from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from vlkg.core.temporal.association import AssociationDecision, AssociationFallbackRequest
from vlkg.core.config.config import LLMFallbackConfig
from vlkg.core.temporal.decision_log import DecisionLogRecord, DecisionLogger, NullDecisionLogger
from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options


logger = logging.getLogger(__name__)


class OpenAICompatibleChatResolver:
    def __init__(
        self,
        config: LLMFallbackConfig,
        *,
        decision_logger: DecisionLogger,
    ) -> None:
        self.config = config
        self.decision_logger = decision_logger
        self.request_count = 0
        self._stats: dict[str, Any] = {
            "candidate_requests": 0,
            "api_calls_attempted": 0,
            "skipped_max_requests": 0,
            "skipped_missing_api_key": 0,
            "request_failures": 0,
            "empty_responses": 0,
            "unparseable_responses": 0,
            "decisions_returned": 0,
            "decision_path_counts": {},
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
            },
        }

    @property
    def provider_name(self) -> str:
        return self.config.provider

    def get_stats(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._stats))

    def resolve(
        self,
        *,
        request: AssociationFallbackRequest,
    ) -> AssociationDecision | None:
        self._stats["candidate_requests"] += 1
        if self.request_count >= self.config.max_requests_per_run:
            self._stats["skipped_max_requests"] += 1
            return None
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            self._stats["skipped_missing_api_key"] += 1
            return None

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": self._build_messages(request),
        }
        apply_reasoning_options(
            payload,
            reasoning_effort=self.config.reasoning_effort,
            thinking_enabled=self.config.thinking_enabled,
        )
        endpoint = urljoin(
            self.config.api_base_url.rstrip("/") + "/",
            self.config.endpoint_path.lstrip("/"),
        )
        raw_response: dict[str, Any] | None = None
        try:
            self.request_count += 1
            self._stats["api_calls_attempted"] += 1
            http_request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(http_request, timeout=self.config.timeout_sec) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("llm_fallback_request_failed=%s", exc)
            self._stats["request_failures"] += 1
            self._log_decision(
                request=request,
                decision=AssociationDecision(
                    matched_entity_id=None,
                    path="new",
                    reason="llm_request_failed",
                ),
                metadata={"error": str(exc), "endpoint": endpoint},
            )
            return None

        self._record_token_usage(raw_response.get("usage"))
        content = self._extract_content(raw_response)
        if content is None:
            self._stats["empty_responses"] += 1
            self._log_decision(
                request=request,
                decision=AssociationDecision(
                    matched_entity_id=None,
                    path="new",
                    reason="llm_empty_response",
                ),
                metadata={"endpoint": endpoint, "raw_response": raw_response},
            )
            return None

        parsed = self._parse_json_content(content)
        if parsed is None:
            self._stats["unparseable_responses"] += 1
            self._log_decision(
                request=request,
                decision=AssociationDecision(
                    matched_entity_id=None,
                    path="new",
                    reason="llm_unparseable_response",
                ),
                metadata={"endpoint": endpoint, "content": content},
            )
            return None

        decision = self._decision_from_payload(
            parsed=parsed,
            request=request,
        )
        self._record_decision(decision)
        self._log_decision(
            request=request,
            decision=decision,
            metadata={"endpoint": endpoint, "raw_response": raw_response},
        )
        return decision

    def _build_messages(
        self,
        request: AssociationFallbackRequest,
    ) -> list[dict[str, str]]:
        confidence_threshold = self.config.override_confidence_threshold
        return [
            {
                "role": "system",
                "content": (
                    "You resolve exactly one ambiguous entity association case in a temporal graph pipeline. "
                    "Return strict JSON only; do not include markdown or extra text. "
                    "Use only the provided current observation, candidates, scores, and context. "
                    "Do not invent labels, entities, frames, relations, or visual details. "
                    "A deterministic SigLIP/geometry default decision is provided. "
                    "Treat that default as the normal answer, and override it only when "
                    "the provided evidence makes the override high-confidence. "
                    "Evidence priority: 1) temporal continuity, 2) spatial continuity "
                    "(IoU, center distance, area ratio), 3) visual similarity score, "
                    "4) label and parent-label consistency, 5) track_id consistency when present. "
                    "If you are not sure, choose abstain so the pipeline can keep the default. "
                    "Only choose match, new, or reject when your confidence is at least "
                    f"{confidence_threshold:.2f}; otherwise choose abstain. "
                    "Choose new or reject only when you are confident the default should not be used. "
                    "Image paths are identifiers only; you cannot inspect image pixels in this request. "
                    "Choose one schema: "
                    '{"decision":"match","matched_entity_id":"...","confidence":0.0,"reason":"..."} '
                    '{"decision":"new","matched_entity_id":null,"confidence":0.0,"reason":"..."} '
                    '{"decision":"reject","matched_entity_id":null,"confidence":0.0,"reason":"..."} '
                    '{"decision":"abstain","matched_entity_id":null,"confidence":0.0,"reason":"..."}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    self._request_payload(request),
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    def _request_payload(
        self,
        request: AssociationFallbackRequest,
    ) -> dict[str, Any]:
        current = request.current
        return {
            "task": "entity_association_fallback",
            "ambiguity_reason": request.ambiguity_reason,
            "deterministic_default": {
                "decision_path": request.default_decision.path,
                "matched_entity_id": request.default_decision.matched_entity_id,
                "score": request.default_decision.score,
                "reason": request.default_decision.reason,
                "instruction": (
                    "Use this SigLIP/geometry decision unless the evidence supports "
                    "a high-confidence override."
                ),
            },
            "guidance": {
                "prefer_match_when": (
                    "one candidate has strong temporal continuity and spatial compatibility "
                    "with the current observation"
                ),
                "prefer_abstain_when": (
                    "the default is plausible or the evidence is not strong enough to override it"
                ),
                "prefer_new_when": (
                    "all candidates have weak spatial and temporal support for the current observation"
                ),
                "prefer_reject_when": (
                    "the candidate set is not trustworthy enough to match, but creating a new entity would also be unsafe"
                ),
            },
            "current_observation": {
                "video_id": current.video_id,
                "frame_name": current.frame_name,
                "frame_index": current.current_frame_index,
                "frame_rank": current.current_frame_rank,
                "object_index": current.object_index,
                "label": current.current_label,
                "parent_label": current.current_parent_label,
                "state_values": list(current.current_state_values),
                "track_id": current.current_track_id,
                "is_static": current.current_is_static,
                "is_noise": current.current_is_noise,
                "bbox": current.current_bbox.model_dump(mode="json"),
                "image_path": current.current_image_path,
                "image_path_is_identifier_only": True,
            },
            "candidates": [
                {
                    "entity_id": candidate.entity_id,
                    "label": candidate.label,
                    "last_frame_index": candidate.last_frame_index,
                    "last_frame_rank": candidate.last_frame_rank,
                    "track_id": candidate.track_id,
                    "parent_label": candidate.parent_label,
                    "is_static": candidate.is_static,
                    "is_noise": candidate.is_noise,
                    "observation_count": candidate.observation_count,
                    "bbox": candidate.last_bbox.model_dump(mode="json"),
                    "candidate_score": self._candidate_score(
                        candidate.entity_id,
                        request=request,
                    ),
                    "context": self._candidate_context(
                        candidate=candidate,
                        request=request,
                    ),
                }
                for candidate in request.filtered_candidates
            ],
        }

    def _record_token_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        totals = self._stats["token_usage"]
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    def _record_decision(self, decision: AssociationDecision) -> None:
        self._stats["decisions_returned"] += 1
        counts = self._stats["decision_path_counts"]
        counts[decision.path] = counts.get(decision.path, 0) + 1

    def _candidate_score(
        self,
        entity_id: str,
        *,
        request: AssociationFallbackRequest,
    ) -> float | None:
        for candidate_id, score in request.ranked_candidate_scores:
            if candidate_id == entity_id:
                return score
        return None

    def _candidate_context(
        self,
        *,
        candidate,
        request: AssociationFallbackRequest,
    ) -> dict[str, Any]:
        current = request.current
        current_bbox = current.current_bbox
        candidate_bbox = candidate.last_bbox
        return {
            "frame_gap": current.current_frame_rank - candidate.last_frame_rank,
            "frame_index_gap": current.current_frame_index - candidate.last_frame_index,
            "same_label": candidate.label == current.current_label,
            "same_parent_label": (
                bool(current.current_parent_label)
                and candidate.parent_label == current.current_parent_label
            ),
            "same_track_hint": (
                bool(current.current_track_id)
                and candidate.track_id == current.current_track_id
            ),
            "spatial_iou": current_bbox.iou(candidate_bbox),
            "center_distance": current_bbox.center_distance(candidate_bbox),
            "area_ratio": current_bbox.area_ratio(candidate_bbox),
        }

    def _extract_content(self, raw_response: dict[str, Any]) -> str | None:
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            ]
            merged = "".join(text_chunks).strip()
            return merged or None
        return None

    def _parse_json_content(self, content: str) -> dict[str, Any] | None:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _decision_from_payload(
        self,
        *,
        parsed: dict[str, Any],
        request: AssociationFallbackRequest,
    ) -> AssociationDecision:
        decision = str(parsed.get("decision", "")).strip().lower()
        confidence = parsed.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
        reason = str(parsed.get("reason", "")).strip() or request.ambiguity_reason
        candidate_ids = {
            candidate.entity_id
            for candidate in request.filtered_candidates
        }
        matched_entity_id = parsed.get("matched_entity_id")
        if decision == "abstain":
            return AssociationDecision(
                matched_entity_id=None,
                path="abstain",
                confidence=confidence_value,
                reason=reason or "llm_abstain",
            )
        if (
            decision == "match"
            and isinstance(matched_entity_id, str)
            and matched_entity_id in candidate_ids
        ):
            return AssociationDecision(
                matched_entity_id=matched_entity_id,
                path="fallback",
                confidence=confidence_value,
                reason=reason,
            )
        if decision == "reject":
            return AssociationDecision(
                matched_entity_id=None,
                path="reject",
                confidence=confidence_value,
                reason=reason,
            )
        if decision == "new":
            return AssociationDecision(
                matched_entity_id=None,
                path="new",
                confidence=confidence_value,
                reason=reason or "llm_new_entity",
            )
        return AssociationDecision(
            matched_entity_id=None,
            path="abstain",
            confidence=confidence_value,
            reason=reason or "llm_invalid_decision",
        )

    def _log_decision(
        self,
        *,
        request: AssociationFallbackRequest,
        decision: AssociationDecision,
        metadata: dict[str, Any],
    ) -> None:
        self.decision_logger.log(
            DecisionLogRecord(
                phase="phase5_llm_fallback",
                provider=self.provider_name,
                reason=request.ambiguity_reason,
                video_id=request.current.video_id,
                frame_name=request.current.frame_name,
                frame_index=request.current.current_frame_index,
                object_index=request.current.object_index,
                current_label=request.current.current_label,
                decision_path=decision.path,
                matched_entity_id=decision.matched_entity_id,
                confidence=decision.confidence,
                candidate_entity_ids=tuple(
                    candidate.entity_id
                    for candidate in request.filtered_candidates
                ),
                metadata=metadata,
            )
        )


def build_llm_fallback_resolver(
    config: LLMFallbackConfig,
    *,
    decision_logger: DecisionLogger | None = None,
) -> OpenAICompatibleChatResolver | None:
    if not config.enabled:
        return None
    return OpenAICompatibleChatResolver(
        config=config,
        decision_logger=decision_logger or NullDecisionLogger(),
    )


def resolve_decision_log_path(
    path: str | None,
    *,
    base_dir: str | Path,
) -> Path | None:
    """Resolve relative decision logs against the run workspace, not input data."""
    if not path:
        return None
    decision_log_path = Path(path)
    if decision_log_path.is_absolute():
        return decision_log_path
    return Path(base_dir) / decision_log_path
