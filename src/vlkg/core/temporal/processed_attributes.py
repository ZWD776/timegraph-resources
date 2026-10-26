from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options
from vlkg.core.temporal.pathing import resolve_repo_input_path


PROCESSED_ATTRIBUTES_FILENAME = "processed_attributes.json"


def _processed_attributes_storage_path(hsgg_input_dir: Path) -> Path:
    """Keep configured derived caches outside organized source observations."""
    from vlkg.core.assets.source_paths import resolve_processed_attributes_dir

    cache_dir = resolve_processed_attributes_dir(hsgg_input_dir)
    return (cache_dir or hsgg_input_dir) / PROCESSED_ATTRIBUTES_FILENAME
PROCESSED_ATTRIBUTES_CONTRACT = "hsgg_processed_attributes_v2"
PROCESSED_ATTRIBUTE_ENTRY_CONTRACT = "hsgg_processed_attribute_entry_v2"
PROCESSED_ATTRIBUTE_REFERENCE_CONTRACT = "hsgg_processed_attribute_reference_v2"
PROCESSED_ATTRIBUTES_PROGRESS_CONTRACT = "hsgg_processed_attributes_progress_v1"

FACT_KINDS = (
    "appearance",
    "color",
    "clothing",
    "material",
    "pattern",
    "shape_size",
    "part",
    "state",
    "action",
    "relation",
    "spatial",
    "text",
    "count",
)
CONFIDENCE_VALUES = {"low", "medium", "high"}


PROCESSED_ATTRIBUTE_SCHEMA: dict[str, Any] = {
    "contract": "hsgg_processed_attribute_schema_v2",
    "description": (
        "Deduplicated object-bound attribute facts parsed from HSGG attributes. "
        "The file is a persistent HSGG sidecar: downstream CG, FAISS, evidence "
        "layer, and IR-QA code should read it instead of reparsing raw "
        "natural-language attributes."
    ),
    "storage": {
        "unique_attributes": (
            "One structured entry per unique (label, normalized raw_attributes) "
            "pair. This is the only section that stores parsed facts."
        ),
        "frames": (
            "Frame/object references with bbox and image anchors. Each reference "
            "points to one unique attribute entry by attribute_key."
        ),
    },
    "structured_fields": {
        "summary": "Short faithful summary of the supplied raw attribute text.",
        "identity_signature": (
            "Stable visible cues that may distinguish this object from another "
            "same-label object, such as blue jacket or white door."
        ),
        "facts": (
            "Atomic object-bound facts. Each fact has kind, target, value, "
            "optional modifiers, raw_span, and confidence."
        ),
        "uncertain": (
            "Raw phrases that are ambiguous, weakly grounded, or not safely "
            "convertible into an atomic fact."
        ),
    },
    "fact_kinds": list(FACT_KINDS),
}


@dataclass(frozen=True)
class AttributeRecordInput:
    record_id: str
    frame_name: str
    object_index: int
    label: str
    raw_attributes: tuple[str, ...]
    image_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    score: float | None = None


@dataclass(frozen=True)
class ProcessedAttributesBuildConfig:
    hsgg_input_dir: Path
    output_path: Path | None = None
    max_frames: int | None = None
    max_records: int | None = None
    priority_filter: str | None = None
    batch_size: int = 16
    force: bool = False
    generator_name: str = "manual"
    generator_model: str | None = None
    generator_mode: str = "llm"
    resume: bool = False
    progress_path: Path | None = None
    checkpoint_every_batches: int = 1
    max_retries: int = 2
    retry_sleep_sec: float = 5.0
    continue_on_batch_failure: bool = False


@dataclass
class AttributeStructuringStats:
    records_seen: int = 0
    records_structured: int = 0
    batches: int = 0
    api_calls_attempted: int = 0
    request_failures: int = 0
    unparseable_responses: int = 0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "records_structured": self.records_structured,
            "batches": self.batches,
            "api_calls_attempted": self.api_calls_attempted,
            "request_failures": self.request_failures,
            "unparseable_responses": self.unparseable_responses,
            "token_usage": dict(self.token_usage),
        }


class AttributeStructurer(Protocol):
    mode: str
    model: str | None

    def structure_batch(
        self,
        records: list[AttributeRecordInput],
    ) -> dict[str, dict[str, Any]]:
        ...

    def stats(self) -> dict[str, Any]:
        ...


class PassthroughAttributeStructurer:
    """Offline smoke-test structurer that preserves raw text without parsing it."""

    mode = "passthrough"
    model = None

    def structure_batch(
        self,
        records: list[AttributeRecordInput],
    ) -> dict[str, dict[str, Any]]:
        return {
            record.record_id: {
                "summary": " | ".join(record.raw_attributes),
                "identity_signature": [],
                "facts": [],
                "uncertain": list(record.raw_attributes),
            }
            for record in records
        }

    def stats(self) -> dict[str, Any]:
        return {}


class SimpleAttributeStructurer:
    """Fast local structurer for on-demand QA evidence.

    It only extracts explicit surface facts from HSGG attribute strings. The
    output is intentionally conservative and question-independent, so it can be
    stored as a persistent processed_attributes sidecar without leaking QA
    assumptions into future runs.
    """

    mode = "simple"
    model = None

    def structure_batch(
        self,
        records: list[AttributeRecordInput],
    ) -> dict[str, dict[str, Any]]:
        return {
            record.record_id: simple_structure_attribute(
                label=record.label,
                raw_attributes=record.raw_attributes,
            )
            for record in records
        }

    def stats(self) -> dict[str, Any]:
        return {}


class OpenAICompatibleAttributeStructurer:
    mode = "llm"

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        api_base_url: str = "https://api.deepseek.com",
        endpoint_path: str = "/chat/completions",
        api_key_env_var: str = "DEEPSEEK_API_KEY",
        timeout_sec: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 1800,
        reasoning_effort: str | None = None,
        thinking_enabled: bool | None = False,
    ) -> None:
        self.model = model
        self.api_base_url = api_base_url
        self.endpoint_path = endpoint_path
        self.api_key_env_var = api_key_env_var
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self._stats = AttributeStructuringStats()

    def stats(self) -> dict[str, Any]:
        return self._stats.as_dict()

    def structure_batch(
        self,
        records: list[AttributeRecordInput],
    ) -> dict[str, dict[str, Any]]:
        if not records:
            return {}
        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {self.api_key_env_var}")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": self._messages(records),
        }
        apply_reasoning_options(
            payload,
            reasoning_effort=self.reasoning_effort,
            thinking_enabled=self.thinking_enabled,
        )
        endpoint = urljoin(
            self.api_base_url.rstrip("/") + "/",
            self.endpoint_path.lstrip("/"),
        )
        self._stats.api_calls_attempted += 1
        try:
            request = Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=self.timeout_sec) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self._stats.request_failures += 1
            raise RuntimeError(f"attribute structuring request failed: {exc}") from exc

        self._record_token_usage(raw_response.get("usage"))
        content = _extract_openai_content(raw_response)
        parsed = _parse_json_content(content)
        if not isinstance(parsed, dict):
            self._stats.unparseable_responses += 1
            raise RuntimeError("attribute structuring response was not a JSON object")
        rows = parsed.get("records")
        if not isinstance(rows, list):
            self._stats.unparseable_responses += 1
            raise RuntimeError("attribute structuring response missing records list")
        output: dict[str, dict[str, Any]] = {}
        requested_ids = {record.record_id for record in records}
        for row in rows:
            if not isinstance(row, dict):
                continue
            record_id = str(row.get("record_id") or "").strip()
            if record_id not in requested_ids:
                continue
            structured = row.get("structured") if isinstance(row.get("structured"), dict) else row
            output[record_id] = normalize_structured_attribute(structured)
        missing = requested_ids - set(output)
        if missing:
            raise RuntimeError(
                "attribute structuring response missing record ids: "
                + ", ".join(sorted(missing)[:5])
            )
        return output

    def _messages(self, records: list[AttributeRecordInput]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You convert HSGG object attribute text into strict JSON. "
                    "Return JSON only, no markdown. Use only the supplied label "
                    "and raw_attributes text. Do not inspect or infer from image "
                    "paths. Do not use questions, answers, dataset-specific "
                    "assumptions, or hidden knowledge. Preserve uncertainty rather "
                    "than guessing. Split only explicit text into atomic facts. For "
                    "every input record, return exactly one item in records with "
                    "the same record_id."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "hsgg_processed_attribute_structuring",
                        "schema_contract": PROCESSED_ATTRIBUTE_SCHEMA["contract"],
                        "required_output_shape": {
                            "records": [
                                {
                                    "record_id": "same as input",
                                    "structured": {
                                        "summary": "short faithful text",
                                        "identity_signature": ["stable visible cue"],
                                        "facts": [
                                            {
                                                "kind": "|".join(FACT_KINDS),
                                                "target": "object or object part",
                                                "value": "atomic fact value",
                                                "modifiers": ["explicit modifier"],
                                                "raw_span": "exact supporting phrase",
                                                "confidence": "low|medium|high",
                                            }
                                        ],
                                        "uncertain": ["ambiguous raw phrase"],
                                    },
                                }
                            ]
                        },
                        "rules": [
                            "Use facts only for explicit content in raw_attributes.",
                            "Put colors, clothing, parts, text, spatial phrases, states, and actions into facts when explicit.",
                            "Put stable distinguishing cues into identity_signature.",
                            "If a phrase is vague or cannot be safely attached to the target, put it in uncertain.",
                            "Do not invent relations or objects that are not in the raw text.",
                        ],
                        "records": [
                            {
                                "record_id": record.record_id,
                                "label": record.label,
                                "raw_attributes": list(record.raw_attributes),
                                "image_path_is_identifier_only": True,
                            }
                            for record in records
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    def _record_token_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        mapping = {
            "prompt_tokens": "prompt_tokens",
            "completion_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }
        details = usage.get("prompt_tokens_details")
        top_level_cache_hit_seen = False
        top_level_cache_miss_seen = False
        for source, target in (
            ("prompt_cache_hit_tokens", "prompt_cache_hit_tokens"),
            ("prompt_cache_miss_tokens", "prompt_cache_miss_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int):
                self._stats.token_usage[target] += value
                if source == "prompt_cache_hit_tokens":
                    top_level_cache_hit_seen = True
                else:
                    top_level_cache_miss_seen = True
        if isinstance(details, dict):
            if not top_level_cache_hit_seen:
                for key in ("cached_tokens", "cache_hit_tokens"):
                    if isinstance(details.get(key), int):
                        self._stats.token_usage["prompt_cache_hit_tokens"] += int(details[key])
                        break
            if not top_level_cache_miss_seen:
                for key in ("cache_miss_tokens",):
                    if isinstance(details.get(key), int):
                        self._stats.token_usage["prompt_cache_miss_tokens"] += int(details[key])
                        break
        if (
            not top_level_cache_miss_seen
            and isinstance(usage.get("prompt_tokens"), int)
            and self._stats.token_usage["prompt_cache_miss_tokens"] == 0
        ):
            prompt_tokens = int(usage["prompt_tokens"])
            cache_hit = min(self._stats.token_usage["prompt_cache_hit_tokens"], prompt_tokens)
            self._stats.token_usage["prompt_cache_miss_tokens"] += max(
                0, prompt_tokens - cache_hit
            )
        for source, target in mapping.items():
            value = usage.get(source)
            if isinstance(value, int):
                self._stats.token_usage[target] += value


def build_processed_attributes(
    config: ProcessedAttributesBuildConfig,
    structurer: AttributeStructurer,
) -> dict[str, Any]:
    hsgg_input_dir = resolve_repo_input_path(config.hsgg_input_dir)
    output_path = (
        resolve_repo_input_path(config.output_path)
        if config.output_path is not None
        else _processed_attributes_storage_path(hsgg_input_dir)
    )
    if output_path.exists() and not config.force:
        raise FileExistsError(
            f"Processed attributes already exist: {output_path}. Use --force to overwrite."
        )

    records = collect_attribute_inputs(
        hsgg_input_dir,
        max_frames=config.max_frames,
        max_records=config.max_records,
    )
    unique_records, record_to_attribute_key = deduplicate_attribute_inputs(records)
    if config.priority_filter:
        unique_records = filter_priority_attribute_records(
            unique_records,
            config.priority_filter,
        )
        selected_keys = set(unique_records)
        record_to_attribute_key = {
            record_id: attribute_key
            for record_id, attribute_key in record_to_attribute_key.items()
            if attribute_key in selected_keys
        }
        records = [
            record
            for record in records
            if record_to_attribute_key.get(record.record_id) in selected_keys
        ]
    unique_list = list(unique_records.values())
    stats = AttributeStructuringStats(records_seen=len(records))
    progress_path = (
        resolve_repo_input_path(config.progress_path)
        if config.progress_path is not None
        else output_path.with_suffix(output_path.suffix + ".progress.json")
    )
    resume_state = (
        load_processed_attributes_progress(progress_path)
        if config.resume and progress_path.exists()
        else None
    )
    selected_attribute_keys = set(unique_records)
    unique_attributes = {
        key: entry
        for key, entry in _resumable_unique_attributes(resume_state).items()
        if key in selected_attribute_keys
    }
    failed_batches = _remaining_failed_batches(
        _resumable_failed_batches(resume_state),
        set(unique_attributes),
        allowed_keys=selected_attribute_keys,
    )
    batch_size = max(1, int(config.batch_size))
    for batch in _batches(unique_list, batch_size):
        batch_keys = [record.record_id for record in batch]
        pending_batch = [
            record for record in batch if record.record_id not in unique_attributes
        ]
        if not pending_batch:
            continue
        stats.batches += 1
        try:
            structured_by_id = structure_batch_with_retries(
                structurer,
                pending_batch,
                max_retries=config.max_retries,
                retry_sleep_sec=config.retry_sleep_sec,
            )
        except Exception as exc:
            failed_batches.append(
                {
                    "batch_keys": batch_keys,
                    "pending_keys": [record.record_id for record in pending_batch],
                    "error": _short_string(exc, limit=600),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_processed_attributes_progress(
                progress_path,
                build_processed_attributes_progress_payload(
                    config=config,
                    output_path=output_path,
                    hsgg_input_dir=hsgg_input_dir,
                    unique_attributes=unique_attributes,
                    failed_batches=failed_batches,
                    reference_count=len(records),
                    total_unique_attribute_count=len(unique_records),
                    stats=stats,
                    structurer=structurer,
                ),
            )
            if config.continue_on_batch_failure:
                continue
            raise
        for record in pending_batch:
            structured = normalize_structured_attribute(
                structured_by_id.get(record.record_id, empty_structured_attribute())
            )
            unique_attributes[record.record_id] = {
                "contract": PROCESSED_ATTRIBUTE_ENTRY_CONTRACT,
                "attribute_key": record.record_id,
                "label": record.label,
                "raw_attributes": list(record.raw_attributes),
                "structured": structured,
                "provenance": {
                    "source": "hsgg.attributes",
                    "source_is_question_independent": True,
                    "uses_questions": False,
                    "uses_gold_answers": False,
                    "uses_retrieval_outputs": False,
                },
            }
            stats.records_structured += 1
        failed_batches = _remaining_failed_batches(failed_batches, set(unique_attributes))
        if stats.batches % max(1, int(config.checkpoint_every_batches)) == 0:
            write_processed_attributes_progress(
                progress_path,
                build_processed_attributes_progress_payload(
                    config=config,
                    output_path=output_path,
                    hsgg_input_dir=hsgg_input_dir,
                    unique_attributes=unique_attributes,
                    failed_batches=failed_batches,
                    reference_count=len(records),
                    total_unique_attribute_count=len(unique_records),
                    stats=stats,
                    structurer=structurer,
                ),
            )

    if failed_batches:
        write_processed_attributes_progress(
            progress_path,
            build_processed_attributes_progress_payload(
                config=config,
                output_path=output_path,
                hsgg_input_dir=hsgg_input_dir,
                unique_attributes=unique_attributes,
                failed_batches=failed_batches,
                reference_count=len(records),
                total_unique_attribute_count=len(unique_records),
                stats=stats,
                structurer=structurer,
            ),
        )
        raise RuntimeError(
            f"processed attribute build has {len(failed_batches)} failed batch(es); "
            f"progress saved to {progress_path}"
        )

    frames: dict[str, dict[str, Any]] = {}
    for record in records:
        attribute_key = record_to_attribute_key[record.record_id]
        frames.setdefault(record.frame_name, {})[str(record.object_index)] = {
            "contract": PROCESSED_ATTRIBUTE_REFERENCE_CONTRACT,
            "record_id": record.record_id,
            "frame_name": record.frame_name,
            "object_index": record.object_index,
            "label": record.label,
            "bbox": list(record.bbox) if record.bbox is not None else None,
            "score": record.score,
            "image_path": record.image_path,
            "attribute_key": attribute_key,
        }

    reference_count = len(records)
    unique_attribute_count = len(unique_attributes)
    dedupe_reduction_count = max(0, reference_count - unique_attribute_count)
    dedupe_reduction_ratio = (
        dedupe_reduction_count / reference_count if reference_count else 0.0
    )
    payload = {
        "contract": PROCESSED_ATTRIBUTES_CONTRACT,
        "schema": PROCESSED_ATTRIBUTE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hsgg_input_dir": _display_path(hsgg_input_dir),
            "attributes_json": _display_path(hsgg_input_dir / "attributes.json"),
            "objects_json": _display_path(hsgg_input_dir / "objects.json"),
        },
        "generation": {
            "generator_name": config.generator_name,
            "generator_mode": config.generator_mode,
            "generator_model": config.generator_model,
            "structurer_mode": structurer.mode,
            "structurer_model": structurer.model,
            "batch_size": batch_size,
            "dedupe_strategy": "label_plus_normalized_raw_attributes",
            "priority_filter": config.priority_filter,
            "resume_enabled": bool(config.resume),
            "progress_path": _display_path(progress_path),
            "max_retries": int(config.max_retries),
            "continue_on_batch_failure": bool(config.continue_on_batch_failure),
            "uses_questions": False,
            "uses_gold_answers": False,
            "uses_retrieval_outputs": False,
            "writes_hsgg_source_files": False,
        },
        "summary": {
            **stats.as_dict(),
            "frame_count": len(frames),
            "reference_count": reference_count,
            "unique_attribute_count": unique_attribute_count,
            "dedupe_reduction_count": dedupe_reduction_count,
            "dedupe_reduction_ratio": round(dedupe_reduction_ratio, 6),
            "structurer": structurer.stats(),
        },
        "unique_attributes": unique_attributes,
        "frames": frames,
    }
    validate_processed_attributes_payload(payload)
    write_json(output_path, payload)
    if progress_path.exists():
        progress_path.unlink()
    return {
        "contract": "hsgg_processed_attributes_build_summary_v2",
        "output_path": _display_path(output_path),
        "hsgg_input_dir": _display_path(hsgg_input_dir),
        "frame_count": len(frames),
        "record_count": reference_count,
        "unique_attribute_count": unique_attribute_count,
        "batch_count": stats.batches,
        "mode": structurer.mode,
    }


def filter_priority_attribute_records(
    unique_records: dict[str, AttributeRecordInput],
    priority_filter: str,
) -> dict[str, AttributeRecordInput]:
    normalized = str(priority_filter or "").strip().lower().replace("_", "-")
    if normalized in {"", "all", "none"}:
        return dict(unique_records)
    allowed = {
        key: record
        for key, record in unique_records.items()
        if priority_attribute_flags(record)
    }
    if normalized in {"text-number", "text-number-v1", "priority-v1"}:
        return allowed
    raise ValueError(f"Unsupported processed attribute priority filter: {priority_filter}")


def priority_attribute_flags(record: AttributeRecordInput) -> tuple[str, ...]:
    texts = normalize_raw_attributes(record.raw_attributes)
    flags: list[str] = []
    for text in texts:
        flags.extend(_priority_attribute_flags_for_text(text))
    return tuple(_unique_strings(flags))


def structure_batch_with_retries(
    structurer: AttributeStructurer,
    records: list[AttributeRecordInput],
    *,
    max_retries: int,
    retry_sleep_sec: float,
) -> dict[str, dict[str, Any]]:
    attempts = max(0, int(max_retries)) + 1
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        try:
            return structurer.structure_batch(records)
        except Exception as exc:
            last_error = exc
            if attempt_index + 1 >= attempts:
                break
            time.sleep(max(0.0, float(retry_sleep_sec)))
    assert last_error is not None
    raise last_error


def build_processed_attributes_progress_payload(
    *,
    config: ProcessedAttributesBuildConfig,
    output_path: Path,
    hsgg_input_dir: Path,
    unique_attributes: dict[str, dict[str, Any]],
    failed_batches: list[dict[str, Any]],
    reference_count: int,
    total_unique_attribute_count: int,
    stats: AttributeStructuringStats,
    structurer: AttributeStructurer,
) -> dict[str, Any]:
    completed_count = len(unique_attributes)
    return {
        "contract": PROCESSED_ATTRIBUTES_PROGRESS_CONTRACT,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hsgg_input_dir": _display_path(hsgg_input_dir),
            "output_path": _display_path(output_path),
        },
        "generation": {
            "generator_name": config.generator_name,
            "generator_mode": config.generator_mode,
            "generator_model": config.generator_model,
            "structurer_mode": structurer.mode,
            "structurer_model": structurer.model,
            "batch_size": max(1, int(config.batch_size)),
            "dedupe_strategy": "label_plus_normalized_raw_attributes",
            "max_retries": int(config.max_retries),
            "continue_on_batch_failure": bool(config.continue_on_batch_failure),
            "uses_questions": False,
            "uses_gold_answers": False,
            "uses_retrieval_outputs": False,
        },
        "summary": {
            **stats.as_dict(),
            "reference_count": reference_count,
            "total_unique_attribute_count": total_unique_attribute_count,
            "completed_unique_attribute_count": completed_count,
            "remaining_unique_attribute_count": max(
                0, total_unique_attribute_count - completed_count
            ),
            "failed_batch_count": len(failed_batches),
            "structurer": structurer.stats(),
        },
        "unique_attributes": unique_attributes,
        "failed_batches": failed_batches,
    }


def load_processed_attributes_progress(path: Path | str) -> dict[str, Any]:
    payload = read_json(resolve_repo_input_path(path))
    if payload.get("contract") != PROCESSED_ATTRIBUTES_PROGRESS_CONTRACT:
        raise ValueError(
            f"Unsupported processed attributes progress contract: {payload.get('contract')!r}"
        )
    unique_attributes = payload.get("unique_attributes")
    if not isinstance(unique_attributes, dict):
        raise ValueError("processed attributes progress must contain unique_attributes object")
    failed_batches = payload.get("failed_batches")
    if not isinstance(failed_batches, list):
        raise ValueError("processed attributes progress must contain failed_batches list")
    return payload


def write_processed_attributes_progress(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _resumable_unique_attributes(
    resume_state: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not resume_state:
        return {}
    unique_attributes = resume_state.get("unique_attributes", {})
    if not isinstance(unique_attributes, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for attribute_key, entry in unique_attributes.items():
        if not isinstance(attribute_key, str) or not isinstance(entry, dict):
            continue
        if entry.get("contract") != PROCESSED_ATTRIBUTE_ENTRY_CONTRACT:
            continue
        if entry.get("attribute_key") != attribute_key:
            continue
        output[attribute_key] = entry
    return output


def _resumable_failed_batches(
    resume_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not resume_state:
        return []
    failed_batches = resume_state.get("failed_batches", [])
    if not isinstance(failed_batches, list):
        return []
    return [item for item in failed_batches if isinstance(item, dict)]


def _remaining_failed_batches(
    failed_batches: list[dict[str, Any]],
    completed_keys: set[str],
    *,
    allowed_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in failed_batches:
        pending_keys = item.get("pending_keys")
        if not isinstance(pending_keys, list):
            output.append(item)
            continue
        normalized_pending = [str(key) for key in pending_keys]
        if allowed_keys is not None:
            normalized_pending = [key for key in normalized_pending if key in allowed_keys]
            if not normalized_pending:
                continue
        if all(key in completed_keys for key in normalized_pending):
            continue
        output.append(item)
    return output


def collect_attribute_inputs(
    hsgg_input_dir: Path | str,
    *,
    max_frames: int | None = None,
    max_records: int | None = None,
) -> list[AttributeRecordInput]:
    hsgg_input_dir = resolve_repo_input_path(hsgg_input_dir)
    attributes_path = hsgg_input_dir / "attributes.json"
    objects_path = hsgg_input_dir / "objects.json"
    if not attributes_path.exists():
        raise FileNotFoundError(f"Missing HSGG attributes.json: {attributes_path}")
    attributes_table = read_json(attributes_path)
    objects_table = read_json(objects_path) if objects_path.exists() else {}
    if not isinstance(attributes_table, dict):
        raise ValueError(f"HSGG attributes.json must be an object: {attributes_path}")
    if not isinstance(objects_table, dict):
        objects_table = {}

    records: list[AttributeRecordInput] = []
    frame_names = sorted_frame_names(list(attributes_table.keys()))
    if max_frames is not None:
        frame_names = frame_names[: max(0, int(max_frames))]
    for frame_name in frame_names:
        frame_attrs = attributes_table.get(frame_name, {})
        if not isinstance(frame_attrs, dict):
            continue
        object_payload = objects_table.get(frame_name, {})
        if not isinstance(object_payload, dict):
            object_payload = {}
        labels = object_payload.get("labels") if isinstance(object_payload.get("labels"), list) else []
        boxes = object_payload.get("boxes") if isinstance(object_payload.get("boxes"), list) else []
        scores = object_payload.get("scores") if isinstance(object_payload.get("scores"), list) else []
        image_path = _optional_string(object_payload.get("image_path"))
        for object_key in sorted(frame_attrs.keys(), key=_object_key_sort):
            object_index = _optional_int(object_key)
            if object_index is None:
                continue
            raw_attributes = tuple(_unique_strings(_string_values(frame_attrs.get(object_key))))
            if not raw_attributes:
                continue
            label = (
                str(labels[object_index]).strip()
                if 0 <= object_index < len(labels)
                else ""
            )
            bbox = _bbox_tuple(boxes[object_index]) if 0 <= object_index < len(boxes) else None
            score = _optional_float(scores[object_index]) if 0 <= object_index < len(scores) else None
            records.append(
                AttributeRecordInput(
                    record_id=f"{frame_name}:object:{object_index}",
                    frame_name=frame_name,
                    object_index=object_index,
                    label=label,
                    raw_attributes=raw_attributes,
                    image_path=image_path,
                    bbox=bbox,
                    score=score,
                )
            )
            if max_records is not None and len(records) >= max(0, int(max_records)):
                return records
    return records


def deduplicate_attribute_inputs(
    records: list[AttributeRecordInput],
) -> tuple[dict[str, AttributeRecordInput], dict[str, str]]:
    unique: dict[str, AttributeRecordInput] = {}
    record_to_attribute_key: dict[str, str] = {}
    for record in records:
        attribute_key = make_attribute_key(record.label, record.raw_attributes)
        record_to_attribute_key[record.record_id] = attribute_key
        if attribute_key in unique:
            continue
        unique[attribute_key] = AttributeRecordInput(
            record_id=attribute_key,
            frame_name=record.frame_name,
            object_index=record.object_index,
            label=_normalize_label(record.label),
            raw_attributes=normalize_raw_attributes(record.raw_attributes),
            image_path=record.image_path,
            bbox=record.bbox,
            score=record.score,
        )
    return unique, record_to_attribute_key


def summarize_attribute_inputs(records: list[AttributeRecordInput]) -> dict[str, Any]:
    unique_records, _ = deduplicate_attribute_inputs(records)
    reference_count = len(records)
    unique_attribute_count = len(unique_records)
    dedupe_reduction_count = max(0, reference_count - unique_attribute_count)
    dedupe_reduction_ratio = (
        dedupe_reduction_count / reference_count if reference_count else 0.0
    )
    return {
        "record_count": reference_count,
        "unique_attribute_count": unique_attribute_count,
        "dedupe_reduction_count": dedupe_reduction_count,
        "dedupe_reduction_ratio": round(dedupe_reduction_ratio, 6),
    }


def make_attribute_key(label: str, raw_attributes: tuple[str, ...] | list[str]) -> str:
    canonical = json.dumps(
        {
            "label": _normalize_label(label),
            "raw_attributes": list(normalize_raw_attributes(raw_attributes)),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    return f"pa_{digest[:20]}"


def normalize_raw_attributes(raw_attributes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(_unique_strings(_string_values(list(raw_attributes))))


def load_processed_attributes(path_or_hsgg_dir: Path | str) -> dict[str, Any]:
    path = resolve_repo_input_path(path_or_hsgg_dir)
    if path.is_dir():
        path = _processed_attributes_storage_path(path)
    payload = read_json(path)
    validate_processed_attributes_payload(payload)
    return payload


def processed_attribute_record(
    payload: dict[str, Any],
    *,
    frame_name: str,
    object_index: int,
) -> dict[str, Any] | None:
    frames = payload.get("frames", {})
    if not isinstance(frames, dict):
        return None
    frame = frames.get(frame_name, {})
    if not isinstance(frame, dict):
        return None
    reference = frame.get(str(object_index))
    if not isinstance(reference, dict):
        return None
    attribute_key = reference.get("attribute_key")
    entry = processed_attribute_entry(payload, str(attribute_key or ""))
    if entry is None:
        return dict(reference)
    return {
        **reference,
        "raw_attributes": entry.get("raw_attributes", []),
        "structured": entry.get("structured", empty_structured_attribute()),
        "attribute_entry": entry,
    }


def processed_attribute_entry(
    payload: dict[str, Any],
    attribute_key: str,
) -> dict[str, Any] | None:
    unique_attributes = payload.get("unique_attributes", {})
    if not isinstance(unique_attributes, dict):
        return None
    entry = unique_attributes.get(attribute_key)
    return entry if isinstance(entry, dict) else None


def get_or_create_processed_attribute_record(
    hsgg_input_dir: Path | str,
    *,
    frame_name: str,
    object_index: int,
    label: str = "",
    raw_attributes: tuple[str, ...] | list[str] | None = None,
    image_path: str | None = None,
    bbox: tuple[float, float, float, float] | list[float] | None = None,
    score: float | None = None,
    structurer: AttributeStructurer | None = None,
) -> dict[str, Any] | None:
    """Return a processed attribute record, creating a simple sidecar entry if missing."""

    hsgg_dir = resolve_repo_input_path(hsgg_input_dir)
    path = _processed_attributes_storage_path(hsgg_dir)
    with _processed_attributes_file_lock(path):
        payload = load_or_initialize_processed_attributes(hsgg_dir)
        existing = processed_attribute_record(
            payload,
            frame_name=str(frame_name),
            object_index=int(object_index),
        )
        if existing is not None:
            return existing

        normalized_raw = normalize_raw_attributes(list(raw_attributes or []))
        if not normalized_raw:
            normalized_raw = tuple(
                _raw_attributes_for_object(hsgg_dir, frame_name, object_index)
            )
        if not normalized_raw:
            return None

        if not label:
            label = _label_for_object(hsgg_dir, frame_name, object_index)
        record = AttributeRecordInput(
            record_id=f"{frame_name}:object:{int(object_index)}",
            frame_name=str(frame_name),
            object_index=int(object_index),
            label=label,
            raw_attributes=normalized_raw,
            image_path=image_path,
            bbox=_bbox_tuple(bbox) if bbox is not None else None,
            score=score,
        )
        attribute_key = make_attribute_key(record.label, record.raw_attributes)
        unique_attributes = payload.setdefault("unique_attributes", {})
        if not isinstance(unique_attributes, dict):
            unique_attributes = {}
            payload["unique_attributes"] = unique_attributes
        if attribute_key not in unique_attributes:
            active_structurer = structurer or SimpleAttributeStructurer()
            structured_by_id = active_structurer.structure_batch(
                [
                    AttributeRecordInput(
                        record_id=attribute_key,
                        frame_name=record.frame_name,
                        object_index=record.object_index,
                        label=_normalize_label(record.label),
                        raw_attributes=record.raw_attributes,
                        image_path=record.image_path,
                        bbox=record.bbox,
                        score=record.score,
                    )
                ]
            )
            structured = normalize_structured_attribute(
                structured_by_id.get(attribute_key, empty_structured_attribute())
            )
            unique_attributes[attribute_key] = {
                "contract": PROCESSED_ATTRIBUTE_ENTRY_CONTRACT,
                "attribute_key": attribute_key,
                "label": _normalize_label(record.label),
                "raw_attributes": list(record.raw_attributes),
                "structured": structured,
                "provenance": {
                    "source": "hsgg.attributes",
                    "source_is_question_independent": True,
                    "uses_questions": False,
                    "uses_gold_answers": False,
                    "uses_retrieval_outputs": False,
                    "generator_mode": active_structurer.mode,
                    "generator_model": active_structurer.model,
                },
            }

        frames = payload.setdefault("frames", {})
        if not isinstance(frames, dict):
            frames = {}
            payload["frames"] = frames
        frame_payload = frames.setdefault(str(frame_name), {})
        if not isinstance(frame_payload, dict):
            frame_payload = {}
            frames[str(frame_name)] = frame_payload
        object_key = str(int(object_index))
        frame_payload[object_key] = {
            "contract": PROCESSED_ATTRIBUTE_REFERENCE_CONTRACT,
            "frame_name": str(frame_name),
            "object_index": int(object_index),
            "label": record.label,
            "attribute_key": attribute_key,
            "image_path": record.image_path,
            "bbox": list(record.bbox) if record.bbox is not None else None,
            "score": record.score,
        }
        write_json(path, payload)
        return processed_attribute_record(
            payload,
            frame_name=str(frame_name),
            object_index=int(object_index),
        )


def load_or_initialize_processed_attributes(hsgg_input_dir: Path | str) -> dict[str, Any]:
    hsgg_dir = resolve_repo_input_path(hsgg_input_dir)
    path = _processed_attributes_storage_path(hsgg_dir)
    if path.exists():
        try:
            return load_processed_attributes(path)
        except Exception:
            payload = read_json(path)
            if isinstance(payload, dict):
                payload.setdefault("contract", PROCESSED_ATTRIBUTES_CONTRACT)
                payload.setdefault("unique_attributes", {})
                payload.setdefault("frames", {})
                return payload
    return {
        "contract": PROCESSED_ATTRIBUTES_CONTRACT,
        "schema": PROCESSED_ATTRIBUTE_SCHEMA,
        "source": {
            "hsgg_input_dir": _display_path(hsgg_dir),
            "source_is_question_independent": True,
            "uses_questions": False,
            "uses_gold_answers": False,
            "uses_retrieval_outputs": False,
        },
        "generation": {
            "generator_name": "on_demand_simple_attribute_structurer",
            "generator_mode": "simple",
            "generator_model": None,
            "dedupe_strategy": "label_plus_normalized_raw_attributes",
        },
        "unique_attributes": {},
        "frames": {},
    }


def validate_processed_attributes_payload(payload: dict[str, Any]) -> None:
    if payload.get("contract") != PROCESSED_ATTRIBUTES_CONTRACT:
        raise ValueError(
            f"Unsupported processed attributes contract: {payload.get('contract')!r}"
        )
    unique_attributes = payload.get("unique_attributes")
    if not isinstance(unique_attributes, dict):
        raise ValueError("processed attributes payload must contain unique_attributes object")
    for attribute_key, entry in unique_attributes.items():
        if not isinstance(attribute_key, str) or not isinstance(entry, dict):
            raise ValueError("unique_attributes must map attribute keys to objects")
        if entry.get("contract") != PROCESSED_ATTRIBUTE_ENTRY_CONTRACT:
            raise ValueError(
                f"Unsupported processed attribute entry contract: {entry.get('contract')!r}"
            )
        if entry.get("attribute_key") != attribute_key:
            raise ValueError(f"attribute_key mismatch for {attribute_key}")
        if not isinstance(entry.get("raw_attributes"), list):
            raise ValueError(f"raw_attributes must be a list in {attribute_key}")
        normalize_structured_attribute(entry.get("structured", {}))

    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("processed attributes payload must contain frames object")
    for frame_name, frame_payload in frames.items():
        if not isinstance(frame_name, str) or not isinstance(frame_payload, dict):
            raise ValueError("processed attributes frames must map frame names to objects")
        for object_key, reference in frame_payload.items():
            if not isinstance(object_key, str) or not isinstance(reference, dict):
                raise ValueError("processed attribute frame entries must be objects")
            if reference.get("contract") != PROCESSED_ATTRIBUTE_REFERENCE_CONTRACT:
                raise ValueError(
                    "Unsupported processed attribute reference contract: "
                    f"{reference.get('contract')!r}"
                )
            if int(reference.get("object_index")) != int(object_key):
                raise ValueError(
                    f"object_index mismatch in frame {frame_name}: {object_key}"
                )
            attribute_key = str(reference.get("attribute_key") or "")
            if attribute_key not in unique_attributes:
                raise ValueError(
                    f"attribute reference missing unique entry in frame {frame_name}: "
                    f"{object_key} -> {attribute_key}"
                )


def empty_structured_attribute() -> dict[str, Any]:
    return {
        "summary": "",
        "identity_signature": [],
        "facts": [],
        "uncertain": [],
    }


def normalize_structured_attribute(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    result = empty_structured_attribute()
    result["summary"] = _short_string(payload.get("summary"), limit=240)
    result["identity_signature"] = _unique_strings(
        _string_values(payload.get("identity_signature"))
    )[:16]
    result["facts"] = _facts(payload.get("facts"))
    result["uncertain"] = _unique_strings(_string_values(payload.get("uncertain")))[:24]
    return result


def simple_structure_attribute(
    *,
    label: str,
    raw_attributes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    texts = normalize_raw_attributes(list(raw_attributes))
    combined = " | ".join(texts)
    facts: list[dict[str, Any]] = []
    for text in texts:
        facts.extend(_simple_attribute_facts(label=label, text=text))
    return normalize_structured_attribute(
        {
            "summary": combined,
            "identity_signature": _simple_identity_signature(label=label, texts=texts),
            "facts": facts,
            "uncertain": [],
        }
    )


def sorted_frame_names(frame_names: list[str]) -> list[str]:
    def sort_key(frame_name: str) -> tuple[int, str]:
        numbers = re.findall(r"\d+", frame_name)
        if numbers:
            return int(numbers[0]), frame_name
        return 10**12, frame_name

    return sorted(frame_names, key=sort_key)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


@contextmanager
def _processed_attributes_file_lock(
    path: Path,
    *,
    timeout_sec: float = 60.0,
    poll_sec: float = 0.1,
    stale_after_sec: float = 600.0,
):
    lock_path = path.with_name(f"{path.name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                fd,
                f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n".encode(
                    "utf-8"
                ),
            )
        except FileExistsError:
            try:
                age_sec = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age_sec = 0.0
            if age_sec > stale_after_sec:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            if time.monotonic() - start > timeout_sec:
                raise TimeoutError(f"Timed out waiting for processed attribute lock: {lock_path}")
            time.sleep(poll_sec)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _batches(
    records: list[AttributeRecordInput],
    batch_size: int,
) -> list[list[AttributeRecordInput]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def _string_values(value: Any) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            values.extend(_string_values(item))
        return values
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_string_values(item))
        return values
    stripped = str(value).strip()
    return [stripped] if stripped else []


_SIMPLE_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow",
}
_SIMPLE_PART_WORDS = {
    "stripe",
    "stripes",
    "tire",
    "tires",
    "wheel",
    "wheels",
    "window",
    "windows",
    "door",
    "doors",
    "handle",
    "handles",
    "logo",
    "sign",
    "lettering",
    "text",
    "writing",
    "grill",
    "grille",
    "mirror",
    "mirrors",
    "roof",
    "shirt",
    "jacket",
    "pants",
    "shoes",
    "hat",
}
_SIMPLE_MATERIAL_WORDS = {
    "brick",
    "chrome",
    "concrete",
    "glass",
    "leather",
    "metal",
    "metallic",
    "plastic",
    "stone",
    "wood",
    "wooden",
}
_SIMPLE_STATE_WORDS = {
    "attached",
    "closed",
    "driving",
    "hanging",
    "moving",
    "open",
    "opened",
    "parked",
    "standing",
    "walking",
}
_PRIORITY_TEXT_WORDS = {
    "lettering",
    "logo",
    "logos",
    "number",
    "numbers",
    "printed",
    "reading",
    "reads",
    "says",
    "sign",
    "signage",
    "signs",
    "text",
    "word",
    "words",
    "writing",
}
_PRIORITY_NUMBER_WORDS = {
    "double",
    "eight",
    "five",
    "four",
    "multiple",
    "nine",
    "one",
    "pair",
    "several",
    "seven",
    "six",
    "ten",
    "three",
    "two",
}


def _simple_attribute_facts(*, label: str, text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    clean_text = " ".join(str(text or "").strip().split())
    lower = clean_text.lower()
    clean_label = _normalize_label(label) or "object"
    if not clean_text:
        return facts

    body_colors, part_colors = _simple_color_facts(lower)
    for color in body_colors:
        facts.append(
            _simple_fact(
                "color",
                clean_label,
                color,
                clean_text,
                modifiers=["body", "primary"],
                confidence="medium",
            )
        )
    for part, color in part_colors:
        facts.append(
            _simple_fact(
                "color",
                part,
                color,
                clean_text,
                modifiers=["part"],
                confidence="medium",
            )
        )
        facts.append(
            _simple_fact(
                "part",
                clean_label,
                part,
                clean_text,
                modifiers=[f"{part}_color={color}"],
                confidence="medium",
            )
        )

    for material in sorted(_SIMPLE_MATERIAL_WORDS):
        if re.search(rf"\b{re.escape(material)}\b", lower):
            facts.append(
                _simple_fact(
                    "material",
                    clean_label,
                    material,
                    clean_text,
                    confidence="low",
                )
            )
    for state in sorted(_SIMPLE_STATE_WORDS):
        if re.search(rf"\b{re.escape(state)}\b", lower):
            facts.append(
                _simple_fact(
                    "state",
                    clean_label,
                    state,
                    clean_text,
                    confidence="low",
                )
            )
    for phrase in _simple_text_phrases(clean_text):
        facts.append(
            _simple_fact(
                "text",
                clean_label,
                phrase,
                clean_text,
                confidence="medium",
            )
        )
    if not facts:
        facts.append(
            _simple_fact(
                "appearance",
                clean_label,
                clean_text,
                clean_text,
                confidence="low",
            )
        )
    return facts


def _priority_attribute_flags_for_text(text: str) -> list[str]:
    lower = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9&-]+", lower))
    flags: list[str] = []
    if tokens & _PRIORITY_TEXT_WORDS or re.search(r'"[^"]{2,}"', str(text or "")):
        flags.append("text_logo_sign")
    if tokens & _PRIORITY_NUMBER_WORDS or re.search(r"\b\d+\b", lower):
        flags.append("number_quantity")
    return flags


def _simple_color_facts(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    body_colors: list[str] = []
    part_colors: list[tuple[str, str]] = []
    colors_pattern = "|".join(sorted(_SIMPLE_COLOR_WORDS))
    part_pattern = "|".join(sorted(_SIMPLE_PART_WORDS, key=len, reverse=True))

    for match in re.finditer(
        rf"\b({colors_pattern})\b(?:\s+(?:and\s+)?\b({colors_pattern})\b)?\s+({part_pattern})\b",
        text,
    ):
        color1 = _normalize_color(match.group(1))
        color2 = _normalize_color(match.group(2))
        part = _normalize_part(match.group(3))
        if part in {"stripe", "tire", "wheel", "window", "door", "handle", "logo", "sign", "lettering", "text", "writing", "grill", "grille", "mirror", "roof", "shirt", "jacket", "pants", "shoes", "hat"}:
            part_colors.append((part, color1))
            if color2:
                part_colors.append((part, color2))
        else:
            body_colors.append(color1)
            if color2:
                body_colors.append(color2)

    for match in re.finditer(
        rf"\b({part_pattern})\b\s+(?:with|of|in|on|featuring)\s+(?:a\s+|an\s+|the\s+)?\b({colors_pattern})\b",
        text,
    ):
        part_colors.append((_normalize_part(match.group(1)), _normalize_color(match.group(2))))

    for match in re.finditer(
        rf"\b({colors_pattern})\b(?:\s+and\s+\b({colors_pattern})\b)?\s+[a-z0-9_-]*\s*(?:truck|car|van|bus|vehicle|person|man|woman|door|building|couch|sofa|trailer)\b",
        text,
    ):
        body_colors.append(_normalize_color(match.group(1)))
        second = _normalize_color(match.group(2))
        if second:
            body_colors.append(second)

    if not body_colors:
        for color in sorted(_SIMPLE_COLOR_WORDS):
            if re.search(rf"\b{re.escape(color)}\b", text):
                body_colors.append(_normalize_color(color))
                break

    part_color_values = {color for _part, color in part_colors}
    if len(body_colors) > 1:
        body_colors = [
            color
            for color in body_colors
            if color not in part_color_values or _color_precedes_main_object(text, color)
        ]
    return _unique_strings(body_colors), _unique_part_color_pairs(part_colors)


def _simple_text_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for pattern in (
        r"\b(?:logo|sign|text|writing|lettering)\s+(?:with|says|showing|reading|of)\s+([^,;.]+)",
        r"\b([A-Z][A-Z0-9&-]{2,})\b",
    ):
        for match in re.finditer(pattern, text):
            value = " ".join(str(match.group(1)).strip(" '\"").split())
            if value:
                phrases.append(value)
    return _unique_strings(phrases)[:6]


def _simple_identity_signature(
    *,
    label: str,
    texts: tuple[str, ...],
) -> list[str]:
    signatures: list[str] = []
    for text in texts:
        lower = text.lower()
        colors, part_colors = _simple_color_facts(lower)
        for color in colors:
            signatures.append(f"{color} {_normalize_label(label) or 'object'}")
        for part, color in part_colors:
            signatures.append(f"{color} {part}")
        for phrase in _simple_text_phrases(text):
            signatures.append(f"text {phrase}")
    return _unique_strings(signatures)[:16]


def _simple_fact(
    kind: str,
    target: str,
    value: str,
    raw_span: str,
    *,
    modifiers: list[str] | None = None,
    confidence: str = "medium",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "target": target,
        "value": value,
        "modifiers": modifiers or [],
        "raw_span": raw_span,
        "confidence": confidence,
    }


def _normalize_color(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "grey":
        return "gray"
    return text


def _normalize_part(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    singular = {
        "doors": "door",
        "handles": "handle",
        "mirrors": "mirror",
        "pants": "pants",
        "shoes": "shoes",
        "stripes": "stripe",
        "tires": "tire",
        "wheels": "wheel",
        "windows": "window",
    }
    return singular.get(text, text)


def _color_precedes_main_object(text: str, color: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(color)}\b\s+[a-z0-9_-]*\s*(?:truck|car|van|bus|vehicle|person|man|woman|door|building|couch|sofa|trailer)\b",
            text,
        )
    )


def _unique_part_color_pairs(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[tuple[str, str]] = []
    for part, color in values:
        key = (_normalize_part(part), _normalize_color(color))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def _raw_attributes_for_object(
    hsgg_dir: Path,
    frame_name: str,
    object_index: int,
) -> list[str]:
    path = hsgg_dir / "attributes.json"
    if not path.exists():
        return []
    try:
        attributes = read_json(path)
    except Exception:
        return []
    if not isinstance(attributes, dict):
        return []
    frame = attributes.get(str(frame_name), {})
    if not isinstance(frame, dict):
        return []
    return _unique_strings(_string_values(frame.get(str(int(object_index)))))


def _label_for_object(
    hsgg_dir: Path,
    frame_name: str,
    object_index: int,
) -> str:
    path = hsgg_dir / "objects.json"
    if not path.exists():
        return ""
    try:
        objects = read_json(path)
    except Exception:
        return ""
    if not isinstance(objects, dict):
        return ""
    frame = objects.get(str(frame_name), {})
    if not isinstance(frame, dict):
        return ""
    labels = frame.get("labels")
    if isinstance(labels, list) and 0 <= int(object_index) < len(labels):
        return str(labels[int(object_index)] or "").strip()
    return ""


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = " ".join(str(value).strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _facts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in FACT_KINDS:
            kind = "appearance"
        target = _short_string(item.get("target"), limit=120)
        fact_value = _short_string(item.get("value"), limit=160)
        raw_span = _short_string(item.get("raw_span"), limit=220)
        modifiers = _unique_strings(_string_values(item.get("modifiers")))[:12]
        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in CONFIDENCE_VALUES:
            confidence = "low"
        if not fact_value and not raw_span:
            continue
        key = (kind, target.lower(), fact_value.lower(), raw_span.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "kind": kind,
                "target": target,
                "value": fact_value,
                "modifiers": modifiers,
                "raw_span": raw_span,
                "confidence": confidence,
            }
        )
    return rows[:32]


def _short_string(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    numbers = [_optional_float(item) for item in value[:4]]
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = [float(item) for item in numbers if item is not None]
    return (x1, y1, x2, y2)


def _object_key_sort(value: Any) -> tuple[int, str]:
    object_index = _optional_int(value)
    if object_index is None:
        return 10**12, str(value)
    return object_index, str(value)


def _normalize_label(label: Any) -> str:
    return " ".join(str(label or "").strip().split()).lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_repo_input_path("."))).replace("\\", "/")
    except ValueError:
        return str(path)


def _extract_openai_content(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = first.get("text")
    return content if isinstance(content, str) else ""


def _parse_json_content(content: str) -> Any:
    text = content.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
