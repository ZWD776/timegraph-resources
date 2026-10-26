from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from vlkg.core.models.temporal_graph import BoundingBox


@dataclass(frozen=True)
class AssociationConfig:
    entity_match_max_frame_gap: int = 2
    static_entity_match_max_frame_gap: int | None = None
    local_tracklet_mode: bool = False
    local_tracklet_max_frame_gap: int = 1
    local_tracklet_embedding_similarity_threshold: float = 0.72
    local_tracklet_min_score: float = 0.60
    local_tracklet_score_margin: float = 0.08
    local_tracklet_motion_weight: float = 0.35
    local_tracklet_embedding_weight: float = 0.45
    local_tracklet_iou_weight: float = 0.15
    local_tracklet_object_index_weight: float = 0.05
    continuity_max_center_distance_ratio: float | None = None
    continuity_min_area_ratio: float | None = None
    object_index_jump_max_delta: int | None = None
    object_index_jump_embedding_veto_threshold: float | None = None
    state_conflict_embedding_veto_threshold: float | None = None
    state_conflict_min_token_overlap_ratio: float = 0.75
    label_conflict_embedding_veto_threshold: float | None = None
    fast_iou_threshold: float = 0.50
    embedding_similarity_threshold: float = 0.75
    static_fast_iou_threshold: float = 0.15
    static_center_distance_ratio: float = 0.20
    human_label_dynamic_center_distance_ratio: float = 2.50
    cross_label_dynamic_center_distance_ratio: float = 1.25
    cross_label_dynamic_min_iou: float = 0.01
    fast_match_embedding_veto_threshold: float | None = None
    static_fast_match_embedding_veto_threshold: float | None = None
    enable_same_label_dynamic_spatial_gate: bool = False


@dataclass(frozen=True)
class AssociationCandidate:
    entity_id: str
    label: str
    last_frame_rank: int
    last_frame_index: int
    track_id: str | None
    object_index: int
    last_bbox: BoundingBox
    parent_label: str | None
    last_state_values: Sequence[str]
    is_static: bool
    is_noise: bool
    observation_count: int
    canonical_image_path: str | None
    embedding: np.ndarray | None


@dataclass(frozen=True)
class AssociationDecision:
    matched_entity_id: str | None
    path: str
    score: float | None = None
    confidence: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssociationFallbackRequest:
    current: AssociationInput
    filtered_candidates: tuple[AssociationCandidate, ...]
    ranked_candidate_scores: tuple[tuple[str, float | None], ...]
    ambiguity_reason: str
    default_decision: AssociationDecision


@dataclass
class AssociationPolicyStats:
    fallback_requests_built: int = 0
    fallback_request_reason_counts: dict[str, int] = field(default_factory=dict)
    fallback_request_label_counts: dict[str, int] = field(default_factory=dict)
    fallback_request_default_path_counts: dict[str, int] = field(default_factory=dict)
    fallback_skipped_disabled: int = 0
    fallback_skipped_static_current: int = 0
    fallback_skipped_noise_current: int = 0
    fallback_skipped_excess_candidates: int = 0
    fallback_skipped_insufficient_candidates: int = 0
    fallback_skipped_no_ambiguity: int = 0

    def record_request(
        self,
        *,
        reason: str,
        current_label: str,
        default_path: str,
    ) -> None:
        self.fallback_requests_built += 1
        self.fallback_request_reason_counts[reason] = (
            self.fallback_request_reason_counts.get(reason, 0) + 1
        )
        self.fallback_request_label_counts[current_label] = (
            self.fallback_request_label_counts.get(current_label, 0) + 1
        )
        self.fallback_request_default_path_counts[default_path] = (
            self.fallback_request_default_path_counts.get(default_path, 0) + 1
        )


class AssociationFallbackResolver(Protocol):
    def resolve(
        self,
        *,
        request: AssociationFallbackRequest,
    ) -> AssociationDecision | None:
        ...


@dataclass(frozen=True)
class AssociationInput:
    video_id: str
    frame_name: str
    current_label: str
    current_frame_rank: int
    current_frame_index: int
    current_track_id: str | None
    object_index: int
    current_image_path: str
    current_parent_label: str | None
    current_state_values: Sequence[str]
    current_is_static: bool
    current_is_noise: bool
    current_bbox: BoundingBox
    current_embedding: np.ndarray
    label_compatibility_groups: Sequence[set[str]]
    current_match_max_frame_gap: int


class AssociationConstraint(Protocol):
    def filter_candidates(
        self,
        *,
        filtered_candidates: list[AssociationCandidate],
        current: AssociationInput,
    ) -> list[AssociationCandidate]:
        ...


@dataclass(frozen=True)
class ParentLabelConstraint:
    enabled: bool = False

    def filter_candidates(
        self,
        *,
        filtered_candidates: list[AssociationCandidate],
        current: AssociationInput,
    ) -> list[AssociationCandidate]:
        if not self.enabled or not current.current_parent_label:
            return filtered_candidates
        constrained = [
            candidate
            for candidate in filtered_candidates
            if candidate.parent_label is None
            or candidate.parent_label == current.current_parent_label
        ]
        return constrained or filtered_candidates


class AssociationPolicy:
    def __init__(
        self,
        config: AssociationConfig,
        constraints: Sequence[AssociationConstraint] | None = None,
        enable_track_hints: bool = True,
        enable_llm_fallback: bool = False,
        fallback_resolver: AssociationFallbackResolver | None = None,
        fallback_max_candidates: int = 4,
        fallback_min_candidate_count: int = 1,
        fallback_ambiguity_margin: float = 0.05,
        fallback_low_similarity_floor: float = 0.55,
        fallback_low_similarity_ambiguity_margin: float = 0.03,
        fallback_single_candidate_score_margin: float = 0.03,
        fallback_enable_static: bool = False,
        fallback_enable_noise: bool = False,
        fallback_enable_candidate_context: bool = False,
        fallback_require_second_candidate_for_low_similarity: bool = True,
        fallback_override_confidence_threshold: float = 0.80,
    ) -> None:
        self.config = config
        self.constraints = list(constraints or [])
        self.enable_track_hints = enable_track_hints
        self.enable_llm_fallback = enable_llm_fallback
        self.fallback_resolver = fallback_resolver
        self.fallback_max_candidates = fallback_max_candidates
        self.fallback_min_candidate_count = fallback_min_candidate_count
        self.fallback_ambiguity_margin = fallback_ambiguity_margin
        self.fallback_low_similarity_floor = fallback_low_similarity_floor
        self.fallback_low_similarity_ambiguity_margin = (
            fallback_low_similarity_ambiguity_margin
        )
        self.fallback_single_candidate_score_margin = (
            fallback_single_candidate_score_margin
        )
        self.fallback_enable_static = fallback_enable_static
        self.fallback_enable_noise = fallback_enable_noise
        self.fallback_enable_candidate_context = fallback_enable_candidate_context
        self.fallback_require_second_candidate_for_low_similarity = (
            fallback_require_second_candidate_for_low_similarity
        )
        self.fallback_override_confidence_threshold = (
            fallback_override_confidence_threshold
        )
        self.stats = AssociationPolicyStats()

    def reset_stats(self) -> None:
        self.stats = AssociationPolicyStats()

    def get_stats(self) -> dict[str, Any]:
        return {
            "fallback_requests_built": self.stats.fallback_requests_built,
            "fallback_request_reason_counts": dict(
                self.stats.fallback_request_reason_counts
            ),
            "fallback_request_label_counts": dict(
                self.stats.fallback_request_label_counts
            ),
            "fallback_request_default_path_counts": dict(
                self.stats.fallback_request_default_path_counts
            ),
            "fallback_skipped_disabled": self.stats.fallback_skipped_disabled,
            "fallback_skipped_static_current": (
                self.stats.fallback_skipped_static_current
            ),
            "fallback_skipped_noise_current": self.stats.fallback_skipped_noise_current,
            "fallback_skipped_excess_candidates": (
                self.stats.fallback_skipped_excess_candidates
            ),
            "fallback_skipped_insufficient_candidates": (
                self.stats.fallback_skipped_insufficient_candidates
            ),
            "fallback_skipped_no_ambiguity": self.stats.fallback_skipped_no_ambiguity,
        }

    def select_entity_match(
        self,
        *,
        candidates: Iterable[AssociationCandidate],
        current: AssociationInput,
    ) -> AssociationDecision:
        filtered_candidates = self._filter_candidates(
            candidates=candidates,
            current=current,
        )
        if self.config.local_tracklet_mode:
            local_tracklet_match = find_local_tracklet_match(
                candidates=filtered_candidates,
                current=current,
                max_frame_gap=self.config.local_tracklet_max_frame_gap,
                embedding_similarity_threshold=(
                    self.config.local_tracklet_embedding_similarity_threshold
                ),
                min_score=self.config.local_tracklet_min_score,
                score_margin=self.config.local_tracklet_score_margin,
                motion_weight=self.config.local_tracklet_motion_weight,
                embedding_weight=self.config.local_tracklet_embedding_weight,
                iou_weight=self.config.local_tracklet_iou_weight,
                object_index_weight=self.config.local_tracklet_object_index_weight,
            )
            if local_tracklet_match is not None:
                matched_candidate, score = local_tracklet_match
                return AssociationDecision(
                    matched_entity_id=matched_candidate.entity_id,
                    path="local_tracklet",
                    score=score,
                )
            return AssociationDecision(matched_entity_id=None, path="new")

        fast_match = find_fast_match(
            candidates=filtered_candidates,
            current_bbox=current.current_bbox,
            current_embedding=current.current_embedding,
            fast_iou_threshold=self.config.fast_iou_threshold,
            static_fast_iou_threshold=self.config.static_fast_iou_threshold,
            static_center_distance_ratio=self.config.static_center_distance_ratio,
            fast_match_embedding_veto_threshold=(
                self.config.fast_match_embedding_veto_threshold
            ),
            static_fast_match_embedding_veto_threshold=(
                self.config.static_fast_match_embedding_veto_threshold
            ),
        )
        if fast_match is not None:
            matched_candidate, score = fast_match
            return AssociationDecision(
                matched_entity_id=matched_candidate.entity_id,
                path="fast",
                score=score,
            )

        ranked_slow_matches = rank_slow_matches(
            candidates=filtered_candidates,
            current_embedding=current.current_embedding,
        )
        slow_match = find_slow_match(
            ranked_matches=ranked_slow_matches,
            embedding_similarity_threshold=self.config.embedding_similarity_threshold,
        )
        if slow_match is not None:
            matched_candidate, score = slow_match
            default_decision = AssociationDecision(
                matched_entity_id=matched_candidate.entity_id,
                path="slow",
                score=score,
            )
        else:
            default_decision = AssociationDecision(matched_entity_id=None, path="new")

        fallback_request = self._build_fallback_request(
            filtered_candidates=filtered_candidates,
            current=current,
            ranked_slow_matches=ranked_slow_matches,
            default_decision=default_decision,
        )
        if fallback_request is not None and self.fallback_resolver is not None:
            fallback_decision = self.fallback_resolver.resolve(
                request=fallback_request,
            )
            if fallback_decision is not None:
                return self._apply_fallback_decision(
                    fallback_decision=fallback_decision,
                    default_decision=default_decision,
                    filtered_candidates=filtered_candidates,
                )

        return default_decision

    def _build_fallback_request(
        self,
        *,
        filtered_candidates: list[AssociationCandidate],
        current: AssociationInput,
        ranked_slow_matches: Sequence[tuple[AssociationCandidate, float]],
        default_decision: AssociationDecision,
    ) -> AssociationFallbackRequest | None:
        if not self.enable_llm_fallback or self.fallback_resolver is None:
            self.stats.fallback_skipped_disabled += 1
            return None
        if current.current_is_static and not self.fallback_enable_static:
            self.stats.fallback_skipped_static_current += 1
            return None
        if current.current_is_noise and not self.fallback_enable_noise:
            self.stats.fallback_skipped_noise_current += 1
            return None
        if not filtered_candidates:
            self.stats.fallback_skipped_insufficient_candidates += 1
            return None
        if len(filtered_candidates) > self.fallback_max_candidates:
            self.stats.fallback_skipped_excess_candidates += 1
            return None

        ambiguity_reason: str | None = None
        if ranked_slow_matches:
            top_candidate, top_score = ranked_slow_matches[0]
            second_score = ranked_slow_matches[1][1] if len(ranked_slow_matches) > 1 else None
            if (
                top_score >= self.config.embedding_similarity_threshold
                and second_score is not None
                and (top_score - second_score) <= self.fallback_ambiguity_margin
            ):
                ambiguity_reason = "slow_margin_ambiguous"
            elif (
                top_score >= self.fallback_low_similarity_floor
                and top_score < self.config.embedding_similarity_threshold
                and (
                    not self.fallback_require_second_candidate_for_low_similarity
                    or (
                        second_score is not None
                        and (
                            top_score - second_score
                        ) <= self.fallback_low_similarity_ambiguity_margin
                    )
                    or (
                        second_score is None
                        and top_candidate.observation_count >= 2
                        and top_score >= (
                            self.config.embedding_similarity_threshold
                            - self.fallback_single_candidate_score_margin
                        )
                    )
                )
            ):
                ambiguity_reason = "low_similarity_uncertain"
        elif (
            self.fallback_enable_candidate_context
            and len(filtered_candidates) >= max(self.fallback_min_candidate_count, 2)
        ):
            ambiguity_reason = "candidate_context_without_embedding_match"

        if ambiguity_reason is None:
            self.stats.fallback_skipped_no_ambiguity += 1
            return None

        self.stats.record_request(
            reason=ambiguity_reason,
            current_label=current.current_label,
            default_path=default_decision.path,
        )
        ranked_candidate_scores = tuple(
            (candidate.entity_id, score)
            for candidate, score in ranked_slow_matches[: self.fallback_max_candidates]
        )
        if not ranked_candidate_scores:
            ranked_candidate_scores = tuple(
                (candidate.entity_id, None)
                for candidate in filtered_candidates[: self.fallback_max_candidates]
            )
        return AssociationFallbackRequest(
            current=current,
            filtered_candidates=tuple(filtered_candidates[: self.fallback_max_candidates]),
            ranked_candidate_scores=ranked_candidate_scores,
            ambiguity_reason=ambiguity_reason,
            default_decision=default_decision,
        )

    def _apply_fallback_decision(
        self,
        *,
        fallback_decision: AssociationDecision,
        default_decision: AssociationDecision,
        filtered_candidates: Sequence[AssociationCandidate],
    ) -> AssociationDecision:
        if fallback_decision.path == "abstain":
            return default_decision

        candidate_ids = {
            candidate.entity_id
            for candidate in filtered_candidates
        }
        if fallback_decision.path == "fallback":
            matched_entity_id = fallback_decision.matched_entity_id
            if matched_entity_id not in candidate_ids:
                return default_decision
            if matched_entity_id == default_decision.matched_entity_id:
                return default_decision
            if self._has_high_fallback_confidence(fallback_decision):
                return fallback_decision
            return default_decision

        if fallback_decision.path in {"new", "reject"}:
            if (
                fallback_decision.path == default_decision.path
                and fallback_decision.matched_entity_id
                == default_decision.matched_entity_id
            ):
                return default_decision
            if self._has_high_fallback_confidence(fallback_decision):
                return fallback_decision
            return default_decision

        return default_decision

    def _has_high_fallback_confidence(
        self,
        decision: AssociationDecision,
    ) -> bool:
        if decision.confidence is None:
            return False
        return decision.confidence >= self.fallback_override_confidence_threshold

    def _filter_candidates(
        self,
        *,
        candidates: Iterable[AssociationCandidate],
        current: AssociationInput,
    ) -> list[AssociationCandidate]:
        filtered = iter_candidate_entities(
            candidates=candidates,
            current_label=current.current_label,
            current_frame_rank=current.current_frame_rank,
            current_is_static=current.current_is_static,
            current_object_index=current.object_index,
            current_state_values=current.current_state_values,
            current_embedding=current.current_embedding,
            current_bbox=current.current_bbox,
            entity_match_max_frame_gap=self.config.entity_match_max_frame_gap,
            current_match_max_frame_gap=current.current_match_max_frame_gap,
            static_entity_match_max_frame_gap=(
                self.config.static_entity_match_max_frame_gap
            ),
            label_compatibility_groups=current.label_compatibility_groups,
            human_label_dynamic_center_distance_ratio=(
                self.config.human_label_dynamic_center_distance_ratio
            ),
            cross_label_dynamic_center_distance_ratio=(
                self.config.cross_label_dynamic_center_distance_ratio
            ),
            cross_label_dynamic_min_iou=self.config.cross_label_dynamic_min_iou,
            enable_same_label_dynamic_spatial_gate=(
                self.config.enable_same_label_dynamic_spatial_gate
            ),
            continuity_max_center_distance_ratio=(
                self.config.continuity_max_center_distance_ratio
            ),
            continuity_min_area_ratio=self.config.continuity_min_area_ratio,
            object_index_jump_max_delta=self.config.object_index_jump_max_delta,
            object_index_jump_embedding_veto_threshold=(
                self.config.object_index_jump_embedding_veto_threshold
            ),
            state_conflict_embedding_veto_threshold=(
                self.config.state_conflict_embedding_veto_threshold
            ),
            state_conflict_min_token_overlap_ratio=(
                self.config.state_conflict_min_token_overlap_ratio
            ),
            label_conflict_embedding_veto_threshold=(
                self.config.label_conflict_embedding_veto_threshold
            ),
        )
        filtered = self.apply_track_hint(
            filtered_candidates=filtered,
            current=current,
        )
        return self.apply_post_filters(filtered_candidates=filtered, current=current)

    def apply_track_hint(
        self,
        *,
        filtered_candidates: list[AssociationCandidate],
        current: AssociationInput,
    ) -> list[AssociationCandidate]:
        if not self.enable_track_hints or current.current_track_id is None:
            return filtered_candidates
        tracked = [
            candidate
            for candidate in filtered_candidates
            if candidate.track_id == current.current_track_id
        ]
        return tracked or filtered_candidates

    def apply_post_filters(
        self,
        *,
        filtered_candidates: list[AssociationCandidate],
        current: AssociationInput,
    ) -> list[AssociationCandidate]:
        constrained = filtered_candidates
        for constraint in self.constraints:
            constrained = constraint.filter_candidates(
                filtered_candidates=constrained,
                current=current,
            )
        return constrained


def iter_candidate_entities(
    *,
    candidates: Iterable[AssociationCandidate],
    current_label: str,
    current_frame_rank: int,
    current_is_static: bool,
    current_object_index: int,
    current_state_values: Sequence[str],
    current_embedding: np.ndarray,
    current_bbox: BoundingBox,
    entity_match_max_frame_gap: int,
    current_match_max_frame_gap: int,
    static_entity_match_max_frame_gap: int | None,
    label_compatibility_groups: Sequence[set[str]],
    human_label_dynamic_center_distance_ratio: float,
    cross_label_dynamic_center_distance_ratio: float,
    cross_label_dynamic_min_iou: float,
    enable_same_label_dynamic_spatial_gate: bool,
    continuity_max_center_distance_ratio: float | None,
    continuity_min_area_ratio: float | None,
    object_index_jump_max_delta: int | None,
    object_index_jump_embedding_veto_threshold: float | None,
    state_conflict_embedding_veto_threshold: float | None,
    state_conflict_min_token_overlap_ratio: float,
    label_conflict_embedding_veto_threshold: float | None,
) -> list[AssociationCandidate]:
    filtered: list[AssociationCandidate] = []
    max_gap = max(1, current_match_max_frame_gap or entity_match_max_frame_gap)
    static_max_gap = (
        max(1, static_entity_match_max_frame_gap)
        if static_entity_match_max_frame_gap is not None
        else None
    )
    for candidate in candidates:
        gap = current_frame_rank - candidate.last_frame_rank
        if gap <= 0:
            continue
        if current_is_static:
            if not candidate.is_static:
                continue
            if static_max_gap is not None and gap > static_max_gap:
                continue
        elif candidate.is_static or gap > max_gap:
            continue
        if not labels_compatible(
            current_label=current_label,
            candidate_label=candidate.label,
            label_compatibility_groups=label_compatibility_groups,
        ):
            continue
        if not temporal_continuity_compatible(
            candidate=candidate,
            current_label=current_label,
            current_bbox=current_bbox,
            current_object_index=current_object_index,
            current_state_values=current_state_values,
            current_embedding=current_embedding,
            continuity_max_center_distance_ratio=(
                continuity_max_center_distance_ratio
            ),
            continuity_min_area_ratio=continuity_min_area_ratio,
            object_index_jump_max_delta=object_index_jump_max_delta,
            object_index_jump_embedding_veto_threshold=(
                object_index_jump_embedding_veto_threshold
            ),
            state_conflict_embedding_veto_threshold=(
                state_conflict_embedding_veto_threshold
            ),
            state_conflict_min_token_overlap_ratio=(
                state_conflict_min_token_overlap_ratio
            ),
            label_conflict_embedding_veto_threshold=(
                label_conflict_embedding_veto_threshold
            ),
        ):
            continue
        if (
            not current_is_static
            and current_label == candidate.label
            and (
                enable_same_label_dynamic_spatial_gate
                or label_uses_dynamic_spatial_gate(
                    label=current_label,
                    label_compatibility_groups=label_compatibility_groups,
                )
            )
            and not cross_label_dynamic_spatially_compatible(
                current_bbox=current_bbox,
                candidate_bbox=candidate.last_bbox,
                max_center_distance_ratio=(
                    cross_label_dynamic_center_distance_ratio
                    if enable_same_label_dynamic_spatial_gate
                    else human_label_dynamic_center_distance_ratio
                ),
                min_iou=cross_label_dynamic_min_iou,
            )
        ):
            continue
        if (
            not current_is_static
            and current_label != candidate.label
            and not cross_label_dynamic_spatially_compatible(
                current_bbox=current_bbox,
                candidate_bbox=candidate.last_bbox,
                max_center_distance_ratio=(
                    cross_label_dynamic_center_distance_ratio
                ),
                min_iou=cross_label_dynamic_min_iou,
            )
        ):
            continue
        filtered.append(candidate)
    return filtered


def labels_compatible(
    *,
    current_label: str,
    candidate_label: str,
    label_compatibility_groups: Sequence[set[str]],
) -> bool:
    if current_label == candidate_label:
        return True
    return any(
        current_label in group and candidate_label in group
        for group in label_compatibility_groups
    )


def label_uses_dynamic_spatial_gate(
    *,
    label: str,
    label_compatibility_groups: Sequence[set[str]],
) -> bool:
    return any(
        label in group and len(group) > 1
        for group in label_compatibility_groups
    )


def temporal_continuity_compatible(
    *,
    candidate: AssociationCandidate,
    current_label: str,
    current_bbox: BoundingBox,
    current_object_index: int,
    current_state_values: Sequence[str],
    current_embedding: np.ndarray,
    continuity_max_center_distance_ratio: float | None,
    continuity_min_area_ratio: float | None,
    object_index_jump_max_delta: int | None,
    object_index_jump_embedding_veto_threshold: float | None,
    state_conflict_embedding_veto_threshold: float | None,
    state_conflict_min_token_overlap_ratio: float,
    label_conflict_embedding_veto_threshold: float | None,
) -> bool:
    if (
        continuity_max_center_distance_ratio is not None
        and not center_motion_within_ratio(
            current_bbox=current_bbox,
            candidate_bbox=candidate.last_bbox,
            max_center_distance_ratio=continuity_max_center_distance_ratio,
        )
    ):
        return False
    if (
        continuity_min_area_ratio is not None
        and current_bbox.area_ratio(candidate.last_bbox) < continuity_min_area_ratio
    ):
        return False
    if (
        object_index_jump_max_delta is not None
        and abs(current_object_index - candidate.object_index)
        > max(0, object_index_jump_max_delta)
        and not _passes_embedding_veto(
            current_embedding=current_embedding,
            candidate_embedding=candidate.embedding,
            threshold=object_index_jump_embedding_veto_threshold,
        )
    ):
        return False
    if (
        label_conflict_embedding_veto_threshold is not None
        and current_label != candidate.label
        and not _passes_embedding_veto(
            current_embedding=current_embedding,
            candidate_embedding=candidate.embedding,
            threshold=label_conflict_embedding_veto_threshold,
        )
    ):
        return False
    if (
        state_conflict_embedding_veto_threshold is not None
        and state_values_conflict(
            current_state_values,
            candidate.last_state_values,
            min_token_overlap_ratio=state_conflict_min_token_overlap_ratio,
        )
        and not _passes_embedding_veto(
            current_embedding=current_embedding,
            candidate_embedding=candidate.embedding,
            threshold=state_conflict_embedding_veto_threshold,
        )
    ):
        return False
    return True


def center_motion_within_ratio(
    *,
    current_bbox: BoundingBox,
    candidate_bbox: BoundingBox,
    max_center_distance_ratio: float,
) -> bool:
    center_distance = current_bbox.center_distance(candidate_bbox)
    max_diagonal = max(current_bbox.diagonal, candidate_bbox.diagonal, 1.0)
    return center_distance / max_diagonal <= max_center_distance_ratio


def state_values_conflict(
    current_state_values: Sequence[str],
    candidate_state_values: Sequence[str],
    *,
    min_token_overlap_ratio: float,
) -> bool:
    current_tokens = state_value_tokens(current_state_values)
    candidate_tokens = state_value_tokens(candidate_state_values)
    if not current_tokens or not candidate_tokens:
        return False
    overlap = current_tokens & candidate_tokens
    overlap_ratio = len(overlap) / max(1, min(len(current_tokens), len(candidate_tokens)))
    return overlap_ratio < max(0.0, min_token_overlap_ratio)


def state_value_tokens(values: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[0-9a-z]+", str(value).lower()):
            if len(token) <= 1:
                continue
            tokens.add(token)
    return tokens


def find_local_tracklet_match(
    *,
    candidates: Sequence[AssociationCandidate],
    current: AssociationInput,
    max_frame_gap: int,
    embedding_similarity_threshold: float,
    min_score: float,
    score_margin: float,
    motion_weight: float,
    embedding_weight: float,
    iou_weight: float,
    object_index_weight: float,
) -> tuple[AssociationCandidate, float] | None:
    scored: list[tuple[AssociationCandidate, float, float]] = []
    for candidate in candidates:
        gap = current.current_frame_rank - candidate.last_frame_rank
        if gap <= 0 or gap > max(1, max_frame_gap):
            continue
        if candidate.embedding is None:
            continue
        embedding_similarity = cosine_similarity(
            current.current_embedding,
            candidate.embedding,
        )
        if embedding_similarity < embedding_similarity_threshold:
            continue
        score = local_tracklet_score(
            candidate=candidate,
            current=current,
            embedding_similarity=embedding_similarity,
            motion_weight=motion_weight,
            embedding_weight=embedding_weight,
            iou_weight=iou_weight,
            object_index_weight=object_index_weight,
        )
        if score >= min_score:
            scored.append((candidate, score, embedding_similarity))

    if not scored:
        return None
    scored.sort(
        key=lambda item: (item[1], item[2], -item[0].observation_count),
        reverse=True,
    )
    best_candidate, best_score, _ = scored[0]
    if len(scored) >= 2 and best_score - scored[1][1] < max(0.0, score_margin):
        return None
    return best_candidate, best_score


def local_tracklet_score(
    *,
    candidate: AssociationCandidate,
    current: AssociationInput,
    embedding_similarity: float,
    motion_weight: float,
    embedding_weight: float,
    iou_weight: float,
    object_index_weight: float,
) -> float:
    motion_score = center_motion_score(
        current_bbox=current.current_bbox,
        candidate_bbox=candidate.last_bbox,
    )
    iou_score = current.current_bbox.iou(candidate.last_bbox)
    object_index_score = object_index_continuity_score(
        current_object_index=current.object_index,
        candidate_object_index=candidate.object_index,
    )
    total_weight = (
        max(0.0, motion_weight)
        + max(0.0, embedding_weight)
        + max(0.0, iou_weight)
        + max(0.0, object_index_weight)
    )
    if total_weight <= 0.0:
        return embedding_similarity
    return (
        max(0.0, motion_weight) * motion_score
        + max(0.0, embedding_weight) * embedding_similarity
        + max(0.0, iou_weight) * iou_score
        + max(0.0, object_index_weight) * object_index_score
    ) / total_weight


def center_motion_score(
    *,
    current_bbox: BoundingBox,
    candidate_bbox: BoundingBox,
) -> float:
    center_distance = current_bbox.center_distance(candidate_bbox)
    scale = max(
        min(current_bbox.diagonal, candidate_bbox.diagonal),
        1.0,
    )
    normalized_distance = center_distance / scale
    return max(0.0, 1.0 - normalized_distance)


def object_index_continuity_score(
    *,
    current_object_index: int,
    candidate_object_index: int,
) -> float:
    delta = abs(current_object_index - candidate_object_index)
    return 1.0 / (1.0 + float(delta))


def find_fast_match(
    *,
    candidates: Sequence[AssociationCandidate],
    current_bbox: BoundingBox,
    current_embedding: np.ndarray,
    fast_iou_threshold: float,
    static_fast_iou_threshold: float,
    static_center_distance_ratio: float,
    fast_match_embedding_veto_threshold: float | None,
    static_fast_match_embedding_veto_threshold: float | None,
) -> tuple[AssociationCandidate, float] | None:
    best_match: AssociationCandidate | None = None
    best_score = -1.0
    for candidate in candidates:
        if candidate.is_static:
            score = static_spatial_score(
                current_bbox=current_bbox,
                candidate_bbox=candidate.last_bbox,
                static_fast_iou_threshold=static_fast_iou_threshold,
                static_center_distance_ratio=static_center_distance_ratio,
            )
            if score < static_fast_iou_threshold or score <= best_score:
                continue
            if not _passes_embedding_veto(
                current_embedding=current_embedding,
                candidate_embedding=candidate.embedding,
                threshold=static_fast_match_embedding_veto_threshold,
            ):
                continue
            best_match = candidate
            best_score = score
            continue

        score = current_bbox.iou(candidate.last_bbox)
        if score >= fast_iou_threshold and score > best_score:
            if not _passes_embedding_veto(
                current_embedding=current_embedding,
                candidate_embedding=candidate.embedding,
                threshold=fast_match_embedding_veto_threshold,
            ):
                continue
            best_match = candidate
            best_score = score

    if best_match is None:
        return None
    return best_match, best_score


def _passes_embedding_veto(
    *,
    current_embedding: np.ndarray,
    candidate_embedding: np.ndarray | None,
    threshold: float | None,
) -> bool:
    if threshold is None:
        return True
    if candidate_embedding is None:
        return False
    return cosine_similarity(current_embedding, candidate_embedding) >= threshold


def find_slow_match(
    *,
    ranked_matches: Sequence[tuple[AssociationCandidate, float]],
    embedding_similarity_threshold: float,
) -> tuple[AssociationCandidate, float] | None:
    if not ranked_matches:
        return None
    best_match, best_similarity = ranked_matches[0]
    if best_similarity < embedding_similarity_threshold:
        return None
    return best_match, best_similarity


def rank_slow_matches(
    *,
    candidates: Sequence[AssociationCandidate],
    current_embedding: np.ndarray,
) -> list[tuple[AssociationCandidate, float]]:
    ranked: list[tuple[AssociationCandidate, float]] = []
    for candidate in candidates:
        if candidate.embedding is None:
            continue
        ranked.append(
            (candidate, cosine_similarity(current_embedding, candidate.embedding))
        )
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return -1.0
    return float(np.dot(left, right) / denominator)


def static_spatial_score(
    *,
    current_bbox: BoundingBox,
    candidate_bbox: BoundingBox,
    static_fast_iou_threshold: float,
    static_center_distance_ratio: float,
) -> float:
    iou = current_bbox.iou(candidate_bbox)
    if iou >= static_fast_iou_threshold:
        return iou

    center_distance = current_bbox.center_distance(candidate_bbox)
    max_diagonal = max(current_bbox.diagonal, candidate_bbox.diagonal, 1.0)
    normalized_distance = center_distance / max_diagonal
    area_ratio = current_bbox.area_ratio(candidate_bbox)
    if (
        normalized_distance <= static_center_distance_ratio
        and area_ratio >= 0.25
    ):
        return max(iou, area_ratio * (1.0 - normalized_distance))
    return iou


def cross_label_dynamic_spatially_compatible(
    *,
    current_bbox: BoundingBox,
    candidate_bbox: BoundingBox,
    max_center_distance_ratio: float,
    min_iou: float,
) -> bool:
    if current_bbox.iou(candidate_bbox) >= min_iou:
        return True
    center_distance = current_bbox.center_distance(candidate_bbox)
    max_diagonal = max(current_bbox.diagonal, candidate_bbox.diagonal, 1.0)
    normalized_distance = center_distance / max_diagonal
    return normalized_distance <= max_center_distance_ratio
