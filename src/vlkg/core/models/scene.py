from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from vlkg.core.config.config import SceneSegmentationConfig
from vlkg.core.models.temporal_graph import Observation


@dataclass(frozen=True)
class FrameSceneSignature:
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    static_entity_ids: frozenset[str]
    static_label_counts: Counter[str]


@dataclass(frozen=True)
class SceneRange:
    scene_index: int
    start_frame_rank: int
    end_frame_rank: int
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    previous_boundary_reason: str | None = None
    next_boundary_reason: str | None = None
    boundary_from_previous_frames: int | None = None
    boundary_to_next_frames: int | None = None
    boundary_from_previous_time_gap_sec: float | None = None
    boundary_to_next_time_gap_sec: float | None = None
    boundary_from_previous_entity_similarity: float | None = None
    boundary_to_next_entity_similarity: float | None = None
    boundary_from_previous_label_similarity: float | None = None
    boundary_to_next_label_similarity: float | None = None
    entry_static_labels: tuple[str, ...] = ()
    exit_static_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneBoundary:
    left_end_idx: int
    right_start_idx: int
    reason: str
    frame_gap: int
    time_gap_sec: float
    entity_similarity: float | None
    label_similarity: float | None
    left_static_labels: tuple[str, ...]
    right_static_labels: tuple[str, ...]


def build_frame_scene_signatures(
    observations: Iterable[Observation],
    *,
    min_anchor_entity_support_frames: int = 1,
) -> list[FrameSceneSignature]:
    min_anchor_entity_support_frames = max(1, min_anchor_entity_support_frames)
    static_entity_support: dict[str, set[int]] = defaultdict(set)
    materialized_observations = list(observations)
    for observation in materialized_observations:
        if observation.is_static and not observation.is_noise:
            static_entity_support[observation.entity_id].add(observation.frame_rank)

    grouped: dict[tuple[int, int, float], list[Observation]] = defaultdict(list)
    for observation in materialized_observations:
        grouped[
            (
                observation.frame_rank,
                observation.frame_index,
                observation.timestamp_sec,
            )
        ].append(observation)

    signatures: list[FrameSceneSignature] = []
    for (frame_rank, frame_index, timestamp_sec), frame_observations in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        anchor_observations = [
            observation
            for observation in frame_observations
            if observation.is_static
            and not observation.is_noise
            and len(static_entity_support.get(observation.entity_id, ()))
            >= min_anchor_entity_support_frames
        ]
        static_entities = {
            observation.entity_id
            for observation in anchor_observations
        }
        static_label_counts: Counter[str] = Counter(
            observation.label
            for observation in anchor_observations
        )
        signatures.append(
            FrameSceneSignature(
                frame_rank=frame_rank,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                static_entity_ids=frozenset(static_entities),
                static_label_counts=static_label_counts,
            )
        )
    return signatures


def detect_scene_ranges(
    frame_signatures: list[FrameSceneSignature],
    *,
    config: SceneSegmentationConfig,
) -> list[SceneRange]:
    if not frame_signatures:
        return []
    if not config.enabled or len(frame_signatures) == 1:
        return [_scene_range_from_slice(frame_signatures, 0, len(frame_signatures) - 1, 1)]

    boundary_window_size = max(1, config.boundary_window_size)

    scene_ranges: list[SceneRange] = []
    boundaries: list[SceneBoundary] = []
    scene_start_idx = 0
    pending_boundary_idx: int | None = None
    pending_anchor: FrameSceneSignature | None = None
    pending_confirmation = 0

    for current_idx in range(1, len(frame_signatures)):
        previous_signature = frame_signatures[current_idx - 1]
        current_signature = frame_signatures[current_idx]
        left_signature = build_window_scene_signature(
            frame_signatures,
            start_idx=max(scene_start_idx, current_idx - boundary_window_size),
            end_idx=current_idx - 1,
        )
        right_signature = build_window_scene_signature(
            frame_signatures,
            start_idx=current_idx,
            end_idx=min(
                len(frame_signatures) - 1,
                current_idx + boundary_window_size - 1,
            ),
        )

        proposed_scene_length = current_idx - scene_start_idx
        hard_gap = (
            current_signature.timestamp_sec - previous_signature.timestamp_sec
            >= config.max_timestamp_gap_sec
        )
        if hard_gap and proposed_scene_length > 0:
            boundaries.append(
                _build_scene_boundary(
                    previous_signature=previous_signature,
                    current_signature=current_signature,
                    reason="hard_gap",
                )
            )
            scene_ranges.append(
                _scene_range_from_slice(
                    frame_signatures,
                    scene_start_idx,
                    current_idx - 1,
                    len(scene_ranges) + 1,
                )
            )
            scene_start_idx = current_idx
            pending_boundary_idx = None
            pending_anchor = None
            pending_confirmation = 0
            continue

        if pending_boundary_idx is not None and pending_anchor is not None:
            if matches_pending_scene_anchor(
                anchor_signature=pending_anchor,
                current_signature=right_signature,
                config=config,
            ):
                pending_confirmation += 1
            else:
                pending_boundary_idx = None
                pending_anchor = None
                pending_confirmation = 0

            if (
                pending_boundary_idx is not None
                and pending_confirmation >= config.boundary_confirmation_frames
                and pending_boundary_idx - scene_start_idx >= config.min_scene_length
            ):
                boundaries.append(
                    _build_scene_boundary(
                        previous_signature=frame_signatures[pending_boundary_idx - 1],
                        current_signature=frame_signatures[pending_boundary_idx],
                        reason="static_signature_shift",
                    )
                )
                scene_ranges.append(
                    _scene_range_from_slice(
                        frame_signatures,
                        scene_start_idx,
                        pending_boundary_idx - 1,
                        len(scene_ranges) + 1,
                    )
                )
                scene_start_idx = pending_boundary_idx
                pending_boundary_idx = None
                pending_anchor = None
                pending_confirmation = 0
            continue

        if proposed_scene_length < config.min_scene_length:
            continue

        if not is_scene_boundary_candidate(
            previous_signature=left_signature,
            current_signature=right_signature,
            config=config,
        ):
            continue

        pending_boundary_idx = current_idx
        pending_anchor = right_signature
        pending_confirmation = 1

    scene_ranges.append(
        _scene_range_from_slice(
            frame_signatures,
            scene_start_idx,
            len(frame_signatures) - 1,
            len(scene_ranges) + 1,
        )
    )
    return _attach_scene_boundary_metadata(
        scene_ranges=scene_ranges,
        boundaries=boundaries,
        frame_signatures=frame_signatures,
    )


def is_scene_boundary_candidate(
    *,
    previous_signature: FrameSceneSignature,
    current_signature: FrameSceneSignature,
    config: SceneSegmentationConfig,
) -> bool:
    previous_static_count = len(previous_signature.static_entity_ids)
    current_static_count = len(current_signature.static_entity_ids)
    if (
        previous_static_count < config.min_static_entities
        or current_static_count < config.min_static_entities
    ):
        return False

    entity_similarity = jaccard_similarity(
        previous_signature.static_entity_ids,
        current_signature.static_entity_ids,
    )
    label_similarity = weighted_jaccard_similarity(
        previous_signature.static_label_counts,
        current_signature.static_label_counts,
    )
    return (
        entity_similarity <= config.static_entity_jaccard_threshold
        and label_similarity <= config.static_label_jaccard_threshold
    )


def matches_pending_scene_anchor(
    *,
    anchor_signature: FrameSceneSignature,
    current_signature: FrameSceneSignature,
    config: SceneSegmentationConfig,
) -> bool:
    entity_similarity = jaccard_similarity(
        anchor_signature.static_entity_ids,
        current_signature.static_entity_ids,
    )
    label_similarity = weighted_jaccard_similarity(
        anchor_signature.static_label_counts,
        current_signature.static_label_counts,
    )
    return (
        entity_similarity > config.static_entity_jaccard_threshold
        or label_similarity > config.static_label_jaccard_threshold
    )


def build_window_scene_signature(
    frame_signatures: list[FrameSceneSignature],
    *,
    start_idx: int,
    end_idx: int,
) -> FrameSceneSignature:
    window = frame_signatures[start_idx : end_idx + 1]
    if not window:
        raise ValueError("Window signature requires at least one frame signature.")

    support_threshold = max(1, math.ceil(len(window) / 2))
    entity_support: Counter[str] = Counter()
    label_support: Counter[str] = Counter()
    for signature in window:
        entity_support.update(signature.static_entity_ids)
        label_support.update(signature.static_label_counts.keys())

    stable_entities = frozenset(
        entity_id
        for entity_id, support in entity_support.items()
        if support >= support_threshold
    )
    stable_labels = Counter(
        {
            label: support
            for label, support in label_support.items()
            if support >= support_threshold
        }
    )
    representative_signature = window[len(window) // 2]
    return FrameSceneSignature(
        frame_rank=representative_signature.frame_rank,
        frame_index=representative_signature.frame_index,
        timestamp_sec=representative_signature.timestamp_sec,
        static_entity_ids=stable_entities,
        static_label_counts=stable_labels,
    )


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def weighted_jaccard_similarity(
    left: Counter[str],
    right: Counter[str],
) -> float:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return 1.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    if union <= 0:
        return 1.0
    return intersection / union


def summarize_static_labels(
    signature: FrameSceneSignature,
    *,
    limit: int = 4,
) -> tuple[str, ...]:
    labels = [
        label
        for label, _ in sorted(
            signature.static_label_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]
    return tuple(labels)


def _build_scene_boundary(
    *,
    previous_signature: FrameSceneSignature,
    current_signature: FrameSceneSignature,
    reason: str,
) -> SceneBoundary:
    return SceneBoundary(
        left_end_idx=previous_signature.frame_rank,
        right_start_idx=current_signature.frame_rank,
        reason=reason,
        frame_gap=current_signature.frame_index - previous_signature.frame_index,
        time_gap_sec=current_signature.timestamp_sec - previous_signature.timestamp_sec,
        entity_similarity=jaccard_similarity(
            previous_signature.static_entity_ids,
            current_signature.static_entity_ids,
        ),
        label_similarity=weighted_jaccard_similarity(
            previous_signature.static_label_counts,
            current_signature.static_label_counts,
        ),
        left_static_labels=summarize_static_labels(previous_signature),
        right_static_labels=summarize_static_labels(current_signature),
    )


def _attach_scene_boundary_metadata(
    *,
    scene_ranges: list[SceneRange],
    boundaries: list[SceneBoundary],
    frame_signatures: list[FrameSceneSignature],
) -> list[SceneRange]:
    if not scene_ranges:
        return []

    boundary_by_left_end_rank = {
        boundary.left_end_idx: boundary
        for boundary in boundaries
    }
    boundary_by_right_start_rank = {
        boundary.right_start_idx: boundary
        for boundary in boundaries
    }
    signature_by_rank = {
        signature.frame_rank: signature
        for signature in frame_signatures
    }

    enriched_ranges: list[SceneRange] = []
    for scene_range in scene_ranges:
        previous_boundary = boundary_by_right_start_rank.get(scene_range.start_frame_rank)
        next_boundary = boundary_by_left_end_rank.get(scene_range.end_frame_rank)
        start_signature = signature_by_rank.get(scene_range.start_frame_rank)
        end_signature = signature_by_rank.get(scene_range.end_frame_rank)
        enriched_ranges.append(
            SceneRange(
                scene_index=scene_range.scene_index,
                start_frame_rank=scene_range.start_frame_rank,
                end_frame_rank=scene_range.end_frame_rank,
                start_frame=scene_range.start_frame,
                end_frame=scene_range.end_frame,
                start_ts=scene_range.start_ts,
                end_ts=scene_range.end_ts,
                support_frames=scene_range.support_frames,
                previous_boundary_reason=(
                    previous_boundary.reason if previous_boundary is not None else None
                ),
                next_boundary_reason=(
                    next_boundary.reason if next_boundary is not None else None
                ),
                boundary_from_previous_frames=(
                    previous_boundary.frame_gap if previous_boundary is not None else None
                ),
                boundary_to_next_frames=(
                    next_boundary.frame_gap if next_boundary is not None else None
                ),
                boundary_from_previous_time_gap_sec=(
                    previous_boundary.time_gap_sec
                    if previous_boundary is not None
                    else None
                ),
                boundary_to_next_time_gap_sec=(
                    next_boundary.time_gap_sec if next_boundary is not None else None
                ),
                boundary_from_previous_entity_similarity=(
                    previous_boundary.entity_similarity
                    if previous_boundary is not None
                    else None
                ),
                boundary_to_next_entity_similarity=(
                    next_boundary.entity_similarity if next_boundary is not None else None
                ),
                boundary_from_previous_label_similarity=(
                    previous_boundary.label_similarity
                    if previous_boundary is not None
                    else None
                ),
                boundary_to_next_label_similarity=(
                    next_boundary.label_similarity if next_boundary is not None else None
                ),
                entry_static_labels=(
                    previous_boundary.right_static_labels
                    if previous_boundary is not None
                    else summarize_static_labels(start_signature)
                    if start_signature is not None
                    else ()
                ),
                exit_static_labels=(
                    next_boundary.left_static_labels
                    if next_boundary is not None
                    else summarize_static_labels(end_signature)
                    if end_signature is not None
                    else ()
                ),
            )
        )
    return enriched_ranges


def _scene_range_from_slice(
    frame_signatures: list[FrameSceneSignature],
    start_idx: int,
    end_idx: int,
    scene_index: int,
) -> SceneRange:
    start_signature = frame_signatures[start_idx]
    end_signature = frame_signatures[end_idx]
    return SceneRange(
        scene_index=scene_index,
        start_frame_rank=start_signature.frame_rank,
        end_frame_rank=end_signature.frame_rank,
        start_frame=start_signature.frame_index,
        end_frame=end_signature.frame_index,
        start_ts=start_signature.timestamp_sec,
        end_ts=end_signature.timestamp_sec,
        support_frames=end_idx - start_idx + 1,
    )
