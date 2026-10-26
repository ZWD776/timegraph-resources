from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.models.stable_identity import StableIdentityDocument
from vlkg.core.transformers.temporal_graph_to_stable_identity import (
    build_stable_identity_document,
)

HUMAN_LABELS = {"player", "person", "people", "man", "woman", "boy", "girl"}
BALL_LABELS = {"basketball", "ball"}
KEY_OBJECT_LABELS = {"basketball", "ball", "scoreboard"}


def build_improvement_maturity_metrics(
    document: TemporalGraphDocument,
    *,
    stable_identity_document: StableIdentityDocument | None = None,
    canonical_graph: CanonicalTemporalMemoryGraph | None = None,
) -> dict[str, Any]:
    """Summarize the three original optimization goals without mutating graphs."""

    stable_identity_document = (
        stable_identity_document or build_stable_identity_document(document)
    )

    return {
        "time_axis": _time_axis_metrics(document),
        "base_delta": _base_delta_metrics(document),
        "entity_identity": _identity_metrics(
            document,
            stable_identity_document,
        ),
        "canonical_graph": (
            _canonical_graph_metrics(canonical_graph)
            if canonical_graph is not None
            else None
        ),
    }


def _time_axis_metrics(document: TemporalGraphDocument) -> dict[str, Any]:
    entity_ids = {entity.entity_id for entity in document.entities}
    segmented_entity_ids = {
        segment.entity_id
        for segment in document.entity_segments
    }
    segments_by_entity: dict[str, int] = defaultdict(int)
    for segment in document.entity_segments:
        segments_by_entity[segment.entity_id] += 1

    event_type_counts = Counter(
        event.event_type
        for event in document.transition_events
    )

    return {
        "frame_count": document.frame_count,
        "scene_segment_count": len(document.scene_segments),
        "entity_segment_count": len(document.entity_segments),
        "state_episode_count": len(document.state_episodes),
        "relation_episode_count": len(document.relation_episodes),
        "transition_event_count": len(document.transition_events),
        "event_type_counts": _sorted_counter(event_type_counts),
        "entity_segment_coverage_ratio": _safe_ratio(
            len(segmented_entity_ids & entity_ids),
            len(entity_ids),
        ),
        "avg_entity_segments_per_entity": _safe_ratio(
            len(document.entity_segments),
            len(entity_ids),
        ),
        "multi_segment_entity_count": sum(
            1
            for count in segments_by_entity.values()
            if count > 1
        ),
        "has_temporal_axis": bool(
            document.frame_count
            and (
                document.scene_segments
                or document.entity_segments
                or document.state_episodes
                or document.relation_episodes
                or document.transition_events
            )
        ),
    }


def _base_delta_metrics(document: TemporalGraphDocument) -> dict[str, Any]:
    static_entity_count = sum(1 for entity in document.entities if entity.is_static)
    noise_entity_count = sum(1 for entity in document.entities if entity.is_noise)
    fragment_entity_count = sum(1 for entity in document.entities if entity.is_fragment)
    dynamic_entity_count = len(document.entities) - static_entity_count

    return {
        "static_entity_count": static_entity_count,
        "dynamic_entity_count": dynamic_entity_count,
        "noise_entity_count": noise_entity_count,
        "fragment_entity_count": fragment_entity_count,
        "scene_base_node_count": len(document.scene_base_nodes),
        "scene_base_edge_count": len(document.scene_base_edges),
        "scene_delta_node_count": len(document.scene_delta_nodes),
        "scene_delta_observation_count": len(document.scene_delta_observations),
        "scene_delta_edge_count": len(document.scene_delta_edges),
        "delta_to_observation_ratio": _safe_ratio(
            len(document.scene_delta_observations),
            len(document.observations),
        ),
        "delta_to_entity_ratio": _safe_ratio(
            len(document.scene_delta_nodes),
            len(document.entities),
        ),
        "base_delta_available": bool(
            document.scene_base_nodes
            or document.scene_delta_nodes
            or document.scene_delta_observations
        ),
    }


def _identity_metrics(
    document: TemporalGraphDocument,
    stable_identity_document: StableIdentityDocument,
) -> dict[str, Any]:
    identities = stable_identity_document.identities
    member_count_distribution = Counter(
        len(identity.member_entities)
        for identity in identities
    )
    identity_type_counts = Counter(
        identity.identity_type
        for identity in identities
    )
    covered_entity_ids = {
        member.entity_id
        for identity in identities
        for member in identity.member_entities
    }
    eligible_entity_ids = {
        entity.entity_id
        for entity in document.entities
        if not entity.is_noise and not entity.is_fragment
    }
    singleton_identity_count = member_count_distribution.get(1, 0)
    merged_identity_count = sum(
        count
        for member_count, count in member_count_distribution.items()
        if member_count > 1
    )
    total_members = sum(
        len(identity.member_entities)
        for identity in identities
    )
    quality_diagnostics = _identity_quality_diagnostics(
        document,
        stable_identity_document,
    )

    return {
        "stable_identity_count": len(identities),
        "covered_entity_count": len(covered_entity_ids),
        "eligible_entity_count": len(eligible_entity_ids),
        "entity_coverage_ratio": _safe_ratio(
            len(covered_entity_ids & eligible_entity_ids),
            len(eligible_entity_ids),
        ),
        "singleton_identity_count": singleton_identity_count,
        "merged_identity_count": merged_identity_count,
        "singleton_identity_ratio": _safe_ratio(
            singleton_identity_count,
            len(identities),
        ),
        "avg_members_per_identity": _safe_ratio(
            total_members,
            len(identities),
        ),
        "member_count_distribution": _sorted_counter(member_count_distribution),
        "identity_type_counts": _sorted_counter(identity_type_counts),
        "identity_view_mode": (
            "conservative"
            if singleton_identity_count >= merged_identity_count
            else "merged"
        ),
        "quality_diagnostics": quality_diagnostics,
    }


def _identity_quality_diagnostics(
    document: TemporalGraphDocument,
    stable_identity_document: StableIdentityDocument,
) -> dict[str, Any]:
    stable_by_entity = {
        member.entity_id: identity
        for identity in stable_identity_document.identities
        for member in identity.member_entities
    }
    relation_count_by_entity: Counter[str] = Counter()
    for episode in document.relation_episodes:
        relation_count_by_entity[episode.subject_entity_id] += 1
        relation_count_by_entity[episode.object_entity_id] += 1

    state_count_by_entity = Counter(
        episode.entity_id
        for episode in document.state_episodes
    )
    segment_count_by_entity = Counter(
        segment.entity_id
        for segment in document.entity_segments
    )
    observations_by_entity: dict[str, list[Any]] = defaultdict(list)
    for observation in document.observations:
        observations_by_entity[observation.entity_id].append(observation)

    features = [
        _entity_quality_feature(
            entity,
            stable_by_entity=stable_by_entity,
            relation_count=relation_count_by_entity[entity.entity_id],
            state_count=state_count_by_entity[entity.entity_id],
            segment_count=segment_count_by_entity[entity.entity_id],
            observations=observations_by_entity[entity.entity_id],
        )
        for entity in document.entities
    ]

    weak_candidates = [
        feature
        for feature in features
        if feature["flags"]
        and (
            feature["observation_count"] <= 8
            or feature["segment_count"] >= 3
            or "key_object" in feature["flags"]
        )
    ]
    weak_human_singletons = [
        feature
        for feature in weak_candidates
        if feature["identity_type"] == "human"
        and feature["stable_member_count"] <= 1
    ]
    sparse_key_objects = [
        feature
        for feature in weak_candidates
        if "key_object" in feature["flags"]
    ]
    merge_suggestions = _identity_merge_suggestions(features)
    uncovered_merge_suggestions = [
        suggestion
        for suggestion in merge_suggestions
        if not suggestion["same_stable_identity"]
    ]
    covered_merge_suggestions = [
        suggestion
        for suggestion in merge_suggestions
        if suggestion["same_stable_identity"]
    ]
    review_suggestions = [
        suggestion
        for suggestion in uncovered_merge_suggestions
        if suggestion["verdict"] == "review_candidate"
    ]
    already_covered_review_suggestions = [
        suggestion
        for suggestion in covered_merge_suggestions
        if suggestion["verdict"] == "review_candidate"
    ]
    blocked_overlap_suggestions = [
        suggestion
        for suggestion in uncovered_merge_suggestions
        if suggestion["verdict"] == "blocked_by_time_overlap"
    ]
    weak_similarity_suggestions = [
        suggestion
        for suggestion in uncovered_merge_suggestions
        if suggestion["verdict"] == "weak_similarity_only"
    ]

    return {
        "weak_candidate_count": len(weak_candidates),
        "weak_candidate_type_counts": _sorted_counter(
            Counter(feature["identity_type"] for feature in weak_candidates)
        ),
        "weak_human_singleton_count": len(weak_human_singletons),
        "weak_human_singletons": [
            _compact_identity_feature(feature)
            for feature in sorted(
                weak_human_singletons,
                key=_identity_feature_sort_key,
            )[:10]
        ],
        "sparse_key_object_count": len(sparse_key_objects),
        "sparse_key_objects": [
            _compact_identity_feature(feature)
            for feature in sorted(
                sparse_key_objects,
                key=_identity_feature_sort_key,
            )[:10]
        ],
        "merge_suggestion_count": len(merge_suggestions),
        "merge_suggestion_verdict_counts": _sorted_counter(
            Counter(suggestion["verdict"] for suggestion in merge_suggestions)
        ),
        "uncovered_merge_suggestion_verdict_counts": _sorted_counter(
            Counter(suggestion["verdict"] for suggestion in uncovered_merge_suggestions)
        ),
        "already_covered_merge_suggestion_verdict_counts": _sorted_counter(
            Counter(suggestion["verdict"] for suggestion in covered_merge_suggestions)
        ),
        "review_merge_suggestion_count": len(review_suggestions),
        "already_covered_review_merge_suggestion_count": len(
            already_covered_review_suggestions
        ),
        "top_review_merge_suggestions": review_suggestions[:8],
        "top_blocked_overlap_merge_suggestions": blocked_overlap_suggestions[:8],
        "top_weak_similarity_merge_suggestions": weak_similarity_suggestions[:8],
    }


def _entity_quality_feature(
    entity: Any,
    *,
    stable_by_entity: dict[str, Any],
    relation_count: int,
    state_count: int,
    segment_count: int,
    observations: list[Any],
) -> dict[str, Any]:
    stable_identity = stable_by_entity.get(entity.entity_id)
    labels = {
        str(value).lower()
        for value in [
            entity.category,
            *entity.labels_seen,
            *entity.raw_labels_seen,
        ]
        if str(value).strip()
    }
    identity_type = (
        stable_identity.identity_type
        if stable_identity is not None
        else _entity_identity_type(entity)
    )
    stable_member_count = (
        len(stable_identity.member_entities)
        if stable_identity is not None
        else 0
    )
    flags: list[str] = []
    if entity.observation_count <= 5:
        flags.append("obs<=5")
    if segment_count >= 3:
        flags.append("many_segments")
    if (
        relation_count <= 1
        and state_count <= 1
        and identity_type == "human"
    ):
        flags.append("weak_human_semantics")
    if (
        relation_count == 0
        and state_count == 0
        and identity_type in {"ball", "static_object"}
    ):
        flags.append("weak_object_semantics")
    if entity.last_seen_frame - entity.first_seen_frame <= 30:
        flags.append("short_time_span")
    if labels & KEY_OBJECT_LABELS:
        flags.append("key_object")

    mean_center_x, mean_center_y, mean_area = _mean_bbox_profile(observations)

    return {
        "entity_id": entity.entity_id,
        "short_id": _short_entity_id(entity.entity_id),
        "category": entity.category,
        "identity_type": identity_type,
        "stable_identity": (
            stable_identity.display_name
            if stable_identity is not None
            else None
        ),
        "stable_id": (
            stable_identity.stable_id
            if stable_identity is not None
            else None
        ),
        "stable_member_count": stable_member_count,
        "observation_count": entity.observation_count,
        "first_seen_frame": entity.first_seen_frame,
        "last_seen_frame": entity.last_seen_frame,
        "segment_count": segment_count,
        "state_episode_count": state_count,
        "relation_episode_count": relation_count,
        "flags": flags,
        "embedding": entity.canonical_embedding or [],
        "signature_tokens": _signature_tokens(
            [
                entity.category,
                *entity.labels_seen,
                *entity.raw_labels_seen,
                *entity.states_seen,
                *entity.raw_states_seen,
                *entity.parent_labels_seen,
            ]
        ),
        "mean_center_x": mean_center_x,
        "mean_center_y": mean_center_y,
        "mean_area": mean_area,
    }


def _entity_identity_type(entity: Any) -> str:
    labels = {
        str(value).lower()
        for value in [
            entity.category,
            *entity.labels_seen,
            *entity.raw_labels_seen,
        ]
        if str(value).strip()
    }
    if labels & HUMAN_LABELS:
        return "human"
    if labels & BALL_LABELS:
        return "ball"
    if entity.is_static:
        return "static_object"
    return "object"


def _identity_merge_suggestions(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    humans = [
        feature
        for feature in features
        if feature["identity_type"] == "human"
    ]
    suggestions: list[dict[str, Any]] = []
    for index, left in enumerate(humans):
        for right in humans[index + 1:]:
            if min(left["observation_count"], right["observation_count"]) > 5:
                continue
            similarity = _cosine_similarity(left["embedding"], right["embedding"])
            if similarity < 0.85:
                continue
            overlap = _frame_overlap(left, right)
            distance = _bbox_distance(left, right)
            ratio = _area_ratio(left, right)
            if overlap <= 3 and distance <= 140 and ratio >= 0.70:
                verdict = "review_candidate"
            elif overlap > 3:
                verdict = "blocked_by_time_overlap"
            else:
                verdict = "weak_similarity_only"
            suggestions.append(
                {
                    "left_entity_id": left["entity_id"],
                    "right_entity_id": right["entity_id"],
                    "pair": [left["short_id"], right["short_id"]],
                    "embedding_similarity": round(similarity, 6),
                    "token_jaccard": round(_token_jaccard(left, right), 6),
                    "overlap": overlap,
                    "gap": _frame_gap(left, right),
                    "bbox_distance": round(distance, 6),
                    "area_ratio": round(ratio, 6),
                    "verdict": verdict,
                    "same_stable_identity": (
                        bool(left["stable_id"])
                        and left["stable_id"] == right["stable_id"]
                    ),
                }
            )

    verdict_rank = {
        "review_candidate": 0,
        "blocked_by_time_overlap": 1,
        "weak_similarity_only": 2,
    }
    return sorted(
        suggestions,
        key=lambda item: (
            verdict_rank[item["verdict"]],
            -item["embedding_similarity"],
            item["bbox_distance"],
        ),
    )


def _compact_identity_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": feature["entity_id"],
        "short_id": feature["short_id"],
        "category": feature["category"],
        "identity_type": feature["identity_type"],
        "stable_identity": feature["stable_identity"],
        "stable_member_count": feature["stable_member_count"],
        "observation_count": feature["observation_count"],
        "frame_range": [
            feature["first_seen_frame"],
            feature["last_seen_frame"],
        ],
        "segment_count": feature["segment_count"],
        "state_episode_count": feature["state_episode_count"],
        "relation_episode_count": feature["relation_episode_count"],
        "flags": list(feature["flags"]),
    }


def _identity_feature_sort_key(feature: dict[str, Any]) -> tuple[Any, ...]:
    return (
        feature["observation_count"],
        feature["first_seen_frame"],
        feature["category"],
        feature["short_id"],
    )


def _mean_bbox_profile(observations: list[Any]) -> tuple[float, float, float]:
    if not observations:
        return 0.0, 0.0, 0.0
    centers: list[tuple[float, float]] = []
    areas: list[float] = []
    for observation in observations:
        bbox = observation.bbox
        centers.append((bbox.center_x, bbox.center_y))
        areas.append(bbox.area)
    return (
        sum(x for x, _ in centers) / len(centers),
        sum(y for _, y in centers) / len(centers),
        sum(areas) / len(areas),
    )


def _signature_tokens(values: list[Any]) -> set[str]:
    color_tokens = {
        "black",
        "blue",
        "brown",
        "gold",
        "gray",
        "green",
        "maroon",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
    }
    wearable_tokens = {
        "dress",
        "jersey",
        "pants",
        "shirt",
        "shoes",
        "shorts",
        "sneakers",
        "sock",
        "socks",
        "uniform",
    }
    tokens: set[str] = set()
    for value in values:
        words = {
            token.strip(".,;:()[]{}").lower()
            for token in str(value).split()
        }
        for color in color_tokens:
            if color in words:
                tokens.add(f"color:{color}")
        for wearable in wearable_tokens:
            if wearable in words:
                tokens.add(f"wear:{wearable}")
    return tokens


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(vector_a, vector_b)) / (norm_a * norm_b)


def _frame_overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    start = max(left["first_seen_frame"], right["first_seen_frame"])
    end = min(left["last_seen_frame"], right["last_seen_frame"])
    return max(0, end - start + 1)


def _frame_gap(left: dict[str, Any], right: dict[str, Any]) -> int:
    if _frame_overlap(left, right) > 0:
        return 0
    earlier, later = sorted(
        [left, right],
        key=lambda item: item["first_seen_frame"],
    )
    return max(0, later["first_seen_frame"] - earlier["last_seen_frame"])


def _area_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_area = float(left.get("mean_area") or 0.0)
    right_area = float(right.get("mean_area") or 0.0)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    return min(left_area, right_area) / max(left_area, right_area)


def _bbox_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    dx = float(left["mean_center_x"]) - float(right["mean_center_x"])
    dy = float(left["mean_center_y"]) - float(right["mean_center_y"])
    return math.sqrt(dx * dx + dy * dy)


def _token_jaccard(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = left["signature_tokens"]
    right_tokens = right["signature_tokens"]
    if not left_tokens and not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _short_entity_id(entity_id: Any) -> str:
    return str(entity_id).split(":")[-1]


def _canonical_graph_metrics(
    graph: CanonicalTemporalMemoryGraph,
) -> dict[str, Any]:
    edge_type_counts = Counter(edge.edge_type for edge in graph.edges)
    episode_kind_counts = Counter(node.episode_kind for node in graph.episode_nodes)
    event_type_counts = Counter(
        node.event_type
        for node in graph.transition_event_nodes
    )

    return {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "scene_node_count": len(graph.scene_nodes),
        "entity_node_count": len(graph.entity_nodes),
        "stable_identity_node_count": len(graph.stable_identity_nodes),
        "state_value_node_count": len(graph.state_value_nodes),
        "interaction_type_node_count": len(graph.interaction_type_nodes),
        "episode_node_count": len(graph.episode_nodes),
        "transition_event_node_count": len(graph.transition_event_nodes),
        "relation_observed_sample_count": len(graph.relation_observed_samples),
        "episode_kind_counts": _sorted_counter(episode_kind_counts),
        "event_type_counts": _sorted_counter(event_type_counts),
        "edge_type_counts": _sorted_counter(edge_type_counts),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter.keys(), key=lambda item: str(item))
    }
