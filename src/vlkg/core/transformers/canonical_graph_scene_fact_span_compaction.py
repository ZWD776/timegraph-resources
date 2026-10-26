from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)


@dataclass(frozen=True)
class SceneFactSpanCompactionResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def compact_scene_facts_by_span(
    graph: CanonicalTemporalMemoryGraph,
    *,
    sample_limit: int = 16,
) -> SceneFactSpanCompactionResult:
    """Compact repeated scene-local state and relation facts into spans.

    It keeps the explicit scene/time axis and only merges facts that have the
    same scene and the same bound entities plus value/predicate signature. It
    does not inspect QA text, gold answers, retrieval output, or QA performance.
    """

    episode_nodes, episode_node_id_map, episode_report = _compact_fact_episodes(
        graph.episode_nodes,
        sample_limit=max(1, sample_limit),
    )
    edges, edge_report = _compact_edges(
        graph.edges,
        episode_node_id_map=episode_node_id_map,
        sample_limit=max(1, sample_limit),
    )
    compact_graph = graph.model_copy(
        update={
            "episode_nodes": episode_nodes,
            "edges": edges,
        }
    )
    report = {
        "contract": "canonical_scene_fact_span_compaction_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "sample_limit": max(1, sample_limit),
            "scope": (
                "Only STATE and RELATION episodes with complete scene-local fact "
                "signatures are compacted. STATE requires same scene, primary "
                "entity, attr_key, and attr_value. RELATION requires same scene, "
                "subject entity, object entity, predicate, and raw predicate. "
                "Scene, entity, stable identity, transition-event, scene-segment, "
                "and entity-segment records are preserved."
            ),
            "graph_semantic_rule": True,
            "does_not_inspect_questions_or_gold_answers": True,
            "frozen_mainline_unchanged": True,
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(compact_graph),
        "episode_compaction": episode_report,
        "edge_compaction": edge_report,
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
        "episode_node_count": (
            report["after"]["episode_node_count"]
            - report["before"]["episode_node_count"]
        ),
    }
    return SceneFactSpanCompactionResult(graph=compact_graph, report=report)


def _compact_fact_episodes(
    nodes: list[CanonicalEpisodeNode],
    *,
    sample_limit: int,
) -> tuple[list[CanonicalEpisodeNode], dict[str, str], dict[str, Any]]:
    passthrough: list[CanonicalEpisodeNode] = []
    grouped: dict[tuple[str, ...], list[CanonicalEpisodeNode]] = defaultdict(list)
    id_map: dict[str, str] = {}
    skipped_by_kind: Counter[str] = Counter()

    for node in nodes:
        signature = _fact_signature(node)
        if signature is None:
            passthrough.append(node)
            id_map[node.node_id] = node.node_id
            skipped_by_kind[node.episode_kind] += 1
            continue
        grouped[signature].append(node)

    compacted: list[CanonicalEpisodeNode] = []
    compact_group_count_by_kind: Counter[str] = Counter()
    source_count_by_kind: Counter[str] = Counter()
    compact_group_size_histogram: Counter[str] = Counter()

    for signature, signature_nodes in grouped.items():
        ordered = sorted(signature_nodes, key=_episode_sort_key)
        if len(ordered) == 1:
            node = ordered[0]
            compacted.append(node)
            id_map[node.node_id] = node.node_id
            continue
        compact_node = _build_scene_fact_span(
            signature=signature,
            nodes=ordered,
            sample_limit=sample_limit,
        )
        compacted.append(compact_node)
        kind = ordered[0].episode_kind
        compact_group_count_by_kind[kind] += 1
        source_count_by_kind[kind] += len(ordered)
        compact_group_size_histogram[_histogram_bucket(len(ordered))] += 1
        for node in ordered:
            id_map[node.node_id] = compact_node.node_id

    merged = sorted(
        [*passthrough, *compacted],
        key=lambda node: (
            _sort_optional_number(node.time_range.start_frame),
            _sort_optional_number(node.time_range.start_ts),
            node.episode_kind,
            node.node_id,
        ),
    )
    return (
        merged,
        id_map,
        {
            "original_episode_nodes": len(nodes),
            "compact_episode_nodes": len(merged),
            "removed_episode_nodes": len(nodes) - len(merged),
            "compact_group_count": sum(compact_group_count_by_kind.values()),
            "source_episode_count_in_compact_groups": sum(source_count_by_kind.values()),
            "compact_group_count_by_kind": dict(sorted(compact_group_count_by_kind.items())),
            "source_episode_count_by_kind": dict(sorted(source_count_by_kind.items())),
            "passthrough_count_by_kind": dict(sorted(skipped_by_kind.items())),
            "compact_group_size_histogram": dict(sorted(compact_group_size_histogram.items())),
        },
    )


def _build_scene_fact_span(
    *,
    signature: tuple[str, ...],
    nodes: list[CanonicalEpisodeNode],
    sample_limit: int,
) -> CanonicalEpisodeNode:
    first = nodes[0]
    compact_id = _compact_episode_id(signature, nodes)
    compact_kind = (
        "scene_state_fact_span"
        if first.episode_kind == "STATE"
        else "scene_relation_fact_span"
    )
    attributes: dict[str, object] = dict(first.attributes)
    attributes.update(
        {
            "compaction_kind": compact_kind,
            "source_episode_kind": first.episode_kind,
            "source_episode_count": len(nodes),
            "source_episode_id_samples": _sample_values(
                [node.episode_id for node in nodes],
                sample_limit,
            ),
            "source_ref_id_samples": _sample_values(
                [node.source_ref_id for node in nodes],
                sample_limit,
            ),
            "source_node_id_samples": _sample_values(
                [node.node_id for node in nodes],
                sample_limit,
            ),
            "observed_interval_samples": _sample_values(
                [
                    _time_range_payload(node.time_range, episode_id=node.episode_id)
                    for node in nodes
                ],
                sample_limit,
            ),
            "observed_interval_count": len(nodes),
            "time_axis_preserved": True,
        }
    )
    return CanonicalEpisodeNode(
        node_id=f"episode:compact:scene_fact_span:{compact_id}",
        video_id=first.video_id,
        episode_id=f"compact:scene_fact_span:{compact_id}",
        episode_kind=first.episode_kind,
        label=_compact_fact_label(first),
        time_range=_span_time_ranges([node.time_range for node in nodes]),
        scene_id=first.scene_id,
        primary_entity_id=first.primary_entity_id,
        subject_entity_id=first.subject_entity_id,
        object_entity_id=first.object_entity_id,
        participant_entity_ids=_merge_unique_strings(
            entity_id
            for node in nodes
            for entity_id in [
                *node.participant_entity_ids,
                node.primary_entity_id,
                node.subject_entity_id,
                node.object_entity_id,
            ]
        ),
        attr_key=first.attr_key,
        attr_value=first.attr_value,
        predicate=first.predicate,
        raw_predicate=first.raw_predicate,
        segment_index=first.segment_index,
        segment_count=first.segment_count,
        source_ref_id=f"compact:scene_fact_span:{compact_id}",
        attributes=attributes,
    )


def _compact_edges(
    edges: list[CanonicalGraphEdge],
    *,
    episode_node_id_map: dict[str, str],
    sample_limit: int,
) -> tuple[list[CanonicalGraphEdge], dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, tuple[tuple[str, str], ...]],
        list[CanonicalGraphEdge],
    ] = defaultdict(list)
    skipped_self_loop_count = 0

    for edge in edges:
        source = episode_node_id_map.get(edge.source_node_id, edge.source_node_id)
        target = episode_node_id_map.get(edge.target_node_id, edge.target_node_id)
        if source == target:
            skipped_self_loop_count += 1
            continue
        attributes = _remap_edge_attributes(edge.attributes, episode_node_id_map)
        normalized_edge = edge.model_copy(update={"attributes": attributes})
        grouped[
            (
                edge.edge_type,
                source,
                target,
                _edge_signature_attrs(normalized_edge),
            )
        ].append(normalized_edge)

    compact_edges: list[CanonicalGraphEdge] = []
    collapsed_group_count = 0
    source_edge_count = 0
    edge_counts_by_type: Counter[str] = Counter()

    for key, source_edges in sorted(grouped.items(), key=lambda item: item[0]):
        edge_type, source, target, signature_attrs = key
        ordered = sorted(source_edges, key=lambda item: item.edge_id)
        attributes = dict(ordered[0].attributes)
        if (
            len(ordered) > 1
            or source != ordered[0].source_node_id
            or target != ordered[0].target_node_id
        ):
            attributes.update(
                {
                    "compaction_kind": "scene_fact_span_edge_group",
                    "source_edge_count": len(ordered),
                    "source_edge_id_samples": _sample_values(
                        [edge.edge_id for edge in ordered],
                        sample_limit,
                    ),
                }
            )
            collapsed_group_count += 1
            source_edge_count += len(ordered)
            edge_counts_by_type[edge_type] += 1
        compact_edges.append(
            CanonicalGraphEdge(
                edge_id=_compact_edge_id(
                    edge_type=edge_type,
                    source_node_id=source,
                    target_node_id=target,
                    signature_attrs=signature_attrs,
                ),
                edge_type=edge_type,
                video_id=ordered[0].video_id,
                source_node_id=source,
                target_node_id=target,
                time_range=_span_time_ranges(
                    [edge.time_range for edge in ordered if edge.time_range is not None]
                ),
                attributes=attributes,
            )
        )

    return (
        compact_edges,
        {
            "original_edges": len(edges),
            "compact_edges": len(compact_edges),
            "removed_edges": len(edges) - len(compact_edges),
            "skipped_self_loop_edges": skipped_self_loop_count,
            "collapsed_group_count": collapsed_group_count,
            "source_edge_count_in_collapsed_groups": source_edge_count,
            "collapsed_group_count_by_type": dict(sorted(edge_counts_by_type.items())),
        },
    )


def _fact_signature(node: CanonicalEpisodeNode) -> tuple[str, ...] | None:
    if node.episode_kind == "STATE":
        if not node.scene_id or not node.primary_entity_id:
            return None
        if not node.attr_key or not node.attr_value:
            return None
        return (
            node.episode_kind,
            node.scene_id,
            node.primary_entity_id,
            node.attr_key,
            node.attr_value,
        )
    if node.episode_kind == "RELATION":
        if not node.scene_id or not node.subject_entity_id or not node.object_entity_id:
            return None
        if not node.predicate and not node.raw_predicate:
            return None
        return (
            node.episode_kind,
            node.scene_id,
            node.subject_entity_id,
            node.object_entity_id,
            node.predicate or "",
            node.raw_predicate or "",
        )
    return None


def _compact_fact_label(node: CanonicalEpisodeNode) -> str:
    if node.episode_kind == "STATE":
        return (
            f"compact state {node.primary_entity_id} "
            f"{node.attr_key}={node.attr_value} in {node.scene_id}"
        )
    return (
        f"compact relation {node.subject_entity_id} "
        f"{node.predicate or node.raw_predicate or 'RELATION'} "
        f"{node.object_entity_id} in {node.scene_id}"
    )


def _edge_signature_attrs(edge: CanonicalGraphEdge) -> tuple[tuple[str, str], ...]:
    keys_by_type = {
        "HAS_EPISODE": ("role", "scope"),
        "IN_SCENE": ("source_segment_id",),
        "HAS_STATE": ("role",),
        "HAS_INTERACTION_TYPE": ("role",),
        "INTERACTS_WITH": ("predicate", "raw_predicate"),
        "HAS_MEMBER_ENTITY": ("role",),
        "BELONGS_TO_SEGMENT": ("role",),
        "TRIGGERED_BY": ("role", "event_type"),
        "STARTS_AT": ("temporal_relation", "event_type"),
        "ENDS_AT": ("temporal_relation", "event_type"),
        "OVERLAPS_WITH": ("temporal_relation",),
        "TEMPORALLY_NEAR": ("temporal_relation",),
        "NEXT_EPISODE": ("temporal_relation",),
        "PREVIOUS_EPISODE": ("temporal_relation",),
    }
    pairs: list[tuple[str, str]] = []
    for key in keys_by_type.get(edge.edge_type, ()):
        value = edge.attributes.get(key)
        if value is None:
            continue
        pairs.append((key, str(value)))
    return tuple(pairs)


def _remap_edge_attributes(
    attributes: dict[str, object],
    episode_node_id_map: dict[str, str],
) -> dict[str, object]:
    remapped = dict(attributes)
    for key in (
        "episode_node_id",
        "source_episode_node_id",
        "target_episode_node_id",
    ):
        value = remapped.get(key)
        if isinstance(value, str) and value in episode_node_id_map:
            remapped[key] = episode_node_id_map[value]
    for key in ("episode_node_ids", "source_episode_node_ids", "target_episode_node_ids"):
        value = remapped.get(key)
        if isinstance(value, list):
            remapped[key] = [
                episode_node_id_map.get(str(item), str(item))
                for item in value
            ]
    return remapped


def _episode_sort_key(node: CanonicalEpisodeNode) -> tuple[float, float, str]:
    return (
        _sort_optional_number(node.time_range.start_frame),
        _sort_optional_number(node.time_range.start_ts),
        node.node_id,
    )


def _compact_episode_id(
    signature: tuple[str, ...],
    nodes: list[CanonicalEpisodeNode],
) -> str:
    first = nodes[0]
    last = nodes[-1]
    text = "||".join(
        [
            *signature,
            str(first.time_range.start_frame),
            str(last.time_range.end_frame),
            str(first.time_range.start_ts),
            str(last.time_range.end_ts),
            str(len(nodes)),
        ]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _compact_edge_id(
    *,
    edge_type: str,
    source_node_id: str,
    target_node_id: str,
    signature_attrs: tuple[tuple[str, str], ...],
) -> str:
    text = "||".join(
        [
            edge_type,
            source_node_id,
            target_node_id,
            *[f"{key}={value}" for key, value in signature_attrs],
        ]
    )
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"edge:compact:scene_fact_span:{edge_type.lower()}:{digest}"


def _span_time_ranges(time_ranges: list[CanonicalTimeRange]) -> CanonicalTimeRange:
    if not time_ranges:
        return CanonicalTimeRange()
    return CanonicalTimeRange(
        start_frame=_min_optional(item.start_frame for item in time_ranges),
        end_frame=_max_optional(item.end_frame for item in time_ranges),
        start_ts=_min_optional(item.start_ts for item in time_ranges),
        end_ts=_max_optional(item.end_ts for item in time_ranges),
        support_frames=sum(
            int(item.support_frames)
            for item in time_ranges
            if item.support_frames is not None
        )
        or None,
    )


def _time_range_payload(
    time_range: CanonicalTimeRange,
    *,
    episode_id: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "start_frame": time_range.start_frame,
        "end_frame": time_range.end_frame,
        "start_ts": time_range.start_ts,
        "end_ts": time_range.end_ts,
        "support_frames": time_range.support_frames,
    }


def _graph_size_report(graph: CanonicalTemporalMemoryGraph) -> dict[str, int]:
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
    }


def _min_optional(values: Any) -> Any:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _max_optional(values: Any) -> Any:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _merge_unique_strings(values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _sample_values(values: list[Any], sample_limit: int) -> list[Any]:
    if len(values) <= sample_limit:
        return list(values)
    if sample_limit <= 2:
        return values[:sample_limit]
    head_count = sample_limit // 2
    tail_count = sample_limit - head_count
    return [*values[:head_count], *values[-tail_count:]]


def _sort_optional_number(value: int | float | None) -> float:
    return float(value) if value is not None else float("inf")


def _histogram_bucket(value: int) -> str:
    if value <= 5:
        return str(value)
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    return "51+"
