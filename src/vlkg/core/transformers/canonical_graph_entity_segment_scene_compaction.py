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
class EntitySegmentSceneCompactionResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def compact_entity_segments_by_scene_span(
    graph: CanonicalTemporalMemoryGraph,
    *,
    sample_limit: int = 16,
) -> EntitySegmentSceneCompactionResult:
    """Compact repeated entity-segment episodes into scene-level presence spans.

    It keeps the explicit time axis, but represents "entity X appears in scene Y"
    once per scene/entity/segment signature instead of repeating the same
    presence evidence for many sampled moments.
    """

    episode_nodes, episode_node_id_map, episode_report = _compact_entity_segments(
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
        "contract": "canonical_entity_segment_scene_span_compaction_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "sample_limit": max(1, sample_limit),
            "scope": (
                "Only ENTITY_SEGMENT episodes with the same scene, entity, and "
                "segment-type signature are compacted into one scene-level "
                "presence span. State, relation, transition-event, and scene "
                "segment records are preserved."
            ),
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
    return EntitySegmentSceneCompactionResult(graph=compact_graph, report=report)


def _compact_entity_segments(
    nodes: list[CanonicalEpisodeNode],
    *,
    sample_limit: int,
) -> tuple[list[CanonicalEpisodeNode], dict[str, str], dict[str, Any]]:
    passthrough: list[CanonicalEpisodeNode] = []
    grouped: dict[tuple[str, ...], list[CanonicalEpisodeNode]] = defaultdict(list)
    id_map: dict[str, str] = {}

    for node in nodes:
        if node.episode_kind != "ENTITY_SEGMENT":
            passthrough.append(node)
            id_map[node.node_id] = node.node_id
            continue
        grouped[_entity_segment_signature(node)].append(node)

    compacted: list[CanonicalEpisodeNode] = []
    compact_group_count = 0
    source_episode_count = 0

    for signature, signature_nodes in grouped.items():
        ordered = sorted(signature_nodes, key=_episode_sort_key)
        if len(ordered) == 1:
            node = ordered[0]
            compacted.append(node)
            id_map[node.node_id] = node.node_id
            continue
        compact_node = _build_scene_span_entity_segment(
            signature=signature,
            nodes=ordered,
            sample_limit=sample_limit,
        )
        compacted.append(compact_node)
        compact_group_count += 1
        source_episode_count += len(ordered)
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
    report = {
        "original_episode_nodes": len(nodes),
        "compact_episode_nodes": len(merged),
        "removed_episode_nodes": len(nodes) - len(merged),
        "compact_group_count": compact_group_count,
        "source_episode_count_in_compact_groups": source_episode_count,
        "episode_kind": "ENTITY_SEGMENT",
    }
    return merged, id_map, report


def _build_scene_span_entity_segment(
    *,
    signature: tuple[str, ...],
    nodes: list[CanonicalEpisodeNode],
    sample_limit: int,
) -> CanonicalEpisodeNode:
    first = nodes[0]
    compact_id = _compact_episode_id(signature, nodes)
    attributes: dict[str, object] = dict(first.attributes)
    attributes.update(
        {
            "compaction_kind": "scene_entity_segment_span",
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
                [_time_range_payload(node.time_range, episode_id=node.episode_id) for node in nodes],
                sample_limit,
            ),
            "observed_interval_count": len(nodes),
            "time_axis_preserved": True,
        }
    )
    return CanonicalEpisodeNode(
        node_id=f"episode:compact:entity_segment_scene_span:{compact_id}",
        video_id=first.video_id,
        episode_id=f"compact:entity_segment_scene_span:{compact_id}",
        episode_kind=first.episode_kind,
        label=f"compact entity segment {first.primary_entity_id} in {first.scene_id}",
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
        source_ref_id=f"compact:entity_segment_scene_span:{compact_id}",
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
        if len(ordered) > 1 or source != ordered[0].source_node_id or target != ordered[0].target_node_id:
            attributes.update(
                {
                    "compaction_kind": "scene_entity_segment_edge_group",
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

    report = {
        "original_edges": len(edges),
        "compact_edges": len(compact_edges),
        "removed_edges": len(edges) - len(compact_edges),
        "skipped_self_loop_edges": skipped_self_loop_count,
        "collapsed_group_count": collapsed_group_count,
        "source_edge_count_in_collapsed_groups": source_edge_count,
        "collapsed_group_count_by_type": dict(sorted(edge_counts_by_type.items())),
    }
    return compact_edges, report


def _entity_segment_signature(node: CanonicalEpisodeNode) -> tuple[str, ...]:
    return (
        node.episode_kind,
        node.scene_id or "",
        node.primary_entity_id or "",
        str(node.attributes.get("segment_type") or ""),
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
    for key in ("episode_node_ids", "source_episode_node_ids"):
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
    return f"edge:compact:scene_entity_segment:{edge_type.lower()}:{digest}"


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
