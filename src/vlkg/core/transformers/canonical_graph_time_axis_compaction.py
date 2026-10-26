from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from vlkg.core.models.canonical_graph import (
    CanonicalEdgeType,
    CanonicalEpisodeKind,
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)


CompactEpisodeKind = Literal["ENTITY_SEGMENT", "STATE", "RELATION"]
SignatureEntityScope = Literal["entity", "stable_identity"]


DEFAULT_COMPACT_EPISODE_KINDS: tuple[CompactEpisodeKind, ...] = (
    "ENTITY_SEGMENT",
    "STATE",
    "RELATION",
)


@dataclass(frozen=True)
class TimeAxisCompactionResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def compact_canonical_graph_by_time_axis(
    graph: CanonicalTemporalMemoryGraph,
    *,
    compact_episode_kinds: Iterable[CompactEpisodeKind] = DEFAULT_COMPACT_EPISODE_KINDS,
    max_gap_frames: int = 10,
    max_gap_seconds: float = 1.0,
    sample_limit: int = 12,
    signature_entity_scope: SignatureEntityScope = "entity",
) -> TimeAxisCompactionResult:
    """Compact redundant episode bursts while keeping the time axis explicit.

    This compacts only repeated episode-level facts that share the same scene,
    entity/relation/state signature, and are adjacent or overlapping in time.
    The compact node retains the span and sampled source intervals, so the
    result is a derived canonical graph view rather than a QA-specific rewrite.
    """

    compact_kinds = set(compact_episode_kinds)
    entity_signature_resolver = _build_entity_signature_resolver(
        graph,
        signature_entity_scope=signature_entity_scope,
    )
    episode_nodes, episode_node_id_map, episode_report = _compact_episode_nodes(
        graph.episode_nodes,
        compact_episode_kinds=compact_kinds,
        max_gap_frames=max_gap_frames,
        max_gap_seconds=max_gap_seconds,
        sample_limit=max(1, sample_limit),
        entity_signature_resolver=entity_signature_resolver,
        signature_entity_scope=signature_entity_scope,
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
        "contract": "canonical_time_axis_compaction_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "compact_episode_kinds": sorted(compact_kinds),
            "max_gap_frames": max_gap_frames,
            "max_gap_seconds": max_gap_seconds,
            "sample_limit": max(1, sample_limit),
            "signature_entity_scope": signature_entity_scope,
            "time_axis_policy": (
                "Only adjacent/overlapping same-signature episode bursts are compacted; "
                "scene, entity/relation/state signature, span, and sampled source "
                "intervals are retained on the compact episode node."
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
    return TimeAxisCompactionResult(graph=compact_graph, report=report)


def _compact_episode_nodes(
    nodes: list[CanonicalEpisodeNode],
    *,
    compact_episode_kinds: set[str],
    max_gap_frames: int,
    max_gap_seconds: float,
    sample_limit: int,
    entity_signature_resolver: dict[str, str],
    signature_entity_scope: SignatureEntityScope,
) -> tuple[list[CanonicalEpisodeNode], dict[str, str], dict[str, Any]]:
    passthrough: list[CanonicalEpisodeNode] = []
    grouped: dict[tuple[str, ...], list[CanonicalEpisodeNode]] = defaultdict(list)
    id_map: dict[str, str] = {}

    for node in nodes:
        if node.episode_kind not in compact_episode_kinds:
            passthrough.append(node)
            id_map[node.node_id] = node.node_id
            continue
        grouped[
            _episode_signature(
                node,
                entity_signature_resolver=entity_signature_resolver,
            )
        ].append(node)

    compacted: list[CanonicalEpisodeNode] = []
    compact_group_count = 0
    source_episode_count = 0
    compacted_by_kind: dict[str, int] = defaultdict(int)
    source_by_kind: dict[str, int] = defaultdict(int)

    for signature, signature_nodes in grouped.items():
        ordered = sorted(signature_nodes, key=_episode_sort_key)
        for burst in _split_temporal_bursts(
            ordered,
            max_gap_frames=max_gap_frames,
            max_gap_seconds=max_gap_seconds,
        ):
            if len(burst) == 1:
                node = burst[0]
                compacted.append(node)
                id_map[node.node_id] = node.node_id
                continue
            compact_node = _build_compact_episode_node(
                signature=signature,
                nodes=burst,
                max_gap_frames=max_gap_frames,
                max_gap_seconds=max_gap_seconds,
                sample_limit=sample_limit,
                entity_signature_resolver=entity_signature_resolver,
                signature_entity_scope=signature_entity_scope,
            )
            compacted.append(compact_node)
            compact_group_count += 1
            source_episode_count += len(burst)
            compacted_by_kind[compact_node.episode_kind] += 1
            source_by_kind[compact_node.episode_kind] += len(burst)
            for node in burst:
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
        "compact_group_count_by_kind": dict(sorted(compacted_by_kind.items())),
        "source_episode_count_by_kind": dict(sorted(source_by_kind.items())),
    }
    return merged, id_map, report


def _build_compact_episode_node(
    *,
    signature: tuple[str, ...],
    nodes: list[CanonicalEpisodeNode],
    max_gap_frames: int,
    max_gap_seconds: float,
    sample_limit: int,
    entity_signature_resolver: dict[str, str],
    signature_entity_scope: SignatureEntityScope,
) -> CanonicalEpisodeNode:
    first = nodes[0]
    time_range = _span_time_ranges([node.time_range for node in nodes])
    compact_id = _compact_episode_id(signature, nodes)
    source_episode_ids = [node.episode_id for node in nodes]
    source_ref_ids = [node.source_ref_id for node in nodes]
    source_node_ids = [node.node_id for node in nodes]
    attributes: dict[str, object] = dict(first.attributes)
    attributes.update(
        {
            "compaction_kind": "time_axis_episode_burst",
            "signature_entity_scope": signature_entity_scope,
            "source_episode_kind": first.episode_kind,
            "source_episode_count": len(nodes),
            "source_episode_id_samples": _sample_values(source_episode_ids, sample_limit),
            "source_ref_id_samples": _sample_values(source_ref_ids, sample_limit),
            "source_node_id_samples": _sample_values(source_node_ids, sample_limit),
            "observed_interval_samples": _sample_values(
                [_time_range_payload(node.time_range, episode_id=node.episode_id) for node in nodes],
                sample_limit,
            ),
            "observed_interval_count": len(nodes),
            "max_gap_frames": max_gap_frames,
            "max_gap_seconds": max_gap_seconds,
            "time_axis_preserved": True,
        }
    )
    if signature_entity_scope == "stable_identity":
        attributes.update(
            {
                "signature_primary_entity_id": _resolve_entity_signature(
                    first.primary_entity_id,
                    entity_signature_resolver,
                ),
                "signature_subject_entity_id": _resolve_entity_signature(
                    first.subject_entity_id,
                    entity_signature_resolver,
                ),
                "signature_object_entity_id": _resolve_entity_signature(
                    first.object_entity_id,
                    entity_signature_resolver,
                ),
                "source_primary_entity_id_samples": _sample_values(
                    [node.primary_entity_id for node in nodes if node.primary_entity_id],
                    sample_limit,
                ),
                "source_subject_entity_id_samples": _sample_values(
                    [node.subject_entity_id for node in nodes if node.subject_entity_id],
                    sample_limit,
                ),
                "source_object_entity_id_samples": _sample_values(
                    [node.object_entity_id for node in nodes if node.object_entity_id],
                    sample_limit,
                ),
            }
        )
    return CanonicalEpisodeNode(
        node_id=f"episode:compact:{first.episode_kind.lower()}:{compact_id}",
        video_id=first.video_id,
        episode_id=f"compact:{first.episode_kind.lower()}:{compact_id}",
        episode_kind=first.episode_kind,
        label=_compact_episode_label(first),
        time_range=time_range,
        scene_id=first.scene_id,
        primary_entity_id=first.primary_entity_id,
        subject_entity_id=first.subject_entity_id,
        object_entity_id=first.object_entity_id,
        participant_entity_ids=_merge_unique_strings(
            entity_id
            for node in nodes
            for entity_id in node.participant_entity_ids
        ),
        attr_key=first.attr_key,
        attr_value=first.attr_value,
        predicate=first.predicate,
        raw_predicate=_join_unique_strings(node.raw_predicate for node in nodes),
        segment_index=first.segment_index,
        segment_count=first.segment_count,
        source_ref_id=f"compact:{first.episode_kind.lower()}:{compact_id}",
        attributes=attributes,
    )


def _compact_edges(
    edges: list[CanonicalGraphEdge],
    *,
    episode_node_id_map: dict[str, str],
    sample_limit: int,
) -> tuple[list[CanonicalGraphEdge], dict[str, Any]]:
    grouped: dict[
        tuple[CanonicalEdgeType, str, str, tuple[tuple[str, str], ...]],
        list[CanonicalGraphEdge],
    ] = defaultdict(list)

    for edge in edges:
        source = episode_node_id_map.get(edge.source_node_id, edge.source_node_id)
        target = episode_node_id_map.get(edge.target_node_id, edge.target_node_id)
        if source == target:
            continue
        attributes = _remap_edge_attributes(edge.attributes, episode_node_id_map)
        normalized_edge = edge.model_copy(update={"attributes": attributes})
        key = (
            normalized_edge.edge_type,
            source,
            target,
            _edge_signature_attrs(normalized_edge),
        )
        grouped[key].append(normalized_edge)

    compact_edges: list[CanonicalGraphEdge] = []
    collapsed_group_count = 0
    source_edge_count = 0
    edge_counts_by_type: dict[str, int] = defaultdict(int)

    for key, source_edges in sorted(grouped.items(), key=lambda item: item[0]):
        edge_type, source, target, signature_attrs = key
        ordered = sorted(source_edges, key=lambda item: item.edge_id)
        time_range = _span_time_ranges(
            [edge.time_range for edge in ordered if edge.time_range is not None]
        )
        attributes = dict(ordered[0].attributes)
        if len(ordered) > 1 or (
            source != ordered[0].source_node_id or target != ordered[0].target_node_id
        ):
            attributes.update(
                {
                    "compaction_kind": "time_axis_edge_burst",
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
                time_range=time_range,
                attributes=attributes,
            )
        )

    report = {
        "original_edges": len(edges),
        "compact_edges": len(compact_edges),
        "removed_edges": len(edges) - len(compact_edges),
        "collapsed_group_count": collapsed_group_count,
        "source_edge_count_in_collapsed_groups": source_edge_count,
        "collapsed_group_count_by_type": dict(sorted(edge_counts_by_type.items())),
    }
    return compact_edges, report


def _split_temporal_bursts(
    nodes: list[CanonicalEpisodeNode],
    *,
    max_gap_frames: int,
    max_gap_seconds: float,
) -> list[list[CanonicalEpisodeNode]]:
    if not nodes:
        return []
    bursts: list[list[CanonicalEpisodeNode]] = []
    current = [nodes[0]]
    for node in nodes[1:]:
        previous = current[-1]
        if _can_merge_temporally(
            previous.time_range,
            node.time_range,
            max_gap_frames=max_gap_frames,
            max_gap_seconds=max_gap_seconds,
        ):
            current.append(node)
            continue
        bursts.append(current)
        current = [node]
    bursts.append(current)
    return bursts


def _can_merge_temporally(
    left: CanonicalTimeRange,
    right: CanonicalTimeRange,
    *,
    max_gap_frames: int,
    max_gap_seconds: float,
) -> bool:
    frame_gap = _gap(left.end_frame, right.start_frame)
    second_gap = _gap(left.end_ts, right.start_ts)
    if frame_gap is None and second_gap is None:
        return False
    frame_ok = frame_gap is not None and frame_gap <= max_gap_frames
    second_ok = second_gap is not None and second_gap <= max_gap_seconds
    return frame_ok or second_ok


def _episode_signature(
    node: CanonicalEpisodeNode,
    *,
    entity_signature_resolver: dict[str, str],
) -> tuple[str, ...]:
    if node.episode_kind == "ENTITY_SEGMENT":
        return (
            node.episode_kind,
            node.scene_id or "",
            _resolve_entity_signature(node.primary_entity_id, entity_signature_resolver),
            str(node.attributes.get("segment_type") or ""),
        )
    if node.episode_kind == "STATE":
        return (
            node.episode_kind,
            node.scene_id or "",
            _resolve_entity_signature(node.primary_entity_id, entity_signature_resolver),
            node.attr_key or "",
            node.attr_value or "",
        )
    if node.episode_kind == "RELATION":
        return (
            node.episode_kind,
            node.scene_id or "",
            _resolve_entity_signature(node.subject_entity_id, entity_signature_resolver),
            node.predicate or "",
            _resolve_entity_signature(node.object_entity_id, entity_signature_resolver),
        )
    return (node.episode_kind, node.node_id)


def _build_entity_signature_resolver(
    graph: CanonicalTemporalMemoryGraph,
    *,
    signature_entity_scope: SignatureEntityScope,
) -> dict[str, str]:
    if signature_entity_scope == "entity":
        return {}
    return {
        node.entity_id: node.stable_identity_id or node.entity_id
        for node in graph.entity_nodes
    }


def _resolve_entity_signature(
    entity_id: str | None,
    resolver: dict[str, str],
) -> str:
    if not entity_id:
        return ""
    return resolver.get(entity_id, entity_id)


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


def _compact_episode_label(node: CanonicalEpisodeNode) -> str:
    if node.episode_kind == "ENTITY_SEGMENT":
        return f"compact entity segment {node.primary_entity_id}"
    if node.episode_kind == "STATE":
        return f"compact state {node.primary_entity_id} {node.attr_key}={node.attr_value}"
    return f"compact relation {node.subject_entity_id} {node.predicate} {node.object_entity_id}"


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
    return f"edge:compact:{edge_type.lower()}:{digest}"


def _span_time_ranges(time_ranges: list[CanonicalTimeRange]) -> CanonicalTimeRange:
    if not time_ranges:
        return CanonicalTimeRange()
    start_frame = _min_optional(item.start_frame for item in time_ranges)
    end_frame = _max_optional(item.end_frame for item in time_ranges)
    start_ts = _min_optional(item.start_ts for item in time_ranges)
    end_ts = _max_optional(item.end_ts for item in time_ranges)
    support_frames = sum(
        int(item.support_frames)
        for item in time_ranges
        if item.support_frames is not None
    )
    return CanonicalTimeRange(
        start_frame=start_frame,
        end_frame=end_frame,
        start_ts=start_ts,
        end_ts=end_ts,
        support_frames=support_frames or None,
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


def _sample_values(values: list[Any], sample_limit: int) -> list[Any]:
    if len(values) <= sample_limit:
        return list(values)
    if sample_limit <= 2:
        return values[:sample_limit]
    head_count = sample_limit // 2
    tail_count = sample_limit - head_count
    return [*values[:head_count], *values[-tail_count:]]


def _merge_unique_strings(values: Iterable[str | None]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return merged


def _join_unique_strings(values: Iterable[str | None]) -> str | None:
    merged = _merge_unique_strings(values)
    return " | ".join(merged) if merged else None


def _gap(left_end: int | float | None, right_start: int | float | None) -> float | None:
    if left_end is None or right_start is None:
        return None
    return max(0.0, float(right_start) - float(left_end))


def _min_optional(values: Iterable[int | float | None]) -> Any:
    known = [value for value in values if value is not None]
    return min(known) if known else None


def _max_optional(values: Iterable[int | float | None]) -> Any:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _sort_optional_number(value: int | float | None) -> float:
    return float(value) if value is not None else math.inf
