from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalGraphEdge,
    CanonicalRelationObservedSample,
    CanonicalSceneNode,
    CanonicalStableIdentityNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)


@dataclass(frozen=True)
class StableCoreCompactionResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def compact_canonical_graph_to_stable_core(
    graph: CanonicalTemporalMemoryGraph,
    *,
    compact_scene_label_entities: bool = True,
    min_scene_label_group_size: int = 2,
    max_raw_visual_attributes: int = 24,
    sample_limit: int = 16,
) -> StableCoreCompactionResult:
    """Normalize duplicate entity fragments into a stable referent core.

    The policy is intentionally generic:
    - members of the same stable identity are represented by one entity node;
    - otherwise, repeated entities with the same scene footprint, label, and role
      are represented by one entity node;
    - episode/event entity references and entity-to-entity edges are remapped to
      the representative ids.

    It does not mutate the input graph and does not inspect questions, answers,
    retrieval output, or QA performance.
    """

    scene_ids_by_entity = _scene_ids_by_entity(graph)
    groups = _group_entities(
        graph.entity_nodes,
        scene_ids_by_entity=scene_ids_by_entity,
        compact_scene_label_entities=compact_scene_label_entities,
        min_scene_label_group_size=max(2, min_scene_label_group_size),
    )
    entity_id_map, entity_node_id_map, entity_nodes, entity_report = _compact_entities(
        groups,
        max_raw_visual_attributes=max(0, max_raw_visual_attributes),
        sample_limit=max(1, sample_limit),
    )

    scene_nodes = [
        _remap_scene_node(node, entity_id_map=entity_id_map)
        for node in graph.scene_nodes
    ]
    stable_identity_nodes = [
        _remap_stable_identity_node(
            node,
            entity_id_map=entity_id_map,
            sample_limit=max(1, sample_limit),
        )
        for node in graph.stable_identity_nodes
    ]
    episode_nodes = [
        node.model_copy(
            update={
                "primary_entity_id": _remap_entity_id(node.primary_entity_id, entity_id_map),
                "subject_entity_id": _remap_entity_id(node.subject_entity_id, entity_id_map),
                "object_entity_id": _remap_entity_id(node.object_entity_id, entity_id_map),
                "participant_entity_ids": _remap_entity_id_list(
                    [
                        *node.participant_entity_ids,
                        node.primary_entity_id,
                        node.subject_entity_id,
                        node.object_entity_id,
                    ],
                    entity_id_map,
                ),
            }
        )
        for node in graph.episode_nodes
    ]
    transition_event_nodes = [
        node.model_copy(
            update={
                "entity_id": _remap_entity_id(node.entity_id, entity_id_map),
                "subject_entity_id": _remap_entity_id(node.subject_entity_id, entity_id_map),
                "object_entity_id": _remap_entity_id(node.object_entity_id, entity_id_map),
            }
        )
        for node in graph.transition_event_nodes
    ]
    relation_observed_samples = [
        _remap_relation_observed_sample(node, entity_id_map=entity_id_map)
        for node in graph.relation_observed_samples
    ]
    edges, edge_report = _compact_edges_after_entity_remap(
        graph.edges,
        entity_node_id_map=entity_node_id_map,
        entity_id_map=entity_id_map,
        sample_limit=max(1, sample_limit),
    )

    compact_graph = graph.model_copy(
        update={
            "scene_nodes": scene_nodes,
            "entity_nodes": entity_nodes,
            "stable_identity_nodes": stable_identity_nodes,
            "episode_nodes": episode_nodes,
            "transition_event_nodes": transition_event_nodes,
            "relation_observed_samples": relation_observed_samples,
            "edges": edges,
        }
    )
    report = {
        "contract": "canonical_stable_core_compaction_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "compact_scene_label_entities": compact_scene_label_entities,
            "min_scene_label_group_size": max(2, min_scene_label_group_size),
            "max_raw_visual_attributes": max(0, max_raw_visual_attributes),
            "sample_limit": max(1, sample_limit),
            "scope": (
                "Entity fragments are collapsed by stable identity when "
                "available, otherwise by same scene footprint, normalized "
                "label, and node role. The rule is graph-semantic and does not "
                "use QA supervision."
            ),
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(compact_graph),
        "entity_compaction": entity_report,
        "edge_compaction": edge_report,
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
        "entity_node_count": (
            report["after"]["entity_node_count"]
            - report["before"]["entity_node_count"]
        ),
    }
    return StableCoreCompactionResult(graph=compact_graph, report=report)


def _scene_ids_by_entity(graph: CanonicalTemporalMemoryGraph) -> dict[str, set[str]]:
    by_entity: dict[str, set[str]] = defaultdict(set)
    scene_ids_by_node_id = {
        node.node_id: node.scene_id
        for node in graph.scene_nodes
    }
    entity_ids_by_node_id = {
        node.node_id: node.entity_id
        for node in graph.entity_nodes
    }
    for edge in graph.edges:
        if edge.edge_type != "IN_SCENE":
            continue
        entity_id = entity_ids_by_node_id.get(edge.source_node_id)
        scene_id = scene_ids_by_node_id.get(edge.target_node_id)
        if entity_id and scene_id:
            by_entity[entity_id].add(scene_id)

    for episode in graph.episode_nodes:
        if not episode.scene_id:
            continue
        for entity_id in episode.participant_entity_ids:
            if entity_id:
                by_entity[entity_id].add(episode.scene_id)
        for entity_id in (
            episode.primary_entity_id,
            episode.subject_entity_id,
            episode.object_entity_id,
        ):
            if entity_id:
                by_entity[entity_id].add(episode.scene_id)
    return by_entity


def _group_entities(
    nodes: list[CanonicalEntityNode],
    *,
    scene_ids_by_entity: dict[str, set[str]],
    compact_scene_label_entities: bool,
    min_scene_label_group_size: int,
) -> list[tuple[tuple[str, ...], list[CanonicalEntityNode]]]:
    initial: dict[tuple[str, ...], list[CanonicalEntityNode]] = defaultdict(list)
    singles: list[tuple[tuple[str, ...], list[CanonicalEntityNode]]] = []

    for node in nodes:
        key = _entity_group_key(node, scene_ids_by_entity=scene_ids_by_entity)
        initial[key].append(node)

    groups: list[tuple[tuple[str, ...], list[CanonicalEntityNode]]] = []
    for key, group_nodes in initial.items():
        if key[0] == "stable_identity" or (
            compact_scene_label_entities
            and len(group_nodes) >= min_scene_label_group_size
        ):
            groups.append((key, group_nodes))
            continue
        for node in group_nodes:
            singles.append((("entity", node.entity_id), [node]))
    groups.extend(singles)
    return groups


def _entity_group_key(
    node: CanonicalEntityNode,
    *,
    scene_ids_by_entity: dict[str, set[str]],
) -> tuple[str, ...]:
    if node.stable_identity_id:
        return ("stable_identity", node.stable_identity_id)
    scene_key = "|".join(sorted(scene_ids_by_entity.get(node.entity_id, set()))) or "unknown_scene"
    label_key = _normalized_label(node)
    return (
        "scene_label_role",
        scene_key,
        label_key,
        str(node.node_role or ""),
        "static" if node.is_static else "dynamic",
        "noise" if node.is_noise else "not_noise",
    )


def _compact_entities(
    groups: list[tuple[tuple[str, ...], list[CanonicalEntityNode]]],
    *,
    max_raw_visual_attributes: int,
    sample_limit: int,
) -> tuple[dict[str, str], dict[str, str], list[CanonicalEntityNode], dict[str, Any]]:
    entity_id_map: dict[str, str] = {}
    entity_node_id_map: dict[str, str] = {}
    compact_nodes: list[CanonicalEntityNode] = []
    group_count_by_scope: Counter[str] = Counter()
    source_count_by_scope: Counter[str] = Counter()
    compacted_source_entities = 0

    for key, nodes in groups:
        representative = _representative_entity(nodes)
        for node in nodes:
            entity_id_map[node.entity_id] = representative.entity_id
            entity_node_id_map[node.node_id] = representative.node_id

        if len(nodes) == 1:
            compact_nodes.append(representative)
            continue

        scope = key[0]
        group_count_by_scope[scope] += 1
        source_count_by_scope[scope] += len(nodes)
        compacted_source_entities += len(nodes)
        compact_nodes.append(
            _merge_entity_group(
                key,
                nodes,
                representative=representative,
                max_raw_visual_attributes=max_raw_visual_attributes,
                sample_limit=sample_limit,
            )
        )

    compact_nodes.sort(
        key=lambda node: (
            _sort_optional_number(node.time_range.start_frame),
            _sort_optional_number(node.time_range.start_ts),
            node.node_id,
        )
    )
    report = {
        "original_entity_nodes": sum(len(nodes) for _, nodes in groups),
        "compact_entity_nodes": len(compact_nodes),
        "removed_entity_nodes": sum(len(nodes) for _, nodes in groups) - len(compact_nodes),
        "compact_group_count": sum(group_count_by_scope.values()),
        "source_entity_count_in_compact_groups": compacted_source_entities,
        "compact_group_count_by_scope": dict(sorted(group_count_by_scope.items())),
        "source_entity_count_by_scope": dict(sorted(source_count_by_scope.items())),
    }
    return entity_id_map, entity_node_id_map, compact_nodes, report


def _representative_entity(nodes: list[CanonicalEntityNode]) -> CanonicalEntityNode:
    return sorted(
        nodes,
        key=lambda node: (
            bool(node.is_noise),
            -int(node.observation_count or 0),
            _sort_optional_number(node.time_range.start_frame),
            node.entity_id,
        ),
    )[0]


def _merge_entity_group(
    key: tuple[str, ...],
    nodes: list[CanonicalEntityNode],
    *,
    representative: CanonicalEntityNode,
    max_raw_visual_attributes: int,
    sample_limit: int,
) -> CanonicalEntityNode:
    labels_seen = _merge_unique_strings(
        item for node in nodes for item in [node.label, *node.labels_seen]
    )
    states_seen = _merge_unique_strings(item for node in nodes for item in node.states_seen)
    parent_labels_seen = _merge_unique_strings(
        item for node in nodes for item in node.parent_labels_seen
    )
    raw_visual_attributes = _merge_unique_strings(
        str(item)
        for node in nodes
        for item in node.attributes.get("raw_visual_attributes", [])
        if str(item).strip()
    )
    if max_raw_visual_attributes > 0:
        raw_visual_attributes = raw_visual_attributes[:max_raw_visual_attributes]
    else:
        raw_visual_attributes = []

    attributes = dict(representative.attributes)
    attributes.update(
        {
            "compaction_kind": "stable_core_entity_group",
            "stable_core_group_scope": key[0],
            "stable_core_group_key": list(key),
            "source_entity_count": len(nodes),
            "source_entity_id_samples": _sample_values(
                [node.entity_id for node in nodes],
                sample_limit,
            ),
            "source_entity_node_id_samples": _sample_values(
                [node.node_id for node in nodes],
                sample_limit,
            ),
            "raw_visual_attributes": raw_visual_attributes,
        }
    )
    return representative.model_copy(
        update={
            "time_range": _span_time_ranges([node.time_range for node in nodes]),
            "observation_count": sum(int(node.observation_count or 0) for node in nodes),
            "is_static": all(bool(node.is_static) for node in nodes),
            "is_noise": all(bool(node.is_noise) for node in nodes),
            "is_fragment": any(bool(node.is_fragment) for node in nodes),
            "labels_seen": labels_seen,
            "states_seen": states_seen,
            "parent_labels_seen": parent_labels_seen,
            "attributes": attributes,
        }
    )


def _remap_scene_node(
    node: CanonicalSceneNode,
    *,
    entity_id_map: dict[str, str],
) -> CanonicalSceneNode:
    return node.model_copy(
        update={
            "representative_static_entity_ids": _remap_entity_id_list(
                node.representative_static_entity_ids,
                entity_id_map,
            )
        }
    )


def _remap_stable_identity_node(
    node: CanonicalStableIdentityNode,
    *,
    entity_id_map: dict[str, str],
    sample_limit: int,
) -> CanonicalStableIdentityNode:
    member_entity_ids = _remap_entity_id_list(node.member_entity_ids, entity_id_map)
    if member_entity_ids == node.member_entity_ids:
        return node
    attributes = dict(node.attributes)
    attributes.update(
        {
            "stable_core_member_entity_ids_remapped": True,
            "source_member_entity_id_samples": _sample_values(
                node.member_entity_ids,
                sample_limit,
            ),
            "source_member_entity_count": len(node.member_entity_ids),
        }
    )
    return node.model_copy(
        update={
            "member_entity_ids": member_entity_ids,
            "attributes": attributes,
        }
    )


def _remap_relation_observed_sample(
    node: CanonicalRelationObservedSample,
    *,
    entity_id_map: dict[str, str],
) -> CanonicalRelationObservedSample:
    return node.model_copy(
        update={
            "subject_entity_id": _remap_entity_id(node.subject_entity_id, entity_id_map),
            "object_entity_id": _remap_entity_id(node.object_entity_id, entity_id_map),
        }
    )


def _compact_edges_after_entity_remap(
    edges: list[CanonicalGraphEdge],
    *,
    entity_node_id_map: dict[str, str],
    entity_id_map: dict[str, str],
    sample_limit: int,
) -> tuple[list[CanonicalGraphEdge], dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, tuple[tuple[str, str], ...]],
        list[CanonicalGraphEdge],
    ] = defaultdict(list)

    for edge in edges:
        source = entity_node_id_map.get(edge.source_node_id, edge.source_node_id)
        target = entity_node_id_map.get(edge.target_node_id, edge.target_node_id)
        if source == target:
            continue
        attributes = _remap_edge_attributes(edge.attributes, entity_id_map=entity_id_map)
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
    collapsed_by_type: Counter[str] = Counter()
    removed_self_loop_count = len(edges) - sum(len(items) for items in grouped.values())

    for key, source_edges in sorted(grouped.items(), key=lambda item: item[0]):
        edge_type, source, target, signature_attrs = key
        ordered = sorted(source_edges, key=lambda edge: edge.edge_id)
        attributes = dict(ordered[0].attributes)
        if len(ordered) > 1 or source != ordered[0].source_node_id or target != ordered[0].target_node_id:
            attributes.update(
                {
                    "compaction_kind": "stable_core_edge_group",
                    "source_edge_count": len(ordered),
                    "source_edge_id_samples": _sample_values(
                        [edge.edge_id for edge in ordered],
                        sample_limit,
                    ),
                }
            )
            collapsed_group_count += 1
            source_edge_count += len(ordered)
            collapsed_by_type[edge_type] += 1
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
        "removed_self_loop_edges": removed_self_loop_count,
        "collapsed_group_count": collapsed_group_count,
        "source_edge_count_in_collapsed_groups": source_edge_count,
        "collapsed_group_count_by_type": dict(sorted(collapsed_by_type.items())),
    }
    return compact_edges, report


def _edge_signature_attrs(edge: CanonicalGraphEdge) -> tuple[tuple[str, str], ...]:
    keys_by_type = {
        "HAS_EPISODE": ("role", "scope"),
        "HAS_STATE": ("role",),
        "HAS_INTERACTION_TYPE": ("role",),
        "INTERACTS_WITH": ("predicate", "raw_predicate"),
        "HAS_MEMBER_ENTITY": ("role",),
        "TRIGGERED_BY": ("role", "event_type"),
        "STARTS_AT": ("temporal_relation", "event_type"),
        "ENDS_AT": ("temporal_relation", "event_type"),
        "OVERLAPS_WITH": ("temporal_relation",),
        "TEMPORALLY_NEAR": ("temporal_relation",),
    }
    pairs: list[tuple[str, str]] = []
    for key in keys_by_type.get(edge.edge_type, ()):
        value = edge.attributes.get(key)
        if value is not None:
            pairs.append((key, _stable_json(value)))
    return tuple(pairs)


def _remap_edge_attributes(
    attributes: dict[str, object],
    *,
    entity_id_map: dict[str, str],
) -> dict[str, object]:
    remapped = dict(attributes)
    for key, value in attributes.items():
        if "entity_id" in key or key in {"member_entity_id", "shared_entity_ids"}:
            remapped[key] = _remap_attribute_value(value, entity_id_map=entity_id_map)
    return remapped


def _remap_attribute_value(value: object, *, entity_id_map: dict[str, str]) -> object:
    if isinstance(value, str):
        return entity_id_map.get(value, value)
    if isinstance(value, list):
        return [_remap_attribute_value(item, entity_id_map=entity_id_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_attribute_value(item, entity_id_map=entity_id_map) for item in value)
    if isinstance(value, dict):
        return {
            key: _remap_attribute_value(item, entity_id_map=entity_id_map)
            for key, item in value.items()
        }
    return value


def _remap_entity_id(entity_id: str | None, entity_id_map: dict[str, str]) -> str | None:
    if not entity_id:
        return entity_id
    return entity_id_map.get(entity_id, entity_id)


def _remap_entity_id_list(
    entity_ids: list[str | None],
    entity_id_map: dict[str, str],
) -> list[str]:
    remapped: list[str] = []
    seen: set[str] = set()
    for entity_id in entity_ids:
        if not entity_id:
            continue
        value = entity_id_map.get(entity_id, entity_id)
        if value in seen:
            continue
        seen.add(value)
        remapped.append(value)
    return remapped


def _normalized_label(node: CanonicalEntityNode) -> str:
    for value in (
        node.attributes.get("canonical_label"),
        node.attributes.get("semantic_group"),
        node.stable_identity_canonical_label,
        node.label,
    ):
        text = str(value or "").strip().lower()
        if text:
            return " ".join(text.split())
    return "unknown"


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


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
    return f"edge:stable_core:{edge_type.lower()}:{digest}"


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
