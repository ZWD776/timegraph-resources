from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from vlkg.core.models.canonical_graph import (
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTransitionEventNode,
)


@dataclass(frozen=True)
class TransitionEventPruningResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


DEFAULT_PRUNED_STATIC_LIFECYCLE_EVENT_TYPES: tuple[str, ...] = (
    "APPEAR",
    "DISAPPEAR",
)


def prune_static_lifecycle_transition_events(
    graph: CanonicalTemporalMemoryGraph,
    *,
    prune_event_types: tuple[str, ...] = DEFAULT_PRUNED_STATIC_LIFECYCLE_EVENT_TYPES,
) -> TransitionEventPruningResult:
    """Prune static-entity lifecycle events from an experimental graph.

    This graph-only compression removes APPEAR/DISAPPEAR transition nodes only
    when they are attached to an entity marked static. Dynamic lifecycle events,
    relation boundary events, STATE/RELATION episodes, and entity records are
    preserved so the downstream retrieval and QA code can run unchanged.
    """

    pruned_event_types = set(prune_event_types)
    static_entity_ids = {
        node.entity_id
        for node in graph.entity_nodes
        if node.entity_id and (node.is_static or node.node_role == "static")
    }

    retained_transition_events: list[CanonicalTransitionEventNode] = []
    removed_transition_node_ids: set[str] = set()
    removed_event_count_by_type: Counter[str] = Counter()
    retained_event_count_by_type: Counter[str] = Counter()
    skipped_event_count_by_reason: Counter[str] = Counter()

    for node in graph.transition_event_nodes:
        if node.event_type not in pruned_event_types:
            retained_transition_events.append(node)
            retained_event_count_by_type[node.event_type] += 1
            skipped_event_count_by_reason["event_type_not_pruned"] += 1
            continue
        if node.entity_id not in static_entity_ids:
            retained_transition_events.append(node)
            retained_event_count_by_type[node.event_type] += 1
            skipped_event_count_by_reason["non_static_or_unknown_entity"] += 1
            continue
        removed_transition_node_ids.add(node.node_id)
        removed_event_count_by_type[node.event_type] += 1

    retained_edges: list[CanonicalGraphEdge] = []
    removed_edge_count_by_type: Counter[str] = Counter()
    removed_edge_count_by_reason: Counter[str] = Counter()

    for edge in graph.edges:
        source_removed = edge.source_node_id in removed_transition_node_ids
        target_removed = edge.target_node_id in removed_transition_node_ids
        if source_removed or target_removed:
            removed_edge_count_by_type[edge.edge_type] += 1
            if source_removed:
                removed_edge_count_by_reason["source_transition_event_pruned"] += 1
            if target_removed:
                removed_edge_count_by_reason["target_transition_event_pruned"] += 1
            continue
        retained_edges.append(edge)

    pruned_graph = graph.model_copy(
        update={
            "transition_event_nodes": retained_transition_events,
            "edges": retained_edges,
        }
    )
    report = {
        "contract": "canonical_static_lifecycle_transition_event_pruning_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "pruned_event_types": sorted(pruned_event_types),
            "static_entity_rule": "entity.is_static == true or entity.node_role == 'static'",
            "scope": (
                "Graph-only experiment. Removes static-entity APPEAR/DISAPPEAR "
                "transition-event nodes and incident edges. Dynamic lifecycle "
                "events, relation boundary events, facts, scenes, entities, and "
                "state values are preserved."
            ),
            "frozen_mainline_unchanged": True,
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(pruned_graph),
        "transition_event_pruning": {
            "static_entity_count": len(static_entity_ids),
            "removed_transition_event_nodes": len(removed_transition_node_ids),
            "removed_event_count_by_type": dict(sorted(removed_event_count_by_type.items())),
            "retained_event_count_by_type": dict(sorted(retained_event_count_by_type.items())),
            "skipped_event_count_by_reason": dict(
                sorted(skipped_event_count_by_reason.items())
            ),
        },
        "edge_pruning": {
            "removed_edges": len(graph.edges) - len(retained_edges),
            "removed_edge_count_by_type": dict(sorted(removed_edge_count_by_type.items())),
            "removed_edge_count_by_reason": dict(
                sorted(removed_edge_count_by_reason.items())
            ),
        },
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
        "transition_event_node_count": (
            report["after"]["transition_event_node_count"]
            - report["before"]["transition_event_node_count"]
        ),
    }
    return TransitionEventPruningResult(graph=pruned_graph, report=report)


def prune_lifecycle_transition_events(
    graph: CanonicalTemporalMemoryGraph,
    *,
    prune_event_types: tuple[str, ...] = DEFAULT_PRUNED_STATIC_LIFECYCLE_EVENT_TYPES,
) -> TransitionEventPruningResult:
    """Prune all APPEAR/DISAPPEAR transition nodes from an experiment graph.

    This is a deliberately aggressive graph-only ablation. It keeps relation
    boundary events and every fact episode, but removes lifecycle event nodes and
    their incident edges regardless of entity role. It is intended to measure the
    effect of graph size/noise reduction, not as a proposed frozen mainline.
    """

    pruned_event_types = set(prune_event_types)
    retained_transition_events: list[CanonicalTransitionEventNode] = []
    removed_transition_node_ids: set[str] = set()
    removed_event_count_by_type: Counter[str] = Counter()
    retained_event_count_by_type: Counter[str] = Counter()

    for node in graph.transition_event_nodes:
        if node.event_type in pruned_event_types:
            removed_transition_node_ids.add(node.node_id)
            removed_event_count_by_type[node.event_type] += 1
            continue
        retained_transition_events.append(node)
        retained_event_count_by_type[node.event_type] += 1

    retained_edges: list[CanonicalGraphEdge] = []
    removed_edge_count_by_type: Counter[str] = Counter()
    removed_edge_count_by_reason: Counter[str] = Counter()

    for edge in graph.edges:
        source_removed = edge.source_node_id in removed_transition_node_ids
        target_removed = edge.target_node_id in removed_transition_node_ids
        if source_removed or target_removed:
            removed_edge_count_by_type[edge.edge_type] += 1
            if source_removed:
                removed_edge_count_by_reason["source_transition_event_pruned"] += 1
            if target_removed:
                removed_edge_count_by_reason["target_transition_event_pruned"] += 1
            continue
        retained_edges.append(edge)

    pruned_graph = graph.model_copy(
        update={
            "transition_event_nodes": retained_transition_events,
            "edges": retained_edges,
        }
    )
    report = {
        "contract": "canonical_lifecycle_transition_event_pruning_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "pruned_event_types": sorted(pruned_event_types),
            "scope": (
                "Aggressive graph-only ablation. Removes all APPEAR/DISAPPEAR "
                "transition-event nodes and incident edges. Relation boundary "
                "events, facts, scenes, entities, and state values are preserved."
            ),
            "frozen_mainline_unchanged": True,
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(pruned_graph),
        "transition_event_pruning": {
            "removed_transition_event_nodes": len(removed_transition_node_ids),
            "removed_event_count_by_type": dict(sorted(removed_event_count_by_type.items())),
            "retained_event_count_by_type": dict(sorted(retained_event_count_by_type.items())),
        },
        "edge_pruning": {
            "removed_edges": len(graph.edges) - len(retained_edges),
            "removed_edge_count_by_type": dict(sorted(removed_edge_count_by_type.items())),
            "removed_edge_count_by_reason": dict(
                sorted(removed_edge_count_by_reason.items())
            ),
        },
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
        "transition_event_node_count": (
            report["after"]["transition_event_node_count"]
            - report["before"]["transition_event_node_count"]
        ),
    }
    return TransitionEventPruningResult(graph=pruned_graph, report=report)


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
