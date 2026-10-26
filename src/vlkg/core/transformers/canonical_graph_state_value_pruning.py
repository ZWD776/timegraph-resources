from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from vlkg.core.models.canonical_graph import (
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
)


@dataclass(frozen=True)
class StateValuePruningResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def prune_state_value_index_layer(
    graph: CanonicalTemporalMemoryGraph,
) -> StateValuePruningResult:
    """Remove the state-value index layer from an experimental graph.

    STATE episode nodes already carry attr_key/attr_value fields. This pass
    removes the separate state_value nodes and HAS_STATE edges, while preserving
    every STATE/RELATION episode and all entity/scene/time fields. It is a
    graph-only compression experiment and does not require retrieval changes.
    """

    removed_state_value_node_ids = {node.node_id for node in graph.state_value_nodes}
    retained_edges: list[CanonicalGraphEdge] = []
    removed_edge_count_by_type: Counter[str] = Counter()
    removed_edge_count_by_reason: Counter[str] = Counter()

    for edge in graph.edges:
        source_removed = edge.source_node_id in removed_state_value_node_ids
        target_removed = edge.target_node_id in removed_state_value_node_ids
        edge_type_removed = edge.edge_type == "HAS_STATE"
        if source_removed or target_removed or edge_type_removed:
            removed_edge_count_by_type[edge.edge_type] += 1
            if source_removed:
                removed_edge_count_by_reason["source_state_value_pruned"] += 1
            if target_removed:
                removed_edge_count_by_reason["target_state_value_pruned"] += 1
            if edge_type_removed:
                removed_edge_count_by_reason["has_state_edge_pruned"] += 1
            continue
        retained_edges.append(edge)

    pruned_graph = graph.model_copy(
        update={
            "state_value_nodes": [],
            "edges": retained_edges,
        }
    )
    report = {
        "contract": "canonical_state_value_index_pruning_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "scope": (
                "Graph-only experiment. Removes state_value nodes and HAS_STATE "
                "index edges. STATE episodes keep attr_key/attr_value fields; "
                "RELATION episodes, entities, scenes, and transition events are "
                "preserved."
            ),
            "frozen_mainline_unchanged": True,
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(pruned_graph),
        "state_value_pruning": {
            "removed_state_value_nodes": len(removed_state_value_node_ids),
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
        "state_value_node_count": (
            report["after"]["state_value_node_count"]
            - report["before"]["state_value_node_count"]
        ),
    }
    return StateValuePruningResult(graph=pruned_graph, report=report)


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
