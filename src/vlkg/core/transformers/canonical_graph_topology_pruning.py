from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
)


@dataclass(frozen=True)
class TopologyPruningResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


DEFAULT_PRUNED_EPISODE_KINDS: tuple[str, ...] = (
    "SCENE_SEGMENT",
    "ENTITY_SEGMENT",
)


def prune_low_information_topology(
    graph: CanonicalTemporalMemoryGraph,
    *,
    prune_episode_kinds: tuple[str, ...] = DEFAULT_PRUNED_EPISODE_KINDS,
    prune_interaction_type_nodes: bool = True,
) -> TopologyPruningResult:
    """Remove derivable topology/index layers from a canonical graph view.

    This pass removes low-information topology/index layers while preserving
    scene, entity, state, relation, and transition fact records. It does not
    modify retrieval logic and does not inspect questions or gold answers.
    """

    pruned_episode_kinds = set(prune_episode_kinds)
    retained_episodes: list[CanonicalEpisodeNode] = []
    removed_episode_node_ids: set[str] = set()
    removed_episode_count_by_kind: Counter[str] = Counter()

    for node in graph.episode_nodes:
        if node.episode_kind in pruned_episode_kinds:
            removed_episode_node_ids.add(node.node_id)
            removed_episode_count_by_kind[node.episode_kind] += 1
            continue
        retained_episodes.append(node)

    removed_interaction_node_ids: set[str] = set()
    if prune_interaction_type_nodes:
        removed_interaction_node_ids = {
            node.node_id
            for node in graph.interaction_type_nodes
        }
        interaction_type_nodes = []
    else:
        interaction_type_nodes = graph.interaction_type_nodes

    removed_node_ids = removed_episode_node_ids | removed_interaction_node_ids
    retained_edges: list[CanonicalGraphEdge] = []
    removed_edge_count_by_type: Counter[str] = Counter()
    removed_edge_count_by_reason: Counter[str] = Counter()

    for edge in graph.edges:
        source_removed = edge.source_node_id in removed_node_ids
        target_removed = edge.target_node_id in removed_node_ids
        if source_removed or target_removed:
            removed_edge_count_by_type[edge.edge_type] += 1
            if source_removed:
                removed_edge_count_by_reason["source_node_pruned"] += 1
            if target_removed:
                removed_edge_count_by_reason["target_node_pruned"] += 1
            continue
        if prune_interaction_type_nodes and edge.edge_type == "HAS_INTERACTION_TYPE":
            removed_edge_count_by_type[edge.edge_type] += 1
            removed_edge_count_by_reason["interaction_type_edge_pruned"] += 1
            continue
        retained_edges.append(edge)

    pruned_graph = graph.model_copy(
        update={
            "episode_nodes": retained_episodes,
            "interaction_type_nodes": interaction_type_nodes,
            "edges": retained_edges,
        }
    )
    report = {
        "contract": "canonical_topology_pruning_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "pruned_episode_kinds": sorted(pruned_episode_kinds),
            "prune_interaction_type_nodes": prune_interaction_type_nodes,
            "scope": (
                "Graph-only derived view. Prunes scene/entity segment episodes "
                "and interaction-type hub nodes, plus incident edges. Relation "
                "and state fact episodes keep their predicate/value fields."
            ),
            "graph_semantic_rule": True,
            "does_not_inspect_questions_or_gold_answers": True,
            "does_not_inspect_retrieval_or_qa_outputs": True,
            "frozen_mainline_unchanged": True,
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(pruned_graph),
        "node_pruning": {
            "removed_episode_nodes": len(removed_episode_node_ids),
            "removed_episode_count_by_kind": dict(sorted(removed_episode_count_by_kind.items())),
            "removed_interaction_type_nodes": len(removed_interaction_node_ids),
        },
        "edge_pruning": {
            "removed_edges": len(graph.edges) - len(retained_edges),
            "removed_edge_count_by_type": dict(sorted(removed_edge_count_by_type.items())),
            "removed_edge_count_by_reason": dict(sorted(removed_edge_count_by_reason.items())),
        },
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
        "episode_node_count": (
            report["after"]["episode_node_count"]
            - report["before"]["episode_node_count"]
        ),
        "interaction_type_node_count": (
            report["after"]["interaction_type_node_count"]
            - report["before"]["interaction_type_node_count"]
        ),
    }
    return TopologyPruningResult(graph=pruned_graph, report=report)


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
