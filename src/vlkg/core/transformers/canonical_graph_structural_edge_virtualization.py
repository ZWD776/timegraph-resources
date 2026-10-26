from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from vlkg.core.models.canonical_graph import (
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
)


VirtualizedStructuralEdgeType = Literal[
    "HAS_EPISODE",
    "BELONGS_TO_SEGMENT",
    "IN_SCENE",
]

DEFAULT_VIRTUALIZED_STRUCTURAL_EDGE_TYPES: tuple[VirtualizedStructuralEdgeType, ...] = (
    "HAS_EPISODE",
    "BELONGS_TO_SEGMENT",
    "IN_SCENE",
)


@dataclass(frozen=True)
class StructuralEdgeVirtualizationResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]


def virtualize_canonical_graph_structure_edges(
    graph: CanonicalTemporalMemoryGraph,
    *,
    edge_types: Iterable[VirtualizedStructuralEdgeType] = DEFAULT_VIRTUALIZED_STRUCTURAL_EDGE_TYPES,
) -> StructuralEdgeVirtualizationResult:
    """Build an experimental graph that omits derivable structural/index edges.

    The full canonical graph remains the audit layer. This derived graph keeps
    nodes and semantic/temporal edges intact, while retrieval projection can
    rebuild the removed structural links from node fields such as scene_id,
    participant_entity_ids, and episode time ranges.
    """

    selected_edge_types = set(edge_types)
    retained_edges: list[CanonicalGraphEdge] = []
    removed_by_type: Counter[str] = Counter()
    retained_by_type: Counter[str] = Counter()

    for edge in graph.edges:
        if edge.edge_type in selected_edge_types:
            removed_by_type[edge.edge_type] += 1
            continue
        retained_edges.append(edge)
        retained_by_type[edge.edge_type] += 1

    virtualized_graph = graph.model_copy(update={"edges": retained_edges})
    report = {
        "contract": "canonical_structural_edge_virtualization_experiment_v1",
        "source_video_id": graph.video_id,
        "policy": {
            "virtualized_edge_types": sorted(selected_edge_types),
            "scope": (
                "Only derivable structure/index edges are removed from the "
                "derived graph. Retrieval projection is responsible for "
                "reconstructing them when the edge type is absent."
            ),
        },
        "before": _graph_size_report(graph),
        "after": _graph_size_report(virtualized_graph),
        "edge_virtualization": {
            "removed_edge_count": sum(removed_by_type.values()),
            "removed_edge_count_by_type": dict(sorted(removed_by_type.items())),
            "retained_edge_count_by_type": dict(sorted(retained_by_type.items())),
        },
    }
    report["delta"] = {
        "node_count": report["after"]["node_count"] - report["before"]["node_count"],
        "edge_count": report["after"]["edge_count"] - report["before"]["edge_count"],
    }
    return StructuralEdgeVirtualizationResult(graph=virtualized_graph, report=report)


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
