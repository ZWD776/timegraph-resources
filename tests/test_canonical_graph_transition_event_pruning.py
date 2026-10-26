from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
    CanonicalTransitionEventNode,
)
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transformers.canonical_graph_transition_event_pruning import (
    prune_lifecycle_transition_events,
    prune_static_lifecycle_transition_events,
)


def test_prunes_only_static_lifecycle_transition_events() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        entity_nodes=[
            entity("static", is_static=True, node_role="static"),
            entity("dynamic", is_static=False, node_role="dynamic"),
        ],
        transition_event_nodes=[
            transition_event("static_appear", "APPEAR", entity_id="entity:static"),
            transition_event("static_disappear", "DISAPPEAR", entity_id="entity:static"),
            transition_event("dynamic_appear", "APPEAR", entity_id="entity:dynamic"),
            transition_event("relation_start", "RELATION_START", subject_entity_id="entity:static"),
        ],
        edges=[
            edge("e1", "IN_SCENE", "transition_event:static_appear", "scene:1"),
            edge("e2", "STARTS_AT", "episode:state", "transition_event:static_appear"),
            edge("e3", "IN_SCENE", "transition_event:static_disappear", "scene:1"),
            edge("e4", "IN_SCENE", "transition_event:dynamic_appear", "scene:1"),
            edge("e5", "TRIGGERED_BY", "episode:rel", "transition_event:relation_start"),
        ],
    )

    result = prune_static_lifecycle_transition_events(graph)

    assert {
        node.node_id
        for node in result.graph.transition_event_nodes
    } == {
        "transition_event:dynamic_appear",
        "transition_event:relation_start",
    }
    assert {edge.edge_id for edge in result.graph.edges} == {"e4", "e5"}
    assert result.report["transition_event_pruning"]["removed_event_count_by_type"] == {
        "APPEAR": 1,
        "DISAPPEAR": 1,
    }
    assert result.report["edge_pruning"]["removed_edge_count_by_type"] == {
        "IN_SCENE": 2,
        "STARTS_AT": 1,
    }


def test_static_lifecycle_pruned_graph_remains_retrieval_compatible() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        entity_nodes=[
            entity("door", is_static=True, node_role="static", label="blue door"),
            entity("person", is_static=False, node_role="dynamic", label="person"),
        ],
        transition_event_nodes=[
            transition_event("door_appear", "APPEAR", entity_id="entity:door", label="blue door appears"),
            transition_event("person_appear", "APPEAR", entity_id="entity:person", label="person appears"),
        ],
        edges=[
            edge("e1", "IN_SCENE", "transition_event:door_appear", "scene:1"),
            edge("e2", "IN_SCENE", "transition_event:person_appear", "scene:1"),
        ],
    )
    pruned_graph = prune_static_lifecycle_transition_events(graph).graph

    session = CanonicalHybridRetrievalSession(pruned_graph)
    output = session.retrieve(
        MemoryQuery(query_text="person appears", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any(
        item.get("source_kind") == "transition_event"
        and item.get("metadata", {}).get("event_type") == "APPEAR"
        for item in output.items
    )


def test_prunes_all_lifecycle_transition_events_but_keeps_relation_boundaries() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        entity_nodes=[
            entity("static", is_static=True, node_role="static"),
            entity("dynamic", is_static=False, node_role="dynamic"),
        ],
        transition_event_nodes=[
            transition_event("static_appear", "APPEAR", entity_id="entity:static"),
            transition_event("dynamic_disappear", "DISAPPEAR", entity_id="entity:dynamic"),
            transition_event("relation_start", "RELATION_START", subject_entity_id="entity:static"),
            transition_event("relation_end", "RELATION_END", subject_entity_id="entity:dynamic"),
        ],
        edges=[
            edge("e1", "IN_SCENE", "transition_event:static_appear", "scene:1"),
            edge("e2", "ENDS_AT", "episode:state", "transition_event:dynamic_disappear"),
            edge("e3", "TRIGGERED_BY", "episode:rel1", "transition_event:relation_start"),
            edge("e4", "TRIGGERED_BY", "episode:rel2", "transition_event:relation_end"),
        ],
    )

    result = prune_lifecycle_transition_events(graph)

    assert {
        node.node_id
        for node in result.graph.transition_event_nodes
    } == {
        "transition_event:relation_start",
        "transition_event:relation_end",
    }
    assert {edge.edge_id for edge in result.graph.edges} == {"e3", "e4"}
    assert result.report["transition_event_pruning"]["removed_event_count_by_type"] == {
        "APPEAR": 1,
        "DISAPPEAR": 1,
    }


def entity(
    suffix: str,
    *,
    is_static: bool,
    node_role: str,
    label: str | None = None,
) -> CanonicalEntityNode:
    return CanonicalEntityNode(
        node_id=f"entity:{suffix}",
        video_id="video",
        entity_id=f"entity:{suffix}",
        label=label or suffix,
        time_range=CanonicalTimeRange(start_frame=1, end_frame=10),
        observation_count=1,
        node_role=node_role,  # type: ignore[arg-type]
        is_static=is_static,
    )


def transition_event(
    suffix: str,
    event_type: str,
    *,
    entity_id: str | None = None,
    subject_entity_id: str | None = None,
    object_entity_id: str | None = None,
    label: str | None = None,
) -> CanonicalTransitionEventNode:
    return CanonicalTransitionEventNode(
        node_id=f"transition_event:{suffix}",
        video_id="video",
        event_id=suffix,
        event_type=event_type,  # type: ignore[arg-type]
        label=label or f"{event_type} {suffix}",
        time_range=CanonicalTimeRange(start_frame=1, end_frame=1, start_ts=1.0, end_ts=1.0),
        scene_id="scene:1",
        entity_id=entity_id,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        source_ref_id=suffix,
    )


def edge(
    edge_id: str,
    edge_type: str,
    source: str,
    target: str,
) -> CanonicalGraphEdge:
    return CanonicalGraphEdge(
        edge_id=edge_id,
        edge_type=edge_type,  # type: ignore[arg-type]
        video_id="video",
        source_node_id=source,
        target_node_id=target,
        time_range=CanonicalTimeRange(start_frame=1, end_frame=1),
    )
