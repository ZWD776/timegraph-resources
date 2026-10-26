from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalStateValueNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transformers.canonical_graph_state_value_pruning import (
    prune_state_value_index_layer,
)


def test_prunes_state_value_nodes_and_has_state_edges_only() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        state_value_nodes=[
            CanonicalStateValueNode(
                node_id="state_value:color:blue",
                video_id="video",
                attr_key="color",
                attr_value="blue",
                label="color=blue",
            )
        ],
        episode_nodes=[
            state_episode("state1"),
            relation_episode("rel1"),
        ],
        edges=[
            edge("e1", "HAS_STATE", "episode:state1", "state_value:color:blue"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:state1"),
            edge("e3", "IN_SCENE", "episode:state1", "scene:1"),
            edge("e4", "HAS_EPISODE", "entity:a", "episode:rel1"),
        ],
    )

    result = prune_state_value_index_layer(graph)

    assert result.graph.state_value_nodes == []
    assert {edge.edge_id for edge in result.graph.edges} == {"e2", "e3", "e4"}
    assert {node.node_id for node in result.graph.episode_nodes} == {
        "episode:state1",
        "episode:rel1",
    }
    state = next(node for node in result.graph.episode_nodes if node.episode_kind == "STATE")
    assert state.attr_key == "color"
    assert state.attr_value == "blue"
    assert result.report["state_value_pruning"]["removed_state_value_nodes"] == 1
    assert result.report["edge_pruning"]["removed_edge_count_by_type"] == {"HAS_STATE": 1}


def test_state_value_pruned_graph_remains_retrieval_compatible() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        state_value_nodes=[
            CanonicalStateValueNode(
                node_id="state_value:color:blue",
                video_id="video",
                attr_key="color",
                attr_value="blue",
                label="color=blue",
            )
        ],
        episode_nodes=[state_episode("state1")],
        edges=[
            edge("e1", "HAS_STATE", "episode:state1", "state_value:color:blue"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:state1"),
        ],
    )
    pruned_graph = prune_state_value_index_layer(graph).graph

    session = CanonicalHybridRetrievalSession(pruned_graph)
    output = session.retrieve(
        MemoryQuery(query_text="blue jacket", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any(
        item.get("source_kind") == "episode"
        and item.get("metadata", {}).get("attr_value") == "blue"
        for item in output.items
    )


def state_episode(suffix: str) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind="STATE",
        label="entity:a color=blue",
        time_range=CanonicalTimeRange(start_frame=1, end_frame=2, start_ts=1.0, end_ts=2.0),
        scene_id="scene:1",
        primary_entity_id="entity:a",
        participant_entity_ids=["entity:a"],
        attr_key="color",
        attr_value="blue",
        source_ref_id=suffix,
    )


def relation_episode(suffix: str) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind="RELATION",
        label="entity:a NEAR entity:b",
        time_range=CanonicalTimeRange(start_frame=1, end_frame=2, start_ts=1.0, end_ts=2.0),
        scene_id="scene:1",
        primary_entity_id="entity:a",
        subject_entity_id="entity:a",
        object_entity_id="entity:b",
        participant_entity_ids=["entity:a", "entity:b"],
        predicate="NEAR",
        raw_predicate="near",
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
