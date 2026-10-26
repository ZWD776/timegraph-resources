from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalInteractionTypeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transformers.canonical_graph_topology_pruning import (
    prune_low_information_topology,
)


def test_prunes_segment_episodes_and_interaction_hubs_only() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        interaction_type_nodes=[
            CanonicalInteractionTypeNode(
                node_id="interaction_type:near",
                video_id="video",
                predicate="NEAR",
                label="NEAR",
            )
        ],
        episode_nodes=[
            episode("scene_seg", "SCENE_SEGMENT"),
            episode("entity_seg", "ENTITY_SEGMENT"),
            episode("state", "STATE"),
            episode("relation", "RELATION"),
        ],
        edges=[
            edge("e1", "IN_SCENE", "episode:state", "scene:1"),
            edge("e2", "BELONGS_TO_SEGMENT", "episode:state", "episode:entity_seg"),
            edge("e3", "BELONGS_TO_SEGMENT", "episode:relation", "episode:scene_seg"),
            edge("e4", "HAS_INTERACTION_TYPE", "episode:relation", "interaction_type:near"),
            edge("e5", "HAS_EPISODE", "entity:a", "episode:relation"),
        ],
    )

    result = prune_low_information_topology(graph)

    assert [node.episode_kind for node in result.graph.episode_nodes] == [
        "STATE",
        "RELATION",
    ]
    assert not result.graph.interaction_type_nodes
    assert {edge.edge_id for edge in result.graph.edges} == {"e1", "e5"}
    assert result.report["node_pruning"]["removed_episode_count_by_kind"] == {
        "ENTITY_SEGMENT": 1,
        "SCENE_SEGMENT": 1,
    }
    assert result.report["node_pruning"]["removed_interaction_type_nodes"] == 1


def test_topology_pruned_graph_remains_retrieval_compatible() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        interaction_type_nodes=[
            CanonicalInteractionTypeNode(
                node_id="interaction_type:near",
                video_id="video",
                predicate="NEAR",
                label="NEAR",
            )
        ],
        episode_nodes=[
            episode("entity_seg", "ENTITY_SEGMENT"),
            episode("state", "STATE"),
            episode("relation", "RELATION"),
        ],
        edges=[
            edge("e1", "HAS_EPISODE", "entity:a", "episode:state"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:relation"),
            edge("e3", "BELONGS_TO_SEGMENT", "episode:state", "episode:entity_seg"),
            edge("e4", "HAS_INTERACTION_TYPE", "episode:relation", "interaction_type:near"),
        ],
    )
    pruned_graph = prune_low_information_topology(graph).graph

    session = CanonicalHybridRetrievalSession(pruned_graph)
    output = session.retrieve(
        MemoryQuery(query_text="blue jacket near door", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items


def episode(suffix: str, kind: str) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind=kind,  # type: ignore[arg-type]
        label="entity:a NEAR entity:b" if kind == "RELATION" else "entity:a color=blue",
        time_range=CanonicalTimeRange(
            start_frame=1,
            end_frame=2,
            start_ts=1.0,
            end_ts=2.0,
            support_frames=2,
        ),
        scene_id="scene:1",
        primary_entity_id="entity:a",
        subject_entity_id="entity:a" if kind == "RELATION" else None,
        object_entity_id="entity:b" if kind == "RELATION" else None,
        participant_entity_ids=["entity:a", "entity:b"] if kind == "RELATION" else ["entity:a"],
        attr_key="color" if kind == "STATE" else None,
        attr_value="blue" if kind == "STATE" else None,
        predicate="NEAR" if kind == "RELATION" else None,
        raw_predicate="near" if kind == "RELATION" else None,
        source_ref_id=suffix,
    )


def edge(
    edge_id: str,
    edge_type: str,
    source: str,
    target: str,
    **attributes: object,
) -> CanonicalGraphEdge:
    return CanonicalGraphEdge(
        edge_id=edge_id,
        edge_type=edge_type,  # type: ignore[arg-type]
        video_id="video",
        source_node_id=source,
        target_node_id=target,
        time_range=CanonicalTimeRange(start_frame=1, end_frame=1),
        attributes=attributes,
    )
