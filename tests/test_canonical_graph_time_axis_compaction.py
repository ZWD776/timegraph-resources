from __future__ import annotations

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import RelationEpisode, StateEpisode
from vlkg.core.transformers import (
    build_canonical_temporal_memory_graph,
    compact_canonical_graph_by_time_axis,
)

from tests.test_canonical_temporal_memory_graph import _sample_temporal_graph_document


def test_compact_canonical_graph_by_time_axis_merges_adjacent_episode_bursts() -> None:
    document = _sample_temporal_graph_document()
    document = document.model_copy(
        update={
            "state_episodes": [
                *document.state_episodes,
                StateEpisode(
                    state_episode_id="state_1_continued",
                    video_id="video_001",
                    entity_id="player_1",
                    attr_key="action",
                    attr_value="dribbling",
                    start_frame=5,
                    end_frame=6,
                    start_ts=0.5,
                    end_ts=0.6,
                    support_frames=2,
                ),
            ],
            "relation_episodes": [
                *document.relation_episodes,
                RelationEpisode(
                    relation_episode_id="rel_1_continued",
                    video_id="video_001",
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="next to",
                    object_entity_id="hoop_1",
                    start_frame=14,
                    end_frame=14,
                    start_ts=1.4,
                    end_ts=1.4,
                    support_frames=1,
                ),
            ],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    result = compact_canonical_graph_by_time_axis(
        graph,
        max_gap_frames=2,
        max_gap_seconds=0.25,
    )
    compact_graph = result.graph

    assert isinstance(compact_graph, CanonicalTemporalMemoryGraph)
    assert graph.node_count > compact_graph.node_count
    assert graph.edge_count > compact_graph.edge_count
    assert result.report["episode_compaction"]["removed_episode_nodes"] == 2

    compact_nodes = [
        node
        for node in compact_graph.episode_nodes
        if node.attributes.get("compaction_kind") == "time_axis_episode_burst"
    ]
    assert {node.episode_kind for node in compact_nodes} == {"STATE", "RELATION"}

    dribbling = next(node for node in compact_nodes if node.episode_kind == "STATE")
    assert dribbling.time_range.start_frame == 1
    assert dribbling.time_range.end_frame == 6
    assert dribbling.attributes["source_episode_count"] == 2
    assert dribbling.attributes["observed_interval_count"] == 2
    assert dribbling.primary_entity_id == "player_1"

    near = next(node for node in compact_nodes if node.episode_kind == "RELATION")
    assert near.time_range.start_frame == 12
    assert near.time_range.end_frame == 14
    assert near.subject_entity_id == "player_1"
    assert near.object_entity_id == "hoop_1"
    assert "standing near" in str(near.raw_predicate)
    assert "next to" in str(near.raw_predicate)


def test_time_axis_compact_graph_remains_retrieval_compatible() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    compact_graph = compact_canonical_graph_by_time_axis(graph).graph

    session = CanonicalHybridRetrievalSession(compact_graph)
    output = session.retrieve(
        MemoryQuery(query_text="when was the player near the hoop", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any("near" in str(item.get("summary", "")).lower() for item in output.items)


def test_stable_identity_time_axis_compaction_merges_member_entity_facts() -> None:
    document = _sample_temporal_graph_document()
    document = document.model_copy(
        update={
            "relation_episodes": [
                *document.relation_episodes,
                RelationEpisode(
                    relation_episode_id="rel_1_same_identity",
                    video_id="video_001",
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="beside",
                    object_entity_id="hoop_1",
                    start_frame=14,
                    end_frame=14,
                    start_ts=1.4,
                    end_ts=1.4,
                    support_frames=1,
                ),
            ],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    result = compact_canonical_graph_by_time_axis(
        graph,
        max_gap_frames=2,
        max_gap_seconds=0.25,
        signature_entity_scope="stable_identity",
    )
    compact_graph = result.graph

    compact_relations = [
        node
        for node in compact_graph.episode_nodes
        if node.episode_kind == "RELATION"
        and node.attributes.get("compaction_kind") == "time_axis_episode_burst"
    ]
    assert len(compact_relations) == 1
    relation = compact_relations[0]
    assert relation.attributes["signature_entity_scope"] == "stable_identity"
    assert relation.attributes["source_episode_count"] == 2
    assert relation.attributes["signature_subject_entity_id"]
    assert relation.attributes["signature_object_entity_id"]
