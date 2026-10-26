from __future__ import annotations

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import Entity, EntitySegment, RelationEpisode
from vlkg.core.transformers import (
    build_canonical_temporal_memory_graph,
    compact_canonical_graph_to_stable_core,
)

from tests.test_canonical_temporal_memory_graph import _sample_temporal_graph_document


def test_stable_core_compaction_collapses_same_scene_label_fragments() -> None:
    document = _sample_temporal_graph_document()
    cone_entities = [
        Entity(
            entity_id=f"cone_{index}",
            video_id="video_001",
            category="cone",
            first_seen_frame=index,
            last_seen_frame=index,
            first_seen_ts=index / 10,
            last_seen_ts=index / 10,
            observation_count=1,
            node_role="dynamic",
            is_fragment=True,
        )
        for index in range(3)
    ]
    cone_segments = [
        EntitySegment(
            segment_id=f"seg_cone_{index}",
            video_id="video_001",
            entity_id=f"cone_{index}",
            scene_id="scene_001",
            segment_index=1,
            segment_count=1,
            start_frame_rank=index,
            end_frame_rank=index,
            start_frame=index,
            end_frame=index,
            start_ts=index / 10,
            end_ts=index / 10,
            support_frames=1,
            is_static=False,
        )
        for index in range(3)
    ]
    document = document.model_copy(
        update={
            "entities": [*document.entities, *cone_entities],
            "entity_segments": [*document.entity_segments, *cone_segments],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    result = compact_canonical_graph_to_stable_core(
        graph,
        min_scene_label_group_size=2,
    )

    assert isinstance(result.graph, CanonicalTemporalMemoryGraph)
    assert result.graph.node_count < graph.node_count
    assert result.report["entity_compaction"]["removed_entity_nodes"] >= 2
    compact_cones = [
        node
        for node in result.graph.entity_nodes
        if node.label == "cone"
        and node.attributes.get("compaction_kind") == "stable_core_entity_group"
    ]
    assert len(compact_cones) == 1
    assert compact_cones[0].attributes["source_entity_count"] == 3


def test_stable_core_compaction_can_disable_scene_label_fragment_merge() -> None:
    document = _sample_temporal_graph_document()
    cone_entities = [
        Entity(
            entity_id=f"cone_{index}",
            video_id="video_001",
            category="cone",
            first_seen_frame=index,
            last_seen_frame=index,
            first_seen_ts=index / 10,
            last_seen_ts=index / 10,
            observation_count=1,
            node_role="dynamic",
            is_fragment=True,
        )
        for index in range(3)
    ]
    cone_segments = [
        EntitySegment(
            segment_id=f"seg_cone_{index}",
            video_id="video_001",
            entity_id=f"cone_{index}",
            scene_id="scene_001",
            segment_index=1,
            segment_count=1,
            start_frame_rank=index,
            end_frame_rank=index,
            start_frame=index,
            end_frame=index,
            start_ts=index / 10,
            end_ts=index / 10,
            support_frames=1,
            is_static=False,
        )
        for index in range(3)
    ]
    document = document.model_copy(
        update={
            "entities": [*document.entities, *cone_entities],
            "entity_segments": [*document.entity_segments, *cone_segments],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    result = compact_canonical_graph_to_stable_core(
        graph,
        compact_scene_label_entities=False,
        min_scene_label_group_size=2,
    )

    compact_cones = [
        node
        for node in result.graph.entity_nodes
        if node.label == "cone"
        and node.attributes.get("compaction_kind") == "stable_core_entity_group"
    ]
    assert compact_cones == []
    assert {node.entity_id for node in result.graph.entity_nodes if node.label == "cone"} == {
        "cone_0",
        "cone_1",
        "cone_2",
    }


def test_stable_core_compaction_remaps_episode_entity_references() -> None:
    document = _sample_temporal_graph_document()
    document = document.model_copy(
        update={
            "relation_episodes": [
                *document.relation_episodes,
                RelationEpisode(
                    relation_episode_id="rel_same_identity",
                    video_id="video_001",
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="beside",
                    object_entity_id="hoop_1",
                    start_frame=13,
                    end_frame=13,
                    start_ts=1.3,
                    end_ts=1.3,
                    support_frames=1,
                ),
            ],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    compact_graph = compact_canonical_graph_to_stable_core(graph).graph

    entity_ids = {node.entity_id for node in compact_graph.entity_nodes}
    for episode in compact_graph.episode_nodes:
        for entity_id in episode.participant_entity_ids:
            assert entity_id in entity_ids


def test_stable_core_compact_graph_remains_retrieval_compatible() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    compact_graph = compact_canonical_graph_to_stable_core(graph).graph

    session = CanonicalHybridRetrievalSession(compact_graph)
    output = session.retrieve(
        MemoryQuery(query_text="when was the player near the hoop", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any("near" in str(item.get("summary", "")).lower() for item in output.items)
