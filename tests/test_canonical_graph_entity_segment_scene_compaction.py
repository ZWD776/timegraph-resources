from __future__ import annotations

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import EntitySegment
from vlkg.core.transformers import (
    build_canonical_temporal_memory_graph,
    compact_entity_segments_by_scene_span,
)

from tests.test_canonical_temporal_memory_graph import _sample_temporal_graph_document


def test_entity_segment_scene_compaction_merges_same_scene_entity_segments() -> None:
    document = _sample_temporal_graph_document()
    document = document.model_copy(
        update={
            "entity_segments": [
                *document.entity_segments,
                EntitySegment(
                    segment_id="seg_player_1_c",
                    video_id="video_001",
                    entity_id="player_1",
                    scene_id="scene_002",
                    segment_index=3,
                    segment_count=3,
                    start_frame_rank=5,
                    end_frame_rank=6,
                    start_frame=16,
                    end_frame=17,
                    start_ts=1.6,
                    end_ts=1.7,
                    support_frames=2,
                    is_static=False,
                ),
            ],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    result = compact_entity_segments_by_scene_span(graph)
    compact_graph = result.graph

    assert isinstance(compact_graph, CanonicalTemporalMemoryGraph)
    assert compact_graph.node_count < graph.node_count
    assert compact_graph.edge_count < graph.edge_count
    assert result.report["episode_compaction"]["removed_episode_nodes"] >= 1
    compact_segments = [
        node
        for node in compact_graph.episode_nodes
        if node.attributes.get("compaction_kind") == "scene_entity_segment_span"
    ]
    assert compact_segments
    assert any(node.primary_entity_id == "player_1" for node in compact_segments)


def test_entity_segment_scene_compact_graph_remains_retrieval_compatible() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    compact_graph = compact_entity_segments_by_scene_span(graph).graph

    session = CanonicalHybridRetrievalSession(compact_graph)
    output = session.retrieve(
        MemoryQuery(query_text="when was the player near the hoop", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any("near" in str(item.get("summary", "")).lower() for item in output.items)
