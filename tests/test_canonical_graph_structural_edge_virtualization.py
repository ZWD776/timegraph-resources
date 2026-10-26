from __future__ import annotations

from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
    _build_projection,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transformers import (
    build_canonical_temporal_memory_graph,
    virtualize_canonical_graph_structure_edges,
)

from tests.test_canonical_temporal_memory_graph import _sample_temporal_graph_document


def test_virtualize_canonical_graph_structure_edges_removes_only_selected_edges() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = virtualize_canonical_graph_structure_edges(graph)
    virtual_graph = result.graph

    assert virtual_graph.node_count == graph.node_count
    assert virtual_graph.edge_count < graph.edge_count
    assert {
        edge.edge_type
        for edge in virtual_graph.edges
    }.isdisjoint({"HAS_EPISODE", "BELONGS_TO_SEGMENT", "IN_SCENE"})
    assert result.report["edge_virtualization"]["removed_edge_count_by_type"]


def test_virtualized_structure_edges_are_rebuilt_in_projection() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    virtual_graph = virtualize_canonical_graph_structure_edges(graph).graph

    projection = _build_projection(virtual_graph)

    relation_scene_links = {
        link.edge_type
        for link in projection.adjacency["episode:relation:rel_1"]
        if link.neighbor_id == "scene:scene_002"
    }
    player_relation_links = {
        link.edge_type
        for link in projection.adjacency["entity:player_1"]
        if link.neighbor_id == "episode:relation:rel_1"
    }
    relation_segment_links = {
        link.edge_type
        for link in projection.adjacency["episode:relation:rel_1"]
        if link.neighbor_id == "episode:entity_segment:seg_player_1_b"
    }

    assert "IN_SCENE" in relation_scene_links
    assert "HAS_EPISODE" in player_relation_links
    assert "BELONGS_TO_SEGMENT" in relation_segment_links


def test_virtualized_structure_graph_remains_retrieval_compatible() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    virtual_graph = virtualize_canonical_graph_structure_edges(graph).graph

    session = CanonicalHybridRetrievalSession(virtual_graph)
    output = session.retrieve(
        MemoryQuery(query_text="when was the player near the hoop", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items
    assert any("near" in str(item.get("summary", "")).lower() for item in output.items)
