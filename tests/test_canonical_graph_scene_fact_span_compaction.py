from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transformers.canonical_graph_scene_fact_span_compaction import (
    compact_scene_facts_by_span,
)


def test_compacts_scene_local_state_and_relation_fact_spans() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        episode_nodes=[
            state_episode("state1", frame=10),
            state_episode("state2", frame=12),
            state_episode("state3", frame=20, attr_value="red"),
            relation_episode("rel1", frame=11),
            relation_episode("rel2", frame=13),
            relation_episode("rel3", frame=14, scene_id="scene:2"),
        ],
        edges=[
            edge("e1", "HAS_EPISODE", "entity:a", "episode:state1", role="subject"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:state2", role="subject"),
            edge("e3", "HAS_STATE", "episode:state1", "state_value:blue", role="value"),
            edge("e4", "HAS_STATE", "episode:state2", "state_value:blue", role="value"),
            edge("e5", "IN_SCENE", "episode:state1", "scene:1"),
            edge("e6", "IN_SCENE", "episode:state2", "scene:1"),
            edge("e7", "HAS_EPISODE", "entity:a", "episode:rel1", role="subject"),
            edge("e8", "HAS_EPISODE", "entity:a", "episode:rel2", role="subject"),
            edge("e9", "HAS_EPISODE", "entity:b", "episode:rel1", role="object"),
            edge("e10", "HAS_EPISODE", "entity:b", "episode:rel2", role="object"),
            edge("e11", "INTERACTS_WITH", "entity:a", "entity:b", predicate="NEAR"),
        ],
    )

    result = compact_scene_facts_by_span(graph)

    compact_nodes = [
        node
        for node in result.graph.episode_nodes
        if str(node.attributes.get("compaction_kind", "")).endswith("_fact_span")
    ]
    assert len(compact_nodes) == 2
    assert result.report["episode_compaction"]["compact_group_count_by_kind"] == {
        "RELATION": 1,
        "STATE": 1,
    }
    state_span = next(node for node in compact_nodes if node.episode_kind == "STATE")
    assert state_span.attr_value == "blue"
    assert state_span.time_range.start_frame == 10
    assert state_span.time_range.end_frame == 12
    relation_span = next(node for node in compact_nodes if node.episode_kind == "RELATION")
    assert relation_span.subject_entity_id == "entity:a"
    assert relation_span.object_entity_id == "entity:b"
    assert relation_span.time_range.start_frame == 11
    assert relation_span.time_range.end_frame == 13
    assert result.report["episode_compaction"]["removed_episode_nodes"] == 2
    assert result.report["edge_compaction"]["removed_edges"] >= 3


def test_scene_fact_span_compact_graph_remains_retrieval_compatible() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        episode_nodes=[
            state_episode("state1", frame=10),
            state_episode("state2", frame=12),
            relation_episode("rel1", frame=11),
            relation_episode("rel2", frame=13),
        ],
        edges=[
            edge("e1", "HAS_EPISODE", "entity:a", "episode:state1", role="subject"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:state2", role="subject"),
            edge("e3", "HAS_STATE", "episode:state1", "state_value:blue", role="value"),
            edge("e4", "HAS_STATE", "episode:state2", "state_value:blue", role="value"),
            edge("e5", "HAS_EPISODE", "entity:a", "episode:rel1", role="subject"),
            edge("e6", "HAS_EPISODE", "entity:a", "episode:rel2", role="subject"),
            edge("e7", "HAS_EPISODE", "entity:b", "episode:rel1", role="object"),
            edge("e8", "HAS_EPISODE", "entity:b", "episode:rel2", role="object"),
        ],
    )
    compact_graph = compact_scene_facts_by_span(graph).graph

    session = CanonicalHybridRetrievalSession(compact_graph)
    output = session.retrieve(
        MemoryQuery(query_text="blue jacket near door", limit=5),
        mode="temporal",
        limit=5,
    )

    assert output.items


def state_episode(
    suffix: str,
    *,
    frame: int,
    scene_id: str = "scene:1",
    attr_value: str = "blue",
) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind="STATE",
        label=f"entity:a color={attr_value}",
        time_range=CanonicalTimeRange(
            start_frame=frame,
            end_frame=frame,
            start_ts=float(frame),
            end_ts=float(frame),
            support_frames=1,
        ),
        scene_id=scene_id,
        primary_entity_id="entity:a",
        participant_entity_ids=["entity:a"],
        attr_key="color",
        attr_value=attr_value,
        source_ref_id=suffix,
    )


def relation_episode(
    suffix: str,
    *,
    frame: int,
    scene_id: str = "scene:1",
) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind="RELATION",
        label="entity:a NEAR entity:b",
        time_range=CanonicalTimeRange(
            start_frame=frame,
            end_frame=frame,
            start_ts=float(frame),
            end_ts=float(frame),
            support_frames=1,
        ),
        scene_id=scene_id,
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
