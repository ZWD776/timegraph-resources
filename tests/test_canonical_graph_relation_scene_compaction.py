from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.core.transformers.canonical_graph_relation_scene_span_compaction import (
    compact_relations_by_scene_span,
)


def test_compacts_repeated_scene_local_relation_episodes() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        episode_nodes=[
            relation_episode("r1", frame=10),
            relation_episode("r2", frame=12),
            relation_episode("r3", frame=20, scene_id="scene:2"),
        ],
        edges=[
            edge("e1", "HAS_EPISODE", "entity:a", "episode:r1", role="subject"),
            edge("e2", "HAS_EPISODE", "entity:a", "episode:r2", role="subject"),
            edge("e3", "HAS_EPISODE", "entity:a", "episode:r3", role="subject"),
            edge("e4", "IN_SCENE", "episode:r1", "scene:1"),
            edge("e5", "IN_SCENE", "episode:r2", "scene:1"),
            edge("e6", "IN_SCENE", "episode:r3", "scene:2"),
            edge("e7", "HAS_INTERACTION_TYPE", "episode:r1", "interaction_type:near"),
            edge("e8", "HAS_INTERACTION_TYPE", "episode:r2", "interaction_type:near"),
        ],
    )

    result = compact_relations_by_scene_span(graph)

    relation_nodes = [
        node for node in result.graph.episode_nodes if node.episode_kind == "RELATION"
    ]
    assert len(relation_nodes) == 2
    compact = next(
        node
        for node in relation_nodes
        if node.attributes.get("compaction_kind") == "scene_relation_span"
    )
    assert compact.scene_id == "scene:1"
    assert compact.subject_entity_id == "entity:a"
    assert compact.object_entity_id == "entity:b"
    assert compact.time_range.start_frame == 10
    assert compact.time_range.end_frame == 12
    assert compact.attributes["source_episode_count"] == 2
    assert result.report["episode_compaction"]["removed_episode_nodes"] == 1
    assert result.report["edge_compaction"]["removed_edges"] >= 2


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
