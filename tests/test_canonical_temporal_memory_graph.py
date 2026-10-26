from __future__ import annotations

from pathlib import Path

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.models.temporal_graph import (
    Entity,
    EntitySegment,
    RelationEpisode,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.transformers import build_canonical_temporal_memory_graph


def _sample_temporal_graph_document() -> TemporalGraphDocument:
    return TemporalGraphDocument(
        video_id="video_001",
        source_dir=Path("data/examples/video_001"),
        frame_count=20,
        observations=[],
        entities=[
            Entity(
                entity_id="player_1",
                video_id="video_001",
                category="man",
                first_seen_frame=0,
                last_seen_frame=19,
                first_seen_ts=0.0,
                last_seen_ts=1.9,
                observation_count=20,
                node_role="dynamic",
                states_seen=["dribbling", "shooting"],
            ),
            Entity(
                entity_id="hoop_1",
                video_id="video_001",
                category="basketball hoop",
                first_seen_frame=0,
                last_seen_frame=19,
                first_seen_ts=0.0,
                last_seen_ts=1.9,
                observation_count=20,
                node_role="static",
                is_static=True,
            ),
        ],
        relation_episodes=[
            RelationEpisode(
                relation_episode_id="rel_1",
                video_id="video_001",
                subject_entity_id="player_1",
                predicate="near",
                raw_predicate="standing near",
                object_entity_id="hoop_1",
                start_frame=12,
                end_frame=13,
                start_ts=1.2,
                end_ts=1.3,
                support_frames=2,
            )
        ],
        state_episodes=[
            StateEpisode(
                state_episode_id="state_1",
                video_id="video_001",
                entity_id="player_1",
                attr_key="action",
                attr_value="dribbling",
                start_frame=1,
                end_frame=4,
                start_ts=0.1,
                end_ts=0.4,
                support_frames=4,
            ),
            StateEpisode(
                state_episode_id="state_2",
                video_id="video_001",
                entity_id="player_1",
                attr_key="action",
                attr_value="shooting",
                start_frame=11,
                end_frame=14,
                start_ts=1.1,
                end_ts=1.4,
                support_frames=4,
            ),
        ],
        scene_segments=[
            SceneSegment(
                scene_id="scene_001",
                video_id="video_001",
                scene_index=1,
                scene_count=2,
                start_frame_rank=0,
                end_frame_rank=9,
                start_frame=0,
                end_frame=9,
                start_ts=0.0,
                end_ts=0.9,
                support_frames=10,
                next_scene_id="scene_002",
            ),
            SceneSegment(
                scene_id="scene_002",
                video_id="video_001",
                scene_index=2,
                scene_count=2,
                start_frame_rank=10,
                end_frame_rank=19,
                start_frame=10,
                end_frame=19,
                start_ts=1.0,
                end_ts=1.9,
                support_frames=10,
                previous_scene_id="scene_001",
            ),
        ],
        entity_segments=[
            EntitySegment(
                segment_id="seg_player_1_a",
                video_id="video_001",
                entity_id="player_1",
                scene_id="scene_001",
                segment_index=1,
                segment_count=2,
                start_frame_rank=0,
                end_frame_rank=9,
                start_frame=0,
                end_frame=9,
                start_ts=0.0,
                end_ts=0.9,
                support_frames=10,
                next_segment_id="seg_player_1_b",
            ),
            EntitySegment(
                segment_id="seg_player_1_b",
                video_id="video_001",
                entity_id="player_1",
                scene_id="scene_002",
                segment_index=2,
                segment_count=2,
                start_frame_rank=10,
                end_frame_rank=19,
                start_frame=10,
                end_frame=19,
                start_ts=1.0,
                end_ts=1.9,
                support_frames=10,
                previous_segment_id="seg_player_1_a",
            ),
        ],
        transition_events=[
            TransitionEvent(
                event_id="event_1",
                video_id="video_001",
                event_type="STATE_TRANSITION",
                frame_rank=11,
                frame_index=11,
                timestamp_sec=1.1,
                entity_id="player_1",
                attr_key="action",
                previous_state_value="dribbling",
                current_state_value="shooting",
                source_segment_id="seg_player_1_b",
                source_episode_id="state_2",
            )
        ],
    )


def _sample_temporal_graph_with_observed_relation_events() -> TemporalGraphDocument:
    document = _sample_temporal_graph_document()
    return document.model_copy(
        update={
            "transition_events": [
                *document.transition_events,
                TransitionEvent(
                    event_id="obs_1",
                    video_id="video_001",
                    event_type="RELATION_OBSERVED",
                    frame_rank=12,
                    frame_index=12,
                    timestamp_sec=1.2,
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="standing near",
                    object_entity_id="hoop_1",
                ),
                TransitionEvent(
                    event_id="obs_2",
                    video_id="video_001",
                    event_type="RELATION_OBSERVED",
                    frame_rank=14,
                    frame_index=14,
                    timestamp_sec=1.4,
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="standing near",
                    object_entity_id="hoop_1",
                ),
                TransitionEvent(
                    event_id="obs_3",
                    video_id="video_001",
                    event_type="RELATION_OBSERVED",
                    frame_rank=16,
                    frame_index=16,
                    timestamp_sec=1.6,
                    subject_entity_id="player_1",
                    predicate="near",
                    raw_predicate="standing near",
                    object_entity_id="hoop_1",
                ),
            ],
        }
    )


def _sample_temporal_graph_with_nearby_relation_episode() -> TemporalGraphDocument:
    document = _sample_temporal_graph_document()
    return document.model_copy(
        update={
            "relation_episodes": [
                *document.relation_episodes,
                RelationEpisode(
                    relation_episode_id="rel_near_1",
                    video_id="video_001",
                    subject_entity_id="player_1",
                    predicate="looking_at",
                    raw_predicate="looking at",
                    object_entity_id="hoop_1",
                    start_frame=6,
                    end_frame=6,
                    start_ts=0.6,
                    end_ts=0.6,
                    support_frames=1,
                ),
            ]
        }
    )


def test_build_canonical_temporal_memory_graph_creates_core_schema() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_document(),
        source_graph_path="data/examples/video_001/temporal_graph.json",
    )

    assert isinstance(graph, CanonicalTemporalMemoryGraph)
    assert graph.schema_version == "canonical_temporal_memory_graph.v1"
    assert graph.video_id == "video_001"
    assert graph.source_graph_path == Path(
        "data/examples/video_001/temporal_graph.json"
    ).resolve()

    assert len(graph.scene_nodes) == 2
    assert len(graph.entity_nodes) == 2
    assert len(graph.stable_identity_nodes) == 2
    assert len(graph.state_value_nodes) == 2
    assert len(graph.interaction_type_nodes) == 1
    assert len(graph.episode_nodes) == 7
    assert len(graph.transition_event_nodes) == 1
    assert graph.node_count == 17
    assert graph.edge_count > 0

    edge_types = {edge.edge_type for edge in graph.edges}
    assert "HAS_EPISODE" in edge_types
    assert "IN_SCENE" in edge_types
    assert "HAS_MEMBER_ENTITY" in edge_types
    assert "HAS_STATE" in edge_types
    assert "HAS_INTERACTION_TYPE" in edge_types
    assert "INTERACTS_WITH" in edge_types
    assert "BELONGS_TO_SEGMENT" in edge_types
    assert "TRIGGERED_BY" in edge_types

    player_node = next(
        node
        for node in graph.entity_nodes
        if node.entity_id == "player_1"
    )
    assert player_node.stable_identity_id is not None
    assert player_node.stable_identity_canonical_label == "person"


def test_build_canonical_temporal_memory_graph_compacts_relation_observed_events() -> None:
    document = _sample_temporal_graph_with_observed_relation_events()
    default_graph = build_canonical_temporal_memory_graph(document)
    compact_graph = build_canonical_temporal_memory_graph(
        document,
        compact_relation_observed_events=True,
    )

    assert [
        node.event_id
        for node in default_graph.transition_event_nodes
        if node.event_type == "RELATION_OBSERVED"
    ] == ["obs_1", "obs_2", "obs_3"]
    assert not [
        node
        for node in compact_graph.transition_event_nodes
        if node.event_type == "RELATION_OBSERVED"
    ]
    assert [sample.event_id for sample in compact_graph.relation_observed_samples] == [
        "obs_1",
        "obs_2",
        "obs_3",
    ]
    assert compact_graph.node_count == default_graph.node_count - 3
    assert compact_graph.edge_count < default_graph.edge_count

    sample = compact_graph.relation_observed_samples[0]
    assert sample.sample_id == "event:obs_1"
    assert sample.scene_id == "scene_002"
    assert sample.subject_entity_id == "player_1"
    assert sample.predicate == "near"
    assert sample.object_entity_id == "hoop_1"
    assert sample.time_range.start_frame == 12


def test_build_canonical_temporal_memory_graph_compacts_derived_temporal_edges() -> None:
    document = _sample_temporal_graph_with_nearby_relation_episode()
    materialized_graph = build_canonical_temporal_memory_graph(
        document,
        compact_derived_temporal_edges=False,
    )
    compact_graph = build_canonical_temporal_memory_graph(
        document,
        compact_derived_temporal_edges=True,
    )

    materialized_edge_types = {edge.edge_type for edge in materialized_graph.edges}
    compact_edge_types = {edge.edge_type for edge in compact_graph.edges}

    assert "OVERLAPS_WITH" in materialized_edge_types
    assert "TEMPORALLY_NEAR" in materialized_edge_types
    assert "OVERLAPS_WITH" not in compact_edge_types
    assert "TEMPORALLY_NEAR" not in compact_edge_types
    assert "STARTS_AT" in compact_edge_types
    assert compact_graph.node_count == materialized_graph.node_count
    assert compact_graph.edge_count < materialized_graph.edge_count


def test_build_canonical_temporal_memory_graph_adds_temporal_chains() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    next_edges = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.edge_type == "NEXT_EPISODE"
    }

    assert (
        "episode:scene_segment:scene_001",
        "episode:scene_segment:scene_002",
    ) in next_edges
    assert (
        "episode:entity_segment:seg_player_1_a",
        "episode:entity_segment:seg_player_1_b",
    ) in next_edges
    assert (
        "episode:state:state_1",
        "episode:state:state_2",
    ) in next_edges
    assert "PREVIOUS_EPISODE" not in {edge.edge_type for edge in graph.edges}


def test_build_canonical_temporal_memory_graph_compacts_previous_episode_edges_by_default() -> None:
    default_graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    materialized_graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_document(),
        compact_previous_episode_edges=False,
    )

    default_edge_types = {edge.edge_type for edge in default_graph.edges}
    materialized_edge_types = {edge.edge_type for edge in materialized_graph.edges}

    assert "NEXT_EPISODE" in default_edge_types
    assert "PREVIOUS_EPISODE" not in default_edge_types
    assert "NEXT_EPISODE" in materialized_edge_types
    assert "PREVIOUS_EPISODE" in materialized_edge_types
    assert default_graph.node_count == materialized_graph.node_count
    assert default_graph.edge_count < materialized_graph.edge_count


def test_build_canonical_temporal_memory_graph_links_relation_and_event_context() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    relation_episode = next(
        node
        for node in graph.episode_nodes
        if node.node_id == "episode:relation:rel_1"
    )
    assert relation_episode.predicate == "near"
    assert relation_episode.subject_entity_id == "player_1"
    assert relation_episode.object_entity_id == "hoop_1"

    interaction_edge = next(
        edge
        for edge in graph.edges
        if edge.edge_type == "INTERACTS_WITH"
    )
    assert interaction_edge.source_node_id == "entity:player_1"
    assert interaction_edge.target_node_id == "entity:hoop_1"
    assert interaction_edge.attributes["episode_node_id"] == "episode:relation:rel_1"

    triggered_by_edges = [
        edge
        for edge in graph.edges
        if edge.edge_type == "TRIGGERED_BY"
    ]
    assert any(
        edge.source_node_id == "episode:state:state_2"
        and edge.target_node_id == "event:event_1"
        for edge in triggered_by_edges
    )


def test_build_canonical_temporal_memory_graph_adds_temporal_semantic_edges() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_document(),
        compact_derived_temporal_edges=False,
    )

    edges_by_type: dict[str, set[tuple[str, str]]] = {}
    for edge in graph.edges:
        edges_by_type.setdefault(edge.edge_type, set()).add(
            (edge.source_node_id, edge.target_node_id)
        )

    assert (
        "episode:state:state_2",
        "event:event_1",
    ) in edges_by_type["STARTS_AT"]
    assert (
        "episode:state:state_2",
        "episode:relation:rel_1",
    ) in edges_by_type["OVERLAPS_WITH"]

    document = _sample_temporal_graph_document()
    near_relation = RelationEpisode(
        relation_episode_id="rel_near",
        video_id="video_001",
        subject_entity_id="player_1",
        predicate="near",
        raw_predicate="near",
        object_entity_id="hoop_1",
        start_frame=5,
        end_frame=5,
        start_ts=0.5,
        end_ts=0.5,
        support_frames=1,
    )
    end_event = TransitionEvent(
        event_id="event_end",
        video_id="video_001",
        event_type="STATE_TRANSITION",
        frame_rank=14,
        frame_index=14,
        timestamp_sec=1.4,
        entity_id="player_1",
        attr_key="action",
        previous_state_value="shooting",
        current_state_value="running",
        source_segment_id="seg_player_1_b",
        source_episode_id="state_2",
    )
    graph_with_near_edges = build_canonical_temporal_memory_graph(
        document.model_copy(
            update={
                "relation_episodes": [
                    *document.relation_episodes,
                    near_relation,
                ],
                "transition_events": [
                    *document.transition_events,
                    end_event,
                ],
            }
        ),
        compact_derived_temporal_edges=False,
    )
    near_edges = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph_with_near_edges.edges
        if edge.edge_type == "TEMPORALLY_NEAR"
    }
    end_edges = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph_with_near_edges.edges
        if edge.edge_type == "ENDS_AT"
    }

    assert (
        "episode:state:state_1",
        "episode:relation:rel_near",
    ) in near_edges
    assert (
        "episode:state:state_2",
        "event:event_end",
    ) in end_edges


def test_build_canonical_temporal_memory_graph_adds_stable_identity_layer() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    stable_identity = next(
        node
        for node in graph.stable_identity_nodes
        if node.canonical_label == "person"
    )
    assert stable_identity.member_entity_ids == ["player_1"]
    assert stable_identity.identity_type == "human"

    member_edge = next(
        edge
        for edge in graph.edges
        if edge.edge_type == "HAS_MEMBER_ENTITY"
        and edge.source_node_id == stable_identity.node_id
    )
    assert member_edge.target_node_id == "entity:player_1"


def test_build_canonical_temporal_memory_graph_compacts_scene_label_entities() -> None:
    document = _sample_temporal_graph_document()
    window_entities = [
        Entity(
            entity_id=f"window_{index}",
            video_id="video_001",
            category="window",
            first_seen_frame=index,
            last_seen_frame=index,
            first_seen_ts=index / 10,
            last_seen_ts=index / 10,
            observation_count=1,
            node_role="dynamic",
            is_fragment=True,
        )
        for index in range(4)
    ]
    window_segments = [
        EntitySegment(
            segment_id=f"seg_window_{index}",
            video_id="video_001",
            entity_id=f"window_{index}",
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
        for index in range(4)
    ]
    document = document.model_copy(
        update={
            "entities": [*document.entities, *window_entities],
            "entity_segments": [*document.entity_segments, *window_segments],
        }
    )

    full_graph = build_canonical_temporal_memory_graph(document)
    compact_graph = build_canonical_temporal_memory_graph(
        document,
        compact_scene_label_entities=True,
        compact_scene_label_min_members=4,
    )

    assert len(compact_graph.entity_nodes) == len(full_graph.entity_nodes) - 3
    compact_entity = next(
        node
        for node in compact_graph.entity_nodes
        if node.attributes.get("compaction_kind") == "scene_label_entity_anchor"
    )
    assert compact_entity.label == "window"
    assert compact_entity.attributes["member_count"] == 4
    assert compact_entity.attributes["scene_id"] == "scene_001"

    compact_segment = next(
        node
        for node in compact_graph.episode_nodes
        if node.attributes.get("compaction_kind") == "scene_label_entity_segment_anchor"
    )
    assert compact_segment.primary_entity_id == compact_entity.entity_id
    assert compact_entity.entity_id in compact_segment.participant_entity_ids
    assert compact_segment.attributes["source_segment_count"] == 4


def test_build_canonical_temporal_memory_graph_preserves_salient_scene_labels() -> None:
    document = _sample_temporal_graph_document()
    couch_entities = [
        Entity(
            entity_id=f"couch_{index}",
            video_id="video_001",
            category="couch",
            first_seen_frame=index,
            last_seen_frame=index,
            first_seen_ts=index / 10,
            last_seen_ts=index / 10,
            observation_count=1,
            node_role="dynamic",
            is_fragment=True,
        )
        for index in range(4)
    ]
    couch_segments = [
        EntitySegment(
            segment_id=f"seg_couch_{index}",
            video_id="video_001",
            entity_id=f"couch_{index}",
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
        for index in range(4)
    ]
    document = document.model_copy(
        update={
            "entities": [*document.entities, *couch_entities],
            "entity_segments": [*document.entity_segments, *couch_segments],
        }
    )

    compact_graph = build_canonical_temporal_memory_graph(
        document,
        compact_scene_label_entities=True,
        compact_scene_label_min_members=4,
    )

    scene = next(node for node in compact_graph.scene_nodes if node.scene_id == "scene_001")
    assert "sofa" in scene.salient_labels
    assert "sofa" in scene.representative_static_labels
    assert {"sofa", "couch", "couches"}.issubset(set(scene.semantic_alias_terms))

    couch_nodes = [
        node
        for node in compact_graph.entity_nodes
        if node.label == "couch"
    ]
    assert len(couch_nodes) == 4
    assert not any(
        node.attributes.get("compaction_kind") == "scene_label_entity_anchor"
        for node in couch_nodes
    )
