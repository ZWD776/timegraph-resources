from __future__ import annotations

from pathlib import Path

from vlkg.core.temporal.improvement_metrics import build_improvement_maturity_metrics
from vlkg.core.models.temporal_graph import (
    BoundingBox,
    Entity,
    EntitySegment,
    Observation,
    RelationEpisode,
    SceneBaseEdge,
    SceneBaseNode,
    SceneDeltaEdge,
    SceneDeltaNode,
    SceneDeltaObservation,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.models.stable_identity import (
    StableIdentityDocument,
    StableIdentityMember,
    StableIdentityNode,
    StableIdentityTimeRange,
)
from vlkg.core.transformers import build_canonical_temporal_memory_graph


def _sample_document() -> TemporalGraphDocument:
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
    return TemporalGraphDocument(
        video_id="video_001",
        source_dir=Path("data/examples/video_001"),
        frame_count=4,
        observations=[
            Observation(
                observation_id="obs_1",
                video_id="video_001",
                entity_id="player_1",
                frame_name="frame_000000",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="frame_000000.png",
                object_index=1,
                bbox=bbox,
                label="man",
                score=0.99,
                state="holding basketball",
                state_values=["holding basketball"],
                scene_id="scene_001",
            ),
            Observation(
                observation_id="obs_2",
                video_id="video_001",
                entity_id="player_2",
                frame_name="frame_000003",
                frame_rank=3,
                frame_index=3,
                timestamp_sec=0.3,
                image_path="frame_000003.png",
                object_index=1,
                bbox=bbox,
                label="person",
                score=0.98,
                state="holding basketball",
                state_values=["holding basketball"],
                scene_id="scene_001",
            ),
        ],
        entities=[
            Entity(
                entity_id="player_1",
                video_id="video_001",
                category="man",
                first_seen_frame=0,
                last_seen_frame=1,
                first_seen_ts=0.0,
                last_seen_ts=0.1,
                observation_count=1,
                labels_seen=["man"],
                states_seen=["holding basketball"],
            ),
            Entity(
                entity_id="player_2",
                video_id="video_001",
                category="person",
                first_seen_frame=2,
                last_seen_frame=3,
                first_seen_ts=0.2,
                last_seen_ts=0.3,
                observation_count=1,
                labels_seen=["person"],
                states_seen=["holding basketball"],
            ),
        ],
        relation_episodes=[
            RelationEpisode(
                relation_episode_id="rel_1",
                video_id="video_001",
                subject_entity_id="player_1",
                predicate="PASSING_TO",
                raw_predicate="pass ball to",
                object_entity_id="player_2",
                start_frame=1,
                end_frame=2,
                start_ts=0.1,
                end_ts=0.2,
                support_frames=2,
            )
        ],
        state_episodes=[
            StateEpisode(
                state_episode_id="state_1",
                video_id="video_001",
                entity_id="player_1",
                attr_value="holding basketball",
                start_frame=0,
                end_frame=1,
                start_ts=0.0,
                end_ts=0.1,
                support_frames=2,
            )
        ],
        scene_segments=[
            SceneSegment(
                scene_id="scene_001",
                video_id="video_001",
                start_frame_rank=0,
                end_frame_rank=3,
                start_frame=0,
                end_frame=3,
                start_ts=0.0,
                end_ts=0.3,
                support_frames=4,
            )
        ],
        entity_segments=[
            EntitySegment(
                segment_id="seg_1",
                video_id="video_001",
                entity_id="player_1",
                scene_id="scene_001",
                start_frame_rank=0,
                end_frame_rank=1,
                start_frame=0,
                end_frame=1,
                start_ts=0.0,
                end_ts=0.1,
                support_frames=2,
            ),
            EntitySegment(
                segment_id="seg_2",
                video_id="video_001",
                entity_id="player_2",
                scene_id="scene_001",
                start_frame_rank=2,
                end_frame_rank=3,
                start_frame=2,
                end_frame=3,
                start_ts=0.2,
                end_ts=0.3,
                support_frames=2,
            ),
        ],
        scene_base_nodes=[
            SceneBaseNode(
                node_id="base_1",
                scene_id="scene_001",
                video_id="video_001",
                entity_id="court_1",
                label="court",
                first_frame=0,
                last_frame=3,
                support_frames=4,
                coverage_ratio=1.0,
            )
        ],
        scene_base_edges=[
            SceneBaseEdge(
                edge_id="base_edge_1",
                scene_id="scene_001",
                video_id="video_001",
                predicate="SCENE_CONTAINS",
                source_id="scene_001",
                target_id="base_1",
                entity_id="court_1",
                support_frames=4,
                coverage_ratio=1.0,
            )
        ],
        scene_delta_nodes=[
            SceneDeltaNode(
                node_id="delta_1",
                scene_id="scene_001",
                video_id="video_001",
                entity_id="player_1",
                label="man",
                start_frame=0,
                end_frame=1,
                start_ts=0.0,
                end_ts=0.1,
                support_frames=2,
                transition_event_count=1,
                relation_episode_count=1,
                state_episode_count=1,
            )
        ],
        scene_delta_observations=[
            SceneDeltaObservation(
                delta_observation_id="delta_obs_1",
                scene_id="scene_001",
                video_id="video_001",
                entity_id="player_1",
                scene_delta_node_id="delta_1",
                source_segment_id="seg_1",
                source_observation_id="obs_1",
                frame_name="frame_000000",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="frame_000000.png",
                bbox=bbox,
                label="man",
                trigger_event_ids=["event_appear"],
                trigger_event_types=["APPEAR"],
                marks_segment_start=True,
            )
        ],
        scene_delta_edges=[
            SceneDeltaEdge(
                edge_id="delta_edge_1",
                scene_id="scene_001",
                video_id="video_001",
                predicate="SCENE_UPDATES",
                source_id="base_1",
                target_id="delta_1",
                entity_id="player_1",
                support_frames=2,
                transition_event_count=1,
            )
        ],
        transition_events=[
            TransitionEvent(
                event_id="event_appear",
                video_id="video_001",
                event_type="APPEAR",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                entity_id="player_1",
                source_segment_id="seg_1",
            ),
            TransitionEvent(
                event_id="event_relation_start",
                video_id="video_001",
                event_type="RELATION_START",
                frame_rank=1,
                frame_index=1,
                timestamp_sec=0.1,
                subject_entity_id="player_1",
                predicate="PASSING_TO",
                object_entity_id="player_2",
                source_episode_id="rel_1",
            ),
        ],
    )


def _sample_stable_identity_document() -> StableIdentityDocument:
    return StableIdentityDocument(
        video_id="video_001",
        identities=[
            StableIdentityNode(
                stable_id="video_001:stable:human:001",
                video_id="video_001",
                identity_type="human",
                canonical_label="person",
                display_name="human_001",
                labels_seen=["man", "person"],
                member_entities=[
                    StableIdentityMember(
                        entity_id="player_1",
                        category="man",
                        first_seen_frame=0,
                        last_seen_frame=1,
                        first_seen_ts=0.0,
                        last_seen_ts=0.1,
                        observation_count=1,
                    ),
                    StableIdentityMember(
                        entity_id="player_2",
                        category="person",
                        first_seen_frame=2,
                        last_seen_frame=3,
                        first_seen_ts=0.2,
                        last_seen_ts=0.3,
                        observation_count=1,
                    ),
                ],
                time_range=StableIdentityTimeRange(
                    start_frame=0,
                    end_frame=3,
                    start_ts=0.0,
                    end_ts=0.3,
                    support_frames=2,
                ),
                observation_count=2,
            )
        ],
    )


def test_improvement_maturity_metrics_cover_original_optimization_goals() -> None:
    document = _sample_document()
    canonical_graph = build_canonical_temporal_memory_graph(document)

    metrics = build_improvement_maturity_metrics(
        document,
        stable_identity_document=_sample_stable_identity_document(),
        canonical_graph=canonical_graph,
    )

    assert metrics["time_axis"]["has_temporal_axis"] is True
    assert metrics["time_axis"]["entity_segment_coverage_ratio"] == 1.0
    assert metrics["time_axis"]["transition_event_count"] == 2
    assert metrics["base_delta"]["base_delta_available"] is True
    assert metrics["base_delta"]["delta_to_observation_ratio"] == 0.5
    assert metrics["entity_identity"]["entity_coverage_ratio"] == 1.0
    assert metrics["entity_identity"]["member_count_distribution"] == {"2": 1}
    assert metrics["entity_identity"]["singleton_identity_ratio"] == 0.0
    assert (
        metrics["entity_identity"]["quality_diagnostics"][
            "weak_human_singleton_count"
        ]
        == 0
    )
    assert metrics["canonical_graph"]["node_count"] == canonical_graph.node_count
    assert metrics["canonical_graph"]["edge_count"] == canonical_graph.edge_count


def test_improvement_metrics_include_identity_quality_diagnostics() -> None:
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
    document = TemporalGraphDocument(
        video_id="video_quality",
        source_dir=Path("data/examples/video_quality"),
        frame_count=6,
        observations=[
            Observation(
                observation_id="obs_player_1",
                video_id="video_quality",
                entity_id="player_1",
                frame_name="frame_000000",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="frame_000000.png",
                object_index=1,
                bbox=bbox,
                label="man",
                score=0.99,
                state_values=["red jersey"],
            ),
            Observation(
                observation_id="obs_player_2",
                video_id="video_quality",
                entity_id="player_2",
                frame_name="frame_000005",
                frame_rank=5,
                frame_index=5,
                timestamp_sec=0.5,
                image_path="frame_000005.png",
                object_index=1,
                bbox=bbox,
                label="person",
                score=0.98,
                state_values=["red jersey"],
            ),
            Observation(
                observation_id="obs_ball_1",
                video_id="video_quality",
                entity_id="ball_1",
                frame_name="frame_000003",
                frame_rank=3,
                frame_index=3,
                timestamp_sec=0.3,
                image_path="frame_000003.png",
                object_index=2,
                bbox=bbox,
                label="basketball",
                score=0.97,
            ),
        ],
        entities=[
            Entity(
                entity_id="player_1",
                video_id="video_quality",
                category="man",
                first_seen_frame=0,
                last_seen_frame=0,
                first_seen_ts=0.0,
                last_seen_ts=0.0,
                observation_count=1,
                labels_seen=["man"],
                states_seen=["red jersey"],
                canonical_embedding=[1.0, 0.0],
            ),
            Entity(
                entity_id="player_2",
                video_id="video_quality",
                category="person",
                first_seen_frame=5,
                last_seen_frame=5,
                first_seen_ts=0.5,
                last_seen_ts=0.5,
                observation_count=1,
                labels_seen=["person"],
                states_seen=["red jersey"],
                canonical_embedding=[0.99, 0.01],
            ),
            Entity(
                entity_id="ball_1",
                video_id="video_quality",
                category="basketball",
                first_seen_frame=3,
                last_seen_frame=3,
                first_seen_ts=0.3,
                last_seen_ts=0.3,
                observation_count=1,
                labels_seen=["basketball"],
                canonical_embedding=[0.0, 1.0],
            ),
        ],
        relation_episodes=[],
        state_episodes=[],
        entity_segments=[
            EntitySegment(
                segment_id="seg_player_1",
                video_id="video_quality",
                entity_id="player_1",
                start_frame_rank=0,
                end_frame_rank=0,
                start_frame=0,
                end_frame=0,
                start_ts=0.0,
                end_ts=0.0,
                support_frames=1,
            ),
            EntitySegment(
                segment_id="seg_player_2",
                video_id="video_quality",
                entity_id="player_2",
                start_frame_rank=5,
                end_frame_rank=5,
                start_frame=5,
                end_frame=5,
                start_ts=0.5,
                end_ts=0.5,
                support_frames=1,
            ),
            EntitySegment(
                segment_id="seg_ball_1",
                video_id="video_quality",
                entity_id="ball_1",
                start_frame_rank=3,
                end_frame_rank=3,
                start_frame=3,
                end_frame=3,
                start_ts=0.3,
                end_ts=0.3,
                support_frames=1,
            ),
        ],
    )
    stable_identity_document = StableIdentityDocument(
        video_id="video_quality",
        identities=[
            StableIdentityNode(
                stable_id="video_quality:stable:human:001",
                video_id="video_quality",
                identity_type="human",
                canonical_label="person",
                display_name="human_001",
                member_entities=[
                    StableIdentityMember(
                        entity_id="player_1",
                        category="man",
                        first_seen_frame=0,
                        last_seen_frame=0,
                        first_seen_ts=0.0,
                        last_seen_ts=0.0,
                        observation_count=1,
                    )
                ],
                time_range=StableIdentityTimeRange(
                    start_frame=0,
                    end_frame=0,
                    start_ts=0.0,
                    end_ts=0.0,
                    support_frames=1,
                ),
                observation_count=1,
            ),
            StableIdentityNode(
                stable_id="video_quality:stable:human:002",
                video_id="video_quality",
                identity_type="human",
                canonical_label="person",
                display_name="human_002",
                member_entities=[
                    StableIdentityMember(
                        entity_id="player_2",
                        category="person",
                        first_seen_frame=5,
                        last_seen_frame=5,
                        first_seen_ts=0.5,
                        last_seen_ts=0.5,
                        observation_count=1,
                    )
                ],
                time_range=StableIdentityTimeRange(
                    start_frame=5,
                    end_frame=5,
                    start_ts=0.5,
                    end_ts=0.5,
                    support_frames=1,
                ),
                observation_count=1,
            ),
            StableIdentityNode(
                stable_id="video_quality:stable:ball:001",
                video_id="video_quality",
                identity_type="ball",
                canonical_label="basketball",
                display_name="ball_001",
                member_entities=[
                    StableIdentityMember(
                        entity_id="ball_1",
                        category="basketball",
                        first_seen_frame=3,
                        last_seen_frame=3,
                        first_seen_ts=0.3,
                        last_seen_ts=0.3,
                        observation_count=1,
                    )
                ],
                time_range=StableIdentityTimeRange(
                    start_frame=3,
                    end_frame=3,
                    start_ts=0.3,
                    end_ts=0.3,
                    support_frames=1,
                ),
                observation_count=1,
            ),
        ],
    )

    metrics = build_improvement_maturity_metrics(
        document,
        stable_identity_document=stable_identity_document,
    )
    diagnostics = metrics["entity_identity"]["quality_diagnostics"]

    assert diagnostics["weak_human_singleton_count"] == 2
    assert diagnostics["sparse_key_object_count"] == 1
    assert diagnostics["merge_suggestion_count"] == 1
    assert diagnostics["merge_suggestion_verdict_counts"] == {
        "review_candidate": 1,
    }
    assert diagnostics["uncovered_merge_suggestion_verdict_counts"] == {
        "review_candidate": 1,
    }
    assert diagnostics["already_covered_merge_suggestion_verdict_counts"] == {}
    assert diagnostics["review_merge_suggestion_count"] == 1
    assert diagnostics["already_covered_review_merge_suggestion_count"] == 0
    assert diagnostics["top_review_merge_suggestions"][0]["pair"] == [
        "player_1",
        "player_2",
    ]
    assert diagnostics["top_blocked_overlap_merge_suggestions"] == []
    assert diagnostics["top_weak_similarity_merge_suggestions"] == []
    assert diagnostics["weak_candidate_type_counts"] == {
        "ball": 1,
        "human": 2,
    }


def test_improvement_maturity_metrics_do_not_mutate_document() -> None:
    document = _sample_document()
    before = document.model_dump(mode="json")

    build_improvement_maturity_metrics(
        document,
        stable_identity_document=_sample_stable_identity_document(),
    )

    assert document.model_dump(mode="json") == before
