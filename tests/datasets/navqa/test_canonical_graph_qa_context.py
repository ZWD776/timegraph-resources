from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
import uuid

import numpy as np
import pytest

from vlkg.datasets.navqa.qa.canonical_graph_qa_context import (
    answer_canonical_graph_query,
    build_canonical_llm_ready_bundle,
    retrieve_canonical_graph_context,
    _time_range_text,
)
from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalSceneNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.core.config.config import GraphQAConfig
from vlkg.datasets.navqa.qa.graph_qa_answering import (
    OpenAICompatibleGraphQAResponder,
    _build_llm_evidence_selection_candidates,
    _bundle_with_answer_candidate_extraction,
    _bundle_with_llm_evidence_selection,
    _normalize_selected_evidence_lines,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import (
    BoundingBox,
    Entity,
    EntitySegment,
    Observation,
    RelationEpisode,
    SceneDeltaNode,
    SceneDeltaObservation,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.transformers import build_canonical_temporal_memory_graph
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import (
    VisualFaissConfig,
    VisualFaissGrounder,
    _concise_object,
    _without_schema_match_terms,
    expand_visual_faiss_graph_evidence,
    fuse_visual_faiss_ger_evidence,
    fuse_visual_faiss_unified_candidates,
    fuse_visual_faiss_role_aware_evidence,
    fuse_visual_faiss_evidence,
    _compressed_visual_faiss_rows,
    _visual_query_phrases,
    _visual_query_phrase_records,
)
from vlkg.datasets.navqa.retrieval.visual_entity_candidate_packs import (
    fuse_visual_entity_candidate_packs,
    visual_entity_pack_query_phrase_records,
)


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


def test_live_graph_qa_prompt_requires_exact_binary_answer_text() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig(enabled=True))
    messages = responder._build_messages(
        question=(
            "Did you see a red fire hydrant?\n\n"
            "NaVQA evaluation adapter: for binary yes/no questions, the final "
            "answer_text must be exactly 'yes' or exactly 'no'."
        ),
        llm_ready_bundle={
            "answer_brief": "red fire hydrant observed",
            "evidence_lines": ["rank=1 frame=10 object=red fire hydrant"],
            "focus_frame_lines": ["rank=1 frame=10"],
            "question_ir": {"answer_type": "binary", "rendering": "yes_no"},
        },
        graph_result={
            "query_type": "canonical_temporal",
            "llm_ready_bundle": {
                "answer_brief": "red fire hydrant observed",
                "evidence_lines": ["rank=1 frame=10 object=red fire hydrant"],
                "focus_frame_lines": ["rank=1 frame=10"],
                "question_ir": {"answer_type": "binary", "rendering": "yes_no"},
            },
            "retrieval_bundle": {"focus_frames": [{"frame_index": 10}]},
            "matches": [],
        },
    )

    system_prompt = str(messages[0]["content"])
    user_payload = json.loads(str(messages[1]["content"]))

    assert "answer_text must be exactly 'yes' or exactly 'no'" in system_prompt
    assert user_payload["instructions"][
        "binary_yes_no_answer_text_must_be_exact_label_only"
    ] is True
    assert "answer_text must be exactly 'yes' or exactly 'no'" in (
        user_payload["answering_policy"]["question_ir_policy"][4]
    )


def _sample_temporal_graph_with_multi_member_identity() -> TemporalGraphDocument:
    video_id = "video_identity"
    scene_id = "scene_entry"
    return TemporalGraphDocument(
        video_id=video_id,
        source_dir=Path("data/examples/video_identity"),
        frame_count=20,
        observations=[
            Observation(
                observation_id="obs_person_a",
                video_id=video_id,
                entity_id="person_a",
                frame_name="frame_000000",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="synthetic.png",
                object_index=0,
                bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=120.0),
                label="man",
                score=0.99,
                state_values=["blue coat"],
            ),
            Observation(
                observation_id="obs_person_b",
                video_id=video_id,
                entity_id="person_b",
                frame_name="frame_000009",
                frame_rank=9,
                frame_index=9,
                timestamp_sec=0.9,
                image_path="synthetic.png",
                object_index=0,
                bbox=BoundingBox(x1=12.0, y1=10.0, x2=52.0, y2=120.0),
                label="person",
                score=0.99,
                state_values=["blue coat"],
            ),
        ],
        entities=[
            Entity(
                entity_id="person_a",
                video_id=video_id,
                category="man",
                first_seen_frame=0,
                last_seen_frame=2,
                first_seen_ts=0.0,
                last_seen_ts=0.2,
                observation_count=3,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=["blue coat"],
                node_role="dynamic",
            ),
            Entity(
                entity_id="person_b",
                video_id=video_id,
                category="person",
                first_seen_frame=9,
                last_seen_frame=12,
                first_seen_ts=0.9,
                last_seen_ts=1.2,
                observation_count=4,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["person"],
                states_seen=["blue coat"],
                node_role="dynamic",
            ),
        ],
        relation_episodes=[],
        state_episodes=[],
        scene_segments=[
            SceneSegment(
                scene_id=scene_id,
                video_id=video_id,
                scene_index=1,
                scene_count=1,
                start_frame_rank=0,
                end_frame_rank=19,
                start_frame=0,
                end_frame=19,
                start_ts=0.0,
                end_ts=1.9,
                support_frames=20,
                representative_dynamic_labels=["person"],
                scene_kind_hint="entry",
            )
        ],
        entity_segments=[
            EntitySegment(
                segment_id="seg_person_a",
                video_id=video_id,
                entity_id="person_a",
                scene_id=scene_id,
                start_frame_rank=0,
                end_frame_rank=2,
                start_frame=0,
                end_frame=2,
                start_ts=0.0,
                end_ts=0.2,
                support_frames=3,
            ),
            EntitySegment(
                segment_id="seg_person_b",
                video_id=video_id,
                entity_id="person_b",
                scene_id=scene_id,
                start_frame_rank=9,
                end_frame_rank=12,
                start_frame=9,
                end_frame=12,
                start_ts=0.9,
                end_ts=1.2,
                support_frames=4,
            ),
        ],
        transition_events=[],
    )


def _sample_temporal_graph_with_observed_relation_burst() -> TemporalGraphDocument:
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
            ]
        }
    )


def _sample_temporal_graph_with_scene_transition_chain() -> TemporalGraphDocument:
    video_id = "video_scene_chain"
    return TemporalGraphDocument(
        video_id=video_id,
        source_dir=Path("data/examples/video_scene_chain"),
        frame_count=30,
        observations=[
            Observation(
                observation_id="obs_tree_1",
                video_id=video_id,
                entity_id="tree_1",
                frame_name="frame_000000.jpg",
                frame_rank=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="frames/frame_000000.jpg",
                object_index=0,
                bbox=BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.6),
                label="tree",
                score=0.9,
                node_role="static",
                is_static=True,
                scene_id="scene_outside",
            ),
            Observation(
                observation_id="obs_door_1",
                video_id=video_id,
                entity_id="door_1",
                frame_name="frame_000010.jpg",
                frame_rank=10,
                frame_index=10,
                timestamp_sec=1.0,
                image_path="frames/frame_000010.jpg",
                object_index=0,
                bbox=BoundingBox(x1=0.3, y1=0.0, x2=0.7, y2=0.8),
                label="door",
                score=0.9,
                node_role="static",
                is_static=True,
                scene_id="scene_doorway",
            ),
            Observation(
                observation_id="obs_floor_1",
                video_id=video_id,
                entity_id="floor_1",
                frame_name="frame_000020.jpg",
                frame_rank=20,
                frame_index=20,
                timestamp_sec=2.0,
                image_path="frames/frame_000020.jpg",
                object_index=0,
                bbox=BoundingBox(x1=0.0, y1=0.5, x2=1.0, y2=1.0),
                label="floor",
                score=0.9,
                node_role="static",
                is_static=True,
                scene_id="scene_inside",
            ),
        ],
        entities=[
            Entity(
                entity_id="tree_1",
                video_id=video_id,
                category="tree",
                first_seen_frame=0,
                last_seen_frame=9,
                first_seen_ts=0.0,
                last_seen_ts=0.9,
                observation_count=1,
                node_role="static",
                is_static=True,
            ),
            Entity(
                entity_id="door_1",
                video_id=video_id,
                category="door",
                first_seen_frame=10,
                last_seen_frame=19,
                first_seen_ts=1.0,
                last_seen_ts=1.9,
                observation_count=1,
                node_role="static",
                is_static=True,
            ),
            Entity(
                entity_id="floor_1",
                video_id=video_id,
                category="floor",
                first_seen_frame=20,
                last_seen_frame=29,
                first_seen_ts=2.0,
                last_seen_ts=2.9,
                observation_count=1,
                node_role="static",
                is_static=True,
            ),
        ],
        relation_episodes=[],
        state_episodes=[],
        scene_segments=[
            SceneSegment(
                scene_id="scene_outside",
                video_id=video_id,
                scene_index=1,
                scene_count=3,
                start_frame_rank=0,
                end_frame_rank=9,
                start_frame=0,
                end_frame=9,
                start_ts=0.0,
                end_ts=0.9,
                support_frames=10,
                next_scene_id="scene_doorway",
                representative_static_labels=["tree"],
                scene_kind_hint="tree",
            ),
            SceneSegment(
                scene_id="scene_doorway",
                video_id=video_id,
                scene_index=2,
                scene_count=3,
                start_frame_rank=10,
                end_frame_rank=19,
                start_frame=10,
                end_frame=19,
                start_ts=1.0,
                end_ts=1.9,
                support_frames=10,
                previous_scene_id="scene_outside",
                next_scene_id="scene_inside",
                representative_static_labels=["door", "building"],
                scene_kind_hint="door / building",
            ),
            SceneSegment(
                scene_id="scene_inside",
                video_id=video_id,
                scene_index=3,
                scene_count=3,
                start_frame_rank=20,
                end_frame_rank=29,
                start_frame=20,
                end_frame=29,
                start_ts=2.0,
                end_ts=2.9,
                support_frames=10,
                previous_scene_id="scene_doorway",
                representative_static_labels=["floor"],
                scene_kind_hint="floor",
            ),
        ],
        entity_segments=[
            EntitySegment(
                segment_id="seg_tree_1",
                video_id=video_id,
                entity_id="tree_1",
                scene_id="scene_outside",
                segment_index=1,
                segment_count=1,
                start_frame_rank=0,
                end_frame_rank=9,
                start_frame=0,
                end_frame=9,
                start_ts=0.0,
                end_ts=0.9,
                support_frames=1,
                is_static=True,
            ),
            EntitySegment(
                segment_id="seg_door_1",
                video_id=video_id,
                entity_id="door_1",
                scene_id="scene_doorway",
                segment_index=1,
                segment_count=1,
                start_frame_rank=10,
                end_frame_rank=19,
                start_frame=10,
                end_frame=19,
                start_ts=1.0,
                end_ts=1.9,
                support_frames=1,
                is_static=True,
            ),
            EntitySegment(
                segment_id="seg_floor_1",
                video_id=video_id,
                entity_id="floor_1",
                scene_id="scene_inside",
                segment_index=1,
                segment_count=1,
                start_frame_rank=20,
                end_frame_rank=29,
                start_frame=20,
                end_frame=29,
                start_ts=2.0,
                end_ts=2.9,
                support_frames=1,
                is_static=True,
            ),
        ],
        transition_events=[],
    )


def test_retrieve_canonical_graph_context_builds_llm_bundle() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="near hoop", limit=6),
        retrieval_mode="temporal",
    )

    assert graph_result["query_type"] == "canonical_temporal"
    assert graph_result["matches"]
    assert graph_result["retrieval_bundle"]["focus_frames"]
    assert graph_result["retrieval_bundle"]["scene_summaries"]
    assert graph_result["llm_ready_bundle"]["answer_brief"]
    assert graph_result["llm_ready_bundle"]["evidence_lines"]
    assert graph_result["llm_ready_bundle"]["focus_frame_lines"]
    assert graph_result["llm_ready_bundle"]["scene_lines"]
    assert "canonical_temporal" in graph_result["llm_ready_bundle"]["llm_context_text"]


def test_vector_profile_exposes_two_supplements_after_lexical_top_five() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    matches = [
        {
            "source_record_id": f"lexical:{index}",
            "source_kind": "entity",
            "summary": f"lexical evidence {index}",
            "score": 100.0 - index,
            "metadata": {},
        }
        for index in range(1, 6)
    ]
    matches.extend(
        {
            "source_record_id": f"vector:{index}",
            "source_kind": "entity_attribute",
            "summary": f"vector evidence {index}",
            "score": 50.0 - index,
            "metadata": {
                "retrieval_channel": "canonical_vector_supplement",
            },
        }
        for index in range(1, 3)
    )
    retrieval_bundle = {
        "primary_item_ids": [item["source_record_id"] for item in matches],
        "raw_payload": {},
    }

    vector_bundle = build_canonical_llm_ready_bundle(
        graph=graph,
        query=MemoryQuery(
            query_text="semantic question",
            metadata={"graph_retrieval_profile": "canonical_vector_hybrid_v1"},
        ),
        retrieval_mode="temporal",
        filters={},
        summary="test",
        matches=matches,
        retrieval_bundle=retrieval_bundle,
    )
    navqa_bundle = build_canonical_llm_ready_bundle(
        graph=graph,
        query=MemoryQuery(
            query_text="semantic question",
            metadata={"graph_retrieval_profile": "navqa_temporal"},
        ),
        retrieval_mode="temporal",
        filters={},
        summary="test",
        matches=matches,
        retrieval_bundle=retrieval_bundle,
    )

    assert len(vector_bundle["evidence_lines"]) == 7
    assert "vector evidence 1" in vector_bundle["evidence_lines"][5]
    assert "vector evidence 2" in vector_bundle["evidence_lines"][6]
    assert len(navqa_bundle["evidence_lines"]) == 6
    assert "vector evidence 2" not in "\n".join(navqa_bundle["evidence_lines"])

    seeded_bundle = build_canonical_llm_ready_bundle(
        graph=graph,
        query=MemoryQuery(
            query_text="semantic question",
            metadata={"graph_retrieval_profile": "canonical_vector_seeded_v2"},
        ),
        retrieval_mode="temporal",
        filters={},
        summary="test",
        matches=matches,
        retrieval_bundle=retrieval_bundle,
    )
    assert len(seeded_bundle["evidence_lines"]) == 6
    assert "vector evidence 1" in seeded_bundle["evidence_lines"][5]
    assert "vector evidence 2" not in "\n".join(seeded_bundle["evidence_lines"])


def test_retrieve_canonical_graph_context_exposes_temporal_chain_lines() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="How long was it inside after entering?", limit=8),
        retrieval_mode="temporal",
    )

    temporal_chains = graph_result["retrieval_bundle"]["temporal_chains"]
    assert temporal_chains
    assert temporal_chains[0]["scene_ids"] == [
        "scene_outside",
        "scene_doorway",
        "scene_inside",
    ]
    chain_lines = graph_result["llm_ready_bundle"]["temporal_chain_lines"]
    assert any("scene_outside" in line and "scene_inside" in line for line in chain_lines)
    assert "Temporal chains:" in graph_result["llm_ready_bundle"]["llm_context_text"]
    structured = graph_result["llm_ready_bundle"]["structured_graph_context"]
    assert structured["temporal_chains"][0]["scene_ids"][-1] == "scene_inside"


def test_question_ir_execution_exposes_semantic_event_candidates() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="When did you first enter the building?",
            answer_type="time",
            metadata={
                "question_ir": {
                    "answer_type": "time",
                    "operation": "onset",
                    "target_state": "enter building",
                    "target_entities": ["building"],
                    "temporal_relation": "first",
                    "aggregation": "single_best",
                    "rendering": "minutes_ago",
                    "evidence_requirements": ["timestamp of first entry event"],
                    "graph_information_needs": ["ego trajectory with building entry events"],
                }
            },
            limit=8,
        ),
        retrieval_mode="temporal",
    )

    execution = graph_result["llm_ready_bundle"]["question_ir_execution"]
    candidates = execution["semantic_event_candidates"]
    assert execution["available"] is True
    assert candidates
    assert candidates[0]["event_type"] == "enter"
    assert candidates[0]["boundary_frame"] == 20
    assert "semantic_event_candidate" in "\n".join(execution["execution_lines"])


def test_llm_bundle_exposes_evidence_aggregation_groups() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="What action was the man doing near the hoop?",
            answer_type="text",
            metadata={
                "question_ir": {
                    "answer_type": "text",
                    "operation": "attribute_query",
                    "target_entities": ["man"],
                    "attribute_dimension": "action",
                    "aggregation": "multiple",
                    "rendering": "short_phrase",
                    "hard_constraints": [
                        {
                            "kind": "spatial",
                            "text": "near the hoop",
                            "role": "spatial_condition",
                            "target_entities": ["hoop"],
                            "must_bind_to": ["same_scene"],
                        }
                    ],
                    "evidence_requirements": [
                        "state or relation evidence for the man near the hoop"
                    ],
                    "graph_information_needs": ["entity state and relation episodes"],
                }
            },
            limit=8,
        ),
        retrieval_mode="temporal",
    )

    bundle = graph_result["llm_ready_bundle"]
    aggregation = bundle["evidence_aggregation"]
    lines = bundle["evidence_aggregation_lines"]

    assert aggregation["available"] is True
    assert aggregation["question_ir_summary"]["operation"] == "attribute_query"
    assert aggregation["question_ir_summary"]["hard_constraints"][0]["text"] == "near the hoop"
    assert aggregation["hard_constraint_terms"]
    assert lines
    assert "ranks=" in lines[0]
    assert any(
        group["group_type"] == "entity" and "player_1" in group["group_id"]
        for group in aggregation["groups"]
    )
    assert any(
        "dribbling" in " ".join(group.get("states", []))
        or "shooting" in " ".join(group.get("states", []))
        for group in aggregation["groups"]
    )
    assert aggregation["contract"] == "canonical_evidence_aggregation_v2"
    assert aggregation["candidate_facts"]
    assert any(
        fact.get("fact_type") in {"state", "relation"}
        and fact.get("binding_strength")
        in {
            "direct_target_question_factor",
            "direct_target_requirement",
            "direct_target",
        }
        for fact in aggregation["candidate_facts"]
    )
    assert any(
        group.get("hard_constraint_hits")
        for group in aggregation["groups"]
    )
    assert any("hard_constraint_hits=" in line for line in lines)
    assert any(line.startswith("candidate_fact=") for line in lines)
    assert "Evidence aggregation:" in bundle["llm_context_text"]
    assert bundle["structured_graph_context"]["evidence_aggregation"]["available"] is True


def test_live_graph_qa_prompt_prioritizes_temporal_chains() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="How long was it inside after entering?", limit=8),
        retrieval_mode="temporal",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question="How long was it inside after entering?",
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    system_prompt = messages[0]["content"]
    user_payload = messages[1]["content"]
    assert "canonical temporal memory graph" in system_prompt
    assert "first-person robot navigation video" in system_prompt
    assert "'I', 'you', and 'we' in a question refer to the robot/ego camera" in system_prompt
    assert "viewpoint background only disambiguates wording and is not observation evidence" in system_prompt
    assert "temporal_chain_lines" in user_payload
    assert "evidence_aggregation_lines" in user_payload
    assert "candidate_facts" in user_payload
    assert "structured_graph_context.temporal_chains" in system_prompt
    assert "Do not answer 'insufficient evidence'" in system_prompt
    assert "main time axis" in system_prompt
    assert "Question metadata and answer-format hints are not observation evidence" in system_prompt
    assert "bind attributes, states, and descriptions only to the entity" in system_prompt
    assert "time_adaptation" in user_payload
    assert "dataset_context" in user_payload
    assert "context_is_not_evidence" in user_payload
    assert "used_time_anchor_ids" in system_prompt
    assert "Do not manually recalculate" in system_prompt
    assert "direct state or attribute evidence bound to the queried entity" in user_payload
    assert "generic spatial or co-occurrence relations, used only as weak context" in user_payload
    assert "do not blindly use rank 1" in user_payload
    assert "graph_summary.evidence_aggregation_lines" in system_prompt
    assert "graph_summary.candidate_facts" in system_prompt
    assert "question_metadata_and_format_hints_are_not_visual_evidence" in user_payload


def test_evidence_aggregation_folds_stable_identity_members() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_multi_member_identity()
    )

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="Did you see a person holding a door?",
            answer_type="binary",
            metadata={
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["person"],
                    "target_state": "holding door",
                    "aggregation": "yes_no",
                    "rendering": "yes_no",
                    "evidence_requirements": [
                        "visual evidence of a person holding a door"
                    ],
                    "graph_information_needs": ["object detections with actions"],
                }
            },
            limit=8,
        ),
        retrieval_mode="temporal",
    )

    aggregation = graph_result["llm_ready_bundle"]["evidence_aggregation"]
    identity_groups = [
        group
        for group in aggregation["groups"]
        if group.get("group_type") == "identity"
    ]

    assert len(identity_groups) <= 1
    if identity_groups:
        assert len(identity_groups[0]["entity_ids"]) > 1
        assert not any(
            group.get("group_type") == "entity"
            and set(group.get("entity_ids", [])) >= set(identity_groups[0]["entity_ids"])
            for group in aggregation["groups"]
        )


def test_live_graph_qa_prompt_includes_question_ir_contract() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="When did you first enter the building?",
            answer_type="time",
            metadata={
                "question_ir": {
                    "answer_type": "time",
                    "operation": "onset",
                    "target_state": "enter building",
                    "target_entities": ["building"],
                    "aggregation": "single_best",
                    "rendering": "minutes_ago",
                    "evidence_requirements": ["timestamp of first entry event"],
                }
            },
            limit=8,
        ),
        retrieval_mode="temporal",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question="When did you first enter the building?",
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    system_prompt = messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])
    assert "use it as the answer contract" in system_prompt
    assert user_payload["question_ir"]["operation"] == "onset"
    assert user_payload["question_ir_execution"]["semantic_event_candidates"]
    assert "question_ir_policy" in user_payload["answering_policy"]


def test_live_graph_qa_prompt_compacts_faiss_payloads() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "\n".join(
            [
                "Video: toy",
                "Frame FAISS evidence:",
                "- SHORT_FRAME_FAISS_LINE",
                "FULL_CONTEXT_PAYLOAD_SHOULD_NOT_APPEAR",
            ]
        ),
        "evidence_lines": ["rank=1 kind=episode score=8 :: graph evidence"],
        "focus_frame_lines": ["rank=1 frame=10 score=7 reasons=graph"],
        "scene_lines": [],
        "temporal_chain_lines": ["chain keeps normal prompt context"],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "visual_faiss_grounding": {
            "groups": [{"raw": "FULL_VISUAL_PAYLOAD_SHOULD_NOT_APPEAR"}]
        },
        "visual_faiss_payloads": [
            {"raw": "FULL_VISUAL_PAYLOAD_LIST_SHOULD_NOT_APPEAR"}
        ],
        "visual_faiss_evidence_lines": ["SHORT_VISUAL_FAISS_LINE"],
        "frame_faiss_grounding": {
            "groups": [{"raw": "FULL_FRAME_PAYLOAD_SHOULD_NOT_APPEAR"}]
        },
        "frame_faiss_payloads": [
            {"raw": "FULL_FRAME_PAYLOAD_LIST_SHOULD_NOT_APPEAR"}
        ],
        "frame_faiss_evidence_lines": ["SHORT_FRAME_FAISS_LINE"],
        "merged_frame_faiss_grounding": {
            "groups": [{"raw": "FULL_MERGED_PAYLOAD_SHOULD_NOT_APPEAR"}]
        },
        "merged_frame_faiss_payloads": [
            {"raw": "FULL_MERGED_PAYLOAD_LIST_SHOULD_NOT_APPEAR"}
        ],
        "merged_frame_faiss_evidence_lines": ["SHORT_MERGED_FAISS_LINE"],
        "structured_graph_context": {
            "temporal_chains": [{"summary": "NORMAL_TEMPORAL_CHAIN_SHOULD_APPEAR"}],
            "entity_aliases": {
                "entity_1": {"display_text": "NORMAL_ALIAS_SHOULD_APPEAR"}
            },
            "visual_faiss_grounding": {
                "raw": "STRUCTURED_VISUAL_PAYLOAD_SHOULD_NOT_APPEAR"
            },
            "frame_faiss_grounding": {
                "raw": "STRUCTURED_FRAME_PAYLOAD_SHOULD_NOT_APPEAR"
            },
            "merged_frame_faiss_grounding": {
                "raw": "STRUCTURED_MERGED_PAYLOAD_SHOULD_NOT_APPEAR"
            },
            "source_merged_frame_faiss": {
                "raw": "STRUCTURED_SOURCE_MERGED_SHOULD_NOT_APPEAR"
            },
        },
    }

    messages = responder._build_messages(
        question="What did the graph and visual recall show?",
        llm_ready_bundle=bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )

    prompt_text = messages[1]["content"]
    user_payload = json.loads(prompt_text)

    for forbidden in [
        "FULL_CONTEXT_PAYLOAD_SHOULD_NOT_APPEAR",
        "FULL_VISUAL_PAYLOAD_SHOULD_NOT_APPEAR",
        "FULL_VISUAL_PAYLOAD_LIST_SHOULD_NOT_APPEAR",
        "FULL_FRAME_PAYLOAD_SHOULD_NOT_APPEAR",
        "FULL_FRAME_PAYLOAD_LIST_SHOULD_NOT_APPEAR",
        "FULL_MERGED_PAYLOAD_SHOULD_NOT_APPEAR",
        "FULL_MERGED_PAYLOAD_LIST_SHOULD_NOT_APPEAR",
        "STRUCTURED_VISUAL_PAYLOAD_SHOULD_NOT_APPEAR",
        "STRUCTURED_FRAME_PAYLOAD_SHOULD_NOT_APPEAR",
        "STRUCTURED_MERGED_PAYLOAD_SHOULD_NOT_APPEAR",
        "STRUCTURED_SOURCE_MERGED_SHOULD_NOT_APPEAR",
    ]:
        assert forbidden not in prompt_text

    assert user_payload["graph_summary"]["visual_faiss_evidence_lines"] == [
        "SHORT_VISUAL_FAISS_LINE"
    ]
    assert user_payload["graph_summary"]["frame_faiss_evidence_lines"] == [
        "SHORT_FRAME_FAISS_LINE"
    ]
    assert user_payload["graph_summary"]["merged_frame_faiss_evidence_lines"] == [
        "SHORT_MERGED_FAISS_LINE"
    ]
    assert "visual_faiss_grounding" not in user_payload["graph_summary"]
    assert (
        user_payload["structured_graph_context"]["temporal_chains"][0]["summary"]
        == "NORMAL_TEMPORAL_CHAIN_SHOULD_APPEAR"
    )
    assert (
        user_payload["structured_graph_context"]["entity_aliases"]["entity_1"][
            "display_text"
        ]
        == "NORMAL_ALIAS_SHOULD_APPEAR"
    )
    assert "SHORT_FRAME_FAISS_LINE" in user_payload["graph_context"]


def test_live_graph_qa_prompt_stays_dataset_agnostic() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="How long was it inside after entering?", limit=8),
        retrieval_mode="temporal",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question=(
            "How long was it inside after entering?\n\n"
            "Question time metadata, not answer evidence:\n"
            "current_time_unix=1000\n\n"
            "Official answer format hint: answer primarily as a duration in minutes."
        ),
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    prompt_text = "\n".join(message["content"] for message in messages)
    for forbidden in [
        "NaVQA",
        "CODa",
        "seq0",
        "Q17",
        "Q20",
        "red couches",
        "police call",
    ]:
        assert forbidden not in prompt_text
    assert "Question time metadata, not answer evidence" in prompt_text
    assert "metadata_policy" in prompt_text
    assert "Answer-format hints can shape wording only" in prompt_text


def test_live_graph_qa_non_position_prompt_excludes_archived_position_policy() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_scene_transition_chain()
    )
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="How long was it inside after entering?", limit=8),
        retrieval_mode="temporal",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question="How long was it inside after entering?",
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    prompt_text = "\n".join(message["content"] for message in messages)
    for forbidden in [
        "for_position_place_questions_return_event_time_not_coordinates",
        "position_rendering_policy",
        "ego-pose adapter",
        "x/y/z",
        "GPS",
        "object 3D coordinates",
        "inventing coordinates for position questions",
        "graph_summary_positions",
    ]:
        assert forbidden not in prompt_text
    assert "used_evidence_ranks_must_reference_graph_summary_indices" in prompt_text


def test_live_graph_qa_non_position_answer_type_overrides_where_query_plan() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="Where is the basketball hoop?", answer_type="text", limit=6),
        retrieval_mode="temporal",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question="Where is the basketball hoop?",
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    query_plan = graph_result["retrieval_bundle"]["raw_payload"]["query_plan"]
    assert query_plan["question_compiler"]["answer_type"] == "text"
    assert query_plan["query_family"] != "position"
    user_payload = json.loads(messages[1]["content"])
    assert "position_rendering_policy" not in user_payload["answering_policy"]
    assert (
        "for_position_place_questions_return_event_time_not_coordinates"
        not in user_payload["instructions"]
    )


def test_live_graph_qa_position_prompt_keeps_archived_position_policy() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="Where did you see the basketball hoop?", limit=6),
        retrieval_mode="temporal",
    )
    graph_result["retrieval_bundle"]["raw_payload"]["query_plan"] = {
        "query_family": "position",
        "question_compiler": {
            "answer_type": "position",
            "ranking_policy": ["target_event_time_then_pose_render"],
        },
    }
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    messages = responder._build_messages(
        question="Where did you see the basketball hoop?",
        llm_ready_bundle=graph_result["llm_ready_bundle"],
        graph_result=graph_result,
    )

    user_payload = json.loads(messages[1]["content"])
    assert user_payload["instructions"][
        "for_position_place_questions_return_event_time_not_coordinates"
    ] is True
    assert "position_rendering_policy" in user_payload["answering_policy"]
    assert any(
        "Do not invent x/y/z" in item
        for item in user_payload["answering_policy"]["position_rendering_policy"]
    )
    assert (
        "inventing coordinates for position questions without coordinate evidence"
        in user_payload["graph_contract"]["forbidden_behaviors"]
    )


def test_live_graph_qa_payload_applies_deepseek_reasoning_options() -> None:
    responder = OpenAICompatibleGraphQAResponder(
        GraphQAConfig(
            model="deepseek-v4-pro",
            reasoning_effort="high",
            thinking_enabled=True,
        )
    )

    payload = responder._build_payload(
        question="What happened?",
        llm_ready_bundle={
            "llm_context_text": "Evidence:",
            "structured_graph_context": {},
        },
        graph_result={
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            }
        },
    )

    assert payload["model"] == "deepseek-v4-pro"
    assert payload["reasoning_effort"] == "high"
    assert payload["thinking"]["type"] == "enabled"


def test_live_graph_qa_retries_when_reasoning_exhausts_content() -> None:
    class _RetryingResponder(OpenAICompatibleGraphQAResponder):
        def __init__(self) -> None:
            super().__init__(
                GraphQAConfig(
                    model="deepseek-v4-pro",
                    reasoning_effort="high",
                    thinking_enabled=True,
                    min_evidence_lines=0,
                    min_focus_frames=0,
                )
            )
            self.payloads: list[dict[str, object]] = []

        def _post_payload(self, payload, *, api_key):  # type: ignore[no-untyped-def]
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "thinking only",
                            },
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer_text":"ok","used_focus_frames":[],'
                                '"used_evidence_ranks":[],"used_time_anchor_ids":[],'
                                '"uncertainty":[]}'
                            ),
                        },
                    }
                ]
            }

    responder = _RetryingResponder()
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    result = responder.answer(
        question="What happened?",
        graph_result={
            "answer_brief": "fallback",
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
            "llm_ready_bundle": {
                "evidence_lines": [],
                "focus_frame_lines": [],
                "structured_graph_context": {},
            },
        },
    )

    assert result["graph_rag_status"] == "live_validated"
    assert result["answer_text"] == "ok"
    assert len(responder.payloads) == 2
    assert responder.payloads[0]["reasoning_effort"] == "high"
    assert "thinking" in responder.payloads[0]
    assert "reasoning_effort" not in responder.payloads[1]
    assert responder.payloads[1]["thinking"] == {"type": "disabled"}


def test_live_graph_qa_retries_invalid_json_with_thinking_disabled() -> None:
    class _RetryingResponder(OpenAICompatibleGraphQAResponder):
        def __init__(self) -> None:
            super().__init__(
                GraphQAConfig(
                    model="deepseek-v4-pro",
                    reasoning_effort="high",
                    thinking_enabled=True,
                    min_evidence_lines=0,
                    min_focus_frames=0,
                )
            )
            self.payloads: list[dict[str, object]] = []

        def _post_payload(self, payload, *, api_key):  # type: ignore[no-untyped-def]
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "not json",
                            },
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer_text":"fixed","used_focus_frames":[],'
                                '"used_evidence_ranks":[],"used_time_anchor_ids":[],'
                                '"uncertainty":[]}'
                            ),
                        },
                    }
                ]
            }

    responder = _RetryingResponder()
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    result = responder.answer(
        question="What happened?",
        graph_result={
            "answer_brief": "fallback",
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
            "llm_ready_bundle": {
                "evidence_lines": [],
                "focus_frame_lines": [],
                "structured_graph_context": {},
            },
        },
    )

    assert result["graph_rag_status"] == "live_validated"
    assert result["answer_text"] == "fixed"
    assert len(responder.payloads) == 2
    assert responder.payloads[0]["thinking"] == {"type": "enabled"}
    assert responder.payloads[1]["thinking"] == {"type": "disabled"}


def test_generic_llm_evidence_selection_candidates_are_not_dataset_specific() -> None:
    bundle = {
        "evidence_lines": [
            "rank=1 kind=entity score=9 :: control panel near the doorway",
            "rank=2 kind=episode score=8 :: machine status changed to ready",
        ],
        "evidence_aggregation_lines": [
            "candidate_fact=1 type=state binding=direct_target ranks=2 fact=status ready"
        ],
        "evidence_aggregation": {
            "candidate_facts": [
                {
                    "fact_type": "state",
                    "fact_text": "status ready",
                    "binding_strength": "direct_target",
                    "support_ranks": [2],
                    "support_examples": ["machine status changed to ready"],
                }
            ]
        },
        "temporal_chain_lines": ["rank=1 sequence=setup -> active"],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
    )

    assert [item["source"] for item in candidates[:3]] == [
        "evidence_line",
        "evidence_line",
        "aggregation_line",
    ]
    assert any(
        item["source"] == "candidate_fact"
        and item["binding_strength"] == "direct_target"
        and item["support_ranks"] == [2]
        for item in candidates
    )
    joined = json.dumps(candidates, ensure_ascii=False)
    for forbidden in ["NaVQA", "seq0", "car", "sculpture", "color"]:
        assert forbidden not in joined


def test_llm_evidence_selection_accepts_visual_faiss_object_local_lines() -> None:
    bundle = {
        "evidence_lines": ["rank=1 kind=entity score=9 :: graph context"],
        "visual_faiss_object_local_fact_lines": [
            (
                "rank=1 kind=visual_faiss_object_local_fact score=5.1 frame=42 "
                "obj=3 label='sign' :: object-local visual fact; "
                "direct_attributes=object obj=3 label='sign' attr=red sign; "
                "direct_relations=object obj=3 relation=sign beside doorway"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
    )

    assert any(
        item["source"] == "visual_faiss_object_local"
        and item["fact_type"] == "object_local_visual_fact"
        for item in candidates
    )


def test_llm_evidence_selection_accepts_visual_faiss_local_subgraph_lines() -> None:
    bundle = {
        "evidence_lines": ["rank=1 kind=entity score=9 :: graph context"],
        "visual_faiss_local_subgraph_fact_lines": [
            (
                "rank=1 kind=visual_faiss_local_subgraph_fact score=8.1 "
                "entity=entity_sign label='sign' frame=42 "
                "binding=entity_local_subgraph_relation_bound :: "
                "entity-local subgraph fact; entity=entity_sign; label='sign'; "
                "time=frames=40-45; direct_graph='red sign beside doorway'"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
    )

    assert any(
        item["source"] == "visual_faiss_local_subgraph"
        and item["fact_type"] == "entity_local_subgraph_fact"
        for item in candidates
    )


def test_llm_evidence_selection_accepts_visual_faiss_evidence_layer_lines() -> None:
    bundle = {
        "evidence_lines": ["rank=1 kind=entity score=9 :: graph context"],
        "visual_faiss_evidence_layer_lines": [
            (
                "rank=1 kind=visual_faiss_evidence_layer_fact score=88.0 "
                "binding=direct_object_bound :: evidence_layer_object_pack "
                "obs=seq6:frame:000001_2d_rect_cam0_6_10:object:0 "
                "label='person' frame=10 similarity=0.710 "
                "attrs=person wearing blue jacket relations=anchor_subject person touching glass door"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
    )

    assert any(
        item["source"] == "visual_faiss_evidence_layer"
        and item["fact_type"] == "evidence_layer_object_pack"
        for item in candidates
    )


def test_llm_evidence_selection_accepts_visual_faiss_role_aware_lines() -> None:
    bundle = {
        "evidence_lines": ["rank=1 kind=entity score=9 :: graph context"],
        "visual_faiss_role_aware_lines": [
            (
                "rank=1 kind=visual_faiss_role_aware_candidate score=9.2 "
                "binding=target_query_with_structured_direct_attribute "
                "coverage=target=yes/attribute=yes/context=1 :: "
                "role_aware_candidate_subject seq=6 frame=10 obj=2 label='couch' "
                "query_role=target direct_processed_attributes="
                "direct_object_attribute owner=frame=000010 obj=2 label='couch' "
                "kind=color target='object' value='red'; "
                "constraint_context=context_query='entered building' role=event_constraint "
                "seq=6 frame=10 obj=7 label='building'"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
    )

    assert any(
        item["source"] == "visual_faiss_role_aware"
        and item["fact_type"] == "visual_faiss_role_aware_candidate"
        for item in candidates
    )


def test_visual_faiss_role_aware_keeps_constraint_object_as_context(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample"
    image_dir = sample_root / "images"
    hsgg_dir = sample_root / "hsgg_local_paths"
    image_dir.mkdir(parents=True)
    hsgg_dir.mkdir(parents=True)
    image_path = image_dir / "frame_000010.jpg"
    image_path.write_bytes(b"not-a-real-jpeg")
    (hsgg_dir / "objects.json").write_text(
        json.dumps(
            {
                "frame_000010": [
                    {"object": "couch", "bbox": [0, 0, 10, 10]},
                    {"object": "building", "bbox": [20, 0, 40, 20]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (hsgg_dir / "attributes.json").write_text(
        json.dumps(
            {
                "frame_000010": {
                    "0": ["Red leather couch beside entrance"],
                    "1": ["Building entrance with glass door"],
                }
            }
        ),
        encoding="utf-8",
    )

    graph_answer = {
        "llm_ready_bundle": {
            "evidence_lines": ["rank=1 kind=scene :: entered building"],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "llm_context_text": "Video: toy",
            "structured_graph_context": {},
        },
        "retrieval_bundle": {"evidence_digest": []},
    }
    visual_payload = {
        "groups": [
            {
                "query_phrase": "couches",
                "results": [
                    {
                        "faiss_id": 1,
                        "seq_id": "6",
                        "frame_key": "frame_000010",
                        "image_name": "frame_000010.jpg",
                        "graph_frame_index": 10,
                        "object_index": 0,
                        "label": "couch",
                        "similarity": 0.91,
                        "bbox": [0, 0, 10, 10],
                        "image_path": str(image_path),
                        "visual_attribute": "Red leather couch beside entrance",
                        "hsgg_visual_facts": ["attr=Red leather couch beside entrance"],
                    }
                ],
            },
            {
                "query_phrase": "entered building",
                "results": [
                    {
                        "faiss_id": 2,
                        "seq_id": "6",
                        "frame_key": "frame_000010",
                        "image_name": "frame_000010.jpg",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "building",
                        "similarity": 0.93,
                        "bbox": [20, 0, 40, 20],
                        "image_path": str(image_path),
                        "visual_attribute": "Building entrance with glass door",
                        "hsgg_visual_facts": ["attr=Building entrance with glass door"],
                    }
                ],
            },
        ]
    }

    updated = fuse_visual_faiss_role_aware_evidence(
        graph_answer,
        visual_payload,
        question="What color were the couches in the building?",
        question_ir={
            "target_entities": ["couches"],
            "hard_constraints": ["entered building"],
            "attribute_dimension": "color",
        },
        object_limit=4,
        context_limit=4,
    )
    bundle = updated["llm_ready_bundle"]
    role_lines = bundle["visual_faiss_role_aware_lines"]

    assert role_lines
    assert "label='couch'" in role_lines[0]
    assert "constraint_context=" in role_lines[0]
    assert "label='building'" not in role_lines[0].split("constraint_context=", 1)[0]
    assert (hsgg_dir / "processed_attributes.json").exists()
    assert json.dumps(bundle["visual_faiss_role_aware_evidence"], ensure_ascii=False)


def test_answerable_candidate_packs_group_late_object_local_evidence_without_gold() -> None:
    bundle = {
        "evidence_lines": [
            "rank=1 kind=entity score=9 frame=10 obj=1 label='person' :: graph context"
        ],
        "visual_faiss_object_local_fact_lines": [
            (
                "rank=1 kind=visual_faiss_object_local_fact score=9.0 frame=10 "
                "obj=1 label='person' :: object-local visual fact; "
                "object=frame=10 obj=1 label='person'; "
                "direct_attributes=object obj=1 label='person' attr=dark pants"
            ),
            (
                "rank=8 kind=visual_faiss_object_local_fact score=4.1 frame=42 "
                "obj=3 label='sneakers' :: object-local visual fact; "
                "object=frame=42 obj=3 label='sneakers'; "
                "direct_attributes=object obj=3 label='sneakers' attr=white sneakers; "
                "direct_relations=object obj=3 relation=on foot"
            ),
        ],
        "visual_faiss_ger_bound_fact_lines": [
            (
                "rank=8 kind=visual_faiss_ger_bound_fact score=4.0 frame=42 "
                "label='sneakers' q_terms=sneakers,foot :: GER-like bound visual fact; "
                "attributes=white sneakers; relations=on foot"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    without_packs = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=2,
    )
    with_packs = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        question="What color were the shoes on the foot?",
        max_candidates=5,
        answerable_candidate_packs=True,
        answerable_candidate_pack_max_items=4,
    )

    assert not any("white sneakers" in item["text"] for item in without_packs)
    pack = next(
        item
        for item in with_packs
        if item["source"] == "answerable_candidate_pack"
        and "white sneakers" in item["text"]
    )
    assert "direct_attributes=" in pack["text"]
    assert "direct_relations=" in pack["text"]
    assert "visual_faiss_object_local:2" in pack["support_sources"]
    assert "visual_faiss_ger_bound:1" in pack["support_sources"]
    assert pack["uses_gold_answers"] is False
    assert pack["uses_question_type_branches"] is False
    assert pack["uses_dataset_or_domain_wordlist"] is False
    joined = json.dumps(with_packs, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth", "seq0"]:
        assert forbidden not in joined


def test_answerable_candidate_packs_skip_unanchored_label_only_evidence() -> None:
    bundle = {
        "visual_faiss_ger_bound_fact_lines": [
            (
                "rank=1 kind=visual_faiss_ger_bound_fact score=8.1 "
                "label='person' binding=strong :: attributes=wearing blue jacket"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=5,
        answerable_candidate_packs=True,
        answerable_candidate_pack_max_items=4,
    )

    assert not any(
        item["source"] == "answerable_candidate_pack"
        and "key=label=" in item["text"]
        for item in candidates
    )


def test_answerable_candidate_packs_do_not_backfill_ambiguous_frame_label() -> None:
    bundle = {
        "visual_faiss_object_local_fact_lines": [
            (
                "rank=1 kind=visual_faiss_object_local_fact score=9.0 frame=42 "
                "obj=1 label='person' :: object-local visual fact; "
                "direct_attributes=object obj=1 label='person' attr=black jacket"
            ),
            (
                "rank=2 kind=visual_faiss_object_local_fact score=8.0 frame=42 "
                "obj=2 label='person' :: object-local visual fact; "
                "direct_attributes=object obj=2 label='person' attr=blue jacket"
            ),
        ],
        "visual_faiss_ger_bound_fact_lines": [
            (
                "rank=3 kind=visual_faiss_ger_bound_fact score=7.0 frame=42 "
                "label='person' :: GER-like bound visual fact; "
                "attributes=wearing jacket"
            )
        ],
        "evidence_aggregation": {"candidate_facts": []},
        "evidence_aggregation_lines": [],
    }

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result={"retrieval_bundle": {"evidence_digest": []}},
        max_candidates=10,
        answerable_candidate_packs=True,
        answerable_candidate_pack_max_items=8,
    )

    for item in candidates:
        if item["source"] != "answerable_candidate_pack":
            continue
        if "obj=1" in item["text"] or "obj=2" in item["text"]:
            assert "visual_faiss_ger_bound:1" not in item["support_sources"]


def test_visual_faiss_fusion_promotes_recall_into_evidence_pool() -> None:
    graph_answer = {
        "retrieval_bundle": {
            "focus_frames": [{"frame_index": 10}],
            "evidence_digest": [
                {
                    "rank": 1,
                    "source_record_id": "graph:1",
                    "source_kind": "entity",
                    "summary": "graph evidence",
                    "score": 1.0,
                }
            ],
        },
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy\nEvidence lines:\n- graph evidence",
            "evidence_lines": ["rank=1 kind=entity score=1 :: graph evidence"],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of red sign",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "red sign",
                        "similarity": 0.42,
                        "elapsed_sec_from_sequence_start": 4.2,
                        "visual_attribute": "red rectangular sign",
                        "visual_relations": ["sign near doorway"],
                        "hsgg_neighbor_attributes": [
                            "neighbor obj=7 label='doorway' attr=open doorway"
                        ],
                        "hsgg_visual_facts": [
                            "object obj=3 label='red sign' attr=red rectangular sign",
                            "object obj=3 relation=sign near doorway",
                            "neighbor obj=7 label='doorway' attr=open doorway",
                        ],
                    },
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 52,
                        "object_index": 8,
                        "label": "plain wall",
                        "similarity": 0.99,
                    }
                ],
            }
        ],
        "lines": ["visual query 'a photo of red sign' recalled object evidence"],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = updated["llm_ready_bundle"]
    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result=updated,
        max_candidates=10,
        balanced=True,
    )

    assert any("visual_faiss_object" in line for line in bundle["evidence_lines"])
    assert any(
        fact["fact_type"] == "visual_faiss_object_recall"
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
    )
    assert any(
        item["source_kind"] == "visual_faiss_object"
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    assert any(item["source"] == "visual_faiss" for item in candidates)
    assert any(
        item["source"] == "candidate_fact"
        and item.get("fact_type") == "visual_faiss_object_recall"
        for item in candidates
    )
    joined_evidence = "\n".join(bundle["evidence_lines"])
    assert "red rectangular sign" in joined_evidence
    assert "sign near doorway" in joined_evidence
    assert "open doorway" in joined_evidence
    visual_facts = [
        fact
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
        if fact.get("fact_type") == "visual_faiss_object_recall"
    ]
    assert visual_facts
    red_sign_facts = [
        fact for fact in visual_facts if "red rectangular sign" in fact["fact_text"]
    ]
    assert red_sign_facts
    assert "sign near doorway" in red_sign_facts[0]["fact_text"]
    assert red_sign_facts[0]["hsgg_visual_facts"] == [
        "object obj=3 label='red sign' attr=red rectangular sign",
        "object obj=3 relation=sign near doorway",
        "neighbor obj=7 label='doorway' attr=open doorway",
    ]
    assert "compression_score" in red_sign_facts[0]
    assert len([line for line in bundle["evidence_lines"] if "visual_faiss_object" in line]) <= 2
    serialized = json.dumps(visual_facts, ensure_ascii=False)
    for forbidden in ["gold_answer", "answer_label", "ground_truth", "question_type"]:
        assert forbidden not in serialized


def test_visual_faiss_concise_object_keeps_hsgg_fact_payload() -> None:
    compact = _concise_object(
        {
            "faiss_id": "12",
            "seq_id": "toy",
            "frame_key": "frame_a",
            "graph_frame_index": "42",
            "object_index": "3",
            "label": "red sign",
            "visual_attribute": "red rectangular sign",
            "visual_relations": ["sign near doorway"],
            "visual_relation_records": [
                {
                    "subject_index": 3,
                    "object_index": 7,
                    "subject_label": "sign",
                    "predicate": "near",
                    "object_label": "doorway",
                    "text": "sign near doorway",
                }
            ],
            "hsgg_neighbor_attributes": [
                "neighbor obj=7 label='doorway' attr=open doorway"
            ],
            "hsgg_visual_facts": [
                "object obj=3 label='red sign' attr=red rectangular sign",
                "object obj=3 relation=sign near doorway",
            ],
        }
    )

    assert compact["faiss_id"] == 12
    assert compact["graph_frame_index"] == 42
    assert compact["object_index"] == 3
    assert compact["visual_relation_records"][0]["predicate"] == "near"
    assert compact["hsgg_neighbor_attributes"] == [
        "neighbor obj=7 label='doorway' attr=open doorway"
    ]
    assert compact["hsgg_visual_facts"] == [
        "object obj=3 label='red sign' attr=red rectangular sign",
        "object obj=3 relation=sign near doorway",
    ]


def test_visual_faiss_query_phrases_are_question_ir_derived_without_word_filters() -> None:
    phrases = _visual_query_phrases(
        question="Did the robot stop near the stop sign?",
        question_ir={
            "target_state": "stopped near stop sign",
            "target_entities": ["stop sign"],
            "spatial_constraints": ["near"],
            "core_terms": ["side of", "for the first time", "red banner"],
        },
        limit=8,
    )

    assert "a photo of stopped near stop sign" in phrases
    assert "a photo of stop sign" in phrases
    assert "a photo of red banner" in phrases
    assert "a photo of side of" in phrases
    assert "a photo of for the first time" in phrases


def test_visual_faiss_role_aware_phrase_records_separate_target_and_constraints() -> None:
    records = _visual_query_phrase_records(
        question="What were the color of the couches near where I entered the building?",
        question_ir={
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "hard_constraints": ["entered building"],
            "core_terms": ["building"],
        },
        limit=8,
    )

    by_text = {record["text"]: record for record in records}

    assert by_text["couches"]["role"] == "primary_target"
    assert by_text["color"]["role"] == "target_attribute"
    assert by_text["entered building"]["role"] == "hard_constraint"
    assert by_text["building"]["role"] == "fallback"


def test_role_aware_visual_faiss_compression_keeps_target_over_stronger_context() -> None:
    visual_payload = {
        "question": "What were the color of the couches near where I entered the building?",
        "question_ir": {
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "hard_constraints": ["entered building"],
        },
        "groups": [
            {
                "query_phrase": "a photo of couches",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "faiss_id": 1,
                        "seq_id": "6",
                        "graph_frame_index": 10,
                        "object_index": 0,
                        "label": "couch",
                        "similarity": 0.12,
                        "visual_attribute": "Red leather couch beside entrance",
                        "hsgg_visual_facts": ["attr=Red leather couch beside entrance"],
                    }
                ],
            },
            {
                "query_phrase": "a photo of entered building",
                "query_phrase_role": "hard_constraint",
                "results": [
                    {
                        "faiss_id": 2,
                        "seq_id": "6",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "building",
                        "similarity": 0.99,
                        "visual_attribute": "Building entrance with glass door",
                        "hsgg_visual_facts": ["attr=Building entrance with glass door"],
                    }
                ],
            },
        ],
        "compression": {"selected_object_limit": 1},
    }

    default_rows = _compressed_visual_faiss_rows(visual_payload)
    role_aware_rows = _compressed_visual_faiss_rows(
        {
            **visual_payload,
            "compression": {
                "selected_object_limit": 1,
                "role_aware_compression_enabled": True,
            },
        }
    )

    assert default_rows[0][1]["label"] == "building"
    assert role_aware_rows[0][1]["label"] == "couch"
    assert role_aware_rows[0][1]["query_phrase_role"] == "primary_target"
    assert role_aware_rows[0][1]["role_aware_selection_bucket"] == "primary_target"


def test_entity_pack_phrase_records_keep_events_out_of_primary_target() -> None:
    records = visual_entity_pack_query_phrase_records(
        question="What color jacket was the person who opened the door for you wearing?",
        question_ir={
            "target_entities": ["person"],
            "attribute_dimension": "color",
            "event_constraints": [
                {
                    "relation": "during",
                    "operation": "occurrence",
                    "target_state": "opened the door for you",
                    "target_entities": ["person", "door"],
                }
            ],
            "hard_constraints": [
                {
                    "kind": "event",
                    "text": "person who opened the door for you",
                    "target_entities": ["person", "door"],
                }
            ],
        },
        limit=10,
    )

    primary_texts = [
        record["text"] for record in records if record["role"] == "primary_target"
    ]
    hard_texts = [
        record["text"] for record in records if record["role"] == "hard_constraint"
    ]
    event_texts = [
        record["text"] for record in records if record["role"] == "event_constraint"
    ]

    assert primary_texts == ["person"]
    assert "person who opened the door for" in hard_texts
    assert "door" in event_texts
    assert all("opened the door" not in text for text in primary_texts)


def test_visual_entity_candidate_pack_prioritizes_target_pack_and_keeps_neighbor_context(
    tmp_path: Path,
) -> None:
    temporal_graph_path = tmp_path / "temporal_graph.json"
    temporal_graph_path.write_text(
        json.dumps(
            {
                "video_id": "toy",
                "observations": [
                    {
                        "observation_id": "toy:frame_000010:obj_0001",
                        "video_id": "toy",
                        "entity_id": "entity_person",
                        "frame_index": 10,
                        "object_index": 1,
                        "label": "person",
                        "raw_state_values": ["blue jacket"],
                        "state_values": ["blue jacket"],
                    },
                    {
                        "observation_id": "toy:frame_000010:obj_0002",
                        "video_id": "toy",
                        "entity_id": "entity_door",
                        "frame_index": 10,
                        "object_index": 2,
                        "label": "door",
                        "raw_state_values": ["glass door"],
                        "state_values": ["glass door"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        source_graph_path=temporal_graph_path,
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:entity_person",
                video_id="toy",
                entity_id="entity_person",
                label="person",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                labels_seen=["person"],
                states_seen=["blue jacket"],
                attributes={"raw_visual_attributes": ["blue jacket"]},
            ),
            CanonicalEntityNode(
                node_id="entity:entity_door",
                video_id="toy",
                entity_id="entity_door",
                label="door",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                labels_seen=["door"],
                states_seen=["glass door"],
                attributes={"raw_visual_attributes": ["glass door"]},
            ),
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_person",
                video_id="toy",
                episode_id="episode_person",
                episode_kind="STATE",
                label="person wearing a blue jacket",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                scene_id="scene_1",
                primary_entity_id="entity_person",
                attr_key="color",
                attr_value="blue",
                source_ref_id="state_person",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": ["rank=1 kind=entity label='door'"],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color jacket was the person who opened the door wearing?",
        "query_phrase_records": [
            {
                "text": "person",
                "rendered": "a photo of person",
                "role": "primary_target",
            },
            {
                "text": "opened the door",
                "rendered": "a photo of opened the door",
                "role": "event_constraint",
            },
        ],
        "groups": [
            {
                "query_phrase": "a photo of person",
                "query_phrase_text": "person",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "person",
                        "similarity": 0.7,
                        "visual_attribute": "blue jacket",
                        "hsgg_visual_facts": [
                            "object obj=1 label='person' attr=blue jacket"
                        ],
                        "hsgg_neighbor_attributes": [
                            "neighbor obj=2 label='door' attr=black door mat"
                        ],
                    }
                ],
            },
            {
                "query_phrase": "a photo of opened the door",
                "query_phrase_text": "opened the door",
                "query_phrase_role": "event_constraint",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "door",
                        "similarity": 0.95,
                        "visual_attribute": "glass door",
                        "hsgg_visual_facts": [
                            "object obj=2 label='door' attr=glass door"
                        ],
                    }
                ],
            },
        ],
    }

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What color jacket was the person who opened the door wearing?",
        question_ir={
            "target_entities": ["person"],
            "attribute_dimension": "color",
            "event_constraints": ["opened the door"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    packs = bundle["visual_entity_candidate_packs"]["packs"]

    assert bundle["visual_entity_candidate_pack_lines"]
    assert packs[0]["anchor_entity_id"] == "entity_person"
    assert packs[0]["candidate_role"] == "target_direct"
    assert any("blue jacket" in value for value in packs[0]["direct_attributes"])
    assert any("black door mat" in value for value in packs[0]["neighbor_context"])
    assert bundle["evidence_lines"][0] == bundle["visual_entity_candidate_pack_lines"][0]

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result=updated,
        max_candidates=8,
    )
    assert candidates[0]["source"] == "visual_entity_candidate_pack"
    assert candidates[0]["fact_type"] == "visual_entity_candidate_pack"


def test_entity_pack_query_recall_payload_preserves_phrase_roles(monkeypatch) -> None:
    grounder = VisualFaissGrounder.__new__(VisualFaissGrounder)
    monkeypatch.setattr(
        grounder,
        "_encode_texts",
        lambda phrases: [
            np.asarray([float(index + 1), 0.0], dtype=np.float32)
            for index, _phrase in enumerate(phrases)
        ],
    )
    counter = {"value": 0}

    def fake_search_by_vector(**_kwargs: object) -> list[dict[str, object]]:
        counter["value"] += 1
        index = counter["value"]
        return [
            {
                "seq_id": "toy",
                "graph_frame_index": index,
                "object_index": index,
                "label": "couches" if index == 1 else "building",
                "similarity": 0.1,
            }
        ]

    monkeypatch.setattr(grounder, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(
        grounder,
        "_attach_hsgg_object_facts",
        lambda row, **_kwargs: None,
    )

    payload = grounder.build_query_recall_evidence(
        seq_id="toy",
        question="What were the color of the couches near where I entered the building?",
        graph_answer={},
        question_ir={
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "event_constraints": [
                {
                    "relation": "near",
                    "operation": "occurrence",
                    "target_state": "enter building",
                    "target_entities": ["building"],
                }
            ],
            "hard_constraints": [
                {
                    "kind": "event",
                    "text": "near where I entered the building",
                    "target_entities": ["couches", "building"],
                }
            ],
        },
        config=VisualFaissConfig(
            entity_pack_enabled=True,
            entity_pack_tracklet_index_enabled=False,
            query_phrase_limit=4,
            query_recall_top_k=1,
            max_lines=4,
        ),
    )

    groups = payload["groups"]
    assert groups[0]["query_phrase_text"] == "couches"
    assert groups[0]["query_phrase_role"] == "primary_target"
    assert groups[1]["query_phrase_text"] == "couches color"
    assert groups[1]["query_phrase_role"] == "target_attribute"
    assert groups[3]["query_phrase_text"] == "enter building"
    assert groups[3]["query_phrase_role"] == "event_constraint"


def test_entity_pack_fallback_keeps_target_entity_ahead_of_constraints() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:couch",
                video_id="toy",
                entity_id="entity_couch",
                label="couches",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["red couch"],
            ),
            CanonicalEntityNode(
                node_id="entity:building",
                video_id="toy",
                entity_id="entity_building",
                label="building",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of couches",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "couches",
                        "anchor_entity_id": "entity_couch",
                        "similarity": 0.1,
                        "visual_attribute": "red couches in lobby",
                    }
                ],
            },
            {
                "query_phrase": "a photo of enter building",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "building",
                        "anchor_entity_id": "entity_building",
                        "similarity": 0.3,
                        "visual_attribute": "building entrance",
                    }
                ],
            },
        ],
    }

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What were the color of the couches near where I entered the building?",
        question_ir={
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "event_constraints": [
                {
                    "relation": "near",
                    "target_state": "enter building",
                    "target_entities": ["building"],
                }
            ],
            "hard_constraints": [
                {
                    "kind": "event",
                    "text": "near where I entered the building",
                    "target_entities": ["couches", "building"],
                }
            ],
        },
    )

    packs = updated["llm_ready_bundle"]["visual_entity_candidate_packs"]["packs"]
    assert packs[0]["anchor_entity_id"] == "entity_couch"
    assert packs[0]["candidate_role"] == "target_direct"
    assert "red couches in lobby" in packs[0]["direct_attributes"]


def test_entity_pack_semantic_align_is_disabled_by_default() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:sculpture",
                video_id="toy",
                entity_id="entity_sculpture",
                label="sculpture",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["metal sculpture"],
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of art installation",
                "query_phrase_text": "art installation",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "sculpture",
                        "anchor_entity_id": "entity_sculpture",
                        "similarity": 0.8,
                        "visual_attribute": "metal sculpture",
                    }
                ],
            },
        ],
    }

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What was the large art installation made of?",
        question_ir={"target_entities": ["art installation"]},
    )

    metadata = updated["llm_ready_bundle"]["visual_entity_candidate_packs"]
    pack = metadata["packs"][0]
    assert "semantic_alignment_enabled" not in metadata
    assert "target_semantic_similarity" not in pack
    assert "semantic_target=" not in pack["fact_text"]


def test_entity_pack_semantic_align_boosts_primary_target_without_wordlist() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:sculpture",
                video_id="toy",
                entity_id="entity_sculpture",
                label="sculpture",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["metal sculpture"],
            ),
            CanonicalEntityNode(
                node_id="entity:building",
                video_id="toy",
                entity_id="entity_building",
                label="building",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["building entrance"],
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of art installation",
                "query_phrase_text": "art installation",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "building",
                        "anchor_entity_id": "entity_building",
                        "similarity": 0.9,
                        "visual_attribute": "building entrance",
                    },
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "sculpture",
                        "anchor_entity_id": "entity_sculpture",
                        "similarity": 0.86,
                        "visual_attribute": "metal sculpture",
                    },
                ],
            },
            {
                "query_phrase": "a photo of entered building",
                "query_phrase_text": "entered building",
                "query_phrase_role": "event_constraint",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "building",
                        "anchor_entity_id": "entity_building",
                        "similarity": 0.99,
                        "visual_attribute": "building entrance",
                    }
                ],
            },
        ],
    }

    def fake_encoder(texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            if "art installation" in lowered or "sculpture" in lowered:
                vectors.append(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            elif "building" in lowered or "entrance" in lowered:
                vectors.append(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        return vectors

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What was the large art installation made of after I entered the building?",
        question_ir={
            "target_entities": ["art installation"],
            "event_constraints": ["entered building"],
        },
        semantic_align_enabled=True,
        semantic_text_encoder=fake_encoder,
    )

    metadata = updated["llm_ready_bundle"]["visual_entity_candidate_packs"]
    packs = metadata["packs"]
    building_pack = next(pack for pack in packs if pack["anchor_entity_id"] == "entity_building")

    assert metadata["semantic_alignment_enabled"] is True
    assert metadata["semantic_alignment_provider"] == "siglip_text"
    assert metadata["uses_dataset_or_domain_wordlist"] is False
    assert metadata["uses_gold_answers"] is False
    assert metadata["llm_semantic_alignment"] is False
    assert packs[0]["anchor_entity_id"] == "entity_sculpture"
    assert packs[0]["target_semantic_query_text"] == "art installation"
    assert packs[0]["target_semantic_match_text"] in {"sculpture", "metal sculpture"}
    assert packs[0]["semantic_alignment_bonus"] > 0.0
    assert packs[0]["semantic_alignment_provider"] == "siglip_text"
    assert packs[0]["llm_semantic_alignment"] is False
    assert packs[0]["uses_dataset_or_domain_wordlist"] is False
    assert "semantic_target=" in packs[0]["fact_text"]
    assert building_pack["candidate_role"] in {"target_query", "constraint_context"}
    assert building_pack["target_semantic_query_text"] == "art installation"


def test_entity_pack_semantic_gated_role_downgrades_wrong_primary_recall() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:sculpture",
                video_id="toy",
                entity_id="entity_sculpture",
                label="sculpture",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["metal sculpture"],
            ),
            CanonicalEntityNode(
                node_id="entity:bicycle",
                video_id="toy",
                entity_id="entity_bicycle",
                label="bicycle",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["bicycle by building entrance"],
            ),
            CanonicalEntityNode(
                node_id="entity:building",
                video_id="toy",
                entity_id="entity_building",
                label="building",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["building entrance"],
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of art installation",
                "query_phrase_text": "art installation",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "bicycle",
                        "anchor_entity_id": "entity_bicycle",
                        "similarity": 0.96,
                        "visual_attribute": "bicycle by building entrance",
                    },
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "sculpture",
                        "anchor_entity_id": "entity_sculpture",
                        "similarity": 0.78,
                        "visual_attribute": "metal sculpture",
                    },
                ],
            },
            {
                "query_phrase": "a photo of entered building",
                "query_phrase_text": "entered building",
                "query_phrase_role": "event_constraint",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 3,
                        "label": "building",
                        "anchor_entity_id": "entity_building",
                        "similarity": 0.99,
                        "visual_attribute": "building entrance",
                    }
                ],
            },
        ],
    }

    def fake_encoder(texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            if "art installation" in lowered or "sculpture" in lowered:
                vectors.append(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            elif "building" in lowered or "entrance" in lowered or "bicycle" in lowered:
                vectors.append(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        return vectors

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What was the large art installation made of after I entered the building?",
        question_ir={
            "target_entities": ["art installation"],
            "event_constraints": ["entered building"],
        },
        semantic_gated_role_enabled=True,
        semantic_text_encoder=fake_encoder,
    )

    metadata = updated["llm_ready_bundle"]["visual_entity_candidate_packs"]
    packs = metadata["packs"]
    bicycle_pack = next(pack for pack in packs if pack["anchor_entity_id"] == "entity_bicycle")

    assert metadata["semantic_role_gate_enabled"] is True
    assert metadata["semantic_alignment_provider"] == "siglip_text"
    assert metadata["uses_dataset_or_domain_wordlist"] is False
    assert packs[0]["anchor_entity_id"] == "entity_sculpture"
    assert packs[0]["candidate_role"] == "target_query"
    assert packs[0]["candidate_role_before_semantic_gate"] == "provisional_target_query"
    assert packs[0]["candidate_role_after_semantic_gate"] == "target_query"
    assert packs[0]["semantic_role_gate_passed"] is True
    assert bicycle_pack["candidate_role"] == "visual_recall_candidate"
    assert bicycle_pack["candidate_role_before_semantic_gate"] == "provisional_target_query"
    assert bicycle_pack["candidate_role_after_semantic_gate"] == "visual_recall_candidate"
    assert bicycle_pack["semantic_role_gate_passed"] is False
    assert "target_query" not in bicycle_pack["ranking_score_breakdown"]
    assert bicycle_pack["semantic_alignment_bonus"] == 0.0
    assert bicycle_pack["uses_dataset_or_domain_wordlist"] is False
    assert bicycle_pack["llm_semantic_alignment"] is False


def test_entity_pack_semantic_gated_role_keeps_direct_label_hit() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:car",
                video_id="toy",
                entity_id="entity_car",
                label="car",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["blue car"],
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of car",
                "query_phrase_text": "car",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "car",
                        "anchor_entity_id": "entity_car",
                        "similarity": 0.9,
                        "visual_attribute": "blue car",
                    }
                ],
            }
        ],
    }

    def fake_encoder(texts: list[str]) -> list[np.ndarray]:
        return [np.asarray([0.0, 1.0, 0.0], dtype=np.float32) for _text in texts]

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What color was the car?",
        question_ir={"target_entities": ["car"], "attribute_dimension": "color"},
        semantic_gated_role_enabled=True,
        semantic_text_encoder=fake_encoder,
    )

    pack = updated["llm_ready_bundle"]["visual_entity_candidate_packs"]["packs"][0]
    assert pack["candidate_role"] == "target_direct"
    assert pack["candidate_role_before_semantic_gate"] == "target_direct"
    assert pack["candidate_role_after_semantic_gate"] == "target_direct"
    assert pack["semantic_role_gate_passed"] is True
    assert pack["semantic_role_gate_bypass"] == "target_direct"


def test_score_contract_ranks_semantic_target_pack_and_limits_rerank_pool() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:sofa",
                video_id="toy",
                entity_id="entity_sofa",
                label="sofa",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["red sofa"],
            ),
            CanonicalEntityNode(
                node_id="entity:building",
                video_id="toy",
                entity_id="entity_building",
                label="building",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                states_seen=["building entrance"],
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [
                "rank=1 kind=entity label='building' :: old graph evidence",
                "rank=2 kind=entity label='door' :: extra old graph evidence",
            ],
            "visual_faiss_ger_evidence_lines": [
                "rank=1 kind=visual_faiss_ger score=999 :: should not enter contract"
            ],
            "evidence_aggregation": {"candidate_facts": [{"fact_text": "old fact"}]},
            "evidence_aggregation_lines": ["old aggregation"],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "query_phrase_records": [
            {
                "text": "couches",
                "rendered": "a photo of couches",
                "role": "primary_target",
            },
            {
                "text": "entered building",
                "rendered": "a photo of entered building",
                "role": "event_constraint",
            },
        ],
        "groups": [
            {
                "query_phrase": "a photo of couches",
                "query_phrase_text": "couches",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "label": "sofa",
                        "anchor_entity_id": "entity_sofa",
                        "similarity": 0.4,
                        "visual_attribute": "red sofa in lobby",
                    }
                ],
            },
            {
                "query_phrase": "a photo of entered building",
                "query_phrase_text": "entered building",
                "query_phrase_role": "event_constraint",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 10,
                        "object_index": 2,
                        "label": "building",
                        "anchor_entity_id": "entity_building",
                        "similarity": 0.95,
                        "visual_attribute": "building entrance",
                    }
                ],
            },
        ],
    }

    def fake_encoder(texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "couches" in lowered or "sofa" in lowered:
                vectors.append(np.asarray([1.0, 0.0], dtype=np.float32))
            elif "building" in lowered:
                vectors.append(np.asarray([0.0, 1.0], dtype=np.float32))
            else:
                vectors.append(np.asarray([0.1, 0.1], dtype=np.float32))
        return vectors

    updated = fuse_visual_entity_candidate_packs(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        question="What were the color of the couches near where I entered the building?",
        question_ir={
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "event_constraints": ["entered building"],
        },
        score_contract_enabled=True,
        semantic_text_encoder=fake_encoder,
    )
    bundle = updated["llm_ready_bundle"]
    metadata = bundle["visual_entity_candidate_packs"]
    packs = metadata["packs"]

    assert metadata["score_contract_enabled"] is True
    assert bundle["score_contract"]["enabled"] is True
    assert packs[0]["anchor_entity_id"] == "entity_sofa"
    assert packs[0]["score_contract"] == "visual_candidate_pack_score_contract_v1"
    assert "score" not in packs[0]
    assert packs[0]["retrieval_score"] < packs[1]["retrieval_score"]
    assert packs[0]["semantic_score"] > packs[1]["semantic_score"]
    assert packs[0]["answerability_tier"] < packs[1]["answerability_tier"]
    assert packs[0]["semantic_match_text"] in {"sofa", "red sofa", "red sofa in lobby"}
    assert bundle["visual_faiss_ger_evidence_lines"] == []
    assert len(bundle["fallback_evidence_lines"]) == 2

    candidates = _build_llm_evidence_selection_candidates(
        bundle,
        graph_result=updated,
        max_candidates=8,
    )
    assert candidates[0]["source"] == "candidate_pack"
    assert candidates[0]["candidate_unit"] == "candidate_pack"
    assert {candidate["source"] for candidate in candidates} <= {
        "candidate_pack",
        "fallback_evidence",
    }


def test_visual_entity_tracklet_faiss_hit_maps_back_to_cg6_entity(
    tmp_path: Path,
) -> None:
    faiss = pytest.importorskip("faiss")
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    index_name = "object_siglip_flat_ip"
    index = faiss.IndexFlatIP(2)
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / f"{index_name}.faiss"))

    rows = [
        {
            "faiss_id": 0,
            "seq_id": "toy",
            "frame_key": "000001",
            "graph_frame_index": 1,
            "object_index": 0,
            "label": "person",
            "score": 0.91,
            "bbox": [0.1, 0.1, 0.3, 0.5],
            "image_path": "toy/images/000001.jpg",
        },
        {
            "faiss_id": 1,
            "seq_id": "toy",
            "frame_key": "000002",
            "graph_frame_index": 2,
            "object_index": 0,
            "label": "person",
            "score": 0.88,
            "bbox": [0.11, 0.1, 0.31, 0.5],
            "image_path": "toy/images/000002.jpg",
        },
        {
            "faiss_id": 2,
            "seq_id": "toy",
            "frame_key": "000001",
            "graph_frame_index": 1,
            "object_index": 1,
            "label": "door",
            "score": 0.93,
            "bbox": [0.6, 0.1, 0.8, 0.6],
            "image_path": "toy/images/000001.jpg",
        },
    ]
    connection = sqlite3.connect(index_dir / f"{index_name}.metadata.sqlite")
    connection.execute(
        """
        CREATE TABLE object_metadata (
            faiss_id INTEGER PRIMARY KEY,
            seq_id TEXT,
            graph_frame_index INTEGER,
            object_index INTEGER,
            raw_json TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO object_metadata (
            faiss_id, seq_id, graph_frame_index, object_index, raw_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                row["faiss_id"],
                row["seq_id"],
                row["graph_frame_index"],
                row["object_index"],
                json.dumps(row),
            )
            for row in rows
        ],
    )
    connection.commit()
    connection.close()

    temporal_graph_path = tmp_path / "temporal_graph.json"
    temporal_graph_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "observation_id": "obs_person_1",
                        "entity_id": "entity_person",
                        "frame_index": 1,
                        "object_index": 0,
                        "label": "person",
                        "bbox": [0.1, 0.1, 0.3, 0.5],
                        "state_values": ["blue jacket"],
                    },
                    {
                        "observation_id": "obs_person_2",
                        "entity_id": "entity_person",
                        "frame_index": 2,
                        "object_index": 0,
                        "label": "person",
                        "bbox": [0.11, 0.1, 0.31, 0.5],
                        "state_values": ["blue jacket"],
                    },
                    {
                        "observation_id": "obs_door",
                        "entity_id": "entity_door",
                        "frame_index": 1,
                        "object_index": 1,
                        "label": "door",
                        "bbox": [0.6, 0.1, 0.8, 0.6],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        source_graph_path=temporal_graph_path,
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity_person",
                video_id="toy",
                entity_id="entity_person",
                label="person",
                time_range=CanonicalTimeRange(start_frame=1, end_frame=2),
                observation_count=2,
                states_seen=["blue jacket"],
            ),
            CanonicalEntityNode(
                node_id="entity_door",
                video_id="toy",
                entity_id="entity_door",
                label="door",
                time_range=CanonicalTimeRange(start_frame=1, end_frame=1),
                observation_count=1,
            ),
        ],
    )

    grounder = VisualFaissGrounder(index_dir=index_dir, index_name=index_name)
    try:
        hits = grounder._search_entity_tracklets_by_vector(
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            seq_id="toy",
            graph=graph,
            top_k=2,
            search_k=2,
            min_score=0.0,
        )
    finally:
        grounder.close()

    assert hits
    assert hits[0]["entity_id"] == "entity_person"
    assert hits[0]["tracklet_hit"] is True
    assert hits[0]["tracklet_observation_count"] == 2
    assert hits[0]["tracklet_time_span"]["start_frame"] == 1
    assert hits[0]["tracklet_time_span"]["end_frame"] == 2


def test_visual_faiss_graph_anchor_expansion_promotes_nearby_graph_evidence() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="visibility",
                attr_value="visible",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {
            "focus_frames": [],
            "evidence_digest": [],
        },
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "groups": [
            {
                "query_phrase": "a photo of red sign",
                "results": [
                        {
                            "seq_id": "toy",
                            "graph_frame_index": 42,
                            "object_index": 3,
                            "label": "red sign",
                            "similarity": 0.51,
                            "visual_attribute": "red sign beside doorway",
                            "visual_relations": ["sign beside doorway"],
                            "hsgg_visual_facts": [
                                "object obj=3 label='red sign' attr=red sign beside doorway",
                                "object obj=3 relation=sign beside doorway",
                            ],
                        }
                    ],
                }
            ],
        "lines": ["visual query 'a photo of red sign' recalled object evidence"],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = expand_visual_faiss_graph_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        frame_radius=2,
        max_items=5,
    )
    bundle = updated["llm_ready_bundle"]

    assert any("red sign is visible" in line for line in bundle["evidence_lines"])
    assert any(
        "visual_faiss_soft_graph_context" in line
        and "object_facts=" in line
        and "nearby_graph=" in line
        for line in bundle["evidence_lines"]
    )
    assert any(
        fact["fact_type"] == "visual_faiss_soft_graph_context"
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
    )
    assert any(
        fact["fact_type"] == "visual_faiss_anchored_graph_evidence"
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
    )
    assert any(
        item["source_kind"] == "visual_faiss_soft_graph_context"
        and item.get("visual_anchor", {}).get("frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    assert any(
        item["source_kind"] == "episode"
        and item.get("visual_anchor", {}).get("frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    assert (
        bundle["visual_faiss_anchor_expansion"]["contract"]
        == "experimental_visual_faiss_graph_anchor_expansion_v1"
    )
    assert bundle["visual_faiss_anchor_expansion"]["soft_context_count"] == 1
    serialized = json.dumps(bundle["visual_faiss_anchor_expansion"], ensure_ascii=False)
    for forbidden in ["gold_answer", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_ger_fusion_promotes_object_and_kg_context_together() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with sign and doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="visibility",
                attr_value="visible",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {
            "focus_frames": [],
            "evidence_digest": [],
        },
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "Did you see the red sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of red sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "red sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign beside doorway",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='red sign' attr=red sign beside doorway",
                            "object obj=3 relation=sign beside doorway",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_ger_evidence_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_ger_evidence_lines"][0]
    assert bundle["evidence_aggregation_lines"][0].startswith("group=visual_faiss_ger:1")
    assert any("visual_faiss_ger_object_context" in line for line in bundle["evidence_lines"])
    assert any("red sign is visible" in line for line in bundle["evidence_lines"])
    ger_fact = facts[0]
    assert ger_fact["fact_type"] == "visual_faiss_ger_object_context"
    assert ger_fact["binding_strength"] == "visual_similarity_plus_kg_context"
    assert ger_fact["support_ranks"] == [1]
    assert "object_facts=" in ger_fact["fact_text"]
    assert "nearby_graph=" in ger_fact["fact_text"]
    assert bundle["structured_graph_context"]["visual_faiss_ger_evidence_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_ger_object_context"
        and item.get("visual_object", {}).get("graph_frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_ger_fusion"]
    assert metadata["contract"] == "experimental_visual_faiss_ger_fusion_v1"
    assert metadata["object_count"] == 1
    assert metadata["context_count"] == 1
    serialized = json.dumps(metadata, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized
    assert metadata["uses_gold_answers"] is False
    assert metadata["uses_question_type_branches"] is False


def test_visual_faiss_ger_binding_adds_generic_bound_facts_without_gold() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                        "hsgg_frame_context": [
                            "same_frame_anchor obj=3 label='sign' bbox=[10,20,40,60] center=(25,40) area=1200",
                            "same_frame_object obj=4 label='doorway' attr=open doorway spatial=right_of_anchor,below_anchor",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
        binding_enabled=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "hard_constraints": ["beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_ger_bound_fact_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_ger_bound_fact_lines"][0]
    assert facts[0]["fact_type"] == "visual_faiss_ger_bound_fact"
    assert facts[0]["source_kind"] == "visual_faiss_ger_bound_fact"
    assert "red sign" in facts[0]["fact_text"]
    assert "doorway" in facts[0]["fact_text"]
    assert "frame_context=" in facts[0]["fact_text"]
    assert facts[0]["frame_context_fragments"]
    assert bundle["structured_graph_context"]["visual_faiss_ger_bound_fact_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_ger_bound_fact"
        and item.get("visual_object", {}).get("graph_frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_ger_fusion"]
    binding = metadata["binding"]
    assert binding["contract"] == "experimental_visual_faiss_ger_binding_v1"
    assert binding["uses_gold_answers"] is False
    assert binding["uses_question_type_branches"] is False
    assert binding["uses_dataset_or_domain_wordlist"] is False
    assert metadata["frame_context"]["contract"] == (
        "experimental_visual_faiss_object_anchored_frame_context_v1"
    )
    assert metadata["frame_context"]["uses_gold_answers"] is False
    assert metadata["frame_context"]["uses_question_type_branches"] is False
    assert metadata["frame_context"]["uses_dataset_or_domain_wordlist"] is False
    serialized = json.dumps(metadata, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_fragment_bridge_adds_object_local_facts_without_gold() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                        "hsgg_frame_context": [
                            "same_frame_anchor obj=3 label='sign' bbox=[10,20,40,60] center=(25,40) area=1200",
                            "same_frame_object obj=4 label='doorway' attr=open doorway spatial=right_of_anchor,below_anchor",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
        binding_enabled=True,
        fragment_bridge_enabled=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_fragment_bridge_lines"]
    assert facts[0]["fact_type"] == "visual_faiss_fragment_bridge_fact"
    assert facts[0]["source_kind"] == "visual_faiss_fragment_bridge_fact"
    assert "red sign" in facts[0]["fact_text"]
    assert "same_frame_context=" in facts[0]["fact_text"]
    assert "nearby_graph=" in facts[0]["fact_text"]
    assert bundle["structured_graph_context"]["visual_faiss_fragment_bridge_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_fragment_bridge_fact"
        and item.get("visual_object", {}).get("graph_frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_ger_fusion"]
    bridge = metadata["fragment_bridge"]
    assert bridge["contract"] == "experimental_visual_faiss_fragment_bridge_v1"
    assert bridge["graph_mutation"] is False
    assert bridge["uses_gold_answers"] is False
    assert bridge["uses_question_type_branches"] is False
    assert bridge["uses_dataset_or_domain_wordlist"] is False
    serialized = json.dumps(metadata, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_object_local_facts_prioritize_recalled_object_without_gold() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                        "hsgg_frame_context": [
                            "same_frame_anchor obj=3 label='sign' bbox=[10,20,40,60] center=(25,40) area=1200",
                            "same_frame_object obj=4 label='doorway' attr=open doorway spatial=right_of_anchor,below_anchor",
                        ],
                        "hsgg_neighbor_attributes": [
                            "neighbor obj=4 label='doorway' attr=open doorway",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
        binding_enabled=True,
        object_local_facts_enabled=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_object_local_fact_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_object_local_fact_lines"][0]
    assert facts[0]["fact_type"] == "visual_faiss_object_local_fact"
    assert facts[0]["source_kind"] == "visual_faiss_object_local_fact"
    assert facts[0]["support_ranks"] == [1]
    assert "red sign" in facts[0]["fact_text"]
    assert "sign beside doorway" in facts[0]["fact_text"]
    assert "nearby_graph=" not in facts[0]["fact_text"]
    assert bundle["structured_graph_context"]["visual_faiss_object_local_fact_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_object_local_fact"
        and item.get("visual_object", {}).get("graph_frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_ger_fusion"]
    object_local = metadata["object_local_facts"]
    assert object_local["contract"] == "experimental_visual_faiss_object_local_facts_v1"
    assert object_local["graph_mutation"] is False
    assert object_local["uses_gold_answers"] is False
    assert object_local["uses_question_type_branches"] is False
    assert object_local["uses_dataset_or_domain_wordlist"] is False
    serialized = json.dumps(metadata, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_local_subgraph_uses_temporal_observation_anchor(
    tmp_path: Path,
) -> None:
    temporal_graph_path = tmp_path / "temporal_graph.json"
    temporal_graph_path.write_text(
        json.dumps(
            {
                "video_id": "toy",
                "observations": [
                    {
                        "observation_id": "toy:frame_000042:obj_0003",
                        "video_id": "toy",
                        "entity_id": "entity_sign",
                        "frame_name": "frame_000042",
                        "frame_rank": 42,
                        "frame_index": 42,
                        "timestamp_sec": 4.2,
                        "image_path": "frame_000042.jpg",
                        "object_index": 3,
                        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0},
                        "label": "sign",
                        "score": 0.9,
                        "state_values": ["red sign"],
                        "raw_state_values": ["red sign beside doorway"],
                        "scene_id": "scene_1",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        source_graph_path=temporal_graph_path,
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with sign and doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:entity_sign",
                video_id="toy",
                entity_id="entity_sign",
                label="sign",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
                observation_count=1,
                labels_seen=["sign"],
                states_seen=["red sign"],
                attributes={"raw_visual_attributes": ["red sign beside doorway"]},
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.7,
                        "bbox": [10.0, 20.0, 40.0, 60.0],
                        "visual_attribute": "red sign",
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=0,
        context_limit=4,
        binding_enabled=True,
        local_subgraph_enabled=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_local_subgraph_fact_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_local_subgraph_fact_lines"][0]
    assert facts[0]["fact_type"] == "visual_faiss_local_subgraph_fact"
    assert facts[0]["source_kind"] == "visual_faiss_local_subgraph_fact"
    assert facts[0]["entity_id"] == "entity_sign"
    assert "red sign is visible" in facts[0]["fact_text"]
    assert "entity=entity_sign" in bundle["visual_faiss_local_subgraph_fact_lines"][0]
    assert bundle["structured_graph_context"]["visual_faiss_local_subgraph_fact_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_local_subgraph_fact"
        and item.get("entity_id") == "entity_sign"
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_ger_fusion"]
    local_subgraph = metadata["local_subgraph_facts"]
    assert local_subgraph["contract"] == "experimental_visual_faiss_local_subgraph_facts_v1"
    assert local_subgraph["graph_mutation"] is False
    assert local_subgraph["uses_gold_answers"] is False
    assert local_subgraph["uses_question_type_branches"] is False
    assert local_subgraph["uses_dataset_or_domain_wordlist"] is False
    serialized = json.dumps(metadata, ensure_ascii=False)
    assert "visual_faiss_temporal_observation_anchor_v1" in serialized
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_local_subgraph_anchor_only_keeps_neighbors_as_context(
    tmp_path: Path,
) -> None:
    temporal_graph_path = tmp_path / "temporal_graph.json"
    temporal_graph_path.write_text(
        json.dumps(
            {
                "video_id": "toy",
                "observations": [
                    {
                        "observation_id": "toy:frame_000042:obj_0003",
                        "video_id": "toy",
                        "entity_id": "entity_sign",
                        "frame_index": 42,
                        "frame_rank": 42,
                        "timestamp_sec": 4.2,
                        "object_index": 3,
                        "bbox": [10.0, 20.0, 40.0, 60.0],
                        "label": "sign",
                        "state_values": ["red sign"],
                        "raw_state_values": ["red sign beside doorway"],
                        "scene_id": "scene_1",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        source_graph_path=temporal_graph_path,
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:entity_sign",
                video_id="toy",
                entity_id="entity_sign",
                label="sign",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=42),
                observation_count=1,
                labels_seen=["sign"],
                states_seen=["red sign"],
                attributes={"raw_visual_attributes": ["red sign beside doorway"]},
            ),
            CanonicalEntityNode(
                node_id="entity:entity_doorway",
                video_id="toy",
                entity_id="entity_doorway",
                label="doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=42),
                observation_count=1,
                labels_seen=["doorway"],
                attributes={"raw_visual_attributes": ["glass doorway"]},
            ),
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_sign",
                video_id="toy",
                episode_id="episode_sign",
                episode_kind="STATE",
                label="red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=42),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                object_entity_id="entity_doorway",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_sign",
            ),
            CanonicalEpisodeNode(
                node_id="episode_doorway",
                video_id="toy",
                episode_id="episode_doorway",
                episode_kind="STATE",
                label="glass doorway beside a red sign",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=42),
                scene_id="scene_1",
                primary_entity_id="entity_doorway",
                source_ref_id="state_doorway",
            ),
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.7,
                        "bbox": [10.0, 20.0, 40.0, 60.0],
                        "visual_attribute": "red sign",
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=0,
        context_limit=6,
        binding_enabled=True,
        local_subgraph_enabled=True,
        local_subgraph_anchor_only=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "attribute_dimension": "color",
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = [
        fact
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
        if fact.get("fact_type") == "visual_faiss_local_subgraph_fact"
    ]

    assert facts
    assert [fact["entity_id"] for fact in facts] == ["entity_sign"]
    assert "entity_doorway" in facts[0]["neighbor_entity_ids"]
    assert all("entity=entity_doorway" not in line for line in bundle["evidence_lines"])
    local_subgraph = bundle["visual_faiss_ger_fusion"]["local_subgraph_facts"]
    assert local_subgraph["candidate_entity_policy"] == "observation_anchor_entity_only"
    assert local_subgraph["anchor_only"] is True


def test_visual_faiss_balanced_graph_visual_keeps_graph_evidence_in_primary_pool() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_lines = [f"graph evidence {index}" for index in range(1, 5)]
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": graph_lines,
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {
                "candidate_facts": [
                    {
                        "fact_type": "graph_fact",
                        "fact_text": "graph fact",
                        "source_kind": "graph",
                    }
                ]
            },
            "evidence_aggregation_lines": ["graph aggregation"],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_ger_evidence(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
        binding_enabled=True,
        object_local_facts_enabled=True,
        balanced_graph_visual_evidence=True,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )

    bundle = updated["llm_ready_bundle"]
    assert bundle["evidence_lines"][:3] == graph_lines[:3]
    assert any(
        "visual_faiss_object_local_fact" in line
        for line in bundle["evidence_lines"][:6]
    )
    assert bundle["evidence_aggregation"]["candidate_facts"][0]["fact_type"] == "graph_fact"
    assert "balanced_graph_visual_evidence" in bundle["visual_faiss_ger_fusion"]
    assert (
        bundle["visual_faiss_ger_fusion"]["balanced_graph_visual_evidence"][
            "uses_question_type_branches"
        ]
        is False
    )


def test_visual_faiss_schema_terms_are_not_counted_as_match_terms() -> None:
    assert _without_schema_match_terms(
        ["object", "text", "frame", "van", "sidewalk", "relation"]
    ) == ["van", "sidewalk"]


def test_visual_faiss_unified_candidates_promote_object_graph_binding() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene_node_1",
                video_id="toy",
                scene_id="scene_1",
                label="hallway with red sign beside the doorway",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="color",
                attr_value="red",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What color was the sign beside the doorway?",
        "groups": [
            {
                "query_phrase": "a photo of sign beside doorway",
                "results": [
                    {
                        "seq_id": "toy",
                        "graph_frame_index": 42,
                        "object_index": 3,
                        "label": "sign",
                        "similarity": 0.51,
                        "visual_attribute": "red sign",
                        "visual_relations": ["sign beside doorway"],
                        "hsgg_visual_facts": [
                            "object obj=3 label='sign' attr=red sign",
                            "object obj=3 relation=sign beside doorway",
                        ],
                    }
                ],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42, "timestamp_sec": 4.2}],
    }

    updated = fuse_visual_faiss_unified_candidates(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=4,
        context_radius=2,
        context_limit=4,
        question="What color was the sign beside the doorway?",
        question_ir={
            "target_entities": ["sign beside doorway"],
            "hard_constraints": ["beside doorway"],
            "attribute_dimension": "color",
            "evidence_requirements": ["object attribute and relation evidence"],
        },
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_unified_candidate_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_unified_candidate_lines"][0]
    assert facts[0]["fact_type"] == "visual_faiss_unified_candidate"
    assert facts[0]["binding_strength"] == "strong_object_graph_bound"
    assert facts[0]["binding_checks"]["attribute_bound_to_object"] is True
    assert facts[0]["binding_checks"]["relation_bound_to_object"] is True
    assert "red sign" in facts[0]["fact_text"]
    assert bundle["structured_graph_context"]["visual_faiss_unified_candidate_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_unified_candidate"
        and item.get("visual_object", {}).get("graph_frame_index") == 42
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_unified_candidates"]
    assert metadata["contract"] == "experimental_visual_faiss_unified_candidates_v1"
    assert metadata["uses_gold_answers"] is False
    assert metadata["uses_question_type_branches"] is False
    assert metadata["uses_dataset_or_domain_wordlist"] is False
    serialized = json.dumps(metadata, ensure_ascii=False)
    for forbidden in ["gold_answer_text", "answer_label", "ground_truth"]:
        assert forbidden not in serialized


def test_visual_faiss_ger_bound_facts_are_added_to_final_prompt() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": [],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "visual_faiss_ger_bound_fact_lines": [
            "rank=1 kind=visual_faiss_ger_bound_fact score=8.1 frame=42 "
            "label='sign' q_terms=sign,doorway :: GER-like bound visual fact; "
            "attributes=object_attribute=red sign"
        ],
        "structured_graph_context": {},
    }

    messages = responder._build_messages(
        question="What color was the sign beside the doorway?",
        llm_ready_bundle=bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["graph_summary"]["visual_faiss_ger_bound_fact_lines"]
    assert "visual_faiss_ger_bound_fact_lines" in " ".join(
        user_payload["answering_policy"]["visual_faiss_policy"]
    )
    assert "Visual FAISS GER-like bound facts:" in user_payload["graph_context"]


def test_visual_faiss_unified_candidates_are_added_to_final_prompt() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": [],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "visual_faiss_unified_candidate_lines": [
            "rank=1 kind=visual_faiss_unified_candidate score=12.1 "
            "binding=strong_object_graph_bound frame=42 label='sign' "
            "object_terms=sign,doorway :: Unified graph+FAISS candidate; "
            "attributes=object_attribute=red sign"
        ],
        "structured_graph_context": {},
    }

    messages = responder._build_messages(
        question="What color was the sign beside the doorway?",
        llm_ready_bundle=bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["graph_summary"]["visual_faiss_unified_candidate_lines"]
    assert "visual_faiss_unified_candidate_lines" in " ".join(
        user_payload["answering_policy"]["visual_faiss_policy"]
    )
    assert "Visual FAISS unified graph candidates:" in user_payload["graph_context"]


def test_visual_faiss_role_aware_lines_are_added_to_final_prompt() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": [],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "visual_faiss_role_aware_lines": [
            (
                "rank=1 kind=visual_faiss_role_aware_candidate score=9.2 "
                "binding=target_query_with_structured_direct_attribute "
                "coverage=target=yes/attribute=yes/context=1 :: "
                "role_aware_candidate_subject seq=6 frame=10 obj=2 label='couch' "
                "query_role=target direct_processed_attributes="
                "direct_object_attribute owner=frame=000010 obj=2 label='couch' "
                "kind=color target='object' value='red'; "
                "constraint_context=context_query='entered building' role=event_constraint "
                "seq=6 frame=10 obj=7 label='building'"
            )
        ],
        "structured_graph_context": {},
    }

    messages = responder._build_messages(
        question="What color were the couches inside the building?",
        llm_ready_bundle=bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["graph_summary"]["visual_faiss_role_aware_lines"]
    assert "visual_faiss_role_aware_lines" in " ".join(
        user_payload["answering_policy"]["visual_faiss_policy"]
    )
    assert "Visual FAISS role-aware candidate subjects:" in user_payload["graph_context"]


def test_visual_faiss_role_aware_aux_lines_are_prompted_as_auxiliary() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": [],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "visual_faiss_role_aware_aux_lines": [
            (
                "rank=1 kind=visual_faiss_role_aware_aux_candidate score=7.1 "
                "binding=target_query_with_structured_direct_attribute "
                "coverage=target=yes/attribute=yes/context=1 :: "
                "role_aware_aux_candidate_subject seq=6 frame=10 obj=2 "
                "label='couch' query_role=target direct_processed_attributes="
                "direct_object_attribute owner=frame=000010 obj=2 label='couch' "
                "kind=color target='object' value='red'"
            )
        ],
        "structured_graph_context": {},
    }

    messages = responder._build_messages(
        question="What color were the couches inside the building?",
        llm_ready_bundle=bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])
    policy_text = " ".join(user_payload["answering_policy"]["visual_faiss_policy"])

    assert user_payload["graph_summary"]["visual_faiss_role_aware_aux_lines"]
    assert "auxiliary role checks" in policy_text
    assert "do not replace GER-like" in policy_text
    assert "Visual FAISS role-aware auxiliary role checks:" in user_payload["graph_context"]


def test_generic_llm_evidence_selection_is_added_to_final_prompt() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": ["rank=1 kind=episode score=8 :: machine status changed to ready"],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "structured_graph_context": {},
    }
    updated_bundle = _bundle_with_llm_evidence_selection(
        bundle,
        llm_selection={
            "contract": "generic_llm_evidence_selection_v1",
            "available": True,
            "selected_evidence": [
                {
                    "id": "evidence_line:1",
                    "source": "evidence_line",
                    "evidence_rank": 1,
                    "text": "rank=1 kind=episode score=8 :: machine status changed to ready",
                }
            ],
            "selected_ids": ["evidence_line:1"],
            "answer_cue": "ready",
            "insufficient_evidence": False,
        },
    )

    messages = responder._build_messages(
        question="Which status did the machine report?",
        llm_ready_bundle=updated_bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert (
        user_payload["graph_summary"]["llm_evidence_selection"]["contract"]
        == "generic_llm_evidence_selection_v1"
    )
    assert user_payload["graph_summary"]["llm_selected_evidence_lines"]
    assert "use_llm_evidence_selection_when_available" in user_payload["instructions"]
    assert "generic preselection" in " ".join(
        user_payload["answering_policy"]["llm_evidence_selection_policy"]
    )


def test_generic_answer_candidate_extraction_is_added_to_final_prompt() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": ["rank=1 kind=episode score=8 :: machine status changed to ready"],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "structured_graph_context": {},
    }
    updated_bundle = _bundle_with_answer_candidate_extraction(
        bundle,
        extraction={
            "contract": "generic_answer_candidate_extraction_v1",
            "available": True,
            "candidates": [
                {
                    "answer": "ready",
                    "answer_type": "text",
                    "support_ids": ["evidence_line:1"],
                    "support_text": "machine status changed to ready",
                    "confidence": "high",
                    "reasoning": "direct state evidence",
                }
            ],
            "insufficient_evidence": False,
        },
        max_candidates=6,
    )

    messages = responder._build_messages(
        question="Which status did the machine report?",
        llm_ready_bundle=updated_bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert (
        user_payload["graph_summary"]["answer_candidate_extraction"]["contract"]
        == "generic_answer_candidate_extraction_v1"
    )
    assert user_payload["graph_summary"]["answer_candidate_lines"]
    assert "use_answer_candidates_when_available" in user_payload["instructions"]
    assert "all_question_types" not in json.dumps(user_payload, ensure_ascii=False).lower()
    assert "binary" in " ".join(
        user_payload["answering_policy"]["answer_candidate_policy"]
    )
    assert "not as a gold label" in " ".join(
        user_payload["answering_policy"]["answer_candidate_policy"]
    )


def test_experimental_atomic_answer_facts_are_default_off_and_prompted() -> None:
    bundle = {
        "answer_brief": "retrieved evidence",
        "llm_context_text": "Video: toy",
        "evidence_lines": [
            "rank=1 kind=object_attribute score=8 frame=42 "
            "label='person' :: attributes=wearing blue jacket"
        ],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {
            "candidate_facts": [
                {
                    "fact_type": "attribute",
                    "fact_text": "subject=person attribute=wearing blue jacket frame=42",
                    "binding_strength": "target_bound",
                    "support_ranks": [1],
                }
            ]
        },
        "visual_faiss_ger_bound_fact_lines": [
            "rank=1 kind=visual_faiss_ger_bound_fact score=8.1 frame=42 "
            "label='person' binding=strong :: attributes=wearing blue jacket"
        ],
        "structured_graph_context": {},
    }

    unchanged = OpenAICompatibleGraphQAResponder(
        GraphQAConfig(experimental_atomic_answer_facts=False)
    )._graph_result_with_atomic_answer_facts(
        graph_result={"llm_ready_bundle": dict(bundle)}
    )
    assert "atomic_answer_fact_lines" not in unchanged["llm_ready_bundle"]

    responder = OpenAICompatibleGraphQAResponder(
        GraphQAConfig(experimental_atomic_answer_facts=True, atomic_answer_fact_max_items=4)
    )
    updated_result = responder._graph_result_with_atomic_answer_facts(
        graph_result={"llm_ready_bundle": dict(bundle), "matches": [], "retrieval_bundle": {}}
    )
    updated_bundle = updated_result["llm_ready_bundle"]

    assert updated_bundle["atomic_answer_facts"]["contract"] == (
        "experimental_atomic_answer_facts_v1"
    )
    assert updated_bundle["atomic_answer_facts"]["uses_gold_answers"] is False
    assert updated_bundle["atomic_answer_facts"]["uses_dataset_or_domain_wordlist"] is False
    assert updated_bundle["atomic_answer_fact_lines"]
    assert "subject='person'" in updated_bundle["atomic_answer_fact_lines"][0]

    messages = responder._build_messages(
        question="What color jacket was the person wearing?",
        llm_ready_bundle=updated_bundle,
        graph_result={
            "matches": [],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
            },
        },
    )
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["graph_summary"]["atomic_answer_fact_lines"]
    assert "use_atomic_answer_facts_when_available" in user_payload["instructions"]
    assert "Atomic answer facts:" in user_payload["graph_context"]
    assert "subject, predicate, object or attribute" in " ".join(
        user_payload["answering_policy"]["atomic_answer_fact_policy"]
    )


def test_live_graph_qa_runs_generic_evidence_selection_before_answer() -> None:
    class _SelectingResponder(OpenAICompatibleGraphQAResponder):
        def __init__(self) -> None:
            super().__init__(
                GraphQAConfig(
                    evidence_rerank_enabled=True,
                    min_evidence_lines=0,
                    min_focus_frames=0,
                )
            )
            self.payloads: list[dict[str, object]] = []

        def _post_payload(self, payload, *, api_key):  # type: ignore[no-untyped-def]
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"selected_ids":["evidence_line:2"],'
                                    '"answer_cue":"ready",'
                                    '"insufficient_evidence":false,'
                                    '"insufficiency_reason":"",'
                                    '"rationale":"second item directly answers"}'
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer_text":"The status was ready.",'
                                '"used_focus_frames":[],"used_evidence_ranks":[2],'
                                '"used_time_anchor_ids":[],"uncertainty":[]}'
                            )
                        }
                    }
                ]
            }

    responder = _SelectingResponder()
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    result = responder.answer(
        question="Which status did the machine report?",
        graph_result={
            "answer_brief": "fallback",
            "matches": [{"source_kind": "entity"}, {"source_kind": "episode"}],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
                "evidence_digest": [],
            },
            "llm_ready_bundle": {
                "llm_context_text": "Video: toy",
                "evidence_lines": [
                    "rank=1 kind=entity score=9 :: control panel near the doorway",
                    "rank=2 kind=episode score=8 :: machine status changed to ready",
                ],
                "focus_frame_lines": [],
                "scene_lines": [],
                "temporal_chain_lines": [],
                "evidence_aggregation_lines": [],
                "evidence_aggregation": {"candidate_facts": []},
                "structured_graph_context": {},
            },
        },
    )

    assert result["graph_rag_status"] == "live_validated"
    assert result["answer_text"] == "The status was ready."
    assert (
        result["graph_rag_metadata"]["evidence_selection"]["selected_ids"]
        == ["evidence_line:2"]
    )
    assert len(responder.payloads) == 2
    selection_payload = json.loads(responder.payloads[0]["messages"][1]["content"])
    assert selection_payload["task"] == "select_answer_supporting_evidence"
    answer_payload = json.loads(responder.payloads[1]["messages"][1]["content"])
    assert answer_payload["graph_summary"]["llm_evidence_selection"]["selected_ids"] == [
        "evidence_line:2"
    ]


def test_live_graph_qa_runs_generic_answer_candidate_extraction_before_answer() -> None:
    class _CandidateResponder(OpenAICompatibleGraphQAResponder):
        def __init__(self) -> None:
            super().__init__(
                GraphQAConfig(
                    answer_candidate_extraction_enabled=True,
                    min_evidence_lines=0,
                    min_focus_frames=0,
                )
            )
            self.payloads: list[dict[str, object]] = []

        def _post_payload(self, payload, *, api_key):  # type: ignore[no-untyped-def]
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"available":true,'
                                    '"candidates":[{"answer":"ready",'
                                    '"answer_type":"text",'
                                    '"support_ids":["evidence_line:2"],'
                                    '"support_text":"machine status changed to ready",'
                                    '"confidence":"high",'
                                    '"reasoning":"direct evidence"}],'
                                    '"insufficient_evidence":false,'
                                    '"insufficiency_reason":""}'
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer_text":"The status was ready.",'
                                '"used_focus_frames":[],"used_evidence_ranks":[2],'
                                '"used_time_anchor_ids":[],"uncertainty":[]}'
                            )
                        }
                    }
                ]
            }

    responder = _CandidateResponder()
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    result = responder.answer(
        question="Which status did the machine report?",
        graph_result={
            "answer_brief": "fallback",
            "matches": [{"source_kind": "entity"}, {"source_kind": "episode"}],
            "retrieval_bundle": {
                "focus_frames": [],
                "scene_summaries": [],
                "temporal_chains": [],
                "evidence_digest": [],
            },
            "llm_ready_bundle": {
                "llm_context_text": "Video: toy",
                "evidence_lines": [
                    "rank=1 kind=entity score=9 :: control panel near the doorway",
                    "rank=2 kind=episode score=8 :: machine status changed to ready",
                ],
                "focus_frame_lines": [],
                "scene_lines": [],
                "temporal_chain_lines": [],
                "evidence_aggregation_lines": [],
                "evidence_aggregation": {"candidate_facts": []},
                "structured_graph_context": {},
            },
        },
    )

    assert result["graph_rag_status"] == "live_validated"
    assert result["answer_text"] == "The status was ready."
    assert (
        result["graph_rag_metadata"]["answer_candidate_extraction"]["contract"]
        == "generic_answer_candidate_extraction_v1"
    )
    assert len(responder.payloads) == 2
    extraction_payload = json.loads(responder.payloads[0]["messages"][1]["content"])
    assert extraction_payload["task"] == "extract_generic_answer_candidates"
    answer_payload = json.loads(responder.payloads[1]["messages"][1]["content"])
    assert answer_payload["graph_summary"]["answer_candidate_lines"]


def test_answer_canonical_graph_query_graph_only_returns_structured_answer() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    result = answer_canonical_graph_query(
        graph,
        MemoryQuery(query_text="state transition shooting", scene_ids=["scene_002"], limit=6),
        question="What state transition happens in scene 2?",
        retrieval_mode="temporal",
        answer_mode="graph_only",
    )

    assert result["graph_answer"]["matches"]
    assert result["llm_answer"]["mode"] == "graph_only"
    assert result["llm_answer"]["graph_rag_status"] == "graph_only"
    assert result["llm_answer"]["structured_answer"]["used_evidence_lines"]
    assert result["llm_answer"]["structured_answer"]["used_focus_frames"]


def test_answer_canonical_graph_query_mock_reuses_llm_ready_bundle() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    result = answer_canonical_graph_query(
        graph,
        MemoryQuery(entity_ids=["player_1"], limit=6),
        question="Tell me about player_1",
        retrieval_mode="entity",
        answer_mode="mock",
    )

    assert result["graph_answer"]["query_type"] == "canonical_entity"
    assert result["llm_answer"]["mode"] == "mock"
    assert result["llm_answer"]["graph_rag_status"] == "mock_graph_qa"
    assert result["llm_answer"]["structured_answer"]["used_evidence_lines"]
    assert "Evidence:" in result["llm_answer"]["answer_text"]


def test_retrieve_canonical_graph_context_uses_observed_relation_burst_focus_frames() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_observed_relation_burst()
    )

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="player near hoop",
            scene_ids=["scene_002"],
            limit=8,
        ),
        retrieval_mode="temporal",
    )

    burst_match = next(
        item
        for item in graph_result["matches"]
        if item["metadata"].get("episode_kind") == "RELATION_OBSERVED_BURST"
    )
    assert burst_match["metadata"]["observed_frames"] == [12, 14, 16]

    focus_frames = [
        item["frame_index"]
        for item in graph_result["retrieval_bundle"]["focus_frames"]
        if isinstance(item, dict) and item.get("frame_index") is not None
    ]
    assert 12 in focus_frames
    assert 14 in focus_frames
    assert 16 in focus_frames


def test_retrieve_canonical_graph_context_uses_compact_observed_relation_focus_frames() -> None:
    graph = build_canonical_temporal_memory_graph(
        _sample_temporal_graph_with_observed_relation_burst(),
        compact_relation_observed_events=True,
    )

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="player near hoop",
            scene_ids=["scene_002"],
            limit=8,
        ),
        retrieval_mode="temporal",
    )

    burst_match = next(
        item
        for item in graph_result["matches"]
        if item["metadata"].get("episode_kind") == "RELATION_OBSERVED_BURST"
    )
    assert burst_match["metadata"]["source_event_ids"] == ["obs_1", "obs_2", "obs_3"]
    assert burst_match["metadata"]["observed_frames"] == [12, 14, 16]

    focus_frames = [
        item["frame_index"]
        for item in graph_result["retrieval_bundle"]["focus_frames"]
        if isinstance(item, dict) and item.get("frame_index") is not None
    ]
    assert 12 in focus_frames
    assert 14 in focus_frames
    assert 16 in focus_frames


def test_retrieve_canonical_graph_context_uses_base_delta_projection_focus_frames() -> None:
    document = _sample_temporal_graph_document().model_copy(
        update={
            "scene_delta_nodes": [
                SceneDeltaNode(
                    node_id="scene_002:delta:player_1",
                    scene_id="scene_002",
                    video_id="video_001",
                    entity_id="player_1",
                    label="man",
                    start_frame=11,
                    end_frame=14,
                    start_ts=1.1,
                    end_ts=1.4,
                    support_frames=4,
                    transition_event_count=1,
                    relation_episode_count=1,
                    state_episode_count=1,
                )
            ],
            "scene_delta_observations": [
                SceneDeltaObservation(
                    delta_observation_id="delta_obs_11",
                    scene_id="scene_002",
                    video_id="video_001",
                    entity_id="player_1",
                    scene_delta_node_id="scene_002:delta:player_1",
                    source_segment_id="seg_player_1_b",
                    source_observation_id="obs_player_11",
                    frame_name="frame_000011.jpg",
                    frame_rank=11,
                    frame_index=11,
                    timestamp_sec=1.1,
                    image_path="frames/frame_000011.jpg",
                    bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.5),
                    label="man",
                    trigger_event_ids=["event_1"],
                    trigger_event_types=["STATE_TRANSITION"],
                    marks_segment_start=True,
                ),
                SceneDeltaObservation(
                    delta_observation_id="delta_obs_13",
                    scene_id="scene_002",
                    video_id="video_001",
                    entity_id="player_1",
                    scene_delta_node_id="scene_002:delta:player_1",
                    source_segment_id="seg_player_1_b",
                    source_observation_id="obs_player_13",
                    frame_name="frame_000013.jpg",
                    frame_rank=13,
                    frame_index=13,
                    timestamp_sec=1.3,
                    image_path="frames/frame_000013.jpg",
                    bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.4, y2=0.5),
                    label="man",
                    trigger_event_ids=["event_1"],
                    trigger_event_types=["STATE_TRANSITION"],
                ),
            ],
        }
    )
    graph = build_canonical_temporal_memory_graph(document)

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(query_text="scene delta dynamic change shooting", limit=8),
        retrieval_mode="temporal",
        temporal_document=document,
        include_base_delta_projection=True,
    )

    delta_match = next(
        item for item in graph_result["matches"] if item["source_kind"] == "scene_delta"
    )
    assert delta_match["metadata"]["observed_frames"] == [11, 13]
    assert graph_result["filters"]["include_base_delta_projection"] is True

    focus_frames = [
        item["frame_index"]
        for item in graph_result["retrieval_bundle"]["focus_frames"]
        if isinstance(item, dict) and item.get("frame_index") is not None
    ]
    assert 11 in focus_frames
    assert 13 in focus_frames


def test_retrieve_canonical_graph_context_keeps_query_window_focus_boundaries() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())

    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(
            query_text="near hoop",
            start_ts=0.5,
            end_ts=0.9,
            limit=1,
        ),
        retrieval_mode="temporal",
    )

    focus_by_frame = {
        item["frame_index"]: item
        for item in graph_result["retrieval_bundle"]["focus_frames"]
    }
    assert "query_window_boundary" in focus_by_frame[5]["reasons"]
    assert "query_window_boundary" in focus_by_frame[9]["reasons"]


def test_llm_ready_time_range_text_keeps_frame_axis_distinct() -> None:
    assert (
        _time_range_text(
            {
                "start_frame": 190,
                "end_frame": 195,
                "start_ts": 190.0,
                "end_ts": 195.0,
            }
        )
        == "frames 190-195"
    )
    assert (
        _time_range_text(
            {
                "start_frame": 210,
                "end_frame": 210,
                "start_ts": 210.0,
                "end_ts": 210.0,
            }
        )
        == "frame 210"
    )
    assert (
        _time_range_text(
            {
                "start_frame": 70,
                "end_frame": 120,
                "start_ts": 1.42,
                "end_ts": 2.43,
            }
        )
        == "frames 70-120 1.42s->2.43s"
    )


def test_canonical_entity_live_normalization_backfills_richer_evidence_set() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(entity_ids=["player_1"], limit=6),
        retrieval_mode="entity",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    available_evidence_lines = list(graph_result["llm_ready_bundle"]["evidence_lines"])
    available_focus_frames = [
        item.get("frame_index")
        for item in graph_result["retrieval_bundle"]["focus_frames"]
        if isinstance(item, dict)
    ]

    normalized = responder._normalize_structured_answer(
        parsed={
            "answer_text": "player_1 appears over time.",
            "used_focus_frames": available_focus_frames[:2],
            "used_evidence_lines": available_evidence_lines[:3],
            "uncertainty": [],
        },
        graph_result=graph_result,
    )

    assert normalized is not None
    assert len(normalized["used_evidence_lines"]) == 4
    assert normalized["used_evidence_lines"] == available_evidence_lines[:4]
    assert set(normalized["used_focus_frames"]).issubset(set(available_focus_frames))


def test_canonical_entity_live_normalization_accepts_evidence_ranks() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(entity_ids=["player_1"], limit=6),
        retrieval_mode="entity",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    available_evidence_lines = list(graph_result["llm_ready_bundle"]["evidence_lines"])

    normalized = responder._normalize_structured_answer(
        parsed={
            "answer_text": "player_1 changes behavior over time.",
            "used_focus_frames": [11],
            "used_evidence_ranks": [2, 1],
            "uncertainty": [],
        },
        graph_result=graph_result,
    )

    assert normalized is not None
    assert normalized["used_evidence_lines"][:2] == [
        available_evidence_lines[1],
        available_evidence_lines[0],
    ]


def test_canonical_entity_live_normalization_enriches_raw_entity_reference() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    graph_result = retrieve_canonical_graph_context(
        graph,
        MemoryQuery(entity_ids=["player_1"], limit=6),
        retrieval_mode="entity",
    )
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())

    normalized = responder._normalize_structured_answer(
        parsed={
            "answer_text": "player_1 changes behavior over time.",
            "used_focus_frames": [11],
            "used_evidence_ranks": [1, 2],
            "uncertainty": [],
        },
        graph_result=graph_result,
    )

    assert normalized is not None
    assert "human_001" in normalized["answer_text"]
    assert "player_1" in normalized["answer_text"]


def test_live_normalization_filters_unknown_time_anchor_ids() -> None:
    graph_result = {
        "query_type": "canonical_temporal",
        "matches": [],
        "retrieval_bundle": {
            "focus_frames": [
                {
                    "frame_index": 12,
                    "timestamp_sec": 1673884940.0,
                    "scene_id": "scene_001",
                    "reasons": ["query_match"],
                }
            ],
            "scene_summaries": [],
            "temporal_chains": [],
        },
        "llm_ready_bundle": {
            "answer_brief": "The event is visible around frame 12.",
            "evidence_lines": ["rank=1 :: event appears around frame 12."],
            "focus_frame_lines": ["frame=12 ts=1673884940.0"],
            "scene_lines": [],
            "structured_graph_context": {},
        },
    }
    responder = OpenAICompatibleGraphQAResponder(
        GraphQAConfig(target_evidence_lines=1, target_focus_frames=1)
    )

    normalized = responder._normalize_structured_answer(
        parsed={
            "answer_text": "The event happened about 1.0 minutes ago.",
            "used_focus_frames": [12],
            "used_evidence_ranks": [1],
            "used_time_anchor_ids": ["time_anchor:01", "time_anchor:99"],
            "uncertainty": [],
        },
        graph_result=graph_result,
        question="current_time_unix=1673885000.0",
    )

    assert normalized is not None
    assert normalized["used_time_anchor_ids"] == ["time_anchor:01"]


def test_normalize_selected_evidence_lines_drops_conflicting_action_object_phrase() -> None:
    available = [
        "rank=1 :: entity:000007 has state=holding basketball from 85.00s-95.00s.",
        "rank=2 :: entity:000023 has state=holding basketball from 85.00s.",
        "rank=3 :: entity:000007 has state=holding basketball from 215.00s.",
        "rank=4 :: entity:000007 has state=holding baseball bat from 140.00s.",
    ]

    normalized = _normalize_selected_evidence_lines(
        current=list(available),
        available=available,
        target_count=4,
        min_count=1,
        answer_text=(
            "human_004 (man, entity:000007) is holding the basketball from 85.00s "
            "to 95.00s and again at 215.00s."
        ),
    )

    assert available[0] in normalized
    assert available[1] in normalized
    assert available[2] in normalized
    assert available[3] not in normalized


def test_parse_json_content_accepts_json_wrapped_in_extra_text() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    parsed = responder._parse_json_content(
        'Here is the result:\n```json\n{"answer_text":"ok","used_focus_frames":[1],"used_evidence_ranks":[1],"uncertainty":[]}\n```\nThank you.'
    )

    assert parsed is not None
    assert parsed["answer_text"] == "ok"
    assert parsed["used_evidence_ranks"] == [1]


def test_answer_canonical_graph_query_live_handles_missing_api_key() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    temp_root = Path(".scratch/test-artifacts")
    temp_root.mkdir(parents=True, exist_ok=True)
    config_path = temp_root / f"canonical_graph_qa_test_{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        "\n".join(
            [
                'extends: "../../configs/default.yaml"',
                "graph_qa:",
                "  api_key_env_var: CANONICAL_GRAPH_QA_TEST_MISSING_KEY",
            ]
        ),
        encoding="utf-8",
    )
    os.environ.pop("CANONICAL_GRAPH_QA_TEST_MISSING_KEY", None)
    try:
        result = answer_canonical_graph_query(
            graph,
            MemoryQuery(query_text="near hoop", limit=4),
            question="Who is near whom?",
            retrieval_mode="temporal",
            answer_mode="live",
            config_path=str(config_path.resolve()),
        )
    finally:
        config_path.unlink(missing_ok=True)

    assert result["graph_answer"]["matches"]
    assert result["llm_answer"]["mode"] == "live"
    assert result["llm_answer"]["graph_rag_status"] == "live_request_error"
    assert "Missing environment variable" in result["llm_answer"]["error"]
