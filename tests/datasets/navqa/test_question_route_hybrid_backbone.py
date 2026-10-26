from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalSceneNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
    CanonicalTransitionEventNode,
)
from vlkg.datasets.navqa.retrieval.question_route_hybrid_backbone import (
    attach_temporal_interval_boundary_pack,
    attach_temporal_point_candidate_pack,
    fuse_count_candidate_set_pack,
    fuse_question_route_hybrid_backbone,
    hybrid_visual_faiss_config,
    question_route_backbone_policy,
)
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import VisualFaissConfig


def _entity(entity_id: str, label: str, stable_id: str | None = None) -> CanonicalEntityNode:
    return CanonicalEntityNode(
        node_id=f"entity:{entity_id}",
        video_id="test",
        entity_id=entity_id,
        label=label,
        time_range=CanonicalTimeRange(start_frame=0, end_frame=20),
        observation_count=2,
        stable_identity_id=stable_id,
    )


def test_hybrid_backbone_policy_keeps_binary_on_ger_v3() -> None:
    policy = question_route_backbone_policy({"route_id": "binary_existence"})

    assert policy["route_backbone_id"] == "ger_v3_binding"
    assert policy["route_candidate_unit"] == "ger_bound_fact"


def test_specialized_backbone_policy_keeps_binary_on_exact_ger_v3() -> None:
    policy = question_route_backbone_policy(
        {"route_id": "binary_existence"},
        specialized=True,
    )

    assert policy["route_backbone_id"] == "ger_v3_binding"
    assert policy["route_candidate_unit"] == "ger_bound_fact"
    assert policy["route_strategy_scope"] == "stable_shared"
    assert policy["specialized_strategy_applied"] is False


def test_specialized_backbone_does_not_change_stable_ger_routes() -> None:
    stable_routes = [
        "binary_existence",
        "open_descriptive",
        "",
    ]
    behavior_keys = [
        "route_backbone_id",
        "route_candidate_unit",
        "evidence_builder_id",
        "binding_enabled",
        "attribute_owner_enabled",
        "object_local_facts_enabled",
        "temporal_interval_boundary_enabled",
        "temporal_point_candidate_enabled",
        "visual_faiss_entity_pack_query_enabled",
        "entity_pack_score_contract_enabled",
        "entity_pack_semantic_align_enabled",
        "entity_pack_semantic_gated_role_enabled",
        "count_candidate_set_enabled",
        "route_strategy_scope",
    ]

    for route_id in stable_routes:
        base = question_route_backbone_policy({"route_id": route_id})
        specialized = question_route_backbone_policy(
            {"route_id": route_id},
            specialized=True,
        )

        for key in behavior_keys:
            assert specialized[key] == base[key]
        assert specialized["specialized_strategy_applied"] is False


def test_specialized_backbone_marks_only_routes_with_specialized_strategy() -> None:
    count_policy = question_route_backbone_policy(
        {"route_id": "count_set"},
        specialized=True,
    )
    interval_policy = question_route_backbone_policy(
        {"route_id": "temporal_interval"},
        specialized=True,
    )
    attribute_policy = question_route_backbone_policy(
        {"route_id": "entity_attribute"},
        specialized=True,
    )

    assert count_policy["specialized_strategy_applied"] is True
    assert count_policy["binding_enabled"] is False
    assert interval_policy["specialized_strategy_applied"] is True
    assert interval_policy["binding_enabled"] is True
    assert attribute_policy["specialized_strategy_applied"] is False
    assert attribute_policy["binding_enabled"] is False
    assert attribute_policy["entity_pack_semantic_align_enabled"] is False
    assert attribute_policy["entity_pack_semantic_gated_role_enabled"] is False


def test_specialized_stable_route_config_disables_entity_pack_query() -> None:
    config = VisualFaissConfig(
        entity_pack_enabled=True,
        entity_pack_score_contract_enabled=True,
        entity_pack_semantic_align_enabled=True,
        entity_pack_semantic_gated_role_enabled=True,
    )

    updated = hybrid_visual_faiss_config(
        config,
        question_ir={"question_route": {"route_id": "binary_existence"}},
        specialized=True,
    )

    assert updated.entity_pack_enabled is False
    assert updated.entity_pack_score_contract_enabled is False
    assert updated.entity_pack_semantic_align_enabled is False
    assert updated.entity_pack_semantic_gated_role_enabled is False


def test_specialized_temporal_point_uses_route_scoped_candidate_pack() -> None:
    base = question_route_backbone_policy({"route_id": "temporal_point"})
    specialized = question_route_backbone_policy(
        {"route_id": "temporal_point"},
        specialized=True,
    )

    assert base["route_backbone_id"] == "ger_v3_binding"
    assert base["temporal_point_candidate_enabled"] is False
    assert specialized["route_backbone_id"] == "ger_temporal_point_candidate"
    assert specialized["route_candidate_unit"] == "temporal_point_candidate"
    assert specialized["temporal_point_candidate_enabled"] is True
    assert specialized["specialized_strategy_applied"] is True
    assert specialized["route_strategy_scope"] == "route_specific"


def test_specialized_binary_fusion_uses_plain_ger_binding_flags(monkeypatch) -> None:
    calls = []

    def fake_fuse(graph_answer, visual_payload, **kwargs):
        calls.append(kwargs)
        return graph_answer

    monkeypatch.setattr(
        "vlkg.question_route_hybrid_backbone.fuse_visual_faiss_ger_evidence",
        fake_fuse,
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": ["rank=1 fallback"]}}

    updated = fuse_question_route_hybrid_backbone(
        graph_answer,
        {"groups": []},
        graph=CanonicalTemporalMemoryGraph(video_id="test"),
        config=VisualFaissConfig(),
        question="Did you see a car?",
        question_ir={"question_route": {"route_id": "binary_existence"}},
        specialized=True,
    )

    assert len(calls) == 1
    assert calls[0]["binding_enabled"] is True
    assert calls[0]["attribute_owner_enabled"] is False
    assert calls[0]["object_local_facts_enabled"] is False
    metadata = updated["llm_ready_bundle"]["question_route_hybrid_backbone"]
    assert metadata["route_backbone_id"] == "ger_v3_binding"
    assert metadata["specialized_strategy_applied"] is False


def test_count_candidate_set_deduplicates_entity_across_frames() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            _entity("001", "car", stable_id="stable-car-a"),
            _entity("002", "car", stable_id="stable-car-b"),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "cars",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": "001",
                        "label": "car",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "score": 0.9,
                    },
                    {
                        "entity_id": "001",
                        "label": "car",
                        "graph_frame_index": 12,
                        "object_index": 3,
                        "score": 0.8,
                    },
                    {
                        "entity_id": "002",
                        "label": "car",
                        "graph_frame_index": 18,
                        "object_index": 2,
                        "score": 0.7,
                    },
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["cars"],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["cars"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many cars did you see?",
        question_ir=question_ir,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 2
    assert [item["entity_key"] for item in count_set["candidates"]] == [
        "stable-car-a",
        "stable-car-b",
    ]
    assert "count=2" in updated["llm_ready_bundle"]["question_route_count_set_lines"][0]
    assert "answer=2" in updated["llm_ready_bundle"]["answer_candidate_lines"][0]
    assert (
        updated["llm_ready_bundle"]["answer_candidate_extraction"]["candidates"][0][
            "answer"
        ]
        == "2"
    )
    assert updated["llm_ready_bundle"]["evidence_lines"][0].startswith(
        "kind=count_candidate_set"
    )


def test_count_candidate_set_filters_candidates_by_retrieval_window() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:001",
                video_id="test",
                entity_id="001",
                label="car",
                time_range=CanonicalTimeRange(
                    start_frame=10,
                    end_frame=20,
                    start_ts=100.0,
                    end_ts=110.0,
                ),
                observation_count=2,
                stable_identity_id="stable-car-in-window",
            ),
            CanonicalEntityNode(
                node_id="entity:002",
                video_id="test",
                entity_id="002",
                label="car",
                time_range=CanonicalTimeRange(
                    start_frame=300,
                    end_frame=320,
                    start_ts=300.0,
                    end_ts=310.0,
                ),
                observation_count=2,
                stable_identity_id="stable-car-outside-window",
            ),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "cars",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": "001",
                        "label": "car",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "score": 0.7,
                    },
                    {
                        "entity_id": "002",
                        "label": "car",
                        "graph_frame_index": 300,
                        "object_index": 2,
                        "score": 0.99,
                    },
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["cars"],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["cars"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many cars did you see near the curb?",
        question_ir=question_ir,
        retrieval_window={"applied": True, "start_ts": 95.0, "end_ts": 120.0},
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 1
    assert count_set["candidates"][0]["entity_key"] == "stable-car-in-window"
    assert count_set["time_window"]["applied"] is True
    assert count_set["excluded_items_reason"][0]["reason"] == (
        "outside_retrieval_time_window"
    )


def test_global_count_scope_does_not_apply_current_retrieval_window() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:001",
                video_id="test",
                entity_id="001",
                label="car",
                time_range=CanonicalTimeRange(start_ts=100.0, end_ts=110.0),
                observation_count=1,
                stable_identity_id="stable-car-a",
            ),
            CanonicalEntityNode(
                node_id="entity:002",
                video_id="test",
                entity_id="002",
                label="car",
                time_range=CanonicalTimeRange(start_ts=300.0, end_ts=310.0),
                observation_count=1,
                stable_identity_id="stable-car-b",
            ),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "cars",
                "query_phrase_role": "primary_target",
                "results": [
                    {"entity_id": "001", "label": "car", "score": 0.8},
                    {"entity_id": "002", "label": "car", "score": 0.7},
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["cars"],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["cars"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many cars did you see in total?",
        question_ir=question_ir,
        retrieval_window={"applied": True, "start_ts": 95.0, "end_ts": 120.0},
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 2
    assert count_set["time_window"]["applied"] is False
    assert count_set["time_window"]["reason"] == "global_count_scope"


def test_count_candidate_set_uses_explicit_quantity_phrase_as_answer_candidate() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:001",
                video_id="test",
                entity_id="001",
                label="parked cars",
                time_range=CanonicalTimeRange(start_frame=10, end_frame=10),
                observation_count=1,
                stable_identity_id="stable-car-group",
            )
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "cars",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": "001",
                        "label": "parked cars",
                        "visual_attribute": "Three parked cars in a parking lot",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "score": 0.8,
                    }
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["cars"],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["cars"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many cars did you see in total?",
        question_ir=question_ir,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 1
    assert count_set["explicit_quantity_answer"]["quantity"] == 3
    assert "answer=3" in updated["llm_ready_bundle"]["answer_candidate_lines"][0]


def test_specialized_count_set_filters_non_target_labels_before_counting() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            _entity("001", "door", stable_id="stable-door-a"),
            _entity("002", "building", stable_id="stable-building-a"),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "opened doors",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": "001",
                        "label": "door",
                        "visual_attribute": "open glass door",
                        "graph_frame_index": 10,
                        "object_index": 1,
                        "score": 0.9,
                    },
                    {
                        "entity_id": "002",
                        "label": "building",
                        "visual_attribute": "open building entrance",
                        "graph_frame_index": 12,
                        "object_index": 3,
                        "score": 0.95,
                    },
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["doors"],
        "hard_constraints": [{"text": "opened for letting you out"}],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["doors"],
            "event_constraints": ["opened for letting you out"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many doors did they open for letting you out?",
        question_ir=question_ir,
        specialized=True,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count_policy"] == "entity_set_count"
    assert count_set["count"] == 1
    assert count_set["candidate_items"][0]["label"] == "door"
    assert count_set["excluded_items_reason"][0]["reason"] == (
        "target_label_or_direct_attribute_mismatch"
    )


def test_specialized_event_count_rejects_weak_relation_and_uses_target_state_entity() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:door-1",
                video_id="test",
                entity_id="door-1",
                label="door",
                time_range=CanonicalTimeRange(start_frame=50, end_frame=60),
                observation_count=2,
                stable_identity_id="stable-door-1",
                states_seen=["open"],
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:weak:1",
                video_id="test",
                episode_id="weak-1",
                episode_kind="RELATION",
                label="lamp RAW_RELATION_LETTING_LIGHT_OUT window",
                time_range=CanonicalTimeRange(start_frame=40, end_frame=45),
                predicate="RAW_RELATION_LETTING_LIGHT_OUT",
                source_ref_id="weak-1",
            )
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "open door",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": "door-1",
                        "label": "door",
                        "visual_attribute": "open white door",
                        "graph_frame_index": 50,
                        "object_index": 1,
                        "score": 0.8,
                    }
                ],
            }
        ]
    }
    question_ir = {
        "hard_constraints": [{"text": "letting you out"}],
        "question_route": {
            "route_id": "count_set",
            "event_constraints": ["letting you out"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many door did they open for letting you out?",
        question_ir=question_ir,
        specialized=True,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count_policy"] == "event_episode_count"
    assert count_set["count"] == 1
    assert count_set["target_terms"] == ["door"]
    assert count_set["candidate_items"][0]["candidate_source"] == (
        "event_target_state_visual_entity"
    )
    assert count_set["candidate_items"][0]["entity_key"] == "stable-door-1"
    assert "weak-1" not in {
        str(item.get("event_id") or "") for item in count_set["candidate_items"]
    }


def test_specialized_count_set_uses_ego_action_policy_for_turn_counts() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:turn:1",
                video_id="test",
                episode_id="turn-1",
                episode_kind="RELATION",
                label="ego RAW_TURN_LEFT path",
                time_range=CanonicalTimeRange(start_frame=100, end_frame=110),
                predicate="RAW_TURN_LEFT",
                source_ref_id="turn-1",
            ),
            CanonicalEpisodeNode(
                node_id="episode:turn:2",
                video_id="test",
                episode_id="turn-2",
                episode_kind="RELATION",
                label="ego RAW_TURN_RIGHT path",
                time_range=CanonicalTimeRange(start_frame=400, end_frame=410),
                predicate="RAW_TURN_RIGHT",
                source_ref_id="turn-2",
            ),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    question_ir = {
        "target_state": "turns made",
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["turns made"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        {"groups": []},
        graph=graph,
        object_limit=8,
        question="How many turns did you make?",
        question_ir=question_ir,
        specialized=True,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count_policy"] == "ego_action_count"
    assert count_set["dedupe_key"] == "ego_action_frame_cluster"
    assert count_set["count"] == 2


def test_specialized_count_set_is_not_capped_by_object_limit() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        entity_nodes=[
            _entity(f"{index:03d}", "car", stable_id=f"stable-car-{index}")
            for index in range(10)
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "cars",
                "query_phrase_role": "primary_target",
                "results": [
                    {
                        "entity_id": f"{index:03d}",
                        "label": "car",
                        "graph_frame_index": index * 10,
                        "object_index": index,
                        "score": 1.0 - index * 0.01,
                    }
                    for index in range(10)
                ],
            }
        ]
    }
    question_ir = {
        "target_entities": ["cars"],
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["cars"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        object_limit=8,
        question="How many cars did you see in total?",
        question_ir=question_ir,
        specialized=True,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 10
    assert count_set["candidate_limit"] > 8
    assert "count=10" in updated["llm_ready_bundle"]["question_route_count_set_lines"][0]


def test_specialized_ego_action_count_rejects_spatial_relations() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:spatial:right",
                video_id="test",
                episode_id="spatial-right",
                episode_kind="RELATION",
                label="person RAW_RELATION_RIGHT_OF door",
                time_range=CanonicalTimeRange(start_frame=100, end_frame=110),
                predicate="RAW_RELATION_RIGHT_OF",
                source_ref_id="spatial-right",
            ),
            CanonicalEpisodeNode(
                node_id="episode:spatial:around",
                video_id="test",
                episode_id="spatial-around",
                episode_kind="RELATION",
                label="vine RAW_RELATION_GROWING_AROUND pole",
                time_range=CanonicalTimeRange(start_frame=160, end_frame=170),
                predicate="RAW_RELATION_GROWING_AROUND",
                source_ref_id="spatial-around",
            ),
            CanonicalEpisodeNode(
                node_id="episode:turn:left",
                video_id="test",
                episode_id="turn-left",
                episode_kind="RELATION",
                label="ego RAW_TURN_LEFT path",
                time_range=CanonicalTimeRange(start_frame=400, end_frame=410),
                predicate="RAW_TURN_LEFT",
                source_ref_id="turn-left",
            ),
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": []}}
    question_ir = {
        "target_state": "turns made",
        "question_route": {
            "route_id": "count_set",
            "primary_target_terms": ["turns made"],
        },
    }

    updated = fuse_count_candidate_set_pack(
        graph_answer,
        {"groups": []},
        graph=graph,
        object_limit=8,
        question="How many turns did you make?",
        question_ir=question_ir,
        specialized=True,
    )

    count_set = updated["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count_policy"] == "ego_action_count"
    assert count_set["count"] == 1
    assert count_set["candidate_items"][0]["entity_key"].startswith(
        "ego_action_cluster"
    )


def test_temporal_point_candidate_pack_selects_occurrence_candidate() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene:1",
                video_id="test",
                scene_id="scene:1",
                label="street crossing scene",
                time_range=CanonicalTimeRange(
                    start_frame=0,
                    end_frame=200,
                    start_ts=1000.0,
                    end_ts=1020.0,
                ),
            )
        ],
    )
    graph_answer = {"llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]}}
    visual_payload = {
        "groups": [
            {
                "query_phrase": "a photo of crossing street",
                "results": [
                    {
                        "graph_frame_index": 100,
                        "label": "pedestrian street crossing",
                        "visual_attribute": "Pedestrian crossing road",
                        "similarity": 0.2,
                        "score": 0.9,
                    }
                ],
            }
        ]
    }

    updated = attach_temporal_point_candidate_pack(
        graph_answer,
        visual_payload,
        graph=graph,
        question="When did you cross the street?",
        question_ir={
            "operation": "occurrence",
            "target_state": "crossing street",
            "target_entities": ["street"],
            "question_route": {
                "route_id": "temporal_point",
                "primary_target_terms": ["street", "crossing street"],
            },
        },
        retrieval_window={"start_ts": 1000.0, "end_ts": 1020.0},
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_point"]
    assert pack["selection_status"] == "selected"
    assert pack["selected_candidate"]["frame"] == 100
    assert pack["selected_candidate"]["timestamp_sec"] == 1010.0
    assert updated["llm_ready_bundle"]["evidence_lines"][0].startswith(
        "kind=temporal_point_candidate_pack"
    )
    assert updated["llm_ready_bundle"]["answer_candidate_lines"][0].startswith(
        "kind=answer_candidate source=temporal_point_candidate"
    )


def test_temporal_point_candidate_pack_does_not_select_boundary_operations() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene:1",
                video_id="test",
                scene_id="scene:1",
                label="building scene",
                time_range=CanonicalTimeRange(
                    start_frame=0,
                    end_frame=200,
                    start_ts=1000.0,
                    end_ts=1020.0,
                ),
            )
        ],
    )

    updated = attach_temporal_point_candidate_pack(
        {"llm_ready_bundle": {"evidence_lines": []}},
        {
            "groups": [
                {
                    "query_phrase": "a photo of leaving the building",
                    "results": [
                        {
                            "graph_frame_index": 120,
                            "label": "building",
                            "visual_attribute": "Building entrance",
                            "similarity": 0.2,
                            "score": 0.9,
                        }
                    ],
                }
            ]
        },
        graph=graph,
        question="When did you leave the building?",
        question_ir={
            "operation": "offset",
            "target_state": "leave building",
            "target_entities": ["building"],
            "question_route": {"route_id": "temporal_point"},
        },
        retrieval_window={"start_ts": 1000.0, "end_ts": 1020.0},
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_point"]
    assert pack["selection_status"] == "unsupported_operation"
    assert pack["selected_candidate"] == {}
    assert "answer_candidate_lines" not in updated["llm_ready_bundle"]

    updated_onset = attach_temporal_point_candidate_pack(
        {"llm_ready_bundle": {"evidence_lines": []}},
        {
            "groups": [
                {
                    "query_phrase": "a photo of first entering the building",
                    "results": [
                        {
                            "graph_frame_index": 120,
                            "label": "building",
                            "visual_attribute": "Building entrance",
                            "similarity": 0.2,
                            "score": 0.9,
                        }
                    ],
                }
            ]
        },
        graph=graph,
        question="When did you first enter the building?",
        question_ir={
            "operation": "onset",
            "target_state": "first enter building",
            "target_entities": ["building"],
            "question_route": {"route_id": "temporal_point"},
        },
        retrieval_window={"start_ts": 1000.0, "end_ts": 1020.0},
    )
    onset_pack = updated_onset["llm_ready_bundle"]["question_route_temporal_point"]
    assert onset_pack["selection_status"] == "unsupported_operation"
    assert onset_pack["selected_candidate"] == {}
    assert "answer_candidate_lines" not in updated_onset["llm_ready_bundle"]


def test_specialized_temporal_interval_adds_paired_boundary_candidate() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="test")
    graph_answer = {
        "llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]},
        "retrieval_bundle": {
            "semantic_event_candidates": [
                {
                    "event_type": "enter",
                    "boundary_frame": 100,
                    "summary": "enter building",
                    "score": 5.0,
                },
                {
                    "event_type": "leave",
                    "boundary_frame": 220,
                    "summary": "leave building",
                    "score": 5.0,
                },
            ]
        },
    }

    updated = attach_temporal_interval_boundary_pack(
        graph_answer,
        graph=graph,
        question="How long were you in the building for?",
        question_ir={
            "target_entities": ["building"],
            "question_route": {"route_id": "temporal_interval"},
        },
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_interval"]
    assert pack["interval_policy"] == "paired_boundary"
    assert pack["paired_duration_candidates"][0]["start_frame"] == 100
    assert pack["paired_duration_candidates"][0]["end_frame"] == 220
    structured = updated["llm_ready_bundle"]["structured_graph_context"]
    assert structured["temporal_span_candidates"][0]["source"] == (
        "question_route_paired_boundary"
    )


def test_specialized_temporal_interval_pairs_nearest_start_with_state_confirmation() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:inside",
                video_id="test",
                episode_id="inside-1",
                episode_kind="RELATION",
                label="person RAW_RELATION_INSIDE room",
                time_range=CanonicalTimeRange(start_frame=620, end_frame=620),
                predicate="RAW_RELATION_INSIDE",
                raw_predicate="inside",
                source_ref_id="inside-1",
            )
        ],
    )
    graph_answer = {
        "llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]},
        "retrieval_bundle": {
            "temporal_span_candidates": [
                {
                    "source_kind": "transition_event",
                    "start_frame": 100,
                    "end_frame": 100,
                    "score": 9.0,
                    "summary": "Transition event RELATION_START near room",
                },
                {
                    "source_kind": "transition_event",
                    "start_frame": 500,
                    "end_frame": 500,
                    "score": 8.0,
                    "summary": "Transition event RELATION_START near room",
                },
            ]
        },
    }

    updated = attach_temporal_interval_boundary_pack(
        graph_answer,
        graph=graph,
        question="How long were you in the room?",
        question_ir={
            "target_entities": ["room"],
            "target_state": "inside room",
            "question_route": {"route_id": "temporal_interval"},
        },
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_interval"]
    first = pack["paired_duration_candidates"][0]
    assert pack["interval_policy"] == "state_confirmation_pair"
    assert first["source"] == "question_route_state_confirmation_pair"
    assert first["start_frame"] == 500
    assert first["end_frame"] == 620


def test_specialized_temporal_interval_rejects_short_state_confirmation_flicker() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:inside",
                video_id="test",
                episode_id="inside-1",
                episode_kind="RELATION",
                label="person RAW_RELATION_INSIDE building",
                time_range=CanonicalTimeRange(start_frame=516, end_frame=516),
                predicate="RAW_RELATION_INSIDE",
                raw_predicate="inside",
                source_ref_id="inside-1",
            )
        ],
    )
    graph_answer = {
        "llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]},
        "retrieval_bundle": {
            "temporal_span_candidates": [
                {
                    "source_kind": "transition_event",
                    "start_frame": 500,
                    "end_frame": 500,
                    "score": 9.0,
                    "summary": "Transition event RELATION_START near building",
                },
                {
                    "source_kind": "transition_event",
                    "start_frame": 700,
                    "end_frame": 700,
                    "score": 8.0,
                    "summary": "Transition event RELATION_END near building",
                },
            ]
        },
    }

    updated = attach_temporal_interval_boundary_pack(
        graph_answer,
        graph=graph,
        question="How long were you in the building?",
        question_ir={
            "target_entities": ["building"],
            "target_state": "inside building",
            "question_route": {"route_id": "temporal_interval"},
        },
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_interval"]
    first = pack["paired_duration_candidates"][0]
    assert pack["interval_policy"] == "paired_boundary"
    assert first["source"] == "question_route_paired_boundary"
    assert first["start_frame"] == 500
    assert first["end_frame"] == 700


def test_specialized_temporal_interval_pairs_travel_to_window_start_with_arrival_state() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        scene_nodes=[
            CanonicalSceneNode(
                node_id="scene:1",
                video_id="test",
                scene_id="scene:1",
                label="route scene",
                time_range=CanonicalTimeRange(
                    start_frame=0,
                    end_frame=1000,
                    start_ts=0.0,
                    end_ts=100.0,
                ),
            )
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:inside",
                video_id="test",
                episode_id="inside-1",
                episode_kind="RELATION",
                label="person RAW_RELATION_INSIDE destination",
                time_range=CanonicalTimeRange(
                    start_frame=700,
                    end_frame=700,
                    start_ts=70.0,
                    end_ts=70.0,
                ),
                predicate="RAW_RELATION_INSIDE",
                raw_predicate="inside",
                source_ref_id="inside-1",
            )
        ],
    )
    graph_answer = {
        "llm_ready_bundle": {"evidence_lines": []},
        "retrieval_bundle": {},
    }

    updated = attach_temporal_interval_boundary_pack(
        graph_answer,
        graph=graph,
        question="How long did it take you to get to the destination?",
        question_ir={
            "target_entities": ["destination"],
            "target_state": "getting to the destination",
            "question_route": {"route_id": "temporal_interval"},
        },
        retrieval_window={"applied": True, "start_ts": 10.0, "end_ts": 90.0},
    )

    first = updated["llm_ready_bundle"]["question_route_temporal_interval"][
        "paired_duration_candidates"
    ][0]
    assert first["source"] == "question_route_window_to_state_confirmation"
    assert first["start_frame"] == 100
    assert first["end_frame"] == 700


def test_specialized_temporal_interval_does_not_scan_graph_transition_boundaries() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="test",
        transition_event_nodes=[
            CanonicalTransitionEventNode(
                node_id="transition:start",
                video_id="test",
                event_id="transition:start",
                event_type="APPEAR",
                label="building APPEAR",
                time_range=CanonicalTimeRange(start_frame=100, end_frame=100),
                entity_id="entity:building",
                source_ref_id="transition:start",
            ),
            CanonicalTransitionEventNode(
                node_id="transition:end",
                video_id="test",
                event_id="transition:end",
                event_type="DISAPPEAR",
                label="building DISAPPEAR",
                time_range=CanonicalTimeRange(start_frame=300, end_frame=300),
                entity_id="entity:building",
                source_ref_id="transition:end",
            ),
        ],
    )
    graph_answer = {
        "llm_ready_bundle": {"evidence_lines": ["rank=1 graph fallback"]},
        "retrieval_bundle": {},
    }

    updated = attach_temporal_interval_boundary_pack(
        graph_answer,
        graph=graph,
        question="How long were you in the building?",
        question_ir={
            "target_entities": ["building"],
            "target_state": "inside building",
            "question_route": {"route_id": "temporal_interval"},
        },
    )

    pack = updated["llm_ready_bundle"]["question_route_temporal_interval"]
    assert pack["start_boundary_candidates"] == []
    assert pack["end_boundary_candidates"] == []
    assert pack["paired_duration_candidates"] == []
