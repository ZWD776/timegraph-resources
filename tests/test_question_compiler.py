from __future__ import annotations

from vlkg.core.graph_retrieval.question_compiler import compile_question


def test_compiler_extracts_generic_attribute_without_domain_targets() -> None:
    plan = compile_question(
        "Where did you see a red object?",
        answer_type="position",
        category="spatial",
    )

    assert plan.metadata["domain_profile"] == "generic"
    assert plan.answer_type == "position"
    assert plan.query_family == "position_target"
    assert plan.target_entities == ()
    assert "red" in plan.attributes
    assert plan.answer_rendering == "ego_pose_at_retrieved_event_time"
    assert plan.retrieval_strategy == "target_entity_scene_then_event_time"


def test_compiler_default_profile_is_generic() -> None:
    plan = compile_question(
        "Where did you see a red object?",
        answer_type="position",
        category="spatial",
    )

    assert plan.metadata["domain_profile"] == "generic"
    assert plan.target_entities == ()
    assert "red" in plan.attributes
    assert plan.query_family == "position_target"


def test_compiler_rejects_removed_domain_profile() -> None:
    try:
        compile_question(
            "Where did you see a red object?",
            answer_type="position",
            category="spatial",
            domain_profile="urban_navigation",
        )
    except ValueError as exc:
        assert "unknown domain profile" in str(exc).lower()
    else:
        raise AssertionError("urban_navigation profile should not be available")


def test_compiler_recognizes_person_as_generic_target() -> None:
    plan = compile_question(
        "What was the appearance of the person?",
        answer_type="text",
        category="attribute",
    )

    assert plan.metadata["domain_profile"] == "generic"
    assert "person" in plan.target_entities
    assert "appearance" in plan.attributes
    assert plan.query_family == "attribute"


def test_compiler_keeps_generic_relative_event_constraint_without_exact_phrase() -> None:
    plan = compile_question(
        "What color was the chair near where I left the building?",
        answer_type="text",
        category="descriptive",
        domain_profile="generic",
    )

    assert "near_reference_event" in plan.spatial_constraints
    assert "leave" in plan.events


def test_compiler_marks_right_now_questions_as_current_window() -> None:
    plan = compile_question(
        "Are you stopped right now?",
        answer_type="binary",
        category="environmental",
    )

    assert plan.answer_type == "binary"
    assert plan.query_family == "current_state"
    assert "stop" in plan.events
    assert plan.current_window_required is True
    assert plan.retrieval_strategy == "question_window_scene_first"
    assert "must_not_answer_from_anywhere_in_video" in plan.risk_notes


def test_compiler_routes_enter_building_to_temporal_axis() -> None:
    plan = compile_question(
        "When did you first enter the building?",
        answer_type="time",
        category="temporal",
    )

    assert plan.answer_type == "time"
    assert plan.query_family == "temporal_event"
    assert "enter" in plan.events
    assert "earliest" in plan.temporal_constraints
    assert "event_time" in plan.temporal_constraints
    assert plan.retrieval_strategy == "temporal_axis_then_target_filter"
    assert "earliest_event" in plan.ranking_policy


def test_compiler_routes_closest_place_to_sit_with_ranking_policy() -> None:
    plan = compile_question(
        "Where is the closest place to sit?",
        answer_type="position",
        category="spatial",
    )

    assert "closest" in plan.spatial_constraints
    assert "distance_or_recency_proxy" in plan.ranking_policy
    assert "target_event_time_then_pose_render" in plan.ranking_policy
    assert "closest_requires_spatial_pose_or_proxy_ranking" in plan.risk_notes


def test_compiler_routes_ego_direction_to_motion_family() -> None:
    plan = compile_question(
        "Which direction did you turn after leaving the building?",
        answer_type="text",
        category="ego",
    )

    assert plan.query_family == "ego_motion"
    assert "turn" in plan.events
    assert "leave" in plan.events
    assert "after_event" in plan.temporal_constraints
    assert plan.retrieval_strategy == "ego_event_or_scene_transition"


def test_compiler_routes_appearance_question_to_attribute_binding() -> None:
    plan = compile_question(
        "What was the appearance of the person?",
        answer_type="text",
        category="attribute",
    )

    assert plan.query_family == "attribute"
    assert "person" in plan.target_entities
    assert "appearance" in plan.attributes
    assert plan.retrieval_strategy == "target_entity_attribute_binding"
    assert "attribute_must_bind_to_target_entity" in plan.risk_notes
