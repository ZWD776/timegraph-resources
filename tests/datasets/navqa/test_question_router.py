from __future__ import annotations

from vlkg.datasets.navqa.retrieval.question_router import (
    QUESTION_EVIDENCE_TAXONOMY_CONTRACT,
    QUESTION_ROUTE_CONTRACT,
    ROUTE_BINARY_EXISTENCE,
    ROUTE_COUNT_SET,
    ROUTE_EGO_MOTION_SIDE_TURN,
    ROUTE_ENTITY_ATTRIBUTE,
    ROUTE_OCR_LOGO_TEXT,
    ROUTE_TEMPORAL_INTERVAL,
    ROUTE_TEMPORAL_POINT,
    attach_question_route_to_ir,
    route_question,
)


def test_router_routes_duration_to_temporal_interval() -> None:
    route = route_question(
        "How long were you stopped near the gate?",
        official_question_type="duration",
        question_ir={"answer_type": "duration", "operation": "interval"},
    )

    assert route.contract == QUESTION_ROUTE_CONTRACT
    assert route.route_id == ROUTE_TEMPORAL_INTERVAL
    assert route.taxonomy_contract == QUESTION_EVIDENCE_TAXONOMY_CONTRACT
    assert route.taxonomy_path == "temporal.interval"
    assert route.evidence_family == "temporal_interval_estimation"
    assert route.answer_shape == "duration"
    assert route.time_requirements["requires_interval"] is True
    assert route.retrieval_policy == "target_entity_state_start_end_boundary"


def test_router_routes_when_to_temporal_point() -> None:
    route = route_question(
        "When did you enter the building?",
        official_question_type="time",
        question_ir={
            "answer_type": "time",
            "operation": "onset",
            "target_entities": ["building"],
        },
    )

    assert route.route_id == ROUTE_TEMPORAL_POINT
    assert route.taxonomy_path == "temporal.point"
    assert route.answer_shape == "time"
    assert "building" in route.primary_target_terms
    assert route.time_requirements["requires_point"] is True


def test_router_time_answer_type_beats_empty_attribute_field() -> None:
    route = route_question(
        "When did you leave the building?",
        official_question_type="time",
        question_ir={
            "answer_type": "time",
            "operation": "offset",
            "target_entities": ["building"],
            "target_state": "leave building",
            "attribute_dimension": "",
        },
    )

    assert route.route_id == ROUTE_TEMPORAL_POINT


def test_router_routes_attribute_question_from_ir() -> None:
    route = route_question(
        "What color was the car near the intersection?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["car"],
            "attribute_dimension": "color",
            "spatial_constraints": ["near the intersection"],
        },
    )

    assert route.route_id == ROUTE_ENTITY_ATTRIBUTE
    assert route.taxonomy_path == "non_temporal.entity.attribute"
    assert route.primary_target_terms == ("car",)
    assert "color" in route.attribute_terms
    assert "near the intersection" in route.spatial_constraints


def test_router_prefers_question_route_embedded_in_ir() -> None:
    route = route_question(
        "What color was the car near the intersection?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["car"],
            "attribute_dimension": "color",
            "question_route": {"route_id": ROUTE_ENTITY_ATTRIBUTE},
        },
    )

    assert route.route_id == ROUTE_ENTITY_ATTRIBUTE
    assert route.source == "llm_question_ir.question_route"
    assert "ir_question_route" in route.route_signals


def test_router_uses_evidence_question_type_from_ir() -> None:
    route = route_question(
        "How many doors did the person open?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "occurrence",
            "evidence_question_type": "non_temporal.entity.count",
            "target_entities": ["doors"],
        },
    )

    assert route.route_id == ROUTE_COUNT_SET
    assert route.source == "llm_question_ir.evidence_question_type"


def test_router_routes_count_question_to_entity_set() -> None:
    route = route_question(
        "How many doors did the person open?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "aggregation": "count",
            "target_entities": ["doors"],
            "event_constraints": ["person open doors"],
        },
    )

    assert route.route_id == ROUTE_COUNT_SET
    assert route.taxonomy_path == "non_temporal.entity.count"
    assert route.retrieval_policy == "time_condition_filtered_entity_set"
    assert "doors" in route.primary_target_terms
    assert "person open doors" in route.event_constraints


def test_router_how_many_occurrence_stays_count_set() -> None:
    route = route_question(
        "How many cars did you see that passed the gate?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "occurrence",
            "aggregation": "multiple",
            "target_entities": ["cars"],
            "event_constraints": ["passed the gate"],
        },
    )

    assert route.route_id == ROUTE_COUNT_SET


def test_router_multiple_attribute_answers_are_not_count_sets() -> None:
    route = route_question(
        "What color are the banners on the lamp posts?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "aggregation": "multiple",
            "target_entities": ["banners"],
            "attribute_dimension": "color",
        },
    )

    assert route.route_id == ROUTE_ENTITY_ATTRIBUTE


def test_router_routes_text_logo_requests_to_ocr_logo_text() -> None:
    route = route_question(
        "What word was written on the trailer?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "target_entities": ["trailer"],
            "evidence_requirements": ["read visible text or logo"],
        },
    )

    assert route.route_id == ROUTE_OCR_LOGO_TEXT
    assert route.taxonomy_path == "non_temporal.symbolic.ocr_logo"
    assert route.evidence_policy == "hsgg_text_logo_attribute_candidate_pack"


def test_router_plain_stop_sign_is_not_ocr_without_text_intent() -> None:
    route = route_question(
        "What color is the sidewalk next to the stop sign?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["sidewalk"],
            "attribute_dimension": "color",
            "spatial_constraints": ["next to stop sign"],
        },
    )

    assert route.route_id == ROUTE_ENTITY_ATTRIBUTE


def test_router_routes_side_turn_to_ego_motion() -> None:
    route = route_question(
        "Which side did you turn toward after leaving the building?",
        official_question_type="text",
        question_ir={"answer_type": "text", "qa_focus": "ego_direction"},
    )

    assert route.route_id == ROUTE_EGO_MOTION_SIDE_TURN
    assert route.taxonomy_path == "non_temporal.ego.direction"
    assert route.retrieval_policy == "ego_pose_heading_or_spatial_proxy"


def test_router_side_of_street_with_ego_actor_routes_to_ego_motion() -> None:
    route = route_question(
        "What side of the street are you driving on?",
        official_question_type="text",
        question_ir={
            "answer_type": "text",
            "operation": "spatial_relation",
            "target_entities": ["street"],
            "attribute_dimension": "side",
            "spatial_constraints": ["side_of", "street"],
        },
    )

    assert route.route_id == ROUTE_EGO_MOTION_SIDE_TURN


def test_router_keeps_plain_binary_as_binary_existence() -> None:
    route = route_question(
        "Did you pass a gate?",
        official_question_type="binary",
        question_ir={"answer_type": "binary", "target_entities": ["gate"]},
    )

    assert route.route_id == ROUTE_BINARY_EXISTENCE
    assert route.taxonomy_path == "non_temporal.entity.existence"
    assert route.answer_shape == "binary"


def test_router_binary_existence_beats_attribute_modifiers() -> None:
    route = route_question(
        "Did you see a red fire hydrant?",
        official_question_type="binary",
        question_ir={
            "answer_type": "binary",
            "operation": "existence_query",
            "target_entities": ["fire hydrant"],
            "attribute_dimension": "color",
            "target_state": "red fire hydrant",
        },
    )

    assert route.route_id == ROUTE_BINARY_EXISTENCE
    assert route.taxonomy_path == "non_temporal.entity.existence"


def test_router_right_now_does_not_trigger_ego_side() -> None:
    route = route_question(
        "Are you stopped in a loading zone right now?",
        official_question_type="binary",
        question_ir={
            "answer_type": "binary",
            "operation": "current_state",
            "target_entities": ["loading zone"],
            "attribute_dimension": "state",
            "target_state": "stopped in a loading zone",
        },
    )

    assert route.route_id == ROUTE_BINARY_EXISTENCE
    assert route.taxonomy_path == "non_temporal.state.object"


def test_router_records_compiler_conflict_when_ir_is_more_specific() -> None:
    route = route_question(
        "Did you pass a sign?",
        official_question_type="binary",
        question_ir={
            "answer_type": "binary",
            "operation": "attribute_query",
            "attribute_dimension": "text",
            "target_entities": ["sign"],
        },
    )

    assert route.route_id == ROUTE_OCR_LOGO_TEXT
    assert any("compiler_route=" in conflict for conflict in route.conflicts)


def test_attach_question_route_to_ir_preserves_original_fields() -> None:
    route = route_question(
        "When did you stop?",
        official_question_type="time",
        question_ir={"answer_type": "time", "operation": "onset"},
    )

    expanded = attach_question_route_to_ir({"answer_type": "time"}, route)

    assert expanded is not None
    assert expanded["answer_type"] == "time"
    assert expanded["question_route"]["route_id"] == ROUTE_TEMPORAL_POINT
    assert expanded["question_route"]["taxonomy_path"] == "temporal.point"


def test_router_binary_event_existence_keeps_route_but_marks_event_taxonomy() -> None:
    route = route_question(
        "Did the person open the door?",
        official_question_type="binary",
        question_ir={
            "answer_type": "binary",
            "operation": "existence_query",
            "target_entities": ["person", "door"],
            "event_constraints": ["person open door"],
        },
    )

    assert route.route_id == ROUTE_BINARY_EXISTENCE
    assert route.taxonomy_path == "non_temporal.event.existence"
    assert route.evidence_family == "event_existence_verification"
