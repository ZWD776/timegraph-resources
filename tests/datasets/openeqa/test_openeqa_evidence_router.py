from __future__ import annotations

import pytest

from vlkg.datasets.openeqa.retrieval.evidence_router import (
    EVIDENCE_NEEDS,
    NEED_ENTITY_ATTRIBUTE,
    NEED_EXISTENCE_OR_STATE,
    NEED_FUNCTIONAL_WORLD,
    NEED_COUNT_SET,
    NEED_OBJECT_IDENTITY,
    NEED_OCR_LOGO_TEXT,
    NEED_OPEN_DESCRIPTIVE,
    NEED_SPATIAL_LOCALIZATION,
    OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS,
    OPEN_EQA_V0_OFFICIAL_CATEGORIES,
    ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3,
    ROUTING_PROFILE_IR_FIRST_V4,
    canonical_route_id_for_need,
    open_eqa_v0_route_taxonomy,
    compile_evidence_demand,
    question_route_decision_from_ir,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import route_handler_id
from vlkg.datasets.navqa.retrieval.question_router import ROUTE_ENTITY_ATTRIBUTE


def test_openeqa_v0_route_taxonomy_is_seven_active_routes_plus_fallback() -> None:
    taxonomy = open_eqa_v0_route_taxonomy()

    assert len(OPEN_EQA_V0_OFFICIAL_CATEGORIES) == 7
    assert len(OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS) == 7
    assert NEED_OPEN_DESCRIPTIVE not in OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS
    assert OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS | {NEED_OPEN_DESCRIPTIVE} == EVIDENCE_NEEDS
    assert taxonomy["official_category_count"] == 7
    assert taxonomy["active_route_count"] == 7
    assert taxonomy["fallback_route"] == NEED_OPEN_DESCRIPTIVE
    assert taxonomy["temporal_route_enabled"] is False


def test_openeqa_router_has_no_temporal_route() -> None:
    demand = compile_evidence_demand(
        "Where did I leave my medicine?",
        category="object localization",
        question_ir={
            "target_entities": ["medicine"],
            "temporal_relation": "current",
            "evidence_question_type": "temporal.point",
        },
    )

    assert demand.primary_need == NEED_SPATIAL_LOCALIZATION
    assert demand.temporal_route_enabled is False
    assert all("temporal" not in value for value in EVIDENCE_NEEDS)


def test_turned_on_is_state_not_temporal() -> None:
    demand = compile_evidence_demand(
        "Is the lamp turned on?",
        category="object state recognition",
        question_ir={"target_entities": ["lamp"], "target_state": "turned on"},
    )

    assert demand.primary_need == NEED_EXISTENCE_OR_STATE
    assert "turned on" in demand.state_terms
    assert demand.requires_negative_coverage is True


def test_route_and_evidence_demand_are_one_persisted_decision() -> None:
    decision = compile_evidence_demand(
        "What color is the chair?",
        category="attribute recognition",
        question_ir={"target_entities": ["chair"], "attribute_dimension": "color"},
    ).to_dict()
    ir = {"question_route": decision}

    assert decision["route_id"] == ROUTE_ENTITY_ATTRIBUTE
    assert decision["route_id"] == canonical_route_id_for_need(decision["primary_need"])
    assert question_route_decision_from_ir(ir) == decision
    assert "evidence_demand" not in ir


def test_official_categories_select_evidence_backbones() -> None:
    cases = {
        "object recognition": NEED_OBJECT_IDENTITY,
        "attribute recognition": NEED_ENTITY_ATTRIBUTE,
        "object localization": NEED_SPATIAL_LOCALIZATION,
        "functional reasoning": NEED_FUNCTIONAL_WORLD,
        "world knowledge": NEED_FUNCTIONAL_WORLD,
    }
    for category, expected in cases.items():
        demand = compile_evidence_demand(
            "What is this object?",
            category=category,
            question_ir={"target_entities": ["object"]},
        )
        assert demand.primary_need == expected


def test_spatial_anchors_are_not_promoted_to_answer_targets() -> None:
    demand = compile_evidence_demand(
        "What is the white object on the wall above the TV?",
        category="object recognition",
        question_ir={
            "target_entities": ["white object", "wall", "TV"],
            "attribute_dimension": "type",
            "spatial_constraints": [
                {"relation": "on", "target": "wall"},
                {"relation": "above", "target": "TV"},
            ],
        },
    )

    assert demand.target_terms == ("white object",)
    assert demand.anchor_terms == ("wall", "TV")
    assert demand.spatial_relations == ("on", "above")
    assert demand.spatial_constraints == (
        {"anchor": "wall", "relation": "on"},
        {"anchor": "TV", "relation": "above"},
    )
    assert demand.open_target_slot is True
    roles = {intent.role for intent in demand.query_intents}
    assert {"target", "anchor"}.issubset(roles)
    anchor_relations = {
        intent.text: intent.spatial_relation
        for intent in demand.query_intents
        if intent.role == "anchor"
    }
    assert anchor_relations == {"wall": "on", "TV": "above"}
    assert all("visual" not in intent.text.lower() for intent in demand.query_intents)


def test_open_spatial_answer_slot_is_explicit() -> None:
    demand = compile_evidence_demand(
        "What is to the left of the mirror?",
        category="spatial understanding",
        question_ir={
            "target_entities": ["mirror"],
            "spatial_constraints": [{"relation": "left of", "target": "mirror"}],
        },
    )

    assert demand.target_terms == ()
    assert demand.anchor_terms == ("mirror",)
    assert demand.spatial_relations == ("left_of",)
    assert demand.open_target_slot is True


def test_surface_capacity_uses_state_coverage_evidence() -> None:
    demand = compile_evidence_demand(
        "Does the table with the monitor have space for a few more cups?",
        category="spatial understanding",
        question_ir={"target_entities": ["table"]},
    )

    assert demand.primary_need == NEED_EXISTENCE_OR_STATE
    assert demand.open_target_slot is True
    assert demand.requires_negative_coverage is True
    assert demand.metadata["surface_capacity_query"] is True


@pytest.mark.parametrize(
    ("question", "question_ir", "expected"),
    [
        (
            "Is there room on the dining table to eat?",
            {"operation": "functional_query", "target_entities": ["dining table"]},
            "surface_capacity",
        ),
        (
            "Is the front door open?",
            {
                "operation": "attribute_query",
                "target_entities": ["front door"],
                "target_state": "open",
            },
            "open_closed_state",
        ),
        (
            "Are the ceiling lights turned on?",
            {
                "operation": "state_query",
                "target_entities": ["ceiling lights"],
                "target_state": "turned on",
            },
            "power_state",
        ),
        (
            "Is there a shower curtain in the bathroom?",
            {
                "operation": "existence_query",
                "target_entities": ["shower curtain"],
            },
            "binary_existence",
        ),
    ],
)
def test_v3_state_subvariants_are_deterministic(
    question: str,
    question_ir: dict,
    expected: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="object state recognition",
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3,
    )

    assert demand.primary_need == NEED_EXISTENCE_OR_STATE
    assert demand.strategy_variant == expected
    assert demand.metadata["routing_profile"] == (
        ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3
    )


def test_v2_state_variant_remains_legacy_for_manifest_compatibility() -> None:
    demand = compile_evidence_demand(
        "Is the front door open?",
        category="object state recognition",
        question_ir={
            "operation": "attribute_query",
            "target_entities": ["front door"],
            "target_state": "open",
        },
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert demand.strategy_variant == "state_or_existence"


@pytest.mark.parametrize(
    ("question", "category", "question_ir", "expected_unit"),
    [
        (
            "What object is beside the table?",
            "object recognition",
            {"target_entities": ["chair", "table"]},
            "canonical_entity_tracklet",
        ),
        (
            "What color is the chair?",
            "attribute recognition",
            {"target_entities": ["chair"], "attribute_dimension": "color"},
            "owner_attribute",
        ),
        (
            "Is the door open?",
            "object state recognition",
            {"target_entities": ["door"], "target_state": "open"},
            "owner_attribute",
        ),
        (
            "What is to the left of the table?",
            "spatial understanding",
            {
                "target_entities": ["table"],
                "spatial_constraints": [{"relation": "left of", "target": "table"}],
            },
            "scene_frame",
        ),
        (
            "How many chairs are there?",
            "object recognition",
            {"target_entities": ["chair"], "aggregation": "count"},
            "candidate_set",
        ),
        (
            "What text is written on the sign?",
            "attribute recognition",
            {"target_entities": ["sign"], "attribute_dimension": "text"},
            "owner_attribute",
        ),
        (
            "What can I use to sit down?",
            "functional reasoning",
            {"target_entities": ["chair"]},
            "scene_frame",
        ),
    ],
)
def test_each_evidence_route_emits_its_candidate_unit(
    question: str,
    category: str,
    question_ir: dict,
    expected_unit: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category=category,
        question_ir=question_ir,
    )

    assert expected_unit in {intent.evidence_unit for intent in demand.query_intents}
    assert demand.temporal_route_enabled is False


def test_v2_category_grounding_prevents_broad_functional_phrase_override() -> None:
    demand = compile_evidence_demand(
        "Where can I find the pink bowl?",
        category="object localization",
        question_ir={"operation": "location_query", "target_entities": ["pink bowl"]},
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert demand.primary_need == NEED_SPATIAL_LOCALIZATION
    assert demand.strategy_variant == "target_location"
    assert demand.source == "official_question_category"
    assert demand.contract == "openeqa_question_route_decision.v2"


def test_v2_placement_feasibility_is_isolated_from_v1_routing() -> None:
    question = "Can I put something big under my bed?"
    question_ir = {
        "operation": "spatial_relation",
        "target_entities": ["something", "bed"],
    }

    v1 = compile_evidence_demand(
        question,
        category="spatial understanding",
        question_ir=question_ir,
    )
    v2 = compile_evidence_demand(
        question,
        category="spatial understanding",
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert v1.primary_need == NEED_FUNCTIONAL_WORLD
    assert v2.primary_need == NEED_EXISTENCE_OR_STATE
    assert v2.metadata["surface_capacity_query"] is True


def test_v2_where_can_i_place_remains_functional() -> None:
    demand = compile_evidence_demand(
        "Where can I put my jacket?",
        category="functional reasoning",
        question_ir={"operation": "functional_query", "target_entities": ["jacket"]},
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert demand.primary_need == NEED_FUNCTIONAL_WORLD
    assert demand.metadata["surface_capacity_query"] is False


def test_v2_spatial_variants_keep_open_and_closed_targets_separate() -> None:
    target_location = compile_evidence_demand(
        "Where is the tissue box?",
        category="object localization",
        question_ir={"operation": "position_query", "target_entities": ["tissue box"]},
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )
    open_relation = compile_evidence_demand(
        "What is to the left of the mirror?",
        category="spatial understanding",
        question_ir={
            "operation": "spatial_relation",
            "target_entities": ["mirror"],
            "spatial_constraints": [{"relation": "left of", "target": "mirror"}],
        },
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert target_location.strategy_variant == "target_location"
    assert open_relation.strategy_variant == "open_target_relation"


@pytest.mark.parametrize(
    ("question", "category", "expected"),
    [
        (
            "How many chairs are there in the scene?",
            "object recognition",
            "direct_or_structural_entity_count",
        ),
        (
            "How many chairs are in this room?",
            "object recognition",
            "direct_or_structural_entity_count",
        ),
        (
            "How many boxes are on the fourth shelf?",
            "spatial understanding",
            "filtered_relation_attribute_count",
        ),
        (
            "How many coat hooks are available?",
            "object state recognition",
            "filtered_relation_attribute_count",
        ),
        (
            "How many burners are on the stove?",
            "attribute recognition",
            "direct_or_structural_entity_count",
        ),
        (
            "How many people can sit on chairs in this room?",
            "functional reasoning",
            "capacity_or_functional_count",
        ),
    ],
)
def test_v2_count_variants_are_persisted(
    question: str,
    category: str,
    expected: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category=category,
        question_ir={
            "operation": "count_query",
            "aggregation": "count",
            "attribute_dimension": "state" if "available" in question else "",
        },
        routing_profile=ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2,
    )

    assert demand.primary_need == "count_set"
    assert demand.strategy_variant == expected


@pytest.mark.parametrize(
    ("question", "question_ir", "expected_need"),
    [
        (
            "What brand is shown on the appliance?",
            {"operation": "attribute_query", "attribute_dimension": "brand"},
            NEED_OCR_LOGO_TEXT,
        ),
        (
            "How many chairs are there?",
            {"operation": "count_query", "answer_type": "count"},
            NEED_COUNT_SET,
        ),
        (
            "Which objects form the counted set?",
            {"operation": "count_query", "answer_type": "entity"},
            NEED_OBJECT_IDENTITY,
        ),
        (
            "Is the door open?",
            {"operation": "current_state", "answer_type": "binary"},
            NEED_EXISTENCE_OR_STATE,
        ),
        (
            "Are the lights on?",
            {"operation": "state_query", "answer_type": "binary"},
            NEED_EXISTENCE_OR_STATE,
        ),
        (
            "Is there a chair?",
            {"operation": "existence_query", "answer_type": "binary"},
            NEED_EXISTENCE_OR_STATE,
        ),
        (
            "Is the lamp switched on?",
            {"operation": "attribute_query", "attribute_dimension": "state"},
            NEED_EXISTENCE_OR_STATE,
        ),
        (
            "What color is the chair?",
            {"operation": "attribute_query", "attribute_dimension": "color"},
            NEED_ENTITY_ATTRIBUTE,
        ),
        (
            "Where is the chair?",
            {"operation": "location_query", "answer_type": "location"},
            NEED_SPATIAL_LOCALIZATION,
        ),
        (
            "What position is the chair in?",
            {"operation": "position_query", "answer_type": "location"},
            NEED_SPATIAL_LOCALIZATION,
        ),
        (
            "Is the chair beside the table?",
            {"operation": "spatial_relation", "answer_type": "binary"},
            NEED_SPATIAL_LOCALIZATION,
        ),
        (
            "Which direction leads to the kitchen?",
            {"operation": "navigation_direction", "answer_type": "direction"},
            NEED_SPATIAL_LOCALIZATION,
        ),
        (
            "What is beside the table?",
            {"operation": "identity_query", "spatial_constraints": [
                {"relation": "beside", "target": "table"}
            ]},
            NEED_OBJECT_IDENTITY,
        ),
        (
            "What is the appliance used for?",
            {"operation": "functional_query", "answer_type": "text"},
            NEED_FUNCTIONAL_WORLD,
        ),
        (
            "Why are sinks commonly found in kitchens?",
            {"operation": "world_knowledge_query", "answer_type": "text"},
            NEED_FUNCTIONAL_WORLD,
        ),
        (
            "Describe the room.",
            {"operation": "descriptive_query", "answer_type": "text"},
            NEED_OPEN_DESCRIPTIVE,
        ),
        (
            "Answer this unsupported request.",
            {"operation": "unsupported", "answer_type": "text"},
            NEED_OPEN_DESCRIPTIVE,
        ),
    ],
)
def test_ir_first_v4_operation_mapping(
    question: str,
    question_ir: dict,
    expected_need: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="deliberately conflicting category",
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == expected_need
    assert demand.source.startswith("question_ir")
    assert demand.metadata["category"] == ""
    assert demand.metadata["official_category_used_for_routing"] is False
    assert demand.metadata["routing_input_contract"] == (
        "question_and_semantic_ir_only"
    )


def test_ir_first_v4_category_counterfactual_is_route_handler_invariant() -> None:
    question = "What is beside the table?"
    question_ir = {
        "operation": "identity_query",
        "answer_type": "entity",
        "target_entities": ["object", "table"],
        "spatial_constraints": [{"relation": "beside", "target": "table"}],
    }
    signatures = []
    for category in ("object recognition", "", "random category 8472"):
        demand = compile_evidence_demand(
            question,
            category=category,
            question_ir=question_ir,
            routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
        )
        signatures.append(
            (
                demand.primary_need,
                demand.route_id,
                demand.strategy_variant,
                route_handler_id(demand.to_dict()),
            )
        )

    assert len(set(signatures)) == 1
    assert signatures[0] == (
        NEED_OBJECT_IDENTITY,
        "open_descriptive",
        "anchor_bound_open_target_identity",
        "object_identity.anchor_bound_relation_incremental_v1",
    )


def test_ir_first_v4_surface_capacity_requires_capacity_binary_ir() -> None:
    capacity = compile_evidence_demand(
        "Is there room on the table for another plate?",
        category="spatial understanding",
        question_ir={
            "operation": "functional_query",
            "answer_type": "binary",
            "target_entities": ["table"],
            "target_state": "space available",
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )
    text_only_false_positive = compile_evidence_demand(
        "Can I put something big under my bed?",
        category="object state recognition",
        question_ir={
            "operation": "spatial_relation",
            "answer_type": "binary",
            "target_entities": ["something", "bed"],
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert capacity.primary_need == NEED_EXISTENCE_OR_STATE
    assert capacity.strategy_variant == "surface_capacity"
    assert capacity.open_target_slot is True
    assert capacity.metadata["surface_capacity_query"] is True
    assert text_only_false_positive.primary_need == NEED_SPATIAL_LOCALIZATION
    assert text_only_false_positive.strategy_variant != "surface_capacity"
    assert text_only_false_positive.metadata["surface_capacity_query"] is False


def test_ir_first_v4_surface_capacity_accepts_semantic_state_ir() -> None:
    demand = compile_evidence_demand(
        "Is there room on my desk?",
        category="spatial understanding",
        question_ir={
            "operation": "state_query",
            "answer_type": "binary",
            "target_entities": ["desk"],
            "target_state": "available space",
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == NEED_EXISTENCE_OR_STATE
    assert demand.strategy_variant == "surface_capacity"
    assert demand.metadata["surface_capacity_query"] is True


@pytest.mark.parametrize(
    ("question", "question_ir", "expected_need", "expected_variant"),
    [
        (
            "What type of room is this?",
            {
                "operation": "identity_query",
                "answer_type": "text",
                "attribute_dimension": "type",
                "target_entities": ["room"],
                "qa_focus": "descriptive",
            },
            NEED_OPEN_DESCRIPTIVE,
            "scene_or_content_description",
        ),
        (
            "What is today's date on the calendar?",
            {
                "operation": "attribute_query",
                "answer_type": "text",
                "target_entities": ["calendar"],
                "target_state": "today's date",
                "temporal_relation": "current",
            },
            NEED_OCR_LOGO_TEXT,
            "owner_aware_text",
        ),
    ],
)
def test_ir_first_v4_structured_room_and_calendar_boundaries(
    question: str,
    question_ir: dict,
    expected_need: str,
    expected_variant: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="deliberately conflicting category",
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == expected_need
    assert demand.strategy_variant == expected_variant
    assert demand.metadata["official_category_used_for_routing"] is False


def test_ir_first_v4_room_boundary_does_not_capture_object_type() -> None:
    demand = compile_evidence_demand(
        "What type of chair is this?",
        question_ir={
            "operation": "identity_query",
            "answer_type": "text",
            "attribute_dimension": "type",
            "target_entities": ["chair"],
            "qa_focus": "descriptive",
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == NEED_OBJECT_IDENTITY
    assert demand.strategy_variant == "direct_identity"


@pytest.mark.parametrize(
    ("question", "operation", "target_state"),
    [
        (
            "Where would you suggest placing a computer in this room?",
            "location_query",
            "suitable placement location",
        ),
        (
            "Where can I get recipes for what to cook?",
            "location_query",
            "recipes for what to cook",
        ),
        (
            "What two items should be moved before using the shower?",
            "identity_query",
            "should be moved",
        ),
        (
            "What type of furniture is used to display items?",
            "identity_query",
            "display items",
        ),
        (
            "Where should I sit without moving anything?",
            "location_query",
            "suitable for sitting without moving anything",
        ),
        (
            "Where can I sit down and rest?",
            "location_query",
            "sit down and rest",
        ),
        (
            "The printer ran out of paper, where can I get more?",
            "location_query",
            "available paper",
        ),
        (
            "Where can I set up a place to study?",
            "location_query",
            "suitable for setting up a study place",
        ),
    ],
)
def test_ir_first_v4_structured_functional_state_overrides_surface_operation(
    question: str,
    operation: str,
    target_state: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="spatial understanding",
        question_ir={
            "operation": operation,
            "answer_type": "location_text" if operation == "location_query" else "text",
            "target_state": target_state,
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == NEED_FUNCTIONAL_WORLD
    assert demand.strategy_variant == "grounded_function_or_world"
    assert demand.source == "question_ir.structured_functional_intent"


@pytest.mark.parametrize(
    ("question", "question_ir", "expected_need"),
    [
        (
            "Where would the garbage disposal switch be located?",
            {"operation": "location_query", "answer_type": "location_text"},
            NEED_SPATIAL_LOCALIZATION,
        ),
        (
            "What type of windows does the house have?",
            {
                "operation": "identity_query",
                "answer_type": "text",
                "attribute_dimension": "type",
                "target_entities": ["windows"],
            },
            NEED_OBJECT_IDENTITY,
        ),
        (
            "What is the used towel?",
            {
                "operation": "identity_query",
                "answer_type": "text",
                "target_state": "used",
                "target_entities": ["towel"],
            },
            NEED_OBJECT_IDENTITY,
        ),
    ],
)
def test_ir_first_v4_functional_boundary_preserves_nonfunctional_ir(
    question: str,
    question_ir: dict,
    expected_need: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="functional reasoning",
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == expected_need
    assert demand.metadata["official_category_used_for_routing"] is False


def test_ir_first_v4_surface_capacity_does_not_treat_room_entity_as_capacity() -> None:
    demand = compile_evidence_demand(
        "Can I go out this room without opening the door?",
        category="functional reasoning",
        question_ir={
            "operation": "functional_query",
            "answer_type": "binary",
            "target_entities": ["room", "door"],
            "target_state": "exit room without opening the door",
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == NEED_FUNCTIONAL_WORLD
    assert demand.strategy_variant == "grounded_function_or_world"
    assert demand.metadata["surface_capacity_query"] is False

    quantified_capacity = compile_evidence_demand(
        "Is there some room to hang more clothes?",
        category="spatial understanding",
        question_ir={
            "operation": "functional_query",
            "answer_type": "binary",
            "target_entities": ["clothes"],
            "target_state": "room to hang more clothes",
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )
    assert quantified_capacity.primary_need == NEED_EXISTENCE_OR_STATE
    assert quantified_capacity.strategy_variant == "surface_capacity"


@pytest.mark.parametrize(
    ("question", "target_state", "expected_variant"),
    [
        ("Is the television on or off?", "on or off", "power_state"),
        ("Is this home on the first floor?", "on the first floor", "other_state"),
        (
            "Is the tabletop with a postal bin on it clean?",
            "clean",
            "cleanliness_state",
        ),
        ("Is the table with monitors on it messy?", "messy", "cleanliness_state"),
        ("Is the towel on the toilet used?", "used", "other_state"),
    ],
)
def test_ir_first_v4_power_state_uses_state_predicate_not_spatial_on(
    question: str,
    target_state: str,
    expected_variant: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        category="object state recognition",
        question_ir={
            "operation": "state_query",
            "answer_type": "binary",
            "target_state": target_state,
        },
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == NEED_EXISTENCE_OR_STATE
    assert demand.strategy_variant == expected_variant


@pytest.mark.parametrize(
    ("question", "question_ir", "expected_need"),
    [
        (
            "Where can I read a book?",
            {"operation": "functional_query", "attribute_dimension": "function"},
            NEED_FUNCTIONAL_WORLD,
        ),
        (
            "What color is the sign?",
            {"operation": "attribute_query", "attribute_dimension": "color"},
            NEED_ENTITY_ATTRIBUTE,
        ),
    ],
)
def test_ir_first_v4_does_not_trigger_ocr_from_ambiguous_words(
    question: str,
    question_ir: dict,
    expected_need: str,
) -> None:
    demand = compile_evidence_demand(
        question,
        question_ir=question_ir,
        routing_profile=ROUTING_PROFILE_IR_FIRST_V4,
    )

    assert demand.primary_need == expected_need
    assert demand.primary_need != NEED_OCR_LOGO_TEXT


def test_legacy_v1_v2_v3_profiles_remain_category_aware() -> None:
    question = "Where is the chair?"
    question_ir = {"operation": "location_query", "target_entities": ["chair"]}
    profiles = (
        ("openeqa_v0_category_first_v1", "openeqa_question_route_decision.v1"),
        (ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2, "openeqa_question_route_decision.v2"),
        (ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3, "openeqa_question_route_decision.v2"),
    )

    for profile, contract in profiles:
        demand = compile_evidence_demand(
            question,
            category="object recognition",
            question_ir=question_ir,
            routing_profile=profile,
        )
        assert demand.primary_need == NEED_OBJECT_IDENTITY
        assert demand.source == "official_question_category"
        assert demand.contract == contract
