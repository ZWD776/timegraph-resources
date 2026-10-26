from __future__ import annotations

from vlkg.core.graph_retrieval.evidence_demand import (
    AGGREGATION_COUNT,
    EVIDENCE_DEMAND_CONTRACT,
    EVIDENCE_UNIT_FULL_FRAME,
    EVIDENCE_UNIT_GRAPH_FACT,
    EVIDENCE_UNIT_OBJECT_CROP,
    compile_evidence_demand,
    visual_query_intents,
)


def test_attribute_demand_keeps_facet_bound_to_owner() -> None:
    demand = compile_evidence_demand(
        "What material is the ceiling?",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["ceiling"],
            "attribute_dimension": "material",
            "evidence_requirements": ["visual material recognition"],
            "question_route": {"route_id": "entity_attribute"},
        },
    )

    assert demand.contract == EVIDENCE_DEMAND_CONTRACT
    assert demand.primary_targets == ("ceiling",)
    assert demand.requested_facets == ("material",)
    assert demand.owner_binding_required is True
    assert EVIDENCE_UNIT_GRAPH_FACT in demand.evidence_units
    assert EVIDENCE_UNIT_OBJECT_CROP in demand.evidence_units
    assert [row["text"] for row in visual_query_intents(demand)] == ["ceiling"]


def test_spatial_identity_demand_separates_target_and_references() -> None:
    demand = compile_evidence_demand(
        "What is the white object on the wall above the TV?",
        question_ir={
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["white object", "wall", "TV"],
            "attribute_dimension": "type",
            "spatial_constraints": [
                {"relation": "on", "target": "wall"},
                {"relation": "above", "target": "TV"},
            ],
            "hard_constraints": [
                {"kind": "attribute", "text": "white", "role": "target_modifier"}
            ],
            "evidence_question_type": "non_temporal.entity.identity",
            "question_route": {"route_id": "open_descriptive"},
        },
    )

    assert demand.primary_targets == ("white object",)
    assert demand.reference_entities == ("wall", "TV")
    assert demand.spatial_relations[:2] == ("on", "above")
    assert demand.reference_binding_required is True
    assert demand.direction_binding_required is True
    assert EVIDENCE_UNIT_FULL_FRAME in demand.evidence_units
    intents = visual_query_intents(demand)
    assert intents[0]["text"] == "white object"
    assert {row["role"] for row in intents} >= {
        "primary_target",
        "reference_entity",
        "scene_context",
    }


def test_count_demand_requires_set_aggregation_and_frame_coverage() -> None:
    demand = compile_evidence_demand(
        "How many chairs are in the room?",
        question_ir={
            "answer_type": "count",
            "operation": "count_query",
            "aggregation": "count",
            "target_entities": ["chairs"],
            "question_route": {"route_id": "count_set"},
        },
    )

    assert demand.aggregation == AGGREGATION_COUNT
    assert EVIDENCE_UNIT_OBJECT_CROP in demand.evidence_units
    assert EVIDENCE_UNIT_FULL_FRAME in demand.evidence_units


def test_binary_existence_marks_negative_answer_as_coverage_sensitive() -> None:
    demand = compile_evidence_demand(
        "Is there a fire extinguisher?",
        question_ir={
            "answer_type": "binary",
            "operation": "existence_query",
            "target_entities": ["fire extinguisher"],
            "question_route": {"route_id": "binary_existence"},
        },
    )

    assert demand.negative_evidence_coverage_required is True
    assert EVIDENCE_UNIT_FULL_FRAME in demand.evidence_units
