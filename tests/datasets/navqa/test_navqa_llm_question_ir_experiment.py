from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / 'scripts' / 'datasets' / 'navqa' / 'retrieval' / 'build_question_ir.py'
)


def _load_module():
    spec = importlib.util.spec_from_file_location("navqa_llm_question_ir", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_color_question_ir_validation_accepts_descriptive_attribute() -> None:
    module = _load_module()
    row = {
        module.FIELD_UUID: "q-color",
        module.FIELD_QUESTION: "What were the color of the couches near the entrance?",
        module.FIELD_TYPE: "text",
    }

    ir = module.normalize_ir_item(
        {
            "uuid": "q-color",
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["couches"],
            "attribute_dimension": "color",
            "rendering": "color_only",
            "qa_focus": "descriptive",
            "confidence": 0.9,
        },
        row=row,
    )
    validation = module.validate_ir_item(ir, row=row)

    assert validation["auto_reasonable"] is True
    assert ir["operation"] == "attribute_query"
    assert ir["rendering"] == "color_only"


def test_question_ir_normalizes_hard_constraints() -> None:
    module = _load_module()
    row = {
        module.FIELD_UUID: "q-hard",
        module.FIELD_QUESTION: "What color was the jacket worn by the person who opened the door?",
        module.FIELD_TYPE: "text",
    }

    ir = module.normalize_ir_item(
        {
            "uuid": "q-hard",
            "answer_type": "text",
            "operation": "attribute_query",
            "target_entities": ["jacket"],
            "attribute_dimension": "color",
            "hard_constraints": [
                {
                    "kind": "event",
                    "text": "person who opened the door",
                    "role": "event_condition",
                    "target_entities": ["person", "door"],
                    "must_bind_to": ["answer_entity", "same_time_window"],
                }
            ],
            "rendering": "color_only",
            "qa_focus": "descriptive",
        },
        row=row,
    )

    assert ir["hard_constraints"][0]["text"] == "person who opened the door"
    assert module.validate_ir_item(ir, row=row)["auto_reasonable"] is True


def test_time_question_ir_validation_rejects_attribute_operation() -> None:
    module = _load_module()
    row = {
        module.FIELD_UUID: "q-time",
        module.FIELD_QUESTION: "When did you leave the building?",
        module.FIELD_TYPE: "time",
    }

    ir = module.normalize_ir_item(
        {
            "uuid": "q-time",
            "answer_type": "time",
            "operation": "attribute_query",
            "rendering": "free_text",
            "qa_focus": "descriptive",
        },
        row=row,
    )
    validation = module.validate_ir_item(ir, row=row)

    assert validation["auto_reasonable"] is False
    assert "time_operation_unexpected:attribute_query" in validation["issues"]
    assert "time_rendering_unexpected:free_text" in validation["issues"]


def test_side_question_ir_validation_expects_spatial_geometry_focus() -> None:
    module = _load_module()
    row = {
        module.FIELD_UUID: "q-side",
        module.FIELD_QUESTION: "What side of the street are you driving on?",
        module.FIELD_TYPE: "text",
    }

    ir = module.normalize_ir_item(
        {
            "uuid": "q-side",
            "answer_type": "text",
            "operation": "spatial_relation",
            "target_entities": ["street"],
            "attribute_dimension": "side",
            "rendering": "free_text",
            "qa_focus": "spatial_geometry",
        },
        row=row,
    )
    validation = module.validate_ir_item(ir, row=row)

    assert validation["auto_reasonable"] is True
    assert validation["warnings"] == []


def test_mock_compiler_produces_records_for_rows() -> None:
    module = _load_module()
    rows = [
        {
            module.FIELD_UUID: "q1",
            module.FIELD_QUESTION: "Did you see a red object?",
            module.FIELD_TYPE: "binary",
            module.FIELD_CATEGORY: "environmental",
        }
    ]

    raw_items = module.compile_questions_mock(rows)
    records = module.build_records(rows, raw_items, domain_profile="generic")

    assert len(records) == 1
    assert records[0]["llm_question_ir"]["answer_type"] == "binary"
    assert records[0]["llm_question_ir"]["evidence_question_type"]
    assert records[0]["llm_question_ir"]["question_route"]["route_id"] == "binary_existence"
    assert records[0]["llm_question_ir"]["question_route_id"] == "binary_existence"
    assert "validation" in records[0]


def test_build_records_adds_route_ready_question_type_to_ir() -> None:
    module = _load_module()
    rows = [
        {
            module.FIELD_UUID: "q-count",
            module.FIELD_QUESTION: "How many doors did the person open?",
            module.FIELD_TYPE: "text",
            module.FIELD_CATEGORY: "counting",
        }
    ]

    records = module.build_records(
        rows,
        [
            {
                "uuid": "q-count",
                "answer_type": "text",
                "operation": "occurrence",
                "aggregation": "multiple",
                "target_entities": ["doors"],
                "event_constraints": ["person open doors"],
            }
        ],
        domain_profile="generic",
    )

    ir = records[0]["llm_question_ir"]
    assert ir["evidence_question_type"] == "non_temporal.entity.count"
    assert ir["question_route_id"] == "count_set"
    assert ir["question_route"]["route_id"] == "count_set"
    assert ir["question_route"]["taxonomy_path"] == "non_temporal.entity.count"


def test_extract_ir_items_accepts_results_shape() -> None:
    module = _load_module()

    items, log = module._extract_ir_items(
        {
            "results": [
                {
                    "uuid": "q1",
                    "answer_type": "text",
                    "operation": "attribute_query",
                }
            ]
        }
    )

    assert log["source"] == "results"
    assert items == [
        {
            "uuid": "q1",
            "answer_type": "text",
            "operation": "attribute_query",
        }
    ]


def test_align_ir_items_falls_back_to_batch_order_when_uuid_missing() -> None:
    module = _load_module()
    rows = [
        {module.FIELD_UUID: "q1", module.FIELD_QUESTION: "When did you leave?"},
        {module.FIELD_UUID: "q2", module.FIELD_QUESTION: "What color was the car?"},
    ]
    raw_items = [
        {"answer_type": "time", "operation": "offset"},
        {"answer_type": "text", "operation": "attribute_query"},
    ]

    aligned, log = module._align_ir_items_to_rows(raw_items, rows)

    assert log["method"] == "position"
    assert [item["uuid"] for item in aligned] == ["q1", "q2"]
    assert aligned[0]["operation"] == "offset"


def test_llm_payload_contains_semantic_compiler_contract() -> None:
    module = _load_module()

    class DummyConfig:
        model = "dummy-model"
        reasoning_effort = None
        thinking_enabled = False

    payload = module._build_llm_payload(
        [
            {
                module.FIELD_UUID: "q1",
                module.FIELD_QUESTION: "When were you in the lobby with blue chairs?",
                module.FIELD_TYPE: "time",
                module.FIELD_CATEGORY: "temporal",
            }
        ],
        config=DummyConfig(),
        max_tokens=1000,
        temperature=0.0,
    )
    user_payload = payload["messages"][1]["content"]

    assert "Return exactly one item for each input question" in user_payload
    assert "first-person robot/ego-camera navigation video" in user_payload
    assert "evidence_question_type_taxonomy" in user_payload
    assert "dataset-independent evidence-need type used by downstream routing" in user_payload
    assert "'I', 'you', and 'we' refer to the robot/ego camera" in user_payload
    assert "must not be treated as an answer or as graph evidence" in user_payload
    assert "When were you in/at/on X?" in user_payload
    assert "do not use interval unless the wording asks how long" in user_payload
    assert "operation=position_query" in user_payload
    assert "Do not force side or turn questions into descriptive" in user_payload
    assert "hard_constraints" in user_payload
    assert "Use hard_constraints to preserve question factors" in user_payload
    assert "not an instruction to reject reasonable graph-supported inference" in user_payload
    assert "relative clauses, action clauses" in user_payload
