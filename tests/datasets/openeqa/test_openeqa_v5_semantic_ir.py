from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import pytest

from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import (
    ROUTE_IDS,
    TAXONOMY_CONTRACT,
)


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SCRIPT_DIR = REPO_ROOT / "scripts" / "openeqa"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

compiler = import_module("scripts.datasets.openeqa.retrieval.run_openeqa_v5_semantic_ir")
merger = import_module("scripts.datasets.openeqa.retrieval.merge_openeqa_v5_semantic_ir")
route_builder = import_module("scripts.datasets.openeqa.retrieval.build_openeqa_single_route_ir")


def _row() -> dict:
    return {
        "question_id": "q1",
        "question": "Is there room on the table to work?",
        "category": "must not enter the compiler",
        "answer": "must not enter the compiler",
        "episode_history": "hm3d-v0/episode-1",
    }


def _raw(**overrides: object) -> dict:
    value = {
        "uuid": "q1",
        "question_type": "surface_capacity",
        "operation": "functional_query",
        "answer_type": "binary",
        "target_entities": ["table"],
        "attribute_dimension": "",
        "target_state": "available space",
        "spatial_constraints": [],
        "hard_constraints": [],
        "aggregation": "yes_no",
        "rendering": "yes_no",
        "confidence": 0.9,
        "rationale": "The question asks about usable table space.",
    }
    value.update(overrides)
    return value


def test_frozen_taxonomy_config_matches_exactly_fourteen_runtime_types() -> None:
    payload = json.loads(
        (REPO_ROOT / "configs/datasets/openeqa/taxonomy.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["contract"] == TAXONOMY_CONTRACT
    assert payload["question_type_count"] == 14
    assert set(payload["question_types"]) == set(ROUTE_IDS)
    assert payload["unsupported_fallback_is_question_type"] is False
    assert "unsupported_fallback" not in ROUTE_IDS


def test_prompt_exposes_only_uuid_and_raw_question_per_item() -> None:
    prompt = compiler.build_prompt([_row()])

    assert prompt["questions"] == [
        {"uuid": "q1", "question": "Is there room on the table to work?"}
    ]
    serialized_questions = json.dumps(prompt["questions"], ensure_ascii=False)
    assert "must not enter" not in serialized_questions
    assert prompt["input_contract"]["allowed_fields"] == ["uuid", "question"]


def test_question_id_filter_is_selection_only() -> None:
    rows = [
        _row(),
        {
            **_row(),
            "question_id": "q2",
            "episode_history": "hm3d-v0/episode-2",
        },
    ]

    selected = compiler.select_rows(
        rows,
        data_source="hm3d-v0",
        question_ids=["q2"],
    )

    assert [row["question_id"] for row in selected] == ["q2"]
    assert compiler.build_prompt(selected)["questions"] == [
        {"uuid": "q2", "question": "Is there room on the table to work?"}
    ]


def test_api_key_slot_is_selected_without_retaining_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = argparse.Namespace(api_key_env_var="TEST_OPENEQA_API_KEY")
    monkeypatch.setenv("TEST_OPENEQA_API_KEYS", "first,second")

    compiler.pin_api_key(config, 1)

    assert os.environ["TEST_OPENEQA_API_KEY"] == "second"
    assert "TEST_OPENEQA_API_KEYS" not in os.environ


def test_response_items_align_by_uuid_not_model_order() -> None:
    rows = [
        _row(),
        {**_row(), "question_id": "q2"},
    ]

    aligned = compiler.align_response_items(
        [{"uuid": "q2"}, {"uuid": "q1"}],
        rows,
    )

    assert [item["uuid"] for item in aligned] == ["q1", "q2"]


def test_response_item_uuid_coverage_fails_closed() -> None:
    with pytest.raises(ValueError, match="uuid coverage mismatch"):
        compiler.align_response_items([{"uuid": "other"}], [_row()])


def test_semantic_response_retry_recovers_uuid_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = argparse.Namespace(
        api_key_env_var="TEST_OPENEQA_RETRY_KEY",
        model="gpt-5.6-luna",
    )
    monkeypatch.setenv("TEST_OPENEQA_RETRY_KEY", "secret")
    responses = iter(
        [
            {
                "choices": [
                    {"message": {"content": json.dumps({"items": [{"uuid": "typo"}]})}}
                ]
            },
            {
                "choices": [
                    {"message": {"content": json.dumps({"items": [{"uuid": "q1"}]})}}
                ]
            },
        ]
    )
    monkeypatch.setattr(
        compiler,
        "_openeqa_post_payload",
        lambda *args, **kwargs: next(responses),
    )

    items, log = compiler.compile_batch(
        [_row()],
        config=config,
        max_tokens=100,
        temperature=0.0,
        semantic_retries=1,
    )

    assert items == [{"uuid": "q1"}]
    assert log["semantic_response_attempts"] == 2


def test_normalized_ir_has_one_type_and_no_route_or_second_layer() -> None:
    ir = compiler.normalize_ir_item(_raw(), row=_row(), data_source="hm3d-v0")

    assert ir["question_type"] == "surface_capacity"
    assert ir["compiler_inputs"] == ["raw_question"]
    assert ir["uses_gold_answer"] is False
    assert ir["uses_graph_evidence"] is False
    assert ir["uses_visual_evidence"] is False
    assert not (compiler.FORBIDDEN_IR_FIELDS & set(ir))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_type", "unsupported_fallback"),
        ("question_type", ""),
        ("operation", "unknown"),
        ("answer_type", "unknown"),
    ],
)
def test_invalid_semantic_contract_fails_closed(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        compiler.normalize_ir_item(
            _raw(**{field: value}),
            row=_row(),
            data_source="hm3d-v0",
        )


@pytest.mark.parametrize(
    "forbidden_field",
    sorted(compiler.FORBIDDEN_IR_FIELDS),
)
def test_route_and_label_fields_fail_closed(forbidden_field: str) -> None:
    with pytest.raises(ValueError, match="forbidden IR fields"):
        compiler.normalize_ir_item(
            _raw(**{forbidden_field: "leak"}),
            row=_row(),
            data_source="hm3d-v0",
        )


def test_output_contract_declares_semantic_ir_only() -> None:
    ir = compiler.normalize_ir_item(_raw(), row=_row(), data_source="hm3d-v0")
    args = argparse.Namespace(
        data_source="hm3d-v0",
        temperature=0.0,
        batch_size=5,
    )
    payload = compiler.build_output(
        args=args,
        input_path=Path("open-eqa-v0.json"),
        model="gpt-5.6-luna",
        records=[
            {
                "index": 1,
                "question_id": "q1",
                "llm_question_ir": ir,
            }
        ],
        batch_logs=[],
        status="complete",
    )

    assert payload["taxonomy_contract"] == TAXONOMY_CONTRACT
    assert payload["compiler_input_fields"] == ["uuid", "question"]
    assert payload["official_question_category_provided"] is False
    assert payload["persists_question_route"] is False
    assert payload["summary"]["question_type_counts"] == {"surface_capacity": 1}


def test_complete_disjoint_semantic_ir_shards_merge_in_dataset_order() -> None:
    rows = [
        _row(),
        {
            **_row(),
            "question_id": "q2",
            "episode_history": "hm3d-v0/episode-2",
        },
    ]
    ir1 = compiler.normalize_ir_item(_raw(), row=rows[0], data_source="hm3d-v0")
    ir2 = compiler.normalize_ir_item(
        _raw(uuid="q2", question_type="entity_state", operation="state_query"),
        row=rows[1],
        data_source="hm3d-v0",
    )

    def shard(record: dict) -> dict:
        return {
            "contract": compiler.OUTPUT_CONTRACT,
            "semantic_ir_schema": compiler.IR_SCHEMA,
            "taxonomy_contract": TAXONOMY_CONTRACT,
            "data_source": "hm3d-v0",
            "status": "complete",
            "records": [record],
        }

    merged = merger.merge_records(
        [
            shard({"question_id": "q2", "llm_question_ir": ir2}),
            shard({"question_id": "q1", "llm_question_ir": ir1}),
        ],
        dataset_rows=rows,
        data_source="hm3d-v0",
    )

    assert [record["question_id"] for record in merged] == ["q1", "q2"]
    assert [record["index"] for record in merged] == [1, 2]


def test_semantic_ir_merge_rejects_incomplete_coverage() -> None:
    ir = compiler.normalize_ir_item(_raw(), row=_row(), data_source="hm3d-v0")
    payload = {
        "contract": compiler.OUTPUT_CONTRACT,
        "semantic_ir_schema": compiler.IR_SCHEMA,
        "taxonomy_contract": TAXONOMY_CONTRACT,
        "data_source": "hm3d-v0",
        "status": "complete",
        "records": [{"question_id": "q1", "llm_question_ir": ir}],
    }
    rows = [
        _row(),
        {
            **_row(),
            "question_id": "q2",
            "episode_history": "hm3d-v0/episode-2",
        },
    ]

    with pytest.raises(ValueError, match="source coverage mismatch"):
        merger.merge_records([payload], dataset_rows=rows, data_source="hm3d-v0")


def test_route_builder_merges_multiple_semantic_ir_sources(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for question_id, question_type in (
        ("q1", "surface_capacity"),
        ("q2", "entity_state"),
    ):
        path = tmp_path / f"{question_id}.json"
        path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "question_id": question_id,
                            "llm_question_ir": {
                                "question_type": question_type,
                                "uses_gold_answer": False,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    records = route_builder.load_source_records(paths)

    assert set(records) == {"q1", "q2"}
    normalized = route_builder.normalize_ir(
        records["q1"],
        question="Is there room on the table?",
        question_id="q1",
        data_source="hm3d-v0",
        episode_history="hm3d-v0/episode-1",
    )
    route = normalized["question_route"]
    assert route["route_id"] == "surface_capacity"
    assert route["handler_id"].startswith("openeqa_v5.")
    assert route["visual_query_builder_id"] == "surface_occupancy_v1"
    assert route["visual_strategy_contract"] == (
        "openeqa_v5_visual_strategy.v1"
    )
    assert normalized.get("target_state", "") == ""


def test_surface_capacity_route_persists_required_strategy_parameters() -> None:
    normalized = route_builder.normalize_ir(
        {
            "llm_question_ir": {
                "question_type": "surface_capacity",
                "target_entities": ["dining table"],
                "target_state": "",
                "attribute_dimension": "",
                "spatial_constraints": [],
                "hard_constraints": [
                    {"kind": "activity", "text": "space available to eat"}
                ],
                "uses_gold_answer": False,
            }
        },
        question="Is there room on the dining table to eat?",
        question_id="surface-q",
        data_source="hm3d-v0",
        episode_history="hm3d-v0/episode-1",
    )

    assert normalized["question_type"] == "surface_capacity"
    assert normalized["target_state"] == "has room to eat"
    assert normalized["attribute_dimension"] == "state"
    assert normalized["spatial_constraints"] == [
        {"relation": "on", "target": "dining table"}
    ]
    assert normalized["hard_constraints"] == [
        {
            "kind": "state",
            "text": "room on the dining table to eat",
            "role": "target_modifier",
            "target_entities": ["dining table"],
            "must_bind_to": ["answer_entity"],
        }
    ]
    assert normalized["strategy_parameter_normalization"][
        "runtime_reclassification"
    ] is False


@pytest.mark.parametrize(
    ("question_type", "source", "expected"),
    [
        (
            "entity_state",
            {
                "operation": "state_query",
                "target_entities": ["door"],
                "attribute_dimension": "",
                "target_state": "open",
            },
            {"operation": "attribute_query", "attribute_dimension": "state"},
        ),
        (
            "grounded_world_knowledge",
            {
                "operation": "world_knowledge_query",
                "target_entities": ["kitchen", "setting"],
                "target_state": "",
                "hard_constraints": [
                    {"kind": "classification", "text": "home or public space"}
                ],
            },
            {
                "target_entities": ["kitchen"],
                "target_state": "home or public space",
            },
        ),
        (
            "content_description",
            {
                "operation": "descriptive_query",
                "target_entities": ["animal"],
            },
            {"operation": "identity_query"},
        ),
        (
            "scene_description",
            {
                "operation": "descriptive_query",
                "target_entities": ["staircase"],
            },
            {"operation": "attribute_query"},
        ),
        (
            "text_reading",
            {
                "operation": "location_query",
                "target_entities": ["Bless this home"],
            },
            {"target_entities": ["written text", "Bless this home"]},
        ),
    ],
)
def test_route_builder_normalizes_type_constrained_strategy_parameters(
    question_type: str,
    source: dict,
    expected: dict,
) -> None:
    normalized = route_builder.normalize_strategy_parameters(
        {"question_type": question_type, **source},
        question_type=question_type,
    )

    for key, value in expected.items():
        assert normalized[key] == value
    assert normalized["strategy_parameter_normalization"][
        "runtime_reclassification"
    ] is False
