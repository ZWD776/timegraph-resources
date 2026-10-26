from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval" / "audit_navqa_question_routes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("navqa_question_route_audit_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_question_routes_uses_ir_and_keeps_route_fields() -> None:
    module = _load_module()
    rows = [
        {
            module.FIELD_SEQ_ID: "4",
            module.FIELD_UUID: "q-duration",
            module.FIELD_QUESTION: "How long was the person at the door?",
            module.FIELD_TYPE: "duration",
            module.FIELD_CATEGORY: "temporal",
        },
        {
            module.FIELD_SEQ_ID: "4",
            module.FIELD_UUID: "q-count",
            module.FIELD_QUESTION: "How many doors did the person open?",
            module.FIELD_TYPE: "text",
            module.FIELD_CATEGORY: "count",
        },
    ]
    ir_records = {
        "q-duration": {"answer_type": "duration", "operation": "interval"},
        "q-count": {
            "answer_type": "text",
            "aggregation": "count",
            "target_entities": ["doors"],
            "event_constraints": ["person open doors"],
        },
    }

    records = module.audit_question_routes(rows, question_ir_records=ir_records)

    assert records[0]["question_route_id"] == "temporal_interval"
    assert records[0]["taxonomy_path"] == "temporal.interval"
    assert records[0]["time_requirements"]["requires_interval"] is True
    assert records[1]["question_route_id"] == "count_set"
    assert records[1]["taxonomy_path"] == "non_temporal.entity.count"
    assert "doors" in records[1]["primary_target_terms"]
    assert "person open doors" in records[1]["event_constraints"]


def test_route_audit_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "qa.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                module.FIELD_SEQ_ID,
                module.FIELD_UUID,
                module.FIELD_QUESTION,
                module.FIELD_TYPE,
                module.FIELD_CATEGORY,
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                module.FIELD_SEQ_ID: "21",
                module.FIELD_UUID: "q-color",
                module.FIELD_QUESTION: "What color was the car near the intersection?",
                module.FIELD_TYPE: "text",
                module.FIELD_CATEGORY: "attribute",
            }
        )
    ir_path = tmp_path / "ir.json"
    ir_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "uuid": "q-color",
                        "llm_question_ir": {
                            "answer_type": "text",
                            "operation": "attribute_query",
                            "target_entities": ["car"],
                            "attribute_dimension": "color",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = module.audit_question_routes(
        module.filter_rows(
            module.load_rows(csv_path),
            seq_ids=None,
            include_types=list(module.NON_POSITION_QA_TYPES),
            exclude_types=None,
        ),
        question_ir_records=module.load_question_ir_records(ir_path),
    )
    output_json = tmp_path / "routes.json"
    output_csv = tmp_path / "routes.csv"
    output_md = tmp_path / "routes.md"

    module.write_outputs(
        records,
        csv_path=csv_path,
        question_ir_json=ir_path,
        output_json=output_json,
        output_csv=output_csv,
        output_md=output_md,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["question_count"] == 1
    assert payload["route_distribution"] == {"entity_attribute": 1}
    assert payload["taxonomy_distribution"] == {"non_temporal.entity.attribute": 1}
    csv_text = output_csv.read_text(encoding="utf-8-sig")
    assert "entity_attribute" in csv_text
    assert "non_temporal.entity.attribute" in csv_text
    md_text = output_md.read_text(encoding="utf-8")
    assert "NaVQA QuestionRoute Audit" in md_text
    assert "Taxonomy Distribution" in md_text
