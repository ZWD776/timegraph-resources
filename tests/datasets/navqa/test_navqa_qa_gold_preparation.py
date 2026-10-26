from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / "scripts"
    / "prepare_navqa_qa_gold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_navqa_qa_gold", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_question_metadata_aliases_match_sequence_prefixed_uuid() -> None:
    module = _load_module()
    metadata = module.question_metadata_aliases(
        {
            "data": [
                {
                    "id": "3_LONG_example",
                    "start_time": 1673986500.0,
                    "end_time": 1673987100.0,
                }
            ]
        },
        seq_id="3",
    )

    assert metadata["3_LONG_example"]["end_time"] == 1673987100.0
    assert metadata["LONG_example"]["end_time"] == 1673987100.0


def test_build_records_materializes_time_gold_for_prefixed_uuid() -> None:
    module = _load_module()
    rows = [
        {
            "UUID": "3_LONG_example",
            "Seq ID": "3",
            "Question": "When did I see the marker?",
            "Timestamp \nwith answer": "12:16:08",
            "Type \n(binary, position, time, text)": "time",
            "Text answer": "",
            "Parsable answer": "[minutes] ago",
            "Question\nCategory": "temporal",
        }
    ]
    metadata = {
        "3_LONG_example": {
            "id": "3_LONG_example",
            "start_time": 1673986544.1946297,
            "end_time": 1673987134.9523191,
        }
    }

    records = module.build_records(
        rows,
        seq_id="3",
        question_metadata=metadata,
    )

    assert records[0]["gold_answer"] == "9.45 minutes ago"
    assert records[0]["official_label"]["semantics"] == "minutes_before_question_end_time"
    assert records[0]["is_current_mainline"] is True
