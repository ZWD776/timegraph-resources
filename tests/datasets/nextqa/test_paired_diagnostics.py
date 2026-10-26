from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

from vlkg.datasets.nextqa.audit.paired_diagnostics import compare_paired_runs


def _ir(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "video_id": "v1",
        "normalized_question": "what did the person do",
        "target_entities": [],
        "actors": ["person"],
        "actions": [],
        "objects": [],
        "temporal_direction": "none",
        "temporal_anchor": "none",
        "referenced_event": "",
        "causal_query": "none",
        "count_query": False,
        "location_query": False,
        "attribute_query": "none",
        "answer_format": "multiple_choice_5",
    }


def _retrieval(record_id: str = "episode:1") -> dict:
    return {
        "question_id": "q1",
        "video_id": "v1",
        "status": "completed",
        "graph_record_ids": [record_id],
        "evidence": [
            {
                "rank": 1,
                "source_record_id": record_id,
                "source_kind": "episode",
                "time_range": {"start_ts": 1.0, "end_ts": 2.0},
                "evidence_text": "person moves",
            }
        ],
    }


def _prediction(choice: int, request_hash: str = "same") -> dict:
    return {
        "question_id": "q1",
        "video_id": "v1",
        "status": "completed",
        "predicted_choice_index": choice,
        "request_sha256": request_hash,
        "evidence_sufficiency": "partial",
    }


def _evaluation(choice: int, *, correct: bool) -> dict:
    return {
        "question_id": "q1",
        "video_id": "v1",
        "official_type": "DO",
        "status": "completed",
        "predicted_choice_index": choice,
        "gold_choice_index": 1,
        "correct": correct,
    }


def test_same_request_answer_change_is_model_variance() -> None:
    result = compare_paired_runs(
        old_ir=[_ir()],
        new_ir=[_ir()],
        old_retrieval=[_retrieval()],
        new_retrieval=[_retrieval()],
        old_predictions=[_prediction(0)],
        new_predictions=[_prediction(1)],
        old_evaluation=[_evaluation(0, correct=False)],
        new_evaluation=[_evaluation(1, correct=True)],
    )

    row = result["question_results"][0]
    assert row["outcome"] == "improved"
    assert row["mechanism"] == "same_request_model_variance"
    assert row["candidate_ir_effect"] is False
    assert result["summary"]["overall"]["same_request_answer_changed"] == 1
    assert result["summary"]["overall"]["same_request_score_flip"] == 1


def test_ir_and_evidence_change_is_only_a_candidate_ir_effect() -> None:
    new_ir = deepcopy(_ir())
    new_ir["target_entities"] = ["ball"]
    new_retrieval = _retrieval("episode:2")
    new_retrieval["evidence"][0]["evidence_text"] = "person holds ball"

    result = compare_paired_runs(
        old_ir=[_ir()],
        new_ir=[new_ir],
        old_retrieval=[_retrieval()],
        new_retrieval=[new_retrieval],
        old_predictions=[_prediction(0, "old")],
        new_predictions=[_prediction(1, "new")],
        old_evaluation=[_evaluation(0, correct=False)],
        new_evaluation=[_evaluation(1, correct=True)],
        repaired_video_ids={"v1"},
    )

    row = result["question_results"][0]
    assert row["mechanism"] == "ir_and_retrieval_changed"
    assert row["candidate_ir_effect"] is True
    assert row["candidate_ir_effect_path"] == "direct_prompt_and_retrieval"
    assert row["video_cohort"] == "repaired_graph_29"
    assert row["ir"]["term_additions"] == {"target_entities": ["ball"]}


def test_paired_sources_must_have_identical_question_ids() -> None:
    mismatched = _ir("q2")
    try:
        compare_paired_runs(
            old_ir=[_ir()],
            new_ir=[mismatched],
            old_retrieval=[_retrieval()],
            new_retrieval=[_retrieval()],
            old_predictions=[_prediction(0)],
            new_predictions=[_prediction(0)],
            old_evaluation=[_evaluation(0, correct=False)],
            new_evaluation=[_evaluation(0, correct=False)],
        )
    except ValueError as exc:
        assert "do not align" in str(exc)
    else:
        raise AssertionError("expected mismatched question IDs to fail")


def test_ir_change_with_same_evidence_is_a_direct_prompt_effect() -> None:
    new_ir = deepcopy(_ir())
    new_ir["actions"] = ["hold"]

    result = compare_paired_runs(
        old_ir=[_ir()],
        new_ir=[new_ir],
        old_retrieval=[_retrieval()],
        new_retrieval=[_retrieval()],
        old_predictions=[_prediction(0, "old")],
        new_predictions=[_prediction(1, "new")],
        old_evaluation=[_evaluation(0, correct=False)],
        new_evaluation=[_evaluation(1, correct=True)],
    )

    row = result["question_results"][0]
    assert row["mechanism"] == "semantic_ir_changed_same_evidence"
    assert row["candidate_ir_effect_path"] == "direct_prompt_only"
    assert result["summary"]["overall"]["candidate_ir_direct_prompt_only"] == 1
    assert result["summary"]["overall"]["regressed_with_new_incomplete_evidence"] == 0


def test_paired_diagnostic_code_is_dataset_isolated() -> None:
    repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    paths = [
        repo_root / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'audit' / 'paired_diagnostics.py',
        repo_root / "src" / "vlkg" / "datasets" / "nextqa" / "audit" / "paired_diagnostics.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("eval_adapters.navqa" in name for name in imports)
        assert not any("eval_adapters.openeqa" in name for name in imports)
