from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from vlkg.datasets.nextqa.retrieval.llm_question_ir import (
    LLMQuestionIRConfig,
    OpenAICompatibleIRClient,
    audit_prompt_isolation,
    build_question_ir_messages,
    compile_question_ir,
    parse_semantic_ir_response,
    request_fingerprint,
    semantic_only_view,
)
from vlkg.datasets.nextqa.retrieval.question_ir import retrieval_input_from_ir


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
RUNNER_PATH = REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'retrieval' / 'build_question_ir_llm.py'


def _config(*, max_attempts: int = 1) -> LLMQuestionIRConfig:
    return LLMQuestionIRConfig(
        profile_name="nextqa_question_only_llm_ir_v3",
        provider="openai_compatible",
        model="gpt-5.6-luna",
        api_base_url="https://example.invalid/v1",
        api_base_url_env_var=None,
        endpoint_path="/chat/completions",
        api_key_env_var="NEXTQA_TEST_API_KEY",
        api_key_pool_env_var="NEXTQA_TEST_API_KEYS",
        timeout_sec=1.0,
        temperature=0.1,
        max_tokens=500,
        max_attempts=max_attempts,
        thinking_enabled=False,
        retry_backoff_sec=0.0,
        retry_jitter_sec=0.0,
    )


def _question() -> dict[str, object]:
    return {
        "question_id": "nextqa-val/10030609934/q1",
        "video_id": "10030609934",
        "normalized_question": (
            "why was the boy at the brown table near the stairs at the beginning "
            "of the video"
        ),
        "official_type": "CW",
        "answer_choices": [
            "sentinel-choice-zero",
            "sentinel-choice-one",
            "sentinel-choice-two",
            "sentinel-choice-three",
            "sentinel-choice-four",
        ],
        "answer_format": "multiple_choice_5",
        "gold_choice_index": 3,
        "gold_answer": "sentinel-gold-answer",
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "question_id": "nextqa-val/10030609934/q1",
        "intent": "causal_why",
        "answer_target": "reason",
        "actors": ["boy"],
        "actions": [],
        "objects": ["brown table", "stairs"],
        "states": [],
        "anchor_event": {
            "text": "boy was at the brown table near the stairs",
            "actors": ["boy"],
            "actions": [],
            "objects": ["brown table", "stairs"],
            "states": [],
        },
        "temporal_relation": "none",
        "causal_relation": "reason_for_effect",
        "visual_evidence_demand": "preceding_context",
        "confidence": 0.96,
    }


def _chat_response(payload: dict[str, object] | None = None) -> dict[str, object]:
    content = payload or _semantic_payload()
    return {
        "id": "fake-response-id",
        "choices": [
            {
                "message": {"content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("nextqa_llm_ir_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_contains_only_question_identity_and_text_from_the_row() -> None:
    row = _question()
    messages = build_question_ir_messages(row)
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["input"] == {
        "question_id": row["question_id"],
        "question": row["normalized_question"],
    }
    serialized = json.dumps(messages)
    assert "sentinel-choice" not in serialized
    assert "sentinel-gold-answer" not in serialized
    assert '"official_type"' not in serialized
    assert '"answer_choices"' not in serialized
    assert '"gold_choice_index"' not in serialized
    assert "while/as/when/during" in serialized
    assert "means, manner, method" in serialized


def test_prompt_fingerprint_is_invariant_to_forbidden_fields() -> None:
    row = _question()
    baseline = request_fingerprint(row)
    checks = audit_prompt_isolation(row)

    assert checks
    assert all(checks.values())
    assert request_fingerprint({**row, "official_type": "TN"}) == baseline
    assert request_fingerprint({**row, "answer_choices": ["x"] * 5}) == baseline
    assert request_fingerprint({**row, "gold_choice_index": 0}) == baseline


def test_response_schema_accepts_valid_semantics_and_rejects_inconsistency() -> None:
    parsed = parse_semantic_ir_response(
        _chat_response(), expected_question_id=str(_question()["question_id"])
    )
    assert parsed["intent"] == "causal_why"
    assert parsed["anchor_event"]["actors"] == ["boy"]

    invalid = _semantic_payload()
    invalid["visual_evidence_demand"] = "direct_visual"
    with pytest.raises(ValueError, match="visual_evidence_demand is inconsistent"):
        parse_semantic_ir_response(
            _chat_response(invalid),
            expected_question_id=str(_question()["question_id"]),
        )


def test_object_purpose_causal_why_does_not_require_an_event_anchor() -> None:
    purpose = _semantic_payload()
    purpose.update(
        {
            "actors": [],
            "actions": [],
            "objects": ["pole"],
            "states": [],
            "anchor_event": {
                "text": "",
                "actors": [],
                "actions": [],
                "objects": ["pole"],
                "states": [],
            },
        }
    )

    parsed = parse_semantic_ir_response(
        _chat_response(purpose),
        expected_question_id=str(_question()["question_id"]),
    )
    assert parsed["intent"] == "causal_why"
    assert parsed["anchor_event"]["text"] == ""
    assert parsed["objects"] == ["pole"]


def test_fake_live_api_success_and_output_views_are_pure(monkeypatch) -> None:
    monkeypatch.setenv("NEXTQA_TEST_API_KEY", "test-only-key")
    config = _config()
    client = OpenAICompatibleIRClient(
        config,
        post_json=lambda payload, api_key: _chat_response(),
    )

    runtime = compile_question_ir(
        _question(), config=config, mode="live", client=client
    )
    semantic = semantic_only_view(runtime)
    retrieval = retrieval_input_from_ir(runtime).to_dict()

    assert runtime["status"] == "completed"
    assert runtime["model"] == "gpt-5.6-luna"
    assert runtime["official_type"] == "CW"
    assert len(runtime["answer_choices"]) == 5
    assert runtime["rule_fallback_used"] is False
    assert "official_type" not in semantic
    assert "answer_choices" not in semantic
    assert "official_type" not in retrieval
    assert "answer_choices" not in retrieval
    assert "visual_evidence_demand" not in retrieval


def test_api_failure_is_an_error_without_rule_fallback(monkeypatch) -> None:
    monkeypatch.setenv("NEXTQA_TEST_API_KEY", "test-only-key")
    config = _config(max_attempts=2)

    def fail_request(payload, api_key):
        raise URLError("intentional test failure")

    client = OpenAICompatibleIRClient(config, post_json=fail_request)
    record = compile_question_ir(
        _question(), config=config, mode="live", client=client
    )

    assert record["status"] == "error"
    assert record["attempt_count"] == 2
    assert record["rule_fallback_used"] is False
    assert "intent" not in record
    assert "answer_choices" not in record


def test_resume_selection_preserves_completed_and_retries_errors() -> None:
    runner = _load_runner_module()
    ids = ["q1", "q2", "q3"]
    existing = [
        {"question_id": "q1", "status": "completed"},
        {"question_id": "q2", "status": "error"},
    ]
    assert runner.select_pending_question_ids(ids, existing) == ["q2", "q3"]


def test_smoke_scope_requires_and_selects_exactly_one_question() -> None:
    runner = _load_runner_module()
    rows = [
        {"question_id": "q1", "video_id": "v1"},
        {"question_id": "q2", "video_id": "v2"},
    ]

    assert runner.select_scope_rows(rows, scope="smoke", question_id="q2") == [
        rows[1]
    ]
    with pytest.raises(ValueError, match="requires --question-id"):
        runner.select_scope_rows(rows, scope="smoke", question_id=None)
    with pytest.raises(ValueError, match="only with smoke"):
        runner.select_scope_rows(rows, scope="pilot", question_id="q1")


def test_checkpoint_materializes_views_only_when_requested(
    monkeypatch, nextqa_tmp_dir: Path
) -> None:
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "semantic_only_view", lambda row: {"view": "semantic"})
    monkeypatch.setattr(
        runner,
        "retrieval_input_from_ir",
        lambda row: type("Projection", (), {"to_dict": lambda self: {"view": "retrieval"}})(),
    )
    records = [
        {
            "question_id": "q1",
            "status": "completed",
        }
    ]

    runner._checkpoint(nextqa_tmp_dir, records)
    assert (nextqa_tmp_dir / "question_ir.records.json").is_file()
    assert not (nextqa_tmp_dir / "question_ir.runtime.json").exists()
    assert not (nextqa_tmp_dir / "question_ir.semantic.json").exists()
    assert not (nextqa_tmp_dir / "question_ir.retrieval.json").exists()

    runner._checkpoint(nextqa_tmp_dir, records, materialize_views=True)
    assert (nextqa_tmp_dir / "question_ir.runtime.json").is_file()
    assert (nextqa_tmp_dir / "question_ir.semantic.json").is_file()
    assert (nextqa_tmp_dir / "question_ir.retrieval.json").is_file()


def test_seed_import_accepts_only_completed_matching_request_hash(
    monkeypatch, nextqa_tmp_dir: Path
) -> None:
    monkeypatch.setenv("NEXTQA_TEST_API_KEY", "test-only-key")
    config = _config()
    client = OpenAICompatibleIRClient(
        config,
        post_json=lambda payload, api_key: _chat_response(),
    )
    row = _question()
    completed = compile_question_ir(row, config=config, mode="live", client=client)
    error = {
        "question_id": "nextqa-val/other/q2",
        "status": "error",
    }
    seed_path = nextqa_tmp_dir / "seed.json"
    seed_path.write_text(json.dumps([completed, error]), encoding="utf-8")
    runner = _load_runner_module()

    imported, audit = runner.load_seed_completed_records(
        seed_path,
        question_index={
            str(row["question_id"]): row,
            "nextqa-val/other/q2": {
                "question_id": "nextqa-val/other/q2",
                "normalized_question": "what happened next",
            },
        },
        model="gpt-5.6-luna",
    )

    assert [record["question_id"] for record in imported] == [row["question_id"]]
    assert audit["imported_completed_count"] == 1
    assert audit["skipped_error_count"] == 1
    assert audit["request_fingerprint_verified"] is True


def test_llm_ir_files_do_not_import_navqa_or_openeqa() -> None:
    paths = [
        REPO_ROOT / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'retrieval' / 'llm_question_ir.py',
        RUNNER_PATH,
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
