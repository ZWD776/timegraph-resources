from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from scripts.datasets.nextqa.qa.run_graph_only_qa_pilot import (
    prediction_from_retrieval_error,
    select_pending_questions,
)
from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy
from vlkg.datasets.nextqa.qa.graph_only_qa import (
    OpenAICompatibleMCQClient,
    COMPATIBLE_BROWSER_USER_AGENT,
    answer_graph_only_mcq,
    build_chat_payload,
    build_qa_messages,
    load_graph_only_mcq_config,
    parse_mcq_response,
    parse_api_key_pool,
    stable_api_key_slot,
)


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
CONFIG = REPO_ROOT / 'configs' / 'datasets' / 'nextqa' / 'graph_qa.yaml'
CONFIG_EVIDENCE24 = REPO_ROOT / 'tests/fixtures/release_configs/nextqa/nextqa_graph_qa_luna_evidence24_v1.yaml'


def _question() -> dict:
    return {
        "question_id": "nextqa-val/123/q7",
        "video_id": "123",
        "normalized_question": "what did the boy do after he stopped",
        "answer_choices": ["walk", "sit", "wave", "ride", "look"],
        "official_type": "TN",
        "target_entities": ["boy"],
        "actors": ["boy"],
        "actions": ["stop"],
        "objects": [],
        "temporal_direction": "after",
        "temporal_anchor": "",
        "referenced_event": "boy stopped",
        "causal_query": "none",
        "count_query": False,
        "location_query": False,
        "attribute_query": "none",
        "answer_format": "multiple_choice_5",
        "uses_gold_answer": False,
    }


def _retrieval() -> dict:
    evidence = [
        {
            "rank": rank,
            "source_record_id": f"episode:{rank}",
            "source_kind": "episode",
            "time_range": {"start_ts": float(rank), "end_ts": float(rank)},
            "evidence_text": f"boy event {rank}",
        }
        for rank in range(1, 17)
    ]
    return {
        "question_id": "nextqa-val/123/q7",
        "video_id": "123",
        "status": "completed",
        "graph_retrieval_profile": "nextqa_graph_only_unified_v2",
        "evidence": evidence,
        "graph_record_ids": [row["source_record_id"] for row in evidence],
        "purity": {
            "uses_gold_answer": False,
            "uses_official_type": False,
            "uses_answer_choices": False,
            "reads_rgb": False,
            "reads_object_crop": False,
            "reads_visual_embedding_cache": False,
            "uses_visual_index": False,
            "shared_hybrid_retrieve_called": False,
        },
    }


def test_prompt_uses_one_shared_graph_budget_and_excludes_type_and_gold() -> None:
    messages = build_qa_messages(_question(), _retrieval(), evidence_limit=16)
    prompt = json.loads(messages[1]["content"])

    assert len(prompt["answer_choices"]) == 5
    assert len(prompt["graph_evidence"]) == 16
    assert prompt["policy"]["same_graph_evidence_budget_for_every_choice"] == 16
    serialized = json.dumps(messages).lower()
    assert "official_type" not in serialized
    assert '"tn"' not in serialized
    assert "gold_choice" not in serialized
    assert "answer key" not in messages[1]["content"].lower()


def test_official_type_perturbation_does_not_change_prompt() -> None:
    question = _question()
    changed = deepcopy(question)
    changed["official_type"] = "CW"

    assert build_qa_messages(question, _retrieval(), evidence_limit=16) == (
        build_qa_messages(changed, _retrieval(), evidence_limit=16)
    )


def test_chat_payload_is_fixed_to_gpt5mini_json_mode() -> None:
    config = load_graph_only_mcq_config(CONFIG)
    payload = build_chat_payload(
        build_qa_messages(_question(), _retrieval(), evidence_limit=16),
        config=config,
    )

    assert payload["model"] == "gpt-5.6-luna"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}


def test_evidence24_qa_profile_is_isolated_and_matches_retrieval_budget() -> None:
    config = load_graph_only_mcq_config(CONFIG_EVIDENCE24)
    assert config.profile_name == "nextqa_graph_only_mcq_luna_evidence24_v1"
    assert config.model == "gpt-5.6-luna"
    assert config.evidence_limit == 24


def test_parser_derives_canonical_choice_text_and_validates_evidence_ranks() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "choice_index": 3,
                            "reasoning": "Episode 9 follows the stop.",
                            "used_evidence_ranks": [9],
                            "evidence_sufficiency": "partial",
                        }
                    )
                }
            }
        ]
    }
    parsed = parse_mcq_response(
        response,
        answer_choices=_question()["answer_choices"],
        evidence_count=16,
    )

    assert parsed["choice_index"] == 3
    assert parsed["choice_text"] == "ride"
    assert parsed["used_evidence_ranks"] == [9]


def test_api_failure_is_an_error_and_never_a_fallback_choice(monkeypatch) -> None:
    config = load_graph_only_mcq_config(CONFIG)
    monkeypatch.setenv(config.api_key_env_var, "test-key")

    def fail(_payload, _key):
        raise URLError("offline")

    result = answer_graph_only_mcq(
        _question(),
        _retrieval(),
        config=config,
        answer_mode="live",
        client=OpenAICompatibleMCQClient(config, post_json=fail),
    )

    assert result["status"] == "error"
    assert result["predicted_choice_index"] is None
    assert result["predicted_choice_text"] is None
    assert result["attempt_count"] == config.max_attempts


def test_live_client_uses_gateway_compatible_request_headers(monkeypatch) -> None:
    config = load_graph_only_mcq_config(CONFIG)
    monkeypatch.delenv(config.api_key_pool_env_var, raising=False)
    monkeypatch.setenv(config.api_key_env_var, "test-key")
    captured: dict[str, Request] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "{}"}}]}'

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "vlkg.eval_adapters.nextqa.graph_only_qa.urlopen",
        fake_urlopen,
    )
    client = OpenAICompatibleMCQClient(config)
    client.post({"model": config.model, "messages": []})

    request = captured["request"]
    assert request.get_header("User-agent") == COMPATIBLE_BROWSER_USER_AGENT
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") == "Bearer test-key"


def test_api_key_pool_is_deduplicated_and_slots_are_stable(monkeypatch) -> None:
    config = load_graph_only_mcq_config(CONFIG)
    monkeypatch.setenv(config.api_key_pool_env_var, "k1,k2\nk1; k3")
    monkeypatch.delenv(config.api_key_env_var, raising=False)
    assert parse_api_key_pool() == ["k1", "k2", "k3"]
    assert stable_api_key_slot("question-1", 3) == stable_api_key_slot("question-1", 3)

    client = OpenAICompatibleMCQClient(config)
    assert client.api_key_pool_size == 3
    assert [client._api_key_for_attempt(i) for i in range(4)] == [
        "k1", "k2", "k3", "k1"
    ]


def test_resume_selection_preserves_completed_and_retries_only_error() -> None:
    existing = [
        {"question_id": "q1", "status": "completed"},
        {"question_id": "q2", "status": "error"},
    ]

    assert select_pending_questions(
        ["q1", "q2", "q3"], existing_predictions=existing
    ) == ["q2", "q3"]


def test_qa_runner_exposes_serial_default_and_explicit_worker_flag() -> None:
    from scripts.datasets.nextqa.qa.run_graph_only_qa_pilot import build_parser

    assert build_parser().parse_args(["--run-id", "r"]).workers == 1
    assert build_parser().parse_args(["--run-id", "r", "--workers", "4"]).workers == 4


def test_special_scope_has_exact_repair_counts() -> None:
    from scripts.datasets.nextqa.qa.run_graph_only_qa_pilot import build_parser

    args = build_parser().parse_args(["--run-id", "r", "--scope", "special"])
    assert args.scope == "special"


def test_full_retrieval_error_is_propagated_without_api_or_choice() -> None:
    question = _question()
    retrieval = {
        "question_id": question["question_id"],
        "video_id": question["video_id"],
        "status": "error",
        "error_type": "RuntimeError",
        "error_message": "graph retrieval returned no evidence",
        "uses_gold_answer": False,
    }

    result = prediction_from_retrieval_error(
        question,
        retrieval,
        model="gpt-5.6-luna",
        qa_profile="nextqa_graph_only_mcq_luna_v1",
        graph_retrieval_profile="nextqa_graph_only_unified_v2",
        answer_mode="live",
    )

    assert result["status"] == "error"
    assert result["graph_rag_status"] == "graph_only_retrieval_error"
    assert result["predicted_choice_index"] is None
    assert result["predicted_choice_text"] is None
    assert result["attempt_count"] == 0
    assert result["evidence_budget_per_choice"] == [0] * 5
    assert result["uses_gold_answer"] is False


def test_exact_choice_evaluation_counts_errors_as_incorrect() -> None:
    runtime = [
        {
            "question_id": "q1",
            "video_id": "v1",
            "official_type": "CW",
            "answer_choices": ["a", "b", "c", "d", "e"],
        },
        {
            "question_id": "q2",
            "video_id": "v2",
            "official_type": "TP",
            "answer_choices": ["a", "b", "c", "d", "e"],
        },
    ]
    labels = [
        {"question_id": "q1", "gold_choice_index": 1},
        {"question_id": "q2", "gold_choice_index": 3},
    ]
    predictions = [
        {
            "question_id": "q1",
            "status": "completed",
            "predicted_choice_index": 1,
            "predicted_choice_text": "b",
            "graph_record_ids": ["episode:1"],
            "evidence_sufficiency": "sufficient",
        },
        {
            "question_id": "q2",
            "status": "error",
            "predicted_choice_index": None,
            "predicted_choice_text": None,
            "graph_record_ids": ["episode:2"],
        },
    ]

    metrics = evaluate_exact_choice_accuracy(predictions, labels, runtime)

    assert metrics["overall_accuracy"] == 0.5
    assert metrics["completed_only_accuracy"] == 1.0
    assert metrics["error_question_count"] == 1
    assert metrics["provisional_error_attribution"] == {
        "QA_generation_or_parse_gap": 1
    }


def test_exact_choice_evaluation_attributes_retrieval_errors_to_recall() -> None:
    runtime = [
        {
            "question_id": "q1",
            "video_id": "v1",
            "official_type": "CW",
            "answer_choices": ["a", "b", "c", "d", "e"],
        }
    ]
    labels = [{"question_id": "q1", "gold_choice_index": 1}]
    predictions = [
        {
            "question_id": "q1",
            "status": "error",
            "error_stage": "graph_retrieval",
            "graph_rag_status": "graph_only_retrieval_error",
            "predicted_choice_index": None,
            "predicted_choice_text": None,
            "graph_record_ids": [],
        }
    ]

    metrics = evaluate_exact_choice_accuracy(predictions, labels, runtime)

    assert metrics["provisional_error_attribution"] == {"candidate_recall_gap": 1}


def test_nextqa_qa_code_is_dataset_and_visual_isolated() -> None:
    paths = [
        REPO_ROOT / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'qa' / 'graph_only_qa.py',
        REPO_ROOT / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'qa' / 'evaluation.py',
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'qa' / 'run_graph_only_qa_pilot.py',
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'audit' / 'evaluate_graph_only_qa_pilot.py',
    ]
    forbidden_roots = {"faiss", "PIL", "cv2", "torch", "open_clip"}
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
        assert not any(name.split(".")[0] in forbidden_roots for name in imports)


def test_qa_runner_has_no_evaluator_label_or_official_type_input() -> None:
    source = (
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'qa' / 'run_graph_only_qa_pilot.py'
    ).read_text(encoding="utf-8").lower()
    assert "evaluator_labels" not in source
    assert "gold_choice_index" not in source
    assert 'row["official_type"]' not in source
    assert 'row.get("official_type")' not in source


def test_config_rejects_a_non_gpt5mini_model(nextqa_tmp_dir: Path) -> None:
    config_path = nextqa_tmp_dir / "bad.yaml"
    config_path.write_text(
        CONFIG.read_text(encoding="utf-8").replace("gpt-5.6-luna", "gpt-4.1-mini"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires gpt-5.6-luna"):
        load_graph_only_mcq_config(config_path)
