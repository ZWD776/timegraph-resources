from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlkg.datasets.nextqa.qa.no_routing_qa import build_no_routing_messages, validate_no_routing_input
from vlkg.datasets.nextqa.qa.graph_only_qa import GraphOnlyMCQConfig, STRICT_IR_FIELDS
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.visual_qa import (
    VisualMCQConfig,
    build_visual_qa_messages,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_no_routing import (
    ABLATION_LABEL,
    EVIDENCE_LIMIT,
    TOP_K_PER_KIND,
    render_neutral_hit,
    retrieve_uniform_candidates,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [{
            "candidate_id": f"{kwargs['kind']}:candidate",
            "candidate_kind": kwargs["kind"],
            "video_id": kwargs["video_id"],
            "rank_within_kind": 1,
            "similarity": 0.5,
            "metadata": {"label": "person", "timestamp_sec": 1.0, "graph_bound": True},
        }]


def _question() -> dict:
    return {
        "question_id": "nextqa-val/v/q1",
        "video_id": "v",
        "normalized_question": "what happens next",
        "answer_choices": ["a", "b", "c", "d", "e"],
        "answer_format": "multiple_choice_5",
        "target_entities": ["person"],
        "actors": ["person"],
        "actions": ["move"],
        "objects": [],
        "temporal_direction": "next",
        "temporal_anchor": "person moves",
        "referenced_event": "person moves",
        "causal_query": "none",
        "count_query": False,
        "location_query": False,
        "attribute_query": "none",
        "uses_gold_answer": False,
    }


def _graph() -> dict:
    return {
        "question_id": "nextqa-val/v/q1",
        "video_id": "v",
        "status": "completed",
        "graph_retrieval_profile": "test",
        "graph_record_ids": ["r1"],
        "graph_record_ids_sha256": "hash",
        "uses_gold_answer": False,
        "uses_official_type": False,
        "evidence": [{"rank": 1, "source_record_id": "r1", "evidence_text": "person moves"}],
    }


def _visual() -> dict:
    return {
        "question_id": "nextqa-val/v/q1",
        "video_id": "v",
        "ablation_label": ABLATION_LABEL,
        "graph_record_ids": ["r1"],
        "visual_evidence": [{"rank": 1, "candidate_id": "frame:candidate", "candidate_kind": "frame", "text": "visual candidate kind=frame; objects=person."}],
        "used_visual_candidate_ids": ["frame:candidate"],
        "uses_gold_answer": False,
        "uses_official_type": False,
        "uses_answer_choices": False,
        "qa_receives_images": False,
        "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
    }


def test_uniform_retrieval_uses_fixed_inventory_and_budget() -> None:
    store = FakeStore()
    hits = retrieve_uniform_candidates(store, video_id="v", query_vector=[1.0, 0.0])
    assert [call["kind"] for call in store.calls] == list(CANDIDATE_KINDS)
    assert all(call["top_k"] == TOP_K_PER_KIND for call in store.calls)
    assert len(hits) == len(CANDIDATE_KINDS)
    assert len(hits) <= EVIDENCE_LIMIT


def test_neutral_renderer_has_no_route_or_path_semantics() -> None:
    hit = {
        "candidate_kind": "before_after_pair",
        "candidate_id": "pair",
        "video_id": "v",
        "metadata": {"boundary_ts": 2.0, "before_labels": ["person"], "after_labels": ["chair"]},
    }
    rendered = render_neutral_hit(hit)
    assert rendered is not None
    assert "route" not in rendered.lower()
    assert "handler" not in rendered.lower()
    assert ".png" not in rendered.lower()


def test_no_routing_prompt_is_neutral_and_symmetric() -> None:
    messages = build_no_routing_messages(_question(), _graph(), _visual())
    serialized = json.dumps(messages, ensure_ascii=False).lower()
    prompt = json.loads(messages[1]["content"])
    assert "route_id" not in serialized
    assert "handler_id" not in serialized
    assert "official_type" not in serialized
    assert "gold_choice" not in serialized
    assert serialized.count("choice_text") == 5
    assert prompt["task"] == "nextqa_graph_plus_retrieved_visual_text_multiple_choice"
    assert set(prompt["question_ir"]) == set(STRICT_IR_FIELDS)
    assert "retrieved_graph_bound_visual_evidence" in prompt
    assert "uniform_visual_evidence" not in prompt
    assert all(row["evidence_role"] == "" for row in prompt["retrieved_graph_bound_visual_evidence"])


def test_forbidden_routing_metadata_does_not_change_prompt() -> None:
    baseline = build_no_routing_messages(_question(), _graph(), _visual())
    changed = {
        **_question(),
        "intent": "causal_why",
        "visual_evidence_demand": "preceding_context",
        "official_type": "CW",
        "gold_choice_index": 4,
        "route_id": "causal_why",
        "handler_id": "forbidden",
    }
    assert build_no_routing_messages(changed, _graph(), _visual()) == baseline


def test_prompt_matches_routed_contract_except_visual_evidence() -> None:
    transport = GraphOnlyMCQConfig(
        profile_name="nextqa_visual_enhanced_mcq_test",
        provider="openai_compatible",
        model="gpt-5.6-luna",
        api_base_url="https://example.test/v1",
        api_base_url_env_var=None,
        endpoint_path="/chat/completions",
        api_key_env_var="OPENAI_API_KEY",
        timeout_sec=300.0,
        temperature=0.1,
        max_tokens=600,
        max_attempts=2,
        evidence_limit=16,
        thinking_enabled=False,
    )
    config = VisualMCQConfig(transport=transport, graph_evidence_limit=16, visual_evidence_limit=12)
    routed_visual = {
        **_visual(),
        "route_id": "temporal_next",
        "handler_id": "nextqa_visual_handler_v2:temporal_next",
        "handler_status": "completed",
        "visual_evidence": [{"rank": 1, "role_id": "following_window", "text": "routed evidence"}],
    }
    no_routing_messages = build_no_routing_messages(_question(), _graph(), _visual())
    routed_messages = build_visual_qa_messages(_question(), _graph(), routed_visual, config=config)
    assert no_routing_messages[0] == routed_messages[0]
    no_routing_prompt = json.loads(no_routing_messages[1]["content"])
    routed_prompt = json.loads(routed_messages[1]["content"])
    no_routing_evidence = no_routing_prompt.pop("retrieved_graph_bound_visual_evidence")
    routed_evidence = routed_prompt.pop("retrieved_graph_bound_visual_evidence")
    assert no_routing_prompt == routed_prompt
    assert all(row["evidence_role"] == "" for row in no_routing_evidence)
    assert routed_evidence[0]["evidence_role"] == "following_window"


def test_no_routing_rejects_route_fields_and_preserves_graph_ids() -> None:
    visual = _visual()
    visual["route_id"] = "temporal_next"
    with pytest.raises(ValueError, match="route fields"):
        validate_no_routing_input(_question(), _graph(), visual)

    visual = _visual()
    visual["graph_record_ids"] = ["changed"]
    with pytest.raises(ValueError, match="frozen graph record IDs"):
        validate_no_routing_input(_question(), _graph(), visual)


def test_no_routing_source_is_isolated() -> None:
    root = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    source = (root / "src" / "vlkg" / "datasets" / "nextqa" / "runtime" / "visual_runtime_no_routing.py").read_text(encoding="utf-8")
    qa_source = (root / "src" / "vlkg" / "datasets" / "nextqa" / "qa" / "no_routing_qa.py").read_text(encoding="utf-8")
    runner_source = (root / "scripts" / "datasets" / "nextqa" / "qa" / "run_visual_no_routing_qa.py").read_text(encoding="utf-8")
    assert "visual_runtime_v2.handlers" not in source
    assert "EvidenceDemand" not in source
    assert "visual_runtime_v2.visual_qa" not in qa_source
    assert "STRICT_IR_FIELDS" in runner_source
    assert 'row.get("official_type")' not in runner_source
    assert 'questions_ir[qid].get("intent")' not in runner_source
