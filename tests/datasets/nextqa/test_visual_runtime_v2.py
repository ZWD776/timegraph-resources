from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vlkg.datasets.nextqa.runtime.visual_runtime_v2.handlers import HANDLERS, dispatch_handler
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import build_strict_before_after_candidates
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.route_contracts import (
    INTENT_CONTRACTS,
    audit_forbidden_field_isolation,
    compile_route_record,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.visual_qa import (
    build_visual_qa_messages,
    load_visual_mcq_config,
)


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())


def _ir(intent: str = "causal_why") -> dict[str, object]:
    contract = INTENT_CONTRACTS[intent]
    return {
        "question_id": "nextqa-val/v1/q1",
        "video_id": "v1",
        "normalized_question": "why did the person move the object",
        "intent": intent,
        "answer_target": sorted(contract["answer_targets"])[0],
        "visual_evidence_demand": contract["visual_evidence_demand"],
        "anchor_event": {"text": "the person moved the object"},
        "referenced_event": "the person moved the object",
        "target_entities": ["person", "object"],
        "actors": ["person"],
        "actions": ["move"],
        "objects": ["object"],
        "states": [],
        "answer_choices": ["a", "b", "c", "d", "e"],
        "official_type": "CW",
        "uses_gold_answer": False,
        "status": "completed",
    }


def _hit(kind: str, candidate_id: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "video_id": "v1",
        "graph_bound": True,
    }
    if kind == "object":
        metadata.update(entity_id="e1", label="person", timestamp_sec=1.0, state_values=[])
    elif kind == "tracklet":
        metadata.update(entity_id="e1", label="person", labels_seen=["person"], states_seen=[], start_ts=0.0, end_ts=2.0)
    elif kind == "frame":
        metadata.update(frame_name="frame_1", frame_index=1, timestamp_sec=1.0, labels=["person"], states=[], entity_ids=["e1"])
    elif kind == "temporal_window":
        metadata.update(start_ts=0.0, end_ts=2.0, labels=["person"], states=[], entity_ids=["e1"], relations=[], events=[])
    elif kind == "event_boundary":
        metadata.update(event_id="ev", event_type="STATE_CHANGE", timestamp_sec=1.0, labels=["person"], states=[], entity_ids=["e1"])
    elif kind == "before_after_pair":
        metadata.update(
            event_id="ev", event_type="STATE_CHANGE", boundary_ts=1.0,
            before_frame_name="f0", before_ts=0.0, before_labels=["person"],
            before_states=[], before_entity_ids=["e1"], after_frame_name="f2",
            after_ts=2.0, after_labels=["person"], after_states=[],
            after_entity_ids=["e1"], strict_temporal_pair=True, pair_status="complete",
            missing_sides=[],
        )
    return {
        "candidate_id": candidate_id, "candidate_kind": kind, "video_id": "v1",
        "rank_within_kind": 1, "similarity": 0.9, "metadata": metadata,
    }


def test_route_is_exactly_luna_intent_and_forbidden_fields_are_invariant() -> None:
    for intent in INTENT_CONTRACTS:
        row = _ir(intent)
        route = compile_route_record(row)
        assert route["route_id"] == intent
        assert route["route_source"] == "nextqa_luna_ir_intent_only"
        assert route["uses_question_text_routing"] is False
        assert all(audit_forbidden_field_isolation(row).values())


def test_route_rejects_demand_or_target_inconsistent_with_intent() -> None:
    bad = _ir("causal_why")
    bad["visual_evidence_demand"] = "direct_visual"
    with pytest.raises(ValueError, match="visual_evidence_demand is inconsistent"):
        compile_route_record(bad)
    bad = _ir("causal_why")
    bad["answer_target"] = "action"
    with pytest.raises(ValueError, match="answer_target is inconsistent"):
        compile_route_record(bad)


def test_strict_pair_builder_never_uses_boundary_frame_on_either_side() -> None:
    frames = [
        {"frame_name": f"f{i}", "frame_index": i, "timestamp_sec": float(i), "labels": ["person"], "states": [], "entity_ids": ["e"]}
        for i in range(3)
    ]
    vectors = [np.eye(3, dtype=np.float32)[i] for i in range(3)]
    graph = {
        "video_id": "nextqa-val/v1",
        "transition_events": [
            {"event_id": "start", "timestamp_sec": 0.0},
            {"event_id": "middle", "timestamp_sec": 1.0},
            {"event_id": "end", "timestamp_sec": 2.0},
        ],
    }
    metadata, output_vectors, audit = build_strict_before_after_candidates(graph, frames, vectors)
    by_event = {row["event_id"]: row for row in metadata}
    assert by_event["middle"]["before_frame_name"] == "f0"
    assert by_event["middle"]["after_frame_name"] == "f2"
    assert by_event["start"]["before_ts"] is None
    assert by_event["start"]["after_ts"] == 1.0
    assert by_event["end"]["before_ts"] == 1.0
    assert by_event["end"]["after_ts"] is None
    assert len(output_vectors) == 3
    assert audit["same_frame_pair_count"] == 0
    assert audit["strict_order_violation_count"] == 0


def test_all_eight_handlers_are_independent_and_obey_role_budgets() -> None:
    assert set(HANDLERS) == set(INTENT_CONTRACTS)
    assert len({handler.__module__ for handler in HANDLERS.values()}) == 8
    for intent in INTENT_CONTRACTS:
        route = compile_route_record(_ir(intent))
        role_hits: dict[str, list[dict[str, object]]] = {}
        for query in route["evidence_demand"]["query_intents"]:
            hits = []
            for kind in query["candidate_kinds"]:
                for index in range(int(query["search_top_k"])):
                    hits.append(_hit(str(kind), f"{query['role_id']}:{kind}:{index}"))
            role_hits[str(query["role_id"])] = hits
        pack = dispatch_handler(route, role_hits)
        assert pack["handler_id"] == route["handler_id"]
        assert pack["handler_module"] == intent
        assert pack["evidence"]
        assert len(pack["evidence"]) <= int(route["evidence_demand"]["evidence_limit"])
        if intent not in {"descriptive_count"}:
            selected_kinds = {row["candidate_kind"] for row in pack["evidence"]}
            expected_kinds = {
                kind
                for query in route["evidence_demand"]["query_intents"]
                for kind, quota in query["kind_quotas"].items()
                if quota > 0
            }
            assert selected_kinds == expected_kinds


def test_visual_qa_prompt_is_text_only_and_hides_route_type(tmp_path: Path) -> None:
    config = load_visual_mcq_config(REPO_ROOT / 'configs' / 'datasets' / 'nextqa' / 'visual_qa.yaml')
    question = {
        **_ir("causal_why"),
        "target_entities": ["person"], "actors": ["person"], "actions": ["move"],
        "objects": ["object"], "temporal_direction": "none", "temporal_anchor": "none",
        "causal_query": "why", "count_query": False, "location_query": False,
        "attribute_query": "none", "answer_format": "multiple_choice_5",
    }
    graph = {
        "question_id": question["question_id"], "video_id": "v1", "status": "completed",
        "graph_retrieval_profile": "nextqa_graph_only_unified_v2",
        "graph_record_ids": ["r1"],
        "evidence": [{"rank": 1, "source_record_id": "r1", "source_kind": "event", "time_range": {}, "evidence_text": "person moved"}],
        "uses_gold_answer": False, "uses_official_type": False,
    }
    visual = {
        "question_id": question["question_id"], "video_id": "v1",
        "route_id": "causal_why", "handler_id": "nextqa_visual_handler_v2:causal_why",
        "handler_status": "completed", "graph_record_ids": ["r1"],
        "visual_evidence": [{"rank": 1, "role_id": "effect_event", "text": "Effect event at 1.000s."}],
        "used_visual_candidate_ids": ["v1"], "uses_gold_answer": False,
        "uses_official_type": False, "uses_answer_choices": False,
        "qa_receives_images": False, "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
    }
    messages = build_visual_qa_messages(question, graph, visual, config=config)
    serialized = json.dumps(messages, ensure_ascii=False).lower()
    assert "official_type" not in serialized
    assert '"intent"' not in serialized
    assert '"route_id"' not in serialized
    assert "image_path" not in serialized
    assert "embedding" not in serialized
    assert ".png" not in serialized
    prompt = json.loads(messages[1]["content"])
    assert prompt["policy"]["same_combined_evidence_budget_for_every_choice"] == 2
    assert len(prompt["answer_choices"]) == 5


def test_visual_v2_code_has_no_navqa_or_openeqa_adapter_imports() -> None:
    root = REPO_ROOT / "src" / "vlkg" / "datasets" / "nextqa" / "runtime" / "visual_runtime_v2"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "eval_adapters.navqa" not in source
    assert "eval_adapters.openeqa" not in source
