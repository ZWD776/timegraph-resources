from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import causal_how_uniform, descriptive_count_uniform
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.route_policy_v3 import TARGET_IDENTITIES, TARGET_ROUTE_IDS, materialize_visual_record


def _route(route_id: str) -> dict:
    return {"question_id": "q1", "video_id": "v1", "route_id": route_id, "route_source": "nextqa_luna_ir_intent_only"}


def _uniform() -> dict:
    return {"question_id": "q1", "video_id": "v1", "graph_retrieval_status": "completed", "graph_retrieval_profile": "frozen", "graph_record_ids": ["g1"], "graph_record_ids_sha256": "hash", "graph_evidence_count": 1, "visual_evidence": [{"rank": 1, "candidate_id": "c1", "candidate_kind": "frame", "similarity": 0.9, "text": "visual candidate kind=frame; objects=person."}], "visual_candidate_ids": ["c1"], "uses_gold_answer": False, "uses_official_type": False, "uses_answer_choices": False, "uses_route": False, "uses_handler": False, "uses_intent": False, "uses_visual_evidence_demand": False}


def _routed(route_id: str) -> dict:
    return {"question_id": "q1", "video_id": "v1", "route_id": route_id, "handler_id": f"nextqa_visual_handler_v2:{route_id}", "visual_evidence": [{"rank": 1, "role_id": "role", "text": "old"}]}


@pytest.mark.parametrize("module", [causal_how_uniform, descriptive_count_uniform])
def test_uniform_handlers_keep_route_identity_and_exact_evidence(module) -> None:
    source = _uniform()
    record = module.build(_route(module.ROUTE_ID), source)
    assert record["route_id"] == module.ROUTE_ID
    assert record["strategy_id"] == module.STRATEGY_ID
    assert record["handler_id"] == module.HANDLER_ID
    assert record["visual_evidence"] == source["visual_evidence"]
    assert all("role_id" not in row for row in record["visual_evidence"])


def test_two_routes_have_distinct_handlers_and_same_strategy_family() -> None:
    assert TARGET_ROUTE_IDS == {"causal_how", "descriptive_count"}
    assert len({item["handler_id"] for item in TARGET_IDENTITIES.values()}) == 2
    assert causal_how_uniform.STRATEGY_ID != descriptive_count_uniform.STRATEGY_ID


def test_non_target_route_is_byte_equivalent_copy() -> None:
    routed = _routed("temporal_next")
    frozen = deepcopy(routed)
    output = materialize_visual_record(_route("temporal_next"), routed, _uniform())
    assert output == frozen
    assert output is not routed


def test_uniform_handler_rejects_wrong_route_and_impure_source() -> None:
    with pytest.raises(ValueError, match="requires route_id"):
        causal_how_uniform.build(_route("temporal_next"), _uniform())
    source = _uniform()
    source["uses_official_type"] = True
    with pytest.raises(ValueError, match="purity"):
        causal_how_uniform.build(_route("causal_how"), source)
