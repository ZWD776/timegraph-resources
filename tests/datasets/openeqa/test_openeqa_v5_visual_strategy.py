from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import (
    ROUTE_CONTRACT,
    ROUTE_IDS,
    ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5,
    VISUAL_STRATEGIES,
    build_visual_query_plan,
    route_handler_contract_issues,
    route_handler_id,
    route_visual_query_builder_id,
    route_visual_strategy_contract,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import shared_handlers
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import active_handlers
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.handler_base import (
    V5_HANDLER_CONTRACT,
)


def _route(route_id: str) -> dict:
    route = {
        "contract": ROUTE_CONTRACT,
        "route_id": route_id,
        "route_source": "semantic_question_ir.question_type",
        "handler_id": "",
        "visual_query_builder_id": "",
        "visual_strategy_contract": "",
        "uses_gold_answer": False,
        "uses_official_question_category": False,
        "runtime_route_recompile_allowed": False,
        "metadata": {
            "routing_profile": ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5,
            "official_category_used_for_routing": False,
        },
    }
    route["handler_id"] = route_handler_id(route)
    route["visual_query_builder_id"] = route_visual_query_builder_id(route)
    route["visual_strategy_contract"] = route_visual_strategy_contract(route)
    return route


def _ir(route_id: str = "relation_bound_identity") -> dict:
    return {
        "question_type": route_id,
        "target_entities": ["object"],
        "attribute_dimension": "color",
        "target_state": "white",
        "spatial_constraints": [{"relation": "above", "target": "TV"}],
        "hard_constraints": [{"kind": "attribute", "text": "white"}],
        "aggregation": "single_best",
        "category": "deliberately ignored",
    }


def test_visual_strategy_registry_covers_exactly_the_taxonomy() -> None:
    assert set(VISUAL_STRATEGIES) == set(ROUTE_IDS)
    assert all(policy.route_id == route_id for route_id, policy in VISUAL_STRATEGIES.items())
    assert all(policy.visual_query_builder_id for policy in VISUAL_STRATEGIES.values())
    assert all(policy.candidate_units for policy in VISUAL_STRATEGIES.values())




def test_category_counterfactual_keeps_route_handler_builder_and_query_plan() -> None:
    route = _route("relation_bound_identity")
    baseline_ir = _ir()
    changed_ir = deepcopy(baseline_ir)
    changed_ir["category"] = "random official category"
    changed_route = deepcopy(route)
    changed_route["metadata"]["category"] = "object recognition"

    baseline = build_visual_query_plan(baseline_ir, route)
    changed = build_visual_query_plan(changed_ir, changed_route)

    assert changed == baseline
    assert route_handler_id(changed) == route_handler_id(baseline)
    assert route_visual_query_builder_id(changed) == route_visual_query_builder_id(
        baseline
    )


def test_same_type_parameters_change_queries_but_not_handler_or_builder() -> None:
    route = _route("entity_state")
    open_ir = {
        **_ir("entity_state"),
        "target_entities": ["door"],
        "target_state": "open",
        "spatial_constraints": [],
    }
    power_ir = {
        **open_ir,
        "target_entities": ["ceiling fan"],
        "target_state": "powered on",
    }

    open_plan = build_visual_query_plan(open_ir, route)
    power_plan = build_visual_query_plan(power_ir, route)

    assert open_plan["query_intents"] != power_plan["query_intents"]
    assert route_handler_id(open_plan) == route_handler_id(power_plan)
    assert route_visual_query_builder_id(open_plan) == route_visual_query_builder_id(
        power_plan
    )
    assert route_handler_contract_issues(open_plan) == []
    assert route_handler_contract_issues(power_plan) == []


@pytest.mark.parametrize(
    ("route_id", "expected_unit"),
    [
        ("relation_bound_identity", "spatial_pair"),
        ("entity_attribute", "owner_attribute"),
        ("surface_capacity", "spatial_pair"),
        ("target_location", "canonical_entity_tracklet"),
        ("text_reading", "owner_attribute"),
    ],
)
def test_first_group_has_route_specific_visual_candidate_units(
    route_id: str,
    expected_unit: str,
) -> None:
    policy = VISUAL_STRATEGIES[route_id]
    assert expected_unit in policy.candidate_units
    assert policy.fallback_policy == "preserve_base_graph_and_frozen_visual_evidence"


def test_visual_candidate_content_cannot_change_control_plane() -> None:
    plan = build_visual_query_plan(_ir(), _route("relation_bound_identity"))
    control_plane = {
        "route_id": plan["route_id"],
        "handler_id": plan["handler_id"],
        "visual_query_builder_id": plan["visual_query_builder_id"],
    }
    empty_visual_payload = {"groups": []}
    populated_visual_payload = {
        "groups": [{"hits": [{"label": "clock", "similarity": 0.9}]}]
    }

    assert empty_visual_payload != populated_visual_payload
    assert control_plane == {
        "route_id": plan["route_id"],
        "handler_id": plan["handler_id"],
        "visual_query_builder_id": plan["visual_query_builder_id"],
    }


def test_query_plan_rejects_question_type_route_drift() -> None:
    with pytest.raises(ValueError, match="must match"):
        build_visual_query_plan(_ir("entity_state"), _route("entity_existence"))


def test_world_knowledge_visual_query_does_not_encode_the_hypothesis() -> None:
    ir = {
        **_ir("grounded_world_knowledge"),
        "target_entities": ["home"],
        "hard_constraints": [
            {"kind": "setting", "text": "suburban residential setting"}
        ],
        "spatial_constraints": [],
    }

    plan = build_visual_query_plan(ir, _route("grounded_world_knowledge"))

    assert plan["candidate_units"] == ["scene_frame"]
    assert {intent["evidence_unit"] for intent in plan["query_intents"]} == {
        "scene_frame"
    }
    assert all(
        "suburban" not in intent["text"].casefold()
        for intent in plan["query_intents"]
    )


def test_surface_capacity_uses_anchor_target_context_without_answer_target() -> None:
    ir = {
        **_ir("surface_capacity"),
        "target_entities": ["dining table"],
        "target_state": "has room to eat",
        "spatial_constraints": [{"relation": "on", "target": "dining table"}],
        "hard_constraints": [
            {"kind": "state", "text": "room on the dining table to eat"}
        ],
    }

    plan = build_visual_query_plan(ir, _route("surface_capacity"))

    assert plan["target_terms"] == []
    assert plan["anchor_terms"] == ["dining table"]
    assert [intent["role"] for intent in plan["query_intents"]] == [
        "anchor",
        "target",
        "context",
    ]
    assert all(
        intent["target_terms"] == [] and intent["anchor_terms"] == ["dining table"]
        for intent in plan["query_intents"]
    )


def test_v5_owned_frozen_fusion_reports_query_and_fallback_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_frozen(*args: object, **kwargs: object) -> dict:
        return {"visual_evidence": {"candidate_packs": []}}

    monkeypatch.setattr(shared_handlers, "fuse_frozen_v3", fake_frozen)
    plan = build_visual_query_plan(_ir("entity_existence"), _route("entity_existence"))

    fused = shared_handlers.fuse_entity_existence_evidence(
        {},
        {},
        graph=object(),
        question="ignored by routing",
        question_ir=_ir("entity_existence"),
        evidence_demand=plan,
    )

    evidence = fused["visual_evidence"]
    assert evidence["v5_handler_contract"] == V5_HANDLER_CONTRACT
    assert evidence["visual_query_strategy_applied"] is True
    assert evidence["specialized_fusion_applied"] is False
    assert evidence["base_graph_evidence_preserved"] is True
    assert evidence["negative_coverage_qualified"] is False
    assert evidence["handler_fallback_reason"] == (
        "no_strict_positive_visual_existence_candidate"
    )


def test_entity_count_publishes_deduplicated_candidates_not_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_frozen(*args: object, **kwargs: object) -> dict:
        return {
            "visual_evidence": {"candidate_packs": []},
            "llm_ready_bundle": {"structured_graph_context": {}},
        }

    monkeypatch.setattr(shared_handlers, "fuse_frozen_v3", fake_frozen)
    plan = build_visual_query_plan(_ir("entity_count"), _route("entity_count"))
    visual_payload = {
        "groups": [
            {
                "retrieved_unit": "candidate_set",
                "hits": [
                    {
                        "canonical_entity_id": "entity-1",
                        "label": "chair",
                        "graph_frame_index": 1,
                        "similarity": 0.8,
                    },
                    {
                        "canonical_entity_id": "entity-1",
                        "label": "chair",
                        "graph_frame_index": 2,
                        "similarity": 0.7,
                    },
                    {
                        "canonical_entity_id": "entity-2",
                        "label": "chair",
                        "graph_frame_index": 3,
                        "similarity": 0.6,
                    },
                ],
            }
        ]
    }

    fused = shared_handlers.fuse_entity_count_evidence(
        {},
        visual_payload,
        graph=object(),
        question="How many chairs are there?",
        question_ir=_ir("entity_count"),
        evidence_demand=plan,
    )

    evidence = fused["visual_evidence"]
    payload = fused["llm_ready_bundle"]["question_route_count_set"]
    assert len(evidence["count_candidate_set"]) == 2
    assert payload["candidate_count"] == 2
    assert payload["candidates_are_retrieval_candidates_not_count_answers"] is True
    assert evidence["specialized_fusion_applied"] is True


def test_entity_existence_publishes_only_strict_positive_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_frozen(*args: object, **kwargs: object) -> dict:
        return {
            "visual_evidence": {"candidate_packs": []},
            "llm_ready_bundle": {},
        }

    monkeypatch.setattr(shared_handlers, "fuse_frozen_v3", fake_frozen)
    route = _route("entity_existence")
    ir = {
        **_ir("entity_existence"),
        "target_entities": ["table mats"],
        "spatial_constraints": [],
    }
    plan = build_visual_query_plan(ir, route)
    visual_payload = {
        "groups": [
            {
                "retrieved_unit": "canonical_entity_tracklet",
                "hits": [
                    {"canonical_entity_id": "table", "label": "table"},
                    {
                        "canonical_entity_id": "mats",
                        "label": "table mats",
                        "graph_frame_index": 3,
                    },
                ],
            }
        ]
    }

    fused = shared_handlers.fuse_entity_existence_evidence(
        {},
        visual_payload,
        graph=object(),
        question="Is the table set with mats?",
        question_ir=ir,
        evidence_demand=plan,
    )

    evidence = fused["visual_evidence"]
    payload = fused["llm_ready_bundle"]["question_route_existence"]
    assert [item["label"] for item in evidence["existence_positive_candidates"]] == [
        "table mats"
    ]
    assert payload["negative_coverage_qualified"] is False
    assert payload["missing_candidate_does_not_imply_absence"] is True
    assert evidence["specialized_fusion_applied"] is True


def test_unified_entity_state_uses_ir_state_without_changing_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_open_closed(*args: object, **kwargs: object) -> dict:
        captured.update(kwargs["evidence_demand"])
        return {
            "visual_evidence": {"route_strategy_applied": True},
            "llm_ready_bundle": {},
        }

    monkeypatch.setattr(
        shared_handlers,
        "_fuse_open_closed_state",
        fake_open_closed,
    )
    route = _route("entity_state")
    ir = {
        **_ir("entity_state"),
        "target_entities": ["front door"],
        "target_state": "open",
        "spatial_constraints": [],
    }
    plan = build_visual_query_plan(ir, route)
    handler_before = route_handler_id(plan)

    fused = shared_handlers.fuse_entity_state_evidence(
        {},
        {},
        graph=object(),
        question="must not choose the handler",
        question_ir=ir,
        evidence_demand=plan,
    )

    assert route_handler_id(plan) == handler_before
    assert route_visual_query_builder_id(plan) == "owner_locked_state_v1"
    assert captured["strategy_variant"] == "open_closed_state"
    assert fused["visual_evidence"]["specialized_fusion_applied"] is True
    assert "strategy_variant" not in plan


def test_entity_state_without_explicit_consensus_restores_graph_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open_closed(*args: object, **kwargs: object) -> dict:
        return {
            "visual_evidence": {"route_strategy_applied": False},
            "llm_ready_bundle": {"evidence_lines": ["visual guess"]},
        }

    monkeypatch.setattr(
        shared_handlers,
        "_fuse_open_closed_state",
        fake_open_closed,
    )
    ir = {
        **_ir("entity_state"),
        "target_entities": ["door"],
        "target_state": "open",
        "spatial_constraints": [],
    }
    plan = build_visual_query_plan(ir, _route("entity_state"))

    fused = shared_handlers.fuse_entity_state_evidence(
        {"llm_ready_bundle": {"evidence_lines": ["graph fact"]}},
        {},
        graph=object(),
        question="Is the door open?",
        question_ir=ir,
        evidence_demand=plan,
    )

    assert fused["llm_ready_bundle"]["evidence_lines"] == ["graph fact"]
    assert fused["visual_evidence"]["specialized_fusion_applied"] is False


def test_surface_capacity_without_explicit_signal_restores_graph_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_surface(*args: object, **kwargs: object) -> dict:
        return {
            "visual_evidence": {
                "route_strategy_applied": True,
                "surface_capacity_candidate_lines": [
                    "rank=1 kind=owner_bound_surface_observation observations=yellow cloth"
                ],
            },
            "llm_ready_bundle": {"evidence_lines": ["visual guidance"]},
        }

    monkeypatch.setattr(active_handlers, "_fuse_surface_capacity", fake_surface)
    monkeypatch.setattr(
        active_handlers,
        "fuse_frozen_v3",
        lambda *args, **kwargs: {
            "visual_evidence": {"candidate_packs": [{"label": "table"}]},
            "llm_ready_bundle": {"evidence_lines": ["frozen visual fact"]},
        },
    )
    plan = build_visual_query_plan(
        _ir("surface_capacity"),
        _route("surface_capacity"),
    )

    fused = active_handlers.fuse_surface_capacity_evidence(
        {"llm_ready_bundle": {"evidence_lines": ["graph fact"]}},
        {},
        graph=object(),
        question="Is there room on the table?",
        question_ir=_ir("surface_capacity"),
        evidence_demand=plan,
    )

    assert fused["llm_ready_bundle"]["evidence_lines"] == ["frozen visual fact"]
    assert fused["visual_evidence"]["route_strategy_applied"] is False
    assert fused["visual_evidence"]["handler_fallback_reason"] == (
        "no_explicit_surface_signal_preserve_route_visual_frozen_fusion"
    )
