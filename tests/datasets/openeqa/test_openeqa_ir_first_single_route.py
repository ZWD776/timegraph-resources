from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import (
    ROUTE_CONTRACT,
    ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5,
    ROUTE_IDS,
    handler_provenance,
    route_handler_contract_issues,
    route_handler_id,
    route_visual_query_builder_id,
    route_visual_strategy_contract,
    single_route_decision_from_ir,
    single_route_decision_issues,
    fuse_single_route_visual_evidence,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.policy_registry import (
    POLICIES,
    RoutePolicy,
)


def _ir(route_id: str = "surface_capacity") -> dict:
    policy = POLICIES[route_id]
    return {
        "schema_version": "question_ir.v3",
        "dataset": "OpenEQA",
        "question_type": route_id,
        "uses_gold_answer": False,
        "route_decision_is_persisted": True,
        "route_decision_runtime_recompile_allowed": False,
        "question_route_id": route_id,
        "category": "deliberately ignored",
        "question_route": {
            "contract": ROUTE_CONTRACT,
            "route_id": route_id,
            "route_source": "semantic_question_ir.question_type",
            "handler_id": policy.handler_id,
            "visual_query_builder_id": policy.visual_query_builder_id,
            "visual_strategy_contract": policy.visual_strategy_contract,
            "uses_gold_answer": False,
            "uses_official_question_category": False,
            "runtime_route_recompile_allowed": False,
            "metadata": {
                "routing_profile": ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5,
                "official_category_used_for_routing": False,
            },
        },
    }


def test_single_route_is_directly_persisted_from_ir_question_type() -> None:
    ir = _ir()
    route = single_route_decision_from_ir(ir)

    assert route["route_id"] == "surface_capacity"
    assert route["handler_id"] == route_handler_id(route)
    assert single_route_decision_issues(ir) == []
    assert route_handler_contract_issues(route) == []


def test_single_route_does_not_reclassify_from_category_or_ir_parameters() -> None:
    baseline = _ir("entity_state")
    changed = deepcopy(baseline)
    changed["category"] = "object recognition"
    changed["target_state"] = "blue, closed, near the table"
    changed["question_route"]["metadata"]["category"] = "object recognition"

    assert single_route_decision_from_ir(changed)["route_id"] == (
        single_route_decision_from_ir(baseline)["route_id"]
    )


def test_single_route_requires_explicit_question_type_and_rejects_unknown_type() -> None:
    missing = _ir()
    missing.pop("question_type")
    assert single_route_decision_from_ir(missing) == {}

    unknown = _ir("surface_capacity")
    unknown["question_type"] = "new_runtime_subtype"
    unknown["question_route"]["route_id"] = "new_runtime_subtype"
    assert single_route_decision_from_ir(unknown) == {}


def test_single_route_registry_covers_exactly_the_flat_route_set() -> None:
    assert set(POLICIES) == set(ROUTE_IDS)
    assert len(ROUTE_IDS) == 14
    assert "unsupported_fallback" not in ROUTE_IDS
    assert len({policy.handler_id for policy in POLICIES.values()}) == 14
    assert all(
        policy.handler.__module__.startswith(
            "vlkg.datasets.openeqa.runtime.visual_runtime_v5"
        )
        for policy in POLICIES.values()
    )


def test_single_route_parameters_do_not_select_a_second_handler() -> None:
    route = single_route_decision_from_ir(_ir("entity_state"))
    expected = route_handler_id(route)

    for target_state in ("open", "powered on", "clean", "full"):
        changed = deepcopy(route)
        changed["target_state"] = target_state
        changed["spatial_constraints"] = [
            {"relation": "near", "target": "table"}
        ]
        changed["attribute_dimension"] = "state"
        assert route_handler_id(changed) == expected


@pytest.mark.parametrize("route_id", sorted(ROUTE_IDS))
def test_single_route_handler_contract_has_no_persisted_second_layer(route_id: str) -> None:
    ir = _ir(route_id)
    route = single_route_decision_from_ir(ir)
    assert route_handler_id(route) == POLICIES[route_id].handler_id
    assert handler_provenance(route) == POLICIES[route_id].provenance
    assert "strategy_variant" not in route
    assert "primary_need" not in route


def test_single_route_dispatch_does_not_persist_compatibility_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_id = "surface_capacity"
    original = POLICIES[route_id]

    def fake_handler(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "visual_evidence": {
                "primary_need": "existence_or_state",
                "strategy_variant": "surface_capacity",
                "route_strategy_variant": "surface_capacity",
                "route_strategy_handler": "legacy.handler",
                "route_strategy_applied": True,
            }
        }

    monkeypatch.setitem(
        POLICIES,
        route_id,
        RoutePolicy(
            route_id=route_id,
            primary_need=original.primary_need,
            handler_id="test.handler.v1",
            handler=fake_handler,
            handler_variant=original.handler_variant,
            provenance="active_v5",
            visual_strategy_contract=original.visual_strategy_contract,
            visual_query_builder_id=original.visual_query_builder_id,
        ),
    )
    route = single_route_decision_from_ir(_ir(route_id))
    fused = fuse_single_route_visual_evidence(
        {},
        {},
        graph=object(),
        question="",
        question_ir=_ir(route_id),
        evidence_demand=route,
    )

    evidence = fused["visual_evidence"]
    assert evidence["handler_id"] == "test.handler.v1"
    assert "primary_need" not in evidence
    assert "strategy_variant" not in evidence
    assert "route_strategy_variant" not in evidence
    assert "route_strategy_handler" not in evidence
