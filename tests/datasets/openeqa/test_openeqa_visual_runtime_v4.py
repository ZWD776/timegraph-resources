from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import (
    VISUAL_BACKBONE_OPENEQA_QR_V4,
    VISUAL_BUILDER_ROUTE_DISPATCHED_V4,
    select_visual_builder,
)
from vlkg.datasets.openeqa.retrieval.evidence_router import (
    ROUTING_PROFILE_IR_FIRST_V4,
    STATE_STRATEGY_VARIANTS,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v3 import (
    fuse_spatial_guarded_visual_evidence,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import (
    ACTIVE_HANDLER_VARIANTS,
    FROZEN_V3_DELEGATION_VARIANTS,
    OPENEQA_VISUAL_RUNTIME_CONTRACT,
    fuse_route_dispatched_visual_evidence,
    handler_provenance,
    route_handler_contract_issues,
    route_handler_id,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.open_target_relation import (
    _bind_unbound_anchor_hits,
    _geometry_precision_score,
    _generic_target,
    _group_anchor_hits,
    _merge_relation_pairs,
    _target_semantic_scores,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.entity_attribute import (
    _attribute_consensus,
    _publish_owner_attribute_evidence,
    _required_route_binding_tokens,
    _select_owner_candidate_packs,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.existence_or_state import (
    _open_closed_consensus,
    _publish_open_closed_evidence,
    _select_open_closed_candidate_packs,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.surface_capacity import (
    _publish_surface_capacity_evidence,
    _select_surface_observations,
)


def _graph_result() -> dict:
    return {
        "answer_brief": "graph answer",
        "llm_ready_bundle": {
            "answer_brief": "graph answer",
            "evidence_lines": ["kind=entity label='table'"],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
        "retrieval_bundle": {"evidence_digest": []},
    }


def _demand(need: str) -> dict:
    return {
        "primary_need": need,
        "strategy_variant": f"{need}_baseline",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "spatial_relations": [],
        "spatial_constraints": [],
        "attribute_dimensions": [],
        "state_terms": [],
        "constraint_terms": [],
        "query_intents": [],
    }


def test_v4_is_an_explicit_mainline_builder() -> None:
    assert (
        select_visual_builder(
            VISUAL_BACKBONE_OPENEQA_QR_V4,
            evidence_demand=_demand("spatial_localization"),
        )
        == VISUAL_BUILDER_ROUTE_DISPATCHED_V4
    )


@pytest.mark.parametrize(
    ("need", "variant", "expected_handler"),
    [
        (
            "spatial_localization",
            "target_location",
            "spatial_localization.variant_v1",
        ),
        (
            "spatial_localization",
            "open_target_relation",
            "spatial_localization.open_target_relation_graph_only_v1",
        ),
        (
            "spatial_localization",
            "relation_description_or_verification",
            "spatial_localization.relation_description.frozen_v3",
        ),
        (
            "spatial_localization",
            "unregistered_spatial_variant",
            "spatial_localization.frozen_v3",
        ),
        (
            "object_identity",
            "anchor_bound_open_target_identity",
            "object_identity.anchor_bound_relation_incremental_v1",
        ),
        (
            "count_set",
            "capacity_or_functional_count",
            "count_set.capacity_or_functional.frozen_v3",
        ),
        (
            "entity_attribute",
            "owner_attribute",
            "entity_attribute.owner_locked_consensus_v2",
        ),
        (
            "existence_or_state",
            "state_or_existence",
            "existence_or_state.frozen_v3",
        ),
        (
            "existence_or_state",
            "surface_capacity",
            "existence_or_state.surface_capacity_observation_v1",
        ),
        (
            "existence_or_state",
            "open_closed_state",
            "existence_or_state.open_closed_consensus_v1",
        ),
        (
            "functional_world",
            "grounded_function_or_world",
            "functional_world.frozen_v3",
        ),
        (
            "ocr_logo_text",
            "owner_aware_text",
            "ocr_logo_text.owner_aware_text_boundary_v1",
        ),
        (
            "open_descriptive",
            "scene_or_content_description",
            "open_descriptive.scene_or_content_boundary_v1",
        ),
        (
            "open_descriptive",
            "unsupported_fallback",
            "open_descriptive.unsupported.frozen_v3",
        ),
    ],
)
def test_v4_dispatches_by_persisted_primary_need_and_variant(
    need: str,
    variant: str,
    expected_handler: str,
) -> None:
    demand = _demand(need)
    demand["strategy_variant"] = variant
    demand["metadata"] = {"category": "deliberately unrelated"}

    assert route_handler_id(demand) == expected_handler


def test_v4_handler_provenance_contract_has_disjoint_active_and_frozen_sets() -> None:
    assert ACTIVE_HANDLER_VARIANTS.isdisjoint(FROZEN_V3_DELEGATION_VARIANTS)
    assert ACTIVE_HANDLER_VARIANTS == {
        ("spatial_localization", "target_location"),
        ("spatial_localization", "open_target_relation"),
        ("object_identity", "anchor_bound_open_target_identity"),
        ("existence_or_state", "surface_capacity"),
        ("existence_or_state", "open_closed_state"),
        ("entity_attribute", "owner_attribute"),
        ("ocr_logo_text", "owner_aware_text"),
        ("open_descriptive", "scene_or_content_description"),
    }
    assert all(
        handler_provenance(
            {
                "primary_need": need,
                "strategy_variant": variant,
            }
        )
        == "active_v4"
        for need, variant in ACTIVE_HANDLER_VARIANTS
    )
    assert all(
        not route_handler_id(
            {"primary_need": need, "strategy_variant": variant}
        ).endswith(".frozen_v3")
        for need, variant in ACTIVE_HANDLER_VARIANTS
    )
    assert all(
        handler_provenance(
            {
                "primary_need": need,
                "strategy_variant": variant,
            }
        )
        == "frozen_v3"
        for need, variant in FROZEN_V3_DELEGATION_VARIANTS
    )


def test_v4_contract_rejects_unregistered_variant_fallback() -> None:
    demand = _demand("spatial_localization")
    demand.update(
        {
            "strategy_variant": "new_variant_not_in_contract",
            "route_strategy_handler": "spatial_localization.frozen_v3",
            "metadata": {"routing_profile": ROUTING_PROFILE_IR_FIRST_V4},
        }
    )

    issues = route_handler_contract_issues(demand)

    assert handler_provenance(demand) == "unregistered_fallback"
    assert len(issues) == 1
    assert "not registered" in issues[0]


@pytest.mark.parametrize("need,variant", sorted(FROZEN_V3_DELEGATION_VARIANTS))
def test_v4_frozen_delegation_is_explicitly_registered(
    need: str,
    variant: str,
) -> None:
    demand = _demand(need)
    demand.update(
        {
            "strategy_variant": variant,
            "route_strategy_handler": route_handler_id(
                {"primary_need": need, "strategy_variant": variant}
            ),
            "metadata": {"routing_profile": ROUTING_PROFILE_IR_FIRST_V4},
        }
    )

    assert handler_provenance(demand) == "frozen_v3"
    assert route_handler_contract_issues(demand) == []
    assert route_handler_id(demand).endswith(".frozen_v3")


@pytest.mark.parametrize(
    "variant",
    sorted(STATE_STRATEGY_VARIANTS - {"surface_capacity", "open_closed_state"}),
)
def test_v4_state_variants_explicitly_delegate_frozen_v3(variant: str) -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = {"groups": []}
    demand = _demand("existence_or_state")
    demand["strategy_variant"] = variant
    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Is the object open?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    candidate = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Is the object open?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert route_handler_id(demand) == "existence_or_state.frozen_v3"
    assert candidate["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert candidate["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert candidate["visual_evidence"]["frozen_v3_delegation"] is True


def test_v4_direct_identity_remains_exact_frozen_v3_evidence() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = {"groups": []}
    demand = _demand("object_identity")
    demand["strategy_variant"] = "direct_identity"
    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What object is visible?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    candidate = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What object is visible?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert candidate["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert candidate["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert candidate["visual_evidence"]["spatial_pairs"] == baseline[
        "visual_evidence"
    ]["spatial_pairs"]
    assert candidate["visual_evidence"]["registered_handler_provenance"] == (
        "frozen_v3"
    )
    assert candidate["visual_evidence"]["execution_evidence_provenance"] == (
        "frozen_v3"
    )


@pytest.mark.parametrize(
    ("need", "variant", "question", "fallback_reason"),
    [
        (
            "ocr_logo_text",
            "owner_aware_text",
            "What is today's date on the calendar?",
            "no_dedicated_ocr_observation_layer",
        ),
        (
            "open_descriptive",
            "scene_or_content_description",
            "What type of room is this?",
            "no_dedicated_scene_description_layer",
        ),
    ],
)
def test_phase5_boundary_only_handlers_preserve_exact_frozen_prompt(
    need: str,
    variant: str,
    question: str,
    fallback_reason: str,
) -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = {"groups": []}
    demand = _demand(need)
    demand["strategy_variant"] = variant
    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question=question,
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    candidate = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question=question,
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert candidate["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert candidate["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert candidate["visual_evidence"]["route_strategy_applied"] is False
    assert candidate["visual_evidence"]["frozen_v3_delegation"] is True
    assert candidate["visual_evidence"]["handler_fallback_reason"] == fallback_reason


@pytest.mark.parametrize(
    "need",
    ["spatial_localization", "object_identity", "count_set", "entity_attribute"],
)
def test_v4_initial_handlers_preserve_v3_answer_visible_evidence(need: str) -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = {"groups": []}
    demand = _demand(need)
    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is visible?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    candidate = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is visible?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert candidate["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert candidate["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert candidate["visual_evidence"]["spatial_pairs"] == baseline[
        "visual_evidence"
    ]["spatial_pairs"]
    assert candidate["visual_evidence"]["visual_runtime_contract"] == (
        OPENEQA_VISUAL_RUNTIME_CONTRACT
    )
    assert candidate["visual_evidence"]["strategy_variant"] == (
        f"{need}_baseline"
    )


def test_owner_attribute_exact_owner_suppresses_unrelated_target_query() -> None:
    unrelated = {
        "anchor_entity_id": "lamp-1",
        "entity_label": "floor lamp",
        "candidate_role": "target_query",
        "target_hits": ["ceiling"],
        "direct_attributes": ["white lamp hanging from ceiling"],
        "constraint_hits": [],
        "retrieval_score": 0.95,
    }
    owner = {
        "anchor_entity_id": "ceiling-1",
        "entity_label": "ceiling",
        "candidate_role": "visual_recall_candidate",
        "target_hits": [],
        "direct_attributes": ["wooden ceiling"],
        "constraint_hits": ["living", "room"],
        "retrieval_score": 0.10,
    }

    selected = _select_owner_candidate_packs(
        [unrelated, owner],
        target_phrases=["ceiling"],
        limit=3,
    )

    assert [pack["anchor_entity_id"] for pack in selected] == ["ceiling-1"]
    assert selected[0]["owner_binding"]["strict_owner_match"] is True
    assert selected[0]["owner_binding"]["uses_attribute_text_for_owner_binding"] is False

    assert _select_owner_candidate_packs(
        [
            {
                "anchor_entity_id": "ceiling-fan-1",
                "entity_label": "ceiling fan",
                "candidate_role": "target_query",
                "direct_attributes": ["white ceiling fan"],
            }
        ],
        target_phrases=["ceiling"],
        limit=3,
    ) == []

    patio_required = _required_route_binding_tokens(
        evidence_demand={
            "target_terms": ["patio chairs"],
            "anchor_terms": [],
            "constraint_terms": ["patio chairs"],
        }
    )
    assert patio_required == {"patio", "chair"}
    assert _select_owner_candidate_packs(
        [
            {
                "anchor_entity_id": "chairs-1",
                "entity_label": "chairs",
                "candidate_role": "target_query",
                "target_hits": ["chairs"],
                "constraint_hits": ["chairs"],
                "direct_attributes": ["black chair in kitchen"],
            }
        ],
        target_phrases=["patio chairs"],
        required_binding_tokens=patio_required,
        limit=3,
    ) == []

    assert _required_route_binding_tokens(
        evidence_demand={
            "target_terms": ["two trash cans"],
            "anchor_terms": [],
            "constraint_terms": [],
        }
    ) == {"trash", "can"}
    assert _select_owner_candidate_packs(
        [
            {
                "anchor_entity_id": "bed-1",
                "entity_label": "bed",
                "candidate_role": "target_query",
                "direct_attributes": ["blue bed with a white comforter"],
            }
        ],
        target_phrases=["bed comforter"],
        limit=3,
    ) == []


def test_open_closed_state_requires_owner_and_route_binding() -> None:
    selected = _select_open_closed_candidate_packs(
        [
            {
                "anchor_entity_id": "front-door-1",
                "entity_label": "door",
                "target_hits": ["front", "door"],
                "direct_attributes": ["the front door is open"],
            },
            {
                "anchor_entity_id": "cabinet-door-1",
                "entity_label": "cabinet door",
                "target_hits": ["door"],
                "direct_attributes": ["the cabinet door is closed"],
            },
        ],
        target_phrases=["front door"],
        required_tokens={"front", "door"},
        limit=4,
    )

    assert [pack["owner_entity_id"] for pack in selected] == ["front-door-1"]
    consensus = _open_closed_consensus(selected)
    assert consensus["available"] is True
    assert consensus["observed_state"] == "open"
    assert consensus["conflict_detected"] is False


def test_open_closed_state_conflict_restores_exact_graph_prompt() -> None:
    selected = _select_open_closed_candidate_packs(
        [
            {
                "anchor_entity_id": "door-1",
                "entity_label": "door",
                "target_hits": ["door"],
                "direct_attributes": ["open door", "closed door"],
            }
        ],
        target_phrases=["door"],
        required_tokens={"door"},
        limit=4,
    )
    consensus = _open_closed_consensus(selected)
    graph_bundle = deepcopy(_graph_result()["llm_ready_bundle"])
    fused = {
        "llm_ready_bundle": {"evidence_lines": ["wrong visual evidence"]},
        "visual_evidence": {"candidate_packs": [{"image_path": "forbidden.png"}]},
    }
    frozen_bundle = deepcopy(fused["llm_ready_bundle"])
    frozen_packs = deepcopy(fused["visual_evidence"]["candidate_packs"])

    result = _publish_open_closed_evidence(
        fused,
        graph_bundle=graph_bundle,
        raw_pack_count=1,
        selected_packs=selected,
        consensus=consensus,
        target_phrases=["door"],
        requested_states=["open"],
        required_tokens={"door"},
        max_visual_lines=8,
    )

    assert consensus["conflict_detected"] is True
    assert result["llm_ready_bundle"] == frozen_bundle
    assert result["visual_evidence"]["candidate_packs"] == frozen_packs
    assert result["visual_evidence"]["graph_only_fallback"] is True


def test_open_closed_state_publishes_text_only_consensus() -> None:
    selected = _select_open_closed_candidate_packs(
        [
            {
                "anchor_entity_id": "front-door-1",
                "entity_label": "door",
                "target_hits": ["front", "door"],
                "direct_attributes": ["front door is open"],
                "visual_anchors": [{"image_path": "forbidden.png"}],
            }
        ],
        target_phrases=["front door"],
        required_tokens={"front", "door"},
        limit=4,
    )
    consensus = _open_closed_consensus(selected)
    fused = {
        "llm_ready_bundle": {
            "evidence_lines": ["kind=frozen_graph_evidence"],
            "visual_entity_candidate_pack_lines": ["kind=frozen_visual_candidate"],
        },
        "visual_evidence": {
            "candidate_packs": [{"entity_label": "frozen candidate"}],
        },
    }
    frozen_evidence_lines = deepcopy(fused["llm_ready_bundle"]["evidence_lines"])
    frozen_pack_lines = deepcopy(
        fused["llm_ready_bundle"]["visual_entity_candidate_pack_lines"]
    )
    frozen_packs = deepcopy(fused["visual_evidence"]["candidate_packs"])
    result = _publish_open_closed_evidence(
        fused,
        graph_bundle=deepcopy(_graph_result()["llm_ready_bundle"]),
        raw_pack_count=1,
        selected_packs=selected,
        consensus=consensus,
        target_phrases=["front door"],
        requested_states=["open"],
        required_tokens={"front", "door"},
        max_visual_lines=8,
    )

    serialized_bundle = str(result["llm_ready_bundle"])
    serialized_visual = str(result["visual_evidence"])
    assert "observed_state=open" in serialized_bundle
    assert "forbidden.png" not in serialized_bundle
    assert "forbidden.png" not in serialized_visual
    assert result["llm_ready_bundle"]["evidence_lines"] == frozen_evidence_lines
    assert result["llm_ready_bundle"][
        "visual_entity_candidate_pack_lines"
    ] == frozen_pack_lines
    assert result["visual_evidence"]["candidate_packs"] == frozen_packs
    assert result["visual_evidence"]["graph_only_fallback"] is False


def test_owner_attribute_conflicting_observations_are_explicit() -> None:
    consensus = _attribute_consensus(
        {
            "entity_label": "chair",
            "direct_attributes": [
                "black chair",
                "brown chair",
                "black upholstered chair",
            ],
        },
        target_phrases=["bedroom chair"],
        attribute_dimensions=["color"],
    )

    assert consensus["conflict_detected"] is True
    assert consensus["agreement_state"] == "partial_token_consensus_with_conflict"
    assert consensus["consensus_descriptor_tokens"][0] == {
        "token": "black",
        "support_count": 2,
    }
    assert consensus["first_observation_is_not_privileged"] is True


def test_owner_attribute_no_owner_restores_graph_only_prompt_fields() -> None:
    graph_bundle = deepcopy(_graph_result()["llm_ready_bundle"])
    fused = _graph_result()
    fused["llm_ready_bundle"].update(
        {
            "evidence_lines": ["rank=1 kind=candidate_pack label='lamp'"],
            "evidence_aggregation_lines": ["group=candidate_pack:1"],
            "evidence_aggregation": {"candidate_facts": [{"wrong": True}]},
            "visual_entity_candidate_pack_lines": ["wrong visual pack"],
            "visual_evidence_contract": "frozen-v3",
        }
    )
    fused["visual_evidence"] = {
        "candidate_packs": [{"entity_label": "lamp"}],
        "stats": {},
    }
    frozen_bundle = deepcopy(fused["llm_ready_bundle"])
    frozen_packs = deepcopy(fused["visual_evidence"]["candidate_packs"])

    result = _publish_owner_attribute_evidence(
        fused,
        graph_bundle=graph_bundle,
        raw_packs=list(fused["visual_evidence"]["candidate_packs"]),
        selected_packs=[],
        target_phrases=["ceiling"],
        attribute_dimensions=["material"],
        required_binding_tokens={"ceiling"},
        max_visual_lines=8,
    )

    assert result["llm_ready_bundle"] == frozen_bundle
    assert result["visual_evidence"]["candidate_packs"] == frozen_packs
    assert result["visual_evidence"]["owner_attribute_available"] is False
    assert result["visual_evidence"]["route_strategy_applied"] is False
    assert result["visual_evidence"]["graph_only_fallback"] is True


def test_owner_attribute_adds_only_supplemental_owner_evidence() -> None:
    fused = {
        "llm_ready_bundle": {
            "evidence_lines": ["kind=frozen_graph_evidence"],
            "visual_entity_candidate_pack_lines": ["kind=frozen_visual_candidate"],
            "structured_graph_context": {"video_id": "episode-1"},
        },
        "visual_evidence": {
            "candidate_packs": [{"entity_label": "frozen candidate"}],
            "stats": {},
        },
    }
    selected = [
        {
            "anchor_entity_id": "ceiling-1",
            "entity_label": "ceiling",
            "owner_binding": {
                "matched_target_phrase": "ceiling",
                "overlap_token_count": 1,
            },
            "owner_route_binding": {"complete": True},
            "owner_attribute_consensus": {
                "observations": ["wooden ceiling"],
                "attribute_dimensions": ["material"],
                "consensus_descriptor_tokens": [],
                "conflict_detected": False,
            },
        }
    ]
    frozen_evidence_lines = deepcopy(fused["llm_ready_bundle"]["evidence_lines"])
    frozen_pack_lines = deepcopy(
        fused["llm_ready_bundle"]["visual_entity_candidate_pack_lines"]
    )
    frozen_packs = deepcopy(fused["visual_evidence"]["candidate_packs"])

    result = _publish_owner_attribute_evidence(
        fused,
        graph_bundle=deepcopy(_graph_result()["llm_ready_bundle"]),
        raw_packs=deepcopy(frozen_packs),
        selected_packs=selected,
        target_phrases=["ceiling"],
        attribute_dimensions=["material"],
        required_binding_tokens={"ceiling"},
        max_visual_lines=8,
    )

    assert result["llm_ready_bundle"]["evidence_lines"] == frozen_evidence_lines
    assert result["llm_ready_bundle"][
        "visual_entity_candidate_pack_lines"
    ] == frozen_pack_lines
    assert result["visual_evidence"]["candidate_packs"] == frozen_packs
    assert result["llm_ready_bundle"][
        "question_route_specialized_evidence_lines"
    ][0].startswith("kind=owner_attribute_guidance")
    assert result["visual_evidence"]["route_strategy_applied"] is True


def test_surface_capacity_aggregates_owner_observations_without_replacing_frozen() -> None:
    candidates = _select_surface_observations(
        [
            {
                "source_record_id": "desk-pack",
                "anchor_entity_id": "desk-1",
                "entity_label": "desk",
                "direct_attributes": [
                    "desk covered with a laptop, food, and drinks",
                    "cluttered desk with electronics",
                ],
            },
            {
                "source_record_id": "chair-pack",
                "anchor_entity_id": "chair-1",
                "entity_label": "chair",
                "direct_attributes": ["empty chair"],
            },
        ],
        target_phrases=["desk"],
        limit=6,
    )
    assert len(candidates) == 1
    assert candidates[0]["owner"] == "desk-1"
    assert len(candidates[0]["observations"]) == 2

    fused = {
        "llm_ready_bundle": {
            "evidence_lines": ["kind=frozen_graph_evidence"],
            "visual_entity_candidate_pack_lines": ["kind=frozen_visual_candidate"],
            "structured_graph_context": {},
        },
        "visual_evidence": {
            "candidate_packs": [{"entity_label": "desk"}],
            "stats": {},
        },
    }
    frozen_evidence_lines = deepcopy(fused["llm_ready_bundle"]["evidence_lines"])
    frozen_pack_lines = deepcopy(
        fused["llm_ready_bundle"]["visual_entity_candidate_pack_lines"]
    )
    frozen_packs = deepcopy(fused["visual_evidence"]["candidate_packs"])

    result = _publish_surface_capacity_evidence(
        fused,
        candidates=candidates,
        raw_pack_count=2,
        target_phrases=["desk"],
        question="Is there any room to work on the desks?",
        max_visual_lines=8,
    )

    assert result["llm_ready_bundle"]["evidence_lines"] == frozen_evidence_lines
    assert result["llm_ready_bundle"][
        "visual_entity_candidate_pack_lines"
    ] == frozen_pack_lines
    assert result["visual_evidence"]["candidate_packs"] == frozen_packs
    assert result["llm_ready_bundle"]["structured_graph_context"][
        "surface_capacity_evidence"
    ]["plural_or_existential_scope"] is True
    assert "clutter_is_direct_occupancy_evidence=true" in result[
        "llm_ready_bundle"
    ]["question_route_specialized_evidence_lines"][0]
    assert result["visual_evidence"]["route_strategy_applied"] is True


def test_surface_capacity_without_observations_keeps_exact_frozen_bundle() -> None:
    fused = {
        "llm_ready_bundle": {"evidence_lines": ["kind=frozen_graph_evidence"]},
        "visual_evidence": {"candidate_packs": [], "stats": {}},
    }
    frozen_bundle = deepcopy(fused["llm_ready_bundle"])

    result = _publish_surface_capacity_evidence(
        fused,
        candidates=[],
        raw_pack_count=0,
        target_phrases=["desk"],
        question="Is there room on my desk?",
        max_visual_lines=8,
    )

    assert result["llm_ready_bundle"] == frozen_bundle
    assert result["visual_evidence"]["candidate_packs"] == []
    assert result["visual_evidence"]["route_strategy_applied"] is False
    assert result["visual_evidence"]["handler_fallback_reason"] == (
        "no_owner_bound_surface_observations"
    )


def test_v4_target_location_adds_entity_pack_without_changing_other_routes() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:chair-1",
                video_id="episode-1",
                entity_id="chair-1",
                label="chair",
                time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
                observation_count=1,
            )
        ],
    )
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "groups": [
            {
                "query_intent": {
                    "text": "chair",
                    "role": "target",
                    "source": "target_identity",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "observation_id": "obs-1",
                        "graph_frame_index": 2,
                        "object_index": 1,
                        "label": "chair",
                        "bbox": [0, 0, 10, 10],
                        "similarity": 0.9,
                        "canonical_entity_id": "chair-1",
                    }
                ],
            }
        ],
    }
    spatial_demand = _demand("spatial_localization")
    spatial_demand["strategy_variant"] = "target_location"
    identity_demand = _demand("object_identity")
    identity_demand["strategy_variant"] = "target_location"

    spatial = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Where is the chair?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=spatial_demand,
    )
    identity = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is the chair?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=identity_demand,
    )
    frozen_spatial = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Where is the chair?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=spatial_demand,
    )

    assert spatial["visual_evidence"]["route_strategy_applied"] is True
    assert spatial["visual_evidence"]["frozen_v3_delegation"] is False
    assert any(
        "kind=visual_entity_candidate_pack" in line
        for line in spatial["llm_ready_bundle"]["visual_entity_candidate_pack_lines"]
    )
    assert len(spatial["llm_ready_bundle"]["visual_entity_candidate_pack_lines"]) <= 4
    for field in (
        "evidence_lines",
        "evidence_aggregation",
        "evidence_aggregation_lines",
        "llm_context_text",
        "structured_graph_context",
    ):
        assert spatial["llm_ready_bundle"].get(field) == frozen_spatial[
            "llm_ready_bundle"
        ].get(field)
    assert "route_strategy_applied" not in identity["visual_evidence"]
    assert identity["visual_evidence"]["frozen_v3_delegation"] is True


def test_anchor_bound_identity_adds_relation_candidate_incrementally() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity-plant",
                video_id="episode-1",
                entity_id="plant-1",
                label="plant",
                time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
                observation_count=1,
            ),
            CanonicalEntityNode(
                node_id="entity-mirror",
                video_id="episode-1",
                entity_id="mirror-1",
                label="mirror",
                time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
                observation_count=1,
            ),
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode-relation",
                video_id="episode-1",
                episode_id="relation-1",
                episode_kind="RELATION",
                label="plant left of mirror",
                time_range=CanonicalTimeRange(
                    start_frame=2,
                    end_frame=2,
                    support_frames=1,
                ),
                scene_id="scene-1",
                subject_entity_id="plant-1",
                object_entity_id="mirror-1",
                predicate="left_of",
                raw_predicate="left of",
                source_ref_id="source-1",
            )
        ],
    )
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "groups": [
            {
                "query_intent": {
                    "text": "mirror",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "observation_id": "mirror-observation",
                        "graph_frame_index": 2,
                        "object_index": 1,
                        "label": "mirror",
                        "bbox": [20, 0, 30, 10],
                        "similarity": 0.9,
                        "canonical_entity_id": "mirror-1",
                    }
                ],
            },
            {
                "query_intent": {
                    "text": "objects near mirror",
                    "role": "spatial_target",
                },
                "hits": [
                    {
                        "faiss_id": 2,
                        "observation_id": "plant-observation",
                        "graph_frame_index": 2,
                        "object_index": 2,
                        "label": "plant",
                        "bbox": [0, 0, 10, 10],
                        "similarity": 0.4,
                        "canonical_entity_id": "plant-1",
                    }
                ],
            },
        ],
    }
    demand = _demand("object_identity")
    demand.update(
        {
            "strategy_variant": "anchor_bound_open_target_identity",
            "open_target_slot": True,
            "target_terms": [],
            "anchor_terms": ["mirror"],
            "spatial_relations": ["left_of"],
            "spatial_constraints": [
                {"anchor": "mirror", "relation": "left_of"}
            ],
        }
    )

    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is to the left of the mirror?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    result = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is to the left of the mirror?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    pair = result["visual_evidence"]["object_identity_relation_candidates"][0]
    assert pair["candidate_role"] == "anchor_bound_identity_candidate"
    assert pair["canonical_entity_id"] == "plant-1"
    assert pair["anchor_entity_id"] == "mirror-1"
    assert result["visual_evidence"]["route_strategy_applied"] is True
    assert result["visual_evidence"]["frozen_v3_delegation"] is False
    assert result["visual_evidence"]["registered_handler_provenance"] == (
        "active_v4"
    )
    assert result["visual_evidence"]["execution_evidence_provenance"] == (
        "active_v4_supplemental"
    )
    assert result["visual_evidence"]["object_identity_relation_verification"][
        "relation_verified_candidates_only"
    ] is True
    assert result["llm_ready_bundle"]["evidence_lines"] == baseline[
        "llm_ready_bundle"
    ]["evidence_lines"]
    assert result["llm_ready_bundle"].get(
        "visual_entity_candidate_pack_lines"
    ) == baseline["llm_ready_bundle"].get("visual_entity_candidate_pack_lines")
    assert result["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert result["llm_ready_bundle"][
        "question_route_anchor_bound_identity_lines"
    ][
        0
    ].startswith("rank=1 kind=openeqa_anchor_bound_identity_candidate")


def test_anchor_bound_identity_falls_back_when_no_relation_is_verified() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    demand = _demand("object_identity")
    demand.update(
        {
            "strategy_variant": "anchor_bound_open_target_identity",
            "open_target_slot": True,
            "anchor_terms": ["mirror"],
            "spatial_relations": ["left_of"],
            "spatial_constraints": [
                {"anchor": "mirror", "relation": "left_of"}
            ],
        }
    )

    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        {"groups": []},
        graph=graph,
        question="What is to the left of the mirror?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    result = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        {"groups": []},
        graph=graph,
        question="What is to the left of the mirror?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert result["visual_evidence"]["route_strategy_applied"] is False
    assert result["visual_evidence"]["frozen_v3_delegation"] is True
    assert result["visual_evidence"]["object_identity_relation_verification"][
        "fallback_reason"
    ] == "no_relation_verified_candidates"
    assert result["visual_evidence"]["execution_evidence_provenance"] == (
        "frozen_v3"
    )
    assert result["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]


def test_open_target_relation_publishes_direction_preserving_role_bound_pair() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            CanonicalEntityNode(
                node_id="entity:plant-1",
                video_id="episode-1",
                entity_id="plant-1",
                label="plant",
                time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
                observation_count=1,
            ),
            CanonicalEntityNode(
                node_id="entity:mirror-1",
                video_id="episode-1",
                entity_id="mirror-1",
                label="mirror",
                time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
                observation_count=1,
            ),
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:relation-1",
                video_id="episode-1",
                episode_id="relation-1",
                episode_kind="RELATION",
                label="plant left of mirror",
                time_range=CanonicalTimeRange(
                    start_frame=2,
                    end_frame=2,
                    support_frames=1,
                ),
                scene_id="scene-1",
                subject_entity_id="plant-1",
                object_entity_id="mirror-1",
                predicate="left_of",
                raw_predicate="left of",
                source_ref_id="source-1",
            )
        ],
    )
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "groups": [
            {
                "query_intent": {
                    "text": "mirror",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "observation_id": "mirror-observation",
                        "graph_frame_index": 2,
                        "object_index": 1,
                        "label": "mirror",
                        "bbox": [20, 0, 30, 10],
                        "image_path": "C:/frames/frame_2.png",
                        "similarity": 0.9,
                        "canonical_entity_id": "mirror-1",
                    }
                ],
            },
            {
                "query_intent": {
                    "text": "objects co-visible with mirror",
                    "role": "spatial_target",
                },
                "hits": [
                    {
                        "faiss_id": 2,
                        "observation_id": "plant-observation",
                        "graph_frame_index": 2,
                        "object_index": 2,
                        "label": "plant",
                        "bbox": [0, 0, 10, 10],
                        "image_path": "C:/frames/frame_2.png",
                        "similarity": 0.4,
                        "canonical_entity_id": "plant-1",
                    }
                ],
            },
        ],
    }
    demand = _demand("spatial_localization")
    demand.update(
        {
            "strategy_variant": "open_target_relation",
            "open_target_slot": True,
            "target_terms": [],
            "anchor_terms": ["mirror"],
            "spatial_relations": ["left_of"],
            "spatial_constraints": [
                {"anchor": "mirror", "relation": "left_of"}
            ],
        }
    )

    result = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is to the left of the mirror?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    pair = result["visual_evidence"]["spatial_pairs"][0]
    assert pair["target_entity_id"] == "plant-1"
    assert pair["anchor_entity_id"] == "mirror-1"
    assert pair["relation"] == "left_of"
    assert pair["canonical_relation_direction"] == "target_to_anchor"
    assert pair["uses_gold_answer"] is False
    assert result["visual_evidence"]["route_strategy_applied"] is True
    assert result["visual_evidence"]["frozen_v3_delegation"] is False
    assert result["visual_evidence"]["open_target_relation_verification"][
        "target_anchor_roles_are_separate"
    ] is True
    assert result["llm_ready_bundle"]["evidence_lines"][0].startswith(
        "kind=openeqa_open_target_relation_guidance"
    )
    assert result["llm_ready_bundle"]["evidence_lines"][1].startswith(
        "rank=1 kind=openeqa_open_target_relation_pair"
    )
    assert result["llm_ready_bundle"]["open_target_relation_image_frames"] == []
    assert result["visual_evidence"]["open_target_relation_verification"][
        "multimodal_relation_frame_count"
    ] == 0


def test_open_target_relation_change_does_not_change_other_spatial_variant() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = {"groups": []}
    demand = _demand("spatial_localization")
    demand["strategy_variant"] = "relation_description_or_verification"
    baseline = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Is the chair left of the table?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )
    candidate = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Is the chair left of the table?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    assert candidate["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert candidate["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    assert candidate["visual_evidence"]["spatial_pairs"] == baseline[
        "visual_evidence"
    ]["spatial_pairs"]
    assert candidate["visual_evidence"]["frozen_v3_delegation"] is True


def test_open_target_relation_groups_repeated_anchor_observations() -> None:
    anchor = CanonicalEntityNode(
        node_id="entity:mirror-1",
        video_id="episode-1",
        entity_id="mirror-1",
        label="mirror",
        time_range=CanonicalTimeRange(start_frame=2, end_frame=5),
        observation_count=2,
    )
    hits = [
        {
            "faiss_id": index,
            "canonical_entity_id": "mirror-1",
            "requested_relation": "left_of",
            "graph_frame_index": frame,
            "similarity": score,
        }
        for index, frame, score in ((1, 2, 0.8), (2, 5, 0.9))
    ]

    groups = _group_anchor_hits(hits, entity_by_id={"mirror-1": anchor})

    assert len(groups) == 1
    assert groups[0][0] is anchor
    assert groups[0][1] == "left_of"
    assert [hit["faiss_id"] for hit in groups[0][2]] == [1, 2]


def test_open_target_relation_binds_unique_unbound_anchor_without_gold() -> None:
    table = CanonicalEntityNode(
        node_id="entity:table-1",
        video_id="episode-1",
        entity_id="table-1",
        label="table",
        time_range=CanonicalTimeRange(start_frame=2, end_frame=4),
        observation_count=3,
    )
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1", entity_nodes=[table])
    hit = {
        "faiss_id": 7,
        "graph_frame_index": 3,
        "label": "table",
        "intent_text": "dining table",
        "requested_relation": "on",
        "canonical_entity_id": "",
    }

    bound = _bind_unbound_anchor_hits(
        [hit],
        graph=graph,
        entity_by_id={"table-1": table},
    )

    assert len(bound) == 1
    assert bound[0]["canonical_entity_id"] == "table-1"
    assert bound[0]["binding_kind"] == "route_local_label_time_anchor_resolution"
    assert bound[0]["uses_gold_answer"] is False


def test_open_target_relation_recovers_canonical_on_relation_from_unbound_anchor() -> None:
    table = CanonicalEntityNode(
        node_id="entity:table-1",
        video_id="episode-1",
        entity_id="table-1",
        label="table",
        time_range=CanonicalTimeRange(start_frame=2, end_frame=4),
        observation_count=3,
    )
    cup = CanonicalEntityNode(
        node_id="entity:cup-1",
        video_id="episode-1",
        entity_id="cup-1",
        label="cup",
        time_range=CanonicalTimeRange(start_frame=3, end_frame=3),
        observation_count=1,
    )
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[table, cup],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode:on-1",
                video_id="episode-1",
                episode_id="on-1",
                episode_kind="RELATION",
                label="cup on table",
                time_range=CanonicalTimeRange(
                    start_frame=3,
                    end_frame=3,
                    support_frames=1,
                ),
                scene_id="scene-1",
                subject_entity_id="cup-1",
                object_entity_id="table-1",
                predicate="on",
                raw_predicate="on",
                source_ref_id="source:on-1",
            )
        ],
    )
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "dining table",
                    "role": "anchor",
                    "spatial_relation": "on",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "observation_id": "table-observation",
                        "graph_frame_index": 3,
                        "label": "table",
                        "bbox": [0, 10, 30, 30],
                        "similarity": 0.8,
                        "canonical_entity_id": "",
                    }
                ],
            },
            {
                "query_intent": {
                    "text": "objects co-visible with dining table",
                    "role": "spatial_target",
                },
                "hits": [
                    {
                        "faiss_id": 2,
                        "observation_id": "cup-observation",
                        "graph_frame_index": 3,
                        "label": "cup",
                        "bbox": [8, 0, 16, 10],
                        "similarity": 0.4,
                        "canonical_entity_id": "cup-1",
                    }
                ],
            },
        ]
    }
    demand = _demand("spatial_localization")
    demand.update(
        {
            "strategy_variant": "open_target_relation",
            "open_target_slot": True,
            "target_terms": [],
            "anchor_terms": ["dining table"],
            "spatial_relations": ["on"],
            "spatial_constraints": [{"anchor": "dining table", "relation": "on"}],
        }
    )

    result = fuse_route_dispatched_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is on the dining table?",
        question_ir={"question_route": demand},
        evidence_demand=demand,
    )

    pair = result["visual_evidence"]["spatial_pairs"][0]
    assert pair["target_entity_id"] == "cup-1"
    assert pair["anchor_entity_id"] == "table-1"
    assert pair["relation"] == "on"
    assert pair["anchor_observations"][0]["binding_kind"] == (
        "route_local_label_time_anchor_resolution"
    )
    assert pair["uses_gold_answer"] is False


def test_open_target_relation_plural_generic_and_component_candidates_are_demoted() -> None:
    def pair(label: str, score: float) -> dict:
        return {
            "target_entity_id": label,
            "target_label": label,
            "relation": "below",
            "anchor_entity_id": "sink-1",
            "anchor_label": "sink",
            "pair_score": score,
            "verification_sources": ["same_frame_bbox_geometry"],
            "support_frames": [2],
            "support_frame_count": 1,
        }

    ranked = _merge_relation_pairs(
        [
            pair("tiles", 1.20),
            pair("floor drain", 1.00),
            pair("burners", 1.10),
            pair("pan", 0.95),
        ],
        limit=4,
        target_terms=[],
    )
    scores = {item["target_label"]: item["pair_score"] for item in ranked}

    assert _generic_target("tiles") is True
    assert scores["floor drain"] > scores["tiles"]
    assert scores["pan"] > scores["burners"]


def test_open_target_relation_ranking_is_label_diverse_and_geometry_aware() -> None:
    def pair(
        target_id: str,
        label: str,
        score: float,
        sources: list[str],
    ) -> dict:
        return {
            "target_entity_id": target_id,
            "target_label": label,
            "relation": "left_of",
            "anchor_entity_id": "mirror-1",
            "anchor_label": "mirror",
            "pair_score": score,
            "verification_sources": sources,
            "support_frames": [2],
            "support_frame_count": 1,
        }

    ranked = _merge_relation_pairs(
        [
            pair("chair-1", "chair", 1.8, ["canonical_relation"]),
            pair("chair-2", "chairs", 1.7, ["canonical_relation"]),
            pair("table-1", "table", 1.4, ["canonical_relation"]),
            pair("lamp-1", "lamp", 0.5, ["same_frame_bbox_geometry"]),
            pair("vase-1", "vase", 0.4, ["same_frame_bbox_geometry"]),
        ],
        limit=4,
        target_terms=[],
    )

    labels = [item["target_label"] for item in ranked]
    assert labels[:2] == ["lamp", "vase"]
    assert sum(label in {"chair", "chairs"} for label in labels) == 1


def test_open_target_relation_spatial_target_hits_affect_semantic_score() -> None:
    plant = CanonicalEntityNode(
        node_id="entity:plant-1",
        video_id="episode-1",
        entity_id="plant-1",
        label="plant",
        time_range=CanonicalTimeRange(start_frame=2, end_frame=2),
        observation_count=1,
    )

    scores = _target_semantic_scores(
        [
            {
                "retrieval_role": "spatial_target",
                "canonical_entity_id": "plant-1",
                "similarity": 0.75,
            }
        ],
        entity_by_id={"plant-1": plant},
    )

    assert scores == {"plant-1": 0.75}


def test_open_target_relation_directional_geometry_rewards_axis_alignment() -> None:
    clock_bbox = [802.0, 478.0, 842.0, 517.0]
    chair_score = _geometry_precision_score(
        {
            "relation": "left_of",
            "target_bbox": [740.0, 662.0, 801.0, 695.0],
            "anchor_bbox": clock_bbox,
        }
    )
    window_score = _geometry_precision_score(
        {
            "relation": "left_of",
            "target_bbox": [582.0, 402.0, 729.0, 711.0],
            "anchor_bbox": clock_bbox,
        }
    )

    assert window_score > chair_score


def test_open_target_relation_between_geometry_rewards_midpoint() -> None:
    anchor_boxes = [
        [353.0, 408.0, 393.0, 484.0],
        [920.0, 613.0, 1003.0, 669.0],
    ]
    centered_score = _geometry_precision_score(
        {
            "relation": "between",
            "target_bbox": [556.0, 470.0, 795.0, 618.0],
            "anchor_bbox": anchor_boxes,
        }
    )
    off_center_score = _geometry_precision_score(
        {
            "relation": "between",
            "target_bbox": [1027.0, 628.0, 1129.0, 683.0],
            "anchor_bbox": anchor_boxes,
        }
    )

    assert centered_score > off_center_score


def test_open_target_relation_duplicate_anchor_labels_are_one_constraint() -> None:
    pairs = [
        {
            "target_entity_id": "plant-1",
            "target_label": "plant",
            "relation": "left_of",
            "anchor_entity_id": anchor_id,
            "anchor_label": "mirror",
            "pair_score": score,
            "verification_sources": ["same_frame_bbox_geometry"],
            "support_frames": [frame],
            "support_frame_count": 1,
        }
        for anchor_id, score, frame in (
            ("mirror-1", 0.7, 2),
            ("mirror-2", 0.6, 3),
        )
    ]

    ranked = _merge_relation_pairs(pairs, limit=4, target_terms=[])

    assert ranked[0]["constraint_support_count"] == 1
