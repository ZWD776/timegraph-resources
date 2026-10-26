from __future__ import annotations

import inspect

import numpy as np

import pytest

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.qa.canonical_graph_qa_context import (
    retrieve_canonical_graph_context,
)
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import (
    CanonicalHybridRetrievalSession,
)
from vlkg.datasets.openeqa.retrieval.evidence_router import compile_evidence_demand
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import (
    OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT,
    adapt_visual_payload_for_navqa_backbone,
    fuse_navqa_equivalent_visual_evidence,
    navqa_equivalent_question_ir,
    navqa_route_id_for_evidence_demand,
)
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import (
    VISUAL_BACKBONE_EVIDENCE_DEMAND_V4C_GUARDED,
    VISUAL_BACKBONE_EVIDENCE_DEMAND_V4D_GUARDED,
    VISUAL_BACKBONE_EVIDENCE_DEMAND_V4_GUARDED,
    VISUAL_BACKBONE_EVIDENCE_DEMAND_V3,
    VISUAL_BACKBONE_GENERIC_GER_V1,
    VISUAL_BACKBONE_LEGACY_V4,
    VISUAL_BACKBONE_NAVQA_EQUIVALENT,
    VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1,
    VISUAL_BACKBONE_ROUTE_HYBRID_V2,
    VISUAL_BUILDER_EVIDENCE_DEMAND_V2,
    VISUAL_BUILDER_EVIDENCE_DEMAND_V4_GUARDED,
    VISUAL_BUILDER_EVIDENCE_DEMAND_V4C_GUARDED,
    VISUAL_BUILDER_EVIDENCE_DEMAND_V4D_GUARDED,
    answer_visual_enhanced_canonical_query,
    generic_ger_visual_demand,
    select_visual_builder,
)
from vlkg.datasets.navqa.retrieval.question_route_hybrid_backbone import (
    BACKBONE_ENTITY_PACK_SCORE_CONTRACT,
    BACKBONE_GER_ATTRIBUTE_OWNER,
    BACKBONE_GER_OBJECT_LOCAL_RELATION,
    BACKBONE_GER_V3_BINDING,
)
from vlkg.datasets.navqa.retrieval.question_router import (
    ROUTE_BINARY_EXISTENCE,
    ROUTE_COUNT_SET,
    ROUTE_ENTITY_ATTRIBUTE,
    ROUTE_ENTITY_RELATION_SPATIAL,
    ROUTE_OCR_LOGO_TEXT,
    ROUTE_OPEN_DESCRIPTIVE,
)
from vlkg.core.memory.types import MemoryQuery


def _graph() -> CanonicalTemporalMemoryGraph:
    return CanonicalTemporalMemoryGraph.model_validate(
        {
            "video_id": "episode-1",
            "entity_nodes": [
                {
                    "node_id": "node-chair",
                    "video_id": "episode-1",
                    "entity_id": "chair-1",
                    "label": "chair",
                    "time_range": {"start_frame": 0, "end_frame": 4},
                    "observation_count": 3,
                    "attributes": {
                        "raw_visual_attributes": ["blue fabric chair", "wooden legs"]
                    },
                },
                {
                    "node_id": "node-table",
                    "video_id": "episode-1",
                    "entity_id": "table-1",
                    "label": "table",
                    "time_range": {"start_frame": 0, "end_frame": 4},
                    "observation_count": 3,
                },
            ],
            "episode_nodes": [
                {
                    "node_id": "state-chair",
                    "video_id": "episode-1",
                    "episode_id": "state-chair",
                    "episode_kind": "STATE",
                    "label": "chair color blue",
                    "time_range": {"start_frame": 0, "end_frame": 4},
                    "primary_entity_id": "chair-1",
                    "attr_key": "color",
                    "attr_value": "blue",
                    "source_ref_id": "source-state",
                },
                {
                    "node_id": "relation-chair-table",
                    "video_id": "episode-1",
                    "episode_id": "relation-chair-table",
                    "episode_kind": "RELATION",
                    "label": "chair left of table",
                    "time_range": {"start_frame": 2, "end_frame": 2},
                    "subject_entity_id": "chair-1",
                    "object_entity_id": "table-1",
                    "predicate": "left_of",
                    "source_ref_id": "source-relation",
                },
            ],
        }
    )


def _graph_result() -> dict:
    return {
        "answer_brief": "graph answer",
        "llm_ready_bundle": {
            "answer_brief": "graph answer",
            "evidence_lines": ["kind=graph label='chair'"],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "uncertainty_notes": [],
            "llm_context_text": "Graph context",
            "structured_graph_context": {},
        },
        "retrieval_bundle": {"evidence_digest": []},
    }


def _payload() -> dict:
    chair = {
        "faiss_id": 1,
        "graph_frame_index": 2,
        "object_index": 0,
        "label": "chair",
        "bbox": [0, 0, 10, 10],
        "similarity": 0.8,
        "observation_id": "obs-chair",
        "canonical_entity_id": "chair-1",
        "binding_kind": "exact_frame_object_to_canonical_entity",
        "binding_confidence": 1.0,
    }
    table = {
        "faiss_id": 2,
        "graph_frame_index": 2,
        "object_index": 1,
        "label": "table",
        "bbox": [20, 0, 30, 10],
        "similarity": 0.75,
        "observation_id": "obs-table",
        "canonical_entity_id": "table-1",
        "binding_kind": "exact_frame_object_to_canonical_entity",
        "binding_confidence": 1.0,
    }
    return {
        "contract": "openeqa_episode_scoped_visual_search.v1",
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "query_intent_count": 2,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {
                    "text": "chair color",
                    "role": "target",
                    "source": "target_attribute_or_state",
                },
                "hits": [chair],
            },
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "source": "spatial_anchor",
                },
                "hits": [table],
            },
        ],
    }


class _FakeVisualIndex:
    def search_evidence_demand(self, **_kwargs) -> dict:
        return _payload()

    @staticmethod
    def encode_texts(texts: list[str]) -> list[np.ndarray]:
        return [
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in texts
        ]


@pytest.mark.parametrize(
    ("need", "route_id"),
    [
        ("object_identity", ROUTE_OPEN_DESCRIPTIVE),
        ("entity_attribute", ROUTE_ENTITY_ATTRIBUTE),
        ("existence_or_state", ROUTE_BINARY_EXISTENCE),
        ("spatial_localization", ROUTE_ENTITY_RELATION_SPATIAL),
        ("count_set", ROUTE_COUNT_SET),
        ("ocr_logo_text", ROUTE_OCR_LOGO_TEXT),
        ("functional_world", ROUTE_OPEN_DESCRIPTIVE),
        ("open_descriptive", ROUTE_OPEN_DESCRIPTIVE),
    ],
)
def test_openeqa_need_maps_to_frozen_navqa_route(need: str, route_id: str) -> None:
    assert navqa_route_id_for_evidence_demand({"primary_need": need}) == route_id


def test_visual_payload_adapter_supplies_navqa_owner_and_object_local_fields() -> None:
    demand = {
        "primary_need": "spatial_localization",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "spatial_relations": ["left_of"],
        "spatial_constraints": [
            {"relation": "left_of", "anchor": "table", "target": "chair"}
        ],
    }
    ir = navqa_equivalent_question_ir({}, evidence_demand=demand)
    payload = adapt_visual_payload_for_navqa_backbone(
        _payload(),
        graph=_graph(),
        question="What is left of the table?",
        question_ir=ir,
    )

    assert ir["question_route"]["route_id"] == ROUTE_ENTITY_RELATION_SPATIAL
    assert all("results" in group for group in payload["groups"])
    chair = payload["groups"][0]["results"][0]
    assert chair["entity_id"] == "chair-1"
    assert any("attr=color=blue" in fact for fact in chair["hsgg_visual_facts"])
    assert any("relation=chair left_of table" in fact for fact in chair["hsgg_visual_facts"])
    assert any("label='table'" in fact for fact in chair["hsgg_frame_context"])


def test_entity_attribute_uses_navqa_candidate_pack_score_contract() -> None:
    demand = {
        "primary_need": "entity_attribute",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "attribute_dimensions": ["color"],
        "state_terms": [],
        "constraint_terms": [],
        "confidence": 0.94,
    }
    fused = fuse_navqa_equivalent_visual_evidence(
        _graph_result(),
        _payload(),
        graph=_graph(),
        question="What color is the chair?",
        question_ir={"target_entities": ["chair"], "attribute_dimension": "color"},
        evidence_demand=demand,
    )

    bundle = fused["llm_ready_bundle"]
    backbone = bundle["question_route_hybrid_backbone"]
    assert backbone["route_backbone_id"] == BACKBONE_ENTITY_PACK_SCORE_CONTRACT
    assert bundle["score_contract"]["enabled"] is True
    assert bundle["visual_entity_candidate_packs"]["packs"][0]["anchor_entity_id"] == "chair-1"
    assert fused["visual_evidence"]["contract"] == OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT


def test_spatial_and_open_routes_use_navqa_ger_backbones() -> None:
    spatial_demand = {
        "primary_need": "spatial_localization",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "spatial_relations": ["left_of"],
        "spatial_constraints": [
            {"relation": "left_of", "anchor": "table", "target": "chair"}
        ],
        "confidence": 0.94,
    }
    spatial = fuse_navqa_equivalent_visual_evidence(
        _graph_result(),
        _payload(),
        graph=_graph(),
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=spatial_demand,
    )
    spatial_bundle = spatial["llm_ready_bundle"]
    assert (
        spatial_bundle["question_route_hybrid_backbone"]["route_backbone_id"]
        == BACKBONE_GER_OBJECT_LOCAL_RELATION
    )
    assert spatial_bundle["visual_faiss_object_local_fact_lines"]

    open_demand = {
        "primary_need": "object_identity",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "confidence": 0.94,
    }
    opened = fuse_navqa_equivalent_visual_evidence(
        _graph_result(),
        _payload(),
        graph=_graph(),
        question="What object is near the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=open_demand,
    )
    assert (
        opened["llm_ready_bundle"]["question_route_hybrid_backbone"][
            "route_backbone_id"
        ]
        == BACKBONE_GER_V3_BINDING
    )


def test_count_and_ocr_use_navqa_specialized_evidence_builders() -> None:
    count_demand = {
        "primary_need": "count_set",
        "answer_shape": "count",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "confidence": 0.97,
    }
    counted = fuse_navqa_equivalent_visual_evidence(
        _graph_result(),
        _payload(),
        graph=_graph(),
        question="How many chairs are there?",
        question_ir={"target_entities": ["chair"], "aggregation": "count"},
        evidence_demand=count_demand,
    )
    count_set = counted["llm_ready_bundle"]["question_route_count_set"]
    assert count_set["count"] == 1
    assert "entity_id" in count_set["dedupe_key"]

    ocr_demand = {
        "primary_need": "ocr_logo_text",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "attribute_dimensions": ["text"],
        "confidence": 0.96,
    }
    ocr = fuse_navqa_equivalent_visual_evidence(
        _graph_result(),
        _payload(),
        graph=_graph(),
        question="What text is written on the chair?",
        question_ir={"target_entities": ["chair"], "attribute_dimension": "text"},
        evidence_demand=ocr_demand,
    )
    bundle = ocr["llm_ready_bundle"]
    assert (
        bundle["question_route_hybrid_backbone"]["route_backbone_id"]
        == BACKBONE_GER_ATTRIBUTE_OWNER
    )
    assert bundle["visual_faiss_ger_fusion"]["binding"][
        "attribute_owner_enabled"
    ] is True


def test_route_hybrid_selects_specialized_builder_without_temporal_route() -> None:
    signature = inspect.signature(answer_visual_enhanced_canonical_query)
    assert (
        signature.parameters["visual_backbone"].default
        == VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1
    )
    for need in (
        "object_identity",
        "entity_attribute",
        "existence_or_state",
        "spatial_localization",
        "count_set",
        "ocr_logo_text",
        "functional_world",
        "open_descriptive",
    ):
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BACKBONE_NAVQA_EQUIVALENT
        )
    for need in ("spatial_localization", "count_set", "ocr_logo_text"):
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_ROUTE_HYBRID_V2,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BACKBONE_NAVQA_EQUIVALENT
        )
    for need in (
        "object_identity",
        "entity_attribute",
        "existence_or_state",
        "functional_world",
        "open_descriptive",
    ):
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_EVIDENCE_DEMAND_V4D_GUARDED,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BUILDER_EVIDENCE_DEMAND_V4D_GUARDED
        )
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_ROUTE_HYBRID_V2,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BACKBONE_LEGACY_V4
        )
    assert "temporal" not in {
        navqa_route_id_for_evidence_demand({"primary_need": need})
        for need in (
            "object_identity",
            "entity_attribute",
            "existence_or_state",
            "spatial_localization",
            "count_set",
            "ocr_logo_text",
            "functional_world",
            "open_descriptive",
        )
    }


def test_visual_mainline_preserves_graph_retrieval_source_ids() -> None:
    graph = _graph()
    question = "What color is the chair?"
    decision = compile_evidence_demand(
        question,
        category="attribute recognition",
        question_ir={"target_entities": ["chair"], "attribute_dimension": "color"},
    ).to_dict()
    question_ir = {
        "target_entities": ["chair"],
        "attribute_dimension": "color",
        "question_route": decision,
    }
    query = MemoryQuery(
        query_text=question,
        metadata={
            "graph_retrieval_profile": "navqa_temporal",
            "question_ir": question_ir,
        },
        limit=10,
    )
    graph_only = retrieve_canonical_graph_context(
        graph,
        query,
        retrieval_mode="temporal",
        limit=10,
        retrieval_session=CanonicalHybridRetrievalSession(graph),
    )
    visual = answer_visual_enhanced_canonical_query(
        graph,
        query,
        question=question,
        episode_history="episode-1",
        category="attribute recognition",
        question_ir=question_ir,
        visual_index=_FakeVisualIndex(),
        retrieval_mode="temporal",
        answer_mode="graph_only",
        limit=10,
        retrieval_session=CanonicalHybridRetrievalSession(graph),
    )

    graph_only_ids = [row["source_record_id"] for row in graph_only["matches"]]
    visual_ids = [
        row["source_record_id"] for row in visual["graph_answer"]["matches"]
    ]
    assert visual_ids == graph_only_ids
    assert visual["question_route_decision_source"] == "question_ir.question_route"
    assert visual["selected_visual_builder"] == VISUAL_BACKBONE_NAVQA_EQUIVALENT


def test_new_visual_backbones_are_versioned_and_generic_control_is_not_routed() -> None:
    for need in (
        "object_identity",
        "entity_attribute",
        "existence_or_state",
        "spatial_localization",
        "count_set",
        "ocr_logo_text",
        "functional_world",
    ):
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_EVIDENCE_DEMAND_V4C_GUARDED,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BUILDER_EVIDENCE_DEMAND_V4C_GUARDED
        )
        assert (
            select_visual_builder(
                VISUAL_BACKBONE_EVIDENCE_DEMAND_V4_GUARDED,
                evidence_demand={"primary_need": need},
            )
            == VISUAL_BUILDER_EVIDENCE_DEMAND_V4_GUARDED
        )
    assert (
        select_visual_builder(
            VISUAL_BACKBONE_EVIDENCE_DEMAND_V3,
            evidence_demand={"primary_need": "spatial_localization"},
        )
        == VISUAL_BUILDER_EVIDENCE_DEMAND_V2
    )
    assert (
        select_visual_builder(
            VISUAL_BACKBONE_EVIDENCE_DEMAND_V3,
            evidence_demand={"primary_need": "existence_or_state"},
        )
        == VISUAL_BACKBONE_LEGACY_V4
    )
    assert (
        select_visual_builder(
            VISUAL_BACKBONE_GENERIC_GER_V1,
            evidence_demand={"primary_need": "entity_attribute"},
        )
        == VISUAL_BACKBONE_GENERIC_GER_V1
    )
    generic = generic_ger_visual_demand(
        "What color is the chair beside the table?",
        {
            "answer_shape": "text",
            "target_terms": ["chair"],
            "anchor_terms": ["table"],
        },
    )
    assert generic["route_id"] == "generic_ger"
    assert generic["metadata"]["question_route_conditioned"] is False
    assert {item["role"] for item in generic["query_intents"]} <= {
        "target",
        "context",
    }
    assert generic["temporal_route_enabled"] is False
