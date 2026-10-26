from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import (
    adapt_visual_payload_for_navqa_backbone,
    navqa_equivalent_question_ir,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.question_route_hybrid_backbone import (
    fuse_question_route_hybrid_backbone as fork_fuse,
    question_route_backbone_policy as fork_policy,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.route_contracts import (
    ROUTE_BINARY_EXISTENCE,
    ROUTE_COUNT_SET,
    ROUTE_EGO_MOTION_SIDE_TURN,
    ROUTE_ENTITY_ATTRIBUTE,
    ROUTE_ENTITY_RELATION_SPATIAL,
    ROUTE_OCR_LOGO_TEXT,
    ROUTE_OPEN_DESCRIPTIVE,
    ROUTE_TEMPORAL_INTERVAL,
    ROUTE_TEMPORAL_POINT,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_faiss_grounding import (
    VisualFaissConfig as ForkVisualFaissConfig,
)
from vlkg.datasets.navqa.retrieval.question_route_hybrid_backbone import (
    fuse_question_route_hybrid_backbone as navqa_fuse,
    question_route_backbone_policy as navqa_policy,
)
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import VisualFaissConfig as NavqaVisualFaissConfig


def _graph() -> CanonicalTemporalMemoryGraph:
    return CanonicalTemporalMemoryGraph.model_validate(
        {
            "video_id": "parity-episode",
            "entity_nodes": [
                {
                    "node_id": "node-chair",
                    "video_id": "parity-episode",
                    "entity_id": "chair-1",
                    "label": "chair",
                    "time_range": {"start_frame": 0, "end_frame": 4},
                    "observation_count": 3,
                    "attributes": {"raw_visual_attributes": ["blue fabric chair"]},
                },
                {
                    "node_id": "node-table",
                    "video_id": "parity-episode",
                    "entity_id": "table-1",
                    "label": "table",
                    "time_range": {"start_frame": 0, "end_frame": 4},
                    "observation_count": 3,
                },
            ],
            "episode_nodes": [
                {
                    "node_id": "state-chair",
                    "video_id": "parity-episode",
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
                    "video_id": "parity-episode",
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


def _graph_answer() -> dict:
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


def _raw_visual_payload() -> dict:
    return {
        "contract": "openeqa_episode_scoped_visual_search.v1",
        "episode_history": "parity-episode",
        "episode_scoped_search": True,
        "groups": [
            {
                "query_intent": {
                    "text": "chair color",
                    "role": "target",
                    "source": "target_attribute_or_state",
                },
                "hits": [
                    {
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
                ],
            },
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "source": "spatial_anchor",
                },
                "hits": [
                    {
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
                ],
            },
        ],
    }


ROUTES = (
    ROUTE_BINARY_EXISTENCE,
    ROUTE_COUNT_SET,
    ROUTE_EGO_MOTION_SIDE_TURN,
    ROUTE_ENTITY_ATTRIBUTE,
    ROUTE_ENTITY_RELATION_SPATIAL,
    ROUTE_OCR_LOGO_TEXT,
    ROUTE_OPEN_DESCRIPTIVE,
    ROUTE_TEMPORAL_INTERVAL,
    ROUTE_TEMPORAL_POINT,
)


@pytest.mark.parametrize("route_id", ROUTES)
def test_visual_runtime_v1_policy_matches_frozen_navqa(route_id: str) -> None:
    route = {"route_id": route_id}
    assert fork_policy(route, specialized=False) == navqa_policy(
        route, specialized=False
    )
    assert fork_policy(route, specialized=True) == navqa_policy(
        route, specialized=True
    )


CASES = (
    (
        "What color is the chair?",
        {
            "primary_need": "entity_attribute",
            "answer_shape": "text",
            "target_terms": ["chair"],
            "attribute_dimensions": ["color"],
        },
    ),
    (
        "How many chairs are there?",
        {
            "primary_need": "count_set",
            "answer_shape": "count",
            "target_terms": ["chair"],
        },
    ),
    (
        "What is left of the table?",
        {
            "primary_need": "spatial_localization",
            "answer_shape": "text",
            "target_terms": ["chair"],
            "anchor_terms": ["table"],
            "spatial_relations": ["left_of"],
            "spatial_constraints": [
                {"relation": "left_of", "anchor": "table", "target": "chair"}
            ],
        },
    ),
    (
        "What text is written on the chair?",
        {
            "primary_need": "ocr_logo_text",
            "answer_shape": "text",
            "target_terms": ["chair"],
            "attribute_dimensions": ["text"],
        },
    ),
    (
        "Is there a chair?",
        {
            "primary_need": "existence_or_state",
            "answer_shape": "binary",
            "target_terms": ["chair"],
        },
    ),
    (
        "What object is near the table?",
        {
            "primary_need": "object_identity",
            "answer_shape": "text",
            "target_terms": ["chair"],
            "anchor_terms": ["table"],
        },
    ),
)


@pytest.mark.parametrize(("question", "demand"), CASES)
def test_visual_runtime_v1_fusion_matches_frozen_navqa(
    question: str,
    demand: dict,
) -> None:
    graph = _graph()
    question_ir = navqa_equivalent_question_ir({}, evidence_demand=demand)
    payload = adapt_visual_payload_for_navqa_backbone(
        _raw_visual_payload(),
        graph=graph,
        question=question,
        question_ir=question_ir,
    )
    config_values = {
        "max_lines": 8,
        "ger_object_limit": 8,
        "ger_context_limit": 8,
        "local_subgraph_neighbor_limit": 6,
    }
    navqa_result = navqa_fuse(
        deepcopy(_graph_answer()),
        deepcopy(payload),
        graph=graph,
        config=NavqaVisualFaissConfig(**config_values),
        question=question,
        question_ir=deepcopy(question_ir),
        specialized=True,
        retrieval_window=None,
    )
    fork_result = fork_fuse(
        deepcopy(_graph_answer()),
        deepcopy(payload),
        graph=graph,
        config=ForkVisualFaissConfig(**config_values),
        question=question,
        question_ir=deepcopy(question_ir),
        specialized=True,
        retrieval_window=None,
    )
    assert fork_result == navqa_result
