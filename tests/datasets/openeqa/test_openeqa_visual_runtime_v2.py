from __future__ import annotations

from copy import deepcopy

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v2.spatial_geometry import (
    SPATIAL_VERIFIED_BACKBONE_CONTRACT,
    fuse_spatial_verified_visual_evidence,
    verify_spatial_pairs,
)
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import (
    VISUAL_BACKBONE_OPENEQA_QR_V2,
    VISUAL_BUILDER_SPATIAL_VERIFIED_V2,
    select_visual_builder,
)


def _graph() -> CanonicalTemporalMemoryGraph:
    return CanonicalTemporalMemoryGraph.model_validate(
        {
            "video_id": "episode-1",
            "entity_nodes": [
                {
                    "node_id": "chair-node",
                    "video_id": "episode-1",
                    "entity_id": "chair-1",
                    "label": "chair",
                    "time_range": {"start_frame": 0, "end_frame": 2},
                    "observation_count": 1,
                },
                {
                    "node_id": "table-node",
                    "video_id": "episode-1",
                    "entity_id": "table-1",
                    "label": "table",
                    "time_range": {"start_frame": 0, "end_frame": 2},
                    "observation_count": 1,
                },
            ],
            "episode_nodes": [
                {
                    "node_id": "relation-node",
                    "video_id": "episode-1",
                    "episode_id": "relation-1",
                    "episode_kind": "RELATION",
                    "label": "chair left of table",
                    "time_range": {"start_frame": 2, "end_frame": 2},
                    "subject_entity_id": "chair-1",
                    "object_entity_id": "table-1",
                    "predicate": "left_of",
                    "source_ref_id": "source-1",
                }
            ],
        }
    )


def _hit(
    faiss_id: int,
    label: str,
    bbox: list[float],
    *,
    entity_id: str,
    frame: int = 2,
    similarity: float = 0.8,
) -> dict:
    return {
        "faiss_id": faiss_id,
        "observation_id": f"obs-{faiss_id}",
        "graph_frame_index": frame,
        "object_index": faiss_id,
        "label": label,
        "bbox": bbox,
        "similarity": similarity,
        "canonical_entity_id": entity_id,
        "binding_kind": "exact_frame_object_to_canonical_entity",
        "binding_confidence": 1.0,
    }


def _payload(*, chair_bbox: list[float] | None = None) -> dict:
    chair = _hit(1, "chair", chair_bbox or [0, 0, 10, 10], entity_id="chair-1")
    table = _hit(2, "table", [20, 0, 30, 10], entity_id="table-1", similarity=0.75)
    vase = _hit(3, "vase", [40, 0, 50, 10], entity_id="vase-1", similarity=0.75)
    return {
        "contract": "openeqa_episode_scoped_visual_search.v1",
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "query_intent_count": 3,
        "episode_candidate_count": 3,
        "groups": [
            {
                "query_intent": {"text": "chair", "role": "target"},
                "hits": [chair],
            },
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [table],
            },
            {
                "query_intent": {
                    "text": "objects co-visible with recalled spatial anchors",
                    "role": "spatial_target",
                },
                "hits": [chair, vase],
            },
        ],
    }


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


def _demand(relation: str = "left_of") -> dict:
    return {
        "primary_need": "spatial_localization",
        "route_id": "entity_relation_spatial",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "spatial_relations": [relation],
        "spatial_constraints": [{"anchor": "table", "relation": relation}],
        "open_target_slot": False,
        "attribute_dimensions": [],
        "state_terms": [],
        "constraint_terms": [],
        "query_intents": [],
    }


def test_v2_is_an_explicit_builder_and_does_not_replace_v1_default() -> None:
    assert (
        select_visual_builder(VISUAL_BACKBONE_OPENEQA_QR_V2, evidence_demand=_demand())
        == VISUAL_BUILDER_SPATIAL_VERIFIED_V2
    )


def test_left_relation_keeps_only_geometry_verified_target() -> None:
    pairs = verify_spatial_pairs(
        _payload(), graph=_graph(), evidence_demand=_demand(), limit=8
    )
    assert pairs
    assert {pair["target_label"] for pair in pairs} == {"chair"}
    assert all(pair["relation"] == "left_of" for pair in pairs)
    assert all(pair["relation_verified"] is True for pair in pairs)


def test_wrong_direction_has_no_geometry_pair_without_canonical_support() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    pairs = verify_spatial_pairs(
        _payload(chair_bbox=[40, 0, 50, 10]),
        graph=graph,
        evidence_demand=_demand(),
        limit=8,
    )
    assert pairs == []


def test_between_uses_second_anchor_from_same_frame_completion() -> None:
    first_frame = _hit(10, "picture frame", [0, 0, 10, 20], entity_id="frame-1")
    target = _hit(11, "clock", [18, 2, 28, 18], entity_id="clock-1")
    second_frame = _hit(12, "picture frame", [36, 0, 46, 20], entity_id="frame-2")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "two picture frames",
                    "role": "anchor",
                    "spatial_relation": "between",
                },
                "hits": [first_frame],
            },
            {
                "query_intent": {
                    "text": "objects co-visible with recalled spatial anchors",
                    "role": "spatial_target",
                },
                "hits": [target, second_frame],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["two picture frames"],
        "spatial_relations": ["between"],
        "spatial_constraints": [
            {"anchor": "two picture frames", "relation": "between"}
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    pairs = verify_spatial_pairs(
        payload,
        graph=CanonicalTemporalMemoryGraph(video_id="episode-1"),
        evidence_demand=demand,
        limit=8,
    )
    assert any(pair["target_label"] == "clock" for pair in pairs)
    clock_pair = next(pair for pair in pairs if pair["target_label"] == "clock")
    assert clock_pair["relation"] == "between"
    assert len(clock_pair["anchor_observations"]) == 2


def test_no_verified_pair_removes_visual_spatial_noise_and_keeps_graph() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    fused = fuse_spatial_verified_visual_evidence(
        _graph_result(),
        _payload(chair_bbox=[40, 0, 50, 10]),
        graph=graph,
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=_demand(),
    )
    visual = fused["visual_evidence"]
    assert visual["contract"] == SPATIAL_VERIFIED_BACKBONE_CONTRACT
    assert visual["available"] is False
    assert visual["candidate_packs"] == []
    assert visual["spatial_pairs"] == []
    assert visual["spatial_verification"]["fallback_to_graph_evidence"] is True
    lines = fused["llm_ready_bundle"]["evidence_lines"]
    assert any("fallback=graph_evidence_only" in line for line in lines)
    assert any("kind=entity" in line for line in lines)
    assert not any("visual_faiss" in line for line in lines)


def test_non_verifiable_spatial_question_is_v1_evidence_passthrough() -> None:
    demand = _demand("near")
    baseline_payload = _payload()
    fused = fuse_spatial_verified_visual_evidence(
        _graph_result(),
        baseline_payload,
        graph=_graph(),
        question="Where is the chair near the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=demand,
    )
    assert fused["visual_evidence"]["spatial_verification"]["applied"] is False
    assert fused["visual_evidence"]["visual_runtime_contract"] == "openeqa_visual_runtime_v2"


def test_input_payload_is_not_mutated() -> None:
    payload = _payload()
    before = deepcopy(payload)
    verify_spatial_pairs(payload, graph=_graph(), evidence_demand=_demand(), limit=8)
    assert payload == before
