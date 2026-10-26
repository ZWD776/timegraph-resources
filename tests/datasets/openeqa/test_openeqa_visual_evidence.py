from __future__ import annotations

import json

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.config.config import GraphQAConfig
from vlkg.datasets.openeqa.qa.graph_qa_answering import (
    OpenAICompatibleGraphQAResponder,
)
from vlkg.datasets.openeqa.retrieval.visual_evidence import (
    VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT,
    VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT,
    VISUAL_EVIDENCE_GUARDED_V4_CONTRACT,
    VISUAL_OVERRIDE_GUARDED_V4C_CONTRACT,
    VISUAL_OVERRIDE_GUARDED_V4D_CONTRACT,
    VISUAL_OVERRIDE_GUARDED_V4_CONTRACT,
    VISUAL_EVIDENCE_V2_CONTRACT,
    fuse_graph_visual_evidence,
    fuse_graph_visual_evidence_guarded_v4,
    fuse_graph_visual_evidence_guarded_v4c,
    fuse_graph_visual_evidence_guarded_v4d,
    fuse_graph_visual_evidence_v2,
)


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
            "uncertainty_notes": [],
            "llm_context_text": "Graph context",
            "structured_graph_context": {},
        },
    }


def test_visual_attribute_pack_keeps_owner_binding() -> None:
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "query_intent_count": 1,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {"text": "chair color", "role": "target"},
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
            }
        ],
    }
    demand = {
        "primary_need": "entity_attribute",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "attribute_dimensions": ["color"],
        "requires_negative_coverage": False,
    }

    fused = fuse_graph_visual_evidence(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    evidence = fused["visual_evidence"]
    pack = evidence["candidate_packs"][0]
    assert pack["canonical_entity_id"] == "chair-1"
    assert "blue fabric chair" in pack["raw_visual_attributes"]
    assert any("owner_bound_visual_attributes" in line for line in evidence["evidence_lines"])
    assert fused["llm_ready_bundle"]["evidence_lines"] == [
        "kind=graph label='chair'"
    ]
    assert fused["llm_ready_bundle"]["visual_entity_candidate_pack_lines"][0].startswith(
        "kind=graph_bound_visual_candidate"
    )
    assert fused["llm_ready_bundle"]["visual_faiss_evidence_lines"] == []


def test_unbound_visual_hit_is_not_promoted_to_graph_fact() -> None:
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "query_intent_count": 1,
        "episode_candidate_count": 1,
        "groups": [
            {
                "query_intent": {"text": "lamp", "role": "target"},
                "hits": [
                    {
                        "faiss_id": 7,
                        "graph_frame_index": 3,
                        "label": "lamp",
                        "bbox": [0, 0, 4, 4],
                        "similarity": 0.7,
                        "observation_id": "obs-lamp",
                        "canonical_entity_id": "",
                        "binding_kind": "hsgg_observation_without_canonical_entity",
                        "binding_confidence": 0.45,
                    }
                ],
            }
        ],
    }
    demand = {
        "primary_need": "existence_or_state",
        "target_terms": ["lamp"],
        "anchor_terms": [],
        "attribute_dimensions": [],
        "requires_negative_coverage": True,
    }

    fused = fuse_graph_visual_evidence(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    evidence = fused["visual_evidence"]
    assert evidence["candidate_packs"][0]["graph_bound"] is False
    assert evidence["candidate_packs"][0]["unbound_visual_observation_is_graph_fact"] is False
    assert any("lack a canonical entity binding" in note for note in fused["llm_ready_bundle"]["uncertainty_notes"])


def test_spatial_route_builds_same_frame_target_anchor_pair() -> None:
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "query_intent_count": 2,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {"text": "chair", "role": "target"},
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 2,
                        "label": "chair",
                        "bbox": [0, 0, 10, 10],
                        "similarity": 0.8,
                        "canonical_entity_id": "chair-1",
                        "binding_confidence": 1.0,
                    }
                ],
            },
            {
                "query_intent": {"text": "table", "role": "anchor"},
                "hits": [
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 2,
                        "label": "table",
                        "bbox": [20, 0, 30, 10],
                        "similarity": 0.75,
                        "canonical_entity_id": "table-1",
                        "binding_confidence": 1.0,
                    }
                ],
            },
        ],
    }
    demand = {
        "primary_need": "spatial_localization",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "attribute_dimensions": [],
        "requires_negative_coverage": False,
    }

    fused = fuse_graph_visual_evidence(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    pair = fused["visual_evidence"]["spatial_pairs"][0]
    assert pair["relation"] == "left_of"
    assert pair["target_entity_id"] == "chair-1"
    assert pair["anchor_entity_id"] == "table-1"


def test_spatial_route_rejects_same_entity_self_pair() -> None:
    shared = {
        "faiss_id": 1,
        "observation_id": "obs-chair",
        "graph_frame_index": 2,
        "label": "chair",
        "bbox": [0, 0, 10, 10],
        "similarity": 0.8,
        "canonical_entity_id": "chair-1",
        "binding_confidence": 1.0,
    }
    payload = {
        "episode_history": "episode-1",
        "groups": [
            {"query_intent": {"text": "left chair", "role": "target"}, "hits": [shared]},
            {"query_intent": {"text": "chair", "role": "anchor"}, "hits": [shared]},
        ],
    }
    demand = {
        "primary_need": "spatial_localization",
        "target_terms": [],
        "anchor_terms": ["chair"],
        "spatial_relations": ["left_of"],
        "open_target_slot": True,
    }

    fused = fuse_graph_visual_evidence(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    assert fused["visual_evidence"]["spatial_pairs"] == []
    assert fused["visual_evidence"]["evidence_lines"] == []


def test_v2_keeps_unverified_coframe_candidate_without_claiming_relation() -> None:
    payload = {
        "episode_history": "episode-1",
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {"text": "chair", "role": "anchor"},
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 2,
                        "label": "chair",
                        "bbox": [0, 0, 10, 10],
                        "similarity": 0.8,
                        "canonical_entity_id": "chair-1",
                    }
                ],
            },
            {
                "query_intent": {"text": "co-visible", "role": "spatial_target"},
                "hits": [
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 2,
                        "label": "table",
                        "bbox": [20, 0, 30, 10],
                        "similarity": 0.8,
                        "canonical_entity_id": "table-1",
                    }
                ],
            },
        ],
    }
    demand = {
        "primary_need": "spatial_localization",
        "target_terms": [],
        "anchor_terms": ["chair"],
        "spatial_relations": ["behind"],
        "open_target_slot": True,
    }

    fused = fuse_graph_visual_evidence_v2(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    evidence = fused["visual_evidence"]
    assert evidence["contract"] == VISUAL_EVIDENCE_V2_CONTRACT
    assert evidence["spatial_pairs"] == []
    assert any(
        "kind=visual_coframe_candidate" in line and "relation_verified=false" in line
        for line in evidence["evidence_lines"]
    )


def test_v2_deduplicates_candidate_visual_anchors_and_adds_coverage() -> None:
    hit = {
        "faiss_id": 1,
        "observation_id": "obs-chair",
        "graph_frame_index": 2,
        "label": "chair",
        "bbox": [0, 0, 10, 10],
        "similarity": 0.8,
        "canonical_entity_id": "chair-1",
    }
    payload = {
        "episode_history": "episode-1",
        "episode_candidate_count": 2,
        "groups": [
            {"query_intent": {"text": "chair", "role": "target"}, "hits": [hit]},
            {
                "query_intent": {"text": "chair open", "role": "target"},
                "hits": [hit],
            },
        ],
    }
    demand = {
        "primary_need": "existence_or_state",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "requires_negative_coverage": True,
    }

    fused = fuse_graph_visual_evidence_v2(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    evidence = fused["visual_evidence"]
    assert len(evidence["candidate_packs"][0]["visual_anchors"]) == 1
    assert evidence["stats"]["duplicate_candidate_anchor_count_removed"] == 1
    assert any(
        "kind=visual_episode_coverage" in line and "absence_proven=false" in line
        for line in evidence["evidence_lines"]
    )


def _bound_pair_payload(*, chair_bbox: list[int], table_bbox: list[int]) -> dict:
    return {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {"text": "chair", "role": "spatial_target"},
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 2,
                        "label": "chair",
                        "bbox": chair_bbox,
                        "similarity": 0.8,
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
                },
                "hits": [
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 2,
                        "label": "table",
                        "bbox": table_bbox,
                        "similarity": 0.8,
                        "canonical_entity_id": "table-1",
                        "binding_kind": "exact_frame_object_to_canonical_entity",
                        "binding_confidence": 1.0,
                    }
                ],
            },
        ],
    }


def _spatial_demand(relation: str) -> dict:
    return {
        "primary_need": "spatial_localization",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "spatial_relations": [relation],
        "open_target_slot": False,
        "attribute_dimensions": [],
        "requires_negative_coverage": False,
    }


def test_v4_rejects_bbox_pair_without_same_direction_canonical_relation() -> None:
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        _bound_pair_payload(chair_bbox=[20, 0, 30, 10], table_bbox=[0, 0, 10, 10]),
        graph=_graph(),
        evidence_demand=_spatial_demand("right_of"),
    )

    evidence = fused["visual_evidence"]
    assert evidence["contract"] == VISUAL_EVIDENCE_GUARDED_V4_CONTRACT
    assert evidence["spatial_pairs"][0]["proof"]["relation_verified"] is False
    assert evidence["verified_spatial_pairs"] == []
    assert evidence["evidence_lines"] == []
    assert fused["llm_ready_bundle"]["evidence_lines"] == [
        "kind=graph label='chair'"
    ]


def test_v4_accepts_graph_bound_pair_with_same_direction_canonical_relation() -> None:
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        _bound_pair_payload(chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]),
        graph=_graph(),
        evidence_demand=_spatial_demand("left_of"),
    )

    evidence = fused["visual_evidence"]
    pair = evidence["verified_spatial_pairs"][0]
    assert pair["proof"]["target_bound"] is True
    assert pair["proof"]["anchor_bound"] is True
    assert pair["proof"]["relation_verified"] is True
    assert pair["proof"]["can_override_graph"] is True
    assert any("relation_verified=canonical_direct_relation" in line for line in evidence["evidence_lines"])


def test_v4_crop_presence_cannot_override_requested_state_without_state_fact() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    demand = {
        "primary_need": "existence_or_state",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "state_terms": ["open"],
        "attribute_dimensions": [],
        "requires_negative_coverage": True,
    }

    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    pack = fused["visual_evidence"]["candidate_packs"][0]
    assert pack["proof"]["owner_bound"] is True
    assert pack["proof"]["state_bearing"] is False
    assert pack["proof"]["can_override_graph"] is False
    assert all("owner_bound_states=" not in line for line in fused["visual_evidence"]["evidence_lines"])
    assert fused["visual_evidence"]["evidence_lines"] == []
    assert fused["visual_evidence"]["visual_override_contract"]["absence_proven"] is False


def test_v4_attribute_requires_matching_owner_component_and_dimension() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    demand = {
        "primary_need": "entity_attribute",
        "target_terms": ["handle", "chair"],
        "anchor_terms": [],
        "attribute_dimensions": ["material"],
        "requires_negative_coverage": False,
    }

    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    proof = fused["visual_evidence"]["candidate_packs"][0]["proof"]
    assert proof["target_bound"] is False
    assert proof["owner_bound"] is False
    assert proof["attribute_dimension"]["matched"] is False
    assert proof["can_override_graph"] is False
    assert fused["visual_evidence"]["evidence_lines"] == []


def test_v4_keeps_functional_identity_and_count_candidate_set() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    functional = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "functional_world",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "attribute_dimensions": [],
            "allows_common_world_knowledge": True,
            "answer_shape": "text",
        },
    )
    counted = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "count_set",
            "target_terms": ["chairs"],
            "anchor_terms": [],
            "attribute_dimensions": [],
        },
    )

    assert any("graph_bound_object_identity_for_common_world_reasoning" in line for line in functional["visual_evidence"]["evidence_lines"])
    assert any("unique_graph_bound_target_candidates=1" in line for line in counted["visual_evidence"]["evidence_lines"])


def test_v4_binary_functional_question_does_not_use_identity_as_proof() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "functional_world",
            "answer_shape": "binary",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "attribute_dimensions": [],
            "allows_common_world_knowledge": True,
        },
    )

    assert fused["visual_evidence"]["candidate_packs"][0]["proof"]["can_override_graph"] is False
    assert fused["visual_evidence"]["evidence_lines"] == []


def test_v4c_binary_functional_requires_exact_graph_bound_target() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    exact = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "functional_world",
            "answer_shape": "binary",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "state_terms": [],
            "attribute_dimensions": [],
            "allows_common_world_knowledge": True,
        },
    )
    unrelated = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "functional_world",
            "answer_shape": "binary",
            "target_terms": ["poster"],
            "anchor_terms": [],
            "state_terms": [],
            "attribute_dimensions": [],
            "allows_common_world_knowledge": True,
        },
    )

    assert any(
        "kind=guarded_functional_identity_candidate" in line
        for line in exact["visual_evidence"]["evidence_lines"]
    )
    assert unrelated["visual_evidence"]["evidence_lines"] == []


def test_v4c_keeps_state_crop_guard_unchanged() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    graph = _graph().model_copy(
        update={
            "episode_nodes": [
                episode
                for episode in _graph().episode_nodes
                if episode.episode_kind != "STATE"
            ]
        }
    )
    fused = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        payload,
        graph=graph,
        evidence_demand={
            "primary_need": "existence_or_state",
            "answer_shape": "binary",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "state_terms": ["open"],
            "attribute_dimensions": [],
            "requires_negative_coverage": True,
        },
    )

    assert fused["visual_evidence"]["evidence_lines"] == []
    assert fused["visual_evidence"]["candidate_packs"][0]["proof"][
        "state_bearing"
    ] is False


def test_v4_simple_object_identity_allows_top_semantic_graph_bound_candidate() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "object_identity",
            "answer_shape": "text",
            "target_terms": ["furniture for sitting"],
            "anchor_terms": [],
            "spatial_relations": [],
            "attribute_dimensions": [],
        },
    )

    proof = fused["visual_evidence"]["candidate_packs"][0]["proof"]
    assert proof["target_bound"] is True
    assert proof["can_override_graph"] is True
    assert any("observed_graph_bound_object_identity" in line for line in fused["visual_evidence"]["evidence_lines"])


def test_v4_owner_bound_localization_without_anchor_does_not_require_pair() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "spatial_localization",
            "answer_shape": "location_text",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "spatial_relations": ["where"],
            "attribute_dimensions": [],
        },
    )

    assert fused["visual_evidence"]["candidate_packs"][0]["proof"]["can_override_graph"] is True
    assert fused["visual_evidence"]["evidence_lines"]


def test_v4_raw_attribute_requires_one_unconflicted_dimension_value() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    graph = _graph()
    entity_nodes = list(graph.entity_nodes)
    entity_nodes[0] = entity_nodes[0].model_copy(
        update={"attributes": {"raw_visual_attributes": ["white smoke detector"]}}
    )
    graph = graph.model_copy(
        update={
            "entity_nodes": entity_nodes,
            "episode_nodes": [
                episode
                for episode in graph.episode_nodes
                if episode.episode_kind != "STATE"
            ],
        }
    )
    fused = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(),
        payload,
        graph=graph,
        evidence_demand={
            "primary_need": "entity_attribute",
            "answer_shape": "text",
            "target_terms": ["chair"],
            "anchor_terms": [],
            "attribute_dimensions": ["color"],
        },
    )

    dimension = fused["visual_evidence"]["candidate_packs"][0]["proof"]["attribute_dimension"]
    assert dimension["values"] == ["white"]
    assert dimension["conflicting_values"] is False
    assert fused["visual_evidence"]["evidence_lines"]


def test_v4c_identity_candidate_does_not_prove_unverified_relation() -> None:
    fused = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        _bound_pair_payload(
            chair_bbox=[20, 0, 30, 10],
            table_bbox=[0, 0, 10, 10],
        ),
        graph=_graph(),
        evidence_demand={
            "primary_need": "object_identity",
            "answer_shape": "text",
            "target_terms": [],
            "anchor_terms": ["table"],
            "spatial_relations": ["right_of"],
            "open_target_slot": True,
            "attribute_dimensions": [],
        },
    )

    evidence = fused["visual_evidence"]
    chair = next(
        pack for pack in evidence["candidate_packs"] if pack["visual_label"] == "chair"
    )
    assert evidence["contract"] == VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    assert chair["proof"]["relation_verified"] is False
    assert chair["proof"]["can_override_graph"] is False
    assert chair["proof"]["identity_observation"]["can_support_object_identity"] is True
    assert any(
        "can_support_object_identity=true" in line
        and "can_prove_spatial_relation=false" in line
        for line in evidence["evidence_lines"]
    )
    assert (
        evidence["visual_override_contract"]["contract"]
        == VISUAL_OVERRIDE_GUARDED_V4C_CONTRACT
    )


def test_v4c_repeated_unbound_identity_is_non_graph_observation() -> None:
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {"text": "lamp", "role": "target"},
                "hits": [
                    {
                        "faiss_id": 11,
                        "graph_frame_index": 1,
                        "label": "lamp",
                        "similarity": 0.8,
                        "observation_id": "lamp-1",
                        "canonical_entity_id": "",
                    },
                    {
                        "faiss_id": 12,
                        "graph_frame_index": 2,
                        "label": "lamp",
                        "similarity": 0.79,
                        "observation_id": "lamp-2",
                        "canonical_entity_id": "",
                    },
                ],
            }
        ],
    }
    fused = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "object_identity",
            "answer_shape": "text",
            "target_terms": ["lamp"],
            "anchor_terms": [],
            "spatial_relations": [],
            "attribute_dimensions": [],
        },
    )

    identity = fused["visual_evidence"]["identity_support_candidates"]
    assert len(identity) == 2
    assert all(not pack["graph_bound"] for pack in identity)
    assert all(
        pack["proof"]["identity_observation"]["visual_observation_is_graph_fact"]
        is False
        for pack in identity
    )
    assert all(
        "visual_observation_is_graph_fact=false" in line
        for line in fused["visual_evidence"]["evidence_lines"]
    )


def test_v4c_no_anchor_localization_exposes_direct_canonical_relation() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10],
        table_bbox=[20, 0, 30, 10],
    )
    payload["groups"] = payload["groups"][:1]
    fused = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(),
        payload,
        graph=_graph(),
        evidence_demand={
            "primary_need": "spatial_localization",
            "answer_shape": "location_text",
            "target_terms": ["blue chair"],
            "anchor_terms": [],
            "spatial_relations": [],
            "attribute_dimensions": ["color"],
        },
    )

    lines = fused["visual_evidence"]["evidence_lines"]
    assert any("kind=guarded_graph_bound_location_candidate" in line for line in lines)
    assert any("chair left_of table" in line for line in lines)


def test_v4c_resolves_stable_dominant_attribute_but_keeps_v4_isolated() -> None:
    graph = _graph()
    entities = list(graph.entity_nodes)
    entities[0] = entities[0].model_copy(
        update={
            "attributes": {
                "raw_visual_attributes": [
                    "red shirt on hanger",
                    "red cotton shirt",
                    "white shirt on person",
                    "red shirt",
                ]
            }
        }
    )
    graph = graph.model_copy(
        update={
            "entity_nodes": entities,
            "episode_nodes": [
                episode
                for episode in graph.episode_nodes
                if episode.episode_kind != "STATE"
            ],
        }
    )
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10],
        table_bbox=[20, 0, 30, 10],
    )
    payload["groups"] = payload["groups"][:1]
    demand = {
        "primary_need": "entity_attribute",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": [],
        "attribute_dimensions": ["color"],
    }

    v4 = fuse_graph_visual_evidence_guarded_v4(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )
    v4c = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )

    assert v4["visual_evidence"]["evidence_lines"] == []
    dimension = v4c["visual_evidence"]["candidate_packs"][0]["proof"][
        "attribute_dimension"
    ]
    assert dimension["answer_values"] == ["red"]
    assert any("owner_bound_color=red" in line for line in v4c["visual_evidence"]["evidence_lines"])


def test_v4d_promotes_complete_object_over_unrequested_components() -> None:
    graph_payload = _graph().model_dump(mode="json")
    graph_payload["entity_nodes"][0]["attributes"] = {
        "raw_visual_attributes": [
            "gray armchair in the corner",
            "gray armchair with rounded arms",
        ]
    }
    graph_payload["entity_nodes"].append(
        {
            "node_id": "node-armrest",
            "video_id": "episode-1",
            "entity_id": "armrest-1",
            "label": "armrest",
            "time_range": {"start_frame": 0, "end_frame": 4},
            "observation_count": 3,
            "attributes": {
                "raw_visual_attributes": ["gray fabric armrest of a chair"]
            },
        }
    )
    graph = CanonicalTemporalMemoryGraph.model_validate(graph_payload)
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {
                    "text": "grey object in corner",
                    "role": "spatial_target",
                    "source": "question_bound_target",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 1,
                        "label": "armrest",
                        "similarity": 0.95,
                        "canonical_entity_id": "armrest-1",
                        "binding_confidence": 1.0,
                    },
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 2,
                        "label": "chair",
                        "similarity": 0.75,
                        "canonical_entity_id": "chair-1",
                        "binding_confidence": 1.0,
                    },
                ],
            }
        ],
    }
    demand = {
        "primary_need": "object_identity",
        "answer_shape": "text",
        "target_terms": [],
        "anchor_terms": ["corner"],
        "spatial_relations": ["inside"],
        "open_target_slot": True,
        "attribute_dimensions": ["type"],
        "state_terms": ["grey"],
        "constraint_terms": ["grey", "in the corner"],
    }

    v4c = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )
    v4d = fuse_graph_visual_evidence_guarded_v4d(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )

    assert "label='armrest'" in v4c["visual_evidence"]["evidence_lines"][0]
    assert "label='chair'" in v4d["visual_evidence"]["evidence_lines"][0]
    assert "complete_object_preferred=true" in v4d["visual_evidence"]["evidence_lines"][0]
    assert v4d["visual_evidence"]["contract"] == VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT
    assert (
        v4d["visual_evidence"]["visual_override_contract"]["contract"]
        == VISUAL_OVERRIDE_GUARDED_V4D_CONTRACT
    )


def test_v4d_does_not_use_spatial_anchor_words_to_reorder_open_identity() -> None:
    graph_payload = _graph().model_dump(mode="json")
    graph_payload["entity_nodes"].extend(
        [
            {
                "node_id": "node-calculator",
                "video_id": "episode-1",
                "entity_id": "calculator-1",
                "label": "calculator",
                "time_range": {"start_frame": 0, "end_frame": 4},
                "observation_count": 2,
            },
            {
                "node_id": "node-blinds",
                "video_id": "episode-1",
                "entity_id": "blinds-1",
                "label": "blinds",
                "time_range": {"start_frame": 0, "end_frame": 4},
                "observation_count": 2,
                "attributes": {"raw_visual_attributes": ["blinds covering window"]},
            },
        ]
    )
    graph = CanonicalTemporalMemoryGraph.model_validate(graph_payload)
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "groups": [
            {
                "query_intent": {
                    "text": "object on desk leaning against wall with window",
                    "role": "spatial_target",
                    "source": "question_bound_target",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 1,
                        "label": "calculator",
                        "similarity": 0.9,
                        "canonical_entity_id": "calculator-1",
                        "binding_confidence": 1.0,
                    },
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 1,
                        "label": "blinds",
                        "similarity": 0.8,
                        "canonical_entity_id": "blinds-1",
                        "binding_confidence": 1.0,
                    },
                ],
            }
        ],
    }
    demand = {
        "primary_need": "object_identity",
        "answer_shape": "text",
        "target_terms": [],
        "anchor_terms": ["desk", "wall", "window"],
        "spatial_relations": ["on"],
        "open_target_slot": True,
        "state_terms": [],
        "constraint_terms": ["on desk", "against wall", "with window"],
    }

    v4d = fuse_graph_visual_evidence_guarded_v4d(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )

    assert "label='calculator'" in v4d["visual_evidence"]["evidence_lines"][0]
    assert v4d["visual_evidence"]["contract"] == VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT


def test_v4d_falls_back_to_v4c_for_missing_functional_question() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"] = payload["groups"][:1]
    payload["groups"][0]["query_intent"]["source"] = "target_attribute_or_state"
    demand = {
        "primary_need": "functional_world",
        "answer_shape": "text",
        "target_terms": ["objects"],
        "anchor_terms": [],
        "spatial_relations": [],
        "state_terms": ["missing for sleeping"],
        "constraint_terms": ["objects missing to sleep"],
        "query_intents": [
            {
                "text": "objects missing for sleeping",
                "source": "target_attribute_or_state",
            }
        ],
    }

    v4c = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )
    v4d = fuse_graph_visual_evidence_guarded_v4d(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    assert v4c["visual_evidence"]["evidence_lines"]
    assert (
        v4d["visual_evidence"]["evidence_lines"]
        == v4c["visual_evidence"]["evidence_lines"]
    )
    assert (
        v4d["visual_evidence"]["contract"]
        == VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    )


def test_v4d_falls_back_to_v4c_for_capacity_functional_question() -> None:
    payload = _bound_pair_payload(
        chair_bbox=[0, 0, 10, 10], table_bbox=[20, 0, 30, 10]
    )
    payload["groups"][0]["query_intent"]["source"] = "question_bound_target"
    demand = {
        "primary_need": "functional_world",
        "answer_shape": "text",
        "target_terms": ["items"],
        "anchor_terms": ["table"],
        "spatial_relations": ["between"],
        "state_terms": ["store items"],
        "constraint_terms": ["store items between tables"],
        "query_intents": [
            {"text": "store items between tables", "source": "question_bound_target"}
        ],
    }

    v4c = fuse_graph_visual_evidence_guarded_v4c(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )
    v4d = fuse_graph_visual_evidence_guarded_v4d(
        _graph_result(), payload, graph=_graph(), evidence_demand=demand
    )

    assert any("guarded_functional_identity_candidate" in line for line in v4c["visual_evidence"]["evidence_lines"])
    assert (
        v4d["visual_evidence"]["evidence_lines"]
        == v4c["visual_evidence"]["evidence_lines"]
    )
    assert (
        v4d["visual_evidence"]["contract"]
        == VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    )


def test_v4d_preserves_v4c_functional_candidate_order() -> None:
    graph_payload = _graph().model_dump(mode="json")
    graph_payload["entity_nodes"].extend(
        [
            {
                "node_id": "node-cabinet",
                "video_id": "episode-1",
                "entity_id": "cabinet-1",
                "label": "cabinet",
                "time_range": {"start_frame": 0, "end_frame": 4},
                "observation_count": 2,
            },
            {
                "node_id": "node-burner",
                "video_id": "episode-1",
                "entity_id": "burner-1",
                "label": "burner",
                "time_range": {"start_frame": 0, "end_frame": 4},
                "observation_count": 2,
            },
        ]
    )
    graph = CanonicalTemporalMemoryGraph.model_validate(graph_payload)
    payload = {
        "episode_history": "episode-1",
        "episode_scoped_search": True,
        "episode_candidate_count": 2,
        "groups": [
            {
                "query_intent": {
                    "text": "kitchen",
                    "role": "target",
                    "source": "target_identity",
                },
                "hits": [
                    {
                        "faiss_id": 1,
                        "graph_frame_index": 1,
                        "label": "cabinet",
                        "similarity": 0.95,
                        "canonical_entity_id": "cabinet-1",
                        "binding_confidence": 1.0,
                    }
                ],
            },
            {
                "query_intent": {
                    "text": "kitchen warm up burrito",
                    "role": "target",
                    "source": "target_attribute_or_state",
                },
                "hits": [
                    {
                        "faiss_id": 2,
                        "graph_frame_index": 2,
                        "label": "burner",
                        "similarity": 0.70,
                        "canonical_entity_id": "burner-1",
                        "binding_confidence": 1.0,
                    }
                ],
            },
        ],
    }
    demand = {
        "primary_need": "functional_world",
        "answer_shape": "text",
        "target_terms": ["kitchen"],
        "anchor_terms": [],
        "spatial_relations": ["inside"],
        "state_terms": ["warm up burrito"],
        "constraint_terms": [],
        "query_intents": [group["query_intent"] for group in payload["groups"]],
    }

    v4d = fuse_graph_visual_evidence_guarded_v4d(
        _graph_result(), payload, graph=graph, evidence_demand=demand
    )

    assert "label='cabinet'" in v4d["visual_evidence"]["evidence_lines"][0]
    assert (
        v4d["visual_evidence"]["contract"]
        == VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    )


def test_v4_prompt_guard_is_contract_gated() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    base_bundle = {
        "answer_brief": "graph answer",
        "llm_context_text": "Video: toy",
        "evidence_lines": ["kind=graph label='chair'"],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "structured_graph_context": {},
        "visual_evidence_contract": VISUAL_EVIDENCE_V2_CONTRACT,
        "visual_entity_candidate_pack_lines": ["kind=legacy_visual_candidate"],
    }
    graph_result = {"matches": [], "retrieval_bundle": {}}
    legacy_messages = responder._build_messages(
        question="What color is the chair?",
        llm_ready_bundle=base_bundle,
        graph_result=graph_result,
    )
    guarded_bundle = dict(base_bundle)
    guarded_bundle["visual_evidence_contract"] = VISUAL_EVIDENCE_GUARDED_V4_CONTRACT
    guarded_bundle["visual_override_contract"] = {
        "contract": VISUAL_OVERRIDE_GUARDED_V4_CONTRACT,
        "default_authority": "canonical_graph_evidence",
        "can_override_graph": False,
    }
    guarded_messages = responder._build_messages(
        question="What color is the chair?",
        llm_ready_bundle=guarded_bundle,
        graph_result=graph_result,
    )

    legacy_payload = json.loads(legacy_messages[1]["content"])
    guarded_payload = json.loads(guarded_messages[1]["content"])
    assert "graph_evidence_is_default_authority" not in legacy_payload["instructions"]
    assert guarded_payload["instructions"]["graph_evidence_is_default_authority"] is True
    assert guarded_payload["visual_override_contract"]["can_override_graph"] is False


def test_v4_without_safe_visual_lines_uses_graph_only_prompt_policy() -> None:
    responder = OpenAICompatibleGraphQAResponder(GraphQAConfig())
    bundle = {
        "answer_brief": "graph answer",
        "llm_context_text": "Video: toy",
        "evidence_lines": ["kind=graph label='chair'"],
        "focus_frame_lines": [],
        "scene_lines": [],
        "temporal_chain_lines": [],
        "evidence_aggregation_lines": [],
        "evidence_aggregation": {"candidate_facts": []},
        "structured_graph_context": {},
        "visual_evidence_contract": VISUAL_EVIDENCE_GUARDED_V4_CONTRACT,
        "visual_override_contract": {
            "contract": VISUAL_OVERRIDE_GUARDED_V4_CONTRACT,
            "can_override_graph": False,
            "answerable_visual_line_count": 0,
        },
        "visual_entity_candidate_pack_lines": [],
    }
    messages = responder._build_messages(
        question="What is under the television?",
        llm_ready_bundle=bundle,
        graph_result={"matches": [], "retrieval_bundle": {}},
    )

    payload = json.loads(messages[1]["content"])
    assert payload["instructions"]["use_only_graph_evidence"] is True
    assert "visual_override_contract" not in payload
    assert "graph_evidence_is_default_authority" not in payload["instructions"]
