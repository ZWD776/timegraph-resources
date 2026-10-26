from __future__ import annotations

from copy import deepcopy

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import (
    fuse_navqa_equivalent_visual_evidence,
)
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import (
    VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1,
    VISUAL_BACKBONE_OPENEQA_QR_V3,
    VISUAL_BUILDER_SPATIAL_GUARDED_V3,
    select_visual_builder,
)
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import verify_spatial_pairs
from vlkg.datasets.openeqa.runtime.visual_runtime_v3 import (
    OPENEQA_VISUAL_RUNTIME_CONTRACT,
    fuse_spatial_guarded_visual_evidence,
)


def _hit(
    faiss_id: int,
    label: str,
    bbox: list[float],
    *,
    entity_id: str = "",
    frame: int = 2,
) -> dict:
    return {
        "faiss_id": faiss_id,
        "observation_id": f"obs-{faiss_id}",
        "graph_frame_index": frame,
        "object_index": faiss_id,
        "label": label,
        "bbox": bbox,
        "similarity": 0.8,
        "canonical_entity_id": entity_id,
        "binding_kind": (
            "exact_frame_object_to_canonical_entity"
            if entity_id
            else "hsgg_observation_without_canonical_entity"
        ),
        "binding_confidence": 1.0 if entity_id else 0.45,
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


def _entity(entity_id: str, label: str) -> CanonicalEntityNode:
    return CanonicalEntityNode(
        node_id=f"entity:{entity_id}",
        video_id="episode-1",
        entity_id=entity_id,
        label=label,
        time_range=CanonicalTimeRange(start_frame=0, end_frame=10),
        observation_count=1,
    )


def _relation(
    subject_id: str,
    predicate: str,
    object_id: str,
    *,
    frame: int = 2,
) -> CanonicalEpisodeNode:
    episode_id = f"{subject_id}:{predicate}:{object_id}:{frame}"
    return CanonicalEpisodeNode(
        node_id=f"episode:{episode_id}",
        video_id="episode-1",
        episode_id=episode_id,
        episode_kind="RELATION",
        label=f"{subject_id} {predicate} {object_id}",
        time_range=CanonicalTimeRange(start_frame=frame, end_frame=frame),
        primary_entity_id=subject_id,
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        participant_entity_ids=[subject_id, object_id],
        predicate=f"RAW_RELATION_{predicate.upper()}",
        raw_predicate=predicate.replace("_", " "),
        source_ref_id=episode_id,
    )


def _directional_demand() -> dict:
    return {
        "primary_need": "spatial_localization",
        "route_id": "entity_relation_spatial",
        "answer_shape": "text",
        "target_terms": ["chair"],
        "anchor_terms": ["table"],
        "spatial_relations": ["left_of"],
        "spatial_constraints": [{"anchor": "table", "relation": "left_of"}],
        "open_target_slot": False,
        "attribute_dimensions": [],
        "state_terms": [],
        "constraint_terms": [],
        "query_intents": [],
    }


def _directional_payload(*, chair_left: bool) -> dict:
    chair_bbox = [0, 0, 10, 10] if chair_left else [40, 0, 50, 10]
    chair = _hit(1, "chair", chair_bbox, entity_id="chair-1")
    table = _hit(2, "table", [20, 0, 30, 10], entity_id="table-1")
    return {
        "groups": [
            {"query_intent": {"text": "chair", "role": "target"}, "hits": [chair]},
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [table],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [chair],
            },
        ]
    }


def test_v3_is_explicit_and_v1_remains_default() -> None:
    demand = _directional_demand()
    assert (
        select_visual_builder(VISUAL_BACKBONE_OPENEQA_QR_V3, evidence_demand=demand)
        == VISUAL_BUILDER_SPATIAL_GUARDED_V3
    )
    assert select_visual_builder(
        VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, evidence_demand=demand
    ) != VISUAL_BUILDER_SPATIAL_GUARDED_V3


def test_v3_no_pair_preserves_v1_answer_visible_evidence() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = _directional_payload(chair_left=False)
    demand = _directional_demand()
    baseline = fuse_navqa_equivalent_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=demand,
    )
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=demand,
    )
    assert fused["llm_ready_bundle"]["evidence_lines"] == baseline[
        "llm_ready_bundle"
    ]["evidence_lines"]
    assert fused["llm_ready_bundle"] == baseline["llm_ready_bundle"]
    assert fused["visual_evidence"]["candidate_packs"] == baseline[
        "visual_evidence"
    ]["candidate_packs"]
    verifier = fused["visual_evidence"]["spatial_verification"]
    assert verifier["fallback_to_v1_visual_evidence"] is True
    assert verifier["baseline_visual_evidence_preserved"] is True


def test_v3_inserts_pair_after_preserved_baseline_prefix() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = _directional_payload(chair_left=True)
    demand = _directional_demand()
    baseline = fuse_navqa_equivalent_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=demand,
    )
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is left of the table?",
        question_ir={"target_entities": ["chair"]},
        evidence_demand=demand,
    )
    old_lines = baseline["llm_ready_bundle"]["evidence_lines"]
    new_lines = fused["llm_ready_bundle"]["evidence_lines"]
    prefix_size = min(4, len(old_lines))
    assert new_lines[:prefix_size] == old_lines[:prefix_size]
    assert "kind=openeqa_spatial_geometry_candidate" in new_lines[prefix_size]
    assert all(line in new_lines for line in old_lines)
    assert fused["visual_evidence"]["visual_runtime_contract"] == (
        OPENEQA_VISUAL_RUNTIME_CONTRACT
    )


def test_v3_suppresses_answer_visible_pair_when_targets_are_ambiguous() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    table = _hit(1, "table", [40, 0, 50, 10], entity_id="table-1")
    chair = _hit(2, "chair", [0, 0, 10, 10], entity_id="chair-1")
    vase = _hit(3, "vase", [1, 0, 11, 10], entity_id="vase-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [table],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [chair, vase],
            },
        ]
    }
    demand = _directional_demand()
    demand["target_terms"] = []
    demand["open_target_slot"] = True
    baseline = fuse_navqa_equivalent_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        payload,
        graph=graph,
        question="What is left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    verifier = fused["visual_evidence"]["spatial_verification"]
    assert verifier["verified_pair_count"] == 2
    assert verifier["answer_visible_pair_count"] == 0
    assert verifier["fallback_to_v1_visual_evidence"] is True
    assert not any(
        "openeqa_spatial_geometry_candidate" in line
        for line in fused["llm_ready_bundle"]["evidence_lines"]
    )
    assert fused["llm_ready_bundle"] == baseline["llm_ready_bundle"]


def test_v3_promotes_component_target_to_unique_canonical_owner() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            _entity("armrest-1", "armrest"),
            _entity("chair-1", "chair"),
            _entity("table-1", "table"),
        ],
        episode_nodes=[_relation("chair-1", "supporting", "armrest-1")],
    )
    table = _hit(1, "table", [20, 0, 30, 10], entity_id="table-1")
    armrest = _hit(2, "armrest", [0, 0, 10, 10], entity_id="armrest-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [table],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [armrest],
            },
        ]
    }
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        payload,
        graph=graph,
        question="What is left of the table?",
        question_ir={},
        evidence_demand=_directional_demand(),
    )
    visual = fused["visual_evidence"]
    assert visual["raw_spatial_pairs"][0]["target_label"] == "armrest"
    assert visual["spatial_pairs"][0]["target_label"] == "chair"
    assert visual["spatial_pairs"][0]["original_target_label"] == "armrest"
    assert visual["stats"]["owner_promoted_spatial_pair_count"] == 1
    assert visual["spatial_verification"]["answer_visible_pair_count"] == 1
    assert any(
        "target='chair'" in line and "component_target='armrest'" in line
        for line in fused["llm_ready_bundle"]["evidence_lines"]
    )


def test_v3_owner_resolution_does_not_inflate_raw_geometry_margin() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            _entity("armrest-1", "armrest"),
            _entity("chair-1", "chair"),
            _entity("table-1", "table"),
        ],
        episode_nodes=[_relation("chair-1", "supporting", "armrest-1")],
    )
    table = _hit(1, "table", [40, 0, 50, 10], entity_id="table-1")
    chair = _hit(2, "chair", [0.1, 0, 10.1, 10], entity_id="chair-1")
    armrest = _hit(3, "armrest", [0, 0, 10, 10], entity_id="armrest-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "table",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [table],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [chair, armrest],
            },
        ]
    }
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        payload,
        graph=graph,
        question="What is left of the table?",
        question_ir={},
        evidence_demand=_directional_demand(),
    )
    visual = fused["visual_evidence"]
    assert len(visual["raw_spatial_pairs"]) == 2
    assert len(visual["spatial_pairs"]) == 1
    assert visual["spatial_verification"]["answer_visible_pair_count"] == 0


def test_v3_suppresses_component_whose_owner_is_a_between_anchor() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="episode-1",
        entity_nodes=[
            _entity("armrest-1", "armrest"),
            _entity("chair-1", "chair"),
            _entity("bed-1", "bed"),
        ],
        episode_nodes=[_relation("chair-1", "supporting", "armrest-1")],
    )
    chair = _hit(1, "chair", [0, 0, 10, 20], entity_id="chair-1")
    bed = _hit(2, "bed", [40, 0, 50, 20], entity_id="bed-1")
    armrest = _hit(3, "armrest", [20, 2, 30, 18], entity_id="armrest-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "bed and chair",
                    "role": "anchor",
                    "spatial_relation": "between",
                },
                "hits": [bed, chair],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [armrest],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["bed and chair"],
        "spatial_relations": ["between"],
        "spatial_constraints": [
            {"anchor": "bed and chair", "relation": "between"}
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    baseline = fuse_navqa_equivalent_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="What is between the bed and the chair?",
        question_ir={},
        evidence_demand=demand,
    )
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        payload,
        graph=graph,
        question="What is between the bed and the chair?",
        question_ir={},
        evidence_demand=demand,
    )
    visual = fused["visual_evidence"]
    assert visual["raw_spatial_pairs"][0]["target_label"] == "armrest"
    assert visual["spatial_pairs"] == []
    assert visual["stats"]["owner_anchor_collision_suppressed_count"] == 1
    assert visual["spatial_verification"]["answer_visible_pair_count"] == 0
    assert fused["llm_ready_bundle"] == baseline["llm_ready_bundle"]


def test_v3_does_not_apply_object_pair_to_yes_no_spatial_context() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = _directional_payload(chair_left=True)
    demand = _directional_demand()
    baseline = fuse_navqa_equivalent_visual_evidence(
        deepcopy(_graph_result()),
        deepcopy(payload),
        graph=graph,
        question="Is there an object to the left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    fused = fuse_spatial_guarded_visual_evidence(
        deepcopy(_graph_result()),
        payload,
        graph=graph,
        question="Is there an object to the left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    verifier = fused["visual_evidence"]["spatial_verification"]
    assert verifier["applied"] is False
    assert verifier["verified_pair_count"] == 0
    assert verifier["answer_visible_pair_count"] == 0
    assert fused["llm_ready_bundle"] == baseline["llm_ready_bundle"]


def test_v3_does_not_treat_which_instance_question_as_open_target() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = _directional_payload(chair_left=True)
    demand = _directional_demand()
    fused = fuse_spatial_guarded_visual_evidence(
        _graph_result(),
        payload,
        graph=graph,
        question="Which chair is to the left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    verifier = fused["visual_evidence"]["spatial_verification"]
    assert verifier["applied"] is False
    assert verifier["answer_visible_pair_count"] == 0


def test_directional_verifier_ignores_anchor_from_other_relation() -> None:
    light_switch = _hit(1, "light switch", [100, 0, 110, 10])
    hinge = _hit(2, "hinge", [0, 0, 10, 10])
    bed = _hit(3, "bed", [20, 0, 30, 10])
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "light switch",
                    "role": "anchor",
                    "spatial_relation": "right_of",
                },
                "hits": [light_switch],
            },
            {
                "query_intent": {
                    "text": "bathroom",
                    "role": "anchor",
                    "spatial_relation": "inside",
                },
                "hits": [hinge],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [bed],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["light switch", "bathroom"],
        "spatial_relations": ["right_of", "inside"],
        "spatial_constraints": [
            {"anchor": "light switch", "relation": "right_of"},
            {"anchor": "bathroom", "relation": "inside"},
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    pairs = verify_spatial_pairs(
        payload,
        graph=CanonicalTemporalMemoryGraph(video_id="episode-1"),
        evidence_demand=demand,
    )
    assert pairs == []


def test_directional_verifier_rejects_same_canonical_entity_self_pair() -> None:
    anchor = _hit(1, "trash can", [20, 0, 30, 10], entity_id="bin-1")
    duplicate = _hit(2, "trash can", [0, 0, 10, 10], entity_id="bin-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "trash can",
                    "role": "anchor",
                    "spatial_relation": "left_of",
                },
                "hits": [anchor],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [duplicate],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["trash can"],
        "spatial_relations": ["left_of"],
        "spatial_constraints": [
            {"anchor": "trash can", "relation": "left_of"}
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    pairs = verify_spatial_pairs(
        payload,
        graph=CanonicalTemporalMemoryGraph(video_id="episode-1"),
        evidence_demand=demand,
    )
    assert pairs == []


def test_v3_suppresses_unverified_multi_direction_conjunction() -> None:
    graph = CanonicalTemporalMemoryGraph(video_id="episode-1")
    payload = _directional_payload(chair_left=True)
    demand = _directional_demand()
    demand["spatial_relations"] = ["left_of", "right_of"]
    demand["spatial_constraints"] = [
        {"anchor": "table", "relation": "left_of"},
        {"anchor": "plant", "relation": "right_of"},
    ]
    fused = fuse_spatial_guarded_visual_evidence(
        _graph_result(),
        payload,
        graph=graph,
        question="What is right of the plant and left of the table?",
        question_ir={},
        evidence_demand=demand,
    )
    verifier = fused["visual_evidence"]["spatial_verification"]
    assert verifier["applied"] is False
    assert verifier["answer_visible_pair_count"] == 0


def test_between_keeps_distinct_observations_despite_shared_entity_binding() -> None:
    first = _hit(1, "picture frame", [0, 0, 10, 20], entity_id="frame-merged")
    second = _hit(2, "picture frame", [40, 0, 50, 20], entity_id="frame-merged")
    door_frame = _hit(3, "door frame", [60, 0, 70, 20], entity_id="door-frame")
    television = _hit(4, "television", [20, 2, 30, 18], entity_id="tv-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "two picture frames",
                    "role": "anchor",
                    "spatial_relation": "between",
                },
                "hits": [first],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [television, second, door_frame],
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
    )
    assert any(pair["target_label"] == "television" for pair in pairs)
    tv_pair = next(pair for pair in pairs if pair["target_label"] == "television")
    assert tv_pair["anchor_label"] == "picture frame + picture frame"
    assert "door frame" not in tv_pair["anchor_label"]


def test_between_requires_both_named_anchor_classes() -> None:
    sofa = _hit(1, "sofa", [0, 0, 10, 20], entity_id="sofa-1")
    chair = _hit(2, "chair", [40, 0, 50, 20], entity_id="chair-1")
    other_chair = _hit(3, "chair", [60, 0, 70, 20], entity_id="chair-2")
    table = _hit(4, "table", [20, 2, 30, 18], entity_id="table-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "sofa and chair",
                    "role": "anchor",
                    "spatial_relation": "between",
                },
                "hits": [sofa, chair, other_chair],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [table],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["sofa and chair"],
        "spatial_relations": ["between"],
        "spatial_constraints": [
            {"anchor": "sofa and chair", "relation": "between"}
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    pairs = verify_spatial_pairs(
        payload,
        graph=CanonicalTemporalMemoryGraph(video_id="episode-1"),
        evidence_demand=demand,
    )
    assert pairs
    assert all("sofa" in pair["anchor_label"] for pair in pairs)
    assert all("chair + chair" not in pair["anchor_label"] for pair in pairs)


def test_between_rejects_target_contained_inside_anchor() -> None:
    first = _hit(1, "armchair", [0, 0, 30, 30], entity_id="chair-1")
    second = _hit(2, "armchair", [50, 0, 80, 30], entity_id="chair-2")
    cushion = _hit(3, "cushion", [5, 5, 20, 20], entity_id="cushion-1")
    payload = {
        "groups": [
            {
                "query_intent": {
                    "text": "two armchairs",
                    "role": "anchor",
                    "spatial_relation": "between",
                },
                "hits": [first, second],
            },
            {
                "query_intent": {"text": "co-visible objects", "role": "spatial_target"},
                "hits": [cushion],
            },
        ]
    }
    demand = {
        "primary_need": "spatial_localization",
        "anchor_terms": ["two armchairs"],
        "spatial_relations": ["between"],
        "spatial_constraints": [
            {"anchor": "two armchairs", "relation": "between"}
        ],
        "open_target_slot": True,
        "target_terms": [],
    }
    pairs = verify_spatial_pairs(
        payload,
        graph=CanonicalTemporalMemoryGraph(video_id="episode-1"),
        evidence_demand=demand,
    )
    assert pairs == []
