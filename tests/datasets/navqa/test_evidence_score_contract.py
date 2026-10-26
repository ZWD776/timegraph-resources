from __future__ import annotations

from typing import Any

from vlkg.datasets.navqa.retrieval.evidence_score_contract import apply_candidate_pack_score_contract


def _candidate_pack(
    entity_id: str,
    *,
    start_ts: float,
    end_ts: float,
    retrieval_similarity: float,
    attribute: str,
) -> dict[str, Any]:
    return {
        "anchor_entity_id": entity_id,
        "entity_label": "truck",
        "candidate_role": "target_direct",
        "target_hits": ["truck"],
        "attribute_hits": ["color"],
        "constraint_hits": [],
        "direct_attributes": [attribute],
        "direct_relations": [],
        "time_span": {"start_ts": start_ts, "end_ts": end_ts},
        "visual_anchors": [{"similarity": retrieval_similarity}],
        "ranking_score_breakdown": {
            "visual_similarity": retrieval_similarity * 8.0,
        },
        "source": "unit_test",
        "source_kind": "visual_entity_candidate_pack",
        "fact_text": f"{entity_id} {attribute}",
    }


def test_entity_attribute_score_contract_prefers_window_aligned_attribute_candidate() -> None:
    scored = apply_candidate_pack_score_contract(
        [
            _candidate_pack(
                "outside-window",
                start_ts=300.0,
                end_ts=310.0,
                retrieval_similarity=0.99,
                attribute="white truck",
            ),
            _candidate_pack(
                "inside-window",
                start_ts=100.0,
                end_ts=110.0,
                retrieval_similarity=0.55,
                attribute="yellow truck",
            ),
        ],
        question="What color was the truck?",
        question_ir={
            "target_entities": ["truck"],
            "attribute_dimension": "color",
        },
        question_route={"route_id": "entity_attribute"},
        retrieval_window={"applied": True, "start_ts": 95.0, "end_ts": 120.0},
    )

    assert scored[0]["anchor_entity_id"] == "inside-window"
    assert scored[0]["question_window_alignment"]["overlaps"] is True
    assert scored[0]["final_rank_key"]["question_window_alignment_score"] > scored[1][
        "final_rank_key"
    ]["question_window_alignment_score"]
    assert scored[0]["uses_gold_answers"] is False


def test_question_window_alignment_does_not_change_binary_existence_ranking() -> None:
    scored = apply_candidate_pack_score_contract(
        [
            _candidate_pack(
                "outside-window",
                start_ts=300.0,
                end_ts=310.0,
                retrieval_similarity=0.99,
                attribute="white truck",
            ),
            _candidate_pack(
                "inside-window",
                start_ts=100.0,
                end_ts=110.0,
                retrieval_similarity=0.55,
                attribute="yellow truck",
            ),
        ],
        question="Did you see a truck?",
        question_ir={"target_entities": ["truck"]},
        question_route={"route_id": "binary_existence"},
        retrieval_window={"applied": True, "start_ts": 95.0, "end_ts": 120.0},
    )

    assert scored[0]["anchor_entity_id"] == "outside-window"
    assert "question_window_alignment" not in scored[0]
    assert "question_window_alignment_score" not in scored[0]["final_rank_key"]
