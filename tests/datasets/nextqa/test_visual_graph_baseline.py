from __future__ import annotations

from vlkg.datasets.nextqa.runtime.visual_runtime_v1.baseline import (
    compare_prediction_record_ids,
    freeze_graph_records,
    graph_record_ids_sha256,
)


def _retrieval_row() -> dict[str, object]:
    return {
        "question_id": "nextqa-val/100/q1",
        "video_id": "100",
        "status": "completed",
        "graph_retrieval_profile": "unified",
        "graph_record_ids": ["record:1"],
        "evidence": [
            {
                "rank": 1,
                "source_record_id": "record:1",
                "source_kind": "entity",
                "evidence_text": "person at 1.0s",
                "entity_ids": ["entity:1"],
                "time_range": {"start_ts": 1.0, "end_ts": 1.0},
                "score": 1.0,
            }
        ],
    }


def test_freeze_preserves_record_order_and_text() -> None:
    frozen = freeze_graph_records([_retrieval_row()])
    assert frozen[0]["graph_record_ids"] == ["record:1"]
    assert frozen[0]["evidence"][0]["evidence_text"] == "person at 1.0s"
    assert frozen[0]["graph_record_ids_sha256"] == graph_record_ids_sha256(
        ["record:1"]
    )


def test_prediction_record_ids_must_match_exactly() -> None:
    frozen = freeze_graph_records([_retrieval_row()])
    predictions = [
        {
            "question_id": "nextqa-val/100/q1",
            "status": "completed",
            "graph_record_ids": ["record:1"],
        }
    ]
    comparison = compare_prediction_record_ids(frozen, predictions)
    assert comparison["passed"] is True
    assert comparison["completed_record_id_matches"] == 1


def test_prediction_record_id_reordering_fails() -> None:
    row = _retrieval_row()
    row["graph_record_ids"] = ["record:1", "record:2"]
    row["evidence"] = [
        {**row["evidence"][0], "source_record_id": "record:1"},
        {**row["evidence"][0], "source_record_id": "record:2", "rank": 2},
    ]
    frozen = freeze_graph_records([row])
    comparison = compare_prediction_record_ids(
        frozen,
        [
            {
                "question_id": "nextqa-val/100/q1",
                "status": "completed",
                "graph_record_ids": ["record:2", "record:1"],
            }
        ],
    )
    assert comparison["passed"] is False
    assert comparison["mismatch_count"] == 1
