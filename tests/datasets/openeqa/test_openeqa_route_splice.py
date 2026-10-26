from __future__ import annotations

import pytest

from vlkg.datasets.openeqa.retrieval.route_splice import splice_route_records


def test_route_splice_replaces_only_exact_manifest_ids() -> None:
    baseline = [
        {"question_id": "q1", "answer": "old-1"},
        {"question_id": "q2", "answer": "old-2"},
        {"question_id": "q3", "answer": "old-3"},
    ]
    candidate = [{"question_id": "q2", "answer": "new-2"}]

    merged, report = splice_route_records(
        baseline,
        candidate,
        expected_question_ids=["q2"],
    )

    assert merged == [baseline[0], candidate[0], baseline[2]]
    assert report["non_target_changed_count"] == 0
    assert report["question_order_preserved"] is True


def test_route_splice_rejects_missing_or_cross_route_candidates() -> None:
    baseline = [
        {"question_id": "q1", "answer": "old-1"},
        {"question_id": "q2", "answer": "old-2"},
    ]

    with pytest.raises(ValueError, match="exactly equal"):
        splice_route_records(
            baseline,
            [{"question_id": "q1", "answer": "new-1"}],
            expected_question_ids=["q2"],
        )
