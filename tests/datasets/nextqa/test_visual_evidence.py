from __future__ import annotations

from vlkg.datasets.nextqa.runtime.visual_runtime_v1.evidence import (
    qa_safe_visual_hit,
    render_route_evidence,
)


def _route(route_id: str) -> dict[str, object]:
    return {
        "route_id": route_id,
        "video_id": "100",
        "evidence_demand": {"candidate_kinds": ["before_after_pair"]},
    }


def _pair_hit() -> dict[str, object]:
    return {
        "candidate_id": "pair:1",
        "candidate_kind": "before_after_pair",
        "video_id": "100",
        "rank_within_kind": 1,
        "similarity": 0.8,
        "metadata": {
            "candidate_id": "pair:1",
            "boundary_ts": 2.0,
            "before_ts": 1.0,
            "before_labels": ["person", "cup"],
            "after_ts": 3.0,
            "after_labels": ["person", "table"],
            "image_path": "must-not-reach-qa.png",
        },
    }


def test_previous_and_next_handlers_select_the_correct_side() -> None:
    previous, _ = render_route_evidence(
        _route("temporal_previous"), [_pair_hit()], evidence_limit=4
    )
    following, _ = render_route_evidence(
        _route("temporal_next"), [_pair_hit()], evidence_limit=4
    )
    assert "frame at 1.000s" in previous[0]
    assert "person, cup" in previous[0]
    assert "frame at 3.000s" in following[0]
    assert "person, table" in following[0]


def test_count_handler_deduplicates_entity_ids() -> None:
    hits = [
        {
            "candidate_id": f"candidate:{index}",
            "candidate_kind": "tracklet",
            "metadata": {
                "entity_id": entity_id,
                "label": "person",
                "start_ts": 0.0,
                "end_ts": 2.0,
            },
        }
        for index, entity_id in enumerate(("entity:1", "entity:1", "entity:2"))
    ]
    lines, used = render_route_evidence(
        _route("descriptive_count"), hits, evidence_limit=10
    )
    assert len(lines) == 1
    assert len(used) == 2


def test_qa_safe_hit_removes_all_image_paths() -> None:
    safe = qa_safe_visual_hit(_pair_hit())
    assert "image_path" not in safe["metadata"]
    assert "must-not-reach-qa.png" not in repr(safe)
