from __future__ import annotations

import pytest

from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import (
    HANDLER_ID,
    STRATEGY_ID,
    build_coherent_causal_chain_pack,
)


def _route() -> dict[str, str]:
    return {"route_id": "causal_why", "video_id": "v1"}


def _hit(
    kind: str,
    candidate_id: str,
    similarity: float,
    *,
    timestamp: float | None = None,
    boundary: float | None = None,
    before: float | None = None,
    after: float | None = None,
    start: float | None = None,
    end: float | None = None,
    entity_ids: list[str] | None = None,
    labels: list[str] | None = None,
    rank: int = 1,
) -> dict[str, object]:
    metadata: dict[str, object] = {"graph_bound": True}
    if kind == "event_boundary":
        metadata.update(
            timestamp_sec=timestamp,
            event_id=f"event:{candidate_id}",
            event_type="DISAPPEAR",
            entity_ids=entity_ids or [],
            labels=labels or [],
        )
    elif kind == "before_after_pair":
        metadata.update(
            boundary_ts=boundary,
            before_ts=before,
            after_ts=after,
            event_id=f"event:{candidate_id}",
            before_entity_ids=entity_ids or [],
            after_entity_ids=entity_ids or [],
            before_labels=labels or [],
            after_labels=labels or [],
            strict_temporal_pair=True,
        )
    elif kind == "temporal_window":
        metadata.update(
            start_ts=start,
            end_ts=end,
            entity_ids=entity_ids or [],
            labels=labels or [],
            states=[],
            relations=[],
            events=[],
        )
    elif kind == "object":
        metadata.update(
            timestamp_sec=timestamp,
            entity_id=(entity_ids or [""])[0],
            label=(labels or ["object"])[0],
            state_values=[],
        )
    elif kind == "tracklet":
        metadata.update(
            start_ts=start,
            end_ts=end,
            entity_id=(entity_ids or [""])[0],
            label=(labels or ["object"])[0],
            labels_seen=labels or [],
            states_seen=[],
        )
    return {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "video_id": "v1",
        "rank_within_kind": rank,
        "similarity": similarity,
        "metadata": metadata,
    }


def test_coherent_chain_prefers_strict_preceding_and_entity_bound_interaction() -> None:
    effect = _hit(
        "event_boundary", "effect", 0.70, timestamp=10.0, entity_ids=["e1"], labels=["baby"]
    )
    wrong_after = _hit(
        "temporal_window", "after", 0.99, start=11.0, end=13.0, entity_ids=["e9"], labels=["car"]
    )
    right_before = _hit(
        "temporal_window", "before", 0.45, start=7.0, end=9.0, entity_ids=["e1"], labels=["baby", "hold"]
    )
    right_interaction = _hit(
        "object", "interaction", 0.30, timestamp=9.0, entity_ids=["e1"], labels=["baby"]
    )
    pack = build_coherent_causal_chain_pack(
        _route(),
        {
            "effect_event": [effect],
            "preceding_action_or_state": [wrong_after, right_before],
            "actor_object_interaction": [right_interaction],
        },
    )
    ids = pack["used_visual_candidate_ids"]
    assert pack["status"] == "completed"
    assert "effect" in ids
    assert "before" in ids
    assert "interaction" in ids
    assert "after" not in ids
    assert pack["strategy_id"] == STRATEGY_ID
    assert pack["handler_id"] == HANDLER_ID


def test_duplicate_candidate_is_merged_and_can_cover_two_roles() -> None:
    shared = _hit(
        "before_after_pair",
        "shared",
        0.8,
        boundary=10.0,
        before=9.0,
        after=11.0,
        entity_ids=["e1"],
        labels=["person"],
    )
    effect = _hit(
        "event_boundary", "effect", 0.5, timestamp=10.0, entity_ids=["e1"], labels=["person"]
    )
    pack = build_coherent_causal_chain_pack(
        _route(),
        {
            "effect_event": [effect, shared],
            "preceding_action_or_state": [shared],
            "actor_object_interaction": [],
        },
        role_quotas={"actor_object_interaction": 1},
    )
    shared_rows = [row for row in pack["evidence"] if row["candidate_id"] == "shared"]
    assert len(shared_rows) == 1
    assert "effect_event" in shared_rows[0]["role_ids"]
    assert "preceding_action_or_state" in shared_rows[0]["role_ids"]
    assert pack["excluded_reason_counts"]["duplicate_candidate_merged"] == 1


def test_boundary_free_questions_use_one_explicit_fallback_group() -> None:
    window = _hit(
        "temporal_window", "window", 0.5, start=1.0, end=2.0, entity_ids=["e1"], labels=["person"]
    )
    tracklet = _hit(
        "tracklet", "tracklet", 0.4, start=1.0, end=2.0, entity_ids=["e1"], labels=["person"]
    )
    pack = build_coherent_causal_chain_pack(
        _route(),
        {
            "effect_event": [],
            "preceding_action_or_state": [window],
            "actor_object_interaction": [tracklet],
        },
    )
    assert pack["status"] == "boundary_unavailable_fallback"
    assert pack["boundary_available"] is False
    assert pack["role_status"]["effect_event"] == "empty"
    assert pack["strategy_metadata"]["uses_gold_answer"] is False


def test_rejects_non_causal_route() -> None:
    with pytest.raises(ValueError, match="route_id=causal_why"):
        build_coherent_causal_chain_pack(
            {"route_id": "causal_how", "video_id": "v1"},
            {},
        )
