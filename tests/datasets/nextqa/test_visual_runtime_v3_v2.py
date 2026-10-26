from __future__ import annotations

import pytest

from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain_v2 import (
    HANDLER_ID,
    STRATEGY_ID,
    build_coherent_causal_chain_pack_v2,
)


def _hit(kind: str, candidate_id: str, similarity: float, **metadata: object) -> dict[str, object]:
    base = {"graph_bound": True}
    base.update(metadata)
    return {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "video_id": "v1",
        "rank_within_kind": 1,
        "similarity": similarity,
        "metadata": base,
    }


def _route() -> dict[str, str]:
    return {"route_id": "causal_why", "video_id": "v1"}


def test_shared_pair_cannot_fill_effect_and_preceding_quota() -> None:
    shared = _hit(
        "before_after_pair",
        "shared",
        0.99,
        event_id="event:shared",
        boundary_ts=9.0,
        before_ts=8.0,
        after_ts=10.0,
        before_labels=["person"],
        after_labels=["person"],
        strict_temporal_pair=True,
    )
    effect = _hit(
        "event_boundary",
        "effect",
        0.9,
        event_id="event:effect",
        timestamp_sec=10.0,
        labels=["person"],
    )
    effect_candidates = [
        _hit(
            "event_boundary",
            f"effect-{index}",
            0.85 - index * 0.01,
            event_id=f"event:effect-{index}",
            timestamp_sec=10.0,
            labels=["person"],
        )
        for index in range(1, 4)
    ]
    preceding = [
        _hit(
            "before_after_pair",
            f"before-{index}",
            0.8 - index * 0.01,
            event_id=f"event:before-{index}",
            boundary_ts=7.0 - index,
            before_ts=6.0 - index,
            after_ts=8.0 - index,
            before_labels=["person"],
            after_labels=["person"],
            strict_temporal_pair=True,
        )
        for index in range(1, 5)
    ]
    interaction = [
        _hit(
            "object",
            f"interaction-{index}",
            0.6,
            entity_id=f"e{index}",
            timestamp_sec=9.0,
            label="person",
            state_values=[],
        )
        for index in range(1, 5)
    ]
    pack = build_coherent_causal_chain_pack_v2(
        _route(),
        {
            "effect_event": [effect, *effect_candidates, shared],
            "preceding_action_or_state": [shared, *preceding],
            "actor_object_interaction": interaction,
        },
    )
    ids = pack["used_visual_candidate_ids"]
    assert len(ids) == len(set(ids)) == 12
    assert ids.count("shared") == 1
    assert pack["strategy_metadata"]["role_counts"] == {
        "actor_object_interaction": 4,
        "effect_event": 4,
        "preceding_action_or_state": 4,
    }
    assert all(len(row["role_ids"]) == 1 for row in pack["evidence"])


def test_pair_boundary_must_be_before_effect_anchor() -> None:
    effect = _hit(
        "event_boundary", "effect", 0.9, event_id="event:effect", timestamp_sec=10.0, labels=["person"]
    )
    same_boundary = _hit(
        "before_after_pair",
        "same-boundary",
        0.99,
        event_id="event:same",
        boundary_ts=10.0,
        before_ts=9.0,
        after_ts=11.0,
        before_labels=["person"],
        after_labels=["person"],
        strict_temporal_pair=True,
    )
    early_window = _hit(
        "temporal_window",
        "early-window",
        0.2,
        start_ts=6.0,
        end_ts=8.0,
        labels=["person"],
        states=["moving"],
        relations=[],
        events=[],
    )
    interactions = [
        _hit("object", f"object-{index}", 0.2, entity_id=f"e{index}", timestamp_sec=9.0, label="person", state_values=[])
        for index in range(1, 5)
    ]
    pack = build_coherent_causal_chain_pack_v2(
        _route(),
        {
            "effect_event": [effect],
            "preceding_action_or_state": [same_boundary, early_window],
            "actor_object_interaction": interactions,
        },
    )
    preceding_ids = [row["candidate_id"] for row in pack["evidence"] if row["role_id"] == "preceding_action_or_state"]
    assert "same-boundary" not in preceding_ids
    assert "early-window" in preceding_ids


def test_temporal_window_fallback_is_explicit_when_strict_candidates_are_short() -> None:
    effect = _hit(
        "event_boundary", "effect", 0.9, event_id="event:effect", timestamp_sec=10.0, labels=["person"]
    )
    overlap_window = _hit(
        "temporal_window",
        "overlap",
        0.5,
        start_ts=8.0,
        end_ts=11.0,
        labels=["person"],
        states=["moving"],
        relations=[],
        events=[],
    )
    interactions = [
        _hit("object", f"object-{index}", 0.2, entity_id=f"e{index}", timestamp_sec=9.0, label="person", state_values=[])
        for index in range(1, 5)
    ]
    pack = build_coherent_causal_chain_pack_v2(
        _route(),
        {
            "effect_event": [effect],
            "preceding_action_or_state": [overlap_window],
            "actor_object_interaction": interactions,
        },
    )
    row = next(row for row in pack["evidence"] if row["candidate_id"] == "overlap")
    assert row["selection_mode"] == "temporal_window_fallback"
    assert pack["strategy_metadata"]["selection_mode_counts"]["temporal_window_fallback"] == 1


def test_v4_identity_is_explicit_and_route_is_fixed() -> None:
    assert STRATEGY_ID == "causal_why_coherent_chain_v2"
    assert HANDLER_ID == "nextqa_visual_handler_v4:causal_why_coherent_chain"
    with pytest.raises(ValueError, match="route_id=causal_why"):
        build_coherent_causal_chain_pack_v2({"route_id": "causal_how", "video_id": "v1"}, {})
