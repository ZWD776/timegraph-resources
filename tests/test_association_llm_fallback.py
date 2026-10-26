from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlkg.core.temporal.association import (
    AssociationCandidate,
    AssociationConfig,
    AssociationDecision,
    AssociationFallbackRequest,
    AssociationPolicy,
)
from vlkg.core.models.temporal_graph import BoundingBox


@dataclass
class StubResolver:
    decision: AssociationDecision
    requests: list[AssociationFallbackRequest]

    def resolve(
        self,
        *,
        request: AssociationFallbackRequest,
    ) -> AssociationDecision | None:
        self.requests.append(request)
        return self.decision


def _candidate(
    entity_id: str,
    *,
    embedding: np.ndarray,
) -> AssociationCandidate:
    return AssociationCandidate(
        entity_id=entity_id,
        label="person",
        last_frame_rank=1,
        last_frame_index=10,
        track_id=None,
        object_index=0,
        last_bbox=BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        parent_label=None,
        last_state_values=[],
        is_static=False,
        is_noise=False,
        observation_count=3,
        canonical_image_path=None,
        embedding=embedding,
    )


def _current_input():
    from vlkg.core.temporal.association import AssociationInput

    return AssociationInput(
        video_id="seq_test",
        frame_name="frame_000020.jpg",
        current_label="person",
        current_frame_rank=2,
        current_frame_index=20,
        current_track_id=None,
        object_index=0,
        current_image_path="frame_000020.jpg",
        current_parent_label=None,
        current_state_values=[],
        current_is_static=False,
        current_is_noise=False,
        current_bbox=BoundingBox(x1=300.0, y1=100.0, x2=340.0, y2=180.0),
        current_embedding=np.array([1.0, 0.0]),
        label_compatibility_groups=[],
        current_match_max_frame_gap=2,
    )


def _ambiguous_policy(
    resolver: StubResolver,
) -> AssociationPolicy:
    return AssociationPolicy(
        AssociationConfig(),
        enable_llm_fallback=True,
        fallback_resolver=resolver,
        fallback_override_confidence_threshold=0.80,
    )


def _ambiguous_candidates() -> list[AssociationCandidate]:
    return [
        _candidate("entity_a", embedding=np.array([1.0, 0.0])),
        _candidate("entity_b", embedding=np.array([0.99, 0.1])),
    ]


def test_low_confidence_new_keeps_slow_default() -> None:
    resolver = StubResolver(
        decision=AssociationDecision(
            matched_entity_id=None,
            path="new",
            confidence=0.60,
        ),
        requests=[],
    )

    decision = _ambiguous_policy(resolver).select_entity_match(
        candidates=_ambiguous_candidates(),
        current=_current_input(),
    )

    assert decision.path == "slow"
    assert decision.matched_entity_id == "entity_a"
    assert resolver.requests[0].default_decision.path == "slow"
    assert resolver.requests[0].default_decision.matched_entity_id == "entity_a"


def test_abstain_keeps_slow_default() -> None:
    resolver = StubResolver(
        decision=AssociationDecision(
            matched_entity_id=None,
            path="abstain",
            confidence=0.20,
        ),
        requests=[],
    )

    decision = _ambiguous_policy(resolver).select_entity_match(
        candidates=_ambiguous_candidates(),
        current=_current_input(),
    )

    assert decision.path == "slow"
    assert decision.matched_entity_id == "entity_a"


def test_high_confidence_new_can_override_slow_default() -> None:
    resolver = StubResolver(
        decision=AssociationDecision(
            matched_entity_id=None,
            path="new",
            confidence=0.92,
        ),
        requests=[],
    )

    decision = _ambiguous_policy(resolver).select_entity_match(
        candidates=_ambiguous_candidates(),
        current=_current_input(),
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_high_confidence_match_can_override_slow_default() -> None:
    resolver = StubResolver(
        decision=AssociationDecision(
            matched_entity_id="entity_b",
            path="fallback",
            confidence=0.91,
        ),
        requests=[],
    )

    decision = _ambiguous_policy(resolver).select_entity_match(
        candidates=_ambiguous_candidates(),
        current=_current_input(),
    )

    assert decision.path == "fallback"
    assert decision.matched_entity_id == "entity_b"


def test_fast_bbox_match_can_be_vetoed_by_embedding_similarity() -> None:
    policy = AssociationPolicy(
        AssociationConfig(fast_match_embedding_veto_threshold=0.50),
    )
    current = _current_input()
    candidate = _candidate("entity_opposite", embedding=np.array([-1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "last_bbox": current.current_bbox,
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_same_label_dynamic_spatial_gate_blocks_far_slow_embedding_match() -> None:
    policy = AssociationPolicy(
        AssociationConfig(enable_same_label_dynamic_spatial_gate=True),
    )
    candidate = _candidate("entity_far", embedding=np.array([1.0, 0.0]))

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=_current_input(),
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_static_entity_gap_limit_blocks_stale_static_match() -> None:
    policy = AssociationPolicy(
        AssociationConfig(static_entity_match_max_frame_gap=2),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_is_static": True,
            "current_frame_rank": 5,
            "current_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        }
    )
    candidate = _candidate("static_old", embedding=np.array([1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "is_static": True,
            "last_frame_rank": 1,
            "last_bbox": current.current_bbox,
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_static_entity_gap_limit_allows_recent_static_match() -> None:
    policy = AssociationPolicy(
        AssociationConfig(static_entity_match_max_frame_gap=2),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_is_static": True,
            "current_frame_rank": 3,
            "current_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        }
    )
    candidate = _candidate("static_recent", embedding=np.array([1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "is_static": True,
            "last_frame_rank": 1,
            "last_bbox": current.current_bbox,
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "fast"
    assert decision.matched_entity_id == "static_recent"


def test_continuity_center_gate_blocks_large_bbox_jump() -> None:
    policy = AssociationPolicy(
        AssociationConfig(continuity_max_center_distance_ratio=1.0),
    )
    current = _current_input()
    candidate = _candidate("entity_far_bbox", embedding=np.array([1.0, 0.0]))

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_object_index_jump_requires_embedding_support() -> None:
    policy = AssociationPolicy(
        AssociationConfig(
            object_index_jump_max_delta=2,
            object_index_jump_embedding_veto_threshold=0.80,
        ),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "object_index": 20,
            "current_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        }
    )
    candidate = _candidate("entity_index_jump", embedding=np.array([-1.0, 0.0]))

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_state_conflict_requires_embedding_support() -> None:
    policy = AssociationPolicy(
        AssociationConfig(state_conflict_embedding_veto_threshold=0.80),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
            "current_state_values": ["blue jacket"],
        }
    )
    candidate = _candidate("entity_state_conflict", embedding=np.array([-1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "last_state_values": ["black jacket"],
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_label_conflict_requires_embedding_support() -> None:
    policy = AssociationPolicy(
        AssociationConfig(label_conflict_embedding_veto_threshold=0.80),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_label": "truck",
            "current_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
            "label_compatibility_groups": [{"car", "truck"}],
        }
    )
    candidate = _candidate("entity_label_conflict", embedding=np.array([-1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "label": "car",
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_local_tracklet_mode_matches_adjacent_consistent_candidate() -> None:
    policy = AssociationPolicy(
        AssociationConfig(
            local_tracklet_mode=True,
            local_tracklet_embedding_similarity_threshold=0.70,
            local_tracklet_min_score=0.55,
            local_tracklet_score_margin=0.02,
        ),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_bbox": BoundingBox(x1=106.0, y1=100.0, x2=146.0, y2=180.0),
            "object_index": 1,
        }
    )
    candidate = _candidate("entity_local", embedding=np.array([1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "object_index": 1,
            "last_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=current,
    )

    assert decision.path == "local_tracklet"
    assert decision.matched_entity_id == "entity_local"


def test_local_tracklet_mode_rejects_non_adjacent_candidate() -> None:
    policy = AssociationPolicy(
        AssociationConfig(
            local_tracklet_mode=True,
            local_tracklet_max_frame_gap=1,
            local_tracklet_embedding_similarity_threshold=0.70,
        ),
    )
    candidate = _candidate("entity_old", embedding=np.array([1.0, 0.0]))
    candidate = AssociationCandidate(
        **{
            **candidate.__dict__,
            "last_frame_rank": 0,
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate],
        current=_current_input(),
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None


def test_local_tracklet_mode_rejects_ambiguous_candidate_scores() -> None:
    policy = AssociationPolicy(
        AssociationConfig(
            local_tracklet_mode=True,
            local_tracklet_embedding_similarity_threshold=0.70,
            local_tracklet_min_score=0.50,
            local_tracklet_score_margin=0.20,
        ),
    )
    current = _current_input()
    current = current.__class__(
        **{
            **current.__dict__,
            "current_bbox": BoundingBox(x1=106.0, y1=100.0, x2=146.0, y2=180.0),
            "object_index": 1,
        }
    )
    candidate_a = _candidate("entity_a", embedding=np.array([1.0, 0.0]))
    candidate_a = AssociationCandidate(
        **{
            **candidate_a.__dict__,
            "object_index": 1,
            "last_bbox": BoundingBox(x1=100.0, y1=100.0, x2=140.0, y2=180.0),
        }
    )
    candidate_b = _candidate("entity_b", embedding=np.array([1.0, 0.0]))
    candidate_b = AssociationCandidate(
        **{
            **candidate_b.__dict__,
            "object_index": 2,
            "last_bbox": BoundingBox(x1=102.0, y1=100.0, x2=142.0, y2=180.0),
        }
    )

    decision = policy.select_entity_match(
        candidates=[candidate_a, candidate_b],
        current=current,
    )

    assert decision.path == "new"
    assert decision.matched_entity_id is None
