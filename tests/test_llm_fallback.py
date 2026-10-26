from pathlib import Path

import numpy as np

from vlkg.core.temporal.association import AssociationCandidate, AssociationDecision, AssociationFallbackRequest, AssociationInput
from vlkg.core.config.config import LLMFallbackConfig
from vlkg.core.temporal.decision_log import NullDecisionLogger
from vlkg.core.temporal.llm_fallback import OpenAICompatibleChatResolver, resolve_decision_log_path
from vlkg.core.models.temporal_graph import BoundingBox


def test_relative_decision_log_path_resolves_against_workspace(
    tmp_path: Path,
) -> None:
    resolved = resolve_decision_log_path(
        ".scratch/fallback/run.decisions.jsonl",
        base_dir=tmp_path,
    )

    assert resolved == tmp_path / ".scratch/fallback/run.decisions.jsonl"


def test_absolute_decision_log_path_is_preserved(tmp_path: Path) -> None:
    absolute = tmp_path / "run.decisions.jsonl"

    assert resolve_decision_log_path(str(absolute), base_dir=Path("ignored")) == absolute


def _fallback_request() -> AssociationFallbackRequest:
    current = AssociationInput(
        video_id="seq_test",
        frame_name="frame_000020.jpg",
        current_label="door",
        current_frame_rank=2,
        current_frame_index=20,
        current_track_id=None,
        object_index=0,
        current_image_path="frame_000020.jpg",
        current_parent_label=None,
        current_state_values=[],
        current_is_static=False,
        current_is_noise=False,
        current_bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        current_embedding=np.array([1.0, 0.0]),
        label_compatibility_groups=[],
        current_match_max_frame_gap=2,
    )
    candidate = AssociationCandidate(
        entity_id="entity_1",
        label="door",
        last_frame_rank=1,
        last_frame_index=10,
        track_id=None,
        object_index=0,
        last_bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        parent_label=None,
        last_state_values=[],
        is_static=False,
        is_noise=False,
        observation_count=1,
        canonical_image_path=None,
        embedding=np.array([1.0, 0.0]),
    )
    return AssociationFallbackRequest(
        current=current,
        filtered_candidates=(candidate,),
        ranked_candidate_scores=(("entity_1", 0.76),),
        ambiguity_reason="slow_margin_ambiguous",
        default_decision=AssociationDecision(
            matched_entity_id="entity_1",
            path="slow",
            score=0.76,
        ),
    )


def test_abstain_decision_is_parsed() -> None:
    resolver = OpenAICompatibleChatResolver(
        LLMFallbackConfig(),
        decision_logger=NullDecisionLogger(),
    )

    decision = resolver._decision_from_payload(
        parsed={
            "decision": "abstain",
            "matched_entity_id": None,
            "confidence": 0.1,
            "reason": "not enough evidence",
        },
        request=_fallback_request(),
    )

    assert decision.path == "abstain"
    assert decision.matched_entity_id is None
    assert decision.confidence == 0.1


def test_invalid_llm_decision_defaults_to_abstain() -> None:
    resolver = OpenAICompatibleChatResolver(
        LLMFallbackConfig(),
        decision_logger=NullDecisionLogger(),
    )

    decision = resolver._decision_from_payload(
        parsed={
            "decision": "maybe",
            "matched_entity_id": "entity_1",
            "confidence": 0.9,
        },
        request=_fallback_request(),
    )

    assert decision.path == "abstain"
