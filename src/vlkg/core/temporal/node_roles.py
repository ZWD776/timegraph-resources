from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NodeRole = Literal["static", "dynamic", "dependent", "noise"]


@dataclass(frozen=True)
class NodeRoleEvidence:
    label: str
    parent_label: str | None = None
    lexical_static_prior: bool = False
    lexical_dynamic_prior: bool = False
    lexical_dependent_prior: bool = False
    hierarchy_anchor: bool = False
    scene_anchor: bool = False
    max_prev_same_label_iou: float = 0.0
    max_next_same_label_iou: float = 0.0
    is_noise_label: bool = False

    @property
    def max_neighbor_iou(self) -> float:
        return max(self.max_prev_same_label_iou, self.max_next_same_label_iou)

    @property
    def bidirectional_iou(self) -> float:
        return min(self.max_prev_same_label_iou, self.max_next_same_label_iou)


@dataclass(frozen=True)
class NodeRoleDecision:
    role: NodeRole
    is_static: bool
    confidence: float
    reason: str


def classify_node_role(
    evidence: NodeRoleEvidence,
    *,
    static_iou_threshold: float = 0.65,
    anchor_iou_threshold: float = 0.50,
    dynamic_iou_upper_bound: float = 0.35,
) -> NodeRoleDecision:
    if evidence.is_noise_label:
        return NodeRoleDecision(
            role="noise",
            is_static=False,
            confidence=1.0,
            reason="noise_label",
        )

    if evidence.lexical_dependent_prior:
        return NodeRoleDecision(
            role="dependent",
            is_static=False,
            confidence=0.95,
            reason="dependent_semantic_prior",
        )

    stable_by_iou = (
        evidence.max_neighbor_iou >= static_iou_threshold
        or evidence.bidirectional_iou >= anchor_iou_threshold
    )
    unstable_by_iou = evidence.max_neighbor_iou <= dynamic_iou_upper_bound

    if evidence.lexical_dynamic_prior and not evidence.scene_anchor:
        return NodeRoleDecision(
            role="dynamic",
            is_static=False,
            confidence=0.9,
            reason="dynamic_semantic_prior",
        )

    if evidence.scene_anchor and stable_by_iou:
        return NodeRoleDecision(
            role="static",
            is_static=True,
            confidence=0.92,
            reason="scene_anchor_and_neighbor_iou",
        )

    if evidence.lexical_static_prior and (stable_by_iou or not evidence.lexical_dynamic_prior):
        return NodeRoleDecision(
            role="static",
            is_static=True,
            confidence=0.85,
            reason="static_semantic_prior",
        )

    if evidence.hierarchy_anchor and stable_by_iou:
        return NodeRoleDecision(
            role="static",
            is_static=True,
            confidence=0.82,
            reason="hierarchy_anchor_and_neighbor_iou",
        )

    if evidence.scene_anchor and not evidence.lexical_dynamic_prior and not unstable_by_iou:
        return NodeRoleDecision(
            role="static",
            is_static=True,
            confidence=0.75,
            reason="scene_anchor_prior",
        )

    if evidence.lexical_dynamic_prior:
        return NodeRoleDecision(
            role="dynamic",
            is_static=False,
            confidence=0.8,
            reason="dynamic_semantic_fallback",
        )

    if stable_by_iou and not evidence.lexical_dynamic_prior:
        return NodeRoleDecision(
            role="static",
            is_static=True,
            confidence=0.68,
            reason="neighbor_iou_stability",
        )

    return NodeRoleDecision(
        role="dynamic",
        is_static=False,
        confidence=0.6,
        reason="default_dynamic_fallback",
    )
