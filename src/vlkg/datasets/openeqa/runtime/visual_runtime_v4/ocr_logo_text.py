"""Frozen-preserving OpenEQA OCR/logo/text strategy boundary."""
from __future__ import annotations
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'ocr_logo_text.owner_aware_text_boundary_v1'
OCR_LOGO_TEXT_CONTRACT = 'openeqa_owner_aware_text_boundary.v1'

def fuse_ocr_logo_text_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence['ocr_logo_text_contract'] = OCR_LOGO_TEXT_CONTRACT
        visual_evidence['route_strategy_applied'] = False
        visual_evidence['handler_fallback_reason'] = 'no_dedicated_ocr_observation_layer'
    return fused
__all__ = ['HANDLER_ID', 'OCR_LOGO_TEXT_CONTRACT', 'fuse_ocr_logo_text_evidence']
