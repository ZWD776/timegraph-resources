"""Explicit OpenEQA boundaries for descriptive and unsupported IR routes."""
from __future__ import annotations
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
SCENE_DESCRIPTION_HANDLER_ID = 'open_descriptive.scene_or_content_boundary_v1'
SCENE_DESCRIPTION_CONTRACT = 'openeqa_scene_or_content_boundary.v1'
UNSUPPORTED_FALLBACK_HANDLER_ID = 'open_descriptive.unsupported.frozen_v3'

def fuse_scene_or_content_description_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence['scene_description_contract'] = SCENE_DESCRIPTION_CONTRACT
        visual_evidence['route_strategy_applied'] = False
        visual_evidence['handler_fallback_reason'] = 'no_dedicated_scene_description_layer'
    return fused

def fuse_unsupported_fallback_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    return fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
__all__ = ['SCENE_DESCRIPTION_HANDLER_ID', 'SCENE_DESCRIPTION_CONTRACT', 'UNSUPPORTED_FALLBACK_HANDLER_ID', 'fuse_scene_or_content_description_evidence', 'fuse_unsupported_fallback_evidence']
