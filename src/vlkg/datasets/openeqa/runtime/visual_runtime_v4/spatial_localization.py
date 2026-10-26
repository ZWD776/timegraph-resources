"""OpenEQA spatial-localization strategies, isolated by persisted variant."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import adapt_visual_payload_for_navqa_backbone, navqa_equivalent_question_ir
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_entity_candidate_packs import fuse_visual_entity_candidate_packs
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'spatial_localization.variant_v1'
TARGET_LOCATION_CONTRACT = 'openeqa_target_location_entity_pack.v1'
_FROZEN_V3_PROMPT_FIELDS = ('evidence_lines', 'evidence_aggregation', 'evidence_aggregation_lines', 'llm_context_text', 'structured_graph_context')

def fuse_spatial_localization_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    variant = str(evidence_demand.get('strategy_variant') or '')
    if variant != 'target_location':
        return fused
    adapted_ir = navqa_equivalent_question_ir(question_ir, evidence_demand=evidence_demand)
    adapted_payload = adapt_visual_payload_for_navqa_backbone(visual_payload, graph=graph, question=question, question_ir=adapted_ir, max_lines=max_visual_lines)
    frozen_bundle = fused.get('llm_ready_bundle')
    frozen_bundle = frozen_bundle if isinstance(frozen_bundle, dict) else {}
    frozen_prompt_fields = {key: deepcopy(frozen_bundle[key]) for key in _FROZEN_V3_PROMPT_FIELDS if key in frozen_bundle}
    fused = fuse_visual_entity_candidate_packs(fused, adapted_payload, graph=graph, object_limit=min(4, max(1, int(max_candidate_packs))), neighbor_limit=6, question=question, question_ir=adapted_ir, semantic_align_enabled=False, semantic_gated_role_enabled=False, score_contract_enabled=False, retrieval_window=None, semantic_text_encoder=semantic_text_encoder)
    bundle = fused.get('llm_ready_bundle')
    visual_evidence = fused.get('visual_evidence')
    if not isinstance(bundle, dict) or not isinstance(visual_evidence, dict):
        return fused
    for key in _FROZEN_V3_PROMPT_FIELDS:
        if key in frozen_prompt_fields:
            bundle[key] = frozen_prompt_fields[key]
        else:
            bundle.pop(key, None)
    pack_metadata = bundle.get('visual_entity_candidate_packs')
    pack_metadata = pack_metadata if isinstance(pack_metadata, dict) else {}
    packs = [dict(pack) for pack in pack_metadata.get('packs', []) or [] if isinstance(pack, dict)]
    pack_lines = [str(line) for line in bundle.get('visual_entity_candidate_pack_lines', []) or [] if str(line).strip()]
    visual_evidence['target_location_contract'] = TARGET_LOCATION_CONTRACT
    visual_evidence['target_location_candidate_packs'] = packs
    visual_evidence['target_location_candidate_pack_lines'] = pack_lines
    visual_evidence['route_strategy_applied'] = bool(packs)
    visual_evidence['route_strategy_variant'] = variant
    stats = dict(visual_evidence.get('stats') or {})
    stats['target_location_candidate_pack_count'] = len(packs)
    visual_evidence['stats'] = stats
    return fused
__all__ = ['HANDLER_ID', 'TARGET_LOCATION_CONTRACT', 'fuse_spatial_localization_evidence']
