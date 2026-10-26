"""OpenEQA object-identity evidence strategies."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import verify_spatial_pairs
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.open_target_relation import _canonical_relation_pairs, _entity_lookup, _merge_relation_pairs, _relation_pair_pack, _rerank_geometry_pairs, _resolve_component_pair_owners, _role_hits
HANDLER_ID = 'object_identity.anchor_bound_relation_incremental_v1'
FROZEN_HANDLER_ID = 'object_identity.frozen_v3'
OBJECT_IDENTITY_RELATION_CONTRACT = 'openeqa_object_identity_relation_incremental.v1'
_ANCHOR_BOUND_VARIANT = 'anchor_bound_open_target_identity'

def fuse_object_identity_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Add relation-bound identity candidates without replacing frozen-v3.

    The frozen bundle remains authoritative.  New evidence is published only
    through compact visual candidate-pack channels and is accepted when a
    target is tied to a question anchor by canonical relation or same-frame
    geometry.  No answer text is used for binding.
    """
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    if str(evidence_demand.get('strategy_variant') or '') != _ANCHOR_BOUND_VARIANT:
        return fused
    entity_by_id = _entity_lookup(graph)
    verifier_demand = deepcopy(evidence_demand)
    verifier_demand['open_target_slot'] = True
    verifier_demand['target_terms'] = []
    raw_geometry_pairs = verify_spatial_pairs(visual_payload, graph=graph, evidence_demand=verifier_demand, limit=max(12, max(1, int(max_candidate_packs)) * 3))
    geometry_pairs, owner_resolution = _resolve_component_pair_owners(raw_geometry_pairs, graph=graph)
    _, target_observations = _role_hits(visual_payload, evidence_demand=evidence_demand)
    geometry_pairs = _rerank_geometry_pairs(geometry_pairs, target_observations=target_observations, entity_by_id=entity_by_id, target_terms=[])
    canonical_pairs = _canonical_relation_pairs(visual_payload, graph=graph, evidence_demand=evidence_demand)
    pairs = _merge_relation_pairs([*canonical_pairs, *geometry_pairs], limit=max(1, int(max_visual_lines)), target_terms=[])
    selected = pairs[:max(1, int(max_candidate_packs))]
    visual_evidence = fused.get('visual_evidence')
    bundle = fused.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return fused
    old_packs = [dict(pack) for pack in visual_evidence.get('candidate_packs', []) or [] if isinstance(pack, dict)]
    identity_packs = [_identity_pair_pack(pair) for pair in selected]
    identity_lines = [_identity_pair_line(index, pair) for index, pair in enumerate(selected, start=1)]
    if selected:
        bundle['question_route_anchor_bound_identity_lines'] = identity_lines
        structured = bundle.get('structured_graph_context')
        if isinstance(structured, dict):
            structured = deepcopy(structured)
            structured['question_route_anchor_bound_identity_lines'] = identity_lines
            structured['object_identity_relation_candidates'] = identity_packs
            bundle['structured_graph_context'] = structured
    route_applied = bool(selected)
    visual_evidence.update({'object_identity_relation_contract': OBJECT_IDENTITY_RELATION_CONTRACT, 'object_identity_relation_candidates': identity_packs, 'object_identity_relation_candidate_lines': identity_lines, 'object_identity_relation_verification': {'contract': OBJECT_IDENTITY_RELATION_CONTRACT, 'applied': route_applied, 'canonical_pair_count': len(canonical_pairs), 'raw_geometry_pair_count': len(raw_geometry_pairs), 'answer_visible_pair_count': len(selected), 'target_anchor_roles_are_separate': True, 'relation_verified_candidates_only': True, 'component_owner_resolution': owner_resolution, 'fallback_to_frozen_v3': not route_applied, 'fallback_reason': '' if route_applied else 'no_relation_verified_candidates', 'uses_gold_answer': False}, 'route_strategy_applied': route_applied, 'route_strategy_variant': _ANCHOR_BOUND_VARIANT, 'candidate_packs': old_packs, 'supplemental_candidate_packs': identity_packs, 'stats': {**dict(visual_evidence.get('stats') or {}), 'object_identity_relation_canonical_pair_count': len(canonical_pairs), 'object_identity_relation_geometry_pair_count': len(raw_geometry_pairs), 'object_identity_relation_candidate_count': len(selected)}})
    fused['visual_evidence'] = visual_evidence
    fused['llm_ready_bundle'] = bundle
    return fused

def _identity_pair_pack(pair: dict[str, Any]) -> dict[str, Any]:
    pack = _relation_pair_pack(pair)
    pack.update({'contract': OBJECT_IDENTITY_RELATION_CONTRACT, 'candidate_id': 'object_identity_relation:' + str(pair.get('target_entity_id') or pair.get('target_label') or 'object'), 'candidate_role': 'anchor_bound_identity_candidate', 'binding_strength': 'anchor_relation_verified', 'identity_answer_candidate': True})
    return pack

def _identity_pair_line(index: int, pair: dict[str, Any]) -> str:
    return f"rank={index} kind=openeqa_anchor_bound_identity_candidate target={pair.get('target_label')!r} target_entity={pair.get('target_entity_id') or 'unbound_visual_observation'} relation={pair.get('relation') or ''} anchor={pair.get('anchor_label')!r} anchor_entity={pair.get('anchor_entity_id') or 'unbound_visual_observation'} frame={pair.get('frame_index')} pair_score={float(pair.get('pair_score') or 0.0):.3f} evidence_tier={int(pair.get('relation_evidence_tier') or 0)} binding=anchor_relation_verified candidate_role=answer_entity relation_verified=true unique_answer_not_proven=true uses_gold_answer=false"
__all__ = ['FROZEN_HANDLER_ID', 'HANDLER_ID', 'OBJECT_IDENTITY_RELATION_CONTRACT', 'fuse_object_identity_evidence']
