"""Fuse episode-scoped visual retrieval into OpenEQA canonical graph evidence."""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import re
from typing import Any, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import NEED_COUNT_SET, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_FUNCTIONAL_WORLD, NEED_OBJECT_IDENTITY, NEED_OCR_LOGO_TEXT, NEED_OPEN_DESCRIPTIVE, NEED_SPATIAL_LOCALIZATION
VISUAL_EVIDENCE_CONTRACT = 'openeqa_graph_bound_visual_evidence.v1'
VISUAL_EVIDENCE_V2_CONTRACT = 'openeqa_graph_bound_visual_evidence.v2'
VISUAL_EVIDENCE_GUARDED_V4_CONTRACT = 'openeqa_guarded_visual_evidence.v4'
VISUAL_OVERRIDE_GUARDED_V4_CONTRACT = 'openeqa_visual_override_guard.v4'
VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT = 'openeqa_guarded_visual_evidence.v4c'
VISUAL_OVERRIDE_GUARDED_V4C_CONTRACT = 'openeqa_visual_override_guard.v4c'
VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT = 'openeqa_guarded_visual_evidence.v4d'
VISUAL_OVERRIDE_GUARDED_V4D_CONTRACT = 'openeqa_visual_override_guard.v4d'
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_ROLE_PRIORITY = {'target': 3, 'spatial_target': 3, 'anchor': 2, 'context': 1}
_COMPONENT_OWNER_TOKENS = {'arm', 'back', 'base', 'cushion', 'door', 'drawer', 'frame', 'handle', 'knob', 'leg', 'lid', 'pillowcase', 'screen', 'seat', 'shelf', 'top'}
_ATTRIBUTE_VALUE_TOKENS = {'color': {'beige', 'black', 'blue', 'brown', 'cream', 'cyan', 'gold', 'gray', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'teal', 'white', 'yellow'}, 'material': {'ceramic', 'cloth', 'concrete', 'fabric', 'glass', 'leather', 'marble', 'metal', 'metallic', 'plastic', 'stone', 'velvet', 'wood', 'wooden'}, 'shape': {'circular', 'cylindrical', 'oval', 'rectangular', 'round', 'square', 'triangular'}, 'size': {'big', 'large', 'small', 'tall', 'tiny', 'wide'}, 'texture': {'fuzzy', 'rough', 'smooth', 'soft', 'textured'}, 'pattern': {'checkered', 'floral', 'patterned', 'plain', 'striped'}}
_STRONG_COMPONENT_LABEL_TOKENS = {'armrest', 'backrest', 'button', 'cushion', 'display', 'handle', 'knob', 'panel', 'rung', 'seat', 'slat', 'surface', 'top'}
_WEAK_COMPONENT_LABEL_TOKENS = {'base', 'casing', 'cushion', 'drawer', 'frame', 'leg', 'lid', 'pane', 'screen', 'seat', 'shelf', 'surface'}
_COMPONENT_RELATIONS = {'attached_to', 'belongs_to', 'component_of', 'part_of'}

def fuse_graph_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8) -> dict[str, Any]:
    """Create owner-bound visual packs and prepend them to graph evidence."""
    updated_result = deepcopy(graph_result)
    bundle = dict(updated_result.get('llm_ready_bundle') or {})
    if not bundle:
        return updated_result
    spatial_pairs = _spatial_pair_evidence(visual_payload, evidence_demand=evidence_demand, limit=4)
    candidates = _candidate_packs(visual_payload, graph=graph, evidence_demand=evidence_demand, spatial_pairs=spatial_pairs, limit=max(1, int(max_candidate_packs)))
    visual_lines = _visual_evidence_lines(candidates, spatial_pairs=spatial_pairs, evidence_demand=evidence_demand, limit=max(1, int(max_visual_lines)))
    stats = _visual_stats(visual_payload, candidates=candidates)
    visual_evidence = {'contract': VISUAL_EVIDENCE_CONTRACT, 'available': bool(visual_lines), 'episode_history': visual_payload.get('episode_history'), 'evidence_demand': dict(evidence_demand), 'candidate_packs': candidates, 'spatial_pairs': spatial_pairs, 'evidence_lines': visual_lines, 'stats': stats, 'episode_scoped_search': bool(visual_payload.get('episode_scoped_search')), 'uses_gold_answer': False, 'unbound_observations_are_graph_facts': False}
    original_lines = _ordered_unique(bundle.get('evidence_lines', []))
    bundle.update({'evidence_demand': dict(evidence_demand), 'visual_evidence_contract': VISUAL_EVIDENCE_CONTRACT, 'visual_evidence': visual_evidence, 'visual_entity_candidate_packs': candidates, 'visual_entity_candidate_pack_lines': visual_lines, 'visual_faiss_evidence_lines': [], 'evidence_lines': original_lines})
    structured = dict(bundle.get('structured_graph_context') or {})
    structured.update({'evidence_demand': dict(evidence_demand), 'visual_evidence_contract': VISUAL_EVIDENCE_CONTRACT, 'visual_evidence': visual_evidence, 'evidence_lines': original_lines})
    bundle['structured_graph_context'] = structured
    notes = _ordered_unique(bundle.get('uncertainty_notes', []))
    if stats['unbound_hit_count']:
        notes = _ordered_unique([*notes, 'Some visual candidates lack a canonical entity binding and are visual observations only, not graph facts.'])
    if evidence_demand.get('requires_negative_coverage'):
        notes = _ordered_unique([*notes, 'Visual similarity candidates can support presence or state but do not by themselves prove absence.'])
    bundle['uncertainty_notes'] = notes
    updated_result['llm_ready_bundle'] = bundle
    updated_result['visual_evidence'] = visual_evidence
    return updated_result

def fuse_graph_visual_evidence_v2(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8) -> dict[str, Any]:
    """Build concise routed evidence while preserving the v1/v4 contract."""
    need = str(evidence_demand.get('primary_need') or '')
    internal_limit = max(64, int(max_candidate_packs)) if need == NEED_COUNT_SET else max_candidate_packs
    updated_result = fuse_graph_visual_evidence(graph_result, visual_payload, graph=graph, evidence_demand=evidence_demand, max_candidate_packs=internal_limit, max_visual_lines=max_visual_lines)
    visual_evidence = updated_result.get('visual_evidence')
    bundle = updated_result.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return updated_result
    all_candidates = [dict(item) for item in visual_evidence.get('candidate_packs', []) or [] if isinstance(item, dict)]
    duplicate_anchor_count = 0
    for pack in all_candidates:
        anchors = [item for item in pack.get('visual_anchors', []) or [] if isinstance(item, dict)]
        unique_anchors = _unique_visual_anchors(anchors)
        duplicate_anchor_count += len(anchors) - len(unique_anchors)
        pack['visual_anchors'] = unique_anchors[:6]
    count_candidates = _count_candidate_set(all_candidates, evidence_demand=evidence_demand)
    candidates = all_candidates[:max(1, int(max_candidate_packs))]
    spatial_pairs = [dict(item) for item in visual_evidence.get('spatial_pairs', []) or [] if isinstance(item, dict)]
    visual_lines = _visual_evidence_lines_v2(candidates, spatial_pairs=spatial_pairs, evidence_demand=evidence_demand, count_candidates=count_candidates, visual_payload=visual_payload, limit=max(1, int(max_visual_lines)))
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'candidate_pack_count': len(candidates), 'graph_bound_candidate_count': sum((bool(item.get('graph_bound')) for item in candidates)), 'count_candidate_set_size': len(count_candidates), 'duplicate_candidate_anchor_count_removed': duplicate_anchor_count})
    visual_evidence.update({'contract': VISUAL_EVIDENCE_V2_CONTRACT, 'available': bool(visual_lines), 'candidate_packs': candidates, 'count_candidate_set': count_candidates, 'evidence_lines': visual_lines, 'stats': stats, 'route_conditioned_evidence_organization': str(evidence_demand.get('route_id') or '') != 'generic_ger'})
    bundle['visual_evidence_contract'] = VISUAL_EVIDENCE_V2_CONTRACT
    bundle['visual_evidence'] = visual_evidence
    bundle['visual_entity_candidate_packs'] = {'contract': 'openeqa_visual_candidate_packs.v2', 'packs': candidates, 'count_candidate_set': count_candidates, 'uses_gold_answer': False}
    bundle['visual_entity_candidate_pack_lines'] = visual_lines
    structured = dict(bundle.get('structured_graph_context') or {})
    structured['visual_evidence_contract'] = VISUAL_EVIDENCE_V2_CONTRACT
    structured['visual_evidence'] = visual_evidence
    bundle['structured_graph_context'] = structured
    if evidence_demand.get('open_target_slot') and evidence_demand.get('anchor_terms') and (not spatial_pairs):
        bundle['uncertainty_notes'] = _ordered_unique([*bundle.get('uncertainty_notes', []), 'Co-visible visual candidates were retrieved near the anchor, but the requested spatial relation was not verified by crop geometry.'])
    updated_result['visual_evidence'] = visual_evidence
    return updated_result

def fuse_graph_visual_evidence_guarded_v4(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8) -> dict[str, Any]:
    """Expose visual evidence to QA only when route-specific proof is satisfied."""
    updated_result = fuse_graph_visual_evidence_v2(graph_result, visual_payload, graph=graph, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines)
    visual_evidence = updated_result.get('visual_evidence')
    bundle = updated_result.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return updated_result
    candidates = [dict(item) for item in visual_evidence.get('candidate_packs', []) or [] if isinstance(item, dict)]
    spatial_pairs = [_guarded_spatial_pair(dict(item), graph=graph) for item in visual_evidence.get('spatial_pairs', []) or [] if isinstance(item, dict)]
    verified_pairs = [item for item in spatial_pairs if item['proof']['relation_verified']]
    verified_target_ids = {str(item.get('target_entity_id') or '') for item in verified_pairs} - {''}
    verified_anchor_ids = {str(item.get('anchor_entity_id') or '') for item in verified_pairs} - {''}
    for rank, pack in enumerate(candidates, start=1):
        pack['guard_candidate_rank'] = rank
        pack['proof'] = _candidate_proof(pack, evidence_demand=evidence_demand, verified_target_ids=verified_target_ids, verified_anchor_ids=verified_anchor_ids)
    count_candidates = [dict(item) for item in visual_evidence.get('count_candidate_set', []) or [] if isinstance(item, dict)]
    safe_lines = _guarded_visual_evidence_lines(candidates, verified_pairs=verified_pairs, evidence_demand=evidence_demand, count_candidates=count_candidates, visual_payload=visual_payload, limit=max(1, int(max_visual_lines)))
    override_contract = {'contract': VISUAL_OVERRIDE_GUARDED_V4_CONTRACT, 'default_authority': 'canonical_graph_evidence', 'primary_need': str(evidence_demand.get('primary_need') or ''), 'candidate_count': len(candidates), 'verified_spatial_pair_count': len(verified_pairs), 'answerable_visual_line_count': len(safe_lines), 'can_override_graph': any((bool(item.get('proof', {}).get('can_override_graph')) for item in [*candidates, *verified_pairs])), 'absence_proven': False, 'audit_candidates_are_not_answer_evidence': True, 'proof_required_for_visual_override': True, 'uses_gold_answer': False}
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'guarded_answerable_candidate_count': sum((bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_rejected_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_verified_spatial_pair_count': len(verified_pairs), 'guarded_rejected_spatial_pair_count': len(spatial_pairs) - len(verified_pairs)})
    visual_evidence.update({'contract': VISUAL_EVIDENCE_GUARDED_V4_CONTRACT, 'available': bool(safe_lines), 'candidate_packs': candidates, 'spatial_pairs': spatial_pairs, 'verified_spatial_pairs': verified_pairs, 'evidence_lines': safe_lines, 'audit_candidate_packs': candidates, 'visual_override_contract': override_contract, 'stats': stats})
    bundle['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4_CONTRACT
    bundle['visual_evidence'] = visual_evidence
    bundle['visual_override_contract'] = override_contract
    bundle['visual_entity_candidate_packs'] = {'contract': 'openeqa_guarded_visual_candidate_packs.v4', 'packs': candidates, 'verified_spatial_pairs': verified_pairs, 'count_candidate_set': count_candidates, 'audit_only_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'uses_gold_answer': False}
    bundle['visual_entity_candidate_pack_lines'] = safe_lines
    structured = dict(bundle.get('structured_graph_context') or {})
    structured['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4_CONTRACT
    structured['visual_evidence'] = visual_evidence
    structured['visual_override_contract'] = override_contract
    bundle['structured_graph_context'] = structured
    visual_evidence['guard_note'] = 'Visual candidates were retained for audit but none satisfied the route-specific proof required to override canonical graph evidence.' if not safe_lines else ''
    updated_result['visual_evidence'] = visual_evidence
    return updated_result

def fuse_graph_visual_evidence_guarded_v4c(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8) -> dict[str, Any]:
    """Add positive identity proof without weakening v4 relation/state guards."""
    updated_result = fuse_graph_visual_evidence_guarded_v4(graph_result, visual_payload, graph=graph, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines)
    visual_evidence = updated_result.get('visual_evidence')
    bundle = updated_result.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return updated_result
    candidates = [dict(item) for item in visual_evidence.get('candidate_packs', []) or [] if isinstance(item, dict)]
    verified_pairs = [dict(item) for item in visual_evidence.get('verified_spatial_pairs', []) or [] if isinstance(item, dict)]
    for pack in candidates:
        pack['proof'] = _candidate_proof_v4c(pack, evidence_demand=evidence_demand)
    identity_candidates = _attach_identity_observation_proofs(candidates, evidence_demand=evidence_demand)
    count_candidates = [dict(item) for item in visual_evidence.get('count_candidate_set', []) or [] if isinstance(item, dict)]
    safe_lines = _guarded_visual_evidence_lines_v4c(candidates, verified_pairs=verified_pairs, evidence_demand=evidence_demand, count_candidates=count_candidates, visual_payload=visual_payload, identity_candidates=identity_candidates, limit=max(1, int(max_visual_lines)))
    override_contract = {'contract': VISUAL_OVERRIDE_GUARDED_V4C_CONTRACT, 'default_authority': 'canonical_graph_evidence', 'primary_need': str(evidence_demand.get('primary_need') or ''), 'candidate_count': len(candidates), 'verified_spatial_pair_count': len(verified_pairs), 'identity_support_candidate_count': len(identity_candidates), 'answerable_visual_line_count': len(safe_lines), 'can_override_graph': any((bool(item.get('proof', {}).get('can_override_graph')) for item in [*candidates, *verified_pairs])), 'can_support_object_identity': bool(identity_candidates), 'identity_support_is_not_relation_proof': True, 'absence_proven': False, 'audit_candidates_are_not_answer_evidence': True, 'proof_required_for_visual_override': True, 'uses_gold_answer': False}
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'guarded_answerable_candidate_count': sum((bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_rejected_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_identity_support_candidate_count': len(identity_candidates), 'guarded_unbound_identity_support_candidate_count': sum((not bool(item.get('graph_bound')) for item in identity_candidates))})
    visual_evidence.update({'contract': VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT, 'available': bool(safe_lines), 'candidate_packs': candidates, 'identity_support_candidates': identity_candidates, 'evidence_lines': safe_lines, 'visual_override_contract': override_contract, 'stats': stats, 'guard_note': 'Visual candidates were retained for audit but none satisfied identity or route-specific proof.' if not safe_lines else ''})
    bundle['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    bundle['visual_evidence'] = visual_evidence
    bundle['visual_override_contract'] = override_contract
    bundle['visual_entity_candidate_packs'] = {'contract': 'openeqa_guarded_visual_candidate_packs.v4c', 'packs': candidates, 'identity_support_candidates': identity_candidates, 'verified_spatial_pairs': verified_pairs, 'count_candidate_set': count_candidates, 'audit_only_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) and (not bool(item.get('proof', {}).get('identity_observation', {}).get('can_support_object_identity'))) for item in candidates)), 'uses_gold_answer': False}
    bundle['visual_entity_candidate_pack_lines'] = safe_lines
    structured = dict(bundle.get('structured_graph_context') or {})
    structured['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4C_CONTRACT
    structured['visual_evidence'] = visual_evidence
    structured['visual_override_contract'] = override_contract
    bundle['structured_graph_context'] = structured
    updated_result['llm_ready_bundle'] = bundle
    updated_result['visual_evidence'] = visual_evidence
    return updated_result

def fuse_graph_visual_evidence_guarded_v4d(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8) -> dict[str, Any]:
    """Rerank identity evidence and guard constrained functional questions."""
    requested_candidate_limit = max(1, int(max_candidate_packs))
    updated_result = fuse_graph_visual_evidence_guarded_v4c(graph_result, visual_payload, graph=graph, evidence_demand=evidence_demand, max_candidate_packs=requested_candidate_limit, max_visual_lines=max_visual_lines)
    visual_evidence = updated_result.get('visual_evidence')
    bundle = updated_result.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return updated_result
    need = str(evidence_demand.get('primary_need') or '')
    if need not in {NEED_OBJECT_IDENTITY, NEED_OPEN_DESCRIPTIVE}:
        return updated_result
    candidates = [dict(item) for item in visual_evidence.get('candidate_packs', []) or [] if isinstance(item, dict)]
    candidates = _rerank_open_identity_candidates(candidates, evidence_demand=evidence_demand)
    if not any(('identity_selection_score' in pack for pack in candidates)):
        return updated_result
    candidates = candidates[:requested_candidate_limit]
    verified_pairs = [dict(item) for item in visual_evidence.get('verified_spatial_pairs', []) or [] if isinstance(item, dict)]
    verified_target_ids = {str(item.get('target_entity_id') or '') for item in verified_pairs} - {''}
    verified_anchor_ids = {str(item.get('anchor_entity_id') or '') for item in verified_pairs} - {''}
    for rank, pack in enumerate(candidates, start=1):
        pack['guard_candidate_rank'] = rank
        pack['proof'] = _candidate_proof(pack, evidence_demand=evidence_demand, verified_target_ids=verified_target_ids, verified_anchor_ids=verified_anchor_ids)
        pack['proof'] = _candidate_proof_v4c(pack, evidence_demand=evidence_demand)
    identity_candidates = _attach_identity_observation_proofs(candidates, evidence_demand=evidence_demand)
    count_candidates = [dict(item) for item in visual_evidence.get('count_candidate_set', []) or [] if isinstance(item, dict)]
    safe_lines = _guarded_visual_evidence_lines_v4c(candidates, verified_pairs=verified_pairs, evidence_demand=evidence_demand, count_candidates=count_candidates, visual_payload=visual_payload, identity_candidates=identity_candidates, limit=max(1, int(max_visual_lines)))
    override_contract = {'contract': VISUAL_OVERRIDE_GUARDED_V4D_CONTRACT, 'default_authority': 'canonical_graph_evidence', 'primary_need': need, 'candidate_count': len(candidates), 'verified_spatial_pair_count': len(verified_pairs), 'identity_support_candidate_count': len(identity_candidates), 'answerable_visual_line_count': len(safe_lines), 'can_override_graph': any((bool(item.get('proof', {}).get('can_override_graph')) for item in [*candidates, *verified_pairs])), 'can_support_object_identity': bool(identity_candidates), 'identity_support_is_not_relation_proof': True, 'open_identity_component_suppression': True, 'absence_proven': False, 'audit_candidates_are_not_answer_evidence': True, 'proof_required_for_visual_override': True, 'uses_gold_answer': False}
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'candidate_pack_count': len(candidates), 'guarded_answerable_candidate_count': sum((bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_rejected_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) for item in candidates)), 'guarded_identity_support_candidate_count': len(identity_candidates), 'guarded_unbound_identity_support_candidate_count': sum((not bool(item.get('graph_bound')) for item in identity_candidates)), 'open_identity_component_candidate_count': sum((bool(item.get('identity_component_candidate')) for item in candidates))})
    visual_evidence.update({'contract': VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT, 'available': bool(safe_lines), 'candidate_packs': candidates, 'identity_support_candidates': identity_candidates, 'evidence_lines': safe_lines, 'visual_override_contract': override_contract, 'stats': stats, 'guard_note': 'Visual candidates were retained for audit but none satisfied the v4d identity or route-specific proof.' if not safe_lines else ''})
    bundle['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT
    bundle['visual_evidence'] = visual_evidence
    bundle['visual_override_contract'] = override_contract
    bundle['visual_entity_candidate_packs'] = {'contract': 'openeqa_guarded_visual_candidate_packs.v4d', 'packs': candidates, 'identity_support_candidates': identity_candidates, 'verified_spatial_pairs': verified_pairs, 'count_candidate_set': count_candidates, 'audit_only_candidate_count': sum((not bool(item.get('proof', {}).get('can_override_graph')) and (not bool(item.get('proof', {}).get('identity_observation', {}).get('can_support_object_identity'))) for item in candidates)), 'uses_gold_answer': False}
    bundle['visual_entity_candidate_pack_lines'] = safe_lines
    structured = dict(bundle.get('structured_graph_context') or {})
    structured['visual_evidence_contract'] = VISUAL_EVIDENCE_GUARDED_V4D_CONTRACT
    structured['visual_evidence'] = visual_evidence
    structured['visual_override_contract'] = override_contract
    bundle['structured_graph_context'] = structured
    updated_result['llm_ready_bundle'] = bundle
    updated_result['visual_evidence'] = visual_evidence
    return updated_result

def _guarded_spatial_pair(pair: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph) -> dict[str, Any]:
    target_id = str(pair.get('target_entity_id') or '')
    anchor_id = str(pair.get('anchor_entity_id') or '')
    canonical_support = _canonical_relation_support(graph, pair)
    relation_verified = bool(target_id and anchor_id and canonical_support)
    pair['proof'] = {'target_bound': bool(target_id), 'owner_bound': bool(target_id), 'attribute_dimension': None, 'anchor_bound': bool(anchor_id), 'relation_verified': relation_verified, 'state_bearing': False, 'negative_coverage': False, 'can_override_graph': relation_verified}
    pair['canonical_relation_support'] = canonical_support
    return pair

def _canonical_relation_support(graph: CanonicalTemporalMemoryGraph, pair: dict[str, Any]) -> list[dict[str, Any]]:
    target_id = str(pair.get('target_entity_id') or '')
    anchor_ids = [str(pair.get('anchor_entity_id') or '')]
    anchor_ids.extend((str(value) for value in pair.get('anchor_entity_ids', []) or []))
    anchor_ids = [value for value in _ordered_unique(anchor_ids) if value]
    relation = _normalize_relation(str(pair.get('relation') or ''))
    if not target_id or not anchor_ids or (not relation):
        return []
    support: list[dict[str, Any]] = []
    for episode in graph.episode_nodes:
        if episode.episode_kind != 'RELATION':
            continue
        subject_id = str(episode.subject_entity_id or '')
        object_id = str(episode.object_entity_id or '')
        predicate = _normalize_relation(str(episode.raw_predicate or episode.predicate or ''))
        direct = subject_id == target_id and object_id in anchor_ids and _relation_matches_any(predicate, {relation})
        reverse = object_id == target_id and subject_id in anchor_ids and _relation_matches_any(_inverse_relation(predicate), {relation})
        if not direct and (not reverse):
            continue
        support.append({'episode_id': episode.episode_id, 'subject_entity_id': subject_id, 'predicate': predicate, 'object_entity_id': object_id, 'direction': 'direct' if direct else 'inverse'})
    return support[:4]

def _candidate_proof(pack: dict[str, Any], *, evidence_demand: dict[str, Any], verified_target_ids: set[str], verified_anchor_ids: set[str]) -> dict[str, Any]:
    need = str(evidence_demand.get('primary_need') or '')
    entity_id = str(pack.get('canonical_entity_id') or '')
    role = str(pack.get('candidate_role') or '')
    graph_bound = bool(pack.get('graph_bound') and entity_id)
    requires_spatial_pair = _requires_guarded_spatial_pair(evidence_demand)
    owner_matches = _owner_matches_target(pack, evidence_demand=evidence_demand)
    rank = int(pack.get('guard_candidate_rank') or 999)
    answer_shape = str(evidence_demand.get('answer_shape') or '')
    semantic_open_target = bool(rank <= 3 and (need == NEED_FUNCTIONAL_WORLD and answer_shape != 'binary' or (need == NEED_OBJECT_IDENTITY and (not requires_spatial_pair)) or need == NEED_OPEN_DESCRIPTIVE))
    target_bound = bool(graph_bound and role in {'target', 'spatial_target'} and (owner_matches or requires_spatial_pair or semantic_open_target))
    owner_bound = bool(target_bound and owner_matches)
    anchor_bound = bool(entity_id in verified_anchor_ids or (graph_bound and role == 'anchor'))
    relation_verified = bool(entity_id in verified_target_ids)
    dimension_contract = _attribute_dimension_contract(pack, evidence_demand=evidence_demand, owner_bound=owner_bound)
    state_bearing = _state_bearing(pack, evidence_demand=evidence_demand, owner_bound=owner_bound)
    surface_capacity = bool((evidence_demand.get('metadata') or {}).get('surface_capacity_query'))
    if surface_capacity:
        can_override = False
    elif requires_spatial_pair:
        can_override = bool(target_bound and relation_verified)
    elif need == NEED_SPATIAL_LOCALIZATION:
        can_override = owner_bound
    elif need == NEED_ENTITY_ATTRIBUTE:
        can_override = bool(owner_bound and dimension_contract['matched'] and (not dimension_contract['conflicting_values']))
    elif need == NEED_EXISTENCE_OR_STATE:
        state_requested = bool(evidence_demand.get('state_terms'))
        can_override = bool(owner_bound and (state_bearing if state_requested else True))
    elif need == NEED_COUNT_SET:
        can_override = target_bound
    elif need == NEED_OCR_LOGO_TEXT:
        can_override = bool(owner_bound and dimension_contract['matched'])
    elif need == NEED_FUNCTIONAL_WORLD:
        can_override = bool(target_bound and answer_shape != 'binary')
    elif need in {NEED_OBJECT_IDENTITY, NEED_OPEN_DESCRIPTIVE}:
        can_override = target_bound
    else:
        can_override = False
    return {'target_bound': target_bound, 'owner_bound': owner_bound, 'attribute_dimension': dimension_contract, 'anchor_bound': anchor_bound, 'relation_verified': relation_verified, 'state_bearing': state_bearing, 'negative_coverage': False, 'can_override_graph': can_override}

def _candidate_proof_v4c(pack: dict[str, Any], *, evidence_demand: dict[str, Any]) -> dict[str, Any]:
    proof = dict(pack.get('proof') or {})
    dimension = _dominant_attribute_dimension_contract(pack, contract=dict(proof.get('attribute_dimension') or {}))
    proof['attribute_dimension'] = dimension
    need = str(evidence_demand.get('primary_need') or '')
    answer_shape = str(evidence_demand.get('answer_shape') or '')
    role = str(pack.get('candidate_role') or '')
    rank = int(pack.get('guard_candidate_rank') or 999)
    graph_bound = bool(pack.get('graph_bound') and pack.get('canonical_entity_id'))
    requires_spatial_pair = _requires_guarded_spatial_pair(evidence_demand)
    strict_owner_match = _owner_matches_target(pack, evidence_demand=evidence_demand)
    descriptive_owner_match = _descriptive_owner_matches_target(pack, evidence_demand=evidence_demand)
    surface_capacity = bool((evidence_demand.get('metadata') or {}).get('surface_capacity_query'))
    if need == NEED_ENTITY_ATTRIBUTE and graph_bound and (role in {'target', 'spatial_target'}) and (strict_owner_match or descriptive_owner_match) and dimension.get('answer_values'):
        proof['target_bound'] = True
        proof['owner_bound'] = True
        proof['can_override_graph'] = True
    elif need == NEED_SPATIAL_LOCALIZATION and (not requires_spatial_pair) and (answer_shape == 'location_text') and graph_bound and (role in {'target', 'spatial_target'}) and (strict_owner_match or descriptive_owner_match) and pack.get('direct_relations'):
        proof['target_bound'] = True
        proof['owner_bound'] = True
        proof['can_override_graph'] = True
        proof['location_relation_bearing'] = True
    elif need == NEED_FUNCTIONAL_WORLD and (not surface_capacity):
        if answer_shape != 'binary' and graph_bound and (role in {'target', 'spatial_target'}) and (rank <= 5):
            proof['target_bound'] = True
            proof['owner_bound'] = bool(strict_owner_match)
            proof['can_override_graph'] = True
            proof['functional_identity_only'] = True
        elif answer_shape == 'binary' and graph_bound and (role in {'target', 'spatial_target'}) and strict_owner_match and (not evidence_demand.get('anchor_terms')) and (not evidence_demand.get('state_terms')):
            proof['target_bound'] = True
            proof['owner_bound'] = True
            proof['can_override_graph'] = True
            proof['functional_identity_only'] = True
    return proof

def _descriptive_owner_matches_target(pack: dict[str, Any], *, evidence_demand: dict[str, Any]) -> bool:
    if _owner_matches_target(pack, evidence_demand=evidence_demand):
        return True
    target_terms = _flatten_text(evidence_demand.get('target_terms', []))
    if not target_terms:
        return False
    label_tokens = _normalized_owner_tokens([pack.get('entity_label'), pack.get('graph_label'), pack.get('visual_label')])
    attribute_tokens = _normalized_owner_tokens(pack.get('raw_visual_attributes', []))
    if not label_tokens:
        return False
    for target in target_terms:
        target_tokens = _normalized_owner_tokens([target])
        requested_components = target_tokens & _COMPONENT_OWNER_TOKENS
        if requested_components and (not requested_components <= label_tokens):
            continue
        shared = target_tokens & label_tokens
        if not shared:
            continue
        modifiers = target_tokens - label_tokens - {'object', 'item', 'thing'}
        if not modifiers or modifiers <= attribute_tokens:
            return True
    return False

def _dominant_attribute_dimension_contract(pack: dict[str, Any], *, contract: dict[str, Any]) -> dict[str, Any]:
    requested = list(contract.get('requested') or [])
    values = list(contract.get('values') or [])
    answer_values = list(values) if len(values) == 1 else []
    structured_values = _ordered_unique((str(state.get('value') or '').strip().lower() for state in pack.get('owner_bound_states', []) or [] if isinstance(state, dict) and (not requested or set(requested) & _term_tokens([state.get('attribute')])) and str(state.get('value') or '').strip()))
    if len(structured_values) == 1:
        answer_values = structured_values
    counts: Counter[str] = Counter()
    if len(requested) == 1 and (not structured_values):
        vocabulary = _ATTRIBUTE_VALUE_TOKENS.get(str(requested[0]))
        if vocabulary:
            for text in _flatten_text(pack.get('raw_visual_attributes', [])):
                tokens = _TOKEN_PATTERN.findall(text.lower())
                primary = next((token for token in tokens if token in vocabulary), '')
                if primary:
                    counts[_normalize_attribute_value(primary)] += 1
            first_text = next(iter(_flatten_text(pack.get('raw_visual_attributes', []))), '')
            first_value = next((_normalize_attribute_value(token) for token in _TOKEN_PATTERN.findall(first_text.lower()) if token in vocabulary), '')
            if first_value and len(counts) > 1:
                counts[first_value] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not answer_values and ranked:
        top_value, top_count = ranked[0]
        next_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count >= 2 and top_count > next_count:
            answer_values = [top_value]
    contract.update({'primary_value_counts': dict(ranked), 'dominant_value': answer_values[0] if len(answer_values) == 1 else None, 'answer_values': answer_values, 'dominant_value_resolved': bool(answer_values)})
    return contract

def _normalize_attribute_value(value: str) -> str:
    return 'wood' if value == 'wooden' else 'gray' if value == 'grey' else value

def _attach_identity_observation_proofs(candidates: list[dict[str, Any]], *, evidence_demand: dict[str, Any]) -> list[dict[str, Any]]:
    if str(evidence_demand.get('primary_need') or '') not in {NEED_OBJECT_IDENTITY, NEED_OPEN_DESCRIPTIVE}:
        return []
    if (evidence_demand.get('metadata') or {}).get('surface_capacity_query'):
        return []
    label_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pack in candidates:
        key = _identity_label_key(pack)
        if key:
            label_groups[key].append(pack)
    requires_spatial_pair = _requires_guarded_spatial_pair(evidence_demand)
    supported: list[dict[str, Any]] = []
    for pack in candidates:
        rank = int(pack.get('guard_candidate_rank') or 999)
        role = str(pack.get('candidate_role') or '')
        matched_roles = set(pack.get('matched_roles') or [])
        graph_bound = bool(pack.get('graph_bound') and pack.get('canonical_entity_id'))
        key = _identity_label_key(pack)
        peers = label_groups.get(key, [])
        frames = {int(anchor.get('frame_index')) for peer in peers for anchor in peer.get('visual_anchors', []) or [] if isinstance(anchor, dict) and isinstance(anchor.get('frame_index'), int)}
        graph_bound_same_label = any((bool(peer.get('graph_bound') and peer.get('canonical_entity_id')) for peer in peers))
        target_match = _descriptive_owner_matches_target(pack, evidence_demand=evidence_demand)
        spatial_target = role == 'spatial_target' or 'spatial_target' in matched_roles
        eligible = bool(rank <= 5 and role in {'target', 'spatial_target'} and (graph_bound and (spatial_target if requires_spatial_pair else True) or (not graph_bound and target_match and (len(frames) >= 2 or graph_bound_same_label))))
        identity_proof = {'eligible': eligible, 'canonical_entity_bound': graph_bound, 'visual_observation_is_graph_fact': False, 'target_term_match': target_match, 'independent_frame_count': len(frames), 'same_label_candidate_count': len(peers), 'graph_bound_same_label_support': graph_bound_same_label, 'can_support_object_identity': eligible, 'can_prove_spatial_relation': False, 'can_prove_state': False, 'can_prove_absence': False, 'can_prove_owner_attribute': False}
        proof = dict(pack.get('proof') or {})
        proof['identity_observation'] = identity_proof
        pack['proof'] = proof
        if eligible:
            supported.append(pack)
    return supported

def _identity_label_key(pack: dict[str, Any]) -> str:
    tokens = _normalized_owner_tokens([pack.get('entity_label'), pack.get('graph_label'), pack.get('visual_label')])
    return ' '.join(sorted(tokens))

def _rerank_open_identity_candidates(candidates: list[dict[str, Any]], *, evidence_demand: dict[str, Any]) -> list[dict[str, Any]]:
    if not evidence_demand.get('open_target_slot'):
        return candidates
    state_tokens = _identity_attribute_constraint_tokens(evidence_demand.get('state_terms', []))
    constraint_tokens = _identity_attribute_constraint_tokens(evidence_demand.get('constraint_terms', []))
    anchor_tokens = _identity_attribute_constraint_tokens(evidence_demand.get('anchor_terms', []))
    constraint_tokens -= anchor_tokens
    attribute_tokens = state_tokens | constraint_tokens
    if not attribute_tokens or not any((_identity_component_penalty(pack) > 0 for pack in candidates)):
        return candidates
    reranked: list[dict[str, Any]] = []
    for original_rank, source in enumerate(candidates, start=1):
        pack = dict(source)
        candidate_tokens = _identity_candidate_evidence_tokens(pack)
        state_overlap = len(state_tokens & candidate_tokens)
        constraint_overlap = len(constraint_tokens & candidate_tokens)
        component_penalty = _identity_component_penalty(pack)
        selection_score = 4 * state_overlap + constraint_overlap - component_penalty
        pack.update({'identity_original_rank': int(pack.get('guard_candidate_rank') or original_rank), 'identity_state_constraint_overlap': state_overlap, 'identity_other_constraint_overlap': constraint_overlap, 'identity_component_penalty': component_penalty, 'identity_component_candidate': component_penalty > 0, 'identity_complete_object_preferred': bool(component_penalty == 0 and (state_overlap or constraint_overlap)), 'identity_selection_score': selection_score})
        reranked.append(pack)
    reranked.sort(key=lambda pack: (-int(pack.get('identity_selection_score') or 0), -int(bool(pack.get('identity_complete_object_preferred'))), -int(bool(pack.get('graph_bound'))), int(pack.get('identity_original_rank') or 999), str(pack.get('candidate_id') or '')))
    return reranked

def _normalized_identity_constraint_tokens(values: Any) -> set[str]:
    stop = {'at', 'by', 'for', 'from', 'in', 'inside', 'into', 'of', 'on', 'to', 'with'}
    return {_normalize_attribute_value(token) for token in _normalized_owner_tokens(values) if token not in stop}

def _identity_attribute_constraint_tokens(values: Any) -> set[str]:
    vocabulary = {_normalize_attribute_value(token) for tokens in _ATTRIBUTE_VALUE_TOKENS.values() for token in tokens}
    return _normalized_identity_constraint_tokens(values) & vocabulary

def _identity_candidate_evidence_tokens(pack: dict[str, Any]) -> set[str]:
    values: list[Any] = [pack.get('entity_label'), pack.get('graph_label'), pack.get('visual_label'), *list(_flatten_text(pack.get('raw_visual_attributes', [])))]
    for relation in pack.get('direct_relations', []) or []:
        if not isinstance(relation, dict):
            continue
        values.extend([relation.get('subject_label'), relation.get('predicate'), relation.get('object_label')])
    return {_normalize_attribute_value(token) for token in _normalized_owner_tokens(values)}

def _identity_component_penalty(pack: dict[str, Any]) -> int:
    label_tokens = _normalized_owner_tokens([pack.get('entity_label'), pack.get('graph_label'), pack.get('visual_label')])
    if label_tokens & _STRONG_COMPONENT_LABEL_TOKENS:
        return 6
    relation_marks_component = any((_normalize_relation(str(relation.get('predicate') or '')) in _COMPONENT_RELATIONS for relation in pack.get('direct_relations', []) or [] if isinstance(relation, dict)))
    if relation_marks_component and label_tokens & _WEAK_COMPONENT_LABEL_TOKENS:
        return 3
    return 0

def _owner_matches_target(pack: dict[str, Any], *, evidence_demand: dict[str, Any]) -> bool:
    target_terms = _flatten_text(evidence_demand.get('target_terms', []))
    if not target_terms:
        return True
    label_tokens = _normalized_owner_tokens([pack.get('entity_label'), pack.get('graph_label'), pack.get('visual_label')])
    requested_tokens = _normalized_owner_tokens(target_terms)
    requested_components = requested_tokens & _COMPONENT_OWNER_TOKENS
    if requested_components and (not requested_components <= label_tokens):
        return False
    for target in target_terms:
        target_tokens = _normalized_owner_tokens([target])
        if target_tokens and target_tokens <= label_tokens:
            return True
    return False

def _normalized_owner_tokens(values: Any) -> set[str]:
    tokens = _term_tokens(values)
    normalized: set[str] = set()
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            token = token[:-3] + 'y'
        elif len(token) > 3 and token.endswith('s') and (not token.endswith('ss')):
            token = token[:-1]
        if token not in {'a', 'an', 'the', 'object', 'thing', 'item'}:
            normalized.add(token)
    return normalized

def _attribute_dimension_contract(pack: dict[str, Any], *, evidence_demand: dict[str, Any], owner_bound: bool) -> dict[str, Any]:
    requested = sorted(_term_tokens(evidence_demand.get('attribute_dimensions', [])))
    matched_states: list[dict[str, Any]] = []
    for state in pack.get('owner_bound_states', []) or []:
        if not isinstance(state, dict):
            continue
        attribute_tokens = _term_tokens([state.get('attribute')])
        if requested and (not set(requested) & attribute_tokens):
            continue
        if requested or attribute_tokens:
            matched_states.append(state)
    structured_values = _ordered_unique((str(item.get('value') or '').strip().lower() for item in matched_states if str(item.get('value') or '').strip()))
    raw_values = _raw_attribute_dimension_values(pack.get('raw_visual_attributes', []), requested=requested)
    values = _ordered_unique([*structured_values, *raw_values])
    return {'requested': requested, 'matched': bool(owner_bound and values), 'matched_state_count': len(matched_states), 'matched_raw_attribute_count': len(raw_values), 'values': values, 'conflicting_values': len(values) > 1}

def _raw_attribute_dimension_values(attributes: Any, *, requested: list[str]) -> list[str]:
    if len(requested) != 1:
        return []
    dimension = requested[0]
    vocabulary = _ATTRIBUTE_VALUE_TOKENS.get(dimension)
    if not vocabulary:
        return []
    values: list[str] = []
    for text in _flatten_text(attributes):
        values.extend((token for token in _TOKEN_PATTERN.findall(text.lower()) if token in vocabulary))
    normalized = ['wood' if value == 'wooden' else 'gray' if value == 'grey' else value for value in values]
    return _ordered_unique(normalized)

def _state_bearing(pack: dict[str, Any], *, evidence_demand: dict[str, Any], owner_bound: bool) -> bool:
    if not owner_bound:
        return False
    requested = _term_tokens(evidence_demand.get('state_terms', []))
    if not requested:
        return False
    for state in pack.get('owner_bound_states', []) or []:
        if not isinstance(state, dict):
            continue
        state_tokens = _term_tokens([state.get('attribute'), state.get('value')])
        if requested & state_tokens:
            return True
    return False

def _guarded_visual_evidence_lines(candidates: list[dict[str, Any]], *, verified_pairs: list[dict[str, Any]], evidence_demand: dict[str, Any], count_candidates: list[dict[str, Any]], visual_payload: dict[str, Any], limit: int) -> list[str]:
    need = str(evidence_demand.get('primary_need') or '')
    surface_capacity = bool((evidence_demand.get('metadata') or {}).get('surface_capacity_query'))
    if surface_capacity:
        return []
    lines: list[str] = []
    for pair in verified_pairs:
        lines.append(f"kind=guarded_visual_spatial_pair frame={pair.get('frame_index')} target={pair.get('target_label')!r} target_entity={pair.get('target_entity_id')} relation={pair.get('relation')} anchor={pair.get('anchor_label')!r} anchor_entity={pair.get('anchor_entity_id')} target_bound=true anchor_bound=true relation_verified=canonical_direct_relation can_override_graph=true")
    if need == NEED_COUNT_SET:
        safe_count = [item for item in count_candidates if _owner_matches_target(item, evidence_demand=evidence_demand)]
        examples = ', '.join((f"{item.get('entity_label')}({item.get('canonical_entity_id')})" for item in safe_count[:8])) or 'none'
        lines.append(f'kind=guarded_visual_count_candidate_set unique_graph_bound_target_candidates={len(safe_count)} entities={examples} owner_bound=true exact_count_not_guaranteed=true')
    answerable = [item for item in candidates if item.get('proof', {}).get('can_override_graph')]
    if need == NEED_EXISTENCE_OR_STATE and answerable:
        lines.append(f"kind=guarded_visual_episode_coverage episode_candidates={int(visual_payload.get('episode_candidate_count') or 0)} absence_proven=false no_match_does_not_prove_absence=true crop_presence_does_not_prove_state=true")
    requires_spatial_pair = _requires_guarded_spatial_pair(evidence_demand)
    if requires_spatial_pair:
        return _ordered_unique(lines)[:limit]
    if need == NEED_FUNCTIONAL_WORLD:
        answerable = answerable[:3]
    for index, pack in enumerate(answerable, start=1):
        proof = pack['proof']
        if need == NEED_ENTITY_ATTRIBUTE:
            dimension = proof['attribute_dimension']
            values = dimension.get('answer_values') or dimension['values']
            facts = f"owner_bound_{','.join(dimension['requested']) or 'attribute'}=" + ' | '.join(values)
        elif need == NEED_EXISTENCE_OR_STATE and evidence_demand.get('state_terms'):
            requested = _term_tokens(evidence_demand.get('state_terms', []))
            matched = [row for row in pack.get('owner_bound_states', []) or [] if requested & _term_tokens([row.get('attribute'), row.get('value')])]
            facts = 'owner_bound_states=' + ' | '.join((f"{row.get('attribute')}={row.get('value')}" for row in matched[:3]))
        elif need == NEED_FUNCTIONAL_WORLD:
            facts = 'observed_graph_bound_object_identity_for_common_world_reasoning'
        elif need in {NEED_OBJECT_IDENTITY, NEED_SPATIAL_LOCALIZATION}:
            facts = 'observed_graph_bound_object_identity'
        else:
            facts = _pack_fact_text(pack, need=need)
        lines.append(f"kind=guarded_graph_bound_visual_candidate rank={index} role={pack.get('candidate_role')} label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id')} target_bound={str(proof['target_bound']).lower()} owner_bound={str(proof['owner_bound']).lower()} state_bearing={str(proof['state_bearing']).lower()} can_override_graph=true :: {facts}")
        if len(lines) >= limit:
            break
    return _ordered_unique(lines)[:limit]

def _guarded_visual_evidence_lines_v4c(candidates: list[dict[str, Any]], *, verified_pairs: list[dict[str, Any]], evidence_demand: dict[str, Any], count_candidates: list[dict[str, Any]], visual_payload: dict[str, Any], identity_candidates: list[dict[str, Any]], limit: int) -> list[str]:
    need = str(evidence_demand.get('primary_need') or '')
    lines = _guarded_visual_evidence_lines(candidates, verified_pairs=verified_pairs, evidence_demand=evidence_demand, count_candidates=count_candidates, visual_payload=visual_payload, limit=limit)
    if need in {NEED_OBJECT_IDENTITY, NEED_OPEN_DESCRIPTIVE}:
        lines = [line for line in lines if not line.startswith('kind=guarded_graph_bound_visual_candidate')]
        lines.extend(_identity_support_lines(identity_candidates))
    elif need == NEED_SPATIAL_LOCALIZATION and (not _requires_guarded_spatial_pair(evidence_demand)):
        lines = [line for line in lines if not line.startswith('kind=guarded_graph_bound_visual_candidate')]
        lines.extend(_location_support_lines(candidates))
    elif need == NEED_FUNCTIONAL_WORLD:
        lines = [line for line in lines if not line.startswith('kind=guarded_graph_bound_visual_candidate')]
        lines.extend(_functional_identity_lines(candidates))
    return _ordered_unique(lines)[:limit]

def _identity_support_lines(candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, pack in enumerate(candidates[:5], start=1):
        proof = pack.get('proof', {})
        identity = proof.get('identity_observation', {})
        descriptions = _ordered_unique(pack.get('raw_visual_attributes', []))[:2]
        description_text = ' | '.join(descriptions) or 'none'
        v4d_selection = ''
        if 'identity_selection_score' in pack:
            v4d_selection = f"component_candidate={str(bool(pack.get('identity_component_candidate'))).lower()}  complete_object_preferred={str(bool(pack.get('identity_complete_object_preferred'))).lower()} constraint_score={int(pack.get('identity_selection_score') or 0)} "
        lines.append(f"kind=guarded_visual_identity_candidate rank={index} label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id') or 'unbound_visual_observation'} canonical_entity_bound={str(bool(identity.get('canonical_entity_bound'))).lower()} visual_observation_is_graph_fact=false can_support_object_identity=true can_prove_spatial_relation=false relation_verified={str(bool(proof.get('relation_verified'))).lower()} can_override_graph={str(bool(proof.get('can_override_graph'))).lower()} independent_frames={int(identity.get('independent_frame_count') or 0)} {v4d_selection}:: candidate_descriptions={description_text}")
    return lines

def _location_support_lines(candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for pack in candidates:
        proof = pack.get('proof', {})
        if not (proof.get('can_override_graph') and proof.get('location_relation_bearing')):
            continue
        relations = [row for row in pack.get('direct_relations', []) or [] if isinstance(row, dict)]
        if not relations:
            continue
        relation_text = ' | '.join((f"{row.get('subject_label')} {row.get('predicate')} {row.get('object_label')}" for row in relations[:4]))
        lines.append(f"kind=guarded_graph_bound_location_candidate label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id')} owner_bound=true location_relation_bearing=true can_override_graph=true :: direct_canonical_relations={relation_text}")
    return lines

def _functional_identity_lines(candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for pack in candidates:
        proof = pack.get('proof', {})
        if not (proof.get('can_override_graph') and proof.get('functional_identity_only')):
            continue
        descriptions = _ordered_unique(pack.get('raw_visual_attributes', []))[:2]
        relations = [row for row in pack.get('direct_relations', []) or [] if isinstance(row, dict)][:2]
        facts: list[str] = []
        if descriptions:
            facts.append('descriptions=' + ' | '.join(descriptions))
        if relations:
            facts.append('relations=' + ' | '.join((f"{row.get('subject_label')} {row.get('predicate')} {row.get('object_label')}" for row in relations)))
        lines.append(f"kind=guarded_functional_identity_candidate label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id')} target_bound=true identity_grounded=true state_or_negative_not_proven=true can_override_graph=true :: " + ('; '.join(facts) or 'graph_bound_object_identity'))
    return lines

def _normalize_relation(value: str) -> str:
    normalized = '_'.join(_TOKEN_PATTERN.findall(str(value or '').lower()))
    aliases = {'left': 'left_of', 'right': 'right_of', 'in_front': 'in_front_of', 'front_of': 'in_front_of', 'under': 'below', 'underneath': 'below', 'beneath': 'below', 'beside': 'near', 'next_to': 'near'}
    return aliases.get(normalized, normalized)

def _requires_guarded_spatial_pair(evidence_demand: dict[str, Any]) -> bool:
    if not evidence_demand.get('anchor_terms'):
        return False
    relations = {_normalize_relation(value) for value in _flatten_text(evidence_demand.get('spatial_relations', []))} - {'', 'where'}
    return bool(relations)

def _inverse_relation(value: str) -> str:
    relation = _normalize_relation(value)
    return {'left_of': 'right_of', 'right_of': 'left_of', 'above': 'below', 'below': 'above', 'in_front_of': 'behind', 'behind': 'in_front_of', 'inside': 'contains', 'contains': 'inside'}.get(relation, relation)

def _candidate_packs(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], spatial_pairs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    entity_by_id = {node.entity_id: node for node in graph.entity_nodes}
    state_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels = {node.entity_id: node.label for node in graph.entity_nodes}
    for episode in graph.episode_nodes:
        if episode.episode_kind == 'STATE' and episode.primary_entity_id:
            state_by_entity[episode.primary_entity_id].append({'owner_entity_id': episode.primary_entity_id, 'attribute': episode.attr_key, 'value': episode.attr_value, 'start_frame': episode.time_range.start_frame, 'end_frame': episode.time_range.end_frame})
        elif episode.episode_kind == 'RELATION':
            fact = {'subject_entity_id': episode.subject_entity_id, 'subject_label': labels.get(str(episode.subject_entity_id or ''), ''), 'predicate': episode.raw_predicate or episode.predicate, 'object_entity_id': episode.object_entity_id, 'object_label': labels.get(str(episode.object_entity_id or ''), ''), 'start_frame': episode.time_range.start_frame, 'end_frame': episode.time_range.end_frame}
            for entity_id in (episode.subject_entity_id, episode.object_entity_id):
                if entity_id:
                    relation_by_entity[entity_id].append(fact)
    grouped: dict[str, dict[str, Any]] = {}
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent') if isinstance(group.get('query_intent'), dict) else {}
        role = str(intent.get('role') or 'context')
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            entity_id = str(hit.get('canonical_entity_id') or '').strip()
            observation_id = str(hit.get('observation_id') or '').strip()
            key = entity_id or f"visual:{observation_id or hit.get('faiss_id')}"
            pack = grouped.setdefault(key, {'candidate_id': key, 'canonical_entity_id': entity_id, 'graph_bound': bool(entity_id), 'binding_confidence': float(hit.get('binding_confidence') or (1.0 if entity_id else 0.45)), 'binding_kind': str(hit.get('binding_kind') or ''), 'candidate_role': role, 'matched_roles': set(), 'visual_label': str(hit.get('label') or ''), 'visual_anchors': [], 'matched_query_intents': []})
            if _ROLE_PRIORITY.get(role, 0) > _ROLE_PRIORITY.get(str(pack['candidate_role']), 0):
                pack['candidate_role'] = role
            pack['matched_roles'].add(role)
            pack['visual_anchors'].append(_compact_hit(hit, role=role))
            pack['matched_query_intents'].append({'text': str(intent.get('text') or ''), 'role': role, 'source': str(intent.get('source') or '')})
    target_tokens = _term_tokens(evidence_demand.get('target_terms', []))
    attribute_tokens = _term_tokens(evidence_demand.get('attribute_dimensions', []))
    need = str(evidence_demand.get('primary_need') or '')
    open_target_slot = bool(evidence_demand.get('open_target_slot'))
    pair_support_by_entity = Counter((str(pair.get('target_entity_id') or '') for pair in spatial_pairs if pair.get('target_entity_id')))
    packs: list[dict[str, Any]] = []
    for pack in grouped.values():
        entity_id = str(pack.get('canonical_entity_id') or '')
        entity = entity_by_id.get(entity_id)
        graph_label = entity.label if entity is not None else ''
        raw_attributes = _entity_attributes(entity) if entity is not None else []
        states = state_by_entity.get(entity_id, [])[:6]
        relations = relation_by_entity.get(entity_id, [])[:8]
        anchors = pack.get('visual_anchors', [])
        max_similarity = max((float(item.get('similarity') or 0.0) for item in anchors), default=0.0)
        matched_roles = set(pack.pop('matched_roles', set()))
        if open_target_slot and 'anchor' in matched_roles and ('spatial_target' not in matched_roles):
            pack['candidate_role'] = 'anchor'
        role_score = _ROLE_PRIORITY.get(str(pack.get('candidate_role') or ''), 0)
        label_tokens = _term_tokens([graph_label, pack.get('visual_label')])
        target_overlap = len(target_tokens & label_tokens)
        attribute_support = sum((1 for value in raw_attributes if not attribute_tokens or attribute_tokens & set(_TOKEN_PATTERN.findall(value.lower()))))
        answerability = _answerability_score(need, raw_attributes=raw_attributes, states=states, relations=relations)
        spatial_pair_support = int(pair_support_by_entity.get(entity_id, 0))
        pack.update({'entity_label': graph_label or pack.get('visual_label') or 'unknown', 'graph_label': graph_label, 'raw_visual_attributes': raw_attributes[:8], 'owner_bound_states': states, 'direct_relations': relations, 'max_visual_similarity': round(max_similarity, 6), 'target_label_overlap': target_overlap, 'requested_attribute_support': attribute_support, 'answerability_score': answerability, 'spatial_pair_support': spatial_pair_support, 'matched_roles': sorted(matched_roles), 'ranking_score': round(max_similarity + 0.18 * role_score + 0.2 * float(pack.get('binding_confidence') or 0.0) + 0.12 * min(2, target_overlap) + 0.04 * answerability + 0.12 * min(2, spatial_pair_support), 6), 'uses_gold_answer': False, 'unbound_visual_observation_is_graph_fact': False})
        pack['visual_anchors'] = sorted(anchors, key=lambda item: (-float(item.get('similarity') or 0.0), int(item.get('frame_index') or 0)))[:6]
        pack['matched_query_intents'] = _unique_dicts(pack.get('matched_query_intents', []), keys=('text', 'role'))
        packs.append(pack)
    packs.sort(key=lambda pack: (-float(pack.get('ranking_score') or 0.0), -float(pack.get('binding_confidence') or 0.0), str(pack.get('candidate_id') or '')))
    return packs[:limit]

def _spatial_pair_evidence(visual_payload: dict[str, Any], *, evidence_demand: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    need = str(evidence_demand.get('primary_need') or '')
    if need != NEED_SPATIAL_LOCALIZATION and (not evidence_demand.get('anchor_terms')):
        return []
    targets: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent') if isinstance(group.get('query_intent'), dict) else {}
        role = str(intent.get('role') or 'context')
        if role not in {'target', 'spatial_target', 'anchor'}:
            continue
        for hit in group.get('hits', []) or []:
            if isinstance(hit, dict):
                enriched = dict(hit)
                enriched['retrieval_role'] = role
                enriched['intent_text'] = str(intent.get('text') or '')
                enriched['requested_relation'] = str(intent.get('spatial_relation') or '')
                (anchors if role == 'anchor' else targets).append(enriched)
    pairs: list[dict[str, Any]] = []
    requested_relations = set(_flatten_text(evidence_demand.get('spatial_relations', [])))
    if 'between' in requested_relations:
        pairs.extend(_between_spatial_pairs(targets, anchors))
    for target in targets:
        for anchor in anchors:
            if target.get('graph_frame_index') != anchor.get('graph_frame_index'):
                continue
            if _same_visual_object(target, anchor):
                continue
            relation = _bbox_relation(target.get('bbox'), anchor.get('bbox'))
            if not relation:
                continue
            if 'between' in requested_relations:
                continue
            anchor_relations = {str(anchor.get('requested_relation') or '')} - {''}
            applicable_relations = anchor_relations or requested_relations
            if applicable_relations and (not _relation_matches_any(relation, applicable_relations)):
                continue
            score = min(float(target.get('similarity') or 0.0), float(anchor.get('similarity') or 0.0))
            score += 0.1 if target.get('canonical_entity_id') else 0.0
            score += 0.1 if anchor.get('canonical_entity_id') else 0.0
            pairs.append({'frame_index': target.get('graph_frame_index'), 'scene_id': target.get('scene_id') or anchor.get('scene_id'), 'target_label': target.get('label'), 'target_entity_id': target.get('canonical_entity_id') or '', 'relation': relation, 'anchor_label': anchor.get('label'), 'anchor_entity_id': anchor.get('canonical_entity_id') or '', 'target_bbox': target.get('bbox'), 'anchor_bbox': anchor.get('bbox'), 'pair_score': round(score, 6), 'evidence_kind': 'same_frame_visual_geometry', 'uses_gold_answer': False})
    pairs.sort(key=lambda row: (-float(row['pair_score']), int(row.get('frame_index') or 0)))
    return _unique_dicts(pairs, keys=('frame_index', 'target_entity_id', 'relation', 'anchor_entity_id'))[:limit]

def _visual_evidence_lines(candidates: list[dict[str, Any]], *, spatial_pairs: list[dict[str, Any]], evidence_demand: dict[str, Any], limit: int) -> list[str]:
    need = str(evidence_demand.get('primary_need') or '')
    lines: list[str] = []
    if spatial_pairs:
        for pair in spatial_pairs:
            lines.append(f"kind=visual_spatial_pair frame={pair.get('frame_index')} target={pair.get('target_label')!r} relation={pair.get('relation')} anchor={pair.get('anchor_label')!r} pair_score={float(pair.get('pair_score') or 0.0):.3f} source=same_frame_object_crop_geometry")
    if need == NEED_COUNT_SET:
        bound = [row for row in candidates if row.get('graph_bound') and row.get('candidate_role') == 'target']
        lines.append(f'kind=visual_count_candidate_set unique_bound_candidates={len(bound)} coverage=top_semantic_candidates exact_count_not_guaranteed=true')
    answer_candidates = [pack for pack in candidates if pack.get('candidate_role') != 'anchor' and (not evidence_demand.get('open_target_slot') or not evidence_demand.get('anchor_terms') or int(pack.get('spatial_pair_support') or 0) > 0)]
    for index, pack in enumerate(answer_candidates, start=1):
        facts = _pack_fact_text(pack, need=need)
        evidence_kind = 'graph_bound_visual_candidate' if pack.get('graph_bound') else 'visual_observation_candidate'
        lines.append(f"kind={evidence_kind} rank={index} role={pack.get('candidate_role')} label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id') or 'unbound_visual_observation'} similarity={float(pack.get('max_visual_similarity') or 0.0):.3f} binding={pack.get('binding_kind')} binding_confidence={float(pack.get('binding_confidence') or 0.0):.2f} :: {facts}")
        if len(lines) >= limit:
            break
    if need == NEED_EXISTENCE_OR_STATE and (not candidates):
        lines.append('kind=visual_coverage result=no_positive_candidate absence_proven=false')
    return _ordered_unique(lines)[:limit]

def _visual_evidence_lines_v2(candidates: list[dict[str, Any]], *, spatial_pairs: list[dict[str, Any]], evidence_demand: dict[str, Any], count_candidates: list[dict[str, Any]], visual_payload: dict[str, Any], limit: int) -> list[str]:
    need = str(evidence_demand.get('primary_need') or '')
    lines: list[str] = []
    for pair in spatial_pairs:
        lines.append(f"kind=visual_spatial_pair frame={pair.get('frame_index')} target={pair.get('target_label')!r} relation={pair.get('relation')} anchor={pair.get('anchor_label')!r} pair_score={float(pair.get('pair_score') or 0.0):.3f} relation_verified_by=same_frame_crop_geometry")
    if need == NEED_COUNT_SET:
        examples = ', '.join((f"{item.get('entity_label')}({item.get('canonical_entity_id')})" for item in count_candidates[:8])) or 'none'
        lines.append(f'kind=visual_count_candidate_set unique_graph_bound_target_candidates={len(count_candidates)} shown={min(8, len(count_candidates))}/{len(count_candidates)} entities={examples} exact_count_not_guaranteed=true')
    if need == NEED_EXISTENCE_OR_STATE:
        bound = sum((bool(item.get('graph_bound')) for item in candidates))
        lines.append(f"kind=visual_episode_coverage episode_candidates={int(visual_payload.get('episode_candidate_count') or 0)} retrieved_candidate_packs={len(candidates)} graph_bound_packs={bound} topk_positive_candidates_do_not_prove_presence=true no_match_does_not_prove_absence=true absence_proven=false")
    open_spatial_slot = bool(evidence_demand.get('open_target_slot') and evidence_demand.get('anchor_terms'))
    answer_candidates = [pack for pack in candidates if pack.get('candidate_role') != 'anchor']
    if open_spatial_slot and spatial_pairs:
        answer_candidates = [pack for pack in answer_candidates if int(pack.get('spatial_pair_support') or 0) > 0]
    elif open_spatial_slot:
        answer_candidates = [pack for pack in answer_candidates if pack.get('candidate_role') == 'spatial_target']
    for index, pack in enumerate(answer_candidates, start=1):
        if open_spatial_slot and (not spatial_pairs):
            evidence_kind = 'visual_coframe_candidate'
            relation_suffix = ' relation_verified=false'
        else:
            evidence_kind = 'graph_bound_visual_candidate' if pack.get('graph_bound') else 'visual_observation_candidate'
            relation_suffix = ''
        facts = _pack_fact_text(pack, need=need)
        lines.append(f"kind={evidence_kind} rank={index} role={pack.get('candidate_role')} label={pack.get('entity_label')!r} entity={pack.get('canonical_entity_id') or 'unbound_visual_observation'} similarity={float(pack.get('max_visual_similarity') or 0.0):.3f} binding={pack.get('binding_kind')} binding_confidence={float(pack.get('binding_confidence') or 0.0):.2f}{relation_suffix} :: {facts}")
        if len(lines) >= limit:
            break
    return _ordered_unique(lines)[:limit]

def _count_candidate_set(candidates: list[dict[str, Any]], *, evidence_demand: dict[str, Any]) -> list[dict[str, Any]]:
    if str(evidence_demand.get('primary_need') or '') != NEED_COUNT_SET:
        return []
    target_tokens = _count_tokens(evidence_demand.get('target_terms', []))
    if not target_tokens:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pack in candidates:
        entity_id = str(pack.get('canonical_entity_id') or '')
        if not entity_id or entity_id in seen or pack.get('candidate_role') not in {'target', 'spatial_target'}:
            continue
        label_tokens = _count_tokens([pack.get('entity_label'), pack.get('visual_label')])
        if not target_tokens & label_tokens:
            continue
        seen.add(entity_id)
        rows.append(pack)
    return rows

def _count_tokens(values: Any) -> set[str]:
    tokens = _term_tokens(values)
    expanded = set(tokens)
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            expanded.add(token[:-3] + 'y')
        elif len(token) > 2 and token.endswith('s'):
            expanded.add(token[:-1])
            if len(token) > 3 and token.endswith('es'):
                expanded.add(token[:-2])
    return expanded

def _unique_visual_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for anchor in anchors:
        faiss_id = anchor.get('faiss_id')
        observation_id = anchor.get('observation_id')
        key = ('faiss', faiss_id) if faiss_id not in (None, '') else ('observation', observation_id) if observation_id not in (None, '') else ('fallback', anchor.get('frame_index'), anchor.get('object_index'), anchor.get('label'), anchor.get('canonical_entity_id'))
        if key in seen:
            continue
        seen.add(key)
        result.append(anchor)
    return result

def _pack_fact_text(pack: dict[str, Any], *, need: str) -> str:
    attributes = list(pack.get('raw_visual_attributes', []) or [])
    states = list(pack.get('owner_bound_states', []) or [])
    relations = list(pack.get('direct_relations', []) or [])
    if need in {NEED_ENTITY_ATTRIBUTE, NEED_OCR_LOGO_TEXT} and attributes:
        return 'owner_bound_visual_attributes=' + ' | '.join(attributes[:3])
    if need == NEED_EXISTENCE_OR_STATE and states:
        return 'owner_bound_states=' + ' | '.join((f"{row.get('attribute')}={row.get('value')}" for row in states[:3]))
    if need == NEED_SPATIAL_LOCALIZATION and relations:
        return 'direct_graph_relations=' + ' | '.join((f"{row.get('subject_label')} {row.get('predicate')} {row.get('object_label')}" for row in relations[:3]))
    if need == NEED_FUNCTIONAL_WORLD:
        return 'grounded_object_identity_for_functional_reasoning'
    if attributes:
        return 'owner_bound_visual_attributes=' + ' | '.join(attributes[:2])
    if relations:
        return 'direct_graph_relations=' + ' | '.join((f"{row.get('subject_label')} {row.get('predicate')} {row.get('object_label')}" for row in relations[:2]))
    return 'visual_crop_match_only; no canonical owner facts' if not pack.get('graph_bound') else 'graph_bound_visual_identity'

def _entity_attributes(entity: Any) -> list[str]:
    attributes = entity.attributes if isinstance(getattr(entity, 'attributes', None), dict) else {}
    values: list[str] = []
    for key in ('raw_visual_attributes', 'visual_attributes', 'processed_attributes', 'attribute_values'):
        values.extend(_flatten_text(attributes.get(key)))
    values.extend(_flatten_text(getattr(entity, 'states_seen', [])))
    return _ordered_unique(values)[:16]

def _answerability_score(need: str, *, raw_attributes: list[str], states: list[dict[str, Any]], relations: list[dict[str, Any]]) -> float:
    if need in {NEED_ENTITY_ATTRIBUTE, NEED_OCR_LOGO_TEXT}:
        return min(4.0, float(len(raw_attributes)))
    if need == NEED_EXISTENCE_OR_STATE:
        return min(4.0, 1.0 + float(len(states)))
    if need == NEED_SPATIAL_LOCALIZATION:
        return min(4.0, float(len(relations)))
    return min(3.0, 0.5 * len(raw_attributes) + 0.5 * len(relations))

def _compact_hit(hit: dict[str, Any], *, role: str) -> dict[str, Any]:
    return {'faiss_id': hit.get('faiss_id'), 'role': role, 'frame_index': hit.get('graph_frame_index'), 'scene_id': hit.get('scene_id'), 'image_name': hit.get('image_name'), 'object_index': hit.get('object_index'), 'label': hit.get('label'), 'bbox': hit.get('bbox'), 'similarity': hit.get('similarity'), 'observation_id': hit.get('observation_id'), 'canonical_entity_id': hit.get('canonical_entity_id'), 'binding_kind': hit.get('binding_kind'), 'binding_confidence': hit.get('binding_confidence'), 'retrieval_kind': hit.get('retrieval_kind') or 'semantic_object_recall', 'anchor_recall_similarity': hit.get('anchor_recall_similarity')}

def _same_visual_object(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for key in ('faiss_id', 'observation_id'):
        left = first.get(key)
        right = second.get(key)
        if left not in (None, '') and right not in (None, '') and (left == right):
            return True
    left_entity = str(first.get('canonical_entity_id') or '')
    right_entity = str(second.get('canonical_entity_id') or '')
    return bool(left_entity and right_entity and (left_entity == right_entity))

def _relation_matches_any(relation: str, requested: set[str]) -> bool:
    compatibility = {'left_of': {'left_of'}, 'right_of': {'right_of'}, 'above': {'above', 'on'}, 'below': {'below', 'under'}, 'overlapping_or_near': {'near', 'on', 'inside'}}
    return bool(compatibility.get(relation, {relation}) & requested)

def _between_spatial_pairs(targets: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    targets_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        frame = anchor.get('graph_frame_index')
        if isinstance(frame, int):
            anchors_by_frame[frame].append(anchor)
    for target in targets:
        frame = target.get('graph_frame_index')
        if isinstance(frame, int):
            targets_by_frame[frame].append(target)
    pairs: list[dict[str, Any]] = []
    for frame, frame_anchors in anchors_by_frame.items():
        for first_index, first in enumerate(frame_anchors):
            first_box = _bbox(first.get('bbox'))
            if first_box is None:
                continue
            for second in frame_anchors[first_index + 1:]:
                if _same_visual_object(first, second):
                    continue
                if str(first.get('requested_relation') or '') != 'between':
                    continue
                if str(second.get('requested_relation') or '') != 'between':
                    continue
                if str(first.get('intent_text') or '').lower() != str(second.get('intent_text') or '').lower():
                    continue
                second_box = _bbox(second.get('bbox'))
                if second_box is None:
                    continue
                first_center = ((first_box[0] + first_box[2]) / 2.0, (first_box[1] + first_box[3]) / 2.0)
                second_center = ((second_box[0] + second_box[2]) / 2.0, (second_box[1] + second_box[3]) / 2.0)
                horizontal = abs(first_center[0] - second_center[0]) >= abs(first_center[1] - second_center[1])
                lower = min(first_center[0 if horizontal else 1], second_center[0 if horizontal else 1])
                upper = max(first_center[0 if horizontal else 1], second_center[0 if horizontal else 1])
                if upper - lower < 1.0:
                    continue
                for target in targets_by_frame.get(frame, []):
                    if _same_visual_object(target, first) or _same_visual_object(target, second):
                        continue
                    target_box = _bbox(target.get('bbox'))
                    if target_box is None:
                        continue
                    target_center = ((target_box[0] + target_box[2]) / 2.0, (target_box[1] + target_box[3]) / 2.0)
                    coordinate = target_center[0 if horizontal else 1]
                    if not lower < coordinate < upper:
                        continue
                    score = min(float(first.get('similarity') or 0.0), float(second.get('similarity') or 0.0))
                    score += 0.1 if target.get('canonical_entity_id') else 0.0
                    score += 0.05 if first.get('canonical_entity_id') else 0.0
                    score += 0.05 if second.get('canonical_entity_id') else 0.0
                    pairs.append({'frame_index': frame, 'scene_id': target.get('scene_id') or first.get('scene_id'), 'target_label': target.get('label'), 'target_entity_id': target.get('canonical_entity_id') or '', 'relation': 'between', 'anchor_label': f"{first.get('label')} + {second.get('label')}", 'anchor_entity_id': ' + '.join((value for value in (str(first.get('canonical_entity_id') or ''), str(second.get('canonical_entity_id') or '')) if value)), 'target_bbox': target.get('bbox'), 'anchor_bbox': [first.get('bbox'), second.get('bbox')], 'pair_score': round(score, 6), 'evidence_kind': 'same_frame_visual_geometry', 'uses_gold_answer': False})
    return pairs

def _bbox_relation(target_value: Any, anchor_value: Any) -> str:
    target = _bbox(target_value)
    anchor = _bbox(anchor_value)
    if target is None or anchor is None:
        return ''
    tx1, ty1, tx2, ty2 = target
    ax1, ay1, ax2, ay2 = anchor
    tcx, tcy = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    acx, acy = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    tw, th = (max(1.0, tx2 - tx1), max(1.0, ty2 - ty1))
    aw, ah = (max(1.0, ax2 - ax1), max(1.0, ay2 - ay1))
    horizontal_gap = abs(tcx - acx) / max(tw, aw)
    vertical_gap = abs(tcy - acy) / max(th, ah)
    if tx2 <= ax1 or (tcx < acx and horizontal_gap > 0.35):
        return 'left_of'
    if tx1 >= ax2 or (tcx > acx and horizontal_gap > 0.35):
        return 'right_of'
    if ty2 <= ay1 or (tcy < acy and vertical_gap > 0.35):
        return 'above'
    if ty1 >= ay2 or (tcy > acy and vertical_gap > 0.35):
        return 'below'
    return 'overlapping_or_near'

def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return tuple((float(value[index]) for index in range(4)))
    except (TypeError, ValueError):
        return None

def _visual_stats(visual_payload: dict[str, Any], *, candidates: list[dict[str, Any]]) -> dict[str, int]:
    hit_count = sum((len(group.get('hits', []) or []) for group in visual_payload.get('groups', []) or [] if isinstance(group, dict)))
    return {'query_intent_count': int(visual_payload.get('query_intent_count') or 0), 'coframe_expansion_hit_count': int(visual_payload.get('coframe_expansion_hit_count') or 0), 'episode_candidate_count': int(visual_payload.get('episode_candidate_count') or 0), 'hit_count': hit_count, 'candidate_pack_count': len(candidates), 'graph_bound_candidate_count': sum((1 for row in candidates if row.get('graph_bound'))), 'unbound_hit_count': sum((1 for group in visual_payload.get('groups', []) or [] if isinstance(group, dict) for hit in group.get('hits', []) or [] if isinstance(hit, dict) and (not hit.get('canonical_entity_id'))))}

def _term_tokens(values: Any) -> set[str]:
    return set(_TOKEN_PATTERN.findall(' '.join(_flatten_text(values)).lower()))

def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = ' '.join(value.split())
        return [text] if text else []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if isinstance(item, (str, int, float)):
                result.append(f'{key}={item}')
            else:
                result.extend(_flatten_text(item))
        return result
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _flatten_text(item)]
    return [str(value)] if str(value).strip() else []

def _ordered_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = ' '.join(str(value or '').split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

def _unique_dicts(rows: Iterable[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple((row.get(name) for name in keys))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result
