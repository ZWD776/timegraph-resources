"""OpenEQA v2 same-frame spatial relation verification.

The frozen v1 runtime remains responsible for broad visual recall. This layer
only changes answer-visible evidence when a requested relation can be proven
by same-frame bounding-box geometry or an explicit canonical relation.
"""
from __future__ import annotations
from copy import deepcopy
import math
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import NEED_SPATIAL_LOCALIZATION
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import fuse_navqa_equivalent_visual_evidence
SPATIAL_VERIFIER_CONTRACT = 'openeqa_spatial_geometry_verifier.v1'
SPATIAL_VERIFIED_BACKBONE_CONTRACT = 'openeqa_navqa_equivalent_visual_backbone.spatial_verified_v2'
_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v2'
_GEOMETRY_RELATIONS = {'left_of', 'right_of', 'above', 'below', 'between'}
_INVERSE_RELATION = {'left_of': 'right_of', 'right_of': 'left_of', 'above': 'below', 'below': 'above'}
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_GENERIC_ANCHOR_TOKENS = {'a', 'an', 'area', 'frame', 'in', 'item', 'light', 'object', 'of', 'on', 'one', 'room', 'side', 'the', 'thing', 'two', 'wall'}
_VISUAL_LINE_FIELDS = ('visual_faiss_evidence_lines', 'visual_entity_candidate_pack_lines', 'visual_faiss_unified_candidate_lines', 'visual_faiss_evidence_layer_lines', 'visual_faiss_local_subgraph_fact_lines', 'visual_faiss_role_aware_lines', 'visual_faiss_role_aware_aux_lines', 'visual_faiss_object_local_fact_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_ger_evidence_lines', 'frame_faiss_evidence_lines', 'merged_frame_faiss_evidence_lines')

def fuse_spatial_verified_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Run frozen v1, then verify only geometry-applicable spatial evidence."""
    fused = fuse_navqa_equivalent_visual_evidence(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    requested = _requested_geometry_relations(evidence_demand)
    if not requested or not evidence_demand.get('anchor_terms'):
        return _mark_v2_passthrough(fused)
    pairs = verify_spatial_pairs(visual_payload, graph=graph, evidence_demand=evidence_demand, limit=max(1, int(max_candidate_packs)))
    return _replace_answer_visible_spatial_evidence(fused, pairs=pairs, requested_relations=requested, max_visual_lines=max_visual_lines)

def verify_spatial_pairs(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any], limit: int=8) -> list[dict[str, Any]]:
    """Return relation-proven target/anchor pairs without using answer labels."""
    requested = _requested_geometry_relations(evidence_demand)
    if not requested:
        return []
    relation_anchor_terms = _relation_anchor_terms(evidence_demand)
    anchors, semantic_targets, coframe_targets = _role_hits(visual_payload)
    if not anchors:
        return []
    if 'between' in requested:
        anchors = _augment_between_anchors(anchors, coframe_targets=coframe_targets, evidence_demand=evidence_demand)
    semantic_scores = {}
    for hit in semantic_targets:
        key = _hit_key(hit)
        semantic_scores[key] = max(semantic_scores.get(key, float('-inf')), float(hit.get('similarity') or 0.0))
    targets = _target_candidates(semantic_targets=semantic_targets, coframe_targets=coframe_targets, target_terms=evidence_demand.get('target_terms', []), open_target_slot=bool(evidence_demand.get('open_target_slot')), semantic_scores=semantic_scores)
    if bool(evidence_demand.get('open_target_slot')) and 'between' in requested:
        excluded_anchor_terms = [term for relation in requested for term in relation_anchor_terms.get(relation, [])]
        targets = [hit for hit in targets if not _hit_matches_terms(hit, excluded_anchor_terms)]
    if not targets:
        return []
    canonical_relations = _canonical_relation_index(graph)
    raw_pairs: list[dict[str, Any]] = []
    if 'between' in requested:
        raw_pairs.extend(_between_pairs(targets, anchors, anchor_terms=relation_anchor_terms.get('between', []), canonical_relations=canonical_relations, semantic_scores=semantic_scores))
    for relation in sorted(requested - {'between'}):
        raw_pairs.extend(_directional_pairs(targets, anchors, relation=relation, anchor_terms=relation_anchor_terms.get(relation, []), canonical_relations=canonical_relations, semantic_scores=semantic_scores))
    return _aggregate_pairs(raw_pairs, limit=max(1, int(limit)))

def _role_hits(visual_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    semantic_targets: list[dict[str, Any]] = []
    coframe_targets: list[dict[str, Any]] = []
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent')
        intent = intent if isinstance(intent, dict) else {}
        role = str(intent.get('role') or 'context')
        for raw_hit in group.get('hits', []) or []:
            if not isinstance(raw_hit, dict) or _bbox(raw_hit.get('bbox')) is None:
                continue
            hit = deepcopy(raw_hit)
            hit['retrieval_role'] = role
            hit['intent_text'] = str(intent.get('text') or '')
            hit['requested_relation'] = _normalize_relation(intent.get('spatial_relation'))
            if role == 'anchor':
                anchors.append(hit)
            elif role == 'spatial_target':
                coframe_targets.append(hit)
            elif role == 'target':
                semantic_targets.append(hit)
    return (_dedupe_hits(anchors), _dedupe_hits(semantic_targets), _dedupe_hits(coframe_targets))

def _target_candidates(*, semantic_targets: list[dict[str, Any]], coframe_targets: list[dict[str, Any]], target_terms: Any, open_target_slot: bool, semantic_scores: dict[str, float]) -> list[dict[str, Any]]:
    if open_target_slot:
        return coframe_targets or semantic_targets
    semantic_keys = set(semantic_scores)
    matched_coframe = [hit for hit in coframe_targets if _hit_key(hit) in semantic_keys or _hit_matches_terms(hit, target_terms)]
    return _dedupe_hits([*semantic_targets, *matched_coframe])

def _augment_between_anchors(anchors: list[dict[str, Any]], *, coframe_targets: list[dict[str, Any]], evidence_demand: dict[str, Any]) -> list[dict[str, Any]]:
    between_terms = [row.get('anchor') for row in evidence_demand.get('spatial_constraints', []) or [] if isinstance(row, dict) and _normalize_relation(row.get('relation')) == 'between' and row.get('anchor')]
    if not between_terms:
        between_terms = [hit.get('intent_text') for hit in anchors if hit.get('requested_relation') == 'between']
    additions: list[dict[str, Any]] = []
    for hit in coframe_targets:
        if not _hit_matches_terms(hit, between_terms):
            continue
        item = deepcopy(hit)
        item['retrieval_role'] = 'anchor'
        item['requested_relation'] = 'between'
        item['intent_text'] = ' | '.join((str(value) for value in between_terms))
        item['anchor_source'] = 'same_frame_anchor_completion'
        additions.append(item)
    return _dedupe_hits([*anchors, *additions])

def _directional_pairs(targets: list[dict[str, Any]], anchors: list[dict[str, Any]], *, relation: str, anchor_terms: list[str], canonical_relations: set[tuple[str, str, str]], semantic_scores: dict[str, float]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for target in targets:
        for anchor in anchors:
            if _frame(target) != _frame(anchor) or _same_object(target, anchor) or _same_canonical_entity(target, anchor):
                continue
            if anchor_terms and (not _hit_matches_terms(anchor, anchor_terms)):
                continue
            anchor_relation = str(anchor.get('requested_relation') or '')
            if anchor_relation in _GEOMETRY_RELATIONS and anchor_relation != relation:
                continue
            geometry_confidence = _geometry_confidence(target.get('bbox'), anchor.get('bbox'), relation)
            canonical_verified = _canonical_relation_verified(target, anchor, relation=relation, canonical_relations=canonical_relations)
            if geometry_confidence <= 0.0 and (not canonical_verified):
                continue
            verification_sources = []
            if geometry_confidence > 0.0:
                verification_sources.append('same_frame_bbox_geometry')
            if canonical_verified:
                verification_sources.append('canonical_relation')
            pairs.append(_pair_record(target, anchor, relation=relation, geometry_confidence=geometry_confidence, canonical_verified=canonical_verified, semantic_score=semantic_scores.get(_hit_key(target), 0.0), verification_sources=verification_sources))
    return pairs

def _between_pairs(targets: list[dict[str, Any]], anchors: list[dict[str, Any]], *, anchor_terms: list[str], canonical_relations: set[tuple[str, str, str]], semantic_scores: dict[str, float]) -> list[dict[str, Any]]:
    del canonical_relations
    between_anchors = [hit for hit in anchors if hit.get('requested_relation') == 'between'] or anchors
    if anchor_terms:
        between_anchors = [hit for hit in between_anchors if _hit_matches_terms(hit, anchor_terms)]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for anchor in between_anchors:
        frame = _frame(anchor)
        if frame is not None:
            by_frame.setdefault(frame, []).append(anchor)
    pairs: list[dict[str, Any]] = []
    for target in targets:
        frame = _frame(target)
        if frame is None:
            continue
        frame_anchors = by_frame.get(frame, [])
        for index, first in enumerate(frame_anchors):
            for second in frame_anchors[index + 1:]:
                if _same_object(first, second) or _same_object(target, first) or _same_object(target, second) or _same_canonical_entity(target, first) or _same_canonical_entity(target, second):
                    continue
                if anchor_terms and (not _between_anchor_pair_matches(first, second, anchor_terms)):
                    continue
                confidence = _between_confidence(target.get('bbox'), first.get('bbox'), second.get('bbox'))
                if confidence <= 0.0:
                    continue
                record = _pair_record(target, first, relation='between', geometry_confidence=confidence, canonical_verified=False, semantic_score=semantic_scores.get(_hit_key(target), 0.0), verification_sources=['same_frame_bbox_geometry_two_anchors'])
                record['anchor_label'] = f"{first.get('label') or 'object'} + {second.get('label') or 'object'}"
                record['anchor_entity_id'] = ' + '.join((value for value in (str(first.get('canonical_entity_id') or ''), str(second.get('canonical_entity_id') or '')) if value))
                record['anchor_bbox'] = [first.get('bbox'), second.get('bbox')]
                record['anchor_observations'] = [_compact_hit(first), _compact_hit(second)]
                record['pair_score'] = round(float(record['pair_score']) + 0.5 * float(second.get('similarity') or 0.0) + (0.05 if second.get('canonical_entity_id') else 0.0), 6)
                pairs.append(record)
    return pairs

def _pair_record(target: dict[str, Any], anchor: dict[str, Any], *, relation: str, geometry_confidence: float, canonical_verified: bool, semantic_score: float, verification_sources: list[str]) -> dict[str, Any]:
    target_bound = bool(target.get('canonical_entity_id'))
    anchor_bound = bool(anchor.get('canonical_entity_id'))
    anchor_score = float(anchor.get('similarity') or 0.0)
    pair_score = anchor_score + 0.45 * max(0.0, semantic_score) + 0.25 * geometry_confidence + (0.12 if target_bound else 0.0) + (0.08 if anchor_bound else 0.0) + (0.5 if canonical_verified else 0.0)
    frame = _frame(target)
    return {'contract': SPATIAL_VERIFIER_CONTRACT, 'frame_index': frame, 'scene_id': target.get('scene_id') or anchor.get('scene_id'), 'target_label': str(target.get('label') or 'object'), 'target_entity_id': str(target.get('canonical_entity_id') or ''), 'target_bbox': target.get('bbox'), 'anchor_label': str(anchor.get('label') or 'object'), 'anchor_entity_id': str(anchor.get('canonical_entity_id') or ''), 'anchor_bbox': anchor.get('bbox'), 'relation': relation, 'relation_verified': True, 'verification_sources': verification_sources, 'geometry_confidence': round(geometry_confidence, 6), 'target_semantic_similarity': round(max(0.0, semantic_score), 6), 'anchor_recall_similarity': round(anchor_score, 6), 'pair_score': round(pair_score, 6), 'target_observation': _compact_hit(target), 'anchor_observations': [_compact_hit(anchor)], 'graph_bound': target_bound and anchor_bound, 'uses_gold_answer': False}

def _aggregate_pairs(pairs: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pair in sorted(pairs, key=lambda row: (-float(row.get('pair_score') or 0.0), int(row.get('frame_index') or 0), str(row.get('target_label') or ''))):
        target_key = str(pair.get('target_entity_id') or '') or _hit_key(pair.get('target_observation') or {})
        anchor_key = str(pair.get('anchor_entity_id') or '') or str(pair.get('anchor_label') or '')
        key = (target_key, str(pair.get('relation') or ''), anchor_key)
        existing = grouped.get(key)
        if existing is None:
            item = deepcopy(pair)
            item['best_pair_score'] = float(pair.get('pair_score') or 0.0)
            item['support_frames'] = [pair.get('frame_index')]
            item['support_frame_count'] = 1
            grouped[key] = item
            continue
        frames = [*existing.get('support_frames', []), pair.get('frame_index')]
        existing['support_frames'] = _ordered_unique(frames)
        existing['support_frame_count'] = len(existing['support_frames'])
        existing['pair_score'] = round(float(existing.get('best_pair_score') or 0.0) + min(0.2, 0.04 * (existing['support_frame_count'] - 1)), 6)
    result = list(grouped.values())
    for item in result:
        item.pop('best_pair_score', None)
    result.sort(key=lambda row: (-float(row.get('pair_score') or 0.0), -int(row.get('support_frame_count') or 0), str(row.get('target_label') or '')))
    return result[:limit]

def _replace_answer_visible_spatial_evidence(graph_result: dict[str, Any], *, pairs: list[dict[str, Any]], requested_relations: set[str], max_visual_lines: int) -> dict[str, Any]:
    bundle = graph_result.get('llm_ready_bundle')
    if not isinstance(bundle, dict):
        return graph_result
    updated = deepcopy(bundle)
    structured = deepcopy(updated.get('structured_graph_context') or {})
    old_visual = deepcopy(graph_result.get('visual_evidence') or {})
    graph_lines = [str(line) for line in updated.get('evidence_lines', []) or [] if str(line).strip() and (not _is_visual_line(line))]
    graph_aggregation_lines = [str(line) for line in updated.get('evidence_aggregation_lines', []) or [] if str(line).strip() and (not _is_visual_line(line))]
    pair_lines = [_spatial_pair_line(index, pair) for index, pair in enumerate(pairs, 1)]
    if not pair_lines:
        pair_lines = [f"kind=openeqa_spatial_verification relation_verified=false requested_relations={','.join(sorted(requested_relations))} visual_candidates_are_not_spatial_proof=true fallback=graph_evidence_only"]
    visible_pair_lines = pair_lines[:max(1, int(max_visual_lines))]
    for field in _VISUAL_LINE_FIELDS:
        updated[field] = []
        if isinstance(structured, dict):
            structured[field] = []
    updated['question_route_spatial_pair_lines'] = visible_pair_lines
    updated['evidence_lines'] = _ordered_unique([*visible_pair_lines, *graph_lines])
    updated['evidence_aggregation_lines'] = _ordered_unique([*visible_pair_lines, *graph_aggregation_lines])
    aggregation = deepcopy(updated.get('evidence_aggregation') or {})
    graph_facts = [fact for fact in aggregation.get('candidate_facts', []) or [] if isinstance(fact, dict) and (not _is_visual_fact(fact))]
    aggregation['candidate_facts'] = [*[_pair_candidate_fact(pair) for pair in pairs], *graph_facts]
    updated['evidence_aggregation'] = aggregation
    verifier = {'contract': SPATIAL_VERIFIER_CONTRACT, 'applied': True, 'requested_relations': sorted(requested_relations), 'verified_pair_count': len(pairs), 'fallback_to_graph_evidence': not bool(pairs), 'unverified_visual_candidates_are_answer_evidence': False, 'uses_gold_answer': False}
    visual_evidence = {**old_visual, 'contract': SPATIAL_VERIFIED_BACKBONE_CONTRACT, 'visual_runtime_contract': _RUNTIME_CONTRACT, 'available': bool(pairs), 'candidate_packs': [_pair_candidate_pack(pair) for pair in pairs], 'object_local_facts': [], 'bound_facts': [], 'spatial_pairs': pairs, 'evidence_lines': visible_pair_lines, 'spatial_verification': verifier, 'stats': {**dict(old_visual.get('stats') or {}), 'v1_candidate_pack_count_before_verification': len(old_visual.get('candidate_packs', []) or []), 'verified_spatial_pair_count': len(pairs), 'answer_visible_unverified_spatial_candidate_count': 0}}
    updated['visual_runtime_contract'] = _RUNTIME_CONTRACT
    updated['visual_evidence_contract'] = SPATIAL_VERIFIED_BACKBONE_CONTRACT
    updated['visual_evidence'] = visual_evidence
    updated['question_route_spatial_verification'] = verifier
    if isinstance(structured, dict):
        structured['visual_runtime_contract'] = _RUNTIME_CONTRACT
        structured['visual_evidence_contract'] = SPATIAL_VERIFIED_BACKBONE_CONTRACT
        structured['visual_evidence'] = visual_evidence
        structured['question_route_spatial_pair_lines'] = visible_pair_lines
        structured['question_route_spatial_verification'] = verifier
        structured['visual_faiss_object_local_facts'] = []
        structured['visual_faiss_ger_bound_facts'] = []
    updated['structured_graph_context'] = structured
    graph_result['llm_ready_bundle'] = updated
    graph_result['visual_evidence'] = visual_evidence
    return graph_result

def _mark_v2_passthrough(graph_result: dict[str, Any]) -> dict[str, Any]:
    bundle = graph_result.get('llm_ready_bundle')
    if not isinstance(bundle, dict):
        return graph_result
    updated = deepcopy(bundle)
    visual = deepcopy(graph_result.get('visual_evidence') or {})
    visual['visual_runtime_contract'] = _RUNTIME_CONTRACT
    visual['spatial_verification'] = {'contract': SPATIAL_VERIFIER_CONTRACT, 'applied': False, 'reason': 'no_bbox_verifiable_relation_requested', 'uses_gold_answer': False}
    updated['visual_runtime_contract'] = _RUNTIME_CONTRACT
    updated['visual_evidence'] = visual
    structured = deepcopy(updated.get('structured_graph_context') or {})
    if isinstance(structured, dict):
        structured['visual_runtime_contract'] = _RUNTIME_CONTRACT
        structured['visual_evidence'] = visual
    updated['structured_graph_context'] = structured
    graph_result['llm_ready_bundle'] = updated
    graph_result['visual_evidence'] = visual
    return graph_result

def _requested_geometry_relations(evidence_demand: dict[str, Any]) -> set[str]:
    if str(evidence_demand.get('primary_need') or '') != NEED_SPATIAL_LOCALIZATION:
        return set()
    return {relation for value in evidence_demand.get('spatial_relations', []) or [] if (relation := _normalize_relation(value)) in _GEOMETRY_RELATIONS}

def requested_geometry_relations(evidence_demand: dict[str, Any]) -> set[str]:
    """Expose the verifier applicability check to later OpenEQA runtimes."""
    return _requested_geometry_relations(evidence_demand)

def _relation_anchor_terms(evidence_demand: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in evidence_demand.get('spatial_constraints', []) or []:
        if not isinstance(item, dict):
            continue
        relation = _normalize_relation(item.get('relation'))
        anchor = ' '.join(str(item.get('anchor') or '').split())
        if relation in _GEOMETRY_RELATIONS and anchor:
            result.setdefault(relation, []).append(anchor)
    if not result:
        fallback = [' '.join(str(value or '').split()) for value in evidence_demand.get('anchor_terms', []) or [] if str(value or '').strip()]
        for relation in _requested_geometry_relations(evidence_demand):
            result[relation] = fallback
    return result

def _geometry_confidence(target_value: Any, anchor_value: Any, relation: str) -> float:
    target = _bbox(target_value)
    anchor = _bbox(anchor_value)
    if target is None or anchor is None:
        return 0.0
    tx1, ty1, tx2, ty2 = target
    ax1, ay1, ax2, ay2 = anchor
    tcx, tcy = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    acx, acy = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    tw, th = (max(1.0, tx2 - tx1), max(1.0, ty2 - ty1))
    aw, ah = (max(1.0, ax2 - ax1), max(1.0, ay2 - ay1))
    if relation == 'left_of':
        gap = (acx - tcx) / max(tw, aw)
        separated = tx2 <= acx
        orthogonal_offset = abs(tcy - acy) / max(th, ah)
        orthogonal_overlap = _axis_overlap_ratio(ty1, ty2, ay1, ay2)
    elif relation == 'right_of':
        gap = (tcx - acx) / max(tw, aw)
        separated = tx1 >= acx
        orthogonal_offset = abs(tcy - acy) / max(th, ah)
        orthogonal_overlap = _axis_overlap_ratio(ty1, ty2, ay1, ay2)
    elif relation == 'above':
        gap = (acy - tcy) / max(th, ah)
        separated = ty2 <= acy
        orthogonal_offset = abs(tcx - acx) / max(tw, aw)
        orthogonal_overlap = _axis_overlap_ratio(tx1, tx2, ax1, ax2)
    elif relation == 'below':
        gap = (tcy - acy) / max(th, ah)
        separated = ty1 >= acy
        orthogonal_offset = abs(tcx - acx) / max(tw, aw)
        orthogonal_overlap = _axis_overlap_ratio(tx1, tx2, ax1, ax2)
    else:
        return 0.0
    if gap < 0.25 or not separated:
        return 0.0
    direction_confidence = min(1.0, gap)
    proximity = 1.0 / (1.0 + max(0.0, gap - 0.25))
    alignment = max(orthogonal_overlap, 1.0 / (1.0 + orthogonal_offset))
    return round(0.2 * direction_confidence + 0.5 * proximity + 0.3 * alignment, 6)

def _between_confidence(target_value: Any, first_value: Any, second_value: Any) -> float:
    target = _bbox(target_value)
    first = _bbox(first_value)
    second = _bbox(second_value)
    if target is None or first is None or second is None:
        return 0.0
    tc = _center(target)
    fc = _center(first)
    sc = _center(second)
    if _point_in_box(tc, first) or _point_in_box(tc, second):
        return 0.0
    if _box_overlap_ratio(target, first) > 0.35 or _box_overlap_ratio(target, second) > 0.35:
        return 0.0
    vx, vy = (sc[0] - fc[0], sc[1] - fc[1])
    length_sq = vx * vx + vy * vy
    if length_sq < 4.0:
        return 0.0
    wx, wy = (tc[0] - fc[0], tc[1] - fc[1])
    projection = (wx * vx + wy * vy) / length_sq
    if projection <= 0.08 or projection >= 0.92:
        return 0.0
    closest = (fc[0] + projection * vx, fc[1] + projection * vy)
    perpendicular = math.dist(tc, closest) / math.sqrt(length_sq)
    if perpendicular > 0.45:
        return 0.0
    centered = 1.0 - min(1.0, abs(projection - 0.5) / 0.5)
    aligned = 1.0 - min(1.0, perpendicular / 0.45)
    return round(max(0.05, 0.55 * centered + 0.45 * aligned), 6)

def _canonical_relation_index(graph: CanonicalTemporalMemoryGraph) -> set[tuple[str, str, str]]:
    relations: set[tuple[str, str, str]] = set()
    for episode in graph.episode_nodes:
        if str(episode.episode_kind or '').upper() != 'RELATION':
            continue
        subject = str(episode.subject_entity_id or '')
        obj = str(episode.object_entity_id or '')
        relation = _normalize_relation(episode.raw_predicate or episode.predicate)
        if subject and obj and relation:
            relations.add((subject, relation, obj))
    return relations

def _canonical_relation_verified(target: dict[str, Any], anchor: dict[str, Any], *, relation: str, canonical_relations: set[tuple[str, str, str]]) -> bool:
    target_id = str(target.get('canonical_entity_id') or '')
    anchor_id = str(anchor.get('canonical_entity_id') or '')
    if not target_id or not anchor_id:
        return False
    if (target_id, relation, anchor_id) in canonical_relations:
        return True
    inverse = _INVERSE_RELATION.get(relation)
    return bool(inverse and (anchor_id, inverse, target_id) in canonical_relations)

def _spatial_pair_line(index: int, pair: dict[str, Any]) -> str:
    return f"rank={index} kind=openeqa_verified_spatial_pair frame={pair.get('frame_index')} target={pair.get('target_label')!r} target_entity={pair.get('target_entity_id') or 'unbound_visual_observation'} relation={pair.get('relation')} anchor={pair.get('anchor_label')!r} anchor_entity={pair.get('anchor_entity_id') or 'unbound_visual_observation'} pair_score={float(pair.get('pair_score') or 0.0):.3f} support_frames={pair.get('support_frames') or [pair.get('frame_index')]} relation_verified=true uses_gold_answer=false"

def _pair_candidate_pack(pair: dict[str, Any]) -> dict[str, Any]:
    return {'contract': SPATIAL_VERIFIER_CONTRACT, 'candidate_id': 'spatial_pair:' + _hit_key(pair.get('target_observation') or {}), 'candidate_role': 'spatial_target', 'canonical_entity_id': pair.get('target_entity_id'), 'entity_label': pair.get('target_label'), 'graph_bound': bool(pair.get('target_entity_id')), 'binding_strength': 'canonical_relation_verified' if 'canonical_relation' in pair.get('verification_sources', []) else 'same_frame_geometry_verified', 'binding_confidence': pair.get('geometry_confidence'), 'ranking_score': pair.get('pair_score'), 'relation': pair.get('relation'), 'relation_verified': True, 'visual_anchors': [pair.get('target_observation') or {}, *list(pair.get('anchor_observations') or [])], 'spatial_pair': pair, 'uses_gold_answer': False}

def _pair_candidate_fact(pair: dict[str, Any]) -> dict[str, Any]:
    return {'fact_type': 'openeqa_verified_spatial_pair', 'fact_text': _spatial_pair_line(1, pair), 'binding_strength': 'relation_verified', 'source_record_id': 'spatial_pair:' + _hit_key(pair.get('target_observation') or {}), 'source': SPATIAL_VERIFIER_CONTRACT, 'source_kind': 'openeqa_verified_spatial_pair', 'entity_ids': [value for value in (pair.get('target_entity_id'), pair.get('anchor_entity_id')) if value], 'score': pair.get('pair_score'), 'uses_gold_answers': False}

def _is_visual_line(value: Any) -> bool:
    text = str(value or '').lower()
    return 'visual_faiss' in text or 'kind=visual_' in text or 'source=visual_' in text or ('visual crop' in text)

def _is_visual_fact(fact: dict[str, Any]) -> bool:
    values = ' '.join((str(fact.get(key) or '') for key in ('fact_type', 'source', 'source_kind', 'source_record_id'))).lower()
    return 'visual' in values or 'faiss' in values

def _hit_matches_terms(hit: dict[str, Any], terms: Any) -> bool:
    hit_tokens = _term_tokens([hit.get('label'), hit.get('visual_attribute'), hit.get('raw_visual_attributes')])
    normalized_hit_tokens = _singularized(hit_tokens)
    values = terms if isinstance(terms, (list, tuple, set)) else [terms]
    for value in values:
        target_tokens = _singularized(_term_tokens(value))
        if not target_tokens:
            continue
        discriminative = target_tokens - _GENERIC_ANCHOR_TOKENS
        required = discriminative or target_tokens
        if required & normalized_hit_tokens:
            return True
    return False

def _between_anchor_pair_matches(first: dict[str, Any], second: dict[str, Any], terms: list[str]) -> bool:
    for value in terms:
        components = [part.strip() for part in re.split('\\band\\b|,', str(value), flags=re.IGNORECASE) if part.strip()]
        if len(components) >= 2:
            left, right = (components[0], components[1])
            if _hit_matches_terms(first, [left]) and _hit_matches_terms(second, [right]) or (_hit_matches_terms(first, [right]) and _hit_matches_terms(second, [left])):
                return True
            continue
        if _hit_matches_terms(first, [value]) and _hit_matches_terms(second, [value]):
            return True
    return False

def _term_tokens(values: Any) -> set[str]:
    flattened: list[str] = []
    if isinstance(values, (list, tuple, set)):
        flattened.extend((str(value or '') for value in values))
    else:
        flattened.append(str(values or ''))
    return set(_TOKEN_PATTERN.findall(' '.join(flattened).lower()))

def _singularized(tokens: set[str]) -> set[str]:
    result = set(tokens)
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            result.add(token[:-3] + 'y')
        elif len(token) > 3 and token.endswith('s'):
            result.add(token[:-1])
    return result

def _normalize_relation(value: Any) -> str:
    relation = '_'.join(_TOKEN_PATTERN.findall(str(value or '').lower()))
    if not relation:
        return ''
    if 'between' in relation:
        return 'between'
    if 'left' in relation:
        return 'left_of'
    if 'right' in relation:
        return 'right_of'
    if relation in {'above', 'over', 'on_top_of'}:
        return 'above'
    if relation in {'below', 'under', 'beneath', 'underneath'}:
        return 'below'
    return relation

def _dedupe_hits(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        key = _hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result

def _hit_key(hit: dict[str, Any]) -> str:
    for key in ('faiss_id', 'observation_id'):
        value = hit.get(key)
        if value not in (None, ''):
            return f'{key}:{value}'
    return 'fallback:' + '|'.join((str(hit.get(key) or '') for key in ('graph_frame_index', 'object_index', 'label', 'canonical_entity_id')))

def _same_object(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_observation = first.get('faiss_id', first.get('observation_id'))
    second_observation = second.get('faiss_id', second.get('observation_id'))
    if first_observation not in (None, '') and second_observation not in (None, ''):
        return _hit_key(first) == _hit_key(second)
    first_entity = str(first.get('canonical_entity_id') or '')
    second_entity = str(second.get('canonical_entity_id') or '')
    return bool(first_entity and first_entity == second_entity)

def _same_canonical_entity(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_entity = str(first.get('canonical_entity_id') or '')
    second_entity = str(second.get('canonical_entity_id') or '')
    return bool(first_entity and first_entity == second_entity)

def _frame(hit: dict[str, Any]) -> int | None:
    value = hit.get('graph_frame_index')
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def _axis_overlap_ratio(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / max(1.0, min(first_end - first_start, second_end - second_start))

def _point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

def _box_overlap_ratio(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    return intersection / first_area

def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {key: hit.get(key) for key in ('faiss_id', 'observation_id', 'graph_frame_index', 'object_index', 'label', 'bbox', 'similarity', 'canonical_entity_id', 'binding_kind', 'binding_confidence', 'retrieval_kind')}

def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
