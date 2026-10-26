"""Reusable visual evidence primitives for OpenEQA V5 handlers."""
from __future__ import annotations
from typing import Any
import re
COUNT_CANDIDATE_SET_CONTRACT = 'openeqa_v5_count_candidate_set.v1'
EXISTENCE_CANDIDATE_CONTRACT = 'openeqa_v5_existence_candidate.v1'
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_GENERIC_TARGET_TOKENS = {'entity', 'item', 'object', 'thing'}

def deduplicated_candidate_set(visual_payload: dict[str, Any], *, limit: int=64) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict) or group.get('retrieved_unit') != 'candidate_set':
            continue
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            canonical_id = _text(hit.get('canonical_entity_id'))
            label = _text(hit.get('label'))
            frame_index = _optional_int(hit.get('graph_frame_index'))
            key = canonical_id or f'{label.casefold()}@{frame_index}'
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append({'canonical_entity_id': canonical_id, 'label': label, 'graph_frame_index': frame_index, 'similarity': float(hit.get('similarity') or 0.0), 'binding_kind': _text(hit.get('binding_kind')), 'binding_confidence': float(hit.get('binding_confidence') or 0.0), 'candidate_source': 'route_visual_candidate_set'})
            if len(candidates) >= max(1, int(limit)):
                return candidates
    return candidates

def count_candidate_lines(candidates: list[dict[str, Any]]) -> list[str]:
    return [f"kind=v5_count_candidate label={candidate.get('label')!r} canonical_entity_id={candidate.get('canonical_entity_id')!r} frame={candidate.get('graph_frame_index')} similarity={float(candidate.get('similarity') or 0.0):.4f}" for candidate in candidates]

def publish_count_candidate_set(fused: dict[str, Any], *, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    visual_evidence = fused.get('visual_evidence')
    bundle = fused.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return fused
    lines = count_candidate_lines(candidates)
    payload = {'contract': COUNT_CANDIDATE_SET_CONTRACT, 'candidate_count': len(candidates), 'deduplicated_by_canonical_entity': True, 'candidates_are_retrieval_candidates_not_count_answers': True, 'candidates': candidates}
    visual_evidence['count_candidate_set'] = candidates
    visual_evidence['count_candidate_set_contract'] = COUNT_CANDIDATE_SET_CONTRACT
    visual_evidence['route_strategy_applied'] = bool(candidates)
    stats = dict(visual_evidence.get('stats') or {})
    stats['v5_count_candidate_set_count'] = len(candidates)
    visual_evidence['stats'] = stats
    bundle['question_route_count_set'] = payload
    bundle['question_route_count_set_lines'] = lines
    structured = bundle.get('structured_graph_context')
    if isinstance(structured, dict):
        structured['question_route_count_set'] = payload
    return fused

def positive_existence_candidates(visual_payload: dict[str, Any], *, target_terms: list[str], limit: int=8) -> list[dict[str, Any]]:
    target_token_sets = [tokens for term in target_terms if (tokens := set(_TOKEN_PATTERN.findall(term.casefold()))) and (not tokens.issubset(_GENERIC_TARGET_TOKENS))]
    if not target_token_sets:
        return []
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict) or group.get('retrieved_unit') not in {'canonical_entity_tracklet', 'object_crop'}:
            continue
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            label = _text(hit.get('label'))
            label_tokens = set(_TOKEN_PATTERN.findall(label.casefold()))
            if not any((tokens.issubset(label_tokens) for tokens in target_token_sets)):
                continue
            canonical_id = _text(hit.get('canonical_entity_id'))
            frame_index = _optional_int(hit.get('graph_frame_index'))
            key = canonical_id or f'{label.casefold()}@{frame_index}'
            if key in seen:
                continue
            seen.add(key)
            candidates.append({'canonical_entity_id': canonical_id, 'label': label, 'graph_frame_index': frame_index, 'similarity': float(hit.get('similarity') or 0.0), 'candidate_source': 'strict_target_label_visual_match'})
            if len(candidates) >= max(1, int(limit)):
                return candidates
    return candidates

def publish_existence_candidates(fused: dict[str, Any], *, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    visual_evidence = fused.get('visual_evidence')
    bundle = fused.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict):
        return fused
    lines = [f"kind=v5_existence_positive_candidate label={candidate.get('label')!r} canonical_entity_id={candidate.get('canonical_entity_id')!r} frame={candidate.get('graph_frame_index')}" for candidate in candidates]
    payload = {'contract': EXISTENCE_CANDIDATE_CONTRACT, 'positive_candidate_count': len(candidates), 'negative_coverage_qualified': False, 'missing_candidate_does_not_imply_absence': True, 'candidates': candidates}
    visual_evidence['existence_positive_candidates'] = candidates
    visual_evidence['existence_candidate_contract'] = EXISTENCE_CANDIDATE_CONTRACT
    visual_evidence['negative_coverage_qualified'] = False
    visual_evidence['route_strategy_applied'] = bool(candidates)
    stats = dict(visual_evidence.get('stats') or {})
    stats['v5_existence_positive_candidate_count'] = len(candidates)
    visual_evidence['stats'] = stats
    if isinstance(bundle, dict):
        bundle['question_route_existence'] = payload
        bundle['question_route_existence_lines'] = lines
    return fused

def _text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
__all__ = ['COUNT_CANDIDATE_SET_CONTRACT', 'EXISTENCE_CANDIDATE_CONTRACT', 'count_candidate_lines', 'deduplicated_candidate_set', 'positive_existence_candidates', 'publish_count_candidate_set', 'publish_existence_candidates']
