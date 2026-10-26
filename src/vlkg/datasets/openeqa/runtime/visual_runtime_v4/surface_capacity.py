"""Supplemental surface-occupancy evidence for OpenEQA capacity questions."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import STATE_VARIANT_SURFACE_CAPACITY
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'existence_or_state.surface_capacity_observation_v1'
SURFACE_CAPACITY_CONTRACT = 'openeqa_surface_capacity_observation_evidence.v1'
SPECIALIZED_EVIDENCE_FIELD = 'question_route_specialized_evidence_lines'
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_OWNER_PATTERN = re.compile("owner=(?P<owner>.+?)\\s+label='(?P<label>[^']+)'.*?raw_visual_attributes=(?P<attribute>.+)$", re.IGNORECASE)
_STOPWORDS = {'a', 'all', 'an', 'any', 'are', 'for', 'is', 'my', 'on', 'room', 'space', 'the', 'there', 'to', 'work'}

def fuse_surface_capacity_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    if str(evidence_demand.get('strategy_variant') or '') != STATE_VARIANT_SURFACE_CAPACITY:
        return fused
    visual_evidence = fused.get('visual_evidence')
    if not isinstance(visual_evidence, dict):
        return fused
    raw_packs = [dict(pack) for pack in visual_evidence.get('candidate_packs', []) or [] if isinstance(pack, dict)]
    target_phrases = _target_phrases(evidence_demand=evidence_demand, question_ir=question_ir)
    candidates = _select_surface_observations(raw_packs, target_phrases=target_phrases, limit=min(6, max(1, int(max_candidate_packs))))
    return _publish_surface_capacity_evidence(fused, candidates=candidates, raw_pack_count=len(raw_packs), target_phrases=target_phrases, question=question, max_visual_lines=max_visual_lines)

def _select_surface_observations(packs: list[dict[str, Any]], *, target_phrases: list[str], limit: int) -> list[dict[str, Any]]:
    target_tokens = _canonical_tokens(target_phrases)
    grouped: dict[str, dict[str, Any]] = {}
    for pack in packs:
        fact_values = [*(pack.get('hsgg_visual_facts', []) or []), *(pack.get('hsgg_neighbor_attributes', []) or [])]
        entity_label = str(pack.get('entity_label') or '').strip()
        if entity_label and _label_matches(entity_label, target_tokens=target_tokens):
            owner = str(pack.get('anchor_entity_id') or pack.get('source_record_id') or '')
            for attribute in pack.get('direct_attributes', []) or []:
                _add_observation(grouped, owner=owner, label=entity_label, attribute=attribute, source_record_id=pack.get('source_record_id'))
        for fact in fact_values:
            match = _OWNER_PATTERN.search(str(fact or ''))
            if not match:
                continue
            label = match.group('label').strip()
            if not _label_matches(label, target_tokens=target_tokens):
                continue
            _add_observation(grouped, owner=match.group('owner').strip(), label=label, attribute=match.group('attribute').strip(), source_record_id=pack.get('source_record_id'))
    candidates = list(grouped.values())
    candidates.sort(key=lambda item: (-len(item.get('observations', []) or []), str(item.get('owner') or '')))
    return candidates[:max(1, int(limit))]

def _add_observation(grouped: dict[str, dict[str, Any]], *, owner: Any, label: Any, attribute: Any, source_record_id: Any) -> None:
    owner_text = ' '.join(str(owner or '').split())
    label_text = ' '.join(str(label or '').split())
    attribute_text = ' '.join(str(attribute or '').split())
    if not owner_text or not label_text or (not attribute_text):
        return
    key = owner_text.lower()
    candidate = grouped.setdefault(key, {'owner': owner_text, 'owner_label': label_text, 'observations': [], 'source_record_ids': [], 'uses_gold_answer': False})
    candidate['observations'] = _ordered_unique([*candidate['observations'], attribute_text])
    candidate['source_record_ids'] = _ordered_unique([*candidate['source_record_ids'], source_record_id])

def _publish_surface_capacity_evidence(fused: dict[str, Any], *, candidates: list[dict[str, Any]], raw_pack_count: int, target_phrases: list[str], question: str, max_visual_lines: int) -> dict[str, Any]:
    bundle = fused.get('llm_ready_bundle')
    visual_evidence = fused.get('visual_evidence')
    if not isinstance(bundle, dict) or not isinstance(visual_evidence, dict):
        return fused
    lines = [f'kind=surface_capacity_guidance contract={SURFACE_CAPACITY_CONTRACT} compare_all_target_surface_observations=true clutter_is_direct_occupancy_evidence=true object_presence_alone_does_not_prove_no_capacity=true absence_of_objects_does_not_prove_capacity=true for_plural_or_any_queries_one_usable_surface_is_sufficient=true uses_gold_answer=false', *[_surface_candidate_line(index, candidate) for index, candidate in enumerate(candidates, start=1)]] if candidates else []
    if lines:
        existing = list(bundle.get(SPECIALIZED_EVIDENCE_FIELD, []) or [])
        bundle[SPECIALIZED_EVIDENCE_FIELD] = _ordered_unique([*existing, *lines])[:max(1, int(max_visual_lines))]
        structured = deepcopy(bundle.get('structured_graph_context') or {})
        structured['surface_capacity_evidence'] = {'contract': SURFACE_CAPACITY_CONTRACT, 'target_phrases': target_phrases, 'candidate_count': len(candidates), 'plural_or_existential_scope': bool(re.search('\\b(?:any|desks|tables|surfaces)\\b', question, re.IGNORECASE)), 'absence_is_negative_evidence': False, 'uses_gold_answer': False}
        bundle['structured_graph_context'] = structured
    visual_evidence['surface_capacity_contract'] = SURFACE_CAPACITY_CONTRACT
    visual_evidence['surface_capacity_candidate_lines'] = lines
    visual_evidence['supplemental_candidate_packs'] = candidates
    visual_evidence['route_strategy_applied'] = bool(candidates)
    visual_evidence['graph_only_fallback'] = not bool(candidates)
    visual_evidence['handler_fallback_reason'] = '' if candidates else 'no_owner_bound_surface_observations'
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'surface_capacity_raw_candidate_pack_count': raw_pack_count, 'surface_capacity_owner_candidate_count': len(candidates), 'surface_capacity_observation_count': sum((len(candidate.get('observations', []) or []) for candidate in candidates))})
    visual_evidence['stats'] = stats
    return fused

def _surface_candidate_line(index: int, candidate: dict[str, Any]) -> str:
    observations = ' | '.join((_short_text(value, limit=130) for value in candidate.get('observations', []) or []))
    return f"rank={index} kind=owner_bound_surface_observation contract={SURFACE_CAPACITY_CONTRACT} owner={candidate.get('owner')!r} owner_label={candidate.get('owner_label')!r} observations={observations or 'none'} capacity_label_is_not_precomputed=true uses_gold_answer=false"

def _target_phrases(*, evidence_demand: dict[str, Any], question_ir: dict[str, Any] | None) -> list[str]:
    question_ir = question_ir if isinstance(question_ir, dict) else {}
    values = [*(evidence_demand.get('target_terms', []) or []), *(evidence_demand.get('primary_target_terms', []) or [])] or [*(question_ir.get('target_entities', []) or []), *(question_ir.get('target_entity_aliases', []) or [])]
    return _ordered_unique(values)

def _label_matches(label: str, *, target_tokens: set[str]) -> bool:
    label_tokens = _canonical_tokens([label])
    return bool(label_tokens and target_tokens and label_tokens & target_tokens)

def _canonical_tokens(values: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for token in _TOKEN_PATTERN.findall(str(value or '').lower()):
            if token in _STOPWORDS:
                continue
            if len(token) > 4 and token.endswith('ies'):
                token = token[:-3] + 'y'
            elif len(token) > 3 and token.endswith('s'):
                token = token[:-1]
            result.add(token)
    return result

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

def _short_text(value: Any, *, limit: int) -> str:
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else text[:max(1, limit - 3)].rstrip() + '...'
__all__ = ['HANDLER_ID', 'SPECIALIZED_EVIDENCE_FIELD', 'SURFACE_CAPACITY_CONTRACT', 'fuse_surface_capacity_evidence']
