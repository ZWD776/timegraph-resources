"""Isolated text-evidence strategies for OpenEQA existence and state routes."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import STATE_VARIANT_OPEN_CLOSED
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
FROZEN_HANDLER_ID = 'existence_or_state.frozen_v3'
HANDLER_ID = 'existence_or_state.open_closed_consensus_v1'
OPEN_CLOSED_STATE_CONTRACT = 'openeqa_open_closed_state_evidence.v1'
SPECIALIZED_EVIDENCE_FIELD = 'question_route_specialized_evidence_lines'
fuse_existence_or_state_frozen_evidence = fuse_frozen_v3
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_OPEN_PATTERN = re.compile('\\bopen(?:ed)?\\b', re.IGNORECASE)
_CLOSED_PATTERN = re.compile('\\b(?:closed|shut)\\b', re.IGNORECASE)
_STOPWORDS = {'a', 'all', 'an', 'and', 'are', 'at', 'by', 'for', 'from', 'in', 'is', 'of', 'on', 'or', 'the', 'to', 'was', 'were', 'with'}
_STATE_TOKENS = {'close', 'closed', 'open', 'opened', 'shut', 'state'}
_RELATION_TOKENS = {'above', 'behind', 'below', 'beside', 'between', 'inside', 'near', 'next', 'outside', 'under', 'within'}
_QUANTIFIER_TOKENS = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}

def fuse_existence_or_state_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Expose explicit, owner-bound open/closed observations or graph fallback."""
    graph_bundle = deepcopy(graph_result.get('llm_ready_bundle') or {})
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    if str(evidence_demand.get('strategy_variant') or '') != STATE_VARIANT_OPEN_CLOSED:
        return fused
    visual_evidence = fused.get('visual_evidence')
    raw_packs = [dict(pack) for pack in (visual_evidence.get('candidate_packs', []) if isinstance(visual_evidence, dict) else []) if isinstance(pack, dict)]
    target_phrases = _target_phrases(evidence_demand=evidence_demand, question_ir=question_ir)
    required_tokens = _required_binding_tokens(evidence_demand=evidence_demand, target_phrases=target_phrases)
    selected = _select_open_closed_candidate_packs(raw_packs, target_phrases=target_phrases, required_tokens=required_tokens, limit=min(4, max(1, int(max_candidate_packs))))
    consensus = _open_closed_consensus(selected)
    return _publish_open_closed_evidence(fused, graph_bundle=graph_bundle, raw_pack_count=len(raw_packs), selected_packs=selected, consensus=consensus, target_phrases=target_phrases, requested_states=_ordered_unique(evidence_demand.get('state_terms', []) or []), required_tokens=required_tokens, max_visual_lines=max_visual_lines)

def _select_open_closed_candidate_packs(packs: list[dict[str, Any]], *, target_phrases: list[str], required_tokens: set[str], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for pack in packs:
        label = _pack_label(pack)
        binding = _owner_binding(label, target_phrases=target_phrases)
        if not binding['strict_owner_match']:
            continue
        hit_tokens = _binding_hit_tokens(pack, label=label)
        missing = sorted(required_tokens - hit_tokens)
        if missing:
            continue
        observations = _pack_direct_attributes(pack)
        state_observations = [state for text in observations for state in [_observation_state(text)] if state]
        if not state_observations:
            continue
        selected.append({'source_record_id': pack.get('source_record_id'), 'owner_entity_id': pack.get('anchor_entity_id'), 'owner_label': label, 'owner_binding': binding, 'route_binding': {'required_tokens': sorted(required_tokens), 'hit_tokens': sorted(hit_tokens), 'missing_tokens': [], 'complete': True}, 'state_observations': [{'state': state, 'text': text} for text in observations for state in [_observation_state(text)] if state], 'state_values': sorted(set(state_observations)), 'uses_gold_answer': False})
    selected.sort(key=lambda pack: (-int(bool((pack.get('owner_binding') or {}).get('exact_label_match'))), -len(pack.get('state_observations', []) or []), str(pack.get('owner_entity_id') or '')))
    return selected[:max(1, int(limit))]

def _open_closed_consensus(packs: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [dict(observation) for pack in packs for observation in pack.get('state_observations', []) or [] if isinstance(observation, dict)]
    counts = Counter((str(row.get('state') or '') for row in observations))
    counts.pop('', None)
    conflict = len(counts) > 1
    state = next(iter(counts)) if len(counts) == 1 else ''
    return {'contract': 'openeqa_open_closed_consensus.v1', 'available': bool(state) and (not conflict), 'observed_state': state, 'state_support_counts': dict(sorted(counts.items())), 'observation_count': len(observations), 'owner_candidate_count': len(packs), 'conflict_detected': conflict, 'absence_is_negative_evidence': False, 'uses_gold_answer': False}

def _publish_open_closed_evidence(fused: dict[str, Any], *, graph_bundle: dict[str, Any], raw_pack_count: int, selected_packs: list[dict[str, Any]], consensus: dict[str, Any], target_phrases: list[str], requested_states: list[str], required_tokens: set[str], max_visual_lines: int) -> dict[str, Any]:
    visible = bool(consensus.get('available'))
    bundle = fused['llm_ready_bundle']
    visual_evidence = fused['visual_evidence']
    state_lines: list[str] = []
    if visible:
        guidance = 'kind=open_closed_state_guidance explicit_owner_bound_state_only=true answer_yes_only_when_observed_state_matches_requested_state=true absence_is_not_negative_evidence=true uses_gold_answer=false'
        state_lines = [_state_candidate_line(index, pack, consensus=consensus) for index, pack in enumerate(selected_packs, start=1)]
        existing = list(bundle.get(SPECIALIZED_EVIDENCE_FIELD, []) or [])
        bundle[SPECIALIZED_EVIDENCE_FIELD] = _ordered_unique([*existing, guidance, *state_lines])[:max(1, int(max_visual_lines))]
        structured = deepcopy(bundle.get('structured_graph_context') or {})
        structured['open_closed_state_evidence'] = {'contract': OPEN_CLOSED_STATE_CONTRACT, 'target_phrases': target_phrases, 'requested_states': requested_states, 'required_binding_tokens': sorted(required_tokens), 'consensus': consensus, 'uses_gold_answer': False}
        bundle['structured_graph_context'] = structured
    visual_evidence.update({'open_closed_state_contract': OPEN_CLOSED_STATE_CONTRACT, 'open_closed_state_available': visible, 'supplemental_candidate_packs': selected_packs if visible else [], 'state_candidate_lines': state_lines, 'state_consensus': consensus, 'target_phrases': target_phrases, 'requested_states': requested_states, 'required_binding_tokens': sorted(required_tokens), 'route_strategy_applied': visible, 'route_strategy_variant': STATE_VARIANT_OPEN_CLOSED, 'graph_only_fallback': not visible, 'handler_fallback_reason': '' if visible else 'no_unambiguous_owner_bound_open_closed_state', 'open_closed_state_stats': {'raw_candidate_pack_count': raw_pack_count, 'strict_owner_state_candidate_count': len(selected_packs), 'explicit_state_observation_count': int(consensus.get('observation_count') or 0), 'state_conflict_count': int(bool(consensus.get('conflict_detected')))}, 'uses_gold_answer': False})
    return fused

def _state_candidate_line(index: int, pack: dict[str, Any], *, consensus: dict[str, Any]) -> str:
    observations = ' | '.join((_short_text(row.get('text'), limit=100) for row in pack.get('state_observations', []) or [] if isinstance(row, dict)))
    return f"rank={index} kind=owner_locked_open_closed_state contract={OPEN_CLOSED_STATE_CONTRACT} owner_entity={pack.get('owner_entity_id')} owner_label={pack.get('owner_label')!r} observed_state={consensus.get('observed_state')} observations={observations or 'none'} explicit_state_only=true absence_is_not_negative_evidence=true uses_gold_answer=false"

def _state_candidate_fact(index: int, pack: dict[str, Any], *, consensus: dict[str, Any]) -> dict[str, Any]:
    return {'rank': index, 'fact_type': 'owner_locked_open_closed_state', 'source': 'openeqa_visual_runtime_v4', 'source_kind': 'owner_bound_text_state_evidence', 'source_record_id': pack.get('source_record_id'), 'owner_entity_id': pack.get('owner_entity_id'), 'owner_label': pack.get('owner_label'), 'observed_state': consensus.get('observed_state'), 'state_observations': deepcopy(pack.get('state_observations') or []), 'owner_binding': deepcopy(pack.get('owner_binding') or {}), 'route_binding': deepcopy(pack.get('route_binding') or {}), 'uses_gold_answer': False}

def _target_phrases(*, evidence_demand: dict[str, Any], question_ir: dict[str, Any] | None) -> list[str]:
    question_ir = question_ir if isinstance(question_ir, dict) else {}
    route_values = [*(evidence_demand.get('target_terms', []) or []), *(evidence_demand.get('primary_target_terms', []) or [])]
    values = route_values or [*(question_ir.get('target_entities', []) or []), *(question_ir.get('target_entity_aliases', []) or [])]
    return _ordered_unique(values)

def _required_binding_tokens(*, evidence_demand: dict[str, Any], target_phrases: list[str]) -> set[str]:
    values = [*target_phrases, *(evidence_demand.get('anchor_terms', []) or []), *(evidence_demand.get('constraint_terms', []) or [])]
    excluded = _STOPWORDS | _STATE_TOKENS | _RELATION_TOKENS | _QUANTIFIER_TOKENS
    return {token for token in _tokens(' '.join((str(value or '') for value in values))) if token not in excluded and (not token.isdigit())}

def _owner_binding(label: str, *, target_phrases: list[str]) -> dict[str, Any]:
    label_tokens = _tokens(label)
    best: tuple[float, int, int, float, float, str] | None = None
    for phrase in target_phrases:
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens or not label_tokens:
            continue
        sequence = [token for token in _TOKEN_PATTERN.findall(str(phrase).lower()) if token not in _STOPWORDS]
        head = _token_variants(sequence[-1]) if sequence else set()
        overlap = phrase_tokens & label_tokens
        candidate = (float(_normalized_text(phrase) == _normalized_text(label)), int(bool(head & label_tokens)), len(overlap), len(overlap) / len(phrase_tokens), len(overlap) / len(label_tokens), phrase)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
    exact, head_match, overlap, target_coverage, label_coverage, phrase = best or (0.0, 0, 0, 0.0, 0.0, '')
    strict = bool(overlap and head_match and (target_coverage >= 0.5) and (label_coverage >= 0.75))
    return {'contract': 'openeqa_open_closed_owner_binding.v1', 'strict_owner_match': strict, 'owner_label': label, 'matched_target_phrase': phrase, 'exact_label_match': bool(exact), 'target_head_match': bool(head_match), 'overlap_token_count': overlap, 'target_token_coverage': round(target_coverage, 6), 'label_token_coverage': round(label_coverage, 6), 'uses_attribute_text_for_owner_binding': False, 'uses_gold_answer': False}

def _binding_hit_tokens(pack: dict[str, Any], *, label: str) -> set[str]:
    values = [label, *(pack.get('target_hits', []) or []), *(pack.get('constraint_hits', []) or []), *(pack.get('question_hits', []) or []), *(pack.get('direct_relations', []) or [])]
    return _tokens(' '.join((str(value or '') for value in values)))

def _pack_label(pack: dict[str, Any]) -> str:
    visual_object = pack.get('visual_object')
    visual_object = visual_object if isinstance(visual_object, dict) else {}
    return str(pack.get('entity_label') or pack.get('graph_label') or pack.get('visual_label') or visual_object.get('label') or '').strip()

def _pack_direct_attributes(pack: dict[str, Any]) -> list[str]:
    values = list(pack.get('direct_attributes', []) or [])
    visual_object = pack.get('visual_object')
    visual_object = visual_object if isinstance(visual_object, dict) else {}
    for fact in visual_object.get('hsgg_visual_facts', []) or []:
        text = str(fact or '')
        marker = 'attr=raw_visual_attributes='
        if marker in text:
            values.append(text.partition(marker)[2])
    return _ordered_unique(values)

def _observation_state(text: Any) -> str:
    value = str(text or '')
    has_open = bool(_OPEN_PATTERN.search(value))
    has_closed = bool(_CLOSED_PATTERN.search(value))
    if has_open == has_closed:
        return ''
    return 'open' if has_open else 'closed'

def _tokens(value: Any) -> set[str]:
    raw = set(_TOKEN_PATTERN.findall(str(value or '').lower()))
    result: set[str] = set()
    for token in raw:
        if token in _STOPWORDS:
            continue
        result.update(_token_variants(token))
    return result

def _token_variants(token: str) -> set[str]:
    result = {token}
    if token == 'doorway':
        result.add('door')
    if len(token) > 4 and token.endswith('ies'):
        result.add(token[:-3] + 'y')
    elif len(token) > 3 and token.endswith('s'):
        result.add(token[:-1])
    return result

def _normalized_text(value: Any) -> str:
    return ' '.join(sorted(_tokens(value)))

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
__all__ = ['FROZEN_HANDLER_ID', 'HANDLER_ID', 'OPEN_CLOSED_STATE_CONTRACT', 'SPECIALIZED_EVIDENCE_FIELD', 'fuse_existence_or_state_frozen_evidence', 'fuse_existence_or_state_evidence']
