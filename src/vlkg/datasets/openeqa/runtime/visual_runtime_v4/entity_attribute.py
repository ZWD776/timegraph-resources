"""Experimental owner-locked OpenEQA entity-attribute evidence.

This runtime only reorganizes text evidence already produced by frozen v3. It
does not change retrieval, inspect pixels, or use reference answers. The v2
candidate is retained for audit reproducibility but is not registered in the
formal v4 dispatcher because its full-route acceptance gate did not pass.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'entity_attribute.owner_locked_consensus_v2'
OWNER_ATTRIBUTE_CONTRACT = 'openeqa_owner_locked_attribute_evidence.v2'
SPECIALIZED_EVIDENCE_FIELD = 'question_route_specialized_evidence_lines'
_STRATEGY_VARIANT = 'owner_attribute'
_MAX_OWNER_CANDIDATES = 3
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'by', 'for', 'from', 'has', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'this', 'to', 'was', 'were', 'with'}
_RELATION_TOKENS = {'above', 'behind', 'below', 'beside', 'between', 'inside', 'near', 'next', 'outside', 'under', 'within'}
_CARDINAL_QUANTIFIER_TOKENS = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
_VISUAL_LINE_FIELDS = ('visual_faiss_evidence_lines', 'visual_faiss_ger_evidence_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_unified_candidate_lines', 'visual_faiss_object_local_fact_lines', 'visual_faiss_local_subgraph_fact_lines', 'visual_faiss_role_aware_lines', 'visual_faiss_role_aware_aux_lines', 'visual_faiss_evidence_layer_lines', 'frame_faiss_evidence_lines', 'merged_frame_faiss_evidence_lines')

def fuse_entity_attribute_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Publish only attribute evidence whose owner label matches the target."""
    graph_bundle = deepcopy(graph_result.get('llm_ready_bundle') or {})
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    if str(evidence_demand.get('strategy_variant') or '') != _STRATEGY_VARIANT:
        return fused
    visual_evidence = fused.get('visual_evidence')
    bundle = fused.get('llm_ready_bundle')
    if not isinstance(visual_evidence, dict) or not isinstance(bundle, dict):
        return fused
    raw_packs = [dict(pack) for pack in visual_evidence.get('candidate_packs', []) or [] if isinstance(pack, dict)]
    target_phrases = _target_phrases(evidence_demand=evidence_demand, question_ir=question_ir)
    required_binding_tokens = _required_route_binding_tokens(evidence_demand=evidence_demand)
    selected = _select_owner_candidate_packs(raw_packs, target_phrases=target_phrases, required_binding_tokens=required_binding_tokens, limit=min(_MAX_OWNER_CANDIDATES, max(1, int(max_candidate_packs))))
    attribute_dimensions = _ordered_unique(evidence_demand.get('attribute_dimensions', []) or [])
    for pack in selected:
        pack['owner_attribute_consensus'] = _attribute_consensus(pack, target_phrases=target_phrases, attribute_dimensions=attribute_dimensions)
    return _publish_owner_attribute_evidence(fused, graph_bundle=graph_bundle, raw_packs=raw_packs, selected_packs=selected, target_phrases=target_phrases, attribute_dimensions=attribute_dimensions, required_binding_tokens=required_binding_tokens, max_visual_lines=max_visual_lines)

def _select_owner_candidate_packs(packs: list[dict[str, Any]], *, target_phrases: list[str], required_binding_tokens: set[str] | None=None, limit: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source in packs:
        binding = _owner_binding(source, target_phrases=target_phrases)
        if not binding['strict_owner_match']:
            continue
        route_binding = _route_binding(source, required_tokens=required_binding_tokens or set())
        if not route_binding['complete']:
            continue
        pack = deepcopy(source)
        pack['owner_binding'] = binding
        pack['owner_route_binding'] = route_binding
        candidates.append(pack)
    if not candidates:
        return []
    best_overlap = max((int(pack['owner_binding']['overlap_token_count']) for pack in candidates))
    candidates = [pack for pack in candidates if int(pack['owner_binding']['overlap_token_count']) == best_overlap]
    candidates.sort(key=_owner_pack_sort_key)
    return candidates[:max(1, int(limit))]

def _owner_binding(pack: dict[str, Any], *, target_phrases: list[str]) -> dict[str, Any]:
    label = str(pack.get('entity_label') or '').strip()
    label_tokens = _tokens(label)
    best: tuple[float, int, int, float, float, str] | None = None
    for phrase in target_phrases:
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens or not label_tokens:
            continue
        phrase_sequence = [token for token in _TOKEN_PATTERN.findall(str(phrase).lower()) if token not in _STOPWORDS]
        head_tokens = _token_variants(phrase_sequence[-1]) if phrase_sequence else set()
        head_match = bool(head_tokens & label_tokens)
        overlap = phrase_tokens & label_tokens
        target_coverage = len(overlap) / len(phrase_tokens)
        label_coverage = len(overlap) / len(label_tokens)
        exact = float(_normalized_text(phrase) == _normalized_text(label))
        candidate = (exact, int(head_match), len(overlap), target_coverage, label_coverage, phrase)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
    exact, head_match, overlap_count, target_coverage, label_coverage, phrase = best or (0.0, 0, 0, 0.0, 0.0, '')
    strict = bool(overlap_count and head_match and (target_coverage >= 0.5) and (label_coverage >= 0.75))
    return {'contract': 'openeqa_owner_label_binding.v1', 'strict_owner_match': strict, 'owner_label': label, 'matched_target_phrase': phrase, 'exact_label_match': bool(exact), 'target_head_match': bool(head_match), 'overlap_token_count': overlap_count, 'target_token_coverage': round(target_coverage, 6), 'label_token_coverage': round(label_coverage, 6), 'uses_attribute_text_for_owner_binding': False, 'uses_gold_answer': False}

def _owner_pack_sort_key(pack: dict[str, Any]) -> tuple[Any, ...]:
    binding = pack.get('owner_binding') or {}
    role = str(pack.get('candidate_role') or '')
    role_priority = 0 if role == 'target_direct' else 1 if role == 'target_query' else 2
    final_key = pack.get('final_rank_key') or {}
    return (-int(bool(binding.get('exact_label_match'))), -float(binding.get('target_token_coverage') or 0.0), -float(binding.get('label_token_coverage') or 0.0), -len(pack.get('constraint_hits', []) or []), role_priority, int(final_key['answerability_tier']) if final_key.get('answerability_tier') is not None else 9, -float(pack.get('retrieval_score') or 0.0), str(pack.get('anchor_entity_id') or ''))

def _attribute_consensus(pack: dict[str, Any], *, target_phrases: list[str], attribute_dimensions: list[str]) -> dict[str, Any]:
    observations = _ordered_unique(pack.get('direct_attributes', []) or [])
    excluded = set().union(*[_tokens(pack.get('entity_label'))] + [_tokens(value) for value in target_phrases] + [_tokens(value) for value in attribute_dimensions])
    support: Counter[str] = Counter()
    for observation in observations:
        descriptor_tokens = {token for token in _tokens(observation) if token not in excluded and token not in _STOPWORDS and (len(token) > 2)}
        support.update(descriptor_tokens)
    consensus = [{'token': token, 'support_count': count} for token, count in sorted(support.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:8]
    conflict = len(observations) > 1
    if not observations:
        state = 'missing'
    elif len(observations) == 1:
        state = 'single_observation'
    elif consensus:
        state = 'partial_token_consensus_with_conflict'
    else:
        state = 'conflicting_observations_no_consensus'
    return {'contract': 'openeqa_owner_attribute_consensus.v1', 'attribute_dimensions': attribute_dimensions, 'observation_count': len(observations), 'observations': observations[:8], 'agreement_state': state, 'consensus_descriptor_tokens': consensus, 'conflict_detected': conflict, 'first_observation_is_not_privileged': True, 'graph_evidence_breaks_unresolved_conflicts': True, 'uses_gold_answer': False}

def _publish_owner_attribute_evidence(fused: dict[str, Any], *, graph_bundle: dict[str, Any], raw_packs: list[dict[str, Any]], selected_packs: list[dict[str, Any]], target_phrases: list[str], attribute_dimensions: list[str], required_binding_tokens: set[str] | None=None, max_visual_lines: int) -> dict[str, Any]:
    bundle = fused['llm_ready_bundle']
    visual_evidence = fused['visual_evidence']
    owner_lines = [_owner_attribute_line(index, pack) for index, pack in enumerate(selected_packs, start=1)]
    owner_facts = [_owner_attribute_fact(index, pack) for index, pack in enumerate(selected_packs, start=1)]
    guidance = 'kind=owner_attribute_guidance strict_owner_label_binding=true attributes_belong_only_to_owner=true compare_all_observations=true first_observation_is_not_privileged=true graph_evidence_breaks_unresolved_conflicts=true uses_gold_answer=false'
    visible_owner_lines = [guidance, *owner_lines] if owner_lines else []
    if selected_packs:
        existing = list(bundle.get(SPECIALIZED_EVIDENCE_FIELD, []) or [])
        bundle[SPECIALIZED_EVIDENCE_FIELD] = _ordered_unique([*existing, *visible_owner_lines])[:max(1, int(max_visual_lines))]
        structured = deepcopy(bundle.get('structured_graph_context') or {})
        structured['owner_attribute_evidence'] = {'contract': OWNER_ATTRIBUTE_CONTRACT, 'target_phrases': target_phrases, 'attribute_dimensions': attribute_dimensions, 'required_route_binding_tokens': sorted(required_binding_tokens or set()), 'owner_candidate_count': len(selected_packs), 'graph_fallback_preserved': True, 'uses_gold_answer': False}
        bundle['structured_graph_context'] = structured
    visual_evidence.update({'owner_attribute_contract': OWNER_ATTRIBUTE_CONTRACT, 'owner_attribute_available': bool(selected_packs), 'supplemental_candidate_packs': selected_packs, 'owner_attribute_candidate_lines': owner_lines, 'owner_attribute_candidate_facts': owner_facts, 'owner_attribute_target_phrases': target_phrases, 'owner_attribute_dimensions': attribute_dimensions, 'owner_attribute_required_route_binding_tokens': sorted(required_binding_tokens or set()), 'route_strategy_applied': bool(selected_packs), 'route_strategy_variant': _STRATEGY_VARIANT, 'graph_only_fallback': not bool(selected_packs), 'handler_fallback_reason': '' if selected_packs else 'no_strict_owner_attribute_candidates', 'uses_gold_answer': False})
    stats = dict(visual_evidence.get('stats') or {})
    stats.update({'owner_attribute_raw_candidate_pack_count': len(raw_packs), 'owner_attribute_strict_owner_candidate_count': len(selected_packs), 'owner_attribute_unrelated_candidate_suppressed_count': len(raw_packs) - len(selected_packs), 'owner_attribute_graph_fallback_count': len(bundle.get('evidence_lines', []) or []), 'owner_attribute_conflict_candidate_count': sum((bool((pack.get('owner_attribute_consensus') or {}).get('conflict_detected')) for pack in selected_packs))})
    visual_evidence['stats'] = stats
    return fused

def _owner_attribute_line(index: int, pack: dict[str, Any]) -> str:
    binding = pack.get('owner_binding') or {}
    consensus = pack.get('owner_attribute_consensus') or {}
    observations = ' | '.join((_short_text(value, limit=110) for value in consensus.get('observations', []) or []))
    descriptors = ','.join((f"{item.get('token')}:{item.get('support_count')}" for item in consensus.get('consensus_descriptor_tokens', []) or []))
    return f"rank={index} kind=owner_locked_attribute_candidate contract={OWNER_ATTRIBUTE_CONTRACT} owner_entity={pack.get('anchor_entity_id')} owner_label={pack.get('entity_label')!r} matched_target={binding.get('matched_target_phrase')!r} owner_overlap={binding.get('overlap_token_count')} route_binding_complete={str(bool((pack.get('owner_route_binding') or {}).get('complete'))).lower()} attribute_dimensions={consensus.get('attribute_dimensions') or []} agreement={consensus.get('agreement_state')} conflict={str(bool(consensus.get('conflict_detected'))).lower()} consensus_descriptors={descriptors or 'none'} observations={observations or 'none'} first_observation_is_not_privileged=true uses_gold_answer=false"

def _owner_attribute_fact(index: int, pack: dict[str, Any]) -> dict[str, Any]:
    consensus = deepcopy(pack.get('owner_attribute_consensus') or {})
    return {'rank': index, 'fact_type': 'owner_locked_attribute_candidate', 'source': 'openeqa_visual_runtime_v4', 'source_kind': 'owner_locked_attribute_evidence', 'source_record_id': pack.get('source_record_id'), 'owner_entity_id': pack.get('anchor_entity_id'), 'owner_label': pack.get('entity_label'), 'owner_binding': deepcopy(pack.get('owner_binding') or {}), 'route_binding': deepcopy(pack.get('owner_route_binding') or {}), 'attribute_consensus': consensus, 'binding_strength': 'strict_owner_label_attribute_bound', 'uses_gold_answer': False}

def _target_phrases(*, evidence_demand: dict[str, Any], question_ir: dict[str, Any] | None) -> list[str]:
    question_ir = question_ir if isinstance(question_ir, dict) else {}
    route_values: list[Any] = [*(evidence_demand.get('target_terms', []) or []), *(evidence_demand.get('primary_target_terms', []) or [])]
    values = route_values or [*(question_ir.get('target_entities', []) or []), *(question_ir.get('target_entity_aliases', []) or [])]
    return _ordered_unique(values)

def _required_route_binding_tokens(*, evidence_demand: dict[str, Any]) -> set[str]:
    target = _canonical_value_tokens(evidence_demand.get('target_terms', []) or [])
    anchor = _canonical_value_tokens(evidence_demand.get('anchor_terms', []) or [])
    constraints = _canonical_value_tokens(evidence_demand.get('constraint_terms', []) or [])
    residual_constraints = constraints - target - anchor - _RELATION_TOKENS
    return {token for token in target | anchor | residual_constraints if token not in _CARDINAL_QUANTIFIER_TOKENS and (not token.isdigit())}

def _route_binding(pack: dict[str, Any], *, required_tokens: set[str]) -> dict[str, Any]:
    hit_values: list[Any] = [pack.get('entity_label'), *(pack.get('target_hits', []) or []), *(pack.get('constraint_hits', []) or [])]
    hit_tokens = _canonical_value_tokens(hit_values)
    missing = sorted(required_tokens - hit_tokens)
    return {'contract': 'openeqa_owner_route_binding.v1', 'required_tokens': sorted(required_tokens), 'hit_tokens': sorted(hit_tokens), 'missing_tokens': missing, 'complete': not bool(missing), 'uses_gold_answer': False}

def _tokens(value: Any) -> set[str]:
    tokens = set(_TOKEN_PATTERN.findall(str(value or '').lower()))
    result = set(tokens)
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            result.add(token[:-3] + 'y')
        elif len(token) > 3 and token.endswith('s'):
            result.add(token[:-1])
    return {token for token in result if token not in _STOPWORDS}

def _token_variants(token: str) -> set[str]:
    result = {token}
    if len(token) > 4 and token.endswith('ies'):
        result.add(token[:-3] + 'y')
    elif len(token) > 3 and token.endswith('s'):
        result.add(token[:-1])
    return result

def _canonical_value_tokens(values: Iterable[Any]) -> set[str]:
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
__all__ = ['HANDLER_ID', 'OWNER_ATTRIBUTE_CONTRACT', 'SPECIALIZED_EVIDENCE_FIELD', 'fuse_entity_attribute_evidence']
