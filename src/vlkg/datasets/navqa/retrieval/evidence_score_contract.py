from __future__ import annotations
import re
from typing import Any, Callable
SCORE_CONTRACT_ID = 'visual_candidate_pack_score_contract_v1'
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'how', 'i', 'in', 'is', 'it', 'me', 'near', 'of', 'on', 'or', 'that', 'the', 'there', 'this', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'with', 'you'}
_SEMANTIC_TARGET_THRESHOLD = 0.5

def apply_candidate_pack_score_contract(packs: list[dict[str, Any]], *, question: str, question_ir: dict[str, Any], question_route: dict[str, Any] | None=None, retrieval_window: dict[str, Any] | None=None, text_encoder: Callable[[list[str]], list[Any]] | None=None) -> list[dict[str, Any]]:
    """Attach a normalized, auditable score contract to candidate packs.

    The contract keeps retrieval, binding, semantic alignment, and answerability
    as separate fields. It does not use gold answers, question ids, seq ids, or a
    fixed domain wordlist.
    """
    if not packs:
        return []
    question_route = question_route if isinstance(question_route, dict) else {}
    route_id = str(question_route.get('route_id') or '').strip()
    target_texts = _target_texts(question_ir)
    candidate_texts_by_pack = [_candidate_texts(pack) for pack in packs]
    semantic_matches = _semantic_matches(target_texts=target_texts, candidate_texts_by_pack=candidate_texts_by_pack, text_encoder=text_encoder)
    scored: list[dict[str, Any]] = []
    for pack, semantic in zip(packs, semantic_matches, strict=True):
        updated = dict(pack)
        legacy_score = updated.pop('score', None)
        if legacy_score is not None:
            updated['legacy_entity_pack_score'] = legacy_score
        retrieval_score = _retrieval_score(updated)
        binding = _binding_components(updated, question_ir=question_ir)
        binding_score = round(sum(binding.values()), 6)
        semantic_score = round(float(semantic['score']), 6)
        window_alignment = _question_window_alignment(updated, retrieval_window=retrieval_window) if route_id == 'entity_attribute' else None
        window_alignment_score = round(float(window_alignment.get('score') or 0.0), 6) if isinstance(window_alignment, dict) else 0.0
        answerability = _answerability(updated, question=question, question_ir=question_ir, question_route=question_route, binding_components=binding, semantic_score=semantic_score)
        final_rank_key = {'answerability_tier': answerability['answerability_tier'], 'binding_score': binding_score, 'semantic_score': semantic_score, 'retrieval_score': retrieval_score, 'source_priority': _source_priority(updated), 'entity_id': str(updated.get('anchor_entity_id') or '')}
        if window_alignment is not None:
            final_rank_key['question_window_alignment_score'] = window_alignment_score
        updated.update({'score_contract': SCORE_CONTRACT_ID, 'candidate_unit': 'candidate_pack', 'target_terms': target_texts, 'semantic_match_text': semantic['match_text'], 'semantic_query_text': semantic['query_text'], 'semantic_score': semantic_score, 'semantic_alignment_provider': 'siglip_text' if semantic['provider_available'] else 'unavailable', 'binding_components': binding, 'binding_score': binding_score, 'retrieval_score': retrieval_score, 'answerability_score': answerability['answerability_score'], 'answerability_tier': answerability['answerability_tier'], 'answerability_reason': answerability['answerability_reason'], 'final_rank_key': final_rank_key, 'source_contributions': _source_contributions(updated), 'question_route_id': str(question_route.get('route_id') or ''), 'question_route_evidence_policy': str(question_route.get('evidence_policy') or ''), 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False, 'llm_semantic_alignment': False})
        if window_alignment is not None:
            updated['question_window_alignment'] = window_alignment
            updated['question_window_alignment_score'] = window_alignment_score
            updated['source_contributions']['question_window_alignment'] = window_alignment
        scored.append(updated)
    scored.sort(key=_score_contract_sort_key)
    return scored

def score_contract_pack_line(*, index: int, pack: dict[str, Any]) -> str:
    return f"rank={index} kind=candidate_pack contract={SCORE_CONTRACT_ID} tier={pack.get('answerability_tier')} retrieval_score:{_format_float(pack.get('retrieval_score'))} binding_score:{_format_float(pack.get('binding_score'))} semantic_score:{_format_float(pack.get('semantic_score'))} answerability_score:{_format_float(pack.get('answerability_score'))} entity={pack.get('anchor_entity_id')} label={pack.get('entity_label')!r} role={pack.get('candidate_role')} semantic_match={_short_text(pack.get('semantic_match_text'), limit=72)!r} reason={_short_text(pack.get('answerability_reason'), limit=96)!r} :: {pack.get('fact_text')}"

def score_contract_aggregation_line(*, index: int, pack: dict[str, Any]) -> str:
    components = pack.get('binding_components', {})
    component_text = ','.join((f'{key}:{value:g}' for key, value in components.items() if value))
    return f"group=candidate_pack:{index} contract={SCORE_CONTRACT_ID} tier={pack.get('answerability_tier')} entity={pack.get('anchor_entity_id')} label={pack.get('entity_label')!r} binding_components={component_text} semantic_match={_short_text(pack.get('semantic_match_text'), limit=72)!r} answerability={_short_text(pack.get('answerability_reason'), limit=96)}"

def _score_contract_sort_key(pack: dict[str, Any]) -> tuple[Any, ...]:
    key = pack.get('final_rank_key', {})
    if not isinstance(key, dict):
        key = {}
    tier = _optional_int(key.get('answerability_tier'))
    source_priority = _optional_int(key.get('source_priority'))
    return (tier if tier is not None else 9, -float(key.get('question_window_alignment_score') or 0.0), -float(key.get('binding_score') or 0.0), -float(key.get('semantic_score') or 0.0), -float(key.get('retrieval_score') or 0.0), source_priority if source_priority is not None else 9, str(key.get('entity_id') or ''))

def _retrieval_score(pack: dict[str, Any]) -> float:
    anchors = pack.get('visual_anchors', []) or []
    values = []
    for anchor in anchors:
        if isinstance(anchor, dict):
            value = _optional_float(anchor.get('similarity'))
            if value is not None:
                values.append(value)
    breakdown = pack.get('ranking_score_breakdown', {})
    if isinstance(breakdown, dict):
        visual_similarity = _optional_float(breakdown.get('visual_similarity'))
        if visual_similarity is not None:
            values.append(visual_similarity / 8.0)
    return round(max(values, default=0.0), 6)

def _binding_components(pack: dict[str, Any], *, question_ir: dict[str, Any]) -> dict[str, float]:
    role = str(pack.get('candidate_role') or '')
    target_hits = list(pack.get('target_hits', []) or [])
    attribute_hits = list(pack.get('attribute_hits', []) or [])
    constraint_hits = list(pack.get('constraint_hits', []) or [])
    direct_attributes = list(pack.get('direct_attributes', []) or [])
    direct_relations = list(pack.get('direct_relations', []) or [])
    time_span = pack.get('time_span', {}) if isinstance(pack.get('time_span'), dict) else {}
    attribute_needed = bool(str(question_ir.get('attribute_dimension') or '').strip())
    return {'target_exact': 4.0 if role == 'target_direct' or target_hits else 0.0, 'target_query_role': 1.5 if role == 'target_query' else 0.0, 'direct_attribute': min(3.0, len(direct_attributes) * (1.0 if attribute_needed else 0.45)), 'attribute_term': min(1.2, len(attribute_hits) * 0.6), 'direct_relation': min(2.4, len(direct_relations) * 0.6), 'constraint': min(2.0, len(constraint_hits) * 0.45), 'time_span': 0.6 if time_span else 0.0}

def _answerability(pack: dict[str, Any], *, question: str, question_ir: dict[str, Any], question_route: dict[str, Any], binding_components: dict[str, float], semantic_score: float) -> dict[str, Any]:
    del question
    route_id = str(question_route.get('route_id') or '').strip()
    direct_target = bool(binding_components.get('target_exact') or binding_components.get('target_query_role'))
    semantic_target = semantic_score >= _SEMANTIC_TARGET_THRESHOLD
    has_direct_attribute = bool(pack.get('direct_attributes'))
    has_direct_relation = bool(pack.get('direct_relations'))
    has_constraint = bool(pack.get('constraint_hits'))
    has_time_span = bool(pack.get('time_span'))
    attribute_needed = bool(str(question_ir.get('attribute_dimension') or '').strip())
    has_requested_slot = has_direct_attribute if attribute_needed else has_direct_attribute or has_direct_relation
    route_answerability = _route_answerability(route_id=route_id, direct_target=direct_target, semantic_target=semantic_target, has_direct_attribute=has_direct_attribute, has_direct_relation=has_direct_relation, has_constraint=has_constraint, has_time_span=has_time_span)
    if route_answerability is not None:
        return route_answerability
    answerability_score = 0.0
    if direct_target:
        answerability_score += 4.0
    elif semantic_target:
        answerability_score += 2.5
    if has_requested_slot:
        answerability_score += 3.0
    if has_direct_relation:
        answerability_score += 1.0
    if has_constraint:
        answerability_score += 0.8
    if (direct_target or semantic_target) and has_requested_slot:
        tier = 0 if direct_target else 1
        reason = 'target candidate has direct answer-bearing evidence'
    elif direct_target or semantic_target:
        tier = 2
        reason = 'target candidate found but direct answer slot is weak'
    elif has_constraint:
        tier = 3
        reason = 'constraint or context candidate without target binding'
    else:
        tier = 4
        reason = 'fallback visual candidate'
    return {'answerability_score': round(answerability_score, 6), 'answerability_tier': tier, 'answerability_reason': reason}

def _route_answerability(*, route_id: str, direct_target: bool, semantic_target: bool, has_direct_attribute: bool, has_direct_relation: bool, has_constraint: bool, has_time_span: bool) -> dict[str, Any] | None:
    targetish = direct_target or semantic_target
    target_score = 4.0 if direct_target else 2.5 if semantic_target else 0.0
    if route_id == 'temporal_interval':
        score = target_score + (3.0 if has_time_span else 0.0) + (1.0 if has_constraint else 0.0)
        if targetish and has_time_span:
            tier = 0
            reason = 'route temporal_interval: target candidate has usable time span'
        elif has_time_span:
            tier = 2
            reason = 'route temporal_interval: span exists but target binding is weak'
        else:
            tier = 4
            reason = 'route temporal_interval: no usable start/end span'
        return _answerability_payload(score, tier, reason)
    if route_id == 'temporal_point':
        score = target_score + (2.0 if has_time_span else 0.0) + (1.0 if has_constraint else 0.0)
        if targetish and (has_time_span or has_constraint):
            tier = 0
            reason = 'route temporal_point: target/event candidate can anchor time'
        elif has_time_span or has_constraint:
            tier = 2
            reason = 'route temporal_point: time/event evidence exists but target binding is weak'
        else:
            tier = 4
            reason = 'route temporal_point: no clear time anchor'
        return _answerability_payload(score, tier, reason)
    if route_id == 'count_set':
        score = target_score + (1.5 if has_constraint else 0.0) + (0.8 if has_time_span else 0.0)
        if targetish:
            tier = 1 if has_constraint or has_time_span else 2
            reason = 'route count_set: candidate contributes to filtered entity set'
        elif has_constraint:
            tier = 3
            reason = 'route count_set: context candidate may define count condition'
        else:
            tier = 4
            reason = 'route count_set: weak counted-entity binding'
        return _answerability_payload(score, tier, reason)
    if route_id == 'ocr_logo_text':
        score = target_score + (4.0 if has_direct_attribute else 0.0) + (0.8 if has_constraint else 0.0)
        if targetish and has_direct_attribute:
            tier = 0
            reason = 'route ocr_logo_text: target has direct text/logo/attribute evidence'
        elif has_direct_attribute:
            tier = 1
            reason = 'route ocr_logo_text: text-like evidence exists but target binding is weak'
        else:
            tier = 4
            reason = 'route ocr_logo_text: no direct text/logo evidence'
        return _answerability_payload(score, tier, reason)
    if route_id == 'entity_relation_spatial':
        score = target_score + (3.0 if has_direct_relation else 0.0) + (1.2 if has_constraint else 0.0)
        if targetish and (has_direct_relation or has_constraint):
            tier = 0
            reason = 'route entity_relation_spatial: target has relation/spatial support'
        elif has_direct_relation or has_constraint:
            tier = 2
            reason = 'route entity_relation_spatial: relation evidence exists but target binding is weak'
        else:
            tier = 4
            reason = 'route entity_relation_spatial: no relation/spatial support'
        return _answerability_payload(score, tier, reason)
    if route_id == 'ego_motion_side_turn':
        score = target_score + (2.0 if has_direct_relation else 0.0) + (1.2 if has_time_span else 0.0)
        if has_direct_relation or has_time_span:
            tier = 2 if not targetish else 1
            reason = 'route ego_motion_side_turn: uses relation/time evidence as ego-motion proxy'
        else:
            tier = 4
            reason = 'route ego_motion_side_turn: missing pose/heading or proxy evidence'
        return _answerability_payload(score, tier, reason)
    if route_id == 'binary_existence':
        score = target_score + (1.5 if has_constraint else 0.0) + (1.0 if has_direct_relation else 0.0)
        if targetish or has_constraint:
            tier = 1
            reason = 'route binary_existence: evidence supports existence/event decision'
        else:
            tier = 4
            reason = 'route binary_existence: weak existence evidence'
        return _answerability_payload(score, tier, reason)
    return None

def _answerability_payload(score: float, tier: int, reason: str) -> dict[str, Any]:
    return {'answerability_score': round(score, 6), 'answerability_tier': tier, 'answerability_reason': reason}

def _semantic_matches(*, target_texts: list[str], candidate_texts_by_pack: list[list[str]], text_encoder: Callable[[list[str]], list[Any]] | None) -> list[dict[str, Any]]:
    if not target_texts:
        return [{'score': 0.0, 'query_text': '', 'match_text': '', 'provider_available': False} for _ in candidate_texts_by_pack]
    all_texts = _ordered_unique([*target_texts, *[text for texts in candidate_texts_by_pack for text in texts]])
    embeddings = _text_embeddings(all_texts, text_encoder=text_encoder)
    provider_available = bool(embeddings)
    matches: list[dict[str, Any]] = []
    for candidate_texts in candidate_texts_by_pack:
        best_score = 0.0
        best_query = ''
        best_match = ''
        for target_text in target_texts:
            target_vector = embeddings.get(target_text)
            if target_vector is None:
                continue
            for candidate_text in candidate_texts:
                candidate_vector = embeddings.get(candidate_text)
                if candidate_vector is None:
                    continue
                similarity = _cosine_similarity(target_vector, candidate_vector)
                if similarity > best_score or not best_query:
                    best_score = similarity
                    best_query = target_text
                    best_match = candidate_text
        matches.append({'score': best_score, 'query_text': best_query, 'match_text': best_match, 'provider_available': provider_available})
    return matches

def _text_embeddings(texts: list[str], *, text_encoder: Callable[[list[str]], list[Any]] | None) -> dict[str, Any]:
    cleaned = _ordered_unique((_semantic_embedding_text(text) for text in texts))
    if not cleaned or text_encoder is None:
        return {}
    rendered = [f'a photo of {text}' for text in cleaned]
    vectors = text_encoder(rendered)
    return {text: vector for text, vector in zip(cleaned, vectors, strict=False) if vector is not None}

def _candidate_texts(pack: dict[str, Any]) -> list[str]:
    values: list[Any] = [pack.get('entity_label'), *(pack.get('direct_attributes', []) or []), *(pack.get('direct_relations', []) or [])]
    for anchor in pack.get('visual_anchors', []) or []:
        if isinstance(anchor, dict):
            values.append(anchor.get('summary'))
    return _ordered_unique((_semantic_embedding_text(value) for value in values))

def _target_texts(question_ir: dict[str, Any]) -> list[str]:
    targets = _ordered_unique((_semantic_embedding_text(value) for value in _iter_text_values(question_ir.get('target_entities'))))
    if targets:
        return targets
    return _ordered_unique((_semantic_embedding_text(value) for value in _iter_text_values(question_ir.get('target_state'))))

def _source_priority(pack: dict[str, Any]) -> int:
    role = str(pack.get('candidate_role') or '')
    if role == 'target_direct':
        return 0
    if role == 'target_query':
        return 1
    if role == 'attribute_support':
        return 2
    if role == 'visual_recall_candidate':
        return 3
    if role == 'constraint_context':
        return 4
    return 5

def _source_contributions(pack: dict[str, Any]) -> dict[str, Any]:
    return {'source': pack.get('source'), 'source_kind': pack.get('source_kind'), 'legacy_entity_pack_score': pack.get('legacy_entity_pack_score'), 'ranking_score_breakdown': dict(pack.get('ranking_score_breakdown', {}) or {}), 'visual_anchor_count': len(pack.get('visual_anchors', []) or []), 'direct_attribute_count': len(pack.get('direct_attributes', []) or []), 'direct_relation_count': len(pack.get('direct_relations', []) or []), 'constraint_count': len(pack.get('constraint_hits', []) or [])}

def _question_window_alignment(pack: dict[str, Any], *, retrieval_window: dict[str, Any] | None) -> dict[str, Any]:
    window_bounds = _retrieval_window_bounds(retrieval_window)
    candidate_bounds = _candidate_time_bounds(pack)
    if window_bounds is None:
        return {'applied': False, 'reason': 'missing_or_disabled_question_retrieval_window', 'score': 0.0}
    if candidate_bounds is None:
        return {'applied': False, 'reason': 'candidate_missing_time_span', 'window_start_ts': window_bounds[0], 'window_end_ts': window_bounds[1], 'score': 0.0}
    window_start, window_end = window_bounds
    candidate_start, candidate_end = candidate_bounds
    overlap = max(0.0, min(window_end, candidate_end) - max(window_start, candidate_start))
    overlaps = overlap > 0.0 or (candidate_start <= window_end and candidate_end >= window_start)
    if overlaps:
        window_duration = max(1.0, window_end - window_start)
        overlap_ratio = min(1.0, overlap / window_duration)
        score = 4.0 + overlap_ratio
        nearest_gap = 0.0
        reason = 'candidate_time_span_overlaps_question_window'
    else:
        if candidate_end < window_start:
            nearest_gap = window_start - candidate_end
        else:
            nearest_gap = candidate_start - window_end
        if nearest_gap <= 30.0:
            score = 2.5
        elif nearest_gap <= 120.0:
            score = 1.5
        elif nearest_gap <= 300.0:
            score = 0.5
        else:
            score = 0.0
        overlap_ratio = 0.0
        reason = 'candidate_time_span_near_question_window' if score else 'candidate_time_span_far_from_question_window'
    return {'applied': True, 'window_start_ts': round(window_start, 6), 'window_end_ts': round(window_end, 6), 'candidate_start_ts': round(candidate_start, 6), 'candidate_end_ts': round(candidate_end, 6), 'overlaps': overlaps, 'overlap_seconds': round(overlap, 6), 'overlap_ratio': round(overlap_ratio, 6), 'nearest_gap_seconds': round(nearest_gap, 6), 'score': round(score, 6), 'reason': reason, 'uses_gold_answers': False}

def _retrieval_window_bounds(retrieval_window: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(retrieval_window, dict):
        return None
    applied = retrieval_window.get('applied')
    if applied is not None and (not bool(applied)):
        return None
    start = _optional_float(retrieval_window.get('start_ts') if retrieval_window.get('start_ts') is not None else retrieval_window.get('start_time'))
    end = _optional_float(retrieval_window.get('end_ts') if retrieval_window.get('end_ts') is not None else retrieval_window.get('end_time'))
    if start is None or end is None:
        return None
    if end < start:
        start, end = (end, start)
    return (start, end)

def _candidate_time_bounds(pack: dict[str, Any]) -> tuple[float, float] | None:
    time_span = pack.get('time_span', {})
    if not isinstance(time_span, dict):
        return None
    start = _optional_float(time_span.get('start_ts') if time_span.get('start_ts') is not None else time_span.get('start_time'))
    end = _optional_float(time_span.get('end_ts') if time_span.get('end_ts') is not None else time_span.get('end_time'))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    if end < start:
        start, end = (end, start)
    return (start, end)

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target_state', 'relation', 'operation', 'target'):
            if value.get(key):
                values.append(str(value[key]))
        for key in ('target_entities', 'must_bind_to'):
            values.extend(_iter_text_values(value.get(key)))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_iter_text_values(item))
        return values
    return [str(value)]

def _clean_phrase(value: Any) -> str:
    text = re.sub('\\s+', ' ', str(value or '').strip())
    text = re.sub('[^A-Za-z0-9_\\- /]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def _semantic_embedding_text(value: Any, *, max_words: int=18, max_chars: int=160) -> str:
    text = str(value or '')
    text = re.sub('\\b[\\w_]+:entity:\\d+\\b', ' ', text)
    text = re.sub('\\bentity:\\d+\\b', ' ', text)
    text = re.sub('\\bRAW_RELATION_([A-Z0-9_]+)\\b', lambda match: match.group(1).lower().replace('_', ' '), text)
    cleaned = _clean_phrase(text)
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = ' '.join(words[:max_words])
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(' ', 1)[0].strip() or cleaned[:max_chars]
    return cleaned

def _ordered_unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        if not text:
            continue
        normalized = ' '.join(text.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result

def _cosine_similarity(left: Any, right: Any) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    dot = sum((a * b for a, b in zip(left_values, right_values, strict=True)))
    left_norm = sum((value * value for value in left_values)) ** 0.5
    right_norm = sum((value * value for value in right_values)) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _format_float(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return '0'
    return f'{number:.6g}'

def _short_text(value: Any, *, limit: int) -> str:
    text = re.sub('\\s+', ' ', str(value or '').strip())
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rstrip() + '...'
