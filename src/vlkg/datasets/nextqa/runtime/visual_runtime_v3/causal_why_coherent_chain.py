"""Unified causal-chain evidence selection for the NExT-QA causal_why route.

This module is deliberately independent from visual_runtime_v2.  It keeps one
strategy for every causal_why question and only changes visual candidate
selection.  Graph retrieval, question routing, choices, and evaluator labels
are outside this handler's input contract.
"""
from __future__ import annotations
from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.rendering import qa_safe_hit, render_hit
ROUTE_ID = 'causal_why'
STRATEGY_ID = 'causal_why_coherent_chain_v1'
HANDLER_ID = 'nextqa_visual_handler_v3:causal_why_coherent_chain'
EFFECT_ROLE = 'effect_event'
PRECEDING_ROLE = 'preceding_action_or_state'
INTERACTION_ROLE = 'actor_object_interaction'
ROLE_ORDER = (EFFECT_ROLE, PRECEDING_ROLE, INTERACTION_ROLE)
ROLE_QUOTAS = {EFFECT_ROLE: 4, PRECEDING_ROLE: 4, INTERACTION_ROLE: 4}
ALLOWED_KINDS = {EFFECT_ROLE: frozenset({'event_boundary', 'before_after_pair'}), PRECEDING_ROLE: frozenset({'before_after_pair', 'temporal_window'}), INTERACTION_ROLE: frozenset({'object', 'tracklet'})}
TOKEN_PATTERN = re.compile('[a-z0-9]+')

def build_coherent_causal_chain_pack(route_record: Mapping[str, Any], role_hits: Mapping[str, Sequence[Mapping[str, Any]]], *, evidence_limit: int=12, role_quotas: Mapping[str, int] | None=None) -> dict[str, Any]:
    """Build one globally deduplicated, temporally coherent causal pack.

    Candidate retrieval is expected to be performed by the caller.  This
    function reranks the retrieved candidates jointly, so it can be tested
    without loading an index or an embedding model.
    """
    if str(route_record.get('route_id') or '') != ROUTE_ID:
        raise ValueError('coherent causal-chain strategy requires route_id=causal_why')
    video_id = str(route_record.get('video_id') or '').strip()
    if not video_id:
        raise ValueError('causal_why route record is missing video_id')
    if evidence_limit < 1:
        raise ValueError('evidence_limit must be positive')
    quotas = dict(ROLE_QUOTAS)
    if role_quotas is not None:
        quotas.update({str(key): int(value) for key, value in role_quotas.items()})
    if any((quotas.get(role, 0) < 1 for role in ROLE_ORDER)):
        raise ValueError('all causal chain role quotas must be positive')
    candidates, excluded = _merge_candidates(role_hits, video_id=video_id)
    by_role = {role: [candidate for candidate in candidates.values() if role in candidate['source_roles'] and candidate['candidate_kind'] in ALLOWED_KINDS[role] and candidate['rendered']] for role in ROLE_ORDER}
    anchor = _select_anchor(by_role[EFFECT_ROLE], by_role[PRECEDING_ROLE], by_role[INTERACTION_ROLE])
    selected: dict[str, dict[str, Any]] = {}
    role_selection: dict[str, list[str]] = {}
    if anchor is not None:
        _select_role_candidates(selected, role_selection, EFFECT_ROLE, _rank_effect_candidates(by_role[EFFECT_ROLE], anchor), quota=quotas[EFFECT_ROLE])
        _select_role_candidates(selected, role_selection, PRECEDING_ROLE, _rank_temporal_candidates(by_role[PRECEDING_ROLE], anchor, role=PRECEDING_ROLE), quota=quotas[PRECEDING_ROLE])
        _select_role_candidates(selected, role_selection, INTERACTION_ROLE, _rank_interaction_candidates(by_role[INTERACTION_ROLE], anchor, by_role[PRECEDING_ROLE]), quota=quotas[INTERACTION_ROLE])
    else:
        _select_role_candidates(selected, role_selection, PRECEDING_ROLE, _rank_base_candidates(by_role[PRECEDING_ROLE]), quota=quotas[PRECEDING_ROLE])
        _select_role_candidates(selected, role_selection, INTERACTION_ROLE, _rank_base_candidates(by_role[INTERACTION_ROLE]), quota=quotas[INTERACTION_ROLE])
    selected_rows: list[dict[str, Any]] = []
    for candidate_id, candidate in selected.items():
        assigned = tuple((role for role in ROLE_ORDER if role in role_selection.get(candidate_id, [])))
        if not assigned:
            continue
        text = _render_candidate(candidate, assigned)
        if not text:
            excluded['not_renderable_as_graph_bound_text'] += 1
            continue
        selected_rows.append({'role_id': assigned[0], 'role_ids': list(assigned), 'candidate_id': candidate_id, 'candidate_kind': candidate['candidate_kind'], 'similarity': candidate['similarity'], 'text': text})
    selected_rows.sort(key=lambda row: _selected_sort_key(row, selected, anchor))
    selected_rows = selected_rows[:evidence_limit]
    selected_ids = {str(row['candidate_id']) for row in selected_rows}
    role_status = {role: 'filled' if any((role in role_selection.get(cid, []) for cid in selected_ids)) else 'empty' for role in ROLE_ORDER}
    boundary_available = anchor is not None and any((row['candidate_kind'] in ALLOWED_KINDS[EFFECT_ROLE] for row in selected_rows))
    status = 'completed' if boundary_available else 'boundary_unavailable_fallback'
    if not selected_rows:
        status = 'boundary_unavailable'
    safe_hits = []
    for row in selected_rows:
        candidate = selected[str(row['candidate_id'])]
        safe = qa_safe_hit(candidate['hit'])
        safe['strategy_role_ids'] = list(row['role_ids'])
        safe_hits.append(safe)
    return {'schema_version': 'nextqa_visual_candidate_pack_v3', 'strategy_id': STRATEGY_ID, 'route_id': ROUTE_ID, 'handler_id': HANDLER_ID, 'handler_module': 'causal_why_coherent_chain', 'status': status, 'boundary_available': boundary_available, 'role_status': role_status, 'evidence': [{key: value for key, value in row.items()} for row in selected_rows], 'visual_hits': safe_hits, 'used_visual_candidate_ids': [str(row['candidate_id']) for row in selected_rows], 'excluded_reason_counts': dict(sorted(excluded.items())), 'strategy_metadata': {'anchor_candidate_id': str(anchor['candidate_id']) if anchor else None, 'anchor_timestamp_sec': _effect_time(anchor) if anchor else None, 'merged_candidate_count': len(candidates), 'selected_unique_candidate_count': len(selected_rows), 'role_quotas': quotas, 'uses_gold_answer': False, 'uses_official_type': False, 'changes_graph_retrieval': False}}

def _merge_candidates(role_hits: Mapping[str, Sequence[Mapping[str, Any]]], *, video_id: str) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    candidates: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    for role in ROLE_ORDER:
        for raw_hit in role_hits.get(role, ()):
            hit = dict(raw_hit)
            candidate_id = str(hit.get('candidate_id') or '').strip()
            if not candidate_id:
                excluded['missing_candidate_id'] += 1
                continue
            if str(hit.get('video_id') or '') != video_id:
                excluded['cross_video_candidate'] += 1
                continue
            kind = str(hit.get('candidate_kind') or '')
            if kind not in ALLOWED_KINDS[role]:
                excluded['role_kind_mismatch'] += 1
                continue
            rendered = render_hit(ROUTE_ID, role, hit)
            if candidate_id in candidates:
                candidates[candidate_id]['source_roles'].add(role)
                excluded['duplicate_candidate_merged'] += 1
                continue
            if not rendered:
                excluded['not_renderable_as_graph_bound_text'] += 1
                continue
            candidates[candidate_id] = {'candidate_id': candidate_id, 'candidate_kind': kind, 'video_id': video_id, 'hit': hit, 'source_roles': {role}, 'rendered': True, 'similarity': float(hit.get('similarity') or 0.0)}
    return (candidates, excluded)

def _select_anchor(effects: Sequence[Mapping[str, Any]], preceding: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not effects:
        return None
    ranked: list[tuple[float, str, Mapping[str, Any]]] = []
    for candidate in effects:
        anchor_time = _effect_time(candidate)
        best_preceding = max((_causal_compatibility(candidate, item, role=PRECEDING_ROLE) for item in preceding), default=0.0)
        best_interaction = max((_causal_compatibility(candidate, item, role=INTERACTION_ROLE) for item in interactions), default=0.0)
        metadata_bonus = 0.0
        if anchor_time is not None:
            metadata_bonus += 0.04
        if _entity_tokens(candidate):
            metadata_bonus += 0.04
        score = _base_score(candidate) + 0.24 * best_preceding + 0.18 * best_interaction + metadata_bonus
        ranked.append((score, str(candidate['candidate_id']), candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2]

def _rank_effect_candidates(candidates: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    anchor_time = _effect_time(anchor)
    ranked = sorted(candidates, key=lambda item: (-(_base_score(item) + 0.2 * _same_event_or_entity(item, anchor) + 0.1 * _time_proximity(item, anchor_time)), str(item['candidate_id'])))
    return ranked

def _rank_temporal_candidates(candidates: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], *, role: str) -> list[Mapping[str, Any]]:
    filtered = list(candidates)
    if role == PRECEDING_ROLE:
        anchor_time = _effect_time(anchor)
        if anchor_time is not None:
            filtered = [item for item in filtered if (end_time := _preceding_end(item)) is not None and end_time < anchor_time - 1e-09]
    ranked = sorted(filtered, key=lambda item: (-(_base_score(item) + 0.48 * _causal_compatibility(anchor, item, role=role)), str(item['candidate_id'])))
    return ranked

def _rank_interaction_candidates(candidates: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], preceding: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ranked = sorted(candidates, key=lambda item: (-(_base_score(item) + 0.28 * _causal_compatibility(anchor, item, role=INTERACTION_ROLE) + 0.16 * max((_same_event_or_entity(item, previous) for previous in preceding), default=0.0)), str(item['candidate_id'])))
    return ranked

def _rank_base_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(candidates, key=lambda item: (-_base_score(item), str(item['candidate_id'])))

def _select_role_candidates(selected: dict[str, dict[str, Any]], role_selection: dict[str, list[str]], role: str, ranked: Sequence[Mapping[str, Any]], *, quota: int) -> None:
    selected_for_role = 0
    for candidate in ranked:
        candidate_id = str(candidate['candidate_id'])
        source_roles = candidate['source_roles']
        if candidate_id in selected:
            if role in source_roles and role not in role_selection.setdefault(candidate_id, []):
                role_selection[candidate_id].append(role)
                selected_for_role += 1
            if selected_for_role >= quota:
                break
            continue
        selected[candidate_id] = dict(candidate)
        role_selection[candidate_id] = [role]
        selected_for_role += 1
        if selected_for_role >= quota:
            break

def _render_candidate(candidate: Mapping[str, Any], roles: Sequence[str]) -> str | None:
    primary = roles[0]
    rendered = render_hit(ROUTE_ID, primary, candidate['hit'])
    if not rendered:
        return None
    if len(roles) == 1:
        return rendered
    role_text = ', '.join((role.replace('_', ' ') for role in roles))
    return f"Causal chain roles ({role_text}): {rendered.split(': ', 1)[-1]}"

def _selected_sort_key(row: Mapping[str, Any], selected: Mapping[str, Mapping[str, Any]], anchor: Mapping[str, Any] | None) -> tuple[int, float, str]:
    roles = row.get('role_ids') or []
    role_index = min((ROLE_ORDER.index(str(role)) for role in roles), default=len(ROLE_ORDER))
    candidate = selected[str(row['candidate_id'])]
    proximity = _time_proximity(candidate, _effect_time(anchor) if anchor else None)
    return (role_index, -proximity, str(row['candidate_id']))

def _base_score(candidate: Mapping[str, Any]) -> float:
    similarity = max(-1.0, min(1.0, float(candidate.get('similarity') or 0.0)))
    hit = candidate.get('hit') or {}
    rank = max(1, int(hit.get('rank_within_kind') or 1))
    return 0.6 * ((similarity + 1.0) / 2.0) + 0.4 / (1.0 + math.log1p(rank - 1))

def _causal_compatibility(anchor: Mapping[str, Any], candidate: Mapping[str, Any], *, role: str) -> float:
    anchor_time = _effect_time(anchor)
    if anchor_time is None:
        return 0.0
    if role == PRECEDING_ROLE:
        end_time = _preceding_end(candidate)
        if end_time is None or end_time >= anchor_time - 1e-09:
            return 0.0
        return _bounded_proximity(anchor_time - end_time, horizon=8.0)
    center = _candidate_center(candidate)
    if center is None:
        return 0.0
    return _bounded_proximity(abs(center - anchor_time), horizon=10.0)

def _same_event_or_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_meta = left.get('hit', {}).get('metadata') or {}
    right_meta = right.get('hit', {}).get('metadata') or {}
    left_event = str(left_meta.get('event_id') or '')
    right_event = str(right_meta.get('event_id') or '')
    if left_event and left_event == right_event:
        return 1.0
    left_entities = _entity_tokens(left)
    right_entities = _entity_tokens(right)
    if left_entities and right_entities and left_entities & right_entities:
        return 1.0
    left_labels = _label_tokens(left)
    right_labels = _label_tokens(right)
    if left_labels and right_labels and left_labels & right_labels:
        return 0.5
    return 0.0

def _time_proximity(candidate: Mapping[str, Any], anchor_time: float | None) -> float:
    center = _candidate_center(candidate)
    if center is None or anchor_time is None:
        return 0.0
    return _bounded_proximity(abs(center - anchor_time), horizon=10.0)

def _bounded_proximity(distance: float, *, horizon: float) -> float:
    if distance < 0:
        return 0.0
    return max(0.0, 1.0 - min(float(distance), horizon) / horizon)

def _effect_time(candidate: Mapping[str, Any] | None) -> float | None:
    if candidate is None:
        return None
    metadata = candidate.get('hit', {}).get('metadata') or {}
    kind = str(candidate.get('candidate_kind') or '')
    value = metadata.get('timestamp_sec') if kind == 'event_boundary' else metadata.get('boundary_ts')
    return _float_or_none(value)

def _preceding_end(candidate: Mapping[str, Any]) -> float | None:
    metadata = candidate.get('hit', {}).get('metadata') or {}
    kind = str(candidate.get('candidate_kind') or '')
    if kind == 'before_after_pair':
        return _float_or_none(metadata.get('before_ts'))
    if kind == 'temporal_window':
        return _float_or_none(metadata.get('end_ts'))
    return _candidate_center(candidate)

def _candidate_center(candidate: Mapping[str, Any]) -> float | None:
    metadata = candidate.get('hit', {}).get('metadata') or {}
    kind = str(candidate.get('candidate_kind') or '')
    if kind == 'event_boundary':
        return _float_or_none(metadata.get('timestamp_sec'))
    if kind == 'before_after_pair':
        before = _float_or_none(metadata.get('before_ts'))
        after = _float_or_none(metadata.get('after_ts'))
        boundary = _float_or_none(metadata.get('boundary_ts'))
        if before is not None and after is not None:
            return (before + after) / 2.0
        return boundary if boundary is not None else before if before is not None else after
    if kind == 'temporal_window':
        start = _float_or_none(metadata.get('start_ts'))
        end = _float_or_none(metadata.get('end_ts'))
        if start is not None and end is not None:
            return (start + end) / 2.0
        return start if start is not None else end
    if kind == 'object':
        return _float_or_none(metadata.get('timestamp_sec'))
    if kind == 'tracklet':
        start = _float_or_none(metadata.get('start_ts'))
        end = _float_or_none(metadata.get('end_ts'))
        if start is not None and end is not None:
            return (start + end) / 2.0
        return start if start is not None else end
    return None

def _entity_tokens(candidate: Mapping[str, Any]) -> set[str]:
    metadata = candidate.get('hit', {}).get('metadata') or {}
    values: list[Any] = []
    for key in ('entity_id', 'entity_ids', 'subject_entity_id', 'object_entity_id', 'before_entity_ids', 'after_entity_ids'):
        value = metadata.get(key)
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return {str(value).strip() for value in values if str(value or '').strip()}

def _label_tokens(candidate: Mapping[str, Any]) -> set[str]:
    metadata = candidate.get('hit', {}).get('metadata') or {}
    values: list[Any] = []
    for key in ('label', 'labels', 'labels_seen', 'before_labels', 'after_labels', 'state_values', 'states', 'states_seen', 'before_states', 'after_states'):
        value = metadata.get(key)
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return set(TOKEN_PATTERN.findall(' '.join((str(value or '') for value in values)).lower()))

def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
__all__ = ['HANDLER_ID', 'ROUTE_ID', 'STRATEGY_ID', 'build_coherent_causal_chain_pack']
