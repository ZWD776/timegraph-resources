"""Fixed coherent-chain strategy for the NExT-QA causal_why route.

This is a new strategy identity.  It does not alter the v2 handler or the
previous coherent-chain v1 outputs.  Role quotas are counted on unique
candidate IDs, and a candidate shared by retrieval roles cannot satisfy both
roles at once.
"""
from __future__ import annotations
from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.rendering import qa_safe_hit, render_hit
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain import ALLOWED_KINDS, EFFECT_ROLE, INTERACTION_ROLE, PRECEDING_ROLE, ROLE_ORDER
ROUTE_ID = 'causal_why'
STRATEGY_ID = 'causal_why_coherent_chain_v2'
HANDLER_ID = 'nextqa_visual_handler_v4:causal_why_coherent_chain'
ROLE_QUOTAS = {EFFECT_ROLE: 4, PRECEDING_ROLE: 4, INTERACTION_ROLE: 4}
TOKEN_PATTERN = re.compile('[a-z0-9]+')

def build_coherent_causal_chain_pack_v2(route_record: Mapping[str, Any], role_hits: Mapping[str, Sequence[Mapping[str, Any]]], *, evidence_limit: int=12, role_quotas: Mapping[str, int] | None=None) -> dict[str, Any]:
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
    by_role = {role: [item for item in candidates.values() if role in item['source_roles'] and item['candidate_kind'] in ALLOWED_KINDS[role]] for role in ROLE_ORDER}
    anchor = _select_anchor(by_role[EFFECT_ROLE], by_role[PRECEDING_ROLE], by_role[INTERACTION_ROLE])
    anchor_time = _effect_time(anchor) if anchor else None
    selected: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    selection_modes: Counter[str] = Counter()
    if anchor is not None:
        _select_unique(selected, selected_rows, role_counts, selection_modes, EFFECT_ROLE, _rank_effect(by_role[EFFECT_ROLE], anchor), quota=quotas[EFFECT_ROLE])
        strict_preceding = [item for item in by_role[PRECEDING_ROLE] if _is_strict_preceding(item, anchor_time)]
        fallback_preceding = [item for item in by_role[PRECEDING_ROLE] if _is_temporal_fallback(item, anchor_time)]
        _select_unique(selected, selected_rows, role_counts, selection_modes, PRECEDING_ROLE, _rank_preceding(strict_preceding, anchor), quota=quotas[PRECEDING_ROLE], selection_mode='strict_preceding')
        if role_counts[PRECEDING_ROLE] < quotas[PRECEDING_ROLE]:
            _select_unique(selected, selected_rows, role_counts, selection_modes, PRECEDING_ROLE, _rank_preceding(fallback_preceding, anchor), quota=quotas[PRECEDING_ROLE] - role_counts[PRECEDING_ROLE], selection_mode='temporal_window_fallback')
            excluded['preceding_temporal_fallback_used'] += 1
        _select_unique(selected, selected_rows, role_counts, selection_modes, INTERACTION_ROLE, _rank_interaction(by_role[INTERACTION_ROLE], anchor, by_role[PRECEDING_ROLE]), quota=quotas[INTERACTION_ROLE])
    else:
        _select_unique(selected, selected_rows, role_counts, selection_modes, PRECEDING_ROLE, _rank_base(by_role[PRECEDING_ROLE]), quota=quotas[PRECEDING_ROLE], selection_mode='boundary_free_fallback')
        _select_unique(selected, selected_rows, role_counts, selection_modes, INTERACTION_ROLE, _rank_base(by_role[INTERACTION_ROLE]), quota=quotas[INTERACTION_ROLE], selection_mode='boundary_free_fallback')
    if len(selected_rows) < evidence_limit:
        remaining = sorted((item for item in candidates.values() if item['candidate_id'] not in selected and any((_candidate_allowed_for_role(item, role, anchor_time=anchor_time) for role in item['source_roles']))), key=lambda item: (-_base_score(item), item['candidate_id']))
        for item in remaining:
            roles = [role for role in ROLE_ORDER if role in item['source_roles']]
            role = next((candidate_role for candidate_role in roles if candidate_role != EFFECT_ROLE and _candidate_allowed_for_role(item, candidate_role, anchor_time=anchor_time)), None)
            role = role or (roles[0] if roles else None)
            if role is None:
                continue
            _append_candidate(selected, selected_rows, role_counts, selection_modes, item, role, 'global_budget_fill')
            if len(selected_rows) >= evidence_limit:
                break
    selected_rows = selected_rows[:evidence_limit]
    final_ids = {str(row['candidate_id']) for row in selected_rows}
    role_status = {role: 'filled' if role_counts[role] > 0 and any((role == row['role_id'] and row['candidate_id'] in final_ids for row in selected_rows)) else 'empty' for role in ROLE_ORDER}
    boundary_available = anchor is not None and any((row['role_id'] == EFFECT_ROLE for row in selected_rows))
    status = 'completed' if boundary_available else 'boundary_unavailable_fallback'
    if not selected_rows:
        status = 'boundary_unavailable'
    evidence = []
    safe_hits = []
    for row in selected_rows:
        item = selected[row['candidate_id']]
        evidence.append(dict(row))
        safe = qa_safe_hit(item['hit'])
        safe['strategy_role_ids'] = [row['role_id']]
        safe['strategy_selection_mode'] = row['selection_mode']
        safe_hits.append(safe)
    return {'schema_version': 'nextqa_visual_candidate_pack_v4', 'strategy_id': STRATEGY_ID, 'route_id': ROUTE_ID, 'handler_id': HANDLER_ID, 'handler_module': 'causal_why_coherent_chain_v2', 'status': status, 'boundary_available': boundary_available, 'role_status': role_status, 'evidence': evidence, 'visual_hits': safe_hits, 'used_visual_candidate_ids': [row['candidate_id'] for row in selected_rows], 'excluded_reason_counts': dict(sorted(excluded.items())), 'strategy_metadata': {'anchor_candidate_id': str(anchor['candidate_id']) if anchor else None, 'anchor_timestamp_sec': anchor_time, 'role_counts': dict(sorted(role_counts.items())), 'selection_mode_counts': dict(sorted(selection_modes.items())), 'selected_unique_candidate_count': len(selected_rows), 'uses_gold_answer': False, 'uses_official_type': False, 'changes_graph_retrieval': False}}

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
            if not render_hit(ROUTE_ID, role, hit):
                excluded['not_renderable_as_graph_bound_text'] += 1
                continue
            if candidate_id in candidates:
                candidates[candidate_id]['source_roles'].add(role)
                excluded['duplicate_candidate_merged'] += 1
                continue
            candidates[candidate_id] = {'candidate_id': candidate_id, 'candidate_kind': kind, 'video_id': video_id, 'hit': hit, 'source_roles': {role}, 'similarity': float(hit.get('similarity') or 0.0)}
    return (candidates, excluded)

def _select_anchor(effects: Sequence[Mapping[str, Any]], preceding: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not effects:
        return None
    ranked = sorted(effects, key=lambda item: -(_base_score(item) + 0.24 * max((_compatibility(item, other, PRECEDING_ROLE) for other in preceding), default=0.0) + 0.18 * max((_compatibility(item, other, INTERACTION_ROLE) for other in interactions), default=0.0)))
    return ranked[0]

def _select_unique(selected: dict[str, dict[str, Any]], rows: list[dict[str, Any]], role_counts: Counter[str], modes: Counter[str], role: str, ranked: Sequence[Mapping[str, Any]], *, quota: int, selection_mode: str='role_ranked') -> None:
    for item in ranked:
        if role_counts[role] >= quota:
            break
        if item['candidate_id'] in selected:
            continue
        _append_candidate(selected, rows, role_counts, modes, item, role, selection_mode)

def _append_candidate(selected: dict[str, dict[str, Any]], rows: list[dict[str, Any]], role_counts: Counter[str], modes: Counter[str], item: Mapping[str, Any], role: str, mode: str) -> None:
    candidate_id = str(item['candidate_id'])
    selected[candidate_id] = dict(item)
    rows.append({'role_id': role, 'role_ids': [role], 'candidate_id': candidate_id, 'candidate_kind': item['candidate_kind'], 'similarity': float(item['similarity']), 'selection_mode': mode, 'text': _render(item, role, mode)})
    role_counts[role] += 1
    modes[mode] += 1

def _render(item: Mapping[str, Any], role: str, mode: str) -> str:
    text = render_hit(ROUTE_ID, role, item['hit'])
    if mode == 'temporal_window_fallback':
        return f"Temporal context fallback before/around the effect: {text.split(': ', 1)[-1]}"
    if mode == 'boundary_free_fallback':
        return f"Boundary-free causal context: {text.split(': ', 1)[-1]}"
    return text or ''

def _rank_effect(items: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(items, key=lambda item: -(_base_score(item) + 0.2 * _same_signal(item, anchor) + 0.1 * _time_proximity(item, _effect_time(anchor))))

def _rank_preceding(items: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(items, key=lambda item: -(_base_score(item) + 0.48 * _compatibility(anchor, item, PRECEDING_ROLE)))

def _rank_interaction(items: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], preceding: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(items, key=lambda item: -(_base_score(item) + 0.28 * _compatibility(anchor, item, INTERACTION_ROLE) + 0.16 * max((_same_signal(item, other) for other in preceding), default=0.0)))

def _rank_base(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(items, key=lambda item: -_base_score(item))

def _is_strict_preceding(item: Mapping[str, Any], anchor_time: float | None) -> bool:
    if anchor_time is None:
        return False
    metadata = item['hit'].get('metadata') or {}
    kind = str(item['candidate_kind'])
    if kind == 'before_after_pair':
        value = metadata.get('boundary_ts')
    elif kind == 'temporal_window':
        value = metadata.get('end_ts')
    else:
        return False
    try:
        return value is not None and float(value) < anchor_time - 1e-09
    except (TypeError, ValueError):
        return False

def _is_temporal_fallback(item: Mapping[str, Any], anchor_time: float | None) -> bool:
    if anchor_time is None or item['candidate_kind'] != 'temporal_window':
        return False
    metadata = item['hit'].get('metadata') or {}
    try:
        start = float(metadata.get('start_ts'))
        end = float(metadata.get('end_ts'))
    except (TypeError, ValueError):
        return False
    return start < anchor_time - 1e-09 <= end

def _candidate_allowed_for_role(item: Mapping[str, Any], role: str, *, anchor_time: float | None) -> bool:
    if role == PRECEDING_ROLE:
        return _is_strict_preceding(item, anchor_time) or _is_temporal_fallback(item, anchor_time)
    return str(item.get('candidate_kind') or '') in ALLOWED_KINDS.get(role, frozenset())

def _effect_time(item: Mapping[str, Any] | None) -> float | None:
    if item is None:
        return None
    metadata = item['hit'].get('metadata') or {}
    value = metadata.get('timestamp_sec') if item['candidate_kind'] == 'event_boundary' else metadata.get('boundary_ts')
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

def _candidate_center(item: Mapping[str, Any]) -> float | None:
    metadata = item['hit'].get('metadata') or {}
    kind = str(item['candidate_kind'])
    values = {'event_boundary': (metadata.get('timestamp_sec'), metadata.get('timestamp_sec')), 'before_after_pair': (metadata.get('before_ts'), metadata.get('after_ts')), 'temporal_window': (metadata.get('start_ts'), metadata.get('end_ts')), 'object': (metadata.get('timestamp_sec'), metadata.get('timestamp_sec')), 'tracklet': (metadata.get('start_ts'), metadata.get('end_ts'))}.get(kind, (None, None))
    try:
        left = None if values[0] is None else float(values[0])
        right = None if values[1] is None else float(values[1])
    except (TypeError, ValueError):
        return None
    if left is not None and right is not None:
        return (left + right) / 2.0
    return left if left is not None else right

def _compatibility(anchor: Mapping[str, Any], item: Mapping[str, Any], role: str) -> float:
    anchor_time = _effect_time(anchor)
    if anchor_time is None:
        return 0.0
    if role == PRECEDING_ROLE:
        end = item['hit'].get('metadata', {}).get('boundary_ts') if item['candidate_kind'] == 'before_after_pair' else item['hit'].get('metadata', {}).get('end_ts')
        try:
            end_value = float(end)
        except (TypeError, ValueError):
            return 0.0
        return _bounded(anchor_time - end_value, 8.0) if end_value < anchor_time else 0.0
    center = _candidate_center(item)
    return 0.0 if center is None else _bounded(abs(center - anchor_time), 10.0)

def _time_proximity(item: Mapping[str, Any], anchor_time: float | None) -> float:
    center = _candidate_center(item)
    return 0.0 if center is None or anchor_time is None else _bounded(abs(center - anchor_time), 10.0)

def _same_signal(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    lm = left['hit'].get('metadata') or {}
    rm = right['hit'].get('metadata') or {}
    if str(lm.get('event_id') or '') and str(lm.get('event_id')) == str(rm.get('event_id') or ''):
        return 1.0
    left_ids = _entity_ids(lm)
    right_ids = _entity_ids(rm)
    if left_ids & right_ids:
        return 1.0
    left_labels = _labels(lm)
    right_labels = _labels(rm)
    return 0.5 if left_labels & right_labels else 0.0

def _entity_ids(metadata: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ('entity_id', 'entity_ids', 'subject_entity_id', 'object_entity_id', 'before_entity_ids', 'after_entity_ids'):
        value = metadata.get(key)
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return {str(value).strip() for value in values if str(value or '').strip()}

def _labels(metadata: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ('label', 'labels', 'labels_seen', 'before_labels', 'after_labels', 'state_values', 'states', 'states_seen', 'before_states', 'after_states'):
        value = metadata.get(key)
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return set(TOKEN_PATTERN.findall(' '.join((str(value or '') for value in values)).lower()))

def _base_score(item: Mapping[str, Any]) -> float:
    similarity = max(-1.0, min(1.0, float(item.get('similarity') or 0.0)))
    rank = max(1, int(item['hit'].get('rank_within_kind') or 1))
    return 0.6 * ((similarity + 1.0) / 2.0) + 0.4 / (1.0 + math.log1p(rank - 1))

def _bounded(distance: float, horizon: float) -> float:
    return max(0.0, 1.0 - min(max(0.0, float(distance)), horizon) / horizon)
__all__ = ['HANDLER_ID', 'ROUTE_ID', 'STRATEGY_ID', 'ROLE_QUOTAS', 'build_coherent_causal_chain_pack_v2']
