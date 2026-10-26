"""Path-free text rendering for graph-bound visual candidates."""
from __future__ import annotations
from typing import Any, Mapping
SAFE_FIELDS_BY_KIND = {'object': {'observation_id', 'entity_id', 'frame_name', 'frame_index', 'timestamp_sec', 'bbox', 'label', 'state_values', 'scene_id', 'graph_bound'}, 'tracklet': {'entity_id', 'label', 'labels_seen', 'states_seen', 'start_ts', 'end_ts', 'observation_count', 'graph_bound'}, 'frame': {'frame_name', 'frame_index', 'timestamp_sec', 'labels', 'states', 'entity_ids', 'observation_count', 'hsgg_slot_status', 'graph_bound'}, 'temporal_window': {'start_ts', 'end_ts', 'frame_names', 'labels', 'states', 'entity_ids', 'relations', 'events', 'graph_bound'}, 'event_boundary': {'event_id', 'event_type', 'timestamp_sec', 'entity_id', 'subject_entity_id', 'object_entity_id', 'raw_predicate', 'frame_names', 'labels', 'states', 'entity_ids', 'graph_bound'}, 'before_after_pair': {'event_id', 'event_type', 'boundary_ts', 'pair_status', 'missing_sides', 'before_frame_name', 'before_ts', 'before_labels', 'before_states', 'before_entity_ids', 'after_frame_name', 'after_ts', 'after_labels', 'after_states', 'after_entity_ids', 'strict_temporal_pair', 'graph_bound'}}

def qa_safe_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(hit.get('candidate_kind') or '')
    allowed = SAFE_FIELDS_BY_KIND.get(kind, set())
    metadata = {key: _compact(key, value) for key, value in dict(hit.get('metadata') or {}).items() if key in allowed}
    return {'candidate_id': str(hit.get('candidate_id') or ''), 'candidate_kind': kind, 'video_id': str(hit.get('video_id') or ''), 'rank_within_kind': int(hit.get('rank_within_kind') or 0), 'similarity': float(hit.get('similarity') or 0.0), 'metadata': metadata}

def render_hit(route_id: str, role_id: str, hit: Mapping[str, Any]) -> str | None:
    kind = str(hit.get('candidate_kind') or '')
    item = hit.get('metadata') or {}
    if kind == 'object':
        states = _joined(item.get('state_values'))
        return f"{_role(role_id)}: object-crop similarity selected graph entity {item.get('label') or 'object'} at {_seconds(item.get('timestamp_sec'))}" + (f' with states {states}.' if states else '.')
    if kind == 'tracklet':
        labels = _joined(item.get('labels_seen')) or str(item.get('label') or 'object')
        states = _joined(item.get('states_seen'))
        return f"{_role(role_id)}: selected tracklet is visible from {_seconds(item.get('start_ts'))} to {_seconds(item.get('end_ts'))} as {labels}" + (f' with states {states}.' if states else '.')
    if kind == 'frame':
        labels = _joined(item.get('labels'))
        states = _joined(item.get('states'))
        if not labels and (not states):
            return None
        return f"{_role(role_id)}: full-frame visual similarity selected graph evidence at {_seconds(item.get('timestamp_sec'))}; objects {labels or 'unspecified'}" + (f'; states {states}.' if states else '.')
    if kind == 'temporal_window':
        details = _window_details(item)
        if not details:
            return None
        return f"{_role(role_id)}: selected temporal window {_seconds(item.get('start_ts'))}-{_seconds(item.get('end_ts'))}; {details}."
    if kind == 'event_boundary':
        labels = _joined(item.get('labels'))
        states = _joined(item.get('states'))
        suffix = '; '.join((value for value in (f'nearby objects {labels}' if labels else '', f'states {states}' if states else '') if value))
        return f"{_role(role_id)}: graph event {item.get('event_type') or 'EVENT'} at {_seconds(item.get('timestamp_sec'))}" + (f'; {suffix}.' if suffix else '.')
    if kind == 'before_after_pair':
        return _render_pair(route_id, role_id, item)
    return None

def _render_pair(route_id: str, role_id: str, item: Mapping[str, Any]) -> str | None:
    boundary = _seconds(item.get('boundary_ts'))
    before_ts = item.get('before_ts')
    after_ts = item.get('after_ts')
    before = _joined(item.get('before_labels')) or 'no retained HSGG objects'
    after = _joined(item.get('after_labels')) or 'no retained HSGG objects'
    if route_id == 'temporal_previous':
        if before_ts is None:
            return None
        return f'{_role(role_id)}: strictly before boundary {boundary}, frame {_seconds(before_ts)} contains {before}.'
    if route_id == 'temporal_next':
        if after_ts is None:
            return None
        return f'{_role(role_id)}: strictly after boundary {boundary}, frame {_seconds(after_ts)} contains {after}.'
    if before_ts is not None and after_ts is not None:
        return f'{_role(role_id)}: across boundary {boundary}, strict predecessor {_seconds(before_ts)} contains {before}; strict successor {_seconds(after_ts)} contains {after}.'
    available = f'strict predecessor {_seconds(before_ts)} contains {before}' if before_ts is not None else f'strict successor {_seconds(after_ts)} contains {after}'
    return f'{_role(role_id)}: boundary {boundary} has one missing side; {available}.'

def _window_details(item: Mapping[str, Any]) -> str:
    labels = _joined(item.get('labels'))
    states = _joined(item.get('states'))
    relations = _render_relations(item.get('relations'))
    events = _render_events(item.get('events'))
    return '; '.join((value for value in (f'objects {labels}' if labels else '', f'states {states}' if states else '', f'relations {relations}' if relations else '', f'events {events}' if events else '') if value))

def _render_relations(value: Any) -> str:
    if not isinstance(value, list):
        return ''
    return ', '.join((f"{item.get('subject_entity_id')} {item.get('predicate')} {item.get('object_entity_id')}" for item in value[:6] if isinstance(item, Mapping)))

def _render_events(value: Any) -> str:
    if not isinstance(value, list):
        return ''
    return ', '.join((f"{item.get('event_type')} at {_seconds(item.get('timestamp_sec'))}" for item in value[:6] if isinstance(item, Mapping)))

def _compact(key: str, value: Any) -> Any:
    if not isinstance(value, list):
        return value
    limit = 6 if key in {'relations', 'events'} else 24
    return value[:limit]

def _joined(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ''
    return ', '.join(dict.fromkeys((str(item).strip() for item in value if str(item).strip())))

def _seconds(value: Any) -> str:
    return f'{float(value or 0.0):.3f}s'

def _role(role_id: str) -> str:
    return role_id.replace('_', ' ').capitalize()
