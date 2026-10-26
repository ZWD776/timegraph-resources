"""Uniform, route-free visual retrieval for the NExT-QA ablation."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS, l2_normalize
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import VisualShardStore
ABLATION_LABEL = 'no_routing'
EVIDENCE_LIMIT = 12
TOP_K_PER_KIND = 4

def retrieve_uniform_candidates(store: VisualShardStore, *, video_id: str, query_vector: Sequence[float], top_k_per_kind: int=TOP_K_PER_KIND) -> list[dict[str, Any]]:
    """Retrieve the same candidate inventory and budget for every question."""
    if top_k_per_kind < 1:
        raise ValueError('top_k_per_kind must be positive')
    query = l2_normalize(query_vector)
    hits: list[dict[str, Any]] = []
    for kind in CANDIDATE_KINDS:
        hits.extend(store.search(video_id=str(video_id), kind=str(kind), query_vector=query, top_k=top_k_per_kind))
    best: dict[str, dict[str, Any]] = {}
    for hit in hits:
        candidate_id = str(hit['candidate_id'])
        current = best.get(candidate_id)
        if current is None or float(hit.get('similarity') or 0.0) > float(current.get('similarity') or 0.0):
            best[candidate_id] = dict(hit)
    return sorted(best.values(), key=lambda row: (-float(row.get('similarity') or 0.0), str(row['candidate_id'])))[:EVIDENCE_LIMIT]

def render_neutral_hit(hit: Mapping[str, Any]) -> str | None:
    """Render index metadata without a question-specific or route-specific role."""
    kind = str(hit.get('candidate_kind') or 'candidate')
    metadata = dict(hit.get('metadata') or {})
    parts: list[str] = [f'visual candidate kind={kind}']
    if kind == 'object':
        label = str(metadata.get('label') or 'object')
        timestamp = metadata.get('timestamp_sec')
        parts.append(f'object={label}')
        if timestamp is not None:
            parts.append(f'time={float(timestamp):.3f}s')
        states = _join(metadata.get('state_values'))
        if states:
            parts.append(f'states={states}')
    elif kind == 'tracklet':
        labels = _join(metadata.get('labels_seen')) or str(metadata.get('label') or 'object')
        parts.append(f'tracklet={labels}')
        if metadata.get('start_ts') is not None and metadata.get('end_ts') is not None:
            parts.append(f"interval={float(metadata['start_ts']):.3f}-{float(metadata['end_ts']):.3f}s")
        states = _join(metadata.get('states_seen'))
        if states:
            parts.append(f'states={states}')
    elif kind == 'frame':
        labels = _join(metadata.get('labels'))
        states = _join(metadata.get('states'))
        if labels:
            parts.append(f'objects={labels}')
        if states:
            parts.append(f'states={states}')
        if metadata.get('timestamp_sec') is not None:
            parts.append(f"time={float(metadata['timestamp_sec']):.3f}s")
    elif kind == 'temporal_window':
        if metadata.get('start_ts') is not None and metadata.get('end_ts') is not None:
            parts.append(f"interval={float(metadata['start_ts']):.3f}-{float(metadata['end_ts']):.3f}s")
        _append_summary(parts, metadata)
    elif kind == 'event_boundary':
        event_type = str(metadata.get('event_type') or 'event')
        parts.append(f'event={event_type}')
        if metadata.get('timestamp_sec') is not None:
            parts.append(f"time={float(metadata['timestamp_sec']):.3f}s")
        _append_summary(parts, metadata)
    elif kind == 'before_after_pair':
        if metadata.get('boundary_ts') is not None:
            parts.append(f"boundary={float(metadata['boundary_ts']):.3f}s")
        before = _join(metadata.get('before_labels'))
        after = _join(metadata.get('after_labels'))
        if before:
            parts.append(f'preceding_objects={before}')
        if after:
            parts.append(f'following_objects={after}')
    else:
        return None
    return '; '.join(parts) + '.'

def _append_summary(parts: list[str], metadata: Mapping[str, Any]) -> None:
    labels = _join(metadata.get('labels'))
    states = _join(metadata.get('states'))
    if labels:
        parts.append(f'objects={labels}')
    if states:
        parts.append(f'states={states}')
    relations = metadata.get('relations')
    if isinstance(relations, list) and relations:
        rendered = []
        for item in relations[:6]:
            if isinstance(item, Mapping):
                rendered.append(f"{item.get('subject_entity_id')} {item.get('predicate')} {item.get('object_entity_id')}")
        if rendered:
            parts.append('relations=' + ', '.join(rendered))

def _join(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ''
    return ', '.join(dict.fromkeys((str(item).strip() for item in value if str(item).strip())))
