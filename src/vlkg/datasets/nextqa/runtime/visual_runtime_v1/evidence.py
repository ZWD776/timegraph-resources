"""Retrieve graph-bound visual candidates and render text-only evidence."""
from __future__ import annotations
import json
from pathlib import Path
from vlkg.core.assets.source_paths import relocate_image_references
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS, l2_normalize
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.routing import ROUTE_CAUSAL_HOW, ROUTE_CAUSAL_WHY, ROUTE_DESCRIPTIVE_COUNT, ROUTE_TEMPORAL_NEXT, ROUTE_TEMPORAL_PREVIOUS

class VisualShardStore:

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.embedding_dimension = int(payload['embedding_dimension'])
        self.video_manifests = {str(item['video_id']): item for item in payload.get('videos', [])}
        self._cache: dict[tuple[str, str], tuple[Any, list[dict[str, Any]]]] = {}

    @property
    def video_ids(self) -> frozenset[str]:
        return frozenset(self.video_manifests)

    def release_video(self, video_id: str) -> None:
        for key in [key for key in self._cache if key[0] == video_id]:
            del self._cache[key]

    def search(self, *, video_id: str, kind: str, query_vector: Sequence[float] | np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if video_id not in self.video_manifests:
            raise KeyError(f'video has no visual index shard: {video_id}')
        if kind not in CANDIDATE_KINDS:
            raise ValueError(f'unsupported candidate kind: {kind}')
        index, metadata = self._load(video_id, kind)
        if index is None or not metadata:
            return []
        query = l2_normalize(query_vector).reshape(1, -1).astype(np.float32, copy=False)
        if query.shape[1] != self.embedding_dimension:
            raise ValueError('visual query dimension does not match the index')
        scores, indices = index.search(query, min(max(1, int(top_k)), len(metadata)))
        output: list[dict[str, Any]] = []
        for rank, (faiss_id, score) in enumerate(zip(indices[0].tolist(), scores[0].tolist(), strict=True), start=1):
            if faiss_id < 0:
                continue
            item = dict(metadata[faiss_id])
            output.append({'candidate_id': str(item['candidate_id']), 'candidate_kind': kind, 'video_id': video_id, 'rank_within_kind': rank, 'similarity': round(float(score), 6), 'metadata': item})
        return output

    def _load(self, video_id: str, kind: str) -> tuple[Any, list[dict[str, Any]]]:
        key = (video_id, kind)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        shard = self.video_manifests[video_id]['shards'][kind]
        metadata_path = Path(str(shard['metadata_path'])).resolve()
        metadata = relocate_image_references(json.loads(metadata_path.read_text(encoding='utf-8')))
        if not isinstance(metadata, list):
            raise ValueError(f'visual metadata must be a list: {metadata_path}')
        index_path = shard.get('index_path')
        if index_path is None:
            loaded = (None, metadata)
        else:
            import faiss
            index = faiss.read_index(str(Path(str(index_path)).resolve()))
            if int(index.ntotal) != len(metadata):
                raise ValueError(f'FAISS/metadata count mismatch for {video_id}/{kind}')
            loaded = (index, metadata)
        self._cache[key] = loaded
        return loaded

def retrieve_route_candidates(store: VisualShardStore, route_record: Mapping[str, Any], query_vector: Sequence[float] | np.ndarray, *, top_k_per_kind: int) -> list[dict[str, Any]]:
    demand = route_record.get('evidence_demand') or {}
    video_id = str(route_record['video_id'])
    output: list[dict[str, Any]] = []
    for kind in demand.get('candidate_kinds', []):
        output.extend(store.search(video_id=video_id, kind=str(kind), query_vector=query_vector, top_k=top_k_per_kind))
    return output

def render_route_evidence(route_record: Mapping[str, Any], hits: Iterable[Mapping[str, Any]], *, evidence_limit: int) -> tuple[list[str], list[str]]:
    route_id = str(route_record['route_id'])
    hit_rows = list(hits)
    if route_id == ROUTE_DESCRIPTIVE_COUNT:
        return _render_count_evidence(hit_rows, evidence_limit=evidence_limit)
    lines: list[str] = []
    used_ids: list[str] = []
    for hit in hit_rows:
        line = _render_hit(route_id, hit)
        if not line or line in lines:
            continue
        lines.append(line)
        used_ids.append(str(hit['candidate_id']))
        if len(lines) >= evidence_limit:
            break
    return (lines, used_ids)

def qa_safe_visual_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a compact, path-free trace; full metadata remains in the index shard."""
    metadata = dict(hit.get('metadata') or {})
    common = {'candidate_id', 'candidate_kind', 'video_id', 'graph_bound'}
    fields_by_kind = {'object': {'observation_id', 'entity_id', 'frame_name', 'frame_index', 'timestamp_sec', 'bbox', 'label', 'state_values', 'scene_id'}, 'tracklet': {'entity_id', 'label', 'labels_seen', 'states_seen', 'start_ts', 'end_ts', 'observation_count'}, 'frame': {'frame_name', 'frame_index', 'timestamp_sec', 'labels', 'states', 'entity_ids', 'observation_count', 'hsgg_slot_status'}, 'temporal_window': {'start_ts', 'end_ts', 'frame_names', 'labels', 'states', 'entity_ids', 'relations', 'events'}, 'event_boundary': {'event_id', 'event_type', 'timestamp_sec', 'entity_id', 'subject_entity_id', 'object_entity_id', 'raw_predicate', 'frame_names', 'labels', 'states', 'entity_ids'}, 'before_after_pair': {'event_id', 'event_type', 'boundary_ts', 'before_frame_name', 'before_ts', 'before_labels', 'before_states', 'before_entity_ids', 'after_frame_name', 'after_ts', 'after_labels', 'after_states', 'after_entity_ids'}}
    kind = str(hit.get('candidate_kind') or '')
    allowed = common | fields_by_kind.get(kind, set())
    metadata = {key: _compact_metadata_value(key, value) for key, value in metadata.items() if key in allowed}
    return {'candidate_id': str(hit.get('candidate_id') or ''), 'candidate_kind': str(hit.get('candidate_kind') or ''), 'video_id': str(hit.get('video_id') or ''), 'rank_within_kind': int(hit.get('rank_within_kind') or 0), 'similarity': float(hit.get('similarity') or 0.0), 'metadata': metadata}

def _compact_metadata_value(key: str, value: Any) -> Any:
    if not isinstance(value, list):
        return value
    limits = {'relations': 6, 'events': 6, 'entity_ids': 24, 'before_entity_ids': 16, 'after_entity_ids': 16, 'labels': 24, 'states': 24, 'frame_names': 8}
    return value[:limits.get(key, 24)]

def _render_count_evidence(hits: Sequence[Mapping[str, Any]], *, evidence_limit: int) -> tuple[list[str], list[str]]:
    entities: list[tuple[str, str, float, float, str]] = []
    used_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        metadata = hit.get('metadata') or {}
        entity_id = str(metadata.get('entity_id') or '')
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        label = str(metadata.get('label') or 'object')
        start_ts = float(metadata.get('start_ts', metadata.get('timestamp_sec', 0.0)) or 0.0)
        end_ts = float(metadata.get('end_ts', metadata.get('timestamp_sec', 0.0)) or 0.0)
        entities.append((entity_id, label, start_ts, end_ts, str(hit['candidate_id'])))
        used_ids.append(str(hit['candidate_id']))
        if len(entities) >= evidence_limit:
            break
    if not entities:
        return ([], [])
    compact = '; '.join((f'{label} [{start_ts:.3f}-{end_ts:.3f}s]' for _entity_id, label, start_ts, end_ts, _candidate_id in entities))
    return ([f'Temporally deduplicated graph-bound visual entity set: {compact}.'], used_ids)

def _render_hit(route_id: str, hit: Mapping[str, Any]) -> str | None:
    kind = str(hit['candidate_kind'])
    item = hit.get('metadata') or {}
    if kind == 'object':
        states = _joined(item.get('state_values'))
        suffix = f'; states: {states}' if states else ''
        return f"Object-crop recall at {float(item.get('timestamp_sec') or 0.0):.3f}s is bound to {item.get('label') or 'object'} ({item.get('entity_id')}){suffix}."
    if kind == 'tracklet':
        labels = _joined(item.get('labels_seen')) or str(item.get('label') or 'object')
        states = _joined(item.get('states_seen'))
        suffix = f'; states: {states}' if states else ''
        return f"Tracklet {item.get('entity_id')} is observed from {float(item.get('start_ts') or 0.0):.3f}s to {float(item.get('end_ts') or 0.0):.3f}s as {labels}{suffix}."
    if kind == 'frame':
        labels = _joined(item.get('labels'))
        states = _joined(item.get('states'))
        if not labels and (not states):
            return None
        suffix = f'; states: {states}' if states else ''
        return f"HSGG-bound full-frame recall at {float(item.get('timestamp_sec') or 0.0):.3f}s contains {labels or 'objects'}{suffix}."
    if kind == 'temporal_window':
        labels = _joined(item.get('labels'))
        states = _joined(item.get('states'))
        relations = _render_relations(item.get('relations'))
        events = _render_events(item.get('events'))
        details = '; '.join((value for value in (f'objects: {labels}' if labels else '', f'states: {states}' if states else '', f'relations: {relations}' if relations else '', f'events: {events}' if events else '') if value))
        if not details:
            return None
        prefix = 'Ordered mechanism window' if route_id == ROUTE_CAUSAL_HOW else 'Temporal window'
        return f"{prefix} {float(item.get('start_ts') or 0.0):.3f}-{float(item.get('end_ts') or 0.0):.3f}s; {details}."
    if kind == 'event_boundary':
        labels = _joined(item.get('labels'))
        states = _joined(item.get('states'))
        details = '; '.join((value for value in (f'nearby objects: {labels}' if labels else '', f'states: {states}' if states else '') if value))
        return f"Graph event boundary {item.get('event_type') or 'EVENT'} at {float(item.get('timestamp_sec') or 0.0):.3f}s" + (f'; {details}.' if details else '.')
    if kind == 'before_after_pair':
        boundary = float(item.get('boundary_ts') or 0.0)
        before = _joined(item.get('before_labels')) or 'no retained HSGG objects'
        after = _joined(item.get('after_labels')) or 'no retained HSGG objects'
        if route_id == ROUTE_TEMPORAL_PREVIOUS:
            return f"Before the graph event boundary at {boundary:.3f}s, the sampled frame at {float(item.get('before_ts') or 0.0):.3f}s contains {before}."
        if route_id == ROUTE_TEMPORAL_NEXT:
            return f"After the graph event boundary at {boundary:.3f}s, the sampled frame at {float(item.get('after_ts') or 0.0):.3f}s contains {after}."
        if route_id == ROUTE_CAUSAL_WHY:
            return f'Around the candidate effect boundary at {boundary:.3f}s, the preceding frame contains {before}, and the following frame contains {after}.'
        return f'Across the graph event boundary at {boundary:.3f}s, visual context changes from {before} to {after}.'
    return None

def _joined(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ''
    return ', '.join(dict.fromkeys((str(item).strip() for item in value if str(item).strip())))

def _render_relations(value: Any) -> str:
    if not isinstance(value, list):
        return ''
    return ', '.join((f"{item.get('subject_entity_id')} {item.get('predicate')} {item.get('object_entity_id')}" for item in value[:6] if isinstance(item, Mapping)))

def _render_events(value: Any) -> str:
    if not isinstance(value, list):
        return ''
    return ', '.join((f"{item.get('event_type')} at {float(item.get('timestamp_sec') or 0.0):.3f}s" for item in value[:6] if isinstance(item, Mapping)))
