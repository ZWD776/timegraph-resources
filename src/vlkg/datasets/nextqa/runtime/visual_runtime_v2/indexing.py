"""Read visual shards and build strict temporal pair candidates for visual v2."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from vlkg.core.assets.source_paths import relocate_image_references
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS, l2_normalize, mean_normalized

class VisualShardStore:

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 'nextqa_visual_index_manifest_v2':
            raise ValueError('visual v2 requires a v2 index manifest')
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
        hits: list[dict[str, Any]] = []
        for rank, (faiss_id, score) in enumerate(zip(indices[0].tolist(), scores[0].tolist(), strict=True), start=1):
            if faiss_id < 0:
                continue
            item = dict(metadata[faiss_id])
            hits.append({'candidate_id': str(item['candidate_id']), 'candidate_kind': kind, 'video_id': video_id, 'rank_within_kind': rank, 'similarity': round(float(score), 6), 'metadata': item})
        return hits

    def _load(self, video_id: str, kind: str) -> tuple[Any, list[dict[str, Any]]]:
        key = (video_id, kind)
        if key in self._cache:
            return self._cache[key]
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

def retrieve_role_candidates(store: VisualShardStore, route_record: Mapping[str, Any], query_vectors: Mapping[str, Sequence[float] | np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    video_id = str(route_record['video_id'])
    output: dict[str, list[dict[str, Any]]] = {}
    for query in route_record['evidence_demand']['query_intents']:
        role_id = str(query['role_id'])
        vector = query_vectors[role_id]
        role_hits: list[dict[str, Any]] = []
        for kind in query['candidate_kinds']:
            role_hits.extend(store.search(video_id=video_id, kind=str(kind), query_vector=vector, top_k=int(query['search_top_k'])))
        output[role_id] = role_hits
    return output

def reconstruct_faiss_vectors(index_path: str | Path) -> list[np.ndarray]:
    import faiss
    index = faiss.read_index(str(Path(index_path).resolve()))
    return [np.asarray(index.reconstruct(i), dtype=np.float32) for i in range(index.ntotal)]

def build_strict_before_after_candidates(graph: Mapping[str, Any], frame_metadata: Sequence[Mapping[str, Any]], frame_vectors: Sequence[np.ndarray]) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, int]]:
    if len(frame_metadata) != len(frame_vectors):
        raise ValueError('frame metadata/vector count mismatch')
    ordered = sorted(zip(frame_metadata, frame_vectors, strict=True), key=lambda item: (float(item[0]['timestamp_sec']), int(item[0]['frame_index'])))
    frames = [item for item, _vector in ordered]
    vectors_in_time = [vector for _item, vector in ordered]
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    counters = {'transition_event_count': 0, 'complete_pair_count': 0, 'missing_before_count': 0, 'missing_after_count': 0, 'dropped_no_strict_side_count': 0, 'same_frame_pair_count': 0, 'strict_order_violation_count': 0}
    video_id = _official_video_id(graph)
    for event in graph.get('transition_events', []):
        counters['transition_event_count'] += 1
        boundary = float(event.get('timestamp_sec') or 0.0)
        before_indices = [index for index, frame in enumerate(frames) if float(frame['timestamp_sec']) < boundary - 1e-09]
        after_indices = [index for index, frame in enumerate(frames) if float(frame['timestamp_sec']) > boundary + 1e-09]
        before_index = before_indices[-1] if before_indices else None
        after_index = after_indices[0] if after_indices else None
        if before_index is None:
            counters['missing_before_count'] += 1
        if after_index is None:
            counters['missing_after_count'] += 1
        if before_index is None and after_index is None:
            counters['dropped_no_strict_side_count'] += 1
            continue
        before = frames[before_index] if before_index is not None else None
        after = frames[after_index] if after_index is not None else None
        if before is not None and after is not None:
            counters['complete_pair_count'] += 1
            if str(before['frame_name']) == str(after['frame_name']):
                counters['same_frame_pair_count'] += 1
            if not float(before['timestamp_sec']) < boundary < float(after['timestamp_sec']):
                counters['strict_order_violation_count'] += 1
            pair_status = 'complete'
            pair_vector = mean_normalized([vectors_in_time[before_index], vectors_in_time[after_index]])
        elif before is None:
            pair_status = 'missing_before'
            pair_vector = l2_normalize(vectors_in_time[after_index])
        else:
            pair_status = 'missing_after'
            pair_vector = l2_normalize(vectors_in_time[before_index])
        metadata.append({'candidate_id': f"{event['event_id']}:visual_before_after_v2", 'candidate_kind': 'before_after_pair', 'video_id': video_id, 'event_id': str(event['event_id']), 'event_type': str(event.get('event_type') or ''), 'boundary_ts': boundary, 'pair_status': pair_status, 'missing_sides': [side for side, item in (('before', before), ('after', after)) if item is None], **_side_metadata('before', before), **_side_metadata('after', after), 'strict_temporal_pair': True, 'graph_bound': True})
        vectors.append(pair_vector)
    return (metadata, vectors, counters)

def _side_metadata(prefix: str, frame: Mapping[str, Any] | None) -> dict[str, Any]:
    if frame is None:
        return {f'{prefix}_frame_name': None, f'{prefix}_ts': None, f'{prefix}_labels': [], f'{prefix}_states': [], f'{prefix}_entity_ids': []}
    return {f'{prefix}_frame_name': str(frame['frame_name']), f'{prefix}_ts': float(frame['timestamp_sec']), f'{prefix}_labels': _strings(frame.get('labels')), f'{prefix}_states': _strings(frame.get('states')), f'{prefix}_entity_ids': _strings(frame.get('entity_ids'))}

def _official_video_id(graph: Mapping[str, Any]) -> str:
    value = str(graph.get('video_id') or '')
    if value.startswith('nextqa-val/'):
        return value.split('/', 1)[1]
    if not value:
        raise ValueError('temporal graph is missing video_id')
    return value

def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys((str(item).strip() for item in value if str(item).strip())))
