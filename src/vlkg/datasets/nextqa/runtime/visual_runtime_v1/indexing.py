"""Question-independent visual candidate indexes for NExT-QA."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from PIL import Image
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json
from vlkg.core.models.temporal_graph import BoundingBox
from vlkg.core.indexing.vision import EmbeddingCache, VisionFeatureExtractor
CANDIDATE_KINDS = ('object', 'frame', 'tracklet', 'temporal_window', 'event_boundary', 'before_after_pair')

@dataclass(frozen=True)
class VisualIndexConfig:
    profile_name: str
    embedding_model: str
    embedding_dimension: int
    window_size_sec: float
    window_stride_sec: float
    event_boundary_radius_sec: float
    frame_embedding_batch_size: int

def load_visual_index_config(path: str | Path) -> VisualIndexConfig:
    payload = json.loads(Path(path).resolve().read_text(encoding='utf-8'))
    kinds = tuple((str(value) for value in payload.get('candidate_kinds', [])))
    if kinds != CANDIDATE_KINDS:
        raise ValueError(f'candidate kind contract mismatch: {kinds}')
    config = VisualIndexConfig(profile_name=str(payload['profile_name']), embedding_model=str(payload['embedding_model']), embedding_dimension=int(payload['embedding_dimension']), window_size_sec=float(payload['window_size_sec']), window_stride_sec=float(payload['window_stride_sec']), event_boundary_radius_sec=float(payload['event_boundary_radius_sec']), frame_embedding_batch_size=int(payload['frame_embedding_batch_size']))
    if config.embedding_dimension <= 0 or config.frame_embedding_batch_size <= 0:
        raise ValueError('embedding dimensions and batch size must be positive')
    if config.window_size_sec <= 0 or config.window_stride_sec <= 0:
        raise ValueError('window size and stride must be positive')
    return config

def l2_normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('visual embedding must have a finite non-zero norm')
    return array / norm

def mean_normalized(vectors: Iterable[Sequence[float] | np.ndarray]) -> np.ndarray:
    rows = [l2_normalize(vector) for vector in vectors]
    if not rows:
        raise ValueError('cannot aggregate an empty vector collection')
    return l2_normalize(np.mean(np.vstack(rows), axis=0))

def bbox_key(bbox: Mapping[str, Any]) -> str:
    return f"{float(bbox['x1']):.3f},{float(bbox['y1']):.3f},{float(bbox['x2']):.3f},{float(bbox['y2']):.3f}"

def open_embedding_cache_read_only(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    return sqlite3.connect(f'file:{resolved.as_posix()}?mode=ro', uri=True)

def build_object_candidates(graph: Mapping[str, Any], connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[np.ndarray], int]:
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    misses = 0
    video_id = _official_video_id(graph)
    for observation in graph.get('observations', []):
        image_path = str(Path(str(observation['image_path'])))
        key = bbox_key(observation['bbox'])
        row = connection.execute('SELECT embedding, dim FROM embedding_cache WHERE image_path=? AND bbox_key=?', (image_path, key)).fetchone()
        if row is None:
            misses += 1
            continue
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        if vector.size != int(row[1]):
            misses += 1
            continue
        metadata.append({'candidate_id': f"{observation['observation_id']}:visual_object", 'candidate_kind': 'object', 'video_id': video_id, 'observation_id': str(observation['observation_id']), 'entity_id': str(observation['entity_id']), 'frame_name': str(observation['frame_name']), 'frame_index': int(observation['frame_index']), 'timestamp_sec': float(observation['timestamp_sec']), 'image_path': image_path, 'bbox': dict(observation['bbox']), 'label': str(observation.get('label') or 'object'), 'state_values': _strings(observation.get('state_values')), 'scene_id': str(observation.get('scene_id') or ''), 'graph_bound': True})
        vectors.append(l2_normalize(vector))
    return (metadata, vectors, misses)

def build_tracklet_candidates(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[np.ndarray], int]:
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    misses = 0
    video_id = _official_video_id(graph)
    for entity in graph.get('entities', []):
        embedding = entity.get('canonical_embedding')
        if not isinstance(embedding, list) or not embedding:
            misses += 1
            continue
        metadata.append({'candidate_id': f"{entity['entity_id']}:visual_tracklet", 'candidate_kind': 'tracklet', 'video_id': video_id, 'entity_id': str(entity['entity_id']), 'label': str(entity.get('category') or 'object'), 'labels_seen': _strings(entity.get('labels_seen')), 'states_seen': _strings(entity.get('states_seen')), 'start_ts': float(entity.get('first_seen_ts') or 0.0), 'end_ts': float(entity.get('last_seen_ts') or 0.0), 'observation_count': int(entity.get('observation_count') or 0), 'canonical_image_path': str(entity.get('canonical_image_path') or ''), 'graph_bound': True})
        vectors.append(l2_normalize(embedding))
    return (metadata, vectors, misses)

def encode_full_frames(frame_paths: Sequence[Path], *, cache: EmbeddingCache, extractor: VisionFeatureExtractor, batch_size: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for start in range(0, len(frame_paths), batch_size):
        paths = frame_paths[start:start + batch_size]
        items: list[tuple[Path, BoundingBox]] = []
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            items.append((path, BoundingBox(x1=0.0, y1=0.0, x2=float(width), y2=float(height))))
        cached = cache.get_many(items)
        missing = [item for item in items if (str(Path(item[0])), EmbeddingCache.bbox_to_key(item[1])) not in cached]
        if missing:
            computed = extractor.encode_batch(missing)
            cache.set_many([(image_path, bbox, vector) for (image_path, bbox), vector in zip(missing, computed, strict=True)])
            for (image_path, bbox), vector in zip(missing, computed, strict=True):
                cached[str(Path(image_path)), EmbeddingCache.bbox_to_key(bbox)] = vector
        for image_path, bbox in items:
            key = (str(Path(image_path)), EmbeddingCache.bbox_to_key(bbox))
            output[str(Path(image_path))] = l2_normalize(cached[key])
    return output

def build_frame_candidates(graph: Mapping[str, Any], frame_paths: Sequence[Path], frame_embeddings: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    video_id = _official_video_id(graph)
    observations_by_stem: dict[str, list[Mapping[str, Any]]] = {}
    for observation in graph.get('observations', []):
        observations_by_stem.setdefault(str(observation['frame_name']), []).append(observation)
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for path in sorted(frame_paths, key=_frame_sort_key):
        stem = path.stem
        frame_index, timestamp_sec = _parse_frame_stem(stem)
        observations = observations_by_stem.get(stem, [])
        summary = _observation_summary(observations)
        metadata.append({'candidate_id': f'nextqa-val/{video_id}:{stem}:visual_frame', 'candidate_kind': 'frame', 'video_id': video_id, 'frame_name': stem, 'frame_index': frame_index, 'timestamp_sec': timestamp_sec, 'image_path': str(path), **summary, 'graph_bound': bool(observations), 'hsgg_slot_status': 'completed' if observations else 'skipped_or_empty'})
        vectors.append(l2_normalize(frame_embeddings[str(path)]))
    return (metadata, vectors)

def build_temporal_window_candidates(graph: Mapping[str, Any], frame_metadata: Sequence[Mapping[str, Any]], frame_vectors: Sequence[np.ndarray], *, window_size_sec: float, window_stride_sec: float) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    last_start: float | None = None
    for index, frame in enumerate(frame_metadata):
        start_ts = float(frame['timestamp_sec'])
        if last_start is not None and start_ts - last_start < window_stride_sec - 1e-06:
            continue
        members = [(item, vector) for item, vector in zip(frame_metadata[index:], frame_vectors[index:], strict=True) if float(item['timestamp_sec']) <= start_ts + window_size_sec + 1e-06]
        if not members:
            continue
        last_start = start_ts
        member_metadata = [item for item, _vector in members]
        end_ts = float(member_metadata[-1]['timestamp_sec'])
        summary = _merge_frame_summaries(member_metadata)
        metadata.append({'candidate_id': f"nextqa-val/{frame['video_id']}:window:{frame['frame_name']}:{member_metadata[-1]['frame_name']}", 'candidate_kind': 'temporal_window', 'video_id': str(frame['video_id']), 'start_ts': start_ts, 'end_ts': end_ts, 'frame_names': [str(item['frame_name']) for item in member_metadata], 'image_paths': [str(item['image_path']) for item in member_metadata], **summary, **_graph_signals_in_range(graph, start_ts, end_ts), 'graph_bound': any((bool(item.get('graph_bound')) for item in member_metadata))})
        vectors.append(mean_normalized((vector for _item, vector in members)))
    return (metadata, vectors)

def build_event_boundary_candidates(graph: Mapping[str, Any], frame_metadata: Sequence[Mapping[str, Any]], frame_vectors: Sequence[np.ndarray], *, radius_sec: float) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for event in graph.get('transition_events', []):
        timestamp = float(event.get('timestamp_sec') or 0.0)
        members = [(item, vector) for item, vector in zip(frame_metadata, frame_vectors, strict=True) if abs(float(item['timestamp_sec']) - timestamp) <= radius_sec + 1e-06]
        if not members and frame_metadata:
            nearest_index = min(range(len(frame_metadata)), key=lambda idx: abs(float(frame_metadata[idx]['timestamp_sec']) - timestamp))
            members = [(frame_metadata[nearest_index], frame_vectors[nearest_index])]
        if not members:
            continue
        member_metadata = [item for item, _vector in members]
        summary = _merge_frame_summaries(member_metadata)
        metadata.append({'candidate_id': f"{event['event_id']}:visual_boundary", 'candidate_kind': 'event_boundary', 'video_id': _official_video_id(graph), 'event_id': str(event['event_id']), 'event_type': str(event.get('event_type') or ''), 'timestamp_sec': timestamp, 'entity_id': str(event.get('entity_id') or ''), 'subject_entity_id': str(event.get('subject_entity_id') or ''), 'object_entity_id': str(event.get('object_entity_id') or ''), 'raw_predicate': str(event.get('raw_predicate') or ''), 'frame_names': [str(item['frame_name']) for item in member_metadata], **summary, 'graph_bound': True})
        vectors.append(mean_normalized((vector for _item, vector in members)))
    return (metadata, vectors)

def build_before_after_candidates(graph: Mapping[str, Any], frame_metadata: Sequence[Mapping[str, Any]], frame_vectors: Sequence[np.ndarray]) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    if not frame_metadata:
        return (metadata, vectors)
    for event in graph.get('transition_events', []):
        timestamp = float(event.get('timestamp_sec') or 0.0)
        before_indices = [index for index, frame in enumerate(frame_metadata) if float(frame['timestamp_sec']) <= timestamp + 1e-06]
        after_indices = [index for index, frame in enumerate(frame_metadata) if float(frame['timestamp_sec']) >= timestamp - 1e-06]
        before_index = before_indices[-1] if before_indices else 0
        after_index = after_indices[0] if after_indices else len(frame_metadata) - 1
        before = frame_metadata[before_index]
        after = frame_metadata[after_index]
        metadata.append({'candidate_id': f"{event['event_id']}:visual_before_after", 'candidate_kind': 'before_after_pair', 'video_id': _official_video_id(graph), 'event_id': str(event['event_id']), 'event_type': str(event.get('event_type') or ''), 'boundary_ts': timestamp, 'before_frame_name': str(before['frame_name']), 'before_ts': float(before['timestamp_sec']), 'before_labels': _strings(before.get('labels')), 'before_states': _strings(before.get('states')), 'before_entity_ids': _strings(before.get('entity_ids')), 'after_frame_name': str(after['frame_name']), 'after_ts': float(after['timestamp_sec']), 'after_labels': _strings(after.get('labels')), 'after_states': _strings(after.get('states')), 'after_entity_ids': _strings(after.get('entity_ids')), 'graph_bound': True})
        vectors.append(mean_normalized([frame_vectors[before_index], frame_vectors[after_index]]))
    return (metadata, vectors)

def write_faiss_shard(output_root: str | Path, *, kind: str, metadata: Sequence[Mapping[str, Any]], vectors: Sequence[np.ndarray], expected_dimension: int) -> dict[str, Any]:
    import faiss
    if kind not in CANDIDATE_KINDS:
        raise ValueError(f'unsupported candidate kind: {kind}')
    if len(metadata) != len(vectors):
        raise ValueError(f'metadata/vector count mismatch for {kind}')
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / f'{kind}.metadata.json'
    atomic_write_json(metadata_path, list(metadata))
    if not vectors:
        return {'kind': kind, 'count': 0, 'dimension': expected_dimension, 'index_path': None, 'metadata_path': str(metadata_path)}
    matrix = np.vstack([l2_normalize(vector) for vector in vectors]).astype(np.float32, copy=False)
    if matrix.shape[1] != expected_dimension:
        raise ValueError(f'{kind} dimension mismatch: {matrix.shape[1]} != {expected_dimension}')
    index = faiss.IndexFlatIP(expected_dimension)
    index.add(matrix)
    index_path = root / f'{kind}.faiss'
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=f'.{kind}.', suffix='.faiss.tmp', delete=False) as handle:
            temporary = Path(handle.name)
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {'kind': kind, 'count': int(index.ntotal), 'dimension': expected_dimension, 'index_path': str(index_path), 'metadata_path': str(metadata_path), 'index_bytes': index_path.stat().st_size, 'metadata_bytes': metadata_path.stat().st_size}

def _official_video_id(graph: Mapping[str, Any]) -> str:
    value = str(graph.get('video_id') or '')
    if value.startswith('nextqa-val/'):
        return value.split('/', 1)[1]
    if not value:
        raise ValueError('temporal graph is missing video_id')
    return value

def _parse_frame_stem(stem: str) -> tuple[int, float]:
    parts = stem.split('_')
    if len(parts) < 3 or not parts[-1].endswith('s'):
        raise ValueError(f'invalid NExT-QA frame name: {stem}')
    return (int(parts[-2]), float(parts[-1][:-1]))

def _frame_sort_key(path: Path) -> tuple[float, int, str]:
    frame_index, timestamp = _parse_frame_stem(path.stem)
    return (timestamp, frame_index, path.name)

def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys((str(item).strip() for item in value if str(item).strip())))

def _observation_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'labels': list(dict.fromkeys((str(item.get('label') or '') for item in observations if item.get('label')))), 'states': list(dict.fromkeys((state for item in observations for state in _strings(item.get('state_values'))))), 'entity_ids': list(dict.fromkeys((str(item.get('entity_id') or '') for item in observations if item.get('entity_id')))), 'observation_count': len(observations)}

def _merge_frame_summaries(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'labels': list(dict.fromkeys((label for frame in frames for label in _strings(frame.get('labels'))))), 'states': list(dict.fromkeys((state for frame in frames for state in _strings(frame.get('states'))))), 'entity_ids': list(dict.fromkeys((entity_id for frame in frames for entity_id in _strings(frame.get('entity_ids')))))}

def _graph_signals_in_range(graph: Mapping[str, Any], start_ts: float, end_ts: float) -> dict[str, Any]:
    relations = [{'subject_entity_id': str(item.get('subject_entity_id') or ''), 'predicate': str(item.get('raw_predicate') or item.get('predicate') or ''), 'object_entity_id': str(item.get('object_entity_id') or ''), 'start_ts': float(item.get('start_ts') or 0.0), 'end_ts': float(item.get('end_ts') or 0.0)} for item in graph.get('relation_episodes', []) if float(item.get('end_ts') or 0.0) >= start_ts - 1e-06 and float(item.get('start_ts') or 0.0) <= end_ts + 1e-06]
    events = [{'event_id': str(item.get('event_id') or ''), 'event_type': str(item.get('event_type') or ''), 'timestamp_sec': float(item.get('timestamp_sec') or 0.0), 'entity_id': str(item.get('entity_id') or '')} for item in graph.get('transition_events', []) if start_ts - 1e-06 <= float(item.get('timestamp_sec') or 0.0) <= end_ts + 1e-06]
    return {'relations': relations, 'events': events}
