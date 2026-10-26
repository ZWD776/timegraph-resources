from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable
import faiss
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import repo_path, write_json_atomic
from vlkg.core.temporal.pathing import default_cache_db_path
from vlkg.datasets.openeqa.runtime.visual_object_index import GraphObservationBindingIndex, OBJECT_VISUAL_INDEX_CONTRACT
DEFAULT_INDEX_NAME = 'openeqa_object_siglip_flat_ip'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build an OpenEQA object-level SigLIP FAISS index from existing embeddings and bind every indexed row to an HSGG observation and, when available, a canonical graph entity.')
    parser.add_argument('--graph-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-db', type=Path, default=default_cache_db_path())
    parser.add_argument('--image-root', type=Path, default=None)
    parser.add_argument('--index-name', default=DEFAULT_INDEX_NAME)
    parser.add_argument('--episode-history', action='append', default=None)
    parser.add_argument('--limit-episodes', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=20000)
    parser.add_argument('--force', action='store_true')
    return parser

def bbox_values(value: Iterable[Any]) -> list[float] | None:
    values = list(value)
    if len(values) < 4:
        return None
    try:
        return [float(values[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None

def bbox_key(value: Iterable[Any]) -> str | None:
    values = bbox_values(value)
    if values is None:
        return None
    return ','.join((f'{item:.3f}' for item in values))

def load_frame_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('frames') if isinstance(payload, dict) else payload
    result: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        for key in (row.get('source_frame_name'), row.get('image_name'), Path(str(row.get('image_name') or '')).stem, row.get('frame_name')):
            if key:
                result[str(key)] = row
    return result

def image_name(frame_key: str, payload: dict[str, Any]) -> str:
    raw_path = str(payload.get('image_path') or '')
    if raw_path:
        return Path(raw_path.replace('\\', '/')).name
    return frame_key if Path(frame_key).suffix else f'{frame_key}.png'

def cache_image_candidates(*, episode: dict[str, Any], image_root: Path | None, image_name_value: str, payload: dict[str, Any], binding_image_path: str) -> list[str]:
    candidates: list[Path] = []
    episode_history = Path(str(episode['episode_history']))
    if image_root is not None:
        candidates.append(image_root / episode_history / image_name_value)
    frames_dir = Path(str(episode.get('frames_dir') or ''))
    if str(frames_dir):
        candidates.append(frames_dir / image_name_value)
    for value in (binding_image_path, payload.get('image_path')):
        if value:
            candidates.append(Path(str(value)))
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        key = os.path.normcase(os.path.normpath(normalized))
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result

def cached_embedding(cursor: sqlite3.Cursor, *, image_paths: list[str], bbox_key_value: str) -> tuple[np.ndarray | None, str]:
    for image_path_value in image_paths:
        row = cursor.execute('SELECT embedding, dim FROM embedding_cache WHERE image_path = ? AND bbox_key = ?', (str(Path(image_path_value)), bbox_key_value)).fetchone()
        if row is None:
            continue
        vector = np.frombuffer(row[0], dtype=np.float32)
        if vector.size == int(row[1]):
            return (vector.copy(), str(Path(image_path_value)))
    return (None, '')

def create_metadata_schema(connection: sqlite3.Connection) -> None:
    connection.executescript('\n        CREATE TABLE object_metadata (\n            faiss_id INTEGER PRIMARY KEY,\n            data_source TEXT NOT NULL,\n            episode_history TEXT NOT NULL,\n            frame_key TEXT NOT NULL,\n            image_name TEXT NOT NULL,\n            image_path TEXT NOT NULL,\n            graph_frame_index INTEGER NOT NULL,\n            object_index INTEGER NOT NULL,\n            label TEXT,\n            object_score REAL,\n            bbox_json TEXT NOT NULL,\n            bbox_key TEXT NOT NULL,\n            observation_id TEXT NOT NULL,\n            source_entity_id TEXT NOT NULL,\n            canonical_entity_id TEXT NOT NULL,\n            scene_id TEXT,\n            binding_kind TEXT NOT NULL,\n            binding_confidence REAL NOT NULL,\n            temporal_graph_json TEXT NOT NULL,\n            canonical_graph_json TEXT NOT NULL,\n            raw_json TEXT NOT NULL\n        );\n        CREATE UNIQUE INDEX idx_object_episode_frame_object\n            ON object_metadata(episode_history, graph_frame_index, object_index);\n        CREATE INDEX idx_object_episode_entity\n            ON object_metadata(episode_history, canonical_entity_id);\n        CREATE INDEX idx_object_label ON object_metadata(label);\n        ')

def insert_metadata(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    connection.execute('\n        INSERT INTO object_metadata (\n            faiss_id, data_source, episode_history, frame_key, image_name,\n            image_path, graph_frame_index, object_index, label, object_score,\n            bbox_json, bbox_key, observation_id, source_entity_id,\n            canonical_entity_id, scene_id, binding_kind, binding_confidence,\n            temporal_graph_json, canonical_graph_json, raw_json\n        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n        ', (row['faiss_id'], row['data_source'], row['episode_history'], row['frame_key'], row['image_name'], row['image_path'], row['graph_frame_index'], row['object_index'], row['label'], row['object_score'], json.dumps(row['bbox']), row['bbox_key'], row['observation_id'], row['source_entity_id'], row['canonical_entity_id'], row['scene_id'], row['binding_kind'], row['binding_confidence'], row['temporal_graph_json'], row['canonical_graph_json'], json.dumps(row, ensure_ascii=False)))

def add_vectors(index: Any, vectors: list[np.ndarray]) -> None:
    if not vectors:
        return
    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    faiss.normalize_L2(matrix)
    index.add(matrix)

def build_index(args: argparse.Namespace) -> dict[str, Any]:
    graph_manifest_path = repo_path(args.graph_manifest)
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding='utf-8'))
    episodes = [row for row in graph_manifest.get('episodes', []) if isinstance(row, dict) and row.get('status') == 'built']
    if args.episode_history:
        selected = {str(value) for value in args.episode_history}
        episodes = [row for row in episodes if row.get('episode_history') in selected]
    if args.limit_episodes is not None:
        episodes = episodes[:args.limit_episodes]
    if not episodes:
        raise ValueError('No built episodes selected')
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f'{args.index_name}.faiss'
    metadata_path = output_dir / f'{args.index_name}.metadata.sqlite'
    manifest_path = output_dir / f'{args.index_name}.manifest.json'
    for path in (index_path, metadata_path, manifest_path):
        if path.exists() and (not args.force):
            raise FileExistsError(f'Output exists: {path}; pass --force')
    cache_db = repo_path(args.cache_db)
    image_root = repo_path(args.image_root) if args.image_root is not None else None
    temp_metadata = metadata_path.with_name(metadata_path.name + f'.{os.getpid()}.tmp')
    temp_index = index_path.with_name(index_path.name + f'.{os.getpid()}.tmp')
    temp_metadata.unlink(missing_ok=True)
    temp_index.unlink(missing_ok=True)
    cache_connection = sqlite3.connect(cache_db)
    metadata_connection = sqlite3.connect(temp_metadata)
    create_metadata_schema(metadata_connection)
    cache_cursor = cache_connection.cursor()
    index: Any | None = None
    dimension: int | None = None
    vectors: list[np.ndarray] = []
    faiss_id = 0
    episode_stats: list[dict[str, Any]] = []
    try:
        for ordinal, episode in enumerate(episodes, start=1):
            hsgg_dir = Path(str(episode['hsgg_dir']))
            objects = json.loads((hsgg_dir / 'objects.json').read_text(encoding='utf-8'))
            frame_index = load_frame_index(Path(str(episode['frame_time_index_json'])))
            binding_index = GraphObservationBindingIndex.from_graphs(temporal_graph_path=episode['temporal_graph_json'], canonical_graph_path=episode['canonical_graph_json'])
            stats = {'episode_history': episode['episode_history'], 'hsgg_objects': 0, 'selected_visual_anchor_objects': 0, 'skipped_non_visual_anchor_objects': 0, 'visual_anchor_frames': 0, 'valid_boxes': 0, 'exact_graph_bindings': 0, 'observation_only_bindings': 0, 'indexed_exact_graph_bindings': 0, 'indexed_observation_only_bindings': 0, 'missing_graph_binding': 0, 'embedding_cache_hits': 0, 'missing_embedding': 0, 'indexed': 0}
            for frame_key_value, payload in objects.items():
                if not isinstance(payload, dict):
                    continue
                image_name_value = image_name(str(frame_key_value), payload)
                boxes = payload.get('boxes') or []
                stats['hsgg_objects'] += len(boxes)
                if image_root is not None:
                    visual_anchor_path = image_root / Path(str(episode['episode_history'])) / image_name_value
                    if not visual_anchor_path.is_file():
                        stats['skipped_non_visual_anchor_objects'] += len(boxes)
                        continue
                stats['visual_anchor_frames'] += 1
                stats['selected_visual_anchor_objects'] += len(boxes)
                frame_info = frame_index.get(str(frame_key_value)) or frame_index.get(image_name_value) or frame_index.get(Path(image_name_value).stem) or {}
                graph_frame_index = frame_info.get('graph_frame_index')
                if graph_frame_index is None:
                    graph_frame_index = frame_info.get('frame_index')
                labels = payload.get('labels') or []
                scores = payload.get('scores') or []
                for object_index, raw_box in enumerate(boxes):
                    box = bbox_values(raw_box)
                    box_key_value = bbox_key(raw_box)
                    if box is None or box_key_value is None or graph_frame_index is None:
                        continue
                    stats['valid_boxes'] += 1
                    binding = binding_index.bind(frame_index=int(graph_frame_index), object_index=object_index)
                    if binding is None:
                        stats['missing_graph_binding'] += 1
                        continue
                    if binding.graph_bound:
                        stats['exact_graph_bindings'] += 1
                    else:
                        stats['observation_only_bindings'] += 1
                    candidates = cache_image_candidates(episode=episode, image_root=image_root, image_name_value=image_name_value, payload=payload, binding_image_path=binding.image_path)
                    embedding, cache_image_path = cached_embedding(cache_cursor, image_paths=candidates, bbox_key_value=box_key_value)
                    if embedding is None:
                        stats['missing_embedding'] += 1
                        continue
                    stats['embedding_cache_hits'] += 1
                    if dimension is None:
                        dimension = int(embedding.size)
                        index = faiss.IndexFlatIP(dimension)
                    if int(embedding.size) != dimension:
                        stats['missing_embedding'] += 1
                        continue
                    if binding.graph_bound:
                        stats['indexed_exact_graph_bindings'] += 1
                    else:
                        stats['indexed_observation_only_bindings'] += 1
                    label = str(labels[object_index]) if object_index < len(labels) else binding.label
                    score = scores[object_index] if object_index < len(scores) else None
                    row = {'contract': OBJECT_VISUAL_INDEX_CONTRACT, 'faiss_id': faiss_id, 'data_source': graph_manifest.get('data_source', ''), 'episode_history': episode['episode_history'], 'frame_key': str(frame_key_value), 'image_name': image_name_value, 'image_path': cache_image_path, 'graph_image_path': binding.image_path, 'graph_frame_index': int(graph_frame_index), 'object_index': object_index, 'label': label, 'object_score': float(score) if isinstance(score, (int, float)) else None, 'bbox': box, 'bbox_key': box_key_value, 'observation_id': binding.observation_id, 'source_entity_id': binding.source_entity_id, 'canonical_entity_id': binding.canonical_entity_id, 'scene_id': binding.scene_id, 'temporal_graph_json': str(episode['temporal_graph_json']), 'canonical_graph_json': str(episode['canonical_graph_json']), 'binding_kind': binding.binding_kind, 'binding_confidence': binding.binding_confidence, 'is_graph_fact': binding.graph_bound, 'uses_gold_answer': False}
                    insert_metadata(metadata_connection, row)
                    vectors.append(embedding)
                    faiss_id += 1
                    stats['indexed'] += 1
                    if len(vectors) >= max(1, int(args.batch_size)):
                        add_vectors(index, vectors)
                        vectors = []
            episode_stats.append(stats)
            metadata_connection.commit()
            print(f"[{ordinal}/{len(episodes)}] {episode['episode_history']} indexed={stats['indexed']} missing_embedding={stats['missing_embedding']} exact_binding={stats['exact_graph_bindings']} observation_only={stats['observation_only_bindings']} skipped_non_anchor={stats['skipped_non_visual_anchor_objects']} missing_binding={stats['missing_graph_binding']}", flush=True)
        if index is None or dimension is None:
            raise RuntimeError('No graph-bound cached embeddings were indexed')
        add_vectors(index, vectors)
        faiss.write_index(index, str(temp_index))
        metadata_connection.commit()
        metadata_count = int(metadata_connection.execute('SELECT COUNT(*) FROM object_metadata').fetchone()[0])
    finally:
        cache_connection.close()
        metadata_connection.close()
    if metadata_count != int(index.ntotal):
        raise RuntimeError(f'FAISS/metadata mismatch: index={index.ntotal}, metadata={metadata_count}')
    os.replace(temp_index, index_path)
    os.replace(temp_metadata, metadata_path)
    totals = {key: sum((int(row[key]) for row in episode_stats)) for key in ('hsgg_objects', 'selected_visual_anchor_objects', 'skipped_non_visual_anchor_objects', 'visual_anchor_frames', 'valid_boxes', 'exact_graph_bindings', 'observation_only_bindings', 'indexed_exact_graph_bindings', 'indexed_observation_only_bindings', 'missing_graph_binding', 'embedding_cache_hits', 'missing_embedding', 'indexed')}
    manifest = {'contract': OBJECT_VISUAL_INDEX_CONTRACT, 'generated_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'data_source': graph_manifest.get('data_source'), 'graph_manifest': str(graph_manifest_path), 'graph_manifest_run_id': graph_manifest.get('run_id'), 'cache_db': str(cache_db), 'image_root': str(image_root) if image_root is not None else '', 'index_path': str(index_path), 'metadata_path': str(metadata_path), 'manifest_path': str(manifest_path), 'index_name': args.index_name, 'dimension': dimension, 'ntotal': int(index.ntotal), 'metadata_rows': metadata_count, 'faiss_metric': 'inner_product_on_l2_normalized_vectors', 'episode_count': len(episode_stats), 'uses_existing_embeddings_only': True, 'computes_new_embeddings': False, 'uses_gold_answer': False, 'hsgg_observation_binding_required': True, 'canonical_binding_required': False, 'observation_only_rows_are_graph_facts': False, 'totals': totals, 'episodes': episode_stats}
    write_json_atomic(manifest_path, manifest)
    return manifest

def main() -> None:
    args = build_parser().parse_args()
    manifest = build_index(args)
    print(json.dumps({'status': manifest['status'], **manifest['totals']}, sort_keys=True))
    print(f"manifest={manifest['manifest_path']}")
if __name__ == '__main__':
    main()
