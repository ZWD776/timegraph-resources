"""Frozen OpenEQA-owned parity fork of NAVQA visual evidence fusion.

Do not modify this v1 baseline.  Create a new versioned OpenEQA runtime for
strategy changes.
"""
from __future__ import annotations
import json
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.indexing.evidence_layer_reader import EvidenceLayerIndex
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.temporal.processed_attributes import get_or_create_processed_attribute_record
from vlkg.core.ir.query_annotations import synonym_expansion_enabled, synonym_expansion_metadata
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_entity_candidate_packs import visual_entity_pack_query_phrase_records
DEFAULT_VISUAL_FAISS_INDEX_DIR = Path('.scratch/navqa_graph_experiments/visual_faiss_grounding')
DEFAULT_VISUAL_FAISS_INDEX_NAME = 'object_siglip_flat_ip'
VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT = 2
VISUAL_FAISS_fragment_FRAGMENT_BRIDGE_LIMIT = 8
_OBSERVATION_ANCHOR_MAP_CONTRACT = 'visual_faiss_temporal_observation_anchor_map_v1'
_OBSERVATION_ANCHOR_MAP_SUFFIX = '.observation_anchor_map.v1.json'
_OBSERVATION_ANCHOR_STREAM_CHUNK_SIZE = 1024 * 1024
_OBSERVATION_ANCHOR_CACHE: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
_ENTITY_TRACKLET_INDEX_CONTRACT = 'visual_faiss_entity_tracklet_index_v1'
_ENTITY_TRACKLET_INDEX_CACHE_VERSION = 1
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'before', 'behind', 'by', 'can', 'did', 'do', 'does', 'during', 'for', 'from', 'has', 'have', 'how', 'i', 'in', 'into', 'is', 'it', 'near', 'of', 'on', 'or', 'see', 'seen', 'the', 'there', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'while', 'with', 'you'}
_VISUAL_FAISS_SCHEMA_MATCH_TOKENS = {'answer', 'answers', 'attribute', 'attributes', 'camera', 'detection', 'detections', 'ego', 'entity', 'entities', 'evidence', 'frame', 'frames', 'graph', 'image', 'images', 'label', 'labels', 'object', 'objects', 'observation', 'observations', 'photo', 'photos', 'query', 'relation', 'relations', 'state', 'target', 'targets', 'text', 'trajectory', 'type', 'types', 'visual'}

@dataclass(frozen=True)
class VisualFaissConfig:
    index_dir: Path = DEFAULT_VISUAL_FAISS_INDEX_DIR
    index_name: str = DEFAULT_VISUAL_FAISS_INDEX_NAME
    seed_limit: int = 4
    top_k: int = 5
    max_lines: int = 8
    same_sequence_only: bool = True
    per_frame_object_limit: int = 80
    query_phrase_limit: int = 6
    query_recall_top_k: int = 8
    query_recall_search_k: int = 1000
    query_recall_min_score: float = 0.0
    ger_object_limit: int = 8
    ger_context_radius: int = 8
    ger_context_limit: int = 12
    frame_context_enabled: bool = False
    frame_context_neighbor_limit: int = 4
    frame_context_relation_limit: int = 4
    structural_binding_enabled: bool = False
    local_subgraph_entity_limit: int = 8
    local_subgraph_neighbor_limit: int = 6
    local_subgraph_time_radius: int = 8
    evidence_layer_sqlite: Path = Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime/evidence_layer.sqlite')
    evidence_layer_object_limit: int = 8
    evidence_layer_neighbor_limit: int = 4
    role_aware_object_limit: int = 8
    role_aware_context_limit: int = 8
    role_aware_compression_enabled: bool = False
    entity_pack_enabled: bool = False
    entity_pack_semantic_align_enabled: bool = False
    entity_pack_semantic_gated_role_enabled: bool = False
    entity_pack_score_contract_enabled: bool = False
    entity_pack_tracklet_index_enabled: bool = True
    entity_pack_tracklet_top_k: int = 6
    entity_pack_tracklet_search_k: int = 1000

class VisualFaissGrounder:
    """Experimental object-level visual similarity lookup for NaVQA QA.

    The class keeps FAISS and SQLite metadata open in memory for a QA run. It
    never reads answer labels; seed objects come from graph focus frames and
    question text/IR only.
    """

    def __init__(self, *, index_dir: Path=DEFAULT_VISUAL_FAISS_INDEX_DIR, index_name: str=DEFAULT_VISUAL_FAISS_INDEX_NAME) -> None:
        import faiss
        self.index_dir = resolve_repo_input_path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f'{index_name}.faiss'
        self.metadata_db_path = self.index_dir / f'{index_name}.metadata.sqlite'
        self.manifest_path = self.index_dir / f'{index_name}.manifest.json'
        if not self.index_path.exists():
            raise FileNotFoundError(f'Missing FAISS index: {self.index_path}')
        if not self.metadata_db_path.exists():
            raise FileNotFoundError(f'Missing FAISS metadata DB: {self.metadata_db_path}')
        self.index = faiss.read_index(str(self.index_path))
        self.connection = sqlite3.connect(self.metadata_db_path)
        self.connection.row_factory = sqlite3.Row
        self._faiss = faiss
        self._text_encoder = None
        self._hsgg_cache: dict[str, dict[str, Any]] = {}
        self._entity_tracklet_index_cache: dict[str, tuple[Any, list[dict[str, Any]], dict[str, Any]]] = {}
        metadata_count = self.count_metadata_rows()
        if int(self.index.ntotal) != metadata_count:
            raise ValueError(f'FAISS index and metadata DB size mismatch: index={self.index.ntotal}, metadata={metadata_count}')
        self.manifest = self._load_manifest()

    def close(self) -> None:
        self.connection.close()

    def count_metadata_rows(self) -> int:
        return int(self.connection.execute('SELECT COUNT(*) FROM object_metadata').fetchone()[0])

    def describe(self) -> dict[str, Any]:
        return {'contract': 'experimental_visual_faiss_grounding_v1', 'index_path': str(self.index_path), 'metadata_db_path': str(self.metadata_db_path), 'manifest_path': str(self.manifest_path) if self.manifest_path.exists() else '', 'index_ntotal': int(self.index.ntotal), 'metadata_rows': self.count_metadata_rows(), 'dimension': int(getattr(self.index, 'd', 0) or 0), 'manifest': self.manifest}

    def build_query_recall_evidence(self, *, seq_id: str, question: str, graph_answer: dict[str, Any], question_ir: dict[str, Any] | None, config: VisualFaissConfig, graph: CanonicalTemporalMemoryGraph | None=None) -> dict[str, Any]:
        phrase_records = []
        if config.entity_pack_enabled:
            phrase_records = visual_entity_pack_query_phrase_records(question=question, question_ir=question_ir or {}, limit=max(1, config.query_phrase_limit))
        elif config.role_aware_compression_enabled:
            phrase_records = _visual_query_phrase_records(question=question, question_ir=question_ir or {}, limit=max(1, config.query_phrase_limit))
        phrases = [str(record['rendered']) for record in phrase_records] if phrase_records else _visual_query_phrases(question=question, question_ir=question_ir or {}, limit=max(1, config.query_phrase_limit))
        if not phrases:
            return {'contract': 'experimental_visual_query_faiss_recall_v1', 'available': False, 'seq_id': str(seq_id), 'query_phrases': [], 'groups': [], 'lines': [], 'visual_focus_frames': []}
        vectors = self._encode_texts(phrases)
        groups: list[dict[str, Any]] = []
        seen_result_keys: set[tuple[Any, ...]] = set()
        phrase_record_by_rendered = {str(record.get('rendered') or ''): record for record in phrase_records}
        for phrase, vector in zip(phrases, vectors, strict=True):
            results = self._search_by_vector(vector=vector, seq_id=str(seq_id), top_k=max(1, config.query_recall_top_k), search_k=max(1, config.query_recall_search_k), same_sequence_only=config.same_sequence_only, min_score=float(config.query_recall_min_score))
            if config.entity_pack_enabled and config.entity_pack_tracklet_index_enabled and (graph is not None):
                results.extend(self._search_entity_tracklets_by_vector(vector=vector, seq_id=str(seq_id), graph=graph, top_k=max(1, config.entity_pack_tracklet_top_k), search_k=max(1, config.entity_pack_tracklet_search_k), min_score=float(config.query_recall_min_score)))
            filtered_results: list[dict[str, Any]] = []
            for row in results:
                self._attach_hsgg_object_facts(row, include_frame_context=config.frame_context_enabled, frame_context_neighbor_limit=config.frame_context_neighbor_limit, frame_context_relation_limit=config.frame_context_relation_limit)
                key = _visual_query_recall_result_key(row)
                if key in seen_result_keys:
                    continue
                seen_result_keys.add(key)
                filtered_results.append(row)
            if filtered_results:
                group = {'query_phrase': phrase, 'results': filtered_results}
                phrase_record = phrase_record_by_rendered.get(phrase)
                if phrase_record:
                    group['query_phrase_role'] = phrase_record.get('role')
                    group['query_phrase_source'] = phrase_record.get('source')
                    group['query_phrase_text'] = phrase_record.get('text')
                groups.append(group)
        lines = _render_visual_query_recall_lines(groups, max_lines=max(1, config.max_lines), question=question, question_ir=question_ir or {}, phrase_records=phrase_records, role_aware_compression_enabled=config.role_aware_compression_enabled or config.entity_pack_enabled)
        focus_frames = _visual_focus_frames_from_groups(groups, max_frames=max(1, config.max_lines))
        uses_synonym_lexicon = synonym_expansion_enabled(question_ir)
        question_route = question_ir.get('question_route') if isinstance(question_ir, dict) and isinstance(question_ir.get('question_route'), dict) else {}
        compression_metadata = {'contract': 'object_faiss_entity_pack_query_compression_v1' if config.entity_pack_enabled else 'object_faiss_hsgg_fact_role_aware_compression_v1' if config.role_aware_compression_enabled else 'object_faiss_hsgg_fact_generic_compression_v1', 'selected_object_limit': VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT, 'entity_pack_enabled': bool(config.entity_pack_enabled), 'role_aware_compression_enabled': bool(config.role_aware_compression_enabled), 'frame_context_enabled': bool(config.frame_context_enabled), 'frame_context_neighbor_limit': int(config.frame_context_neighbor_limit), 'frame_context_relation_limit': int(config.frame_context_relation_limit), 'tracklet_index_enabled': bool(config.entity_pack_enabled and config.entity_pack_tracklet_index_enabled and (graph is not None)), 'tracklet_top_k': int(config.entity_pack_tracklet_top_k), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon, 'uses_question_route': bool(question_route), 'ranking_signals': ['faiss_similarity', 'hsgg_fact_richness', 'question_ir_role_target_context_separation' if config.role_aware_compression_enabled or config.entity_pack_enabled else 'generic_question_candidate_token_overlap']}
        if uses_synonym_lexicon:
            compression_metadata['synonym_expansion'] = synonym_expansion_metadata(question_ir)
        if question_route:
            compression_metadata['question_route'] = question_route
        return {'contract': 'experimental_visual_query_faiss_recall_v1', 'available': bool(lines), 'seq_id': str(seq_id), 'question': str(question), 'query_phrases': phrases, 'top_k': max(1, config.query_recall_top_k), 'search_k': max(1, config.query_recall_search_k), 'max_lines': max(1, config.max_lines), 'min_score': float(config.query_recall_min_score), 'same_sequence_only': bool(config.same_sequence_only), 'query_phrase_records': phrase_records, 'compression': compression_metadata, 'groups': [{'query_phrase': group['query_phrase'], 'query_phrase_role': group.get('query_phrase_role'), 'query_phrase_source': group.get('query_phrase_source'), 'query_phrase_text': group.get('query_phrase_text'), 'results': [_concise_object(row) for row in group['results']]} for group in groups], 'visual_focus_frames': focus_frames, 'lines': lines}

    def build_question_evidence(self, *, seq_id: str, question: str, graph_answer: dict[str, Any], question_ir: dict[str, Any] | None, config: VisualFaissConfig) -> dict[str, Any]:
        focus_frames = _focus_frame_indices(graph_answer)
        terms = _question_terms(question=question, question_ir=question_ir or {})
        seed_objects = self._select_seed_objects(seq_id=str(seq_id), focus_frames=focus_frames, terms=terms, limit=max(1, config.seed_limit), per_frame_object_limit=max(1, config.per_frame_object_limit))
        groups: list[dict[str, Any]] = []
        seen_result_keys: set[tuple[str, int | None, int | None]] = set()
        for seed in seed_objects:
            results = self._search_similar_objects(seed=seed, seq_id=str(seq_id), top_k=max(1, config.top_k), same_sequence_only=config.same_sequence_only)
            filtered_results: list[dict[str, Any]] = []
            for row in results:
                self._attach_hsgg_object_facts(row, include_frame_context=config.frame_context_enabled, frame_context_neighbor_limit=config.frame_context_neighbor_limit, frame_context_relation_limit=config.frame_context_relation_limit)
                key = (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))
                if key in seen_result_keys:
                    continue
                seen_result_keys.add(key)
                filtered_results.append(row)
            if filtered_results:
                groups.append({'seed': seed, 'results': filtered_results})
        lines = _render_visual_evidence_lines(groups, max_lines=max(1, config.max_lines))
        return {'contract': 'experimental_visual_faiss_grounding_v1', 'available': bool(lines), 'seq_id': str(seq_id), 'focus_frames': focus_frames, 'question_terms': terms, 'seed_limit': max(1, config.seed_limit), 'top_k': max(1, config.top_k), 'max_lines': max(1, config.max_lines), 'same_sequence_only': bool(config.same_sequence_only), 'seed_objects': [_concise_object(row) for row in seed_objects], 'groups': [{'seed': _concise_object(group['seed']), 'results': [_concise_object(row) for row in group['results']]} for group in groups], 'lines': lines}

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _select_seed_objects(self, *, seq_id: str, focus_frames: list[int], terms: list[str], limit: int, per_frame_object_limit: int) -> list[dict[str, Any]]:
        if not focus_frames:
            return []
        rows: list[dict[str, Any]] = []
        for frame_rank, frame_index in enumerate(focus_frames, start=1):
            frame_rows = self.connection.execute('\n                SELECT raw_json\n                FROM object_metadata\n                WHERE seq_id = ? AND graph_frame_index = ?\n                ORDER BY COALESCE(score, 0.0) DESC, object_index ASC\n                LIMIT ?\n                ', (str(seq_id), int(frame_index), int(per_frame_object_limit))).fetchall()
            for row in frame_rows:
                item = json.loads(str(row['raw_json']))
                item['_focus_frame_rank'] = frame_rank
                item['_seed_score'] = _seed_score(item, terms=terms, frame_rank=frame_rank)
                rows.append(item)
        if not rows:
            return []
        rows.sort(key=lambda item: (-float(item.get('_seed_score') or 0.0), int(item.get('_focus_frame_rank') or 9999), int(item.get('object_index') or 999999)))
        selected: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in rows:
            faiss_id = _optional_int(item.get('faiss_id'))
            if faiss_id is None or faiss_id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(faiss_id)
            if len(selected) >= limit:
                break
        return selected

    def _search_similar_objects(self, *, seed: dict[str, Any], seq_id: str, top_k: int, same_sequence_only: bool) -> list[dict[str, Any]]:
        seed_id = _optional_int(seed.get('faiss_id'))
        if seed_id is None:
            return []
        vector = np.asarray(self.index.reconstruct(seed_id), dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(vector)
        search_k = top_k + 1
        if same_sequence_only:
            search_k = min(int(self.index.ntotal), max(top_k * 25, top_k + 25))
        distances, indices = self.index.search(vector, search_k)
        candidate_ids = [int(value) for value in indices[0] if int(value) >= 0 and int(value) != int(seed_id)]
        metadata_by_id = self._metadata_by_ids(candidate_ids)
        results: list[dict[str, Any]] = []
        for score, result_id in zip(distances[0], indices[0], strict=True):
            result_id = int(result_id)
            if result_id < 0 or result_id == int(seed_id):
                continue
            row = metadata_by_id.get(result_id)
            if row is None:
                continue
            if same_sequence_only and str(row.get('seq_id') or '') != str(seq_id):
                continue
            row['similarity'] = float(score)
            row['rank'] = len(results) + 1
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def _search_by_vector(self, *, vector: np.ndarray, seq_id: str, top_k: int, search_k: int, same_sequence_only: bool, min_score: float) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)
        resolved_search_k = min(int(self.index.ntotal), max(top_k, search_k))
        distances, indices = self.index.search(query, resolved_search_k)
        candidate_ids = [int(value) for value in indices[0] if int(value) >= 0]
        metadata_by_id = self._metadata_by_ids(candidate_ids)
        results: list[dict[str, Any]] = []
        for score, result_id in zip(distances[0], indices[0], strict=True):
            result_id = int(result_id)
            if result_id < 0:
                continue
            if float(score) < min_score:
                continue
            row = metadata_by_id.get(result_id)
            if row is None:
                continue
            if same_sequence_only and str(row.get('seq_id') or '') != str(seq_id):
                continue
            row['similarity'] = float(score)
            row['rank'] = len(results) + 1
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def _search_entity_tracklets_by_vector(self, *, vector: np.ndarray, seq_id: str, graph: CanonicalTemporalMemoryGraph, top_k: int, search_k: int, min_score: float) -> list[dict[str, Any]]:
        tracklet_index, metadata, summary = self._load_or_build_entity_tracklet_index(graph=graph, seq_id=seq_id)
        if int(tracklet_index.ntotal) <= 0 or not metadata:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)
        resolved_search_k = min(int(tracklet_index.ntotal), max(top_k, search_k))
        distances, indices = tracklet_index.search(query, resolved_search_k)
        results: list[dict[str, Any]] = []
        for score, result_id in zip(distances[0], indices[0], strict=True):
            result_id = int(result_id)
            if result_id < 0 or result_id >= len(metadata):
                continue
            if float(score) < min_score:
                continue
            row = dict(metadata[result_id])
            row['similarity'] = float(score)
            row['rank'] = len(results) + 1
            row['tracklet_hit'] = True
            row['entity_pack_recall_level'] = 'tracklet'
            row['tracklet_index_summary'] = summary
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def _load_or_build_entity_tracklet_index(self, *, graph: CanonicalTemporalMemoryGraph, seq_id: str) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        cache_key, index_path, metadata_path = self._entity_tracklet_cache_paths(graph=graph, seq_id=seq_id)
        cached = self._entity_tracklet_index_cache.get(cache_key)
        if cached is not None:
            return cached
        if index_path.exists() and metadata_path.exists():
            try:
                index = self._faiss.read_index(str(index_path))
                payload = json.loads(metadata_path.read_text(encoding='utf-8'))
                if isinstance(payload, dict) and payload.get('contract') == _ENTITY_TRACKLET_INDEX_CONTRACT and (payload.get('cache_key') == cache_key):
                    metadata = [item for item in payload.get('tracklets', []) or [] if isinstance(item, dict)]
                    if len(metadata) == int(index.ntotal):
                        summary = dict(payload.get('summary', {}) or {})
                        result = (index, metadata, summary)
                        self._entity_tracklet_index_cache[cache_key] = result
                        return result
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
                pass
        index, metadata, summary = self._build_entity_tracklet_index(graph=graph, seq_id=seq_id, cache_key=cache_key)
        self._write_entity_tracklet_index_cache(index=index, metadata=metadata, summary=summary, cache_key=cache_key, index_path=index_path, metadata_path=metadata_path)
        result = (index, metadata, summary)
        self._entity_tracklet_index_cache[cache_key] = result
        return result

    def _entity_tracklet_cache_paths(self, *, graph: CanonicalTemporalMemoryGraph, seq_id: str) -> tuple[str, Path, Path]:
        source_path = getattr(graph, 'source_graph_path', None)
        resolved_source = resolve_repo_input_path(Path(source_path)) if source_path is not None else None
        source_stat = resolved_source.stat() if resolved_source and resolved_source.exists() else None
        index_stat = self.index_path.stat()
        metadata_stat = self.metadata_db_path.stat()
        raw_key = json.dumps({'contract': _ENTITY_TRACKLET_INDEX_CONTRACT, 'cache_version': _ENTITY_TRACKLET_INDEX_CACHE_VERSION, 'seq_id': str(seq_id), 'source_graph_path': str(resolved_source or ''), 'source_graph_size': int(source_stat.st_size) if source_stat else None, 'source_graph_mtime_ns': int(source_stat.st_mtime_ns) if source_stat else None, 'object_index_path': str(self.index_path), 'object_index_size': int(index_stat.st_size), 'object_index_mtime_ns': int(index_stat.st_mtime_ns), 'object_metadata_path': str(self.metadata_db_path), 'object_metadata_size': int(metadata_stat.st_size), 'object_metadata_mtime_ns': int(metadata_stat.st_mtime_ns)}, sort_keys=True)
        cache_key = hashlib.sha1(raw_key.encode('utf-8')).hexdigest()[:20]
        safe_seq = re.sub('[^a-zA-Z0-9_.-]+', '_', str(seq_id)).strip('_') or 'seq'
        cache_dir = self.index_dir / 'entity_pack_tracklet_cache'
        base_name = f'{self.index_name}.seq{safe_seq}.{cache_key}.entity_tracklet_v{_ENTITY_TRACKLET_INDEX_CACHE_VERSION}'
        return (cache_key, cache_dir / f'{base_name}.faiss', cache_dir / f'{base_name}.metadata.json')

    def _build_entity_tracklet_index(self, *, graph: CanonicalTemporalMemoryGraph, seq_id: str, cache_key: str) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        anchor_index = _visual_faiss_observation_anchor_index(graph)
        entity_by_id = {str(entity.entity_id): entity for entity in graph.entity_nodes}
        dim = int(getattr(self.index, 'd', 0) or 0)
        buckets: dict[str, dict[str, Any]] = {}
        rows = self.connection.execute('\n            SELECT raw_json\n            FROM object_metadata\n            WHERE seq_id = ?\n              AND graph_frame_index IS NOT NULL\n              AND object_index IS NOT NULL\n            ORDER BY graph_frame_index ASC, object_index ASC\n            ', (str(seq_id),)).fetchall()
        indexed_observations = 0
        for db_row in rows:
            try:
                item = json.loads(str(db_row['raw_json']))
            except json.JSONDecodeError:
                continue
            frame = _optional_int(item.get('graph_frame_index'))
            object_index = _optional_int(item.get('object_index'))
            faiss_id = _optional_int(item.get('faiss_id'))
            if frame is None or object_index is None or faiss_id is None:
                continue
            anchor = anchor_index.get((frame, object_index))
            if not isinstance(anchor, dict):
                continue
            entity_id = str(anchor.get('entity_id') or '').strip()
            if not entity_id:
                continue
            try:
                vector = np.asarray(self.index.reconstruct(faiss_id), dtype=np.float32)
            except (RuntimeError, ValueError):
                continue
            if vector.size != dim:
                continue
            bucket = buckets.setdefault(entity_id, {'vector_sum': np.zeros(dim, dtype=np.float32), 'count': 0, 'labels': {}, 'start_frame': frame, 'end_frame': frame, 'start_sec': item.get('elapsed_sec_from_sequence_start'), 'end_sec': item.get('elapsed_sec_from_sequence_start'), 'representative_objects': []})
            bucket['vector_sum'] += vector
            bucket['count'] = int(bucket['count']) + 1
            bucket['start_frame'] = min(int(bucket['start_frame']), frame)
            bucket['end_frame'] = max(int(bucket['end_frame']), frame)
            elapsed = item.get('elapsed_sec_from_sequence_start')
            if isinstance(elapsed, (int, float)):
                start_sec = bucket.get('start_sec')
                end_sec = bucket.get('end_sec')
                bucket['start_sec'] = float(elapsed) if not isinstance(start_sec, (int, float)) else min(float(start_sec), float(elapsed))
                bucket['end_sec'] = float(elapsed) if not isinstance(end_sec, (int, float)) else max(float(end_sec), float(elapsed))
            label = str(anchor.get('label') or item.get('label') or '').strip()
            if label:
                labels = bucket['labels']
                labels[label] = int(labels.get(label, 0)) + 1
            representative = _entity_tracklet_representative_object(item, anchor)
            representatives = list(bucket.get('representative_objects', []) or [])
            representatives.append(representative)
            representatives.sort(key=lambda row: (-float(row.get('object_score') or 0.0), _optional_int(row.get('graph_frame_index')) or 0, _optional_int(row.get('object_index')) or 0))
            bucket['representative_objects'] = representatives[:4]
            indexed_observations += 1
        metadata: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        for entity_id in sorted(buckets):
            bucket = buckets[entity_id]
            count = int(bucket.get('count') or 0)
            if count <= 0:
                continue
            vector = np.asarray(bucket['vector_sum'], dtype=np.float32) / float(count)
            vectors.append(vector)
            entity = entity_by_id.get(entity_id)
            label_counts = dict(bucket.get('labels', {}) or {})
            top_label = ''
            if label_counts:
                top_label = sorted(label_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
            entity_label = str(getattr(entity, 'label', '') or top_label).strip()
            representatives = list(bucket.get('representative_objects', []) or [])
            representative = representatives[0] if representatives else {}
            metadata.append({'tracklet_faiss_id': len(metadata), 'faiss_id': _optional_int(representative.get('faiss_id')), 'object_faiss_id': _optional_int(representative.get('faiss_id')), 'seq_id': str(seq_id), 'entity_id': entity_id, 'anchor_entity_id': entity_id, 'label': entity_label, 'graph_frame_index': _optional_int(representative.get('graph_frame_index')), 'frame_key': representative.get('frame_key'), 'image_name': representative.get('image_name'), 'object_index': _optional_int(representative.get('object_index')), 'bbox': representative.get('bbox'), 'image_path': representative.get('image_path'), 'cache_image_path': representative.get('cache_image_path'), 'bbox_key': representative.get('bbox_key'), 'score': representative.get('object_score'), 'elapsed_sec_from_sequence_start': representative.get('elapsed_sec_from_sequence_start'), 'tracklet_hit': True, 'entity_pack_recall_level': 'tracklet', 'tracklet_observation_count': count, 'tracklet_time_span': {'start_frame': bucket.get('start_frame'), 'end_frame': bucket.get('end_frame'), 'start_sec': bucket.get('start_sec'), 'end_sec': bucket.get('end_sec')}, 'representative_labels': label_counts, 'representative_objects': representatives, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
        index = self._faiss.IndexFlatIP(dim)
        if vectors:
            matrix = np.vstack(vectors).astype('float32', copy=False)
            self._faiss.normalize_L2(matrix)
            index.add(matrix)
        summary = {'contract': _ENTITY_TRACKLET_INDEX_CONTRACT, 'cache_key': cache_key, 'seq_id': str(seq_id), 'entity_count': len(metadata), 'indexed_observation_count': indexed_observations, 'object_metadata_rows_scanned': len(rows), 'dimension': dim, 'aggregation': 'mean_l2_normalized_object_embeddings_per_cg6_entity', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}
        return (index, metadata, summary)

    def _write_entity_tracklet_index_cache(self, *, index: Any, metadata: list[dict[str, Any]], summary: dict[str, Any], cache_key: str, index_path: Path, metadata_path: Path) -> None:
        payload = {'contract': _ENTITY_TRACKLET_INDEX_CONTRACT, 'cache_version': _ENTITY_TRACKLET_INDEX_CACHE_VERSION, 'cache_key': cache_key, 'summary': summary, 'tracklets': metadata}
        index_tmp = index_path.with_name(f'{index_path.name}.{os.getpid()}.tmp')
        metadata_tmp = metadata_path.with_name(f'{metadata_path.name}.{os.getpid()}.tmp')
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(index, str(index_tmp))
            metadata_tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            os.replace(index_tmp, index_path)
            os.replace(metadata_tmp, metadata_path)
        except OSError:
            for path in (index_tmp, metadata_tmp):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        if self._text_encoder is None:
            from vlkg.core.indexing.vision import VisionFeatureExtractor
            self._text_encoder = VisionFeatureExtractor(local_files_only=True)
        return self._text_encoder.encode_text_batch(texts)

    def _metadata_by_ids(self, faiss_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not faiss_ids:
            return {}
        placeholders = ','.join(('?' for _ in faiss_ids))
        rows = self.connection.execute(f'\n            SELECT faiss_id, raw_json\n            FROM object_metadata\n            WHERE faiss_id IN ({placeholders})\n            ', faiss_ids).fetchall()
        return {int(row['faiss_id']): json.loads(str(row['raw_json'])) for row in rows}

    def _attach_hsgg_object_facts(self, row: dict[str, Any], *, include_frame_context: bool=False, frame_context_neighbor_limit: int=4, frame_context_relation_limit: int=4) -> None:
        facts = self._hsgg_object_facts(row, include_frame_context=include_frame_context, frame_context_neighbor_limit=frame_context_neighbor_limit, frame_context_relation_limit=frame_context_relation_limit)
        if not facts:
            return
        row.update(facts)

    def _hsgg_object_facts(self, row: dict[str, Any], *, include_frame_context: bool=False, frame_context_neighbor_limit: int=4, frame_context_relation_limit: int=4) -> dict[str, Any]:
        frame_key = str(row.get('frame_key') or '').strip()
        object_index = _optional_int(row.get('object_index'))
        if not frame_key or object_index is None:
            return {}
        bundle = self._hsgg_bundle_for_row(row)
        if not bundle:
            return {}
        facts: dict[str, Any] = {}
        attributes = bundle.get('attributes', {})
        frame_attributes = attributes.get(frame_key, {}) if isinstance(attributes, dict) else {}
        frame_attribute_lookup = _frame_attribute_lookup(frame_attributes)
        attribute = frame_attribute_lookup.get(object_index, '')
        if isinstance(frame_attributes, dict):
            if attribute:
                facts['visual_attribute'] = attribute
        relations = bundle.get('relations', {})
        frame_relations = relations.get(frame_key, []) if isinstance(relations, dict) else []
        relation_lines: list[str] = []
        relation_records: list[dict[str, Any]] = []
        neighbor_attributes: list[str] = []
        seen_neighbor_attributes: set[str] = set()
        if isinstance(frame_relations, list):
            for relation in frame_relations:
                if not isinstance(relation, dict):
                    continue
                indices = relation.get('idx') or []
                relation_indices = [_optional_int(value) for value in indices]
                if object_index not in relation_indices:
                    continue
                subject_index = relation_indices[0] if len(relation_indices) >= 1 else None
                related_object_index = relation_indices[1] if len(relation_indices) >= 2 else None
                subject = str(relation.get('subject_label') or '').strip()
                predicate = str(relation.get('predicate') or '').strip()
                obj = str(relation.get('object_label') or '').strip()
                if not (subject and predicate and obj):
                    raw_output = str(relation.get('raw_output') or '').strip()
                    if raw_output:
                        relation_lines.append(raw_output)
                        relation_records.append({'subject_index': subject_index, 'object_index': related_object_index, 'text': _short_text(raw_output, limit=120)})
                    continue
                relation_text = f'{subject} {predicate} {obj}'
                relation_lines.append(relation_text)
                relation_records.append({'subject_index': subject_index, 'object_index': related_object_index, 'subject_label': subject, 'predicate': predicate, 'object_label': obj, 'text': _short_text(relation_text, limit=120)})
                for endpoint_index, endpoint_label in ((subject_index, subject), (related_object_index, obj)):
                    if endpoint_index is None or endpoint_index == object_index:
                        continue
                    endpoint_attribute = frame_attribute_lookup.get(endpoint_index, '')
                    if not endpoint_attribute:
                        continue
                    neighbor_text = _neighbor_attribute_text(index=endpoint_index, label=endpoint_label, attribute=endpoint_attribute)
                    if neighbor_text in seen_neighbor_attributes:
                        continue
                    neighbor_attributes.append(neighbor_text)
                    seen_neighbor_attributes.add(neighbor_text)
                    if len(neighbor_attributes) >= 4:
                        break
                if len(relation_lines) >= 3:
                    break
        if relation_lines:
            facts['visual_relations'] = relation_lines
        if relation_records:
            facts['visual_relation_records'] = relation_records
        if neighbor_attributes:
            facts['hsgg_neighbor_attributes'] = neighbor_attributes
        hsgg_visual_facts = _hsgg_visual_fact_lines(label=str(row.get('label') or 'object').strip() or 'object', object_index=object_index, attribute=attribute, relations=relation_lines, neighbor_attributes=neighbor_attributes)
        if hsgg_visual_facts:
            facts['hsgg_visual_facts'] = hsgg_visual_facts
        if include_frame_context:
            frame_context = self._hsgg_frame_context_lines(row=row, bundle=bundle, frame_key=frame_key, object_index=object_index, frame_attribute_lookup=frame_attribute_lookup, relation_records=relation_records, neighbor_limit=max(1, int(frame_context_neighbor_limit)), relation_limit=max(1, int(frame_context_relation_limit)))
            if frame_context:
                facts['hsgg_frame_context'] = frame_context
        return facts

    def _hsgg_frame_context_lines(self, *, row: dict[str, Any], bundle: dict[str, Any], frame_key: str, object_index: int, frame_attribute_lookup: dict[int, str], relation_records: list[dict[str, Any]], neighbor_limit: int, relation_limit: int) -> list[str]:
        objects_payload = self._hsgg_objects_payload_for_row(row=row, bundle=bundle)
        frame_objects = objects_payload.get(frame_key, {}) if isinstance(objects_payload, dict) else {}
        labels = frame_objects.get('labels', []) if isinstance(frame_objects, dict) else []
        boxes = frame_objects.get('boxes', []) if isinstance(frame_objects, dict) else []
        anchor_label = str(row.get('label') or _label_at(labels, object_index) or 'object')
        anchor_bbox = _bbox_values(row.get('bbox'))
        if anchor_bbox is None and object_index < len(boxes):
            anchor_bbox = _bbox_values(boxes[object_index])
        lines: list[str] = []
        anchor_spatial = _bbox_spatial_text(anchor_bbox)
        if anchor_spatial:
            lines.append(f'same_frame_anchor obj={object_index} label={anchor_label!r} {anchor_spatial}')
        relation_lines: list[str] = []
        related_indices: list[int] = []
        for record in relation_records:
            if not isinstance(record, dict):
                continue
            text = _short_text(record.get('text'), limit=120)
            if text:
                relation_lines.append(f'same_frame_relation={text}')
            for key in ('subject_index', 'object_index'):
                index = _optional_int(record.get(key))
                if index is None or index == object_index or index in related_indices:
                    continue
                related_indices.append(index)
        lines.extend(_ordered_unique_texts(relation_lines)[:relation_limit])
        neighbor_indices = _ordered_unique_ints([*related_indices, *_nearest_frame_object_indices(anchor_bbox=anchor_bbox, boxes=boxes, anchor_index=object_index)])
        neighbor_lines: list[str] = []
        for index in neighbor_indices:
            if len(neighbor_lines) >= neighbor_limit:
                break
            label = _label_at(labels, index)
            attribute = _short_text(frame_attribute_lookup.get(index), limit=100)
            if not (label or attribute):
                continue
            neighbor_bbox = _bbox_values(boxes[index]) if index < len(boxes) else None
            spatial = _bbox_relative_text(anchor_bbox=anchor_bbox, candidate_bbox=neighbor_bbox)
            pieces = [f'same_frame_object obj={index}', f'label={label!r}' if label else '', f'attr={attribute}' if attribute else '', spatial]
            neighbor_lines.append(' '.join((piece for piece in pieces if piece)))
        lines.extend(neighbor_lines)
        return _ordered_unique_texts(lines)[:1 + relation_limit + neighbor_limit]

    def _hsgg_objects_payload_for_row(self, *, row: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
        if bundle.get('_objects_loaded'):
            objects = bundle.get('objects', {})
            return objects if isinstance(objects, dict) else {}
        hsgg_root = _hsgg_root_for_metadata_row(row)
        bundle['_objects_loaded'] = True
        if hsgg_root is None:
            bundle['objects'] = {}
            return {}
        objects_path = hsgg_root / 'objects.json'
        if not objects_path.exists():
            bundle['objects'] = {}
            return {}
        try:
            objects = json.loads(objects_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            objects = {}
        bundle['objects'] = objects if isinstance(objects, dict) else {}
        return bundle['objects']

    def _hsgg_bundle_for_row(self, row: dict[str, Any]) -> dict[str, Any]:
        hsgg_root = _hsgg_root_for_metadata_row(row)
        if hsgg_root is None:
            return {}
        cache_key = str(hsgg_root)
        cached = self._hsgg_cache.get(cache_key)
        if cached is not None:
            return cached
        attributes_path = hsgg_root / 'attributes.json'
        relations_path = hsgg_root / 'relations.json'
        bundle: dict[str, Any] = {}
        if attributes_path.exists():
            try:
                bundle['attributes'] = json.loads(attributes_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                bundle['attributes'] = {}
        if relations_path.exists():
            try:
                bundle['relations'] = json.loads(relations_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                bundle['relations'] = {}
        self._hsgg_cache[cache_key] = bundle
        return bundle

def attach_visual_faiss_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any]) -> dict[str, Any]:
    lines = [str(line).strip() for line in visual_payload.get('lines', []) if str(line).strip()]
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_payloads = list(updated.get('visual_faiss_payloads', []) or [])
    existing_payloads.append(visual_payload)
    existing_lines = [str(line).strip() for line in updated.get('visual_faiss_evidence_lines', []) if str(line).strip()]
    merged_lines = _ordered_unique_texts([*existing_lines, *lines])
    updated['visual_faiss_grounding'] = visual_payload
    updated['visual_faiss_payloads'] = existing_payloads
    updated['visual_faiss_evidence_lines'] = merged_lines
    context_lines = list(updated.get('llm_context_text', '').splitlines())
    if lines:
        context_lines.extend(['Visual FAISS evidence:'])
        context_lines.extend((f'- {line}' for line in lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = updated.get('structured_graph_context', {})
    if isinstance(structured, dict):
        structured = dict(structured)
        structured['visual_faiss_grounding'] = visual_payload
        structured['visual_faiss_payloads'] = existing_payloads
        structured['visual_faiss_evidence_lines'] = merged_lines
        updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    _append_visual_focus_frames(graph_answer, visual_payload)
    return graph_answer

def fuse_visual_faiss_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any]) -> dict[str, Any]:
    """Promote visual FAISS recall into the main evidence candidate pool.

    This is an experiment-level retrieval fusion step. It does not change the
    graph retrieval scores or use answer labels; it only gives downstream
    evidence selection a unified pool containing both graph and visual-object
    evidence.
    """
    lines = [str(line).strip() for line in _visual_faiss_fused_evidence_lines(visual_payload) if str(line).strip()]
    if not lines:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    updated['evidence_lines'] = _ordered_unique_texts([*existing_evidence, *lines])
    fusion_facts = _visual_faiss_candidate_facts(visual_payload, evidence_offset=len(existing_evidence))
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*candidate_facts, *fusion_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    fusion_lines = [_visual_faiss_aggregation_line(index, fact) for index, fact in enumerate(fusion_facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*existing_aggregation_lines, *fusion_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS fused evidence:'])
    context_lines.extend((f'- {line}' for line in lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_fusion'] = {'contract': 'experimental_visual_faiss_fused_evidence_v1', 'fused_evidence_count': len(lines), 'fused_candidate_fact_count': len(fusion_facts)}
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for index, line in enumerate(lines, start=1):
            digest.append({'rank': next_rank, 'source_record_id': f'visual_faiss:{index}', 'source_kind': 'visual_faiss_object', 'summary': line, 'score': _visual_faiss_line_score(index, visual_payload)})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def expand_visual_faiss_graph_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, frame_radius: int=8, max_items: int=12) -> dict[str, Any]:
    """Use visual FAISS frame anchors to promote nearby graph evidence.

    The promoted rows are still canonical-graph evidence: scenes, entities,
    state/relation episodes, transition events, and relation samples whose time
    ranges touch a visual anchor frame. FAISS supplies the anchor only.
    """
    anchors = _visual_anchor_rows(visual_payload, max_anchors=max(1, max_items))
    if not anchors:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    graph_items = _graph_evidence_for_visual_anchors(graph=graph, anchors=anchors, frame_radius=max(0, frame_radius), max_items=max(1, max_items))
    soft_contexts = _visual_anchor_soft_contexts(anchors=anchors, graph_items=graph_items, max_contexts=max(1, min(len(anchors), VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT)))
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    if not graph_items:
        _store_visual_anchor_expansion_metadata(graph_answer, anchors=anchors, graph_items=[], soft_contexts=[], frame_radius=frame_radius)
        return graph_answer
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    soft_context_lines = [_visual_anchor_soft_context_line(index=index, context=context) for index, context in enumerate(soft_contexts, start=1)]
    promoted_lines = [_visual_anchor_graph_evidence_line(index=index, item=item) for index, item in enumerate(graph_items, start=1)]
    evidence_offset = len(existing_evidence)
    promoted_offset = evidence_offset + len(soft_context_lines)
    updated['evidence_lines'] = _ordered_unique_texts([*existing_evidence, *soft_context_lines, *promoted_lines])
    soft_context_facts = _visual_anchor_soft_context_candidate_facts(soft_contexts, evidence_offset=evidence_offset)
    expansion_facts = _visual_anchor_candidate_facts(graph_items, evidence_offset=promoted_offset)
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*candidate_facts, *soft_context_facts, *expansion_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    expansion_lines = [_visual_anchor_aggregation_line(index, fact) for index, fact in enumerate([*soft_context_facts, *expansion_facts], start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*existing_aggregation_lines, *expansion_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS soft graph context:'])
    context_lines.extend((f'- {line}' for line in soft_context_lines))
    context_lines.extend(['Visual FAISS anchored graph evidence:'])
    context_lines.extend((f'- {line}' for line in promoted_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_anchor_expansion'] = {'contract': 'experimental_visual_faiss_graph_anchor_expansion_v1', 'frame_radius': int(frame_radius), 'anchor_count': len(anchors), 'soft_context_count': len(soft_contexts), 'promoted_graph_evidence_count': len(graph_items), 'anchors': anchors, 'soft_contexts': soft_contexts, 'promoted_graph_evidence': graph_items}
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for context in soft_contexts:
            digest.append({'rank': next_rank, 'source_record_id': context.get('source_record_id'), 'source_kind': 'visual_faiss_soft_graph_context', 'summary': context.get('summary'), 'score': context.get('score'), 'visual_anchor': context.get('visual_anchor')})
            next_rank += 1
        for item in graph_items:
            digest.append({'rank': next_rank, 'source_record_id': item.get('source_record_id'), 'source_kind': item.get('source_kind'), 'summary': item.get('summary'), 'score': item.get('score'), 'visual_anchor': item.get('visual_anchor')})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    _store_visual_anchor_expansion_metadata(graph_answer, anchors=anchors, graph_items=graph_items, soft_contexts=soft_contexts, frame_radius=frame_radius)
    return graph_answer

def fuse_visual_faiss_ger_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, object_limit: int=8, context_radius: int=8, context_limit: int=12, binding_enabled: bool=False, structural_binding_enabled: bool=False, attribute_owner_enabled: bool=False, fragment_fragment_bridge_enabled: bool=False, object_local_facts_enabled: bool=False, local_subgraph_enabled: bool=False, local_subgraph_anchor_only: bool=False, balanced_graph_visual_evidence: bool=False, role_aware_compression_enabled: bool=False, question: str='', question_ir: dict[str, Any] | None=None) -> dict[str, Any]:
    """GER-like object fusion for the experimental object-FAISS branch.

    Visual FAISS retrieves object candidates; each retrieved object then carries
    its HSGG visual facts plus nearby canonical graph context into the same
    evidence/candidate-fact pool used by graph retrieval. This keeps FAISS as a
    retrieval mechanism, not a dataset-specific answer rule.
    """
    selected_objects = _visual_faiss_ger_object_rows(visual_payload, object_limit=max(1, object_limit), role_aware_compression_enabled=role_aware_compression_enabled, question=question, question_ir=question_ir or {})
    if not selected_objects:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    observation_anchor_index = _visual_faiss_observation_anchor_index(graph) if local_subgraph_enabled else None
    contexts = _visual_faiss_ger_contexts(graph=graph, object_rows=selected_objects, context_radius=max(0, context_radius), context_limit=max(1, context_limit), observation_anchor_index=observation_anchor_index)
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    bound_facts = _visual_faiss_ger_bound_facts(contexts, question=question, question_ir=question_ir or {}, structural_binding_enabled=structural_binding_enabled, attribute_owner_enabled=attribute_owner_enabled, exclude_schema_terms=balanced_graph_visual_evidence) if binding_enabled else []
    bound_lines = [_visual_faiss_ger_bound_fact_line(index=index, fact=fact) for index, fact in enumerate(bound_facts, start=1)]
    bridge_facts = _visual_faiss_fragment_fragment_bridge_facts(contexts, question=question, question_ir=question_ir or {}, structural_binding_enabled=structural_binding_enabled, limit=min(VISUAL_FAISS_fragment_FRAGMENT_BRIDGE_LIMIT, max(1, int(object_limit))), exclude_schema_terms=balanced_graph_visual_evidence) if binding_enabled and fragment_fragment_bridge_enabled else []
    bridge_lines = [_visual_faiss_fragment_fragment_bridge_line(index=index, fact=fact) for index, fact in enumerate(bridge_facts, start=1)]
    object_local_facts = _visual_faiss_object_local_facts(contexts, question=question, question_ir=question_ir or {}, limit=max(1, int(object_limit)), exclude_schema_terms=balanced_graph_visual_evidence) if binding_enabled and object_local_facts_enabled else []
    for rank, fact in enumerate(object_local_facts, start=1):
        fact['support_ranks'] = [rank]
        fact['object_local_rank'] = rank
    object_local_lines = [_visual_faiss_object_local_fact_line(index=index, fact=fact) for index, fact in enumerate(object_local_facts, start=1)]
    local_subgraph_facts = _visual_faiss_local_subgraph_facts(contexts, graph=graph, question=question, question_ir=question_ir or {}, limit=max(1, int(object_limit)), anchor_only=local_subgraph_anchor_only) if binding_enabled and local_subgraph_enabled else []
    for rank, fact in enumerate(local_subgraph_facts, start=1):
        fact['support_ranks'] = [len(object_local_lines) + rank]
        fact['local_subgraph_rank'] = rank
    local_subgraph_lines = [_visual_faiss_local_subgraph_fact_line(index=index, fact=fact) for index, fact in enumerate(local_subgraph_facts, start=1)]
    bound_line_offset = len(object_local_lines) + len(local_subgraph_lines)
    for rank, fact in enumerate(bound_facts, start=1):
        fact['support_ranks'] = [bound_line_offset + rank]
        fact['bound_rank'] = rank
    bridge_line_offset = len(object_local_lines) + len(local_subgraph_lines) + len(bound_lines)
    for rank, fact in enumerate(bridge_facts, start=1):
        fact['support_ranks'] = [bridge_line_offset + rank]
        fact['bridge_rank'] = rank
    ger_lines = [_visual_faiss_ger_context_line(index=index, context=context) for index, context in enumerate(contexts, start=1)]
    ger_line_offset = len(object_local_lines) + len(local_subgraph_lines) + len(bound_lines) + len(bridge_lines)
    updated['visual_faiss_ger_evidence_lines'] = ger_lines
    if bound_lines:
        updated['visual_faiss_ger_bound_fact_lines'] = bound_lines
    if bridge_lines:
        updated['visual_faiss_fragment_fragment_bridge_lines'] = bridge_lines
    if object_local_lines:
        updated['visual_faiss_object_local_fact_lines'] = object_local_lines
    if local_subgraph_lines:
        updated['visual_faiss_local_subgraph_fact_lines'] = local_subgraph_lines
    visual_line_channels = [local_subgraph_lines, object_local_lines, bound_lines, bridge_lines, ger_lines]
    if balanced_graph_visual_evidence:
        updated['evidence_lines'] = _balanced_graph_visual_lines(graph_lines=existing_evidence, visual_channels=visual_line_channels)
    else:
        updated['evidence_lines'] = _ordered_unique_texts([*local_subgraph_lines, *object_local_lines, *bound_lines, *bridge_lines, *ger_lines, *existing_evidence])
    if balanced_graph_visual_evidence:
        _sync_visual_fact_support_ranks(evidence_lines=updated['evidence_lines'], fact_line_pairs=[*zip(local_subgraph_facts, local_subgraph_lines, strict=False), *zip(object_local_facts, object_local_lines, strict=False), *zip(bound_facts, bound_lines, strict=False), *zip(bridge_facts, bridge_lines, strict=False)])
    ger_facts = _visual_faiss_ger_candidate_facts(contexts, support_rank_offset=0 if balanced_graph_visual_evidence else ger_line_offset)
    if balanced_graph_visual_evidence:
        _sync_visual_fact_support_ranks(evidence_lines=updated['evidence_lines'], fact_line_pairs=zip(ger_facts, ger_lines, strict=False))
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    visual_fact_channels = [local_subgraph_facts, object_local_facts, bound_facts, bridge_facts, ger_facts]
    if balanced_graph_visual_evidence:
        evidence_aggregation['candidate_facts'] = _balanced_graph_visual_items(graph_items=candidate_facts, visual_channels=visual_fact_channels)
    else:
        evidence_aggregation['candidate_facts'] = [*local_subgraph_facts, *object_local_facts, *bridge_facts, *bound_facts, *ger_facts, *candidate_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    bridge_aggregation_lines = [_visual_faiss_fragment_fragment_bridge_aggregation_line(index, fact) for index, fact in enumerate(bridge_facts, start=1)]
    bound_aggregation_lines = [_visual_faiss_ger_bound_aggregation_line(index, fact) for index, fact in enumerate(bound_facts, start=1)]
    object_local_aggregation_lines = [_visual_faiss_object_local_aggregation_line(index, fact) for index, fact in enumerate(object_local_facts, start=1)]
    local_subgraph_aggregation_lines = [_visual_faiss_local_subgraph_aggregation_line(index, fact) for index, fact in enumerate(local_subgraph_facts, start=1)]
    ger_aggregation_lines = [_visual_faiss_ger_aggregation_line(index, fact) for index, fact in enumerate(ger_facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    visual_aggregation_channels = [local_subgraph_aggregation_lines, object_local_aggregation_lines, bound_aggregation_lines, bridge_aggregation_lines, ger_aggregation_lines]
    if balanced_graph_visual_evidence:
        updated['evidence_aggregation_lines'] = _balanced_graph_visual_lines(graph_lines=existing_aggregation_lines, visual_channels=visual_aggregation_channels)
    else:
        updated['evidence_aggregation_lines'] = _ordered_unique_texts([*local_subgraph_aggregation_lines, *object_local_aggregation_lines, *bridge_aggregation_lines, *bound_aggregation_lines, *ger_aggregation_lines, *existing_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    if local_subgraph_lines:
        context_lines.extend(['Visual FAISS local-subgraph facts:'])
        context_lines.extend((f'- {line}' for line in local_subgraph_lines))
    if object_local_lines:
        context_lines.extend(['Visual FAISS object-local facts:'])
        context_lines.extend((f'- {line}' for line in object_local_lines))
    if bound_lines:
        context_lines.extend(['Visual FAISS GER-like bound facts:'])
        context_lines.extend((f'- {line}' for line in bound_lines))
    if bridge_lines:
        context_lines.extend(['Visual FAISS fragment fragment-bridge facts:'])
        context_lines.extend((f'- {line}' for line in bridge_lines))
    context_lines.extend(['Visual FAISS GER-like unified evidence:'])
    context_lines.extend((f'- {line}' for line in ger_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    uses_synonym_lexicon = synonym_expansion_enabled(question_ir)
    metadata = {'contract': 'experimental_visual_faiss_ger_fusion_v1', 'object_count': len(selected_objects), 'context_count': len(contexts), 'object_limit': int(max(1, object_limit)), 'context_radius': int(max(0, context_radius)), 'context_limit': int(max(1, context_limit)), 'role_aware_compression_enabled': bool(role_aware_compression_enabled), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon, 'objects': [_concise_visual_faiss_ger_object(row) for _, row in selected_objects], 'contexts': contexts}
    if uses_synonym_lexicon:
        metadata['synonym_expansion'] = synonym_expansion_metadata(question_ir)
    if role_aware_compression_enabled:
        metadata['role_aware_compression'] = {'contract': 'experimental_visual_faiss_ger_role_aware_compression_v1', 'scope': 'FAISS object-row compression before GER evidence packaging', 'policy': ['question-IR target rows are selected before constraint/context rows', 'constraint/context rows cannot exhaust selected object slots', 'support and fallback rows fill remaining slots only'], 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon}
    if any((context.get('hsgg_frame_context') for context in contexts)):
        metadata['frame_context'] = {'contract': 'experimental_visual_faiss_object_anchored_frame_context_v1', 'context_count': sum((1 for context in contexts if context.get('hsgg_frame_context'))), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon, 'signals': ['same-frame HSGG object attributes', 'same-frame HSGG object relations', 'generic bbox relative geometry']}
    if binding_enabled:
        metadata['binding'] = {'contract': 'experimental_visual_faiss_ger_structural_binding_v1' if structural_binding_enabled else 'experimental_visual_faiss_ger_attribute_owner_binding_v1' if attribute_owner_enabled else 'experimental_visual_faiss_ger_binding_v1', 'bound_fact_count': len(bound_facts), 'bound_line_count': len(bound_lines), 'structural_binding_enabled': bool(structural_binding_enabled), 'attribute_owner_enabled': bool(attribute_owner_enabled), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon, 'ranking_signals': ['separate target/entity, attribute, event, spatial, and graph need coverage' if structural_binding_enabled else 'generic question token coverage', 'query/entity tokens do not become hard constraints by default' if structural_binding_enabled else 'question-IR constraint token coverage', 'target label/object fact coverage', 'HSGG attributes are marked as direct_object_attribute or neighbor_object_attribute with owner object indices' if attribute_owner_enabled else 'HSGG object attributes and relation context', 'same-frame nearby graph evidence', 'FAISS similarity as a tie-breaker']}
    if fragment_fragment_bridge_enabled:
        metadata['fragment_fragment_bridge'] = {'contract': 'experimental_visual_faiss_fragment_fragment_bridge_v1', 'bridge_fact_count': len(bridge_facts), 'bridge_line_count': len(bridge_lines), 'bridge_limit': min(VISUAL_FAISS_fragment_FRAGMENT_BRIDGE_LIMIT, max(1, int(object_limit))), 'graph_mutation': False, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'ranking_signals': ['object-level FAISS visual similarity', 'HSGG object attributes and relations attached to the retrieved object', 'same-frame HSGG context attached to the retrieved object', 'nearby canonical graph fragments around the retrieved object frame', 'generic structural slot coverage from question IR as a soft signal']}
    if object_local_facts_enabled:
        metadata['object_local_facts'] = {'contract': 'experimental_visual_faiss_object_local_facts_v1', 'fact_count': len(object_local_facts), 'line_count': len(object_local_lines), 'graph_mutation': False, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'scope': 'FAISS-recalled object metadata plus same-frame HSGG direct attributes, direct relations, object bbox, and direct neighbor attributes', 'ranking_signals': ['object-local attribute coverage', 'object-local relation coverage', 'same-frame anchor/bbox context', 'generic question and question-IR token coverage', 'FAISS similarity as a tie-breaker']}
    if local_subgraph_enabled:
        metadata['local_subgraph_facts'] = {'contract': 'experimental_visual_faiss_local_subgraph_facts_v1', 'fact_count': len(local_subgraph_facts), 'line_count': len(local_subgraph_lines), 'graph_mutation': False, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'scope': 'FAISS-recalled object candidates are mapped back to canonical entity ids, then each candidate entity is represented by its own attributes, direct relation/state evidence, time range, visual anchor, and bounded same-frame/near-time neighbors.', 'ranking_unit': 'candidate local subgraph, not isolated evidence line', 'candidate_entity_policy': 'observation_anchor_entity_only' if local_subgraph_anchor_only else 'observation_anchor_plus_nearby_graph_entities', 'anchor_only': bool(local_subgraph_anchor_only), 'ranking_signals': ['object-level FAISS visual similarity', 'exact temporal-graph observation anchor when frame/object_index is available', 'canonical entity id recovered from nearby graph support as fallback', 'direct entity attributes and states', 'direct relations involving the candidate entity', 'bounded same-frame or near-time neighbor context', 'generic question and question-IR token coverage as a soft signal']}
    if balanced_graph_visual_evidence:
        metadata['balanced_graph_visual_evidence'] = {'contract': 'experimental_visual_faiss_balanced_graph_visual_evidence_v1', 'graph_mutation': False, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'scope': 'preserve original graph evidence in the primary evidence pool while adding object-local, bound, bridge, and GER-like visual FAISS evidence as supplemental channels', 'primary_evidence_policy': {'head_size': 6, 'graph_quota': 3, 'visual_quota': 3}, 'schema_match_tokens_removed': sorted(_VISUAL_FAISS_SCHEMA_MATCH_TOKENS), 'ranking_signals': ['original graph retrieval order', 'object-local visual fact rank', 'GER-like bound visual fact rank', 'GER-like object context rank', 'generic schema-token filtering for match scores']}
    updated['visual_faiss_ger_fusion'] = metadata
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_ger_evidence_lines'] = ger_lines
    if local_subgraph_lines:
        structured['visual_faiss_local_subgraph_fact_lines'] = local_subgraph_lines
        structured['visual_faiss_local_subgraph_facts'] = local_subgraph_facts
    if object_local_lines:
        structured['visual_faiss_object_local_fact_lines'] = object_local_lines
        structured['visual_faiss_object_local_facts'] = object_local_facts
    if bound_lines:
        structured['visual_faiss_ger_bound_fact_lines'] = bound_lines
        structured['visual_faiss_ger_bound_facts'] = bound_facts
    if bridge_lines:
        structured['visual_faiss_fragment_fragment_bridge_lines'] = bridge_lines
        structured['visual_faiss_fragment_fragment_bridge_facts'] = bridge_facts
    structured['visual_faiss_ger_fusion'] = metadata
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for fact in local_subgraph_facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_local_subgraph_fact', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'entity_id': fact.get('entity_id'), 'local_subgraph_checks': fact.get('local_subgraph_checks', {}), 'support_context_ranks': fact.get('support_context_ranks', [])})
            next_rank += 1
        for fact in object_local_facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_object_local_fact', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'object_local_checks': fact.get('object_local_checks', {}), 'support_context_ranks': fact.get('support_context_ranks', [])})
            next_rank += 1
        for fact in bridge_facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_fragment_fragment_bridge_fact', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'bridge_checks': fact.get('bridge_checks', {}), 'support_context_ranks': fact.get('support_context_ranks', [])})
            next_rank += 1
        for fact in bound_facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_ger_bound_fact', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'matched_question_terms': fact.get('matched_question_terms', []), 'matched_constraint_terms': fact.get('matched_constraint_terms', []), 'support_context_ranks': fact.get('support_context_ranks', [])})
            next_rank += 1
        for context in contexts:
            digest.append({'rank': next_rank, 'source_record_id': context.get('source_record_id'), 'source_kind': 'visual_faiss_ger_object_context', 'summary': context.get('summary'), 'score': context.get('score'), 'visual_object': context.get('visual_object'), 'nearby_graph_evidence': context.get('nearby_graph_evidence', [])})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def fuse_visual_faiss_unified_candidates(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, object_limit: int=8, context_radius: int=8, context_limit: int=12, question: str='', question_ir: dict[str, Any] | None=None) -> dict[str, Any]:
    """Build unified graph+FAISS object candidates before final QA.

    This default-off experiment keeps the canonical graph immutable. It uses
    graph retrieval and object FAISS recall as two candidate sources, then ranks
    each visual object by frame-level HSGG object facts plus nearby canonical
    graph support. It never reads gold answers or dataset-specific lexicons.
    """
    selected_objects = _visual_faiss_ger_object_rows(visual_payload, object_limit=max(1, object_limit))
    if not selected_objects:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    contexts = _visual_faiss_ger_contexts(graph=graph, object_rows=selected_objects, context_radius=max(0, context_radius), context_limit=max(1, context_limit))
    candidates = _visual_faiss_unified_candidates(contexts, question=question, question_ir=question_ir or {})
    if not candidates:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    candidate_lines = [_visual_faiss_unified_candidate_line(index=index, candidate=candidate) for index, candidate in enumerate(candidates, start=1)]
    updated['visual_faiss_unified_candidate_lines'] = candidate_lines
    updated['evidence_lines'] = _ordered_unique_texts([*candidate_lines, *existing_evidence])
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    unified_facts = _visual_faiss_unified_candidate_facts(candidates)
    evidence_aggregation['candidate_facts'] = [*unified_facts, *candidate_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    unified_aggregation_lines = [_visual_faiss_unified_aggregation_line(index=index, fact=fact) for index, fact in enumerate(unified_facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*unified_aggregation_lines, *existing_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS unified graph candidates:'])
    context_lines.extend((f'- {line}' for line in candidate_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    metadata = {'contract': 'experimental_visual_faiss_unified_candidates_v1', 'object_count': len(selected_objects), 'context_count': len(contexts), 'candidate_count': len(candidates), 'object_limit': int(max(1, object_limit)), 'context_radius': int(max(0, context_radius)), 'context_limit': int(max(1, context_limit)), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'ranking_signals': ['object FAISS similarity', 'frame-level HSGG target term coverage', 'frame-level HSGG constraint term coverage', 'object-owned attribute facts', 'object-owned relation facts', 'nearby canonical graph support', 'graph retrieval evidence overlap'], 'candidates': candidates}
    updated['visual_faiss_unified_candidates'] = metadata
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_unified_candidate_lines'] = candidate_lines
    structured['visual_faiss_unified_candidate_facts'] = unified_facts
    structured['visual_faiss_unified_candidates'] = metadata
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for candidate in candidates:
            digest.append({'rank': next_rank, 'source_record_id': candidate.get('source_record_id'), 'source_kind': 'visual_faiss_unified_candidate', 'summary': candidate.get('summary'), 'score': candidate.get('score'), 'binding_status': candidate.get('binding_status'), 'visual_object': candidate.get('visual_object'), 'graph_support': candidate.get('graph_support'), 'binding_checks': candidate.get('binding_checks')})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def fuse_visual_faiss_evidence_layer(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, evidence_layer_sqlite: Path=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime/evidence_layer.sqlite'), object_limit: int=8, neighbor_limit: int=4, question: str='', question_ir: dict[str, Any] | None=None, constraint_binding: bool=False) -> dict[str, Any]:
    """Fuse object FAISS hits through the standalone evidence-layer index.

    The experiment keeps CG and FAISS indexes immutable. FAISS decides which
    object observations to inspect; the evidence layer supplies object-bound
    attributes, HSGG relations, same-frame bbox neighbors, and image/time anchors.
    It never reads gold answers or question-type labels.
    """
    selected_objects = _visual_faiss_ger_object_rows(visual_payload, object_limit=max(1, int(object_limit)))
    if not selected_objects:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    index = EvidenceLayerIndex(evidence_layer_sqlite)
    try:
        facts = _visual_faiss_evidence_layer_facts(index, selected_objects, neighbor_limit=max(0, int(neighbor_limit)), question=question, question_ir=question_ir or {}, constraint_binding=constraint_binding)
    finally:
        index.close()
    if not facts:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    fact_lines = [_visual_faiss_evidence_layer_fact_line(index=index, fact=fact) for index, fact in enumerate(facts, start=1)]
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    updated['visual_faiss_evidence_layer_lines'] = fact_lines
    updated['evidence_lines'] = _ordered_unique_texts([*fact_lines, *existing_evidence])
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*facts, *candidate_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    evidence_layer_aggregation_lines = [_visual_faiss_evidence_layer_aggregation_line(index=index, fact=fact) for index, fact in enumerate(facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*evidence_layer_aggregation_lines, *existing_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS evidence-layer local packs:'])
    context_lines.extend((f'- {line}' for line in fact_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    metadata = {'contract': 'experimental_visual_faiss_evidence_layer_v2_constraint_binding' if constraint_binding else 'experimental_visual_faiss_evidence_layer_v1', 'source_index': str(resolve_repo_input_path(evidence_layer_sqlite)), 'object_count': len(selected_objects), 'fact_count': len(facts), 'object_limit': int(max(1, object_limit)), 'neighbor_limit': int(max(0, neighbor_limit)), 'constraint_binding_enabled': bool(constraint_binding), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'retrieval_unit': 'object observation local evidence pack', 'facts': facts}
    updated['visual_faiss_evidence_layer'] = metadata
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_evidence_layer_lines'] = fact_lines
    structured['visual_faiss_evidence_layer_facts'] = facts
    structured['visual_faiss_evidence_layer'] = metadata
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for fact in facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_evidence_layer_fact', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'observation_id': fact.get('observation_id'), 'support_counts': fact.get('support_counts', {}), 'structural_binding_checks': fact.get('structural_binding_checks', {}), 'constraint_binding_enabled': bool(fact.get('constraint_binding_enabled'))})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def fuse_visual_faiss_role_aware_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, object_limit: int=8, context_limit: int=8, question: str='', question_ir: dict[str, Any] | None=None) -> dict[str, Any]:
    """Fuse visual FAISS hits as role-aware candidate subjects.

    FAISS still supplies visual recall, but question-IR target objects are the
    only normal candidate subjects. Hits retrieved for event/spatial/need
    phrases are retained as context attachments, not promoted as answer
    candidates. HSGG attributes are read from processed_attributes.json when
    present; missing object attributes are structured locally on demand and
    written back to that sidecar.
    """
    role_request = _visual_faiss_role_aware_request(question=question, question_ir=question_ir or {})
    rows = _visual_faiss_role_aware_rows(visual_payload, role_request=role_request)
    if not rows:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    primary_rows, context_rows = _visual_faiss_role_aware_split_rows(rows, role_request=role_request)
    primary_rows = primary_rows[:max(1, int(object_limit))]
    context_rows = context_rows[:max(0, int(context_limit))]
    facts = _visual_faiss_role_aware_candidate_facts(primary_rows, context_rows=context_rows, role_request=role_request)
    if not facts:
        return attach_visual_faiss_evidence(graph_answer, visual_payload)
    graph_answer = attach_visual_faiss_evidence(graph_answer, visual_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    fact_lines = [_visual_faiss_role_aware_fact_line(index=index, fact=fact) for index, fact in enumerate(facts, start=1)]
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    updated['visual_faiss_role_aware_lines'] = fact_lines
    updated['evidence_lines'] = _ordered_unique_texts([*fact_lines, *existing_evidence])
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*facts, *candidate_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    role_aware_aggregation_lines = [_visual_faiss_role_aware_aggregation_line(index=index, fact=fact) for index, fact in enumerate(facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*role_aware_aggregation_lines, *existing_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS role-aware candidate subjects:'])
    context_lines.extend((f'- {line}' for line in fact_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    metadata = {'contract': 'experimental_visual_faiss_role_aware_evidence_v1', 'row_count': len(rows), 'primary_candidate_count': len(primary_rows), 'context_only_count': len(context_rows), 'fact_count': len(facts), 'object_limit': int(max(1, object_limit)), 'context_limit': int(max(0, context_limit)), 'role_request': role_request, 'retrieval_unit': 'question-IR target-bound object observation', 'constraint_policy': 'event/spatial/evidence-need FAISS hits are context attachments only, not independent primary candidate subjects', 'processed_attribute_policy': 'read processed_attributes.json for each selected HSGG object; create a simple question-independent local entry when missing', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'facts': facts}
    updated['visual_faiss_role_aware_evidence'] = metadata
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_role_aware_lines'] = fact_lines
    structured['visual_faiss_role_aware_facts'] = facts
    structured['visual_faiss_role_aware_evidence'] = metadata
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for fact in facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_role_aware_candidate', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'role_checks': fact.get('role_checks', {}), 'processed_attribute': fact.get('processed_attribute', {}), 'attached_context': fact.get('attached_context', [])})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def attach_visual_faiss_role_aware_auxiliary_evidence(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, object_limit: int=6, context_limit: int=6, question: str='', question_ir: dict[str, Any] | None=None) -> dict[str, Any]:
    """Attach role-aware FAISS facts as auxiliary evidence.

    This is intentionally weaker than ``fuse_visual_faiss_role_aware_evidence``:
    it does not prepend role-aware facts to the main evidence_lines and does not
    replace GER-like/object-local evidence. The facts are exposed as a separate
    prompt/candidate channel so target/constraint role checks can guide QA
    without letting noisy target-query FAISS hits become the primary route.
    """
    role_request = _visual_faiss_role_aware_request(question=question, question_ir=question_ir or {})
    rows = _visual_faiss_role_aware_rows(visual_payload, role_request=role_request)
    if not rows:
        return graph_answer
    primary_rows, context_rows = _visual_faiss_role_aware_split_rows(rows, role_request=role_request)
    primary_rows = primary_rows[:max(1, int(object_limit))]
    context_rows = context_rows[:max(0, int(context_limit))]
    facts = _visual_faiss_role_aware_candidate_facts(primary_rows, context_rows=context_rows, role_request=role_request)
    if not facts:
        return graph_answer
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    fact_lines = [_visual_faiss_role_aware_aux_line(index=index, fact=fact) for index, fact in enumerate(facts, start=1)]
    updated['visual_faiss_role_aware_aux_lines'] = fact_lines
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    aux_facts = []
    for fact in facts:
        aux_fact = dict(fact)
        aux_fact['fact_type'] = 'visual_faiss_role_aware_aux_candidate'
        aux_fact['source'] = 'visual_faiss_role_aware_aux'
        aux_fact['source_kind'] = 'visual_faiss_role_aware_aux_candidate'
        aux_fact['source_record_id'] = str(aux_fact.get('source_record_id') or '').replace('visual_faiss_role_aware:', 'visual_faiss_role_aware_aux:')
        aux_facts.append(aux_fact)
    evidence_aggregation['candidate_facts'] = [*candidate_facts, *aux_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    aux_aggregation_lines = [_visual_faiss_role_aware_aux_aggregation_line(index=index, fact=fact) for index, fact in enumerate(aux_facts, start=1)]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*existing_aggregation_lines, *aux_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.append('Visual FAISS role-aware auxiliary role checks:')
    context_lines.extend((f'- {line}' for line in fact_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    metadata = {'contract': 'experimental_visual_faiss_role_aware_auxiliary_v1', 'row_count': len(rows), 'primary_candidate_count': len(primary_rows), 'context_only_count': len(context_rows), 'fact_count': len(facts), 'object_limit': int(max(1, object_limit)), 'context_limit': int(max(0, context_limit)), 'role_request': role_request, 'retrieval_unit': 'auxiliary question-IR role check over object FAISS hits', 'constraint_policy': 'role-aware hits are auxiliary constraints only; GER/object-local evidence remains the primary evidence route', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'facts': aux_facts}
    updated['visual_faiss_role_aware_auxiliary'] = metadata
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_aggregation'] = evidence_aggregation
    structured['visual_faiss_role_aware_aux_lines'] = fact_lines
    structured['visual_faiss_role_aware_auxiliary'] = metadata
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for fact in aux_facts:
            digest.append({'rank': next_rank, 'source_record_id': fact.get('source_record_id'), 'source_kind': 'visual_faiss_role_aware_aux_candidate', 'summary': fact.get('fact_text'), 'score': fact.get('score'), 'visual_object': fact.get('visual_object'), 'role_checks': fact.get('role_checks', {}), 'processed_attribute': fact.get('processed_attribute', {}), 'attached_context': fact.get('attached_context', [])})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def _visual_faiss_role_aware_request(*, question: str, question_ir: dict[str, Any]) -> dict[str, Any]:
    target_phrases = _visual_faiss_role_phrases(question_ir, 'target_entities', 'target_state')
    attribute_phrases = _visual_faiss_role_phrases(question_ir, 'attribute_dimension')
    event_phrases = _visual_faiss_role_phrases(question_ir, 'event_constraints', 'hard_constraints', 'temporal_relation')
    spatial_phrases = _visual_faiss_role_phrases(question_ir, 'spatial_constraints')
    need_phrases = _visual_faiss_role_phrases(question_ir, 'graph_information_needs', 'evidence_requirements')
    return {'target_phrases': target_phrases, 'attribute_phrases': attribute_phrases, 'event_phrases': event_phrases, 'spatial_phrases': spatial_phrases, 'need_phrases': need_phrases, 'target_terms': sorted(_visual_faiss_role_term_set(target_phrases)), 'attribute_terms': sorted(_visual_faiss_role_term_set(attribute_phrases)), 'event_terms': sorted(_visual_faiss_role_term_set(event_phrases)), 'spatial_terms': sorted(_visual_faiss_role_term_set(spatial_phrases)), 'need_terms': sorted(_visual_faiss_role_term_set(need_phrases)), 'question_terms': sorted(_visual_faiss_role_term_set([question]))}

def _visual_faiss_role_phrases(question_ir: dict[str, Any], *field_names: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for field_name in field_names:
        for value in _iter_text_values(question_ir.get(field_name)):
            phrase = _clean_phrase(value)
            key = ' '.join(phrase.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
    return phrases[:32]

def _visual_faiss_role_term_set(values: list[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        for token in _visual_faiss_text_tokens(value):
            terms.update(_visual_faiss_role_token_variants(token))
    return terms

def _visual_faiss_role_token_variants(token: str) -> set[str]:
    text = str(token or '').lower().strip()
    if not text:
        return set()
    variants = {text}
    if len(text) > 4 and text.endswith('ies'):
        variants.add(text[:-3] + 'y')
    if len(text) > 4 and text.endswith('es'):
        variants.add(text[:-2])
    if len(text) > 3 and text.endswith('s'):
        variants.add(text[:-1])
    return {value for value in variants if value and value not in _STOPWORDS}

def _visual_faiss_role_aware_rows(visual_payload: dict[str, Any], *, role_request: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for group in visual_payload.get('groups', []) or []:
        phrase = str(group.get('query_phrase') or '').strip()
        explicit_role = _visual_faiss_normalized_query_role(group.get('query_phrase_role'))
        phrase_role = explicit_role or _visual_faiss_role_for_phrase(phrase, role_request=role_request)
        phrase_role_label = str(group.get('query_phrase_role') or phrase_role).strip()
        for result in group.get('results', []) or []:
            if not isinstance(result, dict):
                continue
            key = (str(result.get('seq_id') or ''), _optional_int(result.get('graph_frame_index')), _optional_int(result.get('object_index')), phrase)
            if key in seen:
                continue
            seen.add(key)
            row = dict(result)
            row['query_phrase_role'] = phrase_role_label
            row['query_phrase_source'] = str(group.get('query_phrase_source') or '').strip()
            row['query_phrase_text'] = str(group.get('query_phrase_text') or '').strip()
            row['role_query_phrase'] = phrase
            row['role_query_role'] = phrase_role
            row['role_anchor_terms'] = sorted(_visual_faiss_role_anchor_terms(row))
            row['role_context_terms'] = sorted(_visual_faiss_role_context_terms(row))
            row['role_checks'] = _visual_faiss_role_checks(row, phrase=phrase, phrase_role=phrase_role, role_request=role_request)
            row['role_score'] = _visual_faiss_role_row_score(row)
            rows.append((phrase, row))
    rows.sort(key=lambda pair: (-int(bool(pair[1].get('role_checks', {}).get('is_primary_candidate'))), -int(bool(pair[1].get('role_checks', {}).get('target_query_hit'))), -len(pair[1].get('role_checks', {}).get('target_hits', []) or []), -float(pair[1].get('role_score') or 0.0), -(_optional_float(pair[1].get('similarity')) or 0.0), _optional_int(pair[1].get('graph_frame_index')) or 0, _optional_int(pair[1].get('object_index')) or 0))
    return rows

def _visual_faiss_normalized_query_role(value: Any) -> str:
    role = str(value or '').strip().lower().replace('-', '_')
    mapping = {'primary_target': 'target', 'target': 'target', 'target_entity': 'target', 'target_entities': 'target', 'target_state': 'target', 'target_attribute': 'attribute', 'attribute': 'attribute', 'attribute_dimension': 'attribute', 'event': 'event_constraint', 'event_constraint': 'event_constraint', 'hard_constraint': 'event_constraint', 'spatial': 'spatial_constraint', 'spatial_constraint': 'spatial_constraint', 'support': 'evidence_need', 'evidence_need': 'evidence_need', 'evidence_requirement': 'evidence_need', 'evidence_requirements': 'evidence_need', 'fallback': 'question_context', 'question': 'question_context', 'question_context': 'question_context', 'core_terms': 'question_context'}
    return mapping.get(role, '')

def _visual_faiss_role_for_phrase(phrase: str, *, role_request: dict[str, Any]) -> str:
    cleaned = _clean_phrase(str(phrase).replace('a photo of', ' '))
    phrase_terms = _visual_faiss_role_term_set([cleaned])
    if phrase_terms & set(role_request.get('target_terms', set())):
        return 'target'
    if phrase_terms & set(role_request.get('attribute_terms', set())):
        return 'attribute'
    if phrase_terms & set(role_request.get('event_terms', set())):
        return 'event_constraint'
    if phrase_terms & set(role_request.get('spatial_terms', set())):
        return 'spatial_constraint'
    if phrase_terms & set(role_request.get('need_terms', set())):
        return 'evidence_need'
    return 'question_context'

def _visual_faiss_role_anchor_terms(row: dict[str, Any]) -> set[str]:
    values = [str(row.get('label') or ''), str(row.get('visual_attribute') or ''), str(row.get('object_summary') or '')]
    return _visual_faiss_role_term_set(values)

def _visual_faiss_role_context_terms(row: dict[str, Any]) -> set[str]:
    values = [*[str(value) for value in row.get('visual_relations', []) or []], *[str(value) for value in row.get('hsgg_visual_facts', []) or []], *[str(value) for value in row.get('hsgg_neighbor_attributes', []) or []], *[str(value) for value in row.get('hsgg_frame_context', []) or []]]
    return _visual_faiss_role_term_set(values)

def _visual_faiss_role_checks(row: dict[str, Any], *, phrase: str, phrase_role: str, role_request: dict[str, Any]) -> dict[str, Any]:
    anchor_terms = set(row.get('role_anchor_terms', []) or [])
    context_terms = set(row.get('role_context_terms', []) or [])
    phrase_terms = _visual_faiss_role_term_set([phrase])
    target_terms = set(role_request.get('target_terms', set()))
    attribute_terms = set(role_request.get('attribute_terms', set()))
    event_terms = set(role_request.get('event_terms', set()))
    spatial_terms = set(role_request.get('spatial_terms', set()))
    need_terms = set(role_request.get('need_terms', set()))
    target_query_hit = bool(phrase_role == 'target' or phrase_terms & target_terms)
    target_hits = sorted(anchor_terms & target_terms)
    attribute_hits = sorted((anchor_terms | context_terms | phrase_terms) & attribute_terms)
    event_hits = sorted((context_terms | phrase_terms) & event_terms)
    spatial_hits = sorted((context_terms | phrase_terms) & spatial_terms)
    need_hits = sorted((context_terms | phrase_terms) & need_terms)
    target_required = bool(target_terms)
    is_primary = bool(target_query_hit or target_hits or (not target_required))
    return {'query_role': phrase_role, 'target_required': target_required, 'is_primary_candidate': is_primary, 'target_query_hit': target_query_hit, 'target_hits': target_hits, 'attribute_hits': attribute_hits, 'event_hits': event_hits, 'spatial_hits': spatial_hits, 'need_hits': need_hits, 'context_only_reason': '' if is_primary else 'query/object did not bind to target_entities; kept as context only'}

def _visual_faiss_role_row_score(row: dict[str, Any]) -> float:
    checks = row.get('role_checks', {}) if isinstance(row.get('role_checks'), dict) else {}
    score = (_optional_float(row.get('similarity')) or 0.0) * 8.0
    if checks.get('target_query_hit'):
        score += 4.0
    score += len(checks.get('target_hits', []) or []) * 3.0
    score += len(checks.get('attribute_hits', []) or []) * 1.4
    score += len(checks.get('event_hits', []) or []) * 1.1
    score += len(checks.get('spatial_hits', []) or []) * 1.1
    score += len(checks.get('need_hits', []) or []) * 0.8
    if row.get('visual_attribute'):
        score += 1.0
    if row.get('visual_relations'):
        score += 0.7
    return round(score, 6)

def _visual_faiss_role_aware_split_rows(rows: list[tuple[str, dict[str, Any]]], *, role_request: dict[str, Any]) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    primary: list[tuple[str, dict[str, Any]]] = []
    context: list[tuple[str, dict[str, Any]]] = []
    seen_primary: set[tuple[str, int | None, int | None]] = set()
    seen_context: set[tuple[str, int | None, int | None]] = set()
    for phrase, row in rows:
        key = _visual_faiss_role_row_key(row)
        checks = row.get('role_checks', {}) if isinstance(row.get('role_checks'), dict) else {}
        if checks.get('is_primary_candidate'):
            if key in seen_primary:
                continue
            seen_primary.add(key)
            primary.append((phrase, row))
        else:
            if key in seen_context:
                continue
            seen_context.add(key)
            context.append((phrase, row))
    if not primary:
        for phrase, row in rows:
            key = _visual_faiss_role_row_key(row)
            if key in seen_primary:
                continue
            seen_primary.add(key)
            primary.append((phrase, row))
            if len(primary) >= 3:
                break
    return (primary, context)

def _visual_faiss_role_row_key(row: dict[str, Any]) -> tuple[str, int | None, int | None]:
    return (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))

def _visual_faiss_role_aware_candidate_facts(primary_rows: list[tuple[str, dict[str, Any]]], *, context_rows: list[tuple[str, dict[str, Any]]], role_request: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, (phrase, row) in enumerate(primary_rows, start=1):
        processed = _visual_faiss_processed_attribute_for_row(row)
        neighbor_processed = _visual_faiss_processed_neighbor_attributes(row)
        attached_context = _visual_faiss_role_attached_contexts(row, context_rows=context_rows, role_request=role_request)
        direct_attribute_lines = _visual_faiss_processed_attribute_lines(processed, role='direct_object_attribute')
        neighbor_attribute_lines = [line for record in neighbor_processed for line in _visual_faiss_processed_attribute_lines(record, role='neighbor_object_attribute')][:4]
        fact_text = _visual_faiss_role_aware_fact_text(row=row, processed=processed, direct_attribute_lines=direct_attribute_lines, neighbor_attribute_lines=neighbor_attribute_lines, attached_context=attached_context)
        score = _visual_faiss_role_candidate_score(row=row, processed=processed, direct_attribute_lines=direct_attribute_lines, attached_context=attached_context)
        facts.append({'fact_type': 'visual_faiss_role_aware_candidate', 'fact_text': fact_text, 'binding_strength': _visual_faiss_role_binding_strength(row, processed), 'support_ranks': [index], 'support_examples': [_compressed_visual_object_text(row), *direct_attribute_lines[:3], *neighbor_attribute_lines[:2], *[str(item.get('summary') or '').strip() for item in attached_context[:2]]], 'source_record_id': f'visual_faiss_role_aware:{index}', 'source': 'visual_faiss_role_aware', 'source_kind': 'visual_faiss_role_aware_candidate', 'query_phrase': phrase, 'visual_object': _concise_visual_faiss_ger_object(row), 'role_checks': row.get('role_checks', {}), 'processed_attribute': processed or {}, 'neighbor_processed_attributes': neighbor_processed, 'direct_attribute_fragments': direct_attribute_lines, 'neighbor_attribute_fragments': neighbor_attribute_lines, 'attached_context': attached_context, 'score': score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    facts.sort(key=_visual_faiss_role_candidate_sort_key)
    for rank, fact in enumerate(facts, start=1):
        fact['support_ranks'] = [rank]
        fact['role_aware_rank'] = rank
    return facts

def _visual_faiss_processed_attribute_for_row(row: dict[str, Any]) -> dict[str, Any] | None:
    hsgg_root = _hsgg_root_for_metadata_row(row)
    frame_name = str(row.get('frame_key') or '').strip()
    object_index = _optional_int(row.get('object_index'))
    if hsgg_root is None or not frame_name or object_index is None:
        return None
    try:
        return get_or_create_processed_attribute_record(hsgg_root, frame_name=frame_name, object_index=object_index, label=str(row.get('label') or ''), raw_attributes=[str(row.get('visual_attribute') or '').strip()], image_path=str(row.get('image_path') or '') or None, bbox=row.get('bbox'), score=_optional_float(row.get('score')))
    except Exception as exc:
        return {'processed_attribute_error': str(exc), 'frame_name': frame_name, 'object_index': object_index, 'label': row.get('label')}

def _visual_faiss_processed_neighbor_attributes(row: dict[str, Any]) -> list[dict[str, Any]]:
    hsgg_root = _hsgg_root_for_metadata_row(row)
    frame_name = str(row.get('frame_key') or '').strip()
    if hsgg_root is None or not frame_name:
        return []
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for text in row.get('hsgg_neighbor_attributes', []) or []:
        parsed = _visual_faiss_parse_neighbor_attribute(text)
        object_index = _optional_int(parsed.get('object_index'))
        if object_index is None or object_index in seen:
            continue
        seen.add(object_index)
        try:
            record = get_or_create_processed_attribute_record(hsgg_root, frame_name=frame_name, object_index=object_index, label=str(parsed.get('label') or ''), raw_attributes=[str(parsed.get('attribute') or '').strip()])
        except Exception as exc:
            record = {'processed_attribute_error': str(exc), 'frame_name': frame_name, 'object_index': object_index, 'label': parsed.get('label')}
        if record:
            records.append(record)
        if len(records) >= 3:
            break
    return records

def _visual_faiss_parse_neighbor_attribute(text: Any) -> dict[str, Any]:
    value = str(text or '')
    object_match = re.search('\\bobj=(\\d+)\\b', value)
    label_match = re.search('\\blabel=(?:\'([^\']+)\'|\\"([^\\"]+)\\"|([^ ]+))', value)
    attr_match = re.search('\\battr=(.+)$', value)
    return {'object_index': int(object_match.group(1)) if object_match else None, 'label': next((group for group in (label_match.group(1) if label_match else '', label_match.group(2) if label_match else '', label_match.group(3) if label_match else '') if group), '') if label_match else '', 'attribute': attr_match.group(1).strip() if attr_match else ''}

def _visual_faiss_processed_attribute_lines(record: dict[str, Any] | None, *, role: str) -> list[str]:
    if not isinstance(record, dict) or record.get('processed_attribute_error'):
        return []
    owner_parts = [f"frame={record.get('frame_name')}", f"obj={record.get('object_index')}", f"label={record.get('label')!r}"]
    owner = ' '.join((part for part in owner_parts if part))
    structured = record.get('structured', {}) if isinstance(record.get('structured'), dict) else {}
    lines: list[str] = []
    for fact in structured.get('facts', []) or []:
        if not isinstance(fact, dict):
            continue
        kind = str(fact.get('kind') or '').strip()
        target = str(fact.get('target') or '').strip()
        value = str(fact.get('value') or '').strip()
        modifiers = [str(item).strip() for item in fact.get('modifiers', []) or [] if str(item).strip()]
        if not value:
            continue
        modifier_text = f" modifiers={','.join(modifiers[:4])}" if modifiers else ''
        lines.append(_short_text(f'{role} owner={owner} kind={kind} target={target!r} value={value!r}{modifier_text}', limit=220))
    if not lines:
        summary = str(structured.get('summary') or '').strip()
        if summary:
            lines.append(_short_text(f'{role} owner={owner} summary={summary}', limit=220))
    return _ordered_unique_texts(lines)[:6]

def _visual_faiss_role_attached_contexts(row: dict[str, Any], *, context_rows: list[tuple[str, dict[str, Any]]], role_request: dict[str, Any]) -> list[dict[str, Any]]:
    anchor_frame = _optional_int(row.get('graph_frame_index'))
    scored: list[dict[str, Any]] = []
    for phrase, context_row in context_rows:
        if _visual_faiss_role_row_key(context_row) == _visual_faiss_role_row_key(row):
            continue
        frame = _optional_int(context_row.get('graph_frame_index'))
        frame_distance = abs(frame - anchor_frame) if frame is not None and anchor_frame is not None else 10 ** 9
        checks = context_row.get('role_checks', {}) if isinstance(context_row.get('role_checks'), dict) else {}
        support_score = _optional_float(context_row.get('similarity')) or 0.0
        support_score += max(0.0, 3.0 - min(frame_distance, 30) / 10.0)
        support_score += len(checks.get('event_hits', []) or []) * 1.0
        support_score += len(checks.get('spatial_hits', []) or []) * 1.0
        support_score += len(checks.get('need_hits', []) or []) * 0.7
        summary_parts = [f'context_query={phrase!r}' if phrase else '', f"role={checks.get('query_role')}", _compressed_visual_object_text(context_row)]
        fact_summary = _visual_faiss_fact_summary(context_row, max_facts=3)
        if fact_summary:
            summary_parts.append(f'hsgg_facts={fact_summary}')
        scored.append({'source_record_id': f"visual_faiss_role_context:{context_row.get('seq_id')}:{frame}:{context_row.get('object_index')}", 'summary': _short_text(' '.join((part for part in summary_parts if part)), limit=360), 'score': round(support_score, 6), 'frame_distance': None if frame_distance == 10 ** 9 else frame_distance, 'query_phrase': phrase, 'query_role': checks.get('query_role'), 'visual_object': _concise_visual_faiss_ger_object(context_row), 'role_checks': checks})
    scored.sort(key=lambda item: (_optional_int(item.get('frame_distance')) if item.get('frame_distance') is not None else 10 ** 9, -float(item.get('score') or 0.0), str(item.get('source_record_id') or '')))
    return scored[:3]

def _visual_faiss_role_aware_fact_text(*, row: dict[str, Any], processed: dict[str, Any] | None, direct_attribute_lines: list[str], neighbor_attribute_lines: list[str], attached_context: list[dict[str, Any]]) -> str:
    pieces = ['role_aware_candidate_subject', _compressed_visual_object_text(row)]
    checks = row.get('role_checks', {}) if isinstance(row.get('role_checks'), dict) else {}
    pieces.append(f"query_role={checks.get('query_role')}")
    if checks.get('target_hits'):
        pieces.append('target_hits=' + ','.join(checks.get('target_hits', [])[:6]))
    if checks.get('attribute_hits'):
        pieces.append('attribute_hits=' + ','.join(checks.get('attribute_hits', [])[:6]))
    if direct_attribute_lines:
        pieces.append('direct_processed_attributes=' + ' | '.join(direct_attribute_lines[:4]))
    elif row.get('visual_attribute'):
        pieces.append('direct_object_attribute=' + _short_text(row.get('visual_attribute'), limit=160))
    relations = [_short_text(value, limit=130) for value in row.get('visual_relations', []) or [] if _short_text(value, limit=130)]
    if relations:
        pieces.append('direct_relations=' + ' | '.join(relations[:3]))
    if neighbor_attribute_lines:
        pieces.append('neighbor_processed_attributes=' + ' | '.join(neighbor_attribute_lines[:2]))
    context_summaries = [str(item.get('summary') or '').strip() for item in attached_context if str(item.get('summary') or '').strip()]
    if context_summaries:
        pieces.append('constraint_context=' + ' | '.join(context_summaries[:2]))
    if isinstance(processed, dict) and processed.get('processed_attribute_error'):
        pieces.append('processed_attribute_error=' + _short_text(processed.get('processed_attribute_error'), limit=140))
    return _short_text(' '.join((piece for piece in pieces if piece)), limit=1400)

def _visual_faiss_role_candidate_score(*, row: dict[str, Any], processed: dict[str, Any] | None, direct_attribute_lines: list[str], attached_context: list[dict[str, Any]]) -> float:
    score = _optional_float(row.get('role_score')) or 0.0
    if direct_attribute_lines:
        score += min(3.0, len(direct_attribute_lines) * 0.7)
    structured = processed.get('structured', {}) if isinstance(processed, dict) else {}
    if isinstance(structured, dict) and structured.get('identity_signature'):
        score += 0.8
    score += min(2.0, len(attached_context) * 0.5)
    return round(score, 6)

def _visual_faiss_role_binding_strength(row: dict[str, Any], processed: dict[str, Any] | None) -> str:
    checks = row.get('role_checks', {}) if isinstance(row.get('role_checks'), dict) else {}
    structured = processed.get('structured', {}) if isinstance(processed, dict) else {}
    has_structured_attrs = bool(isinstance(structured, dict) and structured.get('facts'))
    if checks.get('target_query_hit') and has_structured_attrs:
        return 'target_query_with_structured_direct_attribute'
    if checks.get('target_hits') and has_structured_attrs:
        return 'target_anchor_with_structured_direct_attribute'
    if checks.get('target_query_hit') or checks.get('target_hits'):
        return 'target_bound_visual_subject'
    return 'fallback_visual_subject'

def _visual_faiss_role_candidate_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    checks = fact.get('role_checks', {}) if isinstance(fact.get('role_checks'), dict) else {}
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-int(bool(checks.get('target_query_hit') or checks.get('target_hits'))), -len(fact.get('direct_attribute_fragments', []) or []), -len(fact.get('attached_context', []) or []), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, _optional_int(visual_object.get('object_index')) or 0)

def _visual_faiss_role_aware_fact_line(*, index: int, fact: dict[str, Any]) -> str:
    checks = fact.get('role_checks', {}) if isinstance(fact.get('role_checks'), dict) else {}
    coverage_parts = [f"target={('yes' if checks.get('target_query_hit') or checks.get('target_hits') else 'no')}", f"attribute={('yes' if checks.get('attribute_hits') or fact.get('direct_attribute_fragments') else 'no')}", f"context={len(fact.get('attached_context', []) or [])}"]
    return f"rank={index} kind=visual_faiss_role_aware_candidate score={float(fact.get('score') or 0.0):.2f} binding={fact.get('binding_strength')} coverage={'/'.join(coverage_parts)} :: {fact.get('fact_text')}"

def _visual_faiss_role_aware_aux_line(*, index: int, fact: dict[str, Any]) -> str:
    checks = fact.get('role_checks', {}) if isinstance(fact.get('role_checks'), dict) else {}
    coverage_parts = [f"target={('yes' if checks.get('target_query_hit') or checks.get('target_hits') else 'no')}", f"attribute={('yes' if checks.get('attribute_hits') or fact.get('direct_attribute_fragments') else 'no')}", f"context={len(fact.get('attached_context', []) or [])}"]
    return f"rank={index} kind=visual_faiss_role_aware_aux_candidate score={float(fact.get('score') or 0.0):.2f} binding={fact.get('binding_strength')} coverage={'/'.join(coverage_parts)} :: {fact.get('fact_text')}"

def _visual_faiss_role_aware_aggregation_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    checks = fact.get('role_checks', {}) if isinstance(fact.get('role_checks'), dict) else {}
    pieces = [f'group=visual_faiss_role_aware:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"obj={visual_object.get('object_index')}", f"label={visual_object.get('label')!r}", f"query_role={checks.get('query_role')}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        pieces.append('examples=' + ' | '.join(examples[:3]))
    return _short_text(' '.join((piece for piece in pieces if piece)), limit=1200)

def _visual_faiss_role_aware_aux_aggregation_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    checks = fact.get('role_checks', {}) if isinstance(fact.get('role_checks'), dict) else {}
    pieces = [f'group=visual_faiss_role_aware_aux:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"obj={visual_object.get('object_index')}", f"label={visual_object.get('label')!r}", f"query_role={checks.get('query_role')}", 'auxiliary_only=true', str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        pieces.append('examples=' + ' | '.join(examples[:3]))
    return _short_text(' '.join((piece for piece in pieces if piece)), limit=1200)

def _visual_faiss_fused_evidence_lines(visual_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for phrase, row in _compressed_visual_faiss_rows(visual_payload):
        text = _visual_faiss_object_evidence_line(phrase=phrase, row=row)
        if not text or text in seen:
            continue
        lines.append(text)
        seen.add(text)
    return lines

def _visual_faiss_evidence_layer_facts(index: EvidenceLayerIndex, object_rows: list[tuple[str, dict[str, Any]]], *, neighbor_limit: int, question: str='', question_ir: dict[str, Any] | None=None, constraint_binding: bool=False) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    structural_request = _visual_faiss_structural_binding_request(question=question, question_ir=question_ir or {}) if constraint_binding else {}
    for context_index, (phrase, row) in enumerate(object_rows, start=1):
        observation_id = _visual_faiss_evidence_layer_observation_id(row)
        if not observation_id or observation_id in seen_observations:
            continue
        seen_observations.add(observation_id)
        try:
            pack = index.local_evidence_pack(observation_id, neighbor_limit=neighbor_limit)
        except KeyError:
            continue
        fact = _visual_faiss_evidence_layer_fact(context_index=context_index, phrase=phrase, row=row, pack=pack, structural_request=structural_request, constraint_binding=constraint_binding)
        if fact is not None:
            facts.append(fact)
    facts.sort(key=_visual_faiss_evidence_layer_constraint_sort_key if constraint_binding else _visual_faiss_evidence_layer_sort_key)
    return facts

def _visual_faiss_evidence_layer_observation_id(row: dict[str, Any]) -> str | None:
    seq_id = str(row.get('seq_id') or '').strip()
    object_index = _optional_int(row.get('object_index'))
    frame_key = str(row.get('frame_key') or '').strip()
    if not frame_key:
        image_name = str(row.get('image_name') or '').strip()
        frame_key = Path(image_name).stem if image_name else ''
    if not seq_id or not frame_key or object_index is None:
        return None
    return f'seq{seq_id}:frame:{frame_key}:object:{object_index}'

def _visual_faiss_evidence_layer_fact(*, context_index: int, phrase: str, row: dict[str, Any], pack: dict[str, Any], structural_request: dict[str, list[str]] | None=None, constraint_binding: bool=False) -> dict[str, Any] | None:
    observation = pack.get('observation')
    if not isinstance(observation, dict):
        return None
    attributes = [str(item.get('attribute') or '').strip() for item in pack.get('attributes', []) or [] if isinstance(item, dict) and str(item.get('attribute') or '').strip()]
    relation_texts = _visual_faiss_evidence_layer_relation_texts(pack)
    neighbor_texts = _visual_faiss_evidence_layer_neighbor_texts(pack)
    anchor = _visual_faiss_evidence_layer_anchor_text(observation, row=row)
    fact_text = _visual_faiss_evidence_layer_fact_text(observation=observation, anchor=anchor, attributes=attributes, relations=relation_texts, neighbors=neighbor_texts)
    if not fact_text:
        return None
    score = _visual_faiss_evidence_layer_score(row=row, attributes=attributes, relations=relation_texts, neighbors=neighbor_texts)
    binding_payload = _visual_faiss_evidence_layer_constraint_binding(row=row, observation=observation, phrase=phrase, attributes=attributes, relations=relation_texts, neighbors=neighbor_texts, structural_request=structural_request or {}, base_score=score) if constraint_binding else {}
    if binding_payload:
        score = float(binding_payload.get('score') or score)
    fact = {'fact_type': 'visual_faiss_evidence_layer_fact', 'fact_text': fact_text, 'score': score, 'binding_strength': _visual_faiss_evidence_layer_strength(attributes=attributes, relations=relation_texts, neighbors=neighbor_texts), 'source_record_id': f"visual_faiss_evidence_layer:{context_index}:{observation.get('observation_id')}", 'source': 'visual_faiss_evidence_layer', 'source_kind': 'visual_faiss_evidence_layer_fact', 'query_phrase': phrase, 'observation_id': observation.get('observation_id'), 'visual_object': _concise_visual_faiss_ger_object(row), 'evidence_layer_anchor': anchor, 'attributes': attributes[:4], 'relations': relation_texts[:4], 'bbox_neighbors': neighbor_texts[:4], 'support_counts': {'attributes': len(attributes), 'relations': len(relation_texts), 'bbox_neighbors': len(neighbor_texts)}}
    fact.update(binding_payload)
    return fact

def _visual_faiss_evidence_layer_relation_texts(pack: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    anchor_id = ''
    observation = pack.get('observation')
    if isinstance(observation, dict):
        anchor_id = str(observation.get('observation_id') or '')
    for item in pack.get('relations', []) or []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get('subject_label') or '').strip()
        predicate = str(item.get('predicate') or '').strip()
        obj = str(item.get('object_label') or '').strip()
        if not predicate and item.get('raw_output'):
            predicate = str(item.get('raw_output') or '').strip()
        direction = ''
        if anchor_id:
            if item.get('subject_observation_id') == anchor_id:
                direction = 'anchor_subject'
            elif item.get('object_observation_id') == anchor_id:
                direction = 'anchor_object'
        text = ' '.join((part for part in [direction, subject, predicate, obj] if part))
        if text:
            lines.append(_short_text(text, limit=140))
    return _ordered_unique_texts(lines)

def _visual_faiss_evidence_layer_neighbor_texts(pack: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in pack.get('bbox_neighbors', []) or []:
        if not isinstance(item, dict):
            continue
        obj_label = str(item.get('object_label') or '').strip()
        relations = ', '.join((str(value) for value in item.get('relations', []) or []))
        geometry = item.get('geometry', {}) if isinstance(item.get('geometry'), dict) else {}
        distance = geometry.get('normalized_center_distance')
        distance_text = f'dist={float(distance):.3f}' if isinstance(distance, (int, float)) else ''
        text = ' '.join((part for part in [f'neighbor={obj_label!r}' if obj_label else 'neighbor', relations, distance_text] if part))
        lines.append(_short_text(text, limit=140))
    return _ordered_unique_texts(lines)

def _visual_faiss_evidence_layer_anchor_text(observation: dict[str, Any], *, row: dict[str, Any]) -> dict[str, Any]:
    return {'seq_id': observation.get('seq_id'), 'frame_key': observation.get('frame_key'), 'graph_frame_index': observation.get('graph_frame_index'), 'elapsed_sec_from_sequence_start': observation.get('elapsed_sec_from_sequence_start'), 'object_index': observation.get('object_index'), 'label': observation.get('label'), 'bbox': observation.get('bbox'), 'image_name': observation.get('image_name'), 'image_path': observation.get('image_path'), 'faiss_similarity': row.get('similarity'), 'query_overlap_tokens': list(row.get('generic_query_overlap_tokens', []) or [])}

def _visual_faiss_evidence_layer_fact_text(*, observation: dict[str, Any], anchor: dict[str, Any], attributes: list[str], relations: list[str], neighbors: list[str]) -> str:
    pieces = ['evidence_layer_object_pack', f"obs={observation.get('observation_id')}", f"label={observation.get('label')!r}", f"frame={observation.get('graph_frame_index')}"]
    similarity = anchor.get('faiss_similarity')
    if isinstance(similarity, (int, float)):
        pieces.append(f'similarity={float(similarity):.3f}')
    if attributes:
        pieces.append('attrs=' + ' | '.join((_short_text(value, limit=110) for value in attributes[:3])))
    if relations:
        pieces.append('relations=' + ' | '.join(relations[:3]))
    if neighbors:
        pieces.append('bbox_neighbors=' + ' | '.join(neighbors[:3]))
    return ' '.join((part for part in pieces if part))

def _visual_faiss_evidence_layer_score(*, row: dict[str, Any], attributes: list[str], relations: list[str], neighbors: list[str]) -> float:
    score = 70.0
    similarity = row.get('similarity')
    if isinstance(similarity, (int, float)):
        score += float(similarity) * 12.0
    score += min(8.0, len(attributes) * 2.0)
    score += min(8.0, len(relations) * 2.0)
    score += min(4.0, len(neighbors) * 1.0)
    return score

def _visual_faiss_evidence_layer_constraint_binding(*, row: dict[str, Any], observation: dict[str, Any], phrase: str, attributes: list[str], relations: list[str], neighbors: list[str], structural_request: dict[str, list[str]], base_score: float) -> dict[str, Any]:
    context = _visual_faiss_evidence_layer_binding_context(row=row, observation=observation, phrase=phrase, attributes=attributes, relations=relations, neighbors=neighbors)
    anchor_tokens = _visual_faiss_text_tokens(context['anchor_text'])
    object_tokens = _visual_faiss_text_tokens(context['object_text'])
    graph_tokens = _visual_faiss_text_tokens(context['context_text'])
    combined_tokens = anchor_tokens | object_tokens | graph_tokens
    structural_checks = _visual_faiss_structural_binding_checks(combined_tokens=combined_tokens, anchor_tokens=anchor_tokens, object_tokens=object_tokens, graph_tokens=graph_tokens, request=structural_request, attribute_available=bool(attributes))
    binding_score = _visual_faiss_evidence_layer_constraint_score(base_score=base_score, row=row, structural_checks=structural_checks, attributes=attributes, relations=relations, neighbors=neighbors)
    coverage_text = _visual_faiss_evidence_layer_coverage_text(structural_checks)
    return {'score': binding_score, 'binding_strength': _visual_faiss_structural_binding_strength(structural_checks), 'constraint_binding_enabled': True, 'structural_binding_checks': structural_checks, 'constraint_coverage': coverage_text, 'binding_context': context, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}

def _visual_faiss_evidence_layer_binding_context(*, row: dict[str, Any], observation: dict[str, Any], phrase: str, attributes: list[str], relations: list[str], neighbors: list[str]) -> dict[str, str]:
    label = str(observation.get('label') or row.get('label') or '').strip()
    visual_attribute = str(row.get('visual_attribute') or '').strip()
    object_summary = str(row.get('object_summary') or '').strip()
    row_relations = [str(value).strip() for value in row.get('visual_relations', []) or [] if str(value).strip()]
    anchor_text = ' '.join((value for value in [label, object_summary] if str(value).strip()))
    object_text = ' '.join((value for value in [label, visual_attribute, object_summary, *attributes, *row_relations] if str(value).strip()))
    context_text = ' '.join((value for value in [*relations, *neighbors] if str(value).strip()))
    return {'anchor_text': _short_text(anchor_text, limit=500), 'object_text': _short_text(object_text, limit=700), 'context_text': _short_text(context_text, limit=900), 'query_text': _short_text(phrase, limit=200)}

def _visual_faiss_evidence_layer_constraint_score(*, base_score: float, row: dict[str, Any], structural_checks: dict[str, Any], attributes: list[str], relations: list[str], neighbors: list[str]) -> float:
    similarity = _optional_float(row.get('similarity')) or 0.0
    target_hits = list(structural_checks.get('target_hits', []) or [])
    attribute_hits = list(structural_checks.get('attribute_hits', []) or [])
    event_hits = list(structural_checks.get('event_hits', []) or [])
    spatial_hits = list(structural_checks.get('spatial_hits', []) or [])
    need_hits = list(structural_checks.get('need_hits', []) or [])
    graph_hits = list(structural_checks.get('graph_hits', []) or [])
    required_count = int(structural_checks.get('required_slot_count') or 0)
    satisfied_count = int(structural_checks.get('satisfied_slot_count') or 0)
    score = 60.0 + similarity * 6.0
    score += min(4.0, len(attributes) * 0.8)
    score += min(3.0, len(relations) * 0.6)
    score += min(1.5, len(neighbors) * 0.25)
    score += len(target_hits) * 4.0
    score += len(attribute_hits) * 2.0
    score += len(event_hits) * 2.4
    score += len(spatial_hits) * 2.2
    score += len(need_hits) * 1.2
    score += min(2.0, len(graph_hits) * 0.3)
    if required_count:
        score += satisfied_count * 3.0
        if satisfied_count == required_count:
            score += 4.0
        elif satisfied_count == 0:
            score -= 4.0
    if target_hits and (attribute_hits or event_hits or spatial_hits or need_hits):
        score += 2.0
    if not target_hits and structural_checks.get('required_slots', {}).get('target'):
        score -= 2.5
    return round(max(score, float(base_score) * 0.5), 6)

def _visual_faiss_evidence_layer_strength(*, attributes: list[str], relations: list[str], neighbors: list[str]) -> str:
    direct_count = len(attributes) + len(relations)
    if direct_count >= 2:
        return 'direct_object_bound'
    if direct_count == 1:
        return 'partial_object_bound'
    if neighbors:
        return 'bbox_neighbor_context'
    return 'visual_anchor_only'

def _visual_faiss_evidence_layer_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    strength_order = {'direct_object_bound': 0, 'partial_object_bound': 1, 'bbox_neighbor_context': 2, 'visual_anchor_only': 3}
    return (strength_order.get(str(fact.get('binding_strength') or ''), 9), -float(fact.get('score') or 0.0), str(fact.get('observation_id') or ''))

def _visual_faiss_evidence_layer_constraint_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    checks = fact.get('structural_binding_checks', {}) if isinstance(fact.get('structural_binding_checks'), dict) else {}
    required = checks.get('required_slots', {}) if isinstance(checks.get('required_slots'), dict) else {}
    target_required = bool(required.get('target'))
    target_bound = 1 if checks.get('target_hits') else 0
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-(target_bound if target_required else 0), -int(checks.get('satisfied_slot_count') or 0), -len(checks.get('target_hits', []) or []), -len(checks.get('attribute_hits', []) or []), -len(checks.get('event_hits', []) or []), -len(checks.get('spatial_hits', []) or []), -len(checks.get('need_hits', []) or []), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, str(fact.get('observation_id') or ''))

def _visual_faiss_evidence_layer_coverage_text(structural_checks: dict[str, Any]) -> str:
    required = structural_checks.get('required_slots', {}) if isinstance(structural_checks.get('required_slots'), dict) else {}
    satisfied = structural_checks.get('satisfied_slots', {}) if isinstance(structural_checks.get('satisfied_slots'), dict) else {}
    pieces: list[str] = []
    missing: list[str] = []
    for slot in ('target', 'attribute', 'event', 'spatial', 'need'):
        if not required.get(slot):
            continue
        ok = bool(satisfied.get(slot))
        pieces.append(f"{slot}:{('yes' if ok else 'no')}")
        if not ok:
            missing.append(slot)
    if missing:
        pieces.append('missing=' + ','.join(missing))
    return ' '.join(pieces)

def _visual_faiss_evidence_layer_fact_line(*, index: int, fact: dict[str, Any]) -> str:
    coverage = str(fact.get('constraint_coverage') or '').strip()
    coverage_prefix = f'coverage={coverage} ' if coverage else ''
    return f"rank={index} kind=visual_faiss_evidence_layer_fact score={float(fact.get('score') or 0.0):.2f} binding={fact.get('binding_strength')} {coverage_prefix}:: {fact.get('fact_text')}"

def _visual_faiss_evidence_layer_aggregation_line(*, index: int, fact: dict[str, Any]) -> str:
    counts = fact.get('support_counts', {}) if isinstance(fact.get('support_counts'), dict) else {}
    return f"group=visual_faiss_evidence_layer:{index} binding={fact.get('binding_strength')} coverage={fact.get('constraint_coverage') or 'n/a'} attrs={counts.get('attributes', 0)} relations={counts.get('relations', 0)} bbox_neighbors={counts.get('bbox_neighbors', 0)} {fact.get('fact_text')}"

def _visual_faiss_object_evidence_line(*, phrase: str, row: dict[str, Any]) -> str:
    parts = ['visual_faiss_object', _compressed_visual_object_text(row)]
    fact_summary = _visual_faiss_fact_summary(row, max_facts=4)
    if fact_summary:
        parts.append(f'hsgg_facts={fact_summary}')
    query_role = str(row.get('query_phrase_role') or row.get('role_query_role') or '').strip()
    if query_role:
        parts.append(f'query_role={query_role!r}')
    if phrase:
        parts.append(f'query={phrase!r}')
    return ' '.join(parts)

def _visual_faiss_candidate_facts(visual_payload: dict[str, Any], *, evidence_offset: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    evidence_rank = evidence_offset + 1
    for phrase, row in _compressed_visual_faiss_rows(visual_payload):
        key = (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))
        if key in seen:
            continue
        seen.add(key)
        label = str(row.get('label') or 'object').strip() or 'object'
        frame = _optional_int(row.get('graph_frame_index'))
        fact_summary = _visual_faiss_fact_summary(row, max_facts=4)
        fact_text = f'Visual FAISS recalled {label!r}'
        if frame is not None:
            fact_text += f' at frame {frame}'
        if fact_summary:
            fact_text += f' with HSGG facts: {fact_summary}'
        if phrase:
            fact_text += f' for query {phrase!r}'
        facts.append({'fact_type': 'visual_faiss_object_recall', 'fact_text': fact_text, 'binding_strength': 'visual_similarity', 'support_ranks': [evidence_rank], 'support_examples': [_compressed_visual_object_text(row)], 'source': 'visual_faiss', 'query_phrase': phrase, 'label': label, 'frame_index': frame, 'similarity': row.get('similarity'), 'hsgg_visual_facts': list(row.get('hsgg_visual_facts', []) or []), 'hsgg_neighbor_attributes': list(row.get('hsgg_neighbor_attributes', []) or []), 'compression_score': row.get('compression_score'), 'generic_query_overlap_tokens': row.get('generic_query_overlap_tokens', [])})
        evidence_rank += 1
    return facts

def _visual_faiss_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    parts = [f'group=visual_faiss:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:2]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_line_score(index: int, visual_payload: dict[str, Any]) -> float | None:
    rows = _compressed_visual_faiss_rows(visual_payload)
    if index < 1 or index > len(rows):
        return None
    row = rows[index - 1][1]
    try:
        return float(row.get('similarity'))
    except (TypeError, ValueError):
        return None
    return None

def _visual_anchor_rows(visual_payload: dict[str, Any], *, max_anchors: int) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None, str]] = set()
    for phrase, row in _compressed_visual_faiss_rows(visual_payload):
        frame = _optional_int(row.get('graph_frame_index'))
        if frame is None:
            continue
        key = (frame, _optional_int(row.get('object_index')), phrase)
        if key in seen:
            continue
        seen.add(key)
        anchors.append({'frame_index': frame, 'timestamp_sec': row.get('elapsed_sec_from_sequence_start'), 'query_phrase': phrase, 'label': row.get('label'), 'object_index': _optional_int(row.get('object_index')), 'similarity': row.get('similarity'), 'visual_attribute': row.get('visual_attribute'), 'visual_relations': list(row.get('visual_relations', []) or []), 'hsgg_visual_facts': list(row.get('hsgg_visual_facts', []) or []), 'hsgg_neighbor_attributes': list(row.get('hsgg_neighbor_attributes', []) or []), 'compression_score': row.get('compression_score'), 'generic_query_overlap_tokens': list(row.get('generic_query_overlap_tokens', []) or []), 'object_summary': _compressed_visual_object_text(row)})
        if len(anchors) >= max_anchors:
            return anchors
    return anchors

def _graph_evidence_for_visual_anchors(*, graph: CanonicalTemporalMemoryGraph, anchors: list[dict[str, Any]], frame_radius: int, max_items: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor_rank, anchor in enumerate(anchors, start=1):
        frame = _optional_int(anchor.get('frame_index'))
        if frame is None:
            continue
        for item in _graph_items_touching_frame(graph, frame=frame, radius=frame_radius):
            item['visual_anchor'] = dict(anchor)
            item['visual_anchor_rank'] = anchor_rank
            item['anchor_distance'] = _frame_distance_to_range(frame, item.get('time_range', {}))
            item['score'] = _visual_anchor_graph_item_score(item, anchor=anchor)
            rows.append(item)
    rows.sort(key=lambda item: (int(item.get('visual_anchor_rank') or 9999), int(item.get('anchor_distance') or 999999), _source_kind_order(str(item.get('source_kind') or '')), -float(item.get('score') or 0.0), str(item.get('source_record_id') or '')))
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in rows:
        record_id = str(item.get('source_record_id') or '')
        if not record_id or record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        item['rank'] = len(selected) + 1
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected

def _graph_items_touching_frame(graph: CanonicalTemporalMemoryGraph, *, frame: int, radius: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in graph.scene_nodes:
        if _time_range_touches_frame(scene.time_range, frame=frame, radius=radius):
            rows.append({'source_record_id': scene.node_id, 'source_kind': 'scene', 'summary': _short_quote(scene.label, limit=140), 'time_range': _time_range_payload(scene.time_range), 'scene_id': scene.scene_id, 'entity_ids': [], 'metadata': {'scene_id': scene.scene_id, 'scene_kind_hint': scene.scene_kind_hint, 'representative_static_labels': list(scene.representative_static_labels), 'representative_dynamic_labels': list(scene.representative_dynamic_labels), 'salient_labels': list(scene.salient_labels)}})
    for entity in graph.entity_nodes:
        if _time_range_touches_frame(entity.time_range, frame=frame, radius=radius):
            rows.append({'source_record_id': entity.node_id, 'source_kind': 'entity', 'summary': _short_quote(entity.label, limit=140), 'time_range': _time_range_payload(entity.time_range), 'scene_id': None, 'entity_ids': [entity.entity_id], 'metadata': {'entity_id': entity.entity_id, 'label': entity.label, 'labels_seen': list(entity.labels_seen), 'states_seen': list(entity.states_seen), 'stable_identity_id': entity.stable_identity_id, 'stable_identity_display_name': entity.stable_identity_display_name, 'attributes': dict(entity.attributes)}})
    for episode in graph.episode_nodes:
        if _time_range_touches_frame(episode.time_range, frame=frame, radius=radius):
            rows.append({'source_record_id': episode.node_id, 'source_kind': 'episode', 'summary': _episode_summary(episode), 'time_range': _time_range_payload(episode.time_range), 'scene_id': episode.scene_id, 'entity_ids': [value for value in [episode.primary_entity_id, episode.subject_entity_id, episode.object_entity_id, *episode.participant_entity_ids] if value], 'metadata': {'episode_kind': episode.episode_kind, 'label': episode.label, 'attr_key': episode.attr_key, 'attr_value': episode.attr_value, 'predicate': episode.predicate, 'raw_predicate': episode.raw_predicate}})
    for event in graph.transition_event_nodes:
        if _time_range_touches_frame(event.time_range, frame=frame, radius=radius):
            rows.append({'source_record_id': event.node_id, 'source_kind': 'transition_event', 'summary': _transition_event_summary(event), 'time_range': _time_range_payload(event.time_range), 'scene_id': event.scene_id, 'entity_ids': [value for value in [event.entity_id, event.subject_entity_id, event.object_entity_id] if value], 'metadata': {'event_type': event.event_type, 'label': event.label, 'predicate': event.predicate, 'raw_predicate': event.raw_predicate, 'attr_key': event.attr_key, 'previous_state_value': event.previous_state_value, 'current_state_value': event.current_state_value}})
    for sample in graph.relation_observed_samples:
        if _time_range_touches_frame(sample.time_range, frame=frame, radius=radius):
            rows.append({'source_record_id': sample.sample_id, 'source_kind': 'relation_observed_sample', 'summary': _short_quote(sample.label, limit=140), 'time_range': _time_range_payload(sample.time_range), 'scene_id': sample.scene_id, 'entity_ids': [sample.subject_entity_id, sample.object_entity_id], 'metadata': {'predicate': sample.predicate, 'raw_predicate': sample.raw_predicate, 'label': sample.label}})
    return rows

def _time_range_touches_frame(time_range: Any, *, frame: int, radius: int) -> bool:
    start = _optional_int(getattr(time_range, 'start_frame', None))
    end = _optional_int(getattr(time_range, 'end_frame', None))
    if start is None and end is None:
        return False
    if start is None:
        start = end
    if end is None:
        end = start
    assert start is not None and end is not None
    lower = min(start, end) - radius
    upper = max(start, end) + radius
    return lower <= frame <= upper

def _frame_distance_to_range(frame: int, time_range: Any) -> int:
    if isinstance(time_range, dict):
        start = _optional_int(time_range.get('start_frame'))
        end = _optional_int(time_range.get('end_frame'))
    else:
        start = _optional_int(getattr(time_range, 'start_frame', None))
        end = _optional_int(getattr(time_range, 'end_frame', None))
    if start is None and end is None:
        return 999999
    if start is None:
        start = end
    if end is None:
        end = start
    assert start is not None and end is not None
    low = min(start, end)
    high = max(start, end)
    if low <= frame <= high:
        return 0
    return min(abs(frame - low), abs(frame - high))

def _time_range_payload(time_range: Any) -> dict[str, Any]:
    return {'start_frame': getattr(time_range, 'start_frame', None), 'end_frame': getattr(time_range, 'end_frame', None), 'start_ts': getattr(time_range, 'start_ts', None), 'end_ts': getattr(time_range, 'end_ts', None), 'support_frames': getattr(time_range, 'support_frames', None)}

def _visual_anchor_graph_item_score(item: dict[str, Any], *, anchor: dict[str, Any]) -> float:
    similarity = _optional_float(anchor.get('similarity')) or 0.0
    distance = int(item.get('anchor_distance') or 0)
    score = similarity * 10.0 + max(0.0, 8.0 - distance)
    source_kind = str(item.get('source_kind') or '')
    if source_kind in {'episode', 'transition_event', 'relation_observed_sample'}:
        score += 3.0
    elif source_kind == 'entity':
        score += 2.0
    elif source_kind == 'observation_anchor':
        score += 4.0
    elif source_kind == 'scene':
        score += 1.0
    label = str(anchor.get('label') or '').lower()
    summary = str(item.get('summary') or '').lower()
    if label and label in summary:
        score += 2.0
    return round(score, 6)

def _source_kind_order(source_kind: str) -> int:
    order = {'observation_anchor': -1, 'transition_event': 0, 'episode': 1, 'relation_observed_sample': 2, 'entity': 3, 'scene': 4}
    return order.get(source_kind, 9)

def _visual_anchor_graph_evidence_line(*, index: int, item: dict[str, Any]) -> str:
    anchor = item.get('visual_anchor', {}) if isinstance(item.get('visual_anchor'), dict) else {}
    time_range = item.get('time_range', {}) if isinstance(item.get('time_range'), dict) else {}
    parts = [f'rank={index}', f"kind={item.get('source_kind')}", f"score={item.get('score')}"]
    if item.get('scene_id'):
        parts.append(f"scene={item.get('scene_id')}")
    entity_ids = [str(value) for value in item.get('entity_ids', []) if str(value).strip()]
    if entity_ids:
        parts.append('entities=' + ','.join(entity_ids[:4]))
    time_text = _time_range_text(time_range)
    if time_text:
        parts.append(f'time={time_text}')
    anchor_frame = anchor.get('frame_index')
    parts.append(f"visual_anchor=frame={anchor_frame} label={anchor.get('label')!r} query={anchor.get('query_phrase')!r}")
    return ' '.join(parts) + f" :: {item.get('summary')}"

def _visual_anchor_candidate_facts(graph_items: list[dict[str, Any]], *, evidence_offset: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, item in enumerate(graph_items, start=1):
        anchor = item.get('visual_anchor', {}) if isinstance(item.get('visual_anchor'), dict) else {}
        fact_text = f"Visual FAISS anchored graph evidence {item.get('source_kind')} near frame {anchor.get('frame_index')}: {item.get('summary')}"
        facts.append({'fact_type': 'visual_faiss_anchored_graph_evidence', 'fact_text': fact_text, 'binding_strength': 'visual_anchor_graph_support', 'support_ranks': [evidence_offset + index], 'support_examples': [str(item.get('summary') or '')], 'source': 'visual_faiss_graph_anchor', 'source_kind': item.get('source_kind'), 'scene_ids': [item.get('scene_id')] if item.get('scene_id') else [], 'entity_ids': list(item.get('entity_ids', []) or []), 'time_ranges': [item.get('time_range', {})], 'visual_anchor': anchor, 'score': item.get('score')})
    return facts

def _visual_anchor_soft_contexts(*, anchors: list[dict[str, Any]], graph_items: list[dict[str, Any]], max_contexts: int) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for anchor_rank, anchor in enumerate(anchors[:max_contexts], start=1):
        nearby_items = [item for item in graph_items if int(item.get('visual_anchor_rank') or 0) == anchor_rank][:4]
        if not nearby_items:
            continue
        object_facts = _visual_anchor_object_fact_text(anchor)
        graph_context = _visual_anchor_nearby_graph_text(nearby_items)
        summary_parts = [f"Visual object anchor at frame {anchor.get('frame_index')}", f"label={anchor.get('label')!r}"]
        if object_facts:
            summary_parts.append(f'object_facts={object_facts}')
        if graph_context:
            summary_parts.append(f'nearby_graph={graph_context}')
        summary = '; '.join(summary_parts)
        contexts.append({'source_record_id': f'visual_faiss_soft_graph_context:{anchor_rank}', 'source_kind': 'visual_faiss_soft_graph_context', 'summary': _short_quote(summary, limit=420), 'score': _visual_anchor_soft_context_score(anchor, nearby_items), 'visual_anchor': dict(anchor), 'nearby_graph_evidence': [_concise_visual_anchor_graph_item(item) for item in nearby_items], 'time_range': _visual_anchor_soft_time_range(nearby_items), 'scene_ids': _ordered_unique_texts([str(item.get('scene_id') or '').strip() for item in nearby_items if str(item.get('scene_id') or '').strip()]), 'entity_ids': _ordered_unique_texts([str(entity_id).strip() for item in nearby_items for entity_id in item.get('entity_ids', []) or [] if str(entity_id).strip()])})
    return contexts

def _visual_anchor_soft_context_line(*, index: int, context: dict[str, Any]) -> str:
    anchor = context.get('visual_anchor', {}) if isinstance(context.get('visual_anchor'), dict) else {}
    time_range = context.get('time_range', {}) if isinstance(context.get('time_range'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_soft_graph_context', f"score={context.get('score')}"]
    scene_ids = [str(value).strip() for value in context.get('scene_ids', []) or [] if str(value).strip()]
    if scene_ids:
        parts.append('scenes=' + ','.join(scene_ids[:3]))
    entity_ids = [str(value).strip() for value in context.get('entity_ids', []) or [] if str(value).strip()]
    if entity_ids:
        parts.append('entities=' + ','.join(entity_ids[:4]))
    time_text = _time_range_text(time_range)
    if time_text:
        parts.append(f'time={time_text}')
    parts.append(f"visual_anchor=frame={anchor.get('frame_index')} label={anchor.get('label')!r} query={anchor.get('query_phrase')!r}")
    return ' '.join(parts) + f" :: {context.get('summary')}"

def _visual_anchor_soft_context_candidate_facts(soft_contexts: list[dict[str, Any]], *, evidence_offset: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, context in enumerate(soft_contexts, start=1):
        anchor = context.get('visual_anchor', {}) if isinstance(context.get('visual_anchor'), dict) else {}
        support_examples = [str(anchor.get('object_summary') or '').strip(), *[str(item.get('summary') or '').strip() for item in context.get('nearby_graph_evidence', []) or []]]
        support_examples = [value for value in _ordered_unique_texts(support_examples) if value][:4]
        facts.append({'fact_type': 'visual_faiss_soft_graph_context', 'fact_text': str(context.get('summary') or ''), 'binding_strength': 'visual_anchor_plus_nearby_graph', 'support_ranks': [evidence_offset + index], 'support_examples': support_examples, 'source': 'visual_faiss_graph_anchor', 'source_kind': 'visual_faiss_soft_graph_context', 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'visual_anchor': anchor, 'nearby_graph_evidence': list(context.get('nearby_graph_evidence', []) or []), 'score': context.get('score')})
    return facts

def _visual_anchor_object_fact_text(anchor: dict[str, Any]) -> str:
    facts = [str(value).strip() for value in anchor.get('hsgg_visual_facts', []) or [] if str(value).strip()]
    if facts:
        return ' | '.join((_short_quote(value, limit=120) for value in facts[:4]))
    values = [str(anchor.get('visual_attribute') or '').strip(), *[str(value).strip() for value in anchor.get('visual_relations', []) or [] if str(value).strip()]]
    values = [value for value in values if value]
    return ' | '.join((_short_quote(value, limit=120) for value in values[:4]))

def _visual_anchor_nearby_graph_text(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items[:4]:
        source_kind = str(item.get('source_kind') or 'graph').strip() or 'graph'
        summary = _short_quote(str(item.get('summary') or '').strip(), limit=120)
        if not summary:
            continue
        parts.append(f'{source_kind}: {summary}')
    return ' | '.join(parts)

def _visual_faiss_observation_anchor_index(graph: CanonicalTemporalMemoryGraph) -> dict[tuple[int, int], dict[str, Any]]:
    source_path = getattr(graph, 'source_graph_path', None)
    if source_path is None:
        return {}
    resolved = resolve_repo_input_path(Path(source_path))
    cache_key = str(resolved)
    cached = _OBSERVATION_ANCHOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not resolved.exists():
        _OBSERVATION_ANCHOR_CACHE[cache_key] = {}
        return {}
    anchors = _load_observation_anchor_map_cache(resolved)
    if anchors is None:
        anchors = _build_observation_anchor_map_from_temporal_graph(resolved)
        _write_observation_anchor_map_cache(resolved, anchors)
    _OBSERVATION_ANCHOR_CACHE[cache_key] = anchors
    return anchors

def _observation_anchor_map_cache_path(source_graph_path: Path) -> Path:
    return source_graph_path.with_name(source_graph_path.name + _OBSERVATION_ANCHOR_MAP_SUFFIX)

def _load_observation_anchor_map_cache(source_graph_path: Path) -> dict[tuple[int, int], dict[str, Any]] | None:
    cache_path = _observation_anchor_map_cache_path(source_graph_path)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('contract') != _OBSERVATION_ANCHOR_MAP_CONTRACT:
        return None
    source = payload.get('source_graph', {})
    if not isinstance(source, dict):
        return None
    if str(source.get('path') or '') != str(source_graph_path):
        return None
    try:
        stat = source_graph_path.stat()
    except OSError:
        return None
    if int(source.get('size_bytes') or -1) != int(stat.st_size):
        return None
    cached_mtime = float(source.get('mtime') or -1.0)
    if abs(cached_mtime - float(stat.st_mtime)) > 0.001:
        return None
    rows = payload.get('anchors', [])
    if not isinstance(rows, list):
        return None
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        anchor = _observation_anchor_from_row(row)
        if anchor is None:
            continue
        anchors[int(anchor['frame_index']), int(anchor['object_index'])] = anchor
    return anchors

def _write_observation_anchor_map_cache(source_graph_path: Path, anchors: dict[tuple[int, int], dict[str, Any]]) -> None:
    cache_path = _observation_anchor_map_cache_path(source_graph_path)
    try:
        stat = source_graph_path.stat()
    except OSError:
        return
    rows = sorted(anchors.values(), key=lambda item: (int(item.get('frame_index') or 0), int(item.get('object_index') or 0)))
    payload = {'contract': _OBSERVATION_ANCHOR_MAP_CONTRACT, 'source_graph': {'path': str(source_graph_path), 'size_bytes': int(stat.st_size), 'mtime': float(stat.st_mtime)}, 'anchor_count': len(rows), 'anchors': rows}
    tmp_path = cache_path.with_name(f'{cache_path.name}.{os.getpid()}.tmp')
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

def _build_observation_anchor_map_from_temporal_graph(source_graph_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    try:
        with source_graph_path.open('r', encoding='utf-8') as handle:
            for row in _iter_json_array_objects(handle, '"observations"'):
                anchor = _observation_anchor_from_row(row)
                if anchor is None:
                    continue
                anchors[int(anchor['frame_index']), int(anchor['object_index'])] = anchor
    except (OSError, ValueError):
        return {}
    return anchors

def _iter_json_array_objects(handle: Any, property_name: str) -> Any:
    decoder = json.JSONDecoder()
    buffer = ''
    found_property = False
    in_array = False
    finished = False
    while not finished:
        chunk = handle.read(_OBSERVATION_ANCHOR_STREAM_CHUNK_SIZE)
        if not chunk:
            finished = True
        buffer += chunk
        while True:
            if not found_property:
                index = buffer.find(property_name)
                if index < 0:
                    buffer = buffer[-max(0, len(property_name) - 1):]
                    break
                buffer = buffer[index + len(property_name):]
                found_property = True
            if not in_array:
                index = buffer.find('[')
                if index < 0:
                    if finished:
                        raise ValueError('observations array not found')
                    break
                buffer = buffer[index + 1:]
                in_array = True
            stripped = buffer.lstrip()
            if stripped != buffer:
                buffer = stripped
            if buffer.startswith(','):
                buffer = buffer[1:]
                continue
            if buffer.startswith(']'):
                return
            if not buffer:
                break
            try:
                item, end_index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if finished:
                    raise
                break
            if isinstance(item, dict):
                yield item
            buffer = buffer[end_index:]

def _observation_anchor_from_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    frame = _optional_int(row.get('frame_index'))
    object_index = _optional_int(row.get('object_index'))
    entity_id = str(row.get('entity_id') or '').strip()
    if frame is None or object_index is None or (not entity_id):
        return None
    return {'source_record_id': row.get('observation_id'), 'entity_id': entity_id, 'frame_index': frame, 'frame_rank': _optional_int(row.get('frame_rank')), 'object_index': object_index, 'label': row.get('label'), 'bbox': row.get('bbox'), 'image_path': row.get('image_path'), 'scene_id': row.get('scene_id'), 'timestamp_sec': row.get('timestamp_sec'), 'state_values': list(row.get('state_values', []) or []), 'raw_state_values': list(row.get('raw_state_values', []) or []), 'node_role': row.get('node_role'), 'graph_origin': row.get('graph_origin')}

def _visual_faiss_observation_anchor_graph_item(anchor: dict[str, Any], *, row: dict[str, Any]) -> dict[str, Any]:
    frame = _optional_int(anchor.get('frame_index'))
    timestamp = _optional_float(anchor.get('timestamp_sec'))
    time_range = {'start_frame': frame, 'end_frame': frame, 'start_ts': timestamp, 'end_ts': timestamp, 'support_frames': 1}
    label = _short_text(anchor.get('label') or row.get('label'), limit=80) or 'object'
    state_values = _ordered_unique_texts([*[str(value) for value in anchor.get('state_values', []) or []], *[str(value) for value in anchor.get('raw_state_values', []) or []]])[:4]
    summary_parts = [f"FAISS hit maps to temporal observation entity={anchor.get('entity_id')}", f'label={label!r}', f'frame={frame}', f"obj={anchor.get('object_index')}"]
    if state_values:
        summary_parts.append('states=' + ' | '.join(state_values))
    bbox_text = _bbox_spatial_text(_bbox_values(anchor.get('bbox')))
    if bbox_text:
        summary_parts.append(bbox_text)
    return {'source_record_id': anchor.get('source_record_id'), 'source_kind': 'observation_anchor', 'summary': _short_quote('; '.join(summary_parts), limit=220), 'time_range': time_range, 'scene_id': anchor.get('scene_id'), 'entity_ids': [str(anchor.get('entity_id') or '').strip()], 'metadata': {'contract': 'visual_faiss_temporal_observation_anchor_v1', 'frame_index': frame, 'frame_rank': anchor.get('frame_rank'), 'object_index': anchor.get('object_index'), 'label': label, 'bbox': anchor.get('bbox'), 'image_path': anchor.get('image_path'), 'node_role': anchor.get('node_role'), 'graph_origin': anchor.get('graph_origin'), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}}

def _concise_visual_anchor_graph_item(item: dict[str, Any]) -> dict[str, Any]:
    return {'source_record_id': item.get('source_record_id'), 'source_kind': item.get('source_kind'), 'summary': item.get('summary'), 'scene_id': item.get('scene_id'), 'entity_ids': list(item.get('entity_ids', []) or []), 'time_range': item.get('time_range', {}), 'score': item.get('score'), 'anchor_distance': item.get('anchor_distance'), 'metadata': item.get('metadata', {})}

def _visual_anchor_soft_context_score(anchor: dict[str, Any], items: list[dict[str, Any]]) -> float:
    similarity = _optional_float(anchor.get('similarity')) or 0.0
    graph_score = max((_optional_float(item.get('score')) or 0.0 for item in items), default=0.0)
    fact_bonus = 0.0
    if anchor.get('hsgg_visual_facts') or anchor.get('visual_attribute'):
        fact_bonus += 1.0
    if anchor.get('visual_relations') or anchor.get('hsgg_neighbor_attributes'):
        fact_bonus += 0.5
    return round(similarity * 10.0 + graph_score + fact_bonus, 6)

def _visual_anchor_soft_time_range(items: list[dict[str, Any]]) -> dict[str, Any]:
    starts: list[int] = []
    ends: list[int] = []
    start_ts_values: list[float] = []
    end_ts_values: list[float] = []
    support_frames = 0
    for item in items:
        time_range = item.get('time_range', {})
        if not isinstance(time_range, dict):
            continue
        start = _optional_int(time_range.get('start_frame'))
        end = _optional_int(time_range.get('end_frame'))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
        start_ts = _optional_float(time_range.get('start_ts'))
        end_ts = _optional_float(time_range.get('end_ts'))
        if start_ts is not None:
            start_ts_values.append(start_ts)
        if end_ts is not None:
            end_ts_values.append(end_ts)
        frame_count = _optional_int(time_range.get('support_frames')) or 0
        support_frames += max(0, frame_count)
    return {'start_frame': min(starts) if starts else None, 'end_frame': max(ends) if ends else None, 'start_ts': min(start_ts_values) if start_ts_values else None, 'end_ts': max(end_ts_values) if end_ts_values else None, 'support_frames': support_frames or None}

def _visual_faiss_ger_object_rows(visual_payload: dict[str, Any], *, object_limit: int, role_aware_compression_enabled: bool=False, question: str='', question_ir: dict[str, Any] | None=None) -> list[tuple[str, dict[str, Any]]]:
    payload = dict(visual_payload)
    compression = dict(payload.get('compression', {}) or {})
    compression['selected_object_limit'] = max(1, object_limit)
    if role_aware_compression_enabled:
        compression['role_aware_compression_enabled'] = True
        compression['contract'] = 'object_faiss_hsgg_fact_role_aware_compression_v1'
        if question:
            payload['question'] = str(question)
        payload['question_ir'] = question_ir or {}
    payload['compression'] = compression
    return _compressed_visual_faiss_rows(payload)

def _visual_faiss_ger_contexts(*, graph: CanonicalTemporalMemoryGraph, object_rows: list[tuple[str, dict[str, Any]]], context_radius: int, context_limit: int, observation_anchor_index: dict[tuple[int, int], dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for index, (phrase, row) in enumerate(object_rows, start=1):
        frame = _optional_int(row.get('graph_frame_index'))
        object_index = _optional_int(row.get('object_index'))
        nearby_items: list[dict[str, Any]] = []
        if frame is not None:
            nearby_items = _graph_items_touching_frame(graph, frame=frame, radius=context_radius)
            observation_anchor = observation_anchor_index.get((frame, object_index)) if observation_anchor_index is not None and object_index is not None else None
            if observation_anchor is not None:
                nearby_items.insert(0, _visual_faiss_observation_anchor_graph_item(observation_anchor, row=row))
            for item in nearby_items:
                item['anchor_distance'] = _frame_distance_to_range(frame, item.get('time_range', {}))
                item['score'] = _visual_faiss_ger_graph_item_score(item, row=row)
            nearby_items.sort(key=lambda item: (_optional_int(item.get('anchor_distance')) or 0, _source_kind_order(str(item.get('source_kind') or '')), -float(item.get('score') or 0.0), str(item.get('source_record_id') or '')))
            nearby_items = nearby_items[:context_limit]
        concise_graph_items = [_concise_visual_anchor_graph_item(item) for item in nearby_items]
        object_facts = _visual_faiss_fact_summary(row, max_facts=6)
        nearby_graph = _visual_anchor_nearby_graph_text(nearby_items)
        summary_parts = ['Visual FAISS object retrieved as GER candidate', f"label={row.get('label')!r}"]
        if frame is not None:
            summary_parts.append(f'frame={frame}')
        if phrase:
            summary_parts.append(f'query={phrase!r}')
        if object_facts:
            summary_parts.append(f'object_facts={object_facts}')
        if nearby_graph:
            summary_parts.append(f'kg_context={nearby_graph}')
        context = {'source_record_id': f'visual_faiss_ger:{index}', 'source_kind': 'visual_faiss_ger_object_context', 'summary': _short_quote('; '.join(summary_parts), limit=640), 'score': _visual_faiss_ger_context_score(row, nearby_items), 'query_phrase': phrase, 'visual_object': _concise_visual_faiss_ger_object(row), 'nearby_graph_evidence': concise_graph_items, 'time_range': _visual_anchor_soft_time_range(nearby_items), 'scene_ids': _ordered_unique_texts([str(item.get('scene_id') or '').strip() for item in nearby_items if str(item.get('scene_id') or '').strip()]), 'entity_ids': _ordered_unique_texts([str(entity_id).strip() for item in nearby_items for entity_id in item.get('entity_ids', []) or [] if str(entity_id).strip()]), 'hsgg_visual_facts': list(row.get('hsgg_visual_facts', []) or []), 'hsgg_neighbor_attributes': list(row.get('hsgg_neighbor_attributes', []) or []), 'hsgg_frame_context': list(row.get('hsgg_frame_context', []) or [])}
        contexts.append(context)
    return contexts

def _visual_faiss_ger_graph_item_score(item: dict[str, Any], *, row: dict[str, Any]) -> float:
    similarity = _optional_float(row.get('similarity')) or 0.0
    distance = int(item.get('anchor_distance') or 0)
    score = similarity * 10.0 + max(0.0, 8.0 - distance)
    source_kind = str(item.get('source_kind') or '')
    if source_kind in {'episode', 'transition_event', 'relation_observed_sample'}:
        score += 3.0
    elif source_kind == 'entity':
        score += 2.0
    elif source_kind == 'observation_anchor':
        score += 4.0
    elif source_kind == 'scene':
        score += 1.0
    label = str(row.get('label') or '').lower()
    summary = str(item.get('summary') or '').lower()
    if label and label in summary:
        score += 2.0
    return round(score, 6)

def _visual_faiss_ger_context_score(row: dict[str, Any], nearby_items: list[dict[str, Any]]) -> float:
    similarity = _optional_float(row.get('similarity')) or 0.0
    graph_score = max((_optional_float(item.get('score')) or 0.0 for item in nearby_items), default=0.0)
    fact_bonus = 0.0
    if row.get('hsgg_visual_facts') or row.get('visual_attribute'):
        fact_bonus += 1.0
    if row.get('visual_relations') or row.get('hsgg_neighbor_attributes'):
        fact_bonus += 0.5
    return round(similarity * 10.0 + graph_score + fact_bonus, 6)

def _visual_faiss_ger_context_line(*, index: int, context: dict[str, Any]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    time_range = context.get('time_range', {}) if isinstance(context.get('time_range'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_ger_object_context', f"score={context.get('score')}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    label = str(visual_object.get('label') or '').strip()
    if label:
        parts.append(f'label={label!r}')
    scene_ids = [str(value).strip() for value in context.get('scene_ids', []) or [] if str(value).strip()]
    if scene_ids:
        parts.append('scenes=' + ','.join(scene_ids[:3]))
    entity_ids = [str(value).strip() for value in context.get('entity_ids', []) or [] if str(value).strip()]
    if entity_ids:
        parts.append('entities=' + ','.join(entity_ids[:4]))
    time_text = _time_range_text(time_range)
    if time_text:
        parts.append(f'time={time_text}')
    query_phrase = str(context.get('query_phrase') or '').strip()
    if query_phrase:
        parts.append(f'query={query_phrase!r}')
    return ' '.join(parts) + f" :: {context.get('summary')}"

def _visual_faiss_ger_candidate_facts(contexts: list[dict[str, Any]], *, support_rank_offset: int=0) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        support_examples = [str(visual_object.get('object_summary') or '').strip(), *[str(value).strip() for value in context.get('hsgg_visual_facts', []) or [] if str(value).strip()], *[str(item.get('summary') or '').strip() for item in context.get('nearby_graph_evidence', []) or [] if str(item.get('summary') or '').strip()]]
        support_examples = [value for value in _ordered_unique_texts(support_examples) if value][:6]
        fact_text = _visual_faiss_ger_answerable_fact_text(context)
        facts.append({'fact_type': 'visual_faiss_ger_object_context', 'fact_text': fact_text, 'binding_strength': 'visual_similarity_plus_kg_context', 'support_ranks': [support_rank_offset + index], 'support_examples': support_examples, 'source': 'visual_faiss_ger_fusion', 'source_kind': 'visual_faiss_ger_object_context', 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'nearby_graph_evidence': list(context.get('nearby_graph_evidence', []) or []), 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'hsgg_visual_facts': list(context.get('hsgg_visual_facts', []) or []), 'hsgg_neighbor_attributes': list(context.get('hsgg_neighbor_attributes', []) or []), 'hsgg_frame_context': list(context.get('hsgg_frame_context', []) or []), 'score': context.get('score')})
    return facts

def _balanced_graph_visual_lines(*, graph_lines: list[str], visual_channels: list[list[str]], head_size: int=6, graph_quota: int=3, visual_quota: int=3) -> list[str]:
    """Preserve graph evidence while adding visual FAISS evidence generically."""
    graph = _ordered_unique_texts([str(line).strip() for line in graph_lines if str(line).strip()])
    visual = _round_robin_unique_strings(visual_channels)
    head: list[str] = []
    head.extend(graph[:max(0, graph_quota)])
    head.extend(visual[:max(0, visual_quota)])
    if len(head) < max(1, head_size):
        head.extend(graph[max(0, graph_quota):])
    if len(head) < max(1, head_size):
        head.extend(visual[max(0, visual_quota):])
    return _ordered_unique_texts([*head[:max(1, head_size)], *graph, *visual])

def _balanced_graph_visual_items(*, graph_items: list[dict[str, Any]], visual_channels: list[list[dict[str, Any]]], head_size: int=6, graph_quota: int=3, visual_quota: int=3) -> list[dict[str, Any]]:
    graph = [item for item in graph_items if isinstance(item, dict)]
    visual = _round_robin_unique_items(visual_channels)
    head: list[dict[str, Any]] = []
    head.extend(graph[:max(0, graph_quota)])
    head.extend(visual[:max(0, visual_quota)])
    if len(head) < max(1, head_size):
        head.extend(graph[max(0, graph_quota):])
    if len(head) < max(1, head_size):
        head.extend(visual[max(0, visual_quota):])
    return _ordered_unique_fact_items([*head[:max(1, head_size)], *graph, *visual])

def _round_robin_unique_strings(channels: list[list[str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    max_len = max((len(channel) for channel in channels), default=0)
    for depth in range(max_len):
        for channel in channels:
            if depth >= len(channel):
                continue
            text = str(channel[depth]).strip()
            if not text or text in seen:
                continue
            ordered.append(text)
            seen.add(text)
    return ordered

def _round_robin_unique_items(channels: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_len = max((len(channel) for channel in channels), default=0)
    for depth in range(max_len):
        for channel in channels:
            if depth >= len(channel):
                continue
            item = channel[depth]
            if not isinstance(item, dict):
                continue
            signature = _fact_item_signature(item)
            if signature in seen:
                continue
            ordered.append(item)
            seen.add(signature)
    return ordered

def _ordered_unique_fact_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        signature = _fact_item_signature(item)
        if signature in seen:
            continue
        ordered.append(item)
        seen.add(signature)
    return ordered

def _fact_item_signature(item: dict[str, Any]) -> str:
    return '|'.join([str(item.get('source_record_id') or ''), str(item.get('source_kind') or ''), str(item.get('fact_type') or ''), str(item.get('fact_text') or '')])

def _sync_visual_fact_support_ranks(*, evidence_lines: list[str], fact_line_pairs: Any) -> None:
    rank_by_line = {str(line).strip(): index for index, line in enumerate(evidence_lines, start=1) if str(line).strip()}
    for fact, line in fact_line_pairs:
        if not isinstance(fact, dict):
            continue
        rank = rank_by_line.get(str(line).strip())
        if rank is not None:
            fact['support_ranks'] = [rank]

def _visual_faiss_unified_candidates(contexts: list[dict[str, Any]], *, question: str, question_ir: dict[str, Any]) -> list[dict[str, Any]]:
    query_terms = _question_terms(question=question, question_ir=question_ir)
    constraint_terms = _visual_faiss_ger_constraint_terms(question_ir)
    query_token_set = set(query_terms)
    constraint_token_set = set(constraint_terms)
    candidates: list[dict[str, Any]] = []
    for index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        object_text = _visual_faiss_ger_object_text_for_binding(context)
        graph_text = _visual_faiss_ger_graph_text_for_binding(context)
        object_tokens = _visual_faiss_text_tokens(object_text)
        graph_tokens = _visual_faiss_text_tokens(graph_text)
        matched_object_terms = sorted(query_token_set & object_tokens)
        matched_graph_terms = sorted(query_token_set & graph_tokens)
        matched_constraint_terms = sorted(constraint_token_set & (object_tokens | graph_tokens))
        attribute_fragments = _visual_faiss_ger_attribute_fragments(context)
        relation_fragments = _visual_faiss_ger_relation_fragments(context)
        graph_fragments = _visual_faiss_ger_graph_fragments(context)
        graph_support = _visual_faiss_unified_graph_support(context, query_terms=query_terms, constraint_terms=constraint_terms)
        binding_checks = _visual_faiss_unified_binding_checks(matched_object_terms=matched_object_terms, matched_graph_terms=matched_graph_terms, matched_constraint_terms=matched_constraint_terms, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, graph_support=graph_support)
        score = _visual_faiss_unified_candidate_score(context, matched_object_terms=matched_object_terms, matched_graph_terms=matched_graph_terms, matched_constraint_terms=matched_constraint_terms, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, graph_support=graph_support, binding_status=str(binding_checks.get('binding_status') or ''))
        if score <= 0:
            continue
        summary = _visual_faiss_unified_candidate_summary(context=context, binding_checks=binding_checks, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, graph_fragments=graph_fragments)
        candidate = {'source_record_id': f'visual_faiss_unified:{index}', 'source_kind': 'visual_faiss_unified_candidate', 'summary': summary, 'score': score, 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'graph_support': graph_support, 'nearby_graph_evidence': list(context.get('nearby_graph_evidence', []) or []), 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'matched_object_terms': matched_object_terms, 'matched_graph_terms': matched_graph_terms, 'matched_constraint_terms': matched_constraint_terms, 'attribute_fragments': attribute_fragments, 'relation_fragments': relation_fragments, 'graph_fragments': graph_fragments, 'binding_checks': binding_checks, 'binding_status': binding_checks.get('binding_status'), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: (_visual_faiss_unified_binding_order(str(candidate.get('binding_status') or '')), -float(candidate.get('score') or 0.0), -len(candidate.get('matched_constraint_terms', []) or []), -len(candidate.get('matched_object_terms', []) or []), _optional_int((candidate.get('visual_object', {}) if isinstance(candidate.get('visual_object'), dict) else {}).get('graph_frame_index')) or 0, str((candidate.get('visual_object', {}) if isinstance(candidate.get('visual_object'), dict) else {}).get('label') or '')))
    for rank, candidate in enumerate(candidates, start=1):
        candidate['candidate_rank'] = rank
        candidate['support_ranks'] = [rank]
    return candidates

def _visual_faiss_unified_graph_support(context: dict[str, Any], *, query_terms: list[str], constraint_terms: list[str]) -> dict[str, Any]:
    query_token_set = set(query_terms)
    constraint_token_set = set(constraint_terms)
    supported_items: list[dict[str, Any]] = []
    for item in context.get('nearby_graph_evidence', []) or []:
        if not isinstance(item, dict):
            continue
        summary = str(item.get('summary') or '')
        tokens = _visual_faiss_text_tokens(summary)
        query_hits = sorted(query_token_set & tokens)
        constraint_hits = sorted(constraint_token_set & tokens)
        if not query_hits and (not constraint_hits):
            continue
        supported_items.append({'source_record_id': item.get('source_record_id'), 'source_kind': item.get('source_kind'), 'summary': item.get('summary'), 'scene_id': item.get('scene_id'), 'entity_ids': list(item.get('entity_ids', []) or []), 'time_range': item.get('time_range', {}), 'score': item.get('score'), 'query_hits': query_hits, 'constraint_hits': constraint_hits})
    supported_items.sort(key=lambda item: (-len(item.get('constraint_hits', []) or []), -len(item.get('query_hits', []) or []), _source_kind_order(str(item.get('source_kind') or '')), -float(item.get('score') or 0.0)))
    return {'available': bool(supported_items), 'support_count': len(supported_items), 'items': supported_items[:4]}

def _visual_faiss_unified_binding_checks(*, matched_object_terms: list[str], matched_graph_terms: list[str], matched_constraint_terms: list[str], attribute_fragments: list[str], relation_fragments: list[str], graph_support: dict[str, Any]) -> dict[str, Any]:
    has_object_target = bool(matched_object_terms)
    has_graph_target = bool(matched_graph_terms)
    has_constraints = bool(matched_constraint_terms)
    has_attribute = bool(attribute_fragments)
    has_relation = bool(relation_fragments)
    has_graph_support = bool(graph_support.get('available'))
    if has_object_target and has_constraints and has_graph_support and (has_attribute or has_relation):
        status = 'strong_object_graph_bound'
    elif has_object_target and (has_attribute or has_relation) and (has_constraints or has_graph_support):
        status = 'medium_object_bound'
    elif has_object_target or (has_graph_target and has_graph_support):
        status = 'weak_context_bound'
    else:
        status = 'visual_similarity_only'
    if has_graph_support and (not has_object_target) and (not has_constraints) and (has_attribute or has_relation):
        status = 'possible_context_conflict'
    return {'binding_status': status, 'object_target_match': has_object_target, 'graph_target_match': has_graph_target, 'constraint_match': has_constraints, 'attribute_bound_to_object': has_attribute, 'relation_bound_to_object': has_relation, 'graph_support': has_graph_support}

def _visual_faiss_unified_candidate_score(context: dict[str, Any], *, matched_object_terms: list[str], matched_graph_terms: list[str], matched_constraint_terms: list[str], attribute_fragments: list[str], relation_fragments: list[str], graph_support: dict[str, Any], binding_status: str) -> float:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    similarity = _optional_float(visual_object.get('similarity')) or 0.0
    score = similarity * 4.0
    score += len(matched_object_terms) * 1.8
    score += len(matched_graph_terms) * 0.8
    score += len(matched_constraint_terms) * 2.0
    if attribute_fragments:
        score += 1.6
    if relation_fragments:
        score += 1.2
    if graph_support.get('available'):
        score += min(3.0, 1.0 + 0.5 * int(graph_support.get('support_count') or 0))
    score += {'strong_object_graph_bound': 3.0, 'medium_object_bound': 1.5, 'weak_context_bound': 0.5, 'visual_similarity_only': 0.0, 'possible_context_conflict': -1.0}.get(binding_status, 0.0)
    return round(score, 6)

def _visual_faiss_unified_binding_order(binding_status: str) -> int:
    return {'strong_object_graph_bound': 0, 'medium_object_bound': 1, 'weak_context_bound': 2, 'visual_similarity_only': 3, 'possible_context_conflict': 4}.get(binding_status, 9)

def _visual_faiss_unified_candidate_summary(*, context: dict[str, Any], binding_checks: dict[str, Any], attribute_fragments: list[str], relation_fragments: list[str], graph_fragments: list[str]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    pieces = ['Unified graph+FAISS candidate', f"binding={binding_checks.get('binding_status')}", f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', 'object=' + _short_text(visual_object.get('object_summary'), limit=120) if visual_object.get('object_summary') else '', 'attributes=' + ' | '.join((_short_text(value, limit=120) for value in attribute_fragments[:3])) if attribute_fragments else '', 'relations=' + ' | '.join((_short_text(value, limit=110) for value in relation_fragments[:3])) if relation_fragments else '', 'graph_support=' + ' | '.join((_short_text(value, limit=110) for value in graph_fragments[:3])) if graph_fragments else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_unified_candidate_line(*, index: int, candidate: dict[str, Any]) -> str:
    visual_object = candidate.get('visual_object', {}) if isinstance(candidate.get('visual_object'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_unified_candidate', f"score={candidate.get('score')}", f"binding={candidate.get('binding_status')}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    label = str(visual_object.get('label') or '').strip()
    if label:
        parts.append(f'label={label!r}')
    matched_object_terms = [str(value).strip() for value in candidate.get('matched_object_terms', []) or [] if str(value).strip()]
    if matched_object_terms:
        parts.append('object_terms=' + ','.join(matched_object_terms[:8]))
    matched_constraint_terms = [str(value).strip() for value in candidate.get('matched_constraint_terms', []) or [] if str(value).strip()]
    if matched_constraint_terms:
        parts.append('constraint_terms=' + ','.join(matched_constraint_terms[:8]))
    entity_ids = [str(value).strip() for value in candidate.get('entity_ids', []) or [] if str(value).strip()]
    if entity_ids:
        parts.append('entities=' + ','.join(entity_ids[:4]))
    return ' '.join(parts) + f" :: {candidate.get('summary')}"

def _visual_faiss_unified_candidate_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        support_examples = _ordered_unique_texts([str(candidate.get('summary') or ''), *[str(value).strip() for value in candidate.get('attribute_fragments', []) or [] if str(value).strip()], *[str(value).strip() for value in candidate.get('relation_fragments', []) or [] if str(value).strip()], *[str(value).strip() for value in candidate.get('graph_fragments', []) or [] if str(value).strip()]])[:6]
        facts.append({'fact_type': 'visual_faiss_unified_candidate', 'fact_text': candidate.get('summary'), 'binding_strength': candidate.get('binding_status'), 'support_ranks': [index], 'support_examples': support_examples, 'source_record_id': candidate.get('source_record_id'), 'source': 'visual_faiss_unified_candidates', 'source_kind': 'visual_faiss_unified_candidate', 'query_phrase': candidate.get('query_phrase'), 'visual_object': candidate.get('visual_object'), 'graph_support': candidate.get('graph_support'), 'binding_checks': candidate.get('binding_checks'), 'nearby_graph_evidence': list(candidate.get('nearby_graph_evidence', []) or []), 'scene_ids': list(candidate.get('scene_ids', []) or []), 'entity_ids': list(candidate.get('entity_ids', []) or []), 'time_ranges': list(candidate.get('time_ranges', []) or []), 'matched_object_terms': list(candidate.get('matched_object_terms', []) or []), 'matched_graph_terms': list(candidate.get('matched_graph_terms', []) or []), 'matched_constraint_terms': list(candidate.get('matched_constraint_terms', []) or []), 'score': candidate.get('score'), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    return facts

def _visual_faiss_unified_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'group=visual_faiss_unified:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"label={visual_object.get('label')!r}"]
    matched_object_terms = [str(value).strip() for value in fact.get('matched_object_terms', []) or [] if str(value).strip()]
    if matched_object_terms:
        parts.append('object_terms=' + ','.join(matched_object_terms[:8]))
    matched_constraint_terms = [str(value).strip() for value in fact.get('matched_constraint_terms', []) or [] if str(value).strip()]
    if matched_constraint_terms:
        parts.append('constraint_terms=' + ','.join(matched_constraint_terms[:8]))
    parts.append(str(fact.get('fact_text') or ''))
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_object_local_facts(contexts: list[dict[str, Any]], *, question: str, question_ir: dict[str, Any], limit: int, exclude_schema_terms: bool=False) -> list[dict[str, Any]]:
    """Build short object-local facts from FAISS-recalled HSGG objects.

    This layer keeps the graph immutable and only repackages direct object
    metadata: the recalled object's label, bbox/frame, direct attributes,
    direct relations, same-frame anchor context, and directly attached neighbor
    attributes. It avoids broad nearby graph context so the final QA prompt has
    less room to transfer attributes between unrelated objects.
    """
    query_terms = _question_terms(question=question, question_ir=question_ir)
    constraint_terms = _visual_faiss_ger_constraint_terms(question_ir)
    if exclude_schema_terms:
        query_terms = _without_schema_match_terms(query_terms)
        constraint_terms = _without_schema_match_terms(constraint_terms)
    query_tokens = set(query_terms)
    constraint_tokens = set(constraint_terms)
    facts: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        object_text = _visual_faiss_object_local_text(context)
        object_tokens = _visual_faiss_text_tokens(object_text)
        matched_question_terms = sorted(query_tokens & object_tokens)
        matched_constraint_terms = sorted(constraint_tokens & object_tokens)
        attribute_fragments = _visual_faiss_ger_attribute_fragments(context)
        relation_fragments = _visual_faiss_ger_relation_fragments(context)
        anchor_fragments = _visual_faiss_object_local_anchor_fragments(context)
        neighbor_fragments = _visual_faiss_object_local_neighbor_fragments(context)
        if not (attribute_fragments or relation_fragments or anchor_fragments or neighbor_fragments):
            continue
        checks = _visual_faiss_object_local_checks(attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, anchor_fragments=anchor_fragments, neighbor_fragments=neighbor_fragments, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms)
        score = _visual_faiss_object_local_score(context, checks=checks, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms)
        if score <= 0:
            continue
        fact_text = _visual_faiss_object_local_fact_text(context=context, checks=checks, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, anchor_fragments=anchor_fragments, neighbor_fragments=neighbor_fragments)
        support_examples = _ordered_unique_texts([str(visual_object.get('object_summary') or ''), *anchor_fragments, *attribute_fragments, *relation_fragments, *neighbor_fragments])[:6]
        facts.append({'fact_type': 'visual_faiss_object_local_fact', 'fact_text': fact_text, 'binding_strength': _visual_faiss_object_local_strength(checks), 'support_ranks': [context_index], 'support_context_ranks': [context_index], 'support_examples': support_examples, 'source_record_id': f'visual_faiss_object_local:{context_index}', 'source': 'visual_faiss_object_local_facts', 'source_kind': 'visual_faiss_object_local_fact', 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'matched_question_terms': matched_question_terms, 'matched_constraint_terms': matched_constraint_terms, 'attribute_fragments': attribute_fragments, 'relation_fragments': relation_fragments, 'anchor_fragments': anchor_fragments, 'neighbor_fragments': neighbor_fragments, 'object_local_checks': checks, 'score': score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    facts.sort(key=_visual_faiss_object_local_sort_key)
    return facts[:max(1, int(limit))]

def _visual_faiss_object_local_text(context: dict[str, Any]) -> str:
    values = [_visual_faiss_ger_anchor_text_for_binding(context), *(_visual_faiss_ger_attribute_fragments(context) or []), *(_visual_faiss_ger_relation_fragments(context) or []), *(_visual_faiss_object_local_anchor_fragments(context) or []), *(_visual_faiss_object_local_neighbor_fragments(context) or [])]
    return ' '.join((str(value).strip() for value in values if str(value).strip()))

def _visual_faiss_object_local_anchor_fragments(context: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    object_index = _optional_int(visual_object.get('object_index'))
    for item in context.get('hsgg_frame_context', []) or []:
        text = _short_text(item, limit=150)
        if not text:
            continue
        if 'same_frame_anchor' in text:
            fragments.append(text)
            continue
        if object_index is not None and f'obj={object_index}' in text:
            fragments.append(text)
    return _ordered_unique_texts(fragments)[:3]

def _visual_faiss_object_local_neighbor_fragments(context: dict[str, Any]) -> list[str]:
    fragments = [_short_text(item, limit=140) for item in context.get('hsgg_neighbor_attributes', []) or [] if _short_text(item, limit=140)]
    return _ordered_unique_texts(fragments)[:4]

def _visual_faiss_object_local_checks(*, attribute_fragments: list[str], relation_fragments: list[str], anchor_fragments: list[str], neighbor_fragments: list[str], matched_question_terms: list[str], matched_constraint_terms: list[str]) -> dict[str, Any]:
    return {'attribute_count': len(attribute_fragments), 'relation_count': len(relation_fragments), 'anchor_count': len(anchor_fragments), 'neighbor_count': len(neighbor_fragments), 'matched_question_term_count': len(matched_question_terms), 'matched_constraint_term_count': len(matched_constraint_terms), 'has_direct_attribute': bool(attribute_fragments), 'has_direct_relation': bool(relation_fragments), 'has_anchor_bbox': bool(anchor_fragments), 'has_neighbor_attribute': bool(neighbor_fragments)}

def _visual_faiss_object_local_score(context: dict[str, Any], *, checks: dict[str, Any], matched_question_terms: list[str], matched_constraint_terms: list[str]) -> float:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    similarity = _optional_float(visual_object.get('similarity')) or 0.0
    score = similarity * 4.0
    score += min(3.0, int(checks.get('attribute_count') or 0) * 0.9)
    score += min(2.4, int(checks.get('relation_count') or 0) * 0.8)
    score += min(1.2, int(checks.get('anchor_count') or 0) * 0.6)
    score += min(1.2, int(checks.get('neighbor_count') or 0) * 0.3)
    score += min(2.4, len(matched_question_terms) * 0.3)
    score += min(2.4, len(matched_constraint_terms) * 0.45)
    if checks.get('has_direct_attribute') and checks.get('has_direct_relation'):
        score += 1.0
    if checks.get('has_anchor_bbox'):
        score += 0.5
    return round(score, 6)

def _visual_faiss_object_local_strength(checks: dict[str, Any]) -> str:
    if checks.get('has_direct_attribute') and checks.get('has_direct_relation') and checks.get('matched_constraint_term_count'):
        return 'object_local_attribute_relation_constraint_bound'
    if checks.get('has_direct_attribute') and checks.get('has_direct_relation'):
        return 'object_local_attribute_relation_bound'
    if checks.get('has_direct_attribute'):
        return 'object_local_attribute_bound'
    if checks.get('has_direct_relation'):
        return 'object_local_relation_bound'
    return 'object_local_visual_context_bound'

def _visual_faiss_object_local_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    checks = fact.get('object_local_checks', {}) if isinstance(fact.get('object_local_checks'), dict) else {}
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-int(bool(checks.get('has_direct_attribute'))), -int(bool(checks.get('has_direct_relation'))), -int(checks.get('matched_constraint_term_count') or 0), -int(checks.get('matched_question_term_count') or 0), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, str(visual_object.get('label') or ''))

def _visual_faiss_object_local_fact_text(*, context: dict[str, Any], checks: dict[str, Any], matched_question_terms: list[str], matched_constraint_terms: list[str], attribute_fragments: list[str], relation_fragments: list[str], anchor_fragments: list[str], neighbor_fragments: list[str]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    time_range = context.get('time_range', {}) if isinstance(context.get('time_range'), dict) else {}
    object_summary = _short_text(visual_object.get('object_summary'), limit=120)
    if not object_summary:
        label = str(visual_object.get('label') or '').strip()
        frame = visual_object.get('graph_frame_index')
        object_index = visual_object.get('object_index')
        object_summary = f'frame={frame} obj={object_index} label={label!r}'
    pieces = ['object-local visual fact', f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', f'time={_time_range_text(time_range)}' if _time_range_text(time_range) else '', f'object={object_summary}', f'binding={_visual_faiss_object_local_strength(checks)}', 'matched_question_terms=' + ','.join(matched_question_terms[:10]) if matched_question_terms else '', 'matched_constraint_terms=' + ','.join(matched_constraint_terms[:10]) if matched_constraint_terms else '', 'direct_attributes=' + ' | '.join((_short_text(value, limit=105) for value in attribute_fragments[:3])) if attribute_fragments else '', 'direct_relations=' + ' | '.join((_short_text(value, limit=100) for value in relation_fragments[:3])) if relation_fragments else '', 'anchor=' + ' | '.join((_short_text(value, limit=100) for value in anchor_fragments[:2])) if anchor_fragments else '', 'direct_neighbors=' + ' | '.join((_short_text(value, limit=100) for value in neighbor_fragments[:3])) if neighbor_fragments else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_object_local_fact_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_object_local_fact', f"score={fact.get('score')}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    object_index = visual_object.get('object_index')
    if object_index is not None:
        parts.append(f'obj={object_index}')
    label = str(visual_object.get('label') or '').strip()
    if label:
        parts.append(f'label={label!r}')
    strength = str(fact.get('binding_strength') or '').strip()
    if strength:
        parts.append(f'binding={strength}')
    matched_terms = [str(value).strip() for value in fact.get('matched_question_terms', []) or [] if str(value).strip()]
    if matched_terms:
        parts.append('q_terms=' + ','.join(matched_terms[:8]))
    return ' '.join(parts) + f" :: {fact.get('fact_text')}"

def _visual_faiss_object_local_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'group=visual_faiss_object_local:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"obj={visual_object.get('object_index')}", f"label={visual_object.get('label')!r}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_local_subgraph_facts(contexts: list[dict[str, Any]], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], limit: int, anchor_only: bool=False) -> list[dict[str, Any]]:
    """Build entity-centered local-subgraph facts for CG6-style retrieval.

    FAISS supplies visual anchors, then the canonical graph supplies entity,
    relation, state, time, and neighbor context. The unit ranked here is a
    candidate entity subgraph rather than an isolated text evidence line.
    """
    query_tokens = set(_question_terms(question=question, question_ir=question_ir))
    entity_index = {entity.entity_id: entity for entity in graph.entity_nodes}
    facts: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for context_index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        candidate_entity_ids = _visual_faiss_context_candidate_entity_ids(context, anchor_only=anchor_only)
        for entity_id in candidate_entity_ids:
            if entity_id in seen_entities:
                continue
            entity = entity_index.get(entity_id)
            if entity is None:
                continue
            seen_entities.add(entity_id)
            subgraph = _visual_faiss_entity_local_subgraph(graph=graph, entity=entity, anchor_frame=_optional_int(visual_object.get('graph_frame_index')), neighbor_limit=max(1, min(12, int(limit) + 2)))
            fact = _visual_faiss_entity_local_subgraph_fact(context=context, context_index=context_index, entity=entity, subgraph=subgraph, query_tokens=query_tokens)
            if fact:
                facts.append(fact)
            if len(facts) >= max(1, int(limit)) * 3:
                break
        if len(facts) >= max(1, int(limit)) * 3:
            break
    facts.sort(key=_visual_faiss_local_subgraph_sort_key)
    return facts[:max(1, int(limit))]

def _visual_faiss_context_candidate_entity_ids(context: dict[str, Any], *, anchor_only: bool=False) -> list[str]:
    ids: list[str] = []
    for item in context.get('nearby_graph_evidence', []) or []:
        if not isinstance(item, dict):
            continue
        source_kind = str(item.get('source_kind') or '').strip()
        metadata = item.get('metadata', {})
        is_observation_anchor = source_kind == 'observation_anchor'
        if isinstance(metadata, dict):
            is_observation_anchor = is_observation_anchor or metadata.get('contract') == 'visual_faiss_temporal_observation_anchor_v1'
        if anchor_only and (not is_observation_anchor):
            continue
        for value in item.get('entity_ids', []) or []:
            text = str(value).strip()
            if text:
                ids.append(text)
        if isinstance(metadata, dict):
            text = str(metadata.get('entity_id') or '').strip()
            if text:
                ids.append(text)
    if not anchor_only:
        for value in context.get('entity_ids', []) or []:
            text = str(value).strip()
            if text:
                ids.append(text)
    return _ordered_unique_texts(ids)

def _visual_faiss_entity_local_subgraph(*, graph: CanonicalTemporalMemoryGraph, entity: Any, anchor_frame: int | None, neighbor_limit: int) -> dict[str, Any]:
    entity_id = str(getattr(entity, 'entity_id', '') or '')
    direct_items: list[dict[str, Any]] = []
    neighbor_ids: list[str] = []

    def add_item(item: dict[str, Any]) -> None:
        direct_items.append(item)
        for value in item.get('entity_ids', []) or []:
            text = str(value).strip()
            if text and text != entity_id:
                neighbor_ids.append(text)
    for episode in graph.episode_nodes:
        ids = _episode_entity_ids(episode)
        if entity_id not in ids:
            continue
        add_item({'source_record_id': episode.node_id, 'source_kind': 'episode', 'summary': _episode_summary(episode), 'time_range': _time_range_payload(episode.time_range), 'scene_id': episode.scene_id, 'entity_ids': ids, 'anchor_distance': _frame_distance_to_range(anchor_frame, _time_range_payload(episode.time_range)) if anchor_frame is not None else None})
    for event in graph.transition_event_nodes:
        ids = _transition_event_entity_ids(event)
        if entity_id not in ids:
            continue
        add_item({'source_record_id': event.node_id, 'source_kind': 'transition_event', 'summary': _transition_event_summary(event), 'time_range': _time_range_payload(event.time_range), 'scene_id': event.scene_id, 'entity_ids': ids, 'anchor_distance': _frame_distance_to_range(anchor_frame, _time_range_payload(event.time_range)) if anchor_frame is not None else None})
    for sample in graph.relation_observed_samples:
        ids = [sample.subject_entity_id, sample.object_entity_id]
        if entity_id not in ids:
            continue
        add_item({'source_record_id': sample.sample_id, 'source_kind': 'relation_observed_sample', 'summary': _short_quote(sample.label, limit=160), 'time_range': _time_range_payload(sample.time_range), 'scene_id': sample.scene_id, 'entity_ids': ids, 'anchor_distance': _frame_distance_to_range(anchor_frame, _time_range_payload(sample.time_range)) if anchor_frame is not None else None})
    direct_items.sort(key=lambda item: (_optional_int(item.get('anchor_distance')) or 999999, _source_kind_order(str(item.get('source_kind') or '')), str(item.get('source_record_id') or '')))
    neighbor_ids = _ordered_unique_texts(neighbor_ids)[:max(1, neighbor_limit)]
    return {'direct_items': direct_items[:max(1, neighbor_limit * 2)], 'neighbor_entity_ids': neighbor_ids}

def _episode_entity_ids(episode: Any) -> list[str]:
    return _ordered_unique_texts([str(value).strip() for value in [getattr(episode, 'primary_entity_id', None), getattr(episode, 'subject_entity_id', None), getattr(episode, 'object_entity_id', None), *(getattr(episode, 'participant_entity_ids', None) or [])] if str(value or '').strip()])

def _transition_event_entity_ids(event: Any) -> list[str]:
    return _ordered_unique_texts([str(value).strip() for value in [getattr(event, 'entity_id', None), getattr(event, 'subject_entity_id', None), getattr(event, 'object_entity_id', None)] if str(value or '').strip()])

def _visual_faiss_entity_local_subgraph_fact(*, context: dict[str, Any], context_index: int, entity: Any, subgraph: dict[str, Any], query_tokens: set[str]) -> dict[str, Any] | None:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    direct_items = [item for item in subgraph.get('direct_items', []) or [] if isinstance(item, dict)]
    entity_text = _visual_faiss_entity_text(entity)
    item_texts = [str(item.get('summary') or '') for item in direct_items]
    all_text = ' '.join([entity_text, *item_texts])
    matched_terms = sorted(query_tokens & _visual_faiss_text_tokens(all_text))
    visual_similarity = _optional_float(visual_object.get('similarity')) or 0.0
    direct_relation_count = sum((1 for item in direct_items if str(item.get('source_kind') or '') in {'episode', 'relation_observed_sample'}))
    direct_event_count = sum((1 for item in direct_items if str(item.get('source_kind') or '') == 'transition_event'))
    checks = {'has_entity_anchor': True, 'direct_item_count': len(direct_items), 'direct_relation_count': direct_relation_count, 'direct_event_count': direct_event_count, 'neighbor_entity_count': len(subgraph.get('neighbor_entity_ids', []) or []), 'matched_question_term_count': len(matched_terms)}
    score = round(visual_similarity * 5.0 + min(4.0, len(direct_items) * 0.45) + min(2.0, direct_relation_count * 0.6) + min(1.5, direct_event_count * 0.5) + min(2.0, len(matched_terms) * 0.25), 6)
    if score <= 0.0 and (not direct_items):
        return None
    fact_text = _visual_faiss_local_subgraph_fact_text(context=context, entity=entity, direct_items=direct_items, neighbor_entity_ids=list(subgraph.get('neighbor_entity_ids', []) or []), matched_terms=matched_terms)
    support_examples = _ordered_unique_texts([entity_text, str(visual_object.get('object_summary') or ''), *item_texts])[:8]
    return {'fact_type': 'visual_faiss_local_subgraph_fact', 'fact_text': fact_text, 'binding_strength': _visual_faiss_local_subgraph_strength(checks), 'support_ranks': [context_index], 'support_context_ranks': [context_index], 'support_examples': support_examples, 'source_record_id': f'visual_faiss_local_subgraph:{context_index}:{entity.entity_id}', 'source': 'visual_faiss_local_subgraph_facts', 'source_kind': 'visual_faiss_local_subgraph_fact', 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'entity_id': entity.entity_id, 'entity_label': entity.label, 'entity_time_range': _time_range_payload(entity.time_range), 'entity_attributes': dict(getattr(entity, 'attributes', {}) or {}), 'neighbor_entity_ids': list(subgraph.get('neighbor_entity_ids', []) or []), 'direct_graph_items': direct_items, 'matched_question_terms': matched_terms, 'local_subgraph_checks': checks, 'score': score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}

def _visual_faiss_entity_text(entity: Any) -> str:
    attrs = getattr(entity, 'attributes', {}) or {}
    values = [getattr(entity, 'entity_id', ''), getattr(entity, 'label', ''), *(getattr(entity, 'labels_seen', None) or []), *(getattr(entity, 'states_seen', None) or []), attrs.get('canonical_label') if isinstance(attrs, dict) else '', attrs.get('semantic_group') if isinstance(attrs, dict) else '', *(attrs.get('raw_visual_attributes', []) if isinstance(attrs, dict) and isinstance(attrs.get('raw_visual_attributes'), list) else [])]
    return ' '.join((str(value).strip() for value in values if str(value).strip()))

def _visual_faiss_local_subgraph_strength(checks: dict[str, Any]) -> str:
    if checks.get('direct_relation_count') and checks.get('direct_event_count'):
        return 'entity_local_subgraph_relation_event_bound'
    if checks.get('direct_relation_count'):
        return 'entity_local_subgraph_relation_bound'
    if checks.get('direct_event_count'):
        return 'entity_local_subgraph_event_bound'
    if checks.get('direct_item_count'):
        return 'entity_local_subgraph_direct_bound'
    return 'entity_local_subgraph_anchor_bound'

def _visual_faiss_local_subgraph_fact_text(*, context: dict[str, Any], entity: Any, direct_items: list[dict[str, Any]], neighbor_entity_ids: list[str], matched_terms: list[str]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    attrs = getattr(entity, 'attributes', {}) or {}
    raw_attrs = attrs.get('raw_visual_attributes', []) if isinstance(attrs, dict) and isinstance(attrs.get('raw_visual_attributes'), list) else []
    direct_summaries = _ordered_unique_texts([_short_text(item.get('summary'), limit=115) for item in direct_items])[:5]
    pieces = ['entity-local subgraph fact', f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', f'entity={entity.entity_id}', f'label={entity.label!r}', f'time={_time_range_text(_time_range_payload(entity.time_range))}', 'visual_anchor=' + _short_text(visual_object.get('object_summary'), limit=110) if visual_object.get('object_summary') else '', 'entity_attributes=' + ' | '.join((_short_text(value, limit=90) for value in raw_attrs[:3])) if raw_attrs else '', 'direct_graph=' + ' | '.join((value for value in direct_summaries if value)) if direct_summaries else '', 'neighbors=' + ','.join(neighbor_entity_ids[:6]) if neighbor_entity_ids else '', 'matched_question_terms=' + ','.join(matched_terms[:10]) if matched_terms else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_local_subgraph_fact_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_local_subgraph_fact', f"score={fact.get('score')}", f"entity={fact.get('entity_id')}", f"label={fact.get('entity_label')!r}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    strength = str(fact.get('binding_strength') or '').strip()
    if strength:
        parts.append(f'binding={strength}')
    return ' '.join(parts) + f" :: {fact.get('fact_text')}"

def _visual_faiss_local_subgraph_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    parts = [f'group=visual_faiss_local_subgraph:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"entity={fact.get('entity_id')}", f"label={fact.get('entity_label')!r}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_local_subgraph_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    checks = fact.get('local_subgraph_checks', {}) if isinstance(fact.get('local_subgraph_checks'), dict) else {}
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-int(checks.get('direct_relation_count') or 0), -int(checks.get('direct_event_count') or 0), -int(checks.get('direct_item_count') or 0), -int(checks.get('matched_question_term_count') or 0), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, str(fact.get('entity_id') or ''))

def _visual_faiss_fragment_fragment_bridge_facts(contexts: list[dict[str, Any]], *, question: str, question_ir: dict[str, Any], structural_binding_enabled: bool=False, limit: int=VISUAL_FAISS_fragment_FRAGMENT_BRIDGE_LIMIT, exclude_schema_terms: bool=False) -> list[dict[str, Any]]:
    """Build object-local bridge facts for fragmented fragment-style graphs.

    The bridge is a default-off post-graph experiment. It never rewrites graph
    identity; it only packages a retrieved object's HSGG facts, same-frame
    context, and nearby canonical graph fragments into compact answerable facts.
    """
    query_terms = _question_terms(question=question, question_ir=question_ir)
    constraint_terms = _visual_faiss_ger_constraint_terms(question_ir)
    if exclude_schema_terms:
        query_terms = _without_schema_match_terms(query_terms)
        constraint_terms = _without_schema_match_terms(constraint_terms)
    query_token_set = set(query_terms)
    constraint_token_set = set(constraint_terms)
    structural_request = _visual_faiss_structural_binding_request(question=question, question_ir=question_ir)
    bridge_facts: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        object_text = _visual_faiss_ger_object_text_for_binding(context)
        anchor_text = _visual_faiss_ger_anchor_text_for_binding(context)
        graph_text = _visual_faiss_ger_graph_text_for_binding(context)
        object_tokens = _visual_faiss_text_tokens(object_text)
        anchor_tokens = _visual_faiss_text_tokens(anchor_text)
        graph_tokens = _visual_faiss_text_tokens(graph_text)
        combined_tokens = object_tokens | graph_tokens
        matched_question_terms = sorted(query_token_set & combined_tokens)
        matched_object_terms = sorted(query_token_set & object_tokens)
        matched_graph_terms = sorted(query_token_set & graph_tokens)
        matched_constraint_terms = sorted(constraint_token_set & combined_tokens)
        attribute_fragments = _visual_faiss_ger_attribute_fragments(context)
        relation_fragments = _visual_faiss_ger_relation_fragments(context)
        frame_context_fragments = _visual_faiss_ger_frame_context_fragments(context)
        graph_fragments = _visual_faiss_ger_graph_fragments(context)
        same_frame_graph_fragments = _visual_faiss_fragment_same_frame_graph_fragments(context)
        if not (attribute_fragments or relation_fragments or frame_context_fragments or graph_fragments):
            continue
        structural_checks = _visual_faiss_structural_binding_checks(combined_tokens=combined_tokens, anchor_tokens=anchor_tokens, object_tokens=object_tokens, graph_tokens=graph_tokens, request=structural_request, attribute_available=bool(attribute_fragments))
        bridge_checks = _visual_faiss_fragment_fragment_bridge_checks(structural_checks=structural_checks, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, frame_context_fragments=frame_context_fragments, graph_fragments=graph_fragments, same_frame_graph_fragments=same_frame_graph_fragments, matched_object_terms=matched_object_terms, matched_graph_terms=matched_graph_terms, structural_binding_enabled=structural_binding_enabled)
        score = _visual_faiss_fragment_fragment_bridge_score(context, bridge_checks=bridge_checks, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms)
        if score <= 0:
            continue
        support_examples = _ordered_unique_texts([str(visual_object.get('object_summary') or ''), *attribute_fragments, *relation_fragments, *frame_context_fragments, *same_frame_graph_fragments, *graph_fragments])[:8]
        fact_text = _visual_faiss_fragment_fragment_bridge_fact_text(context=context, bridge_checks=bridge_checks, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, frame_context_fragments=frame_context_fragments, same_frame_graph_fragments=same_frame_graph_fragments, graph_fragments=graph_fragments)
        bridge_facts.append({'fact_type': 'visual_faiss_fragment_fragment_bridge_fact', 'fact_text': fact_text, 'binding_strength': _visual_faiss_fragment_fragment_bridge_strength(bridge_checks), 'support_ranks': [context_index], 'support_context_ranks': [context_index], 'support_examples': support_examples, 'source_record_id': f'visual_faiss_fragment_fragment_bridge:{context_index}', 'source': 'visual_faiss_fragment_fragment_bridge', 'source_kind': 'visual_faiss_fragment_fragment_bridge_fact', 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'nearby_graph_evidence': list(context.get('nearby_graph_evidence', []) or []), 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'matched_question_terms': matched_question_terms, 'matched_constraint_terms': matched_constraint_terms, 'matched_object_terms': matched_object_terms, 'matched_graph_terms': matched_graph_terms, 'attribute_fragments': attribute_fragments, 'relation_fragments': relation_fragments, 'frame_context_fragments': frame_context_fragments, 'same_frame_graph_fragments': same_frame_graph_fragments, 'graph_fragments': graph_fragments, 'bridge_checks': bridge_checks, 'score': score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    bridge_facts.sort(key=_visual_faiss_fragment_fragment_bridge_sort_key)
    return bridge_facts[:max(1, int(limit))]

def _visual_faiss_fragment_fragment_bridge_checks(*, structural_checks: dict[str, Any], attribute_fragments: list[str], relation_fragments: list[str], frame_context_fragments: list[str], graph_fragments: list[str], same_frame_graph_fragments: list[str], matched_object_terms: list[str], matched_graph_terms: list[str], structural_binding_enabled: bool) -> dict[str, Any]:
    visual_fragment_count = len(attribute_fragments) + len(relation_fragments) + len(frame_context_fragments)
    graph_fragment_count = len(graph_fragments)
    same_frame_graph_count = len(same_frame_graph_fragments)
    return {'visual_fragment_count': visual_fragment_count, 'graph_fragment_count': graph_fragment_count, 'same_frame_graph_count': same_frame_graph_count, 'object_graph_co_support': bool(visual_fragment_count and graph_fragment_count), 'same_frame_co_support': bool(visual_fragment_count and same_frame_graph_count), 'matched_object_term_count': len(matched_object_terms), 'matched_graph_term_count': len(matched_graph_terms), 'structural_binding_enabled': bool(structural_binding_enabled), 'structural_checks': structural_checks, 'required_slot_count': int(structural_checks.get('required_slot_count') or 0), 'satisfied_slot_count': int(structural_checks.get('satisfied_slot_count') or 0), 'target_hits': list(structural_checks.get('target_hits', []) or []), 'attribute_hits': list(structural_checks.get('attribute_hits', []) or []), 'event_hits': list(structural_checks.get('event_hits', []) or []), 'spatial_hits': list(structural_checks.get('spatial_hits', []) or []), 'graph_hits': list(structural_checks.get('graph_hits', []) or [])}

def _visual_faiss_fragment_fragment_bridge_score(context: dict[str, Any], *, bridge_checks: dict[str, Any], matched_question_terms: list[str], matched_constraint_terms: list[str]) -> float:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    similarity = _optional_float(visual_object.get('similarity')) or 0.0
    visual_count = int(bridge_checks.get('visual_fragment_count') or 0)
    graph_count = int(bridge_checks.get('graph_fragment_count') or 0)
    same_frame_count = int(bridge_checks.get('same_frame_graph_count') or 0)
    satisfied_count = int(bridge_checks.get('satisfied_slot_count') or 0)
    score = similarity * 2.5
    score += min(3.0, visual_count * 0.6)
    score += min(2.5, graph_count * 0.45)
    score += min(1.8, same_frame_count * 0.6)
    score += min(1.2, len(matched_question_terms) * 0.12)
    score += min(0.8, len(matched_constraint_terms) * 0.12)
    score += min(2.0, satisfied_count * 0.45)
    if bridge_checks.get('object_graph_co_support'):
        score += 1.2
    if bridge_checks.get('same_frame_co_support'):
        score += 0.8
    if bridge_checks.get('target_hits') and (bridge_checks.get('attribute_hits') or bridge_checks.get('event_hits') or bridge_checks.get('spatial_hits') or bridge_checks.get('graph_hits')):
        score += 0.8
    return round(score, 6)

def _visual_faiss_fragment_fragment_bridge_strength(bridge_checks: dict[str, Any]) -> str:
    if bridge_checks.get('same_frame_co_support') and bridge_checks.get('target_hits'):
        return 'fragment_fragment_same_frame_visual_graph_bound'
    if bridge_checks.get('object_graph_co_support') and bridge_checks.get('target_hits'):
        return 'fragment_fragment_visual_graph_target_bound'
    if bridge_checks.get('object_graph_co_support'):
        return 'fragment_fragment_visual_graph_bound'
    if int(bridge_checks.get('visual_fragment_count') or 0) > 0:
        return 'fragment_fragment_visual_object_bound'
    return 'fragment_fragment_similarity_bound'

def _visual_faiss_fragment_fragment_bridge_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    checks = fact.get('bridge_checks', {}) if isinstance(fact.get('bridge_checks'), dict) else {}
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-int(bool(checks.get('same_frame_co_support'))), -int(bool(checks.get('object_graph_co_support'))), -int(checks.get('satisfied_slot_count') or 0), -int(checks.get('visual_fragment_count') or 0), -int(checks.get('same_frame_graph_count') or 0), -int(checks.get('graph_fragment_count') or 0), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, str(visual_object.get('label') or ''))

def _visual_faiss_fragment_same_frame_graph_fragments(context: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    for item in context.get('nearby_graph_evidence', []) or []:
        if not isinstance(item, dict):
            continue
        anchor_distance = _optional_int(item.get('anchor_distance'))
        if anchor_distance is None or anchor_distance != 0:
            continue
        summary = _short_text(item.get('summary'), limit=150)
        if not summary:
            continue
        source_kind = str(item.get('source_kind') or 'graph').strip() or 'graph'
        fragments.append(f'{source_kind}: {summary}')
    return _ordered_unique_texts(fragments)[:5]

def _visual_faiss_fragment_fragment_bridge_fact_text(*, context: dict[str, Any], bridge_checks: dict[str, Any], matched_question_terms: list[str], matched_constraint_terms: list[str], attribute_fragments: list[str], relation_fragments: list[str], frame_context_fragments: list[str], same_frame_graph_fragments: list[str], graph_fragments: list[str]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    time_range = context.get('time_range', {}) if isinstance(context.get('time_range'), dict) else {}
    pieces = ['fragment fragment-bridge visual fact', f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', f'time={_time_range_text(time_range)}' if _time_range_text(time_range) else '', 'object=' + _short_text(visual_object.get('object_summary'), limit=120) if visual_object.get('object_summary') else '', 'bridge_strength=' + _visual_faiss_fragment_fragment_bridge_strength(bridge_checks), 'matched_question_terms=' + ','.join(matched_question_terms[:10]) if matched_question_terms else '', 'matched_constraint_terms=' + ','.join(matched_constraint_terms[:10]) if matched_constraint_terms else '', 'object_attributes=' + ' | '.join((_short_text(value, limit=120) for value in attribute_fragments[:4])) if attribute_fragments else '', 'object_relations=' + ' | '.join((_short_text(value, limit=110) for value in relation_fragments[:3])) if relation_fragments else '', 'same_frame_context=' + ' | '.join((_short_text(value, limit=110) for value in frame_context_fragments[:3])) if frame_context_fragments else '', 'same_frame_graph=' + ' | '.join((_short_text(value, limit=110) for value in same_frame_graph_fragments[:3])) if same_frame_graph_fragments else '', 'nearby_graph=' + ' | '.join((_short_text(value, limit=110) for value in graph_fragments[:3])) if graph_fragments else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_fragment_fragment_bridge_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_fragment_fragment_bridge_fact', f"score={fact.get('score')}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    label = str(visual_object.get('label') or '').strip()
    if label:
        parts.append(f'label={label!r}')
    checks = fact.get('bridge_checks', {}) if isinstance(fact.get('bridge_checks'), dict) else {}
    parts.append(f"visual_fragments={checks.get('visual_fragment_count', 0)}")
    parts.append(f"graph_fragments={checks.get('graph_fragment_count', 0)}")
    return ' '.join(parts) + f" :: {fact.get('fact_text')}"

def _visual_faiss_fragment_fragment_bridge_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'group=visual_faiss_fragment_fragment_bridge:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"label={visual_object.get('label')!r}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_ger_bound_facts(contexts: list[dict[str, Any]], *, question: str, question_ir: dict[str, Any], structural_binding_enabled: bool=False, attribute_owner_enabled: bool=False, exclude_schema_terms: bool=False) -> list[dict[str, Any]]:
    """Create generic object/event/attribute binding facts for GER-like recall.

    This is a deterministic experiment layer. It uses only question text/IR and
    retrieved graph/visual facts; it does not inspect gold answers, question
    type, or dataset-specific lexicons.
    """
    query_terms = _question_terms(question=question, question_ir=question_ir)
    constraint_terms = _visual_faiss_ger_constraint_terms(question_ir)
    if exclude_schema_terms:
        query_terms = _without_schema_match_terms(query_terms)
        constraint_terms = _without_schema_match_terms(constraint_terms)
    structural_request = _visual_faiss_structural_binding_request(question=question, question_ir=question_ir)
    bound: list[dict[str, Any]] = []
    for index, context in enumerate(contexts, start=1):
        visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
        object_text = _visual_faiss_ger_object_text_for_binding(context)
        anchor_text = _visual_faiss_ger_anchor_text_for_binding(context)
        graph_text = _visual_faiss_ger_graph_text_for_binding(context)
        combined_text = ' '.join([object_text, graph_text]).strip()
        object_tokens = _visual_faiss_text_tokens(object_text)
        anchor_tokens = _visual_faiss_text_tokens(anchor_text)
        graph_tokens = _visual_faiss_text_tokens(graph_text)
        combined_tokens = object_tokens | graph_tokens
        query_token_set = set(query_terms)
        constraint_token_set = set(constraint_terms)
        matched_question_terms = sorted(query_token_set & combined_tokens)
        matched_constraint_terms = sorted(constraint_token_set & combined_tokens)
        object_only_question_terms = sorted(query_token_set & object_tokens)
        graph_only_question_terms = sorted(query_token_set & graph_tokens)
        attribute_fragments = _visual_faiss_ger_attribute_fragments(context, owner_aware=attribute_owner_enabled)
        relation_fragments = _visual_faiss_ger_relation_fragments(context)
        frame_context_fragments = _visual_faiss_ger_frame_context_fragments(context)
        graph_fragments = _visual_faiss_ger_graph_fragments(context)
        structural_checks = _visual_faiss_structural_binding_checks(combined_tokens=combined_tokens, anchor_tokens=anchor_tokens, object_tokens=object_tokens, graph_tokens=graph_tokens, request=structural_request, attribute_available=bool(attribute_fragments)) if structural_binding_enabled else {}
        binding_score = _visual_faiss_structural_binding_score(context, structural_checks=structural_checks, matched_question_terms=matched_question_terms, object_only_question_terms=object_only_question_terms, graph_only_question_terms=graph_only_question_terms) if structural_binding_enabled else _visual_faiss_ger_binding_score(context, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms, object_only_question_terms=object_only_question_terms, graph_only_question_terms=graph_only_question_terms)
        if binding_score <= 0:
            continue
        support_examples = _ordered_unique_texts([str(visual_object.get('object_summary') or ''), *attribute_fragments, *relation_fragments, *frame_context_fragments, *graph_fragments])[:6]
        fact_text = _visual_faiss_ger_bound_fact_text(context=context, matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms, attribute_fragments=attribute_fragments, relation_fragments=relation_fragments, frame_context_fragments=frame_context_fragments, graph_fragments=graph_fragments)
        bound.append({'fact_type': 'visual_faiss_ger_bound_fact', 'fact_text': fact_text, 'binding_strength': _visual_faiss_structural_binding_strength(structural_checks) if structural_binding_enabled else _visual_faiss_ger_binding_strength(matched_question_terms=matched_question_terms, matched_constraint_terms=matched_constraint_terms, object_only_question_terms=object_only_question_terms, graph_only_question_terms=graph_only_question_terms), 'support_ranks': [index], 'support_context_ranks': [index], 'support_examples': support_examples, 'source_record_id': f'visual_faiss_ger_bound:{index}', 'source': 'visual_faiss_ger_binding', 'source_kind': 'visual_faiss_ger_bound_fact', 'query_phrase': context.get('query_phrase'), 'visual_object': visual_object, 'nearby_graph_evidence': list(context.get('nearby_graph_evidence', []) or []), 'scene_ids': list(context.get('scene_ids', []) or []), 'entity_ids': list(context.get('entity_ids', []) or []), 'time_ranges': [context.get('time_range', {})], 'matched_question_terms': matched_question_terms, 'matched_constraint_terms': matched_constraint_terms, 'object_only_question_terms': object_only_question_terms, 'graph_only_question_terms': graph_only_question_terms, 'attribute_fragments': attribute_fragments, 'attribute_owner_enabled': bool(attribute_owner_enabled), 'relation_fragments': relation_fragments, 'frame_context_fragments': frame_context_fragments, 'graph_fragments': graph_fragments, 'structural_binding_checks': structural_checks, 'structural_binding_enabled': bool(structural_binding_enabled), 'score': binding_score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    bound.sort(key=_visual_faiss_structural_bound_fact_sort_key if structural_binding_enabled else _visual_faiss_ger_bound_fact_sort_key)
    for rank, fact in enumerate(bound, start=1):
        fact['support_ranks'] = [rank]
        fact['bound_rank'] = rank
    return bound

def _visual_faiss_structural_binding_request(*, question: str, question_ir: dict[str, Any]) -> dict[str, list[str]]:
    return {'target_terms': _visual_faiss_terms_from_ir_fields(question_ir, 'target_entities', 'target_state'), 'attribute_terms': _visual_faiss_terms_from_ir_fields(question_ir, 'attribute_dimension'), 'event_terms': _visual_faiss_terms_from_ir_fields(question_ir, 'event_constraints', 'hard_constraints', 'temporal_relation'), 'spatial_terms': _visual_faiss_terms_from_ir_fields(question_ir, 'spatial_constraints'), 'need_terms': _visual_faiss_terms_from_ir_fields(question_ir, 'graph_information_needs', 'evidence_requirements'), 'question_terms': sorted(_visual_faiss_text_tokens(question))}

def _visual_faiss_terms_from_ir_fields(question_ir: dict[str, Any], *field_names: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for field_name in field_names:
        for value in _iter_text_values(question_ir.get(field_name)):
            for token in _visual_faiss_text_tokens(value):
                if token in seen:
                    continue
                terms.append(token)
                seen.add(token)
    return terms[:40]

def _visual_faiss_structural_binding_checks(*, combined_tokens: set[str], anchor_tokens: set[str], object_tokens: set[str], graph_tokens: set[str], request: dict[str, list[str]], attribute_available: bool) -> dict[str, Any]:
    target_hits = sorted(set(request.get('target_terms', [])) & anchor_tokens)
    attribute_hits = sorted(set(request.get('attribute_terms', [])) & combined_tokens)
    event_hits = sorted(set(request.get('event_terms', [])) & combined_tokens)
    spatial_hits = sorted(set(request.get('spatial_terms', [])) & combined_tokens)
    need_hits = sorted(set(request.get('need_terms', [])) & combined_tokens)
    graph_hits = sorted((set(request.get('event_terms', [])) | set(request.get('spatial_terms', [])) | set(request.get('need_terms', [])) | set(request.get('question_terms', []))) & graph_tokens)
    required_slots = {'target': bool(request.get('target_terms')), 'attribute': bool(request.get('attribute_terms')), 'event': bool(request.get('event_terms')), 'spatial': bool(request.get('spatial_terms')), 'need': bool(request.get('need_terms'))}
    satisfied_slots = {'target': bool(target_hits), 'attribute': bool(attribute_hits) or bool(attribute_available), 'event': bool(event_hits), 'spatial': bool(spatial_hits), 'need': bool(need_hits)}
    required_count = sum((1 for value in required_slots.values() if value))
    satisfied_count = sum((1 for slot, required in required_slots.items() if required and satisfied_slots.get(slot)))
    return {'target_hits': target_hits, 'attribute_hits': attribute_hits, 'event_hits': event_hits, 'spatial_hits': spatial_hits, 'need_hits': need_hits, 'graph_hits': graph_hits, 'attribute_available': bool(attribute_available), 'required_slots': required_slots, 'satisfied_slots': satisfied_slots, 'required_slot_count': required_count, 'satisfied_slot_count': satisfied_count}

def _visual_faiss_structural_binding_score(context: dict[str, Any], *, structural_checks: dict[str, Any], matched_question_terms: list[str], object_only_question_terms: list[str], graph_only_question_terms: list[str]) -> float:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    similarity = _optional_float(visual_object.get('similarity')) or 0.0
    target_hits = list(structural_checks.get('target_hits', []) or [])
    attribute_hits = list(structural_checks.get('attribute_hits', []) or [])
    event_hits = list(structural_checks.get('event_hits', []) or [])
    spatial_hits = list(structural_checks.get('spatial_hits', []) or [])
    need_hits = list(structural_checks.get('need_hits', []) or [])
    graph_hits = list(structural_checks.get('graph_hits', []) or [])
    required_count = int(structural_checks.get('required_slot_count') or 0)
    satisfied_count = int(structural_checks.get('satisfied_slot_count') or 0)
    score = similarity * 4.0
    score += len(target_hits) * 2.0
    score += len(attribute_hits) * 1.4
    score += len(event_hits) * 1.7
    score += len(spatial_hits) * 1.4
    score += len(need_hits) * 1.0
    score += min(1.5, len(graph_hits) * 0.35)
    score += min(1.0, len(matched_question_terms) * 0.12)
    score += min(0.6, len(object_only_question_terms) * 0.12)
    score += min(0.6, len(graph_only_question_terms) * 0.12)
    if _visual_faiss_ger_attribute_fragments(context):
        score += 1.0
    if _visual_faiss_ger_relation_fragments(context):
        score += 0.9
    if _visual_faiss_ger_graph_fragments(context):
        score += 0.8
    if _visual_faiss_ger_frame_context_fragments(context):
        score += 0.5
    if required_count:
        score += satisfied_count * 1.2
        if satisfied_count == required_count:
            score += 2.0
        elif satisfied_count == 0:
            score -= 1.0
    if target_hits and (event_hits or spatial_hits or need_hits or graph_hits):
        score += 1.5
    if attribute_hits and target_hits:
        score += 1.2
    return round(score, 6)

def _visual_faiss_structural_binding_strength(structural_checks: dict[str, Any]) -> str:
    required_count = int(structural_checks.get('required_slot_count') or 0)
    satisfied_count = int(structural_checks.get('satisfied_slot_count') or 0)
    target_hits = bool(structural_checks.get('target_hits'))
    attribute_hits = bool(structural_checks.get('attribute_hits'))
    event_hits = bool(structural_checks.get('event_hits'))
    spatial_hits = bool(structural_checks.get('spatial_hits'))
    graph_hits = bool(structural_checks.get('graph_hits'))
    if required_count and satisfied_count == required_count and target_hits:
        return 'structural_all_required_slots_bound'
    if target_hits and attribute_hits and (event_hits or spatial_hits or graph_hits):
        return 'structural_target_attribute_context_bound'
    if target_hits and (event_hits or spatial_hits or graph_hits):
        return 'structural_target_context_bound'
    if target_hits:
        return 'structural_target_only_bound'
    if satisfied_count:
        return 'structural_partial_context_bound'
    return 'structural_visual_similarity_bound'

def _visual_faiss_ger_bound_fact_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    return (-float(fact.get('score') or 0.0), -len(fact.get('matched_constraint_terms', []) or []), -len(fact.get('matched_question_terms', []) or []), _optional_int(visual_object.get('graph_frame_index')) or 0, str(visual_object.get('label') or ''))

def _visual_faiss_structural_bound_fact_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    checks = fact.get('structural_binding_checks', {}) if isinstance(fact.get('structural_binding_checks'), dict) else {}
    return (-int(checks.get('satisfied_slot_count') or 0), -len(checks.get('target_hits', []) or []), -len(checks.get('attribute_hits', []) or []), -len(checks.get('event_hits', []) or []), -len(checks.get('spatial_hits', []) or []), -len(checks.get('graph_hits', []) or []), -float(fact.get('score') or 0.0), _optional_int(visual_object.get('graph_frame_index')) or 0, str(visual_object.get('label') or ''))

def _visual_faiss_ger_constraint_terms(question_ir: dict[str, Any]) -> list[str]:
    raw_values: list[str] = []
    for key in ('target_entities', 'hard_constraints', 'event_constraints', 'spatial_constraints', 'attribute_dimension', 'graph_information_needs', 'evidence_requirements', 'target_state'):
        raw_values.extend(_iter_text_values(question_ir.get(key)))
    terms: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for token in _visual_faiss_text_tokens(value):
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms[:40]

def _visual_faiss_ger_anchor_text_for_binding(context: dict[str, Any]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    values = [str(visual_object.get('label') or ''), str(visual_object.get('visual_attribute') or ''), str(visual_object.get('object_summary') or '')]
    return ' '.join((value for value in values if value.strip()))

def _visual_faiss_ger_object_text_for_binding(context: dict[str, Any]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    values = [str(visual_object.get('label') or ''), str(visual_object.get('visual_attribute') or ''), str(visual_object.get('object_summary') or ''), *[str(value) for value in context.get('hsgg_visual_facts', []) or []], *[str(value) for value in context.get('hsgg_neighbor_attributes', []) or []], *[str(value) for value in context.get('hsgg_frame_context', []) or []]]
    for relation in visual_object.get('visual_relations', []) or []:
        values.append(str(relation))
    return ' '.join((value for value in values if value.strip()))

def _visual_faiss_ger_graph_text_for_binding(context: dict[str, Any]) -> str:
    values = [str(item.get('summary') or '') for item in context.get('nearby_graph_evidence', []) or [] if isinstance(item, dict)]
    return ' '.join((value for value in values if value.strip()))

def _visual_faiss_ger_frame_context_fragments(context: dict[str, Any]) -> list[str]:
    fragments = [_short_text(item, limit=160) for item in context.get('hsgg_frame_context', []) or [] if _short_text(item, limit=160)]
    return _ordered_unique_texts(fragments)[:6]

def _visual_faiss_ger_attribute_fragments(context: dict[str, Any], *, owner_aware: bool=False) -> list[str]:
    fragments: list[str] = []
    seen_owner_attribute_pairs: set[tuple[str, str, str]] = set()
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    owner = _visual_object_owner_text(visual_object)
    attribute = _short_text(visual_object.get('visual_attribute'), limit=140)
    if attribute:
        if owner_aware:
            _append_owner_attribute_fragment(fragments, seen_owner_attribute_pairs, role='direct_object_attribute', owner=owner, attribute=attribute)
        else:
            fragments.append(f'object_attribute={attribute}')
    for fact in context.get('hsgg_visual_facts', []) or []:
        text = _short_text(fact, limit=160)
        if text and (' attr=' in text or 'visual_attribute' in text):
            _append_owner_aware_attribute_text(fragments, seen_owner_attribute_pairs, _owner_aware_hsgg_attribute_text(text, anchor_owner=owner, owner_aware=owner_aware))
    for neighbor in context.get('hsgg_neighbor_attributes', []) or []:
        text = _short_text(neighbor, limit=160)
        if text:
            _append_owner_aware_attribute_text(fragments, seen_owner_attribute_pairs, _owner_aware_hsgg_attribute_text(text, anchor_owner=owner, owner_aware=owner_aware))
    return _ordered_unique_texts(fragments)[:6]

def _append_owner_attribute_fragment(fragments: list[str], seen: set[tuple[str, str, str]], *, role: str, owner: str, attribute: str) -> None:
    clean_owner = _short_text(owner, limit=120)
    clean_attribute = _short_text(attribute, limit=140)
    if not clean_owner or not clean_attribute:
        return
    key = (role, _owner_identity_key(clean_owner), clean_attribute.lower())
    if key in seen:
        return
    seen.add(key)
    fragments.append(f'{role} owner={clean_owner}: {clean_attribute}')

def _owner_identity_key(owner: str) -> str:
    obj_match = re.search('\\bobj=(\\d+)\\b', owner)
    label_match = re.search('\\blabel=(\'[^\']+\'|\\"[^\\"]+\\"|\\S+)', owner)
    if obj_match:
        label = label_match.group(1) if label_match else ''
        return f'obj={obj_match.group(1)}|label={label}'
    return owner

def _append_owner_aware_attribute_text(fragments: list[str], seen: set[tuple[str, str, str]], text: str) -> None:
    clean_text = _short_text(text, limit=180)
    if not clean_text:
        return
    parsed = _parse_owner_aware_attribute_text(clean_text)
    if parsed is None:
        fragments.append(clean_text)
        return
    role, owner, attribute = parsed
    _append_owner_attribute_fragment(fragments, seen, role=role, owner=owner, attribute=attribute)

def _parse_owner_aware_attribute_text(text: str) -> tuple[str, str, str] | None:
    match = re.match('^(direct_object_attribute|neighbor_object_attribute)\\s+(.+?)\\s+attr=(.+)$', text)
    if match:
        return (match.group(1), match.group(2).strip(), match.group(3).strip())
    match = re.match('^(direct_object_attribute|neighbor_object_attribute)\\s+owner=(.+?):\\s*(.+)$', text)
    if match:
        return (match.group(1), match.group(2).strip(), match.group(3).strip())
    return None

def _visual_object_owner_text(visual_object: dict[str, Any]) -> str:
    frame = _optional_int(visual_object.get('graph_frame_index'))
    object_index = _optional_int(visual_object.get('object_index'))
    label = _short_text(visual_object.get('label'), limit=48) or 'object'
    parts: list[str] = []
    if frame is not None:
        parts.append(f'frame={frame}')
    if object_index is not None:
        parts.append(f'obj={object_index}')
    parts.append(f'label={label!r}')
    return ' '.join(parts)

def _owner_aware_hsgg_attribute_text(text: str, *, anchor_owner: str, owner_aware: bool) -> str:
    if not owner_aware:
        return text
    if text.startswith('direct_object_attribute ') or text.startswith('neighbor_object_attribute '):
        return text
    if text.startswith('neighbor '):
        return 'neighbor_object_attribute ' + text[len('neighbor '):]
    if text.startswith('object ') and ' attr=' in text:
        attribute = text.split(' attr=', 1)[1].strip()
        return f'direct_object_attribute owner={anchor_owner}: {attribute}'
    if 'visual_attribute' in text or ' attr=' in text:
        return f'direct_object_attribute owner={anchor_owner}: {text}'
    return text

def _visual_faiss_ger_relation_fragments(context: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    for relation in visual_object.get('visual_relations', []) or []:
        text = _short_text(relation, limit=140)
        if text:
            fragments.append(f'object_relation={text}')
    for fact in context.get('hsgg_visual_facts', []) or []:
        text = _short_text(fact, limit=160)
        if text and ' relation=' in text:
            fragments.append(text)
    return _ordered_unique_texts(fragments)[:5]

def _visual_faiss_ger_graph_fragments(context: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    for item in context.get('nearby_graph_evidence', []) or []:
        if not isinstance(item, dict):
            continue
        summary = _short_text(item.get('summary'), limit=150)
        if not summary:
            continue
        source_kind = str(item.get('source_kind') or 'graph').strip() or 'graph'
        fragments.append(f'{source_kind}: {summary}')
    return _ordered_unique_texts(fragments)[:5]

def _visual_faiss_ger_binding_score(context: dict[str, Any], *, matched_question_terms: list[str], matched_constraint_terms: list[str], object_only_question_terms: list[str], graph_only_question_terms: list[str]) -> float:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    similarity = _optional_float(visual_object.get('similarity')) or 0.0
    score = similarity * 4.0
    score += len(matched_question_terms) * 1.2
    score += len(matched_constraint_terms) * 1.6
    score += len(object_only_question_terms) * 0.6
    score += len(graph_only_question_terms) * 0.4
    if _visual_faiss_ger_attribute_fragments(context):
        score += 1.2
    if _visual_faiss_ger_relation_fragments(context):
        score += 0.8
    if _visual_faiss_ger_graph_fragments(context):
        score += 0.8
    if _visual_faiss_ger_frame_context_fragments(context):
        score += 0.6
    if object_only_question_terms and graph_only_question_terms:
        score += 1.0
    return round(score, 6)

def _visual_faiss_ger_binding_strength(*, matched_question_terms: list[str], matched_constraint_terms: list[str], object_only_question_terms: list[str], graph_only_question_terms: list[str]) -> str:
    if matched_constraint_terms and object_only_question_terms and graph_only_question_terms:
        return 'question_constraints_object_and_graph_bound'
    if matched_constraint_terms and object_only_question_terms:
        return 'question_constraints_object_bound'
    if object_only_question_terms and graph_only_question_terms:
        return 'question_terms_object_and_graph_bound'
    if matched_question_terms:
        return 'question_terms_object_bound'
    return 'visual_similarity_bound'

def _visual_faiss_ger_bound_fact_text(*, context: dict[str, Any], matched_question_terms: list[str], matched_constraint_terms: list[str], attribute_fragments: list[str], relation_fragments: list[str], frame_context_fragments: list[str], graph_fragments: list[str]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    pieces = ['GER-like bound visual fact', f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', 'object=' + _short_text(visual_object.get('object_summary'), limit=120) if visual_object.get('object_summary') else '', 'matched_question_terms=' + ','.join(matched_question_terms[:10]) if matched_question_terms else '', 'matched_constraint_terms=' + ','.join(matched_constraint_terms[:10]) if matched_constraint_terms else '', 'attributes=' + ' | '.join((_short_text(value, limit=120) for value in attribute_fragments[:4])) if attribute_fragments else '', 'relations=' + ' | '.join((_short_text(value, limit=110) for value in relation_fragments[:3])) if relation_fragments else '', 'frame_context=' + ' | '.join((_short_text(value, limit=110) for value in frame_context_fragments[:3])) if frame_context_fragments else '', 'nearby_graph=' + ' | '.join((_short_text(value, limit=110) for value in graph_fragments[:3])) if graph_fragments else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_ger_bound_fact_line(*, index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'rank={index}', 'kind=visual_faiss_ger_bound_fact', f"score={fact.get('score')}"]
    frame = visual_object.get('graph_frame_index')
    if frame is not None:
        parts.append(f'frame={frame}')
    label = str(visual_object.get('label') or '').strip()
    if label:
        parts.append(f'label={label!r}')
    matched_question_terms = [str(value).strip() for value in fact.get('matched_question_terms', []) or [] if str(value).strip()]
    if matched_question_terms:
        parts.append('q_terms=' + ','.join(matched_question_terms[:8]))
    matched_constraint_terms = [str(value).strip() for value in fact.get('matched_constraint_terms', []) or [] if str(value).strip()]
    if matched_constraint_terms:
        parts.append('constraint_terms=' + ','.join(matched_constraint_terms[:8]))
    return ' '.join(parts) + f" :: {fact.get('fact_text')}"

def _visual_faiss_ger_bound_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'group=visual_faiss_ger_bound:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"label={visual_object.get('label')!r}"]
    matched_question_terms = [str(value).strip() for value in fact.get('matched_question_terms', []) or [] if str(value).strip()]
    if matched_question_terms:
        parts.append('q_terms=' + ','.join(matched_question_terms[:8]))
    matched_constraint_terms = [str(value).strip() for value in fact.get('matched_constraint_terms', []) or [] if str(value).strip()]
    if matched_constraint_terms:
        parts.append('constraint_terms=' + ','.join(matched_constraint_terms[:8]))
    parts.append(str(fact.get('fact_text') or ''))
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _visual_faiss_ger_answerable_fact_text(context: dict[str, Any]) -> str:
    visual_object = context.get('visual_object', {}) if isinstance(context.get('visual_object'), dict) else {}
    facts = [str(value).strip() for value in context.get('hsgg_visual_facts', []) or [] if str(value).strip()]
    if not facts:
        attribute = str(visual_object.get('visual_attribute') or '').strip()
        if attribute:
            facts.append(f'visual_attribute={attribute}')
    neighbor_facts = [str(value).strip() for value in context.get('hsgg_neighbor_attributes', []) or [] if str(value).strip()]
    frame_context = [str(value).strip() for value in context.get('hsgg_frame_context', []) or [] if str(value).strip()]
    graph_facts = [str(item.get('summary') or '').strip() for item in context.get('nearby_graph_evidence', []) or [] if str(item.get('summary') or '').strip()]
    pieces = ['GER-like visual object and graph context', f"query={context.get('query_phrase')!r}" if context.get('query_phrase') else '', f"object={visual_object.get('object_summary')}" if visual_object.get('object_summary') else '', 'object_facts=' + ' | '.join((_short_text(value, limit=140) for value in facts[:6])) if facts else '', 'neighbor_facts=' + ' | '.join((_short_text(value, limit=120) for value in neighbor_facts[:4])) if neighbor_facts else '', 'frame_context=' + ' | '.join((_short_text(value, limit=120) for value in frame_context[:4])) if frame_context else '', 'nearby_graph=' + ' | '.join((_short_text(value, limit=120) for value in graph_facts[:4])) if graph_facts else '']
    return '; '.join((piece for piece in pieces if piece))

def _visual_faiss_ger_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    visual_object = fact.get('visual_object', {}) if isinstance(fact.get('visual_object'), dict) else {}
    parts = [f'group=visual_faiss_ger:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"frame={visual_object.get('graph_frame_index')}", f"label={visual_object.get('label')!r}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:3]))
    return ' '.join((part for part in parts if part))

def _concise_visual_faiss_ger_object(row: dict[str, Any]) -> dict[str, Any]:
    return {'faiss_id': _optional_int(row.get('faiss_id')), 'seq_id': str(row.get('seq_id') or ''), 'frame_key': row.get('frame_key'), 'image_name': row.get('image_name'), 'graph_frame_index': _optional_int(row.get('graph_frame_index')), 'object_index': _optional_int(row.get('object_index')), 'label': row.get('label'), 'similarity': row.get('similarity'), 'elapsed_sec_from_sequence_start': row.get('elapsed_sec_from_sequence_start'), 'bbox': row.get('bbox'), 'image_path': row.get('image_path'), 'visual_attribute': row.get('visual_attribute'), 'visual_relations': list(row.get('visual_relations', []) or []), 'hsgg_visual_facts': list(row.get('hsgg_visual_facts', []) or []), 'hsgg_neighbor_attributes': list(row.get('hsgg_neighbor_attributes', []) or []), 'hsgg_frame_context': list(row.get('hsgg_frame_context', []) or []), 'compression_score': row.get('compression_score'), 'query_phrase_role': row.get('query_phrase_role'), 'role_query_role': row.get('role_query_role'), 'role_aware_selection_bucket': row.get('role_aware_selection_bucket'), 'role_aware_compression_score': row.get('role_aware_compression_score'), 'role_checks': row.get('role_checks', {}), 'generic_query_overlap_tokens': list(row.get('generic_query_overlap_tokens', []) or []), 'object_summary': _compressed_visual_object_text(row)}

def _visual_anchor_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    anchor = fact.get('visual_anchor', {}) if isinstance(fact.get('visual_anchor'), dict) else {}
    parts = [f'group=visual_faiss_anchor:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", f"anchor_frame={anchor.get('frame_index')}", f"anchor_label={anchor.get('label')!r}", str(fact.get('fact_text') or '')]
    return ' '.join((part for part in parts if part))

def _store_visual_anchor_expansion_metadata(graph_answer: dict[str, Any], *, anchors: list[dict[str, Any]], graph_items: list[dict[str, Any]], soft_contexts: list[dict[str, Any]], frame_radius: int) -> None:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return
    metadata = {'contract': 'experimental_visual_faiss_graph_anchor_expansion_v1', 'frame_radius': int(frame_radius), 'anchor_count': len(anchors), 'soft_context_count': len(soft_contexts), 'promoted_graph_evidence_count': len(graph_items), 'anchors': anchors, 'soft_contexts': soft_contexts, 'promoted_graph_evidence': graph_items}
    bundle['visual_faiss_anchor_expansion'] = metadata
    structured = bundle.get('structured_graph_context', {})
    if isinstance(structured, dict):
        structured['visual_faiss_anchor_expansion'] = metadata

def _episode_summary(episode: Any) -> str:
    parts = [str(getattr(episode, 'label', '') or '').strip()]
    attr_key = getattr(episode, 'attr_key', None)
    attr_value = getattr(episode, 'attr_value', None)
    predicate = getattr(episode, 'predicate', None) or getattr(episode, 'raw_predicate', None)
    if attr_key or attr_value:
        parts.append(f'state {attr_key}={attr_value}')
    if predicate:
        parts.append(f'relation {predicate}')
    return _short_quote(' '.join((part for part in parts if part)), limit=180)

def _transition_event_summary(event: Any) -> str:
    parts = [str(getattr(event, 'label', '') or '').strip()]
    event_type = getattr(event, 'event_type', None)
    if event_type:
        parts.append(f'type={event_type}')
    previous_value = getattr(event, 'previous_state_value', None)
    current_value = getattr(event, 'current_state_value', None)
    if previous_value or current_value:
        parts.append(f'state {previous_value}->{current_value}')
    predicate = getattr(event, 'predicate', None) or getattr(event, 'raw_predicate', None)
    if predicate:
        parts.append(f'relation {predicate}')
    return _short_quote(' '.join((part for part in parts if part)), limit=180)

def _time_range_text(time_range: dict[str, Any]) -> str:
    start = time_range.get('start_frame')
    end = time_range.get('end_frame')
    if start is None and end is None:
        return ''
    if start == end or end is None:
        return f'frame={start}'
    if start is None:
        return f'frame={end}'
    return f'frames={start}-{end}'

def _focus_frame_indices(graph_answer: dict[str, Any]) -> list[int]:
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if not isinstance(retrieval_bundle, dict):
        return []
    ordered: list[int] = []
    seen: set[int] = set()
    for item in retrieval_bundle.get('focus_frames', []):
        if not isinstance(item, dict):
            continue
        frame_index = _optional_int(item.get('frame_index'))
        if frame_index is None or frame_index in seen:
            continue
        ordered.append(frame_index)
        seen.add(frame_index)
    return ordered

def _question_terms(*, question: str, question_ir: dict[str, Any]) -> list[str]:
    raw_values: list[str] = [question]
    for key in ('target_entities', 'core_terms', 'hard_constraints', 'attribute_dimension', 'graph_information_needs', 'evidence_requirements'):
        value = question_ir.get(key)
        if isinstance(value, list):
            raw_values.extend((str(item) for item in value))
        elif value is not None:
            raw_values.append(str(value))
    terms: list[str] = []
    seen: set[str] = set()
    for text in raw_values:
        for token in re.findall('[a-zA-Z][a-zA-Z0-9_-]{2,}', text.lower()):
            token = token.replace('_', '-').strip('-')
            if not token or token in _STOPWORDS or token in seen:
                continue
            terms.append(token)
            seen.add(token)
    return terms[:40]

def _without_schema_match_terms(terms: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        token = str(term or '').strip().lower()
        if not token or token in _VISUAL_FAISS_SCHEMA_MATCH_TOKENS or token in seen:
            continue
        filtered.append(token)
        seen.add(token)
    return filtered

def _seed_score(item: dict[str, Any], *, terms: list[str], frame_rank: int) -> float:
    label = str(item.get('label') or '').lower()
    label_tokens = set(re.findall('[a-zA-Z][a-zA-Z0-9_-]{1,}', label))
    term_hits = 0
    for term in terms:
        if term in label or term in label_tokens:
            term_hits += 1
    hsgg_score = item.get('score')
    try:
        object_score = float(hsgg_score or 0.0)
    except (TypeError, ValueError):
        object_score = 0.0
    frame_bonus = max(0.0, 1.0 - 0.08 * max(0, frame_rank - 1))
    return frame_bonus + object_score + term_hits * 2.5

def _render_visual_evidence_lines(groups: list[dict[str, Any]], *, max_lines: int) -> list[str]:
    lines: list[str] = []
    for group in groups:
        seed = group['seed']
        seed_text = _object_display(seed)
        result_parts = [_object_display(row, include_similarity=True) for row in group['results']]
        if not result_parts:
            continue
        lines.append(f'seed {seed_text} has visually similar object evidence: ' + '; '.join(result_parts[:4]))
        if len(lines) >= max_lines:
            break
    return lines

def _render_visual_query_recall_lines(groups: list[dict[str, Any]], *, max_lines: int, question: str='', question_ir: dict[str, Any] | None=None, phrase_records: list[dict[str, Any]] | None=None, role_aware_compression_enabled: bool=False) -> list[str]:
    payload = {'question': str(question or ''), 'question_ir': question_ir or {}, 'query_phrase_records': phrase_records or [], 'groups': groups, 'compression': {'selected_object_limit': VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT, 'role_aware_compression_enabled': bool(role_aware_compression_enabled)}}
    lines = [_visual_faiss_object_evidence_line(phrase=phrase, row=row) for phrase, row in _compressed_visual_faiss_rows(payload)]
    return lines

def _visual_focus_frames_from_groups(groups: list[dict[str, Any]], *, max_frames: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in groups:
        phrase = str(group.get('query_phrase') or '').strip()
        for row in group.get('results', []):
            frame_index = _optional_int(row.get('graph_frame_index'))
            if frame_index is None or frame_index in seen:
                continue
            seen.add(frame_index)
            frames.append({'frame_index': frame_index, 'timestamp_sec': row.get('elapsed_sec_from_sequence_start'), 'scene_id': None, 'score': round(float(row.get('similarity') or 0.0), 6), 'reasons': ['visual_query_faiss'], 'query_phrase': phrase, 'label': row.get('label'), 'object_index': row.get('object_index')})
            if len(frames) >= max_frames:
                return frames
    return frames

def _visual_query_recall_result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    entity_id = str(row.get('entity_id') or row.get('anchor_entity_id') or '').strip()
    if row.get('tracklet_hit') and entity_id:
        return (str(row.get('seq_id') or ''), 'entity_tracklet', entity_id)
    return (str(row.get('seq_id') or ''), 'object_observation', _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))

def _append_visual_focus_frames(graph_answer: dict[str, Any], visual_payload: dict[str, Any]) -> None:
    visual_frames = [item for item in visual_payload.get('visual_focus_frames', []) if isinstance(item, dict)]
    if not visual_frames:
        return
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if not isinstance(retrieval_bundle, dict):
        return
    focus_frames = [item for item in retrieval_bundle.get('focus_frames', []) if isinstance(item, dict)]
    seen = {_optional_int(item.get('frame_index')) for item in focus_frames if _optional_int(item.get('frame_index')) is not None}
    next_rank = len(focus_frames) + 1
    appended_lines: list[str] = []
    for item in visual_frames:
        frame_index = _optional_int(item.get('frame_index'))
        if frame_index is None or frame_index in seen:
            continue
        seen.add(frame_index)
        focus = {'rank': next_rank, 'frame_index': frame_index, 'frame_rank': frame_index, 'timestamp_sec': item.get('timestamp_sec'), 'scene_id': item.get('scene_id'), 'score': item.get('score'), 'reasons': list(item.get('reasons', ['visual_query_faiss']))}
        focus_frames.append(focus)
        appended_lines.append('rank={rank} frame={frame} score={score} reasons={reasons} query={query!r} label={label!r} obj={obj}'.format(rank=next_rank, frame=frame_index, score=item.get('score'), reasons=','.join((str(value) for value in focus['reasons'])), query=item.get('query_phrase') or '', label=item.get('label') or '', obj=item.get('object_index')))
        next_rank += 1
    if not appended_lines:
        return
    retrieval_bundle['focus_frames'] = focus_frames
    bundle = graph_answer.get('llm_ready_bundle', {})
    if isinstance(bundle, dict):
        existing = [str(line).strip() for line in bundle.get('focus_frame_lines', []) if str(line).strip()]
        bundle['focus_frame_lines'] = [*existing, *appended_lines]
        structured = bundle.get('structured_graph_context', {})
        if isinstance(structured, dict):
            structured['focus_frames'] = focus_frames

def _object_display(row: dict[str, Any], *, include_similarity: bool=False) -> str:
    label = str(row.get('label') or 'object').strip() or 'object'
    seq = str(row.get('seq_id') or '?')
    frame = row.get('graph_frame_index')
    obj = row.get('object_index')
    elapsed = row.get('elapsed_sec_from_sequence_start')
    parts = [f'seq{seq}', f'frame={frame}', f'obj={obj}', f'label={label!r}']
    if elapsed is not None:
        parts.append(f't={_format_number(elapsed)}s')
    if include_similarity:
        parts.append(f"sim={_format_number(row.get('similarity'))}")
    attribute = str(row.get('visual_attribute') or '').strip()
    if attribute:
        parts.append(f'attr={_short_quote(attribute, limit=80)}')
    relations = [str(item).strip() for item in row.get('visual_relations', []) if str(item).strip()]
    if relations:
        parts.append('relations=' + ' | '.join((_short_quote(item, limit=54) for item in relations[:2])))
    neighbor_attributes = [str(item).strip() for item in row.get('hsgg_neighbor_attributes', []) if str(item).strip()]
    if neighbor_attributes:
        parts.append('neighbor_attrs=' + ' | '.join((_short_quote(item, limit=64) for item in neighbor_attributes[:2])))
    return '(' + ', '.join(parts) + ')'

def _compressed_visual_faiss_rows(visual_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    compression = visual_payload.get('compression', {})
    if isinstance(compression, dict) and compression.get('role_aware_compression_enabled'):
        return _role_aware_compressed_visual_faiss_rows(visual_payload)
    question = str(visual_payload.get('question') or '').strip()
    query_texts = [question]
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for group in visual_payload.get('groups', []) or []:
        phrase = str(group.get('query_phrase') or '').strip()
        if phrase:
            query_texts.append(phrase)
        for row in group.get('results', []) or []:
            key = (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))
            if key in seen:
                continue
            seen.add(key)
            item = dict(row)
            item['compression_score'] = _visual_faiss_compression_score(item, query_tokens=_visual_faiss_query_tokens(query_texts))
            item['generic_query_overlap_tokens'] = _visual_faiss_query_overlap_tokens(item, query_tokens=_visual_faiss_query_tokens(query_texts))
            rows.append((phrase, item))
    rows.sort(key=lambda pair: (-float(pair[1].get('compression_score') or 0.0), -_visual_faiss_query_relevance_score(pair[1], query_tokens=_visual_faiss_query_tokens(query_texts)), -(_optional_float(pair[1].get('similarity')) or 0.0), str(pair[1].get('label') or ''), _optional_int(pair[1].get('graph_frame_index')) or 0, _optional_int(pair[1].get('object_index')) or 0))
    compression = visual_payload.get('compression', {})
    if isinstance(compression, dict):
        limit = _optional_int(compression.get('selected_object_limit'))
    else:
        limit = None
    resolved_limit = max(1, limit or VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT)
    return rows[:resolved_limit]

def _role_aware_compressed_visual_faiss_rows(visual_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    compression = visual_payload.get('compression', {})
    limit = _optional_int(compression.get('selected_object_limit')) if isinstance(compression, dict) else None
    resolved_limit = max(1, limit or VISUAL_FAISS_COMPRESSED_OBJECT_LIMIT)
    question = str(visual_payload.get('question') or '').strip()
    question_ir = visual_payload.get('question_ir', {})
    if not isinstance(question_ir, dict):
        question_ir = {}
    role_request = visual_payload.get('role_request')
    if not isinstance(role_request, dict):
        role_request = _visual_faiss_role_aware_request(question=question, question_ir=question_ir)
    rows = _visual_faiss_role_aware_rows(visual_payload, role_request=role_request)
    if not rows:
        return []
    primary_rows, context_rows = _visual_faiss_role_aware_split_rows(rows, role_request=role_request)
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_keys: set[tuple[str, int | None, int | None]] = set()

    def append_rows(candidates: list[tuple[str, dict[str, Any]]]) -> None:
        for phrase, row in candidates:
            if len(selected) >= resolved_limit:
                return
            key = _visual_faiss_role_row_key(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            item = dict(row)
            item['compression_score'] = item.get('role_score')
            item['role_aware_compression_score'] = item.get('role_score')
            checks = item.get('role_checks', {})
            item['role_aware_selection_bucket'] = 'primary_target' if isinstance(checks, dict) and checks.get('is_primary_candidate') else 'constraint_context'
            item['generic_query_overlap_tokens'] = sorted(set(item.get('role_anchor_terms', []) or []) & set(role_request.get('question_terms', []) or []))[:20]
            selected.append((phrase, item))
    append_rows(primary_rows)
    append_rows(context_rows)
    if not selected:
        append_rows(rows)
    return selected

def _visual_faiss_compression_score(row: dict[str, Any], *, query_tokens: set[str]) -> float:
    similarity = _optional_float(row.get('similarity')) or 0.0
    score = similarity * 10.0
    if row.get('hsgg_visual_facts'):
        score += 1.2
    if row.get('visual_attribute'):
        score += 0.7
    relations = list(row.get('visual_relations', []) or [])
    if relations:
        score += min(0.8, 0.3 * len(relations))
    neighbor_attributes = list(row.get('hsgg_neighbor_attributes', []) or [])
    if neighbor_attributes:
        score += min(0.45, 0.15 * len(neighbor_attributes))
    score += _visual_faiss_query_relevance_score(row, query_tokens=query_tokens)
    return round(score, 6)

def _visual_faiss_query_relevance_score(row: dict[str, Any], *, query_tokens: set[str]) -> float:
    candidate_tokens = _visual_faiss_candidate_tokens(row)
    if not query_tokens or not candidate_tokens:
        return 0.0
    direct = len(query_tokens & candidate_tokens)
    soft = 0
    for query_token in query_tokens:
        if len(query_token) < 5:
            continue
        for candidate_token in candidate_tokens:
            if len(candidate_token) < 5:
                continue
            if query_token in candidate_token or candidate_token in query_token:
                soft += 1
                break
    return min(3.0, direct * 0.9 + soft * 0.35)

def _visual_faiss_query_overlap_tokens(row: dict[str, Any], *, query_tokens: set[str]) -> list[str]:
    return sorted(query_tokens & _visual_faiss_candidate_tokens(row))[:20]

def _visual_faiss_query_tokens(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(_visual_faiss_text_tokens(value))
    return tokens

def _visual_faiss_candidate_tokens(row: dict[str, Any]) -> set[str]:
    values: list[str] = [str(row.get('label') or '')]
    values.extend((str(item) for item in row.get('hsgg_visual_facts', []) or []))
    values.append(str(row.get('visual_attribute') or ''))
    values.extend((str(item) for item in row.get('visual_relations', []) or []))
    values.extend((str(item) for item in row.get('hsgg_neighbor_attributes', []) or []))
    return _visual_faiss_query_tokens(values)

def _visual_faiss_text_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall('[a-zA-Z][a-zA-Z0-9_-]{2,}', str(value or '').lower()):
        token = token.replace('_', '-').strip('-')
        if token and token not in _STOPWORDS:
            tokens.add(token)
    return tokens

def _compressed_visual_object_text(row: dict[str, Any]) -> str:
    label = str(row.get('label') or 'object').strip() or 'object'
    frame = row.get('graph_frame_index')
    obj = row.get('object_index')
    parts = [f"(seq{row.get('seq_id') or '?'}", f'frame={frame}', f'obj={obj}', f'label={label!r}', f"sim={_format_number(row.get('similarity'))}"]
    compression_score = row.get('compression_score')
    if compression_score is not None:
        parts.append(f'compression_score={_format_number(compression_score)}')
    return ', '.join(parts) + ')'

def _concise_object(row: dict[str, Any]) -> dict[str, Any]:
    return {'faiss_id': _optional_int(row.get('faiss_id')), 'object_faiss_id': _optional_int(row.get('object_faiss_id')), 'tracklet_faiss_id': _optional_int(row.get('tracklet_faiss_id')), 'seq_id': str(row.get('seq_id') or ''), 'frame_key': row.get('frame_key'), 'image_name': row.get('image_name'), 'graph_frame_index': _optional_int(row.get('graph_frame_index')), 'object_index': _optional_int(row.get('object_index')), 'entity_id': row.get('entity_id'), 'anchor_entity_id': row.get('anchor_entity_id'), 'tracklet_hit': bool(row.get('tracklet_hit')), 'entity_pack_recall_level': row.get('entity_pack_recall_level'), 'tracklet_observation_count': row.get('tracklet_observation_count'), 'tracklet_time_span': row.get('tracklet_time_span'), 'representative_labels': row.get('representative_labels', {}), 'representative_objects': row.get('representative_objects', []), 'tracklet_index_summary': row.get('tracklet_index_summary', {}), 'label': row.get('label'), 'score': row.get('score'), 'similarity': row.get('similarity'), 'elapsed_sec_from_sequence_start': row.get('elapsed_sec_from_sequence_start'), 'bbox': row.get('bbox'), 'image_path': row.get('image_path'), 'visual_attribute': row.get('visual_attribute'), 'visual_relations': row.get('visual_relations', []), 'visual_relation_records': row.get('visual_relation_records', []), 'hsgg_neighbor_attributes': row.get('hsgg_neighbor_attributes', []), 'hsgg_visual_facts': row.get('hsgg_visual_facts', []), 'hsgg_frame_context': row.get('hsgg_frame_context', [])}

def _entity_tracklet_representative_object(item: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    return {'faiss_id': _optional_int(item.get('faiss_id')), 'seq_id': str(item.get('seq_id') or ''), 'frame_key': item.get('frame_key'), 'image_name': item.get('image_name'), 'graph_frame_index': _optional_int(item.get('graph_frame_index')), 'object_index': _optional_int(item.get('object_index')), 'label': anchor.get('label') or item.get('label'), 'object_score': item.get('score'), 'bbox': anchor.get('bbox') or item.get('bbox'), 'image_path': anchor.get('image_path') or item.get('image_path'), 'cache_image_path': item.get('cache_image_path'), 'bbox_key': item.get('bbox_key'), 'elapsed_sec_from_sequence_start': item.get('elapsed_sec_from_sequence_start'), 'state_values': list(anchor.get('state_values', []) or []), 'raw_state_values': list(anchor.get('raw_state_values', []) or [])}

def _frame_attribute_lookup(frame_attributes: Any) -> dict[int, str]:
    if not isinstance(frame_attributes, dict):
        return {}
    lookup: dict[int, str] = {}
    for key, value in frame_attributes.items():
        index = _optional_int(key)
        text = _short_text(value, limit=120)
        if index is None or not text:
            continue
        lookup[index] = text
    return lookup

def _neighbor_attribute_text(*, index: int, label: str, attribute: str) -> str:
    display_label = _short_text(label, limit=48) or 'object'
    display_attribute = _short_text(attribute, limit=96)
    if not display_attribute:
        return ''
    return f'neighbor obj={index} label={display_label!r} attr={display_attribute}'

def _label_at(labels: Any, index: int) -> str:
    if not isinstance(labels, list) or index < 0 or index >= len(labels):
        return ''
    return _short_text(labels[index], limit=64)

def _bbox_values(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        values = [value.get(key) for key in ('x1', 'y1', 'x2', 'y2')]
    elif isinstance(value, (list, tuple)):
        values = list(value[:4])
    else:
        return None
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in values[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _bbox_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _bbox_spatial_text(box: tuple[float, float, float, float] | None) -> str:
    if box is None:
        return ''
    x1, y1, x2, y2 = box
    cx, cy = _bbox_center(box)
    return f'bbox=[{_format_number(x1)},{_format_number(y1)},{_format_number(x2)},{_format_number(y2)}] center=({_format_number(cx)},{_format_number(cy)}) area={_format_number(_bbox_area(box))}'

def _bbox_relative_text(*, anchor_bbox: tuple[float, float, float, float] | None, candidate_bbox: tuple[float, float, float, float] | None) -> str:
    if anchor_bbox is None or candidate_bbox is None:
        return ''
    ax, ay = _bbox_center(anchor_bbox)
    cx, cy = _bbox_center(candidate_bbox)
    horizontal = 'right_of_anchor' if cx > ax else 'left_of_anchor'
    vertical = 'below_anchor' if cy > ay else 'above_anchor'
    distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
    area_ratio = _bbox_area(candidate_bbox) / max(1.0, _bbox_area(anchor_bbox))
    return f'spatial={horizontal},{vertical} center_distance={_format_number(distance)} area_ratio={_format_number(area_ratio)}'

def _nearest_frame_object_indices(*, anchor_bbox: tuple[float, float, float, float] | None, boxes: Any, anchor_index: int) -> list[int]:
    if anchor_bbox is None or not isinstance(boxes, list):
        return []
    ax, ay = _bbox_center(anchor_bbox)
    scored: list[tuple[float, int]] = []
    for index, raw_box in enumerate(boxes):
        if index == anchor_index:
            continue
        box = _bbox_values(raw_box)
        if box is None:
            continue
        cx, cy = _bbox_center(box)
        distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
        scored.append((distance, index))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [index for _distance, index in scored]

def _ordered_unique_ints(values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered

def _hsgg_visual_fact_lines(*, label: str, object_index: int, attribute: str, relations: list[str], neighbor_attributes: list[str]) -> list[str]:
    facts: list[str] = []
    clean_label = _short_text(label, limit=64) or 'object'
    clean_attribute = _short_text(attribute, limit=120)
    if clean_attribute:
        facts.append(f'object obj={object_index} label={clean_label!r} attr={clean_attribute}')
    for relation in relations[:3]:
        clean_relation = _short_text(relation, limit=120)
        if clean_relation:
            facts.append(f'object obj={object_index} relation={clean_relation}')
    for neighbor_attribute in neighbor_attributes[:3]:
        clean_neighbor = _short_text(neighbor_attribute, limit=120)
        if clean_neighbor:
            facts.append(clean_neighbor)
    return _ordered_unique_texts(facts)[:6]

def _visual_faiss_fact_summary(row: dict[str, Any], *, max_facts: int) -> str:
    facts = [_short_text(item, limit=120) for item in row.get('hsgg_visual_facts', []) or [] if _short_text(item, limit=120)]
    if not facts:
        attribute = _short_text(row.get('visual_attribute'), limit=120)
        relations = [_short_text(item, limit=120) for item in row.get('visual_relations', []) or [] if _short_text(item, limit=120)]
        if attribute:
            facts.append(f'attr={attribute}')
        facts.extend((f'relation={relation}' for relation in relations[:3]))
    return '; '.join(_ordered_unique_texts(facts)[:max(1, max_facts)])

def _hsgg_root_for_metadata_row(row: dict[str, Any]) -> Path | None:
    path_values = [row.get('image_path'), row.get('image_path_abs'), row.get('cache_image_path'), row.get('cache_image_path_abs')]
    for value in path_values:
        if not value:
            continue
        candidate = Path(str(value).replace('\\', '/'))
        if not candidate.is_absolute():
            candidate = resolve_repo_input_path(candidate)
        parent = candidate.parent
        if parent.name.lower() == 'images':
            hsgg_root = parent.parent / 'hsgg_local_paths'
            if hsgg_root.exists():
                return hsgg_root
    return None

def _short_text(value: Any, *, limit: int) -> str:
    text = re.sub('\\s+', ' ', str(value or '')).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(' ', 1)[0].rstrip(' ,;:.') + '...'
    return text

def _short_quote(value: str, *, limit: int) -> str:
    text = _short_text(value, limit=limit)
    return repr(text)

def _visual_query_phrase_records(*, question: str, question_ir: dict[str, Any], limit: int) -> list[dict[str, str]]:
    raw_records: list[dict[str, str]] = []

    def add(value: Any, *, role: str, source: str) -> None:
        text = _clean_phrase(value)
        if not text:
            return
        raw_records.append({'text': text, 'rendered': f'a photo of {text}', 'role': role, 'source': source})
    add(question_ir.get('target_state'), role='primary_target', source='target_state')
    for value in _iter_text_values(question_ir.get('target_entities')):
        add(value, role='primary_target', source='target_entities')
    for value in _iter_text_values(question_ir.get('target_entity_aliases'))[:4]:
        add(value, role='primary_target', source='target_entity_aliases')
    add(question_ir.get('attribute_dimension'), role='target_attribute', source='attribute_dimension')
    for value in _iter_text_values(question_ir.get('attribute_dimension_aliases'))[:3]:
        add(value, role='target_attribute', source='attribute_dimension_aliases')
    for value in _iter_text_values(question_ir.get('event_constraints')):
        add(value, role='event_constraint', source='event_constraints')
    for value in _iter_text_values(question_ir.get('event_constraint_aliases'))[:3]:
        add(value, role='event_constraint', source='event_constraint_aliases')
    for value in _iter_text_values(question_ir.get('spatial_constraints')):
        add(value, role='spatial_constraint', source='spatial_constraints')
    for value in _iter_text_values(question_ir.get('spatial_constraint_aliases'))[:3]:
        add(value, role='spatial_constraint', source='spatial_constraint_aliases')
    for value in _iter_text_values(question_ir.get('hard_constraints')):
        add(value, role='hard_constraint', source='hard_constraints')
    for value in _iter_text_values(question_ir.get('hard_constraint_aliases'))[:3]:
        add(value, role='hard_constraint', source='hard_constraint_aliases')
    for value in _iter_text_values(question_ir.get('evidence_requirements')):
        add(value, role='support', source='evidence_requirements')
    for value in _iter_text_values(question_ir.get('support_aliases'))[:2]:
        add(value, role='support', source='support_aliases')
    add(_compact_question_visual_phrase(question), role='fallback', source='question')
    for value in _iter_text_values(question_ir.get('core_terms')):
        add(value, role='fallback', source='core_terms')
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in raw_records:
        normalized = ' '.join(record['text'].lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append(record)
        if len(records) >= limit:
            break
    return records

def _visual_query_phrases(*, question: str, question_ir: dict[str, Any], limit: int) -> list[str]:
    raw_phrases: list[str] = []

    def add(value: Any) -> None:
        text = _clean_phrase(value)
        if text:
            raw_phrases.append(text)
    add(question_ir.get('target_state'))
    for key in ('target_entities', 'target_entity_aliases', 'hard_constraints', 'hard_constraint_aliases', 'event_constraints', 'event_constraint_aliases', 'spatial_constraints', 'spatial_constraint_aliases'):
        values = _iter_text_values(question_ir.get(key))
        if key == 'target_entity_aliases':
            values = values[:4]
        elif key.endswith('_aliases'):
            values = values[:3]
        for value in values:
            add(value)
    for value in _iter_text_values(question_ir.get('attribute_dimension_aliases'))[:3]:
        add(value)
    add(_compact_question_visual_phrase(question))
    for value in _iter_text_values(question_ir.get('evidence_requirements')):
        add(value)
    for value in _iter_text_values(question_ir.get('support_aliases'))[:2]:
        add(value)
    for value in _iter_text_values(question_ir.get('core_terms')):
        add(value)
    phrases: list[str] = []
    seen: set[str] = set()
    for phrase in raw_phrases:
        normalized = ' '.join(phrase.lower().split())
        if not normalized or normalized in seen:
            continue
        rendered = f'a photo of {phrase}'
        seen.add(normalized)
        phrases.append(rendered)
        if len(phrases) >= limit:
            break
    return phrases

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target', 'target_state', 'relation'):
            if value.get(key):
                values.append(str(value[key]))
        nested = value.get('target_entities')
        if isinstance(nested, list):
            values.extend((str(item) for item in nested))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_iter_text_values(item))
        return values
    return [str(value)]

def _clean_phrase(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    text = re.sub('[_/]+', ' ', text)
    text = re.sub('\\b(visual observation of|timestamp of|timestamp when)\\b', '', text, flags=re.I)
    text = re.sub('\\b(robot|ego camera|you|i|we|saw|see|seen|seeing)\\b', '', text, flags=re.I)
    text = re.sub('\\s+', ' ', text).strip(' .,:;')
    if len(text) > 120:
        text = text[:120].rsplit(' ', 1)[0]
    return text

def _compact_question_visual_phrase(question: str) -> str:
    text = str(question or '').strip().rstrip('?')
    text = re.sub('^(did|do|does|was|were|is|are|what|when|where|which|how long|how)\\s+', '', text, flags=re.I)
    text = re.sub('\\b(you|i|we|see|saw|seen|did|do|does|was|were|is|are)\\b', '', text, flags=re.I)
    text = re.sub('\\s+', ' ', text).strip(' .,:;')
    return _clean_phrase(text)

def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _format_number(value: Any) -> str:
    try:
        return f'{float(value):.3f}'.rstrip('0').rstrip('.')
    except (TypeError, ValueError):
        return str(value)

def _ordered_unique_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered
