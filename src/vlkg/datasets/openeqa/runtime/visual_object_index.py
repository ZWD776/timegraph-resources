"""OpenEQA-owned graph-bound object visual index and episode-local search."""
from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TextIO
import numpy as np
from vlkg.datasets.openeqa.retrieval.evidence_router import UNIT_CANDIDATE_SET, UNIT_CANONICAL_ENTITY, UNIT_OWNER_ATTRIBUTE, UNIT_SCENE_FRAME, visual_query_intents
OBJECT_VISUAL_INDEX_CONTRACT = 'openeqa_object_visual_graph_index.v2'
OBJECT_VISUAL_SEARCH_CONTRACT = 'openeqa_episode_scoped_visual_search.v1'
_STREAM_CHUNK_SIZE = 1024 * 1024
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_LEXICAL_STOPWORDS = {'a', 'an', 'are', 'in', 'is', 'left', 'object', 'of', 'on', 'photo', 'right', 'the', 'to', 'what', 'where'}

def validate_object_visual_index_manifest(manifest: Mapping[str, Any], *, expected_data_source: str, expected_graph_manifest_run_id: str | None, require_full_coverage: bool=False, forbidden_image_root_markers: Iterable[str]=()) -> list[str]:
    """Validate that a visual index is compatible with the requested graph run."""
    issues: list[str] = []
    if manifest.get('contract') != OBJECT_VISUAL_INDEX_CONTRACT:
        issues.append(f"Visual index contract mismatch: expected={OBJECT_VISUAL_INDEX_CONTRACT} actual={manifest.get('contract')}")
    if manifest.get('status') != 'passed':
        issues.append(f"Visual index status is not passed: {manifest.get('status')}")
    if str(manifest.get('data_source') or '') != expected_data_source:
        issues.append(f"Visual index data source mismatch: expected={expected_data_source} actual={manifest.get('data_source')}")
    actual_run_id = str(manifest.get('graph_manifest_run_id') or '')
    expected_run_id = str(expected_graph_manifest_run_id or '')
    if expected_run_id and actual_run_id != expected_run_id:
        issues.append(f'Visual index graph manifest mismatch: expected_run_id={expected_run_id} actual_run_id={actual_run_id}')
    ntotal = int(manifest.get('ntotal') or 0)
    metadata_rows = int(manifest.get('metadata_rows') or 0)
    if ntotal <= 0 or metadata_rows != ntotal:
        issues.append(f'Visual index row mismatch: ntotal={ntotal} metadata_rows={metadata_rows}')
    image_root = str(manifest.get('image_root') or '')
    normalized_image_root = image_root.replace('\\', '/').casefold()
    for marker in forbidden_image_root_markers:
        normalized_marker = str(marker).replace('\\', '/').casefold()
        if normalized_marker and normalized_marker in normalized_image_root:
            issues.append(f'Visual index image_root contains forbidden marker {marker!r}: {image_root}')
    totals = manifest.get('totals') or {}
    indexed = int(totals.get('indexed') or 0)
    if indexed != ntotal:
        issues.append(f'Visual index total mismatch: indexed={indexed} ntotal={ntotal}')
    if require_full_coverage:
        hsgg_objects = int(totals.get('hsgg_objects') or 0)
        selected_objects = int(totals.get('selected_visual_anchor_objects', hsgg_objects) or 0)
        skipped_objects = int(totals.get('skipped_non_visual_anchor_objects') or 0)
        missing_embeddings = int(totals.get('missing_embedding') or 0)
        if hsgg_objects <= 0:
            issues.append('Full visual index has no HSGG objects')
        if selected_objects != hsgg_objects:
            issues.append(f'Full visual index does not cover all HSGG objects: selected={selected_objects} hsgg_objects={hsgg_objects}')
        if skipped_objects != 0:
            issues.append(f'Full visual index skipped non-anchor objects: skipped_non_visual_anchor_objects={skipped_objects}')
        if missing_embeddings != 0:
            issues.append(f'Full visual index has missing embeddings: {missing_embeddings}')
        if indexed != hsgg_objects:
            issues.append(f'Full visual index row coverage mismatch: indexed={indexed} hsgg_objects={hsgg_objects}')
    return issues

@dataclass(frozen=True)
class ObservationBinding:
    observation_id: str
    frame_name: str
    frame_index: int
    object_index: int
    source_entity_id: str
    canonical_entity_id: str
    scene_id: str
    image_path: str
    label: str
    binding_kind: str
    binding_confidence: float

    @property
    def graph_bound(self) -> bool:
        return bool(self.canonical_entity_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class GraphObservationBindingIndex:
    """Frame/object binding into HSGG observations and optional canonical IDs."""

    def __init__(self, *, observations: dict[tuple[int, int], dict[str, Any]], canonical_entity_by_source: dict[str, str]) -> None:
        self._observations = observations
        self._canonical_entity_by_source = canonical_entity_by_source

    @classmethod
    def from_graphs(cls, *, temporal_graph_path: str | Path, canonical_graph_path: str | Path) -> 'GraphObservationBindingIndex':
        observations: dict[tuple[int, int], dict[str, Any]] = {}
        with Path(temporal_graph_path).open('r', encoding='utf-8') as handle:
            for row in iter_json_array_objects(handle, 'observations'):
                frame_index = _optional_int(row.get('frame_index'))
                object_index = _optional_int(row.get('object_index'))
                source_entity_id = _clean_text(row.get('entity_id'))
                if frame_index is None or object_index is None or (not source_entity_id):
                    continue
                key = (frame_index, object_index)
                if key in observations:
                    observations[key] = {'ambiguous': True}
                else:
                    observations[key] = row
        canonical_entity_by_source: dict[str, str] = {}
        with Path(canonical_graph_path).open('r', encoding='utf-8') as handle:
            for row in iter_json_array_objects(handle, 'entity_nodes'):
                canonical_entity_id = _clean_text(row.get('entity_id'))
                if not canonical_entity_id:
                    continue
                attributes = row.get('attributes')
                attributes = attributes if isinstance(attributes, dict) else {}
                source_ids = [*_text_values(row.get('source_entity_ids')), *_text_values(attributes.get('source_entity_ids')), *_text_values(attributes.get('member_entity_ids'))]
                for source_entity_id in source_ids:
                    previous = canonical_entity_by_source.get(source_entity_id)
                    if previous is not None and previous != canonical_entity_id:
                        canonical_entity_by_source[source_entity_id] = ''
                    elif previous is None:
                        canonical_entity_by_source[source_entity_id] = canonical_entity_id
        return cls(observations=observations, canonical_entity_by_source=canonical_entity_by_source)

    @property
    def observation_count(self) -> int:
        return sum((1 for row in self._observations.values() if not row.get('ambiguous')))

    @property
    def canonical_source_count(self) -> int:
        return sum((1 for value in self._canonical_entity_by_source.values() if value))

    def bind(self, *, frame_index: int, object_index: int) -> ObservationBinding | None:
        row = self._observations.get((int(frame_index), int(object_index)))
        if not isinstance(row, dict) or row.get('ambiguous'):
            return None
        source_entity_id = _clean_text(row.get('entity_id'))
        if not source_entity_id:
            return None
        canonical_entity_id = self._canonical_entity_by_source.get(source_entity_id, '')
        return ObservationBinding(observation_id=_clean_text(row.get('observation_id')), frame_name=_clean_text(row.get('frame_name')), frame_index=int(frame_index), object_index=int(object_index), source_entity_id=source_entity_id, canonical_entity_id=canonical_entity_id, scene_id=_clean_text(row.get('scene_id')), image_path=_clean_text(row.get('image_path')), label=_clean_text(row.get('label')), binding_kind='exact_frame_object_to_canonical_entity' if canonical_entity_id else 'hsgg_observation_without_canonical_entity', binding_confidence=1.0 if canonical_entity_id else 0.45)

class ObjectVisualSearchIndex:
    """Exact similarity search over only the requested episode's object rows."""

    def __init__(self, *, index_dir: str | Path, index_name: str='openeqa_object_siglip_flat_ip', text_encoder: Callable[[Sequence[str]], Sequence[np.ndarray]] | None=None, episode_cache_size: int=8) -> None:
        import faiss
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f'{index_name}.faiss'
        self.metadata_path = self.index_dir / f'{index_name}.metadata.sqlite'
        self.manifest_path = self.index_dir / f'{index_name}.manifest.json'
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding='utf-8-sig'))
            for attribute, key in (('index_path', 'index_path'), ('metadata_path', 'metadata_path')):
                if manifest.get(key):
                    value = Path(manifest[key])
                    if not value.is_absolute():
                        value = self.index_dir / value
                    setattr(self, attribute, value)
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        self.index = faiss.read_index(str(self.index_path))
        self.connection = sqlite3.connect(self.metadata_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._faiss = faiss
        self._text_encoder = text_encoder
        self._episode_cache_size = max(1, int(episode_cache_size))
        self._episode_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]] = OrderedDict()
        self._group_cache: OrderedDict[tuple[str, str], tuple[np.ndarray, list[dict[str, Any]]]] = OrderedDict()
        metadata_count = int(self.connection.execute('SELECT COUNT(*) FROM object_metadata').fetchone()[0])
        if metadata_count != int(self.index.ntotal):
            raise ValueError(f'FAISS/metadata mismatch: index={self.index.ntotal}, metadata={metadata_count}')

    def close(self) -> None:
        with self._lock:
            self._episode_cache.clear()
            self._group_cache.clear()
            self.connection.close()

    def episode_size(self, episode_history: str) -> int:
        with self._lock:
            row = self.connection.execute('SELECT COUNT(*) FROM object_metadata WHERE episode_history = ?', (episode_history,)).fetchone()
            return int(row[0]) if row is not None else 0

    def search_evidence_demand(self, *, episode_history: str, evidence_demand: dict[str, Any], top_k: int=8, min_score: float=0.0) -> dict[str, Any]:
        with self._lock:
            intents = visual_query_intents(evidence_demand)
            rendered = [f"a photo of {row['text']}" for row in intents]
            vectors = self._encode_texts(rendered)
            groups: list[dict[str, Any]] = []
            seen_by_role: dict[tuple[str, str], set[str]] = {}
            for intent, query_text, vector in zip(intents, rendered, vectors, strict=True):
                role = str(intent.get('role') or 'context')
                requested_unit = str(intent.get('evidence_unit') or 'object_crop')
                if requested_unit == UNIT_SCENE_FRAME:
                    hits, stats = self._search_grouped_vectors_with_stats(episode_history=episode_history, vector=vector, top_k=top_k, min_score=min_score, lexical_query=str(intent.get('text') or ''), group_kind=UNIT_SCENE_FRAME)
                elif requested_unit in {UNIT_CANONICAL_ENTITY, UNIT_OWNER_ATTRIBUTE, UNIT_CANDIDATE_SET}:
                    hits, stats = self._search_grouped_vectors_with_stats(episode_history=episode_history, vector=vector, top_k=max(top_k, 32) if requested_unit == UNIT_CANDIDATE_SET else top_k, min_score=min_score, lexical_query=str(intent.get('text') or ''), group_kind=UNIT_CANONICAL_ENTITY)
                else:
                    hits, stats = self._search_vector_with_stats(episode_history=episode_history, vector=vector, top_k=top_k, min_score=min_score, lexical_query=str(intent.get('text') or ''))
                    requested_unit = 'object_crop'
                seen = seen_by_role.setdefault((role, requested_unit), set())
                unique_hits: list[dict[str, Any]] = []
                for hit in hits:
                    hit_key = _visual_hit_key(hit, retrieved_unit=requested_unit)
                    if hit_key in seen:
                        continue
                    seen.add(hit_key)
                    unique_hits.append(hit)
                if unique_hits:
                    groups.append({'query_intent': intent, 'rendered_query': query_text, 'retrieved_unit': requested_unit, 'episode_candidate_count': stats['episode_candidate_count'], 'score_threshold_count': stats['score_threshold_count'], 'hits': unique_hits})
            coframe_group = self._anchor_coframe_expansion_group(episode_history=episode_history, evidence_demand=evidence_demand, groups=groups, max_anchor_frames=max(1, min(8, int(top_k))))
            if coframe_group is not None:
                groups.append(coframe_group)
            unit_counts = Counter((str(group.get('retrieved_unit') or 'unknown') for group in groups))
            return {'contract': OBJECT_VISUAL_SEARCH_CONTRACT, 'available': bool(groups), 'episode_history': episode_history, 'uses_gold_answer': False, 'episode_scoped_search': True, 'global_topk_then_filter': False, 'allows_unbound_visual_observations': True, 'unbound_observations_are_graph_facts': False, 'query_intent_count': len(intents), 'coframe_expansion_hit_count': len(coframe_group.get('hits', [])) if coframe_group else 0, 'episode_candidate_count': self.episode_size(episode_history), 'retrieved_unit_group_counts': dict(sorted(unit_counts.items())), 'groups': groups}

    def _anchor_coframe_expansion_group(self, *, episode_history: str, evidence_demand: dict[str, Any], groups: list[dict[str, Any]], max_anchor_frames: int) -> dict[str, Any] | None:
        if not evidence_demand.get('anchor_terms'):
            return None
        anchor_hits: list[dict[str, Any]] = []
        for group in groups:
            intent = group.get('query_intent') if isinstance(group, dict) else None
            if not isinstance(intent, dict) or intent.get('role') != 'anchor':
                continue
            anchor_hits.extend((hit for hit in group.get('hits', []) if isinstance(hit, dict)))
        anchor_hits.sort(key=lambda hit: (-float(hit.get('similarity') or 0.0), int(hit.get('graph_frame_index') or 0), int(hit.get('faiss_id') or 0)))
        frame_scores: dict[int, float] = {}
        anchor_ids: set[int] = set()
        for hit in anchor_hits:
            frame_index = _optional_int(hit.get('graph_frame_index'))
            faiss_id = _optional_int(hit.get('faiss_id'))
            if frame_index is None or faiss_id is None:
                continue
            anchor_ids.add(faiss_id)
            if frame_index not in frame_scores:
                if len(frame_scores) >= max_anchor_frames:
                    continue
                frame_scores[frame_index] = float(hit.get('similarity') or 0.0)
            else:
                frame_scores[frame_index] = max(frame_scores[frame_index], float(hit.get('similarity') or 0.0))
        if not frame_scores:
            return None
        _, _, metadata = self._episode_vectors(episode_history)
        expanded: list[dict[str, Any]] = []
        for faiss_id, raw_row in metadata.items():
            frame_index = _optional_int(raw_row.get('graph_frame_index'))
            if frame_index not in frame_scores or faiss_id in anchor_ids:
                continue
            if not raw_row.get('bbox'):
                continue
            row = dict(raw_row)
            row['similarity'] = frame_scores[frame_index]
            row['anchor_recall_similarity'] = frame_scores[frame_index]
            row['retrieval_kind'] = 'anchor_coframe_expansion'
            row['search_scope'] = 'episode_exact_anchor_frame'
            row.setdefault('binding_kind', 'exact_frame_object_to_canonical_entity' if row.get('canonical_entity_id') else 'hsgg_observation_without_canonical_entity')
            row.setdefault('binding_confidence', 1.0 if row.get('canonical_entity_id') else 0.45)
            expanded.append(row)
        expanded.sort(key=lambda row: (int(row.get('graph_frame_index') or 0), -float(row.get('binding_confidence') or 0.0), int(row.get('object_index') or 0)))
        for rank, row in enumerate(expanded, start=1):
            row['rank'] = rank
        if not expanded:
            return None
        return {'query_intent': {'text': 'objects co-visible with recalled spatial anchors', 'role': 'spatial_target', 'evidence_unit': 'spatial_pair', 'source': 'anchor_coframe_expansion', 'target_terms': list(evidence_demand.get('target_terms', [])), 'anchor_terms': list(evidence_demand.get('anchor_terms', []))}, 'rendered_query': '', 'retrieved_unit': 'same_frame_object_set', 'episode_candidate_count': self.episode_size(episode_history), 'score_threshold_count': len(expanded), 'hits': expanded}

    def search_vector(self, *, episode_history: str, vector: np.ndarray, top_k: int, min_score: float, search_k: int | None=None) -> list[dict[str, Any]]:
        del search_k
        with self._lock:
            hits, _ = self._search_vector_with_stats(episode_history=episode_history, vector=vector, top_k=top_k, min_score=min_score)
            return hits

    def _search_vector_with_stats(self, *, episode_history: str, vector: np.ndarray, top_k: int, min_score: float, lexical_query: str='') -> tuple[list[dict[str, Any]], dict[str, int]]:
        ids, matrix, metadata = self._episode_vectors(episode_history)
        if ids.size == 0:
            return ([], {'episode_candidate_count': 0, 'score_threshold_count': 0})
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != matrix.shape[1]:
            raise ValueError(f'Visual embedding dimension mismatch: query={query.shape[1]}, index={matrix.shape[1]}')
        self._faiss.normalize_L2(query)
        scores = matrix @ query[0]
        eligible = np.flatnonzero(scores >= float(min_score))
        ranking_scores = np.asarray(scores, dtype=np.float32).copy()
        if lexical_query:
            for position, faiss_id in enumerate(ids):
                ranking_scores[position] += _lexical_label_bonus(lexical_query, metadata[int(faiss_id)].get('label'))
        if eligible.size:
            order = eligible[np.argsort(-ranking_scores[eligible], kind='stable')]
        else:
            order = np.asarray([], dtype=np.int64)
        selected = order[:max(1, int(top_k))]
        hits: list[dict[str, Any]] = []
        for rank, position in enumerate(selected, start=1):
            faiss_id = int(ids[int(position)])
            row = dict(metadata[faiss_id])
            row['similarity'] = float(scores[int(position)])
            row['retrieval_score'] = float(ranking_scores[int(position)])
            row['lexical_label_bonus'] = float(ranking_scores[int(position)] - scores[int(position)])
            row['rank'] = rank
            row['search_scope'] = 'episode_exact'
            row.setdefault('binding_kind', 'exact_frame_object_to_canonical_entity' if row.get('canonical_entity_id') else 'hsgg_observation_without_canonical_entity')
            row.setdefault('binding_confidence', 1.0 if row.get('canonical_entity_id') else 0.45)
            hits.append(row)
        return (hits, {'episode_candidate_count': int(ids.size), 'score_threshold_count': int(eligible.size)})

    def _search_grouped_vectors_with_stats(self, *, episode_history: str, vector: np.ndarray, top_k: int, min_score: float, lexical_query: str, group_kind: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
        matrix, rows = self._grouped_episode_vectors(episode_history, group_kind=group_kind)
        if not rows:
            return ([], {'episode_candidate_count': 0, 'score_threshold_count': 0})
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != matrix.shape[1]:
            raise ValueError(f'Visual embedding dimension mismatch: query={query.shape[1]}, index={matrix.shape[1]}')
        self._faiss.normalize_L2(query)
        scores = matrix @ query[0]
        eligible = np.flatnonzero(scores >= float(min_score))
        ranking_scores = np.asarray(scores, dtype=np.float32).copy()
        if lexical_query:
            for position, row in enumerate(rows):
                ranking_scores[position] += _lexical_label_bonus(lexical_query, row.get('aggregate_labels') or row.get('label'))
        order = eligible[np.argsort(-ranking_scores[eligible], kind='stable')] if eligible.size else np.asarray([], dtype=np.int64)
        selected = order[:max(1, int(top_k))]
        hits: list[dict[str, Any]] = []
        for rank, position in enumerate(selected, start=1):
            row = dict(rows[int(position)])
            row['similarity'] = float(scores[int(position)])
            row['retrieval_score'] = float(ranking_scores[int(position)])
            row['lexical_label_bonus'] = float(ranking_scores[int(position)] - scores[int(position)])
            row['rank'] = rank
            row['search_scope'] = 'episode_grouped'
            hits.append(row)
        return (hits, {'episode_candidate_count': len(rows), 'score_threshold_count': int(eligible.size)})

    def _grouped_episode_vectors(self, episode_history: str, *, group_kind: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        cache_key = (episode_history, group_kind)
        cached = self._group_cache.pop(cache_key, None)
        if cached is not None:
            self._group_cache[cache_key] = cached
            return cached
        ids, object_matrix, metadata = self._episode_vectors(episode_history)
        buckets: OrderedDict[str, list[int]] = OrderedDict()
        for position, faiss_id in enumerate(ids):
            row = metadata[int(faiss_id)]
            if group_kind == UNIT_SCENE_FRAME:
                frame_index = _optional_int(row.get('graph_frame_index'))
                key = f'frame:{frame_index}' if frame_index is not None else ''
            else:
                key = _clean_text(row.get('canonical_entity_id'))
            if key:
                buckets.setdefault(key, []).append(position)
        vectors: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for key, positions in buckets.items():
            group_vector = np.asarray(object_matrix[positions].mean(axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(group_vector))
            if norm <= 0.0:
                continue
            vectors.append(group_vector / norm)
            member_rows = [metadata[int(ids[position])] for position in positions]
            representative = max(member_rows, key=lambda row: (float(row.get('object_score') or 0.0), float(row.get('binding_confidence') or 0.0)))
            row = dict(representative)
            labels = _ordered_unique((_clean_text(item.get('label')) for item in member_rows))
            row['aggregate_labels'] = ' '.join(labels[:24])
            row['aggregate_member_count'] = len(member_rows)
            row['aggregate_frame_count'] = len({_optional_int(item.get('graph_frame_index')) for item in member_rows if _optional_int(item.get('graph_frame_index')) is not None})
            if group_kind == UNIT_SCENE_FRAME:
                row['canonical_entity_id'] = ''
                row['retrieval_kind'] = 'scene_frame_object_embedding_aggregate'
                row['scene_frame_key'] = key
            else:
                row['retrieval_kind'] = 'canonical_entity_tracklet_embedding_aggregate'
                row['canonical_entity_id'] = key
            rows.append(row)
        matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else np.empty((0, int(self.index.d)), dtype=np.float32)
        cached = (matrix, rows)
        self._group_cache[cache_key] = cached
        while len(self._group_cache) > self._episode_cache_size * 2:
            self._group_cache.popitem(last=False)
        return cached

    def _episode_vectors(self, episode_history: str) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
        with self._lock:
            cached = self._episode_cache.pop(episode_history, None)
            if cached is not None:
                self._episode_cache[episode_history] = cached
                return cached
            rows = self.connection.execute('SELECT faiss_id, raw_json FROM object_metadata WHERE episode_history = ? ORDER BY faiss_id', (episode_history,)).fetchall()
            ids = np.asarray([int(row['faiss_id']) for row in rows], dtype=np.int64)
            metadata = {int(row['faiss_id']): json.loads(str(row['raw_json'])) for row in rows}
            if ids.size:
                try:
                    matrix = np.asarray(self.index.reconstruct_batch(ids), dtype=np.float32)
                except (AttributeError, TypeError):
                    matrix = np.vstack([np.asarray(self.index.reconstruct(int(value)), dtype=np.float32) for value in ids])
                self._faiss.normalize_L2(matrix)
            else:
                matrix = np.empty((0, int(self.index.d)), dtype=np.float32)
            cached = (ids, matrix, metadata)
            self._episode_cache[episode_history] = cached
            while len(self._episode_cache) > self._episode_cache_size:
                self._episode_cache.popitem(last=False)
            return cached

    def encode_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Reuse the index text encoder for downstream semantic score contracts."""
        with self._lock:
            return list(self._encode_texts(texts))

    def _encode_texts(self, texts: Sequence[str]) -> Sequence[np.ndarray]:
        if not texts:
            return []
        if self._text_encoder is None:
            from vlkg.core.indexing.vision import VisionFeatureExtractor
            extractor = VisionFeatureExtractor(local_files_only=True)
            self._text_encoder = extractor.encode_text_batch
        return self._text_encoder(texts)

def iter_json_array_objects(handle: TextIO, property_name: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = f'"{property_name}"'
    buffer = ''
    found_property = False
    in_array = False
    finished = False
    while not finished:
        chunk = handle.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            finished = True
        buffer += chunk
        while True:
            if not found_property:
                index = buffer.find(marker)
                if index < 0:
                    buffer = buffer[-max(0, len(marker) - 1):]
                    break
                buffer = buffer[index + len(marker):]
                found_property = True
            if not in_array:
                index = buffer.find('[')
                if index < 0:
                    if finished:
                        raise ValueError(f'JSON array not found: {property_name}')
                    break
                buffer = buffer[index + 1:]
                in_array = True
            buffer = buffer.lstrip()
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

def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _text_values(item)]
    return []

def _lexical_label_bonus(query: Any, label: Any) -> float:
    query_text = _clean_text(query).lower()
    label_text = _clean_text(label).lower()
    if not query_text or not label_text:
        return 0.0
    query_tokens = {token[:-1] if token.endswith('s') and len(token) > 3 else token for token in _TOKEN_PATTERN.findall(query_text) if token not in _LEXICAL_STOPWORDS}
    label_tokens = {token[:-1] if token.endswith('s') and len(token) > 3 else token for token in _TOKEN_PATTERN.findall(label_text) if token not in _LEXICAL_STOPWORDS}
    synonym_groups = ({'stair', 'staircase', 'step'}, {'tv', 'television'}, {'sofa', 'couch'}, {'fridge', 'refrigerator'}, {'picture', 'image', 'painting'})
    for group in synonym_groups:
        if query_tokens & group:
            query_tokens.update(group)
        if label_tokens & group:
            label_tokens.update(group)
    if not query_tokens or not label_tokens:
        return 0.0
    overlap_ratio = len(query_tokens & label_tokens) / max(1, len(label_tokens))
    phrase_bonus = 0.12 if label_text in query_text else 0.0
    return min(0.28, 0.16 * overlap_ratio + phrase_bonus)

def _clean_text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        normalized = cleaned.casefold()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result

def _visual_hit_key(hit: dict[str, Any], *, retrieved_unit: str) -> str:
    if retrieved_unit == UNIT_SCENE_FRAME:
        return f"frame:{hit.get('graph_frame_index')}"
    if retrieved_unit in {UNIT_CANONICAL_ENTITY, UNIT_OWNER_ATTRIBUTE, UNIT_CANDIDATE_SET}:
        entity_id = _clean_text(hit.get('canonical_entity_id'))
        if entity_id:
            return f'entity:{entity_id}'
    return f"object:{hit.get('faiss_id')}"
