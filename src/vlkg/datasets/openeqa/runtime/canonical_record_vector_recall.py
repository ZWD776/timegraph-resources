"""OpenEQA-owned canonical record vector recall fork.

The initial implementation is behavior-equivalent to the NAVQA-compatible
source and is isolated here so later OpenEQA retrieval work cannot affect it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence
import numpy as np
from vlkg.core.graph_retrieval.canonical_retrieval_types import CanonicalProjection, GraphRecordView
from vlkg.core.memory.types import MemoryQuery
DEFAULT_VECTOR_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_CACHE_SCHEMA_VERSION = 1
VECTOR_PROFILE = 'canonical_vector_hybrid_v1'
VECTOR_GRAPH_HYBRID_PROFILE = 'canonical_vector_graph_hybrid_v2'
VECTOR_STRUCTURED_PROFILE = 'canonical_vector_structured_v3'
VECTOR_UNION_PROFILE = 'canonical_vector_union_v4'
VECTOR_EVIDENCE_PROFILE = 'canonical_vector_evidence_v5'
DEFAULT_VECTOR_TOP_K = 48
DEFAULT_VECTOR_MIN_SIMILARITY = 0.2
_MODEL_CACHE: dict[str, Any] = {}
_INTERNAL_ID_PATTERN = re.compile('(?:[a-z][a-z0-9-]*-v\\d+__|scene:|entity:|episode:|transition:|stable:)[a-z0-9_:\\-.]+', flags=re.IGNORECASE)

class TextEncoder(Protocol):

    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool, convert_to_numpy: bool, show_progress_bar: bool) -> np.ndarray:
        ...

@dataclass(frozen=True)
class VectorRecallHit:
    record_id: str
    source_kind: str
    similarity: float
    rank: int

@dataclass
class CanonicalRecordVectorIndex:
    record_ids: tuple[str, ...]
    source_kinds: tuple[str, ...]
    group_ids: tuple[str, ...]
    semantic_texts: tuple[str, ...]
    embeddings: np.ndarray
    model_name: str = DEFAULT_VECTOR_MODEL
    cache_status: str = 'memory'
    _encoder: TextEncoder | None = field(default=None, repr=False)

    @classmethod
    def build_or_load(cls, projection: CanonicalProjection, *, cache_path: str | Path | None=None, model_name: str=DEFAULT_VECTOR_MODEL, encoder: TextEncoder | None=None, batch_size: int=128) -> CanonicalRecordVectorIndex:
        records = sorted(projection.records.values(), key=lambda item: item.source_record_id)
        record_ids = tuple((record.source_record_id for record in records))
        source_kinds = tuple((record.source_kind for record in records))
        group_ids = tuple((record_semantic_group_id(record) for record in records))
        semantic_texts = tuple((record_semantic_text(record) for record in records))
        fingerprint = semantic_text_fingerprint(record_ids=record_ids, semantic_texts=semantic_texts, model_name=model_name)
        resolved_cache = Path(cache_path) if cache_path is not None else None
        if resolved_cache is not None:
            cached = cls._load_cache(resolved_cache, record_ids=record_ids, source_kinds=source_kinds, group_ids=group_ids, semantic_texts=semantic_texts, fingerprint=fingerprint, model_name=model_name, encoder=encoder)
            if cached is not None:
                return cached
        resolved_encoder = encoder or load_local_text_encoder(model_name)
        embeddings = encode_texts(resolved_encoder, semantic_texts, batch_size=batch_size)
        index = cls(record_ids=record_ids, source_kinds=source_kinds, group_ids=group_ids, semantic_texts=semantic_texts, embeddings=embeddings, model_name=model_name, cache_status='built', _encoder=resolved_encoder)
        if resolved_cache is not None:
            index._write_cache(resolved_cache, fingerprint=fingerprint)
        return index

    @classmethod
    def _load_cache(cls, path: Path, *, record_ids: tuple[str, ...], source_kinds: tuple[str, ...], group_ids: tuple[str, ...], semantic_texts: tuple[str, ...], fingerprint: str, model_name: str, encoder: TextEncoder | None) -> CanonicalRecordVectorIndex | None:
        manifest_path = vector_cache_manifest_path(path)
        if not path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('schema_version') != VECTOR_CACHE_SCHEMA_VERSION:
                return None
            if str(manifest.get('model_name') or '') != model_name:
                return None
            if str(manifest.get('semantic_text_fingerprint') or '') != fingerprint:
                return None
            with np.load(path, allow_pickle=False) as payload:
                embeddings = np.asarray(payload['embeddings'], dtype=np.float32)
                cached_ids = tuple((str(value) for value in payload['record_ids'].tolist()))
                cached_kinds = tuple((str(value) for value in payload['source_kinds'].tolist()))
            if cached_ids != record_ids or cached_kinds != source_kinds:
                return None
            if embeddings.ndim != 2 or embeddings.shape[0] != len(record_ids):
                return None
            return cls(record_ids=record_ids, source_kinds=source_kinds, group_ids=group_ids, semantic_texts=semantic_texts, embeddings=embeddings, model_name=model_name, cache_status='loaded', _encoder=encoder)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path, *, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        manifest_path = vector_cache_manifest_path(path)
        temp_manifest_path = manifest_path.with_name(f'.{manifest_path.name}.{os.getpid()}.tmp')
        try:
            with temp_path.open('wb') as handle:
                np.savez_compressed(handle, embeddings=np.asarray(self.embeddings, dtype=np.float32), record_ids=np.asarray(self.record_ids, dtype=np.str_), source_kinds=np.asarray(self.source_kinds, dtype=np.str_))
            manifest = {'schema_version': VECTOR_CACHE_SCHEMA_VERSION, 'model_name': self.model_name, 'semantic_text_fingerprint': fingerprint, 'record_count': len(self.record_ids), 'embedding_dimension': int(self.embeddings.shape[1]), 'normalization': 'l2', 'similarity': 'cosine_via_matrix_dot', 'index_scope': 'hsgg_derived_canonical_graph_records', 'uses_images_or_crops': False, 'uses_visual_faiss': False, 'uses_gold_answers': False}
            temp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(temp_path, path)
            os.replace(temp_manifest_path, manifest_path)
        finally:
            for candidate in (temp_path, temp_manifest_path):
                if candidate.exists():
                    candidate.unlink()

    def search(self, query_text: str, *, top_k: int=DEFAULT_VECTOR_TOP_K, min_similarity: float=DEFAULT_VECTOR_MIN_SIMILARITY, encoder: TextEncoder | None=None) -> list[VectorRecallHit]:
        if not query_text.strip() or top_k <= 0 or (not self.record_ids):
            return []
        resolved_encoder = encoder or self._encoder or load_local_text_encoder(self.model_name)
        query_embedding = encode_texts(resolved_encoder, (query_text,), batch_size=1)[0]
        similarities = self.embeddings @ query_embedding
        candidate_count = min(len(self.record_ids), max(top_k * 4, top_k))
        if candidate_count == len(self.record_ids):
            candidate_indices = np.arange(len(self.record_ids))
        else:
            candidate_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
        ordered_indices = sorted((int(index) for index in candidate_indices), key=lambda index: (-float(similarities[index]), self.record_ids[index]))
        eligible_indices = [index for index in ordered_indices if float(similarities[index]) >= min_similarity]
        selected_indices = _diversified_record_indices(eligible_indices, source_kinds=self.source_kinds, group_ids=self.group_ids, top_k=top_k)
        selected_indices.sort(key=lambda index: (-float(similarities[index]), self.record_ids[index]))
        return [VectorRecallHit(record_id=self.record_ids[index], source_kind=self.source_kinds[index], similarity=float(similarities[index]), rank=rank) for rank, index in enumerate(selected_indices, start=1)]

def load_local_text_encoder(model_name: str=DEFAULT_VECTOR_MODEL) -> TextEncoder:
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('canonical vector recall requires sentence-transformers') from exc
    try:
        model = SentenceTransformer(model_name, device='cpu', local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f'local embedding model is unavailable: {model_name}') from exc
    _MODEL_CACHE[model_name] = model
    return model

def encode_texts(encoder: TextEncoder, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        values = encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        values = encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, np.finfo(np.float32).eps)
    return embeddings.astype(np.float32, copy=False)

def vector_cache_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.manifest.json')

def semantic_text_fingerprint(*, record_ids: Sequence[str], semantic_texts: Sequence[str], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(VECTOR_CACHE_SCHEMA_VERSION).encode('utf-8'))
    digest.update(model_name.encode('utf-8'))
    for record_id, semantic_text in zip(record_ids, semantic_texts, strict=True):
        digest.update(b'\x00')
        digest.update(record_id.encode('utf-8'))
        digest.update(b'\x00')
        digest.update(semantic_text.encode('utf-8'))
    return digest.hexdigest()

def canonical_vector_query_text(query: MemoryQuery) -> str:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    raw_question = str(metadata.get('raw_question') or '').strip()
    if not raw_question:
        raw_question = str(query.query_text or '').split(' ; ', maxsplit=1)[0].strip()
    ir = metadata.get('question_ir')
    semantic_ir = ir if isinstance(ir, dict) else {}
    field_labels = (('target', 'target_entities'), ('target description', 'target_description'), ('target description', 'target_descriptors'), ('state', 'target_state'), ('attribute dimension', 'attribute_dimension'), ('spatial', 'spatial_constraints'), ('event', 'event_constraints'), ('constraint', 'hard_constraints'))
    parts = [raw_question] if raw_question else []
    for label, key in field_labels:
        values = [value for value in _flatten_semantic_values(semantic_ir.get(key)) if value.strip().lower() not in {'i', 'me', 'we', 'you', 'entity', 'object', 'thing', 'unknown'}]
        if values:
            parts.append(f"{label}: {'; '.join(values)}")
    return ' '.join(_dedupe_text(parts))

def record_semantic_text(record: GraphRecordView) -> str:
    metadata = record.metadata
    kind = record.source_kind
    parts: list[str] = [kind.replace('_', ' ')]
    if kind == 'scene':
        parts.extend(_semantic_fields(metadata, 'label', 'scene_kind_hint', 'representative_static_labels', 'representative_dynamic_labels', 'salient_labels', 'semantic_alias_terms', 'entry_static_labels', 'exit_static_labels'))
    elif kind == 'entity':
        parts.extend(_semantic_fields(metadata, 'label', 'canonical_label', 'semantic_group', 'semantic_alias_terms', 'attributes', 'raw_visual_attributes'))
    elif kind == 'entity_attribute':
        label = _first_text(metadata.get('label'), metadata.get('canonical_label'))
        if label:
            parts.append(f'entity {label}')
        attr_key = str(metadata.get('attr_key') or 'attribute').strip()
        attr_value = str(metadata.get('attr_value') or '').strip()
        if attr_value:
            parts.append(f'{attr_key}: {attr_value}')
        parts.extend(_semantic_fields(metadata, 'raw_visual_attribute', 'semantic_group', 'semantic_alias_terms'))
    elif kind == 'stable_identity':
        parts.extend(_semantic_fields(metadata, 'display_name', 'canonical_label', 'labels_seen', 'attributes', 'raw_visual_attributes', 'signature_tokens'))
    elif kind == 'episode':
        episode_kind = str(metadata.get('episode_kind') or 'episode').lower()
        parts.append(episode_kind)
        if episode_kind in {'relation', 'relation_observed_burst'}:
            subject = _first_text(metadata.get('subject_entity_label'), metadata.get('primary_entity_label'))
            predicate = _first_text(metadata.get('raw_predicate'), metadata.get('predicate'))
            object_label = _first_text(metadata.get('object_entity_label'))
            if subject and predicate and object_label:
                parts.append(f'{subject} {predicate} {object_label}')
        else:
            label = _first_text(metadata.get('primary_entity_label'), metadata.get('label'))
            if label:
                parts.append(f'entity {label}')
            attr_key = str(metadata.get('attr_key') or '').strip()
            attr_value = str(metadata.get('attr_value') or '').strip()
            if attr_value:
                parts.append(f"{attr_key or 'state'}: {attr_value}")
            parts.extend(_semantic_fields(metadata, 'attributes'))
    elif kind == 'transition_event':
        parts.extend(_semantic_fields(metadata, 'event_type', 'entity_label'))
        previous = str(metadata.get('previous_state_value') or '').strip()
        current = str(metadata.get('current_state_value') or '').strip()
        if previous or current:
            parts.append(f"state changed from {previous or 'unknown'} to {current or 'unknown'}")
        subject = _first_text(metadata.get('subject_entity_label'))
        predicate = _first_text(metadata.get('raw_predicate'), metadata.get('predicate'))
        object_label = _first_text(metadata.get('object_entity_label'))
        if subject and predicate and object_label:
            parts.append(f'{subject} {predicate} {object_label}')
    else:
        parts.extend(_semantic_fields(metadata, 'label', 'attr_key', 'attr_value'))
    cleaned = [_clean_semantic_text(value) for value in _dedupe_text(parts)]
    semantic_text = '. '.join((value for value in cleaned if value))
    if semantic_text:
        return semantic_text[:1600]
    return _clean_semantic_text(record.summary)[:1600]

def record_semantic_group_id(record: GraphRecordView) -> str:
    metadata = record.metadata
    if record.source_kind == 'entity_attribute':
        owner = str(metadata.get('entity_id') or '').strip()
        if owner:
            return f'entity_attribute:{owner}'
    if record.source_kind == 'episode':
        episode_kind = str(metadata.get('episode_kind') or '').upper()
        if episode_kind in {'RELATION', 'RELATION_OBSERVED_BURST'}:
            subject = str(metadata.get('subject_entity_id') or '').strip()
            predicate = str(metadata.get('predicate') or '').strip()
            object_id = str(metadata.get('object_entity_id') or '').strip()
            if subject or object_id:
                return f'relation:{subject}:{predicate}:{object_id}'
        owner = str(metadata.get('primary_entity_id') or '').strip()
        attr_key = str(metadata.get('attr_key') or episode_kind).strip()
        if owner:
            return f'episode:{owner}:{attr_key}'
    if record.entity_ids:
        return f"{record.source_kind}:{'|'.join(record.entity_ids)}"
    return f'{record.source_kind}:{record.source_record_id}'

def _diversified_record_indices(ordered_indices: Sequence[int], *, source_kinds: Sequence[str], group_ids: Sequence[str], top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    base_kind_quotas = {'episode': 16, 'entity_attribute': 12, 'entity': 6, 'transition_event': 4, 'stable_identity': 4, 'scene': 2}
    scale = top_k / 48.0
    kind_quotas = {kind: max(1, round(quota * scale)) for kind, quota in base_kind_quotas.items()}
    selected: list[int] = []
    selected_set: set[int] = set()
    group_counts: dict[str, int] = {}

    def add_index(index: int, *, max_per_group: int) -> bool:
        if index in selected_set or len(selected) >= top_k:
            return False
        group_id = group_ids[index]
        if group_counts.get(group_id, 0) >= max_per_group:
            return False
        selected.append(index)
        selected_set.add(index)
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
        return True
    for kind, quota in kind_quotas.items():
        added = 0
        for index in ordered_indices:
            if source_kinds[index] != kind:
                continue
            if add_index(index, max_per_group=3):
                added += 1
            if added >= quota or len(selected) >= top_k:
                break
    for index in ordered_indices:
        if len(selected) >= top_k:
            break
        add_index(index, max_per_group=3)
    if len(selected) < top_k:
        for index in ordered_indices:
            if len(selected) >= top_k:
                break
            add_index(index, max_per_group=top_k)
    return selected

def _semantic_fields(metadata: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_flatten_semantic_values(metadata.get(key)))
    return values

def _flatten_semantic_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, dict):
        values: list[str] = []
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key.endswith('_id') or normalized_key in {'category', 'evidence_question_type', 'kind', 'must_bind_to', 'operation', 'question_route', 'question_route_id', 'role', 'routing'}:
                continue
            nested_values = _flatten_semantic_values(nested)
            if nested_values:
                if normalized_key in {'relation', 'target', 'target_entities', 'target_state', 'text'}:
                    values.extend(nested_values)
                else:
                    values.append(f"{str(key).replace('_', ' ')} {' '.join(nested_values)}")
        return values
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_semantic_values(item))
        return values
    normalized = str(value).strip()
    return [normalized] if normalized else []

def _first_text(*values: Any) -> str:
    for value in values:
        flattened = _flatten_semantic_values(value)
        if flattened:
            return flattened[0]
    return ''

def _dedupe_text(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = ' '.join(str(value or '').split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result

def _clean_semantic_text(value: str) -> str:
    without_ids = _INTERNAL_ID_PATTERN.sub(' ', str(value or ''))
    return ' '.join(without_ids.replace('__', ' ').split()).strip(' .,:;-')
