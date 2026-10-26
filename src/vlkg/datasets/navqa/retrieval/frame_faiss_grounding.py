from __future__ import annotations
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import expand_visual_faiss_graph_evidence
DEFAULT_FRAME_FAISS_INDEX_DIR = Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/frame_faiss')
DEFAULT_FRAME_FAISS_INDEX_NAME = 'frame_siglip_flat_ip'
_QUERY_IR_FIELDS = ('target_state', 'target_entities', 'attribute_dimension', 'temporal_relation', 'event_constraints', 'spatial_constraints', 'hard_constraints', 'evidence_requirements', 'graph_information_needs', 'qa_focus')

@dataclass(frozen=True)
class FrameFaissConfig:
    index_dir: Path = DEFAULT_FRAME_FAISS_INDEX_DIR
    index_name: str = DEFAULT_FRAME_FAISS_INDEX_NAME
    query_text_limit: int = 6
    top_k: int = 12
    search_k: int = 80
    min_score: float = 0.0
    same_sequence_only: bool = True
    cluster_max_frame_gap: int = 30
    cluster_max_sec_gap: float | None = None
    max_clusters: int = 8
    max_lines: int = 8

class FrameFaissGrounder:
    """Frame-level visual recall over a FAISS side index.

    This is a retrieval-only component. It uses text/image embeddings and frame
    metadata, but does not read answer labels or apply dataset-specific rules.
    """

    def __init__(self, *, index_dir: Path=DEFAULT_FRAME_FAISS_INDEX_DIR, index_name: str=DEFAULT_FRAME_FAISS_INDEX_NAME) -> None:
        import faiss
        self.index_dir = resolve_repo_input_path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f'{index_name}.faiss'
        self.metadata_db_path = self.index_dir / f'{index_name}.metadata.sqlite'
        self.manifest_path = self.index_dir / f'{index_name}.manifest.json'
        if not self.index_path.exists():
            raise FileNotFoundError(f'Missing frame FAISS index: {self.index_path}')
        if not self.metadata_db_path.exists():
            raise FileNotFoundError(f'Missing frame FAISS metadata DB: {self.metadata_db_path}')
        self.index = faiss.read_index(str(self.index_path))
        self.connection = sqlite3.connect(self.metadata_db_path)
        self.connection.row_factory = sqlite3.Row
        self._faiss = faiss
        self._text_encoder = None
        metadata_count = self.count_metadata_rows()
        if int(self.index.ntotal) != metadata_count:
            raise ValueError(f'Frame FAISS index and metadata DB size mismatch: index={self.index.ntotal}, metadata={metadata_count}')
        self.manifest = self._load_manifest()

    def close(self) -> None:
        self.connection.close()

    def count_metadata_rows(self) -> int:
        return int(self.connection.execute('SELECT COUNT(*) FROM frame_metadata').fetchone()[0])

    def describe(self) -> dict[str, Any]:
        return {'contract': 'navqa_frame_faiss_grounding_v1', 'index_path': str(self.index_path), 'metadata_db_path': str(self.metadata_db_path), 'manifest_path': str(self.manifest_path) if self.manifest_path.exists() else '', 'index_ntotal': int(self.index.ntotal), 'metadata_rows': self.count_metadata_rows(), 'dimension': int(getattr(self.index, 'd', 0) or 0), 'manifest': self.manifest}

    def build_query_recall(self, *, seq_id: str, question: str, question_ir: dict[str, Any] | None, config: FrameFaissConfig) -> dict[str, Any]:
        query_texts = frame_query_texts(question=question, question_ir=question_ir or {}, limit=max(1, config.query_text_limit))
        if not query_texts:
            return {'contract': 'navqa_frame_faiss_query_recall_v1', 'available': False, 'seq_id': str(seq_id), 'query_texts': [], 'groups': [], 'clusters': []}
        vectors = self._encode_texts(query_texts)
        groups: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int | None, str]] = set()
        for query_text, vector in zip(query_texts, vectors, strict=True):
            rows = self._search_by_vector(vector=vector, seq_id=str(seq_id), top_k=max(1, config.top_k), search_k=max(1, config.search_k), same_sequence_only=config.same_sequence_only, min_score=float(config.min_score))
            group_results: list[dict[str, Any]] = []
            for row in rows:
                key = (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), str(row.get('image_name') or ''))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                group_results.append(row)
                all_rows.append({**row, '_query_text': query_text})
            if group_results:
                groups.append({'query_text': query_text, 'results': [_concise_frame(row) for row in group_results]})
        clusters = cluster_frame_results(all_rows, max_frame_gap=max(0, int(config.cluster_max_frame_gap)), max_sec_gap=config.cluster_max_sec_gap, max_clusters=max(1, int(config.max_clusters)))
        lines = _render_frame_recall_lines(clusters, max_lines=max(1, int(config.max_lines)))
        visual_focus_frames = _frame_focus_frames_from_clusters(clusters, max_frames=max(1, int(config.max_lines)))
        return {'contract': 'navqa_frame_faiss_query_recall_v1', 'available': bool(groups), 'seq_id': str(seq_id), 'query_texts': query_texts, 'top_k': max(1, config.top_k), 'search_k': max(1, config.search_k), 'min_score': float(config.min_score), 'same_sequence_only': bool(config.same_sequence_only), 'cluster_max_frame_gap': max(0, int(config.cluster_max_frame_gap)), 'cluster_max_sec_gap': config.cluster_max_sec_gap, 'groups': groups, 'clusters': clusters, 'lines': lines, 'visual_focus_frames': visual_focus_frames}

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _search_by_vector(self, *, vector: np.ndarray, seq_id: str, top_k: int, search_k: int, same_sequence_only: bool, min_score: float) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)
        resolved_search_k = search_k if same_sequence_only else top_k
        distances, indices = self.index.search(query, resolved_search_k)
        ids = [int(value) for value in indices[0] if int(value) >= 0]
        metadata = self._metadata_by_ids(ids)
        rows: list[dict[str, Any]] = []
        for score, frame_id in zip(distances[0], indices[0], strict=True):
            frame_id = int(frame_id)
            if frame_id < 0:
                continue
            if float(score) < min_score:
                continue
            row = metadata.get(frame_id)
            if row is None:
                continue
            if same_sequence_only and str(row.get('seq_id') or '') != str(seq_id):
                continue
            item = dict(row)
            item['similarity'] = float(score)
            item['rank'] = len(rows) + 1
            rows.append(item)
            if len(rows) >= top_k:
                break
        return rows

    def _encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        if self._text_encoder is None:
            from vlkg.core.indexing.vision import VisionFeatureExtractor
            self._text_encoder = VisionFeatureExtractor(local_files_only=True)
        return self._text_encoder.encode_text_batch(texts)

    def _metadata_by_ids(self, faiss_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not faiss_ids:
            return {}
        placeholders = ','.join(('?' for _ in faiss_ids))
        rows = self.connection.execute(f'\n            SELECT faiss_id, raw_json\n            FROM frame_metadata\n            WHERE faiss_id IN ({placeholders})\n            ', faiss_ids).fetchall()
        return {int(row['faiss_id']): json.loads(str(row['raw_json'])) for row in rows}

def frame_query_texts(*, question: str, question_ir: dict[str, Any], limit: int) -> list[str]:
    raw_texts: list[str] = [str(question or '').strip()]
    for field in _QUERY_IR_FIELDS:
        raw_texts.extend(_iter_text_values(question_ir.get(field)))
    texts: list[str] = []
    seen: set[str] = set()
    for raw in raw_texts:
        text = ' '.join(str(raw or '').split())
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        texts.append(f'a video frame of {text}')
        if len(texts) >= limit:
            break
    return texts

def cluster_frame_results(rows: list[dict[str, Any]], *, max_frame_gap: int, max_sec_gap: float | None, max_clusters: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    sortable = []
    for row in rows:
        frame = _optional_int(row.get('graph_frame_index'))
        timestamp = _optional_float(row.get('elapsed_sec_from_sequence_start'))
        if frame is None and timestamp is None:
            continue
        sortable.append(row)
    sortable.sort(key=lambda row: (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')) is None, _optional_int(row.get('graph_frame_index')) or 0, _optional_float(row.get('elapsed_sec_from_sequence_start')) or 0.0, -float(row.get('similarity') or 0.0)))
    clusters: list[list[dict[str, Any]]] = []
    for row in sortable:
        if not clusters or not _frame_row_close(clusters[-1][-1], row, max_frame_gap=max_frame_gap, max_sec_gap=max_sec_gap):
            clusters.append([row])
        else:
            clusters[-1].append(row)
    payloads = [_cluster_payload(index, rows) for index, rows in enumerate(clusters, start=1)]
    payloads.sort(key=lambda item: (-float(item.get('max_similarity') or 0.0), -int(item.get('frame_count') or 0), str(item.get('seq_id') or ''), int(item.get('start_frame') or 0)))
    for rank, item in enumerate(payloads, start=1):
        item['rank'] = rank
    return payloads[:max_clusters]

def attach_frame_faiss_evidence(graph_answer: dict[str, Any], frame_payload: dict[str, Any], *, expose_as_visual: bool=True, include_context_lines: bool=True) -> dict[str, Any]:
    """Attach full-frame FAISS recall as supplemental visual evidence."""
    lines = _frame_faiss_evidence_lines(frame_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_payloads = list(updated.get('frame_faiss_payloads', []) or [])
    existing_payloads.append(frame_payload)
    existing_frame_lines = [str(line).strip() for line in updated.get('frame_faiss_evidence_lines', []) if str(line).strip()]
    merged_frame_lines = _ordered_unique_texts([*existing_frame_lines, *lines])
    updated['frame_faiss_grounding'] = frame_payload
    updated['frame_faiss_payloads'] = existing_payloads
    updated['frame_faiss_evidence_lines'] = merged_frame_lines
    if expose_as_visual:
        existing_visual_lines = [str(line).strip() for line in updated.get('visual_faiss_evidence_lines', []) if str(line).strip()]
        merged_visual_lines = _ordered_unique_texts([*existing_visual_lines, *lines])
        updated['visual_faiss_grounding'] = frame_payload
        updated['visual_faiss_evidence_lines'] = merged_visual_lines
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    if lines and include_context_lines:
        context_lines.extend(['Frame FAISS evidence:'])
        context_lines.extend((f'- {line}' for line in lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = updated.get('structured_graph_context', {})
    if isinstance(structured, dict):
        structured = dict(structured)
        structured['frame_faiss_grounding'] = frame_payload
        structured['frame_faiss_payloads'] = existing_payloads
        structured['frame_faiss_evidence_lines'] = merged_frame_lines
        if expose_as_visual:
            structured['visual_faiss_grounding'] = frame_payload
            structured['visual_faiss_evidence_lines'] = updated.get('visual_faiss_evidence_lines', [])
        updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    _append_frame_focus_frames(graph_answer, frame_payload)
    return graph_answer

def fuse_frame_faiss_evidence(graph_answer: dict[str, Any], frame_payload: dict[str, Any]) -> dict[str, Any]:
    """Promote full-frame FAISS recall into the main evidence candidate pool."""
    lines = _frame_faiss_evidence_lines(frame_payload)
    if not lines:
        return attach_frame_faiss_evidence(graph_answer, frame_payload)
    graph_answer = attach_frame_faiss_evidence(graph_answer, frame_payload)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) if str(line).strip()]
    updated['evidence_lines'] = _ordered_unique_texts([*existing_evidence, *lines])
    fusion_facts = _frame_faiss_candidate_facts(frame_payload, evidence_offset=len(existing_evidence))
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    candidate_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*candidate_facts, *fusion_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    fusion_lines = [_frame_faiss_aggregation_line(index, fact) for index, fact in enumerate(fusion_facts, start=1)]
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) if str(line).strip()]
    updated['evidence_aggregation_lines'] = _ordered_unique_texts([*existing_aggregation_lines, *fusion_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Frame FAISS fused evidence:'])
    context_lines.extend((f'- {line}' for line in lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = dict(updated.get('structured_graph_context', {}))
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    structured['frame_faiss_fusion'] = {'contract': 'navqa_frame_faiss_fused_evidence_v1', 'fused_evidence_count': len(lines), 'fused_candidate_fact_count': len(fusion_facts)}
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for index, line in enumerate(lines, start=1):
            digest.append({'rank': next_rank, 'source_record_id': f'frame_faiss:{index}', 'source_kind': 'frame_faiss_cluster', 'summary': line, 'score': _frame_faiss_line_score(index, frame_payload)})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def frame_faiss_payload_as_graph_anchor_payload(frame_payload: dict[str, Any], *, max_anchors: int=12) -> dict[str, Any]:
    """Adapt frame-level recall to the generic graph-anchor expansion shape."""
    groups: list[dict[str, Any]] = []
    anchors_added = 0
    for cluster in frame_payload.get('clusters', []) or []:
        if not isinstance(cluster, dict):
            continue
        best = cluster.get('best_frame', {})
        if not isinstance(best, dict):
            continue
        frame_index = _optional_int(best.get('graph_frame_index'))
        if frame_index is None:
            continue
        query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
        query_text = query_texts[0] if query_texts else ''
        row = dict(best)
        row['graph_frame_index'] = frame_index
        row['object_index'] = None
        row['label'] = 'full_frame'
        row['similarity'] = cluster.get('max_similarity')
        row['elapsed_sec_from_sequence_start'] = best.get('elapsed_sec_from_sequence_start')
        groups.append({'query_phrase': query_text, 'results': [row]})
        anchors_added += 1
        if anchors_added >= max(1, int(max_anchors)):
            break
    return {'contract': 'navqa_frame_faiss_graph_anchor_payload_v1', 'available': bool(groups), 'seq_id': frame_payload.get('seq_id'), 'query_texts': list(frame_payload.get('query_texts', []) or []), 'groups': groups, 'lines': _frame_faiss_evidence_lines(frame_payload), 'visual_focus_frames': [item for item in frame_payload.get('visual_focus_frames', []) or [] if isinstance(item, dict)][:max(1, int(max_anchors))], 'source_frame_faiss': frame_payload}

def expand_frame_faiss_graph_evidence(graph_answer: dict[str, Any], frame_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, frame_radius: int=8, max_items: int=12) -> dict[str, Any]:
    """Use frame FAISS anchors to promote nearby canonical graph evidence."""
    graph_answer = attach_frame_faiss_evidence(graph_answer, frame_payload, expose_as_visual=False, include_context_lines=False)
    anchor_payload = frame_faiss_payload_as_graph_anchor_payload(frame_payload, max_anchors=max(1, int(max_items)))
    before_bundle = graph_answer.get('llm_ready_bundle', {})
    before_visual_lines = []
    before_payloads = []
    before_grounding = None
    if isinstance(before_bundle, dict):
        before_visual_lines = list(before_bundle.get('visual_faiss_evidence_lines', []) or [])
        before_payloads = list(before_bundle.get('visual_faiss_payloads', []) or [])
        before_grounding = before_bundle.get('visual_faiss_grounding')
    graph_answer = expand_visual_faiss_graph_evidence(graph_answer, anchor_payload, graph=graph, frame_radius=frame_radius, max_items=max_items)
    bundle = graph_answer.get('llm_ready_bundle', {})
    if isinstance(bundle, dict):
        bundle['visual_faiss_evidence_lines'] = before_visual_lines
        bundle['visual_faiss_payloads'] = before_payloads
        if before_grounding is None:
            bundle.pop('visual_faiss_grounding', None)
        else:
            bundle['visual_faiss_grounding'] = before_grounding
        expansion = bundle.pop('visual_faiss_anchor_expansion', None)
        if isinstance(expansion, dict):
            expansion = dict(expansion)
            expansion['contract'] = 'navqa_frame_faiss_graph_anchor_expansion_v1'
            expansion['source_contract'] = 'experimental_visual_faiss_graph_anchor_expansion_v1'
            bundle['frame_faiss_anchor_expansion'] = expansion
        structured = bundle.get('structured_graph_context', {})
        if isinstance(structured, dict):
            structured['visual_faiss_evidence_lines'] = before_visual_lines
            if before_grounding is None:
                structured.pop('visual_faiss_grounding', None)
            else:
                structured['visual_faiss_grounding'] = before_grounding
            if 'visual_faiss_anchor_expansion' in structured:
                expansion = structured.pop('visual_faiss_anchor_expansion')
                if isinstance(expansion, dict):
                    expansion = dict(expansion)
                    expansion['contract'] = 'navqa_frame_faiss_graph_anchor_expansion_v1'
                    expansion['source_contract'] = 'experimental_visual_faiss_graph_anchor_expansion_v1'
                structured['frame_faiss_anchor_expansion'] = expansion
        _rewrite_frame_anchor_bundle_labels(bundle)
    return graph_answer

def _rewrite_frame_anchor_bundle_labels(bundle: dict[str, Any]) -> None:
    evidence_lines = [_frame_anchor_text(str(line)) for line in bundle.get('evidence_lines', []) or []]
    bundle['evidence_lines'] = evidence_lines
    aggregation_lines = [_frame_anchor_text(str(line)) for line in bundle.get('evidence_aggregation_lines', []) or []]
    bundle['evidence_aggregation_lines'] = aggregation_lines
    aggregation = bundle.get('evidence_aggregation', {})
    if isinstance(aggregation, dict):
        facts = []
        for fact in aggregation.get('candidate_facts', []) or []:
            if not isinstance(fact, dict):
                facts.append(fact)
                continue
            updated = dict(fact)
            if updated.get('fact_type') == 'visual_faiss_anchored_graph_evidence':
                updated['fact_type'] = 'frame_faiss_anchored_graph_evidence'
            if updated.get('source') == 'visual_faiss_graph_anchor':
                updated['source'] = 'frame_faiss_graph_anchor'
            if 'fact_text' in updated:
                updated['fact_text'] = _frame_anchor_text(str(updated['fact_text']))
            if 'visual_anchor' in updated:
                updated['frame_faiss_anchor'] = updated['visual_anchor']
            facts.append(updated)
        aggregation['candidate_facts'] = facts
    structured = bundle.get('structured_graph_context', {})
    if isinstance(structured, dict):
        structured['evidence_lines'] = evidence_lines
        structured['evidence_aggregation_lines'] = aggregation_lines
        structured['evidence_aggregation'] = aggregation

def _frame_anchor_text(text: str) -> str:
    return text.replace('group=visual_faiss_anchor:', 'group=frame_faiss_anchor:').replace('Visual FAISS anchored graph evidence', 'Frame FAISS anchored graph evidence').replace('visual_anchor=', 'frame_faiss_anchor=').replace("anchor_label='full_frame'", "anchor_label='frame'")

def _frame_row_close(previous: dict[str, Any], current: dict[str, Any], *, max_frame_gap: int, max_sec_gap: float | None) -> bool:
    if str(previous.get('seq_id') or '') != str(current.get('seq_id') or ''):
        return False
    previous_frame = _optional_int(previous.get('graph_frame_index'))
    current_frame = _optional_int(current.get('graph_frame_index'))
    if previous_frame is not None and current_frame is not None:
        if abs(current_frame - previous_frame) <= max_frame_gap:
            return True
    if max_sec_gap is None:
        return False
    previous_sec = _optional_float(previous.get('elapsed_sec_from_sequence_start'))
    current_sec = _optional_float(current.get('elapsed_sec_from_sequence_start'))
    if previous_sec is None or current_sec is None:
        return False
    return abs(current_sec - previous_sec) <= float(max_sec_gap)

def _cluster_payload(index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [_optional_int(row.get('graph_frame_index')) for row in rows if _optional_int(row.get('graph_frame_index')) is not None]
    times = [_optional_float(row.get('elapsed_sec_from_sequence_start')) for row in rows if _optional_float(row.get('elapsed_sec_from_sequence_start')) is not None]
    similarities = [float(row.get('similarity') or 0.0) for row in rows]
    best = max(rows, key=lambda row: float(row.get('similarity') or 0.0))
    return {'cluster_index': index, 'seq_id': best.get('seq_id'), 'frame_count': len(rows), 'start_frame': min(frames) if frames else None, 'end_frame': max(frames) if frames else None, 'start_elapsed_sec': min(times) if times else None, 'end_elapsed_sec': max(times) if times else None, 'max_similarity': max(similarities) if similarities else None, 'mean_similarity': sum(similarities) / len(similarities) if similarities else None, 'best_frame': _concise_frame(best), 'query_texts': _ordered_unique_texts((str(row.get('_query_text') or '').strip() for row in rows if str(row.get('_query_text') or '').strip())), 'frames': [_concise_frame(row) for row in rows]}

def _render_frame_recall_lines(clusters: list[dict[str, Any]], *, max_lines: int) -> list[str]:
    lines: list[str] = []
    for cluster in clusters:
        text = _frame_faiss_cluster_line(cluster)
        if not text:
            continue
        lines.append(text)
        if len(lines) >= max_lines:
            break
    return lines

def _frame_faiss_evidence_lines(frame_payload: dict[str, Any]) -> list[str]:
    payload_lines = [str(line).strip() for line in frame_payload.get('lines', []) if str(line).strip()]
    if payload_lines:
        return _ordered_unique_texts(payload_lines)
    return _ordered_unique_texts((_frame_faiss_cluster_line(cluster) for cluster in frame_payload.get('clusters', []) or [] if isinstance(cluster, dict)))

def _frame_faiss_cluster_line(cluster: dict[str, Any]) -> str:
    best = cluster.get('best_frame', {})
    if not isinstance(best, dict):
        best = {}
    seq = str(cluster.get('seq_id') or best.get('seq_id') or '?')
    start_frame = _optional_int(cluster.get('start_frame'))
    end_frame = _optional_int(cluster.get('end_frame'))
    start_sec = _optional_float(cluster.get('start_elapsed_sec'))
    end_sec = _optional_float(cluster.get('end_elapsed_sec'))
    best_frame = _optional_int(best.get('graph_frame_index'))
    query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
    frame_count = _optional_int(cluster.get('frame_count'))
    parts = ['frame_faiss_cluster', f'seq{seq}', f'frames={_frame_range_text(start_frame, end_frame)}']
    if start_sec is not None or end_sec is not None:
        parts.append(f't={_time_range_text(start_sec, end_sec)}')
    if best_frame is not None:
        parts.append(f'best_frame={best_frame}')
    if frame_count is not None:
        parts.append(f'count={frame_count}')
    parts.append(f"max_sim={_format_number(cluster.get('max_similarity'))}")
    if query_texts:
        parts.append('queries=' + ' | '.join((_short_quote(text, limit=96) for text in query_texts[:3])))
    return ' '.join((part for part in parts if part))

def _frame_focus_frames_from_clusters(clusters: list[dict[str, Any]], *, max_frames: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cluster in clusters:
        best = cluster.get('best_frame', {})
        if not isinstance(best, dict):
            continue
        frame_index = _optional_int(best.get('graph_frame_index'))
        if frame_index is None or frame_index in seen:
            continue
        seen.add(frame_index)
        query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
        frames.append({'frame_index': frame_index, 'timestamp_sec': best.get('elapsed_sec_from_sequence_start'), 'scene_id': None, 'score': round(float(best.get('similarity') or 0.0), 6), 'reasons': ['frame_faiss_query_recall'], 'query_text': query_texts[0] if query_texts else '', 'cluster_rank': cluster.get('rank'), 'cluster_start_frame': cluster.get('start_frame'), 'cluster_end_frame': cluster.get('end_frame')})
        if len(frames) >= max_frames:
            break
    return frames

def _append_frame_focus_frames(graph_answer: dict[str, Any], frame_payload: dict[str, Any]) -> None:
    visual_frames = [item for item in frame_payload.get('visual_focus_frames', []) if isinstance(item, dict)]
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
        focus = {'rank': next_rank, 'frame_index': frame_index, 'frame_rank': frame_index, 'timestamp_sec': item.get('timestamp_sec'), 'scene_id': item.get('scene_id'), 'score': item.get('score'), 'reasons': list(item.get('reasons', ['frame_faiss_query_recall']))}
        focus_frames.append(focus)
        appended_lines.append('rank={rank} frame={frame} score={score} reasons={reasons} query={query!r} cluster={cluster}'.format(rank=next_rank, frame=frame_index, score=item.get('score'), reasons=','.join((str(value) for value in focus['reasons'])), query=item.get('query_text') or '', cluster=item.get('cluster_rank')))
        next_rank += 1
    if not appended_lines:
        return
    retrieval_bundle['focus_frames'] = focus_frames
    graph_answer['retrieval_bundle'] = retrieval_bundle
    bundle = graph_answer.get('llm_ready_bundle', {})
    if isinstance(bundle, dict):
        existing = [str(line).strip() for line in bundle.get('focus_frame_lines', []) if str(line).strip()]
        bundle['focus_frame_lines'] = [*existing, *appended_lines]
        structured = bundle.get('structured_graph_context', {})
        if isinstance(structured, dict):
            structured['focus_frames'] = focus_frames

def _frame_faiss_candidate_facts(frame_payload: dict[str, Any], *, evidence_offset: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    evidence_rank = evidence_offset + 1
    for cluster in frame_payload.get('clusters', []) or []:
        if not isinstance(cluster, dict):
            continue
        line = _frame_faiss_cluster_line(cluster)
        if not line:
            continue
        best = cluster.get('best_frame', {})
        if not isinstance(best, dict):
            best = {}
        query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
        fact_text = f"Frame FAISS recalled a visual cluster in seq{cluster.get('seq_id') or best.get('seq_id') or '?'} at frames {_frame_range_text(cluster.get('start_frame'), cluster.get('end_frame'))}"
        if query_texts:
            fact_text += f' for query {query_texts[0]!r}'
        facts.append({'fact_type': 'frame_faiss_cluster_recall', 'fact_text': fact_text, 'binding_strength': 'visual_frame_similarity', 'support_ranks': [evidence_rank], 'support_examples': [line], 'source': 'frame_faiss', 'query_texts': query_texts, 'start_frame': _optional_int(cluster.get('start_frame')), 'end_frame': _optional_int(cluster.get('end_frame')), 'best_frame_index': _optional_int(best.get('graph_frame_index')), 'similarity': cluster.get('max_similarity')})
        evidence_rank += 1
    return facts

def _frame_faiss_aggregation_line(index: int, fact: dict[str, Any]) -> str:
    parts = [f'group=frame_faiss:{index}', f"binding={fact.get('binding_strength')}", f"support_ranks={','.join((str(value) for value in fact.get('support_ranks', [])))}", str(fact.get('fact_text') or '')]
    examples = [str(value).strip() for value in fact.get('support_examples', []) if str(value).strip()]
    if examples:
        parts.append('examples=' + ' | '.join(examples[:2]))
    return ' '.join((part for part in parts if part))

def _frame_faiss_line_score(index: int, frame_payload: dict[str, Any]) -> float | None:
    current = 0
    for cluster in frame_payload.get('clusters', []) or []:
        current += 1
        if current == index:
            return _optional_float(cluster.get('max_similarity'))
    return None

def _concise_frame(row: dict[str, Any]) -> dict[str, Any]:
    return {'rank': row.get('rank'), 'similarity': _optional_float(row.get('similarity')), 'faiss_id': _optional_int(row.get('faiss_id')), 'seq_id': str(row.get('seq_id') or ''), 'cam': row.get('cam'), 'stride': row.get('stride'), 'image_name': row.get('image_name'), 'image_path': row.get('image_path'), 'sampled_index': _optional_int(row.get('sampled_index')), 'graph_frame_index': _optional_int(row.get('graph_frame_index')), 'source_frame': _optional_int(row.get('source_frame')), 'timestamp_unix': row.get('timestamp_unix'), 'timestamp_iso': row.get('timestamp_iso'), 'elapsed_sec_from_sequence_start': row.get('elapsed_sec_from_sequence_start')}

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_iter_text_values(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for item in value.values():
            texts.extend(_iter_text_values(item))
        return texts
    return [str(value)]

def _ordered_unique_texts(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        normalized = re.sub('\\s+', ' ', text).lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result

def _frame_range_text(start: Any, end: Any) -> str:
    start_frame = _optional_int(start)
    end_frame = _optional_int(end)
    if start_frame is None and end_frame is None:
        return '?'
    if start_frame == end_frame or end_frame is None:
        return str(start_frame)
    if start_frame is None:
        return str(end_frame)
    return f'{start_frame}-{end_frame}'

def _time_range_text(start: float | None, end: float | None) -> str:
    if start is None and end is None:
        return '?'
    if start is None:
        return f'{_format_number(end)}s'
    if end is None or abs(float(end) - float(start)) < 1e-09:
        return f'{_format_number(start)}s'
    return f'{_format_number(start)}-{_format_number(end)}s'

def _format_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return '?'
    text = f'{number:.3f}'.rstrip('0').rstrip('.')
    return text or '0'

def _short_quote(value: str, *, limit: int) -> str:
    text = ' '.join(str(value or '').split())
    if len(text) > limit:
        text = text[:max(0, limit - 1)].rstrip() + '...'
    return repr(text)

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
