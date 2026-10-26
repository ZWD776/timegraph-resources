from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.navqa.retrieval.frame_faiss_grounding import FrameFaissConfig, FrameFaissGrounder, attach_frame_faiss_evidence, expand_frame_faiss_graph_evidence, frame_query_texts, fuse_frame_faiss_evidence
from vlkg.core.temporal.pathing import resolve_repo_input_path
DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR = Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/merged_frame_faiss')
DEFAULT_MERGED_FRAME_FAISS_INDEX_NAME = 'merged_frame_siglip_flat_ip_w5_s2_grid2x3'

@dataclass(frozen=True)
class MergedFrameFaissConfig:
    index_dir: Path = DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR
    index_name: str = DEFAULT_MERGED_FRAME_FAISS_INDEX_NAME
    query_text_limit: int = 6
    top_k: int = 8
    search_k: int = 80
    min_score: float = 0.0
    same_sequence_only: bool = True
    cluster_max_frame_gap: int = 48
    cluster_max_sec_gap: float | None = None
    max_clusters: int = 8
    max_lines: int = 8

@dataclass(frozen=True)
class FrameWindowFaissConfig:
    """Frame-embedding window recall without storyboard image embeddings.

    This keeps visual retrieval at the single-frame SigLIP level and promotes
    nearby frame hits into temporal windows. It is question/gold-answer agnostic.
    """
    frame_config: FrameFaissConfig = FrameFaissConfig()
    window_size_sec: float = 5.0
    window_stride_sec: float = 2.0
    top_k: int = 8
    search_k: int = 80
    max_clusters: int = 8
    max_lines: int = 8
    max_frames_per_window: int = 8

class MergedFrameFaissGrounder:
    """Merged-frame visual recall over a FAISS side index.

    The retrieval unit is a question-independent storyboard window built from
    consecutive sampled frames. This component reads question text/IR and index
    metadata only; it does not read answer labels or apply dataset-specific rules.
    """

    def __init__(self, *, index_dir: Path=DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR, index_name: str=DEFAULT_MERGED_FRAME_FAISS_INDEX_NAME) -> None:
        import faiss
        self.index_dir = resolve_repo_input_path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f'{index_name}.faiss'
        self.metadata_db_path = self.index_dir / f'{index_name}.metadata.sqlite'
        self.manifest_path = self.index_dir / f'{index_name}.manifest.json'
        if not self.index_path.exists():
            raise FileNotFoundError(f'Missing merged-frame FAISS index: {self.index_path}')
        if not self.metadata_db_path.exists():
            raise FileNotFoundError(f'Missing merged-frame FAISS metadata DB: {self.metadata_db_path}')
        self.index = faiss.read_index(str(self.index_path))
        self.connection = sqlite3.connect(self.metadata_db_path)
        self.connection.row_factory = sqlite3.Row
        self._faiss = faiss
        self._text_encoder = None
        metadata_count = self.count_metadata_rows()
        if int(self.index.ntotal) != metadata_count:
            raise ValueError(f'Merged-frame FAISS index and metadata DB size mismatch: index={self.index.ntotal}, metadata={metadata_count}')
        self.manifest = self._load_manifest()

    def close(self) -> None:
        self.connection.close()

    def count_metadata_rows(self) -> int:
        return int(self.connection.execute('SELECT COUNT(*) FROM merged_frame_metadata').fetchone()[0])

    def describe(self) -> dict[str, Any]:
        return {'contract': 'navqa_merged_frame_faiss_grounding_v1', 'index_path': str(self.index_path), 'metadata_db_path': str(self.metadata_db_path), 'manifest_path': str(self.manifest_path) if self.manifest_path.exists() else '', 'index_ntotal': int(self.index.ntotal), 'metadata_rows': self.count_metadata_rows(), 'dimension': int(getattr(self.index, 'd', 0) or 0), 'manifest': self.manifest}

    def build_query_recall(self, *, seq_id: str, question: str, question_ir: dict[str, Any] | None, config: MergedFrameFaissConfig) -> dict[str, Any]:
        query_texts = merged_frame_query_texts(question=question, question_ir=question_ir or {}, limit=max(1, config.query_text_limit))
        if not query_texts:
            return {'contract': 'navqa_merged_frame_faiss_query_recall_v1', 'available': False, 'seq_id': str(seq_id), 'query_texts': [], 'groups': [], 'clusters': [], 'lines': [], 'visual_focus_frames': []}
        vectors = self._encode_texts(query_texts)
        groups: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int | None]] = set()
        for query_text, vector in zip(query_texts, vectors, strict=True):
            rows = self._search_by_vector(vector=vector, seq_id=str(seq_id), top_k=max(1, config.top_k), search_k=max(1, config.search_k), same_sequence_only=config.same_sequence_only, min_score=float(config.min_score))
            group_results: list[dict[str, Any]] = []
            for row in rows:
                key = (str(row.get('seq_id') or ''), _optional_int(row.get('window_index')))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                group_results.append(row)
                all_rows.append({**row, '_query_text': query_text})
            if group_results:
                groups.append({'query_text': query_text, 'results': [_concise_merged_window(row) for row in group_results]})
        clusters = cluster_merged_frame_results(all_rows, max_frame_gap=max(0, int(config.cluster_max_frame_gap)), max_sec_gap=config.cluster_max_sec_gap, max_clusters=max(1, int(config.max_clusters)))
        lines = _render_merged_frame_recall_lines(clusters, max_lines=max(1, int(config.max_lines)))
        visual_focus_frames = _merged_frame_focus_frames_from_clusters(clusters, max_frames=max(1, int(config.max_lines)))
        return {'contract': 'navqa_merged_frame_faiss_query_recall_v1', 'available': bool(groups), 'seq_id': str(seq_id), 'query_texts': query_texts, 'top_k': max(1, config.top_k), 'search_k': max(1, config.search_k), 'min_score': float(config.min_score), 'same_sequence_only': bool(config.same_sequence_only), 'cluster_max_frame_gap': max(0, int(config.cluster_max_frame_gap)), 'cluster_max_sec_gap': config.cluster_max_sec_gap, 'groups': groups, 'clusters': clusters, 'lines': lines, 'visual_focus_frames': visual_focus_frames}

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
        resolved_search_k = min(int(self.index.ntotal), max(top_k, search_k))
        distances, indices = self.index.search(query, resolved_search_k)
        ids = [int(value) for value in indices[0] if int(value) >= 0]
        metadata = self._metadata_by_ids(ids)
        rows: list[dict[str, Any]] = []
        for score, window_id in zip(distances[0], indices[0], strict=True):
            window_id = int(window_id)
            if window_id < 0:
                continue
            if float(score) < min_score:
                continue
            row = metadata.get(window_id)
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
        rows = self.connection.execute(f'\n            SELECT faiss_id, raw_json\n            FROM merged_frame_metadata\n            WHERE faiss_id IN ({placeholders})\n            ', faiss_ids).fetchall()
        return {int(row['faiss_id']): json.loads(str(row['raw_json'])) for row in rows}

class FrameWindowFaissGrounder:
    """Merged-frame style recall by aggregating frame-level FAISS hits."""

    def __init__(self, *, index_dir: Path=DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR.parent / 'frame_faiss_grounding', index_name: str='frame_siglip_flat_ip') -> None:
        self.frame_grounder = FrameFaissGrounder(index_dir=index_dir, index_name=index_name)

    def close(self) -> None:
        self.frame_grounder.close()

    def describe(self) -> dict[str, Any]:
        description = self.frame_grounder.describe()
        return {'contract': 'navqa_frame_window_faiss_grounding_v1', 'source_frame_faiss': description, 'aggregation': 'frame_hits_to_temporal_windows'}

    def build_query_recall(self, *, seq_id: str, question: str, question_ir: dict[str, Any] | None, config: FrameWindowFaissConfig) -> dict[str, Any]:
        frame_config = config.frame_config
        frame_config = FrameFaissConfig(index_dir=frame_config.index_dir, index_name=frame_config.index_name, query_text_limit=frame_config.query_text_limit, top_k=max(int(config.top_k), int(frame_config.top_k)), search_k=max(int(config.search_k), int(frame_config.search_k)), min_score=frame_config.min_score, same_sequence_only=frame_config.same_sequence_only, cluster_max_frame_gap=frame_config.cluster_max_frame_gap, cluster_max_sec_gap=frame_config.cluster_max_sec_gap, max_clusters=max(int(config.max_clusters), int(frame_config.max_clusters)), max_lines=frame_config.max_lines)
        frame_payload = self.frame_grounder.build_query_recall(seq_id=seq_id, question=question, question_ir=question_ir or {}, config=frame_config)
        return frame_payload_as_window_payload(frame_payload, window_size_sec=max(0.1, float(config.window_size_sec)), window_stride_sec=max(0.1, float(config.window_stride_sec)), max_clusters=max(1, int(config.max_clusters)), max_lines=max(1, int(config.max_lines)), max_frames_per_window=max(1, int(config.max_frames_per_window)))

def frame_payload_as_window_payload(frame_payload: dict[str, Any], *, window_size_sec: float, window_stride_sec: float, max_clusters: int, max_lines: int, max_frames_per_window: int) -> dict[str, Any]:
    rows = _frame_rows_from_frame_payload(frame_payload)
    windows = aggregate_frame_rows_to_windows(rows, window_size_sec=window_size_sec, window_stride_sec=window_stride_sec, max_windows=max_clusters, max_frames_per_window=max_frames_per_window)
    clusters = [_frame_window_as_merged_cluster(index, window) for index, window in enumerate(windows, start=1)]
    visual_focus_frames = _merged_frame_focus_frames_from_clusters(clusters, max_frames=max_lines)
    lines = _render_merged_frame_recall_lines(clusters, max_lines=max_lines)
    return {'contract': 'navqa_frame_window_faiss_query_recall_v1', 'source_contract': frame_payload.get('contract'), 'available': bool(clusters), 'seq_id': frame_payload.get('seq_id'), 'query_texts': list(frame_payload.get('query_texts', []) or []), 'window_size_sec': float(window_size_sec), 'window_stride_sec': float(window_stride_sec), 'max_clusters': max_clusters, 'max_lines': max_lines, 'groups': list(frame_payload.get('groups', []) or []), 'clusters': clusters, 'lines': lines, 'visual_focus_frames': visual_focus_frames, 'source_frame_faiss': frame_payload}

def aggregate_frame_rows_to_windows(rows: list[dict[str, Any]], *, window_size_sec: float, window_stride_sec: float, max_windows: int, max_frames_per_window: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    candidate_rows = []
    for row in rows:
        seq_id = str(row.get('seq_id') or '')
        elapsed = _optional_float(row.get('elapsed_sec_from_sequence_start'))
        frame_index = _optional_int(row.get('graph_frame_index'))
        if elapsed is None or frame_index is None:
            continue
        candidate_rows.append((dict(row), seq_id, elapsed))
    candidate_rows.sort(key=lambda item: (-float(item[0].get('similarity') or 0.0), str(item[1]), float(item[2])))
    windows: list[dict[str, Any]] = []
    assigned: set[tuple[str, int | None, str]] = set()
    for seed, seq_id, elapsed in candidate_rows:
        seed_key = _frame_assignment_key(seed)
        if seed_key in assigned:
            continue
        start_sec = _window_start_for_time(elapsed, window_size_sec=window_size_sec, window_stride_sec=window_stride_sec)
        end_sec = start_sec + window_size_sec
        frames: list[dict[str, Any]] = []
        for row, row_seq_id, row_elapsed in candidate_rows:
            if row_seq_id != seq_id:
                continue
            row_key = _frame_assignment_key(row)
            if row_key in assigned:
                continue
            if start_sec <= row_elapsed <= end_sec:
                frames.append(row)
        if not frames:
            continue
        for frame in frames:
            assigned.add(_frame_assignment_key(frame))
        windows.append({'seq_id': seq_id, 'window_index': len(windows), 'start_elapsed_sec': start_sec, 'end_elapsed_sec': end_sec, 'frames': frames})
    payloads = []
    for window in windows:
        frames = sorted(window['frames'], key=lambda item: (-float(item.get('similarity') or 0.0), _optional_float(item.get('elapsed_sec_from_sequence_start')) or 0.0))[:max_frames_per_window]
        if not frames:
            continue
        elapsed_values = [value for value in (_optional_float(item.get('elapsed_sec_from_sequence_start')) for item in frames) if value is not None]
        similarities = [float(item.get('similarity') or 0.0) for item in frames]
        best = frames[0]
        graph_range = _index_range_for_time_window(frames, index_key='graph_frame_index', start_sec=float(window['start_elapsed_sec']), end_sec=float(window['end_elapsed_sec']))
        source_range = _index_range_for_time_window(frames, index_key='source_frame', start_sec=float(window['start_elapsed_sec']), end_sec=float(window['end_elapsed_sec']))
        payloads.append({'seq_id': window['seq_id'], 'window_index': window['window_index'], 'start_elapsed_sec': window['start_elapsed_sec'], 'end_elapsed_sec': window['end_elapsed_sec'], 'start_graph_frame_index': graph_range[0], 'end_graph_frame_index': graph_range[1], 'start_source_frame': source_range[0], 'end_source_frame': source_range[1], 'frame_count': len(frames), 'max_similarity': max(similarities) if similarities else None, 'mean_similarity': sum(similarities) / len(similarities) if similarities else None, 'best_frame': best, 'frames': frames, 'observed_start_elapsed_sec': min(elapsed_values) if elapsed_values else None, 'observed_end_elapsed_sec': max(elapsed_values) if elapsed_values else None})
    payloads.sort(key=lambda item: (-float(item.get('max_similarity') or 0.0), -float(item.get('mean_similarity') or 0.0), str(item.get('seq_id') or ''), float(item.get('start_elapsed_sec') or 0.0)))
    return payloads[:max_windows]

def _frame_assignment_key(row: dict[str, Any]) -> tuple[str, int | None, str]:
    return (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), str(row.get('image_name') or row.get('image_path') or ''))

def _index_range_for_time_window(rows: list[dict[str, Any]], *, index_key: str, start_sec: float, end_sec: float) -> tuple[int | None, int | None]:
    anchors = [(_optional_float(row.get('elapsed_sec_from_sequence_start')), _optional_int(row.get(index_key))) for row in rows]
    anchors = [(elapsed, frame) for elapsed, frame in anchors if elapsed is not None and frame is not None]
    if not anchors:
        return (None, None)
    anchors.sort(key=lambda item: item[0])
    best_elapsed, best_index = anchors[0]
    units_per_sec = _median_units_per_second(anchors)
    if units_per_sec is None:
        units_per_sec = abs(float(best_index) / float(best_elapsed)) if abs(float(best_elapsed)) > 1e-06 else 1.0
    start_index = int(round(float(best_index) + (start_sec - best_elapsed) * units_per_sec))
    end_index = int(round(float(best_index) + (end_sec - best_elapsed) * units_per_sec))
    if start_index > end_index:
        start_index, end_index = (end_index, start_index)
    return (max(0, start_index), max(0, end_index))

def _median_units_per_second(anchors: list[tuple[float, int]]) -> float | None:
    rates: list[float] = []
    for (prev_sec, prev_index), (cur_sec, cur_index) in zip(anchors, anchors[1:]):
        delta_sec = abs(cur_sec - prev_sec)
        if delta_sec <= 1e-06:
            continue
        rates.append(abs(float(cur_index - prev_index)) / delta_sec)
    if not rates:
        return None
    rates.sort()
    return rates[len(rates) // 2]

def _window_start_for_time(elapsed_sec: float, *, window_size_sec: float, window_stride_sec: float) -> float:
    centered_start = max(0.0, elapsed_sec - window_size_sec / 2.0)
    stride_index = int(centered_start // window_stride_sec)
    return float(stride_index) * window_stride_sec

def _frame_window_as_merged_cluster(rank: int, window: dict[str, Any]) -> dict[str, Any]:
    best_frame = dict(window.get('best_frame') or {})
    best_window = {'rank': rank, 'similarity': _optional_float(window.get('max_similarity')), 'faiss_id': _optional_int(best_frame.get('faiss_id')), 'seq_id': str(window.get('seq_id') or ''), 'cam': best_frame.get('cam'), 'stride': best_frame.get('stride'), 'window_index': _optional_int(window.get('window_index')), 'window_size': _optional_int(window.get('frame_count')), 'window_stride': None, 'image_name': best_frame.get('image_name'), 'image_path': best_frame.get('image_path'), 'frame_count': _optional_int(window.get('frame_count')), 'start_sampled_index': _optional_int(best_frame.get('sampled_index')), 'end_sampled_index': _optional_int(best_frame.get('sampled_index')), 'start_graph_frame_index': _optional_int(window.get('start_graph_frame_index')), 'end_graph_frame_index': _optional_int(window.get('end_graph_frame_index')), 'start_source_frame': _optional_int(window.get('start_source_frame')), 'end_source_frame': _optional_int(window.get('end_source_frame')), 'start_timestamp_unix': best_frame.get('timestamp_unix'), 'end_timestamp_unix': best_frame.get('timestamp_unix'), 'start_timestamp_iso': best_frame.get('timestamp_iso'), 'end_timestamp_iso': best_frame.get('timestamp_iso'), 'start_elapsed_sec': _optional_float(window.get('start_elapsed_sec')), 'end_elapsed_sec': _optional_float(window.get('end_elapsed_sec')), 'mean_similarity': _optional_float(window.get('mean_similarity')), 'aggregation': 'frame_window_faiss', 'frames': [_frame_as_window_frame(item) for item in window.get('frames', []) or [] if isinstance(item, dict)]}
    return {'rank': rank, 'cluster_index': rank, 'seq_id': window.get('seq_id'), 'window_count': 1, 'frame_count': window.get('frame_count'), 'start_frame': window.get('start_graph_frame_index'), 'end_frame': window.get('end_graph_frame_index'), 'start_elapsed_sec': window.get('start_elapsed_sec'), 'end_elapsed_sec': window.get('end_elapsed_sec'), 'observed_start_elapsed_sec': window.get('observed_start_elapsed_sec'), 'observed_end_elapsed_sec': window.get('observed_end_elapsed_sec'), 'max_similarity': window.get('max_similarity'), 'mean_similarity': window.get('mean_similarity'), 'best_window': best_window, 'query_texts': _ordered_unique_texts((str(item.get('_query_text') or '').strip() for item in window.get('frames', []) or [] if isinstance(item, dict) and str(item.get('_query_text') or '').strip())), 'windows': [best_window], 'source_kind': 'frame_window_faiss'}

def _frame_as_window_frame(row: dict[str, Any]) -> dict[str, Any]:
    return {'rank': row.get('rank'), 'similarity': _optional_float(row.get('similarity')), 'faiss_id': _optional_int(row.get('faiss_id')), 'seq_id': str(row.get('seq_id') or ''), 'graph_frame_index': _optional_int(row.get('graph_frame_index')), 'source_frame': _optional_int(row.get('source_frame')), 'sampled_index': _optional_int(row.get('sampled_index')), 'elapsed_sec_from_sequence_start': _optional_float(row.get('elapsed_sec_from_sequence_start')), 'image_name': row.get('image_name'), 'image_path': row.get('image_path'), 'query_text': row.get('_query_text')}

def _frame_rows_from_frame_payload(frame_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str, str]] = set()
    for group in frame_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        query_text = str(group.get('query_text') or '').strip()
        for row in group.get('results', []) or []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get('seq_id') or ''), _optional_int(row.get('graph_frame_index')), str(row.get('image_name') or ''), query_text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, '_query_text': query_text})
    return rows

def merged_frame_query_texts(*, question: str, question_ir: dict[str, Any], limit: int) -> list[str]:
    return [text.replace('a video frame of ', 'a storyboard video window of ', 1) for text in frame_query_texts(question=question, question_ir=question_ir, limit=limit)]

def cluster_merged_frame_results(rows: list[dict[str, Any]], *, max_frame_gap: int, max_sec_gap: float | None, max_clusters: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    sortable = []
    for row in rows:
        start_frame = _optional_int(row.get('start_graph_frame_index'))
        end_frame = _optional_int(row.get('end_graph_frame_index'))
        timestamp = _optional_float(row.get('start_elapsed_sec'))
        if start_frame is None and end_frame is None and (timestamp is None):
            continue
        sortable.append(row)
    sortable.sort(key=lambda row: (str(row.get('seq_id') or ''), _optional_int(row.get('start_graph_frame_index')) is None, _optional_int(row.get('start_graph_frame_index')) or 0, _optional_float(row.get('start_elapsed_sec')) or 0.0, -float(row.get('similarity') or 0.0)))
    clusters: list[list[dict[str, Any]]] = []
    for row in sortable:
        if not clusters or not _window_row_close(clusters[-1][-1], row, max_frame_gap=max_frame_gap, max_sec_gap=max_sec_gap):
            clusters.append([row])
        else:
            clusters[-1].append(row)
    payloads = [_merged_cluster_payload(index, rows) for index, rows in enumerate(clusters, start=1)]
    payloads.sort(key=lambda item: (-float(item.get('max_similarity') or 0.0), -int(item.get('window_count') or 0), str(item.get('seq_id') or ''), int(item.get('start_frame') or 0)))
    for rank, item in enumerate(payloads, start=1):
        item['rank'] = rank
    return payloads[:max_clusters]

def attach_merged_frame_faiss_evidence(graph_answer: dict[str, Any], merged_frame_payload: dict[str, Any], *, expose_as_visual: bool=True, include_context_lines: bool=True) -> dict[str, Any]:
    frame_channel = _capture_frame_faiss_channel(graph_answer)
    visual_channel = _capture_visual_faiss_channel(graph_answer)
    frame_payload = merged_frame_payload_as_frame_payload(merged_frame_payload)
    graph_answer = attach_frame_faiss_evidence(graph_answer, frame_payload, expose_as_visual=expose_as_visual, include_context_lines=include_context_lines)
    _restore_frame_faiss_channel(graph_answer, frame_channel)
    _restore_visual_faiss_channel(graph_answer, visual_channel)
    _store_merged_frame_payload(graph_answer, merged_frame_payload)
    return graph_answer

def fuse_merged_frame_faiss_evidence(graph_answer: dict[str, Any], merged_frame_payload: dict[str, Any]) -> dict[str, Any]:
    frame_channel = _capture_frame_faiss_channel(graph_answer)
    visual_channel = _capture_visual_faiss_channel(graph_answer)
    frame_payload = merged_frame_payload_as_frame_payload(merged_frame_payload)
    graph_answer = fuse_frame_faiss_evidence(graph_answer, frame_payload)
    _restore_frame_faiss_channel(graph_answer, frame_channel)
    _restore_visual_faiss_channel(graph_answer, visual_channel)
    _store_merged_frame_payload(graph_answer, merged_frame_payload)
    return graph_answer

def expand_merged_frame_faiss_graph_evidence(graph_answer: dict[str, Any], merged_frame_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, frame_radius: int=8, max_items: int=12) -> dict[str, Any]:
    frame_channel = _capture_frame_faiss_channel(graph_answer)
    visual_channel = _capture_visual_faiss_channel(graph_answer)
    frame_payload = merged_frame_payload_as_frame_payload(merged_frame_payload)
    graph_answer = expand_frame_faiss_graph_evidence(graph_answer, frame_payload, graph=graph, frame_radius=frame_radius, max_items=max_items)
    _restore_frame_faiss_channel(graph_answer, frame_channel)
    _restore_visual_faiss_channel(graph_answer, visual_channel)
    _store_merged_frame_payload(graph_answer, merged_frame_payload)
    return graph_answer

def _capture_visual_faiss_channel(graph_answer: dict[str, Any]) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return {}
    structured = bundle.get('structured_graph_context', {})
    return {'visual_faiss_grounding': bundle.get('visual_faiss_grounding'), 'visual_faiss_payloads': list(bundle.get('visual_faiss_payloads', []) or []), 'visual_faiss_evidence_lines': list(bundle.get('visual_faiss_evidence_lines', []) or []), 'has_visual_faiss_grounding': 'visual_faiss_grounding' in bundle, 'has_visual_faiss_payloads': 'visual_faiss_payloads' in bundle, 'has_visual_faiss_evidence_lines': 'visual_faiss_evidence_lines' in bundle, 'structured_visual_faiss_grounding': structured.get('visual_faiss_grounding') if isinstance(structured, dict) else None, 'structured_visual_faiss_payloads': list(structured.get('visual_faiss_payloads', []) or []) if isinstance(structured, dict) else [], 'structured_visual_faiss_evidence_lines': list(structured.get('visual_faiss_evidence_lines', []) or []) if isinstance(structured, dict) else [], 'structured_has_visual_faiss_grounding': isinstance(structured, dict) and 'visual_faiss_grounding' in structured, 'structured_has_visual_faiss_payloads': isinstance(structured, dict) and 'visual_faiss_payloads' in structured, 'structured_has_visual_faiss_evidence_lines': isinstance(structured, dict) and 'visual_faiss_evidence_lines' in structured}

def _capture_frame_faiss_channel(graph_answer: dict[str, Any]) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return {}
    structured = bundle.get('structured_graph_context', {})
    return {'frame_faiss_grounding': bundle.get('frame_faiss_grounding'), 'frame_faiss_payloads': list(bundle.get('frame_faiss_payloads', []) or []), 'frame_faiss_evidence_lines': list(bundle.get('frame_faiss_evidence_lines', []) or []), 'has_frame_faiss_grounding': 'frame_faiss_grounding' in bundle, 'has_frame_faiss_payloads': 'frame_faiss_payloads' in bundle, 'has_frame_faiss_evidence_lines': 'frame_faiss_evidence_lines' in bundle, 'structured_frame_faiss_grounding': structured.get('frame_faiss_grounding') if isinstance(structured, dict) else None, 'structured_frame_faiss_payloads': list(structured.get('frame_faiss_payloads', []) or []) if isinstance(structured, dict) else [], 'structured_frame_faiss_evidence_lines': list(structured.get('frame_faiss_evidence_lines', []) or []) if isinstance(structured, dict) else [], 'structured_has_frame_faiss_grounding': isinstance(structured, dict) and 'frame_faiss_grounding' in structured, 'structured_has_frame_faiss_payloads': isinstance(structured, dict) and 'frame_faiss_payloads' in structured, 'structured_has_frame_faiss_evidence_lines': isinstance(structured, dict) and 'frame_faiss_evidence_lines' in structured}

def _restore_visual_faiss_channel(graph_answer: dict[str, Any], snapshot: dict[str, Any]) -> None:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return
    _restore_optional_key(bundle, 'visual_faiss_grounding', snapshot.get('visual_faiss_grounding'), bool(snapshot.get('has_visual_faiss_grounding')))
    _restore_optional_key(bundle, 'visual_faiss_payloads', snapshot.get('visual_faiss_payloads', []), bool(snapshot.get('has_visual_faiss_payloads')))
    _restore_optional_key(bundle, 'visual_faiss_evidence_lines', snapshot.get('visual_faiss_evidence_lines', []), bool(snapshot.get('has_visual_faiss_evidence_lines')))
    structured = bundle.get('structured_graph_context', {})
    if not isinstance(structured, dict):
        return
    _restore_optional_key(structured, 'visual_faiss_grounding', snapshot.get('structured_visual_faiss_grounding'), bool(snapshot.get('structured_has_visual_faiss_grounding')))
    _restore_optional_key(structured, 'visual_faiss_payloads', snapshot.get('structured_visual_faiss_payloads', []), bool(snapshot.get('structured_has_visual_faiss_payloads')))
    _restore_optional_key(structured, 'visual_faiss_evidence_lines', snapshot.get('structured_visual_faiss_evidence_lines', []), bool(snapshot.get('structured_has_visual_faiss_evidence_lines')))

def _restore_frame_faiss_channel(graph_answer: dict[str, Any], snapshot: dict[str, Any]) -> None:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return
    _restore_optional_key(bundle, 'frame_faiss_grounding', snapshot.get('frame_faiss_grounding'), bool(snapshot.get('has_frame_faiss_grounding')))
    _restore_optional_key(bundle, 'frame_faiss_payloads', snapshot.get('frame_faiss_payloads', []), bool(snapshot.get('has_frame_faiss_payloads')))
    _restore_optional_key(bundle, 'frame_faiss_evidence_lines', snapshot.get('frame_faiss_evidence_lines', []), bool(snapshot.get('has_frame_faiss_evidence_lines')))
    structured = bundle.get('structured_graph_context', {})
    if not isinstance(structured, dict):
        return
    _restore_optional_key(structured, 'frame_faiss_grounding', snapshot.get('structured_frame_faiss_grounding'), bool(snapshot.get('structured_has_frame_faiss_grounding')))
    _restore_optional_key(structured, 'frame_faiss_payloads', snapshot.get('structured_frame_faiss_payloads', []), bool(snapshot.get('structured_has_frame_faiss_payloads')))
    _restore_optional_key(structured, 'frame_faiss_evidence_lines', snapshot.get('structured_frame_faiss_evidence_lines', []), bool(snapshot.get('structured_has_frame_faiss_evidence_lines')))

def _restore_optional_key(target: dict[str, Any], key: str, value: Any, existed: bool) -> None:
    if existed:
        target[key] = value
    else:
        target.pop(key, None)

def merged_frame_payload_as_frame_payload(merged_frame_payload: dict[str, Any]) -> dict[str, Any]:
    clusters = []
    for cluster in merged_frame_payload.get('clusters', []) or []:
        if not isinstance(cluster, dict):
            continue
        best_window = cluster.get('best_window', {})
        if not isinstance(best_window, dict):
            best_window = {}
        best_frame = _window_as_best_frame(best_window)
        windows = [_window_as_best_frame(window) for window in cluster.get('windows', []) or [] if isinstance(window, dict)]
        clusters.append({'rank': cluster.get('rank'), 'seq_id': cluster.get('seq_id'), 'frame_count': cluster.get('window_count'), 'start_frame': cluster.get('start_frame'), 'end_frame': cluster.get('end_frame'), 'start_elapsed_sec': cluster.get('start_elapsed_sec'), 'end_elapsed_sec': cluster.get('end_elapsed_sec'), 'max_similarity': cluster.get('max_similarity'), 'best_frame': best_frame, 'query_texts': list(cluster.get('query_texts', []) or []), 'frames': windows, 'source_kind': 'merged_frame_faiss_window'})
    return {'contract': 'navqa_merged_frame_faiss_as_frame_query_recall_v1', 'source_contract': merged_frame_payload.get('contract'), 'available': bool(clusters), 'seq_id': merged_frame_payload.get('seq_id'), 'query_texts': list(merged_frame_payload.get('query_texts', []) or []), 'top_k': merged_frame_payload.get('top_k'), 'search_k': merged_frame_payload.get('search_k'), 'min_score': merged_frame_payload.get('min_score'), 'same_sequence_only': merged_frame_payload.get('same_sequence_only'), 'groups': list(merged_frame_payload.get('groups', []) or []), 'clusters': clusters, 'lines': _merged_frame_faiss_evidence_lines(merged_frame_payload), 'visual_focus_frames': [item for item in merged_frame_payload.get('visual_focus_frames', []) or [] if isinstance(item, dict)], 'source_merged_frame_faiss': merged_frame_payload}

def _store_merged_frame_payload(graph_answer: dict[str, Any], merged_frame_payload: dict[str, Any]) -> None:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return
    updated = dict(bundle)
    existing_payloads = list(updated.get('merged_frame_faiss_payloads', []) or [])
    existing_payloads.append(merged_frame_payload)
    lines = _merged_frame_faiss_evidence_lines(merged_frame_payload)
    existing_lines = [str(line).strip() for line in updated.get('merged_frame_faiss_evidence_lines', []) if str(line).strip()]
    updated['merged_frame_faiss_grounding'] = merged_frame_payload
    updated['merged_frame_faiss_payloads'] = existing_payloads
    updated['merged_frame_faiss_evidence_lines'] = _ordered_unique_texts([*existing_lines, *lines])
    structured = updated.get('structured_graph_context', {})
    if isinstance(structured, dict):
        structured = dict(structured)
        structured['merged_frame_faiss_grounding'] = merged_frame_payload
        structured['merged_frame_faiss_payloads'] = existing_payloads
        structured['merged_frame_faiss_evidence_lines'] = updated['merged_frame_faiss_evidence_lines']
        updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated

def _merged_cluster_payload(index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: (-float(row.get('similarity') or 0.0), _optional_int(row.get('start_graph_frame_index')) or 0))
    best = sorted_rows[0]
    start_frames = [value for value in (_optional_int(row.get('start_graph_frame_index')) for row in rows) if value is not None]
    end_frames = [value for value in (_optional_int(row.get('end_graph_frame_index')) for row in rows) if value is not None]
    start_secs = [value for value in (_optional_float(row.get('start_elapsed_sec')) for row in rows) if value is not None]
    end_secs = [value for value in (_optional_float(row.get('end_elapsed_sec')) for row in rows) if value is not None]
    return {'cluster_index': index, 'seq_id': best.get('seq_id'), 'window_count': len(rows), 'start_frame': min(start_frames) if start_frames else None, 'end_frame': max(end_frames) if end_frames else None, 'start_elapsed_sec': min(start_secs) if start_secs else None, 'end_elapsed_sec': max(end_secs) if end_secs else None, 'max_similarity': max((float(row.get('similarity') or 0.0) for row in rows)), 'best_window': _concise_merged_window(best), 'query_texts': _ordered_unique_texts((str(row.get('_query_text') or '').strip() for row in rows if str(row.get('_query_text') or '').strip())), 'windows': [_concise_merged_window(row) for row in sorted_rows]}

def _window_row_close(previous: dict[str, Any], current: dict[str, Any], *, max_frame_gap: int, max_sec_gap: float | None) -> bool:
    if str(previous.get('seq_id') or '') != str(current.get('seq_id') or ''):
        return False
    prev_end = _optional_int(previous.get('end_graph_frame_index'))
    cur_start = _optional_int(current.get('start_graph_frame_index'))
    if prev_end is not None and cur_start is not None:
        if cur_start - prev_end <= max_frame_gap:
            return True
    if max_sec_gap is None:
        return False
    prev_sec = _optional_float(previous.get('end_elapsed_sec'))
    cur_sec = _optional_float(current.get('start_elapsed_sec'))
    return prev_sec is not None and cur_sec is not None and (cur_sec - prev_sec <= max_sec_gap)

def _render_merged_frame_recall_lines(clusters: list[dict[str, Any]], *, max_lines: int) -> list[str]:
    lines: list[str] = []
    for cluster in clusters:
        line = _merged_frame_cluster_line(cluster)
        if line:
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines

def _merged_frame_faiss_evidence_lines(payload: dict[str, Any]) -> list[str]:
    payload_lines = [str(line).strip() for line in payload.get('lines', []) if str(line).strip()]
    if payload_lines:
        return _ordered_unique_texts(payload_lines)
    return _ordered_unique_texts((_merged_frame_cluster_line(cluster) for cluster in payload.get('clusters', []) or [] if isinstance(cluster, dict)))

def _merged_frame_cluster_line(cluster: dict[str, Any]) -> str:
    best = cluster.get('best_window', {})
    if not isinstance(best, dict):
        best = {}
    seq = str(cluster.get('seq_id') or best.get('seq_id') or '?')
    start_frame = _optional_int(cluster.get('start_frame'))
    end_frame = _optional_int(cluster.get('end_frame'))
    start_sec = _optional_float(cluster.get('start_elapsed_sec'))
    end_sec = _optional_float(cluster.get('end_elapsed_sec'))
    best_window = _optional_int(best.get('window_index'))
    query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
    window_count = _optional_int(cluster.get('window_count'))
    parts = [_merged_frame_cluster_prefix(cluster, best), f'seq{seq}', f'frames={_range_text(start_frame, end_frame)}']
    if start_sec is not None or end_sec is not None:
        parts.append(f't={_time_range_text(start_sec, end_sec)}')
    if best_window is not None:
        parts.append(f'best_window={best_window}')
    if window_count is not None:
        parts.append(f'count={window_count}')
    parts.append(f"max_sim={_format_number(cluster.get('max_similarity'))}")
    if query_texts:
        parts.append('queries=' + ' | '.join((_short_quote(text, limit=96) for text in query_texts[:3])))
    return ' '.join((part for part in parts if part))

def _merged_frame_cluster_prefix(cluster: dict[str, Any], best: dict[str, Any]) -> str:
    if str(cluster.get('source_kind') or '') == 'frame_window_faiss' or str(best.get('aggregation') or '') == 'frame_window_faiss':
        return 'frame_window_faiss_cluster'
    return 'merged_frame_faiss_cluster'

def _merged_frame_focus_frames_from_clusters(clusters: list[dict[str, Any]], *, max_frames: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cluster in clusters:
        best = cluster.get('best_window', {})
        if not isinstance(best, dict):
            continue
        frame_index = _best_graph_frame_for_window(best)
        if frame_index is None or frame_index in seen:
            continue
        seen.add(frame_index)
        query_texts = [str(text).strip() for text in cluster.get('query_texts', []) or [] if str(text).strip()]
        frames.append({'frame_index': frame_index, 'timestamp_sec': best.get('start_elapsed_sec'), 'scene_id': None, 'score': round(float(best.get('similarity') or 0.0), 6), 'reasons': ['merged_frame_faiss_query_recall'], 'query_text': query_texts[0] if query_texts else '', 'cluster_rank': cluster.get('rank'), 'cluster_start_frame': cluster.get('start_frame'), 'cluster_end_frame': cluster.get('end_frame'), 'window_index': best.get('window_index')})
        if len(frames) >= max_frames:
            break
    return frames

def _window_as_best_frame(window: dict[str, Any]) -> dict[str, Any]:
    frame_index = _best_graph_frame_for_window(window)
    return {'rank': window.get('rank'), 'similarity': _optional_float(window.get('similarity')), 'faiss_id': _optional_int(window.get('faiss_id')), 'seq_id': str(window.get('seq_id') or ''), 'cam': window.get('cam'), 'stride': window.get('stride'), 'image_name': window.get('image_name'), 'image_path': window.get('image_path'), 'sampled_index': _optional_int(window.get('start_sampled_index')), 'graph_frame_index': frame_index, 'source_frame': _optional_int(window.get('start_source_frame')), 'timestamp_unix': window.get('start_timestamp_unix'), 'timestamp_iso': window.get('start_timestamp_iso'), 'elapsed_sec_from_sequence_start': window.get('start_elapsed_sec'), 'merged_frame_window': {'window_index': _optional_int(window.get('window_index')), 'start_graph_frame_index': _optional_int(window.get('start_graph_frame_index')), 'end_graph_frame_index': _optional_int(window.get('end_graph_frame_index')), 'start_elapsed_sec': _optional_float(window.get('start_elapsed_sec')), 'end_elapsed_sec': _optional_float(window.get('end_elapsed_sec')), 'frame_count': _optional_int(window.get('frame_count'))}}

def _best_graph_frame_for_window(window: dict[str, Any]) -> int | None:
    start = _optional_int(window.get('start_graph_frame_index'))
    end = _optional_int(window.get('end_graph_frame_index'))
    if start is None:
        return end
    if end is None:
        return start
    return int(round((start + end) / 2))

def _concise_merged_window(row: dict[str, Any]) -> dict[str, Any]:
    return {'rank': row.get('rank'), 'similarity': _optional_float(row.get('similarity')), 'faiss_id': _optional_int(row.get('faiss_id')), 'seq_id': str(row.get('seq_id') or ''), 'cam': row.get('cam'), 'stride': row.get('stride'), 'window_index': _optional_int(row.get('window_index')), 'window_size': _optional_int(row.get('window_size')), 'window_stride': _optional_int(row.get('window_stride')), 'image_name': row.get('image_name'), 'image_path': row.get('image_path'), 'frame_count': _optional_int(row.get('frame_count')), 'start_sampled_index': _optional_int(row.get('start_sampled_index')), 'end_sampled_index': _optional_int(row.get('end_sampled_index')), 'start_graph_frame_index': _optional_int(row.get('start_graph_frame_index')), 'end_graph_frame_index': _optional_int(row.get('end_graph_frame_index')), 'start_source_frame': _optional_int(row.get('start_source_frame')), 'end_source_frame': _optional_int(row.get('end_source_frame')), 'start_timestamp_unix': row.get('start_timestamp_unix'), 'end_timestamp_unix': row.get('end_timestamp_unix'), 'start_timestamp_iso': row.get('start_timestamp_iso'), 'end_timestamp_iso': row.get('end_timestamp_iso'), 'start_elapsed_sec': _optional_float(row.get('start_elapsed_sec')), 'end_elapsed_sec': _optional_float(row.get('end_elapsed_sec')), 'frames': list(row.get('frames', []) or [])}

def _ordered_unique_texts(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result

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

def _range_text(start: Any, end: Any) -> str:
    start_int = _optional_int(start)
    end_int = _optional_int(end)
    if start_int is None and end_int is None:
        return '?'
    if start_int == end_int or end_int is None:
        return str(start_int)
    if start_int is None:
        return str(end_int)
    return f'{start_int}-{end_int}'

def _time_range_text(start: float | None, end: float | None) -> str:
    if start is None and end is None:
        return '?'
    if start is None:
        return f'{_format_number(end)}s'
    if end is None or abs(float(end) - float(start)) < 1e-06:
        return f'{_format_number(start)}s'
    return f'{_format_number(start)}-{_format_number(end)}s'

def _short_quote(value: str, *, limit: int) -> str:
    text = ' '.join(str(value or '').split())
    if len(text) > limit:
        text = text[:limit].rsplit(' ', 1)[0].rstrip(' ,;:.') + '...'
    return repr(text)
