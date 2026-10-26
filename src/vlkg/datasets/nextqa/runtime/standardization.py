"""Pure NExT-QA annotation standardization.

The official ``qid`` column is scoped to a video.  Runtime question IDs are
therefore stable composites of split, official video ID, and per-video qid.
Gold labels are deliberately emitted through a separate function so they
cannot leak into graph construction, retrieval, routing, or QA prompts.
"""
from __future__ import annotations
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
REQUIRED_VAL_COLUMNS = ('video', 'frame_count', 'width', 'height', 'question', 'answer', 'qid', 'type', 'a0', 'a1', 'a2', 'a3', 'a4')
_WHITESPACE = re.compile('\\s+')

def normalize_question(value: str) -> str:
    """Normalize whitespace without changing the question's semantics."""
    return _WHITESPACE.sub(' ', value.strip())

def stable_question_id(video_id: str, qid: str) -> str:
    """Return an unambiguous validation question ID."""
    return f'nextqa-val/{video_id}/q{qid}'

def load_official_mapping(path: Path) -> dict[str, str]:
    """Load and minimally validate the official video-to-VidOR mapping."""
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('NExT-QA video mapping must be a JSON object')
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        video_id = str(key).strip()
        vidor_id = str(value).strip()
        if not video_id or not vidor_id:
            raise ValueError('NExT-QA video mapping contains an empty key/value')
        mapping[video_id] = vidor_id
    return mapping

def load_validation_annotations(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load official validation rows and return normalized rows plus audit data."""
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing_columns = sorted(set(REQUIRED_VAL_COLUMNS) - set(columns))
        if missing_columns:
            raise ValueError(f'validation annotation is missing columns: {missing_columns}')
        source_rows = list(reader)
    rows: list[dict[str, Any]] = []
    composite_ids: list[str] = []
    qids_by_video: dict[str, list[int]] = defaultdict(list)
    choice_count_errors: list[str] = []
    empty_choice_errors: list[str] = []
    gold_index_errors: list[str] = []
    invalid_qid_errors: list[str] = []
    for row_number, source in enumerate(source_rows, start=2):
        video_id = str(source['video']).strip()
        qid_text = str(source['qid']).strip()
        try:
            qid_int = int(qid_text)
        except ValueError:
            invalid_qid_errors.append(f'row {row_number}: {qid_text!r}')
            qid_int = -1
        choices = [str(source[f'a{index}']).strip() for index in range(5)]
        if len(choices) != 5:
            choice_count_errors.append(f'row {row_number}')
        if any((not choice for choice in choices)):
            empty_choice_errors.append(f'row {row_number}')
        gold_text = str(source['answer']).strip()
        try:
            gold_index = int(gold_text)
        except ValueError:
            gold_index = -1
        if not 0 <= gold_index < len(choices):
            gold_index_errors.append(f'row {row_number}: {gold_text!r}')
        question_id = stable_question_id(video_id, qid_text)
        composite_ids.append(question_id)
        qids_by_video[video_id].append(qid_int)
        rows.append({'question_id': question_id, 'official_video_id': video_id, 'official_qid': qid_text, 'official_type': str(source['type']).strip(), 'normalized_question': normalize_question(str(source['question'])), 'answer_choices': choices, 'gold_choice_index': gold_index, 'source_frame_count': int(str(source['frame_count']).strip()), 'source_width': int(str(source['width']).strip()), 'source_height': int(str(source['height']).strip()), 'source_row_number': row_number})
    duplicate_question_ids = sorted((question_id for question_id, count in Counter(composite_ids).items() if count > 1))
    missing_qids_by_video: dict[str, list[int]] = {}
    duplicate_qids_by_video: dict[str, list[int]] = {}
    nonzero_start_qids: dict[str, int] = {}
    for video_id, qids in qids_by_video.items():
        valid_qids = [qid for qid in qids if qid >= 0]
        counts = Counter(valid_qids)
        duplicates = sorted((qid for qid, count in counts.items() if count > 1))
        if duplicates:
            duplicate_qids_by_video[video_id] = duplicates
        if valid_qids:
            minimum = min(valid_qids)
            maximum = max(valid_qids)
            missing = sorted(set(range(minimum, maximum + 1)) - set(valid_qids))
            if missing:
                missing_qids_by_video[video_id] = missing
            if minimum != 0:
                nonzero_start_qids[video_id] = minimum
    question_counts = Counter((row['official_video_id'] for row in rows))
    type_distribution = Counter((row['official_type'] for row in rows))
    gold_distribution = Counter((str(row['gold_choice_index']) for row in rows))
    bare_qid_distribution = Counter((row['official_qid'] for row in rows))
    audit = {'columns': list(columns), 'row_count': len(rows), 'video_count': len(question_counts), 'qid_scope': 'per_video', 'stable_question_id_format': 'nextqa-val/<official_video_id>/q<official_qid>', 'unique_stable_question_id_count': len(set(composite_ids)), 'unique_bare_qid_count': len(bare_qid_distribution), 'duplicate_stable_question_ids': duplicate_question_ids, 'duplicate_qids_within_video': duplicate_qids_by_video, 'missing_qids_within_video': missing_qids_by_video, 'nonzero_start_qids': nonzero_start_qids, 'invalid_qids': invalid_qid_errors, 'choice_count_errors': choice_count_errors, 'empty_choice_errors': empty_choice_errors, 'gold_index_errors': gold_index_errors, 'official_type_distribution': dict(sorted(type_distribution.items())), 'gold_choice_distribution': dict(sorted(gold_distribution.items())), 'questions_per_video': {'min': min(question_counts.values(), default=0), 'max': max(question_counts.values(), default=0), 'mean': sum(question_counts.values()) / len(question_counts) if question_counts else 0.0, 'distribution': dict(sorted(Counter(question_counts.values()).items()))}}
    return (rows, audit)

def build_runtime_questions(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build gold-free records permitted in runtime Question IR preparation."""
    return [{'question_id': str(row['question_id']), 'video_id': str(row['official_video_id']), 'official_qid': str(row['official_qid']), 'official_type': str(row['official_type']), 'normalized_question': str(row['normalized_question']), 'answer_choices': list(row['answer_choices']), 'answer_format': 'multiple_choice_5'} for row in rows]

def build_evaluator_labels(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the evaluator-only label file kept outside runtime artifacts."""
    return [{'question_id': str(row['question_id']), 'video_id': str(row['official_video_id']), 'gold_choice_index': int(row['gold_choice_index'])} for row in rows]

def select_pilot_video_ids(manifest_videos: Sequence[Mapping[str, Any]], requested_local_ids: Sequence[int]=(1, 2, 285, 286, 570)) -> list[str]:
    """Select a deterministic five-video boundary pilot spanning both halves."""
    by_local_id = {int(video['local_sequential_id']): video for video in manifest_videos}
    selected = [str(by_local_id[local_id]['official_video_id']) for local_id in requested_local_ids if local_id in by_local_id]
    if len(selected) < 5:
        for video in sorted(manifest_videos, key=lambda item: int(item['local_sequential_id'])):
            video_id = str(video['official_video_id'])
            if video_id not in selected:
                selected.append(video_id)
            if len(selected) == 5:
                break
    return selected

def contains_gold_label(value: Any) -> bool:
    """Recursively detect evaluator-only fields in a runtime artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {'gold', 'gold_choice_index', 'correct_answer', 'answer_index'}:
                return True
            if contains_gold_label(child):
                return True
    elif isinstance(value, list):
        return any((contains_gold_label(child) for child in value))
    return False
