from __future__ import annotations
import csv
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
FIELD_SEQ_ID = 'Seq ID'
FIELD_UUID = 'UUID'
FIELD_QUESTION = 'Question'
FIELD_TIMESTAMP = 'Timestamp \nwith answer'
FIELD_TYPE = 'Type \n(binary, position, time, text)'
FIELD_TEXT_ANSWER = 'Text answer'
FIELD_PARSABLE_ANSWER = 'Parsable answer'
FIELD_CATEGORY = 'Question\nCategory'

@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float] | None = None
    lidar_frame: int | None = None

@dataclass(frozen=True)
class PositionWindow:
    window_id: str
    start_time: float
    end_time: float
    position: tuple[float, float, float]
    sample_count: int
    file_start: str
    file_end: str

@dataclass(frozen=True)
class PoseLookupResult:
    timestamp: float
    position: tuple[float, float, float]
    window_id: str
    window_start_time: float
    window_end_time: float
    delta_seconds: float
    sample_count: int

def load_navqa_rows(csv_path: Path, *, seq_id: str) -> list[dict[str, str]]:
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        return [row for row in csv.DictReader(handle) if str(row.get(FIELD_SEQ_ID, '')).strip() == str(seq_id)]

def load_question_metadata(qa_unfilled_json: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(qa_unfilled_json.read_text(encoding='utf-8-sig'))
    metadata: dict[str, dict[str, Any]] = {}
    for item in payload.get('data', []):
        if not isinstance(item, dict):
            continue
        question_id = str(item.get('id', '')).strip()
        if question_id:
            metadata[question_id] = item
    return metadata

def load_timestamps(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped:
            values.append(float(stripped.split()[0]))
    if not values:
        raise ValueError(f'No timestamps found in {path}')
    return values

def load_dense_poses(path: Path) -> list[PoseSample]:
    samples: list[PoseSample] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(item) for item in stripped.split()]
        if len(values) != 8:
            raise ValueError(f'Expected 8 pose values on line {line_number} in {path}')
        timestamp, x, y, z, qw, qx, qy, qz = values
        samples.append(PoseSample(timestamp=timestamp, position=(x, y, z), rotation=(qw, qx, qy, qz)))
    if not samples:
        raise ValueError(f'No dense poses found in {path}')
    return samples

def coda_preprocessed_pose_samples(dense_poses: list[PoseSample], lidar_timestamps: list[float]) -> list[PoseSample]:
    """Mimic ReMEmbR's CODa preprocessing without materializing image pkl files."""
    output: list[PoseSample] = []
    last_frame = 0
    for pose in dense_poses:
        frame = bisect_left(lidar_timestamps, pose.timestamp)
        if frame == last_frame:
            continue
        last_frame = frame
        output.append(PoseSample(timestamp=pose.timestamp, position=pose.position, rotation=pose.rotation, lidar_frame=frame))
    if not output:
        raise ValueError('No preprocessed pose samples were produced')
    return output

def build_position_windows_from_pose_samples(samples: list[PoseSample], *, seconds_per_caption: float=3.0, include_final_segment: bool=False) -> list[PositionWindow]:
    """Build caption-aligned position windows following ReMEmbR's caption segmentation."""
    ordered = sorted(samples, key=lambda item: item.timestamp)
    if not ordered:
        raise ValueError('Cannot build position windows from empty samples')
    windows: list[PositionWindow] = []
    current: list[PoseSample] = []
    time_start = ordered[0].timestamp
    for sample in ordered:
        if sample.timestamp - time_start > seconds_per_caption:
            if current:
                windows.append(_position_window_from_samples(current))
            current = [sample]
            time_start = sample.timestamp
        else:
            current.append(sample)
    if include_final_segment and current:
        windows.append(_position_window_from_samples(current))
    if not windows:
        raise ValueError('No position windows were produced')
    return windows

def load_position_windows_from_captions(captions_json: Path) -> list[PositionWindow]:
    payload = json.loads(captions_json.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, list):
        raise ValueError(f'Captions JSON must contain a list: {captions_json}')
    windows: list[PositionWindow] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        position = _position_tuple(item.get('position'))
        start = _time_from_caption_file(item.get('file_start'), fallback=item.get('time'))
        end = _time_from_caption_file(item.get('file_end'), fallback=item.get('time'))
        if position is None or start is None:
            continue
        end = end if end is not None else start
        windows.append(PositionWindow(window_id=str(item.get('id') or f'caption:{index:06d}'), start_time=float(start), end_time=float(end), position=position, sample_count=int(item.get('sample_count') or 0), file_start=str(item.get('file_start') or f'{start}.pkl'), file_end=str(item.get('file_end') or f'{end}.pkl')))
    if not windows:
        raise ValueError(f'No caption position windows found in {captions_json}')
    return sorted(windows, key=lambda item: item.start_time)

def lookup_position_for_timestamp(position_windows: list[PositionWindow], timestamp: float) -> PoseLookupResult | None:
    """Return the closest ego-pose window for a graph-predicted event time."""
    windows = sorted(position_windows, key=lambda item: item.start_time)
    if not windows:
        return None
    starts = [item.start_time for item in windows]
    index = _nearest_window_index(starts, float(timestamp))
    window = windows[index]
    if window.start_time <= timestamp <= window.end_time:
        delta = 0.0
    else:
        delta = min(abs(timestamp - window.start_time), abs(timestamp - window.end_time))
    return PoseLookupResult(timestamp=float(timestamp), position=window.position, window_id=window.window_id, window_start_time=window.start_time, window_end_time=window.end_time, delta_seconds=float(delta), sample_count=window.sample_count)

def build_position_reference(*, navqa_rows: list[dict[str, str]], question_metadata: dict[str, dict[str, Any]], position_windows: list[PositionWindow], seq_id: str) -> dict[str, Any]:
    windows = sorted(position_windows, key=lambda item: item.start_time)
    window_starts = [item.start_time for item in windows]
    answers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in navqa_rows:
        qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
        if qtype != 'position':
            continue
        uuid = str(row.get(FIELD_UUID, '')).strip()
        metadata = question_metadata.get(uuid)
        timestamps = _answer_timestamp_values(row.get(FIELD_TIMESTAMP, ''))
        if metadata is None:
            skipped.append({'uuid': uuid, 'reason': 'missing_question_metadata'})
            continue
        if len(timestamps) != 1:
            skipped.append({'uuid': uuid, 'reason': 'position_answer_requires_one_timestamp', 'timestamp_text': row.get(FIELD_TIMESTAMP, '')})
            continue
        answer_timestamp = timestamp_from_hms_near_reference(timestamps[0], question_start_time=float(metadata['start_time']), reference_time=float(metadata.get('end_time', metadata['start_time'])))
        window_index = _official_context_window_index(window_starts, answer_timestamp)
        window = windows[window_index]
        answers.append({'uuid': uuid, 'seq_id': seq_id, 'question': str(row.get(FIELD_QUESTION, '')).strip(), 'question_type': row.get(FIELD_TYPE, ''), 'question_category': row.get(FIELD_CATEGORY, ''), 'answer_timestamp_text': timestamps[0], 'answer_timestamp_unix': answer_timestamp, 'answer_timestamp_iso': datetime.fromtimestamp(answer_timestamp).isoformat(timespec='milliseconds'), 'position': [round(value, 6) for value in window.position], 'position_semantics': 'ego_pose_at_semantic_event_time', 'event_time_source': 'official_answer_timestamp_for_label_materialization_only', 'source_window': {'window_id': window.window_id, 'start_time': window.start_time, 'end_time': window.end_time, 'file_start': window.file_start, 'file_end': window.file_end, 'sample_count': window.sample_count}, 'generation_policy': 'official_answer_timestamp_to_ego_pose_window'})
    return {'contract': 'navqa_position_reference_v1', 'seq_id': seq_id, 'position_semantics': 'ego_pose_at_semantic_event_time', 'uses_graph': False, 'inject_into_graph': False, 'uses_entity_3d_coordinate': False, 'uses_official_position_coordinate': False, 'uses_official_answer_timestamp': True, 'intended_use': 'evaluation_label_materialization_only; graph QA must retrieve the semantic event time before any external ego-pose rendering', 'position_answer_count': len(answers), 'skipped_count': len(skipped), 'position_answers': answers, 'skipped': skipped}

def write_position_reference_csv(path: Path, reference: dict[str, Any]) -> None:
    rows = list(reference.get('position_answers', []))
    fieldnames = ['uuid', 'seq_id', 'question', 'question_type', 'question_category', 'answer_timestamp_text', 'answer_timestamp_unix', 'position_semantics', 'event_time_source', 'x', 'y', 'z', 'source_window_start_time', 'source_window_end_time', 'source_window_sample_count']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            position = item.get('position', [None, None, None])
            source = item.get('source_window', {})
            writer.writerow({'uuid': item.get('uuid'), 'seq_id': item.get('seq_id'), 'question': item.get('question'), 'question_type': item.get('question_type'), 'question_category': item.get('question_category'), 'answer_timestamp_text': item.get('answer_timestamp_text'), 'answer_timestamp_unix': item.get('answer_timestamp_unix'), 'position_semantics': item.get('position_semantics'), 'event_time_source': item.get('event_time_source'), 'x': position[0], 'y': position[1], 'z': position[2], 'source_window_start_time': source.get('start_time'), 'source_window_end_time': source.get('end_time'), 'source_window_sample_count': source.get('sample_count')})

def write_completed_navqa_csv(*, source_csv: Path, out_csv: Path, reference: dict[str, Any]) -> None:
    answers = {str(item.get('uuid')): item for item in reference.get('position_answers', []) if isinstance(item, dict)}
    with source_csv.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    extra_fields = ['Completed position answer', 'Completed position source']
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        item = answers.get(str(row.get(FIELD_UUID, '')).strip())
        if not item:
            row.setdefault('Completed position answer', '')
            row.setdefault('Completed position source', '')
            continue
        position = item.get('position', [])
        row['Completed position answer'] = json.dumps(position, ensure_ascii=False)
        row['Completed position source'] = str(item.get('generation_policy', ''))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _position_window_from_samples(samples: list[PoseSample]) -> PositionWindow:
    count = len(samples)
    position = tuple((sum((sample.position[index] for sample in samples)) / count for index in range(3)))
    start = samples[0].timestamp
    end_sample = samples[1] if len(samples) > 1 else samples[0]
    return PositionWindow(window_id=f'pose_window:{start:.6f}', start_time=start, end_time=end_sample.timestamp, position=(float(position[0]), float(position[1]), float(position[2])), sample_count=count, file_start=f'{start}.pkl', file_end=f'{end_sample.timestamp}.pkl')

def _answer_timestamp_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or '').split(',') if item.strip()]

def timestamp_from_hms_near_reference(hms: str, *, question_start_time: float, reference_time: float | None=None) -> float:
    """Resolve a NaVQA wall-clock HMS label to the Unix time nearest a question.

    Official NaVQA annotations store answer timestamps as displayed H:M:S strings.
    The display timezone is not encoded in the CSV, so this adapter enumerates
    plausible UTC offsets and picks the candidate nearest the question window.
    This keeps evaluation-label materialization independent from the machine's
    local timezone and separate from graph evidence.
    """
    reference = reference_time if reference_time is not None else question_start_time
    parsed_hms = datetime.strptime(str(hms).strip(), '%H:%M:%S').time()
    base_date = datetime.fromtimestamp(question_start_time, UTC).date()
    candidates: list[float] = []
    for day_offset in range(-1, 2):
        candidate_dt = datetime.combine(base_date + timedelta(days=day_offset), parsed_hms, tzinfo=UTC)
        wall_clock_as_utc = float(candidate_dt.timestamp())
        for hour_offset in range(-14, 15):
            candidates.append(wall_clock_as_utc + hour_offset * 3600.0)
    return min(candidates, key=lambda value: abs(value - reference))

def _official_context_window_index(window_starts: list[float], timestamp: float) -> int:
    first_after = bisect_right(window_starts, timestamp)
    return first_after - 1 if first_after > 0 else -1

def _nearest_window_index(window_starts: list[float], timestamp: float) -> int:
    first_after = bisect_left(window_starts, timestamp)
    if first_after <= 0:
        return 0
    if first_after >= len(window_starts):
        return len(window_starts) - 1
    before = first_after - 1
    after = first_after
    if abs(timestamp - window_starts[before]) <= abs(timestamp - window_starts[after]):
        return before
    return after

def _position_tuple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None

def _time_from_caption_file(value: Any, *, fallback: Any=None) -> float | None:
    text = str(value or '').strip()
    if text.endswith('.pkl'):
        text = text[:-4]
    if not text and fallback is not None:
        text = str(fallback)
    try:
        return float(Path(text).name)
    except (TypeError, ValueError):
        return None
