from __future__ import annotations
import json
import math
import re
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from vlkg.core.models.temporal_graph import BoundingBox
from vlkg.core.temporal.pathing import REPO_ROOT, resolve_repo_input_path
FRAME_NAME_NUMBERS_RE = re.compile('\\d+')

@dataclass(frozen=True)
class CGGraphReference:
    cg_id: str
    canonical_graph_json: Path | None = None
    temporal_graph_json: Path | None = None

@dataclass(frozen=True)
class EvidenceSequenceSpec:
    seq_id: str
    cam: str = '0'
    stride: int | None = 10
    sample_root: Path = Path()
    hsgg_input_dir: Path | None = None
    images_dir: Path | None = None
    frame_time_index_json: Path | None = None
    qa_unfilled_json: Path | None = None
    cg_graphs: tuple[CGGraphReference, ...] = ()

@dataclass(frozen=True)
class EvidenceBuildConfig:
    sequence_manifest: Path = Path('configs/datasets/navqa/sequences.json')
    cg_manifests: tuple[Path, ...] = (Path('configs/datasets/navqa/sequences.json'),)
    output_dir: Path = Path('datasets/navqa/runs/new_runs/payload/evidence_index_build')
    seq_ids: tuple[str, ...] = ()
    max_frames: int | None = None
    max_neighbors_per_object: int = 8
    write_jsonl_sidecars: bool = False
    write_spatial_jsonl: bool = False
    materialize_spatial_table: bool = False
    force: bool = False

@dataclass
class EvidenceBuildStats:
    sequences: int = 0
    frames: int = 0
    images: int = 0
    frame_jsons: int = 0
    observations: int = 0
    attribute_facts: int = 0
    relation_facts: int = 0
    spatial_facts: int = 0
    missing_images: int = 0
    missing_frame_time_rows: int = 0
    skipped_invalid_boxes: int = 0
    sequence_stats: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {'sequences': self.sequences, 'frames': self.frames, 'images': self.images, 'frame_jsons': self.frame_jsons, 'observations': self.observations, 'attribute_facts': self.attribute_facts, 'relation_facts': self.relation_facts, 'spatial_facts': self.spatial_facts, 'missing_images': self.missing_images, 'missing_frame_time_rows': self.missing_frame_time_rows, 'skipped_invalid_boxes': self.skipped_invalid_boxes, 'sequence_stats': self.sequence_stats}

@dataclass(frozen=True)
class FrameEvidenceRecords:
    observations: list[dict[str, Any]]
    attribute_facts: list[dict[str, Any]]
    relation_facts: list[dict[str, Any]]
    spatial_facts: list[dict[str, Any]]
    skipped_invalid_boxes: int = 0

def repo_relative_or_absolute(path: Path | str | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)

def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def append_jsonl(handle: Any, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        count += 1
    return count

def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip('-').isdigit():
            return int(text)
    return None

def optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None

class EvidenceLayerSQLiteWriter:

    def __init__(self, path: Path, *, force: bool=False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not force:
                raise FileExistsError(f'Output already exists: {self.path}')
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('PRAGMA journal_mode=OFF')
        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('PRAGMA temp_store=MEMORY')
        self.connection.execute('PRAGMA locking_mode=EXCLUSIVE')
        self.create_schema()

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def create_schema(self) -> None:
        self.connection.executescript('\n            CREATE TABLE metadata (\n                key TEXT PRIMARY KEY,\n                value_json TEXT NOT NULL\n            );\n\n            CREATE TABLE sequences (\n                seq_id TEXT PRIMARY KEY,\n                cam TEXT,\n                stride INTEGER,\n                sample_root TEXT,\n                images_dir TEXT,\n                hsgg_input_dir TEXT,\n                frame_time_index_json TEXT,\n                qa_unfilled_json TEXT,\n                jpg_count INTEGER,\n                frame_json_count INTEGER,\n                resource_json TEXT NOT NULL\n            );\n\n            CREATE TABLE cg_graph_refs (\n                seq_id TEXT NOT NULL,\n                cg_id TEXT NOT NULL,\n                canonical_graph_json TEXT,\n                canonical_graph_bytes INTEGER,\n                temporal_graph_json TEXT,\n                temporal_graph_bytes INTEGER,\n                PRIMARY KEY (seq_id, cg_id)\n            );\n\n            CREATE TABLE frames (\n                seq_id TEXT NOT NULL,\n                frame_key TEXT NOT NULL,\n                image_name TEXT,\n                image_path TEXT,\n                image_exists INTEGER NOT NULL,\n                sampled_index INTEGER,\n                graph_frame_index INTEGER,\n                source_frame INTEGER,\n                timestamp_unix REAL,\n                timestamp_iso TEXT,\n                elapsed_sec_from_sequence_start REAL,\n                has_frame_time_row INTEGER NOT NULL,\n                observation_count INTEGER NOT NULL,\n                attribute_fact_count INTEGER NOT NULL,\n                relation_fact_count INTEGER NOT NULL,\n                spatial_fact_count INTEGER NOT NULL,\n                PRIMARY KEY (seq_id, frame_key)\n            );\n\n            CREATE TABLE observations (\n                observation_id TEXT PRIMARY KEY,\n                seq_id TEXT NOT NULL,\n                frame_key TEXT NOT NULL,\n                graph_frame_index INTEGER,\n                elapsed_sec_from_sequence_start REAL,\n                image_name TEXT,\n                image_path TEXT,\n                image_exists INTEGER NOT NULL,\n                object_index INTEGER NOT NULL,\n                label TEXT,\n                score REAL,\n                x1 REAL NOT NULL,\n                y1 REAL NOT NULL,\n                x2 REAL NOT NULL,\n                y2 REAL NOT NULL,\n                bbox_key TEXT,\n                attribute_values_json TEXT NOT NULL,\n                raw_json TEXT NOT NULL\n            );\n\n            CREATE TABLE attribute_facts (\n                fact_id TEXT PRIMARY KEY,\n                seq_id TEXT NOT NULL,\n                frame_key TEXT NOT NULL,\n                graph_frame_index INTEGER,\n                observation_id TEXT NOT NULL,\n                object_index INTEGER NOT NULL,\n                label TEXT,\n                attribute TEXT NOT NULL,\n                image_name TEXT,\n                raw_json TEXT NOT NULL\n            );\n\n            CREATE TABLE relation_facts (\n                fact_id TEXT PRIMARY KEY,\n                seq_id TEXT NOT NULL,\n                frame_key TEXT NOT NULL,\n                graph_frame_index INTEGER,\n                subject_observation_id TEXT NOT NULL,\n                subject_object_index INTEGER NOT NULL,\n                subject_label TEXT,\n                predicate TEXT,\n                object_observation_id TEXT NOT NULL,\n                object_object_index INTEGER NOT NULL,\n                object_label TEXT,\n                hsgg_source TEXT,\n                raw_output TEXT,\n                raw_json TEXT NOT NULL\n            );\n\n            CREATE TABLE spatial_facts (\n                fact_id TEXT PRIMARY KEY,\n                seq_id TEXT NOT NULL,\n                frame_key TEXT NOT NULL,\n                graph_frame_index INTEGER,\n                subject_observation_id TEXT NOT NULL,\n                subject_object_index INTEGER NOT NULL,\n                subject_label TEXT,\n                object_observation_id TEXT NOT NULL,\n                object_object_index INTEGER NOT NULL,\n                object_label TEXT,\n                relations_json TEXT NOT NULL,\n                center_dx REAL,\n                center_dy REAL,\n                normalized_center_distance REAL,\n                iou REAL,\n                raw_json TEXT NOT NULL\n            );\n            ')

    def insert_metadata(self, key: str, value: Any) -> None:
        self.connection.execute('INSERT OR REPLACE INTO metadata(key, value_json) VALUES (?, ?)', (key, json_dumps(value)))

    def insert_sequence(self, entry: dict[str, Any]) -> None:
        self.connection.execute('\n            INSERT OR REPLACE INTO sequences (\n                seq_id, cam, stride, sample_root, images_dir, hsgg_input_dir,\n                frame_time_index_json, qa_unfilled_json, jpg_count,\n                frame_json_count, resource_json\n            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (str(entry.get('seq_id') or ''), str(entry.get('cam') or ''), optional_int(entry.get('stride')), entry.get('sample_root'), entry.get('images_dir'), entry.get('hsgg_input_dir'), entry.get('frame_time_index_json'), entry.get('qa_unfilled_json'), optional_int(entry.get('jpg_count')), optional_int(entry.get('frame_json_count')), json_dumps(entry)))
        for ref in entry.get('cg_graphs') or []:
            if not isinstance(ref, dict):
                continue
            self.connection.execute('\n                INSERT OR REPLACE INTO cg_graph_refs (\n                    seq_id, cg_id, canonical_graph_json, canonical_graph_bytes,\n                    temporal_graph_json, temporal_graph_bytes\n                ) VALUES (?, ?, ?, ?, ?, ?)\n                ', (str(entry.get('seq_id') or ''), str(ref.get('cg_id') or ''), ref.get('canonical_graph_json'), optional_int(ref.get('canonical_graph_bytes')), ref.get('temporal_graph_json'), optional_int(ref.get('temporal_graph_bytes'))))

    def insert_frame_records(self, records: FrameEvidenceRecords) -> None:
        if records.observations:
            first = records.observations[0]
            self.connection.execute('\n                INSERT OR REPLACE INTO frames (\n                    seq_id, frame_key, image_name, image_path, image_exists,\n                    sampled_index, graph_frame_index, source_frame,\n                    timestamp_unix, timestamp_iso, elapsed_sec_from_sequence_start,\n                    has_frame_time_row, observation_count, attribute_fact_count,\n                    relation_fact_count, spatial_fact_count\n                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                ', (str(first.get('seq_id') or ''), str(first.get('frame_key') or ''), first.get('image_name'), first.get('image_path'), int(bool(first.get('image_exists'))), optional_int(first.get('sampled_index')), optional_int(first.get('graph_frame_index')), optional_int(first.get('source_frame')), optional_float(first.get('timestamp_unix')), first.get('timestamp_iso'), optional_float(first.get('elapsed_sec_from_sequence_start')), int(bool(first.get('has_frame_time_row'))), len(records.observations), len(records.attribute_facts), len(records.relation_facts), len(records.spatial_facts)))
        self.insert_observations(records.observations)
        self.insert_attribute_facts(records.attribute_facts)
        self.insert_relation_facts(records.relation_facts)
        self.insert_spatial_facts(records.spatial_facts)

    def insert_observations(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.connection.executemany('\n            INSERT OR REPLACE INTO observations (\n                observation_id, seq_id, frame_key, graph_frame_index,\n                elapsed_sec_from_sequence_start, image_name, image_path,\n                image_exists, object_index, label, score, x1, y1, x2, y2,\n                bbox_key, attribute_values_json, raw_json\n            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', [observation_row(row) for row in rows])

    def insert_attribute_facts(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.connection.executemany('\n            INSERT OR REPLACE INTO attribute_facts (\n                fact_id, seq_id, frame_key, graph_frame_index, observation_id,\n                object_index, label, attribute, image_name, raw_json\n            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', [attribute_fact_row(row) for row in rows])

    def insert_relation_facts(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.connection.executemany('\n            INSERT OR REPLACE INTO relation_facts (\n                fact_id, seq_id, frame_key, graph_frame_index,\n                subject_observation_id, subject_object_index, subject_label,\n                predicate, object_observation_id, object_object_index,\n                object_label, hsgg_source, raw_output, raw_json\n            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', [relation_fact_row(row) for row in rows])

    def insert_spatial_facts(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.connection.executemany('\n            INSERT OR REPLACE INTO spatial_facts (\n                fact_id, seq_id, frame_key, graph_frame_index,\n                subject_observation_id, subject_object_index, subject_label,\n                object_observation_id, object_object_index, object_label,\n                relations_json, center_dx, center_dy,\n                normalized_center_distance, iou, raw_json\n            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', [spatial_fact_row(row) for row in rows])

    def create_indexes(self) -> None:
        self.connection.executescript('\n            CREATE INDEX IF NOT EXISTS idx_frames_seq_graph_frame\n                ON frames(seq_id, graph_frame_index);\n            CREATE INDEX IF NOT EXISTS idx_observations_seq_frame\n                ON observations(seq_id, frame_key, object_index);\n            CREATE INDEX IF NOT EXISTS idx_observations_seq_graph_frame\n                ON observations(seq_id, graph_frame_index, object_index);\n            CREATE INDEX IF NOT EXISTS idx_observations_label\n                ON observations(label);\n            CREATE INDEX IF NOT EXISTS idx_observations_bbox\n                ON observations(seq_id, frame_key, bbox_key);\n            CREATE INDEX IF NOT EXISTS idx_attribute_observation\n                ON attribute_facts(observation_id);\n            CREATE INDEX IF NOT EXISTS idx_attribute_seq_frame\n                ON attribute_facts(seq_id, frame_key);\n            CREATE INDEX IF NOT EXISTS idx_relation_subject\n                ON relation_facts(subject_observation_id);\n            CREATE INDEX IF NOT EXISTS idx_relation_object\n                ON relation_facts(object_observation_id);\n            CREATE INDEX IF NOT EXISTS idx_relation_seq_frame\n                ON relation_facts(seq_id, frame_key);\n            CREATE INDEX IF NOT EXISTS idx_spatial_subject\n                ON spatial_facts(subject_observation_id);\n            CREATE INDEX IF NOT EXISTS idx_spatial_object\n                ON spatial_facts(object_observation_id);\n            CREATE INDEX IF NOT EXISTS idx_spatial_seq_frame\n                ON spatial_facts(seq_id, frame_key);\n            ')
        self.connection.execute('ANALYZE')

def observation_row(row: dict[str, Any]) -> tuple[Any, ...]:
    box = row.get('bbox') or [None, None, None, None]
    return (row['observation_id'], str(row.get('seq_id') or ''), str(row.get('frame_key') or ''), optional_int(row.get('graph_frame_index')), optional_float(row.get('elapsed_sec_from_sequence_start')), row.get('image_name'), row.get('image_path'), int(bool(row.get('image_exists'))), optional_int(row.get('object_index')), row.get('label'), optional_float(row.get('score')), float(box[0]), float(box[1]), float(box[2]), float(box[3]), row.get('bbox_key'), json_dumps(row.get('attribute_values') or []), json_dumps(row))

def attribute_fact_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row['fact_id'], str(row.get('seq_id') or ''), str(row.get('frame_key') or ''), optional_int(row.get('graph_frame_index')), row['observation_id'], optional_int(row.get('object_index')), row.get('label'), str(row.get('attribute') or ''), row.get('image_name'), json_dumps(row))

def relation_fact_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row['fact_id'], str(row.get('seq_id') or ''), str(row.get('frame_key') or ''), optional_int(row.get('graph_frame_index')), row['subject_observation_id'], optional_int(row.get('subject_object_index')), row.get('subject_label'), row.get('predicate'), row['object_observation_id'], optional_int(row.get('object_object_index')), row.get('object_label'), row.get('hsgg_source'), row.get('raw_output'), json_dumps(row))

def spatial_fact_row(row: dict[str, Any]) -> tuple[Any, ...]:
    geometry = row.get('geometry') or {}
    return (row['fact_id'], str(row.get('seq_id') or ''), str(row.get('frame_key') or ''), optional_int(row.get('graph_frame_index')), row['subject_observation_id'], optional_int(row.get('subject_object_index')), row.get('subject_label'), row['object_observation_id'], optional_int(row.get('object_object_index')), row.get('object_label'), json_dumps(row.get('relations') or []), optional_float(geometry.get('center_dx')), optional_float(geometry.get('center_dy')), optional_float(geometry.get('normalized_center_distance')), optional_float(geometry.get('iou')), json_dumps(row))

from vlkg.core.indexing.evidence_layer_reader import EvidenceLayerIndex

def build_spatial_fact_for_pair(subject: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    subject_box = bounding_box(subject.get('bbox') or [])
    obj_box = bounding_box(obj.get('bbox') or [])
    if subject_box is None or obj_box is None:
        raise ValueError('Both observations must have valid bbox values.')
    return {'contract': 'navqa_evidence_bbox_spatial_fact_v1', 'fact_id': f"{subject['seq_id']}:frame:{subject['frame_key']}:spatial:{subject['object_index']}:{obj['object_index']}", 'seq_id': subject['seq_id'], 'frame_key': subject['frame_key'], 'graph_frame_index': subject.get('graph_frame_index'), 'subject_observation_id': subject['observation_id'], 'subject_object_index': subject['object_index'], 'subject_label': subject['label'], 'object_observation_id': obj['observation_id'], 'object_object_index': obj['object_index'], 'object_label': obj['label'], 'relations': spatial_relation_candidates(subject_box, obj_box), 'geometry': bbox_geometry(subject_box, obj_box)}

def normalize_seq_id(seq_id: str | int) -> str:
    value = str(seq_id).strip()
    if value.startswith('seq'):
        value = value[3:]
    if not value:
        raise ValueError('seq_id must not be empty')
    return value

def bbox_values(raw_box: Iterable[Any]) -> list[float] | None:
    values = list(raw_box)
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(values[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all((math.isfinite(value) for value in (x1, y1, x2, y2))):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return [x1, y1, x2, y2]

def bbox_key(raw_box: Iterable[Any]) -> str | None:
    values = bbox_values(raw_box)
    if values is None:
        return None
    return ','.join((f'{value:.3f}' for value in values))

def bounding_box(raw_box: Iterable[Any]) -> BoundingBox | None:
    values = bbox_values(raw_box)
    if values is None:
        return None
    return BoundingBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3])

def normalize_attribute_values(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else []
    if isinstance(raw_value, (int, float, bool)):
        return [str(raw_value)]
    if isinstance(raw_value, list):
        values: list[str] = []
        for item in raw_value:
            values.extend(normalize_attribute_values(item))
        return values
    if isinstance(raw_value, dict):
        values = []
        for key, value in raw_value.items():
            nested = normalize_attribute_values(value)
            if nested:
                values.extend((f'{key}: {item}' for item in nested))
        return values
    return [str(raw_value)]

def load_frame_time_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get('frames') if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        keys = {str(row.get('image_name') or ''), Path(str(row.get('image_name') or '')).stem}
        for field_name in ('sampled_index', 'graph_frame_index', 'source_frame'):
            value = row.get(field_name)
            if value is not None:
                keys.add(str(value))
        for key in keys:
            if key:
                by_key[key] = row
    return by_key

def image_name_from_frame(frame_key: str, frame_payload: dict[str, Any]) -> str:
    image_path = str(frame_payload.get('image_path') or '')
    if image_path:
        return Path(image_path.replace('\\', '/')).name
    if Path(frame_key).suffix:
        return Path(frame_key).with_suffix('.jpg').name
    return f'{frame_key}.jpg'

def resolve_image_path(*, images_dir: Path | None, frame_key: str, frame_payload: dict[str, Any]) -> Path | None:
    image_name = image_name_from_frame(frame_key, frame_payload)
    if images_dir is not None:
        candidate = images_dir / image_name
        if candidate.exists():
            return candidate
        matches = sorted(images_dir.glob(f'{Path(image_name).stem}.*'))
        if matches:
            return matches[0]
    image_path = str(frame_payload.get('image_path') or '')
    if image_path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            if candidate.exists() or images_dir is None:
                return candidate
        resolved = resolve_repo_input_path(candidate)
        if resolved.exists():
            return resolved
    if images_dir is None:
        return None
    return images_dir / image_name

def frame_info_for(*, frame_key: str, image_name: str, frame_time_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    keys = [image_name, Path(image_name).stem, frame_key]
    parsed = parse_navqa_frame_name(frame_key)
    for field_name in ('sampled_index', 'graph_frame_index', 'source_frame'):
        value = parsed.get(field_name)
        if value is not None:
            keys.append(str(value))
    for key in keys:
        if key in frame_time_index:
            return (frame_time_index[key], True)
    fallback = {key: value for key, value in parsed.items() if value is not None}
    return (fallback, False)

def parse_navqa_frame_name(frame_key: str) -> dict[str, int | None]:
    stem = Path(frame_key).stem
    numbers = [int(item) for item in FRAME_NAME_NUMBERS_RE.findall(stem)]
    sampled_index = numbers[0] if numbers else None
    seq_id = numbers[-2] if len(numbers) >= 2 else None
    source_frame = numbers[-1] if numbers else None
    return {'sampled_index': sampled_index, 'graph_frame_index': source_frame, 'source_frame': source_frame, 'parsed_seq_id': seq_id}

def relation_indices(relation: dict[str, Any]) -> tuple[int, int] | None:
    raw_idx = relation.get('idx')
    if not isinstance(raw_idx, list) or len(raw_idx) < 2:
        return None
    try:
        return (int(raw_idx[0]), int(raw_idx[1]))
    except (TypeError, ValueError):
        return None

def build_frame_evidence_records(*, spec: EvidenceSequenceSpec, frame_key: str, frame_payload: dict[str, Any], frame_time_index: dict[str, dict[str, Any]] | None=None, max_neighbors_per_object: int=8) -> FrameEvidenceRecords:
    objects_payload = frame_payload.get('objects')
    if not isinstance(objects_payload, dict):
        objects_payload = frame_payload
    labels = objects_payload.get('labels') or []
    boxes = objects_payload.get('boxes') or []
    scores = objects_payload.get('scores') or []
    attributes = frame_payload.get('attributes') or {}
    relations = frame_payload.get('relations') or []
    image_path = resolve_image_path(images_dir=spec.images_dir, frame_key=frame_key, frame_payload={**objects_payload, 'image_path': frame_payload.get('image_path') or objects_payload.get('image_path')})
    image_name = image_path.name if image_path else image_name_from_frame(frame_key, frame_payload)
    frame_info, has_frame_time_row = frame_info_for(frame_key=frame_key, image_name=image_name, frame_time_index=frame_time_index or {})
    observations: list[dict[str, Any]] = []
    observation_by_index: dict[int, dict[str, Any]] = {}
    skipped_invalid_boxes = 0
    for object_index, raw_box in enumerate(boxes):
        box = bbox_values(raw_box)
        key = bbox_key(raw_box)
        if box is None or key is None:
            skipped_invalid_boxes += 1
            continue
        label = labels[object_index] if object_index < len(labels) else ''
        score = scores[object_index] if object_index < len(scores) else None
        observation_id = f'seq{spec.seq_id}:frame:{Path(frame_key).stem}:object:{object_index}'
        attr_values = normalize_attribute_values(attributes.get(str(object_index)))
        observation = {'contract': 'navqa_evidence_observation_v1', 'observation_id': observation_id, 'seq_id': spec.seq_id, 'cam': spec.cam, 'stride': spec.stride, 'frame_key': Path(frame_key).stem, 'image_name': image_name, 'image_path': repo_relative_or_absolute(image_path) if image_path else None, 'image_exists': bool(image_path and image_path.exists()), 'object_index': object_index, 'label': str(label), 'score': float(score) if isinstance(score, (int, float)) else None, 'bbox': box, 'bbox_key': key, 'attribute_values': attr_values, 'sampled_index': frame_info.get('sampled_index'), 'graph_frame_index': frame_info.get('graph_frame_index'), 'source_frame': frame_info.get('source_frame'), 'timestamp_unix': frame_info.get('timestamp_unix'), 'timestamp_iso': frame_info.get('timestamp_iso'), 'elapsed_sec_from_sequence_start': frame_info.get('elapsed_sec_from_sequence_start'), 'has_frame_time_row': has_frame_time_row}
        observations.append(observation)
        observation_by_index[object_index] = observation
    attribute_facts = build_attribute_facts(observations)
    relation_facts = build_relation_facts(seq_id=spec.seq_id, frame_key=Path(frame_key).stem, relations=relations if isinstance(relations, list) else [], observation_by_index=observation_by_index)
    spatial_facts = build_spatial_facts(observations=observations, max_neighbors_per_object=max_neighbors_per_object)
    return FrameEvidenceRecords(observations=observations, attribute_facts=attribute_facts, relation_facts=relation_facts, spatial_facts=spatial_facts, skipped_invalid_boxes=skipped_invalid_boxes)

def build_attribute_facts(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for observation in observations:
        for index, value in enumerate(observation.get('attribute_values') or []):
            fact_id = f"{observation['observation_id']}:attr:{index}"
            facts.append({'contract': 'navqa_evidence_attribute_fact_v1', 'fact_id': fact_id, 'scope': 'direct_object_attribute', 'seq_id': observation['seq_id'], 'frame_key': observation['frame_key'], 'graph_frame_index': observation.get('graph_frame_index'), 'owner_observation_id': observation['observation_id'], 'owner_object_index': observation['object_index'], 'owner_label': observation['label'], 'observation_id': observation['observation_id'], 'object_index': observation['object_index'], 'label': observation['label'], 'attribute': value, 'bbox': observation['bbox'], 'image_name': observation['image_name']})
    return facts

def build_relation_facts(*, seq_id: str, frame_key: str, relations: list[Any], observation_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            continue
        pair = relation_indices(relation)
        if pair is None:
            continue
        subject_index, object_index = pair
        subject = observation_by_index.get(subject_index)
        obj = observation_by_index.get(object_index)
        if subject is None or obj is None:
            continue
        fact_id = f'seq{seq_id}:frame:{frame_key}:relation:{index}:{subject_index}:{object_index}'
        facts.append({'contract': 'navqa_evidence_hsgg_relation_fact_v1', 'fact_id': fact_id, 'seq_id': seq_id, 'frame_key': frame_key, 'graph_frame_index': subject.get('graph_frame_index'), 'subject_observation_id': subject['observation_id'], 'subject_object_index': subject_index, 'subject_label': relation.get('subject_label') or subject['label'], 'predicate': relation.get('predicate'), 'raw_output': relation.get('raw_output'), 'object_observation_id': obj['observation_id'], 'object_object_index': object_index, 'object_label': relation.get('object_label') or obj['label'], 'hsgg_source': relation.get('hsgg_source'), 'selected_from_candidates': relation.get('selected_from_candidates')})
    return facts

def build_spatial_facts(*, observations: list[dict[str, Any]], max_neighbors_per_object: int=8) -> list[dict[str, Any]]:
    if max_neighbors_per_object <= 0:
        return []
    boxes: dict[str, BoundingBox] = {}
    for observation in observations:
        box = bounding_box(observation.get('bbox') or [])
        if box is not None:
            boxes[observation['observation_id']] = box
    facts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for subject in observations:
        subject_box = boxes.get(subject['observation_id'])
        if subject_box is None:
            continue
        neighbors: list[tuple[float, dict[str, Any]]] = []
        for obj in observations:
            if obj['observation_id'] == subject['observation_id']:
                continue
            obj_box = boxes.get(obj['observation_id'])
            if obj_box is None:
                continue
            neighbors.append((subject_box.center_distance(obj_box), obj))
        neighbors.sort(key=lambda item: item[0])
        for _, obj in neighbors[:max_neighbors_per_object]:
            pair_key = tuple(sorted((subject['observation_id'], obj['observation_id'])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            facts.append(build_spatial_fact_for_pair(subject, obj))
    return facts

def bbox_geometry(subject_box: BoundingBox, object_box: BoundingBox) -> dict[str, float]:
    diagonal = max(subject_box.diagonal, object_box.diagonal, 1.0)
    return {'subject_center_x': subject_box.center_x, 'subject_center_y': subject_box.center_y, 'object_center_x': object_box.center_x, 'object_center_y': object_box.center_y, 'center_dx': object_box.center_x - subject_box.center_x, 'center_dy': object_box.center_y - subject_box.center_y, 'normalized_center_distance': subject_box.center_distance(object_box) / diagonal, 'iou': subject_box.iou(object_box), 'subject_inside_object_ratio': subject_box.inside_ratio(object_box), 'object_inside_subject_ratio': object_box.inside_ratio(subject_box), 'area_ratio': subject_box.area_ratio(object_box)}

def spatial_relation_candidates(subject_box: BoundingBox, object_box: BoundingBox) -> list[str]:
    relations: list[str] = []
    dx = object_box.center_x - subject_box.center_x
    dy = object_box.center_y - subject_box.center_y
    horizontal_threshold = max(subject_box.width, object_box.width, 1.0) * 0.15
    vertical_threshold = max(subject_box.height, object_box.height, 1.0) * 0.15
    if dx > horizontal_threshold:
        relations.append('subject_left_of_object')
    elif dx < -horizontal_threshold:
        relations.append('subject_right_of_object')
    if dy > vertical_threshold:
        relations.append('subject_above_object')
    elif dy < -vertical_threshold:
        relations.append('subject_below_object')
    if subject_box.iou(object_box) > 0.0:
        relations.append('overlap')
    if subject_box.inside_ratio(object_box) >= 0.75:
        relations.append('subject_inside_object')
    if object_box.inside_ratio(subject_box) >= 0.75:
        relations.append('object_inside_subject')
    distance = subject_box.center_distance(object_box)
    near_threshold = 1.25 * max(subject_box.diagonal, object_box.diagonal, 1.0)
    if distance <= near_threshold:
        relations.append('near_by_bbox')
    return relations

def load_evidence_sequence_specs(*, sequence_manifest: Path, cg_manifests: Iterable[Path]=(), seq_ids: Iterable[str]=()) -> list[EvidenceSequenceSpec]:
    seq_filter = {normalize_seq_id(seq_id) for seq_id in seq_ids}
    sequence_manifest = resolve_repo_input_path(sequence_manifest)
    payload = read_json(sequence_manifest)
    selected: dict[str, dict[str, Any]] = {}
    for item in payload.get('sequences') or []:
        if not isinstance(item, dict):
            continue
        seq_id = normalize_seq_id(item['seq_id'])
        if seq_filter and seq_id not in seq_filter:
            continue
        selected[seq_id] = item
    if not selected:
        raise ValueError(f'No sequences selected from {sequence_manifest}.')
    cg_refs = load_cg_graph_references(cg_manifests, selected.keys())
    specs: list[EvidenceSequenceSpec] = []
    for seq_id, item in sorted(selected.items(), key=lambda pair: int(pair[0])):
        sample_root = resolve_repo_input_path(item['sample_root'])
        hsgg_path = item.get('hsgg_input_dir') or sample_root / 'hsgg_local_paths'
        images_path = item.get('images_dir') or sample_root / 'images'
        frame_time = item.get('frame_time_index_json')
        qa_path = item.get('qa_unfilled_json')
        specs.append(EvidenceSequenceSpec(seq_id=seq_id, cam=str(item.get('cam', '0')), stride=item.get('stride'), sample_root=sample_root, hsgg_input_dir=resolve_repo_input_path(hsgg_path), images_dir=resolve_repo_input_path(images_path), frame_time_index_json=resolve_repo_input_path(frame_time) if frame_time else None, qa_unfilled_json=resolve_repo_input_path(qa_path) if qa_path else None, cg_graphs=tuple(cg_refs.get(seq_id, []))))
    return specs

def load_cg_graph_references(cg_manifests: Iterable[Path], seq_ids: Iterable[str]) -> dict[str, list[CGGraphReference]]:
    seq_set = {normalize_seq_id(seq_id) for seq_id in seq_ids}
    refs: dict[str, list[CGGraphReference]] = {seq_id: [] for seq_id in seq_set}
    for manifest in cg_manifests:
        path = resolve_repo_input_path(manifest)
        if not path.exists():
            continue
        payload = read_json(path)
        cg_id = str(payload.get('cg_id') or path.stem)
        for item in payload.get('sequences') or []:
            if not isinstance(item, dict):
                continue
            seq_id = normalize_seq_id(item.get('seq_id', ''))
            if seq_id not in refs:
                continue
            canonical = item.get('canonical_graph_json')
            temporal = item.get('temporal_graph_json')
            refs[seq_id].append(CGGraphReference(cg_id=cg_id, canonical_graph_json=resolve_repo_input_path(canonical) if canonical else None, temporal_graph_json=resolve_repo_input_path(temporal) if temporal else None))
    return refs

def iter_frame_payloads(spec: EvidenceSequenceSpec, *, max_frames: int | None=None) -> Iterable[tuple[str, dict[str, Any], Path | None]]:
    if spec.hsgg_input_dir is None:
        return
    frame_dir = spec.hsgg_input_dir / 'frame_scene_graphs'
    count = 0
    if frame_dir.exists():
        for path in sorted(frame_dir.glob('*.json')):
            if max_frames is not None and count >= max_frames:
                break
            payload = read_json(path)
            if isinstance(payload, dict):
                count += 1
                yield (path.stem, payload, path)
        return
    objects_path = spec.hsgg_input_dir / 'objects.json'
    if not objects_path.exists():
        return
    objects = read_json(objects_path)
    attributes = read_json(spec.hsgg_input_dir / 'attributes.json') if (spec.hsgg_input_dir / 'attributes.json').exists() else {}
    relations = read_json(spec.hsgg_input_dir / 'relations.json') if (spec.hsgg_input_dir / 'relations.json').exists() else {}
    if not isinstance(objects, dict):
        return
    for frame_key, object_payload in sorted(objects.items()):
        if max_frames is not None and count >= max_frames:
            break
        if not isinstance(object_payload, dict):
            continue
        count += 1
        yield (str(frame_key), {'image_path': object_payload.get('image_path'), 'objects': object_payload, 'attributes': attributes.get(frame_key, {}) if isinstance(attributes, dict) else {}, 'relations': relations.get(frame_key, []) if isinstance(relations, dict) else []}, None)

def resource_manifest_entry(spec: EvidenceSequenceSpec) -> dict[str, Any]:
    hsgg_dir = spec.hsgg_input_dir
    images_dir = spec.images_dir
    frame_scene_dir = hsgg_dir / 'frame_scene_graphs' if hsgg_dir else None
    jpg_count = len(list(images_dir.glob('*.jpg'))) if images_dir and images_dir.exists() else 0
    frame_json_count = len(list(frame_scene_dir.glob('*.json'))) if frame_scene_dir and frame_scene_dir.exists() else 0
    return {'seq_id': spec.seq_id, 'cam': spec.cam, 'stride': spec.stride, 'sample_root': repo_relative_or_absolute(spec.sample_root), 'images_dir': repo_relative_or_absolute(images_dir), 'images_dir_exists': bool(images_dir and images_dir.exists()), 'jpg_count': jpg_count, 'hsgg_input_dir': repo_relative_or_absolute(hsgg_dir), 'hsgg_input_dir_exists': bool(hsgg_dir and hsgg_dir.exists()), 'frame_scene_graphs_dir': repo_relative_or_absolute(frame_scene_dir), 'frame_json_count': frame_json_count, 'objects_json_exists': bool(hsgg_dir and (hsgg_dir / 'objects.json').exists()), 'attributes_json_exists': bool(hsgg_dir and (hsgg_dir / 'attributes.json').exists()), 'relations_json_exists': bool(hsgg_dir and (hsgg_dir / 'relations.json').exists()), 'frame_time_index_json': repo_relative_or_absolute(spec.frame_time_index_json), 'frame_time_index_exists': bool(spec.frame_time_index_json and spec.frame_time_index_json.exists()), 'qa_unfilled_json': repo_relative_or_absolute(spec.qa_unfilled_json), 'qa_unfilled_exists': bool(spec.qa_unfilled_json and spec.qa_unfilled_json.exists()), 'cg_graphs': [{'cg_id': ref.cg_id, 'canonical_graph_json': repo_relative_or_absolute(ref.canonical_graph_json), 'canonical_graph_exists': bool(ref.canonical_graph_json and ref.canonical_graph_json.exists()), 'canonical_graph_bytes': ref.canonical_graph_json.stat().st_size if ref.canonical_graph_json and ref.canonical_graph_json.exists() else None, 'temporal_graph_json': repo_relative_or_absolute(ref.temporal_graph_json), 'temporal_graph_exists': bool(ref.temporal_graph_json and ref.temporal_graph_json.exists()), 'temporal_graph_bytes': ref.temporal_graph_json.stat().st_size if ref.temporal_graph_json and ref.temporal_graph_json.exists() else None} for ref in spec.cg_graphs]}

def build_evidence_layer_index(config: EvidenceBuildConfig) -> dict[str, Any]:
    output_dir = resolve_repo_input_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'manifest': output_dir / 'manifest.json', 'summary': output_dir / 'summary.json', 'summary_md': output_dir / 'summary.md', 'sqlite': output_dir / 'evidence_layer.sqlite', 'observations': output_dir / 'observations.jsonl', 'attribute_facts': output_dir / 'attribute_facts.jsonl', 'relation_facts': output_dir / 'relation_facts.jsonl', 'spatial_facts': output_dir / 'spatial_facts.jsonl'}
    active_output_paths = [paths['manifest'], paths['summary'], paths['summary_md'], paths['sqlite']]
    if config.write_jsonl_sidecars:
        active_output_paths.extend([paths['observations'], paths['attribute_facts'], paths['relation_facts']])
    if config.write_spatial_jsonl:
        active_output_paths.append(paths['spatial_facts'])
    if not config.force:
        existing = [path for path in active_output_paths if path.exists()]
        if existing:
            raise FileExistsError('Output files already exist. Pass force=True/--force to overwrite: ' + ', '.join((str(path) for path in existing)))
    else:
        for path in paths.values():
            if path.exists():
                path.unlink()
    specs = load_evidence_sequence_specs(sequence_manifest=config.sequence_manifest, cg_manifests=config.cg_manifests, seq_ids=config.seq_ids)
    manifest_entries = [resource_manifest_entry(spec) for spec in specs]
    stats = EvidenceBuildStats(sequences=len(specs))
    sqlite_writer = EvidenceLayerSQLiteWriter(paths['sqlite'], force=True)
    try:
        sqlite_writer.insert_metadata('build_config', {'sequence_manifest': repo_relative_or_absolute(resolve_repo_input_path(config.sequence_manifest)), 'cg_manifests': [repo_relative_or_absolute(resolve_repo_input_path(path)) for path in config.cg_manifests], 'max_frames': config.max_frames, 'max_neighbors_per_object': config.max_neighbors_per_object, 'write_jsonl_sidecars': config.write_jsonl_sidecars, 'write_spatial_jsonl': config.write_spatial_jsonl, 'materialize_spatial_table': config.materialize_spatial_table})
        for manifest_entry in manifest_entries:
            sqlite_writer.insert_sequence(manifest_entry)
        with ExitStack() as stack:
            observation_handle = stack.enter_context(paths['observations'].open('w', encoding='utf-8')) if config.write_jsonl_sidecars else None
            attribute_handle = stack.enter_context(paths['attribute_facts'].open('w', encoding='utf-8')) if config.write_jsonl_sidecars else None
            relation_handle = stack.enter_context(paths['relation_facts'].open('w', encoding='utf-8')) if config.write_jsonl_sidecars else None
            spatial_handle = stack.enter_context(paths['spatial_facts'].open('w', encoding='utf-8')) if config.write_spatial_jsonl else None
            for spec, manifest_entry in zip(specs, manifest_entries, strict=True):
                frame_time_index = load_frame_time_index(spec.frame_time_index_json)
                seq_stats = {'seq_id': spec.seq_id, 'frames': 0, 'observations': 0, 'attribute_facts': 0, 'relation_facts': 0, 'spatial_facts': 0, 'spatial_facts_materialized': 0, 'missing_images': 0, 'missing_frame_time_rows': 0, 'skipped_invalid_boxes': 0}
                for frame_key, payload, _path in iter_frame_payloads(spec, max_frames=config.max_frames):
                    records = build_frame_evidence_records(spec=spec, frame_key=frame_key, frame_payload=payload, frame_time_index=frame_time_index, max_neighbors_per_object=config.max_neighbors_per_object)
                    db_records = records
                    if not config.materialize_spatial_table:
                        db_records = FrameEvidenceRecords(observations=records.observations, attribute_facts=records.attribute_facts, relation_facts=records.relation_facts, spatial_facts=[], skipped_invalid_boxes=records.skipped_invalid_boxes)
                    sqlite_writer.insert_frame_records(db_records)
                    seq_stats['frames'] += 1
                    seq_stats['observations'] += len(records.observations)
                    seq_stats['attribute_facts'] += len(records.attribute_facts)
                    seq_stats['relation_facts'] += len(records.relation_facts)
                    seq_stats['spatial_facts'] += len(records.spatial_facts)
                    seq_stats['spatial_facts_materialized'] += len(db_records.spatial_facts)
                    if observation_handle is not None:
                        append_jsonl(observation_handle, records.observations)
                    if attribute_handle is not None:
                        append_jsonl(attribute_handle, records.attribute_facts)
                    if relation_handle is not None:
                        append_jsonl(relation_handle, records.relation_facts)
                    if spatial_handle is not None:
                        append_jsonl(spatial_handle, records.spatial_facts)
                    seq_stats['skipped_invalid_boxes'] += records.skipped_invalid_boxes
                    seq_stats['missing_images'] += sum((1 for item in records.observations if not item.get('image_exists')))
                    seq_stats['missing_frame_time_rows'] += sum((1 for item in records.observations if not item.get('has_frame_time_row')))
                stats.frames += seq_stats['frames']
                stats.observations += seq_stats['observations']
                stats.attribute_facts += seq_stats['attribute_facts']
                stats.relation_facts += seq_stats['relation_facts']
                stats.spatial_facts += seq_stats['spatial_facts']
                stats.missing_images += seq_stats['missing_images']
                stats.missing_frame_time_rows += seq_stats['missing_frame_time_rows']
                stats.skipped_invalid_boxes += seq_stats['skipped_invalid_boxes']
                stats.images += int(manifest_entry['jpg_count'] or 0)
                stats.frame_jsons += int(manifest_entry['frame_json_count'] or 0)
                stats.sequence_stats.append(seq_stats)
                sqlite_writer.commit()
        sqlite_writer.insert_metadata('build_stats', stats.as_dict())
        sqlite_writer.create_indexes()
        sqlite_writer.commit()
    finally:
        sqlite_writer.close()
    manifest = {'contract': 'navqa_evidence_layer_index_manifest_v1', 'description': 'Standalone evidence-layer resource index over existing HSGG frame JSON, jpg files, bbox/object observations, relation facts, bbox geometry facts, frame_time_index, and CG graph references. It does not modify CG, IR, or QA outputs.', 'sequence_manifest': repo_relative_or_absolute(resolve_repo_input_path(config.sequence_manifest)), 'cg_manifests': [repo_relative_or_absolute(resolve_repo_input_path(path)) for path in config.cg_manifests], 'output_dir': repo_relative_or_absolute(output_dir), 'max_frames': config.max_frames, 'max_neighbors_per_object': config.max_neighbors_per_object, 'write_jsonl_sidecars': config.write_jsonl_sidecars, 'write_spatial_jsonl': config.write_spatial_jsonl, 'materialize_spatial_table': config.materialize_spatial_table, 'primary_query_store': repo_relative_or_absolute(paths['sqlite']), 'files': {name: repo_relative_or_absolute(path) for name, path in paths.items()}, 'sequences': manifest_entries, 'stats': stats.as_dict()}
    write_json(paths['manifest'], manifest)

    def build_file_summaries() -> dict[str, dict[str, Any]]:
        return {name: {'path': repo_relative_or_absolute(path), 'exists': path.exists(), 'bytes': path.stat().st_size if path.exists() else None} for name, path in paths.items()}
    summary = {'contract': 'navqa_evidence_layer_index_summary_v1', 'output_dir': repo_relative_or_absolute(output_dir), 'stats': stats.as_dict(), 'files': build_file_summaries()}
    write_json(paths['summary'], summary)
    paths['summary_md'].write_text(render_summary_markdown(summary), encoding='utf-8')
    summary['files'] = build_file_summaries()
    write_json(paths['summary'], summary)
    paths['summary_md'].write_text(render_summary_markdown(summary), encoding='utf-8')
    return summary

def render_summary_markdown(summary: dict[str, Any]) -> str:
    stats = summary['stats']
    lines = ['# NAVQA Evidence Layer Index Summary', '', 'This is a standalone resource/evidence index. It does not modify CG, IR, or QA outputs.', '', '## Overall', '', f"- Sequences: {stats['sequences']}", f"- Frames scanned: {stats['frames']}", f"- HSGG frame json files available: {stats['frame_jsons']}", f"- JPG images available: {stats['images']}", f"- Observations: {stats['observations']}", f"- Attribute facts: {stats['attribute_facts']}", f"- HSGG relation facts: {stats['relation_facts']}", f"- BBox spatial facts available on demand: {stats['spatial_facts']}", f"- Missing image references: {stats['missing_images']}", f"- Missing frame_time_index rows: {stats['missing_frame_time_rows']}", f"- Skipped invalid boxes: {stats['skipped_invalid_boxes']}", '', '## By Sequence', '', '| seq | frames | observations | attributes | relations | spatial on demand | missing image | missing time | invalid boxes |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in stats['sequence_stats']:
        lines.append('| {seq_id} | {frames} | {observations} | {attribute_facts} | {relation_facts} | {spatial_facts} | {missing_images} | {missing_frame_time_rows} | {skipped_invalid_boxes} |'.format(**item))
    lines.append('')
    lines.append('## Files')
    lines.append('')
    for name, file_info in summary['files'].items():
        size_text = f"{file_info['bytes']} bytes" if file_info.get('exists') else 'not written'
        lines.append(f"- {name}: `{file_info['path']}` ({size_text})")
    lines.append('')
    return '\n'.join(lines)
