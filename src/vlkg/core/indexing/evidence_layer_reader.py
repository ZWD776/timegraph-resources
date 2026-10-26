"""Read an explicitly supplied evidence-layer SQLite schema without dataset defaults."""
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

class EvidenceLayerIndex:
    """Query helper for the standalone evidence-layer SQLite sidecar."""

    def __init__(self, path: Path | str) -> None:
        self.path = resolve_repo_input_path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'Evidence layer SQLite not found: {self.path}')
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def count(self, table: str) -> int:
        if table not in {'sequences', 'frames', 'observations', 'attribute_facts', 'relation_facts', 'spatial_facts', 'cg_graph_refs'}:
            raise ValueError(f'Unsupported table: {table}')
        return int(self.connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])

    def observation(self, observation_id: str) -> dict[str, Any] | None:
        row = self.connection.execute('SELECT raw_json FROM observations WHERE observation_id = ?', (observation_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row['raw_json'])

    def frame_observations(self, *, seq_id: str, frame_key: str | None=None, graph_frame_index: int | None=None) -> list[dict[str, Any]]:
        if frame_key is not None:
            rows = self.connection.execute('\n                SELECT raw_json FROM observations\n                WHERE seq_id = ? AND frame_key = ?\n                ORDER BY object_index\n                ', (str(seq_id), str(frame_key))).fetchall()
        elif graph_frame_index is not None:
            rows = self.connection.execute('\n                SELECT raw_json FROM observations\n                WHERE seq_id = ? AND graph_frame_index = ?\n                ORDER BY object_index\n                ', (str(seq_id), int(graph_frame_index))).fetchall()
        else:
            raise ValueError('Either frame_key or graph_frame_index is required.')
        return [json.loads(row['raw_json']) for row in rows]

    def observation_attributes(self, observation_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute('\n            SELECT raw_json FROM attribute_facts\n            WHERE observation_id = ?\n            ORDER BY fact_id\n            ', (observation_id,)).fetchall()
        return [json.loads(row['raw_json']) for row in rows]

    def observation_relations(self, observation_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute('\n            SELECT raw_json FROM relation_facts\n            WHERE subject_observation_id = ? OR object_observation_id = ?\n            ORDER BY fact_id\n            ', (observation_id, observation_id)).fetchall()
        return [json.loads(row['raw_json']) for row in rows]

    def nearest_bbox_neighbors(self, observation_id: str, *, limit: int=8) -> list[dict[str, Any]]:
        observation = self.observation(observation_id)
        if observation is None:
            return []
        same_frame = self.frame_observations(seq_id=str(observation['seq_id']), frame_key=str(observation['frame_key']))
        subject_box = bounding_box(observation.get('bbox') or [])
        if subject_box is None:
            return []
        rows: list[tuple[float, dict[str, Any]]] = []
        for candidate in same_frame:
            if candidate['observation_id'] == observation_id:
                continue
            candidate_box = bounding_box(candidate.get('bbox') or [])
            if candidate_box is None:
                continue
            rows.append((subject_box.center_distance(candidate_box), build_spatial_fact_for_pair(observation, candidate)))
        rows.sort(key=lambda item: item[0])
        return [row for _distance, row in rows[:max(0, int(limit))]]

    def local_evidence_pack(self, observation_id: str, *, neighbor_limit: int=8) -> dict[str, Any]:
        observation = self.observation(observation_id)
        if observation is None:
            raise KeyError(f'Unknown observation_id: {observation_id}')
        return {'contract': 'navqa_evidence_layer_local_pack_v1', 'observation': observation, 'attributes': self.observation_attributes(observation_id), 'relations': self.observation_relations(observation_id), 'bbox_neighbors': self.nearest_bbox_neighbors(observation_id, limit=neighbor_limit)}

def build_spatial_fact_for_pair(subject: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    subject_box = bounding_box(subject.get('bbox') or [])
    obj_box = bounding_box(obj.get('bbox') or [])
    if subject_box is None or obj_box is None:
        raise ValueError('Both observations must have valid bbox values.')
    return {'contract': 'navqa_evidence_bbox_spatial_fact_v1', 'fact_id': f"{subject['seq_id']}:frame:{subject['frame_key']}:spatial:{subject['object_index']}:{obj['object_index']}", 'seq_id': subject['seq_id'], 'frame_key': subject['frame_key'], 'graph_frame_index': subject.get('graph_frame_index'), 'subject_observation_id': subject['observation_id'], 'subject_object_index': subject['object_index'], 'subject_label': subject['label'], 'object_observation_id': obj['observation_id'], 'object_object_index': obj['object_index'], 'object_label': obj['label'], 'relations': spatial_relation_candidates(subject_box, obj_box), 'geometry': bbox_geometry(subject_box, obj_box)}

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

def bounding_box(raw_box: Iterable[Any]) -> BoundingBox | None:
    values = bbox_values(raw_box)
    if values is None:
        return None
    return BoundingBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3])

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
