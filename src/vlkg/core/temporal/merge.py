from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlkg.core.temporal.association import AssociationCandidate
from vlkg.core.models.temporal_graph import BoundingBox


@dataclass
class EntityStateRecord:
    entity_id: str
    label: str
    last_frame_rank: int
    last_frame_index: int
    last_track_id: str | None
    last_object_index: int
    last_bbox: BoundingBox
    parent_label: str | None
    last_state_values: list[str]
    is_static: bool
    is_noise: bool
    canonical_image_path: str | None
    canonical_embedding: np.ndarray | None
    observation_count: int


def merge_l2_normalized_embeddings(
    *,
    current: np.ndarray | None,
    new_embedding: np.ndarray,
    current_count: int,
) -> np.ndarray:
    if current is None:
        return new_embedding.copy()
    merged = (current * current_count + new_embedding) / (current_count + 1)
    norm = np.linalg.norm(merged)
    if norm <= 0.0:
        return new_embedding.copy()
    return (merged / norm).astype(np.float32)


def create_entity_state_record(
    *,
    entity_id: str,
    label: str,
    frame_rank: int,
    frame_index: int,
    track_id: str | None,
    object_index: int,
    bbox: BoundingBox,
    parent_label: str | None,
    state_values: list[str],
    is_static: bool,
    is_noise: bool,
    image_path: str,
    embedding: np.ndarray,
) -> EntityStateRecord:
    return EntityStateRecord(
        entity_id=entity_id,
        label=label,
        last_frame_rank=frame_rank,
        last_frame_index=frame_index,
        last_track_id=track_id,
        last_object_index=object_index,
        last_bbox=bbox,
        parent_label=parent_label,
        last_state_values=list(state_values),
        is_static=is_static,
        is_noise=is_noise,
        canonical_image_path=image_path,
        canonical_embedding=embedding.copy(),
        observation_count=1,
    )


def update_entity_state_record(
    *,
    entity_state: EntityStateRecord,
    frame_rank: int,
    frame_index: int,
    track_id: str | None,
    object_index: int,
    bbox: BoundingBox,
    parent_label: str | None,
    state_values: list[str],
    is_static: bool,
    is_noise: bool,
    image_path: str,
    embedding: np.ndarray,
) -> None:
    entity_state.canonical_embedding = merge_l2_normalized_embeddings(
        current=entity_state.canonical_embedding,
        new_embedding=embedding,
        current_count=entity_state.observation_count,
    )
    entity_state.canonical_image_path = image_path
    entity_state.last_frame_rank = frame_rank
    entity_state.last_frame_index = frame_index
    entity_state.last_track_id = track_id
    entity_state.last_object_index = object_index
    entity_state.last_bbox = bbox
    entity_state.parent_label = parent_label
    entity_state.last_state_values = list(state_values)
    entity_state.is_static = entity_state.is_static or is_static
    entity_state.is_noise = entity_state.is_noise or is_noise
    entity_state.observation_count += 1


def build_association_candidates(
    entity_states: list[EntityStateRecord],
) -> list[AssociationCandidate]:
    return [
        AssociationCandidate(
            entity_id=entity_state.entity_id,
            label=entity_state.label,
            last_frame_rank=entity_state.last_frame_rank,
            last_frame_index=entity_state.last_frame_index,
            track_id=entity_state.last_track_id,
            object_index=entity_state.last_object_index,
            last_bbox=entity_state.last_bbox,
            parent_label=entity_state.parent_label,
            last_state_values=entity_state.last_state_values,
            is_static=entity_state.is_static,
            is_noise=entity_state.is_noise,
            observation_count=entity_state.observation_count,
            canonical_image_path=entity_state.canonical_image_path,
            embedding=entity_state.canonical_embedding,
        )
        for entity_state in entity_states
    ]
