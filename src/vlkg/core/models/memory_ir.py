from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SceneAnchor(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    label: str
    first_frame: int
    last_frame: int
    support_frames: int
    coverage_ratio: float
    canonical_image_path: str | None = None
    parent_labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)


class SceneAnchorSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    video_id: str
    scene_id: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    representative_static_labels: list[str] = Field(default_factory=list)
    representative_dynamic_labels: list[str] = Field(default_factory=list)
    scene_kind_hint: str | None = None
    anchors: list[SceneAnchor] = Field(default_factory=list)


class EntityIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_entity_id: str
    video_id: str
    entity_id: str
    label: str
    node_role: Literal["dynamic"] = "dynamic"
    first_seen_frame: int
    last_seen_frame: int
    first_seen_ts: float
    last_seen_ts: float
    observation_count: int
    labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)
    parent_labels_seen: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    canonical_image_path: str | None = None
    is_fragment: bool = False


class EntityStateUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    update_id: str
    video_id: str
    entity_id: str
    label: str
    scene_id: str | None = None
    attr_key: str = "state"
    attr_value: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    source_state_episode_id: str


class InteractionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    video_id: str
    scene_id: str | None = None
    subject_entity_id: str
    subject_label: str
    subject_role: Literal["dynamic"] = "dynamic"
    predicate: str
    raw_predicate: str | None = None
    object_entity_id: str
    object_label: str
    object_role: Literal["static", "dynamic"] = "dynamic"
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    source_relation_episode_id: str


class MemoryIRDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "memory_ir.v1"
    video_id: str
    frame_count: int
    source_graph_path: Path | None = None
    scene_anchor_snapshots: list[SceneAnchorSnapshot] = Field(default_factory=list)
    entity_identities: list[EntityIdentity] = Field(default_factory=list)
    entity_state_updates: list[EntityStateUpdate] = Field(default_factory=list)
    interaction_events: list[InteractionEvent] = Field(default_factory=list)
