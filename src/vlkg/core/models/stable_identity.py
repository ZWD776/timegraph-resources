from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


StableIdentityType = Literal[
    "human",
    "ball",
    "static_object",
    "object",
]


class StableIdentityTimeRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_frame: int | None = None
    end_frame: int | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    support_frames: int | None = None


class StableIdentityMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    category: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_ts: float
    last_seen_ts: float
    observation_count: int
    labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)
    parent_labels_seen: list[str] = Field(default_factory=list)


class StableIdentityNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    stable_id: str
    video_id: str
    identity_type: StableIdentityType
    canonical_label: str
    display_name: str
    labels_seen: list[str] = Field(default_factory=list)
    signature_tokens: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    member_entities: list[StableIdentityMember] = Field(default_factory=list)
    time_range: StableIdentityTimeRange
    observation_count: int
    attributes: dict[str, object] = Field(default_factory=dict)


class StableIdentityDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "stable_identity_graph.v1"
    video_id: str
    source_graph_path: Path | None = None
    identities: list[StableIdentityNode] = Field(default_factory=list)

    @property
    def identity_count(self) -> int:
        return len(self.identities)
