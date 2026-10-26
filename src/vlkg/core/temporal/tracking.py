from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vlkg.core.models.temporal_graph import BoundingBox


@dataclass(frozen=True)
class TrackingObservationInput:
    video_id: str
    frame_name: str
    frame_rank: int
    frame_index: int
    object_index: int
    label: str
    image_path: str
    bbox: BoundingBox
    is_static: bool
    is_noise: bool


class TrackingProvider(Protocol):
    def assign_track_id(
        self,
        observation: TrackingObservationInput,
    ) -> str | None:
        ...


class NoOpTrackingProvider:
    def assign_track_id(
        self,
        observation: TrackingObservationInput,
    ) -> str | None:
        return None
