from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vlkg.core.models.memory_ir import MemoryIRDocument


class SceneAnchorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    video_id: str
    scene_id: str
    snapshot_id: str
    anchor_entity_id: str
    label: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    coverage_ratio: float
    canonical_image_path: str | None = None


class EntityIdentityRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    video_id: str
    entity_id: str
    label: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_ts: float
    last_seen_ts: float
    observation_count: int
    scene_ids: list[str] = Field(default_factory=list)
    canonical_image_path: str | None = None


class EntityStateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
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


class InteractionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    video_id: str
    scene_id: str | None = None
    subject_entity_id: str
    subject_label: str
    predicate: str
    object_entity_id: str
    object_label: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int


class MemoryQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_text: str | None = None
    answer_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    video_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    start_ts: float | None = None
    end_ts: float | None = None
    limit: int = 10


class MemoryRecordBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    scene_anchors: list[SceneAnchorRecord] = Field(default_factory=list)
    entity_identities: list[EntityIdentityRecord] = Field(default_factory=list)
    entity_states: list[EntityStateRecord] = Field(default_factory=list)
    interactions: list[InteractionRecord] = Field(default_factory=list)

    @classmethod
    def from_memory_ir(cls, document: MemoryIRDocument) -> "MemoryRecordBatch":
        scene_anchors: list[SceneAnchorRecord] = []
        for snapshot in document.scene_anchor_snapshots:
            for anchor in snapshot.anchors:
                scene_anchors.append(
                    SceneAnchorRecord(
                        record_id=f"{snapshot.snapshot_id}:{anchor.entity_id}",
                        video_id=document.video_id,
                        scene_id=snapshot.scene_id,
                        snapshot_id=snapshot.snapshot_id,
                        anchor_entity_id=anchor.entity_id,
                        label=anchor.label,
                        start_frame=snapshot.start_frame,
                        end_frame=snapshot.end_frame,
                        start_ts=snapshot.start_ts,
                        end_ts=snapshot.end_ts,
                        support_frames=anchor.support_frames,
                        coverage_ratio=anchor.coverage_ratio,
                        canonical_image_path=anchor.canonical_image_path,
                    )
                )

        entity_identities = [
            EntityIdentityRecord(
                record_id=item.memory_entity_id,
                video_id=item.video_id,
                entity_id=item.entity_id,
                label=item.label,
                first_seen_frame=item.first_seen_frame,
                last_seen_frame=item.last_seen_frame,
                first_seen_ts=item.first_seen_ts,
                last_seen_ts=item.last_seen_ts,
                observation_count=item.observation_count,
                scene_ids=list(item.scene_ids),
                canonical_image_path=item.canonical_image_path,
            )
            for item in document.entity_identities
        ]

        entity_states = [
            EntityStateRecord(
                record_id=item.update_id,
                video_id=item.video_id,
                entity_id=item.entity_id,
                label=item.label,
                scene_id=item.scene_id,
                attr_key=item.attr_key,
                attr_value=item.attr_value,
                start_frame=item.start_frame,
                end_frame=item.end_frame,
                start_ts=item.start_ts,
                end_ts=item.end_ts,
                support_frames=item.support_frames,
            )
            for item in document.entity_state_updates
        ]

        interactions = [
            InteractionRecord(
                record_id=item.event_id,
                video_id=item.video_id,
                scene_id=item.scene_id,
                subject_entity_id=item.subject_entity_id,
                subject_label=item.subject_label,
                predicate=item.predicate,
                object_entity_id=item.object_entity_id,
                object_label=item.object_label,
                start_frame=item.start_frame,
                end_frame=item.end_frame,
                start_ts=item.start_ts,
                end_ts=item.end_ts,
                support_frames=item.support_frames,
            )
            for item in document.interaction_events
        ]

        return cls(
            video_id=document.video_id,
            scene_anchors=scene_anchors,
            entity_identities=entity_identities,
            entity_states=entity_states,
            interactions=interactions,
        )
