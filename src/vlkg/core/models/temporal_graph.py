from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def diagonal(self) -> float:
        return (self.width**2 + self.height**2) ** 0.5

    def iou(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def intersection_area(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        return inter_w * inter_h

    def inside_ratio(self, other: "BoundingBox") -> float:
        if self.area <= 0.0:
            return 0.0
        return self.intersection_area(other) / self.area

    def area_ratio(self, other: "BoundingBox") -> float:
        if self.area <= 0.0 or other.area <= 0.0:
            return 0.0
        return min(self.area, other.area) / max(self.area, other.area)

    def center_distance(self, other: "BoundingBox") -> float:
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return (dx**2 + dy**2) ** 0.5

    def contains_center_of(self, other: "BoundingBox") -> bool:
        return (
            self.x1 <= other.center_x <= self.x2
            and self.y1 <= other.center_y <= self.y2
        )


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id: str
    video_id: str
    entity_id: str
    frame_name: str
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    image_path: str
    object_index: int
    bbox: BoundingBox
    label: str
    score: float
    state: str | None = None
    state_values: list[str] = Field(default_factory=list)
    raw_state_values: list[str] = Field(default_factory=list)
    parent_label: str | None = None
    node_role: Literal["static", "dynamic", "dependent", "noise"] = "dynamic"
    scene_id: str | None = None
    is_static: bool = False
    is_noise: bool = False
    graph_origin: Literal["SCENE_BASE", "SCENE_DELTA"] = "SCENE_DELTA"
    source: Literal["vlm_json"] = "vlm_json"


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    video_id: str
    category: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_ts: float
    last_seen_ts: float
    observation_count: int
    canonical_image_path: str | None = None
    canonical_embedding: list[float] | None = None
    labels_seen: list[str] = Field(default_factory=list)
    raw_labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)
    raw_states_seen: list[str] = Field(default_factory=list)
    parent_labels_seen: list[str] = Field(default_factory=list)
    node_role: Literal["static", "dynamic", "dependent", "noise"] = "dynamic"
    is_static: bool = False
    is_noise: bool = False
    is_fragment: bool = False
    graph_origin: Literal["SCENE_BASE", "SCENE_DELTA", "HYBRID"] = "SCENE_DELTA"


class RelationEpisode(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation_episode_id: str
    video_id: str
    subject_entity_id: str
    predicate: str
    raw_predicate: str | None = None
    object_entity_id: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int


class StateEpisode(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_episode_id: str
    video_id: str
    entity_id: str
    attr_key: str = "state"
    attr_value: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int


class SceneSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene_id: str
    video_id: str
    scene_index: int = 1
    scene_count: int = 1
    start_frame_rank: int
    end_frame_rank: int
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    previous_scene_id: str | None = None
    next_scene_id: str | None = None
    boundary_from_previous_frames: int | None = None
    boundary_to_next_frames: int | None = None
    boundary_from_previous_time_gap_sec: float | None = None
    boundary_to_next_time_gap_sec: float | None = None
    boundary_from_previous_entity_similarity: float | None = None
    boundary_to_next_entity_similarity: float | None = None
    boundary_from_previous_label_similarity: float | None = None
    boundary_to_next_label_similarity: float | None = None
    boundary_from_previous_reason: str | None = None
    boundary_to_next_reason: str | None = None
    entry_static_labels: list[str] = Field(default_factory=list)
    exit_static_labels: list[str] = Field(default_factory=list)
    representative_static_entity_ids: list[str] = Field(default_factory=list)
    representative_static_labels: list[str] = Field(default_factory=list)
    representative_dynamic_labels: list[str] = Field(default_factory=list)
    scene_kind_hint: str | None = None


class EntitySegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    video_id: str
    entity_id: str
    scene_id: str | None = None
    segment_type: Literal["ENTITY_LIFECYCLE"] = "ENTITY_LIFECYCLE"
    segment_index: int = 1
    segment_count: int = 1
    start_frame_rank: int
    end_frame_rank: int
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    previous_segment_id: str | None = None
    next_segment_id: str | None = None
    gap_from_previous_frames: int | None = None
    gap_to_next_frames: int | None = None
    gap_from_previous_ranks: int | None = None
    gap_to_next_ranks: int | None = None
    resumes_after_gap: bool = False
    is_static: bool = False
    is_noise: bool = False


class SceneBaseNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    scene_id: str
    video_id: str
    entity_id: str
    label: str
    first_frame: int
    last_frame: int
    support_frames: int
    coverage_ratio: float
    canonical_image_path: str | None = None
    parent_labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)


class SceneBaseEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    scene_id: str
    video_id: str
    predicate: Literal["SCENE_CONTAINS"]
    source_id: str
    target_id: str
    entity_id: str
    support_frames: int
    coverage_ratio: float


class SceneDeltaNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    scene_id: str
    video_id: str
    entity_id: str
    label: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    support_frames: int
    transition_event_count: int
    relation_episode_count: int
    state_episode_count: int
    graph_origin: Literal["SCENE_DELTA"] = "SCENE_DELTA"


class SceneDeltaObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta_observation_id: str
    scene_id: str
    video_id: str
    entity_id: str
    scene_delta_node_id: str
    source_segment_id: str
    source_observation_id: str
    frame_name: str
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    image_path: str
    bbox: BoundingBox
    label: str
    trigger_event_ids: list[str] = Field(default_factory=list)
    trigger_event_types: list[str] = Field(default_factory=list)
    marks_segment_start: bool = False
    marks_segment_end: bool = False
    graph_origin: Literal["SCENE_DELTA"] = "SCENE_DELTA"


class SceneDeltaEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    scene_id: str
    video_id: str
    predicate: Literal["SCENE_UPDATES"]
    source_id: str
    target_id: str
    entity_id: str
    support_frames: int
    transition_event_count: int


class TransitionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    video_id: str
    event_type: Literal[
        "APPEAR",
        "DISAPPEAR",
        "STATE_TRANSITION",
        "RELATION_START",
        "RELATION_END",
        "RELATION_OBSERVED",
    ]
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    entity_id: str | None = None
    subject_entity_id: str | None = None
    predicate: str | None = None
    raw_predicate: str | None = None
    object_entity_id: str | None = None
    attr_key: str | None = None
    previous_state_value: str | None = None
    current_state_value: str | None = None
    source_segment_id: str | None = None
    source_episode_id: str | None = None


class TemporalGraphDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    source_dir: Path
    frame_count: int
    observations: list[Observation]
    entities: list[Entity]
    relation_episodes: list[RelationEpisode]
    state_episodes: list[StateEpisode]
    scene_segments: list[SceneSegment] = Field(default_factory=list)
    entity_segments: list[EntitySegment] = Field(default_factory=list)
    scene_base_nodes: list[SceneBaseNode] = Field(default_factory=list)
    scene_base_edges: list[SceneBaseEdge] = Field(default_factory=list)
    scene_delta_nodes: list[SceneDeltaNode] = Field(default_factory=list)
    scene_delta_observations: list[SceneDeltaObservation] = Field(default_factory=list)
    scene_delta_edges: list[SceneDeltaEdge] = Field(default_factory=list)
    transition_events: list[TransitionEvent] = Field(default_factory=list)

    def filtered_copy(
        self,
        *,
        exclude_noise: bool = False,
        exclude_fragments: bool = False,
        keep_fragment_labels: set[str] | None = None,
    ) -> "TemporalGraphDocument":
        keep_fragment_labels = {
            str(label).strip().lower()
            for label in (keep_fragment_labels or set())
            if str(label).strip()
        }
        excluded_entity_ids = {
            entity.entity_id
            for entity in self.entities
            if (exclude_noise and entity.is_noise)
            or (
                exclude_fragments
                and entity.is_fragment
                and entity.category.strip().lower() not in keep_fragment_labels
            )
        }
        entities = [
            entity
            for entity in self.entities
            if entity.entity_id not in excluded_entity_ids
        ]
        observations = [
            observation
            for observation in self.observations
            if observation.entity_id not in excluded_entity_ids
        ]
        state_episodes = [
            episode
            for episode in self.state_episodes
            if episode.entity_id not in excluded_entity_ids
        ]
        relation_episodes = [
            episode
            for episode in self.relation_episodes
            if episode.subject_entity_id not in excluded_entity_ids
            and episode.object_entity_id not in excluded_entity_ids
        ]
        entity_segments = [
            segment
            for segment in self.entity_segments
            if segment.entity_id not in excluded_entity_ids
        ]
        scene_base_nodes = [
            node
            for node in self.scene_base_nodes
            if node.entity_id not in excluded_entity_ids
        ]
        allowed_scene_base_node_ids = {
            node.node_id
            for node in scene_base_nodes
        }
        scene_base_edges = [
            edge
            for edge in self.scene_base_edges
            if edge.entity_id not in excluded_entity_ids
            and edge.target_id in allowed_scene_base_node_ids
        ]
        scene_delta_nodes = [
            node
            for node in self.scene_delta_nodes
            if node.entity_id not in excluded_entity_ids
        ]
        allowed_scene_delta_node_ids = {
            node.node_id
            for node in scene_delta_nodes
        }
        scene_delta_observations = [
            observation
            for observation in self.scene_delta_observations
            if observation.entity_id not in excluded_entity_ids
            and observation.scene_delta_node_id in allowed_scene_delta_node_ids
        ]
        scene_delta_edges = [
            edge
            for edge in self.scene_delta_edges
            if edge.entity_id not in excluded_entity_ids
            and edge.target_id in allowed_scene_delta_node_ids
        ]
        transition_events = [
            event
            for event in self.transition_events
            if (
                event.entity_id is None
                or event.entity_id not in excluded_entity_ids
            )
            and (
                event.subject_entity_id is None
                or event.subject_entity_id not in excluded_entity_ids
            )
            and (
                event.object_entity_id is None
                or event.object_entity_id not in excluded_entity_ids
            )
        ]
        scene_segments = self._refresh_scene_segments(
            scene_segments=self.scene_segments,
            observations=observations,
            entities=entities,
        )
        return self.model_copy(
            update={
                "observations": observations,
                "entities": entities,
                "state_episodes": state_episodes,
                "relation_episodes": relation_episodes,
                "scene_segments": scene_segments,
                "entity_segments": entity_segments,
                "scene_base_nodes": scene_base_nodes,
                "scene_base_edges": scene_base_edges,
                "scene_delta_nodes": scene_delta_nodes,
                "scene_delta_observations": scene_delta_observations,
                "scene_delta_edges": scene_delta_edges,
                "transition_events": transition_events,
            }
        )

    def _refresh_scene_segments(
        self,
        *,
        scene_segments: list[SceneSegment],
        observations: list[Observation],
        entities: list[Entity],
    ) -> list[SceneSegment]:
        if not scene_segments:
            return scene_segments

        entity_by_id = {
            entity.entity_id: entity
            for entity in entities
        }
        refreshed: list[SceneSegment] = []
        for scene in scene_segments:
            static_entity_frame_support: dict[str, set[int]] = defaultdict(set)
            static_label_frame_support: dict[str, set[int]] = defaultdict(set)
            dynamic_label_frame_support: dict[str, set[int]] = defaultdict(set)
            entry_static_labels: set[str] = set()
            exit_static_labels: set[str] = set()

            for observation in observations:
                if not (
                    scene.start_frame <= observation.frame_index <= scene.end_frame
                ):
                    continue
                if observation.is_static:
                    static_entity_frame_support[observation.entity_id].add(
                        observation.frame_rank
                    )
                    static_label_frame_support[observation.label].add(
                        observation.frame_rank
                    )
                    if observation.frame_index == scene.start_frame:
                        entry_static_labels.add(observation.label)
                    if observation.frame_index == scene.end_frame:
                        exit_static_labels.add(observation.label)
                    continue

                entity = entity_by_id.get(observation.entity_id)
                dynamic_label = (
                    entity.category
                    if entity is not None
                    else observation.label
                )
                dynamic_label_frame_support[dynamic_label].add(
                    observation.frame_rank
                )

            representative_static_entity_ids = [
                entity_id
                for entity_id, _ in sorted(
                    static_entity_frame_support.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:6]
            ]
            representative_static_labels = [
                label
                for label, _ in sorted(
                    static_label_frame_support.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:6]
            ]
            representative_dynamic_labels = [
                label
                for label, _ in sorted(
                    dynamic_label_frame_support.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:6]
            ]
            scene_kind_hint_parts = (
                representative_static_labels[:3]
                or representative_dynamic_labels[:3]
            )
            scene_kind_hint = (
                " / ".join(scene_kind_hint_parts)
                if scene_kind_hint_parts
                else None
            )
            refreshed.append(
                scene.model_copy(
                    update={
                        "entry_static_labels": sorted(entry_static_labels),
                        "exit_static_labels": sorted(exit_static_labels),
                        "representative_static_entity_ids": representative_static_entity_ids,
                        "representative_static_labels": representative_static_labels,
                        "representative_dynamic_labels": representative_dynamic_labels,
                        "scene_kind_hint": scene_kind_hint,
                    }
                )
            )
        return refreshed
