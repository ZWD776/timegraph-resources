from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vlkg.core.models.temporal_graph import (
    Entity,
    EntitySegment,
    RelationEpisode,
    SceneBaseEdge,
    SceneBaseNode,
    SceneDeltaEdge,
    SceneDeltaNode,
    SceneDeltaObservation,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)


@dataclass(frozen=True)
class EntityTemporalSnapshot:
    entity: Entity
    frame_index: int
    frame_rank: int | None
    scene_segment: SceneSegment | None
    entity_segments: tuple[EntitySegment, ...]
    active_states: tuple[StateEpisode, ...]
    active_relations: tuple[RelationEpisode, ...]
    transition_events: tuple[TransitionEvent, ...]


@dataclass(frozen=True)
class SceneDeltaSummary:
    scene_id: str
    video_id: str
    scene_index: int
    scene_count: int
    frame_index: int | None
    frame_rank: int | None
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    previous_scene_id: str | None
    next_scene_id: str | None
    boundary_from_previous_frames: int | None
    boundary_to_next_frames: int | None
    boundary_from_previous_time_gap_sec: float | None
    boundary_to_next_time_gap_sec: float | None
    boundary_from_previous_entity_similarity: float | None
    boundary_to_next_entity_similarity: float | None
    boundary_from_previous_label_similarity: float | None
    boundary_to_next_label_similarity: float | None
    boundary_from_previous_reason: str | None
    boundary_to_next_reason: str | None
    entry_static_labels: tuple[str, ...]
    exit_static_labels: tuple[str, ...]
    scene_base_node_count: int
    scene_base_edge_count: int
    scene_delta_node_count: int
    scene_delta_edge_count: int
    scene_delta_observation_count: int
    active_delta_node_count: int
    active_delta_edge_count: int
    active_delta_observation_count: int
    active_delta_entity_ids: tuple[str, ...]
    relation_episode_count: int
    state_episode_count: int
    transition_event_count: int
    transition_event_type_breakdown: dict[str, int]
    representative_static_entity_ids: tuple[str, ...]
    representative_static_labels: tuple[str, ...]
    representative_dynamic_labels: tuple[str, ...]
    active_delta_labels: tuple[str, ...]
    key_transition_frames: tuple[int, ...]
    scene_kind_hint: str | None


class TemporalIndex:
    def __init__(self, document: TemporalGraphDocument) -> None:
        self.document = document
        self.entities_by_id = {
            entity.entity_id: entity
            for entity in document.entities
        }
        self.frame_index_to_rank = self._build_frame_index_to_rank(document)
        self.scene_segments = sorted(
            document.scene_segments,
            key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
        )
        self.entity_segments_by_entity: dict[str, list[EntitySegment]] = defaultdict(list)
        self.scene_base_nodes_by_scene: dict[str, list[SceneBaseNode]] = defaultdict(list)
        self.scene_base_nodes_by_entity: dict[str, list[SceneBaseNode]] = defaultdict(list)
        self.scene_base_edges_by_scene: dict[str, list[SceneBaseEdge]] = defaultdict(list)
        self.scene_delta_nodes_by_scene: dict[str, list[SceneDeltaNode]] = defaultdict(list)
        self.scene_delta_nodes_by_entity: dict[str, list[SceneDeltaNode]] = defaultdict(list)
        self.scene_delta_observations_by_scene: dict[str, list[SceneDeltaObservation]] = defaultdict(list)
        self.scene_delta_observations_by_entity: dict[str, list[SceneDeltaObservation]] = defaultdict(list)
        self.scene_delta_observations_by_frame: dict[int, list[SceneDeltaObservation]] = defaultdict(list)
        self.scene_delta_edges_by_scene: dict[str, list[SceneDeltaEdge]] = defaultdict(list)
        self.state_episodes_by_entity: dict[str, list[StateEpisode]] = defaultdict(list)
        self.relation_episodes_by_entity: dict[str, list[RelationEpisode]] = defaultdict(list)
        self.transition_events_by_frame_index: dict[int, list[TransitionEvent]] = (
            defaultdict(list)
        )
        self.transition_events_by_frame_rank: dict[int, list[TransitionEvent]] = (
            defaultdict(list)
        )
        self.transition_events_by_entity: dict[str, list[TransitionEvent]] = (
            defaultdict(list)
        )

        for segment in document.entity_segments:
            self.entity_segments_by_entity[segment.entity_id].append(segment)
        for entity_id, segments in self.entity_segments_by_entity.items():
            segments.sort(key=lambda item: (item.start_frame, item.end_frame, item.segment_id))

        for node in document.scene_base_nodes:
            self.scene_base_nodes_by_scene[node.scene_id].append(node)
            self.scene_base_nodes_by_entity[node.entity_id].append(node)
        for scene_id, nodes in self.scene_base_nodes_by_scene.items():
            nodes.sort(key=lambda item: (item.label, item.entity_id, item.node_id))
        for entity_id, nodes in self.scene_base_nodes_by_entity.items():
            nodes.sort(key=lambda item: (item.scene_id, item.node_id))

        for edge in document.scene_base_edges:
            self.scene_base_edges_by_scene[edge.scene_id].append(edge)
        for scene_id, edges in self.scene_base_edges_by_scene.items():
            edges.sort(key=lambda item: (item.target_id, item.edge_id))

        for node in document.scene_delta_nodes:
            self.scene_delta_nodes_by_scene[node.scene_id].append(node)
            self.scene_delta_nodes_by_entity[node.entity_id].append(node)
        for scene_id, nodes in self.scene_delta_nodes_by_scene.items():
            nodes.sort(key=lambda item: (item.label, item.entity_id, item.node_id))
        for entity_id, nodes in self.scene_delta_nodes_by_entity.items():
            nodes.sort(key=lambda item: (item.scene_id, item.node_id))

        for observation in document.scene_delta_observations:
            self.scene_delta_observations_by_scene[observation.scene_id].append(observation)
            self.scene_delta_observations_by_entity[observation.entity_id].append(observation)
            self.scene_delta_observations_by_frame[observation.frame_index].append(observation)
        for scene_id, observations in self.scene_delta_observations_by_scene.items():
            observations.sort(
                key=lambda item: (item.frame_index, item.entity_id, item.delta_observation_id)
            )
        for entity_id, observations in self.scene_delta_observations_by_entity.items():
            observations.sort(
                key=lambda item: (item.frame_index, item.scene_id, item.delta_observation_id)
            )
        for frame_index, observations in self.scene_delta_observations_by_frame.items():
            observations.sort(
                key=lambda item: (item.scene_id, item.entity_id, item.delta_observation_id)
            )

        for edge in document.scene_delta_edges:
            self.scene_delta_edges_by_scene[edge.scene_id].append(edge)
        for scene_id, edges in self.scene_delta_edges_by_scene.items():
            edges.sort(key=lambda item: (item.target_id, item.edge_id))

        for episode in document.state_episodes:
            self.state_episodes_by_entity[episode.entity_id].append(episode)
        for entity_id, episodes in self.state_episodes_by_entity.items():
            episodes.sort(
                key=lambda item: (item.start_frame, item.end_frame, item.attr_value)
            )

        for episode in document.relation_episodes:
            self.relation_episodes_by_entity[episode.subject_entity_id].append(episode)
            if episode.object_entity_id != episode.subject_entity_id:
                self.relation_episodes_by_entity[episode.object_entity_id].append(episode)
        for entity_id, episodes in self.relation_episodes_by_entity.items():
            episodes.sort(
                key=lambda item: (
                    item.start_frame,
                    item.end_frame,
                    item.predicate,
                    item.relation_episode_id,
                )
            )

        for event in document.transition_events:
            self.transition_events_by_frame_index[event.frame_index].append(event)
            self.transition_events_by_frame_rank[event.frame_rank].append(event)
            for entity_id in self._event_entity_ids(event):
                self.transition_events_by_entity[entity_id].append(event)

        for frame_index, events in self.transition_events_by_frame_index.items():
            events.sort(key=lambda item: (item.timestamp_sec, item.event_type, item.event_id))
        for frame_rank, events in self.transition_events_by_frame_rank.items():
            events.sort(key=lambda item: (item.timestamp_sec, item.event_type, item.event_id))
        for entity_id, events in self.transition_events_by_entity.items():
            deduped = {event.event_id: event for event in events}
            self.transition_events_by_entity[entity_id] = sorted(
                deduped.values(),
                key=lambda item: (
                    item.frame_rank,
                    item.frame_index,
                    item.timestamp_sec,
                    item.event_type,
                    item.event_id,
                ),
            )

    @classmethod
    def build(cls, document: TemporalGraphDocument) -> "TemporalIndex":
        return cls(document)

    def scene_at_frame(self, frame_index: int) -> SceneSegment | None:
        for segment in self.scene_segments:
            if self._covers_frame(segment.start_frame, segment.end_frame, frame_index):
                return segment
        return None

    def scene_by_id(self, scene_id: str) -> SceneSegment | None:
        for segment in self.scene_segments:
            if segment.scene_id == scene_id:
                return segment
        return None

    def frame_rank_for_index(self, frame_index: int) -> int | None:
        return self.frame_index_to_rank.get(frame_index)

    def scene_base_nodes_for_scene(self, scene_id: str) -> list[SceneBaseNode]:
        return list(self.scene_base_nodes_by_scene.get(scene_id, []))

    def scene_base_edges_for_scene(self, scene_id: str) -> list[SceneBaseEdge]:
        return list(self.scene_base_edges_by_scene.get(scene_id, []))

    def scene_base_nodes_at_frame(self, frame_index: int) -> list[SceneBaseNode]:
        scene = self.scene_at_frame(frame_index)
        if scene is None:
            return []
        return self.scene_base_nodes_for_scene(scene.scene_id)

    def scene_delta_nodes_for_scene(self, scene_id: str) -> list[SceneDeltaNode]:
        return list(self.scene_delta_nodes_by_scene.get(scene_id, []))

    def scene_delta_edges_for_scene(self, scene_id: str) -> list[SceneDeltaEdge]:
        return list(self.scene_delta_edges_by_scene.get(scene_id, []))

    def scene_delta_observations_for_scene(self, scene_id: str) -> list[SceneDeltaObservation]:
        return list(self.scene_delta_observations_by_scene.get(scene_id, []))

    def scene_delta_observations_at_frame(self, frame_index: int) -> list[SceneDeltaObservation]:
        scene = self.scene_at_frame(frame_index)
        if scene is None:
            return []
        return [
            observation
            for observation in self.scene_delta_observations_by_frame.get(frame_index, [])
            if observation.scene_id == scene.scene_id
        ]

    def scene_delta_nodes_at_frame(self, frame_index: int) -> list[SceneDeltaNode]:
        scene = self.scene_at_frame(frame_index)
        if scene is None:
            return []
        return [
            node
            for node in self.scene_delta_nodes_for_scene(scene.scene_id)
            if self._covers_frame(node.start_frame, node.end_frame, frame_index)
        ]

    def scene_delta_summary(self, scene_id: str) -> SceneDeltaSummary:
        scene = self.scene_by_id(scene_id)
        if scene is None:
            raise KeyError(f"Unknown scene_id: {scene_id}")
        return self._build_scene_delta_summary(scene=scene, frame_index=None)

    def scene_delta_summary_at_frame(self, frame_index: int) -> SceneDeltaSummary | None:
        scene = self.scene_at_frame(frame_index)
        if scene is None:
            return None
        return self._build_scene_delta_summary(scene=scene, frame_index=frame_index)

    def entity_segments_for_entity(self, entity_id: str) -> list[EntitySegment]:
        return list(self.entity_segments_by_entity.get(entity_id, []))

    def entity_is_active(self, entity_id: str, frame_index: int) -> bool:
        return any(
            self._covers_frame(segment.start_frame, segment.end_frame, frame_index)
            for segment in self.entity_segments_by_entity.get(entity_id, [])
        )

    def active_entities(self, frame_index: int) -> list[Entity]:
        active: list[Entity] = []
        for entity_id, entity in self.entities_by_id.items():
            if self.entity_is_active(entity_id, frame_index):
                active.append(entity)
        active.sort(key=lambda item: item.entity_id)
        return active

    def active_states(
        self,
        frame_index: int,
        *,
        entity_id: str | None = None,
    ) -> list[StateEpisode]:
        entity_ids = [entity_id] if entity_id else list(self.state_episodes_by_entity.keys())
        episodes: list[StateEpisode] = []
        for current_entity_id in entity_ids:
            for episode in self.state_episodes_by_entity.get(current_entity_id, []):
                if self._covers_frame(episode.start_frame, episode.end_frame, frame_index):
                    episodes.append(episode)
        episodes.sort(key=lambda item: (item.entity_id, item.start_frame, item.attr_value))
        return episodes

    def active_relations(
        self,
        frame_index: int,
        *,
        entity_id: str | None = None,
        predicate: str | None = None,
    ) -> list[RelationEpisode]:
        entity_ids = [entity_id] if entity_id else list(self.relation_episodes_by_entity.keys())
        deduped: dict[str, RelationEpisode] = {}
        for current_entity_id in entity_ids:
            for episode in self.relation_episodes_by_entity.get(current_entity_id, []):
                if predicate and episode.predicate != predicate:
                    continue
                if self._covers_frame(episode.start_frame, episode.end_frame, frame_index):
                    deduped[episode.relation_episode_id] = episode
        episodes = list(deduped.values())
        episodes.sort(
            key=lambda item: (
                item.start_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
            )
        )
        return episodes

    def transition_events_at_frame(
        self,
        *,
        frame_index: int | None = None,
        frame_rank: int | None = None,
        entity_id: str | None = None,
        event_type: str | None = None,
    ) -> list[TransitionEvent]:
        if frame_index is None and frame_rank is None:
            raise ValueError("Either frame_index or frame_rank must be provided.")

        if frame_index is not None:
            events = list(self.transition_events_by_frame_index.get(frame_index, []))
        else:
            events = list(self.transition_events_by_frame_rank.get(frame_rank or 0, []))

        if entity_id is not None:
            events = [
                event
                for event in events
                if entity_id in self._event_entity_ids(event)
            ]
        if event_type is not None:
            normalized_event_type = event_type.strip().upper()
            events = [
                event
                for event in events
                if event.event_type == normalized_event_type
            ]
        return events

    def transition_events_for_entity(
        self,
        entity_id: str,
        *,
        event_type: str | None = None,
    ) -> list[TransitionEvent]:
        events = list(self.transition_events_by_entity.get(entity_id, []))
        if event_type is None:
            return events
        normalized_event_type = event_type.strip().upper()
        return [
            event
            for event in events
            if event.event_type == normalized_event_type
        ]

    def entity_snapshot(
        self,
        entity_id: str,
        *,
        frame_index: int,
    ) -> EntityTemporalSnapshot:
        entity = self.entities_by_id[entity_id]
        return EntityTemporalSnapshot(
            entity=entity,
            frame_index=frame_index,
            frame_rank=self.frame_rank_for_index(frame_index),
            scene_segment=self.scene_at_frame(frame_index),
            entity_segments=tuple(self.entity_segments_for_entity(entity_id)),
            active_states=tuple(self.active_states(frame_index, entity_id=entity_id)),
            active_relations=tuple(self.active_relations(frame_index, entity_id=entity_id)),
            transition_events=tuple(
                self.transition_events_at_frame(
                    frame_index=frame_index,
                    entity_id=entity_id,
                )
            ),
        )

    def _build_frame_index_to_rank(
        self,
        document: TemporalGraphDocument,
    ) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for observation in document.observations:
            mapping.setdefault(observation.frame_index, observation.frame_rank)
        for scene_segment in document.scene_segments:
            mapping.setdefault(scene_segment.start_frame, scene_segment.start_frame_rank)
            mapping.setdefault(scene_segment.end_frame, scene_segment.end_frame_rank)
        return mapping

    def _event_entity_ids(self, event: TransitionEvent) -> set[str]:
        entity_ids = {
            value
            for value in (
                event.entity_id,
                event.subject_entity_id,
                event.object_entity_id,
            )
            if value is not None
        }
        return entity_ids

    def _build_scene_delta_summary(
        self,
        *,
        scene: SceneSegment,
        frame_index: int | None,
    ) -> SceneDeltaSummary:
        scene_base_nodes = self.scene_base_nodes_for_scene(scene.scene_id)
        scene_base_edges = self.scene_base_edges_for_scene(scene.scene_id)
        scene_delta_nodes = self.scene_delta_nodes_for_scene(scene.scene_id)
        scene_delta_observations = self.scene_delta_observations_for_scene(scene.scene_id)
        scene_delta_edges = self.scene_delta_edges_for_scene(scene.scene_id)

        if frame_index is None:
            active_delta_nodes = list(scene_delta_nodes)
            active_delta_observations = list(scene_delta_observations)
        else:
            active_delta_nodes = [
                node
                for node in scene_delta_nodes
                if self._covers_frame(node.start_frame, node.end_frame, frame_index)
            ]
            active_delta_observations = self.scene_delta_observations_at_frame(frame_index)

        active_delta_entity_ids = tuple(
            sorted({node.entity_id for node in active_delta_nodes})
        )
        active_delta_entity_id_set = set(active_delta_entity_ids)

        if frame_index is None:
            relevant_events = [
                event
                for event in self.document.transition_events
                if self._covers_frame(scene.start_frame, scene.end_frame, event.frame_index)
            ]
            relation_episode_count = len(
                {
                    episode.relation_episode_id
                    for episode in self.document.relation_episodes
                    if not (
                        episode.end_frame < scene.start_frame
                        or episode.start_frame > scene.end_frame
                    )
                    and (
                        episode.subject_entity_id in active_delta_entity_id_set
                        or episode.object_entity_id in active_delta_entity_id_set
                    )
                }
            )
            state_episode_count = len(
                {
                    episode.state_episode_id
                    for episode in self.document.state_episodes
                    if episode.entity_id in active_delta_entity_id_set
                    and not (
                        episode.end_frame < scene.start_frame
                        or episode.start_frame > scene.end_frame
                    )
                }
            )
        else:
            relevant_events = self.transition_events_at_frame(frame_index=frame_index)
            if active_delta_entity_id_set:
                relevant_events = [
                    event
                    for event in relevant_events
                    if self._event_entity_ids(event) & active_delta_entity_id_set
                ]
            else:
                relevant_events = []
            relation_episode_count = len(
                {
                    episode.relation_episode_id
                    for episode in self.active_relations(frame_index)
                    if (
                        episode.subject_entity_id in active_delta_entity_id_set
                        or episode.object_entity_id in active_delta_entity_id_set
                    )
                }
            )
            state_episode_count = len(
                {
                    episode.state_episode_id
                    for episode in self.active_states(frame_index)
                    if episode.entity_id in active_delta_entity_id_set
                }
            )

        transition_event_type_breakdown: dict[str, int] = defaultdict(int)
        for event in relevant_events:
            transition_event_type_breakdown[event.event_type] += 1

        active_delta_label_counts: dict[str, int] = defaultdict(int)
        for node in active_delta_nodes:
            active_delta_label_counts[node.label] += 1
        active_delta_labels = tuple(
            label
            for label, _ in sorted(
                active_delta_label_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:6]
        )

        transition_frame_counts: dict[int, int] = defaultdict(int)
        for event in relevant_events:
            transition_frame_counts[event.frame_index] += 1
        key_transition_frames = tuple(
            frame_index
            for frame_index, _ in sorted(
                transition_frame_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:4]
        )

        active_delta_node_ids = {node.node_id for node in active_delta_nodes}
        active_delta_edge_count = sum(
            1
            for edge in scene_delta_edges
            if edge.target_id in active_delta_node_ids
        )

        return SceneDeltaSummary(
            scene_id=scene.scene_id,
            video_id=scene.video_id,
            scene_index=scene.scene_index,
            scene_count=scene.scene_count,
            frame_index=frame_index,
            frame_rank=self.frame_rank_for_index(frame_index) if frame_index is not None else None,
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            start_ts=scene.start_ts,
            end_ts=scene.end_ts,
            previous_scene_id=scene.previous_scene_id,
            next_scene_id=scene.next_scene_id,
            boundary_from_previous_frames=scene.boundary_from_previous_frames,
            boundary_to_next_frames=scene.boundary_to_next_frames,
            boundary_from_previous_time_gap_sec=scene.boundary_from_previous_time_gap_sec,
            boundary_to_next_time_gap_sec=scene.boundary_to_next_time_gap_sec,
            boundary_from_previous_entity_similarity=(
                scene.boundary_from_previous_entity_similarity
            ),
            boundary_to_next_entity_similarity=scene.boundary_to_next_entity_similarity,
            boundary_from_previous_label_similarity=(
                scene.boundary_from_previous_label_similarity
            ),
            boundary_to_next_label_similarity=scene.boundary_to_next_label_similarity,
            boundary_from_previous_reason=scene.boundary_from_previous_reason,
            boundary_to_next_reason=scene.boundary_to_next_reason,
            entry_static_labels=tuple(scene.entry_static_labels),
            exit_static_labels=tuple(scene.exit_static_labels),
            scene_base_node_count=len(scene_base_nodes),
            scene_base_edge_count=len(scene_base_edges),
            scene_delta_node_count=len(scene_delta_nodes),
            scene_delta_edge_count=len(scene_delta_edges),
            scene_delta_observation_count=len(scene_delta_observations),
            active_delta_node_count=len(active_delta_nodes),
            active_delta_edge_count=active_delta_edge_count,
            active_delta_observation_count=len(active_delta_observations),
            active_delta_entity_ids=active_delta_entity_ids,
            relation_episode_count=relation_episode_count,
            state_episode_count=state_episode_count,
            transition_event_count=len(relevant_events),
            transition_event_type_breakdown=dict(sorted(transition_event_type_breakdown.items())),
            representative_static_entity_ids=tuple(scene.representative_static_entity_ids),
            representative_static_labels=tuple(scene.representative_static_labels),
            representative_dynamic_labels=tuple(scene.representative_dynamic_labels),
            active_delta_labels=active_delta_labels,
            key_transition_frames=key_transition_frames,
            scene_kind_hint=scene.scene_kind_hint,
        )

    def _covers_frame(self, start_frame: int, end_frame: int, frame_index: int) -> bool:
        return start_frame <= frame_index <= end_frame
