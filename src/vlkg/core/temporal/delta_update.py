from __future__ import annotations

from dataclasses import dataclass

from vlkg.core.models.temporal_graph import Entity, Observation, SceneDeltaEdge, SceneDeltaNode
from vlkg.core.models.temporal_graph import (
    EntitySegment,
    RelationEpisode,
    SceneBaseEdge,
    SceneBaseNode,
    SceneDeltaObservation,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.models.temporal_index import TemporalIndex


@dataclass(frozen=True)
class DeltaUpdatePlan:
    video_id: str
    start_frame: int
    end_frame: int
    touched_scene_ids: tuple[str, ...]
    touched_frame_indices: tuple[int, ...]
    reused_scene_base_node_ids: tuple[str, ...]
    reused_scene_base_edge_ids: tuple[str, ...]
    delta_observation_ids: tuple[str, ...]
    impacted_entity_ids: tuple[str, ...]
    impacted_entity_segment_ids: tuple[str, ...]
    impacted_state_episode_ids: tuple[str, ...]
    impacted_relation_episode_ids: tuple[str, ...]
    transition_event_ids: tuple[str, ...]
    full_scene_observation_count: int
    full_window_observation_count: int
    delta_observation_count: int
    reused_scene_base_node_count: int
    reused_scene_base_edge_count: int
    impacted_entity_count: int
    impacted_entity_segment_count: int
    impacted_state_episode_count: int
    impacted_relation_episode_count: int
    transition_event_count: int
    delta_to_window_observation_ratio: float
    delta_to_scene_observation_ratio: float


@dataclass(frozen=True)
class DeltaWindowSlice:
    plan: DeltaUpdatePlan
    scene_segments: tuple[SceneSegment, ...]
    scene_base_nodes: tuple[SceneBaseNode, ...]
    scene_base_edges: tuple[SceneBaseEdge, ...]
    scene_delta_nodes: tuple[SceneDeltaNode, ...]
    scene_delta_observations: tuple[SceneDeltaObservation, ...]
    scene_delta_edges: tuple[SceneDeltaEdge, ...]
    entities: tuple[Entity, ...]
    entity_segments: tuple[EntitySegment, ...]
    state_episodes: tuple[StateEpisode, ...]
    relation_episodes: tuple[RelationEpisode, ...]
    transition_events: tuple[TransitionEvent, ...]


@dataclass(frozen=True)
class IncrementalWindowDocument:
    plan: DeltaUpdatePlan
    slice: DeltaWindowSlice
    document: TemporalGraphDocument


@dataclass(frozen=True)
class IncrementalExecutionResult:
    plan: DeltaUpdatePlan
    window_document: IncrementalWindowDocument
    updated_document: TemporalGraphDocument
    removed_counts: dict[str, int]
    inserted_counts: dict[str, int]
    document_unchanged: bool


class DeltaUpdatePlanner:
    def __init__(self, document: TemporalGraphDocument) -> None:
        self.document = document
        self.index = TemporalIndex.build(document)
        self.observations_by_scene: dict[str, list] = {}
        for scene in document.scene_segments:
            self.observations_by_scene[scene.scene_id] = []
        for observation in document.observations:
            if observation.scene_id is None:
                continue
            self.observations_by_scene.setdefault(observation.scene_id, []).append(observation)
        for scene_id, observations in self.observations_by_scene.items():
            observations.sort(
                key=lambda item: (item.frame_index, item.frame_rank, item.observation_id)
            )

    @classmethod
    def build(cls, document: TemporalGraphDocument) -> "DeltaUpdatePlanner":
        return cls(document)

    def frame_indices(self) -> list[int]:
        return sorted(self.index.frame_index_to_rank.keys())

    def plan_frame_window(
        self,
        *,
        start_frame: int,
        end_frame: int,
        scoped_entity_ids: tuple[str, ...] | set[str] | None = None,
    ) -> DeltaUpdatePlan:
        if start_frame > end_frame:
            raise ValueError("start_frame must be <= end_frame")

        touched_scenes = [
            scene
            for scene in self.document.scene_segments
            if self._overlaps(
                scene.start_frame,
                scene.end_frame,
                start_frame,
                end_frame,
            )
        ]
        touched_scene_ids = tuple(scene.scene_id for scene in touched_scenes)
        touched_frame_indices = tuple(
            frame_index
            for frame_index in self.frame_indices()
            if start_frame <= frame_index <= end_frame
        )

        delta_observations = self._delta_observations_for_window(
            touched_scenes=touched_scenes,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        transition_events = self._transition_events_for_window(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        window_observation_entity_ids = (
            set(scoped_entity_ids)
            if scoped_entity_ids is not None
            else self._window_observation_entity_ids(
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        if scoped_entity_ids is not None:
            scoped_entity_id_set = set(scoped_entity_ids)
            delta_observations = [
                observation
                for observation in delta_observations
                if observation.entity_id in scoped_entity_id_set
            ]
            transition_events = [
                event
                for event in transition_events
                if self.index._event_entity_ids(event) & scoped_entity_id_set
            ]

        impacted_entity_ids = sorted(
            set(window_observation_entity_ids)
            | {
                observation.entity_id
                for observation in delta_observations
            }
            | {
                entity_id
                for event in transition_events
                for entity_id in self.index._event_entity_ids(event)
            }
        )
        impacted_entity_id_set = set(impacted_entity_ids)
        reused_scene_base_nodes = self._reused_scene_base_nodes(
            touched_scenes=touched_scenes,
            relevant_entity_ids=impacted_entity_id_set,
        )
        reused_scene_base_edges = self._reused_scene_base_edges(
            touched_scenes=touched_scenes,
            relevant_entity_ids=impacted_entity_id_set,
        )

        impacted_entity_segments = self._impacted_entity_segments(
            impacted_entity_ids=impacted_entity_id_set,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        impacted_state_episodes = self._impacted_state_episodes(
            impacted_entity_ids=impacted_entity_id_set,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        impacted_relation_episodes = self._impacted_relation_episodes(
            impacted_entity_ids=impacted_entity_id_set,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        full_scene_observation_count = sum(
            len(self.observations_by_scene.get(scene.scene_id, []))
            for scene in touched_scenes
        )
        full_window_observation_count = sum(
            1
            for observation in self.document.observations
            if start_frame <= observation.frame_index <= end_frame
        )
        delta_observation_count = len(delta_observations)

        return DeltaUpdatePlan(
            video_id=self.document.video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            touched_scene_ids=touched_scene_ids,
            touched_frame_indices=touched_frame_indices,
            reused_scene_base_node_ids=tuple(node.node_id for node in reused_scene_base_nodes),
            reused_scene_base_edge_ids=tuple(edge.edge_id for edge in reused_scene_base_edges),
            delta_observation_ids=tuple(
                observation.delta_observation_id
                for observation in delta_observations
            ),
            impacted_entity_ids=tuple(impacted_entity_ids),
            impacted_entity_segment_ids=tuple(
                segment.segment_id
                for segment in impacted_entity_segments
            ),
            impacted_state_episode_ids=tuple(
                episode.state_episode_id
                for episode in impacted_state_episodes
            ),
            impacted_relation_episode_ids=tuple(
                episode.relation_episode_id
                for episode in impacted_relation_episodes
            ),
            transition_event_ids=tuple(event.event_id for event in transition_events),
            full_scene_observation_count=full_scene_observation_count,
            full_window_observation_count=full_window_observation_count,
            delta_observation_count=delta_observation_count,
            reused_scene_base_node_count=len(reused_scene_base_nodes),
            reused_scene_base_edge_count=len(reused_scene_base_edges),
            impacted_entity_count=len(impacted_entity_ids),
            impacted_entity_segment_count=len(impacted_entity_segments),
            impacted_state_episode_count=len(impacted_state_episodes),
            impacted_relation_episode_count=len(impacted_relation_episodes),
            transition_event_count=len(transition_events),
            delta_to_window_observation_ratio=(
                delta_observation_count / full_window_observation_count
                if full_window_observation_count
                else 0.0
            ),
            delta_to_scene_observation_ratio=(
                delta_observation_count / full_scene_observation_count
                if full_scene_observation_count
                else 0.0
            ),
        )

    def materialize_frame_window(
        self,
        *,
        start_frame: int,
        end_frame: int,
        scoped_entity_ids: tuple[str, ...] | set[str] | None = None,
    ) -> DeltaWindowSlice:
        plan = self.plan_frame_window(
            start_frame=start_frame,
            end_frame=end_frame,
            scoped_entity_ids=scoped_entity_ids,
        )
        impacted_entity_id_set = set(plan.impacted_entity_ids)
        touched_scene_id_set = set(plan.touched_scene_ids)
        delta_observation_id_set = set(plan.delta_observation_ids)
        reused_scene_base_node_id_set = set(plan.reused_scene_base_node_ids)
        reused_scene_base_edge_id_set = set(plan.reused_scene_base_edge_ids)
        impacted_entity_segment_id_set = set(plan.impacted_entity_segment_ids)
        impacted_state_episode_id_set = set(plan.impacted_state_episode_ids)
        impacted_relation_episode_id_set = set(plan.impacted_relation_episode_ids)
        transition_event_id_set = set(plan.transition_event_ids)
        scene_delta_node_id_set = {
            observation.scene_delta_node_id
            for observation in self.document.scene_delta_observations
            if observation.delta_observation_id in delta_observation_id_set
        }

        return DeltaWindowSlice(
            plan=plan,
            scene_segments=tuple(
                scene
                for scene in self.document.scene_segments
                if scene.scene_id in touched_scene_id_set
            ),
            scene_base_nodes=tuple(
                node
                for node in self.document.scene_base_nodes
                if node.node_id in reused_scene_base_node_id_set
            ),
            scene_base_edges=tuple(
                edge
                for edge in self.document.scene_base_edges
                if edge.edge_id in reused_scene_base_edge_id_set
            ),
            scene_delta_nodes=tuple(
                node
                for node in self.document.scene_delta_nodes
                if node.node_id in scene_delta_node_id_set
            ),
            scene_delta_observations=tuple(
                observation
                for observation in self.document.scene_delta_observations
                if observation.delta_observation_id in delta_observation_id_set
            ),
            scene_delta_edges=tuple(
                edge
                for edge in self.document.scene_delta_edges
                if edge.target_id in scene_delta_node_id_set
            ),
            entities=tuple(
                entity
                for entity in self.document.entities
                if entity.entity_id in impacted_entity_id_set
            ),
            entity_segments=tuple(
                segment
                for segment in self.document.entity_segments
                if segment.segment_id in impacted_entity_segment_id_set
            ),
            state_episodes=tuple(
                episode
                for episode in self.document.state_episodes
                if episode.state_episode_id in impacted_state_episode_id_set
            ),
            relation_episodes=tuple(
                episode
                for episode in self.document.relation_episodes
                if episode.relation_episode_id in impacted_relation_episode_id_set
            ),
            transition_events=tuple(
                event
                for event in self.document.transition_events
                if event.event_id in transition_event_id_set
            ),
        )

    def materialize_window_document(
        self,
        *,
        start_frame: int,
        end_frame: int,
        scoped_entity_ids: tuple[str, ...] | set[str] | None = None,
        preserve_scene_segments: bool = False,
    ) -> IncrementalWindowDocument:
        window_slice = self.materialize_frame_window(
            start_frame=start_frame,
            end_frame=end_frame,
            scoped_entity_ids=scoped_entity_ids,
        )
        plan = window_slice.plan
        included_entity_ids = self._window_document_entity_ids(window_slice)
        observations = self._window_document_observations(
            start_frame=start_frame,
            end_frame=end_frame,
            included_entity_ids=included_entity_ids,
        )
        entity_segments = self._window_document_entity_segments(
            start_frame=start_frame,
            end_frame=end_frame,
            included_entity_ids=included_entity_ids,
        )
        state_episodes = self._window_document_state_episodes(
            start_frame=start_frame,
            end_frame=end_frame,
            included_entity_ids=included_entity_ids,
        )
        relation_episodes = self._window_document_relation_episodes(
            start_frame=start_frame,
            end_frame=end_frame,
            included_entity_ids=included_entity_ids,
        )

        window_document = TemporalGraphDocument(
            video_id=self.document.video_id,
            source_dir=self.document.source_dir,
            frame_count=len(plan.touched_frame_indices),
            observations=list(observations),
            entities=[
                entity
                for entity in self.document.entities
                if entity.entity_id in included_entity_ids
            ],
            relation_episodes=list(relation_episodes),
            state_episodes=list(state_episodes),
            scene_segments=(
                []
                if preserve_scene_segments
                else list(window_slice.scene_segments)
            ),
            entity_segments=list(entity_segments),
            scene_base_nodes=list(window_slice.scene_base_nodes),
            scene_base_edges=list(window_slice.scene_base_edges),
            scene_delta_nodes=list(window_slice.scene_delta_nodes),
            scene_delta_observations=list(window_slice.scene_delta_observations),
            scene_delta_edges=list(window_slice.scene_delta_edges),
            transition_events=list(window_slice.transition_events),
        )
        return IncrementalWindowDocument(
            plan=plan,
            slice=window_slice,
            document=window_document,
        )

    def execute_window_update(
        self,
        *,
        start_frame: int,
        end_frame: int,
        target_document: TemporalGraphDocument | None = None,
        scoped_entity_ids: tuple[str, ...] | set[str] | None = None,
        preserve_scene_segments: bool = False,
    ) -> IncrementalExecutionResult:
        base_document = target_document or self.document
        if base_document.video_id != self.document.video_id:
            raise ValueError("target_document.video_id must match planner.document.video_id")

        window_document = self.materialize_window_document(
            start_frame=start_frame,
            end_frame=end_frame,
            scoped_entity_ids=scoped_entity_ids,
            preserve_scene_segments=preserve_scene_segments,
        )
        updated_document, removed_counts, inserted_counts = self._apply_window_document(
            base_document=base_document,
            window_document=window_document,
            preserve_scene_segments=preserve_scene_segments,
        )
        document_unchanged = (
            self._document_signature(updated_document)
            == self._document_signature(base_document)
        )
        return IncrementalExecutionResult(
            plan=window_document.plan,
            window_document=window_document,
            updated_document=updated_document,
            removed_counts=removed_counts,
            inserted_counts=inserted_counts,
            document_unchanged=document_unchanged,
        )

    def _reused_scene_base_nodes(
        self,
        *,
        touched_scenes: list[SceneSegment],
        relevant_entity_ids: set[str],
    ) -> list[SceneBaseNode]:
        nodes: list[SceneBaseNode] = []
        for scene in touched_scenes:
            nodes.extend(
                node
                for node in self.index.scene_base_nodes_for_scene(scene.scene_id)
                if node.entity_id in relevant_entity_ids
            )
        nodes.sort(key=lambda item: (item.scene_id, item.node_id))
        return nodes

    def _reused_scene_base_edges(
        self,
        *,
        touched_scenes: list[SceneSegment],
        relevant_entity_ids: set[str],
    ) -> list[SceneBaseEdge]:
        edges: list[SceneBaseEdge] = []
        for scene in touched_scenes:
            edges.extend(
                edge
                for edge in self.index.scene_base_edges_for_scene(scene.scene_id)
                if edge.entity_id in relevant_entity_ids
            )
        edges.sort(key=lambda item: (item.scene_id, item.edge_id))
        return edges

    def _window_observation_entity_ids(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> set[str]:
        return {
            observation.entity_id
            for observation in self.document.observations
            if start_frame <= observation.frame_index <= end_frame
        }

    def _delta_observations_for_window(
        self,
        *,
        touched_scenes: list[SceneSegment],
        start_frame: int,
        end_frame: int,
    ) -> list[SceneDeltaObservation]:
        observations: list[SceneDeltaObservation] = []
        for scene in touched_scenes:
            for observation in self.index.scene_delta_observations_for_scene(scene.scene_id):
                if start_frame <= observation.frame_index <= end_frame:
                    observations.append(observation)
        observations.sort(
            key=lambda item: (item.frame_index, item.scene_id, item.entity_id, item.delta_observation_id)
        )
        return observations

    def _transition_events_for_window(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> list[TransitionEvent]:
        events = [
            event
            for event in self.document.transition_events
            if start_frame <= event.frame_index <= end_frame
        ]
        events.sort(key=lambda item: (item.frame_index, item.event_type, item.event_id))
        return events

    def _impacted_entity_segments(
        self,
        *,
        impacted_entity_ids: set[str],
        start_frame: int,
        end_frame: int,
    ) -> list[EntitySegment]:
        segments = [
            segment
            for segment in self.document.entity_segments
            if segment.entity_id in impacted_entity_ids
            and self._overlaps(segment.start_frame, segment.end_frame, start_frame, end_frame)
        ]
        segments.sort(key=lambda item: (item.start_frame, item.entity_id, item.segment_id))
        return segments

    def _impacted_state_episodes(
        self,
        *,
        impacted_entity_ids: set[str],
        start_frame: int,
        end_frame: int,
    ) -> list[StateEpisode]:
        episodes = [
            episode
            for episode in self.document.state_episodes
            if episode.entity_id in impacted_entity_ids
            and self._overlaps(episode.start_frame, episode.end_frame, start_frame, end_frame)
        ]
        episodes.sort(key=lambda item: (item.start_frame, item.entity_id, item.state_episode_id))
        return episodes

    def _impacted_relation_episodes(
        self,
        *,
        impacted_entity_ids: set[str],
        start_frame: int,
        end_frame: int,
    ) -> list[RelationEpisode]:
        episodes = [
            episode
            for episode in self.document.relation_episodes
            if (
                episode.subject_entity_id in impacted_entity_ids
                or episode.object_entity_id in impacted_entity_ids
            )
            and self._overlaps(episode.start_frame, episode.end_frame, start_frame, end_frame)
        ]
        episodes.sort(
            key=lambda item: (
                item.start_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            )
        )
        return episodes

    def _window_document_entity_ids(
        self,
        window_slice: DeltaWindowSlice,
    ) -> set[str]:
        entity_ids = {
            entity.entity_id
            for entity in window_slice.entities
        }
        entity_ids.update(
            node.entity_id
            for node in window_slice.scene_base_nodes
        )
        entity_ids.update(
            node.entity_id
            for node in window_slice.scene_delta_nodes
        )
        entity_ids.update(
            entity_id
            for event in window_slice.transition_events
            for entity_id in self.index._event_entity_ids(event)
        )
        entity_ids.update(
            episode.entity_id
            for episode in window_slice.state_episodes
        )
        entity_ids.update(
            episode.subject_entity_id
            for episode in window_slice.relation_episodes
        )
        entity_ids.update(
            episode.object_entity_id
            for episode in window_slice.relation_episodes
        )
        return entity_ids

    def _window_document_observations(
        self,
        *,
        start_frame: int,
        end_frame: int,
        included_entity_ids: set[str],
    ) -> list[Observation]:
        observations = [
            observation
            for observation in self.document.observations
            if observation.entity_id in included_entity_ids
            and start_frame <= observation.frame_index <= end_frame
        ]
        observations.sort(
            key=lambda item: (item.frame_index, item.frame_rank, item.entity_id, item.observation_id)
        )
        return observations

    def _window_document_entity_segments(
        self,
        *,
        start_frame: int,
        end_frame: int,
        included_entity_ids: set[str],
    ) -> list[EntitySegment]:
        segments = [
            segment
            for segment in self.document.entity_segments
            if segment.entity_id in included_entity_ids
            and self._overlaps(segment.start_frame, segment.end_frame, start_frame, end_frame)
        ]
        segments.sort(key=lambda item: (item.start_frame, item.entity_id, item.segment_id))
        return segments

    def _window_document_state_episodes(
        self,
        *,
        start_frame: int,
        end_frame: int,
        included_entity_ids: set[str],
    ) -> list[StateEpisode]:
        episodes = [
            episode
            for episode in self.document.state_episodes
            if episode.entity_id in included_entity_ids
            and self._overlaps(episode.start_frame, episode.end_frame, start_frame, end_frame)
        ]
        episodes.sort(key=lambda item: (item.start_frame, item.entity_id, item.state_episode_id))
        return episodes

    def _window_document_relation_episodes(
        self,
        *,
        start_frame: int,
        end_frame: int,
        included_entity_ids: set[str],
    ) -> list[RelationEpisode]:
        episodes = [
            episode
            for episode in self.document.relation_episodes
            if (
                episode.subject_entity_id in included_entity_ids
                and episode.object_entity_id in included_entity_ids
            )
            and self._overlaps(episode.start_frame, episode.end_frame, start_frame, end_frame)
        ]
        episodes.sort(
            key=lambda item: (
                item.start_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            )
        )
        return episodes

    def _apply_window_document(
        self,
        *,
        base_document: TemporalGraphDocument,
        window_document: IncrementalWindowDocument,
        preserve_scene_segments: bool = False,
    ) -> tuple[TemporalGraphDocument, dict[str, int], dict[str, int]]:
        plan = window_document.plan
        incremental_document = window_document.document
        included_entity_ids = {
            entity.entity_id
            for entity in incremental_document.entities
        }
        touched_scene_ids = set(plan.touched_scene_ids)
        start_frame = plan.start_frame
        end_frame = plan.end_frame

        observations, removed_observation_count = self._replace_collection(
            existing=base_document.observations,
            additions=incremental_document.observations,
            should_remove=lambda item: (
                item.entity_id in included_entity_ids
                and start_frame <= item.frame_index <= end_frame
            ),
            sort_key=lambda item: (
                item.frame_index,
                item.frame_rank,
                item.entity_id,
                item.observation_id,
            ),
        )
        entities, removed_entity_count = self._replace_collection(
            existing=base_document.entities,
            additions=incremental_document.entities,
            should_remove=lambda item: item.entity_id in included_entity_ids,
            sort_key=lambda item: (item.first_seen_frame, item.entity_id),
        )
        relation_episodes, removed_relation_episode_count = self._replace_collection(
            existing=base_document.relation_episodes,
            additions=incremental_document.relation_episodes,
            should_remove=lambda item: (
                (
                    item.subject_entity_id in included_entity_ids
                    and item.object_entity_id in included_entity_ids
                )
                and self._overlaps(item.start_frame, item.end_frame, start_frame, end_frame)
            ),
            sort_key=lambda item: (
                item.start_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            ),
        )
        state_episodes, removed_state_episode_count = self._replace_collection(
            existing=base_document.state_episodes,
            additions=incremental_document.state_episodes,
            should_remove=lambda item: (
                item.entity_id in included_entity_ids
                and self._overlaps(item.start_frame, item.end_frame, start_frame, end_frame)
            ),
            sort_key=lambda item: (
                item.start_frame,
                item.entity_id,
                item.state_episode_id,
            ),
        )
        if preserve_scene_segments:
            scene_segments = list(base_document.scene_segments)
            removed_scene_segment_count = 0
        else:
            scene_segments, removed_scene_segment_count = self._replace_collection(
                existing=base_document.scene_segments,
                additions=incremental_document.scene_segments,
                should_remove=lambda item: item.scene_id in touched_scene_ids,
                sort_key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
            )
        entity_segments, removed_entity_segment_count = self._replace_collection(
            existing=base_document.entity_segments,
            additions=incremental_document.entity_segments,
            should_remove=lambda item: (
                item.entity_id in included_entity_ids
                and self._overlaps(item.start_frame, item.end_frame, start_frame, end_frame)
            ),
            sort_key=lambda item: (item.start_frame, item.entity_id, item.segment_id),
        )
        scene_base_nodes, removed_scene_base_node_count = self._replace_collection(
            existing=base_document.scene_base_nodes,
            additions=incremental_document.scene_base_nodes,
            should_remove=lambda item: (
                item.scene_id in touched_scene_ids
                and item.entity_id in included_entity_ids
            ),
            sort_key=lambda item: (item.scene_id, item.node_id),
        )
        scene_base_edges, removed_scene_base_edge_count = self._replace_collection(
            existing=base_document.scene_base_edges,
            additions=incremental_document.scene_base_edges,
            should_remove=lambda item: (
                item.scene_id in touched_scene_ids
                and item.entity_id in included_entity_ids
            ),
            sort_key=lambda item: (item.scene_id, item.edge_id),
        )
        scene_delta_nodes, removed_scene_delta_node_count = self._replace_collection(
            existing=base_document.scene_delta_nodes,
            additions=incremental_document.scene_delta_nodes,
            should_remove=lambda item: (
                item.entity_id in included_entity_ids
                and self._overlaps(item.start_frame, item.end_frame, start_frame, end_frame)
            ),
            sort_key=lambda item: (item.scene_id, item.start_frame, item.entity_id, item.node_id),
        )
        scene_delta_observations, removed_scene_delta_observation_count = self._replace_collection(
            existing=base_document.scene_delta_observations,
            additions=incremental_document.scene_delta_observations,
            should_remove=lambda item: (
                item.entity_id in included_entity_ids
                and start_frame <= item.frame_index <= end_frame
            ),
            sort_key=lambda item: (
                item.frame_index,
                item.scene_id,
                item.entity_id,
                item.delta_observation_id,
            ),
        )
        scene_delta_edges, removed_scene_delta_edge_count = self._replace_collection(
            existing=base_document.scene_delta_edges,
            additions=incremental_document.scene_delta_edges,
            should_remove=lambda item: item.entity_id in included_entity_ids,
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        transition_events, removed_transition_event_count = self._replace_collection(
            existing=base_document.transition_events,
            additions=incremental_document.transition_events,
            should_remove=lambda item: (
                start_frame <= item.frame_index <= end_frame
                and bool(self.index._event_entity_ids(item) & included_entity_ids)
            ),
            sort_key=lambda item: (item.frame_index, item.event_type, item.event_id),
        )

        updated_document = TemporalGraphDocument(
            video_id=base_document.video_id,
            source_dir=self.document.source_dir,
            frame_count=max(base_document.frame_count, self.document.frame_count),
            observations=observations,
            entities=entities,
            relation_episodes=relation_episodes,
            state_episodes=state_episodes,
            scene_segments=scene_segments,
            entity_segments=entity_segments,
            scene_base_nodes=scene_base_nodes,
            scene_base_edges=scene_base_edges,
            scene_delta_nodes=scene_delta_nodes,
            scene_delta_observations=scene_delta_observations,
            scene_delta_edges=scene_delta_edges,
            transition_events=transition_events,
        )

        removed_counts = {
            "observations": removed_observation_count,
            "entities": removed_entity_count,
            "relation_episodes": removed_relation_episode_count,
            "state_episodes": removed_state_episode_count,
            "scene_segments": removed_scene_segment_count,
            "entity_segments": removed_entity_segment_count,
            "scene_base_nodes": removed_scene_base_node_count,
            "scene_base_edges": removed_scene_base_edge_count,
            "scene_delta_nodes": removed_scene_delta_node_count,
            "scene_delta_observations": removed_scene_delta_observation_count,
            "scene_delta_edges": removed_scene_delta_edge_count,
            "transition_events": removed_transition_event_count,
        }
        inserted_counts = {
            "observations": len(incremental_document.observations),
            "entities": len(incremental_document.entities),
            "relation_episodes": len(incremental_document.relation_episodes),
            "state_episodes": len(incremental_document.state_episodes),
            "scene_segments": len(incremental_document.scene_segments),
            "entity_segments": len(incremental_document.entity_segments),
            "scene_base_nodes": len(incremental_document.scene_base_nodes),
            "scene_base_edges": len(incremental_document.scene_base_edges),
            "scene_delta_nodes": len(incremental_document.scene_delta_nodes),
            "scene_delta_observations": len(incremental_document.scene_delta_observations),
            "scene_delta_edges": len(incremental_document.scene_delta_edges),
            "transition_events": len(incremental_document.transition_events),
        }
        return updated_document, removed_counts, inserted_counts

    def _replace_collection(
        self,
        *,
        existing: list,
        additions: list,
        should_remove,
        sort_key,
    ) -> tuple[list, int]:
        kept = [
            item
            for item in existing
            if not should_remove(item)
        ]
        removed_count = len(existing) - len(kept)
        merged = [*kept, *additions]
        merged.sort(key=sort_key)
        return merged, removed_count

    def _document_signature(
        self,
        document: TemporalGraphDocument,
    ) -> dict:
        return {
            "video_id": document.video_id,
            "source_dir": document.source_dir.as_posix(),
            "frame_count": document.frame_count,
            "observations": self._sorted_model_dump(
                document.observations,
                key=lambda item: (
                    item.frame_index,
                    item.frame_rank,
                    item.entity_id,
                    item.observation_id,
                ),
            ),
            "entities": self._sorted_model_dump(
                document.entities,
                key=lambda item: (
                    item.first_seen_frame,
                    item.last_seen_frame,
                    item.entity_id,
                ),
            ),
            "relation_episodes": self._sorted_model_dump(
                document.relation_episodes,
                key=lambda item: (
                    item.start_frame,
                    item.end_frame,
                    item.subject_entity_id,
                    item.predicate,
                    item.object_entity_id,
                    item.relation_episode_id,
                ),
            ),
            "state_episodes": self._sorted_model_dump(
                document.state_episodes,
                key=lambda item: (
                    item.start_frame,
                    item.end_frame,
                    item.entity_id,
                    item.state_episode_id,
                ),
            ),
            "scene_segments": self._sorted_model_dump(
                document.scene_segments,
                key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
            ),
            "entity_segments": self._sorted_model_dump(
                document.entity_segments,
                key=lambda item: (
                    item.start_frame,
                    item.end_frame,
                    item.entity_id,
                    item.segment_id,
                ),
            ),
            "scene_base_nodes": self._sorted_model_dump(
                document.scene_base_nodes,
                key=lambda item: (item.scene_id, item.entity_id, item.node_id),
            ),
            "scene_base_edges": self._sorted_model_dump(
                document.scene_base_edges,
                key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
            ),
            "scene_delta_nodes": self._sorted_model_dump(
                document.scene_delta_nodes,
                key=lambda item: (
                    item.scene_id,
                    item.start_frame,
                    item.end_frame,
                    item.entity_id,
                    item.node_id,
                ),
            ),
            "scene_delta_observations": self._sorted_model_dump(
                document.scene_delta_observations,
                key=lambda item: (
                    item.frame_index,
                    item.scene_id,
                    item.entity_id,
                    item.delta_observation_id,
                ),
            ),
            "scene_delta_edges": self._sorted_model_dump(
                document.scene_delta_edges,
                key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
            ),
            "transition_events": self._sorted_model_dump(
                document.transition_events,
                key=lambda item: (
                    item.frame_index,
                    item.frame_rank,
                    item.event_type,
                    item.event_id,
                ),
            ),
        }

    def _sorted_model_dump(
        self,
        values: list,
        *,
        key,
    ) -> list[dict]:
        return [
            value.model_dump(mode="json")
            for value in sorted(values, key=key)
        ]

    def _overlaps(
        self,
        left_start: int,
        left_end: int,
        right_start: int,
        right_end: int,
    ) -> bool:
        return not (left_end < right_start or left_start > right_end)
