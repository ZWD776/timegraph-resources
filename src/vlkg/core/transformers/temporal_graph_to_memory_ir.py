from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from vlkg.core.models.memory_ir import (
    EntityIdentity,
    EntityStateUpdate,
    InteractionEvent,
    MemoryIRDocument,
    SceneAnchor,
    SceneAnchorSnapshot,
)
from vlkg.core.models.temporal_graph import Entity, RelationEpisode, SceneSegment, StateEpisode, TemporalGraphDocument


class TemporalGraphToMemoryIRConverter:
    def __init__(
        self,
        document: TemporalGraphDocument,
        *,
        source_graph_path: str | Path | None = None,
        include_fragments: bool = True,
        include_unmapped_relations: bool = False,
    ) -> None:
        self.document = document
        self.source_graph_path = (
            Path(source_graph_path).resolve()
            if source_graph_path is not None
            else None
        )
        self.include_fragments = include_fragments
        self.include_unmapped_relations = include_unmapped_relations
        self._entity_by_id = {entity.entity_id: entity for entity in document.entities}
        self._scene_segments = sorted(
            document.scene_segments,
            key=lambda item: (item.start_frame, item.scene_id),
        )

    def convert(self) -> MemoryIRDocument:
        dynamic_entities = self._collect_dynamic_entities()
        dynamic_entity_ids = {entity.entity_id for entity in dynamic_entities}

        return MemoryIRDocument(
            video_id=self.document.video_id,
            frame_count=self.document.frame_count,
            source_graph_path=self.source_graph_path,
            scene_anchor_snapshots=self._build_scene_anchor_snapshots(),
            entity_identities=self._build_entity_identities(dynamic_entities),
            entity_state_updates=self._build_entity_state_updates(dynamic_entity_ids),
            interaction_events=self._build_interaction_events(dynamic_entity_ids),
        )

    def _collect_dynamic_entities(self) -> list[Entity]:
        entities = [
            entity
            for entity in self.document.entities
            if self._is_dynamic_principal_entity(entity)
        ]
        if not self.include_fragments:
            entities = [entity for entity in entities if not entity.is_fragment]
        return sorted(
            entities,
            key=lambda item: (item.first_seen_frame, item.category, item.entity_id),
        )

    def _build_scene_anchor_snapshots(self) -> list[SceneAnchorSnapshot]:
        anchors_by_scene: dict[str, list[SceneAnchor]] = defaultdict(list)
        for node in sorted(
            self.document.scene_base_nodes,
            key=lambda item: (item.scene_id, -item.coverage_ratio, item.label, item.entity_id),
        ):
            entity = self._entity_by_id.get(node.entity_id)
            if entity is None or not self._is_static_anchor_entity(entity):
                continue
            anchors_by_scene[node.scene_id].append(
                SceneAnchor(
                    entity_id=node.entity_id,
                    label=node.label,
                    first_frame=node.first_frame,
                    last_frame=node.last_frame,
                    support_frames=node.support_frames,
                    coverage_ratio=node.coverage_ratio,
                    canonical_image_path=node.canonical_image_path,
                    parent_labels_seen=list(node.parent_labels_seen),
                    states_seen=list(node.states_seen),
                )
            )

        snapshots: list[SceneAnchorSnapshot] = []
        for scene in self._scene_segments:
            snapshots.append(
                SceneAnchorSnapshot(
                    snapshot_id=f"{scene.scene_id}:anchors",
                    video_id=scene.video_id,
                    scene_id=scene.scene_id,
                    start_frame=scene.start_frame,
                    end_frame=scene.end_frame,
                    start_ts=scene.start_ts,
                    end_ts=scene.end_ts,
                    support_frames=scene.support_frames,
                    representative_static_labels=list(scene.representative_static_labels),
                    representative_dynamic_labels=list(scene.representative_dynamic_labels),
                    scene_kind_hint=scene.scene_kind_hint,
                    anchors=anchors_by_scene.get(scene.scene_id, []),
                )
            )
        return snapshots

    def _build_entity_identities(
        self,
        entities: list[Entity],
    ) -> list[EntityIdentity]:
        scene_ids_by_entity = self._scene_ids_by_entity()
        identities: list[EntityIdentity] = []
        for entity in entities:
            identities.append(
                EntityIdentity(
                    memory_entity_id=entity.entity_id,
                    video_id=entity.video_id,
                    entity_id=entity.entity_id,
                    label=entity.category,
                    first_seen_frame=entity.first_seen_frame,
                    last_seen_frame=entity.last_seen_frame,
                    first_seen_ts=entity.first_seen_ts,
                    last_seen_ts=entity.last_seen_ts,
                    observation_count=entity.observation_count,
                    labels_seen=list(entity.labels_seen),
                    states_seen=list(entity.states_seen),
                    parent_labels_seen=list(entity.parent_labels_seen),
                    scene_ids=scene_ids_by_entity.get(entity.entity_id, []),
                    canonical_image_path=entity.canonical_image_path,
                    is_fragment=entity.is_fragment,
                )
            )
        return identities

    def _build_entity_state_updates(
        self,
        dynamic_entity_ids: set[str],
    ) -> list[EntityStateUpdate]:
        updates: list[EntityStateUpdate] = []
        for episode in sorted(
            self.document.state_episodes,
            key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.entity_id,
                item.attr_value,
                item.state_episode_id,
            ),
        ):
            if episode.entity_id not in dynamic_entity_ids:
                continue
            entity = self._entity_by_id.get(episode.entity_id)
            if entity is None:
                continue
            updates.append(
                EntityStateUpdate(
                    update_id=episode.state_episode_id,
                    video_id=episode.video_id,
                    entity_id=episode.entity_id,
                    label=entity.category,
                    scene_id=self._scene_id_for_range(
                        episode.start_frame,
                        episode.end_frame,
                    ),
                    attr_key=episode.attr_key,
                    attr_value=episode.attr_value,
                    start_frame=episode.start_frame,
                    end_frame=episode.end_frame,
                    start_ts=episode.start_ts,
                    end_ts=episode.end_ts,
                    support_frames=episode.support_frames,
                    source_state_episode_id=episode.state_episode_id,
                )
            )
        return updates

    def _build_interaction_events(
        self,
        dynamic_entity_ids: set[str],
    ) -> list[InteractionEvent]:
        events: list[InteractionEvent] = []
        for episode in sorted(
            self.document.relation_episodes,
            key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            ),
        ):
            if not self._should_keep_relation_episode(
                episode=episode,
                dynamic_entity_ids=dynamic_entity_ids,
            ):
                continue
            subject = self._entity_by_id.get(episode.subject_entity_id)
            object_entity = self._entity_by_id.get(episode.object_entity_id)
            if subject is None or object_entity is None:
                continue
            events.append(
                InteractionEvent(
                    event_id=episode.relation_episode_id,
                    video_id=episode.video_id,
                    scene_id=self._scene_id_for_range(
                        episode.start_frame,
                        episode.end_frame,
                    ),
                    subject_entity_id=episode.subject_entity_id,
                    subject_label=subject.category,
                    predicate=episode.predicate,
                    raw_predicate=episode.raw_predicate,
                    object_entity_id=episode.object_entity_id,
                    object_label=object_entity.category,
                    object_role=("static" if self._is_static_anchor_entity(object_entity) else "dynamic"),
                    start_frame=episode.start_frame,
                    end_frame=episode.end_frame,
                    start_ts=episode.start_ts,
                    end_ts=episode.end_ts,
                    support_frames=episode.support_frames,
                    source_relation_episode_id=episode.relation_episode_id,
                )
            )
        return events

    def _should_keep_relation_episode(
        self,
        *,
        episode: RelationEpisode,
        dynamic_entity_ids: set[str],
    ) -> bool:
        if episode.subject_entity_id not in dynamic_entity_ids:
            return False
        object_entity = self._entity_by_id.get(episode.object_entity_id)
        if object_entity is None or object_entity.is_noise:
            return False
        if not (
            self._is_dynamic_principal_entity(object_entity)
            or self._is_static_anchor_entity(object_entity)
        ):
            return False
        if (
            not self.include_fragments
            and self._is_dynamic_principal_entity(object_entity)
            and object_entity.is_fragment
        ):
            return False
        if (
            not self.include_unmapped_relations
            and episode.predicate == "UNMAPPED_RELATION"
        ):
            return False
        return True

    def _scene_ids_by_entity(self) -> dict[str, list[str]]:
        scene_ids: dict[str, list[str]] = defaultdict(list)
        for segment in sorted(
            self.document.entity_segments,
            key=lambda item: (
                item.entity_id,
                item.start_frame,
                item.segment_id,
            ),
        ):
            if not segment.scene_id:
                continue
            existing = scene_ids[segment.entity_id]
            if segment.scene_id not in existing:
                existing.append(segment.scene_id)
        return dict(scene_ids)

    def _scene_id_for_range(
        self,
        start_frame: int,
        end_frame: int,
    ) -> str | None:
        best_scene: SceneSegment | None = None
        best_overlap = -1
        for scene in self._scene_segments:
            overlap = min(end_frame, scene.end_frame) - max(start_frame, scene.start_frame) + 1
            if overlap <= 0:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_scene = scene
        if best_scene is not None:
            return best_scene.scene_id
        for scene in self._scene_segments:
            if scene.start_frame <= start_frame <= scene.end_frame:
                return scene.scene_id
        return None

    def _is_dynamic_principal_entity(self, entity: Entity) -> bool:
        if entity.is_noise or entity.is_static:
            return False
        return entity.node_role not in {"dependent", "noise", "static"}

    def _is_static_anchor_entity(self, entity: Entity) -> bool:
        if entity.is_noise:
            return False
        return entity.is_static or entity.node_role == "static"


def build_memory_ir(
    document: TemporalGraphDocument,
    *,
    source_graph_path: str | Path | None = None,
    include_fragments: bool = True,
    include_unmapped_relations: bool = False,
) -> MemoryIRDocument:
    converter = TemporalGraphToMemoryIRConverter(
        document,
        source_graph_path=source_graph_path,
        include_fragments=include_fragments,
        include_unmapped_relations=include_unmapped_relations,
    )
    return converter.convert()
