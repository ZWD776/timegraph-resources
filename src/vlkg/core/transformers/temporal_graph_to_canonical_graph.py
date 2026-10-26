from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from vlkg.core.models.canonical_graph import (
    CanonicalEdgeType,
    CanonicalEpisodeKind,
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalInteractionTypeNode,
    CanonicalRelationObservedSample,
    CanonicalSceneNode,
    CanonicalStableIdentityNode,
    CanonicalStateValueNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
    CanonicalTransitionEventNode,
    CanonicalEntityNode,
)
from vlkg.core.models.temporal_graph import (
    Entity,
    EntitySegment,
    RelationEpisode,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.config.semantic_labels import (
    canonical_label,
    is_salient_label,
    label_alias_terms,
    semantic_group,
)
from vlkg.core.transformers.temporal_graph_to_stable_identity import (
    StableIdentityMergeMode,
    build_stable_identity_document,
)


_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")
_TEMPORAL_SEMANTIC_EPISODE_KINDS = {"STATE", "RELATION"}
_TEMPORAL_EVENT_EPISODE_KINDS = {"ENTITY_SEGMENT", "STATE", "RELATION"}
_TEMPORAL_NEAR_MAX_FRAME_GAP = 5
_TEMPORAL_NEAR_MAX_TS_GAP = 1.0
_START_EVENT_TYPES = {"APPEAR", "RELATION_START"}
_END_EVENT_TYPES = {"DISAPPEAR", "RELATION_END"}
_HUMAN_LIKE_LABELS = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "human",
    "player",
}


class TemporalGraphToCanonicalGraphConverter:
    def __init__(
        self,
        document: TemporalGraphDocument,
        *,
        source_graph_path: str | Path | None = None,
        include_stable_identities: bool = True,
        compact_relation_observed_events: bool = False,
        compact_derived_temporal_edges: bool = True,
        compact_previous_episode_edges: bool = True,
        compact_scene_label_entities: bool = False,
        compact_scene_label_min_members: int = 4,
        compact_scene_label_entity_segments: bool = True,
        include_scene_segment_episodes: bool = True,
        include_entity_segment_episodes: bool = True,
        include_interaction_type_nodes: bool = True,
        stable_identity_merge_mode: StableIdentityMergeMode = "semantic",
        stable_identity_include_fragments: bool = False,
        stable_identity_include_noise: bool = False,
    ) -> None:
        self.document = document
        self.source_graph_path = (
            Path(source_graph_path).resolve()
            if source_graph_path is not None
            else None
        )
        self.include_stable_identities = include_stable_identities
        self.compact_relation_observed_events = compact_relation_observed_events
        self.compact_derived_temporal_edges = compact_derived_temporal_edges
        self.compact_previous_episode_edges = compact_previous_episode_edges
        self.compact_scene_label_entities = compact_scene_label_entities
        self.compact_scene_label_min_members = max(2, compact_scene_label_min_members)
        self.compact_scene_label_entity_segments = compact_scene_label_entity_segments
        self.include_scene_segment_episodes = include_scene_segment_episodes
        self.include_entity_segment_episodes = include_entity_segment_episodes
        self.include_interaction_type_nodes = include_interaction_type_nodes

        self._entity_by_id = {
            entity.entity_id: entity
            for entity in document.entities
        }
        self._entity_ids_by_scene: dict[str, set[str]] = defaultdict(set)
        for segment in document.entity_segments:
            if segment.scene_id:
                self._entity_ids_by_scene[segment.scene_id].add(segment.entity_id)
        self._stable_identity_document = (
            build_stable_identity_document(
                document,
                source_graph_path=source_graph_path,
                include_fragments=stable_identity_include_fragments,
                include_noise=stable_identity_include_noise,
                merge_mode=stable_identity_merge_mode,
            )
            if include_stable_identities
            else None
        )
        self._stable_identity_by_entity_id: dict[str, object] = {}
        self._stable_identity_node_ids: dict[str, str] = {}
        if self._stable_identity_document is not None:
            for identity in self._stable_identity_document.identities:
                for member in identity.member_entities:
                    self._stable_identity_by_entity_id[member.entity_id] = identity
        self._scene_segments = sorted(
            document.scene_segments,
            key=lambda item: (item.start_frame, item.scene_id),
        )
        self._entity_segments = sorted(
            document.entity_segments,
            key=lambda item: (item.entity_id, item.start_frame, item.segment_id),
        )
        self._scene_node_ids: dict[str, str] = {}
        self._entity_node_ids: dict[str, str] = {}
        self._scene_segment_episode_node_ids: dict[str, str] = {}
        self._entity_segment_episode_node_ids: dict[str, str] = {}
        self._state_episode_node_ids: dict[str, str] = {}
        self._relation_episode_node_ids: dict[str, str] = {}
        self._state_value_node_ids: dict[tuple[str, str], str] = {}
        self._interaction_type_node_ids: dict[str, str] = {}
        self._episode_time_ranges: dict[str, CanonicalTimeRange] = {}
        self._edges: dict[str, CanonicalGraphEdge] = {}
        self._relation_observed_samples: list[CanonicalRelationObservedSample] = []
        self._compact_entity_id_by_original_entity_id: dict[str, str] = {}
        self._compact_entity_members_by_anchor_id: dict[str, list[Entity]] = {}
        self._compact_entity_scene_by_anchor_id: dict[str, str] = {}
        self._compact_segment_node_id_by_key: dict[tuple[str | None, str, str], str] = {}

    def convert(self) -> CanonicalTemporalMemoryGraph:
        scene_nodes, scene_episode_nodes = self._build_scene_nodes_and_episodes()
        entity_nodes = self._build_entity_nodes()
        stable_identity_nodes = self._build_stable_identity_nodes()
        entity_segment_nodes = self._build_entity_segment_episode_nodes()
        state_value_nodes, state_episode_nodes = self._build_state_nodes_and_episodes()
        interaction_type_nodes, relation_episode_nodes = self._build_relation_nodes_and_episodes()
        transition_event_nodes = self._build_transition_event_nodes()

        episode_nodes = (
            scene_episode_nodes
            + entity_segment_nodes
            + state_episode_nodes
            + relation_episode_nodes
        )
        self._add_temporal_semantic_edges(
            episode_nodes=episode_nodes,
            transition_event_nodes=transition_event_nodes,
        )

        return CanonicalTemporalMemoryGraph(
            video_id=self.document.video_id,
            source_graph_path=self.source_graph_path,
            scene_nodes=scene_nodes,
            entity_nodes=entity_nodes,
            stable_identity_nodes=stable_identity_nodes,
            state_value_nodes=state_value_nodes,
            interaction_type_nodes=interaction_type_nodes,
            episode_nodes=episode_nodes,
            transition_event_nodes=transition_event_nodes,
            relation_observed_samples=self._relation_observed_samples,
            edges=list(self._edges.values()),
        )

    def _build_scene_nodes_and_episodes(
        self,
    ) -> tuple[list[CanonicalSceneNode], list[CanonicalEpisodeNode]]:
        scene_nodes: list[CanonicalSceneNode] = []
        episode_nodes: list[CanonicalEpisodeNode] = []
        chain_node_ids: list[str] = []

        for scene in self._scene_segments:
            time_range = _time_range(
                start_frame=scene.start_frame,
                end_frame=scene.end_frame,
                start_ts=scene.start_ts,
                end_ts=scene.end_ts,
                support_frames=scene.support_frames,
            )
            scene_node_id = f"scene:{scene.scene_id}"
            episode_node_id = f"episode:scene_segment:{scene.scene_id}"
            label = scene.scene_kind_hint or f"scene {scene.scene_index}"
            scene_entity_labels = [
                self._entity_by_id[entity_id].category
                for entity_id in sorted(self._entity_ids_by_scene.get(scene.scene_id, set()))
                if entity_id in self._entity_by_id
            ]
            salient_labels = _salient_scene_labels(scene, entity_labels=scene_entity_labels)
            scene_alias_terms = _alias_terms_for_labels(
                [
                    label,
                    *(scene.representative_static_labels or []),
                    *(scene.representative_dynamic_labels or []),
                    *(scene.entry_static_labels or []),
                    *(scene.exit_static_labels or []),
                    *salient_labels,
                ]
            )

            self._scene_node_ids[scene.scene_id] = scene_node_id

            scene_nodes.append(
                CanonicalSceneNode(
                    node_id=scene_node_id,
                    video_id=scene.video_id,
                    scene_id=scene.scene_id,
                    label=label,
                    time_range=time_range,
                    scene_index=scene.scene_index,
                    scene_count=scene.scene_count,
                    previous_scene_id=scene.previous_scene_id,
                    next_scene_id=scene.next_scene_id,
                    representative_static_entity_ids=list(scene.representative_static_entity_ids),
                    representative_static_labels=_merge_unique(
                        list(scene.representative_static_labels),
                        salient_labels,
                        limit=10,
                    ),
                    representative_dynamic_labels=list(scene.representative_dynamic_labels),
                    entry_static_labels=list(scene.entry_static_labels),
                    exit_static_labels=list(scene.exit_static_labels),
                    salient_labels=salient_labels,
                    semantic_alias_terms=scene_alias_terms,
                    scene_kind_hint=scene.scene_kind_hint,
                    boundary_from_previous_reason=scene.boundary_from_previous_reason,
                    boundary_to_next_reason=scene.boundary_to_next_reason,
                    boundary_from_previous_entity_similarity=scene.boundary_from_previous_entity_similarity,
                    boundary_to_next_entity_similarity=scene.boundary_to_next_entity_similarity,
                    boundary_from_previous_label_similarity=scene.boundary_from_previous_label_similarity,
                    boundary_to_next_label_similarity=scene.boundary_to_next_label_similarity,
                )
            )
            if self.include_scene_segment_episodes:
                self._scene_segment_episode_node_ids[scene.scene_id] = episode_node_id
                self._episode_time_ranges[episode_node_id] = time_range
                episode_nodes.append(
                    CanonicalEpisodeNode(
                        node_id=episode_node_id,
                        video_id=scene.video_id,
                        episode_id=scene.scene_id,
                        episode_kind="SCENE_SEGMENT",
                        label=f"scene segment {scene.scene_id}",
                        time_range=time_range,
                        scene_id=scene.scene_id,
                        segment_index=scene.scene_index,
                        segment_count=scene.scene_count,
                        source_ref_id=scene.scene_id,
                        attributes={
                            "scene_kind_hint": scene.scene_kind_hint,
                            "semantic_alias_terms": scene_alias_terms,
                            "salient_labels": salient_labels,
                            "previous_scene_id": scene.previous_scene_id,
                            "next_scene_id": scene.next_scene_id,
                        },
                    )
                )
                chain_node_ids.append(episode_node_id)
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=scene_node_id,
                    target_node_id=episode_node_id,
                    suffix="scene_segment",
                    time_range=time_range,
                    attributes={"scope": "scene_segment"},
                )
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=episode_node_id,
                    target_node_id=scene_node_id,
                    suffix="scene_segment",
                    time_range=time_range,
                )

        self._add_temporal_chain(
            chain_node_ids,
            suffix_prefix="scene_segment_chain",
        )
        return scene_nodes, episode_nodes

    def _build_entity_nodes(self) -> list[CanonicalEntityNode]:
        entity_nodes: list[CanonicalEntityNode] = []
        entities = self._canonical_entity_inputs()
        for entity in sorted(
            entities,
            key=lambda item: (item.first_seen_frame, item.category, item.entity_id),
        ):
            node_id = f"entity:{entity.entity_id}"
            self._entity_node_ids[entity.entity_id] = node_id
            stable_identity = self._stable_identity_by_entity_id.get(entity.entity_id)
            compact_members = self._compact_entity_members_by_anchor_id.get(entity.entity_id)
            entity_canonical_label = canonical_label(entity.category) or entity.category
            entity_semantic_group = semantic_group(entity.category)
            entity_alias_terms = _alias_terms_for_labels(
                [
                    entity.category,
                    *entity.labels_seen,
                ]
            )
            attributes: dict[str, object] = {
                "canonical_label": entity_canonical_label,
                "semantic_group": entity_semantic_group,
                "semantic_alias_terms": entity_alias_terms,
                "is_salient_label": is_salient_label(entity.category),
            }
            raw_visual_attributes = _dedupe_text_values(
                [
                    *entity.raw_states_seen,
                    *entity.states_seen,
                ]
            )
            if raw_visual_attributes:
                attributes["raw_visual_attributes"] = raw_visual_attributes
            if compact_members:
                compact_alias_terms = _alias_terms_for_labels(
                    [
                        entity.category,
                        *entity.labels_seen,
                        *[
                            label
                            for member in compact_members
                            for label in (member.labels_seen or [member.category])
                        ],
                    ]
                )
                attributes.update({
                    "compaction_kind": "scene_label_entity_anchor",
                    "member_entity_ids": [member.entity_id for member in compact_members],
                    "member_count": len(compact_members),
                    "scene_id": self._compact_entity_scene_by_anchor_id.get(entity.entity_id),
                    "labels_seen": sorted(
                        {
                            label
                            for member in compact_members
                            for label in (member.labels_seen or [member.category])
                        }
                    ),
                    "node_roles_seen": sorted({member.node_role for member in compact_members}),
                    "source_entity_ids": [member.entity_id for member in compact_members],
                    "semantic_alias_terms": compact_alias_terms,
                })
            entity_nodes.append(
                CanonicalEntityNode(
                    node_id=node_id,
                    video_id=entity.video_id,
                    entity_id=entity.entity_id,
                    label=entity.category,
                    time_range=_time_range(
                        start_frame=entity.first_seen_frame,
                        end_frame=entity.last_seen_frame,
                        start_ts=entity.first_seen_ts,
                        end_ts=entity.last_seen_ts,
                        support_frames=entity.observation_count,
                    ),
                    observation_count=entity.observation_count,
                    node_role=entity.node_role,
                    is_static=entity.is_static,
                    is_noise=entity.is_noise,
                    is_fragment=entity.is_fragment,
                    stable_identity_id=(
                        stable_identity.stable_id
                        if stable_identity is not None
                        else None
                    ),
                    stable_identity_display_name=(
                        stable_identity.display_name
                        if stable_identity is not None
                        else None
                    ),
                    stable_identity_canonical_label=(
                        stable_identity.canonical_label
                        if stable_identity is not None
                        else None
                    ),
                    labels_seen=list(entity.labels_seen),
                    states_seen=list(entity.states_seen),
                    parent_labels_seen=list(entity.parent_labels_seen),
                    canonical_image_path=entity.canonical_image_path,
                    attributes=attributes,
                )
            )
        return entity_nodes

    def _canonical_entity_inputs(self) -> list[Entity]:
        if not self.compact_scene_label_entities:
            return list(self.document.entities)

        segment_scene_by_entity: dict[str, set[str]] = defaultdict(set)
        for segment in self._entity_segments:
            if segment.scene_id:
                segment_scene_by_entity[segment.entity_id].add(segment.scene_id)

        groups: dict[tuple[str, str, str], list[Entity]] = defaultdict(list)
        passthrough: list[Entity] = []
        for entity in self.document.entities:
            scenes = sorted(segment_scene_by_entity.get(entity.entity_id, set()))
            if len(scenes) != 1 or not self._eligible_for_scene_label_compaction(entity):
                passthrough.append(entity)
                continue
            key = (scenes[0], _stable_token(entity.category), entity.node_role)
            groups[key].append(entity)

        compact_entities: list[Entity] = []
        for key, members in sorted(groups.items()):
            scene_id, label_token, node_role = key
            if len(members) < self.compact_scene_label_min_members:
                passthrough.extend(members)
                continue
            if any(is_salient_label(member.category) for member in members):
                passthrough.extend(members)
                continue
            members = sorted(members, key=lambda item: (item.first_seen_frame, item.entity_id))
            anchor_id = (
                f"{self.document.video_id}:compact_entity:"
                f"{_stable_token(scene_id)}:{label_token}:{node_role}"
            )
            first_seen_frame = min(member.first_seen_frame for member in members)
            last_seen_frame = max(member.last_seen_frame for member in members)
            first_seen_ts = min(member.first_seen_ts for member in members)
            last_seen_ts = max(member.last_seen_ts for member in members)
            observation_count = sum(member.observation_count for member in members)
            labels_seen = sorted(
                {
                    label
                    for member in members
                    for label in (member.labels_seen or [member.category])
                }
            )
            states_seen = sorted(
                {
                    state
                    for member in members
                    for state in member.states_seen
                }
            )
            raw_states_seen = _dedupe_text_values(
                value
                for member in members
                for value in (member.raw_states_seen or member.states_seen)
            )
            parent_labels_seen = sorted(
                {
                    label
                    for member in members
                    for label in member.parent_labels_seen
                }
            )
            is_static = any(member.is_static for member in members) or node_role == "static"
            is_noise = all(member.is_noise for member in members)
            compact_entity = Entity(
                entity_id=anchor_id,
                video_id=members[0].video_id,
                category=members[0].category,
                first_seen_frame=first_seen_frame,
                last_seen_frame=last_seen_frame,
                first_seen_ts=first_seen_ts,
                last_seen_ts=last_seen_ts,
                observation_count=observation_count,
                canonical_image_path=members[0].canonical_image_path,
                labels_seen=labels_seen,
                raw_labels_seen=labels_seen,
                states_seen=states_seen,
                raw_states_seen=raw_states_seen,
                parent_labels_seen=parent_labels_seen,
                node_role=node_role,
                is_static=is_static,
                is_noise=is_noise,
                is_fragment=all(member.is_fragment for member in members),
                graph_origin="HYBRID",
            )
            compact_entities.append(compact_entity)
            self._compact_entity_members_by_anchor_id[anchor_id] = members
            self._compact_entity_scene_by_anchor_id[anchor_id] = scene_id
            for member in members:
                self._compact_entity_id_by_original_entity_id[member.entity_id] = anchor_id

        return passthrough + compact_entities

    def _eligible_for_scene_label_compaction(self, entity: Entity) -> bool:
        if entity.is_noise:
            return False
        if entity.node_role in {"static", "dependent"}:
            return True
        if entity.is_fragment:
            return True
        if entity.observation_count <= 2 and entity.category.strip().lower() not in _HUMAN_LIKE_LABELS:
            return True
        return False

    def _canonical_entity_id(self, entity_id: str | None) -> str | None:
        if entity_id is None:
            return None
        return self._compact_entity_id_by_original_entity_id.get(entity_id, entity_id)

    def _build_stable_identity_nodes(self) -> list[CanonicalStableIdentityNode]:
        if self._stable_identity_document is None:
            return []

        stable_identity_nodes: list[CanonicalStableIdentityNode] = []
        for identity in self._stable_identity_document.identities:
            node_id = f"stable_identity:{identity.stable_id}"
            self._stable_identity_node_ids[identity.stable_id] = node_id
            member_entity_ids = [
                member.entity_id
                for member in identity.member_entities
            ]
            raw_visual_attributes = _dedupe_text_values(
                value
                for entity_id in member_entity_ids
                for value in (
                    self._entity_by_id.get(entity_id).raw_states_seen
                    if self._entity_by_id.get(entity_id) is not None
                    else []
                )
            )
            identity_attributes = dict(identity.attributes)
            if raw_visual_attributes:
                identity_attributes["raw_visual_attributes"] = raw_visual_attributes
            time_range = _time_range(
                start_frame=identity.time_range.start_frame,
                end_frame=identity.time_range.end_frame,
                start_ts=identity.time_range.start_ts,
                end_ts=identity.time_range.end_ts,
                support_frames=identity.time_range.support_frames,
            )
            stable_identity_nodes.append(
                CanonicalStableIdentityNode(
                    node_id=node_id,
                    video_id=identity.video_id,
                    stable_id=identity.stable_id,
                    identity_type=identity.identity_type,
                    canonical_label=identity.canonical_label,
                    display_name=identity.display_name,
                    time_range=time_range,
                    observation_count=identity.observation_count,
                    member_entity_ids=member_entity_ids,
                    labels_seen=list(identity.labels_seen),
                    signature_tokens=list(identity.signature_tokens),
                    scene_ids=list(identity.scene_ids),
                    attributes=identity_attributes,
                )
            )

            for member in identity.member_entities:
                entity_node_id = self._entity_node_ids.get(member.entity_id)
                if entity_node_id is None:
                    continue
                self._add_edge(
                    edge_type="HAS_MEMBER_ENTITY",
                    source_node_id=node_id,
                    target_node_id=entity_node_id,
                    suffix=member.entity_id,
                    time_range=_time_range(
                        start_frame=member.first_seen_frame,
                        end_frame=member.last_seen_frame,
                        start_ts=member.first_seen_ts,
                        end_ts=member.last_seen_ts,
                        support_frames=member.observation_count,
                    ),
                    attributes={
                        "stable_id": identity.stable_id,
                        "member_entity_id": member.entity_id,
                    },
                )

            for scene_id in identity.scene_ids:
                scene_node_id = self._scene_node_ids.get(scene_id)
                if scene_node_id is None:
                    continue
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=scene_node_id,
                    suffix=f"stable_identity:{identity.stable_id}:{scene_id}",
                    time_range=time_range,
                    attributes={"stable_id": identity.stable_id},
                )
        return stable_identity_nodes

    def _build_entity_segment_episode_nodes(self) -> list[CanonicalEpisodeNode]:
        if not self.include_entity_segment_episodes:
            self._add_direct_entity_scene_presence_edges()
            return []

        if self.compact_scene_label_entities and self.compact_scene_label_entity_segments:
            return self._build_compact_entity_segment_episode_nodes()

        return self._build_full_entity_segment_episode_nodes()

    def _add_direct_entity_scene_presence_edges(self) -> None:
        segment_groups: dict[tuple[str, str], list[EntitySegment]] = defaultdict(list)
        for segment in self._entity_segments:
            if not segment.scene_id:
                continue
            canonical_entity_id = self._canonical_entity_id(segment.entity_id) or segment.entity_id
            segment_groups[(canonical_entity_id, segment.scene_id)].append(segment)

        for (canonical_entity_id, scene_id), segments in sorted(segment_groups.items()):
            entity_node_id = self._entity_node_ids.get(canonical_entity_id)
            scene_node_id = self._scene_node_ids.get(scene_id)
            if entity_node_id is None or scene_node_id is None:
                continue
            ordered = sorted(segments, key=lambda item: (item.start_frame, item.segment_id))
            time_range = _time_range(
                start_frame=min(segment.start_frame for segment in ordered),
                end_frame=max(segment.end_frame for segment in ordered),
                start_ts=min(segment.start_ts for segment in ordered),
                end_ts=max(segment.end_ts for segment in ordered),
                support_frames=sum(segment.support_frames for segment in ordered),
            )
            self._add_edge(
                edge_type="IN_SCENE",
                source_node_id=entity_node_id,
                target_node_id=scene_node_id,
                suffix=f"direct_entity_presence:{canonical_entity_id}:{scene_id}",
                time_range=time_range,
                attributes={
                    "role": "direct_entity_scene_presence",
                    "source_segment_count": len(ordered),
                    "source_segment_id_samples": [
                        segment.segment_id for segment in ordered[:16]
                    ],
                    "source_entity_ids": sorted(
                        {segment.entity_id for segment in ordered}
                    ),
                    "canonical_entity_id": canonical_entity_id,
                    "scene_id": scene_id,
                    "time_axis_preserved": True,
                },
            )

    def _build_full_entity_segment_episode_nodes(self) -> list[CanonicalEpisodeNode]:
        episode_nodes: list[CanonicalEpisodeNode] = []
        chain_by_entity: dict[str, list[str]] = defaultdict(list)

        for segment in self._entity_segments:
            canonical_entity_id = self._canonical_entity_id(segment.entity_id) or segment.entity_id
            node_id = f"episode:entity_segment:{segment.segment_id}"
            time_range = _time_range(
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                start_ts=segment.start_ts,
                end_ts=segment.end_ts,
                support_frames=segment.support_frames,
            )
            self._entity_segment_episode_node_ids[segment.segment_id] = node_id
            self._episode_time_ranges[node_id] = time_range
            episode_nodes.append(
                CanonicalEpisodeNode(
                    node_id=node_id,
                    video_id=segment.video_id,
                    episode_id=segment.segment_id,
                    episode_kind="ENTITY_SEGMENT",
                    label=f"entity segment {canonical_entity_id}",
                    time_range=time_range,
                    scene_id=segment.scene_id,
                    primary_entity_id=canonical_entity_id,
                    participant_entity_ids=[canonical_entity_id],
                    segment_index=segment.segment_index,
                    segment_count=segment.segment_count,
                    source_ref_id=segment.segment_id,
                    attributes={
                        "segment_type": segment.segment_type,
                        "resumes_after_gap": segment.resumes_after_gap,
                        "previous_segment_id": segment.previous_segment_id,
                        "next_segment_id": segment.next_segment_id,
                        "source_entity_id": segment.entity_id,
                        "canonical_entity_id": canonical_entity_id,
                    },
                )
            )
            chain_by_entity[canonical_entity_id].append(node_id)

            entity_node_id = self._entity_node_ids.get(canonical_entity_id)
            if entity_node_id is not None:
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=entity_node_id,
                    target_node_id=node_id,
                    suffix="entity_segment",
                    time_range=time_range,
                    attributes={"role": "segment"},
                )
            if segment.scene_id and segment.scene_id in self._scene_node_ids:
                scene_node_id = self._scene_node_ids[segment.scene_id]
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=scene_node_id,
                    suffix="entity_segment",
                    time_range=time_range,
                )
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=scene_node_id,
                    target_node_id=node_id,
                    suffix=f"entity_segment:{segment.segment_id}",
                    time_range=time_range,
                    attributes={"scope": "scene"},
                )
                scene_segment_episode_node_id = self._scene_segment_episode_node_ids.get(segment.scene_id)
                if scene_segment_episode_node_id is not None:
                    self._add_edge(
                        edge_type="BELONGS_TO_SEGMENT",
                        source_node_id=node_id,
                        target_node_id=scene_segment_episode_node_id,
                        suffix="scene_segment",
                        time_range=time_range,
                    )
            if segment.scene_id and entity_node_id is not None and segment.scene_id in self._scene_node_ids:
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=entity_node_id,
                    target_node_id=self._scene_node_ids[segment.scene_id],
                    suffix=f"entity_segment:{segment.segment_id}",
                    time_range=time_range,
                    attributes={"source_segment_id": segment.segment_id},
                )

        for entity_id, node_ids in chain_by_entity.items():
            self._add_temporal_chain(
                node_ids,
                suffix_prefix=f"entity_segment_chain:{entity_id}",
            )
        return episode_nodes

    def _build_compact_entity_segment_episode_nodes(self) -> list[CanonicalEpisodeNode]:
        episode_nodes: list[CanonicalEpisodeNode] = []
        compact_groups: dict[tuple[str | None, str, str], list[EntitySegment]] = defaultdict(list)
        passthrough_segments: list[EntitySegment] = []

        for segment in self._entity_segments:
            canonical_entity_id = self._canonical_entity_id(segment.entity_id) or segment.entity_id
            if canonical_entity_id == segment.entity_id:
                passthrough_segments.append(segment)
                continue
            compact_groups[
                (segment.scene_id, canonical_entity_id, segment.segment_type)
            ].append(segment)

        passthrough_chain_by_entity: dict[str, list[str]] = defaultdict(list)
        for segment in passthrough_segments:
            built_nodes = self._build_full_entity_segment_episode_nodes_for_segment(segment)
            episode_nodes.extend(built_nodes)
            canonical_entity_id = self._canonical_entity_id(segment.entity_id) or segment.entity_id
            passthrough_chain_by_entity[canonical_entity_id].extend(
                node.node_id for node in built_nodes
            )

        chain_by_entity: dict[str, list[str]] = defaultdict(list)
        for key, segments in sorted(
            compact_groups.items(),
            key=lambda item: (
                item[0][0] or "",
                item[0][1],
                item[0][2],
            ),
        ):
            scene_id, canonical_entity_id, segment_type = key
            segments = sorted(segments, key=lambda item: (item.start_frame, item.segment_id))
            start_frame = min(segment.start_frame for segment in segments)
            end_frame = max(segment.end_frame for segment in segments)
            start_ts = min(segment.start_ts for segment in segments)
            end_ts = max(segment.end_ts for segment in segments)
            support_frames = sum(segment.support_frames for segment in segments)
            compact_segment_id = (
                f"{canonical_entity_id}:segment:"
                f"{_stable_token(scene_id or 'scene_unknown')}:{start_frame:06d}:{end_frame:06d}"
            )
            node_id = f"episode:entity_segment:{compact_segment_id}"
            time_range = _time_range(
                start_frame=start_frame,
                end_frame=end_frame,
                start_ts=start_ts,
                end_ts=end_ts,
                support_frames=support_frames,
            )
            self._compact_segment_node_id_by_key[key] = node_id
            for segment in segments:
                self._entity_segment_episode_node_ids[segment.segment_id] = node_id
            self._episode_time_ranges[node_id] = time_range
            episode_nodes.append(
                CanonicalEpisodeNode(
                    node_id=node_id,
                    video_id=segments[0].video_id,
                    episode_id=compact_segment_id,
                    episode_kind="ENTITY_SEGMENT",
                    label=f"entity segment {canonical_entity_id}",
                    time_range=time_range,
                    scene_id=scene_id,
                    primary_entity_id=canonical_entity_id,
                    participant_entity_ids=[canonical_entity_id],
                    segment_index=1,
                    segment_count=1,
                    source_ref_id=compact_segment_id,
                    attributes={
                        "segment_type": segment_type,
                        "compaction_kind": "scene_label_entity_segment_anchor",
                        "source_segment_ids": [segment.segment_id for segment in segments],
                        "source_entity_ids": sorted({segment.entity_id for segment in segments}),
                        "source_segment_count": len(segments),
                    },
                )
            )
            chain_by_entity[canonical_entity_id].append(node_id)

            entity_node_id = self._entity_node_ids.get(canonical_entity_id)
            if entity_node_id is not None:
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=entity_node_id,
                    target_node_id=node_id,
                    suffix="entity_segment",
                    time_range=time_range,
                    attributes={"role": "segment", "compacted": True},
                )
            if scene_id and scene_id in self._scene_node_ids:
                scene_node_id = self._scene_node_ids[scene_id]
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=scene_node_id,
                    suffix="entity_segment",
                    time_range=time_range,
                )
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=scene_node_id,
                    target_node_id=node_id,
                    suffix=f"entity_segment:{compact_segment_id}",
                    time_range=time_range,
                    attributes={"scope": "scene", "compacted": True},
                )
                scene_segment_episode_node_id = self._scene_segment_episode_node_ids.get(scene_id)
                if scene_segment_episode_node_id is not None:
                    self._add_edge(
                        edge_type="BELONGS_TO_SEGMENT",
                        source_node_id=node_id,
                        target_node_id=scene_segment_episode_node_id,
                        suffix="scene_segment",
                        time_range=time_range,
                    )
            if scene_id and entity_node_id is not None and scene_id in self._scene_node_ids:
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=entity_node_id,
                    target_node_id=self._scene_node_ids[scene_id],
                    suffix=f"entity_segment:{compact_segment_id}",
                    time_range=time_range,
                    attributes={"source_segment_id": compact_segment_id, "compacted": True},
                )

        for entity_id, node_ids in chain_by_entity.items():
            self._add_temporal_chain(
                node_ids,
                suffix_prefix=f"entity_segment_chain:{entity_id}",
            )
        for entity_id, node_ids in passthrough_chain_by_entity.items():
            self._add_temporal_chain(
                node_ids,
                suffix_prefix=f"entity_segment_chain:{entity_id}",
            )
        return episode_nodes

    def _build_full_entity_segment_episode_nodes_for_segment(
        self,
        segment: EntitySegment,
    ) -> list[CanonicalEpisodeNode]:
        node_id = f"episode:entity_segment:{segment.segment_id}"
        canonical_entity_id = self._canonical_entity_id(segment.entity_id) or segment.entity_id
        time_range = _time_range(
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            start_ts=segment.start_ts,
            end_ts=segment.end_ts,
            support_frames=segment.support_frames,
        )
        self._entity_segment_episode_node_ids[segment.segment_id] = node_id
        self._episode_time_ranges[node_id] = time_range
        node = CanonicalEpisodeNode(
            node_id=node_id,
            video_id=segment.video_id,
            episode_id=segment.segment_id,
            episode_kind="ENTITY_SEGMENT",
            label=f"entity segment {canonical_entity_id}",
            time_range=time_range,
            scene_id=segment.scene_id,
            primary_entity_id=canonical_entity_id,
            participant_entity_ids=[canonical_entity_id],
            segment_index=segment.segment_index,
            segment_count=segment.segment_count,
            source_ref_id=segment.segment_id,
            attributes={
                "segment_type": segment.segment_type,
                "resumes_after_gap": segment.resumes_after_gap,
                "previous_segment_id": segment.previous_segment_id,
                "next_segment_id": segment.next_segment_id,
                "source_entity_id": segment.entity_id,
                "canonical_entity_id": canonical_entity_id,
            },
        )

        entity_node_id = self._entity_node_ids.get(canonical_entity_id)
        if entity_node_id is not None:
            self._add_edge(
                edge_type="HAS_EPISODE",
                source_node_id=entity_node_id,
                target_node_id=node_id,
                suffix="entity_segment",
                time_range=time_range,
                attributes={"role": "segment"},
            )
        if segment.scene_id and segment.scene_id in self._scene_node_ids:
            scene_node_id = self._scene_node_ids[segment.scene_id]
            self._add_edge(
                edge_type="IN_SCENE",
                source_node_id=node_id,
                target_node_id=scene_node_id,
                suffix="entity_segment",
                time_range=time_range,
            )
            self._add_edge(
                edge_type="HAS_EPISODE",
                source_node_id=scene_node_id,
                target_node_id=node_id,
                suffix=f"entity_segment:{segment.segment_id}",
                time_range=time_range,
                attributes={"scope": "scene"},
            )
            scene_segment_episode_node_id = self._scene_segment_episode_node_ids.get(segment.scene_id)
            if scene_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=scene_segment_episode_node_id,
                    suffix="scene_segment",
                    time_range=time_range,
                )
        if segment.scene_id and entity_node_id is not None and segment.scene_id in self._scene_node_ids:
            self._add_edge(
                edge_type="IN_SCENE",
                source_node_id=entity_node_id,
                target_node_id=self._scene_node_ids[segment.scene_id],
                suffix=f"entity_segment:{segment.segment_id}",
                time_range=time_range,
                attributes={"source_segment_id": segment.segment_id},
            )
        return [node]

    def _build_state_nodes_and_episodes(
        self,
    ) -> tuple[list[CanonicalStateValueNode], list[CanonicalEpisodeNode]]:
        state_value_nodes: list[CanonicalStateValueNode] = []
        episode_nodes: list[CanonicalEpisodeNode] = []
        chain_by_entity_and_attr: dict[tuple[str, str], list[str]] = defaultdict(list)

        for episode in sorted(
            self.document.state_episodes,
            key=lambda item: (
                item.entity_id,
                item.attr_key,
                item.start_frame,
                item.end_frame,
                item.state_episode_id,
            ),
        ):
            canonical_entity_id = self._canonical_entity_id(episode.entity_id) or episode.entity_id
            state_key = (episode.attr_key, episode.attr_value)
            state_value_node_id = self._state_value_node_ids.get(state_key)
            if state_value_node_id is None:
                state_value_node_id = (
                    f"state_value:{episode.attr_key}:{_stable_token(episode.attr_value)}"
                )
                self._state_value_node_ids[state_key] = state_value_node_id
                state_value_nodes.append(
                    CanonicalStateValueNode(
                        node_id=state_value_node_id,
                        video_id=episode.video_id,
                        attr_key=episode.attr_key,
                        attr_value=episode.attr_value,
                        label=f"{episode.attr_key}={episode.attr_value}",
                    )
                )

            scene_id = self._scene_id_for_range(
                episode.start_frame,
                episode.end_frame,
            )
            node_id = f"episode:state:{episode.state_episode_id}"
            time_range = _time_range(
                start_frame=episode.start_frame,
                end_frame=episode.end_frame,
                start_ts=episode.start_ts,
                end_ts=episode.end_ts,
                support_frames=episode.support_frames,
            )
            self._state_episode_node_ids[episode.state_episode_id] = node_id
            self._episode_time_ranges[node_id] = time_range
            episode_nodes.append(
                CanonicalEpisodeNode(
                    node_id=node_id,
                    video_id=episode.video_id,
                    episode_id=episode.state_episode_id,
                    episode_kind="STATE",
                    label=f"{canonical_entity_id} {episode.attr_key}={episode.attr_value}",
                    time_range=time_range,
                    scene_id=scene_id,
                    primary_entity_id=canonical_entity_id,
                    participant_entity_ids=[canonical_entity_id],
                    attr_key=episode.attr_key,
                    attr_value=episode.attr_value,
                    source_ref_id=episode.state_episode_id,
                    attributes={
                        "source_entity_id": episode.entity_id,
                        "canonical_entity_id": canonical_entity_id,
                    },
                )
            )
            chain_by_entity_and_attr[(canonical_entity_id, episode.attr_key)].append(node_id)

            entity_node_id = self._entity_node_ids.get(canonical_entity_id)
            if entity_node_id is not None:
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=entity_node_id,
                    target_node_id=node_id,
                    suffix="state_episode",
                    time_range=time_range,
                    attributes={"role": "state"},
                )
            self._add_edge(
                edge_type="HAS_STATE",
                source_node_id=node_id,
                target_node_id=state_value_node_id,
                suffix="state_value",
                time_range=time_range,
            )
            if scene_id and scene_id in self._scene_node_ids:
                scene_node_id = self._scene_node_ids[scene_id]
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=scene_node_id,
                    suffix="state_episode",
                    time_range=time_range,
                )
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=scene_node_id,
                    target_node_id=node_id,
                    suffix=f"state_episode:{episode.state_episode_id}",
                    time_range=time_range,
                    attributes={"scope": "scene"},
                )

            entity_segment_episode_node_id = self._best_entity_segment_episode_node_id(
                episode.entity_id,
                episode.start_frame,
                episode.end_frame,
            )
            if entity_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=entity_segment_episode_node_id,
                    suffix="entity_segment",
                    time_range=time_range,
                )
            scene_segment_episode_node_id = self._best_scene_segment_episode_node_id(
                episode.start_frame,
                episode.end_frame,
            )
            if scene_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=scene_segment_episode_node_id,
                    suffix="scene_segment",
                    time_range=time_range,
                )

        for chain_key, node_ids in chain_by_entity_and_attr.items():
            entity_id, attr_key = chain_key
            self._add_temporal_chain(
                node_ids,
                suffix_prefix=f"state_chain:{entity_id}:{attr_key}",
            )
        return state_value_nodes, episode_nodes

    def _build_relation_nodes_and_episodes(
        self,
    ) -> tuple[list[CanonicalInteractionTypeNode], list[CanonicalEpisodeNode]]:
        interaction_type_nodes: list[CanonicalInteractionTypeNode] = []
        episode_nodes: list[CanonicalEpisodeNode] = []
        raw_predicates_by_predicate: dict[str, set[str]] = defaultdict(set)
        chain_by_relation: dict[tuple[str, str, str], list[str]] = defaultdict(list)

        for episode in sorted(
            self.document.relation_episodes,
            key=lambda item: (
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.start_frame,
                item.end_frame,
                item.relation_episode_id,
            ),
        ):
            if episode.raw_predicate:
                raw_predicates_by_predicate[episode.predicate].add(episode.raw_predicate)
            canonical_subject_entity_id = (
                self._canonical_entity_id(episode.subject_entity_id)
                or episode.subject_entity_id
            )
            canonical_object_entity_id = (
                self._canonical_entity_id(episode.object_entity_id)
                or episode.object_entity_id
            )
            interaction_type_node_id: str | None = None
            if self.include_interaction_type_nodes:
                interaction_type_node_id = self._interaction_type_node_ids.get(
                    episode.predicate
                )
                if interaction_type_node_id is None:
                    interaction_type_node_id = (
                        f"interaction_type:{_stable_token(episode.predicate)}"
                    )
                    self._interaction_type_node_ids[episode.predicate] = (
                        interaction_type_node_id
                    )
                    interaction_type_nodes.append(
                        CanonicalInteractionTypeNode(
                            node_id=interaction_type_node_id,
                            video_id=episode.video_id,
                            predicate=episode.predicate,
                            label=episode.predicate,
                            raw_predicates_seen=[],
                        )
                    )

            scene_id = self._scene_id_for_range(
                episode.start_frame,
                episode.end_frame,
            )
            node_id = f"episode:relation:{episode.relation_episode_id}"
            time_range = _time_range(
                start_frame=episode.start_frame,
                end_frame=episode.end_frame,
                start_ts=episode.start_ts,
                end_ts=episode.end_ts,
                support_frames=episode.support_frames,
            )
            self._relation_episode_node_ids[episode.relation_episode_id] = node_id
            self._episode_time_ranges[node_id] = time_range
            episode_nodes.append(
                CanonicalEpisodeNode(
                    node_id=node_id,
                    video_id=episode.video_id,
                    episode_id=episode.relation_episode_id,
                    episode_kind="RELATION",
                    label=(
                        f"{canonical_subject_entity_id} {episode.predicate} "
                        f"{canonical_object_entity_id}"
                    ),
                    time_range=time_range,
                    scene_id=scene_id,
                    primary_entity_id=canonical_subject_entity_id,
                    subject_entity_id=canonical_subject_entity_id,
                    object_entity_id=canonical_object_entity_id,
                    participant_entity_ids=[
                        canonical_subject_entity_id,
                        canonical_object_entity_id,
                    ],
                    predicate=episode.predicate,
                    raw_predicate=episode.raw_predicate,
                    source_ref_id=episode.relation_episode_id,
                    attributes={
                        "source_subject_entity_id": episode.subject_entity_id,
                        "source_object_entity_id": episode.object_entity_id,
                        "canonical_subject_entity_id": canonical_subject_entity_id,
                        "canonical_object_entity_id": canonical_object_entity_id,
                    },
                )
            )
            chain_by_relation[
                (
                    canonical_subject_entity_id,
                    episode.predicate,
                    canonical_object_entity_id,
                )
            ].append(node_id)

            subject_node_id = self._entity_node_ids.get(canonical_subject_entity_id)
            object_node_id = self._entity_node_ids.get(canonical_object_entity_id)
            if subject_node_id is not None:
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=subject_node_id,
                    target_node_id=node_id,
                    suffix="relation_subject",
                    time_range=time_range,
                    attributes={"role": "subject"},
                )
            if object_node_id is not None:
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=object_node_id,
                    target_node_id=node_id,
                    suffix="relation_object",
                    time_range=time_range,
                    attributes={"role": "object"},
                )
            if scene_id and scene_id in self._scene_node_ids:
                scene_node_id = self._scene_node_ids[scene_id]
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=scene_node_id,
                    suffix="relation_episode",
                    time_range=time_range,
                )
                self._add_edge(
                    edge_type="HAS_EPISODE",
                    source_node_id=scene_node_id,
                    target_node_id=node_id,
                    suffix=f"relation_episode:{episode.relation_episode_id}",
                    time_range=time_range,
                    attributes={"scope": "scene"},
                )

            if interaction_type_node_id is not None:
                self._add_edge(
                    edge_type="HAS_INTERACTION_TYPE",
                    source_node_id=node_id,
                    target_node_id=interaction_type_node_id,
                    suffix="interaction_type",
                    time_range=time_range,
                )
            if subject_node_id is not None and object_node_id is not None:
                self._add_edge(
                    edge_type="INTERACTS_WITH",
                    source_node_id=subject_node_id,
                    target_node_id=object_node_id,
                    suffix=episode.relation_episode_id,
                    time_range=time_range,
                    attributes={
                        "episode_node_id": node_id,
                        "predicate": episode.predicate,
                        "raw_predicate": episode.raw_predicate,
                    },
                )

            subject_segment_episode_node_id = self._best_entity_segment_episode_node_id(
                episode.subject_entity_id,
                episode.start_frame,
                episode.end_frame,
            )
            if subject_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=subject_segment_episode_node_id,
                    suffix="subject_segment",
                    time_range=time_range,
                    attributes={"role": "subject"},
                )
            object_segment_episode_node_id = self._best_entity_segment_episode_node_id(
                episode.object_entity_id,
                episode.start_frame,
                episode.end_frame,
            )
            if object_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=object_segment_episode_node_id,
                    suffix="object_segment",
                    time_range=time_range,
                    attributes={"role": "object"},
                )
            scene_segment_episode_node_id = self._best_scene_segment_episode_node_id(
                episode.start_frame,
                episode.end_frame,
            )
            if scene_segment_episode_node_id is not None:
                self._add_edge(
                    edge_type="BELONGS_TO_SEGMENT",
                    source_node_id=node_id,
                    target_node_id=scene_segment_episode_node_id,
                    suffix="scene_segment",
                    time_range=time_range,
                )

        raw_predicates_lookup = {
            node.predicate: sorted(raw_predicates_by_predicate.get(node.predicate, set()))
            for node in interaction_type_nodes
        }
        interaction_type_nodes = [
            node.model_copy(
                update={
                    "raw_predicates_seen": raw_predicates_lookup.get(node.predicate, []),
                }
            )
            for node in interaction_type_nodes
        ]

        for relation_key, node_ids in chain_by_relation.items():
            subject_entity_id, predicate, object_entity_id = relation_key
            self._add_temporal_chain(
                node_ids,
                suffix_prefix=(
                    f"relation_chain:{subject_entity_id}:{predicate}:{object_entity_id}"
                ),
            )
        return interaction_type_nodes, episode_nodes

    def _build_transition_event_nodes(self) -> list[CanonicalTransitionEventNode]:
        event_nodes: list[CanonicalTransitionEventNode] = []
        for event in sorted(
            self.document.transition_events,
            key=lambda item: (
                item.frame_index,
                item.timestamp_sec,
                item.event_id,
            ),
        ):
            scene_id = self._scene_id_for_range(
                event.frame_index,
                event.frame_index,
            )
            node_id = f"event:{event.event_id}"
            time_range = _time_range(
                start_frame=event.frame_index,
                end_frame=event.frame_index,
                start_ts=event.timestamp_sec,
                end_ts=event.timestamp_sec,
                support_frames=1,
            )
            canonical_entity_id = self._canonical_entity_id(event.entity_id)
            canonical_subject_entity_id = self._canonical_entity_id(event.subject_entity_id)
            canonical_object_entity_id = self._canonical_entity_id(event.object_entity_id)
            if (
                self.compact_relation_observed_events
                and event.event_type == "RELATION_OBSERVED"
                and event.subject_entity_id
                and event.predicate
                and event.object_entity_id
            ):
                self._relation_observed_samples.append(
                    CanonicalRelationObservedSample(
                        sample_id=node_id,
                        video_id=event.video_id,
                        event_id=event.event_id,
                        label=_event_label(event),
                        time_range=time_range,
                        scene_id=scene_id,
                        subject_entity_id=canonical_subject_entity_id or event.subject_entity_id,
                        predicate=event.predicate,
                        raw_predicate=event.raw_predicate,
                        object_entity_id=canonical_object_entity_id or event.object_entity_id,
                        source_ref_id=event.event_id,
                    )
                )
                continue

            event_nodes.append(
                CanonicalTransitionEventNode(
                    node_id=node_id,
                    video_id=event.video_id,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    label=_event_label(event),
                    time_range=time_range,
                    scene_id=scene_id,
                    entity_id=canonical_entity_id or event.entity_id,
                    subject_entity_id=canonical_subject_entity_id or event.subject_entity_id,
                    object_entity_id=canonical_object_entity_id or event.object_entity_id,
                    predicate=event.predicate,
                    raw_predicate=event.raw_predicate,
                    attr_key=event.attr_key,
                    previous_state_value=event.previous_state_value,
                    current_state_value=event.current_state_value,
                    source_ref_id=event.event_id,
                )
            )

            if scene_id and scene_id in self._scene_node_ids:
                self._add_edge(
                    edge_type="IN_SCENE",
                    source_node_id=node_id,
                    target_node_id=self._scene_node_ids[scene_id],
                    suffix="transition_event",
                    time_range=time_range,
                )

            source_episode_node_id = None
            if event.source_episode_id:
                source_episode_node_id = (
                    self._state_episode_node_ids.get(event.source_episode_id)
                    or self._relation_episode_node_ids.get(event.source_episode_id)
                )
            if source_episode_node_id is not None:
                self._add_edge(
                    edge_type="TRIGGERED_BY",
                    source_node_id=source_episode_node_id,
                    target_node_id=node_id,
                    suffix="source_episode",
                    time_range=time_range,
                )
            if event.source_segment_id and event.source_segment_id in self._entity_segment_episode_node_ids:
                self._add_edge(
                    edge_type="TRIGGERED_BY",
                    source_node_id=self._entity_segment_episode_node_ids[event.source_segment_id],
                    target_node_id=node_id,
                    suffix="source_segment",
                    time_range=time_range,
                )
        return event_nodes

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
        return best_scene.scene_id if best_scene is not None else None

    def _best_scene_segment_episode_node_id(
        self,
        start_frame: int,
        end_frame: int,
    ) -> str | None:
        scene_id = self._scene_id_for_range(start_frame, end_frame)
        if scene_id is None:
            return None
        return self._scene_segment_episode_node_ids.get(scene_id)

    def _best_entity_segment_episode_node_id(
        self,
        entity_id: str,
        start_frame: int,
        end_frame: int,
    ) -> str | None:
        best_segment: EntitySegment | None = None
        best_overlap = -1
        for segment in self._entity_segments:
            if segment.entity_id != entity_id:
                continue
            overlap = min(end_frame, segment.end_frame) - max(start_frame, segment.start_frame) + 1
            if overlap <= 0:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = segment
        if best_segment is None:
            return None
        return self._entity_segment_episode_node_ids.get(best_segment.segment_id)

    def _add_temporal_chain(
        self,
        node_ids: list[str],
        *,
        suffix_prefix: str,
    ) -> None:
        for index in range(len(node_ids) - 1):
            current_node_id = node_ids[index]
            next_node_id = node_ids[index + 1]
            time_range = self._episode_time_ranges.get(next_node_id)
            self._add_edge(
                edge_type="NEXT_EPISODE",
                source_node_id=current_node_id,
                target_node_id=next_node_id,
                suffix=f"{suffix_prefix}:{index}",
                time_range=time_range,
            )
            if not self.compact_previous_episode_edges:
                self._add_edge(
                    edge_type="PREVIOUS_EPISODE",
                    source_node_id=next_node_id,
                    target_node_id=current_node_id,
                    suffix=f"{suffix_prefix}:{index}",
                    time_range=self._episode_time_ranges.get(current_node_id),
                )

    def _add_temporal_semantic_edges(
        self,
        *,
        episode_nodes: list[CanonicalEpisodeNode],
        transition_event_nodes: list[CanonicalTransitionEventNode],
    ) -> None:
        self._add_event_temporal_semantic_edges(
            episode_nodes=episode_nodes,
            transition_event_nodes=transition_event_nodes,
        )
        self._add_episode_temporal_semantic_edges(episode_nodes)

    def _add_event_temporal_semantic_edges(
        self,
        *,
        episode_nodes: list[CanonicalEpisodeNode],
        transition_event_nodes: list[CanonicalTransitionEventNode],
    ) -> None:
        candidate_episodes = [
            node
            for node in episode_nodes
            if node.episode_kind in _TEMPORAL_EVENT_EPISODE_KINDS
        ]
        for event_node in transition_event_nodes:
            if event_node.event_type == "RELATION_OBSERVED":
                continue
            event_frame = event_node.time_range.start_frame
            if event_frame is None:
                continue
            event_entities = _transition_event_entity_ids(event_node)
            if not event_entities:
                continue
            for episode_node in candidate_episodes:
                if not _same_known_scene(episode_node.scene_id, event_node.scene_id):
                    continue
                shared_entities = _episode_entity_ids(episode_node) & event_entities
                if not shared_entities:
                    continue
                start_frame = episode_node.time_range.start_frame
                end_frame = episode_node.time_range.end_frame
                if start_frame is None or end_frame is None:
                    continue
                is_start_boundary = event_frame == start_frame
                is_end_boundary = event_frame == end_frame
                if is_start_boundary and event_node.event_type not in _END_EVENT_TYPES:
                    self._add_edge(
                        edge_type="STARTS_AT",
                        source_node_id=episode_node.node_id,
                        target_node_id=event_node.node_id,
                        suffix="temporal_semantic:start_boundary",
                        time_range=event_node.time_range,
                        attributes={
                            "temporal_relation": "starts_at",
                            "event_type": event_node.event_type,
                            "shared_entity_ids": sorted(shared_entities),
                        },
                    )
                if is_end_boundary and event_node.event_type not in _START_EVENT_TYPES:
                    self._add_edge(
                        edge_type="ENDS_AT",
                        source_node_id=episode_node.node_id,
                        target_node_id=event_node.node_id,
                        suffix="temporal_semantic:end_boundary",
                        time_range=event_node.time_range,
                        attributes={
                            "temporal_relation": "ends_at",
                            "event_type": event_node.event_type,
                            "shared_entity_ids": sorted(shared_entities),
                        },
                    )
                if (
                    episode_node.episode_kind in _TEMPORAL_SEMANTIC_EPISODE_KINDS
                    and start_frame < event_frame < end_frame
                    and not self.compact_derived_temporal_edges
                ):
                    self._add_edge(
                        edge_type="OVERLAPS_WITH",
                        source_node_id=episode_node.node_id,
                        target_node_id=event_node.node_id,
                        suffix="temporal_semantic:event_inside_interval",
                        time_range=event_node.time_range,
                        attributes={
                            "temporal_relation": "event_inside_interval",
                            "event_type": event_node.event_type,
                            "shared_entity_ids": sorted(shared_entities),
                        },
                    )

    def _add_episode_temporal_semantic_edges(
        self,
        episode_nodes: list[CanonicalEpisodeNode],
    ) -> None:
        candidate_episodes = [
            node
            for node in episode_nodes
            if node.episode_kind in _TEMPORAL_SEMANTIC_EPISODE_KINDS
        ]
        for left_index, left in enumerate(candidate_episodes):
            for right in candidate_episodes[left_index + 1:]:
                if left.episode_kind == right.episode_kind:
                    continue
                if not _same_known_scene(left.scene_id, right.scene_id):
                    continue
                shared_entities = _episode_entity_ids(left) & _episode_entity_ids(right)
                if not shared_entities:
                    continue
                overlap = _overlap_time_range(left.time_range, right.time_range)
                if overlap is not None:
                    if not self.compact_derived_temporal_edges:
                        source_node, target_node = _ordered_temporal_pair(left, right)
                        self._add_edge(
                            edge_type="OVERLAPS_WITH",
                            source_node_id=source_node.node_id,
                            target_node_id=target_node.node_id,
                            suffix="temporal_semantic:episode_overlap",
                            time_range=overlap,
                            attributes={
                                "temporal_relation": "episode_overlap",
                                "shared_entity_ids": sorted(shared_entities),
                            },
                        )
                    continue
                gap = _time_range_gap(left.time_range, right.time_range)
                if gap is None:
                    continue
                gap_frames, gap_ts = gap
                if (
                    gap_frames <= _TEMPORAL_NEAR_MAX_FRAME_GAP
                    or gap_ts <= _TEMPORAL_NEAR_MAX_TS_GAP
                ):
                    if self.compact_derived_temporal_edges:
                        continue
                    source_node, target_node = _ordered_temporal_pair(left, right)
                    self._add_edge(
                        edge_type="TEMPORALLY_NEAR",
                        source_node_id=source_node.node_id,
                        target_node_id=target_node.node_id,
                        suffix="temporal_semantic:episode_near",
                        time_range=_span_time_range(left.time_range, right.time_range),
                        attributes={
                            "temporal_relation": "episode_near",
                            "gap_frames": gap_frames,
                            "gap_ts": round(gap_ts, 6),
                            "shared_entity_ids": sorted(shared_entities),
                        },
                    )

    def _add_edge(
        self,
        *,
        edge_type: CanonicalEdgeType,
        source_node_id: str,
        target_node_id: str,
        suffix: str,
        time_range: CanonicalTimeRange | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        edge_id = (
            f"edge:{edge_type.lower()}:{source_node_id}:{target_node_id}:{suffix}"
        )
        if edge_id in self._edges:
            return
        self._edges[edge_id] = CanonicalGraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            video_id=self.document.video_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            time_range=time_range,
            attributes=attributes or {},
        )


def build_canonical_temporal_memory_graph(
    document: TemporalGraphDocument,
    *,
    source_graph_path: str | Path | None = None,
    include_stable_identities: bool = True,
    compact_relation_observed_events: bool = False,
    compact_derived_temporal_edges: bool = True,
    compact_previous_episode_edges: bool = True,
    compact_scene_label_entities: bool = False,
    compact_scene_label_min_members: int = 4,
    compact_scene_label_entity_segments: bool = True,
    include_scene_segment_episodes: bool = True,
    include_entity_segment_episodes: bool = True,
    include_interaction_type_nodes: bool = True,
    stable_identity_merge_mode: StableIdentityMergeMode = "semantic",
    stable_identity_include_fragments: bool = False,
    stable_identity_include_noise: bool = False,
) -> CanonicalTemporalMemoryGraph:
    converter = TemporalGraphToCanonicalGraphConverter(
        document,
        source_graph_path=source_graph_path,
        include_stable_identities=include_stable_identities,
        compact_relation_observed_events=compact_relation_observed_events,
        compact_derived_temporal_edges=compact_derived_temporal_edges,
        compact_previous_episode_edges=compact_previous_episode_edges,
        compact_scene_label_entities=compact_scene_label_entities,
        compact_scene_label_min_members=compact_scene_label_min_members,
        compact_scene_label_entity_segments=compact_scene_label_entity_segments,
        include_scene_segment_episodes=include_scene_segment_episodes,
        include_entity_segment_episodes=include_entity_segment_episodes,
        include_interaction_type_nodes=include_interaction_type_nodes,
        stable_identity_merge_mode=stable_identity_merge_mode,
        stable_identity_include_fragments=stable_identity_include_fragments,
        stable_identity_include_noise=stable_identity_include_noise,
    )
    return converter.convert()


def _time_range(
    *,
    start_frame: int | None,
    end_frame: int | None,
    start_ts: float | None,
    end_ts: float | None,
    support_frames: int | None,
) -> CanonicalTimeRange:
    return CanonicalTimeRange(
        start_frame=start_frame,
        end_frame=end_frame,
        start_ts=start_ts,
        end_ts=end_ts,
        support_frames=support_frames,
    )


def _episode_entity_ids(node: CanonicalEpisodeNode) -> set[str]:
    return {
        entity_id
        for entity_id in (
            node.primary_entity_id,
            node.subject_entity_id,
            node.object_entity_id,
            *node.participant_entity_ids,
        )
        if entity_id
    }


def _transition_event_entity_ids(node: CanonicalTransitionEventNode) -> set[str]:
    return {
        entity_id
        for entity_id in (
            node.entity_id,
            node.subject_entity_id,
            node.object_entity_id,
        )
        if entity_id
    }


def _merge_unique(
    labels: list[str],
    extra_labels: list[str],
    *,
    limit: int,
) -> list[str]:
    merged: list[str] = []
    for label in [*labels, *extra_labels]:
        normalized = canonical_label(label) or label
        if normalized and normalized not in merged:
            merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _salient_scene_labels(
    scene: SceneSegment,
    *,
    entity_labels: list[str] | None = None,
) -> list[str]:
    labels: list[str] = []
    for label in [
        scene.scene_kind_hint or "",
        *scene.representative_static_labels,
        *scene.representative_dynamic_labels,
        *scene.entry_static_labels,
        *scene.exit_static_labels,
        *(entity_labels or []),
    ]:
        canonical = canonical_label(label)
        if canonical and is_salient_label(canonical) and canonical not in labels:
            labels.append(canonical)
    return labels


def _alias_terms_for_labels(labels: list[str]) -> list[str]:
    terms: set[str] = set()
    for label in labels:
        terms.update(label_alias_terms(label))
    return sorted(term for term in terms if term)


def _dedupe_text_values(values: object) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        deduped.append(text)
        seen.add(key)
    return deduped


def _same_known_scene(left_scene_id: str | None, right_scene_id: str | None) -> bool:
    if left_scene_id is None or right_scene_id is None:
        return True
    return left_scene_id == right_scene_id


def _ordered_temporal_pair(
    left: CanonicalEpisodeNode,
    right: CanonicalEpisodeNode,
) -> tuple[CanonicalEpisodeNode, CanonicalEpisodeNode]:
    left_key = (
        _sort_frame(left.time_range.start_frame),
        _sort_frame(left.time_range.end_frame),
        left.node_id,
    )
    right_key = (
        _sort_frame(right.time_range.start_frame),
        _sort_frame(right.time_range.end_frame),
        right.node_id,
    )
    return (left, right) if left_key <= right_key else (right, left)


def _overlap_time_range(
    left: CanonicalTimeRange,
    right: CanonicalTimeRange,
) -> CanonicalTimeRange | None:
    if (
        left.start_frame is None
        or left.end_frame is None
        or right.start_frame is None
        or right.end_frame is None
    ):
        return None
    start_frame = max(left.start_frame, right.start_frame)
    end_frame = min(left.end_frame, right.end_frame)
    if start_frame > end_frame:
        return None
    return _time_range(
        start_frame=start_frame,
        end_frame=end_frame,
        start_ts=_max_optional(left.start_ts, right.start_ts),
        end_ts=_min_optional(left.end_ts, right.end_ts),
        support_frames=end_frame - start_frame + 1,
    )


def _time_range_gap(
    left: CanonicalTimeRange,
    right: CanonicalTimeRange,
) -> tuple[int, float] | None:
    if (
        left.start_frame is None
        or left.end_frame is None
        or right.start_frame is None
        or right.end_frame is None
    ):
        return None
    if left.end_frame < right.start_frame:
        frame_gap = right.start_frame - left.end_frame
        ts_gap = _optional_gap(left.end_ts, right.start_ts)
        return frame_gap, ts_gap
    if right.end_frame < left.start_frame:
        frame_gap = left.start_frame - right.end_frame
        ts_gap = _optional_gap(right.end_ts, left.start_ts)
        return frame_gap, ts_gap
    return 0, 0.0


def _span_time_range(
    left: CanonicalTimeRange,
    right: CanonicalTimeRange,
) -> CanonicalTimeRange:
    start_frame = _min_optional(left.start_frame, right.start_frame)
    end_frame = _max_optional(left.end_frame, right.end_frame)
    support_frames = (
        end_frame - start_frame + 1
        if start_frame is not None and end_frame is not None
        else None
    )
    return _time_range(
        start_frame=start_frame,
        end_frame=end_frame,
        start_ts=_min_optional(left.start_ts, right.start_ts),
        end_ts=_max_optional(left.end_ts, right.end_ts),
        support_frames=support_frames,
    )


def _sort_frame(value: int | None) -> int:
    return value if value is not None else 10**12


def _min_optional(left: int | float | None, right: int | float | None):
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _max_optional(left: int | float | None, right: int | float | None):
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _optional_gap(left_end: float | None, right_start: float | None) -> float:
    if left_end is None or right_start is None:
        return float("inf")
    return max(0.0, right_start - left_end)


def _stable_token(value: str) -> str:
    lowered = value.strip().lower()
    slug = _TOKEN_PATTERN.sub("_", lowered).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    if not slug:
        slug = "value"
    return f"{slug}-{digest}"


def _event_label(event: TransitionEvent) -> str:
    if event.event_type == "STATE_TRANSITION":
        return (
            f"state transition {event.entity_id} "
            f"{event.previous_state_value} -> {event.current_state_value}"
        )
    if event.event_type in {"RELATION_START", "RELATION_END", "RELATION_OBSERVED"}:
        return (
            f"{event.event_type.lower()} {event.subject_entity_id} "
            f"{event.predicate} {event.object_entity_id}"
        )
    return f"{event.event_type.lower()} {event.entity_id or event.subject_entity_id or 'unknown'}"
