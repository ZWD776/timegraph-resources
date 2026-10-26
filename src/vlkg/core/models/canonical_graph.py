from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CanonicalEpisodeKind = Literal[
    "SCENE_SEGMENT",
    "ENTITY_SEGMENT",
    "STATE",
    "RELATION",
]

CanonicalEdgeType = Literal[
    "IN_SCENE",
    "HAS_STATE",
    "HAS_INTERACTION_TYPE",
    "INTERACTS_WITH",
    "HAS_MEMBER_ENTITY",
    "HAS_EPISODE",
    "NEXT_EPISODE",
    "PREVIOUS_EPISODE",
    "BELONGS_TO_SEGMENT",
    "TRIGGERED_BY",
    "STARTS_AT",
    "ENDS_AT",
    "OVERLAPS_WITH",
    "TEMPORALLY_NEAR",
]


class CanonicalTimeRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_frame: int | None = None
    end_frame: int | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    support_frames: int | None = None


class CanonicalSceneNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    scene_id: str
    label: str
    time_range: CanonicalTimeRange
    scene_index: int = 1
    scene_count: int = 1
    previous_scene_id: str | None = None
    next_scene_id: str | None = None
    representative_static_entity_ids: list[str] = Field(default_factory=list)
    representative_static_labels: list[str] = Field(default_factory=list)
    representative_dynamic_labels: list[str] = Field(default_factory=list)
    entry_static_labels: list[str] = Field(default_factory=list)
    exit_static_labels: list[str] = Field(default_factory=list)
    salient_labels: list[str] = Field(default_factory=list)
    semantic_alias_terms: list[str] = Field(default_factory=list)
    scene_kind_hint: str | None = None
    boundary_from_previous_reason: str | None = None
    boundary_to_next_reason: str | None = None
    boundary_from_previous_entity_similarity: float | None = None
    boundary_to_next_entity_similarity: float | None = None
    boundary_from_previous_label_similarity: float | None = None
    boundary_to_next_label_similarity: float | None = None


class CanonicalEntityNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    entity_id: str
    label: str
    time_range: CanonicalTimeRange
    observation_count: int
    node_role: Literal["static", "dynamic", "dependent", "noise"] = "dynamic"
    is_static: bool = False
    is_noise: bool = False
    is_fragment: bool = False
    stable_identity_id: str | None = None
    stable_identity_display_name: str | None = None
    stable_identity_canonical_label: str | None = None
    labels_seen: list[str] = Field(default_factory=list)
    states_seen: list[str] = Field(default_factory=list)
    parent_labels_seen: list[str] = Field(default_factory=list)
    canonical_image_path: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)


class CanonicalStableIdentityNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    stable_id: str
    identity_type: Literal["human", "ball", "static_object", "object"]
    canonical_label: str
    display_name: str
    time_range: CanonicalTimeRange
    observation_count: int
    member_entity_ids: list[str] = Field(default_factory=list)
    labels_seen: list[str] = Field(default_factory=list)
    signature_tokens: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, object] = Field(default_factory=dict)


class CanonicalStateValueNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    attr_key: str
    attr_value: str
    label: str


class CanonicalInteractionTypeNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    predicate: str
    label: str
    raw_predicates_seen: list[str] = Field(default_factory=list)


class CanonicalEpisodeNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    episode_id: str
    episode_kind: CanonicalEpisodeKind
    label: str
    time_range: CanonicalTimeRange
    scene_id: str | None = None
    primary_entity_id: str | None = None
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    participant_entity_ids: list[str] = Field(default_factory=list)
    attr_key: str | None = None
    attr_value: str | None = None
    predicate: str | None = None
    raw_predicate: str | None = None
    segment_index: int | None = None
    segment_count: int | None = None
    source_ref_id: str
    attributes: dict[str, object] = Field(default_factory=dict)


class CanonicalTransitionEventNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    video_id: str
    event_id: str
    event_type: Literal[
        "APPEAR",
        "DISAPPEAR",
        "STATE_TRANSITION",
        "RELATION_START",
        "RELATION_END",
        "RELATION_OBSERVED",
    ]
    label: str
    time_range: CanonicalTimeRange
    scene_id: str | None = None
    entity_id: str | None = None
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    predicate: str | None = None
    raw_predicate: str | None = None
    attr_key: str | None = None
    previous_state_value: str | None = None
    current_state_value: str | None = None
    source_ref_id: str


class CanonicalRelationObservedSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    video_id: str
    event_id: str
    label: str
    time_range: CanonicalTimeRange
    scene_id: str | None = None
    subject_entity_id: str
    predicate: str
    raw_predicate: str | None = None
    object_entity_id: str
    source_ref_id: str


class CanonicalGraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    edge_type: CanonicalEdgeType
    video_id: str
    source_node_id: str
    target_node_id: str
    time_range: CanonicalTimeRange | None = None
    attributes: dict[str, object] = Field(default_factory=dict)


class CanonicalTemporalMemoryGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "canonical_temporal_memory_graph.v1"
    video_id: str
    source_graph_path: Path | None = None
    scene_nodes: list[CanonicalSceneNode] = Field(default_factory=list)
    entity_nodes: list[CanonicalEntityNode] = Field(default_factory=list)
    stable_identity_nodes: list[CanonicalStableIdentityNode] = Field(default_factory=list)
    state_value_nodes: list[CanonicalStateValueNode] = Field(default_factory=list)
    interaction_type_nodes: list[CanonicalInteractionTypeNode] = Field(default_factory=list)
    episode_nodes: list[CanonicalEpisodeNode] = Field(default_factory=list)
    transition_event_nodes: list[CanonicalTransitionEventNode] = Field(default_factory=list)
    relation_observed_samples: list[CanonicalRelationObservedSample] = Field(default_factory=list)
    edges: list[CanonicalGraphEdge] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        return (
            len(self.scene_nodes)
            + len(self.entity_nodes)
            + len(self.stable_identity_nodes)
            + len(self.state_value_nodes)
            + len(self.interaction_type_nodes)
            + len(self.episode_nodes)
            + len(self.transition_event_nodes)
        )

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def episode_nodes_by_kind(
        self,
        episode_kind: CanonicalEpisodeKind,
    ) -> list[CanonicalEpisodeNode]:
        return [
            node
            for node in self.episode_nodes
            if node.episode_kind == episode_kind
        ]
