from __future__ import annotations

from vlkg.workspace import workspace_root
from vlkg.core.assets.source_paths import resolve_data_path

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from vlkg.core.temporal.association import (
    AssociationConfig,
    AssociationInput,
    AssociationPolicy,
    ParentLabelConstraint,
    labels_compatible,
)
from vlkg.core.config.config import DomainConfig, load_domain_config
from vlkg.core.temporal.decision_log import JSONLDecisionLogger, NullDecisionLogger
from vlkg.core.temporal.llm_fallback import build_llm_fallback_resolver, resolve_decision_log_path
from vlkg.core.temporal.merge import (
    EntityStateRecord,
    build_association_candidates,
    create_entity_state_record,
    update_entity_state_record,
)
from vlkg.core.temporal.node_roles import NodeRoleDecision, NodeRoleEvidence, classify_node_role
from vlkg.core.models.temporal_graph import (
    BoundingBox,
    Entity,
    EntitySegment,
    SceneBaseEdge,
    SceneBaseNode,
    SceneDeltaEdge,
    SceneDeltaNode,
    SceneDeltaObservation,
    Observation,
    RelationEpisode,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
    TransitionEvent,
)
from vlkg.core.temporal.pathing import default_cache_db_path
from vlkg.core.models.scene import build_frame_scene_signatures, detect_scene_ranges
from vlkg.core.config.semantic_labels import canonical_label, is_salient_label, label_alias_terms
from vlkg.core.temporal.tracking import NoOpTrackingProvider, TrackingObservationInput, TrackingProvider
from vlkg.core.indexing.vision import EmbeddingCache, VisionFeatureExtractor


FRAME_NAME_PATTERN = re.compile(
    r"^frame_(?P<frame_idx>\d+)_(?P<timestamp>\d+(?:\.\d+)?)s$"
)
FRAME_NAME_INDEX_ONLY_PATTERN = re.compile(r"^frame_(?P<frame_idx>\d+)$")
CODA_FRAME_NAME_PATTERN = re.compile(
    r"^(?P<sample_idx>\d+)_(?P<frame_idx>\d+)_2d_rect_cam\d+_\d+_(?P<raw_idx>\d+)$"
)
CODA_SAMPLED_FRAME_NAME_PATTERN = re.compile(
    r"^(?P<sample_idx>\d+)_2d_rect_cam\d+_\d+_(?P<raw_idx>\d+)$"
)
REPO_ROOT = workspace_root()
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
UNMAPPED_RELATION = "UNMAPPED_RELATION"
WEARABLE_COMPONENT_LABELS = {
    "clothing",
    "dress",
    "glove",
    "headband",
    "helmet",
    "jersey",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "socks",
    "sleeve",
    "wristband",
}
HUMAN_LIKE_LABELS = {
    "player",
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
}
PLAYER_LIKE_KEYWORDS = {
    "basketball player",
    "player",
    "dribbling",
    "shooting",
    "guarding",
    "passing",
    "rebound",
    "playing basketball",
    "playing with",
    "defend",
    "defending",
    "running",
}
APPEARANCE_ONLY_KEYWORDS = {
    "wearing",
    "dress",
    "shirt",
    "pants",
    "shorts",
    "shoes",
    "sneakers",
    "socks",
    "jersey",
    "outfit",
}
OBJECT_SENSITIVE_STATE_PREFIXES = {
    "holding",
    "dribbling",
    "shooting",
}
STATE_OBJECT_STOPWORDS = {
    "a",
    "an",
    "and",
    "another",
    "at",
    "away",
    "from",
    "in",
    "into",
    "near",
    "of",
    "on",
    "onto",
    "other",
    "the",
    "to",
    "toward",
    "towards",
    "with",
}


@dataclass
class _FrameMeta:
    frame_name: str
    frame_index: int
    timestamp_sec: float


@dataclass
class _FrameObjectInput:
    object_index: int
    bbox: BoundingBox
    label: str
    score: float
    state: str | None
    parent_label: str | None
    is_static: bool
    is_noise: bool
    node_role: str = "dynamic"
    folded_states: list[str] = field(default_factory=list)
    folded_into_host: bool = False
    skip_entity_creation: bool = False

    @property
    def state_values(self) -> list[str]:
        values: list[str] = []
        if self.state:
            values.append(self.state)
        values.extend(self.folded_states)
        deduped: list[str] = []
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped
@dataclass
class _AlignmentStats:
    total_alignments: int = 0
    fast_path_hits: int = 0
    slow_path_hits: int = 0
    fallback_path_hits: int = 0
    reject_path_hits: int = 0
    new_entity_creations: int = 0


@dataclass
class _StateFilterStats:
    candidates_seen: int = 0
    kept_candidates: int = 0
    rejected_candidates: int = 0
    rejected_empty_candidates: int = 0
    kept_exact_allowlist: int = 0
    kept_object_insensitive: int = 0
    kept_object_free: int = 0
    kept_object_context_supported: int = 0
    kept_object_context_unchecked: int = 0
    rejected_object_context_unsupported: int = 0


def _merge_scene_representative_labels(
    labels: list[str],
    salient_labels: list[str],
    *,
    limit: int,
) -> list[str]:
    merged: list[str] = []
    for label in [*labels, *salient_labels]:
        normalized = canonical_label(label) or label
        if normalized and normalized not in merged:
            merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


class JSONToGraphTransformer:
    def __init__(
        self,
        input_dir: str | Path,
        *,
        video_id: str | None = None,
        config: DomainConfig | None = None,
        frame_time_index_path: str | Path | None = None,
        frame_image_root: str | Path | None = None,
        entity_match_max_frame_gap: int = 2,
        episode_max_frame_gap: int = 1,
        fast_iou_threshold: float = 0.50,
        embedding_similarity_threshold: float = 0.75,
        label_compatibility_groups: list[set[str]] | None = None,
        vision_feature_extractor: VisionFeatureExtractor | None = None,
        embedding_cache: EmbeddingCache | None = None,
        tracking_provider: TrackingProvider | None = None,
        llm_fallback_resolver=None,
        decision_logger=None,
        cache_db_path: str | Path | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.frame_image_root = (
            Path(frame_image_root).resolve() if frame_image_root is not None else None
        )
        self.json_dir = (
            self.input_dir / "json"
            if (self.input_dir / "json").exists()
            else self.input_dir
        )
        self.video_id = video_id or self.input_dir.name
        self.frame_time_index = self._load_frame_time_index(frame_time_index_path)
        self.config = config or load_domain_config(config_dir=DEFAULT_CONFIG_DIR)
        self.entity_match_max_frame_gap = entity_match_max_frame_gap
        self.episode_max_frame_gap = episode_max_frame_gap
        self.fast_iou_threshold = fast_iou_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.static_labels = self.config.static_label_set
        self.hierarchy_static_labels = self.config.hierarchy_static_label_set
        self.dynamic_labels = self.config.dynamic_label_set
        self.noise_labels = self.config.noise_label_set
        self.component_labels = self.config.component_label_set
        self.component_host_labels = self.config.component_host_label_set
        self.component_inside_ratio = self.config.component_inside_ratio
        self.state_subject_labels = self.config.state_subject_label_set
        self.state_prefix_allowlist = self.config.normalized_state_prefix_allowlist
        self.state_exact_allowlist = self.config.normalized_state_exact_allowlist
        self.state_episode_exclude_prefixes = (
            self.config.normalized_state_episode_exclude_prefixes
        )
        self.state_summary_prefixes = sorted(
            set(self.state_prefix_allowlist),
            key=len,
            reverse=True,
        )
        self.state_aliases = self.config.normalized_state_aliases
        self.enable_structured_state_extraction = (
            self.config.enable_structured_state_extraction
        )
        self.min_state_episode_length_frames = max(
            1,
            self.config.min_state_episode_length_frames,
        )
        self.min_state_transition_support_frames = max(
            1,
            self.config.min_state_transition_support_frames,
        )
        self.state_transition_min_dominance_margin_frames = max(
            0,
            self.config.state_transition_min_dominance_margin_frames,
        )
        self.state_transition_min_dominance_share = min(
            1.0,
            max(0.0, self.config.state_transition_min_dominance_share),
        )
        self.state_transition_max_gap_frames = max(
            1,
            self.config.state_transition_max_gap_frames,
        )
        self.relation_subject_labels = self.config.relation_subject_label_set
        self.relation_object_labels = self.config.relation_object_label_set
        self.allow_dynamic_static_relations = self.config.allow_dynamic_static_relations
        self.dynamic_static_relation_predicates = (
            self.config.normalized_dynamic_static_relation_predicates
        )
        self.enable_semantic_relation_assignment_cleanup = (
            self.config.enable_semantic_relation_assignment_cleanup
        )
        self.relation_episode_exclude_predicates = (
            self.config.normalized_relation_episode_exclude_predicates
        )
        self.relation_episode_gap_overrides = (
            self.config.normalized_relation_episode_gap_overrides
        )
        self.relation_episode_min_support_overrides = (
            self.config.normalized_relation_episode_min_support_overrides
        )
        self.min_relation_episode_support_frames = max(
            1,
            self.config.min_relation_episode_support_frames,
        )
        self.min_relation_episode_length_frames = max(
            1,
            self.config.min_relation_episode_length_frames,
        )
        self.min_relation_event_support_frames = max(
            1,
            self.config.min_relation_event_support_frames,
        )
        self.max_entity_absence_tolerance_frames = max(
            1,
            self.config.max_entity_absence_tolerance_frames,
        )
        self.min_entity_segment_event_support_frames = max(
            1,
            self.config.min_entity_segment_event_support_frames,
        )
        self.entity_segment_event_support_overrides = (
            self.config.normalized_entity_segment_event_support_overrides
        )
        self.scene_segmentation_config = self.config.scene_segmentation
        self.scene_base_graph_config = self.config.scene_base_graph
        self.label_aliases = (
            {}
            if self.config.association.identity_role_strategy == "tracklet"
            else self.config.normalized_label_aliases
        )
        self.label_min_scores = self.config.normalized_label_min_scores
        self.label_gap_overrides = (
            self.config.normalized_association_label_gap_overrides
        )
        config_label_compatibility_groups = [
            set(group)
            for group in self.config.normalized_label_compatibility_groups
        ]
        runtime_label_compatibility_groups = [
            {self._normalize_label(item) for item in group}
            for group in (label_compatibility_groups or [])
        ]
        self.label_compatibility_groups = (
            config_label_compatibility_groups
            + runtime_label_compatibility_groups
        )
        self.enable_relation_predicate_mapping = (
            self.config.enable_relation_predicate_mapping
        )
        self.relation_map = self.config.normalized_relation_map
        self.static_fast_iou_threshold = min(self.fast_iou_threshold, 0.15)
        self.static_center_distance_ratio = 0.20
        self.component_overlap_ratio = 0.20
        self.component_nearest_distance_ratio = 0.75
        self.role_static_iou_threshold = 0.65
        self.role_anchor_iou_threshold = 0.50
        self.role_dynamic_iou_upper_bound = 0.35
        self.association_config = AssociationConfig(
            entity_match_max_frame_gap=self.entity_match_max_frame_gap,
            static_entity_match_max_frame_gap=(
                max(1, self.config.association.static_entity_match_max_frame_gap)
                if self.config.association.static_entity_match_max_frame_gap
                is not None
                else None
            ),
            local_tracklet_mode=self.config.association.local_tracklet_mode,
            local_tracklet_max_frame_gap=max(
                1,
                self.config.association.local_tracklet_max_frame_gap,
            ),
            local_tracklet_embedding_similarity_threshold=(
                self.config.association.local_tracklet_embedding_similarity_threshold
            ),
            local_tracklet_min_score=(
                self.config.association.local_tracklet_min_score
            ),
            local_tracklet_score_margin=(
                self.config.association.local_tracklet_score_margin
            ),
            local_tracklet_motion_weight=(
                self.config.association.local_tracklet_motion_weight
            ),
            local_tracklet_embedding_weight=(
                self.config.association.local_tracklet_embedding_weight
            ),
            local_tracklet_iou_weight=(
                self.config.association.local_tracklet_iou_weight
            ),
            local_tracklet_object_index_weight=(
                self.config.association.local_tracklet_object_index_weight
            ),
            continuity_max_center_distance_ratio=(
                self.config.association.continuity_max_center_distance_ratio
            ),
            continuity_min_area_ratio=(
                self.config.association.continuity_min_area_ratio
            ),
            object_index_jump_max_delta=(
                self.config.association.object_index_jump_max_delta
            ),
            object_index_jump_embedding_veto_threshold=(
                self.config.association.object_index_jump_embedding_veto_threshold
            ),
            state_conflict_embedding_veto_threshold=(
                self.config.association.state_conflict_embedding_veto_threshold
            ),
            state_conflict_min_token_overlap_ratio=(
                self.config.association.state_conflict_min_token_overlap_ratio
            ),
            label_conflict_embedding_veto_threshold=(
                self.config.association.label_conflict_embedding_veto_threshold
            ),
            fast_iou_threshold=self.fast_iou_threshold,
            embedding_similarity_threshold=self.embedding_similarity_threshold,
            static_fast_iou_threshold=self.static_fast_iou_threshold,
            static_center_distance_ratio=self.static_center_distance_ratio,
            fast_match_embedding_veto_threshold=(
                self.config.association.fast_match_embedding_veto_threshold
            ),
            static_fast_match_embedding_veto_threshold=(
                self.config.association.static_fast_match_embedding_veto_threshold
            ),
            enable_same_label_dynamic_spatial_gate=(
                self.config.association.enable_same_label_dynamic_spatial_gate
            ),
        )
        decision_log_path = resolve_decision_log_path(
            self.config.llm_fallback.decision_log_path,
            base_dir=Path.cwd(),
        )
        self.decision_logger = decision_logger or (
            JSONLDecisionLogger(decision_log_path)
            if decision_log_path is not None
            else NullDecisionLogger()
        )
        self.llm_fallback_resolver = (
            llm_fallback_resolver
            or build_llm_fallback_resolver(
                self.config.llm_fallback,
                decision_logger=self.decision_logger,
            )
        )
        self.association_policy = AssociationPolicy(
            self.association_config,
            constraints=[
                # Start conservatively; later phases can enable or extend constraints.
                ParentLabelConstraint(
                    enabled=self.config.association.enable_parent_label_constraint
                ),
            ],
            enable_track_hints=self.config.association.enable_track_hints,
            enable_llm_fallback=self.config.llm_fallback.enabled,
            fallback_resolver=self.llm_fallback_resolver,
            fallback_max_candidates=self.config.llm_fallback.max_candidates,
            fallback_min_candidate_count=self.config.llm_fallback.min_candidate_count,
            fallback_ambiguity_margin=self.config.llm_fallback.ambiguity_margin,
            fallback_low_similarity_floor=self.config.llm_fallback.low_similarity_floor,
            fallback_low_similarity_ambiguity_margin=(
                self.config.llm_fallback.low_similarity_ambiguity_margin
            ),
            fallback_single_candidate_score_margin=(
                self.config.llm_fallback.single_candidate_score_margin
            ),
            fallback_enable_static=self.config.llm_fallback.enable_static_fallback,
            fallback_enable_noise=self.config.llm_fallback.enable_noise_fallback,
            fallback_enable_candidate_context=(
                self.config.llm_fallback.enable_candidate_context_fallback
            ),
            fallback_require_second_candidate_for_low_similarity=(
                self.config.llm_fallback.require_second_candidate_for_low_similarity
            ),
            fallback_override_confidence_threshold=(
                self.config.llm_fallback.override_confidence_threshold
            ),
        )
        self.tracking_provider = tracking_provider or NoOpTrackingProvider()
        self.vision_feature_extractor = (
            vision_feature_extractor or VisionFeatureExtractor()
        )
        self._owns_embedding_cache = embedding_cache is None
        self.embedding_cache = embedding_cache or EmbeddingCache(
            cache_db_path or default_cache_db_path()
        )
        self.logger = logging.getLogger(__name__)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        self._reset_runtime_state()

    def transform(self) -> TemporalGraphDocument:
        self._reset_runtime_state()
        started_at = time.perf_counter()
        try:
            self.logger.info("temporal_graph_stage=start input_dir=%s", self.input_dir)
            objects = self._load_json("objects.json")
            attributes = self._load_json("attributes.json")
            relations = self._load_json("relations.json")
            hierarchy = self._load_json("hierarchy_mapping.json")
            self._log_stage(
                "load_json",
                started_at,
                objects_frames=len(objects),
                attributes_frames=len(attributes),
                relations_frames=len(relations),
                hierarchy_frames=len(hierarchy),
            )

            frame_metas = self._sorted_frame_metas(objects.keys())
            self._frame_rank_by_index = {
                meta.frame_index: frame_rank
                for frame_rank, meta in enumerate(frame_metas)
            }
            self._log_stage("parse_frames", started_at, frames=len(frame_metas))
            observations, observation_lookup = self._build_observations(
                frame_metas=frame_metas,
                objects_table=objects,
                attributes_table=attributes,
                hierarchy_table=hierarchy,
            )
            self._log_stage(
                "build_observations",
                started_at,
                observations=len(observations),
                observation_lookup=len(observation_lookup),
            )
            entities = self._build_entities(observations)
            self._log_stage("build_entities", started_at, entities=len(entities))
            state_episodes = self._build_state_episodes(observations)
            self._log_stage(
                "build_state_episodes",
                started_at,
                state_episodes=len(state_episodes),
            )
            relation_episodes, transient_relation_episodes = self._build_relation_episodes(
                frame_metas=frame_metas,
                relations_table=relations,
                observation_lookup=observation_lookup,
            )
            self._log_stage(
                "build_relation_episodes",
                started_at,
                relation_episodes=len(relation_episodes),
                transient_relation_episodes=len(transient_relation_episodes),
            )
            frame_rank_lookup = dict(self._frame_rank_by_index)
            scene_segments = self._build_scene_segments(
                observations=observations,
                entities=entities,
            )
            self._log_stage(
                "build_scene_segments",
                started_at,
                scene_segments=len(scene_segments),
            )
            entity_segments = self._build_entity_segments(
                entities=entities,
                observations=observations,
                scene_segments=scene_segments,
                frame_rank_lookup=frame_rank_lookup,
            )
            self._log_stage(
                "build_entity_segments",
                started_at,
                entity_segments=len(entity_segments),
            )
            scene_base_nodes, scene_base_edges = self._build_scene_base_graph(
                entities=entities,
                observations=observations,
                scene_segments=scene_segments,
            )
            self._log_stage(
                "build_scene_base_graph",
                started_at,
                scene_base_nodes=len(scene_base_nodes),
                scene_base_edges=len(scene_base_edges),
            )
            observations = self._annotate_observation_origins(
                observations=observations,
                scene_segments=scene_segments,
                scene_base_nodes=scene_base_nodes,
            )
            entities = self._annotate_entity_origins(
                entities=entities,
                scene_base_nodes=scene_base_nodes,
            )
            transition_events = self._build_transition_events(
                entities=entities,
                state_episodes=state_episodes,
                relation_episodes=relation_episodes,
                transient_relation_episodes=transient_relation_episodes,
                entity_segments=entity_segments,
                frame_rank_lookup=frame_rank_lookup,
            )
            self._log_stage(
                "build_transition_events",
                started_at,
                transition_events=len(transition_events),
            )
            scene_delta_nodes, scene_delta_edges = self._build_scene_delta_graph(
                entities=entities,
                entity_segments=entity_segments,
                scene_base_nodes=scene_base_nodes,
                state_episodes=state_episodes,
                relation_episodes=relation_episodes,
                transition_events=transition_events,
            )
            self._log_stage(
                "build_scene_delta_graph",
                started_at,
                scene_delta_nodes=len(scene_delta_nodes),
                scene_delta_edges=len(scene_delta_edges),
            )
            scene_delta_observations = self._build_scene_delta_observations(
                observations=observations,
                entity_segments=entity_segments,
                scene_base_nodes=scene_base_nodes,
                transition_events=transition_events,
            )
            self._log_stage(
                "build_scene_delta_observations",
                started_at,
                scene_delta_observations=len(scene_delta_observations),
            )

            document = TemporalGraphDocument(
                video_id=self.video_id,
                source_dir=self.input_dir.resolve(),
                frame_count=len(frame_metas),
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
            self._log_alignment_summary()
            self._log_stage("complete", started_at, frame_count=len(frame_metas))
            return document
        finally:
            if self._owns_embedding_cache:
                self.embedding_cache.close()

    def apply_garbage_collection(
        self,
        document: TemporalGraphDocument,
        min_lifespan: int | None = None,
    ) -> TemporalGraphDocument:
        threshold = (
            self.config.garbage_collection.min_lifespan
            if min_lifespan is None
            else min_lifespan
        )
        observations_by_entity: dict[str, list[Observation]] = defaultdict(list)
        for observation in document.observations:
            observations_by_entity[observation.entity_id].append(observation)

        semantic_fragment_demotions_enabled = (
            self.config.association.identity_role_strategy != "tracklet"
        )
        human_area_median = (
            self._human_entity_area_median(
                entities=document.entities,
                observations_by_entity=observations_by_entity,
            )
            if semantic_fragment_demotions_enabled
            else None
        )
        static_duplicate_entity_ids = self._detect_static_duplicate_entity_ids(
            entities=document.entities,
            observations_by_entity=observations_by_entity,
        )
        static_fragment_entity_ids = self._detect_static_fragment_entity_ids(
            entities=document.entities,
        )

        updated_entities: list[Entity] = []
        for entity in document.entities:
            lifespan = entity.last_seen_frame - entity.first_seen_frame + 1
            is_fragment = False
            if not entity.is_noise:
                if entity.is_static:
                    is_fragment = entity.entity_id in static_fragment_entity_ids
                else:
                    is_fragment = (
                        lifespan <= threshold
                        or entity.observation_count <= threshold
                    )
            is_noise = entity.is_noise
            if (
                is_fragment
                and self.config.fragment_noise_max_score is not None
                and entity.category in self.config.normalized_fragment_noise_labels
            ):
                entity_observations = observations_by_entity.get(entity.entity_id, [])
                if entity_observations:
                    max_score = max(observation.score for observation in entity_observations)
                    if max_score <= self.config.fragment_noise_max_score:
                        is_noise = True
            entity_observations = observations_by_entity.get(entity.entity_id, [])
            if (
                semantic_fragment_demotions_enabled
                and self._should_demote_small_appearance_human_entity(
                    entity=entity,
                    observations=entity_observations,
                    human_area_median=human_area_median,
                )
            ):
                is_noise = True
            if entity.entity_id in static_duplicate_entity_ids:
                is_noise = True
            updated_entities.append(
                entity.model_copy(
                    update={
                        "is_fragment": is_fragment,
                        "is_noise": is_noise,
                    }
                )
            )
        return document.model_copy(update={"entities": updated_entities})

    def _human_entity_area_median(
        self,
        *,
        entities: list[Entity],
        observations_by_entity: dict[str, list[Observation]],
    ) -> float | None:
        areas: list[float] = []
        for entity in entities:
            if entity.is_static or entity.is_noise:
                continue
            if not self._is_human_like_entity(entity):
                continue
            observations = observations_by_entity.get(entity.entity_id, [])
            if not observations:
                continue
            areas.append(mean(observation.bbox.area for observation in observations))
        if not areas:
            return None
        return float(median(areas))

    def _detect_static_fragment_entity_ids(
        self,
        *,
        entities: list[Entity],
    ) -> set[str]:
        max_observations = self.config.garbage_collection.static_fragment_max_observations
        if max_observations is None or max_observations < 1:
            return set()

        return {
            entity.entity_id
            for entity in entities
            if entity.is_static
            and not entity.is_noise
            and entity.observation_count <= max_observations
        }

    def _should_demote_small_appearance_human_entity(
        self,
        *,
        entity: Entity,
        observations: list[Observation],
        human_area_median: float | None,
    ) -> bool:
        if entity.is_static or entity.is_noise:
            return False
        if not self._is_human_like_entity(entity):
            return False
        if entity.observation_count > 12:
            return False
        if human_area_median is None or not observations:
            return False
        average_area = mean(observation.bbox.area for observation in observations)
        if average_area > human_area_median * 0.55:
            return False
        context = self._human_entity_context(
            entity=entity,
            observations=observations,
        )
        return context == "appearance_only"

    def _human_entity_context(
        self,
        *,
        entity: Entity,
        observations: list[Observation],
    ) -> str:
        texts = [
            *entity.labels_seen,
            *(entity.raw_states_seen or entity.states_seen),
        ]
        for observation in observations:
            texts.append(observation.label)
            texts.extend(observation.raw_state_values or observation.state_values)

        player_like_hits = 0
        appearance_hits = 0
        for text in texts:
            normalized_text = self._normalize_text(text)
            if any(keyword in normalized_text for keyword in PLAYER_LIKE_KEYWORDS):
                player_like_hits += 1
            if any(keyword in normalized_text for keyword in APPEARANCE_ONLY_KEYWORDS):
                appearance_hits += 1

        if player_like_hits >= 2:
            return "player_like"
        if player_like_hits == 0 and appearance_hits > 0:
            return "appearance_only"
        return "generic"

    def _is_human_like_entity(self, entity: Entity) -> bool:
        labels = {entity.category, *entity.labels_seen}
        return any(
            label in (self.component_host_labels or HUMAN_LIKE_LABELS)
            for label in labels
        )

    def _detect_static_duplicate_entity_ids(
        self,
        *,
        entities: list[Entity],
        observations_by_entity: dict[str, list[Observation]],
    ) -> set[str]:
        grouped: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            if entity.is_static and not entity.is_noise:
                grouped[entity.category].append(entity)

        duplicate_ids: set[str] = set()
        aggregate_boxes = {
            entity.entity_id: self._average_observation_bbox(
                observations_by_entity.get(entity.entity_id, [])
            )
            for entity in entities
        }

        for same_label_entities in grouped.values():
            ordered = sorted(
                same_label_entities,
                key=lambda item: (-item.observation_count, item.first_seen_frame, item.entity_id),
            )
            for index, entity in enumerate(ordered[1:], start=1):
                current_bbox = aggregate_boxes.get(entity.entity_id)
                if current_bbox is None:
                    continue
                for stronger in ordered[:index]:
                    if stronger.observation_count < entity.observation_count * 2:
                        continue
                    stronger_bbox = aggregate_boxes.get(stronger.entity_id)
                    if stronger_bbox is None:
                        continue
                    if self._static_entities_look_duplicate(
                        current_bbox=current_bbox,
                        stronger_bbox=stronger_bbox,
                    ):
                        duplicate_ids.add(entity.entity_id)
                        break
        return duplicate_ids

    def _average_observation_bbox(
        self,
        observations: list[Observation],
    ) -> BoundingBox | None:
        if not observations:
            return None
        return BoundingBox(
            x1=mean(observation.bbox.x1 for observation in observations),
            y1=mean(observation.bbox.y1 for observation in observations),
            x2=mean(observation.bbox.x2 for observation in observations),
            y2=mean(observation.bbox.y2 for observation in observations),
        )

    def _static_entities_look_duplicate(
        self,
        *,
        current_bbox: BoundingBox,
        stronger_bbox: BoundingBox,
    ) -> bool:
        if current_bbox.iou(stronger_bbox) >= 0.05:
            return True
        if current_bbox.inside_ratio(stronger_bbox) >= 0.60:
            return True
        if stronger_bbox.inside_ratio(current_bbox) >= 0.60:
            return True
        return False

    def _reset_runtime_state(self) -> None:
        self._entity_counter = 0
        self._entity_states: dict[str, EntityStateRecord] = {}
        self._entity_state_ids_by_label: dict[str, set[str]] = defaultdict(set)
        self._static_entity_state_ids: set[str] = set()
        self._dynamic_entity_state_ids: set[str] = set()
        self._frame_rank_by_index: dict[int, int] = {}
        self._alignment_stats = _AlignmentStats()
        self._state_filter_stats = _StateFilterStats()
        self._state_filter_kept_counter: Counter[str] = Counter()
        self._state_filter_rejected_counter: Counter[str] = Counter()
        self.association_policy.reset_stats()
        if hasattr(self.embedding_cache, "reset_stats"):
            self.embedding_cache.reset_stats()

    def get_runtime_stats(self) -> dict[str, Any]:
        alignment = asdict(self._alignment_stats)
        state_filter = asdict(self._state_filter_stats)
        state_filter["top_kept_states"] = self._state_filter_kept_counter.most_common(10)
        state_filter["top_rejected_states"] = (
            self._state_filter_rejected_counter.most_common(10)
        )
        cache_stats: dict[str, Any] = {}
        if hasattr(self.embedding_cache, "get_stats"):
            cache_stats = self.embedding_cache.get_stats()
        llm_fallback_stats: dict[str, Any] = {}
        if hasattr(self.llm_fallback_resolver, "get_stats"):
            llm_fallback_stats = self.llm_fallback_resolver.get_stats()
        return {
            "alignment": alignment,
            "state_filter": state_filter,
            "association_policy": self.association_policy.get_stats(),
            "llm_fallback": llm_fallback_stats,
            "embedding_cache": cache_stats,
        }

    def _load_json(self, file_name: str) -> dict[str, Any]:
        file_path = self.json_dir / file_name
        with file_path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)

    def _load_frame_time_index(
        self,
        frame_time_index_path: str | Path | None,
    ) -> dict[int, float]:
        if frame_time_index_path is None:
            return {}
        path = Path(frame_time_index_path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError(f"frame_time_index frames must be a list: {path}")
        index: dict[int, float] = {}
        for item in frames:
            if not isinstance(item, dict):
                continue
            frame_value = item.get("graph_frame_index", item.get("frame_index"))
            timestamp_value = item.get("timestamp_unix", item.get("timestamp_sec"))
            if frame_value is None or timestamp_value is None:
                continue
            index[int(frame_value)] = float(timestamp_value)
        return index

    def _sorted_frame_metas(self, frame_names: Any) -> list[_FrameMeta]:
        metas = [self._parse_frame_name(frame_name) for frame_name in frame_names]
        return sorted(metas, key=lambda item: (item.frame_index, item.timestamp_sec))

    def _parse_frame_name(self, frame_name: str) -> _FrameMeta:
        match = FRAME_NAME_PATTERN.match(frame_name)
        if match:
            return _FrameMeta(
                frame_name=frame_name,
                frame_index=int(match.group("frame_idx")),
                timestamp_sec=float(match.group("timestamp")),
            )

        match = FRAME_NAME_INDEX_ONLY_PATTERN.match(frame_name)
        if match:
            frame_index = int(match.group("frame_idx"))
            return _FrameMeta(
                frame_name=frame_name,
                frame_index=frame_index,
                timestamp_sec=self.frame_time_index.get(frame_index, float(frame_index)),
            )

        match = CODA_FRAME_NAME_PATTERN.match(frame_name)
        if match:
            frame_index = int(match.group("frame_idx"))
            return _FrameMeta(
                frame_name=frame_name,
                frame_index=frame_index,
                timestamp_sec=self.frame_time_index.get(frame_index, float(frame_index)),
            )

        match = CODA_SAMPLED_FRAME_NAME_PATTERN.match(frame_name)
        if match:
            raw_index = int(match.group("raw_idx"))
            return _FrameMeta(
                frame_name=frame_name,
                frame_index=raw_index,
                timestamp_sec=self.frame_time_index.get(raw_index, float(raw_index)),
            )

        raise ValueError(f"Unsupported frame name format: {frame_name}")

    def _build_observations(
        self,
        *,
        frame_metas: list[_FrameMeta],
        objects_table: dict[str, Any],
        attributes_table: dict[str, Any],
        hierarchy_table: dict[str, Any],
    ) -> tuple[list[Observation], dict[tuple[str, int], Observation]]:
        observations: list[Observation] = []
        observation_lookup: dict[tuple[str, int], Observation] = {}
        state_context_by_frame = (
            self._build_state_context_index(objects_table)
            if self.enable_structured_state_extraction
            else {}
        )
        global_state_context = (
            set().union(*state_context_by_frame.values())
            if state_context_by_frame
            else set()
        )

        for frame_rank, meta in enumerate(frame_metas):
            frame_payload = objects_table[meta.frame_name]
            frame_attributes = attributes_table.get(meta.frame_name, {})
            hierarchy_entry = hierarchy_table.get(meta.frame_name, {})
            parent_lookup = self._build_parent_lookup(hierarchy_entry)
            prev_label_boxes = self._neighbor_label_boxes(
                frame_metas=frame_metas,
                objects_table=objects_table,
                frame_rank=frame_rank - 1,
            )
            next_label_boxes = self._neighbor_label_boxes(
                frame_metas=frame_metas,
                objects_table=objects_table,
                frame_rank=frame_rank + 1,
            )

            boxes = frame_payload["boxes"]
            labels = frame_payload["labels"]
            scores = frame_payload["scores"]
            if self.frame_image_root is not None:
                image_path = str(self.frame_image_root / f"{meta.frame_name}.png")
            else:
                raw_image_path = Path(frame_payload['image_path'])
                if not raw_image_path.is_absolute():
                    raw_image_path = self.input_dir / raw_image_path
                image_path = str(resolve_data_path(raw_image_path))

            frame_box_data: list[_FrameObjectInput] = []
            for object_index, (box_values, label, score) in enumerate(
                zip(boxes, labels, scores, strict=True)
            ):
                normalized_label = self._normalize_label(label)
                min_score = self.label_min_scores.get(normalized_label, 0.0)
                if float(score) < min_score:
                    continue
                bbox = BoundingBox(
                    x1=float(box_values[0]),
                    y1=float(box_values[1]),
                    x2=float(box_values[2]),
                    y2=float(box_values[3]),
                )
                parent_label = parent_lookup.get(normalized_label)
                role_decision = self._classify_frame_object_role(
                    label=normalized_label,
                    parent_label=parent_label,
                    bbox=bbox,
                    hierarchy_entry=hierarchy_entry,
                    prev_label_boxes=prev_label_boxes,
                    next_label_boxes=next_label_boxes,
                )
                frame_box_data.append(
                    _FrameObjectInput(
                        object_index=object_index,
                        bbox=bbox,
                        label=normalized_label,
                        score=float(score),
                        state=self._normalize_state_value(
                            frame_attributes.get(str(object_index))
                        ),
                        parent_label=parent_label,
                        is_static=role_decision.is_static,
                        is_noise=role_decision.role == "noise",
                        node_role=role_decision.role,
                    )
                )

            if self.config.association.identity_role_strategy != "tracklet":
                frame_box_data = self._fold_component_states(frame_box_data)
            kept_items = [
                item
                for item in frame_box_data
                if not item.folded_into_host and not item.skip_entity_creation
            ]

            embeddings = self._prepare_frame_embeddings(
                image_path=image_path,
                boxes=[item.bbox for item in kept_items],
                labels=[item.label for item in kept_items],
            )

            for item, embedding in zip(kept_items, embeddings, strict=True):
                track_id = self.tracking_provider.assign_track_id(
                    TrackingObservationInput(
                        video_id=self.video_id,
                        frame_name=meta.frame_name,
                        frame_rank=frame_rank,
                        frame_index=meta.frame_index,
                        object_index=item.object_index,
                        label=item.label,
                        image_path=image_path,
                        bbox=item.bbox,
                        is_static=item.is_static,
                        is_noise=item.is_noise,
                    )
                )
                raw_state_values: list[str] = []
                for state_value in item.state_values:
                    normalized_state_value = self._normalize_state_value(state_value)
                    if (
                        normalized_state_value
                        and normalized_state_value not in raw_state_values
                    ):
                        raw_state_values.append(normalized_state_value)
                state_values = []
                if self.enable_structured_state_extraction:
                    state_context_terms = (
                        state_context_by_frame.get(meta.frame_name, set())
                        | global_state_context
                    )
                    state_values = self._structure_state_values(
                        raw_state_values,
                        state_context_terms=state_context_terms,
                    )
                entity_id = self._assign_entity_id(
                    frame_name=meta.frame_name,
                    frame_rank=frame_rank,
                    frame_index=meta.frame_index,
                    track_id=track_id,
                    image_path=image_path,
                    object_index=item.object_index,
                    label=item.label,
                    bbox=item.bbox,
                    parent_label=item.parent_label,
                    state_values=raw_state_values,
                    is_static=item.is_static,
                    is_noise=item.is_noise,
                    embedding=embedding,
                )
                observation = Observation(
                    observation_id=(
                        f"{self.video_id}:{meta.frame_name}:obj_{item.object_index:04d}"
                    ),
                    video_id=self.video_id,
                    entity_id=entity_id,
                    frame_name=meta.frame_name,
                    frame_rank=frame_rank,
                    frame_index=meta.frame_index,
                    timestamp_sec=meta.timestamp_sec,
                    image_path=image_path,
                    object_index=item.object_index,
                    bbox=item.bbox,
                    label=item.label,
                    score=item.score,
                    state=state_values[0] if state_values else None,
                    state_values=state_values,
                    raw_state_values=raw_state_values,
                    parent_label=item.parent_label,
                    node_role=item.node_role,
                    is_static=item.is_static,
                    is_noise=item.is_noise,
                )
                observations.append(observation)
                observation_lookup[(meta.frame_name, item.object_index)] = observation

        return observations, observation_lookup

    def _fold_component_states(
        self,
        items: list[_FrameObjectInput],
    ) -> list[_FrameObjectInput]:
        if not self.component_labels:
            return items

        hosts = [
            item
            for item in items
            if item.label in self.component_host_labels
            and not item.is_noise
        ]
        if not hosts:
            return items

        for item in items:
            if item.label not in self.component_labels:
                continue
            host = self._find_component_host(component=item, hosts=hosts)
            if host is None:
                item.is_noise = True
                item.skip_entity_creation = True
                continue
            folded_state = self._component_to_state(item)
            if folded_state:
                host.folded_states.append(folded_state)
            item.folded_into_host = True

        return items

    def _find_component_host(
        self,
        *,
        component: _FrameObjectInput,
        hosts: list[_FrameObjectInput],
    ) -> _FrameObjectInput | None:
        best_host: _FrameObjectInput | None = None
        best_ratio = 0.0
        for host in hosts:
            if host.object_index == component.object_index:
                continue
            inside_ratio = component.bbox.inside_ratio(host.bbox)
            if inside_ratio < self.component_inside_ratio:
                continue
            if inside_ratio > best_ratio:
                best_ratio = inside_ratio
                best_host = host
        if best_host is not None:
            return best_host

        for host in hosts:
            if host.object_index == component.object_index:
                continue
            if host.bbox.contains_center_of(component.bbox):
                return host

        best_distance = float("inf")
        fallback_host: _FrameObjectInput | None = None
        for host in hosts:
            overlap_ratio = component.bbox.inside_ratio(host.bbox)
            if overlap_ratio < self.component_overlap_ratio:
                continue
            distance = component.bbox.center_distance(host.bbox)
            max_diagonal = max(component.bbox.diagonal, host.bbox.diagonal, 1.0)
            normalized_distance = distance / max_diagonal
            if normalized_distance <= self.component_nearest_distance_ratio and distance < best_distance:
                best_distance = distance
                fallback_host = host
        return fallback_host

    def _component_to_state(self, component: _FrameObjectInput) -> str | None:
        if component.label not in WEARABLE_COMPONENT_LABELS:
            return None

        component_state = self._normalize_state_value(component.state)
        if component_state:
            if component_state.startswith("wearing "):
                return component_state
            return f"wearing {component_state}"

        return f"wearing {component.label}"

    def _build_state_context_index(
        self,
        objects_table: dict[str, Any],
    ) -> dict[str, set[str]]:
        context_by_frame: dict[str, set[str]] = {}
        for frame_name, frame_payload in objects_table.items():
            frame_context: set[str] = set()
            for label in frame_payload.get("labels", []):
                frame_context.update(self._state_context_terms(label))
                frame_context.update(self._state_context_terms(self._normalize_label(label)))
            context_by_frame[frame_name] = frame_context
        return context_by_frame

    def _structure_state_values(
        self,
        raw_state_values: list[str],
        *,
        state_context_terms: set[str] | None = None,
    ) -> list[str]:
        structured: list[str] = []
        for raw_state_value in raw_state_values:
            for candidate in self._structured_state_candidates_from_text(raw_state_value):
                if not self._should_keep_structured_state_candidate(
                    candidate,
                    state_context_terms=state_context_terms,
                ):
                    continue
                if candidate not in structured:
                    structured.append(candidate)
        return structured

    def _should_keep_structured_state_candidate(
        self,
        candidate: str,
        *,
        state_context_terms: set[str] | None,
    ) -> bool:
        normalized = self._normalize_state_value(candidate)
        if not normalized:
            self._record_state_filter_decision(
                normalized,
                kept=False,
                reason="empty",
            )
            return False
        if normalized in self.state_exact_allowlist:
            self._record_state_filter_decision(
                normalized,
                kept=True,
                reason="exact_allowlist",
            )
            return True

        prefix = self._state_prefix_for_candidate(normalized)
        if prefix not in OBJECT_SENSITIVE_STATE_PREFIXES:
            self._record_state_filter_decision(
                normalized,
                kept=True,
                reason="object_insensitive",
            )
            return True

        object_terms = self._state_object_terms_from_candidate(
            normalized_state_value=normalized,
            prefix=prefix,
        )
        if not object_terms:
            self._record_state_filter_decision(
                normalized,
                kept=True,
                reason="object_free",
            )
            return True
        if not state_context_terms:
            self._record_state_filter_decision(
                normalized,
                kept=True,
                reason="object_context_unchecked",
            )
            return True
        support_terms = self._state_object_support_terms_from_candidate(
            normalized_state_value=normalized,
            prefix=prefix,
        )
        if support_terms & state_context_terms:
            self._record_state_filter_decision(
                normalized,
                kept=True,
                reason="object_context_supported",
            )
            return True
        self._record_state_filter_decision(
            normalized,
            kept=False,
            reason="object_context_unsupported",
        )
        return False

    def _record_state_filter_decision(
        self,
        state_value: str,
        *,
        kept: bool,
        reason: str,
    ) -> None:
        self._state_filter_stats.candidates_seen += 1
        if kept:
            self._state_filter_stats.kept_candidates += 1
            if state_value:
                self._state_filter_kept_counter[state_value] += 1
        else:
            self._state_filter_stats.rejected_candidates += 1
            if state_value:
                self._state_filter_rejected_counter[state_value] += 1

        if reason == "empty":
            self._state_filter_stats.rejected_empty_candidates += 1
        elif reason == "exact_allowlist":
            self._state_filter_stats.kept_exact_allowlist += 1
        elif reason == "object_insensitive":
            self._state_filter_stats.kept_object_insensitive += 1
        elif reason == "object_free":
            self._state_filter_stats.kept_object_free += 1
        elif reason == "object_context_supported":
            self._state_filter_stats.kept_object_context_supported += 1
        elif reason == "object_context_unchecked":
            self._state_filter_stats.kept_object_context_unchecked += 1
        elif reason == "object_context_unsupported":
            self._state_filter_stats.rejected_object_context_unsupported += 1

    def _state_prefix_for_candidate(self, normalized_state_value: str) -> str | None:
        for prefix in self.state_summary_prefixes:
            if (
                normalized_state_value == prefix
                or normalized_state_value.startswith(f"{prefix} ")
            ):
                return prefix
        return None

    def _state_object_terms_from_candidate(
        self,
        *,
        normalized_state_value: str,
        prefix: str,
    ) -> set[str]:
        object_phrase = normalized_state_value.removeprefix(prefix).strip()
        return self._state_context_terms(object_phrase)

    def _state_object_support_terms_from_candidate(
        self,
        *,
        normalized_state_value: str,
        prefix: str,
    ) -> set[str]:
        object_phrase = normalized_state_value.removeprefix(prefix).strip()
        normalized_phrase = self._normalize_text(object_phrase)
        if not normalized_phrase:
            return set()

        tokens = [
            token
            for token in re.findall(r"[0-9a-z]+", normalized_phrase)
            if token not in STATE_OBJECT_STOPWORDS
        ]
        if not tokens:
            return set()
        if len(tokens) == 1:
            return self._state_context_terms(tokens[0])

        full_phrase = " ".join(tokens)
        head_token = tokens[-1]
        support_terms = {full_phrase}
        full_phrase_alias = self._normalize_label(full_phrase)
        if full_phrase_alias and full_phrase_alias != full_phrase:
            support_terms.add(full_phrase_alias)
        support_terms.update(self._state_context_terms(head_token))
        return support_terms

    def _state_context_terms(self, value: str | None) -> set[str]:
        normalized = self._normalize_text(value)
        if not normalized:
            return set()

        terms = {normalized}
        alias = self._normalize_label(normalized)
        if alias and alias != normalized:
            terms.add(alias)

        for token in re.findall(r"[0-9a-z]+", normalized):
            if token in STATE_OBJECT_STOPWORDS:
                continue
            terms.add(token)
            singular = self._singularize_state_token(token)
            terms.add(singular)
            token_alias = self._normalize_label(token)
            if token_alias and token_alias != token:
                terms.update(self._state_context_terms(token_alias))
        return {term for term in terms if term}

    def _singularize_state_token(self, token: str) -> str:
        if len(token) > 3 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _structured_state_candidates_from_text(
        self,
        state_value: str,
    ) -> list[str]:
        normalized = self._normalize_state_value(state_value)
        if not normalized:
            return []

        candidates: list[str] = []
        if normalized in self.state_exact_allowlist:
            candidates.append(normalized)

        for prefix in self.state_summary_prefixes:
            candidate = self._extract_state_prefix_candidate(
                normalized_state_value=normalized,
                prefix=prefix,
            )
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _extract_state_prefix_candidate(
        self,
        *,
        normalized_state_value: str,
        prefix: str,
    ) -> str | None:
        if not prefix:
            return None
        if (
            normalized_state_value == prefix
            or normalized_state_value.startswith(f"{prefix} ")
        ):
            return normalized_state_value

        match = re.search(
            rf"(?<!\S){re.escape(prefix)}(?:\s+.+)?$",
            normalized_state_value,
        )
        if match is None:
            return None

        candidate = self._normalize_state_value(match.group(0))
        return candidate or None

    def _prepare_frame_embeddings(
        self,
        *,
        image_path: str,
        boxes: list[BoundingBox],
        labels: list[str],
    ) -> list[np.ndarray]:
        if not boxes:
            return []

        if not self.config.association.enable_visual_embeddings:
            return [
                self._build_structural_embedding(label=label, bbox=bbox)
                for label, bbox in zip(labels, boxes, strict=True)
            ]

        requests = [(image_path, bbox) for bbox in boxes]
        cache_hits = self.embedding_cache.get_many(requests)

        embeddings: list[np.ndarray | None] = [None] * len(requests)
        misses: list[tuple[int, str, BoundingBox]] = []
        for index, (req_image_path, req_bbox) in enumerate(requests):
            cache_key = (
                str(Path(req_image_path)),
                self.embedding_cache.bbox_to_key(req_bbox),
            )
            cached = cache_hits.get(cache_key)
            if cached is not None:
                embeddings[index] = cached
            else:
                misses.append((index, req_image_path, req_bbox))

        if misses:
            miss_requests = [
                (req_image_path, req_bbox)
                for _, req_image_path, req_bbox in misses
            ]
            miss_embeddings = self.vision_feature_extractor.encode_batch(miss_requests)
            self.embedding_cache.set_many(
                [
                    (req_image_path, req_bbox, embedding)
                    for (_, req_image_path, req_bbox), embedding in zip(
                        misses, miss_embeddings, strict=True
                    )
                ]
            )
            for (index, _, _), embedding in zip(misses, miss_embeddings, strict=True):
                embeddings[index] = embedding

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("Failed to populate all frame embeddings.")
        return [embedding for embedding in embeddings if embedding is not None]

    @staticmethod
    def _build_structural_embedding(*, label: str, bbox: BoundingBox) -> np.ndarray:
        """Build a deterministic non-visual identity vector for graph-only builds.

        The vector is deliberately derived only from HSGG labels and box geometry.
        It is used for temporal association when RGB files are unavailable and is
        not a visual feature or a retrieval index embedding.
        """
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        label_features = np.asarray(
            [(byte / 127.5) - 1.0 for byte in digest[:32]], dtype=np.float32
        )
        geometry = np.asarray(
            [
                bbox.center_x / 2048.0,
                bbox.center_y / 2048.0,
                bbox.width / 2048.0,
                bbox.height / 2048.0,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate([label_features, geometry * 0.15])
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).astype(np.float32)

    def _neighbor_label_boxes(
        self,
        *,
        frame_metas: list[_FrameMeta],
        objects_table: dict[str, Any],
        frame_rank: int,
    ) -> dict[str, list[BoundingBox]]:
        if frame_rank < 0 or frame_rank >= len(frame_metas):
            return {}
        frame_payload = objects_table[frame_metas[frame_rank].frame_name]
        label_boxes: dict[str, list[BoundingBox]] = defaultdict(list)
        for box_values, raw_label in zip(
            frame_payload.get("boxes", []),
            frame_payload.get("labels", []),
            strict=True,
        ):
            label_boxes[self._normalize_label(raw_label)].append(
                BoundingBox(
                    x1=float(box_values[0]),
                    y1=float(box_values[1]),
                    x2=float(box_values[2]),
                    y2=float(box_values[3]),
                )
            )
        return label_boxes

    def _classify_frame_object_role(
        self,
        *,
        label: str,
        parent_label: str | None,
        bbox: BoundingBox,
        hierarchy_entry: dict[str, Any],
        prev_label_boxes: dict[str, list[BoundingBox]],
        next_label_boxes: dict[str, list[BoundingBox]],
    ) -> NodeRoleDecision:
        if self.config.association.identity_role_strategy == "tracklet":
            if label in self.noise_labels:
                return NodeRoleDecision(
                    role="noise",
                    is_static=False,
                    confidence=1.0,
                    reason="noise_label",
                )
            return NodeRoleDecision(
                role="dynamic",
                is_static=False,
                confidence=0.6,
                reason="tracklet_identity_observation",
            )

        hierarchy_anchor = self._is_hierarchy_anchor(
            label=label,
            hierarchy_entry=hierarchy_entry,
        )
        evidence = NodeRoleEvidence(
            label=label,
            parent_label=parent_label,
            lexical_static_prior=label in self.static_labels,
            lexical_dynamic_prior=(
                label in self.dynamic_labels
                or label in self.component_host_labels
                or label in self.state_subject_labels
            ),
            lexical_dependent_prior=label in self.component_labels,
            hierarchy_anchor=hierarchy_anchor,
            scene_anchor=(
                label in self.static_labels
                or hierarchy_anchor
            ),
            max_prev_same_label_iou=self._max_same_label_iou(
                bbox=bbox,
                other_boxes=prev_label_boxes.get(label, []),
            ),
            max_next_same_label_iou=self._max_same_label_iou(
                bbox=bbox,
                other_boxes=next_label_boxes.get(label, []),
            ),
            is_noise_label=label in self.noise_labels,
        )
        decision = classify_node_role(
            evidence,
            static_iou_threshold=self.role_static_iou_threshold,
            anchor_iou_threshold=self.role_anchor_iou_threshold,
            dynamic_iou_upper_bound=self.role_dynamic_iou_upper_bound,
        )
        if (
            not self.config.association.allow_visual_stability_static_role
            and decision.reason == "neighbor_iou_stability"
        ):
            return NodeRoleDecision(
                role="dynamic",
                is_static=False,
                confidence=min(decision.confidence, 0.6),
                reason="visual_stability_not_static_identity",
            )
        return decision

    def _max_same_label_iou(
        self,
        *,
        bbox: BoundingBox,
        other_boxes: list[BoundingBox],
    ) -> float:
        if not other_boxes:
            return 0.0
        return max(bbox.iou(other_bbox) for other_bbox in other_boxes)

    def _build_parent_lookup(self, hierarchy_entry: dict[str, Any]) -> dict[str, str]:
        parent_lookup: dict[str, str] = {}
        layer2_mapping = hierarchy_entry.get("layer2_mapping", {})
        for parent_label, child_labels in layer2_mapping.items():
            normalized_parent = self._normalize_label(parent_label)
            for child_label in child_labels:
                parent_lookup.setdefault(
                    self._normalize_label(child_label),
                    normalized_parent,
                )
        return parent_lookup

    def _is_hierarchy_anchor(
        self,
        *,
        label: str,
        hierarchy_entry: dict[str, Any],
    ) -> bool:
        if label not in self.hierarchy_static_labels:
            return False
        layer1_nodes = {
            self._normalize_label(node)
            for node in hierarchy_entry.get("layer1_nodes", [])
        }
        if label in layer1_nodes:
            return True
        layer2_parents = {
            self._normalize_label(node)
            for node in hierarchy_entry.get("layer2_mapping", {}).keys()
        }
        return label in layer2_parents

    def _infer_static(
        self,
        *,
        label: str,
        hierarchy_entry: dict[str, Any],
    ) -> bool:
        if label in self.dynamic_labels:
            return False
        if label in self.static_labels:
            return True
        if label not in self.hierarchy_static_labels:
            return False
        layer1_nodes = {
            self._normalize_label(node)
            for node in hierarchy_entry.get("layer1_nodes", [])
        }
        if label in layer1_nodes:
            return True
        layer2_parents = {
            self._normalize_label(node)
            for node in hierarchy_entry.get("layer2_mapping", {}).keys()
        }
        if label in layer2_parents:
            return True
        return False

    def _assign_entity_id(
        self,
        *,
        frame_name: str,
        frame_rank: int,
        frame_index: int,
        track_id: str | None,
        image_path: str,
        object_index: int,
        label: str,
        bbox: BoundingBox,
        parent_label: str | None,
        state_values: list[str],
        is_static: bool,
        is_noise: bool,
        embedding: np.ndarray,
    ) -> str:
        self._alignment_stats.total_alignments += 1
        candidates = build_association_candidates(
            self._candidate_entity_states_for_association(
                label=label,
                frame_rank=frame_rank,
                is_static=is_static,
            )
        )
        decision = self.association_policy.select_entity_match(
            candidates=candidates,
            current=AssociationInput(
                video_id=self.video_id,
                frame_name=frame_name,
                current_label=label,
                current_frame_rank=frame_rank,
                current_frame_index=frame_index,
                current_track_id=track_id,
                object_index=object_index,
                current_image_path=image_path,
                current_parent_label=parent_label,
                current_state_values=state_values,
                current_is_static=is_static,
                current_is_noise=is_noise,
                current_bbox=bbox,
                current_embedding=embedding,
                label_compatibility_groups=self.label_compatibility_groups,
                current_match_max_frame_gap=self._match_max_frame_gap_for_label(label),
            ),
        )
        if decision.path == "local_tracklet" and decision.matched_entity_id is not None:
            self._alignment_stats.fast_path_hits += 1
            local_match = self._entity_states[decision.matched_entity_id]
            self._update_entity_state(
                entity_state=local_match,
                frame_rank=frame_rank,
                frame_index=frame_index,
                track_id=track_id,
                image_path=image_path,
                object_index=object_index,
                bbox=bbox,
                parent_label=parent_label,
                state_values=state_values,
                is_static=is_static,
                is_noise=is_noise,
                embedding=embedding,
            )
            return local_match.entity_id

        if decision.path == "fast" and decision.matched_entity_id is not None:
            self._alignment_stats.fast_path_hits += 1
            fast_match = self._entity_states[decision.matched_entity_id]
            self._update_entity_state(
                entity_state=fast_match,
                frame_rank=frame_rank,
                frame_index=frame_index,
                track_id=track_id,
                image_path=image_path,
                object_index=object_index,
                bbox=bbox,
                parent_label=parent_label,
                state_values=state_values,
                is_static=is_static,
                is_noise=is_noise,
                embedding=embedding,
            )
            return fast_match.entity_id

        if decision.path == "slow" and decision.matched_entity_id is not None:
            self._alignment_stats.slow_path_hits += 1
            slow_match = self._entity_states[decision.matched_entity_id]
            self._update_entity_state(
                entity_state=slow_match,
                frame_rank=frame_rank,
                frame_index=frame_index,
                track_id=track_id,
                image_path=image_path,
                object_index=object_index,
                bbox=bbox,
                parent_label=parent_label,
                state_values=state_values,
                is_static=is_static,
                is_noise=is_noise,
                embedding=embedding,
            )
            return slow_match.entity_id

        if decision.path == "fallback" and decision.matched_entity_id is not None:
            self._alignment_stats.fallback_path_hits += 1
            fallback_match = self._entity_states[decision.matched_entity_id]
            self._update_entity_state(
                entity_state=fallback_match,
                frame_rank=frame_rank,
                frame_index=frame_index,
                track_id=track_id,
                image_path=image_path,
                object_index=object_index,
                bbox=bbox,
                parent_label=parent_label,
                state_values=state_values,
                is_static=is_static,
                is_noise=is_noise,
                embedding=embedding,
            )
            return fallback_match.entity_id

        if decision.path == "reject":
            self._alignment_stats.reject_path_hits += 1

        self._alignment_stats.new_entity_creations += 1
        entity_state = self._create_entity_state(
            frame_rank=frame_rank,
            frame_index=frame_index,
            track_id=track_id,
            image_path=image_path,
            label=label,
            object_index=object_index,
            bbox=bbox,
            parent_label=parent_label,
            state_values=state_values,
            is_static=is_static,
            is_noise=is_noise,
            embedding=embedding,
        )
        return entity_state.entity_id

    def _create_entity_state(
        self,
        *,
        frame_rank: int,
        frame_index: int,
        track_id: str | None,
        image_path: str,
        label: str,
        object_index: int,
        bbox: BoundingBox,
        parent_label: str | None,
        state_values: list[str],
        is_static: bool,
        is_noise: bool,
        embedding: np.ndarray,
    ) -> EntityStateRecord:
        entity_id = self._new_entity_id()
        entity_state = create_entity_state_record(
            entity_id=entity_id,
            label=label,
            frame_rank=frame_rank,
            frame_index=frame_index,
            track_id=track_id,
            object_index=object_index,
            bbox=bbox,
            parent_label=parent_label,
            state_values=state_values,
            is_static=is_static,
            is_noise=is_noise,
            image_path=image_path,
            embedding=embedding,
        )
        self._entity_states[entity_id] = entity_state
        self._index_entity_state(entity_state)
        return entity_state

    def _update_entity_state(
        self,
        *,
        entity_state: EntityStateRecord,
        frame_rank: int,
        frame_index: int,
        track_id: str | None,
        image_path: str,
        object_index: int,
        bbox: BoundingBox,
        parent_label: str | None,
        state_values: list[str],
        is_static: bool,
        is_noise: bool,
        embedding: np.ndarray,
    ) -> None:
        update_entity_state_record(
            entity_state=entity_state,
            frame_rank=frame_rank,
            frame_index=frame_index,
            track_id=track_id,
            object_index=object_index,
            bbox=bbox,
            parent_label=parent_label,
            state_values=state_values,
            is_static=is_static,
            is_noise=is_noise,
            image_path=image_path,
            embedding=embedding,
        )
        self._index_entity_state(entity_state)

    def _index_entity_state(self, entity_state: EntityStateRecord) -> None:
        self._entity_state_ids_by_label[entity_state.label].add(entity_state.entity_id)
        if entity_state.is_static:
            self._static_entity_state_ids.add(entity_state.entity_id)
            self._dynamic_entity_state_ids.discard(entity_state.entity_id)
        else:
            self._dynamic_entity_state_ids.add(entity_state.entity_id)

    def _candidate_entity_states_for_association(
        self,
        *,
        label: str,
        frame_rank: int,
        is_static: bool,
    ) -> list[EntityStateRecord]:
        compatible_labels = self._compatible_labels_for_association(label)
        compatible_ids: set[str] = set()
        for compatible_label in compatible_labels:
            compatible_ids.update(self._entity_state_ids_by_label.get(compatible_label, set()))

        if is_static:
            scoped_ids = compatible_ids & self._static_entity_state_ids
            return [
                self._entity_states[entity_id]
                for entity_id in scoped_ids
                if frame_rank > self._entity_states[entity_id].last_frame_rank
            ]

        max_gap = max(1, self._match_max_frame_gap_for_label(label))
        scoped_ids = compatible_ids & self._dynamic_entity_state_ids
        return [
            self._entity_states[entity_id]
            for entity_id in scoped_ids
            if 0 < frame_rank - self._entity_states[entity_id].last_frame_rank <= max_gap
        ]

    def _compatible_labels_for_association(self, label: str) -> set[str]:
        compatible = {label}
        for candidate_label in self._entity_state_ids_by_label:
            if candidate_label == label:
                continue
            if labels_compatible(
                current_label=label,
                candidate_label=candidate_label,
                label_compatibility_groups=self.label_compatibility_groups,
            ):
                compatible.add(candidate_label)
        return compatible

    def _new_entity_id(self) -> str:
        self._entity_counter += 1
        return f"{self.video_id}:entity:{self._entity_counter:06d}"

    def _match_max_frame_gap_for_label(self, label: str) -> int:
        return self.label_gap_overrides.get(
            label,
            self.entity_match_max_frame_gap,
        )

    def _build_entities(self, observations: list[Observation]) -> list[Entity]:
        grouped: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            grouped[observation.entity_id].append(observation)

        entities: list[Entity] = []
        for entity_id, entity_observations in grouped.items():
            entity_observations.sort(key=lambda item: item.frame_index)
            entity_state = self._entity_states[entity_id]
            label_counter = Counter(obs.label for obs in entity_observations)
            structured_state_counter = Counter(
                state_value
                for obs in entity_observations
                for state_value in obs.state_values
            )
            raw_state_counter = Counter(
                state_value
                for obs in entity_observations
                for state_value in (obs.raw_state_values or obs.state_values)
            )
            parent_counter = Counter(
                obs.parent_label
                for obs in entity_observations
                if obs.parent_label is not None
            )
            labels_seen = self._select_supported_entity_labels(
                label_counter=label_counter,
                observation_count=len(entity_observations),
            )
            states_seen = self._select_supported_entity_states(
                state_counter=structured_state_counter,
                observation_count=len(entity_observations),
            )
            first_observation = entity_observations[0]
            last_observation = entity_observations[-1]

            entities.append(
                Entity(
                    entity_id=entity_id,
                    video_id=self.video_id,
                    category=label_counter.most_common(1)[0][0],
                    first_seen_frame=first_observation.frame_index,
                    last_seen_frame=last_observation.frame_index,
                    first_seen_ts=first_observation.timestamp_sec,
                    last_seen_ts=last_observation.timestamp_sec,
                    observation_count=len(entity_observations),
                    canonical_image_path=entity_state.canonical_image_path,
                    canonical_embedding=(
                        entity_state.canonical_embedding.tolist()
                        if entity_state.canonical_embedding is not None
                        else None
                    ),
                    labels_seen=labels_seen,
                    raw_labels_seen=sorted(label_counter.keys()),
                    states_seen=states_seen,
                    raw_states_seen=sorted(raw_state_counter.keys()),
                    parent_labels_seen=sorted(parent_counter.keys()),
                    node_role=(
                        "noise"
                        if entity_state.is_noise
                        else "static"
                        if entity_state.is_static
                        else "dynamic"
                    ),
                    is_static=entity_state.is_static,
                    is_noise=entity_state.is_noise,
                )
            )

        entities.sort(key=lambda item: item.entity_id)
        return entities

    def _select_supported_entity_labels(
        self,
        *,
        label_counter: Counter[str],
        observation_count: int,
    ) -> list[str]:
        if not label_counter:
            return []

        max_count = max(label_counter.values())
        dominant_labels = {
            label
            for label, count in label_counter.items()
            if count == max_count
        }
        strong_secondary_labels = {
            label
            for label, count in label_counter.items()
            if count >= 2 and count / max(1, observation_count) >= 0.25
        }
        return sorted(dominant_labels | strong_secondary_labels)

    def _select_supported_entity_states(
        self,
        *,
        state_counter: Counter[str],
        observation_count: int,
    ) -> list[str]:
        if not state_counter:
            return []

        min_support = 2 if observation_count >= 4 else 1
        supported = [
            state_value
            for state_value, count in state_counter.items()
            if count >= min_support
        ]
        return sorted(supported)

    def _build_scene_segments(
        self,
        *,
        observations: list[Observation],
        entities: list[Entity],
    ) -> list[SceneSegment]:
        frame_signatures = build_frame_scene_signatures(
            observations,
            min_anchor_entity_support_frames=(
                self.scene_segmentation_config.min_anchor_entity_support_frames
            ),
        )
        if not frame_signatures:
            return []
        scene_ranges = detect_scene_ranges(
            frame_signatures,
            config=self.scene_segmentation_config,
        )
        scene_segments: list[SceneSegment] = []
        scene_count = len(scene_ranges)
        for scene_range in scene_ranges:
            metadata = self._build_scene_segment_metadata(
                scene_range=scene_range,
                observations=observations,
                entities=entities,
            )
            scene_segments.append(
                SceneSegment(
                    scene_id=f"{self.video_id}:scene:{scene_range.scene_index:06d}",
                    video_id=self.video_id,
                    scene_index=scene_range.scene_index,
                    scene_count=scene_count,
                    start_frame_rank=scene_range.start_frame_rank,
                    end_frame_rank=scene_range.end_frame_rank,
                    start_frame=scene_range.start_frame,
                    end_frame=scene_range.end_frame,
                    start_ts=scene_range.start_ts,
                    end_ts=scene_range.end_ts,
                    support_frames=scene_range.support_frames,
                    boundary_from_previous_frames=scene_range.boundary_from_previous_frames,
                    boundary_to_next_frames=scene_range.boundary_to_next_frames,
                    boundary_from_previous_time_gap_sec=(
                        scene_range.boundary_from_previous_time_gap_sec
                    ),
                    boundary_to_next_time_gap_sec=(
                        scene_range.boundary_to_next_time_gap_sec
                    ),
                    boundary_from_previous_entity_similarity=(
                        scene_range.boundary_from_previous_entity_similarity
                    ),
                    boundary_to_next_entity_similarity=(
                        scene_range.boundary_to_next_entity_similarity
                    ),
                    boundary_from_previous_label_similarity=(
                        scene_range.boundary_from_previous_label_similarity
                    ),
                    boundary_to_next_label_similarity=(
                        scene_range.boundary_to_next_label_similarity
                    ),
                    boundary_from_previous_reason=scene_range.previous_boundary_reason,
                    boundary_to_next_reason=scene_range.next_boundary_reason,
                    entry_static_labels=list(scene_range.entry_static_labels),
                    exit_static_labels=list(scene_range.exit_static_labels),
                    representative_static_entity_ids=metadata[
                        "representative_static_entity_ids"
                    ],
                    representative_static_labels=metadata[
                        "representative_static_labels"
                    ],
                    representative_dynamic_labels=metadata[
                        "representative_dynamic_labels"
                    ],
                    scene_kind_hint=metadata["scene_kind_hint"],
                )
            )
        for index, segment in enumerate(scene_segments):
            previous_scene_id = (
                scene_segments[index - 1].scene_id if index > 0 else None
            )
            next_scene_id = (
                scene_segments[index + 1].scene_id
                if index + 1 < len(scene_segments)
                else None
            )
            scene_segments[index] = segment.model_copy(
                update={
                    "previous_scene_id": previous_scene_id,
                    "next_scene_id": next_scene_id,
                }
            )
        return scene_segments

    def _build_scene_segment_metadata(
        self,
        *,
        scene_range,
        observations: list[Observation],
        entities: list[Entity],
    ) -> dict[str, Any]:
        entity_by_id = {
            entity.entity_id: entity
            for entity in entities
        }
        static_entity_frame_support: dict[str, set[int]] = defaultdict(set)
        static_label_frame_support: dict[str, set[int]] = defaultdict(set)
        dynamic_label_frame_support: dict[str, set[int]] = defaultdict(set)
        salient_label_frame_support: dict[str, set[int]] = defaultdict(set)

        for observation in observations:
            if observation.is_noise:
                continue
            if not (
                scene_range.start_frame
                <= observation.frame_index
                <= scene_range.end_frame
            ):
                continue
            if observation.is_static:
                static_entity_frame_support[observation.entity_id].add(
                    observation.frame_rank
                )
                static_label_frame_support[observation.label].add(
                    observation.frame_rank
                )
            else:
                entity = entity_by_id.get(observation.entity_id)
                dynamic_label = (
                    entity.category
                    if entity is not None
                    else observation.label
                )
                dynamic_label_frame_support[dynamic_label].add(
                    observation.frame_rank
                )
            for label_term in label_alias_terms(observation.label):
                if is_salient_label(label_term):
                    salient_label_frame_support[label_term].add(observation.frame_rank)
            entity = entity_by_id.get(observation.entity_id)
            if entity is not None:
                for label_term in label_alias_terms(entity.category):
                    if is_salient_label(label_term):
                        salient_label_frame_support[label_term].add(observation.frame_rank)

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
        salient_labels = [
            label
            for label, _ in sorted(
                salient_label_frame_support.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:4]
        ]
        representative_static_labels = _merge_scene_representative_labels(
            representative_static_labels,
            salient_labels,
            limit=8,
        )

        scene_kind_hint_parts = representative_static_labels[:3] or representative_dynamic_labels[:3]
        scene_kind_hint = (
            " / ".join(scene_kind_hint_parts)
            if scene_kind_hint_parts
            else None
        )

        return {
            "representative_static_entity_ids": representative_static_entity_ids,
            "representative_static_labels": representative_static_labels,
            "representative_dynamic_labels": representative_dynamic_labels,
            "scene_kind_hint": scene_kind_hint,
        }

    def _build_entity_segments(
        self,
        *,
        entities: list[Entity],
        observations: list[Observation],
        scene_segments: list[SceneSegment],
        frame_rank_lookup: dict[int, int],
    ) -> list[EntitySegment]:
        segments: list[EntitySegment] = []
        observations_by_entity: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            observations_by_entity[observation.entity_id].append(observation)

        max_presence_gap = self.max_entity_absence_tolerance_frames
        for entity in entities:
            entity_observations = sorted(
                observations_by_entity.get(entity.entity_id, []),
                key=lambda item: (
                    item.frame_rank,
                    item.frame_index,
                    item.timestamp_sec,
                    item.observation_id,
                ),
            )
            if not entity_observations:
                continue

            entity_segments: list[EntitySegment] = []
            presence_runs: list[list[Observation]] = []
            current_run: list[Observation] = []
            previous_frame_rank: int | None = None
            for observation in entity_observations:
                if (
                    previous_frame_rank is not None
                    and observation.frame_rank - previous_frame_rank > max_presence_gap
                    and current_run
                ):
                    presence_runs.append(current_run)
                    current_run = []
                current_run.append(observation)
                previous_frame_rank = observation.frame_rank
            if current_run:
                presence_runs.append(current_run)

            for run in presence_runs:
                current_scene_id: str | None = None
                current_scene_observations: list[Observation] = []
                for observation in run:
                    scene_segment = self._scene_segment_for_frame(
                        scene_segments=scene_segments,
                        frame_index=observation.frame_index,
                    )
                    scene_id = scene_segment.scene_id if scene_segment is not None else None
                    if current_scene_observations and scene_id != current_scene_id:
                        entity_segments.append(
                            self._build_entity_segment_from_observations(
                                entity=entity,
                                observations=current_scene_observations,
                                scene_id=current_scene_id,
                                frame_rank_lookup=frame_rank_lookup,
                            )
                        )
                        current_scene_observations = []
                    current_scene_observations.append(observation)
                    current_scene_id = scene_id

                if current_scene_observations:
                    entity_segments.append(
                        self._build_entity_segment_from_observations(
                            entity=entity,
                            observations=current_scene_observations,
                            scene_id=current_scene_id,
                            frame_rank_lookup=frame_rank_lookup,
                        )
                    )

            segments.extend(self._annotate_entity_segments(entity_segments))

        segments.sort(
            key=lambda item: (
                item.start_frame_rank,
                item.entity_id,
                item.segment_id,
            )
        )
        return segments

    def _annotate_entity_segments(
        self,
        segments: list[EntitySegment],
    ) -> list[EntitySegment]:
        if not segments:
            return []

        ordered = sorted(
            segments,
            key=lambda item: (
                item.start_frame_rank,
                item.end_frame_rank,
                item.segment_id,
            ),
        )
        segment_count = len(ordered)
        annotated: list[EntitySegment] = []
        for index, segment in enumerate(ordered, start=1):
            previous_segment = ordered[index - 2] if index > 1 else None
            next_segment = ordered[index] if index < segment_count else None

            gap_from_previous_frames: int | None = None
            gap_from_previous_ranks: int | None = None
            if previous_segment is not None:
                gap_from_previous_frames = max(
                    segment.start_frame - previous_segment.end_frame - 1,
                    0,
                )
                gap_from_previous_ranks = max(
                    segment.start_frame_rank - previous_segment.end_frame_rank - 1,
                    0,
                )

            gap_to_next_frames: int | None = None
            gap_to_next_ranks: int | None = None
            if next_segment is not None:
                gap_to_next_frames = max(
                    next_segment.start_frame - segment.end_frame - 1,
                    0,
                )
                gap_to_next_ranks = max(
                    next_segment.start_frame_rank - segment.end_frame_rank - 1,
                    0,
                )

            annotated.append(
                segment.model_copy(
                    update={
                        "segment_index": index,
                        "segment_count": segment_count,
                        "previous_segment_id": (
                            previous_segment.segment_id
                            if previous_segment is not None
                            else None
                        ),
                        "next_segment_id": (
                            next_segment.segment_id if next_segment is not None else None
                        ),
                        "gap_from_previous_frames": gap_from_previous_frames,
                        "gap_to_next_frames": gap_to_next_frames,
                        "gap_from_previous_ranks": gap_from_previous_ranks,
                        "gap_to_next_ranks": gap_to_next_ranks,
                        "resumes_after_gap": (
                            previous_segment is not None
                            and gap_from_previous_ranks is not None
                            and gap_from_previous_ranks > 0
                        ),
                    }
                )
            )
        return annotated

    def _build_entity_segment_from_observations(
        self,
        *,
        entity: Entity,
        observations: list[Observation],
        scene_id: str | None,
        frame_rank_lookup: dict[int, int],
    ) -> EntitySegment:
        ordered = sorted(
            observations,
            key=lambda item: (
                item.frame_rank,
                item.frame_index,
                item.timestamp_sec,
                item.observation_id,
            ),
        )
        start = ordered[0]
        end = ordered[-1]
        scene_suffix = (
            scene_id.rsplit(":", maxsplit=1)[-1]
            if scene_id is not None
            else "global"
        )
        return EntitySegment(
            segment_id=(
                f"{entity.entity_id}:segment:{scene_suffix}:"
                f"{start.frame_index:06d}:{end.frame_index:06d}"
            ),
            video_id=self.video_id,
            entity_id=entity.entity_id,
            scene_id=scene_id,
            start_frame_rank=frame_rank_lookup[start.frame_index],
            end_frame_rank=frame_rank_lookup[end.frame_index],
            start_frame=start.frame_index,
            end_frame=end.frame_index,
            start_ts=start.timestamp_sec,
            end_ts=end.timestamp_sec,
            support_frames=len({observation.frame_rank for observation in ordered}),
            is_static=entity.is_static,
            is_noise=entity.is_noise,
        )

    def _build_scene_base_graph(
        self,
        *,
        entities: list[Entity],
        observations: list[Observation],
        scene_segments: list[SceneSegment],
    ) -> tuple[list[SceneBaseNode], list[SceneBaseEdge]]:
        if not self.scene_base_graph_config.enabled:
            return [], []

        entity_lookup = {
            entity.entity_id: entity
            for entity in entities
        }
        observations_by_scene: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            if not observation.is_static or observation.is_noise:
                continue
            scene_segment = self._scene_segment_for_frame(
                scene_segments=scene_segments,
                frame_index=observation.frame_index,
            )
            if scene_segment is None:
                continue
            observations_by_scene[scene_segment.scene_id].append(observation)

        scene_lookup = {
            scene.scene_id: scene
            for scene in scene_segments
        }
        scene_base_nodes: list[SceneBaseNode] = []
        scene_base_edges: list[SceneBaseEdge] = []

        for scene_id, scene_observations in observations_by_scene.items():
            scene_segment = scene_lookup[scene_id]
            grouped_by_entity: dict[str, list[Observation]] = defaultdict(list)
            for observation in scene_observations:
                grouped_by_entity[observation.entity_id].append(observation)

            for entity_id, entity_observations in grouped_by_entity.items():
                entity = entity_lookup.get(entity_id)
                if entity is None:
                    continue
                support_frames = len(
                    {
                        observation.frame_rank
                        for observation in entity_observations
                    }
                )
                coverage_ratio = support_frames / max(scene_segment.support_frames, 1)
                if (
                    support_frames < self.scene_base_graph_config.min_anchor_support_frames
                    or coverage_ratio
                    < self.scene_base_graph_config.min_anchor_coverage_ratio
                ):
                    continue

                first_frame = min(
                    observation.frame_index
                    for observation in entity_observations
                )
                last_frame = max(
                    observation.frame_index
                    for observation in entity_observations
                )
                scene_base_node = SceneBaseNode(
                    node_id=f"{scene_id}:base:{entity_id}",
                    scene_id=scene_id,
                    video_id=self.video_id,
                    entity_id=entity_id,
                    label=entity.category,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    support_frames=support_frames,
                    coverage_ratio=coverage_ratio,
                    canonical_image_path=entity.canonical_image_path,
                    parent_labels_seen=entity.parent_labels_seen,
                    states_seen=entity.states_seen,
                )
                scene_base_nodes.append(scene_base_node)
                scene_base_edges.append(
                    SceneBaseEdge(
                        edge_id=f"{scene_id}:contains:{entity_id}",
                        scene_id=scene_id,
                        video_id=self.video_id,
                        predicate="SCENE_CONTAINS",
                        source_id=scene_id,
                        target_id=scene_base_node.node_id,
                        entity_id=entity_id,
                        support_frames=support_frames,
                        coverage_ratio=coverage_ratio,
                    )
                )

        scene_base_nodes.sort(key=lambda item: (item.scene_id, item.label, item.entity_id))
        scene_base_edges.sort(key=lambda item: (item.scene_id, item.target_id))
        return scene_base_nodes, scene_base_edges

    def _annotate_observation_origins(
        self,
        *,
        observations: list[Observation],
        scene_segments: list[SceneSegment],
        scene_base_nodes: list[SceneBaseNode],
    ) -> list[Observation]:
        anchored_entities_by_scene: dict[str, set[str]] = defaultdict(set)
        for node in scene_base_nodes:
            anchored_entities_by_scene[node.scene_id].add(node.entity_id)

        updated: list[Observation] = []
        for observation in observations:
            scene_segment = self._scene_segment_for_frame(
                scene_segments=scene_segments,
                frame_index=observation.frame_index,
            )
            scene_id = scene_segment.scene_id if scene_segment is not None else None
            graph_origin = "SCENE_DELTA"
            if (
                scene_id is not None
                and observation.entity_id in anchored_entities_by_scene.get(scene_id, set())
            ):
                graph_origin = "SCENE_BASE"
            updated.append(
                observation.model_copy(
                    update={
                        "scene_id": scene_id,
                        "graph_origin": graph_origin,
                    }
                )
            )
        return updated

    def _annotate_entity_origins(
        self,
        *,
        entities: list[Entity],
        scene_base_nodes: list[SceneBaseNode],
    ) -> list[Entity]:
        anchored_entity_ids = {
            node.entity_id
            for node in scene_base_nodes
        }
        updated: list[Entity] = []
        for entity in entities:
            if entity.entity_id in anchored_entity_ids and entity.is_static:
                graph_origin = "SCENE_BASE"
            elif entity.entity_id in anchored_entity_ids:
                graph_origin = "HYBRID"
            else:
                graph_origin = "SCENE_DELTA"
            updated.append(
                entity.model_copy(update={"graph_origin": graph_origin})
            )
        return updated

    def _build_scene_delta_graph(
        self,
        *,
        entities: list[Entity],
        entity_segments: list[EntitySegment],
        scene_base_nodes: list[SceneBaseNode],
        state_episodes: list[StateEpisode],
        relation_episodes: list[RelationEpisode],
        transition_events: list[TransitionEvent],
    ) -> tuple[list[SceneDeltaNode], list[SceneDeltaEdge]]:
        entity_lookup = {
            entity.entity_id: entity
            for entity in entities
        }
        anchored_entities_by_scene: dict[str, set[str]] = defaultdict(set)
        for node in scene_base_nodes:
            anchored_entities_by_scene[node.scene_id].add(node.entity_id)

        delta_nodes: list[SceneDeltaNode] = []
        delta_edges: list[SceneDeltaEdge] = []

        for segment in entity_segments:
            if segment.scene_id is None:
                continue
            if segment.entity_id in anchored_entities_by_scene.get(segment.scene_id, set()):
                continue

            entity = entity_lookup.get(segment.entity_id)
            if entity is None or entity.is_noise:
                continue

            transition_event_count = len(
                {
                    event.event_id
                    for event in transition_events
                    if segment.entity_id in self._transition_event_entity_ids(event)
                    and segment.start_frame <= event.frame_index <= segment.end_frame
                }
            )
            relation_episode_count = len(
                {
                    episode.relation_episode_id
                    for episode in relation_episodes
                    if (
                        segment.entity_id == episode.subject_entity_id
                        or segment.entity_id == episode.object_entity_id
                    )
                    and not (
                        episode.end_frame < segment.start_frame
                        or episode.start_frame > segment.end_frame
                    )
                }
            )
            state_episode_count = len(
                {
                    episode.state_episode_id
                    for episode in state_episodes
                    if episode.entity_id == segment.entity_id
                    and not (
                        episode.end_frame < segment.start_frame
                        or episode.start_frame > segment.end_frame
                    )
                }
            )
            delta_node_id = self._scene_delta_node_id(
                scene_id=segment.scene_id,
                entity_id=segment.entity_id,
                segment_id=segment.segment_id,
            )
            delta_node = SceneDeltaNode(
                node_id=delta_node_id,
                scene_id=segment.scene_id,
                video_id=self.video_id,
                entity_id=segment.entity_id,
                label=entity.category,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                start_ts=segment.start_ts,
                end_ts=segment.end_ts,
                support_frames=segment.support_frames,
                transition_event_count=transition_event_count,
                relation_episode_count=relation_episode_count,
                state_episode_count=state_episode_count,
            )
            delta_nodes.append(delta_node)
            delta_edges.append(
                SceneDeltaEdge(
                    edge_id=f"{segment.scene_id}:updates:{segment.entity_id}:{segment.segment_id}",
                    scene_id=segment.scene_id,
                    video_id=self.video_id,
                    predicate="SCENE_UPDATES",
                    source_id=segment.scene_id,
                    target_id=delta_node.node_id,
                    entity_id=segment.entity_id,
                    support_frames=segment.support_frames,
                    transition_event_count=transition_event_count,
                )
            )

        delta_nodes.sort(key=lambda item: (item.scene_id, item.label, item.entity_id, item.node_id))
        delta_edges.sort(key=lambda item: (item.scene_id, item.target_id, item.edge_id))
        return delta_nodes, delta_edges

    def _build_scene_delta_observations(
        self,
        *,
        observations: list[Observation],
        entity_segments: list[EntitySegment],
        scene_base_nodes: list[SceneBaseNode],
        transition_events: list[TransitionEvent],
    ) -> list[SceneDeltaObservation]:
        anchored_entities_by_scene: dict[str, set[str]] = defaultdict(set)
        for node in scene_base_nodes:
            anchored_entities_by_scene[node.scene_id].add(node.entity_id)

        observations_by_entity: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            observations_by_entity[observation.entity_id].append(observation)
        for entity_id, entity_observations in observations_by_entity.items():
            entity_observations.sort(
                key=lambda item: (item.frame_index, item.frame_rank, item.observation_id)
            )

        events_by_entity_and_frame: dict[tuple[str, int], list[TransitionEvent]] = defaultdict(list)
        for event in transition_events:
            for entity_id in self._transition_event_entity_ids(event):
                events_by_entity_and_frame[(entity_id, event.frame_index)].append(event)

        delta_observations: list[SceneDeltaObservation] = []
        seen_delta_observation_ids: set[str] = set()

        for segment in entity_segments:
            if segment.scene_id is None:
                continue
            if segment.entity_id in anchored_entities_by_scene.get(segment.scene_id, set()):
                continue

            interesting_frames = {segment.start_frame, segment.end_frame}
            for event in events_by_entity_and_frame.get((segment.entity_id, segment.start_frame), []):
                if segment.start_frame <= event.frame_index <= segment.end_frame:
                    interesting_frames.add(event.frame_index)
            for event_key, events in events_by_entity_and_frame.items():
                entity_id, frame_index = event_key
                if entity_id != segment.entity_id:
                    continue
                if segment.start_frame <= frame_index <= segment.end_frame:
                    interesting_frames.add(frame_index)

            scene_delta_node_id = self._scene_delta_node_id(
                scene_id=segment.scene_id,
                entity_id=segment.entity_id,
                segment_id=segment.segment_id,
            )
            for observation in observations_by_entity.get(segment.entity_id, []):
                if observation.scene_id != segment.scene_id:
                    continue
                if observation.is_noise or observation.graph_origin != "SCENE_DELTA":
                    continue
                if observation.frame_index not in interesting_frames:
                    continue
                if not (segment.start_frame <= observation.frame_index <= segment.end_frame):
                    continue

                trigger_events = sorted(
                    events_by_entity_and_frame.get(
                        (segment.entity_id, observation.frame_index),
                        [],
                    ),
                    key=lambda item: (item.event_type, item.event_id),
                )
                delta_observation_id = f"{scene_delta_node_id}:{observation.frame_index:06d}"
                if delta_observation_id in seen_delta_observation_ids:
                    continue
                seen_delta_observation_ids.add(delta_observation_id)
                delta_observations.append(
                    SceneDeltaObservation(
                        delta_observation_id=delta_observation_id,
                        scene_id=segment.scene_id,
                        video_id=self.video_id,
                        entity_id=segment.entity_id,
                        scene_delta_node_id=scene_delta_node_id,
                        source_segment_id=segment.segment_id,
                        source_observation_id=observation.observation_id,
                        frame_name=observation.frame_name,
                        frame_rank=observation.frame_rank,
                        frame_index=observation.frame_index,
                        timestamp_sec=observation.timestamp_sec,
                        image_path=observation.image_path,
                        bbox=observation.bbox,
                        label=observation.label,
                        trigger_event_ids=[event.event_id for event in trigger_events],
                        trigger_event_types=sorted(
                            {event.event_type for event in trigger_events}
                        ),
                        marks_segment_start=observation.frame_index == segment.start_frame,
                        marks_segment_end=observation.frame_index == segment.end_frame,
                    )
                )

        delta_observations.sort(
            key=lambda item: (
                item.scene_id,
                item.frame_index,
                item.entity_id,
                item.delta_observation_id,
            )
        )
        return delta_observations

    def _scene_delta_node_id(
        self,
        *,
        scene_id: str,
        entity_id: str,
        segment_id: str,
    ) -> str:
        return f"{scene_id}:delta:{entity_id}:{segment_id}"

    def _scene_segment_for_frame(
        self,
        *,
        scene_segments: list[SceneSegment],
        frame_index: int,
    ) -> SceneSegment | None:
        for scene_segment in scene_segments:
            if scene_segment.start_frame <= frame_index <= scene_segment.end_frame:
                return scene_segment
        return None

    def _build_transition_events(
        self,
        *,
        entities: list[Entity],
        state_episodes: list[StateEpisode],
        relation_episodes: list[RelationEpisode],
        transient_relation_episodes: list[RelationEpisode],
        entity_segments: list[EntitySegment],
        frame_rank_lookup: dict[int, int],
    ) -> list[TransitionEvent]:
        events: list[TransitionEvent] = []
        entity_lookup = {
            entity.entity_id: entity
            for entity in entities
        }
        segments_by_entity: dict[str, list[EntitySegment]] = defaultdict(list)
        for segment in entity_segments:
            segments_by_entity[segment.entity_id].append(segment)
        for entity_id, segments in segments_by_entity.items():
            segments.sort(key=lambda item: (item.start_frame, item.end_frame, item.segment_id))

        for entity_id, segments in segments_by_entity.items():
            entity = entity_lookup.get(entity_id)
            min_support_frames = self._entity_segment_event_min_support(entity)
            for segment in segments:
                if segment.support_frames < min_support_frames:
                    continue
                if self._segment_starts_new_presence(segment):
                    events.append(
                        TransitionEvent(
                            event_id=f"{segment.segment_id}:APPEAR:{segment.start_frame:06d}",
                            video_id=self.video_id,
                            event_type="APPEAR",
                            frame_rank=frame_rank_lookup[segment.start_frame],
                            frame_index=segment.start_frame,
                            timestamp_sec=segment.start_ts,
                            entity_id=entity_id,
                            source_segment_id=segment.segment_id,
                        )
                    )
                if not self._segment_ends_presence(segment):
                    continue
                if (
                    segment.start_frame == segment.end_frame
                    and min_support_frames > 1
                ):
                    continue
                events.append(
                    TransitionEvent(
                        event_id=(
                            f"{segment.segment_id}:DISAPPEAR:{segment.end_frame:06d}"
                        ),
                        video_id=self.video_id,
                        event_type="DISAPPEAR",
                        frame_rank=frame_rank_lookup[segment.end_frame],
                        frame_index=segment.end_frame,
                        timestamp_sec=segment.end_ts,
                        entity_id=entity_id,
                        source_segment_id=segment.segment_id,
                    )
                )

        state_episode_groups: dict[str, list[StateEpisode]] = defaultdict(list)
        for episode in state_episodes:
            state_episode_groups[episode.entity_id].append(episode)

        for entity_id, episodes in state_episode_groups.items():
            ordered = sorted(
                episodes,
                key=lambda item: (item.start_frame, item.end_frame, item.attr_value),
            )
            grouped_by_start_frame: dict[int, list[StateEpisode]] = defaultdict(list)
            for episode in ordered:
                grouped_by_start_frame[episode.start_frame].append(episode)

            representative_groups: list[StateEpisode] = []
            for start_frame in sorted(grouped_by_start_frame.keys()):
                current_group = grouped_by_start_frame[start_frame]
                representative = self._select_state_transition_representative(
                    current_group
                )
                if representative is None:
                    continue
                representative_groups.append(representative)

            previous_episode: StateEpisode | None = None
            for episode in representative_groups:
                if previous_episode is None:
                    previous_episode = episode
                    continue

                previous_value = previous_episode.attr_value
                current_value = episode.attr_value
                if previous_value == current_value:
                    previous_episode = episode
                    continue
                frame_gap = episode.start_frame - previous_episode.end_frame
                if frame_gap < 1 or frame_gap > self.state_transition_max_gap_frames:
                    previous_episode = episode
                    continue

                previous_segment_id = self._entity_segment_id_for_frame(
                    segments=segments_by_entity.get(entity_id, []),
                    frame_index=previous_episode.start_frame,
                )
                current_segment_id = self._entity_segment_id_for_frame(
                    segments=segments_by_entity.get(entity_id, []),
                    frame_index=episode.start_frame,
                )
                if previous_segment_id != current_segment_id:
                    previous_episode = episode
                    continue
                events.append(
                    TransitionEvent(
                        event_id=(
                            f"{entity_id}:STATE_TRANSITION:{episode.start_frame:06d}:"
                            f"{previous_value.replace(' ', '_')}:"
                            f"{current_value.replace(' ', '_')}"
                        ),
                        video_id=self.video_id,
                        event_type="STATE_TRANSITION",
                        frame_rank=frame_rank_lookup[episode.start_frame],
                        frame_index=episode.start_frame,
                        timestamp_sec=episode.start_ts,
                        entity_id=entity_id,
                        attr_key=episode.attr_key,
                        previous_state_value=previous_value,
                        current_state_value=current_value,
                        source_segment_id=self._entity_segment_id_for_frame(
                            segments=segments_by_entity.get(entity_id, []),
                            frame_index=episode.start_frame,
                        ),
                        source_episode_id=episode.state_episode_id,
                    )
                )
                previous_episode = episode

        for episode in relation_episodes:
            if (
                episode.support_frames < self.min_relation_event_support_frames
                or episode.start_frame == episode.end_frame
            ):
                continue
            events.append(
                TransitionEvent(
                    event_id=(
                        f"{episode.relation_episode_id}:RELATION_START:"
                        f"{episode.start_frame:06d}"
                    ),
                    video_id=self.video_id,
                    event_type="RELATION_START",
                    frame_rank=frame_rank_lookup[episode.start_frame],
                    frame_index=episode.start_frame,
                    timestamp_sec=episode.start_ts,
                    subject_entity_id=episode.subject_entity_id,
                    predicate=episode.predicate,
                    raw_predicate=episode.raw_predicate,
                    object_entity_id=episode.object_entity_id,
                    source_episode_id=episode.relation_episode_id,
                )
            )
            events.append(
                TransitionEvent(
                    event_id=(
                        f"{episode.relation_episode_id}:RELATION_END:"
                        f"{episode.end_frame:06d}"
                    ),
                    video_id=self.video_id,
                    event_type="RELATION_END",
                    frame_rank=frame_rank_lookup[episode.end_frame],
                    frame_index=episode.end_frame,
                    timestamp_sec=episode.end_ts,
                    subject_entity_id=episode.subject_entity_id,
                    predicate=episode.predicate,
                    raw_predicate=episode.raw_predicate,
                    object_entity_id=episode.object_entity_id,
                    source_episode_id=episode.relation_episode_id,
                )
            )

        for episode in transient_relation_episodes:
            events.append(
                TransitionEvent(
                    event_id=(
                        f"{episode.relation_episode_id}:RELATION_OBSERVED:"
                        f"{episode.start_frame:06d}"
                    ),
                    video_id=self.video_id,
                    event_type="RELATION_OBSERVED",
                    frame_rank=frame_rank_lookup[episode.start_frame],
                    frame_index=episode.start_frame,
                    timestamp_sec=episode.start_ts,
                    subject_entity_id=episode.subject_entity_id,
                    predicate=episode.predicate,
                    raw_predicate=episode.raw_predicate,
                    object_entity_id=episode.object_entity_id,
                    source_episode_id=episode.relation_episode_id,
                )
            )

        events.sort(
            key=lambda item: (
                item.frame_rank,
                item.frame_index,
                item.timestamp_sec,
                item.event_type,
                item.event_id,
            )
        )
        return events

    def _segment_starts_new_presence(self, segment: EntitySegment) -> bool:
        gap = segment.gap_from_previous_ranks
        return gap is None or gap > 0

    def _segment_ends_presence(self, segment: EntitySegment) -> bool:
        gap = segment.gap_to_next_ranks
        return gap is None or gap > 0

    def _entity_segment_event_min_support(self, entity: Entity | None) -> int:
        if entity is None:
            return self.min_entity_segment_event_support_frames
        normalized_label = self._normalize_label(entity.category)
        return self.entity_segment_event_support_overrides.get(
            normalized_label,
            self.min_entity_segment_event_support_frames,
        )

    def _select_state_transition_representative(
        self,
        episodes: list[StateEpisode],
    ) -> StateEpisode | None:
        if not episodes:
            return None
        ordered = sorted(
            episodes,
            key=lambda item: (
                -item.support_frames,
                -(item.end_frame - item.start_frame),
                item.attr_value,
            ),
        )
        representative = ordered[0]
        if representative.support_frames < self.min_state_transition_support_frames:
            return None
        if len(ordered) == 1:
            return representative

        second_support = ordered[1].support_frames
        if (
            representative.support_frames
            < second_support + self.state_transition_min_dominance_margin_frames
        ):
            return None

        total_support = sum(item.support_frames for item in ordered)
        if total_support <= 0:
            return None
        dominance_share = representative.support_frames / total_support
        if dominance_share < self.state_transition_min_dominance_share:
            return None
        return representative

    def _entity_segment_id_for_frame(
        self,
        *,
        segments: list[EntitySegment],
        frame_index: int,
    ) -> str | None:
        for segment in segments:
            if segment.start_frame <= frame_index <= segment.end_frame:
                return segment.segment_id
        if not segments:
            return None
        return segments[0].segment_id

    def _transition_event_entity_ids(
        self,
        event: TransitionEvent,
    ) -> set[str]:
        return {
            value
            for value in (
                event.entity_id,
                event.subject_entity_id,
                event.object_entity_id,
            )
            if value is not None
        }

    def _build_state_episodes(
        self,
        observations: list[Observation],
    ) -> list[StateEpisode]:
        state_facts: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)

        for observation in observations:
            if observation.is_static or observation.is_noise:
                continue
            if self.state_subject_labels and observation.label not in self.state_subject_labels:
                continue
            for state_value in observation.state_values:
                if not self._should_keep_state_value(state_value):
                    continue
                state_facts[(observation.entity_id, state_value)].append(
                    (
                        observation.frame_rank,
                        observation.frame_index,
                        observation.timestamp_sec,
                    )
                )

        episodes: list[StateEpisode] = []
        for (entity_id, state_value), frame_points in state_facts.items():
            frame_points.sort(key=lambda item: item[0])
            current: StateEpisode | None = None
            current_end_rank: int | None = None

            for frame_rank, frame_index, timestamp_sec in frame_points:
                if current is None:
                    current = self._new_state_episode(
                        entity_id=entity_id,
                        state_value=state_value,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                    current_end_rank = frame_rank
                    continue

                contiguous = (
                    current_end_rank is not None
                    and frame_rank - current_end_rank - 1
                    <= self._state_episode_missing_rank_gap()
                )
                if contiguous:
                    current = current.model_copy(
                        update={
                            "end_frame": frame_index,
                            "end_ts": timestamp_sec,
                            "support_frames": current.support_frames + 1,
                        }
                    )
                    current_end_rank = frame_rank
                    continue

                episodes.append(current)
                current = self._new_state_episode(
                    entity_id=entity_id,
                    state_value=state_value,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                )
                current_end_rank = frame_rank

            if current is not None:
                episodes.append(current)

        episodes = self._debounce_state_episodes(episodes)
        episodes.sort(key=lambda item: (item.start_frame, item.entity_id, item.attr_value))
        return episodes

    def _state_episode_missing_rank_gap(self) -> int:
        return max(0, self.episode_max_frame_gap)

    def _new_state_episode(
        self,
        *,
        entity_id: str,
        state_value: str,
        frame_index: int,
        timestamp_sec: float,
    ) -> StateEpisode:
        safe_state = state_value.replace(" ", "_")
        return StateEpisode(
            state_episode_id=(
                f"{entity_id}:state:{frame_index:06d}:{safe_state}"
            ),
            video_id=self.video_id,
            entity_id=entity_id,
            attr_value=state_value,
            start_frame=frame_index,
            end_frame=frame_index,
            start_ts=timestamp_sec,
            end_ts=timestamp_sec,
            support_frames=1,
        )

    def _build_relation_episodes(
        self,
        *,
        frame_metas: list[_FrameMeta],
        relations_table: dict[str, Any],
        observation_lookup: dict[tuple[str, int], Observation],
    ) -> tuple[list[RelationEpisode], list[RelationEpisode]]:
        relation_facts: dict[
            tuple[str, str, str], dict[int, dict[str, Any]]
        ] = defaultdict(dict)

        for frame_rank, meta in enumerate(frame_metas):
            for relation_item in relations_table.get(meta.frame_name, []):
                subject_idx, object_idx = relation_item["idx"]
                subject_observation = observation_lookup.get(
                    (meta.frame_name, int(subject_idx))
                )
                object_observation = observation_lookup.get(
                    (meta.frame_name, int(object_idx))
                )
                if subject_observation is None or object_observation is None:
                    continue
                predicate, raw_predicate = self._map_relation_predicate(
                    relation_item["predicate"]
                )
                resolved = self._resolve_relation_assignment(
                    subject_observation=subject_observation,
                    object_observation=object_observation,
                    predicate=predicate,
                    raw_predicate=raw_predicate,
                )
                if resolved is None:
                    continue
                subject_observation, predicate, raw_predicate, object_observation = resolved
                if not self._should_keep_relation_episode(
                    subject_observation=subject_observation,
                    object_observation=object_observation,
                    predicate=predicate,
                ):
                    continue

                key = (
                    subject_observation.entity_id,
                    predicate,
                    object_observation.entity_id,
                )
                frame_bucket = relation_facts[key].setdefault(
                    frame_rank,
                    {
                        "frame_index": meta.frame_index,
                        "timestamp_sec": meta.timestamp_sec,
                        "raw_predicates": set(),
                    },
                )
                frame_bucket["raw_predicates"].add(raw_predicate)

        episodes: list[RelationEpisode] = []
        for (
            subject_entity_id,
            predicate,
            object_entity_id,
        ), frame_points in relation_facts.items():
            ordered_frame_points = sorted(frame_points.items(), key=lambda item: item[0])
            current: RelationEpisode | None = None
            current_end_rank: int | None = None

            for frame_rank, frame_payload in ordered_frame_points:
                frame_index = int(frame_payload["frame_index"])
                timestamp_sec = float(frame_payload["timestamp_sec"])
                raw_predicate = self._merge_metadata_values(
                    *sorted(frame_payload["raw_predicates"])
                )
                if current is None:
                    current = self._new_relation_episode(
                        subject_entity_id=subject_entity_id,
                        predicate=predicate,
                        raw_predicate=raw_predicate,
                        object_entity_id=object_entity_id,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                    current_end_rank = frame_rank
                    continue

                contiguous = (
                    current_end_rank is not None
                    and (
                        frame_rank - current_end_rank - 1
                        <= self._relation_episode_missing_rank_gap(predicate)
                    )
                )
                if contiguous:
                    current = current.model_copy(
                        update={
                            "end_frame": frame_index,
                            "end_ts": timestamp_sec,
                            "support_frames": current.support_frames + 1,
                            "raw_predicate": self._merge_metadata_values(
                                current.raw_predicate,
                                raw_predicate,
                            ),
                        }
                    )
                    current_end_rank = frame_rank
                    continue

                episodes.append(current)
                current = self._new_relation_episode(
                    subject_entity_id=subject_entity_id,
                    predicate=predicate,
                    raw_predicate=raw_predicate,
                    object_entity_id=object_entity_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                )
                current_end_rank = frame_rank

            if current is not None:
                episodes.append(current)

        episodes = self._debounce_relation_episodes(episodes)
        durable_episodes: list[RelationEpisode] = []
        transient_episodes: list[RelationEpisode] = []
        for episode in episodes:
            predicate_min_support = self._relation_episode_min_support(
                episode.predicate
            )
            if episode.support_frames < predicate_min_support:
                continue
            if episode.support_frames >= self.min_relation_episode_support_frames:
                durable_episodes.append(episode)
                continue
            transient_episodes.append(episode)
        durable_episodes.sort(
            key=lambda item: (item.start_frame, item.subject_entity_id, item.predicate)
        )
        transient_episodes.sort(
            key=lambda item: (item.start_frame, item.subject_entity_id, item.predicate)
        )
        return durable_episodes, transient_episodes

    def _debounce_state_episodes(
        self,
        episodes: list[StateEpisode],
    ) -> list[StateEpisode]:
        if self.min_state_episode_length_frames <= 1 or len(episodes) < 3:
            return episodes

        smoothed: list[StateEpisode] = []
        grouped: dict[tuple[str, str], list[StateEpisode]] = defaultdict(list)
        for episode in episodes:
            grouped[(episode.entity_id, episode.attr_key)].append(episode)

        for key in sorted(grouped.keys()):
            sequence = sorted(
                grouped[key],
                key=lambda item: (
                    self._episode_start_rank(item),
                    self._episode_end_rank(item),
                    item.attr_value,
                    item.state_episode_id,
                ),
            )
            smoothed.extend(
                self._debounce_episode_sequence(
                    sequence=sequence,
                    min_length_frames=self.min_state_episode_length_frames,
                    same_value=lambda left, right: left.attr_value == right.attr_value,
                    merge=self._merge_state_episode_pair,
                    can_bridge=self._can_bridge_state_blip,
                )
            )

        return smoothed

    def _debounce_relation_episodes(
        self,
        episodes: list[RelationEpisode],
    ) -> list[RelationEpisode]:
        if self.min_relation_episode_length_frames <= 1 or len(episodes) < 3:
            return episodes

        smoothed: list[RelationEpisode] = []
        grouped: dict[tuple[str, str], list[RelationEpisode]] = defaultdict(list)
        for episode in episodes:
            grouped[(episode.subject_entity_id, episode.object_entity_id)].append(episode)

        for key in sorted(grouped.keys()):
            sequence = sorted(
                grouped[key],
                key=lambda item: (
                    self._episode_start_rank(item),
                    self._episode_end_rank(item),
                    item.predicate,
                    item.relation_episode_id,
                ),
            )
            smoothed.extend(
                self._debounce_episode_sequence(
                    sequence=sequence,
                    min_length_frames=self.min_relation_episode_length_frames,
                    same_value=lambda left, right: left.predicate == right.predicate,
                    merge=self._merge_relation_episode_pair,
                    can_bridge=self._can_bridge_relation_blip,
                )
            )

        return smoothed

    def _debounce_episode_sequence(
        self,
        *,
        sequence: list[Any],
        min_length_frames: int,
        same_value,
        merge,
        can_bridge,
    ) -> list[Any]:
        updated = list(sequence)
        changed = True
        while changed and len(updated) >= 3:
            changed = False
            rebuilt: list[Any] = []
            index = 0
            while index < len(updated):
                if (
                    rebuilt
                    and index + 1 < len(updated)
                ):
                    previous = rebuilt[-1]
                    current = updated[index]
                    following = updated[index + 1]
                    if (
                        current.support_frames < min_length_frames
                        and same_value(previous, following)
                        and can_bridge(previous, current, following)
                    ):
                        rebuilt[-1] = merge(previous, following)
                        index += 2
                        changed = True
                        continue
                rebuilt.append(updated[index])
                index += 1
            updated = rebuilt
        return updated

    def _can_bridge_state_blip(
        self,
        previous: StateEpisode,
        current: StateEpisode,
        following: StateEpisode,
    ) -> bool:
        if previous.attr_value == current.attr_value:
            return False
        if current.attr_value == following.attr_value:
            return False
        if previous.entity_id != current.entity_id or current.entity_id != following.entity_id:
            return False
        return (
            self._episodes_are_adjacent(previous, current)
            and self._episodes_are_adjacent(current, following)
        )

    def _can_bridge_relation_blip(
        self,
        previous: RelationEpisode,
        current: RelationEpisode,
        following: RelationEpisode,
    ) -> bool:
        if previous.predicate == current.predicate:
            return False
        if current.predicate == following.predicate:
            return False
        if previous.subject_entity_id != current.subject_entity_id:
            return False
        if previous.object_entity_id != current.object_entity_id:
            return False
        if current.subject_entity_id != following.subject_entity_id:
            return False
        if current.object_entity_id != following.object_entity_id:
            return False
        return (
            self._episodes_are_adjacent(previous, current)
            and self._episodes_are_adjacent(current, following)
        )

    def _merge_state_episode_pair(
        self,
        previous: StateEpisode,
        following: StateEpisode,
    ) -> StateEpisode:
        return previous.model_copy(
            update={
                "end_frame": following.end_frame,
                "end_ts": following.end_ts,
                "support_frames": previous.support_frames + following.support_frames,
            }
        )

    def _merge_relation_episode_pair(
        self,
        previous: RelationEpisode,
        following: RelationEpisode,
    ) -> RelationEpisode:
        return previous.model_copy(
            update={
                "end_frame": following.end_frame,
                "end_ts": following.end_ts,
                "support_frames": previous.support_frames + following.support_frames,
                "raw_predicate": self._merge_metadata_values(
                    previous.raw_predicate,
                    following.raw_predicate,
                ),
            }
        )

    def _episodes_are_adjacent(
        self,
        left: StateEpisode | RelationEpisode,
        right: StateEpisode | RelationEpisode,
    ) -> bool:
        return self._episode_start_rank(right) - self._episode_end_rank(left) == 1

    def _episode_start_rank(
        self,
        episode: StateEpisode | RelationEpisode,
    ) -> int:
        return self._frame_rank_by_index.get(episode.start_frame, episode.start_frame)

    def _episode_end_rank(
        self,
        episode: StateEpisode | RelationEpisode,
    ) -> int:
        return self._frame_rank_by_index.get(episode.end_frame, episode.end_frame)

    def _resolve_relation_assignment(
        self,
        *,
        subject_observation: Observation,
        object_observation: Observation,
        predicate: str,
        raw_predicate: str,
    ) -> tuple[Observation, str, str, Observation] | None:
        if not self.enable_semantic_relation_assignment_cleanup:
            return subject_observation, predicate, raw_predicate, object_observation

        raw = raw_predicate

        if any(
            marker in raw
            for marker in (
                " are both ",
                " both play ",
                " both wear ",
                " play basketball together",
            )
        ):
            return None

        if predicate == "WEARING":
            return None

        reverse_markers = (
            " held by ",
            " blocked by ",
            " chased by ",
            " pushed by ",
        )
        if any(marker in f" {raw} " for marker in reverse_markers):
            subject_observation, object_observation = object_observation, subject_observation

        human_like_labels = self.component_host_labels or {
            "person",
            "man",
            "woman",
            "player",
            "boy",
            "girl",
        }
        human_first_predicates = {
            "HOLDING",
            "LOOKING_AT",
            "RECEIVING_FROM",
            "PASSING_TO",
            "DRIBBLING_PAST",
            "GUARDING",
            "MOVING_TOWARD",
            "SHOOTING",
            "SHOOTING_TOWARD",
            "BLOCKING_SHOT_FROM",
            "INTERACTING_WITH",
            "PLAYING_WITH",
            "COMPETING_WITH",
            "RUNNING_PAST",
            "FACING_AWAY_FROM",
            "WALKING_AWAY_FROM",
            "GESTURING_TOWARD",
            "BEING_CHASED_BY",
            "BEING_PUSHED_BY",
            "SHOOTING_OVER",
        }
        if (
            predicate in human_first_predicates
            and subject_observation.label not in human_like_labels
            and object_observation.label in human_like_labels
        ):
            subject_observation, object_observation = object_observation, subject_observation

        if (
            predicate == "UNMAPPED_RELATION"
            and subject_observation.label == "bottle"
            and object_observation.label in human_like_labels
            and (" wear" in raw or " held by " in f" {raw} ")
        ):
            return None

        return subject_observation, predicate, raw_predicate, object_observation

    def _map_relation_predicate(self, predicate: str) -> tuple[str, str]:
        normalized = self._normalize_relation_text(predicate)
        if not self.enable_relation_predicate_mapping:
            return self._raw_relation_predicate_label(normalized), normalized
        if normalized in self.relation_map:
            return self.relation_map[normalized], normalized
        for source_phrase in sorted(self.relation_map.keys(), key=len, reverse=True):
            if source_phrase in normalized:
                return self.relation_map[source_phrase], normalized
        return UNMAPPED_RELATION, normalized

    def _raw_relation_predicate_label(self, normalized_predicate: str) -> str:
        fragment = self._identifier_fragment(normalized_predicate)
        if fragment == "unknown":
            return UNMAPPED_RELATION
        return f"RAW_RELATION_{fragment.upper()}"

    def _should_keep_relation_episode(
        self,
        *,
        subject_observation: Observation,
        object_observation: Observation,
        predicate: str,
    ) -> bool:
        if predicate in self.relation_episode_exclude_predicates:
            return False
        if subject_observation.entity_id == object_observation.entity_id:
            return False
        if subject_observation.is_static or object_observation.is_static:
            if not self._should_keep_dynamic_static_relation(
                subject_observation=subject_observation,
                object_observation=object_observation,
                predicate=predicate,
            ):
                return False
        if subject_observation.is_noise or object_observation.is_noise:
            return False
        return True

    def _relation_episode_missing_rank_gap(
        self,
        predicate: str,
    ) -> int:
        base_gap = max(0, self.episode_max_frame_gap - 1)
        return max(
            base_gap,
            self.relation_episode_gap_overrides.get(predicate, base_gap),
        )

    def _relation_episode_min_support(
        self,
        predicate: str,
    ) -> int:
        return self.relation_episode_min_support_overrides.get(predicate, 1)

    def _should_keep_dynamic_static_relation(
        self,
        *,
        subject_observation: Observation,
        object_observation: Observation,
        predicate: str,
    ) -> bool:
        if not self.allow_dynamic_static_relations:
            return False
        if subject_observation.is_static:
            return False
        if not object_observation.is_static:
            return True
        if (
            self.dynamic_static_relation_predicates
            and predicate not in self.dynamic_static_relation_predicates
        ):
            return False
        if (
            self.relation_subject_labels
            and subject_observation.label not in self.relation_subject_labels
        ):
            return False
        if (
            self.relation_object_labels
            and object_observation.label not in self.relation_object_labels
        ):
            return False
        return True

    def _should_keep_state_value(self, state_value: str) -> bool:
        normalized = self._normalize_state_value(state_value)
        if not normalized:
            return False
        if normalized in self.state_exact_allowlist:
            return True
        if any(
            normalized.startswith(prefix)
            for prefix in self.state_episode_exclude_prefixes
        ):
            return False
        return any(
            normalized.startswith(prefix)
            for prefix in self.state_prefix_allowlist
        )

    def _new_relation_episode(
        self,
        *,
        subject_entity_id: str,
        predicate: str,
        raw_predicate: str,
        object_entity_id: str,
        frame_index: int,
        timestamp_sec: float,
    ) -> RelationEpisode:
        raw_fragment = self._identifier_fragment(raw_predicate or predicate)
        return RelationEpisode(
            relation_episode_id=(
                f"{subject_entity_id}:{predicate}:{object_entity_id}:"
                f"{frame_index:06d}:{raw_fragment}"
            ),
            video_id=self.video_id,
            subject_entity_id=subject_entity_id,
            predicate=predicate,
            raw_predicate=raw_predicate,
            object_entity_id=object_entity_id,
            start_frame=frame_index,
            end_frame=frame_index,
            start_ts=timestamp_sec,
            end_ts=timestamp_sec,
            support_frames=1,
        )

    def _normalize_label(self, value: str | None) -> str:
        normalized = self._normalize_text(value)
        seen: set[str] = set()
        while normalized in self.label_aliases and normalized not in seen:
            seen.add(normalized)
            normalized = self.label_aliases[normalized]
        return normalized

    def _normalize_relation_text(self, value: str | None) -> str:
        return self._normalize_text(value)

    def _normalize_state_value(self, value: str | None) -> str:
        normalized = self._normalize_text(value)
        seen: set[str] = set()
        while normalized in self.state_aliases and normalized not in seen:
            seen.add(normalized)
            normalized = self.state_aliases[normalized]
        return normalized

    def _merge_metadata_values(self, *values: str | None) -> str | None:
        merged: list[str] = []
        for value in values:
            if not value:
                continue
            for token in str(value).split(" | "):
                normalized_token = self._normalize_text(token)
                if normalized_token and normalized_token not in merged:
                    merged.append(normalized_token)
        if not merged:
            return None
        return " | ".join(merged)

    def _normalize_text(self, value: str | None) -> str:
        if value is None:
            return ""
        normalized = str(value).strip().lower()
        normalized = re.sub(r"[_/\\]+", " ", normalized)
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _identifier_fragment(self, value: str | None) -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return "unknown"
        normalized = normalized.replace(" ", "_")
        normalized = re.sub(r"[^0-9a-z_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "unknown"

    def _log_alignment_summary(self) -> None:
        self.logger.info(
            "alignment_summary=%s",
            json.dumps(asdict(self._alignment_stats), ensure_ascii=False),
        )

    def _log_stage(self, stage: str, started_at: float, **values: Any) -> None:
        payload = {
            "stage": stage,
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
            **values,
        }
        self.logger.info(
            "temporal_graph_progress=%s",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
