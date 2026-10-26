from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _normalize_config_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized


def _apply_alias_map(value: str, alias_map: dict[str, str]) -> str:
    normalized = _normalize_config_text(value)
    seen: set[str] = set()
    while normalized in alias_map and normalized not in seen:
        seen.add(normalized)
        normalized = alias_map[normalized]
    return normalized


def _merge_compatibility_groups(
    groups: list[set[str]],
) -> list[set[str]]:
    pending = [set(group) for group in groups if group]
    merged: list[set[str]] = []

    while pending:
        current = pending.pop(0)
        overlap_found = True
        while overlap_found:
            overlap_found = False
            remaining: list[set[str]] = []
            for candidate in pending:
                if current & candidate:
                    current |= candidate
                    overlap_found = True
                    continue
                remaining.append(candidate)
            pending = remaining
        merged.append(current)

    return merged


class GarbageCollectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_lifespan: int = 2
    static_fragment_max_observations: int | None = None


class ExportConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    exclude_noise: bool = False
    exclude_fragments: bool = False
    protected_fragment_labels: list[str] = Field(default_factory=list)

    @property
    def normalized_protected_fragment_labels(self) -> set[str]:
        return {
            _normalize_config_text(item)
            for item in self.protected_fragment_labels
            if _normalize_config_text(item)
        }


class SceneSegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_scene_length: int = 8
    boundary_confirmation_frames: int = 2
    boundary_window_size: int = 3
    min_static_entities: int = 2
    min_anchor_entity_support_frames: int = 2
    static_entity_jaccard_threshold: float = 0.25
    static_label_jaccard_threshold: float = 0.45
    max_timestamp_gap_sec: float = 30.0


class SceneBaseGraphConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_anchor_coverage_ratio: float = 0.6
    min_anchor_support_frames: int = 3


class AssociationPolicyOptionsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enable_visual_embeddings: bool = True
    enable_parent_label_constraint: bool = False
    enable_track_hints: bool = True
    identity_role_strategy: Literal["semantic_static_dynamic", "tracklet"] = (
        "semantic_static_dynamic"
    )
    allow_visual_stability_static_role: bool = True
    static_entity_match_max_frame_gap: int | None = None
    local_tracklet_mode: bool = False
    local_tracklet_max_frame_gap: int = 1
    local_tracklet_embedding_similarity_threshold: float = 0.72
    local_tracklet_min_score: float = 0.60
    local_tracklet_score_margin: float = 0.08
    local_tracklet_motion_weight: float = 0.35
    local_tracklet_embedding_weight: float = 0.45
    local_tracklet_iou_weight: float = 0.15
    local_tracklet_object_index_weight: float = 0.05
    continuity_max_center_distance_ratio: float | None = None
    continuity_min_area_ratio: float | None = None
    object_index_jump_max_delta: int | None = None
    object_index_jump_embedding_veto_threshold: float | None = None
    state_conflict_embedding_veto_threshold: float | None = None
    state_conflict_min_token_overlap_ratio: float = 0.75
    label_conflict_embedding_veto_threshold: float | None = None
    fast_match_embedding_veto_threshold: float | None = None
    static_fast_match_embedding_veto_threshold: float | None = None
    enable_same_label_dynamic_spatial_gate: bool = False


class LLMFallbackConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider: str = "deepseek_openai_compatible"
    model: str = "deepseek-v4-flash"
    api_base_url: str = "https://api.deepseek.com"
    endpoint_path: str = "/chat/completions"
    api_key_env_var: str = "DEEPSEEK_API_KEY"
    timeout_sec: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 512
    reasoning_effort: str | None = None
    thinking_enabled: bool | None = None
    max_candidates: int = 4
    max_requests_per_run: int = 5
    min_candidate_count: int = 1
    ambiguity_margin: float = 0.05
    low_similarity_floor: float = 0.55
    low_similarity_ambiguity_margin: float = 0.03
    single_candidate_score_margin: float = 0.03
    enable_static_fallback: bool = False
    enable_noise_fallback: bool = False
    enable_candidate_context_fallback: bool = False
    require_second_candidate_for_low_similarity: bool = True
    override_confidence_threshold: float = 0.80
    decision_log_path: str | None = None


class GraphQAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider: str = "deepseek_openai_compatible"
    model: str = "deepseek-v4-flash"
    api_base_url: str = "https://api.deepseek.com"
    endpoint_path: str = "/chat/completions"
    api_key_env_var: str = "DEEPSEEK_API_KEY"
    timeout_sec: float = 30.0
    temperature: float = 0.1
    max_tokens: int = 700
    reasoning_effort: str | None = None
    thinking_enabled: bool | None = None
    prompt_style: Literal["full", "owner_minimal"] = "full"
    max_evidence_lines: int = 6
    max_focus_frames: int = 6
    max_scene_lines: int = 4
    min_evidence_lines: int = 1
    target_evidence_lines: int = 2
    min_focus_frames: int = 1
    target_focus_frames: int = 2
    complex_query_bonus_evidence_lines: int = 1
    complex_query_bonus_focus_frames: int = 1
    enforce_supported_evidence_only: bool = True
    fallback_to_graph_only_on_invalid_response: bool = True
    evidence_rerank_enabled: bool = False
    evidence_rerank_max_candidates: int = 18
    evidence_rerank_max_selected: int = 6
    evidence_rerank_max_tokens: int = 700
    evidence_rerank_temperature: float = 0.0
    answer_candidate_extraction_enabled: bool = False
    answer_candidate_extraction_max_candidates: int = 6
    answer_candidate_extraction_max_tokens: int = 700
    answer_candidate_extraction_temperature: float = 0.0
    experimental_balanced_evidence_candidates: bool = False
    experimental_answerable_candidate_packs: bool = False
    answerable_candidate_pack_max_items: int = 12
    experimental_unified_evidence_layer: bool = False
    unified_evidence_max_items: int = 10
    unified_evidence_graph_quota: int = 5
    unified_evidence_visual_quota: int = 3
    unified_evidence_frame_quota: int = 2
    experimental_atomic_answer_facts: bool = False
    atomic_answer_fact_max_items: int = 8


class DomainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_name: str = "default"
    static_labels: list[str] = Field(default_factory=list)
    hierarchy_static_labels: list[str] = Field(default_factory=list)
    dynamic_labels: list[str] = Field(default_factory=list)
    noise_labels: list[str] = Field(default_factory=list)
    component_labels: list[str] = Field(default_factory=list)
    component_host_labels: list[str] = Field(default_factory=list)
    component_inside_ratio: float = 0.85
    state_subject_labels: list[str] = Field(default_factory=list)
    state_prefix_allowlist: list[str] = Field(default_factory=list)
    state_exact_allowlist: list[str] = Field(default_factory=list)
    state_episode_exclude_prefixes: list[str] = Field(default_factory=list)
    state_aliases: dict[str, str] = Field(default_factory=dict)
    enable_structured_state_extraction: bool = True
    min_state_episode_length_frames: int = 2
    min_state_transition_support_frames: int = 3
    state_transition_min_dominance_margin_frames: int = 2
    state_transition_min_dominance_share: float = 0.6
    state_transition_max_gap_frames: int = 10
    relation_subject_labels: list[str] = Field(default_factory=list)
    relation_object_labels: list[str] = Field(default_factory=list)
    allow_dynamic_static_relations: bool = False
    dynamic_static_relation_predicates: list[str] = Field(default_factory=list)
    enable_semantic_relation_assignment_cleanup: bool = True
    label_aliases: dict[str, str] = Field(default_factory=dict)
    label_compatibility_groups: list[list[str]] = Field(default_factory=list)
    label_min_scores: dict[str, float] = Field(default_factory=dict)
    association_label_gap_overrides: dict[str, int] = Field(default_factory=dict)
    fragment_noise_labels: list[str] = Field(default_factory=list)
    fragment_noise_max_score: float | None = None
    relation_map: dict[str, str] = Field(default_factory=dict)
    enable_relation_predicate_mapping: bool = True
    relation_episode_exclude_predicates: list[str] = Field(default_factory=list)
    relation_episode_gap_overrides: dict[str, int] = Field(default_factory=dict)
    relation_episode_min_support_overrides: dict[str, int] = Field(default_factory=dict)
    min_relation_episode_support_frames: int = 1
    min_relation_episode_length_frames: int = 2
    min_relation_event_support_frames: int = 3
    max_entity_absence_tolerance_frames: int = 3
    min_entity_segment_event_support_frames: int = 3
    entity_segment_event_support_overrides: dict[str, int] = Field(default_factory=dict)
    scene_segmentation: SceneSegmentationConfig = Field(
        default_factory=SceneSegmentationConfig
    )
    scene_base_graph: SceneBaseGraphConfig = Field(
        default_factory=SceneBaseGraphConfig
    )
    association: AssociationPolicyOptionsConfig = Field(
        default_factory=AssociationPolicyOptionsConfig
    )
    llm_fallback: LLMFallbackConfig = Field(
        default_factory=LLMFallbackConfig
    )
    graph_qa: GraphQAConfig = Field(
        default_factory=GraphQAConfig
    )
    garbage_collection: GarbageCollectionConfig = Field(
        default_factory=GarbageCollectionConfig
    )
    export: ExportConfig = Field(default_factory=ExportConfig)

    @property
    def static_label_set(self) -> set[str]:
        return {label.lower() for label in self.static_labels}

    @property
    def dynamic_label_set(self) -> set[str]:
        return {label.lower() for label in self.dynamic_labels}

    @property
    def hierarchy_static_label_set(self) -> set[str]:
        return {label.lower() for label in self.hierarchy_static_labels}

    @property
    def noise_label_set(self) -> set[str]:
        return {label.lower() for label in self.noise_labels}

    @property
    def component_label_set(self) -> set[str]:
        return {label.lower() for label in self.component_labels}

    @property
    def component_host_label_set(self) -> set[str]:
        return {label.lower() for label in self.component_host_labels}

    @property
    def state_subject_label_set(self) -> set[str]:
        return {label.lower() for label in self.state_subject_labels}

    @property
    def normalized_state_prefix_allowlist(self) -> list[str]:
        alias_map = self.normalized_state_aliases
        return [
            _apply_alias_map(item, alias_map)
            for item in self.state_prefix_allowlist
            if _apply_alias_map(item, alias_map)
        ]

    @property
    def normalized_state_exact_allowlist(self) -> set[str]:
        alias_map = self.normalized_state_aliases
        return {
            _apply_alias_map(item, alias_map)
            for item in self.state_exact_allowlist
            if _apply_alias_map(item, alias_map)
        }

    @property
    def normalized_state_episode_exclude_prefixes(self) -> list[str]:
        alias_map = self.normalized_state_aliases
        return [
            _apply_alias_map(item, alias_map)
            for item in self.state_episode_exclude_prefixes
            if _apply_alias_map(item, alias_map)
        ]

    @property
    def normalized_state_aliases(self) -> dict[str, str]:
        normalized_pairs = {
            _normalize_config_text(key): _normalize_config_text(value)
            for key, value in self.state_aliases.items()
            if _normalize_config_text(key) and _normalize_config_text(value)
        }
        resolved: dict[str, str] = {}
        for key in normalized_pairs:
            resolved[key] = _apply_alias_map(normalized_pairs[key], normalized_pairs)
        return resolved

    @property
    def relation_subject_label_set(self) -> set[str]:
        return {label.lower() for label in self.relation_subject_labels}

    @property
    def relation_object_label_set(self) -> set[str]:
        return {label.lower() for label in self.relation_object_labels}

    @property
    def normalized_dynamic_static_relation_predicates(self) -> set[str]:
        return {
            item.strip().upper()
            for item in self.dynamic_static_relation_predicates
            if item and item.strip()
        }

    @property
    def normalized_label_aliases(self) -> dict[str, str]:
        return {
            key.strip().lower(): value.strip().lower()
            for key, value in self.label_aliases.items()
        }

    @property
    def normalized_label_compatibility_groups(self) -> list[set[str]]:
        alias_map = self.normalized_label_aliases
        normalized_groups: list[set[str]] = []
        for group in self.label_compatibility_groups:
            normalized_group = {
                _apply_alias_map(item, alias_map)
                for item in group
                if _apply_alias_map(item, alias_map)
            }
            if len(normalized_group) >= 2:
                normalized_groups.append(normalized_group)
        return _merge_compatibility_groups(normalized_groups)

    @property
    def normalized_label_min_scores(self) -> dict[str, float]:
        alias_map = self.normalized_label_aliases
        normalized: dict[str, float] = {}
        for key, value in self.label_min_scores.items():
            normalized_key = _apply_alias_map(key, alias_map)
            if not normalized_key:
                continue
            normalized[normalized_key] = float(value)
        return normalized

    @property
    def normalized_association_label_gap_overrides(self) -> dict[str, int]:
        alias_map = self.normalized_label_aliases
        normalized: dict[str, int] = {}
        for key, value in self.association_label_gap_overrides.items():
            normalized_key = _apply_alias_map(key, alias_map)
            if not normalized_key:
                continue
            normalized[normalized_key] = max(1, int(value))
        return normalized

    @property
    def normalized_fragment_noise_labels(self) -> set[str]:
        alias_map = self.normalized_label_aliases
        return {
            _apply_alias_map(item, alias_map)
            for item in self.fragment_noise_labels
            if _apply_alias_map(item, alias_map)
        }

    @property
    def normalized_relation_map(self) -> dict[str, str]:
        return {
            key.strip().lower(): value.strip().upper()
            for key, value in self.relation_map.items()
        }

    @property
    def normalized_relation_episode_exclude_predicates(self) -> set[str]:
        return {
            item.strip().upper()
            for item in self.relation_episode_exclude_predicates
            if item and item.strip()
        }

    @property
    def normalized_relation_episode_gap_overrides(self) -> dict[str, int]:
        return {
            key.strip().upper(): max(0, int(value))
            for key, value in self.relation_episode_gap_overrides.items()
            if key and str(key).strip()
        }

    @property
    def normalized_relation_episode_min_support_overrides(self) -> dict[str, int]:
        return {
            key.strip().upper(): max(1, int(value))
            for key, value in self.relation_episode_min_support_overrides.items()
            if key and str(key).strip()
        }

    @property
    def normalized_entity_segment_event_support_overrides(self) -> dict[str, int]:
        alias_map = self.normalized_label_aliases
        normalized: dict[str, int] = {}
        for key, value in self.entity_segment_event_support_overrides.items():
            normalized_key = _apply_alias_map(key, alias_map)
            if not normalized_key:
                continue
            normalized[normalized_key] = max(1, int(value))
        return normalized


def load_domain_config(
    config_path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
) -> DomainConfig:
    base_dir = Path(config_dir) if config_dir is not None else Path.cwd() / "configs"
    default_path = base_dir / "default.yaml"
    merged = _read_yaml(default_path)

    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = base_dir / override_path
        override_payload = _load_with_extends(override_path, default_path=default_path)
        merged = _deep_merge_dicts(merged, override_payload)

    return DomainConfig.model_validate(merged)


def _load_with_extends(
    config_path: Path,
    *,
    default_path: Path,
) -> dict[str, Any]:
    payload = _read_yaml(config_path)
    parent_ref = payload.pop("extends", None)
    if parent_ref is None:
        return payload

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    if parent_path.resolve() == default_path.resolve():
        base_payload = _read_yaml(default_path)
    else:
        base_payload = _load_with_extends(parent_path, default_path=default_path)
    return _deep_merge_dicts(base_payload, payload)


def _read_yaml(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {file_path}")
    return data


def _deep_merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
            continue
        if isinstance(value, list):
            existing = merged.get(key, [])
            if not isinstance(existing, list):
                existing = []
            deduped: list[Any] = []
            for item in [*existing, *value]:
                if item not in deduped:
                    deduped.append(item)
            merged[key] = deduped
            continue
        merged[key] = deepcopy(value)
    return merged
