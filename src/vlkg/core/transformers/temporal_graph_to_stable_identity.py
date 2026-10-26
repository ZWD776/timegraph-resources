from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vlkg.core.models.temporal_graph import Entity, Observation, TemporalGraphDocument
from vlkg.core.models.stable_identity import (
    StableIdentityDocument,
    StableIdentityMember,
    StableIdentityNode,
    StableIdentityTimeRange,
    StableIdentityType,
)


StableIdentityMergeMode = Literal["semantic", "none"]

_HUMAN_LABELS = {
    "player",
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
}
_BALL_LABELS = {"basketball", "ball"}
_COLOR_TOKENS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "maroon",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
_WEARABLE_TOKENS = {
    "jersey",
    "shorts",
    "shirt",
    "pants",
    "dress",
    "shoes",
    "sneakers",
    "sock",
    "socks",
    "uniform",
}
_LOW_DISTINCTIVE_WEARABLE_TOKENS = {
    "shoes",
    "sneakers",
    "sock",
    "socks",
}
_PLAYER_LIKE_KEYWORDS = {
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
    "watching",
    "looking at",
}
_APPEARANCE_ONLY_KEYWORDS = {
    "wearing",
    "dress",
    "shirt",
    "pants",
    "shorts",
    "shoes",
    "sneakers",
    "socks",
    "jersey",
}
_OBJECT_SENSITIVE_ACTION_PREFIXES = (
    "holding",
    "dribbling",
    "shooting",
)
_SPARSE_TAIL_MAX_OBSERVATIONS = 4
_SPARSE_TAIL_MAX_GAP_FRAMES = 80
_SPARSE_TAIL_MIN_EMBEDDING_SIMILARITY = 0.90
_SPARSE_TAIL_MIN_SIGNATURE_JACCARD = 0.45
_SPARSE_TAIL_MIN_AREA_SIMILARITY = 0.85
_SPARSE_TAIL_MAX_NORMALIZED_CENTER_DISTANCE = 0.60
_NUMBER_PATTERN = re.compile(r"\b(?:number\s*)?(\d{1,3})\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class _ObservationProfile:
    center_x: float
    center_y: float
    area: float


@dataclass
class _EntityFeatures:
    entity: Entity
    observations: list[Observation]
    identity_type: StableIdentityType
    signature_tokens: set[str]
    appearance_pair_tokens: set[str]
    human_label_hints: set[str]
    human_context: str
    scene_ids: set[str]
    mean_center_x: float
    mean_center_y: float
    mean_area: float


@dataclass
class _StableIdentityCluster:
    stable_id: str
    identity_type: StableIdentityType
    canonical_label: str
    display_name: str
    members: list[_EntityFeatures] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        return sum(member.entity.observation_count for member in self.members)

    @property
    def labels_seen(self) -> set[str]:
        labels: set[str] = set()
        for member in self.members:
            if member.identity_type == "human" and member.human_label_hints:
                labels.update(member.human_label_hints)
                continue
            labels.update(member.entity.labels_seen or [member.entity.category])
        return labels

    @property
    def signature_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for member in self.members:
            tokens.update(member.signature_tokens)
        return tokens

    @property
    def characteristic_appearance_pair_tokens(self) -> set[str]:
        if not self.members:
            return set()
        support: Counter[str] = Counter()
        for member in self.members:
            support.update(member.appearance_pair_tokens)
        min_support = max(1, math.ceil(len(self.members) * 0.5))
        return {
            token
            for token, count in support.items()
            if count >= min_support
        }

    @property
    def scene_ids(self) -> set[str]:
        scene_ids: set[str] = set()
        for member in self.members:
            scene_ids.update(member.scene_ids)
        return scene_ids

    @property
    def human_contexts(self) -> set[str]:
        return {
            member.human_context
            for member in self.members
            if member.identity_type == "human"
        }

    @property
    def raw_labels_seen(self) -> set[str]:
        labels: set[str] = set()
        for member in self.members:
            labels.update(
                member.entity.raw_labels_seen
                or member.entity.labels_seen
                or [member.entity.category]
            )
        return labels

    @property
    def start_frame(self) -> int:
        return min(member.entity.first_seen_frame for member in self.members)

    @property
    def end_frame(self) -> int:
        return max(member.entity.last_seen_frame for member in self.members)

    @property
    def start_ts(self) -> float:
        return min(member.entity.first_seen_ts for member in self.members)

    @property
    def end_ts(self) -> float:
        return max(member.entity.last_seen_ts for member in self.members)

    @property
    def mean_center_x(self) -> float:
        return sum(member.mean_center_x for member in self.members) / len(self.members)

    @property
    def mean_center_y(self) -> float:
        return sum(member.mean_center_y for member in self.members) / len(self.members)

    @property
    def mean_area(self) -> float:
        return sum(member.mean_area for member in self.members) / len(self.members)


class TemporalGraphToStableIdentityConverter:
    def __init__(
        self,
        document: TemporalGraphDocument,
        *,
        source_graph_path: str | Path | None = None,
        include_fragments: bool = False,
        include_noise: bool = False,
        merge_mode: StableIdentityMergeMode = "semantic",
    ) -> None:
        if merge_mode not in {"semantic", "none"}:
            raise ValueError(
                "stable identity merge_mode must be 'semantic' or 'none'"
            )
        self.document = document
        self.source_graph_path = (
            Path(source_graph_path).resolve()
            if source_graph_path is not None
            else None
        )
        self.include_fragments = include_fragments
        self.include_noise = include_noise
        self.merge_mode = merge_mode
        self._observations_by_entity: dict[str, list[Observation]] = defaultdict(list)
        self._scene_ids_by_entity: dict[str, set[str]] = defaultdict(set)
        for observation in document.observations:
            self._observations_by_entity[observation.entity_id].append(observation)
            if observation.scene_id:
                self._scene_ids_by_entity[observation.entity_id].add(observation.scene_id)

    def convert(self) -> StableIdentityDocument:
        features = [
            self._build_entity_features(entity)
            for entity in self._collect_entities()
        ]
        clusters: list[_StableIdentityCluster] = []
        counters: Counter[str] = Counter()

        for entity_features in features:
            matched_cluster = None
            if self.merge_mode == "semantic":
                matched_cluster = self._find_best_cluster(
                    clusters=clusters,
                    entity_features=entity_features,
                )
            if matched_cluster is None:
                counters[entity_features.identity_type] += 1
                stable_id = (
                    f"{self.document.video_id}:stable:"
                    f"{entity_features.identity_type}:{counters[entity_features.identity_type]:03d}"
                )
                matched_cluster = _StableIdentityCluster(
                    stable_id=stable_id,
                    identity_type=entity_features.identity_type,
                    canonical_label=self._canonical_label_for_identity(entity_features),
                    display_name=(
                        f"{entity_features.identity_type}_{counters[entity_features.identity_type]:03d}"
                    ),
                )
                clusters.append(matched_cluster)
            matched_cluster.members.append(entity_features)

        identities = [
            self._build_identity_node(cluster)
            for cluster in clusters
        ]
        identities.sort(key=lambda item: (item.identity_type, item.display_name))

        return StableIdentityDocument(
            video_id=self.document.video_id,
            source_graph_path=self.source_graph_path,
            identities=identities,
        )

    def _collect_entities(self) -> list[Entity]:
        entities = [
            entity
            for entity in self.document.entities
            if (self.include_noise or not entity.is_noise)
            and (self.include_fragments or not entity.is_fragment)
        ]
        return sorted(
            entities,
            key=lambda item: (item.first_seen_frame, item.category, item.entity_id),
        )

    def _build_entity_features(self, entity: Entity) -> _EntityFeatures:
        observations = sorted(
            self._observations_by_entity.get(entity.entity_id, []),
            key=lambda item: (item.frame_index, item.object_index),
        )
        profiles = [
            _ObservationProfile(
                center_x=observation.bbox.center_x,
                center_y=observation.bbox.center_y,
                area=observation.bbox.area,
            )
            for observation in observations
        ]
        mean_center_x = (
            sum(profile.center_x for profile in profiles) / len(profiles)
            if profiles
            else 0.0
        )
        mean_center_y = (
            sum(profile.center_y for profile in profiles) / len(profiles)
            if profiles
            else 0.0
        )
        mean_area = (
            sum(profile.area for profile in profiles) / len(profiles)
            if profiles
            else 0.0
        )
        identity_type = (
            "object"
            if self.merge_mode == "none"
            else self._identity_type_for_entity(entity)
        )
        signature_tokens = (
            set()
            if self.merge_mode == "none"
            else self._extract_signature_tokens(
                entity,
                observations,
            )
        )
        appearance_pair_tokens = (
            set()
            if self.merge_mode == "none"
            else self._extract_appearance_pair_tokens(
                entity,
                observations,
            )
        )
        human_label_hints = (
            set()
            if self.merge_mode == "none"
            else self._human_label_hints_for_entity(
                entity,
                observations,
            )
        )
        human_context = (
            "generic"
            if self.merge_mode == "none"
            else self._human_context_for_entity(
                entity,
                observations,
            )
        )
        return _EntityFeatures(
            entity=entity,
            observations=observations,
            identity_type=identity_type,
            signature_tokens=signature_tokens,
            appearance_pair_tokens=appearance_pair_tokens,
            human_label_hints=human_label_hints,
            human_context=human_context,
            scene_ids=set(self._scene_ids_by_entity.get(entity.entity_id, set())),
            mean_center_x=mean_center_x,
            mean_center_y=mean_center_y,
            mean_area=mean_area,
        )

    def _identity_type_for_entity(self, entity: Entity) -> StableIdentityType:
        labels = {entity.category, *entity.labels_seen}
        if any(label in _HUMAN_LABELS for label in labels):
            return "human"
        if any(label in _BALL_LABELS for label in labels):
            return "ball"
        if entity.is_static:
            return "static_object"
        return "object"

    def _extract_signature_tokens(
        self,
        entity: Entity,
        observations: list[Observation],
    ) -> set[str]:
        token_support: Counter[str] = Counter()

        if observations:
            for observation in observations:
                per_observation_tokens: set[str] = set()
                texts = [observation.label]
                texts.extend(self._identity_state_texts_for_observation(observation))
                if observation.parent_label:
                    texts.append(observation.parent_label)
                for text in texts:
                    per_observation_tokens.update(
                        self._signature_tokens_from_text(str(text))
                    )
                token_support.update(per_observation_tokens)
            min_support = 2 if len(observations) >= 4 else 1
            supported_tokens = {
                token
                for token, count in token_support.items()
                if count >= min_support
            }
            if supported_tokens:
                return supported_tokens

        fallback_tokens: set[str] = set()
        texts = [
            entity.category,
            *entity.labels_seen,
            *self._identity_state_texts_for_entity(entity),
            *entity.parent_labels_seen,
        ]
        for text in texts:
            fallback_tokens.update(self._signature_tokens_from_text(str(text)))
        return fallback_tokens

    def _extract_appearance_pair_tokens(
        self,
        entity: Entity,
        observations: list[Observation],
    ) -> set[str]:
        token_support: Counter[str] = Counter()

        if observations:
            for observation in observations:
                per_observation_tokens: set[str] = set()
                texts = [observation.label]
                texts.extend(self._identity_state_texts_for_observation(observation))
                if observation.parent_label:
                    texts.append(observation.parent_label)
                for text in texts:
                    per_observation_tokens.update(
                        self._appearance_pair_tokens_from_text(str(text))
                    )
                token_support.update(per_observation_tokens)
            min_support = 2 if len(observations) >= 4 else 1
            supported_tokens = {
                token
                for token, count in token_support.items()
                if count >= min_support
            }
            if supported_tokens:
                return supported_tokens

        fallback_tokens: set[str] = set()
        texts = [
            entity.category,
            *entity.labels_seen,
            *self._identity_state_texts_for_entity(entity),
            *entity.parent_labels_seen,
        ]
        for text in texts:
            fallback_tokens.update(self._appearance_pair_tokens_from_text(str(text)))
        return fallback_tokens

    def _identity_state_texts_for_observation(
        self,
        observation: Observation,
    ) -> list[str]:
        source_values = observation.raw_state_values or observation.state_values
        supported_values = set(observation.state_values)
        return [
            text
            for text in (
                self._strip_unsupported_action_object_tail(
                    value,
                    supported_state_values=supported_values,
                )
                for value in source_values
            )
            if text
        ]

    def _identity_state_texts_for_entity(
        self,
        entity: Entity,
    ) -> list[str]:
        source_values = entity.raw_states_seen or entity.states_seen
        supported_values = set(entity.states_seen)
        return [
            text
            for text in (
                self._strip_unsupported_action_object_tail(
                    value,
                    supported_state_values=supported_values,
                )
                for value in source_values
            )
            if text
        ]

    def _strip_unsupported_action_object_tail(
        self,
        value: str,
        *,
        supported_state_values: set[str],
    ) -> str:
        normalized_text = str(value).lower()
        normalized_supported = {
            " ".join(_TOKEN_PATTERN.findall(item.lower()))
            for item in supported_state_values
        }
        for prefix in _OBJECT_SENSITIVE_ACTION_PREFIXES:
            match = re.search(rf"\b{re.escape(prefix)}\b(?:\s+.+)?$", normalized_text)
            if match is None:
                continue
            action_tail = " ".join(_TOKEN_PATTERN.findall(match.group(0)))
            if (
                action_tail
                and any(
                    state == action_tail or state.startswith(f"{action_tail} ")
                    for state in normalized_supported
                )
            ):
                return value
            stripped = str(value)[: match.start()].strip(" ,;:-")
            return stripped
        return value

    def _signature_tokens_from_text(self, text: str) -> set[str]:
        tokens: set[str] = set()
        normalized_text = str(text).lower()
        normalized_tokens = _TOKEN_PATTERN.findall(normalized_text)
        token_set = set(normalized_tokens)
        for color in _COLOR_TOKENS:
            if color in token_set:
                tokens.add(f"color:{color}")
        for wearable in _WEARABLE_TOKENS:
            if wearable in token_set:
                tokens.add(f"wear:{wearable}")
        for match in _NUMBER_PATTERN.findall(normalized_text):
            tokens.add(f"num:{match}")
        return tokens

    def _appearance_pair_tokens_from_text(self, text: str) -> set[str]:
        tokens: set[str] = set()
        normalized_tokens = _TOKEN_PATTERN.findall(str(text).lower())
        for index, token in enumerate(normalized_tokens):
            if token not in _COLOR_TOKENS:
                continue
            for offset in (1, 2, 3):
                target_index = index + offset
                if target_index >= len(normalized_tokens):
                    break
                candidate = normalized_tokens[target_index]
                if candidate in _WEARABLE_TOKENS:
                    tokens.add(f"appearance:{token}:{candidate}")
                    break
            for offset in (1, 2):
                target_index = index - offset
                if target_index < 0:
                    break
                candidate = normalized_tokens[target_index]
                if candidate in _WEARABLE_TOKENS:
                    tokens.add(f"appearance:{token}:{candidate}")
                    break
        return tokens

    def _human_context_for_entity(
        self,
        entity: Entity,
        observations: list[Observation],
    ) -> str:
        if entity.category == "player" or "player" in entity.labels_seen:
            return "player_like"

        player_like_hits = 0
        appearance_hits = 0
        texts = [
            *self._identity_state_texts_for_entity(entity),
            *entity.labels_seen,
        ]
        for observation in observations:
            texts.extend(self._identity_state_texts_for_observation(observation))
            texts.append(observation.label)

        for text in texts:
            normalized_text = str(text).lower()
            if any(keyword in normalized_text for keyword in _PLAYER_LIKE_KEYWORDS):
                player_like_hits += 1
            if any(keyword in normalized_text for keyword in _APPEARANCE_ONLY_KEYWORDS):
                appearance_hits += 1

        if player_like_hits >= 2:
            return "player_like"
        if player_like_hits == 0 and appearance_hits > 0:
            return "appearance_only"
        return "generic"

    def _human_label_hints_for_entity(
        self,
        entity: Entity,
        observations: list[Observation],
    ) -> set[str]:
        raw_labels = [
            entity.category,
            *(entity.raw_labels_seen or entity.labels_seen),
        ]
        raw_human_labels = {
            label
            for label in raw_labels
            if label in _HUMAN_LABELS
        }

        observation_label_counts: Counter[str] = Counter()
        for observation in observations:
            if observation.label in _HUMAN_LABELS:
                observation_label_counts[observation.label] += 1

        if not observation_label_counts:
            return raw_human_labels

        max_count = max(observation_label_counts.values())
        dominant_labels = {
            label
            for label, count in observation_label_counts.items()
            if count == max_count
        }
        strong_secondary_labels = {
            label
            for label, count in observation_label_counts.items()
            if count >= 2 and count / max(1, len(observations)) >= 0.35
        }
        return dominant_labels | strong_secondary_labels

    def _find_best_cluster(
        self,
        *,
        clusters: list[_StableIdentityCluster],
        entity_features: _EntityFeatures,
    ) -> _StableIdentityCluster | None:
        best_cluster: _StableIdentityCluster | None = None
        best_score = -math.inf

        for cluster in clusters:
            score = self._identity_merge_score(
                cluster=cluster,
                entity_features=entity_features,
            )
            if score is None or score <= best_score:
                continue
            best_score = score
            best_cluster = cluster
        return best_cluster

    def _identity_merge_score(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
    ) -> float | None:
        if cluster.identity_type != entity_features.identity_type:
            return None

        if entity_features.identity_type == "human":
            return self._human_merge_score(
                cluster=cluster,
                entity_features=entity_features,
            )
        if entity_features.identity_type == "ball":
            return self._ball_merge_score(
                cluster=cluster,
                entity_features=entity_features,
            )
        if entity_features.identity_type == "static_object":
            return self._static_object_merge_score(
                cluster=cluster,
                entity_features=entity_features,
            )
        return None

    def _human_merge_score(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
    ) -> float | None:
        overlap_frames = self._time_overlap_frames(
            cluster_start=cluster.start_frame,
            cluster_end=cluster.end_frame,
            entity_start=entity_features.entity.first_seen_frame,
            entity_end=entity_features.entity.last_seen_frame,
        )
        if overlap_frames > 3:
            return None

        cluster_numbers = {
            token
            for token in cluster.signature_tokens
            if token.startswith("num:")
        }
        entity_numbers = {
            token
            for token in entity_features.signature_tokens
            if token.startswith("num:")
        }
        if cluster_numbers and entity_numbers and cluster_numbers.isdisjoint(entity_numbers):
            return None

        shared_appearance_pairs = (
            cluster.characteristic_appearance_pair_tokens
            & entity_features.appearance_pair_tokens
        )
        shared_non_footwear_pairs = {
            token
            for token in shared_appearance_pairs
            if not self._appearance_pair_is_low_distinctive(token)
        }
        cluster_contexts = cluster.human_contexts
        context_pair = cluster_contexts | {entity_features.human_context}
        if (
            "player_like" in context_pair
            and "appearance_only" in context_pair
            and (
                not cluster_numbers
                or not entity_numbers
                or cluster_numbers.isdisjoint(entity_numbers)
            )
        ):
            return None

        embed_similarity = self._embedding_similarity(
            cluster=cluster,
            entity_features=entity_features,
        )
        signature_jaccard = self._signature_jaccard(
            cluster_tokens=cluster.signature_tokens,
            entity_tokens=entity_features.signature_tokens,
        )
        area_similarity = self._ratio_similarity(
            cluster.mean_area,
            entity_features.mean_area,
        )
        if (
            entity_features.entity.observation_count <= _SPARSE_TAIL_MAX_OBSERVATIONS
            and not (cluster_numbers & entity_numbers)
            and not shared_non_footwear_pairs
            and not self._is_strong_sparse_tail_continuation(
                cluster=cluster,
                entity_features=entity_features,
                overlap_frames=overlap_frames,
                embedding_similarity=embed_similarity,
                signature_jaccard=signature_jaccard,
                area_similarity=area_similarity,
                cluster_numbers=cluster_numbers,
                entity_numbers=entity_numbers,
            )
        ):
            return None
        if embed_similarity < 0.60 and signature_jaccard == 0.0:
            return None

        score = (
            embed_similarity * 0.65
            + signature_jaccard * 0.25
            + area_similarity * 0.10
        )
        if cluster_numbers and entity_numbers and not cluster_numbers.isdisjoint(entity_numbers):
            score += 0.40
        if len(shared_non_footwear_pairs) >= 2:
            score += 0.15
        elif shared_non_footwear_pairs:
            score += 0.05
        elif shared_appearance_pairs:
            score += 0.01
        elif self._color_overlap(
            cluster_tokens=cluster.signature_tokens,
            entity_tokens=entity_features.signature_tokens,
        ):
            score += 0.05
        if entity_features.human_context == "player_like" and "player_like" in cluster_contexts:
            score += 0.05
        if overlap_frames <= 0:
            score += 0.05
        return score if score >= 0.85 else None

    def _is_strong_sparse_tail_continuation(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
        overlap_frames: int,
        embedding_similarity: float,
        signature_jaccard: float,
        area_similarity: float,
        cluster_numbers: set[str],
        entity_numbers: set[str],
    ) -> bool:
        if overlap_frames != 0:
            return False
        if cluster_numbers and entity_numbers and cluster_numbers.isdisjoint(entity_numbers):
            return False
        if entity_features.human_context not in cluster.human_contexts:
            return False
        gap_frames = self._time_gap_frames(
            cluster_start=cluster.start_frame,
            cluster_end=cluster.end_frame,
            entity_start=entity_features.entity.first_seen_frame,
            entity_end=entity_features.entity.last_seen_frame,
        )
        if gap_frames > _SPARSE_TAIL_MAX_GAP_FRAMES:
            return False
        if embedding_similarity < _SPARSE_TAIL_MIN_EMBEDDING_SIMILARITY:
            return False
        if signature_jaccard < _SPARSE_TAIL_MIN_SIGNATURE_JACCARD:
            return False
        if area_similarity < _SPARSE_TAIL_MIN_AREA_SIMILARITY:
            return False
        return (
            self._normalized_center_distance(
                cluster_mean_center_x=cluster.mean_center_x,
                cluster_mean_center_y=cluster.mean_center_y,
                cluster_mean_area=cluster.mean_area,
                entity_mean_center_x=entity_features.mean_center_x,
                entity_mean_center_y=entity_features.mean_center_y,
                entity_mean_area=entity_features.mean_area,
            )
            <= _SPARSE_TAIL_MAX_NORMALIZED_CENTER_DISTANCE
        )

    def _ball_merge_score(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
    ) -> float | None:
        overlap_frames = self._time_overlap_frames(
            cluster_start=cluster.start_frame,
            cluster_end=cluster.end_frame,
            entity_start=entity_features.entity.first_seen_frame,
            entity_end=entity_features.entity.last_seen_frame,
        )
        if overlap_frames > 5:
            return None
        embed_similarity = self._embedding_similarity(
            cluster=cluster,
            entity_features=entity_features,
        )
        area_similarity = self._ratio_similarity(
            cluster.mean_area,
            entity_features.mean_area,
        )
        score = embed_similarity * 0.75 + area_similarity * 0.25
        return score if score >= 0.80 else None

    def _static_object_merge_score(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
    ) -> float | None:
        if cluster.canonical_label != entity_features.entity.category:
            return None
        embed_similarity = self._embedding_similarity(
            cluster=cluster,
            entity_features=entity_features,
        )
        area_similarity = self._ratio_similarity(
            cluster.mean_area,
            entity_features.mean_area,
        )
        score = embed_similarity * 0.60 + area_similarity * 0.40
        return score if score >= 0.90 else None

    def _embedding_similarity(
        self,
        *,
        cluster: _StableIdentityCluster,
        entity_features: _EntityFeatures,
    ) -> float:
        similarities: list[float] = []
        current_embedding = entity_features.entity.canonical_embedding
        if current_embedding is None:
            return 0.0
        current_norm = _vector_norm(current_embedding)
        if current_norm == 0.0:
            return 0.0
        for member in cluster.members:
            candidate_embedding = member.entity.canonical_embedding
            if candidate_embedding is None:
                continue
            candidate_norm = _vector_norm(candidate_embedding)
            if candidate_norm == 0.0:
                continue
            similarities.append(
                _dot(current_embedding, candidate_embedding)
                / (current_norm * candidate_norm)
            )
        return max(similarities) if similarities else 0.0

    def _signature_jaccard(
        self,
        *,
        cluster_tokens: set[str],
        entity_tokens: set[str],
    ) -> float:
        if not cluster_tokens or not entity_tokens:
            return 0.0
        intersection = len(cluster_tokens & entity_tokens)
        union = len(cluster_tokens | entity_tokens)
        return intersection / union if union else 0.0

    def _color_overlap(
        self,
        *,
        cluster_tokens: set[str],
        entity_tokens: set[str],
    ) -> bool:
        cluster_colors = {
            token
            for token in cluster_tokens
            if token.startswith("color:")
        }
        entity_colors = {
            token
            for token in entity_tokens
            if token.startswith("color:")
        }
        return bool(cluster_colors & entity_colors)

    def _appearance_pair_is_low_distinctive(self, token: str) -> bool:
        if not token.startswith("appearance:"):
            return False
        _, _, wearable = token.split(":", maxsplit=2)
        return wearable in _LOW_DISTINCTIVE_WEARABLE_TOKENS

    def _ratio_similarity(self, value_a: float, value_b: float) -> float:
        if value_a <= 0.0 or value_b <= 0.0:
            return 0.0
        return min(value_a, value_b) / max(value_a, value_b)

    def _time_overlap_frames(
        self,
        *,
        cluster_start: int,
        cluster_end: int,
        entity_start: int,
        entity_end: int,
    ) -> int:
        overlap_start = max(cluster_start, entity_start)
        overlap_end = min(cluster_end, entity_end)
        if overlap_end < overlap_start:
            return 0
        return overlap_end - overlap_start + 1

    def _time_gap_frames(
        self,
        *,
        cluster_start: int,
        cluster_end: int,
        entity_start: int,
        entity_end: int,
    ) -> int:
        if self._time_overlap_frames(
            cluster_start=cluster_start,
            cluster_end=cluster_end,
            entity_start=entity_start,
            entity_end=entity_end,
        ):
            return 0
        if cluster_end < entity_start:
            return entity_start - cluster_end
        return cluster_start - entity_end

    def _normalized_center_distance(
        self,
        *,
        cluster_mean_center_x: float,
        cluster_mean_center_y: float,
        cluster_mean_area: float,
        entity_mean_center_x: float,
        entity_mean_center_y: float,
        entity_mean_area: float,
    ) -> float:
        scale = math.sqrt(max(cluster_mean_area, entity_mean_area, 1.0))
        dx = cluster_mean_center_x - entity_mean_center_x
        dy = cluster_mean_center_y - entity_mean_center_y
        return math.sqrt(dx * dx + dy * dy) / scale

    def _canonical_label_for_identity(self, entity_features: _EntityFeatures) -> str:
        if entity_features.identity_type == "human":
            return "person"
        if entity_features.identity_type == "ball":
            return "basketball"
        return entity_features.entity.category

    def _build_identity_node(
        self,
        cluster: _StableIdentityCluster,
    ) -> StableIdentityNode:
        members = sorted(
            cluster.members,
            key=lambda item: (
                item.entity.first_seen_frame,
                item.entity.category,
                item.entity.entity_id,
            ),
        )
        labels_seen = sorted(cluster.labels_seen)
        signature_tokens = sorted(cluster.signature_tokens)
        scene_ids = sorted(cluster.scene_ids)
        observation_count = cluster.observation_count

        return StableIdentityNode(
            stable_id=cluster.stable_id,
            video_id=self.document.video_id,
            identity_type=cluster.identity_type,
            canonical_label=cluster.canonical_label,
            display_name=cluster.display_name,
            labels_seen=labels_seen,
            signature_tokens=signature_tokens,
            scene_ids=scene_ids,
            member_entities=[
                StableIdentityMember(
                    entity_id=member.entity.entity_id,
                    category=member.entity.category,
                    first_seen_frame=member.entity.first_seen_frame,
                    last_seen_frame=member.entity.last_seen_frame,
                    first_seen_ts=member.entity.first_seen_ts,
                    last_seen_ts=member.entity.last_seen_ts,
                    observation_count=member.entity.observation_count,
                    labels_seen=list(member.entity.labels_seen),
                    states_seen=list(member.entity.states_seen),
                    parent_labels_seen=list(member.entity.parent_labels_seen),
                )
                for member in members
            ],
            time_range=StableIdentityTimeRange(
                start_frame=cluster.start_frame,
                end_frame=cluster.end_frame,
                start_ts=cluster.start_ts,
                end_ts=cluster.end_ts,
                support_frames=observation_count,
            ),
            observation_count=observation_count,
            attributes={
                "member_count": len(members),
                "mean_center_x": round(cluster.mean_center_x, 3),
                "mean_center_y": round(cluster.mean_center_y, 3),
                "mean_area": round(cluster.mean_area, 3),
                "raw_labels_seen": sorted(cluster.raw_labels_seen),
            },
        )


def _dot(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        return 0.0
    return sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def build_stable_identity_document(
    document: TemporalGraphDocument,
    *,
    source_graph_path: str | Path | None = None,
    include_fragments: bool = False,
    include_noise: bool = False,
    merge_mode: StableIdentityMergeMode = "semantic",
) -> StableIdentityDocument:
    return TemporalGraphToStableIdentityConverter(
        document,
        source_graph_path=source_graph_path,
        include_fragments=include_fragments,
        include_noise=include_noise,
        merge_mode=merge_mode,
    ).convert()
