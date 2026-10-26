from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from vlkg.core.graph_retrieval.graph_qa_context import build_llm_ready_bundle
from vlkg.core.models.temporal_graph import Entity, Observation, TemporalGraphDocument
from vlkg.core.models.temporal_index import TemporalIndex


@dataclass(frozen=True)
class QueryResolution:
    entity_ids: tuple[str, ...]
    matched_by: str


FOCUS_FRAME_REASON_WEIGHTS = {
    "snapshot_frame": 7,
    "entity_first_seen": 5,
    "entity_last_seen": 5,
    "first_observation": 4,
    "last_observation": 4,
    "scene_start": 4,
    "scene_end": 4,
    "entity_segment_start": 3,
    "entity_segment_end": 3,
    "episode_start": 3,
    "episode_end": 3,
    "transition_event": 6,
}


class TemporalGraphQueryEngine:
    def __init__(self, document: TemporalGraphDocument) -> None:
        self.document = document
        self.index = TemporalIndex.build(document)
        self.entities_by_id = self.index.entities_by_id
        self.observations_by_entity: dict[str, list[Observation]] = defaultdict(list)
        self.frame_timestamps = self._build_frame_timestamp_map(document)

        for observation in document.observations:
            self.observations_by_entity[observation.entity_id].append(observation)
        for entity_id, observations in self.observations_by_entity.items():
            observations.sort(
                key=lambda item: (
                    item.frame_index,
                    item.frame_rank,
                    item.timestamp_sec,
                    item.object_index,
                )
            )

    def answer_entity_timeline(
        self,
        *,
        entity_id: str | None = None,
        label: str | None = None,
        text: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        resolution = self.resolve_entities(
            entity_id=entity_id,
            label=label,
            text=text,
            limit=limit,
        )
        matches = [
            self._entity_timeline_match(current_entity_id)
            for current_entity_id in resolution.entity_ids
        ]
        filters = {
            "entity_id": entity_id,
            "label": label,
            "text": text,
            "limit": limit,
        }
        summary = (
            f"Matched {len(matches)} entity timeline records."
            if matches
            else "No entity timeline matches found."
        )
        retrieval_bundle = self._build_retrieval_bundle(
            query_type="entity_timeline",
            matches=matches,
        )
        return self._assemble_answer_payload(
            query_type="entity_timeline",
            matched_by=resolution.matched_by,
            filters=filters,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )

    def answer_scene_summary(
        self,
        *,
        scene_id: str | None = None,
        frame_index: int | None = None,
        entity_id: str | None = None,
        label: str | None = None,
        text: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        matched_by: Any = "scene_id" if scene_id else "frame_index" if frame_index is not None else "scene_container"
        scene_ids: list[str] = []

        if scene_id is not None:
            scene = self.index.scene_by_id(scene_id)
            if scene is None:
                scene_ids = []
            else:
                scene_ids = [scene.scene_id]
        elif frame_index is not None:
            scene = self.index.scene_at_frame(frame_index)
            if scene is not None:
                scene_ids = [scene.scene_id]
        else:
            resolution = self.resolve_entities(
                entity_id=entity_id,
                label=label,
                text=text,
                limit=limit,
            )
            matched_by = resolution.matched_by
            seen_scene_ids: set[str] = set()
            for current_entity_id in resolution.entity_ids:
                for segment in self.index.entity_segments_for_entity(current_entity_id):
                    if segment.scene_id is None or segment.scene_id in seen_scene_ids:
                        continue
                    scene_ids.append(segment.scene_id)
                    seen_scene_ids.add(segment.scene_id)
                    if len(scene_ids) >= limit:
                        break
                if len(scene_ids) >= limit:
                    break

        matches = [
            self._scene_summary_match(current_scene_id)
            for current_scene_id in scene_ids[:limit]
        ]
        filters = {
            "scene_id": scene_id,
            "frame_index": frame_index,
            "entity_id": entity_id,
            "label": label,
            "text": text,
            "limit": limit,
        }
        summary = (
            f"Matched {len(matches)} scene summaries."
            if matches
            else "No scene summary matches found."
        )
        retrieval_bundle = self._build_retrieval_bundle(
            query_type="scene_summary",
            matches=matches,
        )
        return self._assemble_answer_payload(
            query_type="scene_summary",
            matched_by=matched_by,
            filters=filters,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )

    def answer_frame_snapshot(
        self,
        *,
        frame_index: int,
        entity_id: str | None = None,
        label: str | None = None,
        text: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        scene = self.index.scene_at_frame(frame_index)
        scene_summary = self.index.scene_delta_summary_at_frame(frame_index)
        if scene is None or scene_summary is None:
            matches: list[dict[str, Any]] = []
            matched_by: Any = "frame_index"
        else:
            resolution = self.resolve_entities(
                entity_id=entity_id,
                label=label,
                text=text,
                limit=limit,
            )
            filtered_entity_ids = set(resolution.entity_ids) if any((entity_id, label, text)) else None
            matched_by = resolution.matched_by if filtered_entity_ids is not None else "frame_index"
            if filtered_entity_ids is not None and not filtered_entity_ids:
                matches = []
            else:
                matches = [
                    self._frame_snapshot_match(
                        frame_index=frame_index,
                        scene_id=scene.scene_id,
                        filtered_entity_ids=filtered_entity_ids,
                    )
                ]

        filters = {
            "frame_index": frame_index,
            "entity_id": entity_id,
            "label": label,
            "text": text,
            "limit": limit,
        }
        summary = (
            f"Matched {len(matches)} frame snapshots."
            if matches
            else "No frame snapshot matches found."
        )
        retrieval_bundle = self._build_retrieval_bundle(
            query_type="frame_snapshot",
            matches=matches,
        )
        return self._assemble_answer_payload(
            query_type="frame_snapshot",
            matched_by=matched_by,
            filters=filters,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )

    def answer_state_history(
        self,
        *,
        entity_id: str | None = None,
        label: str | None = None,
        text: str | None = None,
        state_text: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        resolution = self.resolve_entities(
            entity_id=entity_id,
            label=label,
            text=text,
            limit=limit,
        )
        normalized_state_text = _normalize_text(state_text)
        matches: list[dict[str, Any]] = []
        for current_entity_id in resolution.entity_ids:
            entity = self.entities_by_id[current_entity_id]
            for episode in self.index.state_episodes_by_entity.get(current_entity_id, []):
                if normalized_state_text and normalized_state_text not in _normalize_text(
                    episode.attr_value
                ):
                    continue
                matches.append(
                    {
                        "entity_id": entity.entity_id,
                        "entity_category": entity.category,
                        "state_episode_id": episode.state_episode_id,
                        "state_value": episode.attr_value,
                        "start_frame": episode.start_frame,
                        "end_frame": episode.end_frame,
                        "start_ts": episode.start_ts,
                        "end_ts": episode.end_ts,
                        "support_frames": episode.support_frames,
                        "evidence": {
                            "transition_events": [
                                event.model_dump(mode="json")
                                for event in self.index.transition_events_for_entity(
                                    current_entity_id,
                                    event_type="STATE_TRANSITION",
                                )
                                if event.source_episode_id == episode.state_episode_id
                            ],
                        },
                    }
                )
        matches.sort(
            key=lambda item: (
                item["start_frame"],
                item["entity_category"],
                item["entity_id"],
                item["state_value"],
            )
        )
        matches = matches[:limit]
        filters = {
            "entity_id": entity_id,
            "label": label,
            "text": text,
            "state_text": state_text,
            "limit": limit,
        }
        summary = (
            f"Matched {len(matches)} state history records."
            if matches
            else "No state history matches found."
        )
        retrieval_bundle = self._build_retrieval_bundle(
            query_type="state_history",
            matches=matches,
        )
        return self._assemble_answer_payload(
            query_type="state_history",
            matched_by=resolution.matched_by,
            filters=filters,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )

    def answer_relation_history(
        self,
        *,
        subject_entity_id: str | None = None,
        subject_label: str | None = None,
        subject_text: str | None = None,
        object_entity_id: str | None = None,
        object_label: str | None = None,
        object_text: str | None = None,
        predicate: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        subject_has_filter = any((subject_entity_id, subject_label, subject_text))
        object_has_filter = any((object_entity_id, object_label, object_text))
        subject_resolution = (
            self.resolve_entities(
                entity_id=subject_entity_id,
                label=subject_label,
                text=subject_text,
                limit=limit,
            )
            if subject_has_filter
            else QueryResolution(entity_ids=tuple(), matched_by="all_entities")
        )
        object_resolution = (
            self.resolve_entities(
                entity_id=object_entity_id,
                label=object_label,
                text=object_text,
                limit=limit,
            )
            if object_has_filter
            else QueryResolution(entity_ids=tuple(), matched_by="all_entities")
        )
        subject_entity_id_set = set(subject_resolution.entity_ids)
        object_entity_id_set = set(object_resolution.entity_ids)
        normalized_predicate = predicate.strip().upper() if predicate else None
        matches: list[dict[str, Any]] = []
        for episode in self.document.relation_episodes:
            if subject_entity_id_set and episode.subject_entity_id not in subject_entity_id_set:
                continue
            if object_entity_id_set and episode.object_entity_id not in object_entity_id_set:
                continue
            if normalized_predicate and episode.predicate != normalized_predicate:
                continue
            subject_entity = self.entities_by_id[episode.subject_entity_id]
            object_entity = self.entities_by_id[episode.object_entity_id]
            matches.append(
                {
                    "relation_episode_id": episode.relation_episode_id,
                    "predicate": episode.predicate,
                    "raw_predicate": episode.raw_predicate,
                    "subject": {
                        "entity_id": subject_entity.entity_id,
                        "category": subject_entity.category,
                    },
                    "object": {
                        "entity_id": object_entity.entity_id,
                        "category": object_entity.category,
                    },
                    "start_frame": episode.start_frame,
                    "end_frame": episode.end_frame,
                    "start_ts": episode.start_ts,
                    "end_ts": episode.end_ts,
                    "support_frames": episode.support_frames,
                    "evidence": {
                        "transition_events": [
                            event.model_dump(mode="json")
                            for event in self._relation_transition_events(
                                episode.relation_episode_id
                            )
                        ],
                    },
                }
            )
        matches.sort(
            key=lambda item: (
                item["start_frame"],
                item["predicate"],
                item["subject"]["entity_id"],
                item["object"]["entity_id"],
            )
        )
        matches = matches[:limit]
        filters = {
            "subject_entity_id": subject_entity_id,
            "subject_label": subject_label,
            "subject_text": subject_text,
            "object_entity_id": object_entity_id,
            "object_label": object_label,
            "object_text": object_text,
            "predicate": normalized_predicate,
            "limit": limit,
        }
        matched_by = {
            "subject": subject_resolution.matched_by,
            "object": object_resolution.matched_by,
        }
        summary = (
            f"Matched {len(matches)} relation history records."
            if matches
            else "No relation history matches found."
        )
        retrieval_bundle = self._build_retrieval_bundle(
            query_type="relation_history",
            matches=matches,
        )
        return self._assemble_answer_payload(
            query_type="relation_history",
            matched_by=matched_by,
            filters=filters,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )

    def resolve_entities(
        self,
        *,
        entity_id: str | None = None,
        label: str | None = None,
        text: str | None = None,
        limit: int = 20,
    ) -> QueryResolution:
        if entity_id:
            if entity_id not in self.entities_by_id:
                raise KeyError(f"Unknown entity_id: {entity_id}")
            return QueryResolution(entity_ids=(entity_id,), matched_by="entity_id")

        normalized_label = _normalize_text(label)
        normalized_text = _normalize_text(text)
        if not normalized_label and not normalized_text:
            return QueryResolution(
                entity_ids=tuple(sorted(self.entities_by_id.keys())[:limit]),
                matched_by="all_entities",
            )

        ranked_matches: list[tuple[int, str]] = []
        for current_entity_id, entity in self.entities_by_id.items():
            score = self._entity_match_score(
                entity=entity,
                normalized_label=normalized_label,
                normalized_text=normalized_text,
            )
            if score <= 0:
                continue
            ranked_matches.append((score, current_entity_id))

        ranked_matches.sort(
            key=lambda item: (
                -item[0],
                self.entities_by_id[item[1]].first_seen_frame,
                item[1],
            )
        )
        resolved_entity_ids = tuple(entity_key for _, entity_key in ranked_matches[:limit])
        if not resolved_entity_ids:
            return QueryResolution(entity_ids=tuple(), matched_by="no_match")

        if normalized_label and normalized_text:
            matched_by = "label+text"
        elif normalized_label:
            matched_by = "label"
        else:
            matched_by = "text"
        return QueryResolution(entity_ids=resolved_entity_ids, matched_by=matched_by)

    def _entity_timeline_match(self, entity_id: str) -> dict[str, Any]:
        entity = self.entities_by_id[entity_id]
        observations = self.observations_by_entity.get(entity_id, [])
        first_observation = observations[0] if observations else None
        last_observation = observations[-1] if observations else None
        entity_segments = self.index.entity_segments_for_entity(entity_id)
        transition_events = self.index.transition_events_for_entity(entity_id)
        gap_segments = [
            segment
            for segment in entity_segments
            if segment.gap_from_previous_ranks is not None
            and segment.gap_from_previous_ranks > 0
        ]
        max_gap_frames = max(
            (
                segment.gap_from_previous_frames or 0
                for segment in gap_segments
            ),
            default=0,
        )
        max_gap_ranks = max(
            (
                segment.gap_from_previous_ranks or 0
                for segment in gap_segments
            ),
            default=0,
        )
        return {
            "entity_id": entity.entity_id,
            "category": entity.category,
            "first_seen_frame": entity.first_seen_frame,
            "last_seen_frame": entity.last_seen_frame,
            "first_seen_ts": entity.first_seen_ts,
            "last_seen_ts": entity.last_seen_ts,
            "observation_count": entity.observation_count,
            "segment_count": len(entity_segments),
            "reappearance_count": len(gap_segments),
            "max_gap_frames": max_gap_frames,
            "max_gap_ranks": max_gap_ranks,
            "labels_seen": entity.labels_seen,
            "states_seen": entity.states_seen,
            "parent_labels_seen": entity.parent_labels_seen,
            "is_static": entity.is_static,
            "is_noise": entity.is_noise,
            "is_fragment": entity.is_fragment,
            "evidence": {
                "entity_segments": [
                    segment.model_dump(mode="json")
                    for segment in entity_segments
                ],
                "transition_events": [
                    event.model_dump(mode="json")
                    for event in transition_events
                ],
                "appear_events": [
                    event.model_dump(mode="json")
                    for event in transition_events
                    if event.event_type == "APPEAR"
                ],
                "disappear_events": [
                    event.model_dump(mode="json")
                    for event in transition_events
                    if event.event_type == "DISAPPEAR"
                ],
                "first_observation": (
                    first_observation.model_dump(mode="json")
                    if first_observation is not None
                    else None
                ),
                "last_observation": (
                    last_observation.model_dump(mode="json")
                    if last_observation is not None
                    else None
                ),
            },
        }

    def _entity_match_score(
        self,
        *,
        entity: Entity,
        normalized_label: str,
        normalized_text: str,
    ) -> int:
        label_haystacks = {
            _normalize_text(entity.category),
            *[_normalize_text(value) for value in entity.labels_seen],
            *[_normalize_text(value) for value in entity.parent_labels_seen],
        }
        text_haystacks = {
            *label_haystacks,
            *[_normalize_text(value) for value in entity.states_seen],
        }
        score = 0
        if normalized_label:
            if normalized_label == _normalize_text(entity.category):
                score += 100
            elif normalized_label in label_haystacks:
                score += 80
            elif any(normalized_label in value for value in label_haystacks if value):
                score += 50
        if normalized_text:
            if any(normalized_text == value for value in text_haystacks if value):
                score += 60
            elif any(normalized_text in value for value in text_haystacks if value):
                score += 30
        return score

    def _relation_transition_events(self, relation_episode_id: str) -> list[Any]:
        return [
            event
            for event in self.document.transition_events
            if event.source_episode_id == relation_episode_id
        ]

    def _scene_summary_match(self, scene_id: str) -> dict[str, Any]:
        scene = self.index.scene_by_id(scene_id)
        if scene is None:
            raise KeyError(f"Unknown scene_id: {scene_id}")
        scene_summary = self.index.scene_delta_summary(scene_id)
        base_nodes = self.index.scene_base_nodes_for_scene(scene_id)
        delta_nodes = self.index.scene_delta_nodes_for_scene(scene_id)
        delta_observations = self.index.scene_delta_observations_for_scene(scene_id)
        return {
            "scene_id": scene.scene_id,
            "start_frame": scene.start_frame,
            "end_frame": scene.end_frame,
            "start_ts": scene.start_ts,
            "end_ts": scene.end_ts,
            "support_frames": scene.support_frames,
            "scene_delta_summary": asdict(scene_summary),
            "evidence": {
                "scene_segment": scene.model_dump(mode="json"),
                "scene_base_nodes": [node.model_dump(mode="json") for node in base_nodes],
                "scene_delta_nodes": [node.model_dump(mode="json") for node in delta_nodes],
                "scene_delta_observations": [
                    observation.model_dump(mode="json")
                    for observation in delta_observations[:40]
                ],
            },
        }

    def _frame_snapshot_match(
        self,
        *,
        frame_index: int,
        scene_id: str,
        filtered_entity_ids: set[str] | None,
    ) -> dict[str, Any]:
        scene = self.index.scene_by_id(scene_id)
        scene_summary = self.index.scene_delta_summary_at_frame(frame_index)
        if scene is None or scene_summary is None:
            raise KeyError(f"Unknown frame snapshot context for frame_index={frame_index}")

        frame_rank = self.index.frame_rank_for_index(frame_index)
        timestamp_sec = self.frame_timestamps.get(frame_index)

        scene_base_nodes = self.index.scene_base_nodes_at_frame(frame_index)
        delta_nodes = self.index.scene_delta_nodes_at_frame(frame_index)
        delta_observations = self.index.scene_delta_observations_at_frame(frame_index)
        active_states = self.index.active_states(frame_index)
        active_relations = self.index.active_relations(frame_index)
        transition_events = self.index.transition_events_at_frame(frame_index=frame_index)

        if filtered_entity_ids is not None:
            scene_base_nodes = [
                node for node in scene_base_nodes if node.entity_id in filtered_entity_ids
            ]
            delta_nodes = [
                node for node in delta_nodes if node.entity_id in filtered_entity_ids
            ]
            delta_observations = [
                observation
                for observation in delta_observations
                if observation.entity_id in filtered_entity_ids
            ]
            active_states = [
                episode
                for episode in active_states
                if episode.entity_id in filtered_entity_ids
            ]
            active_relations = [
                episode
                for episode in active_relations
                if (
                    episode.subject_entity_id in filtered_entity_ids
                    or episode.object_entity_id in filtered_entity_ids
                )
            ]
            transition_events = [
                event
                for event in transition_events
                if self.index._event_entity_ids(event) & filtered_entity_ids
            ]

        filtered_scene_summary = {
            **asdict(scene_summary),
            "scene_base_node_count": len(scene_base_nodes),
            "active_delta_node_count": len(delta_nodes),
            "active_delta_observation_count": len(delta_observations),
            "active_delta_entity_ids": tuple(
                sorted({node.entity_id for node in delta_nodes})
            ),
            "relation_episode_count": len(
                {
                    episode.relation_episode_id
                    for episode in active_relations
                }
            ),
            "state_episode_count": len(
                {
                    episode.state_episode_id
                    for episode in active_states
                }
            ),
            "transition_event_count": len(transition_events),
        }
        transition_event_type_breakdown: dict[str, int] = defaultdict(int)
        for event in transition_events:
            transition_event_type_breakdown[event.event_type] += 1
        filtered_scene_summary["transition_event_type_breakdown"] = dict(
            sorted(transition_event_type_breakdown.items())
        )

        return {
            "frame_index": frame_index,
            "frame_rank": frame_rank,
            "timestamp_sec": timestamp_sec,
            "scene_id": scene.scene_id,
            "scene_start_frame": scene.start_frame,
            "scene_end_frame": scene.end_frame,
            "scene_delta_summary": filtered_scene_summary,
            "evidence": {
                "scene_segment": scene.model_dump(mode="json"),
                "scene_base_nodes": [
                    node.model_dump(mode="json")
                    for node in scene_base_nodes
                ],
                "active_delta_nodes": [
                    node.model_dump(mode="json")
                    for node in delta_nodes
                ],
                "active_delta_observations": [
                    observation.model_dump(mode="json")
                    for observation in delta_observations
                ],
                "active_states": [
                    episode.model_dump(mode="json")
                    for episode in active_states
                ],
                "active_relations": [
                    episode.model_dump(mode="json")
                    for episode in active_relations
                ],
                "transition_events": [
                    event.model_dump(mode="json")
                    for event in transition_events
                ],
            },
        }

    def _build_frame_timestamp_map(
        self,
        document: TemporalGraphDocument,
    ) -> dict[int, float]:
        mapping: dict[int, float] = {}
        for observation in document.observations:
            mapping.setdefault(observation.frame_index, observation.timestamp_sec)
        for event in document.transition_events:
            mapping.setdefault(event.frame_index, event.timestamp_sec)
        for segment in document.scene_segments:
            mapping.setdefault(segment.start_frame, segment.start_ts)
            mapping.setdefault(segment.end_frame, segment.end_ts)
        return mapping

    def _assemble_answer_payload(
        self,
        *,
        query_type: str,
        matched_by: Any,
        filters: dict[str, Any],
        summary: str,
        matches: list[dict[str, Any]],
        retrieval_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        llm_ready_bundle = build_llm_ready_bundle(
            video_id=self.document.video_id,
            query_type=query_type,
            filters=filters,
            matched_by=matched_by,
            summary=summary,
            matches=matches,
            retrieval_bundle=retrieval_bundle,
        )
        return {
            "query_type": query_type,
            "matched_by": matched_by,
            "filters": filters,
            "summary": summary,
            "matches": matches,
            "retrieval_bundle": retrieval_bundle,
            "llm_ready_bundle": llm_ready_bundle,
            "answer_brief": llm_ready_bundle["answer_brief"],
        }

    def _build_retrieval_bundle(
        self,
        *,
        query_type: str,
        matches: list[dict[str, Any]],
        max_focus_frames: int = 8,
        max_scene_summaries: int = 4,
        max_evidence_items: int = 8,
    ) -> dict[str, Any]:
        focus_frame_reasons: dict[int, set[str]] = defaultdict(set)
        for match in matches:
            self._register_focus_frames(
                query_type=query_type,
                match=match,
                focus_frame_reasons=focus_frame_reasons,
            )

        ranked_focus_frames = self._rank_focus_frames(
            focus_frame_reasons=focus_frame_reasons,
            max_focus_frames=max_focus_frames,
        )
        scene_summaries: list[dict[str, Any]] = []
        seen_scene_ids: set[str] = set()

        for item in ranked_focus_frames:
            scene = self.index.scene_at_frame(item["frame_index"])
            if scene is None:
                continue
            item["scene_id"] = scene.scene_id
            if scene.scene_id in seen_scene_ids:
                continue
            scene_summaries.append(asdict(self.index.scene_delta_summary(scene.scene_id)))
            seen_scene_ids.add(scene.scene_id)
            if len(scene_summaries) >= max_scene_summaries:
                break

        return {
            "video_id": self.document.video_id,
            "query_type": query_type,
            "match_count": len(matches),
            "focus_frames": ranked_focus_frames,
            "scene_summaries": scene_summaries,
            "evidence_digest": self._build_evidence_digest(
                query_type=query_type,
                matches=matches,
                limit=max_evidence_items,
            ),
        }

    def _rank_focus_frames(
        self,
        *,
        focus_frame_reasons: dict[int, set[str]],
        max_focus_frames: int,
    ) -> list[dict[str, Any]]:
        ranked_items: list[tuple[int, int, list[str]]] = []
        for frame_index, reasons in focus_frame_reasons.items():
            sorted_reasons = sorted(reasons)
            score = sum(
                FOCUS_FRAME_REASON_WEIGHTS.get(reason, 1)
                for reason in sorted_reasons
            )
            ranked_items.append((score, frame_index, sorted_reasons))

        ranked_items.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, frame_index, reasons) in enumerate(
            ranked_items[:max_focus_frames],
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "frame_index": frame_index,
                    "frame_rank": self.index.frame_rank_for_index(frame_index),
                    "timestamp_sec": self.frame_timestamps.get(frame_index),
                    "scene_id": None,
                    "score": score,
                    "reasons": reasons,
                }
            )
        return results

    def _build_evidence_digest(
        self,
        *,
        query_type: str,
        matches: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        digest: list[dict[str, Any]] = []
        for match in matches[:limit]:
            if query_type == "entity_timeline":
                segment_ranges = [
                    {
                        "segment_index": segment["segment_index"],
                        "segment_count": segment["segment_count"],
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "scene_id": segment.get("scene_id"),
                        "resumes_after_gap": segment.get("resumes_after_gap", False),
                        "gap_from_previous_frames": segment.get("gap_from_previous_frames"),
                        "gap_from_previous_ranks": segment.get("gap_from_previous_ranks"),
                    }
                    for segment in match.get("evidence", {}).get("entity_segments", [])
                ]
                digest.append(
                    {
                        "entity_id": match["entity_id"],
                        "category": match["category"],
                        "first_seen_frame": match["first_seen_frame"],
                        "last_seen_frame": match["last_seen_frame"],
                        "observation_count": match["observation_count"],
                        "segment_count": match.get("segment_count", len(segment_ranges)),
                        "reappearance_count": match.get("reappearance_count", 0),
                        "max_gap_frames": match.get("max_gap_frames", 0),
                        "max_gap_ranks": match.get("max_gap_ranks", 0),
                        "segment_ranges": segment_ranges,
                    }
                )
            elif query_type == "state_history":
                transition_events = list(
                    match.get("evidence", {}).get("transition_events", [])
                )
                digest.append(
                    {
                        "entity_id": match["entity_id"],
                        "entity_category": match["entity_category"],
                        "state_episode_id": match["state_episode_id"],
                        "state_value": match["state_value"],
                        "start_frame": match["start_frame"],
                        "end_frame": match["end_frame"],
                        "start_ts": match["start_ts"],
                        "end_ts": match["end_ts"],
                        "support_frames": match["support_frames"],
                        "transition_event_count": len(transition_events),
                        "transition_frames": [
                            event.get("frame_index")
                            for event in transition_events[:4]
                            if isinstance(event.get("frame_index"), int)
                        ],
                        "is_instantaneous": (
                            match["support_frames"] == 1
                            or match["start_frame"] == match["end_frame"]
                        ),
                    }
                )
            elif query_type == "relation_history":
                transition_events = list(
                    match.get("evidence", {}).get("transition_events", [])
                )
                digest.append(
                    {
                        "relation_episode_id": match["relation_episode_id"],
                        "predicate": match["predicate"],
                        "raw_predicate": match.get("raw_predicate"),
                        "subject_entity_id": match["subject"]["entity_id"],
                        "object_entity_id": match["object"]["entity_id"],
                        "start_frame": match["start_frame"],
                        "end_frame": match["end_frame"],
                        "start_ts": match["start_ts"],
                        "end_ts": match["end_ts"],
                        "support_frames": match["support_frames"],
                        "transition_event_count": len(transition_events),
                        "transition_frames": [
                            event.get("frame_index")
                            for event in transition_events[:4]
                            if isinstance(event.get("frame_index"), int)
                        ],
                        "is_instantaneous": (
                            match["support_frames"] == 1
                            or match["start_frame"] == match["end_frame"]
                        ),
                    }
                )
            elif query_type == "scene_summary":
                scene_summary = match["scene_delta_summary"]
                digest.append(
                    {
                        "scene_id": match["scene_id"],
                        "scene_index": scene_summary.get("scene_index"),
                        "scene_count": scene_summary.get("scene_count"),
                        "start_frame": match["start_frame"],
                        "end_frame": match["end_frame"],
                        "scene_kind_hint": scene_summary.get("scene_kind_hint"),
                        "previous_scene_id": scene_summary.get("previous_scene_id"),
                        "next_scene_id": scene_summary.get("next_scene_id"),
                        "boundary_from_previous_reason": scene_summary.get(
                            "boundary_from_previous_reason"
                        ),
                        "boundary_to_next_reason": scene_summary.get(
                            "boundary_to_next_reason"
                        ),
                        "boundary_from_previous_label_similarity": scene_summary.get(
                            "boundary_from_previous_label_similarity"
                        ),
                        "boundary_to_next_label_similarity": scene_summary.get(
                            "boundary_to_next_label_similarity"
                        ),
                        "entry_static_labels": scene_summary.get("entry_static_labels", ()),
                        "exit_static_labels": scene_summary.get("exit_static_labels", ()),
                        "representative_static_labels": scene_summary.get(
                            "representative_static_labels",
                            (),
                        ),
                        "representative_dynamic_labels": scene_summary.get(
                            "representative_dynamic_labels",
                            (),
                        ),
                        "active_delta_labels": scene_summary.get(
                            "active_delta_labels",
                            (),
                        ),
                        "key_transition_frames": scene_summary.get(
                            "key_transition_frames",
                            (),
                        ),
                        "active_delta_entity_ids": scene_summary["active_delta_entity_ids"],
                        "transition_event_count": scene_summary["transition_event_count"],
                        "transition_event_type_breakdown": scene_summary.get(
                            "transition_event_type_breakdown",
                            {},
                        ),
                        "scene_base_node_count": scene_summary["scene_base_node_count"],
                        "scene_delta_node_count": scene_summary["scene_delta_node_count"],
                    }
                )
            elif query_type == "frame_snapshot":
                scene_summary = match["scene_delta_summary"]
                active_states = list(match.get("evidence", {}).get("active_states", []))
                active_relations = list(
                    match.get("evidence", {}).get("active_relations", [])
                )
                transition_events = list(
                    match.get("evidence", {}).get("transition_events", [])
                )
                digest.append(
                    {
                        "frame_index": match["frame_index"],
                        "scene_id": match["scene_id"],
                        "scene_base_node_count": scene_summary["scene_base_node_count"],
                        "active_delta_node_count": scene_summary["active_delta_node_count"],
                        "active_delta_entity_ids": scene_summary["active_delta_entity_ids"],
                        "scene_kind_hint": scene_summary.get("scene_kind_hint"),
                        "representative_static_labels": scene_summary.get(
                            "representative_static_labels",
                            (),
                        ),
                        "representative_dynamic_labels": scene_summary.get(
                            "representative_dynamic_labels",
                            (),
                        ),
                        "active_state_count": scene_summary["state_episode_count"],
                        "active_relation_count": scene_summary["relation_episode_count"],
                        "transition_event_count": scene_summary["transition_event_count"],
                        "transition_event_type_breakdown": scene_summary.get(
                            "transition_event_type_breakdown",
                            {},
                        ),
                        "active_state_preview": [
                            _describe_state_snapshot(item)
                            for item in active_states[:3]
                        ],
                        "active_relation_preview": [
                            _describe_relation_snapshot(item)
                            for item in active_relations[:3]
                        ],
                        "transition_event_preview": [
                            _describe_transition_event(event)
                            for event in _select_transition_event_highlights(
                                transition_events,
                                limit=3,
                            )
                        ],
                    }
                )
        return digest

    def _register_focus_frames(
        self,
        *,
        query_type: str,
        match: dict[str, Any],
        focus_frame_reasons: dict[int, set[str]],
    ) -> None:
        if query_type == "entity_timeline":
            self._add_focus_frame(
                focus_frame_reasons,
                match.get("first_seen_frame"),
                "entity_first_seen",
            )
            self._add_focus_frame(
                focus_frame_reasons,
                match.get("last_seen_frame"),
                "entity_last_seen",
            )
            evidence = match.get("evidence", {})
            for segment in evidence.get("entity_segments", []):
                self._add_focus_frame(
                    focus_frame_reasons,
                    segment.get("start_frame"),
                    "entity_segment_start",
                )
                self._add_focus_frame(
                    focus_frame_reasons,
                    segment.get("end_frame"),
                    "entity_segment_end",
                )
            for event in evidence.get("transition_events", []):
                self._add_focus_frame(
                    focus_frame_reasons,
                    event.get("frame_index"),
                    "transition_event",
                )
            for key in ("first_observation", "last_observation"):
                observation = evidence.get(key)
                if observation is None:
                    continue
                self._add_focus_frame(
                    focus_frame_reasons,
                    observation.get("frame_index"),
                    key,
                )
            return

        if query_type == "scene_summary":
            self._add_focus_frame(
                focus_frame_reasons,
                match.get("start_frame"),
                "scene_start",
            )
            self._add_focus_frame(
                focus_frame_reasons,
                match.get("end_frame"),
                "scene_end",
            )
            for event in match.get("scene_delta_summary", {}).get(
                "transition_event_type_breakdown",
                {},
            ):
                _ = event
            return

        if query_type == "frame_snapshot":
            self._add_focus_frame(
                focus_frame_reasons,
                match.get("frame_index"),
                "snapshot_frame",
            )
            evidence = match.get("evidence", {})
            for event in evidence.get("transition_events", []):
                self._add_focus_frame(
                    focus_frame_reasons,
                    event.get("frame_index"),
                    "transition_event",
                )
            return

        self._add_focus_frame(
            focus_frame_reasons,
            match.get("start_frame"),
            "episode_start",
        )
        self._add_focus_frame(
            focus_frame_reasons,
            match.get("end_frame"),
            "episode_end",
        )
        evidence = match.get("evidence", {})
        for event in evidence.get("transition_events", []):
            self._add_focus_frame(
                focus_frame_reasons,
                event.get("frame_index"),
                "transition_event",
            )

    def _add_focus_frame(
        self,
        focus_frame_reasons: dict[int, set[str]],
        frame_index: Any,
        reason: str,
    ) -> None:
        if not isinstance(frame_index, int):
            return
        focus_frame_reasons[frame_index].add(reason)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _describe_state_snapshot(episode: dict[str, Any]) -> str:
    state_value = (
        episode.get("state_value")
        or episode.get("attr_value")
        or episode.get("current_state_value")
    )
    if not state_value:
        attr_key = str(episode.get("attr_key") or "").strip()
        if attr_key and attr_key.lower() != "state":
            state_value = attr_key
    if not state_value:
        return str(episode.get("entity_id"))
    return f"{episode.get('entity_id')}:{state_value}"


def _describe_relation_snapshot(episode: dict[str, Any]) -> str:
    return (
        f"{episode.get('subject_entity_id')}:{episode.get('predicate')}:"
        f"{episode.get('object_entity_id')}"
    )


def _describe_transition_event(event: dict[str, Any]) -> str:
    frame_index = event.get("frame_index")
    event_type = str(event.get("event_type", "UNKNOWN")).strip()
    prefix = f"frame={frame_index} {event_type}"
    if event_type in {"APPEAR", "DISAPPEAR"}:
        return f"{prefix} entity={event.get('entity_id')}"
    if event_type == "STATE_TRANSITION":
        return (
            f"{prefix} entity={event.get('entity_id')} "
            f"{event.get('previous_state_value')}->{event.get('current_state_value')}"
        )
    if event_type in {"RELATION_START", "RELATION_END", "RELATION_OBSERVED"}:
        return (
            f"{prefix} subject={event.get('subject_entity_id')} "
            f"predicate={event.get('predicate')} object={event.get('object_entity_id')}"
        )
    return prefix


def _select_transition_event_highlights(
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    priority = {
        "STATE_TRANSITION": 0,
        "RELATION_START": 1,
        "RELATION_END": 2,
        "RELATION_OBSERVED": 3,
        "APPEAR": 4,
        "DISAPPEAR": 5,
    }
    ordered = sorted(
        events,
        key=lambda item: (
            priority.get(str(item.get("event_type", "")).strip(), 99),
            item.get("frame_index", 0),
            str(item.get("entity_id") or item.get("subject_entity_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for event in ordered:
        key = (
            event.get("event_type"),
            event.get("entity_id"),
            event.get("subject_entity_id"),
            event.get("predicate"),
            event.get("object_entity_id"),
            event.get("previous_state_value"),
            event.get("current_state_value"),
        )
        if key in seen_keys:
            continue
        selected.append(event)
        seen_keys.add(key)
        if len(selected) >= limit:
            break
    return selected
