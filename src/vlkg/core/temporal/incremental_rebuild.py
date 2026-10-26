from __future__ import annotations

import json
from dataclasses import dataclass

from vlkg.core.temporal.delta_update import IncrementalExecutionResult
from vlkg.core.temporal.delta_update import DeltaUpdatePlanner
from vlkg.core.models.temporal_graph import TemporalGraphDocument


@dataclass(frozen=True)
class OfflineIncrementalRebuildPlan:
    video_id: str
    existing_frame_count: int
    refreshed_frame_count: int
    existing_last_frame: int | None
    refreshed_last_frame: int | None
    existing_frame_indices: tuple[int, ...]
    refreshed_frame_indices: tuple[int, ...]
    common_frame_indices: tuple[int, ...]
    changed_frame_indices: tuple[int, ...]
    directly_changed_entity_ids: tuple[str, ...]
    appended_frame_indices: tuple[int, ...]
    removed_frame_indices: tuple[int, ...]
    non_prefix_new_frame_indices: tuple[int, ...]
    divergence_start_frame: int | None
    rebuild_start_frame: int | None
    rebuild_end_frame: int | None
    update_pattern: str
    rebuild_mode: str
    execution_scope: str
    append_only: bool
    is_noop: bool
    scene_segments_stable: bool
    supported: bool
    unsupported_reason: str | None
    lookback_frame_count: int


@dataclass(frozen=True)
class OfflineIncrementalRebuildResult:
    plan: OfflineIncrementalRebuildPlan
    refreshed_document: TemporalGraphDocument
    updated_document: TemporalGraphDocument
    execution_result: IncrementalExecutionResult | None


class OfflineIncrementalRebuilder:
    def __init__(
        self,
        *,
        existing_document: TemporalGraphDocument,
        refreshed_document: TemporalGraphDocument,
    ) -> None:
        if existing_document.video_id != refreshed_document.video_id:
            raise ValueError(
                "existing_document.video_id must match refreshed_document.video_id"
            )
        self.existing_document = existing_document
        self.refreshed_document = refreshed_document

    def build_plan(
        self,
        *,
        lookback_frame_count: int = 1,
    ) -> OfflineIncrementalRebuildPlan:
        if lookback_frame_count < 0:
            raise ValueError("lookback_frame_count must be >= 0")

        existing_frame_indices = self._frame_indices(self.existing_document)
        refreshed_frame_indices = self._frame_indices(self.refreshed_document)
        existing_frame_index_set = set(existing_frame_indices)
        refreshed_frame_index_set = set(refreshed_frame_indices)
        common_frame_indices = tuple(
            frame_index
            for frame_index in existing_frame_indices
            if frame_index in refreshed_frame_index_set
        )
        append_only = (
            len(existing_frame_indices) < len(refreshed_frame_indices)
            and tuple(refreshed_frame_indices[: len(existing_frame_indices)])
            == tuple(existing_frame_indices)
        )
        same_frame_set = tuple(existing_frame_indices) == tuple(refreshed_frame_indices)
        removal_only = (
            len(refreshed_frame_indices) < len(existing_frame_indices)
            and all(
                frame_index in existing_frame_index_set
                for frame_index in refreshed_frame_indices
            )
        )
        use_observation_only_diff = append_only or removal_only or not same_frame_set
        changed_frame_indices = (
            self._changed_observation_frame_indices(
                common_frame_indices=common_frame_indices,
            )
            if use_observation_only_diff
            else self._changed_frame_indices(
                common_frame_indices=common_frame_indices,
            )
        )
        scene_drift_changed_frame_indices = (
            self._scene_drift_changed_frame_indices(
                common_frame_indices=common_frame_indices,
            )
            if same_frame_set
            else tuple()
        )
        scene_segments_stable = self._scene_segments_stable()
        if same_frame_set and not scene_segments_stable and scene_drift_changed_frame_indices:
            changed_frame_indices = scene_drift_changed_frame_indices
        appended_frame_indices = tuple(
            frame_index
            for frame_index in refreshed_frame_indices[len(existing_frame_indices) :]
        ) if append_only else tuple()
        removed_frame_indices = tuple(
            frame_index
            for frame_index in existing_frame_indices
            if frame_index not in refreshed_frame_index_set
        )
        non_prefix_new_frame_indices = tuple(
            frame_index
            for frame_index in refreshed_frame_indices
            if frame_index not in existing_frame_index_set
            and frame_index not in appended_frame_indices
        )
        divergence_candidates = sorted(
            {
                *changed_frame_indices,
                *appended_frame_indices,
                *removed_frame_indices,
                *non_prefix_new_frame_indices,
            }
        )
        divergence_start_frame = (
            divergence_candidates[0]
            if divergence_candidates
            else None
        )
        is_noop = (
            same_frame_set
            and not changed_frame_indices
        ) or (
            append_only
            and not changed_frame_indices
            and not appended_frame_indices
        )

        rebuild_mode = "unsupported"
        update_pattern = "non_prefix_update"
        execution_scope = "unsupported"
        supported = False
        unsupported_reason: str | None = None
        if is_noop:
            update_pattern = "noop"
            rebuild_mode = "noop"
            execution_scope = "noop"
            supported = True
        elif append_only:
            update_pattern = "append_only"
            rebuild_mode = (
                "append_with_patch"
                if changed_frame_indices
                else "append_only"
            )
            execution_scope = "window_incremental"
            supported = True
        elif removal_only and removed_frame_indices and not non_prefix_new_frame_indices:
            update_pattern = "frame_removal"
            rebuild_mode = "frame_removal_refresh"
            execution_scope = "window_incremental"
            supported = True
        elif same_frame_set and scene_segments_stable:
            update_pattern = "same_frame_patch"
            rebuild_mode = "inplace_patch"
            execution_scope = "window_incremental"
            supported = True
        elif same_frame_set:
            update_pattern = "scene_drift"
            rebuild_mode = "scene_resegment_incremental"
            execution_scope = "window_incremental"
            supported = True
        elif not same_frame_set:
            update_pattern = "non_prefix_update"
            rebuild_mode = "non_prefix_refresh"
            execution_scope = "window_incremental"
            supported = True

        rebuild_start_frame: int | None = None
        rebuild_end_frame: int | None = None
        directly_changed_entity_ids: tuple[str, ...] = tuple()
        trigger_frame_indices = tuple(
            sorted({*changed_frame_indices, *appended_frame_indices})
        )
        if supported and update_pattern == "same_frame_patch":
            directly_changed_entity_ids = self._directly_changed_entity_ids(
                frame_indices=changed_frame_indices,
            )
        if supported and trigger_frame_indices:
            first_trigger_position = refreshed_frame_indices.index(trigger_frame_indices[0])
            start_position = max(0, first_trigger_position - lookback_frame_count)
            rebuild_start_frame = refreshed_frame_indices[start_position]
            rebuild_end_frame = (
                refreshed_frame_indices[-1]
                if append_only
                else trigger_frame_indices[-1]
            )
        elif supported and rebuild_mode == "frame_removal_refresh" and removed_frame_indices:
            rebuild_start_frame = self._window_start_frame(
                frame_indices=refreshed_frame_indices,
                divergence_start_frame=divergence_start_frame,
                lookback_frame_count=lookback_frame_count,
            )
            rebuild_end_frame = max(
                [*existing_frame_indices, *refreshed_frame_indices],
                default=None,
            )
        elif supported and rebuild_mode == "scene_resegment_incremental" and refreshed_frame_indices:
            rebuild_start_frame = self._window_start_frame(
                frame_indices=refreshed_frame_indices,
                divergence_start_frame=divergence_start_frame,
                lookback_frame_count=lookback_frame_count,
            )
            rebuild_end_frame = changed_frame_indices[-1] if changed_frame_indices else refreshed_frame_indices[-1]
        elif supported and rebuild_mode == "non_prefix_refresh":
            candidate_starts = [
                *removed_frame_indices,
                *non_prefix_new_frame_indices,
            ]
            rebuild_start_frame = self._window_start_frame(
                frame_indices=refreshed_frame_indices,
                divergence_start_frame=(
                    min(candidate_starts)
                    if candidate_starts
                    else divergence_start_frame
                ),
                lookback_frame_count=lookback_frame_count,
            )
            rebuild_end_frame = (
                max(
                    [*existing_frame_indices, *refreshed_frame_indices],
                    default=None,
                )
            )

        return OfflineIncrementalRebuildPlan(
            video_id=self.existing_document.video_id,
            existing_frame_count=len(existing_frame_indices),
            refreshed_frame_count=len(refreshed_frame_indices),
            existing_last_frame=existing_frame_indices[-1] if existing_frame_indices else None,
            refreshed_last_frame=(
                refreshed_frame_indices[-1] if refreshed_frame_indices else None
            ),
            existing_frame_indices=tuple(existing_frame_indices),
            refreshed_frame_indices=tuple(refreshed_frame_indices),
            common_frame_indices=common_frame_indices,
            changed_frame_indices=changed_frame_indices,
            directly_changed_entity_ids=directly_changed_entity_ids,
            appended_frame_indices=appended_frame_indices,
            removed_frame_indices=removed_frame_indices,
            non_prefix_new_frame_indices=non_prefix_new_frame_indices,
            divergence_start_frame=divergence_start_frame,
            rebuild_start_frame=rebuild_start_frame,
            rebuild_end_frame=rebuild_end_frame,
            update_pattern=update_pattern,
            rebuild_mode=rebuild_mode,
            execution_scope=execution_scope,
            append_only=append_only,
            is_noop=is_noop,
            scene_segments_stable=scene_segments_stable,
            supported=supported,
            unsupported_reason=unsupported_reason,
            lookback_frame_count=lookback_frame_count,
        )

    def rebuild(
        self,
        *,
        lookback_frame_count: int = 1,
    ) -> OfflineIncrementalRebuildResult:
        plan = self.build_plan(lookback_frame_count=lookback_frame_count)
        if not plan.supported:
            raise ValueError(
                plan.unsupported_reason
                or "OfflineIncrementalRebuilder cannot incrementally rebuild this update."
            )
        if plan.is_noop:
            return OfflineIncrementalRebuildResult(
                plan=plan,
                refreshed_document=self.refreshed_document,
                updated_document=self.refreshed_document,
                execution_result=None,
            )
        if plan.rebuild_start_frame is None or plan.rebuild_end_frame is None:
            raise ValueError("Rebuild window could not be determined.")

        planner = DeltaUpdatePlanner.build(self.refreshed_document)
        execution_result = planner.execute_window_update(
            start_frame=plan.rebuild_start_frame,
            end_frame=plan.rebuild_end_frame,
            target_document=self.existing_document,
            scoped_entity_ids=(
                plan.directly_changed_entity_ids
                if plan.directly_changed_entity_ids
                else None
            ),
            preserve_scene_segments=(
                plan.update_pattern == "same_frame_patch"
                and plan.scene_segments_stable
            ),
        )
        affected_entity_ids = self._affected_entity_ids_for_window(
            start_frame=plan.rebuild_start_frame,
            end_frame=plan.rebuild_end_frame,
        )
        updated_document = self._normalize_entity_histories(
            document=execution_result.updated_document,
            affected_entity_ids=affected_entity_ids,
            start_frame=plan.rebuild_start_frame,
            end_frame=plan.rebuild_end_frame,
        )
        if plan.rebuild_mode == "scene_resegment_incremental":
            updated_document = self._normalize_scene_histories(
                document=updated_document,
                start_frame=plan.rebuild_start_frame,
                end_frame=plan.rebuild_end_frame,
            )
        return OfflineIncrementalRebuildResult(
            plan=plan,
            refreshed_document=self.refreshed_document,
            updated_document=updated_document,
            execution_result=execution_result,
        )

    def _frame_indices(
        self,
        document: TemporalGraphDocument,
    ) -> list[int]:
        frame_indices = {
            observation.frame_index
            for observation in document.observations
        }
        if not frame_indices:
            frame_indices.update(
                scene_segment.start_frame
                for scene_segment in document.scene_segments
            )
            frame_indices.update(
                scene_segment.end_frame
                for scene_segment in document.scene_segments
        )
        return sorted(frame_indices)

    def _window_start_frame(
        self,
        *,
        frame_indices: tuple[int, ...] | list[int],
        divergence_start_frame: int | None,
        lookback_frame_count: int,
    ) -> int | None:
        ordered = list(frame_indices)
        if not ordered:
            return None
        if divergence_start_frame is None:
            return ordered[0]
        first_position = next(
            (
                index
                for index, frame_index in enumerate(ordered)
                if frame_index >= divergence_start_frame
            ),
            len(ordered) - 1,
        )
        start_position = max(0, first_position - lookback_frame_count)
        return ordered[start_position]

    def _changed_frame_indices(
        self,
        *,
        common_frame_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        existing_signatures = self._frame_signatures(self.existing_document)
        refreshed_signatures = self._frame_signatures(self.refreshed_document)
        changed_frame_indices = [
            frame_index
            for frame_index in common_frame_indices
            if existing_signatures.get(frame_index) != refreshed_signatures.get(frame_index)
        ]
        return tuple(changed_frame_indices)

    def _changed_observation_frame_indices(
        self,
        *,
        common_frame_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        existing_signatures = self._observation_frame_signatures(self.existing_document)
        refreshed_signatures = self._observation_frame_signatures(self.refreshed_document)
        changed_frame_indices = [
            frame_index
            for frame_index in common_frame_indices
            if existing_signatures.get(frame_index) != refreshed_signatures.get(frame_index)
        ]
        return tuple(changed_frame_indices)

    def _scene_drift_changed_frame_indices(
        self,
        *,
        common_frame_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        observation_changed_frame_indices = self._changed_observation_frame_indices(
            common_frame_indices=common_frame_indices,
        )
        scene_assignment_changed_frame_indices = self._changed_scene_assignment_frame_indices(
            common_frame_indices=common_frame_indices,
        )
        changed_frame_indices = sorted(
            {
                *observation_changed_frame_indices,
                *scene_assignment_changed_frame_indices,
            }
        )
        return tuple(changed_frame_indices)

    def _changed_scene_assignment_frame_indices(
        self,
        *,
        common_frame_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        existing_scene_ids = self._scene_ids_by_frame(self.existing_document)
        refreshed_scene_ids = self._scene_ids_by_frame(self.refreshed_document)
        changed_frame_indices = [
            frame_index
            for frame_index in common_frame_indices
            if existing_scene_ids.get(frame_index) != refreshed_scene_ids.get(frame_index)
        ]
        return tuple(changed_frame_indices)

    def _scene_ids_by_frame(
        self,
        document: TemporalGraphDocument,
    ) -> dict[int, str | None]:
        frame_indices = self._frame_indices(document)
        scene_ids_by_frame = {
            frame_index: None
            for frame_index in frame_indices
        }
        for segment in document.scene_segments:
            for frame_index in frame_indices:
                if segment.start_frame <= frame_index <= segment.end_frame:
                    scene_ids_by_frame[frame_index] = segment.scene_id
        return scene_ids_by_frame

    def _directly_changed_entity_ids(
        self,
        *,
        frame_indices: tuple[int, ...],
    ) -> tuple[str, ...]:
        if not frame_indices:
            return tuple()
        changed_entity_ids: set[str] = set()
        changed_entity_ids.update(
            self._changed_observation_entity_ids(frame_indices=frame_indices)
        )
        changed_entity_ids.update(
            self._changed_state_entity_ids(frame_indices=frame_indices)
        )
        changed_entity_ids.update(
            self._changed_relation_entity_ids(frame_indices=frame_indices)
        )
        changed_entity_ids.update(
            self._changed_transition_event_entity_ids(frame_indices=frame_indices)
        )
        return tuple(sorted(changed_entity_ids))

    def _changed_observation_entity_ids(
        self,
        *,
        frame_indices: tuple[int, ...],
    ) -> set[str]:
        frame_index_set = set(frame_indices)
        existing_lookup = {
            (item.frame_index, item.object_index): item
            for item in self.existing_document.observations
            if item.frame_index in frame_index_set
        }
        refreshed_lookup = {
            (item.frame_index, item.object_index): item
            for item in self.refreshed_document.observations
            if item.frame_index in frame_index_set
        }
        changed_entity_ids: set[str] = set()
        for key in set(existing_lookup) | set(refreshed_lookup):
            existing_item = existing_lookup.get(key)
            refreshed_item = refreshed_lookup.get(key)
            existing_signature = (
                self._raw_observation_signature(existing_item)
                if existing_item is not None
                else None
            )
            refreshed_signature = (
                self._raw_observation_signature(refreshed_item)
                if refreshed_item is not None
                else None
            )
            if existing_signature == refreshed_signature:
                continue
            if existing_item is not None:
                changed_entity_ids.add(existing_item.entity_id)
            if refreshed_item is not None:
                changed_entity_ids.add(refreshed_item.entity_id)
        return changed_entity_ids

    def _changed_state_entity_ids(
        self,
        *,
        frame_indices: tuple[int, ...],
    ) -> set[str]:
        frame_index_set = set(frame_indices)
        existing_signatures = {
            self._model_signature(item, exclude={"video_id"}): item.entity_id
            for item in self.existing_document.state_episodes
            if any(item.start_frame <= frame_index <= item.end_frame for frame_index in frame_index_set)
        }
        refreshed_signatures = {
            self._model_signature(item, exclude={"video_id"}): item.entity_id
            for item in self.refreshed_document.state_episodes
            if any(item.start_frame <= frame_index <= item.end_frame for frame_index in frame_index_set)
        }
        changed_signatures = set(existing_signatures) ^ set(refreshed_signatures)
        return {
            entity_id
            for signature in changed_signatures
            for entity_id in (
                existing_signatures.get(signature),
                refreshed_signatures.get(signature),
            )
            if entity_id is not None
        }

    def _changed_relation_entity_ids(
        self,
        *,
        frame_indices: tuple[int, ...],
    ) -> set[str]:
        frame_index_set = set(frame_indices)
        existing_signatures = {
            self._model_signature(item, exclude={"video_id"}): (
                item.subject_entity_id,
                item.object_entity_id,
            )
            for item in self.existing_document.relation_episodes
            if any(item.start_frame <= frame_index <= item.end_frame for frame_index in frame_index_set)
        }
        refreshed_signatures = {
            self._model_signature(item, exclude={"video_id"}): (
                item.subject_entity_id,
                item.object_entity_id,
            )
            for item in self.refreshed_document.relation_episodes
            if any(item.start_frame <= frame_index <= item.end_frame for frame_index in frame_index_set)
        }
        changed_signatures = set(existing_signatures) ^ set(refreshed_signatures)
        changed_entity_ids: set[str] = set()
        for signature in changed_signatures:
            for entity_pair in (
                existing_signatures.get(signature),
                refreshed_signatures.get(signature),
            ):
                if entity_pair is None:
                    continue
                changed_entity_ids.update(entity_pair)
        return changed_entity_ids

    def _changed_transition_event_entity_ids(
        self,
        *,
        frame_indices: tuple[int, ...],
    ) -> set[str]:
        frame_index_set = set(frame_indices)
        existing_signatures = {
            self._model_signature(item, exclude={"video_id"}): self._event_entity_ids(item)
            for item in self.existing_document.transition_events
            if item.frame_index in frame_index_set
        }
        refreshed_signatures = {
            self._model_signature(item, exclude={"video_id"}): self._event_entity_ids(item)
            for item in self.refreshed_document.transition_events
            if item.frame_index in frame_index_set
        }
        changed_signatures = set(existing_signatures) ^ set(refreshed_signatures)
        changed_entity_ids: set[str] = set()
        for signature in changed_signatures:
            changed_entity_ids.update(existing_signatures.get(signature, set()))
            changed_entity_ids.update(refreshed_signatures.get(signature, set()))
        return changed_entity_ids

    def _frame_signatures(
        self,
        document: TemporalGraphDocument,
    ) -> dict[int, str]:
        frame_indices = self._frame_indices(document)
        scene_by_frame = {
            frame_index: None
            for frame_index in frame_indices
        }
        for segment in document.scene_segments:
            for frame_index in frame_indices:
                if segment.start_frame <= frame_index <= segment.end_frame:
                    scene_by_frame[frame_index] = {
                        "scene_id": segment.scene_id,
                        "start_frame": segment.start_frame,
                        "end_frame": segment.end_frame,
                    }

        signatures: dict[int, str] = {}
        for frame_index in frame_indices:
            payload = {
                "scene": scene_by_frame.get(frame_index),
                "observations": sorted(
                    self._observation_signature(item)
                    for item in document.observations
                    if item.frame_index == frame_index
                ),
                "entity_segments": sorted(
                    self._model_signature(
                        item,
                        exclude={"video_id"},
                    )
                    for item in document.entity_segments
                    if item.start_frame <= frame_index <= item.end_frame
                ),
                "state_episodes": sorted(
                    self._model_signature(
                        item,
                        exclude={"video_id"},
                    )
                    for item in document.state_episodes
                    if item.start_frame <= frame_index <= item.end_frame
                ),
                "relation_episodes": sorted(
                    self._model_signature(
                        item,
                        exclude={"video_id"},
                    )
                    for item in document.relation_episodes
                    if item.start_frame <= frame_index <= item.end_frame
                ),
                "scene_delta_observations": sorted(
                    self._scene_delta_observation_signature(item)
                    for item in document.scene_delta_observations
                    if item.frame_index == frame_index
                ),
                "transition_events": sorted(
                    self._model_signature(
                        item,
                        exclude={"video_id"},
                    )
                    for item in document.transition_events
                    if item.frame_index == frame_index
                ),
            }
            signatures[frame_index] = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        return signatures

    def _observation_frame_signatures(
        self,
        document: TemporalGraphDocument,
    ) -> dict[int, str]:
        frame_indices = self._frame_indices(document)
        signatures: dict[int, str] = {}
        for frame_index in frame_indices:
            payload = {
                "observations": sorted(
                    self._raw_observation_signature(item)
                    for item in document.observations
                    if item.frame_index == frame_index
                ),
            }
            signatures[frame_index] = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        return signatures

    def _scene_segments_stable(self) -> bool:
        return [
            self._model_signature(
                segment,
                exclude={"video_id"},
            )
            for segment in self.existing_document.scene_segments
        ] == [
            self._model_signature(
                segment,
                exclude={"video_id"},
            )
            for segment in self.refreshed_document.scene_segments
        ]

    def _normalize_entity_histories(
        self,
        *,
        document: TemporalGraphDocument,
        affected_entity_ids: set[str],
        start_frame: int,
        end_frame: int,
    ) -> TemporalGraphDocument:
        ranges = self._affected_scene_ranges(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        affected_scene_ids = self._affected_scene_ids_for_ranges(
            document=self.refreshed_document,
            ranges=ranges,
        ) | self._affected_scene_ids_for_ranges(
            document=self.existing_document,
            ranges=ranges,
        )
        entities = self._replace_collection(
            existing=document.entities,
            additions=[
                item
                for item in self.refreshed_document.entities
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.first_seen_frame, item.entity_id),
        )
        # Observation-level IDs can drift after longer-range association updates,
        # so rehydrate the full observation history for affected entities.
        observations = self._replace_collection(
            existing=document.observations,
            additions=[
                item
                for item in self.refreshed_document.observations
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.frame_index,
                item.frame_rank,
                item.entity_id,
                item.observation_id,
            ),
        )
        entity_segments = self._replace_collection(
            existing=document.entity_segments,
            additions=[
                item
                for item in self.refreshed_document.entity_segments
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.start_frame, item.end_frame, item.entity_id, item.segment_id),
        )
        scene_base_nodes = self._replace_collection(
            existing=document.scene_base_nodes,
            additions=[
                item
                for item in self.refreshed_document.scene_base_nodes
                if (
                    item.entity_id in affected_entity_ids
                    and item.scene_id in affected_scene_ids
                )
            ],
            should_remove=lambda item: (
                item.entity_id in affected_entity_ids
                and item.scene_id in affected_scene_ids
            ),
            sort_key=lambda item: (item.scene_id, item.entity_id, item.node_id),
        )
        scene_base_edges = self._replace_collection(
            existing=document.scene_base_edges,
            additions=[
                item
                for item in self.refreshed_document.scene_base_edges
                if (
                    item.entity_id in affected_entity_ids
                    and item.scene_id in affected_scene_ids
                )
            ],
            should_remove=lambda item: (
                item.entity_id in affected_entity_ids
                and item.scene_id in affected_scene_ids
            ),
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        relation_episodes = self._replace_collection(
            existing=document.relation_episodes,
            additions=[
                item
                for item in self.refreshed_document.relation_episodes
                if (
                    item.subject_entity_id in affected_entity_ids
                    or item.object_entity_id in affected_entity_ids
                )
            ],
            should_remove=lambda item: (
                item.subject_entity_id in affected_entity_ids
                or item.object_entity_id in affected_entity_ids
            ),
            sort_key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            ),
        )
        state_episodes = self._replace_collection(
            existing=document.state_episodes,
            additions=[
                item
                for item in self.refreshed_document.state_episodes
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.entity_id,
                item.state_episode_id,
            ),
        )
        scene_delta_nodes = self._replace_collection(
            existing=document.scene_delta_nodes,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_nodes
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.scene_id,
                item.start_frame,
                item.end_frame,
                item.entity_id,
                item.node_id,
            ),
        )
        scene_delta_observations = self._replace_collection(
            existing=document.scene_delta_observations,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_observations
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.frame_index,
                item.scene_id,
                item.entity_id,
                item.delta_observation_id,
            ),
        )
        scene_delta_edges = self._replace_collection(
            existing=document.scene_delta_edges,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_edges
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        transition_events = self._replace_collection(
            existing=document.transition_events,
            additions=[
                item
                for item in self.refreshed_document.transition_events
                if self._event_entity_ids(item) & affected_entity_ids
            ],
            should_remove=lambda item: bool(self._event_entity_ids(item) & affected_entity_ids),
            sort_key=lambda item: (
                item.frame_index,
                item.frame_rank,
                item.event_type,
                item.event_id,
            ),
        )

        return document.model_copy(
            update={
                "entities": entities,
                "observations": observations,
                "relation_episodes": relation_episodes,
                "state_episodes": state_episodes,
                "entity_segments": entity_segments,
                "scene_base_nodes": scene_base_nodes,
                "scene_base_edges": scene_base_edges,
                "scene_delta_nodes": scene_delta_nodes,
                "scene_delta_observations": scene_delta_observations,
                "scene_delta_edges": scene_delta_edges,
                "transition_events": transition_events,
                "source_dir": self.refreshed_document.source_dir,
                "frame_count": self.refreshed_document.frame_count,
            }
        )

    def _refresh_affected_scenes(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> TemporalGraphDocument:
        ranges = self._affected_scene_ranges(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        existing_scene_ids = {
            segment.scene_id
            for segment in self.existing_document.scene_segments
            if self._overlaps_any_range(
                segment.start_frame,
                segment.end_frame,
                ranges=ranges,
            )
        }
        refreshed_scene_ids = {
            segment.scene_id
            for segment in self.refreshed_document.scene_segments
            if self._overlaps_any_range(
                segment.start_frame,
                segment.end_frame,
                ranges=ranges,
            )
        }
        affected_entity_ids = self._affected_entity_ids_for_ranges(
            ranges=ranges,
        )

        observations = self._replace_collection(
            existing=self.existing_document.observations,
            additions=[
                item
                for item in self.refreshed_document.observations
                if self._frame_in_ranges(
                    item.frame_index,
                    ranges=ranges,
                )
            ],
            should_remove=lambda item: self._frame_in_ranges(
                item.frame_index,
                ranges=ranges,
            ),
            sort_key=lambda item: (
                item.frame_index,
                item.frame_rank,
                item.entity_id,
                item.observation_id,
            ),
        )
        entities = self._replace_collection(
            existing=self.existing_document.entities,
            additions=[
                item
                for item in self.refreshed_document.entities
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.first_seen_frame, item.entity_id),
        )
        relation_episodes = self._replace_collection(
            existing=self.existing_document.relation_episodes,
            additions=[
                item
                for item in self.refreshed_document.relation_episodes
                if (
                    item.subject_entity_id in affected_entity_ids
                    or item.object_entity_id in affected_entity_ids
                )
            ],
            should_remove=lambda item: (
                item.subject_entity_id in affected_entity_ids
                or item.object_entity_id in affected_entity_ids
            ),
            sort_key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
                item.relation_episode_id,
            ),
        )
        state_episodes = self._replace_collection(
            existing=self.existing_document.state_episodes,
            additions=[
                item
                for item in self.refreshed_document.state_episodes
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.entity_id,
                item.state_episode_id,
            ),
        )
        scene_segments = self._replace_collection(
            existing=self.existing_document.scene_segments,
            additions=[
                item
                for item in self.refreshed_document.scene_segments
                if self._overlaps_any_range(
                    item.start_frame,
                    item.end_frame,
                    ranges=ranges,
                )
            ],
            should_remove=lambda item: self._overlaps_any_range(
                item.start_frame,
                item.end_frame,
                ranges=ranges,
            ),
            sort_key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
        )
        entity_segments = self._replace_collection(
            existing=self.existing_document.entity_segments,
            additions=[
                item
                for item in self.refreshed_document.entity_segments
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.start_frame, item.end_frame, item.entity_id, item.segment_id),
        )
        scene_base_nodes = self._replace_collection(
            existing=self.existing_document.scene_base_nodes,
            additions=[
                item
                for item in self.refreshed_document.scene_base_nodes
                if item.scene_id in refreshed_scene_ids
            ],
            should_remove=lambda item: item.scene_id in existing_scene_ids,
            sort_key=lambda item: (item.scene_id, item.entity_id, item.node_id),
        )
        scene_base_edges = self._replace_collection(
            existing=self.existing_document.scene_base_edges,
            additions=[
                item
                for item in self.refreshed_document.scene_base_edges
                if item.scene_id in refreshed_scene_ids
            ],
            should_remove=lambda item: item.scene_id in existing_scene_ids,
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        scene_delta_nodes = self._replace_collection(
            existing=self.existing_document.scene_delta_nodes,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_nodes
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.scene_id,
                item.start_frame,
                item.end_frame,
                item.entity_id,
                item.node_id,
            ),
        )
        scene_delta_observations = self._replace_collection(
            existing=self.existing_document.scene_delta_observations,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_observations
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (
                item.frame_index,
                item.scene_id,
                item.entity_id,
                item.delta_observation_id,
            ),
        )
        scene_delta_edges = self._replace_collection(
            existing=self.existing_document.scene_delta_edges,
            additions=[
                item
                for item in self.refreshed_document.scene_delta_edges
                if item.entity_id in affected_entity_ids
            ],
            should_remove=lambda item: item.entity_id in affected_entity_ids,
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        transition_events = self._replace_collection(
            existing=self.existing_document.transition_events,
            additions=[
                item
                for item in self.refreshed_document.transition_events
                if self._frame_in_ranges(
                    item.frame_index,
                    ranges=ranges,
                )
            ],
            should_remove=lambda item: self._frame_in_ranges(
                item.frame_index,
                ranges=ranges,
            ),
            sort_key=lambda item: (
                item.frame_index,
                item.frame_rank,
                item.event_type,
                item.event_id,
            ),
        )
        return self.refreshed_document.model_copy(
            update={
                "observations": observations,
                "entities": entities,
                "relation_episodes": relation_episodes,
                "state_episodes": state_episodes,
                "scene_segments": scene_segments,
                "entity_segments": entity_segments,
                "scene_base_nodes": scene_base_nodes,
                "scene_base_edges": scene_base_edges,
                "scene_delta_nodes": scene_delta_nodes,
                "scene_delta_observations": scene_delta_observations,
                "scene_delta_edges": scene_delta_edges,
                "transition_events": transition_events,
                "source_dir": self.refreshed_document.source_dir,
                "frame_count": self.refreshed_document.frame_count,
            }
        )

    def _normalize_scene_histories(
        self,
        *,
        document: TemporalGraphDocument,
        start_frame: int,
        end_frame: int,
    ) -> TemporalGraphDocument:
        ranges = self._affected_scene_ranges(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        existing_scene_ids = {
            segment.scene_id
            for segment in self.existing_document.scene_segments
            if self._overlaps_any_range(
                segment.start_frame,
                segment.end_frame,
                ranges=ranges,
            )
        }
        refreshed_scene_ids = {
            segment.scene_id
            for segment in self.refreshed_document.scene_segments
            if self._overlaps_any_range(
                segment.start_frame,
                segment.end_frame,
                ranges=ranges,
            )
        }
        scene_segments = self._replace_collection(
            existing=document.scene_segments,
            additions=[
                item
                for item in self.refreshed_document.scene_segments
                if self._overlaps_any_range(
                    item.start_frame,
                    item.end_frame,
                    ranges=ranges,
                )
            ],
            should_remove=lambda item: (
                item.scene_id in existing_scene_ids
                or self._overlaps_any_range(
                    item.start_frame,
                    item.end_frame,
                    ranges=ranges,
                )
            ),
            sort_key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
        )
        scene_base_nodes = self._replace_collection(
            existing=document.scene_base_nodes,
            additions=[
                item
                for item in self.refreshed_document.scene_base_nodes
                if item.scene_id in refreshed_scene_ids
            ],
            should_remove=lambda item: (
                item.scene_id in existing_scene_ids
                or item.scene_id in refreshed_scene_ids
            ),
            sort_key=lambda item: (item.scene_id, item.entity_id, item.node_id),
        )
        scene_base_edges = self._replace_collection(
            existing=document.scene_base_edges,
            additions=[
                item
                for item in self.refreshed_document.scene_base_edges
                if item.scene_id in refreshed_scene_ids
            ],
            should_remove=lambda item: (
                item.scene_id in existing_scene_ids
                or item.scene_id in refreshed_scene_ids
            ),
            sort_key=lambda item: (item.scene_id, item.entity_id, item.edge_id),
        )
        return document.model_copy(
            update={
                "scene_segments": scene_segments,
                "scene_base_nodes": scene_base_nodes,
                "scene_base_edges": scene_base_edges,
                "source_dir": self.refreshed_document.source_dir,
                "frame_count": self.refreshed_document.frame_count,
            }
        )

    def _observation_signature(
        self,
        observation,
    ) -> str:
        return self._model_signature(
            observation,
            exclude={"video_id", "image_path"},
        )

    def _scene_delta_observation_signature(
        self,
        observation,
    ) -> str:
        return self._model_signature(
            observation,
            exclude={"video_id", "image_path"},
        )

    def _raw_observation_signature(
        self,
        observation,
    ) -> str:
        return self._model_signature(
            observation,
            exclude={
                "video_id",
                "image_path",
                "entity_id",
                "scene_id",
                "graph_origin",
            },
        )

    def _model_signature(
        self,
        item,
        *,
        exclude: set[str] | None = None,
    ) -> str:
        return json.dumps(
            item.model_dump(
                mode="json",
                exclude=exclude or set(),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )

    def _replace_collection(
        self,
        *,
        existing: list,
        additions: list,
        should_remove,
        sort_key,
    ) -> list:
        kept = [
            item
            for item in existing
            if not should_remove(item)
        ]
        merged = [*kept, *additions]
        merged.sort(key=sort_key)
        return merged

    def _event_entity_ids(
        self,
        event,
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

    def _overlaps(
        self,
        left_start: int,
        left_end: int,
        right_start: int,
        right_end: int,
    ) -> bool:
        return not (left_end < right_start or right_end < left_start)

    def _overlaps_any_range(
        self,
        left_start: int,
        left_end: int,
        *,
        ranges: list[tuple[int, int]],
    ) -> bool:
        return any(
            self._overlaps(
                left_start,
                left_end,
                range_start,
                range_end,
            )
            for range_start, range_end in ranges
        )

    def _frame_in_ranges(
        self,
        frame_index: int,
        *,
        ranges: list[tuple[int, int]],
    ) -> bool:
        return any(
            range_start <= frame_index <= range_end
            for range_start, range_end in ranges
        )

    def _collapse_ranges(
        self,
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not ranges:
            return []
        ordered = sorted(ranges, key=lambda item: (item[0], item[1]))
        merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
        for start_frame, end_frame in ordered[1:]:
            last = merged[-1]
            if start_frame <= last[1] + 1:
                last[1] = max(last[1], end_frame)
                continue
            merged.append([start_frame, end_frame])
        return [
            (start_frame, end_frame)
            for start_frame, end_frame in merged
        ]

    def _affected_scene_ranges(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> list[tuple[int, int]]:
        ranges = [
            (segment.start_frame, segment.end_frame)
            for document in (self.existing_document, self.refreshed_document)
            for segment in document.scene_segments
            if self._overlaps(
                segment.start_frame,
                segment.end_frame,
                start_frame,
                end_frame,
            )
        ]
        if not ranges:
            ranges = [(start_frame, end_frame)]
        return self._collapse_ranges(ranges)

    def _affected_entity_ids_for_window(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> set[str]:
        return self._affected_entity_ids_for_ranges(
            ranges=[(start_frame, end_frame)],
        )

    def _affected_entity_ids_for_ranges(
        self,
        *,
        ranges: list[tuple[int, int]],
    ) -> set[str]:
        affected_entity_ids: set[str] = set()
        documents = (
            self.existing_document,
            self.refreshed_document,
        )
        for document in documents:
            affected_scene_ids = self._affected_scene_ids_for_ranges(
                document=document,
                ranges=ranges,
            )
            affected_entity_ids.update(
                observation.entity_id
                for observation in document.observations
                if self._frame_in_ranges(
                    observation.frame_index,
                    ranges=ranges,
                )
            )
            affected_entity_ids.update(
                entity.entity_id
                for entity in document.entities
                if self._overlaps_any_range(
                    entity.first_seen_frame,
                    entity.last_seen_frame,
                    ranges=ranges,
                )
            )
            affected_entity_ids.update(
                segment.entity_id
                for segment in document.entity_segments
                if self._overlaps_any_range(
                    segment.start_frame,
                    segment.end_frame,
                    ranges=ranges,
                )
            )
            affected_entity_ids.update(
                episode.entity_id
                for episode in document.state_episodes
                if self._overlaps_any_range(
                    episode.start_frame,
                    episode.end_frame,
                    ranges=ranges,
                )
            )
            for episode in document.relation_episodes:
                if not self._overlaps_any_range(
                    episode.start_frame,
                    episode.end_frame,
                    ranges=ranges,
                ):
                    continue
                affected_entity_ids.add(episode.subject_entity_id)
                affected_entity_ids.add(episode.object_entity_id)
            affected_entity_ids.update(
                node.entity_id
                for node in document.scene_delta_nodes
                if self._overlaps_any_range(
                    node.start_frame,
                    node.end_frame,
                    ranges=ranges,
                )
            )
            affected_entity_ids.update(
                observation.entity_id
                for observation in document.scene_delta_observations
                if self._frame_in_ranges(
                    observation.frame_index,
                    ranges=ranges,
                )
            )
            affected_entity_ids.update(
                edge.entity_id
                for edge in document.scene_delta_edges
                if edge.scene_id in affected_scene_ids
            )
            for event in document.transition_events:
                if self._frame_in_ranges(
                    event.frame_index,
                    ranges=ranges,
                ):
                    affected_entity_ids.update(self._event_entity_ids(event))
        return affected_entity_ids

    def _affected_scene_ids_for_ranges(
        self,
        *,
        document: TemporalGraphDocument,
        ranges: list[tuple[int, int]],
    ) -> set[str]:
        return {
            segment.scene_id
            for segment in document.scene_segments
            if self._overlaps_any_range(
                segment.start_frame,
                segment.end_frame,
                ranges=ranges,
            )
        }
