from __future__ import annotations

from typing import Any


def build_llm_ready_bundle(
    *,
    video_id: str,
    query_type: str,
    filters: dict[str, Any],
    matched_by: Any,
    summary: str,
    matches: list[dict[str, Any]],
    retrieval_bundle: dict[str, Any],
) -> dict[str, Any]:
    answer_brief = _build_answer_brief(query_type=query_type, matches=matches)
    evidence_lines = _build_evidence_lines(query_type=query_type, matches=matches, limit=6)
    focus_frame_lines = _build_focus_frame_lines(retrieval_bundle)
    scene_lines = _build_scene_lines(retrieval_bundle)
    match_highlights = _build_match_highlights(query_type=query_type, matches=matches)
    uncertainty_notes = _build_uncertainty_notes(
        query_type=query_type,
        matches=matches,
        retrieval_bundle=retrieval_bundle,
    )

    llm_context_text = "\n".join(
        line
        for line in [
            f"Video: {video_id}",
            f"Query type: {query_type}",
            f"Matched by: {matched_by}",
            f"Filters: {filters}",
            f"Summary: {summary}",
            f"Answer brief: {answer_brief}",
            "Evidence lines:",
            *[f"- {line}" for line in evidence_lines],
            "Focus frames:",
            *[f"- {line}" for line in focus_frame_lines],
            "Scene notes:",
            *[f"- {line}" for line in scene_lines],
            "Match highlights:",
            *[f"- {line}" for line in match_highlights],
            "Uncertainty notes:",
            *[f"- {line}" for line in uncertainty_notes],
        ]
        if line
    )

    return {
        "video_id": video_id,
        "query_type": query_type,
        "answer_brief": answer_brief,
        "evidence_lines": evidence_lines,
        "focus_frame_lines": focus_frame_lines,
        "scene_lines": scene_lines,
        "match_highlights": match_highlights,
        "uncertainty_notes": uncertainty_notes,
        "llm_context_text": llm_context_text,
        "structured_graph_context": {
            "video_id": video_id,
            "query_type": query_type,
            "matched_by": matched_by,
            "filters": filters,
            "summary": summary,
            "answer_brief": answer_brief,
            "evidence_lines": evidence_lines,
            "focus_frames": retrieval_bundle.get("focus_frames", []),
            "scene_summaries": retrieval_bundle.get("scene_summaries", []),
            "evidence_digest": retrieval_bundle.get("evidence_digest", []),
            "match_highlights": match_highlights,
            "uncertainty_notes": uncertainty_notes,
        },
    }


def _build_answer_brief(*, query_type: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No graph evidence matched the query."

    if query_type == "entity_timeline":
        top = matches[0]
        if len(matches) > 1:
            return (
                f"Found {len(matches)} candidate entities. The earliest starts at frame "
                f"{min(match['first_seen_frame'] for match in matches)} and the latest ends at frame "
                f"{max(match['last_seen_frame'] for match in matches)}."
            )
        segments = top.get("evidence", {}).get("entity_segments", [])
        if len(segments) > 1:
            segment_ranges = ", ".join(
                f"{segment['start_frame']}->{segment['end_frame']}"
                for segment in segments[:4]
            )
            if len(segments) > 4:
                segment_ranges = f"{segment_ranges}, ..."
            reappearance_count = int(top.get("reappearance_count", 0) or 0)
            max_gap_frames = int(top.get("max_gap_frames", 0) or 0)
            return (
                f"Entity {top['entity_id']} ({top['category']}) appears in "
                f"{len(segments)} separate presence segments: {segment_ranges}. "
                f"It reappears after {reappearance_count} absence gap(s), "
                f"with the largest gap spanning {max_gap_frames} frame indices. "
                f"The overall observed span is frame {top['first_seen_frame']} to "
                f"{top['last_seen_frame']} with {top['observation_count']} observations."
            )
        return (
            f"Entity {top['entity_id']} ({top['category']}) appears from frame "
            f"{top['first_seen_frame']} to {top['last_seen_frame']} "
            f"({top['first_seen_ts']:.3f}s to {top['last_seen_ts']:.3f}s), "
            f"with {top['observation_count']} observations."
        )

    if query_type == "state_history":
        top = matches[0]
        if len(matches) > 1:
            states = sorted(
                {
                    str(match.get("state_value", "")).strip()
                    for match in matches
                    if str(match.get("state_value", "")).strip()
                }
            )
            state_suffix = (
                f" Top state values include: {', '.join(states[:3])}."
                if states
                else ""
            )
            return (
                f"Found {len(matches)} matching state intervals. The earliest begins at frame "
                f"{min(match['start_frame'] for match in matches)} and the latest ends at frame "
                f"{max(match['end_frame'] for match in matches)}."
                f"{state_suffix}"
            )
        if _is_instant_episode(top):
            return (
                f"Entity {top['entity_id']} ({top['entity_category']}) is observed in state "
                f"'{top['state_value']}' at frame {top['start_frame']} "
                f"({top['start_ts']:.3f}s)."
            )
        return (
            f"Entity {top['entity_id']} ({top['entity_category']}) is in state "
            f"'{top['state_value']}' from frame {top['start_frame']} to {top['end_frame']} "
            f"({top['start_ts']:.3f}s to {top['end_ts']:.3f}s)."
        )

    if query_type == "relation_history":
        top = matches[0]
        if len(matches) > 1:
            predicates = sorted(
                {
                    str(match.get("predicate", "")).strip()
                    for match in matches
                    if str(match.get("predicate", "")).strip()
                }
            )
            relation_pairs = {
                (
                    str(match.get("subject", {}).get("entity_id", "")).strip(),
                    str(match.get("object", {}).get("entity_id", "")).strip(),
                )
                for match in matches
            }
            relation_pairs = {
                (subject, obj)
                for subject, obj in relation_pairs
                if subject and obj
            }
            subjects = sorted({subject for subject, _ in relation_pairs})
            return (
                f"Found {len(matches)} matching relation episodes across "
                f"{len(relation_pairs)} unique entity pairs for predicates "
                f"{predicates}. The earliest begins at frame "
                f"{min(match['start_frame'] for match in matches)} and the latest ends at frame "
                f"{max(match['end_frame'] for match in matches)}. "
                f"Representative subjects include: {', '.join(subjects[:3])}."
            )
        if _is_instant_episode(top):
            return (
                f"Relation {top['predicate']} is observed between "
                f"{top['subject']['entity_id']} and {top['object']['entity_id']} at frame "
                f"{top['start_frame']} ({top['start_ts']:.3f}s)."
            )
        return (
            f"Relation {top['predicate']} holds between "
            f"{top['subject']['entity_id']} and {top['object']['entity_id']} from frame "
            f"{top['start_frame']} to {top['end_frame']} "
            f"({top['start_ts']:.3f}s to {top['end_ts']:.3f}s)."
        )

    if query_type == "scene_summary":
        top = matches[0]
        if len(matches) > 1:
            return (
                f"Found {len(matches)} matching scene containers. The earliest begins at frame "
                f"{min(match['start_frame'] for match in matches)} and the latest ends at frame "
                f"{max(match['end_frame'] for match in matches)}."
            )
        summary = top["scene_delta_summary"]
        kind_hint = summary.get("scene_kind_hint")
        kind_hint_suffix = f" The scene context looks like: {kind_hint}." if kind_hint else ""
        boundary_suffix = _describe_scene_boundary_brief(summary)
        boundary_suffix = f" {boundary_suffix}" if boundary_suffix else ""
        static_labels = list(summary.get("representative_static_labels", []))
        dynamic_labels = list(summary.get("representative_dynamic_labels", []))
        label_suffix_parts: list[str] = []
        if static_labels:
            label_suffix_parts.append(
                "anchor labels include " + ", ".join(static_labels[:3])
            )
        if dynamic_labels:
            label_suffix_parts.append(
                "dynamic labels include " + ", ".join(dynamic_labels[:3])
            )
        label_suffix = (
            " " + "; ".join(label_suffix_parts) + "."
            if label_suffix_parts
            else ""
        )
        return (
            f"Scene {top['scene_id']} spans frame {top['start_frame']} to {top['end_frame']} "
            f"({top['start_ts']:.3f}s to {top['end_ts']:.3f}s), with "
            f"{summary['scene_base_node_count']} base nodes, "
            f"{summary['scene_delta_node_count']} delta nodes, and "
            f"{summary['transition_event_count']} transition events."
            f"{kind_hint_suffix}{boundary_suffix}{label_suffix}"
        )

    if query_type == "frame_snapshot":
        top = matches[0]
        summary = top["scene_delta_summary"]
        kind_hint = str(summary.get("scene_kind_hint") or "").strip()
        kind_hint_suffix = f" Scene context: {kind_hint}." if kind_hint else ""
        breakdown = _format_transition_breakdown(
            summary.get("transition_event_type_breakdown", {})
        )
        breakdown_suffix = f" Transition profile: {breakdown}." if breakdown else ""
        return (
            f"Frame {top['frame_index']} in scene {top['scene_id']} has "
            f"{summary['scene_base_node_count']} base nodes, "
            f"{summary['active_delta_node_count']} active delta nodes, and "
            f"{summary['transition_event_count']} transition events."
            f"{kind_hint_suffix}{breakdown_suffix}"
        )

    return "Graph evidence was found."


def _build_evidence_lines(
    *,
    query_type: str,
    matches: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for match in matches[:limit]:
        if query_type == "entity_timeline":
            segments = match.get("evidence", {}).get("entity_segments", [])
            if len(segments) > 1:
                segment_descriptions: list[str] = []
                for segment in segments[:4]:
                    description = (
                        f"{segment['segment_index']}:{segment['start_frame']}"
                        f"->{segment['end_frame']}"
                    )
                    gap_from_previous = segment.get("gap_from_previous_frames")
                    if gap_from_previous not in (None, 0):
                        description = f"{description}(gap={gap_from_previous})"
                    segment_descriptions.append(description)
                segment_ranges = ", ".join(segment_descriptions)
                if len(segments) > 4:
                    segment_ranges = f"{segment_ranges}, ..."
                line = (
                    f"{match['entity_id']} ({match['category']}): "
                    f"segments={len(segments)} [{segment_ranges}], "
                    f"reappearances={match.get('reappearance_count', 0)}, "
                    f"max_gap_frames={match.get('max_gap_frames', 0)}, "
                    f"overall={match['first_seen_frame']}->{match['last_seen_frame']}, "
                    f"observations={match['observation_count']}"
                )
            else:
                line = (
                    f"{match['entity_id']} ({match['category']}): frames "
                    f"{match['first_seen_frame']}->{match['last_seen_frame']}, "
                    f"observations={match['observation_count']}"
                )
        elif query_type == "state_history":
            line = _describe_state_history_line(match)
        elif query_type == "relation_history":
            line = _describe_relation_history_line(match)
        elif query_type == "scene_summary":
            summary = match["scene_delta_summary"]
            primary_line = (
                f"{match['scene_id']} frames {match['start_frame']}->{match['end_frame']} "
                f"base_nodes={summary['scene_base_node_count']} "
                f"delta_nodes={summary['scene_delta_node_count']} "
                f"transitions={summary['transition_event_count']}"
            )
            _append_evidence_line(lines, seen, primary_line, limit=limit)
            breakdown = _format_transition_breakdown(
                summary.get("transition_event_type_breakdown", {})
            )
            if breakdown:
                _append_evidence_line(
                    lines,
                    seen,
                    f"{match['scene_id']} transition_breakdown={breakdown}",
                    limit=limit,
                )
            kind_hint = str(summary.get("scene_kind_hint") or "").strip()
            if kind_hint:
                _append_evidence_line(
                    lines,
                    seen,
                    f"{match['scene_id']} scene_kind_hint={kind_hint}",
                    limit=limit,
                )
            boundary_note = _describe_scene_boundary_line(summary)
            if boundary_note:
                _append_evidence_line(
                    lines,
                    seen,
                    f"{match['scene_id']} boundary={boundary_note}",
                    limit=limit,
                )
            static_labels = list(summary.get("representative_static_labels", []))
            dynamic_labels = list(summary.get("representative_dynamic_labels", []))
            if static_labels or dynamic_labels:
                label_parts: list[str] = []
                if static_labels:
                    label_parts.append(
                        "static=" + ",".join(static_labels[:4])
                    )
                if dynamic_labels:
                    label_parts.append(
                        "dynamic=" + ",".join(dynamic_labels[:4])
                    )
                _append_evidence_line(
                    lines,
                    seen,
                    f"{match['scene_id']} labels=" + " ".join(label_parts),
                    limit=limit,
                )
            key_transition_frames = list(summary.get("key_transition_frames", []))
            active_delta_entity_ids = list(summary.get("active_delta_entity_ids", []))
            if key_transition_frames or active_delta_entity_ids:
                focus_parts: list[str] = []
                if key_transition_frames:
                    focus_parts.append(
                        "key_frames="
                        + ",".join(str(frame) for frame in key_transition_frames[:4])
                    )
                if active_delta_entity_ids:
                    preview = ",".join(active_delta_entity_ids[:3])
                    if len(active_delta_entity_ids) > 3:
                        preview = f"{preview},..."
                    focus_parts.append(
                        f"active_delta_entities={len(active_delta_entity_ids)}[{preview}]"
                    )
                _append_evidence_line(
                    lines,
                    seen,
                    f"{match['scene_id']} focus=" + " ".join(focus_parts),
                    limit=limit,
                )
            continue
        elif query_type == "frame_snapshot":
            summary = match["scene_delta_summary"]
            primary_line = (
                f"frame={match['frame_index']} scene={match['scene_id']} "
                f"base_nodes={summary['scene_base_node_count']} "
                f"active_delta_nodes={summary['active_delta_node_count']} "
                f"transitions={summary['transition_event_count']}"
            )
            _append_evidence_line(lines, seen, primary_line, limit=limit)
            active_states = match.get("evidence", {}).get("active_states", [])
            if active_states and len(lines) < limit:
                state_preview = ", ".join(
                    _describe_state_snapshot(item)
                    for item in active_states[:2]
                )
                _append_evidence_line(
                    lines,
                    seen,
                    f"frame={match['frame_index']} active_states={state_preview}",
                    limit=limit,
                )
            active_relations = match.get("evidence", {}).get("active_relations", [])
            if active_relations and len(lines) < limit:
                relation_preview = ", ".join(
                    _describe_relation_snapshot(item)
                    for item in active_relations[:2]
                )
                _append_evidence_line(
                    lines,
                    seen,
                    f"frame={match['frame_index']} active_relations={relation_preview}",
                    limit=limit,
                )
            breakdown = _format_transition_breakdown(
                summary.get("transition_event_type_breakdown", {})
            )
            if breakdown:
                _append_evidence_line(
                    lines,
                    seen,
                    f"frame={match['frame_index']} transition_breakdown={breakdown}",
                    limit=limit,
                )
            for event in _select_transition_event_highlights(
                match.get("evidence", {}).get("transition_events", []),
                limit=max(0, limit - len(lines)),
            ):
                if len(lines) >= limit:
                    break
                _append_evidence_line(
                    lines,
                    seen,
                    _describe_transition_event(event),
                    limit=limit,
                )
            continue
        else:
            continue
        _append_evidence_line(lines, seen, line, limit=limit)
    return lines


def _append_evidence_line(
    lines: list[str],
    seen: set[str],
    line: str,
    *,
    limit: int,
) -> None:
    if len(lines) >= limit:
        return
    normalized = str(line).strip()
    if not normalized or normalized in seen:
        return
    lines.append(normalized)
    seen.add(normalized)


def _build_focus_frame_lines(retrieval_bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in retrieval_bundle.get("focus_frames", []):
        lines.append(
            f"rank={item.get('rank')} frame={item.get('frame_index')} "
            f"ts={item.get('timestamp_sec')} score={item.get('score')} "
            f"reasons={','.join(item.get('reasons', []))}"
        )
    return lines


def _build_scene_lines(retrieval_bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for scene in retrieval_bundle.get("scene_summaries", []):
        kind_hint = str(scene.get("scene_kind_hint") or "").strip()
        kind_hint_suffix = f" kind={kind_hint}" if kind_hint else ""
        static_labels = list(scene.get("representative_static_labels", []))
        dynamic_labels = list(scene.get("representative_dynamic_labels", []))
        key_transition_frames = list(scene.get("key_transition_frames", []))
        details: list[str] = []
        if static_labels:
            details.append("static=" + ",".join(static_labels[:3]))
        if dynamic_labels:
            details.append("dynamic=" + ",".join(dynamic_labels[:3]))
        if key_transition_frames:
            details.append(
                "key_frames=" + ",".join(str(frame) for frame in key_transition_frames[:4])
            )
        boundary_note = _describe_scene_boundary_compact(scene)
        if boundary_note:
            details.append(boundary_note)
        detail_suffix = (" " + " ".join(details)) if details else ""
        lines.append(
            f"scene={scene.get('scene_id')} frames={scene.get('start_frame')}->"
            f"{scene.get('end_frame')} active_delta_entities="
            f"{len(scene.get('active_delta_entity_ids', []))} "
            f"transitions={scene.get('transition_event_count')}{kind_hint_suffix}"
            f"{detail_suffix}"
        )
    if not lines:
        lines.append("No scene summary available for the matched evidence.")
    return lines


def _build_match_highlights(
    *,
    query_type: str,
    matches: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if not matches:
        return lines

    if query_type == "frame_snapshot":
        match = matches[0]
        summary = match.get("scene_delta_summary", {})
        active_states = match.get("evidence", {}).get("active_states", [])
        active_relations = match.get("evidence", {}).get("active_relations", [])
        if active_states:
            preview = ", ".join(_describe_state_snapshot(item) for item in active_states[:3])
            lines.append(f"frame {match['frame_index']} active states: {preview}")
        if active_relations:
            preview = ", ".join(
                _describe_relation_snapshot(item) for item in active_relations[:3]
            )
            lines.append(f"frame {match['frame_index']} active relations: {preview}")
        breakdown = _format_transition_breakdown(
            summary.get("transition_event_type_breakdown", {})
        )
        if breakdown:
            lines.append(
                f"frame {match['frame_index']} change profile: {breakdown}"
            )
        highlights = [
            _describe_transition_event(event)
            for event in _select_transition_event_highlights(
                match.get("evidence", {}).get("transition_events", []),
                limit=4,
            )
        ]
        if highlights:
            lines.append("frame event highlights: " + " | ".join(highlights))
        if not (active_states or active_relations or highlights):
            lines.append(
                f"frame {match['frame_index']} is primarily described by scene-base and delta-node context."
            )
        return lines

    if query_type == "state_history":
        state_values = sorted(
            {
                str(match.get("state_value", "")).strip()
                for match in matches
                if str(match.get("state_value", "")).strip()
            }
        )
        if state_values:
            lines.append("state values: " + ", ".join(state_values[:5]))
        supports = [
            int(match.get("support_frames", 0) or 0)
            for match in matches
            if isinstance(match.get("support_frames"), int)
        ]
        if supports:
            lines.append(
                f"state support range: min={min(supports)} max={max(supports)}"
            )
        return lines

    if query_type == "relation_history":
        predicates = sorted(
            {
                str(match.get("predicate", "")).strip()
                for match in matches
                if str(match.get("predicate", "")).strip()
            }
        )
        if predicates:
            lines.append("predicates: " + ", ".join(predicates[:5]))
        relation_pairs = [
            (
                str(match.get("subject", {}).get("entity_id", "")).strip(),
                str(match.get("object", {}).get("entity_id", "")).strip(),
            )
            for match in matches
        ]
        deduped_pairs: list[str] = []
        seen_pairs: set[str] = set()
        for subject, obj in relation_pairs:
            if not subject or not obj:
                continue
            pair_text = f"{subject}->{obj}"
            if pair_text in seen_pairs:
                continue
            seen_pairs.add(pair_text)
            deduped_pairs.append(pair_text)
        relation_pairs = deduped_pairs
        if relation_pairs:
            lines.append("relation pairs: " + ", ".join(relation_pairs[:4]))
        supports = [
            int(match.get("support_frames", 0) or 0)
            for match in matches
            if isinstance(match.get("support_frames"), int)
        ]
        if supports:
            lines.append(
                f"relation support range: min={min(supports)} max={max(supports)}"
            )
        return lines

    if query_type == "entity_timeline":
        match = matches[0]
        if match.get("segment_count", 1) > 1:
            lines.append(
                f"entity reappears {match.get('reappearance_count', 0)} time(s); "
                f"largest gap={match.get('max_gap_frames', 0)} frame indices"
            )
        labels_seen = list(match.get("labels_seen", []))
        if labels_seen:
            lines.append("labels seen: " + ", ".join(labels_seen[:4]))
        return lines

    if query_type == "scene_summary":
        match = matches[0]
        summary = match.get("scene_delta_summary", {})
        breakdown = _format_transition_breakdown(
            summary.get("transition_event_type_breakdown", {})
        )
        if breakdown:
            lines.append(
                f"scene {match['scene_id']} change profile: {breakdown}"
            )
        active_delta_entity_ids = list(summary.get("active_delta_entity_ids", []))
        if active_delta_entity_ids:
            preview = ", ".join(active_delta_entity_ids[:6])
            if len(active_delta_entity_ids) > 6:
                preview = f"{preview}, ..."
            lines.append(
                f"scene active delta entities: {preview}"
            )
        static_labels = list(summary.get("representative_static_labels", []))
        if static_labels:
            lines.append(
                "scene anchor labels: " + ", ".join(static_labels[:6])
            )
        dynamic_labels = list(summary.get("representative_dynamic_labels", []))
        if dynamic_labels:
            lines.append(
                "scene dynamic labels: " + ", ".join(dynamic_labels[:6])
            )
        key_transition_frames = list(summary.get("key_transition_frames", []))
        if key_transition_frames:
            lines.append(
                "scene key transition frames: "
                + ", ".join(str(frame) for frame in key_transition_frames)
            )
        boundary_line = _describe_scene_boundary_line(summary)
        if boundary_line:
            lines.append(boundary_line)
        return lines

    return lines


def _format_transition_breakdown(breakdown: dict[str, Any]) -> str:
    parts = [
        f"{event_type}:{count}"
        for event_type, count in sorted(breakdown.items())
        if isinstance(event_type, str) and isinstance(count, int) and count > 0
    ]
    return ", ".join(parts)


def _describe_transition_event(event: dict[str, Any]) -> str:
    frame_index = event.get("frame_index")
    event_type = str(event.get("event_type", "UNKNOWN")).strip()
    prefix = f"frame={frame_index} {event_type}"
    if event_type in {"APPEAR", "DISAPPEAR"}:
        entity_id = event.get("entity_id")
        return f"{prefix} entity={entity_id}"
    if event_type == "STATE_TRANSITION":
        return (
            f"{prefix} entity={event.get('entity_id')} "
            f"{event.get('previous_state_value')}->{event.get('current_state_value')}"
        )
    if event_type in {"RELATION_START", "RELATION_END"}:
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
        "APPEAR": 3,
        "DISAPPEAR": 4,
    }
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event.get("event_type"),
            event.get("entity_id"),
            event.get("subject_entity_id"),
            event.get("predicate"),
            event.get("object_entity_id"),
            event.get("previous_state_value"),
            event.get("current_state_value"),
        )

    ordered = sorted(
        events,
        key=lambda item: (
            priority.get(str(item.get("event_type", "")).strip(), 99),
            item.get("frame_index", 0),
            str(item.get("entity_id") or item.get("subject_entity_id") or ""),
        ),
    )
    covered_types: set[str] = set()
    for event in ordered:
        event_type = str(event.get("event_type", "")).strip()
        key = event_key(event)
        if not event_type or key in seen_keys or event_type in covered_types:
            continue
        selected.append(event)
        seen_keys.add(key)
        covered_types.add(event_type)
        if len(selected) >= limit:
            return selected

    for event in ordered:
        key = event_key(event)
        if key in seen_keys:
            continue
        selected.append(event)
        seen_keys.add(key)
        if len(selected) >= limit:
            return selected
    return selected


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


def _is_instant_episode(match: dict[str, Any]) -> bool:
    support_frames = match.get("support_frames")
    start_frame = match.get("start_frame")
    end_frame = match.get("end_frame")
    return support_frames == 1 or (
        start_frame is not None
        and end_frame is not None
        and start_frame == end_frame
    )


def _describe_state_history_line(match: dict[str, Any]) -> str:
    if _is_instant_episode(match):
        return (
            f"{match['entity_id']} state='{match['state_value']}' frame "
            f"{match['start_frame']} support={match['support_frames']} instantaneous"
        )
    return (
        f"{match['entity_id']} state='{match['state_value']}' frames "
        f"{match['start_frame']}->{match['end_frame']} "
        f"support={match['support_frames']}"
    )


def _describe_relation_history_line(match: dict[str, Any]) -> str:
    raw_predicate = str(match.get("raw_predicate") or "").strip()
    if _is_instant_episode(match):
        line = (
            f"{match['subject']['entity_id']} {match['predicate']} "
            f"{match['object']['entity_id']} frame "
            f"{match['start_frame']} support={match['support_frames']} instantaneous"
        )
    else:
        line = (
            f"{match['subject']['entity_id']} {match['predicate']} "
            f"{match['object']['entity_id']} frames "
            f"{match['start_frame']}->{match['end_frame']} "
            f"support={match['support_frames']}"
        )
    if raw_predicate and raw_predicate != match["predicate"]:
        line = f"{line} raw='{raw_predicate}'"
    return line


def _build_uncertainty_notes(
    *,
    query_type: str,
    matches: list[dict[str, Any]],
    retrieval_bundle: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    if not matches:
        notes.append("The graph did not return any matching evidence.")
        return notes

    if retrieval_bundle.get("match_count", 0) > len(retrieval_bundle.get("evidence_digest", [])):
        notes.append("Only the leading subset of evidence was condensed into the digest.")
    if not retrieval_bundle.get("scene_summaries"):
        notes.append("Scene-level summaries are unavailable or not present in this graph document.")
    if query_type == "relation_history" and len(matches) > 1:
        notes.append("Multiple relation episodes matched; downstream LLM should avoid assuming uniqueness.")
    if query_type == "relation_history" and any(_is_instant_episode(match) for match in matches):
        notes.append("Some relation matches are instantaneous single-frame observations rather than stable intervals.")
    if query_type == "state_history" and len(matches) > 1:
        notes.append("Multiple state intervals matched; downstream LLM should preserve time ranges.")
    if query_type == "state_history" and any(_is_instant_episode(match) for match in matches):
        notes.append("Some state matches are instantaneous single-frame observations rather than persistent states.")
    if query_type == "entity_timeline" and len(matches) > 1:
        notes.append("Multiple entities matched; downstream LLM should mention ambiguity if needed.")
    if query_type == "entity_timeline" and any(
        len(match.get("evidence", {}).get("entity_segments", [])) > 1
        for match in matches
    ):
        notes.append("At least one matched entity appears in multiple disjoint presence segments.")
    if query_type == "scene_summary" and len(matches) > 1:
        notes.append("Multiple scenes matched; downstream LLM should avoid assuming a single scene.")
    if query_type == "frame_snapshot" and not matches[0].get("evidence", {}).get("active_delta_nodes"):
        notes.append("This frame has no active delta node after filtering; it may be dominated by scene-base context.")
    if query_type == "scene_summary" and not matches[0].get("scene_delta_summary", {}).get("key_transition_frames"):
        notes.append("This scene summary does not expose key transition frames yet; answer should lean on labels and counts.")
    return notes


def _format_similarity(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return None


def _describe_scene_boundary_brief(summary: dict[str, Any]) -> str:
    if not summary.get("previous_scene_id") and not summary.get("next_scene_id"):
        return ""
    parts: list[str] = []
    if summary.get("previous_scene_id"):
        reason = summary.get("boundary_from_previous_reason") or "scene_shift"
        sim = _format_similarity(summary.get("boundary_from_previous_label_similarity"))
        if sim is not None:
            parts.append(f"it follows a {reason} boundary from label similarity {sim}")
        else:
            parts.append(f"it follows a {reason} boundary")
    if summary.get("next_scene_id"):
        reason = summary.get("boundary_to_next_reason") or "scene_shift"
        sim = _format_similarity(summary.get("boundary_to_next_label_similarity"))
        if sim is not None:
            parts.append(f"it precedes a {reason} boundary toward label similarity {sim}")
        else:
            parts.append(f"it precedes a {reason} boundary")
    return " ".join(parts)


def _describe_scene_boundary_line(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    if summary.get("previous_scene_id"):
        part = (
            f"from_previous={summary.get('previous_scene_id')} "
            f"reason={summary.get('boundary_from_previous_reason')} "
            f"frame_gap={summary.get('boundary_from_previous_frames')} "
            f"entity_sim={_format_similarity(summary.get('boundary_from_previous_entity_similarity'))} "
            f"label_sim={_format_similarity(summary.get('boundary_from_previous_label_similarity'))}"
        )
        entry_labels = list(summary.get("entry_static_labels", []))
        if entry_labels:
            part = f"{part} entry_labels={','.join(entry_labels[:4])}"
        parts.append(part)
    if summary.get("next_scene_id"):
        part = (
            f"to_next={summary.get('next_scene_id')} "
            f"reason={summary.get('boundary_to_next_reason')} "
            f"frame_gap={summary.get('boundary_to_next_frames')} "
            f"entity_sim={_format_similarity(summary.get('boundary_to_next_entity_similarity'))} "
            f"label_sim={_format_similarity(summary.get('boundary_to_next_label_similarity'))}"
        )
        exit_labels = list(summary.get("exit_static_labels", []))
        if exit_labels:
            part = f"{part} exit_labels={','.join(exit_labels[:4])}"
        parts.append(part)
    return " | ".join(parts)


def _describe_scene_boundary_compact(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    if summary.get("previous_scene_id"):
        label_sim = _format_similarity(summary.get("boundary_from_previous_label_similarity"))
        parts.append(
            "prev="
            + str(summary.get("boundary_from_previous_reason") or "scene_shift")
            + (f":{label_sim}" if label_sim is not None else "")
        )
    if summary.get("next_scene_id"):
        label_sim = _format_similarity(summary.get("boundary_to_next_label_similarity"))
        parts.append(
            "next="
            + str(summary.get("boundary_to_next_reason") or "scene_shift")
            + (f":{label_sim}" if label_sim is not None else "")
        )
    return " ".join(parts)
