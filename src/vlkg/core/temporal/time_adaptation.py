from __future__ import annotations

import re
from typing import Any


_QUESTION_TIME_RE = re.compile(
    r"\b(?P<key>current_time_unix|current_time|question_current_time_unix|question_time_unix)"
    r"\s*=\s*(?P<value>-?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)
_QUESTION_WINDOW_RE = re.compile(
    r"\b(?P<key>question_window_start_unix|question_start_time_unix|start_time_unix)"
    r"\s*=\s*(?P<value>-?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)


def build_graph_time_adaptation(
    *,
    question: str,
    graph_result: dict[str, Any],
    max_candidates: int = 16,
) -> dict[str, Any]:
    """Build relative-time candidates from graph evidence and question time metadata.

    This deliberately does not choose the answer event. It only normalizes already
    retrieved graph timestamps against the current question time when both appear to
    be on the same time scale.
    """

    current_marker = _extract_first_time_marker(_QUESTION_TIME_RE, question)
    window_start_marker = _extract_first_time_marker(_QUESTION_WINDOW_RE, question)
    current_time = current_marker["value"] if current_marker else None
    current_time_domain = (
        current_marker["domain"] if current_marker else _time_domain(current_time)
    )
    window_start = window_start_marker["value"] if window_start_marker else None
    raw_candidates = _collect_time_anchor_candidates(graph_result)
    candidates = _dedupe_time_candidates(raw_candidates)

    comparable_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        normalized = dict(candidate)
        timestamp = _optional_float(normalized.get("timestamp_sec"))
        normalized["anchor_id"] = f"time_anchor:{index:02d}"
        normalized["time_domain"] = _time_domain(timestamp)
        normalized["relative_to_current"] = None
        if (
            current_time is not None
            and timestamp is not None
            and current_time_domain == normalized["time_domain"]
        ):
            relative = _relative_time_payload(
                current_time=current_time,
                timestamp=timestamp,
            )
            normalized["relative_to_current"] = relative
        comparable_candidates.append(normalized)
        if len(comparable_candidates) >= max_candidates:
            break

    has_relative_candidates = any(
        item.get("relative_to_current")
        for item in comparable_candidates
    )
    return {
        "contract": "graph_time_adaptation_v1",
        "uses_answer_timestamp": False,
        "current_time": current_time,
        "question_window_start": window_start,
        "current_time_domain": current_time_domain,
        "relative_time_candidates": comparable_candidates,
        "has_relative_time_candidates": has_relative_candidates,
        "notes": _time_adaptation_notes(
            current_time=current_time,
            candidates=comparable_candidates,
            has_relative_candidates=has_relative_candidates,
        ),
    }


def _collect_time_anchor_candidates(graph_result: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_bundle = graph_result.get("retrieval_bundle", {})
    candidates: list[dict[str, Any]] = []

    for item in retrieval_bundle.get("focus_frames", []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "source": "focus_frame",
                "rank": item.get("rank"),
                "timestamp_sec": _optional_float(item.get("timestamp_sec")),
                "frame_index": _optional_int(item.get("frame_index")),
                "scene_id": item.get("scene_id"),
                "reasons": list(item.get("reasons", [])),
                "score": item.get("score"),
            }
        )

    for chain_rank, chain in enumerate(retrieval_bundle.get("temporal_chains", []), start=1):
        if not isinstance(chain, dict):
            continue
        chain_time = dict(chain.get("time_range", {}))
        _append_range_candidates(
            candidates,
            source="temporal_chain",
            rank=chain_rank,
            time_range=chain_time,
            scene_id=None,
            reasons=["temporal_chain_boundary"],
        )
        for scene_rank, scene in enumerate(chain.get("scene_sequence", []), start=1):
            if not isinstance(scene, dict):
                continue
            _append_range_candidates(
                candidates,
                source="temporal_chain_scene",
                rank=chain_rank,
                time_range={
                    "start_frame": scene.get("start_frame"),
                    "end_frame": scene.get("end_frame"),
                    "start_ts": scene.get("start_ts"),
                    "end_ts": scene.get("end_ts"),
                },
                scene_id=scene.get("scene_id"),
                reasons=[f"temporal_chain_scene:{scene_rank}"],
            )

    for scene_rank, scene in enumerate(retrieval_bundle.get("scene_summaries", []), start=1):
        if not isinstance(scene, dict):
            continue
        _append_range_candidates(
            candidates,
            source="scene_summary",
            rank=scene_rank,
            time_range={
                "start_frame": scene.get("start_frame"),
                "end_frame": scene.get("end_frame"),
                "start_ts": scene.get("start_ts"),
                "end_ts": scene.get("end_ts"),
            },
            scene_id=scene.get("scene_id"),
            reasons=["scene_summary_boundary"],
        )

    for match_rank, match in enumerate(graph_result.get("matches", [])[:12], start=1):
        if not isinstance(match, dict):
            continue
        metadata = dict(match.get("metadata", {}))
        _append_range_candidates(
            candidates,
            source=f"match:{match.get('source_kind')}",
            rank=match_rank,
            time_range=dict(metadata.get("time_range", {})),
            scene_id=metadata.get("scene_id"),
            reasons=["match_time_range"],
            score=match.get("score"),
        )

    return [
        item
        for item in candidates
        if _optional_float(item.get("timestamp_sec")) is not None
    ]


def _append_range_candidates(
    candidates: list[dict[str, Any]],
    *,
    source: str,
    rank: int | None,
    time_range: dict[str, Any],
    scene_id: Any,
    reasons: list[str],
    score: Any = None,
) -> None:
    start_ts = _optional_float(time_range.get("start_ts"))
    end_ts = _optional_float(time_range.get("end_ts"))
    start_frame = _optional_int(time_range.get("start_frame"))
    end_frame = _optional_int(time_range.get("end_frame"))
    if start_ts is not None:
        candidates.append(
            {
                "source": f"{source}:start",
                "rank": rank,
                "timestamp_sec": start_ts,
                "frame_index": start_frame,
                "scene_id": scene_id,
                "reasons": reasons,
                "score": score,
            }
        )
    if end_ts is not None and end_ts != start_ts:
        candidates.append(
            {
                "source": f"{source}:end",
                "rank": rank,
                "timestamp_sec": end_ts,
                "frame_index": end_frame,
                "scene_id": scene_id,
                "reasons": reasons,
                "score": score,
            }
        )


def _dedupe_time_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, int | None, str, str]] = set()
    for item in candidates:
        timestamp = _optional_float(item.get("timestamp_sec"))
        if timestamp is None:
            continue
        frame_index = _optional_int(item.get("frame_index"))
        source_family = str(item.get("source") or "").split(":", 1)[0]
        scene_id = str(item.get("scene_id") or "")
        key = (round(timestamp, 6), frame_index, source_family, scene_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _relative_time_payload(
    *,
    current_time: float,
    timestamp: float | None,
) -> dict[str, Any] | None:
    if timestamp is None:
        return None
    delta_seconds = current_time - timestamp
    delta_minutes = delta_seconds / 60.0
    direction = "ago" if delta_seconds >= 0 else "after_current_time"
    return {
        "delta_seconds": round(delta_seconds, 3),
        "delta_minutes": round(delta_minutes, 4),
        "absolute_delta_minutes": round(abs(delta_minutes), 4),
        "direction": direction,
        "display_text": _relative_time_text(delta_seconds),
    }


def _relative_time_text(delta_seconds: float) -> str:
    abs_seconds = abs(delta_seconds)
    suffix = "ago" if delta_seconds >= 0 else "after current time"
    if abs_seconds < 60:
        return f"about {abs_seconds:.0f} seconds {suffix}"
    minutes = abs_seconds / 60.0
    if minutes < 10:
        return f"about {minutes:.1f} minutes {suffix}"
    return f"about {minutes:.0f} minutes {suffix}"


def _time_adaptation_notes(
    *,
    current_time: float | None,
    candidates: list[dict[str, Any]],
    has_relative_candidates: bool,
) -> list[str]:
    notes: list[str] = []
    if current_time is None:
        notes.append("No current question time was provided, so relative time should not be computed.")
    elif not has_relative_candidates:
        notes.append(
            "No graph timestamp shared the current-time domain, so keep graph frame/time anchors instead of producing minutes-ago values."
        )
    if candidates:
        notes.append(
            "Relative-time candidates are computed only from retrieved graph evidence and question current time; they do not select the answer event."
        )
    return notes


def _extract_first_time_marker(
    pattern: re.Pattern[str],
    text: str,
) -> dict[str, Any] | None:
    match = pattern.search(str(text or ""))
    if not match:
        return None
    value = _optional_float(match.group("value"))
    if value is None:
        return None
    key = str(match.group("key") or "")
    return {
        "key": key,
        "value": value,
        "domain": _time_domain_from_key_or_value(key, value),
    }


def _time_domain_from_key_or_value(key: str, value: float | None) -> str:
    if "unix" in str(key or "").lower():
        return "unix_seconds"
    return _time_domain(value)


def _time_domain(value: float | None) -> str:
    if value is None:
        return "unknown"
    if abs(value) >= 1_000_000_000:
        return "unix_seconds"
    return "video_seconds"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
