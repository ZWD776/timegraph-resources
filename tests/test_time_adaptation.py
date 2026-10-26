from __future__ import annotations

from vlkg.core.temporal.time_adaptation import build_graph_time_adaptation


def _graph_result_with_focus_timestamp(timestamp_sec: float) -> dict:
    return {
        "retrieval_bundle": {
            "focus_frames": [
                {
                    "rank": 1,
                    "frame_index": 42,
                    "timestamp_sec": timestamp_sec,
                    "scene_id": "scene_001",
                    "reasons": ["query_match"],
                    "score": 0.91,
                }
            ],
        },
        "matches": [
            {
                "source_kind": "scene",
                "score": 0.82,
                "metadata": {
                    "scene_id": "scene_001",
                    "time_range": {
                        "start_frame": 40,
                        "end_frame": 45,
                        "start_ts": timestamp_sec,
                        "end_ts": timestamp_sec + 5,
                    },
                },
            }
        ],
    }


def test_time_adaptation_uses_only_graph_timestamps_not_answer_timestamp() -> None:
    adaptation = build_graph_time_adaptation(
        question=(
            "When did the event happen?\n"
            "Question time metadata, not answer evidence:\n"
            "current_time_unix=1673885000.0\n"
            "Timestamp with answer: 1673884999.0"
        ),
        graph_result=_graph_result_with_focus_timestamp(1673884940.0),
    )

    assert adaptation["uses_answer_timestamp"] is False
    assert adaptation["current_time"] == 1673885000.0
    assert adaptation["relative_time_candidates"][0]["timestamp_sec"] == 1673884940.0
    assert (
        adaptation["relative_time_candidates"][0]["relative_to_current"]["display_text"]
        == "about 1.0 minutes ago"
    )


def test_time_adaptation_same_anchor_changes_with_question_current_time() -> None:
    graph_result = _graph_result_with_focus_timestamp(1673884940.0)

    earlier_question = build_graph_time_adaptation(
        question="current_time_unix=1673885000.0",
        graph_result=graph_result,
    )
    later_question = build_graph_time_adaptation(
        question="current_time_unix=1673885240.0",
        graph_result=graph_result,
    )

    earlier_relative = earlier_question["relative_time_candidates"][0]["relative_to_current"]
    later_relative = later_question["relative_time_candidates"][0]["relative_to_current"]
    assert earlier_relative["delta_seconds"] == 60.0
    assert later_relative["delta_seconds"] == 300.0
    assert earlier_relative["display_text"] == "about 1.0 minutes ago"
    assert later_relative["display_text"] == "about 5.0 minutes ago"


def test_time_adaptation_does_not_mix_unix_question_time_with_video_seconds() -> None:
    adaptation = build_graph_time_adaptation(
        question="current_time_unix=1000.0",
        graph_result=_graph_result_with_focus_timestamp(12.0),
    )

    candidate = adaptation["relative_time_candidates"][0]
    assert adaptation["current_time_domain"] == "unix_seconds"
    assert candidate["time_domain"] == "video_seconds"
    assert candidate["relative_to_current"] is None
    assert adaptation["has_relative_time_candidates"] is False


def test_time_adaptation_can_normalize_video_seconds_when_question_uses_video_seconds() -> None:
    adaptation = build_graph_time_adaptation(
        question="current_time=72.0",
        graph_result=_graph_result_with_focus_timestamp(12.0),
    )

    candidate = adaptation["relative_time_candidates"][0]
    assert adaptation["current_time_domain"] == "video_seconds"
    assert candidate["time_domain"] == "video_seconds"
    assert candidate["relative_to_current"]["delta_seconds"] == 60.0
    assert candidate["relative_to_current"]["display_text"] == "about 1.0 minutes ago"
