from __future__ import annotations

from datetime import datetime

from vlkg.datasets.navqa.qa.position_reference import (
    FIELD_CATEGORY,
    FIELD_PARSABLE_ANSWER,
    FIELD_QUESTION,
    FIELD_SEQ_ID,
    FIELD_TEXT_ANSWER,
    FIELD_TIMESTAMP,
    FIELD_TYPE,
    FIELD_UUID,
    PoseSample,
    build_position_reference,
    build_position_windows_from_pose_samples,
    coda_preprocessed_pose_samples,
    lookup_position_for_timestamp,
)


def test_coda_preprocessed_pose_samples_matches_one_pose_per_new_lidar_frame() -> None:
    poses = [
        PoseSample(timestamp=9.9, position=(0.0, 0.0, 0.0)),
        PoseSample(timestamp=10.0, position=(1.0, 0.0, 0.0)),
        PoseSample(timestamp=10.1, position=(2.0, 0.0, 0.0)),
        PoseSample(timestamp=10.9, position=(3.0, 0.0, 0.0)),
        PoseSample(timestamp=11.0, position=(4.0, 0.0, 0.0)),
    ]
    lidar_timestamps = [10.0, 11.0, 12.0]

    samples = coda_preprocessed_pose_samples(poses, lidar_timestamps)

    assert [item.timestamp for item in samples] == [10.1]
    assert [item.lidar_frame for item in samples] == [1]


def test_position_reference_materializes_label_without_graph_injection() -> None:
    base_time = 1673884185.0
    samples = [
        PoseSample(timestamp=base_time + 0.0, position=(0.0, 0.0, 0.0)),
        PoseSample(timestamp=base_time + 1.0, position=(2.0, 0.0, 0.0)),
        PoseSample(timestamp=base_time + 2.0, position=(4.0, 0.0, 0.0)),
        PoseSample(timestamp=base_time + 4.0, position=(10.0, 0.0, 0.0)),
        PoseSample(timestamp=base_time + 5.0, position=(12.0, 0.0, 0.0)),
    ]
    windows = build_position_windows_from_pose_samples(samples, seconds_per_caption=3.0)
    answer_hms = datetime.fromtimestamp(base_time + 1.0).strftime("%H:%M:%S")
    row = {
        FIELD_SEQ_ID: "0",
        FIELD_UUID: "LONG_example",
        FIELD_QUESTION: "Where is the closest place to sit?",
        FIELD_TIMESTAMP: answer_hms,
        FIELD_TYPE: "position",
        FIELD_TEXT_ANSWER: "",
        FIELD_PARSABLE_ANSWER: "[position]",
        FIELD_CATEGORY: "spatial",
    }
    metadata = {
        "LONG_example": {
            "id": "LONG_example",
            "start_time": base_time,
            "end_time": base_time + 20.0,
        }
    }

    reference = build_position_reference(
        navqa_rows=[row],
        question_metadata=metadata,
        position_windows=windows,
        seq_id="0",
    )

    assert reference["uses_graph"] is False
    assert reference["inject_into_graph"] is False
    assert reference["uses_entity_3d_coordinate"] is False
    assert reference["uses_official_position_coordinate"] is False
    assert reference["uses_official_answer_timestamp"] is True
    assert reference["position_semantics"] == "ego_pose_at_semantic_event_time"
    assert reference["position_answer_count"] == 1
    assert reference["position_answers"][0]["position"] == [2.0, 0.0, 0.0]
    assert reference["position_answers"][0]["position_semantics"] == (
        "ego_pose_at_semantic_event_time"
    )
    assert reference["position_answers"][0]["event_time_source"] == (
        "official_answer_timestamp_for_label_materialization_only"
    )
    assert reference["position_answers"][0]["generation_policy"] == (
        "official_answer_timestamp_to_ego_pose_window"
    )


def test_position_reference_skips_non_single_timestamp_position_rows() -> None:
    row = {
        FIELD_SEQ_ID: "0",
        FIELD_UUID: "LONG_duration_like",
        FIELD_QUESTION: "Where was it?",
        FIELD_TIMESTAMP: "00:01:41,00:01:42",
        FIELD_TYPE: "position",
        FIELD_TEXT_ANSWER: "",
        FIELD_PARSABLE_ANSWER: "[position]",
        FIELD_CATEGORY: "spatial",
    }
    metadata = {
        "LONG_duration_like": {
            "id": "LONG_duration_like",
            "start_time": 100.0,
            "end_time": 120.0,
        }
    }
    windows = build_position_windows_from_pose_samples(
        [PoseSample(timestamp=100.0, position=(0.0, 0.0, 0.0))],
        seconds_per_caption=3.0,
        include_final_segment=True,
    )

    reference = build_position_reference(
        navqa_rows=[row],
        question_metadata=metadata,
        position_windows=windows,
        seq_id="0",
    )

    assert reference["position_answer_count"] == 0
    assert reference["skipped"][0]["reason"] == "position_answer_requires_one_timestamp"


def test_position_reference_hms_parsing_is_independent_of_local_timezone() -> None:
    # The displayed HMS can be many hours away from UTC Unix time. The reference
    # builder should choose the occurrence nearest the question window instead
    # of trusting the machine's local timezone.
    samples = [
        PoseSample(timestamp=1673884650.0, position=(1.0, 0.0, 0.0)),
        PoseSample(timestamp=1673884674.0, position=(2.0, 0.0, 0.0)),
        PoseSample(timestamp=1673884675.0, position=(4.0, 0.0, 0.0)),
        PoseSample(timestamp=1673884690.0, position=(8.0, 0.0, 0.0)),
    ]
    windows = build_position_windows_from_pose_samples(samples, seconds_per_caption=3.0)
    row = {
        FIELD_SEQ_ID: "0",
        FIELD_UUID: "LONG_timezone",
        FIELD_QUESTION: "Where is the object?",
        FIELD_TIMESTAMP: "7:57:53",
        FIELD_TYPE: "position",
        FIELD_TEXT_ANSWER: "",
        FIELD_PARSABLE_ANSWER: "[position]",
        FIELD_CATEGORY: "spatial",
    }
    metadata = {
        "LONG_timezone": {
            "id": "LONG_timezone",
            "start_time": 1673884349.0,
            "end_time": 1673884844.0,
        }
    }

    reference = build_position_reference(
        navqa_rows=[row],
        question_metadata=metadata,
        position_windows=windows,
        seq_id="0",
    )

    answer = reference["position_answers"][0]
    assert answer["answer_timestamp_unix"] == 1673884673.0
    assert answer["position"] == [1.0, 0.0, 0.0]


def test_pose_lookup_uses_event_timestamp_not_question_uuid() -> None:
    windows = build_position_windows_from_pose_samples(
        [
            PoseSample(timestamp=100.0, position=(1.0, 0.0, 0.0)),
            PoseSample(timestamp=101.0, position=(3.0, 0.0, 0.0)),
            PoseSample(timestamp=110.0, position=(9.0, 0.0, 0.0)),
            PoseSample(timestamp=111.0, position=(11.0, 0.0, 0.0)),
        ],
        seconds_per_caption=3.0,
        include_final_segment=True,
    )

    result = lookup_position_for_timestamp(windows, 110.2)

    assert result is not None
    assert result.position == (10.0, 0.0, 0.0)
    assert result.window_start_time == 110.0
