from __future__ import annotations

from vlkg.datasets.navqa.qa.official_qa import (
    load_position_labels,
    materialize_official_answer_label,
)
from vlkg.datasets.navqa.qa.position_reference import (
    FIELD_CATEGORY,
    FIELD_PARSABLE_ANSWER,
    FIELD_QUESTION,
    FIELD_SEQ_ID,
    FIELD_TEXT_ANSWER,
    FIELD_TIMESTAMP,
    FIELD_TYPE,
    FIELD_UUID,
)


def _row(**updates: str) -> dict[str, str]:
    row = {
        FIELD_SEQ_ID: "0",
        FIELD_UUID: "LONG_example",
        FIELD_QUESTION: "When did the event happen?",
        FIELD_TIMESTAMP: "7:57:53",
        FIELD_TYPE: "time",
        FIELD_TEXT_ANSWER: "",
        FIELD_PARSABLE_ANSWER: "[minutes] ago",
        FIELD_CATEGORY: "temporal",
    }
    row.update(updates)
    return row


def test_time_label_uses_question_end_time_and_timezone_safe_hms_resolution() -> None:
    label = materialize_official_answer_label(
        _row(),
        question_metadata={
            "LONG_example": {
                "id": "LONG_example",
                "start_time": 1673884349.0,
                "end_time": 1673884853.0,
            }
        },
    )

    assert label.evaluation_key == "time"
    assert label.value == 3.0
    assert label.display == "3.0 minutes ago"
    assert label.semantics == "minutes_before_question_end_time"
    assert label.inject_into_graph is False


def test_position_label_uses_external_reference_without_graph_injection() -> None:
    reference = {
        "position_answers": [
            {
                "uuid": "LONG_example",
                "position": [-104.492685, 169.162389, -2.406359],
            }
        ]
    }

    label = materialize_official_answer_label(
        _row(
            **{
                FIELD_TYPE: "position",
                FIELD_PARSABLE_ANSWER: "[position]",
                FIELD_CATEGORY: "spatial",
            }
        ),
        position_labels=load_position_labels(reference),
    )

    assert label.evaluation_key == "position"
    assert label.value == [-104.492685, 169.162389, -2.406359]
    assert label.display == "[-104.492685, 169.162389, -2.406359]"
    assert label.semantics == "ego_pose_at_semantic_event_time"
    assert label.inject_into_graph is False


def test_duration_binary_and_text_labels_keep_official_output_modes_separate() -> None:
    duration = materialize_official_answer_label(
        _row(
            **{
                FIELD_TYPE: "duration",
                FIELD_PARSABLE_ANSWER: "2.3",
                FIELD_CATEGORY: "temporal reasoning",
            }
        )
    )
    binary = materialize_official_answer_label(
        _row(
            **{
                FIELD_TYPE: "binary",
                FIELD_PARSABLE_ANSWER: "yes",
                FIELD_CATEGORY: "environmental",
            }
        )
    )
    text = materialize_official_answer_label(
        _row(
            **{
                FIELD_TYPE: "text",
                FIELD_TEXT_ANSWER: "They were wearing a blue puffer jacket",
                FIELD_PARSABLE_ANSWER: "",
                FIELD_CATEGORY: "descriptive",
            }
        )
    )

    assert duration.evaluation_key == "duration"
    assert duration.value == 2.3
    assert binary.evaluation_key == "binary"
    assert binary.value == "yes"
    assert text.evaluation_key == "text"
    assert text.value == "They were wearing a blue puffer jacket"
