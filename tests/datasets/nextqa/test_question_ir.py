from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.datasets.nextqa.retrieval.question_ir import (
    audit_nonsemantic_field_isolation,
    build_question_ir_record,
    retrieval_input_from_ir,
    retrieval_input_fingerprint,
)


def _row(question: str = "What did the boy do after he stopped the bike?") -> dict:
    return {
        "question_id": "nextqa-val/123/q7",
        "video_id": "123",
        "official_qid": "7",
        "official_type": "TN",
        "normalized_question": question,
        "answer_choices": ["one", "two", "three", "four", "five"],
        "answer_format": "multiple_choice_5",
    }


def test_question_ir_keeps_diagnostics_but_retrieval_projection_drops_them() -> None:
    record = build_question_ir_record(_row())
    retrieval = retrieval_input_from_ir(record).to_dict()

    assert record["official_type"] == "TN"
    assert len(record["answer_choices"]) == 5
    assert "official_type" not in retrieval
    assert "answer_choices" not in retrieval
    assert retrieval["temporal_direction"] == "after"
    assert retrieval["actors"] == ["boy"]
    assert "stop" in retrieval["actions"]
    assert "bike" in retrieval["objects"]


def test_nonsemantic_fields_cannot_change_retrieval_fingerprint() -> None:
    record = build_question_ir_record(_row())
    baseline = retrieval_input_fingerprint(record)
    perturbed = deepcopy(record)
    perturbed["official_type"] = "CW"
    perturbed["answer_choices"] = ["x", "y", "z", "u", "v"]

    assert retrieval_input_fingerprint(perturbed) == baseline
    assert audit_nonsemantic_field_isolation(record) == {
        "official_type_invariant": True,
        "answer_choices_invariant": True,
    }


def test_question_ir_recognizes_count_location_and_causal_without_type_code() -> None:
    count = build_question_ir_record(_row("How many children are cycling?"))
    location = build_question_ir_record(_row("Where are the people hanging out?"))
    causal = build_question_ir_record(_row("Why did the woman hold the baby?"))

    assert count["count_query"] is True
    assert count["causal_query"] == "none"
    assert location["location_query"] is True
    assert causal["causal_query"] == "why"


def test_question_ir_avoids_action_idioms_and_records_attribute_demand() -> None:
    activity = build_question_ir_record(_row("What activity is being carried out?"))
    location = build_question_ir_record(_row("Where is this video taken?"))
    color = build_question_ir_record(_row("What colour is the toy car?"))

    assert "carry" not in activity["actions"]
    assert "take" not in location["actions"]
    assert color["attribute_query"] == "color"


def test_question_ir_rejects_identity_mismatch_or_non_five_choice_input() -> None:
    bad_id = _row()
    bad_id["video_id"] = "456"
    with pytest.raises(ValueError, match="identity"):
        build_question_ir_record(bad_id)

    bad_choices = _row()
    bad_choices["answer_choices"] = ["one"]
    with pytest.raises(ValueError, match="five"):
        build_question_ir_record(bad_choices)
