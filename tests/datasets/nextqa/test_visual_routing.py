from __future__ import annotations

from copy import deepcopy

import pytest

from vlkg.datasets.nextqa.runtime.visual_runtime_v1.routing import (
    ROUTE_CAUSAL_HOW,
    ROUTE_CAUSAL_WHY,
    ROUTE_DESCRIPTIVE_COUNT,
    ROUTE_DESCRIPTIVE_LOCATION,
    ROUTE_DESCRIPTIVE_OBJECT_ACTION,
    ROUTE_TEMPORAL_CONCURRENT,
    ROUTE_TEMPORAL_NEXT,
    ROUTE_TEMPORAL_PREVIOUS,
    audit_route_field_isolation,
    compile_route_record,
)


def _ir(question: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "question_id": "nextqa-val/100/q1",
        "video_id": "100",
        "normalized_question": question,
        "target_entities": ["person"],
        "actors": ["person"],
        "actions": [],
        "objects": [],
        "temporal_direction": "none",
        "temporal_anchor": "none",
        "referenced_event": "",
        "causal_query": "none",
        "count_query": False,
        "location_query": False,
        "attribute_query": "none",
        "answer_format": "multiple_choice_5",
        "official_type": "IGNORED",
        "answer_choices": ["a", "b", "c", "d", "e"],
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_ir("why did the person leave", causal_query="why"), ROUTE_CAUSAL_WHY),
        (_ir("how did the person open it", causal_query="how"), ROUTE_CAUSAL_HOW),
        (_ir("what happened before the person left", temporal_direction="before"), ROUTE_TEMPORAL_PREVIOUS),
        (_ir("what happened after the person left", temporal_direction="after"), ROUTE_TEMPORAL_NEXT),
        (_ir("what happened while the person waited", temporal_direction="concurrent"), ROUTE_TEMPORAL_CONCURRENT),
        (_ir("how many people were present", count_query=True), ROUTE_DESCRIPTIVE_COUNT),
        (_ir("where was the person", location_query=True), ROUTE_DESCRIPTIVE_LOCATION),
        (_ir("what did the person hold"), ROUTE_DESCRIPTIVE_OBJECT_ACTION),
    ],
)
def test_question_only_routes(row: dict[str, object], expected: str) -> None:
    route = compile_route_record(row)
    assert route["route_id"] == expected
    assert route["changes_graph_retrieval"] is False


def test_route_is_invariant_to_type_choices_and_gold() -> None:
    row = _ir("what happened after the person left", temporal_direction="after")
    checks = audit_route_field_isolation(row)
    assert all(checks.values())

    route = compile_route_record(row)
    assert "official_type" not in route
    assert "answer_choices" not in route
    assert "gold_choice_index" not in route
    assert route["uses_official_type"] is False
    assert route["uses_answer_choices"] is False


def test_choice_perturbation_cannot_change_route() -> None:
    row = _ir("why did the person leave", causal_query="why")
    changed = deepcopy(row)
    changed["answer_choices"] = ["x0", "x1", "x2", "x3", "x4"]
    assert compile_route_record(row) == compile_route_record(changed)


def test_spatial_next_to_is_not_a_temporal_next_cue() -> None:
    row = _ir("what is the boy next to the woman doing")
    assert compile_route_record(row)["route_id"] == ROUTE_DESCRIPTIVE_OBJECT_ACTION


def test_reaction_when_event_uses_concurrent_evidence() -> None:
    row = _ir(
        "how did the dog react when the person arrived",
        causal_query="how",
        temporal_direction="concurrent",
    )
    assert compile_route_record(row)["route_id"] == ROUTE_TEMPORAL_CONCURRENT
