from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import uuid


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / 'scripts' / 'datasets' / 'navqa' / 'qa' / 'run_navqa_official_qa_on_canonical_graph.py'
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("navqa_official_qa_eval", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scratch_json_path() -> Path:
    root = Path(".scratch/test-artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"qa_unfilled_{uuid.uuid4().hex}.json"


def test_official_qa_parser_exposes_question_router_hybrid_backbone() -> None:
    module = _load_eval_module()

    args = module.build_parser().parse_args(
        [
            "graph.json",
            "--csv",
            "qa.csv",
            "--experimental-question-router",
            "--experimental-question-router-hybrid-backbone",
        ]
    )

    assert args.experimental_question_router is True
    assert args.experimental_question_router_hybrid_backbone is True


def test_official_qa_parser_exposes_question_router_specialized_backbone() -> None:
    module = _load_eval_module()

    args = module.build_parser().parse_args(
        [
            "graph.json",
            "--csv",
            "qa.csv",
            "--experimental-question-router",
            "--experimental-question-router-specialized-backbone",
        ]
    )

    assert args.experimental_question_router is True
    assert args.experimental_question_router_specialized_backbone is True


def test_official_qa_parser_exposes_question_router_mainline() -> None:
    module = _load_eval_module()

    args = module.build_parser().parse_args(
        [
            "graph.json",
            "--csv",
            "qa.csv",
            "--experimental-question-router",
            "--experimental-question-router-mainline",
        ]
    )

    assert args.experimental_question_router is True
    assert args.experimental_question_router_mainline is True


def test_question_time_metadata_preserves_uuid_ids() -> None:
    module = _load_eval_module()
    qa_path = _scratch_json_path()
    try:
        qa_path.write_text(
            json.dumps(
                {
                    "data": [
                        {
                            "id": "LONG_abc",
                            "start_time": 100.0,
                            "end_time": 160.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        metadata = module.load_question_time_metadata(qa_path)
        context = module.question_time_context(
            {"UUID": "LONG_abc"},
            metadata,
        )
    finally:
        qa_path.unlink(missing_ok=True)

    assert "LONG_abc" in metadata
    assert "QLONG_abc" not in metadata
    assert "current_time_unix=160.0" in context
    assert "question_window_start_unix=100.0" in context


def test_question_time_metadata_still_accepts_numeric_question_ids() -> None:
    module = _load_eval_module()
    qa_path = _scratch_json_path()
    try:
        qa_path.write_text(
            json.dumps({"data": [{"id": "17", "start_time": 10.0, "end_time": 20.0}]}),
            encoding="utf-8",
        )

        metadata = module.load_question_time_metadata(qa_path)
    finally:
        qa_path.unlink(missing_ok=True)

    assert metadata["17"]["end_time"] == 20.0
    assert metadata["Q17"]["end_time"] == 20.0


def test_position_question_adaptation_treats_position_as_event_time() -> None:
    module = _load_eval_module()

    context = module.question_adaptation_context(
        {
            module.FIELD_TYPE: "position",
            module.FIELD_PARSABLE_ANSWER: "[position]",
        }
    )
    instruction = module.official_answer_format_instruction(
        {
            module.FIELD_TYPE: "position",
            module.FIELD_PARSABLE_ANSWER: "[position]",
        }
    )

    assert "event-time localization" in context
    assert "robot/ego pose at the semantic event time" in context
    assert "Do not infer object 3D coordinates" in context
    assert "ego-pose renderer" in instruction


def test_binary_answer_format_instruction_requires_exact_yes_no() -> None:
    module = _load_eval_module()

    instruction = module.official_answer_format_instruction(
        {
            module.FIELD_TYPE: "binary",
            module.FIELD_PARSABLE_ANSWER: "yes",
        }
    )

    assert "exactly 'yes' or exactly 'no'" in instruction
    assert "not in answer_text" in instruction


def test_build_memory_query_uses_question_memory_window() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "LONG_window",
            module.FIELD_QUESTION: "Where did you see a red fire hydrant?",
            module.FIELD_TYPE: "position",
            module.FIELD_CATEGORY: "spatial",
        },
        retrieval_limit=12,
        question_time_metadata={
            "LONG_window": {
                "start_time": 1673884200.0,
                "end_time": 1673884500.0,
            }
        },
        graph_bounds=(1673884180.0, 1673885000.0),
    )

    assert query.start_ts == 1673884200.0
    assert query.end_ts == 1673884500.0
    assert query.limit == 12
    assert window["applied"] is True
    assert window["reason"] == "question_memory_window"


def test_build_memory_query_can_use_llm_question_ir_query_text() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "q-ir",
            module.FIELD_QUESTION: "When did you leave the building?",
            module.FIELD_TYPE: "time",
            module.FIELD_CATEGORY: "temporal",
        },
        retrieval_limit=12,
        question_ir={
            "answer_type": "time",
            "operation": "offset",
            "qa_focus": "time_duration",
            "target_entities": ["building"],
            "rendering": "minutes_ago",
            "hard_constraints": [
                {
                    "kind": "event",
                    "text": "leave the building",
                    "role": "event_condition",
                    "target_entities": ["building"],
                    "must_bind_to": ["same_time_window"],
                }
            ],
            "graph_information_needs": ["ego trajectory"],
        },
        question_time_metadata={},
    )

    assert "operation offset" not in query.query_text
    assert "answer_type" not in query.query_text
    assert query.metadata["question_ir"]["operation"] == "offset"
    assert "building" in query.query_text
    assert "hard constraint event leave the building" in query.query_text
    assert query.query_text.startswith("When did you leave the building?")
    assert query.answer_type == "time"
    assert query.limit == 12
    assert window["applied"] is False


def test_build_memory_query_can_attach_question_route_metadata() -> None:
    module = _load_eval_module()

    query, _window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "q-route",
            module.FIELD_QUESTION: "How many doors did the person open?",
            module.FIELD_TYPE: "text",
            module.FIELD_CATEGORY: "count",
        },
        retrieval_limit=8,
        question_ir={"answer_type": "text", "target_entities": ["doors"]},
        question_route={
            "contract": "navqa_question_route_v1",
            "route_id": "count_set",
            "retrieval_policy": "time_condition_filtered_entity_set",
        },
        question_time_metadata={},
    )

    assert query.metadata["question_ir"]["target_entities"] == ["doors"]
    assert query.metadata["question_route"]["route_id"] == "count_set"
    assert "doors" in query.query_text


def test_load_question_ir_records_indexes_llm_question_ir_by_uuid() -> None:
    module = _load_eval_module()
    path = Path(".scratch/test-artifacts") / f"question_ir_{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "uuid": "q1",
                            "llm_question_ir": {
                                "operation": "attribute_query",
                                "target_entities": ["couches"],
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        records = module.load_question_ir_records(path)
    finally:
        path.unlink(missing_ok=True)

    assert records["q1"]["operation"] == "attribute_query"


def test_navqa_official_style_accuracy_summary_exposes_by_type_buckets() -> None:
    module = _load_eval_module()

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "question_type": "text",
                "official_format_prediction": "brown",
                "predicted_answer": "brown",
                "official_evaluation_key": "brown",
                "status": "live_validated",
                "text_llm_judgement": {"correct": True},
            },
            {
                "question_type": "binary",
                "official_format_prediction": "yes",
                "predicted_answer": "yes",
                "official_evaluation_value": "yes",
                "official_evaluation_key": "binary",
                "official_evaluation_label": {"value": "yes", "display": "yes"},
                "status": "live_validated",
            },
            {
                "question_type": "time",
                "official_format_prediction": "1 minute ago",
                "predicted_answer": "1 minute ago",
                "official_evaluation_value": "1 minute ago",
                "status": "live_validated",
            },
        ]
    )

    assert summary["by_type"]["text"]["correct"] == 1
    assert summary["by_type"]["text"]["total"] == 1
    assert summary["by_type"]["binary"]["correct"] == 1
    assert summary["by_type"]["binary"]["total"] == 1
    assert summary["by_type"]["time"]["correct"] == 1


def test_build_memory_query_uses_tail_window_for_current_questions() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "LONG_current",
            module.FIELD_QUESTION: "Are you stopped in a loading zone right now?",
            module.FIELD_TYPE: "binary",
            module.FIELD_CATEGORY: "environmental",
        },
        retrieval_limit=5,
        question_time_metadata={
            "LONG_current": {
                "start_time": 1673884200.0,
                "end_time": 1673884500.0,
            }
        },
        graph_bounds=(1673884180.0, 1673885000.0),
        current_window_sec=20.0,
    )

    assert query.start_ts == 1673884480.0
    assert query.end_ts == 1673884500.0
    assert window["applied"] is True
    assert window["current_window_required"] is True
    assert window["reason"] == "current_question_end_window"


def test_build_memory_query_skips_incompatible_time_scale() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "LONG_scale",
            module.FIELD_QUESTION: "When did you leave the building?",
            module.FIELD_TYPE: "time",
            module.FIELD_CATEGORY: "temporal",
        },
        retrieval_limit=5,
        question_time_metadata={
            "LONG_scale": {
                "start_time": 1673884200.0,
                "end_time": 1673884500.0,
            }
        },
        graph_bounds=(2.0, 1642.0),
    )

    assert query.start_ts is None
    assert query.end_ts is None
    assert window["applied"] is False
    assert window["reason"] == "time_scale_mismatch"


def test_build_memory_query_maps_unix_question_window_to_graph_frames() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "LONG_scale",
            module.FIELD_QUESTION: "When did you first enter the building?",
            module.FIELD_TYPE: "time",
            module.FIELD_CATEGORY: "temporal",
        },
        retrieval_limit=5,
        question_time_metadata={
            "LONG_scale": {
                "start_time": 1_000_000_000.0,
                "end_time": 1_000_000_040.0,
            }
        },
        graph_bounds=(2.0, 1642.0),
        frame_time_index={
            "by_frame": {
                20: {"frame_index": 20, "timestamp_unix": 1_000_000_000.1},
                60: {"frame_index": 60, "timestamp_unix": 1_000_000_039.9},
            },
            "ordered_frames": [20, 60],
            "source_path": "test-index.json",
        },
    )

    assert query.start_ts == 20.0
    assert query.end_ts == 60.0
    assert window["applied"] is True
    assert window["reason"] == "question_memory_window"
    assert window["mapped_from_unix_time"]["source"] == "frame_time_index"


def test_current_question_window_scales_seconds_to_graph_units() -> None:
    module = _load_eval_module()

    query, window = module.build_memory_query_for_question(
        {
            module.FIELD_UUID: "q-current",
            module.FIELD_QUESTION: "Are you stopped in a loading zone right now?",
            module.FIELD_TYPE: "binary",
            module.FIELD_CATEGORY: "environmental",
        },
        retrieval_limit=5,
        question_time_metadata={
            "q-current": {
                "start_time": 1_000_000_000.0,
                "end_time": 1_000_000_040.0,
            }
        },
        graph_bounds=(2.0, 1642.0),
        frame_time_index={
            "by_frame": {
                20: {"frame_index": 20, "timestamp_unix": 1_000_000_000.0},
                100: {"frame_index": 100, "timestamp_unix": 1_000_000_040.0},
            },
            "ordered_frames": [20, 100],
            "source_path": "test-index.json",
        },
        current_window_sec=30.0,
    )

    assert query.start_ts == 40.0
    assert query.end_ts == 100.0
    assert window["reason"] == "current_question_end_window"


def test_filter_questions_by_type_supports_non_position_mainline() -> None:
    module = _load_eval_module()

    rows = [
        {module.FIELD_UUID: "q-binary", module.FIELD_TYPE: "binary"},
        {module.FIELD_UUID: "q-text", module.FIELD_TYPE: "text"},
        {module.FIELD_UUID: "q-time", module.FIELD_TYPE: "time"},
        {module.FIELD_UUID: "q-duration", module.FIELD_TYPE: "duration"},
        {module.FIELD_UUID: "q-position", module.FIELD_TYPE: "position"},
    ]

    selected = module.filter_questions_by_type(
        rows,
        include_types=module.NON_POSITION_QA_TYPES,
    )

    assert [row[module.FIELD_UUID] for row in selected] == [
        "q-binary",
        "q-text",
        "q-time",
        "q-duration",
    ]


def test_filter_questions_by_type_excludes_position_without_deleting_support() -> None:
    module = _load_eval_module()

    rows = [
        {module.FIELD_UUID: "q-position", module.FIELD_TYPE: "position"},
        {module.FIELD_UUID: "q-time", module.FIELD_TYPE: "time"},
    ]

    selected = module.filter_questions_by_type(rows, exclude_types=["position"])

    assert [row[module.FIELD_UUID] for row in selected] == ["q-time"]


def test_non_position_selection_does_not_load_position_artifacts(monkeypatch) -> None:
    module = _load_eval_module()

    assert module.has_position_questions(
        [
            {module.FIELD_UUID: "q-binary", module.FIELD_TYPE: "binary"},
            {module.FIELD_UUID: "q-text", module.FIELD_TYPE: "text"},
            {module.FIELD_UUID: "q-time", module.FIELD_TYPE: "time"},
            {module.FIELD_UUID: "q-duration", module.FIELD_TYPE: "duration"},
        ]
    ) is False

    def _fail_load_position_reference(_path):
        raise AssertionError("position reference should stay archived for non-position runs")

    def _fail_load_pose_timeline(**_kwargs):
        raise AssertionError("pose timeline should stay archived for non-position runs")

    monkeypatch.setattr(module, "load_position_reference", _fail_load_position_reference)
    monkeypatch.setattr(module, "load_pose_timeline", _fail_load_pose_timeline)

    selected_has_position = False
    position_reference = (
        module.load_position_reference(None)
        if selected_has_position
        else None
    )
    position_windows = (
        module.load_pose_timeline(
            pose_timeline_json=None,
            dense_pose=None,
            timestamps=None,
            seconds_per_caption=3.0,
            seq_id="0",
        )
        if selected_has_position
        else []
    )

    assert position_reference is None
    assert position_windows == []


def test_non_position_accuracy_summary_excludes_position_contract() -> None:
    module = _load_eval_module()

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "uuid": "q-time",
                "question_type": "time",
                "official_evaluation_value": "4.04 minutes ago",
                "official_format_prediction": "4.15 minutes ago",
                "status": "live_validated",
            },
            {
                "uuid": "q-text",
                "question_type": "text",
                "official_evaluation_value": "a red car",
                "official_format_prediction": "a red car",
                "status": "live_validated",
            },
        ]
    )

    assert "position_prediction_contract" not in summary
    assert "Position labels" not in summary["note"]


def test_navqa_official_style_accuracy_uses_two_minute_temporal_threshold() -> None:
    module = _load_eval_module()

    results = [
        {
            "uuid": "q-time",
            "question": "When?",
            "question_type": "time",
            "official_evaluation_value": "4.00 minutes ago",
            "official_format_prediction": "5.50 minutes ago",
            "status": "live_validated",
        }
    ]

    official_style = module.navqa_official_style_accuracy_summary(results)

    assert official_style["contract"] == "navqa_official_style_accuracy_v1"
    assert official_style["temporal_threshold_minutes"] == 2.0
    assert official_style["correct"] == 1
    assert official_style["per_question"][0]["method"].startswith(
        "navqa_official_style_numeric_delta_minutes="
    )


def test_navqa_official_style_text_can_use_attached_llm_judgement() -> None:
    module = _load_eval_module()

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "uuid": "q-text",
                "question": "What color was it?",
                "question_type": "text",
                "official_evaluation_value": "They were wearing a blue puffer jacket",
                "official_format_prediction": "navy winter coat",
                "text_llm_judgement": {
                    "correct": True,
                    "reason": "Semantically matches the official answer.",
                    "extracted_answer": "navy winter coat",
                },
                "status": "live_validated",
            }
        ]
    )

    assert summary["correct"] == 1
    assert summary["per_question"][0]["method"] == "navqa_official_style_llm_text_judge"
    assert summary["text_judging"] == "llm_judge_when_available_else_normalized_containment_proxy"


def test_text_judge_result_normalization_accepts_json_fence() -> None:
    module = _load_eval_module()

    parsed = module._parse_json_object_from_text(
        '```json\n{"correct": true, "extracted_answer": "blue", "reason": "ok"}\n```'
    )
    normalized = module._normalize_text_judge_result(parsed)

    assert normalized == {
        "correct": True,
        "reason": "ok",
        "extracted_answer": "blue",
    }


def test_navqa_official_style_accuracy_summary_scores_mixed_answer_types() -> None:
    module = _load_eval_module()

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "uuid": "q1",
                "question": "When?",
                "question_type": "time",
                "official_evaluation_value": "4.04 minutes ago",
                "official_format_prediction": "4.15 minutes ago",
                "status": "live_validated",
            },
            {
                "uuid": "q2",
                "question": "Where?",
                "question_type": "position",
                "official_evaluation_value": "[-1.000000, 2.000000, 3.000000]",
                "official_format_prediction": "[-1.000000, 2.000000, 3.000000]",
                "official_format_prediction_source": "graph_event_time_to_pose_timeline",
                "official_format_prediction_details": {
                    "uses_question_uuid_position_label": False,
                    "uses_official_answer_timestamp": False,
                },
                "status": "live_fallback_graph_only",
            },
            {
                "uuid": "q3",
                "question": "Current?",
                "question_type": "binary",
                "official_evaluation_label": {"value": "yes"},
                "official_evaluation_value": "yes",
                "official_format_prediction": "no",
                "status": "live_validated",
            },
        ],
    )

    assert summary["correct"] == 2
    assert summary["total"] == 3
    assert summary["by_type"]["time"]["correct"] == 1
    assert summary["by_type"]["position"]["correct"] == 1
    assert summary["by_type"]["binary"]["correct"] == 0
    assert summary["status_counts"]["live_validated"] == 2
    assert summary["status_counts"]["live_fallback_graph_only"] == 1


def test_position_prediction_renders_from_graph_event_time_not_uuid_gold_label() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "LONG_position",
            module.FIELD_QUESTION: "Where did you see a fire hydrant?",
            module.FIELD_TYPE: "position",
            module.FIELD_CATEGORY: "spatial",
        },
        graph_answer={
            "retrieval_bundle": {
                "focus_frames": [
                    {
                        "rank": 1,
                        "timestamp_sec": 110.0,
                        "frame_index": 20,
                        "scene_id": "scene:001",
                        "reasons": ["top_match_start"],
                        "score": 10.0,
                    }
                ],
            },
            "matches": [
                {
                    "source_kind": "entity",
                    "score": 10.0,
                    "metadata": {
                        "time_range": {
                            "start_ts": 110.0,
                            "end_ts": 110.0,
                            "start_frame": 20,
                            "end_frame": 20,
                        },
                    },
                }
            ],
        },
        llm_answer={
            "answer_text": "The fire hydrant was seen at frame 20.",
            "structured_answer": {
                "used_focus_frames": [20],
                "used_time_anchor_ids": ["time_anchor:01"],
            },
        },
        official_label={
            "semantics": "ego_pose_at_semantic_event_time",
        },
        position_labels={
            "LONG_position": [-999.0, -999.0, -999.0],
        },
        position_windows=[
            module.PositionWindow(
                window_id="pose_window:110",
                start_time=110.0,
                end_time=111.0,
                position=(1.0, 2.0, 3.0),
                sample_count=2,
                file_start="110.pkl",
                file_end="111.pkl",
            )
        ],
        question_time_metadata={},
    )

    assert prediction["display"] == "[1.000000, 2.000000, 3.000000]"
    assert prediction["source"] == "graph_event_time_to_pose_timeline"
    assert prediction["details"]["uses_question_uuid_position_label"] is False
    assert prediction["details"]["uses_official_answer_timestamp"] is False


def test_time_prediction_renders_from_frame_time_index_without_gold_timestamp() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-time",
            module.FIELD_QUESTION: "When did you leave the building?",
            module.FIELD_TYPE: "time",
        },
        graph_answer={
            "retrieval_bundle": {
                "focus_frames": [
                    {"rank": 1, "frame_index": 1096, "scene_id": "scene:outside"}
                ],
            },
            "matches": [
                {
                    "source_kind": "temporal_composition",
                    "summary": "Indoor scene transitions to outdoor scene.",
                    "metadata": {
                        "time_range": {"start_frame": 1056, "end_frame": 1118}
                    },
                }
            ],
        },
        llm_answer={
            "answer_text": "You left the building around frame 1096.",
            "structured_answer": {"used_focus_frames": [1096]},
        },
        official_label={"semantics": "minutes_before_question_end_time"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                1096: {"frame_index": 1096, "timestamp_unix": 1_000.0},
            },
            "ordered_frames": [1096],
        },
        question_time_metadata={
            "q-time": {"start_time": 900.0, "end_time": 1_120.0}
        },
    )

    assert prediction["display"] == "2.00 minutes ago"
    assert prediction["source"] == "graph_event_frame_time_index_relative_to_question_end"
    assert prediction["details"]["uses_official_answer_timestamp"] is False


def test_time_prediction_prefers_question_ir_semantic_event_candidate() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-time",
            module.FIELD_QUESTION: "When did you first enter the building?",
            module.FIELD_TYPE: "time",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "operation": "onset",
                    "target_state": "enter building",
                    "target_entities": ["building"],
                }
            },
            "retrieval_bundle": {
                "semantic_event_candidates": [
                    {
                        "event_type": "enter",
                        "boundary_frame": 796,
                        "boundary_scene_id": "scene:inside",
                    }
                ],
                "focus_frames": [
                    {"rank": 1, "frame_index": 312, "scene_id": "scene:door"},
                    {"rank": 2, "frame_index": 796, "scene_id": "scene:inside"},
                ],
            },
            "matches": [],
        },
        llm_answer={
            "answer_text": "The first entry occurred at frame 312.",
            "structured_answer": {"used_focus_frames": [312]},
        },
        official_label={"semantics": "minutes_before_question_end_time"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                312: {"frame_index": 312, "timestamp_unix": 1_000.0},
                796: {"frame_index": 796, "timestamp_unix": 1_120.0},
            },
            "ordered_frames": [312, 796],
        },
        question_time_metadata={
            "q-time": {"start_time": 900.0, "end_time": 1_240.0}
        },
    )

    assert prediction["display"] == "2.00 minutes ago"
    assert prediction["details"]["event_frame_index"] == 796
    assert prediction["details"]["event_source"] == "question_ir_semantic_event_candidate"


def test_duration_prediction_renders_current_state_to_question_end() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-duration",
            module.FIELD_QUESTION: "How long have you been outside?",
            module.FIELD_TYPE: "duration",
        },
        graph_answer={
            "retrieval_bundle": {
                "focus_frames": [
                    {"rank": 1, "frame_index": 1096, "scene_id": "scene:outside"}
                ],
            },
            "matches": [
                {
                    "source_kind": "temporal_composition",
                    "summary": "Indoor scene transitions to outdoor scene.",
                    "metadata": {
                        "time_range": {"start_frame": 1056, "end_frame": 1118}
                    },
                }
            ],
        },
        llm_answer={
            "answer_text": "You have been outside from frame 1096 onward.",
            "structured_answer": {"used_focus_frames": [1096]},
        },
        official_label={"semantics": "event_duration_minutes"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                1096: {"frame_index": 1096, "timestamp_unix": 1_000.0},
            },
            "ordered_frames": [1096],
        },
        question_time_metadata={
            "q-duration": {"start_time": 900.0, "end_time": 1_060.0}
        },
    )

    assert prediction["display"] == "1.00 minutes"
    assert prediction["source"] == "graph_event_frame_time_index_current_duration"
    assert prediction["details"]["uses_official_answer_timestamp"] is False


def test_duration_prediction_prefers_temporal_span_candidate_interval() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-duration-span",
            module.FIELD_QUESTION: "How long were you waiting at the intersection?",
            module.FIELD_TYPE: "duration",
        },
        graph_answer={
            "retrieval_bundle": {
                "temporal_span_candidates": [
                    {
                        "source": "retrieved_match_time_range",
                        "source_kind": "scene",
                        "rank": 1,
                        "score": 20.0,
                        "start_frame": 0,
                        "end_frame": 1800,
                        "summary": "A broad road scene is visible.",
                    },
                    {
                        "source": "retrieved_match_time_range",
                        "source_kind": "stable_identity",
                        "rank": 2,
                        "score": 18.0,
                        "start_frame": 300,
                        "end_frame": 390,
                        "target_hits": ["waiting", "intersection"],
                        "summary": "The robot is waiting at the intersection.",
                    },
                ],
            },
            "matches": [
                {
                    "source_kind": "scene",
                    "summary": "Fallback broad scene interval.",
                    "metadata": {
                        "time_range": {"start_frame": 0, "end_frame": 1800}
                    },
                }
            ],
        },
        llm_answer={
            "answer_text": "The graph suggests waiting at the intersection.",
            "structured_answer": {},
        },
        official_label={"semantics": "event_duration_minutes"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                0: {"frame_index": 0, "timestamp_unix": 1_000.0},
                300: {"frame_index": 300, "timestamp_unix": 1_030.0},
                390: {"frame_index": 390, "timestamp_unix": 1_075.0},
                1800: {"frame_index": 1800, "timestamp_unix": 1_900.0},
            },
            "ordered_frames": [0, 300, 390, 1800],
        },
    )

    assert prediction["display"] == "0.75 minutes"
    assert prediction["source"] == "graph_interval_frame_time_index_duration"
    assert (
        prediction["details"]["interval_source"]
        == "temporal_span_candidate:retrieved_match_time_range"
    )
    assert prediction["details"]["start_frame"] == 300
    assert prediction["details"]["end_frame"] == 390
    assert prediction["details"]["uses_official_answer_timestamp"] is False


def test_duration_prediction_rejects_scene_span_as_interval_answer() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-duration-scene-only",
            module.FIELD_QUESTION: "How long were you in the building for?",
            module.FIELD_TYPE: "duration",
        },
        graph_answer={
            "retrieval_bundle": {
                "temporal_span_candidates": [
                    {
                        "source": "retrieved_match_time_range",
                        "source_kind": "scene",
                        "rank": 1,
                        "score": 20.0,
                        "start_frame": 0,
                        "end_frame": 1800,
                        "target_hits": ["building"],
                        "summary": "A broad building scene is visible.",
                    },
                    {
                        "source": "retrieved_match_time_range",
                        "source_kind": "transition_event",
                        "rank": 2,
                        "score": 18.0,
                        "start_frame": 450,
                        "end_frame": 450,
                        "target_hits": ["building"],
                        "summary": "A relation boundary near the building.",
                    },
                ],
            },
            "matches": [],
        },
        llm_answer={
            "answer_text": "insufficient evidence",
            "structured_answer": {},
        },
        official_label={"semantics": "event_duration_minutes"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                0: {"frame_index": 0, "timestamp_unix": 1_000.0},
                450: {"frame_index": 450, "timestamp_unix": 1_225.0},
                1800: {"frame_index": 1800, "timestamp_unix": 1_900.0},
            },
            "ordered_frames": [0, 450, 1800],
        },
    )

    assert prediction["display"] == "insufficient evidence"
    assert prediction["source"] == "llm_answer_text"
    assert prediction["source"] != "graph_interval_frame_time_index_duration"


def test_duration_prediction_rejects_zero_length_match_fallback() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-duration-zero-fallback",
            module.FIELD_QUESTION: "How long were you in the building for?",
            module.FIELD_TYPE: "duration",
        },
        graph_answer={
            "matches": [
                {
                    "source_kind": "transition_event",
                    "summary": "A relation boundary near the building.",
                    "metadata": {
                        "time_range": {"start_frame": 450, "end_frame": 450}
                    },
                },
                {
                    "source_kind": "episode",
                    "summary": "A zero-length episode-like record.",
                    "metadata": {
                        "time_range": {"start_frame": 700, "end_frame": 700}
                    },
                },
            ],
        },
        llm_answer={
            "answer_text": "insufficient evidence",
            "structured_answer": {},
        },
        official_label={"semantics": "event_duration_minutes"},
        frame_time_index={
            "source_path": "test-index.json",
            "by_frame": {
                450: {"frame_index": 450, "timestamp_unix": 1_225.0},
                700: {"frame_index": 700, "timestamp_unix": 1_350.0},
            },
            "ordered_frames": [450, 700],
        },
    )

    assert prediction["display"] == "insufficient evidence"
    assert prediction["source"] == "llm_answer_text"
    assert prediction["source"] != "graph_interval_frame_time_index_duration"


def test_duration_answer_consistency_flags_llm_arithmetic_conflict() -> None:
    module = _load_eval_module()

    official_prediction = {
        "display": "2.50 minutes",
        "source": "graph_interval_frame_time_index_duration",
        "semantics": "event_duration_minutes",
        "details": {
            "start_frame": 796,
            "end_frame": 1096,
            "duration_seconds": 150.016,
            "interval_source": "llm_answer_frame_pair",
            "uses_official_answer_timestamp": False,
        },
    }
    raw_answer = (
        "The graph evidence provides an entry event at frame 796 and a leave "
        "event at frame 1096. The duration between these timestamps is "
        "approximately 5.0 minutes."
    )

    consistency = module.answer_consistency_for_official_render(
        {
            module.FIELD_UUID: "q-duration",
            module.FIELD_QUESTION: "How long were you in the building for?",
            module.FIELD_TYPE: "duration",
        },
        official_format_prediction=official_prediction,
        raw_answer_text=raw_answer,
    )

    assert consistency["status"] == "raw_llm_numeric_conflict_with_graph_render"
    assert consistency["rendered_minutes"] == 2.5
    assert consistency["raw_minutes"] == 5.0
    assert consistency["raw_answer_frames"] == [796, 1096]
    assert (
        module.final_predicted_answer_for_output(
            official_format_prediction=official_prediction,
            raw_answer_text=raw_answer,
        )
        == "2.50 minutes"
    )


def test_duration_answer_consistency_accepts_matching_llm_duration_text() -> None:
    module = _load_eval_module()

    consistency = module.answer_consistency_for_official_render(
        {
            module.FIELD_UUID: "q-duration",
            module.FIELD_QUESTION: "How long were you in the building for?",
            module.FIELD_TYPE: "duration",
        },
        official_format_prediction={
            "display": "2.50 minutes",
            "source": "graph_interval_frame_time_index_duration",
            "semantics": "event_duration_minutes",
            "details": {"uses_official_answer_timestamp": False},
        },
        raw_answer_text="The graph-backed interval is approximately 2 minutes 30 seconds.",
    )

    assert consistency["status"] == "consistent"
    assert consistency["rendered_minutes"] == 2.5
    assert consistency["raw_minutes"] == 2.5


def test_time_answer_consistency_ignores_unix_seconds_and_accepts_ranges() -> None:
    module = _load_eval_module()

    consistency = module.answer_consistency_for_official_render(
        {
            module.FIELD_UUID: "q-time",
            module.FIELD_QUESTION: "When did you get off of the brick sidewalk?",
            module.FIELD_TYPE: "time",
        },
        official_format_prediction={
            "display": "1.71 minutes ago",
            "source": "graph_time_anchor_relative_to_question_end",
            "semantics": "minutes_before_question_end_time",
            "details": {"uses_official_answer_timestamp": False},
        },
        raw_answer_text=(
            "The event is around frame 1378-1380, approximately "
            "1673884874.65s to 1673884875.65s."
        ),
    )

    assert consistency["status"] == "no_explicit_raw_numeric_duration"
    assert consistency["raw_minutes"] is None
    assert consistency["raw_minute_candidates"] == []

    range_consistency = module.answer_consistency_for_official_render(
        {
            module.FIELD_UUID: "q-time",
            module.FIELD_QUESTION: "When were you in the atrium with the red couches?",
            module.FIELD_TYPE: "time",
        },
        official_format_prediction={
            "display": "1.86 minutes ago",
            "source": "graph_time_anchor_relative_to_question_end",
            "semantics": "minutes_before_question_end_time",
            "details": {"uses_official_answer_timestamp": False},
        },
        raw_answer_text="The ego was in the scene from about 1.9 to 1.5 minutes ago.",
    )

    assert range_consistency["status"] == "consistent"
    assert range_consistency["raw_minutes"] == 1.9
    assert range_consistency["raw_minute_candidates"] == [1.9, 1.5]


def test_raw_answer_remains_output_when_render_is_not_graph_backed() -> None:
    module = _load_eval_module()

    assert (
        module.final_predicted_answer_for_output(
            official_format_prediction={
                "display": "2.50 minutes",
                "source": "llm_answer_duration_text",
                "semantics": "event_duration_minutes",
            },
            raw_answer_text="approximately 2.5 minutes",
        )
        == "approximately 2.5 minutes"
    )


def test_time_event_frame_uses_start_for_crossing_interval() -> None:
    module = _load_eval_module()

    selected = module._event_frame_from_interval_for_question(
        {"start_frame": 1238, "end_frame": 1278},
        question="When did you cross the brick street?",
        answer_text="The crossing happened during frames 1238-1278.",
    )

    assert selected == 1238


def test_time_render_uses_temporal_point_candidate_pack() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-time-point",
            module.FIELD_QUESTION: "When did you cross the street?",
            module.FIELD_TYPE: "time",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "operation": "occurrence",
                    "question_route": {"route_id": "temporal_point"},
                },
                "structured_graph_context": {
                    "question_route_temporal_point": {
                        "selection_status": "selected",
                        "selected_candidate": {
                            "frame": 100,
                            "timestamp_sec": 940.0,
                            "score": 31.0,
                            "query_phrase": "a photo of crossing street",
                            "label": "pedestrian crossing",
                            "result_rank": 1,
                        },
                    }
                },
            }
        },
        llm_answer={"answer_text": "about 1 minute ago"},
        official_label={"semantics": "minutes_before_question_end_time"},
        question_time_metadata={"q-time-point": {"end_time": 1000.0}},
    )

    assert prediction["display"] == "1.00 minutes ago"
    assert prediction["source"] == "graph_question_route_temporal_point_candidate"
    assert prediction["details"]["event_frame_index"] == 100
    assert prediction["details"]["question_route_id"] == "temporal_point"


def test_time_render_ignores_temporal_point_pack_for_other_routes() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-time-interval",
            module.FIELD_QUESTION: "When did you cross the street?",
            module.FIELD_TYPE: "time",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "operation": "occurrence",
                    "question_route": {"route_id": "temporal_interval"},
                },
                "structured_graph_context": {
                    "question_route_temporal_point": {
                        "selection_status": "selected",
                        "selected_candidate": {
                            "frame": 100,
                            "timestamp_sec": 940.0,
                            "score": 31.0,
                        },
                    }
                },
            }
        },
        llm_answer={"answer_text": "about 1 minute ago"},
        official_label={"semantics": "minutes_before_question_end_time"},
        question_time_metadata={"q-time-interval": {"end_time": 1000.0}},
    )

    assert prediction["display"] == "about 1 minute ago"
    assert prediction["source"] == "llm_answer_text_no_time_anchor_render"


def test_text_prediction_does_not_override_llm_answer_with_color_evidence() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-color",
            module.FIELD_QUESTION: "What were the color of the couches near the entrance?",
            module.FIELD_TYPE: "text",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "operation": "attribute_query",
                    "target_entities": ["couches"],
                    "attribute_dimension": "color",
                    "aggregation": "multiple",
                    "rendering": "color_only",
                },
                "evidence_aggregation": {
                    "available": True,
                    "groups": [
                        {
                            "group_type": "entity",
                            "group_id": "entity:red_sofa",
                            "match_ranks": [2, 3],
                            "target_hits": ["sofa"],
                            "requirement_hits": ["couches"],
                            "attributes": [
                                "visual_attribute=red cushioned seat near entrance",
                                "visual_attribute=red bench in lobby",
                            ],
                        }
                    ],
                },
                "evidence_lines": [
                    "rank=1 kind=entity_attribute :: sofa visual_attribute=brown leather sofa",
                    "rank=2 kind=entity_attribute :: sofa visual_attribute=red cushioned seat on gray metal support column",
                    "rank=3 kind=entity_attribute :: sofa visual_attribute=red bench in lobby",
                    "rank=4 kind=entity_attribute :: chair visual_attribute=brown wooden chair in lobby",
                ],
            }
        },
        llm_answer={
            "answer_text": "The seating was mixed.",
            "structured_answer": {},
        },
        official_label={"semantics": "free_text_human_judged"},
    )

    assert prediction["display"] == "The seating was mixed."
    assert prediction["source"] == "llm_answer_text"


def test_binary_prediction_rejects_conflicting_llm_yes_no_text() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary",
            module.FIELD_QUESTION: "Did you see any bicyclists?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={},
        llm_answer={
            "answer_text": (
                "Yes, the graph evidence shows bicycles, but no bicyclists are "
                "directly detected."
            ),
            "structured_answer": {},
        },
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"].startswith("Yes,")
    assert prediction["source"] == "llm_answer_text_binary_ambiguous"
    assert prediction["semantics"] == "natural_language"
    assert prediction["details"]["reason"] == "conflicting_yes_no_cues"

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "uuid": "q-binary",
                "question": "Did you see any bicyclists?",
                "question_type": "binary",
                "official_evaluation_label": {"value": "yes"},
                "official_evaluation_value": "yes",
                "official_format_prediction": prediction["display"],
                "official_format_prediction_source": prediction["source"],
                "status": "live_validated",
            }
        ]
    )

    assert summary["correct"] == 0
    assert (
        summary["per_question"][0]["method"]
        == "navqa_official_style_predicted_binary=unknown"
    )


def test_binary_prediction_accepts_clear_llm_yes_no_text() -> None:
    module = _load_eval_module()

    yes_prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-yes",
            module.FIELD_QUESTION: "Did you see a red fire hydrant?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={},
        llm_answer={
            "answer_text": "Yes, direct graph evidence shows a red fire hydrant.",
            "structured_answer": {},
        },
        official_label={"semantics": "yes_no"},
    )
    no_prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-no",
            module.FIELD_QUESTION: "Was the sidewalk busy?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={},
        llm_answer={
            "answer_text": "No, the graph evidence does not show a busy sidewalk.",
            "structured_answer": {},
        },
        official_label={"semantics": "yes_no"},
    )

    assert yes_prediction["display"] == "yes"
    assert yes_prediction["source"] == "llm_answer_text_binary_clear"
    assert no_prediction["display"] == "no"
    assert no_prediction["source"] == "llm_answer_text_binary_clear"


def test_binary_existence_contract_uses_selected_yes_cue() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-selected-yes",
            module.FIELD_QUESTION: "Did you see any bicyclists?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["bicyclists"],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["bicyclists"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:7",
                            "text": "object_attribute=Person riding black bicycle",
                        }
                    ],
                    "answer_cue": "yes",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "yes"
    assert prediction["source"] == "graph_question_route_binary_existence_selected_cue"


def test_binary_existence_contract_ignores_query_echo_only() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-query-echo",
            module.FIELD_QUESTION: "Did you see a scooter?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["scooter"],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["scooter"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:1",
                            "text": (
                                "query='a photo of scooter'; "
                                "object obj=13 label='skateboard' "
                                "attr=Black skateboard with white shoes"
                            ),
                        }
                    ],
                    "answer_cue": "no",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "no"
    assert prediction["source"] == "llm_answer_text_binary_clear"


def test_binary_existence_contract_accepts_observed_target_in_selected_evidence() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-selected-target",
            module.FIELD_QUESTION: "Did you see a scooter?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["scooter"],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["scooter"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:1",
                            "text": (
                                "query='a photo of scooter'; "
                                "object obj=13 label='skateboard'; "
                                "neighbor obj=2 label='man2' "
                                "attr=Man wearing blue shirt riding scooter"
                            ),
                        }
                    ],
                    "answer_cue": "no",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "yes"
    assert prediction["source"] == "graph_question_route_binary_existence_evidence_contract"
    assert prediction["details"]["support_source"] == "selected_evidence"


def test_binary_existence_contract_applies_to_existence_query_with_target_state() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-target-state",
            module.FIELD_QUESTION: "Did the robot stop near the stop sign?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_state": "stopped near stop sign",
                    "target_entities": ["robot", "stop sign"],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["robot", "stop sign", "stopped near stop sign"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:4",
                            "text": "object_attribute=Red and white stop sign on the corner",
                        }
                    ],
                    "answer_cue": "maybe",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "yes"
    assert prediction["source"] == "graph_question_route_binary_existence_evidence_contract"
    assert prediction["details"]["support_source"] == "selected_evidence"


def test_binary_existence_contract_uses_candidate_head_term_when_selection_empty() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-candidate-head",
            module.FIELD_QUESTION: "Did you see an elevated pedestrian bridge?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["elevated pedestrian bridge"],
                    "hard_constraints": [
                        {
                            "kind": "entity",
                            "text": "elevated pedestrian bridge",
                        }
                    ],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["elevated pedestrian bridge"],
                    },
                },
                "visual_faiss_ger_bound_fact_lines": [
                    "rank=1 kind=entity_attribute :: roof visual_attribute=metal bridge under blue sky"
                ],
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "yes"
    assert prediction["source"] == "graph_question_route_binary_existence_evidence_contract"
    assert prediction["details"]["support_source"] == "candidate_evidence"
    assert prediction["details"]["support_match_kind"] == "head_term"


def test_binary_existence_contract_does_not_join_attribute_tokens_across_owner() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-attribute-binding",
            module.FIELD_QUESTION: "Can I ride a bike on the yellow road you saw?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "existence_query",
                    "target_entities": ["yellow road"],
                    "hard_constraints": [
                        {
                            "kind": "attribute",
                            "text": "yellow",
                            "role": "target_modifier",
                            "target_entities": ["yellow road"],
                        }
                    ],
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["yellow road"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:17",
                            "text": "pole visual_attribute=yellow metal pole in middle of road",
                        }
                    ],
                    "answer_cue": "yellow metal pole in middle of road",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "no"
    assert prediction["source"] == "llm_answer_text_binary_clear"


def test_binary_existence_contract_does_not_apply_to_current_state() -> None:
    module = _load_eval_module()

    prediction = module.render_official_format_prediction(
        {
            module.FIELD_UUID: "q-binary-current-state",
            module.FIELD_QUESTION: "Was the brick sidewalk busy?",
            module.FIELD_TYPE: "binary",
        },
        graph_answer={
            "llm_ready_bundle": {
                "question_ir": {
                    "answer_type": "binary",
                    "operation": "current_state",
                    "target_entities": ["brick sidewalk"],
                    "target_state": "busy",
                    "question_route": {
                        "route_id": "binary_existence",
                        "primary_target_terms": ["brick sidewalk", "busy"],
                    },
                },
                "llm_evidence_selection": {
                    "selected_evidence": [
                        {
                            "id": "evidence_line:1",
                            "text": "object obj=1 label='brick pavement'",
                        }
                    ],
                    "answer_cue": "yes",
                    "insufficient_evidence": False,
                },
            }
        },
        llm_answer={"answer_text": "no", "structured_answer": {}},
        official_label={"semantics": "yes_no"},
    )

    assert prediction["display"] == "no"
    assert prediction["source"] == "llm_answer_text_binary_clear"


def test_position_score_rejects_uuid_label_prediction_source() -> None:
    module = _load_eval_module()

    summary = module.navqa_official_style_accuracy_summary(
        [
            {
                "uuid": "q-position",
                "question": "Where did you see the hydrant?",
                "question_type": "position",
                "official_evaluation_value": "[1.000000, 2.000000, 3.000000]",
                "official_format_prediction": "[1.000000, 2.000000, 3.000000]",
                "official_format_prediction_source": "question_uuid_position_label",
                "official_format_prediction_details": {
                    "uses_question_uuid_position_label": True,
                    "uses_official_answer_timestamp": True,
                },
                "status": "live_validated",
            }
        ]
    )

    score = summary["per_question"][0]
    assert summary["correct"] == 0
    assert score["method"] == "navqa_official_style_invalid_position_prediction_source"


def test_pose_timeline_loads_local_seq_metadata_by_default() -> None:
    module = _load_eval_module()
    if not Path(".scratch/coda_seq0_metadata/poses/0.txt").exists():
        return
    windows = module.load_pose_timeline(
        pose_timeline_json=None,
        dense_pose=None,
        timestamps=None,
        seconds_per_caption=3.0,
        seq_id="0",
    )

    assert windows


def test_position_questions_require_pose_timeline() -> None:
    module = _load_eval_module()

    try:
        module.require_pose_timeline_for_position_questions(
            rows=[
                {
                    module.FIELD_TYPE: "position",
                    module.FIELD_QUESTION: "Where did you see the hydrant?",
                }
            ],
            position_windows=[],
            seq_id="missing",
        )
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing pose timeline")

    assert "Selected NaVQA questions include position answers" in message
    assert "download_coda_zip_member.py poses/missing.txt" in message
