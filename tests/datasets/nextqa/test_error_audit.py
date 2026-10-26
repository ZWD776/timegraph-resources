from __future__ import annotations

import json
from pathlib import Path

import vlkg.datasets.nextqa.audit.error_audit as error_audit_module
from vlkg.datasets.nextqa.audit.error_audit import audit_prediction_rows


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_error_audit_distinguishes_retrieval_miss_from_graph_gap(tmp_path: Path) -> None:
    graph_path = _write(
        tmp_path / "canonical.json",
        {
            "entity_nodes": [
                {"node_id": "entity:boy", "label": "boy", "time_range": {"start_ts": 0.0}},
            ],
            "episode_nodes": [
                {"node_id": "episode:walk", "label": "boy walking away", "time_range": {"start_ts": 2.0}},
            ],
            "transition_event_nodes": [],
            "scene_nodes": [],
        },
    )
    input_dir = tmp_path / "input"
    _write(input_dir / "objects.json", {"frame_000000_0000.000s": {"labels": ["boy"]}})
    _write(input_dir / "attributes.json", {"frame_000000_0000.000s": {"0": "boy walking away"}})
    _write(input_dir / "relations.json", {"frame_000000_0000.000s": []})
    predictions = [{"question_id": "nextqa-val/v/q0", "video_id": "v", "status": "completed", "predicted_choice_index": 0, "predicted_choice_text": "boy", "graph_record_ids": [], "reasoning": "", "used_evidence_ranks": []}]
    runtime = [{"question_id": "nextqa-val/v/q0", "video_id": "v", "official_type": "TN", "normalized_question": "what happened after", "answer_choices": ["boy walking away", "sit", "wave", "run", "jump"]}]
    labels = [{"question_id": "nextqa-val/v/q0", "video_id": "v", "gold_choice_index": 0}]
    retrieval = [{"question_id": "nextqa-val/v/q0", "video_id": "v", "evidence": [], "graph_record_ids": []}]
    result = audit_prediction_rows(predictions, runtime, labels, retrieval, graph_paths={"v": graph_path}, input_dirs={"v": input_dir})
    row = result["rows"][0]
    assert row["signal_audit"]["gold_choice_hsgg_hit"] is True
    assert row["signal_audit"]["gold_choice_full_canonical_hit"] is True
    assert row["primary_category"] == "candidate_recall_gap"
    assert result["purity"]["reads_rgb"] is False


def test_error_audit_marks_unrepresented_gold_as_hsgg_gap(tmp_path: Path) -> None:
    graph_path = _write(tmp_path / "canonical.json", {"entity_nodes": [], "episode_nodes": [], "transition_event_nodes": [], "scene_nodes": []})
    input_dir = tmp_path / "input"
    _write(input_dir / "objects.json", {"frame": {"labels": ["person"]}})
    _write(input_dir / "attributes.json", {"frame": {"0": "person"}})
    _write(input_dir / "relations.json", {"frame": []})
    base = {"question_id": "nextqa-val/v/q0", "video_id": "v"}
    result = audit_prediction_rows(
        [{**base, "status": "completed", "predicted_choice_index": 1, "predicted_choice_text": "chair", "graph_record_ids": []}],
        [{**base, "official_type": "CH", "normalized_question": "why", "answer_choices": ["unseen action", "chair", "sit", "walk", "run"]}],
        [{**base, "gold_choice_index": 0}],
        [{**base, "evidence": [], "graph_record_ids": []}],
        graph_paths={"v": graph_path}, input_dirs={"v": input_dir},
    )
    assert result["rows"][0]["primary_category"] == "causal_reasoning_gap"


def test_error_audit_loads_structured_records_once_per_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph_path = _write(
        tmp_path / "canonical.json",
        {
            "entity_nodes": [{"node_id": "entity:room", "label": "room"}],
            "episode_nodes": [],
            "transition_event_nodes": [],
            "scene_nodes": [],
        },
    )
    input_dir = tmp_path / "input"
    _write(input_dir / "objects.json", {"frame": {"labels": ["room"]}})
    _write(input_dir / "attributes.json", {"frame": {}})
    _write(input_dir / "relations.json", {"frame": []})
    canonical_calls = 0
    original = error_audit_module._canonical_records

    def counted(graph):
        nonlocal canonical_calls
        canonical_calls += 1
        return original(graph)

    monkeypatch.setattr(error_audit_module, "_canonical_records", counted)
    base_rows = [
        {"question_id": f"nextqa-val/v/q{index}", "video_id": "v"}
        for index in range(2)
    ]
    audit_prediction_rows(
        [
            {
                **row,
                "status": "completed",
                "predicted_choice_index": 0,
                "predicted_choice_text": "room",
                "graph_record_ids": [],
            }
            for row in base_rows
        ],
        [
            {
                **row,
                "official_type": "DL",
                "normalized_question": "where",
                "answer_choices": ["room", "park", "road", "farm", "stage"],
            }
            for row in base_rows
        ],
        [{**row, "gold_choice_index": 0} for row in base_rows],
        [{**row, "evidence": [], "graph_record_ids": []} for row in base_rows],
        graph_paths={"v": graph_path},
        input_dirs={"v": input_dir},
    )

    assert canonical_calls == 1
