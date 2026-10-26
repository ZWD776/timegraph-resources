from __future__ import annotations

import csv
import json
from pathlib import Path

from vlkg.datasets.nextqa.audit.archive_audit import parse_frame_filename
from vlkg.datasets.nextqa.audit.preflight import _video_timing_profile
from vlkg.datasets.nextqa.runtime.standardization import (
    build_evaluator_labels,
    build_runtime_questions,
    contains_gold_label,
    load_official_mapping,
    load_validation_annotations,
    stable_question_id,
)


def _write_validation(path: Path) -> None:
    columns = [
        "video",
        "frame_count",
        "width",
        "height",
        "question",
        "answer",
        "qid",
        "type",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for video_id in ("100", "200"):
            for qid in (0, 1):
                writer.writerow(
                    {
                        "video": video_id,
                        "frame_count": 90,
                        "width": 640,
                        "height": 480,
                        "question": " what   happened next ",
                        "answer": qid,
                        "qid": qid,
                        "type": "TN",
                        "a0": "choice 0",
                        "a1": "choice 1",
                        "a2": "choice 2",
                        "a3": "choice 3",
                        "a4": "choice 4",
                    }
                )


def test_stable_question_id_scopes_official_qid_to_video(tmp_path: Path) -> None:
    annotation = tmp_path / "val.csv"
    _write_validation(annotation)

    rows, audit = load_validation_annotations(annotation)

    assert audit["row_count"] == 4
    assert audit["unique_bare_qid_count"] == 2
    assert audit["unique_stable_question_id_count"] == 4
    assert audit["duplicate_stable_question_ids"] == []
    assert stable_question_id("100", "0") == "nextqa-val/100/q0"
    assert rows[0]["normalized_question"] == "what happened next"


def test_runtime_questions_keep_gold_in_evaluator_only_file(tmp_path: Path) -> None:
    annotation = tmp_path / "val.csv"
    _write_validation(annotation)
    rows, _ = load_validation_annotations(annotation)

    runtime = build_runtime_questions(rows)
    labels = build_evaluator_labels(rows)

    assert not contains_gold_label(runtime)
    assert all("gold_choice_index" not in record for record in runtime)
    assert all("gold_choice_index" in record for record in labels)
    assert {record["question_id"] for record in runtime} == {
        record["question_id"] for record in labels
    }


def test_mapping_and_real_timestamp_parsing(tmp_path: Path) -> None:
    mapping_path = tmp_path / "map.json"
    mapping_path.write_text(json.dumps({"100": "0001/100"}), encoding="utf-8")

    assert load_official_mapping(mapping_path) == {"100": "0001/100"}
    assert parse_frame_filename("frame_000030_0001.001s.png") == (30, 1.001, "png")


def test_timestamp_profile_respects_per_video_fps_and_rounding() -> None:
    profile = _video_timing_profile(
        {
            "frame_names": [
                "frame_000000_0000.000s.png",
                "frame_000030_0001.251s.png",
                "frame_000060_0002.503s.png",
            ]
        }
    )

    assert profile["timestamp_consistent_with_single_fps_and_3_decimal_rounding"]
    assert 23.97 < float(profile["inferred_source_fps"]) < 23.99
    assert float(profile["median_sample_interval_seconds"]) == 1.2515
