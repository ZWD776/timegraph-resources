from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_NAME = "navqa_official_style_accuracy"
QUESTION_TYPES = ("text", "binary", "time", "duration")


def _percent(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct * 100.0 / total, 2)


def _empty_bucket() -> dict[str, Any]:
    return {"correct": 0, "total": 0, "accuracy_percent": 0.0}


def _add_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["correct"] += int(source.get("correct", 0) or 0)
    target["total"] += int(source.get("total", 0) or 0)
    target["accuracy_percent"] = _percent(target["correct"], target["total"])


def _row_official_style_metric(row: dict[str, Any]) -> dict[str, Any] | None:
    new_summary = row.get("new")
    if not isinstance(new_summary, dict):
        return None

    if new_summary.get("metric") == METRIC_NAME:
        return new_summary

    return None


def summarize_payload(payload: dict[str, Any], *, source_path: str | None = None) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("expected an all-sequence summary with a 'rows' list")

    overall = _empty_bucket()
    by_type = {question_type: _empty_bucket() for question_type in QUESTION_TYPES}
    used_rows = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        metric = _row_official_style_metric(row)
        if metric is None:
            continue

        used_rows += 1
        _add_bucket(overall, metric)

        metric_by_type = metric.get("by_type", {})
        if not isinstance(metric_by_type, dict):
            continue
        for question_type in QUESTION_TYPES:
            bucket = metric_by_type.get(question_type)
            if isinstance(bucket, dict):
                _add_bucket(by_type[question_type], bucket)

    if used_rows == 0:
        raise ValueError(f"no {METRIC_NAME} rows found")

    run_metadata = payload.get("run_metadata", {})
    if not isinstance(run_metadata, dict):
        run_metadata = {}

    return {
        "source_path": source_path,
        "run_name": payload.get("run_name"),
        "route_id": run_metadata.get("route_id"),
        "cg_id": run_metadata.get("cg_id"),
        "ir_qa_id": run_metadata.get("ir_qa_id"),
        "metric": METRIC_NAME,
        "rows_used": used_rows,
        "overall": overall,
        "by_type": by_type,
    }


def summarize_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary JSON root must be an object")
    return summarize_payload(payload, source_path=str(path))


def _format_bucket(bucket: dict[str, Any]) -> str:
    return (
        f"{bucket.get('correct', 0)}/{bucket.get('total', 0)} = "
        f"{float(bucket.get('accuracy_percent', 0.0)):.2f}%"
    )


def format_text_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"source: {summary.get('source_path')}",
        f"run_name: {summary.get('run_name')}",
        f"route_id: {summary.get('route_id')}",
        f"metric: {summary.get('metric')}",
        f"overall: {_format_bucket(summary['overall'])}",
    ]
    by_type = summary.get("by_type", {})
    for question_type in QUESTION_TYPES:
        bucket = by_type.get(question_type, _empty_bucket())
        lines.append(f"{question_type}: {_format_bucket(bucket)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize NaVQA all-sequence results using only "
            "navqa_official_style_accuracy."
        )
    )
    parser.add_argument(
        "summary_json",
        nargs="+",
        type=Path,
        help="Path(s) to mainline_all_sequences_summary.json.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [summarize_file(path) for path in args.summary_json]

    if args.format == "json":
        output: dict[str, Any] | list[dict[str, Any]]
        output = summaries[0] if len(summaries) == 1 else summaries
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    for index, summary in enumerate(summaries):
        if index:
            print()
        print(format_text_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
