from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.official_qa import (  # noqa: E402
    load_position_labels,
    materialize_official_answer_label,
    official_label_to_dict,
)
from vlkg.datasets.navqa.runtime.paths import (  # noqa: E402
    default_navqa_sequence_layout,
)
from vlkg.datasets.navqa.qa.position_reference import (  # noqa: E402
    FIELD_CATEGORY,
    FIELD_PARSABLE_ANSWER,
    FIELD_QUESTION,
    FIELD_SEQ_ID,
    FIELD_TEXT_ANSWER,
    FIELD_TIMESTAMP,
    FIELD_TYPE,
    FIELD_UUID,
)
from vlkg.core.temporal.pathing import resolve_repo_input_path  # noqa: E402


CURRENT_MAINLINE_TYPES = ("binary", "text", "time", "duration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare sequence-scoped NaVQA question and gold-answer files from "
            "the official data.csv and qa_unfilled.json."
        )
    )
    parser.add_argument("--seq-id", required=True)
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path(".scratch/navqa"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(".scratch/navqa_official_data.csv"),
        help="Official NaVQA data.csv.",
    )
    parser.add_argument("--qa-unfilled-json", type=Path, required=True)
    parser.add_argument(
        "--position-reference-json",
        type=Path,
        default=None,
        help=(
            "Optional position reference produced by build_navqa_position_reference.py. "
            "If omitted, position gold remains the official [position] placeholder."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to .scratch/navqa/seq{seq-id}.",
    )
    parser.add_argument(
        "--no-copy-qa-unfilled",
        action="store_true",
        help="Do not copy qa_unfilled.json into the sequence layout.",
    )
    return parser


def load_official_rows(csv_path: Path, *, seq_id: str) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get(FIELD_SEQ_ID, "")).strip() == str(seq_id)
        ]


def load_qa_unfilled(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"qa_unfilled JSON must contain an object: {path}")
    items = payload.get("data", [])
    if not isinstance(items, list):
        raise ValueError(f"qa_unfilled.data must be a list: {path}")
    return payload


def question_metadata_aliases(
    qa_unfilled: dict[str, Any],
    *,
    seq_id: str,
) -> dict[str, dict[str, Any]]:
    items = qa_unfilled.get("data", [])
    metadata: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return metadata
    prefix = f"{seq_id}_"
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id", "")).strip()
        if not raw_id:
            continue
        metadata[raw_id] = item
        if raw_id.startswith(prefix):
            metadata.setdefault(raw_id[len(prefix):], item)
        else:
            metadata.setdefault(f"{prefix}{raw_id}", item)
        if raw_id.isdigit():
            metadata.setdefault(f"Q{raw_id}", item)
    return metadata


def build_records(
    rows: list[dict[str, str]],
    *,
    seq_id: str,
    question_metadata: dict[str, dict[str, Any]],
    position_labels: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        uuid = str(row.get(FIELD_UUID, "")).strip()
        qtype = normalized_question_type(row)
        label = materialize_official_answer_label(
            row,
            question_metadata=question_metadata,
            position_labels=position_labels,
        )
        metadata = question_metadata.get(uuid)
        records.append(
            {
                "index": index,
                "uuid": uuid,
                "seq_id": seq_id,
                "question": str(row.get(FIELD_QUESTION, "")).strip(),
                "question_type": qtype,
                "question_category": str(row.get(FIELD_CATEGORY, "")).strip(),
                "answer_timestamp_text": str(row.get(FIELD_TIMESTAMP, "")).strip(),
                "text_answer": str(row.get(FIELD_TEXT_ANSWER, "")).strip(),
                "parsable_answer": str(row.get(FIELD_PARSABLE_ANSWER, "")).strip(),
                "official_answer": official_answer_text(row),
                "gold_answer": label.display,
                "official_label": official_label_to_dict(label),
                "is_current_mainline": qtype in CURRENT_MAINLINE_TYPES,
                "question_metadata": compact_question_metadata(metadata),
            }
        )
    return records


def compact_question_metadata(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    file_info = item.get("file_info", {})
    if not isinstance(file_info, dict):
        file_info = {}
    return {
        "id": item.get("id"),
        "length_category": item.get("length_category"),
        "length": item.get("length"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
        "qa_start_filename": file_info.get("qa_start_filename"),
        "qa_end_filename": file_info.get("qa_end_filename"),
    }


def metadata_alignment(
    rows: list[dict[str, str]],
    question_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matched: list[str] = []
    missing: list[str] = []
    for row in rows:
        uuid = str(row.get(FIELD_UUID, "")).strip()
        if uuid and uuid in question_metadata:
            matched.append(uuid)
        else:
            missing.append(uuid)
    return {
        "matched_count": len(matched),
        "missing_count": len(missing),
        "missing_uuids": missing,
    }


def counts_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "").strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def official_answer_text(row: dict[str, str]) -> str:
    text = str(row.get(FIELD_TEXT_ANSWER, "")).strip()
    parsable = str(row.get(FIELD_PARSABLE_ANSWER, "")).strip()
    if text and parsable:
        return f"{text} (parsable: {parsable})"
    return text or parsable


def normalized_question_type(row: dict[str, str]) -> str:
    return str(row.get(FIELD_TYPE, "")).strip().lower()


def question_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "UUID": record["uuid"],
            "Seq ID": record["seq_id"],
            "Question": record["question"],
            "Timestamp \nwith answer": record["answer_timestamp_text"],
            "Type \n(binary, position, time, text)": record["question_type"],
            "Text answer": record["text_answer"],
            "Parsable answer": record["parsable_answer"],
            "Question\nCategory": record["question_category"],
        }
        for record in records
    ]


def build_payload(
    *,
    records: list[dict[str, Any]],
    seq_id: str,
    csv_path: Path,
    qa_unfilled_path: Path,
    position_reference_path: Path | None,
    alignment: dict[str, Any],
    current_mainline_types: tuple[str, ...] = CURRENT_MAINLINE_TYPES,
) -> dict[str, Any]:
    non_position = [
        record
        for record in records
        if str(record.get("question_type") or "").strip().lower()
        in current_mainline_types
    ]
    return {
        "contract": "navqa_sequence_qa_gold_v1",
        "seq_id": seq_id,
        "current_mainline_types": list(current_mainline_types),
        "source": {
            "csv": str(csv_path),
            "qa_unfilled_json": str(qa_unfilled_path),
            "position_reference_json": (
                str(position_reference_path) if position_reference_path is not None else None
            ),
        },
        "counts": {
            "all": len(records),
            "current_mainline_non_position": len(non_position),
            "by_type": counts_by(records, "question_type"),
            "by_category": counts_by(records, "question_category"),
        },
        "metadata_alignment": alignment,
        "records": records,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "uuid",
        "seq_id",
        "question_type",
        "question_category",
        "question",
        "answer_timestamp_text",
        "official_answer",
        "gold_answer",
        "evaluation_key",
        "evaluation_value",
        "evaluation_semantics",
        "uses_official_answer_timestamp",
        "is_current_mainline",
        "question_start_time",
        "question_end_time",
        "metadata_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            label = record.get("official_label", {})
            if not isinstance(label, dict):
                label = {}
            metadata = record.get("question_metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            writer.writerow(
                {
                    "index": record.get("index"),
                    "uuid": record.get("uuid"),
                    "seq_id": record.get("seq_id"),
                    "question_type": record.get("question_type"),
                    "question_category": record.get("question_category"),
                    "question": record.get("question"),
                    "answer_timestamp_text": record.get("answer_timestamp_text"),
                    "official_answer": record.get("official_answer"),
                    "gold_answer": record.get("gold_answer"),
                    "evaluation_key": label.get("evaluation_key"),
                    "evaluation_value": label.get("value"),
                    "evaluation_semantics": label.get("semantics"),
                    "uses_official_answer_timestamp": label.get(
                        "uses_official_answer_timestamp"
                    ),
                    "is_current_mainline": record.get("is_current_mainline"),
                    "question_start_time": metadata.get("start_time"),
                    "question_end_time": metadata.get("end_time"),
                    "metadata_id": metadata.get("id"),
                }
            )


def write_question_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "UUID",
        "Seq ID",
        "Question",
        "Timestamp \nwith answer",
        "Type \n(binary, position, time, text)",
        "Text answer",
        "Parsable answer",
        "Question\nCategory",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = list(payload.get("records", []))
    non_position = [record for record in records if record.get("is_current_mainline")]
    counts = payload.get("counts", {})
    alignment = payload.get("metadata_alignment", {})
    lines = [
        f"# NaVQA Seq{payload.get('seq_id')} QA/Gold Preparation",
        "",
        f"- All questions: {counts.get('all', 0)}",
        f"- Current mainline non-position: {counts.get('current_mainline_non_position', 0)}",
        f"- Type counts: {json.dumps(counts.get('by_type', {}), ensure_ascii=False)}",
        (
            "- Metadata alignment: "
            f"{alignment.get('matched_count', 0)} matched, "
            f"{alignment.get('missing_count', 0)} missing"
        ),
        "",
        "## Current Mainline Questions",
        "",
        "| # | Type | Question | Gold answer |",
        "|---:|---|---|---|",
    ]
    for record in non_position:
        lines.append(
            "| {index} | {question_type} | {question} | {gold_answer} |".format(
                index=record.get("index"),
                question_type=_md_escape(record.get("question_type", "")),
                question=_md_escape(record.get("question", "")),
                gold_answer=_md_escape(record.get("gold_answer", "")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = build_parser().parse_args()
    seq_id = str(args.seq_id)
    layout = default_navqa_sequence_layout(
        seq_id=seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    out_dir = args.out_dir or layout.sequence_root
    csv_path = resolve_repo_input_path(args.csv)
    qa_unfilled_path = resolve_repo_input_path(args.qa_unfilled_json)
    position_reference_path = (
        resolve_repo_input_path(args.position_reference_json)
        if args.position_reference_json is not None
        else None
    )

    rows = load_official_rows(csv_path, seq_id=seq_id)
    if not rows:
        raise SystemExit(f"No official NaVQA rows found for seq_id={seq_id}: {csv_path}")
    qa_unfilled = load_qa_unfilled(qa_unfilled_path)
    metadata = question_metadata_aliases(qa_unfilled, seq_id=seq_id)
    position_labels = None
    if position_reference_path is not None:
        reference = json.loads(position_reference_path.read_text(encoding="utf-8-sig"))
        position_labels = load_position_labels(reference)

    records = build_records(
        rows,
        seq_id=seq_id,
        question_metadata=metadata,
        position_labels=position_labels,
    )
    alignment = metadata_alignment(rows, metadata)
    payload = build_payload(
        records=records,
        seq_id=seq_id,
        csv_path=csv_path,
        qa_unfilled_path=qa_unfilled_path,
        position_reference_path=position_reference_path,
        alignment=alignment,
    )
    non_position_records = [
        record for record in records if record["is_current_mainline"]
    ]
    non_position_alignment = metadata_alignment(
        question_rows(non_position_records),
        metadata,
    )
    non_position_payload = build_payload(
        records=non_position_records,
        seq_id=seq_id,
        csv_path=csv_path,
        qa_unfilled_path=qa_unfilled_path,
        position_reference_path=position_reference_path,
        alignment=non_position_alignment,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_qa_unfilled:
        qa_target = out_dir / "qa_unfilled.json"
        if qa_unfilled_path.resolve() != qa_target.resolve():
            shutil.copy2(qa_unfilled_path, qa_target)

    write_json(out_dir / "official_questions.json", question_rows(records))
    write_json(
        out_dir / "official_questions_non_position.json",
        question_rows(non_position_records),
    )
    write_question_csv(out_dir / "official_questions.csv", question_rows(records))
    write_question_csv(
        out_dir / "official_questions_non_position.csv",
        question_rows(non_position_records),
    )
    write_json(out_dir / "gold_answers.json", payload)
    write_json(out_dir / "gold_answers_non_position.json", non_position_payload)
    write_csv(out_dir / "gold_answers.csv", records)
    write_csv(out_dir / "gold_answers_non_position.csv", non_position_records)
    write_summary(out_dir / "qa_gold_summary.md", payload)

    print(f"seq_id={seq_id}")
    print(f"out_dir={out_dir}")
    print(f"questions={len(records)}")
    print(f"non_position={len(non_position_records)}")
    print(f"metadata_matched={alignment['matched_count']}/{len(records)}")


if __name__ == "__main__":
    main()
