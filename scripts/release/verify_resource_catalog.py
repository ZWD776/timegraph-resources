"""Verify the released resource catalogue without models, APIs, or dataset files."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("navqa", "nextqa", "openeqa")
EXPECTED_FRAMES = {"navqa": 9889, "nextqa": 23381, "openeqa": 31742}
EXPECTED_QA = {"navqa": 210, "nextqa": 4996, "openeqa": 1636}
EXPECTED_UNITS = {"navqa": 7, "nextqa": 570, "openeqa": 152}


@lru_cache(maxsize=16)
def resolved_root(root: Path) -> Path:
    return root.resolve()


def safe_reference(root: Path, value: str) -> Path:
    """Resolve only relative catalogue references, including on non-Windows hosts."""
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if not value or value == "." or posix.is_absolute() or windows.drive or ".." in posix.parts or "\\" in value or ":" in value:
        raise ValueError(f"unsafe relative resource reference: {value!r}")
    # Validate lexically without statting 65,012 external references. No dataset
    # is opened by the default check; optional data checks only test existence.
    base = resolved_root(root)
    target = base / value
    if not target.is_relative_to(base):
        raise ValueError("resource reference escapes its selected root")
    return target


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value: str, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}")
    return result


def verify_catalog(catalog: Path, data_root: Path | None = None) -> dict[str, Any]:
    catalog = catalog.resolve()
    ledger = load_json(catalog / "catalog_files.json")
    entries = ledger["files"]
    names: set[str] = set()
    for item in entries:
        name = item["path"]
        if name in names:
            raise ValueError(f"duplicate catalogue ledger entry: {name}")
        names.add(name)
        path = safe_reference(catalog, name)
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"catalogue checksum mismatch: {name}")
        if len(data) >= 100 * 1024**2:
            raise ValueError(f"catalogue file exceeds ordinary Git file limit: {name}")
    actual = {p.relative_to(catalog).as_posix() for p in catalog.rglob("*") if p.is_file()}
    if actual != names | {"catalog_files.json"}:
        raise ValueError("catalogue contains missing or unregistered files")

    units = load_json(catalog / "unit_catalog.json")["units"]
    unit_lookup = {(r["dataset"], r["shard"], str(r["unit_id"])): r for r in units}
    if len(unit_lookup) != len(units):
        raise ValueError("duplicate dataset/shard/unit identity")
    if dict(Counter(r["dataset"] for r in units)) != EXPECTED_UNITS:
        raise ValueError("unit counts differ from the selected release")
    for unit in units:
        for key in ("current_hsgg_reference_root", "canonical_path", "temporal_path"):
            safe_reference(catalog, unit[key])

    frame_counts: dict[str, int] = {}
    qa_counts: dict[str, int] = {}
    hsgg_counts: dict[str, int] = {}
    per_unit: Counter[tuple[str, str, str]] = Counter()
    real_data_missing: list[dict[str, str]] = []
    open_missing_ordinal = nav_selected = 0
    for dataset in DATASETS:
        frame_keys: set[tuple[str, str, str]] = set()
        hsgg_count = 0
        with (catalog / f"{dataset}_frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["shard"], row["unit_id"])
                identity = (row["shard"], row["unit_id"], row["hsgg_record_key"])
                if row["dataset"] != dataset or key not in unit_lookup or identity in frame_keys:
                    raise ValueError(f"invalid or duplicate frame identity in {dataset}")
                frame_keys.add(identity)
                per_unit[key] += 1
                image = safe_reference(data_root or catalog, row["image_path"])
                available = row["hsgg_available"] == "True"
                hsgg_count += int(available)
                if row["hsgg_path"]:
                    hsgg = safe_reference(data_root or catalog, row["hsgg_path"])
                else:
                    hsgg = None
                if available and hsgg is None:
                    raise ValueError("completed HSGG frame has no resource reference")
                if data_root is not None:
                    if not image.is_file():
                        real_data_missing.append({"kind": "image", "path": row["image_path"]})
                    if available and hsgg is not None and not hsgg.is_file():
                        real_data_missing.append({"kind": "hsgg", "path": row["hsgg_path"]})
                if dataset == "navqa":
                    number(row["timestamp_unix"], name="NaVQA Unix timestamp")
                    if number(row["elapsed_seconds"], name="NaVQA elapsed seconds") < 0:
                        raise ValueError("negative elapsed seconds")
                elif dataset == "nextqa":
                    if number(row["timestamp_seconds"], name="NextQA nominal seconds") < 0:
                        raise ValueError("negative nominal seconds")
                else:
                    if row["timestamp_is_real_time"] != "False" or row["timestamp_seconds"] or row["timestamp_unix"]:
                        raise ValueError("OpenEQA ordinal values must not be labelled as real seconds")
                    if row["ordinal_time"]:
                        number(row["ordinal_time"], name="OpenEQA ordering value")
                    else:
                        open_missing_ordinal += 1
        frame_counts[dataset] = len(frame_keys)
        hsgg_counts[dataset] = hsgg_count
        if len(frame_keys) != EXPECTED_FRAMES[dataset]:
            raise ValueError(f"unexpected {dataset} frame count")

        question_ids: set[str] = set()
        with (catalog / f"{dataset}_qa_ids.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if {"question", "answer", "answer_choices", "gold_choice_index"} & set(reader.fieldnames or []):
                raise ValueError("QA mapping unexpectedly includes question or answer content")
            for row in reader:
                key = (row["dataset"], row["shard"], row["unit_id"])
                qid = row["question_id"]
                if key not in unit_lookup or qid in question_ids or row["dataset"] != dataset:
                    raise ValueError(f"invalid or duplicate {dataset} QA mapping")
                question_ids.add(qid)
                if row["question_text_included"] != "False" or row["answer_included"] != "False":
                    raise ValueError("QA identifier export must not include answers")
                if dataset == "navqa":
                    start = number(row["start_time_unix"], name="QA history start")
                    end = number(row["end_time_unix"], name="QA history end")
                    if start > end:
                        raise ValueError("reversed QA history window")
                    nav_selected += int(row["selected_non_position"] == "True")
        qa_counts[dataset] = len(question_ids)
        if len(question_ids) != EXPECTED_QA[dataset]:
            raise ValueError(f"unexpected {dataset} QA count")
    for key, unit in unit_lookup.items():
        if per_unit[key] != unit["sampled_image_count"]:
            raise ValueError(f"frame/unit count mismatch: {key}")
    if nav_selected != 139:
        raise ValueError("NaVQA non-position subset differs from the selected release")
    summary = load_json(catalog / "export_summary.json")
    if summary["frame_counts"] != frame_counts or summary["qa_counts"] != qa_counts:
        raise ValueError("catalogue summary disagrees with CSV content")
    if summary["current_completed_hsgg_counts"] != hsgg_counts:
        raise ValueError("completed HSGG counts disagree")
    if real_data_missing:
        raise ValueError(f"selected external data root lacks {len(real_data_missing)} referenced files; first: {real_data_missing[0]}")
    return {
        "status": "passed", "unit_count": len(units), "frame_counts": frame_counts,
        "qa_counts": qa_counts, "completed_hsgg_counts": hsgg_counts,
        "navqa_non_position_questions": nav_selected,
        "openeqa_frames_without_legacy_ordinal_mapping": open_missing_ordinal,
        "catalog_checksums_verified": True, "external_data_checked": data_root is not None,
        "netdisk_contents_verified": False, "models_or_apis_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog")
    parser.add_argument("--data-root", type=Path, help="Optional explicit external root containing datasets/. Read-only existence checks; never inferred from the parent workspace.")
    args = parser.parse_args()
    try:
        result = verify_catalog(args.catalog, args.data_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
