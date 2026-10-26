from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.runtime.paths import default_navqa_sequence_layout


CODA_IMAGE_RE = re.compile(
    r"(?:(?P<sampled_index>\d+)_)?(?:(?P<graph_frame>\d+)_)?2d_rect_cam"
    r"(?P<cam>\d+)_(?P<seq>\d+)_(?P<source_frame>\d+)\.jpe?g$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact frame->official timestamp index for NaVQA/CODa."
    )
    parser.add_argument("--seq-id", default="0")
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path(".scratch/navqa"))
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--qa-unfilled",
        type=Path,
        default=None,
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def load_timestamps(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values.append(float(stripped.split()[0]))
    if not values:
        raise ValueError(f"No timestamps found in {path}")
    return values


def parse_image(path: Path) -> dict[str, Any]:
    match = CODA_IMAGE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unsupported CODa image name: {path.name}")
    source_frame = int(match.group("source_frame"))
    graph_frame = match.group("graph_frame")
    sampled_index = match.group("sampled_index")
    return {
        "sampled_index": int(sampled_index) if sampled_index is not None else None,
        "graph_frame_index": int(graph_frame) if graph_frame is not None else source_frame,
        "cam": int(match.group("cam")),
        "seq": int(match.group("seq")),
        "source_frame": source_frame,
        "image_name": path.name,
        "image_path": str(path),
    }


def iso_from_unix(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")


def build_frame_rows(image_dir: Path, timestamps: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        row = parse_image(image_path)
        source_frame = int(row["source_frame"])
        if source_frame >= len(timestamps):
            raise ValueError(
                f"Frame {source_frame} from {image_path.name} is outside timestamps length {len(timestamps)}"
            )
        timestamp = float(timestamps[source_frame])
        row.update(
            {
                "timestamp_unix": timestamp,
                "timestamp_iso": iso_from_unix(timestamp),
                "elapsed_sec_from_sequence_start": timestamp - timestamps[0],
                "elapsed_min_from_sequence_start": (timestamp - timestamps[0]) / 60.0,
            }
        )
        rows.append(row)
    return rows


def load_question_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("data", []):
        start_time = float(item["start_time"])
        end_time = float(item["end_time"])
        rows.append(
            {
                "id": item.get("id"),
                "length_category": item.get("length_category"),
                "length_sec": float(item.get("length", end_time - start_time)),
                "start_time_unix": start_time,
                "end_time_unix": end_time,
                "start_time_iso": iso_from_unix(start_time),
                "end_time_iso": iso_from_unix(end_time),
                "qa_start_filename": item.get("file_info", {}).get("qa_start_filename"),
                "qa_end_filename": item.get("file_info", {}).get("qa_end_filename"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    layout = default_navqa_sequence_layout(
        seq_id=args.seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    image_dir = args.image_dir or layout.images_dir
    timestamps_path = args.timestamps or layout.timestamps_path
    qa_unfilled_path = args.qa_unfilled or layout.qa_unfilled_json
    out_json = args.out_json or layout.frame_time_index_json
    out_csv = args.out_csv or layout.frame_time_index_csv

    timestamps = load_timestamps(timestamps_path)
    frame_rows = build_frame_rows(image_dir, timestamps)
    question_rows = load_question_rows(qa_unfilled_path)

    payload = {
        "sequence_id": str(args.seq_id),
        "camera": str(args.cam),
        "stride": args.stride,
        "source": {
            "timestamps": str(timestamps_path),
            "image_dir": str(image_dir),
            "qa_unfilled": str(qa_unfilled_path),
        },
        "timestamp_count": len(timestamps),
        "sequence_start_unix": timestamps[0],
        "sequence_start_iso": iso_from_unix(timestamps[0]),
        "sequence_end_unix": timestamps[-1],
        "sequence_end_iso": iso_from_unix(timestamps[-1]),
        "sequence_duration_sec": timestamps[-1] - timestamps[0],
        "frames": frame_rows,
        "questions": question_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_csv, frame_rows)

    print(f"timestamps: {len(timestamps)}")
    print(f"frames: {len(frame_rows)}")
    print(f"questions: {len(question_rows)}")
    print(f"sequence_duration_sec: {payload['sequence_duration_sec']:.3f}")
    print(f"out_json: {out_json}")
    print(f"out_csv: {out_csv}")


if __name__ == "__main__":
    main()
