from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.position_reference import (
    build_position_reference,
    build_position_windows_from_pose_samples,
    coda_preprocessed_pose_samples,
    load_dense_poses,
    load_navqa_rows,
    load_position_windows_from_captions,
    load_question_metadata,
    load_timestamps,
    write_completed_navqa_csv,
    write_position_reference_csv,
)
from vlkg.datasets.navqa.runtime.paths import default_navqa_sequence_layout
from vlkg.core.temporal.pathing import resolve_repo_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an independent NaVQA position reference file. This materializes "
            "official position labels for evaluation only; it does not modify or "
            "augment the graph."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Official NaVQA data.csv.")
    parser.add_argument("--qa-unfilled-json", type=Path, required=True)
    parser.add_argument("--seq-id", default="0")
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path(".scratch/navqa"))
    parser.add_argument(
        "--dense-pose",
        type=Path,
        default=None,
        help="CODa poses/dense/<seq>.txt. Required unless --captions-json is provided.",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="CODa timestamps/<seq>.txt. Required with --dense-pose.",
    )
    parser.add_argument(
        "--captions-json",
        type=Path,
        default=None,
        help=(
            "Optional existing ReMEmbR captions JSON. If provided, positions are "
            "read from it directly instead of reconstructing caption windows from dense poses."
        ),
    )
    parser.add_argument("--seconds-per-caption", type=float, default=3.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--completed-csv",
        type=Path,
        default=None,
        help="Optional copy of data.csv with a completed-position-answer column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = default_navqa_sequence_layout(
        seq_id=args.seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    out_json = args.out_json or (layout.sequence_root / "position_reference.json")
    out_csv = args.out_csv or (layout.sequence_root / "position_reference.csv")
    csv_path = resolve_repo_input_path(args.csv)
    qa_path = resolve_repo_input_path(args.qa_unfilled_json)
    navqa_rows = load_navqa_rows(csv_path, seq_id=str(args.seq_id))
    question_metadata = load_question_metadata(qa_path)

    if args.captions_json is not None:
        windows = load_position_windows_from_captions(resolve_repo_input_path(args.captions_json))
        position_source = str(resolve_repo_input_path(args.captions_json))
    else:
        if args.dense_pose is None or args.timestamps is None:
            raise SystemExit("--dense-pose and --timestamps are required without --captions-json")
        dense_pose_path = resolve_repo_input_path(args.dense_pose)
        timestamps_path = resolve_repo_input_path(args.timestamps)
        dense_poses = load_dense_poses(dense_pose_path)
        lidar_timestamps = load_timestamps(timestamps_path)
        samples = coda_preprocessed_pose_samples(dense_poses, lidar_timestamps)
        windows = build_position_windows_from_pose_samples(
            samples,
            seconds_per_caption=args.seconds_per_caption,
        )
        position_source = str(dense_pose_path)

    reference = build_position_reference(
        navqa_rows=navqa_rows,
        question_metadata=question_metadata,
        position_windows=windows,
        seq_id=str(args.seq_id),
    )
    reference["source"] = {
        "csv": str(csv_path),
        "qa_unfilled_json": str(qa_path),
        "position_source": position_source,
        "seconds_per_caption": args.seconds_per_caption,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_position_reference_csv(out_csv, reference)
    if args.completed_csv is not None:
        write_completed_navqa_csv(
            source_csv=csv_path,
            out_csv=args.completed_csv,
            reference=reference,
        )

    print(f"position_answers={reference['position_answer_count']}")
    print(f"skipped={reference['skipped_count']}")
    print(f"out_json={out_json}")
    print(f"out_csv={out_csv}")
    if args.completed_csv is not None:
        print(f"completed_csv={args.completed_csv}")


if __name__ == "__main__":
    main()
