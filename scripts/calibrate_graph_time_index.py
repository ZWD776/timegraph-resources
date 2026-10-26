from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FRAME_KEYS = {
    "frame_index": "timestamp_sec",
    "first_seen_frame": "first_seen_ts",
    "last_seen_frame": "last_seen_ts",
    "start_frame": "start_ts",
    "end_frame": "end_ts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite graph timestamps using a frame-time index without rebuilding "
            "the visual temporal graph."
        )
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--frame-time-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a frame key cannot be found in the frame-time index.",
    )
    return parser.parse_args()


def load_frame_time_index(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    index: dict[int, float] = {}
    for item in payload.get("frames", []):
        if not isinstance(item, dict):
            continue
        frame = item.get("graph_frame_index", item.get("frame_index"))
        timestamp = item.get("timestamp_unix", item.get("timestamp_sec"))
        if frame is None or timestamp is None:
            continue
        index[int(frame)] = float(timestamp)
    if not index:
        raise ValueError(f"No frame timestamp mappings found in {path}")
    return index


def timestamp_for_frame(
    frame_index: Any,
    frame_time_index: dict[int, float],
    *,
    strict: bool,
) -> float | None:
    try:
        frame = int(frame_index)
    except (TypeError, ValueError):
        return None
    if frame in frame_time_index:
        return frame_time_index[frame]
    if strict:
        raise KeyError(f"Missing timestamp mapping for frame {frame}")
    return None


def calibrate_value(
    value: Any,
    frame_time_index: dict[int, float],
    *,
    strict: bool,
    stats: dict[str, int],
) -> Any:
    if isinstance(value, list):
        return [
            calibrate_value(item, frame_time_index, strict=strict, stats=stats)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    calibrated = {
        key: calibrate_value(item, frame_time_index, strict=strict, stats=stats)
        for key, item in value.items()
    }
    for frame_key, ts_key in FRAME_KEYS.items():
        if frame_key not in calibrated or ts_key not in calibrated:
            continue
        timestamp = timestamp_for_frame(
            calibrated[frame_key],
            frame_time_index,
            strict=strict,
        )
        if timestamp is None:
            stats["missing"] += 1
            continue
        old_timestamp = calibrated[ts_key]
        if old_timestamp != timestamp:
            calibrated[ts_key] = timestamp
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1
    return calibrated


def main() -> None:
    args = parse_args()
    frame_time_index = load_frame_time_index(args.frame_time_index)
    payload = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    stats = {"updated": 0, "unchanged": 0, "missing": 0}
    calibrated = calibrate_value(
        payload,
        frame_time_index,
        strict=args.strict,
        stats=stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibrated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
