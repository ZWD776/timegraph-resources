from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.temporal.delta_update import DeltaUpdatePlanner
from vlkg.core.models.temporal_graph import TemporalGraphDocument


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one delta update plan/window from a temporal graph document."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="Inclusive start frame index for the delta window.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Inclusive end frame index for the delta window.",
    )
    return parser


def load_document(path: Path) -> TemporalGraphDocument:
    return TemporalGraphDocument.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    document = load_document(args.input_json)
    planner = DeltaUpdatePlanner.build(document)
    window_slice = planner.materialize_frame_window(
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    payload = {
        "video_id": document.video_id,
        "plan": _dump(window_slice.plan),
        "slice_summary": {
            "scene_segments": len(window_slice.scene_segments),
            "scene_base_nodes": len(window_slice.scene_base_nodes),
            "scene_base_edges": len(window_slice.scene_base_edges),
            "scene_delta_nodes": len(window_slice.scene_delta_nodes),
            "scene_delta_observations": len(window_slice.scene_delta_observations),
            "scene_delta_edges": len(window_slice.scene_delta_edges),
            "entities": len(window_slice.entities),
            "entity_segments": len(window_slice.entity_segments),
            "state_episodes": len(window_slice.state_episodes),
            "relation_episodes": len(window_slice.relation_episodes),
            "transition_events": len(window_slice.transition_events),
        },
        "slice": _dump(window_slice),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dump(value):
    if is_dataclass(value):
        return {
            key: _dump(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _dump(item)
            for key, item in value.items()
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


if __name__ == "__main__":
    main()
