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
        description="Execute one incremental window update and inspect the merged result."
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
        help="Inclusive start frame index for the incremental window.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Inclusive end frame index for the incremental window.",
    )
    return parser


def load_document(path: Path) -> TemporalGraphDocument:
    return TemporalGraphDocument.model_validate_json(path.read_text(encoding="utf-8"))


def summarize_document(document: TemporalGraphDocument) -> dict[str, int]:
    return {
        "frame_count": document.frame_count,
        "observations": len(document.observations),
        "entities": len(document.entities),
        "relation_episodes": len(document.relation_episodes),
        "state_episodes": len(document.state_episodes),
        "scene_segments": len(document.scene_segments),
        "entity_segments": len(document.entity_segments),
        "scene_base_nodes": len(document.scene_base_nodes),
        "scene_base_edges": len(document.scene_base_edges),
        "scene_delta_nodes": len(document.scene_delta_nodes),
        "scene_delta_observations": len(document.scene_delta_observations),
        "scene_delta_edges": len(document.scene_delta_edges),
        "transition_events": len(document.transition_events),
    }


def main() -> None:
    args = build_parser().parse_args()
    document = load_document(args.input_json)
    planner = DeltaUpdatePlanner.build(document)
    execution_result = planner.execute_window_update(
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    payload = {
        "video_id": document.video_id,
        "plan": _dump(execution_result.plan),
        "base_document_summary": summarize_document(document),
        "window_document_summary": summarize_document(execution_result.window_document.document),
        "updated_document_summary": summarize_document(execution_result.updated_document),
        "removed_counts": execution_result.removed_counts,
        "inserted_counts": execution_result.inserted_counts,
        "document_unchanged": execution_result.document_unchanged,
        "execution_result": _dump(execution_result),
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
