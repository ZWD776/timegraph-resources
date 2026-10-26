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

from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.models.temporal_index import TemporalIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query scene segments, active episodes, and transition events from a temporal graph document."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Inspect one frame index.",
    )
    parser.add_argument(
        "--entity-id",
        type=str,
        default=None,
        help="Optional entity id for entity-level snapshot or event filtering.",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        default=None,
        help="Optional transition event type filter.",
    )
    parser.add_argument(
        "--predicate",
        type=str,
        default=None,
        help="Optional relation predicate filter when querying active relations.",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Optional scene id for scene-level delta summary queries.",
    )
    return parser


def load_document(path: Path) -> TemporalGraphDocument:
    return TemporalGraphDocument.model_validate_json(path.read_text(encoding="utf-8"))


def summarize_document(document: TemporalGraphDocument) -> dict[str, int]:
    return {
        "frame_count": document.frame_count,
        "entities": len(document.entities),
        "scene_segments": len(document.scene_segments),
        "entity_segments": len(document.entity_segments),
        "scene_base_nodes": len(document.scene_base_nodes),
        "scene_base_edges": len(document.scene_base_edges),
        "scene_delta_nodes": len(document.scene_delta_nodes),
        "scene_delta_observations": len(document.scene_delta_observations),
        "scene_delta_edges": len(document.scene_delta_edges),
        "state_episodes": len(document.state_episodes),
        "relation_episodes": len(document.relation_episodes),
        "transition_events": len(document.transition_events),
    }


def main() -> None:
    args = build_parser().parse_args()
    document = load_document(args.input_json)
    index = TemporalIndex.build(document)

    payload: dict[str, object] = {
        "video_id": document.video_id,
        "summary": summarize_document(document),
    }

    if args.frame_index is not None:
        payload["frame_query"] = {
            "frame_index": args.frame_index,
            "frame_rank": index.frame_rank_for_index(args.frame_index),
            "scene_segment": _dump_optional(index.scene_at_frame(args.frame_index)),
            "scene_delta_summary": _dump_optional(
                index.scene_delta_summary_at_frame(args.frame_index)
            ),
            "scene_base_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_base_nodes_at_frame(args.frame_index)
            ],
            "scene_delta_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_delta_nodes_at_frame(args.frame_index)
            ],
            "scene_delta_observations": [
                observation.model_dump(mode="json")
                for observation in index.scene_delta_observations_at_frame(args.frame_index)
            ],
            "active_entity_ids": [
                entity.entity_id
                for entity in index.active_entities(args.frame_index)
            ],
            "active_states": [
                episode.model_dump(mode="json")
                for episode in index.active_states(
                    args.frame_index,
                    entity_id=args.entity_id,
                )
            ],
            "active_relations": [
                episode.model_dump(mode="json")
                for episode in index.active_relations(
                    args.frame_index,
                    entity_id=args.entity_id,
                    predicate=args.predicate,
                )
            ],
            "transition_events": [
                event.model_dump(mode="json")
                for event in index.transition_events_at_frame(
                    frame_index=args.frame_index,
                    entity_id=args.entity_id,
                    event_type=args.event_type,
                )
            ],
        }

    if args.entity_id is not None:
        frame_index = args.frame_index
        if frame_index is None:
            frame_index = index.entities_by_id[args.entity_id].first_seen_frame
        snapshot = index.entity_snapshot(args.entity_id, frame_index=frame_index)
        payload["entity_query"] = {
            "entity_id": args.entity_id,
            "frame_index": frame_index,
            "frame_rank": snapshot.frame_rank,
            "scene_segment": _dump_optional(snapshot.scene_segment),
            "scene_base_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_base_nodes_by_entity.get(args.entity_id, [])
            ],
            "scene_delta_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_delta_nodes_by_entity.get(args.entity_id, [])
            ],
            "scene_delta_observations": [
                observation.model_dump(mode="json")
                for observation in index.scene_delta_observations_by_entity.get(
                    args.entity_id, []
                )
            ],
            "entity": snapshot.entity.model_dump(mode="json"),
            "entity_segments": [
                segment.model_dump(mode="json")
                for segment in snapshot.entity_segments
            ],
            "active_states": [
                episode.model_dump(mode="json")
                for episode in snapshot.active_states
            ],
            "active_relations": [
                episode.model_dump(mode="json")
                for episode in snapshot.active_relations
            ],
            "transition_events": [
                event.model_dump(mode="json")
                for event in index.transition_events_for_entity(
                    args.entity_id,
                    event_type=args.event_type,
                )
            ],
        }

    if args.scene_id is not None:
        scene = index.scene_by_id(args.scene_id)
        if scene is None:
            raise SystemExit(f"Unknown scene_id: {args.scene_id}")
        payload["scene_query"] = {
            "scene_id": args.scene_id,
            "scene_segment": scene.model_dump(mode="json"),
            "scene_delta_summary": asdict(index.scene_delta_summary(args.scene_id)),
            "scene_base_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_base_nodes_for_scene(args.scene_id)
            ],
            "scene_base_edges": [
                edge.model_dump(mode="json")
                for edge in index.scene_base_edges_for_scene(args.scene_id)
            ],
            "scene_delta_nodes": [
                node.model_dump(mode="json")
                for node in index.scene_delta_nodes_for_scene(args.scene_id)
            ],
            "scene_delta_observations": [
                observation.model_dump(mode="json")
                for observation in index.scene_delta_observations_for_scene(args.scene_id)
            ],
            "scene_delta_edges": [
                edge.model_dump(mode="json")
                for edge in index.scene_delta_edges_for_scene(args.scene_id)
            ],
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dump_optional(value: object | None) -> object | None:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value.model_dump(mode="json")


if __name__ == "__main__":
    main()
