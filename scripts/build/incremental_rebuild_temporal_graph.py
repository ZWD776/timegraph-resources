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

from vlkg.core.config.config import load_domain_config
from vlkg.core.temporal.incremental_rebuild import OfflineIncrementalRebuilder
from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.temporal.pathing import default_cache_db_path, resolve_repo_input_path
from vlkg.core.transformers import JSONToGraphTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally rebuild a temporal graph document from an updated frame/json directory. "
            "Current implementation targets append-only frame growth."
        )
    )
    parser.add_argument(
        "existing_graph_json",
        type=Path,
        help="Path to the existing temporal graph JSON document.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the updated frames and json outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input_dir>/temporal_graph.incremental.json.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write rebuild summary JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("default.yaml"),
        help="Domain profile YAML relative to configs/ or an absolute path.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional video id override. Defaults to the existing document video id.",
    )
    parser.add_argument(
        "--lookback-frames",
        type=int,
        default=1,
        help="How many already-built frames to include before the first appended frame.",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Exclude entities marked as noise from the refreshed document before rebuild.",
    )
    parser.add_argument(
        "--exclude-fragments",
        action="store_true",
        help="Exclude entities marked as fragments from the refreshed document before rebuild.",
    )
    parser.add_argument(
        "--min-lifespan",
        type=int,
        default=None,
        help="Override garbage-collection min lifespan threshold.",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=default_cache_db_path(),
        help="Embedding cache database path.",
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
    existing_graph_json = resolve_repo_input_path(args.existing_graph_json)
    input_dir = resolve_repo_input_path(args.input_dir)
    existing_document = load_document(existing_graph_json)
    config = load_domain_config(args.config, config_dir=REPO_ROOT / "configs")
    transformer = JSONToGraphTransformer(
        input_dir=input_dir,
        video_id=args.video_id or existing_document.video_id,
        config=config,
        cache_db_path=args.cache_db.resolve(),
    )
    refreshed_document = transformer.transform()
    if args.min_lifespan is not None:
        refreshed_document = transformer.apply_garbage_collection(
            refreshed_document,
            min_lifespan=args.min_lifespan,
        )
    else:
        refreshed_document = transformer.apply_garbage_collection(refreshed_document)

    exclude_noise = args.exclude_noise or config.export.exclude_noise
    exclude_fragments = args.exclude_fragments or config.export.exclude_fragments
    refreshed_document = refreshed_document.filtered_copy(
        exclude_noise=exclude_noise,
        exclude_fragments=exclude_fragments,
        keep_fragment_labels=config.export.normalized_protected_fragment_labels,
    )

    rebuilder = OfflineIncrementalRebuilder(
        existing_document=existing_document,
        refreshed_document=refreshed_document,
    )
    rebuild_result = rebuilder.rebuild(lookback_frame_count=args.lookback_frames)

    output_path = args.output or (input_dir / "temporal_graph.incremental.json")
    output_path.write_text(
        rebuild_result.updated_document.model_dump_json(indent=2),
        encoding="utf-8",
    )

    payload = {
        "video_id": rebuild_result.plan.video_id,
        "append_only": rebuild_result.plan.append_only,
        "is_noop": rebuild_result.plan.is_noop,
        "execution_scope": rebuild_result.plan.execution_scope,
        "plan": _dump(rebuild_result.plan),
        "existing_document_summary": summarize_document(existing_document),
        "refreshed_document_summary": summarize_document(rebuild_result.refreshed_document),
        "updated_document_summary": summarize_document(rebuild_result.updated_document),
        "execution": (
            {
                "document_unchanged": rebuild_result.execution_result.document_unchanged,
                "removed_counts": rebuild_result.execution_result.removed_counts,
                "inserted_counts": rebuild_result.execution_result.inserted_counts,
            }
            if rebuild_result.execution_result is not None
            else None
        ),
        "output_path": str(output_path.resolve()),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.summary_output is not None:
        args.summary_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
