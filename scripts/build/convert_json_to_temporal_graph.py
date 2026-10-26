from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.config.config import load_domain_config
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import JSONToGraphTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert frame-wise VLM JSON outputs into a temporal graph document."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the json/ folder or the JSON files directly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input_dir>/temporal_graph.json.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional video id override. Defaults to input directory name.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("default.yaml"),
        help="Domain profile YAML relative to configs/ or an absolute path.",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Exclude entities marked as noise from exported JSON.",
    )
    parser.add_argument(
        "--exclude-fragments",
        action="store_true",
        help="Exclude entities marked as fragments from exported JSON.",
    )
    parser.add_argument(
        "--min-lifespan",
        type=int,
        default=None,
        help="Override garbage-collection min lifespan threshold.",
    )
    parser.add_argument(
        "--frame-time-index",
        type=Path,
        default=None,
        help=(
            "Optional frame-time index JSON. When provided, frame_index values "
            "are mapped to official timestamps during graph construction."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.add_argument('--embedding-cache-db', type=Path, default=None,
                        help='Explicit cache owned by the selected dataset or job.')
    args = parser.parse_args()
    started_at = time.perf_counter()

    input_dir = resolve_repo_input_path(args.input_dir)
    config = load_domain_config(args.config, config_dir=REPO_ROOT / "configs")
    print(f"input_dir={input_dir}", flush=True)
    print(f"config={args.config}", flush=True)
    transformer = JSONToGraphTransformer(
        input_dir=input_dir,
        video_id=args.video_id,
        config=config,
        cache_db_path=resolve_repo_input_path(args.embedding_cache_db) if args.embedding_cache_db else None,
        frame_time_index_path=(
            resolve_repo_input_path(args.frame_time_index)
            if args.frame_time_index is not None
            else None
        ),
    )
    document = transformer.transform()
    print(
        "transform_runtime_stats="
        + json.dumps(transformer.get_runtime_stats(), ensure_ascii=False),
        flush=True,
    )
    if args.min_lifespan is not None:
        document = transformer.apply_garbage_collection(
            document,
            min_lifespan=args.min_lifespan,
        )
    else:
        document = transformer.apply_garbage_collection(document)

    exclude_noise = args.exclude_noise or config.export.exclude_noise
    exclude_fragments = args.exclude_fragments or config.export.exclude_fragments
    document = document.filtered_copy(
        exclude_noise=exclude_noise,
        exclude_fragments=exclude_fragments,
        keep_fragment_labels=config.export.normalized_protected_fragment_labels,
    )

    output_path = args.output or (input_dir / "temporal_graph.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing_output={output_path}", flush=True)
    output_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(
        "output_summary="
        + json.dumps(
            {
                "output": str(output_path.resolve()),
                "bytes": output_path.stat().st_size,
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "frame_count": document.frame_count,
                "observations": len(document.observations),
                "entities": len(document.entities),
                "relation_episodes": len(document.relation_episodes),
                "state_episodes": len(document.state_episodes),
                "scene_segments": len(document.scene_segments),
                "entity_segments": len(document.entity_segments),
                "transition_events": len(document.transition_events),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
