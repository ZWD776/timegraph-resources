from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "registry").is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / "src")]

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "registry").is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.config.config import load_domain_config
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import JSONToGraphTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an OpenEQA temporal graph with the shared graph transformer "
            "and an optional dataset-owned visual embedding cache."
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frame-time-index", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache-db",
        type=Path,
        default=None,
        help=(
            "Optional OpenEQA object-embedding cache. The graph transformer "
            "uses it only for entity association; it is not stored as QA evidence."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.perf_counter()
    input_dir = resolve_repo_input_path(args.input_dir)
    config = load_domain_config(args.config, config_dir=REPO_ROOT / "configs")
    cache_db_path = (
        resolve_repo_input_path(args.embedding_cache_db)
        if args.embedding_cache_db is not None
        else None
    )
    transformer = JSONToGraphTransformer(
        input_dir=input_dir,
        video_id=args.video_id,
        config=config,
        frame_time_index_path=resolve_repo_input_path(args.frame_time_index),
        cache_db_path=cache_db_path,
    )
    document = transformer.apply_garbage_collection(transformer.transform())
    document = document.filtered_copy(
        exclude_noise=config.export.exclude_noise,
        exclude_fragments=config.export.exclude_fragments,
        keep_fragment_labels=config.export.normalized_protected_fragment_labels,
    )
    output_path = resolve_repo_input_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    print(
        "output_summary="
        + json.dumps(
            {
                "output": str(output_path.resolve()),
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "frame_count": document.frame_count,
                "observations": len(document.observations),
                "entities": len(document.entities),
                "relation_episodes": len(document.relation_episodes),
                "state_episodes": len(document.state_episodes),
                "scene_segments": len(document.scene_segments),
                "entity_segments": len(document.entity_segments),
                "embedding_cache_db": str(cache_db_path or ""),
                "association_visual_embeddings": bool(
                    config.association.enable_visual_embeddings
                ),
                "transform_runtime_stats": transformer.get_runtime_stats(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
