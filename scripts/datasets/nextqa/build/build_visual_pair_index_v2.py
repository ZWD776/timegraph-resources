from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json
from vlkg.datasets.nextqa.runtime.run_snapshot import (
    collect_imported_repo_sources,
    fingerprint_file,
    persist_drift_report,
    persist_run_config,
    persist_source_snapshot,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import write_faiss_shard
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import (
    build_strict_before_after_candidates,
    reconstruct_faiss_vectors,
)
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild strict before/after shards and reference immutable visual v1 shards.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-index-manifest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_base_manifest(args, run_root):
    return recorded_or_selected(args.base_index_manifest, resume=args.resume,
                                run_config=run_root/'run_config.json', field='base_visual_index_manifest',
                                selected=lambda: asset_file('nextqa','indexes','manifest'))


def validate_base(base):
    # V2 retains all five non-pair shard families. Its old pair shard is rebuilt,
    # never used as an input to the strict pair construction algorithm.
    if base.get('schema_version') not in {'nextqa_visual_index_manifest_v1','nextqa_visual_index_manifest_v2'}:
        raise ValueError('base visual index must be a supported audited manifest')
    if len(base.get('videos', [])) != 570:
        raise ValueError('base visual index must contain 570 videos')
    for video in base['videos']:
        if set(video['shards']) != {'object','frame','tracklet','temporal_window','event_boundary','before_after_pair'}:
            raise ValueError('base video does not contain the six required shard families')


def _load(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root("visual_indexes", args.run_id, resume=args.resume)
    base_path = resolve_base_manifest(args, run_root)
    base = _load(base_path)
    validate_base(base)
    contract = {
        "dataset": "NExT-QA", "split": "val", "stage": "strict_before_after_pair_index_v2",
        "base_visual_index_manifest": fingerprint_file(base_path),
        "reused_candidate_kinds": ["object", "frame", "tracklet", "temporal_window", "event_boundary"],
        "rebuilt_candidate_kinds": ["before_after_pair"],
        "strict_before_operator": "frame_ts < boundary_ts",
        "strict_after_operator": "frame_ts > boundary_ts",
        "same_frame_fallback_allowed": False,
        "uses_questions": False, "uses_gold_answer": False, "calls_api": False,
    }
    persist_run_config(
        run_root / "run_config.json", contract=contract, resume=args.resume,
        schema_version="nextqa_visual_pair_index_run_config_v2",
    )
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json", source_paths=sources, resume=args.resume
    )
    videos: list[dict[str, Any]] = []
    totals = Counter()
    counter_totals = Counter()
    for base_video in base["videos"]:
        video_id = str(base_video["video_id"])
        graph_path = (REPO_ROOT / Path(str(base_video["temporal_graph_path"]))).resolve()
        graph = _load(graph_path)
        frame_shard = base_video["shards"]["frame"]
        frame_metadata = _load(Path(str(frame_shard["metadata_path"])))
        frame_vectors = reconstruct_faiss_vectors(frame_shard["index_path"])
        pair_metadata, pair_vectors, counters = build_strict_before_after_candidates(
            graph, frame_metadata, frame_vectors
        )
        pair_root = run_root / "videos" / video_id
        pair_shard = write_faiss_shard(
            pair_root, kind="before_after_pair", metadata=pair_metadata,
            vectors=pair_vectors, expected_dimension=int(base["embedding_dimension"]),
        )
        pair_shard["provenance"] = "rebuilt_strict_v2"
        shards: dict[str, Any] = {}
        for kind, shard in base_video["shards"].items():
            if kind == "before_after_pair":
                shards[kind] = pair_shard
            else:
                shards[kind] = {**shard, "provenance": "reused_from_visual_v1"}
            totals[kind] += int(shards[kind]["count"])
        counter_totals.update(counters)
        videos.append(
            {
                **{key: value for key, value in base_video.items() if key != "shards"},
                "schema_version": "nextqa_visual_video_index_manifest_v2",
                "shards": shards,
                "strict_pair_audit": counters,
                "status": "completed",
            }
        )
    checks = {
        "video_count_570": len(videos) == 570,
        "same_frame_pair_count_zero": counter_totals["same_frame_pair_count"] == 0,
        "strict_order_violation_count_zero": counter_totals["strict_order_violation_count"] == 0,
        "all_non_pair_shards_reused": all(
            shard.get("provenance") == "reused_from_visual_v1"
            for video in videos for kind, shard in video["shards"].items()
            if kind != "before_after_pair"
        ),
        "all_pair_shards_rebuilt": all(
            video["shards"]["before_after_pair"].get("provenance") == "rebuilt_strict_v2"
            for video in videos
        ),
    }
    manifest = {
        "schema_version": "nextqa_visual_index_manifest_v2",
        "run_id": args.run_id,
        "profile_name": "nextqa_visual_runtime_v2",
        "embedding_model": base["embedding_model"],
        "embedding_dimension": base["embedding_dimension"],
        "base_visual_index_manifest_path": str(base_path),
        "base_visual_index_manifest_sha256": contract["base_visual_index_manifest"]["sha256"],
        "graph_root": base["graph_root"],
        "candidate_kind_totals": dict(sorted(totals.items())),
        "videos": videos,
        "uses_questions": False,
        "uses_gold_answer": False,
        "qa_receives_images": False,
    }
    audit = {
        "schema_version": "nextqa_visual_pair_index_audit_v2",
        "run_id": args.run_id,
        "completed_video_count": len(videos),
        "candidate_kind_totals": dict(sorted(totals.items())),
        "strict_pair_counters": dict(sorted(counter_totals.items())),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "uses_questions": False, "uses_gold_answer": False, "calls_api": False,
        "status": "passed" if all(checks.values()) else "failed",
    }
    atomic_write_json(run_root / "visual_index_manifest.json", manifest)
    atomic_write_json(run_root / "visual_index_audit.json", audit)
    drift = persist_drift_report(
        run_root / "source_drift_report.json", snapshot=snapshot, current_source_paths=sources
    )
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "visual_index_audit.json", audit)
    atomic_write_json(
        run_root / "run_status.json",
        {"status": audit["status"], "completed_video_count": len(videos), "calls_api": False,
         "code_drift_detected": drift["code_drift_detected"]},
    )
    if audit["status"] != "passed":
        raise SystemExit(f"strict pair index audit failed: {audit['failed_checks']}")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
