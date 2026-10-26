from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from collections import Counter, defaultdict
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
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.handlers import HANDLERS, dispatch_handler
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import (
    VisualShardStore,
    retrieve_role_candidates,
)
from vlkg.core.indexing.vision import VisionFeatureExtractor
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected


DEFAULT_CONFIG = REPO_ROOT / "configs/datasets/nextqa/visual_runtime.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build path-free graph plus visual evidence using independent v2 handlers.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--route-records", type=Path, required=True)
    parser.add_argument("--frozen-graph-evidence", type=Path, required=True)
    parser.add_argument(
        "--visual-index-manifest", type=Path, default=None,
        help="Default: selected NExT-QA registry index; resume keeps the recorded manifest.",
    )
    parser.add_argument("--pilot-manifest", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_index_manifest(args: argparse.Namespace, run_root: Path) -> Path:
    if args.visual_index_manifest is not None:
        return args.visual_index_manifest.resolve()
    if args.resume:
        saved = json.loads((run_root / 'run_config.json').read_text(encoding='utf-8'))
        path = saved['contract']['visual_index_manifest']['path']
        if not isinstance(path, str) or not path.strip():
            raise ValueError('resume contract has no visual index manifest path')
        return (REPO_ROOT / path).resolve()
    return asset_file('nextqa', 'indexes', 'manifest', REPO_ROOT)


def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _select_pilot(rows: list[dict[str, Any]], pilot_path: Path | None) -> list[dict[str, Any]]:
    if pilot_path is None:
        return rows
    payload = json.loads(pilot_path.resolve().read_text(encoding="utf-8"))
    video_ids = {str(item["official_video_id"]) for item in payload.get("videos", [])}
    if len(video_ids) != 5:
        raise ValueError("pilot manifest must contain five videos")
    return [row for row in rows if str(row["video_id"]) in video_ids]


def _strict_pair_is_valid(hit: dict[str, Any]) -> bool:
    if hit.get("candidate_kind") != "before_after_pair":
        return True
    metadata = hit.get("metadata") or {}
    if metadata.get("strict_temporal_pair") is not True:
        return False
    boundary = float(metadata["boundary_ts"])
    before = metadata.get("before_ts")
    after = metadata.get("after_ts")
    if before is not None and not float(before) < boundary:
        return False
    if after is not None and not float(after) > boundary:
        return False
    if before is not None and after is not None:
        if str(metadata.get("before_frame_name")) == str(metadata.get("after_frame_name")):
            return False
        if not float(before) < boundary < float(after):
            return False
    return True


def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root("visual_evidence", args.run_id, resume=args.resume)
    config_path = recorded_or_selected(args.config, resume=args.resume, run_config=run_root / 'run_config.json',
        field='config', selected=lambda: DEFAULT_CONFIG)
    route_path = args.route_records.resolve()
    baseline_path = args.frozen_graph_evidence.resolve()
    index_path = resolve_index_manifest(args, run_root)
    pilot_path = args.pilot_manifest.resolve() if args.pilot_manifest else None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "nextqa_visual_runtime_config_v2":
        raise ValueError("visual evidence v2 requires the v2 runtime config")
    store = VisualShardStore(index_path)
    routes = _select_pilot(_load_rows(route_path, "route records"), pilot_path)
    routes = sorted(routes, key=lambda row: (str(row["video_id"]), str(row["question_id"])))
    baseline_by_id = {
        str(row["question_id"]): row
        for row in _load_rows(baseline_path, "frozen graph evidence")
    }
    if not routes or any(str(row["question_id"]) not in baseline_by_id for row in routes):
        raise ValueError("routes do not align with the frozen graph evidence")
    if any(str(route["video_id"]) not in store.video_ids for route in routes):
        raise ValueError("route references a video outside the visual index")
    contract = {
        "dataset": "NExT-QA", "split": "val", "stage": "graph_bound_text_visual_evidence_v2",
        "config": fingerprint_file(config_path), "route_records": fingerprint_file(route_path),
        "frozen_graph_evidence": fingerprint_file(baseline_path),
        "visual_index_manifest": fingerprint_file(index_path),
        "pilot_manifest": fingerprint_file(pilot_path) if pilot_path else None,
        "question_count": len(routes), "device": args.device,
        "route_source": "nextqa_luna_ir_intent_only", "uses_official_type": False,
        "uses_answer_choices": False, "uses_gold_answer": False,
        "changes_graph_retrieval": False, "calls_api": False,
        "qa_receives_images": False, "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
    }
    persist_run_config(
        run_root / "run_config.json", contract=contract, resume=args.resume,
        schema_version="nextqa_visual_evidence_run_config_v2",
    )
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json", source_paths=sources, resume=args.resume
    )
    extractor = VisionFeatureExtractor(
        model_name=str(config["embedding_model"]),
        device=None if args.device == "auto" else args.device,
        local_files_only=True,
    )
    query_items: list[tuple[str, str, str]] = []
    for route in routes:
        for query in route["evidence_demand"]["query_intents"]:
            query_items.append(
                (str(route["question_id"]), str(query["role_id"]), str(query["query_text"]))
            )
    query_vectors: dict[str, dict[str, Any]] = defaultdict(dict)
    batch_size = int(config["text_embedding_batch_size"])
    for start in range(0, len(query_items), batch_size):
        batch = query_items[start : start + batch_size]
        vectors = extractor.encode_text_batch([item[2] for item in batch])
        for (question_id, role_id, _text), vector in zip(batch, vectors, strict=True):
            query_vectors[question_id][role_id] = vector

    records: list[dict[str, Any]] = []
    preview: list[dict[str, Any]] = []
    route_counts = Counter()
    status_counts = Counter()
    kind_counts = Counter()
    role_counts = Counter()
    active_video_id: str | None = None
    for route in routes:
        question_id = str(route["question_id"])
        video_id = str(route["video_id"])
        if active_video_id is not None and video_id != active_video_id:
            store.release_video(active_video_id)
        active_video_id = video_id
        baseline = baseline_by_id[question_id]
        role_hits = retrieve_role_candidates(store, route, query_vectors[question_id])
        pack = dispatch_handler(route, role_hits)
        route_counts[str(route["route_id"])] += 1
        status_counts[str(pack["status"])] += 1
        kind_counts.update(str(row["candidate_kind"]) for row in pack["evidence"])
        role_counts.update(str(row["role_id"]) for row in pack["evidence"])
        evidence_rows = [
            {"rank": index, **row}
            for index, row in enumerate(pack["evidence"], start=1)
        ]
        record = {
            "schema_version": "nextqa_graph_bound_visual_evidence_record_v2",
            "question_id": question_id, "video_id": video_id,
            "route_id": str(route["route_id"]), "route_source": str(route["route_source"]),
            "handler_id": str(pack["handler_id"]), "handler_module": str(pack["handler_module"]),
            "handler_status": str(pack["status"]), "boundary_available": bool(pack["boundary_available"]),
            "role_status": dict(pack["role_status"]),
            "graph_retrieval_status": str(baseline["status"]),
            "graph_retrieval_profile": str(baseline["graph_retrieval_profile"]),
            "graph_record_ids": list(baseline["graph_record_ids"]),
            "graph_record_ids_sha256": str(baseline["graph_record_ids_sha256"]),
            "graph_evidence_count": len(baseline["evidence"]),
            "visual_evidence": evidence_rows, "visual_hits": list(pack["visual_hits"]),
            "used_visual_candidate_ids": list(pack["used_visual_candidate_ids"]),
            "excluded_reason_counts": dict(pack["excluded_reason_counts"]),
            "uses_official_type": False, "uses_answer_choices": False,
            "uses_gold_answer": False, "changes_graph_retrieval": False,
            "offline_reads_visual_index": True, "qa_receives_images": False,
            "qa_receives_image_paths": False, "qa_receives_visual_embeddings": False,
            "calls_api": False,
        }
        records.append(record)
        preview.append(
            {
                "schema_version": "nextqa_text_only_visual_qa_input_preview_v2",
                "question_id": question_id, "video_id": video_id,
                "graph_evidence": [
                    {"rank": item["rank"], "text": item["evidence_text"]}
                    for item in baseline["evidence"]
                ],
                "visual_evidence": [
                    {"rank": item["rank"], "role_id": item["role_id"], "text": item["text"]}
                    for item in evidence_rows
                ],
                "image_content": [], "qa_receives_images": False,
                "qa_receives_image_paths": False, "qa_receives_visual_embeddings": False,
                "uses_answer_choices": False, "uses_gold_answer": False,
            }
        )
    if active_video_id is not None:
        store.release_video(active_video_id)
    handler_modules = {str(row["handler_module"]) for row in records}
    serialized_preview = json.dumps(preview, ensure_ascii=False).lower()
    checks = {
        "graph_record_ids_preserved": all(
            row["graph_record_ids_sha256"] == baseline_by_id[str(row["question_id"])]["graph_record_ids_sha256"]
            for row in records
        ),
        "same_video_only": all(
            str(hit["video_id"]) == str(row["video_id"])
            for row in records for hit in row["visual_hits"]
        ),
        "route_handler_purity": all(
            row["handler_id"] == f"nextqa_visual_handler_v2:{row['route_id']}"
            and row["handler_module"] == row["route_id"]
            for row in records
        ),
        "eight_independent_handler_modules_observed": handler_modules == set(HANDLERS),
        "strict_pair_purity": all(
            _strict_pair_is_valid(hit) for row in records for hit in row["visual_hits"]
        ),
        "visual_evidence_nonempty": all(bool(row["visual_evidence"]) for row in records),
        "qa_preview_has_no_image_paths": all(ext not in serialized_preview for ext in (".png", ".jpg", ".jpeg")),
        "qa_preview_image_content_empty": all(row["image_content"] == [] for row in preview),
    }
    audit = {
        "schema_version": "nextqa_visual_evidence_audit_v2", "run_id": args.run_id,
        "question_count": len(records), "video_count": len({str(row["video_id"]) for row in records}),
        "route_distribution": dict(sorted(route_counts.items())),
        "handler_status_distribution": dict(sorted(status_counts.items())),
        "selected_candidate_kind_distribution": dict(sorted(kind_counts.items())),
        "selected_role_distribution": dict(sorted(role_counts.items())),
        "boundary_unavailable_question_ids": [
            str(row["question_id"]) for row in records if row["handler_status"].startswith("boundary_unavailable")
        ],
        "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        **checks, "uses_official_type": False, "uses_answer_choices": False,
        "uses_gold_answer": False, "calls_api": False, "qa_receives_images": False,
        "qa_receives_visual_embeddings": False,
        "status": "passed" if all(checks.values()) else "failed",
    }
    atomic_write_json(run_root / "visual_evidence_records.json", records)
    atomic_write_json(run_root / "qa_text_input_preview.json", preview)
    atomic_write_json(run_root / "visual_evidence_audit.json", audit)
    atomic_write_json(
        run_root / "visual_evidence_manifest.json",
        {"schema_version": "nextqa_visual_evidence_manifest_v2", "run_id": args.run_id,
         "visual_evidence_records_path": str((run_root / "visual_evidence_records.json").resolve()),
         "qa_text_input_preview_path": str((run_root / "qa_text_input_preview.json").resolve()),
         "question_count": len(records), "video_count": audit["video_count"],
         "route_records_path": str(route_path), "frozen_graph_evidence_path": str(baseline_path),
         "visual_index_manifest_path": str(index_path), "changes_graph_retrieval": False,
         "uses_gold_answer": False, "qa_receives_images": False},
    )
    drift = persist_drift_report(
        run_root / "source_drift_report.json", snapshot=snapshot, current_source_paths=sources
    )
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "visual_evidence_audit.json", audit)
    atomic_write_json(
        run_root / "run_status.json",
        {"status": audit["status"], "question_count": len(records), "calls_api": False,
         "code_drift_detected": drift["code_drift_detected"]},
    )
    if audit["status"] != "passed":
        raise SystemExit(f"visual evidence v2 audit failed: {audit['failed_checks']}")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
