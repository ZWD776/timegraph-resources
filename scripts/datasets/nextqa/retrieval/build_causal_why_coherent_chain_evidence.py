"""Build an isolated causal_why visual-evidence pilot with the v3 handler."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.runtime.run_snapshot import (
    collect_imported_repo_sources,
    fingerprint_file,
    persist_drift_report,
    persist_run_config,
    persist_source_snapshot,
    sha256_file,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import (
    VisualShardStore,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.visual_qa import load_visual_mcq_config
from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import causal_why_coherent_chain as strategy_v1
from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import causal_why_coherent_chain_v2 as strategy_v2
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain import (
    ALLOWED_KINDS,
    EFFECT_ROLE,
    HANDLER_ID,
    INTERACTION_ROLE,
    PRECEDING_ROLE,
    ROUTE_ID,
    ROLE_ORDER,
    ROLE_QUOTAS,
    STRATEGY_ID,
    build_coherent_causal_chain_pack,
)
from vlkg.core.indexing.vision import VisionFeatureExtractor




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'visual_evidence' / args.run_id / "run_config.json"
    selected = {}
    selected['route_records'] = recorded_or_selected(args.route_records, resume=args.resume, run_config=run_config, field='route_records', selected=lambda: asset_file("nextqa", 'retrieval', 'routes', REPO_ROOT))
    selected['frozen_graph_evidence'] = recorded_or_selected(args.frozen_graph_evidence, resume=args.resume, run_config=run_config, field='frozen_graph_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'frozen_graph', REPO_ROOT))
    selected['visual_index_manifest'] = recorded_or_selected(args.visual_index_manifest, resume=args.resume, run_config=run_config, field='visual_index_manifest', selected=lambda: asset_file("nextqa", 'indexes', 'manifest', REPO_ROOT))
    selected['config'] = recorded_or_selected(args.config, resume=args.resume, run_config=run_config, field='config', selected=lambda: REPO_ROOT / "configs/datasets/nextqa/visual_runtime.json")
    if args.scope == "pilot":
        if args.pilot_manifest is None:
            raise ValueError("pilot scope requires an explicit --pilot-manifest")
        selected["pilot_manifest"] = args.pilot_manifest.resolve()
    else:
        selected["pilot_manifest"] = None
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-records", type=Path, default=None)
    parser.add_argument("--frozen-graph-evidence", type=Path, default=None)
    parser.add_argument("--visual-index-manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pilot-manifest", type=Path, default=None)
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--search-top-k", type=int, default=24)
    parser.add_argument("--evidence-limit", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strategy-version", choices=("v1", "v2"), default="v2")
    return parser


def _strategy_spec(version: str) -> dict[str, Any]:
    if version == "v2":
        return {
            "module": strategy_v2,
            "strategy_id": strategy_v2.STRATEGY_ID,
            "handler_id": strategy_v2.HANDLER_ID,
            "build": strategy_v2.build_coherent_causal_chain_pack_v2,
            "schema_version": "nextqa_graph_bound_visual_evidence_record_v4",
        }
    return {
        "module": strategy_v1,
        "strategy_id": strategy_v1.STRATEGY_ID,
        "handler_id": strategy_v1.HANDLER_ID,
        "build": strategy_v1.build_coherent_causal_chain_pack,
    "schema_version": "nextqa_graph_bound_visual_evidence_record_v3",
    }


def _load_list(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id or question_id in output:
            raise ValueError(f"{name} contains an empty or duplicate question ID")
        output[question_id] = row
    return output


def _pilot_video_ids(path: Path) -> set[str]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    video_ids = {str(row["official_video_id"]) for row in payload.get("videos", [])}
    if len(video_ids) != 5:
        raise ValueError("pilot manifest must contain exactly five videos")
    return video_ids


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


def _route_query_text(route: dict[str, Any], role: str) -> str:
    for query in (route.get("evidence_demand") or {}).get("query_intents", []):
        if str(query.get("role_id") or "") == role:
            return str(query.get("query_text") or "").strip()
    raise ValueError(f"{route['question_id']}: missing persisted query for role {role}")


def _retrieve_role_hits(
    store: VisualShardStore,
    route: dict[str, Any],
    query_vectors: dict[str, Any],
    *,
    search_top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    video_id = str(route["video_id"])
    role_hits: dict[str, list[dict[str, Any]]] = {}
    for role in ROLE_ORDER:
        vector = query_vectors[role]
        hits: list[dict[str, Any]] = []
        for kind in sorted(ALLOWED_KINDS[role]):
            hits.extend(
                store.search(
                    video_id=video_id,
                    kind=kind,
                    query_vector=vector,
                    top_k=search_top_k,
                )
            )
        role_hits[role] = hits
    return role_hits


def _build_preview(
    question: dict[str, Any],
    graph: dict[str, Any],
    visual: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "nextqa_text_only_visual_qa_input_preview_v3",
        "question_id": str(question["question_id"]),
        "video_id": str(question["video_id"]),
        "graph_evidence": [
            {"rank": item.get("rank"), "text": item.get("evidence_text")}
            for item in graph.get("evidence", [])
        ],
        "visual_evidence": [
            {"rank": item.get("rank"), "role_id": item.get("role_id"), "text": item.get("text")}
            for item in visual.get("visual_evidence", [])
        ],
        "image_content": [],
        "qa_receives_images": False,
        "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
        "uses_answer_choices": False,
        "uses_gold_answer": False,
    }


def main() -> int:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    strategy = _strategy_spec(args.strategy_version)
    if args.search_top_k < 1 or args.evidence_limit < 1:
        raise ValueError("search-top-k and evidence-limit must be positive")
    route_path = inputs['route_records']
    graph_path = inputs['frozen_graph_evidence']
    index_path = inputs['visual_index_manifest']
    config_path = inputs['config']
    pilot_path = inputs['pilot_manifest']
    routes_all = _load_list(route_path, "route records")
    graph_all = _load_list(graph_path, "frozen graph evidence")
    graphs_by_id = _index(graph_all, "frozen graph evidence")
    pilot_ids = _pilot_video_ids(pilot_path) if args.scope == "pilot" else set()
    routes = [
        row for row in routes_all
        if str(row.get("route_id") or "") == ROUTE_ID
        and (args.scope == "full" or str(row.get("video_id")) in pilot_ids)
    ]
    routes.sort(key=lambda row: (str(row["video_id"]), str(row["question_id"])))
    if args.scope == "pilot" and not routes:
        raise ValueError("pilot contains no causal_why questions")
    if any(str(row["question_id"]) not in graphs_by_id for row in routes):
        raise ValueError("causal_why route records do not align with frozen graph evidence")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "nextqa_visual_runtime_config_v2":
        raise ValueError("causal_why v3 builder requires the v2 visual runtime config")
    run_root = prepare_stage_run_root("visual_evidence", args.run_id, resume=args.resume)
    contract = {
        "dataset": "NExT-QA",
        "split": "val",
        "stage": f"causal_why_coherent_chain_visual_evidence_{args.strategy_version}",
        "scope": args.scope,
        "run_id": args.run_id,
        "route_id": ROUTE_ID,
        "strategy_id": strategy["strategy_id"],
        "handler_id": strategy["handler_id"],
        "route_records": fingerprint_file(route_path),
        "frozen_graph_evidence": fingerprint_file(graph_path),
        "visual_index_manifest": fingerprint_file(index_path),
        "config": fingerprint_file(config_path),
        "pilot_manifest": fingerprint_file(pilot_path) if args.scope == "pilot" else None,
        "question_count": len(routes),
        "video_count": len({str(row["video_id"]) for row in routes}),
        "search_top_k": args.search_top_k,
        "evidence_limit": args.evidence_limit,
        "role_quotas": strategy["module"].ROLE_QUOTAS,
        "device": args.device,
        "changes_graph_retrieval": False,
        "uses_official_type": False,
        "uses_answer_choices": False,
        "uses_gold_answer": False,
        "calls_api": False,
        "qa_receives_images": False,
        "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
    }
    persist_run_config(
        run_root / "run_config.json",
        contract=contract,
        resume=args.resume,
        schema_version="nextqa_causal_why_coherent_chain_evidence_run_config_v1",
    )
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json",
        source_paths=source_paths,
        resume=args.resume,
    )
    store = VisualShardStore(index_path)
    extractor = VisionFeatureExtractor(
        model_name=str(config["embedding_model"]),
        device=None if args.device == "auto" else args.device,
        local_files_only=True,
    )
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    route_queries: dict[str, list[str]] = {
        str(route["question_id"]): [_route_query_text(route, role) for role in ROLE_ORDER]
        for route in routes
    }
    all_query_items = [
        (question_id, role, query_text)
        for question_id in [str(route["question_id"]) for route in routes]
        for role, query_text in zip(ROLE_ORDER, route_queries[question_id], strict=True)
    ]
    query_vectors_by_id: dict[str, dict[str, Any]] = {}
    batch_size = int(config.get("text_embedding_batch_size", 32))
    for start in range(0, len(all_query_items), batch_size):
        batch = all_query_items[start : start + batch_size]
        vectors = extractor.encode_text_batch([item[2] for item in batch])
        for (question_id, role, _query_text), vector in zip(batch, vectors, strict=True):
            query_vectors_by_id.setdefault(question_id, {})[role] = vector
    status_counts: Counter[str] = Counter()
    candidate_kind_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    active_video_id: str | None = None
    for route in routes:
        question_id = str(route["question_id"])
        video_id = str(route["video_id"])
        if active_video_id is not None and active_video_id != video_id:
            store.release_video(active_video_id)
        active_video_id = video_id
        query_vectors = query_vectors_by_id[question_id]
        role_hits = _retrieve_role_hits(
            store,
            route,
            query_vectors,
            search_top_k=args.search_top_k,
        )
        pack = strategy["build"](
            route,
            role_hits,
            evidence_limit=args.evidence_limit,
        )
        baseline = graphs_by_id[question_id]
        evidence_rows = [
            {"rank": rank, **row}
            for rank, row in enumerate(pack["evidence"], start=1)
        ]
        record = {
            "schema_version": strategy["schema_version"],
            "question_id": question_id,
            "video_id": video_id,
            "route_id": ROUTE_ID,
            "route_source": "nextqa_luna_ir_intent_only",
            "strategy_id": strategy["strategy_id"],
            "handler_id": strategy["handler_id"],
            "handler_module": "causal_why_coherent_chain_v2" if args.strategy_version == "v2" else "causal_why_coherent_chain",
            "handler_status": str(pack["status"]),
            "boundary_available": bool(pack["boundary_available"]),
            "role_status": dict(pack["role_status"]),
            "strategy_metadata": dict(pack["strategy_metadata"]),
            "graph_retrieval_status": str(baseline.get("status")),
            "graph_retrieval_profile": str(baseline.get("graph_retrieval_profile")),
            "graph_record_ids": list(baseline.get("graph_record_ids", [])),
            "graph_record_ids_sha256": str(baseline.get("graph_record_ids_sha256", "")),
            "graph_evidence_count": len(baseline.get("evidence", [])),
            "visual_evidence": evidence_rows,
            "visual_hits": list(pack["visual_hits"]),
            "used_visual_candidate_ids": list(pack["used_visual_candidate_ids"]),
            "excluded_reason_counts": dict(pack["excluded_reason_counts"]),
            "uses_official_type": False,
            "uses_answer_choices": False,
            "uses_gold_answer": False,
            "changes_graph_retrieval": False,
            "offline_reads_visual_index": True,
            "qa_receives_images": False,
            "qa_receives_image_paths": False,
            "qa_receives_visual_embeddings": False,
            "calls_api": False,
        }
        records.append(record)
        previews.append(_build_preview(route, baseline, record))
        status_counts.update([str(pack["status"])])
        candidate_kind_counts.update(str(row["candidate_kind"]) for row in evidence_rows)
        role_counts.update(str(row["role_id"]) for row in evidence_rows)
        excluded_counts.update({key: int(value) for key, value in pack["excluded_reason_counts"].items()})
    if active_video_id is not None:
        store.release_video(active_video_id)
    # Contract flags in the preview intentionally mention image paths and
    # embeddings.  Inspect only the evidence payload for leaked media data.
    preview_evidence_text = json.dumps(
        [
            {
                "visual_evidence": row["visual_evidence"],
                "visual_hits": record["visual_hits"],
            }
            for row, record in zip(previews, records, strict=True)
        ],
        ensure_ascii=False,
    ).lower()
    checks = {
        "causal_why_only": all(row["route_id"] == ROUTE_ID for row in records),
        "question_count_matches_scope": len(records) == len(routes),
        "unique_question_ids": len({row["question_id"] for row in records}) == len(records),
        "graph_record_ids_preserved": all(
            row["graph_record_ids_sha256"] == graphs_by_id[row["question_id"]].get("graph_record_ids_sha256")
            for row in records
        ),
        "all_graph_retrieval_completed": all(row["graph_retrieval_status"] == "completed" for row in records),
        "same_video_only": all(
            str(hit.get("video_id")) == row["video_id"]
            for row in records for hit in row["visual_hits"]
        ),
        "strict_pair_purity": all(
            _strict_pair_is_valid(hit) for row in records for hit in row["visual_hits"]
        ),
        "fixed_evidence_limit": all(len(row["visual_evidence"]) <= args.evidence_limit for row in records),
        "visual_evidence_nonempty": all(bool(row["visual_evidence"]) for row in records),
        "handler_identity_pure": all(
            row["handler_id"] == strategy["handler_id"] for row in records
        ),
        "no_gold_or_official_type": all(
            row["uses_gold_answer"] is False and row["uses_official_type"] is False for row in records
        ),
        "graph_retrieval_unchanged": all(row["changes_graph_retrieval"] is False for row in records),
        "qa_preview_has_no_image_paths": all(
            ext not in preview_evidence_text
            for ext in (".png", ".jpg", ".jpeg", "image_path", "image_url", "embedding")
        ),
        "qa_preview_image_content_empty": all(row["image_content"] == [] for row in previews),
    }
    audit = {
        "schema_version": "nextqa_causal_why_coherent_chain_evidence_audit_v1",
        "run_id": args.run_id,
        "scope": args.scope,
        "route_id": ROUTE_ID,
        "strategy_id": strategy["strategy_id"],
        "handler_id": strategy["handler_id"],
        "question_count": len(records),
        "video_count": len({row["video_id"] for row in records}),
        "handler_status_distribution": dict(sorted(status_counts.items())),
        "selected_candidate_kind_distribution": dict(sorted(candidate_kind_counts.items())),
        "selected_role_distribution": dict(sorted(role_counts.items())),
        "excluded_reason_counts": dict(sorted(excluded_counts.items())),
        "boundary_unavailable_question_ids": [
            row["question_id"] for row in records if not row["boundary_available"]
        ],
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "uses_gold_answer": False,
        "uses_official_type": False,
        "uses_answer_choices": False,
        "changes_graph_retrieval": False,
        "calls_api": False,
        "qa_receives_images": False,
        "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False,
        "status": "passed" if all(checks.values()) else "failed",
    }
    atomic_write_json(run_root / "visual_evidence_records.json", records)
    atomic_write_json(run_root / "qa_text_input_preview.json", previews)
    atomic_write_json(run_root / "visual_evidence_audit.json", audit)
    atomic_write_json(
        run_root / "visual_evidence_manifest.json",
        {
            "schema_version": "nextqa_causal_why_coherent_chain_evidence_manifest_v1",
            "run_id": args.run_id,
            "route_id": ROUTE_ID,
            "strategy_id": strategy["strategy_id"],
            "handler_id": strategy["handler_id"],
            "question_count": len(records),
            "video_count": audit["video_count"],
            "visual_evidence_records_path": str((run_root / "visual_evidence_records.json").resolve()),
            "qa_text_input_preview_path": str((run_root / "qa_text_input_preview.json").resolve()),
            "route_records_path": str(route_path),
            "frozen_graph_evidence_path": str(graph_path),
            "visual_index_manifest_path": str(index_path),
            "changes_graph_retrieval": False,
            "uses_gold_answer": False,
            "uses_official_type": False,
            "qa_receives_images": False,
        },
    )
    atomic_write_text(
        run_root / "summary.md",
        "\n".join(
            [
                "# NExT-QA causal_why coherent-chain evidence v3",
                "",
                f"- Run: `{args.run_id}`",
                f"- Scope: `{args.scope}`",
                f"- Questions: `{len(records)}`",
                f"- Videos: `{audit['video_count']}`",
                f"- Strategy: `{strategy['strategy_id']}`",
                f"- Handler: `{strategy['handler_id']}`",
                f"- Status: `{audit['status']}`",
                f"- Failed checks: `{audit['failed_checks']}`",
                "- API calls: `false`",
                "- QA receives images: `false`",
                "",
            ]
        ),
    )
    drift = persist_drift_report(
        run_root / "source_drift_report.json",
        snapshot=snapshot,
        current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]),
    )
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "visual_evidence_audit.json", audit)
    atomic_write_json(
        run_root / "run_status.json",
        {
            "status": audit["status"],
            "run_id": args.run_id,
            "question_count": len(records),
            "calls_api": False,
            "code_drift_detected": drift["code_drift_detected"],
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
