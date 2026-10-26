"""Merge the two fixed route-wide v3 strategies into a frozen prediction set."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text, require_within_nextqa_scratch, NEXTQA_SCRATCH_ROOT
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.route_policy_v3 import POLICY_ID, TARGET_IDENTITIES, TARGET_ROUTE_IDS




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'ablation' / args.run_id / "run_config.json"
    selected = {}
    selected['route_records'] = recorded_or_selected(args.route_records, resume=args.resume, run_config=run_config, field='route_records', selected=lambda: asset_file("nextqa", 'retrieval', 'routes', REPO_ROOT))
    return selected


def _rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list")
    return payload


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if not qid or qid in output:
            raise ValueError(f"{name} contains an empty or duplicate question ID")
        output[qid] = row
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge NExT-QA visual route policy v3 predictions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--target-overlay-predictions", type=Path, required=True)
    parser.add_argument("--route-records", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    base_path = args.base_predictions.resolve()
    overlay_path = args.target_overlay_predictions.resolve()
    route_path = inputs['route_records']
    base_rows = _rows(base_path, "base predictions")
    base = _index(base_rows, "base predictions")
    overlay = _index(_rows(overlay_path, "target overlay predictions"), "target overlay predictions")
    routes = _index(_rows(route_path, "route records"), "route records")
    expected_ids = set(routes)
    target_ids = {qid for qid, row in routes.items() if row.get("route_id") in TARGET_ROUTE_IDS}
    if len(expected_ids) != 4996 or set(base) != expected_ids or set(overlay) != target_ids:
        raise ValueError("base, route, or target overlay coverage mismatch")
    run_root = require_within_nextqa_scratch(NEXTQA_SCRATCH_ROOT / "ablation" / args.run_id)
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_root}")
    if not run_root.exists() and args.resume:
        raise FileNotFoundError(f"cannot resume missing run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=args.resume)
    contract = {"dataset": "NExT-QA", "split": "val", "stage": "visual_route_policy_v3_prediction_composite", "policy_id": POLICY_ID, "base_predictions": fingerprint_file(base_path), "target_overlay_predictions": fingerprint_file(overlay_path), "route_records": fingerprint_file(route_path), "question_count": 4996, "target_route_ids": sorted(TARGET_ROUTE_IDS), "selects_better_answer_per_question": False, "uses_official_type": False, "uses_gold_answer": False, "calls_api": False}
    persist_run_config(run_root / "run_config.json", contract=contract, resume=args.resume, schema_version="nextqa_visual_route_policy_v3_composite_run_config_v1")
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / "source_snapshot.json", source_paths=sources, resume=args.resume)
    merged = []
    provenance = []
    retained = 0
    graph_mismatches = []
    for base_row in base_rows:
        qid = str(base_row["question_id"])
        route_id = str(routes[qid]["route_id"])
        if qid in target_ids:
            row = overlay[qid]
            identity = TARGET_IDENTITIES[route_id]
            if row.get("route_id") != route_id or row.get("strategy_id") != identity["strategy_id"] or row.get("handler_id") != identity["handler_id"] or row.get("response_reused_by_exact_request_sha256") is not True:
                raise ValueError(f"target strategy identity mismatch: {qid}")
            if list(row.get("graph_record_ids", [])) != list(base_row.get("graph_record_ids", [])):
                graph_mismatches.append(qid)
            merged.append(deepcopy(row))
            source = "uniform_route_handler_exact_request_reuse"
            strategy = identity["strategy_id"]
        else:
            merged.append(deepcopy(base_row))
            retained += 1
            source = "frozen_base_prediction"
            strategy = str(base_row.get("strategy_id") or "nextqa_visual_runtime_v2")
        provenance.append({"question_id": qid, "route_id": route_id, "prediction_source": source, "strategy_id": strategy})
    checks = {"full_question_coverage": len(merged) == 4996, "target_route_exact_coverage": len(overlay) == len(target_ids) == 855, "graph_record_ids_preserved": not graph_mismatches, "non_target_predictions_unchanged": all(merged[index] == base_rows[index] for index in range(len(base_rows)) if str(base_rows[index]["question_id"]) not in target_ids), "no_per_question_answer_selection": True}
    audit = {"schema_version": "nextqa_visual_route_policy_v3_composite_audit_v1", "run_id": args.run_id, "policy_id": POLICY_ID, "question_count": len(merged), "target_question_count": len(target_ids), "retained_base_question_count": retained, "graph_record_mismatch_question_ids": graph_mismatches, "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed], "selects_better_answer_per_question": False, "uses_official_type": False, "uses_gold_answer": False, "calls_api": False, "status": "passed" if all(checks.values()) else "failed"}
    atomic_write_json(run_root / "predictions.json", merged)
    atomic_write_json(run_root / "prediction_provenance.json", provenance)
    atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(run_root / "manifest.json", {"schema_version": "nextqa_visual_route_policy_v3_composite_manifest_v1", "run_id": args.run_id, "policy_id": POLICY_ID, "predictions_path": str((run_root / "predictions.json").resolve()), "prediction_provenance_path": str((run_root / "prediction_provenance.json").resolve()), "base_predictions_path": str(base_path), "question_count": 4996, "target_question_count": len(target_ids), "calls_api": False})
    atomic_write_text(run_root / "summary.md", "\n".join(["# NExT-QA Visual Route Policy V3 Composite", "", f"- Base: `{base_path}`", f"- Replaced fixed routes: `{sorted(TARGET_ROUTE_IDS)}`", f"- Replaced questions: `{len(target_ids)}`", f"- Retained base questions: `{retained}`", "- Per-question answer selection: `False`", f"- Status: `{audit['status']}`", ""]))
    drift = persist_drift_report(run_root / "source_drift_report.json", snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(run_root / "run_status.json", {"status": audit["status"], "question_count": 4996, "calls_api": False, "code_drift_detected": drift["code_drift_detected"]})
    if audit["status"] != "passed":
        raise SystemExit(f"route policy composite failed: {audit['failed_checks']}")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
