"""Materialize the fixed NExT-QA visual route policy v3 and reusable responses."""

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

from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text, require_within_nextqa_scratch
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.visual_qa import build_visual_qa_messages, load_visual_mcq_config
from vlkg.datasets.nextqa.qa.graph_only_qa import request_sha256
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.route_policy_v3 import POLICY_ID, TARGET_IDENTITIES, TARGET_ROUTE_IDS, materialize_visual_record




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'ablation' / args.run_id / "run_config.json"
    selected = {}
    selected['policy'] = recorded_or_selected(args.policy, resume=args.resume, run_config=run_config, field='policy', selected=lambda: REPO_ROOT / "configs/datasets/nextqa/route_policy.json")
    selected['qa_config'] = recorded_or_selected(args.qa_config, resume=args.resume, run_config=run_config, field='qa_config', selected=lambda: REPO_ROOT / "configs/datasets/nextqa/visual_qa.yaml")
    selected['route_records'] = recorded_or_selected(args.route_records, resume=args.resume, run_config=run_config, field='route_records', selected=lambda: asset_file("nextqa", 'retrieval', 'routes', REPO_ROOT))
    selected['routed_evidence'] = recorded_or_selected(args.routed_evidence, resume=args.resume, run_config=run_config, field='routed_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'visual_evidence', REPO_ROOT))
    selected['uniform_evidence'] = recorded_or_selected(args.uniform_evidence, resume=args.resume, run_config=run_config, field='uniform_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'uniform_evidence', REPO_ROOT))
    selected['uniform_predictions'] = recorded_or_selected(args.uniform_predictions, resume=args.resume, run_config=run_config, field='uniform_predictions', selected=lambda: asset_file("nextqa", 'runs', 'B', REPO_ROOT))
    selected['frozen_graph_evidence'] = recorded_or_selected(args.frozen_graph_evidence, resume=args.resume, run_config=run_config, field='frozen_graph_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'frozen_graph', REPO_ROOT))
    selected['questions'] = recorded_or_selected(args.questions, resume=args.resume, run_config=run_config, field='questions', selected=lambda: asset_file("nextqa", 'ir', 'runtime', REPO_ROOT))
    return selected


def _rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in output:
            raise ValueError(f"{name} contains an empty or duplicate question ID")
        output[question_id] = row
    return output


def _run_root(run_id: str, resume: bool) -> Path:
    root = require_within_nextqa_scratch(NEXTQA_SCRATCH_ROOT / "ablation" / run_id)
    if root.exists() and not resume:
        raise FileExistsError(f"run already exists; use --resume explicitly: {root}")
    if not root.exists() and resume:
        raise FileNotFoundError(f"cannot resume missing run: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed eight-route NExT-QA visual policy v3.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--qa-config", type=Path, default=None)
    parser.add_argument("--route-records", type=Path, default=None)
    parser.add_argument("--routed-evidence", type=Path, default=None)
    parser.add_argument("--uniform-evidence", type=Path, default=None)
    parser.add_argument("--uniform-predictions", type=Path, default=None)
    parser.add_argument("--frozen-graph-evidence", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    paths = {name: inputs[arg] for name, arg in {
        "policy": "policy", "qa_config": "qa_config", "routes": "route_records",
        "routed_evidence": "routed_evidence", "uniform_evidence": "uniform_evidence",
        "uniform_predictions": "uniform_predictions", "graph": "frozen_graph_evidence",
        "questions": "questions",
    }.items()}
    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    if policy.get("schema_version") != "nextqa_visual_route_policy_config_v3" or policy.get("policy_id") != POLICY_ID:
        raise ValueError("route policy config identity mismatch")
    configured = policy.get("route_strategies") or {}
    if {route: configured.get(route) for route in TARGET_ROUTE_IDS} != TARGET_IDENTITIES:
        raise ValueError("target route strategy identities do not match code")
    routes = _index(_rows(paths["routes"], "routes"), "routes")
    routed = _index(_rows(paths["routed_evidence"], "routed evidence"), "routed evidence")
    uniform = _index(_rows(paths["uniform_evidence"], "uniform evidence"), "uniform evidence")
    uniform_predictions = _index(_rows(paths["uniform_predictions"], "uniform predictions"), "uniform predictions")
    graphs = _index(_rows(paths["graph"], "frozen graph evidence"), "frozen graph evidence")
    questions = _index(_rows(paths["questions"], "questions"), "questions")
    expected_ids = set(routes)
    if len(expected_ids) != 4996 or any(set(rows) != expected_ids for rows in (routed, uniform, uniform_predictions, graphs, questions)):
        raise ValueError("all policy inputs must exactly cover the fixed 4,996-question set")
    qa_config = load_visual_mcq_config(paths["qa_config"])
    run_root = _run_root(args.run_id, args.resume)
    contract = {
        "dataset": "NExT-QA", "split": "val", "stage": "visual_route_policy_v3_materialization",
        "policy": fingerprint_file(paths["policy"]), "route_records": fingerprint_file(paths["routes"]),
        "routed_evidence": fingerprint_file(paths["routed_evidence"]), "uniform_evidence": fingerprint_file(paths["uniform_evidence"]),
        "uniform_predictions": fingerprint_file(paths["uniform_predictions"]), "frozen_graph_evidence": fingerprint_file(paths["graph"]),
        "questions": fingerprint_file(paths["questions"]), "qa_config": fingerprint_file(paths["qa_config"]),
        "question_count": 4996, "target_route_ids": sorted(TARGET_ROUTE_IDS), "target_route_strategies": TARGET_IDENTITIES,
        "reuses_api_responses_only_after_exact_request_hash_match": True, "calls_api": False,
        "selects_better_answer_per_question": False, "uses_official_type": False, "uses_gold_answer": False,
    }
    persist_run_config(run_root / "run_config.json", contract=contract, resume=args.resume, schema_version="nextqa_visual_route_policy_v3_run_config_v1")
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / "source_snapshot.json", source_paths=sources, resume=args.resume)

    visual_records: list[dict[str, Any]] = []
    overlay_predictions: list[dict[str, Any]] = []
    target_counts = {route: 0 for route in sorted(TARGET_ROUTE_IDS)}
    hash_mismatches: list[str] = []
    graph_mismatches: list[str] = []
    non_target_unchanged = 0
    for question_id in sorted(expected_ids, key=lambda qid: (str(routes[qid]["video_id"]), qid)):
        route = routes[question_id]
        route_id = str(route["route_id"])
        visual = materialize_visual_record(route, routed[question_id], uniform[question_id])
        visual_records.append(visual)
        if route_id not in TARGET_ROUTE_IDS:
            non_target_unchanged += int(visual == routed[question_id])
            continue
        target_counts[route_id] += 1
        if list(visual.get("graph_record_ids", [])) != list(graphs[question_id].get("graph_record_ids", [])):
            graph_mismatches.append(question_id)
        messages = build_visual_qa_messages(questions[question_id], graphs[question_id], visual, config=qa_config)
        expected_hash = str(uniform_predictions[question_id].get("request_sha256") or "")
        actual_hash = request_sha256(messages)
        if not expected_hash or actual_hash != expected_hash:
            hash_mismatches.append(question_id)
            continue
        source = uniform_predictions[question_id]
        identity = TARGET_IDENTITIES[route_id]
        prediction = deepcopy(source)
        prediction.update({
            "schema_version": "nextqa_visual_route_policy_v3_reused_prediction_v1",
            "route_id": route_id,
            "handler_id": identity["handler_id"],
            "handler_status": "completed",
            "strategy_id": identity["strategy_id"],
            "strategy_scope": "route_wide",
            "strategy_family": "uniform_visual_retrieval",
            "route_policy_id": POLICY_ID,
            "source_prediction_schema_version": source.get("schema_version"),
            "source_qa_profile": source.get("qa_profile"),
            "response_reused_by_exact_request_sha256": True,
            "visual_candidate_ids": list(visual.get("used_visual_candidate_ids", [])),
        })
        overlay_predictions.append(prediction)

    target_ids = {qid for qid, row in routes.items() if row.get("route_id") in TARGET_ROUTE_IDS}
    checks = {
        "full_visual_evidence_coverage": len(visual_records) == 4996,
        "target_route_exact_coverage": len(overlay_predictions) == len(target_ids) == sum(target_counts.values()),
        "target_counts_match": target_counts == {"causal_how": 677, "descriptive_count": 178},
        "all_target_request_hashes_match_uniform_qa": not hash_mismatches,
        "graph_record_ids_preserved": not graph_mismatches,
        "six_non_target_routes_unchanged": non_target_unchanged == 4996 - len(target_ids),
        "target_handlers_distinct": len({value["handler_id"] for value in TARGET_IDENTITIES.values()}) == 2,
        "no_per_question_answer_selection": policy.get("selects_better_answer_per_question") is False,
    }
    audit = {
        "schema_version": "nextqa_visual_route_policy_v3_audit_v1", "run_id": args.run_id,
        "policy_id": POLICY_ID, "question_count": 4996, "target_question_count": len(target_ids),
        "retained_v2_question_count": non_target_unchanged, "target_route_counts": target_counts,
        "request_hash_mismatch_question_ids": hash_mismatches, "graph_record_mismatch_question_ids": graph_mismatches,
        "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "selects_better_answer_per_question": False, "uses_official_type": False,
        "uses_gold_answer": False, "calls_api": False, "status": "passed" if all(checks.values()) else "failed",
    }
    atomic_write_json(run_root / "visual_evidence_records.json", visual_records)
    atomic_write_json(run_root / "target_overlay_predictions.json", overlay_predictions)
    atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(run_root / "manifest.json", {
        "schema_version": "nextqa_visual_route_policy_v3_manifest_v1", "run_id": args.run_id,
        "policy_id": POLICY_ID, "visual_evidence_records_path": str((run_root / "visual_evidence_records.json").resolve()),
        "target_overlay_predictions_path": str((run_root / "target_overlay_predictions.json").resolve()),
        "question_count": 4996, "target_question_count": len(target_ids), "calls_api": False,
    })
    atomic_write_text(run_root / "summary.md", "\n".join([
        "# NExT-QA Visual Route Policy V3", "", f"- Policy: `{POLICY_ID}`",
        f"- Target routes: `{sorted(TARGET_ROUTE_IDS)}`", f"- Target questions: `{len(target_ids)}`",
        f"- Exact reused-request hashes: `{len(target_ids) - len(hash_mismatches)}/{len(target_ids)}`",
        f"- Non-target evidence unchanged: `{non_target_unchanged}`", f"- Status: `{audit['status']}`", "",
    ]))
    drift = persist_drift_report(run_root / "source_drift_report.json", snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(run_root / "run_status.json", {"status": audit["status"], "question_count": 4996, "target_question_count": len(target_ids), "calls_api": False, "code_drift_detected": drift["code_drift_detected"]})
    if audit["status"] != "passed":
        raise SystemExit(f"visual route policy v3 audit failed: {audit['failed_checks']}")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
