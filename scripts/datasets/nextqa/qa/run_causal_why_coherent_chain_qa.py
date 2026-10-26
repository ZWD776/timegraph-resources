"""Run isolated Luna QA for the causal_why coherent-chain strategy."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.qa.graph_only_qa import parse_api_key_pool, stable_api_key_slot
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.runtime.run_snapshot import (
    collect_imported_repo_sources,
    fingerprint_file,
    persist_drift_report,
    persist_run_config,
    persist_source_snapshot,
    sha256_file,
    utc_now,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.visual_qa import (
    answer_visual_mcq,
    load_visual_mcq_config,
    validate_visual_qa_inputs,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import (
    HANDLER_ID as V1_HANDLER_ID,
    STRATEGY_ID as V1_STRATEGY_ID,
    V2_HANDLER_ID,
    V2_STRATEGY_ID,
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain import ROUTE_ID




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'qa_pilots' / args.run_id / "run_config.json"
    selected = {}
    selected['question_ir'] = recorded_or_selected(args.question_ir, resume=args.resume, run_config=run_config, field='question_ir', selected=lambda: asset_file("nextqa", 'ir', 'runtime', REPO_ROOT))
    selected['frozen_graph_evidence'] = recorded_or_selected(args.frozen_graph_evidence, resume=args.resume, run_config=run_config, field='frozen_graph_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'frozen_graph', REPO_ROOT))
    selected['visual_evidence'] = recorded_or_selected(args.visual_evidence, resume=args.resume, run_config=run_config, field='visual_evidence', selected=lambda: asset_file("nextqa", 'retrieval', 'causal_evidence', REPO_ROOT))
    selected['config'] = recorded_or_selected(args.config, resume=args.resume, run_config=run_config, field='config', selected=lambda: REPO_ROOT / "configs/datasets/nextqa/causal_why_qa.yaml")
    return selected


def _strategy_spec(version: str) -> tuple[str, str]:
    if version == "v2":
        return V2_STRATEGY_ID, V2_HANDLER_ID
    return V1_STRATEGY_ID, V1_HANDLER_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--question-ir", type=Path, default=None)
    parser.add_argument("--frozen-graph-evidence", type=Path, default=None)
    parser.add_argument("--visual-evidence", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--answer-mode", choices=("mock", "live"), default="live")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strategy-version", choices=("v1", "v2"), default="v2")
    return parser


def _load_list(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id or question_id in result:
            raise ValueError(f"{name} has an empty or duplicate question ID")
        result[question_id] = row
    return result


def _write_checkpoint(
    run_root: Path,
    rows: list[dict[str, Any]],
    *,
    expected_ids: list[str],
    run_id: str,
    answer_mode: str,
    strategy_id: str,
    handler_id: str,
) -> dict[str, Any]:
    by_id = _index(rows, "predictions") if rows else {}
    ordered = [by_id[qid] for qid in expected_ids if qid in by_id]
    completed = [row for row in ordered if row.get("status") == "completed"]
    errors = [row for row in ordered if row.get("status") == "error"]
    checks = {
        "duplicate_question_id_count_zero": len(by_id) == len(rows),
        "unexpected_question_ids_empty": not (set(by_id) - set(expected_ids)),
        "gold_purity": all(row.get("uses_gold_answer") is False for row in ordered),
        "official_type_purity": all(row.get("uses_official_type") is False for row in ordered),
        "text_only_qa_boundary": all(
            row.get("qa_receives_images") is False
            and row.get("qa_receives_image_paths") is False
            and row.get("qa_receives_visual_embeddings") is False
            for row in ordered
        ),
        "equal_choice_budgets": all(
            isinstance(row.get("evidence_budget_per_choice"), list)
            and len(row["evidence_budget_per_choice"]) == 5
            and len(set(row["evidence_budget_per_choice"])) == 1
            for row in ordered
        ),
        "causal_why_strategy_purity": all(
            row.get("route_id") == ROUTE_ID
            and row.get("handler_id") == handler_id
            and row.get("strategy_id") == strategy_id
            for row in ordered
        ),
    }
    audit = {
        "schema_version": "nextqa_causal_why_coherent_chain_qa_audit_v1",
        "run_id": run_id,
        "strategy_id": strategy_id,
        "route_id": ROUTE_ID,
        "handler_id": handler_id,
        "answer_mode": answer_mode,
        "expected_question_count": len(expected_ids),
        "prediction_record_count": len(ordered),
        "completed_question_count": len(completed),
        "error_question_count": len(errors),
        "missing_question_ids": [qid for qid in expected_ids if qid not in by_id],
        "unexpected_question_ids": sorted(set(by_id) - set(expected_ids)),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "uses_gold_answer": False,
        "uses_official_type": False,
        "qa_receives_images": False,
    }
    atomic_write_json(run_root / "predictions.json", ordered)
    atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(
        run_root / "summary.json",
        {key: audit[key] for key in (
            "schema_version", "run_id", "strategy_id", "route_id", "handler_id",
            "answer_mode", "expected_question_count", "prediction_record_count",
            "completed_question_count", "error_question_count", "missing_question_ids",
            "failed_checks",
        )},
    )
    atomic_write_text(
        run_root / "summary.md",
        "\n".join([
            "# NExT-QA causal_why coherent-chain QA",
            "", f"- Run: `{run_id}`", f"- Strategy: `{strategy_id}`",
            f"- Expected: `{len(expected_ids)}`", f"- Completed: `{len(completed)}`",
            f"- Errors: `{len(errors)}`", f"- Failed checks: `{audit['failed_checks']}`", "",
        ]),
    )
    return audit


def main() -> int:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    strategy_id, handler_id = _strategy_spec(args.strategy_version)
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint-every must be positive")
    question_path = inputs['question_ir']
    graph_path = inputs['frozen_graph_evidence']
    visual_path = inputs['visual_evidence']
    config_path = inputs['config']
    questions_all = _load_list(question_path, "question IR")
    graph_all = _load_list(graph_path, "frozen graph evidence")
    visual_all = _load_list(visual_path, "coherent-chain visual evidence")
    questions = _index([row for row in questions_all if str(row.get("intent")) == ROUTE_ID], "causal_why question IR")
    graphs = _index([row for row in graph_all if str(row.get("question_id")) in questions], "causal_why graph evidence")
    visuals = _index(visual_all, "coherent-chain visual evidence")
    if set(questions) != set(graphs) or set(questions) != set(visuals):
        raise ValueError("causal_why question, graph, and v3 visual evidence IDs must match")
    question_ids = sorted(questions, key=lambda qid: (str(questions[qid]["video_id"]), qid))
    if len(question_ids) != len(visuals):
        raise ValueError("causal_why question count does not match v3 visual evidence")
    config = load_visual_mcq_config(config_path)
    for question_id in question_ids:
        validate_visual_qa_inputs(questions[question_id], graphs[question_id], visuals[question_id], config=config)
        if visuals[question_id].get("strategy_id") != strategy_id or visuals[question_id].get("handler_id") != handler_id:
            raise ValueError(f"strategy identity mismatch: {question_id}")
    key_pool = parse_api_key_pool(os.getenv(config.transport.api_key_pool_env_var, ""))
    if args.answer_mode == "live" and not key_pool and not os.getenv(config.transport.api_key_env_var, "").strip():
        raise RuntimeError("live QA requires the configured API key environment variable or key pool")
    run_root = prepare_stage_run_root("qa_pilots", args.run_id, resume=args.resume)
    contract = {
        "dataset": "NExT-QA", "split": "val", "stage": f"causal_why_coherent_chain_mcq_qa_{args.strategy_version}",
        "run_id": args.run_id, "strategy_id": strategy_id, "route_id": ROUTE_ID,
        "handler_id": handler_id, "question_ir": fingerprint_file(question_path),
        "frozen_graph_evidence": fingerprint_file(graph_path), "visual_evidence": fingerprint_file(visual_path),
        "config": fingerprint_file(config_path), "question_count": len(question_ids),
        "video_count": len({str(row["video_id"]) for row in questions.values()}), "answer_mode": args.answer_mode,
        "provider": config.transport.provider, "model": config.transport.model,
        "qa_profile": config.transport.profile_name, "workers": args.workers,
        "checkpoint_every_questions": args.checkpoint_every, "timeout_sec": config.transport.timeout_sec,
        "max_attempts": config.transport.max_attempts, "graph_evidence_limit": config.graph_evidence_limit,
        "visual_evidence_limit": config.visual_evidence_limit, "choice_count": 5,
        "qa_receives_images": False, "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False, "uses_gold_answer": False,
        "uses_official_type": False, "calls_api": args.answer_mode == "live",
        "resume_contract": "preserve_completed_retry_error_only_v3", "output_root": str(run_root),
    }
    persist_run_config(run_root / "run_config.json", contract=contract, resume=args.resume, schema_version="nextqa_causal_why_coherent_chain_qa_run_config_v1")
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / "source_snapshot.json", source_paths=sources, resume=args.resume)
    predictions_path = run_root / "predictions.json"
    existing = _load_list(predictions_path, "existing predictions") if args.resume and predictions_path.is_file() else []
    existing_by_id = _index(existing, "existing predictions") if existing else {}
    current_by_id = {qid: row for qid, row in existing_by_id.items() if row.get("status") == "completed"}
    pending = [qid for qid in question_ids if qid not in current_by_id]
    pool_size = max(1, len(key_pool) or 1)

    def process(question_id: str) -> dict[str, Any]:
        result = answer_visual_mcq(
            questions[question_id], graphs[question_id], visuals[question_id],
            config=config, answer_mode=args.answer_mode,
            api_key_slot=stable_api_key_slot(question_id, pool_size),
        )
        result["schema_version"] = f"nextqa_causal_why_coherent_chain_mcq_prediction_{args.strategy_version}"
        result["strategy_id"] = strategy_id
        result["strategy_scope"] = "route_wide"
        result["route_id"] = ROUTE_ID
        result["handler_id"] = handler_id
        return result

    def checkpoint(offset: int, question_id: str, result: dict[str, Any]) -> None:
        current_by_id[question_id] = result
        if offset % args.checkpoint_every == 0 or offset == len(pending):
            _write_checkpoint(run_root, list(current_by_id.values()), expected_ids=question_ids, run_id=args.run_id, answer_mode=args.answer_mode, strategy_id=strategy_id, handler_id=handler_id)
        print(json.dumps({"progress": f"{offset}/{len(pending)}", "question_id": question_id, "status": result["status"], "attempt_count": result["attempt_count"]}, ensure_ascii=False), flush=True)

    if args.workers == 1:
        for offset, question_id in enumerate(pending, start=1):
            checkpoint(offset, question_id, process(question_id))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, question_id): question_id for question_id in pending}
            for offset, future in enumerate(as_completed(futures), start=1):
                question_id = futures[future]
                checkpoint(offset, question_id, future.result())
    audit = _write_checkpoint(run_root, list(current_by_id.values()), expected_ids=question_ids, run_id=args.run_id, answer_mode=args.answer_mode, strategy_id=strategy_id, handler_id=handler_id)
    drift = persist_drift_report(run_root / "source_drift_report.json", snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    valid = not audit["missing_question_ids"] and not audit["failed_checks"] and not drift["code_drift_detected"]
    status = "completed" if valid and audit["error_question_count"] == 0 else ("completed_with_errors" if valid else "invalid")
    atomic_write_json(run_root / "run_status.json", {"status": status, "run_id": args.run_id, "completed_at_utc": utc_now(), "completed_question_count": audit["completed_question_count"], "error_question_count": audit["error_question_count"], "code_drift_detected": drift["code_drift_detected"], "uses_gold_answer": False})
    print(json.dumps({"status": status, "run_id": args.run_id, "question_count": len(question_ids), "completed": audit["completed_question_count"], "errors": audit["error_question_count"]}, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
