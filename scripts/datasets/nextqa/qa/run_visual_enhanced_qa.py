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




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / ("qa_pilots" if args.scope == "pilot" else "qa") / args.run_id / "run_config.json"
    selected = {}
    selected['question_ir'] = recorded_or_selected(args.question_ir, resume=args.resume, run_config=run_config, field='question_ir_path', selected=lambda: asset_file("nextqa", 'ir', 'runtime', REPO_ROOT))
    selected['frozen_graph_evidence'] = recorded_or_selected(args.frozen_graph_evidence, resume=args.resume, run_config=run_config, field='frozen_graph_evidence_path', selected=lambda: asset_file("nextqa", 'retrieval', 'frozen_graph', REPO_ROOT))
    selected['visual_evidence'] = recorded_or_selected(args.visual_evidence, resume=args.resume, run_config=run_config, field='visual_evidence_path', selected=lambda: asset_file("nextqa", 'retrieval', 'visual_evidence', REPO_ROOT))
    selected['config'] = recorded_or_selected(args.config, resume=args.resume, run_config=run_config, field='config_path', selected=lambda: REPO_ROOT / "configs/datasets/nextqa/visual_qa.yaml")
    if args.scope == "pilot":
        if args.pilot_manifest is None:
            raise ValueError("pilot scope requires an explicit --pilot-manifest")
        selected["pilot_manifest"] = args.pilot_manifest.resolve()
    else:
        selected["pilot_manifest"] = None
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated text-only NExT-QA graph plus visual evidence QA.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--question-ir", type=Path, default=None)
    parser.add_argument("--frozen-graph-evidence", type=Path)
    parser.add_argument("--visual-evidence", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--pilot-manifest", type=Path, default=None)
    parser.add_argument("--answer-mode", choices=("mock", "live"), default="live")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in result:
            raise ValueError(f"{name} has an empty or duplicate question ID: {question_id}")
        result[question_id] = row
    return result


def _pilot_video_ids(path: Path) -> set[str]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    ids = {str(row["official_video_id"]) for row in payload.get("videos", [])}
    if len(ids) != 5:
        raise ValueError("pilot manifest must contain five videos")
    return ids


def _write_checkpoint(
    run_root: Path,
    rows: list[dict[str, Any]],
    *,
    expected_ids: list[str],
    run_id: str,
    answer_mode: str,
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
            and len(set(row["evidence_budget_per_choice"])) == 1
            and len(row["evidence_budget_per_choice"]) == 5
            for row in ordered
        ),
        "route_handler_purity": all(
            row.get("handler_id") == f"nextqa_visual_handler_v2:{row.get('route_id')}"
            for row in ordered
        ),
    }
    audit = {
        "schema_version": "nextqa_visual_enhanced_mcq_audit_v2", "run_id": run_id,
        "answer_mode": answer_mode, "expected_question_count": len(expected_ids),
        "prediction_record_count": len(ordered), "completed_question_count": len(completed),
        "error_question_count": len(errors),
        "missing_question_ids": [qid for qid in expected_ids if qid not in by_id],
        "unexpected_question_ids": sorted(set(by_id) - set(expected_ids)),
        "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "uses_gold_answer": False, "qa_receives_images": False,
    }
    atomic_write_json(run_root / "predictions.json", ordered)
    atomic_write_json(run_root / "audit.json", audit)
    atomic_write_json(
        run_root / "summary.json",
        {key: audit[key] for key in (
            "schema_version", "run_id", "answer_mode", "expected_question_count",
            "prediction_record_count", "completed_question_count", "error_question_count",
            "missing_question_ids", "failed_checks",
        )},
    )
    atomic_write_text(
        run_root / "summary.md",
        "\n".join([
            "# NExT-QA Visual-Enhanced QA",
            "", f"- Run: `{run_id}`", f"- Expected: `{len(expected_ids)}`",
            f"- Completed: `{len(completed)}`", f"- Errors: `{len(errors)}`",
            f"- Missing: `{len(audit['missing_question_ids'])}`",
            f"- Failed purity checks: `{audit['failed_checks']}`", "",
        ]),
    )
    return audit


def main() -> int:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint interval must be positive")
    question_path = inputs['question_ir']
    graph_path = inputs['frozen_graph_evidence']
    visual_path = inputs['visual_evidence']
    config_path = inputs['config']
    pilot_path = inputs['pilot_manifest']
    config = load_visual_mcq_config(config_path)
    questions_all = _load_rows(question_path, "question IR")
    graph_all = _load_rows(graph_path, "frozen graph evidence")
    visual_all = _load_rows(visual_path, "visual evidence")
    if args.scope == "pilot":
        video_ids = _pilot_video_ids(pilot_path)
        questions_all = [row for row in questions_all if str(row["video_id"]) in video_ids]
        graph_all = [row for row in graph_all if str(row["video_id"]) in video_ids]
        visual_all = [row for row in visual_all if str(row["video_id"]) in video_ids]
    questions = _index(questions_all, "question IR")
    graphs = _index(graph_all, "frozen graph evidence")
    visuals = _index(visual_all, "visual evidence")
    if set(questions) != set(graphs) or set(questions) != set(visuals):
        raise ValueError("question, frozen graph, and visual evidence ID sets must match")
    question_ids = sorted(questions, key=lambda qid: (str(questions[qid]["video_id"]), qid))
    expected = 39 if args.scope == "pilot" else 4996
    if len(question_ids) != expected:
        raise ValueError(f"{args.scope} scope requires {expected} questions, found {len(question_ids)}")
    for question_id in question_ids:
        validate_visual_qa_inputs(
            questions[question_id], graphs[question_id], visuals[question_id], config=config
        )
    run_root = prepare_stage_run_root(
        "qa_pilots" if args.scope == "pilot" else "qa", args.run_id, resume=args.resume
    )
    key_pool = parse_api_key_pool(os.getenv(config.transport.api_key_pool_env_var, ""))
    if args.answer_mode == "live" and not key_pool and not os.getenv(config.transport.api_key_env_var, "").strip():
        raise RuntimeError("live visual QA requires OPENAI_API_KEYS or OPENAI_API_KEY")
    contract = {
        "dataset": "NExT-QA", "split": "val", "stage": "text_only_graph_plus_visual_mcq_v2",
        "scope": args.scope, "run_id": args.run_id,
        "question_ir_path": str(question_path), "question_ir_sha256": sha256_file(question_path),
        "frozen_graph_evidence_path": str(graph_path), "frozen_graph_evidence_sha256": sha256_file(graph_path),
        "visual_evidence_path": str(visual_path), "visual_evidence_sha256": sha256_file(visual_path),
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "pilot_manifest_path": str(pilot_path) if args.scope == "pilot" else None,
        "question_count": len(question_ids), "video_count": len({str(row["video_id"]) for row in questions_all}),
        "answer_mode": args.answer_mode, "provider": config.transport.provider,
        "model": config.transport.model, "qa_profile": config.transport.profile_name,
        "endpoint_url": config.transport.endpoint_url() if args.answer_mode == "live" else None,
        "api_key_pool_size": len(key_pool) if args.answer_mode == "live" else None,
        "workers": args.workers, "timeout_sec": config.transport.timeout_sec,
        "max_attempts": config.transport.max_attempts,
        "graph_evidence_limit": config.graph_evidence_limit,
        "visual_evidence_limit": config.visual_evidence_limit,
        "choice_count": 5, "choice_scoring_policy": "single_shared_graph_visual_text_prompt_symmetric_choices_v2",
        "checkpoint_every_questions": args.checkpoint_every,
        "qa_receives_images": False, "qa_receives_image_paths": False,
        "qa_receives_visual_embeddings": False, "uses_official_type": False,
        "uses_gold_answer": False, "output_root": str(run_root),
        "resume_contract": "preserve_completed_retry_error_only_v2",
    }
    persist_run_config(
        run_root / "run_config.json", contract=contract, resume=args.resume,
        schema_version="nextqa_visual_enhanced_mcq_run_config_v2",
    )
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json", source_paths=sources, resume=args.resume
    )
    predictions_path = run_root / "predictions.json"
    existing = _load_rows(predictions_path, "existing predictions") if args.resume and predictions_path.is_file() else []
    existing_by_id = _index(existing, "existing predictions") if existing else {}
    current_by_id = {
        question_id: row for question_id, row in existing_by_id.items()
        if row.get("status") == "completed"
    }
    pending = [question_id for question_id in question_ids if question_id not in current_by_id]
    pool_size = max(1, len(key_pool) or 1)

    def process(question_id: str) -> dict[str, Any]:
        return answer_visual_mcq(
            questions[question_id], graphs[question_id], visuals[question_id],
            config=config, answer_mode=args.answer_mode,
            api_key_slot=stable_api_key_slot(question_id, pool_size),
        )

    def record(offset: int, question_id: str, result: dict[str, Any]) -> None:
        current_by_id[question_id] = result
        if offset % args.checkpoint_every == 0 or offset == len(pending):
            _write_checkpoint(
                run_root, list(current_by_id.values()), expected_ids=question_ids,
                run_id=args.run_id, answer_mode=args.answer_mode,
            )
        print(json.dumps({"progress": f"{offset}/{len(pending)}", "question_id": question_id,
                          "status": result["status"], "attempt_count": result["attempt_count"]},
                         ensure_ascii=False), flush=True)

    if args.workers == 1:
        for offset, question_id in enumerate(pending, start=1):
            record(offset, question_id, process(question_id))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, question_id): question_id for question_id in pending}
            for offset, future in enumerate(as_completed(futures), start=1):
                question_id = futures[future]
                record(offset, question_id, future.result())
    audit = _write_checkpoint(
        run_root, list(current_by_id.values()), expected_ids=question_ids,
        run_id=args.run_id, answer_mode=args.answer_mode,
    )
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    drift = persist_drift_report(
        run_root / "source_drift_report.json", snapshot=snapshot,
        current_source_paths=current_sources,
    )
    valid = (
        audit["prediction_record_count"] == len(question_ids)
        and not audit["missing_question_ids"] and not audit["failed_checks"]
        and not drift["code_drift_detected"]
    )
    status_name = "invalid" if not valid else (
        "completed" if audit["error_question_count"] == 0 else "completed_with_errors"
    )
    status = {
        "status": status_name, "run_id": args.run_id, "completed_at_utc": utc_now(),
        "completed_question_count": audit["completed_question_count"],
        "error_question_count": audit["error_question_count"],
        "code_drift_detected": drift["code_drift_detected"], "uses_gold_answer": False,
    }
    atomic_write_json(run_root / "run_status.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    return 1 if status_name == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
