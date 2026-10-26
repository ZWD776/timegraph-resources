"""Build isolated question-only Luna IR for NExT-QA validation questions."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.retrieval.llm_question_ir import (
    ANCHOR_REQUIRED_INTENTS,
    OFFICIAL_TYPE_LABELS,
    OFFICIAL_TYPE_TO_INTENT,
    PROFILE_NAME,
    audit_prompt_isolation,
    compile_question_ir,
    load_llm_question_ir_config,
    parse_api_key_pool,
    request_fingerprint,
    semantic_only_view,
    stable_api_key_slot,
)
from vlkg.datasets.nextqa.retrieval.question_ir import retrieval_input_from_ir
from vlkg.datasets.nextqa.runtime.run_snapshot import (
    collect_imported_repo_sources,
    fingerprint_file,
    persist_drift_report,
    persist_run_config,
    persist_source_snapshot,
    utc_now,
)


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
PROJECT_PYTHON = Path(sys.executable).resolve()
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'question_ir' / args.run_id / 'run_config.json'
    return {
        'questions': recorded_or_selected(args.questions, resume=args.resume, run_config=run_config,
            field='questions', selected=lambda: asset_file('nextqa', 'source', 'questions', REPO_ROOT)),
        'config': recorded_or_selected(args.config, resume=args.resume, run_config=run_config,
            field='config', selected=lambda: REPO_ROOT / 'configs/datasets/nextqa/question_ir.yaml'),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--scope", choices=("smoke", "pilot", "full"), default="full"
    )
    parser.add_argument(
        "--question-id",
        help="Required for smoke scope; forbidden for pilot/full scopes.",
    )
    parser.add_argument("--mode", choices=("live", "mock"), default="live")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--seed-records",
        type=Path,
        default=None,
        help="Import only completed, request-hash-compatible v3 records into a new run.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def _index_unique(
    rows: list[dict[str, Any]],
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id or question_id in output:
            raise ValueError(f"{name} has an empty or duplicate question ID")
        output[question_id] = row
    return output


def select_pending_question_ids(
    question_ids: list[str],
    existing_records: list[Mapping[str, Any]],
) -> list[str]:
    completed = {
        str(row.get("question_id") or "")
        for row in existing_records
        if row.get("status") == "completed"
    }
    return [question_id for question_id in question_ids if question_id not in completed]


def load_seed_completed_records(
    path: Path,
    *,
    question_index: Mapping[str, Mapping[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_rows(path, name="seed IR records")
    indexed = _index_unique(rows, name="seed IR records")
    unexpected = sorted(set(indexed) - set(question_index))
    if unexpected:
        raise ValueError("seed IR contains question IDs outside the current input")
    completed: list[dict[str, Any]] = []
    skipped_errors = 0
    for question_id, row in indexed.items():
        if row.get("status") != "completed":
            skipped_errors += 1
            continue
        if row.get("ir_builder") != PROFILE_NAME or row.get("model") != model:
            raise ValueError(f"seed IR profile/model mismatch: {question_id}")
        if row.get("uses_gold_answer") is not False or row.get("rule_fallback_used"):
            raise ValueError(f"seed IR purity failed: {question_id}")
        expected_hash = request_fingerprint(question_index[question_id])
        if row.get("request_sha256") != expected_hash:
            raise ValueError(f"seed IR request fingerprint mismatch: {question_id}")
        completed.append(dict(row))
    return completed, {
        "schema_version": "nextqa_question_only_llm_ir_seed_audit_v1",
        "source_path": str(path.resolve()),
        "source_record_count": len(rows),
        "imported_completed_count": len(completed),
        "skipped_error_count": skipped_errors,
        "unexpected_question_id_count": len(unexpected),
        "request_fingerprint_verified": True,
        "uses_gold_answer": False,
    }


def select_scope_rows(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    question_id: str | None,
) -> list[dict[str, Any]]:
    if scope == "smoke":
        selected_id = str(question_id or "").strip()
        if not selected_id:
            raise ValueError("smoke scope requires --question-id")
        selected = [
            row for row in rows if str(row.get("question_id") or "") == selected_id
        ]
        if len(selected) != 1:
            raise ValueError("smoke question ID must match exactly one input row")
        return selected
    if question_id is not None:
        raise ValueError("--question-id is allowed only with smoke scope")
    return rows


def _api_key_pool_size(config: Any, *, mode: str) -> int:
    if mode != "live":
        return 0
    keys = parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ""))
    if keys:
        return len(keys)
    return int(bool(os.getenv(config.api_key_env_var, "").strip()))


def _checkpoint(
    run_root: Path,
    records: list[dict[str, Any]],
    *,
    materialize_views: bool = False,
) -> None:
    ordered = sorted(records, key=lambda row: str(row["question_id"]))
    completed = [row for row in ordered if row.get("status") == "completed"]
    errors = [row for row in ordered if row.get("status") == "error"]
    atomic_write_json(run_root / "question_ir.records.json", ordered)
    atomic_write_json(run_root / "errors.json", errors)
    if materialize_views:
        semantic = [semantic_only_view(row) for row in completed]
        retrieval = [retrieval_input_from_ir(row).to_dict() for row in completed]
        atomic_write_json(run_root / "question_ir.runtime.json", completed)
        atomic_write_json(run_root / "question_ir.semantic.json", semantic)
        atomic_write_json(run_root / "question_ir.retrieval.json", retrieval)


def _diagnostic_summary(
    runtime_records: list[dict[str, Any]],
) -> dict[str, Any]:
    cross_tab: dict[str, Counter[str]] = {}
    agreement = 0
    recognized = 0
    for row in runtime_records:
        official_type = str(row.get("official_type") or "")
        intent = str(row.get("intent") or "")
        cross_tab.setdefault(official_type, Counter())[intent] += 1
        expected = OFFICIAL_TYPE_TO_INTENT.get(official_type)
        if expected is not None:
            recognized += 1
            agreement += int(intent == expected)
    return {
        "note": (
            "Official type was joined only after LLM IR persistence and was never "
            "included in an API request."
        ),
        "recognized_question_count": recognized,
        "expected_intent_agreement_count": agreement,
        "expected_intent_agreement_rate": agreement / recognized if recognized else 0.0,
        "official_type_labels": OFFICIAL_TYPE_LABELS,
        "cross_tab": {
            key: dict(sorted(values.items()))
            for key, values in sorted(cross_tab.items())
        },
    }


def _request_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "question_id",
        "status",
        "model",
        "request_sha256",
        "attempt_count",
        "api_key_slot",
        "api_key_pool_size",
        "response_metadata",
        "error_stage",
        "error_type",
        "error_message",
    )
    return [
        {field: row.get(field) for field in fields if field in row}
        for row in sorted(records, key=lambda item: str(item["question_id"]))
    ]


def _summary_markdown(audit: Mapping[str, Any]) -> str:
    diagnostic = audit["official_type_diagnostic_only"]
    lines = [
        "# NExT-QA Question-only Luna IR v3",
        "",
        f"- Status: `{audit['status']}`",
        f"- Questions: `{audit['question_count']}`",
        f"- Completed/errors: `{audit['completed_question_count']}/{audit['error_question_count']}`",
        f"- Videos: `{audit['video_count']}`",
        f"- Model: `{audit['model']}`",
        f"- Prompt isolation: `{audit['prompt_isolation_passed']}`",
        f"- Rule fallback count: `{audit['rule_fallback_count']}`",
        (
            "- Post-persistence official-type intent agreement: "
            f"`{diagnostic['expected_intent_agreement_rate']:.4%}` "
            f"({diagnostic['expected_intent_agreement_count']}/"
            f"{diagnostic['recognized_question_count']})"
        ),
        "",
        "## Intent Distribution",
        "",
    ]
    for intent, count in audit["intent_distribution"].items():
        lines.append(f"- `{intent}`: {count}")
    lines.extend(["", "## Official Type Diagnostic", ""])
    for code, values in diagnostic["cross_tab"].items():
        label = diagnostic["official_type_labels"].get(code, "unknown")
        rendered = ", ".join(
            f"{intent}={count}" for intent, count in values.items()
        )
        lines.append(f"- `{code}` ({label}): {rendered}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    if Path(sys.executable).resolve() != PROJECT_PYTHON:
        raise RuntimeError(
            f"NExT-QA jobs must use the project virtual environment: {PROJECT_PYTHON}"
        )
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint-every must be positive")

    inputs = resolve_inputs(args)
    questions_path = inputs['questions']
    config_path = inputs['config']
    config = load_llm_question_ir_config(config_path)
    rows = select_scope_rows(
        _load_rows(questions_path, name="runtime questions"),
        scope=args.scope,
        question_id=args.question_id,
    )
    expected_count, expected_videos = {
        "smoke": (1, 1),
        "pilot": (39, 5),
        "full": (4996, 570),
    }[args.scope]
    question_index = _index_unique(rows, name="runtime questions")
    question_ids = list(question_index)
    video_ids = {str(row.get("video_id") or "") for row in rows}
    if len(rows) != expected_count or len(video_ids) != expected_videos:
        raise ValueError(
            f"{args.scope} LLM IR requires {expected_count} questions over "
            f"{expected_videos} videos"
        )

    run_root = prepare_stage_run_root("question_ir", args.run_id, resume=args.resume)
    key_pool_size = _api_key_pool_size(config, mode=args.mode)
    contract = {
        "dataset": "NExT-QA",
        "split": "val",
        "stage": "question_only_llm_ir",
        "scope": args.scope,
        "selected_question_id": args.question_id if args.scope == "smoke" else None,
        "seed_records": (
            fingerprint_file(args.seed_records.resolve()) if args.seed_records else None
        ),
        "run_id": args.run_id,
        "questions": fingerprint_file(questions_path),
        "config": fingerprint_file(config_path),
        "ir_builder": PROFILE_NAME,
        "mode": args.mode,
        "provider": config.provider,
        "model": config.model,
        "endpoint_url": config.endpoint_url() if args.mode == "live" else None,
        "api_key_env_var": config.api_key_env_var if args.mode == "live" else None,
        "api_key_pool_env_var": (
            config.api_key_pool_env_var if args.mode == "live" else None
        ),
        "api_key_pool_size": key_pool_size if args.mode == "live" else None,
        "workers": args.workers,
        "timeout_sec": config.timeout_sec,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "max_attempts": config.max_attempts,
        "request_inputs": ["question_id", "normalized_question"],
        "reads_official_type": False,
        "reads_answer_choices": False,
        "reads_gold_answer": False,
        "reads_graph": False,
        "reads_images": False,
        "uses_visual_index": False,
        "output_root": str(run_root),
        "resume_policy": "preserve_completed_retry_error_only_v1",
        "rule_fallback_policy": "forbidden_in_live_runs",
    }
    persist_run_config(
        run_root / "run_config.json",
        contract=contract,
        resume=args.resume,
        schema_version="nextqa_question_only_llm_ir_run_config_v1",
    )
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    source_snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json",
        source_paths=source_paths,
        resume=args.resume,
    )

    records_path = run_root / "question_ir.records.json"
    seed_completed: list[dict[str, Any]] = []
    if args.seed_records:
        seed_completed, seed_audit = load_seed_completed_records(
            args.seed_records.resolve(),
            question_index=question_index,
            model=config.model,
        )
        atomic_write_json(run_root / "seed_migration_audit.json", seed_audit)
    existing = (
        _load_rows(records_path, name="existing IR records")
        if args.resume and records_path.is_file()
        else []
    )
    existing_index = _index_unique(existing, name="existing IR records") if existing else {}
    completed_by_id = {
        str(row["question_id"]): row for row in seed_completed
    }
    completed_by_id.update({
        question_id: row
        for question_id, row in existing_index.items()
        if row.get("status") == "completed"
    })
    current_by_id = dict(completed_by_id)
    pending = select_pending_question_ids(
        question_ids,
        [*seed_completed, *existing],
    )

    isolation_checks = [audit_prompt_isolation(row) for row in rows]
    prompt_isolation_passed = all(
        all(check.values()) for check in isolation_checks
    )
    if not prompt_isolation_passed:
        raise RuntimeError("NExT-QA LLM IR prompt isolation failed before API execution")

    def process(question_id: str) -> dict[str, Any]:
        return compile_question_ir(
            question_index[question_id],
            config=config,
            mode=args.mode,
            api_key_slot=stable_api_key_slot(
                question_id,
                max(1, key_pool_size),
            ),
        )

    def record_result(offset: int, question_id: str, result: dict[str, Any]) -> None:
        current_by_id[question_id] = result
        if offset % args.checkpoint_every == 0 or offset == len(pending):
            _checkpoint(run_root, list(current_by_id.values()))

    if args.workers == 1:
        for offset, question_id in enumerate(pending, start=1):
            record_result(offset, question_id, process(question_id))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, question_id): question_id
                for question_id in pending
            }
            for offset, future in enumerate(as_completed(futures), start=1):
                question_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "schema_version": "nextqa_question_only_llm_ir_v3",
                        "status": "error",
                        "question_id": question_id,
                        "video_id": str(question_index[question_id].get("video_id") or ""),
                        "ir_builder": PROFILE_NAME,
                        "model": config.model,
                        "error_stage": "worker_execution",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "uses_gold_answer": False,
                        "rule_fallback_used": False,
                    }
                record_result(offset, question_id, result)

    _checkpoint(
        run_root,
        list(current_by_id.values()),
        materialize_views=True,
    )
    ordered = sorted(current_by_id.values(), key=lambda row: str(row["question_id"]))
    completed = [row for row in ordered if row.get("status") == "completed"]
    errors = [row for row in ordered if row.get("status") == "error"]
    output_ids = [str(row["question_id"]) for row in ordered]
    semantic_rows = [semantic_only_view(row) for row in completed]
    retrieval_rows = [retrieval_input_from_ir(row).to_dict() for row in completed]
    diagnostic = _diagnostic_summary(completed)
    rule_fallback_count = sum(bool(row.get("rule_fallback_used")) for row in ordered)
    anchor_completeness = sum(
        bool((row.get("anchor_event") or {}).get("text"))
        for row in completed
        if row.get("intent") in ANCHOR_REQUIRED_INTENTS
    )
    anchor_required_count = sum(
        row.get("intent") in ANCHOR_REQUIRED_INTENTS for row in completed
    )
    audit: dict[str, Any] = {
        "schema_version": "nextqa_question_only_llm_ir_audit_v1",
        "run_id": args.run_id,
        "status": "passed" if not errors else "completed_with_errors",
        "mode": args.mode,
        "model": config.model,
        "question_count": len(rows),
        "completed_question_count": len(completed),
        "error_question_count": len(errors),
        "video_count": len(video_ids),
        "duplicate_output_question_id_count": len(output_ids) - len(set(output_ids)),
        "missing_question_ids": sorted(set(question_ids) - set(output_ids)),
        "intent_distribution": dict(
            sorted(Counter(str(row["intent"]) for row in completed).items())
        ),
        "answer_target_distribution": dict(
            sorted(Counter(str(row["answer_target"]) for row in completed).items())
        ),
        "visual_evidence_demand_distribution": dict(
            sorted(
                Counter(
                    str(row["visual_evidence_demand"]) for row in completed
                ).items()
            )
        ),
        "anchor_required_question_count": anchor_required_count,
        "anchor_complete_question_count": anchor_completeness,
        "prompt_isolation_passed": prompt_isolation_passed,
        "prompt_invariant_fields": sorted(isolation_checks[0]) if isolation_checks else [],
        "semantic_output_contains_official_type": any(
            "official_type" in row for row in semantic_rows
        ),
        "semantic_output_contains_answer_choices": any(
            "answer_choices" in row for row in semantic_rows
        ),
        "retrieval_output_contains_official_type": any(
            "official_type" in row for row in retrieval_rows
        ),
        "retrieval_output_contains_answer_choices": any(
            "answer_choices" in row for row in retrieval_rows
        ),
        "retrieval_output_contains_visual_evidence_demand": any(
            "visual_evidence_demand" in row for row in retrieval_rows
        ),
        "rule_fallback_count": rule_fallback_count,
        "official_type_diagnostic_only": diagnostic,
        "calls_api": args.mode == "live",
        "reads_graph": False,
        "reads_images": False,
        "uses_visual_index": False,
        "uses_gold_answer": False,
    }
    purity_passed = (
        audit["duplicate_output_question_id_count"] == 0
        and not audit["missing_question_ids"]
        and audit["prompt_isolation_passed"]
        and not audit["semantic_output_contains_official_type"]
        and not audit["semantic_output_contains_answer_choices"]
        and not audit["retrieval_output_contains_official_type"]
        and not audit["retrieval_output_contains_answer_choices"]
        and not audit["retrieval_output_contains_visual_evidence_demand"]
        and audit["rule_fallback_count"] == 0
        and anchor_completeness == anchor_required_count
    )
    if not purity_passed:
        audit["status"] = "failed"

    atomic_write_json(run_root / "request_audit.json", _request_audit_rows(ordered))
    atomic_write_json(run_root / "question_ir_audit.json", audit)
    atomic_write_json(
        run_root / "question_ir_manifest.json",
        {
            "schema_version": "nextqa_question_only_llm_ir_manifest_v1",
            "run_id": args.run_id,
            "dataset": "NExT-QA",
            "split": "val",
            "source_questions": str(questions_path),
            "records_path": str(run_root / "question_ir.records.json"),
            "runtime_path": str(run_root / "question_ir.runtime.json"),
            "semantic_path": str(run_root / "question_ir.semantic.json"),
            "retrieval_path": str(run_root / "question_ir.retrieval.json"),
            "question_count": len(rows),
            "completed_question_count": len(completed),
            "error_question_count": len(errors),
            "uses_gold_answer": False,
        },
    )
    atomic_write_json(run_root / "summary.json", audit)
    atomic_write_text(run_root / "summary.md", _summary_markdown(audit))

    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    drift = persist_drift_report(
        run_root / "source_drift_report.json",
        snapshot=source_snapshot,
        current_source_paths=current_sources,
    )
    if drift["code_drift_detected"] or not purity_passed:
        status_name = "invalid"
    elif errors:
        status_name = "completed_with_errors"
    else:
        status_name = "completed"
    status = {
        "status": status_name,
        "run_id": args.run_id,
        "completed_at_utc": utc_now(),
        "question_count": len(rows),
        "completed_question_count": len(completed),
        "error_question_count": len(errors),
        "code_drift_detected": drift["code_drift_detected"],
        "prompt_isolation_passed": prompt_isolation_passed,
        "rule_fallback_count": rule_fallback_count,
        "uses_gold_answer": False,
    }
    atomic_write_json(run_root / "run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 1 if status_name == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
