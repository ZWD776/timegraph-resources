from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
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
)
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.strategies import merge_route_prediction_overlay
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.strategies.strategy_identity import (
    validate_route_strategy_identity,
)




from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


def resolve_inputs(args):
    run_config = NEXTQA_SCRATCH_ROOT / 'qa' / args.run_id / "run_config.json"
    selected = {}
    selected['base_predictions'] = recorded_or_selected(args.base_predictions, resume=args.resume, run_config=run_config, field='base_predictions', selected=lambda: asset_file("nextqa", 'runs', 'visual_base', REPO_ROOT))
    selected['route_records'] = recorded_or_selected(args.route_records, resume=args.resume, run_config=run_config, field='route_records', selected=lambda: asset_file("nextqa", 'retrieval', 'routes', REPO_ROOT))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge one complete NExT-QA route strategy into the frozen visual v2 predictions."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overlay-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, default=None)
    parser.add_argument("--route-records", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} must be a JSON list of objects")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    inputs = resolve_inputs(args)
    base_path = inputs['base_predictions']
    overlay_path = args.overlay_predictions.resolve()
    route_path = inputs['route_records']
    base_rows = _load_rows(base_path, "base predictions")
    overlay_rows = _load_rows(overlay_path, "overlay predictions")
    route_rows = _load_rows(route_path, "route records")
    strategy_ids = {str(row.get("strategy_id") or "") for row in overlay_rows}
    target_route_ids = {
        str(row.get("target_route_id") or row.get("route_id") or "")
        for row in overlay_rows
    }
    handler_ids = {str(row.get("handler_id") or "") for row in overlay_rows}
    if len(strategy_ids) != 1 or len(target_route_ids) != 1 or len(handler_ids) != 1:
        raise ValueError("overlay predictions do not have one fixed strategy identity")
    strategy_id = strategy_ids.pop()
    target_route_id = target_route_ids.pop()
    handler_id = handler_ids.pop()
    strategy_family = validate_route_strategy_identity(
        strategy_id, target_route_id, handler_id
    )
    run_root = prepare_stage_run_root("qa", args.run_id, resume=args.resume)
    contract = {
        "dataset": "NExT-QA",
        "split": "val",
        "stage": "fixed_route_strategy_prediction_overlay_v1",
        "run_id": args.run_id,
        "strategy_id": strategy_id,
        "target_route_id": target_route_id,
        "handler_id": handler_id,
        "strategy_family": strategy_family,
        "base_predictions": fingerprint_file(base_path),
        "overlay_predictions": fingerprint_file(overlay_path),
        "route_records": fingerprint_file(route_path),
        "expected_question_count": 4996,
        "selects_better_answer_per_question": False,
        "uses_official_type": False,
        "uses_gold_answer": False,
        "calls_api": False,
    }
    persist_run_config(
        run_root / "run_config.json",
        contract=contract,
        resume=args.resume,
        schema_version="nextqa_fixed_route_strategy_overlay_run_config_v1",
    )
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(
        run_root / "source_snapshot.json", source_paths=sources, resume=args.resume
    )
    merged, audit, provenance = merge_route_prediction_overlay(
        base_rows,
        overlay_rows,
        route_rows,
        target_route_id=target_route_id,
        strategy_id=strategy_id,
        handler_id=handler_id,
        expected_question_count=4996,
    )
    audit["run_id"] = args.run_id
    atomic_write_json(run_root / "predictions.json", merged)
    atomic_write_json(run_root / "overlay_audit.json", audit)
    atomic_write_json(run_root / "prediction_provenance.json", provenance)
    atomic_write_json(
        run_root / "overlay_manifest.json",
        {
            "schema_version": "nextqa_fixed_route_strategy_overlay_manifest_v1",
            "run_id": args.run_id,
            "predictions_path": str((run_root / "predictions.json").resolve()),
            "prediction_provenance_path": str(
                (run_root / "prediction_provenance.json").resolve()
            ),
            "base_predictions_path": str(base_path),
            "overlay_predictions_path": str(overlay_path),
            "route_records_path": str(route_path),
            "strategy_id": strategy_id,
            "target_route_id": target_route_id,
            "question_count": len(merged),
            "selects_better_answer_per_question": False,
            "uses_gold_answer": False,
        },
    )
    atomic_write_text(
        run_root / "summary.md",
        "\n".join(
            [
                "# NExT-QA Fixed Route Strategy Overlay",
                "",
                f"- Run: `{args.run_id}`",
                f"- Strategy: `{strategy_id}`",
                f"- Target route: `{target_route_id}`",
                f"- Replaced questions: `{audit['overlay_question_count']}`",
                f"- Unchanged base questions: `{audit['retained_base_question_count']}`",
                f"- Non-target unchanged: `{audit['non_target_predictions_unchanged']}`",
                f"- Per-question answer selection: `{audit['selects_better_answer_per_question']}`",
                "",
            ]
        ),
    )
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    drift = persist_drift_report(
        run_root / "source_drift_report.json",
        snapshot=snapshot,
        current_source_paths=current_sources,
    )
    if drift["code_drift_detected"]:
        audit["status"] = "invalid_code_drift"
        atomic_write_json(run_root / "overlay_audit.json", audit)
    atomic_write_json(
        run_root / "run_status.json",
        {
            "status": audit["status"],
            "question_count": len(merged),
            "calls_api": False,
            "code_drift_detected": drift["code_drift_detected"],
        },
    )
    if audit["status"] != "passed":
        raise SystemExit("route strategy prediction overlay failed")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
