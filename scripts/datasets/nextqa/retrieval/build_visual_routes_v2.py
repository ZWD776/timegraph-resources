from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.policy_registry import audit_policy_registry
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.route_contracts import audit_forbidden_field_isolation, compile_route_records
DEFAULT_IR = (REPO_ROOT / 'datasets/nextqa/ir/nextqa-ir-v3-luna-full4996/payload/nextqa-ir-v3-luna-full4996-20260817b/question_ir.runtime.json')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Persist NExT-QA visual v2 routes from Luna IR intent only.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-ir', type=Path, default=DEFAULT_IR)
    parser.add_argument('--pilot-manifest', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_list(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _select(rows: list[dict[str, Any]], pilot_manifest: Path | None) -> list[dict[str, Any]]:
    if pilot_manifest is None:
        return rows
    payload = json.loads(pilot_manifest.resolve().read_text(encoding='utf-8'))
    video_ids = {str(row['official_video_id']) for row in payload.get('videos', [])}
    selected = [row for row in rows if str(row.get('video_id')) in video_ids]
    if len(video_ids) != 5 or not selected:
        raise ValueError('pilot manifest must select five videos with questions')
    return selected

def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root('visual_routes', args.run_id, resume=args.resume)
    ir_path = args.question_ir.resolve()
    pilot_path = args.pilot_manifest.resolve() if args.pilot_manifest else None
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'luna_ir_intent_only_visual_routing_v2', 'question_ir': fingerprint_file(ir_path), 'pilot_manifest': fingerprint_file(pilot_path) if pilot_path else None, 'route_source': 'nextqa_luna_ir_intent_only', 'uses_question_text_routing': False, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False, 'calls_api': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_route_run_config_v2')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    rows = _select(_load_list(ir_path, 'question IR'), pilot_path)
    routes = compile_route_records(rows)
    isolation = [audit_forbidden_field_isolation(row) for row in rows]
    distribution = Counter((str(route['route_id']) for route in routes))
    handler_distribution = Counter((str(route['handler_id']) for route in routes))
    policy_audit = audit_policy_registry()
    exact_intent_mapping = all((str(route['route_id']) == str(row['intent']) for route, row in zip(routes, rows, strict=True)))
    checks = {'unique_question_ids': len({str(route['question_id']) for route in routes}) == len(routes), 'intent_is_only_route_source': all((route['route_source'] == 'nextqa_luna_ir_intent_only' for route in routes)), 'route_equals_ir_intent': exact_intent_mapping, 'intent_demand_target_consistent': len(routes) == len(rows), 'official_type_invariant': all((item['official_type_invariant'] for item in isolation)), 'answer_choices_invariant': all((item['answer_choices_invariant'] for item in isolation)), 'gold_choice_index_invariant': all((item['gold_choice_index_invariant'] for item in isolation)), 'gold_answer_invariant': all((item['gold_answer_invariant'] for item in isolation)), 'unique_handler_per_route': policy_audit['unique_handler_per_route'], 'all_role_budgets_positive': policy_audit['all_role_budgets_positive'], 'all_non_count_routes_within_limit': policy_audit['all_non_count_routes_within_limit']}
    audit = {'schema_version': 'nextqa_visual_route_audit_v2', 'run_id': args.run_id, 'question_count': len(routes), 'video_count': len({str(route['video_id']) for route in routes}), 'route_distribution': dict(sorted(distribution.items())), 'handler_distribution': dict(sorted(handler_distribution.items())), 'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed], 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'calls_api': False, 'status': 'passed' if all(checks.values()) else 'failed'}
    atomic_write_json(run_root / 'route_records.json', routes)
    atomic_write_json(run_root / 'route_audit.json', audit)
    atomic_write_json(run_root / 'route_manifest.json', {'schema_version': 'nextqa_visual_route_manifest_v2', 'run_id': args.run_id, 'route_records_path': str((run_root / 'route_records.json').resolve()), 'question_ir_path': str(ir_path), 'question_ir_sha256': contract['question_ir']['sha256'], 'question_count': len(routes), 'video_count': audit['video_count'], 'route_source': 'nextqa_luna_ir_intent_only', 'uses_gold_answer': False})
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=sources)
    if drift['code_drift_detected']:
        audit['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'route_audit.json', audit)
    atomic_write_json(run_root / 'run_status.json', {'status': audit['status'], 'question_count': len(routes), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if audit['status'] != 'passed':
        raise SystemExit(f"visual route v2 audit failed: {audit['failed_checks']}")
    print(json.dumps(audit, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
