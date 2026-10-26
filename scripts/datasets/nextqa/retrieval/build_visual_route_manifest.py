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
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.routing import audit_route_field_isolation, compile_route_records
DEFAULT_QUESTION_IR = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'question_ir' / 'nextqa_question_ir_v2_full4996_20260812a' / 'question_ir.runtime.json'
DEFAULT_PILOT_MANIFEST = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'dry_run' / 'pilot5' / 'nextqa_val_pilot5_manifest.json'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Persist question-only NExT-QA EvidenceDemand routes without QA or API calls.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-ir', type=Path, default=DEFAULT_QUESTION_IR)
    parser.add_argument('--pilot-manifest', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_list(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _select_pilot(rows: list[dict[str, Any]], manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None:
        return rows
    manifest = json.loads(manifest_path.resolve().read_text(encoding='utf-8'))
    expected = {str(question_id) for video in manifest.get('videos', []) for question_id in video.get('question_ids', [])}
    selected = [row for row in rows if str(row.get('question_id')) in expected]
    actual = {str(row['question_id']) for row in selected}
    if actual != expected:
        raise ValueError('pilot route selection does not match the fixed pilot question IDs')
    return selected

def _contains_forbidden_runtime_key(value: Any) -> bool:
    forbidden = {'official_type', 'answer_choices', 'gold', 'gold_answer', 'gold_choice_index'}
    if isinstance(value, dict):
        return any((str(key).lower() in forbidden or _contains_forbidden_runtime_key(item) for key, item in value.items()))
    if isinstance(value, list):
        return any((_contains_forbidden_runtime_key(item) for item in value))
    return False

def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root('visual_routes', args.run_id, resume=args.resume)
    question_ir_path = args.question_ir.resolve()
    pilot_manifest_path = args.pilot_manifest.resolve() if args.pilot_manifest else None
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'question_only_visual_routing', 'question_ir': fingerprint_file(question_ir_path), 'pilot_manifest': fingerprint_file(pilot_manifest_path) if pilot_manifest_path else None, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'calls_api': False, 'changes_graph_retrieval': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_route_run_config_v1')
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=source_paths, resume=args.resume)
    rows = _select_pilot(_load_list(question_ir_path, 'question IR'), pilot_manifest_path)
    routes = compile_route_records(rows)
    isolation = [audit_route_field_isolation(row) for row in rows]
    distribution = Counter((str(route['route_id']) for route in routes))
    audit = {'schema_version': 'nextqa_visual_route_audit_v1', 'run_id': args.run_id, 'question_count': len(routes), 'video_count': len({str(route['video_id']) for route in routes}), 'route_distribution': dict(sorted(distribution.items())), 'unique_question_ids': len({str(route['question_id']) for route in routes}) == len(routes), 'official_type_invariant': all((check['official_type_invariant'] for check in isolation)), 'answer_choices_invariant': all((check['answer_choices_invariant'] for check in isolation)), 'gold_choice_index_invariant': all((check['gold_choice_index_invariant'] for check in isolation)), 'gold_answer_invariant': all((check['gold_answer_invariant'] for check in isolation)), 'route_output_contains_forbidden_runtime_key': _contains_forbidden_runtime_key(routes), 'changes_graph_retrieval': False, 'calls_api': False, 'status': 'passed'}
    failed = [key for key, value in audit.items() if key.endswith('_invariant') and value is not True]
    if failed or not audit['unique_question_ids'] or audit['route_output_contains_forbidden_runtime_key']:
        audit['status'] = 'failed'
    atomic_write_json(run_root / 'route_records.json', routes)
    atomic_write_json(run_root / 'route_audit.json', audit)
    atomic_write_json(run_root / 'route_manifest.json', {'schema_version': 'nextqa_visual_route_manifest_v1', 'run_id': args.run_id, 'route_records_path': str((run_root / 'route_records.json').resolve()), 'question_count': len(routes), 'video_count': audit['video_count'], 'question_ir_path': str(question_ir_path), 'question_ir_sha256': contract['question_ir']['sha256'], 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False})
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=source_paths)
    if drift['code_drift_detected']:
        audit['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'route_audit.json', audit)
    atomic_write_json(run_root / 'run_status.json', {'status': audit['status'], 'question_count': len(routes), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if audit['status'] != 'passed':
        raise SystemExit(f'route audit failed: {audit}')
    print(json.dumps(audit, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
