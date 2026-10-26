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
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.baseline import compare_prediction_record_ids, freeze_graph_records
DEFAULT_RETRIEVAL = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'graph_retrieval' / 'nextqa_graph_retrieval_v2_full4996_20260812a' / 'retrieval_records.json'
DEFAULT_PREDICTIONS = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'qa' / 'nextqa_graph_qa_luna_pool_full4996_20260814c' / 'predictions.json'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Freeze the existing unified graph retrieval evidence for visual QA pairing.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retrieval-records', type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument('--graph-only-predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root('visual_graph_baseline', args.run_id, resume=args.resume)
    retrieval_path = args.retrieval_records.resolve()
    prediction_path = args.graph_only_predictions.resolve()
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'freeze_route_neutral_graph_evidence', 'retrieval_records': fingerprint_file(retrieval_path), 'graph_only_predictions': fingerprint_file(prediction_path), 'recomputes_graph_retrieval': False, 'uses_gold_answer': False, 'uses_official_type': False, 'uses_visual_index': False, 'calls_api': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_graph_baseline_run_config_v1')
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=source_paths, resume=args.resume)
    retrieval_rows = _load_rows(retrieval_path, 'retrieval records')
    predictions = _load_rows(prediction_path, 'graph-only predictions')
    frozen = freeze_graph_records(retrieval_rows)
    comparison = compare_prediction_record_ids(frozen, predictions)
    status_distribution = Counter((str(row['status']) for row in frozen))
    profile_distribution = Counter((str(row['graph_retrieval_profile']) for row in frozen if row['status'] == 'completed'))
    audit = {'schema_version': 'nextqa_visual_graph_baseline_audit_v1', 'run_id': args.run_id, 'question_count': len(frozen), 'video_count': len({str(row['video_id']) for row in frozen}), 'status_distribution': dict(sorted(status_distribution.items())), 'graph_retrieval_profile_distribution': dict(sorted(profile_distribution.items())), 'prediction_record_id_comparison': comparison, 'recomputes_graph_retrieval': False, 'uses_gold_answer': False, 'uses_official_type': False, 'uses_visual_index': False, 'calls_api': False, 'status': 'passed' if comparison['passed'] else 'failed'}
    atomic_write_json(run_root / 'frozen_graph_evidence.json', frozen)
    atomic_write_json(run_root / 'baseline_audit.json', audit)
    atomic_write_json(run_root / 'baseline_manifest.json', {'schema_version': 'nextqa_visual_graph_baseline_manifest_v1', 'run_id': args.run_id, 'frozen_graph_evidence_path': str((run_root / 'frozen_graph_evidence.json').resolve()), 'source_retrieval_records_path': str(retrieval_path), 'source_retrieval_records_sha256': contract['retrieval_records']['sha256'], 'source_graph_only_predictions_path': str(prediction_path), 'source_graph_only_predictions_sha256': contract['graph_only_predictions']['sha256'], 'question_count': len(frozen), 'video_count': audit['video_count'], 'graph_retrieval_frozen': True, 'changes_graph_retrieval': False, 'uses_gold_answer': False})
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=source_paths)
    if drift['code_drift_detected']:
        audit['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'baseline_audit.json', audit)
    atomic_write_json(run_root / 'run_status.json', {'status': audit['status'], 'question_count': len(frozen), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if audit['status'] != 'passed':
        raise SystemExit(f'graph baseline freeze failed: {audit}')
    print(json.dumps(audit, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
