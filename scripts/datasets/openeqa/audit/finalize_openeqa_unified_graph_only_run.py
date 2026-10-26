"""Validate graph-only execution shards and materialize one OpenEQA run."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import load_open_eqa_rows, repo_path, write_json_array_atomic, write_json_atomic
OUTPUT_CONTRACT = 'openeqa_unified_graph_only_qa_run.v1'
ROUTE_CONTRACT = 'openeqa_unified_question_route_ir.v1'
SHARD_CONFIG_CONTRACT = 'openeqa_hm3d_graph_only_qa_run_config_v1'
EXPECTED_MODEL = 'gpt-5.6-luna'
EXPECTED_PROVIDER = 'openai_compatible'
EXPECTED_STATUS = 'live_validated'
COMMON_CONFIG_FIELDS = ('open_eqa_json', 'question_ir_json', 'skip_question_ir', 'answer_mode', 'retrieval_mode', 'graph_retrieval_profile', 'graph_retrieval_contract', 'retrieval_limit', 'qa_workers', 'max_tokens', 'vector_model')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate graph-only source shards and write one complete, gold-free OpenEQA prediction set.')
    parser.add_argument('--open-eqa-json', type=Path, default=(REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--question-route-json', type=Path, required=True)
    parser.add_argument('--shard-run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f'Expected a JSON list: {path}')
    return [item for item in payload if isinstance(item, dict)]

def _question_id(row: dict[str, Any]) -> str:
    return str(row.get('question_id') or row.get('uuid') or '').strip()

def _data_source(row: dict[str, Any]) -> str:
    return str(row.get('episode_history') or '').partition('/')[0]

def _route_id(ir: dict[str, Any]) -> str:
    route = ir.get('question_route') if isinstance(ir, dict) else None
    return str(route.get('route_id') or '') if isinstance(route, dict) else ''

def _config_signature(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple((json.dumps(config.get(field), ensure_ascii=False, sort_keys=True) for field in COMMON_CONFIG_FIELDS))

def _load_shard(run_dir: Path) -> dict[str, Any]:
    required = {'run_config': run_dir / 'run_config.json', 'summary': run_dir / 'summary.json', 'audit': run_dir / 'audit.json', 'predictions': run_dir / 'predictions.json'}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f'Incomplete graph-only shard {run_dir}; missing={missing}')
    return {'run_dir': str(run_dir), 'run_config': _load_json(required['run_config']), 'summary': _load_json(required['summary']), 'audit': _load_list(required['audit']), 'predictions': _load_list(required['predictions'])}

def validate_and_merge(*, dataset_rows: list[dict[str, Any]], route_payload: dict[str, Any], route_path: Path, shards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    issues: list[str] = []
    expected_by_id = {_question_id(row): row for row in dataset_rows if _question_id(row)}
    expected_sources = sorted({_data_source(row) for row in dataset_rows if _data_source(row)})
    route_records = {_question_id(record): record.get('llm_question_ir') or record.get('question_ir') or {} for record in route_payload.get('records', []) or [] if isinstance(record, dict) and _question_id(record)}
    if route_payload.get('contract') != ROUTE_CONTRACT:
        issues.append(f'question route contract must be {ROUTE_CONTRACT}')
    if route_payload.get('dataset') != 'OpenEQA':
        issues.append('question route dataset must be OpenEQA')
    if route_payload.get('uses_gold_answer') is not False:
        issues.append('question route manifest must declare uses_gold_answer=false')
    if route_payload.get('runtime_route_recompile_allowed') is not False:
        issues.append('question route manifest must forbid runtime recompilation')
    if set(route_records) != set(expected_by_id):
        issues.append('question route IDs do not exactly cover OpenEQA')
    configs = [shard.get('run_config') or {} for shard in shards]
    for config in configs:
        source = str(config.get('data_source') or '')
        if config.get('contract') != SHARD_CONFIG_CONTRACT:
            issues.append(f'{source}: invalid graph-only run_config contract')
        if config.get('skip_question_ir') is not False:
            issues.append(f'{source}: question IR was skipped')
        if config.get('answer_mode') != 'live':
            issues.append(f'{source}: answer_mode must be live')
        configured_route = Path(str(config.get('question_ir_json') or ''))
        if configured_route.resolve() != route_path.resolve():
            issues.append(f'{source}: question route manifest drift')
    if len({_config_signature(config) for config in configs}) != 1:
        issues.append('execution shards do not share one graph-only configuration')
    actual_sources = [str(config.get('data_source') or '') for config in configs]
    if len(actual_sources) != len(set(actual_sources)):
        issues.append(f'duplicate execution shard sources: {actual_sources}')
    if sorted(actual_sources) != expected_sources:
        issues.append(f'execution shard coverage mismatch actual={sorted(actual_sources)} expected={expected_sources}')
    merged_audit: list[dict[str, Any]] = []
    merged_predictions: list[dict[str, Any]] = []
    seen_audit: set[str] = set()
    seen_predictions: set[str] = set()
    source_diagnostics: dict[str, dict[str, Any]] = {}
    for shard in shards:
        config = shard.get('run_config') or {}
        source = str(config.get('data_source') or '')
        audit = shard.get('audit') or []
        predictions = shard.get('predictions') or []
        expected_source_ids = {question_id for question_id, row in expected_by_id.items() if _data_source(row) == source}
        audit_by_id = {_question_id(item): item for item in audit if _question_id(item)}
        prediction_by_id = {_question_id(item): item for item in predictions if _question_id(item)}
        source_diagnostics[source] = {'run_dir': shard.get('run_dir'), 'expected_count': len(expected_source_ids), 'audit_count': len(audit), 'prediction_count': len(predictions), 'error_count': sum((1 for item in audit if item.get('error')))}
        if len(audit_by_id) != len(audit):
            issues.append(f'{source}: audit contains missing or duplicate IDs')
        if len(prediction_by_id) != len(predictions):
            issues.append(f'{source}: predictions contain missing or duplicate IDs')
        if set(audit_by_id) != expected_source_ids:
            issues.append(f'{source}: audit does not exactly cover its OpenEQA shard')
        if set(prediction_by_id) != expected_source_ids:
            issues.append(f'{source}: predictions do not exactly cover its OpenEQA shard')
        for question_id, item in audit_by_id.items():
            if question_id in seen_audit:
                issues.append(f'duplicate audit ID across shards: {question_id}')
                continue
            seen_audit.add(question_id)
            if item.get('error'):
                issues.append(f'{question_id}: QA error is present')
            if item.get('graph_rag_status') != EXPECTED_STATUS:
                issues.append(f'{question_id}: graph_rag_status is not {EXPECTED_STATUS}')
            if item.get('model') != EXPECTED_MODEL:
                issues.append(f'{question_id}: model is not {EXPECTED_MODEL}')
            if item.get('provider') != EXPECTED_PROVIDER:
                issues.append(f'{question_id}: provider is not {EXPECTED_PROVIDER}')
            contract = item.get('graph_only_contract') or {}
            expected_contract = {'uses_hsgg_derived_graph': True, 'uses_images_or_crops': False, 'uses_visual_faiss': False, 'uses_gold_answer_for_prediction': False}
            if contract != expected_contract:
                issues.append(f'{question_id}: graph-only contract drift')
            actual_ir = item.get('question_ir') or {}
            expected_ir = route_records.get(question_id) or {}
            if _route_id(actual_ir) != _route_id(expected_ir):
                issues.append(f'{question_id}: question route drift')
            prediction = prediction_by_id.get(question_id) or {}
            if str(prediction.get('answer') or '') != str(item.get('prediction') or ''):
                issues.append(f'{question_id}: audit/prediction answer mismatch')
            merged_audit.append(item)
        for question_id, item in prediction_by_id.items():
            if question_id in seen_predictions:
                issues.append(f'duplicate prediction ID across shards: {question_id}')
                continue
            seen_predictions.add(question_id)
            merged_predictions.append(item)
    if seen_audit != set(expected_by_id):
        issues.append('merged audit does not exactly cover OpenEQA')
    if seen_predictions != set(expected_by_id):
        issues.append('merged predictions do not exactly cover OpenEQA')
    order = {question_id: index for index, question_id in enumerate(expected_by_id)}
    merged_audit.sort(key=lambda item: order.get(_question_id(item), len(order)))
    merged_predictions.sort(key=lambda item: order.get(_question_id(item), len(order)))
    statuses = Counter((str(item.get('graph_rag_status') or '') for item in merged_audit))
    route_counts = Counter((_route_id(item.get('question_ir') or {}) for item in merged_audit))
    summary = {'contract': OUTPUT_CONTRACT, 'dataset': 'OpenEQA', 'status': 'complete' if not issues else 'invalid', 'generated_at': datetime.now(timezone.utc).isoformat(), 'question_count': len(dataset_rows), 'audit_count': len(merged_audit), 'prediction_count': len(merged_predictions), 'error_count': sum((1 for item in merged_audit if item.get('error'))), 'model': EXPECTED_MODEL, 'provider': EXPECTED_PROVIDER, 'uses_gold_answer': False, 'uses_images_or_crops': False, 'data_source_counts': dict(sorted(Counter((_data_source(row) for row in dataset_rows)).items())), 'data_sources_are_execution_shards_only': True, 'graph_rag_status_counts': dict(sorted(statuses.items())), 'question_route_id_counts': dict(sorted(route_counts.items())), 'common_config': {field: configs[0].get(field) if configs else None for field in COMMON_CONFIG_FIELDS}, 'source_diagnostics': dict(sorted(source_diagnostics.items())), 'contract_issue_count': len(issues), 'contract_issues': issues}
    return (merged_predictions, merged_audit, summary, issues)

def main() -> None:
    args = build_parser().parse_args()
    dataset_path = repo_path(args.open_eqa_json)
    route_path = repo_path(args.question_route_json)
    output_dir = repo_path(args.output_dir)
    shard_dirs = [repo_path(path) for path in args.shard_run_dir]
    predictions, audit, summary, issues = validate_and_merge(dataset_rows=load_open_eqa_rows(dataset_path), route_payload=_load_json(route_path), route_path=route_path, shards=[_load_shard(path) for path in shard_dirs])
    summary.update({'open_eqa_json': str(dataset_path), 'question_route_json': str(route_path), 'shard_run_dirs': [str(path) for path in shard_dirs]})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_array_atomic(output_dir / 'predictions.json', predictions)
    write_json_array_atomic(output_dir / 'audit.json', audit)
    write_json_atomic(output_dir / 'summary.json', summary)
    write_json_atomic(output_dir / 'run_manifest.json', summary)
    print(f"status={summary['status']} questions={summary['question_count']} predictions={summary['prediction_count']} issues={len(issues)}")
    if issues:
        for issue in issues[:20]:
            print(f'- {issue}')
        raise SystemExit(2)
if __name__ == '__main__':
    main()
