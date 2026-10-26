"""Validate execution shards and materialize one formal OpenEQA QA run."""
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
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_CONTRACT as SINGLE_ROUTE_CONTRACT, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5
OUTPUT_CONTRACT = 'openeqa_unified_qa_run.v1'
UNIFIED_ROUTE_CONTRACT = 'openeqa_unified_question_route_ir.v1'
SHARD_CONFIG_CONTRACT = 'openeqa_evidence_demand_visual_qa_run_config_v1'
COMMON_CONFIG_FIELDS = ('open_eqa_json', 'question_ir_json', 'route_subset_manifest', 'route_subset_primary_need', 'route_subset_strategy_variants', 'answer_mode', 'qa_config', 'qa_max_tokens', 'retrieval_mode', 'retrieval_limit', 'graph_retrieval_profile', 'graph_retrieval_contract', 'vector_model', 'visual_index_name', 'visual_backbone', 'visual_runtime_contract', 'visual_top_k', 'visual_min_score', 'visual_max_candidate_packs', 'visual_max_evidence_lines', 'uses_gold_answer')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate source execution shards and write one complete OpenEQA prediction/audit/summary contract.')
    parser.add_argument('--open-eqa-json', type=Path, default=(REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--question-route-json', type=Path, required=True)
    parser.add_argument('--route-subset-manifest', type=Path, default=None)
    parser.add_argument('--strategy-variant', action='append', default=None)
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

def _route_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in payload.get('records', []) or []:
        if not isinstance(record, dict):
            continue
        question_id = _question_id(record)
        ir = record.get('llm_question_ir') or record.get('question_ir') or {}
        if question_id and isinstance(ir, dict):
            if question_id in result:
                raise ValueError(f'Duplicate question route ID: {question_id}')
            result[question_id] = ir
    return result

def expected_rows_for_run(dataset_rows: list[dict[str, Any]], route_subset: dict[str, Any] | None, strategy_variants: list[str] | None=None) -> list[dict[str, Any]]:
    if route_subset is None:
        return list(dataset_rows)
    if route_subset.get('uses_gold_answer') is not False:
        raise ValueError('route subset must declare uses_gold_answer=false')
    requested_variants = {str(value).strip() for value in strategy_variants or [] if str(value).strip()}
    subset_ids = {_question_id(record) for record in route_subset.get('records') or [] if isinstance(record, dict) and _question_id(record) and (not requested_variants or str(record.get('strategy_variant') or '').strip() in requested_variants)}
    return [row for row in dataset_rows if _question_id(row) in subset_ids]

def load_shard_run(run_dir: Path) -> dict[str, Any]:
    required = {'run_config': run_dir / 'run_config.json', 'summary': run_dir / 'summary.json', 'audit': run_dir / 'audit.json', 'predictions': run_dir / 'predictions.json'}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f'Incomplete shard run {run_dir}; missing={missing}')
    return {'run_dir': str(run_dir), 'run_config': _load_json(required['run_config']), 'summary': _load_json(required['summary']), 'audit': _load_list(required['audit']), 'predictions': _load_list(required['predictions'])}

def _config_signature(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple((json.dumps(config.get(field), ensure_ascii=False, sort_keys=True) for field in COMMON_CONFIG_FIELDS))

def validate_and_merge(*, dataset_rows: list[dict[str, Any]], question_route_payload: dict[str, Any], route_subset_payload: dict[str, Any] | None, shards: list[dict[str, Any]], strategy_variants: list[str] | None=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    issues: list[str] = []
    expected_rows = expected_rows_for_run(dataset_rows, route_subset_payload, strategy_variants)
    expected_by_id = {_question_id(row): row for row in expected_rows}
    expected_sources = sorted({_data_source(row) for row in expected_rows if _data_source(row)})
    routes = _route_records(question_route_payload)
    if question_route_payload.get('contract') != UNIFIED_ROUTE_CONTRACT:
        issues.append(f'question route contract must be {UNIFIED_ROUTE_CONTRACT}')
    if question_route_payload.get('dataset') != 'OpenEQA':
        issues.append('question route dataset must be OpenEQA')
    if question_route_payload.get('uses_gold_answer') is not False:
        issues.append('question route manifest must declare uses_gold_answer=false')
    if question_route_payload.get('runtime_route_recompile_allowed') is not False:
        issues.append('question route manifest must forbid runtime recompilation')
    is_single_route = question_route_payload.get('routing_profile') == ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5
    if is_single_route and question_route_payload.get('route_contract') != SINGLE_ROUTE_CONTRACT:
        issues.append(f'single-route manifest route_contract must be {SINGLE_ROUTE_CONTRACT}')
    if is_single_route and route_subset_payload is not None:
        issues.append('V5 single-route execution does not accept route subsets')
    if is_single_route and strategy_variants:
        issues.append('V5 single-route execution does not accept strategy_variant filters')
    if route_subset_payload is not None:
        subset_records = route_subset_payload.get('records') or []
        subset_ids = [_question_id(record) for record in subset_records if isinstance(record, dict) and _question_id(record)]
        dataset_ids = {_question_id(row) for row in dataset_rows}
        if len(subset_ids) != len(set(subset_ids)):
            issues.append('route subset contains duplicate question IDs')
        unknown_subset_ids = sorted(set(subset_ids) - dataset_ids)
        if unknown_subset_ids:
            issues.append(f'route subset contains {len(unknown_subset_ids)} IDs outside OpenEQA; sample={unknown_subset_ids[:8]}')
    if set(expected_by_id) - set(routes):
        missing = sorted(set(expected_by_id) - set(routes))
        issues.append(f'question route manifest missing {len(missing)} IDs; sample={missing[:8]}')
    configs = [shard.get('run_config') or {} for shard in shards]
    for config in configs:
        if config.get('contract') != SHARD_CONFIG_CONTRACT:
            issues.append(f"invalid shard run_config contract for {config.get('data_source')}")
        if config.get('uses_gold_answer') is not False:
            issues.append(f"{config.get('data_source')}: uses_gold_answer must be false")
    signatures = {_config_signature(config) for config in configs}
    if len(signatures) != 1:
        issues.append('execution shards do not share one complete QA/retrieval/visual configuration')
    actual_sources = [str(config.get('data_source') or '') for config in configs]
    if len(actual_sources) != len(set(actual_sources)):
        issues.append(f'duplicate execution shard sources: {actual_sources}')
    if sorted(actual_sources) != expected_sources:
        issues.append(f'execution shard source coverage mismatch: actual={sorted(actual_sources)} expected={expected_sources}')
    merged_audit: list[dict[str, Any]] = []
    merged_predictions: list[dict[str, Any]] = []
    seen_audit: set[str] = set()
    seen_predictions: set[str] = set()
    source_summaries: dict[str, dict[str, Any]] = {}
    for shard in shards:
        config = shard.get('run_config') or {}
        source = str(config.get('data_source') or '')
        expected_source_ids = {question_id for question_id, row in expected_by_id.items() if _data_source(row) == source}
        audit = shard.get('audit') or []
        predictions = shard.get('predictions') or []
        shard_summary = shard.get('summary') or {}
        source_summaries[source] = {'run_dir': shard.get('run_dir'), 'expected_count': len(expected_source_ids), 'audit_count': len(audit), 'prediction_count': len(predictions), 'error_count': sum((1 for item in audit if item.get('error')))}
        if shard_summary.get('error_count') != source_summaries[source]['error_count']:
            issues.append(f'{source}: summary error_count disagrees with audit')
        audit_ids = {_question_id(item) for item in audit if _question_id(item)}
        prediction_ids = {_question_id(item) for item in predictions if _question_id(item)}
        prediction_by_id = {_question_id(item): item for item in predictions if _question_id(item)}
        if len(audit_ids) != len(audit):
            issues.append(f'{source}: audit contains missing or duplicate question IDs')
        if len(prediction_ids) != len(predictions):
            issues.append(f'{source}: predictions contain missing or duplicate question IDs')
        if audit_ids != expected_source_ids:
            missing = sorted(expected_source_ids - audit_ids)
            extra = sorted(audit_ids - expected_source_ids)
            issues.append(f'{source}: audit coverage mismatch missing={len(missing)} extra={len(extra)}')
        if prediction_ids != expected_source_ids:
            missing = sorted(expected_source_ids - prediction_ids)
            extra = sorted(prediction_ids - expected_source_ids)
            issues.append(f'{source}: prediction coverage mismatch missing={len(missing)} extra={len(extra)}')
        for item in audit:
            question_id = _question_id(item)
            if question_id in seen_audit:
                issues.append(f'duplicate audit question ID across shards: {question_id}')
                continue
            seen_audit.add(question_id)
            if item.get('error'):
                issues.append(f'{question_id}: QA error is present')
            expected_row = expected_by_id.get(question_id, {})
            if expected_row and _data_source(expected_row) != source:
                issues.append(f'{question_id}: stored in wrong source shard {source}')
            route = routes.get(question_id, {}).get('question_route') or {}
            if is_single_route:
                expected_route_id = str(route.get('route_id') or '')
                expected_handler = str(route.get('handler_id') or '')
                actual_demand = item.get('evidence_demand') or {}
                if str(actual_demand.get('route_id') or '') != expected_route_id:
                    issues.append(f"{question_id}: route_id drift actual={actual_demand.get('route_id')!r} expected={expected_route_id!r}")
                if actual_demand.get('primary_need') or actual_demand.get('strategy_variant'):
                    issues.append(f'{question_id}: V5 evidence demand leaked second-layer fields')
                if str(item.get('route_handler_id') or '') != expected_handler:
                    issues.append(f"{question_id}: handler drift actual={item.get('route_handler_id')!r} expected={expected_handler!r}")
            else:
                expected_need = str(route.get('primary_need') or '')
                expected_variant = str(route.get('strategy_variant') or '')
                actual_need = str((item.get('evidence_demand') or {}).get('primary_need') or '')
                actual_variant = str(item.get('strategy_variant') or '')
                if actual_need != expected_need:
                    issues.append(f'{question_id}: evidence need drift actual={actual_need!r} expected={expected_need!r}')
                if actual_variant != expected_variant:
                    issues.append(f'{question_id}: strategy variant drift actual={actual_variant!r} expected={expected_variant!r}')
            if item.get('runtime_route_recompiled') is not False:
                issues.append(f'{question_id}: runtime route recompilation detected')
            if item.get('question_route_decision_source') != 'question_ir.question_route':
                issues.append(f'{question_id}: persisted route was not the decision source')
            if item.get('graph_retrieval_profile') != config.get('graph_retrieval_profile'):
                issues.append(f'{question_id}: graph retrieval profile drift')
            if item.get('visual_backbone') != config.get('visual_backbone'):
                issues.append(f'{question_id}: visual backbone drift')
            handler_field = 'route_handler_id' if is_single_route else 'route_strategy_handler'
            if not str(item.get(handler_field) or '').strip():
                issues.append(f'{question_id}: {handler_field} is missing')
            prediction_item = prediction_by_id.get(question_id) or {}
            persisted_answer = str(prediction_item.get('answer') or prediction_item.get('prediction') or '')
            if persisted_answer != str(item.get('prediction') or ''):
                issues.append(f'{question_id}: audit/prediction answer mismatch')
            merged_audit.append(item)
        for item in predictions:
            question_id = _question_id(item)
            if question_id in seen_predictions:
                issues.append(f'duplicate prediction question ID across shards: {question_id}')
                continue
            seen_predictions.add(question_id)
            merged_predictions.append(item)
    order = {_question_id(row): index for index, row in enumerate(dataset_rows)}
    merged_audit.sort(key=lambda item: order.get(_question_id(item), len(order)))
    merged_predictions.sort(key=lambda item: order.get(_question_id(item), len(order)))
    if seen_audit != set(expected_by_id):
        issues.append('merged audit does not exactly cover the selected OpenEQA IDs')
    if seen_predictions != set(expected_by_id):
        issues.append('merged predictions do not exactly cover the selected OpenEQA IDs')
    counts = Counter((_data_source(row) for row in expected_rows))
    common_config = {field: configs[0].get(field) if configs else None for field in COMMON_CONFIG_FIELDS}
    summary = {'contract': OUTPUT_CONTRACT, 'dataset': 'OpenEQA', 'status': 'complete' if not issues else 'invalid', 'generated_at': datetime.now(timezone.utc).isoformat(), 'question_count': len(expected_rows), 'audit_count': len(merged_audit), 'prediction_count': len(merged_predictions), 'error_count': sum((1 for item in merged_audit if item.get('error'))), 'data_source_counts': dict(sorted(counts.items())), 'data_sources_are_execution_shards_only': True, 'common_config': common_config, 'source_diagnostics': dict(sorted(source_summaries.items())), 'contract_issue_count': len(issues), 'contract_issues': issues, 'route_subset_strategy_variants': sorted(strategy_variants or [])}
    return (merged_predictions, merged_audit, summary, issues)

def main() -> None:
    args = build_parser().parse_args()
    dataset_path = repo_path(args.open_eqa_json)
    route_path = repo_path(args.question_route_json)
    subset_path = repo_path(args.route_subset_manifest) if args.route_subset_manifest else None
    shard_dirs = [repo_path(path) for path in args.shard_run_dir]
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, audit, summary, issues = validate_and_merge(dataset_rows=load_open_eqa_rows(dataset_path), question_route_payload=_load_json(route_path), route_subset_payload=_load_json(subset_path) if subset_path else None, shards=[load_shard_run(path) for path in shard_dirs], strategy_variants=args.strategy_variant)
    summary.update({'open_eqa_json': str(dataset_path), 'question_route_json': str(route_path), 'route_subset_manifest': str(subset_path or ''), 'shard_run_dirs': [str(path) for path in shard_dirs]})
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
