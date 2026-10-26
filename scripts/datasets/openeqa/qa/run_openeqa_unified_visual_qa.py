"""Run OpenEQA as one dataset with source-specific execution resources."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import load_open_eqa_rows, repo_path, write_json_atomic
from vlkg.core.config.config import load_domain_config
from vlkg.datasets.openeqa.retrieval.evidence_router import IR_FIRST_ROUTING_PROFILES, question_route_decision_issues
from vlkg.datasets.openeqa.runtime.execution_contract import ir_first_execution_config_issues, SINGLE_ROUTE_VISUAL_BACKBONE
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import route_handler_contract_issues
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_CONTRACT as SINGLE_ROUTE_CONTRACT, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, route_handler_contract_issues as single_route_handler_contract_issues, single_route_decision_issues
RESOURCE_CONTRACT = 'openeqa_unified_execution_resources.v1'
RUN_PLAN_CONTRACT = 'openeqa_unified_visual_qa_plan.v1'
UNIFIED_ROUTE_CONTRACT = 'openeqa_unified_question_route_ir.v1'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Execute one OpenEQA visual QA configuration over all source resource shards, then materialize one parent result.')
    parser.add_argument('--open-eqa-json', type=Path, default=(REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--resource-manifest', type=Path, required=True)
    parser.add_argument('--question-route-json', type=Path, required=True)
    parser.add_argument('--route-subset-manifest', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, default=(REPO_ROOT / 'datasets/openeqa/runs/new_runs/payload'))
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--visual-backbone', default='openeqa_question_router_v4')
    parser.add_argument('--graph-retrieval-profile', default='canonical_vector_evidence_v5')
    parser.add_argument('--answer-mode', choices=('live', 'mock'), default='live')
    parser.add_argument('--config', default='datasets/openeqa/qa.yaml')
    parser.add_argument('--vector-model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--retrieval-limit', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--qa-temperature', type=float, default=None)
    parser.add_argument('--qa-api-key-index', type=int, default=None)
    parser.add_argument('--qa-workers', type=int, default=1, help='Concurrent per-question QA workers for each isolated source shard.')
    parser.add_argument('--visual-top-k', type=int, default=8)
    parser.add_argument('--visual-min-score', type=float, default=0.0)
    parser.add_argument('--visual-max-candidate-packs', type=int, default=8)
    parser.add_argument('--visual-max-evidence-lines', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--strategy-variant', action='append', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run', action='store_true')
    return parser

def load_resource_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError('OpenEQA resource manifest must be an object')
    return payload

def validate_resource_manifest(payload: dict[str, Any], *, dataset_rows: list[dict[str, Any]], require_paths: bool=True) -> list[str]:
    issues: list[str] = []
    if payload.get('contract') != RESOURCE_CONTRACT:
        issues.append(f'resource contract must be {RESOURCE_CONTRACT}')
    resources = payload.get('execution_shards')
    if not isinstance(resources, list) or not resources:
        return [*issues, 'execution_shards must be a non-empty list']
    expected_sources = sorted({str(row.get('episode_history') or '').partition('/')[0] for row in dataset_rows if str(row.get('episode_history') or '').partition('/')[0]})
    actual_sources: list[str] = []
    for resource in resources:
        if not isinstance(resource, dict):
            issues.append('execution_shards contains a non-object entry')
            continue
        source = str(resource.get('data_source') or '').strip()
        actual_sources.append(source)
        for field in ('graph_manifest', 'visual_index_dir', 'visual_index_name'):
            if not str(resource.get(field) or '').strip():
                issues.append(f"{source or '<missing source>'}: missing {field}")
        if require_paths:
            graph_manifest = repo_path(Path(str(resource.get('graph_manifest') or '')))
            visual_index_dir = repo_path(Path(str(resource.get('visual_index_dir') or '')))
            if not graph_manifest.is_file():
                issues.append(f'{source}: graph manifest not found: {graph_manifest}')
            if not visual_index_dir.is_dir():
                issues.append(f'{source}: visual index directory not found: {visual_index_dir}')
    if len(actual_sources) != len(set(actual_sources)):
        issues.append(f'duplicate resource sources: {actual_sources}')
    if sorted(actual_sources) != expected_sources:
        issues.append(f'resource source coverage mismatch actual={sorted(actual_sources)} expected={expected_sources}')
    return issues

def validate_question_route_manifest(payload: dict[str, Any], *, dataset_rows: list[dict[str, Any]], visual_backbone: str | None=None, graph_retrieval_profile: str | None=None) -> list[str]:
    issues: list[str] = []
    if payload.get('contract') != UNIFIED_ROUTE_CONTRACT:
        issues.append(f'question route contract must be {UNIFIED_ROUTE_CONTRACT}')
    if payload.get('dataset') != 'OpenEQA':
        issues.append('question route dataset must be OpenEQA')
    if payload.get('uses_gold_answer') is not False:
        issues.append('question route manifest must declare uses_gold_answer=false')
    if payload.get('runtime_route_recompile_allowed') is not False:
        issues.append('question route manifest must forbid runtime recompilation')
    routing_profile = str(payload.get('routing_profile') or '')
    is_single_route = routing_profile == ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5
    if is_single_route and payload.get('route_contract') != SINGLE_ROUTE_CONTRACT:
        issues.append(f'single-route manifest route_contract must be {SINGLE_ROUTE_CONTRACT}')
    if routing_profile in IR_FIRST_ROUTING_PROFILES or is_single_route:
        if payload.get('uses_official_question_category_for_routing') is not False:
            issues.append('IR-first manifest must forbid official category routing')
        expected_routing_inputs = ['semantic_question_ir.question_type'] if is_single_route else ['question', 'semantic_question_ir']
        if payload.get('routing_inputs') != expected_routing_inputs:
            issues.append('IR-first manifest routing inputs must match its route contract')
        if payload.get('persists_route_strategy_handler') is not True:
            issues.append('IR-first manifest must persist route strategy handlers')
        if visual_backbone is not None and graph_retrieval_profile is not None:
            issues.extend(ir_first_execution_config_issues(routing_profile=routing_profile, visual_backbone=visual_backbone, graph_retrieval_profile=graph_retrieval_profile))
    expected_ids = {str(row.get('question_id') or row.get('uuid') or '').strip() for row in dataset_rows}
    expected_ids.discard('')
    records = payload.get('records')
    if not isinstance(records, list):
        return [*issues, 'question route records must be a list']
    actual_ids = [str(record.get('question_id') or record.get('uuid') or '').strip() for record in records if isinstance(record, dict)]
    if len(actual_ids) != len(set(actual_ids)):
        issues.append('question route manifest contains duplicate question IDs')
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        issues.append(f'question route coverage mismatch missing={len(missing)} extra={len(extra)}')
    if routing_profile in IR_FIRST_ROUTING_PROFILES or is_single_route:
        for record in records:
            if not isinstance(record, dict):
                continue
            item_id = str(record.get('question_id') or record.get('uuid') or '')
            ir = record.get('llm_question_ir')
            if not isinstance(ir, dict):
                issues.append(f'{item_id}: missing semantic question IR')
                continue
            if not is_single_route and str(ir.get('category') or '').strip():
                issues.append(f'{item_id}: IR-first question IR retains category')
            route_issues = single_route_decision_issues(ir) if is_single_route else question_route_decision_issues(ir)
            if route_issues:
                issues.append(f'{item_id}: {route_issues[0]}')
                continue
            route = ir['question_route']
            metadata = route.get('metadata')
            metadata = metadata if isinstance(metadata, dict) else {}
            if metadata.get('routing_profile') != routing_profile:
                issues.append(f'{item_id}: persisted routing profile drift')
            handler_issues = single_route_handler_contract_issues(route) if is_single_route else route_handler_contract_issues(route)
            for handler_issue in handler_issues:
                issues.append(f'{item_id}: {handler_issue}')
    return issues

def build_shard_command(*, args: argparse.Namespace, resource: dict[str, Any], shard_output_root: Path) -> list[str]:
    source = str(resource['data_source'])
    command = [sys.executable, str(SCRIPT_DIR / 'run_openeqa_visual_evidence_qa.py'), '--open-eqa-json', str(repo_path(args.open_eqa_json)), '--graph-manifest', str(repo_path(Path(resource['graph_manifest']))), '--question-ir-json', str(repo_path(args.question_route_json)), '--data-source', source, '--visual-index-dir', str(repo_path(Path(resource['visual_index_dir']))), '--visual-index-name', str(resource['visual_index_name']), '--output-root', str(shard_output_root), '--run-name', source, '--visual-backbone', args.visual_backbone, '--graph-retrieval-profile', args.graph_retrieval_profile, '--answer-mode', args.answer_mode, '--config', args.config, '--vector-model', args.vector_model, '--retrieval-limit', str(args.retrieval_limit), '--max-tokens', str(args.max_tokens), '--qa-workers', str(args.qa_workers), '--visual-top-k', str(args.visual_top_k), '--visual-min-score', str(args.visual_min_score), '--visual-max-candidate-packs', str(args.visual_max_candidate_packs), '--visual-max-evidence-lines', str(args.visual_max_evidence_lines), '--checkpoint-every', str(args.checkpoint_every)]
    if args.qa_temperature is not None:
        command.extend(['--qa-temperature', str(args.qa_temperature)])
    if args.qa_api_key_index is not None:
        command.extend(['--qa-api-key-index', str(args.qa_api_key_index)])
    if args.route_subset_manifest:
        command.extend(['--route-subset-manifest', str(repo_path(args.route_subset_manifest))])
    for variant in args.strategy_variant or []:
        command.extend(['--strategy-variant', variant])
    if args.resume:
        command.append('--resume')
    if args.run:
        command.append('--run')
    return command

def main() -> None:
    args = build_parser().parse_args()
    dataset_path = repo_path(args.open_eqa_json)
    resource_path = repo_path(args.resource_manifest)
    question_route_path = repo_path(args.question_route_json)
    route_subset_path = repo_path(args.route_subset_manifest) if args.route_subset_manifest else None
    output_dir = repo_path(args.output_root) / args.run_name
    shard_output_root = output_dir / 'shards'
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = load_open_eqa_rows(dataset_path)
    resources = load_resource_manifest(resource_path)
    issues = validate_resource_manifest(resources, dataset_rows=dataset_rows)
    if args.visual_backbone == SINGLE_ROUTE_VISUAL_BACKBONE:
        if route_subset_path is not None:
            issues.append('V5 single-route execution does not accept route subsets')
        if args.strategy_variant:
            issues.append('V5 single-route execution does not accept strategy_variant filters')
    if not question_route_path.is_file():
        issues.append(f'question route manifest not found: {question_route_path}')
    else:
        question_route_payload = json.loads(question_route_path.read_text(encoding='utf-8-sig'))
        if not isinstance(question_route_payload, dict):
            issues.append('question route manifest must be an object')
        else:
            issues.extend(validate_question_route_manifest(question_route_payload, dataset_rows=dataset_rows, visual_backbone=args.visual_backbone, graph_retrieval_profile=args.graph_retrieval_profile))
    if route_subset_path is not None and (not route_subset_path.is_file()):
        issues.append(f'route subset manifest not found: {route_subset_path}')
    qa_config = load_domain_config(args.config)
    if args.qa_workers < 1:
        issues.append('qa-workers must be >= 1')
    if args.answer_mode == 'live' and qa_config.graph_qa.model != 'gpt-5.6-luna':
        issues.append(f'formal OpenEQA live QA requires gpt-5.6-luna, got {qa_config.graph_qa.model}')
    plan = {'contract': RUN_PLAN_CONTRACT, 'dataset': 'OpenEQA', 'created_at': datetime.now(timezone.utc).isoformat(), 'run_name': args.run_name, 'open_eqa_json': str(dataset_path), 'resource_manifest': str(resource_path), 'question_route_json': str(question_route_path), 'route_subset_manifest': str(route_subset_path or ''), 'data_sources_are_execution_shards_only': True, 'qa_model': qa_config.graph_qa.model, 'common_strategy': {'visual_backbone': args.visual_backbone, 'graph_retrieval_profile': args.graph_retrieval_profile, 'answer_mode': args.answer_mode, 'config': args.config, 'vector_model': args.vector_model, 'retrieval_limit': args.retrieval_limit, 'max_tokens': args.max_tokens, 'qa_temperature': args.qa_temperature, 'qa_api_key_index': args.qa_api_key_index, 'qa_workers': args.qa_workers, 'visual_top_k': args.visual_top_k, 'visual_min_score': args.visual_min_score, 'visual_max_candidate_packs': args.visual_max_candidate_packs, 'visual_max_evidence_lines': args.visual_max_evidence_lines}, 'execution_shards': resources.get('execution_shards') or [], 'preflight_issues': issues}
    write_json_atomic(output_dir / 'run_plan.json', plan)
    if issues:
        print('preflight_status=failed')
        for issue in issues:
            print(f'- {issue}')
        raise SystemExit(2)
    commands = [build_shard_command(args=args, resource=resource, shard_output_root=shard_output_root) for resource in resources['execution_shards']]
    write_json_atomic(output_dir / 'execution_commands.json', {'commands': commands, 'uses_gold_answer': False})
    print(f'preflight_status=passed shards={len(commands)} qa_model={qa_config.graph_qa.model}')
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.run:
        print('Use --run to execute QA; shard preflights completed.')
        return
    finalize_command = [sys.executable, str(REPO_ROOT / 'scripts/datasets/openeqa/audit/finalize_openeqa_unified_qa_run.py'), '--open-eqa-json', str(dataset_path), '--question-route-json', str(question_route_path), '--output-dir', str(output_dir)]
    if route_subset_path is not None:
        finalize_command.extend(['--route-subset-manifest', str(route_subset_path)])
    for variant in args.strategy_variant or []:
        finalize_command.extend(['--strategy-variant', variant])
    for resource in resources['execution_shards']:
        finalize_command.extend(['--shard-run-dir', str(shard_output_root / str(resource['data_source']))])
    subprocess.run(finalize_command, cwd=REPO_ROOT, check=True)
    print(f'unified_run={output_dir}')
if __name__ == '__main__':
    main()
