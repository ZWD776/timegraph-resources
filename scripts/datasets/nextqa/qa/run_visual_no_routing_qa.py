"""Run isolated text-only QA for the NExT-QA no-routing ablation."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.qa.graph_only_qa import GraphOnlyMCQConfig, STRICT_IR_FIELDS, parse_api_key_pool, stable_api_key_slot
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text, require_within_nextqa_scratch
from vlkg.datasets.nextqa.qa.no_routing_qa import answer_no_routing_mcq, validate_no_routing_input
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot, sha256_file, utc_now
DEFAULT_CONFIG = (REPO_ROOT / 'configs/datasets/nextqa/no_routing_qa.yaml')
DEFAULT_IR = (REPO_ROOT / 'datasets/nextqa/ir/nextqa-ir-v3-luna-full4996/payload/nextqa-ir-v3-luna-full4996-20260817b/question_ir.retrieval.json')
DEFAULT_RUNTIME = (REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/questions.runtime.json')
DEFAULT_GRAPH = (REPO_ROOT / 'datasets/nextqa/retrieval/nextqa-cg6-retrieval-v3-route-policy-v3/payload/nextqa_visual_graph_baseline_v2_irv3_full4996_20260818a/frozen_graph_evidence.json')
DEFAULT_PILOT = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'dry_run' / 'pilot5' / 'nextqa_val_pilot5_manifest.json'

def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get('question_id') or '')
        if not qid or qid in result:
            raise ValueError(f'{name} contains an empty or duplicate question ID: {qid}')
        result[qid] = row
    return result

def _pilot_videos(path: Path) -> set[str]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    ids = {str(item['official_video_id']) for item in payload.get('videos', [])}
    if len(ids) != 5:
        raise ValueError('pilot manifest must contain five videos')
    return ids

def _run_root(run_id: str, resume: bool) -> Path:
    root = require_within_nextqa_scratch((REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'ablation' / run_id)
    if root.exists() and (not resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {root}')
    if not root.exists() and resume:
        raise FileNotFoundError(f'cannot resume missing run: {root}')
    root.mkdir(parents=True, exist_ok=resume)
    return root

def _write_checkpoint(run_root: Path, rows: list[dict[str, Any]], expected_ids: list[str], run_id: str, answer_mode: str) -> dict[str, Any]:
    by_id = _index(rows, 'predictions') if rows else {}
    ordered = [by_id[qid] for qid in expected_ids if qid in by_id]
    completed = [row for row in ordered if row.get('status') == 'completed']
    errors = [row for row in ordered if row.get('status') == 'error']
    checks = {'question_coverage': len(by_id) == len(expected_ids) and (not set(by_id) - set(expected_ids)), 'gold_purity': all((row.get('uses_gold_answer') is False for row in ordered)), 'official_type_purity': all((row.get('uses_official_type') is False for row in ordered)), 'no_routing_label': all((row.get('ablation_label') == 'no_routing' for row in ordered)), 'text_only_boundary': all((row.get('qa_receives_images') is False and row.get('qa_receives_image_paths') is False and (row.get('qa_receives_visual_embeddings') is False) for row in ordered)), 'equal_choice_budgets': all((isinstance(row.get('evidence_budget_per_choice'), list) and len(row['evidence_budget_per_choice']) == 5 and (len(set(row['evidence_budget_per_choice'])) == 1) for row in ordered)), 'no_route_handler_fields': all(('route_id' not in row and 'handler_id' not in row for row in ordered))}
    audit = {'schema_version': 'nextqa_visual_no_routing_qa_audit_v1', 'run_id': run_id, 'answer_mode': answer_mode, 'expected_question_count': len(expected_ids), 'prediction_record_count': len(ordered), 'completed_question_count': len(completed), 'error_question_count': len(errors), 'missing_question_ids': [qid for qid in expected_ids if qid not in by_id], 'checks': checks, 'failed_checks': [key for key, value in checks.items() if not value], 'uses_gold_answer': False, 'uses_official_type': False, 'qa_receives_images': False}
    atomic_write_json(run_root / 'predictions.json', ordered)
    atomic_write_json(run_root / 'audit.json', audit)
    atomic_write_json(run_root / 'summary.json', {key: audit[key] for key in ('schema_version', 'run_id', 'answer_mode', 'expected_question_count', 'prediction_record_count', 'completed_question_count', 'error_question_count', 'missing_question_ids', 'failed_checks')})
    atomic_write_text(run_root / 'summary.md', '\n'.join(['# NExT-QA Visual No-Routing QA', '', f'- Expected: `{len(expected_ids)}`', f'- Completed: `{len(completed)}`', f'- Errors: `{len(errors)}`', f"- Failed checks: `{audit['failed_checks']}`", '']))
    return audit

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run NExT-QA visual retrieval without routing.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--visual-evidence', type=Path, required=True)
    parser.add_argument('--question-ir', type=Path, default=DEFAULT_IR)
    parser.add_argument('--runtime-questions', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--frozen-graph-evidence', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--pilot-manifest', type=Path, default=DEFAULT_PILOT)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--answer-mode', choices=('mock', 'live'), default='mock')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    return parser

def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError('workers and checkpoint-every must be positive')
    config_path = args.config.resolve()
    payload = __import__('yaml').safe_load(config_path.read_text(encoding='utf-8')) or {}
    if payload.get('schema_version') != 'nextqa_visual_no_routing_mcq_config_v2':
        raise ValueError('unsupported no-routing QA config schema')
    qa = payload.get('qa') or {}
    config = GraphOnlyMCQConfig(profile_name=str(qa.get('profile_name') or ''), provider=str(qa.get('provider') or ''), model=str(qa.get('model') or ''), api_base_url=str(qa.get('api_base_url') or ''), api_base_url_env_var=str(qa.get('api_base_url_env_var') or '') or None, endpoint_path=str(qa.get('endpoint_path') or ''), api_key_env_var=str(qa.get('api_key_env_var') or ''), timeout_sec=float(qa.get('timeout_sec', 300)), temperature=float(qa.get('temperature', 0.1)), max_tokens=int(qa.get('max_tokens', 600)), max_attempts=int(qa.get('max_attempts', 2)), evidence_limit=16, thinking_enabled=bool(qa.get('thinking_enabled', False)), api_key_pool_env_var=str(qa.get('api_key_pool_env_var') or 'OPENAI_API_KEYS'), retry_backoff_sec=float(qa.get('retry_backoff_sec', 1)), retry_jitter_sec=float(qa.get('retry_jitter_sec', 0.25)))
    if config.model != 'gpt-5.6-luna' or config.provider != 'openai_compatible' or (not config.profile_name.startswith('nextqa_visual_no_routing_mcq_')):
        raise ValueError('no-routing QA config must be isolated and use gpt-5.6-luna')
    question_ir_path = args.question_ir.resolve()
    runtime_path = args.runtime_questions.resolve()
    graph_path = args.frozen_graph_evidence.resolve()
    visual_path = args.visual_evidence.resolve()
    questions_ir = _index(_load_rows(question_ir_path, 'question IR'), 'question IR')
    runtime_raw = _index(_load_rows(runtime_path, 'runtime questions'), 'runtime questions')
    if set(questions_ir) != set(runtime_raw):
        raise ValueError('question IR and runtime question ID sets must match')
    if any((str(runtime_raw[qid].get('video_id') or '') != str(questions_ir[qid].get('video_id') or '') for qid in runtime_raw)):
        raise ValueError('question IR and runtime video IDs do not align')
    runtime = {qid: {'question_id': str(row['question_id']), 'video_id': str(row['video_id']), 'normalized_question': str(questions_ir[qid].get('normalized_question') or ''), 'answer_choices': list(row.get('answer_choices') or []), 'answer_format': str(row.get('answer_format') or 'multiple_choice_5'), **{field: questions_ir[qid].get(field) for field in STRICT_IR_FIELDS}, 'uses_gold_answer': False} for qid, row in runtime_raw.items()}
    graphs = _index(_load_rows(graph_path, 'graph evidence'), 'graph evidence')
    visuals = _index(_load_rows(visual_path, 'visual evidence'), 'visual evidence')
    if set(questions_ir) != set(runtime) or set(runtime) != set(graphs):
        raise ValueError('question IR, graph, and runtime ID sets must match')
    selected = set(runtime)
    pilot_path = args.pilot_manifest.resolve()
    if args.scope == 'pilot':
        video_ids = _pilot_videos(pilot_path)
        selected = {qid for qid in selected if str(runtime[qid]['video_id']) in video_ids}
    qids = sorted(selected, key=lambda qid: (str(runtime[qid]['video_id']), qid))
    expected = 39 if args.scope == 'pilot' else 4996
    if len(qids) != expected:
        raise ValueError(f'{args.scope} scope requires {expected} questions, found {len(qids)}')
    if set(qids) != set(visuals):
        raise ValueError('selected question and visual evidence ID sets must match')
    for qid in qids:
        validate_no_routing_input(runtime[qid], graphs[qid], visuals[qid])
    run_root = _run_root(args.run_id, args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'visual_no_routing_qa_v2', 'scope': args.scope, 'question_ir': fingerprint_file(question_ir_path), 'runtime_questions': fingerprint_file(runtime_path), 'frozen_graph_evidence': fingerprint_file(graph_path), 'visual_evidence': fingerprint_file(visual_path), 'config': fingerprint_file(config_path), 'pilot_manifest': fingerprint_file(pilot_path) if args.scope == 'pilot' else None, 'question_count': len(qids), 'answer_mode': args.answer_mode, 'model': config.model, 'endpoint_url': config.endpoint_url() if args.answer_mode == 'live' else None, 'workers': args.workers, 'timeout_sec': config.timeout_sec, 'max_attempts': config.max_attempts, 'graph_evidence_limit': 16, 'visual_evidence_limit': 12, 'normalized_question_source': 'question_ir.retrieval.json', 'answer_choices_source': 'standardized_questions.runtime.json', 'qa_question_ir_fields': list(STRICT_IR_FIELDS), 'qa_prompt_contract': 'routed_visual_v2_shape_with_uniform_neutral_visual_evidence', 'ablation_label': 'no_routing', 'uses_route': False, 'uses_handler': False, 'uses_intent': False, 'uses_visual_evidence_demand': False, 'uses_official_type': False, 'uses_gold_answer': False, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'calls_api': args.answer_mode == 'live', 'resume_contract': 'preserve_completed_retry_error_only_v2'}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_no_routing_qa_run_config_v2')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    existing = _load_rows(run_root / 'predictions.json', 'existing predictions') if args.resume and (run_root / 'predictions.json').is_file() else []
    existing_by_id = _index(existing, 'existing predictions') if existing else {}
    current = {qid: row for qid, row in existing_by_id.items() if row.get('status') == 'completed'}
    pending = [qid for qid in qids if qid not in current]
    keys = parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ''))
    if args.answer_mode == 'live' and (not keys) and (not os.getenv(config.api_key_env_var, '').strip()):
        raise RuntimeError('live no-routing QA requires an API key pool')
    pool_size = max(1, len(keys) or 1)

    def process(qid: str) -> dict[str, Any]:
        return answer_no_routing_mcq(runtime[qid], graphs[qid], visuals[qid], config=config, answer_mode=args.answer_mode, api_key_slot=stable_api_key_slot(qid, pool_size))

    def record(offset: int, qid: str, result: dict[str, Any]) -> None:
        current[qid] = result
        if offset % args.checkpoint_every == 0 or offset == len(pending):
            _write_checkpoint(run_root, list(current.values()), qids, args.run_id, args.answer_mode)
        print(json.dumps({'progress': f'{offset}/{len(pending)}', 'question_id': qid, 'status': result['status']}, ensure_ascii=False), flush=True)
    if args.workers == 1:
        for offset, qid in enumerate(pending, start=1):
            record(offset, qid, process(qid))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, qid): qid for qid in pending}
            for offset, future in enumerate(as_completed(futures), start=1):
                record(offset, futures[future], future.result())
    audit = _write_checkpoint(run_root, list(current.values()), qids, args.run_id, args.answer_mode)
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    valid = audit['prediction_record_count'] == len(qids) and (not audit['missing_question_ids']) and (not audit['failed_checks']) and (not drift['code_drift_detected'])
    status = 'invalid' if not valid else 'completed' if audit['error_question_count'] == 0 else 'completed_with_errors'
    atomic_write_json(run_root / 'run_status.json', {'status': status, 'run_id': args.run_id, 'completed_at_utc': utc_now(), 'completed_question_count': audit['completed_question_count'], 'error_question_count': audit['error_question_count'], 'calls_api': args.answer_mode == 'live', 'code_drift_detected': drift['code_drift_detected']})
    print(json.dumps({'status': status, 'completed': audit['completed_question_count'], 'errors': audit['error_question_count']}, ensure_ascii=False, indent=2))
    return 1 if status == 'invalid' else 0
if __name__ == '__main__':
    raise SystemExit(main())
