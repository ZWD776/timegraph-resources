from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.qa.graph_only_qa import answer_graph_only_mcq, load_graph_only_mcq_config, parse_api_key_pool, stable_api_key_slot, validate_qa_input_pair
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, persist_drift_report, persist_run_config, persist_source_snapshot, sha256_file, utc_now
from vlkg.datasets.nextqa.runtime.special_patch import SPECIAL_PATCH_QUESTION_COUNT, SPECIAL_PATCH_VIDEO_COUNT
DEFAULT_QUESTION_IR = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'question_ir' / 'nextqa_question_ir_v2_pilot39_20260811T153915Z' / 'question_ir.runtime.json'
DEFAULT_RETRIEVAL = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'graph_retrieval_pilots' / 'nextqa_graph_retrieval_v2_pilot39_20260811T153940Z' / 'retrieval_records.json'
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'datasets/nextqa/graph_qa.yaml'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the isolated five-video NExT-QA graph-only MCQ pilot.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-ir', type=Path, default=DEFAULT_QUESTION_IR)
    parser.add_argument('--retrieval-records', type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--answer-mode', choices=('mock', 'live'), default='live')
    parser.add_argument('--scope', choices=('pilot', 'full', 'special'), default='pilot')
    parser.add_argument('--checkpoint-every', type=int, default=None, help='atomic checkpoint interval; defaults to 1 for pilot and 10 for full')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed-predictions', type=Path, default=None, help='import completed records from an earlier run into a new run')
    parser.add_argument('--workers', type=int, default=1, help='parallel API workers; default 1 preserves serial runner behavior')
    return parser

def _load_rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _index_unique(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id:
            raise ValueError(f'{name} contains an empty question_id')
        if question_id in indexed:
            raise ValueError(f'{name} contains duplicate question_id: {question_id}')
        indexed[question_id] = row
    return indexed

def _load_completed_seed(path: Path, *, expected_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Import only completed, graph-only records from an earlier run."""
    rows = _load_rows(path, name='seed predictions')
    indexed = _index_unique(rows, name='seed predictions')
    unexpected = sorted(set(indexed) - expected_ids)
    if unexpected:
        raise ValueError('seed predictions contain question IDs outside current input: ' + ', '.join(unexpected[:5]))
    completed: list[dict[str, Any]] = []
    for row in rows:
        if row.get('status') != 'completed':
            continue
        if row.get('uses_gold_answer') is not False:
            raise ValueError(f"seed gold purity failed: {row.get('question_id')}")
        if any((row.get(field) is not False for field in ('reads_rgb', 'reads_object_crop', 'reads_visual_embedding_cache', 'uses_visual_index'))):
            raise ValueError(f"seed visual purity failed: {row.get('question_id')}")
        completed.append(row)
    return (completed, {'schema_version': 'nextqa_completed_seed_migration_v1', 'source_predictions_path': str(path.resolve()), 'source_predictions_sha256': sha256_file(path), 'source_record_count': len(rows), 'source_completed_count': len(completed), 'source_error_count': sum((row.get('status') == 'error' for row in rows)), 'imported_completed_count': len(completed), 'skipped_non_completed_count': len(rows) - len(completed), 'unexpected_question_id_count': len(unexpected), 'uses_gold_answer': False})

def select_pending_questions(question_ids: list[str], *, existing_predictions: list[Mapping[str, Any]]) -> list[str]:
    completed = {str(row.get('question_id')) for row in existing_predictions if row.get('status') == 'completed'}
    return [question_id for question_id in question_ids if question_id not in completed]

def prediction_from_retrieval_error(question_row: Mapping[str, Any], retrieval_row: Mapping[str, Any], *, model: str, qa_profile: str, graph_retrieval_profile: str, answer_mode: str) -> dict[str, Any]:
    """Propagate a full-run retrieval gap without issuing an API request."""
    if retrieval_row.get('status') != 'error':
        raise ValueError('only retrieval error records may be propagated')
    if retrieval_row.get('uses_gold_answer') is not False:
        raise ValueError('retrieval error record violates gold purity')
    return {'schema_version': 'nextqa_graph_only_mcq_prediction_v1', 'question_id': str(question_row['question_id']), 'video_id': str(question_row['video_id']), 'model': model if answer_mode == 'live' else 'nextqa-mock-v1', 'answer_mode': answer_mode, 'qa_profile': qa_profile, 'graph_retrieval_profile': graph_retrieval_profile, 'graph_record_ids': [], 'graph_evidence_count': 0, 'evidence_budget_per_choice': [0] * 5, 'request_sha256': None, 'attempt_count': 0, 'status': 'error', 'graph_rag_status': 'graph_only_retrieval_error', 'predicted_choice_index': None, 'predicted_choice_text': None, 'error_stage': 'graph_retrieval', 'error_type': str(retrieval_row.get('error_type') or 'RetrievalError'), 'error_message': str(retrieval_row.get('error_message') or 'graph retrieval did not complete'), 'response_metadata': None, 'uses_gold_answer': False, 'uses_official_type': False, 'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_embedding_cache': False, 'uses_visual_index': False}

def _audit(predictions: list[Mapping[str, Any]], *, expected_ids: list[str], run_id: str, answer_mode: str) -> dict[str, Any]:
    ids = [str(row.get('question_id') or '') for row in predictions]
    completed = [row for row in predictions if row.get('status') == 'completed']
    errors = [row for row in predictions if row.get('status') == 'error']
    return {'schema_version': 'nextqa_graph_only_mcq_audit_v1', 'run_id': run_id, 'answer_mode': answer_mode, 'expected_question_count': len(expected_ids), 'prediction_record_count': len(predictions), 'completed_question_count': len(completed), 'error_question_count': len(errors), 'duplicate_question_id_count': len(ids) - len(set(ids)), 'missing_question_ids': sorted(set(expected_ids) - set(ids)), 'unexpected_question_ids': sorted(set(ids) - set(expected_ids)), 'video_count': len({str(row.get('video_id')) for row in predictions}), 'status_distribution': dict(sorted(Counter((str(row.get('status')) for row in predictions)).items())), 'graph_evidence_count_distribution': dict(sorted(Counter((int(row.get('graph_evidence_count', 0)) for row in predictions)).items())), 'all_choices_share_equal_evidence_budget': all((len(row.get('evidence_budget_per_choice', [])) == 5 and len(set(row.get('evidence_budget_per_choice', []))) == 1 for row in predictions)), 'gold_purity': all((row.get('uses_gold_answer') is False for row in predictions)), 'official_type_purity': all((row.get('uses_official_type') is False for row in predictions)), 'visual_purity': all((row.get('reads_rgb') is False and row.get('reads_object_crop') is False and (row.get('reads_visual_embedding_cache') is False) and (row.get('uses_visual_index') is False) for row in predictions)), 'api_request_attempt_count': sum((int(row.get('attempt_count', 0)) for row in predictions)) if answer_mode == 'live' else 0, 'uses_gold_answer': False}

def _summary_markdown(audit: Mapping[str, Any]) -> str:
    return '\n'.join(['# NExT-QA Graph-only MCQ QA Pilot', '', f"- Run ID: `{audit['run_id']}`", f"- Answer mode: `{audit['answer_mode']}`", f"- Predictions: {audit['prediction_record_count']}/{audit['expected_question_count']}", f"- Completed: {audit['completed_question_count']}", f"- Errors: {audit['error_question_count']}", f"- Videos: {audit['video_count']}", f"- Equal evidence budget across choices: {audit['all_choices_share_equal_evidence_budget']}", f"- Gold purity: {audit['gold_purity']}", f"- Official-type purity: {audit['official_type_purity']}", f"- Visual purity: {audit['visual_purity']}", '', 'Accuracy is intentionally absent here. Evaluator-only labels are loaded by the separate evaluation stage after this prediction file is complete.', ''])

def _write_checkpoint(run_root: Path, predictions: list[dict[str, Any]], *, expected_ids: list[str], run_id: str, answer_mode: str) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda row: str(row['question_id']))
    audit = _audit(ordered, expected_ids=expected_ids, run_id=run_id, answer_mode=answer_mode)
    atomic_write_json(run_root / 'predictions.json', ordered)
    atomic_write_json(run_root / 'audit.json', audit)
    atomic_write_json(run_root / 'summary.json', audit)
    atomic_write_text(run_root / 'summary.md', _summary_markdown(audit))
    return audit

def main() -> int:
    args = build_parser().parse_args()
    for stream_name in ('stdin', 'stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    question_ir_path = args.question_ir.resolve()
    retrieval_path = args.retrieval_records.resolve()
    config_path = args.config.resolve()
    scope_counts = {'pilot': (39, 5), 'full': (4996, 570), 'special': (SPECIAL_PATCH_QUESTION_COUNT, SPECIAL_PATCH_VIDEO_COUNT)}
    expected_question_count, expected_video_count = scope_counts[args.scope]
    checkpoint_every = args.checkpoint_every or (1 if args.scope == 'pilot' else 10)
    if checkpoint_every < 1:
        raise ValueError('checkpoint interval must be positive')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    config = load_graph_only_mcq_config(config_path)
    if args.answer_mode == 'live':
        pool = parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ''))
        if not pool and (not os.getenv(config.api_key_env_var, '').strip()):
            raise RuntimeError(f'missing environment variable: {config.api_key_env_var} or {config.api_key_pool_env_var}')
    question_rows = _load_rows(question_ir_path, name='question IR')
    retrieval_rows = _load_rows(retrieval_path, name='retrieval records')
    questions = _index_unique(question_rows, name='question IR')
    retrieval = _index_unique(retrieval_rows, name='retrieval records')
    if set(questions) != set(retrieval):
        raise ValueError('question IR and retrieval question ID sets do not match')
    question_ids = sorted(questions)
    if len(question_ids) != expected_question_count or len({str(row['video_id']) for row in question_rows}) != expected_video_count:
        raise ValueError(f'this {args.scope} runner requires {expected_question_count} questions over {expected_video_count} videos')
    retrieval_profiles = sorted({str(row['graph_retrieval_profile']) for row in retrieval_rows if row.get('status') == 'completed' and row.get('graph_retrieval_profile')})
    if len(retrieval_profiles) != 1:
        raise ValueError('QA requires exactly one graph retrieval profile')
    for question_id in question_ids:
        retrieval_row = retrieval[question_id]
        if retrieval_row.get('status') == 'completed':
            validate_qa_input_pair(questions[question_id], retrieval_row, evidence_limit=config.evidence_limit)
        elif args.scope == 'pilot' or retrieval_row.get('status') != 'error':
            raise ValueError(f'retrieval input is not completed for pilot question: {question_id}')
        elif retrieval_row.get('uses_gold_answer') is not False:
            raise ValueError(f'retrieval error purity failed: {question_id}')
    run_root = prepare_stage_run_root('qa_pilots' if args.scope == 'pilot' else 'qa', args.run_id, resume=args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': {'pilot': 'graph_only_mcq_qa_pilot', 'full': 'full_validation_graph_only_mcq_qa', 'special': 'special_patch_29_video_graph_only_mcq_qa'}[args.scope], 'scope': args.scope, 'run_id': args.run_id, 'question_ir_path': str(question_ir_path), 'question_ir_sha256': sha256_file(question_ir_path), 'retrieval_records_path': str(retrieval_path), 'retrieval_records_sha256': sha256_file(retrieval_path), 'config_path': str(config_path), 'config_sha256': sha256_file(config_path), 'question_count': len(question_ids), 'video_count': expected_video_count, 'answer_mode': args.answer_mode, 'provider': config.provider, 'model': config.model, 'qa_profile': config.profile_name, 'graph_retrieval_profiles': retrieval_profiles, 'retrieval_error_propagation': 'record_error_without_api_request_v1' if args.scope != 'pilot' else None, 'endpoint_url': config.endpoint_url() if args.answer_mode == 'live' else None, 'api_key_env_var': config.api_key_env_var if args.answer_mode == 'live' else None, 'api_key_pool_env_var': config.api_key_pool_env_var if args.answer_mode == 'live' else None, 'api_key_pool_size': len(parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ''))) if args.answer_mode == 'live' else None, 'timeout_sec': config.timeout_sec, 'retry_backoff_sec': config.retry_backoff_sec, 'retry_jitter_sec': config.retry_jitter_sec, 'seed_predictions_path': str(args.seed_predictions.resolve()) if args.seed_predictions else None, 'seed_predictions_sha256': sha256_file(args.seed_predictions.resolve()) if args.seed_predictions and args.seed_predictions.resolve().is_file() else None, 'temperature': config.temperature, 'max_tokens': config.max_tokens, 'max_attempts': config.max_attempts, 'evidence_limit': config.evidence_limit, 'choice_count': 5, 'choice_scoring_policy': 'single_shared_evidence_prompt_symmetric_choices_v1', 'checkpoint_every_questions': checkpoint_every, 'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_embedding_cache': False, 'uses_visual_index': False, 'uses_faiss': False, 'uses_official_type': False, 'uses_gold_answer': False, 'output_root': str(run_root), 'resume_contract': 'preserve_completed_retry_error_only_v1'}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_graph_only_mcq_run_config_v1')
    start_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    source_snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=start_sources, resume=args.resume)
    predictions_path = run_root / 'predictions.json'
    seed_completed: list[dict[str, Any]] = []
    if args.seed_predictions:
        seed_path = args.seed_predictions.resolve()
        if not seed_path.is_file():
            raise FileNotFoundError(f'seed predictions are missing: {seed_path}')
        seed_completed, seed_audit = _load_completed_seed(seed_path, expected_ids=set(question_ids))
        atomic_write_json(run_root / 'seed_migration_audit.json', seed_audit)
    existing = _load_rows(predictions_path, name='existing predictions') if args.resume and predictions_path.is_file() else []
    existing_by_id = _index_unique(existing, name='existing predictions') if existing else {}
    completed_by_id = {question_id: row for question_id, row in existing_by_id.items() if row.get('status') == 'completed'}
    pending = select_pending_questions(question_ids, existing_predictions=existing + seed_completed)
    current_by_id = dict(completed_by_id)
    if not args.resume:
        current_by_id.update({str(row['question_id']): row for row in seed_completed})

    def process_question(question_id: str) -> dict[str, Any]:
        retrieval_row = retrieval[question_id]
        if retrieval_row.get('status') == 'error':
            return prediction_from_retrieval_error(questions[question_id], retrieval_row, model=config.model, qa_profile=config.profile_name, graph_retrieval_profile=retrieval_profiles[0], answer_mode=args.answer_mode)
        return answer_graph_only_mcq(questions[question_id], retrieval_row, config=config, answer_mode=args.answer_mode, api_key_slot=stable_api_key_slot(question_id, max(1, len(parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ''))) or 1)))

    def record_result(offset: int, question_id: str, result: dict[str, Any]) -> None:
        current_by_id[question_id] = result
        if offset % checkpoint_every == 0 or offset == len(pending):
            _write_checkpoint(run_root, list(current_by_id.values()), expected_ids=question_ids, run_id=args.run_id, answer_mode=args.answer_mode)
        print(json.dumps({'progress': f'{offset}/{len(pending)}', 'question_id': question_id, 'status': result['status'], 'attempt_count': result['attempt_count']}, ensure_ascii=False), flush=True)
    if args.workers == 1:
        for offset, question_id in enumerate(pending, start=1):
            record_result(offset, question_id, process_question(question_id))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_question, question_id): question_id for question_id in pending}
            for offset, future in enumerate(as_completed(futures), start=1):
                question_id = futures[future]
                record_result(offset, question_id, future.result())
    audit = _write_checkpoint(run_root, list(current_by_id.values()), expected_ids=question_ids, run_id=args.run_id, answer_mode=args.answer_mode)
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=source_snapshot, current_source_paths=current_sources)
    structurally_valid = audit['prediction_record_count'] == len(question_ids) and audit['duplicate_question_id_count'] == 0 and (not audit['missing_question_ids']) and (not audit['unexpected_question_ids']) and audit['all_choices_share_equal_evidence_budget'] and audit['gold_purity'] and audit['official_type_purity'] and audit['visual_purity'] and (not drift['code_drift_detected'])
    status_name = 'invalid' if not structurally_valid else 'completed' if audit['error_question_count'] == 0 else 'completed_with_errors'
    status = {'status': status_name, 'run_id': args.run_id, 'completed_at_utc': utc_now(), 'completed_question_count': audit['completed_question_count'], 'error_question_count': audit['error_question_count'], 'code_drift_detected': drift['code_drift_detected'], 'uses_gold_answer': False}
    atomic_write_json(run_root / 'run_status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 1 if status_name == 'invalid' else 0
if __name__ == '__main__':
    raise SystemExit(main())
