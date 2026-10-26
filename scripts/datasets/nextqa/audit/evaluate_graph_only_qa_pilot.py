from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, persist_drift_report, persist_run_config, persist_source_snapshot, sha256_file, utc_now
DEFAULT_RUNTIME = REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/questions.runtime.json'
DEFAULT_LABELS = REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/labels.evaluator_only.json'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate a complete NExT-QA graph-only MCQ prediction file.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--question-runtime', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--evaluator-labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='full')
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _ids(rows: list[Mapping[str, Any]], *, name: str) -> list[str]:
    values = [str(row.get('question_id') or '') for row in rows]
    if any((not value for value in values)) or len(values) != len(set(values)):
        raise ValueError(f'{name} has empty or duplicate question IDs')
    return values

def _summary_markdown(metrics: Mapping[str, Any], *, run_id: str) -> str:
    lines = ['# NExT-QA Graph-only Exact-choice Evaluation', '', f'- Run ID: `{run_id}`', f"- Overall accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_question_count']}/{metrics['question_count']})", f"- Completed/errors: {metrics['completed_question_count']}/{metrics['error_question_count']}", f"- Video coverage: {metrics['video_coverage']:.4f}", f"- Graph evidence coverage: {metrics['graph_evidence_coverage']:.4f}", '', '## Accuracy by Official Type', '']
    for type_code, row in metrics['official_type_accuracy'].items():
        lines.append(f"- {type_code}: {row['accuracy']:.4f} ({row['correct']}/{row['count']})")
    lines.extend(['', '## Accuracy by Group', ''])
    for group, row in metrics['question_group_accuracy'].items():
        lines.append(f"- {group}: {row['accuracy']:.4f} ({row['correct']}/{row['count']})")
    lines.extend(['', 'Error attribution is provisional. Full-graph signal and HSGG coverage checks are a separate audit gate.', ''])
    return '\n'.join(lines)

def main() -> int:
    args = build_parser().parse_args()
    predictions_path = args.predictions.resolve()
    runtime_path = args.question_runtime.resolve()
    labels_path = args.evaluator_labels.resolve()
    expected_question_count = 39 if args.scope == 'pilot' else 4996
    predictions = _load_rows(predictions_path, name='predictions')
    runtime_questions = _load_rows(runtime_path, name='runtime questions')
    prediction_ids = _ids(predictions, name='predictions')
    runtime_ids = _ids(runtime_questions, name='runtime questions')
    if len(predictions) != expected_question_count or set(prediction_ids) != set(runtime_ids):
        raise ValueError(f'prediction file is not the complete {expected_question_count}-question {args.scope} run')
    labels = _load_rows(labels_path, name='evaluator labels')
    if set(_ids(labels, name='evaluator labels')) != set(prediction_ids):
        raise ValueError('evaluator labels do not match the complete prediction set')
    run_root = prepare_stage_run_root('qa_evaluations', args.run_id, resume=args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'graph_only_mcq_exact_choice_evaluation', 'scope': args.scope, 'run_id': args.run_id, 'predictions_path': str(predictions_path), 'predictions_sha256': sha256_file(predictions_path), 'runtime_questions_path': str(runtime_path), 'runtime_questions_sha256': sha256_file(runtime_path), 'evaluator_labels_path': str(labels_path), 'evaluator_labels_sha256': sha256_file(labels_path), 'question_count': len(predictions), 'metric': 'exact_choice_accuracy', 'gold_access': 'evaluator_only_after_complete_predictions', 'uses_gold_answer': True, 'output_root': str(run_root)}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_exact_choice_evaluation_run_config_v1')
    start_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=start_sources, resume=args.resume)
    metrics = evaluate_exact_choice_accuracy(predictions, labels, runtime_questions)
    atomic_write_json(run_root / 'evaluation.json', metrics)
    summary = {key: value for key, value in metrics.items() if key != 'question_results'}
    atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_text(run_root / 'summary.md', _summary_markdown(metrics, run_id=args.run_id))
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=current_sources)
    status = {'status': 'invalid' if drift['code_drift_detected'] else 'completed', 'run_id': args.run_id, 'completed_at_utc': utc_now(), 'question_count': metrics['question_count'], 'overall_accuracy': metrics['overall_accuracy'], 'code_drift_detected': drift['code_drift_detected'], 'gold_access': 'evaluator_only'}
    atomic_write_json(run_root / 'run_status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 1 if drift['code_drift_detected'] else 0
if __name__ == '__main__':
    raise SystemExit(main())
