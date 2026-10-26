from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
DEFAULT_LABELS = (REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/labels.evaluator_only.json')
DEFAULT_RUNTIME = (REPO_ROOT / 'datasets/nextqa/ir/nextqa-ir-v3-luna-full4996/payload/nextqa-ir-v3-luna-full4996-20260817b/question_ir.runtime.json')
DEFAULT_GRAPH = (REPO_ROOT / 'datasets/nextqa/runs/nextqa-cg6-qa-v3-luna-full4996/payload/nextqa-graph-qa-luna-irv3-full4996-20260817a/predictions.json')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Paired exact-choice evaluation for NExT-QA visual v2 QA.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--visual-predictions', type=Path, required=True)
    parser.add_argument('--graph-only-predictions', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--runtime-questions', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id or question_id in result:
            raise ValueError(f'{name} contains an empty or duplicate question ID')
        result[question_id] = row
    return result

def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root('qa_evaluations', args.run_id, resume=args.resume)
    visual_path = args.visual_predictions.resolve()
    graph_path = args.graph_only_predictions.resolve()
    labels_path = args.labels.resolve()
    runtime_path = args.runtime_questions.resolve()
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'paired_exact_choice_visual_v2', 'visual_predictions': fingerprint_file(visual_path), 'graph_only_predictions': fingerprint_file(graph_path), 'labels': fingerprint_file(labels_path), 'runtime_questions': fingerprint_file(runtime_path), 'metric': 'exact_choice_accuracy', 'uses_gpt_judge': False, 'selects_better_answer_per_question': False, 'calls_api': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_paired_evaluation_run_config_v2')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    visual = _load_rows(visual_path, 'visual predictions')
    graph_all = _index(_load_rows(graph_path, 'graph-only predictions'), 'graph-only predictions')
    labels_all = _index(_load_rows(labels_path, 'labels'), 'labels')
    runtime_all = _index(_load_rows(runtime_path, 'runtime questions'), 'runtime questions')
    visual_by_id = _index(visual, 'visual predictions')
    question_ids = set(visual_by_id)
    if not question_ids <= set(graph_all) or not question_ids <= set(labels_all) or (not question_ids <= set(runtime_all)):
        raise ValueError('visual prediction IDs do not align with graph, labels, and runtime')
    ordered_ids = sorted(question_ids)
    graph = [graph_all[qid] for qid in ordered_ids]
    labels = [labels_all[qid] for qid in ordered_ids]
    runtime = [runtime_all[qid] for qid in ordered_ids]
    visual_ordered = [visual_by_id[qid] for qid in ordered_ids]
    graph_metrics = evaluate_exact_choice_accuracy(graph, labels, runtime)
    visual_metrics = evaluate_exact_choice_accuracy(visual_ordered, labels, runtime)
    paired_rows: list[dict[str, Any]] = []
    outcomes = Counter()
    route_counts = Counter()
    route_correct = Counter()
    type_delta: dict[str, Counter[str]] = defaultdict(Counter)
    for question_id in ordered_ids:
        graph_row = graph_all[question_id]
        visual_row = visual_by_id[question_id]
        label = labels_all[question_id]
        runtime_row = runtime_all[question_id]
        gold = int(label['gold_choice_index'])
        graph_pred = graph_row.get('predicted_choice_index') if graph_row.get('status') == 'completed' else None
        visual_pred = visual_row.get('predicted_choice_index') if visual_row.get('status') == 'completed' else None
        graph_correct = graph_pred == gold
        visual_correct = visual_pred == gold
        if not graph_correct and visual_correct:
            outcome = 'improved'
        elif graph_correct and (not visual_correct):
            outcome = 'regressed'
        elif graph_correct and visual_correct:
            outcome = 'unchanged_correct'
        else:
            outcome = 'unchanged_incorrect'
        changed = graph_pred != visual_pred
        outcomes[outcome] += 1
        outcomes['answer_changed' if changed else 'answer_unchanged'] += 1
        route_id = str(visual_row.get('route_id') or '')
        route_counts[route_id] += 1
        route_correct[route_id] += int(visual_correct)
        type_code = str(runtime_row['official_type'])
        type_delta[type_code][outcome] += 1
        paired_rows.append({'question_id': question_id, 'video_id': str(runtime_row['video_id']), 'official_type': type_code, 'route_id': route_id, 'graph_status': graph_row.get('status'), 'visual_status': visual_row.get('status'), 'gold_choice_index': gold, 'graph_choice_index': graph_pred, 'visual_choice_index': visual_pred, 'graph_correct': graph_correct, 'visual_correct': visual_correct, 'answer_changed': changed, 'outcome': outcome})
    route_purity = all((row.get('handler_id') == f"nextqa_visual_handler_v2:{row.get('route_id')}" and row.get('uses_official_type') is False and (row.get('uses_gold_answer') is False) for row in visual_ordered))
    summary = {'schema_version': 'nextqa_visual_paired_exact_evaluation_v2', 'question_count': len(ordered_ids), 'graph_only_accuracy': graph_metrics['overall_accuracy'], 'visual_enhanced_accuracy': visual_metrics['overall_accuracy'], 'accuracy_delta': visual_metrics['overall_accuracy'] - graph_metrics['overall_accuracy'], 'paired_outcomes': dict(sorted(outcomes.items())), 'graph_only_completed_count': graph_metrics['completed_question_count'], 'graph_only_error_count': graph_metrics['error_question_count'], 'visual_completed_count': visual_metrics['completed_question_count'], 'visual_error_count': visual_metrics['error_question_count'], 'graph_only_official_type_accuracy': graph_metrics['official_type_accuracy'], 'visual_official_type_accuracy': visual_metrics['official_type_accuracy'], 'official_type_paired_outcomes': {type_code: dict(sorted(counts.items())) for type_code, counts in sorted(type_delta.items())}, 'visual_route_accuracy': {route_id: {'correct': route_correct[route_id], 'count': count, 'accuracy': route_correct[route_id] / count} for route_id, count in sorted(route_counts.items())}, 'route_handler_variant_purity': route_purity, 'uses_gpt_judge': False, 'selects_better_answer_per_question': False, 'metric': 'exact_choice_accuracy', 'status': 'passed' if route_purity else 'failed'}
    atomic_write_json(run_root / 'graph_only_metrics.json', graph_metrics)
    atomic_write_json(run_root / 'visual_enhanced_metrics.json', visual_metrics)
    atomic_write_json(run_root / 'paired_results.json', paired_rows)
    atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_text(run_root / 'summary.md', '\n'.join(['# NExT-QA Visual v2 Paired Evaluation', '', f'- Questions: `{len(ordered_ids)}`', f"- Graph-only accuracy: `{summary['graph_only_accuracy']:.6f}`", f"- Visual-enhanced accuracy: `{summary['visual_enhanced_accuracy']:.6f}`", f"- Delta: `{summary['accuracy_delta']:+.6f}`", f"- Improved: `{outcomes['improved']}`", f"- Regressed: `{outcomes['regressed']}`", f"- Answer changed: `{outcomes['answer_changed']}`", f"- Visual API errors: `{summary['visual_error_count']}`", f'- Route/handler purity: `{route_purity}`', '']))
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=sources)
    if drift['code_drift_detected']:
        summary['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_json(run_root / 'run_status.json', {'status': summary['status'], 'question_count': len(ordered_ids), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if summary['status'] != 'passed':
        raise SystemExit('paired evaluation failed purity or source drift')
    print(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
