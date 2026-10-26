"""Evaluate Graph-only, uniform visual, and routed visual NExT-QA results."""
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
from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text, require_within_nextqa_scratch
from vlkg.datasets.nextqa.runtime.run_snapshot import fingerprint_file, persist_run_config, persist_source_snapshot, persist_drift_report, collect_imported_repo_sources
DEFAULT_LABELS = (REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/labels.evaluator_only.json')
DEFAULT_RUNTIME = (REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/questions.runtime.json')
DEFAULT_GRAPH = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'qa' / 'nextqa-graph-qa-luna-irv3-full4996-evalfix-20260817a' / 'predictions.json'
DEFAULT_ROUTED = (REPO_ROOT / 'datasets/nextqa/runs/nextqa-cg6-qa-v3-luna-full4996/payload/nextqa-visual-qa-luna-v2-irv3-full4996-20260818a/predictions.json')

def _load(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get('question_id') or '')
        if not qid or qid in out:
            raise ValueError(f'{name} contains an empty or duplicate question ID: {qid}')
        out[qid] = row
    return out

def _correct(prediction: dict[str, Any], gold: int) -> bool:
    return prediction.get('status') == 'completed' and prediction.get('predicted_choice_index') == gold

def _paired(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]], runtime: dict[str, dict[str, Any]], left_name: str, right_name: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    outcomes = Counter()
    rows: list[dict[str, Any]] = []
    for qid in sorted(runtime):
        gold = int(labels[qid]['gold_choice_index'])
        left_correct = _correct(left[qid], gold)
        right_correct = _correct(right[qid], gold)
        if not left_correct and right_correct:
            outcome = 'improved'
        elif left_correct and (not right_correct):
            outcome = 'regressed'
        elif left_correct and right_correct:
            outcome = 'unchanged_correct'
        else:
            outcome = 'unchanged_incorrect'
        changed = left[qid].get('predicted_choice_index') != right[qid].get('predicted_choice_index')
        outcomes[outcome] += 1
        outcomes['answer_changed' if changed else 'answer_unchanged'] += 1
        rows.append({'question_id': qid, 'video_id': str(runtime[qid]['video_id']), 'official_type': str(runtime[qid]['official_type']), 'gold_choice_index': gold, f'{left_name}_choice_index': left[qid].get('predicted_choice_index'), f'{right_name}_choice_index': right[qid].get('predicted_choice_index'), f'{left_name}_correct': left_correct, f'{right_name}_correct': right_correct, 'answer_changed': changed, 'outcome': outcome})
    return (dict(sorted(outcomes.items())), rows)

def _type_table(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return metrics.get('official_type_accuracy', {})

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Exact-choice three-way NExT-QA ablation evaluation.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--no-routing-predictions', type=Path, required=True)
    parser.add_argument('--graph-only-predictions', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--routed-predictions', type=Path, default=DEFAULT_ROUTED)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--runtime-questions', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--resume', action='store_true')
    return parser

def main() -> int:
    args = build_parser().parse_args()
    labels_path = args.labels.resolve()
    runtime_path = args.runtime_questions.resolve()
    graph_path = args.graph_only_predictions.resolve()
    routed_path = args.routed_predictions.resolve()
    no_route_path = args.no_routing_predictions.resolve()
    labels = _index(_load(labels_path, 'labels'), 'labels')
    runtime = _index(_load(runtime_path, 'runtime questions'), 'runtime questions')
    graph = _index(_load(graph_path, 'graph-only predictions'), 'graph-only predictions')
    routed = _index(_load(routed_path, 'routed visual predictions'), 'routed visual predictions')
    no_route = _index(_load(no_route_path, 'no-routing predictions'), 'no-routing predictions')
    expected = set(runtime)
    if set(labels) != expected or set(graph) != expected or set(routed) != expected or (set(no_route) != expected):
        raise ValueError('all three prediction sets, labels, and runtime questions must have exactly the same IDs')
    run_root = require_within_nextqa_scratch((REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'ablation' / args.run_id)
    if run_root.exists() and (not args.resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {run_root}')
    if not run_root.exists() and args.resume:
        raise FileNotFoundError(f'cannot resume missing run: {run_root}')
    run_root.mkdir(parents=True, exist_ok=args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'three_way_visual_ablation_evaluation_v1', 'metric': 'exact_choice_accuracy', 'uses_gpt_judge': False, 'selects_better_answer_per_question': False, 'labels': fingerprint_file(labels_path), 'runtime_questions': fingerprint_file(runtime_path), 'graph_only_predictions': fingerprint_file(graph_path), 'visual_without_routing_predictions': fingerprint_file(no_route_path), 'visual_with_routing_predictions': fingerprint_file(routed_path), 'question_count': len(expected)}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_three_way_ablation_run_config_v1')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    labels_list = [labels[qid] for qid in sorted(expected)]
    runtime_list = [runtime[qid] for qid in sorted(expected)]
    graph_list = [graph[qid] for qid in sorted(expected)]
    no_route_list = [no_route[qid] for qid in sorted(expected)]
    routed_list = [routed[qid] for qid in sorted(expected)]
    metrics = {'graph_only': evaluate_exact_choice_accuracy(graph_list, labels_list, runtime_list), 'visual_without_routing': evaluate_exact_choice_accuracy(no_route_list, labels_list, runtime_list), 'visual_with_routing': evaluate_exact_choice_accuracy(routed_list, labels_list, runtime_list)}
    ab, ab_rows = _paired(graph, no_route, labels, runtime, 'graph_only', 'visual_without_routing')
    ac, ac_rows = _paired(graph, routed, labels, runtime, 'graph_only', 'visual_with_routing')
    bc, bc_rows = _paired(no_route, routed, labels, runtime, 'visual_without_routing', 'visual_with_routing')
    a = metrics['graph_only']['overall_accuracy']
    b = metrics['visual_without_routing']['overall_accuracy']
    c = metrics['visual_with_routing']['overall_accuracy']
    formal_table = [{'configuration': 'Graph-only', 'correct': metrics['graph_only']['correct_question_count'], 'total': len(expected), 'exact_accuracy': a, 'status': 'formal reference'}, {'configuration': 'Visual without routing', 'correct': metrics['visual_without_routing']['correct_question_count'], 'total': len(expected), 'exact_accuracy': b, 'status': 'ablation'}, {'configuration': 'Visual + routing', 'correct': metrics['visual_with_routing']['correct_question_count'], 'total': len(expected), 'exact_accuracy': c, 'status': 'formal comparison'}]
    summary = {'schema_version': 'nextqa_three_way_ablation_evaluation_v1', 'question_count': len(expected), 'formal_table': formal_table, 'increments': {'visual_gain_b_minus_a': b - a, 'routing_gain_c_minus_b': c - b, 'total_gain_c_minus_a': c - a}, 'paired_outcomes': {'graph_only_vs_visual_without_routing': ab, 'graph_only_vs_visual_with_routing': ac, 'visual_without_routing_vs_visual_with_routing': bc}, 'selects_better_answer_per_question': False, 'uses_gpt_judge': False, 'by_official_type': {'graph_only': _type_table(metrics['graph_only']), 'visual_without_routing': _type_table(metrics['visual_without_routing']), 'visual_with_routing': _type_table(metrics['visual_with_routing'])}, 'api_error_count': {'graph_only': metrics['graph_only']['error_question_count'], 'visual_without_routing': metrics['visual_without_routing']['error_question_count'], 'visual_with_routing': metrics['visual_with_routing']['error_question_count']}, 'purity': {'visual_without_routing_ablation_label': all((row.get('ablation_label') == 'no_routing' and 'route_id' not in row and ('handler_id' not in row) for row in no_route_list)), 'visual_with_routing_formal_route': all(('route_id' in row and 'handler_id' in row for row in routed_list))}, 'status': 'passed'}
    atomic_write_json(run_root / 'graph_only_metrics.json', metrics['graph_only'])
    atomic_write_json(run_root / 'visual_without_routing_metrics.json', metrics['visual_without_routing'])
    atomic_write_json(run_root / 'visual_with_routing_metrics.json', metrics['visual_with_routing'])
    atomic_write_json(run_root / 'paired_graph_vs_no_routing.json', ab_rows)
    atomic_write_json(run_root / 'paired_graph_vs_routed.json', ac_rows)
    atomic_write_json(run_root / 'paired_no_routing_vs_routed.json', bc_rows)
    atomic_write_json(run_root / 'summary.json', summary)
    lines = ['# NExT-QA Three-Way Visual Ablation', '', '| Configuration | Correct / Total | Exact Accuracy | Status |', '|---|---:|---:|---|']
    for row in formal_table:
        lines.append(f"| {row['configuration']} | {row['correct']} / {row['total']} | {row['exact_accuracy']:.3%} | {row['status']} |")
    lines += ['', f'- Visual gain (B-A): `{b - a:+.3%}`', f'- Routing gain (C-B): `{c - b:+.3%}`', f'- Total gain (C-A): `{c - a:+.3%}`', '', '## Paired outcomes', '', f'- Graph-only vs B: `{ab}`', f'- Graph-only vs C: `{ac}`', f'- B vs C: `{bc}`', '- selects_better_answer_per_question: `false`', '', '## Official type', '', '```json', json.dumps(summary['by_official_type'], ensure_ascii=False, indent=2, sort_keys=True), '```', '']
    atomic_write_text(run_root / 'summary.md', '\n'.join(lines))
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    if drift['code_drift_detected']:
        summary['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_json(run_root / 'run_status.json', {'status': summary['status'], 'question_count': len(expected), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary['status'] == 'passed' else 1
if __name__ == '__main__':
    raise SystemExit(main())
