"""Evaluate a fixed NExT-QA visual route policy v3 prediction composite."""
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
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.route_policy_v3 import POLICY_ID, TARGET_IDENTITIES, TARGET_ROUTE_IDS
DEFAULT_LABELS = (REPO_ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation/labels.evaluator_only.json')
DEFAULT_RUNTIME = (REPO_ROOT / 'datasets/nextqa/ir/nextqa-ir-v3-luna-full4996/payload/nextqa-ir-v3-luna-full4996-20260817b/question_ir.runtime.json')
DEFAULT_GRAPH = (REPO_ROOT / 'datasets/nextqa/runs/nextqa-cg6-qa-v3-luna-full4996/payload/nextqa-graph-qa-luna-irv3-full4996-20260817a/predictions.json')
DEFAULT_ROUTES = (REPO_ROOT / 'datasets/nextqa/retrieval/nextqa-cg6-retrieval-v3-route-policy-v3/payload/nextqa_visual_routes_v2_irv3_full4996_20260818a/route_records.json')

def _rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list')
    return payload

def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        qid = str(row.get('question_id') or '')
        if not qid or qid in output:
            raise ValueError(f'{name} contains an empty or duplicate question ID')
        output[qid] = row
    return output

def _correct(row: dict[str, Any], gold: int) -> bool:
    return row.get('status') == 'completed' and row.get('predicted_choice_index') == gold

def _outcome(before: bool, after: bool) -> str:
    if not before and after:
        return 'improved'
    if before and (not after):
        return 'regressed'
    return 'unchanged_correct' if before else 'unchanged_incorrect'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate fixed NExT-QA visual route policy v3.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--composite-predictions', type=Path, required=True)
    parser.add_argument('--base-predictions', type=Path, required=True)
    parser.add_argument('--graph-only-predictions', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--runtime-questions', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--route-records', type=Path, default=DEFAULT_ROUTES)
    parser.add_argument('--resume', action='store_true')
    return parser

def main() -> int:
    args = build_parser().parse_args()
    paths = {name: value.resolve() for name, value in {'composite': args.composite_predictions, 'base': args.base_predictions, 'graph': args.graph_only_predictions, 'labels': args.labels, 'runtime': args.runtime_questions, 'routes': args.route_records}.items()}
    composite = _index(_rows(paths['composite'], 'composite predictions'), 'composite predictions')
    base = _index(_rows(paths['base'], 'base predictions'), 'base predictions')
    graph = _index(_rows(paths['graph'], 'graph predictions'), 'graph predictions')
    labels = _index(_rows(paths['labels'], 'labels'), 'labels')
    runtime = _index(_rows(paths['runtime'], 'runtime'), 'runtime')
    routes = _index(_rows(paths['routes'], 'routes'), 'routes')
    ids = set(routes)
    if len(ids) != 4996 or any((set(rows) != ids for rows in (composite, base, graph, labels, runtime))):
        raise ValueError('all evaluation inputs must exactly cover 4,996 questions')
    run_root = require_within_nextqa_scratch((REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'ablation' / args.run_id)
    if run_root.exists() and (not args.resume):
        raise FileExistsError(f'run already exists: {run_root}')
    if not run_root.exists() and args.resume:
        raise FileNotFoundError(f'cannot resume missing run: {run_root}')
    run_root.mkdir(parents=True, exist_ok=args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'visual_route_policy_v3_exact_choice_evaluation', 'policy_id': POLICY_ID, **{name: fingerprint_file(path) for name, path in paths.items()}, 'question_count': 4996, 'metric': 'exact_choice_accuracy', 'uses_gpt_judge': False, 'selects_better_answer_per_question': False, 'calls_api': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_route_policy_v3_evaluation_run_config_v1')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    ordered = sorted(ids)
    label_rows = [labels[qid] for qid in ordered]
    runtime_rows = [runtime[qid] for qid in ordered]
    graph_metrics = evaluate_exact_choice_accuracy([graph[qid] for qid in ordered], label_rows, runtime_rows)
    base_metrics = evaluate_exact_choice_accuracy([base[qid] for qid in ordered], label_rows, runtime_rows)
    composite_metrics = evaluate_exact_choice_accuracy([composite[qid] for qid in ordered], label_rows, runtime_rows)
    outcomes = Counter()
    route_counts = Counter()
    route_correct = Counter()
    target_outcomes = {route: Counter() for route in sorted(TARGET_ROUTE_IDS)}
    paired = []
    for qid in ordered:
        gold = int(labels[qid]['gold_choice_index'])
        before = _correct(base[qid], gold)
        after = _correct(composite[qid], gold)
        outcome = _outcome(before, after)
        outcomes[outcome] += 1
        route_id = str(routes[qid]['route_id'])
        route_counts[route_id] += 1
        route_correct[route_id] += int(after)
        if route_id in target_outcomes:
            target_outcomes[route_id][outcome] += 1
        paired.append({'question_id': qid, 'video_id': str(runtime[qid]['video_id']), 'official_type': str(runtime[qid]['official_type']), 'route_id': route_id, 'gold_choice_index': gold, 'base_choice_index': base[qid].get('predicted_choice_index'), 'composite_choice_index': composite[qid].get('predicted_choice_index'), 'base_correct': before, 'composite_correct': after, 'outcome': outcome})
    target_pure = all((composite[qid].get('strategy_id') == TARGET_IDENTITIES[str(routes[qid]['route_id'])]['strategy_id'] and composite[qid].get('handler_id') == TARGET_IDENTITIES[str(routes[qid]['route_id'])]['handler_id'] and (composite[qid].get('response_reused_by_exact_request_sha256') is True) for qid in ordered if str(routes[qid]['route_id']) in TARGET_ROUTE_IDS))
    non_target_unchanged = all((composite[qid] == base[qid] for qid in ordered if str(routes[qid]['route_id']) not in TARGET_ROUTE_IDS))
    summary = {'schema_version': 'nextqa_visual_route_policy_v3_evaluation_v1', 'run_id': args.run_id, 'policy_id': POLICY_ID, 'question_count': 4996, 'graph_only_accuracy': graph_metrics['overall_accuracy'], 'base_accuracy': base_metrics['overall_accuracy'], 'composite_accuracy': composite_metrics['overall_accuracy'], 'composite_correct': composite_metrics['correct_question_count'], 'delta_vs_graph_only': composite_metrics['overall_accuracy'] - graph_metrics['overall_accuracy'], 'delta_vs_base': composite_metrics['overall_accuracy'] - base_metrics['overall_accuracy'], 'base_to_composite_outcomes': dict(sorted(outcomes.items())), 'target_route_outcomes': {route: dict(sorted(values.items())) for route, values in target_outcomes.items()}, 'composite_route_accuracy': {route: {'correct': route_correct[route], 'count': count, 'accuracy': route_correct[route] / count} for route, count in sorted(route_counts.items())}, 'composite_official_type_accuracy': composite_metrics['official_type_accuracy'], 'checks': {'full_question_coverage': len(composite) == 4996, 'target_strategy_purity': target_pure, 'six_non_target_routes_unchanged': non_target_unchanged}, 'uses_gpt_judge': False, 'selects_better_answer_per_question': False, 'status': 'passed' if target_pure and non_target_unchanged else 'failed'}
    atomic_write_json(run_root / 'graph_only_metrics.json', graph_metrics)
    atomic_write_json(run_root / 'base_metrics.json', base_metrics)
    atomic_write_json(run_root / 'composite_metrics.json', composite_metrics)
    atomic_write_json(run_root / 'paired_results.json', paired)
    atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_text(run_root / 'summary.md', '\n'.join(['# NExT-QA Visual Route Policy V3 Evaluation', '', f'- Questions: `4996`', f"- Graph-only: `{summary['graph_only_accuracy']:.6f}`", f"- Base: `{summary['base_accuracy']:.6f}`", f"- Composite: `{summary['composite_accuracy']:.6f}`", f"- Delta vs base: `{summary['delta_vs_base']:+.6f}`", f"- Base-to-composite: `{summary['base_to_composite_outcomes']}`", '- Per-question answer selection: `False`', f"- Status: `{summary['status']}`", '']))
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    if drift['code_drift_detected']:
        summary['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'summary.json', summary)
    atomic_write_json(run_root / 'run_status.json', {'status': summary['status'], 'question_count': 4996, 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if summary['status'] != 'passed':
        raise SystemExit('visual route policy v3 evaluation failed')
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
