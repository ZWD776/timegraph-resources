"""Verify migrated NExT-QA evidence and replay frozen predictions offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def mapped(path: str, entries: list[dict], root: Path = ROOT) -> Path:
    source = (root / path).resolve()
    for entry in sorted(entries, key=lambda row: len(row['source']), reverse=True):
        prefix = (root / entry['source']).resolve()
        if source == prefix:
            return root / entry['target']
        if source.is_relative_to(prefix):
            return root / entry['target'] / source.relative_to(prefix)
    raise ValueError(f'not in selected migration map: {path}')


def id_set(rows: list[dict]) -> set[str]:
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids) or len(ids) != 4996:
        raise ValueError('expected exactly 4,996 unique question IDs')
    return set(ids)


def verify(plan_path: Path, root: Path = ROOT) -> dict:
    from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy

    entries = load(plan_path)['entries']
    source_root = '.scratch/nextqa/standardized/nextqa_val_standardized_full4996_20260812a'
    labels_old = load(root / source_root / 'labels.evaluator_only.json')
    runtime_old = load(root / source_root / 'questions.runtime.json')
    labels = load(mapped(source_root + '/labels.evaluator_only.json', entries, root))
    runtime = load(mapped(source_root + '/questions.runtime.json', entries, root))
    if labels != labels_old or runtime != runtime_old:
        raise AssertionError('evaluator labels or standardized runtime changed')
    expected = id_set(runtime)
    paths = {
        'ir': '.scratch/nextqa/question_ir/nextqa-ir-v3-luna-full4996-20260817b/question_ir.runtime.json',
        'retrieval': '.scratch/nextqa/graph_retrieval/nextqa-graph-retrieval-v2-irv3-full4996-20260817b/retrieval_records.json',
        'routes': '.scratch/nextqa/visual_routes/nextqa_visual_routes_v2_irv3_full4996_20260818a/route_records.json',
        'visual_evidence': '.scratch/nextqa/visual_evidence/nextqa_visual_evidence_v2_irv3_full4996_20260818a/visual_evidence_records.json',
    }
    for name, path in paths.items():
        rows = load(mapped(path, entries, root))
        if id_set(rows) != expected:
            raise AssertionError(f'question coverage differs: {name}')
        print(f'validated {name}: 4996 IDs', file=sys.stderr, flush=True)
        del rows
    arms = {
        'A': ('.scratch/nextqa/qa/nextqa-graph-qa-luna-irv3-full4996-20260817a/predictions.json', 3024),
        'B': ('.scratch/nextqa/ablation/nextqa_visual_no_routing_qa_v2_full4996_20260901a/predictions.json', 3257),
        'C': ('.scratch/nextqa/ablation/nextqa_visual_route_policy_v3_mainline_composite_20260902a/predictions.json', 3334),
    }
    scores = {}
    for arm, (path, correct) in arms.items():
        old_result = evaluate_exact_choice_accuracy(load(root / path), labels_old, runtime_old)
        result = evaluate_exact_choice_accuracy(load(mapped(path, entries, root)), labels, runtime)
        if result != old_result:
            raise AssertionError(f'complete evaluation differs: {arm}')
        if result['correct_question_count'] != correct:
            raise AssertionError(f'frozen expected score differs: {arm}')
        scores[arm] = {'correct': correct, 'total': 4996,
                      'accuracy': result['overall_accuracy'], 'complete_evaluation_equal': True}
        print(f'replayed {arm}: {correct}/4996', file=sys.stderr, flush=True)
    return {'status': 'passed', 'question_count': 4996, 'coverage_layers': list(paths),
            'frozen_prediction_replay': scores, 'uses_network': False,
            'generates_new_predictions': False,
            'scope': 'Existing evaluator on copied predictions; no rerun of LLM or end-to-end retrieval.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.plan)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'registry/migrations'):
        raise ValueError('verification output must be under registry/migrations')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
