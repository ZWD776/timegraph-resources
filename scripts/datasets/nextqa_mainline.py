"""Run selected NExT-QA stages from the current asset registry.

Existing run resumes continue through their original entrypoints and contracts.
This entry creates new runs; stage-specific output isolation remains unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from vlkg.core.assets.asset_registry import asset_file  # noqa: E402


def stage_command(stage: str, run_id: str, *, answer_mode='mock', device='cpu', root=ROOT) -> list[str]:
    def asset(layer, name):
        return str(asset_file('nextqa', layer, name, root))

    def config(name):
        return str(root / 'configs/datasets/nextqa' / name)

    if not run_id or '/' in run_id or '\\' in run_id or '..' in run_id or ':' in run_id:
        raise ValueError('invalid new run identifier')
    common = ['--run-id', run_id]
    if stage == 'retrieve':
        script_group = 'retrieval'
        script = 'run_graph_retrieval_pilot.py'
        args = ['--scope', 'full', '--question-ir', asset('ir', 'retrieval'),
                '--graph-run-root', str(Path(asset('graphs', 'manifest')).parent),
                '--config', config('retrieval.yaml')]
    elif stage == 'graph-qa':
        script_group = 'qa'
        script = 'run_graph_only_qa_pilot.py'
        args = ['--scope', 'full', '--question-ir', asset('ir', 'runtime'),
                '--retrieval-records', asset('retrieval', 'records'),
                '--config', config('graph_qa.yaml'), '--answer-mode', answer_mode]
    elif stage == 'visual-evidence':
        script_group = 'retrieval'
        script = 'build_visual_evidence_v2.py'
        args = ['--route-records', asset('retrieval', 'routes'),
                '--frozen-graph-evidence', asset('retrieval', 'frozen_graph'),
                '--visual-index-manifest', asset('indexes', 'manifest'),
                '--config', config('visual_runtime.json'), '--device', device]
    elif stage == 'visual-base-qa':
        script_group = 'qa'
        # This reruns the v2 base, not the later route-policy/causal overlay composite.
        script = 'run_visual_enhanced_qa.py'
        args = ['--scope', 'full', '--question-ir', asset('ir', 'runtime'),
                '--frozen-graph-evidence', asset('retrieval', 'frozen_graph'),
                '--visual-evidence', asset('retrieval', 'visual_evidence'),
                '--config', config('visual_qa.yaml'), '--answer-mode', answer_mode]
    elif stage == 'compose-C':
        script_group = 'qa'
        script = 'merge_visual_route_policy_v3.py'
        args = ['--base-predictions', asset('runs', 'causal_base'),
                '--target-overlay-predictions', asset('runs', 'policy_overlay'),
                '--route-records', asset('retrieval', 'routes')]
    else:
        raise ValueError(f'unknown stage: {stage}')
    entrypoint = root / 'scripts/datasets/nextqa' / script_group / script
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    return [sys.executable, str(entrypoint), *common, *args]


def replay(root=ROOT) -> dict:
    from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy

    def load(layer, name):
        return json.loads(asset_file('nextqa', layer, name, root).read_text(encoding='utf-8'))

    labels, questions = load('source', 'labels'), load('source', 'questions')
    output = {}
    for arm in ('A', 'B', 'C'):
        result = evaluate_exact_choice_accuracy(load('runs', arm), labels, questions)
        output[arm] = {'correct': result['correct_question_count'], 'total': len(questions),
                       'accuracy': result['overall_accuracy']}
    return {'mode': 'offline_frozen_prediction_evaluation', 'dataset': 'nextqa', 'arms': output,
            'calls_api': False, 'generates_predictions': False}


def execution_environment(root=ROOT) -> dict[str, str]:
    environment = dict(os.environ)
    environment['VLKG_NEXTQA_RUN_ROOT'] = str(root / 'datasets/nextqa/runs/new_runs/payload')
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('replay', 'retrieve', 'graph-qa', 'visual-evidence', 'visual-base-qa', 'compose-C'))
    parser.add_argument('--run-id')
    parser.add_argument('--answer-mode', choices=('mock', 'live'), default='mock')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.stage == 'replay':
        print(json.dumps(replay(), indent=2))
        return 0
    if not args.run_id:
        parser.error('new stage runs require --run-id')
    command = stage_command(args.stage, args.run_id, answer_mode=args.answer_mode, device=args.device)
    environment = execution_environment()
    if args.dry_run:
        print(json.dumps({'argv': command, 'cwd': str(ROOT), 'executes': False,
                          'output_root': environment['VLKG_NEXTQA_RUN_ROOT']}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
