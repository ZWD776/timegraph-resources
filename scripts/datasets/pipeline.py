"""Common operational CLI with explicit, equal dataset ownership."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from vlkg.core.assets.asset_registry import DATASET_KEYS, inspect_dataset


def make_plan(dataset, stage, run_id, mode='mock'):
    module = import_module(f'vlkg.datasets.{dataset}.runtime.operations')
    result = module.plan(stage, run_id, mode)
    if stage not in ('build', 'index', 'ir', 'retrieve', 'answer'):
        raise ValueError('use status/validate and the dataset evaluator for control operations')
    if not result['commands']:
        raise ValueError('dataset operation has no concrete executable command')
    return {'dataset': dataset, 'stage': stage, 'run_id': run_id, 'mode': mode, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASET_KEYS, required=True)
    parser.add_argument('--stage', choices=('status', 'validate', 'build', 'index', 'ir', 'retrieve', 'answer', 'evaluate', 'audit'), required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--mode', choices=('mock', 'live'), default='mock')
    parser.add_argument('--execute', action='store_true', help='Execute the concrete plan; default prints it without writing or running.')
    args = parser.parse_args()
    if args.stage in ('status', 'validate'):
        result = inspect_dataset(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(bool(result['issues']))
    if args.stage in ('evaluate', 'audit'):
        module = import_module(f'vlkg.datasets.{args.dataset}.audit.replay')
        result = module.replay()
        if args.stage == 'audit':
            result['registry'] = inspect_dataset(args.dataset)
            if result['registry']['issues']:
                raise ValueError('registry audit failed')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not args.run_id:
        parser.error('operational stages require --run-id')
    plan = make_plan(args.dataset, args.stage, args.run_id, args.mode)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    # Run outputs have one dataset owner; a build cannot overwrite active assets.
    output = Path(plan['output_root'])
    if output.exists():
        raise FileExistsError('new run identifier already exists; choose a fresh identifier')
    output.mkdir(parents=True)
    for name, payload in plan['generated_files'].items():
        p = Path(name).resolve()
        if not p.is_relative_to(output.resolve()):
            raise ValueError('generated input escapes the dataset run')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    environment = dict(os.environ)
    environment['VLKG_NEXTQA_RUN_ROOT'] = str(ROOT / 'datasets/nextqa/runs/new_runs/payload')
    (output / 'operation_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for command in plan['commands']:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
