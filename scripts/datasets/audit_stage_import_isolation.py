"""Exercise all selected stage imports with sibling datasets and retired backends denied."""
from __future__ import annotations

import argparse
import importlib
import importlib.abc
import json
from pathlib import Path
import sys

from scripts.datasets.migrate_asset_storage import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('navqa', 'nextqa', 'openeqa'))
    args = parser.parse_args()
    forbidden = tuple(name for name in ('navqa', 'nextqa', 'openeqa') if name != args.dataset)
    violations = []
    class DenyForeign(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            blocked = fullname.startswith(('graphiti_core', 'cognee', 'vlkg.integrations')) or any(
                fullname == prefix + name or fullname.startswith(prefix + name + '.')
                for prefix in ('vlkg.datasets.', 'vlkg.eval_adapters.') for name in forbidden)
            if blocked:
                violations.append(fullname)
                raise ImportError('selected stage imported retired or sibling dependency: ' + fullname)
    sys.meta_path.insert(0, DenyForeign())
    from scripts.datasets.pipeline import make_plan
    stages = {}
    for stage in ('build', 'index', 'ir', 'retrieve', 'answer'):
        modules = []
        for command in make_plan(args.dataset, stage, 'import-isolation', 'mock')['commands']:
            path = Path(command[1]).relative_to(ROOT).with_suffix('')
            name = '.'.join(path.parts)
            importlib.import_module(name)
            modules.append(name)
        stages[stage] = modules
    foreign_loaded = [name for name in sys.modules if any(
        name == 'vlkg.datasets.' + ds or name.startswith('vlkg.datasets.' + ds + '.') for ds in forbidden)]
    report = {'dataset': args.dataset, 'status': 'passed' if not violations and not foreign_loaded else 'failed',
              'stages': stages, 'blocked_attempts': violations, 'foreign_dataset_modules_loaded': foreign_loaded,
              'calls_api': False, 'scope': 'Executed imports for all selected stage command modules, including dynamic imports at import time. Runtime probes cover executed data access separately.'}
    out = ROOT / 'registry/audits/stage_import_isolation' / (args.dataset + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
