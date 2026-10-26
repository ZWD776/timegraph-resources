"""Prepare the approved reversible retirement of prototypes and optional backends."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil

from scripts.datasets.migrate_asset_storage import ROOT, write_json
from scripts.datasets.retired_scope_inventory import RETIRED


def main():
    base = 'archive/legacy_20260928/shared/retired_features'
    sources = list(RETIRED) + ['deploy', 'src/vlkg/integrations',
        'src/vlkg/memory_backends/factory.py',
        'src/vlkg/core/transformers/graphiti_ingest_to_raw_episodes.py',
        'src/vlkg/core/transformers/memory_ir_to_graphiti_ingest.py',
        'configs/sports_basketball.yaml', 'configs/robot_unitree.yaml',
        'tests/test_cognee_projection_backend.py', 'tests/test_memory_backend_factory.py',
        'tests/test_person_label_compatibility.py']
    sources += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/vlkg').glob('*graphiti*.py')]
    for folder in ('scripts/build', 'scripts/query', 'scripts'):
        sources += [p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob('*graphiti*.py')]
    sources += ['scripts/build/load_memory_ir_to_memory_backend.py',
        'scripts/build/query_memory_backend.py',
        'scripts/build/align_basketball_to_robot_style.py',
        'scripts/build/clean_hsgg_occam_dataset.py']
    sources += [f'scripts/eval/{name}.py' for name in (
        'benchmark_incremental_rebuild', 'benchmark_delta_update',
        'diagnose_entity_fragments', 'benchmark_temporal_graph', 'benchmark_llm_fallback')]
    sources = sorted(set(p for p in sources if (ROOT / p).exists()))
    entries = [{'source': p, 'target': f'{base}/retired/{p}',
                'decision': 'delete_entire_retired_feature_after_final_gate'} for p in sources]
    mutable = ['src/vlkg/__init__.py', 'src/vlkg/memory_backends/__init__.py',
        'src/vlkg/core/transformers/__init__.py', 'src/vlkg/core/temporal/pathing.py',
        'tests/test_domain_config.py', 'tests/test_canonical_module_aliases.py', 'pyproject.toml']
    for relative in mutable:
        target = ROOT / base / 'information/before_retirement' / relative
        if target.exists():
            raise FileExistsError('do not overwrite original source snapshot: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(ROOT / 'registry/gates/retired_features_move_plan.json', {
        'action': 'move', 'deletion_allowed': False,
        'receipt': 'registry/gates/retired_features_move_receipt.json',
        'authorization': 'Human explicitly retains HSGG and retires all other named prototypes/backends; actual deletion remains a later step.',
        'retained_hsgg': ['HSGG', 'HSGG_pri', 'HSGG-main', 'HSGG_upload.tgz'],
        'entries': entries})
    print(json.dumps({'units': len(entries), 'action': 'prepared_non_deleting_move_plan'}))


if __name__ == '__main__':
    main()
