"""Validate the reviewable final deletion boundary; this tool cannot delete files."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

from scripts.datasets.migrate_asset_storage import ROOT, is_reparse, native_path, safe_path, write_json
from vlkg.core.assets.asset_registry import inspect_dataset


def main():
    plan_path = ROOT / 'registry/gates/final_deletion_plan.json'
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding='utf-8'))
        if existing.get('actual_deletion_performed'):
            audit_path = ROOT / existing['completion_audit']
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            print(json.dumps({'status': existing['status'], 'actual_deletion_performed': True,
                              'completion_audit': existing['completion_audit'],
                              'audit_status': audit['status'], 'removed_logical_bytes': audit['removed_logical_bytes']}))
            if audit['status'] != 'passed':
                raise ValueError('completed deletion audit needs review')
            return
        if existing.get('deletion_execution_enabled'):
            raise ValueError('do not overwrite an explicitly authorized execution scope')
    catalog_path = ROOT / 'registry/audits/final_storage_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if catalog['status'] != 'catalog_complete_pending_acceptance':
        raise ValueError('catalog has unresolved storage/hash coverage issues')
    candidates, retained, groups = set(), set(), {}
    for owner, item in catalog['ledgers'].items():
        blob = (ROOT / item['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            raise ValueError('catalog ledger fingerprint changed: ' + owner)
        counts = {'delete_files': 0, 'delete_bytes': 0, 'retain_files': 0, 'retain_bytes': 0}
        for row in json.loads(gzip.decompress(blob)):
            if not row['sha256'] or row['delete_now'] is not False:
                raise ValueError('missing hash or unexpected execution flag')
            path = safe_path(row['path'])
            if not path.is_relative_to(ROOT / 'archive/legacy_20260928'):
                raise ValueError('catalog row escapes the sealed boundary')
            if row['decision'] == 'delete_candidate':
                if row['path'] in candidates or row['path'] in retained:
                    raise ValueError('conflicting retention decision')
                if not native_path(path).is_file() or native_path(path).stat().st_size != row['bytes']:
                    raise ValueError('candidate disappeared or changed size: ' + row['path'])
                candidates.add(row['path'])
                counts['delete_files'] += 1
                counts['delete_bytes'] += row['bytes']
            else:
                if row['path'] in candidates:
                    raise ValueError('candidate is also retained')
                retained.add(row['path'])
                counts['retain_files'] += 1
                counts['retain_bytes'] += row['bytes']
        groups[owner] = counts
    datasets = {ds: inspect_dataset(ds) for ds in ('navqa', 'nextqa', 'openeqa')}
    if any(result['issues'] for result in datasets.values()):
        raise ValueError('current selected assets have unresolved registry issues')
    hsgg = ['HSGG', 'HSGG_pri', 'HSGG-main', 'HSGG_upload.tgz']
    if any(not (ROOT / path).exists() for path in hsgg):
        raise ValueError('explicitly retained HSGG material is absent')
    free = shutil.disk_usage(ROOT).free
    plan = {'status': 'prepared_before_actual_deletion', 'actual_deletion_performed': False,
        'deletion_execution_enabled': False, 'requires_final_user_deletion_instruction': True,
        'catalog': catalog_path.relative_to(ROOT).as_posix(), 'ledger_hashes': catalog['ledgers'],
        'groups': groups, 'protected_current_roots': ['datasets', 'src', 'scripts', 'configs', 'tests', 'docs', 'registry', '.git', '.venv', *hsgg],
        'storage': {'free_bytes': free, 'reserve_bytes': 20 * 1024**3,
                    'candidate_logical_bytes': sum(row['delete_bytes'] for row in groups.values()),
                    'estimated_free_after_deletion_bytes': free + sum(row['delete_bytes'] for row in groups.values()),
                    'estimate_note': 'Logical bytes; physical recovery can differ with sparse files, compression, hard links and recycle behavior.'},
        'boundary_checks': {'all_candidates_inside_single_archive': True, 'candidate_retention_conflicts': 0,
                            'candidate_size_mismatches': 0, 'current_registry_issues': 0, 'hsgg_materials_present': True},
        'execution_order': ['navqa', 'nextqa', 'openeqa', 'shared'],
        'before_each_removal': ['Recheck full current SHA256 against the ledger.',
            'Keep selected owned payloads and sealed information records excluded.',
            'Remove retired junctions as links only; never traverse their targets.',
            'Write per-file outcome and measure free space after each dataset.'],
        'known_algorithm_limitations': 'registry/audits/final_regression_summary.json'}
    write_json(ROOT / 'registry/gates/final_deletion_plan.json', plan)
    print(json.dumps({'status': plan['status'], 'groups': groups,
                      'candidate_gib': round(plan['storage']['candidate_logical_bytes'] / 1024**3, 3),
                      'free_gib': round(free / 1024**3, 3), 'actual_deletion_performed': False}))


if __name__ == '__main__':
    main()
