"""Audit completed cleanup against immutable ledgers and deletion journals."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.datasets.execute_reviewed_deletion import native, prior_journal


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def journal_file(directory):
    manifest_path = directory / 'journal_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entries = [r for r in manifest['journals'] if Path(r['path']).name == 'files.jsonl']
        if len(entries) != 1:
            raise ValueError('journal manifest must identify one file-removal log')
        raw = ROOT / entries[0]['path']
        compressed = ROOT / entries[0]['compressed_path']
        for path in (raw, compressed):
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError('journal index escapes workspace')
        if raw.is_file():
            return raw
        if compressed.is_file():
            return compressed
        raise FileNotFoundError('indexed deletion journal is missing')
    path = directory / 'files.jsonl'
    if path.exists():
        return path
    compressed = directory / 'files.jsonl.gz'
    if not compressed.is_file():
        raise FileNotFoundError('deletion journal is missing')
    return compressed


def main():
    plan = load('registry/gates/final_deletion_plan.json')
    authorization = load(plan['authorization'])
    if authorization['ledger_hashes'] != plan['ledger_hashes']:
        raise ValueError('execution authorization scope differs from original ledgers')
    directory = ROOT / 'registry/deletion_receipts' / authorization['run_id']
    receipt = load((directory / 'summary.json').relative_to(ROOT))
    removed, verified, links = prior_journal(journal_file(directory))
    issues, rows = [], {}
    for item in plan['ledger_hashes'].values():
        blob = (ROOT / item['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            raise ValueError('original reviewed ledger changed')
        for row in json.loads(gzip.decompress(blob)):
            if row['path'] in rows:
                raise ValueError('duplicate original inventory path')
            rows[row['path']] = row
    original_candidates = {p: r for p, r in rows.items() if r['decision'] == 'delete_candidate'}
    if set(removed) != set(original_candidates) or receipt['status'] != 'completed':
        issues.append('primary deletion journal is incomplete or has extra removals')
    for p in set(removed) & set(original_candidates):
        if (removed[p]['sha256'], removed[p]['bytes']) != (original_candidates[p]['sha256'], original_candidates[p]['bytes']):
            issues.append({'journal_checksum_differs': p})
    corrections = {}
    if plan.get('retention_corrections'):
        for row in load(plan['retention_corrections'])['corrections']:
            old = original_candidates[row['path']]
            if (row['sha256'], row['bytes']) != (old['sha256'], old['bytes']):
                raise ValueError('retention correction differs from original candidate')
            if row['corrected_decision'] != 'retain_sealed_information' or not row['actual_file_restored']:
                raise ValueError('unverified retention correction')
            if hashlib.sha256(native(ROOT / row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('restored sealed information differs from original bytes')
            corrections[row['path']] = row
    primary = {p: r for p, r in original_candidates.items() if p not in corrections}
    supplemental = {}
    shared_paper_paths = set()
    for scope_path in plan.get('supplemental_deletions', []):
        scope = load(scope_path)
        shared_paper_paths.update(row['path'] for row in scope['preserved_shared_materials'])
        for row in scope['delete_candidates']:
            original = rows[row['path']]
            if (original['sha256'], original['bytes']) != (row['sha256'], row['bytes']):
                raise ValueError('supplemental scope differs from original inventory')
            supplemental[row['path']] = row
        subdir = ROOT / scope['receipt_directory']
        if load((subdir / 'summary.json').relative_to(ROOT))['status'] != 'completed':
            issues.append('supplemental execution did not complete')
        sub_removed, _, _ = prior_journal(journal_file(subdir))
        if set(sub_removed) != {r['path'] for r in scope['delete_candidates']}:
            issues.append('supplemental journal incomplete or has extra removals')
        for row in scope['delete_candidates']:
            matched = sub_removed.get(row['path'], {})
            if (matched.get('sha256'), matched.get('bytes')) != (row['sha256'], row['bytes']):
                issues.append({'supplemental_checksum_differs': row['path']})
    retained_count = retained_bytes = paper_hashes = 0
    paper = load('registry/papers/paper_preservation_inventory.json')
    paper_roots = [v['path'] + '/' for k, v in paper['inventory'].items() if k != 'aaai27_paper_draft']
    paper_roots += [p + '/' for p in paper['additional_keep_roots']]
    for p, row in rows.items():
        path = native(ROOT / p)
        if p in primary or p in supplemental:
            if path.exists():
                issues.append({'deleted_candidate_still_present': p})
            continue
        if not path.is_file() or path.stat().st_size != row['bytes']:
            issues.append({'retained_file_missing_or_size_changed': p})
            continue
        retained_count += 1
        retained_bytes += row['bytes']
        if p in shared_paper_paths or any(p.startswith(prefix) for prefix in paper_roots):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                issues.append({'retained_paper_hash_changed': p})
            paper_hashes += 1
    core_paper_refs = 0
    for item in paper['paper_artifact_references']:
        for match in item['preserved_matches']:
            path = native(ROOT / match['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                issues.append({'core_paper_reference_missing_or_changed': match['path']})
            core_paper_refs += 1
    expected_links = {r['target'] for r in load('registry/gates/retired_compatibility_links_receipt.json')['links']}
    if links != expected_links:
        issues.append('junction journal differs from reviewed links')
    for p in expected_links:
        try:
            native(ROOT / p).lstat()
        except FileNotFoundError:
            pass
        else:
            issues.append({'removed_junction_still_present': p})
    for p in ('HSGG', 'HSGG_pri', 'HSGG-main', 'HSGG_upload.tgz'):
        if not (ROOT / p).exists():
            issues.append({'protected_hsgg_missing': p})
    for row in load('registry/current_registry_lock.json')['files']:
        if hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() != row['sha256']:
            issues.append({'current_registry_control_changed': row['path']})
    from vlkg.core.assets.asset_registry import inspect_dataset, verify_metadata_copies
    datasets = {ds: inspect_dataset(ds) for ds in ('navqa', 'nextqa', 'openeqa')}
    for ds, result in datasets.items():
        if result['issues']:
            issues.append({'dataset': ds, 'issues': result['issues']})
    metadata_issues = verify_metadata_copies(ROOT)
    if metadata_issues:
        issues.append({'metadata_copy_issues': metadata_issues})
    free = shutil.disk_usage(ROOT).free
    result = {'status': 'passed' if not issues else 'needs_review', 'issues': issues,
              'primary_removed_files': len(primary), 'supplemental_removed_files': len(supplemental),
              'original_primary_removal_operations': len(removed), 'restored_sealed_information_files': len(corrections),
              'removed_logical_bytes': sum(r['bytes'] for r in [*primary.values(), *supplemental.values()]),
              'retained_archive_files_checked': retained_count, 'retained_archive_bytes': retained_bytes,
              'retained_paper_full_hashes_checked': paper_hashes, 'junction_links_removed': len(links),
              'core_paper_references_checked': core_paper_refs,
              'protected_hsgg_present': True, 'current_registry': datasets,
              'free_bytes_before': authorization['free_bytes_before'], 'free_bytes_after': free,
              'observed_free_space_increase_bytes': free - authorization['free_bytes_before'],
              'actual_deletion_performed': True}
    (ROOT / 'registry/audits/postdeletion_inventory_20261002.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'current_registry'}, ensure_ascii=False), flush=True)
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
