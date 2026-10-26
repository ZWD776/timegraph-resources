"""Inventory the final boundary, reusing verified move hashes; never delete files."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time

from scripts.datasets.build_deletion_catalog import classify
from scripts.datasets.migrate_asset_storage import ROOT, is_reparse, native_path, sha256, write_json
from vlkg.core.assets.asset_provenance import historical_control_file


ARCHIVE = 'archive/legacy_20260928'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-archive', action='store_true', help='Refresh the changed archive only; preserve the timestamped complete footprint measurement.')
    args = parser.parse_args()
    previous_path = ROOT / 'registry/audits/final_storage_catalog.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if args.refresh_archive else None
    previous_hashes = {}
    if previous:
        for item in previous['ledgers'].values():
            blob = (ROOT / item['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest() != item['sha256']:
                raise ValueError('previous catalog fingerprint differs')
            previous_hashes.update({row['path']: row for row in json.loads(gzip.decompress(blob))})
    receipt_paths = sorted((ROOT / (ARCHIVE + '/governance/registry/migrations')).glob('*move_receipt.json'))
    receipt_paths += sorted((ROOT / 'registry/gates').glob('*move_receipt.json'))
    moves, entries, incomplete, code_receipts = {}, [], [], []
    for path in receipt_paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'entries' not in data and 'files' in data:
            code_receipts.append({'path': path.relative_to(ROOT).as_posix(), 'historical_status': data.get('status'),
                                 'scope': 'Code namespace receipt, not a storage move ledger.',
                                 'current_acceptance': 'All selected stage imports and current moved-code regressions pass; source snapshots and Git checkpoints preserve originals.'})
            continue
        if data.get('status') != 'verified':
            incomplete.append(path.relative_to(ROOT).as_posix())
            continue
        for entry in data.get('entries', []):
            if entry.get('status') != 'verified' or 'file_ledger' not in entry:
                incomplete.append(path.relative_to(ROOT).as_posix())
                continue
            moves[entry['source']] = entry['target']
            entries.append((path.relative_to(ROOT).as_posix(), entry))

    def relocate(path):
        seen = set()
        while path not in seen:
            seen.add(path)
            parts = path.split('/')
            match = next(('/'.join(parts[:length]) for length in range(len(parts), 0, -1)
                          if '/'.join(parts[:length]) in moves), None)
            if match is None:
                return path
            path = moves[match] + path[len(match):]
        raise ValueError('cyclic move chain: ' + path)

    hashes, direct_priority, conflicts = {}, {}, []
    for receipt, entry in entries:
        ledger = historical_control_file(entry['file_ledger'], ROOT).read_bytes()
        if hashlib.sha256(ledger).hexdigest() != entry['file_ledger_sha256']:
            raise ValueError('ledger checksum differs: ' + entry['file_ledger'])
        for row in json.loads(gzip.decompress(ledger)):
            old_target = entry['target'] + ('/' + row['relative'] if row['relative'] else '')
            target = relocate(old_target)
            priority = int(target == old_target)
            if target in hashes and hashes[target]['sha256'] != row['sha256']:
                if direct_priority[target] == priority:
                    conflicts.append(target)
            if target not in hashes or priority > direct_priority[target]:
                hashes[target] = {'bytes': row['bytes'], 'sha256': row['sha256'], 'move_receipt': receipt}
                direct_priority[target] = priority

    retained_hashes = {}
    for path, proof in hashes.items():
        if path.startswith('datasets/'):
            file = native_path(ROOT / path)
            if file.is_file() and file.stat().st_size == proof['bytes']:
                retained_hashes.setdefault(proof['sha256'], path)
    container_review = ROOT / (ARCHIVE + '/migration/deletion_catalog/container_review.json')
    container_decisions = {}
    if container_review.exists():
        container_decisions = {row['path']: row for row in json.loads(container_review.read_text(encoding='utf-8'))['containers']}

    categories, groups, links, errors, uncovered, mismatches = defaultdict(Counter), defaultdict(list), [], [], [], []
    stats = defaultdict(lambda: {'files': 0, 'bytes': 0})
    started = time.monotonic()
    last_notice = started

    def walk_error(exc):
        errors.append(str(exc))

    walk_root = ROOT / ARCHIVE if args.refresh_archive else ROOT
    for directory, dirs, files in os.walk(native_path(walk_root), onerror=walk_error):
        relative_dir = Path(directory).relative_to(native_path(ROOT)).as_posix()
        for name in list(dirs):
            child = Path(directory, name)
            if is_reparse(child):
                links.append(child.relative_to(native_path(ROOT)).as_posix())
                dirs.remove(name)
        for name in files:
            child = Path(directory, name)
            path = child.relative_to(native_path(ROOT)).as_posix()
            if is_reparse(child):
                links.append(path)
                continue
            size = child.stat().st_size
            parts = path.split('/')
            bucket = '/'.join(parts[:3]) if path.startswith('archive/') else '/'.join(parts[:2]) if path.startswith('datasets/') else parts[0]
            stats[bucket]['files'] += 1
            stats[bucket]['bytes'] += size
            if not path.startswith(ARCHIVE + '/') or path.startswith(ARCHIVE + '/migration/final_deletion_catalog/'):
                continue
            proof = hashes.get(path)
            if proof is not None and proof['bytes'] != size:
                mismatches.append({'path': path, 'ledger_bytes': proof['bytes'], 'actual_bytes': size})
                proof = None
            if proof is None:
                prior = previous_hashes.get(path)
                if prior and prior['bytes'] == size and prior.get('sha256'):
                    proof = {'sha256': prior['sha256'], 'move_receipt': prior['move_receipt']}
                elif size > 64 * 1024**2:
                    uncovered.append({'path': path, 'bytes': size})
                    proof = {'sha256': None, 'move_receipt': None}
                else:
                    proof = {'sha256': sha256(child), 'move_receipt': None}
            owner = parts[2] if parts[2] in ('navqa', 'nextqa', 'openeqa') else 'shared'
            classification = classify(path)
            if owner != 'shared' and '/assets/' in path and name in {
                'retrieval_records.json', 'visual_evidence_records.json',
                'question_ir_records.json', 'question_ir.json',
                'route_manifest.json', 'routes.json', 'visual_routes.json'}:
                classification = 'derived_runtime_asset_candidate'
            if owner != 'shared' and '/original_mainline/index_metadata/' in path and name.endswith(('.metadata.json', '.metadata.jsonl')):
                classification = 'superseded_index_metadata_asset_candidate'
            decision, reason = 'retain_sealed_information', 'Historical code, configuration, reports, logs or provenance remain sealed.'
            if owner == 'shared':
                if '/shared/retired_features/' in path:
                    decision, reason = 'delete_candidate', 'Human explicitly retired robot/basketball prototypes and Graphiti/Cognee.'
                elif '/shared/environment_state/current_closeout/.npm-cache/' in path:
                    decision, reason = 'delete_candidate', 'Generated package cache; retained runtimes do not load it.'
                elif '/shared/environment_state/current_closeout/data/cache/' in path:
                    decision, reason = 'delete_candidate', 'Superseded shared embedding database; selected mainlines use their own verified caches.'
                elif '/shared/assets/scratch_history/' in path and classification.endswith('_candidate'):
                    decision, reason = 'delete_candidate', 'Superseded scratch asset; current runtime denies old scratch roots.'
                else:
                    decision, reason = 'retain_sealed_information', 'Preserve shared research, papers, HSGG and environment evidence; outside authorized dataset/prototype deletion scope.'
            elif '/information/' not in path and classification in {
                'graph_asset_candidate', 'binary_asset_candidate', 'image_asset_candidate',
                'old_structured_input_candidate', 'generated_cache_or_fixture_candidate',
                'derived_runtime_asset_candidate', 'superseded_index_metadata_asset_candidate'}:
                decision, reason = 'delete_candidate', 'Unselected or superseded asset is sealed; selected mainline resources are owned under datasets/.'
            if classification == 'container_requires_content_review' or classification == 'retain_original_video_reference':
                duplicate = retained_hashes.get(proof['sha256'])
                reviewed = container_decisions.get(path)
                if duplicate:
                    decision, reason = 'delete_candidate', 'Byte-identical retained owned asset: ' + duplicate
                elif reviewed and reviewed['decision'] in {'duplicate_container_candidate', 'generated_test_fixture_candidate'}:
                    decision, reason = 'delete_candidate', 'Container members/retained-copy hash reviewed in deletion_catalog/container_review.json.'
                elif '/shared/retired_features/' not in path:
                    decision, reason = 'retain_sealed_source_reference', 'Keep original source/container reference; no unsupported duplicate assumption.'
            if '/governance/' in path or '/migration/' in path or '/information/' in path and owner != 'shared':
                decision, reason = 'retain_sealed_information', 'Preserve sealed audit, code and historical information.'
            row = {'path': path, 'bytes': size, 'sha256': proof['sha256'],
                   'hash_evidence': 'verified_move_receipt' if proof['move_receipt'] else 'current_small_file_hash' if proof['sha256'] else 'missing',
                   'move_receipt': proof['move_receipt'], 'classification': classification,
                   'decision': decision, 'reason': reason, 'delete_now': False,
                   'required_before_deletion': 'Recheck current SHA256 and active-reference guard immediately before any file removal.'}
            groups[owner].append(row)
            categories[owner][decision + '_files'] += 1
            categories[owner][decision + '_bytes'] += size
        if time.monotonic() - last_notice > 20:
            print('inventory progress:', sum(v['files'] for v in stats.values()), 'files', flush=True)
            last_notice = time.monotonic()

    out = ROOT / (ARCHIVE + '/migration/final_deletion_catalog')
    out.mkdir(parents=True, exist_ok=True)
    ledgers = {}
    for owner, rows in groups.items():
        rows.sort(key=lambda row: row['path'])
        blob = gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0)
        path = out / (owner + '.files.json.gz')
        path.write_bytes(blob)
        ledgers[owner] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(blob).hexdigest(),
                         'files': len(rows), **dict(categories[owner])}
    if previous:
        measured = {key: value for key, value in previous['footprint'].items() if not key.startswith('archive/')}
        measured.update(stats)
        stats = measured
    report = {'status': 'catalog_complete_pending_acceptance' if not (errors or uncovered or mismatches or incomplete or conflicts) else 'catalog_requires_review',
              'actual_deletion_performed': False, 'approval_ready': False,
              'footprint': dict(stats), 'ledgers': ledgers, 'reparse_points_not_traversed': links,
              'walk_errors': errors, 'uncovered_large_files': uncovered, 'size_mismatches': mismatches,
              'incomplete_receipts': sorted(set(incomplete)), 'conflicting_hash_evidence': sorted(set(conflicts)),
              'historical_code_receipts': code_receipts,
              'footprint_scope': 'Complete workspace measurement; archive refreshed after closing moves.' if previous else 'Complete workspace, including Git, environment and all local payloads; links not traversed.',
              'hash_policy': 'Reuse verified move SHA256 for large assets; hash uncovered small information files once. Actual deletion must recheck hashes immediately before removing each file.',
              'disk': {'free_bytes': shutil.disk_usage(ROOT).free, 'reserve_bytes': 20 * 1024**3},
              'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(out / 'summary.json', report)
    write_json(ROOT / 'registry/audits/final_storage_catalog.json', report)
    print(json.dumps({'status': report['status'], 'ledgers': ledgers,
                      'issues': {key: len(report[key]) for key in ('walk_errors', 'uncovered_large_files', 'size_mismatches', 'incomplete_receipts', 'conflicting_hash_evidence')},
                      'free_gib': round(report['disk']['free_bytes'] / 1024**3, 2)}))


if __name__ == '__main__':
    main()
