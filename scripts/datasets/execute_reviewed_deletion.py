"""Preview reviewed cleanup; execution requires a later explicit enabled gate.

Remove individual verified files only. Never recursively remove a directory.
This script is maintenance tooling, not a dataset/graph algorithm change.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PREFIX = 'archive/legacy_20260928/'


def native(path):
    value = str(path.absolute())
    return Path('\\\\?\\' + value) if os.name == 'nt' and not value.startswith('\\\\?\\') else path


def checked_candidate(row, root=ROOT):
    value = row.get('path', '')
    if row.get('decision') != 'delete_candidate' or row.get('delete_now') is not False:
        raise ValueError('retained or unreviewed row cannot be removed')
    if (not value.startswith(ARCHIVE_PREFIX) or '\\' in value or ':' in value
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('candidate escapes the declared archive')
    root = root.resolve()
    path = root
    for part in value.split('/'):
        path = path / part
        info = native(path).lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('candidate path traverses a reparse point')
    if not path.resolve().is_relative_to(root / 'archive/legacy_20260928'):
        raise ValueError('resolved target escapes archive')
    if not stat.S_ISREG(info.st_mode) or info.st_size != row['bytes']:
        raise ValueError('candidate type/size changed')
    return path


def verified_file(row, root=ROOT):
    path = checked_candidate(row, root)
    digest = hashlib.sha256()
    with native(path).open('rb') as stream:
        initial = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
        final = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if digest.hexdigest() != row['sha256'] or identity(initial) != identity(final):
        raise ValueError('current candidate SHA256 or file identity changed')
    checked_candidate(row, root)
    if identity(native(path).stat()) != identity(final):
        raise ValueError('candidate changed after hashing')
    return path


def remove_verified_file(row, journal, root=ROOT):
    path = verified_file(row, root)
    # Journal the matched checksum before the irreversible operation.
    journal({'path': row['path'], 'status': 'hash_verified_before_removal', 'sha256': row['sha256'], 'bytes': row['bytes']})
    if getattr(native(path).stat(), 'st_file_attributes', 0) & 1:
        # Only the exact reviewed file may have its read-only flag cleared.
        os.chmod(native(path), stat.S_IWRITE | stat.S_IREAD)
    native(path).unlink()
    journal({'path': row['path'], 'status': 'removed', 'bytes': row['bytes']})


def load_scope(plan, root=ROOT):
    groups = {}
    for owner, item in plan['ledger_hashes'].items():
        blob = (root / item['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            raise ValueError('reviewed ledger changed')
        groups[owner] = [row for row in json.loads(gzip.decompress(blob)) if row['decision'] == 'delete_candidate']
        if (len(groups[owner]) != plan['groups'][owner]['delete_files']
                or sum(row['bytes'] for row in groups[owner]) != plan['groups'][owner]['delete_bytes']):
            raise ValueError('reviewed group size/count differs')
    return groups


def prior_journal(path):
    removed, verified, links = {}, {}, set()
    if not path.exists():
        return removed, verified, links
    opener = gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open('r', encoding='utf-8')
    with opener as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise ValueError('truncated journal requires manual reconciliation before resume')
            row = json.loads(line)
            if row.get('status') == 'hash_verified_before_removal':
                verified[row['path']] = row
            elif row.get('status') in ('removed', 'absent_after_verified_removal'):
                if row['path'] not in verified:
                    raise ValueError('removal journal lacks preceding checksum verification')
                removed[row['path']] = verified[row['path']]
            elif row.get('status') == 'junction_link_only_removed':
                links.add(row['path'])
    return removed, verified, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume only the same reviewed scope and existing recorded run.')
    parser.add_argument('--run-id', help='Required for an explicitly authorized execution; existing journals are never overwritten.')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'registry/gates/final_deletion_plan.json').read_text(encoding='utf-8'))
    groups = load_scope(plan)
    if not args.execute:
        print(json.dumps({'mode': 'preview_only', 'deletion_performed': False,
            'scope_status': plan['status'], 'scope_already_executed': plan.get('actual_deletion_performed', False),
            'execution_enabled': plan['deletion_execution_enabled'], 'order': plan['execution_order'],
            'groups': plan['groups'], 'logical_candidate_gib': round(plan['storage']['candidate_logical_bytes'] / 1024**3, 3)}, indent=2))
        return 0
    if plan.get('deletion_execution_enabled') is not True:
        raise PermissionError('Actual deletion is disabled: a later explicit user deletion instruction and gate update are required.')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('provide a safe unique run identifier')
    ready = json.loads((ROOT / 'registry/gates/predeletion_readiness.json').read_text(encoding='utf-8'))
    if ready['status'] != 'ready_before_actual_deletion' or ready['ledger_hashes'] != plan['ledger_hashes']:
        raise ValueError('readiness scope no longer matches current reviewed ledgers')
    directory = ROOT / 'registry/deletion_receipts' / args.run_id
    scope = {'ledger_hashes': plan['ledger_hashes'], 'groups': plan['groups']}
    if args.resume:
        if json.loads((directory / 'scope.json').read_text(encoding='utf-8')) != scope:
            raise ValueError('cannot resume a different ledger scope')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
    prior_removed, prior_verified, prior_links = prior_journal(directory / 'files.jsonl')
    removed = len(prior_removed)
    bytes_removed = sum(row['bytes'] for row in prior_removed.values())
    current = None
    with (directory / 'files.jsonl').open('a' if args.resume else 'x', encoding='utf-8') as stream:
        def journal(value):
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            for owner in plan['execution_order']:
                for row in groups[owner]:
                    current = row['path']
                    old = prior_verified.get(current)
                    if old and (old['sha256'] != row['sha256'] or old['bytes'] != row['bytes']):
                        raise ValueError('prior checksum differs from reviewed candidate')
                    if current in prior_removed:
                        if native(ROOT / current).exists():
                            raise ValueError('previously removed candidate has reappeared')
                        continue
                    if old and not native(ROOT / current).exists():
                        journal({'path': current, 'status': 'absent_after_verified_removal', 'bytes': row['bytes']})
                        removed += 1
                        bytes_removed += row['bytes']
                        continue
                    remove_verified_file(row, journal)
                    removed += 1
                    bytes_removed += row['bytes']
                    if removed % 1000 == 0:
                        print('removed', removed, 'verified candidate files', flush=True)
                journal({'group': owner, 'status': 'completed', 'free_bytes': shutil.disk_usage(ROOT).free})
            links = json.loads((ROOT / 'registry/gates/retired_compatibility_links_receipt.json').read_text(encoding='utf-8'))['links']
            for row in links:
                value = row['target']
                if not value.startswith(ARCHIVE_PREFIX + 'shared/retired_features/compatibility_links/') or '..' in value.split('/'):
                    raise ValueError('link outside reviewed boundary')
                path = ROOT / value
                if value in prior_links:
                    if native(path).exists():
                        raise ValueError('removed junction has reappeared')
                    continue
                # Check every ancestor without traversing a replacement junction.
                parent = ROOT
                for part in value.split('/')[:-1]:
                    parent = parent / part
                    if native(parent).lstat().st_file_attributes & 0x400:
                        raise ValueError('junction removal parent became a reparse point')
                if not native(path).lstat().st_file_attributes & 0x400:
                    raise ValueError('reviewed junction replaced by a regular directory')
                # rmdir on a Windows directory junction removes that link only.
                os.rmdir(native(path))
                journal({'path': value, 'status': 'junction_link_only_removed'})
            result = {'status': 'completed', 'removed_files': removed, 'removed_logical_bytes': bytes_removed,
                      'free_bytes': shutil.disk_usage(ROOT).free, 'actual_deletion_performed': True}
        except BaseException as exc:
            result = {'status': 'stopped_on_error', 'path': current, 'error': str(exc),
                      'removed_files': removed, 'removed_logical_bytes': bytes_removed, 'actual_deletion_performed': bool(removed)}
            journal(result)
            (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            raise
        (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
