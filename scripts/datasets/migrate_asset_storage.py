"""Perform reviewed, reversible workspace-local asset moves with file hashes.

This tool never deletes assets. A receipt is saved before and after each move.
Use --verify to recheck a completed receipt without changing storage.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def native_path(path: Path) -> Path:
    resolved = str(path.absolute())
    if os.name == 'nt' and not resolved.startswith('\\\\?\\'):
        return Path('\\\\?\\' + resolved)
    return path


def safe_path(value: str, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('use nonempty workspace-relative POSIX paths')
    p = Path(value)
    if p.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'unsafe relative path: {value}')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes workspace: {value}')
    # Resolve alone can hide a junction which points back into the workspace.
    current = root
    for part in p.parts:
        current = current / part
        if current.exists() and is_reparse(current):
            raise ValueError(f'reparse point in asset path: {current}')
    return resolved


def is_reparse(path: Path) -> bool:
    path = native_path(path)
    info = path.lstat()
    return bool(getattr(info, 'st_file_attributes', 0) & 0x400) or path.is_symlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with native_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path):
    path = native_path(path)
    if is_reparse(path):
        raise ValueError(f'reparse asset: {path}')
    if path.is_file():
        return [{'relative': '', 'bytes': path.stat().st_size, 'sha256': sha256(path)}]
    rows = []
    last_notice = time.monotonic()
    def fail_on_walk_error(error):
        # An inaccessible subtree must never produce a successful partial ledger.
        raise error

    for directory, dirs, files in os.walk(path, onerror=fail_on_walk_error):
        for name in dirs:
            if is_reparse(Path(directory, name)):
                raise ValueError(f'reparse child directory: {directory}/{name}')
        for name in files:
            file = Path(directory, name)
            if is_reparse(file):
                raise ValueError(f'reparse child file: {file}')
            rows.append({'relative': file.relative_to(path).as_posix(),
                         'bytes': file.stat().st_size, 'sha256': sha256(file)})
            if time.monotonic() - last_notice > 20:
                print(f'hashing {path.name}: {len(rows)} files', flush=True)
                last_notice = time.monotonic()
    return sorted(rows, key=lambda row: row['relative'])


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def verify_rows(path: Path, rows):
    actual = inventory(path)
    if actual != rows:
        raise RuntimeError(f'file bytes or inventory changed: {path}')


def reconcile_split_move(source: Path, target: Path, rows):
    """Recover a native Move-Item partial move without dropping any source copy."""
    for row in rows:
        old, new = source / row['relative'], target / row['relative']
        old_native, new_native = native_path(old), native_path(new)
        if new_native.is_file():
            if sha256(new) != row['sha256']:
                raise ValueError(f'partial archive target changed: {new}')
            if old_native.is_file():
                if sha256(old) != row['sha256']:
                    raise ValueError(f'partial source copy changed: {old}')
                # Preserve an unexpected duplicate rather than removing it.
                duplicate = target.parent / (target.name + '_split_duplicates') / row['relative']
                native_path(duplicate.parent).mkdir(parents=True, exist_ok=True)
                if native_path(duplicate).exists():
                    raise FileExistsError(duplicate)
                os.rename(old_native, native_path(duplicate))
        elif old_native.is_file():
            if sha256(old) != row['sha256']:
                raise ValueError(f'partial source changed: {old}')
            native_path(new.parent).mkdir(parents=True, exist_ok=True)
            os.rename(old_native, new_native)
        else:
            raise FileNotFoundError(f'file absent from both sides of partial move: {row["relative"]}')
    verify_rows(target, rows)
    # Preserve the now-empty source directory tree as metadata; no asset delete.
    if inventory(source):
        raise ValueError('unrecorded files remain in partial source')
    empty_tree = target.parent / (target.name + '_empty_source_tree')
    if empty_tree.exists():
        raise FileExistsError(empty_tree)
    os.rename(native_path(source), native_path(empty_tree))


def migrate(plan_path: Path, *, resume=False):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('action') != 'move' or plan.get('deletion_allowed') is not False:
        raise ValueError('only explicit non-deleting move plans are accepted')
    receipt_path = safe_path(plan['receipt'])
    if receipt_path.exists() and not resume:
        raise FileExistsError('receipt exists; verify or review it before another move')
    previous = json.loads(receipt_path.read_text(encoding='utf-8')) if resume else None
    if previous and previous['plan_sha256'] != sha256(plan_path):
        raise ValueError('resume plan differs from the original reviewed plan')
    resolved = []
    for row in plan['entries']:
        if row['source'].split('/')[0] in {'.git', '.venv', '.codex', '.agents', '.aws'}:
            raise ValueError('protected environment is outside the asset migration scope')
        source, target = safe_path(row['source']), safe_path(row['target'])
        index = len(resolved)
        prior = previous['entries'][index] if previous and index < len(previous['entries']) else None
        if prior and (prior['source'] != row['source'] or prior['target'] != row['target']):
            raise ValueError('resume receipt does not match plan entry')
        if not prior and (not source.exists() or target.exists()):
            raise ValueError(f'missing source or existing target: {row}')
        if not row['target'].startswith(('datasets/', 'archive/legacy_20260928/')):
            raise ValueError('target must be a dataset asset or the declared archive root')
        if source == target or source.is_relative_to(target) or target.is_relative_to(source):
            raise ValueError('overlapping move source/target')
        resolved.append((row, source, target))
    for i, (_, source, target) in enumerate(resolved):
        for _, other_source, other_target in resolved[i+1:]:
            paths = (source, target)
            if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                   for a in paths for b in (other_source, other_target)):
                raise ValueError('plan contains overlapping entries')
    receipt = previous or {'contract': 'asset_storage_move_receipt_v1', 'plan': plan_path.relative_to(ROOT).as_posix(),
               'plan_sha256': sha256(plan_path), 'status': 'in_progress', 'entries': [],
               'deletion_performed': False}
    write_json(receipt_path, receipt)
    for number, (row, source, target) in enumerate(resolved):
        if number < len(receipt['entries']):
            item = receipt['entries'][number]
            if item['status'] == 'verified':
                continue
            ledger = safe_path(item['file_ledger']).read_bytes()
            if hashlib.sha256(ledger).hexdigest() != item['file_ledger_sha256']:
                raise ValueError('resume file ledger changed')
            files = json.loads(gzip.decompress(ledger))
            if source.exists() and not target.exists():
                verify_rows(source, files)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)
            elif source.exists() and target.exists():
                reconcile_split_move(source, target, files)
                item['partial_native_move_recovered'] = True
            elif source.exists() or not target.exists():
                raise ValueError('ambiguous source/target state during resume')
            verify_rows(target, files)
            item['status'] = 'verified'
            write_json(receipt_path, receipt)
            print('resumed and verified', row['source'], flush=True)
            continue
        print(f'preflight {number+1}/{len(resolved)}: {row["source"]}', flush=True)
        files = inventory(source)
        ledger_path = receipt_path.parent / (receipt_path.stem + f'.{number:03d}.files.json.gz')
        ledger = gzip.compress(json.dumps(files, separators=(',', ':')).encode('utf-8'), mtime=0)
        if ledger_path.exists():
            raise FileExistsError(ledger_path)
        ledger_path.write_bytes(ledger)
        item = {**row, 'status': 'hashed_before_move', 'files': len(files),
                'bytes': sum(r['bytes'] for r in files),
                'file_ledger': ledger_path.relative_to(ROOT).as_posix(),
                'file_ledger_sha256': hashlib.sha256(ledger).hexdigest()}
        receipt['entries'].append(item)
        write_json(receipt_path, receipt)
        target.parent.mkdir(parents=True, exist_ok=True)
        # All resolved paths have been checked to stay inside this workspace.
        # Same-volume rename preserves bytes and is reversible without copying.
        os.rename(source, target)
        item['status'] = 'moved_pending_verification'
        write_json(receipt_path, receipt)
        verify_rows(target, files)
        item['status'] = 'verified'
        write_json(receipt_path, receipt)
        print(f'verified {item["files"]} files, {item["bytes"]} bytes', flush=True)
    receipt['status'] = 'verified'
    write_json(receipt_path, receipt)


def verify_receipt(path: Path):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt['status'] != 'verified':
        raise ValueError('move receipt is incomplete; inspect source/target before recovery')
    for row in receipt['entries']:
        blob = safe_path(row['file_ledger']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['file_ledger_sha256']:
            raise ValueError('file ledger fingerprint changed')
        verify_rows(safe_path(row['target']), json.loads(gzip.decompress(blob)))
    print('receipt verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', type=Path)
    modes.add_argument('--verify', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume the same plan from its verified file ledgers.')
    args = parser.parse_args()
    if args.plan:
        migrate(safe_path(args.plan.as_posix()), resume=args.resume)
    else:
        verify_receipt(safe_path(args.verify.as_posix()))


if __name__ == '__main__':
    main()
