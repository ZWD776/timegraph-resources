"""Execute a reviewed, byte-preserving NExT-QA copy plan; never delete sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(root: Path, value: str) -> Path:
    candidate = root / value
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path escapes workspace: {value}')
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'reparse path not permitted: {parent}')
    return candidate.resolve()


def enumerate_plan(plan: dict, root: Path) -> list[tuple[Path, Path]]:
    if plan.get('dataset') != 'nextqa' or plan.get('delete_sources') is not False:
        raise ValueError('only non-destructive NExT-QA copy plans are supported')
    pairs, targets = [], set()
    for entry in plan['entries']:
        source, target = local(root, entry['source']), local(root, entry['target'])
        if not any(source.is_relative_to(root / allowed) for allowed in ('.scratch/nextqa', 'data/next-qa')):
            raise ValueError(f'unapproved source: {source}')
        if not target.is_relative_to(root / 'datasets/nextqa') or 'payload' not in target.relative_to(root / 'datasets/nextqa').parts:
            raise ValueError(f'unapproved target: {target}')
        if not source.exists():
            raise FileNotFoundError(source)
        files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
        for path in files:
            src = local(root, path.relative_to(root).as_posix())
            dst = target if source.is_file() else target / path.relative_to(source)
            dst = local(root, dst.relative_to(root).as_posix())
            if dst in targets:
                raise ValueError(f'duplicate target: {dst}')
            targets.add(dst)
            pairs.append((src, dst))
    return pairs


def copy_file(source: Path, target: Path) -> dict:
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        src_hash = digest(source)
        if digest(target) != src_hash:
            raise FileExistsError(f'destination differs; refusing to overwrite: {target}')
    else:
        temporary = target.with_name(target.name + '.copying')
        if temporary.exists():
            raise FileExistsError(f'unfinished temporary copy needs inspection: {temporary}')
        h = hashlib.sha256()
        with source.open('rb') as inp, temporary.open('xb') as out:
            last = time.monotonic()
            copied = 0
            while chunk := inp.read(4 * 1024**2):
                out.write(chunk)
                h.update(chunk)
                copied += len(chunk)
                if time.monotonic() - last > 15:
                    print(f'copying {source.name}: {copied / 1024**3:.2f} GiB', flush=True)
                    last = time.monotonic()
            out.flush()
            os.fsync(out.fileno())
        src_hash = h.hexdigest()
        if digest(temporary) != src_hash:
            raise RuntimeError(f'copy verification failed: {temporary}')
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'source changed during copy: {source}')
        # rename fails on Windows if a concurrent writer has claimed the destination.
        temporary.rename(target)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'source changed during verification: {source}')
    return {'bytes': before.st_size, 'sha256': src_hash}


def execute(plan_path: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    pairs = enumerate_plan(plan, root)
    total = sum(src.stat().st_size for src, _ in pairs)
    missing_bytes = sum(src.stat().st_size for src, dst in pairs if not dst.exists())
    if shutil.disk_usage(root).free < missing_bytes + 1024**3:
        raise RuntimeError('insufficient free space for verified copies plus 1 GiB margin')
    print(f'copy plan: {len(pairs)} files, {total / 1024**3:.3f} GiB', flush=True)
    rows = []
    last = time.monotonic()
    for i, (src, dst) in enumerate(pairs, 1):
        row = copy_file(src, dst)
        rows.append({'source': src.relative_to(root).as_posix(),
                     'target': dst.relative_to(root).as_posix(), **row})
        if i % 500 == 0 or i == len(pairs) or time.monotonic() - last > 15:
            print(f'verified {i}/{len(pairs)}', flush=True)
            last = time.monotonic()
    if plan_path.read_bytes() != plan_bytes:
        raise RuntimeError('plan changed during copy')
    report = {'contract': 'nextqa_mainline_copy_receipt_v1', 'status': 'byte_copies_verified',
              'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'files': len(rows),
              'bytes': total, 'sources_deleted': False, 'runtime_references_rewritten': False,
              'rows': rows}
    receipt = root / 'datasets/nextqa/reports/nextqa-reports-v1/payload/copy_receipt.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(report, separators=(',', ':')) + '\n').encode('utf-8')
    if receipt.exists():
        if receipt.read_bytes() != content:
            raise FileExistsError('previous copy receipt differs')
    else:
        receipt.write_bytes(content)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    execute(parser.parse_args().plan)
