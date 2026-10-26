"""Remove empty archive containers after verified file deletion; never recurse-delete."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datasets.execute_reviewed_deletion import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'registry/gates/final_deletion_plan.json').read_text(encoding='utf-8'))
    auth = json.loads((ROOT / plan['authorization']).read_text(encoding='utf-8'))
    receipts = ROOT / 'registry/deletion_receipts' / auth['run_id']
    if json.loads((receipts / 'summary.json').read_text(encoding='utf-8'))['status'] != 'completed':
        raise ValueError('primary file deletion must be complete first')
    archive = ROOT / 'archive/legacy_20260928'
    if archive.resolve() != archive or not archive.is_relative_to(ROOT):
        raise ValueError('archive boundary changed')
    paper = json.loads((ROOT / 'registry/papers/paper_preservation_inventory.json').read_text(encoding='utf-8'))
    protected = [ROOT / v['path'] for k, v in paper['inventory'].items() if k != 'aaai27_paper_draft']
    protected += [ROOT / p for p in paper['additional_keep_roots']]
    protected.append(archive / 'migration/final_deletion_catalog')
    if args.execute and not plan['deletion_execution_enabled']:
        raise PermissionError('cleanup execution gate is disabled')
    log = receipts / ('empty_directories.jsonl' if args.execute else 'empty_directories_preview.jsonl')
    count = 0
    def walk_error(exc):
        raise exc
    directories = []
    for directory, children, _ in os.walk(native(archive), topdown=True, followlinks=False, onerror=walk_error):
        path = ROOT / Path(directory).relative_to(native(ROOT))
        if any(path.is_relative_to(p) for p in protected):
            children.clear()
            continue
        for name in children:
            if native(path / name).lstat().st_file_attributes & 0x400:
                raise ValueError('unexpected reparse container after junction cleanup')
        directories.append(path)
    with log.open('x', encoding='utf-8') as stream:
        for path in reversed(directories):
            if path == archive or any(path.is_relative_to(p) for p in protected):
                continue
            if not path.resolve().is_relative_to(archive):
                raise ValueError('empty container escapes archive')
            parent = ROOT
            for part in path.relative_to(ROOT).parts:
                parent /= part
                if native(parent).lstat().st_file_attributes & 0x400:
                    raise ValueError('empty container traverses a reparse point')
            with os.scandir(native(path)) as entries:
                if next(entries, None) is not None:
                    continue
            if args.execute:
                # rmdir fails if anything appeared; it cannot remove a nonempty directory.
                os.rmdir(native(path))
            stream.write(json.dumps({'path': path.relative_to(ROOT).as_posix(),
                                     'status': 'empty_directory_removed' if args.execute else 'empty_directory_candidate'}) + '\n')
            count += 1
            if count % 1000 == 0:
                stream.flush()
                print('empty containers processed:', count, flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    result = {'status': 'completed', 'empty_directories': count, 'actual_empty_directory_removal': args.execute,
              'recursive_deletion_used': False, 'protected_paper_directories_excluded': True}
    (receipts / 'empty_directories_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
