"""Execute the separately reviewed AAAI-only supplement, preserving ICLR references."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datasets.execute_reviewed_deletion import remove_verified_file, native

SCOPE = ROOT / 'registry/gates/aaai_supplemental_deletion_scope_20261002.json'
PREFIXES = ('archive/legacy_20260928/shared/papers_and_release/aaai27_paper_draft/',
            'archive/legacy_20260928/shared/assets/root_materials/AuthorKit27/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    scope = json.loads(SCOPE.read_text(encoding='utf-8'))
    blob = (ROOT / scope['base_ledger']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != scope['base_ledger_sha256']:
        raise ValueError('original preservation ledger changed')
    original = {r['path']: r for r in json.loads(gzip.decompress(blob))}
    rows = scope['delete_candidates']
    preserved = {r['path'] for r in scope['preserved_shared_materials']}
    if (len({r['path'] for r in rows}) != scope['delete_files']
            or len(rows) != scope['delete_files']
            or sum(r['bytes'] for r in rows) != scope['delete_bytes']):
        raise ValueError('supplemental scope count/size differs')
    for row in rows:
        old = original[row['path']]
        if (not row['path'].startswith(PREFIXES) or row['path'] in preserved
                or (row['bytes'], row['sha256']) != (old['bytes'], old['sha256'])):
            raise ValueError('candidate escapes reviewed AAAI supplement')
    for path in preserved:
        if hashlib.sha256(native(ROOT / path).read_bytes()).hexdigest() != original[path]['sha256']:
            raise ValueError('shared ICLR reference missing or changed')
    if not args.execute:
        print(json.dumps({'mode': 'preview', 'files': len(rows), 'bytes': scope['delete_bytes'],
                          'shared_materials_preserved': len(preserved)}))
        return 0
    plan = json.loads((ROOT / 'registry/gates/final_deletion_plan.json').read_text(encoding='utf-8'))
    authorization = json.loads((ROOT / plan['authorization']).read_text(encoding='utf-8'))
    primary = json.loads((ROOT / 'registry/deletion_receipts' / authorization['run_id'] / 'summary.json').read_text(encoding='utf-8'))
    if (not plan['deletion_execution_enabled'] or primary['status'] != 'completed'
            or scope['status'] != 'authorized_supplemental_scope'):
        raise PermissionError('primary completion and explicit supplemental authorization required')
    directory = ROOT / scope['receipt_directory']
    if directory != ROOT / 'registry/deletion_receipts/aaai_cleanup_20261002':
        raise ValueError('unexpected receipt directory')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'scope.json').write_bytes(SCOPE.read_bytes())
    count = size = 0
    current = None
    with (directory / 'files.jsonl').open('x', encoding='utf-8') as stream:
        def journal(value):
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            for row in rows:
                current = row['path']
                remove_verified_file(row, journal)
                count += 1
                size += row['bytes']
            result = {'status': 'completed', 'removed_files': count, 'removed_logical_bytes': size,
                      'actual_deletion_performed': True, 'shared_materials_preserved': len(preserved),
                      'free_bytes': shutil.disk_usage(ROOT).free}
        except BaseException as exc:
            result = {'status': 'stopped_on_error', 'path': current, 'error': str(exc),
                      'removed_files': count, 'removed_logical_bytes': size,
                      'actual_deletion_performed': bool(count)}
            journal(result)
            (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            raise
        (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
