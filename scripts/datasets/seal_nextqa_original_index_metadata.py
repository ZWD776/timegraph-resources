"""Seal unused original metadata while preserving the active FAISS shard files."""
from __future__ import annotations

import gzip
import hashlib
import json
import os

from scripts.datasets.migrate_asset_storage import ROOT, safe_path, sha256, write_json, verify_rows


def main():
    source_name = 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/videos'
    target_name = 'archive/legacy_20260928/nextqa/original_mainline/index_metadata/videos'
    source, target = safe_path(source_name), safe_path(target_name)
    receipt_path = ROOT / 'registry/gates/nextqa_original_index_metadata_move_receipt.json'
    if target.exists() or receipt_path.exists():
        raise FileExistsError('inspect the existing receipt; this tool never overwrites a move')
    manifest = ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/runtime_paths_v3/index_manifest.json'
    active = json.loads(manifest.read_text(encoding='utf-8'))
    references = []
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'metadata_path' and isinstance(child, str):
                    references.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(active)
    if len(references) != 3420 or any('runtime_paths_v3' not in path for path in references):
        raise ValueError('selected manifest must reference all 3420 current metadata shards')
    files = sorted(source.rglob('*.metadata.json'))
    if len(files) != 3420:
        raise ValueError(f'expected 3420 original metadata shards, found {len(files)}')
    binary_stats = {p.relative_to(source).as_posix(): p.stat().st_size for p in source.rglob('*.faiss')}
    rows = [{'relative': p.relative_to(source).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha256(p)} for p in files]
    ledger_path = receipt_path.with_name('nextqa_original_index_metadata_move_receipt.000.files.json.gz')
    blob = gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0)
    ledger_path.write_bytes(blob)
    entry = {'source': source_name, 'target': target_name, 'dataset': 'nextqa',
        'status': 'hashed_before_move', 'files': len(rows), 'bytes': sum(row['bytes'] for row in rows),
        'subset_move': True, 'source_retains_active_faiss_binaries': True,
        'file_ledger': ledger_path.relative_to(ROOT).as_posix(),
        'file_ledger_sha256': hashlib.sha256(blob).hexdigest()}
    receipt = {'contract': 'asset_storage_move_receipt_v1', 'status': 'in_progress',
        'entries': [entry], 'deletion_performed': False, 'active_metadata_references': len(references),
        'retained_faiss_files': len(binary_stats)}
    write_json(receipt_path, receipt)
    for row in rows:
        old, new = safe_path(source_name + '/' + row['relative']), safe_path(target_name + '/' + row['relative'])
        new.parent.mkdir(parents=True, exist_ok=True)
        os.rename(old, new)
    verify_rows(target, rows)
    if binary_stats != {p.relative_to(source).as_posix(): p.stat().st_size for p in source.rglob('*.faiss')}:
        raise ValueError('active FAISS inventory changed')
    entry['status'] = receipt['status'] = 'verified'
    write_json(receipt_path, receipt)
    print(json.dumps({'status': 'verified', 'files': len(rows), 'bytes': entry['bytes'],
                      'retained_faiss_files': len(binary_stats)}))


if __name__ == '__main__':
    main()
