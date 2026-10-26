"""Split a shared cache into dataset-owned keys without changing embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from vlkg.core.assets.asset_relocation import PathPrefixRewriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('navqa', 'openeqa'), required=True)
    args = parser.parse_args()
    if args.dataset == 'navqa':
        from scripts.datasets.rebase_navqa_runtime import mapping
        asset = 'navqa-cache-external-v1'
    else:
        from scripts.datasets.rebase_openeqa_runtime import mapping
        asset = 'openeqa-cache-external-v1'
    writer = PathPrefixRewriter(mapping(), ROOT)
    source = ROOT / 'data/cache/embedding_cache.db'
    target = ROOT / f'datasets/{args.dataset}/cache/{asset}/payload/embedding_cache.sqlite'
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect('file:' + str(source) + '?mode=ro', uri=True)
    new = sqlite3.connect(target)
    copied = duplicates = total = 0
    before_signature = after_signature = 0
    unmapped = []
    try:
        new.execute(old.execute("SELECT sql FROM sqlite_master WHERE name='embedding_cache'").fetchone()[0])
        cursor = old.execute('SELECT image_path,bbox_key,embedding,dim,updated_at FROM embedding_cache WHERE instr(image_path,?)>0', (args.dataset,))
        while batch := cursor.fetchmany(1000):
            for path, bbox, blob, dim, timestamp in batch:
                relocated = writer.rewrite_text(path)
                if relocated == path:
                    # Cache rows from nonselected experiments stay in the archive.
                    if len(unmapped) < 30:
                        unmapped.append(path)
                    continue
                total += 1
                signature = int.from_bytes(hashlib.sha256(bbox.encode('utf-8') + blob + str(dim).encode('ascii')).digest(), 'big')
                previous = new.execute('SELECT embedding,dim FROM embedding_cache WHERE image_path=? AND bbox_key=?', (relocated, bbox)).fetchone()
                if previous:
                    if previous != (blob, dim):
                        raise ValueError('path rebasing would merge different embeddings for one image/bbox')
                    duplicates += 1
                    continue
                new.execute('INSERT INTO embedding_cache VALUES (?,?,?,?,?)', (relocated, bbox, blob, dim, timestamp))
                before_signature = (before_signature + signature) % (1 << 256)
                after_signature = (after_signature + signature) % (1 << 256)
                copied += 1
            new.commit()
            if total and total % 100000 == 0:
                print(args.dataset, 'cache keys checked', total, flush=True)
        integrity = new.execute('PRAGMA quick_check').fetchone()[0]
        if integrity != 'ok':
            raise RuntimeError('dataset embedding cache integrity check failed')
        new.commit()
        # Validate the stored values independently of insertion bookkeeping.
        stored_signature = 0
        for bbox, blob, dim in new.execute('SELECT bbox_key,embedding,dim FROM embedding_cache'):
            signature = int.from_bytes(hashlib.sha256(bbox.encode('utf-8') + blob + str(dim).encode('ascii')).digest(), 'big')
            stored_signature = (stored_signature + signature) % (1 << 256)
        if stored_signature != before_signature:
            raise RuntimeError('stored embedding/bbox/dimension multiset changed')
    finally:
        old.close(); new.close()
    report = {
        'status': 'verified', 'dataset': args.dataset, 'source': source.relative_to(ROOT).as_posix(),
        'target': target.relative_to(ROOT).as_posix(), 'source_keys_considered': total,
        'copied_keys': copied, 'identical_aliases_collapsed': duplicates,
        'unselected_unmapped_examples': unmapped,
        'embedding_bbox_dimension_signature': f'{before_signature:064x}',
        'stored_signature': f'{stored_signature:064x}', 'bytes': target.stat().st_size,
        'integrity': integrity, 'models_called': False, 'source_deleted': False,
    }
    out = ROOT / f'registry/migrations/{args.dataset}_embedding_cache.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
