"""Repair unvalidated v3 Windows references while preserving affected bytes."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from vlkg.core.assets.asset_relocation import PathPrefixRewriter


def main():
    state = ROOT / 'registry/migrations/nextqa_layer_ownership_activation.json'
    report = json.loads(state.read_text(encoding='utf-8'))
    if report['status'] != 'activated_pending_behavior_validation':
        raise RuntimeError('only unvalidated runtime derivations may be repaired')
    writer = PathPrefixRewriter(report['mapping'], ROOT)
    archive = ROOT / 'archive/legacy_20260928/migration/pending_nextqa_v3_paths'
    repairs = []
    for base in ('graphs/nextqa-cg6-human-v2-time-fact-core-v1', 'indexes/nextqa-visual-index-v2-strict-pairs-full570'):
        original = ROOT / 'datasets/nextqa' / base / 'payload/runtime_paths_v3'
        for count, p in enumerate(sorted(original.rglob('*.json')), 1):
            temporary = p.with_name(p.name + '.path-repair.tmp')
            outcome = writer.stream(p, temporary)
            if outcome['replacements']:
                snapshot = archive / p.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                if snapshot.exists():
                    raise FileExistsError('repair snapshot already exists')
                shutil.copyfile(p, snapshot)
                # Only a newly derived, not-yet-validated runtime file is replaced.
                # Its previous bytes are preserved in the archive first.
                os.replace(temporary, p)
                repairs.append({'path': p.relative_to(ROOT).as_posix(), 'snapshot': snapshot.relative_to(ROOT).as_posix(), **outcome})
            else:
                # This file was created solely as a temporary comparison output.
                temporary.unlink()
            if count % 500 == 0:
                print('pending references checked', base.split('/')[0], count, flush=True)
    # Preserve the incomplete ledger, then update only the repaired target hashes.
    ledger = ROOT / report['file_ledger']
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ledger, archive / ledger.name)
    rows = json.loads(gzip.decompress(ledger.read_bytes()))
    by_path = {r['path']: r for r in repairs}
    for row in rows:
        key = row.get('target') or row.get('target_path')
        if key in by_path:
            row['target_sha256'] = by_path[key]['target_sha256']
            row['target_bytes'] = by_path[key]['target_bytes']
            row['pending_path_repair'] = True
    ledger.write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode('utf-8'), mtime=0))
    report['file_ledger_sha256'] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    report['pending_windows_path_repair'] = 'registry/migrations/nextqa_pending_path_repair.json'
    state.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    result = {'status': 'repaired_pending_validation', 'files_repaired': len(repairs), 'repairs': repairs,
              'source_graph_data_or_vectors_deleted': False, 'only_reviewed_path_prefixes_changed': True}
    (ROOT / 'registry/migrations/nextqa_pending_path_repair.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('repaired files', len(repairs), flush=True)


if __name__ == '__main__':
    main()
