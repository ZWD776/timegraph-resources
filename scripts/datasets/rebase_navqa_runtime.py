"""Derive CG6 runtime views from hash-verified, relocated NAVQA originals."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vlkg.core.assets.asset_relocation import PathPrefixRewriter

SOURCE = 'datasets/navqa/source/navqa-coda-every10-v1/payload'
GRAPH = 'datasets/navqa/graphs/navqa-cg6-time-fact-core-v1/payload'
INDEX = 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload'
CACHE = 'datasets/navqa/cache/navqa-cache-external-v1/payload'
RETRIEVAL = 'datasets/navqa/retrieval/navqa-cg6-retrieval-v1/payload'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def mapping():
    plan = json.loads((ROOT / 'registry/migrations/navqa_mainline_move_plan.json').read_text(encoding='utf-8'))
    values = {row['source']: row['target'] for row in plan['entries']}
    values['.scratch/navqa_cg6_local_tracklet/temporal'] = GRAPH + '/runtime/temporal'
    values['.scratch/navqa_cg6_local_tracklet/time_fact_core'] = GRAPH + '/runtime/time_fact_core'
    values['data/cache/embedding_cache.db'] = CACHE + '/embedding_cache.sqlite'
    values['.scratch/navqa_evidence_layer_index/evidence_layer.sqlite'] = INDEX + '/runtime/evidence_layer.sqlite'
    for seq in ('0', '3', '4', '6', '16', '21', '22'):
        base = f'seq{seq}/cam0/every10'
        values[f'.scratch/navqa_seq{seq}_cam0_every10'] = SOURCE + f'/navqa/{base}/images'
        values[f'.scratch/navqa/{base}/hsgg_local_paths'] = SOURCE + f'/runtime_inputs/{base}/hsgg_local_paths'
    for suffix in ('.metadata.jsonl', '.metadata.sqlite', '.manifest.json'):
        name = 'object_siglip_flat_ip' + suffix
        values['.scratch/navqa_graph_experiments/visual_faiss_grounding/' + name] = INDEX + '/runtime/' + name
    return values


def rebase_sqlite(source, target, rewriter, *, cg6_only=False):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect('file:' + str(source) + '?mode=ro', uri=True)
    new = sqlite3.connect(target)
    old.row_factory = sqlite3.Row
    counts, changed = {}, {}
    try:
        tables = old.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for table, ddl in tables:
            new.execute(ddl)
            columns = [r[1] for r in old.execute(f'PRAGMA table_info("{table}")')]
            where = " WHERE cg_id='CG6-clean'" if cg6_only and table == 'cg_graph_refs' else ''
            cursor = old.execute(f'SELECT * FROM "{table}"{where}')
            inserted = edits = 0
            while batch := cursor.fetchmany(1000):
                records = []
                for row in batch:
                    values = list(row)
                    for i, value in enumerate(values):
                        if not isinstance(value, str):
                            continue
                        if cg6_only and table == 'metadata' and row['key'] == 'build_config' and columns[i] == 'value_json':
                            config = json.loads(value)
                            config['sequence_manifest'] = 'configs/datasets/navqa/sequences.json'
                            config['cg_manifests'] = ['configs/datasets/navqa/sequences.json']
                            values[i] = json.dumps(config, ensure_ascii=False, separators=(',', ':'))
                        elif cg6_only and table == 'sequences' and columns[i] == 'resource_json':
                            resource = json.loads(value)
                            resource['cg_graphs'] = [g for g in resource.get('cg_graphs', []) if g.get('cg_id') == 'CG6-clean']
                            values[i] = json.dumps(rewriter.rewrite_value(resource), ensure_ascii=False, separators=(',', ':'))
                        else:
                            values[i] = rewriter.rewrite_text(value)
                        edits += int(values[i] != value)
                    records.append(values)
                new.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in columns)})', records)
                inserted += len(records)
            counts[table], changed[table] = inserted, edits
            print('sqlite', table, inserted, 'rows', flush=True)
        for ddl, in old.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"):
            new.execute(ddl)
        new.commit()
        if new.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('derived SQLite failed integrity check')
    finally:
        old.close(); new.close()
    return {'row_counts': counts, 'changed_text_cells': changed, 'integrity': 'ok',
            'nonpath_cells_and_blob_values_preserved': True, 'cg6_only': cg6_only}


def main():
    receipt = json.loads((ROOT / 'registry/migrations/navqa_mainline_move_receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'verified':
        raise RuntimeError('NAVQA move is not verified')
    rewriter = PathPrefixRewriter(mapping(), ROOT)
    rows = []
    for part in ('temporal', 'time_fact_core'):
        original = ROOT / GRAPH / 'original' / part
        for p in sorted(original.rglob('*')):
            if p.is_file():
                target = ROOT / GRAPH / 'runtime' / part / p.relative_to(original)
                report = rewriter.stream(p, target, verify_existing=True)
                rows.append({'source': p.relative_to(ROOT).as_posix(), 'target': target.relative_to(ROOT).as_posix(), **report})
                print('graph view', target.name, target.stat().st_size, flush=True)
    for seq in ('0', '3', '4', '6', '16', '21', '22'):
        base = f'seq{seq}/cam0/every10/hsgg_local_paths'
        original = ROOT / SOURCE / 'navqa' / base
        for p in sorted(original.rglob('*.json')):
            target = ROOT / SOURCE / 'runtime_inputs' / base / p.relative_to(original)
            report = rewriter.stream(p, target, verify_existing=True)
            rows.append({'source': p.relative_to(ROOT).as_posix(), 'target': target.relative_to(ROOT).as_posix(), **report})
        print('HSGG runtime inputs', seq, flush=True)
    metadata = ROOT / INDEX / 'visual_faiss/object_siglip_flat_ip.metadata.jsonl'
    rows.append({'source': metadata.relative_to(ROOT).as_posix(), 'target': INDEX + '/runtime/object_siglip_flat_ip.metadata.jsonl',
                 **rewriter.stream(metadata, ROOT / INDEX / 'runtime/object_siglip_flat_ip.metadata.jsonl', verify_existing=True)})
    manifest = json.loads((ROOT / INDEX / 'visual_faiss/object_siglip_flat_ip.manifest.json').read_text(encoding='utf-8-sig'))
    manifest = rewriter.rewrite_value(manifest)
    manifest['sequence_manifest'] = 'configs/datasets/navqa/sequences.json'
    manifest['metadata_sqlite_path'] = INDEX + '/runtime/object_siglip_flat_ip.metadata.sqlite'
    write_json(ROOT / INDEX / 'runtime/object_siglip_flat_ip.manifest.json', manifest)
    sqlite_report = rebase_sqlite(ROOT / INDEX / 'visual_faiss/object_siglip_flat_ip.metadata.sqlite',
                                  ROOT / INDEX / 'runtime/object_siglip_flat_ip.metadata.sqlite', rewriter)
    evidence_report = rebase_sqlite(ROOT / '.scratch/navqa_evidence_layer_index/evidence_layer.sqlite',
                                    ROOT / INDEX / 'runtime/evidence_layer.sqlite', rewriter, cg6_only=True)
    for seq in ('0', '3', '4', '6', '16', '21', '22'):
        original = ROOT / RETRIEVAL / f'seq{seq}_cam0_every10/projection_cache.json.gz'
        target = original.parent / 'runtime_projection_cache.json.gz'
        if target.exists():
            raise FileExistsError(target)
        with gzip.open(original, 'rt', encoding='utf-8') as f:
            before = json.load(f)
        after = rewriter.rewrite_value(before)
        # Graph structure, labels and temporal signatures are immutable.
        if before.get('fingerprint') != after.get('fingerprint'):
            raise RuntimeError('projection fingerprint changed during path rewrite')
        data = json.dumps(after, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        target.write_bytes(gzip.compress(data, mtime=0))
        rows.append({'source': original.relative_to(ROOT).as_posix(), 'target': target.relative_to(ROOT).as_posix(),
                     'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                     'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'projection_fingerprint_unchanged': True})
        print('projection cache', seq, flush=True)
    ledger = ROOT / 'registry/migrations/navqa_runtime_view_hashes.json.gz'
    ledger.write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode('utf-8'), mtime=0))
    write_json(ROOT / 'registry/migrations/navqa_runtime_views.json', {
        'status': 'generated_pending_behavior_validation', 'view_files': len(rows),
        'path_mapping': mapping(), 'file_ledger': ledger.relative_to(ROOT).as_posix(),
        'file_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'visual_metadata': sqlite_report, 'evidence_layer': evidence_report,
        'models_called': False, 'original_assets_deleted': False})


if __name__ == '__main__':
    main()
