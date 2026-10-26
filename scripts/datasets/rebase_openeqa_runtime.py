"""Create relocated OpenEQA runtime manifests, graph JSON and visual metadata."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from vlkg.core.assets.asset_relocation import PathPrefixRewriter
from scripts.datasets.rebase_navqa_runtime import rebase_sqlite, write_json

GRAPH = 'datasets/openeqa/graphs/openeqa-time-fact-core-v1/payload'
INDEX = 'datasets/openeqa/indexes/openeqa-v5-full-visual-indexes/payload'


def mapping():
    plan = json.loads((ROOT / 'registry/migrations/openeqa_mainline_move_plan.json').read_text(encoding='utf-8'))
    values = {row['source']: row['target'] for row in plan['entries']}
    for row in plan['entries']:
        if row['layer'] == 'graphs':
            values[row['source']] = row['target'].replace('/original/', '/runtime/')
        elif row['layer'] == 'indexes':
            for suffix in ('.manifest.json', '.metadata.sqlite'):
                name = 'openeqa_object_siglip_flat_ip' + suffix
                values[row['source'] + '/' + name] = row['target'].replace('/original/', '/runtime/') + '/' + name
    values['configs/openeqa_cg6_unified_full.yaml'] = 'configs/datasets/openeqa/graph.yaml'
    values['data/cache/embedding_cache.db'] = 'datasets/openeqa/cache/openeqa-cache-external-v1/payload/embedding_cache.sqlite'
    return values


def main():
    move = json.loads((ROOT / 'registry/migrations/openeqa_mainline_move_receipt.json').read_text(encoding='utf-8'))
    if move['status'] != 'verified':
        raise RuntimeError('OpenEQA source move is not fully verified')
    writer = PathPrefixRewriter(mapping(), ROOT)
    rows = []
    graph_stats = {}
    for shard in ('hm3d', 'scannet'):
        original = ROOT / GRAPH / 'original' / shard
        count = size = 0
        for p in sorted(original.rglob('*')):
            if not p.is_file() or p.suffix not in ('.json', '.jsonl', '.csv'):
                continue
            target = ROOT / GRAPH / 'runtime' / shard / p.relative_to(original)
            report = writer.stream(p, target, verify_existing=True)
            rows.append({'source': p.relative_to(ROOT).as_posix(), 'target': target.relative_to(ROOT).as_posix(), **report})
            count += 1; size += target.stat().st_size
            if count % 1000 == 0:
                print('runtime graph views', shard, count, flush=True)
        graph_stats[shard] = {'files': count, 'bytes': size}
        print('graph shard finished', shard, count, size, flush=True)
    index_stats = {}
    for shard in ('hm3d', 'scannet'):
        original = ROOT / INDEX / 'original' / shard
        target = ROOT / INDEX / 'runtime' / shard
        name = 'openeqa_object_siglip_flat_ip'
        stats = rebase_sqlite(original / (name + '.metadata.sqlite'), target / (name + '.metadata.sqlite'), writer)
        manifest = json.loads((original / (name + '.manifest.json')).read_text(encoding='utf-8-sig'))
        manifest = writer.rewrite_value(manifest)
        manifest['index_path'] = str((original / (name + '.faiss')).resolve())
        manifest['metadata_path'] = str((target / (name + '.metadata.sqlite')).resolve())
        manifest['runtime_path_contract'] = 'relocated_metadata_unchanged_faiss_v1'
        write_json(target / (name + '.manifest.json'), manifest)
        stats['faiss_binary_unchanged'] = True
        index_stats[shard] = stats
    resources = writer.rewrite_value(json.loads((ROOT / 'configs/openeqa_unified_execution_resources_v1.json').read_text(encoding='utf-8')))
    for row in resources['execution_shards']:
        shard = 'hm3d' if row['data_source'] == 'hm3d-v0' else 'scannet'
        row['visual_index_dir'] = INDEX + '/runtime/' + shard
    resources['effective_graph_config'] = 'configs/datasets/openeqa/graph.yaml'
    write_json(ROOT / 'configs/datasets/openeqa/execution_resources.json', resources)
    ledger = ROOT / 'registry/migrations/openeqa_runtime_view_hashes.json.gz'
    ledger.write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode('utf-8'), mtime=0))
    write_json(ROOT / 'registry/migrations/openeqa_runtime_views.json', {
        'status': 'generated_pending_behavior_validation', 'view_files': len(rows),
        'graph_stats': graph_stats, 'index_stats': index_stats,
        'path_mapping': mapping(), 'file_ledger': ledger.relative_to(ROOT).as_posix(),
        'file_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'models_called': False, 'original_assets_deleted': False})


if __name__ == '__main__':
    main()
