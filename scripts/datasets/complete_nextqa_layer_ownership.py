"""Separate source/cache ownership and derive the final NExT-QA runtime view."""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.datasets.rebase_nextqa_runtime import Relocator, derive_json
from scripts.datasets.rebase_nextqa_embedding_cache import migrate as migrate_cache

GRAPH = 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload'
INDEX = 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload'
SOURCE = 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload'
CACHE = 'datasets/nextqa/cache/nextqa-cache-external-v1/payload'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    move = read('registry/migrations/nextqa_layer_ownership_move_receipt.json')
    if move['status'] != 'verified':
        raise RuntimeError('layer ownership move must be verified first')
    mapping = {
        GRAPH + '/runtime_paths_v2/cache/siglip_object_crop.sqlite': CACHE + '/embedding_cache.sqlite',
        GRAPH + '/nextqa_cg6_human_v2_full570_20260812a/normalized_inputs': SOURCE + '/normalized_inputs',
        GRAPH + '/runtime_paths_v2': GRAPH + '/runtime_paths_v3',
        INDEX + '/runtime_paths_v2': INDEX + '/runtime_paths_v3',
    }
    mapper = Relocator(mapping, ROOT)
    rows = []
    for base in (GRAPH, INDEX):
        original = ROOT / base / 'runtime_paths_v2'
        for number, p in enumerate(sorted(original.rglob('*.json')), 1):
            target = ROOT / base / 'runtime_paths_v3' / p.relative_to(original)
            rows.append(derive_json(p, target, mapper))
            if number % 500 == 0:
                print('nextqa layer views', 'graph' if base == GRAPH else 'index', number, flush=True)
    graph_path = GRAPH + '/runtime_paths_v3/graph_manifest.json'
    index_path = INDEX + '/runtime_paths_v3/index_manifest.json'
    write(graph_path, read(GRAPH + '/runtime_paths_v3/nextqa_graph_build_manifest.json'))
    write(index_path, read(INDEX + '/runtime_paths_v3/visual_index_manifest.json'))
    source_path = SOURCE + '/source_manifest.json'
    write(source_path, read(SOURCE + '/runtime/nextqa_val_manifest_v1.json'))
    cache_source = ROOT / CACHE / 'original_v2/siglip_object_crop.sqlite'
    cache_target = ROOT / CACHE / 'embedding_cache.sqlite'
    cache_report = migrate_cache(cache_source, cache_target, Relocator({
        GRAPH + '/nextqa_cg6_human_v2_full570_20260812a/normalized_inputs': SOURCE + '/normalized_inputs'}, ROOT))
    cache_report.update({'source': cache_source.relative_to(ROOT).as_posix(),
                         'target': cache_target.relative_to(ROOT).as_posix()})
    write('registry/migrations/nextqa_owned_cache_rebase.json', cache_report)
    missing = sorted(value for value in mapper.references if not Path(value).exists())
    if missing:
        raise FileNotFoundError(f'{len(missing)} relocated resource references missing: {missing[:3]}')
    ledger = ROOT / 'registry/migrations/nextqa_layer_view_hashes.json.gz'
    ledger.write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode('utf-8'), mtime=0))
    report = {'status': 'activated_pending_behavior_validation', 'derived_json_files': len(rows),
              'graph_manifest': graph_path, 'index_manifest': index_path,
              'source_manifest': source_path, 'cache': cache_target.relative_to(ROOT).as_posix(),
              'mapping': mapping, 'changed_fields': dict(mapper.changed_fields),
              'missing_targets': missing, 'file_ledger': ledger.relative_to(ROOT).as_posix(),
              'file_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
              'original_vectors_unchanged': True, 'old_run_contracts_modified': False,
              'deletion_performed': False}
    write('registry/migrations/nextqa_layer_ownership_activation.json', report)
    manifest_path = ROOT / 'registry/datasets/nextqa.yaml'
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    for layer, named in (
        ('graphs', {'manifest': graph_path}), ('indexes', {'manifest': index_path}),
        ('cache', {'sqlite': report['cache'], 'embeddings': report['cache']}),
    ):
        p = next((ROOT / f'datasets/nextqa/{layer}').glob('*/asset_pointer.json'))
        value = json.loads(p.read_text(encoding='utf-8'))
        value['previous_paths'] = value['current_paths']
        value['current_paths'] = list(dict.fromkeys(named.values()))
        value['entrypoints'] = named
        value['dependency_paths'] = []
        value['status'] = 'layer_owned_runtime_view_pending_behavior_validation'
        write(p.relative_to(ROOT).as_posix(), value)
        manifest['current_paths']['graph' if layer == 'graphs' else layer] = value['current_paths']
        manifest['entrypoints'][layer] = named
    p = next((ROOT / 'datasets/nextqa/source').glob('*/asset_pointer.json'))
    value = json.loads(p.read_text(encoding='utf-8'))
    value['entrypoints']['manifest'] = source_path
    value['dependency_paths'] = [path for path in value.get('dependency_paths', []) if '/normalized_inputs' not in path]
    write(p.relative_to(ROOT).as_posix(), value)
    manifest['entrypoints']['source']['manifest'] = source_path
    manifest['status'] = 'source_graph_cache_ownership_separated_history_unsealed'
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding='utf-8', newline='\n')
    activation = read('registry/migrations/nextqa_runtime_view_activation.json')
    activation['status'] = 'superseded_by_layer_ownership_v3'
    activation['superseded_by'] = 'registry/migrations/nextqa_layer_ownership_activation.json'
    write('registry/migrations/nextqa_runtime_view_activation.json', activation)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
