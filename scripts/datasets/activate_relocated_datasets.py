"""Activate verified NAVQA/OpenEQA relocation views in the symmetric registry."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from vlkg.core.assets.asset_relocation import PathPrefixRewriter
from scripts.datasets.rebase_navqa_runtime import mapping as navqa_mapping
from scripts.datasets.rebase_openeqa_runtime import mapping as openeqa_mapping


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def write(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def activate(dataset, entries):
    manifest_path = ROOT / f'registry/datasets/{dataset}.yaml'
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    manifest['entrypoints'] = {}
    for layer, named in entries.items():
        for name, value in named.items():
            if not (ROOT / value).exists():
                raise FileNotFoundError(f'{dataset}/{layer}/{name}: {value}')
        pointer_path = next((ROOT / f'datasets/{dataset}/{layer}').glob('*/asset_pointer.json'))
        pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
        pointer['previous_paths'] = pointer['current_paths']
        pointer['current_paths'] = list(dict.fromkeys(named.values()))
        pointer['entrypoints'] = named
        pointer['dependency_paths'] = []
        pointer['status'] = 'relocated_runtime_pending_behavior_validation'
        pointer['payload_migrated'] = True
        pointer['migration'] = {'copy': False, 'move': True, 'delete': False,
                                'next_step': 'offline_resource_and_behavior_validation'}
        write(pointer_path.relative_to(ROOT).as_posix(), pointer)
        manifest['current_paths']['graph' if layer == 'graphs' else layer] = pointer['current_paths']
        manifest['entrypoints'][layer] = named
    manifest['status'] = 'mainline_relocated_pending_behavior_validation'
    manifest['planning_only'] = False
    manifest['migration'] = {'copy_payloads': False, 'move_payloads': True, 'delete_payloads': False,
                             'next_gate': 'offline_behavior_validation_and_history_sealing'}
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding='utf-8', newline='\n')


def main():
    for dataset in ('navqa', 'openeqa'):
        if read(f'registry/migrations/{dataset}_mainline_move_receipt.json')['status'] != 'verified':
            raise RuntimeError('incomplete move receipt')
        read(f'registry/migrations/{dataset}_runtime_views.json')
        if read(f'registry/migrations/{dataset}_embedding_cache.json')['status'] != 'verified':
            raise RuntimeError('cache split not verified')
    writer = PathPrefixRewriter(navqa_mapping(), ROOT)
    graph = 'datasets/navqa/graphs/navqa-cg6-time-fact-core-v1/payload/runtime'
    source = 'datasets/navqa/source/navqa-coda-every10-v1/payload'
    ir = 'datasets/navqa/ir/navqa-cg6-question-ir-v1/payload'
    retrieval = 'datasets/navqa/retrieval/navqa-cg6-retrieval-v1/payload'
    index = 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime'
    runs = 'datasets/navqa/runs/navqa-cg6-qa-v1/payload'
    for name in ('sequences.json', 'sequences_reuse.json'):
        original = read('configs/datasets/navqa/' + name)
        snapshot = ROOT / 'archive/legacy_20260928/migration/before_dataset_activation/navqa' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(ROOT / 'configs/datasets/navqa' / name, snapshot)
        document = writer.rewrite_value(original)
        document['default_output_root'] = 'datasets/navqa/runs/new_runs/payload'
        for row in document['sequences']:
            seq = row['seq_id']
            if 'projection_cache' in row:
                row['projection_cache'] = retrieval + f'/seq{seq}_cam0_every10/runtime_projection_cache.json.gz'
            if 'reuse_cache_source' in row:
                row['reuse_cache_source']['seq_dir'] = retrieval + f'/seq{seq}_cam0_every10'
                row['reuse_cache_source']['root'] = retrieval
        write('configs/datasets/navqa/' + name, document)
        if name == 'sequences.json':
            write(graph + '/graph_manifest.json', document)
            write(source + '/source_manifest.json', {'dataset': 'navqa', 'source_id': 'navqa-coda-every10-v1',
                'sequences': [{key: value for key, value in row.items() if key in
                    ('seq_id', 'cam', 'stride', 'sample_root', 'hsgg_input_dir', 'frame_time_index_json', 'qa_unfilled_json')}
                    for row in document['sequences']], 'official_annotations': source + '/official/qa.csv'})
    routes = read('configs/datasets/navqa/routes.json')
    for row in routes['routes']:
        row['output_root'] = 'datasets/navqa/runs/new_runs/payload'
    write('configs/datasets/navqa/routes.json', routes)
    write(ir + '/question_ir_manifest.json', {'dataset': 'navqa', 'by_sequence': [
        {'seq_id': seq, 'path': ir + f'/seq{seq}_cam0_every10/question_ir.json'} for seq in ('0', '3', '4', '6', '16', '21', '22')]})
    write(retrieval + '/retrieval_manifest.json', {'dataset': 'navqa',
        'sequence_manifest': 'configs/datasets/navqa/sequences_reuse.json', 'routes': 'configs/datasets/navqa/routes.json'})
    faiss_manifest = index + '/object_siglip_flat_ip.manifest.json'
    visual = read(faiss_manifest)
    visual['index_path'] = str((ROOT / 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/visual_faiss/object_siglip_flat_ip.faiss').resolve())
    visual['metadata_sqlite_path'] = str((ROOT / index / 'object_siglip_flat_ip.metadata.sqlite').resolve())
    write(faiss_manifest, visual)
    write(index + '/index_manifest.json', {'dataset': 'navqa', 'graph_profile': 'time_fact_core',
        'visual_manifest': faiss_manifest, 'evidence_layer': index + '/evidence_layer.sqlite'})
    for dataset in ('navqa', 'openeqa'):
        report_asset = 'navqa-reports-v1' if dataset == 'navqa' else 'openeqa-reports-v1'
        for suffix in ('runtime_views', 'embedding_cache'):
            target = f'datasets/{dataset}/reports/{report_asset}/metadata/{suffix}.json'
            write(target, read(f'registry/migrations/{dataset}_{suffix}.json'))
    activate('navqa', {
        'source': {'manifest': source + '/source_manifest.json', 'official_annotations': source + '/official/qa.csv'},
        'graphs': {'manifest': graph + '/graph_manifest.json'},
        'indexes': {'manifest': index + '/index_manifest.json', 'visual_manifest': faiss_manifest,
                    'evidence_layer': index + '/evidence_layer.sqlite'},
        'ir': {'runtime': ir + '/question_ir_manifest.json'},
        'retrieval': {'manifest': retrieval + '/retrieval_manifest.json'},
        'runs': {'A': runs + '/CG6-clean_IR-QA-G-clean_full_20260601/mainline_all_sequences_summary.json',
                 'B': runs + '/CG6-clean_IR-QA-F-GER-v3-binding_full_20260601/mainline_all_sequences_summary.json',
                 'C': runs + '/navqa_deepseek_bestline_composite_20260902/navqa_route_composite.json'},
        'cache': {'embeddings': 'datasets/navqa/cache/navqa-cache-external-v1/payload/embedding_cache.sqlite'},
        'reports': {'runtime_views': 'datasets/navqa/reports/navqa-reports-v1/metadata/runtime_views.json',
                    'embedding_cache': 'datasets/navqa/reports/navqa-reports-v1/metadata/embedding_cache.json'},
    })
    source = 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload'
    graph = 'datasets/openeqa/graphs/openeqa-time-fact-core-v1/payload/runtime'
    index = 'datasets/openeqa/indexes/openeqa-v5-full-visual-indexes/payload/runtime'
    ir = 'datasets/openeqa/ir/openeqa-v5-14class-strategy-v4/payload'
    runs = 'datasets/openeqa/runs/openeqa-v5-qa-mainline-luna/payload'
    write(source + '/source_manifest.json', {'dataset': 'openeqa', 'source_id': 'openeqa-em-eqa-hm3d-scannet-v1',
        'official_annotations': source + '/official/open-eqa-v0.json', 'shards': {
            'hm3d': {'hsgg': source + '/hm3d/hsgg_stride2', 'frames': source + '/hm3d/frames_stride2_offset0'},
            'scannet': {'hsgg': source + '/scannet/hsgg_prepared', 'frames': source + '/scannet/frames_stride2'}}})
    write(graph + '/graph_manifest.json', {'dataset': 'openeqa', 'graph_profile': 'time_fact_core',
        'execution_shards': [{'data_source': shard + '-v0', 'manifest': graph + '/' + shard + '/openeqa_hm3d_graph_build_manifest.json'}
            for shard in ('hm3d', 'scannet')]})
    write(index + '/index_manifest.json', {'dataset': 'openeqa', 'execution_shards': [
        {'data_source': shard + '-v0', 'manifest': index + '/' + shard + '/openeqa_object_siglip_flat_ip.manifest.json'}
        for shard in ('hm3d', 'scannet')]})
    writer = PathPrefixRewriter(openeqa_mapping(), ROOT)
    ir_data = writer.rewrite_value(read(ir + '/question_ir.json'))
    write(ir + '/runtime_question_ir.json', ir_data)
    activate('openeqa', {
        'source': {'manifest': source + '/source_manifest.json', 'questions': source + '/official/open-eqa-v0.json'},
        'graphs': {'manifest': graph + '/graph_manifest.json',
                   'hm3d': graph + '/hm3d/openeqa_hm3d_graph_build_manifest.json',
                   'scannet': graph + '/scannet/openeqa_hm3d_graph_build_manifest.json'},
        'indexes': {'manifest': index + '/index_manifest.json',
                    'hm3d': index + '/hm3d/openeqa_object_siglip_flat_ip.manifest.json',
                    'scannet': index + '/scannet/openeqa_object_siglip_flat_ip.manifest.json'},
        'ir': {'runtime': ir + '/runtime_question_ir.json'},
        'retrieval': {'manifest': 'configs/datasets/openeqa/execution_resources.json'},
        'runs': {'A': runs + '/A/full_v5_ir_key0_temp0/predictions.json',
                 'B': runs + '/B/full_v5_ir_key0_temp0/predictions.json', 'C': runs + '/C/predictions.json'},
        'cache': {'embeddings': 'datasets/openeqa/cache/openeqa-cache-external-v1/payload/embedding_cache.sqlite'},
        'reports': {'runtime_views': 'datasets/openeqa/reports/openeqa-reports-v1/metadata/runtime_views.json',
                    'embedding_cache': 'datasets/openeqa/reports/openeqa-reports-v1/metadata/embedding_cache.json'},
    })
    print('NAVQA and OpenEQA active pointers switched; behavior validation remains pending.')


if __name__ == '__main__':
    main()
