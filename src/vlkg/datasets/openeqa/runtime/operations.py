"""OpenEQA decisions behind the common dataset CLI."""
import json

from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.pipeline_commands import command, output_root
from vlkg.workspace import workspace_root


def plan(stage, run_id, mode='mock'):
    root = workspace_root()
    out = output_root('openeqa', run_id)
    source = json.loads(asset_file('openeqa', 'source', 'manifest').read_text(encoding='utf-8'))
    commands = []
    if stage == 'build':
        for shard, row in source['shards'].items():
            commands.append(command('openeqa', 'build', 'build_openeqa_graphs_from_hsgg.py',
                '--hsgg-root', root / row['hsgg'], '--frames-root', root / row['frames'],
                '--output-root', out / 'graphs' / shard, '--data-source', shard + '-v0', '--run-id', run_id,
                '--embedding-cache-db', asset_file('openeqa', 'cache', 'embeddings')))
    elif stage == 'index':
        for shard in ('hm3d', 'scannet'):
            commands.append(command('openeqa', 'build', 'build_openeqa_object_visual_index.py',
                '--graph-manifest', asset_file('openeqa', 'graphs', shard), '--output-dir', out / 'indexes' / shard,
                '--cache-db', asset_file('openeqa', 'cache', 'embeddings')))
    elif stage == 'ir':
        for shard in ('hm3d', 'scannet'):
            commands.append(command('openeqa', 'retrieval', 'run_openeqa_v5_semantic_ir.py',
                '--open-eqa-json', asset_file('openeqa', 'source', 'questions'), '--data-source', shard + '-v0',
                '--output-json', out / 'ir' / (shard + '.json'), '--output-md', out / 'ir' / (shard + '.md'),
                '--config', root / 'configs/datasets/openeqa/qa.yaml'))
        commands.append(command('openeqa', 'retrieval', 'build_openeqa_single_route_ir.py',
            '--open-eqa-json', asset_file('openeqa', 'source', 'questions'),
            '--source-question-ir', out / 'ir/hm3d.json', '--source-question-ir', out / 'ir/scannet.json',
            '--output-json', out / 'ir/question_ir.json'))
    elif stage in ('retrieve', 'answer'):
        args = ['--open-eqa-json', asset_file('openeqa', 'source', 'questions'),
                '--resource-manifest', asset_file('openeqa', 'retrieval', 'manifest'),
                '--question-route-json', asset_file('openeqa', 'ir', 'runtime'),
                '--output-root', out, '--run-name', run_id, '--visual-backbone', 'openeqa_ir_first_single_route_v5',
                '--config', root / 'configs/datasets/openeqa/qa.yaml', '--answer-mode', 'mock' if stage == 'retrieve' else mode,
                '--run']
        commands.append(command('openeqa', 'qa', 'run_openeqa_unified_visual_qa.py', *args))
    return {'commands': commands, 'generated_files': {}, 'output_root': str(out),
            'scope': 'HM3D/ScanNet are execution shards. Retrieve executes the full V5 evidence pipeline with mock answering; no answer API is called.'}
