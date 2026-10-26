"""NAVQA decisions behind the common dataset CLI."""
import json

from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.pipeline_commands import command, output_root
from vlkg.datasets.navqa.graph.contract import new_build_manifest
from vlkg.workspace import workspace_root


def plan(stage, run_id, mode='mock'):
    root = workspace_root()
    out = output_root('navqa', run_id)
    commands, files = [], {}
    if stage == 'build':
        manifest = out / 'build_manifest.json'
        files[str(manifest)] = new_build_manifest(out / 'graphs')
        commands.append(command('navqa', 'build', 'rebuild_navqa_clean_manifest_graphs.py', manifest, '--rebuild-temporal'))
    elif stage == 'index':
        commands.append(command('navqa', 'build', 'build_navqa_visual_faiss_index.py',
            '--sequence-manifest', root / 'configs/datasets/navqa/sequences.json',
            '--cache-db', asset_file('navqa', 'cache', 'embeddings'), '--output-dir', out / 'indexes'))
    elif stage == 'ir':
        source = json.loads(asset_file('navqa', 'source', 'manifest').read_text(encoding='utf-8'))
        for row in source['sequences']:
            seq = row['seq_id']
            commands.append(command('navqa', 'retrieval', 'build_question_ir.py',
                '--csv', asset_file('navqa', 'source', 'official_annotations'), '--seq-id', seq,
                '--layout-root', root / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa',
                '--config', root / 'configs/datasets/navqa/graph.yaml', '--non-position-only', '--mode', mode,
                '--output-json', out / 'ir' / f'seq{seq}.json', '--output-md', out / 'ir' / f'seq{seq}.md'))
    elif stage in ('retrieve', 'answer'):
        if stage == 'retrieve':
            commands.append(command('navqa', 'qa', 'run_navqa_mainline_all_sequences.py',
                '--sequence-manifest', root / 'configs/datasets/navqa/sequences_reuse.json',
                '--output-root', root / 'datasets/navqa/runs/new_runs/payload', '--run-name', run_id,
                '--config', root / 'configs/datasets/navqa/graph.yaml', '--answer-mode', 'graph_only',
                '--ir-mode', 'mock', '--text-judge-mode', 'none', '--reuse-manifest-question-ir',
                '--reuse-sequence-projection-cache', '--skip-question-ir-generation',
                '--experimental-question-router', '--experimental-question-router-mainline',
                '--experimental-visual-query-faiss-recall', '--run'))
        else:
            args = ['--route', 'CG6-clean+IR-QA-F-question-router-v4-mainline', '--run-name', run_id]
            args += ['--run', '--', '--answer-mode', mode, '--ir-mode', mode, '--text-judge-mode', 'live' if mode == 'live' else 'none']
            commands.append(command('navqa', 'qa', 'run_navqa_route.py', *args))
    return {'commands': commands, 'generated_files': files, 'output_root': str(out),
            'scope': 'Original full CG6. Retrieve executes all selected questions without LLM answering or judging; live reranking belongs to the answer stage.'}
