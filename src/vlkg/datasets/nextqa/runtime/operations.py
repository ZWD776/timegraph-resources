"""NExT-QA decisions behind the common dataset CLI."""
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.pipeline_commands import command, output_root
from vlkg.workspace import workspace_root


def plan(stage, run_id, mode='mock'):
    root = workspace_root()
    out = output_root('nextqa', run_id)
    def asset(layer, name):
        return asset_file('nextqa', layer, name)
    def config(name):
        return root / 'configs/datasets/nextqa' / name
    commands = []
    if stage == 'build':
        commands.append(command('nextqa', 'build', 'build_graph_full.py', '--run-id', run_id,
            '--manifest', asset('source', 'manifest'), '--config', config('graph.yaml')))
    elif stage == 'index':
        commands.append(command('nextqa', 'build', 'build_visual_pair_index_v2.py', '--run-id', run_id,
            '--base-index-manifest', asset('indexes', 'manifest')))
    elif stage == 'ir':
        commands.append(command('nextqa', 'retrieval', 'build_question_ir_llm.py', '--run-id', run_id,
            '--questions', asset('source', 'questions'), '--config', config('question_ir.yaml'), '--scope', 'full', '--mode', mode))
    elif stage == 'retrieve':
        commands.append(command('nextqa', 'retrieval', 'run_graph_retrieval_pilot.py', '--run-id', run_id,
            '--scope', 'full', '--question-ir', asset('ir', 'retrieval'), '--graph-run-root', asset('graphs', 'manifest').parent,
            '--config', config('retrieval.yaml')))
    elif stage == 'answer':
        commands.append(command('nextqa', 'qa', 'run_graph_only_qa_pilot.py', '--run-id', run_id,
            '--scope', 'full', '--question-ir', asset('ir', 'runtime'), '--retrieval-records', asset('retrieval', 'records'),
            '--config', config('graph_qa.yaml'), '--answer-mode', mode))
    return {'commands': commands, 'generated_files': {}, 'output_root': str(out),
            'scope': 'Build/index/IR/retrieval and graph-only QA. Fixed C composition is a separate retained recipe.'}
