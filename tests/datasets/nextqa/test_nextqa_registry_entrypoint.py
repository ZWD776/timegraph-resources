from pathlib import Path
import json

import pytest
import yaml

from scripts.datasets import nextqa_mainline as entry
from scripts.datasets.activate_nextqa_mainline import rebase



def _make_script_entries(root: Path) -> None:
    # stage_command validates the entrypoint without executing it.
    for relative in (
        'retrieval/run_graph_retrieval_pilot.py',
        'qa/run_graph_only_qa_pilot.py',
        'retrieval/build_visual_evidence_v2.py',
        'qa/run_visual_enhanced_qa.py',
        'qa/merge_visual_route_policy_v3.py',
    ):
        path = root / 'scripts/datasets/nextqa' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# test entrypoint placeholder; never executed\n', encoding='utf-8')


def test_owned_retrieval_and_qa_configs_load_after_renaming():
    from vlkg.datasets.nextqa.retrieval.graph_only_retrieval import load_unified_retrieval_config
    from vlkg.datasets.nextqa.qa.graph_only_qa import load_graph_only_mcq_config
    retrieval = load_unified_retrieval_config(entry.ROOT / 'configs/datasets/nextqa/retrieval.yaml')
    qa = load_graph_only_mcq_config(entry.ROOT / 'configs/datasets/nextqa/graph_qa.yaml')
    assert retrieval.profile_name == 'nextqa_graph_only_unified_v2'
    assert retrieval.evidence_limit == 16
    assert qa.evidence_limit == 16


@pytest.mark.parametrize('stage', ['retrieve', 'graph-qa', 'visual-evidence', 'visual-base-qa', 'compose-C'])
def test_new_stage_uses_registry_inputs(stage, tmp_path, monkeypatch):
    _make_script_entries(tmp_path)
    seen = []
    def fake(dataset, layer, name, root):
        seen.append((dataset, layer, name))
        return tmp_path / 'datasets' / dataset / layer / (name + '.json')
    monkeypatch.setattr(entry, 'asset_file', fake)
    command = entry.stage_command(stage, 'new-mainline-run', root=tmp_path)
    assert command[command.index('--run-id') + 1] == 'new-mainline-run'
    assert all(dataset == 'nextqa' for dataset, _, _ in seen)
    assert '--resume' not in command
    if stage != 'compose-C':
        assert str(tmp_path / 'configs/datasets/nextqa') in command[command.index('--config') + 1]
    if 'qa' in stage:
        assert command[command.index('--answer-mode') + 1] == 'mock'


def test_compose_c_uses_registered_components(tmp_path, monkeypatch):
    _make_script_entries(tmp_path)
    calls=[]
    def fake(dataset,layer,name,root):
        calls.append((layer,name))
        return tmp_path/layer/(name+'.json')
    monkeypatch.setattr(entry,'asset_file',fake)
    command=entry.stage_command('compose-C','compose-test',root=tmp_path)
    assert set(calls)=={('runs','causal_base'),('runs','policy_overlay'),('retrieval','routes')}
    assert '--answer-mode' not in command


def test_graph_qa_explicit_live_mode_is_forwarded(tmp_path, monkeypatch):
    _make_script_entries(tmp_path)
    monkeypatch.setattr(entry, 'asset_file', lambda *args: tmp_path / 'input.json')
    cmd = entry.stage_command('graph-qa', 'new-live-run', root=tmp_path, answer_mode='live')
    assert cmd[cmd.index('--answer-mode') + 1] == 'live'


def test_derived_manifest_rewrites_only_path_values(tmp_path):
    source = {'id': 'keep-id', 'metric': 42, 'path': str(tmp_path / '.scratch/nextqa/graphs/v1/graph.json'),
              'rows': [{'file': '.scratch/nextqa/graphs/v1/a.json', 'note': 'historical graph v1'}]}
    before = json.dumps(source)
    updated = rebase(source, {'.scratch/nextqa/graphs/v1': 'datasets/nextqa/graphs/selected/payload'}, tmp_path)
    assert json.dumps(source) == before
    assert updated['id'] == source['id'] and updated['metric'] == 42
    assert updated['rows'][0]['note'] == source['rows'][0]['note']
    assert updated['rows'][0]['file'] == 'datasets/nextqa/graphs/selected/payload/a.json'
    assert updated['path'].endswith('/datasets/nextqa/graphs/selected/payload/graph.json')


@pytest.mark.requires_data
def test_actual_stage_commands_use_migrated_inputs():
    for stage in ('retrieve', 'graph-qa', 'visual-evidence', 'visual-base-qa'):
        try:
            command = entry.stage_command(stage, 'offline-migration-check')
        except FileNotFoundError as exc:
            missing = Path(exc.filename or exc.args[0]).resolve()
            assert missing.is_relative_to(entry.ROOT / 'datasets/nextqa')
            pytest.skip('requires external NextQA downstream payloads; no original-project fallback')
        options = {'--question-ir', '--graph-run-root', '--retrieval-records', '--route-records',
                   '--visual-index-manifest', '--frozen-graph-evidence', '--visual-evidence'}
        for i, arg in enumerate(command):
            if arg in options:
                target = Path(command[i + 1])
                if not target.exists():
                    pytest.skip('requires external NextQA downstream payloads; no original-project fallback')
                assert target.exists()
                assert target.is_relative_to(entry.ROOT / 'datasets/nextqa')


def test_selected_configs_are_declared_in_release_manifest():
    manifest = yaml.safe_load((entry.ROOT / 'configs/datasets/nextqa/config_manifest.yaml').read_text(encoding='utf-8'))
    paths = {row['path'] for row in manifest['configs']}
    assert manifest['dataset'] == 'nextqa'
    assert manifest['graph_config'] == 'configs/datasets/nextqa/graph.yaml'
    assert {'configs/datasets/nextqa/retrieval.yaml', 'configs/datasets/nextqa/graph_qa.yaml'} <= paths
    assert all((entry.ROOT / path).is_file() for path in paths)


@pytest.mark.requires_data
def test_active_source_manifest_covers_repaired_directories():
    from vlkg.core.assets.asset_registry import asset_file
    from vlkg.datasets.nextqa.graph.hsgg_input_adapter import resolve_normalized_input_paths
    try:
        path = asset_file('nextqa', 'source', 'manifest')
    except FileNotFoundError as exc:
        missing = Path(exc.filename or exc.args[0]).resolve()
        assert missing.is_relative_to(entry.ROOT / 'datasets/nextqa/source')
        pytest.skip('requires external normalized NextQA source payload')
    if not path.is_file():
        pytest.skip('requires external normalized NextQA source payload')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert len(manifest['videos']) == 570
    assert manifest['input_storage'] == 'normalized_directory'
    for video in manifest['videos']:
        assert video['input_storage'] == 'normalized_directory'
        input_dir, frames_dir = resolve_normalized_input_paths(video)
        for path in (input_dir, frames_dir):
            assert path.is_relative_to(entry.ROOT / 'datasets/nextqa/source')
            assert path.is_dir()
        for name in ('objects.json', 'attributes.json', 'relations.json', 'hierarchy_mapping.json', 'timeline.json', 'source_trace.json'):
            assert (input_dir / name).is_file()


@pytest.mark.requires_data
def test_current_special_patch_uses_complete_directory_inputs():
    from vlkg.datasets.nextqa.runtime.special_patch import load_special_graph_manifest
    path = entry.ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/manifests/repaired29_manifest.json'
    if not path.is_file():
        pytest.skip('requires external NextQA repaired29 source manifest')
    manifest = load_special_graph_manifest(path)
    assert manifest['input_storage'] == 'normalized_directory'
    assert len(manifest['videos']) == 29
    assert sum(video['skipped_frame_count'] for video in manifest['videos']) == 11


def test_output_root_is_opt_in_and_dataset_isolated(tmp_path):
    from vlkg.datasets.nextqa.runtime.io_utils import select_run_root
    assert select_run_root(None, repo_root=tmp_path) == tmp_path / 'datasets/nextqa/runs/new_runs/payload'
    environment = entry.execution_environment(tmp_path)
    assert select_run_root(environment['VLKG_NEXTQA_RUN_ROOT'], repo_root=tmp_path) == tmp_path / 'datasets/nextqa/runs/new_runs/payload'
    with pytest.raises(ValueError, match='must stay inside'):
        select_run_root('datasets/openeqa/runs/test/payload', repo_root=tmp_path)
    with pytest.raises(ValueError, match='must stay inside'):
        select_run_root('datasets/nextqa/runs', repo_root=tmp_path)
