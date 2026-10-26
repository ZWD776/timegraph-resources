"""Check invocation contracts without starting graph builds or model downloads."""
from __future__ import annotations

import json
from pathlib import Path
from vlkg.core.assets.asset_provenance import relocated_provenance
from types import SimpleNamespace
import importlib
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/openeqa'))
nextqa = importlib.import_module('scripts.nextqa.build_graph_full')
openeqa = importlib.import_module('scripts.datasets.openeqa.build.build_openeqa_graphs_from_hsgg')
visual = importlib.import_module('scripts.nextqa.build_visual_evidence_v2')


def test_nextqa_new_runs_use_materialized_configuration(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['builder', '--run-id', 'new-run'])
    args = nextqa.parse_args()
    assert nextqa.resolve_build_config(args) == ROOT / 'configs/datasets/nextqa/graph.yaml'


def test_nextqa_resume_preserves_recorded_config(tmp_path, monkeypatch):
    monkeypatch.setattr(nextqa, 'REPO_ROOT', tmp_path)
    base = tmp_path / 'datasets/nextqa/runs/new_runs/payload'
    run = base / 'graphs/frozen-run'
    run.mkdir(parents=True)
    monkeypatch.setattr(nextqa, 'NEXTQA_SCRATCH_ROOT', base)
    path = run / 'run_config.json'
    path.write_text(json.dumps({'contract': {'config': {'path': 'configs/recorded.yaml'}}}))
    before = path.read_bytes()
    args = SimpleNamespace(config=None, resume=True, run_id='frozen-run')
    assert nextqa.resolve_build_config(args) == tmp_path / 'configs/recorded.yaml'
    assert path.read_bytes() == before


def test_nextqa_resume_does_not_fallback_when_record_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(nextqa, 'REPO_ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        nextqa.resolve_build_config(SimpleNamespace(config=None, resume=True, run_id='unknown-run'))


def test_nextqa_new_build_manifest_is_registry_selected(tmp_path, monkeypatch):
    selected = tmp_path / 'selected_manifest.json'
    monkeypatch.setattr(nextqa, 'asset_file', lambda *args: selected)
    assert nextqa.resolve_build_manifest(SimpleNamespace(manifest=None, resume=False)) == selected


def test_nextqa_resume_uses_recorded_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(nextqa, 'REPO_ROOT', tmp_path)
    base = tmp_path / 'datasets/nextqa/runs/new_runs/payload'
    monkeypatch.setattr(nextqa, 'NEXTQA_SCRATCH_ROOT', base)
    record = base / 'graphs/frozen-run/run_config.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({'contract': {'manifest': {'path': 'original_manifest.json'}}}))
    args = SimpleNamespace(manifest=None, resume=True, run_id='frozen-run')
    assert nextqa.resolve_build_manifest(args) == tmp_path / 'original_manifest.json'


def test_nextqa_config_adapter_still_clears_component_labels():
    from vlkg.datasets.nextqa.graph.temporal_canonical import load_nextqa_graph_config
    original = load_nextqa_graph_config(ROOT / 'tests/fixtures/release_configs/nextqa/nextqa_cg6_human_compat_v2.yaml', config_dir=ROOT / 'configs')
    updated = load_nextqa_graph_config(nextqa.DEFAULT_CONFIG, config_dir=ROOT / 'configs')
    assert original.model_dump() == updated.model_dump()
    assert updated.component_labels == updated.component_host_labels == []


@pytest.mark.parametrize('argv', [
    ['--config=default.yaml'],
    ['--graph-build-adapter=hm3d_spatial_v1'],
    ['--graph-profile=canonical'],
])
def test_openeqa_guard_cannot_be_bypassed_with_equals(argv):
    with pytest.raises(SystemExit):
        openeqa.normalize_options(argv)


@pytest.mark.parametrize('source', ['hm3d-v0', 'scannet-v0'])
def test_openeqa_shards_use_same_build_contract(source):
    argv = openeqa.normalize_options(['--data-source', source, '--limit-episodes', '1'])
    assert argv[argv.index('--config') + 1] == str(ROOT / 'configs/datasets/openeqa/graph.yaml')
    assert argv[argv.index('--graph-profile') + 1] == 'time_fact_core'
    assert argv[argv.index('--data-source') + 1] == source
    assert argv[argv.index('--limit-episodes') + 1] == '1'


def test_openeqa_legacy_explicit_configuration_stays_available():
    argv = openeqa.normalize_options(['--config=datasets/openeqa/graph.yaml'])
    assert argv[argv.index('--config') + 1] == str(ROOT / 'configs/datasets/openeqa/graph.yaml')


def test_new_visual_run_resolves_registry_index(tmp_path, monkeypatch):
    target = tmp_path / 'selected.json'
    monkeypatch.setattr(visual, 'asset_file', lambda *args: target)
    args = SimpleNamespace(visual_index_manifest=None, resume=False)
    assert visual.resolve_index_manifest(args, tmp_path) == target


def test_visual_resume_preserves_manifest_fingerprint_path(tmp_path):
    path = tmp_path / 'run_config.json'
    recorded = tmp_path / 'original_manifest.json'
    path.write_text(json.dumps({'contract': {'visual_index_manifest': {'path': str(recorded)}}}))
    before = path.read_bytes()
    args = SimpleNamespace(visual_index_manifest=None, resume=True)
    assert visual.resolve_index_manifest(args, tmp_path) == recorded
    assert path.read_bytes() == before


@pytest.mark.requires_data
def test_default_visual_index_is_the_verified_parallel_copy():
    args = SimpleNamespace(visual_index_manifest=None, resume=False)
    try:
        selected = visual.resolve_index_manifest(args, ROOT / 'unused-run')
    except FileNotFoundError as exc:
        missing = Path(exc.filename or exc.args[0]).resolve()
        assert missing.is_relative_to(ROOT / 'datasets/nextqa')
        pytest.skip('requires external NextQA visual-index payloads; no original-project fallback')
    assert selected.is_relative_to(ROOT / 'datasets/nextqa/indexes')
    original = ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/visual_index_manifest.json'
    if not original.is_file() or not selected.is_file():
        pytest.skip('requires external NextQA visual-index payloads; no original-project fallback')
    old = json.loads(original.read_text(encoding='utf-8'))
    new = json.loads(selected.read_text(encoding='utf-8'))
    assert len(new['videos']) == len(old['videos']) == 570
    for before, after in zip(old['videos'], new['videos']):
        assert before['video_id'] == after['video_id']
        for kind, shard in before['shards'].items():
            copied = after['shards'][kind]
            assert {k: v for k, v in shard.items() if k not in {'index_path', 'metadata_path', 'metadata_bytes'}} == {
                k: v for k, v in copied.items() if k not in {'index_path', 'metadata_path', 'metadata_bytes'}
            }
            for field in ('index_path', 'metadata_path'):
                if copied[field] is None:
                    assert shard[field] is None and copied['count'] == 0
                else:
                    path = Path(copied[field])
                    assert path.is_relative_to(ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload')
                    assert path.is_file()
