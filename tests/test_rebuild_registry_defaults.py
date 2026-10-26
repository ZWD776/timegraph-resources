from pathlib import Path
from types import SimpleNamespace
import json

from scripts.nextqa import build_visual_indexes as base
from scripts.nextqa import build_visual_pair_index_v2 as pairs

ROOT=Path(__file__).resolve().parents[1]


def test_base_index_new_run_uses_registry_graph_source_and_config():
    args=SimpleNamespace(config=None,graph_root=None,selection_manifest=None,resume=False)
    values=base.resolve_inputs(args,ROOT/'unused')
    assert values['config']==ROOT/'configs/datasets/nextqa/visual_index_build.json'
    assert values['graph_root']==ROOT/'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/runtime_paths_v3'
    assert values['selection']==ROOT/'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json'
    assert values['object_cache'].is_file()
    assert values['object_cache'].is_relative_to(ROOT/'datasets/nextqa/cache')


def test_pair_index_new_run_uses_current_audited_v2_bundle():
    args=SimpleNamespace(base_index_manifest=None,resume=False)
    path=pairs.resolve_base_manifest(args,ROOT/'unused')
    payload=json.loads(path.read_text(encoding='utf-8'))
    pairs.validate_base(payload)
    assert path==ROOT/'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/runtime_paths_v3/index_manifest.json'


def test_resume_keeps_recorded_rebuild_paths(tmp_path):
    run=tmp_path/'run_config.json'
    run.write_text(json.dumps({'contract':{
        'config':{'path':str(tmp_path/'old-config.json')},
        'graph_manifest':{'path':str(tmp_path/'old-graphs/nextqa_graph_build_manifest.json')},
        'selection_manifest':{'path':str(tmp_path/'old-selection.json')},
        'base_visual_index_manifest':{'path':str(tmp_path/'old-index.json')},
        'object_embedding_cache':{'path':str(tmp_path/'old-cache.sqlite')},
    }}))
    a=SimpleNamespace(config=None,graph_root=None,selection_manifest=None,resume=True)
    resolved=base.resolve_inputs(a,tmp_path)
    assert resolved['config']==tmp_path/'old-config.json'
    assert resolved['graph_root']==tmp_path/'old-graphs'
    assert resolved['selection']==tmp_path/'old-selection.json'
    b=SimpleNamespace(base_index_manifest=None,resume=True)
    assert pairs.resolve_base_manifest(b,tmp_path)==tmp_path/'old-index.json'
