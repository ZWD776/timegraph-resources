from __future__ import annotations
import sys
from importlib import import_module
from pathlib import Path
import json
from PIL import Image
from vlkg.core.config.config import load_domain_config
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'datasets' / 'openeqa'
NAVQA_GRAPH_CONFIG = REPO_ROOT / 'configs/datasets/navqa/graph.yaml'
SCANNET_GRAPH_CONFIG = REPO_ROOT / 'tests/fixtures/release_configs/openeqa_scannet_graph_only.yaml'
UNIFIED_GRAPH_CONFIG = REPO_ROOT / 'configs/datasets/openeqa/graph.yaml'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
graph_builder = import_module('scripts.datasets.openeqa.build.build_hm3d_graphs_from_hsgg')
unified_graph_builder = import_module('scripts.datasets.openeqa.build.build_openeqa_graphs_from_hsgg')
openeqa_converter = import_module('scripts.datasets.openeqa.build.convert_openeqa_json_to_temporal_graph')

def test_scannet_graph_profile_preserves_cg6_visual_association() -> None:
    navqa = load_domain_config(NAVQA_GRAPH_CONFIG, config_dir=REPO_ROOT / 'configs')
    scannet = load_domain_config(SCANNET_GRAPH_CONFIG, config_dir=REPO_ROOT / 'configs')
    assert navqa.association.enable_visual_embeddings is True
    assert scannet.association.enable_visual_embeddings is True
    assert scannet.association.local_tracklet_mode == navqa.association.local_tracklet_mode
    assert scannet.association.local_tracklet_embedding_similarity_threshold == navqa.association.local_tracklet_embedding_similarity_threshold

def test_unified_openeqa_graph_profile_is_effectively_cg6_for_all_sources() -> None:
    navqa = load_domain_config(NAVQA_GRAPH_CONFIG, config_dir=REPO_ROOT / 'configs').model_dump()
    scannet = load_domain_config(SCANNET_GRAPH_CONFIG, config_dir=REPO_ROOT / 'configs').model_dump()
    unified = load_domain_config(UNIFIED_GRAPH_CONFIG, config_dir=REPO_ROOT / 'configs').model_dump()
    navqa.pop('profile_name')
    scannet.pop('profile_name')
    unified.pop('profile_name')
    assert unified == navqa == scannet

def test_unified_openeqa_graph_entrypoint_forces_shared_contract(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(unified_graph_builder.sys, 'argv', ['build_openeqa_graphs_from_hsgg.py', '--data-source', 'scannet-v0'])
    monkeypatch.setattr(unified_graph_builder.legacy_builder, 'main', lambda: captured.extend(unified_graph_builder.sys.argv))
    unified_graph_builder.main()
    config = captured[captured.index('--config') + 1]
    assert Path(config) == REPO_ROOT / 'configs/datasets/openeqa/graph.yaml'
    assert '--graph-build-adapter' in captured
    assert 'none' in captured
    assert '--unified-openeqa-contract' in captured

def test_openeqa_converter_exposes_dataset_owned_embedding_cache() -> None:
    args = openeqa_converter.build_parser().parse_args(['standard_json', '--output', 'temporal_graph.json', '--video-id', 'scannet-v0__episode', '--config', str(SCANNET_GRAPH_CONFIG), '--frame-time-index', 'frame_time_index.json', '--embedding-cache-db', 'scannet_full.sqlite'])
    assert args.embedding_cache_db == Path('scannet_full.sqlite')

def test_graph_build_flow_uses_present_converter_and_owned_cache(tmp_path, monkeypatch):
    hsgg = tmp_path / 'hsgg/hm3d-v0/episode'
    frames = tmp_path / 'frames/hm3d-v0/episode'
    hsgg.mkdir(parents=True)
    frames.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(frames / '0.png')
    for name, data in {'objects': {'0': {'image_path': '0.png', 'labels': ['chair'], 'boxes': [[0, 0, 8, 8]]}}, 'attributes': {'0': {}}, 'relations': {'0': []}, 'hierarchy_mapping': {'0': {}}}.items():
        (hsgg / (name + '.json')).write_text(json.dumps(data), encoding='utf-8')
    cache = tmp_path / 'embeddings.sqlite'
    cache.write_bytes(b'')
    commands = []
    monkeypatch.setattr(graph_builder, 'run_command', lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(graph_builder.sys, 'argv', ['builder', '--hsgg-root', str(tmp_path / 'hsgg'), '--frames-root', str(tmp_path / 'frames'), '--output-root', str(tmp_path / 'output'), '--config', str(REPO_ROOT / 'configs/datasets/openeqa/graph.yaml'), '--embedding-cache-db', str(cache), '--run-id', 'owned-converter-check', '--dry-run'])
    graph_builder.main()
    assert len(commands) == 2
    assert all((Path(command[1]).is_file() for command in commands))
    assert Path(commands[0][1]).is_relative_to(REPO_ROOT / 'scripts/datasets/openeqa/build')
    args = openeqa_converter.build_parser().parse_args(commands[0][2:])
    assert args.embedding_cache_db == cache

def test_graph_builder_records_embedding_cache_argument() -> None:
    args = graph_builder.build_parser().parse_args(['--hsgg-root', 'hsgg', '--frames-root', 'frames', '--output-root', 'graphs', '--embedding-cache-db', 'scannet_full.sqlite'])
    assert args.embedding_cache_db == Path('scannet_full.sqlite')
