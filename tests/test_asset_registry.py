from pathlib import Path
from vlkg.core.assets.asset_provenance import relocated_provenance
import json
import hashlib
import pytest

from vlkg.core.assets.asset_registry import (
    DATASET_KEYS,
    LAYERS,
    current_paths,
    load_dataset_manifest,
    load_pointer,
    metadata_paths,
    validate_symmetric_layout,
    inspect_dataset,
    graph_config_path,
    verify_metadata_copies,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_manifests_are_symmetric() -> None:
    validate_symmetric_layout()
    for dataset in DATASET_KEYS:
        manifest = load_dataset_manifest(dataset)
        assert tuple(manifest["target_layout"]) == LAYERS


def test_mainline_graph_pointers_exist() -> None:
    expected = {
        "navqa": "navqa-cg6-time-fact-core-v1",
        "nextqa": "nextqa-cg6-human-v2-time-fact-core-v1",
        "openeqa": "openeqa-time-fact-core-v1",
    }
    for dataset, asset_id in expected.items():
        pointer = load_pointer(dataset, "graphs", asset_id)
        assert isinstance(pointer["payload_migrated"], bool)
        assert pointer["current_paths"]
        assert all(path.exists() for path in current_paths(dataset, "graphs", asset_id))


def test_navqa_ir_matches_actual_mainline_sequence_manifest() -> None:
    pointer = load_pointer("navqa", "ir", "navqa-cg6-question-ir-v1")
    manifest = json.loads((ROOT / pointer["source_manifest"]).read_text(encoding="utf-8"))
    expected = [s["question_ir_json"] for s in manifest["sequences"]]
    bundle = json.loads((ROOT / pointer['entrypoints']['runtime']).read_text(encoding='utf-8'))
    assert [row['path'] for row in bundle['by_sequence']] == expected
    assert len(expected) == 7
    assert all(p.exists() for p in current_paths("navqa", "ir", pointer["asset_id"]))


def test_metadata_migration_is_hash_preserving_for_graph_reports() -> None:
    paths = metadata_paths("navqa", "graphs", "navqa-cg6-time-fact-core-v1")
    assert len(paths) == 7
    assert all(path.exists() for path in paths)
    assert verify_metadata_copies() == []


@pytest.mark.parametrize("dataset", DATASET_KEYS)
def test_all_registered_paths_agree_and_exist(dataset) -> None:
    assert inspect_dataset(dataset)["issues"] == []


@pytest.mark.parametrize("dataset,source", [
    ("navqa", "navqa_cg6_local_tracklet.yaml"),
    ("nextqa", "nextqa_cg6_human_compat_v2.yaml"),
    ("openeqa", "openeqa_cg6_unified_full.yaml"),
])
def test_materialized_graph_config_preserves_effective_parameters(dataset, source) -> None:
    from vlkg.core.config.config import load_domain_config
    assert load_domain_config(graph_config_path(dataset)).model_dump() == load_domain_config(
        relocated_provenance(ROOT / "configs" / source, ROOT), config_dir=ROOT / "configs"
    ).model_dump()


@pytest.mark.parametrize("asset_id", ["..", "../outside", "x/y", "C:\\outside"])
def test_pointer_rejects_path_as_asset_id(asset_id) -> None:
    with pytest.raises(ValueError, match="invalid asset identifier"):
        load_pointer("navqa", "graphs", asset_id)


def test_metadata_verification_detects_corruption(tmp_path) -> None:
    (tmp_path / 'registry/migrations').mkdir(parents=True)
    (tmp_path / 'source.json').write_bytes(b'{}')
    (tmp_path / 'copy.json').write_bytes(b'{"changed":true}')
    digest = hashlib.sha256(b'{}').hexdigest()
    (tmp_path / 'registry/migrations/metadata_migration_v1.json').write_text(json.dumps({
        'items': [{'source': 'source.json', 'target': 'copy.json',
                   'source_sha256': digest, 'target_sha256': digest}]
    }))
    assert verify_metadata_copies(tmp_path) == ['hash mismatch: copy.json']


def test_dataset_route_configuration_preserves_cg6_route_behavior() -> None:
    old = json.loads(relocated_provenance(ROOT / 'configs/navqa_cg6_routes.json', ROOT).read_text(encoding='utf-8'))
    new = json.loads((ROOT / 'configs/datasets/navqa/routes.json').read_text(encoding='utf-8'))
    default_paths = {'csv', 'layout_root'}
    assert {k: v for k, v in new['defaults'].items() if k not in default_paths} == {
        k: v for k, v in old['defaults'].items() if k not in default_paths}
    assert len(new['routes']) == len(old['routes'])
    for a, b in zip(old['routes'], new['routes']):
        assert b['config'] == 'datasets/navqa/graph.yaml'
        changed = {'config', 'sequence_manifest', 'output_root', 'args'}
        assert {k: v for k, v in a.items() if k not in changed} == {k: v for k, v in b.items() if k not in changed}
        assert [value for value in a['args'] if not value.startswith('.scratch/')] == [
            value for value in b['args'] if not value.startswith('datasets/navqa/')]
        original_path = relocated_provenance(ROOT / a['sequence_manifest'], ROOT)
        original = json.loads(original_path.read_text(encoding='utf-8'))
        updated = json.loads((ROOT / b['sequence_manifest']).read_text(encoding='utf-8'))
        path_fields = {'sample_root', 'hsgg_input_dir', 'temporal_graph_json', 'canonical_graph_json',
                       'frame_time_index_json', 'qa_unfilled_json', 'question_ir_json', 'projection_cache', 'reuse_cache_source'}
        assert [{k: v for k, v in row.items() if k not in path_fields} for row in original['sequences']] == [
            {k: v for k, v in row.items() if k not in path_fields} for row in updated['sequences']]
        assert b['output_root'] == 'datasets/navqa/runs/new_runs/payload'
        assert updated['temporal_graph_config'] == 'datasets/navqa/graph.yaml'
