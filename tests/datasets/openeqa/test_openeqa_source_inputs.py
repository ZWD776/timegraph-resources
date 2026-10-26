from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.datasets.openeqa.build import build_hm3d_graphs_from_hsgg as builder
from vlkg.core.assets import asset_registry


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_episode(tmp_path: Path, *, aggregate_count: int = 2, sidecar_count: int = 39) -> tuple[Path, Path]:
    hsgg = tmp_path / 'hsgg/hm3d-v0/episode'
    frames = tmp_path / 'frames/hm3d-v0/episode'
    frames.mkdir(parents=True)
    tables: dict[str, dict] = {key: {} for key in ('objects', 'attributes', 'relations', 'hierarchy')}
    statuses = {}
    for index in range(sidecar_count):
        key = f'{index * 2:05d}-rgb'
        record = {
            'image_path': f'/cloud/old/frames/{key}.png',
            'objects': {'labels': ['chair'], 'boxes': [[0, 0, 4, 4]], 'scores': [0.8]},
            'attributes': {'0': 'wooden chair'}, 'relations': [], 'hierarchy': {},
        }
        write_json(hsgg / 'frame_scene_graphs' / f'{key}.json', record)
        (frames / f'{key}.png').write_bytes(b'')
        statuses[key] = {'status': 'completed'}
        if index < aggregate_count:
            for field in tables:
                tables[field][key] = record[field]
    for field, filename in (('objects', 'objects.json'), ('attributes', 'attributes.json'),
                            ('relations', 'relations.json'), ('hierarchy', 'hierarchy_mapping.json')):
        write_json(hsgg / filename, tables[field])
    write_json(hsgg / 'run_manifest.json', {'frames': statuses, 'completed_frames': sidecar_count})
    return hsgg, frames


def test_completed_sidecars_restore_37_frames_and_keep_ordinal_time(tmp_path, monkeypatch) -> None:
    hsgg, frames = make_episode(tmp_path)
    before = {path.relative_to(hsgg): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in hsgg.rglob('*.json')}
    def fail_archive_read(*args, **kwargs):
        raise AssertionError('OpenEQA standardization must read the local completed sidecars')
    monkeypatch.setattr(tarfile, 'open', fail_archive_read)
    output = tmp_path / 'output'
    result = builder.standardize_episode(
        hsgg_dir=hsgg, frames_dir=frames, output_dir=output, episode_history='hm3d-v0/episode',
        ordinal_step=1.0, frame_timestamp_index={}, force=False, dry_run=False)
    assert result['completed_frame_count'] == 39
    assert result['hsgg_record_source'] == 'completed_frame_scene_graphs'
    objects = json.loads((output / 'standard_json/objects.json').read_text())
    assert len(objects) == 39
    assert objects['frame_38']['image_path'] == str((frames / '00076-rgb.png').resolve())
    time = json.loads((output / 'frame_time_index.json').read_text())
    assert time['timestamp_semantics'] == 'ordinal_frame_order'
    assert time['timestamp_is_real_time'] is False
    assert [row['timestamp_sec'] for row in time['frames']] == list(range(39))
    assert time['frames'][-1]['source_frame_index'] == 76
    after = {path.relative_to(hsgg): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in hsgg.rglob('*.json')}
    assert after == before


def test_default_roots_select_requested_openeqa_shard(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / 'source_manifest.json'
    roots = {shard: {'hsgg': str(tmp_path / shard / 'hsgg'), 'frames': str(tmp_path / shard / 'frames')}
             for shard in ('hm3d', 'scannet')}
    write_json(manifest, {'dataset': 'openeqa', 'shards': roots})
    requested = []
    def selected_asset(dataset, layer, name, root):
        requested.append((dataset, layer, name))
        return manifest
    monkeypatch.setattr(asset_registry, 'asset_file', selected_asset)
    args = builder.build_parser().parse_args(['--data-source', 'scannet-v0', '--output-root', 'output'])
    hsgg, frames, selected_manifest = builder.resolve_input_roots(args)
    assert requested == [('openeqa', 'source', 'manifest')]
    assert (hsgg, frames) == (Path(roots['scannet']['hsgg']), Path(roots['scannet']['frames']))
    assert selected_manifest == manifest
    assert args.config == Path('datasets/openeqa/graph.yaml')


def test_explicit_legacy_roots_remain_paired_and_manifest_independent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(asset_registry, 'asset_file', lambda *args: pytest.fail('Explicit roots must not read the selected manifest'))
    args = builder.build_parser().parse_args([
        '--hsgg-root', str(tmp_path / 'old_hsgg'), '--frames-root', str(tmp_path / 'old_frames'),
        '--output-root', 'output'])
    assert builder.resolve_input_roots(args) == (tmp_path / 'old_hsgg', tmp_path / 'old_frames', None)
    args.frames_root = None
    with pytest.raises(ValueError, match='supplied together'):
        builder.resolve_input_roots(args)


def test_manifest_completed_frame_mismatch_is_rejected(tmp_path) -> None:
    hsgg, _ = make_episode(tmp_path)
    (hsgg / 'frame_scene_graphs/00076-rgb.json').unlink()
    with pytest.raises(ValueError, match='differ from run manifest'):
        builder.load_episode_hsgg_tables(hsgg)


def test_prepared_scannet_sidecars_do_not_require_a_copied_run_manifest(tmp_path) -> None:
    hsgg, _ = make_episode(tmp_path)
    (hsgg / 'run_manifest.json').unlink()
    tables, record_source = builder.load_episode_hsgg_tables(hsgg)
    assert record_source == 'completed_frame_scene_graphs'
    assert len(tables['objects']) == 39


def test_old_aggregate_standardization_is_not_silently_reused(tmp_path) -> None:
    hsgg, frames = make_episode(tmp_path)
    output = tmp_path / 'output'
    write_json(output / 'standard_json/objects.json', {'frame_0': {}})
    write_json(output / 'frame_time_index.json', {'frames': []})
    with pytest.raises(ValueError, match='use --force or a new --run-id'):
        builder.standardize_episode(
            hsgg_dir=hsgg, frames_dir=frames, output_dir=output, episode_history='hm3d-v0/episode',
            ordinal_step=1.0, frame_timestamp_index={}, force=False, dry_run=False)
