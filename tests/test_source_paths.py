from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_runtime_image_metadata_relocates_without_mutating_original(tmp_path):
    from vlkg.core.assets.source_paths import relocate_image_references

    new = tmp_path / 'datasets/nextqa/source/current/items/42/frames/f.png'
    new.parent.mkdir(parents=True)
    new.write_bytes(b'unchanged image')
    old = 'datasets/nextqa/source/current/normalized_inputs/0001_42/frames/f.png'
    config = tmp_path / 'registry/source_path_aliases.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'aliases': [{'dataset': 'nextqa', 'source': 'datasets/nextqa/source/current/normalized_inputs/0001_42', 'target': 'datasets/nextqa/source/current/items/42'}]}))
    original = {'image_path': old, 'pair': {'image_paths': [old]}, 'note': old}
    updated = relocate_image_references(original, tmp_path)
    assert updated['image_path'] == str(new)
    assert updated['pair']['image_paths'] == [str(new)]
    assert updated['note'] == original['note'] == old
    assert original['image_path'] == old


def test_processed_attributes_use_declared_cache_not_source(tmp_path, monkeypatch):
    from vlkg.core.assets import source_paths
    from vlkg.core.temporal.processed_attributes import (
        get_or_create_processed_attribute_record, load_processed_attributes,
    )

    source = tmp_path / 'datasets/navqa/source/current/hsgg'
    cache = tmp_path / 'datasets/navqa/cache/current/processed'
    source.mkdir(parents=True)
    config = tmp_path / 'registry/source_path_aliases.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'derived_sidecars': [{'input_dir': source.relative_to(tmp_path).as_posix(), 'processed_attributes_dir': cache.relative_to(tmp_path).as_posix()}]}))
    monkeypatch.setattr(source_paths, 'workspace_root', lambda: tmp_path)
    record = get_or_create_processed_attribute_record(source, frame_name='frame_000001', object_index=0, label='person', raw_attributes=['blue shirt'])
    assert record is not None
    assert (cache / 'processed_attributes.json').is_file()
    assert not (source / 'processed_attributes.json').exists()
    assert load_processed_attributes(source)['frames']['frame_000001']
from PIL import Image

from vlkg.core.assets import source_paths
from vlkg.core.assets.source_paths import resolve_data_path, resolve_processed_attributes_dir


def _config(root: Path, aliases: list[dict], sidecars: list[dict] | None=None) -> Path:
    path = root / 'registry/source_path_aliases.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': 1, 'aliases': aliases, 'derived_sidecars': sidecars or []}), encoding='utf-8')
    return path


def _alias(source: str, target: str, dataset: str='nextqa') -> dict:
    return {'dataset': dataset, 'source': source, 'target': target}


def test_longest_prefix_and_transitive_directory_aliases(tmp_path: Path) -> None:
    _config(tmp_path, [_alias('old', 'wrong'), _alias('old/nested', 'middle'), _alias('middle', 'items/100')])
    target = tmp_path / 'items/100/frames/frame.png'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'frame')
    assert resolve_data_path('old/nested/frames/frame.png', tmp_path) == target
    assert resolve_data_path(tmp_path / 'old/nested', tmp_path) == target.parent.parent
    assert resolve_data_path('old-sibling/frame.png', tmp_path) == tmp_path / 'old-sibling/frame.png'


def test_existing_original_file_and_directory_take_priority(tmp_path: Path) -> None:
    _config(tmp_path, [_alias('old', 'new')])
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'input.json').write_text('{}')
    assert resolve_data_path(old, tmp_path) == old
    assert resolve_data_path(old / 'input.json', tmp_path) == old / 'input.json'


def test_external_input_is_untouched_and_never_statted(tmp_path: Path, monkeypatch) -> None:
    external = tmp_path.parent / 'explicit_external/frame.png'
    def deny_exists(path):
        raise AssertionError('external resolution must not inspect paths')
    monkeypatch.setattr(Path, 'exists', deny_exists)
    assert resolve_data_path(external, tmp_path) == external


@pytest.mark.parametrize('aliases', [
    [_alias('a', 'b'), _alias('b', 'a')],
    [_alias('a', 'a/expansion')],
])
def test_cycles_and_recursive_expansion_are_rejected(tmp_path: Path, aliases: list[dict]) -> None:
    _config(tmp_path, aliases)
    with pytest.raises(ValueError, match='cycle'):
        resolve_data_path('a/frame.png', tmp_path)


@pytest.mark.parametrize('source,target', [
    ('../outside', 'datasets/nextqa/source/items'),
    ('datasets/nextqa/source/old', '../outside'),
    ('datasets/navqa/source/old', 'datasets/nextqa/source/items'),
    ('datasets/nextqa/source/old', 'datasets/openeqa/source/items'),
])
def test_aliases_cannot_escape_or_cross_dataset_owners(tmp_path: Path, source: str, target: str) -> None:
    _config(tmp_path, [_alias(source, target)])
    with pytest.raises(ValueError, match='workspace|ownership'):
        resolve_data_path('missing.png', tmp_path)


def test_alias_snapshot_is_cached_and_reloads_after_config_change(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, [_alias('old', 'new')])
    reads = []
    original = Path.read_text
    def record(path, *args, **kwargs):
        if path == config:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', record)
    for index in range(20):
        assert resolve_data_path(f'old/{index}.png', tmp_path) == tmp_path / f'new/{index}.png'
    assert len(reads) == 1
    _config(tmp_path, [_alias('old', 'new_longer')])
    assert resolve_data_path('old/0.png', tmp_path) == tmp_path / 'new_longer/0.png'
    assert len(reads) == 2


def test_sidecar_lookup_follows_source_alias_without_creating_dirs(tmp_path: Path) -> None:
    _config(tmp_path, [_alias('datasets/navqa/source/old', 'datasets/navqa/source/items', 'navqa')], [{'input_dir': 'datasets/navqa/source/items', 'processed_attributes_dir': 'datasets/navqa/cache/attributes'}])
    target = resolve_processed_attributes_dir('datasets/navqa/source/old', tmp_path)
    assert target == tmp_path / 'datasets/navqa/cache/attributes'
    assert not target.exists()


def test_actual_crop_resolves_old_image_reference_without_model_or_cache_key_rewrite(tmp_path: Path, monkeypatch) -> None:
    from vlkg.core.indexing.vision import EmbeddingCache, VisionFeatureExtractor
    from vlkg.core.models.temporal_graph import BoundingBox
    import numpy as np
    _config(tmp_path, [_alias('datasets/nextqa/source/old', 'datasets/nextqa/source/items/100')])
    new = tmp_path / 'datasets/nextqa/source/items/100/frames/frame.png'
    new.parent.mkdir(parents=True)
    Image.new('RGB', (8, 6), color=(12, 34, 56)).save(new)
    old = tmp_path / 'datasets/nextqa/source/old/frames/frame.png'
    monkeypatch.setattr(source_paths, 'workspace_root', lambda: tmp_path)
    extractor = VisionFeatureExtractor(device='cpu')
    bbox = BoundingBox(x1=1, y1=1, x2=5, y2=4)
    crop = extractor._crop_image(image_path=old, bbox=bbox)
    assert crop.size == (4, 3)
    assert crop.getpixel((0, 0)) == (12, 34, 56)
    assert extractor._model is None
    cache = EmbeddingCache(tmp_path / 'cache/test.sqlite')
    try:
        cache.set_many([(old, bbox, np.asarray([1.0, 0.0], dtype=np.float32))])
        assert cache.get(old, bbox) is not None
        assert cache.get(new, bbox) is None
        assert cache._connection.execute('SELECT image_path FROM embedding_cache').fetchone()[0] == str(old)
    finally:
        cache.close()


def test_repo_input_path_follows_alias_for_missing_absolute_and_relative(tmp_path: Path, monkeypatch) -> None:
    from vlkg.core.temporal import pathing
    _config(tmp_path, [_alias('old', 'new')])
    monkeypatch.setattr(pathing, 'REPO_ROOT', tmp_path)
    assert pathing.resolve_repo_input_path('old/input') == tmp_path / 'new/input'
    assert pathing.resolve_repo_input_path(tmp_path / 'old/input') == tmp_path / 'new/input'
    assert not (tmp_path / 'data').exists()
