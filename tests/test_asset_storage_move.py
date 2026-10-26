import json
from pathlib import Path

import pytest

from scripts.datasets import migrate_asset_storage as migration


def test_relative_paths_cannot_escape_or_traverse_symlinks(tmp_path):
    for value in ('../other', '/absolute', 'data/../other', 'data//other', 'data\\other'):
        with pytest.raises(ValueError):
            migration.safe_path(value, tmp_path)
    assert migration.safe_path('data/input.bin', tmp_path) == tmp_path / 'data/input.bin'


def test_hash_verified_move_preserves_files_and_rejects_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, 'ROOT', tmp_path)
    source = tmp_path / 'data/source'
    source.mkdir(parents=True)
    (source / 'a.bin').write_bytes(bytes(range(256)))
    (source / 'b.json').write_text('{"graph":"unchanged"}\n', encoding='utf-8')
    before = migration.inventory(source)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'action': 'move', 'deletion_allowed': False,
        'receipt': 'registry/receipt.json', 'entries': [
            {'source': 'data/source', 'target': 'datasets/navqa/source/test/payload/source'}]}))
    migration.migrate(plan)
    target = tmp_path / 'datasets/navqa/source/test/payload/source'
    assert not source.exists()
    assert migration.inventory(target) == before
    migration.verify_receipt(tmp_path / 'registry/receipt.json')
    with pytest.raises(FileExistsError):
        migration.migrate(plan)


def test_overlapping_plan_fails_before_any_move(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, 'ROOT', tmp_path)
    source = tmp_path / 'data/source/nested'
    source.mkdir(parents=True)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'action': 'move', 'deletion_allowed': False,
        'receipt': 'registry/receipt.json', 'entries': [
            {'source': 'data/source', 'target': 'datasets/navqa/source/test/payload/all'},
            {'source': 'data/source/nested', 'target': 'datasets/navqa/source/test/payload/nested'}]}))
    with pytest.raises(ValueError, match='overlapping entries'):
        migration.migrate(plan)
    assert source.is_dir()
    assert not (tmp_path / 'registry/receipt.json').exists()


def test_unreadable_subtree_cannot_be_recorded_as_complete(tmp_path, monkeypatch):
    def inaccessible_walk(path, *, onerror):
        onerror(PermissionError('unreadable child directory'))
        yield str(path), [], []

    monkeypatch.setattr(migration.os, 'walk', inaccessible_walk)
    with pytest.raises(PermissionError, match='unreadable child'):
        migration.inventory(tmp_path)
