import json
from pathlib import Path

import pytest

from scripts.datasets.copy_mainline_assets import copy_file, enumerate_plan, execute


def plan_for(root):
    source = root / '.scratch/nextqa/selected'
    source.mkdir(parents=True)
    (source / 'record.json').write_bytes(b'{"q":1}\r\n')
    plan = {'dataset': 'nextqa', 'delete_sources': False, 'entries': [
        {'source': '.scratch/nextqa/selected', 'target': 'datasets/nextqa/ir/v1/payload/selected'}]}
    p = root / 'plan.json'
    p.write_text(json.dumps(plan))
    return p, source


def test_copy_is_byte_preserving_and_replayable(tmp_path):
    p, source = plan_for(tmp_path)
    first = execute(p, tmp_path)
    assert execute(p, tmp_path) == first
    assert (tmp_path / first['rows'][0]['target']).read_bytes() == (source / 'record.json').read_bytes()


def test_copy_rejects_existing_different_file(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_bytes(b'original')
    b.write_bytes(b'user content')
    with pytest.raises(FileExistsError):
        copy_file(a, b)
    assert b.read_bytes() == b'user content'


def test_duplicate_destinations_block_before_copy(tmp_path):
    p, source = plan_for(tmp_path)
    plan = json.loads(p.read_text())
    plan['entries'] *= 2
    with pytest.raises(ValueError, match='duplicate target'):
        enumerate_plan(plan, tmp_path)
    assert not (tmp_path / 'datasets').exists()


def test_external_target_rejected(tmp_path):
    p, _ = plan_for(tmp_path)
    plan = json.loads(p.read_text())
    plan['entries'][0]['target'] = '../outside'
    with pytest.raises(ValueError, match='escapes'):
        enumerate_plan(plan, tmp_path)
