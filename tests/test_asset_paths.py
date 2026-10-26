import json
from pathlib import Path

import pytest

from vlkg.core.assets.asset_paths import translate_path, validate_mapping


def test_mapping_does_not_match_sibling_prefix(tmp_path):
    mapping = {'.scratch/navqa': 'datasets/navqa/source/payload'}
    assert translate_path('.scratch/navqa_cg6_local_tracklet/temporal', mapping, tmp_path) == '.scratch/navqa_cg6_local_tracklet/temporal'
    assert translate_path('.scratch/navqa/seq0', mapping, tmp_path) == 'datasets/navqa/source/payload/seq0'


def test_longest_component_prefix_wins(tmp_path):
    mapping = {'source': 'datasets/raw', 'source/nested': 'datasets/derived'}
    assert translate_path('source/nested/file.json', mapping, tmp_path) == 'datasets/derived/file.json'


def test_decoded_windows_absolute_path_preserves_root(tmp_path):
    value = (str(tmp_path) + '/.scratch/navqa/seq0/data.json').replace('/', '\\')
    result = translate_path(value, {'.scratch/navqa': 'datasets/navqa/source'}, tmp_path)
    assert result == tmp_path.resolve().as_posix() + '/datasets/navqa/source/seq0/data.json'


def test_other_workspace_is_untouched(tmp_path):
    assert translate_path('Z:/elsewhere/.scratch/navqa/a', {'.scratch/navqa': 'datasets/navqa'}, tmp_path) == 'Z:/elsewhere/.scratch/navqa/a'


@pytest.mark.parametrize('targets', [('dest', 'dest'), ('dest', 'dest/child')])
def test_overlapping_targets_rejected(targets):
    with pytest.raises(ValueError, match='overlap'):
        validate_mapping(dict(zip(('source_a', 'source_b'), targets)))


def test_project_mapping_has_distinct_target_roots():
    p = Path(__file__).resolve().parents[1] / 'archive/legacy_20260928/governance/registry/migrations/path_mapping_v2.json'
    validate_mapping(json.loads(p.read_text(encoding='utf-8'))['mapping'])
