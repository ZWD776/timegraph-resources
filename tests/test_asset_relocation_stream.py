import json
from pathlib import Path

from vlkg.core.assets.asset_relocation import PathPrefixRewriter


def test_path_rewrite_preserves_other_graph_data_and_prefix_siblings(tmp_path):
    data = {'id': 'entity_1', 'time': 1.25, 'score': 0.33,
            'image_path': '.scratch/selected/images/a.jpg',
            'absolute': str(tmp_path / '.scratch/selected/images/a.jpg'),
            'sibling': '.scratch/selected_extra/images/a.jpg',
            'evidence': ['a person holds a cup', 'time_fact_core']}
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(data, indent=2), encoding='utf-8')
    writer = PathPrefixRewriter({'.scratch/selected': 'datasets/navqa/source/main/payload'}, tmp_path)
    for chunk_size in (1, 7, 31, 4096):
        target = tmp_path / f'output-{chunk_size}.json'
        receipt = writer.stream(source, target, chunk_size=chunk_size)
        result = json.loads(target.read_text(encoding='utf-8'))
        assert result == {**data,
            'image_path': 'datasets/navqa/source/main/payload/images/a.jpg',
            'absolute': str(tmp_path / 'datasets/navqa/source/main/payload/images/a.jpg')}
        assert sum(receipt['replacements'].values()) == 2
        assert result['sibling'] == data['sibling']


def test_windows_escaped_and_raw_paths_are_rewritten(tmp_path):
    root = Path('D:/workspace')
    writer = PathPrefixRewriter({'old/input': 'datasets/openeqa/source/main/payload'}, root)
    assert writer.rewrite_text('old/input/frames/a.jpg') == 'datasets/openeqa/source/main/payload/frames/a.jpg'
    data = {'path': str(root / 'old/input/frames/a.jpg')}
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(data), encoding='utf-8')
    target = tmp_path / 'output.json'
    writer.stream(source, target, chunk_size=3)
    assert json.loads(target.read_text(encoding='utf-8'))['path'] == str(root / 'datasets/openeqa/source/main/payload/frames/a.jpg')
