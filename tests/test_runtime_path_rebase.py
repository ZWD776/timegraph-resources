import json
from pathlib import Path

import pytest

from scripts.datasets.rebase_nextqa_runtime import Relocator, derive_json


def test_only_whole_path_strings_are_relocated(tmp_path):
    mapper = Relocator({'.scratch/nextqa/graphs/main': 'datasets/nextqa/graphs/main/payload'}, tmp_path)
    original = {'id': 'entity:1', 'score': 0.375, 'image_path': str(tmp_path / '.scratch/nextqa/graphs/main/frame.png'),
                'note': 'see .scratch/nextqa/graphs/main/frame.png', 'state': [True, None, 3],
                'other': '.scratch/nextqa/graphs/main_extra/frame.png'}
    transformed = mapper.transform(json.loads(json.dumps(original)))
    assert transformed['id'] == original['id']
    assert transformed['score'] == original['score']
    assert transformed['state'] == original['state']
    assert transformed['note'] == original['note']
    assert transformed['other'] == original['other']
    assert transformed['image_path'] == tmp_path.as_posix()+'/datasets/nextqa/graphs/main/payload/frame.png'
    assert mapper.changed_fields == {'image_path': 1}


def test_derived_file_preserves_source_and_is_idempotent(tmp_path):
    p = tmp_path / 'original.json';p.write_text('{"path":".scratch/nextqa/raw/a","n":1}')
    original = p.read_bytes();target = tmp_path / 'view.json'
    m = {'.scratch/nextqa/raw': 'datasets/nextqa/source/raw'}
    row = derive_json(p,target,Relocator(m,tmp_path))
    assert p.read_bytes()==original
    assert derive_json(p,target,Relocator(m,tmp_path))==row
    assert json.loads(target.read_text())['n']==1


def test_different_destination_is_not_overwritten(tmp_path):
    p=tmp_path/'source.json';p.write_text('{"n":1}')
    target=tmp_path/'view.json';target.write_text('{"n":2}')
    with pytest.raises(FileExistsError):
        derive_json(p,target,Relocator({},tmp_path))
    assert json.loads(target.read_text())['n']==2


def test_cache_relocation_preserves_vector_blob(tmp_path):
    import sqlite3
    from scripts.datasets.rebase_nextqa_embedding_cache import migrate
    source=tmp_path/'old.sqlite';target=tmp_path/'new.sqlite'
    image=tmp_path/'datasets/nextqa/frames/a.png';image.parent.mkdir(parents=True);image.write_bytes(b'image')
    old=str(tmp_path/'.scratch/nextqa/frames/a.png')
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE embedding_cache(image_path TEXT,bbox_key TEXT,embedding BLOB,dim INTEGER,PRIMARY KEY(image_path,bbox_key))')
        db.execute('INSERT INTO embedding_cache VALUES(?,?,?,?)',(old,'box',b'\x01\x02\x03\x04',1))
    result=migrate(source,target,Relocator({'.scratch/nextqa/frames':'datasets/nextqa/frames'},tmp_path))
    assert result['rows']==1 and result['embedding_bytes_and_dimensions_unchanged']
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT image_path,embedding FROM embedding_cache').fetchone()==(str(image),b'\x01\x02\x03\x04')
    with sqlite3.connect(source) as db:
        assert db.execute('SELECT image_path FROM embedding_cache').fetchone()[0]==old


def test_current_runtime_views_have_verified_resources():
    from vlkg.core.assets.asset_registry import asset_file
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'archive/legacy_20260928/governance/registry/migrations/nextqa_runtime_path_rebase.json').read_text(encoding='utf-8'))
    assert report['status']=='passed'
    assert report['missing_targets']==[] and report['unmapped_old_paths']=={}
    assert report['derived_json_files']==5700
    graph=asset_file('nextqa','graphs','manifest')
    index=asset_file('nextqa','indexes','manifest')
    active=json.loads((root/'archive/legacy_20260928/governance/registry/migrations/nextqa_layer_ownership_activation.json').read_text(encoding='utf-8'))
    assert graph==root/active['graph_manifest']
    assert index==root/active['index_manifest']
    assert asset_file('nextqa','cache','sqlite').is_file()
    assert asset_file('nextqa','cache','sqlite').is_relative_to(root/'datasets/nextqa/cache')


def test_projection_and_qa_text_parity_evidence():
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'archive/legacy_20260928/governance/registry/migrations/nextqa_runtime_path_parity.json').read_text(encoding='utf-8'))
    assert report['status']=='passed'
    assert report['projection_records_checked']>0
    assert report['sampled_graph_vector_text_lexical_tokens_evidence_equal']
    assert len(report['query_checks'])==18
    assert all(r['qa_safe_evidence_equal'] and r['rendered_text_equal'] for r in report['query_checks'])


def test_old_activation_cannot_restore_superseded_pointers():
    from scripts.datasets.activate_nextqa_mainline import main
    with pytest.raises(RuntimeError, match='superseded'):
        main()
