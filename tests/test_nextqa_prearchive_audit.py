from pathlib import Path
import pytest

from scripts.datasets.audit_nextqa_execution import AccessGuard


def test_old_data_paths_blocked_but_sibling_prefix_allowed(tmp_path):
    guard=AccessGuard(tmp_path)
    for relative in ('.scratch/nextqa/a.json','data/next-qa/a.tar.gz'):
        with pytest.raises(PermissionError):
            guard.check_path(tmp_path/relative,'open')
    guard.check_path(tmp_path/'.scratch/nextqa_other/a.json','open')
    guard.check_path(tmp_path/'datasets/nextqa/source/a.json','open')
    assert len(guard.denied)==2


def test_sqlite_uri_cannot_bypass_guard(tmp_path):
    guard=AccessGuard(tmp_path)
    uri='file:'+(tmp_path/'.scratch/nextqa/cache.sqlite').as_posix()+'?mode=ro'
    with pytest.raises(PermissionError):
        guard.hook('sqlite3.connect',(uri,))


def test_network_forbidden_and_recorded(tmp_path):
    guard=AccessGuard(tmp_path)
    with pytest.raises(PermissionError):
        guard.hook('socket.connect',(None,None))
    assert guard.network_attempts==1


def test_retained_dependency_overrides_verified_copy():
    from scripts.datasets.audit_nextqa_prearchive import classify
    copies={'.scratch/nextqa/graphs/main':'datasets/nextqa/graphs/main'}
    assert classify('.scratch/nextqa/graphs/main',copies,['.scratch/nextqa/graphs/main'])=='hold_rebuild_or_recipe_dependency'
    assert classify('.scratch/nextqa/graphs/main',copies,[])=='archive_candidate_verified_copy'
    assert classify('.scratch/nextqa/graphs/main_extra',copies,[])=='review_nonselected_asset'


def test_hold_inside_parent_blocks_whole_parent_archive():
    from scripts.datasets.audit_nextqa_prearchive import classify
    assert classify('.scratch/nextqa/graphs',{},['.scratch/nextqa/graphs/main'])=='hold_rebuild_or_recipe_dependency'


def test_persisted_gate_preserves_measured_archive_blockers():
    import json
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'archive/legacy_20260928/governance/registry/audits/nextqa_prearchive/summary.json').read_text(encoding='utf-8'))
    gate=json.loads((root/'archive/legacy_20260928/governance/registry/gates/nextqa_prearchive_gate.json').read_text(encoding='utf-8'))
    assert gate['blanket_archive_allowed'] is False and gate['deletion_allowed'] is False
    assert report['current_inference_guarded_cases']=={
        'replay':'passed','resources':'passed','graph_mock':'passed',
        'index_defaults':'passed','pair_defaults':'passed','causal_inputs':'passed'}
    assert gate['hold_roots']==report['hold_roots']
    assert len(report['historical_fingerprint_mismatches'])==2
    assets=json.loads((root/'archive/legacy_20260928/governance/registry/audits/nextqa_prearchive/asset_inventory.json').read_text(encoding='utf-8'))
    assert all(row['may_delete'] is False for row in assets)
    assert report['hold_roots']==[]
    assert report['observed_third_party_source_files']==[]
    for case in report['current_inference_guarded_cases']:
        execution=json.loads((root/f'archive/legacy_20260928/governance/registry/audits/nextqa_prearchive/execution_{case}.json').read_text(encoding='utf-8'))
        assert execution['graphiti_source_denied'] is True
    assert gate['unclassified_asset_units']==0
    assert gate['shared_code_retirement_allowed'] is False


def test_archive_plan_rejects_unreviewed_roots_and_path_escape():
    from scripts.datasets.plan_nextqa_archive import archive_target
    for source in ('.scratch/nextqa_other/a', '.scratch/nextqa/../navqa/a',
                   '.scratch/nextqa//a', 'data/next-qa/../other/a', '/absolute/a'):
        with pytest.raises(ValueError):
            archive_target(source)
    assert archive_target('.scratch\\nextqa\\qa\\old')=='archive/legacy_20260928/nextqa/assets/scratch/nextqa/qa/old'


def test_archive_proposal_is_complete_and_non_executable():
    import json
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'archive/legacy_20260928/governance/registry/migrations/nextqa_archive_disposition.json').read_text(encoding='utf-8'))
    inventory=json.loads((root/'archive/legacy_20260928/governance/registry/audits/nextqa_prearchive/asset_inventory.json').read_text(encoding='utf-8'))
    assert {r['path'] for r in plan['assets']}=={r['path'] for r in inventory}
    assert plan['unclassified_units']==0 and plan['move_allowed'] is False and plan['deletion_allowed'] is False
    targets=[r['archive_target'] for r in plan['assets']]
    assert len(targets)==len(set(targets))
    assert all(r['move_allowed'] is False and r['may_delete'] is False for r in plan['assets'])
    assert all(r['reason_cn'] and r['category'] for r in plan['assets'])
    assert all(r['rewrite_or_move_allowed'] is False for r in plan['references'])


def test_final_c_recomposition_is_exact():
    import json
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'archive/legacy_20260928/governance/registry/migrations/nextqa_final_c_recomposition.json').read_text(encoding='utf-8'))
    assert report['status']=='passed'
    assert report['question_count']==4996
    assert report['prediction_objects_exactly_equal']
    assert report['new_sha256']==report['frozen_sha256']
