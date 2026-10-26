from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import pytest
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.strategies import merge_route_prediction_overlay
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain_v2 import HANDLER_ID, STRATEGY_ID, ROUTE_ID as TARGET_ROUTE_ID
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.strategies.strategy_identity import validate_route_strategy_identity, validate_strategy_evidence_manifest
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())

def test_route_strategy_identity_accepts_causal_why_v2() -> None:
    assert validate_route_strategy_identity('causal_why_coherent_chain_v2', 'causal_why', 'nextqa_visual_handler_v4:causal_why_coherent_chain') == 'causal_why_coherent_chain_v2'

def _route(question_id: str='nextqa-val/v1/q1') -> dict[str, object]:
    return {'question_id': question_id, 'video_id': 'v1', 'route_id': TARGET_ROUTE_ID, 'route_source': 'nextqa_luna_ir_intent_only', 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False}

def _prediction(question_id: str, route_id: str, graph_ids: list[str]) -> dict[str, object]:
    return {'question_id': question_id, 'video_id': 'v1', 'route_id': route_id, 'handler_id': f'nextqa_visual_handler_v2:{route_id}', 'status': 'completed', 'predicted_choice_index': 1, 'graph_record_ids': graph_ids, 'uses_official_type': False, 'uses_gold_answer': False}

def test_route_overlay_replaces_exact_target_and_preserves_other_records() -> None:
    base = [_prediction('q1', TARGET_ROUTE_ID, ['g1']), _prediction('q2', 'descriptive_direct', ['g2']), _prediction('q3', TARGET_ROUTE_ID, ['g3'])]
    frozen_non_target = deepcopy(base[1])
    overlay = []
    for row in (base[0], base[2]):
        replacement = deepcopy(row)
        replacement.update(handler_id=HANDLER_ID, strategy_id=STRATEGY_ID, strategy_scope='route_wide', predicted_choice_index=2)
        overlay.append(replacement)
    routes = [{**_route('q1')}, {**_route('q2'), 'route_id': 'descriptive_direct'}, {**_route('q3')}]
    merged, audit, provenance = merge_route_prediction_overlay(base, overlay, routes, target_route_id=TARGET_ROUTE_ID, strategy_id=STRATEGY_ID, handler_id=HANDLER_ID, expected_question_count=3)
    assert merged[1] == frozen_non_target
    assert audit['target_question_count'] == 2
    assert audit['retained_base_question_count'] == 1
    assert audit['non_target_predictions_unchanged'] is True
    assert [row['prediction_source'] for row in provenance] == ['route_strategy_overlay', 'frozen_visual_v2', 'route_strategy_overlay']

def test_route_overlay_rejects_partial_target_or_changed_graph_records() -> None:
    base = [_prediction('q1', TARGET_ROUTE_ID, ['g1']), _prediction('q2', TARGET_ROUTE_ID, ['g2'])]
    routes = [_route('q1'), _route('q2')]
    partial = deepcopy(base[0])
    partial.update(handler_id=HANDLER_ID, strategy_id=STRATEGY_ID, strategy_scope='route_wide')
    with pytest.raises(ValueError, match='exactly cover'):
        merge_route_prediction_overlay(base, [partial], routes, target_route_id=TARGET_ROUTE_ID, strategy_id=STRATEGY_ID, handler_id=HANDLER_ID, expected_question_count=2)
    changed = []
    for row in base:
        replacement = deepcopy(row)
        replacement.update(handler_id=HANDLER_ID, strategy_id=STRATEGY_ID, strategy_scope='route_wide')
        changed.append(replacement)
    changed[0]['graph_record_ids'] = ['changed']
    with pytest.raises(ValueError, match='graph record IDs'):
        merge_route_prediction_overlay(base, changed, routes, target_route_id=TARGET_ROUTE_ID, strategy_id=STRATEGY_ID, handler_id=HANDLER_ID, expected_question_count=2)


@pytest.mark.parametrize('field', ('strategy_id', 'target_route_id', 'handler_id', 'schema_version'))
def test_selected_strategy_rejects_changed_identity_or_schema(field):
    manifest = {'strategy_id': STRATEGY_ID, 'target_route_id': TARGET_ROUTE_ID,
                'handler_id': HANDLER_ID, 'schema_version': 'nextqa_visual_route_strategy_evidence_manifest_v1'}
    manifest[field] = 'unsupported'
    with pytest.raises(ValueError):
        validate_strategy_evidence_manifest(manifest)
