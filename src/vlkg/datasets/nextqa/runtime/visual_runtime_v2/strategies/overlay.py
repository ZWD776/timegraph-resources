"""Deterministic route-wide prediction overlay contracts for NExT-QA."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Sequence

def _index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id or question_id in output:
            raise ValueError(f'{name} has an empty or duplicate question ID')
        output[question_id] = row
    return output

def merge_route_prediction_overlay(base_predictions: Sequence[Mapping[str, Any]], overlay_predictions: Sequence[Mapping[str, Any]], route_records: Sequence[Mapping[str, Any]], *, target_route_id: str, strategy_id: str, handler_id: str, expected_question_count: int=4996) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base = _index(base_predictions, 'base predictions')
    overlay = _index(overlay_predictions, 'overlay predictions')
    routes = _index(route_records, 'route records')
    if len(base) != expected_question_count or set(base) != set(routes):
        raise ValueError('base predictions and route manifest must cover the full fixed set')
    target_ids = {question_id for question_id, route in routes.items() if route.get('route_id') == target_route_id}
    if not target_ids or set(overlay) != target_ids:
        raise ValueError('overlay predictions must exactly cover the target route')
    if any((route.get('uses_official_type') is not False or route.get('uses_answer_choices') is not False or route.get('uses_gold_answer') is not False for route in routes.values())):
        raise ValueError('route manifest purity failed')
    merged: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    non_target_equal_count = 0
    graph_record_mismatch_ids = []
    for base_row in base_predictions:
        question_id = str(base_row['question_id'])
        route_id = str(routes[question_id]['route_id'])
        if str(base_row.get('route_id') or '') != route_id:
            raise ValueError(f'base prediction route mismatch: {question_id}')
        if question_id in target_ids:
            row = overlay[question_id]
            if row.get('route_id') != target_route_id:
                raise ValueError(f'overlay route mismatch: {question_id}')
            if row.get('strategy_id') != strategy_id or row.get('handler_id') != handler_id:
                raise ValueError(f'overlay strategy identity mismatch: {question_id}')
            if row.get('strategy_scope') != 'route_wide':
                raise ValueError(f'overlay is not route-wide: {question_id}')
            if row.get('uses_gold_answer') is not False or row.get('uses_official_type') is not False:
                raise ValueError(f'overlay prediction purity failed: {question_id}')
            if list(row.get('graph_record_ids', [])) != list(base_row.get('graph_record_ids', [])):
                graph_record_mismatch_ids.append(question_id)
            merged.append(deepcopy(dict(row)))
            source = 'route_strategy_overlay'
            assigned_strategy = strategy_id
        else:
            copied = deepcopy(dict(base_row))
            merged.append(copied)
            if copied != dict(base_row):
                raise AssertionError('non-target prediction changed while copying')
            non_target_equal_count += 1
            source = 'frozen_visual_v2'
            assigned_strategy = 'nextqa_visual_runtime_v2'
        provenance.append({'question_id': question_id, 'route_id': route_id, 'prediction_source': source, 'strategy_id': assigned_strategy})
    if graph_record_mismatch_ids:
        raise ValueError('overlay changed frozen graph record IDs')
    audit = {'schema_version': 'nextqa_route_strategy_overlay_audit_v1', 'question_count': len(merged), 'target_route_id': target_route_id, 'strategy_id': strategy_id, 'handler_id': handler_id, 'target_question_count': len(target_ids), 'overlay_question_count': len(overlay), 'retained_base_question_count': non_target_equal_count, 'non_target_predictions_unchanged': non_target_equal_count == len(base) - len(target_ids), 'target_route_exact_coverage': set(overlay) == target_ids, 'graph_record_ids_preserved': not graph_record_mismatch_ids, 'selects_better_answer_per_question': False, 'uses_official_type': False, 'uses_gold_answer': False, 'status': 'passed'}
    return (merged, audit, provenance)
