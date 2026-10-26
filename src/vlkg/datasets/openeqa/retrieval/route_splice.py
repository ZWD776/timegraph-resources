"""Strict question-ID replacement for isolated OpenEQA route experiments."""
from __future__ import annotations
from typing import Any
ROUTE_SPLICE_CONTRACT = 'openeqa_route_result_splice.v1'

def manifest_question_ids(manifest: dict[str, Any], *, data_source: str, strategy_variants: list[str] | None=None) -> list[str]:
    requested_variants = {str(value).strip() for value in strategy_variants or [] if str(value).strip()}
    result: list[str] = []
    for row in manifest.get('records', []) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get('data_source') or '') != data_source:
            continue
        variant = str(row.get('strategy_variant') or '').strip()
        if requested_variants and variant not in requested_variants:
            continue
        question_id = str(row.get('question_id') or '').strip()
        if question_id:
            result.append(question_id)
    return result

def splice_route_records(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *, expected_question_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = _index_unique(baseline, label='baseline')
    candidate_by_id = _index_unique(candidate, label='candidate')
    expected = set(expected_question_ids)
    if len(expected) != len(expected_question_ids):
        raise ValueError('route manifest contains duplicate question IDs')
    missing_from_baseline = sorted(expected - set(baseline_by_id))
    if missing_from_baseline:
        raise ValueError(f'baseline is missing {len(missing_from_baseline)} target IDs; sample={missing_from_baseline[:5]}')
    missing_candidate = sorted(expected - set(candidate_by_id))
    extra_candidate = sorted(set(candidate_by_id) - expected)
    if missing_candidate or extra_candidate:
        raise ValueError(f'candidate IDs must exactly equal the route manifest: missing={missing_candidate[:5]} extra={extra_candidate[:5]}')
    merged = [candidate_by_id.get(str(row.get('question_id') or ''), row) for row in baseline]
    merged_by_id = _index_unique(merged, label='merged')
    if list(merged_by_id) != list(baseline_by_id):
        raise AssertionError('merged question order differs from baseline')
    non_target_changed = [question_id for question_id, row in baseline_by_id.items() if question_id not in expected and merged_by_id[question_id] != row]
    if non_target_changed:
        raise AssertionError(f'non-target records changed; sample={non_target_changed[:5]}')
    return (merged, {'contract': ROUTE_SPLICE_CONTRACT, 'baseline_count': len(baseline), 'candidate_count': len(candidate), 'target_question_count': len(expected), 'merged_count': len(merged), 'non_target_question_count': len(baseline) - len(expected), 'non_target_changed_count': 0, 'missing_target_count': 0, 'extra_candidate_count': 0, 'question_order_preserved': True})

def _index_unique(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{label} contains a non-object record')
        question_id = str(row.get('question_id') or '').strip()
        if not question_id:
            raise ValueError(f'{label} contains a record without question_id')
        if question_id in result:
            raise ValueError(f'{label} contains duplicate question_id={question_id}')
        result[question_id] = row
    return result
__all__ = ['ROUTE_SPLICE_CONTRACT', 'manifest_question_ids', 'splice_route_records']
