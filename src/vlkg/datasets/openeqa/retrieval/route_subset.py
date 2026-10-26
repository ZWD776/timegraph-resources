"""Validation helpers for immutable OpenEQA EvidenceDemand subsets."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROUTE_SUBSET_CONTRACT = 'openeqa_evidence_need_question_subset.v1'
ROUTE_CHANGED_SUBSET_CONTRACT = 'openeqa_ir_first_route_change_subset.v1'
ROUTE_SUBSET_CONTRACTS = {ROUTE_SUBSET_CONTRACT, ROUTE_CHANGED_SUBSET_CONTRACT}

def load_route_subset_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError(f'Route subset manifest must be an object: {path}')
    return payload

def validate_route_subset_manifest(payload: dict[str, Any], *, data_source: str, question_ir_records: dict[str, dict[str, Any]], strategy_variants: list[str] | None=None) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    contract = str(payload.get('contract') or '')
    if contract not in ROUTE_SUBSET_CONTRACTS:
        issues.append(f'route subset contract must be one of {sorted(ROUTE_SUBSET_CONTRACTS)!r}')
    changed_union = contract == ROUTE_CHANGED_SUBSET_CONTRACT
    primary_need = str(payload.get('primary_need') or '').strip()
    if not changed_union and (not primary_need):
        issues.append('route subset primary_need is missing')
    if changed_union and payload.get('selection_contract') != 'primary_need_or_strategy_variant_or_handler_changed':
        issues.append('changed route subset selection contract is invalid')
    if payload.get('uses_gold_answer') is not False:
        issues.append('route subset must declare uses_gold_answer=false')
    requested_variants = {str(value).strip() for value in strategy_variants or [] if str(value).strip()}
    known_variants = {str(value).strip() for value in payload.get('strategy_variant_counts') or {} if str(value).strip()}
    unknown_variants = requested_variants - known_variants
    if unknown_variants:
        issues.append(f'unknown strategy variants for {primary_need}: {sorted(unknown_variants)}')
    records = payload.get('records', [])
    if not isinstance(records, list):
        return ([], [*issues, 'route subset records must be a list'])
    selected_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            issues.append('route subset contains a non-object record')
            continue
        if str(record.get('data_source') or '') != data_source:
            continue
        question_id = str(record.get('question_id') or '').strip()
        variant = str(record.get('strategy_variant') or '').strip()
        if requested_variants and variant not in requested_variants:
            continue
        if not question_id:
            issues.append('route subset contains a record without question_id')
            continue
        if question_id in seen:
            issues.append(f'duplicate question_id in route subset: {question_id}')
            continue
        seen.add(question_id)
        question_ir = question_ir_records.get(question_id)
        route = question_ir.get('question_route') if isinstance(question_ir, dict) else None
        if not isinstance(route, dict):
            issues.append(f'{question_id}: persisted question_route is missing')
            continue
        actual_need = str(route.get('primary_need') or '').strip()
        actual_route_id = str(route.get('route_id') or '').strip()
        actual_variant = str(route.get('strategy_variant') or '').strip()
        actual_handler = str(route.get('route_strategy_handler') or '').strip()
        expected_need = str(record.get('primary_need') or '').strip() if changed_union else primary_need
        expected_route_id = str(record.get('route_id') or '').strip()
        expected_handler = str(record.get('route_strategy_handler') or '').strip()
        if actual_need != expected_need:
            issues.append(f'{question_id}: manifest need={expected_need!r}, persisted need={actual_need!r}')
            continue
        if changed_union and actual_route_id != expected_route_id:
            issues.append(f'{question_id}: manifest route_id={expected_route_id!r}, persisted route_id={actual_route_id!r}')
            continue
        if actual_variant != variant:
            issues.append(f'{question_id}: manifest variant={variant!r}, persisted variant={actual_variant!r}')
            continue
        if changed_union and actual_handler != expected_handler:
            issues.append(f'{question_id}: manifest handler={expected_handler!r}, persisted handler={actual_handler!r}')
            continue
        selected_ids.append(question_id)
    expected_dataset_count = int((payload.get('dataset_counts') or {}).get(data_source) or 0)
    if not requested_variants and len(selected_ids) != expected_dataset_count:
        issues.append(f'route subset selected {len(selected_ids)}/{expected_dataset_count} records for {data_source}')
    if not selected_ids:
        issues.append(f'route subset has no selected records for {data_source}')
    return (selected_ids, issues)
__all__ = ['ROUTE_CHANGED_SUBSET_CONTRACT', 'ROUTE_SUBSET_CONTRACT', 'ROUTE_SUBSET_CONTRACTS', 'load_route_subset_manifest', 'validate_route_subset_manifest']
