"""Dataset-independent semantic Question IR contract for OpenEQA."""
from __future__ import annotations
from typing import Any, Mapping
SEMANTIC_QUESTION_IR_CONTRACT = 'openeqa_semantic_question_ir.v1'
SEMANTIC_QUESTION_IR_SOURCE_CONTRACT = 'openeqa_semantic_question_ir_source.v1'
SEMANTIC_QUESTION_IR_FIELDS = frozenset({'aggregation', 'answer_type', 'attribute_dimension', 'event_constraints', 'hard_constraints', 'operation', 'qa_focus', 'rendering', 'spatial_constraints', 'target_description', 'target_descriptors', 'target_entities', 'target_state', 'temporal_relation'})
_ROUTE_OR_TAXONOMY_FIELDS = frozenset({'category', 'evidence_demand', 'evidence_question_type', 'evidence_requirements', 'graph_information_needs', 'question_route', 'question_route_id', 'question_type', 'routing'})

def semantic_question_ir_projection(question_ir: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the only Question IR fields allowed to influence graph retrieval."""
    if not isinstance(question_ir, Mapping):
        return {}
    return {key: value for key, value in question_ir.items() if key in SEMANTIC_QUESTION_IR_FIELDS}

def semantic_source_payload_issues(payload: Mapping[str, Any]) -> list[str]:
    """Validate one gold-free, question-only semantic IR source shard."""
    issues: list[str] = []
    if payload.get('contract') != SEMANTIC_QUESTION_IR_SOURCE_CONTRACT:
        issues.append(f'contract must be {SEMANTIC_QUESTION_IR_SOURCE_CONTRACT}')
    if payload.get('status') != 'complete':
        issues.append('status must be complete')
    if payload.get('uses_gold_answer') is not False:
        issues.append('uses_gold_answer must be false')
    if payload.get('prompt_excludes_official_answers') is not True:
        issues.append('prompt_excludes_official_answers must be true')
    if payload.get('prompt_excludes_graph_evidence') is not True:
        issues.append('prompt_excludes_graph_evidence must be true')
    if payload.get('raw_question_only') is not True:
        issues.append('raw_question_only must be true')
    if payload.get('compiler_receives_open_eqa_category') is not False:
        issues.append('compiler_receives_open_eqa_category must be false')
    if payload.get('retains_route_or_taxonomy_fields') is not False:
        issues.append('retains_route_or_taxonomy_fields must be false')
    if not str(payload.get('domain_profile') or '').strip():
        issues.append('domain_profile is missing')
    if not isinstance(payload.get('batch_size'), int) or payload['batch_size'] < 1:
        issues.append('batch_size must be a positive integer')
    if not isinstance(payload.get('max_tokens'), int) or payload['max_tokens'] < 1:
        issues.append('max_tokens must be a positive integer')
    if not isinstance(payload.get('temperature'), (int, float)):
        issues.append('temperature must be numeric')
    data_source = str(payload.get('data_source') or '').strip()
    if not data_source:
        issues.append('data_source is missing')
    records = payload.get('records')
    if not isinstance(records, list) or not records:
        issues.append('records must be a non-empty list')
        return issues
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f'records[{index}] must be an object')
            continue
        question_ir = record.get('llm_question_ir') or record.get('question_ir')
        if not isinstance(question_ir, Mapping):
            issues.append(f'records[{index}] is missing question IR')
            continue
        populated = sorted((key for key in _ROUTE_OR_TAXONOMY_FIELDS if question_ir.get(key)))
        if populated:
            issues.append(f'records[{index}] retains route/taxonomy fields: {populated}')
    return issues

def semantic_source_signature(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Fields that must be identical across all OpenEQA semantic IR shards."""

    def value(key: str) -> str:
        raw = payload.get(key)
        return '' if raw is None else str(raw)
    return (value('schema_version'), value('provider'), value('model'), value('mode'), value('domain_profile'), value('batch_size'), value('max_tokens'), value('temperature'))
__all__ = ['SEMANTIC_QUESTION_IR_CONTRACT', 'SEMANTIC_QUESTION_IR_SOURCE_CONTRACT', 'SEMANTIC_QUESTION_IR_FIELDS', 'semantic_question_ir_projection', 'semantic_source_payload_issues', 'semantic_source_signature']
