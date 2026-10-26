"""Persisted route contracts derived only from the NExT-QA Luna IR."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping
ROUTE_CAUSAL_WHY = 'causal_why'
ROUTE_CAUSAL_HOW = 'causal_how'
ROUTE_TEMPORAL_PREVIOUS = 'temporal_previous'
ROUTE_TEMPORAL_NEXT = 'temporal_next'
ROUTE_TEMPORAL_CONCURRENT = 'temporal_concurrent'
ROUTE_DESCRIPTIVE_COUNT = 'descriptive_count'
ROUTE_DESCRIPTIVE_LOCATION = 'descriptive_location'
ROUTE_DESCRIPTIVE_DIRECT = 'descriptive_direct'
ROUTE_IDS = frozenset({ROUTE_CAUSAL_WHY, ROUTE_CAUSAL_HOW, ROUTE_TEMPORAL_PREVIOUS, ROUTE_TEMPORAL_NEXT, ROUTE_TEMPORAL_CONCURRENT, ROUTE_DESCRIPTIVE_COUNT, ROUTE_DESCRIPTIVE_LOCATION, ROUTE_DESCRIPTIVE_DIRECT})
INTENT_CONTRACTS: dict[str, dict[str, Any]] = {ROUTE_CAUSAL_WHY: {'visual_evidence_demand': 'preceding_context', 'answer_targets': frozenset({'reason'})}, ROUTE_CAUSAL_HOW: {'visual_evidence_demand': 'ordered_action_chain', 'answer_targets': frozenset({'action_chain'})}, ROUTE_TEMPORAL_PREVIOUS: {'visual_evidence_demand': 'preceding_event_window', 'answer_targets': frozenset({'preceding_event'})}, ROUTE_TEMPORAL_NEXT: {'visual_evidence_demand': 'following_event_window', 'answer_targets': frozenset({'following_event'})}, ROUTE_TEMPORAL_CONCURRENT: {'visual_evidence_demand': 'same_time_window', 'answer_targets': frozenset({'concurrent_event'})}, ROUTE_DESCRIPTIVE_COUNT: {'visual_evidence_demand': 'deduplicated_entities', 'answer_targets': frozenset({'count'})}, ROUTE_DESCRIPTIVE_LOCATION: {'visual_evidence_demand': 'spatial_context', 'answer_targets': frozenset({'location'})}, ROUTE_DESCRIPTIVE_DIRECT: {'visual_evidence_demand': 'direct_visual', 'answer_targets': frozenset({'object', 'action', 'state', 'attribute'})}}
ROUTING_INPUT_FIELDS = ('question_id', 'video_id', 'normalized_question', 'intent', 'answer_target', 'visual_evidence_demand', 'anchor_event', 'referenced_event', 'target_entities', 'actors', 'actions', 'objects', 'states')

def routing_input_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(record.get(field)) for field in ROUTING_INPUT_FIELDS}

def routing_input_sha256(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(routing_input_projection(record), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def validate_ir_route_contract(record: Mapping[str, Any]) -> tuple[str, str, str]:
    question_id = _required_text(record, 'question_id')
    _required_text(record, 'video_id')
    if str(record.get('status') or 'completed') != 'completed':
        raise ValueError(f'question IR is not completed: {question_id}')
    if record.get('uses_gold_answer') is not False:
        raise ValueError(f'question IR gold purity failed: {question_id}')
    intent = _required_text(record, 'intent').lower()
    if intent not in INTENT_CONTRACTS:
        raise ValueError(f'unsupported NExT-QA Luna intent: {intent!r}')
    answer_target = _required_text(record, 'answer_target').lower()
    visual_demand = _required_text(record, 'visual_evidence_demand').lower()
    contract = INTENT_CONTRACTS[intent]
    if visual_demand != contract['visual_evidence_demand']:
        raise ValueError(f'visual_evidence_demand is inconsistent with intent for {question_id}')
    if answer_target not in contract['answer_targets']:
        raise ValueError(f'answer_target is inconsistent with intent for {question_id}')
    return (intent, answer_target, visual_demand)

def compile_route_record(record: Mapping[str, Any]) -> dict[str, Any]:
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.evidence_demand import compile_evidence_demand
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.policy_registry import get_route_policy
    intent, answer_target, visual_demand = validate_ir_route_contract(record)
    policy = get_route_policy(intent)
    demand = compile_evidence_demand(record, policy=policy)
    return {'schema_version': 'nextqa_visual_route_record_v2', 'question_id': _required_text(record, 'question_id'), 'video_id': _required_text(record, 'video_id'), 'route_id': intent, 'route_source': 'nextqa_luna_ir_intent_only', 'handler_id': policy.handler_id, 'answer_target': answer_target, 'visual_evidence_demand': visual_demand, 'evidence_demand': demand, 'question_ir_sha256': routing_input_sha256(record), 'uses_question_text_routing': False, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False}

def compile_route_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    routes = [compile_route_record(record) for record in records]
    question_ids = [str(record['question_id']) for record in routes]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError('route manifest contains duplicate question IDs')
    return routes

def audit_forbidden_field_isolation(record: Mapping[str, Any]) -> dict[str, bool]:
    baseline = compile_route_record(record)
    checks: dict[str, bool] = {}
    for key, value in (('official_type', '__PERTURBED_TYPE__'), ('answer_choices', [f'perturbed-{index}' for index in range(5)]), ('gold_choice_index', 4), ('gold_answer', '__PERTURBED_GOLD__')):
        changed = deepcopy(dict(record))
        changed[key] = value
        checks[f'{key}_invariant'] = compile_route_record(changed) == baseline
    return checks

def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = str(record.get(key) or '').strip()
    if not value:
        raise ValueError(f'question IR is missing {key}')
    return value
