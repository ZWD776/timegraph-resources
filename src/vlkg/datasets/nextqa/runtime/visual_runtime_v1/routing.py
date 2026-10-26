"""Question-only evidence demand and visual routing for NExT-QA."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping
ROUTE_CAUSAL_WHY = 'causal_why'
ROUTE_CAUSAL_HOW = 'causal_how'
ROUTE_TEMPORAL_PREVIOUS = 'temporal_previous'
ROUTE_TEMPORAL_NEXT = 'temporal_next'
ROUTE_TEMPORAL_CONCURRENT = 'temporal_concurrent'
ROUTE_DESCRIPTIVE_COUNT = 'descriptive_count'
ROUTE_DESCRIPTIVE_LOCATION = 'descriptive_location'
ROUTE_DESCRIPTIVE_OBJECT_ACTION = 'descriptive_object_or_action'
ROUTE_IDS = frozenset({ROUTE_CAUSAL_WHY, ROUTE_CAUSAL_HOW, ROUTE_TEMPORAL_PREVIOUS, ROUTE_TEMPORAL_NEXT, ROUTE_TEMPORAL_CONCURRENT, ROUTE_DESCRIPTIVE_COUNT, ROUTE_DESCRIPTIVE_LOCATION, ROUTE_DESCRIPTIVE_OBJECT_ACTION})
QUESTION_ONLY_FIELDS = ('question_id', 'video_id', 'normalized_question', 'target_entities', 'actors', 'actions', 'objects', 'temporal_direction', 'temporal_anchor', 'referenced_event', 'causal_query', 'count_query', 'location_query', 'attribute_query', 'answer_format')
_PREVIOUS_RE = re.compile('\\b(before|previous|previously|prior|earlier)\\b')
_NEXT_RE = re.compile('\\b(after|following|later|then)\\b|\\bnext\\b(?!\\s+to\\b)')
_CONCURRENT_RE = re.compile('\\b(while|during|concurrent|simultaneous|simultaneously)\\b')

@dataclass(frozen=True)
class EvidenceDemand:
    route_id: str
    handler_id: str
    evidence_roles: tuple[str, ...]
    candidate_kinds: tuple[str, ...]
    temporal_policy: str
    target_terms: tuple[str, ...]
    query_text: str
    question_ir_sha256: str
    contract: str = 'nextqa_evidence_demand_v1'
    uses_official_type: bool = False
    uses_answer_choices: bool = False
    uses_gold_answer: bool = False
    changes_graph_retrieval: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['evidence_roles'] = list(self.evidence_roles)
        payload['candidate_kinds'] = list(self.candidate_kinds)
        payload['target_terms'] = list(self.target_terms)
        return payload

def question_only_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Discard type, choices, labels, and any non-semantic runtime fields."""
    return {field: deepcopy(record.get(field)) for field in QUESTION_ONLY_FIELDS}

def question_only_fingerprint(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(question_only_projection(record), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def compile_evidence_demand(record: Mapping[str, Any]) -> EvidenceDemand:
    projected = question_only_projection(record)
    question_id = _required_text(projected, 'question_id')
    _required_text(projected, 'video_id')
    question = _required_text(projected, 'normalized_question').lower()
    causal = str(projected.get('causal_query') or 'none').lower()
    direction = str(projected.get('temporal_direction') or 'none').lower()
    if causal == 'why' or question.startswith('why '):
        route_id = ROUTE_CAUSAL_WHY
    elif bool(projected.get('count_query')) or question.startswith('how many '):
        route_id = ROUTE_DESCRIPTIVE_COUNT
    elif bool(projected.get('location_query')) or question.startswith('where '):
        route_id = ROUTE_DESCRIPTIVE_LOCATION
    elif direction == 'before':
        route_id = ROUTE_TEMPORAL_PREVIOUS
    elif direction == 'after':
        route_id = ROUTE_TEMPORAL_NEXT
    elif direction == 'concurrent':
        route_id = ROUTE_TEMPORAL_CONCURRENT
    elif _PREVIOUS_RE.search(question):
        route_id = ROUTE_TEMPORAL_PREVIOUS
    elif _NEXT_RE.search(question):
        route_id = ROUTE_TEMPORAL_NEXT
    elif _CONCURRENT_RE.search(question):
        route_id = ROUTE_TEMPORAL_CONCURRENT
    elif causal == 'how' or (question.startswith('how ') and (not question.startswith('how many '))):
        route_id = ROUTE_CAUSAL_HOW
    else:
        route_id = ROUTE_DESCRIPTIVE_OBJECT_ACTION
    roles, kinds, temporal_policy = _route_contract(route_id)
    target_terms = _dedupe_strings([*list(projected.get('actors') or []), *list(projected.get('actions') or []), *list(projected.get('objects') or []), *list(projected.get('target_entities') or [])])
    query_text = _visual_query_text(question, target_terms, projected)
    demand = EvidenceDemand(route_id=route_id, handler_id=f'nextqa_visual_handler_v1:{route_id}', evidence_roles=roles, candidate_kinds=kinds, temporal_policy=temporal_policy, target_terms=target_terms, query_text=query_text, question_ir_sha256=question_only_fingerprint(record))
    if demand.route_id not in ROUTE_IDS:
        raise AssertionError(f'unsupported route for {question_id}: {demand.route_id}')
    return demand

def compile_route_record(record: Mapping[str, Any]) -> dict[str, Any]:
    demand = compile_evidence_demand(record)
    return {'schema_version': 'nextqa_visual_route_record_v1', 'question_id': _required_text(record, 'question_id'), 'video_id': _required_text(record, 'video_id'), 'evidence_demand': demand.to_dict(), 'route_id': demand.route_id, 'handler_id': demand.handler_id, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False}

def compile_route_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    routes = [compile_route_record(record) for record in records]
    question_ids = [str(record['question_id']) for record in routes]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError('route manifest contains duplicate question IDs')
    return routes

def audit_route_field_isolation(record: Mapping[str, Any]) -> dict[str, bool]:
    baseline = compile_route_record(record)
    checks: dict[str, bool] = {}
    for key, value in (('official_type', '__PERTURBED_TYPE__'), ('answer_choices', [f'perturbed-{index}' for index in range(5)]), ('gold_choice_index', 4), ('gold_answer', '__PERTURBED_GOLD__')):
        changed = deepcopy(dict(record))
        changed[key] = value
        checks[f'{key}_invariant'] = compile_route_record(changed) == baseline
    return checks

def _route_contract(route_id: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    contracts = {ROUTE_CAUSAL_WHY: (('effect_event', 'preceding_action_or_state', 'actor_object_interaction'), ('event_boundary', 'before_after_pair', 'temporal_window', 'object', 'tracklet'), 'effect_boundary_with_preceding_context'), ROUTE_CAUSAL_HOW: (('ordered_action_chain', 'mechanism_object', 'actor_object_interaction'), ('temporal_window', 'object', 'tracklet', 'frame'), 'ordered_mechanism_window'), ROUTE_TEMPORAL_PREVIOUS: (('target_event_boundary', 'preceding_window'), ('before_after_pair', 'event_boundary', 'temporal_window', 'frame'), 'preceding_side_of_target_boundary'), ROUTE_TEMPORAL_NEXT: (('target_event_boundary', 'following_window'), ('before_after_pair', 'event_boundary', 'temporal_window', 'frame'), 'following_side_of_target_boundary'), ROUTE_TEMPORAL_CONCURRENT: (('same_window_actor_object_relations',), ('temporal_window', 'frame', 'object', 'tracklet'), 'same_time_window'), ROUTE_DESCRIPTIVE_COUNT: (('temporally_deduplicated_entity_set',), ('tracklet', 'object'), 'tracklet_deduplicated_set'), ROUTE_DESCRIPTIVE_LOCATION: (('target_entity', 'spatial_anchor', 'scene_context'), ('object', 'frame', 'tracklet'), 'target_with_spatial_context'), ROUTE_DESCRIPTIVE_OBJECT_ACTION: (('direct_object_or_action',), ('object', 'frame', 'tracklet', 'temporal_window'), 'direct_visual_evidence')}
    return contracts[route_id]

def _visual_query_text(question: str, target_terms: tuple[str, ...], projected: Mapping[str, Any]) -> str:
    parts = [question]
    referenced_event = str(projected.get('referenced_event') or '').strip()
    if referenced_event:
        parts.append(f'referenced event: {referenced_event}')
    if target_terms:
        parts.append('visual targets: ' + ', '.join(target_terms))
    return '. '.join(parts)

def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = str(record.get(key) or '').strip()
    if not value:
        raise ValueError(f'question IR is missing {key}')
    return value

def _dedupe_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((str(value).strip().lower() for value in values if str(value).strip())))
