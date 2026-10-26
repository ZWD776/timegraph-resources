"""Compile role-specific visual query intents from validated Luna IR fields."""
from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.policy_registry import RoutePolicy

def compile_evidence_demand(record: Mapping[str, Any], *, policy: RoutePolicy) -> dict[str, Any]:
    question = str(record.get('normalized_question') or '').strip()
    if not question:
        raise ValueError('question IR is missing normalized_question')
    anchor = record.get('anchor_event')
    anchor_text = str(anchor.get('text') or '').strip() if isinstance(anchor, Mapping) else ''
    referenced_event = str(record.get('referenced_event') or '').strip()
    target_terms = _dedupe([*list(record.get('target_entities') or []), *list(record.get('actors') or []), *list(record.get('actions') or []), *list(record.get('objects') or []), *list(record.get('states') or [])])
    event_text = anchor_text or referenced_event
    answer_target = str(record.get('answer_target') or '').strip().lower()
    query_intents = []
    for role in policy.roles:
        parts = [question, f"evidence role: {role.role_id.replace('_', ' ')}"]
        if event_text:
            parts.append(f'anchor event: {event_text}')
        if target_terms:
            parts.append('visual targets: ' + ', '.join(target_terms))
        if policy.route_id == 'descriptive_direct':
            parts.append(f'answer target: {answer_target}')
        query_intents.append({'role_id': role.role_id, 'query_text': '. '.join(parts), 'candidate_kinds': [kind for kind, _quota in role.kind_quotas], 'kind_quotas': {kind: quota for kind, quota in role.kind_quotas}, 'search_top_k': role.search_top_k})
    return {'schema_version': 'nextqa_evidence_demand_v2', 'route_id': policy.route_id, 'handler_id': policy.handler_id, 'temporal_policy': policy.temporal_policy, 'answer_target': answer_target, 'target_terms': list(target_terms), 'query_intents': query_intents, 'evidence_limit': policy.evidence_limit, 'route_source': 'nextqa_luna_ir_intent_only', 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False}

def _dedupe(values: list[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((str(value).strip().lower() for value in values if str(value).strip())))
