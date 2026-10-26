"""Persisted OpenEQA single-route contract for the IR-first V5 profile."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.taxonomy import ROUTE_IDS
ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5 = 'openeqa_ir_first_single_route_v5'
ROUTE_CONTRACT = 'openeqa_single_route_decision.v1'

def single_route_decision_from_ir(question_ir: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a valid persisted single route, or ``{}`` when invalid.

    V5 deliberately accepts only a route persisted by the semantic IR.  It
    never derives a route from question text, official category, or evidence
    fields at runtime.
    """
    if not isinstance(question_ir, Mapping):
        return {}
    route = question_ir.get('question_route')
    if not isinstance(route, Mapping):
        return {}
    route = dict(route)
    metadata = route.get('metadata')
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if route.get('contract') != ROUTE_CONTRACT:
        return {}
    if metadata.get('routing_profile') != ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5:
        return {}
    question_type = str(question_ir.get('question_type') or '').strip()
    route_id = str(route.get('route_id') or '').strip()
    if not question_type or question_type != route_id or route_id not in ROUTE_IDS:
        return {}
    if question_ir.get('uses_gold_answer') is not False:
        return {}
    if question_ir.get('route_decision_is_persisted') is not True:
        return {}
    if question_ir.get('route_decision_runtime_recompile_allowed') is not False:
        return {}
    if str(question_ir.get('question_route_id') or '') != route_id:
        return {}
    if route.get('uses_gold_answer') is not False:
        return {}
    if route.get('uses_official_question_category') is not False:
        return {}
    if route.get('runtime_route_recompile_allowed') is not False:
        return {}
    if route.get('route_source') != 'semantic_question_ir.question_type':
        return {}
    if not str(route.get('handler_id') or '').strip():
        return {}
    return route

def single_route_decision_issues(question_ir: Mapping[str, Any] | None) -> list[str]:
    if single_route_decision_from_ir(question_ir):
        return []
    return [f'question_ir.question_route must be one valid persisted OpenEQA single-route {ROUTE_CONTRACT} object']

def route_input_projection(question_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Project only fields that the persisted route contract audits."""
    route = question_ir.get('question_route')
    return {'question_type': question_ir.get('question_type'), 'question_route': deepcopy(route) if isinstance(route, Mapping) else None}
__all__ = ['ROUTE_CONTRACT', 'ROUTE_IDS', 'ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5', 'route_input_projection', 'single_route_decision_from_ir', 'single_route_decision_issues']
