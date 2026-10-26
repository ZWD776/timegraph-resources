"""Direct route-to-handler dispatcher for OpenEQA IR-first V5."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.policy_registry import get_route_policy
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.route_contracts import ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5
OPENEQA_VISUAL_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v5'
ROUTE_STRATEGY_CONTRACT = 'openeqa_single_route_strategy_dispatch.v1'

def _route_id(evidence_demand: dict[str, Any]) -> str:
    return str(evidence_demand.get('route_id') or '').strip()

def route_handler_id(evidence_demand: dict[str, Any]) -> str:
    return _policy(evidence_demand).handler_id

def handler_provenance(evidence_demand: dict[str, Any]) -> str:
    return _policy(evidence_demand).provenance

def route_visual_query_builder_id(evidence_demand: dict[str, Any]) -> str:
    return _policy(evidence_demand).visual_query_builder_id

def route_visual_strategy_contract(evidence_demand: dict[str, Any]) -> str:
    return _policy(evidence_demand).visual_strategy_contract

def _policy(evidence_demand: dict[str, Any]) -> Any:
    return get_route_policy(_route_id(evidence_demand), handler_id=str(evidence_demand.get('handler_id') or '').strip())

def route_handler_contract_issues(evidence_demand: dict[str, Any]) -> list[str]:
    metadata = evidence_demand.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get('routing_profile') != ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5:
        return []
    route_id = _route_id(evidence_demand)
    try:
        policy = _policy(evidence_demand)
    except ValueError as exc:
        return [str(exc)]
    persisted = str(evidence_demand.get('handler_id') or '').strip()
    if not persisted:
        return ['single-route handler_id is missing']
    if persisted != policy.handler_id:
        return [f'single-route handler_id drift: persisted={persisted!r} expected={policy.handler_id!r}']
    persisted_builder = str(evidence_demand.get('visual_query_builder_id') or '').strip()
    if persisted_builder != policy.visual_query_builder_id:
        return [f'single-route visual_query_builder_id drift: persisted={persisted_builder!r} expected={policy.visual_query_builder_id!r}']
    persisted_strategy_contract = str(evidence_demand.get('visual_strategy_contract') or '').strip()
    if persisted_strategy_contract != policy.visual_strategy_contract:
        return [f'single-route visual_strategy_contract drift: persisted={persisted_strategy_contract!r} expected={policy.visual_strategy_contract!r}']
    if evidence_demand.get('strategy_variant'):
        return ['single-route execution must not persist strategy_variant']
    if evidence_demand.get('primary_need'):
        return ['single-route execution must not persist primary_need']
    return []

def fuse_single_route_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Dispatch once by persisted route_id.

    The compatibility fields below are private adapter inputs for existing V4
    handler functions.  They are populated from the already selected policy;
    they do not perform a second question classification.
    """
    route_id = _route_id(evidence_demand)
    policy = _policy(evidence_demand)
    issues = route_handler_contract_issues(evidence_demand)
    if issues:
        raise ValueError(issues[0])
    compatible_demand = deepcopy(evidence_demand)
    compatible_demand['primary_need'] = policy.primary_need
    compatible_demand['strategy_variant'] = policy.handler_variant
    compatible_demand['metadata'] = {**dict(compatible_demand.get('metadata') or {}), 'routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5}
    fused = policy.handler(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=compatible_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence.pop('primary_need', None)
        visual_evidence.pop('strategy_variant', None)
        visual_evidence.pop('route_strategy_variant', None)
        visual_evidence.pop('route_strategy_handler', None)
        specialized_fusion_applied = bool(visual_evidence.get('specialized_fusion_applied') or visual_evidence.get('route_strategy_applied'))
        visual_query_strategy_applied = bool(visual_evidence.get('visual_query_strategy_applied'))
        visual_evidence.update({'visual_runtime_contract': OPENEQA_VISUAL_RUNTIME_CONTRACT, 'route_strategy_contract': ROUTE_STRATEGY_CONTRACT, 'route_id': route_id, 'handler_id': policy.handler_id, 'visual_query_builder_id': policy.visual_query_builder_id, 'visual_strategy_contract': policy.visual_strategy_contract, 'registered_handler_provenance': policy.provenance, 'frozen_v3_delegation': not specialized_fusion_applied, 'execution_evidence_provenance': 'active_v5_specialized_fusion' if specialized_fusion_applied else 'v5_route_query_with_frozen_v3_fusion' if visual_query_strategy_applied else 'frozen_v3'})
    return fused
__all__ = ['OPENEQA_VISUAL_RUNTIME_CONTRACT', 'ROUTE_STRATEGY_CONTRACT', 'fuse_single_route_visual_evidence', 'handler_provenance', 'route_handler_contract_issues', 'route_handler_id', 'route_visual_query_builder_id', 'route_visual_strategy_contract']
