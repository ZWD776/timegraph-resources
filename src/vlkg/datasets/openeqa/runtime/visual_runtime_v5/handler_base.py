"""Shared result contract for OpenEQA V5 route handlers."""
from __future__ import annotations
from typing import Any
V5_HANDLER_CONTRACT = 'openeqa_v5_route_handler.v1'

def annotate_v5_handler_result(fused: dict[str, Any], *, evidence_demand: dict[str, Any], specialized_fusion_applied: bool, fallback_reason: str='') -> dict[str, Any]:
    visual_evidence = fused.get('visual_evidence')
    if not isinstance(visual_evidence, dict):
        return fused
    visual_evidence.update({'v5_handler_contract': V5_HANDLER_CONTRACT, 'visual_query_strategy_applied': bool(evidence_demand.get('visual_query_builder_id') and evidence_demand.get('query_intents')), 'specialized_fusion_applied': bool(specialized_fusion_applied), 'base_graph_evidence_preserved': True, 'fallback_policy': evidence_demand.get('fallback_policy')})
    if fallback_reason:
        visual_evidence['handler_fallback_reason'] = fallback_reason
    return fused
__all__ = ['V5_HANDLER_CONTRACT', 'annotate_v5_handler_result']
