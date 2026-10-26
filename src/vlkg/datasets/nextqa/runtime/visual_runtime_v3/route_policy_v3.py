"""Fixed eight-route policy with two uniform visual handlers."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v3 import causal_how_uniform, descriptive_count_uniform
POLICY_ID = 'nextqa_visual_route_policy_v3_uniform_negative_routes'
TARGET_ROUTE_IDS = frozenset({causal_how_uniform.ROUTE_ID, descriptive_count_uniform.ROUTE_ID})
TARGET_HANDLERS = {causal_how_uniform.ROUTE_ID: causal_how_uniform.build, descriptive_count_uniform.ROUTE_ID: descriptive_count_uniform.build}
TARGET_IDENTITIES = {causal_how_uniform.ROUTE_ID: {'strategy_id': causal_how_uniform.STRATEGY_ID, 'handler_id': causal_how_uniform.HANDLER_ID}, descriptive_count_uniform.ROUTE_ID: {'strategy_id': descriptive_count_uniform.STRATEGY_ID, 'handler_id': descriptive_count_uniform.HANDLER_ID}}

def materialize_visual_record(route_record: Mapping[str, Any], routed_v2_record: Mapping[str, Any], no_routing_record: Mapping[str, Any]) -> dict[str, Any]:
    route_id = str(route_record.get('route_id') or '')
    if route_id in TARGET_HANDLERS:
        return TARGET_HANDLERS[route_id](route_record, no_routing_record)
    if str(routed_v2_record.get('route_id') or '') != route_id:
        raise ValueError('v2 visual record does not match persisted route')
    return deepcopy(dict(routed_v2_record))
__all__ = ['POLICY_ID', 'TARGET_HANDLERS', 'TARGET_IDENTITIES', 'TARGET_ROUTE_IDS', 'materialize_visual_record']
