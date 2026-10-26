"""Route-specific uniform visual retrieval for descriptive_count."""
from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.uniform_visual_common import build_uniform_route_record
ROUTE_ID = 'descriptive_count'
STRATEGY_ID = 'descriptive_count_uniform_visual_v1'
HANDLER_ID = 'nextqa_visual_handler_v3:descriptive_count_uniform_visual'

def build(route_record: Mapping[str, Any], no_routing_record: Mapping[str, Any]) -> dict[str, Any]:
    return build_uniform_route_record(route_record, no_routing_record, route_id=ROUTE_ID, strategy_id=STRATEGY_ID, handler_id=HANDLER_ID, handler_module='descriptive_count_uniform')
__all__ = ['HANDLER_ID', 'ROUTE_ID', 'STRATEGY_ID', 'build']
