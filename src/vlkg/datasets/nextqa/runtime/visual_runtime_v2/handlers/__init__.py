"""Independent route handlers for NExT-QA visual runtime v2."""
from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.handlers import causal_how, causal_why, descriptive_count, descriptive_direct, descriptive_location, temporal_concurrent, temporal_next, temporal_previous
HANDLERS = {'causal_why': causal_why.build, 'causal_how': causal_how.build, 'temporal_previous': temporal_previous.build, 'temporal_next': temporal_next.build, 'temporal_concurrent': temporal_concurrent.build, 'descriptive_count': descriptive_count.build, 'descriptive_location': descriptive_location.build, 'descriptive_direct': descriptive_direct.build}

def dispatch_handler(route_record: Mapping[str, Any], role_hits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    route_id = str(route_record['route_id'])
    try:
        handler = HANDLERS[route_id]
    except KeyError as exc:
        raise ValueError(f'no visual v2 handler for route: {route_id}') from exc
    pack = handler(route_record, role_hits)
    if str(pack.get('handler_id')) != str(route_record.get('handler_id')):
        raise ValueError('actual handler does not match persisted route handler')
    return pack
