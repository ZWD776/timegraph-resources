from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.candidate_packs import build_standard_candidate_pack
HANDLER_ID = 'nextqa_visual_handler_v2:temporal_previous'

def _accept(_role: str, hit: Mapping[str, Any]) -> bool:
    if hit.get('candidate_kind') != 'before_after_pair':
        return True
    metadata = hit.get('metadata') or {}
    return metadata.get('before_ts') is not None

def build(route_record: Mapping[str, Any], role_hits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return build_standard_candidate_pack(route_record, role_hits, actual_handler_id=HANDLER_ID, accept_hit=_accept)
