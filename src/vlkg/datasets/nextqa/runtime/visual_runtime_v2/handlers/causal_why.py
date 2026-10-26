from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.candidate_packs import build_standard_candidate_pack
HANDLER_ID = 'nextqa_visual_handler_v2:causal_why'

def build(route_record: Mapping[str, Any], role_hits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return build_standard_candidate_pack(route_record, role_hits, actual_handler_id=HANDLER_ID)
