"""Fail-closed execution constraints for OpenEQA IR-first routing."""
from __future__ import annotations
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES
from vlkg.datasets.openeqa.retrieval.evidence_router import IR_FIRST_ROUTING_PROFILES
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.route_contracts import ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5
IR_FIRST_VISUAL_BACKBONE = 'openeqa_question_router_v4'
SINGLE_ROUTE_VISUAL_BACKBONE = 'openeqa_question_router_v5'

def question_ir_routing_profile(question_ir: object) -> str:
    if not isinstance(question_ir, dict):
        return ''
    route = question_ir.get('question_route')
    if not isinstance(route, dict):
        return ''
    metadata = route.get('metadata')
    if not isinstance(metadata, dict):
        return ''
    return str(metadata.get('routing_profile') or '').strip()

def ir_first_execution_config_issues(*, routing_profile: str, visual_backbone: str, graph_retrieval_profile: str) -> list[str]:
    if routing_profile not in IR_FIRST_ROUTING_PROFILES:
        if routing_profile != ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5:
            return []
    issues: list[str] = []
    expected_backbone = SINGLE_ROUTE_VISUAL_BACKBONE if routing_profile == ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5 else IR_FIRST_VISUAL_BACKBONE
    if visual_backbone != expected_backbone:
        issues.append(f"IR-first routing requires visual_backbone={expected_backbone}, got {visual_backbone or '<empty>'}")
    if graph_retrieval_profile not in UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES:
        issues.append(f"IR-first routing requires a category-independent unified graph retrieval profile, got {graph_retrieval_profile or '<empty>'}")
    return issues
__all__ = ['IR_FIRST_VISUAL_BACKBONE', 'SINGLE_ROUTE_VISUAL_BACKBONE', 'ir_first_execution_config_issues', 'question_ir_routing_profile']
