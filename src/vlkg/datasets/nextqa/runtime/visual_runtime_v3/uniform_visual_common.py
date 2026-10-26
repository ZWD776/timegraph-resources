"""Frozen uniform-visual strategy shared by route-specific v3 handlers."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping

def build_uniform_route_record(route_record: Mapping[str, Any], no_routing_record: Mapping[str, Any], *, route_id: str, strategy_id: str, handler_id: str, handler_module: str) -> dict[str, Any]:
    """Attach a route identity without changing the frozen uniform evidence."""
    question_id = str(route_record.get('question_id') or '')
    video_id = str(route_record.get('video_id') or '')
    if str(route_record.get('route_id') or '') != route_id:
        raise ValueError(f'uniform handler requires route_id={route_id}')
    if not question_id or question_id != str(no_routing_record.get('question_id') or ''):
        raise ValueError('route and no-routing question IDs do not align')
    if not video_id or video_id != str(no_routing_record.get('video_id') or ''):
        raise ValueError('route and no-routing video IDs do not align')
    if any((no_routing_record.get(field) is not False for field in ('uses_gold_answer', 'uses_official_type', 'uses_answer_choices', 'uses_route', 'uses_handler', 'uses_intent', 'uses_visual_evidence_demand'))):
        raise ValueError('no-routing source purity failed')
    evidence = no_routing_record.get('visual_evidence')
    if not isinstance(evidence, list) or not evidence or len(evidence) > 12:
        raise ValueError('uniform visual evidence must contain 1..12 records')
    if any(('role_id' in row for row in evidence if isinstance(row, Mapping))):
        raise ValueError('uniform visual evidence must remain role-neutral')
    return {'schema_version': 'nextqa_graph_bound_visual_evidence_record_v3', 'question_id': question_id, 'video_id': video_id, 'route_id': route_id, 'route_source': str(route_record.get('route_source') or ''), 'handler_id': handler_id, 'handler_module': handler_module, 'handler_status': 'completed', 'strategy_id': strategy_id, 'strategy_scope': 'route_wide', 'strategy_family': 'uniform_visual_retrieval', 'strategy_implementation': 'nextqa_visual_no_routing_evidence_v1', 'boundary_available': False, 'role_status': {'uniform_visual_retrieval': 'filled'}, 'graph_retrieval_status': str(no_routing_record.get('graph_retrieval_status') or ''), 'graph_retrieval_profile': str(no_routing_record.get('graph_retrieval_profile') or ''), 'graph_record_ids': list(no_routing_record.get('graph_record_ids', [])), 'graph_record_ids_sha256': str(no_routing_record.get('graph_record_ids_sha256') or ''), 'graph_evidence_count': int(no_routing_record.get('graph_evidence_count') or 0), 'visual_evidence': deepcopy(evidence), 'visual_hits': [], 'used_visual_candidate_ids': list(no_routing_record.get('visual_candidate_ids') or no_routing_record.get('used_visual_candidate_ids') or []), 'excluded_reason_counts': {}, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False, 'offline_reads_visual_index': True, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'calls_api': False}
