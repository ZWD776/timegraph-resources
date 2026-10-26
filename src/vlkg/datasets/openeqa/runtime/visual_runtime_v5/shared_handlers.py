"""V5-owned route handlers that currently use the frozen fusion primitive."""
from __future__ import annotations
from copy import deepcopy
import re
from typing import Any, Callable
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.existence_or_state import fuse_existence_or_state_evidence as _fuse_open_closed_state
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.handler_base import annotate_v5_handler_result
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.evidence_primitives import deduplicated_candidate_set, positive_existence_candidates, publish_count_candidate_set, publish_existence_candidates
DIRECT_IDENTITY_HANDLER_ID = 'openeqa_v5.direct_identity.visual_v1'
ENTITY_STATE_HANDLER_ID = 'openeqa_v5.entity_state.visual_v1'
ENTITY_EXISTENCE_HANDLER_ID = 'openeqa_v5.entity_existence.visual_v1'
RELATION_QUERY_HANDLER_ID = 'openeqa_v5.relation_query.visual_v1'
ENTITY_COUNT_HANDLER_ID = 'openeqa_v5.entity_count.visual_v1'
FUNCTIONAL_AFFORDANCE_HANDLER_ID = 'openeqa_v5.functional_affordance.visual_v1'
GROUNDED_WORLD_KNOWLEDGE_HANDLER_ID = 'openeqa_v5.grounded_world_knowledge.visual_v1'
SCENE_DESCRIPTION_HANDLER_ID = 'openeqa_v5.scene_description.visual_v1'
CONTENT_DESCRIPTION_HANDLER_ID = 'openeqa_v5.content_description.visual_v1'

def _fuse_with_frozen_fusion(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, evidence_demand: dict[str, Any], route_id: str, **kwargs: Any) -> dict[str, Any]:
    fused = fuse_frozen_v3(graph_result, visual_payload, evidence_demand=evidence_demand, **kwargs)
    visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence['route_strategy_applied'] = False
        if route_id == 'entity_existence':
            visual_evidence['negative_coverage_qualified'] = False
    return annotate_v5_handler_result(fused, evidence_demand=evidence_demand, specialized_fusion_applied=False, fallback_reason='v5_route_query_with_frozen_v3_fusion')

def _handler(route_id: str) -> Callable[..., dict[str, Any]]:

    def fuse(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        evidence_demand = kwargs.pop('evidence_demand')
        return _fuse_with_frozen_fusion(graph_result, visual_payload, evidence_demand=evidence_demand, route_id=route_id, **kwargs)
    fuse.__name__ = f'fuse_{route_id}_evidence'
    return fuse
fuse_direct_identity_evidence = _handler('direct_identity')

def fuse_entity_state_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence_demand = kwargs.pop('evidence_demand')
    baseline_bundle = deepcopy(graph_result.get('llm_ready_bundle'))
    state_tokens = {token for value in evidence_demand.get('state_terms', []) or [] for token in re.findall('[a-z0-9]+', str(value).casefold())}
    if state_tokens & {'open', 'opened', 'closed', 'shut'}:
        compatible_demand = deepcopy(evidence_demand)
        compatible_demand['strategy_variant'] = 'open_closed_state'
        fused = _fuse_open_closed_state(graph_result, visual_payload, evidence_demand=compatible_demand, **kwargs)
        visual_evidence = fused.get('visual_evidence')
        specialized = bool(visual_evidence.get('route_strategy_applied') if isinstance(visual_evidence, dict) else False)
        if not specialized and isinstance(baseline_bundle, dict):
            fused['llm_ready_bundle'] = baseline_bundle
        return annotate_v5_handler_result(fused, evidence_demand=evidence_demand, specialized_fusion_applied=specialized, fallback_reason='' if specialized else 'open_closed_visual_consensus_unavailable')
    return _fuse_with_frozen_fusion(graph_result, visual_payload, evidence_demand=evidence_demand, route_id='entity_state', **kwargs)

def fuse_entity_existence_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence_demand = kwargs.pop('evidence_demand')
    fused = fuse_frozen_v3(graph_result, visual_payload, evidence_demand=evidence_demand, **kwargs)
    candidates = positive_existence_candidates(visual_payload, target_terms=list(evidence_demand.get('target_terms') or []), limit=8)
    fused = publish_existence_candidates(fused, candidates=candidates)
    return annotate_v5_handler_result(fused, evidence_demand=evidence_demand, specialized_fusion_applied=bool(candidates), fallback_reason='' if candidates else 'no_strict_positive_visual_existence_candidate')
fuse_relation_query_evidence = _handler('relation_query')

def fuse_entity_count_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence_demand = kwargs.pop('evidence_demand')
    fused = fuse_frozen_v3(graph_result, visual_payload, evidence_demand=evidence_demand, **kwargs)
    candidates = deduplicated_candidate_set(visual_payload, limit=64)
    fused = publish_count_candidate_set(fused, candidates=candidates)
    return annotate_v5_handler_result(fused, evidence_demand=evidence_demand, specialized_fusion_applied=bool(candidates), fallback_reason='' if candidates else 'no_deduplicated_visual_count_candidates')
fuse_functional_affordance_evidence = _handler('functional_affordance')
fuse_grounded_world_knowledge_evidence = _handler('grounded_world_knowledge')
fuse_scene_description_evidence = _handler('scene_description')
fuse_content_description_evidence = _handler('content_description')
__all__ = ['CONTENT_DESCRIPTION_HANDLER_ID', 'DIRECT_IDENTITY_HANDLER_ID', 'ENTITY_COUNT_HANDLER_ID', 'ENTITY_EXISTENCE_HANDLER_ID', 'ENTITY_STATE_HANDLER_ID', 'FUNCTIONAL_AFFORDANCE_HANDLER_ID', 'GROUNDED_WORLD_KNOWLEDGE_HANDLER_ID', 'RELATION_QUERY_HANDLER_ID', 'SCENE_DESCRIPTION_HANDLER_ID', 'fuse_content_description_evidence', 'fuse_direct_identity_evidence', 'fuse_entity_count_evidence', 'fuse_entity_existence_evidence', 'fuse_entity_state_evidence', 'fuse_functional_affordance_evidence', 'fuse_grounded_world_knowledge_evidence', 'fuse_relation_query_evidence', 'fuse_scene_description_evidence']
