"""V5-owned boundaries for the first route-specific visual handlers."""
from __future__ import annotations
from copy import deepcopy
import re
from typing import Any, Callable
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.entity_attribute import fuse_entity_attribute_evidence as _fuse_entity_attribute
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.object_identity import fuse_object_identity_evidence as _fuse_relation_bound_identity_v1
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.ocr_logo_text import fuse_ocr_logo_text_evidence as _fuse_text_reading
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.spatial_localization import fuse_spatial_localization_evidence as _fuse_target_location
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.surface_capacity import fuse_surface_capacity_evidence as _fuse_surface_capacity
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.handler_base import annotate_v5_handler_result
RELATION_BOUND_IDENTITY_HANDLER_ID = 'openeqa_v5.relation_bound_identity.visual_v1'
ENTITY_ATTRIBUTE_HANDLER_ID = 'openeqa_v5.entity_attribute.visual_v1'
SURFACE_CAPACITY_HANDLER_ID = 'openeqa_v5.surface_capacity.visual_v1'
TARGET_LOCATION_HANDLER_ID = 'openeqa_v5.target_location.visual_v1'
TEXT_READING_HANDLER_ID = 'openeqa_v5.text_reading.visual_v1'

def _delegate(handler: Callable[..., dict[str, Any]], graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence_demand = kwargs.get('evidence_demand')
    fused = handler(graph_result, visual_payload, **kwargs)
    visual_evidence = fused.get('visual_evidence')
    specialized = bool(visual_evidence.get('route_strategy_applied') if isinstance(visual_evidence, dict) else False)
    existing_fallback_reason = str(visual_evidence.get('handler_fallback_reason') if isinstance(visual_evidence, dict) else '').strip()
    return annotate_v5_handler_result(fused, evidence_demand=evidence_demand if isinstance(evidence_demand, dict) else {}, specialized_fusion_applied=specialized, fallback_reason='' if specialized else existing_fallback_reason or 'specialized_visual_evidence_unavailable')

def fuse_relation_bound_identity_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _delegate(_fuse_relation_bound_identity_v1, graph_result, visual_payload, **kwargs)

def fuse_entity_attribute_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _delegate(_fuse_entity_attribute, graph_result, visual_payload, **kwargs)

def fuse_surface_capacity_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence_demand = kwargs.get('evidence_demand')
    specialized_result = _fuse_surface_capacity(deepcopy(graph_result), visual_payload, **kwargs)
    visual_evidence = specialized_result.get('visual_evidence')
    lines = visual_evidence.get('surface_capacity_candidate_lines', []) if isinstance(visual_evidence, dict) else []
    explicit_signal = _surface_capacity_signal(lines)
    if explicit_signal:
        fused = specialized_result
    else:
        fused = fuse_frozen_v3(graph_result, visual_payload, graph=kwargs['graph'], question=kwargs['question'], question_ir=kwargs.get('question_ir'), evidence_demand=evidence_demand if isinstance(evidence_demand, dict) else {}, max_candidate_packs=kwargs.get('max_candidate_packs', 8), max_visual_lines=kwargs.get('max_visual_lines', 8), semantic_text_encoder=kwargs.get('semantic_text_encoder'))
        visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict) and (not explicit_signal):
        visual_evidence['route_strategy_applied'] = False
        visual_evidence['specialized_fusion_applied'] = False
        visual_evidence['graph_only_fallback'] = False
        visual_evidence['handler_fallback_reason'] = 'no_explicit_surface_signal_preserve_route_visual_frozen_fusion'
    return annotate_v5_handler_result(fused, evidence_demand=evidence_demand if isinstance(evidence_demand, dict) else {}, specialized_fusion_applied=explicit_signal, fallback_reason='' if explicit_signal else 'no_explicit_surface_signal_preserve_route_visual_frozen_fusion')

def fuse_target_location_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _delegate(_fuse_target_location, graph_result, visual_payload, **kwargs)

def fuse_text_reading_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _delegate(_fuse_text_reading, graph_result, visual_payload, **kwargs)

def _surface_capacity_signal(lines: Any) -> bool:
    observation_text = ' '.join((str(line) for line in lines or [] if 'kind=owner_bound_surface_observation' in str(line))).casefold()
    return bool(re.search('\\b(?:clear|empty|free|available|cluttered|crowded|covered|occupied)\\b|\\b(?:room|space)\\s+(?:to|for)\\b|\\b(?:objects|items|nothing)\\s+on\\b', observation_text))
__all__ = ['ENTITY_ATTRIBUTE_HANDLER_ID', 'RELATION_BOUND_IDENTITY_HANDLER_ID', 'SURFACE_CAPACITY_HANDLER_ID', 'TARGET_LOCATION_HANDLER_ID', 'TEXT_READING_HANDLER_ID', 'fuse_entity_attribute_evidence', 'fuse_relation_bound_identity_evidence', 'fuse_surface_capacity_evidence', 'fuse_target_location_evidence', 'fuse_text_reading_evidence']
