"""Persisted EvidenceDemand dispatcher for isolated OpenEQA strategies."""
from __future__ import annotations
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import NEED_COUNT_SET, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_FUNCTIONAL_WORLD, NEED_OBJECT_IDENTITY, NEED_OCR_LOGO_TEXT, NEED_OPEN_DESCRIPTIVE, NEED_SPATIAL_LOCALIZATION, ROUTING_PROFILE_IR_FIRST_V4, STATE_STRATEGY_VARIANTS, STATE_VARIANT_LEGACY, STATE_VARIANT_OPEN_CLOSED, STATE_VARIANT_SURFACE_CAPACITY
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.count_set import HANDLER_ID as COUNT_SET_HANDLER_ID, fuse_count_set_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.entity_attribute import HANDLER_ID as ENTITY_ATTRIBUTE_HANDLER_ID, fuse_entity_attribute_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.existence_or_state import FROZEN_HANDLER_ID as EXISTENCE_OR_STATE_FROZEN_HANDLER_ID, HANDLER_ID as OPEN_CLOSED_STATE_HANDLER_ID, fuse_existence_or_state_frozen_evidence, fuse_existence_or_state_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.functional_world import HANDLER_ID as FUNCTIONAL_WORLD_HANDLER_ID, fuse_functional_world_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.object_identity import FROZEN_HANDLER_ID as OBJECT_IDENTITY_FROZEN_HANDLER_ID, HANDLER_ID as OBJECT_IDENTITY_HANDLER_ID, fuse_object_identity_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.open_target_relation import HANDLER_ID as OPEN_TARGET_RELATION_HANDLER_ID, fuse_open_target_relation_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.open_descriptive import SCENE_DESCRIPTION_HANDLER_ID, UNSUPPORTED_FALLBACK_HANDLER_ID, fuse_scene_or_content_description_evidence, fuse_unsupported_fallback_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.ocr_logo_text import HANDLER_ID as OCR_LOGO_TEXT_HANDLER_ID, fuse_ocr_logo_text_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.spatial_localization import HANDLER_ID as SPATIAL_LOCALIZATION_HANDLER_ID, fuse_spatial_localization_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.surface_capacity import HANDLER_ID as SURFACE_CAPACITY_HANDLER_ID, fuse_surface_capacity_evidence
OPENEQA_VISUAL_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v4'
ROUTE_STRATEGY_CONTRACT = 'openeqa_evidence_route_strategy_dispatch.v1'
FALLBACK_HANDLER_ID = 'other_routes.frozen_v3'
_Handler = Callable[..., dict[str, Any]]
_PRIMARY_NEED_HANDLERS: dict[str, tuple[str, _Handler]] = {NEED_SPATIAL_LOCALIZATION: ('spatial_localization.frozen_v3', fuse_frozen_v3), NEED_OBJECT_IDENTITY: (OBJECT_IDENTITY_FROZEN_HANDLER_ID, fuse_frozen_v3), NEED_COUNT_SET: (COUNT_SET_HANDLER_ID, fuse_count_set_evidence)}
_VARIANT_HANDLERS: dict[tuple[str, str], tuple[str, _Handler]] = {(NEED_SPATIAL_LOCALIZATION, 'target_location'): (SPATIAL_LOCALIZATION_HANDLER_ID, fuse_spatial_localization_evidence), (NEED_SPATIAL_LOCALIZATION, 'open_target_relation'): (OPEN_TARGET_RELATION_HANDLER_ID, fuse_open_target_relation_evidence), (NEED_SPATIAL_LOCALIZATION, 'relation_description_or_verification'): ('spatial_localization.relation_description.frozen_v3', fuse_frozen_v3), (NEED_OBJECT_IDENTITY, 'anchor_bound_open_target_identity'): (OBJECT_IDENTITY_HANDLER_ID, fuse_object_identity_evidence), (NEED_OBJECT_IDENTITY, 'direct_identity'): (OBJECT_IDENTITY_FROZEN_HANDLER_ID, fuse_frozen_v3), (NEED_COUNT_SET, 'direct_or_structural_entity_count'): ('count_set.direct_or_structural.frozen_v3', fuse_count_set_evidence), (NEED_COUNT_SET, 'filtered_relation_attribute_count'): ('count_set.filtered_relation_attribute.frozen_v3', fuse_count_set_evidence), (NEED_COUNT_SET, 'capacity_or_functional_count'): ('count_set.capacity_or_functional.frozen_v3', fuse_count_set_evidence), (NEED_ENTITY_ATTRIBUTE, 'owner_attribute'): (ENTITY_ATTRIBUTE_HANDLER_ID, fuse_entity_attribute_evidence), (NEED_EXISTENCE_OR_STATE, STATE_VARIANT_LEGACY): (EXISTENCE_OR_STATE_FROZEN_HANDLER_ID, fuse_existence_or_state_frozen_evidence), (NEED_EXISTENCE_OR_STATE, STATE_VARIANT_SURFACE_CAPACITY): (SURFACE_CAPACITY_HANDLER_ID, fuse_surface_capacity_evidence), (NEED_EXISTENCE_OR_STATE, STATE_VARIANT_OPEN_CLOSED): (OPEN_CLOSED_STATE_HANDLER_ID, fuse_existence_or_state_evidence), (NEED_FUNCTIONAL_WORLD, 'grounded_function_or_world'): (FUNCTIONAL_WORLD_HANDLER_ID, fuse_functional_world_evidence), (NEED_OCR_LOGO_TEXT, 'owner_aware_text'): (OCR_LOGO_TEXT_HANDLER_ID, fuse_ocr_logo_text_evidence), (NEED_OPEN_DESCRIPTIVE, 'scene_or_content_description'): (SCENE_DESCRIPTION_HANDLER_ID, fuse_scene_or_content_description_evidence), (NEED_OPEN_DESCRIPTIVE, 'unsupported_fallback'): (UNSUPPORTED_FALLBACK_HANDLER_ID, fuse_unsupported_fallback_evidence)}
for _state_variant in STATE_STRATEGY_VARIANTS:
    _VARIANT_HANDLERS.setdefault((NEED_EXISTENCE_OR_STATE, _state_variant), (EXISTENCE_OR_STATE_FROZEN_HANDLER_ID, fuse_existence_or_state_frozen_evidence))
FROZEN_V3_DELEGATION_VARIANTS = frozenset({(NEED_SPATIAL_LOCALIZATION, 'relation_description_or_verification'), (NEED_OBJECT_IDENTITY, 'direct_identity'), (NEED_COUNT_SET, 'direct_or_structural_entity_count'), (NEED_COUNT_SET, 'filtered_relation_attribute_count'), (NEED_COUNT_SET, 'capacity_or_functional_count'), (NEED_FUNCTIONAL_WORLD, 'grounded_function_or_world'), (NEED_OPEN_DESCRIPTIVE, 'unsupported_fallback')} | {(NEED_EXISTENCE_OR_STATE, variant) for variant in STATE_STRATEGY_VARIANTS if variant not in {STATE_VARIANT_SURFACE_CAPACITY, STATE_VARIANT_OPEN_CLOSED}})
ACTIVE_HANDLER_VARIANTS = frozenset({(NEED_SPATIAL_LOCALIZATION, 'target_location'), (NEED_SPATIAL_LOCALIZATION, 'open_target_relation'), (NEED_OBJECT_IDENTITY, 'anchor_bound_open_target_identity'), (NEED_EXISTENCE_OR_STATE, STATE_VARIANT_SURFACE_CAPACITY), (NEED_EXISTENCE_OR_STATE, STATE_VARIANT_OPEN_CLOSED), (NEED_ENTITY_ATTRIBUTE, 'owner_attribute'), (NEED_OCR_LOGO_TEXT, 'owner_aware_text'), (NEED_OPEN_DESCRIPTIVE, 'scene_or_content_description')})

def handler_provenance(evidence_demand: dict[str, Any]) -> str:
    """Return the isolated provenance class for a persisted V4 route."""
    key = (str(evidence_demand.get('primary_need') or ''), str(evidence_demand.get('strategy_variant') or ''))
    if key in ACTIVE_HANDLER_VARIANTS:
        return 'active_v4'
    if key in FROZEN_V3_DELEGATION_VARIANTS:
        return 'frozen_v3'
    return 'unregistered_fallback'

def _route_handler(evidence_demand: dict[str, Any]) -> tuple[str, _Handler]:
    need = str(evidence_demand.get('primary_need') or '')
    variant = str(evidence_demand.get('strategy_variant') or '')
    return _VARIANT_HANDLERS.get((need, variant), _PRIMARY_NEED_HANDLERS.get(need, (FALLBACK_HANDLER_ID, fuse_frozen_v3)))

def route_handler_id(evidence_demand: dict[str, Any]) -> str:
    return _route_handler(evidence_demand)[0]

def route_handler_contract_issues(evidence_demand: dict[str, Any]) -> list[str]:
    metadata = evidence_demand.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get('routing_profile') != ROUTING_PROFILE_IR_FIRST_V4:
        return []
    persisted = str(evidence_demand.get('route_strategy_handler') or '').strip()
    expected = route_handler_id(evidence_demand)
    if not persisted:
        return ['IR-first route_strategy_handler is missing']
    if persisted != expected:
        return [f'IR-first route_strategy_handler drift: persisted={persisted!r} expected={expected!r}']
    provenance = handler_provenance(evidence_demand)
    if provenance == 'unregistered_fallback':
        return [f"IR-first route variant is not registered in the V4 isolation contract: primary_need={evidence_demand.get('primary_need')!r} strategy_variant={evidence_demand.get('strategy_variant')!r}"]
    return []

def fuse_route_dispatched_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    handler_id, handler = _route_handler(evidence_demand)
    fused = handler(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    visual_evidence = fused.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence['visual_runtime_contract'] = OPENEQA_VISUAL_RUNTIME_CONTRACT
        visual_evidence['route_strategy_contract'] = ROUTE_STRATEGY_CONTRACT
        visual_evidence['route_strategy_handler'] = handler_id
        visual_evidence['registered_handler_provenance'] = handler_provenance(evidence_demand)
        visual_evidence['strategy_variant'] = str(evidence_demand.get('strategy_variant') or '')
        visual_evidence['frozen_v3_delegation'] = not bool(visual_evidence.get('route_strategy_applied'))
        visual_evidence['execution_evidence_provenance'] = 'active_v4_supplemental' if visual_evidence.get('route_strategy_applied') else 'frozen_v3'
    return fused
__all__ = ['FALLBACK_HANDLER_ID', 'ACTIVE_HANDLER_VARIANTS', 'FROZEN_V3_DELEGATION_VARIANTS', 'OPENEQA_VISUAL_RUNTIME_CONTRACT', 'ROUTE_STRATEGY_CONTRACT', 'fuse_route_dispatched_visual_evidence', 'handler_provenance', 'route_handler_contract_issues', 'route_handler_id']
