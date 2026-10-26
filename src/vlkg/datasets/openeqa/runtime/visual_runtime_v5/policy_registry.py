"""Flat OpenEQA route-to-handler registry for the V5 runtime."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from vlkg.datasets.openeqa.retrieval.evidence_router import NEED_COUNT_SET, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_FUNCTIONAL_WORLD, NEED_OBJECT_IDENTITY, NEED_OCR_LOGO_TEXT, NEED_OPEN_DESCRIPTIVE, NEED_SPATIAL_LOCALIZATION
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.active_handlers import ENTITY_ATTRIBUTE_HANDLER_ID, RELATION_BOUND_IDENTITY_HANDLER_ID, SURFACE_CAPACITY_HANDLER_ID, TARGET_LOCATION_HANDLER_ID, TEXT_READING_HANDLER_ID, fuse_entity_attribute_evidence, fuse_relation_bound_identity_evidence, fuse_surface_capacity_evidence, fuse_target_location_evidence, fuse_text_reading_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.shared_handlers import CONTENT_DESCRIPTION_HANDLER_ID, DIRECT_IDENTITY_HANDLER_ID, ENTITY_COUNT_HANDLER_ID, ENTITY_EXISTENCE_HANDLER_ID, ENTITY_STATE_HANDLER_ID, FUNCTIONAL_AFFORDANCE_HANDLER_ID, GROUNDED_WORLD_KNOWLEDGE_HANDLER_ID, RELATION_QUERY_HANDLER_ID, SCENE_DESCRIPTION_HANDLER_ID, fuse_content_description_evidence, fuse_direct_identity_evidence, fuse_entity_count_evidence, fuse_entity_existence_evidence, fuse_entity_state_evidence, fuse_functional_affordance_evidence, fuse_grounded_world_knowledge_evidence, fuse_relation_query_evidence, fuse_scene_description_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.visual_strategy import get_visual_strategy
_Handler = Callable[..., dict[str, Any]]

@dataclass(frozen=True)
class RoutePolicy:
    route_id: str
    primary_need: str
    handler_id: str
    handler: _Handler
    handler_variant: str
    provenance: str
    visual_strategy_contract: str
    visual_query_builder_id: str

def _policy(route_id: str, primary_need: str, handler_id: str, handler: _Handler, *, provenance: str, handler_variant: str | None=None, visual_query_builder_id: str='') -> RoutePolicy:
    visual_strategy = get_visual_strategy(route_id, visual_query_builder_id=visual_query_builder_id)
    return RoutePolicy(route_id=route_id, primary_need=primary_need, handler_id=handler_id, handler=handler, handler_variant=handler_variant or route_id, provenance=provenance, visual_strategy_contract=visual_strategy.contract, visual_query_builder_id=visual_strategy.visual_query_builder_id)
POLICIES: dict[str, RoutePolicy] = {'relation_bound_identity': _policy('relation_bound_identity', NEED_OBJECT_IDENTITY, RELATION_BOUND_IDENTITY_HANDLER_ID, fuse_relation_bound_identity_evidence, provenance='selected_v5_relation_v1', handler_variant='anchor_bound_open_target_identity'), 'direct_identity': _policy('direct_identity', NEED_OBJECT_IDENTITY, DIRECT_IDENTITY_HANDLER_ID, fuse_direct_identity_evidence, provenance='v5_owned_frozen_fusion'), 'entity_attribute': _policy('entity_attribute', NEED_ENTITY_ATTRIBUTE, ENTITY_ATTRIBUTE_HANDLER_ID, fuse_entity_attribute_evidence, provenance='active_v5', handler_variant='owner_attribute'), 'entity_state': _policy('entity_state', NEED_EXISTENCE_OR_STATE, ENTITY_STATE_HANDLER_ID, fuse_entity_state_evidence, provenance='active_v5', handler_variant='other_state'), 'entity_existence': _policy('entity_existence', NEED_EXISTENCE_OR_STATE, ENTITY_EXISTENCE_HANDLER_ID, fuse_entity_existence_evidence, provenance='active_v5', handler_variant='binary_existence'), 'surface_capacity': _policy('surface_capacity', NEED_EXISTENCE_OR_STATE, SURFACE_CAPACITY_HANDLER_ID, fuse_surface_capacity_evidence, provenance='active_v5'), 'target_location': _policy('target_location', NEED_SPATIAL_LOCALIZATION, TARGET_LOCATION_HANDLER_ID, fuse_target_location_evidence, provenance='active_v5'), 'relation_query': _policy('relation_query', NEED_SPATIAL_LOCALIZATION, RELATION_QUERY_HANDLER_ID, fuse_relation_query_evidence, provenance='v5_owned_frozen_fusion', handler_variant='relation_description_or_verification'), 'entity_count': _policy('entity_count', NEED_COUNT_SET, ENTITY_COUNT_HANDLER_ID, fuse_entity_count_evidence, provenance='active_v5', handler_variant='direct_or_structural_entity_count'), 'text_reading': _policy('text_reading', NEED_OCR_LOGO_TEXT, TEXT_READING_HANDLER_ID, fuse_text_reading_evidence, provenance='active_v5', handler_variant='owner_aware_text'), 'functional_affordance': _policy('functional_affordance', NEED_FUNCTIONAL_WORLD, FUNCTIONAL_AFFORDANCE_HANDLER_ID, fuse_functional_affordance_evidence, provenance='v5_owned_frozen_fusion', handler_variant='grounded_function_or_world'), 'grounded_world_knowledge': _policy('grounded_world_knowledge', NEED_FUNCTIONAL_WORLD, GROUNDED_WORLD_KNOWLEDGE_HANDLER_ID, fuse_grounded_world_knowledge_evidence, provenance='v5_owned_frozen_fusion', handler_variant='grounded_function_or_world'), 'scene_description': _policy('scene_description', NEED_OPEN_DESCRIPTIVE, SCENE_DESCRIPTION_HANDLER_ID, fuse_scene_description_evidence, provenance='v5_owned_frozen_fusion', handler_variant='scene_or_content_description'), 'content_description': _policy('content_description', NEED_OPEN_DESCRIPTIVE, CONTENT_DESCRIPTION_HANDLER_ID, fuse_content_description_evidence, provenance='v5_owned_frozen_fusion', handler_variant='scene_or_content_description')}
LEGACY_RELATION_BOUND_IDENTITY_POLICY = _policy('relation_bound_identity', NEED_OBJECT_IDENTITY, RELATION_BOUND_IDENTITY_HANDLER_ID, fuse_relation_bound_identity_evidence, provenance='active_v5_legacy_relation_v1', handler_variant='anchor_bound_open_target_identity', visual_query_builder_id='anchor_target_intersection_v1')
POLICIES_BY_HANDLER_ID = {policy.handler_id: policy for policy in POLICIES.values()}

def get_route_policy(route_id: str, *, handler_id: str='') -> RoutePolicy:
    cleaned_route_id = str(route_id).strip()
    if handler_id:
        current = POLICIES.get(cleaned_route_id)
        if current is not None and current.handler_id == str(handler_id).strip():
            return current
        try:
            policy = POLICIES_BY_HANDLER_ID[str(handler_id).strip()]
        except KeyError as exc:
            raise ValueError(f'unsupported OpenEQA V5 handler: {handler_id!r}') from exc
        if policy.route_id != cleaned_route_id:
            raise ValueError(f'OpenEQA V5 handler does not belong to route: route={cleaned_route_id!r} handler={handler_id!r}')
        return policy
    try:
        return POLICIES[cleaned_route_id]
    except KeyError as exc:
        raise ValueError(f'unsupported OpenEQA single route: {route_id!r}') from exc
__all__ = ['LEGACY_RELATION_BOUND_IDENTITY_POLICY', 'POLICIES', 'POLICIES_BY_HANDLER_ID', 'RoutePolicy', 'get_route_policy']
