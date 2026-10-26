"""Static route-conditioned visual retrieval policies for OpenEQA V5."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from vlkg.datasets.openeqa.retrieval.evidence_router import UNIT_CANDIDATE_SET, UNIT_CANONICAL_ENTITY, UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE, UNIT_SCENE_FRAME, UNIT_SPATIAL_PAIR
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.taxonomy import ROUTE_IDS
VISUAL_STRATEGY_CONTRACT = 'openeqa_v5_visual_strategy.v1'
VISUAL_QUERY_PLAN_CONTRACT = 'openeqa_v5_visual_query_plan.v1'
RELATION_BOUND_IDENTITY_BUILDER_V1 = 'anchor_target_intersection_v1'
TARGET_LOCATION_BUILDER_V1 = 'target_scene_location_v1'

@dataclass(frozen=True)
class VisualStrategyPolicy:
    route_id: str
    visual_query_builder_id: str
    candidate_units: tuple[str, ...]
    binding_policy: str
    aggregation_policy: str
    negative_coverage_policy: str
    fallback_policy: str = 'preserve_base_graph_and_frozen_visual_evidence'
    allows_world_knowledge: bool = False
    contract: str = VISUAL_STRATEGY_CONTRACT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _policy(route_id: str, builder: str, units: tuple[str, ...], binding: str, aggregation: str, negative_coverage: str='not_applicable', *, allows_world_knowledge: bool=False) -> VisualStrategyPolicy:
    return VisualStrategyPolicy(route_id=route_id, visual_query_builder_id=builder, candidate_units=units, binding_policy=binding, aggregation_policy=aggregation, negative_coverage_policy=negative_coverage, allows_world_knowledge=allows_world_knowledge)
VISUAL_STRATEGIES: dict[str, VisualStrategyPolicy] = {'relation_bound_identity': _policy('relation_bound_identity', RELATION_BOUND_IDENTITY_BUILDER_V1, (UNIT_OBJECT_CROP, UNIT_CANONICAL_ENTITY, UNIT_SPATIAL_PAIR), 'anchor_first_then_target_relation_binding', 'rank_bound_target_identity_candidates'), 'direct_identity': _policy('direct_identity', 'direct_entity_identity_v1', (UNIT_OBJECT_CROP, UNIT_CANONICAL_ENTITY), 'target_entity_binding', 'multi_view_identity_consensus'), 'entity_attribute': _policy('entity_attribute', 'owner_locked_attribute_v1', (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE), 'target_owner_lock', 'owner_attribute_consensus'), 'entity_state': _policy('entity_state', 'owner_locked_state_v1', (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE), 'target_owner_lock', 'state_observation_consensus'), 'entity_existence': _policy('entity_existence', 'episode_entity_coverage_v1', (UNIT_CANONICAL_ENTITY, UNIT_OBJECT_CROP, UNIT_SCENE_FRAME), 'episode_target_binding', 'positive_or_coverage_qualified_existence', 'episode_visual_coverage_required_for_negative'), 'surface_capacity': _policy('surface_capacity', 'surface_occupancy_v1', (UNIT_OBJECT_CROP, UNIT_SPATIAL_PAIR), 'surface_owner_lock', 'surface_occupancy_and_free_space', 'visible_surface_coverage_required'), 'target_location': _policy('target_location', 'target_scene_location_v1', (UNIT_CANONICAL_ENTITY, UNIT_OBJECT_CROP, UNIT_SCENE_FRAME), 'target_to_scene_and_anchor_binding', 'location_relation_pack'), 'relation_query': _policy('relation_query', 'known_entity_pair_relation_v1', (UNIT_CANONICAL_ENTITY, UNIT_OBJECT_CROP, UNIT_SPATIAL_PAIR), 'known_entity_pair_binding', 'relation_pair_consensus'), 'entity_count': _policy('entity_count', 'deduplicated_entity_set_v1', (UNIT_CANDIDATE_SET, UNIT_CANONICAL_ENTITY), 'candidate_to_canonical_entity_binding', 'deduplicate_filter_then_count'), 'text_reading': _policy('text_reading', 'owner_bound_text_v1', (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE, UNIT_SCENE_FRAME), 'text_to_owner_or_location_binding', 'ocr_text_candidate_consensus'), 'functional_affordance': _policy('functional_affordance', 'grounded_affordance_v1', (UNIT_CANONICAL_ENTITY, UNIT_OBJECT_CROP, UNIT_SCENE_FRAME), 'target_and_scene_grounding', 'grounded_affordance_evidence', allows_world_knowledge=True), 'grounded_world_knowledge': _policy('grounded_world_knowledge', 'scene_grounded_world_knowledge_v1', (UNIT_SCENE_FRAME,), 'scene_fact_grounding', 'scene_facts_before_world_knowledge', allows_world_knowledge=True), 'scene_description': _policy('scene_description', 'scene_summary_v1', (UNIT_SCENE_FRAME, UNIT_CANONICAL_ENTITY), 'episode_scene_scope', 'representative_scene_summary'), 'content_description': _policy('content_description', 'content_carrier_v1', (UNIT_OBJECT_CROP, UNIT_SCENE_FRAME), 'content_carrier_binding', 'carrier_content_observation_consensus')}
LEGACY_RELATION_BOUND_IDENTITY_STRATEGY = _policy('relation_bound_identity', RELATION_BOUND_IDENTITY_BUILDER_V1, (UNIT_OBJECT_CROP, UNIT_CANONICAL_ENTITY, UNIT_SPATIAL_PAIR), 'anchor_first_then_target_relation_binding', 'rank_bound_target_identity_candidates')

def get_visual_strategy(route_id: str, *, visual_query_builder_id: str='') -> VisualStrategyPolicy:
    if str(route_id).strip() == 'relation_bound_identity' and visual_query_builder_id == RELATION_BOUND_IDENTITY_BUILDER_V1:
        return LEGACY_RELATION_BOUND_IDENTITY_STRATEGY
    try:
        policy = VISUAL_STRATEGIES[str(route_id).strip()]
    except KeyError as exc:
        raise ValueError(f'unsupported OpenEQA V5 visual strategy: {route_id!r}') from exc
    if visual_query_builder_id and visual_query_builder_id != policy.visual_query_builder_id:
        raise ValueError(f'unsupported OpenEQA V5 visual query builder: {visual_query_builder_id!r}')
    return policy

def build_visual_query_plan(question_ir: Mapping[str, Any], route_decision: Mapping[str, Any]) -> dict[str, Any]:
    route_id = str(route_decision.get('route_id') or '').strip()
    if str(question_ir.get('question_type') or '').strip() != route_id:
        raise ValueError('semantic IR question_type and persisted route_id must match')
    policy = get_visual_strategy(route_id, visual_query_builder_id=str(route_decision.get('visual_query_builder_id') or '').strip())
    targets = _text_list(question_ir.get('target_entities'))
    attributes = _text_list(question_ir.get('attribute_dimension'))
    states = _text_list(question_ir.get('target_state'))
    constraints = _constraint_texts(question_ir.get('hard_constraints'))
    spatial_constraints = _spatial_constraints(question_ir.get('spatial_constraints'))
    anchors = _ordered_unique((str(item.get('target') or item.get('anchor') or '') for item in spatial_constraints))
    intents: list[dict[str, Any]] = []

    def add(text: str, *, role: str, unit: str, source: str, spatial_relation: str='') -> None:
        cleaned = _clean(text)
        if not cleaned:
            return
        intents.append({'text': cleaned, 'role': role, 'evidence_unit': unit, 'source': source, 'spatial_relation': _clean(spatial_relation), 'target_terms': list(targets), 'anchor_terms': list(anchors)})
    _add_route_intents(route_id, add=add, targets=targets, anchors=anchors, attributes=attributes, states=states, constraints=constraints, spatial_constraints=spatial_constraints)
    unique_intents: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for intent in intents:
        key = (intent['text'].casefold(), intent['role'], intent['evidence_unit'])
        if key in seen:
            continue
        seen.add(key)
        unique_intents.append(intent)
    if route_id == 'surface_capacity':
        surface_anchors = anchors or targets
        for intent in unique_intents:
            intent['target_terms'] = []
            intent['anchor_terms'] = list(surface_anchors)
    if not unique_intents:
        raise ValueError(f'{route_id}: semantic IR cannot build a visual query intent')
    route_metadata = route_decision.get('metadata')
    route_metadata = route_metadata if isinstance(route_metadata, Mapping) else {}
    plan = {'route_id': route_id, 'handler_id': route_decision.get('handler_id'), 'metadata': {'routing_profile': route_metadata.get('routing_profile'), 'official_category_used_for_routing': False}, 'contract': VISUAL_QUERY_PLAN_CONTRACT, 'route_contract': route_decision.get('contract'), 'visual_strategy_contract': VISUAL_STRATEGY_CONTRACT, 'visual_query_builder_id': policy.visual_query_builder_id, 'candidate_units': list(policy.candidate_units), 'binding_policy': policy.binding_policy, 'aggregation_policy': policy.aggregation_policy, 'negative_coverage_policy': policy.negative_coverage_policy, 'fallback_policy': policy.fallback_policy, 'allows_common_world_knowledge': policy.allows_world_knowledge, 'target_terms': [] if route_id == 'surface_capacity' else targets, 'anchor_terms': anchors or targets if route_id == 'surface_capacity' else anchors, 'attribute_dimensions': attributes, 'state_terms': states, 'constraint_terms': constraints, 'spatial_constraints': spatial_constraints, 'query_intents': unique_intents[:12], 'uses_gold_answer': False, 'uses_official_question_category': False, 'source': 'semantic_question_ir.parameters'}
    return plan

def _add_route_intents(route_id: str, *, add: Any, targets: list[str], anchors: list[str], attributes: list[str], states: list[str], constraints: list[str], spatial_constraints: list[dict[str, str]]) -> None:
    modifier_text = ' '.join([*attributes, *states, *constraints[:2]])
    if route_id == 'relation_bound_identity':
        for target in targets or ['object']:
            add(f'{target} {modifier_text}', role='target', unit=UNIT_OBJECT_CROP, source='relation_bound_target')
            add(target, role='target', unit=UNIT_CANONICAL_ENTITY, source='relation_bound_entity')
        for item in spatial_constraints:
            anchor = item.get('target', '')
            add(anchor, role='anchor', unit=UNIT_OBJECT_CROP, source='relation_anchor', spatial_relation=item.get('relation', ''))
        return
    if route_id in {'direct_identity', 'entity_attribute', 'entity_state'}:
        unit = UNIT_OWNER_ATTRIBUTE if route_id != 'direct_identity' else UNIT_CANONICAL_ENTITY
        for target in targets:
            add(f'{target} {modifier_text}', role='target', unit=UNIT_OBJECT_CROP, source=f'{route_id}_target')
            add(f'{target} {modifier_text}', role='owner', unit=unit, source=f'{route_id}_owner')
        return
    if route_id == 'entity_existence':
        for target in targets:
            add(target, role='target', unit=UNIT_CANONICAL_ENTITY, source='existence_entity')
            add(target, role='target', unit=UNIT_OBJECT_CROP, source='existence_observation')
        add(' '.join([*targets, *constraints]), role='context', unit=UNIT_SCENE_FRAME, source='existence_coverage')
        return
    if route_id == 'surface_capacity':
        for item in spatial_constraints or [{'relation': 'on', 'target': target} for target in targets]:
            add(item.get('target', ''), role='anchor', unit=UNIT_OBJECT_CROP, source='surface_anchor', spatial_relation=item.get('relation', ''))
        add(' '.join(constraints), role='target', unit=UNIT_OBJECT_CROP, source='question_bound_target')
        add(' '.join([*constraints, *[f'on {target}' for target in targets]]), role='context', unit=UNIT_OBJECT_CROP, source='surface_constraints')
        return
    if route_id == 'target_location':
        for target in targets:
            add(target, role='target', unit=UNIT_CANONICAL_ENTITY, source='location_entity')
            add(target, role='target', unit=UNIT_OBJECT_CROP, source='location_observation')
        add(' '.join([*targets, *constraints, 'room scene']), role='context', unit=UNIT_SCENE_FRAME, source='location_scene')
        return
    if route_id == 'relation_query':
        for index, target in enumerate(targets):
            add(target, role='target' if index == 0 else 'anchor', unit=UNIT_CANONICAL_ENTITY, source='relation_entity')
            add(target, role='target' if index == 0 else 'anchor', unit=UNIT_OBJECT_CROP, source='relation_observation')
        return
    if route_id == 'entity_count':
        for target in targets:
            add(f'{target} {modifier_text}', role='candidate_set', unit=UNIT_CANDIDATE_SET, source='count_candidates')
        return
    if route_id == 'text_reading':
        for target in targets or ['visible text']:
            add(f'{target} visible text writing logo', role='target', unit=UNIT_OBJECT_CROP, source='text_observation')
            add(target, role='owner', unit=UNIT_OWNER_ATTRIBUTE, source='text_owner')
        add(' '.join([*targets, *constraints, 'visible text']), role='context', unit=UNIT_SCENE_FRAME, source='text_scene')
        return
    if route_id == 'functional_affordance':
        for target in targets:
            add(f'{target} {modifier_text}', role='target', unit=UNIT_CANONICAL_ENTITY, source='grounded_entity')
            add(f'{target} {modifier_text}', role='target', unit=UNIT_OBJECT_CROP, source='grounded_observation')
        add(' '.join([*targets, *constraints, 'scene context']), role='context', unit=UNIT_SCENE_FRAME, source='grounded_scene')
        return
    if route_id == 'grounded_world_knowledge':
        add(' '.join([*targets, 'whole scene surroundings context']), role='context', unit=UNIT_SCENE_FRAME, source='world_knowledge_scene')
        return
    if route_id == 'scene_description':
        add(' '.join([*targets, *constraints, 'room scene overview']), role='context', unit=UNIT_SCENE_FRAME, source='scene_overview')
        return
    if route_id == 'content_description':
        for target in targets or ['painting image screen']:
            add(f'{target} depicted content', role='target', unit=UNIT_OBJECT_CROP, source='content_carrier')
        add(' '.join([*targets, *constraints, 'depicted content']), role='context', unit=UNIT_SCENE_FRAME, source='content_context')

def _clean(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clean(value)] if _clean(value) else []
    if isinstance(value, (list, tuple, set)):
        return _ordered_unique((_clean(item) for item in value))
    return []

def _constraint_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _ordered_unique((_clean(item.get('text')) if isinstance(item, Mapping) else _clean(item) for item in value))

def _spatial_constraints(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [{'relation': _clean(item.get('relation')), 'target': _clean(item.get('target') or item.get('anchor'))} for item in value if isinstance(item, Mapping) and (_clean(item.get('relation')) or _clean(item.get('target') or item.get('anchor')))]

def _ordered_unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
if set(VISUAL_STRATEGIES) != set(ROUTE_IDS):
    raise RuntimeError('OpenEQA V5 visual strategies must cover exactly the taxonomy')
__all__ = ['VISUAL_QUERY_PLAN_CONTRACT', 'RELATION_BOUND_IDENTITY_BUILDER_V1', 'TARGET_LOCATION_BUILDER_V1', 'VISUAL_STRATEGIES', 'VISUAL_STRATEGY_CONTRACT', 'VisualStrategyPolicy', 'build_visual_query_plan', 'get_visual_strategy']
