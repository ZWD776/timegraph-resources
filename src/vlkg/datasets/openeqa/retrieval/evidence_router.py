"""OpenEQA evidence-demand router for graph-bound visual retrieval.

The route is the evidence demand: it selects one visual evidence backbone while
leaving the graph-only retrieval path unchanged. OpenEQA v0 has no temporal QA
route, so temporal metadata in an IR record is treated only as scene context.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.route_contracts import ROUTE_BINARY_EXISTENCE, ROUTE_COUNT_SET, ROUTE_ENTITY_ATTRIBUTE, ROUTE_ENTITY_RELATION_SPATIAL, ROUTE_OCR_LOGO_TEXT, ROUTE_OPEN_DESCRIPTIVE
EVIDENCE_DEMAND_CONTRACT = 'openeqa_question_route_decision.v1'
EVIDENCE_DEMAND_CONTRACT_V2 = 'openeqa_question_route_decision.v2'
EVIDENCE_DEMAND_CONTRACTS = {EVIDENCE_DEMAND_CONTRACT, EVIDENCE_DEMAND_CONTRACT_V2}
ROUTING_PROFILE_CATEGORY_FIRST_V1 = 'openeqa_v0_category_first_v1'
ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2 = 'openeqa_v0_evidence_demand_variants_v2'
ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3 = 'openeqa_v0_evidence_demand_variants_v3'
ROUTING_PROFILE_IR_FIRST_V4 = 'openeqa_ir_first_evidence_demand_variants_v4'
IR_FIRST_ROUTING_PROFILES = frozenset({ROUTING_PROFILE_IR_FIRST_V4})
EVIDENCE_DEMAND_VARIANT_PROFILES = frozenset({ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2, ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3, ROUTING_PROFILE_IR_FIRST_V4})
STATE_SUBVARIANT_ROUTING_PROFILES = frozenset({ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3, ROUTING_PROFILE_IR_FIRST_V4})
ROUTING_PROFILES = {ROUTING_PROFILE_CATEGORY_FIRST_V1, ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V2, ROUTING_PROFILE_EVIDENCE_DEMAND_VARIANTS_V3, ROUTING_PROFILE_IR_FIRST_V4}
OPEN_EQA_V0_ROUTE_TAXONOMY_CONTRACT = 'openeqa_v0_evidence_route_taxonomy.v1'
NEED_OBJECT_IDENTITY = 'object_identity'
NEED_ENTITY_ATTRIBUTE = 'entity_attribute'
NEED_EXISTENCE_OR_STATE = 'existence_or_state'
NEED_SPATIAL_LOCALIZATION = 'spatial_localization'
NEED_COUNT_SET = 'count_set'
NEED_OCR_LOGO_TEXT = 'ocr_logo_text'
NEED_FUNCTIONAL_WORLD = 'functional_world'
NEED_OPEN_DESCRIPTIVE = 'open_descriptive'
STATE_VARIANT_LEGACY = 'state_or_existence'
STATE_VARIANT_SURFACE_CAPACITY = 'surface_capacity'
STATE_VARIANT_OPEN_CLOSED = 'open_closed_state'
STATE_VARIANT_POWER = 'power_state'
STATE_VARIANT_CLEANLINESS = 'cleanliness_state'
STATE_VARIANT_FULL_EMPTY = 'full_empty_state'
STATE_VARIANT_EXISTENCE = 'binary_existence'
STATE_VARIANT_OTHER = 'other_state'
STATE_STRATEGY_VARIANTS = frozenset({STATE_VARIANT_SURFACE_CAPACITY, STATE_VARIANT_OPEN_CLOSED, STATE_VARIANT_POWER, STATE_VARIANT_CLEANLINESS, STATE_VARIANT_FULL_EMPTY, STATE_VARIANT_EXISTENCE, STATE_VARIANT_OTHER})
IR_FIRST_STRATEGY_VARIANTS_BY_NEED = {NEED_OBJECT_IDENTITY: frozenset({'anchor_bound_open_target_identity', 'direct_identity'}), NEED_ENTITY_ATTRIBUTE: frozenset({'owner_attribute'}), NEED_EXISTENCE_OR_STATE: STATE_STRATEGY_VARIANTS, NEED_SPATIAL_LOCALIZATION: frozenset({'open_target_relation', 'relation_description_or_verification', 'target_location'}), NEED_COUNT_SET: frozenset({'capacity_or_functional_count', 'direct_or_structural_entity_count', 'filtered_relation_attribute_count'}), NEED_OCR_LOGO_TEXT: frozenset({'owner_aware_text'}), NEED_FUNCTIONAL_WORLD: frozenset({'grounded_function_or_world'}), NEED_OPEN_DESCRIPTIVE: frozenset({'scene_or_content_description', 'unsupported_fallback'})}
EVIDENCE_NEEDS = {NEED_OBJECT_IDENTITY, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_SPATIAL_LOCALIZATION, NEED_COUNT_SET, NEED_OCR_LOGO_TEXT, NEED_FUNCTIONAL_WORLD, NEED_OPEN_DESCRIPTIVE}
CANONICAL_ROUTE_BY_NEED = {NEED_OBJECT_IDENTITY: ROUTE_OPEN_DESCRIPTIVE, NEED_ENTITY_ATTRIBUTE: ROUTE_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE: ROUTE_BINARY_EXISTENCE, NEED_SPATIAL_LOCALIZATION: ROUTE_ENTITY_RELATION_SPATIAL, NEED_COUNT_SET: ROUTE_COUNT_SET, NEED_OCR_LOGO_TEXT: ROUTE_OCR_LOGO_TEXT, NEED_FUNCTIONAL_WORLD: ROUTE_OPEN_DESCRIPTIVE, NEED_OPEN_DESCRIPTIVE: ROUTE_OPEN_DESCRIPTIVE}
OPEN_EQA_V0_OFFICIAL_CATEGORIES = frozenset({'attribute recognition', 'functional reasoning', 'object localization', 'object recognition', 'object state recognition', 'spatial understanding', 'world knowledge'})
OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS = frozenset({NEED_OBJECT_IDENTITY, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_SPATIAL_LOCALIZATION, NEED_COUNT_SET, NEED_OCR_LOGO_TEXT, NEED_FUNCTIONAL_WORLD})
OPEN_EQA_V0_FALLBACK_EVIDENCE_NEED = NEED_OPEN_DESCRIPTIVE
OPEN_EQA_V0_TEMPORAL_ROUTE_ENABLED = False
UNIT_OBJECT_CROP = 'object_crop'
UNIT_CANONICAL_ENTITY = 'canonical_entity_tracklet'
UNIT_SCENE_FRAME = 'scene_frame'
UNIT_GRAPH_NEIGHBORHOOD = 'graph_neighborhood'
UNIT_OWNER_ATTRIBUTE = 'owner_attribute'
UNIT_SPATIAL_PAIR = 'spatial_pair'
UNIT_CANDIDATE_SET = 'candidate_set'
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_COUNT_PATTERNS = ('how many', 'number of', 'count of')
_OCR_TERMS = {'brand', 'label', 'letter', 'letters', 'logo', 'logos', 'printed', 'read', 'reads', 'sign', 'text', 'word', 'words', 'written', 'writing'}
_ATTRIBUTE_TERMS = {'color', 'colored', 'colour', 'material', 'shape', 'size', 'texture', 'pattern', 'appearance'}
_STATE_TERMS = {'closed', 'empty', 'full', 'locked', 'off', 'on', 'open', 'opened', 'running', 'turned', 'unlocked'}
_SPATIAL_TERMS = {'above', 'behind', 'below', 'beneath', 'beside', 'between', 'front', 'inside', 'left', 'near', 'next', 'right', 'under', 'where'}
_FUNCTIONAL_PHRASES = ('can i', 'could i', 'how can', 'how do', 'is used for', 'should i', 'what can', 'what could', 'what do i use', 'what is used', 'what would', 'where should')
_SURFACE_CAPACITY_PHRASES = ('enough room', 'enough space', 'is there room', 'is there space', 'room on the', 'space for', 'space on the')
_V2_PLACEMENT_FEASIBILITY_PATTERN = re.compile('^(?:can|could) i (?:add|fit|hang|place|put|stand|throw)\\b')
_COUNT_ORDINAL_FILTER_PATTERN = re.compile('\\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\\b')
_SPATIAL_PREPOSITIONS = (' above ', ' behind ', ' below ', ' beside ', ' between ', ' in front of ', ' in the ', ' inside ', ' near ', ' next to ', ' on the ', ' under ')
_GENERIC_TARGETS = {'area', 'item', 'object', 'place', 'room', 'scene', 'something', 'thing'}
_CATEGORY_NEED = {'object recognition': NEED_OBJECT_IDENTITY, 'attribute recognition': NEED_ENTITY_ATTRIBUTE, 'object state recognition': NEED_EXISTENCE_OR_STATE, 'object localization': NEED_SPATIAL_LOCALIZATION, 'spatial understanding': NEED_SPATIAL_LOCALIZATION, 'functional reasoning': NEED_FUNCTIONAL_WORLD, 'world knowledge': NEED_FUNCTIONAL_WORLD}

@dataclass(frozen=True)
class VisualQueryIntent:
    text: str
    role: str
    evidence_unit: str = UNIT_OBJECT_CROP
    source: str = ''
    spatial_relation: str = ''
    target_terms: tuple[str, ...] = ()
    anchor_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class EvidenceDemand:
    primary_need: str
    route_id: str
    answer_shape: str
    target_terms: tuple[str, ...] = ()
    anchor_terms: tuple[str, ...] = ()
    spatial_relations: tuple[str, ...] = ()
    spatial_constraints: tuple[dict[str, str], ...] = ()
    open_target_slot: bool = False
    strategy_variant: str = ''
    attribute_dimensions: tuple[str, ...] = ()
    state_terms: tuple[str, ...] = ()
    constraint_terms: tuple[str, ...] = ()
    auxiliary_needs: tuple[str, ...] = ()
    preferred_units: tuple[str, ...] = ()
    query_intents: tuple[VisualQueryIntent, ...] = ()
    retrieval_policy: str = 'episode_scoped_object_and_context_recall'
    evidence_policy: str = 'graph_bound_visual_candidate_pack'
    requires_negative_coverage: bool = False
    allows_common_world_knowledge: bool = False
    confidence: float = 0.0
    source: str = ''
    contract: str = EVIDENCE_DEMAND_CONTRACT
    temporal_route_enabled: bool = False
    uses_gold_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['primary_target_terms'] = list(self.target_terms)
        payload['attribute_terms'] = list(self.attribute_dimensions)
        payload['event_constraints'] = list(self.constraint_terms)
        payload['time_requirements'] = {}
        payload['fallback_route'] = ROUTE_OPEN_DESCRIPTIVE
        payload['taxonomy_path'] = f'openeqa.non_temporal.{self.primary_need}'
        payload['evidence_family'] = f'{self.primary_need}_evidence'
        return payload

def open_eqa_v0_route_taxonomy() -> dict[str, Any]:
    """Return the versioned OpenEQA v0 evidence-routing contract."""
    return {'contract': OPEN_EQA_V0_ROUTE_TAXONOMY_CONTRACT, 'official_category_count': len(OPEN_EQA_V0_OFFICIAL_CATEGORIES), 'official_categories': sorted(OPEN_EQA_V0_OFFICIAL_CATEGORIES), 'active_route_count': len(OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS), 'active_routes': sorted(OPEN_EQA_V0_ACTIVE_EVIDENCE_NEEDS), 'fallback_route': OPEN_EQA_V0_FALLBACK_EVIDENCE_NEED, 'temporal_route_enabled': OPEN_EQA_V0_TEMPORAL_ROUTE_ENABLED}

def compile_evidence_demand(question: str, *, category: str | None=None, question_ir: dict[str, Any] | None=None, routing_profile: str=ROUTING_PROFILE_CATEGORY_FIRST_V1) -> EvidenceDemand:
    """Compile one primary evidence backbone without temporal routing."""
    ir = dict(question_ir or {})
    normalized_question = _normalize(question)
    tokens = set(_TOKEN_PATTERN.findall(normalized_question))
    normalized_category = _normalize(category or ir.get('category'))
    if routing_profile not in ROUTING_PROFILES:
        raise ValueError(f'Unknown OpenEQA routing profile {routing_profile!r}; expected one of {sorted(ROUTING_PROFILES)}')
    if routing_profile in IR_FIRST_ROUTING_PROFILES:
        select_need = _select_need_ir_first_v4
    elif routing_profile in EVIDENCE_DEMAND_VARIANT_PROFILES:
        select_need = _select_need_evidence_demand_v2
    else:
        select_need = _select_need
    routing_category = '' if routing_profile in IR_FIRST_ROUTING_PROFILES else normalized_category
    primary_need, source, confidence = select_need(question=normalized_question, tokens=tokens, category=routing_category, question_ir=ir)
    anchors = _ordered_unique([*_anchor_terms(ir), *_inferred_anchor_terms(normalized_question, ir)])
    targets = _target_terms(ir, anchors=anchors)
    spatial_constraints = _structured_spatial_constraints(ir, question=normalized_question)
    spatial_relations = _ordered_unique([*(row.get('relation', '') for row in spatial_constraints), *_spatial_relations(ir, question=normalized_question)])
    if routing_profile == ROUTING_PROFILE_IR_FIRST_V4:
        surface_capacity_query = _is_ir_surface_capacity_question_v4(normalized_question, question_ir=ir)
    elif routing_profile in EVIDENCE_DEMAND_VARIANT_PROFILES:
        surface_capacity_query = _is_surface_capacity_question_v2(normalized_question)
    else:
        surface_capacity_query = any((phrase in normalized_question for phrase in _SURFACE_CAPACITY_PHRASES))
    open_target_slot = _open_target_slot(primary_need, targets, anchors=anchors, surface_capacity_query=surface_capacity_query)
    attribute_dimensions = _attribute_dimensions(ir, tokens=tokens)
    states = _state_terms(ir, tokens=tokens)
    constraints = _constraint_terms(ir)
    strategy_variant = _strategy_variant(primary_need=primary_need, question=normalized_question, category=routing_category, question_ir=ir, routing_profile=routing_profile, surface_capacity_query=surface_capacity_query, open_target_slot=open_target_slot, anchors=anchors, spatial_relations=spatial_relations, attribute_dimensions=attribute_dimensions, states=states)
    auxiliary = _auxiliary_needs(primary_need=primary_need, tokens=tokens, anchors=anchors, attribute_dimensions=attribute_dimensions, states=states)
    units = _preferred_units(primary_need)
    intents = _build_query_intents(question=question, primary_need=primary_need, targets=targets, anchors=anchors, attribute_dimensions=attribute_dimensions, states=states, constraints=constraints, spatial_constraints=spatial_constraints)
    return EvidenceDemand(primary_need=primary_need, route_id=canonical_route_id_for_need(primary_need), answer_shape=_answer_shape(ir, question=normalized_question), target_terms=targets, anchor_terms=anchors, spatial_relations=spatial_relations, spatial_constraints=spatial_constraints, open_target_slot=open_target_slot, strategy_variant=strategy_variant, attribute_dimensions=attribute_dimensions, state_terms=states, constraint_terms=constraints, auxiliary_needs=auxiliary, preferred_units=units, query_intents=intents, retrieval_policy=_retrieval_policy(primary_need), evidence_policy=_evidence_policy(primary_need), requires_negative_coverage=primary_need == NEED_EXISTENCE_OR_STATE, allows_common_world_knowledge=primary_need == NEED_FUNCTIONAL_WORLD, confidence=confidence, source=source, contract=EVIDENCE_DEMAND_CONTRACT_V2 if routing_profile in EVIDENCE_DEMAND_VARIANT_PROFILES else EVIDENCE_DEMAND_CONTRACT, metadata={'category': routing_category, 'ignored_temporal_fields': _present_temporal_fields(ir), 'question_ir_contract': str(ir.get('schema_version') or ''), 'surface_capacity_query': surface_capacity_query, 'routing_profile': routing_profile, 'official_category_used_for_routing': routing_profile not in IR_FIRST_ROUTING_PROFILES, 'routing_input_contract': 'question_and_semantic_ir_only' if routing_profile in IR_FIRST_ROUTING_PROFILES else 'legacy_category_aware'})

def canonical_route_id_for_need(primary_need: str) -> str:
    return CANONICAL_ROUTE_BY_NEED.get(primary_need, ROUTE_OPEN_DESCRIPTIVE)

def question_route_decision_from_ir(question_ir: dict[str, Any] | None) -> dict[str, Any]:
    """Return the one persisted OpenEQA route/demand object, if valid."""
    if not isinstance(question_ir, dict):
        return {}
    route = question_ir.get('question_route')
    if not isinstance(route, dict):
        return {}
    if route.get('contract') not in EVIDENCE_DEMAND_CONTRACTS:
        return {}
    if route.get('uses_gold_answer') is not False:
        return {}
    primary_need = str(route.get('primary_need') or '').strip()
    route_id = str(route.get('route_id') or '').strip()
    if primary_need not in EVIDENCE_NEEDS:
        return {}
    if route_id != canonical_route_id_for_need(primary_need):
        return {}
    if route.get('temporal_route_enabled') is not False:
        return {}
    metadata = route.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    routing_profile = str(metadata.get('routing_profile') or '')
    if routing_profile in IR_FIRST_ROUTING_PROFILES:
        if question_ir.get('uses_gold_answer') is not False:
            return {}
        if question_ir.get('route_decision_is_persisted') is not True:
            return {}
        if question_ir.get('route_decision_runtime_recompile_allowed') is not False:
            return {}
        if str(question_ir.get('question_route_id') or '') != route_id:
            return {}
        if str(question_ir.get('category') or '').strip():
            return {}
        if metadata.get('official_category_used_for_routing') is not False:
            return {}
        if metadata.get('routing_input_contract') != 'question_and_semantic_ir_only':
            return {}
        if str(metadata.get('category') or '').strip():
            return {}
        if not isinstance(metadata.get('surface_capacity_query'), bool):
            return {}
        if str(route.get('source') or '').startswith('official_question_category'):
            return {}
        strategy_variant = str(route.get('strategy_variant') or '').strip()
        if strategy_variant not in IR_FIRST_STRATEGY_VARIANTS_BY_NEED[primary_need]:
            return {}
        if not str(route.get('route_strategy_handler') or '').strip():
            return {}
    return dict(route)

def question_route_decision_issues(question_ir: dict[str, Any] | None) -> list[str]:
    if question_route_decision_from_ir(question_ir):
        return []
    return [f'question_ir.question_route must be one valid {sorted(EVIDENCE_DEMAND_CONTRACTS)} object']

def visual_query_intents(demand: EvidenceDemand | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(demand, EvidenceDemand):
        return [item.to_dict() for item in demand.query_intents]
    rows = demand.get('query_intents', []) if isinstance(demand, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict) and row.get('text')]

def _select_need(*, question: str, tokens: set[str], category: str, question_ir: dict[str, Any]) -> tuple[str, str, float]:
    if tokens & _OCR_TERMS and tokens & {'read', 'reads', 'text', 'word', 'words', 'logo', 'brand', 'sign'}:
        return (NEED_OCR_LOGO_TEXT, 'question_text.ocr_logo', 0.96)
    if any((pattern in question for pattern in _COUNT_PATTERNS)):
        return (NEED_COUNT_SET, 'question_text.count', 0.97)
    if any((pattern in question for pattern in _SURFACE_CAPACITY_PHRASES)):
        return (NEED_EXISTENCE_OR_STATE, 'question_text.surface_capacity', 0.96)
    if any((phrase in question for phrase in _FUNCTIONAL_PHRASES)):
        return (NEED_FUNCTIONAL_WORLD, 'question_text.functional', 0.95)
    category_need = _CATEGORY_NEED.get(category)
    if category_need:
        return (category_need, 'official_question_category', 0.94)
    evidence_type = _normalize(question_ir.get('evidence_question_type'))
    if evidence_type:
        mapped = _need_from_evidence_type(evidence_type)
        if mapped:
            return (mapped, 'question_ir.evidence_question_type', 0.9)
    if question.startswith('where ') or tokens & _SPATIAL_TERMS:
        return (NEED_SPATIAL_LOCALIZATION, 'question_text.spatial', 0.82)
    if tokens & _ATTRIBUTE_TERMS:
        return (NEED_ENTITY_ATTRIBUTE, 'question_text.attribute', 0.82)
    if tokens & _STATE_TERMS or _normalize(question_ir.get('answer_type')) == 'binary':
        return (NEED_EXISTENCE_OR_STATE, 'question_text.state_or_binary', 0.78)
    if question.startswith(('what ', 'which ', 'who ')):
        return (NEED_OBJECT_IDENTITY, 'question_text.identity', 0.7)
    return (NEED_OPEN_DESCRIPTIVE, 'fallback', 0.45)

def _select_need_evidence_demand_v2(*, question: str, tokens: set[str], category: str, question_ir: dict[str, Any]) -> tuple[str, str, float]:
    if tokens & _OCR_TERMS and tokens & {'read', 'reads', 'text', 'word', 'words', 'logo', 'brand', 'sign'}:
        return (NEED_OCR_LOGO_TEXT, 'question_text.ocr_logo', 0.96)
    if any((pattern in question for pattern in _COUNT_PATTERNS)):
        return (NEED_COUNT_SET, 'question_text.count', 0.97)
    if _is_surface_capacity_question_v2(question):
        return (NEED_EXISTENCE_OR_STATE, 'question_text.surface_capacity', 0.96)
    category_need = _CATEGORY_NEED.get(category)
    if category_need:
        return (category_need, 'official_question_category', 0.95)
    evidence_type = _normalize(question_ir.get('evidence_question_type'))
    if evidence_type:
        mapped = _need_from_evidence_type(evidence_type)
        if mapped:
            return (mapped, 'question_ir.evidence_question_type', 0.9)
    operation_need = _need_from_question_ir_operation_v2(question_ir, question=question, category=category)
    if operation_need:
        return (operation_need, 'question_ir.operation', 0.88)
    if _is_unambiguous_functional_question_v2(question, tokens=tokens):
        return (NEED_FUNCTIONAL_WORLD, 'question_text.functional', 0.86)
    if question.startswith('where ') or tokens & _SPATIAL_TERMS:
        return (NEED_SPATIAL_LOCALIZATION, 'question_text.spatial', 0.82)
    if tokens & _ATTRIBUTE_TERMS:
        return (NEED_ENTITY_ATTRIBUTE, 'question_text.attribute', 0.82)
    if tokens & _STATE_TERMS or _normalize(question_ir.get('answer_type')) == 'binary':
        return (NEED_EXISTENCE_OR_STATE, 'question_text.state_or_binary', 0.78)
    if question.startswith(('what ', 'which ', 'who ')):
        return (NEED_OBJECT_IDENTITY, 'question_text.identity', 0.7)
    return (NEED_OPEN_DESCRIPTIVE, 'fallback', 0.45)

def _select_need_ir_first_v4(*, question: str, tokens: set[str], category: str, question_ir: dict[str, Any]) -> tuple[str, str, float]:
    """Select one route from question-only semantic IR without category input."""
    if category:
        raise ValueError('IR-first OpenEQA routing must not receive a category')
    operation = _normalize(question_ir.get('operation'))
    answer_type = _normalize(question_ir.get('answer_type'))
    attribute_dimension = _normalize(question_ir.get('attribute_dimension'))
    if _is_explicit_ocr_query_v4(question, question_ir=question_ir):
        return (NEED_OCR_LOGO_TEXT, 'question_ir+question_text.ocr', 0.98)
    if _is_ir_calendar_or_date_query_v4(question_ir):
        return (NEED_OCR_LOGO_TEXT, 'question_ir.calendar_date_text', 0.98)
    if _is_ir_room_or_scene_description_v4(question, question_ir=question_ir):
        return (NEED_OPEN_DESCRIPTIVE, 'question_ir.room_or_scene_description', 0.97)
    if operation == 'count_query':
        if answer_type == 'count':
            return (NEED_COUNT_SET, 'question_ir.operation+answer_type', 0.98)
        return (NEED_OBJECT_IDENTITY, 'question_ir.operation+answer_type', 0.9)
    if _is_ir_surface_capacity_question_v4(question, question_ir=question_ir):
        return (NEED_EXISTENCE_OR_STATE, 'question_ir.capacity', 0.97)
    if _is_ir_structured_functional_intent_v4(question_ir):
        return (NEED_FUNCTIONAL_WORLD, 'question_ir.structured_functional_intent', 0.94)
    if operation in {'current_state', 'state_query'}:
        return (NEED_EXISTENCE_OR_STATE, 'question_ir.operation', 0.97)
    if operation == 'existence_query':
        if answer_type == 'binary':
            return (NEED_EXISTENCE_OR_STATE, 'question_ir.operation+answer_type', 0.97)
        return (NEED_FUNCTIONAL_WORLD, 'question_ir.operation+answer_type', 0.86)
    if operation == 'attribute_query':
        if attribute_dimension == 'state':
            return (NEED_EXISTENCE_OR_STATE, 'question_ir.operation+attribute_dimension', 0.97)
        return (NEED_ENTITY_ATTRIBUTE, 'question_ir.operation+attribute_dimension', 0.96)
    if operation in {'location_query', 'navigation_direction', 'position_query', 'spatial_relation'}:
        return (NEED_SPATIAL_LOCALIZATION, 'question_ir.operation', 0.97)
    if operation == 'identity_query':
        return (NEED_OBJECT_IDENTITY, 'question_ir.operation', 0.97)
    if operation in {'functional_query', 'world_knowledge_query'}:
        return (NEED_FUNCTIONAL_WORLD, 'question_ir.operation', 0.97)
    if operation == 'descriptive_query':
        return (NEED_OPEN_DESCRIPTIVE, 'question_ir.operation', 0.95)
    if operation == 'unsupported':
        return (NEED_OPEN_DESCRIPTIVE, 'question_ir.unsupported_fallback', 0.35)
    if any((pattern in question for pattern in _COUNT_PATTERNS)):
        return (NEED_COUNT_SET, 'question_text.fallback.count', 0.72)
    if question.startswith('where ') or tokens & _SPATIAL_TERMS:
        return (NEED_SPATIAL_LOCALIZATION, 'question_text.fallback.spatial', 0.68)
    if tokens & _ATTRIBUTE_TERMS:
        return (NEED_ENTITY_ATTRIBUTE, 'question_text.fallback.attribute', 0.68)
    if tokens & _STATE_TERMS or answer_type == 'binary':
        return (NEED_EXISTENCE_OR_STATE, 'question_text.fallback.state', 0.64)
    if _is_unambiguous_functional_question_v2(question, tokens=tokens):
        return (NEED_FUNCTIONAL_WORLD, 'question_text.fallback.functional', 0.62)
    if question.startswith(('what ', 'which ', 'who ')):
        return (NEED_OBJECT_IDENTITY, 'question_text.fallback.identity', 0.58)
    return (NEED_OPEN_DESCRIPTIVE, 'question_ir.fallback', 0.3)

def _is_explicit_ocr_query_v4(question: str, *, question_ir: dict[str, Any]) -> bool:
    operation = _normalize(question_ir.get('operation'))
    dimension = _normalize(question_ir.get('attribute_dimension'))
    if operation == 'attribute_query' and dimension not in {'brand', 'logo', 'text', 'word', 'words'}:
        return False
    if dimension in {'brand', 'logo', 'text', 'word', 'words'}:
        return True
    if re.search('\\b(?:brand|logo)\\b', question):
        return True
    if re.search('\\b(?:word|words|text)\\b.*\\b(?:printed|written)\\b', question):
        return True
    if re.search('\\b(?:label|sign)\\b.*\\b(?:read|reads|say|says|written)\\b', question):
        return True
    return bool(re.search('\\bwhat (?:is|are) written\\b', question))

def _is_ir_room_or_scene_description_v4(question: str, *, question_ir: dict[str, Any]) -> bool:
    """Recognize scene/type questions before generic identity routing."""
    target_tokens = set(_TOKEN_PATTERN.findall(' '.join(_text_values(question_ir.get('target_entities')))))
    dimension = _normalize(question_ir.get('attribute_dimension'))
    focus = _normalize(question_ir.get('qa_focus'))
    operation = _normalize(question_ir.get('operation'))
    scene_tokens = {'room', 'scene', 'space', 'area', 'home', 'house'}
    if not target_tokens & scene_tokens:
        return False
    if dimension in {'type', 'kind', 'category', 'room_type', 'scene_type'}:
        return True
    if focus in {'descriptive', 'scene', 'scene_description'}:
        return True
    return operation == 'descriptive_query'

def _is_ir_calendar_or_date_query_v4(question_ir: dict[str, Any]) -> bool:
    """Route structured date/time readings to the OCR/text evidence family."""
    target_tokens = set(_TOKEN_PATTERN.findall(' '.join(_text_values(question_ir.get('target_entities')))))
    state_tokens = set(_TOKEN_PATTERN.findall(' '.join([*_text_values(question_ir.get('target_state')), *_text_values(question_ir.get('attribute_dimension'))])))
    owner_tokens = {'calendar', 'clock', 'watch', 'planner', 'schedule'}
    text_tokens = {'date', 'today', 'time', 'month', 'day', 'year', 'schedule'}
    return bool(target_tokens & owner_tokens and state_tokens & text_tokens)

def _is_ir_structured_functional_intent_v4(question_ir: dict[str, Any]) -> bool:
    """Recognize functional intent encoded in semantic state descriptors."""
    operation = _normalize(question_ir.get('operation'))
    if operation not in {'identity_query', 'location_query'}:
        return False
    semantic_tokens = set(_TOKEN_PATTERN.findall(' '.join([*_text_values(question_ir.get('target_state')), *_text_values(question_ir.get('target_description')), *_text_values(question_ir.get('target_descriptors'))])))
    if not semantic_tokens:
        return False
    recommendation_or_resource = {'available', 'recommended', 'suitable'}
    identity_affordance = {'display', 'intended', 'move', 'moved', 'moving', 'purpose', 'use', 'using'}
    location_affordance = {'cook', 'rest', 'sit', 'sitting', 'study'}
    if semantic_tokens & recommendation_or_resource:
        return True
    if operation == 'identity_query':
        return bool(semantic_tokens & identity_affordance)
    return bool(semantic_tokens & location_affordance)

def _is_ir_surface_capacity_question_v4(question: str, *, question_ir: dict[str, Any]) -> bool:
    operation = _normalize(question_ir.get('operation'))
    if operation not in {'functional_query', 'state_query', 'attribute_query'}:
        return False
    if _normalize(question_ir.get('answer_type')) != 'binary':
        return False
    semantic_state = _normalize(' '.join([*_text_values(question_ir.get('target_state')), *_text_values(question_ir.get('target_description')), *_text_values(question_ir.get('target_descriptors'))]))
    state_tokens = set(_TOKEN_PATTERN.findall(semantic_state))
    if state_tokens & {'capacity', 'fit', 'space'}:
        return True
    if re.search('\\bis there (?:(?:any|some)\\s+)?room\\b', question):
        return True
    return _is_surface_capacity_question_v2(question)

def _need_from_question_ir_operation_v2(question_ir: dict[str, Any], *, question: str, category: str) -> str | None:
    operation = _normalize(question_ir.get('operation'))
    aggregation = _normalize(question_ir.get('aggregation'))
    if operation == 'count_query' or aggregation in {'count', 'count_distinct'}:
        return NEED_COUNT_SET
    if operation in {'location_query', 'navigation_direction', 'position_query', 'spatial_relation'}:
        return NEED_SPATIAL_LOCALIZATION
    if operation == 'identity_query':
        return NEED_OBJECT_IDENTITY
    if operation in {'functional_query', 'world_knowledge_query'}:
        return NEED_FUNCTIONAL_WORLD
    if operation in {'current_state', 'state_query'}:
        return NEED_EXISTENCE_OR_STATE
    if operation == 'existence_query':
        if _is_unambiguous_functional_question_v2(question, tokens=set(_TOKEN_PATTERN.findall(question))):
            return NEED_FUNCTIONAL_WORLD
        return NEED_EXISTENCE_OR_STATE
    if operation == 'attribute_query':
        return {'attribute recognition': NEED_ENTITY_ATTRIBUTE, 'functional reasoning': NEED_FUNCTIONAL_WORLD, 'object recognition': NEED_OBJECT_IDENTITY, 'object state recognition': NEED_EXISTENCE_OR_STATE, 'spatial understanding': NEED_SPATIAL_LOCALIZATION, 'world knowledge': NEED_FUNCTIONAL_WORLD}.get(category)
    return None

def _is_unambiguous_functional_question_v2(question: str, *, tokens: set[str]) -> bool:
    if question.startswith(('where can i find', 'where can i see', 'where should i look')):
        return False
    if question.startswith(('what can be found', 'what can be seen')):
        return False
    if 'what would' in question and tokens & _SPATIAL_TERMS:
        return False
    return any((phrase in question for phrase in _FUNCTIONAL_PHRASES))

def _is_surface_capacity_question_v2(question: str) -> bool:
    return any((phrase in question for phrase in _SURFACE_CAPACITY_PHRASES)) or bool(_V2_PLACEMENT_FEASIBILITY_PATTERN.search(question))

def _need_from_evidence_type(value: str) -> str | None:
    if 'ocr' in value or 'logo' in value:
        return NEED_OCR_LOGO_TEXT
    if '.count' in value:
        return NEED_COUNT_SET
    if '.attribute' in value:
        return NEED_ENTITY_ATTRIBUTE
    if '.existence' in value or '.state' in value:
        return NEED_EXISTENCE_OR_STATE
    if '.spatial' in value or '.localization' in value or '.relation' in value:
        return NEED_SPATIAL_LOCALIZATION
    if '.functional' in value or '.world_knowledge' in value:
        return NEED_FUNCTIONAL_WORLD
    if '.ego' in value or '.direction' in value:
        return NEED_SPATIAL_LOCALIZATION
    if '.identity' in value:
        return NEED_OBJECT_IDENTITY
    if 'open_descriptive' in value:
        return NEED_OPEN_DESCRIPTIVE
    return None

def _strategy_variant(*, primary_need: str, question: str, category: str, question_ir: dict[str, Any], routing_profile: str, surface_capacity_query: bool, open_target_slot: bool, anchors: tuple[str, ...], spatial_relations: tuple[str, ...], attribute_dimensions: tuple[str, ...], states: tuple[str, ...]) -> str:
    operation = _normalize(question_ir.get('operation'))
    if primary_need == NEED_SPATIAL_LOCALIZATION:
        if open_target_slot and (anchors or spatial_relations):
            return 'open_target_relation'
        if category == 'object localization' or operation in {'location_query', 'position_query'}:
            return 'target_location'
        if operation == 'navigation_direction' and _normalize(question_ir.get('answer_type')) != 'binary':
            return 'target_location'
        return 'relation_description_or_verification'
    if primary_need == NEED_OBJECT_IDENTITY:
        if open_target_slot or anchors or spatial_relations:
            return 'anchor_bound_open_target_identity'
        return 'direct_identity'
    if primary_need == NEED_COUNT_SET:
        if _is_capacity_or_functional_count(question, category=category):
            return 'capacity_or_functional_count'
        if attribute_dimensions or _COUNT_ORDINAL_FILTER_PATTERN.search(question):
            return 'filtered_relation_attribute_count'
        return 'direct_or_structural_entity_count'
    if primary_need == NEED_EXISTENCE_OR_STATE:
        if routing_profile in STATE_SUBVARIANT_ROUTING_PROFILES:
            return _state_strategy_variant(question=question, question_ir=question_ir, routing_profile=routing_profile, states=states, surface_capacity_query=surface_capacity_query)
        return STATE_VARIANT_LEGACY
    return {NEED_ENTITY_ATTRIBUTE: 'owner_attribute', NEED_FUNCTIONAL_WORLD: 'grounded_function_or_world', NEED_OCR_LOGO_TEXT: 'owner_aware_text', NEED_OPEN_DESCRIPTIVE: 'scene_or_content_description' if operation == 'descriptive_query' or _is_ir_room_or_scene_description_v4(question, question_ir=question_ir) else 'unsupported_fallback'}.get(primary_need, '')

def _state_strategy_variant(*, question: str, question_ir: dict[str, Any], routing_profile: str, states: tuple[str, ...], surface_capacity_query: bool) -> str:
    if surface_capacity_query:
        return STATE_VARIANT_SURFACE_CAPACITY
    operation = _normalize(question_ir.get('operation'))
    if operation == 'existence_query' or question.startswith(('are there ', 'is there ')):
        return STATE_VARIANT_EXISTENCE
    if routing_profile == ROUTING_PROFILE_IR_FIRST_V4:
        semantic_state = _normalize(' '.join([*_text_values(question_ir.get('target_state')), *_text_values(question_ir.get('state_descriptors'))]))
        state_tokens = set(_TOKEN_PATTERN.findall(' '.join([semantic_state, *states])))
        if state_tokens & {'closed', 'open', 'shut'}:
            return STATE_VARIANT_OPEN_CLOSED
        if state_tokens & {'clean', 'dirty', 'messy', 'tidy'}:
            return STATE_VARIANT_CLEANLINESS
        if state_tokens & {'empty', 'filled', 'full', 'occupied'}:
            return STATE_VARIANT_FULL_EMPTY
        if _is_ir_power_state_question_v4(question=question, semantic_state=semantic_state):
            return STATE_VARIANT_POWER
        return STATE_VARIANT_OTHER
    state_text = _normalize(' '.join([*_text_values(question_ir.get('target_state')), *_text_values(question_ir.get('state_descriptors')), *states]))
    state_tokens = set(_TOKEN_PATTERN.findall(state_text))
    if state_tokens & {'closed', 'open', 'shut'}:
        return STATE_VARIANT_OPEN_CLOSED
    if state_tokens & {'lit', 'off', 'on', 'powered', 'switched', 'unlit'}:
        return STATE_VARIANT_POWER
    if state_tokens & {'clean', 'dirty', 'messy', 'tidy'}:
        return STATE_VARIANT_CLEANLINESS
    if state_tokens & {'empty', 'filled', 'full', 'occupied'}:
        return STATE_VARIANT_FULL_EMPTY
    return STATE_VARIANT_OTHER

def _is_ir_power_state_question_v4(*, question: str, semantic_state: str) -> bool:
    state_tokens = set(_TOKEN_PATTERN.findall(semantic_state))
    if state_tokens & {'lit', 'powered', 'switched', 'unlit'}:
        return True
    if semantic_state in {'off', 'off or on', 'on', 'on or off'}:
        return True
    explicit_transition = '\\b(?:powered|switched|turned)\\s+(?:off|on)\\b'
    if re.search(explicit_transition, semantic_state):
        return True
    normalized_question = question.rstrip(' ?.!')
    return bool(re.search(explicit_transition, normalized_question) or re.search('\\b(?:off|on)$', normalized_question))

def _is_capacity_or_functional_count(question: str, *, category: str) -> bool:
    if category == 'functional reasoning':
        return True
    phrases = ('allowed to sleep', 'can be plugged', 'can sit', 'choices do i have', 'does the mr coffee make', 'intended to use', 'maximum number', 'roommate')
    return any((phrase in question for phrase in phrases))

def _target_terms(ir: dict[str, Any], *, anchors: tuple[str, ...]) -> tuple[str, ...]:
    anchor_keys = {_normalize(value) for value in anchors}
    values = [*_text_values(ir.get('target_entities')), *_text_values(ir.get('target_description')), *_text_values(ir.get('target_descriptors'))]
    filtered = [value for value in values if _normalize(value) not in anchor_keys]
    non_generic = [value for value in filtered if _normalize(value) not in _GENERIC_TARGETS]
    return _ordered_unique(non_generic)

def _anchor_terms(ir: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in _items(ir.get('spatial_constraints')):
        if isinstance(item, dict):
            values.extend(_text_values(item.get('target')))
            values.extend(_text_values(item.get('anchor')))
    for item in _items(ir.get('hard_constraints')):
        if not isinstance(item, dict) or _normalize(item.get('kind')) != 'spatial':
            continue
        values.extend(_text_values(item.get('anchor_entities')))
    return _ordered_unique(values)

def _inferred_anchor_terms(question: str, ir: dict[str, Any]) -> tuple[str, ...]:
    if not any((phrase in f' {question} ' for phrase in _SPATIAL_PREPOSITIONS)):
        return ()
    targets = _ordered_unique(_text_values(ir.get('target_entities')))
    if len(targets) <= 1:
        return ()
    return tuple(targets[1:])

def _attribute_dimensions(ir: dict[str, Any], *, tokens: set[str]) -> tuple[str, ...]:
    values = [*_text_values(ir.get('attribute_dimension')), *_text_values(ir.get('attribute_dimension_aliases'))]
    values.extend(sorted(tokens & _ATTRIBUTE_TERMS))
    return _ordered_unique(values)

def _state_terms(ir: dict[str, Any], *, tokens: set[str]) -> tuple[str, ...]:
    values = [*_text_values(ir.get('target_state')), *_text_values(ir.get('state_descriptors'))]
    values.extend(sorted(tokens & _STATE_TERMS))
    return _ordered_unique(values)

def _constraint_terms(ir: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ('hard_constraints', 'spatial_constraints', 'event_constraints'):
        for item in _items(ir.get(key)):
            if isinstance(item, dict):
                values.extend(_text_values(item.get('text')))
                relation = _clean_text(item.get('relation'))
                target = _clean_text(item.get('target'))
                if relation and target:
                    values.append(f'{relation} {target}')
            else:
                values.extend(_text_values(item))
    return _ordered_unique(values)

def _spatial_relations(ir: dict[str, Any], *, question: str) -> tuple[str, ...]:
    values: list[str] = []
    for item in _items(ir.get('spatial_constraints')):
        if isinstance(item, dict):
            raw_relation = _normalize(item.get('relation'))
            if raw_relation.replace('_', ' ') == 'side of':
                if 'left' in question:
                    raw_relation = 'left of'
                elif 'right' in question:
                    raw_relation = 'right of'
            relation = _normalize_spatial_relation(raw_relation)
            if relation:
                values.append(relation)
    for phrase, relation in (('in between', 'between'), ('between', 'between'), ('to the left of', 'left_of'), ('left of', 'left_of'), ('left side', 'left_of'), ('to the right of', 'right_of'), ('right of', 'right_of'), ('right side', 'right_of'), ('above', 'above'), ('below', 'below'), ('under', 'below'), ('next to', 'near'), ('beside', 'near'), ('near', 'near')):
        if phrase in f' {question} ':
            values.append(relation)
    if not values:
        if ' on ' in f' {question} ':
            values.append('on')
        if ' in ' in f' {question} ':
            values.append('inside')
    return _ordered_unique(values)

def _structured_spatial_constraints(ir: dict[str, Any], *, question: str) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    directional_relation = ''
    if 'left' in question:
        directional_relation = 'left_of'
    elif 'right' in question:
        directional_relation = 'right_of'
    for item in _items(ir.get('spatial_constraints')):
        if not isinstance(item, dict):
            continue
        anchor = _clean_text(item.get('target') or item.get('anchor'))
        relation = _normalize_spatial_relation(item.get('relation'))
        if directional_relation and anchor and (_normalize(anchor) in question):
            if relation in {'inside', 'side_of', 'right_of', 'left_of'}:
                relation = directional_relation
        if anchor and relation:
            rows.append({'anchor': anchor, 'relation': relation})
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (_normalize(row['anchor']), row['relation'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return tuple(unique)

def _normalize_spatial_relation(value: Any) -> str:
    relation = _normalize(value).replace('-', ' ').replace('_', ' ')
    if not relation:
        return ''
    if 'between' in relation:
        return 'between'
    if 'left' in relation:
        return 'left_of'
    if 'right' in relation or relation == 'side of':
        return 'right_of'
    if relation in {'above', 'over', 'on top of'}:
        return 'above'
    if relation in {'below', 'beneath', 'under'}:
        return 'below'
    if relation in {'near', 'next to', 'beside', 'by'}:
        return 'near'
    if relation in {'in', 'inside', 'within'}:
        return 'inside'
    if relation in {'on', 'on the'}:
        return 'on'
    return relation.replace(' ', '_')

def _open_target_slot(primary_need: str, targets: tuple[str, ...], *, anchors: tuple[str, ...], surface_capacity_query: bool) -> bool:
    if surface_capacity_query:
        return True
    if primary_need not in {NEED_OBJECT_IDENTITY, NEED_SPATIAL_LOCALIZATION}:
        return False
    if not targets:
        return bool(anchors) or primary_need == NEED_OBJECT_IDENTITY
    generic_suffixes = tuple((f' {value}' for value in sorted(_GENERIC_TARGETS)))
    return all((_normalize(target) in _GENERIC_TARGETS or _normalize(target).endswith(generic_suffixes) for target in targets))

def _auxiliary_needs(*, primary_need: str, tokens: set[str], anchors: tuple[str, ...], attribute_dimensions: tuple[str, ...], states: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    if anchors and primary_need != NEED_SPATIAL_LOCALIZATION:
        values.append(NEED_SPATIAL_LOCALIZATION)
    if attribute_dimensions and primary_need != NEED_ENTITY_ATTRIBUTE:
        values.append(NEED_ENTITY_ATTRIBUTE)
    if states and primary_need != NEED_EXISTENCE_OR_STATE:
        values.append(NEED_EXISTENCE_OR_STATE)
    if tokens & _OCR_TERMS and primary_need != NEED_OCR_LOGO_TEXT:
        values.append(NEED_OCR_LOGO_TEXT)
    return _ordered_unique(values)

def _preferred_units(primary_need: str) -> tuple[str, ...]:
    mapping = {NEED_OBJECT_IDENTITY: (UNIT_OBJECT_CROP, UNIT_CANONICAL_ENTITY, UNIT_GRAPH_NEIGHBORHOOD), NEED_ENTITY_ATTRIBUTE: (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE, UNIT_CANONICAL_ENTITY), NEED_EXISTENCE_OR_STATE: (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE, UNIT_CANONICAL_ENTITY), NEED_SPATIAL_LOCALIZATION: (UNIT_OBJECT_CROP, UNIT_SPATIAL_PAIR, UNIT_SCENE_FRAME, UNIT_GRAPH_NEIGHBORHOOD), NEED_COUNT_SET: (UNIT_OBJECT_CROP, UNIT_CANDIDATE_SET, UNIT_CANONICAL_ENTITY), NEED_OCR_LOGO_TEXT: (UNIT_OBJECT_CROP, UNIT_OWNER_ATTRIBUTE, UNIT_CANONICAL_ENTITY), NEED_FUNCTIONAL_WORLD: (UNIT_OBJECT_CROP, UNIT_SCENE_FRAME, UNIT_GRAPH_NEIGHBORHOOD), NEED_OPEN_DESCRIPTIVE: (UNIT_OBJECT_CROP, UNIT_SCENE_FRAME, UNIT_GRAPH_NEIGHBORHOOD)}
    return mapping[primary_need]

def _build_query_intents(*, question: str, primary_need: str, targets: tuple[str, ...], anchors: tuple[str, ...], attribute_dimensions: tuple[str, ...], states: tuple[str, ...], constraints: tuple[str, ...], spatial_constraints: tuple[dict[str, str], ...]) -> tuple[VisualQueryIntent, ...]:
    intents: list[VisualQueryIntent] = []

    def add(text: str, *, role: str, unit: str, source: str, spatial_relation: str='') -> None:
        cleaned = _clean_text(text)
        if not cleaned:
            return
        intents.append(VisualQueryIntent(text=cleaned, role=role, evidence_unit=unit, source=source, spatial_relation=spatial_relation, target_terms=targets, anchor_terms=anchors))
    modifiers = [*attribute_dimensions, *states]
    for target in targets[:4]:
        add(target, role='target', unit=UNIT_OBJECT_CROP, source='target_identity')
        if modifiers:
            add(' '.join([target, *modifiers[:2]]), role='target', unit=UNIT_OBJECT_CROP, source='target_attribute_or_state')
        if primary_need in {NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_OCR_LOGO_TEXT}:
            add(' '.join([target, *modifiers[:2]]), role='owner', unit=UNIT_OWNER_ATTRIBUTE, source='canonical_owner_entity')
        elif primary_need == NEED_COUNT_SET:
            add(target, role='candidate_set', unit=UNIT_CANDIDATE_SET, source='canonical_count_entities')
        elif primary_need == NEED_OBJECT_IDENTITY:
            add(target, role='target', unit=UNIT_CANONICAL_ENTITY, source='canonical_entity_identity')
    for anchor in anchors[:4]:
        relation = next((row.get('relation', '') for row in spatial_constraints if _normalize(row.get('anchor')) == _normalize(anchor)), '')
        add(anchor, role='anchor', unit=UNIT_OBJECT_CROP, source='spatial_anchor', spatial_relation=relation)
    if anchors or not targets:
        add(_compact_visual_question(question), role='target', unit=UNIT_OBJECT_CROP, source='question_bound_target')
    if primary_need == NEED_OCR_LOGO_TEXT:
        for target in targets[:2] or ('sign',):
            add(f'{target} visible text logo writing', role='target', unit=UNIT_OBJECT_CROP, source='ocr_logo')
    elif primary_need == NEED_COUNT_SET:
        for target in targets[:2]:
            add(f'{target} object instance', role='target', unit=UNIT_OBJECT_CROP, source='count_set')
    elif primary_need in {NEED_SPATIAL_LOCALIZATION, NEED_FUNCTIONAL_WORLD, NEED_OPEN_DESCRIPTIVE}:
        compact = _compact_visual_question(question)
        add(compact, role='context', unit=UNIT_SCENE_FRAME, source='question_context')
    if constraints and (not targets):
        add(' '.join(constraints[:2]), role='context', unit=UNIT_OBJECT_CROP, source='constraints')
    if not intents:
        add(_compact_visual_question(question), role='target', unit=UNIT_OBJECT_CROP, source='question_fallback')
    unique: list[VisualQueryIntent] = []
    seen: set[tuple[str, str, str]] = set()
    for intent in intents:
        key = (_normalize(intent.text), intent.role, intent.evidence_unit)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        unique.append(intent)
    return tuple(unique[:10])

def _retrieval_policy(primary_need: str) -> str:
    return {NEED_OBJECT_IDENTITY: 'target_crop_then_graph_neighborhood', NEED_ENTITY_ATTRIBUTE: 'target_crop_then_owner_attribute', NEED_EXISTENCE_OR_STATE: 'episode_coverage_then_owner_state', NEED_SPATIAL_LOCALIZATION: 'target_anchor_crop_then_coframe_spatial_pair', NEED_COUNT_SET: 'episode_scoped_deduplicated_candidate_set', NEED_OCR_LOGO_TEXT: 'owner_bound_text_logo_attribute', NEED_FUNCTIONAL_WORLD: 'ground_object_identity_then_common_function', NEED_OPEN_DESCRIPTIVE: 'broad_graph_bound_visual_candidates'}[primary_need]

def _evidence_policy(primary_need: str) -> str:
    return {NEED_OBJECT_IDENTITY: 'identity_candidate_pack', NEED_ENTITY_ATTRIBUTE: 'owner_attribute_candidate_pack', NEED_EXISTENCE_OR_STATE: 'positive_or_coverage_qualified_state_pack', NEED_SPATIAL_LOCALIZATION: 'target_anchor_relation_pack', NEED_COUNT_SET: 'deduplicated_count_candidate_set', NEED_OCR_LOGO_TEXT: 'owner_aware_ocr_logo_pack', NEED_FUNCTIONAL_WORLD: 'grounded_identity_function_pack', NEED_OPEN_DESCRIPTIVE: 'ger_style_graph_visual_pack'}[primary_need]

def _answer_shape(ir: dict[str, Any], *, question: str) -> str:
    answer_type = _normalize(ir.get('answer_type'))
    if answer_type:
        return answer_type
    if question.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ', 'can ')):
        return 'binary'
    if any((pattern in question for pattern in _COUNT_PATTERNS)):
        return 'count'
    if question.startswith('where '):
        return 'location_text'
    return 'text'

def _present_temporal_fields(ir: dict[str, Any]) -> list[str]:
    fields = []
    for key in ('temporal_relation', 'temporal_constraints', 'time_requirements'):
        if ir.get(key) not in (None, '', [], {}):
            fields.append(key)
    return fields

def _compact_visual_question(question: str) -> str:
    text = _normalize(question)
    tokens = [token for token in _TOKEN_PATTERN.findall(text) if token not in {'a', 'an', 'are', 'did', 'do', 'does', 'i', 'is', 'of', 'the', 'there', 'to', 'was', 'were', 'what', 'which', 'who'}]
    return ' '.join(tokens[:14])

def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]

def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target', 'anchor', 'value', 'name'):
            values.extend(_text_values(value.get(key)))
        return values
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _text_values(item)]
    return []

def _ordered_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = _normalize(text)
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)

def _clean_text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _normalize(value: Any) -> str:
    return _clean_text(value).lower()
