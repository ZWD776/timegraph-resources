from __future__ import annotations
from dataclasses import asdict, dataclass, field
import re
from typing import Any
from vlkg.core.graph_retrieval.question_compiler import QuestionPlan, compile_question
QUESTION_ROUTE_CONTRACT = 'navqa_question_route_v1'
QUESTION_EVIDENCE_TAXONOMY_CONTRACT = 'navqa_question_evidence_taxonomy_v1'
QUESTION_ROUTER_IR_QA_ID = 'IR-QA-F-question-router-v1'
ROUTE_TEMPORAL_POINT = 'temporal_point'
ROUTE_TEMPORAL_INTERVAL = 'temporal_interval'
ROUTE_ENTITY_ATTRIBUTE = 'entity_attribute'
ROUTE_ENTITY_RELATION_SPATIAL = 'entity_relation_spatial'
ROUTE_COUNT_SET = 'count_set'
ROUTE_BINARY_EXISTENCE = 'binary_existence'
ROUTE_OCR_LOGO_TEXT = 'ocr_logo_text'
ROUTE_EGO_MOTION_SIDE_TURN = 'ego_motion_side_turn'
ROUTE_OPEN_DESCRIPTIVE = 'open_descriptive'
ROUTE_IDS = {ROUTE_TEMPORAL_POINT, ROUTE_TEMPORAL_INTERVAL, ROUTE_ENTITY_ATTRIBUTE, ROUTE_ENTITY_RELATION_SPATIAL, ROUTE_COUNT_SET, ROUTE_BINARY_EXISTENCE, ROUTE_OCR_LOGO_TEXT, ROUTE_EGO_MOTION_SIDE_TURN, ROUTE_OPEN_DESCRIPTIVE}
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_COUNT_TERMS = {'count', 'counts', 'number', 'many'}
_OCR_TERMS = {'letter', 'letters', 'logo', 'logos', 'printed', 'read', 'reads', 'say', 'says', 'text', 'word', 'words', 'written', 'writing'}
_OCR_SIGN_INTENT_TERMS = {'letter', 'letters', 'logo', 'logos', 'name', 'printed', 'read', 'reads', 'say', 'says', 'text', 'word', 'words', 'written', 'writing'}
_SPATIAL_TERMS = {'adjacent', 'alongside', 'beside', 'between', 'behind', 'near', 'nearby', 'next'}
_SPATIAL_PHRASES = {'close to', 'in front of', 'left of', 'next to', 'right of'}
_EGO_TERMS = {'direction', 'driving', 'heading', 'left', 'right', 'side', 'turn', 'turned', 'turning'}
_EGO_ACTOR_TERMS = {'driving', 'i', 'me', 'my', 'robot', 'turn', 'turned', 'turning', 'walking', 'you'}
_ATTRIBUTE_TERMS = {'appearance', 'color', 'colored', 'colour', 'look', 'looks', 'name', 'state', 'wearing', 'wore'}

@dataclass(frozen=True)
class QuestionRoute:
    route_id: str
    answer_shape: str
    primary_target_terms: tuple[str, ...] = ()
    attribute_terms: tuple[str, ...] = ()
    event_constraints: tuple[str, ...] = ()
    spatial_constraints: tuple[str, ...] = ()
    time_requirements: dict[str, Any] = field(default_factory=dict)
    retrieval_policy: str = 'hybrid_candidate_pack'
    evidence_policy: str = 'common_candidate_pack'
    confidence: float = 0.0
    fallback_route: str = ROUTE_OPEN_DESCRIPTIVE
    contract: str = QUESTION_ROUTE_CONTRACT
    source: str = ''
    compiler_family: str = ''
    compiler_answer_type: str = ''
    ir_operation: str = ''
    ir_answer_type: str = ''
    official_question_type: str = ''
    conflicts: tuple[str, ...] = ()
    route_signals: tuple[str, ...] = ()
    taxonomy_contract: str = QUESTION_EVIDENCE_TAXONOMY_CONTRACT
    taxonomy_path: str = ''
    taxonomy_root: str = ''
    taxonomy_domain: str = ''
    taxonomy_task: str = ''
    evidence_family: str = ''
    taxonomy_source: str = 'question_route_derived'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def route_question(question: str, *, official_question_type: str | None=None, question_ir: dict[str, Any] | None=None, compiler_plan: QuestionPlan | None=None, domain_profile: str | None=None, category: str | None=None) -> QuestionRoute:
    qtype = _clean_scalar(official_question_type).lower()
    ir = question_ir if isinstance(question_ir, dict) else {}
    compiler = compiler_plan or compile_question(question, answer_type=qtype or None, category=category, domain_profile=domain_profile)
    text = _normalize(question)
    tokens = set(_TOKEN_PATTERN.findall(text))
    route_id, source, confidence, signals = _select_route(text=text, tokens=tokens, official_question_type=qtype, question_ir=ir, compiler=compiler)
    compiler_route = _route_from_compiler(compiler)
    conflicts = _route_conflicts(selected_route=route_id, compiler_route=compiler_route, official_question_type=qtype, question_ir=ir)
    taxonomy = _taxonomy_for_route(route_id, question_ir=ir, compiler=compiler)
    return QuestionRoute(route_id=route_id, answer_shape=_answer_shape(qtype=qtype, question_ir=ir, compiler=compiler), primary_target_terms=_ordered_unique([*_iter_text_values(ir.get('target_entities')), *_iter_text_values(ir.get('target_state')), *compiler.target_entities]), attribute_terms=_ordered_unique([*_iter_text_values(ir.get('attribute_dimension')), *_iter_text_values(ir.get('attribute_dimension_aliases')), *compiler.attributes]), event_constraints=_ordered_unique([*_iter_text_values(ir.get('event_constraints')), *_iter_text_values(ir.get('hard_constraints')), *compiler.events, *compiler.temporal_constraints]), spatial_constraints=_ordered_unique([*_iter_text_values(ir.get('spatial_constraints')), *compiler.spatial_constraints]), time_requirements=_time_requirements(route_id, question_ir=ir, compiler=compiler), retrieval_policy=_retrieval_policy(route_id), evidence_policy=_evidence_policy(route_id), confidence=confidence, fallback_route=ROUTE_OPEN_DESCRIPTIVE if route_id != ROUTE_OPEN_DESCRIPTIVE else 'rule_text_retrieval', source=source, compiler_family=compiler.query_family, compiler_answer_type=compiler.answer_type, ir_operation=_clean_scalar(ir.get('operation')).lower(), ir_answer_type=_clean_scalar(ir.get('answer_type')).lower(), official_question_type=qtype, conflicts=conflicts, route_signals=tuple(signals), taxonomy_path=taxonomy['taxonomy_path'], taxonomy_root=taxonomy['taxonomy_root'], taxonomy_domain=taxonomy['taxonomy_domain'], taxonomy_task=taxonomy['taxonomy_task'], evidence_family=taxonomy['evidence_family'], taxonomy_source=taxonomy['taxonomy_source'])

def question_route_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get('question_route')
    return value if isinstance(value, dict) else {}

def attach_question_route_to_ir(question_ir: dict[str, Any] | None, route: QuestionRoute | dict[str, Any] | None) -> dict[str, Any] | None:
    if route is None:
        return question_ir
    route_dict = route.to_dict() if isinstance(route, QuestionRoute) else dict(route)
    if not route_dict:
        return question_ir
    updated = dict(question_ir or {})
    updated['question_route'] = route_dict
    return updated

def _select_route(*, text: str, tokens: set[str], official_question_type: str, question_ir: dict[str, Any], compiler: QuestionPlan) -> tuple[str, str, float, list[str]]:
    operation = _clean_scalar(question_ir.get('operation')).lower()
    answer_type = _clean_scalar(question_ir.get('answer_type')).lower()
    aggregation = _clean_scalar(question_ir.get('aggregation')).lower()
    rendering = _clean_scalar(question_ir.get('rendering')).lower()
    focus = _clean_scalar(question_ir.get('qa_focus')).lower()
    ir_route = question_ir.get('question_route')
    if isinstance(ir_route, dict):
        route_id = _clean_scalar(ir_route.get('route_id')).lower()
        if route_id in ROUTE_IDS:
            return (route_id, 'llm_question_ir.question_route', 0.98, ['ir_question_route'])
    evidence_question_type = _clean_scalar(question_ir.get('evidence_question_type') or question_ir.get('question_type_taxonomy') or question_ir.get('taxonomy_path')).lower()
    route_from_type = _route_from_evidence_question_type(evidence_question_type)
    if route_from_type:
        return (route_from_type, 'llm_question_ir.evidence_question_type', 0.96, [f'ir_evidence_question_type:{evidence_question_type}'])
    ir_text = _normalize(' '.join([operation, aggregation, rendering, focus, *_iter_text_values(question_ir.get('attribute_dimension')), *_iter_text_values(question_ir.get('evidence_requirements')), *_iter_text_values(question_ir.get('graph_information_needs'))]))
    all_text = f'{text} {ir_text}'.strip()
    all_tokens = tokens | set(_TOKEN_PATTERN.findall(ir_text))
    explicit_route = _explicit_ir_route(operation=operation, answer_type=answer_type, aggregation=aggregation, rendering=rendering, focus=focus, text=all_text, tokens=all_tokens, question_ir=question_ir)
    if explicit_route:
        route, signal = explicit_route
        return (route, 'llm_question_ir', 0.95, [signal])
    if official_question_type == 'binary':
        return (ROUTE_BINARY_EXISTENCE, 'official_binary_shape', 0.82, ['official_binary'])
    compiler_route = _route_from_compiler(compiler)
    if compiler_route != ROUTE_OPEN_DESCRIPTIVE:
        return (compiler_route, 'question_compiler', 0.78, [f'compiler:{compiler.query_family}'])
    keyword_route = _keyword_route(text=text, tokens=tokens)
    if keyword_route:
        route, signal = keyword_route
        return (route, 'question_text_keywords', 0.72, [signal])
    return (ROUTE_OPEN_DESCRIPTIVE, 'fallback', 0.45, ['fallback_open_descriptive'])

def _explicit_ir_route(*, operation: str, answer_type: str, aggregation: str, rendering: str, focus: str, text: str, tokens: set[str], question_ir: dict[str, Any]) -> tuple[str, str] | None:
    if operation == 'interval' or answer_type == 'duration' or aggregation == 'interval':
        return (ROUTE_TEMPORAL_INTERVAL, 'ir_interval')
    if rendering == 'duration_minutes':
        return (ROUTE_TEMPORAL_INTERVAL, 'ir_duration_rendering')
    if answer_type != 'time' and (aggregation in {'count', 'count_distinct'} or 'how many' in text or bool(tokens & _COUNT_TERMS)):
        return (ROUTE_COUNT_SET, 'ir_count')
    if answer_type == 'time' or operation in {'onset', 'offset', 'occurrence', 'event_time'}:
        return (ROUTE_TEMPORAL_POINT, 'ir_temporal_point')
    if _looks_like_ocr_request(text, tokens=tokens, question_ir=question_ir):
        return (ROUTE_OCR_LOGO_TEXT, 'ir_ocr_logo_text')
    if focus in {'ego_motion', 'ego_direction', 'navigation_direction'} or operation in {'turn', 'ego_motion', 'navigation_direction'} or _looks_like_ego_request(text, tokens=tokens, question_ir=question_ir):
        return (ROUTE_EGO_MOTION_SIDE_TURN, 'ir_ego_motion')
    if answer_type == 'binary' or operation in {'existence_query', 'current_state'}:
        return (ROUTE_BINARY_EXISTENCE, 'ir_binary')
    if operation == 'attribute_query' or _has_ir_attribute(question_ir):
        return (ROUTE_ENTITY_ATTRIBUTE, 'ir_attribute_query')
    if operation in {'spatial_relation', 'relation_query'} or _has_ir_spatial(question_ir):
        return (ROUTE_ENTITY_RELATION_SPATIAL, 'ir_spatial_relation')
    return None

def _route_from_evidence_question_type(value: str) -> str | None:
    mapping = {'temporal.point': ROUTE_TEMPORAL_POINT, 'temporal.interval': ROUTE_TEMPORAL_INTERVAL, 'non_temporal.entity.existence': ROUTE_BINARY_EXISTENCE, 'non_temporal.entity.identity': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.entity.attribute': ROUTE_ENTITY_ATTRIBUTE, 'non_temporal.entity.count': ROUTE_COUNT_SET, 'non_temporal.entity.relation': ROUTE_ENTITY_RELATION_SPATIAL, 'non_temporal.event.existence': ROUTE_BINARY_EXISTENCE, 'non_temporal.event.participant': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.event.outcome': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.state.scene': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.state.object': ROUTE_BINARY_EXISTENCE, 'non_temporal.state.traversability': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.ego.motion': ROUTE_EGO_MOTION_SIDE_TURN, 'non_temporal.ego.direction': ROUTE_EGO_MOTION_SIDE_TURN, 'non_temporal.ego.interaction': ROUTE_EGO_MOTION_SIDE_TURN, 'non_temporal.symbolic.ocr_logo': ROUTE_OCR_LOGO_TEXT, 'non_temporal.spatial.localization': ROUTE_ENTITY_RELATION_SPATIAL, 'non_temporal.spatial.relation': ROUTE_ENTITY_RELATION_SPATIAL, 'non_temporal.functional.affordance': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.world_knowledge.contextual': ROUTE_OPEN_DESCRIPTIVE, 'non_temporal.open_descriptive': ROUTE_OPEN_DESCRIPTIVE}
    return mapping.get(value)

def _route_from_compiler(plan: QuestionPlan) -> str:
    if plan.answer_type == 'duration' or plan.query_family == 'duration':
        return ROUTE_TEMPORAL_INTERVAL
    if plan.answer_type == 'time' or plan.query_family == 'temporal_event':
        return ROUTE_TEMPORAL_POINT
    if plan.query_family == 'attribute':
        return ROUTE_ENTITY_ATTRIBUTE
    if plan.query_family == 'ego_motion':
        return ROUTE_EGO_MOTION_SIDE_TURN
    if plan.query_family in {'existence', 'current_state'} or plan.answer_type == 'binary':
        return ROUTE_BINARY_EXISTENCE
    if plan.spatial_constraints:
        return ROUTE_ENTITY_RELATION_SPATIAL
    return ROUTE_OPEN_DESCRIPTIVE

def _keyword_route(*, text: str, tokens: set[str]) -> tuple[str, str] | None:
    if text.startswith('how long') or 'duration' in tokens:
        return (ROUTE_TEMPORAL_INTERVAL, 'text_duration')
    if text.startswith('when ') or 'when' in tokens:
        return (ROUTE_TEMPORAL_POINT, 'text_when')
    if 'how many' in text or bool(tokens & _COUNT_TERMS):
        return (ROUTE_COUNT_SET, 'text_count')
    if _looks_like_ocr_request(text, tokens=tokens, question_ir={}):
        return (ROUTE_OCR_LOGO_TEXT, 'text_ocr_logo')
    if 'what color' in text or bool(tokens & _ATTRIBUTE_TERMS):
        return (ROUTE_ENTITY_ATTRIBUTE, 'text_attribute')
    if bool(tokens & _EGO_TERMS):
        return (ROUTE_EGO_MOTION_SIDE_TURN, 'text_ego_motion')
    if bool(tokens & _SPATIAL_TERMS) or any((phrase in text for phrase in _SPATIAL_PHRASES)):
        return (ROUTE_ENTITY_RELATION_SPATIAL, 'text_spatial_relation')
    return None

def _looks_like_ocr_request(text: str, *, tokens: set[str], question_ir: dict[str, Any]) -> bool:
    if tokens & _OCR_TERMS:
        return True
    if tokens & {'sign', 'signage'} and tokens & _OCR_SIGN_INTENT_TERMS:
        return True
    if 'license plate' in text:
        return True
    if 'what is the name' in text or 'what was the name' in text:
        return True
    evidence_text = _normalize(' '.join([*_iter_text_values(question_ir.get('evidence_requirements')), *_iter_text_values(question_ir.get('graph_information_needs')), *_iter_text_values(question_ir.get('target_state'))]))
    evidence_tokens = set(_TOKEN_PATTERN.findall(evidence_text))
    return bool(evidence_tokens & _OCR_TERMS) or bool(evidence_tokens & {'sign', 'signage'} and evidence_tokens & _OCR_SIGN_INTENT_TERMS)

def _looks_like_ego_request(text: str, *, tokens: set[str], question_ir: dict[str, Any]) -> bool:
    del question_ir
    ego_terms = set(tokens & _EGO_TERMS)
    if 'right now' in text:
        ego_terms.discard('right')
    if not ego_terms:
        return False
    if tokens & _EGO_ACTOR_TERMS:
        return True
    return 'which direction' in text or 'what side' in text

def _has_ir_attribute(question_ir: dict[str, Any]) -> bool:
    return bool(_iter_text_values(question_ir.get('attribute_dimension')) or _iter_text_values(question_ir.get('attribute_dimension_aliases')))

def _has_ir_spatial(question_ir: dict[str, Any]) -> bool:
    return bool(_iter_text_values(question_ir.get('spatial_constraints')))

def _answer_shape(*, qtype: str, question_ir: dict[str, Any], compiler: QuestionPlan) -> str:
    ir_answer_type = _clean_scalar(question_ir.get('answer_type')).lower()
    if qtype in {'binary', 'position', 'time', 'duration', 'text'}:
        return qtype
    if ir_answer_type:
        return ir_answer_type
    return compiler.answer_type or 'text'

def _time_requirements(route_id: str, *, question_ir: dict[str, Any], compiler: QuestionPlan) -> dict[str, Any]:
    if route_id not in {ROUTE_TEMPORAL_POINT, ROUTE_TEMPORAL_INTERVAL, ROUTE_COUNT_SET}:
        return {}
    operation = _clean_scalar(question_ir.get('operation')).lower()
    return {'requires_time_axis': True, 'requires_interval': route_id == ROUTE_TEMPORAL_INTERVAL, 'requires_point': route_id == ROUTE_TEMPORAL_POINT, 'operation': operation, 'temporal_relation': _clean_scalar(question_ir.get('temporal_relation')).lower(), 'compiler_temporal_constraints': list(compiler.temporal_constraints)}

def _retrieval_policy(route_id: str) -> str:
    policies = {ROUTE_TEMPORAL_POINT: 'temporal_axis_event_boundary_then_target_filter', ROUTE_TEMPORAL_INTERVAL: 'target_entity_state_start_end_boundary', ROUTE_ENTITY_ATTRIBUTE: 'target_entity_direct_attribute_binding', ROUTE_ENTITY_RELATION_SPATIAL: 'target_entity_relation_bbox_spatial_binding', ROUTE_COUNT_SET: 'time_condition_filtered_entity_set', ROUTE_BINARY_EXISTENCE: 'target_or_event_existence_evidence', ROUTE_OCR_LOGO_TEXT: 'text_logo_attribute_first_with_visual_anchor', ROUTE_EGO_MOTION_SIDE_TURN: 'ego_pose_heading_or_spatial_proxy', ROUTE_OPEN_DESCRIPTIVE: 'hybrid_candidate_pack_fallback'}
    return policies.get(route_id, 'hybrid_candidate_pack_fallback')

def _evidence_policy(route_id: str) -> str:
    policies = {ROUTE_TEMPORAL_POINT: 'time_anchor_candidate_pack', ROUTE_TEMPORAL_INTERVAL: 'start_end_boundary_candidate_pack', ROUTE_ENTITY_ATTRIBUTE: 'direct_attribute_candidate_pack', ROUTE_ENTITY_RELATION_SPATIAL: 'direct_relation_or_bbox_candidate_pack', ROUTE_COUNT_SET: 'deduplicated_entity_set_candidate_pack', ROUTE_BINARY_EXISTENCE: 'existence_candidate_pack', ROUTE_OCR_LOGO_TEXT: 'hsgg_text_logo_attribute_candidate_pack', ROUTE_EGO_MOTION_SIDE_TURN: 'ego_motion_candidate_pack', ROUTE_OPEN_DESCRIPTIVE: 'common_candidate_pack'}
    return policies.get(route_id, 'common_candidate_pack')

def _taxonomy_for_route(route_id: str, *, question_ir: dict[str, Any], compiler: QuestionPlan) -> dict[str, str]:
    path, family = _taxonomy_path_and_family(route_id, question_ir=question_ir, compiler=compiler)
    parts = path.split('.')
    root = parts[0] if parts else ''
    domain = parts[1] if len(parts) > 1 else ''
    task = '.'.join(parts[2:]) if len(parts) > 2 else ''
    return {'taxonomy_path': path, 'taxonomy_root': root, 'taxonomy_domain': domain, 'taxonomy_task': task, 'evidence_family': family, 'taxonomy_source': 'question_route_derived'}

def _taxonomy_path_and_family(route_id: str, *, question_ir: dict[str, Any], compiler: QuestionPlan) -> tuple[str, str]:
    if route_id == ROUTE_TEMPORAL_POINT:
        return ('temporal.point', 'temporal_point_localization')
    if route_id == ROUTE_TEMPORAL_INTERVAL:
        return ('temporal.interval', 'temporal_interval_estimation')
    if route_id == ROUTE_ENTITY_ATTRIBUTE:
        return ('non_temporal.entity.attribute', 'entity_bound_attribute')
    if route_id == ROUTE_ENTITY_RELATION_SPATIAL:
        return ('non_temporal.spatial.relation', 'entity_spatial_relation')
    if route_id == ROUTE_COUNT_SET:
        return ('non_temporal.entity.count', 'entity_set_count')
    if route_id == ROUTE_OCR_LOGO_TEXT:
        return ('non_temporal.symbolic.ocr_logo', 'symbolic_text_logo')
    if route_id == ROUTE_EGO_MOTION_SIDE_TURN:
        return _ego_taxonomy(question_ir=question_ir, compiler=compiler)
    if route_id == ROUTE_BINARY_EXISTENCE:
        return _binary_existence_taxonomy(question_ir=question_ir, compiler=compiler)
    if route_id == ROUTE_OPEN_DESCRIPTIVE:
        return ('non_temporal.open_descriptive', 'open_descriptive_evidence')
    return ('non_temporal.open_descriptive', 'open_descriptive_evidence')

def _binary_existence_taxonomy(*, question_ir: dict[str, Any], compiler: QuestionPlan) -> tuple[str, str]:
    operation = _clean_scalar(question_ir.get('operation')).lower()
    answer_type = _clean_scalar(question_ir.get('answer_type')).lower()
    attribute = _clean_scalar(question_ir.get('attribute_dimension')).lower()
    has_event_constraints = bool(_iter_text_values(question_ir.get('event_constraints')) or _has_ir_event_hard_constraint(question_ir) or compiler.events or compiler.temporal_constraints)
    if operation == 'current_state' or attribute in {'state', 'condition', 'status'}:
        return ('non_temporal.state.object', 'state_verification')
    if has_event_constraints or operation in {'occurrence', 'event_presence', 'event_time'}:
        return ('non_temporal.event.existence', 'event_existence_verification')
    if answer_type == 'binary':
        return ('non_temporal.entity.existence', 'entity_existence_verification')
    return ('non_temporal.entity.existence', 'entity_existence_verification')

def _has_ir_event_hard_constraint(question_ir: dict[str, Any]) -> bool:
    hard_constraints = question_ir.get('hard_constraints')
    if not isinstance(hard_constraints, (list, tuple, set)):
        hard_constraints = [hard_constraints] if hard_constraints else []
    for item in hard_constraints:
        if isinstance(item, dict):
            text = _normalize(' '.join([_clean_scalar(item.get('kind')), _clean_scalar(item.get('role')), _clean_scalar(item.get('operation')), _clean_scalar(item.get('relation')), ' '.join(_iter_text_values(item.get('must_bind_to')))]))
        else:
            text = _normalize(item)
        tokens = set(_TOKEN_PATTERN.findall(text))
        if tokens & {'action', 'event', 'event_condition', 'occurrence', 'transition'}:
            return True
    return False

def _ego_taxonomy(*, question_ir: dict[str, Any], compiler: QuestionPlan) -> tuple[str, str]:
    operation = _clean_scalar(question_ir.get('operation')).lower()
    focus = _clean_scalar(question_ir.get('qa_focus')).lower()
    if 'direction' in focus or 'direction' in operation:
        return ('non_temporal.ego.direction', 'ego_direction_evidence')
    if 'interaction' in focus or 'interaction' in operation:
        return ('non_temporal.ego.interaction', 'ego_interaction_evidence')
    if compiler.query_family == 'ego_motion' or 'motion' in focus or 'motion' in operation:
        return ('non_temporal.ego.motion', 'ego_motion_evidence')
    return ('non_temporal.ego.motion', 'ego_motion_evidence')

def _route_conflicts(*, selected_route: str, compiler_route: str, official_question_type: str, question_ir: dict[str, Any]) -> tuple[str, ...]:
    conflicts: list[str] = []
    if compiler_route != ROUTE_OPEN_DESCRIPTIVE and compiler_route != selected_route:
        conflicts.append(f'compiler_route={compiler_route}')
    ir_answer_type = _clean_scalar(question_ir.get('answer_type')).lower()
    if official_question_type and ir_answer_type and (official_question_type != ir_answer_type):
        conflicts.append(f'official_type={official_question_type};ir_answer_type={ir_answer_type}')
    return tuple(conflicts)

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_scalar(value)
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target_state', 'relation', 'operation', 'target', 'value'):
            if value.get(key) not in (None, '', [], {}):
                values.append(_clean_scalar(value.get(key)))
        for key in ('target_entities', 'must_bind_to', 'aliases'):
            values.extend(_iter_text_values(value.get(key)))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_iter_text_values(item))
        return values
    return [_clean_scalar(value)]

def _ordered_unique(values: list[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_scalar(value)
        if not text:
            continue
        normalized = ' '.join(text.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return tuple(result)

def _clean_scalar(value: Any) -> str:
    text = re.sub('\\s+', ' ', str(value or '').strip())
    return text

def _normalize(value: str) -> str:
    return re.sub('\\s+', ' ', str(value or '').strip().lower())
