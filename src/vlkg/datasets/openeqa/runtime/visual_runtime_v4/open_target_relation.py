"""Role-bound evidence for OpenEQA ``open_target_relation`` questions.

This handler is deliberately variant-local.  It preserves the frozen v3
result unless it can construct an explicit ``target -> relation -> anchor``
pair from same-frame geometry or a canonical relation episode reached through
a visually recalled anchor.
"""
from __future__ import annotations
from copy import deepcopy
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import verify_spatial_pairs
from vlkg.datasets.openeqa.runtime.visual_runtime_v3.spatial_guarded import _resolve_component_pair_owners
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'spatial_localization.open_target_relation_graph_only_v1'
OPEN_TARGET_RELATION_CONTRACT = 'openeqa_open_target_relation_graph_only.v1'
_STRATEGY_VARIANT = 'open_target_relation'
_MAX_VISIBLE_PAIRS = 4
_MIN_RAW_GEOMETRY_PAIRS = 24
_MAX_RELATION_IMAGE_FRAMES = 3
_MAX_RELATION_FRAME_DISTANCE = 12
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_RELATION_SELECTION_GUIDANCE = 'kind=openeqa_open_target_relation_guidance candidate_order=relation_support_not_answer_certainty candidate_set=primary_relation_bound_answer_space compare_all_candidates=true prefer_complete_persistent_object=true ignore_unbound_or_self_referential_graph_spatial_candidates=true combine_compatible_candidates=true unique_answer_not_proven=true uses_gold_answer=false'
_VISUAL_LINE_FIELDS = ('visual_faiss_evidence_lines', 'visual_entity_candidate_pack_lines', 'visual_faiss_unified_candidate_lines', 'visual_faiss_evidence_layer_lines', 'visual_faiss_local_subgraph_fact_lines', 'visual_faiss_role_aware_lines', 'visual_faiss_role_aware_aux_lines', 'visual_faiss_object_local_fact_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_ger_evidence_lines', 'frame_faiss_evidence_lines', 'merged_frame_faiss_evidence_lines')
_SYMMETRIC_ALIASES = {'near': {'near', 'next_to', 'beside', 'adjacent_to', 'close_to'}, 'across_from': {'across', 'across_from', 'opposite', 'opposite_to'}, 'opposite': {'across', 'across_from', 'opposite', 'opposite_to'}, 'perpendicular_to': {'perpendicular', 'perpendicular_to'}}
_FORWARD_ALIASES = {'left_of': {'left', 'left_of', 'to_the_left_of'}, 'right_of': {'right', 'right_of', 'to_the_right_of'}, 'above': {'above', 'over', 'on_top_of', 'positioned_above', 'located_above'}, 'below': {'below', 'under', 'underneath', 'beneath', 'located_under'}, 'on': {'on', 'on_top_of', 'resting_on', 'sitting_on', 'standing_on', 'mounted_on', 'hanging_on', 'hanging_from', 'attached_to', 'leaning_on'}, 'inside': {'in', 'inside', 'inside_of', 'within'}, 'outside': {'outside', 'outside_of'}, 'behind': {'behind'}, 'in_front_of': {'in_front_of', 'front_of'}, 'at': {'at', 'located_at'}}
_REVERSE_ALIASES = {'left_of': _FORWARD_ALIASES['right_of'], 'right_of': _FORWARD_ALIASES['left_of'], 'above': _FORWARD_ALIASES['below'], 'below': _FORWARD_ALIASES['above'], 'on': {'support', 'supports', 'supporting', 'hold', 'holds', 'holding'}, 'inside': {'contain', 'contains', 'containing', 'hold', 'holds', 'holding'}, 'behind': _FORWARD_ALIASES['in_front_of'], 'in_front_of': _FORWARD_ALIASES['behind']}
_GENERIC_TARGET_LABELS = {'area', 'background', 'floor', 'room', 'scene', 'space', 'surface', 'tile', 'wall'}
_ANCHOR_STOP_TOKENS = {'a', 'an', 'at', 'black', 'blue', 'brown', 'clear', 'corner', 'first', 'front', 'green', 'grey', 'in', 'large', 'left', 'long', 'of', 'on', 'one', 'other', 'red', 'right', 'second', 'side', 'small', 'the', 'top', 'two', 'upper', 'white', 'with'}
_COMPONENT_TARGET_TOKENS = {'armrest', 'backrest', 'baseboard', 'bracket', 'burner', 'button', 'casing', 'cushion', 'fabric', 'frame', 'grate', 'handle', 'headboard', 'hinge', 'knob', 'lampshade', 'leg', 'lid', 'panel', 'pane', 'pipe', 'pocket', 'rung', 'seat', 'screw', 'slat', 'spout', 'surface', 'thermostat', 'latch'}
_ANCHOR_SYNONYM_GROUPS = ({'stair', 'staircase', 'step'}, {'tv', 'television'}, {'sofa', 'couch'}, {'fridge', 'refrigerator'}, {'picture', 'image', 'painting'}, {'restroom', 'bathroom'})

def fuse_open_target_relation_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Publish only direction-preserving, role-bound relation candidates."""
    fused = fuse_frozen_v3(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    if str(evidence_demand.get('strategy_variant') or '') != _STRATEGY_VARIANT:
        return fused
    verifier_demand = deepcopy(evidence_demand)
    verifier_demand['open_target_slot'] = True
    verifier_demand['target_terms'] = []
    raw_geometry_pairs = verify_spatial_pairs(visual_payload, graph=graph, evidence_demand=verifier_demand, limit=max(_MIN_RAW_GEOMETRY_PAIRS, _MAX_VISIBLE_PAIRS * 3, int(max_candidate_packs) * 3))
    geometry_pairs, owner_resolution = _resolve_component_pair_owners(raw_geometry_pairs, graph=graph)
    entity_by_id = _entity_lookup(graph)
    anchor_observations, target_observations = _role_hits(visual_payload, evidence_demand=evidence_demand)
    geometry_pairs = _rerank_geometry_pairs(geometry_pairs, target_observations=target_observations, entity_by_id=entity_by_id, target_terms=evidence_demand.get('target_terms', []))
    canonical_pairs = _canonical_relation_pairs(visual_payload, graph=graph, evidence_demand=evidence_demand)
    pairs = _merge_relation_pairs([*canonical_pairs, *geometry_pairs], limit=min(_MAX_VISIBLE_PAIRS, max(1, int(max_visual_lines))), target_terms=evidence_demand.get('target_terms', []))
    image_frames: list[dict[str, Any]] = []
    return _publish_relation_pairs(fused, pairs=pairs, raw_geometry_pairs=raw_geometry_pairs, canonical_pair_count=len(canonical_pairs), owner_resolution=owner_resolution, requested_relations=_requested_relations(evidence_demand), image_frames=image_frames, max_visual_lines=max_visual_lines)

def _canonical_relation_pairs(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, evidence_demand: dict[str, Any]) -> list[dict[str, Any]]:
    entity_by_id = _entity_lookup(graph)
    anchors, target_observations = _role_hits(visual_payload, evidence_demand=evidence_demand)
    anchors = _bind_unbound_anchor_hits(anchors, graph=graph, entity_by_id=entity_by_id)
    if not anchors:
        return []
    target_semantic_scores = _target_semantic_scores(target_observations, entity_by_id=entity_by_id)
    pairs: list[dict[str, Any]] = []
    for anchor_entity, requested, anchor_hits in _group_anchor_hits(anchors, entity_by_id=entity_by_id):
        anchor_id = anchor_entity.entity_id
        for episode in graph.episode_nodes:
            if str(episode.episode_kind or '').upper() != 'RELATION':
                continue
            subject = entity_by_id.get(str(episode.subject_entity_id or ''))
            object_ = entity_by_id.get(str(episode.object_entity_id or ''))
            if subject is None or object_ is None:
                continue
            predicate = _normalize_relation(episode.raw_predicate or episode.predicate)
            candidate = None
            direction = ''
            if object_.entity_id == anchor_id and _forward_matches(requested, predicate):
                candidate = subject
                direction = 'target_to_anchor'
            elif subject.entity_id == anchor_id and _reverse_matches(requested, predicate):
                candidate = object_
                direction = 'anchor_inverse_to_target'
            if candidate is None or candidate.entity_id == anchor_id:
                continue
            if candidate.is_noise or _generic_target(candidate.label):
                continue
            relation_frame = _range_representative_frame(episode.time_range)
            anchor = _best_anchor_for_range(anchor_hits, episode.time_range)
            anchor_frame = _frame(anchor)
            frame_distance = _frame_distance(anchor_frame, episode.time_range)
            observation = _best_target_observation(target_observations, candidate_entity_id=candidate.entity_id, entity_by_id=entity_by_id, preferred_frame=anchor_frame if anchor_frame is not None else relation_frame)
            co_visible = bool(observation and anchor_frame is not None and (_frame(observation) == anchor_frame))
            if frame_distance is not None and frame_distance > _MAX_RELATION_FRAME_DISTANCE and (not co_visible):
                continue
            support_count = int(episode.time_range.support_frames or 1)
            score = 1.0 + 0.2 * max(0.0, float(anchor.get('similarity') or 0.0)) + 0.7 * target_semantic_scores.get(candidate.entity_id, 0.0) + (0.18 if co_visible else 0.0) + (0.1 if observation else 0.0) + min(0.12, 0.03 * max(0, support_count - 1)) + (0.08 if frame_distance == 0 else 0.0)
            pairs.append({'contract': OPEN_TARGET_RELATION_CONTRACT, 'frame_index': _frame(observation) if observation else relation_frame, 'scene_id': episode.scene_id or anchor.get('scene_id'), 'target_label': candidate.label, 'target_entity_id': candidate.entity_id, 'target_bbox': (observation or {}).get('bbox'), 'anchor_label': anchor_entity.label, 'anchor_entity_id': anchor_id, 'anchor_bbox': anchor.get('bbox'), 'relation': requested, 'relation_verified': True, 'verification_sources': ['canonical_relation'], 'canonical_predicate': predicate, 'canonical_relation_direction': direction, 'relation_episode_id': episode.episode_id, 'relation_frame_distance': frame_distance, 'target_observation': _compact_hit(observation or {}), 'anchor_observations': [_compact_hit(anchor)], 'graph_bound': True, 'co_visible': co_visible, 'pair_score': round(score, 6), 'support_frames': _support_frames(episode.time_range), 'support_frame_count': support_count, 'uses_gold_answer': False})
    return _coalesce_relation_pairs(pairs)

def _publish_relation_pairs(graph_result: dict[str, Any], *, pairs: list[dict[str, Any]], raw_geometry_pairs: list[dict[str, Any]], canonical_pair_count: int, owner_resolution: dict[str, Any], requested_relations: set[str], image_frames: list[dict[str, Any]], max_visual_lines: int) -> dict[str, Any]:
    bundle = graph_result.get('llm_ready_bundle')
    if not isinstance(bundle, dict):
        return graph_result
    updated = deepcopy(bundle)
    old_visual = deepcopy(graph_result.get('visual_evidence') or {})
    visible_pairs = pairs[:min(_MAX_VISIBLE_PAIRS, max(1, int(max_visual_lines)))]
    pair_lines = [_relation_pair_line(index, pair) for index, pair in enumerate(visible_pairs, start=1)]
    pair_facts = [_relation_pair_fact(pair) for pair in visible_pairs]
    if pair_lines:
        graph_lines = [str(line) for line in updated.get('evidence_lines', []) or [] if str(line).strip() and (not _is_visual_line(line))]
        graph_aggregation_lines = [str(line) for line in updated.get('evidence_aggregation_lines', []) or [] if str(line).strip() and (not _is_visual_line(line))]
        for field in _VISUAL_LINE_FIELDS:
            updated[field] = []
        routed_lines = [_RELATION_SELECTION_GUIDANCE, *pair_lines]
        updated['question_route_open_target_relation_lines'] = routed_lines
        updated['evidence_lines'] = _ordered_unique([*routed_lines, *graph_lines])
        updated['evidence_aggregation_lines'] = _ordered_unique([*routed_lines, *graph_aggregation_lines])
        aggregation = deepcopy(updated.get('evidence_aggregation') or {})
        graph_facts = [fact for fact in aggregation.get('candidate_facts', []) or [] if isinstance(fact, dict) and (not _is_visual_fact(fact))]
        aggregation['candidate_facts'] = [*pair_facts, *graph_facts]
        updated['evidence_aggregation'] = aggregation
    verification = {'contract': OPEN_TARGET_RELATION_CONTRACT, 'applied': True, 'requested_relations': sorted(requested_relations), 'canonical_pair_count': canonical_pair_count, 'raw_geometry_pair_count': len(raw_geometry_pairs), 'answer_visible_pair_count': len(visible_pairs), 'relation_direction_is_target_to_anchor': True, 'target_anchor_roles_are_separate': True, 'fallback_to_frozen_v3': not bool(visible_pairs), 'unverified_candidates_are_relation_proof': False, 'component_owner_resolution': owner_resolution, 'multimodal_relation_frame_count': len(image_frames), 'uses_gold_answer': False}
    candidate_packs = [_relation_pair_pack(pair) for pair in visible_pairs] if visible_pairs else list(old_visual.get('candidate_packs', []) or [])
    visual_evidence = {**old_visual, 'contract': OPEN_TARGET_RELATION_CONTRACT, 'open_target_relation_contract': OPEN_TARGET_RELATION_CONTRACT, 'candidate_packs': candidate_packs, 'spatial_pairs': visible_pairs, 'spatial_candidate_lines': pair_lines, 'open_target_relation_verification': verification, 'open_target_relation_image_frames': image_frames, 'route_strategy_applied': True, 'route_strategy_variant': _STRATEGY_VARIANT, 'stats': {**dict(old_visual.get('stats') or {}), 'open_target_relation_canonical_pair_count': canonical_pair_count, 'open_target_relation_geometry_pair_count': len(raw_geometry_pairs), 'open_target_relation_answer_visible_pair_count': len(visible_pairs), 'answer_visible_spatial_pair_count': len(visible_pairs), 'answer_visible_unverified_spatial_candidate_count': 0 if visible_pairs else len(old_visual.get('candidate_packs', []) or [])}}
    updated['open_target_relation_contract'] = OPEN_TARGET_RELATION_CONTRACT
    updated['open_target_relation_verification'] = verification
    updated['open_target_relation_image_frames'] = image_frames
    updated['visual_evidence'] = visual_evidence
    structured = deepcopy(updated.get('structured_graph_context') or {})
    if isinstance(structured, dict):
        structured['open_target_relation_contract'] = OPEN_TARGET_RELATION_CONTRACT
        structured['open_target_relation_verification'] = verification
        structured['open_target_relation_image_frames'] = [{'frame_index': item.get('frame_index'), 'source': item.get('source'), 'uses_gold_answer': False} for item in image_frames]
        structured['question_route_open_target_relation_lines'] = [_RELATION_SELECTION_GUIDANCE, *pair_lines] if pair_lines else []
        structured['visual_evidence'] = visual_evidence
        for field in _VISUAL_LINE_FIELDS:
            if pair_lines:
                structured[field] = []
    updated['structured_graph_context'] = structured
    graph_result['llm_ready_bundle'] = updated
    graph_result['visual_evidence'] = visual_evidence
    return graph_result

def _role_hits(visual_payload: dict[str, Any], *, evidence_demand: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    constraints = [item for item in evidence_demand.get('spatial_constraints', []) or [] if isinstance(item, dict)]
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent')
        intent = intent if isinstance(intent, dict) else {}
        role = str(intent.get('role') or 'context')
        intent_relation = _normalize_requested_relation(intent.get('spatial_relation'))
        intent_text = str(intent.get('text') or '')
        for raw_hit in group.get('hits', []) or []:
            if not isinstance(raw_hit, dict):
                continue
            hit = deepcopy(raw_hit)
            hit['retrieval_role'] = role
            hit['intent_text'] = intent_text
            if role == 'anchor':
                if not _anchor_hit_matches_intent(hit, intent_text):
                    continue
                relation = intent_relation or _constraint_relation(hit, intent_text=intent_text, constraints=constraints)
                if relation:
                    hit['requested_relation'] = relation
                    anchors.append(hit)
            elif role in {'target', 'spatial_target'}:
                targets.append(hit)
    return (_dedupe_hits(anchors, include_relation=True), _dedupe_hits(targets))

def _relation_image_frames(*, pairs: Iterable[dict[str, Any]], visual_payload: dict[str, Any], anchor_observations: Iterable[dict[str, Any]], target_observations: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select route-local relation views with role-preserving annotations."""
    rows: list[tuple[int | None, str, str]] = []
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent')
        intent = intent if isinstance(intent, dict) else {}
        role = str(intent.get('role') or 'context')
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            path = str(hit.get('image_path') or '').strip()
            if path:
                rows.append((_frame(hit), path, role))
    by_frame: dict[int, tuple[str, str]] = {}
    for frame, path, role in rows:
        if frame is not None:
            by_frame.setdefault(frame, (path, role))
    selected: list[dict[str, Any]] = []
    seen_views: set[tuple[Any, ...]] = set()
    relation_view_frames: set[tuple[int | None, str]] = set()

    def add(frame: int | None, path: str, source: str, *, view_mode: str, crop_box: tuple[float, float, float, float] | None=None, candidate_rank: int | None=None, target_label: str='', relation: str='', anchor_label: str='', target_bbox: tuple[float, float, float, float] | None=None, anchor_bboxes: Iterable[tuple[float, float, float, float]]=()) -> None:
        cleaned = str(path or '').strip()
        normalized_anchors = list(anchor_bboxes)
        view_key = (cleaned, frame, view_mode, candidate_rank, crop_box, target_bbox, tuple(normalized_anchors))
        if not cleaned or view_key in seen_views or len(selected) >= max(1, int(limit)):
            return
        seen_views.add(view_key)
        item: dict[str, Any] = {'frame_index': frame, 'image_path': cleaned, 'source': source, 'view_mode': view_mode, 'overlay_contract': 'openeqa_relation_role_overlay.v1', 'uses_gold_answer': False}
        if crop_box is not None:
            item['crop_box'] = list(crop_box)
        if candidate_rank is not None:
            item['candidate_rank'] = candidate_rank
        if target_label:
            item['target_label'] = target_label
        if relation:
            item['relation'] = relation
        if anchor_label:
            item['anchor_label'] = anchor_label
        if target_bbox is not None:
            item['target_bbox'] = list(target_bbox)
        if normalized_anchors:
            item['anchor_bboxes'] = [list(value) for value in normalized_anchors]
        selected.append(item)
    for candidate_rank, pair in enumerate(pairs, start=1):
        frame = _optional_int(pair.get('frame_index'))
        path, _ = by_frame.get(frame, ('', '')) if frame is not None else ('', '')
        selected_before = len(selected)
        target_bbox = _bbox(pair.get('target_bbox'))
        anchor_bboxes = _bbox_list(pair.get('anchor_bbox'))
        add(frame, path, 'relation_pair_context', view_mode='annotated_relation_context', crop_box=_relation_context_crop(target_bbox, anchor_bboxes), candidate_rank=candidate_rank, target_label=str(pair.get('target_label') or ''), relation=str(pair.get('relation') or ''), anchor_label=str(pair.get('anchor_label') or ''), target_bbox=target_bbox, anchor_bboxes=anchor_bboxes)
        if len(selected) > selected_before:
            relation_view_frames.add((frame, str(path or '').strip()))
    for source_name, observations in (('anchor_recall_frame', anchor_observations), ('relation_context_frame', target_observations)):
        for hit in observations:
            frame = _frame(hit)
            path = str(hit.get('image_path') or '')
            if not path and frame is not None:
                path, _ = by_frame.get(frame, ('', ''))
            if (frame, str(path or '').strip()) in relation_view_frames:
                continue
            hit_bbox = _bbox(hit.get('bbox'))
            is_anchor = source_name == 'anchor_recall_frame'
            add(frame, path, source_name, view_mode='annotated_anchor_context' if is_anchor else 'annotated_retrieval_context', crop_box=_single_object_context_crop(hit_bbox), target_label='' if is_anchor else str(hit.get('label') or ''), anchor_label=str(hit.get('label') or '') if is_anchor else '', target_bbox=None if is_anchor else hit_bbox, anchor_bboxes=[hit_bbox] if is_anchor and hit_bbox else [])
    for frame, path, role in rows:
        if (frame, str(path or '').strip()) in relation_view_frames:
            continue
        add(frame, path, f'retrieved_{role}_frame', view_mode='full_relation_context')
    return selected

def _bbox_list(value: Any) -> list[tuple[float, float, float, float]]:
    single = _bbox(value)
    if single is not None:
        return [single]
    if not isinstance(value, (list, tuple)):
        return []
    return [box for item in value if (box := _bbox(item)) is not None]

def _relation_context_crop(target_bbox: tuple[float, float, float, float] | None, anchor_bboxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    boxes = [box for box in [target_bbox, *anchor_bboxes] if box is not None]
    if not boxes:
        return None
    left = min((box[0] for box in boxes))
    top = min((box[1] for box in boxes))
    right = max((box[2] for box in boxes))
    bottom = max((box[3] for box in boxes))
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    pad_x = max(48.0, width * 0.18)
    pad_y = max(48.0, height * 0.18)
    return (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)

def _single_object_context_crop(bbox: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    pad_x = max(64.0, width * 1.5)
    pad_y = max(64.0, height * 1.5)
    return (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)

def _constraint_relation(hit: dict[str, Any], *, intent_text: str, constraints: list[dict[str, Any]]) -> str:
    hit_tokens = _tokens([hit.get('label'), hit.get('visual_attribute'), intent_text])
    for item in constraints:
        anchor_tokens = _tokens(item.get('anchor'))
        if anchor_tokens and anchor_tokens & hit_tokens:
            relation = _normalize_requested_relation(item.get('relation'))
            if relation:
                return relation
    return ''

def _requested_relations(evidence_demand: dict[str, Any]) -> set[str]:
    return {relation for value in evidence_demand.get('spatial_relations', []) or [] if (relation := _normalize_requested_relation(value))}

def _normalize_requested_relation(value: Any) -> str:
    relation = _normalize_relation(value)
    if not relation:
        return ''
    if 'between' in relation:
        return 'between'
    if 'left' in relation:
        return 'left_of'
    if 'right' in relation:
        return 'right_of'
    if relation in {'above', 'over', 'on_top_of'}:
        return 'above'
    if relation in {'below', 'under', 'beneath', 'underneath'}:
        return 'below'
    if relation in {'front', 'front_of', 'in_front_of'}:
        return 'in_front_of'
    if relation in {'across', 'across_from'}:
        return 'across_from'
    if relation in {'perpendicular', 'perpendicular_to'}:
        return 'perpendicular_to'
    return relation

def _normalize_relation(value: Any) -> str:
    relation = '_'.join(_TOKEN_PATTERN.findall(str(value or '').lower()))
    return relation.removeprefix('raw_relation_')

def _forward_matches(requested: str, predicate: str) -> bool:
    if predicate in _SYMMETRIC_ALIASES.get(requested, set()):
        return True
    return predicate in _FORWARD_ALIASES.get(requested, set())

def _reverse_matches(requested: str, predicate: str) -> bool:
    if predicate in _SYMMETRIC_ALIASES.get(requested, set()):
        return True
    return predicate in _REVERSE_ALIASES.get(requested, set())

def _merge_relation_pairs(pairs: list[dict[str, Any]], *, limit: int, target_terms: Any) -> list[dict[str, Any]]:
    result = _coalesce_relation_pairs(pairs)
    support_by_target: dict[str, set[tuple[str, str]]] = {}
    for pair in result:
        target_key = str(pair.get('target_entity_id') or pair.get('target_label') or '')
        support_by_target.setdefault(target_key, set()).add((str(pair.get('relation') or ''), _anchor_semantic_key(pair)))
    requested_target_tokens = _tokens(target_terms) - {'object', 'thing', 'item'}
    for pair in result:
        target_key = str(pair.get('target_entity_id') or pair.get('target_label') or '')
        support_count = len(support_by_target.get(target_key, set()))
        component_penalty = 0.55 if _component_target(pair.get('target_label')) and (not requested_target_tokens & _tokens(pair.get('target_label'))) else 0.0
        generic_penalty = 0.6 if _generic_target(pair.get('target_label')) else 0.0
        support_frame_count = int(pair.get('support_frame_count') or 0)
        persistence_bonus = 0.0 if str(pair.get('relation') or '') == 'between' else min(0.36, 0.09 * max(0, support_frame_count - 1))
        whole_object_bonus = 0.08 if not component_penalty and (not generic_penalty) else 0.0
        pair['constraint_support_count'] = support_count
        pair['component_target_penalized'] = bool(component_penalty)
        pair['generic_target_penalized'] = bool(generic_penalty)
        pair['persistence_bonus'] = round(persistence_bonus, 6)
        pair['whole_object_bonus'] = round(whole_object_bonus, 6)
        pair['relation_evidence_tier'] = _relation_evidence_tier(pair)
        pair['pair_score'] = round(float(pair.get('pair_score') or 0.0) + min(0.36, 0.18 * max(0, support_count - 1)) + persistence_bonus + whole_object_bonus - component_penalty - generic_penalty, 6)
    result.sort(key=lambda pair: (-int(pair.get('relation_evidence_tier') or 0), -float(pair.get('pair_score') or 0.0), -int(pair.get('constraint_support_count') or 0), -int(pair.get('support_frame_count') or 0), -int('canonical_relation' in pair.get('verification_sources', [])), str(pair.get('target_label') or '')))
    return _select_label_diverse_pairs(result, limit=limit)

def _anchor_semantic_key(pair: dict[str, Any]) -> str:
    label_key = _normalized_label_key(pair.get('anchor_label'))
    if label_key:
        return 'label:' + ' '.join(label_key)
    return 'entity:' + str(pair.get('anchor_entity_id') or '')

def _coalesce_relation_pairs(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in pairs:
        key = (str(source.get('target_entity_id') or source.get('target_label') or ''), str(source.get('relation') or ''), str(source.get('anchor_entity_id') or source.get('anchor_label') or ''))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = deepcopy(source)
            continue
        current_sources = list(existing.get('verification_sources') or [])
        source_sources = list(source.get('verification_sources') or [])
        existing['verification_sources'] = _ordered_unique([*current_sources, *source_sources])
        existing['support_frames'] = _ordered_unique([*list(existing.get('support_frames') or []), *list(source.get('support_frames') or [])])
        existing['support_frame_count'] = len(existing['support_frames'])
        existing['pair_score'] = round(max(float(existing.get('pair_score') or 0.0), float(source.get('pair_score') or 0.0)) + 0.08, 6)
        existing['co_visible'] = bool(existing.get('co_visible') or source.get('co_visible'))
    return list(grouped.values())

def _relation_evidence_tier(pair: dict[str, Any]) -> int:
    sources = set(pair.get('verification_sources') or [])
    has_canonical = 'canonical_relation' in sources
    has_geometry = any((str(source).startswith('same_frame_bbox_geometry') for source in sources))
    if has_canonical and has_geometry:
        return 4
    if 'same_frame_bbox_geometry_two_anchors' in sources:
        return 4
    if has_geometry or (has_canonical and bool(pair.get('co_visible'))):
        return 3
    if has_canonical:
        return 2
    return 1

def _select_label_diverse_pairs(pairs: Iterable[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_labels: set[tuple[str, ...]] = set()
    for pair in pairs:
        label_key = _normalized_label_key(pair.get('target_label'))
        if label_key and label_key in seen_labels:
            continue
        if label_key:
            seen_labels.add(label_key)
        selected.append(pair)
        if len(selected) >= max(1, int(limit)):
            break
    return selected

def _normalized_label_key(value: Any) -> tuple[str, ...]:
    tokens = _TOKEN_PATTERN.findall(str(value or '').lower())
    normalized = []
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            token = token[:-3] + 'y'
        elif len(token) > 4 and token.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
            token = token[:-2]
        elif len(token) > 3 and token.endswith('s') and (not token.endswith('ss')):
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)

def _relation_pair_line(index: int, pair: dict[str, Any]) -> str:
    return f"rank={index} kind=openeqa_open_target_relation_pair target={pair.get('target_label')!r} target_entity={pair.get('target_entity_id') or 'unbound_visual_observation'} relation={pair.get('relation') or ''} anchor={pair.get('anchor_label')!r} anchor_entity={pair.get('anchor_entity_id') or 'unbound_visual_observation'} frame={pair.get('frame_index')} pair_score={float(pair.get('pair_score') or 0.0):.3f} support_frames={int(pair.get('support_frame_count') or 0)} evidence_tier={int(pair.get('relation_evidence_tier') or 0)} verification_sources={','.join(pair.get('verification_sources') or [])} binding=direct_target_relation_verified target_role=open_answer_candidate anchor_role=question_constraint relation_direction=target_to_anchor relation_verified=true unique_answer_not_proven=true uses_gold_answer=false"

def _relation_pair_fact(pair: dict[str, Any]) -> dict[str, Any]:
    return {'fact_type': 'openeqa_open_target_relation_pair', 'source': OPEN_TARGET_RELATION_CONTRACT, 'source_kind': 'role_bound_spatial_relation_pair', 'subject': str(pair.get('target_label') or 'object'), 'predicate': str(pair.get('relation') or ''), 'object': str(pair.get('anchor_label') or 'object'), 'fact_text': f"{pair.get('target_label') or 'object'} {pair.get('relation') or ''} {pair.get('anchor_label') or 'object'}", 'binding_strength': 'direct_target_question_factor', 'requirement_hits': [str(pair.get('relation') or '')], 'hard_constraint_hits': [[str(pair.get('relation') or ''), str(pair.get('anchor_label') or 'object')]], 'hard_constraint_coverage': 1, 'entity_ids': [value for value in (pair.get('target_entity_id'), pair.get('anchor_entity_id')) if value], 'score': float(pair.get('pair_score') or 0.0), 'relation_verified': True, 'target_anchor_roles_are_separate': True, 'unique_answer_proven': False, 'uses_gold_answers': False}

def _relation_pair_pack(pair: dict[str, Any]) -> dict[str, Any]:
    return {'contract': OPEN_TARGET_RELATION_CONTRACT, 'candidate_id': 'open_target_relation:' + str(pair.get('target_entity_id') or pair.get('target_label') or 'object'), 'candidate_role': 'open_answer_target', 'canonical_entity_id': pair.get('target_entity_id'), 'entity_label': pair.get('target_label'), 'graph_bound': bool(pair.get('target_entity_id')), 'binding_strength': 'direct_target_relation_verified', 'ranking_score': pair.get('pair_score'), 'relation': pair.get('relation'), 'anchor_entity_id': pair.get('anchor_entity_id'), 'anchor_label': pair.get('anchor_label'), 'relation_verified': True, 'spatial_pair': pair, 'uses_gold_answer': False}

def _entity_lookup(graph: CanonicalTemporalMemoryGraph) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for entity in graph.entity_nodes:
        lookup[entity.entity_id] = entity
        members = entity.attributes.get('member_entity_ids', [])
        if isinstance(members, list):
            for member in members:
                lookup.setdefault(str(member), entity)
    return lookup

def _bind_unbound_anchor_hits(hits: Iterable[dict[str, Any]], *, graph: CanonicalTemporalMemoryGraph, entity_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    """Conservatively bind visual anchors by label, scene, and time.

    Full visual indexes can contain HSGG observations that were not assigned a
    canonical entity ID.  Dropping those observations prevents explicit graph
    relations such as ``object on table`` from ever being reached.  This
    resolver only accepts a unique best entity and never uses answer text.
    """
    bound: list[dict[str, Any]] = []
    unique_entities = {entity.entity_id: entity for entity in graph.entity_nodes}.values()
    for source in hits:
        existing = entity_by_id.get(str(source.get('canonical_entity_id') or ''))
        if existing is not None:
            bound.append(source)
            continue
        hit_frame = _frame(source)
        hit_scene = str(source.get('scene_id') or '')
        hit_label_key = _normalized_label_key(source.get('label'))
        required_tokens = _semantic_tokens([source.get('intent_text'), source.get('label')]) - _ANCHOR_STOP_TOKENS
        observed_tokens = _semantic_tokens(source.get('label'))
        if not required_tokens and (not observed_tokens):
            continue
        candidates: list[tuple[float, int, str, Any]] = []
        for entity in unique_entities:
            if entity.is_noise:
                continue
            entity_scene = str(entity.attributes.get('scene_id') or '')
            if hit_scene and entity_scene and (hit_scene != entity_scene):
                continue
            distance = _frame_distance(hit_frame, entity.time_range)
            if distance is not None and distance > _MAX_RELATION_FRAME_DISTANCE:
                continue
            label_values = [entity.label, entity.labels_seen, entity.stable_identity_display_name, entity.stable_identity_canonical_label, entity.attributes.get('canonical_label'), entity.attributes.get('semantic_alias_terms')]
            entity_tokens = _semantic_tokens(label_values)
            exact_label = bool(hit_label_key and any((_normalized_label_key(value) == hit_label_key for value in [entity.label, *list(entity.labels_seen or [])])))
            required_coverage = len(required_tokens & entity_tokens) / len(required_tokens) if required_tokens else 0.0
            observed_coverage = len(observed_tokens & entity_tokens) / len(observed_tokens) if observed_tokens else 0.0
            lexical_coverage = max(required_coverage, observed_coverage)
            if not exact_label and lexical_coverage < 0.75:
                continue
            temporal_score = 0.35 if distance == 0 else 0.2 / (1.0 + float(distance or 0))
            score = (1.0 if exact_label else 0.0) + 0.7 * lexical_coverage + temporal_score + min(0.08, 0.01 * int(entity.observation_count or 0))
            candidates.append((score, int(entity.observation_count or 0), entity.entity_id, entity))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05:
            continue
        score, _, _, entity = candidates[0]
        resolved = deepcopy(source)
        resolved['canonical_entity_id'] = entity.entity_id
        resolved['binding_kind'] = 'route_local_label_time_anchor_resolution'
        resolved['binding_confidence'] = round(min(0.95, score / 2.05), 6)
        resolved['uses_gold_answer'] = False
        bound.append(resolved)
    return _dedupe_hits(bound, include_relation=True)

def _group_anchor_hits(hits: Iterable[dict[str, Any]], *, entity_by_id: dict[str, Any]) -> list[tuple[Any, str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    entities: dict[str, Any] = {}
    for hit in hits:
        requested = str(hit.get('requested_relation') or '')
        entity = entity_by_id.get(str(hit.get('canonical_entity_id') or ''))
        if not requested or entity is None:
            continue
        key = (entity.entity_id, requested)
        grouped.setdefault(key, []).append(hit)
        entities[entity.entity_id] = entity
    return [(entities[entity_id], requested, anchor_hits) for (entity_id, requested), anchor_hits in grouped.items()]

def _best_anchor_for_range(hits: Iterable[dict[str, Any]], time_range: Any) -> dict[str, Any]:

    def rank(hit: dict[str, Any]) -> tuple[int, float, str]:
        distance = _frame_distance(_frame(hit), time_range)
        return (distance if distance is not None else 10 ** 6, -float(hit.get('similarity') or 0.0), _hit_key(hit))
    return min(hits, key=rank)

def _target_semantic_scores(hits: list[dict[str, Any]], *, entity_by_id: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for hit in hits:
        if str(hit.get('retrieval_role') or '') not in {'target', 'spatial_target'}:
            continue
        entity = entity_by_id.get(str(hit.get('canonical_entity_id') or ''))
        if entity is None:
            continue
        result[entity.entity_id] = max(result.get(entity.entity_id, 0.0), max(0.0, float(hit.get('similarity') or 0.0)))
    return result

def _rerank_geometry_pairs(pairs: list[dict[str, Any]], *, target_observations: list[dict[str, Any]], entity_by_id: dict[str, Any], target_terms: Any) -> list[dict[str, Any]]:
    semantic_scores = _target_semantic_scores(target_observations, entity_by_id=entity_by_id)
    requested_tokens = _tokens(target_terms) - {'object', 'thing', 'item'}
    result: list[dict[str, Any]] = []
    for source in pairs:
        pair = deepcopy(source)
        entity = entity_by_id.get(str(pair.get('target_entity_id') or ''))
        entity_tokens = _tokens([pair.get('target_label'), getattr(entity, 'labels_seen', []), getattr(entity, 'states_seen', []), getattr(entity, 'attributes', {})])
        modifier_overlap = len(requested_tokens & entity_tokens) / len(requested_tokens) if requested_tokens else 0.0
        semantic_score = semantic_scores.get(str(pair.get('target_entity_id') or ''), 0.0)
        pair['target_query_similarity'] = round(semantic_score, 6)
        pair['target_modifier_overlap'] = round(modifier_overlap, 6)
        geometry_precision = _geometry_precision_score(pair)
        pair['geometry_precision_score'] = round(geometry_precision, 6)
        pair['pair_score'] = round(float(pair.get('pair_score') or 0.0) + 0.7 * semantic_score + 0.3 * modifier_overlap + 0.7 * geometry_precision, 6)
        result.append(pair)
    return result

def _geometry_precision_score(pair: dict[str, Any]) -> float:
    target = _bbox(pair.get('target_bbox'))
    relation = str(pair.get('relation') or '')
    if target is None:
        return 0.0
    raw_anchor = pair.get('anchor_bbox')
    if relation == 'between':
        if not isinstance(raw_anchor, (list, tuple)) or len(raw_anchor) != 2:
            return 0.0
        anchors = [_bbox(item) for item in raw_anchor]
        if any((item is None for item in anchors)):
            return 0.0
        first, second = anchors
        assert first is not None and second is not None
        first_center = _bbox_center(first)
        second_center = _bbox_center(second)
        midpoint = ((first_center[0] + second_center[0]) / 2.0, (first_center[1] + second_center[1]) / 2.0)
        target_center = _bbox_center(target)
        separation = _point_distance(first_center, second_center)
        if separation <= 0.0:
            return 0.0
        midpoint_distance = _point_distance(target_center, midpoint)
        return max(0.0, 1.0 - midpoint_distance / max(1.0, separation / 2.0))
    anchor = _bbox(raw_anchor)
    if anchor is None:
        return 0.0
    if relation in {'left_of', 'right_of'}:
        overlap = _interval_overlap_ratio((target[1], target[3]), (anchor[1], anchor[3]))
        primary_overlap = _interval_overlap_ratio((target[0], target[2]), (anchor[0], anchor[2]))
        if relation == 'left_of':
            gap = max(0.0, anchor[0] - target[2])
        else:
            gap = max(0.0, target[0] - anchor[2])
        scale = max(target[2] - target[0], anchor[2] - anchor[0], 1.0)
    elif relation in {'above', 'below'}:
        overlap = _interval_overlap_ratio((target[0], target[2]), (anchor[0], anchor[2]))
        primary_overlap = _interval_overlap_ratio((target[1], target[3]), (anchor[1], anchor[3]))
        if relation == 'above':
            gap = max(0.0, anchor[1] - target[3])
        else:
            gap = max(0.0, target[1] - anchor[3])
        scale = max(target[3] - target[1], anchor[3] - anchor[1], 1.0)
    else:
        return 0.0
    proximity = 1.0 / (1.0 + gap / scale)
    primary_distinctness = 1.0 - primary_overlap
    return 0.4 * overlap + 0.2 * proximity + 0.4 * primary_distinctness

def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def _point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

def _interval_overlap_ratio(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    denominator = max(1.0, min(first[1] - first[0], second[1] - second[0]))
    return min(1.0, overlap / denominator)

def _best_target_observation(hits: list[dict[str, Any]], *, candidate_entity_id: str, entity_by_id: dict[str, Any], preferred_frame: int | None) -> dict[str, Any] | None:
    candidates = []
    for hit in hits:
        entity = entity_by_id.get(str(hit.get('canonical_entity_id') or ''))
        if entity is None or entity.entity_id != candidate_entity_id:
            continue
        distance = abs(_frame(hit) - preferred_frame) if _frame(hit) is not None and preferred_frame is not None else 10 ** 6
        candidates.append((distance, -float(hit.get('similarity') or 0.0), hit))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None

def _generic_target(label: Any) -> bool:
    tokens = set(_normalized_label_key(label))
    return bool(tokens and tokens <= _GENERIC_TARGET_LABELS)

def _component_target(label: Any) -> bool:
    return bool(_tokens(label) & _COMPONENT_TARGET_TOKENS)

def _anchor_hit_matches_intent(hit: dict[str, Any], intent_text: str) -> bool:
    required = _tokens(intent_text) - _ANCHOR_STOP_TOKENS
    if not required:
        return True
    available = _tokens([hit.get('label'), hit.get('visual_attribute'), hit.get('raw_visual_attributes')])
    return bool(required & available)

def _tokens(values: Any) -> set[str]:
    flattened = values if isinstance(values, (list, tuple, set)) else [values]
    result = set(_TOKEN_PATTERN.findall(' '.join((str(v or '') for v in flattened)).lower()))
    for token in list(result):
        for suffix in ('board', 'door', 'frame', 'shelf', 'table'):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                result.add(suffix)
        if len(token) > 4 and token.endswith('ies'):
            result.add(token[:-3] + 'y')
        elif len(token) > 3 and token.endswith('s'):
            result.add(token[:-1])
    return result

def _semantic_tokens(values: Any) -> set[str]:
    result = _tokens(values)
    for group in _ANCHOR_SYNONYM_GROUPS:
        if result & group:
            result.update(group)
    return result

def _dedupe_hits(hits: Iterable[dict[str, Any]], *, include_relation: bool=False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (_hit_key(hit), str(hit.get('requested_relation') or '') if include_relation else '')
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result

def _hit_key(hit: dict[str, Any]) -> str:
    for key in ('faiss_id', 'observation_id'):
        value = hit.get(key)
        if value not in (None, ''):
            return f'{key}:{value}'
    return 'fallback:' + '|'.join((str(hit.get(key) or '') for key in ('graph_frame_index', 'object_index', 'label', 'canonical_entity_id')))

def _frame(hit: dict[str, Any]) -> int | None:
    return _optional_int(hit.get('graph_frame_index'))

def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _range_representative_frame(time_range: Any) -> int | None:
    start = _optional_int(getattr(time_range, 'start_frame', None))
    end = _optional_int(getattr(time_range, 'end_frame', None))
    if start is not None:
        return start
    return end

def _frame_distance(frame: int | None, time_range: Any) -> int | None:
    if frame is None:
        return None
    start = _optional_int(getattr(time_range, 'start_frame', None))
    end = _optional_int(getattr(time_range, 'end_frame', None))
    if start is None and end is None:
        return None
    start = end if start is None else start
    end = start if end is None else end
    assert start is not None and end is not None
    if start <= frame <= end:
        return 0
    return min(abs(frame - start), abs(frame - end))

def _support_frames(time_range: Any) -> list[int]:
    start = _optional_int(getattr(time_range, 'start_frame', None))
    end = _optional_int(getattr(time_range, 'end_frame', None))
    return _ordered_unique([value for value in (start, end) if value is not None])

def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {key: hit.get(key) for key in ('faiss_id', 'observation_id', 'graph_frame_index', 'object_index', 'label', 'bbox', 'similarity', 'canonical_entity_id', 'binding_kind', 'binding_confidence', 'retrieval_kind')}

def _is_visual_line(value: Any) -> bool:
    text = str(value or '').lower()
    return 'visual_faiss' in text or 'kind=visual_' in text or 'source=visual_' in text or ('visual crop' in text)

def _is_visual_fact(fact: dict[str, Any]) -> bool:
    values = ' '.join((str(fact.get(key) or '') for key in ('fact_type', 'source', 'source_kind', 'source_record_id'))).lower()
    return 'visual' in values or 'faiss' in values

def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
__all__ = ['HANDLER_ID', 'OPEN_TARGET_RELATION_CONTRACT', 'fuse_open_target_relation_evidence']
