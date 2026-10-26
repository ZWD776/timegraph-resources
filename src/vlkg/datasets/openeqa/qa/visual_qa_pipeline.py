"""OpenEQA graph plus evidence-demand visual retrieval QA pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.qa.canonical_graph_qa_context import canonical_query_payload, retrieve_canonical_graph_context
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession, CanonicalRetrievalMode
from vlkg.datasets.openeqa.retrieval.evidence_router import compile_evidence_demand, question_route_decision_from_ir
from vlkg.datasets.openeqa.retrieval.evidence_router import NEED_COUNT_SET, NEED_OCR_LOGO_TEXT, NEED_SPATIAL_LOCALIZATION
from vlkg.datasets.openeqa.qa.graph_qa_pipeline import answer_via_mode
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import fuse_navqa_equivalent_visual_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import fuse_spatial_verified_visual_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v3 import fuse_spatial_guarded_visual_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import fuse_route_dispatched_visual_evidence, route_handler_contract_issues
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import build_visual_query_plan, fuse_single_route_visual_evidence, route_handler_contract_issues as single_route_handler_contract_issues, single_route_decision_from_ir
from vlkg.datasets.openeqa.retrieval.visual_evidence import fuse_graph_visual_evidence, fuse_graph_visual_evidence_guarded_v4, fuse_graph_visual_evidence_guarded_v4c, fuse_graph_visual_evidence_guarded_v4d, fuse_graph_visual_evidence_v2
from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import TemporalGraphDocument
VISUAL_QA_PIPELINE_CONTRACT = 'openeqa_evidence_demand_visual_qa.v1'
VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1 = 'openeqa_navqa_question_router_v1'
VISUAL_BACKBONE_OPENEQA_QR_V2 = 'openeqa_question_router_v2'
VISUAL_BACKBONE_OPENEQA_QR_V3 = 'openeqa_question_router_v3'
VISUAL_BACKBONE_OPENEQA_QR_V4 = 'openeqa_question_router_v4'
VISUAL_BACKBONE_OPENEQA_QR_V5 = 'openeqa_question_router_v5'
VISUAL_BACKBONE_NAVQA_EQUIVALENT = 'navqa_equivalent_v1'
VISUAL_BUILDER_SPATIAL_VERIFIED_V2 = 'navqa_equivalent_spatial_verified_v2'
VISUAL_BUILDER_SPATIAL_GUARDED_V3 = 'navqa_equivalent_spatial_guarded_v3'
VISUAL_BUILDER_ROUTE_DISPATCHED_V4 = 'openeqa_route_dispatched_v4'
VISUAL_BUILDER_ROUTE_DISPATCHED_V5 = 'openeqa_route_dispatched_v5'
VISUAL_BACKBONE_LEGACY_V4 = 'legacy_openeqa_v4'
VISUAL_BACKBONE_ROUTE_HYBRID_V2 = 'route_hybrid_v2'
VISUAL_BACKBONE_GENERIC_GER_V1 = 'generic_ger_v1'
VISUAL_BACKBONE_EVIDENCE_DEMAND_V3 = 'evidence_demand_v3'
VISUAL_BACKBONE_EVIDENCE_DEMAND_V4_GUARDED = 'evidence_demand_v4_guarded'
VISUAL_BACKBONE_EVIDENCE_DEMAND_V4C_GUARDED = 'evidence_demand_v4c_guarded'
VISUAL_BACKBONE_EVIDENCE_DEMAND_V4D_GUARDED = 'evidence_demand_v4d_guarded'
VISUAL_BUILDER_EVIDENCE_DEMAND_V2 = 'evidence_demand_graph_visual_v2'
VISUAL_BUILDER_EVIDENCE_DEMAND_V4_GUARDED = 'evidence_demand_guarded_v4'
VISUAL_BUILDER_EVIDENCE_DEMAND_V4C_GUARDED = 'evidence_demand_guarded_v4c'
VISUAL_BUILDER_EVIDENCE_DEMAND_V4D_GUARDED = 'evidence_demand_guarded_v4d'
VISUAL_BACKBONES = {VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, VISUAL_BACKBONE_OPENEQA_QR_V2, VISUAL_BACKBONE_OPENEQA_QR_V3, VISUAL_BACKBONE_OPENEQA_QR_V4, VISUAL_BACKBONE_OPENEQA_QR_V5, VISUAL_BACKBONE_EVIDENCE_DEMAND_V4D_GUARDED, VISUAL_BACKBONE_EVIDENCE_DEMAND_V4C_GUARDED, VISUAL_BACKBONE_EVIDENCE_DEMAND_V4_GUARDED, VISUAL_BACKBONE_EVIDENCE_DEMAND_V3, VISUAL_BACKBONE_GENERIC_GER_V1, VISUAL_BACKBONE_NAVQA_EQUIVALENT, VISUAL_BACKBONE_LEGACY_V4, VISUAL_BACKBONE_ROUTE_HYBRID_V2}
_NAVQA_SPECIALIZED_EVIDENCE_NEEDS = {NEED_COUNT_SET, NEED_OCR_LOGO_TEXT, NEED_SPATIAL_LOCALIZATION}

def select_visual_builder(visual_backbone: str, *, evidence_demand: dict[str, Any]) -> str:
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1:
        return VISUAL_BACKBONE_NAVQA_EQUIVALENT
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V2:
        return VISUAL_BUILDER_SPATIAL_VERIFIED_V2
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V3:
        return VISUAL_BUILDER_SPATIAL_GUARDED_V3
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V4:
        return VISUAL_BUILDER_ROUTE_DISPATCHED_V4
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
        return VISUAL_BUILDER_ROUTE_DISPATCHED_V5
    if visual_backbone == VISUAL_BACKBONE_EVIDENCE_DEMAND_V4D_GUARDED:
        return VISUAL_BUILDER_EVIDENCE_DEMAND_V4D_GUARDED
    if visual_backbone == VISUAL_BACKBONE_EVIDENCE_DEMAND_V4C_GUARDED:
        return VISUAL_BUILDER_EVIDENCE_DEMAND_V4C_GUARDED
    if visual_backbone == VISUAL_BACKBONE_EVIDENCE_DEMAND_V4_GUARDED:
        return VISUAL_BUILDER_EVIDENCE_DEMAND_V4_GUARDED
    if visual_backbone == VISUAL_BACKBONE_EVIDENCE_DEMAND_V3:
        need = str(evidence_demand.get('primary_need') or '')
        return VISUAL_BUILDER_EVIDENCE_DEMAND_V2 if need == NEED_SPATIAL_LOCALIZATION else VISUAL_BACKBONE_LEGACY_V4
    if visual_backbone == VISUAL_BACKBONE_GENERIC_GER_V1:
        return VISUAL_BACKBONE_GENERIC_GER_V1
    if visual_backbone == VISUAL_BACKBONE_ROUTE_HYBRID_V2:
        need = str(evidence_demand.get('primary_need') or '')
        return VISUAL_BACKBONE_NAVQA_EQUIVALENT if need in _NAVQA_SPECIALIZED_EVIDENCE_NEEDS else VISUAL_BACKBONE_LEGACY_V4
    if visual_backbone in VISUAL_BACKBONES:
        return visual_backbone
    raise ValueError(f'Unknown visual backbone {visual_backbone!r}; expected one of {sorted(VISUAL_BACKBONES)}')

def answer_visual_enhanced_canonical_query(graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, *, question: str, episode_history: str, category: str | None, question_ir: dict[str, Any] | None, visual_index: ObjectVisualSearchIndex, retrieval_mode: CanonicalRetrievalMode='temporal', answer_mode: str='graph_only', config_path: str | None=None, limit: int | None=None, visual_top_k: int=8, visual_min_score: float=0.0, visual_max_candidate_packs: int=8, visual_max_evidence_lines: int=8, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None, retrieval_session: CanonicalHybridRetrievalSession | None=None, domain_profile: str | None=None, graph_qa_config_overrides: dict[str, object] | None=None, visual_backbone: str=VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1) -> dict[str, Any]:
    normalized_question = ' '.join(str(question or '').split())
    if visual_backbone in {VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, VISUAL_BACKBONE_OPENEQA_QR_V2, VISUAL_BACKBONE_OPENEQA_QR_V3, VISUAL_BACKBONE_OPENEQA_QR_V4, VISUAL_BACKBONE_OPENEQA_QR_V5}:
        if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
            demand = single_route_decision_from_ir(question_ir)
        else:
            demand = question_route_decision_from_ir(question_ir)
        if not demand:
            raise ValueError('OpenEQA mainline requires one persisted canonical question route decision')
        if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V4:
            handler_issues = route_handler_contract_issues(demand)
            if handler_issues:
                raise ValueError(handler_issues[0])
        elif visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
            handler_issues = single_route_handler_contract_issues(demand)
            if handler_issues:
                raise ValueError(handler_issues[0])
        route_decision_source = 'question_ir.question_route'
    else:
        demand = compile_evidence_demand(normalized_question, category=category, question_ir=question_ir).to_dict()
        route_decision_source = 'runtime_legacy_compiler'
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
        retrieval_demand = build_visual_query_plan(question_ir or {}, demand)
    else:
        retrieval_demand = generic_ger_visual_demand(normalized_question, demand) if visual_backbone == VISUAL_BACKBONE_GENERIC_GER_V1 else demand
    graph_result = retrieve_canonical_graph_context(graph, query, retrieval_mode=retrieval_mode, limit=limit, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, projection_cache_path=projection_cache_path, retrieval_session=retrieval_session, domain_profile=domain_profile)
    visual_payload = visual_index.search_evidence_demand(episode_history=episode_history, evidence_demand=retrieval_demand, top_k=visual_top_k, min_score=visual_min_score)
    selected_visual_builder = select_visual_builder(visual_backbone, evidence_demand=demand)
    if selected_visual_builder == VISUAL_BACKBONE_NAVQA_EQUIVALENT:
        fused_result = fuse_navqa_equivalent_visual_evidence(graph_result, visual_payload, graph=graph, question=normalized_question, question_ir=question_ir, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines, semantic_text_encoder=visual_index.encode_texts)
    elif selected_visual_builder == VISUAL_BUILDER_SPATIAL_VERIFIED_V2:
        fused_result = fuse_spatial_verified_visual_evidence(graph_result, visual_payload, graph=graph, question=normalized_question, question_ir=question_ir, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines, semantic_text_encoder=visual_index.encode_texts)
    elif selected_visual_builder == VISUAL_BUILDER_SPATIAL_GUARDED_V3:
        fused_result = fuse_spatial_guarded_visual_evidence(graph_result, visual_payload, graph=graph, question=normalized_question, question_ir=question_ir, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines, semantic_text_encoder=visual_index.encode_texts)
    elif selected_visual_builder == VISUAL_BUILDER_ROUTE_DISPATCHED_V4:
        fused_result = fuse_route_dispatched_visual_evidence(graph_result, visual_payload, graph=graph, question=normalized_question, question_ir=question_ir, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines, semantic_text_encoder=visual_index.encode_texts)
    elif selected_visual_builder == VISUAL_BUILDER_ROUTE_DISPATCHED_V5:
        fused_result = fuse_single_route_visual_evidence(graph_result, visual_payload, graph=graph, question=normalized_question, question_ir=question_ir, evidence_demand=retrieval_demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines, semantic_text_encoder=visual_index.encode_texts)
    elif selected_visual_builder == VISUAL_BUILDER_EVIDENCE_DEMAND_V2:
        fused_result = fuse_graph_visual_evidence_v2(graph_result, visual_payload, graph=graph, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    elif selected_visual_builder == VISUAL_BUILDER_EVIDENCE_DEMAND_V4_GUARDED:
        fused_result = fuse_graph_visual_evidence_guarded_v4(graph_result, visual_payload, graph=graph, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    elif selected_visual_builder == VISUAL_BUILDER_EVIDENCE_DEMAND_V4C_GUARDED:
        fused_result = fuse_graph_visual_evidence_guarded_v4c(graph_result, visual_payload, graph=graph, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    elif selected_visual_builder == VISUAL_BUILDER_EVIDENCE_DEMAND_V4D_GUARDED:
        fused_result = fuse_graph_visual_evidence_guarded_v4d(graph_result, visual_payload, graph=graph, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    elif selected_visual_builder == VISUAL_BACKBONE_GENERIC_GER_V1:
        fused_result = fuse_graph_visual_evidence_v2(graph_result, visual_payload, graph=graph, evidence_demand=retrieval_demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    elif selected_visual_builder == VISUAL_BACKBONE_LEGACY_V4:
        fused_result = fuse_graph_visual_evidence(graph_result, visual_payload, graph=graph, evidence_demand=demand, max_candidate_packs=visual_max_candidate_packs, max_visual_lines=visual_max_evidence_lines)
    visual_evidence = fused_result.get('visual_evidence')
    if isinstance(visual_evidence, dict):
        visual_evidence['visual_backbone'] = visual_backbone
        visual_evidence['selected_visual_builder'] = selected_visual_builder
    return {'contract': VISUAL_QA_PIPELINE_CONTRACT, 'visual_backbone': visual_backbone, 'selected_visual_builder': selected_visual_builder, 'question': normalized_question, 'parsed_query': canonical_query_payload(query, retrieval_mode=retrieval_mode), 'evidence_demand': retrieval_demand, 'compiled_question_evidence_demand': demand, 'question_route_decision_source': route_decision_source, 'visual_retrieval': visual_payload, 'graph_answer': fused_result, 'llm_answer': answer_via_mode(answer_mode=answer_mode, config_path=config_path, question=normalized_question, graph_result=fused_result, graph_qa_config_overrides=graph_qa_config_overrides)}

def generic_ger_visual_demand(question: str, compiled_demand: dict[str, Any]) -> dict[str, Any]:
    """Create one broad visual query policy with no question-route branches."""
    terms = _ordered_unique([*compiled_demand.get('target_terms', []), *compiled_demand.get('anchor_terms', [])])
    intents = [{'text': term, 'role': 'target', 'evidence_unit': 'object_crop', 'source': 'generic_entity_recall', 'spatial_relation': '', 'target_terms': terms, 'anchor_terms': []} for term in terms[:6]]
    normalized_question = ' '.join(str(question or '').split())
    if normalized_question:
        intents.append({'text': normalized_question, 'role': 'context', 'evidence_unit': 'object_crop', 'source': 'generic_question_recall', 'spatial_relation': '', 'target_terms': terms, 'anchor_terms': []})
    return {'primary_need': 'open_descriptive', 'route_id': 'generic_ger', 'answer_shape': str(compiled_demand.get('answer_shape') or 'text'), 'target_terms': terms, 'anchor_terms': [], 'spatial_relations': [], 'spatial_constraints': [], 'open_target_slot': False, 'attribute_dimensions': [], 'state_terms': [], 'constraint_terms': [], 'auxiliary_needs': [], 'preferred_units': ['object_crop', 'graph_neighborhood'], 'query_intents': intents, 'retrieval_policy': 'generic_episode_scoped_object_recall', 'evidence_policy': 'generic_ger_graph_bound_candidate_pack', 'requires_negative_coverage': False, 'allows_common_world_knowledge': False, 'confidence': 1.0, 'source': 'generic_non_routed_visual_control', 'contract': 'openeqa_generic_ger_visual_demand.v1', 'temporal_route_enabled': False, 'uses_gold_answer': False, 'metadata': {'question_route_conditioned': False}}

def _ordered_unique(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = ' '.join(str(value or '').split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
