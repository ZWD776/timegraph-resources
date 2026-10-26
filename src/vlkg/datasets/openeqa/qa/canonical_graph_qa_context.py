"""OpenEQA-owned canonical graph QA context fork."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
from typing import Any
from vlkg.core.models.canonical_graph import CanonicalSceneNode, CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.qa.graph_qa_pipeline import answer_via_mode
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE, CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_HYBRID_GRAPH_RETRIEVAL_PROFILE, CanonicalHybridRetrievalSession, CanonicalRetrievalMode, run_canonical_graph_hybrid_retrieval
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import TemporalGraphDocument
FOCUS_FRAME_REASON_WEIGHTS = {'top_match_start': 6, 'top_match_end': 6, 'event_frame': 7, 'query_window_boundary': 7, 'scene_boundary': 4, 'entity_segment_boundary': 4, 'state_interval_boundary': 5, 'relation_interval_boundary': 5, 'scene_base_boundary': 3, 'scene_delta_boundary': 6, 'temporal_chain_boundary': 6}
SCENE_CONTEXT_CUES: dict[str, set[str]] = {'indoor': set(), 'outdoor': set(), 'threshold': set()}
DEFAULT_EVIDENCE_LINE_LIMIT = 6
CANONICAL_VECTOR_HYBRID_EVIDENCE_LINE_LIMIT = 7
CANONICAL_VECTOR_GRAPH_HYBRID_EVIDENCE_LINE_LIMIT = 8
CANONICAL_VECTOR_EVIDENCE_EVIDENCE_LINE_LIMIT = 10

def canonical_query_payload(query: MemoryQuery, *, retrieval_mode: CanonicalRetrievalMode) -> dict[str, object]:
    return {'retrieval_mode': retrieval_mode, 'query_text': query.query_text, 'answer_type': query.answer_type, 'metadata': dict(query.metadata), 'video_id': query.video_id, 'entity_ids': list(query.entity_ids), 'scene_ids': list(query.scene_ids), 'start_ts': query.start_ts, 'end_ts': query.end_ts, 'limit': query.limit}

def retrieve_canonical_graph_context(graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, *, retrieval_mode: CanonicalRetrievalMode='temporal', limit: int | None=None, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None, retrieval_session: CanonicalHybridRetrievalSession | None=None, domain_profile: str | None=None) -> dict[str, Any]:
    resolved_limit = max(1, limit or query.limit or 10)
    if retrieval_session is not None:
        retrieval = retrieval_session.retrieve(query, mode=retrieval_mode, limit=resolved_limit)
    else:
        retrieval = run_canonical_graph_hybrid_retrieval(graph, query, mode=retrieval_mode, limit=resolved_limit, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, projection_cache_path=projection_cache_path, domain_profile=domain_profile)
    matches = list(retrieval.items)
    query_type = f'canonical_{retrieval_mode}'
    filters = canonical_query_payload(query, retrieval_mode=retrieval_mode)
    if include_base_delta_projection:
        filters['include_base_delta_projection'] = temporal_document is not None
    summary = f'Retrieved {len(matches)} canonical graph evidence item(s).' if matches else 'No canonical graph evidence matched the query.'
    retrieval_bundle = _build_retrieval_bundle(graph=graph, query=query, retrieval_mode=retrieval_mode, matches=matches, raw_payload=retrieval.raw_payload)
    llm_ready_bundle = build_canonical_llm_ready_bundle(graph=graph, query=query, retrieval_mode=retrieval_mode, filters=filters, summary=summary, matches=matches, retrieval_bundle=retrieval_bundle)
    return {'query_type': query_type, 'matched_by': 'canonical_graph_hybrid', 'filters': filters, 'summary': summary, 'matches': matches, 'retrieval_bundle': retrieval_bundle, 'llm_ready_bundle': llm_ready_bundle, 'answer_brief': llm_ready_bundle['answer_brief']}

def answer_canonical_graph_query(graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, *, question: str | None=None, retrieval_mode: CanonicalRetrievalMode='temporal', answer_mode: str='graph_only', config_path: str | None=None, limit: int | None=None, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None, retrieval_session: CanonicalHybridRetrievalSession | None=None, domain_profile: str | None=None, graph_qa_config_overrides: dict[str, object] | None=None) -> dict[str, Any]:
    normalized_question = _default_question(query=query, retrieval_mode=retrieval_mode, question=question)
    normalized_query = _query_with_fallback_text(query, fallback_text=normalized_question)
    graph_result = retrieve_canonical_graph_context(graph, normalized_query, retrieval_mode=retrieval_mode, limit=limit, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, projection_cache_path=projection_cache_path, retrieval_session=retrieval_session, domain_profile=domain_profile)
    return {'question': normalized_question, 'parsed_query': canonical_query_payload(normalized_query, retrieval_mode=retrieval_mode), 'graph_answer': graph_result, 'llm_answer': answer_via_mode(answer_mode=answer_mode, config_path=config_path, question=normalized_question, graph_result=graph_result, graph_qa_config_overrides=graph_qa_config_overrides)}

def build_canonical_llm_ready_bundle(*, graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, retrieval_mode: CanonicalRetrievalMode, filters: dict[str, object], summary: str, matches: list[dict[str, Any]], retrieval_bundle: dict[str, Any]) -> dict[str, Any]:
    question_ir = _question_ir_from_query(query)
    question_ir_execution = _build_question_ir_execution_context(question_ir=question_ir, retrieval_bundle=retrieval_bundle)
    primary_matches = _primary_matches_for_display(matches, retrieval_bundle)
    answer_brief = _build_answer_brief(retrieval_mode=retrieval_mode, matches=primary_matches)
    evidence_lines = _build_evidence_lines(primary_matches, limit=_evidence_line_limit(query))
    focus_frame_lines = _build_focus_frame_lines(retrieval_bundle)
    scene_lines = _build_scene_lines(retrieval_bundle)
    temporal_chain_lines = _build_temporal_chain_lines(retrieval_bundle)
    temporal_span_requested = _question_ir_requests_temporal_span(question_ir)
    temporal_span_candidate_lines = _build_temporal_span_candidate_lines(retrieval_bundle, requested=temporal_span_requested)
    evidence_aggregation = _build_evidence_aggregation(question_ir=question_ir, matches=matches, retrieval_bundle=retrieval_bundle)
    evidence_aggregation_lines = _build_evidence_aggregation_lines(evidence_aggregation)
    match_highlights = _build_match_highlights(matches, limit=6)
    uncertainty_notes = _build_uncertainty_notes(query=query, retrieval_mode=retrieval_mode, matches=matches, retrieval_bundle=retrieval_bundle)
    llm_context_text = '\n'.join((line for line in [f'Video: {graph.video_id}', f'Query type: canonical_{retrieval_mode}', f'Filters: {filters}', f'Summary: {summary}', f'Answer brief: {answer_brief}', 'Evidence aggregation:', *[f'- {line}' for line in evidence_aggregation_lines], 'Evidence lines:', *[f'- {line}' for line in evidence_lines], 'Focus frames:', *[f'- {line}' for line in focus_frame_lines], 'Scene notes:', *[f'- {line}' for line in scene_lines], 'Temporal chains:', *[f'- {line}' for line in temporal_chain_lines], *(['Temporal span candidates:', *[f'- {line}' for line in temporal_span_candidate_lines]] if temporal_span_candidate_lines else []), 'Question IR execution:', *[f'- {line}' for line in question_ir_execution.get('execution_lines', [])], 'Match highlights:', *[f'- {line}' for line in match_highlights], 'Uncertainty notes:', *[f'- {line}' for line in uncertainty_notes]] if line))
    return {'video_id': graph.video_id, 'query_type': f'canonical_{retrieval_mode}', 'answer_brief': answer_brief, 'evidence_lines': evidence_lines, 'focus_frame_lines': focus_frame_lines, 'scene_lines': scene_lines, 'temporal_chain_lines': temporal_chain_lines, 'temporal_span_requested': temporal_span_requested, 'temporal_span_candidate_lines': temporal_span_candidate_lines, 'evidence_aggregation': evidence_aggregation, 'evidence_aggregation_lines': evidence_aggregation_lines, 'question_ir': question_ir or {}, 'question_ir_execution': question_ir_execution, 'match_highlights': match_highlights, 'uncertainty_notes': uncertainty_notes, 'llm_context_text': llm_context_text, 'structured_graph_context': {'video_id': graph.video_id, 'query_type': f'canonical_{retrieval_mode}', 'filters': filters, 'summary': summary, 'answer_brief': answer_brief, 'evidence_lines': evidence_lines, 'focus_frames': retrieval_bundle.get('focus_frames', []), 'scene_summaries': retrieval_bundle.get('scene_summaries', []), 'temporal_chains': retrieval_bundle.get('temporal_chains', []), 'temporal_span_requested': temporal_span_requested, 'temporal_span_candidates': retrieval_bundle.get('temporal_span_candidates', []), 'evidence_aggregation': evidence_aggregation, 'question_ir': question_ir or {}, 'question_ir_execution': question_ir_execution, 'evidence_digest': retrieval_bundle.get('evidence_digest', []), 'fragments': retrieval_bundle.get('fragments', []), 'seed_candidates': retrieval_bundle.get('seed_candidates', []), 'entity_aliases': _build_entity_aliases(graph, matches), 'match_highlights': match_highlights, 'uncertainty_notes': uncertainty_notes}}

def _build_retrieval_bundle(*, graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, retrieval_mode: CanonicalRetrievalMode, matches: list[dict[str, Any]], raw_payload: dict[str, Any], max_focus_frames: int=8, max_scene_summaries: int=4) -> dict[str, Any]:
    focus_frames = _build_focus_frames(graph=graph, query=query, matches=matches, max_focus_frames=max_focus_frames)
    temporal_chains = _build_temporal_chains(matches, limit=6)
    scene_summaries = _build_scene_summaries(graph=graph, query=query, matches=matches, focus_frames=focus_frames, temporal_chains=temporal_chains, max_scene_summaries=max_scene_summaries)
    question_ir = _question_ir_from_query(query)
    semantic_event_candidates = _build_semantic_event_candidates_from_ir(graph=graph, question_ir=question_ir)
    temporal_span_candidates = _build_temporal_span_candidates_from_ir(question_ir=question_ir, matches=matches, temporal_chains=temporal_chains)
    return {'video_id': graph.video_id, 'query_type': f'canonical_{retrieval_mode}', 'match_count': len(matches), 'primary_item_ids': list(raw_payload.get('primary_item_ids', [])), 'supplemental_item_ids': list(raw_payload.get('supplemental_item_ids', [])), 'focus_frames': focus_frames, 'scene_summaries': scene_summaries, 'temporal_chains': temporal_chains, 'semantic_event_candidates': semantic_event_candidates, 'temporal_span_candidates': temporal_span_candidates, 'evidence_digest': [{'rank': rank, 'source_record_id': item['source_record_id'], 'source_kind': item['source_kind'], 'summary': item['summary'], 'score': item.get('score')} for rank, item in enumerate(matches[:8], start=1)], 'seed_candidates': list(raw_payload.get('seed_candidates', [])), 'fragments': list(raw_payload.get('fragments', [])), 'query_plan': dict(raw_payload.get('query_plan', {})), 'raw_payload': raw_payload}

def _primary_matches_for_display(matches: list[dict[str, Any]], retrieval_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    primary_ids = [str(value or '').strip() for value in retrieval_bundle.get('primary_item_ids', []) if str(value or '').strip()]
    if not primary_ids:
        return matches
    primary_set = set(primary_ids)
    primary_matches = [item for item in matches if str(item.get('source_record_id') or '').strip() in primary_set]
    return primary_matches or matches

def _evidence_line_limit(query: MemoryQuery) -> int:
    profile = str(query.metadata.get('graph_retrieval_profile') or '').strip()
    if profile == CANONICAL_VECTOR_HYBRID_GRAPH_RETRIEVAL_PROFILE:
        return CANONICAL_VECTOR_HYBRID_EVIDENCE_LINE_LIMIT
    if profile == CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE:
        return CANONICAL_VECTOR_GRAPH_HYBRID_EVIDENCE_LINE_LIMIT
    if profile == CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE:
        return CANONICAL_VECTOR_EVIDENCE_EVIDENCE_LINE_LIMIT
    return DEFAULT_EVIDENCE_LINE_LIMIT

def _build_focus_frames(*, graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, matches: list[dict[str, Any]], max_focus_frames: int) -> list[dict[str, Any]]:
    frame_candidates: dict[int, dict[str, Any]] = {}
    max_match_score = max((float(item.get('score') or 0.0) for item in matches), default=0.0)
    for item in matches[:8]:
        metadata = dict(item.get('metadata', {}))
        time_range = dict(metadata.get('time_range', {}))
        item_score = float(item.get('score') or 0.0)
        scene_id = _scene_id_for_match(metadata)
        frame_specs = _focus_frame_specs_for_item(source_kind=str(item.get('source_kind') or ''), metadata=metadata, time_range=time_range)
        for frame_index, timestamp_sec, reason in frame_specs:
            if frame_index is None:
                continue
            score = item_score + FOCUS_FRAME_REASON_WEIGHTS.get(reason, 1)
            entry = frame_candidates.get(frame_index)
            if entry is None:
                entry = {'frame_index': frame_index, 'frame_rank': frame_index, 'timestamp_sec': timestamp_sec, 'scene_id': scene_id or _scene_id_for_frame(graph, frame_index), 'score': 0.0, 'reasons': set()}
                frame_candidates[frame_index] = entry
            entry['score'] += score
            entry['reasons'].add(reason)
            if entry['timestamp_sec'] is None and timestamp_sec is not None:
                entry['timestamp_sec'] = timestamp_sec
            if entry['scene_id'] is None and scene_id is not None:
                entry['scene_id'] = scene_id
    for frame_index, timestamp_sec, scene_id in _query_window_focus_specs(graph, query):
        score = max_match_score + FOCUS_FRAME_REASON_WEIGHTS['query_window_boundary']
        entry = frame_candidates.get(frame_index)
        if entry is None:
            entry = {'frame_index': frame_index, 'frame_rank': frame_index, 'timestamp_sec': timestamp_sec, 'scene_id': scene_id or _scene_id_for_frame(graph, frame_index), 'score': 0.0, 'reasons': set()}
            frame_candidates[frame_index] = entry
        entry['score'] += score
        entry['reasons'].add('query_window_boundary')
        if entry['timestamp_sec'] is None:
            entry['timestamp_sec'] = timestamp_sec
        if entry['scene_id'] is None and scene_id is not None:
            entry['scene_id'] = scene_id
    ranked = sorted(frame_candidates.values(), key=lambda item: (-float(item['score']), int(item['frame_index'])))
    results: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked[:max_focus_frames], start=1):
        results.append({'rank': rank, 'frame_index': item['frame_index'], 'frame_rank': item['frame_rank'], 'timestamp_sec': item['timestamp_sec'], 'scene_id': item['scene_id'], 'score': round(float(item['score']), 4), 'reasons': sorted(item['reasons'])})
    return results

def _build_scene_summaries(*, graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, matches: list[dict[str, Any]], focus_frames: list[dict[str, Any]], temporal_chains: list[dict[str, Any]], max_scene_summaries: int) -> list[dict[str, Any]]:
    scenes_by_id = {scene.scene_id: scene for scene in graph.scene_nodes}
    ordered_scene_ids: list[str] = []
    seen_scene_ids: set[str] = set()
    for scene_id in query.scene_ids:
        if scene_id in scenes_by_id and scene_id not in seen_scene_ids:
            ordered_scene_ids.append(scene_id)
            seen_scene_ids.add(scene_id)
    for item in matches:
        metadata = dict(item.get('metadata', {}))
        scene_id = _scene_id_for_match(metadata)
        if scene_id and scene_id in scenes_by_id and (scene_id not in seen_scene_ids):
            ordered_scene_ids.append(scene_id)
            seen_scene_ids.add(scene_id)
    for focus_frame in focus_frames:
        scene_id = str(focus_frame.get('scene_id') or '').strip()
        if scene_id and scene_id in scenes_by_id and (scene_id not in seen_scene_ids):
            ordered_scene_ids.append(scene_id)
            seen_scene_ids.add(scene_id)
    for chain in temporal_chains:
        for scene_id in chain.get('scene_ids', []):
            scene_id = str(scene_id or '').strip()
            if scene_id and scene_id in scenes_by_id and (scene_id not in seen_scene_ids):
                ordered_scene_ids.append(scene_id)
                seen_scene_ids.add(scene_id)
    summaries: list[dict[str, Any]] = []
    for scene_id in ordered_scene_ids[:max_scene_summaries]:
        scene = scenes_by_id[scene_id]
        summaries.append(_serialize_scene(scene))
    return summaries

def _serialize_scene(scene: CanonicalSceneNode) -> dict[str, Any]:
    return {'scene_id': scene.scene_id, 'label': scene.label, 'scene_index': scene.scene_index, 'scene_count': scene.scene_count, 'start_frame': scene.time_range.start_frame, 'end_frame': scene.time_range.end_frame, 'start_ts': scene.time_range.start_ts, 'end_ts': scene.time_range.end_ts, 'support_frames': scene.time_range.support_frames, 'scene_kind_hint': scene.scene_kind_hint, 'previous_scene_id': scene.previous_scene_id, 'next_scene_id': scene.next_scene_id, 'representative_static_entity_ids': list(scene.representative_static_entity_ids), 'representative_static_labels': list(scene.representative_static_labels), 'representative_dynamic_labels': list(scene.representative_dynamic_labels), 'entry_static_labels': list(scene.entry_static_labels), 'exit_static_labels': list(scene.exit_static_labels), 'boundary_from_previous_reason': scene.boundary_from_previous_reason, 'boundary_to_next_reason': scene.boundary_to_next_reason, 'boundary_from_previous_entity_similarity': scene.boundary_from_previous_entity_similarity, 'boundary_to_next_entity_similarity': scene.boundary_to_next_entity_similarity, 'boundary_from_previous_label_similarity': scene.boundary_from_previous_label_similarity, 'boundary_to_next_label_similarity': scene.boundary_to_next_label_similarity}

def _build_answer_brief(*, retrieval_mode: CanonicalRetrievalMode, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return 'No canonical graph evidence matched the query.'
    top = matches[0]
    top_summary = str(top.get('summary') or '').strip()
    source_kind = str(top.get('source_kind') or 'unknown').strip()
    if retrieval_mode == 'entity':
        if len(matches) == 1:
            return f'Entity-centered retrieval points to {source_kind}: {top_summary}'
        return f'Entity-centered retrieval found {len(matches)} evidence items. The strongest match is {source_kind}: {top_summary}'
    if len(matches) == 1:
        return f'Temporal retrieval points to {source_kind}: {top_summary}'
    return f'Temporal retrieval found {len(matches)} evidence items. The strongest match is {source_kind}: {top_summary}'

def _build_evidence_lines(matches: list[dict[str, Any]], *, limit: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for rank, item in enumerate(matches[:limit], start=1):
        metadata = dict(item.get('metadata', {}))
        support = dict(metadata.get('hybrid_support', {}))
        scene_id = _scene_id_for_match(metadata)
        entity_ids = _entity_ids_for_match(metadata)
        time_range = dict(metadata.get('time_range', {}))
        relation_types = list(support.get('relation_types', []))
        summary = _compact_summary(str(item.get('summary') or '').strip(), metadata=metadata)
        line_parts = [f'rank={rank}', f"kind={item.get('source_kind')}", f"score={item.get('score')}"]
        if scene_id:
            line_parts.append(f'scene={_display_id(scene_id)}')
        if entity_ids:
            line_parts.append('entities=' + ','.join((_display_id(item) for item in entity_ids[:4])))
        time_text = _time_range_text(time_range)
        if time_text:
            line_parts.append(f'time={time_text}')
        if relation_types:
            line_parts.append('support=' + ','.join(relation_types[:4]))
        line = ' '.join(line_parts) + f' :: {summary}'
        normalized = line.strip()
        if normalized and normalized not in seen:
            lines.append(normalized)
            seen.add(normalized)
    return lines

def _build_focus_frame_lines(retrieval_bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in retrieval_bundle.get('focus_frames', []):
        frame_index = item.get('frame_index')
        timestamp_sec = item.get('timestamp_sec')
        time_part = f'frame={frame_index}'
        if not _timestamp_matches_frame(frame_index, timestamp_sec):
            time_part = f'{time_part} ts={timestamp_sec}'
        lines.append(f"rank={item.get('rank')} {time_part} scene={item.get('scene_id')} score={item.get('score')} reasons={','.join(item.get('reasons', []))}")
    if not lines:
        lines.append('No focus frames were derived from the matched canonical graph evidence.')
    return lines

def _build_scene_lines(retrieval_bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for scene in retrieval_bundle.get('scene_summaries', []):
        details: list[str] = []
        if scene.get('scene_kind_hint'):
            details.append(f"kind={scene.get('scene_kind_hint')}")
        static_labels = list(scene.get('representative_static_labels', []))
        dynamic_labels = list(scene.get('representative_dynamic_labels', []))
        if static_labels:
            details.append('static=' + ','.join(static_labels[:3]))
        if dynamic_labels:
            details.append('dynamic=' + ','.join(dynamic_labels[:3]))
        time_range = {'start_frame': scene.get('start_frame'), 'end_frame': scene.get('end_frame'), 'start_ts': scene.get('start_ts'), 'end_ts': scene.get('end_ts')}
        lines.append(f"scene={scene.get('scene_id')} time={_time_range_text(time_range)} " + ' '.join(details))
    if not lines:
        lines.append('No scene summaries were derived from the matched canonical graph evidence.')
    return lines

def _build_temporal_chains(matches: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in matches:
        if str(item.get('source_kind') or '') != 'temporal_composition':
            continue
        metadata = dict(item.get('metadata', {}))
        scene_sequence = [dict(scene) for scene in metadata.get('scene_sequence', []) if isinstance(scene, dict)]
        scene_ids = tuple((str(scene.get('scene_id') or '').strip() for scene in scene_sequence if str(scene.get('scene_id') or '').strip()))
        if not scene_ids or scene_ids in seen:
            continue
        seen.add(scene_ids)
        chains.append({'source_record_id': item.get('source_record_id'), 'score': item.get('score'), 'composition_type': metadata.get('composition_type'), 'chain_size': metadata.get('chain_size'), 'scene_ids': list(scene_ids), 'time_range': dict(metadata.get('time_range', {})), 'context_sequence': list(metadata.get('context_sequence', [])), 'transition_labels': list(metadata.get('transition_labels', [])), 'scene_sequence': scene_sequence, 'summary': item.get('summary')})
        if len(chains) >= limit:
            break
    return chains

def _build_temporal_chain_lines(retrieval_bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for rank, chain in enumerate(retrieval_bundle.get('temporal_chains', []), start=1):
        time_text = _time_range_text(dict(chain.get('time_range', {})))
        scene_parts: list[str] = []
        for scene in chain.get('scene_sequence', []):
            if not isinstance(scene, dict):
                continue
            scene_time = _time_range_text({'start_frame': scene.get('start_frame'), 'end_frame': scene.get('end_frame'), 'start_ts': scene.get('start_ts'), 'end_ts': scene.get('end_ts')})
            labels = _scene_signature_for_line(scene)
            context = ','.join(scene.get('scene_context_hints', []))
            if context:
                labels = f'{labels};context={context}'
            scene_parts.append(f"{scene.get('scene_id')}[{scene_time or 'unknown time'}:{labels}]")
        transitions = list(chain.get('transition_labels', []))
        line = f"rank={rank} score={chain.get('score')} time={time_text or 'unknown time'} sequence={' -> '.join(scene_parts)}"
        if transitions:
            line += ' transitions=' + ','.join((str(item) for item in transitions[:8]))
        lines.append(line)
    if not lines:
        lines.append('No ordered temporal composition chains were retrieved.')
    return lines

def _build_temporal_span_candidates_from_ir(*, question_ir: dict[str, Any] | None, matches: list[dict[str, Any]], temporal_chains: list[dict[str, Any]], limit: int=8) -> list[dict[str, Any]]:
    if not _question_ir_requests_temporal_span(question_ir):
        return []
    ir = question_ir if isinstance(question_ir, dict) else {}
    target_terms = _temporal_span_target_terms(ir)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, str]] = set()
    for rank, item in enumerate(matches, start=1):
        metadata = dict(item.get('metadata', {}))
        time_range = dict(metadata.get('time_range', {}))
        start_frame = _optional_int(time_range.get('start_frame'))
        end_frame = _optional_int(time_range.get('end_frame'))
        if start_frame is None or end_frame is None:
            continue
        source_kind = str(item.get('source_kind') or '').strip()
        key = (start_frame, end_frame, source_kind)
        if key in seen:
            continue
        seen.add(key)
        summary = _compact_summary(str(item.get('summary') or '').strip(), metadata=metadata)
        evidence_text = _match_semantic_text(item, metadata=metadata, summary=summary)
        evidence_terms = _semantic_term_set([evidence_text])
        target_hits = sorted(target_terms & evidence_terms)
        score = _temporal_span_candidate_score(source_kind=source_kind, rank=rank, target_hits=target_hits, time_range=time_range)
        if score <= 0.0:
            continue
        candidates.append({'candidate_type': 'temporal_span', 'source': 'retrieved_match_time_range', 'rank': rank, 'source_kind': source_kind, 'source_record_id': item.get('source_record_id'), 'score': round(score, 4), 'target_hits': target_hits, 'start_frame': start_frame, 'end_frame': end_frame, 'start_ts': time_range.get('start_ts'), 'end_ts': time_range.get('end_ts'), 'support_frames': time_range.get('support_frames'), 'duration_seconds': _duration_seconds_from_time_range(time_range), 'scene_id': _scene_id_for_match(metadata), 'entity_ids': _entity_ids_for_match(metadata), 'summary': summary, 'warning': _temporal_span_candidate_warning(source_kind)})
    for chain_rank, chain in enumerate(temporal_chains, start=1):
        time_range = dict(chain.get('time_range', {}))
        start_frame = _optional_int(time_range.get('start_frame'))
        end_frame = _optional_int(time_range.get('end_frame'))
        if start_frame is None or end_frame is None:
            continue
        key = (start_frame, end_frame, 'temporal_chain')
        if key in seen:
            continue
        seen.add(key)
        chain_text = ' '.join([str(chain.get('summary') or ''), str(chain.get('context_sequence') or ''), str(chain.get('transition_labels') or ''), str(chain.get('scene_sequence') or '')])
        evidence_terms = _semantic_term_set([chain_text])
        target_hits = sorted(target_terms & evidence_terms)
        if not target_hits:
            continue
        candidates.append({'candidate_type': 'temporal_span', 'source': 'temporal_chain_time_range', 'rank': chain_rank, 'source_kind': 'temporal_chain', 'source_record_id': chain.get('source_record_id'), 'score': round(4.0 + len(target_hits), 4), 'target_hits': target_hits, 'start_frame': start_frame, 'end_frame': end_frame, 'start_ts': time_range.get('start_ts'), 'end_ts': time_range.get('end_ts'), 'support_frames': time_range.get('support_frames'), 'duration_seconds': _duration_seconds_from_time_range(time_range), 'scene_ids': list(chain.get('scene_ids', [])), 'summary': _compact_summary(str(chain.get('summary') or '')), 'warning': 'temporal chain is a candidate span; verify target state before using as duration answer'})
    return sorted(candidates, key=lambda item: (-float(item.get('score') or 0.0), _temporal_span_source_priority(str(item.get('source_kind') or '')), _optional_int(item.get('start_frame')) or 10 ** 12, _optional_int(item.get('end_frame')) or 10 ** 12))[:limit]

def _question_ir_requests_temporal_span(question_ir: dict[str, Any] | None) -> bool:
    if not isinstance(question_ir, dict):
        return False
    operation = str(question_ir.get('operation') or '').strip().lower()
    answer_type = str(question_ir.get('answer_type') or '').strip().lower()
    rendering = str(question_ir.get('rendering') or '').strip().lower()
    aggregation = str(question_ir.get('aggregation') or '').strip().lower()
    return operation == 'interval' or answer_type == 'duration' or rendering == 'duration_minutes' or (aggregation == 'interval')

def _temporal_span_target_terms(question_ir: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for key in ('target_state', 'target_entities', 'event_constraints', 'spatial_constraints', 'hard_constraints'):
        values.extend(_clean_ir_semantic_strings(question_ir.get(key)))
    return _semantic_term_set(values)

def _clean_ir_semantic_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for nested in value.values():
            strings.extend(_clean_ir_semantic_strings(nested))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_clean_ir_semantic_strings(item))
        return strings
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []

def _temporal_span_candidate_score(*, source_kind: str, rank: int, target_hits: list[str], time_range: dict[str, Any]) -> float:
    if not target_hits:
        return 0.0
    base = 10.0 + 2.0 * len(target_hits)
    base += max(0.0, 4.0 - min(rank, 8) * 0.25)
    base += {'episode': 3.0, 'entity': 2.5, 'stable_identity': 2.0, 'entity_attribute': 1.5, 'temporal_composition': 1.0, 'transition_event': -2.0, 'scene': -3.0}.get(source_kind, 0.0)
    duration_seconds = _duration_seconds_from_time_range(time_range)
    if duration_seconds is not None:
        if duration_seconds <= 0.0:
            base -= 4.0
        elif duration_seconds < 5.0:
            base -= 1.5
        elif duration_seconds > 600.0:
            base -= 2.0
    return base

def _duration_seconds_from_time_range(time_range: dict[str, Any]) -> float | None:
    start_ts = _optional_float(time_range.get('start_ts'))
    end_ts = _optional_float(time_range.get('end_ts'))
    if start_ts is None or end_ts is None:
        return None
    return round(abs(end_ts - start_ts), 4)

def _temporal_span_source_priority(source_kind: str) -> int:
    return {'episode': 0, 'entity': 1, 'stable_identity': 2, 'entity_attribute': 3, 'temporal_composition': 4, 'temporal_chain': 5, 'transition_event': 8, 'scene': 9}.get(source_kind, 6)

def _temporal_span_candidate_warning(source_kind: str) -> str:
    if source_kind == 'scene':
        return 'scene-level span is background only; do not use directly as a duration answer unless no finer span exists'
    if source_kind == 'transition_event':
        return 'transition event is a boundary, not a duration span by itself'
    if source_kind == 'temporal_composition':
        return 'temporal composition is a candidate span; verify target entity/state binding'
    return ''

def _build_temporal_span_candidate_lines(retrieval_bundle: dict[str, Any], *, requested: bool, limit: int=8) -> list[str]:
    if not requested:
        return []
    candidates = retrieval_bundle.get('temporal_span_candidates', [])
    if not isinstance(candidates, list) or not candidates:
        return ['No temporal span candidates were derived from Question IR and retrieval evidence.']
    lines: list[str] = []
    for index, item in enumerate(candidates[:limit], start=1):
        if not isinstance(item, dict):
            continue
        time_text = _time_range_text({'start_frame': item.get('start_frame'), 'end_frame': item.get('end_frame'), 'start_ts': item.get('start_ts'), 'end_ts': item.get('end_ts'), 'support_frames': item.get('support_frames')})
        parts = [f'rank={index}', f"source={item.get('source')}", f"kind={item.get('source_kind')}", f"score={item.get('score')}", f"time={time_text or 'unknown time'}"]
        if item.get('duration_seconds') is not None:
            parts.append(f"duration_sec={item.get('duration_seconds')}")
        if item.get('target_hits'):
            parts.append('target_hits=' + ','.join(item['target_hits'][:6]))
        if item.get('entity_ids'):
            parts.append('entities=' + ','.join((_display_id(str(value)) for value in item['entity_ids'][:4])))
        if item.get('scene_id'):
            parts.append(f"scene={_display_id(str(item.get('scene_id')))}")
        if item.get('scene_ids'):
            parts.append('scenes=' + ','.join((_display_id(str(value)) for value in item['scene_ids'][:4])))
        if item.get('warning'):
            parts.append('warning=' + str(item.get('warning')))
        summary = str(item.get('summary') or '').strip()
        if summary:
            parts.append(':: ' + summary)
        lines.append(' '.join(parts))
    return lines

def _build_evidence_aggregation(*, question_ir: dict[str, Any] | None, matches: list[dict[str, Any]], retrieval_bundle: dict[str, Any], max_groups: int=8, max_examples_per_group: int=4) -> dict[str, Any]:
    ir = question_ir if isinstance(question_ir, dict) else {}
    targets = _semantic_term_set(_clean_ir_strings(ir.get('target_entities')))
    target_state_terms = _semantic_term_set(_clean_ir_strings(ir.get('target_state')))
    attribute_terms = _semantic_term_set(_clean_ir_strings(ir.get('attribute_dimension')))
    requirement_terms = _semantic_term_set(_clean_ir_strings(ir.get('evidence_requirements')))
    info_need_terms = _semantic_term_set(_clean_ir_strings(ir.get('graph_information_needs')))
    hard_constraint_groups = _hard_constraint_term_groups(ir)
    hard_constraint_terms = {token for group in hard_constraint_groups for token in group}
    groups_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rank, item in enumerate(matches, start=1):
        metadata = dict(item.get('metadata', {}))
        summary = _compact_summary(str(item.get('summary') or '').strip(), metadata=metadata)
        evidence_text = _match_semantic_text(item, metadata=metadata, summary=summary)
        tokens = _semantic_term_set([evidence_text])
        entity_ids = _entity_ids_for_match(metadata)
        scene_id = _scene_id_for_match(metadata)
        time_range = dict(metadata.get('time_range', {}))
        group_keys = _evidence_group_keys(entity_ids=entity_ids, scene_id=scene_id, metadata=metadata, source_kind=str(item.get('source_kind') or ''))
        for group_key in group_keys:
            group = groups_by_key.setdefault(group_key, {'group_type': group_key[0], 'group_id': group_key[1], 'match_ranks': [], 'source_kinds': [], 'scene_ids': [], 'entity_ids': [], 'time_ranges': [], 'attributes': [], 'states': [], 'relations': [], 'examples': [], 'target_hits': [], 'attribute_hits': [], 'requirement_hits': [], 'info_need_hits': [], 'hard_constraint_hits': [], 'hard_constraint_coverage': 0, 'score': 0.0})
            hard_hits = _hard_constraint_group_hits(tokens, hard_constraint_groups)
            group['score'] = float(group.get('score') or 0.0) + _aggregation_match_score(rank=rank, item=item, tokens=tokens, target_terms=targets | target_state_terms, attribute_terms=attribute_terms, requirement_terms=requirement_terms, info_need_terms=info_need_terms, hard_constraint_group_hits=hard_hits)
            _append_unique(group['match_ranks'], rank)
            _append_unique(group['source_kinds'], str(item.get('source_kind') or ''))
            if scene_id:
                _append_unique(group['scene_ids'], scene_id)
            for entity_id in entity_ids:
                _append_unique(group['entity_ids'], entity_id)
            if time_range:
                _append_unique(group['time_ranges'], time_range)
            _extend_unique(group['attributes'], _attribute_facts_from_metadata(metadata))
            _extend_unique(group['states'], _state_facts_from_metadata(metadata))
            _extend_unique(group['relations'], _relation_facts_from_metadata(metadata))
            _extend_unique(group['target_hits'], sorted(tokens & (targets | target_state_terms)))
            _extend_unique(group['attribute_hits'], sorted(tokens & attribute_terms))
            _extend_unique(group['requirement_hits'], sorted(tokens & requirement_terms))
            _extend_unique(group['info_need_hits'], sorted(tokens & info_need_terms))
            for hit in hard_hits:
                _append_unique(group['hard_constraint_hits'], list(hit))
            group['hard_constraint_coverage'] = len(group['hard_constraint_hits'])
            if len(group['examples']) < max_examples_per_group and summary:
                group['examples'].append({'rank': rank, 'kind': item.get('source_kind'), 'scene_id': scene_id, 'entity_ids': entity_ids, 'time_range': time_range, 'summary': summary})
    groups = sorted(groups_by_key.values(), key=_evidence_aggregation_group_sort_key)[:max_groups]
    candidate_facts = _build_candidate_facts_from_groups(groups, target_terms=targets | target_state_terms, attribute_terms=attribute_terms, requirement_terms=requirement_terms, info_need_terms=info_need_terms, hard_constraint_terms=hard_constraint_terms, limit=max_groups)
    return {'contract': 'canonical_evidence_aggregation_v2', 'available': bool(groups), 'question_ir_summary': {'operation': ir.get('operation', ''), 'answer_type': ir.get('answer_type', ''), 'qa_focus': ir.get('qa_focus', ''), 'aggregation': ir.get('aggregation', ''), 'rendering': ir.get('rendering', ''), 'target_entities': _clean_ir_strings(ir.get('target_entities')), 'target_state': _clean_ir_strings(ir.get('target_state')), 'attribute_dimension': _clean_ir_strings(ir.get('attribute_dimension')), 'hard_constraints': _clean_hard_constraints(ir.get('hard_constraints')), 'evidence_requirements': _clean_ir_strings(ir.get('evidence_requirements')), 'graph_information_needs': _clean_ir_strings(ir.get('graph_information_needs'))}, 'target_terms': sorted(targets | target_state_terms), 'attribute_terms': sorted(attribute_terms), 'requirement_terms': sorted(requirement_terms), 'info_need_terms': sorted(info_need_terms), 'hard_constraint_terms': sorted(hard_constraint_terms), 'groups': groups, 'candidate_facts': candidate_facts, 'temporal_windows': _aggregate_temporal_windows(matches), 'scene_windows': _aggregate_scene_windows(retrieval_bundle)}

def _build_evidence_aggregation_lines(aggregation: dict[str, Any], *, max_lines: int=10) -> list[str]:
    groups = aggregation.get('groups', []) if isinstance(aggregation, dict) else []
    if not groups:
        return ['No evidence aggregation groups were derived from retrieved matches.']
    candidate_facts = aggregation.get('candidate_facts', []) if isinstance(aggregation, dict) else []
    semantic_terms = set(aggregation.get('target_terms', []))
    semantic_terms.update(aggregation.get('attribute_terms', []))
    semantic_terms.update(aggregation.get('requirement_terms', []))
    semantic_terms.update(aggregation.get('info_need_terms', []))
    semantic_terms.update(aggregation.get('hard_constraint_terms', []))
    lines: list[str] = []
    for fact_index, fact in enumerate(candidate_facts[:min(4, max_lines)], start=1):
        if not isinstance(fact, dict):
            continue
        fact_parts = [f'candidate_fact={fact_index}', f"type={fact.get('fact_type')}", f"binding={fact.get('binding_strength')}", 'ranks=' + ','.join((str(rank) for rank in fact.get('support_ranks', [])[:6]))]
        if fact.get('group_id'):
            fact_parts.append(f"group={fact.get('group_type')}:{_display_id(str(fact.get('group_id') or ''))}")
        if fact.get('target_hits'):
            fact_parts.append('target_hits=' + ','.join(fact['target_hits'][:6]))
        if fact.get('requirement_hits'):
            fact_parts.append('requirement_hits=' + ','.join(fact['requirement_hits'][:6]))
        if fact.get('hard_constraint_hits'):
            fact_parts.append('hard_constraint_hits=' + ' | '.join((','.join((str(token) for token in hit[:6])) for hit in fact['hard_constraint_hits'][:4] if isinstance(hit, list))))
        fact_text = str(fact.get('fact_text') or '').strip()
        if fact_text:
            fact_parts.append('fact=' + fact_text)
        examples = [str(value or '').strip() for value in fact.get('support_examples', [])[:2] if str(value or '').strip()]
        if examples:
            fact_parts.append('examples=' + ' || '.join(examples))
        lines.append(' '.join(fact_parts))
    for index, group in enumerate(groups[:max_lines], start=1):
        parts = [f'group={index}', f"{group.get('group_type')}={_display_id(str(group.get('group_id') or ''))}", 'ranks=' + ','.join((str(rank) for rank in group.get('match_ranks', [])[:6]))]
        if group.get('scene_ids'):
            parts.append('scenes=' + ','.join((_display_id(str(scene_id)) for scene_id in group['scene_ids'][:4])))
        if group.get('entity_ids'):
            parts.append('entities=' + ','.join((_display_id(str(entity_id)) for entity_id in group['entity_ids'][:4])))
        if group.get('target_hits'):
            parts.append('target_hits=' + ','.join(group['target_hits'][:6]))
        if group.get('attribute_hits'):
            parts.append('attribute_hits=' + ','.join(group['attribute_hits'][:6]))
        if group.get('requirement_hits'):
            parts.append('requirement_hits=' + ','.join(group['requirement_hits'][:6]))
        if group.get('hard_constraint_hits'):
            parts.append('hard_constraint_hits=' + ' | '.join((','.join((str(token) for token in hit[:6])) for hit in group['hard_constraint_hits'][:4] if isinstance(hit, list))))
        facts = [*[f'attr:{value}' for value in _prioritized_aggregation_values(group.get('attributes', []), semantic_terms=semantic_terms, limit=3)], *[f'state:{value}' for value in _prioritized_aggregation_values(group.get('states', []), semantic_terms=semantic_terms, limit=3)], *[f'relation:{value}' for value in _prioritized_aggregation_values(group.get('relations', []), semantic_terms=semantic_terms, limit=3)]]
        if facts:
            parts.append('facts=' + ' | '.join(facts[:5]))
        examples = _prioritized_aggregation_examples(group.get('examples', []), semantic_terms=semantic_terms, limit=2)
        if examples:
            parts.append('examples=' + ' || '.join(examples))
        lines.append(' '.join(parts))
    return lines

def _build_candidate_facts_from_groups(groups: list[dict[str, Any]], *, target_terms: set[str], attribute_terms: set[str], requirement_terms: set[str], info_need_terms: set[str], hard_constraint_terms: set[str], limit: int) -> list[dict[str, Any]]:
    semantic_terms = set(target_terms)
    semantic_terms.update(attribute_terms)
    semantic_terms.update(requirement_terms)
    semantic_terms.update(info_need_terms)
    semantic_terms.update(hard_constraint_terms)
    facts: list[dict[str, Any]] = []
    for group in groups:
        fact_values = _candidate_fact_values_for_group(group, semantic_terms=semantic_terms)
        binding_strength = _candidate_fact_binding_strength(group)
        for fact_type, fact_text in fact_values:
            facts.append({'fact_type': fact_type, 'fact_text': fact_text, 'binding_strength': binding_strength, 'group_type': group.get('group_type'), 'group_id': group.get('group_id'), 'support_ranks': list(group.get('match_ranks', [])), 'scene_ids': list(group.get('scene_ids', [])), 'entity_ids': list(group.get('entity_ids', [])), 'time_ranges': list(group.get('time_ranges', []))[:3], 'target_hits': list(group.get('target_hits', [])), 'attribute_hits': list(group.get('attribute_hits', [])), 'requirement_hits': list(group.get('requirement_hits', [])), 'info_need_hits': list(group.get('info_need_hits', [])), 'hard_constraint_hits': list(group.get('hard_constraint_hits', [])), 'hard_constraint_coverage': int(group.get('hard_constraint_coverage') or 0), 'support_examples': _prioritized_aggregation_examples(group.get('examples', []), semantic_terms=semantic_terms, limit=2), 'score': float(group.get('score') or 0.0)})
    facts.sort(key=_candidate_fact_sort_key)
    return facts[:limit]

def _candidate_fact_values_for_group(group: dict[str, Any], *, semantic_terms: set[str]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for value in _prioritized_aggregation_values(group.get('attributes', []), semantic_terms=semantic_terms, limit=4):
        values.append(('attribute', value))
    for value in _prioritized_aggregation_values(group.get('states', []), semantic_terms=semantic_terms, limit=4):
        values.append(('state', value))
    for value in _prioritized_aggregation_values(group.get('relations', []), semantic_terms=semantic_terms, limit=4):
        values.append(('relation', value))
    if values:
        return list(dict.fromkeys(values))
    examples = _prioritized_aggregation_examples(group.get('examples', []), semantic_terms=semantic_terms, limit=2)
    return [('summary', value) for value in examples]

def _candidate_fact_binding_strength(group: dict[str, Any]) -> str:
    source_kinds = {str(value or '') for value in group.get('source_kinds', []) or []}
    has_direct_source = bool(source_kinds & {'entity_attribute', 'episode', 'transition_event'})
    has_target = bool(group.get('target_hits'))
    has_requirement = bool(group.get('requirement_hits') or group.get('info_need_hits'))
    has_hard_constraint = bool(group.get('hard_constraint_hits'))
    has_entity_group = str(group.get('group_type') or '') in {'entity', 'identity'}
    if has_direct_source and has_target and has_hard_constraint:
        return 'direct_target_question_factor'
    if has_direct_source and has_target and has_requirement:
        return 'direct_target_requirement'
    if has_direct_source and has_target:
        return 'direct_target'
    if has_entity_group and has_target:
        return 'target_group_context'
    if has_direct_source:
        return 'direct_unbound'
    return 'context_only'

def _candidate_fact_sort_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    binding_order = {'direct_target_question_factor': 0, 'direct_target_requirement': 1, 'direct_target': 2, 'target_group_context': 3, 'direct_unbound': 4, 'context_only': 5}
    fact_type_order = {'attribute': 0, 'state': 1, 'relation': 2, 'summary': 3}
    return (binding_order.get(str(fact.get('binding_strength') or ''), 9), fact_type_order.get(str(fact.get('fact_type') or ''), 9), -len(fact.get('target_hits', []) or []), -int(fact.get('hard_constraint_coverage') or 0), -len(fact.get('requirement_hits', []) or []), -len(fact.get('info_need_hits', []) or []), -float(fact.get('score') or 0.0), min(fact.get('support_ranks') or [10 ** 9]), str(fact.get('fact_text') or ''))

def _evidence_group_keys(*, entity_ids: list[str], scene_id: str | None, metadata: dict[str, Any] | None=None, source_kind: str='') -> list[tuple[str, str]]:
    metadata = metadata if isinstance(metadata, dict) else {}
    keys: list[tuple[str, str]] = []
    if source_kind == 'stable_identity':
        identity_id = str(metadata.get('stable_id') or metadata.get('stable_identity_id') or metadata.get('node_id') or metadata.get('display_name') or metadata.get('stable_identity_display_name') or '').strip()
        if identity_id:
            keys.append(('identity', identity_id))
    else:
        for entity_id in entity_ids:
            if entity_id:
                keys.append(('entity', entity_id))
    if scene_id:
        keys.append(('scene', scene_id))
    if not keys:
        keys.append(('global', 'retrieved_evidence'))
    return keys

def _prioritized_aggregation_values(values: Any, *, semantic_terms: set[str], limit: int) -> list[str]:
    candidates = [str(value or '').strip() for value in values or [] if str(value or '').strip()]
    if not candidates:
        return []
    indexed = list(enumerate(dict.fromkeys(candidates)))

    def sort_key(item: tuple[int, str]) -> tuple[int, int, int, int]:
        original_index, value = item
        tokens = _semantic_term_set([value])
        overlap_count = len(tokens & semantic_terms)
        return (-overlap_count, -len(tokens), original_index, 0)
    return [value for _, value in sorted(indexed, key=sort_key)[:limit]]

def _prioritized_aggregation_examples(examples: Any, *, semantic_terms: set[str], limit: int) -> list[str]:
    rows: list[tuple[int, str]] = []
    for index, example in enumerate(examples or []):
        if not isinstance(example, dict):
            continue
        summary = str(example.get('summary') or '').strip()
        if summary:
            rows.append((index, summary))
    if not rows:
        return []

    def sort_key(item: tuple[int, str]) -> tuple[int, int, int, int]:
        original_index, summary = item
        tokens = _semantic_term_set([summary])
        overlap_count = len(tokens & semantic_terms)
        return (-overlap_count, -len(tokens), original_index, 0)
    return [summary for _, summary in sorted(rows, key=sort_key)[:limit]]

def _aggregation_match_score(*, rank: int, item: dict[str, Any], tokens: set[str], target_terms: set[str], attribute_terms: set[str], requirement_terms: set[str], info_need_terms: set[str], hard_constraint_group_hits: list[tuple[str, ...]]) -> float:
    score = max(0.0, 12.0 - min(rank, 12) * 0.5)
    try:
        score += min(float(item.get('score') or 0.0), 100.0) / 100.0
    except (TypeError, ValueError):
        pass
    score += 4.0 * len(tokens & target_terms)
    score += 3.0 * len(tokens & attribute_terms)
    score += 2.0 * len(tokens & requirement_terms)
    score += 1.5 * len(tokens & info_need_terms)
    score += 5.0 * len(hard_constraint_group_hits)
    score += 1.5 * sum((max(0, len(hit) - 1) for hit in hard_constraint_group_hits))
    source_kind = str(item.get('source_kind') or '')
    if source_kind in {'entity_attribute', 'episode', 'transition_event'}:
        score += 2.0
    elif source_kind in {'entity', 'stable_identity', 'scene'}:
        score += 1.0
    return score

def _evidence_aggregation_group_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    target_coverage = len(group.get('target_hits', []) or [])
    attribute_coverage = len(group.get('attribute_hits', []) or [])
    requirement_coverage = len(group.get('requirement_hits', []) or [])
    info_need_coverage = len(group.get('info_need_hits', []) or [])
    hard_constraint_coverage = int(group.get('hard_constraint_coverage') or 0)
    source_kinds = {str(value or '') for value in group.get('source_kinds', []) or []}
    direct_source = int(bool(source_kinds & {'entity_attribute', 'episode', 'transition_event'}))
    support_count = len(group.get('match_ranks', []) or [])
    best_rank = min(group.get('match_ranks') or [10 ** 9])
    return (-target_coverage, -hard_constraint_coverage, -requirement_coverage, -attribute_coverage, -info_need_coverage, -direct_source, -min(support_count, 5), -float(group.get('score') or 0.0), best_rank, str(group.get('group_type') or ''), str(group.get('group_id') or ''))

def _match_semantic_text(item: dict[str, Any], *, metadata: dict[str, Any], summary: str) -> str:
    values: list[Any] = [item.get('source_kind'), item.get('summary'), summary, metadata.get('label'), metadata.get('canonical_label'), metadata.get('stable_identity_display_name'), metadata.get('stable_identity_canonical_label'), metadata.get('attr_key'), metadata.get('attr_value'), metadata.get('raw_visual_attribute'), metadata.get('predicate'), metadata.get('raw_predicate'), metadata.get('subject_entity_label'), metadata.get('object_entity_label'), metadata.get('event_type'), metadata.get('episode_kind')]
    for key in ('raw_visual_attributes', 'labels_seen', 'states_seen', 'scene_labels'):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(value)
    return ' '.join((str(value) for value in values if str(value or '').strip()))

def _hard_constraint_term_groups(ir: dict[str, Any]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for item in ir.get('hard_constraints') or []:
        texts = _hard_constraint_text_values(item)
        terms = tuple(sorted(_semantic_term_set(texts)))
        if terms:
            groups.append(terms)
    return list(dict.fromkeys(groups))

def _hard_constraint_text_values(item: Any) -> list[str]:
    if isinstance(item, dict):
        values: list[str] = []
        for key in ('text', 'relation', 'operation', 'target_state', 'target', 'target_entities'):
            value = item.get(key)
            if isinstance(value, list):
                values.extend((str(part).strip() for part in value if str(part).strip()))
            elif str(value or '').strip():
                values.append(str(value).strip())
        return values
    if str(item or '').strip():
        return [str(item).strip()]
    return []

def _hard_constraint_group_hits(tokens: set[str], hard_constraint_groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    hits: list[tuple[str, ...]] = []
    for group in hard_constraint_groups:
        overlap = tuple(sorted(tokens & set(group)))
        if overlap:
            hits.append(overlap)
    return hits

def _clean_hard_constraints(value: Any) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in value or []:
        if isinstance(item, dict):
            row: dict[str, Any] = {}
            for key in ('kind', 'text', 'role', 'target_entities', 'must_bind_to'):
                if item.get(key) not in (None, '', [], {}):
                    row[key] = item.get(key)
            if row:
                cleaned.append(row)
        elif str(item or '').strip():
            cleaned.append({'text': str(item).strip()})
    return cleaned

def _semantic_term_set(values: list[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        for token in re.findall('[a-z0-9]+', str(value or '').lower()):
            if token and token not in _AGGREGATION_STOP_TOKENS:
                terms.add(token)
    return terms

def _attribute_facts_from_metadata(metadata: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    attr_key = str(metadata.get('attr_key') or '').strip()
    attr_value = str(metadata.get('attr_value') or '').strip()
    raw_visual_attribute = str(metadata.get('raw_visual_attribute') or '').strip()
    if attr_key and attr_value:
        facts.append(f'{attr_key}={attr_value}')
    elif raw_visual_attribute:
        facts.append(f'visual_attribute={raw_visual_attribute}')
    for value in metadata.get('raw_visual_attributes', []) or []:
        normalized = str(value or '').strip()
        if normalized:
            facts.append(f'visual_attribute={normalized}')
    return facts

def _state_facts_from_metadata(metadata: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    episode_kind = str(metadata.get('episode_kind') or '').strip().upper()
    attr_key = str(metadata.get('attr_key') or '').strip()
    attr_value = str(metadata.get('attr_value') or '').strip()
    if episode_kind == 'STATE' and attr_value:
        facts.append(f"{attr_key or 'state'}={attr_value}")
    for value in metadata.get('states_seen', []) or []:
        normalized = str(value or '').strip()
        if normalized:
            facts.append(normalized)
    return facts

def _relation_facts_from_metadata(metadata: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    predicate = str(metadata.get('raw_predicate') or metadata.get('predicate') or '').strip()
    if predicate:
        subject = str(metadata.get('subject_entity_label') or metadata.get('subject_entity_id') or '').strip()
        obj = str(metadata.get('object_entity_label') or metadata.get('object_entity_id') or '').strip()
        if subject or obj:
            facts.append(f"{subject or '?'} {predicate} {obj or '?'}")
        else:
            facts.append(predicate)
    return facts

def _aggregate_temporal_windows(matches: list[dict[str, Any]], *, limit: int=8) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for rank, item in enumerate(matches, start=1):
        metadata = dict(item.get('metadata', {}))
        time_range = dict(metadata.get('time_range', {}))
        key = (time_range.get('start_frame'), time_range.get('end_frame'), time_range.get('start_ts'), time_range.get('end_ts'))
        if key in seen or all((value is None for value in key)):
            continue
        seen.add(key)
        windows.append({'rank': rank, 'source_kind': item.get('source_kind'), 'scene_id': _scene_id_for_match(metadata), 'time_range': time_range, 'time_text': _time_range_text(time_range)})
        if len(windows) >= limit:
            break
    return windows

def _aggregate_scene_windows(retrieval_bundle: dict[str, Any], *, limit: int=6) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for scene in retrieval_bundle.get('scene_summaries', [])[:limit]:
        if not isinstance(scene, dict):
            continue
        windows.append({'scene_id': scene.get('scene_id'), 'label': scene.get('label'), 'time_text': _time_range_text({'start_frame': scene.get('start_frame'), 'end_frame': scene.get('end_frame'), 'start_ts': scene.get('start_ts'), 'end_ts': scene.get('end_ts')}), 'static_labels': list(scene.get('representative_static_labels', []))[:6], 'dynamic_labels': list(scene.get('representative_dynamic_labels', []))[:6], 'entry_static_labels': list(scene.get('entry_static_labels', []))[:6], 'exit_static_labels': list(scene.get('exit_static_labels', []))[:6]})
    return windows

def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)

def _extend_unique(items: list[Any], values: list[Any]) -> None:
    for value in values:
        if value not in items:
            items.append(value)
_AGGREGATION_STOP_TOKENS = {'a', 'an', 'and', 'answer', 'are', 'as', 'at', 'be', 'been', 'by', 'camera', 'did', 'do', 'does', 'ego', 'evidence', 'for', 'from', 'graph', 'i', 'in', 'is', 'it', 'me', 'need', 'needs', 'of', 'on', 'or', 'robot', 'saw', 'see', 'seen', 'state', 'the', 'this', 'to', 'video', 'visual', 'was', 'we', 'were', 'what', 'when', 'where', 'who', 'with', 'you'}

def _question_ir_from_query(query: MemoryQuery) -> dict[str, Any] | None:
    metadata = getattr(query, 'metadata', {}) or {}
    if not isinstance(metadata, dict):
        return None
    question_ir = metadata.get('question_ir')
    return question_ir if isinstance(question_ir, dict) else None

def _build_question_ir_execution_context(*, question_ir: dict[str, Any] | None, retrieval_bundle: dict[str, Any]) -> dict[str, Any]:
    if not question_ir:
        return {'available': False, 'execution_lines': [], 'semantic_event_candidates': []}
    keys = ['answer_type', 'operation', 'qa_focus', 'target_state', 'target_entities', 'attribute_dimension', 'temporal_relation', 'aggregation', 'rendering', 'evidence_requirements', 'graph_information_needs', 'event_constraints', 'spatial_constraints', 'hard_constraints']
    compact_ir = {key: question_ir.get(key) for key in keys if question_ir.get(key) not in (None, '', [], {})}
    semantic_event_candidates = list(retrieval_bundle.get('semantic_event_candidates', []))
    lines = ['operation={operation} answer_type={answer_type} aggregation={aggregation} rendering={rendering} targets={targets} attributes={attributes}'.format(operation=compact_ir.get('operation', ''), answer_type=compact_ir.get('answer_type', ''), aggregation=compact_ir.get('aggregation', ''), rendering=compact_ir.get('rendering', ''), targets=', '.join(_clean_ir_strings(compact_ir.get('target_entities'))), attributes=', '.join(_clean_ir_strings(compact_ir.get('attribute_dimension'))))]
    evidence_requirements = _clean_ir_strings(compact_ir.get('evidence_requirements'))
    graph_information_needs = _clean_ir_strings(compact_ir.get('graph_information_needs'))
    if evidence_requirements:
        lines.append('evidence_requirements=' + '; '.join(evidence_requirements))
    if graph_information_needs:
        lines.append('graph_information_needs=' + '; '.join(graph_information_needs))
    hard_constraints = _clean_hard_constraints(compact_ir.get('hard_constraints'))
    if hard_constraints:
        lines.append('hard_constraints=' + '; '.join((str(item.get('text') or item).strip() for item in hard_constraints[:6] if str(item.get('text') or item).strip())))
    for index, candidate in enumerate(semantic_event_candidates[:6], start=1):
        if not isinstance(candidate, dict):
            continue
        lines.append('semantic_event_candidate rank={rank} event={event} confidence={confidence} boundary_frame={frame} support={support}'.format(rank=index, event=candidate.get('event_type', ''), confidence=candidate.get('confidence', ''), frame=candidate.get('boundary_frame', ''), support=str(candidate.get('support_summary', ''))))
    return {'available': True, 'ir': compact_ir, 'semantic_event_candidates': semantic_event_candidates, 'execution_lines': lines}

def _build_semantic_event_candidates_from_ir(*, graph: CanonicalTemporalMemoryGraph, question_ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not question_ir:
        return []
    requested_events = _requested_semantic_events(question_ir)
    if not requested_events:
        return []
    scene_rows = [_classify_scene_for_semantic_events(scene) for scene in sorted(graph.scene_nodes, key=lambda item: (item.time_range.start_frame if item.time_range.start_frame is not None else 10 ** 12, item.scene_index, item.scene_id))]
    candidates: list[dict[str, Any]] = []
    for index in range(1, len(scene_rows)):
        previous_scene = scene_rows[index - 1]
        current_scene = scene_rows[index]
        previous_outdoor = 'outdoor' in str(previous_scene['coarse_location'])
        previous_indoor = 'indoor' in str(previous_scene['coarse_location'])
        current_outdoor = 'outdoor' in str(current_scene['coarse_location'])
        current_indoor = 'indoor' in str(current_scene['coarse_location'])
        event_type = ''
        if previous_outdoor and current_indoor:
            event_type = 'enter'
        elif previous_indoor and current_outdoor:
            event_type = 'leave'
        if event_type not in requested_events:
            continue
        confidence = int(previous_scene['context_scores']['outdoor' if previous_outdoor else 'indoor']) + int(current_scene['context_scores']['indoor' if current_indoor else 'outdoor']) + int(previous_scene['context_scores']['threshold']) + int(current_scene['context_scores']['threshold'])
        boundary_frame = current_scene['start_frame']
        candidates.append({'event_type': event_type, 'confidence': confidence, 'boundary_frame': boundary_frame, 'boundary_scene_id': current_scene['scene_id'], 'support_scene_ids': [previous_scene['scene_id'], current_scene['scene_id']], 'support_summary': f"{previous_scene['coarse_location']} -> {current_scene['coarse_location']}", 'support_chain': [_compact_semantic_scene(previous_scene), _compact_semantic_scene(current_scene)]})
    for index in range(1, len(scene_rows) - 1):
        previous_scene = scene_rows[index - 1]
        doorway_scene = scene_rows[index]
        next_scene = scene_rows[index + 1]
        if doorway_scene['coarse_location'] != 'doorway':
            continue
        previous_outdoor = 'outdoor' in str(previous_scene['coarse_location'])
        previous_indoor = 'indoor' in str(previous_scene['coarse_location'])
        next_outdoor = 'outdoor' in str(next_scene['coarse_location'])
        next_indoor = 'indoor' in str(next_scene['coarse_location'])
        event_type = ''
        if previous_outdoor and next_indoor:
            event_type = 'enter'
        elif previous_indoor and next_outdoor:
            event_type = 'leave'
        if event_type not in requested_events:
            continue
        confidence = int(previous_scene['context_scores']['outdoor' if previous_outdoor else 'indoor']) + int(doorway_scene['context_scores']['threshold']) + int(next_scene['context_scores']['indoor' if next_indoor else 'outdoor'])
        candidates.append({'event_type': event_type, 'confidence': confidence, 'boundary_frame': next_scene['start_frame'], 'boundary_scene_id': next_scene['scene_id'], 'support_scene_ids': [previous_scene['scene_id'], doorway_scene['scene_id'], next_scene['scene_id']], 'support_summary': f"{previous_scene['coarse_location']} -> {doorway_scene['coarse_location']} -> {next_scene['coarse_location']}", 'support_chain': [_compact_semantic_scene(previous_scene), _compact_semantic_scene(doorway_scene), _compact_semantic_scene(next_scene)]})
    ordering = str(question_ir.get('temporal_relation') or '').strip().lower()
    if 'first' in ordering:
        return sorted(candidates, key=lambda item: (_semantic_candidate_frame(item, missing=10 ** 12), -int(item.get('confidence') or 0)))
    if 'last' in ordering or 'latest' in ordering:
        return sorted(candidates, key=lambda item: (-_semantic_candidate_frame(item, missing=-1), -int(item.get('confidence') or 0)))
    return sorted(candidates, key=lambda item: -int(item.get('confidence') or 0))

def _semantic_candidate_frame(item: dict[str, Any], *, missing: int) -> int:
    frame = _optional_int(item.get('boundary_frame'))
    return frame if frame is not None else missing

def _requested_semantic_events(question_ir: dict[str, Any]) -> set[str]:
    operation = str(question_ir.get('operation') or '').strip().lower()
    target_state = ' '.join(_clean_ir_strings(question_ir.get('target_state')))
    event_text = ' '.join([operation, target_state, ' '.join(_clean_ir_strings(question_ir.get('target_entities'))), ' '.join(_clean_ir_strings(question_ir.get('evidence_requirements'))), ' '.join(_clean_ir_strings(question_ir.get('graph_information_needs')))])
    for item in question_ir.get('event_constraints') or []:
        if isinstance(item, dict):
            for value in item.values():
                event_text += ' ' + ' '.join(_clean_ir_strings(value))
        else:
            event_text += ' ' + str(item)
    events: set[str] = set()
    if operation == 'onset' or re.search('\\b(enter|entered|entry|inside|in building)\\b', event_text):
        events.add('enter')
    if operation == 'offset' or re.search('\\b(leave|left|exit|exited|outside)\\b', event_text):
        events.add('leave')
    if operation == 'interval':
        events.update({'enter', 'leave'})
    return events

def _classify_scene_for_semantic_events(scene: CanonicalSceneNode) -> dict[str, Any]:
    tokens = _scene_semantic_tokens(scene)
    scores = {context: len(tokens & cues) for context, cues in SCENE_CONTEXT_CUES.items()}
    if scores['indoor'] >= scores['outdoor'] + 2 and scores['indoor'] >= scores['threshold']:
        coarse = 'indoor'
    elif scores['outdoor'] >= scores['indoor'] + 2 and scores['outdoor'] >= scores['threshold']:
        coarse = 'outdoor'
    elif scores['threshold'] >= 2:
        coarse = 'doorway'
    elif scores['indoor'] > scores['outdoor']:
        coarse = 'maybe_indoor'
    elif scores['outdoor'] > scores['indoor']:
        coarse = 'maybe_outdoor'
    else:
        coarse = 'unknown'
    return {'scene_id': scene.scene_id, 'scene_index': scene.scene_index, 'label': scene.label, 'start_frame': scene.time_range.start_frame, 'end_frame': scene.time_range.end_frame, 'start_ts': scene.time_range.start_ts, 'end_ts': scene.time_range.end_ts, 'coarse_location': coarse, 'context_scores': scores, 'matched_cues': {context: sorted(tokens & cues) for context, cues in SCENE_CONTEXT_CUES.items()}}

def _compact_semantic_scene(scene_row: dict[str, Any]) -> dict[str, Any]:
    return {'scene_id': scene_row['scene_id'], 'label': scene_row['label'], 'frames': [scene_row['start_frame'], scene_row['end_frame']], 'coarse_location': scene_row['coarse_location'], 'context_scores': scene_row['context_scores'], 'matched_cues': scene_row['matched_cues']}

def _scene_semantic_tokens(scene: CanonicalSceneNode) -> set[str]:
    tokens: set[str] = set()
    for value in [scene.label, scene.scene_kind_hint, *scene.representative_static_labels, *scene.representative_dynamic_labels, *scene.entry_static_labels, *scene.exit_static_labels, *scene.salient_labels, *scene.semantic_alias_terms]:
        text = str(value or '').strip().lower()
        if not text:
            continue
        for atom in re.split('\\s*/\\s*|\\s*\\|\\s*|,\\s*', text):
            tokens.update(_tokenize_semantic_text(atom))
    return tokens

def _tokenize_semantic_text(text: str) -> set[str]:
    return {token for token in re.findall('[a-z0-9]+', str(text or '').lower()) if token}

def _clean_ir_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item or '').strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []

def _scene_signature_for_line(scene: dict[str, Any]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for value in [scene.get('scene_kind_hint'), scene.get('label'), *list(scene.get('scene_context_hints', [])), *list(scene.get('representative_static_labels', [])), *list(scene.get('representative_dynamic_labels', [])), *list(scene.get('entry_static_labels', [])), *list(scene.get('exit_static_labels', []))]:
        normalized = str(value or '').strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        labels.append(normalized)
        seen.add(key)
        if len(labels) >= 4:
            break
    return ','.join(labels) if labels else 'no scene labels'

def _build_match_highlights(matches: list[dict[str, Any]], *, limit: int) -> list[str]:
    highlights: list[str] = []
    for item in matches[:limit]:
        metadata = dict(item.get('metadata', {}))
        support = dict(metadata.get('hybrid_support', {}))
        scene_id = _scene_id_for_match(metadata)
        entity_ids = _entity_ids_for_match(metadata)
        relation_types = list(support.get('relation_types', []))
        path_edge_types = list(support.get('path_edge_types', []))
        parts = [f"{item.get('source_kind')}"]
        if scene_id:
            parts.append(f'scene={_display_id(scene_id)}')
        if entity_ids:
            parts.append('entities=' + ','.join((_display_id(item) for item in entity_ids[:4])))
        if relation_types:
            parts.append('relations=' + ','.join(relation_types[:4]))
        if path_edge_types:
            parts.append('path=' + '->'.join(path_edge_types[:4]))
        highlights.append(' '.join(parts))
    return highlights

def _build_uncertainty_notes(*, query: MemoryQuery, retrieval_mode: CanonicalRetrievalMode, matches: list[dict[str, Any]], retrieval_bundle: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if not matches:
        notes.append('The canonical temporal memory graph did not return any matching evidence.')
        return notes
    unique_source_kinds = {str(item.get('source_kind') or '').strip() for item in matches if str(item.get('source_kind') or '').strip()}
    unique_scene_ids = {str(scene.get('scene_id') or '').strip() for scene in retrieval_bundle.get('scene_summaries', []) if str(scene.get('scene_id') or '').strip()}
    if len(unique_scene_ids) > 1:
        notes.append('Evidence spans multiple scenes; the final answer should preserve scene boundaries.')
    if retrieval_mode == 'temporal' and len(unique_source_kinds) > 1:
        notes.append('Temporal retrieval mixes multiple node types; the answer should distinguish events, episodes, and scene context.')
    if retrieval_mode == 'entity' and any((str(item.get('source_kind') or '').strip() == 'episode' for item in matches)):
        notes.append('Entity retrieval includes lifecycle episodes, so the answer should preserve temporal ordering rather than collapse everything into one static description.')
    if query.start_ts is not None or query.end_ts is not None:
        notes.append('The query includes an explicit time window; claims outside that window should be avoided unless marked as context.')
    if any((int(dict(item.get('metadata', {})).get('hybrid_support', {}).get('hop_count', 0) or 0) > 1 for item in matches)):
        notes.append('Some evidence is connected through multi-hop graph expansion rather than direct seed matches.')
    if not retrieval_bundle.get('focus_frames'):
        notes.append('No focus frames were derived, so the answer should rely on textual graph evidence only.')
    return notes

def _focus_frame_specs_for_item(*, source_kind: str, metadata: dict[str, Any], time_range: dict[str, Any]) -> list[tuple[int | None, float | None, str]]:
    start_frame = _optional_int(time_range.get('start_frame'))
    end_frame = _optional_int(time_range.get('end_frame'))
    start_ts = _optional_float(time_range.get('start_ts'))
    end_ts = _optional_float(time_range.get('end_ts'))
    episode_kind = str(metadata.get('episode_kind') or '').strip()
    event_type = str(metadata.get('event_type') or '').strip()
    specs: list[tuple[int | None, float | None, str]] = []
    if source_kind == 'transition_event':
        specs.append((start_frame, start_ts, 'event_frame'))
        return specs
    if source_kind == 'scene':
        specs.append((start_frame, start_ts, 'scene_boundary'))
        if end_frame != start_frame:
            specs.append((end_frame, end_ts, 'scene_boundary'))
        return specs
    if source_kind == 'scene_base':
        specs.append((start_frame, start_ts, 'scene_base_boundary'))
        if end_frame != start_frame:
            specs.append((end_frame, end_ts, 'scene_base_boundary'))
        return specs
    if source_kind == 'scene_delta':
        return _scene_delta_focus_specs(metadata, time_range)
    if source_kind == 'entity_attribute':
        specs.append((start_frame, start_ts, 'entity_attribute_boundary'))
        if end_frame != start_frame:
            specs.append((end_frame, end_ts, 'entity_attribute_boundary'))
        return specs
    if source_kind == 'temporal_composition':
        specs.append((start_frame, start_ts, 'temporal_chain_boundary'))
        if end_frame != start_frame:
            specs.append((end_frame, end_ts, 'temporal_chain_boundary'))
        for scene in metadata.get('scene_sequence') or []:
            if not isinstance(scene, dict):
                continue
            scene_start_frame = _optional_int(scene.get('start_frame'))
            scene_end_frame = _optional_int(scene.get('end_frame'))
            scene_start_ts = _optional_float(scene.get('start_ts'))
            scene_end_ts = _optional_float(scene.get('end_ts'))
            specs.append((scene_start_frame, scene_start_ts, 'scene_boundary'))
            if scene_end_frame != scene_start_frame:
                specs.append((scene_end_frame, scene_end_ts, 'scene_boundary'))
        return specs
    if source_kind == 'episode':
        if episode_kind == 'ENTITY_SEGMENT':
            specs.append((start_frame, start_ts, 'entity_segment_boundary'))
            if end_frame != start_frame:
                specs.append((end_frame, end_ts, 'entity_segment_boundary'))
            return specs
        if episode_kind == 'STATE':
            specs.append((start_frame, start_ts, 'state_interval_boundary'))
            if end_frame != start_frame:
                specs.append((end_frame, end_ts, 'state_interval_boundary'))
            return specs
        if episode_kind == 'RELATION_OBSERVED_BURST':
            return _relation_observed_burst_focus_specs(metadata, time_range)
        if episode_kind == 'RELATION':
            specs.append((start_frame, start_ts, 'relation_interval_boundary'))
            if end_frame != start_frame:
                specs.append((end_frame, end_ts, 'relation_interval_boundary'))
            return specs
    specs.append((start_frame, start_ts, 'top_match_start'))
    if end_frame != start_frame:
        specs.append((end_frame, end_ts, 'top_match_end'))
    if event_type and (not specs):
        specs.append((start_frame, start_ts, 'event_frame'))
    return specs

def _relation_observed_burst_focus_specs(metadata: dict[str, Any], time_range: dict[str, Any]) -> list[tuple[int | None, float | None, str]]:
    observed_frames = [_optional_int(value) for value in metadata.get('observed_frames', [])]
    observed_timestamps = [_optional_float(value) for value in metadata.get('observed_timestamps', [])]
    ordered_pairs = [(frame, timestamp) for frame, timestamp in zip(observed_frames, observed_timestamps, strict=False) if frame is not None]
    if not ordered_pairs:
        start_frame = _optional_int(time_range.get('start_frame'))
        end_frame = _optional_int(time_range.get('end_frame'))
        start_ts = _optional_float(time_range.get('start_ts'))
        end_ts = _optional_float(time_range.get('end_ts'))
        specs: list[tuple[int | None, float | None, str]] = []
        specs.append((start_frame, start_ts, 'relation_interval_boundary'))
        if end_frame != start_frame:
            specs.append((end_frame, end_ts, 'relation_interval_boundary'))
        return specs
    sample_indices = {0, len(ordered_pairs) // 2, len(ordered_pairs) - 1}
    specs = [(ordered_pairs[index][0], ordered_pairs[index][1], 'event_frame') for index in sorted(sample_indices)]
    return specs

def _scene_delta_focus_specs(metadata: dict[str, Any], time_range: dict[str, Any]) -> list[tuple[int | None, float | None, str]]:
    observed_frames = [_optional_int(value) for value in metadata.get('observed_frames', [])]
    observed_timestamps = [_optional_float(value) for value in metadata.get('observed_timestamps', [])]
    ordered_pairs = [(frame, timestamp) for frame, timestamp in zip(observed_frames, observed_timestamps, strict=False) if frame is not None]
    if ordered_pairs:
        sample_indices = {0, len(ordered_pairs) // 2, len(ordered_pairs) - 1}
        return [(ordered_pairs[index][0], ordered_pairs[index][1], 'scene_delta_boundary') for index in sorted(sample_indices)]
    start_frame = _optional_int(time_range.get('start_frame'))
    end_frame = _optional_int(time_range.get('end_frame'))
    start_ts = _optional_float(time_range.get('start_ts'))
    end_ts = _optional_float(time_range.get('end_ts'))
    specs: list[tuple[int | None, float | None, str]] = [(start_frame, start_ts, 'scene_delta_boundary')]
    if end_frame != start_frame:
        specs.append((end_frame, end_ts, 'scene_delta_boundary'))
    return specs

def _scene_id_for_match(metadata: dict[str, Any]) -> str | None:
    scene_id = str(metadata.get('scene_id') or '').strip()
    if scene_id:
        return scene_id
    scene_ids = [str(value or '').strip() for value in metadata.get('scene_ids', [])]
    scene_ids = [value for value in scene_ids if value]
    if len(scene_ids) == 1:
        return scene_ids[0]
    return None

def _entity_ids_for_match(metadata: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    candidates = [metadata.get('entity_id'), metadata.get('primary_entity_id'), metadata.get('subject_entity_id'), metadata.get('object_entity_id'), *list(metadata.get('participant_entity_ids', [])), *list(metadata.get('member_entity_ids', []))]
    for value in candidates:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered

def _scene_id_for_frame(graph: CanonicalTemporalMemoryGraph, frame_index: int) -> str | None:
    for scene in graph.scene_nodes:
        start_frame = scene.time_range.start_frame
        end_frame = scene.time_range.end_frame
        if start_frame is None or end_frame is None:
            continue
        if start_frame <= frame_index <= end_frame:
            return scene.scene_id
    return None

def _query_window_focus_specs(graph: CanonicalTemporalMemoryGraph, query: MemoryQuery) -> list[tuple[int, float, str | None]]:
    specs: list[tuple[int, float, str | None]] = []
    seen_frames: set[int] = set()
    for timestamp_sec in (query.start_ts, query.end_ts):
        if timestamp_sec is None:
            continue
        frame_index, scene_id = _frame_for_timestamp(graph, float(timestamp_sec))
        if frame_index is None or frame_index in seen_frames:
            continue
        specs.append((frame_index, float(timestamp_sec), scene_id))
        seen_frames.add(frame_index)
    return specs

def _frame_for_timestamp(graph: CanonicalTemporalMemoryGraph, timestamp_sec: float) -> tuple[int | None, str | None]:
    best_frame: int | None = None
    best_scene_id: str | None = None
    best_gap = float('inf')
    for scene in graph.scene_nodes:
        time_range = scene.time_range
        if time_range.start_frame is None or time_range.end_frame is None or time_range.start_ts is None or (time_range.end_ts is None):
            continue
        if time_range.start_ts <= timestamp_sec <= time_range.end_ts:
            duration = time_range.end_ts - time_range.start_ts
            if duration <= 0:
                frame_index = time_range.start_frame
            else:
                position = (timestamp_sec - time_range.start_ts) / duration
                frame_index = round(time_range.start_frame + position * (time_range.end_frame - time_range.start_frame))
            return (int(frame_index), scene.scene_id)
        scene_start_gap = abs(timestamp_sec - time_range.start_ts)
        scene_end_gap = abs(timestamp_sec - time_range.end_ts)
        scene_gap = min(scene_start_gap, scene_end_gap)
        if scene_gap < best_gap:
            best_gap = scene_gap
            best_scene_id = scene.scene_id
            best_frame = time_range.start_frame if scene_start_gap <= scene_end_gap else time_range.end_frame
    return (best_frame, best_scene_id)

def _build_entity_aliases(graph: CanonicalTemporalMemoryGraph, matches: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    entity_ids: list[str] = []
    seen: set[str] = set()
    for item in matches:
        metadata = dict(item.get('metadata', {}))
        for entity_id in _entity_ids_for_match(metadata):
            if entity_id in seen:
                continue
            entity_ids.append(entity_id)
            seen.add(entity_id)
    entity_by_id = {node.entity_id: node for node in graph.entity_nodes}
    aliases: dict[str, dict[str, str]] = {}
    for entity_id in entity_ids:
        entity = entity_by_id.get(entity_id)
        if entity is None:
            continue
        short_id = _display_id(entity_id)
        stable_name = str(entity.stable_identity_display_name or '').strip()
        label = str(entity.label or '').strip()
        canonical_label = str(entity.stable_identity_canonical_label or '').strip()
        readable_parts = [value for value in (stable_name, label or canonical_label) if value]
        readable_name = ' / '.join(dict.fromkeys(readable_parts))
        aliases[entity_id] = {'entity_id': entity_id, 'short_id': short_id, 'stable_name': stable_name, 'label': label, 'canonical_label': canonical_label, 'readable_name': readable_name, 'display_text': _entity_display_text(stable_name=stable_name, label=label or canonical_label, short_id=short_id)}
    return aliases

def _time_range_text(time_range: dict[str, Any]) -> str | None:
    start_frame = time_range.get('start_frame')
    end_frame = time_range.get('end_frame')
    start_ts = time_range.get('start_ts')
    end_ts = time_range.get('end_ts')
    if start_frame is None and end_frame is None and (start_ts is None) and (end_ts is None):
        return None
    if _timestamps_match_frames(start_frame, end_frame, start_ts, end_ts):
        return _frame_range_text(start_frame, end_frame)
    frame_text = None
    if start_frame is not None and end_frame is not None:
        frame_text = _frame_range_text(start_frame, end_frame)
    ts_text = None
    if start_ts is not None and end_ts is not None:
        ts_text = f'{float(start_ts):.2f}s->{float(end_ts):.2f}s'
    if frame_text and ts_text:
        return f'{frame_text} {ts_text}'
    return frame_text or ts_text

def _timestamps_match_frames(start_frame: Any, end_frame: Any, start_ts: Any, end_ts: Any) -> bool:
    if start_frame is None or end_frame is None or start_ts is None or (end_ts is None):
        return False
    try:
        return abs(float(start_frame) - float(start_ts)) < 1e-06 and abs(float(end_frame) - float(end_ts)) < 1e-06
    except (TypeError, ValueError):
        return False

def _timestamp_matches_frame(frame_index: Any, timestamp_sec: Any) -> bool:
    if frame_index is None or timestamp_sec is None:
        return False
    try:
        return abs(float(frame_index) - float(timestamp_sec)) < 1e-06
    except (TypeError, ValueError):
        return False

def _frame_range_text(start_frame: Any, end_frame: Any) -> str:
    if start_frame == end_frame:
        return f'frame {start_frame}'
    return f'frames {start_frame}-{end_frame}'

def _compact_summary(summary: str, *, metadata: dict[str, Any]) -> str:
    if not summary:
        return summary
    raw_visual_attribute = str(metadata.get('raw_visual_attribute') or '').strip()
    if raw_visual_attribute:
        label = str(metadata.get('label') or metadata.get('canonical_label') or 'entity').strip()
        return f'{label} visual_attribute={raw_visual_attribute}'
    entity_ids = _entity_ids_for_match(metadata)
    compact = summary
    for entity_id in sorted(entity_ids, key=len, reverse=True):
        compact = compact.replace(entity_id, _display_id(entity_id))
    scene_id = _scene_id_for_match(metadata)
    if scene_id:
        compact = compact.replace(scene_id, _display_id(scene_id))
    source_record_id = str(metadata.get('source_ref_id') or '').strip()
    if source_record_id:
        compact = compact.replace(source_record_id, _display_id(source_record_id))
    return compact

def _default_question(*, query: MemoryQuery, retrieval_mode: CanonicalRetrievalMode, question: str | None) -> str:
    if question and str(question).strip():
        return str(question).strip()
    parts: list[str] = [f'canonical {retrieval_mode} query']
    if query.query_text:
        parts.append(f'text={query.query_text}')
    if query.entity_ids:
        parts.append('entities=' + ','.join(query.entity_ids))
    if query.scene_ids:
        parts.append('scenes=' + ','.join(query.scene_ids))
    if query.start_ts is not None or query.end_ts is not None:
        parts.append(f'time={query.start_ts}->{query.end_ts}')
    return ' | '.join(parts)

def _query_with_fallback_text(query: MemoryQuery, *, fallback_text: str) -> MemoryQuery:
    if str(query.query_text or '').strip():
        return query
    if not fallback_text.strip():
        return query
    return query.model_copy(update={'query_text': fallback_text})

def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None

def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None

def _display_id(value: str) -> str:
    text = str(value or '').strip()
    if not text:
        return text
    parts = [part for part in text.split(':') if part]
    if len(parts) >= 2:
        return ':'.join(parts[-2:])
    return text

def _entity_display_text(*, stable_name: str, label: str, short_id: str) -> str:
    if stable_name and label:
        return f'{stable_name} ({label}, {short_id})'
    if stable_name:
        return f'{stable_name} ({short_id})'
    if label:
        return f'{label} ({short_id})'
    return short_id
