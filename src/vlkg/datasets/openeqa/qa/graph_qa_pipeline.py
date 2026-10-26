"""OpenEQA-owned graph QA mode dispatcher fork."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from vlkg.core.config.config import load_domain_config
from vlkg.datasets.openeqa.qa.graph_qa_answering import build_graph_qa_responder, graph_only_answer
from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.graph_retrieval.query_engine import TemporalGraphQueryEngine
from vlkg.core.graph_retrieval.query_parser import ParsedQuery, parse_question, repair_question_text

def load_document(path: str | Path) -> TemporalGraphDocument:
    document_path = Path(path)
    return TemporalGraphDocument.model_validate_json(document_path.read_text(encoding='utf-8'))

def parsed_query_payload(parsed: ParsedQuery) -> dict[str, object]:
    return {'query_type': parsed.query_type, 'filters': parsed.filters, 'confidence': parsed.confidence, 'reasoning': parsed.reasoning, 'extracted_spans': parsed.extracted_spans}

def execute_parsed_query(engine: TemporalGraphQueryEngine, parsed: ParsedQuery) -> dict[str, object]:
    if parsed.query_type == 'entity':
        return engine.answer_entity_timeline(**parsed.filters)
    if parsed.query_type == 'frame':
        return engine.answer_frame_snapshot(**parsed.filters)
    if parsed.query_type == 'scene':
        return engine.answer_scene_summary(**parsed.filters)
    if parsed.query_type == 'state':
        return engine.answer_state_history(**parsed.filters)
    if parsed.query_type == 'relation':
        return engine.answer_relation_history(**parsed.filters)
    raise ValueError(f'Unsupported parsed query type: {parsed.query_type}')

def answer_via_mode(*, answer_mode: str, config_path: str | None, question: str, graph_result: dict[str, object], graph_qa_config_overrides: dict[str, object] | None=None) -> dict[str, object]:
    if answer_mode == 'graph_only':
        return graph_only_answer(question=question, graph_result=graph_result)
    config = load_domain_config(config_path)
    graph_qa_config = config.graph_qa
    if graph_qa_config_overrides:
        graph_qa_config = graph_qa_config.model_copy(update=dict(graph_qa_config_overrides))
    responder = build_graph_qa_responder(graph_qa_config, mode=answer_mode)
    if responder is None:
        return graph_only_answer(question=question, graph_result=graph_result)
    return responder.answer(question=question, graph_result=graph_result)

def answer_question(engine: TemporalGraphQueryEngine, question: str, *, answer_mode: str='graph_only', config_path: str | None=None) -> dict[str, object]:
    normalized_question = repair_question_text(question)
    parsed = parse_question(normalized_question)
    graph_result = execute_parsed_query(engine, parsed)
    return {'question': normalized_question, 'parsed_query': parsed_query_payload(parsed), 'graph_answer': graph_result, 'llm_answer': answer_via_mode(answer_mode=answer_mode, config_path=config_path, question=normalized_question, graph_result=graph_result)}
