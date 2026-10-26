from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import filter_rows_by_data_source, load_open_eqa_rows, repo_path, write_json_atomic
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.config.config import load_domain_config
from vlkg.datasets.openeqa.qa.canonical_graph_qa_context import answer_canonical_graph_query
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession, unified_graph_retrieval_question_ir
from vlkg.core.memory.types import MemoryQuery
DEFAULT_QA_JSON = (REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json')
DEFAULT_GRAPH_MANIFEST = (REPO_ROOT / 'datasets/openeqa/graphs/openeqa-time-fact-core-v1/payload/runtime/hm3d/openeqa_hm3d_graph_build_manifest.json')
DEFAULT_OUTPUT_ROOT = (REPO_ROOT / 'datasets/openeqa/runs/new_runs/payload')
DEFAULT_QUESTION_IR_JSON = Path('datasets/openeqa/ir/openeqa-v5-14class-strategy-v4/payload/runtime_question_ir.json')
DEFAULT_GRAPH_RETRIEVAL_PROFILE = 'navqa_temporal'
CANONICAL_VECTOR_HYBRID_PROFILE = 'canonical_vector_hybrid_v1'
CANONICAL_VECTOR_GRAPH_HYBRID_PROFILE = 'canonical_vector_graph_hybrid_v2'
CANONICAL_VECTOR_STRUCTURED_PROFILE = 'canonical_vector_structured_v3'
CANONICAL_VECTOR_UNION_PROFILE = 'canonical_vector_union_v4'
CANONICAL_VECTOR_EVIDENCE_PROFILE = 'canonical_vector_evidence_v5'
CANONICAL_VECTOR_PROFILES = {CANONICAL_VECTOR_HYBRID_PROFILE, CANONICAL_VECTOR_GRAPH_HYBRID_PROFILE, CANONICAL_VECTOR_STRUCTURED_PROFILE, CANONICAL_VECTOR_UNION_PROFILE, CANONICAL_VECTOR_EVIDENCE_PROFILE}
ROUTE_CONDITIONED_GRAPH_RETRIEVAL_PROFILES = {'openeqa_route_aware', 'openeqa_hm3d_binding_v1', 'openeqa_hm3d_spatial_binding_v2'}
UNIFIED_GRAPH_RETRIEVAL_PROFILES = {'navqa_temporal', 'openeqa_hm3d_unified_v1', 'openeqa_hm3d_unified_v2', CANONICAL_VECTOR_HYBRID_PROFILE, CANONICAL_VECTOR_GRAPH_HYBRID_PROFILE, CANONICAL_VECTOR_STRUCTURED_PROFILE, CANONICAL_VECTOR_UNION_PROFILE, CANONICAL_VECTOR_EVIDENCE_PROFILE}
FIXED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES = {*UNIFIED_GRAPH_RETRIEVAL_PROFILES, *ROUTE_CONDITIONED_GRAPH_RETRIEVAL_PROFILES}
ENTITY_CATEGORIES = {'object recognition', 'attribute recognition', 'spatial understanding', 'object state recognition', 'object localization'}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run OpenEQA HM3D graph-only QA over prebuilt canonical temporal memory graphs. This adapter reads OpenEQA rows, maps each question to its episode graph, and writes official prediction JSON plus audit artifacts.')
    parser.add_argument('--open-eqa-json', type=Path, default=DEFAULT_QA_JSON)
    parser.add_argument('--graph-manifest', type=Path, default=DEFAULT_GRAPH_MANIFEST)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--question-ir-json', type=Path, default=DEFAULT_QUESTION_IR_JSON)
    parser.add_argument('--skip-question-ir', action='store_true', help='Debug only: run without OpenEQA Question IR.')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--data-source', default='hm3d-v0')
    parser.add_argument('--category', action='append', default=None, help='Optional OpenEQA category filter. Can be repeated.')
    parser.add_argument('--episode-history', action='append', default=None, help='Optional episode_history filter. Can be repeated.')
    parser.add_argument('--question-id', action='append', default=None, help='Optional question_id filter. Can be repeated.')
    parser.add_argument('--question-id-json', type=Path, action='append', default=None, help='Optional JSON array(s) whose records provide question_id/uuid selection.')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--retrieval-mode', choices=('auto', 'entity', 'temporal'), default='auto', help='auto resolves through --graph-retrieval-profile. The default OpenEQA profile uses temporal canonical retrieval to match the NAVQA graph-only backbone.')
    parser.add_argument('--graph-retrieval-profile', choices=('legacy_entity_auto', 'navqa_temporal', 'openeqa_route_aware', 'openeqa_hm3d_binding_v1', 'openeqa_hm3d_spatial_binding_v2', 'openeqa_hm3d_unified_v1', 'openeqa_hm3d_unified_v2', CANONICAL_VECTOR_HYBRID_PROFILE, CANONICAL_VECTOR_GRAPH_HYBRID_PROFILE, CANONICAL_VECTOR_STRUCTURED_PROFILE, CANONICAL_VECTOR_UNION_PROFILE, CANONICAL_VECTOR_EVIDENCE_PROFILE), default=DEFAULT_GRAPH_RETRIEVAL_PROFILE, help='Graph-only retrieval profile. legacy_entity_auto preserves the first OpenEQA adapter behavior; navqa_temporal uses the same canonical temporal retrieval mode as NAVQA graph-only and is the unified default without route/category-conditioned scoring; openeqa_route_aware additionally enables IR/route-aware scoring inside canonical graph retrieval without images, FAISS, or gold answers; openeqa_hm3d_binding_v1 keeps that scoring and renders relation evidence with canonical subject/object labels; openeqa_hm3d_spatial_binding_v2 additionally ranks explicit target-anchor relation bindings; openeqa_hm3d_unified_v1 uses one route-independent temporal candidate/scoring path with semantic target, attribute, and relation binding features. openeqa_hm3d_unified_v2 keeps only fully bound spatial relation features so unrelated attribute/entity evidence retains baseline ranking; canonical_vector_hybrid_v1 uniformly unions lexical/exact recall with semantic recall over HSGG-derived canonical graph records, then keeps the same structural graph scoring and expansion path; canonical_vector_graph_hybrid_v2 keeps the same unified path, preserves five lexical graph anchors, and admits up to three vector-seeded graph fragments into a shared evidence context; canonical_vector_structured_v3 keeps the same unified path and adds target-owner and attribute-dimension compatibility to graph reranking; canonical_vector_union_v4 merges lexical, exact, and vector candidates before the shared structural scoring and graph expansion path; canonical_vector_evidence_v5 keeps that unified candidate union and adds four vector-seeded graph evidence fragments near the front of the shared context, without question-category or route branching.')
    parser.add_argument('--vector-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Local sentence-transformers model used only by canonical vector profiles. Network download is disabled.')
    parser.add_argument('--retrieval-limit', type=int, default=10)
    parser.add_argument('--answer-mode', choices=('graph_only', 'mock', 'live'), default='live', help='live calls the configured LLM but still uses graph-only evidence; graph_only writes the deterministic graph evidence brief.')
    parser.add_argument('--config', default='datasets/openeqa/qa.yaml', help='OpenEQA-only QA configuration. This default is isolated from the NAVQA runners, which continue to use their own config paths.')
    parser.add_argument('--qa-workers', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--qa-temperature', type=float, default=None, help='Optional OpenEQA QA sampling temperature override.')
    parser.add_argument('--qa-api-key-index', type=int, default=None, help='Pin OpenEQA QA to one zero-based API key-pool slot.')
    parser.add_argument('--checkpoint-every', type=int, default=1, help='Persist the full audit/prediction checkpoint after this many completed questions.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run', action='store_true')
    return parser

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def load_question_id_jsons(paths: list[Path] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        path = repo_path(raw_path)
        payload = load_json(path)
        records = payload if isinstance(payload, list) else payload.get('records', []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            raise ValueError(f'question ID selection must contain a record list: {path}')
        for record in records:
            if not isinstance(record, dict):
                continue
            question_id = str(record.get('question_id') or record.get('uuid') or '').strip()
            if not question_id:
                continue
            if question_id in seen:
                raise ValueError(f'duplicate selected question_id: {question_id}')
            seen.add(question_id)
            result.append(question_id)
    if paths and (not result):
        raise ValueError('question ID selection files contain no IDs')
    return result

def pin_qa_api_key(config_path: str, index: int | None) -> int:
    if index is None:
        return 0
    if index < 0:
        raise ValueError('qa-api-key-index must be non-negative')
    config = load_domain_config(config_path).graph_qa
    env_var = str(config.api_key_env_var)
    pool_var = env_var if env_var.endswith('_KEYS') else f'{env_var}S'
    raw_pool = os.getenv(pool_var, '')
    try:
        parsed = json.loads(raw_pool)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        pool = [str(value).strip() for value in parsed if str(value).strip()]
    else:
        pool = [value.strip() for value in re.split('[,;\\r\\n]+', raw_pool) if value.strip()]
    if not pool:
        single_key = os.getenv(env_var, '').strip()
        pool = [single_key] if single_key else []
    if index >= len(pool):
        raise ValueError(f'qa-api-key-index {index} is outside configured pool size {len(pool)}')
    os.environ[env_var] = pool[index]
    os.environ.pop(pool_var, None)
    return len(pool)

def summarize_usage(response_metadata: dict[str, Any] | None) -> dict[str, int]:
    usage = (response_metadata or {}).get('usage') or {}
    return {'prompt_tokens': int(usage.get('prompt_tokens') or 0), 'completion_tokens': int(usage.get('completion_tokens') or 0), 'total_tokens': int(usage.get('total_tokens') or 0)}

def graph_by_episode(manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in manifest.get('episodes', []):
        if not isinstance(item, dict):
            continue
        episode_history = str(item.get('episode_history') or '').strip()
        canonical = item.get('canonical_graph_json')
        if episode_history and canonical:
            result[episode_history] = repo_path(Path(str(canonical)))
    return result

def load_question_ir_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f'Missing question IR JSON: {resolved}')
    payload = load_json(resolved)
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise ValueError(f'question IR records must be a list: {resolved}')
    by_question_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = str(record.get('question_id') or record.get('uuid') or '').strip()
        ir = _normalize_question_ir_record(record)
        if question_id and isinstance(ir, dict):
            normalized_ir = dict(ir)
            record_question_type = str(record.get('question_type') or '').strip()
            if record_question_type and (not str(normalized_ir.get('question_type') or '').strip()):
                normalized_ir['question_type'] = record_question_type
            route_id = _question_route_id_from_ir(normalized_ir)
            if route_id:
                normalized_ir.setdefault('question_route_id', route_id)
                normalized_ir.setdefault('routing', {'route_id': route_id})
            elif isinstance(normalized_ir.get('question_route'), dict):
                route = normalized_ir.get('question_route') or {}
                route_id = str(route.get('route_id') or '').strip()
                if route_id:
                    normalized_ir['question_route_id'] = route_id
                normalized_ir.setdefault('routing', {'route_id': route_id})
            if not str(normalized_ir.get('answer_type') or '').strip():
                answer_type_hint = str(record.get('answer_type_hint') or '').strip()
                if answer_type_hint:
                    normalized_ir['answer_type'] = answer_type_hint
            by_question_id[question_id] = normalized_ir
    return by_question_id

def _normalize_question_ir_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    raw_ir = record.get('question_ir') or record.get('llm_question_ir')
    normalized_ir = dict(raw_ir) if isinstance(raw_ir, dict) else {}
    if not normalized_ir:
        for key in ('question_type', 'question_route_id', 'question_route', 'routing', 'answer_type_hint', 'answer_type', 'evidence_question_type', 'target_entities', 'target_description', 'target_state'):
            value = record.get(key)
            if value in (None, '', [], {}):
                continue
            normalized_ir[key] = value
    record_question_type = str(record.get('question_type') or '').strip()
    if record_question_type and (not str(normalized_ir.get('question_type') or '').strip()):
        normalized_ir['question_type'] = record_question_type
    record_route_id = str(record.get('question_route_id') or '').strip()
    if record_route_id and (not str(normalized_ir.get('question_route_id') or '').strip()):
        normalized_ir['question_route_id'] = record_route_id
    record_route = record.get('question_route')
    if isinstance(record_route, dict) and record_route and (not isinstance(normalized_ir.get('question_route'), dict)):
        normalized_ir['question_route'] = record_route
    record_routing = record.get('routing')
    if isinstance(record_routing, dict) and record_routing and (not isinstance(normalized_ir.get('routing'), dict)):
        normalized_ir['routing'] = record_routing
    return normalized_ir

def _question_type_from_ir(ir: dict[str, Any] | None) -> str | None:
    if not isinstance(ir, dict):
        return None
    for key in ('question_type', 'answer_type', 'evidence_question_type'):
        value = str(ir.get(key) or '').strip()
        if value:
            return value
    return None

def _question_route_id_from_ir(ir: dict[str, Any] | None) -> str | None:
    if not isinstance(ir, dict):
        return None
    value = str(ir.get('question_route_id') or '').strip()
    if value:
        return value
    route = ir.get('question_route')
    if isinstance(route, dict):
        value = str(route.get('route_id') or '').strip()
        if value:
            return value
    routing = ir.get('routing')
    if isinstance(routing, dict):
        value = str(routing.get('route_id') or '').strip()
        if value:
            return value
    return None

def _question_route_from_ir(ir: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(ir, dict):
        return {}
    route = ir.get('question_route')
    if isinstance(route, dict) and route:
        return route
    route_id = _question_route_id_from_ir(ir)
    if route_id:
        return {'route_id': route_id}
    routing = ir.get('routing')
    if isinstance(routing, dict) and routing:
        return routing
    return {}

def _question_type_from_audit_item(item: dict[str, Any]) -> str:
    value = str(item.get('question_type') or '').strip()
    if value:
        return value
    return _question_type_from_ir(item.get('question_ir')) or 'unknown'

def _question_route_id_from_audit_item(item: dict[str, Any]) -> str:
    value = str(item.get('question_route_id') or '').strip()
    if value:
        return value
    return _question_route_id_from_ir(item.get('question_ir')) or 'unknown'

def string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ('kind', 'text', 'role', 'relation', 'operation', 'target_state', 'target', 'target_entities', 'must_bind_to'):
            parts.extend(string_values(value.get(key)))
        return parts
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(string_values(item))
        return parts
    return [str(value)] if str(value).strip() else []

def query_text_from_question_ir(question: str, ir: dict[str, Any] | None) -> str:
    if not isinstance(ir, dict):
        return question
    parts = [question]
    for key in ('target_state', 'target_entities', 'target_description', 'target_descriptors', 'attribute_dimension', 'event_constraints', 'spatial_constraints', 'hard_constraints', 'evidence_requirements', 'graph_information_needs', 'normalized_query_terms'):
        parts.extend(string_values(ir.get(key)))
    return ' ; '.join(dict.fromkeys((part.strip() for part in parts if part.strip())))

def select_rows(rows: list[dict[str, Any]], *, data_source: str, categories: list[str] | None, episode_histories: list[str] | None, question_ids: list[str] | None, offset: int, limit: int | None) -> list[dict[str, Any]]:
    selected = filter_rows_by_data_source(rows, data_source=data_source)
    if categories:
        wanted = {item.strip().lower() for item in categories}
        selected = [row for row in selected if str(row.get('category') or '').strip().lower() in wanted]
    if episode_histories:
        wanted = {item.strip() for item in episode_histories}
        selected = [row for row in selected if str(row.get('episode_history') or '').strip() in wanted]
    if question_ids:
        wanted = {item.strip() for item in question_ids}
        selected = [row for row in selected if str(row.get('question_id') or '').strip() in wanted]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected

def resolve_retrieval_mode(category: str, requested: str, *, question_ir: dict[str, Any] | None=None, graph_retrieval_profile: str='legacy_entity_auto') -> str:
    if requested != 'auto':
        return requested
    if graph_retrieval_profile in FIXED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES:
        return 'temporal'
    qtype = _question_type_from_ir(question_ir)
    if qtype:
        return 'temporal' if qtype.startswith('temporal.') else 'entity'
    return 'entity' if category.strip().lower() in ENTITY_CATEGORIES else 'temporal'

def graph_retrieval_contract(graph_retrieval_profile: str) -> dict[str, bool]:
    unified = graph_retrieval_profile in UNIFIED_GRAPH_RETRIEVAL_PROFILES
    return {'unified_candidate_and_scoring_path': unified, 'fixed_temporal_retrieval_mode': graph_retrieval_profile in FIXED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES, 'uses_category_conditioned_scoring': False, 'uses_question_route_conditioned_scoring': graph_retrieval_profile in ROUTE_CONDITIONED_GRAPH_RETRIEVAL_PROFILES, 'uses_evidence_type_conditioned_scoring': False, 'uses_canonical_record_vectors': graph_retrieval_profile in CANONICAL_VECTOR_PROFILES}

def graph_qa_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {'max_tokens': args.max_tokens}
    if args.qa_temperature is not None:
        overrides['temperature'] = float(args.qa_temperature)
    if args.graph_retrieval_profile == CANONICAL_VECTOR_GRAPH_HYBRID_PROFILE:
        overrides['max_evidence_lines'] = 8
    elif args.graph_retrieval_profile == CANONICAL_VECTOR_EVIDENCE_PROFILE:
        overrides['max_evidence_lines'] = 10
    return overrides

def vector_cache_path_for_graph(graph_path: Path) -> Path:
    return graph_path.with_name('canonical_record_vectors_minilm_v1.npz')

def load_graph(path: Path) -> CanonicalTemporalMemoryGraph:
    return CanonicalTemporalMemoryGraph.model_validate_json(path.read_text(encoding='utf-8-sig'))

def output_paths(output_dir: Path) -> dict[str, Path]:
    return {'run_config': output_dir / 'run_config.json', 'predictions': output_dir / 'predictions.json', 'audit': output_dir / 'audit.json', 'summary_json': output_dir / 'summary.json', 'summary_md': output_dir / 'summary.md'}

def load_existing_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    return payload if isinstance(payload, list) else []

def safe_prediction_text(payload: dict[str, Any]) -> str:
    llm_answer = payload.get('llm_answer') or {}
    answer = str(llm_answer.get('answer_text') or '').strip()
    if answer:
        return answer
    graph_answer = payload.get('graph_answer') or {}
    return str(graph_answer.get('answer_brief') or '').strip()

def audit_record_from_result(*, row: dict[str, Any], graph_path: Path, retrieval_mode: str, graph_retrieval_profile: str, question_ir: dict[str, Any] | None, result: dict[str, Any], elapsed_sec: float) -> dict[str, Any]:
    llm_answer = result.get('llm_answer') or {}
    graph_answer = result.get('graph_answer') or {}
    llm_structured = llm_answer.get('structured_answer') or {}
    metadata = llm_answer.get('graph_rag_metadata') or {}
    response_metadata = llm_answer.get('response_metadata') or {}
    graph_matches = [item for item in graph_answer.get('matches', []) or [] if isinstance(item, dict)]
    return {'question_id': str(row.get('question_id') or ''), 'episode_history': str(row.get('episode_history') or ''), 'category': str(row.get('category') or ''), 'question': str(row.get('question') or ''), 'prediction': safe_prediction_text(result), 'retrieval_mode': retrieval_mode, 'graph_retrieval_profile': graph_retrieval_profile, 'question_ir': question_ir or {}, 'question_type': _question_type_from_ir(question_ir), 'question_route_id': _question_route_id_from_ir(question_ir), 'question_route': _question_route_from_ir(question_ir), 'graph_path': str(graph_path), 'graph_only_contract': {'uses_hsgg_derived_graph': True, 'uses_images_or_crops': False, 'uses_visual_faiss': False, 'uses_gold_answer_for_prediction': False}, 'graph_rag_status': llm_answer.get('graph_rag_status'), 'provider': llm_answer.get('provider'), 'model': llm_answer.get('model'), 'match_count': metadata.get('match_count'), 'graph_retrieval_source_record_ids': [str(item.get('source_record_id') or '') for item in graph_matches], 'graph_retrieval_source_kinds': [str(item.get('source_kind') or '') for item in graph_matches], 'evidence_line_count': metadata.get('evidence_line_count'), 'used_focus_frames': llm_structured.get('used_focus_frames', []), 'used_evidence_lines': llm_structured.get('used_evidence_lines', []), 'answer_brief': graph_answer.get('answer_brief'), 'usage': summarize_usage(response_metadata), 'elapsed_sec': round(elapsed_sec, 3), 'error': llm_answer.get('error')}

def error_audit_record(*, row: dict[str, Any], graph_path: Path | None, retrieval_mode: str | None, error: str, graph_retrieval_profile: str='', question_ir: dict[str, Any] | None=None) -> dict[str, Any]:
    return {'question_id': str(row.get('question_id') or ''), 'episode_history': str(row.get('episode_history') or ''), 'category': str(row.get('category') or ''), 'question': str(row.get('question') or ''), 'prediction': '', 'retrieval_mode': retrieval_mode, 'graph_retrieval_profile': graph_retrieval_profile, 'question_ir': question_ir or {}, 'question_type': _question_type_from_ir(question_ir), 'question_route_id': _question_route_id_from_ir(question_ir), 'question_route': _question_route_from_ir(question_ir), 'graph_path': str(graph_path) if graph_path is not None else None, 'graph_only_contract': {'uses_hsgg_derived_graph': True, 'uses_images_or_crops': False, 'uses_visual_faiss': False, 'uses_gold_answer_for_prediction': False}, 'graph_rag_status': 'error', 'provider': None, 'model': None, 'match_count': 0, 'evidence_line_count': 0, 'used_focus_frames': [], 'used_evidence_lines': [], 'answer_brief': '', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'elapsed_sec': 0.0, 'error': error}

def predictions_from_audit(audit: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{'question_id': str(item.get('question_id') or ''), 'answer': str(item.get('prediction') or '')} for item in audit if item.get('question_id')]

def build_summary(*, run_name: str, output_dir: Path, selected_rows: list[dict[str, Any]], audit: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter((str(item.get('graph_rag_status') or 'unknown') for item in audit))
    categories_total = Counter((str(row.get('category') or 'unknown') for row in selected_rows))
    categories_done = Counter((str(item.get('category') or 'unknown') for item in audit))
    retrieval_modes = Counter((str(item.get('retrieval_mode') or 'unknown') for item in audit))
    question_types = Counter((_question_type_from_audit_item(item) for item in audit))
    question_routes = Counter((_question_route_id_from_audit_item(item) for item in audit))
    usage = Counter()
    for item in audit:
        usage.update(item.get('usage') or {})
    errors = [item for item in audit if item.get('error')]
    return {'contract': 'openeqa_hm3d_graph_only_qa_summary_v1', 'run_name': run_name, 'output_dir': str(output_dir), 'generated_at': datetime.now(timezone.utc).isoformat(), 'open_eqa_json': str(repo_path(args.open_eqa_json)), 'graph_manifest': str(repo_path(args.graph_manifest)), 'graph_manifest_run_id': manifest.get('run_id'), 'graph_config': manifest.get('config'), 'graph_profile': manifest.get('graph_profile'), 'time_axis': manifest.get('time_axis'), 'timestamp_is_real_time': manifest.get('timestamp_is_real_time'), 'answer_mode': args.answer_mode, 'retrieval_mode': args.retrieval_mode, 'graph_retrieval_profile': args.graph_retrieval_profile, 'graph_retrieval_contract': graph_retrieval_contract(args.graph_retrieval_profile), 'retrieval_limit': args.retrieval_limit, 'selected_question_count': len(selected_rows), 'completed_question_count': len(audit), 'prediction_count': len(predictions_from_audit(audit)), 'category_counts_total': dict(sorted(categories_total.items())), 'category_counts_completed': dict(sorted(categories_done.items())), 'graph_rag_status_counts': dict(sorted(statuses.items())), 'retrieval_mode_counts': dict(sorted(retrieval_modes.items())), 'question_type_counts': dict(sorted(question_types.items())), 'question_route_id_counts': dict(sorted(question_routes.items())), 'token_usage': {'prompt_tokens': int(usage.get('prompt_tokens', 0)), 'completion_tokens': int(usage.get('completion_tokens', 0)), 'total_tokens': int(usage.get('total_tokens', 0))}, 'error_count': len(errors), 'error_question_ids': [str(item.get('question_id') or '') for item in errors], 'graph_only_contract': {'uses_hsgg_derived_graph': True, 'uses_images_or_crops': False, 'uses_visual_faiss': False, 'uses_gold_answer_for_prediction': False}}

def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"# OpenEQA HM3D Graph-Only QA: {summary['run_name']}", '', f"- selected questions: `{summary['selected_question_count']}`", f"- completed questions: `{summary['completed_question_count']}`", f"- prediction count: `{summary['prediction_count']}`", f"- answer mode: `{summary['answer_mode']}`", f"- retrieval mode: `{summary['retrieval_mode']}`", f"- graph retrieval profile: `{summary.get('graph_retrieval_profile')}`", f"- graph profile: `{summary.get('graph_profile')}`", f"- time axis: `{summary.get('time_axis')}`", f"- timestamp is real time: `{summary.get('timestamp_is_real_time')}`", f"- error count: `{summary['error_count']}`", '', '## Category Counts', '']
    for category, count in summary['category_counts_completed'].items():
        lines.append(f'- `{category}`: `{count}`')
    lines.extend(['', '## Status Counts', ''])
    for status, count in summary['graph_rag_status_counts'].items():
        lines.append(f'- `{status}`: `{count}`')
    lines.extend(['', '## Question Type Counts', ''])
    for question_type, count in summary['question_type_counts'].items():
        lines.append(f'- `{question_type}`: `{count}`')
    lines.extend(['', '## Question Route Counts', ''])
    for route_id, count in summary['question_route_id_counts'].items():
        lines.append(f'- `{route_id}`: `{count}`')
    lines.extend(['', '## Token Usage', '', f"- prompt tokens: `{summary['token_usage']['prompt_tokens']}`", f"- completion tokens: `{summary['token_usage']['completion_tokens']}`", f"- total tokens: `{summary['token_usage']['total_tokens']}`"])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def persist_outputs(*, paths: dict[str, Path], audit: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_json_atomic(paths['audit'], audit)
    write_json_atomic(paths['predictions'], predictions_from_audit(audit))
    write_json_atomic(paths['summary_json'], summary)
    write_summary_md(paths['summary_md'], summary)

def preflight(*, args: argparse.Namespace, manifest: dict[str, Any], selected_rows: list[dict[str, Any]], graph_index: dict[str, Path], question_ir_records: dict[str, dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if args.qa_workers < 1:
        issues.append('qa-workers must be >= 1.')
    if args.checkpoint_every < 1:
        issues.append('checkpoint-every must be >= 1.')
    missing_graphs: list[str] = []
    missing_files: list[str] = []
    missing_ir: list[str] = []
    for row in selected_rows:
        episode_history = str(row.get('episode_history') or '')
        question_id = str(row.get('question_id') or '')
        graph_path = graph_index.get(episode_history)
        if graph_path is None:
            missing_graphs.append(episode_history)
        elif not graph_path.exists():
            missing_files.append(str(graph_path))
        if not args.skip_question_ir and question_id not in question_ir_records:
            missing_ir.append(question_id)
    if missing_graphs:
        issues.append(f'Missing graph manifest entries: {sorted(set(missing_graphs))[:8]}')
    if missing_files:
        issues.append(f'Missing canonical graph files: {missing_files[:8]}')
    if missing_ir:
        issues.append(f'incomplete question IR records: {len(selected_rows) - len(missing_ir)}/{len(selected_rows)} (missing sample={missing_ir[:8]})')
    if manifest.get('timestamp_is_real_time') is not False:
        issues.append('Graph manifest timestamp_is_real_time is not false; inspect time semantics.')
    return issues

def main() -> None:
    args = build_parser().parse_args()
    pin_qa_api_key(args.config, args.qa_api_key_index)
    run_name = args.run_name or datetime.now().strftime('hm3d_graph_only_%Y%m%d_%H%M%S')
    output_dir = repo_path(args.output_root) / run_name
    paths = output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    open_eqa_json = repo_path(args.open_eqa_json)
    graph_manifest_path = repo_path(args.graph_manifest)
    if not open_eqa_json.exists():
        raise SystemExit(f'Missing OpenEQA QA file: {open_eqa_json}')
    if not graph_manifest_path.exists():
        raise SystemExit(f'Missing graph manifest: {graph_manifest_path}')
    rows = load_open_eqa_rows(open_eqa_json)
    manifest = load_json(graph_manifest_path)
    graph_index = graph_by_episode(manifest)
    question_ir_records = {} if args.skip_question_ir else load_question_ir_records(args.question_ir_json)
    selected_question_ids = [*(args.question_id or []), *load_question_id_jsons(args.question_id_json)]
    selected_rows = select_rows(rows, data_source=args.data_source, categories=args.category, episode_histories=args.episode_history, question_ids=selected_question_ids, offset=args.offset, limit=args.limit)
    run_config = {'contract': 'openeqa_hm3d_graph_only_qa_run_config_v1', 'run_name': run_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'open_eqa_json': str(open_eqa_json), 'graph_manifest': str(graph_manifest_path), 'qa_api_key_index': args.qa_api_key_index, 'question_id_jsons': [str(repo_path(path)) for path in args.question_id_json or []], 'question_ir_json': '' if args.skip_question_ir else str(repo_path(args.question_ir_json)), 'skip_question_ir': bool(args.skip_question_ir), 'output_dir': str(output_dir), 'data_source': args.data_source, 'answer_mode': args.answer_mode, 'retrieval_mode': args.retrieval_mode, 'graph_retrieval_profile': args.graph_retrieval_profile, 'graph_retrieval_contract': graph_retrieval_contract(args.graph_retrieval_profile), 'retrieval_limit': args.retrieval_limit, 'qa_workers': args.qa_workers, 'max_tokens': args.max_tokens, 'vector_model': args.vector_model if args.graph_retrieval_profile in CANONICAL_VECTOR_PROFILES else None, 'selected_question_count': len(selected_rows)}
    write_json_atomic(paths['run_config'], run_config)
    issues = preflight(args=args, manifest=manifest, selected_rows=selected_rows, graph_index=graph_index, question_ir_records=question_ir_records)
    print(f'run_name={run_name}')
    print(f'output_dir={output_dir}')
    print(f'selected_questions={len(selected_rows)}')
    print(f"graph_manifest_run_id={manifest.get('run_id')}")
    print(f"time_axis={manifest.get('time_axis')}")
    print(f"timestamp_is_real_time={manifest.get('timestamp_is_real_time')}")
    print(f'graph_retrieval_profile={args.graph_retrieval_profile}')
    print('question_ir=' + ('disabled' if args.skip_question_ir else f'{repo_path(args.question_ir_json)} matched={len(question_ir_records)}'))
    if issues:
        print('preflight_status=failed')
        for issue in issues:
            print(f'- {issue}')
        raise SystemExit(2)
    print('preflight_status=passed')
    if not args.run:
        print('Use --run to start QA.')
        return
    existing_audit = load_existing_audit(paths['audit']) if args.resume else []
    if args.force:
        existing_audit = []
    elif args.resume:
        existing_audit = [item for item in existing_audit if not item.get('error')]
    completed_ids = {str(item.get('question_id') or '') for item in existing_audit if item.get('question_id') and (not item.get('error'))}
    audit_by_id = {str(item.get('question_id') or ''): item for item in existing_audit if item.get('question_id')}
    pending_rows = [(idx, row) for idx, row in enumerate(selected_rows, start=1) if str(row.get('question_id') or '') not in completed_ids]

    def run_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        _idx, row = item
        question_id = str(row.get('question_id') or '')
        episode_history = str(row.get('episode_history') or '')
        category = str(row.get('category') or '')
        question = str(row.get('question') or '')
        question_ir = None if args.skip_question_ir else question_ir_records.get(question_id)
        retrieval_question_ir = unified_graph_retrieval_question_ir(question_ir, graph_retrieval_profile=args.graph_retrieval_profile)
        retrieval_mode = resolve_retrieval_mode(category, args.retrieval_mode, question_ir=question_ir, graph_retrieval_profile=args.graph_retrieval_profile)
        graph_path = graph_index.get(episode_history)
        if graph_path is None:
            return error_audit_record(row=row, graph_path=None, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, error=f'Missing graph for episode_history={episode_history}')
        started = datetime.now()
        try:
            graph = load_graph(graph_path)
            session_kwargs: dict[str, Any] = {}
            if args.graph_retrieval_profile in CANONICAL_VECTOR_PROFILES:
                session_kwargs = {'vector_recall_cache_path': vector_cache_path_for_graph(graph_path), 'vector_recall_model_name': args.vector_model}
            retrieval_session = CanonicalHybridRetrievalSession(graph, **session_kwargs)
            result = answer_canonical_graph_query(graph, MemoryQuery(query_text=query_text_from_question_ir(question, retrieval_question_ir), metadata={'dataset': 'OpenEQA', 'data_source': args.data_source, 'question_id': question_id, 'episode_history': episode_history, **({'category': category} if args.graph_retrieval_profile not in UNIFIED_GRAPH_RETRIEVAL_PROFILES else {}), 'question_ir': retrieval_question_ir or {}, 'question_route': _question_route_from_ir(retrieval_question_ir), 'graph_retrieval_profile': args.graph_retrieval_profile, **({'raw_question': question} if args.graph_retrieval_profile in CANONICAL_VECTOR_PROFILES else {})}, limit=args.retrieval_limit), question=question, retrieval_mode=retrieval_mode, answer_mode=args.answer_mode, config_path=args.config, limit=args.retrieval_limit, retrieval_session=retrieval_session, graph_qa_config_overrides=graph_qa_overrides(args))
            return audit_record_from_result(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, result=result, elapsed_sec=(datetime.now() - started).total_seconds())
        except Exception as exc:
            return error_audit_record(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
    completed_since_start = 0
    with ThreadPoolExecutor(max_workers=args.qa_workers) as executor:
        futures = {executor.submit(run_one, item): item for item in pending_rows}
        for future in as_completed(futures):
            idx, row = futures[future]
            record = future.result()
            question_id = str(record.get('question_id') or row.get('question_id') or '')
            audit_by_id[question_id] = record
            completed_since_start += 1
            audit = [audit_by_id[str(selected.get('question_id') or '')] for selected in selected_rows if str(selected.get('question_id') or '') in audit_by_id]
            print(f"[{idx}/{len(selected_rows)}] completed {question_id} status={record.get('graph_rag_status') or 'error'}", flush=True)
            summary = build_summary(run_name=run_name, output_dir=output_dir, selected_rows=selected_rows, audit=audit, args=args, manifest=manifest)
            if completed_since_start % args.checkpoint_every == 0:
                persist_outputs(paths=paths, audit=audit, summary=summary)
    audit = [audit_by_id[str(selected.get('question_id') or '')] for selected in selected_rows if str(selected.get('question_id') or '') in audit_by_id]
    summary = build_summary(run_name=run_name, output_dir=output_dir, selected_rows=selected_rows, audit=audit, args=args, manifest=manifest)
    persist_outputs(paths=paths, audit=audit, summary=summary)
    print(f"predictions={paths['predictions']}")
    print(f"audit={paths['audit']}")
    print(f"summary={paths['summary_json']}")
    print(f"summary_md={paths['summary_md']}")
if __name__ == '__main__':
    main()
