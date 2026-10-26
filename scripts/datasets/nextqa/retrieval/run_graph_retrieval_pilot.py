"""Run an isolated NExT-QA graph-only retrieval scope offline."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.retrieval.graph_only_retrieval import UnifiedGraphOnlyRetriever, load_unified_retrieval_config
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, require_within_nextqa_scratch
from vlkg.datasets.nextqa.retrieval.question_ir import retrieval_input_from_ir
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot, utc_now
from vlkg.datasets.nextqa.runtime.special_patch import SPECIAL_PATCH_QUESTION_COUNT, SPECIAL_PATCH_VIDEO_COUNT
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
PROJECT_PYTHON = Path(sys.executable).resolve()
DEFAULT_GRAPH_RUN = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'graph_pilots' / 'nextqa_cg6_human_v2_pilot5_20260811T145718Z'
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'datasets/nextqa/retrieval.yaml'
STRICT_RETRIEVAL_KEYS = {'question_id', 'video_id', 'normalized_question', 'target_entities', 'actors', 'actions', 'objects', 'temporal_direction', 'temporal_anchor', 'referenced_event', 'causal_query', 'count_query', 'location_query', 'attribute_query', 'answer_format'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-ir', type=Path, required=True)
    parser.add_argument('--graph-run-root', type=Path, default=DEFAULT_GRAPH_RUN)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--seed-vector-cache-root', type=Path, default=None, help='Read-only prior canonical vector cache cloned into the new isolated run.')
    parser.add_argument('--scope', choices=('pilot', 'full', 'special'), default='pilot')
    parser.add_argument('--checkpoint-every', type=int, default=None, help='questions between atomic checkpoints; defaults to 1 for pilot and 25 otherwise')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def _load_strict_ir(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError('retrieval IR must be a JSON list of objects')
    for row in payload:
        if set(row) != STRICT_RETRIEVAL_KEYS:
            extra = sorted(set(row) - STRICT_RETRIEVAL_KEYS)
            missing = sorted(STRICT_RETRIEVAL_KEYS - set(row))
            raise ValueError(f'retrieval IR violates strict key contract; extra={extra}, missing={missing}')
    return payload

def _load_graph_paths(graph_run_root: Path, *, expected_video_count: int) -> dict[str, Path]:
    manifest_path = graph_run_root / 'nextqa_graph_build_manifest.json'
    manifest = _load_json(manifest_path)
    if manifest.get('video_count') != expected_video_count or manifest.get('completed_video_count') != expected_video_count:
        raise ValueError(f'graph retrieval requires a completed {expected_video_count}-video graph run')
    if manifest.get('error_video_count') != 0 or manifest.get('uses_gold_answer') is not False:
        raise ValueError('graph build manifest purity/completion gate failed')
    paths: dict[str, Path] = {}
    for row in manifest.get('videos', []):
        path = (REPO_ROOT / Path(str(row['canonical_graph_path']))).resolve()
        if row.get('status') != 'completed' or not path.is_file():
            raise ValueError(f'canonical graph is unavailable: {path}')
        paths[str(row['official_video_id'])] = path
    if len(paths) != expected_video_count:
        raise ValueError('canonical graph manifest contains duplicate or missing video IDs')
    return paths

def _load_existing_records(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / 'retrieval_records.json'
    if not path.is_file():
        return []
    payload = _load_json(path)
    return payload if isinstance(payload, list) else []

def _checkpoint(run_root: Path, records: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    combined = sorted([*records, *errors], key=lambda row: str(row.get('question_id') or ''))
    atomic_write_json(run_root / 'retrieval_records.json', combined)
    atomic_write_json(run_root / 'run_status.json', {'status': 'running', 'updated_at_utc': utc_now(), 'completed_question_count': len(records), 'error_question_count': len(errors), 'uses_gold_answer': False})

def _seed_vector_cache(source_root: Path, target_root: Path) -> dict[str, Any]:
    source = require_within_nextqa_scratch(source_root.resolve())
    target = require_within_nextqa_scratch(target_root.resolve())
    if not source.is_dir():
        raise FileNotFoundError(f'seed vector cache root is unavailable: {source}')
    source_files = sorted((path for path in source.iterdir() if path.is_file()))
    npz_count = sum((path.suffix == '.npz' for path in source_files))
    manifest_count = sum((path.name.endswith('.npz.manifest.json') for path in source_files))
    if not source_files or npz_count != manifest_count:
        raise ValueError('seed vector cache must contain paired npz and manifest files')
    target.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    for source_path in source_files:
        target_path = target / source_path.name
        if target_path.exists():
            continue
        try:
            os.link(source_path, target_path)
            linked += 1
        except OSError:
            shutil.copy2(source_path, target_path)
            copied += 1
    return {'schema_version': 'nextqa_vector_cache_seed_audit_v1', 'source_root': str(source), 'target_root': str(target), 'source_file_count': len(source_files), 'source_npz_count': npz_count, 'source_manifest_count': manifest_count, 'source_bytes': sum((path.stat().st_size for path in source_files)), 'hardlinked_file_count': linked, 'copied_file_count': copied, 'old_run_write_allowed': False, 'uses_gold_answer': False}

def _status_name(*, structurally_valid: bool, error_count: int) -> str:
    if not structurally_valid:
        return 'invalid'
    return 'completed_with_errors' if error_count else 'completed'

def main() -> int:
    args = parse_args()
    if Path(sys.executable).resolve() != PROJECT_PYTHON:
        raise RuntimeError(f'NExT-QA jobs must use the project virtual environment: {PROJECT_PYTHON}')
    question_ir_path = args.question_ir.resolve()
    graph_run_root = args.graph_run_root.resolve()
    config_path = args.config.resolve()
    scope_counts = {'pilot': (39, 5), 'full': (4996, 570), 'special': (SPECIAL_PATCH_QUESTION_COUNT, SPECIAL_PATCH_VIDEO_COUNT)}
    expected_question_count, expected_video_count = scope_counts[args.scope]
    checkpoint_every = args.checkpoint_every or (1 if args.scope == 'pilot' else 25)
    if checkpoint_every < 1:
        raise ValueError('checkpoint interval must be positive')
    graph_manifest_path = graph_run_root / 'nextqa_graph_build_manifest.json'
    seed_vector_cache_root = args.seed_vector_cache_root.resolve() if args.seed_vector_cache_root else None
    config = load_unified_retrieval_config(config_path)
    ir_rows = _load_strict_ir(question_ir_path)
    graph_paths = _load_graph_paths(graph_run_root, expected_video_count=expected_video_count)
    question_ids = [str(row['question_id']) for row in ir_rows]
    if len(ir_rows) != expected_question_count or len(question_ids) != len(set(question_ids)):
        raise ValueError(f'retrieval {args.scope} run requires {expected_question_count} unique question IDs')
    if set((str(row['video_id']) for row in ir_rows)) != set(graph_paths):
        raise ValueError('Question IR and graph manifest video IDs do not align')
    run_root = prepare_stage_run_root('graph_retrieval_pilots' if args.scope == 'pilot' else 'graph_retrieval', args.run_id, resume=args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': {'pilot': 'fixed_five_video_graph_only_retrieval', 'full': 'full_validation_graph_only_retrieval', 'special': 'special_patch_29_video_graph_only_retrieval'}[args.scope], 'scope': args.scope, 'run_id': args.run_id, 'graph_run_manifest': fingerprint_file(graph_manifest_path), 'question_ir': fingerprint_file(question_ir_path), 'config': fingerprint_file(config_path), 'profile_parameters': config.__dict__, 'question_count': len(ir_rows), 'video_count': len(graph_paths), 'retrieval_input': 'question_only', 'uses_gold_answer': False, 'reads_official_type': False, 'reads_answer_choices': False, 'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_embedding_cache': False, 'uses_faiss': False, 'calls_api': False, 'checkpoint_every_questions': checkpoint_every, 'seed_vector_cache_root': str(seed_vector_cache_root) if seed_vector_cache_root else None, 'shared_hybrid_retrieve_called': False, 'output_root': str(run_root), 'resume_policy': 'keep completed questions and retry only error records'}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_graph_retrieval_run_config_v1')
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    source_snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=source_paths, resume=args.resume)
    vector_cache_root = run_root / 'cache' / 'canonical_record_vectors'
    if seed_vector_cache_root is not None and (not args.resume):
        cache_seed_audit = _seed_vector_cache(seed_vector_cache_root, vector_cache_root)
        atomic_write_json(run_root / 'vector_cache_seed_audit.json', cache_seed_audit)
    existing = _load_existing_records(run_root) if args.resume else []
    completed_by_id = {str(row['question_id']): row for row in existing if row.get('status') == 'completed'}
    records = list(completed_by_id.values())
    errors: list[dict[str, Any]] = []
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in ir_rows:
        if str(row['question_id']) in completed_by_id:
            continue
        rows_by_video.setdefault(str(row['video_id']), []).append(row)
    processed_count = 0
    for video_id in sorted(rows_by_video):
        graph = CanonicalTemporalMemoryGraph.model_validate_json(graph_paths[video_id].read_text(encoding='utf-8'))
        vector_cache_path = vector_cache_root / f'{video_id}.npz'
        try:
            retriever = UnifiedGraphOnlyRetriever(graph, config=config, vector_cache_path=vector_cache_path)
        except Exception as exc:
            errors.extend(({'schema_version': 'nextqa_graph_retrieval_record_v1', 'question_id': str(row['question_id']), 'video_id': video_id, 'status': 'error', 'error_stage': 'initialize_video_retriever', 'error_type': type(exc).__name__, 'error_message': str(exc), 'uses_gold_answer': False} for row in rows_by_video[video_id]))
            processed_count += len(rows_by_video[video_id])
            _checkpoint(run_root, records, errors)
            continue
        for row in rows_by_video[video_id]:
            try:
                result = retriever.retrieve(retrieval_input_from_ir(row))
                if result['status'] != 'completed':
                    raise RuntimeError('graph retrieval returned no evidence')
                records.append(result)
            except Exception as exc:
                errors.append({'schema_version': 'nextqa_graph_retrieval_record_v1', 'question_id': str(row['question_id']), 'video_id': video_id, 'status': 'error', 'error_stage': 'retrieve_question', 'error_type': type(exc).__name__, 'error_message': str(exc), 'uses_gold_answer': False})
            processed_count += 1
            if processed_count % checkpoint_every == 0:
                _checkpoint(run_root, records, errors)
    records.sort(key=lambda row: str(row['question_id']))
    errors.sort(key=lambda row: str(row['question_id']))
    all_records = [*records, *errors]
    _checkpoint(run_root, records, errors)
    output_ids = [str(row['question_id']) for row in all_records]
    purity_rows = [row.get('purity', {}) for row in records]
    audit = {'schema_version': 'nextqa_graph_retrieval_audit_v1', 'run_id': args.run_id, 'graph_retrieval_profile': config.profile_name, 'question_count': len(ir_rows), 'completed_question_count': len(records), 'error_question_count': len(errors), 'video_count': len({row['video_id'] for row in records}), 'duplicate_output_question_id_count': len(output_ids) - len(set(output_ids)), 'missing_question_ids': sorted(set(question_ids) - set(output_ids)), 'evidence_count_distribution': dict(sorted(Counter((len(row.get('evidence', [])) for row in records)).items())), 'graph_evidence_coverage': sum((bool(row.get('graph_record_ids')) for row in records)) / len(ir_rows), 'official_type_purity': all((not row.get('uses_official_type', True) for row in purity_rows)), 'answer_choice_purity': all((not row.get('uses_answer_choices', True) for row in purity_rows)), 'gold_purity': all((not row.get('uses_gold_answer', True) for row in purity_rows)), 'visual_purity': all((not row.get('reads_rgb', True) and (not row.get('reads_object_crop', True)) and (not row.get('reads_visual_embedding_cache', True)) and (not row.get('uses_visual_index', True)) for row in purity_rows)), 'faiss_used': False, 'api_called': False, 'shared_hybrid_retrieve_called': False, 'uses_gold_answer': False}
    atomic_write_json(run_root / 'retrieval_records.json', sorted(all_records, key=lambda row: row['question_id']))
    atomic_write_json(run_root / 'retrieval_audit.json', audit)
    atomic_write_json(run_root / 'retrieval_manifest.json', {'schema_version': 'nextqa_graph_retrieval_manifest_v1', 'run_id': args.run_id, 'dataset': 'NExT-QA', 'split': 'val', 'graph_run_root': str(graph_run_root), 'question_ir_path': str(question_ir_path), 'config_path': str(config_path), 'retrieval_records_path': str(run_root / 'retrieval_records.json'), 'question_count': len(ir_rows), 'completed_question_count': len(records), 'error_question_count': len(errors), 'uses_gold_answer': False})
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=source_snapshot, current_source_paths=current_sources)
    structurally_valid = len(records) + len(errors) == len(ir_rows) and (not audit['duplicate_output_question_id_count']) and (not audit['missing_question_ids']) and audit['official_type_purity'] and audit['answer_choice_purity'] and audit['gold_purity'] and audit['visual_purity'] and (not drift['code_drift_detected'])
    status_name = _status_name(structurally_valid=structurally_valid, error_count=len(errors))
    status = {'status': status_name, 'run_id': args.run_id, 'completed_at_utc': utc_now(), 'completed_question_count': len(records), 'error_question_count': len(errors), 'code_drift_detected': drift['code_drift_detected'], 'uses_gold_answer': False}
    atomic_write_json(run_root / 'run_status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 1 if status_name == 'invalid' else 0
if __name__ == '__main__':
    raise SystemExit(main())
