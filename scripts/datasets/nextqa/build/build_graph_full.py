"""Build and audit the complete 570-video NExT-QA validation graph set."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping
from vlkg.core.assets.asset_registry import asset_file
from vlkg.datasets.nextqa.runtime.contracts import FULL_GRAPH_STAGE, load_full_manifest, prepare_full_graph_run_root
from vlkg.datasets.nextqa.audit.graph_audit import build_graph_audit, build_graph_manifest, persist_graph_reports
from vlkg.datasets.nextqa.graph.hsgg_input_adapter import load_normalized_video_input, normalize_hsgg_archive
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT, atomic_write_json
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot, utc_now
from vlkg.datasets.nextqa.graph.temporal_canonical import CANONICAL_OPTIONS, BuiltVideoGraph, build_video_graph, load_nextqa_graph_config
from vlkg.core.indexing.vision import EmbeddingCache, VisionFeatureExtractor
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
PROJECT_PYTHON = Path(sys.executable).resolve()
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'datasets' / 'nextqa' / 'graph.yaml'
SIGLIP_MODEL = 'google/siglip-base-patch16-224'
EXPECTED_VIDEO_COUNT = 570

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manifest', type=Path, default=None, help='New builds use the selected source manifest; resumes use the recorded path.')
    parser.add_argument('--config', type=Path, default=None, help='New runs use configs/datasets/nextqa/graph.yaml; resume uses the recorded config.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--normalization-batch-size', type=int, default=285, help='videos per input batch; explicit archive imports are grouped by archive before batching')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()

def resolve_build_config(args: argparse.Namespace) -> Path:
    """Select configuration without creating a run or relaxing resume validation."""
    if args.config is not None:
        return args.config.resolve()
    if not args.resume:
        return DEFAULT_CONFIG
    from vlkg.datasets.nextqa.runtime.contracts import RUN_ID_RE
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError(f'invalid run ID: {args.run_id!r}')
    record = NEXTQA_SCRATCH_ROOT / FULL_GRAPH_STAGE / args.run_id / 'run_config.json'
    payload = json.loads(record.read_text(encoding='utf-8'))
    path = payload['contract']['config']['path']
    if not isinstance(path, str) or not path.strip():
        raise ValueError('resume run_config contains no configuration path')
    return (REPO_ROOT / path).resolve()

def resolve_build_manifest(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return args.manifest.resolve()
    if not args.resume:
        return asset_file('nextqa', 'source', 'manifest', REPO_ROOT)
    from vlkg.datasets.nextqa.runtime.contracts import RUN_ID_RE
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError(f'invalid run ID: {args.run_id!r}')
    record = NEXTQA_SCRATCH_ROOT / FULL_GRAPH_STAGE / args.run_id / 'run_config.json'
    payload = json.loads(record.read_text(encoding='utf-8'))
    path = payload['contract']['manifest']['path']
    if not isinstance(path, str) or not path.strip():
        raise ValueError('resume run_config contains no source manifest path')
    return (REPO_ROOT / path).resolve()

def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())

def _load_completed_results(run_root: Path) -> list[BuiltVideoGraph]:
    manifest_path = run_root / 'nextqa_graph_build_manifest.json'
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    completed: list[BuiltVideoGraph] = []
    for row in payload.get('videos', []):
        if row.get('status') != 'completed':
            continue
        required_paths = {field: (REPO_ROOT / Path(str(row[field]))).resolve() for field in ('temporal_graph_path', 'canonical_graph_path', 'canonical_report_path', 'video_audit_path')}
        if not all((path.is_file() for path in required_paths.values())):
            continue
        audit = json.loads(required_paths['video_audit_path'].read_text(encoding='utf-8'))
        completed.append(BuiltVideoGraph(official_video_id=str(row['official_video_id']), local_sequential_id=int(row['local_sequential_id']), temporal_graph_path=required_paths['temporal_graph_path'], canonical_graph_path=required_paths['canonical_graph_path'], canonical_report_path=required_paths['canonical_report_path'], audit_path=required_paths['video_audit_path'], audit=audit))
    return completed

def select_pending_videos(videos: Iterable[Mapping[str, Any]], *, completed: Iterable[BuiltVideoGraph]) -> list[Mapping[str, Any]]:
    completed_ids = {item.official_video_id for item in completed}
    return [video for video in videos if str(video['official_video_id']) not in completed_ids]

def _expected_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    videos = manifest['videos']
    return {'videos': len(videos), 'sampled_frames': sum((int(video['sampled_image_count']) for video in videos)), 'completed_hsgg_frames': sum((int(video['hsgg_frame_count']) for video in videos)), 'skipped_frames': sum((int(video['skipped_frame_count']) for video in videos))}

def _checkpoint(*, run_id: str, manifest_path: Path, config_path: Path, graph_profile: str, run_root: Path, built: list[BuiltVideoGraph], errors: list[dict[str, Any]]) -> None:
    atomic_write_json(run_root / 'nextqa_graph_build_manifest.json', build_graph_manifest(run_id=run_id, source_manifest_path=manifest_path, config_path=config_path, graph_profile=graph_profile, run_root=run_root, built=built, errors=errors))
    atomic_write_json(run_root / 'run_status.json', {'status': 'running', 'updated_at_utc': utc_now(), 'run_id': run_id, 'expected_video_count': EXPECTED_VIDEO_COUNT, 'completed_video_count': len(built), 'error_video_count': len(errors), 'uses_gold_answer': False})

def _batches(items: list[Mapping[str, Any]], size: int) -> Iterable[list[Mapping[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]

def input_batches(videos: list[Mapping[str, Any]], size: int) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    """Group explicit imports by archive; reuse extracted directories directly."""
    if size < 1:
        raise ValueError('input batch size must be positive')
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for video in videos:
        storage = str(video.get('input_storage', 'archive'))
        if storage not in {'normalized_directory', 'archive'}:
            raise ValueError(f'unsupported input storage: {storage}')
        archive_key = str(Path(str(video['hsgg_archive'])).resolve()) if storage == 'archive' else ''
        groups.setdefault((storage, archive_key), []).append(video)
    for (storage, _), group in groups.items():
        for batch in _batches(group, size):
            yield storage, batch

def main() -> int:
    args = parse_args()
    if Path(sys.executable).resolve() != PROJECT_PYTHON:
        raise RuntimeError(f'NExT-QA jobs must use the project virtual environment: {PROJECT_PYTHON}')
    if args.normalization_batch_size < 1 or args.normalization_batch_size > 570:
        raise ValueError('normalization batch size must be in 1..570')
    manifest_path = resolve_build_manifest(args)
    config_path = resolve_build_config(args)
    manifest = load_full_manifest(manifest_path)
    config = load_nextqa_graph_config(config_path, config_dir=REPO_ROOT / 'configs')
    run_root = prepare_full_graph_run_root(args.run_id, resume=args.resume)
    cache_path = run_root / 'cache' / 'siglip_object_crop.sqlite'
    expected_counts = _expected_counts(manifest)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'full_validation_graph_build', 'run_id': args.run_id, 'graph_profile': config.profile_name, 'canonical_profile': 'time_fact_core', 'manifest': fingerprint_file(manifest_path), 'config': fingerprint_file(config_path), 'expected_counts': expected_counts, 'canonical_options': CANONICAL_OPTIONS, 'profile_parameters': config.model_dump(mode='json'), 'construction_embedding_model': SIGLIP_MODEL, 'construction_embedding_device': args.device, 'construction_embedding_local_files_only': True, 'construction_embedding_cache': str(cache_path), 'normalization_batch_size': args.normalization_batch_size, 'input_storage_modes': sorted({str(video.get('input_storage', 'archive')) for video in manifest['videos']}), 'normalized_directory_policy': 'read selected source tables and frames in place without copying', 'uses_gold_answer': False, 'reads_questions': False, 'reads_answer_choices': False, 'reads_official_question_type': False, 'reads_sampled_frame_pixels_for_graph_construction': True, 'uses_visual_embeddings_for_graph_construction': True, 'qa_reads_rgb': False, 'qa_reads_object_crop': False, 'qa_reads_embedding_cache': False, 'qa_uses_faiss': False, 'uses_uniform32': False, 'resume_policy': 'skip completed videos and retry only error or missing videos', 'output_root': str(run_root)}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_full_graph_run_config_v1')
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    source_snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=source_paths, resume=args.resume)
    atomic_write_json(run_root / 'run_status.json', {'status': 'running', 'started_or_resumed_at_utc': utc_now(), 'run_id': args.run_id, 'expected_video_count': EXPECTED_VIDEO_COUNT, 'uses_gold_answer': False})
    built = _load_completed_results(run_root) if args.resume else []
    pending = select_pending_videos(manifest['videos'], completed=built)
    errors: list[dict[str, Any]] = []
    extractor: VisionFeatureExtractor | None = None
    embedding_cache: EmbeddingCache | None = None
    try:
        if pending:
            extractor = VisionFeatureExtractor(model_name=SIGLIP_MODEL, device=args.device, local_files_only=True)
            embedding_cache = EmbeddingCache(cache_path)
        for batch_index, (storage, batch) in enumerate(input_batches(pending, args.normalization_batch_size), start=1):
            print(f"loading inputs mode={storage} batch={batch_index} videos={len(batch)} local_ids={batch[0]['local_sequential_id']}..{batch[-1]['local_sequential_id']}", flush=True)
            normalized = []
            if storage == 'normalized_directory':
                for video in batch:
                    try:
                        normalized.append(load_normalized_video_input(video, repository_root=REPO_ROOT))
                    except Exception as exc:
                        errors.append({'official_video_id': str(video['official_video_id']), 'local_sequential_id': int(video['local_sequential_id']), 'status': 'error', 'error_stage': 'normalized_input_validation', 'error_type': type(exc).__name__, 'error_message': str(exc)})
                        print(f"input error video_id={video['official_video_id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        _checkpoint(run_id=args.run_id, manifest_path=manifest_path, config_path=config_path, graph_profile=config.profile_name, run_root=run_root, built=built, errors=errors)
            else:
                try:
                    normalized = normalize_hsgg_archive(batch, output_root=run_root / 'normalized_inputs', materialize_frames=True)
                except Exception as exc:
                    errors.extend(({'official_video_id': str(video['official_video_id']), 'local_sequential_id': int(video['local_sequential_id']), 'status': 'error', 'error_stage': 'batch_normalization', 'error_type': type(exc).__name__, 'error_message': str(exc)} for video in batch))
                    _checkpoint(run_id=args.run_id, manifest_path=manifest_path, config_path=config_path, graph_profile=config.profile_name, run_root=run_root, built=built, errors=errors)
                    continue
            for item in normalized:
                try:
                    result = build_video_graph(item, config=config, output_root=run_root / 'graphs', vision_feature_extractor=extractor, embedding_cache=embedding_cache)
                    built.append(result)
                    print(f'completed {len(built)}/{EXPECTED_VIDEO_COUNT} local_id={item.local_sequential_id} video_id={item.official_video_id}', flush=True)
                except Exception as exc:
                    errors.append({'official_video_id': item.official_video_id, 'local_sequential_id': item.local_sequential_id, 'status': 'error', 'error_stage': 'graph_build', 'error_type': type(exc).__name__, 'error_message': str(exc)})
                    print(f'error local_id={item.local_sequential_id} video_id={item.official_video_id}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                _checkpoint(run_id=args.run_id, manifest_path=manifest_path, config_path=config_path, graph_profile=config.profile_name, run_root=run_root, built=built, errors=errors)
    finally:
        if embedding_cache is not None:
            embedding_cache.close()
    current_sources = collect_imported_repo_sources(extra_paths=[Path(__file__).resolve()])
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=source_snapshot, current_source_paths=current_sources)
    graph_manifest = build_graph_manifest(run_id=args.run_id, source_manifest_path=manifest_path, config_path=config_path, graph_profile=config.profile_name, run_root=run_root, built=built, errors=errors)
    graph_audit = build_graph_audit(run_id=args.run_id, graph_profile=config.profile_name, built=built, errors=errors, source_drift=drift, expected_counts=expected_counts)
    persist_graph_reports(run_root, graph_manifest=graph_manifest, graph_audit=graph_audit)
    final_status = 'completed' if graph_audit['status'] == 'passed' else 'invalid'
    atomic_write_json(run_root / 'run_status.json', {'status': final_status, 'finished_at_utc': utc_now(), 'run_id': args.run_id, 'expected_video_count': EXPECTED_VIDEO_COUNT, 'completed_video_count': len(built), 'error_video_count': len(errors), 'code_drift_detected': drift['code_drift_detected'], 'uses_gold_answer': False, 'graph_build_audit': _relative(run_root / 'graph_build_audit.json')})
    print(f'run_root={run_root}', flush=True)
    print(f'status={final_status}', flush=True)
    return 0 if final_status == 'completed' else 1
if __name__ == '__main__':
    raise SystemExit(main())
