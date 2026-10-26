from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.contracts import prepare_stage_run_root
from vlkg.datasets.nextqa.graph.hsgg_input_adapter import resolve_normalized_input_paths
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS, build_before_after_candidates, build_event_boundary_candidates, build_frame_candidates, build_object_candidates, build_temporal_window_candidates, build_tracklet_candidates, encode_full_frames, load_visual_index_config, open_embedding_cache_read_only, write_faiss_shard
from vlkg.core.indexing.vision import EmbeddingCache, VisionFeatureExtractor
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.assets.asset_resume import recorded_or_selected
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'datasets/nextqa/visual_index_build.json'
DEFAULT_GRAPH_ROOT = (REPO_ROOT / 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/runtime_paths_v3')
DEFAULT_SELECTION = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'dry_run' / 'pilot5' / 'nextqa_val_pilot5_manifest.json'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build question-independent NExT-QA visual index shards.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--graph-root', type=Path, default=None)
    parser.add_argument('--selection-manifest', type=Path, default=None)
    parser.add_argument('--object-embedding-cache', type=Path, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', action='store_true')
    return parser

def resolve_inputs(args, run_root):

    def resolve(explicit, field, selected, parent=False):
        return recorded_or_selected(explicit, resume=args.resume, run_config=run_root / 'run_config.json', field=field, selected=selected, parent=parent)
    values = {'config': resolve(args.config, 'config', lambda: REPO_ROOT / 'configs/datasets/nextqa/visual_index_build.json'), 'graph_root': resolve(args.graph_root, 'graph_manifest', lambda: asset_file('nextqa', 'graphs', 'manifest').parent, True), 'selection': resolve(args.selection_manifest, 'selection_manifest', lambda: asset_file('nextqa', 'source', 'manifest'))}
    values['object_cache'] = resolve(getattr(args, 'object_embedding_cache', None), 'object_embedding_cache', lambda: values['graph_root'] / 'cache/siglip_object_crop.sqlite' if args.graph_root is not None else asset_file('nextqa', 'cache', 'sqlite'))
    return values

def _load_object(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{name} must be a JSON object')
    return payload

def resolve_selected_frame_dir(video: dict[str, Any], graph: dict[str, Any], *, repository_root: Path=REPO_ROOT) -> Path:
    """Use the selected source item instead of stale frame paths in frozen graphs."""
    if video.get('input_storage') == 'normalized_directory':
        _, frames_dir = resolve_normalized_input_paths(video, repository_root=repository_root)
        return frames_dir
    source_dir = Path(str(graph.get('source_dir') or ''))
    return ((source_dir if source_dir.is_absolute() else repository_root / source_dir) / 'frames').resolve()

def main() -> None:
    args = build_parser().parse_args()
    run_root = prepare_stage_run_root('visual_indexes', args.run_id, resume=args.resume)
    inputs = resolve_inputs(args, run_root)
    config_path = inputs['config']
    graph_root = inputs['graph_root']
    selection_path = inputs['selection']
    graph_manifest_path = graph_root / 'graph_manifest.json'
    if not graph_manifest_path.is_file():
        graph_manifest_path = graph_root / 'nextqa_graph_build_manifest.json'
    object_cache_path = inputs['object_cache']
    config = load_visual_index_config(config_path)
    selection = _load_object(selection_path, 'selection manifest')
    graph_manifest = _load_object(graph_manifest_path, 'graph build manifest')
    selected = {str(video['official_video_id']): video for video in selection.get('videos', [])}
    graph_rows = {str(video['official_video_id']): video for video in graph_manifest.get('videos', []) if str(video['official_video_id']) in selected}
    if set(graph_rows) != set(selected):
        raise ValueError('selection videos do not match the graph build manifest')
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'question_independent_visual_indexing', 'config': fingerprint_file(config_path), 'graph_manifest': fingerprint_file(graph_manifest_path), 'selection_manifest': fingerprint_file(selection_path), 'object_embedding_cache': fingerprint_file(object_cache_path), 'selected_video_ids': sorted(selected), 'device': args.device, 'uses_questions': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'calls_api': False, 'qa_receives_images': False}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_index_run_config_v1')
    source_paths = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=source_paths, resume=args.resume)
    object_connection = open_embedding_cache_read_only(object_cache_path)
    frame_cache = EmbeddingCache(run_root / 'cache' / 'siglip_full_frame.sqlite')
    extractor = VisionFeatureExtractor(model_name=config.embedding_model, device=None if args.device == 'auto' else args.device, local_files_only=True)
    video_manifests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        for video_id in sorted(selected, key=lambda value: int(selected[value]['local_sequential_id'])):
            video_root = run_root / 'videos' / video_id
            video_manifest_path = video_root / 'video_index_manifest.json'
            if args.resume and video_manifest_path.is_file():
                prior = _load_object(video_manifest_path, 'video index manifest')
                if prior.get('status') == 'completed':
                    video_manifests.append(prior)
                    continue
            try:
                graph_path = (REPO_ROOT / Path(graph_rows[video_id]['temporal_graph_path'])).resolve()
                graph = _load_object(graph_path, 'temporal graph')
                graph_dir = graph_path.parent
                frame_dir = resolve_selected_frame_dir(selected[video_id], graph)
                frame_paths = sorted(frame_dir.glob('*.png'))
                if len(frame_paths) != int(selected[video_id]['sampled_image_count']):
                    raise ValueError(f"sampled frame mismatch for {video_id}: {len(frame_paths)} != {selected[video_id]['sampled_image_count']}")
                object_meta, object_vectors, object_misses = build_object_candidates(graph, object_connection)
                tracklet_meta, tracklet_vectors, tracklet_misses = build_tracklet_candidates(graph)
                frame_embeddings = encode_full_frames(frame_paths, cache=frame_cache, extractor=extractor, batch_size=config.frame_embedding_batch_size)
                frame_meta, frame_vectors = build_frame_candidates(graph, frame_paths, frame_embeddings)
                window_meta, window_vectors = build_temporal_window_candidates(graph, frame_meta, frame_vectors, window_size_sec=config.window_size_sec, window_stride_sec=config.window_stride_sec)
                boundary_meta, boundary_vectors = build_event_boundary_candidates(graph, frame_meta, frame_vectors, radius_sec=config.event_boundary_radius_sec)
                pair_meta, pair_vectors = build_before_after_candidates(graph, frame_meta, frame_vectors)
                candidate_sets = {'object': (object_meta, object_vectors), 'frame': (frame_meta, frame_vectors), 'tracklet': (tracklet_meta, tracklet_vectors), 'temporal_window': (window_meta, window_vectors), 'event_boundary': (boundary_meta, boundary_vectors), 'before_after_pair': (pair_meta, pair_vectors)}
                shards = {kind: write_faiss_shard(video_root, kind=kind, metadata=candidate_sets[kind][0], vectors=candidate_sets[kind][1], expected_dimension=config.embedding_dimension) for kind in CANDIDATE_KINDS}
                video_manifest = {'schema_version': 'nextqa_visual_video_index_manifest_v1', 'status': 'completed', 'video_id': video_id, 'local_sequential_id': int(selected[video_id]['local_sequential_id']), 'temporal_graph_path': str(graph_path), 'graph_dir': str(graph_dir), 'sampled_frame_count': len(frame_paths), 'hsgg_observation_count': len(graph.get('observations', [])), 'object_embedding_miss_count': object_misses, 'tracklet_embedding_miss_count': tracklet_misses, 'shards': shards, 'uses_questions': False, 'uses_gold_answer': False, 'qa_receives_images': False}
                if object_misses:
                    raise RuntimeError(f'object embedding cache misses for {video_id}: {object_misses}')
                atomic_write_json(video_manifest_path, video_manifest)
                video_manifests.append(video_manifest)
                print(f'video_id={video_id} frames={len(frame_paths)} ' + ' '.join((f"{kind}={shards[kind]['count']}" for kind in CANDIDATE_KINDS)), flush=True)
            except Exception as exc:
                errors.append({'video_id': video_id, 'error_type': type(exc).__name__, 'error_message': str(exc)})
                print(f'video_id={video_id} error={type(exc).__name__}: {exc}', flush=True)
    finally:
        object_connection.close()
        frame_cache.close()
    kind_totals = Counter()
    for video in video_manifests:
        for kind, shard in video.get('shards', {}).items():
            kind_totals[kind] += int(shard.get('count') or 0)
    audit = {'schema_version': 'nextqa_visual_index_audit_v1', 'run_id': args.run_id, 'selected_video_count': len(selected), 'completed_video_count': len(video_manifests), 'error_video_count': len(errors), 'errors': errors, 'candidate_kind_totals': dict(sorted(kind_totals.items())), 'all_sampled_frames_indexed': sum((int(video['sampled_frame_count']) for video in video_manifests)) == sum((int(video['shards']['frame']['count']) for video in video_manifests)), 'zero_object_embedding_misses': all((int(video['object_embedding_miss_count']) == 0 for video in video_manifests)), 'same_video_shards': True, 'uses_questions': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'calls_api': False, 'qa_receives_images': False, 'status': 'passed'}
    if errors or len(video_manifests) != len(selected) or (not audit['all_sampled_frames_indexed']):
        audit['status'] = 'failed'
    atomic_write_json(run_root / 'visual_index_audit.json', audit)
    atomic_write_json(run_root / 'visual_index_manifest.json', {'schema_version': 'nextqa_visual_index_manifest_v1', 'run_id': args.run_id, 'profile_name': config.profile_name, 'embedding_model': config.embedding_model, 'embedding_dimension': config.embedding_dimension, 'selection_manifest_path': str(selection_path), 'graph_root': str(graph_root), 'frame_embedding_cache': str((run_root / 'cache' / 'siglip_full_frame.sqlite').resolve()), 'videos': video_manifests, 'candidate_kind_totals': dict(sorted(kind_totals.items())), 'uses_questions': False, 'uses_gold_answer': False, 'qa_receives_images': False})
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=source_paths)
    if drift['code_drift_detected']:
        audit['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'visual_index_audit.json', audit)
    atomic_write_json(run_root / 'run_status.json', {'status': audit['status'], 'selected_video_count': len(selected), 'completed_video_count': len(video_manifests), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if audit['status'] != 'passed':
        raise SystemExit(f'visual index pilot failed: {audit}')
    print(json.dumps(audit, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
