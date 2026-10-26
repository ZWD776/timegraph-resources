from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from scripts.datasets.openeqa._shared import repo_path, write_json_atomic
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
BUILD_DIR = REPO_ROOT / 'scripts' / 'build'
SRC_ROOT = REPO_ROOT / 'src'
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
FRAME_STEM_PATTERN = re.compile('^(?P<index>\\d+)(?:[-_].*)?$')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build OpenEQA temporal/canonical graphs from completed HSGG episode outputs. The script keeps source path/frame-name adaptation outside the generic graph transformer.')
    parser.add_argument('--hsgg-root', type=Path, default=None, help='Explicit legacy HSGG root; requires --frames-root. Otherwise use the selected OpenEQA source manifest.')
    parser.add_argument('--frames-root', type=Path, default=None, help='Explicit legacy RGB root; requires --hsgg-root.')
    parser.add_argument('--source-manifest', type=Path, default=None, help='OpenEQA source manifest with shards.<hm3d/scannet>.hsgg and .frames. Defaults to the selected source asset.')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--data-source', default='hm3d-v0')
    parser.add_argument('--config', type=Path, default=Path('datasets/openeqa/graph.yaml'))
    parser.add_argument('--graph-profile', choices=('time_fact_core',), default='time_fact_core')
    parser.add_argument('--graph-build-adapter', choices=('none',), default='none', help='Optional QA-independent HM3D build adapter. The default keeps the legacy generic graph construction path unchanged.')
    parser.add_argument('--ordinal-step', type=float, default=1.0, help='Step size for the ordinal graph time axis when no real frame timestamps are available. This is not treated as physical seconds.')
    parser.add_argument('--frame-timestamp-index', type=Path, default=None, help='Optional JSON file with real frame timestamps. Expected shape: {"frames": [{"episode_history": ..., "source_frame_name": ..., "timestamp_sec": ...}, ...]}. If omitted, the graph uses an ordinal order axis.')
    parser.add_argument('--episode-history', action='append', default=None)
    parser.add_argument('--limit-episodes', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--unified-openeqa-contract', action='store_true', help='Write the source-neutral OpenEQA graph-build manifest contract. The historical filename/contract remains available for old runs.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--standardize-only', action='store_true', help='Write standard JSON/frame_time_index files but do not build graphs.')
    parser.add_argument('--allow-missing-local-images', action='store_true', help='Keep HSGG image-path metadata when local RGB files are unavailable. This is intended for graph-only builds; it never reads image pixels.')
    parser.add_argument('--embedding-cache-db', type=Path, default=None, help='Optional dataset-owned object embedding cache used only by entity association during graph construction.')
    return parser

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def source_frame_sort_key(frame_name: str) -> tuple[int, str]:
    match = FRAME_STEM_PATTERN.match(frame_name)
    if match:
        return (int(match.group('index')), frame_name)
    return (10 ** 12, frame_name)

def source_frame_index(frame_name: str) -> int | None:
    match = FRAME_STEM_PATTERN.match(frame_name)
    if not match:
        return None
    return int(match.group('index'))

def resolve_input_roots(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Select OpenEQA-owned inputs, keeping paired explicit roots available."""
    if (args.hsgg_root is None) != (args.frames_root is None):
        raise ValueError('--hsgg-root and --frames-root must be supplied together')
    if args.hsgg_root is not None:
        if args.source_manifest is not None:
            raise ValueError('--source-manifest cannot be combined with explicit input roots')
        return repo_path(args.hsgg_root), repo_path(args.frames_root), None
    if args.source_manifest is None:
        from vlkg.core.assets.asset_registry import asset_file
        manifest_path = asset_file('openeqa', 'source', 'manifest', REPO_ROOT)
    else:
        manifest_path = repo_path(args.source_manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or str(manifest.get('dataset', '')).lower() != 'openeqa':
        raise ValueError(f'Not an OpenEQA source manifest: {manifest_path}')
    shard_key = {'hm3d-v0': 'hm3d', 'scannet-v0': 'scannet'}.get(args.data_source)
    shards = manifest.get('shards')
    shard = shards.get(shard_key) if shard_key and isinstance(shards, dict) else None
    if not isinstance(shard, dict) or not all(isinstance(shard.get(key), str) and shard[key] for key in ('hsgg', 'frames')):
        raise ValueError(f'No input roots for OpenEQA data source {args.data_source!r} in {manifest_path}')
    return repo_path(shard['hsgg']), repo_path(shard['frames']), manifest_path

def load_episode_hsgg_tables(hsgg_dir: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Read completed sidecars before legacy aggregates, without changing sources."""
    frame_dir = hsgg_dir / 'frame_scene_graphs'
    if not frame_dir.is_dir():
        tables = {key: load_json(hsgg_dir / filename) for key, filename in (
            ('objects', 'objects.json'), ('attributes', 'attributes.json'),
            ('relations', 'relations.json'), ('hierarchy', 'hierarchy_mapping.json'))}
        return tables, 'aggregate_json_legacy'
    paths = sorted(frame_dir.glob('*.json'), key=lambda path: source_frame_sort_key(path.stem))
    if not paths:
        raise ValueError(f'No completed HSGG frame sidecars: {frame_dir}')
    actual = {path.stem for path in paths}
    run_manifest_path = hsgg_dir / 'run_manifest.json'
    if run_manifest_path.is_file():
        run_manifest = load_json(run_manifest_path)
        frames = run_manifest.get('frames') if isinstance(run_manifest, dict) else None
        if not isinstance(frames, dict):
            raise ValueError(f'HSGG run manifest must contain frame statuses: {run_manifest_path}')
        expected = {key for key, value in frames.items() if isinstance(value, dict) and value.get('status') == 'completed'}
        if actual != expected:
            raise ValueError(f'Completed HSGG sidecars differ from run manifest: {hsgg_dir}; missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}')
        if int(run_manifest.get('completed_frames', -1)) != len(expected):
            raise ValueError(f'Completed HSGG frame count differs from run manifest: {hsgg_dir}')
    tables: dict[str, dict[str, Any]] = {key: {} for key in ('objects', 'attributes', 'relations', 'hierarchy')}
    for path in paths:
        raw = load_json(path)
        if not isinstance(raw, dict) or not all(isinstance(raw.get(key), kind) for key, kind in (
            ('objects', dict), ('attributes', dict), ('relations', list), ('hierarchy', dict))):
            raise ValueError(f'Invalid HSGG frame sidecar: {path}')
        objects = dict(raw['objects'])
        if raw.get('image_path'):
            objects['image_path'] = raw['image_path']
        tables['objects'][path.stem] = objects
        for key in ('attributes', 'relations', 'hierarchy'):
            tables[key][path.stem] = raw[key]
    return tables, 'completed_frame_scene_graphs'

def resolve_hsgg_data_source_root(hsgg_root: Path, data_source: str) -> Path:
    root = repo_path(hsgg_root)
    candidate = root / data_source
    if candidate.exists():
        return candidate
    return root

def resolve_frames_data_source_root(frames_root: Path, data_source: str) -> Path:
    root = repo_path(frames_root)
    candidate = root / data_source
    if candidate.exists():
        return candidate
    return root

def discover_episode_histories(hsgg_root: Path, data_source: str) -> list[str]:
    data_root = resolve_hsgg_data_source_root(hsgg_root, data_source)
    episodes: list[str] = []
    for child in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if (child / 'frame_scene_graphs').is_dir() or (child / 'objects.json').is_file():
            episodes.append(f'{data_source}/{child.name}')
    return episodes

def episode_hsgg_dir(hsgg_root: Path, episode_history: str, data_source: str) -> Path:
    root = repo_path(hsgg_root)
    direct = root / Path(episode_history)
    if direct.exists():
        return direct
    _, _, episode_id = episode_history.partition('/')
    return resolve_hsgg_data_source_root(root, data_source) / episode_id

def episode_frames_dir(frames_root: Path, episode_history: str, data_source: str) -> Path:
    root = repo_path(frames_root)
    direct = root / Path(episode_history)
    if direct.exists():
        return direct
    _, _, episode_id = episode_history.partition('/')
    return resolve_frames_data_source_root(root, data_source) / episode_id

def resolve_local_image_path(*, frame_key: str, source_image_path: str | None, frames_dir: Path, allow_missing: bool=False) -> Path:
    if source_image_path:
        candidate = frames_dir / Path(str(source_image_path).replace('\\', '/')).name
        if candidate.exists():
            return candidate.resolve()
    for suffix in IMAGE_SUFFIXES:
        candidate = frames_dir / f'{frame_key}{suffix}'
        if candidate.exists():
            return candidate.resolve()
    if allow_missing:
        image_name = Path(str(source_image_path).replace('\\', '/')).name if source_image_path else f'{frame_key}.png'
        return (frames_dir / image_name).resolve()
    raise FileNotFoundError(f'Missing local image for frame={frame_key} under {frames_dir}')

def load_frame_timestamp_index(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    resolved = repo_path(path)
    payload = load_json(resolved)
    frames = payload.get('frames', []) if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError(f'Frame timestamp index must contain a frames list: {resolved}')
    index: dict[tuple[str, str], float] = {}
    for item in frames:
        if not isinstance(item, dict):
            continue
        episode_history = str(item.get('episode_history') or '').strip()
        source_frame_name = str(item.get('source_frame_name') or item.get('frame_name') or item.get('image_name') or '').strip()
        if source_frame_name:
            source_frame_name = Path(source_frame_name).stem
        timestamp = item.get('timestamp_sec', item.get('timestamp_unix'))
        if episode_history and source_frame_name and (timestamp is not None):
            index[episode_history, source_frame_name] = float(timestamp)
    return index

def standardize_episode(*, hsgg_dir: Path, frames_dir: Path, output_dir: Path, episode_history: str, ordinal_step: float, frame_timestamp_index: dict[tuple[str, str], float], graph_build_adapter: str='none', allow_missing_local_images: bool=False, force: bool, dry_run: bool) -> dict[str, Any]:
    if graph_build_adapter != 'none':
        raise ValueError('Only the selected source-neutral build adapter is active')
    standard_dir = output_dir / 'standard_json'
    frame_time_index_path = output_dir / 'frame_time_index.json'
    if not force and (standard_dir / 'objects.json').exists() and frame_time_index_path.exists():
        if (hsgg_dir / 'frame_scene_graphs').is_dir():
            cached_time = load_json(frame_time_index_path)
            if cached_time.get('hsgg_record_source') != 'completed_frame_scene_graphs':
                raise ValueError(f'Existing standardized inputs do not record the completed-sidecar source contract: {output_dir}; use --force or a new --run-id')
        cached: dict[str, Any] = {'standard_json_dir': str(standard_dir), 'frame_time_index_json': str(frame_time_index_path), 'standardized': False}
        return cached
    tables, record_source = load_episode_hsgg_tables(hsgg_dir)
    objects, attributes, relations, hierarchy = (tables[key] for key in ('objects', 'attributes', 'relations', 'hierarchy'))
    frame_keys = sorted(objects.keys(), key=source_frame_sort_key)
    key_sets = [set(table.keys()) for table in (attributes, relations, hierarchy)]
    missing = [sorted(set(frame_keys) - keys)[:5] for keys in key_sets]
    if any((items for items in missing)):
        raise ValueError(f'Inconsistent HSGG frame keys for {episode_history}; missing samples={missing}')
    std_objects: dict[str, Any] = {}
    std_attributes: dict[str, Any] = {}
    std_relations: dict[str, Any] = {}
    std_hierarchy: dict[str, Any] = {}
    frame_index_rows: list[dict[str, Any]] = []
    uses_real_timestamps = False
    missing_local_image_count = 0
    for rank, frame_key in enumerate(frame_keys):
        std_frame_name = f'frame_{rank}'
        object_payload = copy.deepcopy(objects[frame_key])
        local_image_path = resolve_local_image_path(frame_key=frame_key, source_image_path=object_payload.get('image_path'), frames_dir=frames_dir, allow_missing=allow_missing_local_images)
        if not local_image_path.exists():
            missing_local_image_count += 1
        object_payload['image_path'] = str(local_image_path)
        std_objects[std_frame_name] = object_payload
        std_attributes[std_frame_name] = copy.deepcopy(attributes.get(frame_key, {}))
        frame_relations = copy.deepcopy(relations.get(frame_key, []))
        std_relations[std_frame_name] = frame_relations
        std_hierarchy[std_frame_name] = copy.deepcopy(hierarchy.get(frame_key, {}))
        timestamp_value = frame_timestamp_index.get((episode_history, frame_key))
        timestamp_source = 'external_frame_timestamp_index'
        if timestamp_value is None:
            timestamp_value = rank * ordinal_step
            timestamp_source = 'ordinal_frame_order'
        else:
            uses_real_timestamps = True
        frame_index_rows.append({'frame_name': std_frame_name, 'frame_index': rank, 'graph_frame_index': rank, 'timestamp_sec': round(float(timestamp_value), 6), 'timestamp_source': timestamp_source, 'timestamp_is_real_time': timestamp_source != 'ordinal_frame_order', 'source_frame_name': frame_key, 'source_frame_index': source_frame_index(frame_key), 'image_name': local_image_path.name})
    if not dry_run:
        from hsgg_standard_bundle import write_hsgg_bundle
        write_json(standard_dir / 'objects.json', std_objects)
        write_json(standard_dir / 'attributes.json', std_attributes)
        write_json(standard_dir / 'relations.json', std_relations)
        write_json(standard_dir / 'hierarchy_mapping.json', std_hierarchy)
        write_hsgg_bundle(output_json_dir=standard_dir, objects_table=std_objects, attributes_table=std_attributes, relations_table=std_relations, hierarchy_table=std_hierarchy, write_legacy_rewritten_scene_graphs=True)
        write_json(frame_time_index_path, {'contract': 'openeqa_hm3d_frame_time_index_v1', 'episode_history': episode_history, 'source_hsgg_dir': str(hsgg_dir), 'source_frames_dir': str(frames_dir), 'hsgg_record_source': record_source, 'timestamp_semantics': 'external_frame_timestamp_index' if uses_real_timestamps else 'ordinal_frame_order', 'timestamp_is_real_time': uses_real_timestamps, 'ordinal_step': ordinal_step, 'note': 'timestamp_sec is an ordinal graph ordering value unless timestamp_is_real_time is true.', 'frames': frame_index_rows})
    result = {'standard_json_dir': str(standard_dir), 'frame_time_index_json': str(frame_time_index_path), 'standardized': True, 'hsgg_record_source': record_source, 'completed_frame_count': len(frame_keys), 'missing_local_image_count': missing_local_image_count, 'allow_missing_local_images': allow_missing_local_images}
    return result

def run_command(command: list[str], *, dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        hsgg_root, frames_root, source_manifest_path = resolve_input_roots(args)
    except ValueError as exc:
        parser.error(str(exc))
    for label, path in (('HSGG', hsgg_root), ('frames', frames_root)):
        if not path.is_dir():
            raise SystemExit(f'Missing {label} source root: {path}')
    output_root = repo_path(args.output_root)
    run_id = args.run_id or (f'{args.data_source}-hsgg' if source_manifest_path is not None else hsgg_root.name)
    run_output_root = output_root / run_id
    selected = discover_episode_histories(hsgg_root, args.data_source)
    if args.episode_history:
        requested = set(args.episode_history)
        selected = [episode for episode in selected if episode in requested]
    if args.limit_episodes is not None:
        selected = selected[:args.limit_episodes]
    if not selected:
        raise SystemExit('No HSGG episodes selected.')
    manifest_path = run_output_root / ('openeqa_graph_build_manifest.json' if args.unified_openeqa_contract else 'openeqa_hm3d_graph_build_manifest.json')
    manifest: dict[str, Any] = {'contract': 'openeqa_graph_build_manifest.v1' if args.unified_openeqa_contract else 'openeqa_hm3d_graph_build_manifest_v1', 'dataset': 'OpenEQA', 'data_sources_are_execution_shards_only': bool(args.unified_openeqa_contract), 'generated_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'hsgg_root': str(hsgg_root), 'frames_root': str(frames_root), 'output_root': str(run_output_root), 'data_source': args.data_source, 'config': str(args.config), 'graph_profile': args.graph_profile, 'graph_build_adapter': args.graph_build_adapter, 'allow_missing_local_images': args.allow_missing_local_images, 'qa_independent_build_adapter': args.graph_build_adapter != 'none', 'time_axis': 'external_frame_timestamp_index' if args.frame_timestamp_index is not None else 'ordinal_frame_order', 'timestamp_is_real_time': args.frame_timestamp_index is not None, 'ordinal_step': args.ordinal_step, 'frame_timestamp_index': str(repo_path(args.frame_timestamp_index)) if args.frame_timestamp_index is not None else None, 'embedding_cache_db': str(repo_path(args.embedding_cache_db)) if args.embedding_cache_db is not None else None, 'association_uses_visual_embeddings': None, 'episode_count': len(selected), 'episodes': []}
    manifest['source_manifest'] = str(source_manifest_path) if source_manifest_path is not None else None
    from vlkg.core.config.config import load_domain_config
    domain_config = load_domain_config(args.config, config_dir=REPO_ROOT / 'configs')
    manifest['association_uses_visual_embeddings'] = bool(domain_config.association.enable_visual_embeddings)
    if args.embedding_cache_db is not None:
        embedding_cache_db = repo_path(args.embedding_cache_db)
        if not embedding_cache_db.is_file():
            raise SystemExit(f'Missing embedding cache DB: {embedding_cache_db}')
    else:
        embedding_cache_db = None
    frame_timestamp_index = load_frame_timestamp_index(args.frame_timestamp_index)
    for episode_history in selected:
        hsgg_dir = episode_hsgg_dir(hsgg_root, episode_history, args.data_source)
        frames_dir = episode_frames_dir(frames_root, episode_history, args.data_source)
        if not hsgg_dir.exists():
            raise SystemExit(f'Missing HSGG episode dir: {hsgg_dir}')
        if not frames_dir.exists():
            raise SystemExit(f'Missing frames episode dir: {frames_dir}')
        episode_output_dir = run_output_root / Path(episode_history)
        standard_info = standardize_episode(hsgg_dir=hsgg_dir, frames_dir=frames_dir, output_dir=episode_output_dir, episode_history=episode_history, ordinal_step=args.ordinal_step, frame_timestamp_index=frame_timestamp_index, graph_build_adapter=args.graph_build_adapter, allow_missing_local_images=args.allow_missing_local_images, force=args.force, dry_run=args.dry_run)
        temporal_graph_json = episode_output_dir / 'temporal_graph.json'
        canonical_graph_json = episode_output_dir / 'canonical_temporal_memory_graph.json'
        report_json = episode_output_dir / 'time_fact_core_report.json'
        item: dict[str, Any] = {'episode_history': episode_history, 'hsgg_dir': str(hsgg_dir), 'frames_dir': str(frames_dir), **standard_info, 'temporal_graph_json': str(temporal_graph_json), 'canonical_graph_json': str(canonical_graph_json), 'status': 'standardized'}
        if not args.standardize_only:
            if args.force or not temporal_graph_json.exists():
                temporal_converter = REPO_ROOT / 'scripts/datasets/openeqa/build/convert_openeqa_json_to_temporal_graph.py'
                temporal_command = [sys.executable, str(temporal_converter), str(standard_info['standard_json_dir']), '--output', str(temporal_graph_json), '--video-id', episode_history.replace('/', '__'), '--config', str(args.config), '--frame-time-index', str(standard_info['frame_time_index_json'])]
                if embedding_cache_db is not None:
                    temporal_command.extend(['--embedding-cache-db', str(embedding_cache_db)])
                run_command(temporal_command, dry_run=args.dry_run)
            else:
                print(f'skip_existing_temporal_graph={temporal_graph_json}', flush=True)
            if args.force or not canonical_graph_json.exists():
                canonical_command = [sys.executable, str(BUILD_DIR / 'convert_temporal_graph_to_canonical_graph.py'), str(temporal_graph_json), '--output', str(canonical_graph_json), '--graph-profile', args.graph_profile]
                if args.graph_profile == 'time_fact_core':
                    canonical_command.extend(['--report-json', str(report_json)])
                run_command(canonical_command, dry_run=args.dry_run)
            else:
                print(f'skip_existing_canonical_graph={canonical_graph_json}', flush=True)
            item['status'] = 'built' if not args.dry_run else 'dry_run'
        manifest['episodes'].append(item)
        if not args.dry_run:
            write_json_atomic(manifest_path, manifest)
    print(f'graph_build_manifest={manifest_path}', flush=True)
    print(f"episodes={len(manifest['episodes'])}", flush=True)
if __name__ == '__main__':
    main()
