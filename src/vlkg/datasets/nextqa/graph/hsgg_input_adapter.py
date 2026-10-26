"""Stream authoritative NExT-QA HSGG frame records into graph-core tables."""
from __future__ import annotations
import hashlib
import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from vlkg.datasets.nextqa.audit.archive_audit import parse_frame_filename
from vlkg.datasets.nextqa.runtime.contracts import graph_safe_video_record
from vlkg.datasets.nextqa.runtime.io_utils import atomic_copy_binary_stream, atomic_write_json
from vlkg.workspace import workspace_root
ALLOWED_FRAME_STATUSES = {'completed', 'skipped_no_scene_graph'}

@dataclass(frozen=True)
class NormalizedVideoInput:
    official_video_id: str
    episode_key: str
    local_sequential_id: int
    input_dir: Path
    timeline_path: Path
    source_trace_path: Path
    materialized_frame_root: Path | None
    materialized_frame_count: int
    completed_frame_count: int
    skipped_frame_count: int
    sampled_frame_count: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float

def resolve_normalized_input_paths(video: Mapping[str, Any], *, repository_root: str | Path | None=None) -> tuple[Path, Path]:
    """Resolve selected materialized inputs without consulting historical archives."""
    root = Path(repository_root).resolve() if repository_root is not None else workspace_root()
    paths: list[Path] = []
    for field in ('normalized_input_dir', 'frames_dir'):
        value = video.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'video manifest row is missing {field}')
        path = Path(value)
        paths.append((path if path.is_absolute() else root / path).resolve())
    return (paths[0], paths[1])

def load_normalized_video_input(video: Mapping[str, Any], *, repository_root: str | Path | None=None, verify_frame_hashes: bool=False) -> NormalizedVideoInput:
    """Validate and reuse one extracted input, without rewriting tables or PNGs.

    Archive paths and embedded image paths remain provenance only. Graph consumers
    receive the selected actual frame root, so relocating inputs needs no rewrites.
    """
    safe = graph_safe_video_record(video)
    input_dir, frame_root = resolve_normalized_input_paths(safe, repository_root=repository_root)
    portable_image_paths = safe.get('image_path_base') == 'input_directory'
    if portable_image_paths and frame_root != input_dir / 'frames':
        raise ValueError(f'portable normalized frames_dir must be input_directory/frames: {input_dir}')
    tables: dict[str, dict[str, Any]] = {}
    for name in ('objects', 'attributes', 'relations', 'hierarchy_mapping'):
        value = json.loads((input_dir / f'{name}.json').read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError(f'normalized {name} table must be an object: {input_dir}')
        tables[name] = value
    timeline_path, trace_path = input_dir / 'timeline.json', input_dir / 'source_trace.json'
    timeline = json.loads(timeline_path.read_text(encoding='utf-8-sig'))
    trace = json.loads(trace_path.read_text(encoding='utf-8-sig'))
    if not isinstance(timeline, dict) or timeline.get('schema_version') != 'nextqa_hsgg_timeline_v1':
        raise ValueError(f'unsupported normalized timeline schema: {input_dir}')
    if not isinstance(trace, dict) or trace.get('schema_version') != 'nextqa_hsgg_source_trace_v1':
        raise ValueError(f'unsupported normalized source trace schema: {input_dir}')
    video_id = str(safe['official_video_id'])
    if str(timeline.get('official_video_id')) != video_id or str(trace.get('official_video_id')) != video_id:
        raise ValueError(f'normalized input video identity mismatch: {input_dir}')
    for field in ('episode_key', 'vidor_id', 'hsgg_archive_root', 'frame_archive_root'):
        if field in safe and str(trace.get(field)) != str(safe[field]):
            raise ValueError(f'normalized source provenance mismatch for {field}: {input_dir}')
    if timeline.get('timestamp_source') != 'frame_filename_fps_derived_nominal_seconds' or timeline.get('timestamp_is_ordinal') is not False:
        raise ValueError(f'normalized timeline must retain filename timestamps: {input_dir}')
    provenance = {'authoritative_record_source': 'frame_scene_graphs/*.json', 'aggregate_json_read': False, 'embedded_hsgg_image_paths_used': False, 'frames_materialized': True}
    for field, expected in provenance.items():
        if trace.get(field) != expected:
            raise ValueError(f'normalized source trace has invalid {field}: {input_dir}')
    frames, source_frames = timeline.get('frames'), trace.get('sampled_frames')
    if not isinstance(frames, list) or not frames or not isinstance(source_frames, list):
        raise ValueError(f'normalized timeline/source trace must contain sampled frames: {input_dir}')
    if not all(isinstance(frame, dict) for frame in frames + source_frames):
        raise ValueError(f'invalid normalized frame metadata: {input_dir}')
    names = [frame.get('frame_name') for frame in frames]
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError(f'normalized timeline frame names must be unique: {input_dir}')
    name_set = set(names)
    if any(set(table) != name_set for table in tables.values()):
        raise ValueError(f'normalized table frame keys differ from timeline: {input_dir}')
    source_by_name = {frame.get('frame_name'): frame for frame in source_frames}
    if len(source_by_name) != len(source_frames) or set(source_by_name) != name_set:
        raise ValueError(f'normalized source trace frame keys differ from timeline: {input_dir}')
    completed_count, skipped_count = 0, 0
    previous_index, previous_time = -1, -1.0
    for rank, frame in enumerate(frames):
        name = frame['frame_name']
        filename = f'{name}.png'
        index, timestamp, _ = parse_frame_filename(filename)
        if frame.get('frame_filename') != filename or frame.get('frame_rank') != rank or frame.get('frame_index') != index or frame.get('timestamp_sec') != timestamp:
            raise ValueError(f'normalized timeline disagrees with frame filename: {name}')
        if not math.isfinite(timestamp) or index <= previous_index or timestamp <= previous_time:
            raise ValueError(f'normalized timeline is not strictly monotonic: {name}')
        previous_index, previous_time = index, timestamp
        status = frame.get('hsgg_status')
        if status not in ALLOWED_FRAME_STATUSES or frame.get('has_scene_graph') is not (status == 'completed'):
            raise ValueError(f'normalized frame has invalid HSGG status: {name}')
        objects, attributes, relations, hierarchy = (tables[key][name] for key in ('objects', 'attributes', 'relations', 'hierarchy_mapping'))
        _normalize_completed_record({'objects': objects, 'attributes': attributes, 'relations': relations, 'hierarchy': hierarchy}, image_reference=str(frame_root / filename), member_name=name)
        source = source_by_name[name]
        if portable_image_paths:
            expected_image_path = f'frames/{filename}'
            if objects.get('image_path') != expected_image_path or source.get('materialized_image_path') != expected_image_path:
                raise ValueError(f'normalized portable image_path mismatch: {name}')
        if source.get('embedded_image_path_ignored') is not True:
            raise ValueError(f'normalized frame provenance uses stale image paths: {name}')
        frame_member = str(source.get('frame_archive_member', ''))
        expected_frame_member = str(safe['frame_member_template']).format(frame_filename=filename) if 'frame_member_template' in safe else f"{trace['frame_archive_root']}/{filename}"
        if _normalized_member_name(frame_member) != _normalized_member_name(expected_frame_member):
            raise ValueError(f'normalized frame member provenance mismatch: {name}')
        if not re.fullmatch('[0-9a-f]{64}', str(source.get('frame_member_sha256', ''))):
            raise ValueError(f'normalized frame lacks SHA256 provenance: {name}')
        if status == 'completed':
            completed_count += 1
            expected_member = f"{trace['hsgg_archive_root']}/frame_scene_graphs/{name}.json"
            if source.get('hsgg_archive_member') != expected_member or not re.fullmatch('[0-9a-f]{64}', str(source.get('hsgg_member_sha256', ''))):
                raise ValueError(f'normalized HSGG provenance mismatch: {name}')
        else:
            skipped_count += 1
            if objects['boxes'] or attributes or relations or any(hierarchy.get(key) for key in ('layer1_nodes', 'layer2_mapping', 'layer3_mapping')) or source.get('hsgg_archive_member') is not None or source.get('hsgg_member_sha256') is not None:
                raise ValueError(f'normalized skipped frame contains scene graph data: {name}')
        image_path = frame_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.stat().st_size != source.get('frame_member_bytes'):
            raise ValueError(f'normalized frame byte count mismatch: {image_path}')
        with image_path.open('rb') as image_handle:
            if image_handle.read(8) != b'\x89PNG\r\n\x1a\n':
                raise ValueError(f'normalized frame is not PNG: {image_path}')
        if verify_frame_hashes and _sha256_file(image_path) != source['frame_member_sha256']:
            raise ValueError(f'normalized frame SHA256 mismatch: {image_path}')
    counts = {'sampled_frame_count': len(frames), 'completed_frame_count': completed_count, 'skipped_frame_count': skipped_count}
    for field, expected in counts.items():
        if timeline.get(field) != expected:
            raise ValueError(f'normalized timeline count mismatch for {field}: {input_dir}')
    manifest_counts = {'sampled_image_count': len(frames), 'hsgg_frame_count': completed_count, 'skipped_frame_count': skipped_count}
    for field, expected in manifest_counts.items():
        if int(safe[field]) != expected:
            raise ValueError(f'normalized manifest count mismatch for {field}: {input_dir}')
    if trace.get('materialized_frame_count') != len(frames):
        raise ValueError(f'normalized trace materialized count mismatch: {input_dir}')
    for field, actual in (('start_timestamp_seconds', float(frames[0]['timestamp_sec'])), ('end_timestamp_seconds', float(frames[-1]['timestamp_sec']))):
        if float(safe[field]) != actual:
            raise ValueError(f'normalized manifest timestamp mismatch for {field}: {input_dir}')
    return NormalizedVideoInput(official_video_id=video_id, episode_key=str(safe['episode_key']), local_sequential_id=int(safe['local_sequential_id']), input_dir=input_dir, timeline_path=timeline_path, source_trace_path=trace_path, materialized_frame_root=frame_root, materialized_frame_count=len(frames), completed_frame_count=completed_count, skipped_frame_count=skipped_count, sampled_frame_count=len(frames), start_timestamp_seconds=float(frames[0]['timestamp_sec']), end_timestamp_seconds=float(frames[-1]['timestamp_sec']))

def _read_json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[Any, str]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f'unable to read tar member: {member.name}')
    raw = extracted.read()
    return (json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest())

def _archive_frame_reference(video: Mapping[str, Any], frame_filename: str) -> tuple[str, str]:
    member = str(video['frame_member_template']).format(frame_filename=frame_filename)
    archive_path = str(Path(str(video['frame_archive'])).resolve())
    return (f'{archive_path}::{member}', member)

def _normalized_member_name(value: str) -> str:
    return value.replace('\\', '/').lstrip('./')

def _sha256_file(path: Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _materialize_selected_frames(videos: list[Mapping[str, Any]], *, run_manifests: Mapping[str, Mapping[str, Any]], output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    requests_by_archive: dict[Path, dict[str, dict[str, Any]]] = {}
    for video in videos:
        archive_root = str(video['hsgg_archive_root']).strip('/')
        run_manifest = run_manifests.get(archive_root)
        if run_manifest is None:
            raise ValueError(f'missing run_manifest.json below {archive_root}')
        frames = run_manifest.get('frames')
        if not isinstance(frames, dict):
            raise ValueError(f'run manifest frames is not an object: {archive_root}')
        archive_path = Path(str(video['frame_archive'])).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_requests = requests_by_archive.setdefault(archive_path, {})
        video_id = str(video['official_video_id'])
        video_dir = output_root / f"{int(video['local_sequential_id']):04d}_{video_id}"
        for frame_stem in frames:
            frame_filename = f'{frame_stem}.png'
            _, raw_member = _archive_frame_reference(video, frame_filename)
            member_name = _normalized_member_name(raw_member)
            if member_name in archive_requests:
                raise ValueError(f'duplicate requested frame archive member: {member_name}')
            archive_requests[member_name] = {'official_video_id': video_id, 'frame_name': frame_stem, 'target_path': video_dir / 'frames' / frame_filename}
    materialized: dict[tuple[str, str], dict[str, Any]] = {}
    for archive_path, requests in sorted(requests_by_archive.items(), key=lambda item: str(item[0]).lower()):
        found: set[str] = set()
        with tarfile.open(archive_path, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_name = _normalized_member_name(member.name)
                request = requests.get(member_name)
                if request is None:
                    continue
                if member_name in found:
                    raise ValueError(f'duplicate frame archive member: {member_name}')
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f'unable to read frame archive member: {member_name}')
                target_path = atomic_copy_binary_stream(request['target_path'], extracted)
                found.add(member_name)
                materialized[request['official_video_id'], request['frame_name']] = {'frame_archive_member': member_name, 'frame_member_sha256': _sha256_file(target_path), 'frame_member_bytes': target_path.stat().st_size, 'materialized_image_path': str(target_path)}
        missing = sorted(set(requests) - found)
        if missing:
            raise ValueError(f'frame archive is missing {len(missing)} requested members: {missing[:5]}')
    return materialized

def _empty_object_record(image_reference: str) -> dict[str, Any]:
    return {'image_path': image_reference, 'boxes': [], 'labels': [], 'scores': [], 'stage2_empty': True, 'stage3_skipped': True, 'record_list': [], 'record_dict': []}

def _normalize_completed_record(raw: Any, *, image_reference: str, member_name: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(f'HSGG frame member is not an object: {member_name}')
    required = {'objects', 'attributes', 'relations', 'hierarchy'}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f'HSGG frame member {member_name} is missing {missing}')
    objects = raw['objects']
    if not isinstance(objects, dict):
        raise ValueError(f'objects is not an object: {member_name}')
    for key in ('boxes', 'labels', 'scores'):
        if not isinstance(objects.get(key), list):
            raise ValueError(f'objects.{key} is not a list: {member_name}')
    if not len(objects['boxes']) == len(objects['labels']) == len(objects['scores']):
        raise ValueError(f'objects boxes/labels/scores length mismatch: {member_name}')
    attributes = raw['attributes']
    relations = raw['relations']
    hierarchy = raw['hierarchy']
    if not isinstance(attributes, dict):
        raise ValueError(f'attributes is not an object: {member_name}')
    if not isinstance(relations, list):
        raise ValueError(f'relations is not a list: {member_name}')
    if not isinstance(hierarchy, dict):
        raise ValueError(f'hierarchy is not an object: {member_name}')
    normalized_objects = dict(objects)
    normalized_objects['image_path'] = image_reference
    return (normalized_objects, dict(attributes), list(relations), dict(hierarchy))

def normalize_hsgg_archive(videos: Iterable[Mapping[str, Any]], *, output_root: str | Path, materialize_frames: bool=True) -> list[NormalizedVideoInput]:
    """Read one HSGG archive pass and persist deterministic per-video inputs."""
    safe_videos = [graph_safe_video_record(video) for video in videos]
    if not safe_videos:
        raise ValueError('at least one video is required')
    archive_paths = {str(Path(str(video['hsgg_archive'])).resolve()) for video in safe_videos}
    if len(archive_paths) != 1:
        raise ValueError('a normalization batch must use one HSGG archive')
    archive_path = Path(next(iter(archive_paths)))
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    by_root = {str(video['hsgg_archive_root']).strip('/'): video for video in safe_videos}
    run_manifests: dict[str, dict[str, Any]] = {}
    frame_records: dict[str, dict[str, tuple[Any, str, str]]] = {root: {} for root in by_root}
    with tarfile.open(archive_path, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = _normalized_member_name(member.name)
            matching_root = next((root for root in by_root if member_name.startswith(f'{root}/')), None)
            if matching_root is None:
                continue
            relative = member_name[len(matching_root) + 1:]
            if relative == 'run_manifest.json':
                value, _ = _read_json_member(archive, member)
                if not isinstance(value, dict):
                    raise ValueError(f'run manifest is not an object: {member_name}')
                run_manifests[matching_root] = value
            elif relative.startswith('frame_scene_graphs/') and relative.endswith('.json'):
                value, digest = _read_json_member(archive, member)
                frame_stem = Path(relative).stem
                if frame_stem in frame_records[matching_root]:
                    raise ValueError(f'duplicate HSGG frame member: {member_name}')
                frame_records[matching_root][frame_stem] = (value, digest, member_name)
    root_path = Path(output_root).resolve()
    materialized_frames = _materialize_selected_frames(safe_videos, run_manifests=run_manifests, output_root=root_path) if materialize_frames else {}
    normalized: list[NormalizedVideoInput] = []
    for archive_root, video in sorted(by_root.items(), key=lambda item: int(item[1]['local_sequential_id'])):
        run_manifest = run_manifests.get(archive_root)
        if run_manifest is None:
            raise ValueError(f'missing run_manifest.json below {archive_root}')
        frames = run_manifest.get('frames')
        if not isinstance(frames, dict):
            raise ValueError(f'run manifest frames is not an object: {archive_root}')
        objects_table: dict[str, Any] = {}
        attributes_table: dict[str, Any] = {}
        relations_table: dict[str, Any] = {}
        hierarchy_table: dict[str, Any] = {}
        timeline: list[dict[str, Any]] = []
        source_frames: list[dict[str, Any]] = []
        completed_names: set[str] = set()
        skipped_names: set[str] = set()
        parsed_frames: list[tuple[int, float, str, Mapping[str, Any]]] = []
        for frame_stem, metadata in frames.items():
            if not isinstance(metadata, dict):
                raise ValueError(f'invalid run manifest frame metadata: {frame_stem}')
            frame_index, timestamp, _ = parse_frame_filename(f'{frame_stem}.png')
            parsed_frames.append((frame_index, timestamp, frame_stem, metadata))
        parsed_frames.sort(key=lambda item: (item[0], item[1], item[2]))
        previous_index: int | None = None
        previous_timestamp: float | None = None
        for frame_rank, (frame_index, timestamp, frame_stem, metadata) in enumerate(parsed_frames):
            if previous_index is not None and frame_index <= previous_index:
                raise ValueError(f'non-monotonic frame index below {archive_root}')
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f'non-monotonic timestamp below {archive_root}')
            previous_index = frame_index
            previous_timestamp = timestamp
            status = str(metadata.get('status', ''))
            if status not in ALLOWED_FRAME_STATUSES:
                raise ValueError(f'unsupported frame status {status!r}: {archive_root}/{frame_stem}')
            frame_filename = f'{frame_stem}.png'
            archive_reference, frame_member = _archive_frame_reference(video, frame_filename)
            frame_source = materialized_frames.get((str(video['official_video_id']), frame_stem))
            if materialize_frames and frame_source is None:
                raise ValueError(f'materialized frame lookup missing: {frame_member}')
            image_reference = str(frame_source['materialized_image_path']) if frame_source is not None else archive_reference
            hsgg_member: str | None = None
            source_sha256: str | None = None
            if status == 'completed':
                record = frame_records[archive_root].get(frame_stem)
                if record is None:
                    raise ValueError(f'completed frame lacks per-frame HSGG JSON: {frame_stem}')
                raw, source_sha256, hsgg_member = record
                objects, attributes, relations, hierarchy = _normalize_completed_record(raw, image_reference=image_reference, member_name=hsgg_member)
                completed_names.add(frame_stem)
            else:
                if frame_stem in frame_records[archive_root]:
                    raise ValueError(f'skipped frame unexpectedly has HSGG JSON: {frame_stem}')
                objects = _empty_object_record(image_reference)
                attributes = {}
                relations = []
                hierarchy = {'layer1_nodes': [], 'layer2_mapping': {}, 'layer3_mapping': {}}
                skipped_names.add(frame_stem)
            objects_table[frame_stem] = objects
            attributes_table[frame_stem] = attributes
            relations_table[frame_stem] = relations
            hierarchy_table[frame_stem] = hierarchy
            timeline.append({'frame_rank': frame_rank, 'frame_name': frame_stem, 'frame_filename': frame_filename, 'frame_index': frame_index, 'timestamp_sec': timestamp, 'hsgg_status': status, 'has_scene_graph': status == 'completed'})
            source_frames.append({'frame_name': frame_stem, 'frame_archive_member': frame_member, 'frame_member_sha256': frame_source['frame_member_sha256'] if frame_source is not None else None, 'frame_member_bytes': frame_source['frame_member_bytes'] if frame_source is not None else None, 'materialized_image_path': frame_source['materialized_image_path'] if frame_source is not None else None, 'hsgg_archive_member': hsgg_member, 'hsgg_member_sha256': source_sha256, 'embedded_image_path_ignored': True})
        unexpected = sorted(set(frame_records[archive_root]) - completed_names)
        if unexpected:
            raise ValueError(f'HSGG members absent from run manifest completion set: {unexpected[:5]}')
        expected_counts = {'total_images': len(parsed_frames), 'completed_frames': len(completed_names), 'skipped_no_scene_graph_frames': len(skipped_names), 'error_frames': 0}
        for key, expected in expected_counts.items():
            if int(run_manifest.get(key, -1)) != expected:
                raise ValueError(f'run manifest count mismatch for {archive_root}: {key}={run_manifest.get(key)!r}, expected {expected}')
        manifest_counts = {'hsgg_frame_count': len(completed_names), 'sampled_image_count': len(parsed_frames), 'skipped_frame_count': len(skipped_names)}
        for key, actual in manifest_counts.items():
            if int(video[key]) != actual:
                raise ValueError(f"pilot manifest count mismatch for {video['official_video_id']}: {key}={video[key]!r}, actual {actual}")
        video_dir = root_path / f"{int(video['local_sequential_id']):04d}_{video['official_video_id']}"
        atomic_write_json(video_dir / 'objects.json', objects_table)
        atomic_write_json(video_dir / 'attributes.json', attributes_table)
        atomic_write_json(video_dir / 'relations.json', relations_table)
        atomic_write_json(video_dir / 'hierarchy_mapping.json', hierarchy_table)
        timeline_path = atomic_write_json(video_dir / 'timeline.json', {'schema_version': 'nextqa_hsgg_timeline_v1', 'official_video_id': str(video['official_video_id']), 'timestamp_source': 'frame_filename_fps_derived_nominal_seconds', 'timestamp_is_ordinal': False, 'sampled_frame_count': len(timeline), 'completed_frame_count': len(completed_names), 'skipped_frame_count': len(skipped_names), 'frames': timeline})
        source_trace_path = atomic_write_json(video_dir / 'source_trace.json', {'schema_version': 'nextqa_hsgg_source_trace_v1', 'official_video_id': str(video['official_video_id']), 'episode_key': str(video['episode_key']), 'vidor_id': str(video['vidor_id']), 'hsgg_archive': str(archive_path), 'hsgg_archive_root': archive_root, 'frame_archive': str(Path(str(video['frame_archive'])).resolve()), 'frame_archive_root': str(video['frame_archive_root']), 'frames_materialized': materialize_frames, 'materialized_frame_count': len(parsed_frames) if materialize_frames else 0, 'authoritative_record_source': 'frame_scene_graphs/*.json', 'aggregate_json_read': False, 'embedded_hsgg_image_paths_used': False, 'sampled_frames': source_frames})
        normalized.append(NormalizedVideoInput(official_video_id=str(video['official_video_id']), episode_key=str(video['episode_key']), local_sequential_id=int(video['local_sequential_id']), input_dir=video_dir, timeline_path=timeline_path, source_trace_path=source_trace_path, materialized_frame_root=video_dir / 'frames' if materialize_frames else None, materialized_frame_count=len(parsed_frames) if materialize_frames else 0, completed_frame_count=len(completed_names), skipped_frame_count=len(skipped_names), sampled_frame_count=len(parsed_frames), start_timestamp_seconds=float(timeline[0]['timestamp_sec']), end_timestamp_seconds=float(timeline[-1]['timestamp_sec'])))
    return normalized
