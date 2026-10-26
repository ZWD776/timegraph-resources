"""Streaming, extraction-free audit helpers for NExT-QA HSGG archives."""
from __future__ import annotations
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
FRAME_NAME_RE = re.compile('^frame_(?P<frame_index>\\d+)_(?P<seconds>\\d+\\.\\d{3})s\\.(?P<extension>png|json)$')
HSGG_MEMBER_RE = re.compile('^(?P<top>hsgg_every30_(?P<half>first|second)_half_default_hsgg)/(?P<local_id>\\d+)_(?P<video_id>[^/]+)/(?P<relative>.+)$')
FRAME_MEMBER_RE = re.compile('^(?P<top>frames_every30_(?P<half>first|second)_half)/(?P<video_id>[^/]+)/(?P<relative>.+)$')
REQUIRED_OBJECT_KEYS = {'image_path', 'boxes', 'labels', 'scores', 'stage2_empty', 'stage3_skipped', 'record_list', 'record_dict'}
REQUIRED_RELATION_KEYS = {'idx', 'subject_label', 'object_label', 'predicate'}
REQUIRED_HIERARCHY_KEYS = {'layer1_nodes', 'layer2_mapping', 'layer3_mapping'}
REQUIRED_VIDEO_FILES = {'run_manifest.json', 'scene_graphs.json', 'objects.json', 'attributes.json', 'relations.json', 'hierarchy_mapping.json'}

def sha256_file(path: Path, chunk_size: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def file_fingerprint(path: Path, *, include_hash: bool=True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if include_hash:
        result['sha256'] = sha256_file(path)
    return result

def parse_frame_filename(name: str) -> tuple[int, float, str]:
    match = FRAME_NAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f'invalid every-30 frame filename: {name}')
    return (int(match.group('frame_index')), float(match.group('seconds')), match.group('extension'))

def _new_hsgg_video(match: re.Match[str]) -> dict[str, Any]:
    return {'official_video_id': match.group('video_id'), 'local_sequential_id': int(match.group('local_id')), 'half': match.group('half'), 'archive_root': f"{match.group('top')}/{match.group('local_id')}_{match.group('video_id')}", 'files_present': [], 'hsgg_frame_names': [], 'frame_key_sets': {}, 'run_manifest': None, 'schema_error_count': 0, 'schema_error_examples': [], 'individual_frame_schema_error_count': 0, 'stale_every60_image_path_count': 0}

def _record_schema_error(video: dict[str, Any], message: str) -> None:
    video['schema_error_count'] += 1
    if len(video['schema_error_examples']) < 20:
        video['schema_error_examples'].append(message)

def _validate_object_record(video: dict[str, Any], frame_name: str, record: Any, *, count_stale_path: bool) -> None:
    if not isinstance(record, dict):
        _record_schema_error(video, f'objects[{frame_name}] is not an object')
        return
    missing = REQUIRED_OBJECT_KEYS - set(record)
    if missing:
        _record_schema_error(video, f'objects[{frame_name}] missing {sorted(missing)}')
    boxes = record.get('boxes')
    labels = record.get('labels')
    scores = record.get('scores')
    if not isinstance(boxes, list) or not isinstance(labels, list) or (not isinstance(scores, list)):
        _record_schema_error(video, f'objects[{frame_name}] boxes/labels/scores type mismatch')
    elif not len(boxes) == len(labels) == len(scores):
        _record_schema_error(video, f'objects[{frame_name}] boxes/labels/scores length mismatch')
    image_path = str(record.get('image_path', ''))
    if count_stale_path and '/frames_every60/' in image_path.replace('\\', '/'):
        video['stale_every60_image_path_count'] += 1

def _validate_objects(video: dict[str, Any], value: Any) -> list[str]:
    if not isinstance(value, dict):
        _record_schema_error(video, 'objects.json is not an object')
        return []
    for frame_name, record in value.items():
        _validate_object_record(video, frame_name, record, count_stale_path=True)
    return list(value)

def _validate_attributes(video: dict[str, Any], value: Any) -> list[str]:
    if not isinstance(value, dict):
        _record_schema_error(video, 'attributes.json is not an object')
        return []
    for frame_name, record in value.items():
        if not isinstance(record, dict):
            _record_schema_error(video, f'attributes[{frame_name}] is not an object')
    return list(value)

def _validate_relations(video: dict[str, Any], value: Any) -> list[str]:
    if not isinstance(value, dict):
        _record_schema_error(video, 'relations.json is not an object')
        return []
    for frame_name, relations in value.items():
        if not isinstance(relations, list):
            _record_schema_error(video, f'relations[{frame_name}] is not a list')
            continue
        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                _record_schema_error(video, f'relations[{frame_name}][{index}] is not an object')
                continue
            missing = REQUIRED_RELATION_KEYS - set(relation)
            if missing:
                _record_schema_error(video, f'relations[{frame_name}][{index}] missing {sorted(missing)}')
    return list(value)

def _validate_hierarchy(video: dict[str, Any], value: Any) -> list[str]:
    if not isinstance(value, dict):
        _record_schema_error(video, 'hierarchy_mapping.json is not an object')
        return []
    for frame_name, record in value.items():
        if not isinstance(record, dict):
            _record_schema_error(video, f'hierarchy[{frame_name}] is not an object')
            continue
        missing = REQUIRED_HIERARCHY_KEYS - set(record)
        if missing:
            _record_schema_error(video, f'hierarchy[{frame_name}] missing {sorted(missing)}')
    return list(value)

def _validate_individual_frame(video: dict[str, Any], frame_name: str, value: Any) -> None:
    """Validate the per-frame combined record used as the authoritative graph source."""
    before = video['schema_error_count']
    if not isinstance(value, dict):
        _record_schema_error(video, f'frame_scene_graphs[{frame_name}] is not an object')
    else:
        required = {'image_path', 'objects', 'hierarchy', 'attributes', 'relations'}
        missing = required - set(value)
        if missing:
            _record_schema_error(video, f'frame_scene_graphs[{frame_name}] missing {sorted(missing)}')
        _validate_object_record(video, frame_name, value.get('objects'), count_stale_path=False)
        hierarchy = value.get('hierarchy')
        if not isinstance(hierarchy, dict):
            _record_schema_error(video, f'hierarchy[{frame_name}] is not an object')
        else:
            hierarchy_missing = REQUIRED_HIERARCHY_KEYS - set(hierarchy)
            if hierarchy_missing:
                _record_schema_error(video, f'hierarchy[{frame_name}] missing {sorted(hierarchy_missing)}')
        if not isinstance(value.get('attributes'), dict):
            _record_schema_error(video, f'attributes[{frame_name}] is not an object')
        relations = value.get('relations')
        if not isinstance(relations, list):
            _record_schema_error(video, f'relations[{frame_name}] is not a list')
        else:
            for index, relation in enumerate(relations):
                if not isinstance(relation, dict):
                    _record_schema_error(video, f'relations[{frame_name}][{index}] is not an object')
                    continue
                relation_missing = REQUIRED_RELATION_KEYS - set(relation)
                if relation_missing:
                    _record_schema_error(video, f'relations[{frame_name}][{index}] missing {sorted(relation_missing)}')
    video['individual_frame_schema_error_count'] += video['schema_error_count'] - before

def _read_json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f'unable to read tar member: {member.name}')
    return json.loads(extracted.read())

def _timestamp_audit(frame_names: Iterable[str]) -> dict[str, Any]:
    parsed: list[tuple[int, float, str]] = []
    invalid: list[str] = []
    for name in frame_names:
        try:
            frame_index, seconds, _ = parse_frame_filename(name)
            parsed.append((frame_index, seconds, name))
        except ValueError:
            invalid.append(name)
    parsed.sort(key=lambda item: (item[0], item[1], item[2]))
    duplicate_frame_indices = sorted((index for index, count in Counter((item[0] for item in parsed)).items() if count > 1))
    non_monotonic_pairs: list[list[str]] = []
    deltas: list[float] = []
    for previous, current in zip(parsed, parsed[1:]):
        deltas.append(current[1] - previous[1])
        if current[1] <= previous[1] or current[0] <= previous[0]:
            non_monotonic_pairs.append([previous[2], current[2]])
    return {'count': len(parsed), 'invalid_filenames': invalid, 'duplicate_frame_indices': duplicate_frame_indices, 'non_monotonic_pairs': non_monotonic_pairs[:20], 'monotonic': not invalid and (not duplicate_frame_indices) and (not non_monotonic_pairs), 'start_timestamp_seconds': parsed[0][1] if parsed else None, 'end_timestamp_seconds': parsed[-1][1] if parsed else None, 'timestamp_delta_seconds': {'min': min(deltas) if deltas else None, 'max': max(deltas) if deltas else None, 'mean': sum(deltas) / len(deltas) if deltas else None}}

def scan_hsgg_archive(path: Path) -> dict[str, Any]:
    """Scan all HSGG members once, without extracting the archive."""
    roots: dict[str, dict[str, Any]] = {}
    invalid_members: list[str] = []
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized_name = member.name.replace('\\', '/').lstrip('./')
            match = HSGG_MEMBER_RE.fullmatch(normalized_name)
            if match is None:
                if len(invalid_members) < 100:
                    invalid_members.append(normalized_name)
                continue
            root = f"{match.group('top')}/{match.group('local_id')}_{match.group('video_id')}"
            video = roots.setdefault(root, _new_hsgg_video(match))
            relative = match.group('relative')
            if '/' not in relative:
                video['files_present'].append(relative)
            if relative.startswith('frame_scene_graphs/') and relative.endswith('.json'):
                frame_filename = Path(relative).name
                video['hsgg_frame_names'].append(frame_filename)
                try:
                    value = _read_json_member(archive, member)
                    _validate_individual_frame(video, Path(frame_filename).stem, value)
                except Exception as exc:
                    _record_schema_error(video, f'frame_scene_graphs/{frame_filename} parse error: {exc}')
                    video['individual_frame_schema_error_count'] += 1
                continue
            if relative == 'run_manifest.json':
                try:
                    manifest = _read_json_member(archive, member)
                    frames = manifest.get('frames', {}) if isinstance(manifest, dict) else {}
                    video['run_manifest'] = {'total_images': manifest.get('total_images'), 'completed_frames': manifest.get('completed_frames'), 'skipped_no_scene_graph_frames': manifest.get('skipped_no_scene_graph_frames'), 'error_frames': manifest.get('error_frames'), 'frame_names': list(frames) if isinstance(frames, dict) else []}
                except Exception as exc:
                    _record_schema_error(video, f'run_manifest.json parse error: {exc}')
                continue
            validators = {'objects.json': _validate_objects, 'attributes.json': _validate_attributes, 'relations.json': _validate_relations, 'hierarchy_mapping.json': _validate_hierarchy}
            validator = validators.get(relative)
            if validator is not None:
                try:
                    value = _read_json_member(archive, member)
                    video['frame_key_sets'][relative] = validator(video, value)
                except Exception as exc:
                    _record_schema_error(video, f'{relative} parse error: {exc}')
    videos: list[dict[str, Any]] = []
    for video in roots.values():
        video['files_present'] = sorted(set(video['files_present']))
        video['hsgg_frame_names'] = sorted(set(video['hsgg_frame_names']))
        video['timestamp_audit'] = _timestamp_audit(video['hsgg_frame_names'])
        expected_frame_keys = {Path(name).stem for name in video['hsgg_frame_names']}
        frame_key_mismatches: dict[str, Any] = {}
        for filename, keys in video['frame_key_sets'].items():
            actual = set(keys)
            if actual != expected_frame_keys:
                frame_key_mismatches[filename] = {'missing': sorted(expected_frame_keys - actual)[:20], 'extra': sorted(actual - expected_frame_keys)[:20], 'missing_count': len(expected_frame_keys - actual), 'extra_count': len(actual - expected_frame_keys)}
        video['frame_key_mismatches'] = frame_key_mismatches
        missing_files = REQUIRED_VIDEO_FILES - set(video['files_present'])
        video['missing_required_files'] = sorted(missing_files)
        if missing_files:
            _record_schema_error(video, f'missing video files: {sorted(missing_files)}')
        videos.append(video)
    videos.sort(key=lambda item: item['local_sequential_id'])
    return {'inventory_schema_version': 2, 'archive': file_fingerprint(path), 'scan_mode': 'streaming_no_extraction', 'invalid_member_examples': invalid_members, 'videos': videos}

def scan_frame_archive(path: Path) -> dict[str, Any]:
    """Inventory all sampled PNGs once, without extracting image bytes."""
    roots: dict[str, dict[str, Any]] = {}
    invalid_members: list[str] = []
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized_name = member.name.replace('\\', '/').lstrip('./')
            match = FRAME_MEMBER_RE.fullmatch(normalized_name)
            if match is None:
                if len(invalid_members) < 100:
                    invalid_members.append(normalized_name)
                continue
            relative = match.group('relative')
            if '/' in relative or not relative.endswith('.png'):
                if len(invalid_members) < 100:
                    invalid_members.append(normalized_name)
                continue
            root = f"{match.group('top')}/{match.group('video_id')}"
            video = roots.setdefault(root, {'official_video_id': match.group('video_id'), 'half': match.group('half'), 'archive_root': root, 'frame_names': []})
            video['frame_names'].append(relative)
    videos: list[dict[str, Any]] = []
    for video in roots.values():
        video['frame_names'] = sorted(set(video['frame_names']))
        video['timestamp_audit'] = _timestamp_audit(video['frame_names'])
        videos.append(video)
    videos.sort(key=lambda item: (item['half'], item['official_video_id']))
    return {'inventory_schema_version': 1, 'archive': file_fingerprint(path), 'scan_mode': 'streaming_no_extraction', 'invalid_member_examples': invalid_members, 'videos': videos}

def load_or_scan_inventory(archive_path: Path, cache_path: Path, scanner: Any, *, reuse_cache: bool) -> tuple[dict[str, Any], bool]:
    """Reuse an exact size/mtime cache or execute the supplied scanner."""
    stat = archive_path.stat()
    if reuse_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        fingerprint = cached.get('archive', {})
        required_schema_version = 2 if scanner is scan_hsgg_archive else 1
        if cached.get('inventory_schema_version') == required_schema_version and fingerprint.get('bytes') == stat.st_size and (fingerprint.get('mtime_ns') == stat.st_mtime_ns):
            return (cached, True)
    inventory = scanner(archive_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return (inventory, False)
