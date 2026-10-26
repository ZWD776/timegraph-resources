"""Contracts for the isolated odd-height and low-FPS NExT-QA repair run."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from vlkg.datasets.nextqa.runtime.contracts import DATASET, SPLIT, graph_safe_video_record, validate_manifest_input_storage
SPECIAL_PATCH_VIDEO_COUNT = 29
SPECIAL_PATCH_QUESTION_COUNT = 241
SPECIAL_PATCH_EXPECTED_COUNTS = {'videos': 29, 'sampled_frames': 1228, 'completed_hsgg_frames': 1217, 'skipped_frames': 11}
SPECIAL_PATCH_SOURCE_COUNTS = {'oddheight': 21, 'lowfps': 8}
NORMAL_FPS_UNCHANGED_IDS = ('10082798964', '11244289523', '11276567104', '11541421563', '11849091804', '14003541505', '14012319794', '2535384528', '2559262403', '2584172238', '2679130168', '2809330695')
FORBIDDEN_KEYS = {'answer', 'answer_choices', 'gold_choice_index', 'question', 'question_count', 'question_id', 'question_ids'}

def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).lower())
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found

def load_special_graph_manifest(path: str | Path, *, verify_files: bool=True) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('dataset') != DATASET or payload.get('split') != SPLIT:
        raise ValueError('special patch manifest must describe NExT-QA validation')
    if payload.get('scope') != 'isolated_oddheight_and_actual_lowfps_patch':
        raise ValueError('unexpected special patch scope')
    if payload.get('hsgg_record_source') != 'frame_scene_graphs_members':
        raise ValueError('per-frame HSGG records must be authoritative')
    if payload.get('aggregate_json_allowed_as_build_source') is not False:
        raise ValueError('aggregate HSGG JSON must remain forbidden')
    if payload.get('uses_uniform32') is not False:
        raise ValueError('special patch graph construction must not use uniform-32')
    if payload.get('uses_hsgg_image_path_metadata') is not False:
        raise ValueError('embedded HSGG image paths must remain disabled')
    forbidden = sorted(_keys(payload).intersection(FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError(f'special graph manifest contains forbidden keys: {forbidden}')
    videos = payload.get('videos')
    if not isinstance(videos, list) or len(videos) != SPECIAL_PATCH_VIDEO_COUNT:
        raise ValueError('special patch manifest must contain exactly 29 videos')
    if int(payload.get('video_count', -1)) != len(videos):
        raise ValueError('special patch video_count mismatch')
    validate_manifest_input_storage(payload)
    safe_rows = [graph_safe_video_record(video) for video in videos]
    video_ids = [str(row['official_video_id']) for row in safe_rows]
    local_ids = [int(row['local_sequential_id']) for row in safe_rows]
    if len(video_ids) != len(set(video_ids)) or len(local_ids) != len(set(local_ids)):
        raise ValueError('special patch contains duplicate video or local IDs')
    source_counts = Counter((str(video.get('special_patch_source_kind')) for video in videos))
    if dict(source_counts) != SPECIAL_PATCH_SOURCE_COUNTS:
        raise ValueError(f'unexpected special patch source counts: {dict(source_counts)}')
    actual_counts = {'videos': len(videos), 'sampled_frames': sum((int(row['sampled_image_count']) for row in safe_rows)), 'completed_hsgg_frames': sum((int(row['hsgg_frame_count']) for row in safe_rows)), 'skipped_frames': sum((int(row['skipped_frame_count']) for row in safe_rows))}
    if actual_counts != SPECIAL_PATCH_EXPECTED_COUNTS:
        raise ValueError(f'special patch count mismatch: {actual_counts}')
    if verify_files:
        directory_rows = [row for row in safe_rows if row.get('input_storage') == 'normalized_directory']
        if directory_rows:
            from vlkg.datasets.nextqa.graph.hsgg_input_adapter import load_normalized_video_input
            for row in directory_rows:
                load_normalized_video_input(row)
        archive_rows = [row for row in safe_rows if row.get('input_storage', 'archive') == 'archive']
        if archive_rows:
            hsgg_paths = {Path(str(row['hsgg_archive'])).resolve() for row in archive_rows}
            if len(hsgg_paths) != 1 or not next(iter(hsgg_paths)).is_file():
                raise FileNotFoundError('special patch HSGG archive is missing or inconsistent')
            frame_paths = {Path(str(row['frame_archive'])).resolve() for row in archive_rows}
            missing = sorted((str(item) for item in frame_paths if not item.is_file()))
            if missing:
                raise FileNotFoundError(f'special patch frame archives are missing: {missing}')
    return payload

def special_patch_video_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return the validated 29-video replacement set in manifest order."""
    videos = manifest.get('videos')
    if not isinstance(videos, list) or len(videos) != SPECIAL_PATCH_VIDEO_COUNT:
        raise ValueError('special patch manifest must contain exactly 29 videos')
    video_ids = tuple((str(row['official_video_id']) for row in videos))
    if len(set(video_ids)) != SPECIAL_PATCH_VIDEO_COUNT:
        raise ValueError('special patch manifest contains duplicate video IDs')
    if set(video_ids).intersection(NORMAL_FPS_UNCHANGED_IDS):
        raise ValueError('normal-FPS unchanged videos leaked into the replacement set')
    return video_ids

def select_special_patch_questions(rows: list[dict[str, Any]], *, target_video_ids: set[str], expected_question_count: int=SPECIAL_PATCH_QUESTION_COUNT) -> list[dict[str, Any]]:
    """Select the complete repair question subset without consulting labels."""
    selected = [row for row in rows if str(row.get('video_id')) in target_video_ids]
    question_ids = [str(row.get('question_id') or '') for row in selected]
    selected_video_ids = {str(row.get('video_id')) for row in selected}
    if len(selected) != expected_question_count:
        raise ValueError(f'special patch question count mismatch: {len(selected)} != {expected_question_count}')
    if not all(question_ids) or len(question_ids) != len(set(question_ids)):
        raise ValueError('special patch questions contain empty or duplicate IDs')
    if selected_video_ids != target_video_ids:
        raise ValueError('special patch questions do not cover exactly the target videos')
    return selected

def build_graph_overlay_manifest(base: dict[str, Any], replacement: dict[str, Any], *, target_video_ids: set[str], run_id: str, run_root: str, expected_base_video_count: int=570) -> dict[str, Any]:
    """Overlay replacement graph rows while retaining every unaffected base row."""
    base_rows = base.get('videos')
    replacement_rows = replacement.get('videos')
    if not isinstance(base_rows, list) or not isinstance(replacement_rows, list):
        raise ValueError('graph manifests must contain video lists')
    if len(base_rows) != expected_base_video_count:
        raise ValueError('base graph manifest video count mismatch')
    if len(replacement_rows) != len(target_video_ids):
        raise ValueError('replacement graph manifest video count mismatch')
    if base.get('completed_video_count') != expected_base_video_count:
        raise ValueError('base graph manifest is incomplete')
    if base.get('error_video_count') != 0 or base.get('uses_gold_answer') is not False:
        raise ValueError('base graph manifest failed completion or purity checks')
    if replacement.get('completed_video_count') != len(target_video_ids):
        raise ValueError('replacement graph manifest is incomplete')
    if replacement.get('error_video_count') != 0 or replacement.get('uses_gold_answer') is not False:
        raise ValueError('replacement graph manifest failed completion or purity checks')
    base_by_id = {str(row['official_video_id']): row for row in base_rows}
    replacement_by_id = {str(row['official_video_id']): row for row in replacement_rows}
    if len(base_by_id) != len(base_rows):
        raise ValueError('base graph manifest contains duplicate video IDs')
    if set(replacement_by_id) != target_video_ids:
        raise ValueError('replacement graph manifest IDs do not match the repair scope')
    if not target_video_ids.issubset(base_by_id):
        raise ValueError('repair scope is not contained in the base graph manifest')
    merged_rows = [replacement_by_id.get(str(row['official_video_id']), row) for row in base_rows]
    merged = dict(base)
    merged.update({'schema_version': 'nextqa_graph_overlay_manifest_v1', 'run_id': run_id, 'run_root': run_root, 'video_count': expected_base_video_count, 'completed_video_count': expected_base_video_count, 'error_video_count': 0, 'uses_gold_answer': False, 'overlay_policy': 'replace_exact_video_ids_only_v1', 'replaced_video_count': len(target_video_ids), 'retained_base_video_count': expected_base_video_count - len(target_video_ids), 'replaced_video_ids': sorted(target_video_ids), 'videos': merged_rows})
    return merged

def overlay_question_records(base_rows: list[dict[str, Any]], replacement_rows: list[dict[str, Any]], *, target_question_ids: set[str], expected_base_question_count: int=4996) -> list[dict[str, Any]]:
    """Replace exactly the target question records while preserving base ordering."""
    if len(base_rows) != expected_base_question_count:
        raise ValueError('base question record count mismatch')
    base_ids = [str(row.get('question_id') or '') for row in base_rows]
    replacement_ids = [str(row.get('question_id') or '') for row in replacement_rows]
    if not all(base_ids) or len(base_ids) != len(set(base_ids)):
        raise ValueError('base question records contain empty or duplicate IDs')
    if not all(replacement_ids) or len(replacement_ids) != len(set(replacement_ids)):
        raise ValueError('replacement question records contain empty or duplicate IDs')
    if set(replacement_ids) != target_question_ids:
        raise ValueError('replacement question IDs do not match the repair scope')
    if not target_question_ids.issubset(base_ids):
        raise ValueError('repair question scope is not contained in the base records')
    replacement_by_id = {str(row['question_id']): row for row in replacement_rows}
    return [replacement_by_id.get(str(row['question_id']), row) for row in base_rows]
