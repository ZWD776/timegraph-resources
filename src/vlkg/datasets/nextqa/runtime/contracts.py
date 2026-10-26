"""Dataset-isolated contracts for the NExT-QA temporal graph pilot."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT, require_within_nextqa_scratch
DATASET = 'NExT-QA'
SPLIT = 'val'
GRAPH_PROFILE = 'nextqa_cg6_local_tracklet_v1'
EXPERIMENTAL_GRAPH_PROFILE = 'nextqa_cg6_human_compat_v2'
ALLOWED_GRAPH_PROFILES = frozenset({GRAPH_PROFILE, EXPERIMENTAL_GRAPH_PROFILE})
FIXED_PILOT_LOCAL_IDS = (1, 2, 285, 286, 570)
FULL_GRAPH_STAGE = 'graphs'
RUN_ID_RE = re.compile('^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$')
SAFE_VIDEO_FIELDS = ('episode_key', 'official_video_id', 'vidor_id', 'local_sequential_id', 'archive_half', 'hsgg_archive', 'hsgg_archive_root', 'frame_archive', 'frame_archive_root', 'frame_member_template', 'hsgg_record_source', 'hsgg_generation_pass', 'hsgg_frame_count', 'sampled_image_count', 'skipped_frame_count', 'start_timestamp_seconds', 'end_timestamp_seconds', 'official_frame_count', 'inferred_source_fps', 'median_sample_interval_seconds', 'timestamp_consistency_status')
ARCHIVE_VIDEO_FIELDS = frozenset({'archive_half', 'hsgg_archive', 'hsgg_archive_root', 'frame_archive', 'frame_archive_root', 'frame_member_template'})
NORMALIZED_DIRECTORY_FIELDS = ('input_storage', 'normalized_input_dir', 'frames_dir')
OPTIONAL_VIDEO_FIELDS = ('image_path_base',)
ALLOWED_INPUT_STORAGE = frozenset({'archive', 'normalized_directory'})

def validate_manifest_input_storage(payload: Mapping[str, Any]) -> None:
    """An explicit storage declaration must agree with each selected video."""
    storage = payload.get('input_storage')
    if storage is None:
        return
    if storage not in ALLOWED_INPUT_STORAGE:
        raise ValueError(f'unsupported NExT-QA input_storage: {storage!r}')
    for video in payload.get('videos', []):
        if video.get('input_storage', 'archive') != storage:
            raise ValueError('manifest and video input_storage must agree')

def load_fixed_pilot_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable five-video pilot selection."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('dataset') != DATASET or payload.get('split') != SPLIT:
        raise ValueError('pilot manifest must describe the NExT-QA validation split')
    if payload.get('hsgg_record_source') != 'frame_scene_graphs_members':
        raise ValueError('frame_scene_graphs members must be the authoritative source')
    if payload.get('aggregate_json_allowed_as_build_source') is not False:
        raise ValueError('aggregate HSGG JSON must be forbidden as a build source')
    if payload.get('uses_uniform32') is not False:
        raise ValueError('the graph pilot must use all available every-30 frames')
    if payload.get('uses_hsgg_image_path_metadata') is not False:
        raise ValueError('stale embedded HSGG image paths must be disabled')
    videos = payload.get('videos')
    if not isinstance(videos, list):
        raise ValueError('pilot manifest videos must be a list')
    local_ids = tuple((int(video['local_sequential_id']) for video in videos))
    if local_ids != FIXED_PILOT_LOCAL_IDS:
        raise ValueError(f'pilot must use fixed local IDs {FIXED_PILOT_LOCAL_IDS}, got {local_ids}')
    if int(payload.get('video_count', -1)) != len(FIXED_PILOT_LOCAL_IDS):
        raise ValueError('pilot manifest video_count must be five')
    video_ids = [str(video['official_video_id']) for video in videos]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError('pilot manifest contains duplicate official video IDs')
    validate_manifest_input_storage(payload)
    for video in videos:
        graph_safe_video_record(video)
    return payload

def load_full_manifest(path: str | Path) -> dict[str, Any]:
    """Load the complete audited validation manifest for full graph construction."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('dataset') != DATASET or payload.get('split') != SPLIT:
        raise ValueError('full manifest must describe the NExT-QA validation split')
    if payload.get('hsgg_record_source') != 'frame_scene_graphs_members':
        raise ValueError('frame_scene_graphs members must be the authoritative source')
    if payload.get('aggregate_json_allowed_as_build_source') is not False:
        raise ValueError('aggregate HSGG JSON must be forbidden as a build source')
    if payload.get('uses_uniform32') is not False:
        raise ValueError('full graph construction must use all available every-30 frames')
    if payload.get('uses_hsgg_image_path_metadata') is not False:
        raise ValueError('stale embedded HSGG image paths must be disabled')
    videos = payload.get('videos')
    expected_count = int(payload.get('video_count', -1))
    if not isinstance(videos, list) or expected_count != len(videos) or expected_count != 570:
        raise ValueError('full manifest must contain exactly 570 validation videos')
    local_ids = [int(video['local_sequential_id']) for video in videos]
    official_ids = [str(video['official_video_id']) for video in videos]
    if len(set(local_ids)) != expected_count or sorted(local_ids) != list(range(1, expected_count + 1)):
        raise ValueError('full manifest local sequential IDs must be unique and cover 1..570')
    if len(set(official_ids)) != expected_count:
        raise ValueError('full manifest contains duplicate official video IDs')
    validate_manifest_input_storage(payload)
    for video in videos:
        graph_safe_video_record(video)
    return payload

def graph_safe_video_record(video: Mapping[str, Any]) -> dict[str, Any]:
    """Project a manifest row onto fields permitted during question-free build."""
    storage = video.get('input_storage', 'archive')
    if storage not in ALLOWED_INPUT_STORAGE:
        raise ValueError(f'unsupported NExT-QA input_storage: {storage!r}')
    required = SAFE_VIDEO_FIELDS if storage == 'archive' else tuple(field for field in SAFE_VIDEO_FIELDS if field not in ARCHIVE_VIDEO_FIELDS) + NORMALIZED_DIRECTORY_FIELDS
    missing = [field for field in required if field not in video]
    if missing:
        raise ValueError(f'video manifest row is missing fields: {missing}')
    record = {field: video[field] for field in (*SAFE_VIDEO_FIELDS, *NORMALIZED_DIRECTORY_FIELDS, *OPTIONAL_VIDEO_FIELDS) if field in video}
    if storage == 'normalized_directory':
        for field in ('normalized_input_dir', 'frames_dir'):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f'normalized directory video requires a nonempty {field}')
        if 'image_path_base' in record and record['image_path_base'] != 'input_directory':
            raise ValueError(f"unsupported normalized image_path_base: {record['image_path_base']!r}")
    expected_episode = f"nextqa-val/{record['official_video_id']}"
    if record['episode_key'] != expected_episode:
        raise ValueError(f"invalid episode key: {record['episode_key']}")
    if record['hsgg_record_source'] != 'frame_scene_graphs_members':
        raise ValueError('video does not use authoritative per-frame HSGG records')
    if record['timestamp_consistency_status'] != 'passed':
        raise ValueError('video timestamp audit did not pass')
    return record

def prepare_run_root(run_id: str, *, resume: bool) -> Path:
    """Create an exclusive graph-pilot run directory or validate resume intent."""
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(f'invalid run ID: {run_id!r}')
    root = require_within_nextqa_scratch(NEXTQA_SCRATCH_ROOT / 'graph_pilots' / run_id)
    if root.exists() and (not resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {root}')
    if not root.exists() and resume:
        raise FileNotFoundError(f'cannot resume missing run: {root}')
    root.mkdir(parents=True, exist_ok=resume)
    return root

def prepare_full_graph_run_root(run_id: str, *, resume: bool) -> Path:
    """Create an exclusive full NExT-QA graph run under the isolated scratch root."""
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(f'invalid run ID: {run_id!r}')
    root = require_within_nextqa_scratch(NEXTQA_SCRATCH_ROOT / FULL_GRAPH_STAGE / run_id)
    if root.exists() and (not resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {root}')
    if not root.exists() and resume:
        raise FileNotFoundError(f'cannot resume missing run: {root}')
    root.mkdir(parents=True, exist_ok=resume)
    return root

def prepare_stage_run_root(stage: str, run_id: str, *, resume: bool) -> Path:
    """Create an exclusive NExT-QA run root for a non-graph pilot stage."""
    allowed_stages = {'standardized', 'question_ir', 'graph_retrieval', 'graph_retrieval_pilots', 'qa', 'qa_pilots', 'qa_evaluations', 'qa_error_audits', 'special_patch_preparation', 'special_patch_overlays', 'visual_routes', 'visual_graph_baseline', 'visual_indexes', 'visual_evidence', 'visual_readiness'}
    if stage not in allowed_stages:
        raise ValueError(f'unsupported NExT-QA run stage: {stage!r}')
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(f'invalid run ID: {run_id!r}')
    root = require_within_nextqa_scratch(NEXTQA_SCRATCH_ROOT / stage / run_id)
    if root.exists() and (not resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {root}')
    if not root.exists() and resume:
        raise FileNotFoundError(f'cannot resume missing run: {root}')
    root.mkdir(parents=True, exist_ok=resume)
    return root
