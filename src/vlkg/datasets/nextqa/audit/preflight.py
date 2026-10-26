"""NExT-QA validation preflight, manifest generation, and pilot dry-run."""
from __future__ import annotations
import json
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from vlkg.datasets.nextqa.audit.archive_audit import file_fingerprint, load_or_scan_inventory, parse_frame_filename, scan_frame_archive, scan_hsgg_archive
from vlkg.datasets.nextqa.runtime.standardization import build_evaluator_labels, build_runtime_questions, contains_gold_label, load_official_mapping, load_validation_annotations, select_pilot_video_ids
OFFICIAL_REPO = 'https://github.com/doc-doc/NExT-QA'
OFFICIAL_COMMIT = '2432e9724f88ed9f40010e2989f104570a91de4e'
HSGG_GENERATION_PROVENANCE = {'source': 'user_confirmed_gpu_run_protocol', 'frame_stride': 30, 'sampling_interpretation': 'source_frame_stride_not_fixed_seconds', 'timestamp_policy': 'use_fps_derived_seconds_encoded_in_frame_filename', 'first_half_strategy': 'every60_initial_run_plus_every30_supplement', 'second_half_strategy': 'direct_every30_run', 'skipped_frame_policy': 'intentional HSGG no-scene-graph filtering, including frames with fewer than two objects', 'skip_condition_verified_in_local_hsgg_code': "len(objects_data['boxes']) < 2"}
OFFICIAL_DATASET_FACTS = {'source_url': 'https://doc-doc.github.io/docs/nextqa.html', 'total_video_count': 5440, 'average_video_length_seconds': 44, 'split_video_counts': {'train': 3870, 'val': 570, 'test': 1000}, 'validation_multiple_choice_question_count': 4996, 'question_family_proportions': {'causal': 0.48, 'temporal': 0.29, 'descriptive': 0.23}, 'official_page_specifies_constant_fps': False}

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any, *, critical: bool=True) -> None:
    checks.append({'name': name, 'passed': bool(passed), 'critical': critical, 'detail': detail})

def _duplicates(values: Sequence[Any]) -> list[Any]:
    return sorted((value for value, count in Counter(values).items() if count > 1))

def _question_ids_by_video(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        result[str(row['official_video_id'])].append((int(row['official_qid']), str(row['question_id'])))
    return {video_id: [question_id for _, question_id in sorted(items)] for video_id, items in result.items()}

def _source_record(path: Path, repository_path: str) -> dict[str, Any]:
    record = file_fingerprint(path)
    record.update({'repository': OFFICIAL_REPO, 'commit': OFFICIAL_COMMIT, 'repository_path': repository_path, 'source_url': f'https://raw.githubusercontent.com/doc-doc/NExT-QA/{OFFICIAL_COMMIT}/{repository_path}'})
    return record

def _summarize_hsgg(inventory: Mapping[str, Any]) -> dict[str, Any]:
    videos = inventory['videos']
    manifests = [video.get('run_manifest') or {} for video in videos]
    half_counts = Counter((video['half'] for video in videos))
    stale_paths_by_half = Counter()
    half_hsgg_frames = Counter()
    half_total_images = Counter()
    half_completed_frames = Counter()
    half_skipped_frames = Counter()
    for video in videos:
        half = video['half']
        manifest = video.get('run_manifest') or {}
        half_hsgg_frames[half] += len(video['hsgg_frame_names'])
        half_total_images[half] += int(manifest.get('total_images') or 0)
        half_completed_frames[half] += int(manifest.get('completed_frames') or 0)
        half_skipped_frames[half] += int(manifest.get('skipped_no_scene_graph_frames') or 0)
        stale_paths_by_half[half] += video['stale_every60_image_path_count']
    schema_error_videos = [video['official_video_id'] for video in videos if video['schema_error_count']]
    monotonic_errors = [video['official_video_id'] for video in videos if not video['timestamp_audit']['monotonic']]
    aggregate_mismatch_videos = [video for video in videos if video['frame_key_mismatches']]
    aggregate_missing_frames = {(video['official_video_id'], missing_name) for video in aggregate_mismatch_videos for mismatch in video['frame_key_mismatches'].values() for missing_name in mismatch['missing']}
    aggregate_file_missing_entries = sum((mismatch['missing_count'] for video in aggregate_mismatch_videos for mismatch in video['frame_key_mismatches'].values()))
    video_half = {video['official_video_id']: video['half'] for video in videos}
    aggregate_missing_by_half = Counter((video_half[video_id] for video_id, _ in aggregate_missing_frames))
    aggregate_missing_by_frame_index_mod60 = Counter((parse_frame_filename(f'{frame_name}.json')[0] % 60 for _, frame_name in aggregate_missing_frames))
    return {'archive': inventory['archive'], 'scan_mode': inventory['scan_mode'], 'video_count': len(videos), 'half_video_counts': dict(sorted(half_counts.items())), 'hsgg_frame_count': sum((len(video['hsgg_frame_names']) for video in videos)), 'half_hsgg_frame_counts': dict(sorted(half_hsgg_frames.items())), 'half_manifest_total_images': dict(sorted(half_total_images.items())), 'half_manifest_completed_frames': dict(sorted(half_completed_frames.items())), 'half_manifest_skipped_frames': dict(sorted(half_skipped_frames.items())), 'manifest_total_images': sum((int(item.get('total_images') or 0) for item in manifests)), 'manifest_completed_frames': sum((int(item.get('completed_frames') or 0) for item in manifests)), 'manifest_skipped_no_scene_graph_frames': sum((int(item.get('skipped_no_scene_graph_frames') or 0) for item in manifests)), 'manifest_error_frames': sum((int(item.get('error_frames') or 0) for item in manifests)), 'schema_error_count': sum((video['schema_error_count'] for video in videos)), 'individual_frame_schema_error_count': sum((video['individual_frame_schema_error_count'] for video in videos)), 'schema_error_video_count': len(schema_error_videos), 'schema_error_video_examples': schema_error_videos[:20], 'timestamp_non_monotonic_video_count': len(monotonic_errors), 'timestamp_non_monotonic_video_examples': monotonic_errors[:20], 'embedded_every60_image_path_count': sum((video['stale_every60_image_path_count'] for video in videos)), 'embedded_every60_image_path_count_by_half': dict(sorted(stale_paths_by_half.items())), 'aggregate_frame_key_mismatch_video_count': len(aggregate_mismatch_videos), 'aggregate_missing_frame_entry_count': len(aggregate_missing_frames), 'aggregate_file_missing_entry_count': aggregate_file_missing_entries, 'aggregate_missing_frame_entries_by_half': dict(sorted(aggregate_missing_by_half.items())), 'aggregate_missing_frame_entries_by_frame_index_mod60': {str(key): value for key, value in sorted(aggregate_missing_by_frame_index_mod60.items())}, 'aggregate_frame_key_mismatch_video_examples': [video['official_video_id'] for video in aggregate_mismatch_videos[:20]], 'invalid_member_examples': inventory['invalid_member_examples']}

def _video_timing_profile(video: Mapping[str, Any]) -> dict[str, float | None]:
    parsed = sorted((parse_frame_filename(name)[:2] for name in video['frame_names']))
    rounding_epsilon = 0.0005000001
    fps_lowers = [frame_index / (seconds + rounding_epsilon) for frame_index, seconds in parsed if frame_index > 0 and seconds > 0]
    fps_uppers = [frame_index / (seconds - rounding_epsilon) for frame_index, seconds in parsed if frame_index > 0 and seconds > rounding_epsilon]
    fps_lower = max(fps_lowers) if fps_lowers else None
    fps_upper = min(fps_uppers) if fps_uppers else None
    rounding_consistent = fps_lower is not None and fps_upper is not None and (fps_lower <= fps_upper)
    inferred_fps = (fps_lower + fps_upper) / 2 if rounding_consistent and fps_lower is not None and (fps_upper is not None) else None
    deltas = [current[1] - previous[1] for previous, current in zip(parsed, parsed[1:])]
    return {'inferred_source_fps': inferred_fps, 'fps_rounding_interval_lower': fps_lower, 'fps_rounding_interval_upper': fps_upper, 'timestamp_consistent_with_single_fps_and_3_decimal_rounding': rounding_consistent, 'median_sample_interval_seconds': statistics.median(deltas) if deltas else None}

def _audit_timestamp_consistency(videos: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    official_frame_counts: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        official_frame_counts[str(row['official_video_id'])].add(int(row['source_frame_count']))
    step_distribution = Counter()
    per_video: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    exact_index_sequence_count = 0
    starts_at_zero_count = 0
    rounding_consistent_count = 0
    for video in videos:
        video_id = str(video['official_video_id'])
        parsed = sorted((parse_frame_filename(name)[:2] for name in video['frame_names']))
        indices = [item[0] for item in parsed]
        timestamps = [item[1] for item in parsed]
        steps = [current - previous for previous, current in zip(indices, indices[1:])]
        step_distribution.update(steps)
        frame_count_values = official_frame_counts.get(video_id, set())
        official_frame_count = next(iter(frame_count_values)) if len(frame_count_values) == 1 else None
        expected_indices = list(range(0, official_frame_count, 30)) if official_frame_count is not None else []
        index_sequence_matches = indices == expected_indices
        starts_at_zero = bool(indices) and indices[0] == 0 and (timestamps[0] == 0.0)
        timing_profile = _video_timing_profile(video)
        rounding_consistent = bool(timing_profile['timestamp_consistent_with_single_fps_and_3_decimal_rounding'])
        exact_index_sequence_count += int(index_sequence_matches)
        starts_at_zero_count += int(starts_at_zero)
        rounding_consistent_count += int(rounding_consistent)
        record = {'official_video_id': video_id, 'official_frame_count': official_frame_count, 'sampled_frame_count': len(indices), 'first_frame_index': indices[0] if indices else None, 'last_frame_index': indices[-1] if indices else None, 'start_timestamp_seconds': timestamps[0] if timestamps else None, 'end_timestamp_seconds': timestamps[-1] if timestamps else None, 'index_sequence_matches_range_0_frame_count_step_30': index_sequence_matches, 'starts_at_frame_zero_time_zero': starts_at_zero, **timing_profile}
        per_video.append(record)
        if not index_sequence_matches or not starts_at_zero or (not rounding_consistent):
            problems.append(record)
    return {'timestamp_formula': 'round(frame_index / source_video_fps, 3)', 'timestamp_semantics': 'fps_derived_nominal_seconds_not_container_pts', 'container_pts_verified': False, 'container_pts_verification_blocker': 'raw videos are not present locally and were intentionally not downloaded', 'extractor_code': 'HSGG/utils/extract_video_frames.py', 'video_count': len(videos), 'sampled_filename_count': sum((len(video['frame_names']) for video in videos)), 'exact_official_frame_count_step30_sequence_video_count': exact_index_sequence_count, 'starts_at_frame_zero_time_zero_video_count': starts_at_zero_count, 'single_fps_3_decimal_rounding_consistent_video_count': rounding_consistent_count, 'frame_index_step_distribution': {str(key): value for key, value in sorted(step_distribution.items())}, 'problem_count': len(problems), 'problem_examples': problems[:20], 'per_video': per_video}

def _rounded_distribution(values: Sequence[float], digits: int=3) -> dict[str, int]:
    counts = Counter((f'{round(value, digits):g}' for value in values))
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))

def _summarize_frames(inventory: Mapping[str, Any]) -> dict[str, Any]:
    videos = inventory['videos']
    half_counts = Counter((video['half'] for video in videos))
    half_frames = Counter()
    timing_profiles = [_video_timing_profile(video) for video in videos]
    inferred_fps = [float(profile['inferred_source_fps']) for profile in timing_profiles if profile['inferred_source_fps'] is not None]
    median_intervals = [float(profile['median_sample_interval_seconds']) for profile in timing_profiles if profile['median_sample_interval_seconds'] is not None]
    for video in videos:
        half_frames[video['half']] += len(video['frame_names'])
    monotonic_errors = [video['official_video_id'] for video in videos if not video['timestamp_audit']['monotonic']]
    return {'archive': inventory['archive'], 'scan_mode': inventory['scan_mode'], 'video_count': len(videos), 'half_video_counts': dict(sorted(half_counts.items())), 'sampled_image_count': sum((len(video['frame_names']) for video in videos)), 'half_sampled_image_counts': dict(sorted(half_frames.items())), 'timing_source': 'frame_filename_fps_derived_nominal_seconds', 'constant_fps_assumed': False, 'inferred_source_fps_distribution_rounded_3': _rounded_distribution(inferred_fps), 'median_sample_interval_seconds_distribution_rounded_3': _rounded_distribution(median_intervals), 'median_sample_interval_seconds_statistics': {'min': min(median_intervals) if median_intervals else None, 'max': max(median_intervals) if median_intervals else None, 'median': statistics.median(median_intervals) if median_intervals else None}, 'timestamp_non_monotonic_video_count': len(monotonic_errors), 'timestamp_non_monotonic_video_examples': monotonic_errors[:20], 'invalid_member_examples': inventory['invalid_member_examples']}

def run_preflight(*, repository_root: Path, annotation_path: Path, mapping_path: Path, readme_path: Path, hsgg_archive_path: Path, frame_archive_path: Path, output_root: Path, reuse_cache: bool=True) -> dict[str, Any]:
    """Run the complete pre-build audit and a five-video standardization dry-run."""
    audit_root = output_root / 'audit'
    cache_root = audit_root / 'cache'
    hsgg_inventory, hsgg_cache_reused = load_or_scan_inventory(hsgg_archive_path, cache_root / 'hsgg_archive_inventory.json', scan_hsgg_archive, reuse_cache=reuse_cache)
    frame_inventory, frame_cache_reused = load_or_scan_inventory(frame_archive_path, cache_root / 'frame_archive_inventory.json', scan_frame_archive, reuse_cache=reuse_cache)
    rows, annotation_audit = load_validation_annotations(annotation_path)
    mapping = load_official_mapping(mapping_path)
    questions_by_video = _question_ids_by_video(rows)
    annotation_video_ids = set(questions_by_video)
    hsgg_videos = hsgg_inventory['videos']
    frame_videos = frame_inventory['videos']
    timestamp_consistency = _audit_timestamp_consistency(frame_videos, rows)
    timestamp_consistency_by_video = {record['official_video_id']: record for record in timestamp_consistency['per_video']}
    hsgg_by_video = {video['official_video_id']: video for video in hsgg_videos}
    frames_by_video = {video['official_video_id']: video for video in frame_videos}
    hsgg_video_ids = set(hsgg_by_video)
    frame_video_ids = set(frames_by_video)
    hsgg_summary = _summarize_hsgg(hsgg_inventory)
    frame_summary = _summarize_frames(frame_inventory)
    local_ids = [video['local_sequential_id'] for video in hsgg_videos]
    hsgg_official_ids = [video['official_video_id'] for video in hsgg_videos]
    frame_official_ids = [video['official_video_id'] for video in frame_videos]
    mapping_values = list(mapping.values())
    mapping_missing = sorted(annotation_video_ids - set(mapping))
    mapping_basename_mismatches = sorted((video_id for video_id in annotation_video_ids & set(mapping) if Path(mapping[video_id]).name != video_id))
    hsgg_missing = sorted(annotation_video_ids - hsgg_video_ids)
    hsgg_extra = sorted(hsgg_video_ids - annotation_video_ids)
    frame_missing = sorted(annotation_video_ids - frame_video_ids)
    frame_extra = sorted(frame_video_ids - annotation_video_ids)
    completed_missing_from_frames: dict[str, list[str]] = {}
    sampled_missing_from_hsgg: dict[str, list[str]] = {}
    run_manifest_mismatches: dict[str, Any] = {}
    half_mismatches: list[str] = []
    per_video_records: list[dict[str, Any]] = []
    for hsgg_video in hsgg_videos:
        video_id = hsgg_video['official_video_id']
        frame_video = frames_by_video.get(video_id)
        if frame_video is None:
            continue
        completed_names = {Path(name).with_suffix('.png').name for name in hsgg_video['hsgg_frame_names']}
        sampled_names = set(frame_video['frame_names'])
        missing_completed = sorted(completed_names - sampled_names)
        skipped_sampled = sorted(sampled_names - completed_names)
        if missing_completed:
            completed_missing_from_frames[video_id] = missing_completed[:20]
        if skipped_sampled:
            sampled_missing_from_hsgg[video_id] = skipped_sampled
        if hsgg_video['half'] != frame_video['half']:
            half_mismatches.append(video_id)
        run_manifest = hsgg_video.get('run_manifest') or {}
        manifest_frame_names = {f'{name}.png' if not str(name).endswith('.png') else str(name) for name in run_manifest.get('frame_names', [])}
        mismatch: dict[str, Any] = {}
        expected_total = len(sampled_names)
        expected_completed = len(completed_names)
        expected_skipped = len(skipped_sampled)
        total_images = run_manifest.get('total_images')
        completed_frames = run_manifest.get('completed_frames')
        skipped_frames = run_manifest.get('skipped_no_scene_graph_frames')
        if total_images is None or int(total_images) != expected_total:
            mismatch['total_images'] = [run_manifest.get('total_images'), expected_total]
        if completed_frames is None or int(completed_frames) != expected_completed:
            mismatch['completed_frames'] = [run_manifest.get('completed_frames'), expected_completed]
        if skipped_frames is None or int(skipped_frames) != expected_skipped:
            mismatch['skipped_frames'] = [run_manifest.get('skipped_no_scene_graph_frames'), expected_skipped]
        if manifest_frame_names != sampled_names:
            mismatch['manifest_frame_name_alignment'] = {'missing': sorted(sampled_names - manifest_frame_names)[:20], 'extra': sorted(manifest_frame_names - sampled_names)[:20]}
        if mismatch:
            run_manifest_mismatches[video_id] = mismatch
        question_ids = questions_by_video.get(video_id, [])
        per_video_records.append({'official_video_id': video_id, 'vidor_id': mapping.get(video_id), 'local_sequential_id': hsgg_video['local_sequential_id'], 'half': hsgg_video['half'], 'question_count': len(question_ids), 'sampled_image_count': len(sampled_names), 'hsgg_frame_count': len(completed_names), 'skipped_frame_count': len(skipped_sampled), 'start_timestamp_seconds': frame_video['timestamp_audit']['start_timestamp_seconds'], 'end_timestamp_seconds': frame_video['timestamp_audit']['end_timestamp_seconds'], 'hsgg_schema_error_count': hsgg_video['schema_error_count'], 'timestamp_consistency': timestamp_consistency_by_video.get(video_id)})
    per_video_records.sort(key=lambda item: item['local_sequential_id'])
    checks: list[dict[str, Any]] = []
    _check(checks, 'official_validation_has_570_videos', annotation_audit['video_count'] == 570, annotation_audit['video_count'])
    _check(checks, 'official_validation_has_4996_questions', annotation_audit['row_count'] == 4996, annotation_audit['row_count'])
    _check(checks, 'stable_question_ids_unique', not annotation_audit['duplicate_stable_question_ids'], annotation_audit['duplicate_stable_question_ids'][:20])
    _check(checks, 'per_video_qids_unique', not annotation_audit['duplicate_qids_within_video'], annotation_audit['duplicate_qids_within_video'])
    _check(checks, 'official_qid_fields_present_and_integer', not annotation_audit['invalid_qids'], annotation_audit['invalid_qids'][:20])
    _check(checks, 'all_questions_have_five_nonempty_choices', not annotation_audit['choice_count_errors'] and (not annotation_audit['empty_choice_errors']), {'choice_count_errors': annotation_audit['choice_count_errors'][:20], 'empty_choice_errors': annotation_audit['empty_choice_errors'][:20]})
    _check(checks, 'all_gold_indices_in_range', not annotation_audit['gold_index_errors'], annotation_audit['gold_index_errors'][:20])
    _check(checks, 'all_annotation_videos_in_official_mapping', not mapping_missing, mapping_missing[:20])
    _check(checks, 'annotation_mapping_values_have_matching_basename', not mapping_basename_mismatches, mapping_basename_mismatches[:20])
    _check(checks, 'mapping_values_unique', not _duplicates(mapping_values), _duplicates(mapping_values)[:20])
    _check(checks, 'hsgg_has_570_unique_videos', len(hsgg_video_ids) == 570 and (not _duplicates(hsgg_official_ids)), {'unique': len(hsgg_video_ids), 'duplicates': _duplicates(hsgg_official_ids)[:20]})
    _check(checks, 'hsgg_local_ids_are_exactly_1_to_570', sorted(local_ids) == list(range(1, 571)), {'duplicates': _duplicates(local_ids), 'missing': sorted(set(range(1, 571)) - set(local_ids))})
    _check(checks, 'hsgg_videos_match_annotation', not hsgg_missing and (not hsgg_extra), {'missing': hsgg_missing[:20], 'extra': hsgg_extra[:20]})
    _check(checks, 'frame_archive_has_570_unique_videos', len(frame_video_ids) == 570 and (not _duplicates(frame_official_ids)), {'unique': len(frame_video_ids), 'duplicates': _duplicates(frame_official_ids)[:20]})
    _check(checks, 'frame_videos_match_annotation', not frame_missing and (not frame_extra), {'missing': frame_missing[:20], 'extra': frame_extra[:20]})
    _check(checks, 'hsgg_both_halves_have_285_videos', hsgg_summary['half_video_counts'] == {'first': 285, 'second': 285}, hsgg_summary['half_video_counts'])
    _check(checks, 'frame_archive_both_halves_have_285_videos', frame_summary['half_video_counts'] == {'first': 285, 'second': 285}, frame_summary['half_video_counts'])
    _check(checks, 'hsgg_frame_halves_align', not half_mismatches, half_mismatches[:20])
    _check(checks, 'completed_hsgg_frames_exist_in_frame_archive', not completed_missing_from_frames, {key: value for key, value in list(completed_missing_from_frames.items())[:20]})
    _check(checks, 'run_manifests_align_with_archives', not run_manifest_mismatches, {key: value for key, value in list(run_manifest_mismatches.items())[:20]})
    _check(checks, 'hsgg_error_frames_zero', hsgg_summary['manifest_error_frames'] == 0, hsgg_summary['manifest_error_frames'])
    _check(checks, 'hsgg_schema_complete', hsgg_summary['schema_error_count'] == 0, {'error_count': hsgg_summary['schema_error_count'], 'videos': hsgg_summary['schema_error_video_examples']})
    _check(checks, 'aggregate_json_frame_sets_match_individual_frames', hsgg_summary['aggregate_frame_key_mismatch_video_count'] == 0, {'video_count': hsgg_summary['aggregate_frame_key_mismatch_video_count'], 'missing_frame_entry_count': hsgg_summary['aggregate_missing_frame_entry_count'], 'video_examples': hsgg_summary['aggregate_frame_key_mismatch_video_examples']}, critical=False)
    _check(checks, 'hsgg_timestamps_monotonic', hsgg_summary['timestamp_non_monotonic_video_count'] == 0, hsgg_summary['timestamp_non_monotonic_video_examples'])
    _check(checks, 'sampled_frame_timestamps_monotonic', frame_summary['timestamp_non_monotonic_video_count'] == 0, frame_summary['timestamp_non_monotonic_video_examples'])
    _check(checks, 'sampled_indices_match_official_frame_count_at_step30', timestamp_consistency['exact_official_frame_count_step30_sequence_video_count'] == 570, timestamp_consistency['problem_examples'])
    _check(checks, 'timestamps_start_at_frame_zero_time_zero', timestamp_consistency['starts_at_frame_zero_time_zero_video_count'] == 570, timestamp_consistency['problem_examples'])
    _check(checks, 'timestamps_match_single_video_fps_with_3_decimal_rounding', timestamp_consistency['single_fps_3_decimal_rounding_consistent_video_count'] == 570, timestamp_consistency['problem_examples'])
    _check(checks, 'sampled_equals_completed_plus_skipped', frame_summary['sampled_image_count'] == hsgg_summary['manifest_completed_frames'] + hsgg_summary['manifest_skipped_no_scene_graph_frames'], {'sampled': frame_summary['sampled_image_count'], 'completed': hsgg_summary['manifest_completed_frames'], 'skipped': hsgg_summary['manifest_skipped_no_scene_graph_frames']})
    critical_failures = [check for check in checks if check['critical'] and (not check['passed'])]
    disk = shutil.disk_usage(repository_root)
    generated_at = datetime.now(timezone.utc).isoformat()
    issues: list[dict[str, Any]] = []
    if hsgg_summary['embedded_every60_image_path_count']:
        issues.append({'severity': 'info', 'code': 'first_half_mixed_run_image_path_provenance', 'count': hsgg_summary['embedded_every60_image_path_count'], 'count_by_half': hsgg_summary['embedded_every60_image_path_count_by_half'], 'explanation': 'the first half was produced by an every60 initial run followed by an every30 supplement; inherited every60 paths encode run provenance', 'disposition': 'do not use embedded image_path values; reconstruct paths from manifest roots and frame filenames'})
    if hsgg_summary['aggregate_frame_key_mismatch_video_count']:
        issues.append({'severity': 'warning', 'code': 'aggregate_hsgg_frame_entries_missing', 'video_count': hsgg_summary['aggregate_frame_key_mismatch_video_count'], 'missing_frame_entry_count': hsgg_summary['aggregate_missing_frame_entry_count'], 'aggregate_file_missing_entry_count': hsgg_summary['aggregate_file_missing_entry_count'], 'missing_by_half': hsgg_summary['aggregate_missing_frame_entries_by_half'], 'missing_by_frame_index_mod60': hsgg_summary['aggregate_missing_frame_entries_by_frame_index_mod60'], 'interpretation': 'not specific to the first-half supplement: affected records occur in both halves and at both every60-aligned and supplement-aligned indices', 'disposition': 'use validated frame_scene_graphs/*.json members as the authoritative NExT-QA build source; do not build from incomplete aggregate JSON'})
    issues.append({'severity': 'info', 'code': 'intentional_hsgg_no_scene_graph_filtering', 'count': hsgg_summary['manifest_skipped_no_scene_graph_frames'], 'count_by_half': hsgg_summary['half_manifest_skipped_frames'], 'explanation': HSGG_GENERATION_PROVENANCE['skipped_frame_policy'], 'disposition': 'retain as skipped coverage, not build errors'})
    if annotation_audit['missing_qids_within_video'] or annotation_audit['nonzero_start_qids']:
        issues.append({'severity': 'info', 'code': 'official_per_video_qids_are_noncontiguous', 'affected_video_count': len(set(annotation_audit['missing_qids_within_video']) | set(annotation_audit['nonzero_start_qids'])), 'disposition': 'not treated as missing rows; official qid is preserved and scoped by video'})
    issues.append({'severity': 'limitation', 'code': 'mixed_rate_every30_sampling', 'inferred_source_fps_distribution': frame_summary['inferred_source_fps_distribution_rounded_3'], 'median_interval_seconds_distribution': frame_summary['median_sample_interval_seconds_distribution_rounded_3'], 'explanation': 'every30 is about 1.001 seconds for the 456 videos near 29.97 FPS, but the validation assets also contain videos near 24, 25, 15, and 10 FPS', 'disposition': 'never convert ordinals with a global FPS; use filename timestamps and audit short-event coverage per video'})
    if critical_failures:
        issues.append({'severity': 'blocker', 'code': 'raw_asset_preflight_failed', 'failed_checks': [check['name'] for check in critical_failures]})
    manifest_videos: list[dict[str, Any]] = []
    for hsgg_video in hsgg_videos:
        video_id = hsgg_video['official_video_id']
        frame_video = frames_by_video.get(video_id)
        if frame_video is None or video_id not in annotation_video_ids or video_id not in mapping:
            continue
        question_ids = questions_by_video[video_id]
        sampled_names = set(frame_video['frame_names'])
        completed_names = {Path(name).with_suffix('.png').name for name in hsgg_video['hsgg_frame_names']}
        timing_profile = _video_timing_profile(frame_video)
        timestamp_record = timestamp_consistency_by_video[video_id]
        manifest_videos.append({'episode_key': f'nextqa-val/{video_id}', 'official_video_id': video_id, 'vidor_id': mapping[video_id], 'local_sequential_id': hsgg_video['local_sequential_id'], 'archive_half': hsgg_video['half'], 'hsgg_archive': str(hsgg_archive_path.resolve()), 'hsgg_archive_root': hsgg_video['archive_root'], 'frame_archive': str(frame_archive_path.resolve()), 'frame_archive_root': frame_video['archive_root'], 'frame_member_template': f"{frame_video['archive_root']}/{{frame_filename}}", 'hsgg_record_source': 'frame_scene_graphs_members', 'hsgg_generation_pass': HSGG_GENERATION_PROVENANCE['first_half_strategy'] if hsgg_video['half'] == 'first' else HSGG_GENERATION_PROVENANCE['second_half_strategy'], 'hsgg_frame_count': len(completed_names), 'sampled_image_count': len(sampled_names), 'skipped_frame_count': len(sampled_names - completed_names), 'start_timestamp_seconds': frame_video['timestamp_audit']['start_timestamp_seconds'], 'end_timestamp_seconds': frame_video['timestamp_audit']['end_timestamp_seconds'], 'official_frame_count': timestamp_record['official_frame_count'], 'inferred_source_fps': timing_profile['inferred_source_fps'], 'median_sample_interval_seconds': timing_profile['median_sample_interval_seconds'], 'timestamp_consistency_status': 'passed' if timestamp_record['index_sequence_matches_range_0_frame_count_step_30'] and timestamp_record['starts_at_frame_zero_time_zero'] and timestamp_record['timestamp_consistent_with_single_fps_and_3_decimal_rounding'] else 'failed', 'question_count': len(question_ids), 'question_ids': question_ids})
    manifest_videos.sort(key=lambda item: item['local_sequential_id'])
    full_manifest = {'schema_version': 'nextqa_val_manifest_v1', 'generated_at_utc': generated_at, 'dataset': 'NExT-QA', 'split': 'val', 'episode_key_format': 'nextqa-val/<official_video_id>', 'timestamp_source': 'frame_filename_fps_derived_nominal_seconds', 'timestamp_is_ordinal': False, 'timestamp_is_container_pts': False, 'sampling_policy': 'all_available_every30_hsgg_frames', 'constant_fps_assumed': False, 'observed_timing_summary': {'inferred_source_fps_distribution_rounded_3': frame_summary['inferred_source_fps_distribution_rounded_3'], 'median_sample_interval_seconds_distribution_rounded_3': frame_summary['median_sample_interval_seconds_distribution_rounded_3']}, 'hsgg_generation_provenance': HSGG_GENERATION_PROVENANCE, 'official_dataset_facts': OFFICIAL_DATASET_FACTS, 'hsgg_record_source': 'frame_scene_graphs_members', 'aggregate_json_allowed_as_build_source': False, 'uses_uniform32': False, 'uses_hsgg_image_path_metadata': False, 'video_count': len(manifest_videos), 'videos': manifest_videos}
    manifest_path = output_root / 'manifests' / 'nextqa_val_manifest_v1.json'
    if not critical_failures:
        _write_json(manifest_path, full_manifest)
    pilot_video_ids = select_pilot_video_ids(manifest_videos)
    pilot_video_set = set(pilot_video_ids)
    pilot_videos = [video for video in manifest_videos if video['official_video_id'] in pilot_video_set]
    pilot_rows = [row for row in rows if row['official_video_id'] in pilot_video_set]
    runtime_questions = build_runtime_questions(pilot_rows)
    evaluator_labels = build_evaluator_labels(pilot_rows)
    dry_run_audit = {'schema_version': 'nextqa_val_standardization_dry_run_v1', 'selection_strategy': 'local_sequence_boundaries_1_2_285_286_570', 'requested_local_sequential_ids': [1, 2, 285, 286, 570], 'selected_video_ids': pilot_video_ids, 'selected_local_sequential_ids': [video['local_sequential_id'] for video in pilot_videos], 'selected_halves': dict(Counter((video['archive_half'] for video in pilot_videos))), 'video_count': len(pilot_videos), 'runtime_question_count': len(runtime_questions), 'evaluator_label_count': len(evaluator_labels), 'runtime_contains_gold_label': contains_gold_label(runtime_questions), 'runtime_and_label_question_ids_align': {item['question_id'] for item in runtime_questions} == {item['question_id'] for item in evaluator_labels}, 'path_reconstruction_uses_manifest': True, 'uses_hsgg_image_path_metadata': False, 'passed': False}
    dry_run_audit['passed'] = not critical_failures and len(pilot_videos) == 5 and (not dry_run_audit['runtime_contains_gold_label']) and dry_run_audit['runtime_and_label_question_ids_align'] and (set(dry_run_audit['selected_halves']) == {'first', 'second'})
    pilot_root = output_root / 'dry_run' / 'pilot5'
    if not critical_failures:
        _write_json(pilot_root / 'nextqa_val_pilot5_manifest.json', {**full_manifest, 'video_count': len(pilot_videos), 'videos': pilot_videos})
        _write_json(pilot_root / 'questions.runtime.json', runtime_questions)
        _write_json(pilot_root / 'labels.evaluator_only.json', evaluator_labels)
        _write_json(pilot_root / 'standardization_dry_run_audit.json', dry_run_audit)
    readiness = {'raw_asset_preflight_passed': not critical_failures, 'five_video_standardization_dry_run_passed': dry_run_audit['passed'], 'ready_for_next_stage_five_video_graph_pilot': not critical_failures and dry_run_audit['passed'], 'ready_for_570_video_graph_build': False, 'ready_for_graph_only_qa': False, 'full_build_blocker': 'five-video graph build and graph audit have not been run', 'qa_blocker': 'graph-only QA is gated on successful 570-video graph build'}
    audit = {'schema_version': 'nextqa_raw_asset_audit_v1', 'generated_at_utc': generated_at, 'scope': 'read-only archives + official validation normalization + five-video dry-run', 'prohibited_actions_confirmed': {'navqa_run_or_modified': False, 'openeqa_behavior_modified': False, 'graph_build_started': False, 'model_api_called': False, 'frame_archive_extracted': False, 'raw_videos_downloaded': False}, 'initially_missing_official_files': ['NExT-QA validation annotation with five choices/gold/type', 'NExT-QA official video-to-VidOR mapping'], 'official_sources': {'readme': _source_record(readme_path, 'README.md'), 'validation_annotation': _source_record(annotation_path, 'dataset/nextqa/val.csv'), 'video_id_mapping': _source_record(mapping_path, 'dataset/nextqa/map_vid_vidorID.json')}, 'official_dataset_facts': OFFICIAL_DATASET_FACTS, 'hsgg_generation_provenance': HSGG_GENERATION_PROVENANCE, 'disk': {'checked_before_extraction': True, 'total_bytes': disk.total, 'used_bytes': disk.used, 'free_bytes': disk.free, 'large_archive_extraction_performed': False}, 'annotation': annotation_audit, 'mapping': {'entry_count': len(mapping), 'annotation_video_missing_count': len(mapping_missing), 'annotation_video_missing_examples': mapping_missing[:20], 'duplicate_value_count': len(_duplicates(mapping_values)), 'basename_mismatch_count': len(mapping_basename_mismatches)}, 'hsgg_archive': hsgg_summary, 'frame_archive': frame_summary, 'timestamp_consistency': timestamp_consistency, 'alignment': {'annotation_video_count': len(annotation_video_ids), 'hsgg_video_count': len(hsgg_video_ids), 'frame_video_count': len(frame_video_ids), 'annotation_hsgg_missing_count': len(hsgg_missing), 'annotation_hsgg_extra_count': len(hsgg_extra), 'annotation_frame_missing_count': len(frame_missing), 'annotation_frame_extra_count': len(frame_extra), 'completed_hsgg_missing_from_frame_archive_count': len(completed_missing_from_frames), 'videos_with_skipped_frames': len(sampled_missing_from_hsgg), 'sampled_image_count': frame_summary['sampled_image_count'], 'completed_hsgg_frame_count': hsgg_summary['manifest_completed_frames'], 'skipped_no_scene_graph_frame_count': hsgg_summary['manifest_skipped_no_scene_graph_frames'], 'completed_hsgg_coverage_rate': hsgg_summary['manifest_completed_frames'] / frame_summary['sampled_image_count'] if frame_summary['sampled_image_count'] else 0.0, 'run_manifest_mismatch_video_count': len(run_manifest_mismatches), 'half_mismatch_video_count': len(half_mismatches)}, 'per_video': per_video_records, 'checks': checks, 'critical_failure_count': len(critical_failures), 'issues': issues, 'cache': {'hsgg_inventory_reused': hsgg_cache_reused, 'frame_inventory_reused': frame_cache_reused}, 'dry_run': dry_run_audit, 'readiness': readiness}
    _write_json(audit_root / 'raw_asset_audit.json', audit)
    _write_json(audit_root / 'timestamp_consistency_audit.json', timestamp_consistency)
    _write_markdown(audit_root / 'raw_asset_audit.md', audit)
    _write_readiness(audit_root / 'readiness_report.md', audit)
    return audit

def _status(value: bool) -> str:
    return 'PASS' if value else 'FAIL'

def _write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    annotation = audit['annotation']
    hsgg = audit['hsgg_archive']
    frames = audit['frame_archive']
    alignment = audit['alignment']
    lines = ['# NExT-QA raw asset audit', '', f"Generated: `{audit['generated_at_utc']}`", '', '## Outcome', '', f"- Raw asset preflight: **{_status(audit['readiness']['raw_asset_preflight_passed'])}**", f"- Five-video standardization dry-run: **{_status(audit['readiness']['five_video_standardization_dry_run_passed'])}**", f"- Critical failures: `{audit['critical_failure_count']}`", '- 570-video graph build started: **NO**', '- GPT/model API called: **NO**', '- NAVQA run or modified: **NO**', '- OpenEQA behavior modified: **NO**', '', '## Official validation data', '', f"- Questions: `{annotation['row_count']}`", f"- Videos: `{annotation['video_count']}`", f"- Official type distribution: `{json.dumps(annotation['official_type_distribution'], sort_keys=True)}`", f"- qid scope: `{annotation['qid_scope']}`; stable ID: `{annotation['stable_question_id_format']}`", f"- Mapping entries: `{audit['mapping']['entry_count']}`", '', '## Archive coverage', '', f"- HSGG videos: `{hsgg['video_count']}`; halves: `{json.dumps(hsgg['half_video_counts'], sort_keys=True)}`", f"- Frame videos: `{frames['video_count']}`; halves: `{json.dumps(frames['half_video_counts'], sort_keys=True)}`", f"- Sampled PNGs: `{frames['sampled_image_count']}`", f"- Completed HSGG frames: `{hsgg['manifest_completed_frames']}`", f"- Skipped/no-scene-graph frames: `{hsgg['manifest_skipped_no_scene_graph_frames']}`", f"- HSGG error frames: `{hsgg['manifest_error_frames']}`", f"- Completed coverage: `{alignment['completed_hsgg_coverage_rate']:.6%}`", f"- HSGG per-frame/aggregate schema errors: `{hsgg['schema_error_count']}`", f"- Aggregate JSON mismatch videos: `{hsgg['aggregate_frame_key_mismatch_video_count']}`", f"- Aggregate missing frame entries: `{hsgg['aggregate_missing_frame_entry_count']}`", f"- Missing entries across four aggregate files: `{hsgg['aggregate_file_missing_entry_count']}`", f"- Non-monotonic HSGG videos: `{hsgg['timestamp_non_monotonic_video_count']}`", f"- Non-monotonic sampled-frame videos: `{frames['timestamp_non_monotonic_video_count']}`", f"- Exact official frame-count/step-30 sequences: `{audit['timestamp_consistency']['exact_official_frame_count_step30_sequence_video_count']}/570`", f"- Single-FPS three-decimal timestamp consistency: `{audit['timestamp_consistency']['single_fps_3_decimal_rounding_consistent_video_count']}/570`", f"- Inferred FPS distribution: `{json.dumps(frames['inferred_source_fps_distribution_rounded_3'], sort_keys=True)}`", f"- Median every30 interval distribution: `{json.dumps(frames['median_sample_interval_seconds_distribution_rounded_3'], sort_keys=True)}`", '', '## Checks', '', '| Check | Result |', '|---|---|']
    lines.extend((f"| `{check['name']}` | **{_status(check['passed'])}**{(' (non-blocking)' if not check['critical'] else '')} |" for check in audit['checks']))
    lines.extend(['', '## Known issues / limitations', ''])
    for issue in audit['issues']:
        lines.append(f"- `{issue['severity']}` / `{issue['code']}`: {issue.get('disposition', issue.get('failed_checks', ''))}")
    lines.extend(['', 'The first-half `frames_every60` image paths record the mixed-run provenance and are metadata', 'only. The generated manifest reconstructs frame members from the audited archive root plus', 'the frame filename; no embedded cloud path is used.', ''])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')

def _write_readiness(path: Path, audit: Mapping[str, Any]) -> None:
    readiness = audit['readiness']
    dry_run = audit['dry_run']
    lines = ['# NExT-QA readiness report', '', f"- Raw preflight: **{_status(readiness['raw_asset_preflight_passed'])}**", f"- Five-video standardization: **{_status(readiness['five_video_standardization_dry_run_passed'])}**", f"- Next gate (five-video graph pilot): **{('READY' if readiness['ready_for_next_stage_five_video_graph_pilot'] else 'BLOCKED')}**", '- 570-video graph build: **NOT AUTHORIZED / NOT READY**', '- Graph-only QA: **NOT READY**', '', f"Pilot local IDs: `{dry_run['selected_local_sequential_ids']}`", f"Pilot official video IDs: `{dry_run['selected_video_ids']}`", f"Pilot halves: `{dry_run['selected_halves']}`", f"Pilot runtime questions: `{dry_run['runtime_question_count']}`", f"Runtime contains gold label: `{dry_run['runtime_contains_gold_label']}`", '', 'The next stage must implement and audit the isolated `nextqa_temporal_canonical_v1`', 'five-video graph build before any 570-video build is permitted.', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
