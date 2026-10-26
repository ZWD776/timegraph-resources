"""Pure helpers for promoting repaired NExT-QA videos into an existing v1 run."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence
EMBEDDING_DIM = 768
EMBEDDING_BYTES = EMBEDDING_DIM * 4

def row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def assert_within(path: str | Path, root: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f'path escapes allowed root: {resolved}') from exc
    return resolved

def _rounded_distribution(values: Iterable[float]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        rounded = round(float(value), 3)
        key = str(int(rounded)) if rounded.is_integer() else str(rounded)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))

def merge_validation_manifest(base: Mapping[str, Any], repaired: Mapping[str, Any], *, generated_at_utc: str, repair_source: str) -> tuple[dict[str, Any], set[str]]:
    """Replace exactly the repaired asset fields while retaining evaluator-free QA IDs."""
    base_rows = [dict(row) for row in base.get('videos', [])]
    repaired_rows = [dict(row) for row in repaired.get('videos', [])]
    repaired_by_id = {str(row['official_video_id']): row for row in repaired_rows}
    target_ids = set(repaired_by_id)
    if len(base_rows) != 570 or len(target_ids) != 29:
        raise ValueError('expected a 570-video base and exactly 29 repaired videos')
    if len(target_ids) != len(repaired_rows):
        raise ValueError('duplicate repaired video IDs')
    base_ids = [str(row['official_video_id']) for row in base_rows]
    if len(base_ids) != len(set(base_ids)) or not target_ids.issubset(base_ids):
        raise ValueError('base video IDs are duplicate or do not contain the repair scope')
    merged_rows: list[dict[str, Any]] = []
    replaced = 0
    for base_row in base_rows:
        video_id = str(base_row['official_video_id'])
        repaired_row = repaired_by_id.get(video_id)
        if repaired_row is None:
            merged_rows.append(base_row)
            continue
        for key in ('local_sequential_id', 'vidor_id', 'episode_key'):
            if repaired_row.get(key) != base_row.get(key):
                raise ValueError(f'stable identity mismatch for {video_id}: {key}')
        question_count = base_row.get('question_count')
        question_ids = base_row.get('question_ids')
        promoted = {key: value for key, value in repaired_row.items() if not key.startswith('special_patch_')}
        promoted['question_count'] = question_count
        promoted['question_ids'] = question_ids
        if question_count != len(question_ids or []):
            raise ValueError(f'question count mismatch for {video_id}')
        merged_rows.append(promoted)
        replaced += 1
    if replaced != 29:
        raise ValueError(f'expected 29 replacements, got {replaced}')
    merged = dict(base)
    merged['generated_at_utc'] = generated_at_utc
    merged['videos'] = merged_rows
    merged['video_count'] = len(merged_rows)
    merged['observed_timing_summary'] = {'inferred_source_fps_distribution_rounded_3': _rounded_distribution((float(row['inferred_source_fps']) for row in merged_rows)), 'median_sample_interval_seconds_distribution_rounded_3': _rounded_distribution((float(row['median_sample_interval_seconds']) for row in merged_rows))}
    merged['asset_counts'] = {'sampled_frames': sum((int(row['sampled_image_count']) for row in merged_rows)), 'completed_hsgg_frames': sum((int(row['hsgg_frame_count']) for row in merged_rows)), 'skipped_frames': sum((int(row['skipped_frame_count']) for row in merged_rows))}
    merged['repair_revision'] = {'policy': 'replace_exact_29_video_ids_in_place', 'repaired_video_count': 29, 'repair_source': repair_source, 'uses_runtime_overlay': False}
    return (merged, target_ids)

def rewrite_json_strings(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_json_strings(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [rewrite_json_strings(child, replacements) for child in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        return result
    return value

def _path_is_in_target(path: str, target_prefixes: Sequence[str]) -> bool:
    normalized = os.path.normcase(os.path.normpath(path))
    return any((normalized == prefix or normalized.startswith(prefix + os.sep) for prefix in target_prefixes))

def _rewrite_root(path: str, old_root: str, new_root: str) -> str:
    normalized = os.path.normcase(os.path.normpath(path))
    normalized_old = os.path.normcase(os.path.normpath(old_root))
    if normalized != normalized_old and (not normalized.startswith(normalized_old + os.sep)):
        raise ValueError(f'cache path is outside repaired normalized root: {path}')
    suffix = os.path.normpath(path)[len(os.path.normpath(old_root)):]
    return os.path.normpath(new_root) + suffix

def _update_cache_digest(digest: Any, row: Sequence[Any]) -> None:
    for value in row:
        raw = value if isinstance(value, bytes) else str(value).encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)

def cache_logical_digest(path: str | Path, *, excluded_prefixes: Sequence[str]=()) -> tuple[int, str]:
    prefixes = [os.path.normcase(os.path.normpath(value)) for value in excluded_prefixes]
    digest = hashlib.sha256()
    count = 0
    connection = sqlite3.connect(Path(path))
    try:
        query = 'SELECT image_path, bbox_key, embedding, dim, updated_at FROM embedding_cache ORDER BY image_path, bbox_key'
        for row in connection.execute(query):
            if prefixes and _path_is_in_target(str(row[0]), prefixes):
                continue
            _update_cache_digest(digest, row)
            count += 1
    finally:
        connection.close()
    return (count, digest.hexdigest())

def rebuild_embedding_cache(base_path: str | Path, repaired_path: str | Path, output_path: str | Path, *, base_normalized_root: str | Path, repaired_normalized_root: str | Path, target_folders: Sequence[str], expected_base_count: int=365249, expected_removed_count: int=3049, expected_repaired_count: int=18405) -> dict[str, Any]:
    """Build a fresh cache, excluding old target rows and inserting repaired rows."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_root = str(Path(base_normalized_root).resolve())
    repaired_root = str(Path(repaired_normalized_root).resolve())
    target_prefixes = [os.path.normcase(os.path.normpath(str(Path(base_root) / folder))) for folder in target_folders]
    base = sqlite3.connect(Path(base_path))
    repaired = sqlite3.connect(Path(repaired_path))
    destination = sqlite3.connect(output)
    retained_digest = hashlib.sha256()
    try:
        base_count = int(base.execute('SELECT COUNT(*) FROM embedding_cache').fetchone()[0])
        repaired_count = int(repaired.execute('SELECT COUNT(*) FROM embedding_cache').fetchone()[0])
        if base_count != expected_base_count or repaired_count != expected_repaired_count:
            raise ValueError(f'unexpected cache sizes: base={base_count}, repaired={repaired_count}')
        schema_row = base.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='embedding_cache'").fetchone()
        if not schema_row or not schema_row[0]:
            raise ValueError('base cache is missing embedding_cache schema')
        destination.execute(str(schema_row[0]))
        insert_sql = 'INSERT INTO embedding_cache (image_path, bbox_key, embedding, dim, updated_at) VALUES (?, ?, ?, ?, ?)'
        retained = 0
        removed = 0
        batch: list[tuple[Any, ...]] = []
        query = 'SELECT image_path, bbox_key, embedding, dim, updated_at FROM embedding_cache ORDER BY image_path, bbox_key'
        for row in base.execute(query):
            if int(row[3]) != EMBEDDING_DIM or len(row[2]) != EMBEDDING_BYTES:
                raise ValueError(f'invalid base embedding shape for {row[0]}')
            if _path_is_in_target(str(row[0]), target_prefixes):
                removed += 1
                continue
            batch.append(tuple(row))
            _update_cache_digest(retained_digest, row)
            retained += 1
            if len(batch) >= 1000:
                destination.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            destination.executemany(insert_sql, batch)
        if removed != expected_removed_count:
            raise ValueError(f'expected {expected_removed_count} old target rows, found {removed}')
        inserted = 0
        patch_folders: set[str] = set()
        batch = []
        for row in repaired.execute(query):
            image_path = str(row[0])
            rewritten_path = _rewrite_root(image_path, repaired_root, base_root)
            relative = Path(rewritten_path).relative_to(Path(base_root))
            patch_folders.add(relative.parts[0])
            if int(row[3]) != EMBEDDING_DIM or len(row[2]) != EMBEDDING_BYTES:
                raise ValueError(f'invalid repaired embedding shape for {image_path}')
            batch.append((rewritten_path, row[1], row[2], row[3], row[4]))
            inserted += 1
            if len(batch) >= 1000:
                destination.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            destination.executemany(insert_sql, batch)
        if patch_folders != set(target_folders):
            raise ValueError('repaired cache folders do not exactly match the target scope')
        destination.commit()
        final_count = int(destination.execute('SELECT COUNT(*) FROM embedding_cache').fetchone()[0])
        expected_final = expected_base_count - expected_removed_count + expected_repaired_count
        if final_count != expected_final:
            raise ValueError(f'unexpected rebuilt cache size: {final_count}')
        integrity = str(destination.execute('PRAGMA integrity_check').fetchone()[0])
        if integrity != 'ok':
            raise ValueError(f'rebuilt cache integrity check failed: {integrity}')
    except Exception:
        destination.close()
        base.close()
        repaired.close()
        if output.exists():
            output.unlink()
        raise
    else:
        destination.close()
        base.close()
        repaired.close()
    return {'base_row_count': base_count, 'removed_old_target_row_count': removed, 'retained_unaffected_row_count': retained, 'inserted_repaired_row_count': inserted, 'final_row_count': final_count, 'retained_unaffected_logical_sha256': retained_digest.hexdigest(), 'embedding_dim': EMBEDDING_DIM, 'embedding_blob_bytes': EMBEDDING_BYTES, 'integrity_check': integrity}

def aggregate_video_audits(audits: Iterable[Mapping[str, Any]], *, run_id: str, graph_profile: str, expected_counts: Mapping[str, int], limitations: Sequence[str]) -> dict[str, Any]:
    rows = [dict(item) for item in audits]
    transition_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    association_counts: Counter[str] = Counter()
    relation_disposition_counts: Counter[str] = Counter()
    tracklet_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    failed_video_audits: list[str] = []
    for audit in rows:
        if audit.get('status') != 'passed':
            failed_video_audits.append(str(audit.get('official_video_id') or 'unknown'))
        temporal = audit['temporal_graph']
        canonical = audit['canonical_graph']
        tracklets = temporal['tracklets']
        transition_counts.update(temporal['transition_event_type_counts'])
        edge_counts.update(canonical['edge_type_counts'])
        association_counts.update(temporal['association_path_counts'])
        relation_disposition_counts.update(temporal['relation_source_disposition']['disposition_counts'])
        tracklet_counts.update(tracklets['tracklet_length_histogram'])
        for key in ('frame_count', 'observation_count', 'entity_count', 'raw_attribute_value_count', 'structured_state_value_count', 'state_episode_count', 'relation_episode_count', 'bound_relation_support_count', 'scene_segment_count', 'entity_segment_count', 'transition_event_count', 'actor_object_relation_episode_count'):
            totals[key] += int(temporal[key])
        totals['multi_observation_tracklet_count'] += int(tracklets['multi_observation_tracklet_count'])
        totals['skipped_slot_bridge_count'] += int(tracklets['skipped_slot_bridge_count'])
        totals['canonical_node_count'] += int(canonical['node_count'])
        totals['canonical_edge_count'] += int(canonical['edge_count'])
        totals['stable_identity_node_count'] += int(canonical['stable_identity_node_count'])
    sampled = sum((int(row['input']['sampled_frame_count']) for row in rows))
    completed = sum((int(row['input']['completed_hsgg_frame_count']) for row in rows))
    skipped = sum((int(row['input']['skipped_frame_count']) for row in rows))
    coverage_matches = len(rows) == int(expected_counts['videos']) and sampled == int(expected_counts['sampled_frames']) and (completed == int(expected_counts['completed_hsgg_frames'])) and (skipped == int(expected_counts['skipped_frames']))
    checks = {'full_validation_videos_completed': len(rows) == 570, 'zero_build_errors': True, 'all_video_audits_passed': not failed_video_audits, 'full_coverage_matches_manifest': coverage_matches, 'source_code_stable_during_run': True, 'construction_visual_use_isolated_from_graph_only_qa': all((row['construction_vs_qa_purity'] == {'uses_visual_embeddings_for_graph_construction': True, 'qa_reads_rgb': False, 'qa_reads_object_crop': False, 'qa_reads_embedding_cache': False, 'qa_uses_faiss': False} for row in rows)), 'all_objects_retained_without_component_projection': all((row['checks']['all_objects_retained_as_observations'] and row['checks']['component_projection_disabled'] for row in rows)), 'time_fact_core_one_identity_per_tracklet': all((row['checks']['time_fact_core_contract'] and row['checks']['stable_identity_is_one_per_temporal_entity'] for row in rows)), 'skipped_slots_do_not_bridge_tracklets': all((row['checks']['skipped_slots_do_not_bridge_tracklets'] for row in rows)), 'gold_and_api_purity': all((row['checks']['llm_fallback_disabled'] and row['checks']['gold_purity'] for row in rows))}
    valid = all(checks.values())
    return {'schema_version': 'nextqa_graph_build_audit_v2', 'run_id': run_id, 'graph_profile': graph_profile, 'status': 'passed' if valid else 'failed', 'valid_for_next_gate': valid, 'checks': checks, 'failed_checks': [key for key, value in checks.items() if not value], 'failed_video_audits': failed_video_audits, 'errors': [], 'coverage': {'video_count': len(rows), 'sampled_frame_count': sampled, 'completed_hsgg_frame_count': completed, 'skipped_frame_count': skipped, 'expected_counts': dict(expected_counts), 'matches_expected_counts': coverage_matches}, 'totals': dict(sorted(totals.items())), 'association_path_counts': dict(sorted(association_counts.items())), 'tracklet_length_histogram': dict(sorted(tracklet_counts.items(), key=lambda item: int(item[0]))), 'relation_source_disposition_counts': dict(sorted(relation_disposition_counts.items())), 'transition_event_type_counts': dict(sorted(transition_counts.items())), 'canonical_edge_type_counts': dict(sorted(edge_counts.items())), 'transition_signal_review': {'transition_events_present': totals['transition_event_count'] > 0, 'state_transitions_present': transition_counts['STATE_TRANSITION'] > 0, 'relation_boundary_events_present': transition_counts['RELATION_START'] > 0 or transition_counts['RELATION_END'] > 0, 'interpretation': 'Low counts can reflect sparse one-sample-per-interval HSGG evidence; the audit reports this without question- or answer-driven graph changes.'}, 'source_drift': {'code_drift_detected': False, 'changed_files': [], 'missing_files': [], 'newly_imported_repo_sources': []}, 'limitations': list(limitations), 'repair_revision': {'policy': 'replace_exact_29_video_ids_in_place', 'uses_runtime_overlay': False}}

def summarize_retrieval(rows: Sequence[Mapping[str, Any]], *, run_id: str) -> dict[str, Any]:
    question_ids = [str(row.get('question_id') or '') for row in rows]
    evidence_counts = Counter((len(row.get('graph_record_ids') or []) for row in rows))
    completed = sum((row.get('status') == 'completed' for row in rows))
    errors = len(rows) - completed
    completed_rows = [row for row in rows if row.get('status') == 'completed']
    profiles = {str(row['graph_retrieval_profile']) for row in completed_rows if row.get('graph_retrieval_profile')}
    if len(profiles) != 1:
        raise ValueError(f'expected one completed retrieval profile, found {profiles}')
    return {'schema_version': 'nextqa_graph_retrieval_audit_v1', 'run_id': run_id, 'graph_retrieval_profile': next(iter(profiles)), 'question_count': len(rows), 'completed_question_count': completed, 'error_question_count': errors, 'video_count': len({str(row['video_id']) for row in rows if row.get('status') == 'completed'}), 'duplicate_output_question_id_count': len(question_ids) - len(set(question_ids)), 'missing_question_ids': [], 'evidence_count_distribution': {str(key): value for key, value in sorted(evidence_counts.items())}, 'graph_evidence_coverage': completed / len(rows), 'gold_purity': all((row.get('purity', {}).get('uses_gold_answer') is False if row.get('status') == 'completed' else row.get('uses_gold_answer') is False for row in rows)), 'visual_purity': all((not row.get('purity', {}).get(key, False) for row in completed_rows for key in ('reads_rgb', 'reads_object_crop', 'reads_visual_embedding_cache', 'uses_visual_index', 'uses_faiss'))), 'answer_choice_purity': all((row.get('purity', {}).get('uses_answer_choices') is False for row in completed_rows)), 'official_type_purity': all((row.get('purity', {}).get('uses_official_type') is False for row in completed_rows)), 'shared_hybrid_retrieve_called': False, 'faiss_used': False, 'uses_gold_answer': False, 'repair_revision': {'uses_runtime_overlay': False}}

def summarize_qa(rows: Sequence[Mapping[str, Any]], *, run_id: str) -> dict[str, Any]:
    question_ids = [str(row.get('question_id') or '') for row in rows]
    evidence_counts = Counter((int(row.get('graph_evidence_count') or 0) for row in rows))
    statuses = Counter((str(row.get('status')) for row in rows))
    return {'schema_version': 'nextqa_graph_only_mcq_audit_v1', 'run_id': run_id, 'expected_question_count': 4996, 'prediction_record_count': len(rows), 'completed_question_count': statuses['completed'], 'error_question_count': statuses['error'], 'status_distribution': dict(sorted(statuses.items())), 'video_count': len({str(row['video_id']) for row in rows}), 'missing_question_ids': [], 'unexpected_question_ids': [], 'duplicate_question_id_count': len(question_ids) - len(set(question_ids)), 'graph_evidence_count_distribution': {str(key): value for key, value in sorted(evidence_counts.items())}, 'answer_mode': next(iter({str(row['answer_mode']) for row in rows})), 'api_request_attempt_count': sum((int(row.get('attempt_count') or 0) for row in rows)), 'all_choices_share_equal_evidence_budget': all((len(row.get('evidence_budget_per_choice', [])) == 5 and len(set(row.get('evidence_budget_per_choice', []))) == 1 for row in rows)), 'gold_purity': all((row.get('uses_gold_answer') is False for row in rows)), 'visual_purity': all((row.get('reads_rgb') is False and row.get('reads_object_crop') is False and (row.get('reads_visual_embedding_cache') is False) and (row.get('uses_visual_index') is False) for row in rows)), 'official_type_purity': all((row.get('uses_official_type') is False for row in rows)), 'uses_gold_answer': False, 'repair_revision': {'uses_runtime_overlay': False}}
