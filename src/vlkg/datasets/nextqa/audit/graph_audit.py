"""Aggregate reporting for the fixed NExT-QA CG6 graph pilot."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, atomic_write_text
from vlkg.datasets.nextqa.graph.temporal_canonical import BuiltVideoGraph

def build_graph_manifest(*, run_id: str, source_manifest_path: str | Path, config_path: str | Path, graph_profile: str, run_root: str | Path, built: Iterable[BuiltVideoGraph], errors: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [{'official_video_id': item.official_video_id, 'local_sequential_id': item.local_sequential_id, 'status': 'completed', 'temporal_graph_path': str(item.temporal_graph_path), 'canonical_graph_path': str(item.canonical_graph_path), 'canonical_report_path': str(item.canonical_report_path), 'video_audit_path': str(item.audit_path)} for item in built]
    rows.extend((dict(error) for error in errors))
    rows.sort(key=lambda item: int(item['local_sequential_id']))
    return {'schema_version': 'nextqa_graph_build_manifest_v2', 'dataset': 'NExT-QA', 'split': 'val', 'run_id': run_id, 'graph_profile': graph_profile, 'canonical_profile': 'time_fact_core', 'source_manifest_path': str(Path(source_manifest_path).resolve()), 'config_path': str(Path(config_path).resolve()), 'run_root': str(Path(run_root).resolve()), 'uses_gold_answer': False, 'reads_questions': False, 'uses_visual_embeddings_for_graph_construction': True, 'qa_reads_rgb': False, 'qa_reads_object_crop': False, 'qa_reads_embedding_cache': False, 'qa_uses_faiss': False, 'uses_uniform32': False, 'video_count': len(rows), 'completed_video_count': sum((row['status'] == 'completed' for row in rows)), 'error_video_count': sum((row['status'] == 'error' for row in rows)), 'videos': rows}

def build_graph_audit(*, run_id: str, graph_profile: str, built: Iterable[BuiltVideoGraph], errors: Iterable[Mapping[str, Any]], source_drift: Mapping[str, Any], expected_counts: Mapping[str, int] | None=None) -> dict[str, Any]:
    built_items = list(built)
    error_items = [dict(item) for item in errors]
    failed_video_audits = [item.official_video_id for item in built_items if item.audit['status'] != 'passed']
    transition_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    association_counts: Counter[str] = Counter()
    relation_disposition_counts: Counter[str] = Counter()
    tracklet_length_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for item in built_items:
        temporal = item.audit['temporal_graph']
        canonical = item.audit['canonical_graph']
        tracklets = temporal['tracklets']
        transition_counts.update(temporal['transition_event_type_counts'])
        edge_counts.update(canonical['edge_type_counts'])
        association_counts.update(temporal['association_path_counts'])
        relation_disposition_counts.update(temporal['relation_source_disposition']['disposition_counts'])
        tracklet_length_counts.update(tracklets['tracklet_length_histogram'])
        for key in ('frame_count', 'observation_count', 'entity_count', 'raw_attribute_value_count', 'structured_state_value_count', 'state_episode_count', 'relation_episode_count', 'bound_relation_support_count', 'scene_segment_count', 'entity_segment_count', 'transition_event_count', 'actor_object_relation_episode_count'):
            totals[key] += int(temporal[key])
        totals['multi_observation_tracklet_count'] += int(tracklets['multi_observation_tracklet_count'])
        totals['skipped_slot_bridge_count'] += int(tracklets['skipped_slot_bridge_count'])
        totals['canonical_node_count'] += int(canonical['node_count'])
        totals['canonical_edge_count'] += int(canonical['edge_count'])
        totals['stable_identity_node_count'] += int(canonical['stable_identity_node_count'])
    expected = dict(expected_counts or {'videos': 5, 'sampled_frames': 179, 'completed_hsgg_frames': 178, 'skipped_frames': 1})
    full_run = expected['videos'] == 570
    completion_check_name = 'full_validation_videos_completed' if full_run else 'fixed_five_videos_completed'
    coverage_check_name = 'full_coverage_matches_manifest' if full_run else 'pilot_coverage_matches_manifest'
    expected_coverage_matches = len(built_items) == expected['videos'] and sum((item.audit['input']['sampled_frame_count'] for item in built_items)) == expected['sampled_frames'] and (sum((item.audit['input']['completed_hsgg_frame_count'] for item in built_items)) == expected['completed_hsgg_frames']) and (sum((item.audit['input']['skipped_frame_count'] for item in built_items)) == expected['skipped_frames'])
    checks = {completion_check_name: len(built_items) == expected['videos'] and (not error_items), 'zero_build_errors': not error_items, 'all_video_audits_passed': not failed_video_audits, coverage_check_name: expected_coverage_matches, 'source_code_stable_during_run': not bool(source_drift.get('code_drift_detected')), 'construction_visual_use_isolated_from_graph_only_qa': all((item.audit['construction_vs_qa_purity'] == {'uses_visual_embeddings_for_graph_construction': True, 'qa_reads_rgb': False, 'qa_reads_object_crop': False, 'qa_reads_embedding_cache': False, 'qa_uses_faiss': False} for item in built_items)), 'all_objects_retained_without_component_projection': all((item.audit['checks']['all_objects_retained_as_observations'] and item.audit['checks']['component_projection_disabled'] for item in built_items)), 'time_fact_core_one_identity_per_tracklet': all((item.audit['checks']['time_fact_core_contract'] and item.audit['checks']['stable_identity_is_one_per_temporal_entity'] for item in built_items)), 'skipped_slots_do_not_bridge_tracklets': all((item.audit['checks']['skipped_slots_do_not_bridge_tracklets'] for item in built_items)), 'gold_and_api_purity': all((item.audit['checks']['llm_fallback_disabled'] and item.audit['checks']['gold_purity'] for item in built_items))}
    expected_sampled = sum((item.audit['input']['sampled_frame_count'] for item in built_items))
    expected_completed = sum((item.audit['input']['completed_hsgg_frame_count'] for item in built_items))
    expected_skipped = sum((item.audit['input']['skipped_frame_count'] for item in built_items))
    return {'schema_version': 'nextqa_graph_build_audit_v2', 'run_id': run_id, 'graph_profile': graph_profile, 'status': 'passed' if all(checks.values()) else 'failed', 'valid_for_next_gate': all(checks.values()), 'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed], 'failed_video_audits': failed_video_audits, 'errors': error_items, 'coverage': {'video_count': len(built_items), 'sampled_frame_count': expected_sampled, 'completed_hsgg_frame_count': expected_completed, 'skipped_frame_count': expected_skipped, 'expected_counts': expected, 'matches_expected_counts': expected_coverage_matches}, 'totals': dict(sorted(totals.items())), 'association_path_counts': dict(sorted(association_counts.items())), 'tracklet_length_histogram': dict(sorted(tracklet_length_counts.items(), key=lambda item: int(item[0]))), 'relation_source_disposition_counts': dict(sorted(relation_disposition_counts.items())), 'transition_event_type_counts': dict(sorted(transition_counts.items())), 'canonical_edge_type_counts': dict(sorted(edge_counts.items())), 'transition_signal_review': {'transition_events_present': totals['transition_event_count'] > 0, 'state_transitions_present': transition_counts['STATE_TRANSITION'] > 0, 'relation_boundary_events_present': transition_counts['RELATION_START'] > 0 or transition_counts['RELATION_END'] > 0, 'interpretation': 'Low counts can reflect sparse one-sample-per-interval HSGG evidence; the audit reports this without question- or answer-driven graph changes.'}, 'source_drift': dict(source_drift), 'limitations': ['Frame times are per-video FPS-derived nominal seconds encoded in filenames, not container PTS.', 'Every-30 sampling can miss events shorter than the per-video sampling interval.', 'A skipped HSGG frame remains an empty time slot and cannot bridge a local tracklet.', 'SigLIP crops are used only for graph-construction association, not graph-only QA.', 'Graph construction is question-independent; no question, choice, type, or evaluator label is read.']}

def render_graph_audit_markdown(audit: Mapping[str, Any]) -> str:
    coverage = audit['coverage']
    totals = audit['totals']
    expected_videos = int(coverage.get('expected_counts', {}).get('videos', 5))
    title = '# NExT-QA Full Validation Graph Build Audit' if expected_videos == 570 else '# NExT-QA Fixed Five-Video Graph Pilot Audit'
    lines = [title, '', f"- Run ID: `{audit['run_id']}`", f"- Status: `{audit['status']}`", f"- Ready for next gate: `{str(audit['valid_for_next_gate']).lower()}`", f"- Videos: `{coverage['video_count']}/{expected_videos}`", f"- sampled / HSGG / skipped: `{coverage['sampled_frame_count']} / {coverage['completed_hsgg_frame_count']} / {coverage['skipped_frame_count']}`", f"- observations / entities: `{totals.get('observation_count', 0)} / {totals.get('entity_count', 0)}`", f"- relation episodes / state episodes: `{totals.get('relation_episode_count', 0)} / {totals.get('state_episode_count', 0)}`", f"- transition events: `{totals.get('transition_event_count', 0)}`", f"- canonical nodes / edges: `{totals.get('canonical_node_count', 0)} / {totals.get('canonical_edge_count', 0)}`", '', '## Gate Checks', '']
    for name, passed in audit['checks'].items():
        lines.append(f"- {name}: `{('passed' if passed else 'failed')}`")
    lines.extend(['', '## Known Limitations', ''])
    lines.extend((f'- {item}' for item in audit['limitations']))
    if audit['errors']:
        lines.extend(['', '## Build Errors', ''])
        lines.extend((f"- `{item['official_video_id']}`: {item['error_type']}: {item['error_message']}" for item in audit['errors']))
    return '\n'.join(lines) + '\n'

def persist_graph_reports(run_root: str | Path, *, graph_manifest: Mapping[str, Any], graph_audit: Mapping[str, Any]) -> None:
    root = Path(run_root).resolve()
    atomic_write_json(root / 'nextqa_graph_build_manifest.json', dict(graph_manifest))
    atomic_write_json(root / 'graph_build_audit.json', dict(graph_audit))
    atomic_write_text(root / 'graph_build_audit.md', render_graph_audit_markdown(graph_audit))
