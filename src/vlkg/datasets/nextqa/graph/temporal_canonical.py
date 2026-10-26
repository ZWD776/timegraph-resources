"""NExT-QA wrapper for the shared CG6 temporal and time-fact graph core."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from vlkg.core.config.config import DomainConfig, load_domain_config
from vlkg.datasets.nextqa.runtime.contracts import ALLOWED_GRAPH_PROFILES
from vlkg.datasets.nextqa.graph.hsgg_input_adapter import NormalizedVideoInput
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json
from vlkg.core.transformers.json_to_graph import JSONToGraphTransformer
from vlkg.core.transformers.time_fact_core_pipeline import build_time_fact_core_graph
HUMAN_LABELS = {'person', 'people', 'man', 'woman', 'boy', 'girl', 'child', 'player'}
CANONICAL_OPTIONS = {'compact_scene_label_entities': False, 'compact_scene_label_entity_segments': False, 'stable_identity_merge_mode': 'none', 'stable_identity_include_fragments': True, 'stable_identity_include_noise': True, 'stable_core_compact_scene_label_entities': False}

class DenyLLMFallbackResolver:
    """Fail closed if graph construction unexpectedly attempts an API call."""

    def resolve(self, *, request: object) -> None:
        raise RuntimeError('NExT-QA graph build attempted LLM fallback')

    def get_stats(self) -> dict[str, Any]:
        return {'backend': 'denied', 'request_count': 0}

@dataclass(frozen=True)
class BuiltVideoGraph:
    official_video_id: str
    local_sequential_id: int
    temporal_graph_path: Path
    canonical_graph_path: Path
    canonical_report_path: Path
    audit_path: Path
    audit: dict[str, Any]

def load_nextqa_graph_config(path: str | Path, *, config_dir: str | Path) -> DomainConfig:
    config = load_domain_config(path, config_dir=config_dir)
    config = config.model_copy(update={'component_labels': [], 'component_host_labels': []})
    if config.profile_name not in ALLOWED_GRAPH_PROFILES:
        raise ValueError(f'unexpected NExT-QA graph profile: {config.profile_name}')
    if not config.association.enable_visual_embeddings:
        raise ValueError('NExT-QA CG6 construction requires visual embeddings')
    if config.association.identity_role_strategy != 'tracklet':
        raise ValueError('NExT-QA CG6 construction requires tracklet identity roles')
    if not config.association.local_tracklet_mode:
        raise ValueError('NExT-QA CG6 construction requires local tracklet association')
    if config.component_label_set:
        raise ValueError('NExT-QA CG6 must retain component observations')
    if config.llm_fallback.enabled:
        raise ValueError('NExT-QA graph construction must disable LLM fallback')
    if config.graph_qa.enabled:
        raise ValueError('NExT-QA graph construction must not enable graph QA')
    if config.enable_relation_predicate_mapping:
        raise ValueError('NExT-QA graph profile must preserve raw HSGG predicates')
    if config.enable_semantic_relation_assignment_cleanup:
        raise ValueError('NExT-QA graph profile must preserve raw relation assignment')
    if not config.enable_structured_state_extraction:
        raise ValueError('NExT-QA graph profile must extract structured state signals')
    return config

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def _count_input_signals(input_dir: Path) -> dict[str, int]:
    objects = _load_json(input_dir / 'objects.json')
    attributes = _load_json(input_dir / 'attributes.json')
    relations = _load_json(input_dir / 'relations.json')
    return {'object_observations': sum((len(item.get('labels', [])) for item in objects.values())), 'attribute_records': sum((len(item) for item in attributes.values())), 'relation_records': sum((len(item) for item in relations.values()))}

def _audit_object_retention(input_dir: Path, *, temporal: Any) -> dict[str, Any]:
    objects = _load_json(input_dir / 'objects.json')
    observed = {(observation.frame_name, observation.object_index) for observation in temporal.observations}
    missing: list[dict[str, Any]] = []
    raw_count = 0
    for frame_name, payload in objects.items():
        for object_index, raw_label in enumerate(payload.get('labels', [])):
            raw_count += 1
            if (frame_name, object_index) not in observed:
                missing.append({'frame_name': frame_name, 'object_index': object_index, 'raw_label': raw_label})
    return {'raw_object_count': raw_count, 'observed_object_count': len(observed), 'component_policy_excluded_count': 0, 'unexpected_unobserved_object_count': len(missing), 'unexpected_unobserved_object_examples': missing[:20]}

def _audit_relation_disposition(input_dir: Path, *, temporal: Any) -> dict[str, Any]:
    relations = _load_json(input_dir / 'relations.json')
    observations = {(observation.frame_name, observation.object_index): observation for observation in temporal.observations}
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_name, frame_relations in relations.items():
        for relation in frame_relations:
            disposition = 'retained'
            indices = relation.get('idx')
            if not isinstance(indices, list) or len(indices) != 2:
                disposition = 'malformed_index'
            else:
                try:
                    subject = observations.get((frame_name, int(indices[0])))
                    object_ = observations.get((frame_name, int(indices[1])))
                except (TypeError, ValueError):
                    subject = None
                    object_ = None
                if subject is None or object_ is None:
                    disposition = 'missing_endpoint'
                elif subject.is_noise or object_.is_noise:
                    disposition = 'noise_endpoint'
                elif subject.entity_id == object_.entity_id:
                    disposition = 'self_entity'
            counts[disposition] += 1
            if disposition != 'retained' and len(examples[disposition]) < 10:
                examples[disposition].append({'frame_name': frame_name, 'idx': indices, 'predicate': relation.get('predicate')})
    return {'source_relation_count': sum(counts.values()), 'disposition_counts': dict(sorted(counts.items())), 'non_retained_examples': dict(sorted(examples.items())), 'accounted_relation_count': sum(counts.values())}

def _forbidden_runtime_keys(value: Any, *, prefix: str='') -> list[str]:
    forbidden = {'gold', 'gold_choice_index', 'correct_answer', 'answer_index'}
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f'{prefix}.{key}' if prefix else str(key)
            if str(key).lower() in forbidden:
                found.append(child_path)
            found.extend(_forbidden_runtime_keys(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_runtime_keys(child, prefix=f'{prefix}[{index}]'))
    return found

def _tracklet_statistics(temporal: Any) -> dict[str, Any]:
    observations_by_entity: dict[str, list[Any]] = defaultdict(list)
    for observation in temporal.observations:
        observations_by_entity[observation.entity_id].append(observation)
    lengths = sorted((len(items) for items in observations_by_entity.values()))
    skipped_bridge_count = 0
    for observations in observations_by_entity.values():
        ordered = sorted(observations, key=lambda item: item.frame_rank)
        skipped_bridge_count += sum((right.frame_rank - left.frame_rank > 1 for left, right in zip(ordered, ordered[1:])))
    histogram = Counter(lengths)
    return {'tracklet_count': len(lengths), 'multi_observation_tracklet_count': sum((length > 1 for length in lengths)), 'longest_tracklet_observations': max(lengths, default=0), 'mean_tracklet_observations': round(sum(lengths) / len(lengths), 6) if lengths else 0.0, 'tracklet_length_histogram': {str(length): count for length, count in sorted(histogram.items())}, 'skipped_slot_bridge_count': skipped_bridge_count}

def _build_video_audit(normalized: NormalizedVideoInput, *, temporal: Any, canonical: Any, canonical_report: Mapping[str, Any], transformer_stats: Mapping[str, Any]) -> dict[str, Any]:
    timeline_payload = _load_json(normalized.timeline_path)
    timeline = timeline_payload['frames']
    input_signals = _count_input_signals(normalized.input_dir)
    object_retention = _audit_object_retention(normalized.input_dir, temporal=temporal)
    relation_disposition = _audit_relation_disposition(normalized.input_dir, temporal=temporal)
    tracklets = _tracklet_statistics(temporal)
    event_types = Counter((event.event_type for event in temporal.transition_events))
    canonical_event_types = Counter((event.event_type for event in canonical.transition_event_nodes))
    episode_kinds = Counter((episode.episode_kind for episode in canonical.episode_nodes))
    edge_types = Counter((edge.edge_type for edge in canonical.edges))
    entity_labels = {entity.entity_id: entity.category for entity in temporal.entities}
    actor_object_relation_episodes = sum((1 for episode in temporal.relation_episodes if entity_labels.get(episode.subject_entity_id) in HUMAN_LABELS or entity_labels.get(episode.object_entity_id) in HUMAN_LABELS))
    relation_support = sum((episode.support_frames for episode in temporal.relation_episodes)) + event_types.get('RELATION_OBSERVED', 0)
    raw_state_value_count = sum((len(observation.raw_state_values) for observation in temporal.observations))
    structured_state_value_count = sum((len(observation.state_values) for observation in temporal.observations))
    timestamps = [float(frame['timestamp_sec']) for frame in timeline]
    frame_indices = [int(frame['frame_index']) for frame in timeline]
    skipped_slots = [frame for frame in timeline if not frame['has_scene_graph']]
    cache_stats = dict(transformer_stats.get('embedding_cache', {}))
    alignment = dict(transformer_stats.get('alignment', {}))
    association_path_counts = {'local_tracklet': int(alignment.get('fast_path_hits', 0)), 'new_entity': int(alignment.get('new_entity_creations', 0)), 'slow': int(alignment.get('slow_path_hits', 0)), 'fallback': int(alignment.get('fallback_path_hits', 0)), 'reject': int(alignment.get('reject_path_hits', 0))}
    output_for_purity = {'temporal': temporal.model_dump(mode='json'), 'canonical': canonical.model_dump(mode='json'), 'canonical_report': dict(canonical_report), 'timeline': timeline_payload}
    purity_hits = _forbidden_runtime_keys(output_for_purity)
    relation_counts = relation_disposition['disposition_counts']
    checks = {'sampled_frame_count_preserved': temporal.frame_count == normalized.sampled_frame_count == len(timeline), 'completed_and_skipped_partition_timeline': normalized.completed_frame_count + normalized.skipped_frame_count == normalized.sampled_frame_count, 'all_selected_frames_materialized': normalized.materialized_frame_count == normalized.sampled_frame_count and normalized.materialized_frame_root is not None, 'skipped_placeholders_preserved': len(skipped_slots) == normalized.skipped_frame_count, 'skipped_slots_do_not_bridge_tracklets': tracklets['skipped_slot_bridge_count'] == 0, 'frame_indices_strictly_monotonic': all((left < right for left, right in zip(frame_indices, frame_indices[1:]))), 'timestamps_strictly_monotonic': all((left < right for left, right in zip(timestamps, timestamps[1:]))), 'timeline_endpoints_match_manifest': bool(timeline) and timestamps[0] == normalized.start_timestamp_seconds and (timestamps[-1] == normalized.end_timestamp_seconds), 'all_objects_retained_as_observations': object_retention['observed_object_count'] == input_signals['object_observations'] and object_retention['unexpected_unobserved_object_count'] == 0, 'component_projection_disabled': object_retention['component_policy_excluded_count'] == 0, 'all_relations_accounted': relation_disposition['source_relation_count'] == input_signals['relation_records'] == relation_disposition['accounted_relation_count'], 'no_static_or_component_relation_loss': relation_counts.get('missing_endpoint', 0) == 0, 'siglip_requests_cover_all_observations': int(cache_stats.get('requests', -1)) == len(temporal.observations), 'all_entities_are_dynamic_tracklets_except_noise': all((not entity.is_static for entity in temporal.entities)), 'association_paths_cover_observations': association_path_counts['local_tracklet'] + association_path_counts['new_entity'] + association_path_counts['slow'] + association_path_counts['fallback'] == len(temporal.observations), 'time_fact_core_contract': canonical_report.get('contract') == 'time_fact_core_hsgg_to_graph_pipeline_v1', 'stable_identity_is_one_per_temporal_entity': len(canonical.stable_identity_nodes) == len(temporal.entities), 'llm_fallback_disabled': transformer_stats.get('llm_fallback', {}).get('request_count') == 0, 'gold_purity': not purity_hits}
    return {'schema_version': 'nextqa_graph_video_audit_v2', 'official_video_id': normalized.official_video_id, 'episode_key': normalized.episode_key, 'local_sequential_id': normalized.local_sequential_id, 'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed], 'input': {'sampled_frame_count': normalized.sampled_frame_count, 'completed_hsgg_frame_count': normalized.completed_frame_count, 'skipped_frame_count': normalized.skipped_frame_count, 'materialized_frame_count': normalized.materialized_frame_count, **input_signals}, 'temporal_graph': {'frame_count': temporal.frame_count, 'observation_count': len(temporal.observations), 'entity_count': len(temporal.entities), 'raw_attribute_value_count': raw_state_value_count, 'structured_state_value_count': structured_state_value_count, 'state_episode_count': len(temporal.state_episodes), 'relation_episode_count': len(temporal.relation_episodes), 'bound_relation_support_count': relation_support, 'relation_source_disposition': relation_disposition, 'object_retention': object_retention, 'scene_segment_count': len(temporal.scene_segments), 'entity_segment_count': len(temporal.entity_segments), 'transition_event_count': len(temporal.transition_events), 'transition_event_type_counts': dict(sorted(event_types.items())), 'actor_object_relation_episode_count': actor_object_relation_episodes, 'tracklets': tracklets, 'association_path_counts': association_path_counts}, 'canonical_graph': {'profile': 'time_fact_core', 'contract': canonical_report.get('contract'), 'node_count': canonical.node_count, 'edge_count': canonical.edge_count, 'scene_node_count': len(canonical.scene_nodes), 'entity_node_count': len(canonical.entity_nodes), 'stable_identity_node_count': len(canonical.stable_identity_nodes), 'state_value_node_count': len(canonical.state_value_nodes), 'interaction_type_node_count': len(canonical.interaction_type_nodes), 'episode_kind_counts': dict(sorted(episode_kinds.items())), 'transition_event_type_counts': dict(sorted(canonical_event_types.items())), 'edge_type_counts': dict(sorted(edge_types.items())), 'profile_sizes': canonical_report.get('sizes', {})}, 'runtime_stats': dict(transformer_stats), 'gold_purity_forbidden_key_hits': purity_hits, 'construction_vs_qa_purity': {'uses_visual_embeddings_for_graph_construction': True, 'qa_reads_rgb': False, 'qa_reads_object_crop': False, 'qa_reads_embedding_cache': False, 'qa_uses_faiss': False}, 'sampling_limitation': {'source': 'all_available_every30_hsgg_frames', 'uses_uniform32': False, 'timestamp_source': 'frame_filename_fps_derived_nominal_seconds', 'short_event_risk': 'Events shorter than the per-video every-30 interval may be absent from HSGG; the graph cannot recover evidence not present in sampled records.'}}

def build_video_graph(normalized: NormalizedVideoInput, *, config: DomainConfig, output_root: str | Path, vision_feature_extractor: Any, embedding_cache: Any) -> BuiltVideoGraph:
    """Build one question-independent local-tracklet temporal/canonical graph."""
    if not config.association.enable_visual_embeddings:
        raise ValueError('visual embeddings are required during CG6 graph construction')
    if normalized.materialized_frame_root is None:
        raise ValueError('CG6 graph construction requires materialized sampled frames')
    video_root = Path(output_root).resolve() / f'{normalized.local_sequential_id:04d}_{normalized.official_video_id}'
    transformer = JSONToGraphTransformer(input_dir=normalized.input_dir, video_id=normalized.episode_key, config=config, frame_image_root=normalized.materialized_frame_root, vision_feature_extractor=vision_feature_extractor, embedding_cache=embedding_cache, llm_fallback_resolver=DenyLLMFallbackResolver())
    temporal = transformer.transform()
    transformer_stats = transformer.get_runtime_stats()
    temporal = transformer.apply_garbage_collection(temporal)
    temporal = temporal.filtered_copy(exclude_noise=config.export.exclude_noise, exclude_fragments=config.export.exclude_fragments, keep_fragment_labels=config.export.normalized_protected_fragment_labels)
    temporal_path = video_root / 'temporal_graph.json'
    atomic_write_json(temporal_path, temporal.model_dump(mode='json'))
    canonical_result = build_time_fact_core_graph(temporal, source_graph_path=temporal_path, **CANONICAL_OPTIONS)
    canonical = canonical_result.graph
    canonical_path = video_root / 'canonical_temporal_memory_graph.json'
    atomic_write_json(canonical_path, canonical.model_dump(mode='json'))
    canonical_report_path = video_root / 'time_fact_core_report.json'
    atomic_write_json(canonical_report_path, canonical_result.report)
    audit = _build_video_audit(normalized, temporal=temporal, canonical=canonical, canonical_report=canonical_result.report, transformer_stats=transformer_stats)
    audit_path = video_root / 'video_graph_audit.json'
    atomic_write_json(audit_path, audit)
    return BuiltVideoGraph(official_video_id=normalized.official_video_id, local_sequential_id=normalized.local_sequential_id, temporal_graph_path=temporal_path, canonical_graph_path=canonical_path, canonical_report_path=canonical_report_path, audit_path=audit_path, audit=audit)
