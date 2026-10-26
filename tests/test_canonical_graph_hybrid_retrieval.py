from __future__ import annotations
from pathlib import Path
import uuid
import vlkg.core.graph_retrieval.canonical_hybrid_retrieval as hybrid_retrieval
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import _PROJECTION_CACHE, _build_projection, _format_time_values, run_canonical_graph_hybrid_retrieval, unified_graph_retrieval_question_ir
from vlkg.core.graph_retrieval.canonical_record_vector_recall import VectorRecallHit
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import BoundingBox, Entity, EntitySegment, Observation, RelationEpisode, SceneBaseNode, SceneDeltaNode, SceneDeltaObservation, SceneSegment, StateEpisode, TemporalGraphDocument, TransitionEvent
from vlkg.core.transformers import build_canonical_temporal_memory_graph

def _sample_temporal_graph_document() -> TemporalGraphDocument:
    return TemporalGraphDocument(video_id='video_001', source_dir=Path('data/examples/video_001'), frame_count=20, observations=[], entities=[Entity(entity_id='player_1', video_id='video_001', category='man', first_seen_frame=0, last_seen_frame=19, first_seen_ts=0.0, last_seen_ts=1.9, observation_count=20, node_role='dynamic', states_seen=['dribbling', 'shooting']), Entity(entity_id='hoop_1', video_id='video_001', category='basketball hoop', first_seen_frame=0, last_seen_frame=19, first_seen_ts=0.0, last_seen_ts=1.9, observation_count=20, node_role='static', is_static=True)], relation_episodes=[RelationEpisode(relation_episode_id='rel_1', video_id='video_001', subject_entity_id='player_1', predicate='near', raw_predicate='standing near', object_entity_id='hoop_1', start_frame=12, end_frame=13, start_ts=1.2, end_ts=1.3, support_frames=2)], state_episodes=[StateEpisode(state_episode_id='state_1', video_id='video_001', entity_id='player_1', attr_key='action', attr_value='dribbling', start_frame=1, end_frame=4, start_ts=0.1, end_ts=0.4, support_frames=4), StateEpisode(state_episode_id='state_2', video_id='video_001', entity_id='player_1', attr_key='action', attr_value='shooting', start_frame=11, end_frame=14, start_ts=1.1, end_ts=1.4, support_frames=4)], scene_segments=[SceneSegment(scene_id='scene_001', video_id='video_001', scene_index=1, scene_count=2, start_frame_rank=0, end_frame_rank=9, start_frame=0, end_frame=9, start_ts=0.0, end_ts=0.9, support_frames=10, next_scene_id='scene_002'), SceneSegment(scene_id='scene_002', video_id='video_001', scene_index=2, scene_count=2, start_frame_rank=10, end_frame_rank=19, start_frame=10, end_frame=19, start_ts=1.0, end_ts=1.9, support_frames=10, previous_scene_id='scene_001')], entity_segments=[EntitySegment(segment_id='seg_player_1_a', video_id='video_001', entity_id='player_1', scene_id='scene_001', segment_index=1, segment_count=2, start_frame_rank=0, end_frame_rank=9, start_frame=0, end_frame=9, start_ts=0.0, end_ts=0.9, support_frames=10, next_segment_id='seg_player_1_b'), EntitySegment(segment_id='seg_player_1_b', video_id='video_001', entity_id='player_1', scene_id='scene_002', segment_index=2, segment_count=2, start_frame_rank=10, end_frame_rank=19, start_frame=10, end_frame=19, start_ts=1.0, end_ts=1.9, support_frames=10, previous_segment_id='seg_player_1_a')], transition_events=[TransitionEvent(event_id='event_1', video_id='video_001', event_type='STATE_TRANSITION', frame_rank=11, frame_index=11, timestamp_sec=1.1, entity_id='player_1', attr_key='action', previous_state_value='dribbling', current_state_value='shooting', source_segment_id='seg_player_1_b', source_episode_id='state_2')])

def _sample_temporal_graph_with_observed_relation_burst() -> TemporalGraphDocument:
    document = _sample_temporal_graph_document()
    return document.model_copy(update={'transition_events': [*document.transition_events, TransitionEvent(event_id='obs_1', video_id='video_001', event_type='RELATION_OBSERVED', frame_rank=12, frame_index=12, timestamp_sec=1.2, subject_entity_id='player_1', predicate='near', raw_predicate='standing near', object_entity_id='hoop_1'), TransitionEvent(event_id='obs_2', video_id='video_001', event_type='RELATION_OBSERVED', frame_rank=14, frame_index=14, timestamp_sec=1.4, subject_entity_id='player_1', predicate='near', raw_predicate='standing near', object_entity_id='hoop_1'), TransitionEvent(event_id='obs_3', video_id='video_001', event_type='RELATION_OBSERVED', frame_rank=16, frame_index=16, timestamp_sec=1.6, subject_entity_id='player_1', predicate='near', raw_predicate='standing near', object_entity_id='hoop_1')]})

def _sample_temporal_graph_with_nearby_relation_episode() -> TemporalGraphDocument:
    document = _sample_temporal_graph_document()
    return document.model_copy(update={'relation_episodes': [*document.relation_episodes, RelationEpisode(relation_episode_id='rel_near_1', video_id='video_001', subject_entity_id='player_1', predicate='looking_at', raw_predicate='looking at', object_entity_id='hoop_1', start_frame=6, end_frame=6, start_ts=0.6, end_ts=0.6, support_frames=1)]})

def _sample_temporal_graph_with_base_delta_projection() -> TemporalGraphDocument:
    document = _sample_temporal_graph_document()
    return document.model_copy(update={'scene_base_nodes': [SceneBaseNode(node_id='scene_002:base:hoop_1', scene_id='scene_002', video_id='video_001', entity_id='hoop_1', label='basketball hoop', first_frame=10, last_frame=19, support_frames=10, coverage_ratio=1.0)], 'scene_delta_nodes': [SceneDeltaNode(node_id='scene_002:delta:player_1', scene_id='scene_002', video_id='video_001', entity_id='player_1', label='man', start_frame=11, end_frame=14, start_ts=1.1, end_ts=1.4, support_frames=4, transition_event_count=1, relation_episode_count=1, state_episode_count=1)], 'scene_delta_observations': [SceneDeltaObservation(delta_observation_id='delta_obs_11', scene_id='scene_002', video_id='video_001', entity_id='player_1', scene_delta_node_id='scene_002:delta:player_1', source_segment_id='seg_player_1_b', source_observation_id='obs_player_11', frame_name='frame_000011.jpg', frame_rank=11, frame_index=11, timestamp_sec=1.1, image_path='frames/frame_000011.jpg', bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.5), label='man', trigger_event_ids=['event_1'], trigger_event_types=['STATE_TRANSITION'], marks_segment_start=True), SceneDeltaObservation(delta_observation_id='delta_obs_13', scene_id='scene_002', video_id='video_001', entity_id='player_1', scene_delta_node_id='scene_002:delta:player_1', source_segment_id='seg_player_1_b', source_observation_id='obs_player_13', frame_name='frame_000013.jpg', frame_rank=13, frame_index=13, timestamp_sec=1.3, image_path='frames/frame_000013.jpg', bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.4, y2=0.5), label='man', trigger_event_ids=['event_1'], trigger_event_types=['STATE_TRANSITION'])]})

def _sample_temporal_graph_with_basketball_named_video() -> TemporalGraphDocument:
    video_id = 'basketball_frame_robot_style'
    player_entity_id = f'{video_id}:man:000001'
    ball_entity_id = f'{video_id}:basketball:000001'
    player_segment_id = f'{player_entity_id}:segment:000001:000000:000040'
    ball_segment_id = f'{ball_entity_id}:segment:000001:000100:000100'
    scene_id = f'{video_id}:scene:000001'
    return TemporalGraphDocument(video_id=video_id, source_dir=Path('data/examples/basketball_frame_robot_style'), frame_count=240, observations=[], entities=[Entity(entity_id=player_entity_id, video_id=video_id, category='man', first_seen_frame=0, last_seen_frame=40, first_seen_ts=0.0, last_seen_ts=40.0, observation_count=41, node_role='dynamic', states_seen=['dribbling ball']), Entity(entity_id=ball_entity_id, video_id=video_id, category='basketball', first_seen_frame=100, last_seen_frame=100, first_seen_ts=100.0, last_seen_ts=100.0, observation_count=1, node_role='dynamic', states_seen=['basketball in air'], is_fragment=True)], relation_episodes=[], state_episodes=[StateEpisode(state_episode_id='player_state_1', video_id=video_id, entity_id=player_entity_id, attr_key='action', attr_value='dribbling ball', start_frame=0, end_frame=40, start_ts=0.0, end_ts=40.0, support_frames=41)], scene_segments=[SceneSegment(scene_id=scene_id, video_id=video_id, scene_index=1, scene_count=1, start_frame_rank=0, end_frame_rank=239, start_frame=0, end_frame=239, start_ts=0.0, end_ts=239.0, support_frames=240)], entity_segments=[EntitySegment(segment_id=player_segment_id, video_id=video_id, entity_id=player_entity_id, scene_id=scene_id, segment_index=1, segment_count=1, start_frame_rank=0, end_frame_rank=40, start_frame=0, end_frame=40, start_ts=0.0, end_ts=40.0, support_frames=41), EntitySegment(segment_id=ball_segment_id, video_id=video_id, entity_id=ball_entity_id, scene_id=scene_id, segment_index=1, segment_count=1, start_frame_rank=100, end_frame_rank=100, start_frame=100, end_frame=100, start_ts=100.0, end_ts=100.0, support_frames=1)], transition_events=[TransitionEvent(event_id='player_appear', video_id=video_id, event_type='APPEAR', frame_rank=0, frame_index=0, timestamp_sec=0.0, entity_id=player_entity_id, source_segment_id=player_segment_id), TransitionEvent(event_id='ball_appear', video_id=video_id, event_type='APPEAR', frame_rank=100, frame_index=100, timestamp_sec=100.0, entity_id=ball_entity_id, source_segment_id=ball_segment_id)])

def _sample_temporal_graph_with_color_attribute_distractors() -> TemporalGraphDocument:
    video_id = 'video_color_binding'
    scene_id = 'scene_atrium'
    return TemporalGraphDocument(video_id=video_id, source_dir=Path('data/examples/video_color_binding'), frame_count=12, observations=[], entities=[Entity(entity_id='couch_1', video_id=video_id, category='couch', first_seen_frame=0, last_seen_frame=11, first_seen_ts=0.0, last_seen_ts=1.1, observation_count=12, node_role='static', is_static=True), Entity(entity_id='person_1', video_id=video_id, category='person', first_seen_frame=0, last_seen_frame=11, first_seen_ts=0.0, last_seen_ts=1.1, observation_count=12, node_role='dynamic')], relation_episodes=[], state_episodes=[StateEpisode(state_episode_id='state_couch_red', video_id=video_id, entity_id='couch_1', attr_key='color', attr_value='red', start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12), StateEpisode(state_episode_id='state_person_red_shoes', video_id=video_id, entity_id='person_1', attr_key='appearance', attr_value='red shoes', start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12)], scene_segments=[SceneSegment(scene_id=scene_id, video_id=video_id, scene_index=1, scene_count=1, start_frame_rank=0, end_frame_rank=11, start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12, representative_static_labels=['couch'], representative_dynamic_labels=['person'], scene_kind_hint='atrium')], entity_segments=[EntitySegment(segment_id='seg_couch_1', video_id=video_id, entity_id='couch_1', scene_id=scene_id, start_frame_rank=0, end_frame_rank=11, start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12, is_static=True), EntitySegment(segment_id='seg_person_1', video_id=video_id, entity_id='person_1', scene_id=scene_id, start_frame_rank=0, end_frame_rank=11, start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12)], transition_events=[])

def _sample_temporal_graph_with_raw_visual_attribute_only() -> TemporalGraphDocument:
    video_id = 'video_raw_attribute'
    scene_id = 'scene_lobby'
    return TemporalGraphDocument(video_id=video_id, source_dir=Path('data/examples/video_raw_attribute'), frame_count=6, observations=[], entities=[Entity(entity_id='couch_raw_1', video_id=video_id, category='couch', first_seen_frame=0, last_seen_frame=5, first_seen_ts=0.0, last_seen_ts=0.5, observation_count=6, raw_states_seen=['red couch in lobby'], node_role='static', is_static=True), Entity(entity_id='person_raw_1', video_id=video_id, category='person', first_seen_frame=0, last_seen_frame=5, first_seen_ts=0.0, last_seen_ts=0.5, observation_count=6, raw_states_seen=['red shoes'], node_role='dynamic')], relation_episodes=[], state_episodes=[], scene_segments=[SceneSegment(scene_id=scene_id, video_id=video_id, scene_index=1, scene_count=1, start_frame_rank=0, end_frame_rank=5, start_frame=0, end_frame=5, start_ts=0.0, end_ts=0.5, support_frames=6, representative_static_labels=['couch'], representative_dynamic_labels=['person'], scene_kind_hint='lobby')], entity_segments=[EntitySegment(segment_id='seg_couch_raw_1', video_id=video_id, entity_id='couch_raw_1', scene_id=scene_id, start_frame_rank=0, end_frame_rank=5, start_frame=0, end_frame=5, start_ts=0.0, end_ts=0.5, support_frames=6, is_static=True), EntitySegment(segment_id='seg_person_raw_1', video_id=video_id, entity_id='person_raw_1', scene_id=scene_id, start_frame_rank=0, end_frame_rank=5, start_frame=0, end_frame=5, start_ts=0.0, end_ts=0.5, support_frames=6)], transition_events=[])

def _sample_temporal_graph_with_attribute_event_constraint() -> TemporalGraphDocument:
    video_id = 'video_attribute_event'
    scene_id = 'scene_entry'
    return TemporalGraphDocument(video_id=video_id, source_dir=Path('data/examples/video_attribute_event'), frame_count=12, observations=[], entities=[Entity(entity_id='door_actor', video_id=video_id, category='man', first_seen_frame=6, last_seen_frame=8, first_seen_ts=0.6, last_seen_ts=0.8, observation_count=3, raw_states_seen=['man in blue coat opening door'], node_role='dynamic'), Entity(entity_id='blue_jacket_bystander', video_id=video_id, category='person', first_seen_frame=0, last_seen_frame=5, first_seen_ts=0.0, last_seen_ts=0.5, observation_count=6, states_seen=['wearing dark blue jacket'], raw_states_seen=['person wearing dark blue jacket'], node_role='dynamic')], relation_episodes=[], state_episodes=[StateEpisode(state_episode_id='state_bystander_jacket', video_id=video_id, entity_id='blue_jacket_bystander', attr_key='state', attr_value='wearing dark blue jacket', start_frame=0, end_frame=5, start_ts=0.0, end_ts=0.5, support_frames=6)], scene_segments=[SceneSegment(scene_id=scene_id, video_id=video_id, scene_index=1, scene_count=1, start_frame_rank=0, end_frame_rank=11, start_frame=0, end_frame=11, start_ts=0.0, end_ts=1.1, support_frames=12, representative_static_labels=['door'], representative_dynamic_labels=['person'], scene_kind_hint='entry')], entity_segments=[EntitySegment(segment_id='seg_door_actor', video_id=video_id, entity_id='door_actor', scene_id=scene_id, start_frame_rank=6, end_frame_rank=8, start_frame=6, end_frame=8, start_ts=0.6, end_ts=0.8, support_frames=3), EntitySegment(segment_id='seg_blue_jacket_bystander', video_id=video_id, entity_id='blue_jacket_bystander', scene_id=scene_id, start_frame_rank=0, end_frame_rank=5, start_frame=0, end_frame=5, start_ts=0.0, end_ts=0.5, support_frames=6)], transition_events=[])

def _sample_temporal_graph_with_attribute_near_temporal_constraint() -> TemporalGraphDocument:
    base = _sample_temporal_graph_with_color_attribute_distractors()
    scene_id = 'scene_entry'
    couch_scene = base.scene_segments[0].model_copy(update={'scene_id': scene_id, 'scene_kind_hint': 'entry lounge', 'representative_static_labels': ['door', 'couch'], 'previous_scene_id': 'scene_outside'})
    outside_scene = SceneSegment(scene_id='scene_outside', video_id=base.video_id, scene_index=0, scene_count=2, start_frame_rank=0, end_frame_rank=4, start_frame=0, end_frame=4, start_ts=0.0, end_ts=0.4, support_frames=5, next_scene_id=scene_id, representative_static_labels=['tree'], scene_kind_hint='outside')
    entity_segments = [segment.model_copy(update={'scene_id': scene_id}) for segment in base.entity_segments]
    return base.model_copy(update={'scene_segments': [outside_scene, couch_scene], 'entity_segments': entity_segments})

def _sample_temporal_graph_with_scene_transition_chain() -> TemporalGraphDocument:
    video_id = 'video_scene_chain'
    return TemporalGraphDocument(video_id=video_id, source_dir=Path('data/examples/video_scene_chain'), frame_count=30, observations=[Observation(observation_id='obs_tree_1', video_id=video_id, entity_id='tree_1', frame_name='frame_000000.jpg', frame_rank=0, frame_index=0, timestamp_sec=0.0, image_path='frames/frame_000000.jpg', object_index=0, bbox=BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.6), label='tree', score=0.9, node_role='static', is_static=True, scene_id='scene_outside'), Observation(observation_id='obs_door_1', video_id=video_id, entity_id='door_1', frame_name='frame_000010.jpg', frame_rank=10, frame_index=10, timestamp_sec=1.0, image_path='frames/frame_000010.jpg', object_index=0, bbox=BoundingBox(x1=0.3, y1=0.0, x2=0.7, y2=0.8), label='door', score=0.9, node_role='static', is_static=True, scene_id='scene_doorway'), Observation(observation_id='obs_floor_1', video_id=video_id, entity_id='floor_1', frame_name='frame_000020.jpg', frame_rank=20, frame_index=20, timestamp_sec=2.0, image_path='frames/frame_000020.jpg', object_index=0, bbox=BoundingBox(x1=0.0, y1=0.5, x2=1.0, y2=1.0), label='floor', score=0.9, node_role='static', is_static=True, scene_id='scene_inside')], entities=[Entity(entity_id='tree_1', video_id=video_id, category='tree', first_seen_frame=0, last_seen_frame=9, first_seen_ts=0.0, last_seen_ts=0.9, observation_count=1, node_role='static', is_static=True), Entity(entity_id='door_1', video_id=video_id, category='door', first_seen_frame=10, last_seen_frame=19, first_seen_ts=1.0, last_seen_ts=1.9, observation_count=1, node_role='static', is_static=True), Entity(entity_id='floor_1', video_id=video_id, category='floor', first_seen_frame=20, last_seen_frame=29, first_seen_ts=2.0, last_seen_ts=2.9, observation_count=1, node_role='static', is_static=True)], relation_episodes=[], state_episodes=[], scene_segments=[SceneSegment(scene_id='scene_outside', video_id=video_id, scene_index=1, scene_count=3, start_frame_rank=0, end_frame_rank=9, start_frame=0, end_frame=9, start_ts=0.0, end_ts=0.9, support_frames=10, next_scene_id='scene_doorway', representative_static_labels=['tree'], scene_kind_hint='tree'), SceneSegment(scene_id='scene_doorway', video_id=video_id, scene_index=2, scene_count=3, start_frame_rank=10, end_frame_rank=19, start_frame=10, end_frame=19, start_ts=1.0, end_ts=1.9, support_frames=10, previous_scene_id='scene_outside', next_scene_id='scene_inside', representative_static_labels=['door'], scene_kind_hint='door'), SceneSegment(scene_id='scene_inside', video_id=video_id, scene_index=3, scene_count=3, start_frame_rank=20, end_frame_rank=29, start_frame=20, end_frame=29, start_ts=2.0, end_ts=2.9, support_frames=10, previous_scene_id='scene_doorway', representative_static_labels=['floor'], scene_kind_hint='floor')], entity_segments=[EntitySegment(segment_id='seg_tree_1', video_id=video_id, entity_id='tree_1', scene_id='scene_outside', segment_index=1, segment_count=1, start_frame_rank=0, end_frame_rank=9, start_frame=0, end_frame=9, start_ts=0.0, end_ts=0.9, support_frames=1, is_static=True), EntitySegment(segment_id='seg_door_1', video_id=video_id, entity_id='door_1', scene_id='scene_doorway', segment_index=1, segment_count=1, start_frame_rank=10, end_frame_rank=19, start_frame=10, end_frame=19, start_ts=1.0, end_ts=1.9, support_frames=1, is_static=True), EntitySegment(segment_id='seg_floor_1', video_id=video_id, entity_id='floor_1', scene_id='scene_inside', segment_index=1, segment_count=1, start_frame_rank=20, end_frame_rank=29, start_frame=20, end_frame=29, start_ts=2.0, end_ts=2.9, support_frames=1, is_static=True)], transition_events=[])

def test_canonical_graph_hybrid_temporal_query_expands_from_relation_seed() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8)
    assert result.raw_payload['retrieval_mode'] == 'canonical_graph_hybrid'
    assert result.raw_payload['fragments']
    assert result.items
    assert result.items[0]['source_record_id'] == 'episode:relation:rel_1'
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'episode:state:state_2' in item_ids
    assert 'scene:scene_002' in item_ids
    state_item = next((item for item in result.items if item['source_record_id'] == 'episode:state:state_2'))
    assert state_item['metadata']['hybrid_support']['path_edge_types']
    assert state_item['metadata']['hybrid_support']['hop_count'] >= 1

def test_canonical_graph_hybrid_temporal_query_returns_scene_sequence_chain() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='How long was it inside after entering?', limit=8), mode='temporal', limit=8)
    chain_item = next((item for item in result.items if item['source_kind'] == 'temporal_composition'))
    assert chain_item['metadata']['composition_type'] == 'scene_sequence'
    assert chain_item['metadata']['scene_ids'] == ['scene_outside', 'scene_doorway', 'scene_inside']
    assert chain_item['metadata']['time_range']['start_frame'] == 0
    assert chain_item['metadata']['time_range']['end_frame'] == 29
    scene_labels = [scene['scene_kind_hint'] for scene in chain_item['metadata']['scene_sequence']]
    assert scene_labels == ['tree', 'door', 'floor']

def test_canonical_graph_hybrid_non_temporal_query_does_not_return_scene_sequence_chain() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='door', limit=8), mode='temporal', limit=8)
    assert result.items
    assert {item['source_kind'] for item in result.items}.isdisjoint({'temporal_composition'})

def test_canonical_graph_hybrid_query_plan_uses_local_budget_for_specific_query() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert plan['global_scope_score'] <= 1
    assert plan['candidate_strategy'] == 'local_seed_index'
    assert plan['candidate_count'] is not None
    assert plan['candidate_count'] < len(_build_projection(graph).records)

def test_canonical_graph_hybrid_query_plan_expands_for_global_query() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you ever see a hoop?', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert plan['global_scope_score'] >= 3
    assert plan['retrieval_budget'] >= 200

def test_canonical_graph_hybrid_query_window_caps_global_scope() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you ever see a hoop?', start_ts=0.0, end_ts=1.2, limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert plan['global_scope_score'] <= 1
    assert plan['retrieval_budget'] <= 180

def test_canonical_graph_hybrid_windowed_position_query_uses_moderate_scope() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Where did you see a red fire hydrant?', start_ts=0.0, end_ts=500.0, limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['query_family'] == 'position'
    assert plan['global_scope_score'] <= 2
    assert plan['retrieval_budget'] <= 320

def test_canonical_graph_hybrid_current_window_caps_scope_to_local() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Are you stopped in a loading zone right now?', start_ts=1.0, end_ts=1.2, limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['question_compiler']['current_window_required'] is True
    assert plan['global_scope_score'] == 0
    assert plan['retrieval_budget'] <= 80

def test_canonical_graph_hybrid_first_event_prefers_query_window_start() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When did you first enter the building?', start_ts=1.0, end_ts=2.9, limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert 'earliest_event' in plan['question_compiler']['ranking_policy']
    assert result.items[0]['source_kind'] == 'temporal_composition'
    assert result.items[0]['metadata']['time_range']['start_ts'] >= 1.0

def test_canonical_graph_hybrid_nearest_position_keeps_target_proxy_signal() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Where is the nearest basketball hoop?', start_ts=0.0, end_ts=1.9, limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert 'distance_or_recency_proxy' in plan['question_compiler']['ranking_policy']
    assert any(('hoop' in item['summary'].lower() for item in result.items))

def test_canonical_graph_hybrid_attribute_query_wins_over_temporal_hint() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color were the couches near where I entered?', limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['query_family'] == 'attribute'
    assert 'couches' in plan['core_terms']
    assert 'couch' not in plan['core_terms']
    assert 'sofa' not in plan['core_terms']

def test_canonical_graph_hybrid_generic_profile_does_not_expand_bicyclist_query_to_bicycle() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you see any bicyclists?', limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['query_family'] == 'existence'
    assert 'bicyclists' in plan['core_terms']
    assert 'bicyclist' not in plan['core_terms']
    assert 'bicycle' not in plan['core_terms']
    assert 'bike' not in plan['core_terms']

def test_canonical_graph_hybrid_default_generic_profile_skips_urban_navigation_alias_expansion() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you see any bicyclists?', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert plan['domain_profile'] == 'generic'
    assert 'bicyclists' in plan['core_terms']
    assert 'bicycle' not in plan['core_terms']
    assert 'bike' not in plan['core_terms']

def test_canonical_graph_hybrid_existence_query_keeps_spatial_phrase_as_constraint() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you see any bicyclists near the entrance?', limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['query_family'] == 'existence'
    assert 'bicyclists' in plan['core_terms']
    assert not any((item['source_kind'] == 'temporal_composition' for item in result.items[:3]))

def test_canonical_graph_hybrid_attribute_query_binds_color_to_target_object() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color was the red couch?', limit=6), mode='temporal', limit=6, domain_profile='generic')
    item_ids = [item['source_record_id'] for item in result.items]
    assert item_ids[0] == 'episode:state:state_couch_red'
    assert 'episode:state:state_person_red_shoes' not in item_ids

def test_canonical_graph_hybrid_attribute_query_uses_raw_visual_attribute_binding() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_raw_visual_attribute_only())
    couch_node = next((node for node in graph.entity_nodes if node.entity_id == 'couch_raw_1'))
    assert couch_node.states_seen == []
    assert couch_node.attributes['raw_visual_attributes'] == ['red couch in lobby']
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color was the couch?', limit=6), mode='temporal', limit=6, domain_profile='generic')
    top_item = result.items[0]
    assert top_item['source_kind'] == 'entity_attribute'
    assert top_item['metadata']['entity_id'] == 'couch_raw_1'
    assert top_item['metadata']['raw_visual_attribute'] == 'red couch in lobby'
    assert top_item['metadata']['attr_key'] == 'visual_attribute'

def test_canonical_graph_hybrid_existence_query_binds_attribute_value_to_target() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_raw_visual_attribute_only())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Did you see a red couch?', limit=6, metadata={'question_ir': {'answer_type': 'binary', 'operation': 'existence_query', 'target_entities': ['couch'], 'attribute_dimension': 'color', 'aggregation': 'yes_no', 'rendering': 'yes_no'}}), mode='temporal', limit=6, domain_profile='generic')
    top_item = result.items[0]
    assert top_item['source_kind'] == 'entity_attribute'
    assert top_item['metadata']['entity_id'] == 'couch_raw_1'
    assert top_item['metadata']['raw_visual_attribute'] == 'red couch in lobby'
    assert result.raw_payload['query_plan']['question_compiler']['attributes'] == ['color', 'red']
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'entity:person_raw_1:raw_visual_attribute:0001' not in item_ids[:3]

def test_canonical_graph_hybrid_exposes_supplemental_same_entity_context() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What did you observe about the couch?', answer_type='text', metadata={'question_ir': {'answer_type': 'text', 'operation': 'attribute_query', 'target_entities': ['couch'], 'attribute_dimension': 'appearance', 'aggregation': 'multiple', 'rendering': 'short_phrase'}}, limit=2), mode='temporal', limit=2, domain_profile='generic')
    primary_ids = result.raw_payload['primary_item_ids']
    supplemental_ids = result.raw_payload['supplemental_item_ids']
    assert primary_ids
    assert supplemental_ids
    assert any(('couch_1' in record_id for record_id in supplemental_ids))
    assert all((record_id not in primary_ids for record_id in supplemental_ids))
    assert any((item['metadata']['hybrid_support'].get('supplemental_context') is True for item in result.items if item['source_record_id'] in supplemental_ids))

def test_canonical_graph_hybrid_attribute_query_uses_event_constraint() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_attribute_event_constraint())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color jacket was the person who opened the door wearing?', limit=6), mode='temporal', limit=6, domain_profile='generic')
    top_item = result.items[0]
    assert top_item['source_kind'] in {'entity_attribute', 'episode'}
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'entity:door_actor:raw_visual_attribute:0001' in item_ids

def test_canonical_graph_hybrid_attribute_query_binds_attribute_key_to_target() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What was the appearance of the person?', limit=6), mode='temporal', limit=6)
    plan = result.raw_payload['query_plan']
    item_ids = [item['source_record_id'] for item in result.items]
    assert plan['query_family'] == 'attribute'
    assert 'appearance' in plan['question_compiler']['attributes']
    assert 'person' in plan['question_compiler']['target_entities']
    assert item_ids[0] == 'episode:state:state_person_red_shoes'
    assert 'episode:state:state_couch_red' not in item_ids[:3]

def test_canonical_graph_hybrid_attribute_query_keeps_temporal_phrase_as_constraint() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_attribute_near_temporal_constraint())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color were the couches near where I entered?', limit=8), mode='temporal', limit=8, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    assert plan['query_family'] == 'attribute'
    assert 'near_reference_event' in plan['question_compiler']['spatial_constraints']
    assert 'enter' in plan['question_compiler']['events']
    assert 'couches' in plan['core_terms']
    assert 'couch' not in plan['core_terms']

def test_canonical_graph_hybrid_projection_indexes_entity_derived_salient_scene_label() -> None:
    document = _sample_temporal_graph_with_color_attribute_distractors()
    scene = document.scene_segments[0].model_copy(update={'representative_static_labels': ['floor'], 'scene_kind_hint': 'atrium'})
    graph = build_canonical_temporal_memory_graph(document.model_copy(update={'scene_segments': [scene]}))
    projection = _build_projection(graph)
    scene_record = projection.records['scene:scene_atrium']
    assert scene_record.metadata['salient_labels'] == []
    assert set(scene_record.metadata['semantic_alias_terms']) <= {'floor', 'atrium', 'couch', 'person'}
    assert 'sofa' not in scene_record.tokens

def test_canonical_graph_hybrid_reuses_projection_cache_file() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    cache_path = Path('.scratch') / f'canonical_projection_test_{uuid.uuid4().hex}.pkl'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.unlink(missing_ok=True)
    try:
        first = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8, projection_cache_path=cache_path)
        assert cache_path.exists()
        assert first.items
        _PROJECTION_CACHE.clear()
        second = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8, projection_cache_path=cache_path)
        assert [item['source_record_id'] for item in second.items] == [item['source_record_id'] for item in first.items]
    finally:
        cache_path.unlink(missing_ok=True)
        _PROJECTION_CACHE.clear()

def test_canonical_graph_hybrid_reuses_light_projection_cache_file() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    cache_path = Path('.scratch') / f'canonical_projection_test_{uuid.uuid4().hex}.json.gz'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.unlink(missing_ok=True)
    try:
        first = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8, projection_cache_path=cache_path)
        assert cache_path.exists()
        assert first.items
        _PROJECTION_CACHE.clear()
        second = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=8), mode='temporal', limit=8, projection_cache_path=cache_path)
        assert [item['source_record_id'] for item in second.items] == [item['source_record_id'] for item in first.items]
    finally:
        cache_path.unlink(missing_ok=True)
        _PROJECTION_CACHE.clear()

def test_canonical_graph_hybrid_query_plan_uses_scene_first_for_temporal_query() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='How long were you inside?', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert plan['global_scope_score'] >= 3
    assert str(plan['candidate_strategy']).startswith('scene_first_')
    assert 'temporal_axis' in str(plan['candidate_strategy'])

def test_canonical_graph_hybrid_query_plan_includes_question_compiler_fields() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Where did you see a red fire hydrant?', limit=5), mode='temporal', limit=5, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    compiler = plan['question_compiler']
    assert plan['query_family'] == 'position'
    assert 'red' in plan['core_terms']
    assert compiler['answer_type'] == 'position'
    assert compiler['target_entities'] == []
    assert compiler['attributes'] == ['red']
    assert compiler['retrieval_strategy'] == 'target_entity_scene_then_event_time'

def test_canonical_graph_hybrid_query_plan_marks_current_window_questions() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='Are you stopped in a loading zone right now?', limit=5), mode='temporal', limit=5, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    compiler = plan['question_compiler']
    assert compiler['current_window_required'] is True
    assert compiler['target_entities'] == []
    assert 'stop' in compiler['events']
    assert compiler['retrieval_strategy'] == 'question_window_scene_first'
    assert plan['global_scope_score'] <= 1

def test_canonical_graph_hybrid_query_plan_uses_structured_question_ir() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When were you in the atrium with the red couches?', answer_type='time', metadata={'question_ir': {'answer_type': 'time', 'operation': 'occurrence', 'qa_focus': 'time_duration', 'target_state': 'being in atrium with red couches', 'target_entities': ['atrium', 'red couches'], 'rendering': 'minutes_ago'}}, limit=5), mode='temporal', limit=5, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    compiler = plan['question_compiler']
    assert plan['query_family'] == 'temporal'
    assert 'couches' in plan['core_terms']
    assert 'couch' not in plan['core_terms']
    assert 'red' in plan['core_terms']
    assert 'operation' not in plan['core_terms']
    assert 'answer_type' not in plan['core_terms']
    assert compiler['answer_type'] == 'time'
    assert compiler['target_entities'] == ['atrium', 'red couches']
    assert compiler['retrieval_strategy'] == 'temporal_axis_then_target_filter'
    assert 'question_ir_experimental_override' in compiler['risk_notes']

def test_canonical_graph_hybrid_query_plan_exposes_hard_constraint_groups() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_color_attribute_distractors())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='What color were the couches near where I entered the building?', answer_type='text', metadata={'question_ir': {'answer_type': 'text', 'operation': 'attribute_query', 'qa_focus': 'descriptive', 'target_entities': ['couches'], 'attribute_dimension': 'color', 'rendering': 'color_only', 'hard_constraints': [{'kind': 'spatial', 'text': 'near where I entered the building', 'role': 'spatial_condition', 'target_entities': ['building'], 'must_bind_to': ['same_scene', 'same_time_window']}]}}, limit=5), mode='temporal', limit=5, domain_profile='generic')
    plan = result.raw_payload['query_plan']
    flattened = {token for group in plan['hard_constraint_token_groups'] for token in group}
    assert 'building' in flattened
    assert 'entered' in flattened
    assert 'building' in plan['constraint_tokens']

def test_canonical_graph_hybrid_temporal_query_promotes_time_axis_seeds() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When did you enter inside?', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert 'temporal_axis' in str(plan['candidate_strategy'])
    seed_kinds = [item['source_kind'] for item in result.raw_payload['seed_candidates']]
    assert result.raw_payload['seed_candidates'][0]['source_kind'] == 'temporal_composition'
    assert 'scene' in seed_kinds
    item_kinds = {item['source_kind'] for item in result.items}
    assert result.items[0]['source_kind'] == 'temporal_composition'
    assert 'temporal_composition' in item_kinds
    assert 'scene' in item_kinds

def test_canonical_graph_hybrid_temporal_query_uses_axis_without_object_seed() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_scene_transition_chain())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When did you leave?', limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert 'temporal_axis' in str(plan['candidate_strategy'])
    assert plan['candidate_count'] is not None
    assert result.raw_payload['seed_candidates'][0]['source_kind'] == 'temporal_composition'
    assert result.items[0]['source_kind'] == 'temporal_composition'
    chain_items = [item for item in result.items if item['source_kind'] == 'temporal_composition']
    assert chain_items
    assert any((scene['scene_id'] == 'scene_doorway' for item in chain_items for scene in item['metadata']['scene_sequence']))

def test_canonical_graph_hybrid_temporal_axis_keeps_scene_chain_while_limiting_details() -> None:
    document = _sample_temporal_graph_with_scene_transition_chain()
    document = document.model_copy(update={'entities': [*document.entities, Entity(entity_id='car_1', video_id=document.video_id, category='car', first_seen_frame=100, last_seen_frame=109, first_seen_ts=10.0, last_seen_ts=10.9, observation_count=1, node_role='static', is_static=True)], 'scene_segments': [*document.scene_segments, SceneSegment(scene_id='scene_late_unrelated', video_id=document.video_id, scene_index=4, scene_count=4, start_frame_rank=100, end_frame_rank=109, start_frame=100, end_frame=109, start_ts=10.0, end_ts=10.9, support_frames=10, representative_static_labels=['car'], scene_kind_hint='parking')], 'entity_segments': [*document.entity_segments, EntitySegment(segment_id='seg_car_1', video_id=document.video_id, entity_id='car_1', scene_id='scene_late_unrelated', segment_index=1, segment_count=1, start_frame_rank=100, end_frame_rank=109, start_frame=100, end_frame=109, start_ts=10.0, end_ts=10.9, support_frames=1, is_static=True)]})
    graph = build_canonical_temporal_memory_graph(document)
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When did you leave the building?', start_ts=0.0, end_ts=2.9, limit=8), mode='temporal', limit=8)
    plan = result.raw_payload['query_plan']
    assert 'temporal_axis' in str(plan['candidate_strategy'])
    assert plan['candidate_count'] is not None
    seed_kinds = [item['source_kind'] for item in result.raw_payload['seed_candidates']]
    item_kinds = {item['source_kind'] for item in result.items}
    assert 'temporal_composition' in seed_kinds
    assert 'temporal_composition' in item_kinds
    assert 'scene' in item_kinds
    assert plan['candidate_count'] < len(_build_projection(graph).records)
    assert all(('scene_late_unrelated' not in item['source_record_id'] for item in result.items))

def test_canonical_graph_hybrid_rebuilds_compact_derived_temporal_links() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_nearby_relation_episode(), compact_derived_temporal_edges=True)
    assert {edge.edge_type for edge in graph.edges}.isdisjoint({'OVERLAPS_WITH', 'TEMPORALLY_NEAR'})
    projection = _build_projection(graph)
    overlap_edge_types = {link.edge_type for link in projection.adjacency['episode:relation:rel_1'] if link.neighbor_id == 'episode:state:state_2'}
    near_edge_types = {link.edge_type for link in projection.adjacency['episode:relation:rel_near_1'] if link.neighbor_id == 'episode:state:state_1'}
    assert 'OVERLAPS_WITH' in overlap_edge_types
    assert 'TEMPORALLY_NEAR' in near_edge_types
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='near hoop', limit=10), mode='temporal', limit=10)
    assert result.items

def test_canonical_graph_hybrid_rebuilds_compact_previous_episode_links() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    assert 'PREVIOUS_EPISODE' not in {edge.edge_type for edge in graph.edges}
    projection = _build_projection(graph)
    previous_edge_types = {link.edge_type for link in projection.adjacency['episode:scene_segment:scene_002'] if link.neighbor_id == 'episode:scene_segment:scene_001'}
    assert 'PREVIOUS_EPISODE' in previous_edge_types
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='what happened before shooting', limit=8), mode='temporal', limit=8)
    assert result.items

def test_canonical_graph_hybrid_base_delta_projection_is_opt_in() -> None:
    document = _sample_temporal_graph_with_base_delta_projection()
    graph = build_canonical_temporal_memory_graph(document)
    default_projection = _build_projection(graph, temporal_document=document)
    default_kinds = {record.source_kind for record in default_projection.records.values()}
    assert 'scene_base' not in default_kinds
    assert 'scene_delta' not in default_kinds
    projected = _build_projection(graph, temporal_document=document, include_base_delta_projection=True)
    projected_kinds = {record.source_kind for record in projected.records.values()}
    assert 'scene_base' in projected_kinds
    assert 'scene_delta' in projected_kinds
    delta_record_id = 'scene_delta:scene_002:delta:player_1'
    scene_links = {link.edge_type for link in projected.adjacency['scene:scene_002'] if link.neighbor_id == delta_record_id}
    entity_links = {link.edge_type for link in projected.adjacency['entity:player_1'] if link.neighbor_id == delta_record_id}
    assert 'SCENE_UPDATES' in scene_links
    assert 'HAS_EPISODE' in entity_links

def test_canonical_graph_hybrid_base_delta_projection_surfaces_virtual_evidence() -> None:
    document = _sample_temporal_graph_with_base_delta_projection()
    graph = build_canonical_temporal_memory_graph(document)
    default_result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='scene delta dynamic change shooting', limit=8), mode='temporal', limit=8, temporal_document=document)
    assert {item['source_kind'] for item in default_result.items}.isdisjoint({'scene_base', 'scene_delta'})
    delta_result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='scene delta dynamic change shooting', limit=8), mode='temporal', limit=8, temporal_document=document, include_base_delta_projection=True)
    delta_item = next((item for item in delta_result.items if item['source_kind'] == 'scene_delta'))
    assert delta_item['metadata']['entity_id'] == 'player_1'
    assert delta_item['metadata']['observed_frames'] == [11, 13]
    assert delta_result.raw_payload['include_base_delta_projection'] is True
    base_result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='static background hoop scene base', limit=8), mode='temporal', limit=8, temporal_document=document, include_base_delta_projection=True)
    base_item = next((item for item in base_result.items if item['source_kind'] == 'scene_base'))
    assert base_item['metadata']['entity_id'] == 'hoop_1'
    assert base_item['metadata']['coverage_ratio'] == 1.0

def test_canonical_graph_hybrid_entity_query_returns_lifecycle_context() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(entity_ids=['player_1'], limit=8), mode='entity', limit=8)
    assert result.items
    assert result.items[0]['source_record_id'] == 'entity:player_1'
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'stable_identity:video_001:stable:human:001' in item_ids
    assert 'entity:player_1' in item_ids
    assert 'episode:entity_segment:seg_player_1_b' in item_ids
    assert 'episode:state:state_2' in item_ids
    assert 'episode:relation:rel_1' in item_ids
    identity_item = next((item for item in result.items if item['source_record_id'] == 'stable_identity:video_001:stable:human:001'))
    assert identity_item['source_kind'] == 'stable_identity'
    assert identity_item['metadata']['member_entity_ids'] == ['player_1']

def test_canonical_graph_hybrid_temporal_query_returns_transition_event() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='state transition shooting', scene_ids=['scene_002'], limit=6), mode='temporal', limit=6)
    assert result.items
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'event:event_1' in item_ids
    event_item = next((item for item in result.items if item['source_record_id'] == 'event:event_1'))
    assert event_item['source_kind'] == 'transition_event'
    assert event_item['metadata']['event_type'] == 'STATE_TRANSITION'

def test_canonical_graph_hybrid_temporal_query_aggregates_relation_observed_events() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_observed_relation_burst())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='player near hoop', scene_ids=['scene_002'], limit=8), mode='temporal', limit=8)
    burst_item = next((item for item in result.items if item['metadata'].get('episode_kind') == 'RELATION_OBSERVED_BURST'))
    assert burst_item['source_kind'] == 'episode'
    assert burst_item['metadata']['predicate'] == 'near'
    assert burst_item['metadata']['subject_entity_id'] == 'player_1'
    assert burst_item['metadata']['object_entity_id'] == 'hoop_1'
    assert burst_item['metadata']['source_event_ids'] == ['obs_1', 'obs_2', 'obs_3']
    assert burst_item['metadata']['observed_frames'] == [12, 14, 16]
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'event:obs_1' not in item_ids
    assert 'event:obs_2' not in item_ids
    assert 'event:obs_3' not in item_ids

def test_canonical_graph_hybrid_temporal_query_aggregates_compact_relation_observed_samples() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_observed_relation_burst(), compact_relation_observed_events=True)
    assert not [node for node in graph.transition_event_nodes if node.event_type == 'RELATION_OBSERVED']
    assert [sample.event_id for sample in graph.relation_observed_samples] == ['obs_1', 'obs_2', 'obs_3']
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='player near hoop', scene_ids=['scene_002'], limit=8), mode='temporal', limit=8)
    burst_item = next((item for item in result.items if item['metadata'].get('episode_kind') == 'RELATION_OBSERVED_BURST'))
    assert burst_item['source_kind'] == 'episode'
    assert burst_item['metadata']['source_event_ids'] == ['obs_1', 'obs_2', 'obs_3']
    assert burst_item['metadata']['source_record_ids'] == ['event:obs_1', 'event:obs_2', 'event:obs_3']
    assert burst_item['metadata']['observed_frames'] == [12, 14, 16]
    item_ids = [item['source_record_id'] for item in result.items]
    assert 'event:obs_1' not in item_ids
    assert 'event:obs_2' not in item_ids
    assert 'event:obs_3' not in item_ids

def test_canonical_graph_hybrid_query_ignores_video_prefix_pollution() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_basketball_named_video())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='basketball', limit=6), mode='entity', limit=6)
    assert result.items
    top_item = result.items[0]
    assert top_item['source_kind'] in {'entity', 'episode', 'transition_event'}
    assert 'basketball:000001' in top_item['source_record_id']

def test_canonical_graph_hybrid_query_prefers_matching_appear_event() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_basketball_named_video())
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='When does basketball appear?', limit=6), mode='temporal', limit=6)
    assert result.items
    top_item = result.items[0]
    assert top_item['source_kind'] == 'transition_event'
    assert top_item['metadata']['event_type'] == 'APPEAR'
    assert top_item['metadata']['entity_id'].endswith('basketball:000001')
    assert 'at frame 100' in top_item['summary']
    assert '100.00s' not in top_item['summary']

def test_canonical_graph_hybrid_time_formatter_keeps_frame_axis_distinct() -> None:
    assert _format_time_values(190, 195, 190.0, 195.0) == 'frames 190-195'
    assert _format_time_values(210, 210, 210.0, 210.0) == 'frame 210'
    assert _format_time_values(70, 120, 1.42, 2.43) == '1.42s-2.43s'

def test_navqa_temporal_profile_strips_route_only_ir_fields() -> None:
    ir = {'operation': 'position_query', 'target_entities': ['player'], 'question_route_id': 'entity_relation_spatial', 'question_route': {'route_id': 'entity_relation_spatial'}, 'routing': {'route_id': 'entity_relation_spatial'}, 'evidence_question_type': 'non_temporal.spatial.localization', 'category': 'object localization'}
    sanitized = unified_graph_retrieval_question_ir(ir, graph_retrieval_profile='navqa_temporal')
    assert sanitized == {'operation': 'position_query', 'target_entities': ['player']}

def test_navqa_temporal_retrieval_is_invariant_to_route_metadata() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    semantic_ir = {'operation': 'position_query', 'target_entities': ['player'], 'spatial_constraints': [{'relation': 'near', 'target': 'hoop'}]}
    base = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='what is near the hoop player near hoop', metadata={'graph_retrieval_profile': 'navqa_temporal', 'question_ir': semantic_ir}, limit=8), mode='temporal', limit=8)
    perturbed = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='what is near the hoop player near hoop', metadata={'graph_retrieval_profile': 'navqa_temporal', 'category': 'object localization', 'question_route': {'route_id': 'entity_relation_spatial'}, 'question_ir': {**semantic_ir, 'category': 'object localization', 'evidence_question_type': 'non_temporal.spatial.localization', 'question_route_id': 'entity_relation_spatial', 'question_route': {'route_id': 'entity_relation_spatial'}, 'routing': {'route_id': 'entity_relation_spatial'}}}, limit=8), mode='temporal', limit=8)
    assert perturbed.items == base.items
    assert perturbed.raw_payload == base.raw_payload

class _TrackingVectorIndex:
    model_name = 'test-vector-index'
    cache_status = 'memory'

    def __init__(self, hits: list[VectorRecallHit]) -> None:
        self.hits = hits
        self.search_calls = 0

    def search(self, _: str, **_kwargs: object) -> list[VectorRecallHit]:
        self.search_calls += 1
        return self.hits

class _EmptyThenTrackingVectorIndex(_TrackingVectorIndex):

    def search(self, _: str, **kwargs: object) -> list[VectorRecallHit]:
        self.search_calls += 1
        if self.search_calls == 1:
            return []
        assert kwargs['min_similarity'] == -1.0
        return self.hits

def test_canonical_vector_profile_retries_empty_semantic_recall_without_threshold() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    vector_record_id = 'state_value:action:shooting-a3b24171'
    vector_index = _EmptyThenTrackingVectorIndex([VectorRecallHit(record_id=vector_record_id, source_kind='state_value', similarity=0.12, rank=1)])
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='where should I store produce', metadata={'graph_retrieval_profile': 'canonical_vector_hybrid_v1', 'raw_question': 'Where should I store produce?'}, limit=8), mode='temporal', limit=8, vector_recall_index=vector_index)
    assert vector_record_id in [item['source_record_id'] for item in result.items]
    assert result.raw_payload['vector_recall']['empty_result_fallback'] is True
    assert vector_index.search_calls == 2

def test_canonical_vector_union_profile_scores_vector_hits_in_primary_path() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    vector_record_id = 'state_value:action:shooting-a3b24171'
    vector_index = _TrackingVectorIndex([VectorRecallHit(record_id=vector_record_id, source_kind='state_value', similarity=0.92, rank=1)])
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='what is below the hoop', metadata={'graph_retrieval_profile': 'canonical_vector_union_v4', 'raw_question': 'What is below the hoop?', 'question_ir': {'operation': 'position_query', 'target_entities': ['object below hoop'], 'spatial_constraints': [{'relation': 'below', 'target': 'hoop'}]}}, limit=8), mode='temporal', limit=8, vector_recall_index=vector_index)
    result_ids = [item['source_record_id'] for item in result.items]
    assert vector_record_id in result_ids
    assert vector_record_id in result.raw_payload['primary_item_ids']
    assert result.raw_payload['vector_supplemental_item_ids'] == []
    assert result.raw_payload['vector_union_candidate_ids'] == [vector_record_id]
    assert '+canonical_vector_union' in result.raw_payload['query_plan']['candidate_strategy']
    assert vector_index.search_calls == 1

def test_canonical_vector_structured_profile_binds_color_to_target_owner() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_with_raw_visual_attribute_only())
    projection = _build_projection(graph)
    query = MemoryQuery(query_text='what color is the couch', metadata={'graph_retrieval_profile': 'canonical_vector_structured_v3', 'raw_question': 'What color is the couch?', 'question_ir': {'operation': 'attribute_query', 'target_entities': ['couch'], 'attribute_dimension': 'color'}})
    intent = hybrid_retrieval._infer_query_intent(query.query_text, hybrid_retrieval._tokenize(query.query_text))
    intent = hybrid_retrieval._apply_question_ir_to_intent(intent, query.metadata['question_ir'], query_text=query.query_text, lexicon=hybrid_retrieval.get_domain_lexicon(hybrid_retrieval.DEFAULT_DOMAIN_PROFILE))
    couch_attribute = next((record for record in projection.records.values() if record.source_kind == 'entity_attribute' and record.metadata.get('entity_id') == 'couch_raw_1'))
    person_attribute = next((record for record in projection.records.values() if record.source_kind == 'entity_attribute' and record.metadata.get('entity_id') == 'person_raw_1'))
    couch_score = hybrid_retrieval._canonical_vector_structural_score_adjustment(couch_attribute, query=query, query_intent=intent)
    person_score = hybrid_retrieval._canonical_vector_structural_score_adjustment(person_attribute, query=query, query_intent=intent)
    assert couch_score > 0.0
    assert person_score == 0.0

def test_navqa_temporal_profile_does_not_query_vector_index() -> None:
    graph = build_canonical_temporal_memory_graph(_sample_temporal_graph_document())
    vector_index = _TrackingVectorIndex([])
    result = run_canonical_graph_hybrid_retrieval(graph, MemoryQuery(query_text='player near hoop', metadata={'graph_retrieval_profile': 'navqa_temporal'}, limit=8), mode='temporal', limit=8, vector_recall_index=vector_index)
    assert result.items
    assert vector_index.search_calls == 0
    assert 'vector_recall' not in result.raw_payload
