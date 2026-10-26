from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.temporal.pathing import resolve_repo_input_path


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-password"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a temporal graph JSON document into Neo4j."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to temporal_graph.siglip.json.",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=NEO4J_URI,
        help="Neo4j Bolt URI.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=NEO4J_USER,
        help="Neo4j username.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=NEO4J_PASSWORD,
        help="Neo4j password.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear the target Neo4j database before export.",
    )
    return parser


def load_graph_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_relationship_type(predicate: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", predicate.strip()).upper()
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "RELATED_TO"
    if sanitized[0].isdigit():
        sanitized = f"REL_{sanitized}"
    return sanitized


def chunked(items: list[dict], size: int = 500) -> Iterable[list[dict]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def reset_database(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def upsert_entities(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        SET e.video_id = row.video_id,
            e.category = row.category,
            e.display_name = row.display_name,
            e.first_seen_frame = row.first_seen_frame,
            e.last_seen_frame = row.last_seen_frame,
            e.first_seen_ts = row.first_seen_ts,
            e.last_seen_ts = row.last_seen_ts,
            e.observation_count = row.observation_count,
            e.canonical_image_path = row.canonical_image_path,
            e.labels_seen = row.labels_seen,
            e.states_seen = row.states_seen,
            e.parent_labels_seen = row.parent_labels_seen,
            e.is_static = row.is_static,
            e.is_noise = row.is_noise,
            e.is_fragment = row.is_fragment,
            e.graph_origin = row.graph_origin
        """,
        rows=rows,
    )


def upsert_frames(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (f:Frame {frame_id: row.frame_id})
        SET f.video_id = row.video_id,
            f.frame_index = row.frame_index,
            f.frame_rank = row.frame_rank,
            f.timestamp_sec = row.timestamp_sec,
            f.frame_name = row.frame_name,
            f.image_path = row.image_path
        """,
        rows=rows,
    )


def upsert_next_frame_edges(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (src:Frame {frame_id: row.source_frame_id})
        MATCH (dst:Frame {frame_id: row.target_frame_id})
        MERGE (src)-[r:NEXT_FRAME]->(dst)
        SET r.video_id = row.video_id,
            r.frame_gap = row.frame_gap,
            r.time_delta_sec = row.time_delta_sec
        """,
        rows=rows,
    )


def upsert_observations(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MATCH (f:Frame {frame_id: row.frame_id})
        MERGE (o:Observation {observation_id: row.observation_id})
        SET o.video_id = row.video_id,
            o.entity_id = row.entity_id,
            o.frame_id = row.frame_id,
            o.frame_name = row.frame_name,
            o.frame_rank = row.frame_rank,
            o.frame_index = row.frame_index,
            o.timestamp_sec = row.timestamp_sec,
            o.image_path = row.image_path,
            o.object_index = row.object_index,
            o.label = row.label,
            o.score = row.score,
            o.state = row.state,
            o.state_values = row.state_values,
            o.parent_label = row.parent_label,
            o.is_static = row.is_static,
            o.is_noise = row.is_noise,
            o.bbox_x1 = row.bbox_x1,
            o.bbox_y1 = row.bbox_y1,
            o.bbox_x2 = row.bbox_x2,
            o.bbox_y2 = row.bbox_y2
        MERGE (e)-[:HAS_OBSERVATION]->(o)
        MERGE (o)-[:OBSERVED_AT]->(f)
        """,
        rows=rows,
    )


def upsert_entity_lifecycles(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MATCH (start:Frame {frame_id: row.start_frame_id})
        MATCH (end:Frame {frame_id: row.end_frame_id})
        MERGE (life:EntityLifecycle {lifecycle_id: row.lifecycle_id})
        SET life.video_id = row.video_id,
            life.entity_id = row.entity_id,
            life.category = row.category,
            life.scene_id = row.scene_id,
            life.source_segment_id = row.source_segment_id,
            life.segment_index = row.segment_index,
            life.segment_count = row.segment_count,
            life.first_seen_frame = row.first_seen_frame,
            life.last_seen_frame = row.last_seen_frame,
            life.first_seen_ts = row.first_seen_ts,
            life.last_seen_ts = row.last_seen_ts,
            life.observation_count = row.observation_count,
            life.previous_segment_id = row.previous_segment_id,
            life.next_segment_id = row.next_segment_id,
            life.gap_from_previous_frames = row.gap_from_previous_frames,
            life.gap_to_next_frames = row.gap_to_next_frames,
            life.gap_from_previous_ranks = row.gap_from_previous_ranks,
            life.gap_to_next_ranks = row.gap_to_next_ranks,
            life.resumes_after_gap = row.resumes_after_gap,
            life.is_static = row.is_static,
            life.is_noise = row.is_noise,
            life.is_fragment = row.is_fragment
        MERGE (e)-[:HAS_LIFECYCLE]->(life)
        MERGE (life)-[:STARTS_AT]->(start)
        MERGE (life)-[:ENDS_AT]->(end)
        WITH life, row
        OPTIONAL MATCH (scene:SceneSegment {scene_id: row.scene_id})
        FOREACH (_ IN CASE WHEN scene IS NULL THEN [] ELSE [1] END |
            MERGE (scene)-[:CONTAINS_LIFECYCLE]->(life)
        )
        WITH life, row
        OPTIONAL MATCH (seg:EntitySegment {segment_id: row.source_segment_id})
        FOREACH (_ IN CASE WHEN seg IS NULL THEN [] ELSE [1] END |
            MERGE (life)-[:BACKED_BY_SEGMENT]->(seg)
        )
        """,
        rows=rows,
    )


def upsert_entity_lifecycle_links(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (current:EntityLifecycle {lifecycle_id: row.lifecycle_id})
        MATCH (next:EntityLifecycle {lifecycle_id: row.next_lifecycle_id})
        MERGE (current)-[rel:NEXT_LIFECYCLE]->(next)
        SET rel.entity_id = row.entity_id,
            rel.gap_frames = row.gap_frames,
            rel.gap_ranks = row.gap_ranks
        """,
        rows=rows,
    )


def upsert_scene_segments(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (s:SceneSegment {scene_id: row.scene_id})
        SET s.video_id = row.video_id,
            s.scene_index = row.scene_index,
            s.scene_count = row.scene_count,
            s.start_frame_rank = row.start_frame_rank,
            s.end_frame_rank = row.end_frame_rank,
            s.start_frame = row.start_frame,
            s.end_frame = row.end_frame,
            s.start_ts = row.start_ts,
            s.end_ts = row.end_ts,
            s.support_frames = row.support_frames,
            s.previous_scene_id = row.previous_scene_id,
            s.next_scene_id = row.next_scene_id,
            s.boundary_from_previous_frames = row.boundary_from_previous_frames,
            s.boundary_to_next_frames = row.boundary_to_next_frames,
            s.boundary_from_previous_time_gap_sec = row.boundary_from_previous_time_gap_sec,
            s.boundary_to_next_time_gap_sec = row.boundary_to_next_time_gap_sec,
            s.boundary_from_previous_entity_similarity = row.boundary_from_previous_entity_similarity,
            s.boundary_to_next_entity_similarity = row.boundary_to_next_entity_similarity,
            s.boundary_from_previous_label_similarity = row.boundary_from_previous_label_similarity,
            s.boundary_to_next_label_similarity = row.boundary_to_next_label_similarity,
            s.boundary_from_previous_reason = row.boundary_from_previous_reason,
            s.boundary_to_next_reason = row.boundary_to_next_reason,
            s.entry_static_labels = row.entry_static_labels,
            s.exit_static_labels = row.exit_static_labels,
            s.representative_static_entity_ids = row.representative_static_entity_ids,
            s.representative_static_labels = row.representative_static_labels,
            s.representative_dynamic_labels = row.representative_dynamic_labels,
            s.scene_kind_hint = row.scene_kind_hint
        WITH s, row
        OPTIONAL MATCH (start:Frame {frame_id: row.start_frame_id})
        FOREACH (_ IN CASE WHEN start IS NULL THEN [] ELSE [1] END |
            MERGE (s)-[:STARTS_AT]->(start)
        )
        WITH s, row
        OPTIONAL MATCH (end:Frame {frame_id: row.end_frame_id})
        FOREACH (_ IN CASE WHEN end IS NULL THEN [] ELSE [1] END |
            MERGE (s)-[:ENDS_AT]->(end)
        )
        """,
        rows=rows,
    )


def upsert_scene_links(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (current:SceneSegment {scene_id: row.scene_id})
        MATCH (next:SceneSegment {scene_id: row.next_scene_id})
        MERGE (current)-[rel:NEXT_SCENE]->(next)
        SET rel.frame_gap = row.frame_gap,
            rel.time_gap_sec = row.time_gap_sec,
            rel.boundary_reason = row.boundary_reason,
            rel.entity_similarity = row.entity_similarity,
            rel.label_similarity = row.label_similarity,
            rel.exit_static_labels = row.exit_static_labels,
            rel.entry_static_labels = row.entry_static_labels
        """,
        rows=rows,
    )


def upsert_scene_frame_edges(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (frame:Frame {frame_id: row.frame_id})
        MERGE (scene)-[:INCLUDES_FRAME]->(frame)
        MERGE (frame)-[:IN_SCENE]->(scene)
        """,
        rows=rows,
    )


def upsert_entity_segments(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MERGE (seg:EntitySegment {segment_id: row.segment_id})
        SET seg.video_id = row.video_id,
            seg.segment_type = row.segment_type,
            seg.entity_id = row.entity_id,
            seg.scene_id = row.scene_id,
            seg.segment_index = row.segment_index,
            seg.segment_count = row.segment_count,
            seg.start_frame_rank = row.start_frame_rank,
            seg.end_frame_rank = row.end_frame_rank,
            seg.start_frame = row.start_frame,
            seg.end_frame = row.end_frame,
            seg.start_ts = row.start_ts,
            seg.end_ts = row.end_ts,
            seg.support_frames = row.support_frames,
            seg.previous_segment_id = row.previous_segment_id,
            seg.next_segment_id = row.next_segment_id,
            seg.gap_from_previous_frames = row.gap_from_previous_frames,
            seg.gap_to_next_frames = row.gap_to_next_frames,
            seg.gap_from_previous_ranks = row.gap_from_previous_ranks,
            seg.gap_to_next_ranks = row.gap_to_next_ranks,
            seg.resumes_after_gap = row.resumes_after_gap,
            seg.is_static = row.is_static,
            seg.is_noise = row.is_noise
        MERGE (e)-[:HAS_SEGMENT]->(seg)
        WITH seg, row
        OPTIONAL MATCH (scene:SceneSegment {scene_id: row.scene_id})
        FOREACH (_ IN CASE WHEN scene IS NULL THEN [] ELSE [1] END |
            MERGE (scene)-[:CONTAINS_SEGMENT]->(seg)
        )
        """,
        rows=rows,
    )


def upsert_entity_segment_links(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (current:EntitySegment {segment_id: row.segment_id})
        MATCH (next:EntitySegment {segment_id: row.next_segment_id})
        MERGE (current)-[rel:NEXT_SEGMENT]->(next)
        SET rel.entity_id = row.entity_id,
            rel.gap_frames = row.gap_frames,
            rel.gap_ranks = row.gap_ranks
        """,
        rows=rows,
    )


def upsert_scene_base_nodes(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (entity:Entity {entity_id: row.entity_id})
        MERGE (node:SceneBaseNode {node_id: row.node_id})
        SET node.scene_id = row.scene_id,
            node.video_id = row.video_id,
            node.entity_id = row.entity_id,
            node.label = row.label,
            node.first_frame = row.first_frame,
            node.last_frame = row.last_frame,
            node.support_frames = row.support_frames,
            node.coverage_ratio = row.coverage_ratio,
            node.canonical_image_path = row.canonical_image_path,
            node.parent_labels_seen = row.parent_labels_seen,
            node.states_seen = row.states_seen
        MERGE (scene)-[:HAS_BASE_NODE]->(node)
        MERGE (node)-[:ANCHORS_ENTITY]->(entity)
        WITH node, row
        OPTIONAL MATCH (start:Frame {frame_id: row.first_frame_id})
        FOREACH (_ IN CASE WHEN start IS NULL THEN [] ELSE [1] END |
            MERGE (node)-[:STARTS_AT]->(start)
        )
        WITH node, row
        OPTIONAL MATCH (end:Frame {frame_id: row.last_frame_id})
        FOREACH (_ IN CASE WHEN end IS NULL THEN [] ELSE [1] END |
            MERGE (node)-[:ENDS_AT]->(end)
        )
        """,
        rows=rows,
    )


def upsert_scene_base_edges(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (target:SceneBaseNode {node_id: row.target_id})
        MERGE (scene)-[r:SCENE_CONTAINS_BASE {edge_id: row.edge_id}]->(target)
        SET r.video_id = row.video_id,
            r.predicate = row.predicate,
            r.entity_id = row.entity_id,
            r.support_frames = row.support_frames,
            r.coverage_ratio = row.coverage_ratio
        """,
        rows=rows,
    )


def upsert_scene_delta_nodes(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (entity:Entity {entity_id: row.entity_id})
        MERGE (node:SceneDeltaNode {node_id: row.node_id})
        SET node.scene_id = row.scene_id,
            node.video_id = row.video_id,
            node.entity_id = row.entity_id,
            node.label = row.label,
            node.start_frame = row.start_frame,
            node.end_frame = row.end_frame,
            node.start_ts = row.start_ts,
            node.end_ts = row.end_ts,
            node.support_frames = row.support_frames,
            node.transition_event_count = row.transition_event_count,
            node.relation_episode_count = row.relation_episode_count,
            node.state_episode_count = row.state_episode_count,
            node.graph_origin = row.graph_origin
        MERGE (scene)-[:HAS_DELTA_NODE]->(node)
        MERGE (node)-[:TRACKS_ENTITY]->(entity)
        WITH node, row
        OPTIONAL MATCH (start:Frame {frame_id: row.start_frame_id})
        FOREACH (_ IN CASE WHEN start IS NULL THEN [] ELSE [1] END |
            MERGE (node)-[:STARTS_AT]->(start)
        )
        WITH node, row
        OPTIONAL MATCH (end:Frame {frame_id: row.end_frame_id})
        FOREACH (_ IN CASE WHEN end IS NULL THEN [] ELSE [1] END |
            MERGE (node)-[:ENDS_AT]->(end)
        )
        """,
        rows=rows,
    )


def upsert_scene_delta_observations(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (entity:Entity {entity_id: row.entity_id})
        MATCH (delta:SceneDeltaNode {node_id: row.scene_delta_node_id})
        MERGE (obs:SceneDeltaObservation {delta_observation_id: row.delta_observation_id})
        SET obs.scene_id = row.scene_id,
            obs.video_id = row.video_id,
            obs.entity_id = row.entity_id,
            obs.scene_delta_node_id = row.scene_delta_node_id,
            obs.source_segment_id = row.source_segment_id,
            obs.source_observation_id = row.source_observation_id,
            obs.frame_name = row.frame_name,
            obs.frame_rank = row.frame_rank,
            obs.frame_index = row.frame_index,
            obs.timestamp_sec = row.timestamp_sec,
            obs.image_path = row.image_path,
            obs.label = row.label,
            obs.trigger_event_ids = row.trigger_event_ids,
            obs.trigger_event_types = row.trigger_event_types,
            obs.marks_segment_start = row.marks_segment_start,
            obs.marks_segment_end = row.marks_segment_end,
            obs.graph_origin = row.graph_origin,
            obs.bbox_x1 = row.bbox_x1,
            obs.bbox_y1 = row.bbox_y1,
            obs.bbox_x2 = row.bbox_x2,
            obs.bbox_y2 = row.bbox_y2
        MERGE (scene)-[:HAS_DELTA_OBSERVATION]->(obs)
        MERGE (delta)-[:HAS_DELTA_OBSERVATION]->(obs)
        MERGE (obs)-[:OBSERVES_ENTITY]->(entity)
        WITH obs, row
        OPTIONAL MATCH (frame:Frame {frame_id: row.frame_id})
        FOREACH (_ IN CASE WHEN frame IS NULL THEN [] ELSE [1] END |
            MERGE (obs)-[:OBSERVED_AT]->(frame)
            MERGE (frame)-[:HAS_DELTA_OBSERVATION]->(obs)
        )
        WITH obs, row
        OPTIONAL MATCH (source:Observation {observation_id: row.source_observation_id})
        FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
            MERGE (obs)-[:BACKED_BY_OBSERVATION]->(source)
        )
        """,
        rows=rows,
    )


def upsert_scene_delta_observation_event_edges(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (obs:SceneDeltaObservation {delta_observation_id: row.delta_observation_id})
        UNWIND row.trigger_event_ids AS event_id
        MATCH (ev:TransitionEvent {event_id: event_id})
        MERGE (obs)-[:TRIGGERED_BY_EVENT]->(ev)
        """,
        rows=rows,
    )


def upsert_scene_delta_edges(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (scene:SceneSegment {scene_id: row.scene_id})
        MATCH (target:SceneDeltaNode {node_id: row.target_id})
        MERGE (scene)-[r:SCENE_UPDATES_DELTA {edge_id: row.edge_id}]->(target)
        SET r.video_id = row.video_id,
            r.predicate = row.predicate,
            r.entity_id = row.entity_id,
            r.support_frames = row.support_frames,
            r.transition_event_count = row.transition_event_count
        """,
        rows=rows,
    )


def upsert_transition_events(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (ev:TransitionEvent {event_id: row.event_id})
        SET ev.video_id = row.video_id,
            ev.event_type = row.event_type,
            ev.frame_rank = row.frame_rank,
            ev.frame_index = row.frame_index,
            ev.timestamp_sec = row.timestamp_sec,
            ev.entity_id = row.entity_id,
            ev.subject_entity_id = row.subject_entity_id,
            ev.predicate = row.predicate,
            ev.raw_predicate = row.raw_predicate,
            ev.object_entity_id = row.object_entity_id,
            ev.attr_key = row.attr_key,
            ev.previous_state_value = row.previous_state_value,
            ev.current_state_value = row.current_state_value,
            ev.source_segment_id = row.source_segment_id,
            ev.source_episode_id = row.source_episode_id
        WITH ev, row
        OPTIONAL MATCH (frame:Frame {frame_id: row.frame_id})
        FOREACH (_ IN CASE WHEN frame IS NULL THEN [] ELSE [1] END |
            MERGE (ev)-[:OCCURS_AT]->(frame)
        )
        WITH ev, row
        OPTIONAL MATCH (seg:EntitySegment {segment_id: row.source_segment_id})
        FOREACH (_ IN CASE WHEN seg IS NULL THEN [] ELSE [1] END |
            MERGE (seg)-[:EMITS_EVENT]->(ev)
        )
        FOREACH (_ IN CASE
            WHEN seg IS NULL OR row.event_type <> 'APPEAR' THEN []
            ELSE [1]
        END |
            MERGE (ev)-[:STARTS_SEGMENT]->(seg)
        )
        FOREACH (_ IN CASE
            WHEN seg IS NULL OR row.event_type <> 'DISAPPEAR' THEN []
            ELSE [1]
        END |
            MERGE (ev)-[:ENDS_SEGMENT]->(seg)
        )
        WITH ev, row
        OPTIONAL MATCH (rel_ep:RelationEpisode {relation_episode_id: row.source_episode_id})
        FOREACH (_ IN CASE
            WHEN rel_ep IS NULL OR row.event_type <> 'RELATION_START' THEN []
            ELSE [1]
        END |
            MERGE (ev)-[:STARTS_RELATION_EPISODE]->(rel_ep)
        )
        FOREACH (_ IN CASE
            WHEN rel_ep IS NULL OR row.event_type <> 'RELATION_END' THEN []
            ELSE [1]
        END |
            MERGE (ev)-[:ENDS_RELATION_EPISODE]->(rel_ep)
        )
        WITH ev, row
        OPTIONAL MATCH (state_ep:StateEpisode {state_episode_id: row.source_episode_id})
        FOREACH (_ IN CASE
            WHEN state_ep IS NULL OR row.event_type <> 'STATE_TRANSITION' THEN []
            ELSE [1]
        END |
            MERGE (ev)-[:STARTS_STATE_EPISODE]->(state_ep)
        )
        WITH ev, row
        OPTIONAL MATCH (entity:Entity {entity_id: row.entity_id})
        FOREACH (_ IN CASE WHEN entity IS NULL THEN [] ELSE [1] END |
            MERGE (entity)-[:HAS_TRANSITION]->(ev)
        )
        WITH ev, row
        OPTIONAL MATCH (subject:Entity {entity_id: row.subject_entity_id})
        FOREACH (_ IN CASE WHEN subject IS NULL THEN [] ELSE [1] END |
            MERGE (subject)-[:SUBJECT_OF_EVENT]->(ev)
        )
        WITH ev, row
        OPTIONAL MATCH (object:Entity {entity_id: row.object_entity_id})
        FOREACH (_ IN CASE WHEN object IS NULL THEN [] ELSE [1] END |
            MERGE (ev)-[:TARGETS_ENTITY]->(object)
        )
        """,
        rows=rows,
    )


def upsert_relation_episode_nodes(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Entity {entity_id: row.subject_entity_id})
        MATCH (o:Entity {entity_id: row.object_entity_id})
        MATCH (start:Frame {frame_id: row.start_frame_id})
        MATCH (end:Frame {frame_id: row.end_frame_id})
        MERGE (ep:RelationEpisode {relation_episode_id: row.relation_episode_id})
        SET ep.video_id = row.video_id,
            ep.subject_entity_id = row.subject_entity_id,
            ep.object_entity_id = row.object_entity_id,
            ep.predicate = row.predicate,
            ep.raw_predicate = row.raw_predicate,
            ep.start_frame = row.start_frame,
            ep.end_frame = row.end_frame,
            ep.start_ts = row.start_ts,
            ep.end_ts = row.end_ts,
            ep.support_frames = row.support_frames
        MERGE (s)-[:SUBJECT_OF_EPISODE]->(ep)
        MERGE (ep)-[:OBJECT_OF_EPISODE]->(o)
        MERGE (ep)-[:STARTS_AT]->(start)
        MERGE (ep)-[:ENDS_AT]->(end)
        """,
        rows=rows,
    )


def upsert_state_episode_nodes(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MATCH (start:Frame {frame_id: row.start_frame_id})
        MATCH (end:Frame {frame_id: row.end_frame_id})
        MERGE (ep:StateEpisode {state_episode_id: row.state_episode_id})
        SET ep.video_id = row.video_id,
            ep.entity_id = row.entity_id,
            ep.attr_key = row.attr_key,
            ep.attr_value = row.attr_value,
            ep.start_frame = row.start_frame,
            ep.end_frame = row.end_frame,
            ep.start_ts = row.start_ts,
            ep.end_ts = row.end_ts,
            ep.support_frames = row.support_frames
        MERGE (e)-[:HAS_STATE_EPISODE]->(ep)
        MERGE (ep)-[:STARTS_AT]->(start)
        MERGE (ep)-[:ENDS_AT]->(end)
        """,
        rows=rows,
    )


def upsert_relation_episodes(tx, rel_type: str, rows: list[dict]) -> None:
    query = f"""
    UNWIND $rows AS row
    MATCH (s:Entity {{entity_id: row.subject_entity_id}})
    MATCH (o:Entity {{entity_id: row.object_entity_id}})
    MERGE (s)-[r:{rel_type} {{relation_key: row.relation_key}}]->(o)
    SET r.predicate = row.predicate,
        r.raw_predicates = row.raw_predicates,
        r.video_id = row.video_id,
        r.start_frame = row.start_frame,
        r.end_frame = row.end_frame,
        r.start_ts = row.start_ts,
        r.end_ts = row.end_ts,
        r.support_frames = row.support_frames,
        r.episode_count = row.episode_count,
        r.frame_ranges = row.frame_ranges
    """
    tx.run(query, rows=rows)


def upsert_state_episodes(tx, rows: list[dict]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MERGE (e)-[r:HAS_STATE {state_key: row.state_key}]->(e)
        SET r.video_id = row.video_id,
            r.attr_key = row.attr_key,
            r.attr_value = row.attr_value,
            r.start_frame = row.start_frame,
            r.end_frame = row.end_frame,
            r.start_ts = row.start_ts,
            r.end_ts = row.end_ts,
            r.support_frames = row.support_frames,
            r.episode_count = row.episode_count,
            r.frame_ranges = row.frame_ranges
        """,
        rows=rows,
    )


def aggregate_relation_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            row["subject_entity_id"],
            row["predicate"],
            row["object_entity_id"],
        )
        aggregated = grouped.get(key)
        if aggregated is None:
            grouped[key] = {
                "relation_key": "|".join(key),
                "subject_entity_id": row["subject_entity_id"],
                "predicate": row["predicate"],
                "object_entity_id": row["object_entity_id"],
                "video_id": row["video_id"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "support_frames": row["support_frames"],
                "episode_count": 1,
                "raw_predicates": sorted(
                    {value for value in [row.get("raw_predicate")] if value}
                ),
                "frame_ranges": [f"{row['start_frame']}-{row['end_frame']}"],
            }
            continue

        aggregated["start_frame"] = min(aggregated["start_frame"], row["start_frame"])
        aggregated["end_frame"] = max(aggregated["end_frame"], row["end_frame"])
        aggregated["start_ts"] = min(aggregated["start_ts"], row["start_ts"])
        aggregated["end_ts"] = max(aggregated["end_ts"], row["end_ts"])
        aggregated["support_frames"] += row["support_frames"]
        aggregated["episode_count"] += 1
        raw_predicates = set(aggregated["raw_predicates"])
        if row.get("raw_predicate"):
            raw_predicates.add(row["raw_predicate"])
        aggregated["raw_predicates"] = sorted(raw_predicates)
        aggregated["frame_ranges"].append(f"{row['start_frame']}-{row['end_frame']}")

    return list(grouped.values())


def aggregate_state_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            row["entity_id"],
            row["attr_key"],
            row["attr_value"],
        )
        aggregated = grouped.get(key)
        if aggregated is None:
            grouped[key] = {
                "state_key": "|".join(key),
                "entity_id": row["entity_id"],
                "video_id": row["video_id"],
                "attr_key": row["attr_key"],
                "attr_value": row["attr_value"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "support_frames": row["support_frames"],
                "episode_count": 1,
                "frame_ranges": [f"{row['start_frame']}-{row['end_frame']}"],
            }
            continue

        aggregated["start_frame"] = min(aggregated["start_frame"], row["start_frame"])
        aggregated["end_frame"] = max(aggregated["end_frame"], row["end_frame"])
        aggregated["start_ts"] = min(aggregated["start_ts"], row["start_ts"])
        aggregated["end_ts"] = max(aggregated["end_ts"], row["end_ts"])
        aggregated["support_frames"] += row["support_frames"]
        aggregated["episode_count"] += 1
        aggregated["frame_ranges"].append(f"{row['start_frame']}-{row['end_frame']}")

    return list(grouped.values())


def _frame_id(video_id: str, frame_index: int) -> str:
    return f"{video_id}:frame:{frame_index:06d}"


def build_frame_rows(document: dict) -> list[dict]:
    frame_map: dict[int, dict] = {}
    video_id = str(document["video_id"])

    def ensure_frame(
        frame_index: int,
        *,
        frame_rank: int | None = None,
        timestamp_sec: float | None = None,
        frame_name: str | None = None,
        image_path: str | None = None,
    ) -> None:
        row = frame_map.get(frame_index)
        if row is None:
            frame_map[frame_index] = {
                "frame_id": _frame_id(video_id, frame_index),
                "video_id": video_id,
                "frame_index": frame_index,
                "frame_rank": frame_rank if frame_rank is not None else frame_index,
                "timestamp_sec": timestamp_sec if timestamp_sec is not None else float(frame_index),
                "frame_name": frame_name,
                "image_path": image_path,
            }
            return
        if frame_rank is not None:
            row["frame_rank"] = min(int(row["frame_rank"]), frame_rank)
        if timestamp_sec is not None:
            row["timestamp_sec"] = min(float(row["timestamp_sec"]), timestamp_sec)
        if not row.get("frame_name") and frame_name:
            row["frame_name"] = frame_name
        if not row.get("image_path") and image_path:
            row["image_path"] = image_path

    for observation in document.get("observations", []):
        ensure_frame(
            int(observation["frame_index"]),
            frame_rank=int(observation.get("frame_rank", observation["frame_index"])),
            timestamp_sec=float(observation.get("timestamp_sec", observation["frame_index"])),
            frame_name=observation.get("frame_name"),
            image_path=observation.get("image_path"),
        )

    for entity in document.get("entities", []):
        ensure_frame(
            int(entity["first_seen_frame"]),
            timestamp_sec=float(entity.get("first_seen_ts", entity["first_seen_frame"])),
        )
        ensure_frame(
            int(entity["last_seen_frame"]),
            timestamp_sec=float(entity.get("last_seen_ts", entity["last_seen_frame"])),
        )

    for episode in document.get("relation_episodes", []):
        ensure_frame(
            int(episode["start_frame"]),
            timestamp_sec=float(episode.get("start_ts", episode["start_frame"])),
        )
        ensure_frame(
            int(episode["end_frame"]),
            timestamp_sec=float(episode.get("end_ts", episode["end_frame"])),
        )

    for episode in document.get("state_episodes", []):
        ensure_frame(
            int(episode["start_frame"]),
            timestamp_sec=float(episode.get("start_ts", episode["start_frame"])),
        )
        ensure_frame(
            int(episode["end_frame"]),
            timestamp_sec=float(episode.get("end_ts", episode["end_frame"])),
        )

    return [frame_map[index] for index in sorted(frame_map.keys())]


def build_next_frame_rows(frame_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index in range(len(frame_rows) - 1):
        source = frame_rows[index]
        target = frame_rows[index + 1]
        rows.append(
            {
                "video_id": source["video_id"],
                "source_frame_id": source["frame_id"],
                "target_frame_id": target["frame_id"],
                "frame_gap": int(target["frame_index"]) - int(source["frame_index"]),
                "time_delta_sec": float(target["timestamp_sec"]) - float(source["timestamp_sec"]),
            }
        )
    return rows


def build_observation_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for observation in document.get("observations", []):
        bbox = observation.get("bbox", {})
        rows.append(
            {
                "observation_id": observation["observation_id"],
                "video_id": observation["video_id"],
                "entity_id": observation["entity_id"],
                "frame_id": _frame_id(observation["video_id"], int(observation["frame_index"])),
                "frame_name": observation["frame_name"],
                "frame_rank": observation["frame_rank"],
                "frame_index": observation["frame_index"],
                "timestamp_sec": observation["timestamp_sec"],
                "image_path": observation["image_path"],
                "object_index": observation["object_index"],
                "label": observation["label"],
                "score": observation["score"],
                "state": observation.get("state"),
                "state_values": observation.get("state_values", []),
                "parent_label": observation.get("parent_label"),
                "is_static": observation.get("is_static", False),
                "is_noise": observation.get("is_noise", False),
                "bbox_x1": bbox.get("x1"),
                "bbox_y1": bbox.get("y1"),
                "bbox_x2": bbox.get("x2"),
                "bbox_y2": bbox.get("y2"),
            }
        )
    return rows


def build_entity_lifecycle_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    entity_lookup = {
        entity["entity_id"]: entity
        for entity in document.get("entities", [])
    }
    for segment in document.get("entity_segments", []):
        entity = entity_lookup.get(segment["entity_id"])
        if entity is None:
            continue
        rows.append(
            {
                "lifecycle_id": f"{segment['segment_id']}:lifecycle",
                "video_id": segment["video_id"],
                "entity_id": segment["entity_id"],
                "category": entity["category"],
                "scene_id": segment.get("scene_id"),
                "source_segment_id": segment["segment_id"],
                "segment_index": segment.get("segment_index", 1),
                "segment_count": segment.get("segment_count", 1),
                "first_seen_frame": segment["start_frame"],
                "last_seen_frame": segment["end_frame"],
                "first_seen_ts": segment["start_ts"],
                "last_seen_ts": segment["end_ts"],
                "observation_count": segment["support_frames"],
                "previous_segment_id": segment.get("previous_segment_id"),
                "next_segment_id": segment.get("next_segment_id"),
                "gap_from_previous_frames": segment.get("gap_from_previous_frames"),
                "gap_to_next_frames": segment.get("gap_to_next_frames"),
                "gap_from_previous_ranks": segment.get("gap_from_previous_ranks"),
                "gap_to_next_ranks": segment.get("gap_to_next_ranks"),
                "resumes_after_gap": segment.get("resumes_after_gap", False),
                "is_static": entity.get("is_static", False),
                "is_noise": entity.get("is_noise", False),
                "is_fragment": entity.get("is_fragment", False),
                "start_frame_id": _frame_id(segment["video_id"], int(segment["start_frame"])),
                "end_frame_id": _frame_id(segment["video_id"], int(segment["end_frame"])),
            }
        )
    return rows


def build_entity_segment_link_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for segment in document.get("entity_segments", []):
        next_segment_id = segment.get("next_segment_id")
        if not next_segment_id:
            continue
        rows.append(
            {
                "segment_id": segment["segment_id"],
                "next_segment_id": next_segment_id,
                "entity_id": segment["entity_id"],
                "gap_frames": segment.get("gap_to_next_frames"),
                "gap_ranks": segment.get("gap_to_next_ranks"),
            }
        )
    return rows


def build_entity_lifecycle_link_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for segment in document.get("entity_segments", []):
        next_segment_id = segment.get("next_segment_id")
        if not next_segment_id:
            continue
        rows.append(
            {
                "lifecycle_id": f"{segment['segment_id']}:lifecycle",
                "next_lifecycle_id": f"{next_segment_id}:lifecycle",
                "entity_id": segment["entity_id"],
                "gap_frames": segment.get("gap_to_next_frames"),
                "gap_ranks": segment.get("gap_to_next_ranks"),
            }
        )
    return rows


def build_relation_episode_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for episode in document.get("relation_episodes", []):
        row = dict(episode)
        row["start_frame_id"] = _frame_id(episode["video_id"], int(episode["start_frame"]))
        row["end_frame_id"] = _frame_id(episode["video_id"], int(episode["end_frame"]))
        rows.append(row)
    return rows


def build_scene_segment_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for scene in document.get("scene_segments", []):
        row = dict(scene)
        row["start_frame_id"] = _frame_id(scene["video_id"], int(scene["start_frame"]))
        row["end_frame_id"] = _frame_id(scene["video_id"], int(scene["end_frame"]))
        rows.append(row)
    return rows


def build_scene_link_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for scene in document.get("scene_segments", []):
        next_scene_id = scene.get("next_scene_id")
        if not next_scene_id:
            continue
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "next_scene_id": next_scene_id,
                "frame_gap": scene.get("boundary_to_next_frames"),
                "time_gap_sec": scene.get("boundary_to_next_time_gap_sec"),
                "boundary_reason": scene.get("boundary_to_next_reason"),
                "entity_similarity": scene.get("boundary_to_next_entity_similarity"),
                "label_similarity": scene.get("boundary_to_next_label_similarity"),
                "exit_static_labels": scene.get("exit_static_labels", []),
                "entry_static_labels": [],
            }
        )
    entry_labels_by_scene = {
        scene["scene_id"]: scene.get("entry_static_labels", [])
        for scene in document.get("scene_segments", [])
    }
    for row in rows:
        row["entry_static_labels"] = entry_labels_by_scene.get(row["next_scene_id"], [])
    return rows


def build_scene_frame_rows(
    scene_rows: list[dict],
    frame_rows: list[dict],
) -> list[dict]:
    frame_indices = {int(row["frame_index"]): row["frame_id"] for row in frame_rows}
    rows: list[dict] = []
    for scene in scene_rows:
        start_frame = int(scene["start_frame"])
        end_frame = int(scene["end_frame"])
        for frame_index, frame_id in frame_indices.items():
            if start_frame <= frame_index <= end_frame:
                rows.append(
                    {
                        "scene_id": scene["scene_id"],
                        "frame_id": frame_id,
                    }
                )
    return rows


def build_state_episode_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for episode in document.get("state_episodes", []):
        row = dict(episode)
        row["start_frame_id"] = _frame_id(episode["video_id"], int(episode["start_frame"]))
        row["end_frame_id"] = _frame_id(episode["video_id"], int(episode["end_frame"]))
        rows.append(row)
    return rows


def build_scene_delta_observation_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for observation in document.get("scene_delta_observations", []):
        row = dict(observation)
        bbox = observation.get("bbox", {})
        row["bbox_x1"] = bbox.get("x1")
        row["bbox_y1"] = bbox.get("y1")
        row["bbox_x2"] = bbox.get("x2")
        row["bbox_y2"] = bbox.get("y2")
        row["frame_id"] = _frame_id(observation["video_id"], int(observation["frame_index"]))
        row.pop("bbox", None)
        rows.append(row)
    return rows


def build_scene_base_node_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for node in document.get("scene_base_nodes", []):
        row = dict(node)
        row["first_frame_id"] = _frame_id(node["video_id"], int(node["first_frame"]))
        row["last_frame_id"] = _frame_id(node["video_id"], int(node["last_frame"]))
        rows.append(row)
    return rows


def build_scene_delta_node_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for node in document.get("scene_delta_nodes", []):
        row = dict(node)
        row["start_frame_id"] = _frame_id(node["video_id"], int(node["start_frame"]))
        row["end_frame_id"] = _frame_id(node["video_id"], int(node["end_frame"]))
        rows.append(row)
    return rows


def build_transition_event_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for event in document.get("transition_events", []):
        row = dict(event)
        row["frame_id"] = _frame_id(event["video_id"], int(event["frame_index"]))
        rows.append(row)
    return rows


def export_document(
    document: dict,
    uri: str,
    user: str,
    password: str,
    *,
    reset_db: bool = False,
) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            if reset_db:
                session.execute_write(reset_database)

            entity_rows = []
            for entity in document["entities"]:
                row = dict(entity)
                row["display_name"] = f"{entity['category']} | {entity['entity_id']}"
                entity_rows.append(row)
            for rows in chunked(entity_rows):
                session.execute_write(upsert_entities, rows)

            frame_rows = build_frame_rows(document)
            for rows in chunked(frame_rows):
                session.execute_write(upsert_frames, rows)

            next_frame_rows = build_next_frame_rows(frame_rows)
            for rows in chunked(next_frame_rows):
                session.execute_write(upsert_next_frame_edges, rows)

            observation_rows = build_observation_rows(document)
            for rows in chunked(observation_rows):
                session.execute_write(upsert_observations, rows)

            scene_segment_rows = build_scene_segment_rows(document)
            for rows in chunked(scene_segment_rows):
                session.execute_write(upsert_scene_segments, rows)

            scene_link_rows = build_scene_link_rows(document)
            for rows in chunked(scene_link_rows):
                session.execute_write(upsert_scene_links, rows)

            scene_frame_rows = build_scene_frame_rows(scene_segment_rows, frame_rows)
            for rows in chunked(scene_frame_rows):
                session.execute_write(upsert_scene_frame_edges, rows)

            for rows in chunked(document.get("entity_segments", [])):
                session.execute_write(upsert_entity_segments, rows)

            entity_segment_link_rows = build_entity_segment_link_rows(document)
            for rows in chunked(entity_segment_link_rows):
                session.execute_write(upsert_entity_segment_links, rows)

            lifecycle_rows = build_entity_lifecycle_rows(document)
            for rows in chunked(lifecycle_rows):
                session.execute_write(upsert_entity_lifecycles, rows)

            lifecycle_link_rows = build_entity_lifecycle_link_rows(document)
            for rows in chunked(lifecycle_link_rows):
                session.execute_write(upsert_entity_lifecycle_links, rows)

            scene_base_node_rows = build_scene_base_node_rows(document)
            for rows in chunked(scene_base_node_rows):
                session.execute_write(upsert_scene_base_nodes, rows)

            for rows in chunked(document.get("scene_base_edges", [])):
                session.execute_write(upsert_scene_base_edges, rows)

            scene_delta_node_rows = build_scene_delta_node_rows(document)
            for rows in chunked(scene_delta_node_rows):
                session.execute_write(upsert_scene_delta_nodes, rows)

            scene_delta_observation_rows = build_scene_delta_observation_rows(document)
            for rows in chunked(scene_delta_observation_rows):
                session.execute_write(upsert_scene_delta_observations, rows)

            for rows in chunked(document.get("scene_delta_edges", [])):
                session.execute_write(upsert_scene_delta_edges, rows)

            relation_episode_rows = build_relation_episode_rows(document)
            for rows in chunked(relation_episode_rows):
                session.execute_write(upsert_relation_episode_nodes, rows)

            state_episode_rows = build_state_episode_rows(document)
            for rows in chunked(state_episode_rows):
                session.execute_write(upsert_state_episode_nodes, rows)

            transition_event_rows = build_transition_event_rows(document)
            for rows in chunked(transition_event_rows):
                session.execute_write(upsert_transition_events, rows)

            for rows in chunked(scene_delta_observation_rows):
                session.execute_write(upsert_scene_delta_observation_event_edges, rows)

            grouped_relations: dict[str, list[dict]] = defaultdict(list)
            aggregated_relation_rows = aggregate_relation_rows(document["relation_episodes"])
            for episode in aggregated_relation_rows:
                rel_type = sanitize_relationship_type(episode["predicate"])
                grouped_relations[rel_type].append(episode)

            for rel_type, relation_rows in grouped_relations.items():
                for rows in chunked(relation_rows):
                    session.execute_write(upsert_relation_episodes, rel_type, rows)

            aggregated_state_rows = aggregate_state_rows(document["state_episodes"])
            for rows in chunked(aggregated_state_rows):
                session.execute_write(upsert_state_episodes, rows)
    finally:
        driver.close()


def main() -> None:
    args = build_parser().parse_args()
    document = load_graph_document(resolve_repo_input_path(args.input_json))
    export_document(
        document=document,
        uri=args.uri,
        user=args.user,
        password=args.password,
        reset_db=args.reset_db,
    )
    print("Neo4j export completed.")


if __name__ == "__main__":
    main()
