from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval as openeqa_retrieval
import vlkg.core.graph_retrieval.canonical_hybrid_retrieval as navqa_retrieval
from vlkg.datasets.openeqa.runtime.canonical_record_vector_recall import (
    record_semantic_text as openeqa_record_semantic_text,
)
from vlkg.core.graph_retrieval.canonical_retrieval_types import GraphRecordView
from vlkg.core.config.config import GraphQAConfig
from vlkg.datasets.navqa.qa.canonical_graph_qa_context import (
    answer_canonical_graph_query as answer_navqa_canonical_graph_query,
)
from vlkg.datasets.openeqa.qa.canonical_graph_qa_context import (
    answer_canonical_graph_query as answer_openeqa_canonical_graph_query,
)
from vlkg.datasets.openeqa.qa.graph_qa_answering import (
    OpenAICompatibleGraphQAResponder as OpenEQAGraphQAResponder,
)
from vlkg.datasets.navqa.qa.graph_qa_answering import (
    OpenAICompatibleGraphQAResponder as FrozenGraphQAResponder,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import (
    Entity,
    RelationEpisode,
    SceneSegment,
    StateEpisode,
    TemporalGraphDocument,
)
from vlkg.core.transformers import build_canonical_temporal_memory_graph


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())


def _sample_graph():
    document = TemporalGraphDocument(
        video_id="parity_video",
        source_dir=Path("data/examples/parity_video"),
        frame_count=12,
        observations=[],
        entities=[
            Entity(
                entity_id="chair_1",
                video_id="parity_video",
                category="chair",
                first_seen_frame=0,
                last_seen_frame=11,
                first_seen_ts=0.0,
                last_seen_ts=11.0,
                observation_count=12,
                node_role="static",
                is_static=True,
            ),
            Entity(
                entity_id="table_1",
                video_id="parity_video",
                category="table",
                first_seen_frame=0,
                last_seen_frame=11,
                first_seen_ts=0.0,
                last_seen_ts=11.0,
                observation_count=12,
                node_role="static",
                is_static=True,
            ),
        ],
        state_episodes=[
            StateEpisode(
                state_episode_id="chair_color",
                video_id="parity_video",
                entity_id="chair_1",
                attr_key="color",
                attr_value="red",
                start_frame=0,
                end_frame=11,
                start_ts=0.0,
                end_ts=11.0,
                support_frames=12,
            )
        ],
        relation_episodes=[
            RelationEpisode(
                relation_episode_id="chair_near_table",
                video_id="parity_video",
                subject_entity_id="chair_1",
                predicate="near",
                raw_predicate="near",
                object_entity_id="table_1",
                start_frame=0,
                end_frame=11,
                start_ts=0.0,
                end_ts=11.0,
                support_frames=12,
            )
        ],
        scene_segments=[
            SceneSegment(
                scene_id="room_1",
                video_id="parity_video",
                scene_index=1,
                scene_count=1,
                start_frame_rank=0,
                end_frame_rank=11,
                start_frame=0,
                end_frame=11,
                start_ts=0.0,
                end_ts=11.0,
                support_frames=12,
            )
        ],
    )
    return build_canonical_temporal_memory_graph(document)


class _FixedVectorIndex:
    model_name = "parity-vector"
    cache_status = "memory"

    def __init__(self, hits: list[SimpleNamespace]) -> None:
        self._hits = hits

    def search(self, *_args, **_kwargs):
        return list(self._hits)


def _assert_same_output(query: MemoryQuery, *, vector: bool = False) -> None:
    graph = _sample_graph()
    source_kwargs = {}
    fork_kwargs = {}
    if vector:
        probe = navqa_retrieval.CanonicalHybridRetrievalSession(graph)
        records = sorted(
            probe.projection.records.values(),
            key=lambda item: item.source_record_id,
        )
        hits = [
            SimpleNamespace(
                record_id=record.source_record_id,
                source_kind=record.source_kind,
                similarity=0.9 - rank * 0.01,
                rank=rank + 1,
            )
            for rank, record in enumerate(records[:8])
        ]
        source_kwargs["vector_recall_index"] = _FixedVectorIndex(hits)
        fork_kwargs["vector_recall_index"] = _FixedVectorIndex(hits)

    source = navqa_retrieval.CanonicalHybridRetrievalSession(graph, **source_kwargs)
    fork = openeqa_retrieval.CanonicalHybridRetrievalSession(graph, **fork_kwargs)

    source_output = source.retrieve(query, mode="temporal", limit=10)
    fork_output = fork.retrieve(query, mode="temporal", limit=10)

    assert fork_output.items == source_output.items
    assert fork_output.raw_payload == source_output.raw_payload


def test_openeqa_fork_matches_navqa_reference_for_unified_graph_retrieval() -> None:
    _assert_same_output(
        MemoryQuery(
            query_text="What color is the chair near the table?",
            answer_type="text",
            metadata={
                "graph_retrieval_profile": "openeqa_hm3d_unified_v2",
                "question_ir": {
                    "answer_type": "text",
                    "target_entities": ["chair"],
                    "attribute_dimension": "color",
                    "spatial_constraints": [{"relation": "near", "target": "table"}],
                },
            },
        )
    )


def test_openeqa_fork_matches_navqa_reference_for_vector_evidence_retrieval() -> None:
    _assert_same_output(
        MemoryQuery(
            query_text="What color is the chair near the table?",
            answer_type="text",
            metadata={
                "graph_retrieval_profile": "canonical_vector_evidence_v5",
                "question_ir": {
                    "answer_type": "text",
                    "target_entities": ["chair"],
                    "attribute_dimension": "color",
                },
            },
        ),
        vector=True,
    )


def test_openeqa_graph_only_answer_bundle_matches_navqa_reference() -> None:
    graph = _sample_graph()
    query = MemoryQuery(
        query_text="What color is the chair near the table?",
        answer_type="text",
        metadata={"graph_retrieval_profile": "openeqa_hm3d_unified_v2"},
        limit=10,
    )
    navqa_session = navqa_retrieval.CanonicalHybridRetrievalSession(graph)
    openeqa_session = openeqa_retrieval.CanonicalHybridRetrievalSession(graph)

    navqa_answer = answer_navqa_canonical_graph_query(
        graph,
        query,
        question="What color is the chair near the table?",
        retrieval_mode="temporal",
        answer_mode="graph_only",
        retrieval_session=navqa_session,
    )
    openeqa_answer = answer_openeqa_canonical_graph_query(
        graph,
        query,
        question="What color is the chair near the table?",
        retrieval_mode="temporal",
        answer_mode="graph_only",
        retrieval_session=openeqa_session,
    )

    assert openeqa_answer == navqa_answer


def test_openeqa_graph_only_prompt_matches_frozen_reference_without_visual_contract() -> None:
    bundle = {
        "answer_brief": "The chair is red.",
        "evidence_lines": ["kind=state owner=chair attribute=color value=red"],
        "focus_frame_lines": ["frame=4 objects=[chair, table]"],
        "scene_lines": ["scene=room_1 frames=0-11"],
        "temporal_chain_lines": [],
        "uncertainty_notes": [],
        "question_ir": {
            "answer_type": "text",
            "target_entities": ["chair"],
            "attribute_dimension": "color",
        },
        "structured_graph_context": {"video_id": "parity_video"},
        "llm_context_text": "Graph context",
    }
    graph_result = {
        "answer_brief": "The chair is red.",
        "llm_ready_bundle": bundle,
        "matches": [],
        "retrieval_bundle": {},
    }
    config = GraphQAConfig(enabled=True)
    frozen = FrozenGraphQAResponder(config)
    openeqa = OpenEQAGraphQAResponder(config)

    for force_json_answer in (False, True):
        frozen_payload = frozen._build_payload(
            question="What color is the chair?",
            llm_ready_bundle=bundle,
            graph_result=graph_result,
            force_json_answer=force_json_answer,
        )
        openeqa_payload = openeqa._build_payload(
            question="What color is the chair?",
            llm_ready_bundle=bundle,
            graph_result=graph_result,
            force_json_answer=force_json_answer,
        )

        assert openeqa_payload == frozen_payload


def test_openeqa_scripts_do_not_import_frozen_navqa_retrieval() -> None:
    forbidden = (
        "vlkg.integrations.cognee.canonical_hybrid_retrieval",
        "vlkg.canonical_graph_qa_context",
    )
    offenders = []
    for path in (REPO_ROOT / "scripts" / "openeqa").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(item in source for item in forbidden):
            offenders.append(path.name)

    assert offenders == []


def test_openeqa_fork_uses_owned_vector_recall() -> None:
    source = (
        REPO_ROOT
        / "src"
        / "vlkg"
        / "datasets"
        / "openeqa"
        / "retrieval"
        / "canonical_hybrid_retrieval.py"
    ).read_text(encoding="utf-8")

    assert "vlkg.datasets.openeqa.runtime.canonical_record_vector_recall" in source
    assert "vlkg.integrations.cognee.canonical_record_vector_recall" not in source


def test_unified_retrieval_ir_is_invariant_to_dataset_and_route_metadata() -> None:
    semantic = {
        "answer_type": "text",
        "operation": "attribute_query",
        "target_entities": ["chair"],
        "attribute_dimension": "color",
        "spatial_constraints": [{"relation": "near", "target": "table"}],
    }
    hm3d_ir = {
        **semantic,
        "dataset": "OpenEQA",
        "data_source": "hm3d-v0",
        "category": "attribute recognition",
        "evidence_question_type": "non_temporal.entity.attribute",
        "evidence_requirements": ["owner-bound attribute"],
        "graph_information_needs": ["attribute evidence"],
        "question_route": {"primary_need": "entity_attribute"},
    }
    scannet_ir = {
        **semantic,
        "dataset": "OpenEQA",
        "data_source": "scannet-v0",
        "category": "deliberately different",
        "question_route": {"primary_need": "spatial_localization"},
    }

    hm3d_projection = openeqa_retrieval.unified_graph_retrieval_question_ir(
        hm3d_ir,
        graph_retrieval_profile="canonical_vector_evidence_v5",
    )
    scannet_projection = openeqa_retrieval.unified_graph_retrieval_question_ir(
        scannet_ir,
        graph_retrieval_profile="canonical_vector_evidence_v5",
    )

    assert hm3d_projection == semantic
    assert scannet_projection == semantic


def test_openeqa_vector_text_strips_all_source_video_ids() -> None:
    for source in ("hm3d-v0", "scannet-v0"):
        record = GraphRecordView(
            source_kind="entity",
            source_record_id=f"entity:{source}__episode:chair-1",
            summary=f"internal {source}__episode chair",
            metadata={
                "entity_id": f"{source}__episode:chair-1",
                "label": "chair",
                "canonical_label": "chair",
            },
            index_text="chair",
            tokens=frozenset({"chair"}),
            video_id=f"{source}__episode",
            scene_ids=(),
            entity_ids=("chair-1",),
            start_ts=None,
            end_ts=None,
            priority=5,
        )

        text = openeqa_record_semantic_text(record)

        assert source not in text
        assert "episode" not in text
