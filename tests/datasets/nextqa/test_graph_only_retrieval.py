from __future__ import annotations

import ast
import json
from pathlib import Path

from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalGraphEdge,
    CanonicalSceneNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.datasets.nextqa.retrieval.graph_only_retrieval import (
    UnifiedGraphOnlyRetriever,
    load_unified_retrieval_config,
)
from vlkg.datasets.nextqa.retrieval.question_ir import (
    build_question_ir_record,
    retrieval_input_from_ir,
)


def test_full_retrieval_runner_checkpoint_writes_records_and_status(
    nextqa_tmp_dir: Path,
) -> None:
    import importlib.util

    script = (
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'retrieval' / 'run_graph_retrieval_pilot.py'
    )
    spec = importlib.util.spec_from_file_location(
        "nextqa_graph_retrieval_runner_checkpoint_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._checkpoint(
        nextqa_tmp_dir,
        [{"question_id": "q2", "status": "completed"}],
        [{"question_id": "q1", "status": "error"}],
    )

    records = json.loads(
        (nextqa_tmp_dir / "retrieval_records.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (nextqa_tmp_dir / "run_status.json").read_text(encoding="utf-8")
    )
    assert [row["question_id"] for row in records] == ["q1", "q2"]
    assert status["completed_question_count"] == 1
    assert status["error_question_count"] == 1
    assert module._status_name(structurally_valid=True, error_count=0) == "completed"
    assert (
        module._status_name(structurally_valid=True, error_count=2)
        == "completed_with_errors"
    )
    assert module._status_name(structurally_valid=False, error_count=0) == "invalid"
from vlkg.core.graph_retrieval.canonical_record_vector_recall import VectorRecallHit


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
CONFIG = REPO_ROOT / 'tests/fixtures/release_configs/nextqa/nextqa_graph_only_unified_v1.yaml'
CONFIG_V2 = REPO_ROOT / 'configs/datasets/nextqa/retrieval.yaml'
CONFIG_V3 = REPO_ROOT / 'tests/fixtures/release_configs/nextqa/nextqa_graph_only_unified_v3_evidence24.yaml'


class _FakeVectorIndex:
    model_name = "fake-text-encoder"
    cache_status = "memory"

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        self.queries: list[str] = []

    def search(self, query_text: str, *, top_k: int, min_similarity: float):
        self.queries.append(query_text)
        return [
            VectorRecallHit(
                record_id=self.record_id,
                source_kind="episode",
                similarity=0.72,
                rank=1,
            )
        ]


def _graph() -> CanonicalTemporalMemoryGraph:
    time = CanonicalTimeRange(
        start_frame=0,
        end_frame=30,
        start_ts=0.0,
        end_ts=1.0,
        support_frames=2,
    )
    scene_id = "nextqa-val/123:scene:000001"
    man_id = "nextqa-val/123:entity:000001"
    bike_id = "nextqa-val/123:entity:000002"
    episode_node_id = "episode:ride"
    return CanonicalTemporalMemoryGraph(
        video_id="nextqa-val/123",
        scene_nodes=[
            CanonicalSceneNode(
                node_id=f"scene:{scene_id}",
                video_id="nextqa-val/123",
                scene_id=scene_id,
                scene_index=1,
                scene_count=1,
                label="outdoor bicycle area",
                time_range=time,
            )
        ],
        entity_nodes=[
            CanonicalEntityNode(
                node_id=f"entity:{man_id}",
                video_id="nextqa-val/123",
                entity_id=man_id,
                label="man",
                node_role="dynamic",
                is_static=False,
                is_noise=False,
                is_fragment=False,
                observation_count=2,
                time_range=time,
                canonical_label="man",
                semantic_alias_terms=["man", "person"],
                attributes={"raw_visual_attributes": ["man riding a bicycle"]},
            ),
            CanonicalEntityNode(
                node_id=f"entity:{bike_id}",
                video_id="nextqa-val/123",
                entity_id=bike_id,
                label="bicycle",
                node_role="dynamic",
                is_static=False,
                is_noise=False,
                is_fragment=False,
                observation_count=2,
                time_range=time,
                canonical_label="bicycle",
                semantic_alias_terms=["bicycle", "bike"],
            ),
        ],
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id=episode_node_id,
                video_id="nextqa-val/123",
                episode_id="ride",
                episode_kind="RELATION",
                label="man riding bicycle",
                time_range=time,
                scene_id=scene_id,
                primary_entity_id=man_id,
                subject_entity_id=man_id,
                object_entity_id=bike_id,
                participant_entity_ids=[man_id, bike_id],
                predicate="RIDING",
                raw_predicate="riding",
                source_ref_id="ride",
            )
        ],
        edges=[
            CanonicalGraphEdge(
                edge_id="edge:man:ride",
                edge_type="HAS_EPISODE",
                video_id="nextqa-val/123",
                source_node_id=f"entity:{man_id}",
                target_node_id=episode_node_id,
                time_range=time,
            ),
            CanonicalGraphEdge(
                edge_id="edge:ride:bike",
                edge_type="INTERACTS_WITH",
                video_id="nextqa-val/123",
                source_node_id=episode_node_id,
                target_node_id=f"entity:{bike_id}",
                time_range=time,
            ),
        ],
    )


def _question():
    complete = build_question_ir_record(
        {
            "question_id": "nextqa-val/123/q7",
            "video_id": "123",
            "official_type": "CH",
            "normalized_question": "How does the boy ride the bike?",
            "answer_choices": ["a", "b", "c", "d", "e"],
            "answer_format": "multiple_choice_5",
        }
    )
    return retrieval_input_from_ir(complete)


def test_unified_retrieval_unions_text_vector_entity_and_graph_neighbors() -> None:
    graph = _graph()
    fake = _FakeVectorIndex("episode:ride")
    retriever = UnifiedGraphOnlyRetriever(
        graph,
        config=load_unified_retrieval_config(CONFIG),
        vector_cache_path=None,
        vector_index=fake,
    )

    result = retriever.retrieve(_question())

    assert result["status"] == "completed"
    assert result["candidate_counts"]["lexical"] > 0
    assert result["candidate_counts"]["vector"] == 1
    assert result["candidate_counts"]["exact_entity"] > 0
    assert result["candidate_counts"]["expanded"] >= 3
    assert "episode:ride" in result["graph_record_ids"]
    assert fake.queries == ["how does the boy ride the bike"]
    assert result["purity"] == {
        "uses_gold_answer": False,
        "uses_official_type": False,
        "uses_answer_choices": False,
        "reads_rgb": False,
        "reads_object_crop": False,
        "reads_visual_embedding_cache": False,
        "uses_visual_index": False,
        "shared_hybrid_retrieve_called": False,
    }


def test_config_has_one_fixed_budget_and_no_type_or_route_keys() -> None:
    config = load_unified_retrieval_config(CONFIG)
    config_v2 = load_unified_retrieval_config(CONFIG_V2)
    assert config.profile_name == "nextqa_graph_only_unified_v1"
    assert config.evidence_limit == 16
    assert config.max_hops == 2
    assert config_v2.profile_name == "nextqa_graph_only_unified_v2"
    assert config_v2.evidence_limit == config.evidence_limit
    assert config_v2.max_hops == config.max_hops
    assert config_v2.demand_weight > 0.0

    config_v3 = load_unified_retrieval_config(CONFIG_V3)
    assert config_v3.profile_name == "nextqa_graph_only_unified_v3_evidence24"
    assert config_v3.evidence_limit == 24
    assert config_v3.lexical_top_k == config_v2.lexical_top_k
    assert config_v3.vector_top_k == config_v2.vector_top_k
    assert config_v3.expanded_candidate_limit == config_v2.expanded_candidate_limit
    assert config_v3.max_hops == config_v2.max_hops
    assert config_v3.demand_weight == config_v2.demand_weight


def test_graph_only_modules_do_not_import_dataset_adapters_visual_or_api_clients() -> None:
    paths = [
        REPO_ROOT / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'retrieval' / 'question_ir.py',
        REPO_ROOT / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'retrieval' / 'graph_only_retrieval.py',
        REPO_ROOT / "src" / "vlkg" / "datasets" / "nextqa" / "retrieval" / "question_ir.py",
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'retrieval' / 'run_graph_retrieval_pilot.py',
    ]
    forbidden_roots = {"faiss", "httpx", "openai", "PIL", "requests"}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        imports = {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("eval_adapters.navqa" in name for name in imports)
        assert not any("eval_adapters.openeqa" in name for name in imports)
        assert not any(name.split(".")[0] in forbidden_roots for name in imports)


def test_retrieval_runner_has_no_evaluator_label_input() -> None:
    source = (
        REPO_ROOT / 'scripts' / 'datasets' / 'nextqa' / 'retrieval' / 'run_graph_retrieval_pilot.py'
    ).read_text(encoding="utf-8").lower()
    assert "labels.evaluator_only.json" not in source
    assert 'row["official_type"]' not in source
    assert 'row.get("official_type")' not in source
    assert 'row["answer_choices"]' not in source
    assert 'row.get("answer_choices")' not in source


def test_graph_manifest_relative_paths_are_anchored_to_repository(tmp_path, monkeypatch):
    from scripts.datasets.nextqa.retrieval import run_graph_retrieval_pilot as runner
    workspace = tmp_path / 'checkout'
    graphs = workspace / 'datasets/nextqa/graphs/test/payload'
    graphs.mkdir(parents=True)
    canonical = graphs / 'canonical.json'
    canonical.write_text('{}', encoding='utf-8')
    (graphs / 'nextqa_graph_build_manifest.json').write_text(json.dumps({
        'video_count': 1,
        'completed_video_count': 1,
        'error_video_count': 0,
        'uses_gold_answer': False,
        'videos': [{
            'official_video_id': '100',
            'status': 'completed',
            'canonical_graph_path': canonical.relative_to(workspace).as_posix(),
        }],
    }), encoding='utf-8')
    unrelated = tmp_path / 'other-working-directory'
    unrelated.mkdir()
    monkeypatch.setattr(runner, 'REPO_ROOT', workspace)
    monkeypatch.chdir(unrelated)
    assert runner._load_graph_paths(graphs, expected_video_count=1) == {'100': canonical.resolve()}
