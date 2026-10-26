from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys


SCRIPT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / "scripts" / "datasets" / "openeqa"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

finalizer = import_module("scripts.datasets.openeqa.audit.finalize_openeqa_unified_qa_run")


def _config(source: str) -> dict:
    config = {
        "contract": "openeqa_evidence_demand_visual_qa_run_config_v1",
        "data_source": source,
        "graph_manifest": f"{source}-graphs.json",
        "visual_index_dir": f"{source}-visual-index",
    }
    for field in finalizer.COMMON_CONFIG_FIELDS:
        config[field] = {
            "open_eqa_json": "open-eqa-v0.json",
            "question_ir_json": "unified-routes.json",
            "route_subset_manifest": "",
            "route_subset_primary_need": "",
            "route_subset_strategy_variants": [],
            "answer_mode": "live",
            "qa_config": 'configs/datasets/openeqa/qa.yaml',
            "qa_max_tokens": 256,
            "retrieval_mode": "auto",
            "retrieval_limit": 10,
            "graph_retrieval_profile": "canonical_vector_evidence_v5",
            "graph_retrieval_contract": "canonical_vector_evidence.v5",
            "vector_model": "sentence-transformers/all-MiniLM-L6-v2",
            "visual_index_name": "openeqa_object_siglip_flat_ip",
            "visual_backbone": "openeqa_question_router_v4",
            "visual_runtime_contract": "openeqa_visual_runtime.v4",
            "visual_top_k": 8,
            "visual_min_score": 0.0,
            "visual_max_candidate_packs": 8,
            "visual_max_evidence_lines": 8,
            "uses_gold_answer": False,
        }[field]
    return config


def _route_record(
    question_id: str,
    source: str,
    *,
    variant: str = "target_location",
) -> dict:
    return {
        "question_id": question_id,
        "data_source": source,
        "llm_question_ir": {
            "question_route": {
                "primary_need": "spatial_localization",
                "strategy_variant": variant,
                "uses_gold_answer": False,
            }
        },
    }


def _shard(
    question_id: str,
    source: str,
    *,
    variant: str = "target_location",
) -> dict:
    return {
        "run_dir": f"{source}-run",
        "run_config": _config(source),
        "summary": {"error_count": 0},
        "audit": [
            {
                "question_id": question_id,
                "episode_history": f"{source}/episode-1",
                "prediction": "beside the table",
                "evidence_demand": {"primary_need": "spatial_localization"},
                "strategy_variant": variant,
                "runtime_route_recompiled": False,
                "question_route_decision_source": "question_ir.question_route",
                "graph_retrieval_profile": "canonical_vector_evidence_v5",
                "visual_backbone": "openeqa_question_router_v4",
                "route_strategy_handler": "spatial_localization.variant_v1",
            }
        ],
        "predictions": [
            {"question_id": question_id, "answer": "beside the table"}
        ],
    }


def test_unified_run_accepts_resource_only_source_differences() -> None:
    rows = [
        {
            "question_id": "h1",
            "episode_history": "hm3d-v0/episode-1",
        },
        {
            "question_id": "s1",
            "episode_history": "scannet-v0/episode-1",
        },
    ]
    routes = {
        "contract": "openeqa_unified_question_route_ir.v1",
        "dataset": "OpenEQA",
        "uses_gold_answer": False,
        "runtime_route_recompile_allowed": False,
        "records": [_route_record("h1", "hm3d-v0"), _route_record("s1", "scannet-v0")],
    }

    predictions, audit, summary, issues = finalizer.validate_and_merge(
        dataset_rows=rows,
        question_route_payload=routes,
        route_subset_payload=None,
        shards=[_shard("h1", "hm3d-v0"), _shard("s1", "scannet-v0")],
    )

    assert issues == []
    assert summary["status"] == "complete"
    assert summary["question_count"] == 2
    assert summary["data_sources_are_execution_shards_only"] is True
    assert [item["question_id"] for item in predictions] == ["h1", "s1"]
    assert [item["question_id"] for item in audit] == ["h1", "s1"]


def test_unified_run_rejects_source_specific_strategy_configuration() -> None:
    rows = [
        {"question_id": "h1", "episode_history": "hm3d-v0/episode-1"},
        {"question_id": "s1", "episode_history": "scannet-v0/episode-1"},
    ]
    routes = {
        "contract": "openeqa_unified_question_route_ir.v1",
        "dataset": "OpenEQA",
        "uses_gold_answer": False,
        "runtime_route_recompile_allowed": False,
        "records": [_route_record("h1", "hm3d-v0"), _route_record("s1", "scannet-v0")],
    }
    scannet = deepcopy(_shard("s1", "scannet-v0"))
    scannet["run_config"]["visual_top_k"] = 99

    _, _, summary, issues = finalizer.validate_and_merge(
        dataset_rows=rows,
        question_route_payload=routes,
        route_subset_payload=None,
        shards=[_shard("h1", "hm3d-v0"), scannet],
    )

    assert summary["status"] == "invalid"
    assert any("one complete QA/retrieval/visual configuration" in issue for issue in issues)


def test_unified_run_rejects_runtime_route_drift() -> None:
    rows = [
        {"question_id": "h1", "episode_history": "hm3d-v0/episode-1"},
        {"question_id": "s1", "episode_history": "scannet-v0/episode-1"},
    ]
    routes = {
        "contract": "openeqa_unified_question_route_ir.v1",
        "dataset": "OpenEQA",
        "uses_gold_answer": False,
        "runtime_route_recompile_allowed": False,
        "records": [_route_record("h1", "hm3d-v0"), _route_record("s1", "scannet-v0")],
    }
    hm3d = deepcopy(_shard("h1", "hm3d-v0"))
    hm3d["audit"][0]["strategy_variant"] = "different_variant"
    hm3d["audit"][0]["runtime_route_recompiled"] = True

    _, _, summary, issues = finalizer.validate_and_merge(
        dataset_rows=rows,
        question_route_payload=routes,
        route_subset_payload=None,
        shards=[hm3d, _shard("s1", "scannet-v0")],
    )

    assert summary["status"] == "invalid"
    assert any("strategy variant drift" in issue for issue in issues)
    assert any("runtime route recompilation" in issue for issue in issues)


def test_unified_run_filters_route_subset_by_exact_strategy_variant() -> None:
    rows = [
        {"question_id": "h1", "episode_history": "hm3d-v0/episode-1"},
        {"question_id": "h2", "episode_history": "hm3d-v0/episode-1"},
    ]
    routes = {
        "contract": "openeqa_unified_question_route_ir.v1",
        "dataset": "OpenEQA",
        "uses_gold_answer": False,
        "runtime_route_recompile_allowed": False,
        "records": [
            _route_record("h1", "hm3d-v0", variant="target_location"),
            _route_record("h2", "hm3d-v0", variant="open_target_relation"),
        ],
    }
    route_subset = {
        "uses_gold_answer": False,
        "records": [
            {
                "question_id": "h1",
                "data_source": "hm3d-v0",
                "strategy_variant": "target_location",
            },
            {
                "question_id": "h2",
                "data_source": "hm3d-v0",
                "strategy_variant": "open_target_relation",
            },
        ],
    }
    shard = _shard(
        "h2",
        "hm3d-v0",
        variant="open_target_relation",
    )
    shard["run_config"]["route_subset_strategy_variants"] = [
        "open_target_relation"
    ]

    predictions, audit, summary, issues = finalizer.validate_and_merge(
        dataset_rows=rows,
        question_route_payload=routes,
        route_subset_payload=route_subset,
        shards=[shard],
        strategy_variants=["open_target_relation"],
    )

    assert issues == []
    assert summary["status"] == "complete"
    assert summary["question_count"] == 1
    assert summary["route_subset_strategy_variants"] == [
        "open_target_relation"
    ]
    assert [item["question_id"] for item in predictions] == ["h2"]
    assert [item["question_id"] for item in audit] == ["h2"]


def test_unified_run_merges_v5_by_flat_route_and_handler() -> None:
    rows = [
        {"question_id": "h1", "episode_history": "hm3d-v0/episode-1"},
        {"question_id": "s1", "episode_history": "scannet-v0/episode-1"},
    ]
    handler = "existence_or_state.surface_capacity_observation_v1"
    routes = {
        "contract": "openeqa_unified_question_route_ir.v1",
        "route_contract": "openeqa_single_route_decision.v1",
        "dataset": "OpenEQA",
        "routing_profile": "openeqa_ir_first_single_route_v5",
        "uses_gold_answer": False,
        "runtime_route_recompile_allowed": False,
        "records": [
            {
                "question_id": "h1",
                "llm_question_ir": {
                    "question_type": "surface_capacity",
                    "question_route": {
                        "route_id": "surface_capacity",
                        "handler_id": handler,
                    },
                },
            },
            {
                "question_id": "s1",
                "llm_question_ir": {
                    "question_type": "surface_capacity",
                    "question_route": {
                        "route_id": "surface_capacity",
                        "handler_id": handler,
                    },
                },
            },
        ],
    }

    def v5_shard(question_id: str, source: str) -> dict:
        shard = _shard(question_id, source)
        config = shard["run_config"]
        config["visual_backbone"] = "openeqa_question_router_v5"
        config["visual_runtime_contract"] = "openeqa_visual_runtime_v5"
        shard["audit"][0].update(
            {
                "evidence_demand": {"route_id": "surface_capacity"},
                "route_handler_id": handler,
                "route_strategy_handler": handler,
                "strategy_variant": None,
                "visual_backbone": "openeqa_question_router_v5",
            }
        )
        return shard

    predictions, audit, summary, issues = finalizer.validate_and_merge(
        dataset_rows=rows,
        question_route_payload=routes,
        route_subset_payload=None,
        shards=[v5_shard("h1", "hm3d-v0"), v5_shard("s1", "scannet-v0")],
    )

    assert issues == []
    assert summary["status"] == "complete"
    assert [item["question_id"] for item in predictions] == ["h1", "s1"]
    assert [item["question_id"] for item in audit] == ["h1", "s1"]
