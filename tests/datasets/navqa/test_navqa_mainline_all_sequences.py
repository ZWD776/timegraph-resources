from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / "scripts"
    / "eval"
    / "run_navqa_mainline_all_sequences.py"
)


def _load_module():
    return importlib.import_module("scripts.datasets.navqa.qa.run_navqa_mainline_all_sequences")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discover_unified_layout_sequence(tmp_path: Path) -> None:
    module = _load_module()
    layout_root = tmp_path / "navqa"
    layout = module.default_navqa_sequence_layout(
        seq_id="3",
        cam="0",
        stride=10,
        layout_root=layout_root,
    )
    _write_json(layout.canonical_graph_json, {"nodes": [], "edges": []})
    _write_json(layout.frame_time_index_json, {"ordered_frames": []})
    _write_json(layout.qa_unfilled_json, {"data": []})
    _write_json(layout.sample_root / "llm_question_ir_non_position.json", {"records": []})

    specs = module.discover_sequences(
        layout_root=layout_root,
        repo_root=tmp_path,
        include_legacy_seq0=False,
    )

    assert len(specs) == 1
    assert specs[0].seq_id == "3"
    assert specs[0].layout_kind == "unified"
    assert specs[0].canonical_graph_json == layout.canonical_graph_json
    assert specs[0].question_ir_json == layout.sample_root / "llm_question_ir_non_position.json"


def test_manifest_overrides_discovered_sequence(tmp_path: Path) -> None:
    module = _load_module()
    layout_root = tmp_path / "navqa"
    layout = module.default_navqa_sequence_layout(
        seq_id="3",
        cam="0",
        stride=10,
        layout_root=layout_root,
    )
    manifest_graph = tmp_path / "manual" / "graph.json"
    _write_json(layout.canonical_graph_json, {"name": "discovered"})
    _write_json(manifest_graph, {"name": "manifest"})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "sequences": [
                {
                    "seq_id": "3",
                    "cam": "0",
                    "stride": 10,
                    "canonical_graph_json": str(manifest_graph),
                }
            ]
        },
    )

    specs = module.discover_sequences(
        layout_root=layout_root,
        repo_root=tmp_path,
        include_legacy_seq0=False,
        manifest_path=manifest,
    )

    assert len(specs) == 1
    assert specs[0].layout_kind == "manifest"
    assert specs[0].canonical_graph_json == manifest_graph


def test_manifest_prefixed_layout_overrides_discovered_sequence(tmp_path: Path) -> None:
    module = _load_module()
    layout_root = tmp_path / "navqa"
    layout = module.default_navqa_sequence_layout(
        seq_id="3",
        cam="0",
        stride=10,
        layout_root=layout_root,
    )
    manifest_graph = tmp_path / "manual" / "graph.json"
    _write_json(layout.canonical_graph_json, {"name": "discovered"})
    _write_json(manifest_graph, {"name": "manifest"})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "sequences": [
                {
                    "seq_id": "3",
                    "cam": "0",
                    "stride": 10,
                    "layout_kind": "manifest_unified",
                    "canonical_graph_json": str(manifest_graph),
                }
            ]
        },
    )

    specs = module.discover_sequences(
        layout_root=layout_root,
        repo_root=tmp_path,
        include_legacy_seq0=False,
        manifest_path=manifest,
    )

    assert len(specs) == 1
    assert specs[0].layout_kind == "manifest_unified"
    assert specs[0].canonical_graph_json == manifest_graph


def test_discovery_does_not_inject_legacy_seq0(tmp_path: Path) -> None:
    module = _load_module()
    legacy_root = tmp_path / ".scratch" / "hsgg_navqa_seq0_every10_full_local_paths"
    _write_json(
        legacy_root
        / "canonical_temporal_memory_graph_rebuild_20260507_compact_scene_label.json",
        {"name": "legacy-seq0"},
    )
    _write_json(
        tmp_path / ".scratch" / "navqa_seq0_frame_time_index.json",
        {"ordered_frames": []},
    )

    specs = module.discover_sequences(
        layout_root=tmp_path / ".scratch" / "navqa",
        repo_root=tmp_path,
        include_legacy_seq0=True,
    )

    assert specs == []


def test_default_mainline_manifest_uses_one_layout_kind_for_all_sequences() -> None:
    manifest_path = (
        next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
        / "configs"
        / "datasets/navqa/sequences.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    layout_kinds = {
        str(item.get("layout_kind"))
        for item in payload["sequences"]
    }

    assert layout_kinds == {"manifest_cg6_local_tracklet"}


def test_default_mainline_manifest_exposes_cg_ir_qa_naming() -> None:
    manifest_path = (
        next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
        / "configs"
        / "datasets/navqa/sequences.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["naming_contract"] == "navqa_cg_ir_qa_naming_v1"
    assert payload["cg_id"] == "CG6-clean"
    assert payload["cg_name"] == "construct_graph_6_local_tracklet"
    assert payload["graph_profile"] == "time_fact_core"
    assert payload["ir_qa_default_id"] == "IR-QA-G-clean"
    assert payload["default_route_id"] == "CG6-clean+IR-QA-G-clean"


def test_ir_qa_id_from_visual_faiss_flags() -> None:
    module = _load_module()

    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=False,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-G"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=True,
            visual_faiss_query_recall=False,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-v2"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=True,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=False,
            visual_faiss_ger_binding=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-v3"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=True,
            visual_faiss_ger_fusion=False,
            visual_faiss_ger_binding=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-v4"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_evidence_layer=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-evidence-layer-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_frame_context=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v3-binding"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            synonym_lexicon=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-synonym-l0-l1-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_role_aware_compression=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v3-role-aware-compression"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            question_router=True,
            question_router_hybrid_backbone=True,
            visual_faiss_entity_pack=False,
            visual_faiss_score_contract=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-question-router-v2-hybrid-backbone"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            question_router=True,
            question_router_hybrid_backbone=True,
            question_router_specialized_backbone=True,
            visual_faiss_entity_pack=False,
            visual_faiss_score_contract=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-question-router-v3-specialized-backbone"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            question_router=True,
            question_router_mainline=True,
            visual_faiss_entity_pack=False,
            visual_faiss_score_contract=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-question-router-v4-mainline"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            question_router=True,
            visual_faiss_entity_pack=True,
            visual_faiss_score_contract=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-question-router-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_entity_pack=True,
            visual_faiss_score_contract=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-score-contract-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_entity_pack=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-entity-pack-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_entity_pack=True,
            visual_faiss_entity_pack_semantic_align=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-entity-pack-v2-semantic-align"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_entity_pack=True,
            visual_faiss_entity_pack_semantic_align=True,
            visual_faiss_entity_pack_semantic_gated_role=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-entity-pack-v3-semantic-gated-role"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_frame_context=True,
            visual_faiss_structural_binding=False,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v5-frame-context"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_frame_context=True,
            visual_faiss_structural_binding=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v6-structural-binding"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_fragment_bridge=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v7-fragment-bridge"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_object_local_facts=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v8-object-local-fact"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_object_local_facts=True,
            visual_faiss_balanced_graph_visual=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v9-balanced-graph-visual"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_object_local_facts=True,
            answerable_candidate_packs=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v10-answerable-candidate-pack"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_role_aware_evidence=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-role-aware-evidence-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=True,
            visual_faiss_attribute_owner=True,
            owner_minimal_prompt=True,
            visual_faiss_role_aware_aux=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v13-role-aware-aux"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=True,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=True,
            visual_faiss_ger_binding=False,
            visual_faiss_unified_candidates=True,
            frame_faiss_query_recall=False,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-GER-v4-unified-candidate"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=False,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=False,
            visual_faiss_ger_binding=False,
            frame_faiss_query_recall=True,
            frame_faiss_fused_evidence=True,
            frame_faiss_graph_anchors=False,
        )
        == "IR-QA-F-frame-v1"
    )
    assert (
        module.ir_qa_id_from_flags(
            visual_faiss_focus_seed=False,
            visual_faiss_query_recall=False,
            visual_faiss_fused_evidence=False,
            visual_faiss_graph_anchors=False,
            visual_faiss_ger_fusion=False,
            visual_faiss_ger_binding=False,
            frame_faiss_query_recall=True,
            frame_faiss_fused_evidence=False,
            frame_faiss_graph_anchors=True,
        )
        == "IR-QA-F-frame-v2"
    )


def test_resolve_run_ir_qa_id_uses_manifest_graph_only_variant() -> None:
    module = _load_module()

    assert (
        module.resolve_run_ir_qa_id(
            flag_ir_qa_id="IR-QA-G",
            pipeline_metadata={"ir_qa_default_id": "IR-QA-G-compressed-fact-v1"},
        )
        == "IR-QA-G-compressed-fact-v1"
    )
    assert (
        module.resolve_run_ir_qa_id(
            flag_ir_qa_id="IR-QA-F-GER-v7-fragment-bridge",
            pipeline_metadata={"ir_qa_default_id": "IR-QA-G-compressed-fact-v1"},
        )
        == "IR-QA-F-GER-v7-fragment-bridge"
    )


def test_question_ir_match_count_requires_llm_ir_records(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "question_ir.json"
    _write_json(
        path,
        {
            "records": [
                {"uuid": "q1", "llm_question_ir": {"operation": "existence_query"}},
                {"uuid": "q2"},
            ]
        },
    )

    count = module.question_ir_match_count(
        path,
        [
            {module.FIELD_UUID: "q1"},
            {module.FIELD_UUID: "q2"},
            {module.FIELD_UUID: "q3"},
        ],
    )

    assert count == 1


def test_paths_for_sequence_do_not_overwrite_existing_ir_when_forced(tmp_path: Path) -> None:
    module = _load_module()
    existing_ir = tmp_path / "old_ir.json"
    graph = tmp_path / "graph.json"
    spec = module.SequenceSpec(
        seq_id="3",
        cam="0",
        stride=10,
        layout_kind="unified",
        sample_root=tmp_path / "sample",
        canonical_graph_json=graph,
        frame_time_index_json=None,
        qa_unfilled_json=None,
        question_ir_json=existing_ir,
        projection_cache=None,
        existing_baseline_json=None,
    )

    reused = module.paths_for_sequence(
        spec,
        output_root=tmp_path / "runs",
        run_name_value="run",
        reuse_existing_question_ir=True,
    )
    forced = module.paths_for_sequence(
        spec,
        output_root=tmp_path / "runs",
        run_name_value="run",
        reuse_existing_question_ir=False,
    )

    assert reused.question_ir_json == existing_ir
    assert forced.question_ir_json != existing_ir
    assert forced.question_ir_json.parent == tmp_path / "runs" / "run" / spec.display_id
    assert forced.projection_cache == (
        tmp_path
        / "runs"
        / "run"
        / spec.display_id
        / "canonical_projection_cache.json.gz"
    )


def test_paths_for_sequence_uses_run_local_projection_cache_by_default(tmp_path: Path) -> None:
    module = _load_module()
    sequence_cache = tmp_path / "sample" / "canonical_projection_cache.json.gz"
    spec = module.SequenceSpec(
        seq_id="3",
        cam="0",
        stride=10,
        layout_kind="unified",
        sample_root=tmp_path / "sample",
        canonical_graph_json=tmp_path / "sample" / "graph.json",
        frame_time_index_json=None,
        qa_unfilled_json=None,
        question_ir_json=None,
        projection_cache=sequence_cache,
        existing_baseline_json=None,
    )

    default_paths = module.paths_for_sequence(
        spec,
        output_root=tmp_path / "runs",
        run_name_value="run",
    )
    reused_paths = module.paths_for_sequence(
        spec,
        output_root=tmp_path / "runs",
        run_name_value="run",
        reuse_sequence_projection_cache=True,
    )

    assert default_paths.projection_cache != sequence_cache
    assert default_paths.projection_cache.parent == tmp_path / "runs" / "run" / spec.display_id
    assert reused_paths.projection_cache == sequence_cache


def test_batch_output_complete_rejects_partial_json(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "batch.json"
    _write_json(path, {"results": [{"uuid": "q1"}]})

    assert module.batch_output_complete(path, expected_count=1) is True
    assert module.batch_output_complete(path, expected_count=2) is False


def test_mainline_parser_enables_llm_evidence_rerank_by_default() -> None:
    module = _load_module()

    args = module.build_parser().parse_args([])
    ablation_args = module.build_parser().parse_args(["--no-llm-evidence-rerank"])
    candidate_args = module.build_parser().parse_args(["--answer-candidate-extraction"])
    atomic_args = module.build_parser().parse_args(["--experimental-atomic-answer-facts"])
    parallel_args = module.build_parser().parse_args(["--qa-workers", "3"])
    diagnostic_args = module.build_parser().parse_args(
        ["--qa-workers", "1", "--qa-scheduler", "sequence"]
    )
    reuse_ir_args = module.build_parser().parse_args(["--reuse-manifest-question-ir"])

    assert args.llm_evidence_rerank is True
    assert ablation_args.llm_evidence_rerank is False
    assert args.answer_candidate_extraction is False
    assert candidate_args.answer_candidate_extraction is True
    assert args.experimental_atomic_answer_facts is False
    assert atomic_args.experimental_atomic_answer_facts is True
    assert args.experimental_answerable_candidate_packs is False
    assert args.experimental_visual_faiss_entity_pack is False
    assert args.experimental_visual_faiss_entity_pack_semantic_align is False
    assert args.experimental_visual_faiss_entity_pack_semantic_gated_role is False
    assert args.experimental_visual_faiss_score_contract is False
    assert args.experimental_question_router is False
    assert args.experimental_question_router_hybrid_backbone is False
    assert args.answerable_candidate_pack_max_items == 12
    assert args.qa_workers == 3
    assert parallel_args.qa_workers == 3
    assert args.qa_scheduler == "global"
    assert diagnostic_args.qa_workers == 1
    assert diagnostic_args.qa_scheduler == "sequence"
    assert args.reuse_manifest_question_ir is False
    assert reuse_ir_args.reuse_manifest_question_ir is True


def test_run_qa_batch_tasks_warms_cache_before_parallel(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    calls: list[str] = []

    def fake_run_command(
        command: list[str],
        *,
        retries: int,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        calls.append(command[0])

    monkeypatch.setattr(module, "run_command", fake_run_command)
    tasks = [
        module.QABatchTask(
            offset=index,
            limit=1,
            batch_json=tmp_path / f"batch_{index}.json",
            command=[str(index)],
        )
        for index in range(3)
    ]

    module.run_qa_batch_tasks(tasks, retries=0, workers=2, cwd=tmp_path)

    assert calls[0] == "0"
    assert sorted(calls) == ["0", "1", "2"]


def test_parse_api_key_pool_trims_and_deduplicates() -> None:
    module = _load_module()

    assert module.parse_api_key_pool(" k1, k2 ,,k1, k3 ") == ["k1", "k2", "k3"]


def test_run_qa_batch_tasks_assigns_api_keys_without_command_leak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    observed: list[tuple[list[str], str | None]] = []

    def fake_run_command(
        command: list[str],
        *,
        retries: int,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        observed.append((command, env.get("DEEPSEEK_API_KEY") if env else None))

    monkeypatch.setattr(module, "run_command", fake_run_command)
    tasks = [
        module.QABatchTask(
            offset=index,
            limit=1,
            batch_json=tmp_path / f"batch_{index}.json",
            command=["python", "qa.py", str(index)],
            api_key_index=index,
        )
        for index in range(5)
    ]

    module.run_qa_batch_tasks(
        tasks,
        retries=0,
        workers=5,
        cwd=tmp_path,
        api_keys=["key-a", "key-b"],
    )

    commands = [command for command, _key in observed]
    assigned_keys = [_key for _command, _key in observed]
    assert all("key-a" not in command and "key-b" not in command for command in commands)
    assert assigned_keys.count("key-a") == 3
    assert assigned_keys.count("key-b") == 2


def test_run_global_qa_batch_tasks_warms_each_sequence_before_shared_fanout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[str] = []

    def fake_run_command(
        command: list[str],
        *,
        retries: int,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        calls.append(command[0])

    monkeypatch.setattr(module, "run_command", fake_run_command)

    def make_plan(seq_id: str) -> object:
        spec = module.SequenceSpec(
            seq_id=seq_id,
            cam="0",
            stride=10,
            layout_kind="test",
            sample_root=tmp_path / f"seq{seq_id}",
            canonical_graph_json=tmp_path / f"seq{seq_id}" / "graph.json",
            frame_time_index_json=None,
            qa_unfilled_json=None,
            question_ir_json=None,
            projection_cache=None,
            existing_baseline_json=None,
        )
        paths = module.MainlinePaths(
            output_dir=tmp_path / f"run{seq_id}",
            question_ir_json=tmp_path / f"ir{seq_id}.json",
            question_ir_md=tmp_path / f"ir{seq_id}.md",
            merged_json=tmp_path / f"merged{seq_id}.json",
            merged_csv=tmp_path / f"merged{seq_id}.csv",
            merged_md=tmp_path / f"merged{seq_id}.md",
            projection_cache=tmp_path / f"cache{seq_id}.json.gz",
        )
        tasks = [
            module.QABatchTask(
                offset=index,
                limit=1,
                batch_json=tmp_path / f"seq{seq_id}_batch_{index}.json",
                command=[f"{seq_id}-{index}"],
            )
            for index in range(2)
        ]
        return module.SequenceRunPlan(
            spec=spec,
            selected_rows=[],
            paths=paths,
            batch_paths=[task.batch_json for task in tasks],
            batch_ranges_for_sequence=[(task.offset, task.limit) for task in tasks],
            qa_tasks=tasks,
        )

    module.run_global_qa_batch_tasks(
        [make_plan("1"), make_plan("2")],
        retries=0,
        workers=2,
        cwd=tmp_path,
    )

    assert calls[:2] == ["1-0", "2-0"]
    assert sorted(calls) == ["1-0", "1-1", "2-0", "2-1"]


def test_run_global_qa_batch_tasks_assigns_api_keys_without_command_leak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    observed: list[tuple[list[str], str | None]] = []

    def fake_run_command(
        command: list[str],
        *,
        retries: int,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        observed.append((command, env.get("DEEPSEEK_API_KEY") if env else None))

    monkeypatch.setattr(module, "run_command", fake_run_command)

    spec = module.SequenceSpec(
        seq_id="1",
        cam="0",
        stride=10,
        layout_kind="test",
        sample_root=tmp_path / "seq1",
        canonical_graph_json=tmp_path / "seq1" / "graph.json",
        frame_time_index_json=None,
        qa_unfilled_json=None,
        question_ir_json=None,
        projection_cache=None,
        existing_baseline_json=None,
    )
    paths = module.MainlinePaths(
        output_dir=tmp_path / "run1",
        question_ir_json=tmp_path / "ir1.json",
        question_ir_md=tmp_path / "ir1.md",
        merged_json=tmp_path / "merged1.json",
        merged_csv=tmp_path / "merged1.csv",
        merged_md=tmp_path / "merged1.md",
        projection_cache=tmp_path / "cache1.json.gz",
    )
    tasks = [
        module.QABatchTask(
            offset=index,
            limit=1,
            batch_json=tmp_path / f"batch_{index}.json",
            command=["python", "qa.py", str(index)],
        )
        for index in range(4)
    ]
    plan = module.SequenceRunPlan(
        spec=spec,
        selected_rows=[],
        paths=paths,
        batch_paths=[task.batch_json for task in tasks],
        batch_ranges_for_sequence=[(task.offset, task.limit) for task in tasks],
        qa_tasks=tasks,
    )

    module.run_global_qa_batch_tasks(
        [plan],
        retries=0,
        workers=3,
        cwd=tmp_path,
        api_keys=["key-a", "key-b"],
    )

    commands = [command for command, _key in observed]
    assigned_keys = [_key for _command, _key in observed]
    assert all("key-a" not in command and "key-b" not in command for command in commands)
    assert assigned_keys.count("key-a") == 2
    assert assigned_keys.count("key-b") == 2


def test_qa_batch_command_can_enable_generic_llm_evidence_rerank(tmp_path: Path) -> None:
    module = _load_module()
    spec = module.SequenceSpec(
        seq_id="3",
        cam="0",
        stride=10,
        layout_kind="unified",
        sample_root=tmp_path / "sample",
        canonical_graph_json=tmp_path / "sample" / "graph.json",
        frame_time_index_json=None,
        qa_unfilled_json=None,
        question_ir_json=None,
        projection_cache=None,
        existing_baseline_json=None,
    )
    paths = module.MainlinePaths(
        output_dir=tmp_path / "run",
        question_ir_json=tmp_path / "ir.json",
        question_ir_md=tmp_path / "ir.md",
        merged_json=tmp_path / "merged.json",
        merged_csv=tmp_path / "merged.csv",
        merged_md=tmp_path / "merged.md",
        projection_cache=tmp_path / "cache.json.gz",
    )

    command = module.build_qa_batch_command(
        spec,
        paths,
        csv_path=tmp_path / "qa.csv",
        config="default.yaml",
        answer_mode="live",
        domain_profile="generic",
        retrieval_limit=10,
        text_judge_mode="none",
        llm_evidence_rerank=True,
        llm_evidence_rerank_max_candidates=12,
        llm_evidence_rerank_max_selected=4,
        answer_candidate_extraction=True,
        answer_candidate_extraction_max_candidates=5,
        answer_candidate_extraction_max_tokens=600,
        experimental_ir_evidence_query_expansion=False,
        experimental_balanced_evidence_candidates=False,
        experimental_answerable_candidate_packs=True,
        answerable_candidate_pack_max_items=11,
        experimental_unified_evidence_layer=False,
        unified_evidence_max_items=10,
        unified_evidence_graph_quota=5,
        unified_evidence_visual_quota=3,
        unified_evidence_frame_quota=2,
        experimental_atomic_answer_facts=True,
        atomic_answer_fact_max_items=7,
        experimental_visual_faiss_evidence=False,
        experimental_visual_query_faiss_recall=True,
        experimental_visual_faiss_fused_evidence=False,
        experimental_visual_faiss_graph_anchors=False,
        experimental_visual_faiss_ger_fusion=True,
        experimental_visual_faiss_ger_binding=True,
        experimental_visual_faiss_frame_context=False,
        experimental_visual_faiss_structural_binding=False,
        experimental_visual_faiss_object_local_facts=True,
        experimental_visual_faiss_unified_candidates=False,
        visual_faiss_index_dir=tmp_path / "faiss",
        visual_faiss_index_name="object_siglip_flat_ip",
        visual_faiss_seed_limit=4,
        visual_faiss_top_k=5,
        visual_faiss_max_lines=8,
        visual_faiss_query_phrase_limit=6,
        visual_faiss_query_top_k=8,
        visual_faiss_query_search_k=1000,
        visual_faiss_query_min_score=0.0,
        visual_faiss_ger_object_limit=8,
        visual_faiss_ger_context_radius=8,
        visual_faiss_ger_context_limit=12,
        visual_faiss_cross_sequence=False,
        experimental_frame_faiss_query_recall=False,
        experimental_frame_faiss_fused_evidence=False,
        experimental_frame_faiss_graph_anchors=False,
        experimental_merged_frame_faiss_query_recall=False,
        experimental_merged_frame_faiss_fused_evidence=False,
        experimental_merged_frame_faiss_graph_anchors=False,
        experimental_frame_window_faiss_query_recall=False,
        experimental_frame_window_faiss_graph_anchors=False,
        offset=0,
        limit=2,
        batch_json=tmp_path / "batch.json",
    )

    assert "--llm-evidence-rerank" in command
    assert command[command.index("--llm-evidence-rerank-max-candidates") + 1] == "12"
    assert command[command.index("--llm-evidence-rerank-max-selected") + 1] == "4"
    assert "--answer-candidate-extraction" in command
    assert command[command.index("--answer-candidate-extraction-max-candidates") + 1] == "5"
    assert command[command.index("--answer-candidate-extraction-max-tokens") + 1] == "600"
    assert "--experimental-atomic-answer-facts" in command
    assert command[command.index("--atomic-answer-fact-max-items") + 1] == "7"
    assert "--experimental-answerable-candidate-packs" in command
    assert command[command.index("--answerable-candidate-pack-max-items") + 1] == "11"


def test_build_qa_batch_command_passes_visual_frame_context_flag(tmp_path: Path) -> None:
    module = _load_module()
    spec = module.SequenceSpec(
        seq_id="0",
        cam="cam0",
        stride=10,
        layout_kind="canonical",
        sample_root=tmp_path / "sample",
        canonical_graph_json=tmp_path / "graph.json",
        frame_time_index_json=None,
        qa_unfilled_json=None,
        question_ir_json=None,
        projection_cache=None,
        existing_baseline_json=None,
    )
    paths = module.MainlinePaths(
        output_dir=tmp_path / "out",
        question_ir_json=tmp_path / "ir.json",
        question_ir_md=tmp_path / "ir.md",
        merged_json=tmp_path / "merged.json",
        merged_csv=tmp_path / "merged.csv",
        merged_md=tmp_path / "merged.md",
        projection_cache=tmp_path / "cache.json",
    )

    command = module.build_qa_batch_command(
        spec,
        paths,
        csv_path=tmp_path / "qa.csv",
        config="default.yaml",
        answer_mode="live",
        domain_profile="generic",
        retrieval_limit=10,
        text_judge_mode="none",
        llm_evidence_rerank=True,
        llm_evidence_rerank_max_candidates=12,
        llm_evidence_rerank_max_selected=4,
        answer_candidate_extraction=False,
        answer_candidate_extraction_max_candidates=5,
        answer_candidate_extraction_max_tokens=600,
        experimental_ir_evidence_query_expansion=False,
        experimental_balanced_evidence_candidates=False,
        experimental_answerable_candidate_packs=False,
        answerable_candidate_pack_max_items=12,
        experimental_unified_evidence_layer=False,
        unified_evidence_max_items=10,
        unified_evidence_graph_quota=5,
        unified_evidence_visual_quota=3,
        unified_evidence_frame_quota=2,
        experimental_atomic_answer_facts=False,
        atomic_answer_fact_max_items=7,
        experimental_visual_faiss_evidence=False,
        experimental_visual_query_faiss_recall=True,
        experimental_visual_faiss_fused_evidence=False,
        experimental_visual_faiss_graph_anchors=False,
        experimental_visual_faiss_ger_fusion=True,
        experimental_visual_faiss_ger_binding=True,
        experimental_visual_faiss_frame_context=True,
        experimental_visual_faiss_structural_binding=True,
        experimental_visual_faiss_fragment_bridge=True,
        experimental_visual_faiss_object_local_facts=True,
        experimental_visual_faiss_balanced_graph_visual=True,
        experimental_visual_faiss_unified_candidates=False,
        visual_faiss_index_dir=tmp_path / "faiss",
        visual_faiss_index_name="object_siglip_flat_ip",
        visual_faiss_seed_limit=4,
        visual_faiss_top_k=5,
        visual_faiss_max_lines=8,
        visual_faiss_query_phrase_limit=6,
        visual_faiss_query_top_k=8,
        visual_faiss_query_search_k=1000,
        visual_faiss_query_min_score=0.0,
        visual_faiss_ger_object_limit=8,
        visual_faiss_ger_context_radius=8,
        visual_faiss_ger_context_limit=12,
        visual_faiss_cross_sequence=False,
        experimental_frame_faiss_query_recall=False,
        experimental_frame_faiss_fused_evidence=False,
        experimental_frame_faiss_graph_anchors=False,
        experimental_merged_frame_faiss_query_recall=False,
        experimental_merged_frame_faiss_fused_evidence=False,
        experimental_merged_frame_faiss_graph_anchors=False,
        experimental_frame_window_faiss_query_recall=False,
        experimental_frame_window_faiss_graph_anchors=False,
        offset=0,
        limit=2,
        batch_json=tmp_path / "batch.json",
    )

    assert "--experimental-visual-query-faiss-recall" in command
    assert "--experimental-visual-faiss-ger-fusion" in command
    assert "--experimental-visual-faiss-ger-binding" in command
    assert "--experimental-visual-faiss-frame-context" in command
    assert "--experimental-visual-faiss-structural-binding" in command
    assert "--experimental-visual-faiss-fragment-bridge" in command
    assert "--experimental-visual-faiss-object-local-facts" in command
    assert "--experimental-visual-faiss-balanced-graph-visual" in command
    assert command[command.index("--visual-faiss-query-search-k") + 1] == "1000"


def test_summarize_payload_uses_official_style_as_primary_metric() -> None:
    module = _load_module()

    summary = module.summarize_payload(
        {
            "navqa_official_style_accuracy": {
                "correct": 3,
                "total": 3,
                "accuracy_percent": 100.0,
                "temporal_threshold_minutes": 2.0,
                "text_judging": "llm_judge_when_available_else_normalized_containment_proxy",
                "by_type": {
                    "time": {"correct": 1, "total": 1, "accuracy_percent": 100.0},
                    "text": {"correct": 1, "total": 1, "accuracy_percent": 100.0},
                    "binary": {"correct": 1, "total": 1, "accuracy_percent": 100.0},
                },
            },
        }
    )

    assert summary["metric"] == "navqa_official_style_accuracy"
    assert summary["correct"] == 3
    assert summary["accuracy_percent"] == 100.0
    assert (
        summary["official_metric_rules"]["judge"]
        == "llm_judge_when_available_else_normalized_containment_proxy"
    )
    assert summary["official_metric_rules"]["temporal_threshold_minutes"] == 2.0
    assert "strict" not in summary
    assert "raw_llm" not in summary
    assert "text_binary" not in summary


def test_all_sequence_summary_markdown_displays_accuracy_metrics(tmp_path: Path) -> None:
    module = _load_module()

    _, summary_md = module.write_all_sequence_summary(
        output_root=tmp_path,
        run_name_value="run",
        run_metadata={
            "cg_id": "CG6",
            "ir_qa_id": "IR-QA-F-v2",
            "route_id": "CG6+IR-QA-F-v2",
        },
        rows=[
            {
                "sequence": "seq0_cam0_every10",
                "layout_kind": "manifest_mainline",
                "previous": {"accuracy_percent": 66.67},
                "new": {
                    "correct": 2,
                    "total": 3,
                    "accuracy_percent": 66.67,
                    "by_type": {
                        "time": {"correct": 1, "total": 1, "accuracy_percent": 100.0}
                    },
                },
                "delta_accuracy_percent": 0.0,
            }
        ],
    )

    text = summary_md.read_text(encoding="utf-8")
    assert "route_id: `CG6+IR-QA-F-v2`" in text
    assert "cg_id: `CG6`" in text
    assert "ir_qa_id: `IR-QA-F-v2`" in text
    assert "| seq | layout | questions | correct | acc | previous acc | delta | by type |" in text
    assert "official-rule LLM acc" not in text
    assert "strict acc" not in text
    assert "raw LLM acc" not in text
    assert "66.67" in text
