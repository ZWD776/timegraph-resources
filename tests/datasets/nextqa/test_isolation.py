from __future__ import annotations

import ast
from pathlib import Path

import vlkg.datasets.nextqa.audit.archive_audit as archive_audit
import vlkg.datasets.nextqa.runtime.contracts as contracts
import vlkg.datasets.nextqa.audit.graph_audit as graph_audit
import vlkg.datasets.nextqa.graph.hsgg_input_adapter as hsgg_input_adapter
import vlkg.datasets.nextqa.runtime.io_utils as io_utils
import vlkg.datasets.nextqa.audit.preflight as preflight
import vlkg.datasets.nextqa.runtime.run_snapshot as run_snapshot
import vlkg.datasets.nextqa.runtime.standardization as standardization
import vlkg.datasets.nextqa.graph.temporal_canonical as temporal_canonical


def test_nextqa_preflight_is_dataset_isolated() -> None:
    for module in (
        archive_audit,
        contracts,
        graph_audit,
        hsgg_input_adapter,
        io_utils,
        preflight,
        run_snapshot,
        standardization,
        temporal_canonical,
    ):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        imported_modules = {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("eval_adapters.navqa" in name for name in imported_modules)
        assert not any("eval_adapters.openeqa" in name for name in imported_modules)


def test_graph_pilot_modules_do_not_import_visual_or_api_clients() -> None:
    modules = (contracts, graph_audit, hsgg_input_adapter, io_utils, run_snapshot)
    forbidden = {"faiss", "openai", "PIL", "requests", "httpx"}
    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(name.split(".")[0] in forbidden for name in imported)


def test_graph_builder_never_reads_question_or_evaluator_files() -> None:
    paths = [
        Path(hsgg_input_adapter.__file__),
        Path(temporal_canonical.__file__),
        Path(graph_audit.__file__),
        next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / 'scripts' / 'datasets' / 'nextqa' / 'build' / 'build_graph_full.py',
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert "questions.runtime.json" not in source
        assert "labels.evaluator_only.json" not in source


def test_graph_construction_visual_use_does_not_relax_graph_only_qa_boundary() -> None:
    source = Path(temporal_canonical.__file__).read_text(encoding="utf-8")
    assert "'uses_visual_embeddings_for_graph_construction': True" in source
    assert "'qa_reads_rgb': False" in source
    assert "'qa_reads_object_crop': False" in source
    assert "'qa_reads_embedding_cache': False" in source
    assert "'qa_uses_faiss': False" in source


def test_nextqa_does_not_trust_cloud_image_paths() -> None:
    source = Path(preflight.__file__).read_text(encoding="utf-8").lower()
    assert "/cloud/" not in source
    assert "frame_member_template" in source


def test_hsgg_generation_provenance_contract() -> None:
    provenance = preflight.HSGG_GENERATION_PROVENANCE
    assert provenance["first_half_strategy"] == (
        "every60_initial_run_plus_every30_supplement"
    )
    assert provenance["second_half_strategy"] == "direct_every30_run"
    assert provenance["frame_stride"] == 30
    assert "video_fps_approx" not in provenance
    assert provenance["timestamp_policy"] == (
        "use_fps_derived_seconds_encoded_in_frame_filename"
    )


def test_nextqa_visual_runtime_does_not_import_other_dataset_adapters() -> None:
    package_root = (
        next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
        / 'src' / 'vlkg' / 'datasets' / 'nextqa' / 'runtime' / 'visual_runtime_v1'
    )
    script_root = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / 'scripts' / 'datasets' / 'nextqa'
    paths = [*package_root.glob("*.py")]
    paths.extend(path for path in script_root.glob("*visual*.py"))
    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("eval_adapters.navqa" in name for name in imported)
        assert not any("eval_adapters.openeqa" in name for name in imported)
