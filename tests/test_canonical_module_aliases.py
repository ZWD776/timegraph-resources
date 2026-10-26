import importlib


def test_compatibility_imports_share_one_model_implementation():
    old = importlib.import_module('vlkg.models')
    new = importlib.import_module('vlkg.core.models.temporal_graph')
    assert old is new
    assert old.Entity is new.Entity
    assert old.__spec__.name == 'vlkg.core.models.temporal_graph'


def test_dataset_package_and_submodules_keep_canonical_identity():
    for dataset in ('nextqa', 'openeqa'):
        old = importlib.import_module('vlkg.eval_adapters.' + dataset)
        new = importlib.import_module('vlkg.datasets.' + dataset)
        assert old is new
    old = importlib.import_module('vlkg.eval_adapters.nextqa.temporal_canonical')
    new = importlib.import_module('vlkg.datasets.nextqa.graph.temporal_canonical')
    assert old is new
    assert new.__spec__.name == 'vlkg.datasets.nextqa.graph.temporal_canonical'


def test_shared_retriever_and_types_have_core_ownership():
    from vlkg import run_canonical_graph_hybrid_retrieval as public_retriever
    new = importlib.import_module('vlkg.core.graph_retrieval.canonical_hybrid_retrieval')
    assert public_retriever is new.run_canonical_graph_hybrid_retrieval
    types = importlib.import_module('vlkg.core.graph_retrieval.canonical_retrieval_types')
    assert types.__name__ == 'vlkg.core.graph_retrieval.canonical_retrieval_types'


def test_selected_script_imports_survive_sealing_the_old_directories():
    from scripts.nextqa import build_visual_indexes
    from scripts.openeqa import build_openeqa_single_route_ir

    assert build_visual_indexes is importlib.import_module(
        'scripts.datasets.nextqa.build.build_visual_indexes')
    assert build_openeqa_single_route_ir is importlib.import_module(
        'scripts.datasets.openeqa.retrieval.build_openeqa_single_route_ir')
