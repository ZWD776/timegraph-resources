from __future__ import annotations
from vlkg.core.assets.asset_provenance import relocated_provenance
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from vlkg.datasets.openeqa.retrieval.evidence_router import ROUTING_PROFILE_IR_FIRST_V4, compile_evidence_demand, visual_query_intents
from vlkg.datasets.openeqa.retrieval.semantic_question_ir import semantic_question_ir_projection
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import VISUAL_BACKBONE_OPENEQA_QR_V4, select_visual_builder
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_CONTRACT as SINGLE_ROUTE_CONTRACT, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, route_handler_id as single_route_handler_id, route_visual_query_builder_id as single_route_visual_query_builder_id, route_visual_strategy_contract as single_route_visual_strategy_contract
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import route_handler_id
SCRIPT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / 'scripts' / 'openeqa'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
launcher = import_module('scripts.datasets.openeqa.qa.run_openeqa_unified_visual_qa')
graph_only = import_module('scripts.datasets.openeqa.qa.run_hm3d_graph_only_qa')
visual_runner = import_module('scripts.datasets.openeqa.qa.run_openeqa_visual_evidence_qa')

def _rows() -> list[dict]:
    return [{'question_id': 'h1', 'episode_history': 'hm3d-v0/episode-1'}, {'question_id': 's1', 'episode_history': 'scannet-v0/episode-1'}]

def test_resource_manifest_treats_sources_as_resource_shards_only() -> None:
    payload = {'contract': 'openeqa_unified_execution_resources.v1', 'execution_shards': [{'data_source': 'hm3d-v0', 'graph_manifest': 'hm3d-graphs.json', 'visual_index_dir': 'hm3d-visual', 'visual_index_name': 'shared-index-name'}, {'data_source': 'scannet-v0', 'graph_manifest': 'scannet-graphs.json', 'visual_index_dir': 'scannet-visual', 'visual_index_name': 'shared-index-name'}]}
    issues = launcher.validate_resource_manifest(payload, dataset_rows=_rows(), require_paths=False)
    assert issues == []

def test_resource_manifest_requires_every_open_eqa_source() -> None:
    payload = {'contract': 'openeqa_unified_execution_resources.v1', 'execution_shards': [{'data_source': 'hm3d-v0', 'graph_manifest': 'hm3d-graphs.json', 'visual_index_dir': 'hm3d-visual', 'visual_index_name': 'shared-index-name'}]}
    issues = launcher.validate_resource_manifest(payload, dataset_rows=_rows(), require_paths=False)
    assert any(('source coverage mismatch' in issue for issue in issues))

def test_unified_launcher_requires_one_full_question_route_manifest() -> None:
    payload = {'contract': 'openeqa_unified_question_route_ir.v1', 'dataset': 'OpenEQA', 'uses_gold_answer': False, 'runtime_route_recompile_allowed': False, 'records': [{'question_id': 'h1'}, {'question_id': 's1'}]}
    assert launcher.validate_question_route_manifest(payload, dataset_rows=_rows()) == []
    payload['records'].pop()
    issues = launcher.validate_question_route_manifest(payload, dataset_rows=_rows())
    assert any(('coverage mismatch' in issue for issue in issues))

def test_ir_first_manifest_rejects_legacy_execution_config() -> None:
    payload = {'contract': 'openeqa_unified_question_route_ir.v1', 'dataset': 'OpenEQA', 'routing_profile': ROUTING_PROFILE_IR_FIRST_V4, 'uses_gold_answer': False, 'uses_official_question_category_for_routing': False, 'routing_inputs': ['question', 'semantic_question_ir'], 'runtime_route_recompile_allowed': False, 'persists_route_strategy_handler': True, 'records': [{'question_id': 'h1'}, {'question_id': 's1'}]}
    issues = launcher.validate_question_route_manifest(payload, dataset_rows=_rows(), visual_backbone='legacy_v4', graph_retrieval_profile='legacy_entity_auto')
    assert any(('requires visual_backbone' in issue for issue in issues))
    assert any(('category-independent unified graph' in issue for issue in issues))

def test_ir_first_shard_runner_rejects_legacy_execution_config() -> None:
    demand = compile_evidence_demand('What object is beside the table?', question_ir={'operation': 'identity_query', 'answer_type': 'entity', 'spatial_constraints': [{'relation': 'beside', 'target': 'table'}]}, routing_profile=ROUTING_PROFILE_IR_FIRST_V4).to_dict()
    issues = visual_runner.ir_first_shard_config_issues(selected_rows=[{'question_id': 'h1'}], question_ir_records={'h1': {'question_route': demand}}, visual_backbone='legacy_v4', graph_retrieval_profile='legacy_entity_auto')
    assert any(('requires visual_backbone' in issue for issue in issues))
    assert any(('category-independent unified graph' in issue for issue in issues))
    assert visual_runner.ir_first_shard_config_issues(selected_rows=[{'question_id': 'h1'}], question_ir_records={'h1': {'question_route': demand}}, visual_backbone=VISUAL_BACKBONE_OPENEQA_QR_V4, graph_retrieval_profile='canonical_vector_evidence_v5') == []

def test_v5_question_type_pilot_filter_is_diagnostic_only() -> None:
    rows = [{'question_id': 'q1'}, {'question_id': 'q2'}, {'question_id': 'q3'}, {'question_id': 'q4'}]
    records = {'q1': {'question_type': 'entity_state'}, 'q2': {'question_type': 'entity_state'}, 'q3': {'question_type': 'entity_state'}, 'q4': {'question_type': 'entity_count'}}
    selected, issues = visual_runner.select_v5_question_type_rows(rows, question_ir_records=records, question_types=['entity_state', 'entity_count'], per_type_limit=2)
    assert issues == []
    assert [row['question_id'] for row in selected] == ['q1', 'q2', 'q4']
    assert records == {'q1': {'question_type': 'entity_state'}, 'q2': {'question_type': 'entity_state'}, 'q3': {'question_type': 'entity_state'}, 'q4': {'question_type': 'entity_count'}}

def test_v5_question_type_pilot_filter_rejects_unknown_type() -> None:
    selected, issues = visual_runner.select_v5_question_type_rows([{'question_id': 'q1'}], question_ir_records={'q1': {'question_type': 'entity_state'}}, question_types=['runtime_subtype'], per_type_limit=1)
    assert selected == [{'question_id': 'q1'}]
    assert any(('unknown V5 question types' in issue for issue in issues))

def test_open_eqa_qa_temperature_override_is_explicit() -> None:
    args = type('Args', (), {'max_tokens': 256, 'qa_temperature': 0.0, 'graph_retrieval_profile': 'canonical_vector_evidence_v5'})()
    overrides = graph_only.graph_qa_overrides(args)
    assert overrides['temperature'] == 0.0
    assert overrides['max_tokens'] == 256

def test_question_id_selection_reuses_audit_json_records(tmp_path: Path) -> None:
    selection = tmp_path / 'audit.json'
    selection.write_text(json.dumps([{'question_id': 'q1', 'prediction': 'ignored'}, {'uuid': 'q2', 'score': 5}]), encoding='utf-8')
    assert graph_only.load_question_id_jsons([selection]) == ['q1', 'q2']

def test_qa_key_pin_keeps_only_selected_pool_slot(monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEYS', 'first,second,third')
    monkeypatch.setenv('OPENAI_API_KEY', 'fallback')
    size = graph_only.pin_qa_api_key('datasets/openeqa/qa.yaml', 1)
    assert size == 3
    assert os.environ['OPENAI_API_KEY'] == 'second'
    assert 'OPENAI_API_KEYS' not in os.environ

def test_ir_first_category_counterfactual_keeps_full_retrieval_control_plane() -> None:
    question = 'What object is beside the table?'
    semantic_ir = {'operation': 'identity_query', 'answer_type': 'entity', 'target_entities': ['object', 'table'], 'spatial_constraints': [{'relation': 'beside', 'target': 'table'}]}
    signatures: list[dict] = []
    for category in ('object recognition', '', 'random category 8472'):
        demand = compile_evidence_demand(question, category=category, question_ir=semantic_ir, routing_profile=ROUTING_PROFILE_IR_FIRST_V4).to_dict()
        graph_ir = semantic_question_ir_projection({**semantic_ir, 'category': category, 'question_route': demand})
        signatures.append({'retrieval_mode': graph_only.resolve_retrieval_mode(category, 'auto', question_ir=graph_ir, graph_retrieval_profile='canonical_vector_evidence_v5'), 'graph_query': graph_only.query_text_from_question_ir(question, graph_ir), 'visual_query_intents': visual_query_intents(demand), 'visual_builder': select_visual_builder(VISUAL_BACKBONE_OPENEQA_QR_V4, evidence_demand=demand), 'handler': route_handler_id(demand)})
    assert signatures[1:] == signatures[:1] * 2
    assert signatures[0]['retrieval_mode'] == 'temporal'
    assert signatures[0]['handler'] == 'object_identity.anchor_bound_relation_incremental_v1'

def test_unified_runner_accepts_v5_single_route_manifest_contract() -> None:
    route_id = 'surface_capacity'
    handler_id = single_route_handler_id({'route_id': route_id})
    visual_query_builder_id = single_route_visual_query_builder_id({'route_id': route_id})
    visual_strategy_contract = single_route_visual_strategy_contract({'route_id': route_id})
    ir = {'question_type': route_id, 'uses_gold_answer': False, 'route_decision_is_persisted': True, 'route_decision_runtime_recompile_allowed': False, 'question_route_id': route_id, 'question_route': {'contract': SINGLE_ROUTE_CONTRACT, 'route_id': route_id, 'route_source': 'semantic_question_ir.question_type', 'handler_id': handler_id, 'visual_query_builder_id': visual_query_builder_id, 'visual_strategy_contract': visual_strategy_contract, 'uses_gold_answer': False, 'uses_official_question_category': False, 'runtime_route_recompile_allowed': False, 'metadata': {'routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5}}}
    payload = {'contract': 'openeqa_unified_question_route_ir.v1', 'route_contract': SINGLE_ROUTE_CONTRACT, 'dataset': 'OpenEQA', 'routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, 'uses_gold_answer': False, 'uses_official_question_category_for_routing': False, 'routing_inputs': ['semantic_question_ir.question_type'], 'runtime_route_recompile_allowed': False, 'persists_route_strategy_handler': True, 'records': [{'question_id': 'h1', 'llm_question_ir': ir}, {'question_id': 's1', 'llm_question_ir': ir}]}
    assert launcher.validate_question_route_manifest(payload, dataset_rows=_rows(), visual_backbone='openeqa_question_router_v5', graph_retrieval_profile='canonical_vector_evidence_v5') == []
    payload['route_contract'] = 'openeqa_question_route_decision.v2'
    assert any(('single-route manifest route_contract' in issue for issue in launcher.validate_question_route_manifest(payload, dataset_rows=_rows())))
