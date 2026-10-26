from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import csv
import json
import os
import re
import socket
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.navqa.qa.canonical_graph_qa_context import answer_canonical_graph_query
from vlkg.core.config.config import GraphQAConfig, load_domain_config
from vlkg.core.config.domain_lexicon import DEFAULT_DOMAIN_PROFILE, available_domain_profiles
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession
from vlkg.datasets.navqa.qa.official_qa import load_position_labels, materialize_official_answer_label, official_answer_format_instruction, official_label_to_dict, question_adaptation_context
from vlkg.datasets.navqa.runtime.paths import default_navqa_sequence_layout
from vlkg.datasets.navqa.qa.position_reference import PositionWindow, build_position_windows_from_pose_samples, coda_preprocessed_pose_samples, load_dense_poses, load_position_windows_from_captions, load_timestamps, lookup_position_for_timestamp
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.graph_retrieval.question_compiler import compile_question
from vlkg.datasets.navqa.retrieval.question_route_hybrid_backbone import QUESTION_ROUTER_HYBRID_IR_QA_ID, QUESTION_ROUTER_MAINLINE_IR_QA_ID, QUESTION_ROUTER_SPECIALIZED_IR_QA_ID, fuse_question_route_hybrid_backbone, hybrid_visual_faiss_config
from vlkg.datasets.navqa.retrieval.question_router import attach_question_route_to_ir, route_question
from vlkg.datasets.navqa.retrieval.synonym_lexicon import DEFAULT_SYNONYM_LEXICON_PATH, SynonymLexicon, expand_question_ir_with_synonyms, load_synonym_lexicon, synonym_expansion_metadata
from vlkg.core.temporal.time_adaptation import build_graph_time_adaptation
from vlkg.datasets.navqa.retrieval.frame_faiss_grounding import DEFAULT_FRAME_FAISS_INDEX_DIR, DEFAULT_FRAME_FAISS_INDEX_NAME, FrameFaissConfig, FrameFaissGrounder, attach_frame_faiss_evidence, expand_frame_faiss_graph_evidence, fuse_frame_faiss_evidence
from vlkg.datasets.navqa.qa.graph_qa_pipeline import answer_via_mode
from vlkg.datasets.navqa.retrieval.merged_frame_faiss_grounding import DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR, DEFAULT_MERGED_FRAME_FAISS_INDEX_NAME, FrameWindowFaissConfig, FrameWindowFaissGrounder, MergedFrameFaissConfig, MergedFrameFaissGrounder, attach_merged_frame_faiss_evidence, expand_merged_frame_faiss_graph_evidence, fuse_merged_frame_faiss_evidence
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import DEFAULT_VISUAL_FAISS_INDEX_DIR, DEFAULT_VISUAL_FAISS_INDEX_NAME, VisualFaissConfig, VisualFaissGrounder, attach_visual_faiss_role_aware_auxiliary_evidence, attach_visual_faiss_evidence, expand_visual_faiss_graph_evidence, fuse_visual_faiss_ger_evidence, fuse_visual_faiss_evidence_layer, fuse_visual_faiss_role_aware_evidence, fuse_visual_faiss_unified_candidates, fuse_visual_faiss_evidence
from vlkg.datasets.navqa.retrieval.visual_entity_candidate_packs import fuse_visual_entity_candidate_packs
FIELD_SEQ_ID = 'Seq ID'
FIELD_UUID = 'UUID'
FIELD_QUESTION = 'Question'
FIELD_TIMESTAMP = 'Timestamp \nwith answer'
FIELD_TYPE = 'Type \n(binary, position, time, text)'
FIELD_TEXT_ANSWER = 'Text answer'
FIELD_PARSABLE_ANSWER = 'Parsable answer'
FIELD_CATEGORY = 'Question\nCategory'
NON_POSITION_QA_TYPES = ('binary', 'text', 'time', 'duration')
NAVQA_OFFICIAL_TEMPORAL_THRESHOLD_MINUTES = 2.0
LLM_REQUEST_EXCEPTIONS = (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError, socket.timeout)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run official NaVQA questions against a saved canonical temporal memory graph.')
    parser.add_argument('canonical_graph_json', type=Path)
    parser.add_argument('--csv', type=Path, required=True, help='Official NaVQA data.csv.')
    parser.add_argument('--seq-id', default='0')
    parser.add_argument('--cam', default='0')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--layout-root', type=Path, default=(REPO_ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa'))
    parser.add_argument('--answer-mode', choices=('graph_only', 'mock', 'live'), default='live')
    parser.add_argument('--config', default='default.yaml')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions.')
    parser.add_argument('--offset', type=int, default=0, help='Skip this many sequence questions.')
    parser.add_argument('--include-types', nargs='+', default=None, help="Optional question types to include after seq-id filtering, e.g. 'binary text time duration'.")
    parser.add_argument('--exclude-types', nargs='+', default=None, help='Optional question types to exclude after include-types filtering.')
    parser.add_argument('--non-position-only', action='store_true', help='Archive-friendly current-mainline shortcut for --include-types binary text time duration.')
    parser.add_argument('--question-ir-json', type=Path, default=None, help='Optional experimental LLM question-IR JSON. When provided, the runner uses the compiled IR to build retrieval query text while preserving the original question for answer generation.')
    parser.add_argument('--retrieval-limit', type=int, default=10)
    parser.add_argument('--experimental-synonym-lexicon', action='store_true', help='Default-off experiment: expand question IR target/constraint terms with an explicit L0/L1 synonym lexicon before graph and FAISS retrieval. Never uses gold answers, question ids, or seq-specific rules.')
    parser.add_argument('--experimental-question-router', action='store_true', help='Default-off experiment: derive a generic QuestionRoute after question IR, attach it to MemoryQuery metadata, and expose it in QA audit outputs. It uses no gold answers, question ids, seq rules, or dataset/domain wordlists.')
    parser.add_argument('--experimental-question-router-hybrid-backbone', action='store_true', help='Default-off experiment: use QuestionRoute to choose a route-specific visual evidence backbone instead of sending every route through the same entity-pack score-contract path. Requires --experimental-question-router.')
    parser.add_argument('--experimental-question-router-specialized-backbone', action='store_true', help='Default-off experiment: QuestionRouter v3 specialized backbone. Keeps QuestionRoute routing, adds policy-aware count sets and temporal start/end boundary packs, and preserves old routes by default. Requires --experimental-question-router.')
    parser.add_argument('--experimental-question-router-mainline', action='store_true', help='Default-off experiment: consolidated QuestionRouter mainline. Uses the current route-specific specialized backbone table as one stable CG6 IR-QA route. Requires --experimental-question-router.')
    parser.add_argument('--synonym-lexicon-json', type=Path, default=DEFAULT_SYNONYM_LEXICON_PATH, help='L0/L1 synonym lexicon JSON used by --experimental-synonym-lexicon.')
    parser.add_argument('--llm-evidence-rerank', action='store_true', help='Run a generic LLM evidence-selection pass over retrieved candidates before final answer verbalization. The selector only sees graph evidence and the question, never gold answers.')
    parser.add_argument('--llm-evidence-rerank-max-candidates', type=int, default=18)
    parser.add_argument('--llm-evidence-rerank-max-selected', type=int, default=6)
    parser.add_argument('--answer-candidate-extraction', action='store_true', help='Experimental default-off generic answer-candidate extraction pass. It sees retrieved graph evidence only, applies to all question types, and supplies candidate answers to the final LLM verbalizer.')
    parser.add_argument('--answer-candidate-extraction-max-candidates', type=int, default=6)
    parser.add_argument('--answer-candidate-extraction-max-tokens', type=int, default=700)
    parser.add_argument('--experimental-ir-evidence-query-expansion', action='store_true', help='Default-off experiment: append question-IR evidence requirements and graph information needs to retrieval query text. This uses question text/IR only, never gold answers.')
    parser.add_argument('--experimental-balanced-evidence-candidates', action='store_true', help='Default-off experiment: balance LLM evidence-selection candidates across evidence lines, aggregation lines, candidate facts, temporal chains, and digest entries before truncation.')
    parser.add_argument('--experimental-answerable-candidate-packs', action='store_true', help='Default-off experiment: before LLM evidence selection, group already retrieved graph/object/visual candidates into compact answerable packs. This uses supplied evidence only, never gold answers, question-type branches, or dataset wordlists.')
    parser.add_argument('--answerable-candidate-pack-max-items', type=int, default=12)
    parser.add_argument('--experimental-unified-evidence-layer', action='store_true', help='Default-off experiment: before final QA, normalize graph, object FAISS, and frame/window FAISS evidence into one generic evidence pool. This uses retrieved evidence only, never gold answers.')
    parser.add_argument('--unified-evidence-max-items', type=int, default=10)
    parser.add_argument('--unified-evidence-graph-quota', type=int, default=5)
    parser.add_argument('--unified-evidence-visual-quota', type=int, default=3)
    parser.add_argument('--unified-evidence-frame-quota', type=int, default=2)
    parser.add_argument('--experimental-atomic-answer-facts', action='store_true', help='Default-off experiment: mechanically summarize supplied evidence into generic subject/predicate/object/attribute/time facts before final QA. This uses retrieved evidence only, never gold answers.')
    parser.add_argument('--atomic-answer-fact-max-items', type=int, default=8)
    parser.add_argument('--experimental-visual-faiss-evidence', action='store_true', help='Default-off experiment: add object-level visual similarity evidence from the FAISS side index to the LLM evidence bundle. This does not modify graph retrieval or use gold answers.')
    parser.add_argument('--experimental-visual-query-faiss-recall', action='store_true', help='Default-off experiment: encode question/IR visual phrases with SigLIP text features, search the object FAISS index across the current seq, and add the recalled object frames as post-graph evidence.')
    parser.add_argument('--experimental-visual-faiss-fused-evidence', action='store_true', help='Default-off experiment: promote visual FAISS object recall into the main evidence/candidate-fact pool before generic evidence selection. This uses question/IR and retrieved object metadata only, never gold answers.')
    parser.add_argument('--experimental-visual-faiss-graph-anchors', action='store_true', help='Default-off experiment: use visual FAISS recall as frame anchors, then promote nearby canonical graph evidence into the main evidence pool. FAISS supplies anchors only; final evidence remains graph evidence.')
    parser.add_argument('--experimental-visual-faiss-ger-fusion', action='store_true', help='Default-off experiment: treat object-level FAISS hits as GER-like retrieval candidates by fusing their HSGG facts and nearby KG context into the same evidence/candidate-fact pool as graph retrieval.')
    parser.add_argument('--experimental-visual-faiss-ger-binding', action='store_true', help='Default-off GER-like extension: before final QA, add generic bound facts that score visual FAISS object candidates by question/IR term coverage, object attributes, relations, and nearby graph context. Requires --experimental-visual-faiss-ger-fusion and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-frame-context', action='store_true', help='Default-off GER-like extension: enrich object-level FAISS hits with object-anchored same-frame HSGG context before GER fusion. Requires --experimental-visual-faiss-ger-fusion plus --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-structural-binding', action='store_true', help='Default-off GER-like extension: use a structural binding score that keeps target entities, attributes, event constraints, and spatial constraints as separate slots instead of treating all matched words as hard constraints. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-attribute-owner', action='store_true', help='Default-off GER-like extension: mark HSGG attributes with explicit owner roles before QA, separating direct_object_attribute from neighbor_object_attribute. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-owner-minimal-prompt', action='store_true', help='Default-off QA verbalization experiment: replace the long graph QA system policy with a compact owner-attribute prompt. This changes prompt wording only, not retrieval, FAISS, graph construction, or judging.')
    parser.add_argument('--experimental-visual-faiss-fragment-bridge', action='store_true', help='Default-off graph-local GER-like extension: package object-level FAISS hits, HSGG object facts, same-frame context, and nearby graph fragments into compact bridge facts before final QA. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-object-local-facts', action='store_true', help='Default-off GER-like extension: add short object-local facts for each recalled object using its direct HSGG attributes, direct relations, same-frame anchor/bbox context, and direct neighbor attributes. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-local-subgraph', action='store_true', help='Default-off CG6-adapted GER-like extension: map object-level FAISS hits back to canonical entity ids, then package each candidate entity as a local subgraph with direct attributes, relations, time range, visual anchor, and bounded neighbors. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-local-subgraph-anchor-only', action='store_true', help='Default-off CG6 local-subgraph v2 extension: only the temporal observation entity directly mapped from a FAISS object hit becomes a candidate local subgraph; nearby entities remain context only. Requires --experimental-visual-faiss-local-subgraph and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-balanced-graph-visual', action='store_true', help='Default-off GER-like extension: keep original graph evidence in the primary evidence pool while adding visual FAISS object-local, bound, and GER context evidence as balanced supplemental channels. Requires --experimental-visual-faiss-ger-binding and never uses gold answers, question-type branches, or dataset wordlists.')
    parser.add_argument('--experimental-visual-faiss-unified-candidates', action='store_true', help='Default-off GER-like extension: build unified graph+object-FAISS candidates before final QA by combining recalled objects, frame-level HSGG attributes/relations, and nearby canonical graph support. Requires --experimental-visual-faiss-ger-fusion and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-evidence-layer', action='store_true', help='Default-off experiment: use object FAISS hits to query the standalone evidence-layer SQLite index, then expose object-local packs with HSGG attributes, HSGG relations, bbox neighbors, image anchors, and time anchors. Never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-evidence-layer-constraint-binding', action='store_true', help='Default-off evidence-layer v2 extension: rank object-local evidence packs by generic question-IR target/attribute/event/spatial coverage. Requires --experimental-visual-faiss-evidence-layer and never uses gold answers, question-type branches, or dataset wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-evidence', action='store_true', help='Default-off object-FAISS route: use question-IR roles so target entities become candidate subjects while event/spatial/evidence-need hits stay as context attachments. Reads processed_attributes.json for selected HSGG objects and creates simple missing entries on demand. Never uses gold answers, question-type branches, or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-aux', action='store_true', help='Default-off auxiliary object-FAISS role-check layer. It can be combined with GER-like fusion: GER/object-local evidence remains primary, while role-aware target/context checks are exposed as auxiliary evidence only. Never uses gold answers, question-type branches, or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-compression', action='store_true', help='Default-off GER-binding experiment: separate question-IR target objects from constraint/context objects during object-FAISS compression before GER evidence packaging. Requires --experimental-visual-faiss-ger-fusion and --experimental-visual-faiss-ger-binding.')
    parser.add_argument('--experimental-visual-faiss-entity-pack', action='store_true', help='Default-off FAISS main route: use object FAISS hits as entity/tracklet candidate generators, map them back to CG6 entities, and expose visual_entity_candidate_pack evidence as the primary rerank unit. Requires visual FAISS recall and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-entity-pack-semantic-align', action='store_true', help='Default-off entity-pack v2 experiment: use SigLIP text embeddings inside each question to weakly align primary target phrases with candidate entity labels/direct attributes/direct relations. Requires --experimental-visual-faiss-entity-pack and never uses LLM semantic alignment, gold answers, or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-entity-pack-semantic-gated-role', action='store_true', help='Default-off entity-pack v3 experiment: use SigLIP text embeddings as a semantic gate for primary-target FAISS recalls. A primary target query hit remains provisional unless the candidate entity direct evidence aligns with the target more than with constraints. Requires --experimental-visual-faiss-entity-pack and never uses LLM semantic alignment, gold answers, or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-score-contract', action='store_true', help='Default-off entity-pack score-contract route: keep FAISS/entity-pack recall, then expose only normalized candidate_pack plus limited fallback evidence to LLM rerank. Requires --experimental-visual-faiss-entity-pack and never uses gold answers or dataset/domain wordlists.')
    parser.add_argument('--experimental-frame-faiss-query-recall', action='store_true', help='Default-off experiment: encode the question/IR as SigLIP text features, search the full-frame FAISS index in the current sequence, and add recalled frame clusters as post-graph visual evidence.')
    parser.add_argument('--experimental-frame-faiss-fused-evidence', action='store_true', help='Default-off experiment: promote full-frame FAISS recall into the main evidence/candidate-fact pool before generic evidence selection. This uses question/IR and frame metadata only, never gold answers.')
    parser.add_argument('--experimental-frame-faiss-graph-anchors', action='store_true', help='Default-off experiment: use full-frame FAISS recall as visual frame anchors, then promote nearby canonical graph evidence into the main evidence pool. FAISS supplies anchors only; final evidence remains graph evidence.')
    parser.add_argument('--experimental-merged-frame-faiss-query-recall', action='store_true', help='Default-off experiment: encode the question/IR as SigLIP text features, search the merged-frame storyboard FAISS index in the current sequence, and add recalled visual windows as post-graph evidence.')
    parser.add_argument('--experimental-merged-frame-faiss-fused-evidence', action='store_true', help='Default-off experiment: promote merged-frame FAISS clusters into the main evidence/candidate-fact pool. This uses question/IR and window metadata only, never gold answers.')
    parser.add_argument('--experimental-merged-frame-faiss-graph-anchors', action='store_true', help='Default-off experiment: use merged-frame FAISS recall as visual time-window anchors, then promote nearby canonical graph evidence. This is the recommended single-channel merged-frame route.')
    parser.add_argument('--experimental-frame-window-faiss-query-recall', action='store_true', help='Default-off experiment: retrieve full frames with frame FAISS, aggregate frame hits into temporal windows, and expose the result through the merged-frame evidence channel. This avoids storyboard image embeddings and uses question/IR plus frame metadata only.')
    parser.add_argument('--experimental-frame-window-faiss-graph-anchors', action='store_true', help='Default-off experiment: use frame-window FAISS recall as visual time-window anchors, then promote nearby canonical graph evidence.')
    parser.add_argument('--visual-faiss-index-dir', type=Path, default=DEFAULT_VISUAL_FAISS_INDEX_DIR)
    parser.add_argument('--visual-faiss-index-name', default=DEFAULT_VISUAL_FAISS_INDEX_NAME)
    parser.add_argument('--visual-faiss-seed-limit', type=int, default=4)
    parser.add_argument('--visual-faiss-top-k', type=int, default=5)
    parser.add_argument('--visual-faiss-max-lines', type=int, default=8)
    parser.add_argument('--visual-faiss-query-phrase-limit', type=int, default=6)
    parser.add_argument('--visual-faiss-query-top-k', type=int, default=8)
    parser.add_argument('--visual-faiss-query-search-k', type=int, default=1000)
    parser.add_argument('--visual-faiss-query-min-score', type=float, default=0.0)
    parser.add_argument('--visual-faiss-ger-object-limit', type=int, default=8)
    parser.add_argument('--visual-faiss-ger-context-radius', type=int, default=8)
    parser.add_argument('--visual-faiss-ger-context-limit', type=int, default=12)
    parser.add_argument('--visual-faiss-evidence-layer-sqlite', type=Path, default=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime/evidence_layer.sqlite'))
    parser.add_argument('--visual-faiss-evidence-layer-object-limit', type=int, default=8)
    parser.add_argument('--visual-faiss-evidence-layer-neighbor-limit', type=int, default=4)
    parser.add_argument('--visual-faiss-role-aware-object-limit', type=int, default=8)
    parser.add_argument('--visual-faiss-role-aware-context-limit', type=int, default=8)
    parser.add_argument('--visual-faiss-cross-sequence', action='store_true', help='Allow visually similar objects from other sequences in the FAISS evidence. By default, QA evidence is restricted to the current seq.')
    parser.add_argument('--frame-faiss-index-dir', type=Path, default=DEFAULT_FRAME_FAISS_INDEX_DIR)
    parser.add_argument('--frame-faiss-index-name', default=DEFAULT_FRAME_FAISS_INDEX_NAME)
    parser.add_argument('--frame-faiss-query-text-limit', type=int, default=6)
    parser.add_argument('--frame-faiss-top-k', type=int, default=12)
    parser.add_argument('--frame-faiss-search-k', type=int, default=80)
    parser.add_argument('--frame-faiss-min-score', type=float, default=0.0)
    parser.add_argument('--frame-faiss-cluster-max-frame-gap', type=int, default=30)
    parser.add_argument('--frame-faiss-cluster-max-sec-gap', type=float, default=None)
    parser.add_argument('--frame-faiss-max-clusters', type=int, default=8)
    parser.add_argument('--frame-faiss-max-lines', type=int, default=8)
    parser.add_argument('--frame-faiss-cross-sequence', action='store_true', help='Allow visually similar frames from other sequences in frame FAISS evidence. By default, QA evidence is restricted to the current seq.')
    parser.add_argument('--merged-frame-faiss-index-dir', type=Path, default=DEFAULT_MERGED_FRAME_FAISS_INDEX_DIR)
    parser.add_argument('--merged-frame-faiss-index-name', default=DEFAULT_MERGED_FRAME_FAISS_INDEX_NAME)
    parser.add_argument('--merged-frame-faiss-query-text-limit', type=int, default=6)
    parser.add_argument('--merged-frame-faiss-top-k', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-search-k', type=int, default=80)
    parser.add_argument('--merged-frame-faiss-min-score', type=float, default=0.0)
    parser.add_argument('--merged-frame-faiss-cluster-max-frame-gap', type=int, default=48)
    parser.add_argument('--merged-frame-faiss-cluster-max-sec-gap', type=float, default=None)
    parser.add_argument('--merged-frame-faiss-max-clusters', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-max-lines', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-cross-sequence', action='store_true', help='Allow visually similar merged-frame windows from other sequences. By default, QA evidence is restricted to the current seq.')
    parser.add_argument('--frame-window-faiss-window-size-sec', type=float, default=5.0)
    parser.add_argument('--frame-window-faiss-window-stride-sec', type=float, default=2.0)
    parser.add_argument('--frame-window-faiss-max-frames-per-window', type=int, default=8)
    parser.add_argument('--projection-cache', type=Path, default=None, help='Optional pickle cache for the canonical retrieval projection. Reused when the graph fingerprint matches.')
    parser.add_argument('--domain-profile', choices=available_domain_profiles(), default=DEFAULT_DOMAIN_PROFILE, help="Question/retrieval lexicon profile. The clean mainline defaults to 'generic' and does not use dataset-specific aliases.")
    parser.add_argument('--qa-unfilled-json', type=Path, default=None, help='Optional NaVQA qa_unfilled.json. Supplies question start/end time metadata without using answer labels.')
    parser.add_argument('--frame-time-index', type=Path, default=None, help='Optional frame-to-Unix-time index for rendering time/duration predictions from graph-selected frames. If omitted, the runner tries .scratch/navqa/seq<seq-id>/cam<cam>/every<stride>/frame_time_index.json.')
    parser.add_argument('--position-reference-json', type=Path, default=None, help='Optional NaVQA position reference produced by the evaluation adapter. Used only to materialize official-format labels in outputs; it is not passed to retrieval or graph QA.')
    parser.add_argument('--pose-timeline-json', type=Path, default=None, help='Optional ReMEmbR captions/pose timeline JSON. Used to render position predictions from graph-selected event timestamps, not from question UUIDs.')
    parser.add_argument('--dense-pose', type=Path, default=None, help='CODa poses/<seq>.txt for graph-time to ego-pose rendering. If omitted, the runner tries .scratch/navqa/seq<seq-id>/metadata/poses/<seq-id>.txt.')
    parser.add_argument('--timestamps', type=Path, default=None, help='CODa timestamps/<seq>.txt used with --dense-pose. If omitted, the runner tries .scratch/navqa/seq<seq-id>/metadata/timestamps/<seq-id>.txt.')
    parser.add_argument('--seconds-per-caption', type=float, default=3.0)
    parser.add_argument('--current-window-sec', type=float, default=30.0, help='Look-back window, in seconds, for current/right-now questions when qa_unfilled question time metadata is available.')
    parser.add_argument('--disable-question-window', action='store_true', help='Do not pass qa_unfilled start/end time metadata into MemoryQuery. The metadata will still be shown in prompts and output labels.')
    parser.add_argument('--text-judge-mode', choices=('none', 'live'), default='live', help="Evaluation-only LLM judge for text/descriptive answers. This is part of the NaVQA official-style metric, uses gold answers only after QA generation, and never feeds judge output back into graph retrieval or answering. Use 'none' only for offline diagnostics.")
    parser.add_argument('--output-json', type=Path, default=None)
    parser.add_argument('--output-csv', type=Path, default=None)
    parser.add_argument('--output-md', type=Path, default=None)
    return parser

def load_questions(csv_path: Path, seq_id: str) -> list[dict[str, str]]:
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get(FIELD_SEQ_ID, '')).strip() == str(seq_id)]
    return rows

def filter_questions_by_type(rows: list[dict[str, str]], *, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> list[dict[str, str]]:
    includes = _normalized_type_set(include_types)
    excludes = _normalized_type_set(exclude_types) or set()
    filtered: list[dict[str, str]] = []
    for row in rows:
        qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
        if includes is not None and qtype not in includes:
            continue
        if qtype in excludes:
            continue
        filtered.append(row)
    return filtered

def _normalized_type_set(values: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized

def has_position_questions(rows: list[dict[str, str]]) -> bool:
    return any((str(row.get(FIELD_TYPE, '')).strip().lower() == 'position' for row in rows))

def output_question(row: dict[str, str]) -> str:
    return str(row.get(FIELD_QUESTION, '')).strip()

def query_text_for_question(row: dict[str, str]) -> str:
    return output_question(row)

def load_question_ir_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = resolve_repo_input_path(path)
    payload = json.loads(resolved.read_text(encoding='utf-8-sig'))
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise ValueError(f'question IR records must be a list: {resolved}')
    by_uuid: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        uuid = str(record.get('uuid') or '').strip()
        ir = record.get('llm_question_ir')
        if uuid and isinstance(ir, dict):
            by_uuid[uuid] = ir
    return by_uuid

def question_ir_for_row(row: dict[str, str], question_ir_records: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if not question_ir_records:
        return None
    return question_ir_records.get(str(row.get(FIELD_UUID, '')).strip())

def query_text_from_question_ir(row: dict[str, str], ir: dict[str, Any] | None, *, include_evidence_needs: bool=False) -> str:
    if not isinstance(ir, dict):
        return query_text_for_question(row)
    parts = [output_question(row)]
    target_state = str(ir.get('target_state') or '').strip()
    if target_state:
        parts.append(target_state)
    targets = _string_list(ir.get('target_entities'))
    if targets:
        parts.extend(targets)
    target_aliases = _string_list(ir.get('target_entity_aliases'))
    if target_aliases:
        parts.extend(target_aliases)
    attribute = str(ir.get('attribute_dimension') or '').strip()
    if attribute:
        parts.append(attribute)
    for alias in _string_list(ir.get('attribute_dimension_aliases')):
        parts.append(alias)
    for constraint in _constraint_texts(ir.get('event_constraints'), prefix='event'):
        parts.append(constraint)
    for constraint in _constraint_texts(ir.get('event_constraint_aliases'), prefix='event'):
        parts.append(constraint)
    for constraint in _constraint_texts(ir.get('spatial_constraints'), prefix='spatial'):
        parts.append(constraint)
    for constraint in _constraint_texts(ir.get('spatial_constraint_aliases'), prefix='spatial'):
        parts.append(constraint)
    for constraint in _constraint_texts(ir.get('hard_constraints'), prefix='hard'):
        parts.append(constraint)
    for constraint in _constraint_texts(ir.get('hard_constraint_aliases'), prefix='hard'):
        parts.append(constraint)
    if include_evidence_needs:
        for text in _string_list(ir.get('evidence_requirements')):
            parts.append(f'evidence requirement {text}')
        for text in _string_list(ir.get('support_aliases')):
            parts.append(f'evidence requirement {text}')
        for text in _string_list(ir.get('graph_information_needs')):
            parts.append(f'graph information need {text}')
    return ' ; '.join(dict.fromkeys((part for part in parts if part.strip())))

def _constraint_texts(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            fragments: list[str] = []
            for key in ('kind', 'text', 'role', 'relation', 'operation', 'target_state', 'target', 'target_entities', 'must_bind_to'):
                raw = item.get(key)
                if isinstance(raw, list):
                    fragments.extend((str(part).strip() for part in raw if str(part).strip()))
                elif str(raw or '').strip():
                    fragments.append(str(raw).strip())
            if fragments:
                texts.append(f'{prefix} constraint ' + ' '.join(fragments))
        elif str(item or '').strip():
            texts.append(f'{prefix} constraint {str(item).strip()}')
    return texts

def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def answer_type_for_question(row: dict[str, str]) -> str | None:
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    return qtype or None

def official_answer(row: dict[str, str]) -> str:
    text = str(row.get(FIELD_TEXT_ANSWER, '')).strip()
    parsable = str(row.get(FIELD_PARSABLE_ANSWER, '')).strip()
    if text and parsable:
        return f'{text} (parsable: {parsable})'
    return text or parsable

def navqa_official_style_accuracy_summary(results: list[dict[str, Any]], *, temporal_threshold_minutes: float=NAVQA_OFFICIAL_TEMPORAL_THRESHOLD_MINUTES) -> dict[str, Any]:
    return _accuracy_summary(results, scorer=lambda item: score_navqa_official_style_prediction(item, temporal_threshold_minutes=temporal_threshold_minutes), contract='navqa_official_style_accuracy_v1', tolerance_minutes=temporal_threshold_minutes, note='NaVQA paper-style correctness proxy: temporal time/duration answers are correct within 2 minutes, binary answers require exact yes/no, and text answers use deterministic normalized containment unless an external LLM judge result is attached.', extra={'temporal_threshold_minutes': temporal_threshold_minutes, 'spatial_threshold_meters': 15.0, 'text_judging': 'llm_judge_when_available_else_normalized_containment_proxy', 'paper_metric_note': 'The NaVQA paper reports Overall Correctness by thresholding spatial L2 error at 15m and temporal L1 error at 2 minutes, with descriptive text answers judged by LLM. Current mainline excludes archived position questions.'})

def _accuracy_summary(results: list[dict[str, Any]], *, scorer: Any, contract: str, tolerance_minutes: float, note: str, extra: dict[str, Any] | None=None) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    status_counts: dict[str, int] = {}
    scored: list[dict[str, Any]] = []
    correct = 0
    for index, item in enumerate(results, start=1):
        score = scorer(item)
        score['index'] = index
        scored.append(score)
        qtype = str(item.get('question_type', '')).strip().lower() or 'unknown'
        bucket = by_type.setdefault(qtype, {'correct': 0, 'total': 0})
        bucket['total'] += 1
        if score['matched']:
            bucket['correct'] += 1
            correct += 1
        status = str(item.get('status', '')).strip() or 'unknown'
        status_counts[status] = status_counts.get(status, 0) + 1
    total = len(results)
    for bucket in by_type.values():
        bucket['accuracy_percent'] = round(bucket['correct'] / bucket['total'] * 100.0 if bucket['total'] else 0.0, 2)
    summary = {'contract': contract, 'tolerance_minutes': tolerance_minutes, 'correct': correct, 'total': total, 'accuracy_percent': round(correct / total * 100.0 if total else 0.0, 2), 'by_type': by_type, 'status_counts': status_counts, 'per_question': scored, 'note': note}
    if extra:
        summary.update(extra)
    if any((str(item.get('question_type', '')).strip().lower() == 'position' for item in results)):
        summary['position_prediction_contract'] = 'Position predictions are only scoreable when the predicted coordinate is rendered from graph-selected event time through an independent pose timeline. Question UUID position labels and official answer timestamps are evaluation-label inputs only.'
        summary['note'] = 'This evaluates the evaluation-layer official_format_prediction against materialized official labels. Position labels are rendered from the external pose reference only after graph evidence selects an event.'
    return summary

def score_navqa_official_style_prediction(item: dict[str, Any], *, temporal_threshold_minutes: float=NAVQA_OFFICIAL_TEMPORAL_THRESHOLD_MINUTES) -> dict[str, Any]:
    return _score_prediction_text(item, prediction=str(item.get('official_format_prediction') or item.get('predicted_answer') or '').strip(), tolerance_minutes=temporal_threshold_minutes, position_source=str(item.get('official_format_prediction_source') or ''), position_details=item.get('official_format_prediction_details', {}), method_prefix='navqa_official_style_', text_judgement=item.get('text_llm_judgement'))

def _score_prediction_text(item: dict[str, Any], *, prediction: str, tolerance_minutes: float, position_source: str, position_details: Any, method_prefix: str='', text_judgement: Any=None) -> dict[str, Any]:
    qtype = str(item.get('question_type', '')).strip().lower()
    label = item.get('official_evaluation_label', {})
    if not isinstance(label, dict):
        label = {}
    gold = str(item.get('official_evaluation_value') or label.get('display') or item.get('official_answer') or '').strip()
    if qtype == 'position':
        source = position_source
        details = position_details
        if not isinstance(details, dict):
            details = {}
        safe_graph_pose_prediction = source == 'graph_event_time_to_pose_timeline' and details.get('uses_official_answer_timestamp') is False and (details.get('uses_question_uuid_position_label') is False)
        if not safe_graph_pose_prediction:
            return _score_record(item, False, f'{method_prefix}invalid_position_prediction_source', gold, prediction, prediction_source=source)
        matched = _normalize_label_text(gold) == _normalize_label_text(prediction)
        return _score_record(item, matched, f'{method_prefix}exact_position', gold, prediction)
    if qtype in {'time', 'duration'}:
        gold_number = _number_from_text(gold)
        prediction_number = _number_from_text(prediction)
        if gold_number is None or prediction_number is None:
            return _score_record(item, False, f'{method_prefix}missing_numeric_value', gold, prediction)
        delta = abs(gold_number - prediction_number)
        return _score_record(item, delta <= tolerance_minutes, f'{method_prefix}numeric_delta_minutes={delta:.2f}', gold, prediction, delta_minutes=round(delta, 4))
    if qtype == 'binary':
        gold_value = _yes_no_value(str(label.get('value') or gold))
        prediction_value = _yes_no_value(prediction)
        return _score_record(item, gold_value is not None and gold_value == prediction_value, f"{method_prefix}predicted_binary={prediction_value or 'unknown'}", gold, prediction)
    if isinstance(text_judgement, dict):
        judged = text_judgement.get('correct')
        if isinstance(judged, bool):
            return _score_record(item, judged, f'{method_prefix}llm_text_judge', gold, prediction, judge_reason=str(text_judgement.get('reason') or ''), judge_extracted_answer=str(text_judgement.get('extracted_answer') or ''))
    gold_text = _normalize_label_text(str(label.get('value') or gold))
    prediction_text = _normalize_label_text(prediction)
    matched = bool(gold_text and prediction_text and (gold_text in prediction_text or prediction_text in gold_text))
    return _score_record(item, matched, f'{method_prefix}normalized_text_containment', gold, prediction)

def _score_record(item: dict[str, Any], matched: bool, method: str, gold: str, prediction: str, **extra: Any) -> dict[str, Any]:
    record = {'uuid': item.get('uuid', ''), 'question': item.get('question', ''), 'question_type': item.get('question_type', ''), 'matched': matched, 'method': method, 'gold': gold, 'prediction': prediction}
    record.update(extra)
    return record

def attach_text_llm_judgements(results: list[dict[str, Any]], *, config: GraphQAConfig) -> list[dict[str, Any]]:
    api_key = os.getenv(config.api_key_env_var)
    if not api_key:
        raise RuntimeError(f'Missing environment variable: {config.api_key_env_var}')
    judged: list[dict[str, Any]] = []
    for item in results:
        updated = dict(item)
        if str(item.get('question_type', '')).strip().lower() != 'text':
            judged.append(updated)
            continue
        updated['text_llm_judgement'] = judge_text_answer_with_llm(item, config=config, api_key=api_key)
        judged.append(updated)
    return judged

def judge_text_answer_with_llm(item: dict[str, Any], *, config: GraphQAConfig, api_key: str) -> dict[str, Any]:
    gold = str(item.get('official_evaluation_value') or item.get('official_answer') or '').strip()
    prediction = str(item.get('llm_predicted_answer_raw') or item.get('predicted_answer') or item.get('official_format_prediction') or '').strip()
    payload = _build_text_judge_payload(question=str(item.get('question') or ''), gold_answer=gold, predicted_answer=prediction, config=config)
    try:
        raw_response = _post_openai_compatible_payload(payload, api_key=api_key, config=config)
        content = _extract_openai_compatible_content(raw_response)
        parsed = _parse_json_object_from_text(content or '')
        judgement = _normalize_text_judge_result(parsed)
        judgement.update({'provider': config.provider, 'model': config.model, 'mode': 'live', 'response_metadata': _openai_compatible_response_metadata(raw_response)})
        return judgement
    except LLM_REQUEST_EXCEPTIONS as exc:
        return {'correct': False, 'reason': f'text judge request failed: {exc}', 'extracted_answer': '', 'provider': config.provider, 'model': config.model, 'mode': 'live_error', 'error': str(exc)}

def _build_text_judge_payload(*, question: str, gold_answer: str, predicted_answer: str, config: GraphQAConfig) -> dict[str, Any]:
    prompt_payload = {'task': 'navqa_text_answer_judge', 'question': question, 'gold_answer': gold_answer, 'predicted_answer': predicted_answer, 'instructions': ['Judge whether the predicted answer is semantically correct for the question.', 'Accept paraphrases, synonyms, and concise answers that preserve the key meaning.', 'Reject answers that add conflicting facts or answer a different entity/event.', 'Use the gold answer only for evaluation. Do not infer from outside knowledge.', 'Return strict JSON with keys: correct, extracted_answer, reason.']}
    payload: dict[str, Any] = {'model': config.model, 'temperature': 0.0, 'max_tokens': min(int(config.max_tokens), 300), 'messages': [{'role': 'system', 'content': 'You are an evaluation judge for NaVQA descriptive text answers. Return only strict JSON.'}, {'role': 'user', 'content': json.dumps(prompt_payload, ensure_ascii=False, indent=2)}]}
    apply_reasoning_options(payload, reasoning_effort=config.reasoning_effort, thinking_enabled=config.thinking_enabled)
    return payload

def _post_openai_compatible_payload(payload: dict[str, Any], *, api_key: str, config: GraphQAConfig) -> dict[str, Any]:
    endpoint = urljoin(config.api_base_url.rstrip('/') + '/', config.endpoint_path.lstrip('/'))
    request = Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36', 'Accept': 'application/json'}, method='POST')
    with urlopen(request, timeout=config.timeout_sec) as response:
        raw_response = json.loads(response.read().decode('utf-8'))
    if not isinstance(raw_response, dict):
        raise json.JSONDecodeError('Response JSON is not an object', str(raw_response), 0)
    return raw_response

def _extract_openai_compatible_content(raw_response: dict[str, Any]) -> str | None:
    choices = raw_response.get('choices')
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get('message', {})
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        merged = ''.join((str(item.get('text') or '') for item in content if isinstance(item, dict))).strip()
        return merged or None
    return None

def _parse_json_object_from_text(content: str) -> dict[str, Any] | None:
    stripped = _strip_code_fences(str(content or '').strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_first_json_object(stripped)
        if candidate is None:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None

def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith('```'):
        return stripped
    stripped = re.sub('^```(?:json)?\\s*', '', stripped, flags=re.I)
    stripped = re.sub('\\s*```$', '', stripped)
    return stripped.strip()

def _extract_first_json_object(content: str) -> str | None:
    start = content.find('{')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return content[start:index + 1]
    return None

def _normalize_text_judge_result(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {'correct': False, 'reason': 'text judge response did not contain a JSON object', 'extracted_answer': ''}
    correct = parsed.get('correct')
    if not isinstance(correct, bool):
        correct = str(correct).strip().lower() in {'true', 'yes', 'correct', '1'}
    return {'correct': bool(correct), 'reason': str(parsed.get('reason') or '').strip(), 'extracted_answer': str(parsed.get('extracted_answer') or '').strip()}

def _openai_compatible_response_metadata(raw_response: dict[str, Any]) -> dict[str, Any]:
    choice = {}
    choices = raw_response.get('choices')
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
    return {'id': raw_response.get('id'), 'model': raw_response.get('model'), 'finish_reason': choice.get('finish_reason'), 'usage': raw_response.get('usage', {})}

def position_prediction_diagnostics(item: dict[str, Any]) -> dict[str, Any]:
    if str(item.get('question_type', '')).strip().lower() != 'position':
        return {}
    details = item.get('official_format_prediction_details', {})
    if not isinstance(details, dict):
        details = {}
    label = item.get('official_evaluation_label', {})
    if not isinstance(label, dict):
        label = {}
    return {'prediction_source': item.get('official_format_prediction_source', ''), 'prediction_semantics': item.get('official_format_prediction_semantics', ''), 'predicted_position': _position_from_text(str(item.get('official_format_prediction') or '')), 'gold_position': label.get('value'), 'graph_event_timestamp_sec': details.get('event_timestamp_sec'), 'graph_event_timestamp_iso': details.get('event_timestamp_iso'), 'graph_event_frame_index': details.get('event_frame_index'), 'graph_event_scene_id': details.get('event_scene_id'), 'graph_event_source': details.get('event_source'), 'pose_window_id': details.get('pose_window_id'), 'pose_window_start_time': details.get('pose_window_start_time'), 'pose_window_end_time': details.get('pose_window_end_time'), 'pose_lookup_delta_seconds': details.get('pose_lookup_delta_seconds'), 'uses_official_answer_timestamp': details.get('uses_official_answer_timestamp'), 'uses_question_uuid_position_label': details.get('uses_question_uuid_position_label')}

def _position_from_text(value: str) -> list[float] | None:
    numbers = re.findall('-?\\d+(?:\\.\\d+)?', value)
    if len(numbers) != 3:
        return None
    return [float(item) for item in numbers]

def _normalize_label_text(value: str) -> str:
    return re.sub('\\s+', ' ', str(value or '').strip().lower())

def _number_from_text(value: str) -> float | None:
    match = re.search('-?\\d+(?:\\.\\d+)?', str(value or ''))
    if not match:
        return None
    return float(match.group(0))

def _duration_minutes_from_text(value: str) -> float | None:
    candidates = _duration_minute_candidates_from_text(value)
    return candidates[0] if candidates else None

def _duration_minute_candidates_from_text(value: str) -> list[float]:
    text = str(value or '').strip()
    if not text:
        return []
    candidates: list[float] = []
    for match in re.finditer('\\b(-?\\d+(?:\\.\\d+)?)\\s*(?:-|to|through|and)\\s*(-?\\d+(?:\\.\\d+)?)\\s*(?:minutes?|mins?|m)\\b', text, flags=re.I):
        candidates.extend([float(match.group(1)), float(match.group(2))])
    for match in re.finditer('\\b(-?\\d+(?:\\.\\d+)?)\\s*(?:minutes?|mins?|m)\\b(?:\\s*(?:and)?\\s*(-?\\d+(?:\\.\\d+)?)\\s*(?:seconds?|secs?|s)\\b)?', text, flags=re.I):
        minutes = float(match.group(1))
        seconds = float(match.group(2)) if match.group(2) is not None else 0.0
        candidates.append(minutes + seconds / 60.0)
    for match in re.finditer('\\b(\\d+):([0-5]\\d)\\b', text):
        candidates.append(float(match.group(1)) + float(match.group(2)) / 60.0)
    for match in re.finditer('\\b(-?\\d+(?:\\.\\d+)?)\\s*(?:seconds?|secs?|s)\\b', text, flags=re.I):
        if _looks_like_absolute_unix_timestamp(match.group(1)):
            continue
        candidates.append(float(match.group(1)) / 60.0)
    rounded: list[float] = []
    seen: set[float] = set()
    for candidate in candidates:
        value_key = round(candidate, 4)
        if value_key in seen:
            continue
        rounded.append(value_key)
        seen.add(value_key)
    return rounded

def _closest_minute_candidate(target: float | None, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    if target is None:
        return candidates[0]
    return min(candidates, key=lambda candidate: abs(candidate - target))

def _minute_candidate_delta(target: float, candidates: list[float]) -> float:
    if not candidates:
        return float('inf')
    return min((abs(target - candidate) for candidate in candidates))

def _looks_like_absolute_unix_timestamp(value: str) -> bool:
    try:
        number = abs(float(value))
    except ValueError:
        return False
    return number >= 1000000000

def _yes_no_value(value: str) -> str | None:
    rendered = _analyze_binary_answer_text(value)
    return str(rendered['value']) if rendered['clear'] else None

def _analyze_binary_answer_text(value: str) -> dict[str, Any]:
    text = str(value or '').strip()
    lowered = text.lower()
    yes_matches = list(re.finditer('\\b(?:yes|yeah|yep|affirmative)\\b', lowered))
    no_matches = list(re.finditer('\\b(?:no|nope|negative)\\b', lowered))
    if yes_matches and no_matches:
        return {'value': None, 'clear': False, 'reason': 'conflicting_yes_no_cues', 'yes_cue_count': len(yes_matches), 'no_cue_count': len(no_matches)}
    if yes_matches:
        return {'value': 'yes', 'clear': True, 'reason': 'single_yes_no_cue', 'yes_cue_count': len(yes_matches), 'no_cue_count': 0}
    if no_matches:
        return {'value': 'no', 'clear': True, 'reason': 'single_yes_no_cue', 'yes_cue_count': 0, 'no_cue_count': len(no_matches)}
    return {'value': None, 'clear': False, 'reason': 'missing_explicit_yes_no_cue', 'yes_cue_count': 0, 'no_cue_count': 0}

def load_position_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = resolve_repo_input_path(path)
    payload = json.loads(resolved.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError(f'position reference must contain a JSON object: {resolved}')
    return payload

def load_question_time_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = resolve_repo_input_path(path)
    payload = json.loads(resolved.read_text(encoding='utf-8-sig'))
    items = payload.get('data', [])
    if not isinstance(items, list):
        raise ValueError(f'qa_unfilled data must be a list: {resolved}')
    by_uuid: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get('id', '')).strip()
        if not raw_id:
            continue
        by_uuid[raw_id] = item
        if raw_id.isdigit():
            by_uuid.setdefault(f'Q{raw_id}', item)
    return by_uuid

def load_frame_time_index(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = resolve_repo_input_path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding='utf-8-sig'))
    frames = payload.get('frames', []) if isinstance(payload, dict) else []
    by_frame: dict[int, dict[str, Any]] = {}
    if isinstance(frames, list):
        for item in frames:
            if not isinstance(item, dict):
                continue
            frame_index = _optional_int(item.get('graph_frame_index'))
            timestamp_unix = _float_or_none(item.get('timestamp_unix'))
            if frame_index is None or timestamp_unix is None:
                continue
            by_frame[frame_index] = {'frame_index': frame_index, 'timestamp_unix': timestamp_unix, 'timestamp_iso': item.get('timestamp_iso'), 'source_frame': item.get('source_frame'), 'sampled_index': item.get('sampled_index'), 'elapsed_sec_from_sequence_start': _float_or_none(item.get('elapsed_sec_from_sequence_start'))}
    ordered_frames = sorted(by_frame)
    return {'source_path': str(resolved), 'sequence_id': payload.get('sequence_id') if isinstance(payload, dict) else None, 'sequence_start_unix': payload.get('sequence_start_unix') if isinstance(payload, dict) else None, 'sequence_end_unix': payload.get('sequence_end_unix') if isinstance(payload, dict) else None, 'sequence_duration_sec': payload.get('sequence_duration_sec') if isinstance(payload, dict) else None, 'by_frame': by_frame, 'ordered_frames': ordered_frames}

def default_frame_time_index_path(seq_id: str, *, cam: str='0', stride: int | None=10, layout_root: Path=(REPO_ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa')) -> Path:
    return default_navqa_sequence_layout(seq_id=seq_id, cam=cam, stride=stride, layout_root=layout_root).frame_time_index_json

def graph_time_bounds(graph: CanonicalTemporalMemoryGraph) -> tuple[float, float] | None:
    values: list[float] = []
    node_groups = (graph.scene_nodes, graph.entity_nodes, graph.stable_identity_nodes, graph.episode_nodes, graph.transition_event_nodes, graph.relation_observed_samples)
    for nodes in node_groups:
        for node in nodes:
            time_range = getattr(node, 'time_range', None)
            if time_range is None:
                continue
            for value in (time_range.start_ts, time_range.end_ts):
                number = _float_or_none(value)
                if number is not None:
                    values.append(number)
    if not values:
        return None
    return (min(values), max(values))

def build_memory_query_for_question(row: dict[str, str], *, retrieval_limit: int, question_ir: dict[str, Any] | None=None, question_route: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None, graph_bounds: tuple[float, float] | None=None, frame_time_index: dict[str, Any] | None=None, current_window_sec: float=30.0, use_question_window: bool=True, domain_profile: str | None=None, include_ir_evidence_needs_in_query: bool=False) -> tuple[MemoryQuery, dict[str, Any]]:
    start_ts: float | None = None
    end_ts: float | None = None
    window_info = question_retrieval_window(row, question_time_metadata or {}, graph_bounds=graph_bounds, frame_time_index=frame_time_index, current_window_sec=current_window_sec, use_question_window=use_question_window, domain_profile=domain_profile)
    if window_info.get('applied'):
        start_ts = _float_or_none(window_info.get('start_ts'))
        end_ts = _float_or_none(window_info.get('end_ts'))
    metadata: dict[str, object] = {}
    if question_ir:
        metadata['question_ir'] = question_ir
    if question_route:
        metadata['question_route'] = question_route
    return (MemoryQuery(query_text=query_text_from_question_ir(row, question_ir, include_evidence_needs=include_ir_evidence_needs_in_query), answer_type=answer_type_for_question(row), metadata=metadata, start_ts=start_ts, end_ts=end_ts, limit=retrieval_limit), window_info)

def question_retrieval_window(row: dict[str, str], metadata: dict[str, dict[str, Any]], *, graph_bounds: tuple[float, float] | None=None, frame_time_index: dict[str, Any] | None=None, current_window_sec: float=30.0, use_question_window: bool=True, domain_profile: str | None=None) -> dict[str, Any]:
    if not use_question_window:
        return {'applied': False, 'reason': 'disabled', 'current_window_required': False}
    item = metadata.get(str(row.get(FIELD_UUID, '')).strip())
    if not item:
        return {'applied': False, 'reason': 'missing_question_time_metadata', 'current_window_required': False}
    raw_start = _float_or_none(item.get('start_time'))
    raw_end = _float_or_none(item.get('end_time'))
    if raw_start is None and raw_end is None:
        return {'applied': False, 'reason': 'missing_start_end_time', 'current_window_required': False}
    if raw_start is None:
        raw_start = raw_end
    if raw_end is None:
        raw_end = raw_start
    if raw_start is None or raw_end is None:
        return {'applied': False, 'reason': 'invalid_start_end_time', 'current_window_required': False}
    if raw_start > raw_end:
        raw_start, raw_end = (raw_end, raw_start)
    input_start_time = raw_start
    input_end_time = raw_end
    question_plan = compile_question(output_question(row), answer_type=str(row.get(FIELD_TYPE, '')).strip().lower() or None, category=str(row.get(FIELD_CATEGORY, '')).strip() or None, domain_profile=domain_profile)
    current_required = question_plan.current_window_required
    if not _time_scales_compatible(graph_bounds, raw_start, raw_end):
        mapped = _map_unix_window_to_graph_time(raw_start, raw_end, frame_time_index=frame_time_index, graph_bounds=graph_bounds)
        if mapped is None:
            return {'applied': False, 'reason': 'time_scale_mismatch', 'current_window_required': current_required, 'raw_start_time': raw_start, 'raw_end_time': raw_end, 'graph_bounds': list(graph_bounds) if graph_bounds else None}
        raw_start, raw_end = (mapped['start_ts'], mapped['end_ts'])
        mapped_window_info = mapped
    else:
        mapped_window_info = None
    if current_required:
        lookback = max(0.0, current_window_sec)
        if mapped_window_info is not None:
            graph_units_per_second = _float_or_none(mapped_window_info.get('graph_units_per_second'))
            if graph_units_per_second is not None and graph_units_per_second > 0:
                lookback *= graph_units_per_second
        window_start = max(raw_start, raw_end - lookback)
        window_reason = 'current_question_end_window'
    else:
        window_start = raw_start
        window_reason = 'question_memory_window'
    window_end = raw_end
    if graph_bounds is not None:
        graph_start, graph_end = graph_bounds
        if window_end < graph_start or window_start > graph_end:
            return {'applied': False, 'reason': 'outside_graph_time_bounds', 'current_window_required': current_required, 'raw_start_time': input_start_time, 'raw_end_time': input_end_time, 'graph_bounds': [graph_start, graph_end]}
        window_start = max(window_start, graph_start)
        window_end = min(window_end, graph_end)
    return {'applied': True, 'reason': window_reason, 'current_window_required': current_required, 'start_ts': window_start, 'end_ts': window_end, 'raw_start_time': input_start_time, 'raw_end_time': input_end_time, 'graph_bounds': list(graph_bounds) if graph_bounds else None, 'mapped_from_unix_time': mapped_window_info}

def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _time_scales_compatible(graph_bounds: tuple[float, float] | None, start_time: float, end_time: float) -> bool:
    if graph_bounds is None:
        return True
    graph_start, graph_end = graph_bounds
    graph_is_absolute = _looks_like_unix_time(graph_start) or _looks_like_unix_time(graph_end)
    query_is_absolute = _looks_like_unix_time(start_time) or _looks_like_unix_time(end_time)
    return graph_is_absolute == query_is_absolute

def _map_unix_window_to_graph_time(start_time: float, end_time: float, *, frame_time_index: dict[str, Any] | None, graph_bounds: tuple[float, float] | None=None) -> dict[str, Any] | None:
    if not (_looks_like_unix_time(start_time) or _looks_like_unix_time(end_time)):
        return None
    start_frame_time = _lookup_unix_time_in_frame_index(frame_time_index, start_time)
    end_frame_time = _lookup_unix_time_in_frame_index(frame_time_index, end_time)
    if start_frame_time is None or end_frame_time is None:
        return None
    use_elapsed_seconds = _graph_time_prefers_elapsed_seconds(graph_bounds, frame_time_index=frame_time_index)
    start_elapsed = _float_or_none(start_frame_time.get('elapsed_sec_from_sequence_start'))
    end_elapsed = _float_or_none(end_frame_time.get('elapsed_sec_from_sequence_start'))
    if use_elapsed_seconds and start_elapsed is not None and (end_elapsed is not None):
        start_ts = start_elapsed
        end_ts = end_elapsed
        graph_time_unit = 'elapsed_sec_from_sequence_start'
    else:
        start_ts = float(start_frame_time['frame_index'])
        end_ts = float(end_frame_time['frame_index'])
        graph_time_unit = 'graph_frame_index'
    if start_ts > end_ts:
        start_ts, end_ts = (end_ts, start_ts)
    elapsed_seconds = abs(float(end_frame_time['timestamp_unix']) - float(start_frame_time['timestamp_unix']))
    graph_units = abs(end_ts - start_ts)
    graph_units_per_second = graph_units / elapsed_seconds if elapsed_seconds > 0 else None
    return {'start_ts': start_ts, 'end_ts': end_ts, 'raw_start_time_unix': start_time, 'raw_end_time_unix': end_time, 'start_frame_index': start_frame_time['frame_index'], 'end_frame_index': end_frame_time['frame_index'], 'start_elapsed_sec_from_sequence_start': start_elapsed, 'end_elapsed_sec_from_sequence_start': end_elapsed, 'start_timestamp_unix': start_frame_time['timestamp_unix'], 'end_timestamp_unix': end_frame_time['timestamp_unix'], 'graph_units_per_second': graph_units_per_second, 'graph_time_unit': graph_time_unit, 'source': 'frame_time_index', 'frame_time_index_source': frame_time_index.get('source_path') if isinstance(frame_time_index, dict) else None}

def _graph_time_prefers_elapsed_seconds(graph_bounds: tuple[float, float] | None, *, frame_time_index: dict[str, Any] | None) -> bool:
    if graph_bounds is None or not isinstance(frame_time_index, dict):
        return False
    graph_start, graph_end = graph_bounds
    if _looks_like_unix_time(graph_start) or _looks_like_unix_time(graph_end):
        return False
    sequence_duration = _float_or_none(frame_time_index.get('sequence_duration_sec'))
    ordered = frame_time_index.get('ordered_frames', [])
    if sequence_duration is None or sequence_duration <= 0:
        return False
    if not isinstance(ordered, list) or not ordered:
        return False
    max_frame = max((float(frame) for frame in ordered if isinstance(frame, (int, float))), default=0.0)
    if max_frame <= 0:
        return False
    elapsed_gap = abs(float(graph_end) - sequence_duration)
    frame_gap = abs(float(graph_end) - max_frame)
    tolerance = max(10.0, sequence_duration * 0.08)
    return elapsed_gap <= tolerance and elapsed_gap < frame_gap

def _lookup_unix_time_in_frame_index(frame_time_index: dict[str, Any] | None, timestamp_unix: float) -> dict[str, Any] | None:
    if not isinstance(frame_time_index, dict):
        return None
    by_frame = frame_time_index.get('by_frame', {})
    ordered = frame_time_index.get('ordered_frames', [])
    if not isinstance(by_frame, dict) or not isinstance(ordered, list) or (not ordered):
        return None
    nearest_item: dict[str, Any] | None = None
    nearest_delta = float('inf')
    for frame in ordered:
        item = by_frame.get(frame)
        if not isinstance(item, dict):
            continue
        item_time = _float_or_none(item.get('timestamp_unix'))
        if item_time is None:
            continue
        delta = abs(item_time - timestamp_unix)
        if delta < nearest_delta:
            nearest_delta = delta
            nearest_item = item
    if nearest_item is None:
        return None
    frame_index = _optional_int(nearest_item.get('frame_index'))
    timestamp = _float_or_none(nearest_item.get('timestamp_unix'))
    if frame_index is None or timestamp is None:
        return None
    return {**nearest_item, 'frame_index': frame_index, 'timestamp_unix': timestamp, 'delta_seconds': round(nearest_delta, 3)}

def _looks_like_unix_time(value: float) -> bool:
    return abs(value) >= 1000000000

def _iso_from_unix(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(value)).isoformat(timespec='milliseconds')
    except (TypeError, ValueError, OSError):
        return None

def question_time_context(row: dict[str, str], metadata: dict[str, dict[str, Any]]) -> str:
    item = metadata.get(str(row.get(FIELD_UUID, '')).strip())
    if not item:
        return ''
    start_time = item.get('start_time')
    end_time = item.get('end_time')
    parts = ['Question time metadata, not answer evidence:', f'current_time_unix={end_time}']
    end_iso = _iso_from_unix(end_time)
    if end_iso:
        parts.append(f'current_time_iso={end_iso}')
    if start_time is not None:
        parts.append(f'question_window_start_unix={start_time}')
        start_iso = _iso_from_unix(start_time)
        if start_iso:
            parts.append(f'question_window_start_iso={start_iso}')
    parts.append("For '[minutes] ago' answers, first choose the graph-backed event evidence; then use graph time adaptation candidates when available instead of manually inventing or recalculating relative-time wording.")
    return '\n'.join(parts)

def question_adaptation_context(row: dict[str, str]) -> str:
    from vlkg.datasets.navqa.qa.official_qa import question_adaptation_context as _context
    return _context(row)

def official_answer_format_instruction(row: dict[str, str]) -> str:
    from vlkg.datasets.navqa.qa.official_qa import official_answer_format_instruction as _instruction
    return _instruction(row)

def run_question(graph: CanonicalTemporalMemoryGraph, row: dict[str, str], *, answer_mode: str, config: str | None, retrieval_limit: int, question_ir_records: dict[str, dict[str, Any]] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None, position_labels: dict[str, list[float]] | None=None, position_windows: list[PositionWindow] | None=None, frame_time_index: dict[str, Any] | None=None, projection_cache_path: Path | None=None, retrieval_session: CanonicalHybridRetrievalSession | None=None, graph_bounds: tuple[float, float] | None=None, current_window_sec: float=30.0, use_question_window: bool=True, domain_profile: str | None=None, graph_qa_config_overrides: dict[str, object] | None=None, include_ir_evidence_needs_in_query: bool=False, synonym_lexicon: SynonymLexicon | None=None, experimental_question_router: bool=False, experimental_question_router_hybrid_backbone: bool=False, experimental_question_router_specialized_backbone: bool=False, experimental_question_router_mainline: bool=False, visual_faiss_grounder: VisualFaissGrounder | None=None, visual_faiss_config: VisualFaissConfig | None=None, visual_faiss_focus_seed_evidence: bool=False, visual_faiss_query_recall: bool=False, visual_faiss_fused_evidence: bool=False, visual_faiss_graph_anchors: bool=False, visual_faiss_ger_fusion: bool=False, visual_faiss_ger_binding: bool=False, visual_faiss_frame_context: bool=False, visual_faiss_structural_binding: bool=False, visual_faiss_attribute_owner: bool=False, visual_faiss_fragment_bridge: bool=False, visual_faiss_object_local_facts: bool=False, visual_faiss_local_subgraph: bool=False, visual_faiss_local_subgraph_anchor_only: bool=False, visual_faiss_balanced_graph_visual: bool=False, visual_faiss_unified_candidates: bool=False, visual_faiss_evidence_layer: bool=False, visual_faiss_evidence_layer_constraint_binding: bool=False, visual_faiss_role_aware_evidence: bool=False, visual_faiss_role_aware_aux: bool=False, visual_faiss_role_aware_compression: bool=False, visual_faiss_entity_pack: bool=False, frame_faiss_grounder: FrameFaissGrounder | None=None, frame_faiss_config: FrameFaissConfig | None=None, frame_faiss_query_recall: bool=False, frame_faiss_fused_evidence: bool=False, frame_faiss_graph_anchors: bool=False, merged_frame_faiss_grounder: MergedFrameFaissGrounder | None=None, merged_frame_faiss_config: MergedFrameFaissConfig | None=None, merged_frame_faiss_query_recall: bool=False, merged_frame_faiss_fused_evidence: bool=False, merged_frame_faiss_graph_anchors: bool=False, frame_window_faiss_grounder: FrameWindowFaissGrounder | None=None, frame_window_faiss_config: FrameWindowFaissConfig | None=None, frame_window_faiss_query_recall: bool=False, frame_window_faiss_graph_anchors: bool=False) -> dict[str, Any]:
    question = output_question(row)
    question_ir = question_ir_for_row(row, question_ir_records)
    has_llm_question_ir = question_ir is not None
    original_question_ir = question_ir
    question_ir = expand_question_ir_with_synonyms(question_ir, synonym_lexicon, question=question)
    question_route = None
    if experimental_question_router:
        question_route = route_question(question, official_question_type=str(row.get(FIELD_TYPE, '')).strip().lower() or None, question_ir=question_ir, domain_profile=domain_profile, category=str(row.get(FIELD_CATEGORY, '')).strip() or None)
        question_ir = attach_question_route_to_ir(question_ir, question_route)
    time_context = question_time_context(row, question_time_metadata or {})
    adaptation_context = question_adaptation_context(row)
    format_instruction = official_answer_format_instruction(row)
    memory_query, retrieval_window = build_memory_query_for_question(row, retrieval_limit=retrieval_limit, question_ir=question_ir, question_route=question_route.to_dict() if question_route is not None else None, question_time_metadata=question_time_metadata, graph_bounds=graph_bounds, frame_time_index=frame_time_index, current_window_sec=current_window_sec, use_question_window=use_question_window, domain_profile=domain_profile, include_ir_evidence_needs_in_query=include_ir_evidence_needs_in_query)
    answer_question_text = f'{question}\n\n{time_context}\n\n{adaptation_context}\n\n{format_instruction}\n\nAnswer using only the supplied graph evidence. Treat ordered temporal chains, scene boundaries, focus frames, entity matches, and relation evidence as valid graph evidence. For temporal questions, infer a reasonable interval or boundary from the ordered evidence when the graph supports it, and state uncertainty when the boundary is approximate. Only say that the graph evidence is insufficient when neither direct evidence nor ordered temporal graph evidence supports the requested location, timestamp, duration, or object.'
    payload = answer_canonical_graph_query(graph, memory_query, question=answer_question_text, retrieval_mode='temporal', answer_mode='graph_only' if visual_faiss_grounder is not None and (visual_faiss_focus_seed_evidence or visual_faiss_query_recall) or (frame_faiss_grounder is not None and frame_faiss_query_recall) or (merged_frame_faiss_grounder is not None and merged_frame_faiss_query_recall) or (frame_window_faiss_grounder is not None and frame_window_faiss_query_recall) else answer_mode, config_path=config, limit=retrieval_limit, projection_cache_path=projection_cache_path, retrieval_session=retrieval_session, domain_profile=domain_profile, graph_qa_config_overrides=None if visual_faiss_grounder is not None and (visual_faiss_focus_seed_evidence or visual_faiss_query_recall) or (frame_faiss_grounder is not None and frame_faiss_query_recall) or (merged_frame_faiss_grounder is not None and merged_frame_faiss_query_recall) or (frame_window_faiss_grounder is not None and frame_window_faiss_query_recall) else graph_qa_config_overrides)
    graph_answer = payload.get('graph_answer', {})
    if visual_faiss_frame_context:
        visual_faiss_config = replace(visual_faiss_config or VisualFaissConfig(), frame_context_enabled=True)
    visual_faiss_config = visual_faiss_config or VisualFaissConfig()
    route_uses_specialized_backbone = experimental_question_router_specialized_backbone or experimental_question_router_mainline
    route_uses_visual_backbone = experimental_question_router_hybrid_backbone or route_uses_specialized_backbone
    if route_uses_visual_backbone:
        visual_faiss_config = hybrid_visual_faiss_config(visual_faiss_config, question_ir=question_ir or {}, specialized=route_uses_specialized_backbone)
    if visual_faiss_grounder is not None and isinstance(graph_answer, dict) and (visual_faiss_focus_seed_evidence or visual_faiss_query_recall):
        if visual_faiss_focus_seed_evidence:
            visual_payload = visual_faiss_grounder.build_question_evidence(seq_id=str(row.get(FIELD_SEQ_ID, '')).strip(), question=question, graph_answer=graph_answer, question_ir=question_ir or {}, config=visual_faiss_config or VisualFaissConfig())
            if visual_faiss_fused_evidence:
                fuse_visual_faiss_evidence(graph_answer, visual_payload)
            elif visual_faiss_graph_anchors:
                expand_visual_faiss_graph_evidence(graph_answer, visual_payload, graph=graph)
            elif visual_faiss_evidence_layer:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_faiss_evidence_layer(graph_answer, visual_payload, evidence_layer_sqlite=config_values.evidence_layer_sqlite, object_limit=config_values.evidence_layer_object_limit, neighbor_limit=config_values.evidence_layer_neighbor_limit, question=question, question_ir=question_ir or {}, constraint_binding=visual_faiss_evidence_layer_constraint_binding)
            elif visual_faiss_role_aware_evidence:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_faiss_role_aware_evidence(graph_answer, visual_payload, object_limit=config_values.role_aware_object_limit, context_limit=config_values.role_aware_context_limit, question=question, question_ir=question_ir or {})
            elif route_uses_visual_backbone:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_question_route_hybrid_backbone(graph_answer, visual_payload, graph=graph, config=config_values, question=question, question_ir=question_ir or {}, semantic_text_encoder=visual_faiss_grounder._encode_texts, specialized=route_uses_specialized_backbone, retrieval_window=retrieval_window)
            elif visual_faiss_entity_pack:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_entity_candidate_packs(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, neighbor_limit=config_values.local_subgraph_neighbor_limit, question=question, question_ir=question_ir or {}, semantic_align_enabled=config_values.entity_pack_semantic_align_enabled, semantic_gated_role_enabled=config_values.entity_pack_semantic_gated_role_enabled, score_contract_enabled=config_values.entity_pack_score_contract_enabled, retrieval_window=retrieval_window, semantic_text_encoder=visual_faiss_grounder._encode_texts)
            elif visual_faiss_ger_fusion:
                config_values = visual_faiss_config or VisualFaissConfig()
                if visual_faiss_unified_candidates:
                    fuse_visual_faiss_unified_candidates(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, context_radius=config_values.ger_context_radius, context_limit=config_values.ger_context_limit, question=question, question_ir=question_ir or {})
                else:
                    fuse_visual_faiss_ger_evidence(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, context_radius=config_values.ger_context_radius, context_limit=config_values.ger_context_limit, binding_enabled=visual_faiss_ger_binding, structural_binding_enabled=visual_faiss_structural_binding, attribute_owner_enabled=visual_faiss_attribute_owner, fragment_bridge_enabled=visual_faiss_fragment_bridge, object_local_facts_enabled=visual_faiss_object_local_facts, local_subgraph_enabled=visual_faiss_local_subgraph, local_subgraph_anchor_only=visual_faiss_local_subgraph_anchor_only, balanced_graph_visual_evidence=visual_faiss_balanced_graph_visual, role_aware_compression_enabled=visual_faiss_role_aware_compression, question=question, question_ir=question_ir or {})
                if visual_faiss_role_aware_aux:
                    config_values = visual_faiss_config or VisualFaissConfig()
                    attach_visual_faiss_role_aware_auxiliary_evidence(graph_answer, visual_payload, object_limit=config_values.role_aware_object_limit, context_limit=config_values.role_aware_context_limit, question=question, question_ir=question_ir or {})
            else:
                attach_visual_faiss_evidence(graph_answer, visual_payload)
        if visual_faiss_query_recall:
            visual_payload = visual_faiss_grounder.build_query_recall_evidence(seq_id=str(row.get(FIELD_SEQ_ID, '')).strip(), question=question, graph_answer=graph_answer, question_ir=question_ir or {}, config=visual_faiss_config or VisualFaissConfig(), graph=graph)
            if visual_faiss_fused_evidence:
                fuse_visual_faiss_evidence(graph_answer, visual_payload)
            elif visual_faiss_graph_anchors:
                expand_visual_faiss_graph_evidence(graph_answer, visual_payload, graph=graph)
            elif visual_faiss_evidence_layer:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_faiss_evidence_layer(graph_answer, visual_payload, evidence_layer_sqlite=config_values.evidence_layer_sqlite, object_limit=config_values.evidence_layer_object_limit, neighbor_limit=config_values.evidence_layer_neighbor_limit, question=question, question_ir=question_ir or {}, constraint_binding=visual_faiss_evidence_layer_constraint_binding)
            elif visual_faiss_role_aware_evidence:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_faiss_role_aware_evidence(graph_answer, visual_payload, object_limit=config_values.role_aware_object_limit, context_limit=config_values.role_aware_context_limit, question=question, question_ir=question_ir or {})
            elif route_uses_visual_backbone:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_question_route_hybrid_backbone(graph_answer, visual_payload, graph=graph, config=config_values, question=question, question_ir=question_ir or {}, semantic_text_encoder=visual_faiss_grounder._encode_texts, specialized=route_uses_specialized_backbone, retrieval_window=retrieval_window)
            elif visual_faiss_entity_pack:
                config_values = visual_faiss_config or VisualFaissConfig()
                fuse_visual_entity_candidate_packs(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, neighbor_limit=config_values.local_subgraph_neighbor_limit, question=question, question_ir=question_ir or {}, semantic_align_enabled=config_values.entity_pack_semantic_align_enabled, semantic_gated_role_enabled=config_values.entity_pack_semantic_gated_role_enabled, score_contract_enabled=config_values.entity_pack_score_contract_enabled, retrieval_window=retrieval_window, semantic_text_encoder=visual_faiss_grounder._encode_texts)
            elif visual_faiss_ger_fusion:
                config_values = visual_faiss_config or VisualFaissConfig()
                if visual_faiss_unified_candidates:
                    fuse_visual_faiss_unified_candidates(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, context_radius=config_values.ger_context_radius, context_limit=config_values.ger_context_limit, question=question, question_ir=question_ir or {})
                else:
                    fuse_visual_faiss_ger_evidence(graph_answer, visual_payload, graph=graph, object_limit=config_values.ger_object_limit, context_radius=config_values.ger_context_radius, context_limit=config_values.ger_context_limit, binding_enabled=visual_faiss_ger_binding, structural_binding_enabled=visual_faiss_structural_binding, attribute_owner_enabled=visual_faiss_attribute_owner, fragment_bridge_enabled=visual_faiss_fragment_bridge, object_local_facts_enabled=visual_faiss_object_local_facts, local_subgraph_enabled=visual_faiss_local_subgraph, local_subgraph_anchor_only=visual_faiss_local_subgraph_anchor_only, balanced_graph_visual_evidence=visual_faiss_balanced_graph_visual, role_aware_compression_enabled=visual_faiss_role_aware_compression, question=question, question_ir=question_ir or {})
                if visual_faiss_role_aware_aux:
                    config_values = visual_faiss_config or VisualFaissConfig()
                    attach_visual_faiss_role_aware_auxiliary_evidence(graph_answer, visual_payload, object_limit=config_values.role_aware_object_limit, context_limit=config_values.role_aware_context_limit, question=question, question_ir=question_ir or {})
            else:
                attach_visual_faiss_evidence(graph_answer, visual_payload)
    if frame_faiss_grounder is not None and isinstance(graph_answer, dict) and frame_faiss_query_recall:
        frame_payload = frame_faiss_grounder.build_query_recall(seq_id=str(row.get(FIELD_SEQ_ID, '')).strip(), question=question, question_ir=question_ir or {}, config=frame_faiss_config or FrameFaissConfig())
        if frame_faiss_graph_anchors:
            expand_frame_faiss_graph_evidence(graph_answer, frame_payload, graph=graph, max_items=(frame_faiss_config or FrameFaissConfig()).max_clusters)
        elif frame_faiss_fused_evidence:
            fuse_frame_faiss_evidence(graph_answer, frame_payload)
        else:
            attach_frame_faiss_evidence(graph_answer, frame_payload)
    if merged_frame_faiss_grounder is not None and isinstance(graph_answer, dict) and merged_frame_faiss_query_recall:
        merged_frame_payload = merged_frame_faiss_grounder.build_query_recall(seq_id=str(row.get(FIELD_SEQ_ID, '')).strip(), question=question, question_ir=question_ir or {}, config=merged_frame_faiss_config or MergedFrameFaissConfig())
        if merged_frame_faiss_graph_anchors:
            expand_merged_frame_faiss_graph_evidence(graph_answer, merged_frame_payload, graph=graph, max_items=(merged_frame_faiss_config or MergedFrameFaissConfig()).max_clusters)
        elif merged_frame_faiss_fused_evidence:
            fuse_merged_frame_faiss_evidence(graph_answer, merged_frame_payload)
        else:
            attach_merged_frame_faiss_evidence(graph_answer, merged_frame_payload)
    if frame_window_faiss_grounder is not None and isinstance(graph_answer, dict) and frame_window_faiss_query_recall:
        frame_window_payload = frame_window_faiss_grounder.build_query_recall(seq_id=str(row.get(FIELD_SEQ_ID, '')).strip(), question=question, question_ir=question_ir or {}, config=frame_window_faiss_config or FrameWindowFaissConfig())
        if frame_window_faiss_graph_anchors:
            expand_merged_frame_faiss_graph_evidence(graph_answer, frame_window_payload, graph=graph, max_items=(frame_window_faiss_config or FrameWindowFaissConfig()).max_clusters)
        else:
            attach_merged_frame_faiss_evidence(graph_answer, frame_window_payload)
    if isinstance(graph_answer, dict) and (visual_faiss_grounder is not None and (visual_faiss_focus_seed_evidence or visual_faiss_query_recall) or (frame_faiss_grounder is not None and frame_faiss_query_recall) or (merged_frame_faiss_grounder is not None and merged_frame_faiss_query_recall) or (frame_window_faiss_grounder is not None and frame_window_faiss_query_recall)):
        payload['llm_answer'] = answer_via_mode(answer_mode=answer_mode, config_path=config, question=answer_question_text, graph_result=graph_answer, graph_qa_config_overrides=graph_qa_config_overrides)
    llm_answer = payload.get('llm_answer', {})
    bundle = graph_answer.get('llm_ready_bundle', {})
    raw_payload = graph_answer.get('retrieval_bundle', {}).get('raw_payload', {})
    query_plan = raw_payload.get('query_plan', {})
    structured = llm_answer.get('structured_answer', {})
    official_label = materialize_official_answer_label(row, question_metadata=question_time_metadata, position_labels=position_labels)
    official_format_prediction = render_official_format_prediction(row, graph_answer=graph_answer, llm_answer=llm_answer, official_label=official_label_to_dict(official_label), position_labels=position_labels, position_windows=position_windows, frame_time_index=frame_time_index, question_time_metadata=question_time_metadata)
    raw_predicted_answer = str(llm_answer.get('answer_text', '') or '')
    answer_consistency = answer_consistency_for_official_render(row, official_format_prediction=official_format_prediction, raw_answer_text=raw_predicted_answer)
    route_backbone = bundle.get('question_route_hybrid_backbone', {})
    if not isinstance(route_backbone, dict):
        route_backbone = {}
    question_route_dict = question_route.to_dict() if question_route is not None else {}
    return {'uuid': row.get(FIELD_UUID, ''), 'seq_id': row.get(FIELD_SEQ_ID, ''), 'question': question, 'question_processing_source': 'llm_question_ir+question_route' if has_llm_question_ir and question_route is not None else 'question_route+rule_question_text' if question_route is not None else 'llm_question_ir' if has_llm_question_ir else 'rule_question_text', 'question_ir': question_ir or {}, 'original_question_ir': original_question_ir or {}, 'question_route': question_route_dict, 'question_route_id': question_route.route_id if question_route is not None else '', 'taxonomy_path': question_route_dict.get('taxonomy_path', ''), 'taxonomy_root': question_route_dict.get('taxonomy_root', ''), 'taxonomy_domain': question_route_dict.get('taxonomy_domain', ''), 'taxonomy_task': question_route_dict.get('taxonomy_task', ''), 'evidence_family': question_route_dict.get('evidence_family', ''), 'taxonomy_source': question_route_dict.get('taxonomy_source', ''), 'taxonomy_contract': question_route_dict.get('taxonomy_contract', ''), 'question_route_hybrid_backbone': route_backbone, 'route_backbone_id': route_backbone.get('route_backbone_id', ''), 'route_backbone_reason': route_backbone.get('route_backbone_reason', ''), 'route_candidate_unit': route_backbone.get('route_candidate_unit', ''), 'uses_question_route_hybrid_backbone': bool(route_backbone.get('uses_question_route_hybrid_backbone')), 'uses_question_route_specialized_backbone': bool(route_backbone.get('uses_question_route_specialized_backbone')), 'synonym_expansion': synonym_expansion_metadata(question_ir), 'question_ir_query_text': memory_query.query_text, 'question_type': row.get(FIELD_TYPE, ''), 'question_category': row.get(FIELD_CATEGORY, ''), 'official_timestamp': row.get(FIELD_TIMESTAMP, ''), 'official_text_answer': row.get(FIELD_TEXT_ANSWER, ''), 'official_parsable_answer': row.get(FIELD_PARSABLE_ANSWER, ''), 'official_answer': official_answer(row), 'official_evaluation_label': official_label_to_dict(official_label), 'official_evaluation_value': official_label.display, 'official_evaluation_key': official_label.evaluation_key, 'official_evaluation_semantics': official_label.semantics, 'official_format_prediction': official_format_prediction['display'], 'official_format_prediction_source': official_format_prediction['source'], 'official_format_prediction_semantics': official_format_prediction['semantics'], 'official_format_prediction_details': official_format_prediction.get('details', {}), 'adapter_injects_into_graph': official_label.inject_into_graph, 'retrieval_query': payload.get('parsed_query', {}), 'question_retrieval_window': retrieval_window, 'query_start_ts': memory_query.start_ts, 'query_end_ts': memory_query.end_ts, 'answer_mode': answer_mode, 'status': llm_answer.get('graph_rag_status', ''), 'provider': llm_answer.get('provider', ''), 'model': llm_answer.get('model', ''), 'llm_error': llm_answer.get('error', ''), 'llm_raw_content': llm_answer.get('raw_content', ''), 'llm_response_metadata': llm_answer.get('response_metadata', {}), 'predicted_answer': raw_predicted_answer, 'llm_predicted_answer_raw': raw_predicted_answer, 'answer_consistency': answer_consistency, 'used_focus_frames': structured.get('used_focus_frames', []), 'used_evidence_ranks': structured.get('used_evidence_ranks', []), 'used_time_anchor_ids': structured.get('used_time_anchor_ids', []), 'uncertainty': structured.get('uncertainty', []), 'answer_brief': graph_answer.get('answer_brief', ''), 'query_plan': query_plan, 'global_scope_score': query_plan.get('global_scope_score', ''), 'query_family': query_plan.get('query_family', ''), 'core_terms': query_plan.get('core_terms', []), 'retrieval_budget': query_plan.get('retrieval_budget', ''), 'candidate_strategy': query_plan.get('candidate_strategy', ''), 'candidate_count': query_plan.get('candidate_count', ''), 'scene_seed_count': query_plan.get('scene_seed_count', ''), 'seed_limit': query_plan.get('seed_limit', ''), 'max_hops': query_plan.get('max_hops', ''), 'evidence_lines': bundle.get('evidence_lines', [])[:8], 'evidence_aggregation_lines': bundle.get('evidence_aggregation_lines', [])[:8], 'focus_frame_lines': bundle.get('focus_frame_lines', [])[:8], 'scene_lines': bundle.get('scene_lines', [])[:6], 'temporal_span_candidate_lines': bundle.get('temporal_span_candidate_lines', [])[:8], 'temporal_span_candidates': (graph_answer.get('retrieval_bundle', {}) if isinstance(graph_answer.get('retrieval_bundle', {}), dict) else {}).get('temporal_span_candidates', [])[:8], 'semantic_event_candidates': (graph_answer.get('retrieval_bundle', {}) if isinstance(graph_answer.get('retrieval_bundle', {}), dict) else {}).get('semantic_event_candidates', [])[:8], 'llm_evidence_selection': bundle.get('llm_evidence_selection', {}), 'llm_selected_evidence_lines': bundle.get('llm_selected_evidence_lines', [])[:8], 'answer_candidate_extraction': bundle.get('answer_candidate_extraction', {}), 'answer_candidate_lines': bundle.get('answer_candidate_lines', [])[:8], 'atomic_answer_facts': bundle.get('atomic_answer_facts', {}), 'atomic_answer_fact_lines': bundle.get('atomic_answer_fact_lines', [])[:8], 'unified_evidence_layer': bundle.get('unified_evidence_layer', {}), 'unified_evidence_lines': bundle.get('unified_evidence_lines', [])[:8], 'visual_faiss_grounding': bundle.get('visual_faiss_grounding', {}), 'visual_faiss_evidence_lines': bundle.get('visual_faiss_evidence_lines', [])[:8], 'visual_entity_candidate_packs': bundle.get('visual_entity_candidate_packs', {}), 'visual_entity_candidate_pack_lines': bundle.get('visual_entity_candidate_pack_lines', [])[:8], 'score_contract': bundle.get('score_contract', {}), 'fallback_evidence_lines': bundle.get('fallback_evidence_lines', [])[:8], 'question_route_count_set': bundle.get('question_route_count_set', {}), 'question_route_count_set_lines': bundle.get('question_route_count_set_lines', [])[:8], 'question_route_temporal_interval': bundle.get('question_route_temporal_interval', {}), 'question_route_temporal_interval_lines': bundle.get('question_route_temporal_interval_lines', [])[:8], 'visual_faiss_ger_fusion': bundle.get('visual_faiss_ger_fusion', {}), 'visual_faiss_ger_bound_fact_lines': bundle.get('visual_faiss_ger_bound_fact_lines', [])[:8], 'visual_faiss_evidence_layer': bundle.get('visual_faiss_evidence_layer', {}), 'visual_faiss_evidence_layer_lines': bundle.get('visual_faiss_evidence_layer_lines', [])[:8], 'visual_faiss_role_aware_evidence': bundle.get('visual_faiss_role_aware_evidence', {}), 'visual_faiss_role_aware_lines': bundle.get('visual_faiss_role_aware_lines', [])[:8], 'visual_faiss_role_aware_auxiliary': bundle.get('visual_faiss_role_aware_auxiliary', {}), 'visual_faiss_role_aware_aux_lines': bundle.get('visual_faiss_role_aware_aux_lines', [])[:8], 'visual_faiss_object_local_fact_lines': bundle.get('visual_faiss_object_local_fact_lines', [])[:8], 'visual_faiss_local_subgraph_fact_lines': bundle.get('visual_faiss_local_subgraph_fact_lines', [])[:8], 'visual_faiss_fragment_bridge_lines': bundle.get('visual_faiss_fragment_bridge_lines', [])[:8], 'visual_faiss_ger_evidence_lines': bundle.get('visual_faiss_ger_evidence_lines', [])[:8], 'frame_faiss_grounding': bundle.get('frame_faiss_grounding', {}), 'frame_faiss_evidence_lines': bundle.get('frame_faiss_evidence_lines', [])[:8], 'merged_frame_faiss_grounding': bundle.get('merged_frame_faiss_grounding', {}), 'merged_frame_faiss_evidence_lines': bundle.get('merged_frame_faiss_evidence_lines', [])[:8], 'match_count': len(graph_answer.get('matches', []))}

def render_official_format_prediction(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], official_label: dict[str, Any], position_labels: dict[str, list[float]] | None=None, position_windows: list[PositionWindow] | None=None, frame_time_index: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, str]:
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    answer_text = str(llm_answer.get('answer_text', '')).strip()
    if qtype == 'position':
        rendered_position = _render_position_from_graph_event_time(row, graph_answer=graph_answer, llm_answer=llm_answer, position_windows=position_windows, question_time_metadata=question_time_metadata)
        if rendered_position is not None:
            position = rendered_position['position']
            return {'display': '[' + ', '.join((f'{item:.6f}' for item in position)) + ']', 'source': 'graph_event_time_to_pose_timeline', 'semantics': 'ego_pose_at_semantic_event_time', 'details': rendered_position}
        return {'display': answer_text, 'source': 'llm_answer_text_no_graph_event_pose_render', 'semantics': 'natural_language', 'details': {'reason': 'No graph-selected event timestamp could be mapped to the pose timeline. Official position labels are not used as predictions.', 'has_position_gold_label': bool(position_labels)}}
    if qtype == 'time':
        rendered = _render_time_prediction(row, graph_answer=graph_answer, llm_answer=llm_answer, frame_time_index=frame_time_index, question_time_metadata=question_time_metadata)
        if rendered:
            return {'display': rendered['display'], 'source': rendered['source'], 'semantics': 'minutes_before_question_end_time', 'details': rendered.get('details', {})}
        return {'display': answer_text, 'source': 'llm_answer_text_no_time_anchor_render', 'semantics': 'natural_language'}
    if qtype == 'duration':
        rendered = _render_duration_from_frame_interval(row, graph_answer=graph_answer, llm_answer=llm_answer, frame_time_index=frame_time_index, question_time_metadata=question_time_metadata)
        if rendered:
            return {'display': rendered['display'], 'source': rendered['source'], 'semantics': 'event_duration_minutes', 'details': rendered.get('details', {})}
        rendered = _render_duration_from_answer(answer_text)
        if rendered:
            return {'display': rendered, 'source': 'llm_answer_duration_text', 'semantics': 'event_duration_minutes'}
    if qtype == 'binary':
        rendered_binary = _render_binary_from_question_route_evidence_contract(row, graph_answer=graph_answer, answer_text=answer_text)
        if rendered_binary is not None:
            return rendered_binary
        rendered_binary = _render_binary_from_answer_text(answer_text)
        if rendered_binary is not None:
            return rendered_binary
    return {'display': answer_text, 'source': 'llm_answer_text', 'semantics': str(official_label.get('semantics') or 'natural_language')}

def final_predicted_answer_for_output(*, official_format_prediction: dict[str, Any], raw_answer_text: str) -> str:
    display = str(official_format_prediction.get('display') or '').strip()
    if _is_graph_backed_official_render(official_format_prediction) and display:
        return display
    return str(raw_answer_text or '')

def answer_consistency_for_official_render(row: dict[str, str], *, official_format_prediction: dict[str, Any], raw_answer_text: str, tolerance_minutes: float=0.3) -> dict[str, Any]:
    raw_text = str(raw_answer_text or '').strip()
    rendered_text = str(official_format_prediction.get('display') or '').strip()
    source = str(official_format_prediction.get('source') or '')
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    graph_backed = _is_graph_backed_official_render(official_format_prediction)
    consistency: dict[str, Any] = {'status': 'not_checked', 'checked': False, 'graph_backed_official_render': graph_backed, 'official_format_prediction': rendered_text, 'official_format_prediction_source': source}
    if raw_text:
        consistency['raw_llm_answer'] = raw_text
    if not graph_backed or qtype not in {'time', 'duration'} or (not rendered_text):
        return consistency
    rendered_minutes = _duration_minutes_from_text(rendered_text)
    raw_minute_candidates = _duration_minute_candidates_from_text(raw_text)
    raw_minutes = _closest_minute_candidate(rendered_minutes, raw_minute_candidates)
    consistency.update({'status': 'no_explicit_raw_numeric_duration', 'checked': True, 'rendered_minutes': rendered_minutes, 'raw_minutes': raw_minutes, 'raw_minute_candidates': raw_minute_candidates, 'tolerance_minutes': tolerance_minutes})
    if rendered_minutes is None:
        consistency['status'] = 'rendered_numeric_unavailable'
        return consistency
    if raw_minutes is None:
        return consistency
    delta = _minute_candidate_delta(rendered_minutes, raw_minute_candidates)
    consistency['delta_minutes'] = round(delta, 4)
    if delta > tolerance_minutes:
        consistency['status'] = 'raw_llm_numeric_conflict_with_graph_render'
        details = official_format_prediction.get('details', {})
        if isinstance(details, dict):
            consistency['render_details'] = {key: details.get(key) for key in ('start_frame', 'end_frame', 'event_frame_index', 'duration_seconds', 'delta_seconds', 'interval_source', 'event_source', 'prediction_data_flow') if key in details}
        text_frames = _frames_from_text(raw_text)
        if text_frames:
            consistency['raw_answer_frames'] = text_frames
        return consistency
    consistency['status'] = 'consistent'
    return consistency

def _is_graph_backed_official_render(official_format_prediction: dict[str, Any]) -> bool:
    source = str(official_format_prediction.get('source') or '')
    if not source.startswith('graph_'):
        return False
    details = official_format_prediction.get('details', {})
    if isinstance(details, dict):
        if details.get('uses_official_answer_timestamp') is True:
            return False
        if details.get('uses_question_uuid_position_label') is True:
            return False
    return True

def _render_binary_from_answer_text(answer_text: str) -> dict[str, Any] | None:
    analysis = _analyze_binary_answer_text(answer_text)
    if analysis['clear']:
        return {'display': str(analysis['value']), 'source': 'llm_answer_text_binary_clear', 'semantics': 'yes_no', 'details': {'fallback_guard': 'explicit_non_conflicting_yes_no_text', 'reason': analysis['reason'], 'yes_cue_count': analysis['yes_cue_count'], 'no_cue_count': analysis['no_cue_count']}}
    if answer_text.strip():
        return {'display': answer_text, 'source': 'llm_answer_text_binary_ambiguous', 'semantics': 'natural_language', 'details': {'fallback_guard': 'ambiguous_binary_text_not_rendered_as_yes_no', 'reason': analysis['reason'], 'yes_cue_count': analysis['yes_cue_count'], 'no_cue_count': analysis['no_cue_count']}}
    return None

def _render_binary_from_question_route_evidence_contract(row: dict[str, str], *, graph_answer: dict[str, Any], answer_text: str) -> dict[str, Any] | None:
    if str(row.get(FIELD_TYPE, '')).strip().lower() != 'binary':
        return None
    if _yes_no_value(answer_text) == 'yes':
        return None
    question_ir = _question_ir_from_graph_answer(graph_answer)
    if not isinstance(question_ir, dict):
        return None
    route = question_ir.get('question_route', {})
    if not isinstance(route, dict) or str(route.get('route_id') or '') != 'binary_existence':
        return None
    if str(question_ir.get('operation') or '').strip().lower() != 'existence_query':
        return None
    if str(question_ir.get('operation') or '').strip().lower() == 'current_state':
        return None
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return None
    selection = bundle.get('llm_evidence_selection', {})
    if not isinstance(selection, dict):
        selection = {}
    selected_evidence = [item for item in selection.get('selected_evidence', []) if isinstance(item, dict)]
    insufficient = bool(selection.get('insufficient_evidence'))
    cue = _yes_no_value(str(selection.get('answer_cue') or ''))
    if selected_evidence and (not insufficient) and (cue == 'yes'):
        return {'display': 'yes', 'source': 'graph_question_route_binary_existence_selected_cue', 'semantics': 'yes_no', 'details': {'reason': 'llm_evidence_selection_yes_cue', 'question_route_id': 'binary_existence', 'selected_evidence_count': len(selected_evidence), 'uses_official_answer_timestamp': False}}
    support = _binary_existence_support_from_evidence(question_ir=question_ir, route=route, selected_evidence=selected_evidence, candidate_lines=_binary_existence_candidate_lines(bundle), allow_candidate_lines=not selected_evidence)
    if support is None:
        return None
    if selected_evidence and insufficient:
        return None
    return {'display': 'yes', 'source': 'graph_question_route_binary_existence_evidence_contract', 'semantics': 'yes_no', 'details': {'reason': support['reason'], 'question_route_id': 'binary_existence', 'support_source': support['source'], 'support_match_kind': support['match_kind'], 'support_terms': support['terms'], 'support_text_excerpt': support['text_excerpt'], 'selected_evidence_count': len(selected_evidence), 'uses_official_answer_timestamp': False}}

def _binary_existence_candidate_lines(bundle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ('llm_selected_evidence_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_ger_evidence_lines', 'evidence_lines', 'evidence_aggregation_lines'):
        value = bundle.get(key)
        if not isinstance(value, list):
            continue
        lines.extend((str(item) for item in value if str(item).strip()))
    return lines

def _binary_existence_support_from_evidence(*, question_ir: dict[str, Any], route: dict[str, Any], selected_evidence: list[dict[str, Any]], candidate_lines: list[str], allow_candidate_lines: bool) -> dict[str, Any] | None:
    target_groups = _binary_existence_target_token_groups(question_ir, route)
    if not target_groups:
        return None
    constraint_groups = _binary_existence_constraint_token_groups(question_ir, target_groups)
    selected_lines = [str(item.get('text') or '') for item in selected_evidence if str(item.get('text') or '').strip()]
    support = _binary_existence_find_support(selected_lines, target_groups=target_groups, constraint_groups=constraint_groups, source='selected_evidence', allow_head_term=True, require_exact_target=bool(constraint_groups))
    if support is not None:
        support['reason'] = 'selected_evidence_target_support'
        return support
    if not allow_candidate_lines:
        return None
    support = _binary_existence_find_support(candidate_lines, target_groups=target_groups, constraint_groups=constraint_groups, source='candidate_evidence', allow_head_term=True, require_exact_target=bool(constraint_groups))
    if support is not None:
        support['reason'] = 'candidate_evidence_target_support'
    return support

def _binary_existence_find_support(lines: list[str], *, target_groups: list[list[str]], constraint_groups: list[list[str]], source: str, allow_head_term: bool, require_exact_target: bool) -> dict[str, Any] | None:
    for line in lines:
        cleaned = _binary_existence_observation_text(line)
        if not cleaned:
            continue
        tokens = set(_binary_contract_tokens(cleaned))
        if not tokens:
            continue
        if not all((_binary_token_group_supported(group, tokens, cleaned, allow_head_term=False)[0] for group in constraint_groups)):
            continue
        for group in target_groups:
            supported, match_kind = _binary_token_group_supported(group, tokens, cleaned, allow_head_term=allow_head_term)
            if require_exact_target and match_kind != 'exact_phrase':
                continue
            if supported:
                return {'source': source, 'match_kind': match_kind, 'terms': group, 'text_excerpt': cleaned[:240]}
    return None

def _binary_existence_target_token_groups(question_ir: dict[str, Any], route: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for value in route.get('primary_target_terms', []):
        if isinstance(value, str):
            groups.append(_binary_contract_significant_tokens(value))
    for value in question_ir.get('target_entities', []):
        if isinstance(value, str):
            groups.append(_binary_contract_significant_tokens(value))
    return _dedupe_token_groups(groups)

def _binary_existence_constraint_token_groups(question_ir: dict[str, Any], target_groups: list[list[str]]) -> list[list[str]]:
    groups: list[list[str]] = []
    hard_constraints = question_ir.get('hard_constraints', [])
    if not isinstance(hard_constraints, list):
        return groups
    target_sets = [set(group) for group in target_groups]
    for item in hard_constraints:
        if not isinstance(item, dict):
            continue
        tokens = _binary_contract_significant_tokens(str(item.get('text') or ''))
        if not tokens:
            continue
        token_set = set(tokens)
        if str(item.get('kind') or '') == 'entity' and any((token_set and token_set.issubset(target_set) for target_set in target_sets)):
            continue
        groups.append(tokens)
    return _dedupe_token_groups(groups)

def _binary_token_group_supported(group: list[str], tokens: set[str], cleaned_text: str, *, allow_head_term: bool) -> tuple[bool, str]:
    if not group:
        return (False, '')
    phrase = ' '.join(group)
    if phrase and re.search(f'\\b{re.escape(phrase)}\\b', cleaned_text):
        return (True, 'exact_phrase')
    matches = [token for token in group if token in tokens]
    if len(matches) == len(group):
        return (True, 'all_terms')
    if allow_head_term and len(group) >= 3 and (group[-1] in tokens):
        return (True, 'head_term')
    return (False, '')

def _binary_existence_observation_text(value: str) -> str:
    text = str(value or '')
    text = re.sub('\\bquery=(?:\'[^\']*\'|\\"[^\\"]*\\"|[^;|]+)', ' ', text)
    text = re.sub('\\bq_terms=[^\\s:;|]+', ' ', text)
    text = re.sub('\\bconstraint_terms=[^\\s:;|]+', ' ', text)
    text = re.sub('\\bmatched_question_terms=[^\\s:;|]+', ' ', text)
    text = re.sub('\\bmatched_constraint_terms=[^\\s:;|]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip().lower()

def _binary_contract_significant_tokens(value: str) -> list[str]:
    return [token for token in _binary_contract_tokens(value) if token not in _BINARY_CONTRACT_STOPWORDS and len(token) > 1]

def _binary_contract_tokens(value: str) -> list[str]:
    return [_binary_contract_normalize_token(token) for token in re.findall('[a-z0-9]+', value.lower())]

def _binary_contract_normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith('ies'):
        return token[:-3] + 'y'
    if len(token) > 5 and token.endswith('ing'):
        return token[:-3]
    if len(token) > 4 and token.endswith('ed'):
        return token[:-2]
    if len(token) > 3 and token.endswith('s'):
        return token[:-1]
    return token

def _dedupe_token_groups(groups: list[list[str]]) -> list[list[str]]:
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        cleaned = [token for token in group if token]
        key = tuple(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped
_BINARY_CONTRACT_STOPWORDS = {'a', 'an', 'and', 'any', 'are', 'at', 'be', 'by', 'can', 'did', 'do', 'does', 'for', 'from', 'have', 'i', 'in', 'is', 'it', 'me', 'of', 'on', 'or', 'see', 'saw', 'the', 'there', 'to', 'was', 'were', 'with', 'you'}

def _has_graph_answer_signal(graph_answer: dict[str, Any], llm_answer: dict[str, Any]) -> bool:
    if graph_answer.get('matches'):
        return True
    structured = llm_answer.get('structured_answer', {})
    return bool(isinstance(structured, dict) and (structured.get('used_focus_frames') or structured.get('used_evidence_lines') or structured.get('used_time_anchor_ids')))

def _render_position_from_graph_event_time(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], position_windows: list[PositionWindow] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    if not position_windows:
        return None
    event = _select_graph_event_time_for_position(row, graph_answer=graph_answer, llm_answer=llm_answer, question_time_metadata=question_time_metadata)
    if event is None:
        return None
    timestamp = event.get('timestamp_sec')
    if not isinstance(timestamp, int | float):
        return None
    lookup = lookup_position_for_timestamp(position_windows, float(timestamp))
    if lookup is None:
        return None
    return {'position': [float(value) for value in lookup.position], 'event_timestamp_sec': float(timestamp), 'event_timestamp_iso': _iso_from_unix(float(timestamp)), 'event_source': event.get('source'), 'event_anchor_id': event.get('anchor_id'), 'event_frame_index': event.get('frame_index'), 'event_scene_id': event.get('scene_id'), 'pose_window_id': lookup.window_id, 'pose_window_start_time': lookup.window_start_time, 'pose_window_end_time': lookup.window_end_time, 'pose_lookup_delta_seconds': round(lookup.delta_seconds, 6), 'pose_window_sample_count': lookup.sample_count, 'prediction_data_flow': 'question -> graph retrieval/LLM evidence selection -> event timestamp -> independent pose timeline -> official-format coordinate', 'uses_official_answer_timestamp': False, 'uses_question_uuid_position_label': False}

def _select_graph_event_time_for_position(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    structured = llm_answer.get('structured_answer', {})
    used_ids = structured.get('used_time_anchor_ids', []) if isinstance(structured, dict) else []
    adaptation = build_graph_time_adaptation(question=question_time_context(row, question_time_metadata or {}), graph_result=graph_answer)
    candidates = [item for item in adaptation.get('relative_time_candidates', []) if isinstance(item, dict) and isinstance(item.get('timestamp_sec'), int | float)]
    by_anchor = {str(item.get('anchor_id')): item for item in candidates}
    if isinstance(used_ids, list):
        for anchor_id in used_ids:
            item = by_anchor.get(str(anchor_id))
            if item is not None:
                return dict(item)
    used_frames = structured.get('used_focus_frames', []) if isinstance(structured, dict) else []
    frame_ids = [int(value) for value in used_frames if isinstance(value, int)]
    for frame_id in frame_ids:
        for item in candidates:
            if item.get('source') != 'focus_frame':
                continue
            if _optional_int(item.get('frame_index')) == frame_id:
                return dict(item)
    for item in candidates:
        source = str(item.get('source') or '')
        if source.startswith('match:') or source == 'focus_frame':
            return dict(item)
    return candidates[0] if candidates else None

def _render_time_prediction(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], frame_time_index: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    rendered_point = _render_time_from_temporal_point_candidate(row, graph_answer=graph_answer, question_time_metadata=question_time_metadata)
    if rendered_point:
        return rendered_point
    rendered = _render_time_from_used_anchor(row, graph_answer=graph_answer, llm_answer=llm_answer, question_time_metadata=question_time_metadata)
    if rendered:
        return {'display': rendered, 'source': 'graph_time_anchor_relative_to_question_end', 'details': {'prediction_data_flow': 'question -> graph retrieval/LLM selected time anchor -> relative time adaptation', 'uses_official_answer_timestamp': False}}
    rendered_from_frame = _render_time_from_frame_time_index(row, graph_answer=graph_answer, llm_answer=llm_answer, frame_time_index=frame_time_index, question_time_metadata=question_time_metadata)
    if rendered_from_frame:
        return rendered_from_frame
    return None

def _render_time_from_temporal_point_candidate(row: dict[str, str], *, graph_answer: dict[str, Any], question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    question_end = _question_end_time(row, question_time_metadata)
    if question_end is None:
        return None
    question_ir = _question_ir_from_graph_answer(graph_answer)
    route = question_ir.get('question_route', {}) if isinstance(question_ir, dict) else {}
    if str(route.get('route_id') or '') != 'temporal_point':
        return None
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return None
    structured = bundle.get('structured_graph_context', {})
    if not isinstance(structured, dict):
        structured = {}
    pack = structured.get('question_route_temporal_point') or bundle.get('question_route_temporal_point')
    if not isinstance(pack, dict):
        return None
    if str(pack.get('selection_status') or '') != 'selected':
        return None
    selected = pack.get('selected_candidate')
    if not isinstance(selected, dict):
        return None
    timestamp = _float_or_none(selected.get('timestamp_sec'))
    if timestamp is None:
        return None
    delta_seconds = question_end - timestamp
    minutes = abs(delta_seconds) / 60.0
    suffix = 'ago' if delta_seconds >= 0 else 'after current time'
    return {'display': f'{minutes:.2f} minutes {suffix}', 'source': 'graph_question_route_temporal_point_candidate', 'details': {'event_frame_index': _optional_int(selected.get('frame')), 'event_source': 'question_route_temporal_point_candidate', 'event_rank': selected.get('result_rank'), 'event_timestamp_unix': timestamp, 'question_end_time_unix': question_end, 'delta_seconds': round(delta_seconds, 3), 'delta_minutes': round(delta_seconds / 60.0, 4), 'candidate_score': selected.get('score'), 'candidate_query_phrase': selected.get('query_phrase'), 'candidate_label': selected.get('label'), 'question_route_id': 'temporal_point', 'selection_status': pack.get('selection_status'), 'prediction_data_flow': 'question route -> temporal_point candidate pack -> graph timestamp -> relative minutes before question end', 'uses_official_answer_timestamp': False}}

def _render_time_from_used_anchor(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], question_time_metadata: dict[str, dict[str, Any]] | None=None) -> str | None:
    structured = llm_answer.get('structured_answer', {})
    used_ids = structured.get('used_time_anchor_ids', []) if isinstance(structured, dict) else []
    if not isinstance(used_ids, list) or not used_ids:
        return None
    adaptation = build_graph_time_adaptation(question=question_time_context(row, question_time_metadata or {}), graph_result=graph_answer)
    candidates = {item.get('anchor_id'): item for item in adaptation.get('relative_time_candidates', []) if isinstance(item, dict)}
    for anchor_id in used_ids:
        item = candidates.get(anchor_id)
        if not item:
            continue
        relative = item.get('relative_to_current')
        if not isinstance(relative, dict):
            continue
        minutes = relative.get('absolute_delta_minutes')
        if isinstance(minutes, int | float):
            return f'{float(minutes):.2f} minutes ago'
        display = relative.get('display_text')
        if isinstance(display, str) and display.strip():
            return display.strip()
    return None

def _render_time_from_frame_time_index(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], frame_time_index: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    question_end = _question_end_time(row, question_time_metadata)
    if question_end is None:
        return None
    event = _select_graph_event_frame_for_time(row, graph_answer=graph_answer, llm_answer=llm_answer)
    if event is None:
        return None
    mapped = _lookup_frame_time(frame_time_index, event.get('frame_index'))
    if mapped is None:
        return None
    delta_seconds = question_end - float(mapped['timestamp_unix'])
    minutes = abs(delta_seconds) / 60.0
    suffix = 'ago' if delta_seconds >= 0 else 'after current time'
    return {'display': f'{minutes:.2f} minutes {suffix}', 'source': 'graph_event_frame_time_index_relative_to_question_end', 'details': {'event_frame_index': event.get('frame_index'), 'event_source': event.get('source'), 'event_rank': event.get('rank'), 'event_scene_id': event.get('scene_id'), 'event_timestamp_unix': mapped['timestamp_unix'], 'event_timestamp_iso': mapped.get('timestamp_iso'), 'question_end_time_unix': question_end, 'delta_seconds': round(delta_seconds, 3), 'delta_minutes': round(delta_seconds / 60.0, 4), 'frame_time_index_source': frame_time_index.get('source_path') if isinstance(frame_time_index, dict) else None, 'prediction_data_flow': 'question -> graph retrieval/LLM selected frame -> frame-time index -> relative minutes before question end', 'uses_official_answer_timestamp': False}}

def _render_duration_from_frame_interval(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], frame_time_index: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    current_duration = _render_current_duration_from_frame_time_index(row, graph_answer=graph_answer, llm_answer=llm_answer, frame_time_index=frame_time_index, question_time_metadata=question_time_metadata)
    if current_duration is not None:
        return current_duration
    interval = _select_duration_frame_interval(row, graph_answer=graph_answer, llm_answer=llm_answer)
    if interval is None:
        return None
    start_mapped = _lookup_frame_time(frame_time_index, interval['start_frame'])
    end_mapped = _lookup_frame_time(frame_time_index, interval['end_frame'])
    if start_mapped is None or end_mapped is None:
        return None
    duration_seconds = abs(float(end_mapped['timestamp_unix']) - float(start_mapped['timestamp_unix']))
    return {'display': f'{duration_seconds / 60.0:.2f} minutes', 'source': 'graph_interval_frame_time_index_duration', 'details': {'start_frame': interval['start_frame'], 'end_frame': interval['end_frame'], 'interval_source': interval.get('source'), 'interval_rank': interval.get('rank'), 'start_timestamp_unix': start_mapped['timestamp_unix'], 'end_timestamp_unix': end_mapped['timestamp_unix'], 'duration_seconds': round(duration_seconds, 3), 'frame_time_index_source': frame_time_index.get('source_path') if isinstance(frame_time_index, dict) else None, 'prediction_data_flow': 'question -> graph retrieval/LLM selected interval -> frame-time index -> duration minutes', 'uses_official_answer_timestamp': False}}

def _render_current_duration_from_frame_time_index(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any], frame_time_index: dict[str, Any] | None=None, question_time_metadata: dict[str, dict[str, Any]] | None=None) -> dict[str, Any] | None:
    question = output_question(row)
    if not re.search('\\bhow\\s+long\\s+ha(?:ve|s)\\b', question, flags=re.I):
        return None
    question_end = _question_end_time(row, question_time_metadata)
    if question_end is None:
        return None
    event = _select_graph_event_frame_for_time(row, graph_answer=graph_answer, llm_answer=llm_answer)
    if event is None:
        return None
    mapped = _lookup_frame_time(frame_time_index, event.get('frame_index'))
    if mapped is None:
        return None
    duration_seconds = max(0.0, question_end - float(mapped['timestamp_unix']))
    return {'display': f'{duration_seconds / 60.0:.2f} minutes', 'source': 'graph_event_frame_time_index_current_duration', 'details': {'event_frame_index': event.get('frame_index'), 'event_source': event.get('source'), 'event_rank': event.get('rank'), 'event_timestamp_unix': mapped['timestamp_unix'], 'question_end_time_unix': question_end, 'duration_seconds': round(duration_seconds, 3), 'frame_time_index_source': frame_time_index.get('source_path') if isinstance(frame_time_index, dict) else None, 'prediction_data_flow': 'question -> graph retrieval/LLM selected start frame -> frame-time index -> duration to question end', 'uses_official_answer_timestamp': False}}

def _select_graph_event_frame_for_time(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any]) -> dict[str, Any] | None:
    structured = llm_answer.get('structured_answer', {})
    if not isinstance(structured, dict):
        structured = {}
    answer_text = str(llm_answer.get('answer_text', '') or '')
    question = output_question(row)
    semantic_candidates = _semantic_event_candidates(graph_answer)
    preferred_semantic = _preferred_semantic_event_candidate_for_time(question, graph_answer=graph_answer, candidates=semantic_candidates)
    if preferred_semantic is not None:
        frame_index = _optional_int(preferred_semantic.get('boundary_frame'))
        if frame_index is not None:
            return {'frame_index': frame_index, 'source': 'question_ir_semantic_event_candidate', 'rank': preferred_semantic.get('rank'), 'scene_id': preferred_semantic.get('boundary_scene_id')}
    text_frames = _singular_frames_from_text(answer_text)
    for frame_index in text_frames:
        semantic = _semantic_candidate_for_frame(semantic_candidates, frame_index=frame_index)
        if semantic is not None:
            return {'frame_index': frame_index, 'source': 'question_ir_semantic_event_candidate', 'rank': semantic.get('rank'), 'scene_id': semantic.get('boundary_scene_id')}
    preferred_frames = _ordered_unique_ints_for_eval([*text_frames, *list(structured.get('used_focus_frames', []))])
    if preferred_frames:
        focus_by_frame = _focus_frames_by_index(graph_answer)
        candidate_records = _time_candidate_records(graph_answer)
        for frame_index in preferred_frames:
            focus = focus_by_frame.get(frame_index)
            if focus is not None:
                return {'frame_index': frame_index, 'source': 'llm_selected_focus_frame', 'rank': focus.get('rank'), 'scene_id': focus.get('scene_id')}
            for item in candidate_records:
                if item['start_frame'] <= frame_index <= item['end_frame']:
                    return {'frame_index': frame_index, 'source': 'llm_answer_frame_in_graph_interval', 'rank': item.get('rank'), 'scene_id': item.get('scene_id')}
    for item in _time_candidate_records(graph_answer):
        selected = _event_frame_from_interval_for_question(item, question=question, answer_text=answer_text)
        if selected is None:
            continue
        return {'frame_index': selected, 'source': item.get('source'), 'rank': item.get('rank'), 'scene_id': item.get('scene_id')}
    return None

def _preferred_semantic_event_candidate_for_time(question: str, *, graph_answer: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    question_ir = _question_ir_from_graph_answer(graph_answer)
    operation = str(question_ir.get('operation') or '').strip().lower()
    if operation not in {'onset', 'offset'}:
        return None
    desired_event = 'enter' if operation == 'onset' else 'leave'
    target_text = ' '.join([question, ' '.join(_clean_ir_strings_for_eval(question_ir.get('target_state'))), ' '.join(_clean_ir_strings_for_eval(question_ir.get('target_entities')))]).lower()
    if desired_event == 'enter' and (not re.search('\\b(enter|entered|entry)\\b', target_text)):
        return None
    if desired_event == 'leave' and (not re.search('\\b(leave|left|exit|exited)\\b', target_text)):
        return None
    for item in candidates:
        if str(item.get('event_type') or '').strip().lower() == desired_event:
            return item
    return None

def _select_duration_frame_interval(row: dict[str, str], *, graph_answer: dict[str, Any], llm_answer: dict[str, Any]) -> dict[str, Any] | None:
    temporal_span_interval = _select_temporal_span_candidate_interval(graph_answer)
    if temporal_span_interval is not None:
        return temporal_span_interval
    answer_text = str(llm_answer.get('answer_text', '') or '')
    question = output_question(row)
    text_frames = _frames_from_text(answer_text)
    if len(text_frames) >= 2:
        start_frame = min(text_frames[0], text_frames[1])
        end_frame = max(text_frames[0], text_frames[1])
        if start_frame != end_frame:
            return {'start_frame': start_frame, 'end_frame': end_frame, 'source': 'llm_answer_frame_pair', 'rank': None}
    semantic_interval = _semantic_event_interval_for_duration(question, graph_answer=graph_answer)
    if semantic_interval is not None:
        return semantic_interval
    records = _time_candidate_records(graph_answer)
    records = [item for item in records if _duration_fallback_record_can_render_interval(item)]
    if not records:
        return None
    return records[0]

def _select_temporal_span_candidate_interval(graph_answer: dict[str, Any]) -> dict[str, Any] | None:
    records = _temporal_span_candidate_records(graph_answer)
    if not records:
        return None
    finer_records = [item for item in records if _duration_span_candidate_can_render_interval(item)]
    if not finer_records:
        return None
    return finer_records[0]

def _duration_span_candidate_can_render_interval(item: dict[str, Any]) -> bool:
    source_kind = str(item.get('source_kind') or '').strip().lower()
    if source_kind in {'scene', 'transition_event'}:
        return False
    start_frame = _optional_int(item.get('start_frame'))
    end_frame = _optional_int(item.get('end_frame'))
    return start_frame is not None and end_frame is not None and (start_frame != end_frame)

def _duration_fallback_record_can_render_interval(item: dict[str, Any]) -> bool:
    source = str(item.get('source') or '').strip().lower()
    if source in {'match:scene', 'match:transition_event'}:
        return False
    start_frame = _optional_int(item.get('start_frame'))
    end_frame = _optional_int(item.get('end_frame'))
    return start_frame is not None and end_frame is not None and (start_frame != end_frame)

def _temporal_span_candidate_records(graph_answer: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates: list[Any] = []
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        raw_candidates.extend(retrieval_bundle.get('temporal_span_candidates', []) or [])
    llm_ready_bundle = graph_answer.get('llm_ready_bundle', {})
    if isinstance(llm_ready_bundle, dict):
        structured = llm_ready_bundle.get('structured_graph_context', {})
        if isinstance(structured, dict):
            raw_candidates.extend(structured.get('temporal_span_candidates', []) or [])
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, str]] = set()
    for raw_rank, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue
        start_frame = _optional_int(item.get('start_frame'))
        end_frame = _optional_int(item.get('end_frame'))
        if start_frame is None or end_frame is None:
            continue
        if start_frame > end_frame:
            start_frame, end_frame = (end_frame, start_frame)
        if start_frame == end_frame:
            continue
        source = str(item.get('source') or 'temporal_span_candidate').strip()
        source_kind = str(item.get('source_kind') or '').strip()
        key = (start_frame, end_frame, source, source_kind)
        if key in seen:
            continue
        seen.add(key)
        records.append({'start_frame': start_frame, 'end_frame': end_frame, 'source': f'temporal_span_candidate:{source}', 'rank': item.get('rank', raw_rank), 'scene_id': item.get('scene_id'), 'summary': str(item.get('summary') or ''), 'source_kind': source_kind, 'score': _float_or_none(item.get('score')), 'warning': item.get('warning'), 'target_hits': _string_list(item.get('target_hits')), 'metadata': item})
    return sorted(records, key=_temporal_span_candidate_sort_key)

def _temporal_span_candidate_sort_key(item: dict[str, Any]) -> tuple[float, int, int, int]:
    score = _float_or_none(item.get('score'))
    source_kind = str(item.get('source_kind') or '').strip().lower()
    rank = _optional_int(item.get('rank')) or 10 ** 9
    start_frame = _optional_int(item.get('start_frame')) or 10 ** 9
    return (-(score if score is not None else 0.0), _temporal_span_candidate_source_priority(source_kind), rank, start_frame)

def _temporal_span_candidate_source_priority(source_kind: str) -> int:
    return {'episode': 0, 'entity': 1, 'stable_identity': 2, 'entity_attribute': 3, 'temporal_composition': 4, 'temporal_chain': 5, 'transition_event': 8, 'scene': 9}.get(str(source_kind or '').strip().lower(), 6)

def _semantic_event_candidates(graph_answer: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if not isinstance(retrieval_bundle, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(retrieval_bundle.get('semantic_event_candidates', []), start=1):
        if not isinstance(item, dict):
            continue
        frame = _optional_int(item.get('boundary_frame'))
        if frame is None:
            continue
        candidates.append({**item, 'rank': rank, 'boundary_frame': frame})
    return candidates

def _semantic_candidate_for_frame(candidates: list[dict[str, Any]], *, frame_index: int) -> dict[str, Any] | None:
    for item in candidates:
        if _optional_int(item.get('boundary_frame')) == frame_index:
            return item
    return None

def _semantic_event_interval_for_duration(question: str, *, graph_answer: dict[str, Any]) -> dict[str, Any] | None:
    question_ir = _question_ir_from_graph_answer(graph_answer)
    operation = str(question_ir.get('operation') or '').strip().lower()
    target_state = ' '.join(_clean_ir_strings_for_eval(question_ir.get('target_state')))
    target_entities = ' '.join(_clean_ir_strings_for_eval(question_ir.get('target_entities')))
    text = f'{question} {target_state} {target_entities}'.lower()
    if operation != 'interval' and 'how long' not in text:
        return None
    candidates = _semantic_event_candidates(graph_answer)
    enters = [item for item in candidates if str(item.get('event_type') or '').strip().lower() == 'enter']
    leaves = [item for item in candidates if str(item.get('event_type') or '').strip().lower() == 'leave']
    best: dict[str, Any] | None = None
    best_key: tuple[int, int, int] | None = None
    for enter in enters:
        enter_frame = _optional_int(enter.get('boundary_frame'))
        if enter_frame is None:
            continue
        for leave in leaves:
            leave_frame = _optional_int(leave.get('boundary_frame'))
            if leave_frame is None or leave_frame <= enter_frame:
                continue
            confidence = int(enter.get('confidence') or 0) + int(leave.get('confidence') or 0)
            duration = leave_frame - enter_frame
            key = (confidence, -duration, -enter_frame)
            if best_key is None or key > best_key:
                best_key = key
                best = {'start_frame': enter_frame, 'end_frame': leave_frame, 'source': 'question_ir_semantic_event_interval', 'rank': f"{enter.get('rank')}->{leave.get('rank')}", 'start_event': enter.get('event_type'), 'end_event': leave.get('event_type')}
    return best

def _clean_ir_strings_for_eval(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item or '').strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []

def _time_candidate_records(graph_answer: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(graph_answer.get('matches', []), start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get('metadata', {})
        if not isinstance(metadata, dict):
            continue
        for interval in _intervals_from_match(metadata):
            start_frame = _optional_int(interval.get('start_frame'))
            end_frame = _optional_int(interval.get('end_frame'))
            if start_frame is None or end_frame is None:
                continue
            if start_frame > end_frame:
                start_frame, end_frame = (end_frame, start_frame)
            records.append({'start_frame': start_frame, 'end_frame': end_frame, 'source': f"match:{item.get('source_kind')}", 'rank': rank, 'scene_id': interval.get('scene_id') or _scene_id_for_match_metadata(metadata), 'summary': str(item.get('summary') or ''), 'metadata': metadata})
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        for rank, chain in enumerate(retrieval_bundle.get('temporal_chains', []), start=1):
            if not isinstance(chain, dict):
                continue
            time_range = chain.get('time_range', {})
            if not isinstance(time_range, dict):
                time_range = {}
            start_frame = _optional_int(time_range.get('start_frame'))
            end_frame = _optional_int(time_range.get('end_frame'))
            if start_frame is None or end_frame is None:
                continue
            if start_frame > end_frame:
                start_frame, end_frame = (end_frame, start_frame)
            records.append({'start_frame': start_frame, 'end_frame': end_frame, 'source': 'temporal_chain', 'rank': rank, 'scene_id': None, 'summary': str(chain.get('summary') or ''), 'metadata': chain})
    return _dedupe_interval_records(records)

def _intervals_from_match(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    time_range = metadata.get('time_range', {})
    if isinstance(time_range, dict):
        intervals.append(dict(time_range))
    for scene in metadata.get('scene_sequence', []) or []:
        if not isinstance(scene, dict):
            continue
        intervals.append({'start_frame': scene.get('start_frame'), 'end_frame': scene.get('end_frame'), 'scene_id': scene.get('scene_id')})
    return intervals

def _dedupe_interval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in records:
        key = (int(item['start_frame']), int(item['end_frame']), str(item.get('source') or ''))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique

def _event_frame_from_interval_for_question(item: dict[str, Any], *, question: str, answer_text: str) -> int | None:
    start_frame = _optional_int(item.get('start_frame'))
    end_frame = _optional_int(item.get('end_frame'))
    if start_frame is None:
        return None
    if end_frame is None:
        return start_frame
    text = f'{question} {answer_text}'.lower()
    if re.search('\\b(left|leave|leaving|exit|exited|outside|off)\\b', text):
        return end_frame
    if re.search('\\b(first\\s+enter|enter|entered|entry)\\b', text):
        return start_frame
    if re.search('\\b(cross|crossed|crossing)\\b', text):
        return start_frame
    if re.search('\\b(get off|got off)\\b', text):
        return end_frame
    midpoint = int(round((start_frame + end_frame) / 2.0))
    return midpoint

def _lookup_frame_time(frame_time_index: dict[str, Any] | None, frame_index: Any) -> dict[str, Any] | None:
    if not isinstance(frame_time_index, dict):
        return None
    frame = _optional_int(frame_index)
    if frame is None:
        return None
    by_frame = frame_time_index.get('by_frame', {})
    if not isinstance(by_frame, dict) or not by_frame:
        return None
    item = by_frame.get(frame)
    if item is None:
        ordered = frame_time_index.get('ordered_frames', [])
        if not isinstance(ordered, list) or not ordered:
            return None
        nearest = min(ordered, key=lambda value: abs(int(value) - frame))
        item = by_frame.get(nearest)
    if not isinstance(item, dict):
        return None
    timestamp_unix = _float_or_none(item.get('timestamp_unix'))
    if timestamp_unix is None:
        return None
    return {**item, 'timestamp_unix': timestamp_unix}

def _question_end_time(row: dict[str, str], question_time_metadata: dict[str, dict[str, Any]] | None) -> float | None:
    if not question_time_metadata:
        return None
    item = question_time_metadata.get(str(row.get(FIELD_UUID, '')).strip())
    if not isinstance(item, dict):
        return None
    return _float_or_none(item.get('end_time'))

def _focus_frames_by_index(graph_answer: dict[str, Any]) -> dict[int, dict[str, Any]]:
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if not isinstance(retrieval_bundle, dict):
        return {}
    by_frame: dict[int, dict[str, Any]] = {}
    for item in retrieval_bundle.get('focus_frames', []):
        if not isinstance(item, dict):
            continue
        frame_index = _optional_int(item.get('frame_index'))
        if frame_index is None:
            continue
        by_frame.setdefault(frame_index, item)
    return by_frame

def _frames_from_text(text: str) -> list[int]:
    frames: list[int] = []
    for match in re.finditer('\\bframes?\\s+(-?\\d+)(?:\\s*(?:-|to|through|->)\\s*(-?\\d+))?', str(text or ''), flags=re.I):
        first = _optional_int(match.group(1))
        second = _optional_int(match.group(2))
        if first is not None:
            frames.append(first)
        if second is not None:
            frames.append(second)
    return _ordered_unique_ints_for_eval(frames)

def _singular_frames_from_text(text: str) -> list[int]:
    frames: list[int] = []
    for match in re.finditer('\\bframes?\\s+(-?\\d+)\\b', str(text or ''), flags=re.I):
        end_position = match.end()
        tail = str(text or '')[end_position:end_position + 16]
        if re.match('\\s*(?:-|to|through|->)\\s*-?\\d+', tail, flags=re.I):
            continue
        frame = _optional_int(match.group(1))
        if frame is not None:
            frames.append(frame)
    return _ordered_unique_ints_for_eval(frames)

def _ordered_unique_ints_for_eval(values: list[Any]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed = _optional_int(value)
        if parsed is None or parsed in seen:
            continue
        ordered.append(parsed)
        seen.add(parsed)
    return ordered

def _scene_id_for_match_metadata(metadata: dict[str, Any]) -> str | None:
    scene_id = str(metadata.get('scene_id') or '').strip()
    if scene_id:
        return scene_id
    scene_ids = metadata.get('scene_ids', [])
    if isinstance(scene_ids, list) and len(scene_ids) == 1:
        return str(scene_ids[0])
    return None

def _render_duration_from_answer(answer_text: str) -> str | None:
    value = _duration_minutes_from_text(answer_text)
    if value is not None:
        return f'{value:g} minutes'
    return None

def _question_ir_from_graph_answer(graph_answer: dict[str, Any]) -> dict[str, Any]:
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        raw_payload = retrieval_bundle.get('raw_payload', {})
        if isinstance(raw_payload, dict):
            query_plan = raw_payload.get('query_plan', {})
            if isinstance(query_plan, dict):
                question_ir = query_plan.get('question_ir')
                if isinstance(question_ir, dict):
                    return question_ir
    bundle = graph_answer.get('llm_ready_bundle', {})
    if isinstance(bundle, dict):
        question_ir = bundle.get('question_ir')
        if isinstance(question_ir, dict):
            return question_ir
    return {}

def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def load_pose_timeline(*, pose_timeline_json: Path | None, dense_pose: Path | None, timestamps: Path | None, seconds_per_caption: float, seq_id: str='0', cam: str='0', stride: int | None=10, layout_root: Path=(REPO_ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa')) -> list[PositionWindow]:
    if pose_timeline_json is not None:
        return load_position_windows_from_captions(resolve_repo_input_path(pose_timeline_json))
    if dense_pose is None and timestamps is None:
        layout = default_navqa_sequence_layout(seq_id=seq_id, cam=cam, stride=stride, layout_root=layout_root)
        default_candidates = ((resolve_repo_input_path(layout.poses_path), resolve_repo_input_path(layout.timestamps_path)), (resolve_repo_input_path(Path(f'.scratch/coda_seq{seq_id}_metadata/poses/{seq_id}.txt')), resolve_repo_input_path(Path(f'.scratch/coda_seq{seq_id}_metadata/timestamps/{seq_id}.txt'))))
        for default_dense_pose, default_timestamps in default_candidates:
            if default_dense_pose.exists() and default_timestamps.exists():
                dense_pose = default_dense_pose
                timestamps = default_timestamps
                break
        if dense_pose is None or timestamps is None:
            return []
    if dense_pose is None or timestamps is None:
        raise ValueError('--dense-pose and --timestamps must be provided together')
    dense_pose_path = resolve_repo_input_path(dense_pose)
    timestamps_path = resolve_repo_input_path(timestamps)
    dense_poses = load_dense_poses(dense_pose_path)
    lidar_timestamps = load_timestamps(timestamps_path)
    samples = coda_preprocessed_pose_samples(dense_poses, lidar_timestamps)
    return build_position_windows_from_pose_samples(samples, seconds_per_caption=seconds_per_caption)

def require_pose_timeline_for_position_questions(*, rows: list[dict[str, str]], position_windows: list[PositionWindow], seq_id: str, cam: str='0', stride: int | None=10, layout_root: Path=(REPO_ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa')) -> None:
    if position_windows or not has_position_questions(rows):
        return
    layout = default_navqa_sequence_layout(seq_id=seq_id, cam=cam, stride=stride, layout_root=layout_root)
    raise SystemExit(f'Selected NaVQA questions include position answers, but no pose timeline was available. Position predictions must be rendered from graph-selected event time through an independent pose/timestamp mapping.\nExpected local files:\n  - {layout.poses_path}\n  - {layout.timestamps_path}\nLegacy .scratch/coda_seq*_metadata paths are still accepted when provided or already present.\nYou can download these small official CODa members with:\n  python scripts/download_coda_zip_member.py poses/{seq_id}.txt --seq-id {seq_id} --out {layout.poses_path}\n  python scripts/download_coda_zip_member.py timestamps/{seq_id}.txt --seq-id {seq_id} --out {layout.timestamps_path}')

def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    include_position_diagnostics = any((str(item.get('question_type', '')).strip().lower() == 'position' for item in results))
    fieldnames = ['uuid', 'seq_id', 'question', 'question_type', 'question_category', 'question_route_id', 'taxonomy_path', 'taxonomy_root', 'taxonomy_domain', 'taxonomy_task', 'evidence_family', 'route_backbone_id', 'route_backbone_reason', 'route_candidate_unit', 'uses_question_route_hybrid_backbone', 'uses_question_route_specialized_backbone', 'official_timestamp', 'official_answer', 'official_evaluation_key', 'official_evaluation_value', 'official_evaluation_semantics', 'official_format_prediction', 'official_format_prediction_source', 'official_format_prediction_semantics', 'official_format_prediction_details', 'query_start_ts', 'query_end_ts', 'status', 'llm_error', 'llm_response_metadata', 'predicted_answer', 'llm_predicted_answer_raw', 'answer_consistency', 'used_focus_frames', 'used_time_anchor_ids', 'uncertainty', 'llm_evidence_selection', 'question_route_count_set_lines', 'question_route_temporal_interval_lines', 'temporal_span_candidate_lines', 'temporal_span_candidates', 'match_count', 'global_scope_score', 'query_family', 'candidate_strategy', 'candidate_count', 'scene_seed_count', 'retrieval_budget', 'seed_limit', 'max_hops']
    if include_position_diagnostics:
        insert_at = fieldnames.index('query_start_ts')
        fieldnames.insert(insert_at, 'position_prediction_diagnostics')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {name: item.get(name, '') for name in fieldnames}
            row['used_focus_frames'] = json.dumps(row['used_focus_frames'], ensure_ascii=False)
            row['used_time_anchor_ids'] = json.dumps(row['used_time_anchor_ids'], ensure_ascii=False)
            row['uncertainty'] = json.dumps(row['uncertainty'], ensure_ascii=False)
            row['llm_evidence_selection'] = json.dumps(row['llm_evidence_selection'], ensure_ascii=False)
            row['question_route_count_set_lines'] = json.dumps(row['question_route_count_set_lines'], ensure_ascii=False)
            row['question_route_temporal_interval_lines'] = json.dumps(row['question_route_temporal_interval_lines'], ensure_ascii=False)
            row['temporal_span_candidate_lines'] = json.dumps(row['temporal_span_candidate_lines'], ensure_ascii=False)
            row['temporal_span_candidates'] = json.dumps(row['temporal_span_candidates'], ensure_ascii=False)
            row['llm_response_metadata'] = json.dumps(row['llm_response_metadata'], ensure_ascii=False)
            row['answer_consistency'] = json.dumps(row['answer_consistency'], ensure_ascii=False)
            row['official_format_prediction_details'] = json.dumps(row['official_format_prediction_details'], ensure_ascii=False)
            if include_position_diagnostics:
                row['position_prediction_diagnostics'] = json.dumps(position_prediction_diagnostics(item), ensure_ascii=False)
            writer.writerow(row)

def write_markdown(path: Path, results: list[dict[str, Any]], *, seq_id: str) -> None:
    lines = [f'# NaVQA Seq{seq_id} Official QA on Canonical Hybrid Retrieval', '', f'- questions: {len(results)}', f"- live_validated: {sum((1 for item in results if item.get('status') == 'live_validated'))}", '- pipeline: canonical temporal memory graph -> canonical graph hybrid retrieval -> evidence bundle -> LLM verbalization', '- retrieval backend: canonical_graph_hybrid', '']
    for index, item in enumerate(results, start=1):
        lines.extend([f"## {index}. {item['question']}", '', f"- UUID: `{item['uuid']}`", f"- type/category: `{item['question_type']}` / `{item['question_category']}`", f"- official answer: {item['official_answer'] or '(empty / position placeholder)'}", f"- official eval label: `{item.get('official_evaluation_key')}` = {item.get('official_evaluation_value') or '(unmaterialized)'} ({item.get('official_evaluation_semantics')})", f"- official-format prediction: {item.get('official_format_prediction') or '(empty)'} ({item.get('official_format_prediction_source')})", f"- status: `{item['status']}`", f"- llm error: {item.get('llm_error') or '(none)'}", f"- retrieval plan: global_scope={item.get('global_scope_score')}, family={item.get('query_family')}, strategy={item.get('candidate_strategy')}, candidates={item.get('candidate_count')}, scene_seeds={item.get('scene_seed_count')}, budget={item.get('retrieval_budget')}, seed_limit={item.get('seed_limit')}, max_hops={item.get('max_hops')}", f"- route backbone: {item.get('route_backbone_id') or '(none)'}, unit={item.get('route_candidate_unit') or '(none)'}, reason={item.get('route_backbone_reason') or '(none)'}", f"- predicted: {item['predicted_answer']}", f"- focus frames: `{item['used_focus_frames']}`", f"- time anchors: `{item.get('used_time_anchor_ids', [])}`"])
        consistency = item.get('answer_consistency') or {}
        if isinstance(consistency, dict) and consistency.get('status') not in {None, '', 'not_checked', 'consistent', 'no_explicit_raw_numeric_duration'}:
            lines.append(f"- answer consistency: `{consistency.get('status')}`")
        uncertainty = item.get('uncertainty') or []
        if uncertainty:
            lines.append(f"- uncertainty: {'; '.join((str(x) for x in uncertainty))}")
        raw_answer = str(item.get('llm_predicted_answer_raw') or '').strip()
        if raw_answer and raw_answer != str(item.get('predicted_answer') or '').strip():
            lines.append(f'- raw LLM answer: {raw_answer}')
        prediction_details = item.get('official_format_prediction_details') or {}
        if prediction_details:
            lines.append(f'- official-format prediction details: `{json.dumps(prediction_details, ensure_ascii=False)}`')
        evidence = item.get('evidence_lines') or []
        if evidence:
            lines.append('- evidence:')
            lines.extend((f'  - {line}' for line in evidence[:3]))
        selected_evidence = item.get('llm_selected_evidence_lines') or []
        if selected_evidence:
            lines.append('- LLM selected evidence:')
            lines.extend((f'  - {line}' for line in selected_evidence[:3]))
        answer_candidates = item.get('answer_candidate_lines') or []
        if answer_candidates:
            lines.append('- answer candidates:')
            lines.extend((f'  - {line}' for line in answer_candidates[:3]))
        atomic_facts = item.get('atomic_answer_fact_lines') or []
        if atomic_facts:
            lines.append('- atomic answer facts:')
            lines.extend((f'  - {line}' for line in atomic_facts[:3]))
        unified_evidence = item.get('unified_evidence_lines') or []
        if unified_evidence:
            lines.append('- unified evidence:')
            lines.extend((f'  - {line}' for line in unified_evidence[:3]))
        visual_evidence = item.get('visual_faiss_evidence_lines') or []
        if visual_evidence:
            lines.append('- visual FAISS evidence:')
            lines.extend((f'  - {line}' for line in visual_evidence[:3]))
        entity_packs = item.get('visual_entity_candidate_pack_lines') or []
        if entity_packs:
            lines.append('- visual FAISS entity candidate packs:')
            lines.extend((f'  - {line}' for line in entity_packs[:3]))
        count_set_lines = item.get('question_route_count_set_lines') or []
        if count_set_lines:
            lines.append('- question route count-set packs:')
            lines.extend((f'  - {line}' for line in count_set_lines[:3]))
        visual_ger_bound = item.get('visual_faiss_ger_bound_fact_lines') or []
        if visual_ger_bound:
            lines.append('- visual FAISS GER-like bound facts:')
            lines.extend((f'  - {line}' for line in visual_ger_bound[:3]))
        object_local_facts = item.get('visual_faiss_object_local_fact_lines') or []
        if object_local_facts:
            lines.append('- visual FAISS object-local facts:')
            lines.extend((f'  - {line}' for line in object_local_facts[:3]))
        local_subgraph_facts = item.get('visual_faiss_local_subgraph_fact_lines') or []
        if local_subgraph_facts:
            lines.append('- visual FAISS local-subgraph facts:')
            lines.extend((f'  - {line}' for line in local_subgraph_facts[:3]))
        visual_fragment_bridge = item.get('visual_faiss_fragment_bridge_lines') or []
        if visual_fragment_bridge:
            lines.append('- visual FAISS Graph fragment-bridge facts:')
            lines.extend((f'  - {line}' for line in visual_fragment_bridge[:3]))
        visual_ger_evidence = item.get('visual_faiss_ger_evidence_lines') or []
        if visual_ger_evidence:
            lines.append('- visual FAISS GER-like evidence:')
            lines.extend((f'  - {line}' for line in visual_ger_evidence[:3]))
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    args = build_parser().parse_args()
    if args.experimental_visual_faiss_fused_evidence and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-fused-evidence requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_question_router_hybrid_backbone and (not args.experimental_question_router):
        raise SystemExit('--experimental-question-router-hybrid-backbone requires --experimental-question-router.')
    if args.experimental_question_router_specialized_backbone and (not args.experimental_question_router):
        raise SystemExit('--experimental-question-router-specialized-backbone requires --experimental-question-router.')
    if args.experimental_question_router_mainline and (not args.experimental_question_router):
        raise SystemExit('--experimental-question-router-mainline requires --experimental-question-router.')
    if args.experimental_question_router_specialized_backbone and args.experimental_question_router_hybrid_backbone:
        raise SystemExit('--experimental-question-router-specialized-backbone and --experimental-question-router-hybrid-backbone are separate routes.')
    if args.experimental_question_router_mainline and (args.experimental_question_router_hybrid_backbone or args.experimental_question_router_specialized_backbone):
        raise SystemExit('--experimental-question-router-mainline, --experimental-question-router-specialized-backbone, and --experimental-question-router-hybrid-backbone are separate routes.')
    if (args.experimental_question_router_hybrid_backbone or args.experimental_question_router_specialized_backbone or args.experimental_question_router_mainline) and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('Question-router visual backbones require --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_graph_anchors and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-graph-anchors requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_ger_fusion and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-ger-fusion requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_ger_binding and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-ger-binding requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_frame_context and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-frame-context requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_frame_context and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-frame-context requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_structural_binding and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-structural-binding requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_attribute_owner and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-attribute-owner requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_owner_minimal_prompt and (not args.experimental_visual_faiss_attribute_owner):
        raise SystemExit('--experimental-owner-minimal-prompt requires --experimental-visual-faiss-attribute-owner.')
    if args.experimental_visual_faiss_fragment_bridge and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-fragment-bridge requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_object_local_facts and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-object-local-facts requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_local_subgraph and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-local-subgraph requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_local_subgraph_anchor_only and (not args.experimental_visual_faiss_local_subgraph):
        raise SystemExit('--experimental-visual-faiss-local-subgraph-anchor-only requires --experimental-visual-faiss-local-subgraph.')
    if args.experimental_visual_faiss_balanced_graph_visual and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-balanced-graph-visual requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_answerable_candidate_packs and (not args.experimental_visual_faiss_object_local_facts):
        raise SystemExit('--experimental-answerable-candidate-packs requires --experimental-visual-faiss-object-local-facts.')
    if args.experimental_visual_faiss_unified_candidates and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-unified-candidates requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_unified_candidates and args.experimental_visual_faiss_ger_binding:
        raise SystemExit('--experimental-visual-faiss-unified-candidates and --experimental-visual-faiss-ger-binding are separate GER-like modes.')
    if args.experimental_visual_faiss_evidence_layer and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('--experimental-visual-faiss-evidence-layer requires --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_role_aware_evidence and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('--experimental-visual-faiss-role-aware-evidence requires --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_role_aware_aux and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-role-aware-aux requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_role_aware_aux and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('--experimental-visual-faiss-role-aware-aux requires --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_role_aware_compression and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_role_aware_compression and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_role_aware_compression and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_entity_pack and (not (args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall)):
        raise SystemExit('--experimental-visual-faiss-entity-pack requires --experimental-visual-query-faiss-recall or --experimental-visual-faiss-evidence.')
    if args.experimental_visual_faiss_entity_pack_semantic_align and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-entity-pack-semantic-align requires --experimental-visual-faiss-entity-pack.')
    if args.experimental_visual_faiss_entity_pack_semantic_gated_role and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-entity-pack-semantic-gated-role requires --experimental-visual-faiss-entity-pack.')
    if args.experimental_visual_faiss_score_contract and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-score-contract requires --experimental-visual-faiss-entity-pack.')
    if args.experimental_visual_faiss_evidence_layer_constraint_binding and (not args.experimental_visual_faiss_evidence_layer):
        raise SystemExit('--experimental-visual-faiss-evidence-layer-constraint-binding requires --experimental-visual-faiss-evidence-layer.')
    if args.experimental_visual_faiss_graph_anchors and args.experimental_visual_faiss_fused_evidence:
        raise SystemExit('--experimental-visual-faiss-graph-anchors and --experimental-visual-faiss-fused-evidence are separate FAISS fusion modes.')
    visual_fusion_mode_count = sum((bool(value) for value in [args.experimental_visual_faiss_fused_evidence, args.experimental_visual_faiss_graph_anchors, args.experimental_visual_faiss_ger_fusion, args.experimental_visual_faiss_evidence_layer, args.experimental_visual_faiss_role_aware_evidence, args.experimental_visual_faiss_entity_pack, args.experimental_question_router_hybrid_backbone, args.experimental_question_router_specialized_backbone, args.experimental_question_router_mainline]))
    if visual_fusion_mode_count > 1:
        raise SystemExit('Choose only one visual FAISS fusion mode: --experimental-visual-faiss-fused-evidence, --experimental-visual-faiss-graph-anchors, or --experimental-visual-faiss-ger-fusion, --experimental-visual-faiss-evidence-layer, or --experimental-visual-faiss-role-aware-evidence, or --experimental-visual-faiss-entity-pack, or --experimental-question-router-hybrid-backbone, or --experimental-question-router-specialized-backbone, or --experimental-question-router-mainline.')
    if args.experimental_frame_faiss_fused_evidence and (not args.experimental_frame_faiss_query_recall):
        raise SystemExit('--experimental-frame-faiss-fused-evidence requires --experimental-frame-faiss-query-recall.')
    if args.experimental_frame_faiss_graph_anchors and (not args.experimental_frame_faiss_query_recall):
        raise SystemExit('--experimental-frame-faiss-graph-anchors requires --experimental-frame-faiss-query-recall.')
    if args.experimental_frame_faiss_graph_anchors and args.experimental_frame_faiss_fused_evidence:
        raise SystemExit('--experimental-frame-faiss-graph-anchors and --experimental-frame-faiss-fused-evidence are separate FAISS fusion modes.')
    if args.experimental_merged_frame_faiss_fused_evidence and (not args.experimental_merged_frame_faiss_query_recall):
        raise SystemExit('--experimental-merged-frame-faiss-fused-evidence requires --experimental-merged-frame-faiss-query-recall.')
    if args.experimental_merged_frame_faiss_graph_anchors and (not args.experimental_merged_frame_faiss_query_recall):
        raise SystemExit('--experimental-merged-frame-faiss-graph-anchors requires --experimental-merged-frame-faiss-query-recall.')
    if args.experimental_merged_frame_faiss_graph_anchors and args.experimental_merged_frame_faiss_fused_evidence:
        raise SystemExit('--experimental-merged-frame-faiss-graph-anchors and --experimental-merged-frame-faiss-fused-evidence are separate FAISS fusion modes.')
    if args.experimental_frame_window_faiss_graph_anchors and (not args.experimental_frame_window_faiss_query_recall):
        raise SystemExit('--experimental-frame-window-faiss-graph-anchors requires --experimental-frame-window-faiss-query-recall.')
    if args.experimental_frame_window_faiss_query_recall and args.experimental_merged_frame_faiss_query_recall:
        raise SystemExit('--experimental-frame-window-faiss-query-recall and --experimental-merged-frame-faiss-query-recall are separate MF routes.')
    if args.experimental_frame_window_faiss_query_recall and args.experimental_frame_faiss_query_recall:
        raise SystemExit('--experimental-frame-window-faiss-query-recall already uses frame FAISS internally; do not combine it with --experimental-frame-faiss-query-recall.')
    layout = default_navqa_sequence_layout(seq_id=args.seq_id, cam=args.cam, stride=args.stride, layout_root=args.layout_root)
    output_json = args.output_json or layout.qa_results_json
    output_csv = args.output_csv or layout.qa_results_csv
    output_md = args.output_md or layout.qa_results_md
    graph_path = resolve_repo_input_path(args.canonical_graph_json)
    csv_path = resolve_repo_input_path(args.csv)
    questions = load_questions(csv_path, args.seq_id)
    include_types = args.include_types
    if args.non_position_only:
        include_types = list(NON_POSITION_QA_TYPES)
    selected = filter_questions_by_type(questions, include_types=include_types, exclude_types=args.exclude_types)
    selected = selected[args.offset:]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit(f'No questions selected for seq_id={args.seq_id!r}')
    question_ir_records = load_question_ir_records(args.question_ir_json)
    question_time_metadata = load_question_time_metadata(args.qa_unfilled_json)
    frame_time_index_path = args.frame_time_index
    if frame_time_index_path is None:
        frame_time_index_path = default_frame_time_index_path(args.seq_id, cam=args.cam, stride=args.stride, layout_root=args.layout_root)
    frame_time_index = load_frame_time_index(frame_time_index_path)
    selected_has_position = has_position_questions(selected)
    position_reference = load_position_reference(args.position_reference_json) if selected_has_position else None
    position_labels = load_position_labels(position_reference) if selected_has_position else {}
    position_windows = load_pose_timeline(pose_timeline_json=args.pose_timeline_json, dense_pose=args.dense_pose, timestamps=args.timestamps, seconds_per_caption=args.seconds_per_caption, seq_id=args.seq_id, cam=args.cam, stride=args.stride, layout_root=args.layout_root) if selected_has_position else []
    require_pose_timeline_for_position_questions(rows=selected, position_windows=position_windows, seq_id=args.seq_id, cam=args.cam, stride=args.stride, layout_root=args.layout_root)
    print(f'graph={graph_path}', flush=True)
    print(f'csv={csv_path}', flush=True)
    print(f'seq_id={args.seq_id}', flush=True)
    print(f'questions={len(selected)}', flush=True)
    graph = CanonicalTemporalMemoryGraph.model_validate_json(graph_path.read_text(encoding='utf-8'))
    bounds = graph_time_bounds(graph)
    if bounds is not None:
        print(f'graph_time_bounds={bounds[0]}->{bounds[1]}', flush=True)
    if frame_time_index:
        print(f"frame_time_index={frame_time_index.get('source_path')} frames={len(frame_time_index.get('ordered_frames', []))}", flush=True)
    if question_ir_records:
        matched_ir = sum((1 for row in selected if str(row.get(FIELD_UUID, '')).strip() in question_ir_records))
        print(f'question_ir={args.question_ir_json} matched={matched_ir}/{len(selected)}', flush=True)
    synonym_lexicon: SynonymLexicon | None = None
    if args.experimental_synonym_lexicon:
        synonym_lexicon = load_synonym_lexicon(args.synonym_lexicon_json)
        print(f"synonym_lexicon=enabled path={synonym_lexicon.path} entries={len(synonym_lexicon.entries)} levels={','.join(synonym_lexicon.levels)}", flush=True)
    graph_qa_config_overrides: dict[str, object] = {}
    if args.llm_evidence_rerank:
        graph_qa_config_overrides.update({'evidence_rerank_enabled': True, 'evidence_rerank_max_candidates': args.llm_evidence_rerank_max_candidates, 'evidence_rerank_max_selected': args.llm_evidence_rerank_max_selected})
    if args.answer_candidate_extraction:
        graph_qa_config_overrides.update({'answer_candidate_extraction_enabled': True, 'answer_candidate_extraction_max_candidates': args.answer_candidate_extraction_max_candidates, 'answer_candidate_extraction_max_tokens': args.answer_candidate_extraction_max_tokens})
    if args.experimental_balanced_evidence_candidates:
        graph_qa_config_overrides.update({'experimental_balanced_evidence_candidates': True})
    if args.experimental_answerable_candidate_packs:
        graph_qa_config_overrides.update({'experimental_answerable_candidate_packs': True, 'answerable_candidate_pack_max_items': args.answerable_candidate_pack_max_items})
    if args.experimental_unified_evidence_layer:
        graph_qa_config_overrides.update({'experimental_unified_evidence_layer': True, 'unified_evidence_max_items': args.unified_evidence_max_items, 'unified_evidence_graph_quota': args.unified_evidence_graph_quota, 'unified_evidence_visual_quota': args.unified_evidence_visual_quota, 'unified_evidence_frame_quota': args.unified_evidence_frame_quota})
    if args.experimental_atomic_answer_facts:
        graph_qa_config_overrides.update({'experimental_atomic_answer_facts': True, 'atomic_answer_fact_max_items': args.atomic_answer_fact_max_items})
    if args.experimental_owner_minimal_prompt:
        graph_qa_config_overrides.update({'prompt_style': 'owner_minimal'})
    if not graph_qa_config_overrides:
        graph_qa_config_overrides = None
    if args.llm_evidence_rerank:
        print(f'llm_evidence_rerank=enabled max_candidates={args.llm_evidence_rerank_max_candidates} max_selected={args.llm_evidence_rerank_max_selected}', flush=True)
    if args.answer_candidate_extraction:
        print(f'answer_candidate_extraction=enabled max_candidates={args.answer_candidate_extraction_max_candidates} max_tokens={args.answer_candidate_extraction_max_tokens}', flush=True)
    if args.experimental_ir_evidence_query_expansion:
        print('experimental_ir_evidence_query_expansion=enabled', flush=True)
    if args.experimental_question_router:
        print('experimental_question_router=enabled', flush=True)
    if args.experimental_question_router_hybrid_backbone:
        print(f'experimental_question_router_hybrid_backbone=enabled ir_qa_id={QUESTION_ROUTER_HYBRID_IR_QA_ID}', flush=True)
    if args.experimental_question_router_specialized_backbone:
        print(f'experimental_question_router_specialized_backbone=enabled ir_qa_id={QUESTION_ROUTER_SPECIALIZED_IR_QA_ID}', flush=True)
    if args.experimental_question_router_mainline:
        print(f'experimental_question_router_mainline=enabled ir_qa_id={QUESTION_ROUTER_MAINLINE_IR_QA_ID}', flush=True)
    if args.experimental_balanced_evidence_candidates:
        print('experimental_balanced_evidence_candidates=enabled', flush=True)
    if args.experimental_answerable_candidate_packs:
        print(f'experimental_answerable_candidate_packs=enabled max_items={args.answerable_candidate_pack_max_items}', flush=True)
    if args.experimental_unified_evidence_layer:
        print(f'experimental_unified_evidence_layer=enabled max_items={args.unified_evidence_max_items} graph_quota={args.unified_evidence_graph_quota} visual_quota={args.unified_evidence_visual_quota} frame_quota={args.unified_evidence_frame_quota}', flush=True)
    if args.experimental_atomic_answer_facts:
        print(f'experimental_atomic_answer_facts=enabled max_items={args.atomic_answer_fact_max_items}', flush=True)
    visual_faiss_grounder: VisualFaissGrounder | None = None
    visual_faiss_config: VisualFaissConfig | None = None
    frame_faiss_grounder: FrameFaissGrounder | None = None
    frame_faiss_config: FrameFaissConfig | None = None
    merged_frame_faiss_grounder: MergedFrameFaissGrounder | None = None
    merged_frame_faiss_config: MergedFrameFaissConfig | None = None
    frame_window_faiss_grounder: FrameWindowFaissGrounder | None = None
    frame_window_faiss_config: FrameWindowFaissConfig | None = None
    if args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall:
        visual_faiss_grounder = VisualFaissGrounder(index_dir=args.visual_faiss_index_dir, index_name=args.visual_faiss_index_name)
        visual_faiss_config = VisualFaissConfig(index_dir=args.visual_faiss_index_dir, index_name=args.visual_faiss_index_name, seed_limit=args.visual_faiss_seed_limit, top_k=args.visual_faiss_top_k, max_lines=args.visual_faiss_max_lines, same_sequence_only=not args.visual_faiss_cross_sequence, query_phrase_limit=args.visual_faiss_query_phrase_limit, query_recall_top_k=args.visual_faiss_query_top_k, query_recall_search_k=args.visual_faiss_query_search_k, query_recall_min_score=args.visual_faiss_query_min_score, ger_object_limit=args.visual_faiss_ger_object_limit, ger_context_radius=args.visual_faiss_ger_context_radius, ger_context_limit=args.visual_faiss_ger_context_limit, frame_context_enabled=bool(args.experimental_visual_faiss_frame_context), evidence_layer_sqlite=args.visual_faiss_evidence_layer_sqlite, evidence_layer_object_limit=args.visual_faiss_evidence_layer_object_limit, evidence_layer_neighbor_limit=args.visual_faiss_evidence_layer_neighbor_limit, role_aware_object_limit=args.visual_faiss_role_aware_object_limit, role_aware_context_limit=args.visual_faiss_role_aware_context_limit, role_aware_compression_enabled=bool(args.experimental_visual_faiss_role_aware_compression), entity_pack_enabled=bool(args.experimental_visual_faiss_entity_pack), entity_pack_semantic_align_enabled=bool(args.experimental_visual_faiss_entity_pack_semantic_align), entity_pack_semantic_gated_role_enabled=bool(args.experimental_visual_faiss_entity_pack_semantic_gated_role), entity_pack_score_contract_enabled=bool(args.experimental_visual_faiss_score_contract))
        visual_description = visual_faiss_grounder.describe()
        print(f"experimental_visual_faiss=enabled index_ntotal={visual_description.get('index_ntotal')} focus_seed={bool(args.experimental_visual_faiss_evidence)} query_recall={bool(args.experimental_visual_query_faiss_recall)} fused={bool(args.experimental_visual_faiss_fused_evidence)} graph_anchors={bool(args.experimental_visual_faiss_graph_anchors)} ger_fusion={bool(args.experimental_visual_faiss_ger_fusion)} ger_binding={bool(args.experimental_visual_faiss_ger_binding)} frame_context={bool(args.experimental_visual_faiss_frame_context)} structural_binding={bool(args.experimental_visual_faiss_structural_binding)} attribute_owner={bool(args.experimental_visual_faiss_attribute_owner)} fragment_bridge={bool(args.experimental_visual_faiss_fragment_bridge)} object_local_facts={bool(args.experimental_visual_faiss_object_local_facts)} local_subgraph={bool(args.experimental_visual_faiss_local_subgraph)} local_subgraph_anchor_only={bool(args.experimental_visual_faiss_local_subgraph_anchor_only)} balanced_graph_visual={bool(args.experimental_visual_faiss_balanced_graph_visual)} unified_candidates={bool(args.experimental_visual_faiss_unified_candidates)} evidence_layer={bool(args.experimental_visual_faiss_evidence_layer)} evidence_layer_constraint_binding={bool(args.experimental_visual_faiss_evidence_layer_constraint_binding)} role_aware={bool(args.experimental_visual_faiss_role_aware_evidence)} role_aware_aux={bool(args.experimental_visual_faiss_role_aware_aux)} role_aware_compression={bool(args.experimental_visual_faiss_role_aware_compression)} entity_pack={bool(args.experimental_visual_faiss_entity_pack)} question_router_hybrid_backbone={bool(args.experimental_question_router_hybrid_backbone)} question_router_specialized_backbone={bool(args.experimental_question_router_specialized_backbone)} question_router_mainline={bool(args.experimental_question_router_mainline)} entity_pack_semantic_align={bool(args.experimental_visual_faiss_entity_pack_semantic_align)} entity_pack_semantic_gated_role={bool(args.experimental_visual_faiss_entity_pack_semantic_gated_role)} score_contract={bool(args.experimental_visual_faiss_score_contract)} seed_limit={visual_faiss_config.seed_limit} top_k={visual_faiss_config.top_k} query_top_k={visual_faiss_config.query_recall_top_k} ger_object_limit={visual_faiss_config.ger_object_limit} evidence_layer_object_limit={visual_faiss_config.evidence_layer_object_limit} evidence_layer_neighbor_limit={visual_faiss_config.evidence_layer_neighbor_limit} role_aware_object_limit={visual_faiss_config.role_aware_object_limit} role_aware_context_limit={visual_faiss_config.role_aware_context_limit} same_sequence_only={visual_faiss_config.same_sequence_only}", flush=True)
    if args.experimental_frame_faiss_query_recall:
        frame_faiss_grounder = FrameFaissGrounder(index_dir=args.frame_faiss_index_dir, index_name=args.frame_faiss_index_name)
        frame_faiss_config = FrameFaissConfig(index_dir=args.frame_faiss_index_dir, index_name=args.frame_faiss_index_name, query_text_limit=args.frame_faiss_query_text_limit, top_k=args.frame_faiss_top_k, search_k=args.frame_faiss_search_k, min_score=args.frame_faiss_min_score, same_sequence_only=not args.frame_faiss_cross_sequence, cluster_max_frame_gap=args.frame_faiss_cluster_max_frame_gap, cluster_max_sec_gap=args.frame_faiss_cluster_max_sec_gap, max_clusters=args.frame_faiss_max_clusters, max_lines=args.frame_faiss_max_lines)
        frame_description = frame_faiss_grounder.describe()
        print(f"experimental_frame_faiss=enabled index_ntotal={frame_description.get('index_ntotal')} query_recall={bool(args.experimental_frame_faiss_query_recall)} fused={bool(args.experimental_frame_faiss_fused_evidence)} graph_anchors={bool(args.experimental_frame_faiss_graph_anchors)} top_k={frame_faiss_config.top_k} search_k={frame_faiss_config.search_k} max_clusters={frame_faiss_config.max_clusters} same_sequence_only={frame_faiss_config.same_sequence_only}", flush=True)
    if args.experimental_merged_frame_faiss_query_recall:
        merged_frame_faiss_grounder = MergedFrameFaissGrounder(index_dir=args.merged_frame_faiss_index_dir, index_name=args.merged_frame_faiss_index_name)
        merged_frame_faiss_config = MergedFrameFaissConfig(index_dir=args.merged_frame_faiss_index_dir, index_name=args.merged_frame_faiss_index_name, query_text_limit=args.merged_frame_faiss_query_text_limit, top_k=args.merged_frame_faiss_top_k, search_k=args.merged_frame_faiss_search_k, min_score=args.merged_frame_faiss_min_score, same_sequence_only=not args.merged_frame_faiss_cross_sequence, cluster_max_frame_gap=args.merged_frame_faiss_cluster_max_frame_gap, cluster_max_sec_gap=args.merged_frame_faiss_cluster_max_sec_gap, max_clusters=args.merged_frame_faiss_max_clusters, max_lines=args.merged_frame_faiss_max_lines)
        merged_description = merged_frame_faiss_grounder.describe()
        print(f"experimental_merged_frame_faiss=enabled index_ntotal={merged_description.get('index_ntotal')} query_recall={bool(args.experimental_merged_frame_faiss_query_recall)} fused={bool(args.experimental_merged_frame_faiss_fused_evidence)} graph_anchors={bool(args.experimental_merged_frame_faiss_graph_anchors)} top_k={merged_frame_faiss_config.top_k} search_k={merged_frame_faiss_config.search_k} max_clusters={merged_frame_faiss_config.max_clusters} same_sequence_only={merged_frame_faiss_config.same_sequence_only}", flush=True)
    if args.experimental_frame_window_faiss_query_recall:
        frame_window_faiss_grounder = FrameWindowFaissGrounder(index_dir=args.frame_faiss_index_dir, index_name=args.frame_faiss_index_name)
        frame_window_faiss_config = FrameWindowFaissConfig(frame_config=FrameFaissConfig(index_dir=args.frame_faiss_index_dir, index_name=args.frame_faiss_index_name, query_text_limit=args.frame_faiss_query_text_limit, top_k=args.frame_faiss_top_k, search_k=args.frame_faiss_search_k, min_score=args.frame_faiss_min_score, same_sequence_only=not args.frame_faiss_cross_sequence, cluster_max_frame_gap=args.frame_faiss_cluster_max_frame_gap, cluster_max_sec_gap=args.frame_faiss_cluster_max_sec_gap, max_clusters=args.frame_faiss_max_clusters, max_lines=args.frame_faiss_max_lines), window_size_sec=args.frame_window_faiss_window_size_sec, window_stride_sec=args.frame_window_faiss_window_stride_sec, top_k=args.frame_faiss_top_k, search_k=args.frame_faiss_search_k, max_clusters=args.merged_frame_faiss_max_clusters, max_lines=args.merged_frame_faiss_max_lines, max_frames_per_window=args.frame_window_faiss_max_frames_per_window)
        frame_window_description = frame_window_faiss_grounder.describe()
        source_frame = frame_window_description.get('source_frame_faiss', {})
        print(f"experimental_frame_window_faiss=enabled index_ntotal={source_frame.get('index_ntotal')} query_recall={bool(args.experimental_frame_window_faiss_query_recall)} graph_anchors={bool(args.experimental_frame_window_faiss_graph_anchors)} window_size_sec={frame_window_faiss_config.window_size_sec} window_stride_sec={frame_window_faiss_config.window_stride_sec} max_clusters={frame_window_faiss_config.max_clusters}", flush=True)
    retrieval_session = CanonicalHybridRetrievalSession(graph, projection_cache_path=args.projection_cache, domain_profile=args.domain_profile)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1 + args.offset):
        print(f'[{len(results) + 1}/{len(selected)}] Q{index}: {output_question(row)}', flush=True)
        result = run_question(graph, row, answer_mode=args.answer_mode, config=args.config, retrieval_limit=args.retrieval_limit, question_ir_records=question_ir_records, question_time_metadata=question_time_metadata, position_labels=position_labels, position_windows=position_windows, frame_time_index=frame_time_index, projection_cache_path=args.projection_cache, retrieval_session=retrieval_session, graph_bounds=bounds, current_window_sec=args.current_window_sec, use_question_window=not args.disable_question_window, domain_profile=args.domain_profile, graph_qa_config_overrides=graph_qa_config_overrides, include_ir_evidence_needs_in_query=args.experimental_ir_evidence_query_expansion, synonym_lexicon=synonym_lexicon, experimental_question_router=args.experimental_question_router, experimental_question_router_hybrid_backbone=args.experimental_question_router_hybrid_backbone, experimental_question_router_specialized_backbone=args.experimental_question_router_specialized_backbone, experimental_question_router_mainline=args.experimental_question_router_mainline, visual_faiss_grounder=visual_faiss_grounder, visual_faiss_config=visual_faiss_config, visual_faiss_focus_seed_evidence=args.experimental_visual_faiss_evidence, visual_faiss_query_recall=args.experimental_visual_query_faiss_recall, visual_faiss_fused_evidence=args.experimental_visual_faiss_fused_evidence, visual_faiss_graph_anchors=args.experimental_visual_faiss_graph_anchors, visual_faiss_ger_fusion=args.experimental_visual_faiss_ger_fusion, visual_faiss_ger_binding=args.experimental_visual_faiss_ger_binding, visual_faiss_frame_context=args.experimental_visual_faiss_frame_context, visual_faiss_structural_binding=args.experimental_visual_faiss_structural_binding, visual_faiss_attribute_owner=args.experimental_visual_faiss_attribute_owner, visual_faiss_fragment_bridge=args.experimental_visual_faiss_fragment_bridge, visual_faiss_object_local_facts=args.experimental_visual_faiss_object_local_facts, visual_faiss_local_subgraph=args.experimental_visual_faiss_local_subgraph, visual_faiss_local_subgraph_anchor_only=args.experimental_visual_faiss_local_subgraph_anchor_only, visual_faiss_balanced_graph_visual=args.experimental_visual_faiss_balanced_graph_visual, visual_faiss_unified_candidates=args.experimental_visual_faiss_unified_candidates, visual_faiss_evidence_layer=args.experimental_visual_faiss_evidence_layer, visual_faiss_evidence_layer_constraint_binding=args.experimental_visual_faiss_evidence_layer_constraint_binding, visual_faiss_role_aware_evidence=args.experimental_visual_faiss_role_aware_evidence, visual_faiss_role_aware_aux=args.experimental_visual_faiss_role_aware_aux, visual_faiss_role_aware_compression=args.experimental_visual_faiss_role_aware_compression, visual_faiss_entity_pack=args.experimental_visual_faiss_entity_pack, frame_faiss_grounder=frame_faiss_grounder, frame_faiss_config=frame_faiss_config, frame_faiss_query_recall=args.experimental_frame_faiss_query_recall, frame_faiss_fused_evidence=args.experimental_frame_faiss_fused_evidence, frame_faiss_graph_anchors=args.experimental_frame_faiss_graph_anchors, merged_frame_faiss_grounder=merged_frame_faiss_grounder, merged_frame_faiss_config=merged_frame_faiss_config, merged_frame_faiss_query_recall=args.experimental_merged_frame_faiss_query_recall, merged_frame_faiss_fused_evidence=args.experimental_merged_frame_faiss_fused_evidence, merged_frame_faiss_graph_anchors=args.experimental_merged_frame_faiss_graph_anchors, frame_window_faiss_grounder=frame_window_faiss_grounder, frame_window_faiss_config=frame_window_faiss_config, frame_window_faiss_query_recall=args.experimental_frame_window_faiss_query_recall, frame_window_faiss_graph_anchors=args.experimental_frame_window_faiss_graph_anchors)
        print(f"    status={result['status']} answer={result['predicted_answer'][:180]}", flush=True)
        results.append(result)
    if visual_faiss_grounder is not None:
        visual_faiss_grounder.close()
    if frame_faiss_grounder is not None:
        frame_faiss_grounder.close()
    if merged_frame_faiss_grounder is not None:
        merged_frame_faiss_grounder.close()
    if frame_window_faiss_grounder is not None:
        frame_window_faiss_grounder.close()
    if args.text_judge_mode == 'live':
        graph_qa_config = load_domain_config(args.config).graph_qa
        results = attach_text_llm_judgements(results, config=graph_qa_config)
    official_adapter = {'contract': 'navqa_official_qa_eval_adapter_v1', 'inject_into_graph': False, 'note': 'Official-format gold labels are materialized for evaluation output only.'}
    if selected_has_position:
        official_adapter.update({'uses_position_reference': bool(position_reference), 'position_semantics': position_reference.get('position_semantics') if isinstance(position_reference, dict) else None, 'note': 'Official-format gold labels are materialized for evaluation output only. Position pose rendering is loaded only when selected questions include the archived position type.', 'position_prediction_policy': 'graph_event_time_to_pose_timeline' if position_windows else 'llm_answer_text_no_pose_timeline', 'pose_timeline_window_count': len(position_windows)})
    payload = {'source_graph': str(graph_path), 'source_csv': str(csv_path), 'seq_id': args.seq_id, 'pipeline': 'canonical temporal memory graph -> canonical graph hybrid retrieval -> evidence bundle -> optional generic LLM evidence selection -> optional experimental atomic answer facts -> optional generic answer-candidate extraction -> LLM verbalization', 'retrieval_backend': 'canonical_graph_hybrid', 'retrieval_mode': 'temporal', 'answer_mode': args.answer_mode, 'retrieval_limit': args.retrieval_limit, 'llm_evidence_rerank': {'enabled': bool(args.llm_evidence_rerank), 'max_candidates': args.llm_evidence_rerank_max_candidates, 'max_selected': args.llm_evidence_rerank_max_selected}, 'answer_candidate_extraction': {'enabled': bool(args.answer_candidate_extraction), 'max_candidates': args.answer_candidate_extraction_max_candidates, 'max_tokens': args.answer_candidate_extraction_max_tokens, 'scope': 'all_question_types', 'experimental': True}, 'experiments': {'ir_evidence_query_expansion': bool(args.experimental_ir_evidence_query_expansion), 'balanced_evidence_candidates': bool(args.experimental_balanced_evidence_candidates), 'answerable_candidate_packs': {'enabled': bool(args.experimental_answerable_candidate_packs), 'max_items': args.answerable_candidate_pack_max_items, 'experimental': True, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}, 'atomic_answer_facts': {'enabled': bool(args.experimental_atomic_answer_facts), 'max_items': args.atomic_answer_fact_max_items, 'experimental': True, 'generation': 'mechanical_from_supplied_evidence'}, 'owner_minimal_prompt': {'enabled': bool(args.experimental_owner_minimal_prompt), 'experimental': True, 'scope': 'qa_verbalization_prompt_only'}, 'synonym_lexicon': {'enabled': bool(args.experimental_synonym_lexicon), 'contract': 'navqa_synonym_l0_l1_v1', 'path': str(args.synonym_lexicon_json), 'levels': ['L0', 'L1'], 'insertion_point': 'question_ir_before_graph_and_faiss_retrieval', 'uses_gold_answers': False, 'uses_question_id_rules': False, 'uses_seq_specific_rules': False}, 'question_router': {'enabled': bool(args.experimental_question_router), 'contract': 'navqa_question_route_v1', 'route_id': QUESTION_ROUTER_MAINLINE_IR_QA_ID if args.experimental_question_router_mainline else QUESTION_ROUTER_SPECIALIZED_IR_QA_ID if args.experimental_question_router_specialized_backbone else QUESTION_ROUTER_HYBRID_IR_QA_ID if args.experimental_question_router_hybrid_backbone else 'IR-QA-F-question-router-v1', 'insertion_point': 'after_question_ir_before_graph_and_faiss_retrieval', 'hybrid_backbone_enabled': bool(args.experimental_question_router_hybrid_backbone), 'mainline_backbone_enabled': bool(args.experimental_question_router_mainline), 'specialized_backbone_enabled': bool(args.experimental_question_router_specialized_backbone or args.experimental_question_router_mainline), 'hybrid_backbone_contract': 'navqa_question_route_specialized_backbone_v1' if args.experimental_question_router_specialized_backbone or args.experimental_question_router_mainline else 'navqa_question_route_hybrid_backbone_v1' if args.experimental_question_router_hybrid_backbone else '', 'uses_gold_answers': False, 'uses_question_id_rules': False, 'uses_seq_specific_rules': False, 'uses_dataset_or_domain_wordlist': False}, 'visual_faiss_evidence': {'enabled': bool(args.experimental_visual_faiss_evidence or args.experimental_visual_query_faiss_recall), 'query_recall_enabled': bool(args.experimental_visual_query_faiss_recall), 'fused_evidence_enabled': bool(args.experimental_visual_faiss_fused_evidence), 'graph_anchor_expansion_enabled': bool(args.experimental_visual_faiss_graph_anchors), 'ger_fusion_enabled': bool(args.experimental_visual_faiss_ger_fusion), 'ger_binding_enabled': bool(args.experimental_visual_faiss_ger_binding), 'frame_context_enabled': bool(args.experimental_visual_faiss_frame_context), 'structural_binding_enabled': bool(args.experimental_visual_faiss_structural_binding), 'attribute_owner_enabled': bool(args.experimental_visual_faiss_attribute_owner), 'fragment_bridge_enabled': bool(args.experimental_visual_faiss_fragment_bridge), 'object_local_facts_enabled': bool(args.experimental_visual_faiss_object_local_facts), 'local_subgraph_enabled': bool(args.experimental_visual_faiss_local_subgraph), 'local_subgraph_anchor_only_enabled': bool(args.experimental_visual_faiss_local_subgraph_anchor_only), 'balanced_graph_visual_enabled': bool(args.experimental_visual_faiss_balanced_graph_visual), 'unified_candidates_enabled': bool(args.experimental_visual_faiss_unified_candidates), 'evidence_layer_enabled': bool(args.experimental_visual_faiss_evidence_layer), 'evidence_layer_constraint_binding_enabled': bool(args.experimental_visual_faiss_evidence_layer_constraint_binding), 'role_aware_evidence_enabled': bool(args.experimental_visual_faiss_role_aware_evidence), 'role_aware_aux_enabled': bool(args.experimental_visual_faiss_role_aware_aux), 'role_aware_compression_enabled': bool(args.experimental_visual_faiss_role_aware_compression), 'question_router_hybrid_backbone_enabled': bool(args.experimental_question_router_hybrid_backbone), 'question_router_specialized_backbone_enabled': bool(args.experimental_question_router_specialized_backbone), 'entity_pack_enabled': bool(args.experimental_visual_faiss_entity_pack), 'entity_pack_semantic_align_enabled': bool(args.experimental_visual_faiss_entity_pack_semantic_align), 'entity_pack_semantic_gated_role_enabled': bool(args.experimental_visual_faiss_entity_pack_semantic_gated_role), 'entity_pack_score_contract_enabled': bool(args.experimental_visual_faiss_score_contract), 'entity_pack_semantic_alignment': {'enabled': bool(args.experimental_visual_faiss_entity_pack_semantic_align or args.experimental_visual_faiss_entity_pack_semantic_gated_role or args.experimental_visual_faiss_score_contract), 'provider': 'siglip_text', 'llm_semantic_alignment': False, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False}, 'entity_pack_semantic_role_gate': {'enabled': bool(args.experimental_visual_faiss_entity_pack_semantic_gated_role), 'threshold': 0.78, 'margin': 0.05, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False}, 'score_contract': {'enabled': bool(args.experimental_visual_faiss_score_contract), 'contract': 'visual_candidate_pack_score_contract_v1', 'candidate_unit': 'candidate_pack', 'fallback_unit': 'fallback_evidence', 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False}, 'entity_pack_policy': 'object FAISS hits are mapped to CG6 entity/tracklet candidate packs; direct attributes, relations, constraints, time span, visual anchors, and neighbor context remain separate', 'index_dir': str(args.visual_faiss_index_dir), 'index_name': args.visual_faiss_index_name, 'seed_limit': args.visual_faiss_seed_limit, 'top_k': args.visual_faiss_top_k, 'max_lines': args.visual_faiss_max_lines, 'same_sequence_only': not args.visual_faiss_cross_sequence, 'query_phrase_limit': args.visual_faiss_query_phrase_limit, 'query_top_k': args.visual_faiss_query_top_k, 'query_search_k': args.visual_faiss_query_search_k, 'query_min_score': args.visual_faiss_query_min_score, 'ger_object_limit': args.visual_faiss_ger_object_limit, 'ger_context_radius': args.visual_faiss_ger_context_radius, 'ger_context_limit': args.visual_faiss_ger_context_limit, 'evidence_layer_sqlite': str(args.visual_faiss_evidence_layer_sqlite), 'evidence_layer_object_limit': args.visual_faiss_evidence_layer_object_limit, 'evidence_layer_neighbor_limit': args.visual_faiss_evidence_layer_neighbor_limit, 'role_aware_object_limit': args.visual_faiss_role_aware_object_limit, 'role_aware_context_limit': args.visual_faiss_role_aware_context_limit}, 'frame_faiss_evidence': {'enabled': bool(args.experimental_frame_faiss_query_recall), 'query_recall_enabled': bool(args.experimental_frame_faiss_query_recall), 'fused_evidence_enabled': bool(args.experimental_frame_faiss_fused_evidence), 'graph_anchor_expansion_enabled': bool(args.experimental_frame_faiss_graph_anchors), 'index_dir': str(args.frame_faiss_index_dir), 'index_name': args.frame_faiss_index_name, 'query_text_limit': args.frame_faiss_query_text_limit, 'top_k': args.frame_faiss_top_k, 'search_k': args.frame_faiss_search_k, 'min_score': args.frame_faiss_min_score, 'same_sequence_only': not args.frame_faiss_cross_sequence, 'cluster_max_frame_gap': args.frame_faiss_cluster_max_frame_gap, 'cluster_max_sec_gap': args.frame_faiss_cluster_max_sec_gap, 'max_clusters': args.frame_faiss_max_clusters, 'max_lines': args.frame_faiss_max_lines}, 'merged_frame_faiss_evidence': {'enabled': bool(args.experimental_merged_frame_faiss_query_recall or args.experimental_frame_window_faiss_query_recall), 'query_recall_enabled': bool(args.experimental_merged_frame_faiss_query_recall or args.experimental_frame_window_faiss_query_recall), 'fused_evidence_enabled': bool(args.experimental_merged_frame_faiss_fused_evidence), 'graph_anchor_expansion_enabled': bool(args.experimental_merged_frame_faiss_graph_anchors or args.experimental_frame_window_faiss_graph_anchors), 'route': 'frame_window_faiss_v2' if args.experimental_frame_window_faiss_query_recall else 'storyboard_merged_frame_faiss_v1', 'index_dir': 'unused_for_frame_window_v2' if args.experimental_frame_window_faiss_query_recall else str(args.merged_frame_faiss_index_dir), 'index_name': 'unused_for_frame_window_v2' if args.experimental_frame_window_faiss_query_recall else args.merged_frame_faiss_index_name, 'query_text_limit': args.merged_frame_faiss_query_text_limit, 'top_k': args.merged_frame_faiss_top_k, 'search_k': args.merged_frame_faiss_search_k, 'min_score': args.merged_frame_faiss_min_score, 'same_sequence_only': not args.merged_frame_faiss_cross_sequence, 'cluster_max_frame_gap': args.merged_frame_faiss_cluster_max_frame_gap, 'cluster_max_sec_gap': args.merged_frame_faiss_cluster_max_sec_gap, 'max_clusters': args.merged_frame_faiss_max_clusters, 'max_lines': args.merged_frame_faiss_max_lines, 'frame_window': {'enabled': bool(args.experimental_frame_window_faiss_query_recall), 'frame_index_dir': str(args.frame_faiss_index_dir), 'frame_index_name': args.frame_faiss_index_name, 'window_size_sec': args.frame_window_faiss_window_size_sec, 'window_stride_sec': args.frame_window_faiss_window_stride_sec, 'max_frames_per_window': args.frame_window_faiss_max_frames_per_window}}}, 'question_processing': {'source': 'llm_question_ir+question_route' if question_ir_records and args.experimental_question_router else 'question_route+rule_question_text' if args.experimental_question_router else 'llm_question_ir' if question_ir_records else 'rule_question_text', 'question_ir_json': str(args.question_ir_json) if args.question_ir_json else '', 'matched_count': sum((1 for row in selected if str(row.get(FIELD_UUID, '')).strip() in question_ir_records)), 'question_router_enabled': bool(args.experimental_question_router)}, 'official_adapter': official_adapter, 'question_count': len(results), 'navqa_official_style_accuracy': navqa_official_style_accuracy_summary(results), 'results': results}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(output_csv, results)
    write_markdown(output_md, results, seq_id=args.seq_id)
    print(f'output_json={output_json.resolve()}', flush=True)
    print(f'output_csv={output_csv.resolve()}', flush=True)
    print(f'output_md={output_md.resolve()}', flush=True)
if __name__ == '__main__':
    main()
