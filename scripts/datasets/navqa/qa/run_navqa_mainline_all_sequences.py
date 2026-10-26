from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.navqa.runtime.paths import NavQASequenceLayout, default_navqa_sequence_layout, normalize_cam, normalize_seq_id, normalize_stride
from vlkg.datasets.navqa.retrieval.question_route_hybrid_backbone import QUESTION_ROUTER_HYBRID_IR_QA_ID, QUESTION_ROUTER_MAINLINE_IR_QA_ID, QUESTION_ROUTER_SPECIALIZED_IR_QA_ID
from vlkg.datasets.navqa.retrieval.question_router import QUESTION_ROUTER_IR_QA_ID
from vlkg.datasets.navqa.retrieval.synonym_lexicon import DEFAULT_SYNONYM_LEXICON_PATH, SYNONYM_ROUTE_IR_QA_ID
FIELD_SEQ_ID = 'Seq ID'
FIELD_UUID = 'UUID'
FIELD_TYPE = 'Type \n(binary, position, time, text)'
NON_POSITION_QA_TYPES = ('binary', 'text', 'time', 'duration')
DEFAULT_SEQUENCE_MANIFEST = Path('configs/datasets/navqa/sequences.json')
DEFAULT_OUTPUT_ROOT = Path('datasets/navqa/runs/new_runs/payload')

@dataclass(frozen=True)
class SequenceSpec:
    seq_id: str
    cam: str
    stride: int | None
    layout_kind: str
    sample_root: Path
    canonical_graph_json: Path
    frame_time_index_json: Path | None
    qa_unfilled_json: Path | None
    question_ir_json: Path | None
    projection_cache: Path | None
    existing_baseline_json: Path | None

    @property
    def display_id(self) -> str:
        stride_text = f'every{self.stride}' if self.stride is not None else 'sampled'
        return f'seq{self.seq_id}_cam{self.cam}_{stride_text}'

@dataclass(frozen=True)
class MainlinePaths:
    output_dir: Path
    question_ir_json: Path
    question_ir_md: Path
    merged_json: Path
    merged_csv: Path
    merged_md: Path
    projection_cache: Path

@dataclass(frozen=True)
class QABatchTask:
    offset: int
    limit: int
    batch_json: Path
    command: list[str]
    api_key_index: int | None = None

@dataclass(frozen=True)
class SequenceRunPlan:
    spec: SequenceSpec
    selected_rows: list[dict[str, str]]
    paths: MainlinePaths
    batch_paths: list[Path]
    batch_ranges_for_sequence: list[tuple[int, int]]
    qa_tasks: list[QABatchTask]

class TeeStream:

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any((getattr(stream, 'isatty', lambda: False)() for stream in self._streams))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the current NaVQA mainline over every available sequence with LLM Question IR, batched QA, resume, retry, and all-sequence summary.')
    parser.add_argument('--csv', type=Path, default=Path('datasets/navqa/source/navqa-coda-every10-v1/payload/official/qa.csv'))
    parser.add_argument('--layout-root', type=Path, default=(REPO_ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa'))
    parser.add_argument('--sequence-manifest', type=Path, default=None)
    parser.add_argument('--no-default-sequence-manifest', action='store_true', help='Ignore the default clean sequence manifest and rely only on unified layout discovery.')
    parser.add_argument('--seq-id', nargs='+', default=None)
    parser.add_argument('--cam', default=None)
    parser.add_argument('--stride', default=None)
    parser.add_argument('--include-types', nargs='+', default=None, help='Optional QA question types to include. Defaults to the current mainline non-position scope: binary text time duration.')
    parser.add_argument('--exclude-types', nargs='+', default=None, help='Optional QA question types to exclude after include-types filtering.')
    parser.add_argument('--no-legacy-seq0', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', default='default.yaml')
    parser.add_argument('--answer-mode', choices=('graph_only', 'mock', 'live'), default='live')
    parser.add_argument('--ir-mode', choices=('live', 'mock'), default='live')
    parser.add_argument('--domain-profile', default='generic')
    parser.add_argument('--retrieval-limit', type=int, default=10)
    parser.add_argument('--experimental-synonym-lexicon', action='store_true', help='Default-off experiment passed to per-sequence QA: expand question IR target/constraint terms with an explicit L0/L1 synonym lexicon before graph and FAISS retrieval.')
    parser.add_argument('--experimental-question-router', action='store_true', help='Default-off experiment passed to per-sequence QA: attach a generic QuestionRoute after question IR and expose route-aware candidate pack metadata without using gold answers, question ids, seq rules, or dataset/domain wordlists.')
    parser.add_argument('--experimental-question-router-hybrid-backbone', action='store_true', help='Default-off experiment passed to per-sequence QA: use QuestionRoute to choose route-specific visual evidence backbones instead of one shared entity-pack score-contract path.')
    parser.add_argument('--experimental-question-router-specialized-backbone', action='store_true', help='Default-off experiment passed to per-sequence QA: QuestionRouter v3 specialized backbone with policy-aware count sets and temporal boundary packs.')
    parser.add_argument('--experimental-question-router-mainline', action='store_true', help='Default-off experiment passed to per-sequence QA: the consolidated QuestionRouter mainline. It reuses the current route-specific specialized backbone table as one stable CG6 IR-QA route.')
    parser.add_argument('--synonym-lexicon-json', type=Path, default=DEFAULT_SYNONYM_LEXICON_PATH, help='L0/L1 synonym lexicon JSON passed to per-sequence QA.')
    evidence_rerank_group = parser.add_mutually_exclusive_group()
    evidence_rerank_group.add_argument('--llm-evidence-rerank', dest='llm_evidence_rerank', action='store_true', default=True, help='Enable the frozen mainline generic LLM evidence-selection pass for every sequence. This is the default. It sees retrieved graph evidence and the question only, not gold answers.')
    evidence_rerank_group.add_argument('--no-llm-evidence-rerank', dest='llm_evidence_rerank', action='store_false', help='Disable LLM evidence selection for ablation/debug runs. This is not the current frozen mainline setting.')
    parser.add_argument('--llm-evidence-rerank-max-candidates', type=int, default=18)
    parser.add_argument('--llm-evidence-rerank-max-selected', type=int, default=6)
    parser.add_argument('--answer-candidate-extraction', action='store_true', help='Experimental default-off generic answer-candidate extraction pass for all question types. The frozen mainline remains unchanged unless this flag is supplied.')
    parser.add_argument('--answer-candidate-extraction-max-candidates', type=int, default=6)
    parser.add_argument('--answer-candidate-extraction-max-tokens', type=int, default=700)
    parser.add_argument('--experimental-ir-evidence-query-expansion', action='store_true', help='Default-off experiment passed to per-sequence QA: append question-IR evidence requirements and graph information needs to retrieval query text.')
    parser.add_argument('--experimental-balanced-evidence-candidates', action='store_true', help='Default-off experiment passed to per-sequence QA: balance LLM evidence selection candidates across evidence source types before truncation.')
    parser.add_argument('--experimental-answerable-candidate-packs', action='store_true', help='Default-off experiment passed to per-sequence QA: before LLM evidence selection, group already retrieved graph/object/visual candidates into compact answerable packs without gold answers, question-type branches, or dataset wordlists.')
    parser.add_argument('--answerable-candidate-pack-max-items', type=int, default=12)
    parser.add_argument('--experimental-unified-evidence-layer', action='store_true', help='Default-off experiment passed to per-sequence QA: normalize graph, object FAISS, and frame/window FAISS evidence into one generic evidence pool before final answering.')
    parser.add_argument('--unified-evidence-max-items', type=int, default=10)
    parser.add_argument('--unified-evidence-graph-quota', type=int, default=5)
    parser.add_argument('--unified-evidence-visual-quota', type=int, default=3)
    parser.add_argument('--unified-evidence-frame-quota', type=int, default=2)
    parser.add_argument('--experimental-atomic-answer-facts', action='store_true', help='Default-off experiment passed to per-sequence QA: mechanically summarize supplied evidence into generic atomic answer facts before final answering.')
    parser.add_argument('--atomic-answer-fact-max-items', type=int, default=8)
    parser.add_argument('--experimental-visual-faiss-evidence', action='store_true', help='Default-off experiment passed to per-sequence QA: add FAISS visual similarity evidence to the answer bundle without changing graph retrieval.')
    parser.add_argument('--experimental-visual-query-faiss-recall', action='store_true', help='Default-off experiment passed to per-sequence QA: encode question/IR visual phrases and retrieve objects from the FAISS index as post-graph evidence.')
    parser.add_argument('--experimental-visual-faiss-fused-evidence', action='store_true', help='Default-off experiment passed to per-sequence QA: promote FAISS visual object recall into the main evidence/candidate-fact pool before generic evidence selection.')
    parser.add_argument('--experimental-visual-faiss-graph-anchors', action='store_true', help='Default-off experiment passed to per-sequence QA: use FAISS recall as visual frame anchors, then promote nearby canonical graph evidence into the main evidence pool.')
    parser.add_argument('--experimental-visual-faiss-ger-fusion', action='store_true', help='Default-off experiment passed to per-sequence QA: treat object-level FAISS hits as GER-like retrieval candidates by fusing HSGG facts and nearby KG context into the unified evidence/candidate-fact pool.')
    parser.add_argument('--experimental-visual-faiss-ger-binding', action='store_true', help='Default-off experiment passed to per-sequence QA: add a generic GER-like binding layer that ranks object FAISS hits by question/IR term coverage, HSGG attributes/relations, and nearby graph context. Requires --experimental-visual-faiss-ger-fusion and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-frame-context', action='store_true', help='Default-off experiment passed to per-sequence QA: enrich object-level FAISS hits with object-anchored same-frame HSGG context before GER fusion. Requires --experimental-visual-faiss-ger-fusion plus --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-structural-binding', action='store_true', help='Default-off experiment passed to per-sequence QA: use structural GER-like binding that separates target, attribute, event, spatial, and evidence-need slots instead of treating all term matches as hard constraints. Requires --experimental-visual-faiss-ger-binding.')
    parser.add_argument('--experimental-visual-faiss-attribute-owner', action='store_true', help='Default-off experiment passed to per-sequence QA: mark HSGG attributes with explicit owner roles, separating direct_object_attribute from neighbor_object_attribute. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-owner-minimal-prompt', action='store_true', help='Default-off experiment passed to per-sequence QA: use a compact owner-attribute QA prompt. This changes only final QA prompt wording, not retrieval, graph construction, FAISS, or judging.')
    parser.add_argument('--experimental-visual-faiss-fragment-bridge', action='store_true', help='Default-off experiment passed to per-sequence QA: package object-level FAISS hits, HSGG object facts, same-frame context, and nearby graph fragments into compact graph-local bridge facts. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-object-local-facts', action='store_true', help='Default-off experiment passed to per-sequence QA: add short object-local facts for recalled objects using direct HSGG attributes, direct relations, same-frame anchor/bbox context, and direct neighbor attributes. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-local-subgraph', action='store_true', help='Default-off experiment passed to per-sequence QA: map object FAISS hits back to canonical entity ids, then rank candidate entity local subgraphs with direct attributes, relations, time range, visual anchor, and bounded neighbors. Requires --experimental-visual-faiss-ger-binding and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-local-subgraph-anchor-only', action='store_true', help='Default-off CG6 local-subgraph v2 experiment passed to per-sequence QA: only the temporal observation entity directly mapped from a FAISS object hit becomes a candidate local subgraph; nearby entities remain context only.')
    parser.add_argument('--experimental-visual-faiss-balanced-graph-visual', action='store_true', help='Default-off experiment passed to per-sequence QA: keep original graph evidence in the primary evidence pool while adding visual FAISS object-local, bound, and GER context evidence as balanced supplemental channels. Requires --experimental-visual-faiss-ger-binding.')
    parser.add_argument('--experimental-visual-faiss-unified-candidates', action='store_true', help='Default-off experiment passed to per-sequence QA: build unified graph+object-FAISS candidates by combining object recall, frame-level HSGG attributes/relations, and nearby canonical graph support. Requires --experimental-visual-faiss-ger-fusion and never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-evidence-layer', action='store_true', help='Default-off experiment passed to per-sequence QA: use object FAISS hits to query the standalone evidence-layer SQLite index, then expose object-observation local packs with HSGG attributes, HSGG relations, bbox neighbors, image anchors, and time anchors. Never uses gold answers.')
    parser.add_argument('--experimental-visual-faiss-evidence-layer-constraint-binding', action='store_true', help='Default-off evidence-layer v2 extension passed to per-sequence QA: rank object-local evidence packs by generic question-IR target, attribute, event, and spatial coverage. Requires --experimental-visual-faiss-evidence-layer and never uses gold answers, question-type branches, or dataset wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-evidence', action='store_true', help='Default-off object-FAISS route passed to per-sequence QA: use question-IR roles so target entities become candidate subjects while event/spatial/evidence-need hits stay as context attachments. It reads processed_attributes.json and creates simple missing entries on demand without gold answers or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-aux', action='store_true', help='Default-off auxiliary object-FAISS role-check layer passed to per-sequence QA. It can be combined with GER-like fusion: GER/object-local evidence remains primary, while role-aware target/context checks are exposed only as auxiliary evidence. Never uses gold answers, question-type branches, or dataset/domain wordlists.')
    parser.add_argument('--experimental-visual-faiss-role-aware-compression', action='store_true', help='Default-off GER-binding experiment passed to per-sequence QA: separate question-IR target objects from constraint/context objects during object-FAISS compression before GER evidence packaging. Requires --experimental-visual-faiss-ger-fusion and --experimental-visual-faiss-ger-binding.')
    parser.add_argument('--experimental-visual-faiss-entity-pack', action='store_true', help='Default-off experiment passed to per-sequence QA: use object FAISS hits as entity/tracklet candidate generators and expose visual_entity_candidate_pack as the primary LLM evidence rerank unit.')
    parser.add_argument('--experimental-visual-faiss-entity-pack-semantic-align', action='store_true', help='Default-off entity-pack v2 experiment passed to per-sequence QA: use SigLIP text embeddings to weakly align primary target phrases with candidate entity labels/direct attributes/direct relations. Requires --experimental-visual-faiss-entity-pack.')
    parser.add_argument('--experimental-visual-faiss-entity-pack-semantic-gated-role', action='store_true', help='Default-off entity-pack v3 experiment passed to per-sequence QA: use SigLIP text embeddings as a semantic gate before primary-target FAISS recalls can become target entity packs. Requires --experimental-visual-faiss-entity-pack.')
    parser.add_argument('--experimental-visual-faiss-score-contract', action='store_true', help='Default-off score-contract experiment passed to per-sequence QA: after entity-pack recall, rank candidate packs with separated retrieval/binding/semantic/answerability fields and expose only candidate packs plus limited fallback evidence to rerank. Requires --experimental-visual-faiss-entity-pack.')
    parser.add_argument('--experimental-frame-faiss-query-recall', action='store_true', help='Default-off experiment passed to per-sequence QA: retrieve full video frames from the frame-level FAISS index using question/IR text.')
    parser.add_argument('--experimental-frame-faiss-fused-evidence', action='store_true', help='Default-off experiment passed to per-sequence QA: promote frame-level FAISS clusters into the main evidence/candidate-fact pool.')
    parser.add_argument('--experimental-frame-faiss-graph-anchors', action='store_true', help='Default-off experiment passed to per-sequence QA: use frame-level FAISS as visual anchors, then promote nearby canonical graph evidence.')
    parser.add_argument('--experimental-merged-frame-faiss-query-recall', action='store_true', help='Default-off experiment passed to per-sequence QA: retrieve merged-frame storyboard windows from the merged-frame FAISS index using question/IR text.')
    parser.add_argument('--experimental-merged-frame-faiss-fused-evidence', action='store_true', help='Default-off experiment passed to per-sequence QA: promote merged-frame FAISS clusters into the main evidence/candidate-fact pool.')
    parser.add_argument('--experimental-merged-frame-faiss-graph-anchors', action='store_true', help='Default-off experiment passed to per-sequence QA: use merged-frame FAISS windows as visual anchors, then promote nearby graph evidence.')
    parser.add_argument('--experimental-frame-window-faiss-query-recall', action='store_true', help='Default-off experiment passed to per-sequence QA: retrieve full frames with frame FAISS, aggregate frame hits into temporal windows, and expose them through the merged-frame evidence channel.')
    parser.add_argument('--experimental-frame-window-faiss-graph-anchors', action='store_true', help='Default-off experiment passed to per-sequence QA: use frame-window FAISS windows as visual anchors, then promote nearby graph evidence.')
    parser.add_argument('--visual-faiss-index-dir', type=Path, default=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime'))
    parser.add_argument('--visual-faiss-index-name', default='object_siglip_flat_ip')
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
    parser.add_argument('--visual-faiss-cross-sequence', action='store_true')
    parser.add_argument('--frame-faiss-index-dir', type=Path, default=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/frame_faiss'))
    parser.add_argument('--frame-faiss-index-name', default='frame_siglip_flat_ip')
    parser.add_argument('--frame-faiss-query-text-limit', type=int, default=6)
    parser.add_argument('--frame-faiss-top-k', type=int, default=12)
    parser.add_argument('--frame-faiss-search-k', type=int, default=80)
    parser.add_argument('--frame-faiss-min-score', type=float, default=0.0)
    parser.add_argument('--frame-faiss-cluster-max-frame-gap', type=int, default=30)
    parser.add_argument('--frame-faiss-cluster-max-sec-gap', type=float, default=None)
    parser.add_argument('--frame-faiss-max-clusters', type=int, default=8)
    parser.add_argument('--frame-faiss-max-lines', type=int, default=8)
    parser.add_argument('--frame-faiss-cross-sequence', action='store_true')
    parser.add_argument('--merged-frame-faiss-index-dir', type=Path, default=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/merged_frame_faiss'))
    parser.add_argument('--merged-frame-faiss-index-name', default='merged_frame_siglip_flat_ip_w5_s2_grid2x3')
    parser.add_argument('--merged-frame-faiss-query-text-limit', type=int, default=6)
    parser.add_argument('--merged-frame-faiss-top-k', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-search-k', type=int, default=80)
    parser.add_argument('--merged-frame-faiss-min-score', type=float, default=0.0)
    parser.add_argument('--merged-frame-faiss-cluster-max-frame-gap', type=int, default=48)
    parser.add_argument('--merged-frame-faiss-cluster-max-sec-gap', type=float, default=None)
    parser.add_argument('--merged-frame-faiss-max-clusters', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-max-lines', type=int, default=8)
    parser.add_argument('--merged-frame-faiss-cross-sequence', action='store_true')
    parser.add_argument('--frame-window-faiss-window-size-sec', type=float, default=5.0)
    parser.add_argument('--frame-window-faiss-window-stride-sec', type=float, default=2.0)
    parser.add_argument('--frame-window-faiss-max-frames-per-window', type=int, default=8)
    parser.add_argument('--qa-batch-size', type=int, default=5)
    parser.add_argument('--qa-collapse-missing-batches', action='store_true', help='Default-off execution optimization for resume runs: keep completed QA batches as-is, but merge consecutive missing batches into a larger per-sequence subprocess so graph/cache/FAISS are loaded fewer times. This changes only runner batching, not retrieval or QA logic.')
    parser.add_argument('--qa-workers', type=int, default=3, help='Run independent QA batches concurrently. The project default is 3, which was the fastest stable setting on the current FAISS workload. Use 1 for serial diagnostics.')
    parser.add_argument('--qa-scheduler', choices=('sequence', 'global'), default='global', help="QA batch scheduling mode. 'global' is the project default: warm one batch per sequence, then run all remaining batches from all sequences through one worker pool. Use 'sequence' to finish each sequence before starting the next one.")
    parser.add_argument('--qa-retries', type=int, default=2)
    parser.add_argument('--text-judge-mode', choices=('none', 'live'), default='live', help="Evaluation-only LLM judge for text/descriptive answers. The NaVQA official-style metric uses this for free-form text answers; use 'none' only for offline diagnostics.")
    parser.add_argument('--ir-batch-size', type=int, default=3)
    parser.add_argument('--ir-max-tokens', type=int, default=3000)
    parser.add_argument('--ir-timeout-sec', type=float, default=120.0)
    parser.add_argument('--force-question-ir', action='store_true')
    parser.add_argument('--reuse-manifest-question-ir', action='store_true', help='Reuse question_ir_json paths listed in the sequence manifest. The clean mainline default is to regenerate run-local Question IR so historical IR artifacts cannot silently enter a new run.')
    parser.add_argument('--force-qa', action='store_true')
    parser.add_argument('--reuse-sequence-projection-cache', action='store_true', help='Use the cache path discovered with each sequence. By default the projection cache is run-local so graph/index reuse is explicit.')
    parser.add_argument('--skip-question-ir-generation', action='store_true', help='Require existing question-IR files. Useful when doing a no-network stability pass.')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--log-file', type=Path, default=None, help='Write a full runner log here. By default, the runner writes mainline_run.log under the run directory.')
    parser.add_argument('--no-run-log', action='store_true', help='Disable the default run log file.')
    parser.add_argument('--preflight-only', action='store_true', help='Validate selected sequence inputs and question-IR availability, then exit without generating IR or running QA.')
    parser.add_argument('--run', action='store_true')
    return parser

def repo_path(path: Path | str | None, *, repo_root: Path=REPO_ROOT) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else repo_root / value

def load_questions(csv_path: Path, seq_id: str) -> list[dict[str, str]]:
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        return [row for row in csv.DictReader(handle) if str(row.get(FIELD_SEQ_ID, '')).strip() == str(seq_id)]

def non_position_questions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed = set(NON_POSITION_QA_TYPES)
    return [row for row in rows if str(row.get(FIELD_TYPE, '')).strip().lower() in allowed]

def normalized_question_type_set(values: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value).strip().lower() for value in values if str(value).strip()}

def selected_questions(rows: list[dict[str, str]], *, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> list[dict[str, str]]:
    includes = normalized_question_type_set(include_types)
    excludes = normalized_question_type_set(exclude_types) or set()
    if includes is None:
        includes = set(NON_POSITION_QA_TYPES)
    selected: list[dict[str, str]] = []
    for row in rows:
        qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
        if qtype not in includes:
            continue
        if qtype in excludes:
            continue
        selected.append(row)
    return selected

def append_question_type_scope(command: list[str], *, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> None:
    if include_types is None:
        command.append('--non-position-only')
    else:
        command.append('--include-types')
        command.extend((str(value) for value in include_types))
    if exclude_types:
        command.append('--exclude-types')
        command.extend((str(value) for value in exclude_types))

def question_scope_label(*, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> str:
    includes = normalized_question_type_set(include_types)
    excludes = normalized_question_type_set(exclude_types) or set()
    include_text = 'non_position_only' if includes is None else '+'.join(sorted(includes))
    if not excludes:
        return include_text
    return f"{include_text}_minus_{'+'.join(sorted(excludes))}"

def discover_sequences(*, layout_root: Path, repo_root: Path=REPO_ROOT, include_legacy_seq0: bool=False, manifest_path: Path | None=None) -> list[SequenceSpec]:
    specs: list[SequenceSpec] = []
    if manifest_path is not None:
        specs.extend(load_sequence_manifest(manifest_path, repo_root=repo_root))
    specs.extend(discover_unified_layout_sequences(layout_root, repo_root=repo_root))
    return dedupe_sequence_specs(specs)

def load_sequence_manifest(path: Path, *, repo_root: Path=REPO_ROOT) -> list[SequenceSpec]:
    manifest_path = repo_path(path, repo_root=repo_root)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    raw_items = payload.get('sequences', payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("sequence manifest must be a list or an object with 'sequences'")
    specs: list[SequenceSpec] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        seq_id = normalize_seq_id(item.get('seq_id', ''))
        cam = normalize_cam(item.get('cam', '0'))
        stride = normalize_stride(item.get('stride', 10))
        sample_root = repo_path(item.get('sample_root') or default_navqa_sequence_layout(seq_id=seq_id, cam=cam, stride=stride, layout_root=repo_root / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa').sample_root, repo_root=repo_root)
        graph_path = repo_path(item.get('canonical_graph_json'), repo_root=repo_root)
        if graph_path is None:
            raise ValueError(f'manifest item for seq{seq_id} missing canonical_graph_json')
        layout_kind = manifest_layout_kind(item.get('layout_kind'))
        specs.append(SequenceSpec(seq_id=seq_id, cam=cam, stride=stride, layout_kind=layout_kind, sample_root=sample_root or graph_path.parent, canonical_graph_json=graph_path, frame_time_index_json=repo_path(item.get('frame_time_index_json'), repo_root=repo_root), qa_unfilled_json=repo_path(item.get('qa_unfilled_json'), repo_root=repo_root), question_ir_json=repo_path(item.get('question_ir_json'), repo_root=repo_root), projection_cache=repo_path(item.get('projection_cache'), repo_root=repo_root), existing_baseline_json=repo_path(item.get('existing_baseline_json'), repo_root=repo_root)))
    return specs

def load_manifest_payload(path: Path | None, *, repo_root: Path=REPO_ROOT) -> dict[str, Any]:
    manifest_path = repo_path(path, repo_root=repo_root)
    if manifest_path is None or not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    return payload if isinstance(payload, dict) else {}

def manifest_default_output_root(path: Path | None, *, repo_root: Path=REPO_ROOT) -> Path | None:
    payload = load_manifest_payload(path, repo_root=repo_root)
    value = payload.get('default_output_root')
    if value is None:
        return None
    return repo_path(value, repo_root=repo_root)

def manifest_pipeline_metadata(path: Path | None, *, repo_root: Path=REPO_ROOT) -> dict[str, Any]:
    payload = load_manifest_payload(path, repo_root=repo_root)
    return {'naming_contract': payload.get('naming_contract', 'navqa_cg_ir_qa_naming_v1'), 'cg_id': payload.get('cg_id'), 'cg_name': payload.get('cg_name'), 'cg_role': payload.get('cg_role') or payload.get('mainline_role'), 'graph_profile': payload.get('graph_profile'), 'ir_qa_default_id': payload.get('ir_qa_default_id', 'IR-QA-G'), 'ir_qa_default_name': payload.get('ir_qa_default_name'), 'default_route_id': payload.get('default_route_id')}

def ir_qa_id_from_flags(*, visual_faiss_focus_seed: bool, visual_faiss_query_recall: bool, visual_faiss_fused_evidence: bool, visual_faiss_graph_anchors: bool=False, visual_faiss_ger_fusion: bool=False, visual_faiss_ger_binding: bool=False, visual_faiss_frame_context: bool=False, visual_faiss_structural_binding: bool=False, visual_faiss_attribute_owner: bool=False, owner_minimal_prompt: bool=False, visual_faiss_fragment_bridge: bool=False, visual_faiss_object_local_facts: bool=False, visual_faiss_local_subgraph: bool=False, visual_faiss_local_subgraph_anchor_only: bool=False, visual_faiss_evidence_layer: bool=False, visual_faiss_evidence_layer_constraint_binding: bool=False, visual_faiss_role_aware_evidence: bool=False, visual_faiss_role_aware_aux: bool=False, visual_faiss_role_aware_compression: bool=False, visual_faiss_entity_pack: bool=False, visual_faiss_entity_pack_semantic_align: bool=False, visual_faiss_entity_pack_semantic_gated_role: bool=False, visual_faiss_score_contract: bool=False, visual_faiss_balanced_graph_visual: bool=False, answerable_candidate_packs: bool=False, visual_faiss_unified_candidates: bool=False, synonym_lexicon: bool=False, question_router: bool=False, question_router_hybrid_backbone: bool=False, question_router_specialized_backbone: bool=False, question_router_mainline: bool=False, frame_faiss_query_recall: bool=False, frame_faiss_fused_evidence: bool=False, frame_faiss_graph_anchors: bool=False, merged_frame_faiss_query_recall: bool=False, merged_frame_faiss_fused_evidence: bool=False, merged_frame_faiss_graph_anchors: bool=False, frame_window_faiss_query_recall: bool=False, frame_window_faiss_graph_anchors: bool=False) -> str:
    if question_router_mainline:
        return QUESTION_ROUTER_MAINLINE_IR_QA_ID
    if question_router_specialized_backbone:
        return QUESTION_ROUTER_SPECIALIZED_IR_QA_ID
    if question_router_hybrid_backbone:
        return QUESTION_ROUTER_HYBRID_IR_QA_ID
    if question_router:
        return QUESTION_ROUTER_IR_QA_ID
    if synonym_lexicon:
        return SYNONYM_ROUTE_IR_QA_ID
    if frame_window_faiss_graph_anchors and visual_faiss_query_recall:
        return 'IR-QA-F-DC-v2'
    if frame_window_faiss_graph_anchors:
        return 'IR-QA-F-MF-v2'
    if frame_window_faiss_query_recall:
        return 'IR-QA-F-MF-v2-attach'
    if merged_frame_faiss_graph_anchors and visual_faiss_query_recall:
        return 'IR-QA-F-DC-v1'
    if merged_frame_faiss_graph_anchors:
        return 'IR-QA-F-MF-v1'
    if merged_frame_faiss_fused_evidence:
        return 'IR-QA-F-MF-fused-v1'
    if merged_frame_faiss_query_recall:
        return 'IR-QA-F-MF-v0'
    if frame_faiss_graph_anchors:
        return 'IR-QA-F-frame-v2'
    if frame_faiss_fused_evidence:
        return 'IR-QA-F-frame-v1'
    if frame_faiss_query_recall:
        return 'IR-QA-F-frame-v0'
    if visual_faiss_graph_anchors:
        return 'IR-QA-F-v4'
    if visual_faiss_evidence_layer:
        if visual_faiss_evidence_layer_constraint_binding:
            return 'IR-QA-F-evidence-layer-v2-constraint-binding'
        return 'IR-QA-F-evidence-layer-v1'
    if visual_faiss_role_aware_evidence:
        return 'IR-QA-F-role-aware-evidence-v1'
    if visual_faiss_entity_pack:
        if visual_faiss_score_contract:
            return 'IR-QA-F-score-contract-v1'
        if visual_faiss_entity_pack_semantic_gated_role:
            return 'IR-QA-F-entity-pack-v3-semantic-gated-role'
        if visual_faiss_entity_pack_semantic_align:
            return 'IR-QA-F-entity-pack-v2-semantic-align'
        return 'IR-QA-F-entity-pack-v1'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_role_aware_compression:
        return 'IR-QA-F-GER-v3-role-aware-compression'
    if visual_faiss_ger_fusion and visual_faiss_unified_candidates:
        return 'IR-QA-F-GER-v4-unified-candidate'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_local_subgraph:
        if visual_faiss_local_subgraph_anchor_only:
            return 'IR-QA-F-CG6-local-subgraph-v2'
        return 'IR-QA-F-CG6-local-subgraph-v1'
    if answerable_candidate_packs and visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_object_local_facts:
        return 'IR-QA-F-GER-v10-answerable-candidate-pack'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_object_local_facts and visual_faiss_balanced_graph_visual:
        return 'IR-QA-F-GER-v9-balanced-graph-visual'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_object_local_facts:
        return 'IR-QA-F-GER-v8-object-local-fact'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_fragment_bridge:
        return 'IR-QA-F-GER-v7-fragment-bridge'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_attribute_owner and owner_minimal_prompt and visual_faiss_role_aware_aux:
        return 'IR-QA-F-GER-v13-role-aware-aux'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_attribute_owner and owner_minimal_prompt:
        return 'IR-QA-F-GER-v12-owner-minprompt'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_attribute_owner:
        return 'IR-QA-F-GER-v11-attribute-owner'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_frame_context and visual_faiss_structural_binding:
        return 'IR-QA-F-GER-v6-structural-binding'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding and visual_faiss_frame_context:
        return 'IR-QA-F-GER-v5-frame-context'
    if visual_faiss_ger_fusion and visual_faiss_ger_binding:
        return 'IR-QA-F-GER-v3-binding'
    if visual_faiss_ger_fusion:
        return 'IR-QA-F-GER-v1'
    if visual_faiss_fused_evidence:
        return 'IR-QA-F-v3'
    if visual_faiss_focus_seed and visual_faiss_query_recall:
        return 'IR-QA-F-v1+v2'
    if visual_faiss_query_recall:
        return 'IR-QA-F-v2'
    if visual_faiss_focus_seed:
        return 'IR-QA-F-v1'
    return 'IR-QA-G'

def route_id_for(*, cg_id: str | None, ir_qa_id: str) -> str:
    return f'{cg_id}+{ir_qa_id}' if cg_id else ir_qa_id

def resolve_run_ir_qa_id(*, flag_ir_qa_id: str, pipeline_metadata: dict[str, Any]) -> str:
    manifest_default = str(pipeline_metadata.get('ir_qa_default_id') or '').strip()
    if flag_ir_qa_id == 'IR-QA-G' and manifest_default and (manifest_default != 'IR-QA-G'):
        return manifest_default
    return flag_ir_qa_id

def manifest_layout_kind(value: Any) -> str:
    layout_kind = str(value or 'manifest').strip() or 'manifest'
    normalized = layout_kind.lower()
    if normalized == 'manifest' or normalized.startswith('manifest_'):
        return layout_kind
    return f'manifest_{layout_kind}'

def discover_unified_layout_sequences(layout_root: Path, *, repo_root: Path=REPO_ROOT) -> list[SequenceSpec]:
    root = repo_path(layout_root, repo_root=repo_root)
    if root is None or not root.exists():
        return []
    specs: list[SequenceSpec] = []
    for sample_root in sorted(root.glob('seq*/cam*/every*')):
        if not sample_root.is_dir():
            continue
        parsed = parse_unified_sample_root(sample_root)
        if parsed is None:
            continue
        seq_id, cam, stride = parsed
        layout = default_navqa_sequence_layout(seq_id=seq_id, cam=cam, stride=stride, layout_root=root)
        if not layout.canonical_graph_json.exists():
            continue
        specs.append(SequenceSpec(seq_id=layout.seq_id, cam=layout.cam, stride=layout.stride, layout_kind='unified', sample_root=layout.sample_root, canonical_graph_json=layout.canonical_graph_json, frame_time_index_json=existing_or_none(layout.frame_time_index_json), qa_unfilled_json=existing_or_none(layout.qa_unfilled_json), question_ir_json=first_existing_path(layout.sample_root / 'llm_question_ir_non_position.json', layout.sample_root / 'llm_question_ir_experiment.json'), projection_cache=layout.projection_cache, existing_baseline_json=first_existing_path(layout.sample_root / 'qa_non_position_live_with_question_ir.json', layout.qa_results_json)))
    return specs

def parse_unified_sample_root(sample_root: Path) -> tuple[str, str, int | None] | None:
    try:
        stride_dir = sample_root.name
        cam_dir = sample_root.parent.name
        seq_dir = sample_root.parent.parent.name
    except IndexError:
        return None
    seq_match = re.fullmatch('seq(.+)', seq_dir)
    cam_match = re.fullmatch('cam(.+)', cam_dir)
    stride_match = re.fullmatch('every(\\d+)', stride_dir)
    if not seq_match or not cam_match or (not stride_match):
        return None
    return (normalize_seq_id(seq_match.group(1)), normalize_cam(cam_match.group(1)), normalize_stride(stride_match.group(1)))

def dedupe_sequence_specs(specs: list[SequenceSpec]) -> list[SequenceSpec]:
    by_key: dict[tuple[str, str, int | None], SequenceSpec] = {}
    for spec in specs:
        key = (spec.seq_id, spec.cam, spec.stride)
        current = by_key.get(key)
        if current is None:
            by_key[key] = spec
            continue
        by_key[key] = choose_preferred_spec(current, spec)
    return [by_key[key] for key in sorted(by_key)]

def choose_preferred_spec(left: SequenceSpec, right: SequenceSpec) -> SequenceSpec:
    return left if layout_priority(left.layout_kind) <= layout_priority(right.layout_kind) else right

def layout_priority(layout_kind: str) -> int:
    normalized = str(layout_kind or '').strip().lower()
    if normalized == 'manifest' or normalized.startswith('manifest_'):
        return 0
    if normalized == 'unified':
        return 1
    return 9

def existing_or_none(path: Path | None) -> Path | None:
    return path if path is not None and path.exists() else None

def first_existing_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None

def filter_specs(specs: list[SequenceSpec], *, seq_ids: list[str] | None=None, cam: str | None=None, stride: str | int | None=None) -> list[SequenceSpec]:
    wanted_seq = {normalize_seq_id(value) for value in seq_ids} if seq_ids else None
    wanted_cam = normalize_cam(cam) if cam is not None else None
    wanted_stride = normalize_stride(stride) if stride is not None else None
    return [spec for spec in specs if (wanted_seq is None or spec.seq_id in wanted_seq) and (wanted_cam is None or spec.cam == wanted_cam) and (wanted_stride is None or spec.stride == wanted_stride)]

def run_name(value: str | None) -> str:
    if value:
        return value
    return 'mainline_' + datetime.now().strftime('%Y%m%d_%H%M%S')

def run_dir_for(*, output_root: Path, run_name_value: str) -> Path:
    return output_root / run_name_value

def default_log_path(*, output_root: Path, run_name_value: str) -> Path:
    return run_dir_for(output_root=output_root, run_name_value=run_name_value) / 'mainline_run.log'

def configure_run_log(log_path: Path | None) -> TextIO | None:
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', encoding='utf-8')
    timestamp = datetime.now().isoformat(timespec='seconds')
    handle.write(f'\n===== navqa mainline run {timestamp} =====\n')
    handle.flush()
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    return handle

def paths_for_sequence(spec: SequenceSpec, *, output_root: Path, run_name_value: str, reuse_existing_question_ir: bool=True, reuse_sequence_projection_cache: bool=False) -> MainlinePaths:
    output_dir = run_dir_for(output_root=output_root, run_name_value=run_name_value) / spec.display_id
    existing_ir = spec.question_ir_json if reuse_existing_question_ir else None
    question_ir_json = existing_ir or output_dir / 'llm_question_ir_non_position.json'
    projection_cache = spec.projection_cache if reuse_sequence_projection_cache and spec.projection_cache is not None else output_dir / 'canonical_projection_cache.json.gz'
    return MainlinePaths(output_dir=output_dir, question_ir_json=question_ir_json, question_ir_md=question_ir_json.with_suffix('.md'), merged_json=output_dir / 'qa_non_position_live_with_question_ir.json', merged_csv=output_dir / 'qa_non_position_live_with_question_ir.csv', merged_md=output_dir / 'qa_non_position_live_with_question_ir.md', projection_cache=projection_cache)

def question_ir_match_count(path: Path | None, selected_rows: list[dict[str, str]]) -> int:
    if path is None or not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    records = payload.get('records', []) if isinstance(payload, dict) else []
    by_uuid = {str(record.get('uuid') or '').strip() for record in records if isinstance(record, dict) and isinstance(record.get('llm_question_ir'), dict)}
    return sum((1 for row in selected_rows if str(row.get(FIELD_UUID, '')).strip() in by_uuid))

def batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    size = max(1, batch_size)
    return [(offset, min(size, total - offset)) for offset in range(0, total, size)]

def build_question_ir_command(spec: SequenceSpec, paths: MainlinePaths, *, csv_path: Path, config: str, ir_mode: str, domain_profile: str, batch_size: int, max_tokens: int, timeout_sec: float, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> list[str]:
    command = [sys.executable, str(REPO_ROOT / 'scripts/datasets/navqa/retrieval/build_question_ir.py'), '--csv', str(csv_path), '--seq-id', spec.seq_id, '--cam', spec.cam, '--stride', str(spec.stride or ''), '--config', config, '--mode', ir_mode, '--domain-profile', domain_profile, '--batch-size', str(batch_size), '--max-tokens', str(max_tokens), '--timeout-sec', str(timeout_sec), '--output-json', str(paths.question_ir_json), '--output-md', str(paths.question_ir_md)]
    append_question_type_scope(command, include_types=include_types, exclude_types=exclude_types)
    return command

def build_qa_batch_command(spec: SequenceSpec, paths: MainlinePaths, *, csv_path: Path, config: str, answer_mode: str, domain_profile: str, retrieval_limit: int, text_judge_mode: str, llm_evidence_rerank: bool, llm_evidence_rerank_max_candidates: int, llm_evidence_rerank_max_selected: int, answer_candidate_extraction: bool, answer_candidate_extraction_max_candidates: int, answer_candidate_extraction_max_tokens: int, experimental_ir_evidence_query_expansion: bool, experimental_balanced_evidence_candidates: bool, experimental_answerable_candidate_packs: bool, answerable_candidate_pack_max_items: int, experimental_unified_evidence_layer: bool, unified_evidence_max_items: int, unified_evidence_graph_quota: int, unified_evidence_visual_quota: int, unified_evidence_frame_quota: int, experimental_synonym_lexicon: bool=False, synonym_lexicon_json: Path=DEFAULT_SYNONYM_LEXICON_PATH, experimental_question_router: bool=False, experimental_question_router_hybrid_backbone: bool=False, experimental_question_router_specialized_backbone: bool=False, experimental_question_router_mainline: bool=False, experimental_owner_minimal_prompt: bool=False, experimental_atomic_answer_facts: bool, atomic_answer_fact_max_items: int, experimental_visual_faiss_evidence: bool, experimental_visual_query_faiss_recall: bool, experimental_visual_faiss_fused_evidence: bool, experimental_visual_faiss_graph_anchors: bool, experimental_visual_faiss_ger_fusion: bool, experimental_visual_faiss_ger_binding: bool, experimental_visual_faiss_frame_context: bool, experimental_visual_faiss_structural_binding: bool, experimental_visual_faiss_attribute_owner: bool=False, experimental_visual_faiss_fragment_bridge: bool=False, experimental_visual_faiss_object_local_facts: bool=False, experimental_visual_faiss_local_subgraph: bool=False, experimental_visual_faiss_local_subgraph_anchor_only: bool=False, experimental_visual_faiss_evidence_layer: bool=False, experimental_visual_faiss_evidence_layer_constraint_binding: bool=False, experimental_visual_faiss_role_aware_evidence: bool=False, experimental_visual_faiss_role_aware_aux: bool=False, experimental_visual_faiss_role_aware_compression: bool=False, experimental_visual_faiss_entity_pack: bool=False, experimental_visual_faiss_entity_pack_semantic_align: bool=False, experimental_visual_faiss_entity_pack_semantic_gated_role: bool=False, experimental_visual_faiss_score_contract: bool=False, experimental_visual_faiss_balanced_graph_visual: bool=False, experimental_visual_faiss_unified_candidates: bool=False, experimental_frame_faiss_query_recall: bool, experimental_frame_faiss_fused_evidence: bool, experimental_frame_faiss_graph_anchors: bool, experimental_merged_frame_faiss_query_recall: bool, experimental_merged_frame_faiss_fused_evidence: bool, experimental_merged_frame_faiss_graph_anchors: bool, experimental_frame_window_faiss_query_recall: bool, experimental_frame_window_faiss_graph_anchors: bool, visual_faiss_index_dir: Path, visual_faiss_index_name: str, visual_faiss_seed_limit: int, visual_faiss_top_k: int, visual_faiss_max_lines: int, visual_faiss_query_phrase_limit: int, visual_faiss_query_top_k: int, visual_faiss_query_search_k: int, visual_faiss_query_min_score: float, visual_faiss_ger_object_limit: int, visual_faiss_ger_context_radius: int, visual_faiss_ger_context_limit: int, visual_faiss_evidence_layer_sqlite: Path=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime/evidence_layer.sqlite'), visual_faiss_evidence_layer_object_limit: int=8, visual_faiss_evidence_layer_neighbor_limit: int=4, visual_faiss_role_aware_object_limit: int=8, visual_faiss_role_aware_context_limit: int=8, visual_faiss_cross_sequence: bool, frame_faiss_index_dir: Path=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/frame_faiss'), frame_faiss_index_name: str='frame_siglip_flat_ip', frame_faiss_query_text_limit: int=6, frame_faiss_top_k: int=12, frame_faiss_search_k: int=80, frame_faiss_min_score: float=0.0, frame_faiss_cluster_max_frame_gap: int=30, frame_faiss_cluster_max_sec_gap: float | None=None, frame_faiss_max_clusters: int=8, frame_faiss_max_lines: int=8, frame_faiss_cross_sequence: bool=False, merged_frame_faiss_index_dir: Path=Path('datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/optional/merged_frame_faiss'), merged_frame_faiss_index_name: str='merged_frame_siglip_flat_ip_w5_s2_grid2x3', merged_frame_faiss_query_text_limit: int=6, merged_frame_faiss_top_k: int=8, merged_frame_faiss_search_k: int=80, merged_frame_faiss_min_score: float=0.0, merged_frame_faiss_cluster_max_frame_gap: int=48, merged_frame_faiss_cluster_max_sec_gap: float | None=None, merged_frame_faiss_max_clusters: int=8, merged_frame_faiss_max_lines: int=8, merged_frame_faiss_cross_sequence: bool=False, frame_window_faiss_window_size_sec: float=5.0, frame_window_faiss_window_stride_sec: float=2.0, frame_window_faiss_max_frames_per_window: int=8, offset: int, limit: int, batch_json: Path, include_types: list[str] | tuple[str, ...] | None=None, exclude_types: list[str] | tuple[str, ...] | None=None) -> list[str]:
    command = [sys.executable, str(REPO_ROOT / 'scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py'), str(spec.canonical_graph_json), '--csv', str(csv_path), '--seq-id', spec.seq_id, '--cam', spec.cam, '--stride', str(spec.stride or ''), '--answer-mode', answer_mode, '--config', config, '--retrieval-limit', str(retrieval_limit), '--text-judge-mode', text_judge_mode, '--projection-cache', str(paths.projection_cache), '--domain-profile', domain_profile, '--question-ir-json', str(paths.question_ir_json), '--offset', str(offset), '--limit', str(limit), '--output-json', str(batch_json), '--output-csv', str(batch_json.with_suffix('.csv')), '--output-md', str(batch_json.with_suffix('.md'))]
    append_question_type_scope(command, include_types=include_types, exclude_types=exclude_types)
    if llm_evidence_rerank:
        command.extend(['--llm-evidence-rerank', '--llm-evidence-rerank-max-candidates', str(llm_evidence_rerank_max_candidates), '--llm-evidence-rerank-max-selected', str(llm_evidence_rerank_max_selected)])
    if answer_candidate_extraction:
        command.extend(['--answer-candidate-extraction', '--answer-candidate-extraction-max-candidates', str(answer_candidate_extraction_max_candidates), '--answer-candidate-extraction-max-tokens', str(answer_candidate_extraction_max_tokens)])
    if experimental_ir_evidence_query_expansion:
        command.append('--experimental-ir-evidence-query-expansion')
    if experimental_balanced_evidence_candidates:
        command.append('--experimental-balanced-evidence-candidates')
    if experimental_answerable_candidate_packs:
        command.extend(['--experimental-answerable-candidate-packs', '--answerable-candidate-pack-max-items', str(answerable_candidate_pack_max_items)])
    if experimental_unified_evidence_layer:
        command.extend(['--experimental-unified-evidence-layer', '--unified-evidence-max-items', str(unified_evidence_max_items), '--unified-evidence-graph-quota', str(unified_evidence_graph_quota), '--unified-evidence-visual-quota', str(unified_evidence_visual_quota), '--unified-evidence-frame-quota', str(unified_evidence_frame_quota)])
    if experimental_atomic_answer_facts:
        command.extend(['--experimental-atomic-answer-facts', '--atomic-answer-fact-max-items', str(atomic_answer_fact_max_items)])
    if experimental_synonym_lexicon:
        command.extend(['--experimental-synonym-lexicon', '--synonym-lexicon-json', str(synonym_lexicon_json)])
    if experimental_question_router:
        command.append('--experimental-question-router')
    if experimental_question_router_hybrid_backbone:
        command.append('--experimental-question-router-hybrid-backbone')
    if experimental_question_router_specialized_backbone:
        command.append('--experimental-question-router-specialized-backbone')
    if experimental_question_router_mainline:
        command.append('--experimental-question-router-mainline')
    if experimental_owner_minimal_prompt:
        command.append('--experimental-owner-minimal-prompt')
    if experimental_visual_faiss_evidence or experimental_visual_query_faiss_recall:
        command.extend(['--visual-faiss-index-dir', str(visual_faiss_index_dir), '--visual-faiss-index-name', visual_faiss_index_name, '--visual-faiss-seed-limit', str(visual_faiss_seed_limit), '--visual-faiss-top-k', str(visual_faiss_top_k), '--visual-faiss-max-lines', str(visual_faiss_max_lines), '--visual-faiss-query-phrase-limit', str(visual_faiss_query_phrase_limit), '--visual-faiss-query-top-k', str(visual_faiss_query_top_k), '--visual-faiss-query-search-k', str(visual_faiss_query_search_k), '--visual-faiss-query-min-score', str(visual_faiss_query_min_score), '--visual-faiss-ger-object-limit', str(visual_faiss_ger_object_limit), '--visual-faiss-ger-context-radius', str(visual_faiss_ger_context_radius), '--visual-faiss-ger-context-limit', str(visual_faiss_ger_context_limit), '--visual-faiss-evidence-layer-sqlite', str(visual_faiss_evidence_layer_sqlite), '--visual-faiss-evidence-layer-object-limit', str(visual_faiss_evidence_layer_object_limit), '--visual-faiss-evidence-layer-neighbor-limit', str(visual_faiss_evidence_layer_neighbor_limit), '--visual-faiss-role-aware-object-limit', str(visual_faiss_role_aware_object_limit), '--visual-faiss-role-aware-context-limit', str(visual_faiss_role_aware_context_limit)])
        if experimental_visual_faiss_evidence:
            command.append('--experimental-visual-faiss-evidence')
        if experimental_visual_query_faiss_recall:
            command.append('--experimental-visual-query-faiss-recall')
        if experimental_visual_faiss_fused_evidence:
            command.append('--experimental-visual-faiss-fused-evidence')
        if experimental_visual_faiss_graph_anchors:
            command.append('--experimental-visual-faiss-graph-anchors')
        if experimental_visual_faiss_ger_fusion:
            command.append('--experimental-visual-faiss-ger-fusion')
        if experimental_visual_faiss_ger_binding:
            command.append('--experimental-visual-faiss-ger-binding')
        if experimental_visual_faiss_frame_context:
            command.append('--experimental-visual-faiss-frame-context')
        if experimental_visual_faiss_structural_binding:
            command.append('--experimental-visual-faiss-structural-binding')
        if experimental_visual_faiss_attribute_owner:
            command.append('--experimental-visual-faiss-attribute-owner')
        if experimental_visual_faiss_fragment_bridge:
            command.append('--experimental-visual-faiss-fragment-bridge')
        if experimental_visual_faiss_object_local_facts:
            command.append('--experimental-visual-faiss-object-local-facts')
        if experimental_visual_faiss_local_subgraph:
            command.append('--experimental-visual-faiss-local-subgraph')
        if experimental_visual_faiss_local_subgraph_anchor_only:
            command.append('--experimental-visual-faiss-local-subgraph-anchor-only')
        if experimental_visual_faiss_balanced_graph_visual:
            command.append('--experimental-visual-faiss-balanced-graph-visual')
        if experimental_visual_faiss_unified_candidates:
            command.append('--experimental-visual-faiss-unified-candidates')
        if experimental_visual_faiss_evidence_layer:
            command.append('--experimental-visual-faiss-evidence-layer')
        if experimental_visual_faiss_evidence_layer_constraint_binding:
            command.append('--experimental-visual-faiss-evidence-layer-constraint-binding')
        if experimental_visual_faiss_role_aware_evidence:
            command.append('--experimental-visual-faiss-role-aware-evidence')
        if experimental_visual_faiss_role_aware_aux:
            command.append('--experimental-visual-faiss-role-aware-aux')
        if experimental_visual_faiss_role_aware_compression:
            command.append('--experimental-visual-faiss-role-aware-compression')
        if experimental_visual_faiss_entity_pack:
            command.append('--experimental-visual-faiss-entity-pack')
        if experimental_visual_faiss_entity_pack_semantic_align:
            command.append('--experimental-visual-faiss-entity-pack-semantic-align')
        if experimental_visual_faiss_entity_pack_semantic_gated_role:
            command.append('--experimental-visual-faiss-entity-pack-semantic-gated-role')
        if experimental_visual_faiss_score_contract:
            command.append('--experimental-visual-faiss-score-contract')
        if visual_faiss_cross_sequence:
            command.append('--visual-faiss-cross-sequence')
    if experimental_frame_faiss_query_recall:
        command.extend(['--experimental-frame-faiss-query-recall', '--frame-faiss-index-dir', str(frame_faiss_index_dir), '--frame-faiss-index-name', frame_faiss_index_name, '--frame-faiss-query-text-limit', str(frame_faiss_query_text_limit), '--frame-faiss-top-k', str(frame_faiss_top_k), '--frame-faiss-search-k', str(frame_faiss_search_k), '--frame-faiss-min-score', str(frame_faiss_min_score), '--frame-faiss-cluster-max-frame-gap', str(frame_faiss_cluster_max_frame_gap), '--frame-faiss-max-clusters', str(frame_faiss_max_clusters), '--frame-faiss-max-lines', str(frame_faiss_max_lines)])
        if frame_faiss_cluster_max_sec_gap is not None:
            command.extend(['--frame-faiss-cluster-max-sec-gap', str(frame_faiss_cluster_max_sec_gap)])
        if experimental_frame_faiss_fused_evidence:
            command.append('--experimental-frame-faiss-fused-evidence')
        if experimental_frame_faiss_graph_anchors:
            command.append('--experimental-frame-faiss-graph-anchors')
        if frame_faiss_cross_sequence:
            command.append('--frame-faiss-cross-sequence')
    if experimental_merged_frame_faiss_query_recall:
        command.extend(['--experimental-merged-frame-faiss-query-recall', '--merged-frame-faiss-index-dir', str(merged_frame_faiss_index_dir), '--merged-frame-faiss-index-name', merged_frame_faiss_index_name, '--merged-frame-faiss-query-text-limit', str(merged_frame_faiss_query_text_limit), '--merged-frame-faiss-top-k', str(merged_frame_faiss_top_k), '--merged-frame-faiss-search-k', str(merged_frame_faiss_search_k), '--merged-frame-faiss-min-score', str(merged_frame_faiss_min_score), '--merged-frame-faiss-cluster-max-frame-gap', str(merged_frame_faiss_cluster_max_frame_gap), '--merged-frame-faiss-max-clusters', str(merged_frame_faiss_max_clusters), '--merged-frame-faiss-max-lines', str(merged_frame_faiss_max_lines)])
        if merged_frame_faiss_cluster_max_sec_gap is not None:
            command.extend(['--merged-frame-faiss-cluster-max-sec-gap', str(merged_frame_faiss_cluster_max_sec_gap)])
        if experimental_merged_frame_faiss_fused_evidence:
            command.append('--experimental-merged-frame-faiss-fused-evidence')
        if experimental_merged_frame_faiss_graph_anchors:
            command.append('--experimental-merged-frame-faiss-graph-anchors')
        if merged_frame_faiss_cross_sequence:
            command.append('--merged-frame-faiss-cross-sequence')
    if experimental_frame_window_faiss_query_recall:
        command.extend(['--experimental-frame-window-faiss-query-recall', '--frame-faiss-index-dir', str(frame_faiss_index_dir), '--frame-faiss-index-name', frame_faiss_index_name, '--frame-faiss-query-text-limit', str(frame_faiss_query_text_limit), '--frame-faiss-top-k', str(frame_faiss_top_k), '--frame-faiss-search-k', str(frame_faiss_search_k), '--frame-faiss-min-score', str(frame_faiss_min_score), '--frame-faiss-cluster-max-frame-gap', str(frame_faiss_cluster_max_frame_gap), '--frame-faiss-max-clusters', str(frame_faiss_max_clusters), '--frame-faiss-max-lines', str(frame_faiss_max_lines), '--merged-frame-faiss-max-clusters', str(merged_frame_faiss_max_clusters), '--merged-frame-faiss-max-lines', str(merged_frame_faiss_max_lines), '--frame-window-faiss-window-size-sec', str(frame_window_faiss_window_size_sec), '--frame-window-faiss-window-stride-sec', str(frame_window_faiss_window_stride_sec), '--frame-window-faiss-max-frames-per-window', str(frame_window_faiss_max_frames_per_window)])
        if frame_faiss_cluster_max_sec_gap is not None:
            command.extend(['--frame-faiss-cluster-max-sec-gap', str(frame_faiss_cluster_max_sec_gap)])
        if experimental_frame_window_faiss_graph_anchors:
            command.append('--experimental-frame-window-faiss-graph-anchors')
        if frame_faiss_cross_sequence:
            command.append('--frame-faiss-cross-sequence')
    if spec.qa_unfilled_json is not None:
        command.extend(['--qa-unfilled-json', str(spec.qa_unfilled_json)])
    if spec.frame_time_index_json is not None:
        command.extend(['--frame-time-index', str(spec.frame_time_index_json)])
    return command

def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)

def run_command(command: list[str], *, retries: int, cwd: Path=REPO_ROOT, env: dict[str, str] | None=None) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 2):
        print(f'attempt {attempt}/{retries + 1}: {command_line(command)}', flush=True)
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.stdout:
            print(completed.stdout, end='', flush=True)
        try:
            completed.check_returncode()
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt > retries:
                break
            sleep_sec = min(30.0, 3.0 * attempt)
            print(f'retrying after failure in {sleep_sec:.1f}s', flush=True)
            time.sleep(sleep_sec)
    if last_error is not None:
        raise last_error

def parse_api_key_pool(value: str | None=None) -> list[str]:
    raw = os.environ.get('DEEPSEEK_API_KEYS', '') if value is None else value
    keys: list[str] = []
    seen: set[str] = set()
    for item in str(raw or '').split(','):
        key = item.strip()
        if not key or key in seen:
            continue
        keys.append(key)
        seen.add(key)
    return keys

def qa_task_env(task: QABatchTask, api_keys: list[str]) -> dict[str, str] | None:
    if task.api_key_index is None or not api_keys:
        return None
    env = os.environ.copy()
    env['DEEPSEEK_API_KEY'] = api_keys[task.api_key_index % len(api_keys)]
    return env

def run_qa_batch_tasks(tasks: list[QABatchTask], *, retries: int, workers: int, cwd: Path=REPO_ROOT, api_keys: list[str] | None=None) -> None:
    if not tasks:
        return
    worker_count = max(1, int(workers))
    key_pool = list(api_keys) if api_keys is not None else parse_api_key_pool()
    if key_pool:
        print(f'  qa api key pool: {len(key_pool)} key(s) available; batch subprocesses receive round-robin DEEPSEEK_API_KEY values', flush=True)
    if worker_count == 1 or len(tasks) == 1:
        for task in tasks:
            run_command(task.command, retries=retries, cwd=cwd, env=qa_task_env(task, key_pool))
        return
    warmup, *parallel_tasks = tasks
    print(f'  qa parallel: warmup batch {warmup.offset}:{warmup.offset + warmup.limit} before launching {min(worker_count, len(parallel_tasks))} worker(s)', flush=True)
    run_command(warmup.command, retries=retries, cwd=cwd, env=qa_task_env(warmup, key_pool))
    if not parallel_tasks:
        return
    print(f'  qa parallel: running {len(parallel_tasks)} remaining batch(es) with qa_workers={worker_count}', flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(run_command, task.command, retries=retries, cwd=cwd, env=qa_task_env(task, key_pool)): task for task in parallel_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            future.result()
            print(f'  qa batch {task.offset}:{task.offset + task.limit}: completed', flush=True)

def assign_api_key_indexes(tasks: list[QABatchTask], *, start_index: int=0) -> list[QABatchTask]:
    assigned: list[QABatchTask] = []
    for index, task in enumerate(tasks, start=start_index):
        assigned.append(replace(task, api_key_index=index))
    return assigned

def run_global_qa_batch_tasks(sequence_plans: list[SequenceRunPlan], *, retries: int, workers: int, cwd: Path=REPO_ROOT, api_keys: list[str] | None=None, warmup_per_sequence: bool=True) -> None:
    sequences_with_tasks = [plan for plan in sequence_plans if plan.qa_tasks]
    if not sequences_with_tasks:
        return
    key_pool = list(api_keys) if api_keys is not None else parse_api_key_pool()
    if key_pool:
        print(f'  qa api key pool: {len(key_pool)} key(s) available; batch subprocesses receive round-robin DEEPSEEK_API_KEY values', flush=True)
    warmups: list[QABatchTask] = []
    parallel_tasks: list[QABatchTask] = []
    api_key_index = 0
    if warmup_per_sequence:
        for plan in sequences_with_tasks:
            first, *remaining = plan.qa_tasks
            warmups.append(replace(first, api_key_index=api_key_index))
            api_key_index += 1
            for task in remaining:
                parallel_tasks.append(replace(task, api_key_index=api_key_index))
                api_key_index += 1
    else:
        for plan in sequences_with_tasks:
            for task in plan.qa_tasks:
                parallel_tasks.append(replace(task, api_key_index=api_key_index))
                api_key_index += 1
    if warmup_per_sequence:
        print(f'  qa global scheduler: warming {len(warmups)} sequence-local batch(es) before shared fanout', flush=True)
    else:
        print('  qa global scheduler: skipped sequence-local warmup for collapsed resume batches', flush=True)
    for task in warmups:
        run_command(task.command, retries=retries, cwd=cwd, env=qa_task_env(task, key_pool))
        print(f'  qa warmup batch {task.offset}:{task.offset + task.limit}: completed', flush=True)
    if not parallel_tasks:
        return
    worker_count = max(1, int(workers))
    if worker_count == 1:
        print(f'  qa global scheduler: running {len(parallel_tasks)} remaining batch(es) serially', flush=True)
        for task in parallel_tasks:
            run_command(task.command, retries=retries, cwd=cwd, env=qa_task_env(task, key_pool))
            print(f'  qa batch {task.offset}:{task.offset + task.limit}: completed', flush=True)
        return
    print(f'  qa global scheduler: running {len(parallel_tasks)} remaining batch(es) from {len(sequences_with_tasks)} sequence(s) with qa_workers={worker_count}', flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(run_command, task.command, retries=retries, cwd=cwd, env=qa_task_env(task, key_pool)): task for task in parallel_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            future.result()
            print(f'  qa batch {task.offset}:{task.offset + task.limit}: completed', flush=True)

def expected_batch_json(paths: MainlinePaths, offset: int, limit: int) -> Path:
    return paths.output_dir / 'batches' / f'qa_batch_{offset:03d}_{offset + limit:03d}.json'

def batch_output_complete(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:
        return False
    return len(payload.get('results', [])) == expected_count

def planned_qa_batch_ranges(*, paths: MainlinePaths, total: int, batch_size: int, force_qa: bool, collapse_missing_batches: bool) -> list[tuple[int, int, bool]]:
    ranges: list[tuple[int, int, bool]] = []
    pending_offset: int | None = None
    pending_limit = 0

    def flush_pending() -> None:
        nonlocal pending_offset, pending_limit
        if pending_offset is None:
            return
        ranges.append((pending_offset, pending_limit, False))
        pending_offset = None
        pending_limit = 0
    for offset, limit in batch_ranges(total, batch_size):
        batch_json = expected_batch_json(paths, offset, limit)
        complete = batch_output_complete(batch_json, limit) and (not force_qa)
        if complete:
            flush_pending()
            ranges.append((offset, limit, True))
            continue
        if not collapse_missing_batches:
            ranges.append((offset, limit, False))
            continue
        if pending_offset is None:
            pending_offset = offset
            pending_limit = limit
            continue
        if pending_offset + pending_limit == offset:
            pending_limit += limit
            continue
        flush_pending()
        pending_offset = offset
        pending_limit = limit
    flush_pending()
    return ranges

def load_batch_payloads(batch_paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding='utf-8-sig')) for path in batch_paths]

def merge_batch_payloads(*, spec: SequenceSpec, paths: MainlinePaths, csv_path: Path, selected_rows: list[dict[str, str]], batch_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    official_module = load_official_qa_module()
    results: list[dict[str, Any]] = []
    for payload in batch_payloads:
        batch_results = payload.get('results', [])
        if not isinstance(batch_results, list):
            raise ValueError('batch payload missing list results')
        results.extend(batch_results)
    first = batch_payloads[0] if batch_payloads else {}
    question_ir_matched = question_ir_match_count(paths.question_ir_json, selected_rows)
    return {'source_graph': str(spec.canonical_graph_json), 'source_csv': str(csv_path), 'seq_id': spec.seq_id, 'cam': spec.cam, 'stride': spec.stride, 'layout_kind': spec.layout_kind, 'pipeline': first.get('pipeline', 'canonical temporal memory graph -> canonical graph hybrid retrieval -> evidence bundle -> LLM verbalization'), 'retrieval_backend': first.get('retrieval_backend', 'canonical_graph_hybrid'), 'retrieval_mode': first.get('retrieval_mode', 'temporal'), 'answer_mode': first.get('answer_mode', ''), 'retrieval_limit': first.get('retrieval_limit', ''), 'llm_evidence_rerank': first.get('llm_evidence_rerank', {'enabled': False}), 'answer_candidate_extraction': first.get('answer_candidate_extraction', {'enabled': False}), 'experiments': first.get('experiments', {}), 'question_processing': {'source': 'llm_question_ir', 'question_ir_json': str(paths.question_ir_json), 'matched_count': question_ir_matched}, 'run_inputs': {'contract': 'navqa_mainline_run_inputs_v1', 'source_graph': str(spec.canonical_graph_json), 'source_csv': str(csv_path), 'sample_root': str(spec.sample_root), 'frame_time_index_json': str(spec.frame_time_index_json) if spec.frame_time_index_json is not None else None, 'qa_unfilled_json': str(spec.qa_unfilled_json) if spec.qa_unfilled_json is not None else None, 'question_ir_json': str(paths.question_ir_json), 'projection_cache': str(paths.projection_cache), 'output_dir': str(paths.output_dir), 'layout_kind': spec.layout_kind}, 'official_adapter': first.get('official_adapter', {'contract': 'navqa_official_qa_eval_adapter_v1', 'inject_into_graph': False}), 'mainline_runner': {'contract': 'navqa_mainline_all_sequences_v1', 'batch_count': len(batch_payloads), 'batch_resume': True, 'position_questions': 'archived_not_selected'}, 'question_count': len(results), 'navqa_official_style_accuracy': official_module.navqa_official_style_accuracy_summary(results), 'results': results}

def write_merged_outputs(payload: dict[str, Any], paths: MainlinePaths) -> None:
    official_module = load_official_qa_module()
    results = payload.get('results', [])
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.merged_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    official_module.write_csv(paths.merged_csv, results)
    official_module.write_markdown(paths.merged_md, results, seq_id=str(payload.get('seq_id', '')))

def load_official_qa_module() -> Any:
    script_path = REPO_ROOT / 'scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py'
    spec = importlib.util.spec_from_file_location('navqa_official_qa_eval', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    official_style_accuracy = payload.get('navqa_official_style_accuracy', {})
    if not isinstance(official_style_accuracy, dict):
        official_style_accuracy = {}
    official_style_by_type = official_style_accuracy.get('by_type', {}) if isinstance(official_style_accuracy, dict) else {}
    return {'correct': official_style_accuracy.get('correct', 0), 'total': official_style_accuracy.get('total', 0), 'accuracy_percent': official_style_accuracy.get('accuracy_percent', 0.0), 'by_type': official_style_by_type, 'metric': 'navqa_official_style_accuracy', 'official_metric_rules': {'temporal_threshold_minutes': official_style_accuracy.get('temporal_threshold_minutes', 2.0), 'spatial_threshold_meters': official_style_accuracy.get('spatial_threshold_meters', 15.0), 'judge': official_style_accuracy.get('text_judging', 'llm_judge_when_available_else_normalized_containment_proxy')}}

def load_existing_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    return summarize_payload(payload)

def completed_summary_row(*, spec: SequenceSpec, selected_rows: list[dict[str, str]], paths: MainlinePaths, csv_path: Path, batch_paths: list[Path], batch_ranges_for_sequence: list[tuple[int, int]]) -> dict[str, Any]:
    missing_batches = [path for path, (_offset, limit) in zip(batch_paths, batch_ranges_for_sequence) if not batch_output_complete(path, limit)]
    if missing_batches:
        raise SystemExit(f'Missing/incomplete QA batches for {spec.display_id}: {missing_batches}')
    merged = merge_batch_payloads(spec=spec, paths=paths, csv_path=csv_path, selected_rows=selected_rows, batch_payloads=load_batch_payloads(batch_paths))
    write_merged_outputs(merged, paths)
    new_summary = summarize_payload(merged)
    previous = load_existing_summary(spec.existing_baseline_json)
    delta = None
    if previous is not None:
        delta = round(float(new_summary.get('accuracy_percent', 0.0)) - float(previous.get('accuracy_percent', 0.0)), 2)
    print(f'  merged={paths.merged_json}', flush=True)
    return {'sequence': spec.display_id, 'layout_kind': spec.layout_kind, 'previous': previous, 'new': new_summary, 'delta_accuracy_percent': delta, 'output_json': str(paths.merged_json), 'status': 'completed'}

def write_all_sequence_summary(*, output_root: Path, run_name_value: str, rows: list[dict[str, Any]], run_metadata: dict[str, Any] | None=None) -> tuple[Path, Path]:
    run_dir = run_dir_for(output_root=output_root, run_name_value=run_name_value)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_json = run_dir / 'mainline_all_sequences_summary.json'
    summary_md = run_dir / 'mainline_all_sequences_summary.md'
    payload = {'contract': 'navqa_mainline_all_sequences_summary_v1', 'run_name': run_name_value, 'sequence_count': len(rows), 'run_metadata': run_metadata or {}, 'rows': rows}
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# NaVQA Mainline All Sequences', '', f'- run_name: `{run_name_value}`', f'- sequence_count: `{len(rows)}`']
    if run_metadata:
        route_id = run_metadata.get('route_id')
        cg_id = run_metadata.get('cg_id')
        ir_qa_id = run_metadata.get('ir_qa_id')
        manifest = run_metadata.get('sequence_manifest')
        output_root_text = run_metadata.get('output_root')
        log_path = run_metadata.get('log_path')
        if route_id:
            lines.append(f'- route_id: `{route_id}`')
        if cg_id:
            lines.append(f'- cg_id: `{cg_id}`')
        if ir_qa_id:
            lines.append(f'- ir_qa_id: `{ir_qa_id}`')
        if manifest:
            lines.append(f'- sequence_manifest: `{manifest}`')
        if output_root_text:
            lines.append(f'- output_root: `{output_root_text}`')
        if log_path:
            lines.append(f'- run_log: `{log_path}`')
        qa_scope = run_metadata.get('qa_scope')
        if qa_scope:
            lines.append(f'- qa_scope: `{qa_scope}`')
        lines.append('- post_graph_qa_logic: shared across selected question scope and mainlines')
    lines.extend(['', '| seq | layout | questions | correct | acc | previous acc | delta | by type |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |'])
    for row in rows:
        new = row.get('new', {})
        by_type = format_by_type(new.get('by_type', {}))
        prev = row.get('previous')
        previous_acc = '' if prev is None else f"{prev.get('accuracy_percent', 0.0):.2f}"
        delta = row.get('delta_accuracy_percent')
        delta_text = '' if delta is None else f'{delta:+.2f}'
        lines.append(f"| {row.get('sequence')} | {row.get('layout_kind')} | {new.get('total', 0)} | {new.get('correct', 0)} | {float(new.get('accuracy_percent', 0.0)):.2f} | {previous_acc} | {delta_text} | {by_type} |")
    summary_md.write_text('\n'.join(lines), encoding='utf-8')
    return (summary_json, summary_md)

def format_by_type(by_type: dict[str, Any]) -> str:
    parts: list[str] = []
    for qtype, bucket in sorted(by_type.items()):
        if not isinstance(bucket, dict):
            continue
        parts.append(f"{qtype} {bucket.get('correct', 0)}/{bucket.get('total', 0)} ({bucket.get('accuracy_percent', 0.0)}%)")
    return '<br>'.join(parts)

def validate_sequence_inputs(spec: SequenceSpec, *, csv_path: Path, selected_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    if not csv_path.exists():
        issues.append(f'missing csv: {csv_path}')
    if not spec.canonical_graph_json.exists():
        issues.append(f'missing canonical graph: {spec.canonical_graph_json}')
    if spec.frame_time_index_json is None or not spec.frame_time_index_json.exists():
        issues.append('missing frame_time_index_json')
    if spec.qa_unfilled_json is None or not spec.qa_unfilled_json.exists():
        issues.append('missing qa_unfilled_json')
    if not selected_rows:
        issues.append('no questions selected')
    return issues

def preflight_sequence(spec: SequenceSpec, paths: MainlinePaths, *, csv_path: Path, selected_rows: list[dict[str, str]], require_existing_question_ir: bool) -> dict[str, Any]:
    input_issues = validate_sequence_inputs(spec, csv_path=csv_path, selected_rows=selected_rows)
    question_ir_matches = question_ir_match_count(paths.question_ir_json, selected_rows)
    issues = list(input_issues)
    if require_existing_question_ir and question_ir_matches < len(selected_rows):
        issues.append(f'incomplete question_ir_json: {question_ir_matches}/{len(selected_rows)} at {paths.question_ir_json}')
    return {'sequence': spec.display_id, 'layout_kind': spec.layout_kind, 'question_count': len(selected_rows), 'question_ir_json': str(paths.question_ir_json), 'question_ir_matches': question_ir_matches, 'canonical_graph_json': str(spec.canonical_graph_json), 'frame_time_index_json': str(spec.frame_time_index_json) if spec.frame_time_index_json is not None else '', 'qa_unfilled_json': str(spec.qa_unfilled_json) if spec.qa_unfilled_json is not None else '', 'projection_cache': str(paths.projection_cache), 'output_dir': str(paths.output_dir), 'issues': issues, 'status': 'ok' if not issues else 'blocked'}

def print_preflight_report(rows: list[dict[str, Any]]) -> None:
    print('', flush=True)
    print('preflight:', flush=True)
    for row in rows:
        print(f"  {row['sequence']}: status={row['status']} questions={row['question_count']} ir={row['question_ir_matches']}/{row['question_count']}", flush=True)
        for issue in row['issues']:
            print(f'    issue: {issue}', flush=True)

def main() -> None:
    args = build_parser().parse_args()
    if args.experimental_visual_faiss_fused_evidence and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-fused-evidence requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
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
    if args.experimental_visual_faiss_evidence_layer and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-evidence-layer requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_role_aware_evidence and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-role-aware-evidence requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_role_aware_aux and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-role-aware-aux requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_role_aware_aux and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-role-aware-aux requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_role_aware_compression and (not args.experimental_visual_faiss_ger_fusion):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-faiss-ger-fusion.')
    if args.experimental_visual_faiss_role_aware_compression and (not args.experimental_visual_faiss_ger_binding):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-faiss-ger-binding.')
    if args.experimental_visual_faiss_role_aware_compression and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-role-aware-compression requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_entity_pack and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('--experimental-visual-faiss-entity-pack requires --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_entity_pack_semantic_align and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-entity-pack-semantic-align requires --experimental-visual-faiss-entity-pack.')
    if args.experimental_visual_faiss_entity_pack_semantic_gated_role and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-entity-pack-semantic-gated-role requires --experimental-visual-faiss-entity-pack.')
    if args.experimental_visual_faiss_score_contract and (not args.experimental_visual_faiss_entity_pack):
        raise SystemExit('--experimental-visual-faiss-score-contract requires --experimental-visual-faiss-entity-pack.')
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
    if (args.experimental_question_router_hybrid_backbone or args.experimental_question_router_specialized_backbone or args.experimental_question_router_mainline) and (not args.experimental_visual_faiss_evidence) and (not args.experimental_visual_query_faiss_recall):
        raise SystemExit('Question-router visual backbones require --experimental-visual-faiss-evidence or --experimental-visual-query-faiss-recall.')
    if args.experimental_visual_faiss_evidence_layer_constraint_binding and (not args.experimental_visual_faiss_evidence_layer):
        raise SystemExit('--experimental-visual-faiss-evidence-layer-constraint-binding requires --experimental-visual-faiss-evidence-layer.')
    if args.experimental_visual_faiss_graph_anchors and args.experimental_visual_faiss_fused_evidence:
        raise SystemExit('--experimental-visual-faiss-graph-anchors and --experimental-visual-faiss-fused-evidence are separate FAISS fusion modes.')
    visual_fusion_mode_count = sum((bool(value) for value in [args.experimental_visual_faiss_fused_evidence, args.experimental_visual_faiss_graph_anchors, args.experimental_visual_faiss_ger_fusion, args.experimental_visual_faiss_evidence_layer, args.experimental_visual_faiss_role_aware_evidence, args.experimental_visual_faiss_entity_pack, args.experimental_question_router_hybrid_backbone, args.experimental_question_router_specialized_backbone, args.experimental_question_router_mainline]))
    if visual_fusion_mode_count > 1:
        raise SystemExit('Choose only one visual FAISS fusion mode: --experimental-visual-faiss-fused-evidence, --experimental-visual-faiss-graph-anchors, --experimental-visual-faiss-ger-fusion, or --experimental-visual-faiss-evidence-layer, or --experimental-visual-faiss-role-aware-evidence, or --experimental-visual-faiss-entity-pack, or --experimental-question-router-hybrid-backbone, or --experimental-question-router-specialized-backbone, or --experimental-question-router-mainline.')
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
    csv_path = repo_path(args.csv)
    if csv_path is None:
        raise SystemExit('missing --csv')
    run_name_value = run_name(args.run_name)
    sequence_manifest = args.sequence_manifest
    if sequence_manifest is None and (not args.no_default_sequence_manifest):
        default_manifest = repo_path(DEFAULT_SEQUENCE_MANIFEST)
        if default_manifest is not None and default_manifest.exists():
            sequence_manifest = DEFAULT_SEQUENCE_MANIFEST
    if args.output_root == DEFAULT_OUTPUT_ROOT:
        output_root = manifest_default_output_root(sequence_manifest) or repo_path(args.output_root) or args.output_root
    else:
        output_root = repo_path(args.output_root) or args.output_root
    log_path = None
    if not args.no_run_log:
        if args.log_file is not None:
            log_path = repo_path(args.log_file)
        else:
            log_path = default_log_path(output_root=output_root, run_name_value=run_name_value)
    log_handle = configure_run_log(log_path)
    try:
        specs = discover_sequences(layout_root=args.layout_root, include_legacy_seq0=False, manifest_path=sequence_manifest)
        specs = filter_specs(specs, seq_ids=args.seq_id, cam=args.cam, stride=args.stride)
        if not specs:
            raise SystemExit('No available sequences found.')
        mode = 'preflight' if args.preflight_only else 'run' if args.run else 'dry-run'
        print(f'mode={mode}', flush=True)
        print(f'run_name={run_name_value}', flush=True)
        print(f'sequence_count={len(specs)}', flush=True)
        print(f"sequence_manifest={(repo_path(sequence_manifest) if sequence_manifest else '')}", flush=True)
        print(f'output_root={output_root}', flush=True)
        if log_path is not None:
            print(f'log_file={log_path}', flush=True)
        planned: list[tuple[SequenceSpec, list[dict[str, str]], MainlinePaths, dict[str, Any]]] = []
        preflight_rows: list[dict[str, Any]] = []
        question_scope = question_scope_label(include_types=args.include_types, exclude_types=args.exclude_types)
        for spec in specs:
            selected_rows = selected_questions(load_questions(csv_path, spec.seq_id), include_types=args.include_types, exclude_types=args.exclude_types)
            paths = paths_for_sequence(spec, output_root=output_root, run_name_value=run_name_value, reuse_existing_question_ir=args.reuse_manifest_question_ir and (not args.force_question_ir), reuse_sequence_projection_cache=args.reuse_sequence_projection_cache)
            preflight = preflight_sequence(spec, paths, csv_path=csv_path, selected_rows=selected_rows, require_existing_question_ir=args.skip_question_ir_generation)
            planned.append((spec, selected_rows, paths, preflight))
            preflight_rows.append(preflight)
        if args.preflight_only:
            print_preflight_report(preflight_rows)
            blocked = [row for row in preflight_rows if row['issues']]
            if blocked:
                raise SystemExit(f'Preflight blocked {len(blocked)} sequence(s).')
            print('Preflight passed.', flush=True)
            return
        summary_rows: list[dict[str, Any]] = []
        sequence_run_plans: list[SequenceRunPlan] = []
        for spec, selected_rows, paths, _preflight in planned:
            issues = validate_sequence_inputs(spec, csv_path=csv_path, selected_rows=selected_rows)
            question_ir_matches = question_ir_match_count(paths.question_ir_json, selected_rows)
            needs_ir = args.force_question_ir or question_ir_matches < len(selected_rows)
            print('', flush=True)
            print(f'{spec.display_id}: layout={spec.layout_kind} questions={len(selected_rows)} ir={question_ir_matches}/{len(selected_rows)}', flush=True)
            print(f'  graph={spec.canonical_graph_json}', flush=True)
            print(f'  frame_time_index={spec.frame_time_index_json}', flush=True)
            print(f'  qa_unfilled={spec.qa_unfilled_json}', flush=True)
            print(f'  projection_cache={paths.projection_cache}', flush=True)
            print(f'  output_dir={paths.output_dir}', flush=True)
            if issues:
                for issue in issues:
                    print(f'  issue: {issue}', flush=True)
                if args.run:
                    raise SystemExit(f"Cannot run {spec.display_id}: {'; '.join(issues)}")
                continue
            if needs_ir:
                if args.skip_question_ir_generation:
                    message = f'{spec.display_id} needs question IR ({question_ir_matches}/{len(selected_rows)})'
                    if args.run:
                        raise SystemExit(message)
                    print(f'  planned issue: {message}', flush=True)
                ir_command = build_question_ir_command(spec, paths, csv_path=csv_path, config=args.config, ir_mode=args.ir_mode, domain_profile=args.domain_profile, batch_size=args.ir_batch_size, max_tokens=args.ir_max_tokens, timeout_sec=args.ir_timeout_sec, include_types=args.include_types, exclude_types=args.exclude_types)
                print(f'  question-ir: {command_line(ir_command)}', flush=True)
                if args.run and (not args.skip_question_ir_generation):
                    paths.question_ir_json.parent.mkdir(parents=True, exist_ok=True)
                    run_command(ir_command, retries=args.qa_retries)
                    question_ir_matches = question_ir_match_count(paths.question_ir_json, selected_rows)
                    if question_ir_matches < len(selected_rows):
                        raise SystemExit(f'Question IR incomplete for {spec.display_id}: {question_ir_matches}/{len(selected_rows)}')
            else:
                print(f'  question-ir: reuse {paths.question_ir_json}', flush=True)
            batch_paths: list[Path] = []
            qa_batch_ranges_for_sequence: list[tuple[int, int]] = []
            qa_tasks: list[QABatchTask] = []
            for offset, limit, complete in planned_qa_batch_ranges(paths=paths, total=len(selected_rows), batch_size=args.qa_batch_size, force_qa=args.force_qa, collapse_missing_batches=args.qa_collapse_missing_batches):
                batch_json = expected_batch_json(paths, offset, limit)
                batch_paths.append(batch_json)
                qa_batch_ranges_for_sequence.append((offset, limit))
                if complete:
                    print(f'  qa batch {offset}:{offset + limit}: reuse {batch_json}', flush=True)
                    continue
                qa_command = build_qa_batch_command(spec, paths, csv_path=csv_path, config=args.config, answer_mode=args.answer_mode, domain_profile=args.domain_profile, retrieval_limit=args.retrieval_limit, text_judge_mode=args.text_judge_mode, llm_evidence_rerank=args.llm_evidence_rerank, llm_evidence_rerank_max_candidates=args.llm_evidence_rerank_max_candidates, llm_evidence_rerank_max_selected=args.llm_evidence_rerank_max_selected, answer_candidate_extraction=args.answer_candidate_extraction, answer_candidate_extraction_max_candidates=args.answer_candidate_extraction_max_candidates, answer_candidate_extraction_max_tokens=args.answer_candidate_extraction_max_tokens, experimental_ir_evidence_query_expansion=args.experimental_ir_evidence_query_expansion, experimental_balanced_evidence_candidates=args.experimental_balanced_evidence_candidates, experimental_answerable_candidate_packs=args.experimental_answerable_candidate_packs, answerable_candidate_pack_max_items=args.answerable_candidate_pack_max_items, experimental_unified_evidence_layer=args.experimental_unified_evidence_layer, unified_evidence_max_items=args.unified_evidence_max_items, unified_evidence_graph_quota=args.unified_evidence_graph_quota, unified_evidence_visual_quota=args.unified_evidence_visual_quota, unified_evidence_frame_quota=args.unified_evidence_frame_quota, experimental_synonym_lexicon=args.experimental_synonym_lexicon, synonym_lexicon_json=args.synonym_lexicon_json, experimental_question_router=args.experimental_question_router, experimental_question_router_hybrid_backbone=args.experimental_question_router_hybrid_backbone, experimental_question_router_specialized_backbone=args.experimental_question_router_specialized_backbone, experimental_question_router_mainline=args.experimental_question_router_mainline, experimental_atomic_answer_facts=args.experimental_atomic_answer_facts, atomic_answer_fact_max_items=args.atomic_answer_fact_max_items, experimental_owner_minimal_prompt=args.experimental_owner_minimal_prompt, experimental_visual_faiss_evidence=args.experimental_visual_faiss_evidence, experimental_visual_query_faiss_recall=args.experimental_visual_query_faiss_recall, experimental_visual_faiss_fused_evidence=args.experimental_visual_faiss_fused_evidence, experimental_visual_faiss_graph_anchors=args.experimental_visual_faiss_graph_anchors, experimental_visual_faiss_ger_fusion=args.experimental_visual_faiss_ger_fusion, experimental_visual_faiss_ger_binding=args.experimental_visual_faiss_ger_binding, experimental_visual_faiss_frame_context=args.experimental_visual_faiss_frame_context, experimental_visual_faiss_structural_binding=args.experimental_visual_faiss_structural_binding, experimental_visual_faiss_attribute_owner=args.experimental_visual_faiss_attribute_owner, experimental_visual_faiss_fragment_bridge=args.experimental_visual_faiss_fragment_bridge, experimental_visual_faiss_object_local_facts=args.experimental_visual_faiss_object_local_facts, experimental_visual_faiss_local_subgraph=args.experimental_visual_faiss_local_subgraph, experimental_visual_faiss_local_subgraph_anchor_only=args.experimental_visual_faiss_local_subgraph_anchor_only, experimental_visual_faiss_evidence_layer=args.experimental_visual_faiss_evidence_layer, experimental_visual_faiss_evidence_layer_constraint_binding=args.experimental_visual_faiss_evidence_layer_constraint_binding, experimental_visual_faiss_role_aware_evidence=args.experimental_visual_faiss_role_aware_evidence, experimental_visual_faiss_role_aware_aux=args.experimental_visual_faiss_role_aware_aux, experimental_visual_faiss_role_aware_compression=args.experimental_visual_faiss_role_aware_compression, experimental_visual_faiss_entity_pack=args.experimental_visual_faiss_entity_pack, experimental_visual_faiss_entity_pack_semantic_align=args.experimental_visual_faiss_entity_pack_semantic_align, experimental_visual_faiss_entity_pack_semantic_gated_role=args.experimental_visual_faiss_entity_pack_semantic_gated_role, experimental_visual_faiss_score_contract=args.experimental_visual_faiss_score_contract, experimental_visual_faiss_balanced_graph_visual=args.experimental_visual_faiss_balanced_graph_visual, experimental_visual_faiss_unified_candidates=args.experimental_visual_faiss_unified_candidates, experimental_frame_faiss_query_recall=args.experimental_frame_faiss_query_recall, experimental_frame_faiss_fused_evidence=args.experimental_frame_faiss_fused_evidence, experimental_frame_faiss_graph_anchors=args.experimental_frame_faiss_graph_anchors, experimental_merged_frame_faiss_query_recall=args.experimental_merged_frame_faiss_query_recall, experimental_merged_frame_faiss_fused_evidence=args.experimental_merged_frame_faiss_fused_evidence, experimental_merged_frame_faiss_graph_anchors=args.experimental_merged_frame_faiss_graph_anchors, experimental_frame_window_faiss_query_recall=args.experimental_frame_window_faiss_query_recall, experimental_frame_window_faiss_graph_anchors=args.experimental_frame_window_faiss_graph_anchors, visual_faiss_index_dir=args.visual_faiss_index_dir, visual_faiss_index_name=args.visual_faiss_index_name, visual_faiss_seed_limit=args.visual_faiss_seed_limit, visual_faiss_top_k=args.visual_faiss_top_k, visual_faiss_max_lines=args.visual_faiss_max_lines, visual_faiss_query_phrase_limit=args.visual_faiss_query_phrase_limit, visual_faiss_query_top_k=args.visual_faiss_query_top_k, visual_faiss_query_search_k=args.visual_faiss_query_search_k, visual_faiss_query_min_score=args.visual_faiss_query_min_score, visual_faiss_ger_object_limit=args.visual_faiss_ger_object_limit, visual_faiss_ger_context_radius=args.visual_faiss_ger_context_radius, visual_faiss_ger_context_limit=args.visual_faiss_ger_context_limit, visual_faiss_evidence_layer_sqlite=args.visual_faiss_evidence_layer_sqlite, visual_faiss_evidence_layer_object_limit=args.visual_faiss_evidence_layer_object_limit, visual_faiss_evidence_layer_neighbor_limit=args.visual_faiss_evidence_layer_neighbor_limit, visual_faiss_role_aware_object_limit=args.visual_faiss_role_aware_object_limit, visual_faiss_role_aware_context_limit=args.visual_faiss_role_aware_context_limit, visual_faiss_cross_sequence=args.visual_faiss_cross_sequence, frame_faiss_index_dir=args.frame_faiss_index_dir, frame_faiss_index_name=args.frame_faiss_index_name, frame_faiss_query_text_limit=args.frame_faiss_query_text_limit, frame_faiss_top_k=args.frame_faiss_top_k, frame_faiss_search_k=args.frame_faiss_search_k, frame_faiss_min_score=args.frame_faiss_min_score, frame_faiss_cluster_max_frame_gap=args.frame_faiss_cluster_max_frame_gap, frame_faiss_cluster_max_sec_gap=args.frame_faiss_cluster_max_sec_gap, frame_faiss_max_clusters=args.frame_faiss_max_clusters, frame_faiss_max_lines=args.frame_faiss_max_lines, frame_faiss_cross_sequence=args.frame_faiss_cross_sequence, merged_frame_faiss_index_dir=args.merged_frame_faiss_index_dir, merged_frame_faiss_index_name=args.merged_frame_faiss_index_name, merged_frame_faiss_query_text_limit=args.merged_frame_faiss_query_text_limit, merged_frame_faiss_top_k=args.merged_frame_faiss_top_k, merged_frame_faiss_search_k=args.merged_frame_faiss_search_k, merged_frame_faiss_min_score=args.merged_frame_faiss_min_score, merged_frame_faiss_cluster_max_frame_gap=args.merged_frame_faiss_cluster_max_frame_gap, merged_frame_faiss_cluster_max_sec_gap=args.merged_frame_faiss_cluster_max_sec_gap, merged_frame_faiss_max_clusters=args.merged_frame_faiss_max_clusters, merged_frame_faiss_max_lines=args.merged_frame_faiss_max_lines, merged_frame_faiss_cross_sequence=args.merged_frame_faiss_cross_sequence, frame_window_faiss_window_size_sec=args.frame_window_faiss_window_size_sec, frame_window_faiss_window_stride_sec=args.frame_window_faiss_window_stride_sec, frame_window_faiss_max_frames_per_window=args.frame_window_faiss_max_frames_per_window, offset=offset, limit=limit, batch_json=batch_json, include_types=args.include_types, exclude_types=args.exclude_types)
                print(f'  qa batch {offset}:{offset + limit}: {command_line(qa_command)}', flush=True)
                if args.run:
                    qa_tasks.append(QABatchTask(offset=offset, limit=limit, batch_json=batch_json, command=qa_command, api_key_index=len(qa_tasks)))
            if not args.run:
                previous = load_existing_summary(spec.existing_baseline_json)
                summary_rows.append({'sequence': spec.display_id, 'layout_kind': spec.layout_kind, 'previous': previous, 'new': {'correct': 0, 'total': len(selected_rows), 'accuracy_percent': 0.0}, 'delta_accuracy_percent': None, 'status': 'planned'})
                continue
            for task in qa_tasks:
                task.batch_json.parent.mkdir(parents=True, exist_ok=True)
            sequence_plan = SequenceRunPlan(spec=spec, selected_rows=selected_rows, paths=paths, batch_paths=batch_paths, batch_ranges_for_sequence=qa_batch_ranges_for_sequence, qa_tasks=qa_tasks)
            if args.qa_scheduler == 'global':
                sequence_run_plans.append(sequence_plan)
                continue
            if qa_tasks:
                run_qa_batch_tasks(assign_api_key_indexes(qa_tasks), retries=args.qa_retries, workers=args.qa_workers)
            summary_rows.append(completed_summary_row(spec=spec, selected_rows=selected_rows, paths=paths, csv_path=csv_path, batch_paths=batch_paths, batch_ranges_for_sequence=qa_batch_ranges_for_sequence))
        if args.run and args.qa_scheduler == 'global':
            run_global_qa_batch_tasks(sequence_run_plans, retries=args.qa_retries, workers=args.qa_workers, warmup_per_sequence=not args.qa_collapse_missing_batches)
            for sequence_plan in sequence_run_plans:
                summary_rows.append(completed_summary_row(spec=sequence_plan.spec, selected_rows=sequence_plan.selected_rows, paths=sequence_plan.paths, csv_path=csv_path, batch_paths=sequence_plan.batch_paths, batch_ranges_for_sequence=sequence_plan.batch_ranges_for_sequence))
        run_metadata = {'sequence_manifest': str(repo_path(sequence_manifest)) if sequence_manifest is not None else '', 'output_root': str(output_root), 'log_path': str(log_path) if log_path is not None else '', 'qa_scope': question_scope, 'include_types': args.include_types or list(NON_POSITION_QA_TYPES), 'exclude_types': args.exclude_types or [], 'answer_mode': args.answer_mode, 'text_judge_mode': args.text_judge_mode, 'qa_scheduler': args.qa_scheduler, 'qa_workers': args.qa_workers, 'qa_batch_size': args.qa_batch_size, 'qa_collapse_missing_batches': bool(args.qa_collapse_missing_batches), 'qa_global_warmup_per_sequence': not args.qa_collapse_missing_batches, 'llm_evidence_rerank': bool(args.llm_evidence_rerank), 'llm_evidence_rerank_max_candidates': args.llm_evidence_rerank_max_candidates, 'llm_evidence_rerank_max_selected': args.llm_evidence_rerank_max_selected, 'answer_candidate_extraction': bool(args.answer_candidate_extraction), 'experimental_unified_evidence_layer': bool(args.experimental_unified_evidence_layer), 'unified_evidence_max_items': args.unified_evidence_max_items, 'unified_evidence_graph_quota': args.unified_evidence_graph_quota, 'unified_evidence_visual_quota': args.unified_evidence_visual_quota, 'unified_evidence_frame_quota': args.unified_evidence_frame_quota, 'experimental_atomic_answer_facts': bool(args.experimental_atomic_answer_facts), 'atomic_answer_fact_max_items': args.atomic_answer_fact_max_items, 'experimental_owner_minimal_prompt': bool(args.experimental_owner_minimal_prompt), 'experimental_synonym_lexicon': bool(args.experimental_synonym_lexicon), 'synonym_lexicon_json': str(args.synonym_lexicon_json), 'experimental_question_router': bool(args.experimental_question_router), 'experimental_question_router_hybrid_backbone': bool(args.experimental_question_router_hybrid_backbone), 'experimental_question_router_specialized_backbone': bool(args.experimental_question_router_specialized_backbone), 'experimental_question_router_mainline': bool(args.experimental_question_router_mainline), 'experimental_visual_faiss_frame_context': bool(args.experimental_visual_faiss_frame_context), 'experimental_visual_faiss_structural_binding': bool(args.experimental_visual_faiss_structural_binding), 'experimental_visual_faiss_attribute_owner': bool(args.experimental_visual_faiss_attribute_owner), 'experimental_visual_faiss_fragment_bridge': bool(args.experimental_visual_faiss_fragment_bridge), 'experimental_visual_faiss_object_local_facts': bool(args.experimental_visual_faiss_object_local_facts), 'experimental_visual_faiss_local_subgraph': bool(args.experimental_visual_faiss_local_subgraph), 'experimental_visual_faiss_local_subgraph_anchor_only': bool(args.experimental_visual_faiss_local_subgraph_anchor_only), 'experimental_visual_faiss_evidence_layer': bool(args.experimental_visual_faiss_evidence_layer), 'experimental_visual_faiss_evidence_layer_constraint_binding': bool(args.experimental_visual_faiss_evidence_layer_constraint_binding), 'experimental_visual_faiss_role_aware_evidence': bool(args.experimental_visual_faiss_role_aware_evidence), 'experimental_visual_faiss_role_aware_aux': bool(args.experimental_visual_faiss_role_aware_aux), 'experimental_visual_faiss_role_aware_compression': bool(args.experimental_visual_faiss_role_aware_compression), 'experimental_visual_faiss_entity_pack': bool(args.experimental_visual_faiss_entity_pack), 'experimental_visual_faiss_entity_pack_semantic_align': bool(args.experimental_visual_faiss_entity_pack_semantic_align), 'experimental_visual_faiss_entity_pack_semantic_gated_role': bool(args.experimental_visual_faiss_entity_pack_semantic_gated_role), 'experimental_visual_faiss_score_contract': bool(args.experimental_visual_faiss_score_contract), 'visual_faiss_evidence_layer_sqlite': str(args.visual_faiss_evidence_layer_sqlite), 'visual_faiss_evidence_layer_object_limit': args.visual_faiss_evidence_layer_object_limit, 'visual_faiss_evidence_layer_neighbor_limit': args.visual_faiss_evidence_layer_neighbor_limit, 'visual_faiss_role_aware_object_limit': args.visual_faiss_role_aware_object_limit, 'visual_faiss_role_aware_context_limit': args.visual_faiss_role_aware_context_limit, 'experimental_visual_faiss_balanced_graph_visual': bool(args.experimental_visual_faiss_balanced_graph_visual), 'experimental_answerable_candidate_packs': bool(args.experimental_answerable_candidate_packs), 'answerable_candidate_pack_max_items': args.answerable_candidate_pack_max_items, 'question_ir_policy': 'reuse manifest/run-local IR; generate missing IR unless --skip-question-ir-generation is supplied'}
        pipeline_metadata = manifest_pipeline_metadata(sequence_manifest)
        ir_qa_id = ir_qa_id_from_flags(visual_faiss_focus_seed=args.experimental_visual_faiss_evidence, visual_faiss_query_recall=args.experimental_visual_query_faiss_recall, visual_faiss_fused_evidence=args.experimental_visual_faiss_fused_evidence, visual_faiss_graph_anchors=args.experimental_visual_faiss_graph_anchors, visual_faiss_ger_fusion=args.experimental_visual_faiss_ger_fusion, visual_faiss_ger_binding=args.experimental_visual_faiss_ger_binding, visual_faiss_frame_context=args.experimental_visual_faiss_frame_context, visual_faiss_structural_binding=args.experimental_visual_faiss_structural_binding, visual_faiss_attribute_owner=args.experimental_visual_faiss_attribute_owner, owner_minimal_prompt=args.experimental_owner_minimal_prompt, visual_faiss_fragment_bridge=args.experimental_visual_faiss_fragment_bridge, visual_faiss_object_local_facts=args.experimental_visual_faiss_object_local_facts, visual_faiss_local_subgraph=args.experimental_visual_faiss_local_subgraph, visual_faiss_local_subgraph_anchor_only=args.experimental_visual_faiss_local_subgraph_anchor_only, visual_faiss_evidence_layer=args.experimental_visual_faiss_evidence_layer, visual_faiss_evidence_layer_constraint_binding=args.experimental_visual_faiss_evidence_layer_constraint_binding, visual_faiss_role_aware_evidence=args.experimental_visual_faiss_role_aware_evidence, visual_faiss_role_aware_aux=args.experimental_visual_faiss_role_aware_aux, visual_faiss_role_aware_compression=args.experimental_visual_faiss_role_aware_compression, visual_faiss_entity_pack=args.experimental_visual_faiss_entity_pack, visual_faiss_entity_pack_semantic_align=args.experimental_visual_faiss_entity_pack_semantic_align, visual_faiss_entity_pack_semantic_gated_role=args.experimental_visual_faiss_entity_pack_semantic_gated_role, visual_faiss_score_contract=args.experimental_visual_faiss_score_contract, visual_faiss_balanced_graph_visual=args.experimental_visual_faiss_balanced_graph_visual, answerable_candidate_packs=args.experimental_answerable_candidate_packs, visual_faiss_unified_candidates=args.experimental_visual_faiss_unified_candidates, synonym_lexicon=args.experimental_synonym_lexicon, question_router=args.experimental_question_router, question_router_hybrid_backbone=args.experimental_question_router_hybrid_backbone, question_router_specialized_backbone=args.experimental_question_router_specialized_backbone, question_router_mainline=args.experimental_question_router_mainline, frame_faiss_query_recall=args.experimental_frame_faiss_query_recall, frame_faiss_fused_evidence=args.experimental_frame_faiss_fused_evidence, frame_faiss_graph_anchors=args.experimental_frame_faiss_graph_anchors, merged_frame_faiss_query_recall=args.experimental_merged_frame_faiss_query_recall, merged_frame_faiss_fused_evidence=args.experimental_merged_frame_faiss_fused_evidence, merged_frame_faiss_graph_anchors=args.experimental_merged_frame_faiss_graph_anchors, frame_window_faiss_query_recall=args.experimental_frame_window_faiss_query_recall, frame_window_faiss_graph_anchors=args.experimental_frame_window_faiss_graph_anchors)
        ir_qa_id = resolve_run_ir_qa_id(flag_ir_qa_id=ir_qa_id, pipeline_metadata=pipeline_metadata)
        pipeline_metadata['ir_qa_id'] = ir_qa_id
        pipeline_metadata['route_id'] = route_id_for(cg_id=pipeline_metadata.get('cg_id'), ir_qa_id=ir_qa_id)
        pipeline_metadata['visual_faiss_version_policy'] = 'CG6 supports graph-only and graph plus visual evidence routes. Current route behavior is declared by configs/datasets/navqa/routes.json.'
        run_metadata.update(pipeline_metadata)
        if args.run:
            summary_json, summary_md = write_all_sequence_summary(output_root=output_root, run_name_value=run_name_value, rows=summary_rows, run_metadata=run_metadata)
            print('', flush=True)
            print(f'summary_json={summary_json}', flush=True)
            print(f'summary_md={summary_md}', flush=True)
        else:
            print('', flush=True)
            print('Dry-run only. Add --run to execute Question IR and QA batches.', flush=True)
    finally:
        if log_handle is not None:
            log_handle.flush()
if __name__ == '__main__':
    main()
