from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
PROMPT_DIR = SCRIPT_DIR / 'prompts'
PROMPT_VARIANTS = ('mmbench', 'mmbench-extra', 'judge-v2')
DEFAULT_DATASET = (REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json')
DEFAULT_MODEL = 'gpt-5.6-luna'
DEFAULT_BASE_URL = 'https://spacetimeai.cc/v1'
OPENAI_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'
_API_KEY_POOL_LOCK = threading.Lock()
_API_KEY_POOL_INDEX = 0

def configured_api_keys(*, explicit_key: str | None, env_var: str) -> list[str]:
    """Return a de-duplicated OpenEQA-only key pool without logging secrets."""
    if explicit_key:
        return [explicit_key.strip()]
    raw = os.environ.get(env_var, '')
    values: list[str] = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            values.extend((str(item).strip() for item in parsed))
        else:
            values.extend((part.strip() for part in re.split('[,;\\r\\n]+', raw)))
    if not values and env_var != 'OPENAI_API_KEY':
        values.append(os.environ.get('OPENAI_API_KEY', '').strip())
    return list(dict.fromkeys((value for value in values if value)))

def next_api_key(keys: list[str]) -> str:
    global _API_KEY_POOL_INDEX
    if not keys:
        return ''
    with _API_KEY_POOL_LOCK:
        key = keys[_API_KEY_POOL_INDEX % len(keys)]
        _API_KEY_POOL_INDEX += 1
    return key

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Vendored OpenEQA-style prediction evaluator. Loads OpenEQA QA rows, pairs them with prediction JSON, and judges semantic correctness with an OpenAI-compatible model.')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--prompt-variant', choices=PROMPT_VARIANTS, default='mmbench')
    parser.add_argument('--data-source', choices=('all', 'hm3d-v0', 'scannet-v0'), default='all', help="Evaluation scope. 'all' is the formal OpenEQA mode and requires one prediction for every dataset question. Source values are retained only for diagnostic or partial-run evaluation.")
    parser.add_argument('--question-id', action='append', default=None, help='Optional question_id filter. Can be repeated.')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--api-key-index', type=int, default=None, help='Pin Judge to one zero-based configured key-pool slot.')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--qa-workers', type=int, default=1, help='Concurrent Judge requests; each request rotates through the configured key pool.')
    parser.add_argument('--api-key-pool-env-var', default='OPENAI_API_KEYS', help='Environment variable containing comma-, newline-, or JSON-list-separated API keys.')
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true')
    return parser

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(6):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt >= 5:
                raise
            time.sleep(0.25 * (attempt + 1))

def load_prompt(variant: str) -> str:
    prompt_path = PROMPT_DIR / f'{variant}.txt'
    if not prompt_path.exists():
        raise FileNotFoundError(f'Missing prompt template: {prompt_path}')
    return prompt_path.read_text(encoding='utf-8')

def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def normalize_predictions(payload: Any) -> dict[str, dict[str, Any]]:
    by_question_id: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault('question_id', key)
            else:
                item = {'question_id': key, 'prediction': value}
            items.append(item)
    else:
        raise ValueError('predictions JSON must be a list or dict')
    for item in items:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get('question_id') or item.get('uuid') or '').strip()
        if not question_id:
            continue
        if question_id in by_question_id:
            raise ValueError(f'duplicate prediction question_id: {question_id}')
        by_question_id[question_id] = item
    return by_question_id

def normalize_dataset_rows(dataset_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(dataset_payload, list):
        raise ValueError('OpenEQA dataset must be a list')
    rows: list[dict[str, Any]] = []
    for item in dataset_payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows

def slice_rows(rows: list[dict[str, Any]], *, offset: int, limit: int | None) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError('--offset must be >= 0')
    if limit is None:
        return rows[offset:]
    if limit < 0:
        raise ValueError('--limit must be >= 0')
    return rows[offset:offset + limit]

def filter_rows_by_question_ids(rows: list[dict[str, Any]], question_ids: list[str] | None) -> list[dict[str, Any]]:
    if not question_ids:
        return rows
    wanted = {item.strip() for item in question_ids if item.strip()}
    return [row for row in rows if question_id_from_row(row) in wanted]

def filter_rows_with_predictions(rows: list[dict[str, Any]], prediction_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if question_id_from_row(row) in prediction_map]

def filter_rows_by_data_source(rows: list[dict[str, Any]], data_source: str) -> list[dict[str, Any]]:
    if data_source == 'all':
        return rows
    prefix = f'{data_source}/'
    return [row for row in rows if str(row.get('episode_history') or '').startswith(prefix)]

def complete_prediction_coverage_issues(rows: list[dict[str, Any]], prediction_map: dict[str, dict[str, Any]]) -> list[str]:
    expected_ids = {question_id_from_row(row) for row in rows}
    expected_ids.discard('')
    actual_ids = set(prediction_map)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    issues: list[str] = []
    if missing:
        issues.append(f'missing {len(missing)} OpenEQA predictions; sample={missing[:8]}')
    if extra:
        issues.append(f'found {len(extra)} predictions outside OpenEQA; sample={extra[:8]}')
    return issues

def data_source_from_row(row: dict[str, Any]) -> str:
    return str(row.get('episode_history') or '').partition('/')[0]

def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []

def judge_result_failed(item: dict[str, Any]) -> bool:
    reason = str(item.get('reason') or '').strip().lower()
    return reason.startswith('judge request failed:')

def question_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get('question_id') or row.get('uuid') or '').strip()

def question_text(row: dict[str, Any]) -> str:
    return str(row.get('question') or row.get('query') or '').strip()

def answer_text(row: dict[str, Any]) -> str:
    return str(row.get('answer') or row.get('gold_answer') or '').strip()

def extra_answers(row: dict[str, Any]) -> list[str]:
    value = row.get('extra_answers')
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def build_messages(*, prompt: str, question: str, gold_answer: str, predicted_answer: str, extra: list[str]) -> list[dict[str, str]]:
    user_lines = [f'Question: {question}', f'Reference answer: {gold_answer}', f'Predicted answer: {predicted_answer}']
    if extra:
        user_lines.append('Extra acceptable answers: ' + '; '.join(extra))
    user_lines.extend(['', 'Return only JSON with keys: score, reason, matched_answer.', 'score must be an integer from 1 to 5.', '5 = exactly or clearly correct.', '4 = essentially correct, small wording difference.', '3 = partially correct or ambiguous.', '2 = weakly related but mostly wrong.', '1 = incorrect.'])
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': '\n'.join(user_lines)}]

def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith('```'):
        text = re.sub('^```(?:json)?\\s*', '', text)
        text = re.sub('\\s*```$', '', text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search('\\{.*\\}', text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def normalize_score(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {'score': 1, 'reason': 'model response was not valid JSON', 'matched_answer': ''}
    score = obj.get('score')
    try:
        score_int = int(score)
    except Exception:
        score_int = 1
    score_int = max(1, min(5, score_int))
    return {'score': score_int, 'reason': str(obj.get('reason') or '').strip(), 'matched_answer': str(obj.get('matched_answer') or '').strip()}

def score_to_percent(score: int) -> float:
    return round((score - 1) / 4.0 * 100.0, 2)

def call_judge(*, client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, retries: int) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, response_format={'type': 'json_object'})
            content = (response.choices[0].message.content or '').strip()
            parsed = extract_json_object(content)
            normalized = normalize_score(parsed)
            normalized['raw_response'] = content
            normalized['usage'] = {'prompt_tokens': int(getattr(response.usage, 'prompt_tokens', 0) or 0), 'completion_tokens': int(getattr(response.usage, 'completion_tokens', 0) or 0), 'total_tokens': int(getattr(response.usage, 'total_tokens', 0) or 0)}
            return normalized
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
    return {'score': 1, 'reason': f"judge request failed: {last_error or 'unknown error'}", 'matched_answer': '', 'raw_response': '', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}

def call_judge_with_key_failover(*, client_cache: dict[str, OpenAI], key_pool: list[str], model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, retries: int) -> dict[str, Any]:
    keys = key_pool or ['']
    last_result: dict[str, Any] | None = None
    for _ in range(len(keys)):
        selected_key = next_api_key(keys)
        result = call_judge(client=client_cache[selected_key], model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, retries=retries)
        if not judge_result_failed(result):
            return result
        last_result = result
    return last_result or {'score': 1, 'reason': 'judge request failed: no configured client', 'matched_answer': '', 'raw_response': '', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}

def main() -> None:
    args = build_parser().parse_args()
    if OpenAI is None:
        raise SystemExit('The OpenAI Python SDK is required for OpenEQA evaluation.')
    if args.qa_workers < 1:
        raise SystemExit('--qa-workers must be >= 1')
    dataset_path = args.dataset if args.dataset.is_absolute() else REPO_ROOT / args.dataset
    predictions_path = args.predictions if args.predictions.is_absolute() else REPO_ROOT / args.predictions
    if not dataset_path.exists():
        raise SystemExit(f'Missing dataset: {dataset_path}')
    if not predictions_path.exists():
        raise SystemExit(f'Missing predictions file: {predictions_path}')
    prediction_map = normalize_predictions(load_json(predictions_path))
    all_rows = normalize_dataset_rows(load_json(dataset_path))
    if args.data_source == 'all':
        if args.question_id or args.offset or args.limit is not None:
            raise SystemExit('--data-source all is the formal full OpenEQA mode and cannot be combined with --question-id, --offset, or --limit')
        coverage_issues = complete_prediction_coverage_issues(all_rows, prediction_map)
        if coverage_issues:
            raise SystemExit('; '.join(coverage_issues))
    source_rows = filter_rows_by_data_source(all_rows, args.data_source)
    dataset_rows = slice_rows(filter_rows_by_question_ids(filter_rows_with_predictions(source_rows, prediction_map), args.question_id), offset=args.offset, limit=args.limit)
    prompt = load_prompt(args.prompt_variant)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = predictions_path.parent / f"open_eqa_eval_{args.model.replace('/', '_')}"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.json'
    summary_path = output_dir / 'summary.json'
    summary_md_path = output_dir / 'summary.md'
    prompt_snapshot_path = output_dir / 'prompt_snapshot.txt'
    prompt_hash = prompt_sha256(prompt)
    if prompt_snapshot_path.exists():
        snapshot = prompt_snapshot_path.read_text(encoding='utf-8')
        if snapshot != prompt:
            raise SystemExit('Prompt drift detected for this output directory; use a new output directory instead of mixing Judge prompt versions.')
    else:
        with prompt_snapshot_path.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(prompt)
    selected_question_ids = {question_id_from_row(row) for row in dataset_rows if question_id_from_row(row)}
    existing_results = load_existing_results(results_path) if args.resume else []
    existing_results = [item for item in existing_results if question_id_from_row(item) in selected_question_ids and (not judge_result_failed(item))]
    result_by_id = {question_id_from_row(item): item for item in existing_results if question_id_from_row(item)}
    token_usage = Counter()
    for item in existing_results:
        usage = item.get('usage') if isinstance(item, dict) else None
        if isinstance(usage, dict):
            token_usage.update(usage)
    base_url: str
    if args.base_url:
        base_url = args.base_url
    elif os.environ.get('OPENAI_BASE_URL'):
        base_url = os.environ['OPENAI_BASE_URL']
    else:
        base_url = DEFAULT_BASE_URL
    key_pool = configured_api_keys(explicit_key=args.api_key, env_var=args.api_key_pool_env_var)
    if args.api_key_index is not None:
        if args.api_key_index < 0 or args.api_key_index >= len(key_pool):
            raise SystemExit(f'api-key-index {args.api_key_index} is outside configured pool size {len(key_pool)}')
        key_pool = [key_pool[args.api_key_index]]
    client_cache: dict[str, OpenAI] = {}
    if not args.dry_run:
        for key in key_pool or ['']:
            client_kwargs: dict[str, Any] = {'base_url': base_url, 'default_headers': {'User-Agent': OPENAI_BROWSER_USER_AGENT}}
            if key:
                client_kwargs['api_key'] = key
            client_cache[key] = OpenAI(timeout=args.timeout, **client_kwargs)
    pending_rows = [(index, row) for index, row in enumerate(dataset_rows, start=1) if (qid := question_id_from_row(row)) and qid not in result_by_id]

    def judge_one(item: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any]]:
        index, row = item
        qid = question_id_from_row(row)
        gold = answer_text(row)
        pred_item = prediction_map.get(qid, {})
        predicted = str(pred_item.get('prediction') or pred_item.get('answer') or pred_item.get('official_prediction') or '').strip()
        messages = build_messages(prompt=prompt, question=question_text(row), gold_answer=gold, predicted_answer=predicted, extra=extra_answers(row))
        if args.dry_run:
            judged = {'score': 1, 'reason': 'dry-run', 'matched_answer': '', 'raw_response': '', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        else:
            judged = call_judge_with_key_failover(client_cache=client_cache, key_pool=key_pool, model=args.model, messages=messages, temperature=args.temperature, max_tokens=args.max_tokens, retries=args.retries)
        record = {'question_id': qid, 'question': question_text(row), 'gold_answer': gold, 'prediction': predicted, 'category': str(row.get('category') or '').strip(), 'episode_history': str(row.get('episode_history') or '').strip(), 'data_source': data_source_from_row(row), 'score': judged['score'], 'normalized_score': score_to_percent(int(judged['score'])), 'reason': judged['reason'], 'matched_answer': judged['matched_answer'], 'raw_response': judged.get('raw_response') or '', 'model': args.model, 'prompt_variant': args.prompt_variant, 'usage': judged.get('usage') or {}, 'evaluated_at': datetime.now(timezone.utc).isoformat()}
        return (index, qid, record)

    def ordered_results() -> list[dict[str, Any]]:
        return [result_by_id[qid] for row in dataset_rows if (qid := question_id_from_row(row)) in result_by_id]
    with ThreadPoolExecutor(max_workers=args.qa_workers) as executor:
        futures = {executor.submit(judge_one, item): item for item in pending_rows}
        for future in as_completed(futures):
            index, qid, record = future.result()
            result_by_id[qid] = record
            token_usage.update(record.get('usage') or {})
            results = ordered_results()
            write_json_atomic(results_path, results)
            print(f"[{index}/{len(dataset_rows)}] judged {qid} score={record.get('score')} status={('error' if judge_result_failed(record) else 'ok')}", flush=True)
    results = ordered_results()
    scores = [int(item.get('score') or 1) for item in results]
    normalized_scores = [float(item.get('normalized_score') or 0.0) for item in results]
    score_counts = Counter((str(item) for item in scores))
    judge_error_count = sum((1 for item in results if judge_result_failed(item)))
    by_data_source: dict[str, dict[str, Any]] = {}
    for data_source in sorted({str(item.get('data_source') or '') or data_source_from_row(item) for item in results}):
        if not data_source:
            continue
        source_results = [item for item in results if (str(item.get('data_source') or '') or data_source_from_row(item)) == data_source]
        source_scores = [int(item.get('score') or 1) for item in source_results]
        source_normalized = [float(item.get('normalized_score') or 0.0) for item in source_results]
        by_data_source[data_source] = {'evaluated_count': len(source_results), 'score_mean': round(sum(source_scores) / len(source_scores), 4), 'normalized_score_mean': round(sum(source_normalized) / len(source_normalized), 4)}
    summary = {'contract': 'open_eqa_official_style_eval_v2', 'dataset_name': 'OpenEQA', 'data_source_scope': args.data_source, 'status': 'complete' if len(results) == len(dataset_rows) and judge_error_count == 0 else 'partial', 'dataset': str(dataset_path), 'predictions': str(predictions_path), 'output_dir': str(output_dir), 'model': args.model, 'prompt_variant': args.prompt_variant, 'prompt_sha256': prompt_hash, 'prompt_snapshot': str(prompt_snapshot_path), 'qa_workers': args.qa_workers, 'api_key_pool_env_var': args.api_key_pool_env_var, 'api_key_pool_size': len(key_pool), 'evaluated_count': len(results), 'expected_count': len(dataset_rows), 'judge_error_count': judge_error_count, 'score_mean': round(sum(scores) / len(scores), 4) if scores else 0.0, 'normalized_score_mean': round(sum(normalized_scores) / len(normalized_scores), 4) if normalized_scores else 0.0, 'score_counts': dict(sorted(score_counts.items())), 'by_data_source': by_data_source, 'usage': {'prompt_tokens': int(token_usage.get('prompt_tokens', 0)), 'completion_tokens': int(token_usage.get('completion_tokens', 0)), 'total_tokens': int(token_usage.get('total_tokens', 0))}, 'generated_at': datetime.now(timezone.utc).isoformat()}
    write_json_atomic(summary_path, summary)
    summary_md_path.write_text('\n'.join([f'# OpenEQA Official-Style Eval: {args.model}', '', f'- dataset: `{dataset_path}`', f'- predictions: `{predictions_path}`', f'- output_dir: `{output_dir}`', f'- prompt_variant: `{args.prompt_variant}`', f"- qa_workers: `{summary['qa_workers']}`", f"- api_key_pool_size: `{summary['api_key_pool_size']}`", f"- evaluated_count: `{summary['evaluated_count']}`", f"- status: `{summary['status']}`", f"- judge_error_count: `{summary['judge_error_count']}`", f"- score_mean: `{summary['score_mean']}`", f"- normalized_score_mean: `{summary['normalized_score_mean']}`", f"- data_source_scope: `{summary['data_source_scope']}`", f"- prompt tokens: `{summary['usage']['prompt_tokens']}`", f"- completion tokens: `{summary['usage']['completion_tokens']}`", f"- total tokens: `{summary['usage']['total_tokens']}`", '', '## Score Counts', '', *[f'- `{k}`: `{v}`' for k, v in summary['score_counts'].items()], '', '## Source Diagnostics', '', *[f"- `{source}`: count=`{values['evaluated_count']}`, normalized=`{values['normalized_score_mean']}`" for source, values in summary['by_data_source'].items()], '']), encoding='utf-8')
    print(f'results={results_path}')
    print(f'summary={summary_path}')
    print(f'summary_md={summary_md_path}')
    if args.data_source == 'all' and summary['status'] != 'complete':
        raise SystemExit(2)
if __name__ == '__main__':
    main()
