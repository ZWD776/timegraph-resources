from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
DEFAULT_ROUTES_CONFIG = Path('configs/datasets/navqa/routes.json')
DEFAULT_JOB_ROOT = Path('datasets/navqa/scratch/navqa-scratch-external-v1/payload/jobs')

@dataclass(frozen=True)
class Route:
    route_id: str
    description: str
    sequence_manifest: str
    output_root: str
    config: str | None
    args: tuple[str, ...]

@dataclass(frozen=True)
class RouteRegistry:
    defaults: dict[str, Any]
    routes: dict[str, Route]

@dataclass(frozen=True)
class QACommandOptions:
    mode: str = 'print'
    run_name: str | None = None
    seq_ids: tuple[str, ...] = ()
    include_types: tuple[str, ...] | None = None
    exclude_types: tuple[str, ...] = ()
    qa_workers: int | None = None
    qa_scheduler: str | None = None
    qa_retries: int | None = None
    qa_batch_size: int | None = None
    retrieval_limit: int | None = None
    force_qa: bool = False
    force_question_ir: bool = False
    extra_args: tuple[str, ...] = ()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run stable NaVQA CG/IR-QA routes through one operational entrypoint. This wrapper only assembles existing commands; it does not change graph, retrieval, QA, or judging logic.')
    parser.add_argument('--routes-config', type=Path, default=DEFAULT_ROUTES_CONFIG, help='Route registry JSON. Defaults to configs/datasets/navqa/routes.json.')
    parser.add_argument('--route', help='Route id from --list.')
    parser.add_argument('--list', action='store_true', help='List available stable routes and exit.')
    parser.add_argument('--status', nargs='?', const='latest', default=None, help='Show a background job status. Omit the value to inspect the latest job.')
    parser.add_argument('--summarize', type=str, default=None, help='Print a compact summary from a run directory, summary JSON, job id, or background job directory.')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true', help='Run the mainline preflight only. No IR or QA is generated.')
    action.add_argument('--run', action='store_true', help='Run QA for the selected route.')
    action.add_argument('--resume', action='store_true', help='Resume QA by reusing complete batches and filling missing batches.')
    action.add_argument('--build-graphs', action='store_true', help='Build the graph artifacts required by the route.')
    action.add_argument('--build-graphs-dry-run', action='store_true', help='Print graph build commands without executing them.')
    parser.add_argument('--background', action='store_true', help='Start --run/--resume/--build-graphs in the background and write logs.')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seq-id', action='append', nargs='+', default=None, help='Restrict to one or more sequence ids. May be repeated.')
    parser.add_argument('--include-types', nargs='+', default=None, help="Override the route's QA type scope.")
    parser.add_argument('--exclude-types', nargs='+', default=None, help='Exclude question types after include filtering.')
    parser.add_argument('--qa-workers', type=int, default=None)
    parser.add_argument('--qa-scheduler', choices=('sequence', 'global'), default=None)
    parser.add_argument('--qa-retries', type=int, default=None)
    parser.add_argument('--qa-batch-size', type=int, default=None)
    parser.add_argument('--retrieval-limit', type=int, default=None)
    parser.add_argument('--force-qa', action='store_true')
    parser.add_argument('--force-question-ir', action='store_true')
    parser.add_argument('--job-root', type=Path, default=DEFAULT_JOB_ROOT, help='Background job directory root.')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, help="Extra arguments passed to the underlying QA runner after '--'.")
    return parser

def repo_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value

def load_registry(path: Path=DEFAULT_ROUTES_CONFIG) -> RouteRegistry:
    registry_path = repo_path(path)
    if registry_path is None or not registry_path.exists():
        raise FileNotFoundError(f'route registry not found: {path}')
    payload = json.loads(registry_path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError('route registry must be a JSON object')
    defaults = payload.get('defaults')
    raw_routes = payload.get('routes')
    if not isinstance(defaults, dict):
        raise ValueError('route registry missing defaults object')
    if not isinstance(raw_routes, list):
        raise ValueError('route registry missing routes list')
    routes: dict[str, Route] = {}
    for item in raw_routes:
        if not isinstance(item, dict):
            raise ValueError('route entries must be JSON objects')
        route_id = str(item.get('id') or '').strip()
        if not route_id:
            raise ValueError('route entry missing id')
        if route_id in routes:
            raise ValueError(f'duplicate route id: {route_id}')
        sequence_manifest = str(item.get('sequence_manifest') or '').strip()
        output_root = str(item.get('output_root') or '').strip()
        if not sequence_manifest or not output_root:
            raise ValueError(f'route {route_id} missing manifest/output_root')
        args = item.get('args') or []
        if not isinstance(args, list) or not all((isinstance(value, str) for value in args)):
            raise ValueError(f'route {route_id} args must be a string list')
        routes[route_id] = Route(route_id=route_id, description=str(item.get('description') or ''), sequence_manifest=sequence_manifest, output_root=output_root, config=str(item['config']) if item.get('config') else None, args=tuple(args))
    return RouteRegistry(defaults=defaults, routes=routes)

def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])

def build_qa_command(route: Route, defaults: dict[str, Any], options: QACommandOptions) -> list[str]:
    runner = str(defaults.get('runner') or '').strip()
    if not runner:
        raise ValueError('route defaults missing runner')
    command = [sys.executable, runner]
    append_option(command, '--csv', defaults.get('csv'))
    append_option(command, '--layout-root', defaults.get('layout_root'))
    append_option(command, '--sequence-manifest', route.sequence_manifest)
    append_option(command, '--output-root', route.output_root)
    append_option(command, '--config', route.config or defaults.get('config'))
    append_option(command, '--answer-mode', defaults.get('answer_mode'))
    append_option(command, '--ir-mode', defaults.get('ir_mode'))
    append_option(command, '--domain-profile', defaults.get('domain_profile'))
    append_option(command, '--retrieval-limit', value_or_default(options.retrieval_limit, defaults.get('retrieval_limit')))
    append_option(command, '--qa-batch-size', value_or_default(options.qa_batch_size, defaults.get('qa_batch_size')))
    append_option(command, '--qa-workers', value_or_default(options.qa_workers, defaults.get('qa_workers')))
    append_option(command, '--qa-scheduler', value_or_default(options.qa_scheduler, defaults.get('qa_scheduler')))
    append_option(command, '--qa-retries', value_or_default(options.qa_retries, defaults.get('qa_retries')))
    include_types = options.include_types
    if include_types is None:
        raw_include_types = defaults.get('include_types') or []
        include_types = tuple((str(value) for value in raw_include_types))
    if include_types:
        command.append('--include-types')
        command.extend(include_types)
    if options.exclude_types:
        command.append('--exclude-types')
        command.extend(options.exclude_types)
    if options.seq_ids:
        command.append('--seq-id')
        command.extend(options.seq_ids)
    if options.run_name:
        append_option(command, '--run-name', options.run_name)
    if options.force_qa:
        command.append('--force-qa')
    if options.force_question_ir:
        command.append('--force-question-ir')
    command.extend(route.args)
    if options.mode in {'run', 'resume'}:
        command.append('--run')
    elif options.mode == 'preflight':
        command.append('--preflight-only')
    elif options.mode != 'print':
        raise ValueError(f'unknown QA command mode: {options.mode}')
    command.extend(strip_remainder_marker(options.extra_args))
    return [str(part) for part in command]

def build_graph_commands(route: Route, *, seq_ids: tuple[str, ...]=()) -> list[list[str]]:
    manifest_path = repo_path(route.sequence_manifest)
    if manifest_path is None:
        raise ValueError(f'route {route.route_id} has no sequence manifest')
    payload = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    cg_id = str(payload.get('cg_id') or '')
    command = [sys.executable, 'scripts/datasets/navqa/build/rebuild_navqa_clean_manifest_graphs.py', route.sequence_manifest]
    clean_policy = payload.get('clean_policy') if isinstance(payload, dict) else {}
    if isinstance(clean_policy, dict) and clean_policy.get('direct_from_hsgg_json') is True:
        command.append('--rebuild-temporal')
    return [[str(part) for part in command]]

def question_ir_run_root_from_manifest(manifest: dict[str, Any]) -> str | None:
    roots: set[str] = set()
    for item in manifest.get('sequences', []):
        if not isinstance(item, dict) or not item.get('question_ir_json'):
            continue
        path = Path(str(item['question_ir_json']))
        if len(path.parts) >= 3:
            roots.add(path.parent.parent.as_posix())
    return next(iter(roots)) if len(roots) == 1 else None

def append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == '':
        return
    command.extend([flag, str(value)])

def value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value

def strip_remainder_marker(values: tuple[str, ...]) -> tuple[str, ...]:
    if values and values[0] == '--':
        return values[1:]
    return values

def flatten_seq_ids(values: list[list[str]] | None) -> tuple[str, ...]:
    if not values:
        return ()
    flattened: list[str] = []
    for group in values:
        flattened.extend((str(value) for value in group))
    return tuple(flattened)

def print_routes(registry: RouteRegistry) -> None:
    print('Available routes:')
    for route_id in sorted(registry.routes):
        route = registry.routes[route_id]
        print(f'- {route.route_id}')
        if route.description:
            print(f'  {route.description}')
        print(f'  manifest: {route.sequence_manifest}')
        print(f'  output_root: {route.output_root}')

def safe_job_id(route_id: str, action: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    route_slug = re.sub('[^A-Za-z0-9_.-]+', '_', route_id).strip('_')
    return f'{timestamp}_{route_slug}_{action}'

def start_background(command: list[str], *, route_id: str, action: str, run_name: str | None, job_root: Path) -> Path:
    job_id = run_name or safe_job_id(route_id, action)
    job_dir = repo_path(job_root / job_id)
    if job_dir is None:
        raise ValueError('job root cannot be resolved')
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / 'run.log'
    job_path = job_dir / 'job.json'
    with log_path.open('ab') as log_handle:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    payload = {'contract': 'navqa_route_background_job_v1', 'job_id': job_id, 'route_id': route_id, 'action': action, 'run_name': run_name, 'pid': process.pid, 'created_at': datetime.now().isoformat(timespec='seconds'), 'cwd': str(REPO_ROOT), 'command': command, 'log_path': str(log_path), 'status_hint': 'started'}
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'job_id={job_id}')
    print(f'pid={process.pid}')
    print(f'job_dir={job_dir}')
    print(f'log={log_path}')
    return job_dir

def latest_job_dir(job_root: Path) -> Path | None:
    root = repo_path(job_root)
    if root is None or not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / 'job.json').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)

def resolve_job_dir(value: str, job_root: Path) -> Path | None:
    if value == 'latest':
        return latest_job_dir(job_root)
    direct = repo_path(value)
    if direct is not None and direct.is_dir() and (direct / 'job.json').exists():
        return direct
    rooted = repo_path(job_root / value)
    if rooted is not None and rooted.is_dir() and (rooted / 'job.json').exists():
        return rooted
    return None

def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def show_status(value: str, job_root: Path) -> None:
    job_dir = resolve_job_dir(value, job_root)
    if job_dir is None:
        raise SystemExit(f'background job not found: {value}')
    payload = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    pid = int(payload.get('pid') or 0)
    log_path = Path(str(payload.get('log_path') or job_dir / 'run.log'))
    running = pid_is_running(pid)
    summary_path = find_summary_path(log_path)
    print(f"job_id={payload.get('job_id')}")
    print(f"route_id={payload.get('route_id')}")
    print(f"action={payload.get('action')}")
    print(f'pid={pid}')
    print(f"status={('running' if running else 'not_running')}")
    print(f'log={log_path}')
    if summary_path:
        print(f'summary_json={summary_path}')
    tail = tail_text(log_path, line_count=20)
    if tail:
        print('')
        print('last_log_lines:')
        print(tail)

def find_summary_path(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    summary: str | None = None
    pattern = re.compile('summary_json=(.+)')
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = pattern.search(line)
        if match:
            summary = match.group(1).strip()
    return summary

def tail_text(path: Path, *, line_count: int) -> str:
    if not path.exists():
        return ''
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-line_count:])

def summarize_target(value: str, job_root: Path) -> None:
    summary_json = resolve_summary_json(value, job_root)
    if summary_json is None:
        raise SystemExit(f'summary not found: {value}')
    payload = json.loads(summary_json.read_text(encoding='utf-8-sig'))
    metadata = payload.get('run_metadata') if isinstance(payload, dict) else {}
    rows = payload.get('rows') if isinstance(payload, dict) else []
    if not isinstance(metadata, dict) or not isinstance(rows, list):
        raise SystemExit(f'not a mainline all-sequence summary: {summary_json}')
    total = new_bucket()
    by_type: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric = row_metric(row)
        add_bucket(total, metric)
        metric_by_type = metric.get('by_type') if isinstance(metric, dict) else {}
        if isinstance(metric_by_type, dict):
            for qtype, bucket in metric_by_type.items():
                target = by_type.setdefault(str(qtype), new_bucket())
                add_bucket(target, bucket if isinstance(bucket, dict) else {})
    print(f'source={summary_json}')
    print(f"run_name={payload.get('run_name')}")
    print(f"route_id={metadata.get('route_id')}")
    print(f'overall={format_bucket(total)}')
    if by_type:
        print('by_type=' + ', '.join((f'{qtype}:{format_bucket(bucket)}' for qtype, bucket in sorted(by_type.items()))))

def resolve_summary_json(value: str, job_root: Path) -> Path | None:
    job_dir = resolve_job_dir(value, job_root)
    if job_dir is not None:
        payload = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
        log_path = Path(str(payload.get('log_path') or job_dir / 'run.log'))
        found = find_summary_path(log_path)
        if found:
            found_path = Path(found)
            return found_path if found_path.exists() else repo_path(found_path)
    path = repo_path(value)
    if path is None:
        return None
    if path.is_dir():
        candidate = path / 'mainline_all_sequences_summary.json'
        return candidate if candidate.exists() else None
    return path if path.exists() else None

def row_metric(row: dict[str, Any]) -> dict[str, Any]:
    new = row.get('new')
    if not isinstance(new, dict):
        return {}
    if new.get('metric') == 'navqa_official_style_accuracy':
        return new
    return new

def new_bucket() -> dict[str, Any]:
    return {'correct': 0, 'total': 0, 'accuracy_percent': 0.0}

def add_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    target['correct'] += int(source.get('correct', 0) or 0)
    target['total'] += int(source.get('total', 0) or 0)
    total = int(target['total'] or 0)
    target['accuracy_percent'] = round(int(target['correct']) * 100.0 / total if total else 0.0, 2)

def format_bucket(bucket: dict[str, Any]) -> str:
    return f"{int(bucket.get('correct', 0) or 0)}/{int(bucket.get('total', 0) or 0)}={float(bucket.get('accuracy_percent', 0.0) or 0.0):.2f}%"

def run_foreground(commands: list[list[str]]) -> None:
    for command in commands:
        print(command_line(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

def route_or_exit(registry: RouteRegistry, route_id: str | None) -> Route:
    if not route_id:
        raise SystemExit('missing --route. Use --list to inspect available routes.')
    route = registry.routes.get(route_id)
    if route is None:
        raise SystemExit(f'unknown route: {route_id}. Use --list to inspect routes.')
    return route

def action_from_args(args: argparse.Namespace) -> str:
    if args.preflight:
        return 'preflight'
    if args.run:
        return 'run'
    if args.resume:
        return 'resume'
    if args.build_graphs:
        return 'build-graphs'
    if args.build_graphs_dry_run:
        return 'build-graphs-dry-run'
    return 'print'

def options_from_args(args: argparse.Namespace, *, mode: str) -> QACommandOptions:
    extra_args = tuple((str(value) for value in args.extra_args or ()))
    return QACommandOptions(mode=mode, run_name=args.run_name, seq_ids=flatten_seq_ids(args.seq_id), include_types=tuple(args.include_types) if args.include_types else None, exclude_types=tuple(args.exclude_types) if args.exclude_types else (), qa_workers=args.qa_workers, qa_scheduler=args.qa_scheduler, qa_retries=args.qa_retries, qa_batch_size=args.qa_batch_size, retrieval_limit=args.retrieval_limit, force_qa=bool(args.force_qa), force_question_ir=bool(args.force_question_ir), extra_args=extra_args)

def main() -> None:
    args = build_parser().parse_args()
    if args.status is not None:
        show_status(args.status, args.job_root)
        return
    if args.summarize:
        summarize_target(args.summarize, args.job_root)
        return
    registry = load_registry(args.routes_config)
    if args.list:
        print_routes(registry)
        return
    route = route_or_exit(registry, args.route)
    action = action_from_args(args)
    seq_ids = flatten_seq_ids(args.seq_id)
    if action in {'build-graphs', 'build-graphs-dry-run'}:
        commands = build_graph_commands(route, seq_ids=seq_ids)
        print('assembled build command(s):', flush=True)
        for command in commands:
            print(command_line(command), flush=True)
        if action == 'build-graphs-dry-run':
            return
        if args.background:
            wrapper_command = [sys.executable, 'scripts/datasets/navqa/qa/run_navqa_route.py', '--routes-config', str(args.routes_config), '--route', route.route_id, '--build-graphs']
            if args.run_name:
                wrapper_command.extend(['--run-name', args.run_name])
            for seq_id in seq_ids:
                wrapper_command.extend(['--seq-id', seq_id])
            start_background(wrapper_command, route_id=route.route_id, action=action, run_name=args.run_name, job_root=args.job_root)
            return
        run_foreground(commands)
        return
    options = options_from_args(args, mode=action)
    if action == 'resume' and (options.force_qa or options.force_question_ir):
        raise SystemExit('--resume cannot be combined with --force-qa/--force-question-ir')
    if args.background and action not in {'run', 'resume'}:
        raise SystemExit('--background is only supported with --run, --resume, or --build-graphs')
    background_run_name = args.run_name
    if args.background and background_run_name is None:
        background_run_name = safe_job_id(route.route_id, action)
        options = replace(options, run_name=background_run_name)
    command = build_qa_command(route, registry.defaults, options)
    print('assembled QA command:', flush=True)
    print(command_line(command), flush=True)
    if action == 'print':
        print('No execution requested. Add --preflight, --run, or --resume.', flush=True)
        return
    if args.background:
        start_background(command, route_id=route.route_id, action=action, run_name=background_run_name, job_root=args.job_root)
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)
if __name__ == '__main__':
    main()
