"""Read-only, two-snapshot health evidence for the pilot400 shared-log driver."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_experiment_health import shared_read, read_json, parse_rows
from process_control_windows import WinProcess, split_command_line


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(row):
    methods = row.get('methods', {})
    return {'groups_completed': row.get('groups_completed', 0),
            'facts_completed': row.get('facts_completed', 0),
            'facts_failed': row.get('facts_failed', 0),
            'qa_completed': sum(v.get('completed', 0) for v in methods.values()),
            'qa_failed': sum(v.get('failed', 0) for v in methods.values()),
            'bank': row.get('bank'), 'status': row.get('status')}


def snapshot(run, stage):
    state = read_json(run / (stage + '_status.json'))
    problems, worker = [], {'pid': state['pid']}
    if state.get('stage') != stage:
        problems.append('stage_mismatch')
    if state['status'] == 'running':
        try:
            handle = WinProcess(state['pid'])
            try:
                worker.update(alive=handle.is_running(), creation_time_100ns=handle.creation_time_100ns)
            finally:
                handle.close()
            cmd = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(state['pid'])}' | Select-Object CommandLine | ConvertTo-Json -Compress"
            raw = subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd], check=True,
                                 capture_output=True, text=True, encoding='utf-8').stdout
            args = split_command_line(json.loads(raw)['CommandLine'])
            same_script = any(Path(x).name == 'run_pilot400_experiments.py' for x in args)
            same_dir = ('--run-dir' in args and Path(args[args.index('--run-dir') + 1]).resolve() == run.resolve())
            worker['command_verified'] = same_script and same_dir and stage in args
            # PID reuse must not validate a different launch of the same command.
            created = worker['creation_time_100ns'] / 10_000_000 - 11644473600
            launched = datetime.fromisoformat(state['started_utc']).timestamp()
            worker['creation_matches_launch'] = -1 <= launched - created <= 120
            if not all(worker.get(k) for k in ('alive', 'command_verified', 'creation_matches_launch')):
                problems.append('worker_identity_not_verified')
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            problems.append('worker_identity_' + type(exc).__name__)
    path = run / 'api_calls.jsonl'
    size = path.stat().st_size if path.exists() else 0
    tail = parse_rows(shared_read(path, tail=100_000)) if size else []
    return {'captured_utc': now(), 'run_dir': str(run), 'stage': stage, 'state': state,
            'worker': worker, 'problems': problems, 'api_bytes': size,
            'last_api_utc': tail[-1].get('timestamp_utc') if tail else None,
            'progress': {ci: progress(row) for ci, row in state['conversations'].items()}}


def compare(a, b, rows, *, historical=None):
    verdicts = []
    duration = (datetime.fromisoformat(b['captured_utc']) - datetime.fromisoformat(a['captured_utc'])).total_seconds()
    for ci, row in b['progress'].items():
        old = a['progress'].get(ci, {})
        relevant = [r for r in rows if str(r.get('context', {}).get('conversation')) == ci]
        success = [r for r in relevant if not r.get('cache_hit') and r.get('status') == 200]
        failed = [r for r in relevant if not r.get('cache_hit') and r.get('status') not in (None, 200)]
        changed = any(row.get(k) != old.get(k) for k in row)
        if row['status'] == 'pending':
            verdict = 'pending_not_started'
        elif row['status'] in {'complete', 'failed'}:
            verdict = 'terminal_requires_review'
        else:
            verdict = 'progress_observed' if changed or success else 'no_successful_progress_investigate'
        # A long unchanged QA/fact/group counter merits inspection even with HTTP200.
        if historical and row['status'] == 'running':
            age = (datetime.fromisoformat(b['captured_utc']) - datetime.fromisoformat(historical['captured_utc'])).total_seconds()
            same_process = all(historical.get('worker', {}).get(k) == b['worker'].get(k)
                               for k in ('pid', 'creation_time_100ns'))
            if age >= 600 and same_process and historical.get('progress', {}).get(ci) == row:
                verdict = 'long_unchanged_stage_counters_investigate_even_with_http'
        verdicts.append({'conversation': int(ci), 'previous': old, 'current': row,
                         'real_http200': len(success), 'real_errors': len(failed),
                         'cache_hits': sum(bool(r.get('cache_hit')) for r in relevant),
                         'unique_success_hashes': len({r.get('request_hash') for r in success}),
                         'successful_stages': dict(Counter(r.get('stage') for r in success)),
                         'successful_query_ids': sorted({str(r.get('context', {}).get('query_id')) for r in success}),
                         'verdict': verdict})
    problems = list(b['problems'])
    if b['api_bytes'] < a['api_bytes']:
        problems.append('api_log_shrank')
    return {'first_utc': a['captured_utc'], 'second_utc': b['captured_utc'], 'seconds': duration,
            'model_calls_made': 0, 'problems': problems, 'stage_status': b['state']['status'],
            'bytes_added': b['api_bytes'] - a['api_bytes'], 'conversations': verdicts,
            'requires_investigation': bool(problems) or any('investigate' in v['verdict'] for v in verdicts),
            'limitations': 'HTTP progress is liveness only. Review terminal failures and long unchanged stage counters; segmentation may precede the first completed writer group.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['write', 'tag', 'qa'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('interval must be 1..60 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    first = snapshot(args.run_dir.resolve(), args.stage)
    with (args.output_dir / 'first.json').open('x', encoding='utf-8') as f:
        json.dump(first, f, ensure_ascii=False, indent=2)
    print(json.dumps({'first_snapshot': first['captured_utc'], 'status': first['state']['status']}), flush=True)
    time.sleep(args.interval)
    second = snapshot(args.run_dir.resolve(), args.stage)
    raw = shared_read(args.run_dir / 'api_calls.jsonl', offset=first['api_bytes']) if second['api_bytes'] else b''
    report = compare(first, second, parse_rows(raw[:second['api_bytes'] - first['api_bytes']]),
                     historical=read_json(args.previous) if args.previous else None)
    for name, obj in [('second.json', second), ('report.json', report)]:
        with (args.output_dir / name).open('x', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
