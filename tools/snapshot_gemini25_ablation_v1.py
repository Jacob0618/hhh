"""Read-only current Gemini 2.5 Writer/source/QA health evidence.

Only --output is written. Hot reads share READ|WRITE|DELETE on Windows. Native
databases, raw caches, scientific payloads and historical runs are not opened.
Use repeatable --run-dir, a fresh --output, and optional --compare SNAPSHOT.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read

SCHEMA = 'gemini25-ablation-health-snapshot-v1'
PHASES = ('r03', 'support', 'multi_compare', 'closure', 'rank')
ARMS = ('baseline', 'r03_original', 'original_multi_source_closure')
STATE_LIMIT = 2 * 1024 * 1024
OUTPUT_LIMIT = 1024 * 1024
SCRIPTS = ('run_gemini25_ablation_writer_v1.py', 'prepare_gemini25_ablation_v1.py',
           'score_gemini25_ablation_v1.py', 'run_gemini25_ablation_qa_v1.py',
           'summarize_gemini25_ablation_v1.py', 'snapshot_gemini25_ablation_v1.py')


def utc(stamp=None):
    return (datetime.fromtimestamp(stamp, timezone.utc) if stamp is not None
            else datetime.now(timezone.utc)).isoformat()


def redact(value):
    text = str(value or '')
    text = re.sub(r'(?i)(\bBearer\s+)[^\s\"\']+', r'\1[REDACTED]', text)
    text = re.sub(r'''(?ix)((?:--?|\b)(?:[a-z0-9]+[_-])*(?:api[_-]?key|token|password|secret|authorization)\s*(?:=|:|\s)\s*)("[^"]*"|'[^']*'|[^\s,;]+)''',
                  r'\1[REDACTED]', text)
    text = re.sub(r'(https?://)[^\s/@:]+:[^\s/@]+@', r'\1[REDACTED]@', text)
    text = re.sub(r'(?i)([?&](?:key|api_key|token|access_token)=)[^&\s\"\']+', r'\1[REDACTED]', text)
    text = re.sub(r'AIza[0-9A-Za-z_-]{20,}|\bsk-[0-9A-Za-z_-]{16,}', '[REDACTED]', text)
    return text


def scalar(value):
    if value is None or type(value) in (int, float, bool):
        return value
    if isinstance(value, str):
        return redact(value)[:240]
    return None


def counts_only(value, depth=0):
    if type(value) is int:
        return value
    if not isinstance(value, dict) or depth > 3:
        return None
    return {redact(k)[:80]: counts_only(v, depth + 1) for k, v in value.items()
            if type(v) is int or isinstance(v, dict)}


def issue(errors, path, exc):
    errors.append({'path': str(path), 'error_type': type(exc).__name__})


def read(path, *, limit=STATE_LIMIT):
    if path.stat().st_size > limit:
        raise ValueError('Bounded status reader: file too large')
    raw = shared_read(path)
    if len(raw) > limit:
        raise ValueError('Bounded status reader: file grew too large')
    return json.loads(raw.decode('utf-8-sig'))


def state(path, errors):
    if not path.is_file():
        return {'exists': False}
    try:
        value = read(path)
        if not isinstance(value, dict):
            raise ValueError('Status must be an object')
        fields = ('schema', 'status', 'phase', 'phase_kind', 'terminal', 'pid',
                  'conversation', 'arm', 'method', 'started_utc', 'updated_utc',
                  'finished_utc', 'created_utc', 'completed_groups', 'expected_groups',
                  'facts', 'active_session', 'last_group', 'completed', 'expected',
                  'expected_requests', 'terminal_requests', 'total', 'conversations_finished')
        out = {key: scalar(value[key]) for key in fields if key in value}
        if isinstance(value.get('counts'), dict):
            out['counts'] = counts_only(value['counts'])
        if isinstance(value.get('conversations'), dict):
            out['conversations'] = {str(ci): {key: scalar(row[key]) for key in
                ('terminal', 'expected', 'last_activity_utc') if key in row}
                for ci, row in value['conversations'].items() if str(ci).isdigit() and isinstance(row, dict)}
        for key in ('local_errors', 'outcomes', 'requests', 'units', 'jobs'):
            if isinstance(value.get(key), list):
                out[key + '_count'] = len(value[key])
        if isinstance(value.get('outcomes'), list):
            rows = [row for row in value['outcomes'] if isinstance(row, dict)]
            out['outcome_status_counts'] = dict(Counter(scalar(row.get('status', 'unspecified')) for row in rows))
            out['outcome_local_error_count'] = sum(len(row.get('local_errors', [])) for row in rows
                                                   if isinstance(row.get('local_errors', []), list))
        return {'exists': True, 'bytes': path.stat().st_size,
                'mtime_utc': utc(path.stat().st_mtime), 'state': out}
    except Exception as exc:
        issue(errors, path, exc)
        return {'exists': True, 'read_error': type(exc).__name__}


def local_path(path, root):
    return not path.is_symlink() and path.resolve().is_relative_to(root)


def files(folder, pattern, root):
    return folder.glob(pattern) if local_path(folder, root) else ()


def metadata(paths, errors, conversation=None, *, root=None):
    total, size, intents, latest, perconv = 0, 0, 0, None, Counter()
    for path in paths:
        try:
            if path.is_symlink() or (root is not None and not local_path(path, root)) or not path.is_file():
                continue
            stat = path.stat()
            if path.name.endswith('.intent.json'):
                intents += 1
                continue
            total += 1
            size += stat.st_size
            ci = conversation
            if ci is None:
                match = re.match(r'^c(\d+)_', path.name)
                ci = match.group(1) if match else 'unknown'
            perconv[str(ci)] += 1
            if latest is None or stat.st_mtime > latest['mtime_unix']:
                latest = {'filename': path.name, 'mtime_unix': stat.st_mtime,
                          'mtime_utc': utc(stat.st_mtime), 'bytes': stat.st_size}
        except OSError as exc:
            issue(errors, path, exc)
    return {'file_count': total, 'intent_file_count': intents, 'bytes': size,
            'latest': latest, 'per_conversation': dict(perconv), 'content_read': False}


def ledger(path, errors, previous=None):
    """Increment only from the previous complete-line boundary when possible."""
    stat = path.stat()
    identity = [stat.st_dev, stat.st_ino]
    reuse = (previous and previous.get('file_identity') == identity
             and stat.st_size >= previous.get('bytes', 0)
             and previous.get('complete_bytes', 0) <= stat.st_size)
    offset = previous['complete_bytes'] if reuse else 0
    raw = shared_read(path, offset=offset)
    after = path.stat()
    if [after.st_dev, after.st_ino] != identity or after.st_size < offset + len(raw):
        raise OSError('Ledger replaced or shrank during shared read')
    complete = raw[:raw.rfind(b'\n') + 1]
    counts = Counter(previous.get('counts', {})) if reuse else Counter()
    by_stage = deepcopy_counts(previous.get('by_stage', {})) if reuse else {}
    latest = previous.get('latest') if reuse else None
    perconv = dict(previous.get('conversation_latest', {})) if reuse else {}
    malformed = 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Non-object ledger row')
            cached = value.get('cache_hit') is True
            status = scalar(value.get('status'))
            if not isinstance(status, (str, int, type(None))):
                status = None
            count = Counter(rows=1)
            count['cache' if cached else 'physical'] += 1
            if not cached:
                count['status_' + str(status)] += 1
                count['successes' if type(status) is int and 200 <= status < 300 else 'errors'] += 1
            if value.get('usage_unknown') is True:
                count['unknown_usage_rows'] += 1
            counts.update(count)
            stage = str(scalar(value.get('stage')))
            by_stage.setdefault(stage, Counter()).update(count)
            context = value.get('context') if isinstance(value.get('context'), dict) else {}
            ci = context.get('conversation')
            ci = str(ci) if type(ci) is int and 0 <= ci < 10 else 'unknown'
            event = {key: scalar(value.get(key)) for key in
                     ('timestamp_utc', 'provider', 'model', 'operation', 'stage', 'status', 'cache_hit', 'attempt')}
            event.update({key: scalar(context.get(key)) for key in ('phase', 'method', 'component')})
            event['conversation'] = ci
            key = str(value.get('request_hash', ''))
            event['request_hash_prefix'] = key[:16] if re.fullmatch('[0-9a-f]{64}', key) else None
            if latest is None or str(event.get('timestamp_utc') or '') >= str(latest.get('timestamp_utc') or ''):
                latest = event
            if ci not in perconv or str(event.get('timestamp_utc') or '') >= str(perconv[ci].get('timestamp_utc') or ''):
                perconv[ci] = event
        except (ValueError, TypeError):
            malformed += 1
    if malformed:
        errors.append({'path': str(path), 'error_type': 'MalformedCompleteLedgerRows', 'count': malformed})
    return {'path': str(path), 'file_identity': identity, 'bytes': offset + len(raw),
            'complete_bytes': offset + len(complete), 'in_flight_tail_bytes': len(raw) - len(complete),
            'bytes_read_this_snapshot': len(raw), 'incremental': bool(reuse),
            'reset_or_shrank': bool(previous and not reuse), 'mtime_utc': utc(stat.st_mtime),
            'counts': dict(counts), 'by_stage': {key: dict(value) for key, value in by_stage.items()},
            'latest': latest, 'conversation_latest': perconv,
            'malformed_complete_rows': malformed + (previous.get('malformed_complete_rows', 0) if reuse else 0)}


def deepcopy_counts(value):
    return {key: Counter(counts) for key, counts in value.items()}


def snapshot(run, previous=None):
    run = Path(run).resolve()
    if not run.name.startswith('gemini25_ablation_'):
        raise ValueError('Only explicit new gemini25_ablation_* runs are supported')
    if not run.is_dir():
        return {'run_dir': str(run), 'exists': False}
    errors, states, artifacts, ledger_paths = [], {}, {}, []
    root_states = ('controller_status.json', 'run_status.json', 'retrieval_outcome.json',
                   'execution_intent.json', 'prepare_intent.json')
    for name in root_states:
        if (run / name).is_file() and local_path(run / name, run):
            states[name] = state(run / name, errors)
    artifacts['run_markers'] = metadata((run / name for name in (
        'run_manifest.json', 'run_outcome.json', 'development_review.json', 'all_bank_review.json',
        'selected_inputs.json')), errors, root=run)
    for ci in range(10):
        conv = run / f'conv_{ci}'
        if not conv.is_dir() or not local_path(conv, run):
            continue
        for name in ('retrieval_status.json', 'retrieval_intent.json', 'post_retrieval.json',
                     'session/write_status.json', 'session/write_outcome.json', 'session/write_intent.json'):
            if (conv / name).is_file() and local_path(conv / name, run):
                states[f'conv_{ci}/{name}'] = state(conv / name, errors)
        artifacts[f'conv_{ci}/writer_journals'] = metadata(
            files(conv / 'session/native/session_audit', '*/*.json', run), errors, ci, root=run)
        artifacts[f'conv_{ci}/writer_bank'] = metadata([conv / 'session/bank.json'], errors, ci, root=run)
        artifacts[f'conv_{ci}/retrieval'] = metadata(files(conv / 'retrieval', '*.json', run), errors, ci, root=run)
        for stage in ('reader', 'judge', 'qa', 'compat'):
            for arm in ARMS:
                folder = conv / stage / arm
                if folder.is_dir() and local_path(folder, run):
                    artifacts[f'conv_{ci}/{stage}/{arm}'] = metadata(files(folder, '*.json', run), errors, ci, root=run)
        ledger_paths.extend(conv / leaf / 'api_calls.jsonl'
                            for leaf in ('session/native', 'retrieval_api', 'api'))
    for phase in PHASES:
        folder = run / 'phases' / phase
        if not folder.is_dir() or not local_path(folder, run):
            continue
        for kind in ('units', 'requests'):
            artifacts[f'phases/{phase}/{kind}'] = metadata(files(folder / kind, '*.json', run), errors, root=run)
        for kind in ('firstpass', 'recovery_v1'):
            producer = folder / kind
            for name in ('status.json', 'outcome.json', 'execution_intent.json'):
                if (producer / name).is_file() and local_path(producer / name, run):
                    states[f'phases/{phase}/{kind}/{name}'] = state(producer / name, errors)
            artifacts[f'phases/{phase}/{kind}/responses'] = metadata(files(producer / 'responses', '*.json', run), errors, root=run)
            ledger_paths.append(producer / 'api_calls.jsonl')
    for arm in ARMS:
        folder = run / 'selections' / arm
        if folder.is_dir() and local_path(folder, run):
            artifacts[f'selections/{arm}'] = metadata(files(folder, '*.json', run), errors, root=run)
    old_ledgers = {row['path']: row for row in (previous or {}).get('api_ledgers', [])}
    ledgers = []
    for path in ledger_paths:
        if path.is_file() and local_path(path, run):
            try:
                ledgers.append(ledger(path, errors, old_ledgers.get(str(path))))
            except Exception as exc:
                issue(errors, path, exc)
    totals = sum((Counter(row['counts']) for row in ledgers), Counter())
    return {'run_dir': str(run), 'exists': True, 'states': states, 'artifacts': artifacts,
            'api_ledgers': ledgers, 'api_totals': {'bytes': sum(row['bytes'] for row in ledgers),
                'bytes_read_this_snapshot': sum(row['bytes_read_this_snapshot'] for row in ledgers),
                'counts': dict(totals)}, 'read_error_count': len(errors), 'read_errors': errors[:30],
            'artifact_semantics': 'Metadata counts only; content validity and semantic success are not inferred.'}


def select_processes(rows, runs):
    by_pid = {row['pid']: row for row in rows if type(row.get('pid')) is int}
    status_pids = {value['state']['pid'] for run in runs for value in run.get('states', {}).values()
                   if type(value.get('state', {}).get('pid')) is int}
    selected = {pid for pid, row in by_pid.items() if pid in status_pids or
                any(name in str(row.get('command_line') or '') for name in SCRIPTS)}
    descendants = set(selected)
    for _ in range(8):
        descendants.update(pid for pid, row in by_pid.items() if row.get('parent_pid') in descendants)
    parents = {by_pid[pid].get('parent_pid') for pid in descendants}
    output = []
    for pid in sorted((descendants | parents) & set(by_pid)):
        row = by_pid[pid]
        command = str(row.get('command_line') or '')
        output.append({'pid': pid, 'parent_pid': row.get('parent_pid'),
            'creation_utc': row.get('creation_utc'), 'command_line': redact(command),
            'role': 'snapshot_observer' if pid == os.getpid() else 'related_process' if pid in descendants else 'parent_process',
            'reported_in_status': pid in status_pids, 'current_command_matches_ablation_script': any(name in command for name in SCRIPTS),
            'identity_note': 'Current process identity; a status PID alone may have been reused.'})
    return output


def processes(runs):
    script = r"""[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();
Get-CimInstance Win32_Process | Select-Object @{n='pid';e={$_.ProcessId}},@{n='parent_pid';e={$_.ParentProcessId}},@{n='creation_utc';e={if ($_.CreationDate) {$_.CreationDate.ToUniversalTime().ToString('o')}}},@{n='command_line';e={$_.CommandLine}} | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
            capture_output=True, text=True, encoding='utf-8', check=True, timeout=20,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        rows = json.loads(result.stdout or '[]')
        return {'rows': select_processes(rows if isinstance(rows, list) else [rows], runs),
                'error_type': None, 'presence_is_not_health': True}
    except Exception as exc:
        return {'rows': [], 'error_type': type(exc).__name__, 'presence_is_not_health': True}


def compare(old, new):
    before = {run['run_dir']: run for run in old.get('runs', [])}
    rows = []
    for run in new['runs']:
        prior = before.get(run['run_dir'], {})
        a, b = prior.get('api_totals', {}), run.get('api_totals', {})
        ca, cb = a.get('counts', {}), b.get('counts', {})
        artifacts = {key: {'file_count_delta': run.get('artifacts', {}).get(key, {}).get('file_count', 0)
                                           - prior.get('artifacts', {}).get(key, {}).get('file_count', 0),
                           'bytes_delta': run.get('artifacts', {}).get(key, {}).get('bytes', 0)
                                        - prior.get('artifacts', {}).get(key, {}).get('bytes', 0)}
                     for key in sorted(set(prior.get('artifacts', {})) | set(run.get('artifacts', {})))}
        state_changes = {}
        for key in sorted(set(prior.get('states', {})) | set(run.get('states', {}))):
            first = prior.get('states', {}).get(key, {}).get('state', {})
            second = run.get('states', {}).get(key, {}).get('state', {})
            changes = {name: {'before': first.get(name), 'after': second.get(name)} for name in
                       ('status', 'terminal', 'pid', 'completed_groups', 'facts', 'active_session',
                        'completed', 'terminal_requests', 'conversations_finished', 'counts')
                       if first.get(name) != second.get(name)}
            if changes:
                state_changes[key] = changes
        rows.append({'run_dir': run['run_dir'], 'prior_run_missing': not bool(prior),
            'api_bytes_delta': b.get('bytes', 0) - a.get('bytes', 0),
            'api_count_deltas': {key: cb.get(key, 0) - ca.get(key, 0) for key in sorted(set(ca) | set(cb))},
            'artifact_deltas': artifacts, 'state_changes': state_changes,
            'ledger_reset_or_shrank': any(row.get('reset_or_shrank') for row in run.get('api_ledgers', []))})
    old_processes = {row['pid']: row for row in old.get('processes', {}).get('rows', [])}
    replacements = [row['pid'] for row in new.get('processes', {}).get('rows', [])
                    if row['pid'] in old_processes and row.get('creation_utc') != old_processes[row['pid']].get('creation_utc')]
    try:
        elapsed = (datetime.fromisoformat(new['captured_utc']) - datetime.fromisoformat(old['captured_utc'])).total_seconds()
    except (KeyError, ValueError, TypeError):
        elapsed = None
    return {'runs': rows, 'elapsed_seconds': elapsed, 'pid_creation_changed': replacements,
            'not_a_health_verdict': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new snapshot output')
    runs = [run.resolve() for run in args.run_dir]
    output = args.output.resolve()
    if any(output.is_relative_to(run) for run in runs) or (ROOT / 'runs').resolve() in output.parents:
        raise ValueError('Snapshot output must stay outside scientific run directories')
    previous = read(args.compare, limit=OUTPUT_LIMIT) if args.compare else None
    if previous and previous.get('schema') != SCHEMA:
        raise ValueError('Comparison requires a Gemini 2.5 health snapshot')
    old_runs = {run['run_dir']: run for run in (previous or {}).get('runs', [])}
    data = {'schema': SCHEMA, 'captured_utc': utc(), 'model_calls': 0, 'credential_reads': 0,
            'runs': [snapshot(run, old_runs.get(str(run))) for run in runs], 'health_verdict': None}
    data['processes'] = processes(data['runs'])
    try:
        disk = shutil.disk_usage('D:/')
        data['disk_D'] = {'free_bytes': disk.free, 'total_bytes': disk.total, 'used_bytes': disk.used}
    except OSError as exc:
        data['disk_D'] = {'error_type': type(exc).__name__}
    if previous:
        data['comparison'] = compare(previous, data)
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
    if len(raw) > OUTPUT_LIMIT:
        raise ValueError('Snapshot exceeds bounded output size')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(raw + b'\n')
    print(json.dumps({'output': str(output), 'runs': len(runs), 'bytes': len(raw) + 1,
                      'read_errors': sum(run.get('read_error_count', 0) for run in data['runs']),
                      'model_calls': 0, 'credential_reads': 0}))


if __name__ == '__main__':
    main()
