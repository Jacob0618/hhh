"""Read-only evidence for the authorized Gemini 3.8 rebuild and QA runs.

No model clients or native databases are opened. Compare fresh snapshots; process
existence or HTTP200 alone is deliberately not reported as a health verdict.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read


def read_json(path):
    return json.loads(shared_read(path).decode('utf-8-sig'))


def snapshot(run):
    run = Path(run).resolve()
    if not run.exists():
        return {'run': str(run), 'exists': False}
    states, journals, ledgers, errors = [], [], [], []
    for p in sorted(run.rglob('*.json')):
        parts = p.relative_to(run).parts
        if any(x in parts for x in ('api_cache', 'request_intents', 'cache_lineage')):
            continue
        is_state = p.name.endswith(('_status.json', '_outcome.json')) or p.name in ('parent_state.json', 'worker_state.json', 'status.json')
        is_journal = any(x in parts for x in ('session_audit', 'group_audit'))
        if not (is_state or is_journal):
            continue
        try:
            obj = read_json(p)
            row = {'path': str(p), 'mtime': p.stat().st_mtime}
            if is_state:
                row['state'] = obj
                states.append(row)
            else:
                row.update({k: obj.get(k) for k in ('user_id', 'group_id', 'session_id', 'sequence', 'status', 'started_at', 'finished_at', 'error_type')})
                row['saved_fact_count'] = len(obj.get('results', []))
                row['llm_calls'] = len(obj.get('llm_calls', []))
                journals.append(row)
        except Exception as exc:
            errors.append({'path': str(p), 'error_type': type(exc).__name__})
    artifacts = Counter()
    for p in run.rglob('c*_q*.json'):
        parts = p.relative_to(run).parts
        if any(x in parts for x in ('api_cache', 'request_intents')):
            continue
        artifacts['/'.join(parts[:-1])] += 1
    for p in sorted(run.rglob('api_calls.jsonl')):
        try:
            data = shared_read(p)
            complete = data[:data.rfind(b'\n') + 1]
            counts, by_context, last = Counter(), {}, None
            for line in complete.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cached = obj.get('cache_hit') is True
                counts['cache' if cached else 'physical'] += 1
                if not cached:
                    counts['http_' + str(obj.get('status'))] += 1
                ctx = obj.get('context') or {}
                key = str((ctx.get('conversation'), ctx.get('method', ctx.get('arm')), ctx.get('query_id', ctx.get('group_id')), obj.get('stage')))
                last = {k: obj.get(k) for k in ('timestamp_utc', 'request_hash', 'stage', 'status', 'cache_hit', 'attempt')}
                by_context[key] = last
            ledgers.append({'path': str(p), 'bytes': len(data), 'complete_bytes': len(complete), 'mtime': p.stat().st_mtime,
                            'counts': dict(counts), 'last': last, 'contexts_last': by_context})
        except Exception as exc:
            errors.append({'path': str(p), 'error_type': type(exc).__name__})
    return {'run': str(run), 'exists': True, 'states': states, 'journals': journals,
            'artifacts': dict(artifacts), 'ledgers': ledgers, 'read_errors': errors}


def processes(runs):
    script = "Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    proc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True,
                          text=True, encoding='utf-8', check=True,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = json.loads(proc.stdout or '[]')
    if isinstance(rows, dict):
        rows = [rows]
    ids = {r['ProcessId'] for r in rows if 'gemini38' in (r.get('CommandLine') or '')}
    for run in runs:
        for r in run.get('states', []):
            obj = r['state']
            for k in ('pid', 'controller_pid', 'worker_pid'):
                if type(obj.get(k)) is int:
                    ids.add(obj[k])
    for _ in range(6):
        ids.update(r['ProcessId'] for r in rows if r['ParentProcessId'] in ids)
        ids.update(r['ParentProcessId'] for r in rows if r['ProcessId'] in ids)
    return [r for r in rows if r['ProcessId'] in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new snapshot path')
    runs = [snapshot(r) for r in a.run]
    data = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0,
            'runs': runs, 'processes': processes(runs), 'health_verdict': None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(a.output.resolve()), 'captured_utc': data['captured_utc'], 'runs': [
        {'run': r['run'], 'exists': r['exists'], 'state_files': len(r.get('states', [])),
         'journals': dict(Counter(x['status'] for x in r.get('journals', []))),
         'saved_facts_in_success_journals': sum(x['saved_fact_count'] for x in r.get('journals', []) if x['status'] == 'success'),
         'artifacts': r.get('artifacts', {}), 'ledger_bytes': sum(x['bytes'] for x in r.get('ledgers', [])),
         'api_counts': dict(sum((Counter(x['counts']) for x in r.get('ledgers', [])), Counter())),
         'read_errors': len(r.get('read_errors', []))} for r in runs]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
