"""Read-only snapshots; compare two snapshots to assess real stage progress."""
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


def read(path):
    return json.loads(shared_read(path).decode('utf-8-sig'))


def snapshot(run):
    run = Path(run).resolve()
    states, artifacts, ledgers, errors = {}, {}, [], []
    if not run.exists():
        return {'run': str(run), 'exists': False}
    for p in sorted(run.glob('*.json')):
        if p.name.endswith(('_status.json', '_outcome.json')):
            try:
                states[p.name] = {'mtime': p.stat().st_mtime, 'state': read(p)}
            except Exception as exc:
                errors.append({'path': str(p), 'error_type': type(exc).__name__})
    folders = [run / 'responses', run / 'pilot_responses']
    folders += [p for p in run.glob('conv_*/*/*') if p.is_dir() and p.parent.name in ('reader', 'judge', 'qa')]
    for folder in folders:
        if not folder.exists():
            continue
        counts, last, perconv = Counter(), None, {}
        for p in folder.glob('c*_q*.json'):
            if p.name.endswith('.intent.json'):
                continue
            try:
                row = read(p)
                counts[str(row.get('status', 'status_missing'))] += 1
                ci = p.stem.split('_')[0]
                item = perconv.setdefault(ci, {'count': 0, 'latest_mtime': 0})
                item['count'] += 1
                item['latest_mtime'] = max(item['latest_mtime'], p.stat().st_mtime)
                if last is None or p.stat().st_mtime > last['mtime']:
                    last = {'query_id': p.stem, 'mtime': p.stat().st_mtime}
            except Exception as exc:
                errors.append({'path': str(p), 'error_type': type(exc).__name__})
        artifacts[str(folder.relative_to(run))] = {'count': sum(counts.values()), 'statuses': dict(counts),
            'per_conversation': perconv, 'last': last}
    for p in sorted(run.rglob('api_calls.jsonl')):
        raw = shared_read(p)
        complete = raw[:raw.rfind(b'\n') + 1]
        counts, contexts, usage = Counter(), {}, Counter()
        for line in complete.splitlines():
            row = json.loads(line)
            cached = row.get('cache_hit') is True
            counts['cache' if cached else 'physical'] += 1
            if not cached:
                counts['http_' + str(row.get('status'))] += 1
                if row.get('usage_unknown'):
                    counts['unknown_usage_attempts'] += 1
                for k, v in (row.get('usage') or {}).items():
                    if isinstance(v, (float, int)) and not isinstance(v, bool):
                        usage[k] += v
            ctx = row.get('context') or {}
            key = '|'.join(str(ctx.get(k)) for k in ('conversation', 'method')) + '|' + str(row.get('stage'))
            contexts[key] = {k: row.get(k) for k in ('timestamp_utc', 'request_hash', 'status', 'cache_hit', 'attempt')}
        ledgers.append({'path': str(p), 'bytes': len(raw), 'complete_bytes': len(complete),
            'mtime': p.stat().st_mtime, 'counts': dict(counts), 'physical_reported_usage_sum': dict(usage), 'contexts_last': contexts})
    return {'run': str(run), 'exists': True, 'states': states, 'artifacts': artifacts,
        'ledgers': ledgers, 'read_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    script = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {$_.CommandLine -like '*evidence_exchange*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    proc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True,
        text=True, encoding='utf-8', check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'runs': [snapshot(p) for p in args.run],
        'processes': json.loads(proc.stdout or '[]'), 'health_verdict': 'Compare with earlier snapshot and stage workload; not inferred from process or HTTP200 alone.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'runs': [{
        'run': r['run'], 'artifacts': {k: v['count'] for k, v in r.get('artifacts', {}).items()},
        'ledgers': [{k: l[k] for k in ('path', 'bytes', 'counts')} for l in r.get('ledgers', [])],
        'read_errors': r.get('read_errors', [])} for r in result['runs']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
