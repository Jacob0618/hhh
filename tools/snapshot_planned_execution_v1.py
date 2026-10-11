"""Read-only snapshots for the 2026-10-03 fixed1540 experiment namespaces."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
from tools.run_sdk20_modules import save_new

RUNS = ('planned_hybrid_retrieval_v1', 'planned_hybrid_qa_v1',
        'boundary_variants_v1_20261003')
QID = re.compile(r'^c\d+_q\d+$')


def snapshot(run):
    run = Path(run)
    if not run.exists(): return {'run': str(run), 'exists': False}
    state_paths = [p for p in run.rglob('*.json') if
        p.name.endswith(('_status.json', '_outcome.json')) or p.name in ('execution_intent.json', 'qa_execution_intent.json')]
    states, errors = [], []
    for p in state_paths:
        try:
            data = json.loads(shared_read(p))
            states.append({'path': str(p), 'mtime': p.stat().st_mtime, 'state': data})
        except Exception as exc: errors.append({'path': str(p), 'error_type': type(exc).__name__})
    count = Counter(); newest = {}
    for p in run.rglob('*.json'):
        if not QID.fullmatch(p.stem): continue
        rel = p.relative_to(run); parts = rel.parts
        if any(t in parts for t in ('api_cache', 'request_intents')): continue
        key = '/'.join(parts[:-1]); count[key] += 1
        modified = p.stat().st_mtime
        if modified > newest.get(key, {}).get('mtime', -1):
            newest[key] = {'query_id': p.stem, 'mtime': modified}
    ledgers = []
    for p in run.rglob('api_calls.jsonl'):
        raw = shared_read(p); end = raw.rfind(b'\n') + 1
        counts, last, contexts = Counter(), None, {}
        for line in raw[:end].splitlines():
            try: r = json.loads(line)
            except Exception as exc:
                errors.append({'path': str(p), 'error_type': type(exc).__name__}); continue
            cached = r.get('cache_hit') is True
            counts['cache' if cached else 'http_attempt'] += 1
            if not cached: counts['status_' + str(r.get('status'))] += 1
            if r.get('stage'): counts['stage_' + r['stage']] += 1
            last = {k: r.get(k) for k in ('timestamp', 'timestamp_utc', 'request_hash', 'stage', 'status', 'cache_hit', 'context', 'elapsed_seconds')}
            ctx = r.get('context') or {}
            contexts[str((ctx.get('conversation'), ctx.get('method', ctx.get('arm')), ctx.get('query_id'), r.get('stage')))] = last
        ledgers.append({'path': str(p), 'bytes': len(raw), 'mtime': p.stat().st_mtime,
            'counts': dict(counts), 'last': last, 'contexts': contexts})
    return {'run': str(run), 'exists': True, 'states': states, 'artifacts': dict(count),
        'newest_artifacts': newest, 'ledgers': ledgers, 'read_errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='append', type=Path)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh health snapshot required')
    runs = a.run or [ROOT / 'runs' / r for r in RUNS]
    value = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0, 'runs': [snapshot(r) for r in runs]}
    save_new(a.output, value)
    print(json.dumps({'snapshot': str(a.output), 'runs': [{'run': r['run'], 'exists': r['exists'],
        'artifacts': r.get('artifacts', {}), 'ledger_bytes': sum(x['bytes'] for x in r.get('ledgers', [])),
        'read_errors': len(r.get('read_errors', []))} for r in value['runs']]}))


if __name__ == '__main__': main()
