"""R08 read-only snapshots, including its separate derived embedding products."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.snapshot_evidence_exchange_r01 import snapshot, read
from tools.score_evidence_exchange_r01 import now


def snapshot_discovery(run):
    result = snapshot(run)
    folder = Path(run).resolve() / 'embedding_responses'
    if folder.exists():
        counts, perconv, last = Counter(), {}, None
        for p in folder.glob('c*_b*.json'):
            try:
                row = read(p); stamp = p.stat().st_mtime
                counts[row.get('status', 'missing_status')] += 1
                ci = str(row['conversation'])
                item = perconv.setdefault(ci, {'count': 0, 'latest_mtime': 0})
                item['count'] += 1; item['latest_mtime'] = max(stamp, item['latest_mtime'])
                if last is None or stamp > last['mtime']:
                    last = {'query_id': row.get('request_id', p.stem), 'mtime': stamp}
            except Exception as exc:
                result['read_errors'].append({'path': str(p), 'error_type': type(exc).__name__})
        result['artifacts']['embedding_responses'] = {'count': sum(counts.values()),
            'statuses': dict(counts), 'per_conversation': perconv, 'last': last}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    script = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {$_.CommandLine -like '*facet_discovery*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    proc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
        capture_output=True, text=True, encoding='utf-8', check=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = {'created_utc': now(), 'runs': [snapshot_discovery(p) for p in args.run],
        'processes': json.loads(proc.stdout or '[]'),
        'health_verdict': 'Compare stage products, real contexts and attempts across snapshots; no PID-only verdict.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'runs': [{
        'run': r['run'], 'artifacts': {k: v['count'] for k, v in r.get('artifacts', {}).items()},
        'ledgers': [{k: v[k] for k in ('bytes', 'counts')} for v in r.get('ledgers', [])],
        'read_errors': r.get('read_errors', [])} for r in result['runs']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
