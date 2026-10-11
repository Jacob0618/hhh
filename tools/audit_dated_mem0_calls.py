"""Offline API accounting for dated Mem0 namespaces, including failed attempts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
from tools.summarize_mem0_sparse1540 import measure_usage


def audit(run):
    all_rows, sources, arms = [], [], []
    for folder in sorted(Path(run).glob('conv_*/*')):
        ledger = folder / 'native/api_calls.jsonl'
        if not ledger.exists():
            continue
        raw = shared_read(ledger)
        rows = [json.loads(line) for line in raw.splitlines()]
        sources.append({'path': str(ledger.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'bytes': len(raw), 'rows': len(rows)})
        failures = []
        for i, row in enumerate(rows):
            if row.get('cache_hit') or row.get('status') == 200:
                continue
            signature = (row.get('context'), row.get('stage'), row.get('request_hash'))
            subsequent = [r for r in rows[i + 1:] if
                          (r.get('context'), r.get('stage'), r.get('request_hash')) == signature
                          and r.get('status') == 200 and not r.get('cache_hit')]
            failures.append({'timestamp_utc': row.get('timestamp_utc'), 'context': row.get('context'),
                             'stage': row.get('stage'), 'request_hash': row.get('request_hash'),
                             'status': row.get('status'), 'attempt': row.get('attempt'),
                             'later_physical_success_same_context_stage_hash': bool(subsequent)})
        arms.append({'folder': str(folder.resolve()), 'usage': measure_usage(rows),
                     'stages': dict(Counter(r.get('stage') for r in rows)), 'failures': failures})
        all_rows.extend(rows)
    return {'run_dir': str(Path(run).resolve()), 'ledgers': sources, 'arms': arms,
            'current_namespace_usage': measure_usage(all_rows),
            'stages': {stage: measure_usage([r for r in all_rows if r.get('stage') == stage])
                       for stage in sorted({r.get('stage') for r in all_rows})},
            'note': 'Each physical attempt counted once; imported cache usage excluded from new costs. '
                    'Inherited generation/embedding costs belong to source namespaces, not free cold work. '
                    'Missing ledgers are not proof of zero model activity; use failure audit and intents. '
                    'Unknown transport or embedding usage remains unknown. Configured costs are not invoices. '
                    'Paced provider latency can include gate waiting; it is not isolated HTTP duration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'runs': [audit(run) for run in args.run_dir]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output.resolve()), 'runs': [
        {'run': r['run_dir'], 'usage': r['current_namespace_usage']} for r in result['runs']]}))


if __name__ == '__main__':
    main()
