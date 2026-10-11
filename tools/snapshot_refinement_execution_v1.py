"""Read-only snapshots for the September 30 refinement runs; no model I/O."""
from pathlib import Path
import argparse
import collections
import datetime
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read

RUNS = ('fullconv_llm_segmentation_v1', 'fullconv_llm_segmentation_recovery_v1',
        'refined_event_grouping_v1', 'mem0_source_local_context_v1',
        'mem0_jev_refined_v1', 'mem0_llm_fullconv_v1')

def read(path):
    return json.loads(shared_read(path))

def snapshot():
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': {}}
    for name in RUNS:
        folder = ROOT / 'runs' / name
        if not folder.exists():
            continue
        item = {'states': {}, 'ledgers': [], 'completed_artifacts': {}}
        for rel in ('status.json', 'run_status.json', 'controller_status.json'):
            p = folder / rel
            if p.exists():
                item['states'][rel] = read(p)
        for p in sorted(folder.glob('conv_*/*/write_status.json')):
            s = read(p)
            item['states'][str(p.relative_to(folder))] = {k: v for k, v in s.items()
                if k not in ('bank', 'facts', 'groups')}
        for p in sorted(folder.rglob('api_calls.jsonl')):
            raw = shared_read(p)
            rows, incomplete = [], 0
            for line in raw.splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: incomplete += 1
            physical = [r for r in rows if not r.get('cache_hit')]
            item['ledgers'].append({'path': str(p.relative_to(folder)),
                'bytes': p.stat().st_size, 'rows': len(rows), 'incomplete_rows': incomplete,
                'physical': len(physical), 'cache': len(rows)-len(physical),
                'statuses': dict(collections.Counter(str(r.get('status')) for r in physical)),
                'stages': dict(collections.Counter(str(r.get('stage')) for r in physical)),
                'last_utc': rows[-1].get('timestamp_utc') if rows else None})
        for p in sorted(folder.glob('conv_*/*/bank.json')):
            b = read(p)
            item['completed_artifacts'][str(p.relative_to(folder))] = {
                'complete': b.get('complete'), 'facts': len(b.get('facts', []))}
        out['runs'][name] = item
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    compact = {'utc': result['utc'], 'runs': {}}
    for name, data in result['runs'].items():
        compact['runs'][name] = {'states': {
            rel: {k: v for k, v in s.items() if k in ('status', 'pid', 'updated_utc',
                'completed_groups', 'expected_groups', 'error', 'conversations', 'outcomes')}
            for rel, s in data['states'].items()}, 'ledgers': data['ledgers'],
            'artifacts': data['completed_artifacts']}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
