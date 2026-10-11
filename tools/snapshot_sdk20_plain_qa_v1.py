"""Read-only progress evidence for fixed-selection Reader/Judge experiments."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.snapshot_mem0_plain_qa import artifact_counts, read
from research.snapshot_sdk20_judge_retest_v1 import ledger_summary, process_rows

STAGES = ('reader', 'judge', 'compat', 'qa')


def snapshot(run):
    run = Path(run).resolve()
    manifest = read(run / 'run_manifest.json')
    start = datetime.now(timezone.utc)
    jobs = manifest['jobs']
    controller = {}
    pids = set()
    for name in ('execution_intent.json', 'run_status.json', 'run_outcome.json'):
        path = run / name
        if path.exists():
            row = read(path)
            controller[name] = {k: v for k, v in row.items() if k != 'results'}
            if type(row.get('pid')) is int:
                pids.add(row['pid'])
    conversations = []
    total = {stage: Counter() for stage in STAGES}
    for ci in sorted({job['conversation'] for job in jobs}):
        subset = [j for j in jobs if j['conversation'] == ci]
        folder = run / f'conv_{ci}'
        states = {}
        for name in ('execution_intent.json', 'qa_status.json', 'worker_status.json'):
            path = folder / name
            if path.exists():
                states[name] = read(path)
                if type(states[name].get('pid')) is int:
                    pids.add(states[name]['pid'])
        arms = {}
        for arm in sorted({j['method'] for j in subset}):
            stages = {stage: artifact_counts(folder / stage / arm) for stage in STAGES}
            for stage in STAGES:
                total[stage].update(stages[stage]['counts'])
            arms[arm] = {'expected': sum(j['method'] == arm for j in subset), 'stages': stages}
        ledger = ledger_summary(folder / 'api/api_calls.jsonl')
        conversations.append({'conversation': ci, 'expected': len(subset), 'states': states,
                              'arms': arms, 'api': ledger})
    processes = process_rows()
    wanted = str(run).casefold().replace('/', '\\')
    pids.update(p['ProcessId'] for p in processes
                if wanted in (p.get('CommandLine') or '').casefold().replace('/', '\\'))
    for _ in range(3):
        pids.update(p['ParentProcessId'] for p in processes if p['ProcessId'] in pids)
    return {'schema': 'sdk20-plain-qa-health-evidence-v1', 'run_dir': str(run),
            'protocol_sha256': manifest['protocol_sha256'], 'phase_kind': manifest['phase_kind'],
            'phase_id': manifest['phase_id'], 'captured_utc': start.isoformat(),
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'expected': len(jobs),
            'stage_counts': {s: dict(v) for s, v in total.items()}, 'controller': controller,
            'processes': [p for p in processes if p['ProcessId'] in pids], 'conversations': conversations,
            'note': 'Evidence only; compare per-conversation artifacts and request context/hash/stage over time. HTTP200 or a live process alone is not a health verdict.'}


def difference(old, current):
    if any(old[k] != current[k] for k in ('run_dir', 'protocol_sha256', 'phase_id')):
        raise ValueError('Snapshots must belong to the same run and phase')
    seconds = (datetime.fromisoformat(current['captured_utc']) - datetime.fromisoformat(old['captured_utc'])).total_seconds()
    prior = {x['conversation']: x for x in old['conversations']}
    rows = []
    for row in current['conversations']:
        before = prior[row['conversation']]
        delta = {key: row['api'][key] - before['api'][key]
                 for key in ('bytes', 'rows', 'physical_attempts', 'http_200', 'physical_errors', 'cache_events')}
        added = {arm: {s: info['stages'][s]['output_files'] - before['arms'][arm]['stages'][s]['output_files']
                       for s in STAGES} for arm, info in row['arms'].items()}
        rows.append({'conversation': row['conversation'], 'artifacts_added': added, 'api_delta': delta,
                     'new_request_hashes': len(set(row['api']['request_activity']) - set(before['api']['request_activity']))})
    return {'seconds_between_starts': seconds, 'conversations': rows, 'automatic_health_verdict': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Use a new audit file outside the experiment namespace')
    value = snapshot(args.run_dir)
    if args.previous:
        value['difference'] = difference(read(args.previous), value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: value[k] for k in ('captured_utc', 'expected', 'stage_counts', 'controller', 'processes')}
                     | {'output': str(args.output), 'difference': value.get('difference')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
