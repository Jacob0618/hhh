"""Read-only stage-aware health evidence for refinement retrieval and fixed400 QA."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
from tools.snapshot_mem0_plain_qa import artifact_counts
from research.snapshot_sdk20_judge_retest_v1 import ledger_summary, process_rows

STAGES = ('selection', 'reader', 'judge', 'compat', 'qa')


def read(path):
    return json.loads(shared_read(Path(path)))


def snapshot(run):
    run = Path(run).resolve()
    m = read(run / 'run_manifest.json')
    if m.get('schema') not in ('regroup-refinement-plain-qa400-v1', 'regroup-refinement-plain-qa400-v2'):
        raise ValueError('Refinement QA manifest required')
    if m['phase_kind'] == 'firstpass':
        qs = {int(ci): ids for ci, ids in m['query_ids'].items()}
        if sum(len(ids) for ids in qs.values()) != 400:
            raise ValueError('The firstpass must retain all400 slots')
    else:
        qs = {ci: [j['query_id'] for j in m['jobs'] if j['conversation'] == ci] for ci in range(10)}
        qs = {ci: ids for ci, ids in qs.items() if ids}
    start = datetime.now(timezone.utc).isoformat()
    states, pids = {}, set()
    for name in ('retrieval_execution_intent.json', 'retrieval_status.json', 'retrieval_outcome.json',
                 'qa_execution_intent.json', 'qa_status.json', 'run_outcome.json'):
        p = run / name
        if p.exists():
            value = read(p)
            states[name] = {k: v for k, v in value.items() if k != 'results'}
            if type(value.get('pid')) is int:
                pids.add(value['pid'])
    rows, totals = [], {stage: Counter() for stage in STAGES}
    for ci, ids in sorted(qs.items()):
        folder = run / f'conv_{ci}'
        local = {}
        for name in ('retrieval_execution_intent.json', 'qa_progress.json'):
            p = folder / name
            if p.exists():
                local[name] = read(p)
                if type(local[name].get('pid')) is int:
                    pids.add(local[name]['pid'])
        stages = {stage: artifact_counts(folder / stage / m['arm']) for stage in STAGES}
        for stage in STAGES:
            totals[stage].update(stages[stage]['counts'])
        rows.append({'conversation': ci, 'expected': len(ids), 'states': local, 'stages': stages,
                     'api': ledger_summary(folder / 'api/api_calls.jsonl')})
    all_processes = process_rows()
    wanted = str(run).casefold().replace('/', '\\')
    pids.update(p['ProcessId'] for p in all_processes
                if wanted in (p.get('CommandLine') or '').casefold().replace('/', '\\'))
    for _ in range(3):
        pids.update(p['ProcessId'] for p in all_processes if p['ParentProcessId'] in pids)
    return {'schema': 'refinement-qa-health-v1', 'captured_utc': start,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
            'protocol_sha256': m['protocol_sha256'], 'phase_id': m['phase_id'], 'phase_kind': m['phase_kind'],
            'expected': sum(len(ids) for ids in qs.values()), 'arm': m['arm'],
            'controller': states, 'stage_counts': {k: dict(v) for k, v in totals.items()},
            'conversations': rows, 'processes': [p for p in all_processes if p['ProcessId'] in pids],
            'note': 'Evidence only: compare actual artifacts and per-context request deltas; live PID or HTTP200 alone is not health.'}


def difference(old, new):
    if any(old[k] != new[k] for k in ('run_dir', 'protocol_sha256', 'phase_id')):
        raise ValueError('Same run and phase required')
    before = {r['conversation']: r for r in old['conversations']}
    rows = []
    for row in new['conversations']:
        prior = before[row['conversation']]
        rows.append({'conversation': row['conversation'],
                     'artifacts_added': {s: row['stages'][s]['output_files'] - prior['stages'][s]['output_files'] for s in STAGES},
                     'api_delta': {k: row['api'][k] - prior['api'][k] for k in
                                   ('bytes', 'rows', 'physical_attempts', 'http_200', 'physical_errors', 'cache_events')}})
    return {'seconds_between_starts': (datetime.fromisoformat(new['captured_utc']) - datetime.fromisoformat(old['captured_utc'])).total_seconds(),
            'conversations': rows, 'automatic_health_verdict': None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--previous', type=Path)
    args = ap.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Fresh audit file outside the experiment required')
    value = snapshot(args.run_dir)
    if args.previous:
        value['difference'] = difference(read(args.previous), value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: value[k] for k in ('captured_utc', 'expected', 'phase_kind', 'stage_counts', 'processes')}
                     | {'output': str(args.output), 'difference': value.get('difference')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
