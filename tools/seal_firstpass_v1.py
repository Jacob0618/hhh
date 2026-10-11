"""Seal the terminal 1540-question first pass without mutating run artifacts."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from check_experiment_health import read_json, shared_read, snapshot
from jev_mem.data import load_data, queries, SOURCE_HASH


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    output = ROOT / 'research/main_batch_firstpass_seal_v1'
    output.mkdir(exist_ok=False)
    runs = [ROOT / 'runs/session_top20_v1', ROOT / 'runs/independent_top20_v1']
    state = snapshot(ROOT / 'runs/orchestration_eight_lane_v2', runs)
    if state['active_jobs'] or state['pending_jobs'] or state['problems']:
        raise RuntimeError('Main batch not terminal and verified')
    # Reject surviving QA processes even if queues became terminal prematurely.
    from check_experiment_health import inventory
    if inventory():
        raise RuntimeError('An old supervisor or QA process still exists')
    write_new(output / 'terminal_state.json', state)
    dataset = load_data()
    frozen = read_json(ROOT / 'research/protocol_snapshot_reader_v2/manifest.json')['source_sha256']
    core = ['jev_mem/providers.py', 'jev_mem/selection.py', 'jev_mem/evaluation.py',
            'jev_mem/data.py', 'tools/run_session_experiments.py', 'tools/run_independent_experiments.py',
            'jev_mem/independent.py', 'jev_mem/mem0_adapter.py']
    code = {}
    for path in core:
        code[path] = sha(shared_read(ROOT / path))
        if code[path] != frozen[path.replace('/', '\\')]:
            raise RuntimeError('Frozen source mismatch: ' + path)
    report = {'schema': 'jev-mem-technical-recovery-seal-v1',
              'status': 'sealed_after_main_terminal', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_dataset_sha256': SOURCE_HASH, 'code_sha256': code, 'runs': [],
              'policy': {'per_hash_max_new_http_attempts': 1, 'max_consumer_workers': 8,
                         'frozen_dependency_completion_allowed': True},
              'per_hash_max_new_http_attempts': 1, 'max_consumer_workers': 8,
              'frozen_dependency_completion_allowed': True,
              'inventory_semantics': 'Original run files retained in place; every byte-hash bound. No original mutation authorized.'}
    failures = []
    for architecture, run in zip(['mem0', 'independent'], runs):
        queue = read_json(run / 'queue_status.json')
        methods = queue['methods']
        counts = Counter()
        for ci in range(10):
            folder = run / f'conv_{ci}'
            raw = shared_read(folder / 'qa_summary.json')
            summary = json.loads(raw)
            with (output / f'{architecture}_conv{ci}_qa_summary.json').open('xb') as handle:
                handle.write(raw)
            expected = {q['id'] for q in queries(dataset[ci], ci)}
            if expected != set(summary['requested_query_ids']):
                raise RuntimeError('Summary requested IDs differ from dataset')
            missing = set()
            for method in methods:
                present = {p.stem for p in (folder / 'qa' / method).glob('*.json')}
                if not present <= expected:
                    raise RuntimeError('Unexpected QA ID')
                counts['requested'] += len(expected)
                counts['success'] += len(present)
                missing.update((method, qid) for qid in expected - present)
            errors = {(e['method'], e['query_id']) for e in summary['errors']}
            if errors != missing or len(errors) != len(summary['errors']):
                raise RuntimeError('Summary errors do not exactly cover missing evaluations')
            for error in summary['errors']:
                failures.append({'architecture': architecture, 'run_dir': str(run),
                                 'conversation': ci, **error})
        files = sorted(p for p in run.rglob('*') if p.is_file())
        def record(path):
            raw = shared_read(path)
            return {'path': path.relative_to(run).as_posix(), 'sha256': sha(raw), 'bytes': len(raw)}
        inventory_path = output / (architecture + '_inventory.jsonl')
        with inventory_path.open('x', encoding='utf-8', newline='\n') as handle:
            with ThreadPoolExecutor(max_workers=4) as workers:
                for row in workers.map(record, files):
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        report['runs'].append({'architecture': architecture, 'run_dir': str(run), 'methods': methods,
                               'conversations': list(range(10)), 'requested_questions': 1540,
                               'inventory_path': inventory_path.name,
                               'inventory_sha256': sha(inventory_path.read_bytes()),
                               'file_count': len(files), 'requested_evaluations': counts['requested'],
                               'successful_evaluations': counts['success'],
                               'failed_evaluations': counts['requested'] - counts['success']})
        print(json.dumps({'architecture': architecture, **report['runs'][-1]}, ensure_ascii=False), flush=True)
    write_new(output / 'failures.json', failures)
    report['failures_path'] = 'failures.json'
    report['failures_sha256'] = sha((output / 'failures.json').read_bytes())
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(output / 'seal.json', report)
    print(json.dumps({'seal': str(output / 'seal.json'), 'failures': len(failures), 'model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
