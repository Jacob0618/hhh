"""Read-only dated Mem0 writer evidence; no automatic health verdict or API use."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, json, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_experiment_health import read_json, shared_read


def snapshot(run):
    rows = []
    for path in sorted(run.glob('conv_*/*/write_status.json')):
        state = read_json(path)
        ledger = path.parent / 'native/api_calls.jsonl'
        data = shared_read(ledger) if ledger.exists() else b''
        events, malformed = [], []
        for index, line in enumerate(data.splitlines()):
            try:
                events.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                malformed.append(index + 1)
        journals = [read_json(p) for p in (path.parent / 'native/group_audit').rglob('*.json')]
        journals.sort(key=lambda j: j['sequence'])
        latest = journals[-1] if journals else {}
        rows.append({'conversation': state['conversation'], 'arm': state['arm'],
            'state': state, 'api_bytes': len(data), 'api_rows': len(events),
            'api_status': dict(Counter(str(e.get('status', 'cache')) for e in events)),
            'last_api_events': [{k: e.get(k) for k in ('timestamp_utc', 'stage', 'request_hash', 'attempt', 'status', 'cache_hit')} for e in events[-12:]],
            'malformed_line_numbers': malformed,
            'journals_by_status': dict(Counter(j['status'] for j in journals)),
            'latest_group': {k: latest.get(k) for k in ('group_id', 'sequence', 'status', 'started_at', 'finished_at', 'error_type', 'persistence_check')},
            'same_group_extraction_calls': len(latest.get('llm_calls', [])),
            'current_results_count': len(latest.get('results', []))})
    # Include Python ancestors, including Windows ProcessPool spawn identities.
    # These experiment invocations contain paths/flags, never credential values.
    command = "Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    processes = json.loads(result.stdout or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    pids = {r['state']['pid'] for r in rows}
    pids.update(p['ProcessId'] for p in processes if 'regroup_sdk20.py' in (p.get('CommandLine') or ''))
    for _ in range(5):
        pids.update(p['ParentProcessId'] for p in processes if p['ProcessId'] in pids)
    return {'captured_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
        'arms': rows, 'processes': [p for p in processes if p['ProcessId'] in pids],
        'note': 'Evidence only; compare two snapshots and historical progress before declaring health.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new audit output; no snapshot replacement')
    value = snapshot(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    compact = [{k: r[k] for k in ('conversation', 'arm', 'api_status', 'latest_group', 'malformed_line_numbers')} |
        {'state': {k: r['state'].get(k) for k in ('pid', 'status', 'completed_groups', 'facts', 'updated_utc')}} for r in value['arms']]
    print(json.dumps({'captured_utc': value['captured_utc'], 'arms': compact,
                      'processes': value['processes'], 'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
