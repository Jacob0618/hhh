"""R02 read-only health snapshot using the tested R01 artifact reader."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.snapshot_evidence_exchange_r01 import snapshot
from tools.score_evidence_exchange_r01 import now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    script = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {$_.CommandLine -like '*evidence_ordering*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    proc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True,
        text=True, encoding='utf-8', check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = {'created_utc': now(), 'runs': [snapshot(p) for p in args.run],
              'processes': json.loads(proc.stdout or '[]'),
              'health_verdict': 'Compare real products and context-specific attempts across snapshots; no PID-only verdict.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'runs': [{
        'run': r['run'], 'artifacts': {k: v['count'] for k, v in r.get('artifacts', {}).items()},
        'ledgers': [{k: l[k] for k in ('bytes', 'counts')} for l in r.get('ledgers', [])],
        'read_errors': r.get('read_errors', [])} for r in result['runs']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
