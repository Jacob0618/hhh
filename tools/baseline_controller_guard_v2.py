"""Read-only one-controller guard, including a verified Windows venv launcher."""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ('run_five_baselines_amem_native_v4', 'run_five_baselines_reviewed_chain_v2')


def active_controllers(processes, current_pid=None):
    current_pid = os.getpid() if current_pid is None else current_pid
    by_id = {int(p['ProcessId']): p for p in processes}
    def active(p):
        command = p.get('CommandLine') or ''
        return (re.fullmatch(r'python(?:w|[0-9.]*)?\.exe', p.get('Name', ''), re.I) is not None
                and re.search(r'run_five_baselines[\w]*(?:\.py)?(?:["\s]|$)', command, re.I)
                and re.search(r'(?:^|["\s])run(?:["\s]|$)', command, re.I))
    current = by_id.get(current_pid)
    if current is None:
        raise ValueError('Current interpreter missing from process snapshot')
    allowed = {current_pid}
    parent = by_id.get(int(current.get('ParentProcessId', -1)))
    if parent and active(current) and active(parent):
        def tail(p):
            found = re.fullmatch(r'\s*(?:"[^"]+"|\S+)(.*)', p.get('CommandLine') or '')
            return found.group(1) if found else None
        normalize = lambda p: os.path.normcase(os.path.abspath(p))
        command = tail(current)
        if (command is not None and command == tail(parent)
                and any(re.search(name + r'(?:\.py)?(?:["\s]|$)', command) for name in LAUNCHERS)
                and normalize(parent.get('ExecutablePath') or '') == normalize(ROOT/'runtime/baseline_env_v1/Scripts/python.exe')
                and normalize(current.get('ExecutablePath') or '') == normalize(sys._base_executable)
                and datetime.fromisoformat(parent['CreatedUtc']) <= datetime.fromisoformat(current['CreatedUtc'])):
            allowed.add(int(parent['ProcessId']))
    return [p for p in processes if active(p) and int(p['ProcessId']) not in allowed]


def assert_stopped():
    if os.name != 'nt':
        raise ValueError('Production process validation requires Windows')
    command = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | "
               "Select-Object ProcessId,ParentProcessId,Name,CommandLine,ExecutablePath,"
               "@{Name='CreatedUtc';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}}) | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True, text=True)
    if result.returncode or not result.stdout.strip():
        raise ValueError('Process enumeration failed')
    values = json.loads(result.stdout)
    if active_controllers(values if isinstance(values, list) else [values]):
        raise ValueError('Another baseline controller is active')
