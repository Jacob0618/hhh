"""Read-only observer reusing module-stack health with the new six stages."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import snapshot_module_stack_v1 as prior

scope={**vars(prior),'SCHEMA':'r03-r05-refinement-health-snapshot-v1',
    'PHASES':('precision','complement','sets','multi_compare','closure','rank')}
for name,fn in tuple(scope.items()):
    if isinstance(fn,FunctionType) and fn.__globals__ is vars(prior):
        scope[name]=FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)


def processes():
    command=r"""[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*r03_r05*' } |
Select-Object @{n='pid';e={$_.ProcessId}},@{n='parent_pid';e={$_.ParentProcessId}},@{n='creation_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}},@{n='command_line';e={$_.CommandLine}} | ConvertTo-Json -Compress
"""
    try:
        result=prior.subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,
            encoding='utf-8',check=True,timeout=20,creationflags=getattr(prior.subprocess,'CREATE_NO_WINDOW',0))
        raw=result.stdout.strip();rows=json.loads(raw) if raw else []
        if isinstance(rows,dict):rows=[rows]
        for row in rows:
            row['command_line']=prior._redact_command(row.get('command_line',''))
            row['role']='snapshot_observer' if row.get('pid')==os.getpid() else 'refinement_python_process'
        return {'rows':rows,'error':None,'presence_is_not_health':True}
    except Exception as exc:
        return {'rows':[],'error':type(exc).__name__,'presence_is_not_health':True}


scope['processes']=processes
snapshot=scope['snapshot'];compare=scope['compare']
if __name__=='__main__':scope['main']()
