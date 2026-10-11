"""Offline v4 entry contract checks; never production prepare/run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_five_baselines_amem_native_v4 as w


class EntryContracts(unittest.TestCase):
    def test_frozen_qa_run_and_transport_audit_bodies_reused(self):
        for name in ('run','replay_leaf','firstpass_sources','audit_new_requests'):
            actual=getattr(w.controller,name);old=getattr(w.previous,name)
            self.assertEqual(actual.__code__.co_code,old.__code__.co_code,name)
            self.assertEqual(actual.__code__.co_consts,old.__code__.co_consts,name)

    def test_actual_individual_qualification_schema_is_accepted(self):
        value=w._qualification(w.DEFAULT_QUALIFICATION)
        self.assertEqual(value['request_hash'],w.FAILED_KEY)
        self.assertEqual(value['retry_limit'],0)

    def test_paid_replacement_and_scope_mutations_rejected(self):
        original=w.read(w.DEFAULT_QUALIFICATION)
        for changes in ({'retry_limit':1},{'scope':'whole_run_admitted'},{'raw_sha256':'a'*64},{'kind':'known_complete_invalid_json'}):
            with self.subTest(changes=changes),patch.object(w,'read',return_value={**original,**changes}):
                with self.assertRaises(ValueError):w._qualification(w.DEFAULT_QUALIFICATION)

    def test_explicit_per_conversation_original_owners(self):
        routes={str(c):str(w.NEW if c==2 else w.previous.SOURCE if c in (0,3,6,8) else w.SOURCE) for c in range(10)}
        with patch.object(w.controller,'_PROOF',{'conversation_sources':routes}):
            for ci in range(10):
                owner,leaf,ret=w.source_for({'conversation':ci,'query_id':f'q{ci}'})
                self.assertEqual(owner,Path(routes[str(ci)]))
                self.assertEqual(leaf,owner/'results/firstpass'/f'q{ci}')
                self.assertEqual(ret,owner/'results'/f'c{ci}'/'retrieval'/f'q{ci}.json')

    def test_proof_cannot_replace_complete_recomputed_evidence(self):
        expected={'schema':w.SCHEMA,'status':'passed','failure_qualification':{'path':'x','sha256':'a'},'conversations':{'2':{'mode':'rebuild'}}}
        with patch.object(w,'read',return_value=expected),patch.object(w,'check_binding'),patch.object(w,'_collect',return_value={**expected,'extra':'tampered'}):
            with self.assertRaises(ValueError):w.validate_proof(Path('fixture.json'))

    def test_only_exact_verified_launcher_parent_is_ignored(self):
        def proc(pid,parent,exe,args,created):
            return {'ProcessId':pid,'ParentProcessId':parent,'Name':'python.exe','ExecutablePath':str(exe),
                'CommandLine':f'"{exe}" {args}','CreatedUtc':created}
        args='-u tools/run_five_baselines_amem_native_v4.py run --method amem'
        parent=proc(100,1,w.ROOT/'runtime/baseline_env_v1/Scripts/python.exe',args,'2026-10-09T00:00:00+00:00')
        child=proc(101,100,sys._base_executable,args,'2026-10-09T00:00:01+00:00')
        self.assertEqual(w._active_controllers([parent,child],101),[])
        for change in ({'ParentProcessId':999},{'ExecutablePath':'C:/foreign/python.exe'},
                       {'CommandLine':parent['CommandLine']+' --foreign'},{'CreatedUtc':'2026-10-09T00:00:02+00:00'}):
            records=[{**parent,**change},child] if 'ParentProcessId' not in change else [parent,{**child,**change}]
            self.assertEqual(len(w._active_controllers(records,101)),1)
        foreign=proc(102,1,sys._base_executable,'-m tools.run_five_baselines_format_v2 run --method amem','2026-10-09T00:00:00+00:00')
        self.assertEqual(w._active_controllers([parent,child,foreign],101),[foreign])

    def test_real_venv_two_process_launcher_gate(self):
        runtime=(w.ROOT/'runtime').resolve()
        with tempfile.TemporaryDirectory(prefix='amem_v4_process_test_',dir=runtime) as directory:
            temp=Path(directory).resolve()
            self.assertEqual(temp.parent,runtime)
            script=temp/'run_five_baselines_amem_native_v4.py'
            script.write_text('import sys\nsys.path.insert(0,'+repr(str(w.ROOT))+')\n'
                'from tools.run_five_baselines_amem_native_v4 import assert_source_stopped\n'
                'assert_source_stopped()\nprint("verified_actual_venv_launcher")\n',encoding='utf-8')
            result=subprocess.run([str(w.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'),str(script),'run'],
                                  cwd=w.ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('verified_actual_venv_launcher',result.stdout)


if __name__=='__main__':unittest.main()
