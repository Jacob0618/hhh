"""Bounded packaging/launcher fixtures only; never derive a production archive."""
from contextlib import ExitStack
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_amem_metric_closure_v2 as a
from tools import run_five_baselines_reviewed_chain_v4 as chain
from tools.baselines_common_v1 import read,sha,save_new
from tests import test_amem_retrieval_lifecycle_v1 as lifecycle_factory

ACTUAL=Path(__file__).resolve().parents[1]


class DerivationTests(unittest.TestCase):
    def setUp(self):
        self.f=lifecycle_factory.LifecycleTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.oldout=a.old.archive(self.f.review,self.f.toolreview)
        self.oldplan=Path(self.oldout['plan']);self.plan=read(self.oldplan)
        self.source={'plan':a.binding(self.oldplan),**{k:self.plan[k] for k in ('archive','archive_manifest','archive_replay')}}
        self.original=read(self.source['archive_manifest']['path'])
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for name,value in {'ROOT':self.f.root,'SOURCE_PLAN':self.oldplan,
            'DEFAULT_PLAN':self.f.final/'cleanup_plan_retrieval_v2.json',
            'OUTPUT':self.f.final/'results/continuation_archive_v2',
            'SOURCE_SHA256':{k:v['sha256'] for k,v in self.source.items()}}.items():
            self.stack.enter_context(patch.object(a,name,value))
        # Frozen lifecycle factory stubs expensive replay, independently covered
        # by the true 1540 ZIP diagnostic and subsequent actual package review.
        self.stack.enter_context(patch.object(a.old,'replay_all',return_value=read(self.source['archive_replay']['path'])))
        self.stack.enter_context(patch.object(a,'prepare_metric_runtime',return_value={'runtime_modules_sha256':{}}))
        for name in (*a.typed_entry.code_hashes(),*a.EXTRA,a.METRIC):
            p=self.f.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ACTUAL/name).read_bytes())
        self.evidence=self.f.root/'author_fixture_evidence.json';save_new(self.evidence,{'status':'passed','scope':'synthetic_packaging_only'})
        self.review=self.f.root/'new_tool_review.json'
        save_new(self.review,{'schema':a.REVIEW_SCHEMA,'status':'passed','scope':'archive_derivation_only_no_scientific_change',
            'code_sha256':a.code_hashes(),'source_archive':self.source,'evidence_sha256':{str(self.evidence):sha(self.evidence)}})

    def derive(self):
        with patch.object(a.old,'file_inventory',side_effect=AssertionError('No ancestor inventory allowed')):
            return a.derive(self.review)

    def test_real_small_zip_addition_and_original_package_immutability(self):
        result=self.derive();newplan=read(a.DEFAULT_PLAN)
        self.assertEqual(result['status'],'derived_archive_verified')
        with a.old.Resolver(newplan['archive']['path'],newplan['archive_manifest']['path']) as view:
            self.assertEqual(view.manifest['records'][:len(self.original['records'])],self.original['records'])
            self.assertEqual(view.bytes(self.f.root/a.METRIC),(ACTUAL/a.METRIC).read_bytes())
            additions=view.manifest['records'][len(self.original['records']):]
            self.assertTrue(additions)
            self.assertTrue(all(not any(Path(r['path']).is_relative_to(root) for root in self.f.roots.values()) for r in additions))
        for b in self.source.values():
            self.assertEqual(sha(b['path']),b['sha256']);self.assertEqual(newplan['keep_sha256'][b['path']],b['sha256'])
        self.assertEqual(newplan['code_sha256'],self.plan['code_sha256'])
        self.assertEqual(newplan['work_inventories'],self.plan['work_inventories'])
        self.assertEqual(a.verify_derivation(),newplan)
        with self.assertRaises(FileExistsError):self.derive()

    def test_changed_original_packaging_or_run_addition_rejected_before_output(self):
        value=read(self.review);value['evidence_sha256'][str(self.f.roots['v4']/'fresh_annotation.json')]='a'*64
        self.review.write_text(json.dumps(value),'utf8')
        with self.assertRaises(FileNotFoundError):self.derive()
        self.assertFalse(a.OUTPUT.exists())
        value['evidence_sha256'].pop(str(self.f.roots['v4']/'fresh_annotation.json'))
        value['source_archive']['archive']['sha256']='0'*64
        self.review.write_text(json.dumps(value),'utf8')
        with self.assertRaisesRegex(ValueError,'authority differs'):self.derive()
        self.assertFalse(a.OUTPUT.exists())

    @unittest.skipUnless(os.name=='nt','Real Windows PowerShell fixture')
    def test_new_plan_real_powershell_then_old_plan_cannot_complete(self):
        self.derive();plan=read(a.DEFAULT_PLAN)
        cleanup=self.f.root/'cleanup_review_new.json'
        save_new(cleanup,{'status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'archive_sha256':plan['archive']['sha256']})
        stub=self.f.root/'fixture_verifier.py'
        stub.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ACTUAL))+')\n'
            'from tools import archive_amem_retrieval_continuation_v1 as a\n'
            'a.ROOT=Path('+repr(str(self.f.root))+');a.ROOTS='+repr({k:str(v) for k,v in self.f.roots.items()})+'\na.ROOTS={k:Path(v) for k,v in a.ROOTS.items()}\n'
            'a.FINAL=Path('+repr(str(self.f.final))+');a.WORK_OWNER=Path('+repr(str(self.f.owner))+')\n'
            'a.engine.ROOT=a.ROOT;a.engine.ROOTS=a.ROOTS;a.engine.FINAL=a.FINAL\n'
            "a.engine.stopped=lambda:None;a.engine.replay_all=lambda r:{'status':'passed','fixture_only':True}\na.main()\n",encoding='utf8')
        quote=lambda x:"'"+str(x).replace("'","''")+"'"
        script=(ACTUAL/a.EXTRA[1]).read_text('utf8')
        script=script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path","(Resolve-Path -LiteralPath "+quote(self.f.root)+').Path')
        script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'",'$python = '+quote(sys.executable))
        script=script.replace("$helper = Join-Path $workspace 'tools\\archive_amem_retrieval_continuation_v1.py'",'$helper = '+quote(stub))
        script=script.replace("$ErrorActionPreference = 'Stop'","function Get-CimInstance { @() }\n$ErrorActionPreference = 'Stop'")
        fixture=self.f.root/'fixture_cleanup.ps1';fixture.write_text(script,'utf8')
        self.assertTrue(all((root/'work').resolve().is_relative_to(self.f.root) for root in self.f.roots.values()))
        result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(fixture),'-CleanupReview',str(cleanup)],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(all(not (p/'work').exists() for p in self.f.roots.values()))
        self.assertEqual(a.old.verify_completion(a.DEFAULT_PLAN)['status'],'passed')
        with self.assertRaises(ValueError):a.old.verify_completion(self.oldplan)
        for b in self.source.values():self.assertEqual(sha(b['path']),b['sha256'])


class LauncherTests(unittest.TestCase):
    def test_prepare_gate_before_real_base_prepare_and_run_single_gate(self):
        base=chain.frozen.frozen.base
        with patch('sys.argv',['entry','prepare','--method','simplemem','--review','fake']),patch.object(chain,'verify_predecessors',side_effect=ValueError('missing completion')),patch.object(base,'prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'missing completion'):chain.main()
            prepare.assert_not_called()
        with patch('sys.argv',['entry','run','--method','simplemem']),patch.object(chain,'verify_predecessors') as gate,patch.object(chain,'verify_code_review'),patch.object(base,'manifest',return_value={'review':{'path':'fake'}}),patch.object(base,'run',return_value={}) as run:
            chain.main();self.assertEqual(gate.call_count,1);run.assert_called_once()

    @unittest.skipUnless(os.name=='nt','Actual venv parent identity')
    def test_real_venv_launcher_parent_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix='metric_guard_only_') as directory:
            p=Path(directory)/'run_five_baselines_reviewed_chain_v4.py'
            p.write_text('import sys\nsys.path.insert(0,'+repr(str(ACTUAL))+')\nfrom tools import run_five_baselines_reviewed_chain_v4 as c\nc.assert_stopped()\nprint("GATE_ONLY_PASSED",flush=True)\n','utf8')
            r=subprocess.run([str(ACTUAL/'runtime/baseline_env_v1/Scripts/python.exe'),str(p),'run'],capture_output=True,text=True,timeout=40,creationflags=0x08000000)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'GATE_ONLY_PASSED')


if __name__=='__main__':unittest.main()
