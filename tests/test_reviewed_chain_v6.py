"""Offline dispatch fixtures: no actual baseline preparation or provider call."""
from contextlib import ExitStack,redirect_stdout
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_five_baselines_reviewed_chain_v6 as c
from tools.baselines_common_v1 import read,sha,save_new
from jev_mem import providers

class BeforeWorker(BaseException):pass

class Dispatch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=c.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runroot=self.root/'run';self.hyper=self.runroot/'hypermem'
        source=c.base.paths('hypermem')/'run_manifest.json';self.original=source.read_bytes()
        self.hyper.mkdir(parents=True);(self.hyper/'run_manifest.json').write_bytes(self.original)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c.base,'RUN',self.runroot))
        self.stack.enter_context(patch.object(c,'assert_stopped',lambda:None))
        self.calls=[]
        refs={}
        for method in ('amem','simplemem','hypermem','vmem'):
            refs[method]={}
            for kind in ('completion','receipt','plan'):
                p=self.root/'predecessors'/method/(kind+'.json');save_new(p,{'fixture':method,'kind':kind})
                refs[method][kind]=p
        self.refs=refs
        def completions(method):
            names=('amem','simplemem','hypermem','vmem')[:{'hypermem':2,'vmem':3,'prism':4}[method]]
            return {n:refs[n] for n in names}
        self.stack.enter_context(patch.object(c,'completion_paths',completions))
        self.stack.enter_context(patch.object(c.previous,'verify_predecessors',lambda method:self.calls.append(method)))
        self.stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No credentials')))
        self.stack.enter_context(patch.object(socket.socket,'connect',side_effect=AssertionError('No network')))
        q=self.root/'qualification.json'
        save_new(q,{'schema':'hypermem-chain5-zero-dispatch-independent-diagnosis-v1','status':'confirmed_local_gate_failure',
            'namespace_inventory':{str(self.hyper/'run_manifest.json'):c.ORIGINAL_MANIFEST_SHA},'namespace_before_after_equal':True,
            'producer_count':0,'logical_consumer_count':0,'provider_intents_count':0,'native_artifacts_count':0})
        self.stack.enter_context(patch.object(c,'QUALIFICATION',str(q.relative_to(c.ROOT))))
        self.stack.enter_context(patch.object(c,'QUALIFICATION_SHA',sha(q)))
        self.review=self.make_review('hypermem')
    def make_review(self,method):
        old=read(c.ROOT/'research/baseline_execution_2026-10-08/downstream_dispatch_simplemem_native_v2'/f'{method}_preparation_review_v1.json')
        value={**old,'schema':c.REVIEW_SCHEMA,'fixture_only':True,'production_launch_authorized':False,
            'code_sha256':{**old['code_sha256'],**c.code_hashes()},
            'evidence_sha256':{c.PRIOR_REVIEW_ROOT+'/'+method+'_preparation_review_v1.json':c.PRIOR_REVIEW_SHA[method]}}
        p=self.root/(method+'_review.json');save_new(p,value);return p
    def main(self,*args):
        with patch.object(sys,'argv',[str(c.ROOT/c.ENTRY),*args]),redirect_stdout(io.StringIO()):c.main()
    def test_actual_prepare_check_run_reaches_load_data_with_both_intents(self):
        self.main('prepare','--method','hypermem','--review',str(self.review))
        self.assertEqual(self.original,(self.hyper/'run_manifest.json').read_bytes())
        self.assertFalse((self.hyper/'execution_intent.json').exists())
        self.main('check','--method','hypermem')
        def stop():
            self.assertTrue((self.hyper/'execution_intent.json').is_file())
            intent=read(c.sidecar('hypermem')/'execution_intent.json')
            self.assertEqual(intent['ticket']['method'],'hypermem')
            self.assertEqual(set(intent['ticket']['predecessor_acceptance']['verified_objects']),{'amem','simplemem'})
            self.assertEqual(intent['ticket']['review']['sha256'],sha(self.review))
            self.assertEqual(read(self.hyper/'execution_intent.json')['manifest_sha256'],c.ORIGINAL_MANIFEST_SHA)
            self.assertEqual(read(self.hyper/'status.json')['stage'],'firstpass')
            raise BeforeWorker()
        with patch.object(c.base,'load_data',stop):
            with self.assertRaises(BeforeWorker):self.main('run','--method','hypermem')
        self.assertEqual(self.calls,['hypermem','hypermem'])
        with self.assertRaisesRegex(ValueError,'namespace|required|directory'):self.main('run','--method','hypermem')
        self.assertEqual(self.original,(self.hyper/'run_manifest.json').read_bytes())
    def test_missing_strong_completion_never_prepares_or_runs(self):
        with patch.object(c.previous,'verify_predecessors',side_effect=ValueError('Missing actual completion')):
            with self.assertRaisesRegex(ValueError,'Missing actual'):self.main('prepare','--method','hypermem','--review',str(self.review))
        self.assertFalse(c.sidecar('hypermem').exists())
        self.main('prepare','--method','hypermem','--review',str(self.review))
        with patch.object(c.previous,'verify_predecessors',side_effect=ValueError('Missing actual completion')):
            with self.assertRaisesRegex(ValueError,'Missing actual'):self.main('run','--method','hypermem')
        self.assertFalse((self.hyper/'execution_intent.json').exists());self.assertFalse((c.sidecar('hypermem')/'execution_intent.json').exists())
    def test_ticket_wrong_method_stale_manifest_and_second_use_rejected(self):
        self.main('prepare','--method','hypermem','--review',str(self.review));m,t=c.manifest('hypermem')
        refs=c.strong_predecessors('hypermem');ticket=c._GateTicket(c._ISSUER,'hypermem',m,t,refs)
        with self.assertRaisesRegex(ValueError,'wrong method'):ticket.consume('vmem')
        cp=self.refs['simplemem']['completion'];original=cp.read_bytes();cp.write_text('{}','utf8')
        with self.assertRaisesRegex(ValueError,'evidence changed'):ticket.consume('hypermem')
        cp.write_bytes(original);ticket.consume('hypermem')
        with self.assertRaisesRegex(ValueError,'reused'):ticket.consume('hypermem')
    def test_extra_empty_work_and_parent_manifest_tamper_block_before_gate(self):
        (self.hyper/'work').mkdir()
        with self.assertRaisesRegex(ValueError,'directory'):self.main('prepare','--method','hypermem','--review',str(self.review))
        self.assertEqual(self.calls,[])
        (self.hyper/'work').rmdir()
        p=self.hyper/'run_manifest.json';p.write_text(p.read_text(encoding='utf8')+'\n','utf8')
        with self.assertRaisesRegex(ValueError,'manifest changed'):self.main('prepare','--method','hypermem','--review',str(self.review))
        self.assertEqual(self.calls,[])
    def test_technical_manifest_tamper_and_old_review_rejected(self):
        self.main('prepare','--method','hypermem','--review',str(self.review))
        p=c.sidecar('hypermem')/'manifest.json';value=read(p);value['historical_cache']='allowed';p.write_text(json.dumps(value),'utf8')
        with self.assertRaisesRegex(ValueError,'Technical dispatch'):self.main('run','--method','hypermem')
        self.assertFalse((self.hyper/'execution_intent.json').exists())
    def test_future_fresh_methods_use_original_prepare_and_root(self):
        for method in ('vmem','prism'):
            self.main('prepare','--method',method,'--review',str(self.make_review(method)))
            m=read(self.runroot/method/'run_manifest.json')
            self.assertEqual(m['method'],method);self.assertEqual(m['code_sha256'][c.ENTRY],sha(c.ROOT/c.ENTRY))
            self.assertFalse(c.sidecar(method).exists())
            with patch.object(c.base,'load_data',side_effect=BeforeWorker()):
                with self.assertRaises(BeforeWorker):self.main('run','--method',method)
            intent=read(c.sidecar(method)/'execution_intent.json')
            self.assertIsNone(intent['ticket']['technical_manifest'])
            self.assertNotIn('zero_call_admission',intent['ticket'])
    def test_private_run_only_changes_exact_legacy_loop(self):
        desc=c.bridge_description()
        self.assertTrue(desc['scientific_ast_outside_gate_equal'])
        self.assertEqual(desc['removed_loop_sha256'],'9cd69df79dc4e625e8863323b7a3927b22c1fbbf492c6d7c90a41490e70b61e6')
        with patch.object(c,'LEGACY_LOOP',c.LEGACY_LOOP+' '):
            with self.assertRaisesRegex(ValueError,'boundary'):c.bridge_description()

class Guard(unittest.TestCase):
    def test_all_deep_guard_bindings_restore_on_exception(self):
        before=(c.guard.active_controllers,c.guard.assert_stopped,c.previous.simple._stop)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with c.guard_context():
                self.assertIs(c.guard.active_controllers,c.active_controllers)
                self.assertIs(c.guard.assert_stopped,c.assert_stopped)
                self.assertIs(c.previous.simple._stop,c.assert_stopped)
                raise RuntimeError('fixture')
        self.assertEqual(before,(c.guard.active_controllers,c.guard.assert_stopped,c.previous.simple._stop))
    def test_real_venv_parent_all_deep_stopped_functions(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as temp:
            script=Path(temp)/'run_five_baselines_reviewed_chain_v6.py'
            script.write_text("import sys\nsys.path.insert(0,"+repr(str(c.ROOT))+")\nfrom tools import run_five_baselines_reviewed_chain_v6 as c\nwith c.guard_context():\n c.assert_stopped()\n c.previous.simple.stopped()\n c.previous.previous.archive.old.engine.stopped()\n c.previous.previous.frozen.lifecycle.stopped()\nprint('FOUR_DEEP_GUARDS_PASSED')\n",encoding='utf8')
            result=subprocess.run([sys.executable,'-B',str(script),'run','--method','hypermem'],cwd=c.ROOT,capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('FOUR_DEEP_GUARDS_PASSED',result.stdout)

if __name__=='__main__':unittest.main()
