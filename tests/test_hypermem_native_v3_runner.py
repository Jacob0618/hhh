"""Temporary main-route fixtures; actual full-prefix evidence is tested separately.

Only terminal collection and long author-preflight validation are fixture
substitutes. Review/hash/code, private original run, exclusive intents, tickets
and worker dispatch are the real new implementation.
"""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests import test_hypermem_prefix_api_v3 as factory
from tools import run_five_baselines_hypermem_native_v3 as c
from tools.baselines_common_v1 import save_new,read,sha,BaselineFatal
from jev_mem import providers


class BeforeWorker(BaseException):pass


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.f=factory.Prefix();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.folder;self.new=self.root/'continuation';self.proof=deepcopy(self.f.proof)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c,'NEW',self.new))
        self.stack.enter_context(patch.object(c.admission,'assert_stopped',lambda:None))
        self.stack.enter_context(patch.object(c.admission,'validate_proof',lambda p:read(p)))
        self.stack.enter_context(patch.object(c,'check_preflight',lambda *a:None))
        parent=read(c.OLD/'run_manifest.json');self.proof.update(questions=parent['questions'],original_usage={},new_usage={})
        self.pp=self.root/'dispatch_proof.json';save_new(self.pp,self.proof)
        self.preflight=self.root/'synthetic_preflight_placeholder.json';save_new(self.preflight,{'scope':'route_fixture_only'})
        self.review=self.root/'dispatch_review.json'
        save_new(self.review,{'schema':c.REVIEW_SCHEMA,'status':'passed','fixture_only':True,'production_launch_authorized':False,
            'proof_sha256':sha(self.pp),'specification':parent['specification'],'code_sha256':c.code_hashes(parent),
            'upstream_sha256':parent['upstream_sha256'],'continuation_policy':c.policy(self.proof,True),
            'hash_environment':c.admission.hash_environment(),'prefix_preflight':c.binding(self.preflight),
            'evidence_sha256':{str(self.preflight):sha(self.preflight)}})
        self.refs={}
        for method in ('amem','simplemem'):
            self.refs[method]={}
            for kind in ('completion','receipt','plan'):
                p=self.root/'prior'/method/(kind+'.json');save_new(p,{'fixture':True,'method':method,'kind':kind});self.refs[method][kind]=p
        self.stack.enter_context(patch.object(c.chain,'completion_paths',lambda method:self.refs))
        self.gates=[];self.stack.enter_context(patch.object(c.chain.previous,'verify_predecessors',lambda m:self.gates.append(m)))
        self.stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))
    def main(self,*args):
        with patch.object(sys,'argv',[str(c.ROOT/c.ENTRY),*args]),redirect_stdout(io.StringIO()):c.main()
    def prepare(self):self.main('prepare','--review',str(self.review),'--proof',str(self.pp))
    def test_real_prepare_check_main_threadpool_and_two_exclusive_intents(self):
        self.prepare();self.main('check');captured=[];original=c.private_controller
        def factory(m,p,ticket):
            controller=original(m,p,ticket);controller.load_data=lambda:[{'fixture':True} for _ in range(10)]
            def worker(folder,method,ci,qs,item):
                captured.append(ci)
                self.assertTrue((self.new/'continuation_execution_intent.json').exists())
                self.assertTrue((self.new/'execution_intent.json').exists())
                self.assertFalse((self.new/'work').exists())
                self.assertIs(controller.adapter(method),c.adapter)
                raise BeforeWorker()
            controller.conv_worker=worker;return controller
        with patch.object(c,'private_controller',factory),self.assertRaises(BeforeWorker):self.main('run')
        self.assertEqual(set(captured),set(range(10)));self.assertEqual(self.gates,['hypermem','hypermem'])
        before=c.admission.inventory(self.new)
        with self.assertRaises(ValueError):self.main('run')
        self.assertEqual(before,c.admission.inventory(self.new))
    def test_guard_admission_prepare_reentry_does_not_mutate_namespace(self):
        with patch.object(c.chain.previous,'verify_predecessors',side_effect=ValueError('Missing predecessor')):
            with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.new.exists());self.prepare();before=c.admission.inventory(self.new)
        with self.assertRaises(ValueError):self.prepare()
        with patch.object(c.chain.previous,'verify_predecessors',side_effect=ValueError('Missing predecessor')):
            with self.assertRaises(ValueError):self.main('run')
        self.assertEqual(before,c.admission.inventory(self.new))
    def test_policy_prospective_and_replacement_are_explicit_and_exact(self):
        saved=self.review.read_bytes()
        for key,value in (('prospective_role_representation',1),('replacement_logical_limit',2),('known_invalid_episode_replacement',False)):
            review=json.loads(saved);review['continuation_policy'][key]=value;self.review.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,'compatibility policy'):self.prepare()
            self.assertFalse(self.new.exists())
        self.review.write_bytes(saved)
    def test_ticket_evidence_changes_wrong_method_and_reuse_rejected(self):
        self.prepare();m,p=c.manifest();ticket=c.GateTicket(m,p,c.chain.strong_predecessors('hypermem'))
        with self.assertRaises(ValueError):ticket.consume('vmem')
        saved=self.preflight.read_bytes();self.preflight.write_text('{}')
        with self.assertRaises(ValueError):ticket.consume('hypermem')
        self.preflight.write_bytes(saved);ticket.consume('hypermem')
        with self.assertRaises(ValueError):ticket.consume('hypermem')
    def test_semantic_failure_in_real_conv_worker_never_reaches_retrieval_or_qa(self):
        self.prepare();m,p=c.manifest();controller=c.private_controller(m,p,SimpleNamespace(consume=lambda m:None))
        class FailedAdapter:
            def build(self,*a):raise BaselineFatal('Still malformed after one replacement')
            def retrieve(self,*a):raise AssertionError('Partial bank cannot retrieve')
        controller.adapter=lambda method:FailedAdapter()
        controller.BaselineAPI=lambda *a,**kw:SimpleNamespace(_redact=lambda s:s)
        controller.qa_one=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No QA after failed build'))
        value=controller.conv_worker(self.new,'hypermem',0,[],{'conversation':{'speaker_a':'A','speaker_b':'B'}})
        self.assertEqual(value['status'],'needs_review');self.assertFalse((self.new/'results/c0/build.json').exists())
    def test_original_scientific_functions_and_global_provider_unchanged(self):
        self.prepare();m,p=c.manifest();controller=c.private_controller(m,p,SimpleNamespace(consume=lambda m:None))
        for name in ('qa_one','conv_worker','known_recoverable','replay_qa_leaf','seal'):
            self.assertEqual(getattr(controller,name).__code__,getattr(c.chain.base,name).__code__)
        self.assertIs(c.chain.base.BaselineAPI,c.chain.BaselineAPI)


class Guard(unittest.TestCase):
    def test_real_parent_and_all_deep_runtime_guards(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp:
            path=Path(tmp)/'run_five_baselines_hypermem_native_v3.py'
            path.write_text('import sys\nsys.path.insert(0,'+repr(str(c.ROOT))+')\nfrom tools import audit_hypermem_terminal_v3 as a\na.check_hash_environment()\nwith a.guard_context():\n a.assert_stopped()\n a.chain.previous.simple.stopped()\n a.chain.previous.previous.archive.old.engine.stopped()\n a.chain.previous.previous.frozen.lifecycle.stopped()\nprint("FOUR_DEEP_GUARDS_PASSED")\n')
            r=subprocess.run([sys.executable,'-B',str(path),'run'],cwd=c.ROOT,text=True,capture_output=True,timeout=45)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('FOUR_DEEP_GUARDS_PASSED',r.stdout)


if __name__=='__main__':unittest.main()
