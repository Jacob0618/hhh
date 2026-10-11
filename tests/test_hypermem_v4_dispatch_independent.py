"""Actual main/prepare/check/run dispatch with isolated admission fixtures.

Human trust, the large terminal inventory/preflight, and completed predecessor
packages are explicit synthetic substitutes. Code/review bindings, manifest
creation, single-use ticket, original run body, exclusive intents and the real
ThreadPoolExecutor execute, stopping before any worker/provider work.
"""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests import test_hypermem_v4_entry_audit_independent as fixtures
from tools import run_five_baselines_hypermem_native_v4 as c
from tools import hypermem_prefix_api_v4 as prefix
from tools.baselines_common_v1 import save_new,read,sha
from jev_mem import providers


class BeforeWorker(BaseException):pass


class Dispatch(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='v4-dispatch-independent-',dir=c.ROOT/'research')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        helper=fixtures.FixedGrantBinding()
        self.grant,self.review,self.pp,_,gv,self.proof=helper.setup_grant(self.root,self.stack)
        self.new=self.root/'new';self.old=self.root/'old';self.ancestor=self.root/'ancestor'
        questions=[{'conversation':ci,'query_id':f'c{ci}_q0','question':'synthetic before-worker'} for ci in range(10)]
        self.proof.update(schema=prefix.PROOF_SCHEMA,status='candidate_evidence_only',
            conversations=[{'conversation':ci,'prefix':[{'ordinal':1,'synthetic':True}]} for ci in range(10)],
            producer_registry={},role_cases=[],questions=questions,corrected_historical_usage={},hash_environment=c.admission.hash_environment())
        self.pp.write_text(json.dumps(self.proof),encoding='utf-8')
        self.parent={'specification':{'fixture':'dispatch-only'},'code_sha256':{},
            'upstream_sha256':c.chain.base.source_inventory('hypermem'),'questions':questions}
        save_new(self.old/'run_manifest.json',self.parent);save_new(self.ancestor/'run_manifest.json',{'fixture':'original'})
        self.preflight=self.root/'preflight-placeholder.json';save_new(self.preflight,{'fixture_only':True})
        rv={'schema':c.REVIEW_SCHEMA,'status':'passed_implementation_and_zero_api_preflight_only','production_launch_authorized':False,
            'proof_sha256':sha(self.pp),'source_bindings':self.proof['source_bindings'],'specification':self.parent['specification'],
            'continuation_policy':c.policy(self.proof),'code_sha256':c.code_hashes(self.parent),'upstream_sha256':self.parent['upstream_sha256'],
            'evidence_sha256':{str(self.preflight):sha(self.preflight)},'hash_environment':c.admission.hash_environment(),
            'prefix_preflight':c.binding(self.preflight)}
        self.review.write_text(json.dumps(rv),encoding='utf-8')
        gv.update(terminal_proof=c.binding(self.pp),implementation_review=c.binding(self.review),code_sha256=rv['code_sha256'])
        self.grant.write_text(json.dumps(gv),encoding='utf-8')
        for key,value in (('NEW',self.new),('OLD',self.old),('ANCESTOR',self.ancestor)):
            self.stack.enter_context(patch.object(c,key,value))
        self.stack.enter_context(patch.dict(c.private_controller.__globals__,{'NEW':self.new,'OLD':self.old,'ANCESTOR':self.ancestor}))
        self.stack.enter_context(patch.object(c.admission,'validate_proof',side_effect=lambda path:read(path)))
        self.stack.enter_context(patch.object(c,'check_preflight',return_value={'fixture_only':True}))
        self.stack.enter_context(patch.object(c.admission,'assert_stopped',return_value=None))
        self.refs={}
        for method in ('amem','simplemem'):
            self.refs[method]={}
            for kind in ('completion','receipt','plan'):
                p=self.root/'predecessors'/method/(kind+'.json');save_new(p,{'fixture':True});self.refs[method][kind]=p
        self.accepted={n:{k:c.binding(p) for k,p in rows.items()} for n,rows in self.refs.items()}
        self.strong=self.stack.enter_context(patch.object(c.chain,'strong_predecessors',return_value=self.accepted))
        self.stack.enter_context(patch.object(c.chain,'completion_paths',return_value=self.refs))
        self.stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))

    def main(self,*args):
        with patch.object(sys,'argv',['independent-dispatch-fixture',*args]),redirect_stdout(io.StringIO()):c.main()

    def prepare(self):self.main('prepare','--proof',str(self.pp),'--review',str(self.review))

    def test_real_three_main_routes_reach_ten_workers_only_after_two_exclusive_intents(self):
        self.prepare();self.main('check');seen=[];original=c.private_controller
        def factory(m,proof,ticket):
            ctl=original(m,proof,ticket);ctl.load_data=lambda:[{'fixture':True} for _ in range(10)]
            for name in ('qa_one','conv_worker','known_recoverable','replay_qa_leaf','seal'):
                self.assertEqual(getattr(ctl,name).__code__,getattr(c.chain.base,name).__code__)
            def worker(folder,method,ci,qs,item):
                seen.append(ci)
                self.assertTrue((self.new/'continuation_execution_intent.json').exists())
                self.assertTrue((self.new/'execution_intent.json').exists())
                self.assertFalse((self.new/'work').exists())
                self.assertIs(ctl.adapter(method),c.adapter)
                raise BeforeWorker()
            ctl.conv_worker=worker;return ctl
        with patch.object(c,'private_controller',side_effect=factory),self.assertRaises(BeforeWorker):self.main('run')
        self.assertEqual(set(seen),set(range(10)));self.assertEqual(self.strong.call_count,2)
        self.assertEqual(read(self.new/'status.json')['stage'],'needs_review')
        before={str(p):p.read_bytes() for p in self.new.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):self.main('run')
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.new.rglob('*') if p.is_file()})

    def test_failed_predecessor_or_changed_evidence_has_no_new_run_output(self):
        with patch.object(c.chain,'strong_predecessors',side_effect=ValueError('independent missing completion')):
            with self.assertRaisesRegex(ValueError,'missing completion'):self.prepare()
        self.assertFalse(self.new.exists())
        original=self.preflight.read_bytes();self.preflight.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Reviewed source/evidence changed'):self.prepare()
        self.assertFalse(self.new.exists());self.preflight.write_bytes(original)
        self.prepare();before={str(p):p.read_bytes() for p in self.new.iterdir()}
        with self.assertRaisesRegex(ValueError,'Fresh unstarted'):self.prepare()
        with patch.object(c.chain,'strong_predecessors',side_effect=ValueError('independent missing completion')):
            with self.assertRaisesRegex(ValueError,'missing completion'):self.main('run')
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.new.iterdir()})


if __name__=='__main__':unittest.main(verbosity=2)
