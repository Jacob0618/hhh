"""Author offline entry, transport and audit fixtures. No production actions."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools import run_five_baselines_hypermem_native_v2 as c
from tools import audit_hypermem_continuation_v2 as audit
from tools import hypermem_prefix_api_v2 as prefix
from tools.baselines_common_v1 import save_new,read,sha,BaselineFatal
from jev_mem import providers
from jev_mem.gemini25_ablation_provider_v1 import generation_body,READER_STAGE,JUDGE_STAGE
import requests


class BeforeWorker(BaseException):pass


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=c.ROOT/'research');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.new=self.root/'hypermem_native_v2'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c,'NEW',self.new))
        self.stack.enter_context(patch.object(c.admission,'assert_stopped',lambda:None))
        source=c.ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/author_hypermem_boundary_terminal_v2.json'
        parent=read(c.OLD/'run_manifest.json');v=read(source)
        self.proof={'rows':v['rows'],'registry':{r['request_hash']:r for r in v['rows']},'questions':parent['questions'],
                    'original_inventory':v['original_inventory'],'original_physical_usage':c.admission.usage_presence([r['physical'] for r in v['rows']])}
        self.proof_path=self.root/'proof.json';save_new(self.proof_path,self.proof)
        self.stack.enter_context(patch.object(c.admission,'validate_proof',lambda p:read(p)))
        self.review=self.root/'review.json'
        save_new(self.review,{'schema':c.REVIEW_SCHEMA,'status':'passed','fixture_only':True,'production_launch_authorized':False,
            'proof_sha256':sha(self.proof_path),'specification':parent['specification'],'code_sha256':c.code_hashes(parent),'upstream_sha256':parent['upstream_sha256'],
            'evidence_sha256':{str(self.proof_path):sha(self.proof_path)}})
        self.refs={}
        for name in ('amem','simplemem'):
            self.refs[name]={}
            for kind in ('completion','receipt','plan'):
                p=self.root/'prior'/name/(kind+'.json');save_new(p,{'fixture':True,'method':name,'kind':kind});self.refs[name][kind]=p
        self.stack.enter_context(patch.object(c.chain,'completion_paths',lambda method:self.refs))
        self.gates=[]
        self.stack.enter_context(patch.object(c.chain.previous,'verify_predecessors',lambda method:self.gates.append(method)))
        self.stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))
        self.stack.enter_context(patch.object(socket.socket,'connect',side_effect=AssertionError('No network')))
    def main(self,*args):
        with patch.object(sys,'argv',[str(c.ROOT/c.ENTRY),*args]),redirect_stdout(io.StringIO()):c.main()
    def prepare(self):self.main('prepare','--review',str(self.review),'--proof',str(self.proof_path))
    def test_actual_prepare_manifest_run_to_real_threadpool_worker(self):
        self.prepare();self.main('check');captured=[];original=c.private_controller
        def factory(m,proof,ticket):
            result=original(m,proof,ticket)
            result.load_data=lambda:[{'fixture':'before native'}]
            def worker(folder,method,ci,qs,item):
                captured.append((folder,method,ci,len(qs),item))
                self.assertTrue((self.new/'continuation_execution_intent.json').is_file())
                self.assertTrue((self.new/'execution_intent.json').is_file())
                self.assertFalse((self.new/'work').exists())
                self.assertEqual(result.adapter(method),c.adapter)
                raise BeforeWorker()
            result.conv_worker=worker
            return result
        with patch.object(c,'private_controller',factory),self.assertRaises(BeforeWorker):self.main('run')
        self.assertEqual(len(captured),1);self.assertEqual(captured[0][:3],(self.new,'hypermem',0))
        self.assertEqual(self.gates,['hypermem','hypermem'])
        before={str(p):sha(p) for p in self.new.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):self.main('run')
        self.assertEqual(before,{str(p):sha(p) for p in self.new.rglob('*') if p.is_file()})
    def test_gate_and_manifest_failure_cannot_write_run_artifacts(self):
        with patch.object(c.chain.previous,'verify_predecessors',side_effect=ValueError('Missing predecessor')):
            with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.new.exists());self.prepare()
        before=(self.new/'run_manifest.json').read_bytes()
        with patch.object(c.chain.previous,'verify_predecessors',side_effect=ValueError('Missing predecessor')):
            with self.assertRaises(ValueError):self.main('run')
        self.assertEqual(sorted(p.name for p in self.new.iterdir()),['run_manifest.json'])
        self.assertEqual(before,(self.new/'run_manifest.json').read_bytes())
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(before,(self.new/'run_manifest.json').read_bytes())
    def test_ticket_stale_evidence_wrong_method_and_once(self):
        self.prepare();m,proof=c.manifest();refs=c.chain.strong_predecessors('hypermem');t=c.GateTicket(m,proof,refs)
        with self.assertRaises(ValueError):t.consume('vmem')
        path=self.refs['simplemem']['receipt'];old=path.read_bytes();path.write_text('{}')
        with self.assertRaises(ValueError):t.consume('hypermem')
        path.write_bytes(old);t.consume('hypermem')
        with self.assertRaises(ValueError):t.consume('hypermem')
        self.assertFalse((self.new/'execution_intent.json').exists())
    def test_changed_review_evidence_rejected_before_prepare(self):
        path=self.root/'support.json';save_new(path,{'fixture':True});v=read(self.review)
        v['evidence_sha256'][str(path)]=sha(path);self.review.write_text(json.dumps(v))
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'supporting evidence'):self.prepare()
        self.assertFalse(self.new.exists())
    def test_preserves_all_science_functions_and_original_globals(self):
        self.prepare();m,p=c.manifest();t=c.GateTicket(m,p,c.chain.strong_predecessors('hypermem'))
        result=c.private_controller(m,p,t)
        for name in ('qa_one','conv_worker','known_recoverable','replay_qa_leaf','seal'):
            self.assertEqual(getattr(result,name).__code__,getattr(c.chain.base,name).__code__)
        self.assertIs(c.chain.base.BaselineAPI,c.chain.BaselineAPI)
        self.assertEqual(result.adapter('hypermem').describe(),c.chain.base.adapter('hypermem').describe())
        self.assertEqual(result.manifest('hypermem')['specification'],read(c.OLD/'run_manifest.json')['specification'])


class PrefixPhases(unittest.TestCase):
    def test_true_prefix_method_and_two_qa_phases_have_separate_journals(self):
        report=read(c.ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/author_hypermem_boundary_terminal_v2.json')
        qs=read(c.OLD/'run_manifest.json')['questions'];rows=report['rows']
        proof={'rows':rows,'registry':{r['request_hash']:r for r in rows},'questions':qs}
        case=rows[0];q=next(q for q in qs if q['conversation']==0)
        class Response:
            status_code=200;ok=True;headers={}
            def json(self):return {'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':'{}'}]}}],'modelVersion':'gemini-2.5-flash'}
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp,patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')):
            root=Path(tmp);API=prefix.admitted_api(proof,root);api=API(root/'work/api_method',context=case['context'],keys={'GEMINI_API_KEY':'fake'})
            sig=case['signature']
            with patch.object(requests.Session,'post',side_effect=AssertionError('No inherited transport')):
                api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=case['stage'])
            with patch.object(requests.Session,'post',lambda *a,**k:Response()):
                api.generate('future boundary',json_output=True,stage='hypermem_boundary')
                for phase in ('firstpass','recovery'):
                    qa=API(root/('work/api_'+phase),context={'method':'hypermem','conversation':0,'query_id':q['query_id'],'phase':phase},keys={'GEMINI_API_KEY':'fake'})
                    # Use the unchanged evaluator to construct exact QA bodies;
                    # intentionally invalid content is retained, then caught.
                    try:c.chain.base.evaluator.read_answer(qa,q['question'],q['reference'],[{'id':'native_context','text':'context','observation_date':None}],reference_date=q['reference_date'])
                    except (Exception,BaselineFatal):pass
            self.assertEqual(len(audit.jsonl(root/'work/prefix/c0.jsonl')),2)
            self.assertEqual(len(audit.jsonl(root/'work/api_method/logical_events.jsonl')),2)
            for name in ('api_method','api_firstpass','api_recovery'):
                value,_,_,_=audit.api_audit(root/'work'/name,proof)
                self.assertEqual(value['physical'],1)
            self.assertEqual(audit.api_audit(root/'work/api_method',proof)[0]['cache'],1)
            # A forged context which retains the exact same raw must be refused.
            path=root/'work/api_firstpass/logical_events.jsonl';v=audit.jsonl(path);v[0]['context']['conversation']=1
            path.write_text('\n'.join(json.dumps(x) for x in v)+'\n')
            with self.assertRaises(ValueError):audit.api_audit(root/'work/api_firstpass',proof)


class Guard(unittest.TestCase):
    def test_real_venv_parent_four_deep_stopped_functions(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp:
            p=Path(tmp)/'run_five_baselines_hypermem_native_v2.py'
            p.write_text('import sys\nsys.path.insert(0,'+repr(str(c.ROOT))+')\nfrom tools import audit_hypermem_terminal_v2 as a\nwith a.guard_context():\n a.assert_stopped()\n a.chain.previous.simple.stopped()\n a.chain.previous.previous.archive.old.engine.stopped()\n a.chain.previous.previous.frozen.lifecycle.stopped()\nprint("FOUR_DEEP_GUARDS_PASSED")\n')
            r=subprocess.run([sys.executable,'-B',str(p),'run'],cwd=c.ROOT,capture_output=True,text=True,timeout=45)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('FOUR_DEEP_GUARDS_PASSED',r.stdout)


if __name__=='__main__':unittest.main()
