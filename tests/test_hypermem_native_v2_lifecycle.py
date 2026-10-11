"""New lifecycle only: synthetic artifacts, never production cleanup/requests."""
from contextlib import ExitStack
from copy import deepcopy
import ast
import io
import os
import subprocess
import sys
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_hypermem_native_v2 as a
from tools import run_five_baselines_reviewed_chain_v7 as chain
from tools.baselines_common_v1 import read,sha,save_new


class Retention(unittest.TestCase):
    def test_only_exact_native_runnable_artifacts_are_disposed(self):
        native=a.NEW/'work/native/c2'
        self.assertEqual(a.disposition(native/'cache/hypergraph_bm25_index_conv_0.pkl'),'runnable_hypermem_bm25_vector_index')
        self.assertEqual(a.disposition(native/('embedding_evidence/query_'+'a'*64+'.json')),'runnable_normalized_native_embedding_cache')
        for p in (native/'calls/x.json',native/'embedding_evidence/.integrity/x.json.sha256',a.NEW/'work/api_method/api_cache/raw.json',native/'__pycache__/unique.json'):
            self.assertIsNone(a.disposition(p))
        for p in (native/'calls/response.json.tmp',native/'cache/unknown.pkl',a.ROOT/'research/temporary/embedding_evidence'/('document_'+'a'*64+'.json')):
            with self.assertRaises(ValueError):a.disposition(p)

    def test_resolver_never_returns_disposed_bytes_or_sha(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td,patch.object(a,'OLD',Path(td)/'old'),patch.object(a,'NEW',Path(td)/'new'):
            p=a.NEW/'work/native/c0/cache/hypergraph_embedding_index_conv_0.pkl';p.parent.mkdir(parents=True);p.write_bytes(b'not in ZIP')
            rows=[{'path':str(p),'sha256':sha(p),'size':p.stat().st_size,'retained':False,'disposal_reason':a.disposition(p)}]
            mp=Path(td)/'manifest.json';zp=Path(td)/'evidence.zip';save_new(mp,{'schema':a.SCHEMA,'records':rows})
            with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
            with a.Resolver(zp,mp) as view:
                for fn in (view.bytes,view.sha,view.read):
                    with self.assertRaises(ValueError):fn(p)
                self.assertFalse(a.virtual_paths(view)(p).exists())


class FullArchive(unittest.TestCase):
    FULL_COHORT=False
    def test_ten_real_banks_full_native_and_qa_after_both_fixture_roots_deleted(self):
        from tests import test_hypermem_native_v2_full_audit as factory
        c=factory.c;original=c.private_controller;captured={}
        class Finished(BaseException):pass
        def controller(m,proof,ticket):
            value=original(m,proof,ticket);original_seal=value.seal
            # The reused fixture's admission reads the real immutable parent
            # manifest, while its native parent is temporary. Route only the
            # summary hook's recorded parent path to that temporary owner.
            value.save_new=a.shared_replay.clone(value.save_new,OLD=c.admission.OLD)
            def seal(method):
                if self.FULL_COHORT:self.expand_cohort(value,c.NEW,m,proof)
                result=original_seal(method)
                self.archive_fixture(c.admission.OLD,c.NEW,m,proof,c.admission.chain.base.load_data())
                captured['passed']=True
                raise Finished()
            value.seal=seal;return value
        with patch.object(c,'private_controller',controller):
            with self.assertRaises(Finished):factory.CompleteNativeAudit().test_ten_real_native_banks_raw_embeddings_qa_and_recovery_full_audit()
        self.assertTrue(captured['passed'])

    def expand_cohort(self,controller,new,m,proof):
        # Synthetic repeated questions, genuine original capture/QA/cache code;
        # this verifies a full 1540-slot archive contract, never task accuracy.
        original=next(q for q in proof['questions'] if q['conversation']==1)
        state=read(new/'work/native/c1/state.json')
        additional=[{**original,'query_id':'c1_fixture'+str(i),'source_qa_index':1000+i} for i in range(1530)]
        proof['questions'].extend(additional)  # m.questions shares this fixture list.
        controller.BaselineAPI.admission={**controller.BaselineAPI.admission,
            'questions':{q['query_id']:q['conversation'] for q in proof['questions']}}
        for q in additional:
            api=controller.BaselineAPI(new/'work/api_method',context={'method':'hypermem','conversation':1,'phase':'retrieval','query_id':q['query_id']})
            ret=controller.capture_stage(new/'results/c1/retrieval'/(q['query_id']+'.json'),lambda:controller.adapter('hypermem').retrieve(q['question'],state,api),api)
            self.assertEqual(ret['status'],'complete')
            row=controller.qa_one(new,'hypermem',q,ret['value'])
            self.assertEqual(row['judge'],'complete')
        pp=Path(m['terminal_proof']['path']);pp.write_text(json.dumps(proof),encoding='utf8')
        rp=Path(m['review']['path']);rv=read(rp);rv['proof_sha256']=sha(pp);rv['evidence_sha256'][str(pp)]=sha(pp);rp.write_text(json.dumps(rv),encoding='utf8')
        m['terminal_proof']=a.binding(pp);m['parent_manifest']=a.binding(pp);m['review']=a.binding(rp)
        (new/'run_manifest.json').write_text(json.dumps(m),encoding='utf8')

    def archive_fixture(self,old,new,m,proof,dataset):
        root=old.parent
        with patch.object(a,'OLD',old),patch.object(a,'NEW',new):
            before=a.inventories();cert={'run_inventories':before,'banks':a.banks_for(proof,before)}
            expected={name:read(new/'results'/name) for name in a.OUTPUTS}
            paths={p for r in (old,new) for p in r.rglob('*') if p.is_file()}
            paths.update(a.ROOT/p for p in (*m['code_sha256'],*m['upstream_sha256']))
            paths.add(a.ROOT/a.metric.METRIC)
            paths.add(a.ROOT/'tools/audit_hypermem_continuation_v2.py')
            data={str(p):p.read_bytes() for p in paths}
            # This single fixture input is synthetic; no original dataset bytes
            # are modified and no real-cohort performance claim is made.
            data[str(a.ROOT/'data/locomo10.json')]=json.dumps(dataset).encode()
            records=[]
            for path,b in sorted(data.items()):
                reason=a.disposition(path);h=hashlib.sha256(b).hexdigest()
                r={'path':path,'size':len(b),'sha256':h,'retained':reason is None}
                if reason:r['disposal_reason']=reason
                else:r['member']='blobs/'+h
                records.append(r)
            mp=root/'zip_manifest.json';zp=root/'evidence.zip';save_new(mp,{'schema':a.SCHEMA,'records':records})
            with zipfile.ZipFile(zp,'x') as z:
                z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
                for r in records:
                    if r['retained'] and r['member'] not in seen:z.writestr(r['member'],data[r['path']]);seen.add(r['member'])
            # Strictly bounded temporary-fixture deletion, never actual runs.
            for r in (old,new):
                self.assertTrue(r.resolve().is_relative_to(root.resolve()));self.assertNotEqual(r.resolve(),root.resolve());shutil.rmtree(r)
            self.assertFalse(old.exists());self.assertFalse(new.exists())
            with a.Resolver(zp,mp) as view:
                a.bank_structure(view,proof,cert)
                auditor,transformations,checked=a.archived_auditors(view,cert,m)
                runtime=a.metric.prepare_metric_runtime()
                with a.metric.only_zip_workspace_reads(view,runtime):
                    actual=auditor(a.virtual_paths(view)(new),proof)
                    outputs=a.reconstruct(view,m,proof,created_utc=expected['summary.json']['created_utc'],raw_audit=actual)
                def first_difference(x,y,path='$'):
                    if type(x) is not type(y):return (path,type(x).__name__,type(y).__name__)
                    if isinstance(x,dict):
                        if x.keys()!=y.keys():return (path,sorted(x),sorted(y))
                        for k in x:
                            result=first_difference(x[k],y[k],path+'.'+k)
                            if result:return result
                    elif isinstance(x,list):
                        if len(x)!=len(y):return (path,len(x),len(y))
                        for i,(u,v) in enumerate(zip(x,y)):
                            result=first_difference(u,v,path+'['+str(i)+']')
                            if result:return result
                    elif x!=y:return (path,x,y)
                    return None
                self.assertIsNone(first_difference(outputs,expected))
                self.assertEqual(actual['parent_producers'],10)
                self.assertEqual(actual['native']['prefix_conversations'],10)
                self.assertEqual(outputs['summary.json']['effective']['reader_valid'],1540 if self.FULL_COHORT else 10)
                self.assertEqual(sum(k=='documents' for k,_ in checked),10)
                self.assertEqual(sum(k=='index' for k,_ in checked),30)
                self.assertEqual(len(transformations['exact_substitutions']),4)
                p=next(r['path'] for r in records if not r['retained'])
                with self.assertRaises(ValueError):view.sha(p)
                # No certificate may change state/document membership or raw
                # vector normalization to make the former bank appear valid.
                tampered=deepcopy(cert);native=next(iter(tampered['banks']))
                ep=next(iter(tampered['banks'][native]['embeddings']))
                tampered['banks'][native]['embeddings'][ep]['normalized_vectors_sha256']='0'*64
                bad,_,_=a.archived_auditors(view,tampered,m)
                with a.metric.only_zip_workspace_reads(view,runtime):
                    with self.assertRaisesRegex(ValueError,'normalized native embedding digest'):bad(a.virtual_paths(view)(new),proof)


class FullCohortArchive(FullArchive):
    FULL_COHORT=True


class OriginalParentArchive(unittest.TestCase):
    def test_actual_immutable_ten_parent_producers_from_temporary_zip_without_live_parent_reads(self):
        # Read only the already independently qualified 380 immutable parent
        # files plus its fixed code/data. This is a disposable test ZIP, never a
        # production archive/plan, and no actual run file is removed or written.
        pp=a.ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v2/terminal_admission_v1.json'
        proof=read(pp);self.assertEqual(len(proof['original_inventory']),380)
        wanted={**proof['original_inventory'],**{str(a.ROOT/p):h for p,h in proof['source_code_sha256'].items()}}
        for p in (a.ROOT/'data/locomo10.json',a.ROOT/a.terminal.AUDIT_SOURCE):wanted[str(p)]=sha(p)
        parent=read(a.OLD/'run_manifest.json');technical=read(a.OLD/'technical_dispatch_v6/manifest.json')
        for v in (parent['review'],technical['review']):wanted[v['path']]=v['sha256']
        from tools.check_experiment_health import shared_read
        data={p:shared_read(Path(p)) for p in wanted}
        self.assertTrue(all(hashlib.sha256(data[p]).hexdigest()==h for p,h in wanted.items()))
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            root=Path(td);mp=root/'manifest.json';zp=root/'evidence.zip'
            records=[{'path':p,'sha256':h,'size':len(data[p]),'retained':True,'member':'blobs/'+h} for p,h in sorted(wanted.items())]
            save_new(mp,{'schema':a.SCHEMA,'records':records})
            with zipfile.ZipFile(zp,'x') as z:
                z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
                for r in records:
                    if r['member'] not in seen:z.writestr(r['member'],data[r['path']]);seen.add(r['member'])
            with a.Resolver(zp,mp) as view:
                runtime=a.metric.prepare_metric_runtime()
                with a.metric.only_zip_workspace_reads(view,runtime):actual=a.audit_original(view,proof)
                self.assertEqual(actual['physical_producers'],10);self.assertEqual(actual['logical_consumers'],10)
                self.assertEqual(actual['source_inventory_sha256'],a.digest(proof['original_inventory']))
        self.assertTrue(all(sha(p)==h for p,h in proof['original_inventory'].items()))


class Dispatch(unittest.TestCase):
    def test_exact_verified_parent_and_all_deep_guards_reject_foreign_controller(self):
        command='tools/run_five_baselines_reviewed_chain_v7.py run --method vmem'
        parent={'ProcessId':9,'ParentProcessId':1,'Name':'python.exe','CommandLine':'"venv python" '+command,
            'ExecutablePath':str(a.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'),'CreatedUtc':'2026-10-09T00:00:00+00:00'}
        current={'ProcessId':10,'ParentProcessId':9,'Name':'python.exe','CommandLine':'"base python" '+command,
            'ExecutablePath':sys._base_executable,'CreatedUtc':'2026-10-09T00:00:01+00:00'}
        self.assertEqual(chain.active_controllers([parent,current],10),[])
        for changes in ({'ExecutablePath':'C:/wrong/python.exe'},{'CommandLine':'python '+command+' --different'}, {'CreatedUtc':'2026-10-09T00:00:02+00:00'}):
            bad={**parent,**changes};self.assertEqual(chain.active_controllers([bad,current],10),[bad])
        other={**parent,'ProcessId':22,'Name':'python3.11.exe','CommandLine':'python -m tools.run_five_baselines_unknown run'}
        self.assertEqual(chain.active_controllers([parent,current,other],10),[other])
        # No run-style OS fixture is started while Hyper production is active.
        # Verify the exact deep dynamic guard calls by providing a CIM result.
        current['ProcessId']=os.getpid();current['ParentProcessId']=9
        class Response:
            returncode=0
            stdout=json.dumps([parent,current,other])
        with chain.guard_context(),patch.object(subprocess,'run',return_value=Response()):
            for fn in (chain.assert_stopped,chain.previous.simple.stopped,chain.previous.previous.archive.old.engine.stopped,chain.previous.previous.frozen.lifecycle.stopped):
                with self.assertRaisesRegex(ValueError,'controller is active'):fn()

    def test_actual_prepare_check_main_to_ten_worker_sentinels_for_both_later_methods(self):
        from contextlib import redirect_stdout
        class BeforeProvider(BaseException):pass
        for method in ('vmem','prism'):
            with self.subTest(method=method),tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td,ExitStack() as stack:
                root=Path(td);run=root/'runs';refs={}
                for prior in ('amem','simplemem','hypermem')+ (('vmem',) if method=='prism' else ()):
                    refs[prior]={}
                    for kind in ('completion','receipt','plan'):
                        p=root/'accepted'/prior/(kind+'.json');save_new(p,{'fixture':prior,'kind':kind});refs[prior][kind]=p
                stack.enter_context(patch.object(chain.base,'RUN',run))
                stack.enter_context(patch.object(chain,'assert_stopped'))
                stack.enter_context(patch.object(chain,'completion_paths',return_value=refs))
                stack.enter_context(patch.object(chain,'strong_predecessors',side_effect=lambda m:{n:{k:chain.binding(p) for k,p in rr.items()} for n,rr in refs.items()}))
                priorpath=chain.ROOT/chain.foundation.PRIOR_REVIEW_ROOT/(method+'_preparation_review_v1.json')
                supporting=root/'supporting.json';save_new(supporting,{'fixture_only':True})
                review={**read(priorpath),'schema':chain.REVIEW_SCHEMA,'fixture_only':True,'production_launch_authorized':False,
                    'code_sha256':{**read(priorpath)['code_sha256'],**chain.code_hashes()},
                    'evidence_sha256':{str(priorpath):sha(priorpath),str(supporting):sha(supporting)}}
                rp=root/'review.json';save_new(rp,review)
                def main(*args):
                    with patch.object(sys,'argv',[chain.ENTRY,*args]),redirect_stdout(io.StringIO()):chain.main()
                main('prepare','--method',method,'--review',str(rp));main('check','--method',method)
                owner=run/method;seen=[]
                support_bytes=supporting.read_bytes()
                def mutate_during_predecessor_check(m):
                    supporting.write_text('{"changed":true}',encoding='utf8')
                    return {n:{k:chain.binding(p) for k,p in rr.items()} for n,rr in refs.items()}
                with patch.object(chain,'strong_predecessors',side_effect=mutate_during_predecessor_check):
                    with self.assertRaisesRegex(ValueError,'Preparation evidence changed'):main('run','--method',method)
                self.assertEqual({p.name for p in owner.iterdir()},{'run_manifest.json'})
                supporting.write_bytes(support_bytes)
                def worker(folder,m,ci,qs,item):
                    self.assertTrue((owner/'execution_intent.json').exists())
                    ticket=read(chain.sidecar(method)/'execution_intent.json')['ticket']
                    self.assertEqual(ticket['method'],method)
                    self.assertEqual(set(ticket['predecessor_acceptance']['verified_objects']),set(refs))
                    self.assertNotIn('strong_predecessors',ticket)
                    self.assertFalse((owner/'work').exists());seen.append(ci)
                    raise BeforeProvider()
                with patch.object(chain.base,'load_data',return_value=[{}]*10),patch.object(chain.base,'conv_worker',worker),patch.object(chain.base.BaselineAPI,'__init__',side_effect=AssertionError('No provider')):
                    with self.assertRaises(BeforeProvider):main('run','--method',method)
                self.assertEqual(set(seen),set(range(10)))
                before={str(p):sha(p) for p in owner.rglob('*') if p.is_file()}
                with self.assertRaises(ValueError):main('run','--method',method)
                self.assertEqual(before,{str(p):sha(p) for p in owner.rglob('*') if p.is_file()})
                self.assertEqual(chain.base.adapter(method).__name__,'tools.baselines_'+method+'_v1')

    def test_prepare_requires_new_hyper_completion_before_writing(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            with patch.object(chain,'frozen_sources'),patch.object(chain,'check_review'),patch.object(chain.base,'paths',return_value=Path(td)/'vmem'),\
                    patch.object(chain,'strong_predecessors',side_effect=ValueError('missing actual Hyper completion')),patch.object(chain.base,'prepare') as prepare:
                with self.assertRaisesRegex(ValueError,'Hyper completion'):chain.prepare('vmem',Path(td)/'review.json')
                prepare.assert_not_called()

    def test_completion_route_has_dual_hyper_root_and_only_vmem_before_prism(self):
        for method in ('vmem','prism'):
            paths=chain.completion_paths(method)
            self.assertEqual(paths['hypermem']['completion'],a.OUTPUT/'predecessor_completion.json')
            self.assertEqual(set(paths),{'amem','simplemem','hypermem'}|({'vmem'} if method=='prism' else set()))
        with self.assertRaises(ValueError):chain.completion_paths('hypermem')

    def test_guard_context_restores_frozen_globals_on_error(self):
        before=chain.guard.assert_stopped;previous=chain.previous.simple._stop
        with self.assertRaises(RuntimeError):
            with chain.guard_context():
                self.assertIs(chain.guard.assert_stopped,chain.assert_stopped)
                self.assertIs(chain.previous.simple._stop,chain.assert_stopped)
                raise RuntimeError('fixture')
        self.assertIs(chain.guard.assert_stopped,before);self.assertIs(chain.previous.simple._stop,previous)





class PowerShellGuards(unittest.TestCase):
    def test_actual_powershell_reparse_original_and_final_ancestor_rejected(self):
        actual=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=actual/'research') as directory:
            root=Path(directory);target=root/'target';target.mkdir();(target/'evidence.json').write_text('{}','utf8')
            script="""$ErrorActionPreference='Stop'
$workspace=WORKSPACE
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile(SOURCE,[ref]$tokens,[ref]$errors)
foreach($fn in $ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true)) {
    if($fn.Name -in @('AssertLiteralChain','AssertUnderWorkspace','AssertStopped')) { Invoke-Expression $fn.Extent.Text }
}
$junction=Join-Path $workspace 'alias'
New-Item -ItemType Junction -Path $junction -Target (Join-Path $workspace 'target') | Out-Null
try {
    $rejected=$false
    try { AssertUnderWorkspace (Join-Path $junction 'evidence.json') | Out-Null } catch { if($_ -match 'Reparse') {$rejected=$true} else {throw} }
    if(-not $rejected) {throw 'Original alias chain was accepted'}
    $rejected=$false
    try { AssertUnderWorkspace $junction | Out-Null } catch { if($_ -match 'Reparse') {$rejected=$true} else {throw} }
    if(-not $rejected) {throw 'Target junction accepted'}
    AssertUnderWorkspace (Join-Path $workspace 'target\evidence.json') | Out-Null
    function Get-CimInstance { [PSCustomObject]@{Name='python.exe';CommandLine='python tools/run_five_baselines_hypermem_native_v2.py run'} }
    $rejected=$false
    try {AssertStopped} catch {if($_ -match 'active') {$rejected=$true} else {throw}}
    if(-not $rejected) {throw 'Active controller accepted'}
    Write-Output 'LITERAL_REPARSE_AND_ACTIVE_REJECT_PASSED'
} finally { [IO.Directory]::Delete($junction) }
"""
            script=script.replace('WORKSPACE',"'"+str(root)+"'").replace('SOURCE',"'"+str(actual/'tools/cleanup_hypermem_native_v2.ps1')+"'")
            p=root/'guard_fixture.ps1';p.write_text(script,'utf8')
            r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(p)],capture_output=True,text=True,timeout=40)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('LITERAL_REPARSE_AND_ACTIVE_REJECT_PASSED',r.stdout)
            self.assertTrue((target/'evidence.json').exists())

class Packaging(unittest.TestCase):
    """Packaging/control fixture. Full scientific replay is tested separately."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);run=self.root/'runs/five_baselines_gemini25_v1_20261008'
        self.old=run/'hypermem';self.new=run/'hypermem_native_v2'
        self.out=self.root/'research/baseline_execution_2026-10-08/hypermem_native_v2_lifecycle'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for k,v in {'ROOT':self.root,'OLD':self.old,'NEW':self.new,'OUTPUT':self.out,'DEFAULT_PLAN':self.out/'cleanup_plan.json'}.items():self.stack.enter_context(patch.object(a,k,v))
        def write(path,value):save_new(path,value);return path
        for r in (self.old,self.new):
            write(r/'run_manifest.json',{'fixture_only':True})
            write(r/'results/summary.json',{'fixture_only':True})
            write(r/'work/api_method/api_cache/embedding.json',{'request':{'operation':'batchEmbedContents'},'response':{'embeddings':[]}})
            p=r/'work/native/c0/cache/hypergraph_embedding_index_conv_0.pkl';p.parent.mkdir(parents=True);p.write_bytes(b'disposable vector bytes')
        for p in (a.metric.METRIC,'data/locomo10.json','configs/sdk20_full1540_locked_v1.json'):write(self.root/p,{'fixture_only':True})
        typed_source=Path(__file__).resolve().parents[1]/'tools/baseline_evidence_closure_v1.py'
        target=self.root/'tools/baseline_evidence_closure_v1.py';target.parent.mkdir();target.write_bytes(typed_source.read_bytes())
        self.source=write(self.root/'fixture_source.json',{'fixture_only':True})
        self.codes={'fixture_source.json':sha(self.source)}
        self.stack.enter_context(patch.object(a,'code_hashes',return_value=self.codes))
        self.stack.enter_context(patch.object(a,'stopped'))
        self.stack.enter_context(patch.object(a,'evidence_closure',side_effect=lambda initial,**kw:a.simple.typed.collect(initial,root=a.ROOT,safe_path=a.safe,read=read,sha=sha,**kw)))
        self.inv=a.inventories();self.result=write(self.root/'result_review.json',{'fixture_only':True})
        self.tool=write(self.root/'tool_review.json',{'schema':a.REVIEW_SCHEMA,'status':'passed','scope':'two_root_lifecycle_with_explicit_predelete_bank_certificate',
            'code_sha256':self.codes,'evidence_sha256':{str(self.source):sha(self.source)}})
        self.cert=write(self.out/'bank_validation.json',{'fixture_only':True})
        self.bank=write(self.root/'bank_review.json',{'certificate':a.binding(self.cert),'fixture_only':True})
        self.stack.enter_context(patch.object(a,'result_review',return_value=({}, {}, {}, {})))
        self.stack.enter_context(patch.object(a,'verify_certificate',return_value={'run_inventories':self.inv}))
        self.replay={'status':'passed','scope':'SYNTHETIC PACKAGING CONTROL ONLY, not scientific replay','runtime':{}}
        self.stack.enter_context(patch.object(a,'strict_replay',return_value=self.replay))

    def do_archive(self):return a.archive(self.result,self.tool,self.bank)

    def test_real_zip_two_root_package_and_postdelete_completion_gate(self):
        self.do_archive();plan=a.verify_archive()
        with a.Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as view:
            for r in (self.old,self.new):
                self.assertEqual(view.read(r/'work/api_method/api_cache/embedding.json')['request']['operation'],'batchEmbedContents')
                with self.assertRaises(ValueError):view.sha(r/'work/native/c0/cache/hypergraph_embedding_index_conv_0.pkl')
        review=self.root/'cleanup_review.json';save_new(review,{'schema':'hypermem-native-v2-cleanup-review-v1','status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),
            'archive_sha256':plan['archive']['sha256'],'scope':a.SCOPE,'allowed_delete_roots':plan['allowed_delete_roots']})
        with self.assertRaises((FileNotFoundError,ValueError)):a.complete(review)
        rb=a.binding(review)
        save_new(self.out/'cleanup_execution_intent.json',{'schema':'hypermem-native-v2-cleanup-intent-v1','targets':plan['allowed_delete_roots'],
            'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        for name,r in (('original',self.old),('continuation',self.new)):
            self.assertTrue((r/'work').resolve().is_relative_to(self.root.resolve()));shutil.rmtree(r/'work')
            save_new(self.out/('removal_'+name+'.json'),{'schema':'hypermem-native-v2-removal-receipt-v1','removed_path':str(r/'work'),
                'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        a.complete(review);self.assertEqual(a.verify_completion()['status'],'passed')
        (self.new/'work').mkdir()
        with self.assertRaisesRegex(ValueError,'remains'):a.verify_completion()

    @unittest.skipUnless(os.name=='nt','Native Windows PowerShell fixture')
    def test_real_powershell_exact_two_targets_and_completion_failure_no_aggregate(self):
        self.do_archive();plan=read(a.DEFAULT_PLAN)
        review=self.root/'cleanup_review.json';save_new(review,{'schema':'hypermem-native-v2-cleanup-review-v1','status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),
            'archive_sha256':plan['archive']['sha256'],'scope':a.SCOPE,'allowed_delete_roots':plan['allowed_delete_roots']})
        bridge=self.root/'fixture_verify.py'
        bridge.write_text('import sys\n# Explicitly fake verifier for PowerShell control-flow test only.\nraise SystemExit(0 if sys.argv[1]=="verify" else 9)\n','utf8')
        actual=Path(__file__).resolve().parents[1]
        script=(actual/'tools/cleanup_hypermem_native_v2.ps1').read_text()
        script=script.replace("'D:\\jev mem'",repr(str(self.root)))
        script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'","$python = '"+sys.executable+"'")
        script=script.replace("$helper = Join-Path $workspace 'tools\\archive_hypermem_native_v2.py'","$helper = '"+str(bridge)+"'")
        # Scope local fixture only; do not query or affect the live controller.
        script=script.replace("$ErrorActionPreference = 'Stop'","$ErrorActionPreference = 'Stop'\nfunction Get-CimInstance { @() }")
        ps=self.root/'fixture_cleanup.ps1';ps.write_text(script,'utf8')
        before=sha(a.DEFAULT_PLAN)
        done=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ps),'-CleanupReview',str(review)],capture_output=True,text=True,timeout=50)
        self.assertNotEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertIn('Postdelete proof failed',done.stderr)
        self.assertEqual(sha(a.DEFAULT_PLAN),before)
        self.assertFalse((self.old/'work').exists());self.assertFalse((self.new/'work').exists())
        self.assertTrue((self.out/'removal_original.json').is_file());self.assertTrue((self.out/'removal_continuation.json').is_file())
        self.assertFalse((self.out/'cleanup_receipt.json').exists());self.assertFalse((self.out/'predecessor_completion.json').exists())

if __name__=='__main__':unittest.main()
