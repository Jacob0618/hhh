"""Temporary-only lifecycle fixtures. No production prepare/archive/deletion."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_simplemem_native_v2 as a
from tools import run_five_baselines_reviewed_chain_v5 as chain
from tools.baselines_common_v1 import read,sha,save_new

class Files(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.old=self.root/'old';self.new=self.root/'new'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for k,v in {'OLD':self.old,'NEW':self.new}.items():self.stack.enter_context(patch.object(a,k,v))
    def write(self,p,data=b'example'):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
    def pack(self,paths):
        records=[]
        for p in paths:
            data=p.read_bytes();h=hashlib.sha256(data).hexdigest();reason=a.disposition(p)
            r={'path':str(p),'sha256':h,'size':len(data),'retained':reason is None}
            if reason:r['disposal_reason']=reason
            else:r['member']='blobs/'+h
            records.append(r)
        mp=self.root/'manifest.json';zpath=self.root/'evidence.zip'
        save_new(mp,{'schema':a.SCHEMA,'records':records})
        with zipfile.ZipFile(zpath,'x') as z:
            z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
            for r in records:
                if r['retained'] and r['member'] not in seen:z.writestr(r['member'],Path(r['path']).read_bytes());seen.add(r['member'])
        return zpath,mp

    def test_zip_only_disposed_bytes_never_exist_or_return_sha(self):
        source=self.write(self.old/'work/api_method/api_cache/same.json',b'{"first":1}')
        other=self.write(self.new/'work/api_method/api_cache/same.json',b'{"second":2}')
        db=self.write(self.new/'work/native/c0/simplemem/lancedb/vector.lance')
        z,mp=self.pack([source,other,db])
        for root in (self.old,self.new):
            self.assertTrue(root.resolve().is_relative_to(self.root.resolve()));shutil.rmtree(root)
        with a.Resolver(z,mp) as view:
            self.assertNotEqual(view.bytes(source),view.bytes(other))
            self.assertFalse(view._simplemem_path_type(db).exists())
            with self.assertRaisesRegex(ValueError,'unavailable'):view.sha(db)
            with self.assertRaisesRegex(ValueError,'unavailable'):view.bytes(self.root/'unknown.json')
            paths=list(view._simplemem_path_type(self.old/'work').rglob('*.json'))
            self.assertEqual(paths,[view._simplemem_path_type(source)])

    def test_unknown_numeric_artifact_and_external_research_db_are_not_retained(self):
        with self.assertRaisesRegex(ValueError,'Unclassified'):a.disposition(self.new/'work/native/c0/simplemem/vector.npy')
        with self.assertRaisesRegex(ValueError,'External runnable'):a.disposition(self.root/'research/native/lancedb/1.lance')
        self.assertIsNone(a.disposition(self.new/'work/api_method/api_cache/embedding.json'))
        self.assertIsNone(a.disposition(self.new/'work/prefix/c0_q0.jsonl'))

    def test_only_exact_pinned_fts_python_source_is_retained(self):
        p=a.ROOT/a.LANCEDB_FTS_SOURCE
        self.assertEqual(sha(p),a.LANCEDB_FTS_SOURCE_SHA)
        self.assertIsNone(a.disposition(p))
        with patch.object(a,'sha',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'Python source changed'):a.disposition(p)
        for other in (p.with_name('other.py'),self.root/'research/native/lancedb/fts.py',self.root/'research/native/lancedb/1.lance'):
            with self.assertRaisesRegex(ValueError,'External runnable'):a.disposition(other)

    def test_crc_valid_tamper_and_duplicate_logical_path_are_rejected(self):
        p=self.write(self.old/'work/native/c0/simplemem/memory.json');z,mp=self.pack([p])
        v=read(mp);v['records'].append(dict(v['records'][0]));mp.write_text(json.dumps(v),'utf8')
        with zipfile.ZipFile(z,'w') as f:
            f.writestr('archive_manifest.json',mp.read_bytes());f.writestr(v['records'][0]['member'],p.read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.Resolver(z,mp)

    def test_private_transform_is_exact_and_declares_bank_semantics(self):
        text=a.function_source(a.seal.LiveView(),'tools/run_five_baselines_simplemem_native_v2.py','audit_sidecar')
        self.assertEqual(text.count(a.DB_BEFORE),1)
        _,report,_=a.archived_auditors(a.seal.LiveView(),{'banks':{}})
        self.assertEqual(report['changed_method_lines'],a.DB_BEFORE.splitlines())
        self.assertIn('no DB search replay',report['scope'])
        with patch.object(a,'DB_BEFORE',a.DB_BEFORE+' '):
            with self.assertRaisesRegex(ValueError,'boundary differs'):a.archived_auditors(a.seal.LiveView(),{'banks':{}})

    def test_two_roots_only_and_partial_cleanup_remains_blocked(self):
        out=self.root/'output';out.mkdir()
        planfile=out/'plan.json';save_new(planfile,{})
        plan={'allowed_delete_roots':[str(self.old/'work'),str(self.new/'work')]};review={'sha256':'a'*64}
        save_new(out/'cleanup_execution_intent.json',{'schema':'simplemem-native-v2-cleanup-intent-v1','targets':plan['allowed_delete_roots'],
            'cleanup_plan_sha256':sha(planfile),'cleanup_review_sha256':review['sha256']})
        (self.new/'work').mkdir(parents=True)
        save_new(out/'removal_original.json',{'schema':'simplemem-native-v2-removal-receipt-v1','removed_path':str(self.old/'work'),
            'cleanup_plan_sha256':sha(planfile),'cleanup_review_sha256':review['sha256']})
        with patch.object(a,'OUTPUT',out),patch.object(a,'DEFAULT_PLAN',planfile):
            with self.assertRaisesRegex(ValueError,'remains'):a.removal_evidence(plan,review)

    def test_certificate_complete_bank_inventory_cannot_be_self_changed(self):
        import copy
        old=self.old/'work/native/c0/simplemem';new=self.new/'work/native/c0/simplemem'
        oldfile=str(old/'lancedb/1.lance');newfile=str(new/'lancedb/1.lance');h='a'*64;state='b'*64
        proof={'banks':{'0':{'bank_dir':str(old),'lancedb_inventory':{oldfile:h},'database_integrity':{'rows':1}}}}
        cert={'run_inventories':{'original':{oldfile:{'sha256':h,'size':10}},'continuation':{newfile:{'sha256':h,'size':10}}},'banks':{}}
        for owner,folder,p in [('original',old,oldfile),('continuation',new,newfile)]:
            cert['banks'][str(folder)]={'owner':owner,'database_files':{p:{'sha256':h,'size':10}},'original_bank':str(old),
                'original_database_inventory':{oldfile:h},'database_integrity':{'rows':1},'state':{'path':str(folder/'state.json'),'sha256':state}}
        view=type('StateOnlyView',(),{'sha':lambda self,p:state})()
        a.bank_structure(view,proof,cert)
        bad=copy.deepcopy(cert);bad['run_inventories']['continuation'][newfile]['sha256']='c'*64
        with self.assertRaises(ValueError):a.bank_structure(view,proof,bad)
        bad=copy.deepcopy(cert);bad['banks'][str(new)]['state']['sha256']='c'*64
        with self.assertRaises(ValueError):a.bank_structure(view,proof,bad)
        bad=copy.deepcopy(cert);bad['banks']['extra']=bad['banks'][str(new)]
        with self.assertRaises(ValueError):a.bank_structure(view,proof,bad)

    def test_summary_only_or_foreign_inventory_result_review_rejected(self):
        path=self.write(self.root/'result.json',json.dumps({'schema':a.RESULT_SCHEMA,'status':'passed','method':'simplemem'}).encode())
        with self.assertRaisesRegex(ValueError,'Both full'):a.result_review(path)
        v=read(path);v['reviewed_run_inventories']={'original':{str(self.root/'foreign.json'):'a'*64},'continuation':{str(self.new/'file.json'):'b'*64}}
        path.write_text(json.dumps(v),'utf8')
        with self.assertRaisesRegex(ValueError,'inventory path'):a.result_review(path)

class Dispatch(unittest.TestCase):
    def test_prepare_missing_simple_completion_never_calls_base_prepare(self):
        with patch.object(chain.simple,'stopped'),patch.object(chain.previous.archive,'verify_completion',return_value={'status':'passed'}), \
             patch.object(chain.simple,'verify_completion',side_effect=ValueError('Simple incomplete')), \
             patch.object(chain.previous.frozen.frozen.base,'prepare') as prepare, \
             patch('sys.argv',['run_five_baselines_reviewed_chain_v5.py','prepare','--method','hypermem','--review','fixture.json']):
            with self.assertRaisesRegex(ValueError,'Simple incomplete'):chain.main()
            prepare.assert_not_called()
    def test_predecessors_route_simple_union_and_all_prior_methods(self):
        with patch.object(chain.simple,'stopped'),patch.object(chain.previous.archive,'verify_completion',return_value={'status':'passed'}) as amem, \
             patch.object(chain.simple,'verify_completion',return_value={'status':'passed'}) as simple, \
             patch.object(chain.previous.frozen.lifecycle,'verify_completion',return_value={'status':'passed'}) as other:
            chain.verify_predecessors('prism');amem.assert_called_once();simple.assert_called_once()
            self.assertEqual([x.args[0] for x in other.call_args_list],['hypermem','vmem'])
    def test_private_context_restores_on_exception(self):
        old={k:getattr(chain.previous,k) for k in ('active_controllers','assert_stopped','verify_code_review','verify_predecessors')}
        with self.assertRaises(RuntimeError):
            with chain.private_context():raise RuntimeError('fixture')
        self.assertTrue(all(getattr(chain.previous,k) is v for k,v in old.items()))

    @unittest.skipUnless(os.name=='nt','Actual venv parent identity fixture')
    def test_actual_venv_launcher_parent_only(self):
        actual=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=actual/'research') as directory:
            p=Path(directory)/'run_five_baselines_reviewed_chain_v5.py'
            p.write_text('import sys\nsys.path.insert(0,'+repr(str(actual))+')\nfrom tools import archive_simplemem_native_v2 as a\na.stopped()\nprint("GATE_ONLY_PASSED",flush=True)\n','utf8')
            r=subprocess.run([str(actual/'runtime/baseline_env_v1/Scripts/python.exe'),str(p),'run'],capture_output=True,text=True,timeout=40,creationflags=0x08000000)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'GATE_ONLY_PASSED')

    @unittest.skipUnless(os.name=='nt','Actual PowerShell literal/reparse check')
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
    function Get-CimInstance { [PSCustomObject]@{Name='python.exe';CommandLine='python tools/run_five_baselines_simplemem_native_v2.py run'} }
    $rejected=$false
    try {AssertStopped} catch {if($_ -match 'active') {$rejected=$true} else {throw}}
    if(-not $rejected) {throw 'Active controller accepted'}
    Write-Output 'LITERAL_REPARSE_AND_ACTIVE_REJECT_PASSED'
} finally { [IO.Directory]::Delete($junction) }
"""
            script=script.replace('WORKSPACE',"'"+str(root)+"'").replace('SOURCE',"'"+str(actual/'tools/cleanup_simplemem_native_v2.ps1')+"'")
            p=root/'guard_fixture.ps1';p.write_text(script,'utf8')
            r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(p)],capture_output=True,text=True,timeout=40)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('LITERAL_REPARSE_AND_ACTIVE_REJECT_PASSED',r.stdout)
            self.assertTrue((target/'evidence.json').exists())

class Packaging(unittest.TestCase):
    """Packaging/control fixture. Full scientific replay is tested separately."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);run=self.root/'runs/five_baselines_gemini25_v1_20261008'
        self.old=run/'simplemem';self.new=run/'simplemem_native_v2'
        self.out=self.root/'research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for k,v in {'ROOT':self.root,'OLD':self.old,'NEW':self.new,'OUTPUT':self.out,'DEFAULT_PLAN':self.out/'cleanup_plan.json'}.items():self.stack.enter_context(patch.object(a,k,v))
        def write(path,value):save_new(path,value);return path
        for r in (self.old,self.new):
            write(r/'run_manifest.json',{'fixture_only':True})
            write(r/'results/summary.json',{'fixture_only':True})
            write(r/'work/api_method/api_cache/embedding.json',{'request':{'operation':'batchEmbedContents'},'response':{'embeddings':[]}})
            p=r/'work/native/c0/simplemem/lancedb/vector.lance';p.parent.mkdir(parents=True);p.write_bytes(b'disposable vector bytes')
        for p in (a.metric.METRIC,'data/locomo10.json','configs/sdk20_full1540_locked_v1.json'):write(self.root/p,{'fixture_only':True})
        typed_source=Path(__file__).resolve().parents[1]/'tools/baseline_evidence_closure_v1.py'
        target=self.root/'tools/baseline_evidence_closure_v1.py';target.parent.mkdir();target.write_bytes(typed_source.read_bytes())
        self.source=write(self.root/'fixture_source.json',{'fixture_only':True})
        self.codes={'fixture_source.json':sha(self.source)}
        self.stack.enter_context(patch.object(a,'code_hashes',return_value=self.codes))
        self.stack.enter_context(patch.object(a,'stopped'))
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
                with self.assertRaises(ValueError):view.sha(r/'work/native/c0/simplemem/lancedb/vector.lance')
        review=self.root/'cleanup_review.json';save_new(review,{'schema':'simplemem-native-v2-cleanup-review-v1','status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),
            'archive_sha256':plan['archive']['sha256'],'scope':a.SCOPE,'allowed_delete_roots':plan['allowed_delete_roots']})
        with self.assertRaises((FileNotFoundError,ValueError)):a.complete(review)
        rb=a.binding(review)
        save_new(self.out/'cleanup_execution_intent.json',{'schema':'simplemem-native-v2-cleanup-intent-v1','targets':plan['allowed_delete_roots'],
            'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        for name,r in (('original',self.old),('continuation',self.new)):
            self.assertTrue((r/'work').resolve().is_relative_to(self.root.resolve()));shutil.rmtree(r/'work')
            save_new(self.out/('removal_'+name+'.json'),{'schema':'simplemem-native-v2-removal-receipt-v1','removed_path':str(r/'work'),
                'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        a.complete(review);self.assertEqual(a.verify_completion()['status'],'passed')
        (self.new/'work').mkdir()
        with self.assertRaisesRegex(ValueError,'remains'):a.verify_completion()

    @unittest.skipUnless(os.name=='nt','Native Windows PowerShell fixture')
    def test_real_powershell_exact_two_targets_and_completion_failure_no_aggregate(self):
        self.do_archive();plan=read(a.DEFAULT_PLAN)
        review=self.root/'cleanup_review.json';save_new(review,{'schema':'simplemem-native-v2-cleanup-review-v1','status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),
            'archive_sha256':plan['archive']['sha256'],'scope':a.SCOPE,'allowed_delete_roots':plan['allowed_delete_roots']})
        bridge=self.root/'fixture_verify.py'
        bridge.write_text('import sys\n# Explicitly fake verifier for PowerShell control-flow test only.\nraise SystemExit(0 if sys.argv[1]=="verify" else 9)\n','utf8')
        actual=Path(__file__).resolve().parents[1]
        script=(actual/'tools/cleanup_simplemem_native_v2.ps1').read_text()
        script=script.replace("'D:\\jev mem'",repr(str(self.root)))
        script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'","$python = '"+sys.executable+"'")
        script=script.replace("$helper = Join-Path $workspace 'tools\\archive_simplemem_native_v2.py'","$helper = '"+str(bridge)+"'")
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

class ActualNativeArchive(unittest.TestCase):
    def full_union(self,old,new,m,proof):
        """1540 synthetic QAs with real frozen prompts/parsers/metrics/audits.

        One resumed native query is real from the temporary Tantivy fixture;
        1539 original retrieval contexts are synthetic repeats. No claim that
        these model answers or 1540 native searches are a measured experiment.
        """
        from jev_mem.gemini25_ablation_provider_v1 import generation_body
        from jev_mem.sdk20_cached_api import digest
        from tools.baselines_common_v1 import append
        base=a.core.base;metric,_=base.verified_official_metric()
        qs=[{'query_id':'c0_q'+str(i),'conversation':0,'source_qa_index':i,'question':'What about Alice?' if i==0 else 'Where did A walk?',
            'reference':'park','category':1,'reference_date':'2023-01-01'} for i in range(1540)]
        parent=old/'run_manifest.json';save_new(parent,{'questions':qs})
        proof['questions']=qs
        Path(m['proof']['path']).write_text(json.dumps(proof),'utf8');m['proof']=a.binding(m['proof']['path'])
        m['parent_manifest']=a.binding(parent);m['questions']=qs
        retrieval=read(new/'results/c0/retrieval/c0_q0.json')
        stamp='2026-01-01T00:00:00+00:00'
        for i,q in enumerate(qs):
            owner=new if i==0 else old;leaf=owner/'results/firstpass'/q['query_id'];rp=owner/'results/c0/retrieval'/(q['query_id']+'.json')
            if i:save_new(rp,retrieval)
            context={'method':'simplemem','conversation':0,'query_id':q['query_id'],'phase':'firstpass'};folder=owner/'work/api_firstpass';events=[]
            class Recorder:
                def generate(self,prompt,*,model,system=None,json_output=False,max_output,stage):
                    sig={'provider':'gemini','model':model,'operation':'generateContent','body':generation_body(prompt,system=system,json_output=json_output,max_output=max_output)}
                    key=digest(sig);path=folder/'api_cache'/(key+'.json');fresh=not path.exists();text='park' if stage==base.evaluator.READER_STAGE else '[[yes]]'
                    raw={'request':sig,'stage':stage,'response':{'modelVersion':model,'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}]}}
                    if fresh:
                        save_new(path,raw);save_new(folder/'request_intents'/(key+'.json'),{'request_hash':key,'signature':sig,'context':context,'stage':stage,'retries':2,'created_utc':stamp})
                    event={'provider':'gemini','model':model,'operation':'generateContent','request_hash':key,'cache_path':str(path),'cache_sha256':sha(path),
                        'context':context,'stage':stage,'error_type':None,'source':'new_request' if fresh else 'existing_run_cache'}
                    events.append(event);append(folder/'logical_events.jsonl',{'recorded_utc':stamp,**event})
                    call={**{k:event[k] for k in ('provider','model','operation','request_hash','context','stage')},'timestamp_utc':stamp,'attempt':1,'status':200,'cache_hit':not fresh,'usage':{},'resolved_model':model}
                    if not fresh:call['usage_semantics']='reused_response_no_new_model_usage'
                    append(folder/'api_calls.jsonl',call);return text
            api=Recorder();r=base.evaluator.read_answer(api,q['question'],q['reference'],[{'id':'native_context','text':retrieval['value']['context'],'observation_date':None}],reference_date=q['reference_date']);r['official_f1']=metric(r['prediction']['answer'],q['reference'],q['category'])
            save_new(leaf/'reader.json',{'status':'complete','value':r,'events':[events[-1]]})
            j=base.evaluator.judge_answer(api,q['question'],q['reference'],r['prediction']['answer']);save_new(leaf/'judge.json',{'status':'complete','value':j,'events':[events[-1]]})
            for kind in ('reader','judge'):save_new(leaf/(kind+'.intent.json'),{'context':context})
            save_new(leaf/'outcome.json',{'query_id':q['query_id'],'conversation':0,'method':'simplemem','phase':'firstpass','reader':'complete','judge':'complete','reader_path':str(leaf/'reader.json'),'judge_path':str(leaf/'judge.json')})
        _,_,sources,jobs=a.seal.pre_recovery(m,proof,a.seal.LiveView());self.assertEqual(jobs,[])
        save_new(new/'firstpass_sources.json',sources)
        save_new(new/'qa_recovery_plan.json',{'round':1,'full_cohort':1540,'jobs':jobs,'firstpass_sources_sha256':sha(new/'firstpass_sources.json')})
        save_new(new/'qa_recovery_outcome.json',{'plan_sha256':sha(new/'qa_recovery_plan.json'),'results':[]})
        return m,proof

    def test_real_temporary_native_and_raw_replay_after_both_banks_removed(self):
        # Reuse the independently exercised real Tantivy/native/fake-transport
        # fixture. Intercept only its first successful audit, before its own
        # corruption negative test, and test our archive on those real artifacts.
        from tests.test_simplemem_native_v2_method_audit import MethodFixture
        import lancedb  # Initialize Windows event-loop self-pipe before network guard.
        controller=a.core;original=controller.audit_sidecar;captured={}
        class Finished(BaseException):pass
        def inspect(m,proof,**kwargs):
            expected=original(m,proof,**kwargs)
            old=Path(m['parent_root']);new=Path(m['self_root']);temp=old.parent
            cert={'banks':{}}
            for ci,b in proof['banks'].items():
                bank=new/'work/native'/('c'+ci)/'simplemem'
                cert['banks'][str(bank)]={'database_files':{str(p):{'sha256':sha(p),'size':p.stat().st_size} for p in (bank/'lancedb').rglob('*') if p.is_file()},
                    'original_database_inventory':b['lancedb_inventory'],'database_integrity':b['database_integrity'],
                    'state':{'path':str(bank/'state.json'),'sha256':sha(bank/'state.json')}}
            with patch.object(a,'OLD',old),patch.object(a,'NEW',new),patch.object(a.seal,'OLD',old),patch.object(a.seal,'NEW',new):
                m,proof=self.full_union(old,new,m,proof)
                with patch.object(controller,'audit_sidecar',original):
                    live_outputs=a.seal.reconstruct(m,proof,created_utc='fixture')
                for name,value in live_outputs.items():save_new(new/'results'/name,value)
                paths=[p for p in temp.rglob('*') if p.is_file()]
                paths += [a.ROOT/'tools/run_five_baselines_simplemem_native_v2.py',a.ROOT/'tools/seal_simplemem_native_v2.py']
                paths += [a.ROOT/'upstream/baselines_v1/simplemem/simplemem/core/answer_generator.py']
                paths += [a.ROOT/'jev_mem/locomo_official_metrics.py',a.ROOT/a.metric.METRIC]
                records=[]
                for p in paths:
                    reason=a.disposition(p);r={'path':str(p),'sha256':sha(p),'size':p.stat().st_size,'retained':reason is None}
                    if reason:r['disposal_reason']=reason
                    else:r['member']='blobs/'+r['sha256']
                    records.append(r)
                mp=temp/'archive_fixture.json';zpath=temp/'archive_fixture.zip';save_new(mp,{'schema':a.SCHEMA,'records':records})
                with zipfile.ZipFile(zpath,'x') as z:
                    z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
                    for r in records:
                        if r['retained'] and r['member'] not in seen:z.writestr(r['member'],Path(r['path']).read_bytes());seen.add(r['member'])
                for r in (old,new):
                    self.assertTrue(r.resolve().is_relative_to(temp.resolve()));shutil.rmtree(r)
                with a.Resolver(zpath,mp) as view:
                    reconstruct,transforms,calls=a.archived_auditors(view,cert)
                    auditor=reconstruct.__globals__['archival_controller'].audit_sidecar
                    runtime=a.metric.prepare_metric_runtime()
                    with a.metric.only_zip_workspace_reads(view,runtime):
                        actual=auditor(m,proof,view=view)
                        archive_outputs=reconstruct(m,proof,view=view,created_utc='fixture')
                    self.assertEqual(actual,expected);self.assertEqual(len(calls),2)
                    self.assertEqual(archive_outputs,live_outputs)
                    self.assertEqual(archive_outputs['summary.json']['effective']['reader_valid'],1540)
                    db=next(p for p in records if not p['retained'])
                    with self.assertRaises(ValueError):view.sha(db['path'])
                    captured.update(status='passed',native_result_equal=True,full_synthetic_union_rows=1540,bank_bytes_after_delete=0,transforms=transforms)
            raise Finished()
        with patch.object(controller,'audit_sidecar',inspect),patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
            with self.assertRaises(Finished):MethodFixture().test_real_native_transport_cache_and_full_text_audit()
        self.assertEqual(captured['status'],'passed')

if __name__=='__main__':unittest.main()
