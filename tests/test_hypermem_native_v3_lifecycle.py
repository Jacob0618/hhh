"""Offline, temporary-root lifecycle fixtures; no production operations.

Structural fixtures do not stand in for the later full native/1540 QA replay.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_hypermem_native_v3 as a
from tools import run_five_baselines_reviewed_chain_v8 as chain
from tools.baselines_common_v1 import read,sha,save_new


class Retention(unittest.TestCase):
    def test_three_exact_roots_and_no_parent_overlap(self):
        self.assertEqual(list(a.roots()),['original','parent','continuation'])
        self.assertEqual([p.parent.name for p in a.work_roots()],['hypermem','hypermem_native_v2','hypermem_native_v3'])
        for x in a.work_roots():
            for y in a.work_roots():
                if x!=y:self.assertFalse(x.is_relative_to(y))

    def test_retains_ancestral_raw_normalization_and_replacement(self):
        for root in a.roots().values():
            for relative in ('api_method/api_cache/same_key.json','api_method/request_intents/same_key.json',
                'native/c0/representations/topic_000035_hypermem_episode_roles.json',
                'replacement_v3/known_invalid_episode_once.json','prefix_v3/c0/requests.jsonl'):
                self.assertIsNone(a.disposition(root/'work'/relative))
            self.assertEqual(a.disposition(root/'work/native/c0/cache/hypergraph_embedding_index_conv_0.pkl'),'runnable_hypermem_bm25_vector_index')
            for relative in ('native/c0/cache/unknown.pkl','native/c0/calls/response.json.tmp'):
                with self.assertRaises(ValueError):a.disposition(root/'work'/relative)
        with self.assertRaises(ValueError):a.disposition(a.ROOT/'research/foreign/embedding_evidence'/('query_'+'a'*64+'.json'))

    def test_same_key_different_raw_retained_for_all_owners_and_no_fallback(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td,ExitStack() as stack:
            base=Path(td);owners={'OLD':base/'old','PARENT':base/'parent','NEW':base/'new'}
            for k,v in owners.items():stack.enter_context(patch.object(a,k,v))
            blobs={};records=[]
            for index,root in enumerate(owners.values()):
                p=root/'work/api_method/api_cache/same_key.json';p.parent.mkdir(parents=True)
                value={'owner':index,'raw':'exact original '+str(index)};p.write_text(json.dumps(value),encoding='utf8')
                data=p.read_bytes();h=hashlib.sha256(data).hexdigest();blobs[h]=data
                records.append({'path':str(p),'sha256':h,'size':len(data),'retained':True,'member':'blobs/'+h})
            mp=base/'manifest.json';zp=base/'evidence.zip';save_new(mp,{'schema':a.SCHEMA,'records':records})
            with zipfile.ZipFile(zp,'x') as z:
                z.writestr('archive_manifest.json',mp.read_bytes())
                for h,data in blobs.items():z.writestr('blobs/'+h,data)
            for root in owners.values():
                self.assertTrue(root.resolve().is_relative_to(base.resolve()));shutil.rmtree(root)
            with a.Resolver(zp,mp) as view:
                for index,root in enumerate(owners.values()):self.assertEqual(view.read(root/'work/api_method/api_cache/same_key.json')['owner'],index)
                with self.assertRaises(ValueError):view.read(base/'unrecorded.json')
            self.assertEqual(len(blobs),3)

    def test_resolver_rejects_marking_unique_old_raw_disposable(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            base=Path(td);mp=base/'manifest.json';zp=base/'evidence.zip'
            save_new(mp,{'schema':a.SCHEMA,'records':[{'path':str(a.PARENT/'work/api_method/api_cache/unique.json'),
                'sha256':'1'*64,'size':1,'retained':False,'disposal_reason':'runnable_hypermem_bm25_vector_index'}]})
            with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
            with self.assertRaisesRegex(ValueError,'retention'):a.Resolver(zp,mp)

    def test_missing_archive_or_bank_certificate_blocks_instead_of_faking_completion(self):
        with self.assertRaisesRegex(ValueError,'Exact archive view'):a.archived_auditors(None,None,None)


class Dispatch(unittest.TestCase):
    def test_only_later_methods_and_current_three_root_completion_route(self):
        self.assertEqual(chain.METHODS,('vmem','prism'))
        for method in chain.METHODS:
            pp=chain.completion_paths(method)
            self.assertEqual(pp['hypermem']['plan'],a.DEFAULT_PLAN)
            self.assertEqual(set(pp),{'amem','simplemem','hypermem'}|({'vmem'} if method=='prism' else set()))
        with self.assertRaises(ValueError):chain.completion_paths('hypermem')

    def test_missing_hyper_completion_blocks_fresh_method_before_prepare(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            with patch.object(chain,'frozen_sources'),patch.object(chain,'check_review'),patch.object(chain.base,'paths',return_value=Path(td)/'vmem'),\
                    patch.object(chain,'strong_predecessors',side_effect=ValueError('missing actual three-root completion')),patch.object(chain.base,'prepare') as prepare:
                with self.assertRaisesRegex(ValueError,'three-root'):chain.prepare('vmem',Path(td)/'review.json')
                prepare.assert_not_called()

    def test_attestation_only_hash_checks_reject_tampering_and_reappeared_work(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td,ExitStack() as stack:
            base=Path(td);run=base/'runs';paths={}
            for name in ('amem','simplemem'):
                paths[name]={}
                for kind in ('completion','receipt','plan'):
                    p=base/name/(kind+'.json');save_new(p,{'fixture':name,'kind':kind});paths[name][kind]=p
            verified={name:{k+'_sha256':sha(p) for k,p in pp.items()} for name,pp in paths.items()}
            ticket={'schema':'hypermem-native-v2-execution-intent-v1','manifest':{'sha256':'474a62558f4f3905b3704e717cae3ac93de61aaa683baf41ffc9d24b5a53ffd0'},
                'predecessor_acceptance':{'schema':'runtime-predecessor-acceptance-record-v1','new_scientific_input':False,'recursive_rearchive_claimed':False,'verified_objects':verified}}
            p=base/'ticket.json';save_new(p,ticket)
            stack.enter_context(patch.object(chain,'ATTESTATION',p));stack.enter_context(patch.object(chain,'ATTESTATION_SHA',sha(p)))
            stack.enter_context(patch.object(chain.foundation,'completion_paths',return_value=paths));stack.enter_context(patch.object(chain.base,'RUN',run))
            with patch.object(chain.previous.simple,'verify_completion',side_effect=AssertionError('No old ZIP replay')),\
                    patch.object(chain.previous.previous.archive,'verify_completion',side_effect=AssertionError('No old A replay')):
                self.assertEqual(set(chain.attested_predecessors()),{'amem','simplemem'})
            original=paths['amem']['completion'].read_bytes();paths['amem']['completion'].write_text('{}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'completion/receipt/plan changed'):chain.attested_predecessors()
            paths['amem']['completion'].write_bytes(original)
            (run/'simplemem_native_v2/work').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'reappeared'):chain.attested_predecessors()

    def test_actual_temporary_prepare_manifest_run_to_worker_sentinels_no_provider(self):
        from tests.test_hypermem_native_v2_lifecycle import Dispatch as Reviewed
        from types import FunctionType
        fn=Reviewed.test_actual_prepare_check_main_to_ten_worker_sentinels_for_both_later_methods
        isolated=FunctionType(fn.__code__,{**fn.__globals__,'a':a,'chain':chain},fn.__name__,fn.__defaults__)
        isolated(self)

    def test_exact_new_parent_shape_and_all_deep_guards_reject_foreign_controller(self):
        from tests.test_hypermem_native_v2_lifecycle import Dispatch as Reviewed
        import inspect,ast
        fn=Reviewed.test_exact_verified_parent_and_all_deep_guards_reject_foreign_controller
        source=inspect.getsource(fn).replace('run_five_baselines_reviewed_chain_v7','run_five_baselines_reviewed_chain_v8')
        node=ast.parse('class Reused:\n'+source).body[0].body[0]
        ns={**fn.__globals__,'a':a,'chain':chain};exec(compile(ast.Module(body=[node],type_ignores=[]),'<new-dispatch-exact-process-dictionary-fixture>','exec'),ns)
        ns[fn.__name__](self)


class FullNativeArchive(unittest.TestCase):
    """Original native/QA science with a synthetic 1540-slot dataset.

    The historical production cohort is audited separately. This fixture uses
    genuine tiny native banks, actual prefix API, UUID journals, one malformed
    JSON replacement and one uniform Judge recovery, with fake HTTP only.
    """
    SLOT_COUNT=1540
    def test_1540_native_raw_and_all_four_outputs_from_zip_after_three_fixture_roots_removed(self):
        from tests import test_hypermem_native_v3_full_audit as factory
        c=factory.c;original=c.private_controller;captured={};original_audit=factory.audit.audit_all
        class Finished(BaseException):pass
        def controller(m,proof,ticket):
            q=proof['questions'][1]
            additional=[{**q,'query_id':'c1_fixture'+str(i),'source_qa_index':1000+i} for i in range(self.SLOT_COUNT-10)]
            proof['questions'].extend(additional)
            # Before any real API object/marker exists, lock this synthetic
            # cohort and the actual source used by its scientific functions.
            m['code_sha256']=a.code_hashes()
            m['upstream_sha256']=c.chain.base.source_inventory('hypermem')
            pp=Path(m['terminal_proof']['path']);pp.write_text(json.dumps(proof),encoding='utf8')
            m['terminal_proof']=a.binding(pp);m['review']=a.binding(pp)
            (c.NEW/'run_manifest.json').write_text(json.dumps(m),encoding='utf8')
            value=original(m,proof,ticket)
            # Author fixture patches admission owners; summary reads its
            # captured top-level owner globals, so route them to those roots.
            value.save_new=a.frozen.shared_replay.clone(value.save_new,OLD=c.admission.OLD,ANCESTOR=c.admission.ANCESTOR)
            original_seal=value.seal
            def expand():
                state=read(c.NEW/'work/native/c1/state.json')
                for q in additional:
                    api=value.BaselineAPI(c.NEW/'work/api_method',context={'method':'hypermem','conversation':1,'phase':'retrieval','query_id':q['query_id']})
                    ret=value.capture_stage(c.NEW/'results/c1/retrieval'/(q['query_id']+'.json'),lambda:value.adapter('hypermem').retrieve(q['question'],state,api),api)
                    self.assertEqual(ret['status'],'complete')
                    self.assertEqual(value.qa_one(c.NEW,'hypermem',q,ret['value'])['judge'],'complete')
            captured['expand']=expand
            def seal(method):
                original_seal(method)
                with patch.object(factory.audit,'audit_all',original_audit):
                    self.archive_fixture(c.admission.ANCESTOR,c.admission.OLD,c.NEW,m,proof,c.admission.chain.base.load_data())
                captured['passed']=True;raise Finished()
            value.seal=seal;return value
        def first_audit(*args,**kwargs):
            expand=captured.pop('expand',None)
            if expand:expand()
            return original_audit(*args,**kwargs)
        with patch.object(c,'private_controller',controller),patch.object(factory.audit,'audit_all',first_audit):
            with self.assertRaises(Finished):factory.CompleteAudit().test_real_prefix_representation_replacement_native_qa_uniform_recovery_and_seal()
        self.assertTrue(captured['passed'])

    def archive_fixture(self,ancestor,parent,new,m,proof,dataset):
        base=ancestor.parent
        with patch.object(a,'OLD',ancestor),patch.object(a,'PARENT',parent),patch.object(a,'NEW',new):
            inv=a.inventories()
            # Synthetic fixture has no production terminal-report permission.
            # Only bank byte certification is fed its exact temporary maps.
            cohort={'original_inventory':proof['original_inventory'],'current_inventory':proof['current_inventory']}
            with patch.object(a,'_cohort',return_value=cohort):banks=a.banks_for(proof,inv)
            marker=new/'work/replacement_v3/known_invalid_episode_once.json'
            intent=new/'work/api_method/request_intents'/(proof['replacement']['request_hash']+'.json')
            stat=marker.stat()
            timing={'marker':a.binding(marker),'marker_mtime_ns':stat.st_mtime_ns,'marker_mtime':stat.st_mtime,
                'transport_intent':a.binding(intent),'intent_created_utc':read(intent)['created_utc']}
            cert={'run_inventories':inv,'banks':banks,'replacement_time':timing}
            expected={name:read(new/'results'/name) for name in a.OUTPUTS}
            paths={p for r in (ancestor,parent,new) for p in r.rglob('*') if p.is_file()}
            paths.update(a.ROOT/p for p in (*m['code_sha256'],*m['upstream_sha256']))
            paths.update((Path(m['terminal_proof']['path']),a.ROOT/a.frozen.metric.METRIC,Path(proof['role_qualification']['path'])))
            data={str(p):p.read_bytes() for p in paths}
            data[str(a.ROOT/'data/locomo10.json')]=json.dumps(dataset).encode()
            records=[]
            for path,b in sorted(data.items()):
                reason=a.disposition(path);h=hashlib.sha256(b).hexdigest()
                r={'path':path,'size':len(b),'sha256':h,'retained':reason is None}
                if reason:r['disposal_reason']=reason
                else:r['member']='blobs/'+h
                records.append(r)
            mp=base/'zip_manifest.json';zp=base/'evidence.zip';save_new(mp,{'schema':a.SCHEMA,'records':records})
            with zipfile.ZipFile(zp,'x') as z:
                z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
                for r in records:
                    if r['retained'] and r['member'] not in seen:z.writestr(r['member'],data[r['path']]);seen.add(r['member'])
            for root in (ancestor,parent,new):
                self.assertTrue(root.resolve().is_relative_to(base.resolve()));self.assertNotEqual(root.resolve(),base.resolve());shutil.rmtree(root)
            with a.Resolver(zp,mp) as view:
                a.bank_structure(view,proof,cert)
                auditor,transformations,checked=a.archived_auditors(view,cert,m,proof)
                runtime=a.frozen.metric.prepare_metric_runtime()
                with a.frozen.metric.only_zip_workspace_reads(view,runtime):
                    actual=auditor(a._engine.virtual_paths(view)(new),proof)
                    outputs=a.reconstruct(view,m,proof,created_utc=expected['summary.json']['created_utc'],raw_audit=actual)
                self.assertEqual(outputs,expected)
                self.assertEqual(outputs['summary.json']['effective']['reader_valid'],self.SLOT_COUNT)
                self.assertEqual(len(outputs['per_query.json']),self.SLOT_COUNT)
                self.assertEqual(sum(k=='index' for k,_ in checked),30)
                self.assertEqual(sum(k=='documents' for k,_ in checked),10)
                self.assertEqual(len(transformations['exact_substitutions']),4)
                self.assertTrue(transformations['sole_stat_substitution']['original_before_send_comparison_retained'])
                key=proof['replacement']['request_hash'];oldraw=proof['replacement']['raw']['path'];newraw=new/'work/api_method/api_cache'/(key+'.json')
                self.assertNotEqual(view.sha(oldraw),view.sha(newraw))
                self.assertEqual(view.sha(oldraw),proof['replacement']['raw']['sha256'])
                # Exact temporal gate survives deletion and cannot be relaxed
                # by substituting an internally consistent later timestamp.
                bad=deepcopy(cert);later=a.frozen.datetime.fromisoformat(timing['intent_created_utc']).timestamp()+1
                bad['replacement_time'].update(marker_mtime=later,marker_mtime_ns=round(later*1e9))
                bad_audit,_,_=a.archived_auditors(view,bad,m,proof)
                with a.frozen.metric.only_zip_workspace_reads(view,runtime):
                    with self.assertRaisesRegex(ValueError,'postdates'):bad_audit(a._engine.virtual_paths(view)(new),proof)
                # Native numeric caches are gone; original raw embeddings and
                # normalization digests remain independently checked.
                bad=deepcopy(cert);bank=next(iter(bad['banks'].values()));e=next(iter(bank['embeddings'].values()));e['normalized_vectors_sha256']='0'*64
                bad_audit,_,_=a.archived_auditors(view,bad,m,proof)
                with a.frozen.metric.only_zip_workspace_reads(view,runtime):
                    with self.assertRaisesRegex(ValueError,'normalized native embedding digest'):bad_audit(a._engine.virtual_paths(view)(new),proof)


class Packaging(unittest.TestCase):
    """Control-flow only; full native/QA replay has its own actual fixture."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);run=self.root/'runs/five_baselines_gemini25_v1_20261008'
        self.owners={'OLD':run/'hypermem','PARENT':run/'hypermem_native_v2','NEW':run/'hypermem_native_v3'}
        self.out=self.root/'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for k,v in {'ROOT':self.root,**self.owners,'OUTPUT':self.out,'DEFAULT_PLAN':self.out/'cleanup_plan.json'}.items():self.stack.enter_context(patch.object(a,k,v))
        def write(path,value):save_new(path,value);return path
        for ci,r in enumerate(self.owners.values()):
            write(r/'run_manifest.json',{'fixture_only':True});write(r/'results/summary.json',{'fixture_only':True})
            write(r/'work/api_method/api_cache/same_key.json',{'owner':ci,'original_evidence':'all distinct retained'})
            p=r/'work/native/c0/cache/hypergraph_embedding_index_conv_0.pkl';p.parent.mkdir(parents=True);p.write_bytes(b'disposable index')
        for p in (a.frozen.metric.METRIC,'data/locomo10.json','configs/sdk20_full1540_locked_v1.json'):write(self.root/p,{'fixture_only':True})
        typed=Path(__file__).resolve().parents[1]/'tools/baseline_evidence_closure_v1.py'
        target=self.root/'tools/baseline_evidence_closure_v1.py';target.parent.mkdir();target.write_bytes(typed.read_bytes())
        self.source=write(self.root/'fixture_source.json',{'fixture_only':True});self.codes={'fixture_source.json':sha(self.source)}
        self.stack.enter_context(patch.object(a,'code_hashes',return_value=self.codes));self.stack.enter_context(patch.object(a,'stopped'))
        self.stack.enter_context(patch.object(a,'evidence_closure',side_effect=lambda initial,**kw:a.frozen.simple.typed.collect(initial,root=a.ROOT,safe_path=a.safe,read=read,sha=sha,**kw)))
        self.inv=a.inventories();self.result=write(self.root/'result_review.json',{'fixture_only':True})
        self.tool=write(self.root/'tool_review.json',{'schema':a.REVIEW_SCHEMA,'status':'passed','scope':'three_root_lifecycle_with_explicit_predelete_bank_certificate',
            'code_sha256':self.codes,'evidence_sha256':{str(self.source):sha(self.source)}})
        self.cert=write(self.out/'bank_validation.json',{'fixture_only':True});self.bank=write(self.root/'bank_review.json',{'certificate':a.binding(self.cert),'fixture_only':True})
        self.stack.enter_context(patch.object(a,'result_review',return_value=({}, {}, {}, {})))
        self.stack.enter_context(patch.object(a._engine,'verify_certificate',return_value={'run_inventories':self.inv}))
        self.stack.enter_context(patch.object(a._engine,'strict_replay',return_value={'status':'passed','scope':'SYNTHETIC PACKAGING ONLY','runtime':{}}))

    def do_archive(self):return a.archive(self.result,self.tool,self.bank)
    def review(self,plan):
        p=self.root/'cleanup_review.json';save_new(p,{'schema':'hypermem-native-v3-cleanup-review-v1','status':'passed','cleanup_plan_sha256':sha(a.DEFAULT_PLAN),
            'archive_sha256':plan['archive']['sha256'],'scope':a.SCOPE,'allowed_delete_roots':plan['allowed_delete_roots']});return p

    def test_real_package_three_removal_receipts_and_postdelete_completion_gate(self):
        self.do_archive();plan=a.verify_archive();self.assertEqual(len(plan['allowed_delete_roots']),3)
        with a.Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as view:
            for ci,r in enumerate(self.owners.values()):self.assertEqual(view.read(r/'work/api_method/api_cache/same_key.json')['owner'],ci)
        review=self.review(plan)
        with self.assertRaises((FileNotFoundError,ValueError)):a.complete(review)
        rb=a.binding(review)
        save_new(self.out/'cleanup_execution_intent.json',{'schema':'hypermem-native-v3-cleanup-intent-v1','targets':plan['allowed_delete_roots'],
            'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        for name,r in a.roots().items():
            self.assertTrue((r/'work').resolve().is_relative_to(self.root.resolve()));shutil.rmtree(r/'work')
            save_new(self.out/('removal_'+name+'.json'),{'schema':'hypermem-native-v3-removal-receipt-v1','removed_path':str(r/'work'),
                'cleanup_plan_sha256':sha(a.DEFAULT_PLAN),'cleanup_review_sha256':rb['sha256']})
        a.complete(review);self.assertEqual(a.verify_completion()['status'],'passed')
        self.assertEqual(len(read(self.out/'cleanup_receipt.json')['removal_receipts']),3)
        (self.owners['PARENT']/'work').mkdir()
        with self.assertRaisesRegex(ValueError,'remains'):a.verify_completion()

    @unittest.skipUnless(os.name=='nt','Windows PowerShell fixture')
    def test_actual_powershell_three_targets_but_postdelete_failure_never_issues_completion(self):
        self.do_archive();plan=read(a.DEFAULT_PLAN);review=self.review(plan)
        bridge=self.root/'fixture_verify.py';bridge.write_text('import sys\n# Fake verify only for PowerShell control flow.\nraise SystemExit(0 if sys.argv[1]=="verify" else 9)\n','utf8')
        actual=Path(__file__).resolve().parents[1];script=(actual/'tools/cleanup_hypermem_native_v3.ps1').read_text()
        script=script.replace("'D:\\jev mem'",repr(str(self.root)))
        script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'","$python = '"+sys.executable+"'")
        script=script.replace("$helper = Join-Path $workspace 'tools\\archive_hypermem_native_v3.py'","$helper = '"+str(bridge)+"'")
        script=script.replace("$ErrorActionPreference = 'Stop'","$ErrorActionPreference = 'Stop'\nfunction Get-CimInstance { @() }")
        ps=self.root/'fixture_cleanup.ps1';ps.write_text(script,'utf8');before=sha(a.DEFAULT_PLAN)
        done=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ps),'-CleanupReview',str(review)],capture_output=True,text=True,timeout=50)
        self.assertNotEqual(done.returncode,0,done.stdout+done.stderr);self.assertIn('Postdelete proof failed',done.stderr)
        self.assertEqual(sha(a.DEFAULT_PLAN),before)
        for name,r in a.roots().items():
            self.assertFalse((r/'work').exists());self.assertTrue((self.out/('removal_'+name+'.json')).is_file())
        self.assertFalse((self.out/'cleanup_receipt.json').exists());self.assertFalse((self.out/'predecessor_completion.json').exists())


class PowerShellGuards(unittest.TestCase):
    def test_actual_three_root_powershell_reparse_and_active_reject(self):
        # Reuse the reviewed guard fixture verbatim with only the NEW source
        # basename. A process object is synthesized; no controller is spawned.
        from tests.test_hypermem_native_v2_lifecycle import PowerShellGuards as Original
        import inspect
        source=inspect.getsource(Original.test_actual_powershell_reparse_original_and_final_ancestor_rejected)
        import ast
        source=source.replace('cleanup_hypermem_native_v2.ps1','cleanup_hypermem_native_v3.ps1').replace('run_five_baselines_hypermem_native_v2','run_five_baselines_hypermem_native_v3')
        node=ast.parse('class Reused:\n'+source).body[0].body[0]
        namespace={**globals(),'__file__':__file__};exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact-new-PS-path-guard-fixture>','exec'),namespace)
        namespace['test_actual_powershell_reparse_original_and_final_ancestor_rejected'](self)


if __name__=='__main__':unittest.main()
