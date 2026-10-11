"""New lifecycle only: no production writes, providers or process fixtures.

The full-cohort ZIP fixture uses the already frozen author factories, with fake
HTTP and fixed seed indices. It verifies the real native formatter, producer
audits, Reader/Judge replay and union; it does not claim numeric search testing.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_amem_retrieval_continuation_v1 as archive
from tools import baseline_lifecycle_v3 as life
from tools import run_five_baselines_reviewed_chain_v3 as launch
from tools import seal_amem_retrieval_continuation_v1 as union
from tools import run_five_baselines_amem_retrieval_v1 as entry
from tools.baselines_common_v1 import read,sha,save_new


ACTUAL=Path(__file__).resolve().parents[1]


def patch_scope(stack,root,roots):
    final=roots['v4']/'retrieval_continuation_v1';owner=roots['v4']/'work/retrieval_continuation_v1'
    for module in (archive,archive.engine):
        for key,value in {'ROOT':root,'ROOTS':roots,'FINAL':final}.items():
            stack.enter_context(patch.object(module,key,value))
    stack.enter_context(patch.object(archive,'WORK_OWNER',owner))
    return final,owner


def pack(root,docs,roots,review):
    records=[];blobs={}
    for name,raw in docs.items():
        h=hashlib.sha256(raw).hexdigest();blobs[h]=raw
        records.append({'path':name,'sha256':h,'size':len(raw),'retained':True,'member':'blobs/'+h})
    mp=root/'archive_manifest.json';zp=root/'evidence.zip'
    save_new(mp,{'schema':archive.SCHEMA,'roots':{k:str(v) for k,v in roots.items()},'records':records,'result_review':review})
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('archive_manifest.json',mp.read_bytes())
        for h,raw in blobs.items():z.writestr('blobs/'+h,raw)
    return zp,mp


class FullZipReplay(unittest.TestCase):
    def test_actual_full1540_union_from_zip_with_no_original_work(self):
        from tests.test_amem_retrieval_union_v1 import UnionTests
        from tests.test_amem_retrieval_continuation_v1 import ContinuationTests
        f=UnionTests();native=ContinuationTests();native.setUp();f.setUp()
        try:
            f.entry_patch.stop();native.owner=f.work;native.final=f.final
            roots={'v1':f.run/'amem','v2':f.run/'amem_format_v2','v3':f.run/'amem_native_v3','v4':f.parent}
            with ExitStack() as stack:
                patch_scope(stack,f.root,roots)
                stack.enter_context(patch.object(entry,'WORK_OWNER',f.work));stack.enter_context(patch.object(entry,'FINAL',f.final))
                native.complete()
                ret_path=f.work/'results/c2/retrieval/c2_q42.json';ret=read(ret_path);count=[0]
                def qa_api(folder,**kwargs):
                    api=entry.BaselineAPI(folder,keys={'GEMINI_API_KEY':'offline-fixture-not-real'},**kwargs)
                    def post(*args,**kw):
                        count[0]+=1;return native.response(text=str(native.q['reference']) if count[0]==1 else '[[yes]]')
                    api.local.session=SimpleNamespace(post=post);return api
                with patch.object(entry.base,'BaselineAPI',side_effect=qa_api):
                    entry.base.qa_one(f.work,'amem',native.q,ret['value'],phase='firstpass')
                self.assertEqual(count[0],2);union.copy_qa('firstpass')
                for p in (ret_path,ret_path.with_suffix('.intent.json')):entry.copy_new(p,f.final/'results/technical_retrieval'/p.name)
                f.q.clear();f.q.update(native.q)
                f.proof.update({k:deepcopy(v) for k,v in native.proof.items() if k!='original_failure'})
                f.proof['original_failure'].update(native.proof['original_failure'])
                for name in list(f.docs):
                    if Path(name).is_relative_to(f.work) or Path(name).is_relative_to(f.final):del f.docs[name]
                for folder in (f.work,f.final):
                    for p in folder.rglob('*'):
                        if p.is_file():f.docs[str(p)]=p.read_bytes()
                for record in (native.proof['replacement_certificate'],native.proof['bank']['state'],native.proof['original_failure']['native_call'],native.proof['original_failure']['raw']):
                    f.docs[record['path']]=Path(record['path']).read_bytes()
                for name,folder in roots.items():
                    if name!='v4':f.put(folder/'run_manifest.json',{'synthetic_history':name})
                parent_proof=f.root/'parent_proof.json'
                f.put(parent_proof,{'history_inventories':{name:{str(Path(p).relative_to(folder)):hashlib.sha256(raw).hexdigest()
                    for p,raw in f.docs.items() if Path(p).is_relative_to(folder)} for name,folder in roots.items() if name!='v4'}})
                f.put(f.parent/'run_manifest.json',{'questions':f.qs,'conversation_sources':f.routes,
                    'technical_continuation':{'proof':{'path':str(parent_proof),'sha256':f.hash(parent_proof)}}})
                f.put(f.parent/'status.json',{'stage':'needs_review','query_id':'c2_q42'})
                f.proof['source_inventory_exclusions']=[str(f.final),str(f.work)]
                f.proof['source_inventory']={p:hashlib.sha256(raw).hexdigest() for p,raw in f.docs.items() if Path(p).is_relative_to(f.parent)
                    and not any(Path(p).is_relative_to(x) for x in (f.final,f.work))}
                f.put(f.root/'proof.json',f.proof);f.manifest['proof']['sha256']=f.hash(f.root/'proof.json')
                f.manifest['parent_manifest']['sha256']=f.hash(f.parent/'run_manifest.json');f.manifest['upstream_sha256']={}
                core_review=f.root/'core_review.json';f.put(core_review,{'status':'passed','scope':'single_query_core_implementation_and_actual_admission',
                    'proof':f.manifest['proof'],'code_sha256':{},'archive_cleanup_downstream_authorized':False,
                    'evidence_sha256':{str(parent_proof):f.hash(parent_proof)}})
                f.manifest['independent_review']={'path':str(core_review),'sha256':f.hash(core_review)}
                f.put(f.final/'run_manifest.json',f.manifest);f.finalize_plan()
                outputs=union.reconstruct(f.manifest,f.proof,view=f.view(),created_utc='fixture')
                for name,value in outputs.items():f.put(f.final/'results'/name,value)
                f.put(f.final/'status.json',{'stage':'complete_awaiting_result_review','summary_sha256':f.hash(f.final/'results/summary.json')})
                key=entry.amem._v1._sha({'question':native.q['question'],'memory_sha256':native.state['memory_sha256']})
                result_path=f.work/'native/c2/retrieval'/(key+'.json')
                seed_path=f.root/'seed_review.json';f.put(seed_path,{'schema':'amem-query-retrieval-seed-independent-review-v1','status':'passed','query_id':'c2_q42',
                    'model_calls':0,'credential_reads':0,'query_embedding_calls':1,'corpus_embedding_calls':0,'query_embedding_recomputed':True,
                    'source_inventory_before_after_equal':True,'full_result_equal':True,'bank_state':native.proof['bank']['state'],
                    'embeddings':native.proof['bank']['embeddings'],'retrieval':{'path':str(ret_path),'sha256':f.hash(ret_path)},
                    'native_result':{'path':str(result_path),'sha256':f.hash(result_path)},'seed_indices':list(range(10)),
                    'synthetic_fixture_only_not_a_numeric_validation':True})
                review_path=f.root/'result_review.json';f.put(review_path,{'status':'passed','retrieval_seed_review':{'path':str(seed_path),'sha256':f.hash(seed_path)}})
                zp,mp=pack(f.root,f.docs,roots,{'path':str(review_path),'sha256':f.hash(review_path)})
                # The only real work in this fixture is temporary. All original
                # production files above were READ, copied into ZIP, never changed.
                for folder in roots.values():
                    work=folder/'work'
                    if work.exists():
                        self.assertTrue(work.resolve().is_relative_to(f.root.resolve()));self.assertEqual(work.name,'work');shutil.rmtree(work)
                self.assertTrue(all(not (p/'work').exists() for p in roots.values()))
                with archive.Resolver(zp,mp) as view:
                    expected=archive.replay_all(view)
                    self.assertEqual(expected['effective_slots'],1540);self.assertEqual(expected['original_success_qa_replays'],1539)
                    self.assertEqual(view.read(f.parent/'status.json')['stage'],'needs_review')
                    original_read=view.read
                    bad=deepcopy(outputs['per_query.json']);bad[0]['phase']='changed'
                    with patch.object(view,'read',side_effect=lambda p:bad if Path(p)==f.final/'results/per_query.json' else original_read(p)):
                        with self.assertRaisesRegex(ValueError,'union seal output differs'):archive.replay_all(view)
                    bad_seed=original_read(seed_path);bad_seed['seed_indices']=list(reversed(bad_seed['seed_indices']))
                    with patch.object(view,'read',side_effect=lambda p:bad_seed if Path(p)==seed_path else original_read(p)):
                        with self.assertRaisesRegex(ValueError,'ordered seeds'):archive.verify_seed_review(original_read(review_path),view=view)
                    records=deepcopy(view.manifest['records']);target=next(r for r in view.manifest['records'] if r['path']==str(f.parent/'status.json'));target['sha256']='0'*64
                    with self.assertRaisesRegex(ValueError,'pre-sidecar'):archive._archived_contract(view)
                    view.manifest['records']=records
        finally:
            f.tearDown();native.doCleanups()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='single_query_lifecycle_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();run=self.root/'runs/five_baselines_gemini25_v1_20261008'
        self.roots=dict(zip(('v1','v2','v3','v4'),(run/x for x in ('amem','amem_format_v2','amem_native_v3','amem_native_v4'))))
        self.stack=ExitStack();self.addCleanup(self.stack.close);self.final,self.owner=patch_scope(self.stack,self.root,self.roots)
        self.stack.enter_context(patch.object(archive.engine,'stopped'))
        self.stack.enter_context(patch.object(archive.engine,'verify_native_seal'))
        self.stack.enter_context(patch.object(archive.engine,'replay_all',return_value={'status':'passed','fixture_only':True}))
        self.stack.enter_context(patch.object(archive,'verify_seed_review',return_value={'fixture_only':True}))
        # Lifecycle/real-PowerShell scope fixture; actual replay has separate full
        # ZIP test above, with no reconstruction or method-audit stubs.
        for p in archive.TOOL_FILES:
            target=self.root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ACTUAL/p).read_bytes())
        for folder in self.roots.values():
            (folder/'work').mkdir(parents=True);save_new(folder/'run_manifest.json',{})
        self.marker=self.owner/'work/one_use_retrieval_replacement.json';save_new(self.marker,{'logical_limit':1})
        self.same=[]
        for name in ('v3','v4'):
            p=self.roots[name]/'work/api_method/api_cache/samekey.json';save_new(p,{'request':'same','response':name});self.same.append(p)
        self.vector=self.roots['v4']/'work/native/c2/cache/embeddings.npy';self.vector.parent.mkdir(parents=True);self.vector.write_bytes(b'numeric fixture')
        self.evidence=self.root/'numeric_seed_review.json';save_new(self.evidence,{'fixture_only':True})
        history=self.root/'history.json';save_new(history,{'history_inventories':{name:{str(Path(p).relative_to(folder)):h for p,h in archive.file_inventory(folder).items()}
            for name,folder in self.roots.items() if name!='v4'}})
        dataset=self.root/'data/locomo10.json';save_new(dataset,{'fixture_only':True});save_new(self.root/'configs/sdk20_full1540_locked_v1.json',{})
        (self.roots['v4']/'run_manifest.json').write_text(json.dumps({'dataset_sha256':sha(dataset),'technical_continuation':{'proof':archive.binding(history)}}))
        save_new(self.roots['v4']/'status.json',{'stage':'needs_review'})
        save_new(self.final/'run_manifest.json',{'proof':archive.binding(history)})
        self.summary=self.final/'results/summary.json';save_new(self.summary,{'schema':union.SCHEMA,'method':'amem','self_root':str(self.final),'parent_root':str(self.roots['v4']),
            'work_owner':str(self.owner),'audit_status':'no_invalid_artifacts','effective':{'slots':1540,'reader_valid':1540,'judge_valid':1540}})
        save_new(self.final/'status.json',{'stage':'complete_awaiting_result_review','summary_sha256':sha(self.summary)})
        self.review=self.root/'result_review.json';save_new(self.review,{'status':'passed','summary_sha256':sha(self.summary),
            'reviewed_run_inventory':archive.file_inventory(self.roots['v4']),'retrieval_seed_review':archive.binding(self.evidence)})
        self.toolreview=self.root/'tool_review.json';save_new(self.toolreview,{'status':'passed','code_sha256':{p:sha(self.root/p) for p in archive.TOOL_FILES}})

    def test_full_inventory_scope_and_result_location(self):
        archive.verify_result_review(self.review)
        value=read(self.review);value.pop('reviewed_run_inventory');self.review.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'complete parent'):archive.verify_result_review(self.review)
        self.assertEqual(read(self.roots['v4']/'status.json')['stage'],'needs_review')

    def test_live_seal_passes_exact_result_review_to_same_replayer(self):
        view=SimpleNamespace(manifest={'roots':{},'records':[]})
        def verify(actual):
            self.assertIs(actual,view);self.assertEqual(actual.manifest['result_review'],archive.binding(self.review))
            return {'status':'passed','fixture_only':True}
        with patch.object(entry,'manifest',return_value=({},{})),patch.object(archive.engine,'LiveResolver',return_value=view),patch.object(archive,'replay_all',side_effect=verify):
            self.assertEqual(archive.verify_native_seal(self.review)['status'],'passed')

    def test_failure_receipt_or_remaining_work_cannot_complete(self):
        outcome=archive.archive(self.review,self.toolreview);plan_path=Path(outcome['plan']);plan=read(plan_path)
        cleanup=self.root/'cleanup_review.json';save_new(cleanup,{'status':'passed','cleanup_plan_sha256':sha(plan_path),'archive_sha256':plan['archive']['sha256']})
        save_new(archive.output_dir()/'cleanup_execution_intent.json',{'schema':'amem-four-root-retrieval-cleanup-intent-v1',
            'cleanup_plan_sha256':sha(plan_path),'cleanup_review_sha256':sha(cleanup),'targets':[{'path':p} for p in plan['allowed_delete_roots']]})
        with self.assertRaisesRegex(ValueError,'still exists'):archive.complete(plan_path,cleanup)
        self.assertFalse((self.final/'results/predecessor_completion.json').exists())
        self.assertFalse((self.final/'results/cleanup_receipt.json').exists())

    def test_archive_exact_four_roots_and_nested_owner_counted_once(self):
        outcome=archive.archive(self.review,self.toolreview);plan=read(outcome['plan'])
        self.assertEqual(plan['allowed_delete_roots'],[str(p/'work') for p in self.roots.values()])
        with archive.Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as view:
            records=view.manifest['records'];self.assertEqual(sum(r['path']==str(self.marker) for r in records),1)
            self.assertEqual(view.read(self.same[0])['response'],'v3');self.assertEqual(view.read(self.same[1])['response'],'v4')
            self.assertEqual(view.read(self.marker),{'logical_limit':1});self.assertEqual(view.read(self.evidence),{'fixture_only':True})
            with self.assertRaisesRegex(ValueError,'not retained'):view.bytes(self.vector)
        archive.verify_cleanup(outcome['plan'])
        plan['allowed_delete_roots'].append(str(self.owner));Path(outcome['plan']).write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'scope'):archive.verify_cleanup(outcome['plan'])

    def test_real_powershell_only_temp_four_root_cleanup_and_refusal(self):
        outcome=archive.archive(self.review,self.toolreview);plan_path=Path(outcome['plan']);plan=read(plan_path)
        cleanup=self.root/'cleanup_review.json';save_new(cleanup,{'status':'passed','cleanup_plan_sha256':sha(plan_path),'archive_sha256':plan['archive']['sha256']})
        stub=self.root/'fixture_verifier.py'
        stub.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ACTUAL))+")\n"
            "from tools import archive_amem_retrieval_continuation_v1 as a\n"
            "a.ROOT=Path("+repr(str(self.root))+");a.ROOTS="+repr({k:str(v) for k,v in self.roots.items()})+"\na.ROOTS={k:Path(v) for k,v in a.ROOTS.items()}\n"
            "a.FINAL=Path("+repr(str(self.final))+");a.WORK_OWNER=Path("+repr(str(self.owner))+")\n"
            "a.engine.ROOT=a.ROOT;a.engine.ROOTS=a.ROOTS;a.engine.FINAL=a.FINAL\n"
            "a.engine.stopped=lambda:None;a.engine.replay_all=lambda r:{'status':'passed','fixture_only':True}\na.main()\n",encoding='utf8')
        quote=lambda x:"'"+str(x).replace("'","''")+"'"
        script=(ACTUAL/'tools/cleanup_amem_retrieval_continuation_v1.ps1').read_text('utf8')
        script=script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path","(Resolve-Path -LiteralPath "+quote(self.root)+").Path")
        script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'",'$python = '+quote(sys.executable))
        script=script.replace("$helper = Join-Path $workspace 'tools\\archive_amem_retrieval_continuation_v1.py'",'$helper = '+quote(stub))
        fixture=self.root/'fixture_cleanup.ps1';needle="$ErrorActionPreference = 'Stop'"
        fixture.write_text(script.replace(needle,"function Get-CimInstance { [pscustomobject]@{Name='python.exe';CommandLine='python -m tools.run_five_baselines_amem_retrieval_v1 run'} }\n"+needle),'utf8')
        cmd=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(fixture),'-CleanupReview',str(cleanup)]
        blocked=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(blocked.returncode,0);self.assertIn('controller is still active',blocked.stderr)
        self.assertTrue(all((p/'work').exists() for p in self.roots.values()))
        fixture.write_text(script.replace(needle,'function Get-CimInstance { @() }\n'+needle),'utf8')
        for p in self.roots.values():self.assertTrue((p/'work').resolve().is_relative_to(self.root))
        completed=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertTrue(all(not (p/'work').exists() for p in self.roots.values()))
        self.assertEqual(archive.verify_completion(plan_path)['status'],'passed')
        self.assertEqual(read(self.roots['v4']/'status.json')['stage'],'needs_review')
        self.assertFalse((self.roots['v4']/'results/predecessor_completion.json').exists())
        (self.roots['v2']/'work').mkdir()
        with self.assertRaisesRegex(ValueError,'reappeared'):archive.verify_completion(plan_path)


class DispatchTests(unittest.TestCase):
    def test_exact_route_provider_and_all_code_bindings(self):
        from tools import run_five_baselines_amem_native_v3_admission_v2 as old
        from tools.baselines_common_v1 import BaselineAPI
        self.assertEqual(launch.frozen.NEW,union.FINAL);self.assertEqual(old.NEW.name,'amem_native_v3')
        self.assertEqual(launch.frozen.main.__code__.co_code,old.main.__code__.co_code)
        self.assertEqual(launch.frozen.main.__code__.co_consts,old.main.__code__.co_consts)
        self.assertIs(launch.frozen.base.BaselineAPI,BaselineAPI)
        self.assertTrue(set(archive.TOOL_FILES).issubset(life.code_hashes()))
        self.assertIn('tools/baseline_lifecycle_v2.py',life.code_hashes())

    def test_every_predecessor_verified_and_failure_prevents_dispatch(self):
        for method in life.METHODS:
            events=[]
            with patch.object(life,'stopped'),patch.object(archive,'verify_completion',side_effect=lambda p:events.append('amem') or {'status':'passed'}),patch.object(life,'verify_completion',side_effect=lambda m:events.append(m) or {'status':'passed'}):
                launch.verify_predecessors(method)
            self.assertEqual(events,['amem',*life.METHODS[:life.METHODS.index(method)]])
        with patch('sys.argv',['entry','run','--method','simplemem']),patch.object(launch.frozen.base,'manifest',return_value={'review':{'path':'fake'}}),patch.object(launch,'verify_code_review'),patch.object(launch,'verify_predecessors',side_effect=ValueError('bad predecessor')),patch.object(launch.frozen.base,'run') as run:
            with self.assertRaisesRegex(ValueError,'bad predecessor'):launch.main()
            run.assert_not_called()

    def test_old_supplement_cannot_prepare_new_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.json';save_new(path,{'status':'passed','code_sha256':{}})
            with patch('sys.argv',['entry','prepare','--method','simplemem','--review',str(path)]),patch.object(launch.frozen,'main') as run:
                with self.assertRaisesRegex(ValueError,'every new'):launch.main()
                run.assert_not_called()


if __name__=='__main__':unittest.main()
