"""Offline archive integrity/resolver/counterexample tests; no actual run cleanup."""
import copy
import json
from pathlib import Path
import zipfile
from unittest.mock import patch
from tests import test_five_baselines_amem_native_v3 as original_tests
from tools import archive_amem_continuation_v3 as archive
from tools.baselines_common_v1 import save_new,sha


class ArchiveTests(original_tests.ContinuationTests):
    def setUp(self):
        super().setUp()
        for name,value in (('ROOT',self.root),('ROOTS',{'v1':self.ancestor,'v2':self.source,'v3':self.new})):
            self.patches.append(patch.object(archive,name,value));self.patches[-1].start()

    def pack(self,paths):
        records=[];payloads={}
        for p in paths:
            raw=p.read_bytes();h=archive.digest_bytes(raw);payloads[h]=raw
            records.append({'path':str(p),'sha256':h,'size':len(raw),'retained':True,'member':'blobs/'+h})
        manifest={'schema':archive.SCHEMA,'records':records,'roots':{k:str(v) for k,v in archive.ROOTS.items()}}
        mp=self.root/'archive_manifest.json';save_new(mp,manifest);zp=self.root/'evidence.zip'
        with zipfile.ZipFile(zp,'x') as z:
            z.writestr('archive_manifest.json',mp.read_bytes())
            for h,raw in payloads.items():z.writestr('blobs/'+h,raw)
        return zp,mp

    def test_content_dedup_preserves_same_request_different_raw_and_all_paths(self):
        a=self.root/'raw_a.json';b=self.root/'raw_b.json';c=self.root/'raw_c.json'
        a.write_text('{"request":"same","response":"first"}');b.write_bytes(a.read_bytes())
        c.write_text('{"request":"same","response":"second"}')
        zp,mp=self.pack([a,b,c])
        with archive.Resolver(zp,mp) as r:
            self.assertEqual(len(r.zip.namelist()),3)
            self.assertEqual(r.bytes(a),r.bytes(b));self.assertNotEqual(r.bytes(a),r.bytes(c))
            self.assertEqual(len(r.records),3)

    def test_actual_cross_root_raw_replay_uses_archive_after_original_raw_removed(self):
        q,ret,Recorder,metric,calls=self.qa_fixture();leaf=self.source/'results/firstpass'/q['query_id']
        reader=archive.read(leaf/'reader.json')
        with patch.object(archive.base,'BaselineAPI',Recorder),patch.object(archive.base,'verified_official_metric',return_value=(metric,{})):
            archive.base.qa_one(self.new,'amem',q,archive.read(ret)['value'],phase='recovery',inherited_reader=reader)
            paths=[p for folder in (self.source,self.new) for p in folder.rglob('*') if p.is_file()]
            zp,mp=self.pack(paths)
            raw_paths=[p for p in paths if p.parent.name=='api_cache']
            for p in raw_paths:
                self.assertTrue(p.resolve().is_relative_to(self.root));p.unlink()
            with archive.Resolver(zp,mp) as resolver:
                r,j=archive.qa_replayer(resolver)(q,self.new/'results/recovery'/q['query_id'],self.new,self.source,leaf,ret)
                self.assertTrue(r['inherited']);self.assertTrue(j['value']['correct'])
        self.assertEqual(len(calls),3)

    def test_archive_rejects_corrupt_blob_and_extra_zip_member(self):
        zp,mp=self.pack([self.raw])
        with zipfile.ZipFile(zp,'a') as z:z.writestr('not-declared','unexpected')
        with self.assertRaisesRegex(ValueError,'ZIP entries'):archive.Resolver(zp,mp)

    def test_resolver_rejects_payload_sha_mismatch_even_when_zip_crc_valid(self):
        zp,mp=self.pack([self.raw]);manifest=archive.read(mp);member=manifest['records'][0]['member']
        with zipfile.ZipFile(zp,'w') as z:
            z.writestr('archive_manifest.json',mp.read_bytes());z.writestr(member,b'changed bytes with valid CRC')
        with self.assertRaisesRegex(ValueError,'blob SHA'):archive.Resolver(zp,mp)

    def test_disposal_is_limited_to_native_sessions_and_encoder_cache(self):
        native=self.source/'work/native/c1'
        self.assertIsNotNone(archive.disposable(native/'sessions/session_1/note_000001.json'))
        self.assertIsNotNone(archive.disposable(native/'cache/embeddings.npy'))
        for p in (native/'calls/note_000001.json',native/'memory_updates.jsonl',native/'checkpoint.json.tmp',self.source/'work/api_method/api_cache/raw.json'):
            self.assertIsNone(archive.disposable(p))

    def test_foreign_path_rejected_before_archive_io(self):
        with self.assertRaisesRegex(ValueError,'Foreign'):archive.safe_path(self.root.parent/'outside-proof.json')

    def test_wrong_cleanup_roots_block_before_result_review(self):
        path=self.root/'plan.json';save_new(path,{'schema':archive.PLAN_SCHEMA,'workspace':str(self.root),
              'allowed_delete_roots':[str(self.root)],'archive_crc_verified':True})
        with patch.object(archive,'stopped'),patch.object(archive,'verify_result_review',side_effect=AssertionError('scope gate first')):
            with self.assertRaisesRegex(ValueError,'scope'):archive.verify_cleanup(path)

    def test_fake_cleanup_receipt_cannot_complete_or_launch_without_verified_archive(self):
        path=self.root/'plan.json';save_new(path,{'not':'a plan'})
        save_new(self.new/'results/cleanup_receipt.json',{'placeholder':True})
        with patch.object(archive,'stopped'):
            with self.assertRaisesRegex(ValueError,'scope'):archive.verify_completion(path)

    def test_review_evidence_and_test_source_sha_maps_are_transitively_retained(self):
        evidence=self.root/'evidence.json';save_new(evidence,{'source':'offline preflight'})
        test=self.root/'test_source.py';test.write_text('pass')
        review=self.root/'external_review.json';save_new(review,{'evidence_sha256':{str(evidence):sha(evidence)},
               'tests':{'files_sha256':{str(test):sha(test)}}})
        result=archive.closure([(str(review),sha(review))])
        self.assertEqual(set(result),{str(evidence),str(test),str(review)})

    def archive_fixture(self):
        actual=Path(archive.__file__).resolve().parents[1]
        for relative in archive.TOOL_FILES:
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((actual/relative).read_bytes())
        for folder in archive.ROOTS.values():
            (folder/'work').mkdir(parents=True,exist_ok=True)
        for folder in (self.ancestor,self.source):save_new(folder/'run_manifest.json',{'code_sha256':{},'upstream_sha256':{}})
        discarded=self.source/'work/native/c0/cache/embeddings.npy';discarded.parent.mkdir(parents=True);discarded.write_bytes(b'disposable fixture vectors')
        evidence=self.root/'preflight.json';save_new(evidence,{'fixture':True})
        source_proof=self.root/'terminal.json';save_new(source_proof,{
            'ancestor_inventory':{str(Path(p).relative_to(self.ancestor)):h for p,h in archive.file_inventory(self.ancestor).items()},
            'source_inventory':{str(Path(p).relative_to(self.source)):h for p,h in archive.file_inventory(self.source).items()}})
        dataset=self.root/'data/locomo10.json';save_new(dataset,{'offline_fixture':True})
        save_new(self.root/'configs/sdk20_full1540_locked_v1.json',{'offline_fixture':True})
        save_new(self.new/'run_manifest.json',{'dataset_sha256':sha(dataset),'code_sha256':{},'upstream_sha256':{},
                  'technical_continuation':{'proof':archive.binding(source_proof)}})
        summary=self.new/'results/summary.json';save_new(summary,{'audit_status':'no_invalid_artifacts','effective':{'slots':1540,'reader_valid':1540,'judge_valid':1540}})
        result_review=self.root/'result_review.json';save_new(result_review,{'status':'passed','summary_sha256':sha(summary),'offline_fixture_only':True,
                   'evidence_sha256':{str(evidence):sha(evidence)},'reviewed_run_inventory':archive.file_inventory(self.new)})
        tool_review=self.root/'tool_review.json';save_new(tool_review,{'status':'passed','offline_fixture_only':True,'code_sha256':{p:sha(self.root/p) for p in archive.TOOL_FILES}})
        return result_review,tool_review,discarded

    def test_result_review_requires_full_snapshot_not_only_unchanged_summary(self):
        review,_,_=self.archive_fixture();value=archive.read(review);original=copy.deepcopy(value)
        value.pop('reviewed_run_inventory');review.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'complete absolute run inventory'):archive.verify_result_review(review)
        review.write_text(json.dumps(original));archive.verify_result_review(review)
        raw=self.new/'work/api_method/api_cache/new.json';save_new(raw,{'request':'same','response':'after review'})
        save_new(self.new/'results/raw_request_audit.json',{'consistent_with_rewritten_raw':sha(raw)})
        self.assertEqual(sha(self.new/'results/summary.json'),original['summary_sha256'])
        with self.assertRaisesRegex(ValueError,'reviewed full inventory'):archive.verify_result_review(review)

    def test_archive_to_real_work_removal_to_completion_and_reappearing_work_refusal(self):
        import shutil
        result_review,tool_review,discarded=self.archive_fixture()
        # This fixture tests lifecycle/retention. Exact original QA replay itself
        # is exercised separately above with the frozen real evaluator.
        replay={'status':'passed','offline_fixture_only':True,'model_calls':0}
        with patch.object(archive,'stopped'),patch.object(archive,'replay_all',return_value=replay),patch.object(archive,'verify_native_seal'):
            result=archive.archive(result_review,tool_review);plan_path=Path(result['plan']);plan=archive.read(plan_path)
            archive.verify_cleanup(plan_path)
            with archive.Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as resolver:
                self.assertEqual(resolver.bytes(self.raw),self.raw.read_bytes())
                with self.assertRaisesRegex(ValueError,'not retained'):resolver.bytes(discarded)
                self.assertEqual(resolver.read(self.root/'preflight.json'),{'fixture':True})
            cleanup_review=self.root/'cleanup_review.json';save_new(cleanup_review,{'status':'passed','offline_fixture_only':True,
                 'cleanup_plan_sha256':sha(plan_path),'archive_sha256':plan['archive']['sha256']})
            save_new(archive.output_dir()/'cleanup_execution_intent.json',{'schema':'amem-cross-root-cleanup-intent-v1',
                 'cleanup_plan_sha256':sha(plan_path),'cleanup_review_sha256':sha(cleanup_review),
                 'targets':[{'path':p} for p in plan['allowed_delete_roots']]})
            for name,folder in archive.ROOTS.items():
                work=(folder/'work').resolve();self.assertTrue(work.is_relative_to(self.root.resolve()))
                self.assertEqual(work.name,'work');shutil.rmtree(work)
                save_new(archive.output_dir()/('cleanup_'+name+'.json'),{'schema':'amem-cross-root-removal-receipt-v1',
                     'removed_path':str(work),'cleanup_plan_sha256':sha(plan_path),'cleanup_review_sha256':sha(cleanup_review)})
            self.assertEqual(archive.complete(plan_path,cleanup_review)['status'],'complete')
            self.assertEqual(archive.verify_completion(plan_path)['status'],'passed')
            (self.source/'work').mkdir()
            with self.assertRaisesRegex(ValueError,'reappeared'):archive.verify_completion(plan_path)

    def test_actual_powershell_path_gate_supports_file_and_directory_without_deletion(self):
        import subprocess
        script=Path(archive.__file__).with_name('cleanup_amem_continuation_v3.ps1')
        sample=self.root/'retained.json';sample.write_text('{}')
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command=("$ErrorActionPreference='Stop'; $tokens=$null; $errors=$null; "
                 "$ast=[System.Management.Automation.Language.Parser]::ParseFile("+quote(script)+",[ref]$tokens,[ref]$errors); "
                 "if($errors.Count){throw 'parse errors'}; "
                 "$definition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'AssertUnderWorkspace'},$true); "
                 ". ([scriptblock]::Create($definition.Extent.Text)); $workspace="+quote(self.root)+"; "
                 "$file=AssertUnderWorkspace "+quote(sample)+"; $directory=AssertUnderWorkspace "+quote(self.source)+"; "
                 "if($file -ne "+quote(sample)+" -or $directory -ne "+quote(self.source)+"){throw 'resolved path differs'}; "
                 "$rejected=$false; try { AssertUnderWorkspace "+quote(script)+" | Out-Null } catch { $rejected=$true }; "
                 "if(-not $rejected){throw 'foreign path admitted'}; Write-Output 'file_directory_foreign_gates_passed'")
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('file_directory_foreign_gates_passed',result.stdout)

    def test_real_powershell_three_root_cleanup_and_active_module_controller_block(self):
        import subprocess,sys
        actual=Path(archive.__file__).resolve().parents[1]
        run=self.root/'runs/five_baselines_gemini25_v1_20261008'
        self.ancestor=run/'amem';self.source=run/'amem_format_v2';self.new=run/'amem_native_v3'
        roots={'v1':self.ancestor,'v2':self.source,'v3':self.new}
        with patch.object(archive,'ROOTS',roots),patch.object(archive,'stopped'),patch.object(archive,'verify_native_seal'),patch.object(archive,'replay_all',return_value={'status':'passed','offline_fixture_only':True,'model_calls':0}):
            result_review,tool_review,discarded=self.archive_fixture();result=archive.archive(result_review,tool_review)
            plan_path=Path(result['plan']);plan=archive.read(plan_path)
            cleanup=self.root/'cleanup_review.json';save_new(cleanup,{'status':'passed','offline_fixture_only':True,
                   'cleanup_plan_sha256':sha(plan_path),'archive_sha256':plan['archive']['sha256']})
            stub=self.root/'OFFLINE_FIXTURE_verifier.py'
            stub.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(actual))+")\n"
                 "from tools import archive_amem_continuation_v3 as x\nx.ROOT=Path("+repr(str(self.root))+")\n"
                 "x.ROOTS="+repr({k:str(v) for k,v in roots.items()})+"\nx.ROOTS={k:Path(v) for k,v in x.ROOTS.items()}\n"
                 "x.stopped=lambda:None\nx.replay_all=lambda r:{'status':'passed','offline_fixture_only':True,'model_calls':0}\nx.main()\n",encoding='utf-8')
            quote=lambda x:"'"+str(x).replace("'","''")+"'"
            script=(actual/'tools/cleanup_amem_continuation_v3.ps1').read_text(encoding='utf-8')
            script=script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path","(Resolve-Path -LiteralPath "+quote(self.root)+").Path")
            script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'",'$python = '+quote(sys.executable))
            script=script.replace("$helper = Join-Path $workspace 'tools\\archive_amem_continuation_v3.py'",'$helper = '+quote(stub))
            fixture=self.root/'OFFLINE_FIXTURE_cleanup.ps1';needle="$ErrorActionPreference = 'Stop'"
            active="function Get-CimInstance { [pscustomobject]@{Name='python3.11.exe';CommandLine='python -m tools.run_five_baselines_reviewed_chain_v1 run --method simplemem'} }\n"
            fixture.write_text(script.replace(needle,active+needle),encoding='utf-8')
            command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(fixture),'-CleanupReview',str(cleanup)]
            blocked=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(blocked.returncode,0);self.assertIn('controller is still active',blocked.stderr)
            self.assertTrue(discarded.exists());self.assertFalse((archive.output_dir()/'cleanup_execution_intent.json').exists())
            fixture.write_text(script.replace(needle,'function Get-CimInstance { @() }\n'+needle),encoding='utf-8')
            for folder in roots.values():self.assertTrue((folder/'work').resolve().is_relative_to(self.root.resolve()))
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            self.assertTrue(all(not (folder/'work').exists() for folder in roots.values()))
            self.assertEqual(archive.verify_completion(plan_path)['status'],'passed')

    def test_real_full_1540_archive_only_cohort_replays_3080_leaves(self):
        q,ret,Recorder,metric,calls=self.qa_fixture();leaf=self.source/'results/firstpass'/q['query_id']
        original_reader=archive.read(leaf/'reader.json');original_judge=archive.read(leaf/'judge.json')
        documents={};questions=[];sources=[];firstpass=[];effective=[]
        encode=lambda v:(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        for stage in (original_reader,original_judge):
            raw_path=Path(stage['events'][0]['cache_path']);documents[str(raw_path)]=raw_path.read_bytes()
        for ordinal in range(1540):
            current={**q,'query_id':f'c0_q{ordinal}'};questions.append(current)
            current_leaf=self.source/'results/firstpass'/current['query_id']
            ret_path=self.source/'results/c0/retrieval'/(current['query_id']+'.json')
            stages={}
            for kind,original in (('reader',original_reader),('judge',original_judge)):
                stage=copy.deepcopy(original);stage['events'][0]['context']['query_id']=current['query_id'];stages[kind]=stage
                documents[str(current_leaf/(kind+'.json'))]=encode(stage)
                documents[str(current_leaf/(kind+'.intent.json'))]=encode({'context':stage['events'][0]['context']})
            outcome={'query_id':current['query_id'],'conversation':0,'method':'amem','phase':'firstpass',
                     'reader':'complete','judge':'complete','reader_path':str(current_leaf/'reader.json'),'judge_path':str(current_leaf/'judge.json')}
            documents[str(current_leaf/'outcome.json')]=encode(outcome);documents[str(ret_path)]=ret.read_bytes()
            sources.append({'query_id':current['query_id'],'conversation':0,'owner':str(self.source),'leaf':str(current_leaf),'retrieval_path':str(ret_path),
                            'bindings':{str(p):archive.digest_bytes(documents[str(p)]) for p in (ret_path,current_leaf/'reader.json',current_leaf/'judge.json',current_leaf/'outcome.json')}})
            value={**current,'source_owner':str(self.source),'source_leaf':str(current_leaf),'retrieval_source':str(ret_path),
                   'answer':'park','judge_correct':True,'official_f1':1.0,'custom_metrics':archive.base.evaluator.lexical_metrics('park','park'),
                   'reader_status':'complete','judge_status':'complete'}
            firstpass.append({**value,'phase':'firstpass'});effective.append({**value,'phase':'effective'})
        proof_path=self.root/'synthetic_proof.json';documents[str(proof_path)]=encode({'conversations':{'0':{'mode':'inherit'}}})
        aggregate={'slots':1540,'reader_valid':1540,'judge_valid':1540,'custom_f1_valid_percent':100.0,
                   'official_f1_valid_percent':100.0,'official_f1_fixed_missing_zero_percent':100.0,'judge_valid_percent':100.0,
                   'judge_fixed_missing_zero_percent':100.0,'missing_reader':0,'missing_judge':0}
        summary={'firstpass':aggregate,'effective':aggregate}
        synthetic_manifest={'questions':questions,'technical_continuation':{'proof':{'path':str(proof_path),'sha256':archive.digest_bytes(documents[str(proof_path)])}}}
        for path,value in ((self.new/'run_manifest.json',synthetic_manifest),(self.new/'results/firstpass_sources.json',sources),
                           (self.new/'results/firstpass_per_query.json',firstpass),(self.new/'results/per_query.json',effective),
                           (self.new/'qa_recovery_plan.json',{'round':1,'jobs':[]}),(self.new/'results/summary.json',summary)):
            documents[str(path)]=encode(value)
        records=[];blobs={}
        for path,raw in documents.items():
            h=archive.digest_bytes(raw);blobs[h]=raw;records.append({'path':path,'sha256':h,'size':len(raw),'retained':True,'member':'blobs/'+h})
        manifest={'schema':archive.SCHEMA,'roots':{k:str(v) for k,v in archive.ROOTS.items()},'records':records}
        mp=self.root/'synthetic_archive_manifest.json';save_new(mp,manifest);zp=self.root/'synthetic_archive.zip'
        with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('archive_manifest.json',mp.read_bytes())
            for h,raw in blobs.items():z.writestr('blobs/'+h,raw)
        # None of the additional 1539 query files exists on disk. Original raw
        # producers are removed as well: the resolver must read ZIP evidence.
        for stage in (original_reader,original_judge):Path(stage['events'][0]['cache_path']).unlink()
        with archive.Resolver(zp,mp) as resolver:
            result=archive.replay_all(resolver)
            original_read=resolver.read
            for field in ('phase','custom_metrics','retrieval_source'):
                altered=copy.deepcopy(effective);altered[0][field]='wrong'
                with self.subTest(field=field),patch.object(resolver,'read',side_effect=lambda p:altered if Path(p)==self.new/'results/per_query.json' else original_read(p)):
                    with self.assertRaisesRegex(ValueError,'complete per-query'):archive.replay_all(resolver)
            changed_summary=copy.deepcopy(summary);changed_summary['effective']['judge_valid_percent']=0
            with patch.object(resolver,'read',side_effect=lambda p:changed_summary if Path(p)==self.new/'results/summary.json' else original_read(p)):
                with self.assertRaisesRegex(ValueError,'summary differs'):archive.replay_all(resolver)
        self.assertEqual(result['firstpass_and_effective_leaves'],3080)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(calls),2)


if __name__=='__main__':
    import unittest
    unittest.main()
