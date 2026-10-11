"""Offline archive integrity/resolver/counterexample tests; no actual run cleanup."""
import copy
import json
from pathlib import Path
import zipfile
import shutil
from unittest.mock import patch
from tests import test_five_baselines_amem_native_v3 as original_tests
from tools import archive_amem_continuation_v4 as archive
from tools.baselines_common_v1 import save_new,sha


class ArchiveTests(original_tests.ContinuationTests):
    def setUp(self):
        super().setUp()
        for name,value in (('ROOT',self.root),('ROOTS',{'v1':self.ancestor,'v2':self.source,'v3':self.root/'third','v4':self.new})):
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

    def test_virtual_path_audits_native_and_raw_after_fixture_work_removed(self):
        # Real frozen native audit, including raw/schema/normalization/exports.
        from tests.test_five_baselines_amem_native_v3 import v3, ORIGINAL_REQUEST
        m=self.native_audit_fixture();expected=v3.audit_native_outputs(m)
        paths=[p for p in self.new.rglob('*') if p.is_file()]
        zp,mp=self.pack(paths)
        work=self.new/'work';self.assertTrue(work.resolve().is_relative_to(self.root.resolve()));shutil.rmtree(work)
        with archive.Resolver(zp,mp) as resolver:
            P=archive.archived_path_class(resolver)
            def bound(p,f):
                p=P(p)
                if not p.is_relative_to(f) or not p.is_file():raise ValueError('foreign fixture')
                return p
            audit=archive.clone(v3.audit_native_outputs,NEW=P(self.new),Path=P,read=resolver.read,sha=resolver.sha,bound_path=bound)
            self.assertEqual(audit(m),expected)

    def test_real_full_seal_replay_1540_all_owners_without_live_work(self):
        from tools import baselines_amem_v4 as adapter
        from tests.test_baselines_amem_v1 import FakeAPI,FakeEncoder,conversation
        from tools.baselines_common_v1 import append
        from jev_mem.sdk20_cached_api import digest
        from jev_mem.gemini25_ablation_provider_v1 import generation_body
        from tools import baseline_lifecycle_v2 as life
        outer=self;roots=archive.ROOTS;documents={};questions=[];sources=[];firstpass=[];effective=[]
        encode=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
        # Build a small real native graph with the same frozen scientific methods.
        class MethodRecorder(FakeAPI):
            def __init__(self):super().__init__();self.events=[]
            def generate(self,prompt,**kwargs):
                value=super().generate(prompt,**kwargs)
                if isinstance(value,str):value=json.loads(value)
                sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
                     'body':generation_body(prompt,system=kwargs.get('system'),json_output=kwargs['json_output'],max_output=kwargs['max_output'])}
                key=digest(sig);folder=outer.new/'work/api_method';p=folder/'api_cache'/(key+'.json')
                raw={'request':sig,'response':{'modelVersion':sig['model'],'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':json.dumps(value)}]}}]}}
                context={'method':'amem','conversation':2,'phase':'build'}
                if kwargs['stage']=='amem_query_keywords':context.update(phase='retrieval',query_id='c2_q0')
                save_new(p,raw)
                parsed,journal=adapter.decode_generation_json(raw,raw_artifact_sha256=sha(p));save_new(folder/'parser_lineage'/(key+'.json'),journal)
                identity={k:sig[k] for k in ('provider','model','operation')}
                event={**identity,'request_hash':key,'cache_path':str(p),'cache_sha256':sha(p),'context':context,'stage':kwargs['stage'],
                       'error_type':None,'source':'new_request','cache_hit':False}
                self.events.append(event);append(folder/'logical_events.jsonl',{'recorded_utc':'fixture',**event})
                save_new(folder/'request_intents'/(key+'.json'),{'signature':sig,'request_hash':key,'context':context,'stage':kwargs['stage'],'retries':2})
                append(folder/'api_calls.jsonl',{**identity,'request_hash':key,'context':context,'stage':kwargs['stage'],'cache_hit':False,'attempt':1,'status':200})
                return parsed
        native=self.new/'work/native/c2';out=self.new/'results/c2';api=MethodRecorder()
        with patch.object(adapter._runtime,'_get_encoder',return_value=FakeEncoder()):
            state=adapter.build(conversation(),native,api)
            memory=archive.base.persist_memory(state,native,out);save_new(out/'build.json',{'status':'complete','memory':memory})
            retrieval=adapter.retrieve('Where did A walk?',state,api)
        (self.new/'work/prefix_consumption.jsonl').write_text('')
        # Obtain actual frozen Reader/Judge captures once with the native context.
        q,_,Recorder,metric,calls=self.qa_fixture()
        with patch.object(archive.base,'BaselineAPI',Recorder),patch.object(archive.base,'verified_official_metric',return_value=(metric,{})):
            archive.base.qa_one(self.new,'amem',{**q,'query_id':'seed','conversation':2},retrieval)
        seed=self.new/'results/firstpass/seed'
        captures={kind:archive.read(seed/(kind+'.json')) for kind in ('reader','judge')}
        original_raw={kind:Path(value['events'][0]['cache_path']).read_bytes() for kind,value in captures.items()}
        # Fixture precursor files are not part of the synthetic complete inventory.
        for p in native.rglob('*'):
            if p.is_file():documents[str(p)]=p.read_bytes()
        for p in (self.new/'work/api_method').rglob('*'):
            if p.is_file():documents[str(p)]=p.read_bytes()
        documents[str(self.new/'work/prefix_consumption.jsonl')]=b''
        documents[str(out/'build.json')]=(out/'build.json').read_bytes()
        for p in (out/'memory').rglob('*'):
            if p.is_file():documents[str(p)]=p.read_bytes()
        ledgers={};events={};intents={}
        routes={str(ci):str(roots['v4'] if ci==2 else roots['v2'] if ci in (0,3,6,8) else roots['v3']) for ci in range(10)}
        for ordinal in range(1540):
            ci=ordinal%10;index=ordinal//10;current={**q,'query_id':f'c{ci}_q{index}','conversation':ci};questions.append(current)
            owner=Path(routes[str(ci)]);leaf=owner/'results/firstpass'/current['query_id'];ret=owner/'results'/f'c{ci}'/'retrieval'/(current['query_id']+'.json')
            for kind,original in captures.items():
                stage=copy.deepcopy(original);event=stage['events'][0]
                context={'method':'amem','conversation':ci,'query_id':current['query_id'],'phase':'firstpass'}
                raw=json.loads(original_raw[kind]);key=event['request_hash'];folder=owner/'work'/('api_'+kind);rp=folder/'api_cache'/(key+'.json')
                event.update(context=context,cache_path=str(rp),cache_sha256=archive.digest_bytes(original_raw[kind]))
                documents[str(rp)]=original_raw[kind];documents[str(leaf/(kind+'.json'))]=encode(stage)
                documents[str(leaf/(kind+'.intent.json'))]=encode({'context':context})
                identity={k:raw['request'][k] for k in ('provider','model','operation')};seen=str(rp) in intents
                if not seen:
                    intents[str(rp)]=True;documents[str(folder/'request_intents'/(key+'.json'))]=encode({'signature':raw['request'],'request_hash':key,'context':context,'stage':event['stage'],'retries':2})
                ledgers.setdefault(str(folder),[]).append({**identity,'request_hash':key,'context':context,'stage':event['stage'],'cache_hit':seen,'attempt':1,'status':200})
                events.setdefault(str(folder),[]).append({**event,**identity,'source':'existing_run_cache' if seen else 'new_request','cache_hit':seen})
            outcome={'query_id':current['query_id'],'conversation':ci,'method':'amem','phase':'firstpass','reader':'complete','judge':'complete',
                     'reader_path':str(leaf/'reader.json'),'judge_path':str(leaf/'judge.json')}
            documents[str(leaf/'outcome.json')]=encode(outcome);documents[str(ret)]=encode({'status':'complete','value':retrieval})
            sources.append({'query_id':current['query_id'],'conversation':ci,'owner':str(owner),'leaf':str(leaf),'retrieval_path':str(ret),'outcome':outcome,
                            'bindings':{str(p):archive.digest_bytes(documents[str(p)]) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}})
            value={**current,'source_owner':str(owner),'source_leaf':str(leaf),'retrieval_source':str(ret),'answer':'park','judge_correct':True,'official_f1':1.0,
                   'custom_metrics':archive.base.evaluator.lexical_metrics('park','park'),'reader_status':'complete','judge_status':'complete'}
            firstpass.append({**value,'phase':'firstpass'});effective.append({**value,'phase':'effective'})
        for folder,rows in ledgers.items():
            documents[str(Path(folder)/'api_calls.jsonl')]=b''.join(encode(r) for r in rows) # replaced with one JSON per line below
            documents[str(Path(folder)/'api_calls.jsonl')]=''.join(json.dumps(r)+'\n' for r in rows).encode()
            documents[str(Path(folder)/'logical_events.jsonl')]=''.join(json.dumps(r)+'\n' for r in events[folder]).encode()
        proof_path=self.root/'synthetic_proof.json'
        proof={'self_root':str(self.new),'conversation_sources':routes,'paid_replacement_permissions':[],
               'conversations':{str(ci):{'mode':'rebuild' if ci==2 else 'inherit','prefix':[]} for ci in range(10)},
               'history_roots':{k:str(v) for k,v in roots.items() if k!='v4'},'history_inventories':{},'history_manifests':{}}
        for name,folder in roots.items():
            if name=='v4':continue
            documents[str(folder/'run_manifest.json')]=encode({'synthetic_history':name})
            proof['history_manifests'][name]=archive.digest_bytes(documents[str(folder/'run_manifest.json')])
            proof['history_inventories'][name]={str(Path(p).relative_to(folder)):archive.digest_bytes(raw) for p,raw in documents.items() if Path(p).is_relative_to(folder)}
        documents[str(proof_path)]=encode(proof)
        m={'questions':questions,'self_root':str(self.new),'conversation_sources':routes,'code_sha256':{},'upstream_sha256':{},
           'technical_continuation':{'rebuild_conversations':[2],'proof':{'path':str(proof_path),'sha256':archive.digest_bytes(documents[str(proof_path)])}}}
        for path,value in ((self.new/'run_manifest.json',m),(self.new/'results/firstpass_sources.json',sources),
                           (self.new/'results/firstpass_per_query.json',firstpass),(self.new/'results/per_query.json',effective)):
            documents[str(path)]=encode(value)
        source_sha=archive.digest_bytes(documents[str(self.new/'results/firstpass_sources.json')])
        plan={'round':1,'full_cohort':1540,'firstpass_sources_sha256':source_sha,'jobs':[]}
        documents[str(self.new/'qa_recovery_plan.json')]=encode(plan)
        documents[str(self.new/'qa_recovery_outcome.json')]=encode({'firstpass_sources_sha256':source_sha,'plan_sha256':archive.digest_bytes(encode(plan)),'results':[]})
        # Build expected frozen audit/usage seals against the same virtual inventory.
        class MemoryResolver:
            manifest={'roots':{k:str(v) for k,v in roots.items()},'records':[{'path':p,'sha256':archive.digest_bytes(raw)} for p,raw in documents.items()]}
            def bytes(self,p):return documents[str(p)]
            def read(self,p):return json.loads(self.bytes(p))
            def sha(self,p):return archive.digest_bytes(self.bytes(p))
        r=MemoryResolver();P=archive.archived_path_class(r)
        common={'NEW':P(self.new),'Path':P,'read':r.read,'sha':r.sha,'_PROOF':proof}
        def bound(p,f):
            p=P(p)
            if not p.is_relative_to(f) or not p.is_file():raise ValueError('foreign synthetic native')
            return p
        raw_audit=archive.clone(archive.continuation.controller.audit_new_requests,**common)()
        native_audit=archive.clone(archive.continuation.controller.audit_native_outputs,**common,bound_path=bound)(m)
        usage=archive.clone(archive.continuation.usage_union,**common,HISTORY={k:P(v) for k,v in roots.items() if k!='v4'})()
        summary={'schema':'five-baseline-result-native-continuation-v4','method':'amem','audit_status':'no_invalid_artifacts','created_utc':'fixture',
                 'firstpass':life.aggregate(firstpass),'effective':life.aggregate(effective),'usage':usage,'self_root':str(self.new),
                 'conversation_sources':routes,'source_manifests':proof['history_manifests'],'continuation_proof':m['technical_continuation']['proof']}
        for name,value in (('summary.json',summary),('raw_request_audit.json',raw_audit),('native_output_audit.json',native_audit)):
            documents[str(self.new/'results'/name)]=encode(value)
        records=[];blobs={}
        for path,raw in documents.items():
            h=archive.digest_bytes(raw);blobs[h]=raw;records.append({'path':path,'sha256':h,'size':len(raw),'retained':True,'member':'blobs/'+h})
        mp=self.root/'cohort_manifest.json';zp=self.root/'cohort.zip'
        save_new(mp,{'schema':archive.SCHEMA,'roots':{k:str(v) for k,v in roots.items()},'records':records})
        with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('archive_manifest.json',mp.read_bytes())
            for h,raw in blobs.items():z.writestr('blobs/'+h,raw)
        for folder in roots.values():
            work=folder/'work'
            if work.exists():self.assertTrue(work.resolve().is_relative_to(self.root.resolve()));shutil.rmtree(work)
        with patch.object(archive.base,'questions',return_value=questions),patch.object(archive.base,'verified_official_metric',return_value=(metric,{})), \
             patch.object(archive.continuation,'HISTORY',{k:v for k,v in roots.items() if k!='v4'}), \
             patch.object(archive,'audit_historical_source',return_value=({'status':'passed'},{'status':'passed'})):
            with archive.Resolver(zp,mp) as resolver:
                result=archive.replay_all(resolver);self.assertEqual(result['firstpass_and_effective_leaves'],3080)
                original_read=resolver.read
                bad=copy.deepcopy(effective);bad[2]['source_owner']=str(roots['v3'])
                with patch.object(resolver,'read',side_effect=lambda p:bad if Path(p)==self.new/'results/per_query.json' else original_read(p)):
                    with self.assertRaisesRegex(ValueError,'seal output differs'):archive.replay_all(resolver)

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

    def test_resolver_never_trusts_false_retention_for_unique_text(self):
        p=archive.ROOTS['v3']/'work/native/c2/sessions/session_1/note_000001.json'
        save_new(p,{'unique':'partial evolution evidence'})
        zp,mp=self.pack([p]);value=archive.read(mp)
        value['records'][0].update(retained=False,disposal_reason='original MiniLM numeric embedding cache; SHA and size retained')
        value['records'][0].pop('member');mp.write_text(json.dumps(value),encoding='utf-8')
        with zipfile.ZipFile(zp,'w') as z:z.writestr('archive_manifest.json',mp.read_bytes())
        with self.assertRaisesRegex(ValueError,'exact numeric-cache policy'):archive.Resolver(zp,mp)

    def test_resolver_disposal_requires_boolean_exact_name_and_reason(self):
        p=archive.ROOTS['v3']/'work/native/c2/cache/embeddings.npy';p.parent.mkdir(parents=True);p.write_bytes(b'numeric fixture')
        zp,mp=self.pack([p]);original=archive.read(mp)
        for retained,reason,member in ((0,archive.disposable(p),False),(False,'wrong reason',False),(False,archive.disposable(p),True)):
            with self.subTest(retained=retained,reason=reason,member=member):
                value=copy.deepcopy(original);row=value['records'][0];row.update(retained=retained,disposal_reason=reason)
                if not member:row.pop('member')
                mp.write_text(json.dumps(value),encoding='utf-8')
                with zipfile.ZipFile(zp,'w') as z:z.writestr('archive_manifest.json',mp.read_bytes())
                with self.assertRaisesRegex(ValueError,'boolean|numeric-cache policy'):archive.Resolver(zp,mp)
        value=copy.deepcopy(original);value['records'][0].update(retained=False,disposal_reason=archive.disposable(p));value['records'][0].pop('member')
        mp.write_text(json.dumps(value),encoding='utf-8')
        with zipfile.ZipFile(zp,'w') as z:z.writestr('archive_manifest.json',mp.read_bytes())
        with archive.Resolver(zp,mp) as resolver:
            with self.assertRaisesRegex(ValueError,'not retained'):resolver.bytes(p)

    def test_retained_text_missing_member_or_blob_is_rejected(self):
        zp,mp=self.pack([self.raw]);original=archive.read(mp)
        for remove_member in (True,False):
            value=copy.deepcopy(original)
            if remove_member:value['records'][0].pop('member')
            mp.write_text(json.dumps(value),encoding='utf-8')
            with zipfile.ZipFile(zp,'w') as z:z.writestr('archive_manifest.json',mp.read_bytes())
            with self.assertRaisesRegex(ValueError,'Content address|ZIP entries'):archive.Resolver(zp,mp)

    def test_resolver_rejects_payload_sha_mismatch_even_when_zip_crc_valid(self):
        zp,mp=self.pack([self.raw]);manifest=archive.read(mp);member=manifest['records'][0]['member']
        with zipfile.ZipFile(zp,'w') as z:
            z.writestr('archive_manifest.json',mp.read_bytes());z.writestr(member,b'changed bytes with valid CRC')
        with self.assertRaisesRegex(ValueError,'blob SHA'):archive.Resolver(zp,mp)

    def test_disposal_is_limited_to_native_sessions_and_encoder_cache(self):
        native=self.source/'work/native/c1'
        self.assertIsNone(archive.disposable(native/'sessions/session_1/note_000001.json'))
        self.assertIsNotNone(archive.disposable(native/'cache/embeddings.npy'))
        self.assertIsNotNone(archive.disposable(native/'cache/embeddings.tmp.npy'))
        for p in (native/'cache/audit.json',native/'cache/error.txt',native/'cache/index.bin',
                  native/'cache/other/embeddings.npy',native/'calls/note_000001.json',native/'memory_updates.jsonl',
                  native/'checkpoint.json.tmp',self.source/'work/api_method/api_cache/raw.json'):
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
        for folder in list(archive.ROOTS.values())[:3]:save_new(folder/'run_manifest.json',{'code_sha256':{},'upstream_sha256':{}})
        discarded=self.source/'work/native/c0/cache/embeddings.npy';discarded.parent.mkdir(parents=True);discarded.write_bytes(b'disposable fixture vectors')
        evidence=self.root/'preflight.json';save_new(evidence,{'fixture':True})
        source_proof=self.root/'terminal.json';save_new(source_proof,{'history_inventories':{
            name:{str(Path(p).relative_to(folder)):h for p,h in archive.file_inventory(folder).items()}
            for name,folder in archive.ROOTS.items() if name!='v4'}})
        dataset=self.root/'data/locomo10.json';save_new(dataset,{'offline_fixture':True})
        save_new(self.root/'configs/sdk20_full1540_locked_v1.json',{'offline_fixture':True})
        save_new(self.new/'run_manifest.json',{'dataset_sha256':sha(dataset),'code_sha256':{},'upstream_sha256':{},
                  'technical_continuation':{'proof':archive.binding(source_proof)}})
        summary=self.new/'results/summary.json';save_new(summary,{'schema':'five-baseline-result-native-continuation-v4','method':'amem','self_root':str(self.new),'audit_status':'no_invalid_artifacts','effective':{'slots':1540,'reader_valid':1540,'judge_valid':1540}})
        save_new(self.new/'status.json',{'stage':'completed_awaiting_result_review','summary_sha256':sha(summary)})
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
            save_new(archive.output_dir()/'cleanup_execution_intent.json',{'schema':'amem-four-root-cleanup-intent-v1',
                 'cleanup_plan_sha256':sha(plan_path),'cleanup_review_sha256':sha(cleanup_review),
                 'targets':[{'path':p} for p in plan['allowed_delete_roots']]})
            for name,folder in archive.ROOTS.items():
                work=(folder/'work').resolve();self.assertTrue(work.is_relative_to(self.root.resolve()))
                self.assertEqual(work.name,'work');shutil.rmtree(work)
                save_new(archive.output_dir()/('cleanup_'+name+'.json'),{'schema':'amem-four-root-removal-receipt-v1',
                     'removed_path':str(work),'cleanup_plan_sha256':sha(plan_path),'cleanup_review_sha256':sha(cleanup_review)})
            self.assertEqual(archive.complete(plan_path,cleanup_review)['status'],'complete')
            self.assertEqual(archive.verify_completion(plan_path)['status'],'passed')
            (self.source/'work').mkdir()
            with self.assertRaisesRegex(ValueError,'reappeared'):archive.verify_completion(plan_path)

    def test_actual_powershell_path_gate_supports_file_and_directory_without_deletion(self):
        import subprocess
        script=Path(archive.__file__).with_name('cleanup_amem_continuation_v4.ps1')
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
        self.ancestor=run/'amem';self.source=run/'amem_format_v2';self.new=run/'amem_native_v4'
        roots={'v1':self.ancestor,'v2':self.source,'v3':run/'amem_native_v3','v4':self.new}
        with patch.object(archive,'ROOTS',roots),patch.object(archive,'stopped'),patch.object(archive,'verify_native_seal'),patch.object(archive,'replay_all',return_value={'status':'passed','offline_fixture_only':True,'model_calls':0}):
            result_review,tool_review,discarded=self.archive_fixture();result=archive.archive(result_review,tool_review)
            plan_path=Path(result['plan']);plan=archive.read(plan_path)
            cleanup=self.root/'cleanup_review.json';save_new(cleanup,{'status':'passed','offline_fixture_only':True,
                   'cleanup_plan_sha256':sha(plan_path),'archive_sha256':plan['archive']['sha256']})
            stub=self.root/'OFFLINE_FIXTURE_verifier.py'
            stub.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(actual))+")\n"
                 "from tools import archive_amem_continuation_v4 as x\nx.ROOT=Path("+repr(str(self.root))+")\n"
                 "x.ROOTS="+repr({k:str(v) for k,v in roots.items()})+"\nx.ROOTS={k:Path(v) for k,v in x.ROOTS.items()}\n"
                 "x.stopped=lambda:None\nx.replay_all=lambda r:{'status':'passed','offline_fixture_only':True,'model_calls':0}\nx.main()\n",encoding='utf-8')
            quote=lambda x:"'"+str(x).replace("'","''")+"'"
            script=(actual/'tools/cleanup_amem_continuation_v4.ps1').read_text(encoding='utf-8')
            script=script.replace("(Resolve-Path -LiteralPath 'D:\\jev mem').Path","(Resolve-Path -LiteralPath "+quote(self.root)+").Path")
            script=script.replace("$python = Join-Path $workspace 'runtime\\baseline_env_v1\\Scripts\\python.exe'",'$python = '+quote(sys.executable))
            script=script.replace("$helper = Join-Path $workspace 'tools\\archive_amem_continuation_v4.py'",'$helper = '+quote(stub))
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



if __name__=='__main__':
    import unittest
    unittest.main()
