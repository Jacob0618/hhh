"""Offline counterexamples for v3 transport and cross-root result routing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_five_baselines_amem_native_v3 as v3
from tools.baselines_common_v1 import BaselineFatal, save_new, sha
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body

ORIGINAL_REQUEST=v3.BaselineAPI.request


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.new=self.root/'new';self.ancestor=self.root/'ancestor'
        self.context={'method':'amem','conversation':0,'phase':'build'}
        self.sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
                  'body':generation_body('same original prompt',system='You must respond with a JSON object.',json_output=True,max_output=16384)}
        self.key=digest(self.sig);self.raw=self.source/'work/api_method/api_cache'/(self.key+'.json')
        save_new(self.raw,{'request':self.sig,'response':{'fixture':'original'}})
        self.review=self.root/'review.json';save_new(self.review,{'status':'passed'})
        self.entry={'request_hash':self.key,'signature':self.sig,'stage':'amem_analysis','context':self.context,
                    'raw_path':str(self.raw),'raw_sha256':sha(self.raw),'mode':'inherit'}
        self.proof={'conversations':{str(i):{'mode':'rebuild' if i==0 else 'inherit','prefix':[self.entry] if i==0 else []} for i in range(10)}}
        self.patches=[patch.object(v3,'ROOT',self.root),patch.object(v3,'SOURCE',self.source),patch.object(v3,'NEW',self.new),
                      patch.object(v3,'ANCESTOR',self.ancestor),patch.object(v3,'_PROOF',self.proof)]
        for p in self.patches:p.start()
        self.dispatched=[]
        def fake_request(api,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
            signature={'provider':provider,'model':model,'operation':operation,'body':body};key=digest(signature)
            target=api.run_dir/'api_cache'/(key+'.json')
            if not target.exists():
                self.dispatched.append(key)
                save_new(api.run_dir/'request_intents'/(key+'.json'),{'request_hash':key,'signature':signature})
                save_new(target,{'request':signature,'response':{'fixture':'replacement'}})
            api.events.append({'request_hash':key,'cache_path':str(target),'cache_sha256':sha(target),
                               'context':copy.deepcopy(api.context),'stage':stage,'error_type':None})
            return json.loads(target.read_text())['response']
        self.patches.append(patch.object(v3.BaselineAPI,'request',fake_request));self.patches[-1].start()
        # Any unexpectedly unmocked physical network attempt or credential read
        # is forbidden, including through an inherited provider implementation.
        self.patches.append(patch('requests.sessions.Session.request',side_effect=AssertionError('network forbidden')));self.patches[-1].start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def api(self,context=None):
        return v3.NativeAPI(self.new/'work/api_method',keys={'gemini':'offline-placeholder'},context=context or self.context)

    def request(self,api,body=None):
        return api.request('gemini','gemini-2.5-flash',body or self.sig['body'],stage='amem_analysis')

    def replacement(self):
        self.entry['mode']='replace_once';self.entry['failure_certificate']={'status':'passed','kind':'known_complete_invalid_json',
            'review_path':str(self.review),'review_sha256':sha(self.review),'original_raw_sha256':sha(self.raw),'retry_limit':1}

    def test_full_signature_prefix_import_has_zero_dispatch_and_exact_lineage(self):
        api=self.api();self.assertEqual(self.request(api),{'fixture':'original'});self.assertEqual(self.dispatched,[])
        lineage=v3.read(api.run_dir/'cache_lineage'/(self.key+'.json'))
        self.assertEqual(lineage['source_sha256'],sha(self.raw));self.assertEqual(api.position,1)
        self.assertEqual((api.run_dir/'api_cache'/self.raw.name).read_bytes(),self.raw.read_bytes())

    def native_audit_fixture(self):
        from tools import baselines_amem_v3 as amem
        from tests.test_baselines_amem_v1 import FakeEncoder,FakeAPI,conversation
        outer=self
        class Recorder(FakeAPI):
            def __init__(self):super().__init__();self.events=[]
            def generate(self,prompt,**kwargs):
                value=super().generate(prompt,**kwargs)
                if isinstance(value,str):value=json.loads(value)
                sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
                     'body':generation_body(prompt,system=kwargs.get('system'),json_output=kwargs['json_output'],max_output=kwargs['max_output'])}
                key=digest(sig);raw={'request':sig,'response':{'modelVersion':sig['model'],'candidates':[
                    {'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':json.dumps(value)}]}}]}}
                path=outer.new/'work/api_method/api_cache'/(key+'.json')
                if not path.exists():save_new(path,raw)
                parsed,journal=amem.decode_generation_json(raw,raw_artifact_sha256=sha(path))
                lineage=outer.new/'work/api_method/parser_lineage'/(key+'.json')
                if not lineage.exists():save_new(lineage,journal)
                context=outer.context if kwargs['stage']!='amem_query_keywords' else {**outer.context,'phase':'retrieval','query_id':'c0_q0'}
                event={'request_hash':key,'cache_path':str(path),'cache_sha256':sha(path),
                                    'context':context,'stage':kwargs['stage'],'error_type':None}
                self.events.append(event);v3.append(outer.new/'work/api_method/logical_events.jsonl',{'recorded_utc':v3.now(),**event})
                return parsed
        native=self.new/'work/native/c0';out=self.new/'results/c0';api=Recorder()
        q={'query_id':'c0_q0','conversation':0,'question':'Where was the friend?'}
        with patch.object(amem._runtime,'_get_encoder',return_value=FakeEncoder()):
            state=amem.build(conversation(),native,api)
            memory=v3.base.persist_memory(state,native,out)
            save_new(out/'build.json',{'status':'complete','memory':memory})
            result=amem.retrieve(q['question'],state,api)
            save_new(out/'retrieval'/(q['query_id']+'.json'),{'status':'complete','value':result})
        return {'technical_continuation':{'rebuild_conversations':[0]},'questions':[q]}

    def test_native_audit_replays_full_method_calls_journals_exports_and_retrieval(self):
        m=self.native_audit_fixture();result=v3.audit_native_outputs(m)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['generation_calls']),5)
        p=next((self.new/'work/api_method/parser_lineage').glob('*.json'))
        before=p.read_bytes();p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'parser lineage'):v3.audit_native_outputs(m)
        p.write_bytes(before)
        call=next((self.new/'work/native/c0/calls').glob('*.json'));value=v3.read(call)
        value['parsed_response']['injected']='invalid';call.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'derived response'):v3.audit_native_outputs(m)

    def test_native_lineage_changed_context_is_rejected_even_with_same_raw_signature(self):
        m=self.native_audit_fixture();v3.audit_native_outputs(m)
        path=next((self.new/'work/native/c0/calls').glob('*.json'));value=v3.read(path)
        value['transport_lineage']['context']['conversation']=9;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'original logical event'):v3.audit_native_outputs(m)

    def test_duplicate_exact_prefix_consumers_keep_two_ordinals_one_raw_zero_dispatch(self):
        self.proof['conversations']['0']['prefix'].append(copy.deepcopy(self.entry))
        api=self.api();self.request(api);self.request(api)
        rows=[json.loads(x) for x in (self.new/'work/prefix_consumption.jsonl').read_text().splitlines()]
        self.assertEqual([r['ordinal'] for r in rows],[0,1]);self.assertEqual(self.dispatched,[])
        self.assertEqual(len(list((api.run_dir/'api_cache').glob('*.json'))),1)

    def test_final_raw_audit_accepts_real_cache_two_consumers_and_rejects_context_tamper(self):
        self.proof['conversations']['0']['prefix'].append(copy.deepcopy(self.entry))
        with patch.object(v3.BaselineAPI,'request',ORIGINAL_REQUEST):
            api=self.api();self.request(api);self.request(api)
        self.assertEqual(v3.audit_new_requests()['status'],'passed')
        path=api.run_dir/'api_calls.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[-1]['context']['conversation']=9
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'attribution'):v3.audit_new_requests()

    def test_final_raw_audit_binds_replacement_certificate_and_physical_history(self):
        self.replacement();api=self.api();self.request(api)
        intent_path=api.run_dir/'request_intents'/(self.key+'.json')
        intent=v3.read(intent_path);intent.update(context=self.context,stage='amem_analysis',retries=2,created_utc=v3.now())
        intent_path.write_text(json.dumps(intent))
        identity={k:self.sig[k] for k in ('provider','model','operation')}
        event={**api.events[0],**identity,'source':'new_request'}
        v3.append(api.run_dir/'logical_events.jsonl',event)
        physical={'request_hash':self.key,**identity,'context':self.context,'stage':'amem_analysis','attempt':1,'status':200,'cache_hit':False}
        v3.append(api.run_dir/'api_calls.jsonl',physical)
        self.assertEqual(v3.audit_new_requests()['status'],'passed')
        path=api.run_dir/'replacements'/(self.key+'.json');before=path.read_bytes();rec=v3.read(path)
        rec['prefix_ordinal']=999;path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError,'Replacement provenance'):v3.audit_new_requests()
        path.write_bytes(before)
        physical['attempt']=2;(api.run_dir/'api_calls.jsonl').write_text(json.dumps(physical)+'\n')
        with self.assertRaisesRegex(ValueError,'attempt/consumer'):v3.audit_new_requests()

    def test_changed_full_body_and_context_never_dispatch(self):
        body=copy.deepcopy(self.sig['body']);body['generationConfig']['temperature']=0
        with self.assertRaises(BaselineFatal):self.request(self.api(),body)
        with self.assertRaises(BaselineFatal):self.api({**self.context,'conversation':1})
        self.assertEqual(self.dispatched,[])

    def test_missing_or_changed_source_raw_never_dispatch(self):
        self.raw.write_text('{}')
        with self.assertRaises(ValueError):self.request(self.api())
        self.raw.unlink()
        with self.assertRaises(ValueError):self.request(self.api())
        self.assertEqual(self.dispatched,[])

    def test_replacement_is_consumed_before_transport_and_cannot_repeat(self):
        self.replacement();api=self.api();self.assertEqual(self.request(api),{'fixture':'replacement'})
        consumed=api.run_dir/'replacements'/(self.key+'.json')
        self.assertEqual(v3.read(consumed)['original_raw_sha256'],sha(self.raw));self.assertEqual(self.dispatched,[self.key])
        with self.assertRaises(FileExistsError):self.request(self.api())
        self.assertEqual(self.dispatched,[self.key]);self.assertEqual(v3.read(self.raw)['response'],{'fixture':'original'})

    def test_unknown_replacement_transport_outcome_remains_consumed(self):
        self.replacement()
        with patch.object(v3.BaselineAPI,'request',side_effect=BaselineFatal('unknown outcome')):
            with self.assertRaises(BaselineFatal):self.request(self.api())
        with self.assertRaises(FileExistsError):self.request(self.api())
        self.assertEqual(self.dispatched,[])

    def test_known_request_after_prefix_cannot_be_redrawn(self):
        api=self.api();self.request(api)
        with self.assertRaises(BaselineFatal):self.request(api)
        self.assertEqual(self.dispatched,[])

    def test_first_unseen_request_after_prefix_may_use_original_transport(self):
        api=self.api();self.request(api)
        body=generation_body('first unseen prompt',system='You must respond with a JSON object.',json_output=True,max_output=16384)
        self.assertEqual(self.request(api,body),{'fixture':'replacement'});self.assertEqual(len(self.dispatched),1)

    def test_successful_conversation_routes_readonly_to_source(self):
        self.proof['conversations']['0']={'mode':'inherit','prefix':[]}
        q={'conversation':0,'query_id':'c0_q0'};folder,leaf,ret=v3.source_for(q)
        self.assertEqual(folder,self.source);self.assertEqual(leaf,self.source/'results/firstpass/c0_q0')
        with self.assertRaises(BaselineFatal):self.api()
        self.assertFalse((self.new/'work/native').exists())

    def test_usage_counts_physical_same_hash_attempts_and_not_lineage_hits(self):
        for folder,records in ((self.ancestor,[{'request_hash':'same','status':200,'usage':{'promptTokenCount':3}}]),
                               (self.source,[{'request_hash':'same','cache_hit':True},{'request_hash':'same','status':'transport_error','usage_unknown':True}]),
                               (self.new,[{'request_hash':'same','status':200,'usage':{'promptTokenCount':5}}])):
            p=folder/'work/api_method/api_calls.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        value=v3.usage_union();self.assertEqual(value['physical_attempts'],3)
        self.assertEqual(value['known_partial_tokens']['promptTokenCount'],8);self.assertEqual(value['usage_unknown_or_missing'],1)

    def test_execution_intent_prevents_even_manifest_validation_reentry(self):
        save_new(self.new/'execution_intent.json',{'prior':True})
        with patch.object(v3,'manifest',side_effect=AssertionError('must not inspect or alter prior run')):
            with self.assertRaises(FileExistsError):v3.run()

    def test_partial_work_cannot_resume_under_fresh_v3_controller(self):
        save_new(self.new/'work/native/c0/checkpoint.json',{'completed_notes':10})
        with patch.object(v3,'manifest',side_effect=AssertionError('must fail before source/provider setup')):
            with self.assertRaises(FileExistsError):v3.run()
        self.assertFalse((self.new/'execution_intent.json').exists())

    def test_ancestor_only_known_request_cannot_be_redrawn_after_prefix(self):
        api=self.api();self.request(api)
        body=generation_body('ancestor-only known prompt',json_output=True,max_output=16384)
        sig={**self.sig,'body':body};key=digest(sig)
        save_new(self.ancestor/'work/api_method/api_cache'/(key+'.json'),{'request':sig,'response':{'fixture':'older'}})
        with self.assertRaises(BaselineFatal):self.request(api,body)
        self.assertEqual(self.dispatched,[])

    def test_source_process_terminal_check_failure_precedes_source_io(self):
        p=self.root/'proof.json';save_new(p,{'schema':v3.SCHEMA,'status':'passed','source_run':str(self.source),'ancestor_run':str(self.ancestor)})
        with patch.object(v3,'assert_source_stopped',side_effect=ValueError('active producer')):
            with self.assertRaisesRegex(ValueError,'active producer'):v3.validate_proof(p)

    def test_wrong_namespace_or_unadmitted_build_precedes_credentials_and_mkdir(self):
        with patch('jev_mem.providers.load_credentials',side_effect=AssertionError('credential read forbidden')):
            with self.assertRaises(BaselineFatal):
                v3.NativeAPI(self.root/'wrong',context=self.context)
            with patch.object(v3,'_PROOF',None):
                with self.assertRaises(BaselineFatal):v3.NativeAPI(self.new/'work/api_method',context=self.context)
            with self.assertRaises(BaselineFatal):
                v3.NativeAPI(self.new/'work/api_method',context={**self.context,'conversation':1})
        self.assertFalse((self.root/'wrong').exists());self.assertFalse(self.new.exists())

    def test_manifest_rechecks_review_specification_source_ancestor_and_routing(self):
        from tools import baselines_amem_v3 as amem
        proof={**self.proof,'source_manifest_sha256':'source-sha','ancestor_manifest_sha256':'ancestor-sha'}
        proof_path=self.root/'terminal-proof.json';save_new(proof_path,proof)
        code={'dispatcher.py':'unused'};(self.root/'dispatcher.py').write_text('offline fixture')
        code['dispatcher.py']=sha(self.root/'dispatcher.py')
        review=self.root/'checked-review.json';save_new(review,{'status':'passed','code_sha256':code,'upstream_sha256':{},'proof_sha256':sha(proof_path)})
        old={'code_sha256':{},'upstream_sha256':{}};save_new(self.source/'run_manifest.json',old)
        m={'questions':[],'code_sha256':code,'upstream_sha256':{},'specification':{'frozen':True},'historical_cache':v3.HISTORICAL_CACHE,
           'review':{'path':str(review),'sha256':sha(review)},'technical_continuation':{
               'source_run':str(self.source),'source_manifest_sha256':'source-sha','ancestor_manifest_sha256':'ancestor-sha',
               'inherit_conversations':list(range(1,10)),'rebuild_conversations':[0],
               'proof':{'path':str(proof_path),'sha256':sha(proof_path)}}}
        path=self.new/'run_manifest.json';save_new(path,m)
        with patch.object(v3,'EXTRA',('dispatcher.py',)),patch.object(v3,'validate_proof',return_value=proof),patch.object(v3.base,'questions',return_value=[]),patch.object(amem,'describe',return_value={'frozen':True}):
            self.assertEqual(v3.manifest(return_proof=True),(m,proof))
            for kind in ('source_run','ancestor_manifest_sha256','rebuild_conversations','specification','review'):
                value=copy.deepcopy(m)
                if kind=='source_run':value['technical_continuation'][kind]=str(self.ancestor)
                elif kind=='ancestor_manifest_sha256':value['technical_continuation'][kind]='wrong'
                elif kind=='rebuild_conversations':value['technical_continuation'][kind]=[]
                elif kind=='specification':value[kind]={'frozen':False}
                else:
                    checked=v3.read(review);checked['status']='needs_review';review.write_text(json.dumps(checked));value['review']['sha256']=sha(review)
                path.write_text(json.dumps(value))
                with self.subTest(kind=kind),self.assertRaises(ValueError):v3.manifest()

    def qa_fixture(self):
        q={'query_id':'c0_q0','conversation':0,'question':'Where did A walk?','reference':'park','category':1,'reference_date':'2023-05-08'}
        calls=[]
        class Recorder:
            def __init__(self,run_dir,*,context):self.run_dir=Path(run_dir);self.context=context;self.events=[]
            def _redact(self,value):return value
            def generate(self,prompt,*,model,system=None,json_output=False,max_output,stage):
                signature={'provider':'gemini','model':model,'operation':'generateContent',
                           'body':generation_body(prompt,system=system,json_output=json_output,max_output=max_output)}
                key=digest(signature);text='park' if stage==v3.base.evaluator.READER_STAGE else '[[yes]]'
                raw={'request':signature,'response':{'modelVersion':model,'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}]}}
                p=self.run_dir/'api_cache'/(key+'.json');save_new(p,raw)
                self.events.append({'request_hash':key,'cache_path':str(p),'cache_sha256':sha(p),'stage':stage,'context':self.context,'error_type':None})
                calls.append((self.context['phase'],stage));return text
        metric=lambda answer,reference,category:float(answer==reference)
        ret=self.source/'results/c0/retrieval/c0_q0.json';save_new(ret,{'status':'complete','value':{'context':'A walked in the park.'}})
        with patch.object(v3.base,'BaselineAPI',Recorder),patch.object(v3.base,'verified_official_metric',return_value=(metric,{})):
            v3.base.qa_one(self.source,'amem',q,{'context':'A walked in the park.'})
        return q,ret,Recorder,metric,calls

    def test_cross_root_judge_only_recovery_reuses_reader_and_exact_replays(self):
        q,ret,Recorder,metric,calls=self.qa_fixture()
        original=self.source/'results/firstpass'/q['query_id'];reader=v3.read(original/'reader.json')
        with patch.object(v3.base,'BaselineAPI',Recorder),patch.object(v3.base,'verified_official_metric',return_value=(metric,{})):
            v3.base.qa_one(self.new,'amem',q,v3.read(ret)['value'],phase='recovery',inherited_reader=reader)
            r,j=v3.replay_leaf(q,self.new/'results/recovery'/q['query_id'],self.new,self.source,original,ret)
        self.assertTrue(r['inherited']);self.assertTrue(j['value']['correct'])
        self.assertEqual(calls,[('firstpass',v3.base.evaluator.READER_STAGE),('firstpass',v3.base.evaluator.JUDGE_STAGE),('recovery',v3.base.evaluator.JUDGE_STAGE)])

    def test_cross_root_replay_rejects_tampered_context_and_wrong_event_owner(self):
        q,ret,Recorder,metric,calls=self.qa_fixture();leaf=self.source/'results/firstpass'/q['query_id']
        with patch.object(v3.base,'verified_official_metric',return_value=(metric,{})):
            r,j=v3.replay_leaf(q,leaf,self.source,self.source,leaf,ret)
            changed=v3.read(ret);changed['value']['context']='A walked somewhere else.';ret.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):v3.replay_leaf(q,leaf,self.source,self.source,leaf,ret)
            changed['value']['context']='A walked in the park.';ret.write_text(json.dumps(changed))
            r['events'][0]['context']['conversation']=9;(leaf/'reader.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'ownership'):v3.replay_leaf(q,leaf,self.source,self.source,leaf,ret)

    def test_failed_saved_stage_with_valid_original_raw_requires_offline_review_not_redraw(self):
        q,ret,Recorder,metric,calls=self.qa_fixture();leaf=self.source/'results/firstpass'/q['query_id']
        stage=v3.read(leaf/'reader.json');stage['status']='failed'
        with self.assertRaisesRegex(ValueError,'valid result'):
            v3.base.known_recoverable(stage,self.source,'amem',q,v3.read(ret)['value'],kind='reader')
        self.assertEqual(len(calls),2)

    def test_incomplete_firstpass_never_reads_sources_or_creates_recovery_plan(self):
        with patch.object(v3,'read',side_effect=AssertionError('barrier must run first')):
            with self.assertRaisesRegex(ValueError,'1540'):v3.firstpass_sources({'questions':[]})
        self.assertFalse((self.new/'qa_recovery_plan.json').exists())

    def test_future_method_cannot_accept_summary_and_fake_cleanup_receipt_alone(self):
        save_new(self.new/'results/summary.json',{'audit_status':'no_invalid_artifacts'})
        save_new(self.new/'results/cleanup_receipt.json',{'placeholder':True})
        with self.assertRaises(FileNotFoundError):v3.require_reviewed_predecessor()


if __name__=='__main__':unittest.main()
