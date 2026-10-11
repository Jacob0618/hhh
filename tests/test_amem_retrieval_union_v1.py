"""Synthetic full cohort and exact raw replay; never calls a live provider."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools import seal_amem_retrieval_continuation_v1 as union
from jev_mem.gemini25_ablation_provider_v1 import generation_body
from jev_mem.sdk20_cached_api import digest


class UnionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='amem_union_offline_')
        self.root=Path(self.temp.name).resolve(); self.run=self.root/'runs'
        self.parent=self.run/'amem_native_v4';self.final=self.parent/'retrieval_continuation_v1';self.work=self.parent/'work/retrieval_continuation_v1'
        self.patches=[patch.object(union,'ROOT',self.root),patch.object(union,'PARENT',self.parent),
                      patch.object(union,'FINAL',self.final),patch.object(union,'WORK_OWNER',self.work),
                      patch.object(union.base,'RUN',self.run)]
        for p in self.patches:p.start()
        self.docs={};self.qs=[];self.sources=[]
        self.routes={str(ci):str(self.run/('amem_format_v2' if ci in (0,3,6,8) else 'amem_native_v4' if ci==2 else 'amem_native_v3')) for ci in range(10)}
        self.metric,_=union.base.verified_official_metric()
        for ordinal in range(1540):
            ci=ordinal%10;qi=ordinal//10
            q={'query_id':f'c{ci}_q{qi}','conversation':ci,'source_qa_index':qi,'question':'Where did A walk?',
               'reference':'park','category':1+ordinal%4,'reference_date':'2023-01-01'}
            self.qs.append(q);owner=Path(self.routes[str(ci)]);leaf=owner/'results/firstpass'/q['query_id'];rp=owner/'results'/f'c{ci}'/'retrieval'/(q['query_id']+'.json')
            if q['query_id']=='c2_q42':
                self.put(rp,{'status':'failed','error':{'type':'BaselineFatal','message':'known query JSON failure'}})
            else:
                self.make_qa(owner,q)
                self.sources.append({'query_id':q['query_id'],'conversation':ci,'owner':str(owner),'leaf':str(leaf),'retrieval_path':str(rp),
                    'bindings':{str(p):self.hash(p) for p in (rp,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}})
        q=next(q for q in self.qs if q['query_id']=='c2_q42');self.q=q
        self.make_qa(self.work,q)
        leaf,rp=union.technical_paths()
        self.docs[str(self.final/'results/technical_retrieval/c2_q42.json')]=self.docs[str(rp)]
        for p in list(self.docs):
            if Path(p).parent==leaf:self.docs[str(self.final/'results/technical_firstpass/c2_q42'/Path(p).name)]=self.docs[p]
        self.proof={'query_id':'c2_q42','original_routes':self.routes,'original_successes':self.sources,
                    'original_failure':{'retrieval':{'path':str(self.parent/'results/c2/retrieval/c2_q42.json'),
                                                    'sha256':self.hash(self.parent/'results/c2/retrieval/c2_q42.json')}}}
        proof_path=self.root/'proof.json';self.put(proof_path,self.proof)
        parent_path=self.parent/'run_manifest.json';self.put(parent_path,{'questions':self.qs,'conversation_sources':self.routes})
        self.manifest={'query_id':'c2_q42','final_root':str(self.final),'work_owner':str(self.work),'questions':self.qs,'code_sha256':{},
                       'proof':{'path':str(proof_path),'sha256':self.hash(proof_path)},'parent_manifest':{'path':str(parent_path),'sha256':self.hash(parent_path)}}
        # Scoped fixture method audit; real method-native audit has separate
        # author/independent tests. QA/replay/union logic is never mocked here.
        self.method_audit=types.SimpleNamespace(audit_sidecar=lambda *args,**kwargs:{'status':'passed','offline_method_fixture':True})
        self.entry_patch=patch.dict('sys.modules',{'tools.run_five_baselines_amem_retrieval_v1':self.method_audit})
        self.entry_patch.start()

    def tearDown(self):
        self.entry_patch.stop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def put(self,p,v):self.docs[str(p)]=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
    def hash(self,p):return hashlib.sha256(self.docs[str(p)]).hexdigest()

    def view(self):
        outer=self
        class View:
            manifest={'records':[{'path':p,'sha256':hashlib.sha256(v).hexdigest()} for p,v in outer.docs.items()]}
            def bytes(self,p):return outer.docs[str(p)]
            def read(self,p):return json.loads(self.bytes(p))
            def sha(self,p):return hashlib.sha256(self.bytes(p)).hexdigest()
        return View()

    def make_qa(self,owner,q,phase='firstpass',judge_text='[[yes]]',reader_text='park'):
        leaf=owner/'results'/phase/q['query_id'];rp=owner/'results'/f"c{q['conversation']}"/'retrieval'/(q['query_id']+'.json')
        retrieval={'question':q['question'],'context':'A walked in the park.'}
        self.put(rp,{'status':'complete','value':retrieval})
        context={'method':'amem','conversation':q['conversation'],'query_id':q['query_id'],'phase':phase}
        outer=self; folder=owner/'work'/('api_'+phase);captured=[]
        class Recorder:
            def generate(self,prompt,*,model,system=None,json_output=False,max_output,stage):
                sig={'provider':'gemini','model':model,'operation':'generateContent','body':generation_body(prompt,system=system,json_output=json_output,max_output=max_output)}
                key=digest(sig);raw_path=folder/'api_cache'/(key+'.json')
                value=reader_text if stage==union.base.evaluator.READER_STAGE else judge_text
                raw={'request':sig,'response':{'modelVersion':model,'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':value}]}}]}}
                outer.put(raw_path,raw)
                event={'provider':'gemini','model':model,'operation':'generateContent','request_hash':key,'cache_path':str(raw_path),
                       'cache_sha256':outer.hash(raw_path),'context':context,'stage':stage,'error_type':None,'source':'new_request'}
                captured.append(event)
                outer.put(folder/'request_intents'/(key+'.json'),{'request_hash':key,'signature':sig,'context':context,'stage':stage,'retries':2})
                return value
        recorder=Recorder();reader=union.base.evaluator.read_answer(recorder,q['question'],q['reference'],[{'id':'native_context','text':retrieval['context'],'observation_date':None}],reference_date=q['reference_date'])
        reader['official_f1']=self.metric(reader['prediction']['answer'],q['reference'],q['category'])
        self.put(leaf/'reader.json',{'status':'complete','value':reader,'events':[captured[-1]]})
        judge=union.base.evaluator.judge_answer(recorder,q['question'],q['reference'],reader['prediction']['answer'])
        self.put(leaf/'judge.json',{'status':'complete','value':judge,'events':[captured[-1]]})
        for kind in ('reader','judge'):self.put(leaf/(kind+'.intent.json'),{'context':context})
        self.put(leaf/'outcome.json',{'query_id':q['query_id'],'conversation':q['conversation'],'method':'amem','phase':phase,'reader':'complete','judge':'complete','reader_path':str(leaf/'reader.json'),'judge_path':str(leaf/'judge.json')})
        self.docs[str(folder/'logical_events.jsonl')]=''.join(json.dumps({'recorded_utc':'fixture',**e})+'\n' for e in captured).encode()
        self.docs[str(folder/'api_calls.jsonl')]=''.join(json.dumps({**{k:e[k] for k in ('provider','model','operation','request_hash','context','stage')},'attempt':1,'status':200,'cache_hit':False})+'\n' for e in captured).encode()
        if owner==self.work:
            self.docs[str(owner/'work/api_method/api_calls.jsonl')]=b''

    def finalize_plan(self):
        first,sources=union.original_firstpass(self.manifest,self.proof,view=self.view())
        self.put(self.final/'firstpass_sources.json',sources)
        r,j,s=union.technical_source(self.view(),self.q,'firstpass')
        self.put(self.final/'technical_firstpass_sources.json',[s])
        self.put(self.final/'qa_recovery_plan.json',{'round':1,'full_cohort':1540,'jobs':[],
                 'firstpass_sources_sha256':self.hash(self.final/'firstpass_sources.json'),
                 'technical_firstpass_sources_sha256':self.hash(self.final/'technical_firstpass_sources.json')})
        self.put(self.final/'qa_recovery_outcome.json',{'plan_sha256':self.hash(self.final/'qa_recovery_plan.json'),'results':[]})
        return first

    def test_full1540_firstpass_missing_and_effective_overlay_real_raw_replay(self):
        self.finalize_plan()
        result=union.reconstruct(self.manifest,self.proof,view=self.view(),created_utc='fixture')
        a=result['summary.json'];self.assertEqual(a['firstpass']['reader_valid'],1539);self.assertEqual(a['effective']['reader_valid'],1540)
        self.assertIsNone(a['firstpass']['official_f1_full_cohort_percent']);self.assertEqual(a['effective']['official_f1_full_cohort_percent'],100.0)
        first=next(r for r in result['firstpass_per_query.json'] if r['query_id']=='c2_q42')
        self.assertEqual((first['retrieval_status'],first['reader_status'],first['judge_status']),('failed','not_run','not_run'))
        self.assertIsNone(first['answer']);self.assertTrue(all(r['judge_correct'] for r in result['per_query.json']))
        self.assertEqual(len(result['raw_request_audit.json']['requests']),2)

    def test_proof_success_drift_and_wrong_owner_block(self):
        s=self.proof['original_successes'][0];p=str(Path(s['leaf'])/'reader.json');v=json.loads(self.docs[p]);v['value']['official_f1']=0
        self.put(p,v)
        with self.assertRaisesRegex(ValueError,'successful result changed'):union.original_firstpass(self.manifest,self.proof,view=self.view())

    def test_extra_qa_logical_consumer_is_rejected(self):
        p=str(self.work/'work/api_firstpass/logical_events.jsonl');self.docs[p]+=self.docs[p].splitlines(keepends=True)[0]
        with self.assertRaisesRegex(ValueError,'logical consumers'):union.audit_qa_requests(self.view(),recovered=False)

    def test_attempt_gap_or_early_success_is_rejected(self):
        p=str(self.work/'work/api_firstpass/api_calls.jsonl');rows=[json.loads(x) for x in self.docs[p].splitlines()]
        rows[0]['attempt']=2;self.docs[p]=''.join(json.dumps(x)+'\n' for x in rows).encode()
        with self.assertRaisesRegex(ValueError,'producer sequence'):union.audit_qa_requests(self.view(),recovered=False)

    def test_extra_qa_namespace_is_rejected(self):
        self.docs[str(self.work/'work/api_extra/api_calls.jsonl')]=b''
        with self.assertRaisesRegex(ValueError,'API owner'):union.audit_qa_requests(self.view(),recovered=False)

    def test_nonretryable_status_and_crossphase_event_metadata_rejected(self):
        original=dict(self.docs)
        p=str(self.work/'work/api_firstpass/api_calls.jsonl');calls=[json.loads(x) for x in self.docs[p].splitlines()]
        bad=copy.deepcopy(calls[0]);bad['status']=400;calls[0]['attempt']=2;calls.insert(0,bad)
        self.docs[p]=''.join(json.dumps(x)+'\n' for x in calls).encode()
        with self.assertRaisesRegex(ValueError,'producer sequence'):union.audit_qa_requests(self.view(),recovered=False)
        for field,value in [('source','existing_run_cache'),('provider','other'),('cache_path',str(self.work/'work/api_recovery/api_cache/foreign.json'))]:
            with self.subTest(field=field):
                self.docs=dict(original)
                leaf,_=union.technical_paths();p=leaf/'reader.json';stage=json.loads(self.docs[str(p)]);stage['events'][0][field]=value;self.put(p,stage)
                lp=str(self.work/'work/api_firstpass/logical_events.jsonl');events=[json.loads(x) for x in self.docs[lp].splitlines()];events[0][field]=value
                self.docs[lp]=''.join(json.dumps(x)+'\n' for x in events).encode()
                with self.assertRaisesRegex(ValueError,'logical event'):union.audit_qa_requests(self.view(),recovered=False)

    def test_valid_no_is_final_not_recoverable(self):
        leaf,_=union.technical_paths();j=json.loads(self.docs[str(leaf/'judge.json')]);j['value']=union.base.evaluator.parse_judge_label('[[no]]')
        r=json.loads(self.docs[str(leaf/'reader.json')])
        self.assertIsNone(union.eligible(self.view(),self.q,r,j))

    def test_failed_stage_with_valid_raw_cannot_be_regenerated(self):
        leaf,_=union.technical_paths();j=json.loads(self.docs[str(leaf/'judge.json')]);j['status']='failed';j.pop('value')
        r=json.loads(self.docs[str(leaf/'reader.json')])
        with self.assertRaisesRegex(ValueError,'valid result'):
            union.eligible(self.view(),self.q,r,j)

    def test_known_invalid_judge_label_is_the_only_eligible_stage(self):
        leaf,_=union.technical_paths();j=json.loads(self.docs[str(leaf/'judge.json')]);j['status']='failed';j.pop('value')
        e=j['events'][0];p=e['cache_path'];raw=json.loads(self.docs[p]);raw['response']['candidates'][0]['content']['parts'][0]['text']='unclear'
        self.put(p,raw);e['cache_sha256']=self.hash(p)
        r=json.loads(self.docs[str(leaf/'reader.json')])
        self.assertEqual(union.eligible(self.view(),self.q,r,j),'judge')

    def test_unknown_qa_without_raw_blocks(self):
        with self.assertRaisesRegex(ValueError,'Unknown QA'):
            union.eligible(self.view(),self.q,{'status':'failed','events':[]},{'status':'not_run'})


if __name__=='__main__':unittest.main()
