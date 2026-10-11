"""Plain QA offline replay fixtures; no live model/provider constructor."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import summarize_mem0_regroup_plain_qa_v1 as s
from jev_mem.providers import ModelAPI
from tools.run_sdk20_modules import save_new, verify, sha, digest


class Recording:
    generate = ModelAPI.generate
    def __init__(self, root, context=None, shared=None):
        self.root, self.context, self.events = root, context or {}, []
        self.shared = shared if shared is not None else {'ledger':defaultdict(list),'reader':'Paris','judge':'[[yes]]'}
    def scoped(self, **context):return Recording(self.root,{**self.context,**context},self.shared)
    def _redact(self, error):return str(error)
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified'):
        s.check_plain_text_request(provider,model,body,operation=operation,stage=stage)
        sig={'provider':provider,'model':model,'operation':operation,'body':body};key=s.request_digest(sig)
        text=self.shared['reader' if stage==s.evaluator.READER_STAGE else 'judge']
        raw={'request':sig,'response':{'modelVersion':model,'usageMetadata':{'promptTokenCount':12,'candidatesTokenCount':3},
             'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]}}
        target=self.root/'api_cache'/(key+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(raw),encoding='utf8')
        ip=self.root/'request_intents'/(key+'.json');ip.parent.mkdir(exist_ok=True)
        ip.write_text(json.dumps({'request_hash':key,'signature':sig,'stage':stage,'context':self.context,'retries':2}),encoding='utf8')
        self.events.append({'request_hash':key,'provider':provider,'model':model,'operation':operation,'stage':stage,
            'context':self.context,'source':'new_request','cache_path':str(target),'cache_sha256':sha(target),'error_type':None})
        self.shared['ledger'][key,stage,s.fingerprint(self.context)].append({'request_hash':key,'provider':provider,
            'model':model,'operation':operation,'stage':stage,'context':self.context,'status':200,'attempt':1,
            'cache_hit':False,'usage':raw['response']['usageMetadata'],'resolved_model':model})
        return raw['response']


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.q={'id':'c0_q0','question':'Where?','reference':'Paris','category':1}
        self.m={'protocol_sha256':'mock','reference_dates':{'0':'2023-06-30'}}
        self.arm=s.ARMS[0]
        self.bank={'facts':[{'id':f'f{i}','text':f'Paris fact{i}','observation_date':None} for i in range(20)]}
        self.api=Recording(self.root/'conv_0/api',{'conversation':0,'architecture':s.driver.SCHEMA})
        self.snapshot=s.Snapshot()
        for target in ('requests.Session.post','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(target,side_effect=AssertionError('Offline only'));p.start();self.addCleanup(p.stop)

    def setup_selection(self,status='complete'):
        from mem0.utils.scoring import score_and_rank
        inputs={'semantic_results':[{'id':f['id'],'score':.9,'payload':{'data':f['text']}} for f in self.bank['facts']],
                'bm25_scores':{},'entity_boosts':{},'threshold':.1,'top_k':20,'explain':False}
        base=score_and_rank(**inputs)
        capture={'inputs':inputs,'base20':base,'base20_ids':[r['id'] for r in base],
                 'candidate_ids':[r['id'] for r in base],'candidate_cap':80}
        capture['capture_sha256']=digest(capture)
        p=self.root/'conv_0/selection'/self.arm/'c0_q0.json'
        save_new(p,{**s.driver.identity(0,self.arm,self.q,self.m),'status':status,'bank_sha256':'bank',
                    'selected_ids':capture['base20_ids'] if status=='complete' else [],
                    'capture':capture if status=='complete' else None,'error':None if status=='complete' else {'type':'SearchError'}})
        return p

    def execute(self,selection='complete'):
        self.setup_selection(selection)
        return s.driver.qa_one(self.root,0,self.arm,self.q,self.m,self.bank,'bank',self.api,lambda *a:.75)

    def audit(self):
        return s.audit_slot(self.snapshot,self.root,0,self.arm,self.q,self.m,self.bank,'bank',self.api.shared['ledger'],lambda *a:.75)

    def test_full_plain_raw_replay_keeps_reader_and_false_judge(self):
        self.api.shared['reader']='Paris\n- museum';self.api.shared['judge']='[[no]]'
        self.execute();row=self.audit()
        self.assertEqual(row['qa_status'],'complete');self.assertEqual(row['scores']['judge_correct'],0.)
        self.assertEqual(row['scores']['official_locomo_f1'],.75)
        self.assertTrue(any(p.parent.name=='api_cache' for p in self.snapshot.files))
        self.assertTrue(any(p.parent.name=='request_intents' for p in self.snapshot.files))
        self.assertEqual(self.snapshot.unchanged(),[])

    def test_bad_judge_retains_f1_and_never_counts_as_false(self):
        self.api.shared['judge']='yes because Paris';self.execute();row=self.audit()
        self.assertEqual((row['reader_status'],row['judge_status'],row['qa_status']),('complete','failed','failed'))
        self.assertEqual(row['scores']['official_locomo_f1'],.75);self.assertIsNone(row['scores']['judge_correct'])

    def test_missing_combined_row_does_not_erase_independent_metrics(self):
        final=self.execute();(self.root/'conv_0/qa'/self.arm/'c0_q0.json').unlink()
        row=self.audit();self.assertEqual(row['qa_status'],'missing')
        self.assertEqual((row['scores']['official_locomo_f1'],row['scores']['judge_correct']),(.75,1.))

    def test_selection_failure_two_no_dispatch_stages(self):
        self.execute('failed');row=self.audit()
        self.assertEqual((row['selection_status'],row['reader_status'],row['judge_status']),('failed','not_run','not_run'))
        self.assertTrue(all(v is None for v in row['scores'].values()));self.assertEqual(self.api.shared['ledger'],{})

    def test_corrupt_raw_or_model_cannot_pass_replay(self):
        self.execute();rows=list(self.api.shared['ledger'].values());rows[0][0]['model']='different'
        row=self.audit();self.assertEqual(row['reader_status'],'invalid');self.assertIsNone(row['scores']['official_locomo_f1'])

    def test_old_qa_lineage_is_rejected(self):
        final=self.execute();reader=verify(final['reader']['path']);event=reader['api_events'][0]
        p=self.root/'conv_0/api/cache_lineage'/(event['request_hash']+'.json');p.parent.mkdir();p.write_text('{}')
        row=self.audit();self.assertEqual(row['reader_status'],'invalid')

    def test_unknown_stage_intent_retains_failure_and_no_dispatch(self):
        self.setup_selection();sp=self.root/'conv_0/selection'/self.arm/'c0_q0.json'
        common={**s.driver.identity(0,self.arm,self.q,self.m),'selection_sha256':sha(sp),'bank_sha256':'bank',
                'selected_ids':[f['id'] for f in self.bank['facts']],'reference_date':'2023-06-30','stage':'reader'}
        rp=self.root/'conv_0/reader'/self.arm/'c0_q0.json'
        s.driver.durable_intent(rp.with_suffix('.intent.json'),common)
        s.driver.qa_one(self.root,0,self.arm,self.q,self.m,self.bank,'bank',self.api,lambda *a:.75)
        row=self.audit();self.assertEqual((row['reader_status'],row['judge_status']),('failed','not_run'))
        self.assertEqual(self.api.shared['ledger'],{})

    def test_unresolved_shared_request_guard_has_zero_dispatch_but_bound_intent(self):
        # Capture a genuine expected body, then remove its output to model an
        # earlier producer's unresolved intent. A different arm consumer is blocked.
        ctx=s.stage_context(0,self.arm,'c0_q0','judge');api=self.api.scoped(**ctx)
        s.evaluator.judge_answer(api,'Where?','Paris','Paris');event=api.events[0]
        Path(event['cache_path']).unlink()
        other={**ctx,'method':s.ARMS[1]}
        replay=s.PlainReplay(api.root,[],other,self.api.shared['ledger'],snapshot=self.snapshot,unresolved_guard=True)
        with self.assertRaises(s.DeclaredTransportFailure):s.evaluator.judge_answer(replay,'Where?','Paris','Paris')
        replay.assert_consumed()
        (api.root/'request_intents'/(event['request_hash']+'.json')).unlink()
        replay=s.PlainReplay(api.root,[],other,self.api.shared['ledger'],unresolved_guard=True)
        with self.assertRaises(s.InvalidArtifact):s.evaluator.judge_answer(replay,'Where?','Paris','Paris')

    def slots(self,n=30):
        return [{'query_id':f'c{i//3}_q{i%3}','conversation':i//3,'category':1,'method':arm,
                 **{k+'_status':'complete' for k in ('selection','reader','judge','qa')},
                 'scores':{k:1. for k in s.METRICS}} for arm in s.ARMS for i in range(n)]

    def test_independent_denominators_and_metric_specific_pairs(self):
        slots=self.slots();slots[0]['judge_status']='failed';slots[0]['scores']['judge_correct']=None
        slots[31]['reader_status']='failed';slots[31]['judge_status']='not_run';slots[31]['scores']={k:None for k in s.METRICS}
        r=s.aggregate(slots,denominator=30,bootstrap_samples=20)
        a=r['arms'][s.ARMS[0]]['metrics']
        self.assertEqual(a['official_locomo_f1']['valid_denominator'],30)
        self.assertEqual(a['judge_correct']['valid_denominator'],29)
        self.assertIsNone(a['judge_correct']['full_cohort_observed_mean'])
        self.assertEqual(a['judge_correct']['complete_only_mean'],1.)
        p=r['paired_merged_minus_unmerged'];self.assertEqual(p['official_locomo_f1']['common_success_count'],29)
        self.assertEqual(p['judge_correct']['common_success_count'],28)
        self.assertFalse(p['judge_correct']['bootstrap']['eligible']) # c0 only one pair
        self.assertTrue(p['official_locomo_f1']['bootstrap']['eligible'])

    def test_missing_conversation_and_duplicate_or_nan_slots_rejected(self):
        slots=self.slots(3);r=s.aggregate(slots,denominator=3,bootstrap_samples=20)
        self.assertFalse(r['paired_merged_minus_unmerged']['judge_correct']['bootstrap']['eligible'])
        slots[0]['scores']['judge_correct']=float('nan')
        with self.assertRaises(ValueError):s.aggregate(slots,denominator=3)
        slots=self.slots(3);slots[0]['query_id']=slots[1]['query_id']
        with self.assertRaises(ValueError):s.aggregate(slots,denominator=3)

    def test_unknown_cost_and_cache_usage_do_not_become_free_bill(self):
        rows=[{'model':'gemini-2.5-flash','status':'transport_error','usage_unknown':True,'stage':'reader'},
              {'model':'gemini-2.5-flash','cache_hit':True,'usage':{'promptTokenCount':1000000},'stage':'reader'}]
        with patch.object(s.cache_pack,'verify_pack',return_value={'entries':[]}):
            r=s.usage_and_lineage(self.snapshot,self.root,{'embedding_cache_pack':{}},rows,[],[])
        self.assertEqual(r['new_namespace_only']['physical_attempts'],1)
        self.assertEqual(r['new_namespace_only']['cache_events'],1)
        self.assertIsNone(r['new_namespace_only']['total_cost_usd'])


if __name__=='__main__':unittest.main()
