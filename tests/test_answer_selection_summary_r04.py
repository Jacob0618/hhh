"""Offline negative/positive provenance and immutable-answer R04 tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import answer_selection_inheritance_r04 as h
from tools import summarize_answer_selection_r04 as s
from jev_mem import answer_selection_r04 as p

def write(path,value,sealed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if sealed:value={**value,'content_sha256':h.digest(value)}
    path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')

def row(method,qid,answer,correct,f1):
    return {'job_id':method+'/'+qid,'method':method,'query_id':qid,'conversation':0,'category':1,
        'question':'Which game?','reference':'RPGs','reference_date':'2023-01-01',
        'reader_status':'complete','judge_status':'complete','qa_status':'complete','judge_parse_mode':'strict','errors':[],
        'reader_answer':answer,'reader_artifact':{'path':f'C:/fixture/{method}/{qid}/reader.json','sha256':'reader-'+method+qid},
        'judge_artifact':{'path':f'C:/fixture/{method}/{qid}/judge.json','sha256':'judge-'+method+qid},
        'scores':{'official_locomo_f1':f1,'custom_token_f1':f1,'judge_correct':float(correct)}}

def inherit_fixture():
    rows=[row('baseline','c0_q0','RPGs.',1,.9),row('jev','c0_q0','RPGs and action games.',0,.4),
          row('baseline','c0_q1','Insufficient evidence.',0,0),row('jev','c0_q1','RPGs',1,1)]
    index=h.candidate_index(rows,population=2)
    qa_jobs={(r['method'],r['query_id']):{'identity':r['job_id'],'type':'qa'} for r in rows}
    source_jobs={(r['method'],r['query_id']):{'identity':r['job_id'],'type':'source'} for r in rows}
    requests={}
    for qid in ('c0_q0','c0_q1'):
        candidates={route:{'answer':index[(method,qid)]['reader_answer'],
            'reader':index[(method,qid)]['reader_artifact'],'source_job_sha256':h.digest(qa_jobs[(method,qid)]),
            'selected_source_job_sha256':h.digest(source_jobs[(method,qid)])} for route,method in h.ROUTES.items()}
        mapping=p.presentation(qid)
        requests[qid]={'query_id':qid,'conversation':0,'candidates':candidates,'presentation':mapping,
            'identical':False,'dispatch_eligible':True,
            'state':{'question':'Which game?','reference_date':'2023-01-01','memories':[],
                'candidate_first':candidates[mapping['first']]['answer'],'candidate_second':candidates[mapping['second']]['answer']}}
    jobs=[]
    for method in h.ARMS:
        for qid,rec in requests.items():
            route='baseline' if method=='baseline' or qid=='c0_q0' else 'jev_r03'
            c=rec['candidates'][route]
            jobs.append({'method':method,'query_id':qid,'conversation':0,'route':route,
                'answer':c['answer'],'reader':c['reader'],'decision':{'path':'C:/fixture/decision','sha256':'d'}})
    return rows,index,qa_jobs,source_jobs,requests,jobs

class ImmutableInheritanceTests(unittest.TestCase):
    def test_selected_valid_no_is_retained_not_vote(self):
        _,index,qa,sources,requests,jobs=inherit_fixture()
        for j in jobs:
            if j['method']=='jev' and j['query_id']=='c0_q0':
                j.update(route='jev_r03',**{k:requests['c0_q0']['candidates']['jev_r03'][k] for k in ('answer','reader')})
        before=deepcopy(index)
        result=h.inherit_jobs(jobs,requests,index,qa,sources,population=2)
        r=next(r for r in result if r['job_id']=='jev/c0_q0')
        self.assertEqual(r['scores']['judge_correct'],0)
        self.assertEqual(r['scores']['official_locomo_f1'],.4)
        self.assertFalse(r['reader_generated_from_union']);self.assertEqual(r['new_reader_calls'],0)
        self.assertEqual(index,before)
    def test_no_semantic_or_whitespace_inheritance(self):
        for answer in ('RPGs','Role playing games.'):
            _,i,q,ss,rs,j=inherit_fixture();j[0]['answer']=answer
            with self.assertRaisesRegex(ValueError,'altered original answer'):h.inherit_jobs(j,rs,i,q,ss,population=2)
    def test_wrong_reader_pointer_rejected(self):
        _,i,q,ss,rs,j=inherit_fixture();j[0]['reader']={**j[0]['reader'],'sha256':'changed'}
        with self.assertRaises(ValueError):h.inherit_jobs(j,rs,i,q,ss,population=2)
    def test_distinct_qa_and_source_job_hashes_required(self):
        _,i,q,ss,rs,j=inherit_fixture();rs['c0_q0']['candidates']['baseline']['selected_source_job_sha256']=h.digest(q[('baseline','c0_q0')])
        with self.assertRaisesRegex(ValueError,'selected-source job'):h.inherit_jobs(j,rs,i,q,ss,population=2)
    def test_gold_key_cannot_enter_model_state(self):
        _,i,q,ss,rs,j=inherit_fixture();rs['c0_q0']['state']['reference']='RPGs'
        with self.assertRaisesRegex(ValueError,'Unexpected state'):h.inherit_jobs(j,rs,i,q,ss,population=2)
    def test_baseline_cannot_choose_other_route(self):
        _,i,q,ss,rs,j=inherit_fixture();j[0].update(route='jev_r03',**{k:rs['c0_q0']['candidates']['jev_r03'][k] for k in ('answer','reader')})
        with self.assertRaisesRegex(ValueError,'Baseline anchor'):h.inherit_jobs(j,rs,i,q,ss,population=2)
    def test_missing_or_duplicate_original_not_smaller_cohort(self):
        rows,*_=inherit_fixture()
        with self.assertRaises(ValueError):h.candidate_index(rows[:-1],population=2)
        with self.assertRaises(ValueError):h.candidate_index(rows+[rows[0]],population=2)
    def test_failed_raw_candidate_cannot_supply_score(self):
        rows,*_=inherit_fixture();rows[0]['errors']=[{'stage':'judge','type':'InvalidArtifact'}]
        with self.assertRaisesRegex(ValueError,'raw-audited'):h.candidate_index(rows,population=2)
    def test_stopping_anchor_does_not_drop_with_invalid_baseline(self):
        rows,*_=inherit_fixture();rows=[r for r in rows if r['method']=='baseline']
        for r in rows:r['scores']['judge_correct']=None
        v=h.stopping(rows,population=2,baseline_correct=1,minimum_correct=2)
        self.assertFalse(v['threshold_eligible']);self.assertEqual(v['baseline_correct'],1)
        self.assertFalse(v['strict_gt_3pp_fixed_denominator']['jev'])
        self.assertEqual(v['successful_independent_components_counted_here'],0)
    def test_comparisons_report_losses_and_f1_separately(self):
        rows,*_=inherit_fixture();v=h.transitions([r for r in rows if r['method']=='baseline'],[r for r in rows if r['method']=='jev'])
        self.assertEqual(v['judge_transitions']['0_to_1']['count'],1)
        self.assertEqual(v['judge_transitions']['1_to_0']['count'],1)
        self.assertEqual(v['f1_directions']['improved']['count'],1)
        self.assertEqual(v['f1_directions']['declined']['count'],1)

class RawWrapperAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name).resolve()
        spec=json.loads((s.ROOT/'research/jev_autoresearch_2026-10-04/R04_answer_selection_protocol_final_v1.json').read_text('utf-8'))
        candidates={'baseline':{'answer':'Insufficient evidence.','reader':{'path':'C:/old/a','sha256':'a'}},
                    'jev_r03':{'answer':'RPGs','reader':{'path':'C:/old/b','sha256':'b'}}}
        facts=[{'id':str(n),'text':'James likes RPGs.' if n==0 else f'Unrelated {n}.','observation_date':'2023-01-01'} for n in range(20)]
        signature=p.build_request('Which games?','2023-01-01',facts,candidates,'c0_q0',spec)
        self.rec={'query_id':'c0_q0','conversation':0,'protocol_sha256':'protocol','signature':signature,
            'request_hash':p.request_hash(signature),'presentation':p.presentation('c0_q0'),'state':signature['body']['state'],
            'candidates':candidates,'identical':False,'dispatch_eligible':True}
        self.snapshot=s.Snapshot();write(self.run/'requests/c0_q0.json',self.rec,sealed=True)
        self.response={'model':'jev-1.13.0','answers':{'answer_choice':{'type':'choice','choice':'first_better',
            'probabilities':{'first_better':.8,'second_better':.1,'no_clear_preference':.05,'neither_supported':.05},'confidence':.7}},
            'usage':{'input_tokens':123,'output_tokens':0}}
        self.event={'request_hash':self.rec['request_hash'],'stage':s.STAGE,'provider':'jev','model':'jev-1.13.0',
            'operation':'systemone','context':s.context('c0_q0'),'attempt':1,'status':200,
            'timestamp_utc':'2026-10-04T00:00:02+00:00','resolved_model':'jev-1.13.0','usage':self.response['usage'],'cache_hit':False}
        write(self.run/'request_intents'/f'{self.rec["request_hash"]}.json',{'request_hash':self.rec['request_hash'],
            'signature':signature,'stage':s.STAGE,'retries':2,'context':s.context('c0_q0'),'created_utc':'2026-10-04T00:00:00+00:00'})
        self.rawpath=self.run/'api_cache'/f'{self.rec["request_hash"]}.json'
        write(self.rawpath,{'request':signature,'response':self.response,'stage':s.STAGE})
        parsed=p.parse_choice(self.response,signature);chosen=p.choose_jev(self.rec,parsed,status='complete');candidate=candidates[chosen['route']]
        ref=self.snapshot.pointer(self.run/'requests/c0_q0.json')
        self.wrapper={'schema':'answer-selection-r04-score-result-v1','query_id':'c0_q0','protocol_sha256':'protocol',
            'request_hash':self.rec['request_hash'],'actual_request_hash':self.rec['request_hash'],
            'request_path':ref['path'],'request_sha256':ref['sha256'],'presentation':self.rec['presentation'],
            'serialized_signature_utf8_bytes':len(p.serialized_signature(signature)),'union_sha256':h.digest(facts),
            'started_utc':'2026-10-04T00:00:01+00:00','finished_utc':'2026-10-04T00:00:03+00:00',
            'status':'complete','choice_result':parsed,'raw':self.snapshot.pointer(self.rawpath),
            'selected_route':chosen['route'],'selection_reason':chosen['reason'],'cheap_route':p.cheap_choice(self.rec),
            'selected_original_reader':candidate['reader'],'selected_exact_answer':candidate['answer']}
    def audit(self):return s.audit_wrapper(self.snapshot,self.run,self.rec,self.wrapper,[self.event])
    def test_actual_raw_consumer_producer_chain_passes(self):self.assertFalse(self.audit()['technical_recovery_eligible'])
    def test_other_context_success_not_own_consumer(self):
        self.event['context']=s.context('c0_q2')
        with self.assertRaisesRegex(ValueError,'Unrelated context'):self.audit()
    def test_wrapper_forged_chosen_route_is_rejected(self):
        self.wrapper['selected_route']='jev_r03' if self.wrapper['selected_route']=='baseline' else 'baseline'
        with self.assertRaisesRegex(ValueError,'route policy'):self.audit()
    def test_wrong_raw_full_signature_cannot_reuse(self):
        raw={'request':deepcopy(self.rec['signature']),'response':self.response,'stage':s.STAGE}
        raw['request']['body']['state']['reference_date']='2023-02-01';write(self.rawpath,raw)
        self.wrapper['raw']={'path':str(self.rawpath),'sha256':s.sha(self.rawpath)};self.snapshot=s.Snapshot()
        with self.assertRaisesRegex(ValueError,'signature/stage'):self.audit()
    def test_successful_malformed_response_falls_back_not_recovery(self):
        self.response['answers']['answer_choice']['probabilities']['first_better']=True
        self.mark_malformed()
        self.assertFalse(self.audit()['technical_recovery_eligible'])
    def mark_malformed(self):
        write(self.rawpath,{'request':self.rec['signature'],'response':self.response,'stage':s.STAGE})
        self.wrapper.update(status='failed',choice_result=None,reason='response_validation_failed',
            raw={'path':str(self.rawpath),'sha256':s.sha(self.rawpath)},selected_route='baseline',selection_reason='failed_fallback_baseline',
            selected_original_reader=self.rec['candidates']['baseline']['reader'],selected_exact_answer='Insufficient evidence.',
            failure_evidence={'consumer_context':s.context('c0_q0'),'producer_context':s.context('c0_q0'),
                'intent':{'path':str(self.run/'request_intents'/f'{self.rec["request_hash"]}.json'),
                    'sha256':s.sha(self.run/'request_intents'/f'{self.rec["request_hash"]}.json')},'terminal_attempts':[]})
        self.snapshot=s.Snapshot()
    def test_rounded_probability_sum_remains_failed_and_valid_qa_inherits(self):
        self.response['answers']['answer_choice']['probabilities']={
            'first_better':.60,'second_better':.20,'no_clear_preference':.10,'neither_supported':.09}
        self.mark_malformed();audit=self.audit();d=audit['schema_rejection']
        self.assertEqual(d['category'],'probability_sum_outside_frozen_tolerance')
        self.assertAlmostEqual(d['raw_probability_sum'],.99)
        self.assertFalse(audit['technical_recovery_eligible']);self.assertEqual(audit['selected_route'],'baseline')
        report=s.choice_validity_report({'c0_q0':audit})
        self.assertEqual(report['schema_rejection_count'],1)
        self.assertEqual(report['valid_choice_semantic_fallback_count'],0)
        _,index,qa,sources,requests,jobs=inherit_fixture()
        selected=h.inherit_jobs(jobs,requests,index,qa,sources,population=2)
        baseline=next(r for r in selected if r['job_id']=='jev/c0_q0')
        self.assertEqual(baseline['qa_status'],'complete');self.assertEqual(baseline['scores']['judge_correct'],1)
    def test_nonmaximum_returned_choice_remains_failed_no_renormalization(self):
        a=self.response['answers']['answer_choice'];a['choice']='no_clear_preference'
        a['probabilities']={'first_better':.15,'second_better':.37,'no_clear_preference':.36,'neither_supported':.12}
        self.mark_malformed();audit=self.audit();d=audit['schema_rejection']
        self.assertEqual(d['category'],'returned_choice_not_maximum')
        self.assertEqual(d['maximum_probability_labels'],['second_better'])
        self.assertEqual(d['returned_choice'],'no_clear_preference')
        self.assertFalse(d['probabilities_renormalized']);self.assertFalse(d['reclassified_as_valid'])
        self.assertFalse(audit['technical_recovery_eligible']);self.assertEqual(audit['status'],'failed')
    def test_missing_usage_is_unknown_not_zero(self):
        value=s.prior.original.measured_usage([{**self.event,'status':'transport_error','usage':None}])
        self.assertEqual(value['usage_unknown_attempts'],1)
        self.assertEqual(value['token_fields']['prompt']['missing_attempts'],1)
    def test_valid_choice_without_usage_is_not_invalid_or_free(self):
        self.response.pop('usage');self.event['usage']={}
        write(self.rawpath,{'request':self.rec['signature'],'response':self.response,'stage':s.STAGE})
        self.wrapper['raw']={'path':str(self.rawpath),'sha256':s.sha(self.rawpath)};self.snapshot=s.Snapshot()
        self.assertEqual(self.audit()['status'],'complete')
        self.assertEqual(s.prior.original.measured_usage([self.event])['usage_unknown_attempts'],1)
    def test_snapshot_detects_input_change(self):
        self.snapshot.raw(self.rawpath);self.rawpath.write_text('{}\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Inputs changed'):self.snapshot.unchanged()

class RealSeamSmokeTests(unittest.TestCase):
    def test_real_frozen_audit_exports_and_parser_callback(self):
        self.assertTrue(callable(s.prior.audit_phase));self.assertTrue(callable(s.prior.record_rows))
        self.assertTrue(callable(s.prior.original.intent_inventory))
        job={'question':'Which game?','reference':'RPGs','reference_date':'2023-01-01',
            'selected_facts':[{'id':str(n),'text':'James likes RPGs.','observation_date':'2023-01-01'} for n in range(20)]}
        self.assertEqual(s.prior.runner.request_signature(job,'reader')['model'],'gemini-3.8-flash')
        self.assertEqual(s.prior.runner.request_signature(job,'judge','RPGs')['model'],'gemini-3.8-flash')
    def test_no_offline_summary_changes_old_globals(self):
        from tools import summarize_evidence_exchange_r01 as old
        self.assertEqual(old.ARMS,('baseline','cheap','jev'))
        self.assertEqual(s.prior.SCHEMA,'evidence-selection-r03-plain-qa-summary-v1')
        self.assertNotEqual(s.SCHEMA,s.prior.SCHEMA)
    def test_request_identity_serializer_is_not_artifact_digest(self):
        job={'question':'Which game?','reference':'RPGs','reference_date':'2023-01-01',
            'selected_facts':[{'id':str(n),'text':'James likes RPGs.','observation_date':'2023-01-01'} for n in range(20)]}
        signature=s.prior.runner.request_signature(job,'reader')
        self.assertNotEqual(h.digest(signature),p.request_hash(signature))
        from jev_mem.sdk20_cached_api import digest as provider_hash
        self.assertEqual(provider_hash(signature),p.request_hash(signature))
    def test_real_prepared_candidate_matches_actual_reader_event_hash(self):
        manifest=s.ROOT/'runs/answer_selection_r04_v1/run_manifest.json'
        if not manifest.exists():self.skipTest('Only runs when the actual immutable R04 preparation exists')
        m=json.loads(manifest.read_text('utf-8'));rec=json.loads(Path(m['items'][0]['request']['path']).read_text('utf-8'))
        for candidate in rec['candidates'].values():
            signature=candidate['reader_request_signature'];key=candidate['reader_request_sha256']
            self.assertNotEqual(h.digest(signature),key);self.assertEqual(p.request_hash(signature),key)
            lineage=candidate['raw_lineage']
            if lineage['kind']=='exact_verified_reader_inheritance':
                stage=json.loads(Path(lineage['producer_reader']['path']).read_text('utf-8'));events=stage['api_events']
            else:events=lineage['api_events']
            self.assertTrue(events);self.assertTrue(all(e['request_hash']==key for e in events))

class ColdCostTests(unittest.TestCase):
    def test_cold_excludes_cheap_judge_and_deduplicates_exact_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp).resolve();legacy=run/'legacy.jsonl';scorer=run/'selector';scorer.mkdir()
            sig={'provider':'gemini','model':'gemini-3.8-flash','body':{'q':'same'}};key=p.request_hash(sig)
            other=h.digest({'not':'required'})
            def event(k,method,component,tokens,time):
                return {'request_hash':k,'context':{'conversation':0,'query_id':'c0_q0','method':method,'component':component},
                    'cache_hit':False,'timestamp_utc':time,'status':200,'model':'gemini-3.8-flash','attempt':1,
                    'usage':{'promptTokenCount':tokens,'candidatesTokenCount':1,'totalTokenCount':tokens+1}}
            old=event(key,'baseline','reader',10,'2026-10-04T00:00:01+00:00')
            unneeded=event(other,'cheap','reader',1000,'2026-10-04T00:00:01+00:00')
            judge=event(key,'baseline','judge',500,'2026-10-04T00:00:01+00:00')
            legacy.write_text(''.join(json.dumps(e)+'\n' for e in (old,unneeded,judge)),encoding='utf-8')
            new=event(key,'jev','reader',20,'2026-10-04T00:00:02+00:00')
            (scorer/'api_calls.jsonl').write_text(json.dumps({'provider':'jev','model':'jev-1.13.0',
                'stage':'evidence_selection_score_r03','status':200,'usage':{'input_tokens':7,'output_tokens':0}})+'\n',encoding='utf-8')
            proof=run/'proof.json';write(proof,{'ledger_sources':[{'path':str(legacy),'sha256':s.sha(legacy)}],
                'inherited_physical_usage':{'scope':'reader+judge legacy, not cold'}})
            phase={'run':str(run),'manifest':{'baseline_reuse_proof':{'path':str(proof),'sha256':s.sha(proof)}},'ledger':[new,unneeded]}
            idx={('baseline','c0_q0'):{'reader_artifact':{'path':'old','sha256':'a'}},
                 ('jev','c0_q0'):{'reader_artifact':{'path':'new','sha256':'b'}}}
            spec={'source_refs':{'r03_selection_manifest':{'path':str(scorer/'run_manifest.json'),'sha256':'unused'}}}
            with patch.object(s.prior.runner,'request_signature',return_value=sig):
                result=s.cold_accounting(spec,phase,idx,{k:{} for k in idx},s.Snapshot())
            self.assertEqual(result['reader_usage']['physical_attempts'],1)
            self.assertEqual(result['reader_usage']['token_fields']['prompt']['recorded_token_sum'],10)
            self.assertEqual(result['unique_reader_request_signatures'],1)
            self.assertEqual(result['exact_reader_signatures_shared_by_routes'],1)
            self.assertEqual(result['r03_upstream_selection_usage']['token_fields']['prompt']['recorded_token_sum'],7)
            self.assertTrue(result['required_producer_coverage_complete'])

if __name__=='__main__':unittest.main()
