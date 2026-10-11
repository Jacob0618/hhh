import json
import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from jev_mem import source_append_r05 as pure
from tools import prepare_source_append_r05 as prep


def facts():
    return [{'id':str(i),'text':'Sam memory '+str(i),'observation_date':None} for i in range(20)]


def candidate(slot, text='Sam: useful detail', ident=None):
    view={'session_id':'session_1','source_observation_date':'2023-01-01',
        'turns':[{'source_turn_id':f'D1:{slot+1}','speaker_text':text,'machine_captions':[]}]}
    a=pure.attachment(view)
    return {'packet_id':ident or pure.digest(view),'packet_view':view,'attachment':a,
        'attachment_utf8_bytes':len(a.encode('utf-8')),'origin':{'anchor_slot':slot,'anchor_id':str(slot)}}


def envelopes(messages,session='session_1'):
    return {'envelopes':[{'session_id':session,'observation_date':'2023-01-01',
        'source_turn_ids':[f'D1:{i+1}' for i in range(len(messages))],
        'messages':[{'content':text} for text in messages]}]}


class ProjectionTests(unittest.TestCase):
    def test_known_media_preserves_exact_caption_and_speaker_offsets(self):
        content='Jo: 前文 [Sharing image - query: secret Q. The image shows: Café sunset] 後文'
        p=pure.project_content(content)
        self.assertEqual(p['speaker_text'],'Jo: 前文  後文')
        self.assertEqual(p['machine_captions'],['Café sunset'])
        self.assertNotIn('secret Q',p['speaker_text'])
        for span in p['speaker_spans']+p['caption_spans']:
            self.assertEqual(content[span['start']:span['end']],span['text'])

    def test_caption_only_rendering_and_no_query_leak(self):
        p=pure.project_content('A: x [Sharing image that shows: a blue bird] y')
        self.assertEqual(p['machine_captions'],['a blue bird'])
        self.assertEqual(p['speaker_text'],'A: x  y')

    def test_query_only_ambiguous_and_unterminated_unknown(self):
        for body in ['[Sharing image - query for: hidden]',
            '[Sharing image - query: hidden. The image shows: a. The image shows: b]',
            '[Sharing image - query: hidden']:
            p=pure.project_content('A: x '+body)
            self.assertEqual(p['speaker_text'],'A: x ')
            self.assertEqual(p['machine_captions'],[])
            self.assertEqual(p['omitted_media'][0]['reason'],'unresolved_media')

    def test_actual_lf_canonical_json_and_byte_accounting(self):
        c=candidate(0,'Sam: 中文')
        self.assertEqual(pure.WRAPPER.count('\n'),4)
        self.assertNotIn('\\n',pure.WRAPPER)
        self.assertEqual(c['attachment_utf8_bytes'],len(c['attachment'].encode('utf-8')))
        self.assertGreater(c['attachment_utf8_bytes'],len(c['attachment']))
        self.assertIn('"machine_captions":[]',c['attachment'])

    def test_exact_zip_and_global_duplicate_rejected(self):
        g=envelopes(['A: one','A: two']);g['envelopes'][0]['source_turn_ids'].pop()
        with self.assertRaises(ValueError):pure.source_turns(g,{'path':'p','sha256':'h'})
        g=envelopes(['A: one']);g['envelopes'].append(deepcopy(g['envelopes'][0]));g['envelopes'][1]['session_id']='session_2'
        with self.assertRaises(ValueError):pure.source_turns(g,{'path':'p','sha256':'h'})


class LocatorTests(unittest.TestCase):
    def test_empty_scope_and_corpus(self):
        o=pure.build_candidates('rare request',facts(),[None]*20,[],conversation=0)
        self.assertEqual(o['candidates'],[]);self.assertEqual(o['source_turn_count'],0)
        self.assertEqual(len(o['exclusions']),20)

    def test_current_scope_not_history_or_neighbour_jump(self):
        ts=pure.source_turns(envelopes(['A: unrelated','A: target red','A: excluded bridge']),{'path':'p','sha256':'h'})
        fs=facts();fs[0]['text']='A target red'
        scopes=[{'session_id':'session_1','source_turn_ids':['D1:2'],
            'historical_turn_ids':['D1:1','D1:3']}]+[None]*19
        c=pure.build_candidates('target',fs,scopes,ts,conversation=0)['candidates'][0]
        self.assertEqual([t['source_turn_id'] for t in c['packet_view']['turns']],['D1:2'])
        self.assertEqual(c['boundary']['left'],'scope_excluded');self.assertEqual(c['boundary']['right'],'scope_excluded')

    def test_wrong_session_or_nonexistent_scope_rejected(self):
        ts=pure.source_turns(envelopes(['A: target']),{'path':'p','sha256':'h'})
        for session,tid in [('session_2','D1:1'),('session_1','D1:99')]:
            with self.assertRaises(ValueError):pure.build_candidates('target',facts(),[{'session_id':session,'source_turn_ids':[tid]}]+[None]*19,ts,conversation=0)

    def test_dedup_canonical_origin_aliases_and_distinct_slots(self):
        ts=pure.source_turns(envelopes(['Sam: rare clue']),{'path':'p','sha256':'h'})
        fs=facts();fs[0]['text']=fs[1]['text']='Sam rare clue'
        scope={'session_id':'session_1','source_turn_ids':['D1:1']}
        out=pure.build_candidates('rare clue',fs,[scope,scope]+[None]*18,ts,conversation=0)
        self.assertEqual(len(out['candidates']),1)
        self.assertEqual(out['candidates'][0]['origin']['anchor_slot'],0)
        self.assertEqual(len(out['candidates'][0]['origin_aliases']),2)

    def test_whole_oversize_rejected_no_shortened_window(self):
        ts=pure.source_turns(envelopes(['A: red target','A: '+('é'*3000),'A: end']),{'path':'p','sha256':'h'})
        fs=facts();fs[0]['text']='red target'
        scope={'session_id':'session_1','source_turn_ids':['D1:1','D1:2','D1:3']}
        o=pure.build_candidates('red',fs,[scope]+[None]*19,ts,conversation=0)
        self.assertEqual(o['candidates'],[])
        self.assertTrue(any(e['reason']=='whole_attachment_oversize' for e in o['exclusions']))

    def test_dates_alone_never_make_center_positive(self):
        ts=pure.source_turns(envelopes(['Sam: unrelated']),{'path':'p','sha256':'h'})
        scope={'session_id':'session_1','source_turn_ids':['D1:1']}
        o=pure.build_candidates('2023-01-01',facts(),[scope]+[None]*19,ts,conversation=0)
        self.assertEqual(o['candidates'],[])

    def test_memo_only_exact_same_corpus_and_anchor(self):
        ts=pure.source_turns(envelopes(['Sam: rare clue']),{'path':'p','sha256':'h'})
        fs=facts();scope={'session_id':'session_1','source_turn_ids':['D1:1']};scopes=[scope]*20;cache={}
        a=pure.build_candidates('rare clue',fs,scopes,ts,conversation=0,score_cache=cache)
        b=pure.build_candidates('rare clue',fs,scopes,ts,conversation=0)
        self.assertEqual(a,b);self.assertEqual(len(cache),20)


class DecisionTests(unittest.TestCase):
    def test_no_ranking_all_take_no_scores(self):
        for n in range(3):
            cs=[candidate(i) for i in range(n)]
            d=pure.choose_jev(facts(),cs,None,score_status='not_dispatched')
            self.assertEqual(d['candidate_indices'],list(range(n)))
            with self.assertRaises(ValueError):pure.choose_jev(facts(),cs,{},score_status='complete')

    def test_low_scores_still_top_two_and_stable_ties(self):
        cs=[candidate(i) for i in range(4)]
        d=pure.choose_jev(facts(),cs,{'f00':.01,'f01':.02,'f02':.02,'f03':.001})
        self.assertEqual(d['candidate_indices'],[1,2])

    def test_any_unknown_entire_original(self):
        fs=facts();cs=[candidate(i) for i in range(3)]
        d=pure.choose_jev(fs,cs,None,score_status='unknown')
        self.assertEqual(pure.selected_facts(fs,cs,d),fs)
        for scores in ({'f00':.5}, {'f00':True,'f01':.5,'f02':.7}, {'f00':math.nan,'f01':.5,'f02':.7}):
            with self.assertRaises(ValueError):pure.choose_jev(fs,cs,scores)

    def test_full_prefix_id_date_order_not_mutated(self):
        fs=facts();original=deepcopy(fs);cs=[candidate(4),candidate(9),candidate(12)]
        d=pure.choose_cheap(fs,cs);v=pure.selected_facts(fs,cs,d)
        self.assertEqual(fs,original)
        self.assertEqual([f['id'] for f in v],[f['id'] for f in original])
        for i,f in enumerate(v):
            self.assertEqual(f['observation_date'],original[i]['observation_date'])
            self.assertEqual(f['text'],original[i]['text']+(cs[0]['attachment'] if i==4 else cs[1]['attachment'] if i==9 else ''))

    def test_duplicate_slots_and_accounting_tamper_rejected(self):
        cs=[candidate(0),candidate(0,'other',ident='another')]
        with self.assertRaises(ValueError):pure.choose_cheap(facts(),cs)
        cs=[candidate(0)];d=pure.choose_cheap(facts(),cs);d['added_utf8_bytes']+=1
        with self.assertRaises(ValueError):pure.selected_facts(facts(),cs,d)

    def test_request_no_answers_gold_or_route_and_strict_parser(self):
        cs=[candidate(i) for i in range(3)];sig=pure.build_request('requested detail','2023-01-01',facts(),cs,{'jev':{'predicate':'P'}})
        self.assertEqual(set(sig['body']['questions']),{'f00','f01','f02'})
        self.assertEqual(sig['body']['state'],{'query':{'text':'requested detail','reference_date':'2023-01-01'}})
        response={'model':'jev-1.13.0','answers':{key:{'type':'noul','noul':.4} for key in sig['body']['questions']}}
        self.assertEqual(pure.parse_scores(response,sig),{'f00':.4,'f01':.4,'f02':.4})
        response['model']='other'
        with self.assertRaises(ValueError):pure.parse_scores(response,sig)


class PreparedSeamsTests(unittest.TestCase):
    def test_skip_replay_decision_job_chain(self):
        for n in (1,2):
            with TemporaryDirectory() as temp:
                run=Path(temp);fs=facts();cs=[candidate(i) for i in range(n)]
                old={'query_id':'c0_q0','conversation':0,'question':'detail','reference_date':None,
                    'selected_facts':fs,'selected_ids':[f['id'] for f in fs],
                    'reference':'offline only','category':1,'job_id':'N0/c0_q0',
                    'protocol_sha256':'old','bank':{'path':'bank','sha256':'banksha'}}
                req=prep._request_value(old,{'candidates':cs},{'jev':{'predicate':'P'}},'p')
                reqref=prep.save_new(run/'requests/c0_q0.json',req)
                prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,
                    'query_id':'c0_q0','protocol_sha256':'p','request_hash':req['request_hash'],
                    'status':'not_dispatched','reason':'no_ranking_choice','scores':None,'raw':None})
                scores,status,resref,rawref=prep._score(prep.Reader(),run,
                    {'query_id':'c0_q0','protocol_sha256':'p','request':reqref})
                self.assertEqual((scores,status,rawref),(None,'not_dispatched',None))
                decision=pure.choose_jev(fs,cs,scores,score_status=status)
                derived=pure.selected_facts(fs,cs,decision)
                record={'selected_ids':old['selected_ids'],'selected_facts':derived,
                    'selected_sha256':pure.digest(derived),'unchanged_ordered_reader_facts':False,
                    'frozen_source':{'path':'old','sha256':'s'},'bank':old['bank'],
                    'original_selection':None,'proposal':{'path':'proposal','sha256':'s'},
                    'request':reqref,'response':resref,'raw_response':rawref}
                job=prep._job(old,'jev',record,{'path':'selection','sha256':'s'},'p')
                self.assertEqual(job['selected_facts'],derived)
                self.assertEqual(job['reference'],'offline only')
                self.assertEqual(job['lineage']['response'],resref)
                self.assertEqual(job['selected_ids'],old['selected_ids'])

    def test_real_imports_do_not_rebind_frozen_prior_score(self):
        self.assertEqual(prep._score_replay.__globals__['SCORE_SCHEMA'],prep.SCORE_SCHEMA)
        self.assertEqual(prep.prior._score.__globals__['SCORE_SCHEMA'],'evidence-selection-r03-score-result-v1')
        self.assertIs(prep._load_baseline, __import__('tools.prepare_evidence_exchange_r01',fromlist=['_load_baseline'])._load_baseline)

    def test_request_skip_n0_n2_and_active_n3(self):
        old={'query_id':'c0_q0','conversation':0,'question':'detail','reference_date':None,'selected_facts':facts()}
        spec={'jev':{'predicate':'P'}}
        for n in range(5):
            req=prep._request_value(old,{'candidates':[candidate(i) for i in range(n)]},spec,'p')
            self.assertEqual(req['logical_noul_count'],n)
            self.assertEqual(req['dispatch_eligible'],n>=3)
            self.assertEqual(req['skip_reason'],'no_ranking_choice' if n<=2 else None)
            self.assertEqual(req['signature_utf8_bytes'],len(pure.serialized_signature(req['signature'])))

    def test_actual_final_protocol_binding_and_wrapper(self):
        reader=prep.Reader();spec,ref=prep._research_protocol(reader)
        self.assertEqual(ref['sha256'],pure.PROTOCOL_SHA256)
        self.assertEqual(spec['reader_representation']['wrapper'],pure.WRAPPER)
        pointers=prep._proof_pointers(reader)
        self.assertEqual(set(pointers),{'frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'})


if __name__=='__main__': unittest.main()
