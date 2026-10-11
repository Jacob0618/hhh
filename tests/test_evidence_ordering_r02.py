"""Offline invariants: exact records, all20 scores, byte-bound request and fallback."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem import evidence_ordering_r02 as x
from tools import prepare_evidence_ordering_r02 as prep


def fixture():
    return [{'id':f'id{i}','text':f'neutral memory {i}','observation_date':None} for i in range(20)]


class OrderingTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture()
        self.spec = json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))

    def test_protocol_bytes_and_same20(self):
        self.assertEqual(x.PROTOCOL_SHA256,prep.file_sha(prep.RESEARCH_PROTOCOL))
        x.validate_records(self.records)
        selected = x.selected_facts(self.records,list(reversed(range(20))))
        self.assertEqual(list(reversed(self.records)),selected)
        self.assertIsNone(selected[0]['observation_date'])
        selected[0]['text'] = 'mutation'
        self.assertEqual('neutral memory 19',self.records[19]['text'])

    def test_rejects_duplicate_and_nonpermutation(self):
        for bad in (self.records[:-1],self.records+[self.records[0]],self.records[:-1]+[self.records[0]]):
            with self.assertRaises(ValueError):x.validate_records(bad)
        for order in ([0]*20,list(range(19)),[True]+list(range(1,20))):
            with self.assertRaises(ValueError):x.selected_facts(self.records,order)

    def test_same_fields_feature_null_only_conversion(self):
        self.records[0]['observation_date']='2023-04-01'
        with patch.object(x,'stable_bm25',return_value=[0.]*20) as scorer:
            d=x.choose_cheap('question',None,self.records)
        query,docs=scorer.call_args.args
        self.assertEqual('question\nReference date: ',query)
        self.assertEqual('neutral memory 0\nObservation date: 2023-04-01',docs[0])
        self.assertTrue(docs[1].endswith('Observation date: '))
        self.assertEqual(list(range(20)),d['permutation'])
        self.assertEqual(190,d['stable_tie_pairs'])
        self.assertIsNone(self.records[1]['observation_date'])

    def test_cheap_rank_matches_frozen_bm25_exact20(self):
        self.records[19]['text']='unique targeted relation'
        d=x.choose_cheap('targeted relation','2023-04-01',self.records)
        self.assertEqual(19,d['permutation'][0])
        self.assertEqual(set(range(20)),set(d['permutation']))
        self.assertEqual(list(range(19)),d['permutation'][1:])

    def test_signature_explicit_full_fact_no_gold_or_id(self):
        sig=x.build_request('targeted question',None,self.records,self.spec)
        self.assertEqual({'query'},set(sig['body']['state']))
        self.assertEqual('',sig['body']['state']['query']['reference_date'])
        self.assertEqual([f'f{i:02d}' for i in range(20)],list(sig['body']['questions']))
        for i,(key,q) in enumerate(sig['body']['questions'].items()):
            self.assertEqual('noul',q['type'])
            literal=q['instructions'].split('Memory record (JSON):\n')[1]
            self.assertEqual({'text':self.records[i]['text'],'observation_date':''},json.loads(literal))
            self.assertNotIn('reference_answer',q['instructions'])
            self.assertIn('not whether it is true',q['instructions'])
        self.assertEqual(x.request_hash(sig),__import__('hashlib').sha256(x.serialized_signature(sig)).hexdigest())

    def test_valid_zero_and_low_values_rank_no_threshold(self):
        scores={f'f{i:02d}':0. for i in range(20)}
        scores['f19']=.01
        d=x.choose_jev(self.records,scores)
        self.assertEqual([19,*range(19)],d['permutation'])
        scores['f03']=.01
        d=x.choose_jev(self.records,scores)
        self.assertEqual([3,19],d['permutation'][:2])

    def test_whole_query_fallback_and_no_synthetic_scores(self):
        for status in ('unknown','failed','not_dispatched'):
            d=x.choose_jev(self.records,None,score_status=status)
            self.assertEqual(list(range(20)),d['permutation'])
            self.assertEqual(status,d['score_status'])
            with self.assertRaises(ValueError):x.choose_jev(self.records,{},score_status=status)

    def test_partial_nonfinite_boolean_refused(self):
        good={f'f{i:02d}':.5 for i in range(20)}
        for value in (None,True,float('inf'),float('nan'),-.1,1.1):
            bad=dict(good,f00=value)
            with self.assertRaises(ValueError):x.choose_jev(self.records,bad)
        del good['f19']
        with self.assertRaises(ValueError):x.choose_jev(self.records,good)

    def test_exact_parser_model_type_keys(self):
        sig=x.build_request('question',None,self.records,self.spec)
        good={'model':'jev-1.13.0','answers':{k:{'type':'noul','noul':.2} for k in sig['body']['questions']}}
        self.assertEqual(20,len(x.parse_scores(good,sig)))
        for model in ('other',None):
            bad=deepcopy(good);bad['model']=model
            with self.assertRaises(ValueError):x.parse_scores(bad,sig)
        bad=deepcopy(good);bad['answers']['f00']['type']='choice'
        with self.assertRaises(ValueError):x.parse_scores(bad,sig)
        bad=deepcopy(good);del bad['answers']['f19']
        with self.assertRaises(ValueError):x.parse_scores(bad,sig)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.records=fixture()
        spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))
        self.sig=x.build_request('question',None,self.records,spec)

    def artifacts(self,run,*,status='complete',raw=True,signature=None,eligible=True):
        sig=self.sig
        request=prep.save_new(run/'requests/c0_q0.json',{'signature':sig,'request_hash':x.request_hash(sig),'dispatch_eligible':eligible})
        rawref=None
        scores={f'f{i:02d}':.3 for i in range(20)}
        if raw:
            p=run/'api/raw.json';p.parent.mkdir()
            response={'model':'jev-1.13.0','answers':{k:{'type':'noul','noul':v} for k,v in scores.items()}}
            p.write_text(json.dumps({'request':signature or sig,'response':response}),encoding='utf8')
            rawref={'path':str(p.resolve()),'sha256':prep.file_sha(p)}
        v={'query_id':'c0_q0','protocol_sha256':'proto','request':request}
        prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,'query_id':'c0_q0',
            'protocol_sha256':'proto','request_hash':x.request_hash(sig),'status':status,
            'raw':rawref,'scores':scores if status=='complete' else None,'error_type':'OfflineMock','reason':'test'})
        return v

    def test_complete_raw_bound_and_scores_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);v=self.artifacts(run)
            scores,status,_,_=prep._score(prep.Reader(),run,v)
            self.assertEqual('complete',status);self.assertEqual(20,len(scores))

    def test_raw_other_signature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad=deepcopy(self.sig);bad['body']['state']['query']['text']='other'
            run=Path(tmp);v=self.artifacts(run,signature=bad)
            with self.assertRaises(ValueError):prep._score(prep.Reader(),run,v)

    def test_oversize_cannot_have_complete_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);v=self.artifacts(run,eligible=False)
            with self.assertRaises(ValueError):prep._score(prep.Reader(),run,v)

    def test_explicit_unknown_without_raw_remains_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);v=self.artifacts(run,status='unknown',raw=False)
            scores,status,_,_=prep._score(prep.Reader(),run,v)
            self.assertIsNone(scores);self.assertEqual('unknown',status)

    def test_missing_wrapper_not_terminal_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            ref=prep.save_new(run/'requests/c0_q0.json',{'signature':self.sig,'request_hash':x.request_hash(self.sig)})
            with self.assertRaisesRegex(ValueError,'Missing result'):
                prep._score(prep.Reader(),run,{'request':ref,'query_id':'c0_q0','protocol_sha256':'proto'})

    def test_actual_r01_qa_manifest_is_raw_sha_and_protocol_bound(self):
        spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))
        reader=prep.Reader()
        ref=prep._canonical_ref(spec['source_basis']['n0_qa_manifest'])
        value=reader.bound(ref,sealed=False)
        self.assertNotIn('content_sha256',value)
        prep._validate_qa_manifest(value)
        self.assertEqual(value['baseline_reuse_proof'],reader.raw_pointer(prep.FROZEN_REUSE_PROOF))
        bad=deepcopy(value);bad['phase_id']='mutated'
        with self.assertRaises(ValueError):prep._validate_qa_manifest(bad)

    def test_full_selected_record_replay_preserves_null_and_gold_lineage_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);v=self.artifacts(run)
            spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))
            source_ref={'path':'offline-source','sha256':'a'*64}
            bank_ref={'path':'offline-bank','sha256':'b'*64}
            old={'query_id':'c0_q0','conversation':0,'job_id':'baseline/c0_q0','question':'question',
                'reference_date':None,'reference':'PRIVATE_GOLD_SENTINEL','category':2,
                'selected_facts':self.records,'selected_ids':[f['id'] for f in self.records],
                'bank':bank_ref,'selection':{'path':'offline-selection','sha256':'c'*64},'protocol_sha256':'original'}
            cheap=x.choose_cheap('question',None,self.records)
            proposal={'schema':prep.PROPOSAL_SCHEMA,'protocol_sha256':'proto','query_id':'c0_q0','conversation':0,
                'frozen_source':source_ref,'original_job':old,'source_job_sha256':x.digest(old),
                'bank':bank_ref,'original_selection':old['selection'],'request':v['request'],
                'baseline_selected_sha256':x.digest(self.records),'cheap_decision':cheap,
                'gold_not_used_by_ordering':True,'lineage_job_contains_offline_reference':True}
            proposal_ref=prep.save_new(run/'proposals/c0_q0.json',proposal)
            # Replace request through a fresh fixture path, retaining the original approved signature.
            reqpath=run/'requests/c0_q0.json'
            req=json.loads(reqpath.read_text(encoding='utf8'))
            req.update(schema=prep.REQUEST_SCHEMA,protocol_sha256='proto',query_id='c0_q0',conversation=0,
                logical_noul_count=20,logical_request_cap=1,max_physical_attempts_per_phase=3,
                signature_utf8_bytes=len(x.serialized_signature(self.sig)),signature_utf8_byte_cap=131072)
            reqpath.unlink();newreq=prep.save_new(reqpath,req)
            # Rebind the independently hashed new metadata, same actual signature.
            proposal['request']=newreq
            pp=run/'proposals/c0_q0.json';pp.unlink();proposal_ref=prep.save_new(pp,proposal)
            p={'frozen_source':source_ref,'frozen_protocol_sha256':'original','bank_refs':{'0':bank_ref},
                'source_dataset_sha256':'dataset','official_metric_provenance':{'kind':'frozen'}}
            m={'protocol':p,'protocol_sha256':'proto','items':[{'query_id':'c0_q0','conversation':0,'proposal':proposal_ref,'request':newreq}]}
            frozen={'protocol_sha256':'original','source_dataset_sha256':'dataset','official_metric_provenance':{'kind':'frozen'}}
            with patch.object(prep,'_load_baseline',return_value=(frozen,source_ref,[old],{0:{'ref':bank_ref}})):
                rows=prep._expected_records(prep.Reader(),run,m)
            self.assertEqual(3,len(rows))
            for old_job,method,record in rows:
                self.assertEqual(self.records,record['selected_facts'])  # all equal score: stable original order
                self.assertTrue(record['record_multiset_equal_baseline'])
                self.assertIsNone(record['selected_facts'][0]['observation_date'])
                self.assertNotIn('PRIVATE_GOLD_SENTINEL',json.dumps(self.sig))


if __name__=='__main__':unittest.main()
