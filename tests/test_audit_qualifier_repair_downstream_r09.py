"""Exact path admission, real frozen captures, and sealed offline source chain."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import audit_qualifier_repair_downstream_r09 as m
from jev_mem.local_bm25_context_v2 import forbid_network

class DownstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=m.protocol();cls.run_dir=m.completed_run()
        cls.original=m.sealed_read(cls.run_dir/'run_manifest.json')
        cls.job=next(j for j in m.parent.old.old_jobs(m.bound(cls.original['protocol']['frozen_source']))
            if j['query_id']=='c2_q3')
        cls.old=m.bound(cls.original['historical_retrieval']['selections']['cheap/c2_q3'],sealed=True)
        cls.new=m.sealed_read(m.parent.retrieval_folder(cls.run_dir,2,'cheap')/'selections/c2_q3.json')
        cls.bank=m.bound(cls.new['bank'],sealed=True)
        cls.snapshot=m.parent.old.snapshot(cls.bank['native_dir'])
    def compare(self,new):
        with forbid_network():return m.compare_pair(self.old,new,self.job,self.bank,self.snapshot)
    def reseal_capture(self,s):
        c=s['capture'];c['capture_sha256']=m.digest({k:v for k,v in c.items() if k!='capture_sha256'})
    def test_actual_pair_scores_differ_each_replay_and_full_reader_equal(self):
        before=copy.deepcopy((self.old,self.new,self.job))
        self.assertNotEqual(m.parent.selection_body(self.old),m.parent.selection_body(self.new))
        self.assertEqual(m.projection(self.old),m.projection(self.new))
        result=self.compare(self.new)
        self.assertTrue(result['full_reader_signature_and_canonical_utf8_exact'])
        self.assertFalse(result['original_strict_capture_equal'])
        self.assertEqual(before,(self.old,self.new,self.job))
    def test_forbidden_semantic_bm25_payload_or_candidate_difference_rejected(self):
        for mode in ('semantic','bm25','payload','candidate','extra','order','fact','date'):
            with self.subTest(mode=mode):
                s=copy.deepcopy(self.new);c=s['capture'];inputs=c['inputs']
                if mode=='semantic':inputs['semantic_results'][0]['score']+=1e-12
                elif mode=='bm25':
                    key=next(iter(inputs['bm25_scores']));inputs['bm25_scores'][key]+=1e-12
                elif mode=='payload':inputs['semantic_results'][0]['payload']['data']+=' '
                elif mode=='candidate':c['candidate_ids'][0]='foreign'
                elif mode=='extra':inputs['foreign_field']=True
                elif mode=='order':s['selected_ids'][0],s['selected_ids'][1]=s['selected_ids'][1],s['selected_ids'][0]
                elif mode=='fact':s['selected_facts'][0]['text']+=' '
                else:s['selected_facts'][0]['observation_date']='2099-01-01'
                self.reseal_capture(s)
                with self.assertRaises(ValueError):self.compare(s)
    def test_boost_key_nonfinite_and_derived_score_shape_rejected(self):
        for mode in ('key','nan','integer','row_extra','score_missing'):
            with self.subTest(mode=mode):
                s=copy.deepcopy(self.new);c=s['capture'];boosts=c['inputs']['entity_boosts']
                key=next(iter(boosts))
                if mode=='key':boosts['foreign']=boosts.pop(key)
                elif mode=='nan':boosts[key]=float('nan')
                elif mode=='integer':boosts[key]=1
                elif mode=='row_extra':c['eligible'][0]['foreign']=True
                else:c['eligible'][0].pop('score')
                with self.assertRaises((ValueError,KeyError)):self.compare(s)
    def test_permitted_path_wrong_derived_score_even_with_valid_seal_rejected(self):
        s=copy.deepcopy(self.new);c=s['capture'];mid=c['eligible'][0]['id']
        c['eligible'][0]['score']+=1e-9;c['base20'][0]['score']=c['eligible'][0]['score']
        c['native_scores'][mid]=c['eligible'][0]['score'];self.reseal_capture(s)
        self.assertEqual(m.projection(self.old),m.projection(s))
        with self.assertRaisesRegex(ValueError,'does not replay'):self.compare(s)
    def test_capture_invalid_hash_and_identity_rejected(self):
        for mode in ('hash','method','qid','model_calls'):
            s=copy.deepcopy(self.new)
            if mode=='hash':s['capture']['capture_sha256']='0'*64
            elif mode=='method':s['method']='jev'
            elif mode=='qid':s['query_id']='c2_q4'
            else:s['model_calls']=1
            with self.assertRaises(ValueError):self.compare(s)
    def test_full_reader_body_change_not_just_ids_rejected(self):
        actual=m.qa.request_signature;calls=[]
        def sig(job,stage):
            value=actual(job,stage);calls.append(value)
            if len(calls)==2:value=copy.deepcopy(value);value['body']['foreign']=True
            return value
        with patch.object(m.qa,'request_signature',side_effect=sig),self.assertRaisesRegex(ValueError,'Reader signature'):
            self.compare(self.new)
    def test_pair_reference_rebinding_and_protocol_change_rejected(self):
        foreign=copy.deepcopy(self.original)
        foreign['historical_retrieval']['selections']['cheap/c2_q3']['sha256']='0'*64
        with self.assertRaises(ValueError):m.pairs(self.spec,foreign)
        with patch.object(m,'PROTOCOL_SHA','0'*64),self.assertRaises(ValueError):m.protocol()
    def test_private_projection_scope_does_not_modify_frozen_globals(self):
        before=m.parent.selection_body;cache=m.parent.validate_cache_events
        scope=m.parent.parent.support.isolated(m.parent,{'selection_body':m.projection,
            'validate_cache_events':m.query.validate_cache_events})
        self.assertEqual(scope.selection_body(self.old),scope.selection_body(self.new))
        self.assertIs(m.parent.selection_body,before);self.assertIs(m.parent.validate_cache_events,cache)
    def test_real_file_prepare_gate_seal_validate_chain_and_tamper_rejection(self):
        # Only expensive cohort replay is replaced. Actual manifests/pair bytes,
        # all source construction, pointers, seals and aliases remain real.
        with tempfile.TemporaryDirectory(prefix='r09-downstream-chain-') as td,\
            patch.object(m,'output_dir',return_value=Path(td)/'derived'),\
            patch.object(m.parent,'manifest',return_value=copy.deepcopy(self.original)):
            m.prepare(self.run_dir)
            gate={'schema':'test-gate','status':'passed','model_calls':0,
                'original_strict_parity':{'status':'FAILED','total':1900,'exact':604,'different':1296},
                'downstream_reader_exact':{'total':1900,'equal':1900,'different':0},
                'capture_self_replay':{'historical':1900,'current':3080}}
            with patch.object(m,'gate_value',return_value=gate),forbid_network():
                g=m.audit(self.run_dir);source=m.seal(self.run_dir)
                s=m.validate_source(source['path'])
                self.assertEqual(len(s['jobs']),4620)
                self.assertEqual(s['retrieval_gate'],g)
                self.assertEqual(s['query_event_audit_manifest'],s['downstream_admission_manifest'])
                self.assertEqual(s['index_admission'],s['effective_bank_gate'])
                q=next(j for j in s['jobs'] if j['query_id']=='c1_q22' and j['method']=='jev')
                self.assertTrue(q['question'].endswith(' '))
                self.assertEqual(s['code_sha256'],self.original['code_sha256'])
                path=Path(source['path']);v=m.body(m.sealed_read(path));v['retrieval_gate']['sha256']='0'*64
                path.write_text(json.dumps({**v,'content_sha256':m.digest(v)}),encoding='utf-8')
                with self.assertRaises(ValueError):m.validate_source(path)

if __name__=='__main__':unittest.main()
