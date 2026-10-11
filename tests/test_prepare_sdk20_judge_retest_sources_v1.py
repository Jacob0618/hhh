"""No models: fixed original answers, blocked arrays and exact source identities."""
import copy
import json
import unittest
from unittest.mock import patch

from tools import prepare_sdk20_judge_retest_sources_v1 as p


class SourceTests(unittest.TestCase):
    def success(self, text=' \nParis\n'):
        row={'status':'complete','result':{'prediction':{'answer':text},'exact_match':0.,
             'judgment':{'correct':False,'completeness':.5,'reason':'original'}},'error':None}
        outcome={'scores':{'official_locomo_f1':.75,'custom_token_f1':.5}}
        return row,outcome,{'answer':text,'used_fact_ids':['f1']}

    def failure(self):
        return ({'status':'failed','result':None,'error':{'type':'ValueError','message':'Reader did not return an answer string'}},
                {'scores':None},{'answer':['Paris','Rome'],'used_fact_ids':['f1','f2']})

    def test_original_string_old_judge_and_f1_preserved_exactly(self):
        row,out,pred=self.success(); before=copy.deepcopy((row,out,pred));value=p.answer_fields(row,out,pred)
        self.assertEqual(value['answer'],' \nParis\n');self.assertEqual(value['answer_sha256'],p.text_sha(value['answer']))
        self.assertEqual(value['old_judge'],row['result']['judgment']);self.assertEqual(value['F1']['official_locomo_f1'],.75)
        self.assertEqual(value['reader_status'],'complete');self.assertEqual((row,out,pred),before)

    def test_looks_like_json_string_is_not_reparsed(self):
        inputs=self.success('{"answer":"Paris"}')
        self.assertEqual(p.answer_fields(*inputs)['answer'],inputs[2]['answer'])

    def test_original_arrays_remain_blocked_main_with_null_scores(self):
        row,out,pred=self.failure();value=p.answer_fields(row,out,pred)
        self.assertEqual(value['status'],'blocked_original_reader');self.assertIsNone(value['answer'])
        self.assertEqual(value['pending_original_list'],['Paris','Rome']);self.assertIsNone(value['old_judge'])
        self.assertTrue(all(v is None for v in value['F1'].values()))

    def test_deferred_normalization_keeps_order_and_whitespace_without_new_reader(self):
        pred={'answer':['  Paris','Rome\nItaly','Rome\nItaly'],'used_fact_ids':['f1','f1']}; before=copy.deepcopy(pred)
        value=p.deferred_resolution(pred,{'reference':'Paris','category':1},lambda *args:.625)
        self.assertEqual(value['normalized_answer'],'  Paris\nRome\nItaly\nRome\nItaly')
        self.assertEqual(value['normalization_proof']['original_answer'],pred['answer'])
        self.assertEqual(value['F1']['official_locomo_f1'],.625);self.assertEqual(pred,before)
        self.assertEqual(value['reader_model_calls'],0);self.assertTrue(value['main_firstpass_metrics_remain_null'])

    def test_raw_prediction_drift_or_other_failure_is_rejected(self):
        row,out,pred=self.success();pred['answer']='London'
        with self.assertRaises(ValueError):p.answer_fields(row,out,pred)
        row,out,pred=self.failure();row['error']['type']='RuntimeError'
        with self.assertRaises(ValueError):p.answer_fields(row,out,pred)
        row,out,pred=self.failure();pred['answer']=['Paris',4]
        with self.assertRaises(ValueError):p.answer_fields(row,out,pred)
        with self.assertRaises(ValueError):p.deferred_resolution({'answer':['X'],'used_fact_ids':['f21']},{'reference':'X','category':1},lambda *a:1.)

    def test_original_model_generate_parser_and_full_signature_replay(self):
        job={'question':'Where?','reference':'Paris','reference_date':'2023-06-30',
             'selected_facts':[{'id':f'f{i}','text':f'Fact{i}','observation_date':None} for i in range(20)]}
        sig=p.recovery.expected_reader(job);prediction={'answer':['Paris','Rome'],'used_fact_ids':['f1']}
        raw={'request':sig,'stage':'qa_reader','response':{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(prediction)}]}}]}}
        self.assertEqual(p.reader_prediction(raw),prediction)
        raw['response']['candidates'][0]['finishReason']='MAX_TOKENS'
        with self.assertRaises(RuntimeError):p.reader_prediction(raw)
        raw['stage']='another_stage'
        with self.assertRaises(ValueError):p.reader_prediction(raw)

    def test_new_judge_signature_uses_answer_verbatim_and_plain_protocol(self):
        answer=' Paris\nRome '
        sig=p.judge_signature('Where?','Paris',answer);body=sig['body']
        self.assertEqual(sig['model'],'gemini-3.5-flash-lite')
        self.assertNotIn('responseMimeType',body['generationConfig']);self.assertNotIn('responseSchema',body['generationConfig'])
        self.assertIn('[The Start of Model’s Response]\n'+answer+'\n[The End of Model’s Response]',body['contents'][0]['parts'][0]['text'])

    def population(self):
        jobs=[]; outcomes={}
        for arm in p.ARMS:
            for i in range(1540):
                status='ready' if i<p.EXPECTED[arm]['ready'] else 'blocked_original_reader'
                jobs.append({'method':arm,'query_id':f'q{i}','status':status});outcomes[arm,f'q{i}']={}
        return {'jobs':jobs,'ready_judges':3042,'blocked_original_readers':38},outcomes

    def test_exact_two_arm_population_and_real_status_counts_required(self):
        v,out=self.population();p.validate_main_jobs(v,out)
        for change in ('duplicate','third_arm','wrong_status','missing'):
            x=copy.deepcopy(v)
            if change=='duplicate':x['jobs'][-1]=x['jobs'][0]
            elif change=='third_arm':x['jobs'][0]['method']='preseg_sdk20'
            elif change=='wrong_status':x['jobs'][0]['status']='blocked_original_reader'
            else:x['jobs'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):p.validate_main_jobs(x,out)


if __name__=='__main__':unittest.main()
