from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests

from tools import recover_sdk20_modules as r


def response(value,*,stop='STOP'):
    return {'candidates':[{'finishReason':stop,'content':{'parts':[{'text':json.dumps(value)}]}}],
            'usageMetadata':{'promptTokenCount':7,'candidatesTokenCount':4}}


class FakeHTTP:
    def __init__(self,value,status=200):
        self.value=value;self.status_code=status;self.ok=200<=status<400;self.headers={};self.text='mock'
    def json(self):return deepcopy(self.value)


class Network:
    def __init__(self,readers,judges):self.readers=list(readers);self.judges=list(judges);self.calls=[]
    def post(self,url,headers,json,timeout):
        stage='strict_judge' if 'gemini-3.5-flash-lite' in url else 'qa_reader'
        self.calls.append((stage,deepcopy(json)))
        value=(self.judges if stage=='strict_judge' else self.readers).pop(0)
        if isinstance(value,BaseException):raise value
        return value if isinstance(value,FakeHTTP) else FakeHTTP(value)


class RecoveryTests(unittest.TestCase):
    def job(self,method='fact_types_sdk20',qid='c0_q0'):
        j={'schema':r.SCHEMA,'conversation':0,'method':method,'query_id':qid,'category':4,
           'question':'Who?','reference':'Alice','reference_date':'2023-10-22',
           'selected_facts':[{'id':f'x{i}','text':'Alice '+str(i),'observation_date':'2023-01-01'} for i in range(20)],
           'source_qa':{'path':'old.json','sha256':'old'},'_sha256':'test-job'}
        j['selected_ids']=[f['id'] for f in j['selected_facts']]
        j['expected_reader_signature']=r.expected_reader(j);j['expected_reader_hash']=r.request_digest(j['expected_reader_signature'])
        return j

    def run_case(self,root,j,network):
        with patch('jev_mem.providers.requests.Session',return_value=network),patch('jev_mem.providers.time.sleep'), \
             patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credential reads in tests')):
            return r.execute_consumer(root,j,{'protocol_sha256':'test-protocol'},keys={'GEMINI_API_KEY':'test-placeholder','TYPESAFE_API_KEY':'test-placeholder'})

    def test_body_exact_original_and_raw_failure_probe(self):
        j=self.job();body=j['expected_reader_signature']['body']
        self.assertNotIn('responseSchema',body['generationConfig'])
        inp=json.loads(body['contents'][0]['parts'][0]['text'])
        self.assertEqual(inp['conversation_reference_date'],j['reference_date'])
        self.assertEqual([f['fact'] for f in inp['facts']],[f['text'] for f in j['selected_facts']])
        for invalid in (17,['Alice'],None):
            with self.assertRaisesRegex(ValueError,'answer string'):r.probe_reader(j,response({'answer':invalid}))

    def test_reader_format_recovery_locks_first_valid_and_false_judge_is_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([response({'answer':17}),response({'answer':'Alice','used_fact_ids':['f1']})],
                [response({'correct':False,'completeness':0.0,'reason':'wrong'})])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status'],'complete');self.assertFalse(out['result']['judgment']['correct'])
            self.assertEqual(Counter(x[0] for x in network.calls),{'qa_reader':2,'strict_judge':1})
            lock=r.verify(out['reader_lock']['path']);self.assertEqual(lock['logical_attempt'],2)
            self.assertEqual(network.calls[0][1],network.calls[1][1])
            before=len(network.calls);again=self.run_case(Path(tmp),j,network)
            self.assertEqual(again['result'],out['result']);self.assertEqual(len(network.calls),before)

    def test_judge_format_retries_only_same_judge_never_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([response({'answer':'Alice','used_fact_ids':['f2']})],
                [response({'correct':'yes'}),response({'correct':True,'completeness':True,'reason':'x'}),
                 response({'correct':True,'completeness':1.0,'reason':'yes'})])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status'],'complete');self.assertEqual(Counter(x[0] for x in network.calls),{'qa_reader':1,'strict_judge':3})
            judge_bodies=[x[1] for x in network.calls if x[0]=='strict_judge']
            self.assertTrue(all(b==judge_bodies[0] for b in judge_bodies))
            self.assertEqual(out['result']['prediction']['used_fact_ids'],['x1'])

    def test_three_invalid_reader_responses_exhaust_without_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([response({'answer':17})]*3,[])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status'],'failed');self.assertIsNone(out['result'])
            self.assertEqual(out['status_reason'],'reader_format_attempts_exhausted')
            self.assertEqual(len(network.calls),3);self.assertIsNone(out['reader_lock'])

    def test_reader_first_valid_then_judge_invalid_then_false_does_not_regenerate_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([response({'answer':'Alice'})],
                [response({'correct':'no'}),response({'correct':False,'completeness':0,'reason':'wrong'})])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status'],'complete');self.assertFalse(out['result']['judgment']['correct'])
            self.assertEqual([s for s,b in network.calls],['qa_reader','strict_judge','strict_judge'])
            self.assertEqual(network.calls[1][1],network.calls[2][1])
            self.assertEqual(r.verify(out['reader_lock']['path'])['logical_attempt'],1)

    def test_network_exhaustion_stops_logical_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([requests.Timeout('mock transport failure')]*3,[])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status_reason'],'reader_technical_failed');self.assertEqual(len(network.calls),3)
            self.assertEqual(len(out['attempt_records']),1)
            self.assertFalse((Path(tmp)/'transport/qa_reader/attempt_2').exists())

    def test_incomplete_generation_not_format_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=self.job();network=Network([response({'answer':'Alice'},stop='MAX_TOKENS')],[])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['status_reason'],'reader_technical_failed');self.assertEqual(len(network.calls),1)

    def test_unknown_intent_without_cache_does_not_redispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);j=self.job();p=root/'transport/qa_reader/attempt_1/request_intents'/(j['expected_reader_hash']+'.json')
            p.parent.mkdir(parents=True);p.write_text(json.dumps({'request_hash':j['expected_reader_hash'],
                'signature':j['expected_reader_signature'],'context':{},'stage':'qa_reader','retries':2}),encoding='utf8')
            network=Network([],[]);out=self.run_case(root,j,network)
            self.assertEqual(out['status_reason'],'reader_technical_failed');self.assertEqual(network.calls,[])
            record=r.verify(out['attempt_records'][0]['path']);self.assertEqual(record['api_events'],[])
            self.assertEqual(record['transport_intent']['path'],str(p.resolve()))

    def test_shared_signature_uses_same_physical_cache_across_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);one=self.job();two=self.job(method='preseg_sdk20')
            network=Network([response({'answer':'Alice'})],[response({'correct':True,'completeness':1,'reason':'same'})])
            a=self.run_case(root,one,network);b=self.run_case(root,two,network)
            self.assertEqual(a['result'],b['result']);self.assertEqual(len(network.calls),2)
            for p in b['attempt_records']:
                record=r.verify(p['path']);self.assertEqual(record['api_events'][0]['source'],'existing_run_cache')

    def test_json_parser_failure_retries_but_no_number_coercion(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken=response({});broken['candidates'][0]['content']['parts'][0]['text']='not JSON'
            j=self.job();network=Network([broken,response({'answer':'17'})],[response({'correct':False,'completeness':0,'reason':'wrong'})])
            out=self.run_case(Path(tmp),j,network)
            self.assertEqual(out['result']['prediction']['answer'],'17')
            first=r.verify(out['attempt_records'][0]['path']);self.assertEqual(first['error']['type'],'JSONDecodeError')

if __name__=='__main__':unittest.main()
