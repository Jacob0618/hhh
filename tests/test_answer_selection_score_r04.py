import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import score_answer_selection_r04 as d
from jev_mem.answer_selection_r04 import presentation


def record(identical=False):
    sig = {'provider':'jev','model':'jev-1.13.0','operation':'systemone','body':{
        'model':'jev-1.13.0','state':{'question':'Q','reference_date':'2023',
            'memories':[], 'candidate_first':'answer A','candidate_second':'answer B'},
        'questions':{'answer_choice':{'type':'choice','instructions':'frozen mock',
            'criteria':{k:k for k in ['first_better','second_better','no_clear_preference','neither_supported']}}}}}
    return dict(signature=sig,request_hash=d.request_hash(sig),query_id='c0_q0',
        protocol_sha256='mock',identical=identical,state=sig['body']['state'],
        presentation=presentation('c0_q0'), candidates={
            'baseline':{'answer':'answer A','reader':{'path':'original A','sha256':'A'}},
            'jev_r03':{'answer':'answer A' if identical else 'answer B',
                       'reader':{'path':'original B','sha256':'B'}}})


class MockAPI:
    def __init__(self, root, mode='good'):
        self.run_dir=root
        self.cache_dir=root/'api_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.mode=mode
        self.calls=0
        self.context={}
    def scoped(self, **context):
        self.context=context
        return self
    def request(self, provider,model,body,**kw):
        self.calls+=1
        sig=dict(provider=provider,model=model,body=body,operation=kw['operation'])
        key=d.request_hash(sig)
        d.publish(self.run_dir/'request_intents'/(key+'.json'),dict(request_hash=key,
            signature=sig,stage=d.JEV_STAGE,retries=2,context=self.context))
        if self.mode in ('transport','partial','wrong_context','unknown'):
            rows=[]
            if self.mode!='unknown':
                for attempt in range(1,3 if self.mode=='partial' else 4):
                    rows.append(dict(request_hash=key,stage=d.JEV_STAGE,provider=provider,
                        model=model,operation=kw['operation'],attempt=attempt,
                        status='transport_error',context={'query_id':'unrelated'}
                        if self.mode=='wrong_context' else self.context))
                (self.run_dir/'api_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            raise RuntimeError('mock failure')
        response={'model':model,'answers':{'answer_choice':{'type':'choice',
            'choice':'second_better','probabilities':{'first_better':.1,'second_better':.7,
                'no_clear_preference':.1,'neither_supported':.1},'confidence':.2}}}
        if self.mode=='invalid':response['answers']['answer_choice']['confidence']=None
        d.publish(self.cache_dir/(key+'.json'),{'request':sig,'response':response})
        return response


class DispatchTests(unittest.TestCase):
    def invoke(self, mode, expected):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r=record();p=root/'requests/c0_q0.json';d.publish(p,r)
            api=MockAPI(root,mode)
            result=d.one(root,p,api)
            self.assertEqual(result['status'],expected)
            self.assertEqual(api.calls,1)
            self.assertEqual(result['selected_exact_answer'],r['candidates'][result['selected_route']]['answer'])
            if expected!='complete':
                self.assertIsNone(result['choice_result'])
                self.assertEqual(result['selected_route'],'baseline')
            with self.assertRaises(FileExistsError):d.one(root,p,api)
            self.assertEqual(api.calls,1)
    def test_valid(self):self.invoke('good','complete')
    def test_invalid_not_zero(self):self.invoke('invalid','failed')
    def test_transport_full(self):self.invoke('transport','failed')
    def test_transport_partial(self):self.invoke('partial','unknown')
    def test_other_context(self):self.invoke('wrong_context','unknown')
    def test_unfinished(self):self.invoke('unknown','unknown')
    def test_identical_skips(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'requests/c0_q0.json';d.publish(p,record(True));api=MockAPI(root)
            result=d.one(root,p,api)
            self.assertEqual(result['status'],'identical_answer_no_dispatch')
            self.assertEqual(result['selected_route'],'baseline')
            self.assertEqual(api.calls,0)
            self.assertIsNone(result['actual_request_hash'])
    def test_oversize_not_sent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'requests/c0_q0.json';d.publish(p,record());api=MockAPI(root)
            with patch.object(d,'MAX_BYTES',1):result=d.one(root,p,api)
            self.assertEqual(result['status'],'not_dispatched')
            self.assertEqual(api.calls,0)
    def test_tamper_no_request(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'requests/c0_q0.json';r=record();r['request_hash']='bad';d.publish(p,r);api=MockAPI(root)
            with self.assertRaises(ValueError):d.one(root,p,api)
            self.assertEqual(api.calls,0)
    def test_question_gate(self):
        for q in ({},{'bad':{'type':'choice'}},{'answer_choice':{'type':'noul'}}):
            with self.assertRaises(ValueError):d.expected_keys({'questions':q})
    def test_no_gemini_entry(self):
        api=object.__new__(d.GateAPI)
        with self.assertRaises(ValueError):api.request('gemini','gemini-3.8-flash',{})


if __name__=='__main__':unittest.main()
