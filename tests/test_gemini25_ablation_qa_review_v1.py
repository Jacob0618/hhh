"""Independent QA regression: real provider bookkeeping with mock HTTP only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_gemini25_ablation_qa_v1 as fixtures
from tools import run_gemini25_ablation_qa_v1 as runner
from tools import summarize_gemini25_ablation_v1 as summary
from jev_mem.gemini25_ablation_provider_v1 import Gemini25API,MODEL


class MockHTTP:
    def post(self,url,**kwargs):
        answer='alpha' if 'systemInstruction' in kwargs['json'] else '[[no]]'
        raw={'modelVersion':'models/'+MODEL,'usageMetadata':{'promptTokenCount':9,'candidatesTokenCount':3},
             'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':answer}]}}]}
        return type('Response',(),{'ok':True,'status_code':200,'headers':{},'json':lambda self:raw})()


class QAIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.create_connection','requests.Session.request','jev_mem.providers.load_credentials'):
            blocker=patch(target,side_effect=AssertionError('No network or credentials'))
            blocker.start();self.addCleanup(blocker.stop)

    def test_resource_spelling_survives_real_provider_ledger_and_full_raw_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'first'
            def make_api(folder,label):
                return Gemini25API(folder,keys={'GEMINI_API_KEY':'offline-placeholder'},gate_path=root/'gate.sqlite')
            with patch.object(fixtures,'RecordedFake',side_effect=make_api),\
                 patch('requests.Session',side_effect=MockHTTP):
                m,out=fixtures.phase(run,[fixtures.job(arm) for arm in runner.ARMS])
            path=run/'conv_0/api/api_calls.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual({row['resolved_model'] for row in rows},{'models/'+MODEL})
            original_bytes=path.read_bytes()
            with patch.object(runner,'terminal',return_value=(m,out)):
                audited=summary.audit_phase(run,{})
            self.assertEqual(audited['errors'],[])
            self.assertEqual(path.read_bytes(),original_bytes)
            self.assertEqual(runner.recovery_jobs(m,out),[])
            usage=summary.measured_usage(audited['ledger'])
            self.assertEqual((usage['physical_attempts'],usage['cache_events']),(2,4))
            self.assertEqual(usage['token_fields']['thinking']['missing_attempts'],2)
            self.assertIsNone(usage['total_cost_usd'])
            # Both spellings are allowed model identities, but producer and
            # consumer ledgers must still preserve the exact raw spelling.
            for row in rows:row['resolved_model']=MODEL
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
            with patch.object(runner,'terminal',return_value=(m,out)):
                mismatched=summary.audit_phase(run,{})
            self.assertTrue(any(row['stage']=='reader' for row in mismatched['errors']))

    def test_usage_unknown_and_cache_values_are_not_filled_or_billed(self):
        values=[{'model':MODEL,'status':'transport_error','attempt':1,'usage_unknown':True},
                {'model':MODEL,'status':200,'attempt':2,'usage':{'promptTokenCount':5}},
                {'model':MODEL,'cache_hit':True,'usage':{'promptTokenCount':900,'totalTokenCount':999}}]
        result=summary.measured_usage(values)
        self.assertEqual((result['physical_attempts'],result['cache_events']),(2,1))
        self.assertEqual(result['token_fields']['prompt']['recorded_token_sum'],5)
        self.assertEqual(result['token_fields']['prompt']['missing_attempts'],1)
        self.assertEqual(result['token_fields']['total']['missing_attempts'],2)
        self.assertEqual(result['token_fields']['thinking']['missing_attempts'],2)
        self.assertIsNone(result['total_cost_usd'])


if __name__=='__main__':unittest.main()
