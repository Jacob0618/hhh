"""Legacy payload equivalence and cache/unknown guards; fake HTTP only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
from jev_mem.evaluation import answer_and_judge, READER_SYSTEM
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import CachedLegacyAPI


KEYS = {'GEMINI_API_KEY': 'unit-test-not-a-key', 'TYPESAFE_API_KEY': 'unit-test-not-a-key'}


class Response:
    ok = True
    status_code = 200
    headers = {}
    def __init__(self, reader):
        value = ({'answer': 'A name', 'used_fact_ids': ['f1']} if reader else
                 {'correct': True, 'completeness': 1.0, 'reason': 'Equivalent.'})
        self.value = {'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': json.dumps(value)}]}}], 'usageMetadata': {'promptTokenCount': 3}}
    def json(self):
        return copy.deepcopy(self.value)


class CachedLegacyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_unchanged_evaluator_bodies_and_source_cache_zero_dispatch(self):
        calls = []
        def fake_http(session, url, **kwargs):
            calls.append({'url': url, 'body': copy.deepcopy(kwargs['json'])})
            return Response('gemini-2.5-flash:' in url)
        facts = [{'id': 'actual-id-'+str(i), 'text': 'Fact '+str(i),
                  'observation_date': '2023-01-01'} for i in range(20)]
        def evaluate(api):
            return answer_and_judge(api, 'Question?', 'Reference only for Judge', facts,
                                    reference_date='2023-02-01')
        with patch('requests.Session.post', new=fake_http):
            legacy = ModelAPI(self.root/'legacy', keys=KEYS)
            original = evaluate(legacy)
            legacy_calls = copy.deepcopy(calls)
            calls.clear()
            fresh = CachedLegacyAPI(self.root/'fresh', keys=KEYS)
            actual = evaluate(fresh)
            self.assertEqual(actual, original)
            self.assertEqual(calls, legacy_calls)
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertNotIn('responseSchema', call['body']['generationConfig'])
            calls.clear()
            inherited = CachedLegacyAPI(self.root/'consumer', keys=KEYS,
                                        cache_roots=[self.root/'legacy/api_cache'])
            self.assertEqual(evaluate(inherited), original)
            self.assertEqual(calls, [])
            self.assertEqual(len(inherited.events), 2)
            self.assertTrue(all(e['source']=='inherited_exact_request' for e in inherited.events))
            self.assertEqual(len(list((self.root/'consumer/cache_lineage').glob('*.json'))), 2)
            self.assertFalse((self.root/'consumer/request_intents').exists())
            rows=[json.loads(s) for s in (self.root/'consumer/api_calls.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['cache_hit'] is True and 'status' not in r for r in rows))

    def test_schema_rejected_and_unknown_not_redispatched(self):
        api = CachedLegacyAPI(self.root/'guards', keys=KEYS, retries=0)
        body = {'contents': [{'role': 'user', 'parts': [{'text': '{}'}]}],
                'systemInstruction': {'parts': [{'text': READER_SYSTEM}]},
                'generationConfig': {'temperature': 0, 'maxOutputTokens': 2048,
                    'responseMimeType': 'application/json', 'thinkingConfig': {'thinkingBudget': 0},
                    'responseSchema': {'type': 'OBJECT'}}}
        with patch('requests.Session.post', side_effect=AssertionError('must not dispatch')) as http:
            with self.assertRaisesRegex(ValueError, 'generation configuration'):
                api.request('gemini', 'gemini-2.5-flash', body, stage='qa_reader')
            self.assertEqual(http.call_count, 0)
        with patch('requests.Session.post', side_effect=requests.ConnectionError('fake unknown')) as http:
            kwargs = dict(system=READER_SYSTEM, json_output=True, max_output=2048, stage='qa_reader')
            with self.assertRaisesRegex(RuntimeError, 'fake unknown'):
                api.generate('{}', **kwargs)
            with self.assertRaisesRegex(RuntimeError, 'Prior unresolved logical request'):
                api.generate('{}', **kwargs)
            self.assertEqual(http.call_count, 1)
        self.assertEqual(len(list((self.root/'guards/request_intents').glob('*.json'))), 1)
        logs=[json.loads(s) for s in (self.root/'guards/api_calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['status'], 'transport_error')
        self.assertTrue(logs[0]['usage_unknown'])


if __name__=='__main__':
    unittest.main()
