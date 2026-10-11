"""Offline boundary tests for the new terminal auditor; no original run reads."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from copy import deepcopy
from tools import audit_simplemem_terminal_v2 as a


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='simplemem-terminal-audit-')
        self.root = Path(self.tmp.name).resolve()
        self.context = {'method': 'simplemem', 'conversation': 0, 'phase': 'retrieval', 'query_id': 'c0_q0'}
        self.stage = 'simplemem_query_analysis'
        self.signature = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
            'body': {'contents': [{'role': 'user', 'parts': [{'text': 'original question'}]}],
                     'generationConfig': {'maxOutputTokens': 16384, 'thinkingConfig': {'thinkingBudget': 0}}}}
        self.key = a.digest(self.signature)
        self.intent = {'request_hash': self.key, 'signature': self.signature, 'context': self.context, 'stage': self.stage, 'retries': 2,
                       'created_utc': '2026-01-01T00:00:00+00:00'}
        self.raw = {'request': self.signature, 'response': {'candidates': [{'content': {'role': 'model', 'parts': [{'text': '{}'}]}, 'finishReason': 'STOP'}]}}
        self.write(self.root/'request_intents'/f'{self.key}.json', self.intent)
        self.write(self.root/'api_cache'/f'{self.key}.json', self.raw)
        self.call = dict(a.identity({**self.intent, **{k:self.signature[k] for k in ('provider','model','operation')}}), attempt=1, status=200,
                         timestamp_utc='2026-01-01T00:00:01+00:00', usage={}, resolved_model=None)
        self.event = {**a.identity(self.call), 'source': 'new_request', 'cache_path': str(self.root/'api_cache'/f'{self.key}.json'),
                      'cache_sha256': a.file_sha(self.root/'api_cache'/f'{self.key}.json'), 'error_type': None, 'recorded_utc':'2026-01-01T00:00:02+00:00'}
        self.lines('api_calls.jsonl', [self.call]); self.lines('logical_events.jsonl', [self.event])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')

    def lines(self, name, values):
        (self.root/name).write_text(''.join(json.dumps(v)+'\n' for v in values), encoding='utf8')

    def test_exact_cache_multiplicity(self):
        cached = {**self.call, 'cache_hit': True, 'usage_semantics':'reused_response_no_new_model_usage'}; cached.pop('attempt'); cached.pop('status')
        consumer = {**self.event, 'source': 'existing_run_cache', 'context': {**self.context, 'query_id': 'c0_q1'}}
        cached['context'] = consumer['context']
        self.lines('api_calls.jsonl', [self.call, cached]); self.lines('logical_events.jsonl', [self.event, consumer])
        summary, _, _ = a.audit_api(self.root)
        self.assertEqual(summary['cache_consumers'], 1)
        self.lines('logical_events.jsonl', [self.event, consumer, consumer])
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            a.audit_api(self.root)

    def test_known_retry_and_unknown_http(self):
        self.lines('api_calls.jsonl', [{**self.call, 'status': 'transport_error'}, {**self.call, 'attempt': 2}])
        self.assertEqual(len(a.audit_api(self.root)[0]['retry_chains']), 1)
        self.lines('api_calls.jsonl', [{**self.call, 'status': 400}, {**self.call, 'attempt': 2}])
        with self.assertRaisesRegex(ValueError, 'Unknown/unresolved'):
            a.audit_api(self.root)

    def test_retry_context_and_attempt_order(self):
        wrong = {**self.call, 'attempt': 2, 'context': {**self.context, 'query_id': 'c0_q2'}}
        self.lines('api_calls.jsonl', [{**self.call, 'status': 429}, wrong])
        with self.assertRaisesRegex(ValueError, 'identity'):
            a.audit_api(self.root)
        self.lines('api_calls.jsonl', [{**self.call, 'attempt': 2}])
        with self.assertRaisesRegex(ValueError, 'order/limit'):
            a.audit_api(self.root)

    def test_orphan_raw_and_consumer_sha(self):
        self.lines('logical_events.jsonl', [{**self.event, 'cache_sha256': '0'*64}])
        with self.assertRaisesRegex(ValueError, 'Consumer raw'):
            a.audit_api(self.root)
        self.lines('logical_events.jsonl', [self.event]); self.write(self.root/'api_cache'/'orphan.json', {})
        with self.assertRaisesRegex(ValueError, 'key sets'):
            a.audit_api(self.root)

    def certificate(self, stage, text, message):
        record = {'raw_value': {'response': {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}, 'finishReason': 'STOP'}]}}}
        if stage == 'simplemem_query_analysis':
            from jev_mem.gemini25_ablation_provider_v1 import generation_body
            messages = a.analysis_messages('original question')
            record['signature'] = {'body':generation_body(messages[1]['content'], system=messages[0]['content'], json_output=False, max_output=16384)}
        prefix = [{'stage': stage, 'request_hash': 'key', 'raw': {'path': 'original', 'sha256': '1'*64}, 'ordinal': 2}]
        return a.failure_certificate({'query_id': 'c0_q0', 'conversation': 0, 'question': 'original question'},
            {'error': {'type': 'BaselineFatal', 'message': message}}, prefix, {'key': record})

    def test_only_exact_empty_missing_queries(self):
        cert = self.certificate('simplemem_missing_queries', '{"targeted_queries": []}', 'SimpleMem retrieve: ValueError: Invalid targeted_queries')
        self.assertEqual(cert['kind'], 'native_empty_missing_queries')
        self.assertFalse(cert['paid_regeneration_permitted'])
        for value in ('{}', '{"targeted_queries": null}', '{"targeted_queries": ""}', '{"targeted_queries": [""]}'):
            with self.assertRaises(ValueError):
                self.certificate('simplemem_missing_queries', value, 'SimpleMem retrieve: ValueError: Invalid targeted_queries')

    def test_field_error_exact_binding(self):
        cert = self.certificate('simplemem_query_analysis', '{"keywords":["CS:GO"]}', "SimpleMem retrieve: ValueError: Field does not exist: 'CS'")
        self.assertEqual(cert['kind'], 'native_keyword_field_empty')
        with self.assertRaises(ValueError):
            self.certificate('simplemem_query_analysis', '{"keywords":["CS:GO"]}', "SimpleMem retrieve: ValueError: Field does not exist: 'OTHER'")

    def test_json_default_scope_is_only_analysis(self):
        self.assertEqual(self.certificate('simplemem_query_analysis', '{broken', 'SimpleMem retrieve: JSONDecodeError: bad')['kind'], 'native_analysis_cached_default')
        with self.assertRaises(ValueError):
            self.certificate('simplemem_query_plan', '{broken', 'SimpleMem retrieve: JSONDecodeError: bad')

    def test_incomplete_jsonl_hard_rejects_terminal(self):
        (self.root/'api_calls.jsonl').write_bytes(b'{"partial":')
        with self.assertRaisesRegex(ValueError, 'Incomplete terminal'):
            a.audit_api(self.root)

    def test_retry_limit_usage_and_timestamp_are_frozen(self):
        self.write(self.root/'request_intents'/f'{self.key}.json', {**self.intent, 'retries':3})
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            a.audit_api(self.root)
        self.write(self.root/'request_intents'/f'{self.key}.json', self.intent)
        self.lines('api_calls.jsonl', [{**self.call, 'usage': {'promptTokenCount':99}}])
        with self.assertRaisesRegex(ValueError, 'usage/model'):
            a.audit_api(self.root)
        self.lines('api_calls.jsonl', [{**self.call, 'timestamp_utc':'2025-01-01T00:00:00+00:00'}])
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            a.audit_api(self.root)

    def test_real_verified_venv_parent_is_allowed(self):
        p = self.root/'run_five_baselines_simplemem_native_v2.py'
        p.write_text("import sys\nsys.path.insert(0, 'D:/jev mem')\nfrom tools.audit_simplemem_terminal_v2 import assert_stopped\nassert_stopped()\nprint('verified-real-venv-parent')\n", encoding='utf8')
        run = subprocess.run([str(a.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'), '-B', str(p), 'run'], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertIn('verified-real-venv-parent', run.stdout)

    def test_native_payload_ignores_only_clock_and_preserves_prompt(self):
        from jev_mem.gemini25_ablation_provider_v1 import generation_body
        event = {'utc':'2026-01-01T00:00:01+00:00', 'event':'generation_request', 'stage':self.stage,
                 'prompt':'original question', 'system':'original system', 'max_output':16384, 'temperature_sent':False}
        event['request_id'] = a.digest({k:event[k] for k in ('stage','prompt','system')}, True)
        signature = a.generation_signature(event); key = a.digest(signature)
        answer = {'utc':'2026-01-01T00:00:02+00:00', 'event':'generation_response', 'request_id':event['request_id'], 'stage':self.stage, 'response':'{}'}
        logical = {**self.event, 'request_hash':key}
        record = {'signature':signature,'raw':{'path':'raw','sha256':'1'*64},'intent':{'path':'intent','sha256':'2'*64},
                  'raw_value':self.raw}
        p = self.root/'native.requests.jsonl'
        p.write_text(json.dumps(event)+'\n'+json.dumps(answer)+'\n',encoding='utf8')
        first = a.audit_native(p,[logical],{key:record},self.context)['prefix'][0]
        event['utc']='2026-02-01T00:00:01+00:00';answer['utc']='2026-02-01T00:00:02+00:00'
        p.write_text(json.dumps(event)+'\n'+json.dumps(answer)+'\n',encoding='utf8')
        second = a.audit_native(p,[logical],{key:record},self.context)['prefix'][0]
        self.assertNotEqual(first['native_request_sha256'],second['native_request_sha256'])
        self.assertNotEqual(first['native_response_sha256'],second['native_response_sha256'])
        self.assertEqual(first['native_request_payload_sha256'],second['native_request_payload_sha256'])
        self.assertEqual(first['native_response_payload_sha256'],second['native_response_payload_sha256'])
        event['prompt']='changed question'
        p.write_text(json.dumps(event)+'\n'+json.dumps(answer)+'\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'request_id'):
            a.audit_native(p,[logical],{key:record},self.context)


if __name__ == '__main__':
    unittest.main()
