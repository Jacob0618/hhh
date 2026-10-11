"""Offline model-only adapter tests; no network, credentials or live models."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from jev_mem import gemini38_provider_v1 as frozen_provider
from jev_mem import evaluation_gemini38_v1 as frozen_evaluation
from jev_mem import gemini25_ablation_provider_v1 as provider
from jev_mem import evaluation_gemini25_ablation_v1 as evaluation
from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from jev_mem.providers import ModelAPI
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    'jev_mem/gemini38_provider_v1.py': 'd27799022569c526c4237009b21335642048f9fd1a9161dcbaed4ba5181aaffe',
    'jev_mem/evaluation_gemini38_v1.py': 'e2693f7ac8fc2c8253a2340dcfcdd77f352909a944a0db31d9d6722a33e65527',
}
KEYS = {'GEMINI_API_KEY': 'offline-gemini-placeholder',
        'TYPESAFE_API_KEY': 'offline-jev-placeholder'}
FACTS = [{'id': 'native-id', 'text': 'Ada visited Paris.',
          'observation_date': '2026-10-01'}]


def response(text='Paris', *, model=provider.MODEL, finish='STOP', parts=None):
    return {'modelVersion': model,
            'candidates': [{'finishReason': finish, 'content': {
                'parts': [{'text': text}] if parts is None else parts}}]}


class CaptureAPI:
    generate = provider.Gemini25API.generate

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def request(self, kind, model, body, *, stage):
        provider.check_request(kind, model, body, 'generateContent', stage)
        self.calls.append({'provider': kind, 'model': model,
                           'body': deepcopy(body), 'stage': stage})
        return self.outputs.pop(0)


class Session:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def post(self, url, **kwargs):
        self.calls.append((url, deepcopy(kwargs)))
        if not self.outputs:
            raise AssertionError('Unexpected extra physical dispatch')
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return type('MockResponse', (), {
            'ok': True, 'status_code': 200, 'headers': {},
            'json': lambda self: deepcopy(output),
        })()


class Gemini25AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assert_frozen()

    @classmethod
    def tearDownClass(cls):
        cls.assert_frozen()

    @classmethod
    def assert_frozen(cls):
        if {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in FROZEN} != FROZEN:
            raise AssertionError('Frozen Gemini 3.8 sources changed')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='jev_gemini25_offline_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for target in ('requests.Session.request', 'requests.Session.post',
                       'socket.create_connection', 'jev_mem.providers.load_credentials'):
            blocker = patch(target, side_effect=AssertionError('No network or vault access'))
            blocker.start()
            self.addCleanup(blocker.stop)

    def api(self, outputs):
        api = provider.Gemini25API(self.root / 'run', keys=KEYS,
                                   gate_path=self.root / 'gate.sqlite')
        session = Session(outputs)
        api.local.session = session
        return api, session

    def read(self, api):
        return evaluation.read_answer(api, 'Where did Ada visit?', 'PRIVATE_GOLD',
                                      FACTS, reference_date='2026-10-07')

    def test_import_is_offline_and_frozen_globals_remain_identical(self):
        before = [dict(vars(module)) for module in (frozen_provider, frozen_evaluation)]
        importlib.reload(provider)
        importlib.reload(evaluation)
        for module, snapshot in zip((frozen_provider, frozen_evaluation), before):
            self.assertEqual(set(vars(module)), set(snapshot))
            for key, value in snapshot.items():
                self.assertIs(vars(module)[key], value, key)
        self.assertEqual(frozen_provider.MODEL, 'gemini-3.8-flash')
        self.assertEqual(frozen_provider.generation_config(8192), {
            'maxOutputTokens': 8192, 'thinkingConfig': {'thinkingLevel': 'low'}})

    def test_scientific_code_objects_and_prompts_are_reused(self):
        self.assertIs(provider.generation_body.__code__, frozen_provider.generation_body.__code__)
        self.assertIs(provider._check_request.__code__, frozen_provider.check_request.__code__)
        self.assertIs(provider._response_text.__code__, frozen_provider.response_text.__code__)
        self.assertIs(provider.Gemini25ExtractionLLM.generate_response.__code__,
                      frozen_provider.Gemini38ExtractionLLM.generate_response.__code__)
        for name in ('read_answer', 'judge_answer', 'answer_and_judge'):
            self.assertIs(getattr(evaluation, name).__code__, getattr(frozen_evaluation, name).__code__)
            self.assertIsNot(getattr(evaluation, name).__globals__, vars(frozen_evaluation))
        for name in ('make_judge_prompt', 'parse_judge_label', 'reader_input', 'lexical_metrics'):
            self.assertIs(getattr(evaluation, name), getattr(frozen_evaluation, name))
        self.assertEqual(evaluation.JUDGE_TEMPLATE_SHA256, frozen_evaluation.JUDGE_TEMPLATE_SHA256)
        self.assertEqual(evaluation.READER_SYSTEM, frozen_evaluation.READER_SYSTEM)

    def test_writer_body_and_native_serializer_keep_frozen_contract(self):
        api = CaptureAPI([response('{"memory": [{"text": "访问 Paris"}]}')])
        llm = provider.Gemini25ExtractionLLM(api)
        output = llm.generate_response([
            {'role': 'system', 'content': ADDITIVE_EXTRACTION_PROMPT},
            {'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'second'},
        ])
        self.assertEqual(llm.config.model, provider.MODEL)
        self.assertEqual(json.loads(output), {'memory': [{'text': '访问 Paris'}]})
        call = api.calls[0]
        self.assertEqual(call['model'], provider.MODEL)
        self.assertEqual(call['stage'], 'mem0_extract')
        self.assertEqual(call['body'], {
            'contents': [{'role': 'user', 'parts': [{'text': 'first\nsecond'}]}],
            'systemInstruction': {'parts': [{'text': ADDITIVE_EXTRACTION_PROMPT}]},
            'generationConfig': {'maxOutputTokens': 16384,
                                 'thinkingConfig': {'thinkingBudget': 0},
                                 'responseMimeType': 'application/json'},
        })

    def test_writer_rejects_invalid_native_memory_json(self):
        for value in ({}, {'memory': 'bad'}, {'memory': [None]},
                      {'memory': [{'text': ''}]}, {'memory': [{'text': 4}]}):
            with self.subTest(value=value):
                api = CaptureAPI([response(json.dumps(value))])
                with self.assertRaises(RuntimeError):
                    provider.Gemini25ExtractionLLM(api).generate_response([
                        {'role': 'system', 'content': ADDITIVE_EXTRACTION_PROMPT},
                        {'role': 'user', 'content': 'source'},
                    ])

    def test_reader_plain_text_preserves_exact_answer_and_never_sends_gold(self):
        for raw in ('  Paris\n', '- Paris\n- Rome', '{"answer": ["Paris", "Rome"]}'):
            with self.subTest(raw=raw):
                api = CaptureAPI([response(raw)])
                result = self.read(api)
                self.assertEqual(result['prediction']['answer'], raw)
                self.assertEqual(result['prediction']['raw'], raw)
                call = api.calls[0]
                self.assertEqual(call['stage'], provider.READER_STAGE)
                self.assertEqual(call['body']['generationConfig'], {
                    'maxOutputTokens': 8192, 'thinkingConfig': {'thinkingBudget': 0}})
                self.assertNotIn('PRIVATE_GOLD', json.dumps(call['body']))
                self.assertEqual(json.loads(call['body']['contents'][0]['parts'][0]['text']),
                                 frozen_evaluation.reader_input('Where did Ada visit?', FACTS,
                                     reference_date='2026-10-07'))
                self.assertEqual(result['reader_model'], provider.MODEL)

    def test_judge_uses_full_gold_and_same_strict_or_terminal_parser(self):
        gold = 'Paris; Rome; 完整答案'
        for raw in ('[[yes]]', '  [[no]]\n', 'Explanation.\n[[yes]]'):
            with self.subTest(raw=raw):
                api = CaptureAPI([response(raw)])
                result = evaluation.judge_answer(api, 'Q', gold, 'A')
                self.assertEqual(result, frozen_evaluation.parse_judge_label(raw))
                body = api.calls[0]['body']
                self.assertEqual(body, provider.generation_body(
                    frozen_evaluation.make_judge_prompt('Q', gold, 'A')))
                self.assertNotIn('systemInstruction', body)
                self.assertIn(gold, body['contents'][0]['parts'][0]['text'])
                self.assertEqual(api.calls[0]['stage'], provider.JUDGE_STAGE)
        for raw in ('YES', '[[yes]]\n[[no]]', '', {'correct': True}):
            with self.subTest(raw=raw), self.assertRaises(evaluation.InvalidJudgeLabel):
                evaluation.parse_judge_label(raw)

    def test_combined_evaluation_uses_new_models_and_stage_bindings(self):
        api = CaptureAPI([response('Paris'), response('[[yes]]')])
        result = evaluation.answer_and_judge(api, 'Q', 'Paris', FACTS, reference_date='2026-10-07')
        self.assertEqual([call['model'] for call in api.calls], [provider.MODEL] * 2)
        self.assertEqual([call['stage'] for call in api.calls],
                         [provider.READER_STAGE, provider.JUDGE_STAGE])
        self.assertEqual(result['evaluation_protocol'], evaluation.PROTOCOL)
        self.assertEqual(result['judge_model'], provider.MODEL)

    def test_request_guard_rejects_other_models_and_configuration_drift(self):
        body = provider.generation_body('Q', system=provider.READER_SYSTEM)
        for model in ('gemini-3.8-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite',
                      'gemini-2.5-flash-preview-05-20', 'models/gemini-2.5-flash'):
            with self.subTest(model=model), self.assertRaises(ValueError):
                provider.check_request('gemini', model, body, 'generateContent', provider.READER_STAGE)
        for replacement in ({'temperature': 0}, {'thinkingConfig': {'thinkingLevel': 'low'}},
                            {'thinkingConfig': {'thinkingBudget': 1024}},
                            {'maxOutputTokens': 2048}, {'responseMimeType': 'application/json'}):
            changed = deepcopy(body)
            changed['generationConfig'].update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                provider.check_request('gemini', provider.MODEL, changed, 'generateContent', provider.READER_STAGE)
        for stage in ('qa_reader', frozen_provider.READER_STAGE, 'unspecified'):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                provider.check_request('gemini', provider.MODEL, body, 'generateContent', stage)

    def test_response_model_guard_and_resource_spelling_do_not_modify_raw(self):
        for model in (provider.MODEL, 'models/' + provider.MODEL, None):
            value = response('  retained\n', model=model)
            before = deepcopy(value)
            self.assertEqual(provider.response_text(value), '  retained\n')
            self.assertEqual(value, before)
        for model in ('gemini-3.8-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite',
                      'gemini-2.5-flash-preview-05-20', 'gemini-2.5-flash-001',
                      'models/gemini-2.5-flash-lite', 'gemini-2.5-flash-extra', ''):
            with self.subTest(model=model), self.assertRaises(RuntimeError):
                provider.response_text(response(model=model))

    def test_response_filters_thoughts_preserves_text_and_rejects_incomplete_or_tools(self):
        self.assertEqual(provider.response_text(response(parts=[
            {'text': 'private thought', 'thought': True}, {'text': '  Paris'}, {'text': '\n'},
        ])), '  Paris\n')
        for value in (response(finish='MAX_TOKENS'), response(text='  '),
                      response(parts=[{'text': 'thought only', 'thought': True}]),
                      response(parts=[{'functionCall': {'name': 'x'}}, {'text': 'Paris'}]),
                      {'candidates': []}, {'candidates': [response()['candidates'][0]] * 2}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                provider.response_text(value)

    def test_cache_roots_and_retry_policy_fail_before_credentials_or_dispatch(self):
        with self.assertRaises(ValueError):
            provider.Gemini25API(self.root / 'invalid', cache_roots=[self.root / 'old'])
        with self.assertRaises(ValueError):
            provider.Gemini25API(self.root / 'invalid', retries=0)
        api, _ = self.api([])
        scoped = api.scoped(phase='reader')
        self.assertIsInstance(scoped, provider.Gemini25API)
        self.assertEqual(scoped.cache_roots, ())
        self.assertEqual(scoped.retries, 2)
        self.assertEqual(scoped.gate_path, api.gate_path)
        self.assertIs(provider.Gemini25API.scoped, PacedEmbeddingAPI.scoped)
        self.assertIs(provider.Gemini25API.embed, ModelAPI.embed)
        self.assertIs(provider.Gemini25API.score_many, ModelAPI.score_many)

    def test_success_cache_keeps_exact_raw_and_reuses_only_current_run(self):
        raw = response('  Paris\n', model='models/' + provider.MODEL)
        api, session = self.api([raw])
        first = self.read(api)
        second = self.read(api)
        self.assertEqual(first, second)
        self.assertEqual(len(session.calls), 1)
        cache = list(api.cache_dir.glob('*.json'))
        self.assertEqual(len(cache), 1)
        self.assertEqual(json.loads(cache[0].read_text(encoding='utf-8'))['response'], raw)
        self.assertEqual(len(list((api.run_dir / 'request_intents').glob('*.json'))), 1)
        self.assertEqual([event['source'] for event in api.events], ['new_request', 'existing_run_cache'])
        self.assertFalse((api.run_dir / 'cache_lineage').exists())

    def test_transport_has_two_retries_and_unresolved_intent_cannot_redispatch(self):
        api, session = self.api([requests.ConnectionError('offline failure')] * 3)
        with patch('jev_mem.providers.time.sleep') as sleep:
            with self.assertRaises(RuntimeError):
                self.read(api)
            self.assertEqual(len(session.calls), 3)
            self.assertEqual(sleep.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, 'no implicit redispatch'):
                self.read(api)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(list((api.run_dir / 'request_intents').glob('*.json'))), 1)
        self.assertEqual(list(api.cache_dir.glob('*.json')), [])

    def test_embedding_and_jev_guards_keep_frozen_scope(self):
        provider.check_request('gemini', 'gemini-embedding-001', {}, 'batchEmbedContents', 'embedding')
        provider.check_request('jev', 'systemone', {'opaque': 'unchanged'}, 'generateContent', 'tag')
        with self.assertRaises(ValueError):
            provider.check_request('gemini', provider.MODEL, {}, 'countTokens', 'token-count')


if __name__ == '__main__':
    unittest.main()
