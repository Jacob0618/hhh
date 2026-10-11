import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import score_evidence_selection_r03 as d


def signature(n=80):
    return {'provider': 'jev', 'model': 'jev-1.13.0', 'operation': 'systemone', 'body': {
        'model': 'jev-1.13.0', 'state': {'query': {'text': 'Q', 'reference_date': '2023'}},
        'questions': {f'f{i:02d}': {'type': 'noul', 'instructions': 'fact' + str(i)} for i in range(n)}}}


class MockAPI:
    def __init__(self, root, mode='good'):
        self.run_dir = root
        self.cache_dir = root / 'api_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.calls = 0
        self.context = {}

    def scoped(self, **context):
        self.context = context
        return self

    def request(self, provider, model, body, **kwargs):
        self.calls += 1
        sig = dict(provider=provider, model=model, body=body, operation=kwargs['operation'])
        key = d.request_hash(sig)
        d.publish(self.run_dir / 'request_intents' / (key + '.json'), {
            'request_hash': key, 'signature': sig, 'stage': d.JEV_STAGE, 'retries': 2, 'context': self.context})
        if self.mode in ('transport', 'partial', 'wrong_context', 'unknown'):
            rows = []
            if self.mode != 'unknown':
                for attempt in range(1, 3 if self.mode == 'partial' else 4):
                    rows.append(dict(request_hash=key, stage=d.JEV_STAGE, provider=provider, model=model,
                        operation=kwargs['operation'], attempt=attempt, status='transport_error',
                        context={'query_id': 'unrelated'} if self.mode == 'wrong_context' else self.context))
                (self.run_dir / 'api_calls.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            raise RuntimeError('mock terminal or incomplete failure')
        response = {'model': model, 'answers': {k: {'type': 'noul', 'noul': 0.0} for k in body['questions']}}
        if self.mode == 'invalid':
            response['answers']['f07']['noul'] = None
        d.publish(self.cache_dir / (key + '.json'), {'request': sig, 'response': response})
        return response


class ScoreTests(unittest.TestCase):
    def invoke(self, mode, expected, n=80):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sig = signature(n)
            request = root / 'requests/c0_q0.json'
            d.publish(request, dict(signature=sig, request_hash=d.request_hash(sig), query_id='c0_q0', protocol_sha256='mock'))
            api = MockAPI(root, mode)
            result = d.one(root, request, api)
            self.assertEqual(result['status'], expected)
            self.assertEqual(api.calls, 1)
            self.assertIsNone(result['input_token_count'])
            if expected != 'complete':
                self.assertIsNone(result['scores'])
            else:
                self.assertEqual(result['scores'], {f'f{i:02d}': 0.0 for i in range(n)})
            with self.assertRaises(FileExistsError):
                d.one(root, request, api)
            self.assertEqual(api.calls, 1)

    def test_valid_zero_not_missing(self):
        self.invoke('good', 'complete')

    def test_minimum_twenty_all_zero_is_valid(self):
        self.invoke('good', 'complete', n=20)

    def test_variable_pool_size_is_valid(self):
        self.invoke('good', 'complete', n=37)

    def test_wrong_count_or_keys_never_dispatch(self):
        for n in (0, 19, 81):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sig = signature(n)
                request = root / 'requests/c0_q0.json'
                d.publish(request, dict(signature=sig, request_hash=d.request_hash(sig), query_id='c0_q0', protocol_sha256='mock'))
                api = MockAPI(root)
                with self.assertRaises(ValueError): d.one(root, request, api)
                self.assertEqual(api.calls, 0)
        body = signature()['body']
        body['questions']['other'] = body['questions'].pop('f79')
        with self.assertRaises(ValueError): d.expected_keys(body)
        body = signature()['body']; body['questions']['f79']['type'] = 'choice'
        with self.assertRaises(ValueError): d.expected_keys(body)

    def test_partial_score_no_zero_fill(self):
        self.invoke('invalid', 'failed')

    def test_all_attempts_required(self):
        self.invoke('transport', 'failed')

    def test_partial_attempts_unknown(self):
        self.invoke('partial', 'unknown')

    def test_other_context_is_not_recovery(self):
        self.invoke('wrong_context', 'unknown')

    def test_unfinished_intent_unknown(self):
        self.invoke('unknown', 'unknown')

    def test_no_dispatch_bytes_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sig = signature()
            path = root / 'requests/c0_q0.json'
            d.publish(path, dict(signature=sig, request_hash=d.request_hash(sig), query_id='c0_q0', protocol_sha256='mock'))
            api = MockAPI(root)
            with patch.object(d, 'MAX_BYTES', 1):
                result = d.one(root, path, api)
            self.assertEqual(result['status'], 'not_dispatched')
            self.assertEqual(api.calls, 0)
            self.assertIsNone(result['raw'])

    def test_hash_change_rejected_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'requests/c0_q0.json'
            d.publish(path, dict(signature=signature(), request_hash='wrong', query_id='c0_q0', protocol_sha256='mock'))
            api = MockAPI(root)
            with self.assertRaises(ValueError):
                d.one(root, path, api)
            self.assertEqual(api.calls, 0)


if __name__ == '__main__':
    unittest.main()
