from pathlib import Path
import json
import unittest
import tempfile
from unittest.mock import patch
from tools import score_evidence_exchange_r01 as dispatch
from jev_mem.evidence_exchange_r01 import request_hash


def signature():
    return {'provider': 'jev', 'model': 'jev-1.13.0', 'operation': 'systemone',
        'body': {'model': 'jev-1.13.0', 'state': {'query': 'Q', 'facts': []},
                 'questions': {f'p0_{k}': {'type': 'noul', 'instructions': k} for k in ('G', 'L', 'X')}}}


def prepared(tmp_path):
    sig = signature()
    path = tmp_path / 'requests' / 'c0_q0.json'
    dispatch.publish(path, {'signature': sig, 'query_id': 'c0_q0',
        'protocol_sha256': 'test', 'request_hash': request_hash(sig)})
    return path, sig


class MockAPI:
    def __init__(self, root, mode='good'):
        self.run_dir = root
        self.cache_dir = root / 'api_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.calls = 0

    def scoped(self, **context):
        return self

    def request(self, provider, model, body, **kwargs):
        self.calls += 1
        sig = dict(provider=provider, model=model, operation=kwargs['operation'], body=body)
        key = request_hash(sig)
        if self.mode == 'unknown':
            raise OSError('Unfinished request; must not re-dispatch')
        if self.mode == 'transport':
            (self.run_dir / 'api_calls.jsonl').write_text(json.dumps({
                'request_hash': key, 'status': 'transport_error', 'attempt': 3}) + '\n')
            raise RuntimeError('Exhausted')
        values = [.9, .01, .01] if self.mode == 'good' else [.9, None, .01]
        response = {'model': 'jev-1.13.0', 'answers': {k: {'type': 'noul', 'noul': v} for k, v in zip(body['questions'], values)}}
        (self.cache_dir / (key + '.json')).write_text(json.dumps({'request': sig, 'response': response}), encoding='utf-8')
        return response


def test_score_keeps_full_exact_request_and_raw(tmp_path):
    path, sig = prepared(tmp_path)
    api = MockAPI(tmp_path)
    row = dispatch.one(tmp_path, path, api, 'jev')
    assert row['status'] == 'complete'
    assert row['scores'] == {'p0_G': .9, 'p0_L': .01, 'p0_X': .01}
    assert row['raw']['sha256'] == dispatch.sha(row['raw']['path'])
    assert row['input_token_count'] is None
    assert api.calls == 1
    saved = dispatch.read(tmp_path / 'responses/c0_q0.json')
    digest = saved.pop('content_sha256')
    assert dispatch.digest(saved) == digest


def test_unavailable_never_becomes_zero_probability(tmp_path, mode, expected):
    path, _ = prepared(tmp_path)
    api = MockAPI(tmp_path, mode)
    row = dispatch.one(tmp_path, path, api, 'jev')
    assert row['status'] == expected
    assert row['scores'] is None
    assert api.calls == 1
    assert bool(row['raw']) == (mode == 'invalid')


def test_signature_mismatch_no_dispatch(tmp_path):
    path, _ = prepared(tmp_path)
    rec = dispatch.read(path)
    rec['signature']['body']['state']['query'] = 'changed'
    path.write_text(json.dumps(rec))
    api = MockAPI(tmp_path)
    with unittest.TestCase().assertRaises(ValueError):
        dispatch.one(tmp_path, path, api, 'jev')
    assert api.calls == 0


def test_cap_is_bytes_and_keeps_baseline_action_unavailable(tmp_path):
    path, _ = prepared(tmp_path)
    api = MockAPI(tmp_path)
    with patch.object(dispatch, 'MAX_BYTES', 1):
        row = dispatch.one(tmp_path, path, api, 'jev')
    assert row['status'] == 'not_dispatched'
    assert row['scores'] is None and row['raw'] is None and api.calls == 0


def test_pilot_configuration_and_unknown_are_not_probabilities():
    sig = signature()
    body = dispatch.pilot_body(sig)
    assert body['generationConfig'] == {'maxOutputTokens': 8192, 'thinkingConfig': {'thinkingLevel': 'low'}}
    assert 'responseMimeType' not in body['generationConfig']
    labels = dispatch.parse_pilot('p0_G=yes\np0_L=unknown\np0_X=no', sig)
    assert labels['p0_L'] == 'unknown'
    for bad in ['p0_G=yes\np0_L=no', 'p0_G=yes\np0_L=no\np0_X=no\nexplanation', 'p0_G=0.9\np0_L=no\np0_X=no']:
        with unittest.TestCase().assertRaises(ValueError):
            dispatch.parse_pilot(bad, sig)


def test_write_never_overwrites_previous_result(tmp_path):
    path = tmp_path / 'terminal.json'
    dispatch.publish(path, {'status': 'failed'})
    with unittest.TestCase().assertRaises(FileExistsError):
        dispatch.publish(path, {'status': 'complete'})
    assert dispatch.read(path)['status'] == 'failed'


class DispatchTests(unittest.TestCase):
    def isolated(self, function, *args):
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory), *args)

    def test_exact_raw(self):
        self.isolated(test_score_keeps_full_exact_request_and_raw)

    def test_unknown(self):
        self.isolated(test_unavailable_never_becomes_zero_probability, 'unknown', 'unknown')

    def test_transport(self):
        self.isolated(test_unavailable_never_becomes_zero_probability, 'transport', 'failed')

    def test_invalid(self):
        self.isolated(test_unavailable_never_becomes_zero_probability, 'invalid', 'failed')

    def test_signature(self):
        self.isolated(test_signature_mismatch_no_dispatch)

    def test_bytes_cap(self):
        self.isolated(test_cap_is_bytes_and_keeps_baseline_action_unavailable)

    def test_pilot(self):
        test_pilot_configuration_and_unknown_are_not_probabilities()

    def test_no_overwrite(self):
        self.isolated(test_write_never_overwrites_previous_result)


if __name__ == '__main__':
    unittest.main()
