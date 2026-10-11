"""Offline contracts for R11 selection and dispatch; no provider requests."""
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from jev_mem import source_set_r11 as pure
from jev_mem import source_append_r05 as old
from tools import score_source_set_r11 as score


def facts():
    return [{'id': str(i), 'text': f'原始 memory {i}', 'observation_date': None} for i in range(20)]


def candidates():
    out = []
    for i in range(4):
        view = {'session_id': 'session_1', 'source_observation_date': '2023-01-01',
            'turns': [{'source_turn_id': f'D1:{i}', 'speaker_text': f'Sam: useful 中文 detail {i}',
                'machine_captions': ['A machine caption']}]}
        rendered = old.attachment(view)
        out.append({'packet_id': old.digest(view), 'packet_view': view, 'attachment': rendered,
            'attachment_utf8_bytes': len(rendered.encode('utf-8')),
            'origin': {'anchor_slot': i * 3, 'anchor_id': str(i * 3)}})
    return out


def protocol():
    return {'g': {'predicate': 'Full memory conditional usefulness.'},
        'p': {'instructions': 'Select using only each option and original memories.',
            'criteria_template': 'The pair {first} and {second} is best.',
            'presentation_seed': pure.POLICY['presentation_seed']}}


def signature(arm):
    args = ('What happened?', '2023-01-02', facts(), candidates(), protocol())
    return pure.build_g_request(*args) if arm == 'g' else pure.build_p_request('c0_q0', *args)


def g_response(values=None):
    values = values or [0.1, 0.2, 0.3, 0.4]
    return {'model': pure.MODEL, 'answers': {f'f{i:02d}': {'type': 'noul', 'noul': v}
        for i, v in enumerate(values)}}


def p_response(probs=None, choice='p00', confidence=0):
    probs = probs or {'p00': 0.5, **{k: 0.1 for k in pure.PAIR_KEYS[1:]}}
    return {'model': pure.MODEL, 'answers': {'pair_choice': {'type': 'choice',
        'probabilities': probs, 'choice': choice, 'confidence': confidence}}}


class CoreTests(unittest.TestCase):
    def test_g_sees_complete20_but_not_other_windows_or_forbidden_fields(self):
        sig = signature('g')
        pure.validate_signature('g', sig)
        state = sig['body']['state']
        self.assertEqual(len(state['memories']), 20)
        self.assertEqual(set(state), {'query', 'memories'})
        self.assertNotIn('id', state['memories'][0])
        for i, q in enumerate(sig['body']['questions'].values()):
            literal = json.loads(q['instructions'].split('\nDisplayed candidate source attachment (JSON):\n', 1)[1])
            self.assertEqual(literal, {'candidate_index': i, 'anchor_slot': i * 3,
                'source_attachment': candidates()[i]['attachment']})
        for field in ('reference', 'category', 'old_scores', 'reader_answer', 'judge_label'):
            other = deepcopy(sig); other['body']['state'][field] = 'forbidden'
            with self.assertRaises(ValueError): pure.validate_signature('g', other)

    def test_pair_mapping_complete_deterministic_and_criteria_matches(self):
        sig = signature('p'); pure.validate_signature('p', sig)
        options = sig['body']['state']['bundle_options']
        self.assertEqual(options, pure.pair_options('c0_q0'))
        self.assertEqual({tuple(v) for v in options.values()}, set(pure.PAIRS))
        for key, (a, b) in options.items():
            self.assertEqual(sig['body']['questions']['pair_choice']['criteria'][key], f'The pair {a} and {b} is best.')
        bad = deepcopy(sig); bad['body']['state']['bundle_options']['p01'] = options['p00']
        with self.assertRaises(ValueError): pure.validate_signature('p', bad)

    def test_full_set_request_differs_from_old_local_request(self):
        local = old.build_request('What happened?', '2023-01-02', facts(), candidates(),
            {'jev': {'predicate': 'Full memory conditional usefulness.'}})
        self.assertNotEqual(pure.request_hash(local), pure.request_hash(signature('g')))

    def test_g_all_low_ties_still_append_two_keep_exact_prefix_and_dates(self):
        fs, cs = facts(), candidates(); snapshot = deepcopy(fs)
        scores = pure.parse_g(g_response([0, 0, 0, 0]), signature('g'))
        d = pure.choose_g(fs, cs, scores)
        self.assertEqual(d['candidate_indices'], [0, 1])
        selected = pure.selected_facts(fs, cs, d)
        self.assertEqual(fs, snapshot)
        self.assertEqual([x['id'] for x in selected], [x['id'] for x in fs])
        for actual, original in zip(selected, fs):
            self.assertTrue(actual['text'].startswith(original['text']))
            self.assertEqual(actual['observation_date'], original['observation_date'])
        self.assertEqual(selected[0]['text'], fs[0]['text'] + cs[0]['attachment'])
        self.assertEqual(selected[3]['text'], fs[3]['text'] + cs[1]['attachment'])

    def test_g_invalid_never_zero_imputed(self):
        for v in (True, math.nan, math.inf, -0.1, 1.1, None):
            resp = g_response(); resp['answers']['f00']['noul'] = v
            with self.assertRaises(ValueError): pure.parse_g(resp, signature('g'))
        resp = g_response(); del resp['answers']['f03']
        with self.assertRaises(ValueError): pure.parse_g(resp, signature('g'))

    def test_p_low_confidence_unique_max_selects_exact_bound_pair(self):
        sig = signature('p'); parsed = pure.parse_p(p_response(), sig)
        d = pure.choose_p(facts(), candidates(), parsed, signature=sig)
        self.assertEqual(d['candidate_indices'], sig['body']['state']['bundle_options']['p00'])
        self.assertEqual(len(d['packet_ids']), 2)
        other = deepcopy(parsed); other['candidate_indices'] = [0, 1] if parsed['candidate_indices'] != [0, 1] else [0, 2]
        with self.assertRaises(ValueError): pure.choose_p(facts(), candidates(), other, signature=sig)
        wrong = deepcopy(sig); wrong['body']['state']['memories'][0]['text'] = 'Changed evidence'
        with self.assertRaises(ValueError): pure.choose_p(facts(), candidates(), parsed, signature=wrong)

    def test_p_maximum_tie_baseline_and_epsilon_boundary(self):
        sig = signature('p')
        probs = {'p00': 0.3, 'p01': 0.3, **{k: 0.1 for k in pure.PAIR_KEYS[2:]}}
        parsed = pure.parse_p(p_response(probs), sig)
        d = pure.choose_p(facts(), candidates(), parsed, signature=sig)
        self.assertEqual(d['reason'], 'probability_maximum_tie_fallback')
        self.assertEqual(pure.selected_facts(facts(), candidates(), d), facts())
        probs['p00'] += 4e-13; probs['p01'] -= 4e-13
        self.assertTrue(pure.parse_p(p_response(probs), sig)['tied_max'])
        probs['p00'] += 2e-12; probs['p01'] -= 2e-12
        self.assertFalse(pure.parse_p(p_response(probs), sig)['tied_max'])

    def test_p_invalid_schema_probability_sum_argmax_and_confidence(self):
        changes = [lambda r: r.update(model='wrong'),
            lambda r: r['answers']['pair_choice'].update(type='noul'),
            lambda r: r['answers']['pair_choice']['probabilities'].pop('p05'),
            lambda r: r['answers']['pair_choice']['probabilities'].update(p00=True),
            lambda r: r['answers']['pair_choice']['probabilities'].update(p00=0.6),
            lambda r: r['answers']['pair_choice'].update(choice='p01'),
            lambda r: r['answers']['pair_choice'].update(confidence=math.nan)]
        for change in changes:
            resp = p_response(); change(resp)
            with self.assertRaises(ValueError): pure.parse_p(resp, signature('p'))

    def test_failure_fallback_requires_explicit_no_scores(self):
        for status in ('failed', 'unknown', 'not_dispatched'):
            for chooser in (pure.choose_g, pure.choose_p):
                d = chooser(facts(), candidates(), None, score_status=status)
                self.assertEqual(pure.selected_facts(facts(), candidates(), d), facts())
                with self.assertRaises(ValueError): chooser(facts(), candidates(), {}, score_status=status)
        with self.assertRaises(ValueError): pure.validate_frozen_input(facts(), candidates()[:3])


class FakeAPI:
    def __init__(self, run, response=None, raise_unknown=False):
        self.run_dir = Path(run); self.cache_dir = self.run_dir / 'api_cache'
        self.cache_dir.mkdir(exist_ok=True); self.context = {}; self.calls = []
        self.response = response; self.raise_unknown = raise_unknown

    def scoped(self, **context):
        self.context = context
        return self

    def request(self, provider, model, body, *, operation, stage):
        self.calls.append(stage)
        if self.raise_unknown: raise RuntimeError('Unknown outcome, no intent')
        sig = {'provider': provider, 'model': model, 'operation': operation, 'body': body}
        key = pure.request_hash(sig)
        folder = self.run_dir / 'request_intents'; folder.mkdir(exist_ok=True)
        (folder / (key + '.json')).write_text(json.dumps({'request_hash': key, 'signature': sig,
            'stage': stage, 'retries': 2, 'context': self.context}), encoding='utf-8')
        (self.cache_dir / (key + '.json')).write_text(json.dumps({'request': sig, 'response': self.response}), encoding='utf-8')
        with (self.run_dir / 'api_calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'request_hash': key, 'stage': stage, 'context': self.context,
                'provider': provider, 'model': model, 'operation': operation, 'attempt': 1,
                'status': 200, 'cache_hit': False, 'timestamp_utc': '2026-10-05T00:00:00Z'}) + '\n')
        return self.response


def request_file(run, arm, sig):
    folder = Path(run) / 'requests' / arm; folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'c0_q0.json'
    size = len(pure.serialized_signature(sig))
    path.write_text(json.dumps({'arm': arm, 'query_id': 'c0_q0', 'conversation': 0,
        'protocol_sha256': 'test_protocol', 'signature': sig, 'request_hash': pure.request_hash(sig),
        'signature_utf8_bytes': size, 'dispatch_eligible': size <= score.MAX_BYTES,
        'skip_reason': None if size <= score.MAX_BYTES else 'signature_oversize'}), encoding='utf-8')
    return path


class DispatcherTests(unittest.TestCase):
    def test_exact_mock_dispatch_and_raw_binding(self):
        for arm, resp in [('g', g_response()), ('p', p_response())]:
            with TemporaryDirectory() as directory:
                api = FakeAPI(directory, resp)
                path = request_file(directory, arm, signature(arm))
                row = score.one(directory, path, api)
                self.assertEqual(row['status'], 'complete')
                self.assertEqual(api.calls, [score.STAGES[arm]])
                self.assertTrue(Path(row['raw']['path']).is_file())
                with self.assertRaises(FileExistsError): score.one(directory, path, api)

    def test_oversize_never_dispatches_or_clips(self):
        with TemporaryDirectory() as directory:
            sig = signature('g'); sig['body']['questions']['f00']['instructions'] += '字' * 50000
            api = FakeAPI(directory)
            row = score.one(directory, request_file(directory, 'g', sig), api)
            self.assertEqual(row['status'], 'not_dispatched'); self.assertEqual(api.calls, [])

    def test_malformed_http200_known_failure_unknown_outcome_stays_unknown(self):
        for resp, unknown, status in [({'model': 'wrong'}, False, 'failed'), (None, True, 'unknown')]:
            with TemporaryDirectory() as directory:
                api = FakeAPI(directory, resp, unknown)
                row = score.one(directory, request_file(directory, 'p', signature('p')), api)
                self.assertEqual(row['status'], status)
                self.assertIsNone(row['scores'])

    def test_gate_blocks_writer_reader_wrongstage_and_schema(self):
        api = object.__new__(score.GateAPI)
        with patch.object(score.Gemini38API, 'request', return_value={'offline': True}) as mock:
            for arm in pure.ARMS:
                sig = signature(arm)
                api.request(sig['provider'], sig['model'], sig['body'], operation=sig['operation'], stage=score.STAGES[arm])
            self.assertEqual(mock.call_count, 2)
            for provider, model, operation, stage in [('gemini', 'gemini-3.8-flash', 'generateContent', 'reader'),
                ('jev', pure.MODEL, 'systemone', 'source_append_score_r05'),
                ('jev', 'wrong', 'systemone', score.STAGES['g'])]:
                with self.assertRaises(ValueError):
                    api.request(provider, model, signature('g')['body'], operation=operation, stage=stage)
            self.assertEqual(mock.call_count, 2)


class SealEvidenceTests(unittest.TestCase):
    def test_failure_audit_accepts_malformed_raw_rejects_valid_raw_falsely_failed(self):
        from tools import prepare_source_set_r11 as prep
        for response, should_reject in [({'model': 'wrong'}, False), (p_response(), True)]:
            with TemporaryDirectory() as directory:
                run = Path(directory); sig = signature('p'); api = FakeAPI(run, response)
                path = request_file(run, 'p', sig)
                result = score.one(run, path, api)
                req = json.loads(path.read_text(encoding='utf-8'))
                raw_path = run / 'api_cache' / (req['request_hash'] + '.json')
                raw = json.loads(raw_path.read_text(encoding='utf-8'))
                if should_reject:
                    result.update(status='failed', reason='response_validation_failed', error_type='ValueError')
                    _, _, evidence = score.classify_failure(api, sig, raw_path, 'p')
                    result['failure_evidence'] = evidence
                    with self.assertRaises(ValueError): prep.verify_dispatch(prep.Reader(), run, req, result, raw, 'p')
                else:
                    prep.verify_dispatch(prep.Reader(), run, req, result, raw, 'p')

    def test_unknown_cannot_be_forged_into_terminal_failure(self):
        from tools import prepare_source_set_r11 as prep
        with TemporaryDirectory() as directory:
            run = Path(directory); path = request_file(run, 'p', signature('p'))
            result = score.one(run, path, FakeAPI(run, raise_unknown=True))
            req = json.loads(path.read_text(encoding='utf-8'))
            result.update(status='failed', reason='transport_exhausted_three_attempts', error_type='ValueError')
            with self.assertRaises((ValueError, OSError)):
                prep.verify_dispatch(prep.Reader(), run, req, result, None, 'p')

    def test_ineligible_complete_raw_is_not_a_legal_score(self):
        from tools import prepare_source_set_r11 as prep
        with TemporaryDirectory() as directory:
            run = Path(directory); sig = signature('g')
            sig['body']['questions']['f00']['instructions'] += '字' * 50000
            path = request_file(run, 'g', sig)
            result = score.one(run, path, FakeAPI(run))
            req = json.loads(path.read_text(encoding='utf-8'))
            req['content_sha256'] = pure.digest(req)
            path.write_text(json.dumps(req), encoding='utf-8')
            request_ref = {'path': str(path.resolve()), 'sha256': score.sha(path)}
            raw_path = run / 'api_cache' / (req['request_hash'] + '.json')
            raw_path.write_text(json.dumps({'request': sig, 'response': g_response()}), encoding='utf-8')
            result.update(status='complete', reason=None, scores=pure.parse_g(g_response(), sig),
                request_sha256=request_ref['sha256'], raw={'path': str(raw_path.resolve()), 'sha256': score.sha(raw_path)})
            result.pop('content_sha256', None); result['content_sha256'] = pure.digest(result)
            (run / 'responses' / 'g' / 'c0_q0.json').write_text(json.dumps(result), encoding='utf-8')
            item = {'query_id': 'c0_q0', 'conversation': 0, 'requests': {'g': request_ref}}
            with self.assertRaises(ValueError): prep.score(prep.Reader(), run, item, 'g')


if __name__ == '__main__':
    unittest.main()
