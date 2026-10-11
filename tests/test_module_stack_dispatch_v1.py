"""Independent zero-network tests of module-stack dispatch and source seams."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev_mem import source_support_stack_v1 as core
from tools import prepare_module_stack_v1 as prepare
from tools import score_module_stack_v1 as score
from tools import run_module_stack_qa_v1 as qa
from tests.test_evidence_exchange_qa_r01 import job


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def signature():
    return {'provider': 'jev', 'model': core.MODEL, 'operation': 'systemone', 'body': {
        'model': core.MODEL, 'state': {'original_memory': {'text': 'A memory', 'observation_date': None}},
        'questions': {'s00': {'type': 'noul', 'instructions': core.SUPPORT_PREDICATE + '\nWindow: A source'}}}}


def response(value=.75):
    return {'model': core.MODEL, 'answers': {'s00': {'type': 'noul', 'noul': value}}}


class Fixture:
    def __init__(self, root, statuses=(200,), *, raw_response=None):
        self.run = Path(root).resolve()
        self.phase = 'support'
        self.producer = self.run / 'phases' / self.phase / 'firstpass'
        self.producer.mkdir(parents=True)
        self.sig = signature()
        self.req = prepare.request_value(self.sig, 'c0_fact_f0', 'c0_fact_f0', 0, 'support', 'protocol')
        self.ref = self.request(self.req)
        self.key = self.req['request_hash']
        self.raw = self.producer / 'api_cache' / (self.key + '.json')
        self.intent = self.producer / 'request_intents' / (self.key + '.json')
        dump(self.intent, {'request_hash': self.key, 'signature': self.sig,
            'context': score.context(self.req), 'stage': score.STAGE, 'retries': 2})
        self.events = [{'request_hash': self.key, 'context': score.context(self.req),
            'stage': score.STAGE, 'provider': 'jev', 'model': core.MODEL, 'operation': 'systemone',
            'cache_hit': False, 'attempt': i + 1, 'status': value, 'timestamp_utc': '2026-10-05T00:00:00Z'}
            for i, value in enumerate(statuses)]
        self.write_events()
        if statuses[-1] == 200 and raw_response is not False:
            dump(self.raw, {'request': self.sig, 'response': response() if raw_response is None else raw_response})

    def request(self, rec):
        path = self.run / 'phases' / self.phase / 'requests' / (rec['request_id'] + '.json')
        score.publish(path, rec)
        return prepare.Reader().pointer(path)

    def write_events(self):
        path = self.producer / 'api_calls.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in self.events), encoding='utf-8')

    def consumer(self):
        rec = {**self.req, 'request_id': 'c0_fact_f1', 'query_id': 'c0_fact_f1'}
        return rec, self.request(rec)

    def execute(self, rec=None, ref=None):
        rec, ref = (self.req, self.ref) if rec is None else (rec, ref)
        # A nonempty synthetic key map bypasses credential loading. Existing raw
        # or exhausted intent is consumed through the real cache implementation.
        # The hard network mock makes accidental physical dispatch fail the test.
        with patch('requests.Session.post', side_effect=AssertionError('Network forbidden')):
            api = score.GateAPI(self.producer, keys={'fixture': 'synthetic_not_a_secret'},
                                gate_path=self.run / 'mock_gate.json', retries=2)
            row = score.one(self.producer, ref, api, 'protocol')
        result_ref = prepare.Reader().pointer(self.producer / 'responses' / (rec['request_id'] + '.json'))
        return row, result_ref

    def terminal(self, refs, rows):
        manifest = self.run / 'phases' / self.phase / 'manifest.json'
        score.publish(manifest, {'schema': 'module-stack-phase-manifest-v1', 'phase': self.phase,
            'protocol_sha256': 'protocol', 'requests': refs, 'units': []})
        score.publish(self.producer / 'execution_intent.json', {'schema': score.SCHEMA,
            'phase': self.phase, 'recovery': False, 'requests': refs, 'transport_retries': 2,
            'dispatch_code_sha256': score.sha(score.__file__), 'manifest_sha256': score.sha(manifest)})
        score.publish(self.producer / 'outcome.json', {'schema': score.SCHEMA, 'status': 'terminal',
            'expected_requests': len(refs), 'terminal_requests': len(rows), 'local_errors': []})


class DispatchTests(unittest.TestCase):
    def test_real_exact_cache_producer_and_consumer_raw_replay(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            row, ref = f.execute()
            self.assertEqual(row['status'], 'complete')
            other, other_ref = f.consumer()
            cached, cached_ref = f.execute(other, other_ref)
            self.assertEqual(cached['scores'], row['scores'])
            self.assertEqual(cached['raw'], row['raw'])
            self.assertEqual(score.verify_result(prepare.Reader(), cached_ref, other_ref, f.producer, 2)['status'], 'complete')
            f.terminal([f.ref, other_ref], [row, cached])
            self.assertEqual(len(prepare.phase_results(prepare.Reader(), f.run, f.phase)), 2)

    def test_exhausted_transport_consumer_without_ledger_event_is_known_failure(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d, ('transport_error',) * 3)
            row, ref = f.execute()
            other, other_ref = f.consumer()
            cached, cached_ref = f.execute(other, other_ref)
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(cached['status'], 'failed')
            self.assertIsNone(cached['raw'])
            self.assertEqual(cached['failure_evidence']['producer_context'], score.context(f.req))
            self.assertEqual(cached['failure_evidence']['consumer_context'], score.context(other))
            self.assertFalse(any(e['context'] == score.context(other) for e in f.events))
            self.assertEqual(score.verify_result(prepare.Reader(), cached_ref, other_ref, f.producer, 2)['status'], 'failed')
            f.terminal([f.ref, other_ref], [row, cached])
            self.assertEqual(len(prepare.phase_results(prepare.Reader(), f.run, f.phase, allow_recovery=False)), 2)
            with self.assertRaisesRegex(ValueError, 'recovery'):
                prepare.phase_results(prepare.Reader(), f.run, f.phase)

    def test_partial_transport_and_http200_without_raw_stay_unknown(self):
        for statuses in [('transport_error',), (200,)]:
            with self.subTest(statuses=statuses), tempfile.TemporaryDirectory() as d:
                f = Fixture(d, statuses, raw_response=False)
                row, ref = f.execute()
                self.assertEqual(row['status'], 'unknown')
                with self.assertRaises(ValueError):
                    score.verify_result(prepare.Reader(), ref, f.ref, f.producer, 2)

    def test_failed_raw_replays_parser_and_changed_signature_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d, raw_response=response(True))
            row, ref = f.execute()
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(score.verify_result(prepare.Reader(), ref, f.ref, f.producer, 2)['status'], 'failed')
            changed = copy.deepcopy(f.sig)
            changed['body']['state']['original_memory']['text'] += ' changed'
            dump(f.raw, {'request': changed, 'response': response(True)})
            with self.assertRaises(ValueError):
                score.verify_result(prepare.Reader(), ref, f.ref, f.producer, 2)

    def test_complete_result_requires_its_consumer_event(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            other, other_ref = f.consumer()
            cached, cached_ref = f.execute(other, other_ref)
            f.write_events()  # Retain physical producer only; remove consumer.
            with self.assertRaisesRegex(ValueError, 'consumer event'):
                score.verify_result(prepare.Reader(), cached_ref, other_ref, f.producer, 2)

    def test_missing_result_and_extra_unmanifested_response_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            row, ref = f.execute()
            f.terminal([f.ref], [row])
            path = Path(ref['path'])
            saved = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                prepare.phase_results(prepare.Reader(), f.run, f.phase)
            path.write_bytes(saved)
            score.publish(f.producer / 'responses' / 'unexpected.json', row)
            with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                prepare.phase_results(prepare.Reader(), f.run, f.phase)

    def test_recovery_uses_original_retries_and_failed_subset_once(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            other, other_ref = f.consumer()
            manifest = f.run / 'phases' / f.phase / 'manifest.json'
            score.publish(manifest, {'protocol_sha256': 'protocol'})
            constructed = []
            def fake_api(producer, *, retries):
                constructed.append(retries)
                return object()
            def fake_one(producer, ref, api, protocol):
                rec = score.read(ref['path'])
                row = {'status': 'complete', 'conversation': rec['conversation'], 'request_id': rec['request_id']}
                score.publish(producer / 'responses' / (rec['request_id'] + '.json'), row)
                return row
            first = {f.req['request_id']: {'status': 'complete'}, other['request_id']: {'status': 'failed'}}
            with patch.object(prepare, 'validate_phase', return_value=({'protocol_sha256': 'protocol'}, [f.ref, other_ref])), \
                 patch.object(prepare, 'phase_results', return_value=first), \
                 patch.object(score, 'GateAPI', side_effect=fake_api), patch.object(score, 'one', side_effect=fake_one):
                out = score.run_phase(f.run, f.phase, recovery=True)
                self.assertEqual(out['status'], 'terminal')
                self.assertEqual(out['expected_requests'], 1)
                self.assertEqual(constructed, [2])
                intent = score.read(f.run / 'phases' / f.phase / 'recovery_v1' / 'execution_intent.json')
                self.assertEqual(intent['requests'], [other_ref])
                with self.assertRaises(FileExistsError):
                    score.run_phase(f.run, f.phase, recovery=True)

    def test_unmanifested_unknown_intent_must_block_whole_phase(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            row, ref = f.execute()
            f.terminal([f.ref], [row])
            foreign = copy.deepcopy(f.sig)
            foreign['body']['state']['original_memory']['text'] = 'Unmanifested interrupted request'
            key = score.request_hash(foreign)
            dump(f.producer / 'request_intents' / (key + '.json'),
                 {'request_hash': key, 'signature': foreign, 'context': score.context(f.req),
                  'stage': score.STAGE, 'retries': 2})
            with self.assertRaises(ValueError):
                prepare.phase_results(prepare.Reader(), f.run, f.phase)

    def test_ledger_hash_and_context_must_be_admitted_as_a_pair(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            other_sig = copy.deepcopy(f.sig)
            other_sig['body']['state']['original_memory']['text'] = 'Different declared request'
            other = prepare.request_value(other_sig, 'c0_fact_f2', 'c0_fact_f2', 0, 'support', 'protocol')
            other_ref = f.request(other)
            other_key = other['request_hash']
            dump(f.producer / 'request_intents' / (other_key + '.json'),
                 {'request_hash': other_key, 'signature': other_sig, 'context': score.context(other),
                  'stage': score.STAGE, 'retries': 2})
            dump(f.producer / 'api_cache' / (other_key + '.json'), {'request': other_sig, 'response': response()})
            f.events.append({**f.events[0], 'request_hash': other_key, 'context': score.context(other)})
            f.write_events()
            row, ref = f.execute()
            other_row, result_ref = f.execute(other, other_ref)
            f.terminal([f.ref, other_ref], [row, other_row])
            # Every individual hash and context is declared, but this extra
            # physical combination is not an admitted producer or consumer.
            with (f.producer / 'api_calls.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({**f.events[0], 'context': score.context(other)}) + '\n')
            with self.assertRaises(ValueError):
                prepare.phase_results(prepare.Reader(), f.run, f.phase)


class SourceJobTests(unittest.TestCase):
    def test_mature_job_keeps_changed_pool_ids_and_all_nine_shapes(self):
        oldjob = {**job(), 'protocol_sha256': 'original-protocol'}
        new = copy.deepcopy(oldjob['selected_facts'])
        new[-1] = {'id': 'replacement', 'text': 'Another frozen memory', 'observation_date': '2023-01-01'}
        imported = {arm: {oldjob['query_id']: {'selected_facts': new}} for arm in prepare.ARMS[1:5]}
        base = ({'protocol_sha256': 'protocol', 'protocol': {'frozen_source': {'path': 'source', 'sha256': 'h'}}},
                {'original_jobs': [oldjob], 'imported': {'arms': imported}}, {})
        unit = {'unit_id': oldjob['query_id'], 'locators': {arm: {'candidates': []} for arm in prepare.SOURCE_ARMS},
                'request_ids': {arm: None for arm in prepare.SOURCE_ARMS}}
        reader = SimpleNamespace(pointer=lambda p: {'path': str(p), 'sha256': 'h'})
        with patch.object(prepare, 'base', return_value=base), patch.object(prepare, 'phase_results', return_value={}), \
             patch.object(prepare, 'units', return_value=[({}, unit)]):
            rows = list(prepare.expected_selections(reader, Path('mock-run')))
        self.assertEqual([arm for _, arm, _ in rows], list(prepare.ARMS))
        jobs = [prepare.old._job(old, arm, rec, {'path': 'selection', 'sha256': 'h'}, 'protocol') for old, arm, rec in rows]
        qa.check_jobs(jobs)
        self.assertEqual(jobs[0]['selected_ids'], oldjob['selected_ids'])
        for j in jobs[1:]:
            self.assertEqual(j['selected_ids'][-1], 'replacement')
            self.assertEqual(j['selected_facts'], new)


if __name__ == '__main__': unittest.main()
