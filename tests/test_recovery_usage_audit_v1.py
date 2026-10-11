"""Offline ledger/transport reconciliation tests; all files are temporary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from jev_mem.recovery_v1 import POLICY, request_hash
from tools.summarize_technical_recovery_v1 import actual_recovery_usage


class RecoveryUsageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='recovery-usage-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seal = 'a' * 64
        self.owner = {'architecture': 'mem0', 'conversation': 0, 'query_id': 'c0_q1', 'method': 'jev_single'}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def setup_attempt(self, *, root=None, started=True, stage='jev_score'):
        root = root or self.root
        self.write(root / 'binding.json', {'seal_sha256': self.seal, 'policy': POLICY, 'denominators': {}})
        if stage == 'jev_score':
            signature = {'provider': 'jev', 'model': 'jev-1.13.0', 'operation': 'systemone',
                         'body': {'state': 'Mock data', 'model': 'jev-1.13.0',
                                  'questions': {'relevant': {'type': 'noul', 'instructions': 'Mock criterion'}}}}
        else:
            signature = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
                         'body': {'contents': [{'role': 'user', 'parts': [{'text': 'Mock prompt'}]}]}}
        h = request_hash(signature)
        folder = root / 'attempts' / h
        proof = {'consumer': copy.deepcopy(self.owner), 'kind': 'same_request_recovery',
                 'basis': 'original_network_failure', 'original_log_lines': [1],
                 'predecessor_request_hashes': [], 'sources': []}
        plan = {'schema': 'jev-mem-recovery-request-v1', 'seal_sha256': self.seal,
                'request_hash': h, 'request': signature, 'stage': stage,
                'proof': proof, 'policy': POLICY, 'max_new_http_attempts': 1}
        self.write(folder / 'planned.json', plan)
        if started:
            self.write(folder / 'attempt_started.json', {'request_hash': h, 'seal_sha256': self.seal,
                       'started_utc': '2026-09-28T00:00:00Z', 'pid': 1, 'proof': proof})
        return {'root': root, 'folder': folder, 'hash': h, 'plan': plan, 'signature': signature}

    def response(self, fixture, *, usage=None, overlay=True, transport=True):
        if usage is None:
            usage = {'input_tokens': 100, 'output_tokens': 2}
        response = {'usage': usage, 'answers': {'relevant': {'noul': .6}}}
        common = {'request': fixture['signature'], 'response': response, 'stage': fixture['plan']['stage']}
        if overlay:
            self.write(fixture['folder'] / 'response.json', {**common, 'request_hash': fixture['hash'],
                       'seal_sha256': self.seal, 'persisted_utc': '2026-09-28T00:00:01Z'})
        if transport:
            self.write(fixture['root'] / 'transport/api_cache' / (fixture['hash'] + '.json'), common)
        return response

    def log(self, fixture, *, status=200, usage=None, cached=False, owner=None):
        if usage is None:
            usage = {'input_tokens': 100, 'output_tokens': 2} if status == 200 else {}
        row = {**{key: fixture['signature'][key] for key in ('provider', 'model', 'operation')},
               'request_hash': fixture['hash'], 'stage': fixture['plan']['stage'],
               'status': status, 'attempt': 1, 'cache_hit': cached, 'usage': usage,
               'context': {'phase': 'technical_recovery_v1', **(owner or self.owner)}}
        path = fixture['root'] / 'transport/api_calls.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        return row

    def outcome(self, fixture, status):
        self.write(fixture['folder'] / 'outcome.json', {'request_hash': fixture['hash'], 'status': status,
                   'finished_utc': '2026-09-28T00:00:02Z'})

    def test_normal_success_and_cache_reuse_do_not_duplicate_fee(self):
        f = self.setup_attempt()
        self.response(f)
        self.log(f)
        self.log(f, cached=True)
        self.outcome(f, 'complete')
        result = actual_recovery_usage(self.root)
        self.assertEqual(result['logged_physical_attempts'], 1)
        self.assertEqual(result['attempt_allowance_slots_consumed'], 1)
        self.assertEqual(result['cache_hits_in_transport_log'], 1)
        self.assertEqual(result['http200'], 1)
        self.assertEqual(result['actual_unknown_cost_calls'], 0)
        self.assertAlmostEqual(result['actual_total_cost_usd'], 100 * .042 / 1e6)

    def test_started_without_log_or_response_keeps_unknown_cost_and_consumed_allowance(self):
        self.setup_attempt()
        result = actual_recovery_usage(self.root)
        self.assertEqual(result['logged_physical_attempts'], 0)
        self.assertEqual(result['attempt_allowance_slots_consumed'], 1)
        self.assertEqual(result['started_but_unlogged_attempts'], 1)
        self.assertEqual(result['started_but_unlogged_without_response'], 1)
        self.assertEqual(result['confirmed_physical_attempts_lower_bound'], 0)
        self.assertEqual(result['possible_physical_attempts_upper_bound'], 1)
        self.assertEqual(result['actual_unknown_cost_calls'], 1)
        self.assertEqual(result['attempts_with_unknown_token_usage'], 1)
        self.assertIsNone(result['actual_total_cost_usd'])
        self.assertIsNone(result['unknown_cost_usd'])
        self.assertIsNone(result['attempt_audit'][0]['recorded_usage'])

    def test_bound_response_without_log_recovers_usage_without_resending(self):
        f = self.setup_attempt()
        self.response(f, overlay=False)
        result = actual_recovery_usage(self.root)
        self.assertEqual(result['response_backed_unlogged_attempts'], 1)
        self.assertEqual(result['confirmed_physical_attempts_lower_bound'], 1)
        self.assertEqual(result['actual_unknown_cost_calls'], 0)
        self.assertEqual(result['logged_physical_attempts'], 0)
        self.assertAlmostEqual(result['actual_known_cost_usd'], 100 * .042 / 1e6)
        self.assertEqual(result['attempt_audit'][0]['billing_source'], 'persisted_response_without_log')

    def test_unlogged_response_without_usage_does_not_become_free(self):
        f = self.setup_attempt()
        self.response(f, usage={})
        result = actual_recovery_usage(self.root)
        self.assertEqual(result['confirmed_physical_attempts_lower_bound'], 1)
        self.assertEqual(result['unlogged_attempts_with_unknown_cost'], 1)
        self.assertFalse(result['actual_cost_complete'])
        self.assertIsNone(result['actual_total_cost_usd'])

    def test_planned_only_consumes_no_allowance_or_usage(self):
        self.setup_attempt(started=False)
        result = actual_recovery_usage(self.root)
        self.assertEqual(result['planned_not_started'], 1)
        self.assertEqual(result['attempt_allowance_slots_consumed'], 0)
        self.assertEqual(result['actual_unknown_cost_calls'], 0)
        self.assertEqual(result['actual_total_cost_usd'], 0.)

    def test_http_failure_transport_error_and_format_failure_are_distinct(self):
        for i, status in enumerate((520, 'transport_error', 200)):
            with self.subTest(status=status):
                root = self.root / str(i)
                f = self.setup_attempt(root=root)
                self.log(f, status=status)
                if status == 200:
                    self.response(f)
                    self.outcome(f, 'failed')
                elif status == 'transport_error':
                    self.outcome(f, 'unknown')
                else:
                    self.outcome(f, 'failed')
                result = actual_recovery_usage(root)
                self.assertEqual(result['http_failure_responses'], int(status == 520))
                self.assertEqual(result['transport_error_attempts'], int(status == 'transport_error'))
                self.assertEqual(result['schema_validation_failed_attempts'], int(status == 200))
                self.assertEqual(result['actual_unknown_cost_calls'], int(status != 200))
                if status == 200:
                    self.assertGreater(result['actual_total_cost_usd'], 0.)

    def test_multiple_physical_logs_for_one_hash_rejected(self):
        f = self.setup_attempt()
        self.log(f)
        self.log(f)
        with self.assertRaisesRegex(ValueError, 'More than one'):
            actual_recovery_usage(self.root)

    def test_response_and_log_require_started_marker(self):
        for i, kind in enumerate(('response', 'log')):
            root = self.root / str(i)
            f = self.setup_attempt(root=root, started=False)
            self.response(f) if kind == 'response' else self.log(f)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                actual_recovery_usage(root)

    def test_each_ledger_hash_and_seal_are_checked(self):
        for i, (name, field) in enumerate((('planned.json', 'request_hash'), ('planned.json', 'seal_sha256'),
                ('attempt_started.json', 'request_hash'), ('attempt_started.json', 'seal_sha256'),
                ('response.json', 'request_hash'), ('response.json', 'seal_sha256'),
                ('outcome.json', 'request_hash'))):
            root = self.root / str(i)
            f = self.setup_attempt(root=root)
            self.response(f)
            self.log(f)
            self.outcome(f, 'complete')
            path = f['folder'] / name
            row = json.loads(path.read_text('utf-8')); row[field] = 'b' * 64; self.write(path, row)
            with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                actual_recovery_usage(root)

    def test_signature_response_usage_and_log_context_mismatches_rejected(self):
        for i, mutation in enumerate(('signature', 'response', 'log_usage', 'owner', 'attempt', 'stage')):
            root = self.root / str(i)
            f = self.setup_attempt(root=root)
            self.response(f)
            self.log(f)
            if mutation == 'signature':
                p = f['folder'] / 'planned.json'; row = json.loads(p.read_text()); row['request']['body']['state'] = 'changed'
            elif mutation == 'response':
                p = f['folder'] / 'response.json'; row = json.loads(p.read_text()); row['response']['answers']['relevant']['noul'] = .3
            else:
                p = root / 'transport/api_calls.jsonl'; row = json.loads(p.read_text())
                if mutation == 'log_usage': row['usage']['input_tokens'] = 101
                elif mutation == 'owner': row['context']['query_id'] = 'c0_q9999'
                elif mutation == 'attempt': row['attempt'] = 2
                else: row['stage'] = 'strict_judge'
            self.write(p, row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                actual_recovery_usage(root)

    def test_planner_and_started_owner_can_be_different_consumers_of_same_hash(self):
        f = self.setup_attempt()
        path = f['folder'] / 'attempt_started.json'
        started = json.loads(path.read_text())
        owner = {**self.owner, 'method': 'conditional'}
        started['proof']['consumer'] = owner
        self.write(path, started)
        self.log(f, owner=owner)
        self.response(f)
        self.assertEqual(actual_recovery_usage(self.root)['logged_physical_attempts'], 1)

    def test_partial_known_usage_preserved_but_total_remains_unknown(self):
        f = self.setup_attempt(stage='qa_reader')
        usage = {'promptTokenCount': 100}
        self.response(f, usage=usage)
        self.log(f, usage=usage)
        result = actual_recovery_usage(self.root)
        self.assertAlmostEqual(result['actual_known_cost_usd'], 100 * .30 / 1e6)
        self.assertEqual(result['actual_unknown_cost_calls'], 1)
        self.assertIsNone(result['actual_total_cost_usd'])

    def test_known_zero_price_output_is_distinct_from_unknown_output_usage(self):
        f = self.setup_attempt()
        self.response(f, usage={'input_tokens': 100})
        self.log(f, usage={'input_tokens': 100})
        result = actual_recovery_usage(self.root)
        # JEV's configured output price is zero, so missing output count need not
        # hide known input cost, but the output token usage is still unknown.
        self.assertEqual(result['actual_unknown_cost_calls'], 0)
        self.assertEqual(result['attempts_with_unknown_token_usage'], 1)
        self.assertTrue(result['attempt_audit'][0]['usage_unknown'])
        self.assertFalse(result['attempt_audit'][0]['cost_unknown'])

    def test_invalid_usage_unknown_status_and_orphan_transport_are_rejected(self):
        for i, mutation in enumerate(('nan_usage', 'status', 'orphan')):
            root = self.root / str(i)
            f = self.setup_attempt(root=root)
            self.log(f)
            p = root / 'transport/api_calls.jsonl'; row = json.loads(p.read_text())
            if mutation == 'nan_usage': row['usage']['input_tokens'] = float('nan')
            elif mutation == 'status': row['status'] = 'made_up'
            else: row['request_hash'] = 'b' * 64
            self.write(p, row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                actual_recovery_usage(root)


if __name__ == '__main__':
    unittest.main()
