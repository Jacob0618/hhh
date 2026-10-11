"""Offline evidence gates, particularly error recovery context and raw vectors."""
import copy
from pathlib import Path
import tempfile
import unittest
from tools import audit_mem0_sparse1540_retrieval as target


class RetrievalAuditTests(unittest.TestCase):
    def setUp(self):
        self.expected = target.expected_requests('Question?', [])
        self.signature = self.expected[0]['signature']
        self.digest = target.driver.request_digest(self.signature)
        self.cached = {'stage': target.STAGE, 'request': self.signature,
                       'response': {'embeddings': [{'values': [1.] + [0.] * 767}]}}

    def event(self, line=1, qid='c1_q1', status=200, cache=False, attempt=1):
        return {'_line': line, 'context': {'architecture': target.driver.POLICY,
            'conversation': 1, 'phase': 'retrieve', 'method': 'regroup_rank', 'query_id': qid},
            'request_hash': self.digest, 'stage': target.STAGE, 'provider': 'gemini',
            'model': target.MODEL, 'operation': 'batchEmbedContents', 'status': status,
            'cache_hit': cache, 'attempt': attempt}

    def audit(self, rows, seeds=None, expected=None):
        return target.audit_consumers(rows, 1, 'c1_q1', expected or self.expected,
            lambda _: (self.cached, {'file_sha256': 'fake'}), seeds)

    def test_missing_consumer_denied_even_with_cache(self):
        self.assertFalse(self.audit([])['admitted'])

    def test_same_context_error_then_http_success_passes(self):
        report = self.audit([self.event(status=503), self.event(2, attempt=2)])
        self.assertTrue(report['admitted'])
        self.assertEqual(report['requests'][0]['recovered_errors'], 1)

    def test_other_query_success_does_not_recover_error(self):
        report = self.audit([self.event(status='transport_error'), self.event(2, qid='c1_q2')])
        self.assertFalse(report['admitted'])
        self.assertIn('unrecovered_same_context_request_error', report['problems'])

    def test_other_query_cache_does_not_recover_error(self):
        report = self.audit([self.event(status='transport_error'), self.event(2, qid='c1_q2', cache=True)], {self.digest})
        self.assertFalse(report['admitted'])

    def test_same_context_valid_seed_cache_can_recover(self):
        self.assertTrue(self.audit([self.event(status=503), self.event(2, cache=True)], {self.digest})['admitted'])

    def test_cache_without_seed_or_prior_producer_denied(self):
        self.assertFalse(self.audit([self.event(cache=True)])['admitted'])

    def test_prior_other_context_is_valid_cache_producer_not_recovery(self):
        self.assertTrue(self.audit([self.event(qid='c1_q2'), self.event(2, cache=True)])['admitted'])

    def test_later_other_context_producer_cannot_justify_cache(self):
        self.assertFalse(self.audit([self.event(cache=True), self.event(2, qid='c1_q2')])['admitted'])

    def test_error_after_success_remains_unrecovered(self):
        self.assertFalse(self.audit([self.event(), self.event(2, status=503)])['admitted'])

    def test_duplicate_query_entity_signature_requires_two_consumers(self):
        expected = target.expected_requests('Question?', [('PROPER', 'Question?')])
        self.assertFalse(self.audit([self.event()], expected=expected)['admitted'])
        self.assertTrue(self.audit([self.event(), self.event(2, cache=True)], expected=expected)['admitted'])

    def test_wrong_request_text_and_nonfinite_vector_denied(self):
        self.cached['request'] = target.signature(['Different'])
        self.assertFalse(self.audit([self.event()])['admitted'])
        self.cached['request'] = self.signature
        self.cached['response']['embeddings'][0]['values'][0] = float('nan')
        self.assertFalse(self.audit([self.event()])['admitted'])

    def test_wrong_phase_or_method_does_not_count(self):
        row = self.event()
        row['context']['phase'] = 'write'
        self.assertFalse(self.audit([row])['admitted'])
        row['context']['phase'] = 'retrieve'
        row['context']['method'] = 'other'
        self.assertFalse(self.audit([row])['admitted'])

    def test_write_error_not_attributed_to_successful_retrieval(self):
        earlier = self.event(status='transport_error')
        earlier['context']['phase'] = 'write'
        earlier['context'].pop('query_id')
        earlier['context']['group_id'] = 'sparse_topic_0001'
        result = self.audit([earlier, self.event(2)])
        self.assertTrue(result['admitted'])
        self.assertEqual(result['requests'][0]['physical_errors'], 0)

    def test_actual_seed_cache_event_without_status_is_valid(self):
        row = self.event(cache=True)
        row.pop('status')
        row.pop('attempt')
        result = self.audit([row], {self.digest})
        self.assertTrue(result['admitted'])
        self.assertEqual(result['requests'][0]['cache_hits'], 1)
        self.assertEqual(result['requests'][0]['physical_success'], 0)

    def test_duplicate_attempts_or_more_than_three_physical_calls_denied(self):
        repeated = [self.event(i, status=503) for i in range(1, 5)] + [self.event(5)]
        result = self.audit(repeated)
        self.assertFalse(result['admitted'])
        self.assertIn('physical_attempt_sequence_invalid', result['problems'])
        self.assertIn('physical_attempt_count_exceeds_uniform_budget', result['problems'])
        self.assertFalse(self.audit([self.event(status=503), self.event(2)])['admitted'])

    def test_cache_true_with_http_error_is_contradictory(self):
        result = self.audit([self.event(status=503, cache=True)], {self.digest})
        self.assertFalse(result['admitted'])
        self.assertIn('contradictory_cache_status', result['problems'])

    def test_nonboolean_cache_flag_rejected(self):
        row = self.event()
        row['cache_hit'] = 'false'
        self.assertFalse(self.audit([row])['admitted'])

    def test_write_success_cannot_recover_query_error(self):
        later = self.event(2)
        later['context']['phase'] = 'write'
        later['context'].pop('query_id')
        self.assertFalse(self.audit([self.event(status=503), later])['admitted'])

    def test_sdk_first8_then_dedup_not_first8_unique(self):
        entries = [('PROPER', 'Same')] * 8 + [('PROPER', 'Must be omitted')]
        got = target.expected_requests('Question?', entries)
        self.assertEqual(len(got[1]['signature']['body']['requests']), 1)

    def test_long_input_token_check_and_dimensions(self):
        got = target.expected_requests('a' * 1801, [])
        self.assertEqual(got[0]['signature']['operation'], 'countTokens')
        self.cached['response']['embeddings'][0]['values'].pop()
        self.assertFalse(self.audit([self.event()])['admitted'])

    def test_native_search_strips_before_both_nlp_and_embedding(self):
        seen = []
        expected = target.native_expected_requests(' \tQuestion?\n', lambda text: seen.append(text) or [])
        self.assertEqual(seen, ['Question?'])
        self.assertEqual(expected, self.expected)
        self.assertTrue(self.audit([self.event()], expected=expected)['admitted'])

    def test_real_c1_q22_normalization_leaves_original_identity_unchanged(self):
        question = next(q for q in target.driver.questions(1) if q['id'] == 'c1_q22')['question']
        before = target.fingerprint(question)
        self.assertNotEqual(question, question.strip())
        observed = []
        expected = target.native_expected_requests(question, lambda text: observed.append(text) or [])
        self.assertEqual(observed, [question.strip()])
        self.assertEqual(expected[0]['signature'], target.signature([question.strip()]))
        self.assertNotEqual(target.driver.request_digest(expected[0]['signature']),
                            target.driver.request_digest(target.signature([question])))
        self.assertEqual(target.fingerprint(question), before)

    def test_stderr_missing_or_entity_warning_denied(self):
        self.assertFalse(target.stderr_audit([])['admitted'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'retrieve.stderr.log'
            path.write_text('Local dependency informational message\n')
            self.assertTrue(target.stderr_audit([path])['admitted'])
            path.write_text('Entity boost search failed for one entity: unknown\n')
            result = target.stderr_audit([path])
            self.assertFalse(result['admitted'])
            self.assertNotIn('unknown', str(result))


if __name__ == '__main__':
    unittest.main()
