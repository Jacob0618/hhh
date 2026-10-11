"""Synthetic ledger reconciliation tests; no experimental QA or API is read."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import unittest

from tools.audit_remaining71_ledger_v2 import audit_qa_ledger, MODELS, METHOD, PHASE


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    consumers, ledger = [], []
    start = datetime(2026, 9, 28, tzinfo=timezone.utc)
    for ci, count in ((1, 21), (7, 50)):
        for q in range(count):
            qid = f'c{ci}_q{q}'
            for stage, model in MODELS.items():
                h = digest(qid + stage)
                usage = {'promptTokenCount': 101, 'candidatesTokenCount': 23}
                consumer = {'conversation': ci, 'query_id': qid, 'method': METHOD,
                            'response_usage': copy.deepcopy(usage), 'resolved_model': model,
                            'event': {'source': 'new_response', 'stage': stage,
                                      'request_hash': h, 'response_sha256': digest(h + 'response'),
                                      'new_physical_attempt': True}}
                row = {'context': {'conversation': ci, 'query_id': qid, 'method': METHOD, 'phase': PHASE},
                       'stage': stage, 'provider': 'gemini', 'model': model,
                       'operation': 'generateContent', 'request_hash': h,
                       'cache_hit': False, 'status': 200, 'attempt': 1,
                       'timestamp_utc': (start + timedelta(seconds=len(ledger))).isoformat(),
                       'usage': usage, 'resolved_model': model}
                consumers.append(consumer)
                ledger.append(row)
    return consumers, ledger


def cached_fixture():
    consumers, ledger = fixture()
    # The second synthetic question reuses the first synthetic reader response.
    source, target = 0, 2
    consumers[target]['event'].update(source='recovery_cache', new_physical_attempt=False,
                                    request_hash=consumers[source]['event']['request_hash'],
                                    response_sha256=consumers[source]['event']['response_sha256'])
    ledger[target].update(cache_hit=True, request_hash=ledger[source]['request_hash'],
                          usage_semantics='reused_response_no_new_model_usage')
    ledger[target].pop('status')
    ledger[target].pop('attempt')
    return consumers, ledger


def transport_fixture():
    consumers, ledger = fixture()
    consumers[0]['event'].update(source='failed_or_unknown_response', response_sha256=None)
    consumers[0].pop('response_usage')
    consumers[0].pop('resolved_model')
    ledger[0].update(status='transport_error', usage_unknown=True)
    for key in ('cache_hit', 'usage', 'resolved_model'):
        ledger[0].pop(key)
    del consumers[1]
    del ledger[1]
    return consumers, ledger


class LedgerTests(unittest.TestCase):
    def test_all142_physical_close(self):
        result = audit_qa_ledger(*fixture())
        self.assertEqual(result['physical_http200'], 142)
        self.assertEqual(result['matched_qa_ledger_rows'], 142)
        self.assertEqual(result['response_usage_bindings_checked'], 142)

    def test_missing_physical_fails_even_when_consumer_event_exists(self):
        consumers, ledger = fixture()
        del ledger[9]
        with self.assertRaisesRegex(ValueError, 'Missing QA ledger'):
            audit_qa_ledger(consumers, ledger)

    def test_wrong_query_context_fails(self):
        consumers, ledger = fixture()
        ledger[0]['context']['query_id'] = 'c1_q9999'
        with self.assertRaises(ValueError):
            audit_qa_ledger(consumers, ledger)

    def test_wrong_phase_or_model_fails(self):
        for change in ('phase', 'model'):
            consumers, ledger = fixture()
            if change == 'phase':
                ledger[0]['context']['phase'] = 'unrelated_phase'
            else:
                ledger[0]['model'] = 'wrong-model'
            with self.assertRaises(ValueError):
                audit_qa_ledger(consumers, ledger)

    def test_extra_qa_ledger_row_fails(self):
        consumers, ledger = fixture()
        ledger.append(copy.deepcopy(ledger[0]))
        with self.assertRaisesRegex(ValueError, 'Extra or repeated'):
            audit_qa_ledger(consumers, ledger)

    def test_extra_qa_with_unexpected_stage_is_not_ignored(self):
        consumers, ledger = fixture()
        extra = copy.deepcopy(ledger[0])
        extra['stage'] = 'unexpected_generation'
        ledger.append(extra)
        with self.assertRaises(ValueError):
            audit_qa_ledger(consumers, ledger)

    def test_unrelated_writer_rows_are_not_qa_fees_or_consumers(self):
        consumers, ledger = fixture()
        ledger.insert(0, {'context': {'conversation': 1, 'phase': 'writer'},
                          'stage': 'remaining71_members_only_extract'})
        result = audit_qa_ledger(consumers, ledger)
        self.assertEqual(result['ignored_non_qa_ledger_rows'], 1)
        self.assertEqual(result['physical_http200'], 142)

    def test_legal_cross_query_cache_reuse(self):
        result = audit_qa_ledger(*cached_fixture())
        self.assertEqual(result['physical_http200'], 141)
        self.assertEqual(result['cache_consumers'], 1)
        self.assertEqual(result['cache_reuse'][0]['source_query_id'], 'c1_q0')
        self.assertEqual(result['cache_reuse'][0]['query_id'], 'c1_q1')

    def test_cache_source_different_response_is_rejected(self):
        consumers, ledger = cached_fixture()
        consumers[2]['event']['response_sha256'] = digest('different cached response')
        with self.assertRaisesRegex(ValueError, 'differs from its physical'):
            audit_qa_ledger(consumers, ledger)

    def test_cache_source_must_be_in_current_ledger(self):
        consumers, ledger = cached_fixture()
        h = digest('unrelated nonexistent cache source')
        consumers[2]['event']['request_hash'] = h
        ledger[2]['request_hash'] = h
        with self.assertRaisesRegex(ValueError, 'no successful physical source'):
            audit_qa_ledger(consumers, ledger)

    def test_cache_cannot_precede_physical_source_in_order(self):
        consumers, ledger = cached_fixture()
        ledger[0], ledger[2] = ledger[2], ledger[0]
        with self.assertRaisesRegex(ValueError, 'does not precede'):
            audit_qa_ledger(consumers, ledger)

    def test_cache_cannot_precede_physical_source_in_time(self):
        consumers, ledger = cached_fixture()
        ledger[2]['timestamp_utc'] = '2026-09-27T23:59:00+00:00'
        with self.assertRaisesRegex(ValueError, 'does not precede'):
            audit_qa_ledger(consumers, ledger)

    def test_usage_changed_vs_raw_response_rejected(self):
        consumers, ledger = fixture()
        ledger[0]['usage']['promptTokenCount'] = 1
        with self.assertRaisesRegex(ValueError, 'Raw response usage'):
            audit_qa_ledger(consumers, ledger)

    def test_cache_usage_must_equal_physical_even_without_optional_raw_usage(self):
        consumers, ledger = cached_fixture()
        for c in consumers:
            c.pop('response_usage')
        ledger[2]['usage']['candidatesTokenCount'] = 999
        with self.assertRaisesRegex(ValueError, 'differs from its physical'):
            audit_qa_ledger(consumers, ledger)

    def test_failed_physical_is_not_recovered_by_another_consumer(self):
        consumers, ledger = fixture()
        ledger[0]['status'] = 503
        with self.assertRaisesRegex(ValueError, 'once-only HTTP200'):
            audit_qa_ledger(consumers, ledger)

    def test_nonuniform_retry_rejected(self):
        consumers, ledger = fixture()
        ledger[0]['attempt'] = 2
        with self.assertRaisesRegex(ValueError, 'once-only HTTP200'):
            audit_qa_ledger(consumers, ledger)

    def test_consumer_physical_cache_flag_inconsistency_rejected(self):
        consumers, ledger = fixture()
        consumers[0]['event']['new_physical_attempt'] = False
        with self.assertRaises(ValueError):
            audit_qa_ledger(consumers, ledger)

    def test_fixed71_cannot_silently_shrink(self):
        consumers, ledger = fixture()
        with self.assertRaisesRegex(ValueError, 'Fixed remaining71'):
            audit_qa_ledger(consumers[:-2], ledger[:-2])

    def test_duplicate_stage_rejected(self):
        consumers, ledger = fixture()
        consumers[1] = copy.deepcopy(consumers[0])
        with self.assertRaises(ValueError):
            audit_qa_ledger(consumers, ledger)

    def test_naive_timestamp_is_rejected(self):
        consumers, ledger = fixture()
        ledger[0]['timestamp_utc'] = '2026-09-28T00:00:00'
        with self.assertRaisesRegex(ValueError, 'include timezone'):
            audit_qa_ledger(consumers, ledger)

    def test_real_shaped_terminal_reader_transport_failure_closes141_events(self):
        result = audit_qa_ledger(*transport_fixture())
        self.assertEqual(result['questions'], 71)
        self.assertEqual(result['consumers'], 141)
        self.assertEqual(result['physical_attempts'], 141)
        self.assertEqual(result['physical_http200'], 140)
        self.assertEqual(result['failed_or_unknown_consumers'], 1)
        self.assertEqual(result['unknown_usage_attempts'], 1)

    def test_unknown_transport_usage_cannot_be_erased(self):
        consumers, ledger = transport_fixture()
        ledger[0].pop('usage_unknown')
        with self.assertRaises(ValueError):
            audit_qa_ledger(consumers, ledger)

    def test_failed_reader_cannot_silently_continue_to_judge(self):
        consumers, ledger = fixture()
        consumers[0]['event'].update(source='failed_or_unknown_response', response_sha256=None)
        with self.assertRaisesRegex(ValueError, 'failed Reader'):
            audit_qa_ledger(consumers, ledger)

    def test_successful_reader_cannot_omit_judge(self):
        consumers, ledger = fixture()
        del consumers[1]
        del ledger[1]
        with self.assertRaisesRegex(ValueError, 'requires a terminal Judge'):
            audit_qa_ledger(consumers, ledger)

    def test_explicit_single_question_retry_namespace_can_close(self):
        consumers, ledger = fixture()
        for row in ledger[:2]:
            row['context']['recovery_version'] = 'new_explicit_namespace'
        result = audit_qa_ledger(consumers[:2], ledger[:2], expected_queries=[(1, 'c1_q0')])
        self.assertEqual(result['questions'], 1)
        self.assertEqual(result['physical_http200'], 2)
        self.assertTrue(result['explicit_retry_subset'])

    def test_explicit_retry_subset_cannot_consume_a_different_question(self):
        consumers, ledger = fixture()
        with self.assertRaisesRegex(ValueError, 'explicit recovery subset'):
            audit_qa_ledger(consumers[:2], ledger[:2], expected_queries=[(1, 'c1_q43')])

    def test_phase_override_is_bound_to_ledger(self):
        consumers, ledger = fixture()
        for row in ledger[:2]:
            row['context']['phase'] = 'explicit_new_phase'
        result = audit_qa_ledger(consumers[:2], ledger[:2], expected_queries=[(1, 'c1_q0')],
                                phase='explicit_new_phase')
        self.assertEqual(result['phase'], 'explicit_new_phase')

    def test_judge_http_failure_is_distinct_from_reader_failure(self):
        consumers, ledger = fixture()
        consumers[1]['event'].update(source='failed_or_unknown_response', response_sha256=None)
        consumers[1].pop('response_usage')
        ledger[1].update(status=503, usage={})
        result = audit_qa_ledger(consumers, ledger)
        self.assertEqual(result['consumers'], 142)
        self.assertEqual(result['physical_http200'], 141)
        self.assertEqual(result['failed_or_unknown_consumers'], 1)

    def test_cache_cannot_come_from_a_failed_physical_source(self):
        consumers, ledger = fixture()
        consumers[1]['event'].update(source='failed_or_unknown_response', response_sha256=None)
        consumers[1].pop('response_usage')
        ledger[1].update(status=503, usage={})
        consumers[3]['event'].update(source='recovery_cache', new_physical_attempt=False,
                                      request_hash=consumers[1]['event']['request_hash'])
        ledger[3].update(request_hash=ledger[1]['request_hash'], cache_hit=True,
                          usage_semantics='reused_response_no_new_model_usage')
        with self.assertRaisesRegex(ValueError, 'failed or unknown physical source'):
            audit_qa_ledger(consumers, ledger)


if __name__ == '__main__':
    unittest.main()
