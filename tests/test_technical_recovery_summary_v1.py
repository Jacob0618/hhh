import copy
import hashlib
import json
import unittest
from tools.summarize_technical_recovery_v1 import accept_overlay, group


def signed(row):
    row = copy.deepcopy(row)
    row.pop('row_sha256', None)
    row['row_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return row


class RecoverySummaryTests(unittest.TestCase):
    def fixture(self):
        q = {'id': 'c0_q0', 'question': 'Where?', 'reference': 'Paris', 'category': 1}
        row = {'architecture': 'mem0', 'conversation': 0, 'method': 'rank', 'query_id': q['id'],
               'seal_sha256': 'seal', 'status': 'complete', 'question': q['question'], 'reference': q['reference'],
               'category': 1, 'reader_protocol': 'brief_answer_local_evidence_ids_v2',
               'selected_ids': [f'f{i}' for i in range(20)], 'selected_count': 20, 'prediction': {'answer': 'London'},
               'judgment': {'correct': False, 'completeness': 0, 'reason': 'Different city'}, 'token_f1': 0, 'exact_match': 0,
               'reader_model': 'gemini-2.5-flash', 'judge_model': 'gemini-3.5-flash-lite',
               'request_events': [{'stage': 'qa_reader'}]}
        return q, signed(row), {('mem0', 0, 'rank', 'c0_q0')}

    def test_wrong_answer_is_technical_success(self):
        q, row, failed = self.fixture()
        self.assertFalse(accept_overlay(row, q, failed_keys=failed, seen=set(), seal_sha='seal')['strict_correct'])

    def test_no_overwrite_first_success(self):
        q, row, _ = self.fixture()
        with self.assertRaises(ValueError):
            accept_overlay(row, q, failed_keys=set(), seen=set(), seal_sha='seal')

    def test_corruption_rejected(self):
        q, row, failed = self.fixture(); row['token_f1'] = .8
        with self.assertRaises(ValueError):
            accept_overlay(row, q, failed_keys=failed, seen=set(), seal_sha='seal')

    def test_signed_metric_drift_rejected(self):
        q, row, failed = self.fixture(); row['token_f1'] = .8; row = signed(row)
        with self.assertRaises(ValueError):
            accept_overlay(row, q, failed_keys=failed, seen=set(), seal_sha='seal')

    def test_failure_keeps_denominator(self):
        self.assertEqual(group([])['missing'], 1540)
        self.assertIsNone(group([])['metrics']['official_locomo_f1']['fully_observed'])
        q, row, failed = self.fixture(); row['status'] = 'unknown'; row = signed(row)
        self.assertIsNone(accept_overlay(row, q, failed_keys=failed, seen=set(), seal_sha='seal'))

    def test_duplicate_rejected(self):
        q, row, failed = self.fixture(); seen = set()
        accept_overlay(row, q, failed_keys=failed, seen=seen, seal_sha='seal')
        with self.assertRaises(ValueError):
            accept_overlay(row, q, failed_keys=failed, seen=seen, seal_sha='seal')

    def test_nonfinite_and_false_budget_rejected(self):
        for field, value in [('token_f1', float('nan')), ('selected_count', 0), ('selected_ids', list(range(20))), ('request_events', [])]:
            q, row, failed = self.fixture(); row[field] = value; row = signed(row)
            with self.assertRaises(ValueError):
                accept_overlay(row, q, failed_keys=failed, seen=set(), seal_sha='seal')


if __name__ == '__main__':
    unittest.main()
