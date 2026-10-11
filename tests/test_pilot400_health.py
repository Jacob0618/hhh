import copy
import unittest
from tools.check_pilot400_health import compare, progress


class HealthTests(unittest.TestCase):
    def base(self):
        return {'captured_utc': '2026-09-27T22:00:00+00:00', 'worker': {'pid': 1, 'creation_time_100ns': 2},
                'problems': [], 'api_bytes': 0, 'state': {'status': 'running'},
                'progress': {'0': progress({'status': 'running', 'groups_completed': 1})}}

    def test_live_pid_alone_not_health(self):
        a = self.base(); b = copy.deepcopy(a); b['captured_utc'] = '2026-09-27T22:00:30+00:00'
        self.assertTrue(compare(a, b, [])['requires_investigation'])

    def test_only_errors_not_health(self):
        a = self.base(); b = copy.deepcopy(a)
        self.assertTrue(compare(a, b, [{'context': {'conversation': 0}, 'status': 520}])['requires_investigation'])

    def test_success_context_only(self):
        a = self.base(); b = copy.deepcopy(a)
        self.assertTrue(compare(a, b, [{'context': {'conversation': 1}, 'status': 200}])['requires_investigation'])
        self.assertFalse(compare(a, b, [{'context': {'conversation': 0}, 'status': 200}])['requires_investigation'])

    def test_long_stall_even_success(self):
        old = self.base(); a = self.base(); b = copy.deepcopy(a)
        b['captured_utc'] = '2026-09-27T22:20:00+00:00'
        self.assertTrue(compare(a, b, [{'context': {'conversation': 0}, 'status': 200}], historical=old)['requires_investigation'])

    def test_qa_and_tags_use_stage_progress(self):
        self.assertEqual(progress({'methods': {'a': {'completed': 8, 'failed': 2}}})['qa_completed'], 8)
        self.assertEqual(progress({'facts_completed': 3})['facts_completed'], 3)

    def test_pending_is_not_stalled(self):
        a = self.base(); a['progress']['0']['status'] = 'pending'
        self.assertFalse(compare(a, copy.deepcopy(a), [])['requires_investigation'])

    def test_log_shrink_requires_investigation(self):
        a = self.base(); a['api_bytes'] = 10
        b = copy.deepcopy(a); b['api_bytes'] = 0
        self.assertTrue(compare(a, b, [{'context': {'conversation': 0}, 'status': 200}])['requires_investigation'])


if __name__ == '__main__':
    unittest.main()
