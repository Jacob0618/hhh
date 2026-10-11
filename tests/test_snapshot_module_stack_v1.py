"""Read-only health counters retain failures, unfinished tails and denominators."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import snapshot_module_stack_v1 as health


def dump(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row), encoding='utf-8')


class HealthTests(unittest.TestCase):
    def test_prepared_files_visible_before_manifest_without_reading_contents(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            before={'captured_utc':'2026-10-05T00:00:00+00:00','runs':[health.snapshot(run)]}
            folder=run/'phases'/'support'
            dump(folder/'units'/'u0.json',{'private_source':'not copied to the snapshot'})
            dump(folder/'requests'/'r0.json',{'private_request':'not copied to the snapshot'})
            dump(folder/'requests'/'r1.json',{'private_request':'another request'})
            # No status/manifest/outcome exists. Metadata observation must not
            # open large units or requests, nor claim that they are validated.
            with patch.object(health,'read',side_effect=AssertionError('Content must not be opened')):
                result=health.snapshot(run)
            support=result['phases']['support']
            self.assertTrue(support['directory_exists'])
            self.assertFalse(support['prepared_manifest']['exists'])
            self.assertFalse(support['firstpass']['exists'])
            self.assertEqual(support['prepared_files']['units']['file_count'],1)
            self.assertEqual(support['prepared_files']['requests']['file_count'],2)
            self.assertGreater(support['prepared_files']['requests']['bytes'],0)
            self.assertIsNotNone(support['prepared_files']['units']['latest']['mtime_utc'])
            self.assertNotIn('private_source',json.dumps(result))
            delta=health.compare(before,{'captured_utc':'2026-10-05T00:00:30+00:00','runs':[result]})
            self.assertEqual(delta['runs'][0]['prepared_file_deltas']['support/units']['file_count_delta'],1)
            self.assertEqual(delta['runs'][0]['prepared_file_deltas']['support/requests']['file_count_delta'],2)
            self.assertTrue(delta['runs'][0]['observable_counters_changed'])

    def test_phase_and_qa_counts_exclude_raw_content_and_keep_nested_failures(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            producer = run / 'phases' / 'support' / 'firstpass'
            dump(run / 'phases' / 'support' / 'manifest.json', {'requests': [{'path': 'secret_request_content'}] * 3, 'units': [{}] * 2})
            dump(producer / 'execution_intent.json', {'pid': 42, 'requests': [{}] * 3, 'created_utc': '2026-10-05'})
            dump(producer / 'status.json', {'status': 'running', 'expected_requests': 3, 'terminal_requests': 2})
            dump(producer / 'responses' / 'good.json', {'status': 'complete', 'conversation': 0, 'scores': {'private': .9}})
            dump(producer / 'responses' / 'bad.json', {'status': 'failed', 'conversation': 1, 'error': 'private message'})
            dump(run / 'conv_0' / 'qa' / 'baseline' / 'q0.json', {'status': 'complete', 'conversation': 0, 'reader_answer': 'private answer'})
            dump(run / 'run_outcome.json', {'terminal': True, 'total': 1,
                 'outcomes': [{'terminal': True, 'local_errors': [{'message': 'private error'}]}]})
            result = health.snapshot(run)
            self.assertEqual(result['phases']['support']['firstpass']['responses']['statuses'], {'failed': 1, 'complete': 1})
            self.assertEqual(result['phases']['support']['prepared_manifest']['state']['request_refs_count'], 3)
            self.assertEqual(result['phases']['support']['firstpass']['execution_intent']['state']['pid'], 42)
            self.assertEqual(result['states']['run_outcome.json']['state']['outcome_local_error_count'], 1)
            self.assertEqual(result['qa_artifacts'][str(Path('conv_0') / 'qa' / 'baseline')]['readable_count'], 1)
            self.assertNotIn('private', json.dumps(result))
            self.assertNotIn('secret_request_content', json.dumps(result))

    def test_ledger_keeps_physical_cache_failure_and_partial_tail_separate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'api_calls.jsonl'
            base = {'provider': 'jev', 'stage': 'module_stack_jev_v1', 'request_hash': 'a' * 64,
                    'timestamp_utc': '2026-10-05T00:00:00Z', 'context': {'conversation': 0, 'phase': 'support'}}
            rows = [{**base, 'status': 200, 'cache_hit': False, 'attempt': 1},
                    {**base, 'cache_hit': True},
                    {**base, 'status': 'transport_error', 'usage_unknown': True, 'attempt': 1}]
            raw = ''.join(json.dumps(row) + '\n' for row in rows).encode() + b'{unfinished'
            path.write_bytes(raw)
            errors = []
            result = health.ledger(path, errors)
            self.assertEqual(result['counts']['physical'], 2)
            self.assertEqual(result['counts']['cache'], 1)
            self.assertEqual(result['counts']['status_transport_error'], 1)
            self.assertEqual(result['counts']['unknown_usage_rows'], 1)
            self.assertEqual(result['in_flight_tail_bytes'], len(b'{unfinished'))
            self.assertEqual(errors, [])

    def test_comparison_reports_observed_progress_without_health_claim(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            before = {'captured_utc': '2026-10-05T00:00:00+00:00', 'runs': [health.snapshot(run)]}
            dump(run / 'phases' / 'rank' / 'firstpass' / 'responses' / 'r0.json', {'status': 'failed', 'conversation': 0})
            after = {'captured_utc': '2026-10-05T00:00:30+00:00', 'runs': [health.snapshot(run)]}
            result = health.compare(before, after)
            self.assertEqual(result['elapsed_seconds'], 30)
            self.assertTrue(result['not_a_health_verdict'])
            self.assertTrue(result['runs'][0]['observable_counters_changed'])
            self.assertEqual(result['runs'][0]['phases']['rank/firstpass']['response_file_delta'], 1)

    def test_command_redaction_and_missing_run(self):
        command = 'python module_stack.py --api-key=EXAMPLE1 --password "EXAMPLE 2" Authorization: Bearer EXAMPLE3'
        redacted = health._redact_command(command)
        for secret in ('EXAMPLE1', 'EXAMPLE 2', 'EXAMPLE3'):
            self.assertNotIn(secret, redacted)
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(health.snapshot(Path(d) / 'absent')['exists'])


if __name__ == '__main__': unittest.main()
