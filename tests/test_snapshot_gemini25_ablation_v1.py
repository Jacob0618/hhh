"""Offline current-run observer tests; no live API, vault or native DB reads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import snapshot_gemini25_ablation_v1 as health
from tools.check_experiment_health import shared_read


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class Gemini25HealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gemini25_health_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'gemini25_ablation_writer_v1_test'
        self.run.mkdir()

    def test_metadata_counts_cover_writer_source_qa_without_payload_reads(self):
        paths = [self.run / 'conv_0/session/native/session_audit/user/session_1.json',
                 self.run / 'conv_0/session/bank.json',
                 self.run / 'conv_0/retrieval/c0_q0.json',
                 self.run / 'phases/r03/units/c0_q0.json',
                 self.run / 'phases/support/requests/c1_fact_abc_b0.json',
                 self.run / 'phases/support/firstpass/responses/c1_fact_abc_b0.json',
                 self.run / 'conv_0/reader/baseline/c0_q0.json',
                 self.run / 'conv_0/reader/baseline/c0_q0.intent.json',
                 self.run / 'selections/r03_original/c0_q0.json']
        for path in paths:
            dump(path, {'payload': 'PRIVATE_CONTENT_SHOULD_NEVER_BE_OPENED'})
        # An old nested run/native-store ledger must not be searched or counted.
        dump(self.run / 'history/old_run/api_calls.jsonl', {'status': 200})
        dump(self.run / 'conv_0/native/qdrant/api_calls.jsonl', {'status': 200})
        dump(self.run / 'conv_0/session/native/api_cache/raw.json', {'api_key': 'PRIVATE_RAW'})
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        with patch.object(health, 'shared_read', side_effect=AssertionError('No payload reads')):
            result = health.snapshot(self.run)
        artifacts = result['artifacts']
        self.assertEqual(artifacts['conv_0/writer_journals']['file_count'], 1)
        self.assertEqual(artifacts['conv_0/writer_bank']['file_count'], 1)
        self.assertEqual(artifacts['conv_0/reader/baseline']['file_count'], 1)
        self.assertEqual(artifacts['conv_0/reader/baseline']['intent_file_count'], 1)
        self.assertEqual(artifacts['phases/support/firstpass/responses']['per_conversation'], {'1': 1})
        self.assertEqual(result['api_ledgers'], [])
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_status_whitelist_retains_failure_counts_excludes_errors_and_payload(self):
        path = self.run / 'conv_0/session/write_status.json'
        dump(path, {'status': 'failed', 'pid': 123, 'completed_groups': 2, 'expected_groups': 19,
                    'facts': 40, 'error': {'message': 'PRIVATE_ERROR'}, 'prompt': 'PRIVATE_PROMPT',
                    'counts': {'failed': 1, 'payload': 'PRIVATE_BAD_COUNT'},
                    'outcomes': [{'status': 'failed', 'local_errors': [{'message': 'PRIVATE_NESTED'}]}]})
        self.assertIs(health.shared_read, shared_read)
        with patch.object(health, 'shared_read', wraps=shared_read) as reads:
            result = health.snapshot(self.run)
        self.assertEqual(reads.call_count, 1)
        row = result['states']['conv_0/session/write_status.json']['state']
        self.assertEqual((row['status'], row['completed_groups'], row['expected_groups'], row['facts']),
                         ('failed', 2, 19, 40))
        self.assertEqual(row['outcome_local_error_count'], 1)
        self.assertEqual(row['counts'], {'failed': 1})
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_incremental_ledger_preserves_physical_cache_errors_partial_tail(self):
        path = self.run / 'conv_0/api/api_calls.jsonl'
        path.parent.mkdir(parents=True)
        base = {'timestamp_utc': '2026-10-07T00:00:00Z', 'stage': 'qa_reader_gemini25_ablation_plain_v1',
                'provider': 'gemini', 'model': 'gemini-2.5-flash', 'request_hash': 'a' * 64,
                'context': {'conversation': 0, 'phase': 'qa'}, 'error': 'PRIVATE_ERROR',
                'body': {'prompt': 'PRIVATE_PROMPT'}}
        values = [{**base, 'status': 200, 'cache_hit': False}, {**base, 'cache_hit': True},
                  {**base, 'status': 'transport_error', 'usage_unknown': True}]
        completed = ''.join(json.dumps(value) + '\n' for value in values).encode()
        future = (json.dumps({**base, 'timestamp_utc': '2026-10-07T00:00:10Z', 'status': 200}) + '\n').encode()
        path.write_bytes(completed + future[:30])
        first = health.snapshot(self.run)
        initial = first['api_ledgers'][0]
        self.assertEqual(initial['counts']['physical'], 2)
        self.assertEqual(initial['counts']['successes'], 1)
        self.assertEqual(initial['counts']['errors'], 1)
        self.assertEqual(initial['counts']['cache'], 1)
        self.assertEqual(initial['in_flight_tail_bytes'], 30)
        with path.open('ab') as handle:
            handle.write(future[30:])
        with patch.object(health, 'shared_read', wraps=shared_read) as reader:
            second = health.snapshot(self.run, first)
        row = second['api_ledgers'][0]
        self.assertTrue(row['incremental'])
        self.assertEqual(reader.call_args.kwargs['offset'], len(completed))
        self.assertEqual(row['counts']['physical'], 3)
        self.assertEqual(row['counts']['successes'], 2)
        self.assertEqual(row['bytes_read_this_snapshot'], len(future))
        self.assertEqual(row['in_flight_tail_bytes'], 0)
        self.assertNotIn('PRIVATE', json.dumps(second))

    def test_shrunk_log_resets_and_malformed_complete_rows_are_explicit(self):
        path = self.run / 'conv_0/api/api_calls.jsonl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{"status":200,"stage":"reader"}\n' * 4)
        first = health.snapshot(self.run)
        path.write_bytes(b'invalid\n{"status":429,"stage":"reader"}\n')
        second = health.snapshot(self.run, first)
        row = second['api_ledgers'][0]
        self.assertTrue(row['reset_or_shrank'])
        self.assertEqual(row['counts']['physical'], 1)
        self.assertEqual(row['counts']['errors'], 1)
        self.assertEqual(row['malformed_complete_rows'], 1)
        self.assertEqual(second['read_error_count'], 1)

    def test_process_descendants_parents_identity_and_secret_redaction(self):
        rows = [
            {'pid': 10, 'parent_pid': 1, 'creation_utc': 'old', 'command_line': 'python tools/run_gemini25_ablation_writer_v1.py --api-key=PRIVATE_A'},
            {'pid': 11, 'parent_pid': 10, 'creation_utc': 'now', 'command_line': 'python -c spawn_main()'},
            {'pid': 1, 'parent_pid': 0, 'creation_utc': 'older', 'command_line': 'parent'},
            {'pid': 99, 'parent_pid': 1, 'creation_utc': 'other', 'command_line': 'python historical_run.py'},
        ]
        selected = health.select_processes(rows, [])
        self.assertEqual({row['pid'] for row in selected}, {1, 10, 11})
        self.assertEqual(next(row for row in selected if row['pid'] == 11)['parent_pid'], 10)
        command = ('GEMINI_API_KEY="PRIVATE B" --token=PRIVATE_C --password \'PRIVATE D\' '
                   'Authorization: Bearer PRIVATE_E https://u:PRIVATE_F@example.org/a?key=PRIVATE_G')
        redacted = health.redact(command)
        self.assertNotIn('PRIVATE', redacted)
        self.assertNotIn('PRIVATE', json.dumps(selected))
        old = {'runs': [], 'processes': {'rows': selected}}
        changed = deepcopy(selected)
        next(row for row in changed if row['pid'] == 10)['creation_utc'] = 'new'
        comparison = health.compare(old, {'runs': [], 'processes': {'rows': changed}})
        self.assertEqual(comparison['pid_creation_changed'], [10])
        self.assertTrue(comparison['not_a_health_verdict'])

    def test_compare_counts_stage_status_and_elapsed_without_health_claim(self):
        first = {'captured_utc': '2026-10-07T00:00:00+00:00', 'runs': [health.snapshot(self.run)]}
        dump(self.run / 'conv_0/session/write_status.json', {'status': 'running', 'completed_groups': 1})
        dump(self.run / 'conv_0/session/native/session_audit/user/session_1.json', {'private': 'not read'})
        second = {'captured_utc': '2026-10-07T00:00:30+00:00', 'runs': [health.snapshot(self.run)]}
        result = health.compare(first, second)
        self.assertEqual(result['elapsed_seconds'], 30)
        row = result['runs'][0]
        self.assertEqual(row['artifact_deltas']['conv_0/writer_journals']['file_count_delta'], 1)
        self.assertEqual(row['state_changes']['conv_0/session/write_status.json']['completed_groups']['after'], 1)
        self.assertTrue(result['not_a_health_verdict'])

    def test_only_new_runs_and_output_outside_science_are_allowed(self):
        with self.assertRaises(ValueError):
            health.snapshot(self.root / 'gemini38_old')
        self.assertFalse(health.snapshot(self.root / 'gemini25_ablation_missing')['exists'])
        with patch('sys.argv', ['snapshot', '--run-dir', str(self.run), '--output', str(self.run / 'health.json')]):
            with self.assertRaises(ValueError):
                health.main()
        self.assertFalse((self.run / 'health.json').exists())
        outside = self.root / 'historical.json'
        dump(outside, {'payload': 'not in selected run'})
        self.assertEqual(health.metadata([outside], [], root=self.run)['file_count'], 0)
        self.assertEqual(list(health.files(self.root, '*.json', self.run)), [])

    def test_cli_repeatable_run_dirs_compare_and_exclusive_output(self):
        missing = self.root / 'gemini25_ablation_qa_missing'
        first, second = self.root / 'health_1.json', self.root / 'health_2.json'
        args = ['snapshot', '--run-dir', str(self.run), '--run-dir', str(missing)]
        disk = type('Disk', (), {'free': 50, 'used': 50, 'total': 100})()
        with patch.object(health, 'processes', return_value={'rows': [], 'error_type': None}), \
             patch.object(health.shutil, 'disk_usage', return_value=disk), patch('builtins.print'):
            with patch('sys.argv', args + ['--output', str(first)]):
                health.main()
                with self.assertRaises(FileExistsError):
                    health.main()
            with patch('sys.argv', args + ['--output', str(second), '--compare', str(first)]):
                health.main()
        value = json.loads(second.read_text(encoding='utf-8'))
        self.assertEqual(len(value['runs']), 2)
        self.assertFalse(value['runs'][1]['exists'])
        self.assertEqual(value['disk_D']['free_bytes'], 50)
        self.assertEqual((value['model_calls'], value['credential_reads']), (0, 0))
        self.assertTrue(value['comparison']['not_a_health_verdict'])


if __name__ == '__main__':
    unittest.main()
