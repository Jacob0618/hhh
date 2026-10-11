"""Independent narrow augmentation counterexamples; temporary files only."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import archive_amem_metric_closure_v2 as subject


class FrontierBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='metric_closure_independent_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, 'ROOT', self.root))
        self.stack.enter_context(patch.object(subject.old.engine, 'ROOT', self.root))
        self.stack.enter_context(patch.object(subject.old, 'ROOT', self.root))
        self.roots = {f'v{i}': self.root / f'run{i}' for i in range(1, 5)}
        self.stack.enter_context(patch.object(subject.old, 'ROOTS', self.roots))

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode() if isinstance(value, (dict, list)) else value
        path.write_bytes(raw)
        return str(path), hashlib.sha256(raw).hexdigest()

    def test_existing_records_are_hash_checked_leaves_without_live_read(self):
        absent = self.roots['v3'] / 'work/raw.json'
        h = 'a' * 64
        review = self.put(self.root / 'review.json', {'path': str(absent), 'sha256': h})
        record = {'path': str(absent), 'sha256': h, 'retained': True,
                  'size': 3, 'member': 'blobs/' + h}
        additions, payloads, kept = subject.frontier([review], [record], {}, None)
        self.assertEqual([r['path'] for r in additions], [review[0]])
        self.assertFalse(absent.exists())
        bad = self.put(self.root / 'bad.json', {'path': str(absent), 'sha256': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'conflicts with original'):
            subject.frontier([bad], [record], {}, None)

    def test_old_packaging_is_keep_only_and_new_run_evidence_rejected(self):
        oldplan = self.put(self.roots['v4'] / 'oldplan.json', {'packaging': True})
        additions, payloads, kept = subject.frontier([oldplan], [], dict([oldplan]), None)
        self.assertEqual(additions, []); self.assertEqual(payloads, {})
        self.assertEqual(kept, dict([oldplan]))
        injected = self.put(self.roots['v4'] / 'unreviewed.json', {'new': True})
        with self.assertRaisesRegex(ValueError, 'original run logical inventory'):
            subject.frontier([injected], [], {}, None)

    def test_new_graph_rejects_conflict_missing_foreign_and_malformed_map(self):
        valid = self.put(self.root / 'leaf.json', {})
        malformed = self.put(self.root / 'bad_map.json', {'code_sha256': valid[1]})
        with self.assertRaisesRegex(ValueError, 'dictionary'):
            subject.frontier([malformed], [], {}, None)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            subject.frontier([(valid[0], 'b' * 64), valid], [], {}, None)
        with self.assertRaises(FileNotFoundError):
            subject.frontier([(str(self.root / 'missing'), 'a' * 64)], [], {}, None)
        with self.assertRaises(ValueError):
            subject.frontier([(str(self.root.parent / 'outside'), 'a' * 64)], [], {}, None)


class StrictReplayBoundaryTests(unittest.TestCase):
    def test_runtime_exact_hash_write_and_unknown_loader_checks(self):
        import importlib.machinery
        with tempfile.TemporaryDirectory(prefix='metric_runtime_hash_') as directory:
            root = Path(directory).resolve(); path = root / 'runtime/known.py'
            path.parent.mkdir(); path.write_bytes(b'known module')
            runtime = {'runtime_modules_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}}
            view = SimpleNamespace(records={})
            original_bytecode = sys.dont_write_bytecode
            with patch.object(subject, 'ROOT', root):
                with subject.only_zip_workspace_reads(view, runtime):
                    self.assertEqual(path.read_bytes(), b'known module')
                    with self.assertRaises(AssertionError): path.write_bytes(b'changed')
                    loader = importlib.machinery.SourceFileLoader('fixture', str(path))
                    with self.assertRaises(AssertionError): loader.path_stats(str(root / 'unknown.py'))
                    self.assertTrue(sys.dont_write_bytecode)
                path.write_bytes(b'changed')
                with subject.only_zip_workspace_reads(view, runtime):
                    with self.assertRaisesRegex(AssertionError, 'runtime module changed'): path.read_bytes()
            self.assertEqual(sys.dont_write_bytecode, original_bytecode)

    def test_unknown_runtime_source_or_artifact_cannot_fall_back_live(self):
        with tempfile.TemporaryDirectory(prefix='metric_runtime_independent_') as directory:
            root = Path(directory).resolve(); path = root / 'runtime/unknown_science.py'
            path.parent.mkdir(); path.write_text('unretained evidence', encoding='utf8')
            with patch.object(subject, 'ROOT', root):
                with subject.only_zip_workspace_reads(SimpleNamespace(records={}), {'files_sha256': {}}):
                    with self.assertRaises(AssertionError):
                        path.read_bytes()

    def test_retained_source_uses_zip_bytes_and_blocks_write(self):
        with tempfile.TemporaryDirectory(prefix='metric_source_independent_') as directory:
            root = Path(directory).resolve(); path = root / 'metric.py'
            path.write_bytes(b'live changed value')
            import os
            view = SimpleNamespace(records={os.path.normcase(str(path)): {'size': 8}},
                                   bytes=lambda p: b'zip only')
            with patch.object(subject, 'ROOT', root):
                with subject.only_zip_workspace_reads(view, {'files_sha256': {}}):
                    self.assertEqual(path.read_bytes(), b'zip only')
                    with self.assertRaises(AssertionError):
                        path.write_bytes(b'mutation')
            self.assertEqual(path.read_bytes(), b'live changed value')


class DispatcherBoundaryTests(unittest.TestCase):
    def test_actual_run_has_one_strong_gate_before_base_run(self):
        from tools import run_five_baselines_reviewed_chain_v4 as launch
        inner = launch.frozen.frozen
        with patch.object(sys, 'argv', ['entry', 'run', '--method', 'simplemem']), \
             patch.object(launch, 'verify_code_review'), patch.object(launch, 'assert_stopped'), \
             patch.object(launch.archive, 'verify_completion', return_value={'status': 'passed'}) as completion, \
             patch.object(inner.base, 'manifest', return_value={'review': {'path': 'fixture.json'}}), \
             patch.object(inner.base, 'run', return_value={'offline_fixture': True}) as run, \
             patch.object(inner.base, 'paths', inner.base.paths), patch('sys.stdout', new_callable=io.StringIO):
            launch.main()
            completion.assert_called_once_with()
            run.assert_called_once_with('simplemem')

    def test_actual_prepare_route_requires_completion_before_base_prepare(self):
        from tools import run_five_baselines_reviewed_chain_v4 as launch
        inner = launch.frozen.frozen
        review = {'code_sha256': {launch.frozen.FROZEN_ENTRY: launch.frozen.FROZEN_SHA}}
        with patch.object(sys, 'argv', ['entry', 'prepare', '--method', 'simplemem', '--review', 'fixture.json']), \
             patch.object(launch, 'verify_code_review'), patch.object(launch, 'assert_stopped'), \
             patch.object(launch.archive, 'verify_completion', side_effect=ValueError('missing predecessor completion')), \
             patch.object(inner, 'read', return_value=review), \
             patch.object(inner.base, 'prepare', return_value={'unexpected': 'must not prepare'}) as prepare, \
             patch.object(inner.base, 'paths', inner.base.paths):
            with self.assertRaisesRegex(ValueError, 'missing predecessor completion'):
                launch.main()
            prepare.assert_not_called()

    def test_exact_guard_parent_and_other_controllers(self):
        from tools import run_five_baselines_reviewed_chain_v4 as launch
        command = 'tools/run_five_baselines_reviewed_chain_v4.py run --method simplemem'
        parent = {'ProcessId': 9, 'ParentProcessId': 1, 'Name': 'python.exe',
                  'CommandLine': '"venv python" ' + command,
                  'ExecutablePath': str(launch.ROOT / 'runtime/baseline_env_v1/Scripts/python.exe'),
                  'CreatedUtc': '2026-10-09T00:00:00+00:00'}
        current = {'ProcessId': 10, 'ParentProcessId': 9, 'Name': 'python.exe',
                   'CommandLine': '"base python" ' + command,
                   'ExecutablePath': sys._base_executable,
                   'CreatedUtc': '2026-10-09T00:00:01+00:00'}
        self.assertEqual(launch.active_controllers([parent, current], 10), [])
        for changes in ({'ExecutablePath': 'C:/wrong/python.exe'},
                        {'CommandLine': 'python ' + command + ' --different'},
                        {'CreatedUtc': '2026-10-09T00:00:02+00:00'}):
            with self.subTest(changes=changes):
                self.assertEqual(launch.active_controllers([{**parent, **changes}, current], 10),
                                 [{**parent, **changes}])
        other = {**parent, 'ProcessId': 22, 'Name': 'python3.11.exe',
                 'CommandLine': 'python -m tools.run_five_baselines_unknown run'}
        self.assertEqual(launch.active_controllers([parent, current, other], 10), [other])
        self.assertNotIn(launch.LAUNCHER, launch.guard.LAUNCHERS)

    def test_context_routes_new_plan_and_restores_on_failure(self):
        from tools import run_five_baselines_reviewed_chain_v4 as launch
        oldguard = launch.guard.assert_stopped
        oldplan = launch.frozen.amem.DEFAULT_PLAN
        oldverify = launch.frozen.verify_predecessors
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with launch.private_context():
                self.assertIs(launch.guard.assert_stopped, launch.assert_stopped)
                self.assertEqual(launch.frozen.amem.DEFAULT_PLAN, launch.archive.DEFAULT_PLAN)
                self.assertIs(launch.frozen.verify_predecessors, launch.verify_predecessors)
                raise RuntimeError('sentinel')
        self.assertIs(launch.guard.assert_stopped, oldguard)
        self.assertEqual(launch.frozen.amem.DEFAULT_PLAN, oldplan)
        self.assertIs(launch.frozen.verify_predecessors, oldverify)

    def test_powershell_only_changes_default_plan_and_checks_keep_before_intent(self):
        root = Path(__file__).resolve().parents[1]
        before = (root / 'tools/cleanup_amem_retrieval_continuation_v1.ps1').read_bytes()
        after = (root / 'tools/cleanup_amem_metric_closure_v2.ps1').read_bytes()
        self.assertEqual(after, before.replace(b'cleanup_plan_retrieval_v1.json',
                                             b'cleanup_plan_retrieval_v2.json'))
        text = after.decode('utf8')
        self.assertLess(text.index('$plan.keep_sha256'), text.index('WriteExclusive $intentPath'))

    def test_old_plan_explicitly_rejected_and_official_source_in_future_code_map(self):
        with self.assertRaisesRegex(ValueError, 'New v2 plan is mandatory'):
            subject.verify_derivation(subject.SOURCE_PLAN)
        table = subject.code_hashes()
        self.assertEqual(len(table), 22)
        self.assertEqual(table[subject.METRIC], hashlib.sha256((subject.ROOT / subject.METRIC).read_bytes()).hexdigest())
        for name in subject.EXTRA: self.assertIn(name, table)


if __name__ == '__main__':
    unittest.main()
