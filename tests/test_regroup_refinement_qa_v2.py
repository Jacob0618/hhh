"""Offline V2 source negatives plus the unchanged V1 QA behavioral suite."""
from concurrent.futures import ProcessPoolExecutor
import copy
import importlib.util
import json
import multiprocessing
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import run_regroup_refinement_qa_v1 as original
from tools import run_regroup_refinement_qa_v2 as driver
from tools import regroup_refinement_qa_sources_v2 as sources

# Isolate the reused tests' d/s globals rather than modifying their original
# module. Every behavioral test runs against the same implementation body in V2.
_spec = importlib.util.spec_from_file_location('_isolated_refinement_qa_v2_cases',
    Path(__file__).with_name('test_regroup_refinement_qa_v1.py'))
_reused = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reused)
_reused.d, _reused.s = driver, sources


class ReusedV1BehaviorTests(_reused.RefinementQATests):
    pass


class BoundExecutionTests(unittest.TestCase):
    def test_all_bodies_identical_and_v1_globals_and_objects_unchanged(self):
        before = {name: value for name, value in vars(original).items()}
        self.assertIs(driver.POLICY, original.POLICY)
        self.assertIs(driver.native_qa.retrieve_arm, original.native_qa.retrieve_arm)
        self.assertIs(driver.plain.one, original.plain.one)
        self.assertIs(driver.sources.copy_bank, original.sources.copy_bank)
        for name in driver.REBOUND_FUNCTIONS:
            first, second = getattr(original, name), getattr(driver, name)
            self.assertIs(second.__code__, first.__code__)
            self.assertIs(second.__globals__, vars(driver))
            self.assertIs(first.__globals__, vars(original))
            self.assertIsNot(first, second)
            self.assertEqual(second.__module__, driver.__name__)
            self.assertEqual(second.__qualname__, first.__qualname__)
            self.assertIs(pickle.loads(pickle.dumps(second)), second)
        driver.check_job(_reused.job(), 'jev_refined')
        self.assertEqual(set(before), set(vars(original)))
        self.assertTrue(all(vars(original)[name] is obj for name, obj in before.items()))
        self.assertEqual(original.SCHEMA, 'regroup-refinement-plain-qa400-v1')
        self.assertEqual(original.sources.GATE_SCHEMA, 'regroup-refinement-qa-gate-v1')
        self.assertIn('tools/review_regroup_refinement_banks_v2.py', driver.CODE)

    def test_real_windows_style_spawn_resolves_rebound_module_entry(self):
        # check_job only validates an in-memory job; no model, native client,
        # credentials or external files are used in the spawned interpreter.
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertIsNone(pool.submit(driver.check_job, _reused.job(), 'jev_refined').result(timeout=45))
            bad = _reused.job(); bad['selected_ids'].pop()
            with self.assertRaises(ValueError):
                pool.submit(driver.check_job, bad, 'jev_refined').result(timeout=45)


class SourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='refinement_sources_v2_offline_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for target in ('requests.Session.request', 'socket.create_connection', 'jev_mem.providers.load_credentials'):
            p = patch(target, side_effect=AssertionError('No real credentials/network'))
            p.start(); self.addCleanup(p.stop)
        self.parents, self.rows, self.bank_data, self.export_refs = {}, [], {}, {}
        self.evidence = self.put(self.root / 'evidence.json', {'mock': 'source/index/semantic report'})
        resource = {'snapshot': str(self.root / 'local_resources'), 'files_sha256': {'mock': 'frozen'}}
        for arm in sources.ARMS:
            run = self.root / arm
            inputs = []
            for ci in range(10):
                folder = run / f'conv_{ci}' / arm
                inp = self.put(folder / 'group_inputs.json', {'envelopes': [{'group_id': 'g0'}]})
                journal = self.put(folder / 'journal.json', {'group_id': 'g0'})
                inputs.append({'conversation': ci, 'arm': arm, 'input': inp})
                native = folder / 'native'; native.mkdir()
                (native / 'history.sqlite').write_bytes(b'closed mock file, never opened as SQLite')
                facts = [{'id': f'c{ci}_f{k}', 'text': f'fact{k}', 'observation_date': '2023-01-01'} for k in range(20)]
                bank = {'conversation': ci, 'arm': arm, 'protocol_sha256': arm + '_protocol', 'complete': True,
                        'source_dataset_sha256': sources.SOURCE_HASH, 'group_inputs': inp, 'journal_files': [journal],
                        'facts': facts, 'native_dir': str(native), 'user_id': f'c{ci}_{arm}', 'fact_support': 'unverified'}
                failed = (ci, arm) == (3, 'source_local_context')
                bank_ref = self.put((self.root / 'exports' / f'conv_{ci}' / arm / 'bank.json') if failed else folder / 'bank.json', bank)
                state = {'conversation': ci, 'arm': arm, 'protocol_sha256': arm + '_protocol',
                         'status': 'failed' if failed else 'complete', 'finished_utc': '2026-09-30T12:00:00+00:00',
                         'completed_groups': 1, 'expected_groups': 1, 'facts': 20}
                if not failed: state['bank'] = bank_ref
                state_ref = self.put(folder / 'write_status.json', state)
                outcome = {k: state[k] for k in ('conversation', 'arm', 'protocol_sha256', 'status')}
                outcome['write_status'] = state_ref
                if not failed: outcome['bank'] = bank_ref
                outcome_ref = self.put(folder / 'write_outcome.json', outcome)
                row = {'conversation': ci, 'arm': arm, 'source_kind': 'reviewed_export' if failed else 'strict_complete',
                       'original_write_status': state['status'], 'bank': None if failed else bank_ref,
                       'write_status': state_ref, 'outcome': outcome_ref, 'group_inputs': inp,
                       'native_dir': str(native), 'native_inventory': sources.inventory(native),
                       'groups': 1, 'expected_groups': 1, 'facts': 20, 'orphan_fact_ids': ['mock_orphan'] if failed else [],
                       'audits': {'source_date': [self.evidence], 'index': [self.evidence]},
                       'native_index_limitations': ['unrepaired orphan'] if failed else ['unverified support'],
                       'proof': self.evidence if failed else None}
                self.rows.append(row)
                self.bank_data[ci, arm] = (sources.verify(bank_ref['path']), bank_ref)
                if failed:
                    self.export_refs[ci] = self.put(self.root / 'exports' / f'conv_{ci}' / arm / 'reviewed_export.json', {'mock': 'export'})
            parent = {'arms': [arm], 'protocol_sha256': arm + '_protocol', 'inputs': inputs,
                      'refinement': {'resources': resource}}
            pref = self.put(run / 'run_manifest.json', parent)
            self.parents[run.resolve()] = parent
            for row in self.rows:
                if row['arm'] == arm: row['source_manifest'] = pref
        self.uniform = {'schema': sources.REVIEW_SCHEMA, 'status': sources.REVIEW_STATUS, 'banks': self.rows}
        self.uniform_ref = self.put(self.root / 'uniform20.json', self.uniform)
        self.exp = SimpleNamespace(check_review=Mock(side_effect=lambda p: self.uniform),
                                   check_export_bank=Mock(side_effect=self.loaded_export))
        p = patch.object(sources, 'export_module', return_value=self.exp); p.start(); self.addCleanup(p.stop)
        p = patch.object(sources.legacy.writer, 'manifest', side_effect=lambda p: self.parents[Path(p).resolve()]); p.start(); self.addCleanup(p.stop)
        p = patch.object(sources.legacy.old, 'validate_resources', return_value=None); p.start(); self.addCleanup(p.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only synthetic test fixtures are rewritten; no experiment path used.
        path.write_text(json.dumps({**value, 'content_sha256': sources.digest(value)}, ensure_ascii=False), encoding='utf-8')
        return sources.pointer(path)

    def loaded_export(self, binding):
        self.assertEqual(binding, self.export_refs[3])
        row = next(r for r in self.rows if (r['conversation'], r['arm']) == (3, 'source_local_context'))
        bank, ref = self.bank_data[3, 'source_local_context']
        return {'bank': bank, 'native': Path(row['native_dir']), 'uniform_row': copy.deepcopy(row),
                'source_kind': 'reviewed_export', 'review': {'bank': ref, 'native_inventory': row['native_inventory']},
                'export': {'uniform_twenty_bank_review': self.uniform_ref}}

    def gate(self, arm='source_local_context'):
        rows = sources.assemble_rows(self.uniform_ref['path'], arm, self.export_refs if arm == 'source_local_context' else {})
        return {'schema': sources.GATE_SCHEMA, 'status': 'passed', 'arm': arm,
                'checks': {k: True for k in sources.CHECKS}, 'limitations': ['native limitations retained'],
                'content_quality_reviews': [self.evidence], 'uniform_twenty_bank_review': self.uniform_ref, 'banks': rows}

    def admit(self, value):
        return sources.admit_gate(self.put(self.root / 'gate.json', value)['path'])

    def test_strict_and_export_preserve_original_failed_and_copy_lineage(self):
        gate, admitted, _ = self.admit(self.gate())
        self.assertEqual(len(admitted), 10)
        source = admitted[3, 'source_local_context']
        self.assertEqual(source['review']['original_review']['original_write_status'], 'failed')
        self.assertFalse((self.root / 'source_local_context/conv_3/source_local_context/bank.json').exists())
        self.assertEqual(source['bank'], self.bank_data[3, 'source_local_context'][0])
        before = sources.inventory(source['native'])
        ref = sources.copy_bank(self.root / 'qa_copy', 3, 'source_local_context', source)
        copied = sources.verify(ref['path'])
        self.assertEqual(copied['facts'], source['bank']['facts'])
        self.assertEqual(copied['source_bank'], source['review']['bank'])
        self.assertEqual(copied['source_review']['original_review']['original_write_status'], 'failed')
        self.assertEqual(sources.inventory(source['native']), before)
        self.assertEqual(sources.inventory(Path(copied['native_dir'])), before)
        _, strict, _ = self.admit(self.gate('jev_refined'))
        self.assertEqual(len(strict), 10)

    def test_requires_uniform20_even_when_selected_arm_has_ten(self):
        gate = self.gate()
        self.uniform['banks'] = self.rows[:10]
        with self.assertRaisesRegex(ValueError, 'twenty'): self.admit(gate)
        self.uniform['banks'] = self.rows + [self.rows[0]]
        with self.assertRaisesRegex(ValueError, 'twenty'): self.admit(gate)

    def test_rejects_partial_unknown_and_unreviewed_error_classes(self):
        base = self.gate()
        for changes in ({'groups': 0}, {'groups': 1, 'expected_groups': 2}, {'facts': 19},
                        {'source_kind': 'partial'}, {'original_write_status': 'running'},
                        {'proof': None}, {'native_index_limitations': []}):
            original = copy.deepcopy(self.rows[3])
            self.rows[3].update(changes)
            try:
                with self.assertRaises(ValueError): self.admit(base)
            finally:
                self.rows[3].clear(); self.rows[3].update(original)

    def test_gate_may_not_drop_duplicate_or_change_kind_or_review_binding(self):
        base = self.gate()
        variants = []
        v = copy.deepcopy(base); v['banks'].pop(); variants.append(v)
        v = copy.deepcopy(base); v['banks'][-1] = v['banks'][0]; variants.append(v)
        v = copy.deepcopy(base); v['banks'][3]['source_kind'] = 'strict_complete'; variants.append(v)
        v = copy.deepcopy(base); v['banks'][3]['review_row_sha256'] = 'wrong'; variants.append(v)
        v = copy.deepcopy(base); v['banks'][3]['reviewed_export'] = None; variants.append(v)
        v = copy.deepcopy(base); v['content_quality_reviews'] = []; variants.append(v)
        v = copy.deepcopy(base); v['checks']['twenty_terminal_banks_reviewed'] = False; variants.append(v)
        for value in variants:
            with self.assertRaises(ValueError): self.admit(value)

    def test_export_wrong_review_or_bank_rejected(self):
        base = self.gate()
        original = self.loaded_export(self.export_refs[3])
        for changes in ({'uniform_row': {}}, {'export': {'uniform_twenty_bank_review': self.evidence}},
                        {'source_kind': 'strict_complete'}, {'bank': {'complete': False}}):
            self.exp.check_export_bank.side_effect = None
            self.exp.check_export_bank.return_value = {**original, **changes}
            with self.assertRaises(ValueError): self.admit(base)

    def test_original_status_native_or_journal_drift_rejected(self):
        gate = self.gate()
        row = self.rows[3]
        status_path = Path(row['write_status']['path'])
        data = status_path.read_bytes(); status_path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError): self.admit(gate)
        status_path.write_bytes(data)
        native = Path(row['native_dir']) / 'history.sqlite'
        data = native.read_bytes(); native.write_bytes(b'changed native')
        with self.assertRaises(ValueError): self.admit(gate)
        native.write_bytes(data)
        jp = Path(self.bank_data[3, 'source_local_context'][0]['journal_files'][0]['path'])
        jp.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError): self.admit(gate)

    def test_source_validator_is_authoritative_before_native_copy(self):
        gate = self.gate()
        self.exp.check_review.side_effect = ValueError('running full population')
        with patch.object(sources, 'copy_bank', side_effect=AssertionError('Must not copy')):
            with self.assertRaisesRegex(ValueError, 'running'): self.admit(gate)


if __name__ == '__main__':
    unittest.main()
