"""Offline complementary1140 wiring and frozen model behavior; no paid calls."""
import copy
import ast
import inspect
import importlib.util
import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import CodeType
from unittest.mock import patch
import unittest

from tools import run_regroup_refinement_extension1140_v1 as d
from tools import regroup_refinement_extension1140_sources_v1 as s
from tools import run_regroup_refinement_qa_v1 as original

_spec = importlib.util.spec_from_file_location('_extension_reused_qa_tests',
    Path(__file__).with_name('test_regroup_refinement_qa_v1.py'))
_reused = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reused)
_reused.d, _reused.s = d, s


class ReusedBehavior(_reused.RefinementQATests):
    def test_actual_fixed_cohort_is_400_with_all_conversations(self):
        rows, dates = s.cohort()
        full, full_dates = s.full_source.cohort()
        old, _ = s.legacy.cohort()
        new_ids = [q['id'] for qs in rows.values() for q in qs]
        old_ids = [q['id'] for qs in old.values() for q in qs]
        all_ids = [q['id'] for qs in full.values() for q in qs]
        self.assertEqual(len(new_ids), 1140)
        self.assertFalse(set(new_ids) & set(old_ids))
        self.assertEqual(set(new_ids) | set(old_ids), set(all_ids))
        self.assertEqual(dates, full_dates)
        self.assertEqual(set(rows), {str(ci) for ci in range(10)})
        self.assertEqual(new_ids, [qid for qid in all_ids if qid not in set(old_ids)])

    def test_no_json_and_reader_is_persisted_before_judge(self):
        j = _reused.job(); j['reference'] = 2022
        transport = _reused.Transport(['2022', '[[yes]]'])
        post0 = transport.post
        def post(url, **kwargs):
            if '3.5-flash-lite' in url:
                self.assertEqual(self.row(j, 'reader')['result']['metrics']['official_locomo_f1'], 1)
            return post0(url, **kwargs)
        transport.post = post
        with patch.object(d, 'execution_manifest', return_value=self.m([j])), patch('requests.Session', return_value=transport):
            out = d.qa(self.run, api_factory=self.factory)
        self.assertEqual(out['planned_per_arm'], 1140)
        self.assertEqual(out['counts']['qa'], {'complete': 1})
        for call in transport.calls:
            self.assertNotIn('responseMimeType', call['generationConfig'])
            self.assertNotIn('responseSchema', call['generationConfig'])
        self.assertEqual(json.loads(transport.calls[0]['contents'][0]['parts'][0]['text']),
            _reused.reader_input(j['question'], j['selected_facts'], reference_date=j['reference_date']))

    def test_prepare_is_model_free_single_arm_400_and_embedding_only(self):
        # Fail closed before a namespace/copy/client can be created when the
        # master baseline/config lock is not verified.
        with patch.object(d, '_verify_master_lock', side_effect=ValueError('changed lock')), \
             patch.object(d, '_base_prepare') as prepare, \
             patch.object(d, 'PlainTextEvaluationAPI', side_effect=AssertionError('API forbidden')):
            with self.assertRaisesRegex(ValueError, 'changed lock'):
                d.prepare(self.root/'runs/new', self.root/'gate.json')
            prepare.assert_not_called()


class ExtensionTests(unittest.TestCase):
    def test_old_functions_unchanged_and_native_evaluation_same_objects(self):
        self.assertIs(d.native_qa.retrieve_arm, original.native_qa.retrieve_arm)
        self.assertIs(d.plain.one, original.plain.one)
        self.assertIs(d.sources.copy_bank, original.sources.copy_bank)
        self.assertEqual(original.POLICY['slots_per_arm'], 400)
        self.assertEqual(d.POLICY['slots_per_arm'], 1140)
        for name in d.REBOUND_FUNCTIONS:
            if name in ('manifest', 'prepare') or name in d.CARDINALITY_FIELDS:
                continue
            new, old = getattr(d, name), getattr(original, name)
            self.assertIs(new.__code__, old.__code__)
            self.assertIs(new.__globals__, vars(d))
            self.assertIsNot(new.__globals__, old.__globals__)
            self.assertIs(pickle.loads(pickle.dumps(new)), new)

    def test_only_three_reporting_constants_change(self):
        def normalize(code):
            return (code.co_code, code.co_names,
                tuple(normalize(v) if isinstance(v, CodeType) else ('COUNT' if type(v) is int and v in (400,1140) else v)
                      for v in code.co_consts))
        for name, field in d.CARDINALITY_FIELDS.items():
            old = getattr(original, name)
            code = d._cardinality_code(old, field)
            unmodified = {}
            exec(compile(ast.parse(inspect.getsource(old)), '<isolated-original>', 'exec'), old.__globals__, unmodified)
            self.assertEqual(normalize(code), normalize(unmodified[name].__code__))
            self.assertIn(1140, code.co_consts)
        with self.assertRaises(ValueError):
            d._cardinality_code(original.qa, 'not_a_known_field')

    def test_real_spawn_resolves_rebound_retrieval_helpers(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertIsNone(pool.submit(d.check_job, _reused.job(), 'jev_refined').result(timeout=45))

    def test_selection_does_not_inspect_scores_and_rejects_bad_population(self):
        full, _ = s.full_source.cohort(); old, _ = s.legacy.cohort()
        baseline = s.split_cohort(full, old)
        polluted = copy.deepcopy(full)
        for rows in polluted.values():
            for q in rows:q.update(scores={'judge': False}, status='failed')
        selected = s.split_cohort(polluted, old)
        self.assertEqual({ci:[q['id'] for q in rows] for ci,rows in selected.items()},
                         {ci:[q['id'] for q in rows] for ci,rows in baseline.items()})
        invalid = copy.deepcopy(old); invalid['0'][0]['question'] += 'altered'
        with self.assertRaises(ValueError):s.split_cohort(full, invalid)
        invalid = copy.deepcopy(old); invalid['0'].append(invalid['0'][0])
        with self.assertRaises(ValueError):s.split_cohort(full, invalid)

    def test_manifest_requires_master_lock_before_live_work(self):
        with patch.object(d, '_verify_master_lock', side_effect=ValueError('changed lock')), \
             patch.object(d, '_base_manifest') as base:
            with self.assertRaisesRegex(ValueError, 'changed lock'):d.manifest('unused')
            base.assert_not_called()

    def test_inheritance_pointer_mutation_hard_fails(self):
        bad = {'schema':'refinement-inherited400-lineage-v1','arm':'jev_refined','references':{}}
        with self.assertRaises(ValueError):s.validate_inheritance(bad)


if __name__ == '__main__':
    unittest.main()

