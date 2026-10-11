"""Offline regressions for the frozen full1540 union and independent denominators."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import summarize_regroup_refinement_full1540_v1 as s


class Full1540Tests(unittest.TestCase):
    def setUp(self):
        self.expected = {f'c{i%10}_q{i}': {'query_id': f'c{i%10}_q{i}',
            'conversation': i%10, 'category': i%4+1, 'question': f'Question{i}',
            'reference': f'Answer{i}', 'reference_date': '2023-01-01'} for i in range(s.N)}
        self.rows = [{'query_id': qid, 'conversation': q['conversation'],
            'category': q['category'], 'method': 'jev_refined', 'selection_status': 'complete',
            'reader_status': 'complete', 'judge_status': 'complete', 'qa_status': 'complete',
            'scores': {'official_locomo_f1': .5, 'custom_token_f1': .4, 'judge_correct': 1.},
            'errors': []} for qid, q in self.expected.items()]

    def parts(self):
        return tuple({'slots_by_view': {v: copy.deepcopy(rows) for v in s.VIEWS}}
                     for rows in (self.rows[:400], self.rows[400:]))

    def test_union_scores_individual_slots_not_equal_weight_subset_means(self):
        first, extension = self.parts()
        for view in s.VIEWS:
            for r in first['slots_by_view'][view]: r['scores']['official_locomo_f1'] = 1.
            for r in extension['slots_by_view'][view]: r['scores']['official_locomo_f1'] = 0.
        joined = s.join_views(first, extension, 'jev_refined', self.expected)
        self.assertEqual(len(joined['final']), 1540)
        value = s.aggregate(joined['final'], 'jev_refined')['metrics']['official_locomo_f1']
        self.assertEqual(value['planned_denominator'], 1540)
        self.assertAlmostEqual(value['complete_only_mean'], 400/1540)

    def test_overlap_missing_and_duplicate_are_rejected(self):
        for change in ('overlap', 'missing', 'duplicate'):
            first, extension = self.parts()
            rows = extension['slots_by_view']['final']
            if change == 'overlap': rows[0] = copy.deepcopy(self.rows[0])
            elif change == 'missing': rows.pop()
            else: rows[0] = copy.deepcopy(rows[1])
            with self.assertRaises(s.InvalidArtifact):
                s.join_views(first, extension, 'jev_refined', self.expected)

    def test_missing_judge_retains_reader_and_full_planned_denominator(self):
        first, extension = self.parts()
        for view in s.VIEWS:
            row = extension['slots_by_view'][view][0]
            row['scores']['judge_correct'] = None
            row['judge_status'] = row['qa_status'] = 'failed'
        joined = s.join_views(first, extension, 'jev_refined', self.expected)
        metrics = s.aggregate(joined['final'], 'jev_refined')['metrics']
        self.assertEqual(metrics['official_locomo_f1']['valid_denominator'], 1540)
        self.assertEqual(metrics['judge_correct']['valid_denominator'], 1539)
        self.assertIsNone(metrics['judge_correct']['full_cohort_observed_mean'])
        self.assertEqual(metrics['judge_correct']['zero_imputed_sensitivity']['imputed_slots'], 1)

    def test_strict_compatible_final_views_do_not_overwrite_one_another(self):
        first, extension = self.parts()
        first['slots_by_view']['strict_firstpass'][0]['scores']['judge_correct'] = None
        first['slots_by_view']['final'][0]['scores']['judge_correct'] = 0.
        before = copy.deepcopy(first)
        joined = s.join_views(first, extension, 'jev_refined', self.expected)
        self.assertIsNone(joined['strict_firstpass'][0]['scores']['judge_correct'])
        self.assertEqual(joined['compatible_firstpass'][0]['scores']['judge_correct'], 1.)
        self.assertEqual(joined['final'][0]['scores']['judge_correct'], 0.)
        self.assertEqual(first, before)

    def test_category_method_and_metrics_cannot_be_changed(self):
        for field, value in [('category', 5), ('conversation', 999), ('method', 'other')]:
            rows = copy.deepcopy(self.rows); rows[0][field] = value
            with self.assertRaises(s.InvalidArtifact): s.validate_slots(rows, 'jev_refined', self.expected)
        rows = copy.deepcopy(self.rows); rows[0]['scores']['other_metric'] = 1.
        with self.assertRaises(s.InvalidArtifact): s.validate_slots(rows, 'jev_refined', self.expected)

    def test_judge_without_reader_and_nonfinite_score_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]['scores']['official_locomo_f1'] = None
        with self.assertRaises(s.InvalidArtifact): s.validate_slots(rows, 'jev_refined', self.expected)
        rows = copy.deepcopy(self.rows); rows[0]['scores']['official_locomo_f1'] = float('nan')
        with self.assertRaises(ValueError): s.aggregate(rows, 'jev_refined')

    def test_question_gold_date_binding_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'source_jobs.json'
            jobs = [{**q, 'method': 'jev_refined'} for q in self.expected.values()]
            s.save_new(path, {'jobs': jobs})
            report = {'source_manifest': s.old.plain.pointer(path)}
            s.validate_jobs(report, self.expected, 'jev_refined')
            changed = copy.deepcopy(self.expected)
            changed[next(iter(changed))]['reference_date'] = '2099-01-01'
            with self.assertRaises(s.InvalidArtifact): s.validate_jobs(report, changed, 'jev_refined')

    def test_summary_cannot_replace_raw_replay_scores(self):
        good = {'slots_by_view': {'final': self.rows}, 'audit_status': 'no_invalid_artifacts',
                'audit_errors': [], 'phase_sources': [], 'source_manifest': {}}
        bad = copy.deepcopy(good)
        bad['slots_by_view']['final'][0]['scores']['judge_correct'] = 0.
        with self.assertRaises(s.InvalidArtifact): s.compare_replayed(bad, good)

    def test_scoped_aggregate_leaves_original400_code_globals_unchanged(self):
        original_n = s.old.DENOMINATOR
        original_arms = s.old.ARMS
        s.aggregate(self.rows, 'jev_refined')
        self.assertEqual(s.old.DENOMINATOR, original_n)
        self.assertEqual(s.old.ARMS, original_arms)

    def test_driver_rejects_wrong_schema_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'run_manifest.json').write_text(json.dumps({'schema':'wrong'}),encoding='utf-8')
            with patch.object(s.importlib, 'import_module', side_effect=AssertionError('Must not import')):
                with self.assertRaises(s.InvalidArtifact): s.extension_driver(path)

    def test_extension_shell_reuses_old_aggregation_with1140_and_no_module_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            jobs = [{**q, 'job_id': 'jev_refined/'+qid, 'method': 'jev_refined'}
                    for qid, q in list(self.expected.items())[400:]]
            source = s.save_new(run/'source_jobs.json', {'jobs': jobs})
            m = {'schema': s.EXTENSION_SCHEMA, 'phase_kind': 'firstpass', 'arm': 'jev_refined',
                 'source_manifest': source, 'gate': {'path':'not-read', 'sha256':'not-read'}}
            (run/'run_manifest.json').write_text(json.dumps(m), encoding='utf-8')
            s.save_new(run/'run_outcome.json', {'counts': {}})
            records = {j['job_id']: {'reader': {'status': 'complete', 'result': {
                'metrics': {'official_locomo_f1': .5, 'custom_token_f1': .4},
                'prediction': {'answer': 'Answer'}}}, 'judge': {'status': 'complete',
                'result': {'correct': True, 'label': '[[yes]]'}}, 'compat': None, 'errors': []} for j in jobs}
            phase = {'run': str(run), 'manifest': m, 'outcome': {'counts': {}},
                'records': records, 'errors': [], 'ledger': [], 'ledger_refs': [],
                'usage': s.old.measure_usage([]), 'inputs': {}}
            account = {'retrieval_records': [], 'combined_lineage': {
                'scope_complete': True, 'new_namespace_only': s.old.measure_usage([])}}
            lock = {'dataset': {'questions': list(self.expected.values())}}
            cohort = {str(ci): [{'id': j['query_id']} for j in jobs if j['conversation'] == ci]
                      for ci in range(10)}
            fake_driver = SimpleNamespace(sources=SimpleNamespace(cohort=lambda: (cohort, {})))
            original = s.old.DENOMINATOR
            # Regular functions retain closures when the adapter creates private
            # function globals. Mock objects intentionally do not stand in for code.
            with patch.object(s.locktool, 'verify_lock', return_value=lock), \
                    patch.object(s, 'extension_driver', return_value=fake_driver), \
                    patch.object(s.old, 'audit_phase', lambda *_a, **_k: phase), \
                    patch.object(s.old, 'audit_retrieval_and_usage', lambda *_a, **_k: account), \
                    patch.object(s.locktool, 'LOCK', run/'source_jobs.json'):
                result = s.summarize_extension(run)
            self.assertEqual(result['planned_per_arm'], 1140)
            self.assertEqual(result['schema'], 'regroup-refinement-extension1140-qa-summary-v1')
            self.assertEqual(result['views']['final']['arms']['jev_refined']['metrics']
                             ['judge_correct']['valid_denominator'], 1140)
            self.assertEqual(s.old.DENOMINATOR, original)

    def test_pair_keeps_shared_valid_ids_and_fixed1540(self):
        left = copy.deepcopy(self.rows); right = copy.deepcopy(self.rows)
        right[0]['scores']['judge_correct'] = None
        right[1]['scores']['judge_correct'] = 0.
        result = s.pair(left, right, bootstrap_samples=2)['judge_correct']
        self.assertEqual(result['common_success_count'], 1539)
        self.assertEqual(result['independent_valid_denominators'],
                         {'baseline':1540, 'other':1539, 'planned_each':1540})
        self.assertEqual(result['wins_ties_losses']['losses'], 1)


if __name__ == '__main__': unittest.main()
