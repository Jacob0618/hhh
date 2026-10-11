"""Offline module-stack cardinality, exact-request, recovery and replay tests."""
import copy
import marshal
from pathlib import Path
import tempfile
from types import FunctionType
import unittest
from unittest.mock import patch

from tests import test_evidence_exchange_qa_r01 as oldtests
from tools import run_evidence_exchange_qa_r01 as prior
from tools import run_evidence_selection_qa_r03 as proof_template
from tools import run_module_stack_qa_v1 as runner
from tools import summarize_module_stack_v1 as summary
from jev_mem.source_append_r05 import attachment


def job(method='baseline'):
    return oldtests.job({'cheap': 'pool_old_bm25', 'jev': 'stack_support'}.get(method, method))


# Reuse the original behavior assertions against new isolated functions. The
# old three-arm stopping-rule fixture is replaced with a nine-arm test below.
_scope = {**vars(oldtests), 'runner': runner, 'summary': summary, 'job': job}
_methods = {name: FunctionType(f.__code__, _scope, name, f.__defaults__, f.__closure__)
            for name, f in vars(oldtests.AdapterTests).items()
            if isinstance(f, FunctionType) and name != 'test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary'}
InheritedBehaviorTests = type('InheritedBehaviorTests', (unittest.TestCase,), _methods)


def cohort():
    seed = job()
    return [{**seed, 'method': arm, 'job_id': arm + '/' + f'c{n % 10}_q{n}',
             'query_id': f'c{n % 10}_q{n}', 'conversation': n % 10}
            for arm in runner.ARMS for n in range(runner.PER_ARM)]


def score_rows():
    rows = []
    for arm in runner.ARMS:
        correct = 1153 if arm == 'baseline' else 1199 if arm == 'pool_old_bm25' else 1200
        for n in range(runner.PER_ARM):
            value = 1 if n < correct else None if n == 1539 and arm != 'baseline' else 0
            rows.append({'method': arm, 'query_id': f'c{n % 10}_q{n}', 'conversation': n % 10,
                'category': 1, 'selection_status': 'complete', 'reader_status': 'complete',
                'judge_status': 'missing' if value is None else 'complete',
                'qa_status': 'failed' if value is None else 'complete',
                'scores': {'judge_correct': value, 'official_locomo_f1': value, 'custom_token_f1': value}})
    return rows


class IsolationTests(unittest.TestCase):
    def test_original_stage_recovery_run_and_replay_bytecode_is_reused(self):
        for name in ('one', 'recovery_jobs', 'prepare_recovery', 'terminal', 'reader_call',
                     'compatibility', 'resolve_pair', 'known_failure', 'run'):
            self.assertEqual(marshal.dumps(getattr(runner, name).__code__),
                             marshal.dumps(getattr(prior, name).__code__), name)
        self.assertEqual(marshal.dumps(runner._bound_manifest.__code__), marshal.dumps(prior.manifest.__code__))
        self.assertEqual(marshal.dumps(runner._revalidate_original_proof.__code__),
                         marshal.dumps(proof_template.build_reuse_proofs.__code__))
        for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
            self.assertEqual(marshal.dumps(getattr(summary, name).__code__),
                             marshal.dumps(getattr(summary.original, name).__code__), name)
        self.assertIs(runner.evaluator, prior.evaluator)
        self.assertIs(runner.Gemini38Replay, prior.Gemini38Replay)
        self.assertIs(runner._new_audit_stage, prior._new_audit_stage)

    def test_scopes_are_isolated_and_policy_is_frozen(self):
        self.assertIsNot(runner._impl, prior._impl)
        self.assertIs(runner.identity.__globals__, runner._impl)
        self.assertIs(runner.one.__globals__, runner._helpers)
        self.assertIs(runner._impl['manifest'], runner.manifest)
        self.assertIs(runner._helpers['source_validate'], runner.source_validate)
        self.assertIs(summary.audit_phase.__globals__['runner'], runner)
        self.assertIs(summary.audit_stage.__globals__['runner'], runner)
        self.assertEqual(prior.POLICY['firstpass_slots'], 4620)
        self.assertEqual(prior._impl['SCHEMA'], prior.SCHEMA)
        self.assertEqual(runner.POLICY['firstpass_slots'], 13860)
        self.assertEqual(runner.POLICY['cache_roots'], [])
        self.assertEqual(runner.POLICY['valid_yes_and_no'], 'never_correctness_retry')
        self.assertEqual(runner.POLICY['word_limit'], None)

    def test_exact_original_gemini38_low_8192_signature(self):
        for stage in ('reader', 'judge'):
            signature = runner.request_signature(job(), stage)
            self.assertEqual(signature, prior.request_signature(job(), stage))
            self.assertEqual(signature['model'], 'gemini-3.8-flash')
            config = signature['body']['generationConfig']
            self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'low'})
            self.assertEqual(config['maxOutputTokens'], 8192)
            self.assertNotIn('temperature', config)
        self.assertNotIn('15 words', str(runner.request_signature(job(), 'reader')).lower())

    def test_absent_archive_or_truncated_proof_cannot_be_inherited(self):
        with self.assertRaisesRegex(ValueError, 'Explicit prior'):
            runner.build_reuse_proofs({'archived_r08_summary': {'correct': 1200}})
        with patch.object(runner, 'bound', return_value={'input_sha256': {'one': 'h'}}):
            with self.assertRaisesRegex(ValueError, '12289'):
                runner.build_reuse_proofs({'frozen_baseline_reuse_proof': {'path': 'p'},
                                          'reuse_proof_qa_manifest': {'path': 'm'}})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'proof.json'
            p.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.build_reuse_proofs({'frozen_baseline_reuse_proof': {'path': str(p), 'sha256': 'wrong'},
                                          'reuse_proof_qa_manifest': {'path': 'm'}})

    def test_proof_template_retains_original_historical_cardinalities(self):
        # 4620 belongs to the genuine R01 producer, not the nine-arm consumer.
        constants = runner._revalidate_original_proof.__code__.co_consts
        self.assertIn(4620, constants)
        self.assertIn(10780, constants)
        self.assertNotIn(13860, constants)
        self.assertIs(runner._proof_scope['prior'], prior)


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = cohort()

    def test_nine_full_arms_and_prepare_are_offline(self):
        runner.check_full_cohort(self.jobs)
        with patch.object(runner, 'source_validate', return_value={'jobs': self.jobs}), \
             patch.object(runner, '_new_manifest', side_effect=lambda *args, **kw: {'jobs': len(kw['jobs']), 'model_calls': 0}):
            self.assertEqual(runner.prepare('unused-run', 'unused-source'), {'jobs': 13860, 'model_calls': 0})

    def test_missing_duplicate_and_foreign_cohort_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nine full'):
            runner.check_full_cohort(self.jobs[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate job'):
            runner.check_full_cohort([*self.jobs[:-1], self.jobs[-2]])
        altered = list(self.jobs)
        altered[-1] = {**altered[-1], 'query_id': 'c9_q9999', 'job_id': runner.ARMS[-1] + '/c9_q9999'}
        with self.assertRaisesRegex(ValueError, 'identical fixed'):
            runner.check_full_cohort(altered)
        altered[-1] = {**self.jobs[-1], 'question': 'Different original query?'}
        with self.assertRaisesRegex(ValueError, 'Cross-arm'):
            runner.check_full_cohort(altered)

    def test_stack_five_turn_packet_allowed_six_rejected(self):
        for count, valid in ((5, True), (6, False)):
            packet = {'session_id': 's', 'source_observation_date': '2023-01-01',
                'turns': [{'source_turn_id': str(i), 'speaker_text': f'A: text {i}', 'machine_captions': []}
                          for i in range(count)]}
            jobs = list(self.jobs)
            index = 8 * runner.PER_ARM
            changed = copy.deepcopy(jobs[index])
            changed['selected_facts'][0]['text'] += attachment(packet)
            jobs[index] = changed
            if valid:
                result = summary.augmentation_audit({'jobs': jobs}, {})
                self.assertEqual(result['attachment_count_distribution'][runner.ARMS[-1]], {0: 1539, 1: 1})
                self.assertEqual(result['per_job'][index]['attachments'][0]['source_turn_count'], 5)
            else:
                with self.assertRaisesRegex(ValueError, 'five source turns'):
                    summary.augmentation_audit({'jobs': jobs}, {})

    def test_stack_uses_pool_new_jev_origin_and_checks_prefix(self):
        jobs = list(self.jobs)
        origin_index = 4 * runner.PER_ARM
        source = copy.deepcopy(jobs[origin_index])
        source['selected_facts'][0]['text'] = 'Changed R08 memory'
        jobs[origin_index] = source
        # The stack still equals baseline, so this must fail against R08.
        with self.assertRaisesRegex(ValueError, 'R08 JEV ID/date/text prefix'):
            summary.augmentation_audit({'jobs': jobs}, {})


class SummaryTests(unittest.TestCase):
    def test_fixed_denominator_retains_missing_and_strict_threshold(self):
        rows = score_rows()
        result = summary.stopping_result(rows)
        self.assertFalse(result['strict_gt_3pp_fixed_denominator']['pool_old_bm25'])
        self.assertTrue(result['strict_gt_3pp_fixed_denominator']['stack_support'])
        self.assertEqual(result['missing']['stack_support'], 1)
        report = summary.aggregate(rows, bootstrap_samples=1)
        metric = report['arms']['stack_support']['metrics']['judge_correct']
        self.assertEqual(metric['planned_denominator'], 1540)
        self.assertEqual(metric['valid_denominator'], 1539)
        self.assertEqual(metric['complete_only_mean'], 1200 / 1539)
        self.assertIsNone(metric['full_cohort_observed_mean'])
        self.assertEqual(metric['zero_imputed_sensitivity']['mean'], 1200 / 1540)
        self.assertEqual(len(report['paired_comparisons']), 8)
        contrasts = summary.stack_contrasts(rows, bootstrap_samples=1)
        self.assertIn('stack_support_closure_minus_stack_source', contrasts)
        with self.assertRaisesRegex(ValueError, 'Full planned denominator'):
            summary.aggregate(rows[:-1], bootstrap_samples=1)
        rows[0]['scores']['judge_correct'] = None
        self.assertFalse(summary.stopping_result(rows)['threshold_eligible'])

    def test_valid_no_has_no_recovery_and_missing_stage_is_rejected(self):
        case = InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            j = job('stack_support')
            case.plan(j)
            runner.one(case.run, j, case.m, None, oldtests.official)
            paths = runner.stage_files(case.run, j)
            row = {**{k: j[k] for k in ('conversation', 'method', 'query_id', 'job_id')},
                   **{s: runner.pointer(p) for s, p in paths.items()}}
            out = {'outcomes': [], 'results': [row]}
            parent = {**case.m, 'jobs': [j]}
            self.assertEqual(runner.recovery_jobs(parent, out), [])
            row['qa'] = None
            with self.assertRaisesRegex(ValueError, 'Missing/unknown'):
                runner.recovery_jobs(parent, out)
        finally:
            case.doCleanups()


if __name__ == '__main__': unittest.main()
