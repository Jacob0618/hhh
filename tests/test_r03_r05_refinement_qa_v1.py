"""Offline adapter semantics, reference inheritance, and fixed denominators."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tools

from tools import run_r03_r05_refinement_qa_v1 as qa
from tools import summarize_r03_r05_refinement_v1 as summary
from tests.test_r03_r05_reuse_proofs_v1 import RawFixture, job, write


def cohort():
    facts = job()['selected_facts']
    out = []
    for arm in qa.ARMS:
        for i in range(1540):
            ci = i % 10; qid = f'c{ci}_q{i // 10}'
            out.append({**job(arm, qid), 'conversation': ci,
                        'selected_facts': facts, 'bank': {'path': 'bank', 'sha256': 'b'},
                        'selection': {'path': 'selected', 'sha256': 's'}})
    return out


class DriverTests(unittest.TestCase):
    def test_frozen_evaluator_and_durable_stage_bytecode_are_reused(self):
        for name in ('run', 'terminal', 'recovery_jobs', 'prepare_recovery', 'known_failure', 'reader_call',
                     'compatibility', 'resolve_pair', 'stage_files', 'identity'):
            self.assertIs(getattr(qa, name).__code__, getattr(qa.prior, name).__code__)
            self.assertIsNot(getattr(qa, name).__globals__, getattr(qa.prior, name).__globals__)
        self.assertIs(qa._impl['stage_once'].__code__, qa.prior._impl['stage_once'].__code__)
        self.assertIs(qa.evaluator, qa.prior.evaluator)
        self.assertEqual(qa.prior.SCHEMA, 'module-stack-plain-qa-run-v1')
        self.assertEqual(qa.POLICY['firstpass_slots'], 10780)
        self.assertEqual(qa.prior.POLICY['firstpass_slots'], 13860)

    def test_seven_full_cohort_missing_duplicate_or_changed_query_fails(self):
        jobs = cohort(); qa.check_full_cohort(jobs)
        with self.assertRaisesRegex(ValueError, 'Seven full'):
            qa.check_full_cohort(jobs[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate job'):
            qa.check_full_cohort(jobs[:-1] + [jobs[-2]])
        changed = deepcopy(jobs[-1]); changed['reference'] = 'Different gold'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            qa.check_full_cohort(jobs[:-1] + [changed])

    def test_reference_reader_and_no_judge_inherit_without_stage_or_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(Path(d) / 'producer')
            p = f.mini_proof(); p['input_sha256'] = {}
            run = Path(d) / 'consumer'; ref = write(run / 'proof.json', p, seal=True)
            m = {**f.m, 'schema': qa.SCHEMA, 'jobs': [f.j], 'prior_qa_reuse_proof': ref,
                 'reuse_plans': qa._plans([f.j], p, 'firstpass')}
            write(run / 'run_manifest.json', m)
            with patch.object(qa, 'proof_for', return_value=p), \
                 patch.dict(qa._impl, {'stage_once': lambda *a, **k: self.fail('Reference dispatch forbidden')}):
                qa.one(run, f.j, m, None, f.official)
            paths = qa.stage_files(run, f.j)
            reader = qa.verify(paths['reader']); judge = qa.verify(paths['judge'])
            self.assertEqual(reader['model_calls'], 0); self.assertEqual(judge['model_calls'], 0)
            self.assertFalse(judge['result']['correct'])
            self.assertFalse(paths['reader'].with_suffix('.intent.json').exists())
            self.assertFalse(paths['judge'].with_suffix('.intent.json').exists())
            row = {**{k: f.j[k] for k in ('conversation', 'method', 'query_id', 'job_id')},
                   **{k: qa.pointer(v) for k, v in paths.items()}}
            self.assertEqual(qa.recovery_jobs(m, {'results': [row], 'outcomes': []}), [])
            snap = summary.original.Snapshot()
            actual = summary.audit_stage(snap, paths['judge'],
                {**qa.identity(f.j, m), 'stage': 'judge', 'reader_sha256': qa.sha(paths['reader'])},
                run / 'conv_0/api', qa.stage_context(f.j, m, 'judge'), {},
                lambda api: qa.evaluator.judge_answer(api, f.j['question'], f.j['reference'], 'Paris'))
            self.assertFalse(actual['result']['correct'])

    def test_reference_miss_fails_before_initializing_or_dispatching(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(d); p = f.mini_proof(); p['entries'] = {}; p['reader_matches'][f.j['job_id']] = None
            with self.assertRaisesRegex(ValueError, 'Reference arms require'):
                qa._plans([f.j], p, 'firstpass')

    def test_prepare_uses_full_new_source_without_dispatch(self):
        jobs = cohort()
        with patch.object(qa, 'source_validate', return_value={'jobs': jobs}), \
             patch.object(qa, '_new_manifest', side_effect=lambda *a, **k: {'n': len(k['jobs']), 'model_calls': 0}):
            self.assertEqual(qa.prepare('unused', 'unused'), {'n': 10780, 'model_calls': 0})

    def test_manifest_rejects_changed_disk_proof_after_lru_hit(self):
        with tempfile.TemporaryDirectory() as d:
            f = RawFixture(Path(d) / 'old'); p = f.mini_proof()
            path = Path(d) / 'proof.json'; ref = write(path, p, seal=True)
            m = {'prior_qa_reuse_proof': ref, 'source_manifest': {'path': 'src'},
                 'phase_kind': 'firstpass', 'jobs': [f.j], 'reuse_plans': {}}
            qa.proof_for(m)  # Warm the LRU with the original bytes.
            p['model_calls_during_proof'] = 99; write(path, p, seal=True)
            with patch.object(qa, '_ordinary_manifest', return_value=m), \
                 patch.object(qa, 'source_validate', return_value={'jobs': [f.j]}), \
                 patch.object(qa.reuse, 'revalidate', side_effect=AssertionError('must reject disk change first')):
                with self.assertRaisesRegex(ValueError, 'Source pointer changed'):
                    qa.manifest('unused')


class SummaryTests(unittest.TestCase):
    def test_missing_scores_keep_fixed1540_and_distinct_valid_denominator(self):
        rows = []
        for j in cohort():
            missing = j['method'] == 'r03_refined' and j['query_id'] == 'c0_q0'
            rows.append({k: j[k] for k in ('method', 'query_id', 'conversation', 'category')} |
                {'selection_status': 'complete', 'reader_status': 'failed' if missing else 'complete',
                 'judge_status': 'missing' if missing else 'complete', 'qa_status': 'failed' if missing else 'complete',
                 'scores': {n: None if missing else 1.0 for n in summary.original.METRICS}})
        result = summary.aggregate(rows, bootstrap_samples=2)['arms']['r03_refined']['metrics']['judge_correct']
        self.assertEqual(result['planned_denominator'], 1540)
        self.assertEqual(result['valid_denominator'], 1539)
        self.assertEqual(result['complete_only_mean'], 1)
        self.assertEqual(result['zero_imputed_sensitivity']['mean'], 1539 / 1540)
        with self.assertRaisesRegex(ValueError, 'Full planned denominator'):
            summary.aggregate(rows[:-1], bootstrap_samples=2)

    def test_factorial_signed_difference_and_four_arm_missingness(self):
        names = ('original_source_closure', 'refined_source_closure',
                 'original_multi_source_closure', 'refined_multi_source_closure')
        rows = []
        for arm, score in zip(names, (1.0, 0.0, 0.0, 1.0)):
            for i in range(20):
                rows.append({'method': arm, 'query_id': f'q{i}', 'conversation': i % 10,
                             'scores': {n: score for n in summary.original.METRICS}})
        out = summary.factorial_interaction(rows, bootstrap_samples=2)['metrics']['judge_correct']
        self.assertEqual(out['common_valid_four_arm_count'], 20)
        self.assertEqual(out['contrast']['metrics']['judge_correct']['other_minus_baseline'], 2.0)
        self.assertEqual(out['contrast']['metrics']['judge_correct']['cluster_bootstrap_95_percentile_interval'], [2.0, 2.0])
        rows[-1]['scores']['judge_correct'] = None
        out = summary.factorial_interaction(rows, bootstrap_samples=2)['metrics']['judge_correct']
        self.assertEqual(out['common_valid_four_arm_count'], 19)
        self.assertFalse(out['contrast']['bootstrap']['eligible'])
        self.assertEqual(out['planned_denominator'], 1540)

    def test_source_prefixes_bind_original_or_refined_not_previous_newpool(self):
        jobs = cohort()
        ix = next(i for i, j in enumerate(jobs) if j['method'] == 'refined_multi_source_closure')
        jobs[ix] = deepcopy(jobs[ix]); jobs[ix]['selected_facts'][0]['text'] += '\n[source data]'
        result = summary.selected_input_audit({'jobs': jobs})
        self.assertEqual(result['changed_slots_by_arm']['refined_multi_source_closure'], {0: 1539, 1: 1})
        jobs[ix]['selected_facts'][0]['text'] = 'Changed original prefix'
        with self.assertRaisesRegex(ValueError, 'text prefix'):
            summary.selected_input_audit({'jobs': jobs})

    def test_bad_raw_audit_fails_before_aggregation(self):
        source = {'r03_r05_protocol': {}}
        phase = {'manifest': {'phase_kind': 'firstpass'}, 'run': 'bad', 'errors': [{'message': 'bad raw'}]}
        with patch.object(summary, 'read', return_value={'source_manifest': {'path': 'src'}}), \
             patch.object(summary.runner, 'source_validate', return_value=source), \
             patch.object(summary, 'bound', return_value={}), patch.object(summary, 'audit_phase', return_value=phase), \
             patch.object(summary, 'aggregate', side_effect=AssertionError('must not aggregate')):
            with self.assertRaisesRegex(ValueError, 'Raw audit failed'):
                summary.summarize('run', bootstrap_samples=2)

    def test_source_cost_excludes_historical_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); run = root / 'runs/r03_r05_refinement_test'
            def ledger(folder, amount):
                path = folder / 'api_calls.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'status': 200, 'model': 'jev-1.13.0', 'usage': {'input_tokens': amount}}) + '\n', encoding='utf-8')
            ledger(run / 'phases/precision/firstpass', 2)
            ledger(root / 'runs/module_stack_old/phases/precision/firstpass', 1000)
            source = {'parent_run': str(run), 'selector_manifest': {'path': str(run / 'run_manifest.json')}}
            prep = SimpleNamespace(SCHEMA='fixture', PHASES=('precision',),
                Reader=lambda: SimpleNamespace(bound=lambda ref: {'schema': 'fixture'}),
                phase_results=lambda reader, parent, phase: {'r': {'status': 'complete'}})
            with patch.dict('sys.modules', {'tools.prepare_r03_r05_refinement_v1': prep}), \
                 patch.object(tools, 'prepare_r03_r05_refinement_v1', prep, create=True), patch.object(summary, 'ROOT', root):
                value = summary.source_usage(source)
            self.assertEqual(value['physical_usage']['token_fields']['prompt']['recorded_token_sum'], 2)
            self.assertFalse(value['historical_source_ledgers_included'])

    def test_summary_keeps_original_audit_aggregate_bytecode_and_offline_guard(self):
        self.assertIs(summary.audit_phase.__code__, summary.prior.audit_phase.__code__)
        self.assertIs(summary.aggregate.__code__, summary.prior.aggregate.__code__)
        self.assertIs(summary.offline, summary.original.historical_guard.historical.offline)
        self.assertEqual(len(summary.CONTRASTS), 8)


if __name__ == '__main__':
    unittest.main()
