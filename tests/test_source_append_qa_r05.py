"""Offline R05 tests: original stage regressions plus derived-text contracts."""
import copy
import importlib.util
import json
from pathlib import Path
from types import FunctionType
import tempfile
import unittest

from tools import run_source_append_qa_r05 as runner
from tools import summarize_source_append_r05 as summary
from tools import run_evidence_exchange_qa_r01 as prior
from tools import run_evidence_selection_qa_r03 as template

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r05_original_adapter_tests', ROOT/'tests/test_evidence_exchange_qa_r01.py')
oldtests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldtests)
scope = {**vars(oldtests), 'runner': runner, 'summary': summary}
methods = {name: FunctionType(f.__code__, scope, name, f.__defaults__, f.__closure__)
    for name, f in vars(oldtests.AdapterTests).items() if isinstance(f, FunctionType)
    and name != 'test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary'}
R05InheritedBehaviorTests = type('R05InheritedBehaviorTests', (unittest.TestCase,), methods)


def packet_text(protocol, text='Caroline: I enjoyed it.'):
    packet = {'session_id': 1, 'source_observation_date': '2023-01-01',
        'turns': [{'source_turn_id': 'D1:1', 'speaker_text': text, 'machine_captions': []}]}
    rendered = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return protocol['reader_representation']['wrapper'].replace('{rendered_packet}', rendered)


class R05Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT/'research/jev_autoresearch_2026-10-04/R05_source_append_protocol_final_v1.json').read_text(encoding='utf-8'))

    def cohort(self):
        return {'jobs': [{**oldtests.job(arm), 'query_id': f'c0_q{i}', 'job_id': f'{arm}/c0_q{i}'}
                         for arm in runner.ARMS for i in range(1540)]}

    def test_isolated_namespaces_original_policy_and_request_capture(self):
        self.assertIsNot(runner._impl, prior._impl)
        self.assertIsNot(runner._impl, template._impl)
        self.assertIs(runner.identity.__globals__, runner._impl)
        self.assertIs(runner.one.__globals__, runner._helpers)
        self.assertIs(summary.audit_stage.__globals__['runner'], runner)
        self.assertIs(summary.audit_phase.__globals__['runner'], runner)
        self.assertEqual(prior._impl['SCHEMA'], prior.SCHEMA)
        self.assertEqual(template._impl['SCHEMA'], template.SCHEMA)
        self.assertEqual(runner.request_signature(oldtests.job(), 'reader'), prior.request_signature(oldtests.job(), 'reader'))
        self.assertEqual(runner.POLICY['conversation_workers'], 8)
        self.assertEqual(runner.POLICY['qa_threads_per_conversation'], 4)

    def test_same_ids_dates_and_prefix_append_requires_reader_but_exact_answer_inherits_no(self):
        case = R05InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            job = oldtests.job('jev')
            original = copy.deepcopy(job)
            job['selected_facts'][0]['text'] += packet_text(self.protocol)
            self.assertEqual(job['selected_ids'], original['selected_ids'])
            self.assertEqual(job['selected_facts'][0]['observation_date'], original['selected_facts'][0]['observation_date'])
            self.assertNotEqual(runner.request_signature(job, 'reader'), runner.request_signature(original, 'reader'))
            case.plan(job)
            self.assertIsNone(case.m['reuse_plans'][job['job_id']]['reader'])
            api = oldtests.MockAPI(case.run, job, label='[[yes]]')
            runner.one(case.run, job, case.m, api, oldtests.official)
            self.assertEqual(api.calls, ['reader'])
            judge = runner.verify(runner.stage_files(case.run, job)['judge'])
            self.assertFalse(judge['result']['correct'])
            self.assertEqual(judge['reader_sha256'], runner.sha(runner.stage_files(case.run, job)['reader']))
        finally: case.doCleanups()

    def test_missing_or_wrong_prior_proof_cannot_use_summary_counts(self):
        with self.assertRaisesRegex(ValueError, 'Explicit prior'):
            runner.build_reuse_proofs({'correct': 1153})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'proof.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                runner.build_reuse_proofs({'frozen_baseline_reuse_proof': {'path': str(path), 'sha256': 'wrong'},
                    'reuse_proof_qa_manifest': {'path': 'unused', 'sha256': 'unused'}})

    def test_actual_lf_packet_preserves_original20_and_bounded_slots(self):
        source = self.cohort()
        suffix = packet_text(self.protocol)
        self.assertIn('\n', suffix)
        self.assertNotIn('\\n', suffix)
        source['jobs'][1540]['selected_facts'][0]['text'] += suffix
        result = summary.augmentation_audit(source, self.protocol)
        self.assertEqual(result['attachment_count_distribution']['cheap'], {0: 1539, 1: 1})
        self.assertEqual(result['attachment_count_distribution']['baseline'], {0: 1540})
        self.assertEqual(result['cheap_jev_same_derived_reader_facts'], 1539)
        self.assertEqual(result['added_utf8_bytes_by_arm']['cheap'], len(suffix.encode('utf-8')))
        self.assertIsNone(result['measured_tokens_from_bytes'])

    def test_packet_prefix_date_order_and_full_cap_violations_rejected(self):
        for field in ('prefix', 'date', 'order', 'cap', 'three_slots', 'noncanonical'):
            with self.subTest(field=field):
                source = self.cohort()
                job = source['jobs'][1540]
                suffix = packet_text(self.protocol)
                if field == 'prefix': job['selected_facts'][0]['text'] = 'Replacement' + suffix
                elif field == 'date': job['selected_facts'][0]['observation_date'] = '2023-01-02'
                elif field == 'order': job['selected_ids'].reverse(); job['selected_facts'].reverse()
                elif field == 'cap': job['selected_facts'][0]['text'] += packet_text(self.protocol, '\u753b'*1400)
                elif field == 'three_slots':
                    for fact in job['selected_facts'][:3]: fact['text'] += suffix
                else: job['selected_facts'][0]['text'] += suffix.replace('"session_id":1', '"session_id": 1')
                with self.assertRaises(ValueError): summary.augmentation_audit(source, self.protocol)

    def test_baseline_cannot_silently_take_source_suffix(self):
        source = self.cohort()
        source['jobs'][0]['selected_facts'][0]['text'] += packet_text(self.protocol)
        with self.assertRaises(ValueError): summary.augmentation_audit(source, self.protocol)

    def test_fixed_stop_anchor_missing_and_component_qualification(self):
        rows = [{'method': arm, 'scores': {'judge_correct': int(i < correct)}}
                for arm, correct in (('baseline',1153),('cheap',1199),('jev',1200)) for i in range(1540)]
        result = summary.stopping_result(rows)
        self.assertTrue(result['strict_gt_3pp_fixed_denominator']['jev'])
        self.assertFalse(result['strict_gt_3pp_fixed_denominator']['cheap'])
        self.assertEqual(result['successful_jev_components_counted_here'], 1)
        self.assertNotIn('independent_component_count', result)
        rows[0]['scores']['judge_correct'] = None
        invalid = summary.stopping_result(rows)
        self.assertEqual(invalid['minimum_strict_gt_3pp'], 1200)
        self.assertFalse(invalid['threshold_eligible'])
        self.assertEqual(invalid['successful_jev_components_counted_here'], 0)


if __name__ == '__main__': unittest.main()
