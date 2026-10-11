"""Offline adapter tests: actual prompts/parsers, mocked physical dispatch only."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_evidence_exchange_qa_r01 as runner
from tools import summarize_evidence_exchange_r01 as summary
from tools import run_gemini38_plain_qa_tag_overlay_v1 as frozen
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import pointer, verify
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.summarize_sdk20_modules import InvalidArtifact


def job(method='baseline'):
    facts = [{'id': f'f{i}', 'text': f'Item {i} is in Paris.', 'observation_date': '2023-01-01'} for i in range(20)]
    return {'job_id': method + '/c0_q0', 'method': method, 'status': 'ready', 'conversation': 0,
            'query_id': 'c0_q0', 'question': 'Where is Item 0?', 'reference': 'Paris', 'category': 1,
            'reference_date': '2023-01-02', 'selected_ids': [f['id'] for f in facts],
            'selected_facts': facts, 'bank': {'path': 'mock-bank', 'sha256': 'mock'},
            'selection': {'path': 'mock-selection', 'sha256': 'mock'}}


def official(answer, reference, category):
    return float(answer == reference)


class MockAPI(runner.SignatureCapture):
    def __init__(self, run, j, label='[[no]]', reader_error=False, reader_text='Paris'):
        super().__init__(reader_text)
        self.run, self.j, self.label, self.reader_error = run, j, label, reader_error
        self.events = []; self.component = None; self.calls = []

    def scoped(self, **context):
        child = copy.copy(self)
        child.events = []; child.component = context['component']
        return child

    def _redact(self, value):
        return str(value)

    def request(self, provider, model, body, **kwargs):
        self.calls.append(self.component)
        if self.component == 'judge':
            # The durable Reader/F1 artifact must precede any Judge dispatch.
            row = verify(runner.stage_files(self.run, self.j)['reader'])
            assert row['status'] == 'complete' and 'official_locomo_f1' in row['result']['metrics']
            self.text = self.label
        elif self.reader_error:
            raise RuntimeError('Empty ordinary generation text')
        return super().request(provider, model, body, **kwargs)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'run'; self.run.mkdir()
        old = job()
        reader_result = runner.reader_call(runner.SignatureCapture('Paris'), old, official)
        judge_result = runner.evaluator.judge_answer(runner.SignatureCapture('[[no]]'), old['question'], old['reference'], 'Paris')
        entry = {'source_job': old, 'reader': {'path': 'old-reader', 'sha256': 'old-reader-sha'},
             'judge': {'path': 'old-judge', 'sha256': 'old-judge-sha'},
             'reader_result': reader_result, 'judge_result': judge_result,
             'signatures': {s: runner.request_signature(old, s, 'Paris') for s in ('reader', 'judge')}}
        self.proof = {'entries': {'c0_q0': entry}, 'input_sha256': {}}
        save_new(self.run / 'proof.json', self.proof)
        self.m = {'schema': runner.SCHEMA, 'protocol_sha256': 'mock-protocol', 'phase_kind': 'firstpass',
              'phase_id': 'r01-test', 'source_manifest': {'path': 'mock-source', 'sha256': 'mock'},
              'baseline_reuse_proof': pointer(self.run / 'proof.json'), 'reuse_plans': {}}

    def plan(self, j):
        self.m['reuse_plans'][j['job_id']] = runner.reuse_plan(j, self.proof)

    def test_actual_reader_and_judge_capture_guard_and_parser(self):
        for stage in ('reader', 'judge'):
            signature = runner.request_signature(job(), stage)
            self.assertEqual(signature['model'], 'gemini-3.8-flash')
            self.assertEqual(signature['body']['generationConfig']['thinkingConfig'], {'thinkingLevel': 'low'})
            self.assertNotIn('temperature', signature['body']['generationConfig'])
        self.assertTrue(summary.audit_phase.__globals__['audit_stage'] is summary.audit_stage)

    def test_old_runner_globals_remain_unchanged(self):
        self.assertNotEqual(frozen.SCHEMA, runner.SCHEMA)
        self.assertEqual(len(frozen.ARMS), 7)
        self.assertEqual(frozen.identity(job(), self.m)['schema'], frozen.SCHEMA)
        self.assertEqual(runner.identity(job(), self.m)['schema'], runner.SCHEMA)
        self.assertEqual(runner.POLICY['conversation_workers'], 8)
        self.assertEqual(runner.POLICY['qa_threads_per_conversation'], 4)

    def test_exact_request_order_text_date_and_gold_boundaries(self):
        j = job('cheap'); self.assertIsNotNone(runner.reuse_plan(j, self.proof)['reader'])
        for field in ('order', 'text', 'fact_date'):
            changed = copy.deepcopy(j)
            if field == 'order':
                changed['selected_ids'].reverse(); changed['selected_facts'].reverse()
            elif field == 'text': changed['selected_facts'][0]['text'] += ' Changed.'
            else: changed['selected_facts'][0]['observation_date'] = '2023-01-02'
            self.assertIsNone(runner.reuse_plan(changed, self.proof)['reader'])
            self.assertIsNotNone(runner.reuse_plan(changed, self.proof)['judge_candidate'])
        for key, value in (('question', 'Other?'), ('reference', 'Berlin'), ('category', 2), ('reference_date', '2023-01-03')):
            changed = copy.deepcopy(j); changed[key] = value
            with self.assertRaises(ValueError): runner.reuse_plan(changed, self.proof)
        changed = copy.deepcopy(job()); changed['selected_facts'][0]['text'] += 'changed'
        with self.assertRaises(ValueError): runner.reuse_plan(changed, self.proof)

    def test_inherited_no_never_dispatches_even_if_client_init_failed(self):
        j = job('jev'); self.plan(j)
        runner.one(self.run, j, self.m, None, official, RuntimeError('mock initialization failure'))
        paths = runner.stage_files(self.run, j)
        qa = verify(paths['qa']); self.assertEqual(qa['status'], 'complete'); self.assertFalse(qa['judge_correct'])
        for stage in ('reader', 'judge'):
            row = verify(paths[stage]); self.assertEqual(row['model_calls'], 0); self.assertEqual(row['api_events'], [])
            self.assertFalse(paths[stage].with_suffix('.intent.json').exists())
        with self.assertRaises(RuntimeError): runner.one(self.run, j, self.m, None, official)

    def test_existing_unknown_intent_cannot_be_reclassified_as_inheritance(self):
        j = job(); self.plan(j)
        intent = runner.stage_files(self.run, j)['reader'].with_suffix('.intent.json')
        save_new(intent, {'unknown': True})
        with self.assertRaises(RuntimeError): runner.one(self.run, j, self.m, None, official)
        self.assertFalse(runner.stage_files(self.run, j)['reader'].exists())

    def test_changed_reader_is_durable_before_judge_and_no_valid_no_retry(self):
        j = job('cheap'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        api = MockAPI(self.run, j, reader_text='Berlin')
        runner.one(self.run, j, self.m, api, official)
        self.assertEqual(api.calls, ['reader', 'judge'])
        paths = runner.stage_files(self.run, j)
        self.assertFalse(verify(paths['qa'])['judge_correct'])
        out = {'outcomes': [], 'results': [{**{k:j[k] for k in ('conversation','method','query_id','job_id')},
                    **{s: pointer(p) for s,p in paths.items()}}]}
        parent = {**self.m, 'jobs': [j]}
        self.assertEqual(runner.recovery_jobs(parent, out), [])

    def test_reader_failure_prevents_judge_dispatch_and_unknown_not_retried(self):
        j = job('cheap'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        api = MockAPI(self.run, j, reader_error=True)
        runner.one(self.run, j, self.m, api, official)
        self.assertEqual(api.calls, ['reader'])
        paths = runner.stage_files(self.run, j)
        self.assertEqual(verify(paths['judge'])['error']['type'], 'ReaderFailed')
        out = {'outcomes': [], 'results': [{**{k:j[k] for k in ('conversation','method','query_id','job_id')},
                  **{s: pointer(p) for s,p in paths.items()}}]}
        # The mock has no ended physical ledger proof, so it must not qualify.
        with self.assertRaises(ValueError): runner.recovery_jobs({**self.m, 'jobs': [j]}, out)
        out['results'][0]['reader'] = None
        with self.assertRaises(ValueError): runner.recovery_jobs({**self.m, 'jobs': [j]}, out)

    def test_changed_facts_same_answer_inherits_old_no_without_new_judge(self):
        j = job('jev'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        api = MockAPI(self.run, j, label='[[yes]]')
        runner.one(self.run, j, self.m, api, official)
        paths = runner.stage_files(self.run, j)
        self.assertEqual(api.calls, ['reader'])
        judge = verify(paths['judge'])
        self.assertFalse(judge['result']['correct'])
        self.assertEqual(judge['original_reader'], self.proof['entries']['c0_q0']['reader'])
        self.assertEqual(judge['reader_sha256'], runner.sha(paths['reader']))
        self.assertEqual(judge['model_calls'], 0)
        save_new(self.run/'run_manifest.json', self.m)
        summary.audit_stage(Snapshot(), paths['judge'], {**runner.identity(j,self.m), 'stage':'judge',
              'reader_sha256':runner.sha(paths['reader'])}, self.run/'conv_0/api', runner.stage_context(j,self.m,'judge'), {},
              lambda capture: runner.evaluator.judge_answer(capture,j['question'],j['reference'],'Paris'))

    def test_semantically_similar_but_nonexact_answer_cannot_inherit_judge(self):
        j = job('jev'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        api = MockAPI(self.run, j, reader_text='Paris ')
        runner.one(self.run,j,self.m,api,official)
        self.assertEqual(api.calls,['reader','judge'])
        self.assertNotIn('inheritance_kind',verify(runner.stage_files(self.run,j)['judge']))

    def test_summary_replays_inherited_callback_and_rejects_tampering(self):
        j = job(); self.plan(j); save_new(self.run / 'run_manifest.json', self.m)
        runner.one(self.run, j, self.m, None, official)
        paths = runner.stage_files(self.run, j)
        expected = {**runner.identity(j, self.m), 'stage': 'reader'}
        row = summary.audit_stage(Snapshot(), paths['reader'], expected, self.run / 'conv_0/api',
               runner.stage_context(j, self.m, 'reader'), {}, lambda api: runner.reader_call(api, j, official))
        self.assertEqual(row['result']['prediction']['answer'], 'Paris')
        judge = summary.audit_stage(Snapshot(), paths['judge'], {**runner.identity(j,self.m), 'stage':'judge',
                    'reader_sha256': runner.sha(paths['reader'])}, self.run / 'conv_0/api',
                    runner.stage_context(j,self.m,'judge'), {},
                    lambda api: runner.evaluator.judge_answer(api,j['question'],j['reference'],'Paris'))
        self.assertFalse(judge['result']['correct'])
        changed = copy.deepcopy(j); changed['question'] = 'Elsewhere?'
        with self.assertRaises(InvalidArtifact):
            summary.audit_stage(Snapshot(), paths['reader'], expected, self.run / 'conv_0/api',
                runner.stage_context(j,self.m,'reader'), {}, lambda api: runner.reader_call(api,changed,official))

    def test_uniform_recovery_inherits_successful_reader_and_runs_judge_once(self):
        j = job('cheap'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        api = MockAPI(self.run, j, label='invalid label', reader_text='Berlin')
        runner.one(self.run, j, self.m, api, official)
        paths = runner.stage_files(self.run, j)
        self.assertEqual(verify(paths['judge'])['error']['type'], 'InvalidJudgeLabel')
        out = {'outcomes': [], 'results': [{**{k:j[k] for k in ('conversation','method','query_id','job_id')},
                  **{s: pointer(p) for s,p in paths.items()}}]}
        parent = {**self.m, 'jobs': [j]}
        # Isolate population/one() semantics after the separate bounded raw/ledger
        # admission predicate succeeds. Another test verifies absent proof rejects.
        with patch.dict(runner._impl, {'known_failure': lambda *a: None}):
            retries = runner.recovery_jobs(parent, out)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]['reader_source'], pointer(paths['reader']))
        recovery = self.run.with_name('recovery'); recovery.mkdir()
        m = {**self.m, 'phase_kind': 'technical_recovery', 'phase_id': 'recovery-test'}
        recovery_api = MockAPI(recovery, retries[0], label='[[no]]')
        runner.one(recovery, retries[0], m, recovery_api, official)
        self.assertEqual(recovery_api.calls, ['judge'])
        rp = runner.stage_files(recovery, retries[0])['reader']
        self.assertEqual(verify(rp)['result'], verify(paths['reader'])['result'])
        self.assertEqual(verify(rp)['model_calls'], 0)
        with self.assertRaises(ValueError): runner.recovery_jobs({**m, 'jobs': retries}, {'outcomes': [], 'results': []})

    def test_reader_recovery_same_answer_dynamically_inherits_prebound_old_judge(self):
        j = job('cheap'); j['selected_facts'][0]['text'] += ' changed'; self.plan(j)
        runner.one(self.run,j,self.m,MockAPI(self.run,j,reader_error=True),official)
        paths = runner.stage_files(self.run,j)
        out = {'outcomes':[], 'results':[{**{k:j[k] for k in ('conversation','method','query_id','job_id')},
                      **{s:pointer(p) for s,p in paths.items()}}]}
        with patch.dict(runner._impl, {'known_failure':lambda *a:None}):
            retry = runner.recovery_jobs({**self.m,'jobs':[j]},out)[0]
        self.assertIsNone(retry['reader_source'])
        recovery=self.run.with_name('recovery'); recovery.mkdir()
        m={**self.m,'phase_kind':'technical_recovery','phase_id':'recovery-test'}
        api=MockAPI(recovery,retry,reader_text='Paris',label='[[yes]]')
        runner.one(recovery,retry,m,api,official)
        self.assertEqual(api.calls,['reader'])
        jp=runner.stage_files(recovery,retry)['judge']
        self.assertFalse(verify(jp)['result']['correct'])
        self.assertEqual(verify(jp)['model_calls'],0)
        save_new(recovery/'run_manifest.json',m)
        summary.audit_stage(Snapshot(),jp,{**runner.identity(retry,m),'stage':'judge',
            'reader_sha256':runner.sha(runner.stage_files(recovery,retry)['reader'])},recovery/'conv_0/api',
            runner.stage_context(retry,m,'judge'),{},
            lambda capture:runner.evaluator.judge_answer(capture,retry['question'],retry['reference'],'Paris'))

    def test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary(self):
        rows = []
        for arm, correct in (('baseline', 1153), ('cheap', 1199), ('jev', 1200)):
            rows.extend({'method': arm, 'scores': {'judge_correct': 1 if i < correct else None if i == 1539 and arm != 'baseline' else 0}}
                        for i in range(1540))
        result = summary.stopping_result(rows)
        self.assertEqual(result['minimum_strict_gt_3pp'], 1200)
        self.assertFalse(result['strict_gt_3pp_fixed_denominator']['cheap'])
        self.assertTrue(result['strict_gt_3pp_fixed_denominator']['jev'])
        self.assertEqual(result['missing']['jev'], 1)
        self.assertEqual(result['independent_component_count'], 1)
        rows[0]['scores']['judge_correct'] = None
        invalid = summary.stopping_result(rows)
        self.assertFalse(invalid['threshold_eligible'])
        self.assertEqual(invalid['minimum_strict_gt_3pp'], 1200)
        self.assertFalse(invalid['strict_gt_3pp_fixed_denominator']['jev'])


if __name__ == '__main__': unittest.main()
