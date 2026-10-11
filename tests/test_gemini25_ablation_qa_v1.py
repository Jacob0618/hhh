"""Offline QA adapter tests: raw replay, immutable stages and three-arm metrics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gemini25_ablation_qa_v1 as d
from tools import summarize_gemini25_ablation_v1 as s
from jev_mem.gemini25_ablation_provider_v1 import Gemini25API, check_request, MODEL
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem import evaluation_gemini25_ablation_v1 as e


class RecordedFake:
    generate = Gemini25API.generate

    def __init__(self, folder, label='Reasoning.\n[[no]]', context=None):
        self.folder = Path(folder)
        self.label = label
        self.context = context or {}
        self.events = []

    def scoped(self, **context):
        return RecordedFake(self.folder, self.label, {**self.context, **context})

    def _redact(self, value):
        return str(value)

    def request(self, provider, model, body, *, stage):
        check_request(provider, model, body, 'generateContent', stage)
        signature = {'provider': provider, 'model': model, 'operation': 'generateContent', 'body': body}
        key = request_digest(signature)
        intent = self.folder / 'request_intents' / (key + '.json')
        raw = self.folder / 'api_cache' / (key + '.json')
        intent.parent.mkdir(parents=True, exist_ok=True)
        raw.parent.mkdir(parents=True, exist_ok=True)
        cached = raw.exists()
        if not cached:
            intent.write_text(json.dumps({'signature': signature, 'request_hash': key, 'stage': stage,
                                          'retries': 2, 'context': self.context}))
            response = {'modelVersion': MODEL, 'usageMetadata': {'promptTokenCount': 9, 'candidatesTokenCount': 3,
                'thoughtsTokenCount': 0, 'totalTokenCount': 12}, 'candidates': [{'finishReason': 'STOP',
                'content': {'parts': [{'text': 'alpha' if stage == e.READER_STAGE else self.label}]}}]}
            raw.write_text(json.dumps({'request': signature, 'response': response, 'stage': stage}))
        else:
            response = json.loads(raw.read_text())['response']
        self.events.append({'request_hash': key, 'stage': stage, 'provider': provider, 'model': model,
            'operation': 'generateContent', 'source': 'existing_run_cache' if cached else 'new_request',
            'context': self.context, 'cache_path': str(raw), 'cache_sha256': d._base.sha(raw), 'error_type': None})
        row = {'timestamp_utc': '2026-10-07T00:00:00+00:00', 'request_hash': key, 'stage': stage,
               'provider': provider, 'model': model, 'operation': 'generateContent', 'context': self.context,
               'cache_hit': cached, 'usage': response['usageMetadata'], 'resolved_model': MODEL}
        row.update({'usage_semantics': 'reused_response_no_new_model_usage'} if cached else {'attempt': 1, 'status': 200})
        with (self.folder / 'api_calls.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        return response


def job(arm='baseline', index=0):
    facts = [{'id': str(i), 'text': 'alpha ' + str(i), 'observation_date': '2023-05-08'} for i in range(20)]
    qid = 'c0_q' + str(index)
    return {'conversation': 0, 'method': arm, 'query_id': qid, 'job_id': arm + '/' + qid, 'status': 'ready',
            'question': 'Which name?', 'reference': 'alpha', 'category': 4, 'reference_date': '2023-05-08',
            'selected_ids': [f['id'] for f in facts], 'selected_facts': facts,
            'bank': {'path': 'newbank', 'sha256': 'new'}, 'selection': {'path': 'newselection', 'sha256': 'new'}}


def phase(run, jobs, kind='firstpass', label='Reasoning.\n[[no]]'):
    m = {'phase_kind': kind, 'phase_id': 'offline_mock_' + kind, 'protocol_sha256': 'offline_mock',
         'source_manifest': {'path': 'mock_fresh_source', 'sha256': 'mock'}, 'jobs': jobs,
         'official_metric_provenance': d._base.verified_official_metric()[1]}
    api = RecordedFake(run / 'conv_0/api', label)
    official, _ = d._base.verified_official_metric()
    for j in jobs:
        d.one(run, j, m, api, official)
    out = {'counts': {}, 'results': [{**{k: j[k] for k in ('job_id', 'conversation', 'method', 'query_id')},
            **{k: d._base.pointer(v) for k, v in d.stage_files(run, j).items()}} for j in jobs]}
    (run / 'run_manifest.json').write_text(json.dumps(m))
    (run / 'run_outcome.json').write_text(json.dumps(out))
    return m, out


class Gemini25QATests(unittest.TestCase):
    def setUp(self):
        for target in ('requests.Session.post', 'requests.Session.request', 'socket.create_connection',
                       'jev_mem.providers.load_credentials'):
            p = patch(target, side_effect=AssertionError('No network or credentials in QA tests'))
            p.start()
            self.addCleanup(p.stop)

    def test_scope_isolation_and_protocol_budget(self):
        from tools import run_gemini38_plain_qa_v1 as old
        self.assertEqual(old.POLICY['reader_model'], 'gemini-3.8-flash')
        self.assertEqual(d.POLICY['reader_model'], MODEL)
        self.assertEqual(d.POLICY['judge_model'], MODEL)
        self.assertEqual(d.POLICY['thinking_budget'], 0)
        self.assertEqual(d.POLICY['qa_max_output_tokens'], 8192)
        self.assertNotIn('thinking_level', d.POLICY)
        self.assertEqual((d.POLICY['conversation_workers'], d.POLICY['qa_threads_per_conversation']), (8, 4))
        self.assertIsNot(d.one.__globals__, old.one.__globals__)
        # Provider tests may reload the evaluator module after this adapter was
        # imported. A frozen adapter retains its original function object; its
        # code object and isolated scientific bindings must still be identical.
        bound_reader = d.one.__globals__['read_answer']
        self.assertIs(bound_reader.__code__, e.read_answer.__code__)
        for name in ('MODEL', 'PROTOCOL', 'READER_PROTOCOL', 'READER_STAGE', 'QA_MAX_OUTPUT'):
            self.assertEqual(bound_reader.__globals__[name], getattr(e, name))
        self.assertIs(d.run.__globals__['PlainTextEvaluationAPI'], Gemini25API)
        self.assertEqual(d.ARMS, ('baseline', 'r03_original', 'original_multi_source_closure'))

    def test_new_three_arm_4620_prepare_is_offline_and_binds_fresh_source(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'runs').mkdir()
            jobs = [job(arm, i) for arm in d.ARMS for i in range(1540)]
            source = {'schema': d.SOURCE_SCHEMA, 'jobs': jobs,
                      'official_metric_provenance': d._base.verified_official_metric()[1]}
            sp = root / 'source.json'
            d._base.save_new(sp, source)
            read_source = lambda path: d._base.verify(path)
            with patch.object(d, 'ROOT', root), patch.object(d, 'CODE', ()), \
                 patch.object(d, 'source_validate', side_effect=read_source), \
                 patch.dict(d._impl, ROOT=root, CODE=(), source_validate=read_source,
                            PlainTextEvaluationAPI=lambda *a, **k: self.fail('No API constructor during prepare')):
                result = d.prepare(root / 'runs' / d.RUN_NAME, sp)
                self.assertEqual(result['jobs'], 4620)
                self.assertEqual(result['model_calls'], 0)
                manifest = d.manifest(result['run_dir'])
                self.assertEqual(len(manifest['jobs']), 4620)
                self.assertEqual(manifest['cache_roots'], [])
                self.assertEqual(manifest['source_manifest'], d._base.pointer(sp))
                self.assertFalse((Path(result['run_dir']) / 'execution_intent.json').exists())

    def test_firstpass_rejects_historical_reader(self):
        jobs = [job(arm, i) for arm in d.ARMS for i in range(1540)]
        jobs[0]['reader_source'] = {'path': 'old', 'sha256': 'old'}
        with patch.object(d, 'source_validate', return_value={'jobs': jobs}):
            with self.assertRaisesRegex(ValueError, 'historical'):
                d.prepare(d.DEFAULT_RUN, 'unused')

    def test_first_terminal_no_and_same_phase_cache_replay_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'first'
            jobs = [job(a) for a in d.ARMS]
            m, out = phase(run, jobs)
            with patch.object(d, 'terminal', return_value=(m, out)):
                report = s.audit_phase(run, {})
            self.assertEqual(report['errors'], [])
            self.assertEqual(d.recovery_jobs(m, out), [])
            rows = s.record_rows(jobs, report)
            self.assertTrue(all(r['scores']['judge_correct'] == 0 for r in rows))
            self.assertTrue(all(r['scores']['official_locomo_f1'] == 1 for r in rows))
            self.assertTrue(all(r['judge_parse_mode'] == 'terminal' for r in rows))
            use = s.measured_usage(report['ledger'])
            self.assertEqual((use['physical_attempts'], use['cache_events']), (2, 4))
            self.assertEqual(use['token_fields']['output']['recorded_token_sum'], 6)
            self.assertIsNone(use['total_cost_usd'])

    def test_invalid_label_recovers_judge_only_and_preserves_reader(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'first'
            j = job()
            m, out = phase(run, [j], label='no legal label')
            candidates = d.recovery_jobs(m, out)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]['attempt_kind'], 'judge_only_after_successful_reader')
            with patch.object(d, 'terminal', return_value=(m, out)):
                first = s.audit_phase(run, {})
            recovery = Path(t) / 'recovery'
            mr, retried = phase(recovery, candidates, 'technical_recovery', '[[yes]]')
            with patch.object(d, 'terminal', return_value=(mr, retried)):
                second = s.audit_phase(recovery, {}, parent=first)
            self.assertEqual(second['errors'], [])
            row = s.record_rows([j], first, second, final=True)[0]
            self.assertEqual(row['scores']['judge_correct'], 1)
            self.assertEqual(row['judge_resolution']['source'], 'recovery_accepted_label')
            self.assertEqual(row['reader_artifact'], d._base.pointer(d.stage_files(run, j)['reader']))
            self.assertEqual(s.measured_usage(second['ledger'])['physical_attempts'], 1)
            with self.assertRaisesRegex(ValueError, 'Only one'):
                d.recovery_jobs(mr, retried)

    def test_unknown_or_incomplete_phase_is_not_recoverable(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'first'
            m, out = phase(run, [job()])
            out['results'][0]['reader'] = None
            with self.assertRaisesRegex(ValueError, 'Missing/unknown'):
                d.recovery_jobs(m, out)
            with self.assertRaises(FileNotFoundError):
                d.prepare_recovery(Path(t) / 'never_executed')

    def test_orphan_unknown_request_intent_blocks_seal_and_recovery_gate(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'first'
            m, out = phase(run, [job()])
            unknown = run / 'conv_0/api/request_intents/unfinished.json'
            unknown.write_text(json.dumps({'request_hash': 'unfinished', 'stage': e.READER_STAGE}))
            with patch.object(d, 'terminal', return_value=(m, out)), \
                 patch.object(d, 'source_validate', return_value={'schema': d.SOURCE_SCHEMA}):
                with self.assertRaisesRegex(ValueError, 'Unknown request intents'):
                    d._resolved_phase(run)
                with self.assertRaisesRegex(ValueError, 'Unknown request intents'):
                    d.prepare_recovery(run)

    def test_unknown_stage_intent_blocks_raw_seal_gate(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'first'
            j = job()
            m, out = phase(run, [j])
            rp = d.stage_files(run, j)['reader']
            row = d._base.verify(rp)
            row.update(status='failed', result=None,
                       error={'type': 'UnresolvedStageIntent', 'message': 'Prior stage is retained; no implicit redispatch'})
            row.pop('content_sha256')
            rp.unlink()
            d._base.save_new(rp, row)
            out['results'][0]['reader'] = d._base.pointer(rp)
            with patch.object(d, 'terminal', return_value=(m, out)):
                with self.assertRaisesRegex(ValueError, 'Unknown/local stage intent'):
                    d._resolved_phase(run)

    def test_rehashed_f1_tamper_and_old_model_raw_fail_replay(self):
        for kind in ('f1', 'model'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                run = Path(t) / 'first'
                j = job()
                m, out = phase(run, [j])
                rp = d.stage_files(run, j)['reader']
                row = d._base.verify(rp)
                if kind == 'f1':
                    row['result']['metrics']['official_locomo_f1'] = 0.123
                    row.pop('content_sha256')
                    rp.unlink()
                    d._base.save_new(rp, row)
                else:
                    rawp = Path(row['api_events'][0]['cache_path'])
                    raw = json.loads(rawp.read_text())
                    raw['request']['model'] = 'gemini-3.8-flash'
                    rawp.write_text(json.dumps(raw))
                with patch.object(d, 'terminal', return_value=(m, out)):
                    report = s.audit_phase(run, {})
                self.assertTrue(any(x['stage'] == 'reader' for x in report['errors']))
                self.assertIsNone(s.record_rows([j], report)[0]['scores']['official_locomo_f1'])

    def test_fixed_denominators_all_pairs_and_category_flips(self):
        rows = []
        for arm in d.ARMS:
            for i in range(1540):
                rows.append({'query_id': 'q' + str(i), 'conversation': i % 10, 'category': i % 4 + 1,
                    'method': arm, 'selection_status': 'complete', 'reader_status': 'complete',
                    'judge_status': 'missing', 'qa_status': 'failed',
                    'scores': {'official_locomo_f1': .5, 'custom_token_f1': .25, 'judge_correct': None}})
        for row in rows:
            if row['query_id'] == 'q0':
                row['scores']['judge_correct'] = 1 if row['method'] == d.ARMS[2] else 0
        result = s.aggregate(rows, bootstrap_samples=1)
        self.assertEqual(len(result['paired_comparisons']), 3)
        for arm in d.ARMS:
            metrics = result['arms'][arm]['metrics']
            self.assertEqual(metrics['official_locomo_f1']['valid_denominator'], 1540)
            self.assertEqual(metrics['judge_correct']['planned_denominator'], 1540)
            self.assertEqual(metrics['judge_correct']['valid_denominator'], 1)
            self.assertIsNone(metrics['judge_correct']['full_cohort_observed_mean'])
        pair = result['paired_comparisons']['original_multi_source_closure_minus_r03_original']
        self.assertEqual(pair['judge_transitions']['0_to_1']['query_ids'], ['q0'])
        self.assertEqual(pair['by_category']['1']['judge_transitions']['0_to_1']['query_ids'], ['q0'])
        self.assertEqual(pair['by_category']['1']['fixed_denominator'], 385)

    def test_readiness_does_not_invent_scores(self):
        with tempfile.TemporaryDirectory() as t:
            value = s.readiness(t)
            self.assertFalse(value['ready_for_summary'])
            self.assertFalse(value['results_computed'])
            self.assertEqual(value['model_calls'], 0)
            with self.assertRaises(FileNotFoundError):
                s.summarize(t)


if __name__ == '__main__':
    unittest.main()
