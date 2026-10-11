"""R06 offline tests: frozen stage regressions plus isolated audit acceleration."""
import copy
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_evidence_coverage_qa_r06 as runner
from tools import summarize_evidence_coverage_r06 as summary
from tools import run_evidence_exchange_qa_r01 as prior
from tools import run_evidence_selection_qa_r03 as template
from tools import run_source_append_qa_r05 as source_append

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r06_original_adapter_tests', ROOT/'tests/test_evidence_exchange_qa_r01.py')
oldtests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldtests)
scope = {**vars(oldtests), 'runner': runner, 'summary': summary}
methods = {name: FunctionType(f.__code__, scope, name, f.__defaults__, f.__closure__)
    for name, f in vars(oldtests.AdapterTests).items() if isinstance(f, FunctionType)
    and name != 'test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary'}
R06InheritedBehaviorTests = type('R06InheritedBehaviorTests', (unittest.TestCase,), methods)


class R06Contracts(unittest.TestCase):
    def test_inherited_no_dispatch_rejects_same_context_ledger_activity(self):
        case = R06InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            j = oldtests.job('jev')
            case.plan(j)
            oldtests.save_new(case.run/'run_manifest.json', case.m)
            runner.one(case.run,j,case.m,None,oldtests.official)
            path = runner.stage_files(case.run,j)['reader']
            context = runner.stage_context(j,case.m,'reader')
            indexed = {('unexpected-request','reader',summary.original.old.fingerprint(context)):
                       [{'context':context, 'cache_hit':True}]}
            with self.assertRaisesRegex(oldtests.InvalidArtifact,'Inherited stage was dispatched'):
                summary.audit_stage(summary.Snapshot(),path,{**runner.identity(j,case.m),'stage':'reader'},
                    case.run/'conv_0/api',context,indexed,
                    lambda replay:runner.reader_call(replay,j,oldtests.official))
        finally:
            case.doCleanups()

    def test_ordinary_raw_replay_matches_frozen_audit_and_checks_other_method_producer(self):
        """Real frozen serialization/parsing plus durable synthetic raw/intent/ledger.

        Reader is an exact same-conversation cache consumer of another method;
        Judge is a new physical request. No model client or HTTP is constructed.
        """
        case = R06InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            j = oldtests.job('jev')
            j['selected_facts'][0]['text'] += ' changed'
            case.plan(j)
            fp = summary.original.old.fingerprint
            provider_hash = runner.Gemini38Replay.__mro__[2].request.__globals__['request_digest']
            indexed = {}
            producer_intents = []

            class TraceAPI(oldtests.MockAPI):
                def scoped(self, **context):
                    child = super().scoped(**context)
                    child.context = context
                    return child

                def request(self, provider, model, body, **kwargs):
                    response = super().request(provider, model, body, **kwargs)
                    response['modelVersion'] = model
                    response['usageMetadata'] = {'promptTokenCount': 19, 'candidatesTokenCount': 2}
                    signature = {'provider': provider, 'model': model,
                        'operation': kwargs.get('operation', 'generateContent'), 'body': body}
                    key = provider_hash(signature)
                    stage = kwargs.get('stage', 'unspecified')
                    api_dir = case.run/'conv_0/api'
                    raw = api_dir/'api_cache'/f'{key}.json'
                    intent = api_dir/'request_intents'/f'{key}.json'
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    intent.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_text(json.dumps({'request': signature, 'response': response}), encoding='utf-8')
                    producer = {**self.context, 'method': 'cheap'} if stage == 'reader' else self.context
                    intent.write_text(json.dumps({'signature': signature, 'request_hash': key,
                        'stage': stage, 'retries': 2, 'context': producer}), encoding='utf-8')
                    producer_intents.append(intent)
                    base = {**{k:signature[k] for k in ('provider','model','operation')},
                        'request_hash':key, 'stage':stage, 'usage':response['usageMetadata'],
                        'resolved_model':model}
                    physical = {**base, 'context':producer, 'attempt':1, 'status':200}
                    indexed.setdefault((key, stage, fp(producer)), []).append(physical)
                    if stage == 'reader':
                        cache_row = {**base, 'context':self.context, 'cache_hit':True,
                            'usage_semantics':'reused_response_no_new_model_usage'}
                        indexed.setdefault((key, stage, fp(self.context)), []).append(cache_row)
                    self.events.append({**{k:base[k] for k in ('provider','model','operation','request_hash','stage')},
                        'context':self.context, 'source':'existing_run_cache' if stage=='reader' else 'new_request',
                        'cache_path':str(raw.resolve()), 'cache_sha256':runner.sha(raw), 'error_type':None})
                    return response

            api = TraceAPI(case.run, j, reader_text='Berlin')
            runner.one(case.run, j, case.m, api, oldtests.official)
            self.assertEqual(api.calls, ['reader', 'judge'])
            paths = runner.stage_files(case.run, j)
            for stage in ('reader', 'judge'):
                expected = {**runner.identity(j, case.m), 'stage':stage}
                if stage == 'judge': expected['reader_sha256'] = runner.sha(paths['reader'])
                callback = (lambda replay: runner.reader_call(replay,j,oldtests.official)) if stage=='reader' else (
                    lambda replay: runner.evaluator.judge_answer(replay,j['question'],j['reference'],'Berlin'))
                args = (paths[stage], expected, case.run/'conv_0/api',
                        runner.stage_context(j,case.m,stage), indexed, callback)
                ordinary = runner._new_audit_stage(summary.original.Snapshot(), *args)
                accelerated_snapshot = summary.Snapshot()
                accelerated = summary.audit_stage(accelerated_snapshot, *args)
                self.assertEqual(accelerated, ordinary)
                self.assertEqual(accelerated_snapshot.unchanged(), [])
            # A scoped .values() must not hide a producer owned by another method.
            wrong = json.loads(producer_intents[0].read_text(encoding='utf-8'))
            wrong['context'] = {**wrong['context'], 'method':'unapproved-producer'}
            producer_intents[0].write_text(json.dumps(wrong), encoding='utf-8')
            expected = {**runner.identity(j,case.m), 'stage':'reader'}
            args = (paths['reader'], expected, case.run/'conv_0/api',
                runner.stage_context(j,case.m,'reader'), indexed,
                lambda replay: runner.reader_call(replay,j,oldtests.official))
            for fn, snap in ((runner._new_audit_stage, summary.original.Snapshot()),
                             (summary.audit_stage, summary.Snapshot())):
                with self.assertRaisesRegex(oldtests.InvalidArtifact, 'intent/context/bounded'):
                    fn(snap, *args)
        finally:
            case.doCleanups()

    def test_clone_globals_isolate_original_modules_and_plain_signatures(self):
        for old in (prior, template, source_append):
            self.assertIsNot(runner._impl, old._impl)
            self.assertEqual(old._impl['SCHEMA'], old.SCHEMA)
        self.assertIs(runner.identity.__globals__, runner._impl)
        self.assertIs(runner.one.__globals__, runner._helpers)
        self.assertIs(summary.audit_phase.__globals__['runner'], runner)
        self.assertIs(summary.audit_phase.__globals__['Snapshot'], summary.Snapshot)
        self.assertIsNot(summary.Snapshot, summary.original.Snapshot)
        job = oldtests.job()
        self.assertEqual(runner.request_signature(job, 'reader'), prior.request_signature(job, 'reader'))
        self.assertEqual(runner.request_signature(job, 'judge', 'No.'), prior.request_signature(job, 'judge', 'No.'))
        self.assertEqual(runner.POLICY['conversation_workers'], 8)
        self.assertEqual(runner.POLICY['qa_threads_per_conversation'], 4)

    def test_new_membership_has_new_reader_request_but_exact_answer_reuses_valid_no(self):
        case = R06InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            original = oldtests.job('jev')
            job = copy.deepcopy(original)
            job['selected_facts'][-1] = {'id': 'new-original-bank-fact', 'text': 'A different original memory.', 'observation_date': '2023-01-01'}
            job['selected_ids'][-1] = job['selected_facts'][-1]['id']
            self.assertNotEqual(runner.request_signature(job, 'reader'), runner.request_signature(original, 'reader'))
            case.plan(job)
            self.assertIsNone(case.m['reuse_plans'][job['job_id']]['reader'])
            api = oldtests.MockAPI(case.run, job, label='[[yes]]')
            runner.one(case.run, job, case.m, api, oldtests.official)
            self.assertEqual(api.calls, ['reader'])
            row = runner.verify(runner.stage_files(case.run, job)['judge'])
            self.assertFalse(row['result']['correct'])
            self.assertEqual(row['reader_sha256'], runner.sha(runner.stage_files(case.run, job)['reader']))
        finally: case.doCleanups()

    def test_source_validator_delegates_r06_schema_not_source_prefix_schema(self):
        calls = []
        def validate(path):
            calls.append(path)
            return {'schema': runner.SOURCE_SCHEMA}
        fake = SimpleNamespace(validate_source=validate)
        with patch.dict(sys.modules, {'tools.prepare_evidence_coverage_r06': fake}):
            self.assertEqual(runner.source_validate('new-sealed-source.json')['schema'], runner.SOURCE_SCHEMA)
            self.assertIsInstance(calls[0], Path)
            fake.validate_source = lambda p: {'schema': source_append.SOURCE_SCHEMA}
            with self.assertRaisesRegex(ValueError, 'R06 sealed'):
                runner.source_validate('wrong-source.json')

    def test_snapshot_one_byte_read_and_final_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'raw.json'
            path.write_text('{"value":1}', encoding='utf-8')
            snap = summary.Snapshot()
            count = []
            original_read = Path.read_bytes
            def track(p):
                count.append(p.resolve())
                return original_read(p)
            with patch.object(Path, 'read_bytes', track):
                self.assertEqual(snap.read(path), {'value':1})
                self.assertEqual(snap.read(path), {'value':1})
                snap.sha(path)
                self.assertEqual(count, [path.resolve()])
                path.write_text('{"value":2}', encoding='utf-8')
                self.assertEqual(snap.read(path), {'value':1})
                self.assertEqual(snap.unchanged(), [str(path.resolve())])
                self.assertEqual(len(count), 2)

    def test_snapshot_prebound_hash_cannot_accept_conflicting_first_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bound.json'
            path.write_text('{}', encoding='utf-8')
            snap = summary.Snapshot()
            snap.files[path.resolve()] = 'wrong-existing-binding'
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                snap.read(path)

    def test_context_values_scoped_but_exact_cache_producer_rows_remain_available(self):
        consumer = {'conversation':0, 'method':'jev', 'query_id':'c0_q1', 'component':'reader'}
        producer = {**consumer, 'method':'cheap'}
        other = {**consumer, 'query_id':'c0_q2'}
        fp = summary.original.old.fingerprint
        produced = {'context':producer, 'cache_hit':False, 'status':200}
        consumed = {'context':consumer, 'cache_hit':True}
        unrelated = {'context':other, 'status':200}
        indexed = {('same', 'reader', fp(producer)):[produced],
                   ('same', 'reader', fp(consumer)):[consumed],
                   ('other', 'reader', fp(other)):[unrelated]}
        snap = summary.Snapshot()
        view = summary.ContextIndexed(snap, indexed, consumer)
        self.assertEqual([r for rs in view.values() for r in rs], [consumed])
        self.assertEqual(dict(view.items()), indexed)
        self.assertEqual(view.get(('same', 'reader', fp(producer))), [produced])
        replay = runner.Gemini38Replay('unused', [], consumer, view)
        self.assertEqual(replay._scope_rows('same', 'reader'), [produced, consumed])
        another = summary.ContextIndexed(snap, indexed, producer)
        self.assertEqual(len(snap._context_indexes), 1)
        self.assertEqual([r for rs in another.values() for r in rs], [produced])

    def test_fixed_stop_count_one_retrieval_family_not_previous_source_family(self):
        rows = [{'method':a, 'scores':{'judge_correct':int(i<n)}}
                for a,n in [('baseline',1153),('cheap',1199),('jev',1200)] for i in range(1540)]
        result = summary.stopping_result(rows)
        self.assertEqual(result['minimum_strict_gt_3pp'], 1200)
        self.assertEqual(result['successful_jev_components_counted_here'], 1)
        self.assertTrue(result['independent_of_source_append'])
        self.assertNotIn('independent_component_count', result)
        self.assertIn('shared_with_r03', result['component_family'])
        rows[0]['scores']['judge_correct'] = None
        invalid = summary.stopping_result(rows)
        self.assertFalse(invalid['threshold_eligible'])
        self.assertEqual(invalid['minimum_strict_gt_3pp'], 1200)
        self.assertEqual(invalid['successful_jev_components_counted_here'], 0)


if __name__ == '__main__': unittest.main()
