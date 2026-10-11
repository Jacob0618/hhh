"""R08 offline tests: frozen stage regressions plus isolated audit acceleration."""
import copy
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_facet_discovery_qa_r08 as runner
from tools import summarize_facet_discovery_r08 as summary
from tools import run_evidence_exchange_qa_r01 as prior
from tools import run_evidence_selection_qa_r03 as template
from tools import run_source_append_qa_r05 as source_append

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r08_original_adapter_tests', ROOT/'tests/test_evidence_exchange_qa_r01.py')
oldtests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldtests)
scope = {**vars(oldtests), 'runner': runner, 'summary': summary}
methods = {name: FunctionType(f.__code__, scope, name, f.__defaults__, f.__closure__)
    for name, f in vars(oldtests.AdapterTests).items() if isinstance(f, FunctionType)
    and name not in {'test_stop_rule_uses_fixed_1540_nulls_and_strict_boundary',
        'test_uniform_recovery_inherits_successful_reader_and_runs_judge_once',
        'test_reader_recovery_same_answer_dynamically_inherits_prebound_old_judge'}}
R08InheritedBehaviorTests = type('R08InheritedBehaviorTests', (unittest.TestCase,), methods)


class TracedFailureAPI(oldtests.MockAPI):
    """Real provider signature, synthetic durable raw/intent/physical ledger.

    Failure admission is never mocked. The actual frozen failure replay and
    the new R08 narrow eligibility filter must independently accept/reject it.
    """
    def __init__(self, run, j, m, failure_stage='judge', statuses=('transport_error',)*3,
                 reader_text='Berlin', malformed=False):
        super().__init__(run,j,reader_text=reader_text,
                         label='invalid label' if malformed else '[[no]]')
        self.m=m;self.failure_stage=failure_stage;self.statuses=statuses;self.malformed=malformed

    def scoped(self, **context):
        child=super().scoped(**context);child.context=context;return child

    def request(self, provider, model, body, **kwargs):
        signature={'provider':provider,'model':model,
                   'operation':kwargs.get('operation','generateContent'),'body':body}
        key=runner.Gemini38Replay.__mro__[2].request.__globals__['request_digest'](signature)
        stage=kwargs.get('stage','unspecified');api_dir=self.run/'conv_0/api'
        raw=api_dir/'api_cache'/f'{key}.json';intent=api_dir/'request_intents'/f'{key}.json'
        raw.parent.mkdir(parents=True,exist_ok=True);intent.parent.mkdir(parents=True,exist_ok=True)
        intent.write_text(json.dumps({'signature':signature,'request_hash':key,'stage':stage,
            'retries':2,'context':self.context}),encoding='utf8')
        fails=self.component==self.failure_stage and not self.malformed
        statuses=self.statuses if fails else (200,)
        ledger=api_dir/'api_calls.jsonl'
        with ledger.open('a',encoding='utf8')as file:
            for i,status in enumerate(statuses,1):
                file.write(json.dumps({**{k:signature[k]for k in ('provider','model','operation')},
                    'request_hash':key,'stage':stage,'context':self.context,'attempt':i,
                    'status':status,'cache_hit':False,'usage':{} if status==200 else None,
                    'resolved_model':model if status==200 else None,
                    'timestamp_utc':f'2026-10-04T00:00:0{i}Z'})+'\n')
        if fails:
            self.calls.append(self.component)
            self.events.append({'provider':provider,'model':model,'operation':signature['operation'],
                'request_hash':key,'stage':stage,'context':self.context,'source':'new_request',
                'cache_path':str(raw.resolve()),'cache_sha256':None,'error_type':'RuntimeError'})
            raise RuntimeError('synthetic ended provider failure')
        response=super().request(provider,model,body,**kwargs)
        if self.malformed and self.component==self.failure_stage=='reader':
            response={'candidates':[]}
        response['modelVersion']=model
        raw.write_text(json.dumps({'request':signature,'response':response}),encoding='utf8')
        self.events.append({'provider':provider,'model':model,'operation':signature['operation'],
            'request_hash':key,'stage':stage,'context':self.context,'source':'new_request',
            'cache_path':str(raw.resolve()),'cache_sha256':runner.sha(raw),'error_type':None})
        return response


class NarrowRecovery(unittest.TestCase):
    def setUp(self):
        self.case=R08InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.run=self.case.run;self.m=self.case.m
        self.j=oldtests.job('cheap');self.j['selected_facts'][0]['text']+=' changed'
        self.case.plan(self.j)

    def firstpass(self,**kwargs):
        api=TracedFailureAPI(self.run,self.j,self.m,**kwargs)
        runner.one(self.run,self.j,self.m,api,oldtests.official)
        paths=runner.stage_files(self.run,self.j)
        out={'outcomes':[],'results':[{**{k:self.j[k]for k in ('conversation','method','query_id','job_id')},
            **{stage:runner.pointer(path)for stage,path in paths.items()}}]}
        return {**self.m,'jobs':[self.j]},out,paths,api

    def test_exhausted_judge_transport_inherits_successful_reader_runs_only_judge(self):
        parent,out,paths,_=self.firstpass()
        retry=runner.recovery_jobs(parent,out)
        self.assertEqual(len(retry),1);self.assertEqual(retry[0]['reader_source'],runner.pointer(paths['reader']))
        recovery=self.run.with_name('recovery');recovery.mkdir()
        m={**self.m,'phase_kind':'technical_recovery','phase_id':'r08-recovery-test'}
        api=oldtests.MockAPI(recovery,retry[0],reader_text='Berlin',label='[[no]]')
        runner.one(recovery,retry[0],m,api,oldtests.official)
        self.assertEqual(api.calls,['judge'])
        row=runner.verify(runner.stage_files(recovery,retry[0])['reader'])
        self.assertEqual(row['result'],runner.verify(paths['reader'])['result'])
        self.assertEqual(row['model_calls'],0)
        with self.assertRaises(ValueError):runner.recovery_jobs({**m,'jobs':retry},{'outcomes':[],'results':[]})

    def test_exhausted_reader_retry_then_exact_answer_inherits_old_valid_no(self):
        parent,out,paths,_=self.firstpass(failure_stage='reader')
        retry=runner.recovery_jobs(parent,out)[0];self.assertIsNone(retry['reader_source'])
        self.assertEqual(runner.verify(paths['judge'])['status'],'not_run')
        recovery=self.run.with_name('recovery');recovery.mkdir()
        m={**self.m,'phase_kind':'technical_recovery','phase_id':'r08-reader-recovery-test'}
        api=oldtests.MockAPI(recovery,retry,reader_text='Paris',label='[[yes]]')
        runner.one(recovery,retry,m,api,oldtests.official)
        self.assertEqual(api.calls,['reader'])
        row=runner.verify(runner.stage_files(recovery,retry)['judge'])
        self.assertFalse(row['result']['correct']);self.assertEqual(row['model_calls'],0)

    def test_malformed_http200_label_is_audited_and_retained_without_retry(self):
        parent,out,paths,_=self.firstpass(malformed=True)
        self.assertEqual(runner.verify(paths['judge'])['error']['type'],'InvalidJudgeLabel')
        self.assertEqual(runner.recovery_jobs(parent,out),[])
        self.assertEqual(runner.verify(paths['judge'])['raw_invalid_label'],'invalid label')

    def test_nonretryable_http_audited_and_skipped(self):
        parent,out,_,_=self.firstpass(statuses=(401,))
        self.assertEqual(runner.recovery_jobs(parent,out),[])

    def test_malformed_reader_http200_is_preserved_and_does_not_trigger_judge(self):
        parent,out,paths,api=self.firstpass(malformed=True,failure_stage='reader')
        self.assertEqual(api.calls,['reader'])
        self.assertEqual(runner.verify(paths['reader'])['status'],'failed')
        self.assertEqual(runner.verify(paths['judge'])['status'],'not_run')
        self.assertEqual(runner.recovery_jobs(parent,out),[])

    def test_retryable_http_requires_three_ended_attempts(self):
        parent,out,_,_=self.firstpass(statuses=(503,503,503))
        self.assertEqual(len(runner.recovery_jobs(parent,out)),1)

    def test_incomplete_transport_tail_remains_unknown(self):
        parent,out,_,_=self.firstpass(statuses=('transport_error','transport_error'))
        with self.assertRaisesRegex(ValueError,'three-attempt'):
            runner.recovery_jobs(parent,out)

    def test_wrong_intent_context_is_unknown_not_permission_to_retry(self):
        parent,out,paths,_=self.firstpass()
        row=runner.verify(paths['judge']);key=row['api_events'][0]['request_hash']
        ip=self.run/'conv_0/api/request_intents'/f'{key}.json';intent=json.loads(ip.read_bytes())
        intent['context']={**intent['context'],'query_id':'c0_q999'}
        ip.write_text(json.dumps(intent),encoding='utf8')
        with self.assertRaises(ValueError):runner.recovery_jobs(parent,out)


class R08Contracts(unittest.TestCase):
    def test_inherited_no_dispatch_rejects_same_context_ledger_activity(self):
        case = R08InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
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
        case = R08InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
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
        case = R08InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
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

    def test_source_validator_delegates_r08_schema_not_source_prefix_schema(self):
        calls = []
        def validate(path):
            calls.append(path)
            return {'schema': runner.SOURCE_SCHEMA}
        fake = SimpleNamespace(validate_source=validate)
        with patch.dict(sys.modules, {'tools.prepare_facet_discovery_r08': fake}):
            self.assertEqual(runner.source_validate('new-sealed-source.json')['schema'], runner.SOURCE_SCHEMA)
            self.assertIsInstance(calls[0], Path)
            fake.validate_source = lambda p: {'schema': source_append.SOURCE_SCHEMA}
            with self.assertRaisesRegex(ValueError, 'R08 sealed'):
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
        self.assertTrue(result['numerical_threshold_met'])
        self.assertEqual(result['successful_jev_components_counted_here'], 0)
        self.assertEqual(result['component_qualification'],'pending_actual_three_reviewer_result_consensus')
        self.assertEqual(result['independent_of_source_append'],'candidate_pending_actual_source_review')
        self.assertNotIn('independent_component_count', result)
        self.assertIn('shared_with_r03', result['component_family'])
        rows[0]['scores']['judge_correct'] = None
        invalid = summary.stopping_result(rows)
        self.assertFalse(invalid['threshold_eligible'])
        self.assertEqual(invalid['minimum_strict_gt_3pp'], 1200)
        self.assertEqual(invalid['successful_jev_components_counted_here'], 0)


if __name__ == '__main__': unittest.main()
