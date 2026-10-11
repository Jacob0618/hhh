"""Independent, zero-network pipeline/order and scientific replay review."""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gemini25_ablation_pipeline_v1 as pipeline
from tools import prepare_gemini25_ablation_v1 as prep
from tools import score_gemini25_ablation_v1 as score
from tools import audit_gemini25_ablation_banks_v1 as banks
from tests import test_module_stack_dispatch_v1 as dispatch
from tests.test_r03_r05_source_wiring_v1 import fixture, rows_for


class PipelineTests(unittest.TestCase):
    def execute(self, folder, *, failures=(), unresolved_phase=None, seal_error=False):
        paths = [Path(folder)/name for name in ('writer', 'source', 'qa', 'live')]
        events = []

        def call(name, result=None):
            def inner(*args, **kwargs):
                events.append(name)
                return {} if result is None else result
            return inner

        def results(reader, run, phase, *, allow_recovery=True):
            events.append(phase + ('_audit_final' if allow_recovery else '_audit_first'))
            if phase == unresolved_phase:
                raise ValueError('Unknown request intent')
            return {'request': {'status': 'failed' if phase in failures else 'complete'}}

        def run_phase(run, phase, *, recovery=False):
            return call(phase + ('_recovery' if recovery else '_firstpass'))()

        def qa_run(run):
            return call('qa_recovery' if str(run).endswith('_recovery_v1') else 'qa_firstpass')()

        def seal(run):
            events.append('qa_raw_audit')
            if seal_error:
                raise ValueError('Unknown request intents cannot be sealed')
            return {'audit_status': 'no_invalid_artifacts'}

        with ExitStack() as stack:
            for target, name, value in (
                (banks, 'admit_review', call('bank_gate')),
                (prep, 'pointer', lambda p: {'path': str(p), 'sha256': 'fixture'}),
                (pipeline, 'file_sha', lambda p: 'fixture'),
                (prep, 'prepare', call('source_prepare')),
                (prep, 'retrieve', call('retrieve')),
                (prep, 'prepare_phase', lambda run, phase: call(phase+'_prepare')()),
                (score, 'run_phase', run_phase),
                (score, 'phase_results', results),
                (prep, 'seal', call('source_seal')),
                (pipeline.qa, 'prepare', call('qa_prepare')),
                (pipeline.qa, 'run', qa_run),
                (pipeline.qa, 'prepare_recovery', call('qa_recovery_prepare')),
                (pipeline.qa, 'seal', seal),
                (pipeline.report, 'summarize', call('summary', {'audit_status':'no_invalid_artifacts'})),
                (pipeline, 'precomputation_usage', call('usage', {'prices_inferred':False})),
                (pipeline.report.original.historical_guard.historical, 'offline', nullcontext),
            ):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch('requests.Session.post', side_effect=AssertionError('Network forbidden')))
            with redirect_stdout(StringIO()):
                try:
                    result = pipeline.run(*paths)
                except Exception as exc:
                    result = exc
        status = json.loads((paths[-1]/'status.json').read_text(encoding='utf8'))
        return events, result, status

    def test_full_order_uniform_recovery_and_raw_seal_before_summary(self):
        with tempfile.TemporaryDirectory() as d:
            events, result, status = self.execute(d, failures=('support','rank'))
        expected = ['bank_gate','source_prepare','retrieve']
        for phase in prep.PHASES:
            expected += [phase+'_prepare', phase+'_firstpass', phase+'_audit_first']
            if phase in ('support','rank'):
                expected += [phase+'_recovery']
            expected += [phase+'_audit_final']
        expected += ['source_seal','qa_prepare','qa_firstpass','qa_recovery_prepare',
                     'qa_recovery','qa_raw_audit','summary','usage']
        self.assertEqual(events, expected)
        self.assertEqual(result['status'], 'completed_awaiting_research_discussion')
        self.assertEqual(status['status'], result['status'])

    def test_unknown_source_result_stops_before_recovery_or_next_phase(self):
        with tempfile.TemporaryDirectory() as d:
            events, result, status = self.execute(d, unresolved_phase='support')
        self.assertIsInstance(result, ValueError)
        self.assertEqual(status['status'], 'needs_review')
        self.assertNotIn('support_recovery', events)
        self.assertNotIn('multi_compare_prepare', events)
        self.assertNotIn('qa_prepare', events)

    def test_recovery_orphan_intent_seal_blocks_successful_summary(self):
        with tempfile.TemporaryDirectory() as d:
            events, result, status = self.execute(d, seal_error=True)
        self.assertIsInstance(result, ValueError)
        self.assertEqual(events[-1], 'qa_raw_audit')
        self.assertNotIn('summary', events)
        self.assertEqual(status['status'], 'needs_review')

    def test_usage_physical_only_unknown_fields_no_tariff(self):
        with tempfile.TemporaryDirectory() as d:
            writer, source = Path(d)/'writer', Path(d)/'source'
            physical = {'provider':'gemini','model':'gemini-2.5-flash','status':200,
                        'cache_hit':False,'usage':{'promptTokenCount':7}}
            cached = {**physical,'cache_hit':True,'usage_semantics':'reused_response_no_new_model_usage'}
            for ci in range(10):
                for path in (writer/f'conv_{ci}/session/native/api_calls.jsonl',
                             source/f'conv_{ci}/retrieval_api/api_calls.jsonl'):
                    path.parent.mkdir(parents=True)
                    path.write_text(''.join(json.dumps(r)+'\n' for r in (physical,cached)),encoding='utf8')
            value = pipeline.precomputation_usage(writer,source)
        self.assertFalse(value['prices_inferred'])
        self.assertFalse(value['historical_cache_inherited'])
        self.assertEqual(len(value['ledger_refs']),20)
        self.assertEqual(value['total']['physical_attempts'],20)
        self.assertEqual(value['total']['cache_events'],20)
        self.assertEqual(value['total']['token_fields']['prompt']['recorded_token_sum'],140)
        self.assertEqual(value['total']['token_fields']['output']['missing_attempts'],20)
        self.assertTrue(value['total']['missing_thinking_is_not_zero'])
        self.assertIsNone(value['total']['total_cost_usd'])


class ScientificGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.m, oldbase, self.sources, _, _ = fixture()
        self.facts = oldbase['pools']['c0_q0']['original_r03']
        self.q = {**oldbase['original_jobs'][0], 'records':deepcopy(self.facts),
                  'retrieval':{'path':'fresh.retrieval'},'category':1}
        self.results = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(prep,'manifest',return_value=self.m))
        self.stack.enter_context(patch.object(prep,'inputs',return_value=(self.m,self.sources,[self.q])))
        self.stack.enter_context(patch.object(prep,'phase_results',side_effect=lambda reader,run,phase: self.results[phase]))
        self.stack.enter_context(patch('requests.Session.post',side_effect=AssertionError('Network forbidden')))
        prep.save_new(self.run/'run_manifest.json',self.m)
        for phase in prep.PHASES:
            prep.prepare_phase(self.run,phase)
            pm = prep.Reader().read(self.run/'phases'/phase/'manifest.json')
            requests = [prep.Reader().bound(ref) for ref in pm['requests']]
            self.results[phase] = rows_for(requests)

    def test_exact_source_support_multi_closure_rank_and_three_views(self):
        reader = prep.Reader()
        checked = set()
        prep.validate_phase(self.run,'rank',reader=reader,checked=checked)
        self.assertEqual(checked,set(prep.PHASES))
        support = prep.units(reader,self.run,'support')
        for _, unit in support:
            f = self.sources[0]['rawfacts'][unit['plan']['fact']['id']]
            expected = prep.stack.build_support_plan(prep.canonical(f),f['input_scope'],self.sources[0]['turns'],conversation=0)
            self.assertEqual(unit['plan'],expected)
        multi = prep.units(reader,self.run,'multi_compare')[0][1]
        closure = prep.units(reader,self.run,'closure')[0][1]
        rank = prep.units(reader,self.run,'rank')[0][1]
        self.assertEqual(multi['records'],self.facts)
        expected = prep.r05.resolve_multi(multi['plan'],
            prep.merged_scores(multi['request_ids'],self.results['multi_compare'])[0],score_status='complete')['locator']
        self.assertEqual(closure['locator'],expected)
        self.assertEqual(rank['records'],self.facts)
        for phase in prep.PHASES:
            pm=reader.read(self.run/'phases'/phase/'manifest.json')
            for ref in pm['requests']:
                self.assertNotIn('FORBIDDEN_GOLD_QA',json.dumps(reader.bound(ref)['signature']))
        selections = list(prep.selections(reader,self.run))
        self.assertEqual([arm for _,arm,_ in selections],list(prep.ARMS))
        for _,arm,value in selections:
            self.assertEqual(len(value['selected_facts']),20)
            self.assertEqual(value['selected_ids'],[f['id'] for f in self.facts])
            for got,old in zip(value['selected_facts'],self.facts):
                self.assertEqual(got['observation_date'],old['observation_date'])
                self.assertTrue(got['text'].startswith(old['text']))
        full = selections[-1][-1]
        self.assertLessEqual(sum(a['text']!=b['text'] for a,b in zip(full['selected_facts'],self.facts)),2)

    def test_resealed_ancestor_unit_mutation_is_rejected_from_rank(self):
        pm_path=self.run/'phases/r03/manifest.json'
        pm=prep.Reader().read(pm_path)
        unit_path=Path(pm['units'][0]['path'])
        unit=prep.Reader().read(unit_path)
        unit.pop('content_sha256');unit['records'][0]['text']='Changed scientific input'
        unit_path.unlink();pm['units'][0]=prep.save_new(unit_path,unit)
        pm.pop('content_sha256');pm_path.unlink();prep.save_new(pm_path,pm)
        with self.assertRaisesRegex(ValueError,'Scientific unit'):
            prep.validate_phase(self.run,'rank')

    def test_unknown_upstream_is_never_conservative_fallback(self):
        for phase,parent in (('closure','multi_compare'),('rank','closure')):
            original=deepcopy(self.results[parent])
            self.assertTrue(original)
            first=next(iter(self.results[parent].values()))
            first.update(status='unknown',scores=None)
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                list(prep.phase_values(prep.Reader(),self.run,phase))
            self.results[parent]=original

    def test_request_router_full_r03_80_and_short_source_rank(self):
        records=[{'id':str(i),'text':f'Alice item {i}','observation_date':None} for i in range(80)]
        sig=prep.r03.build_request('boots',None,records,prep.spec())
        self.assertEqual(score.validate_request(sig),'r03')
        self.assertEqual(list(sig['body']['questions']),[f'f{i:02d}' for i in range(80)])
        rank=prep.units(prep.Reader(),self.run,'rank')[0][1]
        # Use three distinct valid anchors to exercise the short f00..f02
        # rank branch; the normal fixture deduplicates to one packet.
        candidates=[]
        for i in range(3):
            candidate=deepcopy(rank['locator']['candidates'][0])
            candidate['packet_id']='fixture_packet_'+str(i)
            candidate['origin'].update(anchor_slot=i,anchor_id=self.facts[i]['id'])
            candidates.append(candidate)
        sig=prep.stack.build_rank_request('boots',None,self.facts,candidates)
        self.assertEqual(score.validate_request(sig),prep.stack.validate_request(sig))
        self.assertLess(len(sig['body']['questions']),20)


class ScorerProofTests(unittest.TestCase):
    def replay(self,name):
        with patch.object(dispatch,'prepare',prep),patch.object(dispatch,'score',score),\
             patch.object(score,'request_hash',prep.pool_core.request_hash,create=True):
            getattr(dispatch.DispatchTests(name),name)()

    def test_exact_cached_producer_consumer(self):
        self.replay('test_real_exact_cache_producer_and_consumer_raw_replay')

    def test_exhausted_transport_requires_uniform_recovery(self):
        self.replay('test_exhausted_transport_consumer_without_ledger_event_is_known_failure')

    def test_partial_or_missing_raw_unknown(self):
        self.replay('test_partial_transport_and_http200_without_raw_stay_unknown')

    def test_foreign_intent_and_hash_context_pair_rejected(self):
        self.replay('test_unmanifested_unknown_intent_must_block_whole_phase')
        self.replay('test_ledger_hash_and_context_must_be_admitted_as_a_pair')

    def test_original_executor_and_raw_verifier_code_objects(self):
        for name in ('context','classify_failure','one','verify_result'):
            fn=getattr(score,name)
            self.assertIs(fn.__code__,getattr(score.prior,name).__code__)
            self.assertEqual(fn.__globals__['STAGE'],score.STAGE)
        self.assertIs(score.one.__globals__['core'],score.core)
        self.assertIs(score.verify_result.__globals__['classify_failure'],score.classify_failure)

    def test_auth_or_resource_failure_stops_downstream(self):
        for http_status in (401,402,403):
            with self.subTest(http_status=http_status), tempfile.TemporaryDirectory() as d,\
                 patch.object(dispatch,'prepare',prep),patch.object(dispatch,'score',score):
                fixture=dispatch.Fixture(d,(http_status,))
                row,ref=fixture.execute()
                self.assertEqual(row['status'],'failed')
                fixture.terminal([fixture.ref],[row])
                with self.assertRaisesRegex(ValueError,'Resource/auth failure'):
                    score.phase_results(prep.Reader(),fixture.run,fixture.phase,allow_recovery=False)

    def test_recovery_uses_only_failed_subset_once_same_retries(self):
        with tempfile.TemporaryDirectory() as d,patch.object(dispatch,'prepare',prep),patch.object(dispatch,'score',score):
            fixture=dispatch.Fixture(d)
            other,other_ref=fixture.consumer()
            score.publish(fixture.run/'phases/support/manifest.json',{'protocol_sha256':'protocol'})
            first={fixture.req['request_id']:{'status':'complete'},other['request_id']:{'status':'failed'}}
            attempts=[]
            def fake_api(dest,*,retries):
                attempts.append(retries)
                return object()
            def fake_one(dest,ref,api,protocol):
                req=score.read(ref['path'])
                row={'request_id':req['request_id'],'conversation':req['conversation'],'status':'failed'}
                score.publish(dest/'responses'/(req['request_id']+'.json'),row)
                return row
            with patch.object(prep,'validate_phase',return_value=({'protocol_sha256':'protocol'},[fixture.ref,other_ref])),\
                 patch.object(score,'phase_results',return_value=first),\
                 patch.object(score,'GateAPI',side_effect=fake_api),patch.object(score,'one',side_effect=fake_one):
                out=score.run_phase(fixture.run,'support',recovery=True)
                self.assertEqual(out['status'],'terminal')
                self.assertEqual(out['expected_requests'],1)
                self.assertEqual(attempts,[2])
                intent=score.read(fixture.run/'phases/support/recovery_v1/execution_intent.json')
                self.assertEqual(intent['requests'],[other_ref])
                with self.assertRaises(FileExistsError):
                    score.run_phase(fixture.run,'support',recovery=True)


class InstalledSDKTests(unittest.TestCase):
    def test_installed_sdk_bytes_match_pinned_checkout_and_drift_blocks(self):
        with patch('requests.Session.post',side_effect=AssertionError('Network forbidden')):
            prep.validate_sdk()
            with patch.object(prep,'file_sha',side_effect=['changed-installed','pinned-checkout']):
                with self.assertRaisesRegex(ValueError,'Installed Memory source'):
                    prep.validate_sdk()
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'unprepared'
            for invoke in (lambda: prep.manifest(dest),lambda: prep.prepare(dest,dest,dest/'gate.json')):
                with patch.object(prep,'validate_sdk',side_effect=ValueError('SDK drift')) as gate:
                    with self.assertRaisesRegex(ValueError,'SDK drift'):
                        invoke()
                    gate.assert_called_once_with()
                self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
