"""Independent billing-continuation review; temporary artifacts and no network."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
import ast
import hashlib
import inspect
from io import StringIO
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import prepare_gemini25_ablation_v1 as source
from tools import score_gemini25_ablation_v1 as score
from tools import gemini25_billing_admission_v1 as billing
from tools import resume_gemini25_ablation_after_billing_v1 as resume


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class BillingFixture:
    """Build proofs using the actual frozen classifier, cache and executor."""
    def __init__(self, root, *, statuses=(402,), phase='r03', kind='firstpass', malformed=False):
        self.run = Path(root).resolve()
        self.phase, self.kind = phase, kind
        self.producer = self.run/'phases'/phase/kind
        self.producer.mkdir(parents=True)
        facts = [{'id':str(i), 'text':f'Alice keeps item {i}', 'observation_date':None} for i in range(20)]
        self.signature = source.r03.build_request('Which item does Alice keep?', None, facts, source.spec())
        self.request = source.request_value(self.signature, 'c0_q0', 'c0_q0', 0, phase, 'protocol')
        self.ref = source.save_new(self.run/'phases'/phase/'requests/c0_q0.json', self.request)
        self.refs = [self.ref]
        self.key = self.request['request_hash']
        self.context = score.context(self.request)
        self.intent = self.producer/'request_intents'/(self.key+'.json')
        dump(self.intent, {'request_hash':self.key, 'signature':self.signature,
                          'context':self.context, 'stage':score.STAGE, 'retries':2})
        self.events = [{'request_hash':self.key, 'context':self.context, 'stage':score.STAGE,
                        'provider':'jev', 'model':'jev-1.13.0', 'operation':'systemone',
                        'cache_hit':False, 'attempt':i+1, 'status':status,
                        'timestamp_utc':'2026-10-07T00:00:00Z'} for i, status in enumerate(statuses)]
        self.write_events()
        if statuses[-1] == 200:
            answers = {key:{'type':'noul', 'noul':True if malformed else .5}
                       for key in self.signature['body']['questions']}
            dump(self.producer/'api_cache'/(self.key+'.json'),
                 {'request':self.signature, 'response':{'model':'jev-1.13.0','answers':answers}})
        self.rows = [self.execute(self.ref)]
        self.terminal()

    def write_events(self):
        (self.producer/'api_calls.jsonl').write_text(
            ''.join(json.dumps(event)+'\n' for event in self.events), encoding='utf-8')

    def execute(self, ref):
        with patch('requests.Session.post', side_effect=AssertionError('Network forbidden')):
            api = score.GateAPI(self.producer, keys={'fixture':'synthetic_not_a_secret'},
                                gate_path=self.run/'mock_gate.json', retries=2)
            return score.one(self.producer, ref, api, 'protocol')

    def add_duplicate(self):
        req = {**self.request, 'request_id':'c0_q1', 'query_id':'c0_q1'}
        ref = source.save_new(self.run/'phases'/self.phase/'requests/c0_q1.json', req)
        self.refs.append(ref)
        self.rows.append(self.execute(ref))
        self.terminal()
        return ref

    def terminal(self):
        manifest = self.run/'phases'/self.phase/'manifest.json'
        for path in (manifest, self.producer/'execution_intent.json', self.producer/'outcome.json'):
            if path.exists():
                path.unlink()  # This fixture owns only its TemporaryDirectory.
        score.publish(manifest, {'schema':'gemini25-ablation-phase-v1','phase':self.phase,
                                 'protocol_sha256':'protocol','requests':self.refs,'units':[]})
        score.publish(self.producer/'execution_intent.json', {'schema':score.SCHEMA,
                      'phase':self.phase, 'recovery':self.kind=='recovery_v1', 'requests':self.refs,
                      'transport_retries':2, 'dispatch_code_sha256':score.sha(score.__file__),
                      'manifest_sha256':score.sha(manifest)})
        counts = {}
        for row in self.rows:
            counts[row['status']] = counts.get(row['status'],0)+1
        score.publish(self.producer/'outcome.json', {'schema':score.SCHEMA,'phase':self.phase,
                      'recovery':self.kind=='recovery_v1','status':'terminal','expected_requests':len(self.refs),
                      'terminal_requests':len(self.rows),'counts':counts,'local_errors':[]})


class AdmissionProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('requests.Session.post', side_effect=AssertionError('Network forbidden')))

    def admit_fixture(self, fixture):
        """Inject a test-only reviewed binding; production __init__ is tested separately."""
        self.stack.enter_context(patch.object(billing, 'RUN', fixture.run))
        self.stack.enter_context(patch.object(billing, 'FIRSTPASS', fixture.run/'phases/r03/firstpass'))
        self.stack.enter_context(patch.dict(billing.POLICY, {'known_failed_requests':len(fixture.refs)}))
        obj = object.__new__(billing.BillingAdmission)
        obj.review_path = self.root/'fixture_review.json'
        source.save_new(obj.review_path, {'fixture':True})
        obj.review_ref = billing.pointer(obj.review_path)
        obj.review = {'bindings':{},'technical_code_sha256':{}}
        obj.firstpass_inventory = billing.inventory(fixture.producer)
        return obj

    def test_exact_failed_producer_and_unlogged_duplicate_remain_failures(self):
        f = BillingFixture(self.root/'source')
        f.add_duplicate()
        obj = self.admit_fixture(f)
        with self.assertRaisesRegex(ValueError, 'Resource/auth'):
            billing._ORIGINAL_PHASE_RESULTS(source.Reader(), f.run, 'r03', allow_recovery=False)
        rows = obj.phase_results(source.Reader(), f.run, 'r03', allow_recovery=False)
        self.assertEqual([v['status'] for v in rows.values()], ['failed','failed'])
        self.assertTrue(all(v['scores'] is None and v['raw'] is None for v in rows.values()))
        self.assertEqual(len(f.events), 1)
        self.assertEqual(rows['c0_q1']['failure_evidence']['producer_context'], f.context)
        with self.assertRaisesRegex(ValueError, 'Uniform recovery'):
            obj.phase_results(source.Reader(), f.run, 'r03')

    def test_firstpass_resealed_result_or_extra_file_changes_binding(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        path = f.producer/'responses/c0_q0.json'
        row = score.read(path); row.pop('content_sha256'); row['error_type'] = 'Changed'
        path.unlink(); score.publish(path,row)
        with self.assertRaisesRegex(ValueError, 'firstpass bytes'):
            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)

    def test_firstpass_extra_unsealed_file_changes_closed_inventory(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        (f.producer/'extra.txt').write_text('unexpected',encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'firstpass bytes'):
            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)

    def test_newly_pinned_foreign_context_still_fails_original_proof(self):
        f = BillingFixture(self.root/'source')
        f.events.append({**f.events[0], 'context':{**f.context,'query_id':'foreign'}})
        f.write_events(); obj = self.admit_fixture(f)
        with self.assertRaises(ValueError):
            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)

    def test_newly_pinned_auth_failures_are_never_exempted(self):
        for status in (401,403):
            with self.subTest(status=status):
                f = BillingFixture(self.root/str(status),statuses=(status,))
                with patch.object(billing,'RUN',f.run),patch.object(billing,'FIRSTPASS',f.producer):
                    obj = object.__new__(billing.BillingAdmission)
                    obj.bind_review = lambda reader:None
                    obj.firstpass_inventory = billing.inventory(f.producer)
                    with self.assertRaisesRegex(ValueError,'Resource/auth'):
                        obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)

    def test_other_run_phase_or_recovery_never_permitted(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        self.assertTrue(obj.permits_402(f.run,'r03',f.producer))
        for run,phase,producer in ((self.root/'foreign','r03',f.producer),
                (f.run,'support',f.producer),(f.run,'r03',f.producer.with_name('recovery_v1'))):
            self.assertFalse(obj.permits_402(run,phase,producer))
        foreign = BillingFixture(self.root/'foreign')
        with self.assertRaisesRegex(ValueError,'Resource/auth'):
            obj.phase_results(source.Reader(),foreign.run,'r03',allow_recovery=False)

    def recovery(self, first, *, statuses=(200,), malformed=False):
        """Create the same request cohort in the sole temporary recovery namespace."""
        dest = first.producer.with_name('recovery_v1'); dest.mkdir()
        dump(dest/'request_intents'/(first.key+'.json'),score.read(first.intent))
        events = [{**first.events[0],'status':status,'attempt':i+1} for i,status in enumerate(statuses)]
        (dest/'api_calls.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
        if statuses[-1] == 200:
            answers = {key:{'type':'noul','noul':True if malformed else .5}
                       for key in first.signature['body']['questions']}
            dump(dest/'api_cache'/(first.key+'.json'),
                 {'request':first.signature,'response':{'model':'jev-1.13.0','answers':answers}})
        api = score.GateAPI(dest,keys={'fixture':'synthetic_not_a_secret'},
                            gate_path=self.root/'mock_gate.json',retries=2)
        rows = [score.one(dest,ref,api,'protocol') for ref in first.refs]
        intent = score.read(first.producer/'execution_intent.json'); intent.pop('content_sha256')
        intent['recovery'] = True
        score.publish(dest/'execution_intent.json',intent)
        outcome = score.read(first.producer/'outcome.json'); outcome.pop('content_sha256')
        outcome.update(recovery=True,counts={r['status']:sum(x['status']==r['status'] for x in rows) for r in rows})
        score.publish(dest/'outcome.json',outcome)
        return dest

    def test_successful_recovery_is_audited_and_keeps_firstpass_lineage(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        self.recovery(f)
        rows = obj.phase_results(source.Reader(),f.run,'r03')
        self.assertEqual(rows['c0_q0']['status'],'complete')
        self.assertIn('firstpass_result_ref',rows['c0_q0'])
        self.assertEqual(obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)['c0_q0']['status'],'failed')

    def test_existing_recovery_402_auth_unknown_or_zero_parse_blocks_even_firstpass_view(self):
        for statuses,malformed in [((402,),False),((401,),False),((403,),False),
                                   (('transport_error',),False),(('transport_error',)*3,False),((200,),True)]:
            with self.subTest(statuses=statuses,malformed=malformed), tempfile.TemporaryDirectory() as tmp:
                f = BillingFixture(Path(tmp)/'source')
                with patch.object(billing,'RUN',f.run),patch.object(billing,'FIRSTPASS',f.producer),\
                     patch.dict(billing.POLICY,{'known_failed_requests':1}):
                    obj = object.__new__(billing.BillingAdmission)
                    obj.bind_review = lambda reader:None
                    obj.firstpass_inventory = billing.inventory(f.producer)
                    self.recovery(f,statuses=statuses,malformed=malformed)
                    for allow in (True,False):
                        with self.assertRaises(ValueError):
                            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=allow)

    def test_other_phase_known_nonresource_failures_preserve_original_fallback(self):
        f = BillingFixture(self.root/'source',statuses=('transport_error',)*3,phase='support')
        obj = self.admit_fixture(f)
        self.recovery(f,statuses=('transport_error',)*3)
        original = billing._ORIGINAL_PHASE_RESULTS(source.Reader(),f.run,'support')
        resumed = obj.phase_results(source.Reader(),f.run,'support')
        self.assertEqual(resumed,original)
        self.assertEqual(resumed['c0_q0']['status'],'failed')
        self.assertIsNone(resumed['c0_q0']['scores'])

    def test_missing_or_local_error_recovery_blocks(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        dest = f.producer.with_name('recovery_v1'); dest.mkdir()
        with self.assertRaises(FileNotFoundError):
            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)
        dest.rmdir(); dest = self.recovery(f)
        path = dest/'outcome.json'; row=score.read(path); row.pop('content_sha256')
        row['local_errors']=[{'error_type':'OSError'}]; path.unlink(); score.publish(path,row)
        with self.assertRaisesRegex(ValueError,'audited terminal'):
            obj.phase_results(source.Reader(),f.run,'r03')

    def test_scoped_activation_dynamic_source_dispatch_and_exception_restore(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        original = score.phase_results
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with obj.activated():
                rows = source.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)
                self.assertEqual(rows['c0_q0']['status'],'failed')
                with self.assertRaisesRegex(RuntimeError,'already active'):
                    with obj.activated():
                        pass
                raise RuntimeError('intentional')
        self.assertIs(score.phase_results,original)
        with obj.activated():
            self.assertIsNot(score.phase_results,original)
        self.assertIs(score.phase_results,original)

    def test_review_file_mutation_blocks_replay_and_activation(self):
        f = BillingFixture(self.root/'source'); obj = self.admit_fixture(f)
        obj.review_path.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'review changed'):
            obj.phase_results(source.Reader(),f.run,'r03',allow_recovery=False)
        with self.assertRaisesRegex(ValueError,'review changed'):
            with obj.activated():pass

    def test_progress_requires_real_nonempty_success_but_preserves_zero_request_stage(self):
        self.assertEqual(billing.require_progress({},require_success=False)['terminal'],0)
        with self.assertRaises(ValueError):billing.require_progress({})
        for rows in ({'x':{'status':'failed'}},{'x':{'status':'unknown'}},
                     {'x':{'status':'not_dispatched'}}):
            with self.assertRaises(ValueError):billing.require_progress(rows)
        self.assertEqual(billing.require_progress({'x':{'status':'complete'},'y':{'status':'failed'}})['terminal'],2)

    def test_copied_terminal_ast_keeps_every_original_check_except_resource_gate(self):
        """Normalize namespace spelling only; the one explicitly changed gate is marked."""
        class Normalize(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                if node.module == 'tools.check_experiment_health':return None
                return node
            def visit_Name(self,node):
                if node.id == 'file_sha':node.id='sha'
                return node
            def visit_Attribute(self,node):
                self.generic_visit(node)
                if isinstance(node.value,ast.Name) and node.value.id == 'score':
                    return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
                return node
            def visit_If(self,node):
                self.generic_visit(node)
                test=ast.dump(node.test,include_attributes=False)
                if all(f'value={s}' in test for s in (401,402,403)):
                    node.body=[ast.Pass()]
                return node
        def normalized(fn):
            parsed=ast.parse(textwrap.dedent(inspect.getsource(fn)))
            terminal=next(n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name=='terminal')
            return ast.dump(Normalize().visit(terminal),include_attributes=False)
        self.assertEqual(normalized(billing._ORIGINAL_PHASE_RESULTS),normalized(billing.BillingAdmission.phase_results))


class WrapperTests(unittest.TestCase):
    def execute(self, folder, *, r03_status='complete', seal_error=False, ordinary_failed=False, twice=False):
        events=[]; root=Path(folder); active=[]
        class Guard:
            review_ref={'path':str(root/'review.json'),'sha256':'fixture'}
            review={'technical_code_sha256':{},'scientific_code_sha256':{}}
            @contextmanager
            def activated(self):
                active.append(True)
                try:yield self
                finally:active.pop()
            def phase_results(self,reader,run,phase,*,allow_recovery=True):
                assert active
                events.append(phase+('_audit_final' if allow_recovery else '_audit_first'))
                status=r03_status if phase=='r03' else ('failed' if ordinary_failed and phase=='support' else 'complete')
                return {'x':{'status':status}}
        def call(name,value=None):
            def inner(*a,**kw):
                self.assertTrue(active);events.append(name)
                return {} if value is None else value
            return inner
        def seal(*a):
            self.assertTrue(active);events.append('qa_seal')
            if seal_error:raise ValueError('fixture raw seal failure')
            return {'seal':{'path':'fixture','sha256':'fixture'},'audit_status':'no_invalid_artifacts'}
        with ExitStack() as stack:
            for name,path in (('RUN',root/'source'),('QA',root/'qa'),('LIVE',root/'live')):
                stack.enter_context(patch.object(billing,name,path))
            for target,name,value in (
                (billing,'BillingAdmission',lambda p:Guard()),
                (resume,'preflight',call('preflight',{'status':'passed'})),
                (score,'run_phase',lambda run,phase,**kw:call(phase+('_recovery' if kw.get('recovery') else '_firstpass'))()),
                (source,'prepare_phase',lambda run,phase:call(phase+'_prepare')()),
                (source,'seal',call('source_seal')),
                (resume.qa,'prepare',call('qa_prepare')),
                (resume.qa,'run',lambda run:call('qa_recovery' if str(run).endswith('_recovery_v1') else 'qa_firstpass')()),
                (resume.qa,'prepare_recovery',call('qa_recovery_prepare')),
                (resume.qa,'seal',seal),
                (resume,'offline_summary',call('summary',{'audit_status':'no_invalid_artifacts'})),
            ):stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch('requests.Session.post',side_effect=AssertionError('Network forbidden')))
            with redirect_stdout(StringIO()):
                try:result=resume.run(root/'review.json')
                except Exception as exc:result=exc
                if twice:
                    with self.assertRaises(FileExistsError):resume.run(root/'review.json')
        state=json.loads((root/'live/status.json').read_text(encoding='utf-8'))
        return events,result,state

    def test_single_recovery_then_remaining_phases_and_mandatory_qa_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            events,result,state=self.execute(tmp,ordinary_failed=True,twice=True)
        expected=['preflight','r03_recovery','r03_audit_final']
        for phase in source.PHASES[1:]:
            expected += [phase+'_prepare',phase+'_firstpass',phase+'_audit_first']
            if phase=='support':expected += [phase+'_recovery']
            expected += [phase+'_audit_final']
        expected += ['source_seal','qa_prepare','qa_firstpass','qa_recovery_prepare','qa_recovery','qa_seal','summary','preflight']
        self.assertEqual(events,expected)
        self.assertEqual(result['status'],'completed_awaiting_research_discussion')
        self.assertEqual(state['status'],result['status'])
        self.assertNotIn('r03_firstpass',events)
        self.assertEqual(resume.qa.TOTAL,4620)
        self.assertEqual(resume.qa.PER_ARM,1540)

    def test_r03_zero_success_or_unknown_never_prepares_downstream(self):
        for status in ('failed','unknown'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                events,result,state=self.execute(tmp,r03_status=status)
            self.assertIsInstance(result,ValueError)
            self.assertNotIn('support_prepare',events)
            self.assertNotIn('source_seal',events)
            self.assertNotIn('qa_prepare',events)
            self.assertEqual(state['status'],'needs_review')

    def test_qa_seal_failure_prevents_final_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            events,result,state=self.execute(tmp,seal_error=True)
        self.assertIsInstance(result,ValueError)
        self.assertEqual(events[-1],'qa_seal')
        self.assertNotIn('summary',events)
        self.assertEqual(state['status'],'needs_review')

    def test_fixed_namespaces_reject_second_recovery_or_controller(self):
        for suffix in ('phases/r03/recovery_v1','phases/support','selected_inputs.json','selections'):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/'source'/suffix;target.mkdir(parents=True)
                with patch.object(billing,'RUN',root/'source'),patch.object(billing,'QA',root/'qa'),patch.object(billing,'LIVE',root/'live'):
                    with self.assertRaises(FileExistsError):resume.unused_namespaces()
        for target_name in ('qa','qa_recovery_v1','live'):
            with self.subTest(target=target_name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/target_name).mkdir()
                with patch.object(billing,'RUN',root/'source'),patch.object(billing,'QA',root/'qa'),patch.object(billing,'LIVE',root/'live'):
                    with self.assertRaises(FileExistsError):resume.unused_namespaces()

    def test_qa_resource_gate_reads_real_conv_api_paths_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'conv_0/api/api_calls.jsonl';p.parent.mkdir(parents=True)
            for status in (401,402,403):
                p.write_text(json.dumps({'status':status,'cache_hit':False})+'\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'resource/auth'):resume.qa_resource_gate(root)
            p.write_text(json.dumps({'status':200,'cache_hit':False}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Partial QA ledger'):resume.qa_resource_gate(root)
            p.write_text(json.dumps({'status':'transport_error','cache_hit':False})+'\n',encoding='utf-8')
            self.assertEqual(resume.qa_resource_gate(root)['physical_attempts'],1)

    def test_controller_identity_allows_redirector_but_rejects_sibling_script_or_module(self):
        script='python.exe tools/resume_gemini25_ablation_after_billing_v1.py run --review fixture.json'
        own={'pid':100,'parent_pid':99,'creation_utc':'fixture','command_line':script}
        parent={'pid':99,'parent_pid':1,'creation_utc':'fixture','command_line':script}
        observer={'pid':102,'parent_pid':1,'creation_utc':'fixture','command_line':'python.exe tools/snapshot_gemini25_ablation_v1.py'}
        with patch.object(resume.os,'getpid',return_value=100),patch.object(resume,'controller_probe',return_value=[own,parent,observer]):
            self.assertEqual(resume.assert_single_controller()['duplicate_controllers'],0)
        for command in (script,'python.exe -m tools.resume_gemini25_ablation_after_billing_v1 run',
                        'python.exe -m tools.run_gemini25_ablation_pipeline_v1 run',
                        'python.exe tools/score_gemini25_ablation_v1.py'):
            other={'pid':101,'parent_pid':1,'creation_utc':'fixture','command_line':command}
            with self.subTest(command=command),patch.object(resume.os,'getpid',return_value=100),\
                 patch.object(resume,'controller_probe',return_value=[own,parent,other]):
                with self.assertRaisesRegex(ValueError,'Another current'):resume.assert_single_controller()

    def test_offline_summary_requires_current_seal_bindings_and_unknown_intent_free_report(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);qa=root/'qa';recovery=root/'qa_recovery_v1';live=root/'live'
            stack.enter_context(patch.object(billing,'QA',qa));stack.enter_context(patch.object(billing,'LIVE',live))
            stack.enter_context(patch.object(billing,'RUN',root/'source'))
            guard=SimpleNamespace(review_ref={'path':'fixture','sha256':'fixture'},review={'technical_code_sha256':{}})
            source.save_new(live/'execution_intent.json',{'schema':resume.SCHEMA,'review':guard.review_ref,'technical_code_sha256':{}})
            source.save_new(recovery/'run_manifest.json',{})
            source.save_new(recovery/'run_outcome.json',{})
            value={'audit_status':'no_invalid_artifacts','phase_intent_inventory':{},
                   'source_manifest':{'path':'source','sha256':'sourcehash'},'phase_sources':[],
                   'input_sha256':{'retained-firstpass':'bound-hash'},'summary_code_sha256':{'frozen':'codehash'}}
            seal={'schema':resume.qa.SCHEMA+'-raw-audit-seal','audit_status':value['audit_status'],
                  'run_manifest':billing.pointer(recovery/'run_manifest.json'),
                  'run_outcome':billing.pointer(recovery/'run_outcome.json'),
                  **{k:deepcopy(value[k]) for k in ('source_manifest','phase_sources','input_sha256','summary_code_sha256')},
                  'raw_replayed':True,'official_f1_recomputed':True,'custom_f1_recomputed':True,'model_calls':0}
            path=recovery/'qa_raw_audit_seal.json';source.save_new(path,seal)
            summary=stack.enter_context(patch.object(resume.report,'summarize',side_effect=lambda *a,**k:deepcopy(value)))
            usage=stack.enter_context(patch.object(resume.pipeline,'precomputation_usage',return_value={'prices_inferred':False}))
            stack.enter_context(patch('requests.Session.post',side_effect=AssertionError('Network forbidden')))
            before=path.read_bytes();got=resume.offline_summary(guard)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(got['technical_continuation']['review'],guard.review_ref)
            self.assertEqual(got['precomputation_usage'],{'prices_inferred':False})
            stale=deepcopy(seal);stale['input_sha256']={};path.unlink();source.save_new(path,stale)
            with self.assertRaisesRegex(ValueError,'seal does not bind'):resume.offline_summary(guard)
            path.unlink();source.save_new(path,seal)
            value['phase_intent_inventory']={'firstpass':{'0':{'unknown_intent_count':1,'ended_success_missing_raw_count':0}}}
            with self.assertRaisesRegex(ValueError,'Unknown/orphan'):resume.offline_summary(guard)
            self.assertEqual(usage.call_count,1)

    def test_offline_check_rejects_partial_producer_and_uses_scoped_source_validator(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);run=root/'source';active=[]
            class Guard:
                @contextmanager
                def activated(self):
                    active.append(True)
                    try:yield self
                    finally:active.pop()
                def phase_results(self,*a,**k):return {'x':{'status':'complete'}}
            stack.enter_context(patch.object(billing,'RUN',run))
            stack.enter_context(patch.object(billing,'BillingAdmission',return_value=Guard()))
            stack.enter_context(patch.object(resume,'preflight',return_value={'status':'passed'}))
            stack.enter_context(patch.object(source,'validate_phase',side_effect=lambda *a,**k:self.assertTrue(active)))
            source.save_new(run/'phases/support/manifest.json',{})
            (run/'phases/support/firstpass').mkdir()
            with self.assertRaises((ValueError,FileNotFoundError)):
                resume.check(root/'review.json')
            (run/'phases/support/firstpass').rmdir()
            source.save_new(run/'selected_inputs.json',{})
            def source_validate(path):
                self.assertTrue(active)
                return {'jobs':[{}]*4620}
            stack.enter_context(patch.object(source,'validate_source',side_effect=source_validate))
            self.assertEqual(resume.check(root/'review.json')['sealed_source_jobs'],4620)


if __name__ == '__main__':
    unittest.main()
