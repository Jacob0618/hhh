"""Offline candidate seam tests. No production execution or original native open."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

from tools import recover_gemini38_session_io_v1 as r
from jev_mem.local_bm25_context_v2 import forbid_network

RESEARCH = r.ROOT/'research/gemini38_retest_2026-10-03'
PARENT = r.ROOT/'runs/gemini38_rebuild_v1_20261003'


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.evidence = r.prefix_sources(r.pointer(RESEARCH/'failure_c3_session_61.json'),PARENT)

    def test_original_codeobjects_and_globals_unchanged(self):
        self.assertIs(r.JournalRetryAdapter.add_session.__code__,r.mem0_adapter.Mem0SessionAdapter.add_session.__code__)
        self.assertIsNot(r.JournalRetryAdapter.add_session.__globals__,r.mem0_adapter.Mem0SessionAdapter.add_session.__globals__)
        self.assertIsNot(r.mem0_adapter.Mem0SessionAdapter.add_session.__globals__['_atomic_json'],r.journal_json)
        cloned = r.clone(r.frozen.write_sessions,Mem0SessionAdapter=r.JournalRetryAdapter)
        self.assertIs(cloned.__code__,r.frozen.write_sessions.__code__)
        self.assertIs(r.frozen.write_sessions.__globals__['Mem0SessionAdapter'],r.mem0_adapter.Mem0SessionAdapter)

    def test_eighteen_real_exact_imports_standard_cache_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory)/'native'
            r.import_prefix(native,self.rows)
            guard = r.PrefixGuard(self.rows,Path(directory)/'events.jsonl')
            with forbid_network() as guards, patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credentials')):
                api = r.RecoveryAPI(native,prefix_guard=guard,keys={'GEMINI_API_KEY':'offline-placeholder'},
                                    context=self.rows[0]['context'],gate_path=Path(directory)/'unused_gate.sqlite',retries=2)
                for row in self.rows:
                    api.context = copy.deepcopy(row['context'])
                    signature = row['signature']
                    response = api.request(signature['provider'],signature['model'],signature['body'],
                                           operation=signature['operation'],stage=row['stage'])
                    self.assertEqual(response,r.read_bound(row['raw'])['response'])
                self.assertEqual(sum(x.call_count for x in guards),0)
            ledger = [json.loads(x) for x in (native/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(ledger),18)
            self.assertEqual([x['request_hash'] for x in ledger],[x['request_hash'] for x in self.rows])
            self.assertTrue(all(x['cache_hit'] and x['usage_semantics']=='reused_response_no_new_model_usage' for x in ledger))
            self.assertEqual(sum(x['stage']=='mem0_extract' for x in ledger),4)
            from tools import audit_gemini38_banks_v1 as audit
            for row in (x for x in self.rows if x['stage']=='mem0_extract'):
                body=row['signature']['body']; system=body['systemInstruction']['parts'][0]['text']
                prompt=body['contents'][0]['parts'][0]['text']
                parsed=json.loads(r.response_text(r.read_bound(row['raw'])['response']))
                journal={'builder_call_count':1,'session_id':row['context']['session_id'],
                         'llm_calls':[{'kwargs':{'messages':[{'role':'system','content':system},
                                                            {'role':'user','content':prompt}]},
                                       'response':json.dumps(parsed),'parsed_memory':parsed['memory']}]}
                checked,_=audit.check_generation(journal,ledger,native,prompt,
                                                 {'ADDITIVE_EXTRACTION_PROMPT':system},3,'session')
                self.assertEqual(checked,parsed['memory'])
            self.assertEqual(guard.position,18)
            self.assertTrue(guard.fourth_returned)
            self.assertFalse((Path(directory)/'unused_gate.sqlite').exists())
            events = [json.loads(x) for x in (Path(directory)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(x['event']=='request_returned' for x in events),18)
            self.assertTrue(all(x['source']=='inherited_exact_request' for x in events))
            self.assertTrue(all(r.sha(p)==h for p,h in self.evidence.items()))

    def test_known_request_drift_order_raw_context_hard_stop(self):
        row = self.rows[0]
        for kind in ('body','context','stage','order','raw','intent'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                guard = r.PrefixGuard(self.rows,Path(d)/'events')
                request = copy.deepcopy(row['signature']); context=copy.deepcopy(row['context']); stage=row['stage']
                if kind=='body': request['body']['extra']='drift'
                if kind=='context': context['session_id']='session_2'
                if kind=='stage': stage='mem0_extract'
                if kind=='order': request=copy.deepcopy(self.rows[1]['signature'])
                if kind in ('raw','intent'):
                    bad = copy.deepcopy(self.rows)
                    bad[0][kind]['sha256']='0'*64
                    guard.rows=bad
                with self.assertRaises(r.CacheDrift): guard.before(context,request,stage)
                self.assertEqual(guard.position,0)

    def test_after_prefix_never_reextract_and_first_add_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            guard = r.PrefixGuard(self.rows,Path(d)/'events')
            generation = self.rows[-1]
            guard.position=18
            with self.assertRaises(r.CacheDrift): guard.before(generation['context'],generation['signature'],'mem0_extract')
            guard.fourth_returned=True
            with self.assertRaises(r.CacheDrift): guard.before(generation['context'],generation['signature'],'mem0_extract')
            embedding = next(x for x in self.rows if x['stage'].startswith('mem0_embedding_add'))
            context={**embedding['context'],'session_id':'session_4'}
            self.assertIsNone(guard.before(context,embedding['signature'],embedding['stage']))
            context['session_id']='session_3'
            with self.assertRaises(r.CacheDrift): guard.before(context,embedding['signature'],embedding['stage'])

    def test_bounded_retry_only_same_bytes_not_core(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'journal.json'; attempts=[]; sleeps=[]
            def replacing(src,dst):
                attempts.append(Path(src).read_bytes())
                if len(attempts)<3: raise PermissionError('mock sharing')
                return r.os.replace(src,dst)
            r.journal_json(path,{'memory':['same']},replace=replacing,sleeper=sleeps.append)
            self.assertEqual(len(attempts),3); self.assertEqual(len(set(attempts)),1)
            self.assertEqual(sleeps,[.1,.2])
            def denied(*args): raise PermissionError('mock persistent')
            with self.assertRaises(PermissionError), patch.object(r.os,'fsync',wraps=r.os.fsync):
                r.journal_json(path,{'memory':['new']},replace=denied,sleeper=sleeps.append)
            self.assertTrue(path.with_suffix('.json.tmp').exists())
            self.assertEqual(json.loads(path.read_text())['memory'],['same'])
            count=[]
            def other(*args): count.append(1); raise OSError('not permission')
            with self.assertRaises(OSError): r.journal_json(path,{},replace=other,sleeper=sleeps.append)
            self.assertEqual(len(count),1)

    def test_hard_stop_closes_core_then_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            namespace=Path(d); folder=namespace/'conv_3/session'; folder.mkdir(parents=True)
            pm={'protocol_sha256':'scientific'}; m={'execution_recovery_sha256':'technical'}
            closed=[]
            def core(run,ci,manifest):
                try: raise r.CacheDrift('known request changed')
                finally:
                    closed.append(True)
                    r.frozen.publish_json(Path(run)/'conv_3/session/write_status.json',
                                          {'status':'running','facts':0,'finished_utc':'closed'})
            guard=r.PrefixGuard(self.rows,folder/'events')
            result=r.run_core(namespace,m,pm,guard,api_factory=lambda:None,core=core)
            self.assertTrue(closed)
            self.assertEqual(result['status'],'blocked_request_drift')
            self.assertEqual(json.loads((folder/'write_status.json').read_text())['status'],'blocked_request_drift')
            self.assertTrue((folder/'recovery_outcome.json').exists())

    def test_unknown_baseexception_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'conv_3/session'; folder.mkdir(parents=True)
            def core(*args): raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                r.run_core(Path(d),{'execution_recovery_sha256':'technical'},{'protocol_sha256':'scientific'},
                           r.PrefixGuard(self.rows,folder/'events'),api_factory=lambda:None,core=core)
            self.assertEqual(json.loads((folder/'recovery_outcome.json').read_text())['status'],'interrupted')

    def test_no_reentry_and_wrong_namespace_before_api(self):
        with tempfile.TemporaryDirectory() as d:
            ns=Path(d); r.new_json(ns/'recovery_manifest.json',{'parent_manifest':{'path':str(ns/'old/run_manifest.json')}})
            (ns/'conv_3/session/native').mkdir(parents=True)
            with patch.object(r,'closed_parent_lock',return_value=r.contextlib.nullcontext()), \
                 patch.object(r,'check',return_value=({}, {}, ns/'old')), \
                 patch.object(r,'RecoveryAPI',side_effect=AssertionError('Must not construct API')):
                with self.assertRaisesRegex(ValueError,'no reentry'): r.execute(ns)

    def test_atomic_once_intent_concurrent_and_repeat_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'intent.json'; barrier=threading.Barrier(8)
            def contender(n):
                barrier.wait()
                try: r.exclusive_intent(path,{'contender':n}); return True
                except FileExistsError: return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes=list(pool.map(contender,range(8)))
            self.assertEqual(sum(outcomes),1)
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):r.exclusive_intent(path,{'contender':'repeat'})
            self.assertEqual(before,path.read_bytes())

    @unittest.skipUnless(r.os.name=='nt','Production is Windows')
    def test_new_namespace_execution_lock_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            with r.recovery_execution_lock(Path(d)):
                with self.assertRaisesRegex(RuntimeError,'Another recovery executor'):
                    with r.recovery_execution_lock(Path(d)):self.fail('Second lock accepted')
            with r.recovery_execution_lock(Path(d)): pass

    def test_core_status_publisher_is_bounded_isolated(self):
        original=r.frozen.write_sessions.__globals__['publish_json']
        function=r.clone(r.frozen.write_sessions,publish_json=r.journal_json)
        self.assertIs(function.__globals__['publish_json'],r.journal_json)
        self.assertIs(r.frozen.write_sessions.__globals__['publish_json'],original)
        self.assertIs(function.__code__,r.frozen.write_sessions.__code__)

    def test_active_parent_and_no_uniform_review_reject_offline(self):
        with tempfile.TemporaryDirectory() as d:
            ns=Path(d)/'new'; review=Path(d)/'review.json'
            r.new_json(review,{'schema':r.REVIEW_SCHEMA,'status':'not_reviewed','qa_gold_used':False})
            with self.assertRaisesRegex(ValueError,'Uniform stage-end'):
                r.validate_review(r.pointer(review),ns,alive=lambda pid:False)

    def test_uniform_forty_contract_and_negative_gates(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); parent=base/'parent'; parent.mkdir(); ns=base/'recovery'
            manifest={'protocol_sha256':'scientific'}
            r.new_json(parent/'run_manifest.json',manifest)
            r.new_json(parent/'controller_status.json',{'terminal':True,'status':'failed',
                                                       'protocol_sha256':'scientific','pid':987654})
            policy=base/'policy.md'; policy.write_text('bound policy',encoding='utf-8')
            diagnosis=base/'diagnosis.json'; r.new_json(diagnosis,{'mock_only':True})
            rows=[]
            for ci in range(10):
                for arm in r.frozen.ARMS:
                    folder=parent/f'conv_{ci}'/arm; folder.mkdir(parents=True)
                    identity={'schema':r.frozen.SCHEMA,'protocol_sha256':'scientific','conversation':ci,'arm':arm}
                    r.new_json(folder/'group_inputs.json',r.frozen.grouped.sealed({'envelopes':[{},{}]}))
                    status='failed' if (ci,arm)==r.SLOT else 'complete'
                    state={**identity,'status':status,'pid':987655,'finished_utc':'mock',
                           'expected_groups':2,'completed_groups':1 if status=='failed' else 2,'facts':1}
                    r.new_json(folder/'write_status.json',state)
                    outcome={**identity,'status':status,'finished_utc':'mock','write_status':r.pointer(folder/'write_status.json')}
                    if status=='complete':
                        r.new_json(folder/'bank.json',r.frozen.grouped.sealed({**identity,'complete':True,'facts':[{}]}))
                        outcome['bank']=r.pointer(folder/'bank.json')
                    r.new_json(folder/'write_outcome.json',r.frozen.grouped.sealed(outcome))
                    rows.append({'conversation':ci,'arm':arm,'original_status':status,
                                 'write_status':r.pointer(folder/'write_status.json'),'outcome':r.pointer(folder/'write_outcome.json')})
            review={'schema':r.REVIEW_SCHEMA,'status':'reviewed_after_firstpass','qa_gold_used':False,
                    'parent_manifest':r.pointer(parent/'run_manifest.json'),'protocol_sha256':'scientific',
                    'policy':r.pointer(policy),'banks':rows,'failures':[{'conversation':3,'arm':'session',
                    'action':'exact_prefix_io_rebuild','approved_for_once':True,'diagnosis':r.pointer(diagnosis),
                    'recovery_namespace':str(ns.resolve())}]}
            reviewpath=base/'review.json'
            def validating(value,alive=lambda pid:False):
                reviewpath.write_text(json.dumps(value),encoding='utf-8')
                with patch.object(r.frozen,'manifest',return_value=manifest):
                    return r.validate_review(r.pointer(reviewpath),ns,alive=alive)
            validating(review)
            for kind in ('missing_bank','duplicate_bank','missing_failure','unapproved','namespace','qa_gold','alive'):
                with self.subTest(kind=kind):
                    bad=copy.deepcopy(review)
                    if kind=='missing_bank':bad['banks'].pop()
                    if kind=='duplicate_bank':bad['banks'][-1]=bad['banks'][0]
                    if kind=='missing_failure':bad['failures']=[]
                    if kind=='unapproved':bad['failures'][0]['approved_for_once']=False
                    if kind=='namespace':bad['failures'][0]['recovery_namespace']=str(base/'other')
                    if kind=='qa_gold':bad['qa_gold_used']=True
                    with self.assertRaises(ValueError): validating(bad,alive=(lambda pid:True) if kind=='alive' else (lambda pid:False))


if __name__ == '__main__':
    unittest.main()
