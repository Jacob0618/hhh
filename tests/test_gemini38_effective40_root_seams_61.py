"""Independent offline review of root's effective40 retrieval/QA/summary seams."""
import base64
import copy
import json
import hashlib
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools import prepare_gemini38_hybrid_effective40_v1 as retrieval
from tools import run_gemini38_plain_qa_effective40_v1 as qa
from tools import summarize_gemini38_effective40_v1 as summary
from jev_mem.local_bm25_context_v2 import forbid_network


class Snapshot:
    def __init__(self):self.files={}
    def raw(self,path):
        raw=Path(path).read_bytes();self.files[str(path)]=summary.original.sha(path);return raw
    def read(self,path):return json.loads(self.raw(path))
    def sha(self,path):self.raw(path);return self.files[str(path)]


class SeamTests(unittest.TestCase):
    def native_fixture(self, base):
        native=Path(base)/'native'
        for name in ('history.sqlite','qdrant/collection/facts/storage.sqlite','api_calls.jsonl',
                     'api_cache/known.json','request_intents/known.json','session_audit/session1.json','qdrant/.lock'):
            p=native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture-original')
        return native

    def test_stateless_fullinventory_workingcopy_changes_and_history_map_unchanged(self):
        self.assertIs(retrieval.inventory.__code__,retrieval.admission.full_native_inventory.__code__)
        self.assertIs(retrieval._scope['inventory'],retrieval.inventory)
        with tempfile.TemporaryDirectory() as d:
            native=self.native_fixture(d)
            before_map=copy.deepcopy(retrieval.admission.idx.HASHES)
            before=retrieval.inventory(native)
            self.assertTrue({'history.sqlite','api_calls.jsonl','api_cache/known.json','request_intents/known.json',
                             'session_audit/session1.json'}<=set(before))
            self.assertNotIn('qdrant/.lock',before)
            (native/'history.sqlite').write_bytes(b'fixture-workingcopy-open-close-change')
            after=retrieval.inventory(native)
            self.assertNotEqual(before['history.sqlite'],after['history.sqlite'])
            self.assertEqual(after['api_cache/known.json'],before['api_cache/known.json'])
            self.assertEqual(before_map,retrieval.admission.idx.HASHES)
            # The frozen core explicitly compares source inventory with sealed
            # inventory, so fresh hashes still detect immutable source changes.
            self.assertNotEqual(before,after)

    def test_fullinventory_guards_nonempty_wal_symlink_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            native=self.native_fixture(d)
            wal=native/'qdrant/collection/facts/storage.sqlite-wal';wal.write_bytes(b'committed-uncheckpointed')
            with self.assertRaises(Exception):retrieval.inventory(native)
            wal.write_bytes(b'')
            self.assertIn('qdrant/collection/facts/storage.sqlite-wal',retrieval.inventory(native))
            (native/'history.sqlite').unlink()
            with self.assertRaises(Exception):retrieval.inventory(native)
            fake_link=SimpleNamespace(is_symlink=lambda:True)
            with patch.object(Path,'rglob',return_value=[fake_link]):
                with self.assertRaisesRegex(Exception,'Symlink native source refused'):retrieval.inventory(native)

    def test_stateful_inventory_reproduces_old_cache_false_rejection_without_global_clear(self):
        with tempfile.TemporaryDirectory() as d:
            native=self.native_fixture(d);hashes={}
            def stateful(path):
                path=str(Path(path).resolve());value=hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if path in hashes and hashes[path]!=value:raise ValueError('Source changed: '+path)
                hashes[path]=value;return value
            # Same admission inventory code object, a private test hash map:
            # reproduce idx.sha behavior without clearing its real HASHES.
            inventory=retrieval.clone(retrieval.admission.full_native_inventory,
                                      {**retrieval.admission.full_native_inventory.__globals__,
                                       'idx':SimpleNamespace(sha=stateful)},'stateful_fixture_inventory')
            inventory(native);(native/'history.sqlite').write_bytes(b'changed-copy')
            with self.assertRaisesRegex(ValueError,'Source changed'):inventory(native)
            retrieval.inventory(native)

    def test_retrieval_core_codeobjects_and_original_globals(self):
        for name in ('tag_worker','retrieve_worker','run_stage','validate_source','manifest','seal','load_tags'):
            if name not in retrieval._scope:continue
            self.assertIs(retrieval._scope[name].__code__,getattr(retrieval.original,name).__code__)
            self.assertIs(retrieval._scope[name].__globals__,retrieval._scope)
            self.assertIsNot(retrieval._scope[name].__globals__,vars(retrieval.original))
        self.assertIs(retrieval._scope['admit_gate'],retrieval.admission.admit_gate)
        self.assertIs(retrieval._scope['_fact_bank'],retrieval.admission.validate_fact_bank)
        self.assertEqual(retrieval.original.SCHEMA,'gemini38-fresh-seven-arm-retrieval-v1')
        self.assertNotEqual(retrieval.SCHEMA,retrieval.original.SCHEMA)
        self.assertIs(retrieval._scope['writer'],retrieval.original.writer)

    def test_spawn_pickle_retrieval_workers_no_worker_execution(self):
        for name in ('tag_worker','retrieve_worker'):
            function=getattr(retrieval,name)
            self.assertIs(pickle.loads(pickle.dumps(function)),function)
            encoded=base64.b64encode(pickle.dumps(function)).decode()
            code=("import pickle,base64;from jev_mem.local_bm25_context_v2 import forbid_network;"
                  "from unittest.mock import patch;\n"
                  "with forbid_network() as guards,patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credential')):\n"
                  f" f=pickle.loads(base64.b64decode('{encoded}'));assert f.__module__=='tools.prepare_gemini38_hybrid_effective40_v1';"
                  "assert f.__globals__['SOURCE_SCHEMA']=='gemini38-effective40-seven-arm-selected-inputs-v1';"
                  "assert not any(g.call_count for g in guards)\n")
            process=subprocess.run([str(retrieval.ROOT/'.venv/Scripts/python.exe'),'-c',code],cwd=retrieval.ROOT,
                                   text=True,capture_output=True)
            self.assertEqual(process.returncode,0,process.stderr)

    def test_qa_scope_fixed_evaluator_and_no_original_mutation(self):
        original=qa.original
        for name in ('one','run','manifest','reader_call','known_failure','recovery_jobs','prepare_recovery','_new_manifest'):
            self.assertIs(qa._impl[name].__code__,original._impl[name].__code__)
            self.assertIs(qa._impl[name].__globals__,qa._impl)
        self.assertIs(qa._impl['_impl'],qa._impl)
        self.assertIsNot(original._impl,qa._impl)
        self.assertIs(qa._impl['read_answer'],original._impl['read_answer'])
        self.assertIs(qa._impl['judge_answer'],original._impl['judge_answer'])
        self.assertIs(qa._impl['PlainTextEvaluationAPI'],original._impl['PlainTextEvaluationAPI'])
        self.assertIs(qa._impl['audit_stage'],original._impl['audit_stage'])
        self.assertEqual(qa.POLICY,original.POLICY)
        self.assertEqual(qa.ARMS,original.ARMS)
        self.assertEqual(original._impl['SCHEMA'],original.SCHEMA)
        self.assertIs(qa.main.__globals__['prepare'],qa.prepare)
        self.assertIs(qa.main.__globals__['run'],qa.run)

    def test_qa_prepare_source_requires_effective_schema_and_full_cohort(self):
        with patch.object(qa.retrieval,'validate_source',return_value={'schema':qa.original.SOURCE_SCHEMA}):
            with self.assertRaisesRegex(ValueError,'effective40'):qa.source_validate(Path('mock'))
        jobs=[{'method':arm,'query_id':f'q{i}'} for arm in qa.ARMS for i in range(1540)]
        source={'schema':qa.SOURCE_SCHEMA,'jobs':jobs}
        with patch.object(qa.retrieval,'validate_source',return_value=source), \
             patch.dict(qa._impl,{'_new_manifest':lambda *args,**kw:kw}):
            result=qa.prepare(Path('mock-new'),Path('mock-source'))
            self.assertEqual(len(result['jobs']),10780)
            self.assertEqual(result['phase_kind'],'firstpass')
        with patch.object(qa.retrieval,'validate_source',return_value={**source,'jobs':jobs[:-1]}):
            with self.assertRaisesRegex(ValueError,'1540'):qa.prepare(Path('mock-new'),Path('mock-source'))

    def test_summary_score_code_and_runner_binding(self):
        for name in ('audit_phase','record_rows','aggregate','readiness','summarize','main'):
            self.assertIs(summary._scope[name].__code__,getattr(summary.original,name).__code__)
            self.assertIs(summary._scope[name].__globals__['runner'],qa)
        self.assertIs(summary._scope['precomputation'],summary.precomputation)
        self.assertIs(summary._scope['ledger'].__globals__['runner'],qa)

    def test_recovery_accounting_excludes_imports_keeps_unknowns(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); native=root/'new/native';native.mkdir(parents=True)
            origin=root/'old.jsonl';review=root/'gate.json';review.write_text('{}')
            model='gemini-3.8-flash'
            base={'model':model,'context':{'conversation':3,'method':'session'},'stage':'mem0_extract'}
            old={**base,'request_hash':'old','status':200,'cache_hit':False,'usage':{'promptTokenCount':100,'candidatesTokenCount':10,'thoughtsTokenCount':5,'totalTokenCount':115}}
            cached={**old,'cache_hit':True,'usage_semantics':'reused_response_no_new_model_usage'}
            new={**base,'request_hash':'new','status':200,'cache_hit':False,'usage':{'promptTokenCount':20,'candidatesTokenCount':2,'totalTokenCount':22}}
            unknown={**base,'request_hash':'unknown','status':'transport_error','cache_hit':False,'usage_unknown':True}
            origin.write_text(json.dumps(old)+'\n');ledger=native/'api_calls.jsonl';ledger.write_text('\n'.join(json.dumps(x)for x in(cached,new,unknown)))
            retrieval_manifest=root/'manifest.json';retrieval_manifest.write_text(json.dumps({'source_review':{'path':str(review),'sha256':summary.original.sha(review)}}))
            source={'retrieval_manifest':{'path':str(retrieval_manifest)},'parent_run':str(root/'parent')}
            prior={'usage':summary.original.measured_usage([old]),'ledger_sources':[{'path':str(origin),'sha256':summary.original.sha(origin)}],
                   'note':'original failed Writer retained','intent_inventory':{}}
            row={'conversation':3,'arm':'session','source_kind':'io_recovery_complete','original_status':'failed','strict_passed':True}
            admitted={(3,'session'):{'native':native}}
            with patch.object(summary.original,'precomputation',return_value=copy.deepcopy(prior)), \
                 patch.object(summary.retrieval.admission,'admit_gate',return_value=({}, {'banks':[row]}, admitted)), \
                 patch.object(summary.original,'intent_inventory',return_value={'unknown_intent_count':1}):
                result=summary.precomputation(Snapshot(),source)
            usage=result['usage']
            self.assertEqual(usage['physical_attempts'],3)
            self.assertEqual(usage['cache_events'],1)
            self.assertEqual(usage['token_fields']['prompt']['recorded_token_sum'],120)
            self.assertEqual(usage['token_fields']['thinking']['missing_attempts'],2)
            self.assertEqual(usage['usage_unknown_attempts'],1)
            self.assertIsNone(usage['total_cost_usd'])
            self.assertEqual(result['writer_recovery']['inherited_cache_events'],1)
            self.assertEqual(result['writer_recovery']['intent_inventory'][str(native)]['unknown_intent_count'],1)
            for kind in ('wrong_context','duplicate_ledger','changed_gate_sha'):
                with self.subTest(kind=kind):
                    copied=copy.deepcopy(prior);rows=[row]
                    ledger.write_text('\n'.join(json.dumps(x)for x in(cached,new,unknown)))
                    if kind=='wrong_context':
                        bad={**new,'context':{'conversation':4,'method':'session'}}
                        ledger.write_text(json.dumps(bad))
                    if kind=='duplicate_ledger':rows=[row,row]
                    if kind=='changed_gate_sha':review.write_text('{"changed":true}')
                    with patch.object(summary.original,'precomputation',return_value=copied), \
                         patch.object(summary.retrieval.admission,'admit_gate',return_value=({}, {'banks':rows}, admitted)), \
                         patch.object(summary.original,'intent_inventory',return_value={}):
                        with self.assertRaises(summary.original.InvalidArtifact):summary.precomputation(Snapshot(),source)


if __name__=='__main__':unittest.main()
