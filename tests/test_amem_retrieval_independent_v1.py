"""Independent negative gates; temporary data only, no process/model/API launch."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np
from tools import baselines_amem_retrieval_v1 as adapter
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import BaselineFatal, sha


class RetrievalGates(unittest.TestCase):
    def fixture(self, temp, vectors=False):
        bank=Path(temp)/'bank';bank.mkdir()
        state={'build_complete':True,'work_dir':str(bank),'corpus':['Do not encode this corpus'],
               'embeddings_sha256':'0'*64,'memory_sha256':'a'*64}
        if vectors:
            (bank/'cache').mkdir();np.save(bank/'cache/embeddings.npy',np.ones((1,384),dtype=np.float32),allow_pickle=False)
            state['embeddings_sha256']=sha(bank/'cache/embeddings.npy')
        path=bank/'state.json';path.write_text(json.dumps(state),encoding='utf8')
        return path,state,Path(temp)/'new_retrieval'

    def invoke(self,path,state,out):
        return adapter.retrieve('fixed question',path,out,object(),state_sha256=sha(path),
                                embeddings_sha256=state['embeddings_sha256'])

    def test_missing_retained_vectors_cannot_reencode_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path,state,out=self.fixture(temp)
            with patch.object(adapter.frozen._runtime,'_get_encoder',side_effect=AssertionError('encoder forbidden')) as encoder:
                with self.assertRaisesRegex(BaselineFatal,'vectors required unchanged'):
                    self.invoke(path,state,out)
                encoder.assert_not_called()
            self.assertFalse(out.exists())

    def test_matching_digest_invalid_numeric_bytes_are_rejected_before_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            path,state,out=self.fixture(temp)
            vector=path.parent/'cache/embeddings.npy';vector.parent.mkdir();vector.write_bytes(b'invalid numeric bytes')
            state['embeddings_sha256']=sha(vector);path.write_text(json.dumps(state),encoding='utf8')
            with patch.object(adapter.frozen._runtime,'_get_encoder',side_effect=AssertionError('encoder forbidden')) as encoder:
                with self.assertRaisesRegex(BaselineFatal,'numeric embedding cache is invalid'):
                    self.invoke(path,state,out)
                encoder.assert_not_called()
            self.assertFalse(out.exists())

    def test_old_bank_target_and_repeat_target_rejected_without_mutating_frozen_globals(self):
        with tempfile.TemporaryDirectory() as temp:
            path,state,out=self.fixture(temp,True)
            restore=adapter.frozen._runtime._restore;retrieve=adapter.frozen._runtime._retrieve
            state_bytes=path.read_bytes();vector_bytes=(path.parent/'cache/embeddings.npy').read_bytes()
            with patch.object(adapter.frozen._runtime,'_get_encoder',side_effect=AssertionError('encoder forbidden')) as encoder:
                for target in (path.parent,path.parent/'queries',path.parent.parent):
                    with self.assertRaisesRegex(BaselineFatal,'separate from the immutable bank'):
                        self.invoke(path,state,target)
                out.mkdir()
                with self.assertRaises(FileExistsError):self.invoke(path,state,out)
                encoder.assert_not_called()
            self.assertIs(adapter.frozen._runtime._restore,restore)
            self.assertIs(adapter.frozen._runtime._retrieve,retrieve)
            self.assertEqual(path.read_bytes(),state_bytes)
            self.assertEqual((path.parent/'cache/embeddings.npy').read_bytes(),vector_bytes)


class GuardGates(unittest.TestCase):
    def processes(self,entry):
        tail=f' -m tools.{entry} run --method amem'
        return [
            {'ProcessId':10,'ParentProcessId':0,'Name':'python.exe','CommandLine':'"parent"'+tail,
             'ExecutablePath':str(guard.frozen.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'),'CreatedUtc':'2026-10-09T00:00:00+00:00'},
            {'ProcessId':11,'ParentProcessId':10,'Name':'python.exe','CommandLine':'"actual"'+tail,
             'ExecutablePath':sys._base_executable,'CreatedUtc':'2026-10-09T00:00:01+00:00'}]

    def test_only_new_verified_parent_chain_is_excluded(self):
        for entry in ('run_five_baselines_amem_retrieval_v1','run_five_baselines_reviewed_chain_v3'):
            values=self.processes(entry)
            self.assertEqual(guard.active_controllers(values,11),[])
            other={**values[0],'ProcessId':20,'ParentProcessId':0}
            self.assertEqual([x['ProcessId'] for x in guard.active_controllers(values+[other],11)],[20])
            values[0]['ExecutablePath']='C:\\not-authorized\\python.exe'
            self.assertEqual([x['ProcessId'] for x in guard.active_controllers(values,11)],[10])

    def test_unknown_basename_and_different_parent_tail_cannot_be_ignored(self):
        values=self.processes('run_five_baselines_unapproved')
        self.assertEqual([x['ProcessId'] for x in guard.active_controllers(values,11)],[10])
        values=self.processes('run_five_baselines_amem_retrieval_v1')
        values[0]['CommandLine'] += ' --unreviewed'
        self.assertEqual([x['ProcessId'] for x in guard.active_controllers(values,11)],[10])


class UnionLedgerGates(unittest.TestCase):
    def setUp(self):
        # Reuse only the author's synthetic data factory; the actual new audit
        # function and our independent mutations remain unmocked.
        from tests.test_amem_retrieval_union_v1 import UnionTests
        self.fixture=UnionTests();self.fixture.setUp()
        from tools import seal_amem_retrieval_continuation_v1
        self.union=seal_amem_retrieval_continuation_v1

    def tearDown(self):self.fixture.tearDown()

    def test_nonretryable_http_cannot_precede_a_success(self):
        f=self.fixture;p=str(f.work/'work/api_firstpass/api_calls.jsonl')
        rows=[json.loads(x) for x in f.docs[p].splitlines()]
        bad={**rows[0],'status':400,'attempt':1};rows[0]['attempt']=2
        f.docs[p]=''.join(json.dumps(x)+'\n' for x in [bad,*rows]).encode()
        with self.assertRaises(ValueError):self.union.audit_qa_requests(f.view(),recovered=False)

    def mutate_reader_event(self,change):
        f=self.fixture;p=str(f.work/'results/firstpass/c2_q42/reader.json')
        stage=json.loads(f.docs[p]);change(stage['events'][0]);f.put(p,stage)
        ledger=str(f.work/'work/api_firstpass/logical_events.jsonl')
        rows=[json.loads(x) for x in f.docs[ledger].splitlines()]
        change(rows[0]);f.docs[ledger]=''.join(json.dumps(x)+'\n' for x in rows).encode()

    def test_cache_origin_claim_cannot_replace_fresh_qa_producer(self):
        self.mutate_reader_event(lambda e:e.update(source='existing_run_cache'))
        with self.assertRaises(ValueError):self.union.audit_qa_requests(self.fixture.view(),recovered=False)

    def test_same_raw_bytes_from_foreign_logical_cache_path_are_refused(self):
        f=self.fixture;p=str(f.work/'results/firstpass/c2_q42/reader.json')
        event=json.loads(f.docs[p])['events'][0]
        foreign=str(f.work/'work/duplicate_raw'/(event['request_hash']+'.json'))
        f.docs[foreign]=f.docs[event['cache_path']]
        self.mutate_reader_event(lambda e:e.update(cache_path=foreign))
        with self.assertRaises(ValueError):self.union.audit_qa_requests(f.view(),recovered=False)


class ReplacementAuthorization(unittest.TestCase):
    def setUp(self):
        from tools import run_five_baselines_amem_retrieval_v1 as entry
        self.entry=entry
        qpath=entry.ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261008T2147/independent_single_query_failure_qualification_v2.json'
        q=json.loads(qpath.read_text(encoding='utf8'))
        self.proof={'query_id':entry.QID,'replacement':q,
                    'original_failure':{'raw':{'path':q['raw_path'],'sha256':q['raw_sha256']}},
                    'replacement_certificate':{'path':str(qpath),'sha256':sha(qpath)}}
        self.temp=tempfile.TemporaryDirectory(prefix='independent_keyword_quota_')
        self.owner=Path(self.temp.name)/'owner';self.owner_patch=patch.object(entry,'WORK_OWNER',self.owner);self.owner_patch.start()
        self.constructor_calls=[]
        outer=self
        def fake_init(obj,run_dir,*,context,**kwargs):
            outer.constructor_calls.append(str(run_dir));obj.run_dir=Path(run_dir);obj.context=context;obj.events=[]
        self.init_patch=patch.object(entry.BaselineAPI,'__init__',fake_init);self.init_patch.start()

    def tearDown(self):
        self.init_patch.stop();self.owner_patch.stop();self.temp.cleanup()

    def api(self,**changes):
        values={'run_dir':self.owner/'work/api_method','proof':self.proof,'context':deepcopy(self.entry.CONTEXT)}
        values.update(changes);return self.entry.ReplacementAPI(**values)

    def request(self,api,**changes):
        sig=deepcopy(self.proof['replacement']['signature'])
        values={**sig,'stage':'amem_query_keywords'};values.update(changes)
        return api.request(**values)

    def test_wrong_owner_context_history_or_retry_rejected_before_base_constructor(self):
        for changes in ({'run_dir':self.owner/'other'},
                        {'context':{**self.entry.CONTEXT,'query_id':'c2_q41'}},
                        {'cache_roots':[self.owner/'history']},{'retries':3},
                        {'proof':{**self.proof,'query_id':'c2_q41'}}):
            with self.assertRaises(BaselineFatal):self.api(**changes)
        self.assertEqual(self.constructor_calls,[])
        self.assertFalse(self.owner.exists())

    def test_changed_or_writer_request_cannot_consume_quota_or_reach_transport(self):
        api=self.api();body=deepcopy(self.proof['replacement']['signature']['body'])
        body['generationConfig']['maxOutputTokens']=8192
        with patch.object(self.entry.BaselineAPI,'request',side_effect=AssertionError('transport forbidden')) as transport:
            for changes in ({'body':body},{'stage':'amem_analysis'},{'cache':False},{'model':'gemini-2.0-flash'}):
                with self.assertRaises(BaselineFatal):self.request(api,**changes)
            transport.assert_not_called()
        self.assertFalse(api.marker.exists())

    def test_durable_marker_precedes_transport_and_any_second_or_unknown_retry_is_refused(self):
        api=self.api();seen=[]
        def fail_after_marker(*args,**kwargs):
            self.assertTrue(api.marker.is_file())
            marker=json.loads(api.marker.read_text(encoding='utf8'))
            self.assertEqual({k:v for k,v in marker.items() if k!='created_utc'},self.entry.marker_value(self.proof))
            seen.append('transport once after durable marker')
            raise RuntimeError('synthetic unknown transport outcome')
        with patch.object(self.entry.BaselineAPI,'request',side_effect=fail_after_marker):
            with self.assertRaisesRegex(RuntimeError,'synthetic unknown'):self.request(api)
            with self.assertRaises(BaselineFatal):self.request(api)
            with self.assertRaises(BaselineFatal):self.api()
        self.assertEqual(seen,['transport once after durable marker'])
        self.assertEqual(len(self.constructor_calls),1)


if __name__=='__main__':unittest.main()
