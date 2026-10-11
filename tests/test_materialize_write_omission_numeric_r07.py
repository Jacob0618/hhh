"""Offline technical seams. Sources and frozen functions remain unchanged."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from unittest.mock import patch
from tools import materialize_write_omission_numeric_r07 as n
from tools import r07_numeric_native_support as h
from tools import materialize_write_omission_r07 as old

spec=importlib.util.spec_from_file_location('r07_original_test_fixtures_numeric',
    old.ROOT/'tests/test_materialize_write_omission_r07.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def context(arm='cheap'):
    return {'architecture':old.SCHEMA,'conversation':0,'method':arm,'phase':'write_supplement',
            'candidate_id':'c0_s001_m00','group_id':'session_1'}


class IsolatedTests(unittest.TestCase):
    def test_clone_functions_share_isolated_globals_and_originals_unchanged(self):
        original=old.materialize_bank; gd=original.__globals__.copy()
        clone=h.isolated({'manifest':lambda r:{'different':True}})
        self.assertIsNot(clone.materialize_bank,original)
        self.assertIs(clone.materialize_bank.__globals__['manifest'],clone.manifest)
        self.assertIs(clone.audit_added_embeddings.__globals__['audit_embedding'],clone.audit_embedding)
        self.assertEqual(vars(old),gd)
        self.assertIs(old.materialize_bank,original)
    def test_windows_worker_entries_are_pickleable(self):
        for fn in (n.materialize_first_attempt,n.retrieve_conversation):
            self.assertIs(pickle.loads(pickle.dumps(fn)),fn)
    def test_actual_spawn_enters_child_worker_without_dispatch(self):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            future=pool.submit(n.retrieve_conversation,'unused',0,{'admission':{}},{'banks':{}},[],[])
            # Child reached the unchanged bank lookup; it never constructed an API.
            with self.assertRaises(KeyError):future.result(timeout=30)
    def test_no_new_cheap_dispatch(self):
        with self.assertRaises(ValueError):n.materialize_first_attempt('unused',0,{})
    def test_retrieval_copy_namespace_is_separate(self):
        self.assertNotEqual(n.native_folder('new',4,'jev'),n.retrieval_folder('new',4,'jev'))
    def test_numeric_projection_preserves_store_and_checks_other_invariants(self):
        b,a,r=fixtures.fixture_delta();a['entities']['e']['vector'][''][0]=.999
        before=copy.deepcopy(a)
        with patch('tools.r07_native_numeric_replay.replay_original_entities',
                   return_value={'native_search_update_replay_exact':True}):
            self.assertEqual(h.numeric_delta('original',b,a,r,'u')['added_facts'],1)
            self.assertEqual(a,before)
            a['entities']['e']['vector']['bm25']['values'][0]=9.
            with self.assertRaises(ValueError):h.numeric_delta('original',b,a,r,'u')
    def test_no_unexplained_dense_acceptance(self):
        b,a,r=fixtures.fixture_delta()
        with patch('tools.r07_native_numeric_replay.replay_original_entities',
                   return_value={'native_search_update_replay_exact':False}):
            with self.assertRaises(ValueError):h.numeric_delta('original',b,a,r,'u')


class CacheTests(unittest.TestCase):
    def fixture(self,base):
        source,target=base/'source',base/'target'
        original=fixtures.LocalEmbeddingAPI(source,context())
        original.embed(['Gina'],stage='mem0_embedding_add')
        # Actual paced/cache provider; full-signature source import, with no HTTP allowed.
        api=old.EmbeddingOnlyAPI(target,keys={'GEMINI_API_KEY':'offline-placeholder'},
            retries=2,context=context('jev'),cache_roots=(source/'api_cache',),gate_path=base/'gate')
        from jev_mem.local_bm25_context_v2 import forbid_network
        with forbid_network():
            for _ in range(2):api.embed(['Gina'],stage='mem0_embedding_add')
        row={'status':'added','candidate':fixtures.candidate('Gina'),'entity_trace':[{'text':'Gina'}]}
        return source,target,row
    def test_actual_provider_import_no_dispatch_and_legal_samecontext_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp))
            result=h.audit_inventory(target,source,0,{'cheap':[],'jev':[row]})
            self.assertEqual((result['physical_attempts'],result['cache_consumers'],result['imported_signatures']),(0,2,1))
            self.assertFalse(list((target/'request_intents').glob('*.json')))
            self.assertEqual(len(old.embedding_ledger(source,0)),1)
            self.assertTrue(result['raw_producer_bindings'][0]['imported_exact_request'])
    def test_changed_target_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp))
            raw=next((target/'api_cache').glob('*.json'));raw.write_bytes(raw.read_bytes()+b' ')
            with self.assertRaises(ValueError):h.audit_inventory(target,source,0,{'cheap':[],'jev':[row]})
    def test_foreign_source_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp));foreign=Path(tmp)/'foreign'
            shutil.copytree(source,foreign)
            with self.assertRaises(ValueError):h.audit_inventory(target,foreign,0,{'cheap':[],'jev':[row]})
    def test_local_intent_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp));raw=next((source/'request_intents').glob('*.json'))
            (target/'request_intents').mkdir();shutil.copy2(raw,target/'request_intents'/raw.name)
            with self.assertRaises(ValueError):h.audit_inventory(target,source,0,{'cheap':[],'jev':[row]})
    def test_undeclared_cache_context_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp));path=target/'api_calls.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['context']['candidate_id']='unrelated'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):h.audit_inventory(target,source,0,{'cheap':[],'jev':[row]})
    def test_no_import_source_producer_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,row=self.fixture(Path(tmp));path=source/'api_calls.jsonl'
            record=json.loads(path.read_text());record['status']=503
            path.write_text(json.dumps(record)+'\n')
            with self.assertRaises(ValueError):h.audit_inventory(target,source,0,{'cheap':[],'jev':[row]})


class ArtifactTests(unittest.TestCase):
    def test_runtime_pins_cannot_silently_change_between_banks(self):
        self.assertEqual(n.numeric_runtime_pins([{'native_code_sha256':{'path':'one'}},
            {'native_code_sha256':{'path':'one','other':'two'}}]),{'path':'one','other':'two'})
        with self.assertRaises(ValueError):n.numeric_runtime_pins([{'native_code_sha256':{'path':'one'}},
            {'native_code_sha256':{'path':'different'}}])
    def test_addition_records_compare_bodies_and_keep_exact_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);c=fixtures.candidate();row={'candidate':c,'memory_id':'new','status':'added',
                'native_result':{'results':[{'id':'new'}]},'entity_trace':[]}
            old.save_new(folder/'additions/0000.json',row)
            old.save_new(folder/'add_dispatch/0000.json',{'candidate':c,'memory_id':'new',
                'native_result':row['native_result'],'status':'add_persisted_before_entity_link'})
            self.assertEqual(n.addition_rows(folder,[c]),[row])
            with self.assertRaises(ValueError):n.addition_rows(folder,[])
    def test_source_whole_replay_not_pointer_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selected_inputs.json';expected={'jobs':[{'selected_facts':['actual']}],
                'numeric_compatibility_protocol':{'path':'p','sha256':'hash'}}
            old.save_new(path,{**expected,'created_utc':'time'})
            with patch.object(n,'_seal_value',return_value=expected):n.validate_source(path)
            with patch.object(n,'_seal_value',return_value={**expected,'jobs':[]}):
                with self.assertRaises(ValueError):n.validate_source(path)


if __name__=='__main__':unittest.main()
