"""Network-blocked snapshot envelope, imported producers and spawn seam regressions."""
from __future__ import annotations
import copy,importlib.util,json,multiprocessing,pickle,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
from tools import materialize_qualifier_repair_snapshot_r09 as m
from tools import r09_snapshot_native_support as s
from tools import r09_native_update_audit as a
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from jev_mem.local_bm25_context_v2 import forbid_network
old=m.old
spec=importlib.util.spec_from_file_location('r09_original_offline_fixture',old.ROOT/'tests/test_materialize_qualifier_repair_r09.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def spawn_probe():
    return m.SCHEMA,m.original.SCHEMA,m.original.old.SCHEMA,m.materialize_one.__module__

class EnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cm=fixtures.native_update_fixture();cls.f=cls.cm.__enter__()
    @classmethod
    def tearDownClass(cls):cls.cm.__exit__(None,None,None)
    def sealed(self):
        with tempfile.TemporaryDirectory() as td:
            ref=old.save_new(Path(td)/'snapshot.json',self.f['before'])
            return old.bound(ref,sealed=True)
    def test_real_sdk_save_bound_envelope_exact_replay(self):
        f=self.f;sealed=self.sealed()
        with self.assertRaisesRegex(ValueError,'Original UPDATE snapshot differs'):
            a.replay_exact(f['original'],sealed,f['after'],f['rows'],f['resources'])
        result=s.replay_exact(f['original'],sealed,f['after'],f['rows'],f['resources'])
        self.assertTrue(result['native_full_update_store_replay_exact']);self.assertFalse(result['epsilon_used'])
        self.assertEqual(result['before_sha256'],old.digest(f['before']))
        self.assertEqual(old.inventory(f['original']),f['original_inventory'])
    def test_fixed_four_keys_and_tampered_seal_rejected(self):
        sealed=self.sealed()
        for name in ('seal','body','extra'):
            v=copy.deepcopy(sealed)
            if name=='seal':v['content_sha256']='0'*64
            elif name=='body':v['history']=[]
            else:v['foreign']=True
            with self.assertRaises(ValueError):s.snapshot_body(v)
        self.assertEqual(s.snapshot_body(sealed),self.f['before'])
    def test_changed_artifact_bytes_fail_before_body_projection(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.json';ref=old.save_new(p,self.f['before']);p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaises(ValueError):old.bound(ref,sealed=True)
    def test_zero_updates_sealed_before_exact_and_no_activity(self):
        f=self.f
        self.assertTrue(s.replay_exact(f['original'],self.sealed(),f['before'],[],f['resources'])
            ['native_full_update_store_replay_exact'])

class ImportedCacheTests(unittest.TestCase):
    def setUp(self):
        self.cm=tempfile.TemporaryDirectory();self.base=Path(self.cm.name).resolve()
        self.source=self.base/'runs/qualifier_repair_r09_v1/materialization/conv_0/embedding_api';self.target=self.base/'new_api'
        self.c={'fact_id':'x','source_session_id':'session_1'}
        self.ctx={'architecture':a.SCHEMA,'conversation':0,'method':'cheap','phase':'qualifier_repair',
            'candidate_id':'x','group_id':'session_1'}
        api=fixtures.LocalEmbeddingAPI(self.source,self.ctx);es=[]
        for action in ('update','add'):
            v=api.embed(['Gina'],query=False,stage='mem0_embedding_'+action)[0]
            es.append({'text':'Gina','action':action,'vector':v})
        self.cheap=[{'candidate':self.c,'embeddings':es}];self.jev=copy.deepcopy(self.cheap)
        self.refs=[old.pointer(p) for p in sorted(self.source.rglob('*')) if p.is_file()]
        self.rootpatch=patch.object(s,'ROOT',self.base);self.rootpatch.start()
        self.api=CachedLegacyAPI(self.target,keys={'GEMINI_API_KEY':'offline-test-placeholder'},retries=2,
            context={**self.ctx,'method':'jev'},cache_roots=(self.source/'api_cache',))
        with forbid_network():
            for e in es:self.assertEqual(self.api.embed(['Gina'],query=False,stage='mem0_embedding_'+e['action'])[0],e['vector'])
    def tearDown(self):self.rootpatch.stop();self.cm.cleanup()
    def proof(self):return s.audit_imported_embeddings(self.target,self.source,0,self.jev,self.cheap,self.refs)
    def test_actual_cached_provider_import_and_same_context_cross_action(self):
        p=self.proof();self.assertEqual((p['physical_attempts'],p['cache_consumers'],p['logical_invocations']),(0,2,2))
        self.assertEqual(len(p['imports']),1);self.assertFalse(p['original_physical_rows_copied'])
        self.assertEqual(p['original_proof']['physical_attempts'],1)
        self.assertEqual(list((self.target/'request_intents').glob('*.json')),[])
        self.assertEqual(a.arm_embedding_proof(p,'jev')['consumers'],p['consumers'])
    def test_new_local_producer_also_full_audited(self):
        api=fixtures.LocalEmbeddingAPI(self.target,{**self.ctx,'method':'jev'})
        vector=api.embed(['London'],query=False,stage='mem0_embedding_add')[0]
        self.jev[0]['embeddings'].append({'text':'London','action':'add','vector':vector})
        p=self.proof();self.assertEqual((p['physical_attempts'],p['cache_consumers']),(1,2))
        self.assertEqual(sum(x['imported'] for x in p['producers']),1)
    def test_foreign_conversation_import_rejected(self):
        with self.assertRaises(ValueError):s.audit_imported_embeddings(self.target,self.source,1,self.jev,self.cheap,self.refs)
    def test_rogue_consumer_and_rogue_samehash_physical_rejected(self):
        path=self.target/'api_calls.jsonl';original=path.read_bytes();row=json.loads(original.splitlines()[0])
        for kind in ('consumer','physical'):
            bad=copy.deepcopy(row)
            if kind=='consumer':bad['context']['candidate_id']='rogue'
            else:bad.update(cache_hit=False,attempt=1,status=200)
            path.write_bytes(original+json.dumps(bad).encode()+b'\n')
            with self.assertRaises(ValueError):self.proof()
        path.write_bytes(original)
    def test_equal_json_different_raw_bytes_not_inherited(self):
        p=next((self.target/'api_cache').glob('*.json'));p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.proof()
    def test_source_bytes_changed_and_orphan_intent_rejected(self):
        p=next((self.source/'api_cache').glob('*.json'));p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.proof()
    def test_missing_original_producer_and_foreign_lineage_rejected(self):
        intent=next((self.source/'request_intents').glob('*.json'));original=intent.read_bytes();intent.unlink()
        self.refs=[old.pointer(p) for p in sorted(self.source.rglob('*')) if p.is_file()]
        with self.assertRaises(ValueError):self.proof()
        intent.write_bytes(original);self.refs=[old.pointer(p) for p in sorted(self.source.rglob('*')) if p.is_file()]
        p=next((self.target/'cache_lineage').glob('*.json'));v=json.loads(p.read_text())
        v['source_path']=str(self.base/'foreign/api_cache'/Path(v['source_path']).name)
        p.write_text(json.dumps(v),encoding='utf-8')
        with self.assertRaises(ValueError):self.proof()
    def test_empty_native_arm_has_zero_new_requests(self):
        with tempfile.TemporaryDirectory() as td:
            p=s.audit_imported_embeddings(td,self.source,0,[],self.cheap,self.refs)
            self.assertEqual((p['logical_invocations'],p['physical_attempts'],p['cache_consumers'],p['ledger']),(0,0,0,None))

class NamespaceTests(unittest.TestCase):
    def test_isolation_and_pickle(self):
        before=m.original.manifest
        lib=s.isolated(m.original,{'manifest':lambda *a:None})
        self.assertIs(m.original.manifest,before);self.assertIsNot(lib.materialize_bank,m.original.materialize_bank)
        self.assertIs(lib.materialize_bank.__globals__['manifest'],lib.manifest)
        for f in (m.materialize_one,m.retrieve_conversation):pickle.dumps(f)
    def test_actual_spawn_imports_top_level_workers(self):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            out=pool.submit(spawn_probe).result(timeout=60)
        self.assertEqual(out,(a.SCHEMA,a.SCHEMA,'write-omission-r07-native-materialization-v1',m.__name__))
    def test_byte_ref_accepts_multiline_jsonl_without_json_parse(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'api_calls.jsonl';p.write_text('{}\n{}\n',encoding='utf-8');r=old.pointer(p);s.byte_ref(r)
            p.write_text('{}\n{}\n ',encoding='utf-8')
            with self.assertRaises(ValueError):s.byte_ref(r)
    def test_source_aliases_bind_real_parent_and_technical_refs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);old.save_new(p/'materialization/run_manifest.json',{'status':'technical'})
            old.save_new(p/'materialization/index_admission.json',{'status':'passed'})
            old.save_new(p/'run_manifest.json',{'status':'technical'})
            parent=old.save_new(p/'parent.json',{'original_selector':True})
            with patch.object(m.original,'_source_value',return_value={'selector_manifest':old.pointer(p/'run_manifest.json'),
                'index_admission':old.pointer(p/'materialization/index_admission.json')}):
                v=m._source_value(p,{'original_selector_manifest':parent},{})
            self.assertEqual(v['selector_manifest'],parent);self.assertEqual(v['original_selector_manifest'],parent)
            self.assertEqual(v['materialization_manifest'],v['technical_manifest'])
            self.assertEqual(v['index_admission'],v['effective_bank_gate'])
            self.assertEqual(v['snapshot_compatibility_protocol'],old.pointer(m.TECH))

if __name__=='__main__':unittest.main()
