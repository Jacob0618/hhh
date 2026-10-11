"""Offline actual Mem0 UPDATE persistence/cleanup plus adversarial audit tests."""
from __future__ import annotations
import contextlib,copy,hashlib,json,pickle,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import materialize_qualifier_repair_r09 as m
from tools import r09_native_update_audit as a
from tools import materialize_write_omission_r07 as old
from jev_mem.providers import ModelAPI,Embeddings
from jev_mem.sdk20_cached_api import digest as request_hash

def spawn_namespace_probe():
    return m._lib().prepare.__globals__['SCHEMA'],old.SCHEMA

class LocalEmbeddingAPI:
    """Durable local responses; original embedding parser and all SDK writes run."""
    embed=ModelAPI.embed
    keys={'GEMINI_API_KEY':'offline-test-placeholder'}
    def __init__(self,path,context):self.run_dir=Path(path);self.context=context
    def scoped(self,**kw):return type(self)(self.run_dir,{**self.context,**kw})
    def request(self,provider,model,body,*,operation,stage):
        sig={'provider':provider,'model':model,'operation':operation,'body':body};key=request_hash(sig)
        raw=self.run_dir/f'api_cache/{key}.json'
        if raw.exists():response=old.bound(old.pointer(raw))['response'];cached=True
        else:
            texts=[x['content']['parts'][0]['text'] for x in body['requests']]
            response={'embeddings':[{'values':[random.Random(int(hashlib.sha256(t.encode()).hexdigest(),16)+i).uniform(-1,1)
                for i in range(768)]} for t in texts]}
            old.save_new(raw,{'request':sig,'response':response,'stage':stage})
            old.save_new(self.run_dir/f'request_intents/{key}.json',{'signature':sig,'request_hash':key,
                'context':self.context,'stage':stage,'retries':2});cached=False
        event={'provider':provider,'model':model,'operation':operation,'stage':stage,'context':self.context,
            'request_hash':key,'resolved_model':None,'usage':{},'cache_hit':cached}
        if cached:event.update(usage_semantics='reused_response_no_new_model_usage')
        else:event.update(attempt=1,status=200)
        with (self.run_dir/'api_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
        return response
    def count_tokens(self,*args,**kw):raise AssertionError('No network/token stage allowed in fixture')

@contextlib.contextmanager
def native_update_fixture():
    """Public fixture for independent reviewers; never opens an original writable DB."""
    from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network
    resource=old.read(old.ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/run_manifest.json')['resources']
    with tempfile.TemporaryDirectory(prefix='r09-real-sdk-test-') as temp:
        base=Path(temp);original=base/'original';dest=base/'updated';memory=None
        try:
            with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
                memory=old.writer.open_memory(original,LocalEmbeddingAPI(base/'initial_api',{}));memory.entity_store
                ids=[]
                for text in ('Gina moved to Paris.','Gina likes painting.'):
                    result=memory.add(text,user_id='u',infer=False,
                        metadata={'jev_mem_observation_date':'2023-01-01','original_custom':'preserve'})
                    mid=result['results'][0]['id'];ids.append(mid)
                    memory._link_entities_for_memory(mid,text,{'user_id':'u'})
                old.native.close_memory(memory);memory=None
            before=old.snapshot(original);original_inventory=old.inventory(original)
            old.copy_closed(original,dest,original_inventory)
            candidate={'fact_id':ids[0],'candidate_id':ids[0],'candidate_key':'c00',
                'original_fact':{'id':ids[0],'text':'Gina moved to Paris.','observation_date':'2023-01-01'},
                'replacement_text':'Gina plans to move to London. ','source_session_id':'session_1',
                'observation_date':'2023-01-01','proposal_index':0,'edit_kind':'qualifier_repair',
                'anchor_turn_ids':['D1:1'],'evidence':[]}
            context={'architecture':a.SCHEMA,'conversation':0,'method':'cheap','phase':'qualifier_repair',
                'candidate_id':ids[0],'group_id':'session_1'}
            with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
                memory=old.writer.open_memory(dest,LocalEmbeddingAPI(base/'api',context));memory.entity_store
                row=a.update_candidate(memory,candidate,'u')
                old.native.close_memory(memory);memory=None
            after=old.snapshot(dest)
            yield {'base':base,'original':original,'updated':dest,'resources':resource,'before':before,
                'after':after,'rows':[row],'ids':ids,'original_inventory':original_inventory,'api':base/'api'}
            if old.inventory(original)!=original_inventory:raise AssertionError('Original fixture mutated')
        finally:
            if memory is not None:old.native.close_memory(memory)

class ActualUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.manager=native_update_fixture();cls.fixture=cls.manager.__enter__()
    @classmethod
    def tearDownClass(cls):cls.manager.__exit__(None,None,None)
    def test_original_public_update_cleanup_shared_reembed_history_and_exact_replay(self):
        f=self.fixture;b,n=f['before'],f['after'];r=f['rows'][0];mid,other=f['ids']
        self.assertEqual(a.validate_delta(b,n,[r],'u')['history_updates'],1)
        self.assertEqual(n['facts'][mid]['payload']['data'],'Gina plans to move to London. ')
        self.assertEqual(n['facts'][mid]['payload']['created_at'],b['facts'][mid]['payload']['created_at'])
        self.assertEqual(n['facts'][other],b['facts'][other])
        paris=[eid for eid,p in b['entities'].items() if p['payload']['data'].casefold()=='paris']
        self.assertTrue(paris);self.assertTrue(all(eid not in n['entities'] for eid in paris))
        self.assertTrue(any(p['payload']['data'].casefold()=='london' and mid in p['payload']['linked_memory_ids']
            for p in n['entities'].values()))
        self.assertTrue(any(e['action']=='update' and e['text'].casefold()=='gina' for e in r['embeddings']))
        self.assertNotEqual(n['facts'][mid]['vector'],b['facts'][mid]['vector'])
        self.assertTrue(a.replay_exact(f['original'],b,n,[r],f['resources'])['native_full_update_store_replay_exact'])
        bm=old.compare_persisted_fact_vectors([f['updated']],f['resources']['snapshot'],f['resources']['files_sha256'])
        self.assertEqual(bm['facts_compared'],bm['exact_matches'])
        proof=a.audit_embeddings(f['api'],0,{'cheap':[r],'jev':[]})
        self.assertGreater(proof['cache_consumers'],0)
        self.assertTrue(any(x['stage']=='mem0_embedding_update' for x in proof['consumers']))
    def test_old_history_loss_rejected(self):
        f=self.fixture;n=copy.deepcopy(f['after']);n['history']=n['history'][1:]
        with self.assertRaises(ValueError):a.validate_delta(f['before'],n,f['rows'],'u')
    def test_unaccepted_fact_or_original_observation_changed_rejected(self):
        f=self.fixture
        for mid,key,value in ((f['ids'][1],'data','changed'),(f['ids'][0],'jev_mem_observation_date','tomorrow')):
            n=copy.deepcopy(f['after']);n['facts'][mid]['payload'][key]=value
            with self.assertRaises(ValueError):a.validate_delta(f['before'],n,f['rows'],'u')
    def test_shared_unchanged_link_loss_rejected(self):
        f=self.fixture;n=copy.deepcopy(f['after']);other=f['ids'][1]
        for p in n['entities'].values():
            if other in p['payload']['linked_memory_ids']:p['payload']['linked_memory_ids'].remove(other);break
        with self.assertRaises(ValueError):a.validate_delta(f['before'],n,f['rows'],'u')
    def test_wrong_update_action_or_trace_vector_rejected(self):
        f=self.fixture
        for kind in ('action','vector','runtime'):
            rows=copy.deepcopy(f['rows'])
            if kind=='action':rows[0]['embeddings'][0]['action']='add'
            elif kind=='vector':rows[0]['operations'][0]['kwargs']['vector'][0]+=.2
            else:rows[0]['native_runtime_code_sha256']={}
            with self.assertRaises(ValueError):a.validate_delta(f['before'],f['after'],rows,'u')
    def test_wrong_actual_store_trace_cannot_replay(self):
        f=self.fixture;rows=copy.deepcopy(f['rows']);rows[0]['operations'][0]['kwargs']['vector'][0]+=.2
        with self.assertRaises(ValueError):a.replay_exact(f['original'],f['before'],f['after'],rows,f['resources'])
    def test_zero_revision_exact_original_and_zero_embedding_activity(self):
        f=self.fixture
        self.assertEqual(a.validate_delta(f['before'],f['before'],[],'u')['updated'],0)
        self.assertTrue(a.replay_exact(f['original'],f['before'],f['before'],[],f['resources'])
            ['native_full_update_store_replay_exact'])
        with tempfile.TemporaryDirectory() as temp:
            proof=a.audit_embeddings(temp,0,{'cheap':[],'jev':[]})
            self.assertEqual((proof['physical_attempts'],proof['cache_consumers'],proof['ledger']),(0,0,None))
    def test_export_birth_lineage_and_updated_current_text(self):
        f=self.fixture;r=f['rows'][0];original={'facts':[{'id':r['memory_id'],'text':r['candidate']['original_fact']['text'],
            'observation_date':'2023-01-01','event':'ADD','input_scope':{'source_turn_ids':['D1:1']}}]}
        out=m.project_bank(original,f['after'],[r])[0]
        self.assertEqual(out['event'],'ADD');self.assertEqual(out['repair']['native_event'],'UPDATE')
        self.assertFalse(out['repair']['original_raw_certifies_replacement'])
        self.assertEqual(out['text'],r['candidate']['replacement_text']);self.assertEqual(out['input_scope'],original['facts'][0]['input_scope'])

class BoundaryTests(unittest.TestCase):
    def test_isolated_globals_and_windows_worker_pickle(self):
        lib=m._lib();self.assertEqual(old.SCHEMA,'write-omission-r07-native-materialization-v1')
        self.assertIsNot(lib.prepare,old.prepare);self.assertEqual(lib.prepare.__globals__['SCHEMA'],m.SCHEMA)
        for fn in (m.retrieve_conversation,m.materialize_conversation):pickle.dumps(fn)
    def test_actual_windows_spawn_keeps_clone_globals_local(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(pool.submit(spawn_namespace_probe).result(timeout=60),
                (m.SCHEMA,'write-omission-r07-native-materialization-v1'))
    def test_exact_admitted_text_not_stripped_and_no_duplicate_target(self):
        c={'candidate_id':'x','fact_id':'x','replacement_text':' body '}
        ad={'items':[{'conversation':0,'candidates':[c],'accepted':{'cheap':['x'],'jev':[]}}]}
        self.assertEqual(m.accepted(ad,0,'cheap')[0]['replacement_text'],' body ')
        ad['items'].append(ad['items'][0])
        with self.assertRaises(ValueError):m.accepted(ad,0,'cheap')
    def test_update_rows_body_comparison_and_writeahead_binding(self):
        c={'fact_id':'x'};r={'candidate':c}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);old.save_new(p/'updates/0000.json',r)
            old.save_new(p/'update_dispatch/0000.json',{'candidate':c,'status':'native_update_about_to_dispatch'})
            self.assertEqual(m.update_rows(p,[c]),[r])
            with self.assertRaises(ValueError):m.update_rows(p,[])
    def test_retrieval_source_jobs_keep_updated_text_and_new_top_pointers(self):
        # Exercise the actual frozen job assembler in isolated new globals.
        # Capture validation belongs to the unchanged SDK20 primitive; this test
        # specifically protects the R09 UPDATE export/path/schema seam.
        ids=[str(i) for i in range(20)]
        original_facts=[{'id':mid,'text':'original '+mid,'observation_date':'2023-01-01'} for mid in ids]
        repaired=copy.deepcopy(original_facts);repaired[0]['text']='replacement exact '
        jobs=[{'method':'baseline','query_id':f'c0_q{i}','conversation':0,'selected_ids':ids,
            'selected_facts':copy.deepcopy(original_facts),'bank':{'path':'oldbank','sha256':'old'},
            'selection':{'path':'oldselection','sha256':'old'},'lineage':{}} for i in range(1540)]
        bank={'facts':repaired,'native_dir':'approved-native'}
        bref={'path':'repairedbank','sha256':'new'};g={'banks':{'0/cheap':bref,'0/jev':bref}}
        path_calls=[]
        def pointer(path):
            path_calls.append(str(path));return {'path':str(path),'sha256':'new'}
        def bound(ref,sealed=False):
            if ref==bref:return bank
            qid=Path(ref['path']).stem;job=jobs[int(qid.split('q')[1])]
            return {'status':'complete','query_id':qid,'original_job_sha256':m.digest(job),'bank':bref,
                'capture':{},'selected_ids':ids,'selected_facts':repaired}
        lib=m._lib({'bank_folder':m.retrieval_folder,'pointer':pointer,'bound':bound,
            'snapshot':lambda p:{},'validate_capture_bank':lambda *args:ids})
        out=lib.construct_jobs('isolated-run',{'admission':{'path':'admit','sha256':'a'}},g,jobs)
        self.assertEqual(len(out),4620)
        self.assertEqual(out[0]['selected_facts'],original_facts)
        for j in (out[1540],out[3080]):
            self.assertEqual(j['selected_facts'],repaired);self.assertEqual(j['bank'],bref)
            self.assertIn('retrieval',j['selection']['path']);self.assertNotIn('materialization/conv_',j['selection']['path'])
            self.assertEqual(j['original_job'],jobs[0])
        self.assertEqual(lib.construct_jobs.__globals__['SOURCE_SCHEMA'],m.SOURCE_SCHEMA)
    def test_cross_action_same_hash_cache_and_two_arm_proof_stability(self):
        with tempfile.TemporaryDirectory() as temp:
            c={'fact_id':'x','source_session_id':'session_1'}
            rows={}
            for arm in ('cheap','jev'):
                ctx={'architecture':a.SCHEMA,'conversation':0,'method':arm,'phase':'qualifier_repair',
                    'candidate_id':'x','group_id':'session_1'};api=LocalEmbeddingAPI(temp,ctx)
                es=[]
                for action in ('update','add'):
                    vector=api.embed(['Gina'],query=False,stage='mem0_embedding_'+action)[0]
                    es.append({'text':'Gina','action':action,'vector':vector})
                rows[arm]=[{'candidate':c,'embeddings':es}]
                proof=a.audit_embeddings(temp,0,rows)
                if arm=='cheap':first=a.arm_embedding_proof(proof,'cheap')
            self.assertEqual(a.arm_embedding_proof(proof,'cheap'),first)
            self.assertEqual((proof['physical_attempts'],proof['cache_consumers']),(1,3))
    def test_undeclared_cache_or_orphan_intent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx={'architecture':a.SCHEMA,'conversation':0,'method':'cheap','phase':'qualifier_repair',
                'candidate_id':'x','group_id':'session_1'};api=LocalEmbeddingAPI(temp,ctx)
            vector=api.embed(['Gina'],query=False,stage='mem0_embedding_add')[0]
            rows={'cheap':[{'candidate':{'fact_id':'x','source_session_id':'session_1'},
                'embeddings':[{'text':'Gina','action':'add','vector':vector}]}]}
            a.audit_embeddings(temp,0,rows)
            api.scoped(candidate_id='rogue').embed(['Gina'],query=False,stage='mem0_embedding_add')
            with self.assertRaises(ValueError):a.audit_embeddings(temp,0,rows)

if __name__=='__main__':unittest.main()
