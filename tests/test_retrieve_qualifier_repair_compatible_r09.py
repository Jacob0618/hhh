"""Real zero-UPDATE copy/CacheOnly SDK20 and strict historical parity regressions."""
from __future__ import annotations
import contextlib,copy,importlib.util,json,multiprocessing,pickle,tempfile,unittest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from tools import retrieve_qualifier_repair_compatible_r09 as m
from jev_mem.providers import Embeddings
from jev_mem.sdk20_cached_api import digest as request_hash
from jev_mem.local_bm25_context_v2 import forbid_network,local_bm25_context
old=m.old
spec=importlib.util.spec_from_file_location('r09_retrieval_local_fixture',old.ROOT/'tests/test_materialize_qualifier_repair_r09.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

class PositiveEmbeddingAPI(fixtures.LocalEmbeddingAPI):
    """Controlled unit vectors exercise the real SDK's positive-score top20 path."""
    def request(self,provider,model,body,*,operation,stage):
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        key=request_hash(signature);raw=self.run_dir/f'api_cache/{key}.json'
        if raw.exists():response=old.bound(old.pointer(raw))['response'];cached=True
        else:
            response={'embeddings':[{'values':[1.0]+[0.0]*767} for _ in body['requests']]}
            old.save_new(raw,{'request':signature,'response':response,'stage':stage})
            old.save_new(self.run_dir/f'request_intents/{key}.json',{'signature':signature,
                'request_hash':key,'context':self.context,'stage':stage,'retries':2});cached=False
        event={'provider':provider,'model':model,'operation':operation,'stage':stage,'context':self.context,
            'request_hash':key,'resolved_model':None,'usage':{},'cache_hit':cached}
        if cached:event['usage_semantics']='reused_response_no_new_model_usage'
        else:event.update(attempt=1,status=200)
        with (self.run_dir/'api_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
        return response

def spawn_probe():return m.retrieve_conversation.__module__,m.original.SCHEMA,m.parent.CODE==tuple(m.parent.CODE)

@contextlib.contextmanager
def zero_update_sdk20_fixture():
    resource=old.read(old.ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/run_manifest.json')['resources']
    with tempfile.TemporaryDirectory(prefix='r09-zero-copy-sdk20-') as td:
        base=Path(td);origin=base/'original';zero=base/'zero';memory=None
        try:
            with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
                memory=old.writer.open_memory(origin,PositiveEmbeddingAPI(base/'build_api',{}));memory.entity_store
                for i in range(24):
                    text=f'Gina keeps painting number {i} in Paris.'
                    r=memory.add(text,user_id='u',infer=False,metadata={'jev_mem_observation_date':'2023-01-01'})
                    memory._link_entities_for_memory(r['results'][0]['id'],text,{'user_id':'u'})
                old.native.close_memory(memory);memory=None
            inv=old.inventory(origin);before=old.snapshot(origin);old.copy_closed(origin,zero,inv)
            facts=[{'id':mid,'text':p['payload']['data'],'observation_date':'2023-01-01','user_id':'u',
                'embedding':p['vector']['']} for mid,p in before['facts'].items()]
            bank={'facts':facts,'updates':[],'user_id':'u','native_dir':str(zero),'native_inventory':inv}
            br=old.save_new(base/'zero_bank.json',bank);original_bank={**bank,'native_dir':str(origin)}
            originalref=old.save_new(base/'original_bank.json',original_bank)
            info={'native_dir':str(origin),'inventory':inv,'snapshot':old.save_new(base/'snapshot.json',before),
                'bank':originalref,'user_id':'u'}
            # Populate local QUERY raw records by running the real SDK once with
            # deterministic local-only responses, then run actual CacheOnly.
            query='What painting does Gina keep in Paris?';query_native=base/'query_fixture'
            old.copy_closed(origin,query_native,inv)
            with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
                memory=old.writer.open_memory(query_native,PositiveEmbeddingAPI(base/'query_api',{}))
                memory.search(query,top_k=20,filters={'user_id':'u'},rerank=False)
                old.native.close_memory(memory);memory=None
            entries=[]
            for p in sorted((base/'query_api/api_cache').glob('*.json')):
                raw=old.bound(old.pointer(p))
                if raw['request']['body']['requests'][0]['taskType']=='RETRIEVAL_QUERY':
                    old.packs.validate_raw(p.read_bytes(),p.stem)
                    entries.append({'request_hash':p.stem,'source':old.pointer(p),'target':old.pointer(p),'conversation':0})
            cp=m.copy_source(0,'jev',bank,info)
            cheapbank={**bank,'native_dir':str(origin)};cr=old.save_new(base/'cheap_bank.json',cheapbank)
            gate={'banks':{'0/cheap':cr,'0/jev':br}}
            run=base/'retrieval_run';job={'method':'baseline','query_id':'c0_q0','conversation':0,
                'question':query,'reference_date':'2023-01-02','selected_facts':[],'selected_ids':[],'lineage':{}}
            mm={'resources':resource,'admission':{'path':'fixture','sha256':'fixture'},
                'physical_copy_sources':{'0/cheap':m.copy_source(0,'cheap',cheapbank,info),'0/jev':cp}}
            rows=m.retrieve_conversation(run,0,mm,gate,[job],entries)
            yield {'base':base,'origin':origin,'zero':zero,'before':before,'inventory':inv,'bank':bank,'info':info,
                'copy_source':cp,'run':run,'gate':gate,'job':job,'rows':rows,'entries':entries,'manifest':mm}
            if old.inventory(origin)!=inv or old.snapshot(zero)!=before:raise AssertionError('Approved sources mutated')
        finally:
            if memory is not None:old.native.close_memory(memory)

class RealSdkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cm=zero_update_sdk20_fixture();cls.f=cls.cm.__enter__()
    @classmethod
    def tearDownClass(cls):cls.cm.__exit__(None,None,None)
    def test_zero_clone_missing_lock_equivalent_source_copy_and_real_sdk20(self):
        f=self.f
        self.assertFalse((f['zero']/'qdrant/.lock').exists());self.assertTrue((f['origin']/'qdrant/.lock').is_file())
        self.assertEqual(old.inventory(f['zero']),old.inventory(f['origin']))
        self.assertEqual(f['copy_source']['native_dir'],str(f['origin']))
        self.assertEqual([r['status'] for r in f['rows']],['complete','complete'])
        for arm in ('cheap','jev'):
            s=m.sealed_read(m.retrieval_folder(f['run'],0,arm)/'selections/c0_q0.json')
            b=m.bound(f['gate']['banks'][f'0/{arm}'],sealed=True)
            ids=old.validate_capture_bank(s['capture'],b,old.snapshot(b['native_dir']))
            self.assertEqual(len(ids),20);self.assertEqual(s['selected_ids'],ids)
            self.assertEqual(s['selected_facts'],old.selected_facts(b,ids))
            self.assertTrue(all(e['cache_hit'] and e['model_calls']==0 for e in s['cache_events']))
            self.assertEqual(s['bank'],f['gate']['banks'][f'0/{arm}'])
            self.assertNotIn('physical_copy_source',s)
        self.assertEqual(old.inventory(f['origin']),f['inventory'])
    def test_nonzero_update_or_changed_text_not_borrow_original(self):
        f=self.f
        for mode in ('update','text','date'):
            b=copy.deepcopy(f['bank'])
            if mode=='update':b['updates']=[{}]
            elif mode=='text':b['facts'][0]['text']='changed'
            else:b['facts'][0]['observation_date']='2023-01-02'
            with self.assertRaises(ValueError):m.copy_source(0,'jev',b,f['info'])
    def test_nonexception_missing_lock_rejected_and_missing_query_cache_cannot_dispatch(self):
        f=self.f
        with self.assertRaises(ValueError):m.copy_source(2,'jev',f['bank'],f['info'])
        api=old.CacheOnlyEmbeddings({})
        with forbid_network(),self.assertRaises(RuntimeError):api.embed(['missing'],query=True,stage='mem0_embedding_search')
    def test_zero_copy_reader_source_assembler_preserves_full20_id_text_date(self):
        f=self.f;s=m.sealed_read(m.retrieval_folder(f['run'],0,'jev')/'selections/c0_q0.json')
        ids=s['selected_ids'];facts=s['selected_facts']
        # Repeating one controlled identity exercises the legacy fixed-size
        # assembler without mocking any pointer, seal, capture, or bank read.
        jobs=[copy.deepcopy(f['job']) for _ in range(1540)]
        gate_ref=old.save_new(f['base']/'actual_index_gate.json',f['gate'])
        mm={**f['manifest'],'approved_effective20_gate':gate_ref}
        lib=m.assembly_lib(f['run'],mm)
        self.assertFalse((f['run']/'materialization/index_admission.json').exists())
        self.assertEqual(lib.pointer(f['run']/'materialization/index_admission.json'),old.pointer(gate_ref['path']))
        out=lib.construct_jobs(f['run'],mm,f['gate'],jobs)
        self.assertEqual(len(out),4620)
        for job in (out[1540],out[3080]):
            self.assertEqual(job['selected_facts'],facts)
            self.assertEqual(job['lineage']['index_admission'],gate_ref)
        self.assertEqual(out[0]['selected_facts'],f['job']['selected_facts'])
        self.assertEqual(out[3080]['bank'],f['gate']['banks']['0/jev'])
        self.assertEqual(out[3080]['selected_ids'],ids)
    def test_exact_query_event_cohort_rejects_missing_foreign_or_extra_calls(self):
        f=self.f;s=m.sealed_read(m.retrieval_folder(f['run'],0,'jev')/'selections/c0_q0.json')
        resource=f['manifest']['resources']
        with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
            self.assertGreater(m.validate_cache_events(s,f['job'],f['entries'],'jev'),0)
            for mode in ('missing','context','hash','target','extra','usage','joint_method'):
                v=copy.deepcopy(s)
                if mode=='missing':v['cache_events']=[]
                elif mode=='context':v['cache_events'][0]['context']['query_id']='foreign'
                elif mode=='hash':v['cache_events'][0]['request_hash']='0'*64
                elif mode=='target':v['cache_events'][0]['target']['path']='foreign'
                elif mode=='extra':v['cache_events'].append(copy.deepcopy(v['cache_events'][0]))
                elif mode=='usage':v['cache_events'][0]['usage_semantics']='new_model_usage'
                else:
                    v['method']='cheap'
                    for e in v['cache_events']:e['context']['method']='cheap'
                with self.assertRaises(ValueError):m.validate_cache_events(v,f['job'],f['entries'],'jev')

class ParityAndWorkerTests(unittest.TestCase):
    def test_closed_proof_bytes_and_file_inventory_bound_without_jsonl_parsing(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'APPROVED',Path(td)):
            folder=Path(td)/'materialization';folder.mkdir()
            ledger=folder/'calls.jsonl';ledger.write_text('{"a":1}\n{"a":2}\n',encoding='utf-8')
            native=folder/'conv_0/jev/native/qdrant';native.mkdir(parents=True)
            (native/'store.sqlite').write_bytes(b'native-has-its-own-inventory')
            refs=m.approved_input_refs();self.assertEqual(refs,[m.pointer(ledger)])
            m.validate_approved_inputs(refs)
            ledger.write_text('{"a":1}\n{"a":3}\n',encoding='utf-8')
            with self.assertRaises(ValueError):m.validate_approved_inputs(refs)
            refs=m.approved_input_refs();(folder/'orphan.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):m.validate_approved_inputs(refs)
    def test_only_envelope_and_event_timestamp_excluded(self):
        base={'content_sha256':'old','selected_ids':['x'],'selected_facts':[{'id':'x','text':'exact ','observation_date':'today'}],
            'capture':{'native_scores':{'x':.8}},'cache_events':[{'timestamp_utc':'old','target':{'path':'p','sha256':'h'},
                'context':{'query_id':'q'},'request_hash':'h'}]}
        current=copy.deepcopy(base);current['content_sha256']='new';current['cache_events'][0]['timestamp_utc']='new'
        self.assertEqual(m.selection_body(base),m.selection_body(current))
        for key in ('score','text','order','source'):
            v=copy.deepcopy(current)
            if key=='score':v['capture']['native_scores']['x']+=1e-12
            elif key=='text':v['selected_facts'][0]['text']='exact'
            elif key=='order':v['selected_ids']=['y']
            else:v['cache_events'][0]['target']['path']='foreign'
            self.assertNotEqual(m.selection_body(base),m.selection_body(v))
    def test_top_level_worker_pickle_and_real_spawn(self):
        pickle.dumps(m.retrieve_conversation)
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(pool.submit(spawn_probe).result(timeout=60),
                (m.__name__,m.SCHEMA,True))
if __name__=='__main__':unittest.main()
