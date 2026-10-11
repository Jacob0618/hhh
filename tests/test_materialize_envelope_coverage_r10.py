"""R10 true SDK sealed snapshots, consecutive updates, zero-copy and QUERY trim."""
from __future__ import annotations
import contextlib,copy,importlib.util,multiprocessing,pickle,tempfile,unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from tools import materialize_envelope_coverage_r10 as m
from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network
old=m.old
spec=importlib.util.spec_from_file_location('r10_local_response_fixture',
    old.ROOT/'tests/test_retrieve_qualifier_repair_compatible_r09.py')
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
LocalAPI=fixture_module.PositiveEmbeddingAPI

def spawn_probe():
    return (m.native_lib().materialize_bank.__globals__['SCHEMA'],
        m._old_lib().retrieve_conversation.__globals__['SCHEMA'],m.prior.SCHEMA,m.original_audit.SCHEMA)

@contextlib.contextmanager
def r10_sdk_pipeline_fixture():
    resources=old.read(old.ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/run_manifest.json')['resources']
    with tempfile.TemporaryDirectory(prefix='r10-real-sdk-') as td:
        base=Path(td);origin=base/'original';updated=base/'updated';zero=base/'zero';memory=None
        try:
            with local_bm25_context(resources['snapshot'],resources['files_sha256']),forbid_network():
                memory=old.writer.open_memory(origin,LocalAPI(base/'initial_api',{}));memory.entity_store
                ids=[]
                for i in range(24):
                    text=f'Gina keeps painting number {i} in Paris.'
                    result=memory.add(text,user_id='u',infer=False,
                        metadata={'jev_mem_observation_date':'2023-01-01','original_custom':'preserve'})
                    mid=result['results'][0]['id'];ids.append(mid);memory._link_entities_for_memory(mid,text,{'user_id':'u'})
                old.native.close_memory(memory);memory=None
            before=old.snapshot(origin);inv=old.inventory(origin)
            sealed_before=old.save_new(base/'original_native_snapshot.json',before)
            authenticated=m.snapshot_bound(sealed_before,sealed=True)
            old.copy_closed(origin,updated,inv);old.copy_closed(origin,zero,inv)
            rows=[]
            with local_bm25_context(resources['snapshot'],resources['files_sha256']),forbid_network():
                memory=old.writer.open_memory(updated,LocalAPI(base/'update_api',{}));memory.entity_store
                # Continuous same-instance UPDATE path, including shared entity cleanup/reembed.
                for i in range(2):
                    mid=ids[22+i]
                    candidate={'fact_id':mid,'candidate_id':mid,'candidate_key':f'c{i:02d}',
                        'original_fact':{'id':mid,'text':before['facts'][mid]['payload']['data'],
                            'observation_date':'2023-01-01'},'replacement_text':f'Gina plans to paint number {i} in London. ',
                        'source_session_id':'session_1','observation_date':'2023-01-01','proposal_index':i,
                        'edit_kind':'qualifier_repair','anchor_turn_ids':['D1:1'],'evidence':[]}
                    ctx={'architecture':m.SCHEMA,'conversation':0,'method':'cheap','phase':'qualifier_repair',
                        'candidate_id':mid,'group_id':'session_1'}
                    from jev_mem.providers import Embeddings
                    memory.embedding_model=Embeddings(LocalAPI(base/'update_api',ctx))
                    row=m.audit.update_candidate(memory,candidate,'u');rows.append(row)
                old.native.close_memory(memory);memory=None
            after=old.snapshot(updated)
            facts=[{'id':mid,'text':p['payload']['data'],'observation_date':'2023-01-01','user_id':'u',
                'event':'ADD','embedding':p['vector']['']} for mid,p in before['facts'].items()]
            original_bank={'facts':facts,'user_id':'u','native_dir':str(origin),'native_inventory':inv,'updates':[]}
            originalref=old.save_new(base/'original_bank.json',original_bank)
            info={'native_dir':str(origin),'inventory':inv,'snapshot':sealed_before,'bank':originalref,'user_id':'u'}
            zerobank={**original_bank,'native_dir':str(zero)}
            updatedbank={**original_bank,'facts':m.prior.project_bank(original_bank,after,rows),
                'native_dir':str(updated),'native_inventory':old.inventory(updated),'updates':rows}
            cr=old.save_new(base/'cheap_bank.json',updatedbank);jr=old.save_new(base/'jev_bank.json',zerobank)
            gate={'banks':{'0/cheap':cr,'0/jev':jr}}
            gate_ref=old.save_new(base/'actual_index_admission.json',gate)
            # QUERY pack is populated with deterministic local responses; production
            # CacheOnly then reuses raw exact signatures, including native trim.
            query=' \tWhat painting does Gina keep in Paris?\r\n '
            qnative=base/'query_fixture';old.copy_closed(origin,qnative,inv)
            with local_bm25_context(resources['snapshot'],resources['files_sha256']),forbid_network():
                memory=old.writer.open_memory(qnative,LocalAPI(base/'query_api',{}))
                memory.search(query,top_k=20,filters={'user_id':'u'},rerank=False)
                old.native.close_memory(memory);memory=None
            entries=[]
            for path in sorted((base/'query_api/api_cache').glob('*.json')):
                raw=old.bound(old.pointer(path));old.packs.validate_raw(path.read_bytes(),path.stem)
                if raw['request']['body']['requests'][0]['taskType']=='RETRIEVAL_QUERY':
                    entries.append({'request_hash':path.stem,'source':old.pointer(path),'target':old.pointer(path),'conversation':0})
            run=base/'retrieval_run'
            job={'method':'baseline','query_id':'c0_q0','conversation':0,'question':query,'category':1,
                'reference':'fixture','reference_date':'2023-01-02','selected_facts':[], 'selected_ids':[],'lineage':{}}
            mm={'resources':resources,'admission':originalref,'effective_bank_gate':gate_ref,
                'physical_copy_sources':{'0/cheap':m.copy_source(updatedbank,info),'0/jev':m.copy_source(zerobank,info)}}
            outcomes=m.retrieve_conversation(run,0,mm,gate,[job],entries)
            yield {'base':base,'origin':origin,'updated':updated,'zero':zero,'before':before,'after':after,'rows':rows,
                'snapshot_ref':sealed_before,'inventory':inv,'resources':resources,'api':base/'update_api',
                'bank':zerobank,'info':info,'gate':gate,'gate_ref':gate_ref,'run':run,'job':job,'manifest':mm,
                'outcomes':outcomes,'entries':entries}
            if old.inventory(origin)!=inv:raise AssertionError('Original fixture mutated')
        finally:
            if memory is not None:old.native.close_memory(memory)

class NativePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cm=r10_sdk_pipeline_fixture();cls.f=cls.cm.__enter__()
    @classmethod
    def tearDownClass(cls):cls.cm.__exit__(None,None,None)
    def test_authenticated_sealed_before_real_two_updates_full_exact_audit(self):
        f=self.f;b=m.snapshot_bound(f['snapshot_ref'],sealed=True)
        self.assertEqual(b,f['before']);self.assertNotIn('content_sha256',b)
        self.assertEqual(m.audit.validate_delta(b,f['after'],f['rows'],'u')['history_updates'],2)
        self.assertTrue(m.audit.replay_exact(f['origin'],b,f['after'],f['rows'],f['resources'])['native_full_update_store_replay_exact'])
        proof=m.audit.audit_embeddings(f['api'],0,{'cheap':f['rows'],'jev':[]})
        self.assertTrue(any(c['stage']=='mem0_embedding_update' for c in proof['consumers']))
        self.assertTrue(all(c['context']['architecture']==m.SCHEMA for c in proof['consumers']))
        bm=old.compare_persisted_fact_vectors([f['updated']],f['resources']['snapshot'],f['resources']['files_sha256'])
        self.assertEqual(bm['facts_compared'],bm['exact_matches'])
    def test_seal_body_and_unknown_metadata_tamper_rejected(self):
        f=self.f
        for mode in ('seal','body','extra'):
            value=m.bound(f['snapshot_ref'],sealed=True);value=copy.deepcopy(value)
            if mode=='seal':value['content_sha256']='0'*64
            elif mode=='body':value['history'][0]['new_memory']='changed'
            else:value['foreign']=True;value['content_sha256']=m.digest(m.body(value))
            with self.assertRaises(ValueError):m.support.snapshot_body(value)
    def test_zero_copy_and_raw_whitespace_query_actual_sdk20(self):
        f=self.f;self.assertFalse((f['zero']/'qdrant/.lock').exists())
        self.assertEqual([r['status'] for r in f['outcomes']],['complete','complete'])
        self.assertEqual(f['manifest']['physical_copy_sources']['0/jev']['native_dir'],str(f['origin']))
        for arm in ('cheap','jev'):
            s=m.sealed_read(m.retrieval_folder(f['run'],0,arm)/'selections/c0_q0.json')
            b=m.bound(f['gate']['banks'][f'0/{arm}'],sealed=True)
            self.assertEqual(old.validate_capture_bank(s['capture'],b,old.snapshot(b['native_dir'])),s['selected_ids'])
            with local_bm25_context(f['resources']['snapshot'],f['resources']['files_sha256']),forbid_network():
                self.assertGreater(m.validate_cache_events(s,f['job'],f['entries'],arm),0)
        self.assertEqual(f['job']['question'],' \tWhat painting does Gina keep in Paris?\r\n ')
    def test_event_cohort_arm_raw_hash_and_extra_consumer_tampering(self):
        f=self.f;s=m.sealed_read(m.retrieval_folder(f['run'],0,'jev')/'selections/c0_q0.json')
        with local_bm25_context(f['resources']['snapshot'],f['resources']['files_sha256']),forbid_network():
            for mode in ('missing','arm','hash','extra','target'):
                v=copy.deepcopy(s)
                if mode=='missing':v['cache_events']=[]
                elif mode=='arm':
                    v['method']='cheap'
                    for event in v['cache_events']:event['context']['method']='cheap'
                elif mode=='hash':v['cache_events'][0]['request_hash']='0'*64
                elif mode=='extra':v['cache_events'].append(copy.deepcopy(v['cache_events'][0]))
                else:v['cache_events'][0]['target']['sha256']='0'*64
                with self.assertRaises(ValueError):m.validate_cache_events(v,f['job'],f['entries'],'jev')
    def test_zero_source_changed_original_metadata_or_unrelated_fact_not_allowed(self):
        f=self.f
        for key in ('text','observation_date'):
            b=copy.deepcopy(f['bank']);b['facts'][0][key]='changed'
            with self.assertRaises(ValueError):m.copy_source(b,f['info'])
        n=copy.deepcopy(f['after']);unchanged=set(n['facts'])-{r['memory_id'] for r in f['rows']}
        n['facts'][next(iter(unchanged))]['payload']['data']='wrong'
        with self.assertRaises(ValueError):m.audit.validate_delta(f['before'],n,f['rows'],'u')
    def test_durable_update_raw_intent_and_foreign_consumer_rejected(self):
        import json,shutil
        f=self.f
        for mode in ('raw','intent','consumer'):
            with tempfile.TemporaryDirectory(prefix='r10-negative-ledger-') as td:
                folder=Path(td)/'api';shutil.copytree(f['api'],folder)
                if mode in ('raw','intent'):
                    path=next((folder/('api_cache' if mode=='raw' else 'request_intents')).glob('*.json'))
                    value=json.loads(path.read_text(encoding='utf-8'))
                    if mode=='raw':value['request']['body']['requests'][0]['content']['parts'][0]['text']='foreign'
                    else:value['context']['candidate_id']='foreign'
                    path.write_text(json.dumps(value),encoding='utf-8')
                else:
                    path=folder/'api_calls.jsonl';events=[json.loads(line) for line in path.read_text().splitlines()]
                    events[0]['context']['method']='jev';path.write_text(''.join(json.dumps(e)+'\n' for e in events))
                with self.assertRaises(ValueError):m.audit.audit_embeddings(folder,0,{'cheap':f['rows'],'jev':[]})
    def test_source_assembly_uses_real_gate_pointer_and_current_arm_texts(self):
        f=self.f;lib=m.assembly_lib(f['run'],f['manifest'])
        self.assertFalse((f['run']/'materialization/index_admission.json').exists())
        self.assertEqual(lib.pointer(f['run']/'materialization/index_admission.json'),f['gate_ref'])
        jobs=[copy.deepcopy(f['job']) for _ in range(1540)]
        out=lib.construct_jobs(f['run'],f['manifest'],f['gate'],jobs)
        self.assertEqual(len(out),4620)
        for arm,offset in (('cheap',1540),('jev',3080)):
            s=m.sealed_read(m.retrieval_folder(f['run'],0,arm)/'selections/c0_q0.json')
            self.assertEqual(out[offset]['selected_facts'],s['selected_facts'])
            self.assertEqual(out[offset]['question'],f['job']['question'])
            self.assertEqual(out[offset]['lineage']['index_admission'],f['gate_ref'])
        self.assertNotEqual(out[1540]['selected_facts'],out[3080]['selected_facts'])
    def test_actual_seal_validate_source_file_chain_with_heavy_cohort_substitution(self):
        f=self.f;run=f['run'];selector=f['base']/'selector';selector.mkdir()
        selectorref=old.save_new(selector/'run_manifest.json',{'schema':'controlled-selector'})
        old.save_new(selector/'admission_manifest.json',{'schema':'controlled-admission'})
        old.save_new(selector/'score_manifest.json',{'schema':'controlled-compiled'})
        old.save_new(selector/'reviews/jev_terminal_audit_v1.json',{'schema':'controlled-score-audit'})
        old.save_new(run/'retrieval_outcome.json',{'schema':m.SCHEMA,'status':'complete','model_calls':0})
        frozenref=old.save_new(f['base']/'frozen_inputs.json',{'official_metric_provenance':{'fixture':True},
            'source_dataset_sha256':'fixture','jobs':[]})
        r09ref=m.pointer(old.ROOT/'research/jev_autoresearch_2026-10-04/R09_qualifier_repair_protocol_final_v1.json')
        mm={**f['manifest'],'protocol':{'frozen_source':frozenref,'frozen_qa_run_manifest':frozenref,
            'frozen_baseline_reuse_proof':frozenref,'reuse_proof_qa_manifest':frozenref},
            'qualifier_repair_protocol':r09ref,'original_selector_manifest':selectorref,
            'native_manifest':f['gate_ref'],'code_sha256':{}}
        old.save_new(run/'run_manifest.json',mm)
        auditvalue={'schema':m.SCHEMA,'status':'passed','current_capture_self_replay':3080}
        old.save_new(run/'retrieval_gate.json',auditvalue)
        jobs=[copy.deepcopy(f['job']) for _ in range(1540)]
        # Only the population-level audit/manifest are substituted. Every bank,
        # capture, selection, native snapshot, source, gate and pointer is real.
        with patch.object(m,'RETRIEVAL',run),patch.object(m,'SELECTOR',selector),\
            patch.object(m,'retrieval_manifest',return_value=mm),\
            patch.object(m,'retrieval_audit_value',return_value=auditvalue),\
            patch.object(old,'old_jobs',return_value=jobs):
            ref=m.seal(run);actual=m.validate_source(ref['path'])
            self.assertEqual(len(actual['jobs']),4620)
            self.assertEqual(actual['envelope_coverage_protocol'],m.pointer(m.SPEC))
            self.assertEqual(actual['qualifier_repair_protocol'],r09ref)
            self.assertEqual(actual['index_admission'],f['gate_ref'])
            self.assertEqual(actual['jobs'][3080]['question'],f['job']['question'])
            self.assertEqual(actual['retrieval_outcome'],m.pointer(run/'retrieval_outcome.json'))
            changed=m.body(actual);changed['jobs'][3080]['question']=f['job']['question'].strip()
            other=old.save_new(run/'foreign_selected_inputs.json',changed)
            with self.assertRaises(ValueError):m.validate_source(other['path'])
    def test_wrong_capture_score_and_candidate_payload_rejected_after_reseal(self):
        f=self.f;s=m.sealed_read(m.retrieval_folder(f['run'],0,'jev')/'selections/c0_q0.json')
        bank=m.bound(f['gate']['banks']['0/jev'],sealed=True)
        for mode in ('score','payload'):
            c=copy.deepcopy(s['capture'])
            if mode=='score':c['base20'][0]['score']+=.1
            else:c['inputs']['semantic_results'][0]['payload']['data']='wrong'
            c['capture_sha256']=m.digest({k:v for k,v in c.items() if k!='capture_sha256'})
            with self.assertRaises(ValueError):old.validate_capture_bank(c,bank,f['before'])

class BoundaryTests(unittest.TestCase):
    def test_private_clones_and_real_spawn(self):
        self.assertEqual(m.prior.SCHEMA,'qualifier-repair-r09-native-v1')
        self.assertEqual(m.original_audit.SCHEMA,m.prior.SCHEMA)
        for worker in (m.materialize_conversation,m.retrieve_conversation):pickle.dumps(worker)
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(pool.submit(spawn_probe).result(timeout=60),
                (m.SCHEMA,m.SCHEMA,m.prior.SCHEMA,m.prior.SCHEMA))
    def test_actual_sdk_query_type_empty_and_trim_boundary(self):
        from mem0.memory.main import _validate_and_trim_search_query
        self.assertEqual(_validate_and_trim_search_query(' \nQ\t '),'Q')
        for value in (' ',None,1):
            with self.assertRaises((ValueError,TypeError)):_validate_and_trim_search_query(value)

if __name__=='__main__':unittest.main()
