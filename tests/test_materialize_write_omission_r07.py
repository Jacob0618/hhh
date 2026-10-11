"""Offline native seams, source preservation and request restrictions for R07."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r07_materialize_tests_owner',ROOT/'tools/materialize_write_omission_r07.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def vector():
    return {'':[1.0]+[0.0]*767,'bm25':{'indices':[1],'values':[1.0]}}


def candidate(text='Gina planned a store opening.',cid='c0_s001_m00'):
    return {'candidate_id':cid,'statement':text,'source_session_id':'session_1',
            'observation_date':'2023-01-01','proposal_index':0,
            'evidence':[{'turn_id':'D1:1','field':'text','quote':'planned a store opening'}]}


def history(mid,text):
    return {'id':mid+'history','memory_id':mid,'event':'ADD','old_memory':None,
            'new_memory':text,'is_deleted':0,'created_at':'2026-10-04','updated_at':None}


def point(text,session='session_1',date='2023-01-01'):
    return {'payload':{'data':text,'user_id':'u','hash':hashlib.md5(text.encode()).hexdigest(),
        'jev_mem_session_id':session,'jev_mem_observation_date':date,
        'created_at':'2026-10-04T00:00:00+00:00','updated_at':'2026-10-04T00:00:00+00:00'},
        'vector':vector()}


def fixture_delta():
    oldtext='Gina likes art.';newtext=candidate()['statement']
    before={'facts':{'old':point(oldtext)},'entities':{'e':{'payload':{
        'data':'Gina','user_id':'u','linked_memory_ids':['old']},'vector':vector()}},
        'history':[history('old',oldtext)]}
    after=copy.deepcopy(before);after['facts']['new']=point(newtext)
    after['history'].append(history('new',newtext))
    after['entities']['e']['payload']['linked_memory_ids'].append('new')
    rows=[{'candidate':candidate(),'memory_id':'new','status':'added',
        'expected_entity_keys':['gina'],'entity_trace':[{'memory_id':'new','normalized_text':'gina',
        'selected_entity_id':'e','selection_kind':'native_exact'}]}]
    return before,after,rows


class Store:
    def __init__(self,semantic=False,swallow=False):
        self.semantic,self.swallow=semantic,swallow
        self.rows={'entity':SimpleNamespace(id='entity',score=.97,
            payload={'data':'different wording' if semantic else 'Gina','user_id':'u',
                     'linked_memory_ids':['old']})}
    def search(self,**kwargs):
        return list(self.rows.values()) if self.semantic else []
    def insert(self,**kwargs):
        if self.swallow:return
        for eid,p in zip(kwargs['ids'],kwargs['payloads']):
            self.rows[eid]=SimpleNamespace(id=eid,payload=p)
    def update(self,**kwargs):
        if self.swallow:return
        self.rows[kwargs['vector_id']].payload=copy.deepcopy(kwargs['payload'])
    def get(self,eid):
        return self.rows.get(eid)


class Memory:
    def __init__(self,semantic=False,swallow=False):
        from mem0.memory.main import Memory as Actual
        self.entity_store=Store(semantic,swallow)
        self.embedding_model=SimpleNamespace(embed=lambda *args:[1.]+[0.]*767)
        self._normalize_entity_text=Actual._normalize_entity_text
        self._existing_entities_by_text=lambda filters:{} if semantic else {
            'gina':copy.deepcopy(self.entity_store.rows['entity'])}
        self._upsert_entity=MethodType(Actual._upsert_entity,self)
        self._link_entities_for_memory=MethodType(Actual._link_entities_for_memory,self)
        self.calls=[]
    def add(self,text,**kwargs):
        self.calls.append((text,kwargs))
        return {'results':[{'id':'new','event':'ADD','memory':text}]}


class DeltaTests(unittest.TestCase):
    def test_original_export_float32_difference_uses_existing_audit_tolerance(self):
        native=[1.]+[0.]*767;export=native.copy();export[1]=4.47e-8
        self.assertLess(m.compare_export_embedding(native,export),1e-6)
        export[1]=.01
        with self.assertRaises(ValueError):m.compare_export_embedding(native,export)
    def test_valid_added_record_preserves_original_native_and_storage_time(self):
        b,a,r=fixture_delta();self.assertEqual(m.validate_delta(b,a,r,'u')['added_facts'],1)
    def test_original_fact_vector_or_text_tamper_rejected(self):
        for field in ('data','jev_mem_observation_date'):
            b,a,r=fixture_delta();a['facts']['old']['payload'][field]='changed'
            with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
        b,a,r=fixture_delta();a['facts']['old']['vector'][''][0]=.9
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
    def test_removed_original_link_and_history_rejected(self):
        b,a,r=fixture_delta();a['entities']['e']['payload']['linked_memory_ids']=['new']
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
        b,a,r=fixture_delta();a['history'][0]['new_memory']='altered'
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
    def test_missing_new_history_or_sparse_or_entity_link_rejected(self):
        for what in ('history','sparse','link'):
            b,a,r=fixture_delta()
            if what=='history':a['history'].pop()
            elif what=='sparse':del a['facts']['new']['vector']['bm25']
            else:a['entities']['e']['payload']['linked_memory_ids']=['old']
            with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
    def test_zero_native_entities_legitimate_no_artificial_link(self):
        b,a,r=fixture_delta();a['entities']['e']['payload']['linked_memory_ids']=['old']
        r[0].update(expected_entity_keys=[],entity_trace=[])
        self.assertEqual(m.validate_delta(b,a,r,'u')['new_orphan_fact_ids'],[])
    def test_entity_sparse_absence_is_reported_native_limitation(self):
        b,a,r=fixture_delta();del b['entities']['e']['vector']['bm25'];del a['entities']['e']['vector']['bm25']
        self.assertEqual(m.validate_delta(b,a,r,'u')['entity_sparse_missing'],['e'])
    def test_foreign_extra_fact_or_entity_and_update_event_rejected(self):
        b,a,r=fixture_delta();a['facts']['unaccepted']=point('Unapproved')
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
        b,a,r=fixture_delta();a['history'][-1]['event']='UPDATE'
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')
    def test_duplicate_skip_requires_exact_text_and_no_new_history(self):
        b,a,r=fixture_delta();a=copy.deepcopy(b);r=[{'status':'duplicate_skipped','candidate':candidate('Gina likes art.'),
            'memory_id':'old','expected_entity_keys':[],'entity_trace':[]}]
        self.assertEqual(m.validate_delta(b,a,r,'u')['duplicate_skips'],1)
        r[0]['candidate']['statement']='Gina enjoys art.'
        with self.assertRaises(ValueError):m.validate_delta(b,a,r,'u')


class NativeSeamTests(unittest.TestCase):
    def test_exact_and_semantic_targets_record_native_link_without_generation(self):
        for semantic,kind in ((False,'native_exact'),(True,'native_semantic')):
            mem=Memory(semantic)
            with patch('mem0.memory.main.extract_entities',return_value=[('PERSON','Gina')]):
                row=m.add_candidate(mem,candidate(),'u',{})
            self.assertEqual(row['entity_trace'][0]['selection_kind'],kind)
            self.assertEqual(row['entity_trace'][0]['selected_entity_id'],'entity')
            self.assertEqual(mem.calls[0][1],{'user_id':'u','infer':False,
                'metadata':{'jev_mem_session_id':'session_1','jev_mem_observation_date':'2023-01-01'}})
    def test_native_insert_target_and_zero_ner(self):
        mem=Memory();mem._existing_entities_by_text=lambda filters:{}
        with patch('mem0.memory.main.extract_entities',return_value=[('PERSON','Gina')]):
            row=m.add_candidate(mem,candidate(),'u',{})
        self.assertEqual(row['entity_trace'][0]['selection_kind'],'native_insert')
        mem=Memory()
        with patch('mem0.memory.main.extract_entities',return_value=[]):
            row=m.add_candidate(mem,candidate(),'u',{})
        self.assertEqual(row['entity_trace'],[])
    def test_swallowed_entity_failure_blocks_and_add_uuid_persisted_first(self):
        mem=Memory(swallow=True);events=[]
        with patch('mem0.memory.main.extract_entities',return_value=[('PERSON','Gina')]):
            with self.assertRaises(ValueError):
                m.add_candidate(mem,candidate(),'u',{},on_add=lambda mid,result:events.append(mid))
        self.assertEqual(events,['new'])
    def test_exact_duplicate_never_calls_add_or_entity(self):
        mem=Memory();row=m.add_candidate(mem,candidate(),'u',{candidate()['statement']:'existing'})
        self.assertEqual(row['status'],'duplicate_skipped');self.assertEqual(mem.calls,[])
    def test_embedding_only_rejects_generation_query_and_changed_dimensions(self):
        api=object.__new__(m.EmbeddingOnlyAPI)
        with self.assertRaises(ValueError):api.request('gemini','gemini-3.8-flash',{},stage='write_omission_proposer_r07')
        body={'requests':[{'model':'models/gemini-embedding-001','content':{'parts':[{'text':'x'}]},
                          'taskType':'RETRIEVAL_QUERY','outputDimensionality':768}]}
        with self.assertRaises(ValueError):api.request('gemini','gemini-embedding-001',body,
            operation='batchEmbedContents',stage='mem0_embedding_add')
        body['requests'][0]['taskType']='RETRIEVAL_DOCUMENT';body['requests'][0]['outputDimensionality']=10
        with self.assertRaises(ValueError):api.request('gemini','gemini-embedding-001',body,
            operation='batchEmbedContents',stage='mem0_embedding_add')
    def test_actual_create_memory_defaults_storage_times_equal(self):
        from mem0.memory.main import Memory as Actual
        captured={}
        fake=SimpleNamespace(vector_store=SimpleNamespace(insert=lambda **kw:captured.update(kw)),
                             db=SimpleNamespace(add_history=lambda *args,**kw:None))
        with patch('mem0.memory.main.lemmatize_for_bm25',side_effect=lambda x:x):
            Actual._create_memory(fake,'hello',{'hello':[1.]+[0.]*767},metadata={'user_id':'u'})
        p=captured['payloads'][0]
        self.assertEqual(p['created_at'],p['updated_at']);self.assertNotIn('jev_mem_observation_date',p)


class PersistenceTests(unittest.TestCase):
    def test_foreign_top_level_lineage_pointer_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            for name in ('run_manifest.json','materialization/run_manifest.json',
                         'materialization/index_admission.json','retrieval_outcome.json','admission_manifest.json'):
                m.save_new(run/name,{'value':1})
            mm={'admission':m.pointer(run/'admission_manifest.json'),'protocol':{'research_protocol':{'path':'spec','sha256':'s'}}}
            source={'selector_manifest':m.pointer(run/'run_manifest.json'),'admission_manifest':mm['admission'],
                'materialization_manifest':m.pointer(run/'materialization/run_manifest.json'),
                'index_admission':m.pointer(run/'materialization/index_admission.json'),
                'retrieval_outcome':m.pointer(run/'retrieval_outcome.json'),'write_omission_protocol':mm['protocol']['research_protocol']}
            m.validate_source_pointers(source,run,mm)
            source['selector_manifest']=m.pointer(run/'retrieval_outcome.json')
            with self.assertRaises(ValueError):m.validate_source_pointers(source,run,mm)
    def test_save_exclusive_and_bound_tamper_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a.json';ref=m.save_new(p,{'answer':'a'})
            with self.assertRaises(FileExistsError):m.save_new(p,{'answer':'b'})
            p.write_text('{}')
            with self.assertRaises(ValueError):m.bound(ref,sealed=True)
    def test_copy_closed_real_lock_preserves_bytes_and_wal_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'src';dst=Path(temp)/'dst'
            (src/'qdrant/collection/facts').mkdir(parents=True)
            (src/'qdrant/.lock').write_bytes(b'')
            (src/'qdrant/collection/facts/storage.sqlite').write_bytes(b'fixture')
            (src/'history.sqlite').write_bytes(b'history')
            inv=m.inventory(src);m.copy_closed(src,dst,inv)
            self.assertEqual(m.inventory(dst),inv);self.assertEqual(m.inventory(src),inv)
            (src/'history.sqlite-wal').write_bytes(b'pending')
            with self.assertRaises(ValueError):m.inventory(src)
    def test_copy_path_escape_rejected_and_source_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'src';(src/'qdrant/collection/facts').mkdir(parents=True)
            (src/'qdrant/.lock').write_bytes(b'')
            (src/'qdrant/collection/facts/storage.sqlite').write_bytes(b'fixture')
            (src/'history.sqlite').write_bytes(b'history')
            inv=m.inventory(src);bad={**inv,'../escape':'x'}
            with patch.object(m,'inventory',return_value=bad):
                with self.assertRaises(ValueError):m.copy_closed(src,Path(temp)/'dst',bad)
            self.assertFalse((Path(temp)/'escape').exists());self.assertEqual(m.inventory(src),inv)
    def test_accepted_list_keeps_admission_order_and_rejects_reorder(self):
        a,b=candidate(cid='a'),candidate('Other fact',cid='b')
        admit={'items':[{'conversation':0,'candidates':[a,b],'accepted':{'cheap':['a','b'],'jev':['b']}}]}
        self.assertEqual(m.accepted(admit,0,'jev'),[b])
        admit['items'][0]['accepted']['cheap']=['b','a']
        with self.assertRaises(ValueError):m.accepted(admit,0,'cheap')


class CaptureTests(unittest.TestCase):
    @staticmethod
    def captured():
        def scorer(semantic_results,bm25_scores,entity_boosts,threshold,top_k,explain=False):
            return copy.deepcopy(semantic_results[:top_k])
        rows=[{'id':str(i),'score':1-i*.001,'payload':point('text'+str(i))['payload']} for i in range(25)]
        module=SimpleNamespace(score_and_rank=scorer)
        from jev_mem.sdk20_type_adapter import capture_native_pool
        with capture_native_pool(_scorer_module=module) as capture:
            module.score_and_rank(rows,{}, {},.1,20)
        return capture,scorer
    def test_native_capture_exact_replay_and_tamper(self):
        c,s=self.captured();self.assertEqual(m.replay_capture(c,s),[str(i) for i in range(20)])
        c['base20_ids'][0]='other'
        with self.assertRaises(ValueError):m.replay_capture(c,s)
    def test_selected_facts_requires_full_samebank_unique20(self):
        bank={'facts':[{'id':str(i),'text':str(i),'observation_date':'date'} for i in range(25)]}
        self.assertEqual(len(m.selected_facts(bank,[str(i) for i in range(20)])),20)
        for ids in ([str(i) for i in range(19)],['0']*20,[str(i) for i in range(19)]+['foreign']):
            with self.assertRaises(ValueError):m.selected_facts(bank,ids)
    def test_changed_arm_top_pointers_are_new_baseline_value_unchanged(self):
        c,s=self.captured();facts=[{'id':str(i),'text':'text'+str(i),'observation_date':'2023-01-01'} for i in range(25)]
        bank={'facts':facts,'native_dir':'native'};snap={'facts':{str(i):point('text'+str(i)) for i in range(25)}}
        old={'method':'baseline','query_id':'c0_q0','conversation':0,'selected_ids':[str(i) for i in range(20)],
             'selected_facts':facts[:20],'bank':{'path':'oldbank','sha256':'old'},
             'selection':{'path':'oldselection','sha256':'old'},'lineage':{}}
        sel={'status':'complete','query_id':'c0_q0','original_job_sha256':m.digest(old),
             'bank':{'path':'newbank','sha256':'new'},'capture':c,'selected_ids':[str(i) for i in range(20)],
             'selected_facts':m.selected_facts(bank,[str(i) for i in range(20)])}
        refs={'0/cheap':sel['bank'],'0/jev':sel['bank']}
        def bound(ref,sealed=False):return bank if ref['path']=='newbank' else sel
        # Keep the original scorer injection local; no native client or source read.
        with patch.object(m,'bound',side_effect=bound),patch.object(m,'snapshot',return_value=snap),\
             patch.object(m,'pointer',side_effect=lambda p:{'path':str(p),'sha256':'new'}),\
             patch.object(m,'validate_capture_bank',return_value=[str(i) for i in range(20)]),\
             patch.object(m,'require',side_effect=lambda cond,msg:None if msg=='Full three-arm cohort required' else
                         (_ for _ in ()).throw(ValueError(msg)) if not cond else None):
            jobs=m.construct_jobs('run',{'admission':{}},{'banks':refs},[old])
        self.assertEqual(jobs[0]['selected_facts'],old['selected_facts'])
        self.assertEqual(jobs[1]['bank'],sel['bank']);self.assertEqual(jobs[1]['original_job'],old)
        self.assertNotEqual(jobs[1]['selection'],old['selection'])


class LocalEmbeddingAPI:
    """Test response recorder only; all native parsing/persistence stays real."""
    embed=m.ModelAPI.embed
    keys={'GEMINI_API_KEY':'offline-test-placeholder'}
    def __init__(self,run_dir,context):self.run_dir=Path(run_dir);self.context=context
    def scoped(self,**kwargs):return type(self)(self.run_dir,{**self.context,**kwargs})
    def request(self,provider,model,body,*,operation,stage):
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        key=m.request_hash(signature);raw=self.run_dir/f'api_cache/{key}.json'
        response={'embeddings':[{'values':[1.]+[0.]*767}]}
        event={'provider':provider,'model':model,'operation':operation,'stage':stage,
               'context':self.context,'request_hash':key,'resolved_model':None,'usage':{}}
        if raw.exists():
            event.update(cache_hit=True,usage_semantics='reused_response_no_new_model_usage')
        else:
            m.save_new(raw,{'request':signature,'response':response,'stage':stage})
            m.save_new(self.run_dir/f'request_intents/{key}.json',{'request_hash':key,'signature':signature,
                'context':self.context,'stage':stage,'retries':2})
            event.update(cache_hit=False,attempt=1,status=200)
        self.run_dir.mkdir(parents=True,exist_ok=True)
        with (self.run_dir/'api_calls.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(event)+'\n')
        return response
    def count_tokens(self,*args,**kwargs):raise AssertionError('No token/API path in local fixture')


class RawEmbeddingTests(unittest.TestCase):
    def test_original_embed_replay_producer_and_duplicate_samecontext_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            context={'architecture':m.SCHEMA,'conversation':0,'method':'cheap','phase':'write_supplement',
                     'candidate_id':'c0_s001_m00','group_id':'session_1'}
            api=LocalEmbeddingAPI(temp,context)
            for _ in range(2):api.embed(['Gina'],query=False,stage='mem0_embedding_add')
            row={'status':'added','candidate':candidate('Gina'),'entity_trace':[{'text':'Gina'}]}
            audit=m.audit_embedding_inventory(temp,0,{'cheap':[row],'jev':[]})
            self.assertEqual((audit['physical_attempts'],audit['cache_consumers']), (1,1))
            vector,proof=m.audit_embedding(temp,'Gina',context,m.embedding_ledger(temp,0))
            self.assertEqual(m.compare_embedding(vector,[1.]+[0.]*767),0)
            self.assertFalse(proof['resolved_model_available'])
            altered=vector.copy();altered[3]=.1
            with self.assertRaises(ValueError):m.compare_embedding(altered,vector)
    def test_undeclared_physical_or_orphan_intent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            context={'architecture':m.SCHEMA,'conversation':0,'method':'cheap','phase':'write_supplement',
                     'candidate_id':'c0_s001_m00','group_id':'session_1'}
            api=LocalEmbeddingAPI(temp,context);api.embed(['Gina'],stage='mem0_embedding_add')
            rows={'cheap':[{'status':'added','candidate':candidate('Gina'),'entity_trace':[]}],'jev':[]}
            m.audit_embedding_inventory(temp,0,rows)
            other=api.scoped(candidate_id='unrelated');other.embed(['unrelated'],stage='mem0_embedding_add')
            with self.assertRaises(ValueError):m.audit_embedding_inventory(temp,0,rows)
    def test_actual_native_add_qdrant_persistence_and_entity_raw_lineage(self):
        # Temporary native store, fake local responses, actual SDK add/link/Qdrant.
        # The pinned CPU BM25/NLP resources are reused; network is forbidden.
        from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network
        retrieval=m.read(ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/run_manifest.json')
        resource=retrieval['resources']
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);context={'architecture':m.SCHEMA,'conversation':0,'method':'cheap',
                'phase':'write_supplement','candidate_id':'c0_s001_m00','group_id':'session_1'}
            api=LocalEmbeddingAPI(base/'api',context);memory=None
            try:
                with local_bm25_context(resource['snapshot'],resource['files_sha256']),forbid_network():
                    memory=m.writer.open_memory(base/'native',api)
                    memory.entity_store  # Materialize original entity collection while still empty.
                    c=candidate('Gina planned a store opening.')
                    row=m.add_candidate(memory,c,'u',{})
                    self.assertEqual(row['status'],'added')
                    m.native.close_memory(memory);memory=None
                snap=m.snapshot(base/'native')
                before={'facts':{},'entities':{},'history':[]}
                self.assertEqual(m.validate_delta(before,snap,[row],'u')['added_facts'],1)
                proof=m.audit_added_embeddings(base/'api',0,'cheap',[row],snap)
                self.assertTrue(any(p['kind']=='fact' for p in proof))
                closure=m.audit_embedding_inventory(base/'api',0,{'cheap':[row],'jev':[]})
                self.assertGreaterEqual(closure['declared_logical_invocations'],1)
            finally:
                if memory is not None:m.native.close_memory(memory)


if __name__=='__main__':unittest.main()
