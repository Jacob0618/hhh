"""Offline tests using native Memory.add, fake storage and blocked model I/O."""
import json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tests import test_mem0_adapter as fixtures
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT,generate_additive_extraction_prompt
from mem0.memory.utils import parse_messages
from tools import run_mem0_regroup_sdk20 as w


class TokenAPI:
    def __init__(self):self.calls=[]
    def count_tokens(self,text,**kwargs):self.calls.append((text,kwargs));return len(text)


class Embedding:
    def __init__(self):self.config=SimpleNamespace();self.api=TokenAPI();self.calls=[]
    def embed(self,text,memory_action='add'):
        self.calls.append((text,memory_action));v=[0.0]*768;v[0]=1.0;return v
    def embed_batch(self,texts,memory_action='add'):
        return [self.embed(text,memory_action) for text in texts]


def envelope(group_id='g0',dates=('2023-05-01','2023-06-10'),indices=(0,2)):
    messages=[{'role':'user','content':'Alice: Yesterday I moved to Paris.'},
              {'role':'assistant','content':'Bob: Yesterday I visited Lyon.'}]
    sources=[{'source_turn_id':f'D{i+1}:1','session_id':f'session_{i+1}','observation_date':day,'source_order_index':pos}
             for i,(day,pos) in enumerate(zip(dates,indices))]
    ds=sorted(set(dates))
    return {'group_id':group_id,'segment_ids':['s0','s2'],'messages':messages,'turn_sources':sources,
        'source_turn_ids':[s['source_turn_id'] for s in sources],'source_order_indices':list(indices),
        'source_observation_dates':ds,'observation_date':ds[0] if len(ds)==1 else None}


class GroupAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='jev_dated_group_');self.addCleanup(self.tmp.cleanup)
        for name,replacement in {'lemmatize_for_bm25':lambda text:text.lower(),'extract_entities_batch':lambda texts:[[] for _ in texts],
            'capture_event':lambda *a,**k:None,'detect_scale_threshold_from_add_result':lambda *a:None,
            'detect_temporal_usage_from_metadata':lambda *a:None,'display_first_run_notice':lambda *a:None}.items():
            manager=patch.object(fixtures.module,name,replacement);manager.start();self.addCleanup(manager.stop)
    def adapter(self,responses):
        memory=fixtures.fake_memory(responses);memory.embedding_model=Embedding()
        return memory,w.Mem0GroupAdapter(memory,self.tmp.name,current_date=w.CURRENT_DATE)
    def test_mixed_dates_are_null_from_native_store_and_original_prompt_unchanged(self):
        memory,adapter=self.adapter([{'memory':[{'text':'Alice moved to Paris on April 30, 2023.'}]}]);env=envelope()
        before=memory.embedding_model;record=adapter.add_group(user_id='c0',**env)
        self.assertEqual(record['status'],'success');fact=record['fact_records'][0]
        self.assertIsNone(fact['observation_date']);self.assertEqual(fact['source_observation_dates'],['2023-05-01','2023-06-10'])
        stored=memory.vector_store.get(fact['id']);self.assertIsNone(stored.payload['jev_mem_observation_date'])
        prompts=memory.llm.calls[0]['messages'];self.assertEqual(prompts[0]['content'],ADDITIVE_EXTRACTION_PROMPT)
        self.assertIn(w.MULTIPLE_DATES,prompts[1]['content'])
        for msg,src in zip(env['messages'],env['turn_sources']):
            self.assertIn(msg['content'],prompts[1]['content']);self.assertIn(src['observation_date'],prompts[1]['content']);self.assertIn(src['source_turn_id'],prompts[1]['content'])
        self.assertEqual(memory.db.saved[0][0],env['messages'])
        self.assertEqual(before.calls[0],(parse_messages(env['messages']),'search'))
        self.assertIs(memory.embedding_model,before);self.assertIs(fixtures.module.generate_additive_extraction_prompt,generate_additive_extraction_prompt)
        self.assertEqual(fact['verified_support'],[])
    def test_single_date_groups_keep_iso_date(self):
        memory,adapter=self.adapter([{'memory':[{'text':'Alice moved.'}]}]);env=envelope(dates=('2023-05-01','2023-05-01'))
        record=adapter.add_group(user_id='c0',**env)
        self.assertEqual(record['fact_records'][0]['observation_date'],'2023-05-01')
        self.assertIn('## Observation Date\n2023-05-01',memory.llm.calls[0]['messages'][1]['content'])
    def test_fake_group_date_rejected_before_sdk_call(self):
        memory,adapter=self.adapter([]);env=envelope();env['observation_date']='2023-06-10'
        with self.assertRaisesRegex(ValueError,'must not claim'):adapter.add_group(user_id='c0',**env)
        self.assertEqual(memory.llm.calls,[])
    def test_explicit_empty_extraction_preserves_original_contract_and_is_flagged(self):
        memory,adapter=self.adapter([{'memory':[]}]);env=envelope()
        result=adapter.add_group(user_id='c0',**env)
        again=adapter.add_group(user_id='c0',**env)
        record=json.loads(next(Path(self.tmp.name).glob('*/*.json')).read_text(encoding='utf8'))
        self.assertEqual(record['status'],'success');self.assertTrue(record['explicit_empty_extraction'])
        self.assertTrue(record['empty_extraction_source_audit_pending']);self.assertEqual(len(memory.llm.calls),1)
        self.assertEqual(result['fact_records'],[]);self.assertTrue(again['replayed'])
    def test_malformed_json_stops_further_groups(self):
        memory,adapter=self.adapter(['invalid json']);env=envelope()
        with self.assertRaises(Exception):adapter.add_group(user_id='c0',**env)
        with self.assertRaises(w.original.SessionRecoveryRequired):adapter.add_group(user_id='c0',**env)
    def test_partial_embedding_cannot_complete(self):
        memory,adapter=self.adapter([{'memory':[{'text':'fact one'},{'text':'fact two'}]}]);env=envelope()
        memory.embedding_model.embed_batch=lambda texts,action:[[0.0]*767+[1.0]]
        with self.assertRaises(w.original.IncompleteSessionError):adapter.add_group(user_id='c0',**env)
        record=json.loads(next(Path(self.tmp.name).glob('*/*.json')).read_text(encoding='utf8'))
        self.assertEqual(record['status'],'requires_recovery');self.assertEqual(record['persistence_check']['missing_texts'],['fact two'])
    def test_success_resume_makes_no_new_call(self):
        memory,adapter=self.adapter([{'memory':[{'text':'a fact'}]}]);env=envelope()
        first=adapter.add_group(user_id='c0',**env);again=adapter.add_group(user_id='c0',**env)
        self.assertTrue(again['replayed']);self.assertEqual(first['results'],again['results']);self.assertEqual(len(memory.llm.calls),1)


class ContextEmbeddingTests(unittest.TestCase):
    def test_short_group_and_nonmatching_embedding_delegate_unchanged(self):
        delegate=Embedding();messages=[{'role':'user','content':'Hi'}];proxy=w.GroupContextEmbeddings(delegate,messages,['D1:1'])
        vector=proxy.embed(parse_messages(messages),'search')
        self.assertEqual(vector,[1.]+[0.]*767);self.assertFalse(proxy.audit['triggered'])
        count=len(delegate.api.calls);proxy.embed('other query','search');proxy.embed('fact','add');proxy.embed_batch(['a','b'],'add')
        self.assertEqual(len(delegate.api.calls),count)
        self.assertEqual(delegate.calls[-4:],[('other query','search'),('fact','add'),('a','add'),('b','add')])
    def test_long_context_exact_full_coverage_and_weighted_fusion(self):
        delegate=Embedding();messages=[{'role':'user','content':'a'*1100},{'role':'assistant','content':'b'*2200},{'role':'user','content':'c'*900}]
        def vector(text,action='add'):
            delegate.calls.append((text,action));v=[0.]*768;v[0 if 'a' in text else 1]=1.;return v
        delegate.embed=vector;proxy=w.GroupContextEmbeddings(delegate,messages,['D1:1','D2:1','D3:1'])
        actual=proxy.embed(parse_messages(messages),'search');audit=proxy.audit
        self.assertTrue(audit['triggered']);self.assertEqual(audit['status'],'complete')
        self.assertEqual(''.join(t for t,a in delegate.calls),parse_messages(messages))
        self.assertTrue(all(len(t)<=2048 and a=='search' for t,a in delegate.calls))
        self.assertEqual(sum(c['end_char']-c['start_char'] for c in audit['chunks']),len(parse_messages(messages)))
        weight0=sum(c['tokens'] for c,(t,a) in zip(audit['chunks'],delegate.calls) if 'a' in t)
        weight1=sum(c['tokens'] for c,(t,a) in zip(audit['chunks'],delegate.calls) if 'a'not in t)
        norm=math.sqrt(weight0**2+weight1**2)
        self.assertAlmostEqual(actual[0],weight0/norm);self.assertAlmostEqual(actual[1],weight1/norm)
        self.assertEqual({x for c in audit['chunks'] for x in c['source_turn_ids']},{'D1:1','D2:1','D3:1'})
    def test_single_long_turn_character_split_preserves_unicode(self):
        delegate=Embedding();messages=[{'role':'user','content':'中文🙂'*3000}]
        proxy=w.GroupContextEmbeddings(delegate,messages,['D1:1']);proxy.embed(parse_messages(messages),'search')
        self.assertEqual(''.join(t for t,a in delegate.calls),parse_messages(messages))
        self.assertTrue(all(len(t)<=2048 for t,a in delegate.calls))
        self.assertTrue(all(c['source_turn_ids']==['D1:1'] for c in proxy.audit['chunks']))

if __name__=='__main__':unittest.main()
