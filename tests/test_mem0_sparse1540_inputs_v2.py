"""Pure admission checks: namespace isolation, exact union and byte bindings."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sqlite3
import tempfile
import unittest
from tools import audit_mem0_sparse1540_inputs_v2 as target


def minimal_map():
    qs=[{'id':'q'+str(i)} for i in range(1540)]
    return {'policy':target.POLICY,'question_ids':[q['id'] for q in qs],
        'rows':[{'conversation':i,'arm':a,'index_recovery':{'path':'result'} if i==3 else None}
                for a in target.base.ARMS for i in range(10)],
        'provider_retries':2,'conversation_workers':8,'query_workers':4,'final_unique_facts':20},qs


class InputAuditTests(unittest.TestCase):
    def setUp(self):target._AUDITED_FILES.clear()

    def test_complete_twenty_rows_with_two_recoveries(self):
        m,qs=minimal_map()
        self.assertEqual(len(target.validate_map_shape(m,qs)),20)

    def test_missing_duplicate_bool_and_unrepaired_conv3_rejected(self):
        for kind in ('missing','duplicate','boolean','not_recovered'):
            m,qs=minimal_map()
            if kind=='missing':m['rows'].pop()
            if kind=='duplicate':m['rows'][-1]=m['rows'][0]
            if kind=='boolean':m['rows'][1]['conversation']=True
            if kind=='not_recovered':m['rows'][3]['index_recovery']=None
            with self.assertRaises(ValueError):target.validate_map_shape(m,qs)

    def test_budget_or_question_population_change_rejected(self):
        m,qs=minimal_map();m['final_unique_facts']=19
        with self.assertRaises(ValueError):target.validate_map_shape(m,qs)
        m,qs=minimal_map();m['question_ids'][0]='unseen'
        with self.assertRaises(ValueError):target.validate_map_shape(m,qs)

    def entities(self):
        return {'e':{'payload':{'data':'video games','linked_memory_ids':['old']},'vector':[1,0]},
                'other':{'payload':{'data':'Nate','linked_memory_ids':['old']},'vector':[0,1]}}

    def test_exact_one_union_passes_and_preserves_input(self):
        before=self.entities();after=deepcopy(before);after['e']['payload']['linked_memory_ids']=['f','old']
        result=target.validate_exact_union({'f':'fact'},{'f':'fact'},before,after,'f','e')
        self.assertEqual(result,after);self.assertEqual(before,self.entities())

    def test_fact_vector_unrelated_payload_or_partial_link_cleanup_rejected(self):
        before=self.entities();after=deepcopy(before);after['e']['payload']['linked_memory_ids']=['f','old']
        with self.assertRaises(ValueError):target.validate_exact_union({'f':'fact'},{'f':'edited'},before,after,'f','e')
        after['other']['payload']['linked_memory_ids'].append('f')
        with self.assertRaises(ValueError):target.validate_exact_union({'f':'fact'},{'f':'fact'},before,after,'f','e')
        after=deepcopy(before);after['e']['payload']['linked_memory_ids']=['f','old'];after['e']['vector']=[.5,.5]
        with self.assertRaises(ValueError):target.validate_exact_union({'f':'fact'},{'f':'fact'},before,after,'f','e')

    def test_not_orphan_or_ambiguous_exact_entity_rejected(self):
        before=self.entities();before['other']['payload']['linked_memory_ids'].append('f')
        with self.assertRaises(ValueError):target.validate_exact_union({}, {},before,before,'f','e')

    def test_cross_export_tolerance_never_allows_actual_store_mutation(self):
        a=[1.]+[0.]*767;b=[1.-4e-8]+[0.]*767
        self.assertGreater(target.validate_bank_vector(a,b,historical_export=True),0)
        with self.assertRaises(ValueError):target.validate_bank_vector(a,b,historical_export=False)
        before=self.entities();after=deepcopy(before);after['e']['payload']['linked_memory_ids']=['f','old']
        with self.assertRaises(ValueError):target.validate_exact_union({'f':a},{'f':b},before,after,'f','e')
        before=self.entities();before['other']['payload']['data']='Video Games'
        with self.assertRaises(ValueError):target.validate_exact_union({}, {},before,before,'f','e')

    def test_raw_points_read_into_memory_does_not_modify_database(self):
        from qdrant_client.http.models import PointStruct,Record
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'storage.sqlite';db=sqlite3.connect(path)
            db.execute('CREATE TABLE points (id TEXT PRIMARY KEY, point BLOB)')
            p=PointStruct(id='a',vector={'':[1.,0.]},payload={'data':'Fact'})
            db.execute('INSERT INTO points VALUES (?,?)',('a',pickle.dumps(p)));db.commit();db.close()
            raw=path.read_bytes();result,digest=target.canonical_points(path)
            self.assertEqual(result['a'],Record(id='a',vector=[1.,0.],payload={'data':'Fact'}).model_dump(mode='json'))
            self.assertEqual(path.read_bytes(),raw)
            self.assertIn(str(path.resolve()),target._AUDITED_FILES)

    def request(self):
        sig=target.retrieval_audit.signature(['Question?'])
        return [{'role':'query','stage':'mem0_embedding_search','signature':sig}],sig

    def event(self,arm='session_rank',qid='c3_q1',status=200,line=1):
        specs,sig=self.request()
        return {'_line':line,'context':{'architecture':target.POLICY,'conversation':3,'method':arm,
            'phase':'retrieve','query_id':qid},'provider':'gemini','model':'gemini-embedding-001',
            'operation':'batchEmbedContents','stage':'mem0_embedding_search',
            'request_hash':target.base.request_digest(sig),'status':status,'attempt':line,'cache_hit':False}

    def consumer(self,rows):
        specs,sig=self.request();cache={'request':sig,'stage':'mem0_embedding_search',
            'response':{'embeddings':[{'values':[1.]+[0.]*767}]}}
        return target.recovery_consumer_audit(rows,'session_rank','c3_q1',specs,lambda h:(cache,{'file_sha256':'fake'}),set())

    def test_v2_same_context_success_and_retry_are_accepted(self):
        self.assertTrue(self.consumer([self.event()])['admitted'])
        self.assertTrue(self.consumer([self.event(status=503),self.event(line=2)])['admitted'])

    def test_other_arm_or_other_question_cannot_restore_v2_error(self):
        for second in (self.event(arm='regroup_rank',line=2),self.event(qid='c3_q2',line=2)):
            self.assertFalse(self.consumer([self.event(status=503),second])['admitted'])

    def test_namespace_mapping_preserves_original_ledger_objects(self):
        rows=[self.event()];prior=deepcopy(rows);result=self.consumer(rows)
        self.assertEqual(rows,prior)
        self.assertEqual(result['original_context']['architecture'],target.POLICY)
        self.assertEqual(result['original_context_rows_sha256'],target.base.fingerprint(rows))

    def test_audited_files_detect_mid_audit_change(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'evidence.json';p.write_text('{}');target.read(p);p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):target.read(p)


if __name__=='__main__':unittest.main()
