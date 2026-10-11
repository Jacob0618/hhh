"""Real local-Qdrant numeric compatibility tests; no network or models."""
import copy
from pathlib import Path
import sqlite3
import socket
import tempfile
import unittest
from unittest.mock import patch
import uuid

from qdrant_client import QdrantClient, models
from mem0.vector_stores.qdrant import Qdrant
from tools import r07_native_numeric_replay as gate


def uid(value):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


class NumericReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r07-test-numeric-')
        self.root = Path(self.temp.name)
        self.source = self.root/'original'
        self.source.mkdir()
        self.block = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        self.block.start()
        con = sqlite3.connect(self.source/'history.sqlite')
        con.execute('CREATE TABLE history (id TEXT, memory_id TEXT, event TEXT)')
        con.execute('INSERT INTO history VALUES (?,?,?)', ('h', uid('oldfact'), 'ADD'))
        con.commit(); con.close()
        client = QdrantClient(path=str(self.source/'qdrant'))
        for name in ('facts', 'facts_entities'):
            client.create_collection(name, vectors_config={'':models.VectorParams(size=3, distance=models.Distance.COSINE)},
                                     sparse_vectors_config={'bm25':models.SparseVectorParams()})
        def point(name, text, entity=False):
            payload = {'data':text, 'user_id':'u'}
            if entity:
                payload['linked_memory_ids'] = [uid('oldfact')]
            return models.PointStruct(id=uid(name), payload=payload,
                vector={'':[.21,.53,.81], 'bm25':models.SparseVector(indices=[1,5], values=[.5,1.25])})
        client.upsert('facts', [point('oldfact', 'Old fact')])
        client.upsert('facts_entities', [point('e1', 'Alice', True), point('e2', 'Bob', True),
                                        point('e3', 'Unchanged entity', True)])
        client.close()
        self.before = gate.original.snapshot(self.source)
        self.inventory = gate.original.inventory(self.source)
        target = self.root/'observed'
        gate.original.copy_closed(self.source, target, self.inventory)
        client = QdrantClient(path=str(target/'qdrant'))
        store = Qdrant.__new__(Qdrant);store.client=client;store.collection_name='facts_entities'
        # Independent actual search/update execution constructs observed output.
        store.search(query='unmatched name', vectors=[.1,.2,.3], top_k=1, filters={'user_id':'u'})
        payload = copy.deepcopy(self.before['entities'][uid('e1')]['payload'])
        payload['linked_memory_ids'].append(uid('newfact'))
        store.update(vector_id=uid('e1'), vector=None, payload=payload)
        store.search(query='Alice', vectors=[.3,.4,.5], top_k=1, filters={'user_id':'u'})
        payload = copy.deepcopy(self.before['entities'][uid('e2')]['payload'])
        payload['linked_memory_ids'].append(uid('newfact'))
        store.update(vector_id=uid('e2'), vector=None, payload=payload)
        client.close()
        self.after = gate.original.snapshot(target)
        self.after['facts'][uid('newfact')] = {'payload':{'data':'New fact', 'user_id':'u'}, 'vector':{'':[1.,0.,0.]}}
        self.after['entities'][uid('newentity')] = {'payload':{'data':'New entity','user_id':'u','linked_memory_ids':[uid('newfact')]}, 'vector':{'':[1.,0.,0.]}}
        def trace(name, kind):
            return {'memory_id':uid('newfact'),'selected_entity_id':uid(name),'selection_kind':kind,
                    'text':name,'normalized_text':name}
        traces = [trace('newentity','native_insert'),trace('e1','native_exact'),trace('e2','native_semantic')]
        self.additions = [{'candidate':{'candidate_id':'c0_s001_m00','statement':'New fact'},
                           'memory_id':uid('newfact'),'status':'added','entity_trace':traces,
                           'expected_entity_keys':[t['normalized_text'] for t in traces]}]

    def tearDown(self):
        self.block.stop()
        self.temp.cleanup()

    def replay(self, after=None, additions=None):
        return gate.replay_original_entities(self.source, self.before,
            self.after if after is None else after, self.additions if additions is None else additions, 'u')

    def test_real_full_matrix_search_persistence_exact_and_source_unchanged(self):
        report = self.replay()
        self.assertTrue(report['native_search_update_replay_exact'])
        self.assertEqual(report['original_entity_count'],3)
        self.assertEqual(report['semantic_search_operations'],2)
        self.assertFalse(report['epsilon_used']);self.assertFalse(report['full_memory_add_replayed'])
        self.assertEqual(gate.original.inventory(self.source), self.inventory)
        self.assertEqual(self.after['entities'][uid('e3')],self.before['entities'][uid('e3')])

    def test_two_semantic_searches_update_distinct_old_entities(self):
        # The observed native sequence already performed two searches followed
        # by updates of e1/e2; replace the insert/exact trace decomposition with
        # two recorded semantic matches, without changing observed old vectors.
        after=copy.deepcopy(self.after);del after['entities'][uid('newentity')]
        rows=copy.deepcopy(self.additions);rows[0]['entity_trace'].pop(0)
        rows[0]['entity_trace'][0]['selection_kind']='native_semantic'
        rows[0]['expected_entity_keys']=['e1','e2']
        report=self.replay(after,rows)
        self.assertEqual(report['semantic_search_operations'],2)
        self.assertEqual(report['old_entity_payload_updates'],2)

    def test_zero_add_all_old_entities_still_compared(self):
        report = self.replay(copy.deepcopy(self.before), [])
        self.assertEqual(report['trace_operations'],0)
        self.assertEqual(report['changed_dense_entity_ids'],[])

    def test_reject_tiny_unexplained_dense_delta(self):
        after=copy.deepcopy(self.after);after['entities'][uid('e1')]['vector'][''][0]+=1e-12
        with self.assertRaises(ValueError):self.replay(after)

    def test_reject_sparse_payload_lost_link_and_original_fact_changes(self):
        for field in ('sparse','payload','link','fact'):
            after=copy.deepcopy(self.after)
            if field=='sparse':after['entities'][uid('e1')]['vector']['bm25']['values'][0]+=.01
            elif field=='payload':after['entities'][uid('e1')]['payload']['data']='Other person'
            elif field=='link':after['entities'][uid('e1')]['payload']['linked_memory_ids'].remove(uid('oldfact'))
            else:after['facts'][uid('oldfact')]['payload']['data']='Changed'
            with self.subTest(field=field), self.assertRaises(ValueError):self.replay(after)

    def test_reject_unexplained_extra_link(self):
        after=copy.deepcopy(self.after);after['entities'][uid('e1')]['payload']['linked_memory_ids'].append('rogue')
        with self.assertRaises(ValueError):self.replay(after)

    def test_reject_trace_context_unknown_kind_missing_operation_or_target(self):
        for field in ('context','kind','missing','target'):
            rows=copy.deepcopy(self.additions)
            if field=='context':rows[0]['entity_trace'][0]['memory_id']='other'
            elif field=='kind':rows[0]['entity_trace'][0]['selection_kind']='unknown'
            elif field=='missing':rows[0]['entity_trace'].pop()
            else:rows[0]['entity_trace'][0]['selected_entity_id']=uid('e1')
            with self.subTest(field=field), self.assertRaises(ValueError):self.replay(additions=rows)

    def test_reject_match_before_insert_and_wrong_scope(self):
        rows=copy.deepcopy(self.additions);rows[0]['entity_trace'][0]['selection_kind']='native_semantic'
        with self.assertRaises(ValueError):self.replay(additions=rows)
        with self.assertRaises(ValueError):gate.replay_original_entities(self.source,self.before,self.after,self.additions,'other')

    def test_reject_wrong_original_snapshot(self):
        before=copy.deepcopy(self.before);before['history'][0]['event']='UPDATE'
        with self.assertRaises(ValueError):gate.replay_original_entities(self.source,before,self.after,self.additions,'u')


if __name__=='__main__':unittest.main()
