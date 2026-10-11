"""Offline regression checks for post-failure export eligibility and read-only IO."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import mem0_regroup_firstpass_review_v2 as m

class ReviewTests(unittest.TestCase):
    def setUp(self):
        m.HASHES.clear();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.blockers=[patch('socket.socket.connect',side_effect=AssertionError('No network')),patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credentials'))]
        for p in self.blockers:p.start();self.addCleanup(p.stop)
    def eligible(self,arm=None):
        arm=arm or m.ARMS[0]
        return {'conversation':3,'arm':arm,'original_write_status':'failed','complete_fact_groups':True,'groups':40,'expected_groups':40,
            'facts_dense_sparse_and_history_valid':True,'existing_entity_dense_and_links_valid':True,'orphan_fact_ids':[m.ORPHANS[arm]]},
    def test_only_reviewed_twin_full_banks_eligible(self):
        for arm in m.ARMS:
            row=self.eligible(arm)[0];proof={'verified':True,'mechanism':'same_canonical_stale_payload_overwrite','fact_id':m.ORPHANS[arm]}
            self.assertTrue(m.export_eligible(row,proof))
            for field,value in [('complete_fact_groups',False),('groups',13),('conversation',8),('original_write_status','complete'),('orphan_fact_ids',[]),('facts_dense_sparse_and_history_valid',False),('existing_entity_dense_and_links_valid',False)]:
                other=dict(row);other[field]=value;self.assertFalse(m.export_eligible(other,proof))
            self.assertFalse(m.export_eligible(row,dict(proof,verified=False)))
    def test_all16_terminal_required_before_export(self):
        states=[{'conversation':c,'arm':a,'status':'complete'} for c in range(2,10) for a in m.ARMS]
        m.require_terminal(states)
        for status in ('running','not_started',None):
            altered=copy.deepcopy(states);altered[0]['status']=status
            with self.assertRaises(ValueError):m.require_terminal(altered)
        with self.assertRaises(ValueError):m.require_terminal(states[:-1])
    def test_readonly_sqlite_keeps_exact_bytes(self):
        p=self.root/'tiny.sqlite';c=sqlite3.connect(p);c.execute('CREATE TABLE history (memory_id TEXT)');c.execute("INSERT INTO history VALUES ('x')");c.commit();c.close()
        before=p.read_bytes();self.assertEqual(m.rows(p,'history'),[{'memory_id':'x'}]);self.assertEqual(p.read_bytes(),before)
        with self.assertRaises(sqlite3.Error):m.rows(p,'history; DELETE FROM history')
        self.assertEqual(p.read_bytes(),before)
    def fixture(self):
        f={'id':'x','text':'test','observation_date':'2023-01-01','source_observation_dates':['2023-01-01'],'input_scope':{'group_id':'g','source_turn_ids':['D1:1']}}
        payload={'data':'test','hash':hashlib.md5(b'test').hexdigest(),'jev_mem_group_id':'g','jev_mem_observation_date':'2023-01-01','jev_mem_source_observation_dates':['2023-01-01'],'jev_mem_source_turn_ids':['D1:1']}
        p=SimpleNamespace(payload=payload,vector={'':[1.]+[0.]*767,'bm25':SimpleNamespace(indices=[1],values=[.5])})
        h={'memory_id':'x','event':'ADD','is_deleted':0,'old_memory':None,'new_memory':'test'}
        return f,p,h
    def test_missing_fact_history_or_required_vector_blocks(self):
        f,p,h=self.fixture();m.validate_records([f],{'x':p},[h])
        with self.assertRaises(ValueError):m.validate_records([f],{},[h])
        with self.assertRaises(ValueError):m.validate_records([f],{'x':p},[])
        with self.assertRaises(ValueError):m.validate_records([f,f],{'x':p},[h])
        for key in ('','bm25'):
            other=copy.deepcopy(p);other.vector.pop(key)
            with self.assertRaises(ValueError):m.validate_records([f],{'x':other},[h])
        p.payload['jev_mem_observation_date']='2024-01-01'
        with self.assertRaises(ValueError):m.validate_records([f],{'x':p},[h])
    def test_seal_mutation_and_nonempty_destination_rejected(self):
        p=self.root/'r.json';m.save(p,{'a':1});self.assertEqual(m.verify(p)['a'],1)
        with self.assertRaises(FileExistsError):m.save(p,{'a':2})
        p.write_text(json.dumps({'a':2,'content_sha256':'bad'}))
        with self.assertRaises(ValueError):m.verify(p)
    def test_check_rejects_unreviewed_export_before_native_reads(self):
        p=self.root/'bad.json';m.save(p,{'schema':'mem0-regroup-reviewed-export-v2','source_kind':'reviewed_export','original_write_status':'failed','reviewed_export_status':'complete','native_index_policy':'original_unrepaired','conversation':8,'arm':m.ARMS[1]})
        with patch.object(m,'native_inventory',side_effect=AssertionError('Do not read source')):
            with self.assertRaises(ValueError):m.check_export_bank(p)

if __name__=='__main__':unittest.main()
