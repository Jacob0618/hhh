"""Continuation transport tests: actual retained raw, no credentials/network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import run_five_baselines_format_v2 as mod
from tools.baselines_common_v1 import BaselineFatal, read, sha

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.new=Path(self.tmp.name)/'amem_format_v2'
        self.bound=mod.proof()
        for item in (patch.object(mod,'NEW',self.new),patch.object(mod,'_proof',self.bound),
                     patch('jev_mem.providers.load_credentials',side_effect=AssertionError('credentials forbidden')),
                     patch('requests.sessions.Session.request',side_effect=AssertionError('network forbidden'))):
            item.start();self.addCleanup(item.stop)

    def api(self,ci=0):
        return mod.PrefixAPI(self.new/'work/api_method',keys={'unused':'stub-test'},context={'method':'amem','conversation':ci,'phase':'build'})

    def send(self,api,key):
        record=self.bound['records'][key];saved=read(record['raw_path']);sig=saved['request']
        return api.request(sig['provider'],sig['model'],sig['body'],stage=record['stage'])

    def test_all_forty_exact_original_raw_reuse_and_audit(self):
        for ci in range(10):
            api=self.api(ci)
            for key in self.bound['prefixes'][str(ci)]:
                value=self.send(api,key)
                self.assertEqual(value,read(self.bound['records'][key]['raw_path'])['response'])
                self.assertEqual(api.events[-1]['source'],'inherited_exact_request')
            self.assertEqual(api.prefix_position,4)
        audit=mod.audit_requests(self.new)
        self.assertEqual(audit['errors'],[])
        self.assertEqual(audit['inherited_unique_prefix_requests'],40)
        self.assertEqual(list((self.new/'work/api_method/request_intents').glob('*.json')),[])

    def test_wrong_prefix_order_blocks_before_dispatch(self):
        api=self.api()
        with self.assertRaises(BaselineFatal):self.send(api,self.bound['prefixes']['0'][1])
        self.assertEqual(api.events,[])

    def test_wrong_conversation_blocks_before_dispatch(self):
        api=self.api()
        with self.assertRaises(BaselineFatal):self.send(api,self.bound['prefixes']['1'][0])
        self.assertEqual(api.events,[])

    def test_unapproved_root_rejected(self):
        with self.assertRaises(ValueError):
            mod.PrefixAPI(self.new/'work/api_method',keys={'unused':'stub-test'},cache_roots=[self.new/'elsewhere'])

    def test_retrieval_scoped_api_does_not_inherit_build_roots(self):
        api=self.api();child=api.scoped(query_id='c0_q0',phase='retrieval')
        self.assertEqual(child.cache_roots,())
        self.assertFalse(child.prefix_admitted)

    def test_other_method_cannot_inherit(self):
        api=mod.PrefixAPI(self.new/'work/api_method',keys={'unused':'stub-test'},context={'method':'simplemem','conversation':0,'phase':'build'})
        self.assertEqual(api.cache_roots,())

    def test_bad_bound_hash_rejected_without_fallback(self):
        api=self.api();key=self.bound['prefixes']['0'][0]
        self.bound['records'][key]['raw_sha256']='0'*64
        with self.assertRaises(BaselineFatal):self.send(api,key)
        self.assertEqual(api.events,[])

    def test_tampered_lineage_rejected(self):
        for ci in range(10):
            api=self.api(ci)
            for key in self.bound['prefixes'][str(ci)]:self.send(api,key)
        p=next((self.new/'work/api_method/cache_lineage').glob('*.json'))
        value=read(p);value['source_path']='other.json';p.write_text(json.dumps(value),encoding='utf-8')
        self.assertTrue(mod.audit_requests(self.new)['errors'])

if __name__=='__main__':unittest.main()
