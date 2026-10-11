"""Second-review finite observer boundaries; virtual artifacts and no CIM."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T1152/test_hypermem_native_v4_observer_v1.py'
spec=importlib.util.spec_from_file_location('_observer_author_factory_only',P)
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
o=factory.o


def replacement_fixture():
    r,m,p,expected,_=factory.binding_fixture();m['_observed_manifest_sha']=expected['manifest']
    cert=next(iter(p['replacement_cases'].values()));key=cert['request_hash'];api=o.RUN/'work/api_method'
    marker={'schema':'hypermem-v4-exclusive-role-replacement-v1','created_utc':'2026-10-09T13:00:00Z',
        'manifest':o.ref(o.MANIFEST,expected['manifest']),'terminal_proof':m['terminal_proof'],'human_grant':m['human_grant'],
        'certificate':cert,'context':cert['context'],'signature':cert['signature'],'stage':cert['stage'],
        'request_hash':key,'prefix_ordinal':cert['prefix_ordinal'],'permission_consumed_before_transport':True,'logical_replacements':1}
    r.put(o.RUN/'work/replacement_v4'/(key+'.json'),marker)
    r.put(api/'request_intents'/(key+'.json'),{'request_hash':key,'signature':cert['signature'],
        'context':cert['context'],'stage':cert['stage'],'created_utc':'2026-10-09T13:00:01Z','retries':2})
    identity={k:cert['signature'][k] for k in ('provider','model','operation')}
    rows=[{**identity,'request_hash':key,'context':cert['context'],'stage':cert['stage'],
        'timestamp_utc':'2026-10-09T13:00:02Z','attempt':1,'status':'transport_error'},
        {**identity,'request_hash':key,'context':cert['context'],'stage':cert['stage'],
        'timestamp_utc':'2026-10-09T13:00:03Z','attempt':2,'status':200}]
    return r,m,cert,rows


class IndependentObserver(unittest.TestCase):
    def test_exact_transport_error_recovery_and_unknown_status(self):
        r,m,c,rows=replacement_fixture()
        self.assertFalse(o.replacement_health(r,m,c,rows,[])['issues_or_inflight'])
        for status in (None,400,200,True,'unknown'):
            bad=deepcopy(rows);bad[0]['status']=status
            with self.subTest(status=status):
                self.assertTrue(o.replacement_health(r,m,c,bad,[])['issues_or_inflight'])

    def test_physical_full_identity_and_attempt_types(self):
        r,m,c,rows=replacement_fixture()
        for field,value in (('provider','other'),('model','other'),('operation','other'),('attempt',True),('attempt',1.0)):
            bad=deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field,value=value):
                self.assertTrue(o.replacement_health(r,m,c,bad,[])['issues_or_inflight'])

    def test_pending_scope_does_not_probe_process_or_initialize_core(self):
        with patch.object(o,'processes',side_effect=AssertionError('CIM forbidden')), \
                patch.object(o,'prefix_health',side_effect=AssertionError('No production reads')), \
                patch.object(o,'ObserverReader',return_value=factory.MemoryReader()):
            value=o.snapshot({k:None for k in ('manifest','review','entry','proof','human_grant')})
        self.assertEqual(value['binding']['status'],'pending_final_bindings')
        self.assertIsNone(value['usage']['experiment_tokens'])
        self.assertIsNone(value['usage']['experiment_cost_usd'])
        self.assertFalse(value['processes']['probe_ok'])

    def test_foreign_binary_and_reparse_rejected_before_shared_read(self):
        with patch.object(o.native,'shared_read',side_effect=AssertionError('No actual file read')):
            for path in (Path('D:/not_jev_workspace/answer.json'),o.RUN/'work/native/c0/indices/meta.json',
                         o.RUN/'work/native/c0/cache/embedding.npy'):
                with self.subTest(path=str(path)),self.assertRaises(ValueError):o.ObserverReader().raw(path)
            with patch.object(Path,'is_symlink',return_value=True),self.assertRaises(ValueError):
                o.ObserverReader().raw(o.RUN/'status.json')


if __name__=='__main__':unittest.main(verbosity=2)
