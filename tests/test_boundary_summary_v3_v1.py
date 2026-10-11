"""Real C raw-manifest shape regression; no QA/Reader/Judge/native execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from tools import summarize_boundary_variants_full1540_v3 as v3
from tools import summarize_boundary_variants_full1540_v2 as v2
from tools import summarize_boundary_variants_full1540_v1 as v1
from tools import run_boundary_variants_qa_v2 as qa
from tools.check_experiment_health import shared_read

REAL=Path('runs/boundary_multiscale_qa_full1540_v2_20261003')


class HydrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=json.loads(shared_read(REAL/'run_manifest.json'))
        cls.source=json.loads(shared_read(REAL/'source_jobs.json'))
        cls.out=json.loads(shared_read(REAL/'run_outcome.json'))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.m=copy.deepcopy(self.raw);self.s=copy.deepcopy(self.source)
        v3.old.plain.publish_json(self.run/'run_manifest.json',self.m)
        v3.save_new(self.run/'run_outcome.json',{k:v for k,v in self.out.items() if k!='content_sha256'})
        v3.save_new(self.run/'retrieval_outcome.json',{'terminal':True})
        self.s['retrieval_outcome']=v3.old.plain.pointer(self.run/'retrieval_outcome.json')
        self.s.pop('content_sha256',None)
        v3.save_new(self.run/'source_jobs.json',self.s)
        seal={'schema':qa.SCHEMA+'-firstpass-seal','total':1540,
            'protocol_sha256':self.m['protocol_sha256'],
            **{k:v3.old.plain.pointer(self.run/n) for k,n in
               [('manifest','run_manifest.json'),('outcome','run_outcome.json'),('source_jobs','source_jobs.json')]}}
        v3.save_new(self.run/'firstpass_seal.json',seal)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(qa,'manifest',return_value=self.m))
        self.stack.enter_context(patch.object(v3.locktool,'verify_lock',return_value={}))
        expected={j['query_id']:{k:j[k] for k in ('query_id','conversation','category','question','reference','reference_date')}
                  for j in self.s['jobs']}
        self.stack.enter_context(patch.object(v3.full,'canonical',return_value=expected))

    def test_real_raw_shape_hydrated_from_seal_and_original_predicate(self):
        self.assertNotIn('jobs',self.raw)
        received=[]
        with patch.object(qa,'recovery_jobs',side_effect=lambda m,o:received.append(m) or ['pending']),\
             patch.object(qa,'seal_jobs',side_effect=AssertionError('Source write forbidden')),\
             patch.object(qa,'execution_manifest',side_effect=AssertionError('Source write path forbidden')):
            facade,evidence=v3.bookkeeping_driver(self.run)
            self.assertEqual(facade.recovery_jobs(self.m,v3.old.plain.verify(self.run/'run_outcome.json')),['pending'])
        self.assertEqual(received[0]['jobs'],self.s['jobs'])
        self.assertEqual(evidence['jobs'],1540)
        self.assertEqual(evidence['model_calls'],0)
        self.assertNotIn('jobs',self.m)

    def test_missing_or_tampered_source_refused_without_reseal(self):
        path=self.run/'source_jobs.json';original=path.read_bytes()
        for data in (None,original+b' '):
            if data is None:path.unlink()
            else:path.write_bytes(data)
            with self.assertRaises((ValueError,FileNotFoundError,v3.InvalidArtifact)):
                v3.bookkeeping_driver(self.run)
            if data is None:self.assertFalse(path.exists())
            path.write_bytes(original)

    def test_existing_hydrated_firstpass_checked_not_replaced(self):
        facade,_=v3.bookkeeping_driver(self.run)
        m=copy.deepcopy(self.m);m.update(jobs=self.s['jobs'],source_manifest=v3.old.plain.pointer(self.run/'source_jobs.json'))
        out=v3.old.plain.verify(self.run/'run_outcome.json')
        with patch.object(qa,'recovery_jobs',return_value=['pending']) as original:
            self.assertEqual(facade.recovery_jobs(m,out),['pending'])
            m['jobs']=m['jobs'][:-1]
            with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(m,out)
            self.assertEqual(original.call_count,1)

    def test_technical_recovery_subset_preserved_and_original_rejection(self):
        facade,_=v3.bookkeeping_driver(self.run)
        m={'phase_kind':'technical_recovery','jobs':[{'query_id':'subset'}]}
        with patch.object(qa,'recovery_jobs',side_effect=ValueError('Only one postbatch technical recovery allowed')) as original:
            with self.assertRaisesRegex(ValueError,'Only one'):facade.recovery_jobs(m,{})
        self.assertIs(original.call_args.args[0],m)
        self.assertEqual(m['jobs'],[{'query_id':'subset'}])

    def test_original_namespaces_and_functions_unchanged(self):
        before=(v1.driver,v2.driver,qa.recovery_jobs,qa.manifest,v2.summarize,v1.summarize)
        v3.bookkeeping_driver(self.run)
        self.assertEqual(before,(v1.driver,v2.driver,qa.recovery_jobs,qa.manifest,v2.summarize,v1.summarize))
        self.assertIs(v3._summarize_core.__code__,v1.summarize.__code__)
        self.assertEqual(v3._summarize_v2.__code__,v2.summarize.__code__)

    def test_matching_protocol_does_not_admit_drifted_raw_or_outcome(self):
        facade,_=v3.bookkeeping_driver(self.run)
        out=v3.old.plain.verify(self.run/'run_outcome.json')
        for field in ('arm','gate','banks'):
            changed=copy.deepcopy(self.m);changed[field]='drifted'
            with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(changed,out)
        changed=copy.deepcopy(out);changed['total']=1539
        with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(self.m,changed)

    def test_invocation_facade_does_not_escape_to_other_reports(self):
        facade,evidence=v3.bookkeeping_driver(self.run)
        original_driver=v3.driver
        report={'limitations':[]}
        with patch.object(v3,'bookkeeping_driver',return_value=(facade,evidence)),\
             patch.object(v3,'source_admission_evidence',return_value={}),\
             patch.object(v3,'_summarize_core',return_value=report) as core:
            result=v3.summarize(self.run,None,'accounting')
        core.assert_called_once_with(self.run,None,'accounting')
        self.assertEqual(result['summary_bookkeeping_hydration'],evidence)
        self.assertIs(v3.driver,original_driver)
        self.assertIs(v3.load_report.__globals__['summarize'],v3.summarize)
        self.assertIs(v3.compare.__globals__['load_report'],v3.load_report)
        self.assertIn('tools/summarize_boundary_variants_full1540_v3.py',v3.CODE)


class PolicyTests(unittest.TestCase):
    def test_unexecuted_recovery_remains_open_original_bookkeeping(self):
        # Actual original v1 postprocessing body, only raw/core replay mocked.
        scope=dict(v3.__dict__)
        report={'arm':v3.ARMS[0],'audit_errors':[],'slots_by_view':{v:[] for v in v3.VIEWS},
                'phase_sources':[],'current_qa_plus_retrieval_usage':{},'limitations':[]}
        facade=type('Facade',(),{'recovery_jobs':staticmethod(lambda m,o:['known-failure'])})()
        scope.update(driver=facade,require_closed=lambda r:None,validate_report=lambda *a:None,
            validity=lambda *a:{'all_slots_valid':True},failures=lambda *a:[],declared_failures=lambda *a:{},
            qa_estimator_assumptions=lambda *a:{},upstream_accounting=lambda *a:{})
        fn=v3._FunctionType(v1.summarize.__code__,scope,argdefs=v1.summarize.__defaults__)
        with patch.object(v3.locktool,'verify_lock',return_value={}),\
             patch.object(v3.full,'canonical',return_value={}),\
             patch.object(v3.old,'scoped',return_value=lambda *a:copy.deepcopy(report)),\
             patch.object(v3.old.plain,'read',return_value={'phase_kind':'firstpass'}),\
             patch.object(v3.old.plain,'verify',return_value={}),\
             patch.object(v3.old.plain,'pointer',return_value={}):
            result=fn('no-real-qa')
        self.assertEqual(result['recovery_status'],'pending_uniform_technical_recovery')
        self.assertFalse(result['uniform_recovery_policy_closed'])
        self.assertFalse(result['final_view_is_uniform_recovery_complete'])


if __name__=='__main__':unittest.main()
