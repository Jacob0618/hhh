"""Independent v3 lineage negatives on real C firstpass shape; no model calls."""
import copy
import unittest
from types import FunctionType
from unittest.mock import patch

from tests import test_boundary_summary_v3_v1 as fixture
from tools import summarize_boundary_variants_full1540_v3 as v3
from tools import summarize_boundary_variants_full1540_v2 as v2
from tools import summarize_boundary_variants_full1540_v1 as v1
from tools import run_boundary_variants_qa_v2 as qa


class IndependentHydrationTests(unittest.TestCase):
    setUpClass = classmethod(fixture.HydrationTests.setUpClass.__func__)
    setUp = fixture.HydrationTests.setUp

    def reseal_source(self, mutate):
        path=self.run/'source_jobs.json'
        source=v3.old.plain.verify(path)
        source.pop('content_sha256')
        mutate(source)
        v3.old.plain.publish_json(path,{**source,'content_sha256':qa.digest(source)})
        sealpath=self.run/'firstpass_seal.json'
        seal=v3.old.plain.verify(sealpath)
        seal.pop('content_sha256')
        seal['source_jobs']=v3.old.plain.pointer(path)
        v3.old.plain.publish_json(sealpath,{**seal,'content_sha256':qa.digest(seal)})

    def test_raw_manifest_and_outcome_cannot_drift(self):
        facade,_=v3.bookkeeping_driver(self.run)
        out=v3.old.plain.verify(self.run/'run_outcome.json')
        with patch.object(qa,'recovery_jobs',return_value=[]) as predicate:
            for field,value in [('arm','boundary_neighbor'),('banks',{}),('gate',{}),
                                ('protocol_sha256','different')]:
                bad=copy.deepcopy(self.m);bad[field]=value
                with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(bad,out)
            badout=copy.deepcopy(out);badout['total']-=1
            with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(self.m,badout)
        predicate.assert_not_called()

    def test_resealed_question_reference_category_date_drift_rejected(self):
        sourcepath=self.run/'source_jobs.json';sealpath=self.run/'firstpass_seal.json'
        original_source=sourcepath.read_bytes();original_seal=sealpath.read_bytes()
        for field,value in [('question','different question'),('reference','different answer'),
                            ('category',999),('reference_date','1900-01-01')]:
            self.reseal_source(lambda s:s['jobs'][0].__setitem__(field,value))
            with self.assertRaises(v3.InvalidArtifact):v3.bookkeeping_driver(self.run)
            sourcepath.write_bytes(original_source);sealpath.write_bytes(original_seal)

    def test_resealed_source_schema_provenance_and_population_rejected(self):
        sourcepath=self.run/'source_jobs.json';sealpath=self.run/'firstpass_seal.json'
        original_source=sourcepath.read_bytes();original_seal=sealpath.read_bytes()
        changes=[lambda s:s.__setitem__('schema','wrong'),
                 lambda s:s.__setitem__('official_metric_provenance',{}),
                 lambda s:s.__setitem__('retrieval_outcome',{}),
                 lambda s:s['jobs'].__setitem__(0,copy.deepcopy(s['jobs'][1]))]
        for mutate in changes:
            self.reseal_source(mutate)
            with self.assertRaises(v3.InvalidArtifact):v3.bookkeeping_driver(self.run)
            sourcepath.write_bytes(original_source);sealpath.write_bytes(original_seal)

    def test_source_manifest_mismatch_is_not_silently_replaced(self):
        facade,_=v3.bookkeeping_driver(self.run)
        out=v3.old.plain.verify(self.run/'run_outcome.json')
        bad=copy.deepcopy(self.m)
        bad.update(jobs=self.s['jobs'],source_manifest={'path':'wrong','sha256':'wrong'})
        with patch.object(qa,'recovery_jobs',return_value=[]) as predicate:
            with self.assertRaises(v3.InvalidArtifact):facade.recovery_jobs(bad,out)
        predicate.assert_not_called()

    def test_per_invocation_facade_passthrough_both_arms_and_recovery_argument(self):
        def core(firstpass,recovery=None,global_accounting_audit=None):
            return {'schema':SCHEMA,'driver_is_original':driver is _qa,
                    'jobs':driver.recovery_jobs(_qa.manifest(firstpass),
                            old.plain.verify(Path(firstpass)/'run_outcome.json')),
                    'recovery':recovery,'accounting':global_accounting_audit,
                    'cost_usd':None,'reader_valid':1539,'judge_valid':1538,
                    'native_edge_missing':True,'slots_by_view':VIEWS}
        fn=FunctionType(core.__code__,vars(v3),argdefs=core.__defaults__)
        previous=(v3.driver,v2.driver,v1.driver,qa.recovery_jobs)
        with patch.object(v3,'_summarize_v2',fn),patch.object(qa,'recovery_jobs',return_value=['candidate']):
            for arm in qa.sources.ARMS:
                # Fixture changes only the declared arm and source job identity;
                # true production SDK/source/raw validation remains the old core.
                self.m['arm']=arm
                def mutate(s):
                    s['arm']=arm
                    for j in s['jobs']:
                        j['method']=arm;j['job_id']=arm+'/'+j['query_id']
                self.reseal_source(mutate)
                for recovery in (None,'fixed_recovery_v1'):
                    result=v3.summarize(self.run,recovery,'current_global_audit')
                    self.assertFalse(result['driver_is_original'])
                    self.assertEqual(result['jobs'],['candidate'])
                    self.assertEqual(result['recovery'],recovery)
                    self.assertEqual(result['accounting'],'current_global_audit')
                    self.assertIsNone(result['cost_usd'])
                    self.assertEqual((result['reader_valid'],result['judge_valid']),(1539,1538))
                    self.assertTrue(result['native_edge_missing'])
                    self.assertEqual(result['slots_by_view'],v3.VIEWS)
                    self.assertFalse(result['summary_bookkeeping_hydration']['recovery_executed'])
        self.assertEqual(previous,(v3.driver,v2.driver,v1.driver,qa.recovery_jobs))

    def test_compare_and_load_report_keep_original_bodies_with_v3_replay_binding(self):
        self.assertIs(v3.compare.__code__,v2.compare.__code__)
        self.assertIs(v3._compare_core.__code__,v1.compare.__code__)
        self.assertIs(v3.load_report.__code__,v1.load_report.__code__)
        self.assertIs(v3.load_report.__globals__['summarize'],v3.summarize)
        self.assertIs(v3._compare_core.__globals__['load_report'],v3.load_report)
        self.assertEqual(v3.COMPARISON_SCHEMA,'boundary-variants-full1540-comparison-v3')
        self.assertIn('tools/summarize_boundary_variants_full1540_v3.py',v3.CODE)


if __name__=='__main__':unittest.main()
