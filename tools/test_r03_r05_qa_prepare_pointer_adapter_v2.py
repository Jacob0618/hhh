"""Offline semantic manifest repair tests; no API/controller construction."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import r03_r05_qa_prepare_pointer_adapter_v2 as fix

class RepairTests(unittest.TestCase):
    def fixture(self,root):
        old=root/'runs/old';old.mkdir(parents=True);src=root/'source.json'
        fix.frozen.save_new(src,{'schema':fix.frozen.SOURCE_SCHEMA,'jobs':[],'official_metric_provenance':{}})
        fix.frozen.save_new(old/'prior_qa_reuse_proof.json',{'proof':'exact'})
        m={'schema':fix.frozen.SCHEMA,'policy':fix.frozen.POLICY,'jobs':[],
           'official_metric_provenance':{},'source_manifest':fix.frozen.pointer(src),
           'phase_kind':'firstpass','phase_id':'r03_r05_refinement_qa_firstpass_v1',
           'prior_qa_reuse_proof':None,'created_utc':'old','parent':None,'reuse_plans':{},
           'cache_roots':[],'code_sha256':{},'model_calls_during_prepare':0}
        fix.frozen.publish_json(old/'run_manifest.json',{**m,'protocol_sha256':fix.frozen.digest(m)})
        return old,m
    def test_changes_only_pointer_time_digest_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old,m=self.fixture(root);before={p.name:p.read_bytes() for p in old.iterdir()}
            source=fix.frozen.verify(root/'source.json')
            with patch.object(fix.frozen,'ROOT',root),patch.object(fix.frozen,'CODE',()), \
                 patch.object(fix.frozen,'source_validate',side_effect=AssertionError('full source runtime admission deferred')), \
                 patch.object(fix.frozen,'verified_official_metric',return_value=(None,{})), \
                 patch.object(fix.frozen,'check_full_cohort'),patch.object(fix.frozen,'_plans',return_value={}), \
                 patch.object(fix.frozen.reuse,'revalidate') as revalidate, \
                 patch.object(fix.frozen,'_ordinary_manifest',side_effect=lambda run,**kw:fix.frozen.read(run/'run_manifest.json')):
                new=root/'runs/new';report=fix.prepare(new,old,technical_bindings={'adapter':'sha'})
                self.assertEqual(report['full_raw_admission'],'pending_original_qa_run_manifest')
                result=fix.frozen.read(new/'run_manifest.json')
                self.assertEqual({k for k in result if result[k] != {**m,'protocol_sha256':fix.frozen.digest(m)}[k]},
                                 {'prior_qa_reuse_proof','created_utc','protocol_sha256'})
                self.assertEqual(result['prior_qa_reuse_proof'],fix.frozen.pointer(old/'prior_qa_reuse_proof.json'))
                self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})
                revalidate.assert_not_called()
                self.assertFalse((new/'prior_qa_reuse_proof.json').exists())
    def test_existing_namespace_and_unknown_old_files_block(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old,m=self.fixture(root)
            with patch.object(fix.frozen,'ROOT',root):
                with self.assertRaises(FileExistsError):fix.prepare(old,old,technical_bindings={})
                (old/'unexpected.intent.json').write_text('{}')
                with self.assertRaises(ValueError):fix.prepare(root/'runs/new',old,technical_bindings={})
                self.assertFalse((root/'runs/new').exists())
    def test_original_run_manifest_rejection_prevents_intent_and_api(self):
        from unittest.mock import Mock
        from tools import run_r03_r05_refinement_qa_v1 as qa
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'qa';run.mkdir();api=Mock()
            # Patch only the admission dependency in a temporary test context;
            # execute the unmodified original runtime bytecode.
            scope=qa.run.__globals__
            with patch.dict(scope,manifest=Mock(side_effect=ValueError('raw proof rejected')),
                            durable_intent=Mock(),PlainTextEvaluationAPI=Mock(),
                            verified_official_metric=Mock()):
                with self.assertRaises(ValueError):qa.run(run,api_factory=api)
                scope['durable_intent'].assert_not_called()
                scope['PlainTextEvaluationAPI'].assert_not_called()
                scope['verified_official_metric'].assert_not_called()
                api.assert_not_called()
                self.assertEqual(list(run.iterdir()),[])

    def test_runtime_summary_and_scientific_functions_are_original(self):
        from tools import resume_r03_r05_refinement_qa_after_prepare_fix_v2 as entry
        from tools import run_r03_r05_refinement_qa_v1 as qa
        from tools import summarize_r03_r05_refinement_v1 as summary
        self.assertIs(fix.frozen,qa);self.assertIs(summary.runner,qa)
        self.assertIn('qa=fix.frozen',inspect.getsource(entry.run))
        self.assertIs(qa.prepare_recovery.__globals__['_new_manifest'],qa._new_manifest)
        text=inspect.getsource(qa._new_manifest)
        self.assertIn("proof_ref = parent_manifest['prior_qa_reuse_proof']",text)
        self.assertIn('if proof_ref is None:',text)

if __name__=='__main__':unittest.main()
