"""Typed closure and old-reader compatibility. All mutation stays in temp."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import baseline_evidence_closure_v1 as typed
from tools import archive_reviewed_chain_typed_closure_v1 as entry
from tools.baselines_common_v1 import read,sha,save_new

ACTUAL=Path(__file__).resolve().parents[1]


class TypedClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def safe(self,p):
        p=Path(p).resolve()
        if not p.is_relative_to(self.root):raise ValueError('foreign fixture path')
        return p
    def collect(self,*paths,annotations=None):
        return typed.collect([(str(p),sha(p)) for p in paths],root=self.root,safe_path=self.safe,read=read,sha=sha,annotations=annotations)
    def known(self):
        for relative in (typed.ANNOTATION_REPORT,typed.ANNOTATION_SOURCE):
            p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ACTUAL/relative).read_bytes())
        return self.root/typed.ANNOTATION_REPORT
    def test_exact_scalar_retains_report_and_adds_explicit_source(self):
        report=self.known();annotations=[];records=self.collect(report,annotations=annotations)
        self.assertEqual(records,{str(report):typed.ANNOTATION_REPORT_SHA,str(self.root/typed.ANNOTATION_SOURCE):typed.ANNOTATION_SOURCE_SHA})
        self.assertEqual(len(annotations),1);self.assertEqual(annotations[0]['json_pointer'],'$.code_sha256')
    def test_unknown_scalar_same_bytes_foreign_path_and_changed_known_bytes_rejected(self):
        report=self.known();foreign=self.root/'foreign.json';foreign.write_bytes(report.read_bytes())
        with self.assertRaisesRegex(ValueError,'Unknown non-map'):self.collect(foreign)
        original=read(report);original['seconds']=0;report.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError,'Unknown non-map'):self.collect(report)
    def test_known_scalar_still_checks_explicit_source_bytes(self):
        report=self.known();(self.root/typed.ANNOTATION_SOURCE).write_text('altered')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.collect(report)
    def test_all_explicit_pairs_and_nested_maps_remain_transitive(self):
        target=self.root/'source.py';target.write_text('pass')
        nested=self.root/'nested.json';save_new(nested,{'tests':{'files_sha256':{str(target):sha(target)}}})
        report=self.root/'review.json';save_new(report,{'evidence_sha256':{str(nested):sha(nested)},'reference':{'path':str(target),'sha256':sha(target)}})
        self.assertEqual(set(self.collect(report)),{str(report),str(nested),str(target)})
    def test_bad_map_values_unknown_types_and_conflicting_hashes_rejected(self):
        target=self.root/'source.py';target.write_text('pass')
        for value in (None,[],True,12,'a'*64,{'source.py':{'sha256':sha(target)}},{'source.py':'not-sha'}):
            report=self.root/'review.json';report.write_text(json.dumps({'upstream_sha256':value}))
            with self.assertRaises(ValueError):self.collect(report)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            typed.collect([(str(target),'0'*64),(str(target),sha(target))],root=self.root,safe_path=self.safe,read=read,sha=sha)
    def test_complete_explicit_pairs_with_malformed_types_rejected(self):
        for value in ({'path':'source.py','sha256':17},{'raw_path':17,'raw_sha256':'a'*64}):
            p=self.root/'review.json';p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Explicit evidence references'):self.collect(p)
    def test_historical_classification_matches_only_all_four_exact_coordinates(self):
        root=ACTUAL;document=root/typed.HISTORICAL_CONFIG
        value=read(document)['baseline']['summary']
        self.assertTrue(typed._historical_annotation(value,document,typed.HISTORICAL_CONFIG_SHA,'$.baseline.summary',root))
        for v,d,h,p in ((value,document.with_name('other.json'),typed.HISTORICAL_CONFIG_SHA,'$.baseline.summary'),
                        (value,document,'0'*64,'$.baseline.summary'),(value,document,typed.HISTORICAL_CONFIG_SHA,'$.dataset.source'),
                        ({**value,'path':str(root/'other.json')},document,typed.HISTORICAL_CONFIG_SHA,'$.baseline.summary'),
                        ({**value,'sha256':'0'*64},document,typed.HISTORICAL_CONFIG_SHA,'$.baseline.summary')):
            self.assertFalse(typed._historical_annotation(v,d,h,p,root))
        p=self.root/'unknown.json';save_new(p,{'path':str(self.root/'missing.json'),'sha256':'0'*64})
        with self.assertRaises(FileNotFoundError):self.collect(p)


class ArchiveCompatibilityTests(unittest.TestCase):
    def test_new_sources_kept_old_code_table_and_reader_unchanged(self):
        from tests import test_amem_retrieval_lifecycle_v1 as oldtest
        factory=oldtest.LifecycleTests();factory.setUp()
        try:
            with patch.object(entry,'ROOT',factory.root):
                hashes={**entry.lifecycle.code_hashes(),**{p:sha(ACTUAL/p) for p in entry.EXTRA}}
                for p in hashes:
                    target=factory.root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ACTUAL/p).read_bytes())
                # The exact annotation was the previously untested real case.
                for p in (typed.ANNOTATION_REPORT,typed.ANNOTATION_SOURCE):
                    target=factory.root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ACTUAL/p).read_bytes())
                review=read(factory.toolreview);review.update(schema=entry.REVIEW_SCHEMA,
                    scope='archive_generation_typed_closure_compatibility_only',code_sha256=hashes,
                    evidence_sha256={str(factory.root/typed.ANNOTATION_REPORT):typed.ANNOTATION_REPORT_SHA})
                factory.toolreview.write_text(json.dumps(review))
                original_hook=entry.amem.engine.closure
                result=entry.archive('amem',factory.review,factory.toolreview)
                self.assertIs(entry.amem.engine.closure,original_hook)
                plan=read(result['plan']);self.assertEqual(set(plan['code_sha256']),set(entry.amem.TOOL_FILES))
                for p in entry.EXTRA:self.assertEqual(plan['keep_sha256'][str(factory.root/p)],hashes[p])
                entry.amem.verify_cleanup(result['plan'])
                with entry.amem.Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as resolver:
                    for p in entry.EXTRA:self.assertEqual(resolver.sha(factory.root/p),hashes[p])
                    self.assertEqual(resolver.sha(factory.root/typed.ANNOTATION_SOURCE),typed.ANNOTATION_SOURCE_SHA)
                # Every added implementation byte is checked by unchanged verifier.
                target=factory.root/entry.EXTRA[0];target.write_text('changed after archive')
                with self.assertRaisesRegex(ValueError,'Bound evidence changed'):entry.amem.verify_cleanup(result['plan'])
        finally:factory.doCleanups()
    def test_hook_restores_on_error_and_each_method_uses_same_typed_function(self):
        a=entry.amem.engine.closure;l=entry.lifecycle.engine.evidence_closure
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with entry.compatibility('amem'):
                self.assertIsNot(entry.amem.engine.closure,a);raise RuntimeError('fixture')
        self.assertIs(entry.amem.engine.closure,a)
        for method in entry.lifecycle.METHODS:
            with entry.compatibility(method):self.assertIs(entry.lifecycle.engine.evidence_closure,entry.evidence_closure)
            self.assertIs(entry.lifecycle.engine.evidence_closure,l)
    def test_review_missing_added_code_rejected_before_archive(self):
        with tempfile.TemporaryDirectory(dir=ACTUAL/'research') as tmp:
            p=Path(tmp)/'review.json';save_new(p,{'schema':entry.REVIEW_SCHEMA,'status':'passed','scope':'archive_generation_typed_closure_compatibility_only','code_sha256':entry.lifecycle.code_hashes()})
            with patch.object(entry.amem,'archive') as archive:
                with self.assertRaisesRegex(ValueError,'two added'):entry.archive('amem',None,p)
                archive.assert_not_called()


if __name__=='__main__':unittest.main()
