"""Versioned source admission counterexamples; no QA/native/API execution."""
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
from tools import summarize_boundary_variants_full1540_v1 as legacy
from tools import summarize_boundary_variants_full1540_v2 as reviewed


def gate_fixture():
    rows=[{'conversation':ci,'arm':reviewed.ARMS[0],
        'source_kind':'reviewed_export' if ci==3 else 'strict_complete',
        'original_write_status':'failed' if ci==3 else 'complete',
        'proof':{'path':'current-c3-proof','sha256':'current-source-specific'} if ci==3 else None}
        for ci in range(10)]
    return {'schema':'boundary-variants-full1540-qa-gate-v2','arm':reviewed.ARMS[0],
        'source_manifest':{'path':'original-writer','sha256':'original-failed-preserved'},
        'uniform_twenty_bank_review':{'path':'uniform20','sha256':'review'},
        'export_manifest':{'path':'new-metadata-export','sha256':'export'},'banks':rows}


class VersionedSummaryTests(unittest.TestCase):
    def evidence(self,gate):
        m={'schema':reviewed.driver.SCHEMA,'arm':reviewed.ARMS[0],'gate':{}}
        with patch.object(reviewed.old.plain,'read',return_value=m),\
             patch.object(reviewed.full,'checked',return_value=Path('gate.json')),\
             patch.object(reviewed.old.plain,'pointer',return_value={'path':'gate','sha256':'sealed'}),\
             patch.object(reviewed.driver.sources,'admit_gate',return_value=(gate,{i:None for i in range(10)},{})):
            return reviewed.source_admission_evidence('offline-firstpass')

    def test_old_summary_does_not_blindly_admit_v2(self):
        before=legacy.driver
        with patch.object(legacy.old.plain,'read',return_value={'schema':reviewed.driver.SCHEMA}):
            with self.assertRaises(legacy.InvalidArtifact):legacy.driver_for('v2')
            self.assertIs(reviewed.driver_for('v2'),reviewed.driver)
        self.assertIs(legacy.driver,before)
        self.assertIsNot(legacy.driver,reviewed.driver)

    def test_summary_body_and_all_three_views_are_reused(self):
        self.assertIs(reviewed._summarize_core.__code__,legacy.summarize.__code__)
        self.assertIs(reviewed._summarize_core.__globals__['driver'],reviewed.driver)
        self.assertIs(legacy.summarize.__globals__['driver'],legacy.driver)
        report={'arm':reviewed.ARMS[0],'limitations':[],
            'slots_by_view':{v:[{'scores':{'judge_correct':None if v=='strict_firstpass' else 0}}] for v in reviewed.VIEWS},
            'current_qa_plus_retrieval_usage':{'total_cost_usd':None},'planned_per_arm':1540}
        original=copy.deepcopy(report)
        with patch.object(reviewed,'source_admission_evidence',return_value={'original_failed_sources':[3]}),\
             patch.object(reviewed,'_summarize_core',return_value=report) as core:
            out=reviewed.summarize('first','recovery','existing-audit')
        core.assert_called_once_with('first','recovery','existing-audit')
        self.assertEqual(out['slots_by_view'],original['slots_by_view'])
        self.assertEqual(out['current_qa_plus_retrieval_usage'],original['current_qa_plus_retrieval_usage'])
        self.assertEqual(out['planned_per_arm'],1540)

    def test_failed_status_and_current_proof_remain_distinct_from_export(self):
        gate=gate_fixture();out=self.evidence(gate)
        self.assertEqual([r['conversation'] for r in out['original_failed_sources']],[3])
        self.assertEqual(out['reviewed_export_sources'][0]['original_write_status'],'failed')
        self.assertEqual(out['reviewed_export_sources'][0]['proof']['sha256'],'current-source-specific')
        self.assertTrue(out['original_strict_completion_not_implied'])
        self.assertTrue(out['export_is_metadata_admission_not_native_repair'])
        self.assertEqual(out['bound_review_and_export_fields']['export_manifest'],gate['export_manifest'])

    def test_erased_failed_or_unknown_source_kind_refused(self):
        for kind,status in [('reviewed_export','complete'),('strict_complete','failed'),('unreviewed','failed')]:
            gate=gate_fixture();gate['banks'][3].update(source_kind=kind,original_write_status=status)
            with self.assertRaisesRegex(reviewed.InvalidArtifact,'cannot be erased'):self.evidence(gate)

    def test_v1_or_unrecognized_qa_schema_refused_before_v2_source_admission(self):
        for schema in (legacy.driver.SCHEMA,'made-up-full1540'):
            with patch.object(reviewed.old.plain,'read',return_value={'schema':schema}),\
                 patch.object(reviewed.driver.sources,'admit_gate') as admit:
                with self.assertRaises(reviewed.InvalidArtifact):reviewed.source_admission_evidence('offline')
                admit.assert_not_called()


if __name__=='__main__':unittest.main()
