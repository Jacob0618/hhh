"""Independent v2 scientific-lineage replay checks; no real QA/API."""
import copy,unittest
from pathlib import Path
from unittest.mock import patch
from tools import summarize_boundary_variants_full1540_v2 as s
from tools import summarize_boundary_variants_full1540_v1 as old

class IndependentSummaryV2(unittest.TestCase):
    def test_all_v1_audit_scoring_pair_bodies_rebound_privately(self):
        for name in s.REBOUND_FUNCTIONS:
            if name in ('summarize','compare'):continue
            self.assertIs(getattr(s,name).__code__,getattr(old,name).__code__,name)
            self.assertIs(getattr(s,name).__globals__,s.__dict__,name)
        self.assertIs(s._summarize_core.__code__,old.summarize.__code__)
        self.assertIs(s._compare_core.__code__,old.compare.__code__)
        self.assertIsNot(old.driver,s.driver)
    def replay(self,stored,fresh):
        with patch.object(s.old.plain,'verify',return_value=stored),patch.object(s.old.plain,'sha',return_value='sha'),patch.object(s.full,'checked',return_value=Path('f/run_manifest.json')),patch.object(s,'summarize',return_value=fresh):
            return s.load_report('sealed-summary.json')
    def fixture(self):
        return {'schema':s.SCHEMA,'model_calls_in_summary':0,'native_clients':0,'summary_code_sha256':{p:'sha' for p in s.CODE},
                'phase_sources':[{'manifest':{}}],'recovery_supplied':False,'pipeline_accounting':{},
                'source_admission_evidence':{'original_failed_sources':[{'conversation':3,'original_write_status':'failed','proof':{'sha256':'current'}}],
                    'export_is_metadata_admission_not_native_repair':True},'created_utc':'original'}
    def test_exact_current_admission_passes_replay(self):
        stored=self.fixture();fresh=copy.deepcopy(stored);fresh['created_utc']='new'
        self.assertEqual(self.replay(stored,fresh),fresh)
    def test_failed_proof_status_or_admission_erasure_rejected(self):
        for change in ('status','proof','drop','extra'):
            fresh=self.fixture();stored=copy.deepcopy(fresh)
            if change=='status':stored['source_admission_evidence']['original_failed_sources'][0]['original_write_status']='complete'
            if change=='proof':stored['source_admission_evidence']['original_failed_sources'][0]['proof']['sha256']='historical'
            if change=='drop':stored.pop('source_admission_evidence')
            if change=='extra':stored['source_admission_evidence']['unproved_success']=True
            with self.assertRaises(s.InvalidArtifact,msg=change):self.replay(stored,fresh)
if __name__=='__main__':unittest.main()
