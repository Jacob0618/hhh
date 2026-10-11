"""Independent real identity gates and thin-body checks; no production output/API."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from tools import review_boundary_variants_banks_v2 as r
from tools import run_boundary_variants_qa_v2 as q
from tools import run_boundary_variants_qa_v1 as q1
from tools import boundary_variants_qa_sources_v2 as s
from jev_mem.local_bm25_context_v2 import forbid_network

class Independent(unittest.TestCase):
    def setUp(self):r.old.HASHES.clear()
    def target(self,arm):
        run=r.ROOT/'runs/boundary_variants_v1_20261003'/arm/'writer';f=run/'conv_3'/arm
        return {'conversation':3,'arm':arm,'run':run,'folder':f,'manifest':r.old.read(run/'run_manifest.json'),'state':r.old.read(f/'write_status.json')}
    def proof(self,arm):return r.ROOT/'research/planned_execution_2026-10-03/checks/20261003T0658'/('c1_conv3_failure_review.json' if arm=='c1_multiscale' else 'c2_conv3_failure_review.json')
    def test_real_both_current_proofs_bind_without_api_or_export(self):
        with forbid_network() as guards:
            for arm in r.ARMS:
                p,ref=r._failure_binding(self.proof(arm),self.target(arm));i=r._proof_inspection(p)
                self.assertFalse(i['strict_passed']);self.assertEqual(i['facts'],p['facts']);self.assertFalse((self.target(arm)['folder']/'bank.json').exists())
            self.assertFalse(any(g.call_count for g in guards))
    def test_c1_proof_does_not_transfer_to_c2(self):
        with self.assertRaisesRegex(ValueError,'pinned'):r._failure_binding(self.proof('c1_multiscale'),self.target('c2_neighbour'))
    def test_same_failure_message_different_protocol_rejected(self):
        t=self.target('c2_neighbour');t['manifest']=dict(t['manifest'],protocol_sha256='unreviewed')
        with self.assertRaisesRegex(ValueError,'pinned'):r._failure_binding(self.proof('c2_neighbour'),t)
    def test_missing_prompt_or_false_source_check_rejected(self):
        p=r.verify(self.proof('c2_neighbour'));p=copy.deepcopy(p);p['strict_body_inspection']['writer_prompt_reconstructed_and_raw_verified']=False
        with self.assertRaisesRegex(ValueError,'strict-body'):r._proof_inspection(p)
    def test_whole20_population_rejects19_and_duplicate(self):
        rows=[{'conversation':ci,'arm':a} for ci in range(10) for a in r.ARMS]
        r._population(rows)
        with self.assertRaises(ValueError):r._population(rows[:-1])
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):r._population(rows)
    def test_v2_retrieval_reader_recovery_exact_v1_bodies(self):
        for name in ('retrieve','qa','recover','seal_firstpass','validate_source'):
            if hasattr(q,name):self.assertIs(getattr(q,name).__code__,getattr(q1,name).__code__,name)
        self.assertEqual(s.CONFIG,s.legacy.CONFIG);self.assertEqual(s.COHORT,s.legacy.COHORT)
        self.assertIs(q.sources,s);self.assertEqual(q.SCHEMA,'boundary-variants-plain-qa-full1540-v2')

if __name__=='__main__':unittest.main()
