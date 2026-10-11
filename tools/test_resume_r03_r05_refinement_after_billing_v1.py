"""Offline mocks; no provider instantiated and no controller launched."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import resume_r03_r05_refinement_after_billing_v1 as w

class ContinuationTests(unittest.TestCase):
    def test_all_billing_failure_blocks(self):
        with self.assertRaises(ValueError):w.remaining_gate({'x':{'status':'failed'}})
    def test_unknown_blocks(self):
        with self.assertRaises(ValueError):w.remaining_gate({'x':{'status':'complete'},'y':{'status':'unknown'}})
    def test_knownfail_with_success_is_legal(self):
        v=w.remaining_gate({'x':{'status':'complete'},'y':{'status':'failed'}})
        self.assertEqual(v['effective_counts'],{'complete':1,'failed':1})
    def test_fresh_namespace(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w,'BASE',Path(d)):
            with self.assertRaises(ValueError):w.admission('unused',d)
    def test_recovery_existing_blocks_before_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';(run/'phases/rank/recovery_v1').mkdir(parents=True)
            with patch.object(w,'BASE',root),patch.object(w,'RUN',run),patch.object(w,'QA',root/'qa'):
                with self.assertRaises(FileExistsError):w.admission('unused',root/'audit')
    def test_failed_recovery_never_seals_or_runs_qa(self):
        from tools import run_r03_r05_refinement_qa_v1 as qa
        with tempfile.TemporaryDirectory() as d, patch.object(w,'admission',return_value={}), patch.object(w,'sha',return_value='mock-hash'), \
             patch.object(w.score,'run_phase',return_value={'status':'terminal'}) as dispatch, \
             patch.object(w.prep,'phase_results',return_value={'x':{'status':'failed'}}), \
             patch.object(w.prep,'seal') as seal, patch.object(qa,'prepare') as prepare:
            with self.assertRaises(ValueError):w.run('review.json',Path(d)/'audit')
            dispatch.assert_called_once_with(w.RUN,'rank',recovery=True)
            seal.assert_not_called();prepare.assert_not_called()
            import json
            self.assertEqual(json.loads((Path(d)/'audit/outcome.json').read_text())['status'],'needs_review')

if __name__=='__main__':unittest.main()
