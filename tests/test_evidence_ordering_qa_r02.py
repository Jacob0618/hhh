"""Offline R02 adapter: run frozen behavioral regression in isolated globals."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import FunctionType
from tests import test_evidence_exchange_qa_r01 as oldtests
from tools import run_evidence_ordering_qa_r02 as runner
from tools import summarize_evidence_ordering_r02 as summary
from tools import run_evidence_exchange_qa_r01 as prior

# Reuse actual same evaluator/signature/stage regression assertions rather than
# duplicating or weakening them. Only test globals change; production R01 does not.
scope={**vars(oldtests),'runner':runner,'summary':summary}
methods={name:FunctionType(f.__code__,scope,name,f.__defaults__,f.__closure__)
         for name,f in vars(oldtests.AdapterTests).items() if isinstance(f,FunctionType)}
R02InheritedBehaviorTests=type('R02InheritedBehaviorTests',(unittest.TestCase,),methods)

class R02IsolationAndProofTests(unittest.TestCase):
    def test_implementation_and_audit_scopes_are_isolated(self):
        self.assertIsNot(runner._impl,prior._impl)
        self.assertIs(runner.identity.__globals__,runner._impl)
        self.assertIs(runner.one.__globals__,runner._helpers)
        self.assertIs(runner.manifest.__globals__['source_validate'],runner.source_validate)
        self.assertIs(summary.audit_phase.__globals__['runner'],runner)
        self.assertIs(summary.audit_stage.__globals__['runner'],runner)
        self.assertEqual(prior._impl['SCHEMA'],prior.SCHEMA)
        self.assertNotEqual(prior.SCHEMA,runner.SCHEMA)

    def test_missing_proof_binding_cannot_fallback_to_summary(self):
        with self.assertRaisesRegex(ValueError,'Explicit prior'):
            runner.build_reuse_proofs({'summary_score':1153})

    def test_proof_sha_change_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proof.json'; p.write_text('{}')
            source={'frozen_baseline_reuse_proof':{'path':str(p),'sha256':'wrong'},
                    'reuse_proof_qa_manifest':{'path':'unused','sha256':'unused'}}
            with self.assertRaises(ValueError):runner.build_reuse_proofs(source)

    def test_order_change_requires_reader_but_same_answer_still_reuses_old_no(self):
        case=R02InheritedBehaviorTests('test_actual_reader_and_judge_capture_guard_and_parser')
        case.setUp()
        try:
            j=oldtests.job('jev'); j['selected_ids'].reverse(); j['selected_facts'].reverse()
            case.plan(j)
            self.assertIsNone(case.m['reuse_plans'][j['job_id']]['reader'])
            api=oldtests.MockAPI(case.run,j,label='[[yes]]')
            runner.one(case.run,j,case.m,api,oldtests.official)
            self.assertEqual(api.calls,['reader'])
            row=runner.verify(runner.stage_files(case.run,j)['judge'])
            self.assertEqual(row['model_calls'],0); self.assertFalse(row['result']['correct'])
        finally:case.doCleanups()

if __name__=='__main__':unittest.main()
