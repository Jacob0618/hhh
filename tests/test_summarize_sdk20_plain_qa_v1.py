"""No-network raw replay and full-denominator checks for the new summary."""
import copy
import unittest
from unittest.mock import patch

from tools import summarize_sdk20_plain_qa_v1 as s
from tests import test_sdk20_plain_qa_v1 as fixture


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixture.PlainTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)

    def phase(self,outputs):
        job=fixture.job();run,m=self.fx.phase([job]);self.fx.execute(run,outputs)
        source=fixture.d.verify(m['source_manifest']['path'])
        return run,m,job,source

    def test_reader_and_strict_label_raw_replay(self):
        run,m,j,source=self.phase(['Paris','[[no]]'])
        result=s.audit_phase(run,source)
        self.assertEqual(result['errors'],[])
        row=s.record_rows([j],result)[0]
        self.assertEqual(row['scores']['official_locomo_f1'],1.)
        self.assertEqual(row['scores']['judge_correct'],0.)
        self.assertEqual(row['reader_answer_whitespace_words'],1)

    def test_invalid_label_has_F1_and_compat_but_not_strict_judge(self):
        run,m,j,source=self.phase(['Paris','Explanation\n[[yes]]'])
        result=s.audit_phase(run,source);self.assertEqual(result['errors'],[])
        a=s.record_rows([j],result)[0];b=s.record_rows([j],result,compatible=True)[0]
        self.assertEqual(a['scores']['official_locomo_f1'],1.)
        self.assertIsNone(a['scores']['judge_correct']);self.assertEqual(b['scores']['judge_correct'],1.)

    def test_conflicting_compat_stays_missing(self):
        run,m,j,source=self.phase(['Paris','quoted [[no]]\n[[yes]]'])
        result=s.audit_phase(run,source);self.assertEqual(result['errors'],[])
        self.assertIsNone(s.record_rows([j],result,compatible=True)[0]['scores']['judge_correct'])

    def test_recovery_keeps_first_valid_reader_and_prefers_retry_strict_no(self):
        run,m,j,source=self.phase(['Paris','Reason\n[[yes]]']);first=s.audit_phase(run,source)
        fixture.d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1')
        self.fx.execute(retry,['[[no]]']);second=s.audit_phase(retry,source,parent=first)
        self.assertEqual(second['errors'],[])
        row=s.record_rows([j],first,second,final=True)[0]
        self.assertEqual(row['scores']['official_locomo_f1'],1.);self.assertEqual(row['scores']['judge_correct'],0.)

    def test_new_reader_recovery_has_retry_resolution_and_independent_metrics(self):
        invalid={'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'unfinished'}]}}]}
        run,m,j,source=self.phase([invalid]);first=s.audit_phase(run,source)
        fixture.d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1')
        self.fx.execute(retry,['Paris','Reason\n[[yes]]']);second=s.audit_phase(retry,source,parent=first)
        self.assertEqual(first['errors'],[]);self.assertEqual(second['errors'],[])
        self.assertIsNone(s.record_rows([j],first)[0]['scores']['official_locomo_f1'])
        final=s.record_rows([j],first,second,final=True)[0]
        self.assertEqual(final['scores']['official_locomo_f1'],1.);self.assertEqual(final['scores']['judge_correct'],1.)

    def test_api_initialization_failure_no_dispatch_retains_planned_slot(self):
        j=fixture.job();run,m=self.fx.phase([j]);source=fixture.d.verify(m['source_manifest']['path'])
        fixture.d.run(run,api_factory=lambda *args:(_ for _ in ()).throw(ValueError('mock initialization')))
        result=s.audit_phase(run,source);self.assertEqual(result['errors'],[])
        row=s.record_rows([j],result)[0]
        self.assertEqual(row['reader_status'],'not_run');self.assertIsNone(row['scores']['judge_correct'])

    def test_all_missing_3080_preserves_independent_denominators(self):
        jobs=[];records={}
        for arm in s.ARMS:
            for index in range(1540):
                j=fixture.job('c0_q'+str(index),arm);jobs.append(j)
                records[j['job_id']]={'reader':None,'judge':None,'compat':None,'qa':None,'errors':[]}
        values=s.record_rows(jobs,{'records':records});out=s.aggregate(values)
        for a in s.ARMS:
            self.assertEqual(out['arms'][a]['planned_denominator'],1540)
            for metric in s.METRICS:self.assertEqual(out['arms'][a]['metrics'][metric]['valid_denominator'],0)

    def test_wrong_resolved_model_or_payload_is_not_scored(self):
        run,m,j,source=self.phase(['Paris','[[yes]]'])
        path=run/'conv_0/api/api_calls.jsonl';original=path.read_text()
        path.write_text(original.replace('gemini-2.5-flash','gemini-WRONG'),encoding='utf-8')
        with self.assertRaises(s.InvalidArtifact):s.audit_phase(run,source)

    def test_recovery_api_initialization_keeps_inherited_F1(self):
        run,m,j,source=self.phase(['Paris','Reason\n[[yes]]']);first=s.audit_phase(run,source)
        fixture.d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1')
        fixture.d.run(retry,api_factory=lambda *args:(_ for _ in ()).throw(ValueError('offline mock')))
        second=s.audit_phase(retry,source,parent=first);self.assertEqual(second['errors'],[])
        row=s.record_rows([j],first,second,final=True)[0]
        self.assertEqual(row['scores']['official_locomo_f1'],1.)
        self.assertEqual(row['scores']['judge_correct'],1.)

    def test_complete_summary_shell_with_mock_cardinality(self):
        run,m,j,source=self.phase(['Paris','[[no]]'])
        with patch.object(s.sources,'validate',return_value=source),patch.object(s,'aggregate',side_effect=lambda rows:{'mock_rows':len(rows)}):
            result=s.summarize(run)
        self.assertEqual(result['audit_status'],'no_invalid_artifacts')
        self.assertEqual(result['planned_total'],3080)
        self.assertEqual(result['views']['strict_firstpass']['mock_rows'],1)
        self.assertEqual(result['answer_length']['strict_firstpass']['baseline_sdk20']['valid'],1)
        self.assertEqual(result['answer_length']['strict_firstpass']['fact_types_sdk20']['valid'],0)
        self.assertEqual(result['model_calls_in_summary'],0)


if __name__=='__main__':unittest.main()
