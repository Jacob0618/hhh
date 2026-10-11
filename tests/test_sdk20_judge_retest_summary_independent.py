"""Independent pure metric/label-priority checks; no requests or source writes."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import summarize_sdk20_judge_retest_v1 as s


def strict(value=None):
    return {'status':'failed','result':None} if value is None else {
        'status':'complete','result':{'correct':value,'label':'[[yes]]' if value else '[[no]]'}}


def compat(value=None):
    return {'status':'rejected','interpretation':None} if value is None else {
        'status':'compat_valid','interpretation':{'correct':value,'label':'[[yes]]' if value else '[[no]]'}}


class IndependentSummaryTests(unittest.TestCase):
    def test_first_strict_false_is_final_even_if_later_yes_exists(self):
        out=s.runner.resolve_pair(strict(False),compat(True),strict(True),compat(True))
        self.assertEqual(out,{'source':'firstpass_strict','correct':False,'label':'[[no]]'})

    def test_retry_strict_precedes_first_compat_without_correctness_selection(self):
        out=s.runner.resolve_pair(strict(),compat(True),strict(False),compat(True))
        self.assertEqual(out['source'],'retry_strict');self.assertIs(out['correct'],False)

    def test_both_strict_failed_earliest_compat_false_is_final(self):
        out=s.runner.resolve_pair(strict(),compat(False),strict(),compat(True))
        self.assertEqual(out['source'],'firstpass_eligible_compat');self.assertIs(out['correct'],False)
        out=s.runner.resolve_pair(strict(),compat(),strict(),compat(False))
        self.assertEqual(out['source'],'retry_eligible_compat');self.assertIs(out['correct'],False)

    def test_unknown_and_conflict_remain_null(self):
        self.assertEqual(s.runner.resolve_pair(None,compat(),strict(),None),{'source':None,'correct':None,'label':None})

    def fixture(self):
        values=[]
        for arm,valid in zip(s.ARMS,(1525,1517)):
            for i in range(1540):
                ready=i<valid
                values.append({'job_id':arm+f'/c{i//154}_q{i%154}','query_id':f'c{i//154}_q{i%154}',
                    'method':arm,'conversation':i//154,'category':i%4+1,
                    'status':'ready' if ready else 'blocked_original_reader',
                    'reader_status':'complete' if ready else 'failed',
                    'F1':{'official_locomo_f1':.5 if ready else None,'custom_token_f1':.4 if ready else None,'custom_exact_match':0. if ready else None}})
        return values

    def test_fixed1540_independent_f1_judge_denominators_and_pairs(self):
        jobs=self.fixture();labels={j['job_id']:False for j in jobs if j['status']=='ready'}
        labels[jobs[0]['job_id']]=None
        view=s.aggregate(s.rows(jobs,labels))
        a,b=(view['arms'][arm]['metrics'] for arm in s.ARMS)
        self.assertEqual((a['official_locomo_f1']['valid_denominator'],b['official_locomo_f1']['valid_denominator']),(1525,1517))
        self.assertEqual((a['judge_correct']['valid_denominator'],b['judge_correct']['valid_denominator']),(1524,1517))
        self.assertEqual(a['judge_correct']['planned_denominator'],1540)
        self.assertEqual(a['judge_correct']['complete_only_mean'],0.)
        self.assertIsNone(a['judge_correct']['full_cohort_observed_mean'])
        self.assertEqual(view['paired_merged_minus_unmerged']['judge_correct']['common_success_count'],1516)
        self.assertEqual(view['paired_merged_minus_unmerged']['official_locomo_f1']['common_success_count'],1517)

    def test_supplement_f1_only_changes_explicit_separate_view(self):
        jobs=self.fixture();before=copy.deepcopy(jobs)
        labels={j['job_id']:True if j['status']=='ready' else None for j in jobs}
        first=s.aggregate(s.rows(jobs,labels));fixed=s.aggregate(s.rows(jobs,{k:False if v is not None else None for k,v in labels.items()}))
        for arm in s.ARMS:
            for metric in ('official_locomo_f1','custom_token_f1'):
                self.assertEqual(first['arms'][arm]['metrics'][metric],fixed['arms'][arm]['metrics'][metric])
        overrides={j['job_id']:{**j,'F1':{'official_locomo_f1':.75,'custom_token_f1':.6,'custom_exact_match':0.}} for j in jobs if j['status']=='blocked_original_reader'}
        full=s.aggregate(s.rows(jobs,{j['job_id']:True for j in jobs},overrides))
        self.assertTrue(all(full['arms'][arm]['metrics']['official_locomo_f1']['valid_denominator']==1540 for arm in s.ARMS))
        self.assertEqual(jobs,before)

    def test_boolean_flips_do_not_drop_false_or_fill_null(self):
        out=s.confusion({'a':True,'b':False,'c':False,'d':None},{'a':False,'b':True,'c':False,'d':False})
        self.assertEqual(out,{'yes_to_no':1,'no_to_yes':1,'no_to_no':1,'missing':1})

    def test_known_failed_blocked_and_actually_missing_are_separate(self):
        jobs=self.fixture();blocked=next(j for j in jobs if j['status']=='blocked_original_reader')
        selected=[jobs[0],jobs[1],blocked]
        out=s.rows(selected,{},status_by_job={jobs[1]['job_id']:'missing'})
        self.assertEqual([r['judge_status'] for r in out],['failed','missing','blocked_original_reader'])
        self.assertEqual([r['reader_status'] for r in out],['complete','complete','failed'])

    def test_wrong_recovery_parent_or_followup_without_parent_rejected(self):
        source={'path':'mock-source','sha256':'s'}
        first={'run':r'D:\mock-first','manifest':{'phase_kind':'firstpass','source_manifest':source}}
        wrong={'run':r'D:\mock-retry','manifest':{'phase_kind':'postbatch_recovery','source_manifest':source,
            'parent':{'run_dir':r'D:\another-first','manifest':{'path':'wrong','sha256':'x'},'outcome':{'path':'wrong','sha256':'x'}}}}
        with patch.object(s,'phase',side_effect=[first,wrong]),patch.object(s.sources,'validate',return_value={'jobs':[]}), \
             patch.object(s.runner,'pointer',side_effect=lambda p:{'path':str(p),'sha256':'mock'}):
            with self.assertRaisesRegex(ValueError,'another phase/source'):s.summarize('first','retry')
        with patch.object(s,'phase',side_effect=[first,wrong]),patch.object(s.sources,'validate',return_value={'jobs':[]}):
            with self.assertRaisesRegex(ValueError,'requires its recovery parent'):s.summarize('first',followup='orphan')


if __name__=='__main__':unittest.main()
