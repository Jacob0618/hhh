"""Fixed400 identity, firstpass and independent metric pairing; fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import compare_regroup_refinement400_v1 as m


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in ('requests.Session.post','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(name,side_effect=AssertionError('No real API/credentials'));p.start();self.addCleanup(p.stop)
        self.expected={f'c{i%10}_q{i}':{'id':f'c{i%10}_q{i}','conversation':i%10,'category':i%4+1,
            'question':f'Q{i}?','reference':f'A{i}'} for i in range(400)}
        self.dates={str(i):'2023-01-01' for i in range(10)}

    def rows(self,arm,score=.5):
        return [{'query_id':q['id'],'conversation':q['conversation'],'category':q['category'],'method':arm,
                 'selection_status':'complete','reader_status':'complete','judge_status':'complete','qa_status':'complete',
                 'scores':{'official_locomo_f1':score,'custom_token_f1':score,'judge_correct':1.},'errors':[]}
                for q in self.expected.values()]

    def write(self,path,value,sealed=True):
        path.parent.mkdir(parents=True,exist_ok=True)
        if sealed:m.save_new(path,value)
        else:path.write_text(json.dumps(value),encoding='utf-8')
        return m.summary.plain.pointer(path)

    def replace(self,path,value):
        value={k:v for k,v in value.items() if k!='content_sha256'}
        path.write_text(json.dumps({**value,'content_sha256':m.digest(value)}),encoding='utf-8')

    def fixture(self,arm,score=.5):
        run=self.root/arm
        config=self.write(self.root/'config.json',{'plain':True}) if not (self.root/'config.json').exists() else m.summary.plain.pointer(self.root/'config.json')
        fields={'plain_text_config':config,'evaluation_protocol':'plain-fixed','dataset_sha256':'same-data',
                'official_metric_provenance':{'sha':'same-official'},'reference_dates':self.dates}
        manifest={**fields,'arm':arm,'phase_kind':'firstpass'};manifest['protocol_sha256']=m.digest(manifest)
        manifest_ref=self.write(run/'run_manifest.json',manifest,False)
        outcome=self.write(run/'run_outcome.json',{'terminal':True})
        jobs=[{'job_id':arm+'/'+q['id'],'query_id':q['id'],'conversation':q['conversation'],'category':q['category'],
               'question':q['question'],'reference':q['reference'],'reference_date':self.dates[str(q['conversation'])],
               'method':arm} for q in self.expected.values()]
        source=self.write(run/'source_jobs.json',{'arm':arm,'protocol_sha256':manifest['protocol_sha256'],
             'official_metric_provenance':fields['official_metric_provenance'],'jobs':jobs})
        rows=self.rows(arm,score);aggregated=m.slots(rows,arm,self.expected)
        names=('strict_firstpass','compatible_firstpass','final')
        report={'schema':'regroup-refinement-single-arm400-qa-summary-v1','arm':arm,'planned_per_arm':400,'planned_total':400,
            'model_calls_in_summary':0,'native_clients':0,'paired_comparison':None,'audit_status':'no_invalid_artifacts','audit_errors':[],
            'summary_code_sha256':{p:m.summary.plain.sha(m.ROOT/p) for p in m.summary.CODE},
            'phase_sources':[{'manifest':manifest_ref,'outcome':outcome,'ledgers':[]}],'source_manifest':source,
            'recovery_supplied':False,'slots_by_view':{name:copy.deepcopy(rows) for name in names},
            'views':{name:{'arms':{arm:copy.deepcopy(aggregated)},'paired':None} for name in names},
            'qa_usage':{},'retrieval_usage':{},'writer_accounting':{}}
        path=self.root/(arm+'_summary.json');self.write(path,report)
        return path,report,{'manifest':fields}

    def test_missing_swapped_category_or_duplicate400_rejected(self):
        rows=self.rows(m.ARMS[0])
        for bad in (rows[:-1],rows[:-1]+[rows[0]],[dict(rows[0],category=9)]+rows[1:]):
            with self.assertRaises(ValueError):m.slots(bad,m.ARMS[0],self.expected)

    def test_pair_uses_metric_specific_valid_sets_and_full400(self):
        left=self.rows(m.BASELINE,.5);right=self.rows(m.ARMS[0],.51)
        right[0]['scores']['judge_correct']=None
        p=m.pair(left,right,bootstrap_samples=10)
        self.assertEqual(p['official_locomo_f1']['common_success_count'],400)
        self.assertAlmostEqual(p['official_locomo_f1']['metrics']['official_locomo_f1']['other_minus_baseline'],.01)
        self.assertEqual(p['judge_correct']['common_success_count'],399)
        self.assertEqual(p['judge_correct']['cohort_denominator'],400)
        self.assertEqual(p['judge_correct']['independent_valid_denominators'],{'baseline':400,'other':399,'planned_each':400})
        self.assertTrue(p['judge_correct']['missing_as_zero_delta_sensitivity']['is_imputation_not_observed_score'])

    def test_reference_identity_is_pinned_before_scores(self):
        p=self.root/'invented-reference.json';self.write(p,{'schema':'refinement-fixed400-reused-reference-v1'})
        with self.assertRaisesRegex(m.InvalidArtifact,'exact frozen'):m.load_reference(p,self.expected,self.dates)

    def test_summary_and_question_identity_verified(self):
        p,report,ref=self.fixture(m.ARMS[0])
        loaded=m.load_arm(p,m.ARMS[0],self.expected,self.dates,ref)
        self.assertEqual(len(loaded['views'][m.VIEW]),400)
        source=Path(report['source_manifest']['path']);j=m.summary.plain.verify(source);j['jobs'][0]['reference']='altered'
        self.replace(source,j);report['source_manifest']=m.summary.plain.pointer(source);self.replace(p,report)
        with self.assertRaisesRegex(m.InvalidArtifact,'Question, answer'):m.load_arm(p,m.ARMS[0],self.expected,self.dates,ref)

    def test_evaluation_protocol_and_recovered_only_slot_tamper_rejected(self):
        p,report,ref=self.fixture(m.ARMS[0])
        bad=copy.deepcopy(ref);bad['manifest']['evaluation_protocol']='other'
        with self.assertRaisesRegex(m.InvalidArtifact,'evaluation protocol'):m.load_arm(p,m.ARMS[0],self.expected,self.dates,bad)
        report['slots_by_view']['final'][0]['scores']['judge_correct']=0
        report['views']['final']['arms'][m.ARMS[0]]=m.slots(report['slots_by_view']['final'],m.ARMS[0],self.expected)
        self.replace(p,report)
        with self.assertRaisesRegex(m.InvalidArtifact,'Final without'):m.load_arm(p,m.ARMS[0],self.expected,self.dates,ref)

    def test_two_primary_differences_never_select_final_recovery_view(self):
        pa,a,ref=self.fixture(m.ARMS[0],.51);pb,b,_=self.fixture(m.ARMS[1],.52)
        rows=self.rows(m.BASELINE,.5)
        reference={**ref,'rows':rows,'aggregate':m.slots(rows,m.BASELINE,self.expected),
                   'source':{'path':'mock-reference','sha256':'fixture'},'origin_source':{'path':'mock-overlay','sha256':'fixture'}}
        with patch.object(m,'cohort',return_value=(self.expected,self.dates)),patch.object(m,'load_reference',return_value=reference):
            out=m.compare(pa,pb,bootstrap_samples=10)
        self.assertEqual(out['primary_view'],'compatible_firstpass')
        self.assertEqual(set(out['primary_comparisons']),{arm+'_minus_dated_merged' for arm in m.ARMS})
        for arm in m.ARMS:
            self.assertIsNone(out['technical_recovery_supplement'][arm]['final_minus_own_compatible_firstpass'])
            self.assertIsNone(out['technical_recovery_supplement'][arm]['comparison_with_historical_reference'])
        self.assertEqual(out['new_model_calls'],0)


if __name__=='__main__':unittest.main()
