"""Independent C summary scope/direction/replay-adapter checks; no API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_boundary_variants_summary_v1 import rows, expected, summary_fixture
from tools import summarize_boundary_variants_full1540_v1 as s


class IndependentBoundarySummaryReview(unittest.TestCase):
    def test_nonzero_right_minus_left_and_historical_directions(self):
        def changed(arm,f1,correct):
            value=rows(arm)
            for i,r in enumerate(value):r['scores'].update(official_locomo_f1=f1,judge_correct=float(i<correct))
            return value
        c1,c2=(summary_fixture(a) for a in s.ARMS)
        for report,f1,correct in ((c1,.4,400),(c2,.6,600)):
            for view in s.VIEWS:
                report['slots_by_view'][view]=changed(report['arm'],f1,correct)
                report['views'][view]=s.aggregate(report['slots_by_view'][view],report['arm'])
        histories={'baseline_sdk20':changed('baseline_sdk20',.1,200),
                   'dated_merged':changed('dated_merged',.2,300)}
        with (patch.object(s.locktool,'verify_lock',return_value={'baseline':{'summary':{}}}),
              patch.object(s.full,'canonical',return_value=expected()),
              patch.object(s,'load_report',side_effect=[c1,c2]),patch.object(s,'validate_report'),
              patch.object(s.old.plain,'read',return_value={k:'same' for k in
                  ('plain_text_config','evaluation_protocol','dataset_sha256','official_metric_provenance','reference_dates')}),
              patch.object(s,'history',return_value=histories),patch.object(s.old.plain,'pointer',return_value={})):
            result=s.compare('c1','c2',bootstrap_samples=2)
        for view in s.VIEWS:
            pairs=result['views'][view]['paired']
            context=result['views'][view]['pair_context']
            self.assertEqual(context['boundary_neighbor_minus_boundary_multiscale']['left_view'],view)
            self.assertTrue(context['boundary_neighbor_minus_boundary_multiscale']['phase_matched'])
            historic=context['boundary_neighbor_minus_dated_merged']
            self.assertEqual((historic['left_method'],historic['right_method']),('dated_merged',s.ARMS[1]))
            self.assertEqual((historic['left_view'],historic['right_view']),(s.full.FINAL,view))
            self.assertFalse(historic['phase_matched'])
            self.assertTrue(historic['historical_descriptive_only'])
            main=pairs['boundary_neighbor_minus_boundary_multiscale']
            self.assertAlmostEqual(main['official_locomo_f1']['metrics']['official_locomo_f1']['other_minus_baseline'],.2)
            self.assertAlmostEqual(main['judge_correct']['metrics']['judge_correct']['other_minus_baseline'],200/1540)
            baseline=pairs['boundary_multiscale_minus_baseline_sdk20']
            self.assertAlmostEqual(baseline['official_locomo_f1']['metrics']['official_locomo_f1']['other_minus_baseline'],.3)
            dated=pairs['boundary_neighbor_minus_dated_merged']
            self.assertAlmostEqual(dated['official_locomo_f1']['metrics']['official_locomo_f1']['other_minus_baseline'],.4)

    def test_pair_missingness_is_metric_specific_and_planned1540_stays(self):
        left,right=rows(s.ARMS[0]),rows(s.ARMS[1])
        for r in left[:2]:r['scores']={m:None for m in s.METRICS}
        for r in right[2:5]:r['scores']['judge_correct']=None
        paired=s.old.scoped(s.full.comparison.pair,N=1540)(left,right,bootstrap_samples=2)
        self.assertEqual(paired['official_locomo_f1']['common_success_count'],1538)
        self.assertEqual(paired['judge_correct']['common_success_count'],1535)
        self.assertEqual(paired['judge_correct']['independent_valid_denominators'],
                         {'baseline':1538,'other':1537,'planned_each':1540})
        self.assertTrue(paired['judge_correct']['missing_as_zero_delta_sensitivity']['is_imputation_not_observed_score'])

    def test_real_single_arm_adapter_body_all1540_without_mutating_old(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);arm=s.ARMS[0];jobs=[];canonical={}
            for q in expected().values():
                j={**q,'job_id':arm+'/'+q['query_id'],'method':arm,'question':'Q?',
                   'reference':'R','reference_date':'2023-01-01'}
                jobs.append(j);canonical[j['query_id']]=j
            header={'schema':s.driver.SCHEMA,'phase_kind':'firstpass','arm':arm,
                    'jobs':jobs,'gate':{'path':'unused_gate','sha256':'mock'}}
            for name,value in (('run_manifest.json',header),('source_jobs.json',{'jobs':jobs}),
                               ('run_outcome.json',{'results':[]})):
                s.save_new(run/name,value)
            records={j['job_id']:{'reader':{'status':'complete','result':{'metrics':{
                     'official_locomo_f1':.5,'custom_token_f1':.25},'prediction':{'answer':'R'}}},
                    'judge':{'status':'complete','result':{'correct':False,'label':'NO'}},
                    'compat':None,'qa':None,'errors':[]} for j in jobs}
            phase={'run':str(run),'manifest':header,'outcome':{'counts':{}},'records':records,
                   'ledger':[],'ledger_refs':[],'errors':[],'usage':s.measure_usage([]),'inputs':{}}
            accounting={'combined_lineage':{'scope_complete':True,'new_namespace_only':{'total_cost_usd':0}},
                        'retrieval_records':[]}
            old_arms=s.old.ARMS;old_n=s.old.DENOMINATOR;old_code=s.old.summarize.__code__
            with (patch.object(s.locktool,'verify_lock',return_value={}),patch.object(s,'require_closed'),
                  patch.object(s,'audit_phase',return_value=phase),
                  patch.object(s.old,'audit_retrieval_and_usage',return_value=accounting),
                  patch.object(s.full,'canonical',return_value=canonical),
                  patch.object(s.driver,'recovery_jobs',return_value=[]),
                  patch.object(s,'upstream_accounting',return_value={'total_cold_start_cost_usd':None})):
                report=s.summarize(run)
            self.assertEqual((report['arm'],report['source_arm']),(arm,'c1_multiscale'))
            self.assertEqual(report['planned_per_arm'],1540)
            self.assertEqual(report['views']['final']['arms'][arm]['metrics']['judge_correct']['valid_denominator'],1540)
            self.assertEqual(report['views']['final']['arms'][arm]['metrics']['judge_correct']['complete_only_mean'],0.)
            self.assertEqual(report['slots_by_view']['final'],report['slots_by_view']['compatible_firstpass'])
            self.assertEqual((s.old.ARMS,s.old.DENOMINATOR,s.old.summarize.__code__),(old_arms,old_n,old_code))
            self.assertFalse(list(run.rglob('api_calls.jsonl')))

    def test_partial_upstream_ledger_keeps_total_unknown_and_sourcearm_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);first=root/'qa';writer=root/'c1_multiscale/writer';glob=root/'c1_multiscale/global'
            gate=root/'gate.json';parent=root/'manifest.json';globalm=glob/'manifest.json'
            for path,value in ((parent,{}),(globalm,{})):
                path.parent.mkdir(parents=True,exist_ok=True);s.save_new(path,value)
            wm={'boundary_variant':{'arm':'c1_multiscale','parent':s.old.plain.pointer(parent),
                                   'global':s.old.plain.pointer(globalm)}}
            writer.mkdir(parents=True);s.save_new(writer/'run_manifest.json',wm)
            s.save_new(gate,{'source_arm':'c1_multiscale','source_manifest':s.old.plain.pointer(writer/'run_manifest.json')})
            first.mkdir();s.save_new(first/'run_manifest.json',{'gate':s.old.plain.pointer(gate)})
            ledger=root/'c1_multiscale/boundary/conv_0/api/api_calls.jsonl';ledger.parent.mkdir(parents=True)
            ledger.write_text(json.dumps({'provider':'gemini','model':'gemini-2.5-flash',
                'status':200,'cache_hit':False,'usage':{'promptTokenCount':10,'candidatesTokenCount':2}})+'\n',encoding='utf-8')
            report=s.upstream_accounting(first,s.ARMS[0])
            self.assertIsNone(report['stages']['boundary']['total_cost_usd'])
            self.assertFalse(report['stages']['boundary']['scope_complete'])
            self.assertEqual(len(report['stages']['boundary']['missing_ledgers']),9)
            self.assertIsNone(report['total_cold_start_cost_usd'])
            self.assertFalse(report['inherited_turn_embedding_cost_included'])
            with self.assertRaises(s.InvalidArtifact):s.upstream_accounting(first,s.ARMS[1])

    def test_stored_lineage_or_extra_fields_cannot_be_silently_corrected(self):
        fresh=summary_fixture(s.ARMS[0])
        fresh.update(schema=s.SCHEMA,model_calls_in_summary=0,native_clients=0,
            source_arm='c1_multiscale',created_utc='fresh',
            summary_code_sha256={p:'frozen' for p in s.CODE})
        stored=copy.deepcopy(fresh);stored.update(content_sha256='seal',created_utc='older')
        with (patch.object(s.old.plain,'verify',return_value=stored),
              patch.object(s.old.plain,'sha',return_value='frozen'),
              patch.object(s.full,'checked',return_value=Path('first/run_manifest.json')),
              patch.object(s,'summarize',return_value=fresh)):
            self.assertEqual(s.load_report('summary'),fresh)
            stored['source_arm']='c2_neighbour'
            with self.assertRaisesRegex(s.InvalidArtifact,'source_arm'):s.load_report('summary')
            stored['source_arm']='c1_multiscale';stored['hidden_posthoc_selection']=True
            with self.assertRaisesRegex(s.InvalidArtifact,'field/lineage'):s.load_report('summary')

    def test_terminal_recovery_alias_never_means_all_slots_valid(self):
        value=rows(s.ARMS[0]);value[0]['scores']['judge_correct']=None
        detail=s.validity(value,[])
        self.assertEqual(detail['planned_slots'],1540)
        self.assertEqual(detail['judge_unavailable_slots'],1)
        self.assertEqual(detail['valid_denominators']['official_locomo_f1'],1540)
        self.assertFalse(detail['all_slots_valid'])
        self.assertFalse(s.validity(rows(s.ARMS[0]),[{'error':'replay mismatched'}])['all_slots_valid'])

    def test_missing_thought_meter_does_not_become_verified_zero_cost(self):
        physical={'provider':'gemini','model':'gemini-2.5-flash','cache_hit':False,'status':200,
            'usage':{'promptTokenCount':10,'candidatesTokenCount':2}}
        old=s.measure_usage([physical])
        self.assertIsNotNone(old['total_cost_usd'])
        assumptions=s.estimator_assumptions([physical],old)
        self.assertEqual(assumptions['missing_Gemini_thoughtsTokenCount_physical_attempts'],1)
        self.assertIsNone(assumptions['conservative_total_cost_usd'])
        self.assertTrue(assumptions['legacy_Gemini_missing_thoughtsTokenCount_defaults_to_zero'])
        cached={**physical,'cache_hit':True}
        embed={**physical,'model':'gemini-embedding-001'}
        self.assertEqual(s.estimator_assumptions([cached,embed],{'total_cost_usd':None})[
            'missing_Gemini_thoughtsTokenCount_physical_attempts'],0)

    def test_actual_closed_global_accounting_keeps_lost_successes_and_cost_unknown(self):
        parent=s.ROOT/'runs/boundary_variants_v1_20261003'
        if not parent.exists():self.skipTest('Actual closed C raw evidence not present on this checkout')
        with tempfile.TemporaryDirectory() as temp:
            for arm,raw,count,missing,lower in ((s.ARMS[0],'c1_multiscale',1494,10,1500),
                                              (s.ARMS[1],'c2_neighbour',1456,11,1463)):
                fixture=Path(temp)/raw;writer=fixture/'unused_writer';qa=fixture/'qa'
                writer.mkdir(parents=True);qa.mkdir()
                glob=parent/raw/'global'
                audit=s.ROOT/'research/planned_execution_2026-10-03/live'/(
                    'c1_global_pair_raw_audit_v2.json' if raw=='c1_multiscale' else 'c2_global_pair_raw_audit_v2.json')
                s.save_new(writer/'run_manifest.json',{'boundary_variant':{'arm':raw,
                    'parent':s.old.plain.pointer(parent/'manifest.json'),
                    'global':s.old.plain.pointer(glob/'manifest.json')}})
                gate=fixture/'gate.json'
                s.save_new(gate,{'source_arm':raw,'source_manifest':s.old.plain.pointer(writer/'run_manifest.json')})
                s.save_new(qa/'run_manifest.json',{'gate':s.old.plain.pointer(gate)})
                result=s.upstream_accounting(qa,arm,audit)['stages']['global_grouping']
                self.assertEqual(result['unique_successful_raw_requests'],count)
                self.assertEqual(len(result['derived_ledger_evidence']['missing_or_duplicate_successes']),missing)
                self.assertEqual(result['attempt_count_lower_bound_from_result_and_ledger'],lower)
                self.assertFalse(result['scope_complete']);self.assertFalse(result['observed_counts_are_complete_totals'])
                self.assertIsNone(result['total_cost_usd']);self.assertIsNone(result['actual_total_physical_attempts'])

    def test_actual_derived_resealed_false_complete_is_rejected(self):
        root=s.ROOT/'research/planned_execution_2026-10-03/live'
        proof=root/'c2_global_pair_raw_audit_v2.json'
        if not proof.exists():self.skipTest('Actual closed C raw evidence not present on this checkout')
        value=s.old.plain.verify(proof);forged=copy.deepcopy(value);forged.pop('content_sha256')
        forged['ledger']['physical_attempt_ledger_complete']=True
        forged['ledger']['missing_or_duplicate_successes']=[]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'false_complete.json';s.save_new(path,forged)
            with self.assertRaisesRegex(s.InvalidArtifact,'does not reproduce'):
                s.bind_global_accounting_audit(path,Path(value['run_dir']),value['arm'],s.Snapshot())


if __name__=='__main__':unittest.main()
