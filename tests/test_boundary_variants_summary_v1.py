"""Mock/counterexample summary admission, never creates Reader/Judge responses."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import summarize_boundary_variants_full1540_v1 as s


def expected():
    return {f'c{i//154}_q{i%154}':{'query_id':f'c{i//154}_q{i%154}','conversation':i//154,
        'category':i%4+1} for i in range(1540)}


def rows(arm):
    return [{'query_id':q['query_id'],'conversation':q['conversation'],'category':q['category'],'method':arm,
        'selection_status':'complete','reader_status':'complete','judge_status':'complete','qa_status':'complete',
        'scores':{'official_locomo_f1':.5,'custom_token_f1':.25,'judge_correct':1.},'errors':[],
        'reader_answer_characters':3,'reader_answer_whitespace_words':1} for q in expected().values()]


def summary_fixture(arm):
    rs=rows(arm)
    return {'arm':arm,'planned_per_arm':1540,'recovery_supplied':False,
        'slots_by_view':{v:copy.deepcopy(rs) for v in s.VIEWS},
        'views':{v:s.aggregate(rs,arm) for v in s.VIEWS},'source_manifest':{},
        'phase_sources':[{'manifest':{'path':'unused.json'}}],
        'recovery_status':'not_needed','final_view_is_uniform_recovery_complete':True,
        'uniform_recovery_policy_closed':True,'final_all_slots_valid':True,
        'validity_by_view':{v:s.validity(rs,[]) for v in s.VIEWS},
        'failures_by_view':{v:[] for v in s.VIEWS},'audit_errors':[],
        'declared_failure_artifacts_by_phase':{},
        'usage_estimator_assumptions':{'conservative_total_cost_usd':None},
        'current_qa_plus_retrieval_usage':{'total_cost_usd':None},'pipeline_accounting':{'total_cold_start_cost_usd':None}}


class SummaryTests(unittest.TestCase):
    def test_reader_judge_independent_1540_denominators(self):
        rs=rows(s.ARMS[0]);rs[0]['scores']={k:None for k in s.METRICS};rs[0]['reader_status']='failed';rs[0]['judge_status']='failed'
        rs[1]['scores']['judge_correct']=None;rs[1]['judge_status']='failed'
        out=s.aggregate(rs,s.ARMS[0])['arms'][s.ARMS[0]]['metrics']
        self.assertEqual(out['official_locomo_f1']['valid_denominator'],1539)
        self.assertEqual(out['judge_correct']['valid_denominator'],1538)
        self.assertEqual(out['judge_correct']['planned_denominator'],1540)
        self.assertIsNone(out['judge_correct']['full_cohort_observed_mean'])
        self.assertEqual(out['judge_correct']['zero_imputed_sensitivity']['imputed_slots'],2)
        self.assertEqual(len(s.failures(rs)),2)

    def test_all_ids_including_failures_required(self):
        rs=rows(s.ARMS[0])
        for broken in (rs[:-1],rs[:-1]+[rs[0]]):
            with self.assertRaises(s.InvalidArtifact):s.full.validate_slots(broken,s.ARMS[0],expected())
        rs[0]['scores']['official_locomo_f1']=None
        with self.assertRaises(s.InvalidArtifact):s.full.validate_slots(rs,s.ARMS[0],expected())

    def test_views_never_mix_and_inherited_reader_not_overwritten(self):
        job={'job_id':'job','query_id':'q','conversation':0,'category':1,'method':s.ARMS[0]}
        a={'reader':{'status':'complete','result':{'metrics':{'official_locomo_f1':.5,'custom_token_f1':.25},'prediction':{'answer':'old'}}},
            'judge':{'status':'failed'},'compat':{'status':'compat_valid','interpretation':{'correct':True,'label':'YES'}},'errors':[]}
        b=copy.deepcopy(a);b['reader']['result']['metrics']['official_locomo_f1']=1.;b['reader']['result']['prediction']['answer']='new'
        b['judge']={'status':'complete','result':{'correct':False,'label':'NO'}}
        first={'records':{'job':a}};retry={'records':{'job':b}}
        strict=s.old.base.record_rows([job],first)[0]
        compat=s.old.base.record_rows([job],first,compatible=True)[0]
        final=s.old.base.record_rows([job],first,retry,compatible=True,final=True)[0]
        self.assertIsNone(strict['scores']['judge_correct']);self.assertEqual(compat['scores']['judge_correct'],1.)
        self.assertEqual(final['scores']['judge_correct'],0.)  # original resolver strict priority, not any-correct
        self.assertEqual(final['scores']['official_locomo_f1'],.5)
        self.assertEqual(final['reader_answer_characters'],3)

    def test_no_recovery_final_cannot_change_compatible(self):
        r=summary_fixture(s.ARMS[0]);r['slots_by_view']['final'][0]['scores']['judge_correct']=0.
        r['views']['final']=s.aggregate(r['slots_by_view']['final'],r['arm'])
        with patch.object(s.full,'validate_jobs',return_value=None):
            with self.assertRaisesRegex(s.InvalidArtifact,'Final without recovery'):s.validate_report(r,expected())

    def test_pending_active_run_rejected_without_creating_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(s.InvalidArtifact):s.require_closed(tmp)
            self.assertFalse((Path(tmp)/'firstpass_seal.json').exists())

    def test_unknown_usage_and_price_not_zero(self):
        missing=s.measure_usage(None)
        unknown=s.measure_usage([{'provider':'jev','model':'jev-1.13.0','status':200,'cache_hit':False,'usage':{}}])
        self.assertIsNone(missing['total_cost_usd']);self.assertIsNone(unknown['total_cost_usd'])
        self.assertEqual(unknown['physical_attempts'],1)
        cached=s.measure_usage([{'cache_hit':True,'usage':{'promptTokenCount':999}}])
        self.assertEqual(cached['physical_attempts'],0);self.assertEqual(cached['cache_events'],1)
        legacy=s.measure_usage([{'provider':'gemini','model':'gemini-2.5-flash','status':200,
            'usage':{'promptTokenCount':10,'candidatesTokenCount':2}}])
        gap=s.estimator_assumptions([{'provider':'gemini','usage':{'promptTokenCount':10,'candidatesTokenCount':2}}],legacy)
        self.assertEqual(gap['missing_Gemini_thoughtsTokenCount_physical_attempts'],1)
        self.assertIsNone(gap['conservative_total_cost_usd'])

    def test_comparison_keeps_all_three_views_and_pair_direction(self):
        c1,c2=(summary_fixture(a) for a in s.ARMS)
        lock={'baseline':{'summary':{'path':'historical','sha256':'frozen'}}}
        history={'baseline_sdk20':rows('baseline_sdk20'),'dated_merged':rows('dated_merged')}
        with patch.object(s.locktool,'verify_lock',return_value=lock),patch.object(s.full,'canonical',return_value=expected()),\
             patch.object(s,'load_report',side_effect=[c1,c2]),patch.object(s,'validate_report',return_value=None),\
             patch.object(s.old.plain,'read',return_value={k:'same' for k in ('plain_text_config','evaluation_protocol','dataset_sha256','official_metric_provenance','reference_dates')}),\
             patch.object(s,'history',return_value=history),patch.object(s.old.plain,'pointer',return_value={'path':'test','sha256':'fixture'}):
            report=s.compare('c1','c2',bootstrap_samples=2)
        self.assertEqual(set(report['views']),set(s.VIEWS));self.assertEqual(report['model_calls_in_summary'],0)
        pair=report['views']['final']['paired']['boundary_neighbor_minus_boundary_multiscale']['judge_correct']
        self.assertEqual(pair['independent_valid_denominators']['planned_each'],1540)
        self.assertEqual(pair['wins_ties_losses'],{'wins':0,'ties':1540,'losses':0})
        self.assertIsNone(report['usage'][s.ARMS[0]]['QA_and_retrieval']['total_cost_usd'])
        binding=report['views']['compatible_firstpass']['pair_context'][s.ARMS[0]+'_minus_dated_merged']
        self.assertEqual(binding['left_view'],s.full.FINAL)
        self.assertEqual(binding['right_view'],'compatible_firstpass')
        self.assertTrue(binding['historical_descriptive_only']);self.assertFalse(binding['phase_matched'])

    def test_applied_recovery_does_not_prove_allvalid(self):
        rs=rows(s.ARMS[0]);rs[0]['scores']['judge_correct']=None
        value=s.validity(rs,[])
        self.assertFalse(value['all_slots_valid']);self.assertEqual(value['judge_unavailable_slots'],1)
        self.assertFalse(s.validity(rows(s.ARMS[0]),[{'stage':'raw_audit'}])['all_slots_valid'])

    def test_declared_failure_error_and_unknown_stage_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reader=root/'reader.json'
            s.save_new(reader,{'status':'failed','error':{'type':'RuntimeError','message':'offline test failure'}})
            outcome=root/'run_outcome.json';s.save_new(outcome,{'results':[{'conversation':0,'query_id':'q',
                'method':s.ARMS[0],'reader':s.old.plain.pointer(reader),'judge':None,'compat':None,'qa':None}]})
            data=s.declared_failures([{'manifest':{'path':'offline_test_manifest'},'outcome':s.old.plain.pointer(outcome)}])
            values=data['offline_test_manifest']
            self.assertEqual(values[0]['error']['type'],'RuntimeError')
            self.assertEqual(sum(r['status']=='unknown' for r in values),3)

    def test_single_arm_upstream_parent_placement_not_directory_guess(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'arbitrary_parent';global_root=root/'new_global_location';writer=root/'custom_native_writer';qa=root/'single_arm_qa'
            for folder in (parent,global_root,writer,qa):folder.mkdir()
            for path in (parent/'manifest.json',global_root/'manifest.json'):path.write_text('{}',encoding='utf-8')
            meta={'arm':'c1_multiscale','parent':s.old.plain.pointer(parent/'manifest.json'),'global':s.old.plain.pointer(global_root/'manifest.json')}
            (writer/'run_manifest.json').write_text(json.dumps({'boundary_variant':meta}),encoding='utf-8')
            gate=root/'gate.json';s.save_new(gate,{'source_manifest':s.old.plain.pointer(writer/'run_manifest.json'),'source_arm':'c1_multiscale'})
            (qa/'run_manifest.json').write_text(json.dumps({'gate':s.old.plain.pointer(gate)}),encoding='utf-8')
            for ci in range(10):
                bp=parent/'c1_multiscale'/'boundary'/f'conv_{ci}'/'api/api_calls.jsonl';wp=writer/f'conv_{ci}'/'c1_multiscale'/'native/api_calls.jsonl'
                for path in (bp,wp):path.parent.mkdir(parents=True);path.write_text('',encoding='utf-8')
            gp=global_root/'api/api_calls.jsonl';gp.parent.mkdir();gp.write_text('',encoding='utf-8')
            other=parent/'c2_neighbour'/'boundary/conv_0/api/api_calls.jsonl';other.parent.mkdir(parents=True);other.write_text('INVALID OTHER ARM',encoding='utf-8')
            out=s.upstream_accounting(qa,s.ARMS[0])
            self.assertEqual(out['source_writer_manifest'],s.old.plain.pointer(writer/'run_manifest.json'))
            self.assertTrue(out['stages']['boundary']['scope_complete'])
            self.assertTrue(out['stages']['writer']['scope_complete'])
            self.assertFalse(out['stages']['global_grouping']['scope_complete'])  # no derived attempt closure
            self.assertIsNone(out['total_cold_start_cost_usd'])

    def test_damaged_cost_ledger_retains_positions_without_repair(self):
        data=b'{"provider":"jev","status":200,"usage":{}}\n\n}\n[]\n'
        parsed,diagnostics=s.accounting_ledger(data,'offline_ledger.jsonl')
        self.assertEqual(len(parsed),1)
        self.assertEqual(diagnostics['blank_lines'],[2])
        self.assertEqual([r['line'] for r in diagnostics['malformed_or_invalid_rows']],[3,4])
        self.assertEqual(s.measure_usage(parsed)['physical_attempts'],1)
        self.assertIsNone(s.measure_usage(parsed)['total_cost_usd'])

    def test_derived_cost_audit_must_bind_current_bytes_and_full_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gp=root/'global';ledger=gp/'api/api_calls.jsonl';ledger.parent.mkdir(parents=True)
            ledger.write_text('}\n',encoding='utf-8');audit=root/'audit.json'
            value={'schema':'boundary-global-pair-raw-independent-audit-v1','arm':'c1_multiscale',
                'run_dir':str(gp),'result_evidence_complete':True,
                'all_pair_scores_and_full_decisions_replayed':True,'model_calls':0,'source_writes':0,
                'audit_code_sha256':s.old.plain.sha(s.ROOT/'tools/audit_boundary_global_raw_v1.py'),
                'conversations':[{'conversation':i,'validated_pairs':1,'full_decision_replay':True} for i in range(10)],
                'unique_pair_raw_requests':10,'input_sha256':{str(ledger.resolve()):s.old.plain.sha(ledger)}}
            s.save_new(audit,value)
            with patch('tools.audit_boundary_global_raw_v1.audit',return_value=value):
                ref,loaded=s.bind_global_accounting_audit(audit,gp,'c1_multiscale',s.Snapshot())
            self.assertEqual(ref,s.old.plain.pointer(audit));self.assertEqual(loaded['unique_pair_raw_requests'],10)
            with patch('tools.audit_boundary_global_raw_v1.audit',return_value={**value,'unique_pair_raw_requests':11}):
                with self.assertRaisesRegex(s.InvalidArtifact,'does not reproduce'):
                    s.bind_global_accounting_audit(audit,gp,'c1_multiscale',s.Snapshot())
            ledger.write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(s.InvalidArtifact,'snapshot changed'):
                s.bind_global_accounting_audit(audit,gp,'c1_multiscale',s.Snapshot())
            with self.assertRaisesRegex(s.InvalidArtifact,'identity'):
                s.bind_global_accounting_audit(audit,gp,'c2_neighbour',s.Snapshot())


if __name__=='__main__':unittest.main()
