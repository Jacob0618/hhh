"""Offline C full1540 thin replay/summary; no scoring, clients or source edits.

Reuse the original single-arm raw Reader/Judge replay, compatibility resolver,
independent-denominator aggregation and paired statistics in private globals.
Only namespace/cardinality admission and C/historical presentation are added.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import summarize_regroup_refinement_qa_v1 as old
from tools import summarize_regroup_refinement_full1540_v1 as full
from tools import run_boundary_variants_qa_v1 as driver
from tools import freeze_sdk20_full1540_v1 as locktool
from tools.summarize_mem0_sparse1540 import Snapshot,measure_usage
from tools.summarize_sdk20_modules import InvalidArtifact
from tools.run_sdk20_modules import save_new,now
from jev_mem.local_bm25_context_v2 import forbid_network

SCHEMA='boundary-variants-single-arm-full1540-summary-v1'
COMPARISON_SCHEMA='boundary-variants-full1540-comparison-v1'
ARMS=('boundary_multiscale','boundary_neighbor')
RAW_ARMS={'boundary_multiscale':'c1_multiscale','boundary_neighbor':'c2_neighbour'}
N=1540
VIEWS=('strict_firstpass','compatible_firstpass','final')
METRICS=old.base.METRICS
DATED=ROOT/'research/full1540_execution_2026-10-02/dated_final_summary.json'
DATED_SHA='a6a6c76ff99cbe88fbb13da39bec513a3930819933e791080b29ec13649ff825'
CODE=('tools/summarize_boundary_variants_full1540_v1.py',*old.CODE,
      'tools/summarize_regroup_refinement_full1540_v1.py',
      'tools/compare_regroup_refinement400_v1.py','tools/run_boundary_variants_qa_v1.py',
      'tools/boundary_variants_qa_sources_v1.py','tools/freeze_sdk20_full1540_v1.py',
      'tools/audit_boundary_global_raw_v1.py')


def driver_for(run):
    if old.plain.read(Path(run)/'run_manifest.json').get('schema')!=driver.SCHEMA:
        raise InvalidArtifact('Only declared C full1540 QA schema supported')
    return driver


def require_closed(run):
    old.require_closed_files(run)
    header=old.plain.read(Path(run)/'run_manifest.json');driver_for(run)
    root=Path(header['parent']['run_dir']) if header['phase_kind']=='technical_recovery' else Path(run)
    # Summary must never invoke the driver's seal helper to create an artifact.
    if not (root/'firstpass_seal.json').is_file():
        raise InvalidArtifact('Existing whole1540 firstpass seal required; use health while active')
    sealed=old.plain.verify(root/'firstpass_seal.json')
    if (sealed.get('schema')!=driver.SCHEMA+'-firstpass-seal' or sealed.get('total')!=N
        or sealed.get('manifest')!=old.plain.pointer(root/'run_manifest.json')
        or sealed.get('outcome')!=old.plain.pointer(root/'run_outcome.json')
        or sealed.get('source_jobs')!=old.plain.pointer(root/'source_jobs.json')):
        raise InvalidArtifact('Firstpass seal/whole population binding differs')


def aggregate(rows,arm):
    fn=old.scoped(old.aggregate,ARMS=ARMS,DENOMINATOR=N)
    result=fn(rows,arm)
    result['interpretation']='Single-arm locked1540 report; independent Reader/F1 and Judge denominators; no paired comparison in this arm report.'
    return result


def audit_phase(run,source,*,parent=None):
    return old.scoped(old.audit_phase,driver_for=driver_for)(run,source,parent=parent)


def validate_report(report,expected):
    arm=report['arm']
    if arm not in ARMS or report['planned_per_arm']!=N or set(report['slots_by_view'])!=set(VIEWS):
        raise InvalidArtifact('Declared full1540 arm and all three views required')
    full.validate_jobs(report,expected,arm)
    for view,rows in report['slots_by_view'].items():
        full.validate_slots(rows,arm,expected)
        if report['views'][view]['arms']!={arm:aggregate(rows,arm)['arms'][arm]}:
            raise InvalidArtifact('Metric aggregates do not reproduce independent denominators')
    if not report['recovery_supplied'] and report['slots_by_view']['final']!=report['slots_by_view']['compatible_firstpass']:
        raise InvalidArtifact('Final without recovery must preserve compatible firstpass')


def failures(rows):
    return [{k:r.get(k) for k in ('query_id','conversation','category','selection_status','reader_status','judge_status','qa_status','errors','scores')}
            for r in rows if r['reader_status']!='complete' or r['judge_status']!='complete' or r.get('errors')]


def validity(rows,audit_errors):
    denominators={metric:sum(r['scores'][metric] is not None for r in rows) for metric in METRICS}
    return {'planned_slots':N,'valid_denominators':denominators,
        'reader_unavailable_slots':N-denominators['official_locomo_f1'],
        'judge_unavailable_slots':N-denominators['judge_correct'],
        'all_slots_valid':len(rows)==N and all(n==N for n in denominators.values()) and not audit_errors,
        'audit_error_count':len(audit_errors)}


def declared_failures(phases):
    output={}
    for phase in phases:
        outcome=old.plain.verify(full.checked(phase['outcome']))
        values=[]
        for row in outcome['results']:
            for stage in ('reader','judge','compat','qa'):
                ref=row.get(stage)
                value=old.plain.verify(full.checked(ref)) if ref else None
                if value is None or value.get('status') in ('failed','not_run','missing','invalid','unknown') or value.get('error'):
                    values.append({'conversation':row['conversation'],'query_id':row['query_id'],
                        'method':row['method'],'stage':stage,'artifact':ref,
                        'status':value.get('status') if value else 'unknown',
                        'error':value.get('error') if value else None})
        output[phase['manifest']['path']]=values
    return output


def estimator_assumptions(rows,legacy_usage):
    physical=[r for r in rows if r.get('cache_hit') is not True]
    missing_thoughts=sum(r.get('provider')=='gemini' and r.get('model')!='gemini-embedding-001'
                        and 'thoughtsTokenCount' not in (r.get('usage') or {}) for r in physical)
    return {'legacy_estimator':'unchanged_measure_usage_and_original_price_table',
        'legacy_total_is_estimate_not_invoice':True,
        'legacy_Gemini_missing_thoughtsTokenCount_defaults_to_zero':True,
        'missing_Gemini_thoughtsTokenCount_physical_attempts':missing_thoughts,
        'missing_counter_is_not_verified_zero':True,
        'embedding_requests_excluded_from_thought_counter_gap':True,
        'conservative_total_cost_usd':None if missing_thoughts else legacy_usage.get('total_cost_usd'),
        'note':'Preserve historical estimator fields; missing thought counters remain a metering gap, even when request thinkingBudget is0.'}


def qa_estimator_assumptions(phases,legacy_usage):
    rows=[];seen=set()
    for phase in phases:
        for ref in phase['ledgers']:
            path=full.checked(ref)
            if str(path) in seen:continue
            seen.add(str(path))
            rows.extend(json.loads(line) for line in path.read_bytes().decode('utf-8-sig').splitlines() if line.strip())
    return estimator_assumptions(rows,legacy_usage)


def accounting_ledger(data,path):
    """Read damaged cost evidence without repairing it or admitting QA artifacts."""
    rows=[];blank=[];damaged=[]
    for number,line in enumerate(data.decode('utf-8-sig').splitlines(),1):
        if not line.strip():blank.append(number);continue
        try:
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError('Ledger row is not an object')
            measure_usage([value])  # Original semantic usage checks still apply.
            rows.append(value)
        except (ValueError,TypeError,KeyError) as exc:
            damaged.append({'path':str(Path(path).resolve()),'line':number,
                'length':len(line),'error_type':type(exc).__name__})
    return rows,{'path':str(Path(path).resolve()),'blank_lines':blank,'malformed_or_invalid_rows':damaged}


def bind_global_accounting_audit(path,global_run,raw_arm,snapshot):
    """Bind an explicit derived proof to current bytes; never substitute its ledger."""
    if path is None:return None,None
    value=old.plain.verify(path)
    if (value.get('schema')!='boundary-global-pair-raw-independent-audit-v1'
        or value.get('arm')!=raw_arm or Path(value['run_dir']).resolve()!=global_run.resolve()
        or value.get('result_evidence_complete') is not True
        or value.get('all_pair_scores_and_full_decisions_replayed') is not True
        or value.get('model_calls')!=0 or value.get('source_writes')!=0
        or value.get('audit_code_sha256')!=old.plain.sha(ROOT/'tools/audit_boundary_global_raw_v1.py')):
        raise InvalidArtifact('Global accounting audit identity/replay/code binding differs')
    convs=value['conversations']
    if (sorted(r['conversation'] for r in convs)!=list(range(10))
        or any(r.get('full_decision_replay') is not True for r in convs)
        or sum(r['validated_pairs'] for r in convs)!=value['unique_pair_raw_requests']):
        raise InvalidArtifact('Global accounting audit full-conversation population differs')
    ledger_path=str((global_run/'api/api_calls.jsonl').resolve())
    if ledger_path not in value['input_sha256']:
        raise InvalidArtifact('Global accounting audit omits actual ledger SHA')
    for name,digest in value['input_sha256'].items():
        if snapshot.sha(name)!=digest:raise InvalidArtifact('Global accounting audit source snapshot changed: '+name)
    # Replay the independent original-criterion pair/decision proof. A stored
    # metadata claim alone cannot admit missing ledger rows or wrong groupings.
    auditor=importlib.import_module('tools.audit_boundary_global_raw_v1')
    replay=auditor.audit(global_run)
    omit={'created_utc','content_sha256'}
    if ({k:v for k,v in value.items() if k not in omit}
        !={k:v for k,v in replay.items() if k not in omit}):
        raise InvalidArtifact('Global accounting audit does not reproduce raw pair/full decision evidence')
    snapshot.raw(path)
    return old.plain.pointer(path),value


def upstream_accounting(firstpass,arm,global_accounting_audit=None):
    """Snapshot bound C cost evidence; result integrity is not billing integrity."""
    m=old.plain.read(Path(firstpass)/'run_manifest.json')
    gate=old.plain.verify(full.checked(m['gate']))
    writer_run=full.checked(gate['source_manifest']).parent
    writer_manifest=old.plain.read(writer_run/'run_manifest.json')
    meta=writer_manifest['boundary_variant'];raw_arm=RAW_ARMS[arm]
    if meta['arm']!=raw_arm or gate['source_arm']!=raw_arm:
        raise InvalidArtifact('Cost source writer/QA arm differs')
    parent=full.checked(meta['parent']).parent;global_run=full.checked(meta['global']).parent
    paths={'boundary':[parent/raw_arm/'boundary'/f'conv_{ci}'/'api/api_calls.jsonl' for ci in range(10)],
           'global_grouping':[global_run/'api/api_calls.jsonl'],
           'writer':[writer_run/f'conv_{ci}'/raw_arm/'native/api_calls.jsonl' for ci in range(10)]}
    snapshot=Snapshot();output={}
    audit_ref,audit=bind_global_accounting_audit(global_accounting_audit,global_run,raw_arm,snapshot)
    for stage,items in paths.items():
        rows=[];missing=[];refs=[];diagnostics=[]
        for path in items:
            if not path.exists():missing.append(str(path));continue
            data=snapshot.raw(path);refs.append({'path':str(path.resolve()),'sha256':snapshot.sha(path)})
            parsed,diagnostic=accounting_ledger(data,path)
            rows.extend(parsed);diagnostics.append(diagnostic)
        usage=measure_usage(rows if refs else None)
        clean=not missing and not any(d['blank_lines'] or d['malformed_or_invalid_rows'] for d in diagnostics)
        complete=clean and (stage!='global_grouping' or audit is not None and audit['ledger']['physical_attempt_ledger_complete'] is True)
        usage.update(scope_complete=complete,ledger_file_scope_complete=clean,missing_ledgers=missing,
            ledger_sources=refs,ledger_diagnostics=diagnostics,
            observed_valid_rows_usage=copy.deepcopy(usage),
            physical_attempt_count_exact=usage.get('physical_attempts') if complete else None,
            physical_attempt_count_observed_lower_bound=usage.get('physical_attempts'),
            observed_counts_are_complete_totals=complete)
        if stage=='global_grouping':
            usage['derived_result_and_ledger_audit']=audit_ref
            usage['derived_ledger_evidence']=audit['ledger'] if audit else None
            usage['unique_successful_raw_requests']=audit['unique_pair_raw_requests'] if audit else None
            usage['attempt_count_lower_bound_from_result_and_ledger']=max(
                usage.get('physical_attempts',0),audit['unique_pair_raw_requests']+usage.get('transport_or_unknown',0)) if audit else None
            usage['actual_total_physical_attempts']=audit['transport_attempt_count_exact'] if audit else None
            usage['result_integrity_is_not_billing_integrity']=True
        if not complete:usage['total_cost_usd']=None
        usage['usage_estimator_assumptions']=estimator_assumptions(rows,usage)
        output[stage]=usage
    if snapshot.unchanged():raise InvalidArtifact('C upstream ledger changed during closed snapshot')
    return {'scope':'this_C_arm_boundary_global_writer_only; QA/retrieval separate',
            'stages':output,'source_writer_manifest':gate['source_manifest'],
            'derived_global_accounting_audit':audit_ref,
            'input_sha256':{str(p):h for p,h in snapshot.files.items()},
            'inherited_turn_embedding_cost_included':False,'total_cold_start_cost_usd':None,
            'note':'Original price/usage estimator retained. Missing usage/prices/ledgers are unknown, not zero; imported embeddings are inherited computation.'}


def summarize(firstpass,recovery=None,global_accounting_audit=None):
    lock=locktool.verify_lock();require_closed(firstpass)
    fn=old.scoped(old.summarize,ARMS=ARMS,DENOMINATOR=N,driver_for=driver_for,
        require_closed_files=require_closed,audit_phase=audit_phase,aggregate=aggregate)
    result=fn(firstpass,recovery)
    result.update(schema=SCHEMA,full1540_lock=old.plain.pointer(locktool.LOCK),
        source_arm=RAW_ARMS[result['arm']],summary_code_sha256={p:old.plain.sha(ROOT/p) for p in CODE})
    expected=full.canonical(lock);validate_report(result,expected)
    m=old.plain.read(Path(firstpass)/'run_manifest.json');out=old.plain.verify(Path(firstpass)/'run_outcome.json')
    try:
        remaining=driver.recovery_jobs(m,out)
        recovery_status='applied_one_uniform_technical_recovery' if recovery else 'pending_uniform_technical_recovery' if remaining else 'not_needed'
    except ValueError:
        remaining=None;recovery_status='blocked_local_error_requires_review'
    result.update(recovery_status=recovery_status,
        final_view_is_uniform_recovery_complete=recovery_status in ('applied_one_uniform_technical_recovery','not_needed'),
        uniform_recovery_policy_closed=recovery_status in ('applied_one_uniform_technical_recovery','not_needed'),
        recovery_policy_is_not_all_valid_proof=True,
        validity_by_view={v:validity(rows,result['audit_errors']) for v,rows in result['slots_by_view'].items()},
        final_all_slots_valid=validity(result['slots_by_view']['final'],result['audit_errors'])['all_slots_valid'],
        failures_by_view={v:failures(rows) for v,rows in result['slots_by_view'].items()},
        declared_failure_artifacts_by_phase=declared_failures(result['phase_sources']),
        usage_estimator_assumptions=qa_estimator_assumptions(result['phase_sources'],result['current_qa_plus_retrieval_usage']),
        pipeline_accounting=upstream_accounting(firstpass,result['arm'],global_accounting_audit))
    result['limitations']=[s.replace('fixed400','locked1540') for s in result['limitations']]
    result['limitations'] += ['All1540 planned slots are retained; terminal/sealed does not mean all valid.',
        'Final without supplied recovery remains compatible firstpass; pending recovery and remaining failures are explicit.',
        'No Reader/Judge regenerated; compatibility only replays original raw label rules.',
        'Boundary/global/Writer and QA/retrieval costs remain separate; no cold-start total inferred.']
    return result


def load_report(path):
    stored=old.plain.verify(path)
    if stored.get('schema')!=SCHEMA or stored.get('model_calls_in_summary')!=0 or stored.get('native_clients')!=0:
        raise InvalidArtifact('An offline C full1540 summary required')
    if stored.get('summary_code_sha256')!={p:old.plain.sha(ROOT/p) for p in CODE}:
        raise InvalidArtifact('C summary implementation binding changed')
    phases=stored['phase_sources']
    if len(phases)!=(2 if stored['recovery_supplied'] else 1):raise InvalidArtifact('Exactly firstpass plus at most one bounded recovery required')
    first=full.checked(phases[0]['manifest']).parent
    recovery=full.checked(phases[1]['manifest']).parent if stored['recovery_supplied'] else None
    if len(phases)!=(2 if recovery else 1):raise InvalidArtifact('Exactly firstpass plus at most one bounded recovery required')
    audit=stored['pipeline_accounting'].get('derived_global_accounting_audit')
    fresh=summarize(first,recovery,full.checked(audit) if audit else None)
    # Compare every scientific field, source pointer, denominator and failure /
    # recovery flag. Only generation time and the outer storage seal may differ.
    if set(stored)-{'created_utc','content_sha256'}!=set(fresh)-{'created_utc'}:
        raise InvalidArtifact('Stored C summary field/lineage population differs')
    for key in fresh:
        if key=='created_utc':continue
        if stored[key]!=fresh[key]:raise InvalidArtifact('Stored C summary does not reproduce immutable raw sources: '+key)
    return fresh


def history(lock):
    baseline,source=full.replay_baseline(lock)
    if old.plain.sha(DATED)!=DATED_SHA:raise InvalidArtifact('Pinned historical dated summary changed')
    dated=old.plain.verify(DATED)
    module=importlib.import_module('tools.run_dated_judge_uniform_recovery_v1')
    replay=module.summarize(dated['run_dir'])
    if replay['slots_by_view']!=dated['slots_by_view']:raise InvalidArtifact('Historical dated raw replay differs')
    dated_rows=[r for r in replay['slots_by_view'][full.FINAL] if r['method']=='dated_merged']
    full.validate_slots(dated_rows,'dated_merged',full.canonical(lock))
    return {'baseline_sdk20':baseline['baseline_sdk20']['rows'],'dated_merged':dated_rows}
    # Source references are assembled by compare; no historical views rewritten.


def compare(c1_path,c2_path,*,bootstrap_samples=2000):
    if type(bootstrap_samples)is not int or bootstrap_samples<1:raise ValueError('Positive bootstrap sample count required')
    lock=locktool.verify_lock();expected=full.canonical(lock)
    c1,c2=load_report(c1_path),load_report(c2_path)
    if (c1['arm'],c2['arm'])!=ARMS:raise InvalidArtifact('Expected C1 then C2 declared summaries')
    for r in (c1,c2):validate_report(r,expected)
    first=[old.plain.read(r['phase_sources'][0]['manifest']['path']) for r in (c1,c2)]
    for key in ('plain_text_config','evaluation_protocol','dataset_sha256','official_metric_provenance','reference_dates'):
        if first[0][key]!=first[1][key]:raise InvalidArtifact('C1/C2 evaluation protocol differs: '+key)
    histories=history(lock)
    pair=old.scoped(full.comparison.pair,N=N)
    views={}
    for view in VIEWS:
        rows={r['arm']:r['slots_by_view'][view] for r in (c1,c2)}
        pairs={'boundary_neighbor_minus_boundary_multiscale':pair(rows[ARMS[0]],rows[ARMS[1]],bootstrap_samples=bootstrap_samples)}
        pairs.update({arm+'_minus_'+base:pair(hist,rows[arm],bootstrap_samples=bootstrap_samples)
                      for arm in ARMS for base,hist in histories.items()})
        context={'boundary_neighbor_minus_boundary_multiscale':{
            'left_method':ARMS[0],'right_method':ARMS[1],'left_view':view,'right_view':view,
            'phase_matched':True,'historical_descriptive_only':False}}
        context.update({arm+'_minus_'+base:{'left_method':base,'right_method':arm,
            'left_view':full.FINAL,'right_view':view,'phase_matched':False,
            'historical_descriptive_only':True} for arm in ARMS for base in histories})
        views[view]={'arms':{r['arm']:r['views'][view]['arms'][r['arm']] for r in (c1,c2)},'paired':pairs,
                     'pair_context':context,'historical_reference_view':full.FINAL,
                     'valid_denominator_note':'Each metric independently intersects valid IDs; zero-imputed deltas are sensitivity only.'}
    hist_aggregate=old.scoped(old.aggregate,ARMS=tuple(histories),DENOMINATOR=N)
    return {'schema':COMPARISON_SCHEMA,'created_utc':now(),'planned_per_method':N,
        'full1540_lock':old.plain.pointer(locktool.LOCK),'primary_view':'compatible_firstpass',
        'views':views,'historical_locked_final':{arm:hist_aggregate(rows,arm)['arms'][arm] for arm,rows in histories.items()},
        'sources':{'c1':old.plain.pointer(c1_path),'c2':old.plain.pointer(c2_path),
                   'historical_N0':lock['baseline']['summary'],'historical_dated':old.plain.pointer(DATED)},
        'recovery':{r['arm']:{'status':r['recovery_status'],'supplied':r['recovery_supplied'],
                     'uniform_policy_closed':r['uniform_recovery_policy_closed'],
                     'final_all_slots_valid':r['final_all_slots_valid'],
                     'validity_by_view':r['validity_by_view']} for r in (c1,c2)},
        'failures_by_view':{r['arm']:r['failures_by_view'] for r in (c1,c2)},
        'declared_failure_artifacts_by_phase':{r['arm']:r['declared_failure_artifacts_by_phase'] for r in (c1,c2)},
        'audit_errors':{r['arm']:r['audit_errors'] for r in (c1,c2)},
        'usage':{r['arm']:{'QA_and_retrieval':r['current_qa_plus_retrieval_usage'],
                'usage_estimator_assumptions':r['usage_estimator_assumptions'],
                'upstream':r['pipeline_accounting']} for r in (c1,c2)},
        'summary_code_sha256':{p:old.plain.sha(ROOT/p) for p in CODE},'model_calls_in_summary':0,'native_clients':0,
        'limitations':['C uses jev_mem.global_segment_grouping.CRITERION and the unchanged runs/global_segment_grouping_v1 parameters, not the later refinement identity criterion.',
            'C1 versus historical dated_merged compares the boundary-method module plus independent regeneration; C2 differs from C1 only in reserve neighbour policy.',
            'C1/C2 initial JEV caches were not shared: identical initial prompts may score differently, followed by independently regenerated Writer/QA. Differences are realized pipeline outcomes, not pure boundary accuracy.',
            'Historical dated comparison includes historical run/recovery factors; N0 also differs in dated input packaging/grouping. No single-cause or same-cold-budget claim.',
            'Historical N0 remains54.6095082528163 F1/66.03896103896105 Judge; its locked final reference is not overwritten by C firstpass/recovery.',
            'Three C views and independent Reader/Judge denominators remain visible; partial valid results are not a complete1540 observed score.',
            'Known ten-conversation cohort and exploratory cluster bootstrap do not establish blind generalization or causality.',
            'Unknown usage/fees remain null; no repeated Reader/Judge, correctness search or posthoc oracle selection.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    one=sub.add_parser('arm');one.add_argument('--firstpass',type=Path,required=True);one.add_argument('--recovery',type=Path)
    one.add_argument('--global-accounting-audit',type=Path,help='Existing derived raw/decision and ledger audit; binds its current source SHA, never repairs logs')
    both=sub.add_parser('compare');both.add_argument('--c1',type=Path,required=True);both.add_argument('--c2',type=Path,required=True)
    both.add_argument('--bootstrap-samples',type=int,default=2000)
    for parser in (one,both):parser.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New summary namespace required')
    with forbid_network():
        result=summarize(a.firstpass,a.recovery,a.global_accounting_audit) if a.action=='arm' else compare(a.c1,a.c2,bootstrap_samples=a.bootstrap_samples)
    save_new(a.output,result)
    print(json.dumps({'output':str(a.output),'schema':result['schema'],'model_calls':0}))


if __name__=='__main__':main()
