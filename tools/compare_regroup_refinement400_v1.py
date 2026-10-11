"""Compare two audited refinement arms with the frozen dated_merged400 reference.

Primary comparisons always use compatible_firstpass. Technical recovery is a
separate within-arm supplement. No API, new scoring, cohort selection or repair.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import summarize_regroup_refinement_qa_v1 as summary
from tools import regroup_refinement_qa_sources_v1 as sources
from tools.run_sdk20_modules import save_new, now, digest
from tools.summarize_sdk20_modules import InvalidArtifact
from jev_mem.local_bm25_context_v2 import forbid_network

REFERENCE=ROOT/'research/regroup_refinement_execution_2026-09-30/reused_reference400_v1.json'
REFERENCE_SHA='7f99353c9a0b86f4d811b569a9b85f90e5397061f787e6085c5feae2a71db25d'
BASELINE='dated_merged'
ARMS=('source_local_context','jev_refined')
METRICS=summary.base.METRICS
VIEW='compatible_firstpass'
N=400
CODE=('tools/compare_regroup_refinement400_v1.py','tools/summarize_regroup_refinement_qa_v1.py',
      'tools/summarize_mem0_regroup_plain_qa_v1.py','tools/summarize_pilot400.py')


def checked(ref):
    current=summary.plain.pointer(ref['path'])
    if any(current[k]!=ref[k] for k in ('path','sha256')):
        raise InvalidArtifact('Bound comparison source drift')
    return Path(ref['path'])


def cohort():
    rows,dates=sources.cohort()
    flat=[q for ci in range(10) for q in rows[str(ci)]]
    if len(flat)!=N or len({q['id'] for q in flat})!=N:
        raise InvalidArtifact('Original fixed400 cohort required')
    return {q['id']:q for q in flat},dates


def slots(rows, method, expected):
    if len(rows)!=N or len({r['query_id'] for r in rows})!=N or {r['query_id'] for r in rows}!=set(expected):
        raise InvalidArtifact('Every original400 slot, including failures, is required')
    for row in rows:
        q=expected[row['query_id']]
        if row['method']!=method or row['conversation']!=q['conversation'] or row['category']!=q['category']:
            raise InvalidArtifact('Per-question method/conversation/category identity differs')
        if set(row['scores'])!=set(METRICS):
            raise InvalidArtifact('Metric identity differs')
    # Reuse the frozen single-arm function; admit the historical method only in
    # this function's private globals, never mutate the summary module's ARMS.
    aggregate=summary.scoped(summary.aggregate,ARMS=(BASELINE,*ARMS))
    return aggregate(rows,method)['arms'][method]


def load_reference(path, expected, dates):
    path=Path(path).resolve()
    if summary.plain.sha(path)!=REFERENCE_SHA:
        raise InvalidArtifact('Expected the exact frozen reused_reference400 artifact')
    ref=summary.plain.read(path)
    if (ref['schema']!='refinement-fixed400-reused-reference-v1' or ref['view']!=VIEW
            or ref['planned_per_arm']!=N or ref['new_model_calls']!=0):
        raise InvalidArtifact('Reference must be compatible firstpass, never mixed final recovery')
    for bound in ref['sources']:checked(bound)
    if ref['sources'][0]!=summary.plain.pointer(sources.COHORT):
        raise InvalidArtifact('Reference uses another cohort manifest')
    origin=summary.plain.verify(ref['sources'][1]['path'])
    if origin['schema']!='mem0-plain-judge-label-compat-overlay-v1' or origin['reader_and_f1_unchanged'] is not True:
        raise InvalidArtifact('Reference does not derive from the original compatible overlay')
    rows=[r for r in ref['slots'] if r['method']==BASELINE]
    byid={r['query_id']:r for r in origin['slots'] if r['method']==BASELINE}
    if any(r!=byid.get(r['query_id']) for r in rows):
        raise InvalidArtifact('Reference rows differ from their original firstpass source')
    manifest=summary.plain.read(checked(origin['sources']['source_manifest']))
    if (manifest['protocol_sha256']!=digest({k:v for k,v in manifest.items() if k!='protocol_sha256'})
            or manifest['reference_dates']!=dates):
        raise InvalidArtifact('Historical source/date protocol differs')
    checked(manifest['plain_text_config'])
    return {'rows':rows,'aggregate':slots(rows,BASELINE,expected),'manifest':manifest,
            'source':summary.plain.pointer(path),'origin_source':ref['sources'][1]}


def load_arm(path, arm, expected, dates, reference):
    path=Path(path).resolve();report=summary.plain.verify(path)
    if (report['schema']!='regroup-refinement-single-arm400-qa-summary-v1' or report['arm']!=arm
            or report['planned_per_arm']!=N or report['planned_total']!=N
            or report['model_calls_in_summary']!=0 or report['native_clients']!=0
            or report['paired_comparison'] is not None):
        raise InvalidArtifact('Expected a frozen audited single-arm400 summary')
    if report['audit_status'] not in ('no_invalid_artifacts','invalid_artifacts_retained_as_missing'):
        raise InvalidArtifact('Unknown source audit status')
    if (report['audit_status']=='no_invalid_artifacts') != (not report['audit_errors']):
        raise InvalidArtifact('Audit status contradicts recorded audit errors')
    if report['summary_code_sha256']!={p:summary.plain.sha(ROOT/p) for p in summary.CODE}:
        raise InvalidArtifact('Summary implementation provenance changed')
    phases=report['phase_sources']
    if len(phases)!=(2 if report['recovery_supplied'] else 1):
        raise InvalidArtifact('Only firstpass and at most one recovery are supported')
    for phase in phases:
        checked(phase['manifest']);checked(phase['outcome'])
        for bound in phase['ledgers']:checked(bound)
    m=summary.plain.read(phases[0]['manifest']['path'])
    if (m['phase_kind']!='firstpass' or m['arm']!=arm
            or m['protocol_sha256']!=digest({k:v for k,v in m.items() if k!='protocol_sha256'})):
        raise InvalidArtifact('Firstpass manifest identity differs')
    for key in ('plain_text_config','evaluation_protocol','dataset_sha256','official_metric_provenance'):
        if m[key]!=reference['manifest'][key]:raise InvalidArtifact('Historical and new evaluation protocol differ: '+key)
    if m['reference_dates']!=dates:raise InvalidArtifact('Reference dates differ')
    jobs=summary.plain.verify(checked(report['source_manifest']))
    if (Path(report['source_manifest']['path']).resolve()!=Path(phases[0]['manifest']['path']).parent/'source_jobs.json'
            or jobs['protocol_sha256']!=m['protocol_sha256'] or jobs['arm']!=arm
            or jobs['official_metric_provenance']!=m['official_metric_provenance']):
        raise InvalidArtifact('Question source is not the original frozen firstpass input')
    if len(jobs['jobs'])!=N or {j['query_id'] for j in jobs['jobs']}!=set(expected):
        raise InvalidArtifact('Frozen question population differs')
    for j in jobs['jobs']:
        q=expected[j['query_id']]
        if (j['method']!=arm or j['conversation']!=q['conversation'] or j['category']!=q['category']
                or j['question']!=q['question'] or j['reference']!=q['reference']
                or j['reference_date']!=dates[str(q['conversation'])]):
            raise InvalidArtifact('Question, answer or date identity differs')
    if report['recovery_supplied']:
        recovery=summary.plain.read(phases[1]['manifest']['path'])
        if (recovery['phase_kind']!='technical_recovery' or recovery['arm']!=arm
                or recovery['parent']['manifest']!=phases[0]['manifest']
                or recovery['parent']['outcome']!=phases[0]['outcome']):
            raise InvalidArtifact('Recovery source differs from the firstpass')
    views=report['slots_by_view']
    if set(views)!={'strict_firstpass',VIEW,'final'}:
        raise InvalidArtifact('Primary and recovered views must stay distinct')
    aggregates={name:slots(rows,arm,expected) for name,rows in views.items()}
    for name,value in aggregates.items():
        if report['views'][name]['arms']!={arm:value} or report['views'][name]['paired'] is not None:
            raise InvalidArtifact('Stored aggregate does not reproduce its complete400 slots')
    if not report['recovery_supplied'] and views['final']!=views[VIEW]:
        raise InvalidArtifact('Final without a recovery changed the compatible firstpass')
    return {'report':report,'views':views,'aggregates':aggregates,'source':summary.plain.pointer(path)}


def pair(left_rows,right_rows,*,bootstrap_samples):
    output={}
    for metric in METRICS:
        left={r['query_id']:{'query_id':r['query_id'],'conversation':r['conversation'],metric:r['scores'][metric]}
              for r in left_rows if r['scores'][metric] is not None}
        right={r['query_id']:{'query_id':r['query_id'],'conversation':r['conversation'],metric:r['scores'][metric]}
               for r in right_rows if r['scores'][metric] is not None}
        result=summary.base.existing.paired_metric(metric,left,right,samples=bootstrap_samples,denominator=N)
        deltas=[right[q][metric]-left[q][metric] for q in result['common_query_ids']]
        result['wins_ties_losses']={'wins':sum(d>0 for d in deltas),'ties':sum(d==0 for d in deltas),'losses':sum(d<0 for d in deltas)}
        result['independent_valid_denominators']={'baseline':len(left),'other':len(right),'planned_each':N}
        result['missing_as_zero_delta_sensitivity']={'other_minus_baseline':
            (sum(r[metric] for r in right.values())-sum(r[metric] for r in left.values()))/N,
            'is_imputation_not_observed_score':True}
        output[metric]=result
    return output


def compare(context_path,jev_path,reference_path=REFERENCE,*,bootstrap_samples=2000):
    if type(bootstrap_samples)is not int or bootstrap_samples<1:
        raise ValueError('Positive bootstrap count required')
    expected,dates=cohort();reference=load_reference(reference_path,expected,dates)
    arms={arm:load_arm(path,arm,expected,dates,reference) for arm,path in zip(ARMS,(context_path,jev_path))}
    comparisons={arm+'_minus_'+BASELINE:pair(reference['rows'],a['views'][VIEW],bootstrap_samples=bootstrap_samples)
                 for arm,a in arms.items()}
    supplement={arm:{'recovery_supplied':a['report']['recovery_supplied'],
        'final_metrics':a['aggregates']['final'],
        'final_minus_own_compatible_firstpass':pair(a['views'][VIEW],a['views']['final'],bootstrap_samples=bootstrap_samples)
            if a['report']['recovery_supplied'] else None,
        'comparison_with_historical_reference':None,
        'reason':'New technical recoveries are not mixed into the reference firstpass primary comparison'} for arm,a in arms.items()}
    return {'schema':'regroup-refinement400-comparison-v1','created_utc':now(),'primary_view':VIEW,
        'planned_per_method':N,'metric_scale':'unit interval; display differences in percentage points',
        'question_identity_checks':{'same400_ids':True,'same_conversation_and_category':True,
            'new_questions_references_and_dates_match_fixed_dataset':True,'same_reader_judge_config_and_official_metric':True},
        'primary_table':{BASELINE:reference['aggregate'],**{arm:a['aggregates'][VIEW] for arm,a in arms.items()}},
        'primary_comparisons':comparisons,'technical_recovery_supplement':supplement,
        'sources':{'reference':reference['source'],'historical_compatible_overlay':reference['origin_source'],
                   'cohort':summary.plain.pointer(sources.COHORT),'arms':{arm:a['source'] for arm,a in arms.items()}},
        'source_audit_status':{arm:{'status':a['report']['audit_status'],'errors':a['report']['audit_errors']} for arm,a in arms.items()},
        'cost_scope':{arm:{'qa':a['report']['qa_usage'],'retrieval':a['report']['retrieval_usage'],
                           'writer':a['report']['writer_accounting']} for arm,a in arms.items()},
        'code_sha256':{p:summary.plain.sha(ROOT/p) for p in CODE},'new_model_calls':0,'new_scores_generated':0,
        'limitations':['Known fixed400 from ten already exposed conversations, including development data; not blind validation.',
            'Primary effects compare compatible firstpass on identical slots; each metric has its own valid and paired subsets.',
            'All failures retain the planned400 denominator; zero imputation is separately labeled sensitivity, never observed performance.',
            'Conversation-cluster bootstrap follows the original eligibility rule and is exploratory, not a causal or significance claim.',
            'Technical recovery results are separate within-arm follow-up; no unequal-retry historical primary comparison is made.',
            'This comparison validates audited summary lineage and scores; it does not rerun raw model replay or judge predictions.',
            'Each writer variant is a whole module change with its recorded source/index limitations; paired QA outcomes alone do not establish mechanisms.',
            'Cost fields retain separate stage scopes and unknowns; no cold-start total or new reference billing is inferred.']}


def markdown(report):
    text=['# 固定400题：写入优化比较','',
          '主表统一使用 compatible_firstpass。F1与Judge各用独立有效分母；括号显示有效题数/固定400。失败保留在分母，缺失计零仅作为JSON中的敏感度分析。','',
          '| 方法 | 官方F1 % | custom F1 % | Judge % |','|---|---:|---:|---:|']
    for arm,a in report['primary_table'].items():
        cells=[]
        for metric in METRICS:
            v=a['metrics'][metric];mean=v['complete_only_mean']
            cells.append(('缺失' if mean is None else f'{mean*100:.3f}')+f" ({v['valid_denominator']}/400)")
        text.append('| '+arm+' | '+' | '.join(cells)+' |')
    text+=['','| 配对差值（百分点） | 官方F1 | custom F1 | Judge |','|---|---:|---:|---:|']
    for name,metrics in report['primary_comparisons'].items():
        cells=[]
        for metric in METRICS:
            p=metrics[metric];v=p['metrics'][metric]['other_minus_baseline']
            cells.append(('缺失' if v is None else f'{v*100:+.3f}')+f" (配对{p['common_success_count']}/400)")
        text.append('| '+name+' | '+' | '.join(cells)+' |')
    text+=['','技术恢复单独记录在 technical_recovery_supplement；未与旧reference首遍混为主差值。配对区间、逐题ID、胜/平/负和缺失敏感度见JSON。',
           '','这是已接触过的10个conversation固定样本，不能称盲测或仅凭区间作因果、显著性结论。该步骤复用已审模型结果，没有新增模型调用或评分。','']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context-summary',type=Path,required=True);p.add_argument('--jev-summary',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=REFERENCE);p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():raise FileExistsError('Fresh comparison output required')
    with forbid_network():value=compare(args.context_summary,args.jev_summary,args.reference,bootstrap_samples=args.bootstrap_samples)
    save_new(args.output,value)
    with args.output.with_suffix('.md').open('x',encoding='utf-8') as f:f.write(markdown(value))
    print(json.dumps({'output':str(args.output),'primary_view':VIEW,'new_model_calls':0}))


if __name__=='__main__':main()
