"""Five-arm R11 report from audited per-slot raw-replay summaries. No API."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_sdk20_judge_retest_v2 import read,sha,bound,pointer,verify
from tools.run_sdk20_modules import save_new
from tools.run_source_set_qa_r11 import request_signature

ARMS=('N0','cheap','local','g','p')
METRICS=('official_locomo_f1','custom_token_f1','judge_correct')

def stats(rows):
    out={'n':len(rows)}
    for metric in METRICS:
        vals=[r['scores'][metric] for r in rows if r['scores'][metric] is not None]
        out[metric]={'valid':len(vals),'missing':len(rows)-len(vals),
            'mean_percent_valid':100*sum(vals)/len(vals) if vals else None,
            'zero_imputed_fixed_percent':100*sum(vals)/len(rows) if rows else None}
    return out

def compare(left,right):
    transitions=defaultdict(list)
    for q,a in left.items():
        x,y=a['scores']['judge_correct'],right[q]['scores']['judge_correct']
        transitions['missing' if x is None or y is None else f'{int(x)}_to_{int(y)}'].append(q)
    deltas={}
    for metric in METRICS:
        pairs=[(r['scores'][metric],right[q]['scores'][metric]) for q,r in left.items()
            if r['scores'][metric] is not None and right[q]['scores'][metric] is not None]
        deltas[metric]={'paired_n':len(pairs),'delta_pp':100*sum(b-a for a,b in pairs)/len(pairs) if pairs else None,
            'zero_imputed_fixed_delta_pp':100*sum((right[q]['scores'][metric] or 0)-(r['scores'][metric] or 0) for q,r in left.items())/len(left) if left else None,
            'zero_imputed_is_sensitivity_not_observed_missing_score':True}
    return {'metrics':deltas,'judge_transitions':{k:{'count':len(transitions[k]),'queries':transitions[k]} for k in ('0_to_0','0_to_1','1_to_0','1_to_1','missing')}}

def make(new_summary,output,old_summary):
    new=verify(new_summary);old=verify(old_summary)
    for name,summary in [('R11',new),('R05',old)]:
        if summary.get('audit_status')!='no_invalid_artifacts' or summary.get('planned_per_arm')!=1540:
            raise ValueError(name+' must pass complete offline raw audit')
        for path,h in summary['input_sha256'].items():
            if not Path(path).is_file() or sha(path)!=h: raise ValueError(name+' audited input changed: '+path)
    view='final_with_uniform_technical_recovery'
    arms={a:{} for a in ARMS}
    for r in old['slots_by_view'][view]:
        a={'baseline':'N0','cheap':'cheap','jev':'local'}[r['method']];arms[a][r['query_id']]=r
    for r in new['slots_by_view'][view]:
        if r['method']=='baseline':
            orig=arms['N0'][r['query_id']]
            if r['scores']!=orig['scores'] or r['reader_answer']!=orig['reader_answer']:
                raise ValueError('Inherited N0 result changed')
        else: arms[r['method']][r['query_id']]=r
    cohort=set(arms['N0'])
    if len(cohort)!=1540 or any(set(rows)!=cohort for rows in arms.values()): raise ValueError('Five1540 shared cohorts required')
    for q in cohort:
        for a in ARMS:
            if any(arms[a][q][k]!=arms['N0'][q][k] for k in ('question','reference','category','reference_date','conversation')):
                raise ValueError('Question identity changed')
    oldsource=bound(old['source_manifest']);newsource=bound(new['source_manifest'])
    jobs={('N0' if j['method']=='baseline' else 'local' if j['method']=='jev' else j['method'],j['query_id']):j for j in oldsource['jobs']}
    jobs.update({(j['method'],j['query_id']):j for j in newsource['jobs'] if j['method']!='baseline'})
    pairs=[('N0','cheap'),('cheap','local'),('local','g'),('local','p'),('g','p'),('cheap','g'),('cheap','p'),('N0','g'),('N0','p')]
    comparisons={};subgroups={};cases=[]
    for left,right in pairs:
        key=right+'_minus_'+left;comparisons[key]=compare(arms[left],arms[right])
        if right in ('g','p'):
            same=[q for q in sorted(cohort) if request_signature(jobs[left,q],'reader')==request_signature(jobs[right,q],'reader')]
            subgroups[key]={'same_full_reader_signature_n':len(same),
                'same_input_flips':compare({q:arms[left][q] for q in same},{q:arms[right][q] for q in same}),
                'different_input_n':1540-len(same)}
    for a in ('g','p'):
        lost=set(comparisons[a+'_minus_local']['judge_transitions']['1_to_0']['queries'])
        for q in sorted(lost):
            oldrow,newrow=arms['local'][q],arms[a][q]
            proposal=bound(jobs[a,q]['lineage']['proposal'])
            cases.append({'arm':a,'query_id':q,'conversation':newrow['conversation'],'category':newrow['category'],
                'question':newrow['question'],'reference':newrow['reference'],
                'same_reader_signature':request_signature(jobs['local',q],'reader')==request_signature(jobs[a,q],'reader'),
                'original20':proposal['original_job']['selected_facts'],'candidates':proposal['locator']['candidates'],
                'old':{'answer':oldrow['reader_answer'],'scores':oldrow['scores'],'selection':bound(jobs['local',q]['selection'])},
                'new':{'answer':newrow['reader_answer'],'scores':newrow['scores'],'selection':bound(jobs[a,q]['selection'])}})
    bycategory={a:{str(c):stats([r for r in arms[a].values() if r['category']==c]) for c in (1,2,3,4)} for a in ARMS}
    byconv={a:{str(c):stats([r for r in arms[a].values() if r['conversation']==c]) for c in range(10)} for a in ARMS}
    result={'schema':'source-set-r11-five-arm-comparison-v1','sources':{'R11':pointer(new_summary),'R05':pointer(old_summary)},
        'metrics':{a:stats(list(rows.values())) for a,rows in arms.items()},'comparisons':comparisons,
        'same_input_subgroups':subgroups,'by_category':bycategory,'by_conversation':byconv,
        'all_new_regression_case_count':len(cases),'cases':cases,
        'limits':['Single exposed1540 cohort; no independent human accuracy or heldout generalization.',
            'R05 controls are historical; unchanged-input flips expose Reader/Judge sampling variance.',
            'G/P are variants of one source-append module; P-G also changes predicate/primitive.',
            'No persistent writer improvement was tested.']}
    save_new(output,result)
    return {'output':str(output),'metrics':result['metrics'],'regression_case_count':len(cases),'model_calls':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--old-summary',type=Path,default=ROOT/'research/cleanup_2026-10-05/r05_postcleanup_raw_audit_v1.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(make(a.summary,a.output,a.old_summary),ensure_ascii=False))
if __name__=='__main__': main()
