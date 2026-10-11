"""Pure post-selection R04 inheritance checks. No clients or model operations."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from tools.run_sdk20_modules import digest

ARMS=('baseline','cheap','jev')
ROUTES={'baseline':'baseline','jev_r03':'jev'}
METRICS=('official_locomo_f1','custom_token_f1','judge_correct')

def require(condition,message):
    if not condition: raise ValueError(message)

def candidate_index(rows, *, population=1540):
    index={}
    for row in rows:
        if row['method'] not in ROUTES.values(): continue
        key=(row['method'],row['query_id'])
        require(key not in index,'Duplicate original candidate slot')
        require(row.get('reader_status')=='complete' and row.get('judge_status')=='complete'
                and row.get('qa_status')=='complete' and not row.get('errors'),
                'Original candidate is not completely raw-audited')
        require(isinstance(row.get('reader_answer'),str) and bool(row['reader_answer']),
                'Missing original actual Reader answer')
        require(row.get('judge_parse_mode') in ('strict','compat'), 'Original Judge parse mode missing')
        require(all(row['scores'].get(k) is not None for k in METRICS),'Original score missing')
        index[key]=row
    cohorts=[{q for m,q in index if m==method} for method in ROUTES.values()]
    require(all(len(c)==population for c in cohorts) and cohorts[0]==cohorts[1],
            'Complete matching original candidate populations required')
    for qid in cohorts[0]:
        a,b=index[('baseline',qid)],index[('jev',qid)]
        require(all(a.get(k)==b.get(k) for k in ('question','reference','reference_date','category','conversation')),
                'Original candidates have different question/gold/date/category/conversation')
    return index

def verify_candidate(request,route,index,qa_jobs,source_jobs):
    """Bind actual raw-audited answer to both original job identities."""
    qid=request['query_id'];method=ROUTES[route]
    row=index[(method,qid)];candidate=request['candidates'][route]
    require(candidate['answer']==row['reader_answer'],'Selected candidate answer differs from actual Reader')
    require(candidate['reader']==row['reader_artifact'],'Selected original Reader pointer differs')
    require(candidate['source_job_sha256']==digest(qa_jobs[(method,qid)]),'Original QA job hash differs')
    require(candidate['selected_source_job_sha256']==digest(source_jobs[(method,qid)]),
            'Original selected-source job hash differs')
    require(request['conversation']==row['conversation'] and request['state']['question']==row['question']
            and request['state']['reference_date']==row['reference_date'],'Choice query/date differs from source')
    # The offline request record may hash source identity; neither gold nor labels
    # can be a Choice-state key. Memory text is data and is not substring-scanned.
    require(set(request['state'])=={'question','reference_date','memories','candidate_first','candidate_second'},
            'Unexpected state field; gold/labels must never enter Choice input')
    require(request['state']['candidate_first']==request['candidates'][request['presentation']['first']]['answer']
            and request['state']['candidate_second']==request['candidates'][request['presentation']['second']]['answer'],
            'Blind presentation and actual candidate strings differ')
    return row

def inherit_jobs(jobs,requests,index,qa_jobs,source_jobs, *, population=1540):
    """A selected original response is referenced, never forged as a new draw."""
    require(Counter(j['method'] for j in jobs)=={a:population for a in ARMS},'Three complete R04 arms required')
    seen=set();rows=[]
    for job in jobs:
        qid=job['query_id'];key=(job['method'],qid)
        require(key not in seen,'Duplicate selected R04 slot');seen.add(key)
        require(job['route'] in ROUTES,'Unknown selected original route')
        request=requests[qid]
        for route in ROUTES: verify_candidate(request,route,index,qa_jobs,source_jobs)
        original=index[(ROUTES[job['route']],qid)]
        require(job['answer']==original['reader_answer'] and job['reader']==original['reader_artifact'],
                'Persisted selection altered original answer or Reader pointer')
        require(job['conversation']==original['conversation'],'Selected conversation differs')
        if job['method']=='baseline':require(job['route']=='baseline','Baseline anchor changed route')
        row=deepcopy(original)
        row.update(job_id=job.get('job_id',job['method']+'/'+qid),method=job['method'],
            original_route=job['route'],original_qa_method=original['method'],
            selected_original_reader=original['reader_artifact'],selected_original_judge=original['judge_artifact'],
            answer_selection=job['decision'],generation_status='referenced_existing_actual_reader',
            reader_generated_from_union=False,new_reader_calls=0,new_judge_calls=0,
            original_scores_replayed=True)
        rows.append(row)
    cohorts=[{q for m,q in seen if m==arm} for arm in ARMS]
    require(all(c==set(requests) for c in cohorts),'R04 selected cohort differs from prepared request cohort')
    return rows

def stopping(rows, *, population=1540,baseline_correct=1153,minimum_correct=1200):
    counts={a:sum(r['scores']['judge_correct']==1 for r in rows if r['method']==a) for a in ARMS}
    missing={a:sum(r['scores']['judge_correct'] is None for r in rows if r['method']==a) for a in ARMS}
    eligible=(sum(r['method']=='baseline' for r in rows)==population
              and counts['baseline']==baseline_correct and missing['baseline']==0)
    return {'denominator':population,'baseline_correct':baseline_correct,
        'audited_baseline_correct':counts['baseline'],'minimum_strict_gt_3pp':minimum_correct,
        'threshold_eligible':eligible,'correct':counts,'missing':missing,
        'strict_gt_3pp_fixed_denominator':{a:eligible and counts[a]>=minimum_correct for a in ARMS[1:]},
        'component_family_count':1,'successful_independent_components_counted_here':0,
        'answer_selection_is_composite_with_r03':True,
        'cannot_count_as_second_independent_component':True}

def transitions(left,right):
    a={r['query_id']:r for r in left};b={r['query_id']:r for r in right}
    require(len(a)==len(left) and len(b)==len(right) and set(a)==set(b),'Paired cohorts differ')
    groups={k:[] for k in ('0_to_0','0_to_1','1_to_0','1_to_1','missing_either')}
    f1={k:[] for k in ('improved','declined','equal','missing')};answers=[]
    for qid in sorted(a):
        x,y=a[qid]['scores']['judge_correct'],b[qid]['scores']['judge_correct']
        groups['missing_either' if x is None or y is None else f'{int(x)}_to_{int(y)}'].append(qid)
        x,y=a[qid]['scores']['official_locomo_f1'],b[qid]['scores']['official_locomo_f1']
        f1['missing' if x is None or y is None else 'improved' if y>x else 'declined' if y<x else 'equal'].append(qid)
        if a[qid]['reader_answer']!=b[qid]['reader_answer']:answers.append(qid)
    return {'denominator':len(a),'judge_transitions':{k:{'count':len(v),'query_ids':v} for k,v in groups.items()},
            'f1_directions':{k:{'count':len(v),'query_ids':v} for k,v in f1.items()},
            'fixed_denominator_metric_deltas':{m:sum((b[q]['scores'][m] or 0)-(a[q]['scores'][m] or 0) for q in a)/len(a)
                                              for m in METRICS},
            'answer_text_changes':{'count':len(answers),'query_ids':answers}}

def descriptive_strata(rows,requests,decisions,abstention_detector):
    """Descriptions after selection; no strata is a selection feature."""
    totals=Counter();strata={};positions={};row_index={(r['method'],r['query_id']):r for r in rows}
    for qid,rec in requests.items():
        aa=abstention_detector(rec['candidates']['baseline']['answer'])
        bb=abstention_detector(rec['candidates']['jev_r03']['answer'])
        label=f'baseline_abstention_{aa}_r03_abstention_{bb}'
        position=rec['presentation']['first'];d=decisions[qid]
        for scope,key in ((strata,label),(positions,position)):
            group=scope.setdefault(key,{'queries':0,'jev_routes':Counter(),'status':Counter(),'reason':Counter(),
                'correct_by_arm':Counter(),'official_f1_sum_by_arm':Counter()})
            group['queries']+=1;group['jev_routes'][d['selected_route']]+=1
            group['status'][d['status']]+=1;group['reason'][d['selection_reason']]+=1
            for arm in ARMS:
                row=row_index[(arm,qid)]
                group['correct_by_arm'][arm]+=row['scores']['judge_correct']==1
                group['official_f1_sum_by_arm'][arm]+=row['scores']['official_locomo_f1'] or 0
        totals['identical_answers']+=rec['identical'];totals['nonidentical_answers']+=not rec['identical']
        totals['dispatch_eligible']+=rec['dispatch_eligible']
        totals['jev_route_'+d['selected_route']]+=1;totals['score_status_'+d['status']]+=1
        totals['selection_reason_'+d['selection_reason']]+=1
    for scope in (strata,positions):
        for group in scope.values():
            for key in ('jev_routes','status','reason','correct_by_arm','official_f1_sum_by_arm'):group[key]=dict(group[key])
    return {'choice_counts':dict(totals),'cheap_abstention_strata':strata,
            'hash_position_strata':positions,'strata_not_used_for_selection':True}
