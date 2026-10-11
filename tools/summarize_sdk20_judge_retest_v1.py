"""Offline fixed-answer Judge comparison; reuse existing metrics and request replay."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_sdk20_judge_retest_v2 as runner
from tools import prepare_sdk20_judge_retest_sources_v1 as sources
from tools import summarize_mem0_regroup_plain_qa_v1 as existing
from tools.summarize_sdk20_modules import fingerprint, DeclaredTransportFailure
from tools.summarize_mem0_sparse1540 import measure_usage
from tools.run_sdk20_modules import save_new, now, digest
from jev_mem.evaluation_plain_text_v1 import judge_answer, InvalidJudgeLabel
from jev_mem.judge_terminal_label_compat_v1 import derive_label, eligible_raw_response

ARMS=('baseline_sdk20','fact_types_sdk20')

def phase(path):
    m,out=runner.terminal(path);path=Path(path).resolve()
    jobs={j['job_id']:j for j in m['jobs']};ledger=[];indexed=defaultdict(list);ledger_refs=[]
    for ci in range(10):
        p=path/f'conv_{ci}/api/api_calls.jsonl'
        if not p.exists():continue
        ledger_refs.append(runner.pointer(p))
        for line in runner.shared_read(p).decode('utf-8-sig').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r.get('stage')!=existing.evaluator.JUDGE_STAGE or r.get('model')!=existing.evaluator.JUDGE_MODEL:
                raise ValueError('Unexpected non-Judge model flow')
            if r['context']['conversation']!=ci or r['context']['phase_id']!=m['phase_id']:
                raise ValueError('Ledger phase/conversation drift')
            ledger.append(r);indexed[r['request_hash'],r['stage'],fingerprint(r['context'])].append(r)
    records={}
    for r in out['results']:
        j=jobs[r['job_id']];ident=runner.identity(j,m)
        s=runner.verify(runner.bound(r['strict']) and r['strict']['path']) if r['strict'] else None
        c=runner.verify(runner.bound(r['compat']) and r['compat']['path']) if r['compat'] else None
        for row in (s,c):
            if row and any(row.get(k)!=v for k,v in ident.items()):raise ValueError('Result identity changed')
        if c and c['strict']!=r['strict']:raise ValueError('Compatibility pointer changed')
        context={'architecture':runner.SCHEMA,'conversation':j['conversation'],'phase':'judge_retest',
                 'phase_id':m['phase_id'],'method':j['method'],'query_id':j['query_id'],'job_id':j['job_id'],'component':'judge'}
        if j['status']=='blocked_original_reader':
            if not s or s['status']!='blocked_original_reader' or s['api_events'] or any(x['context']==context for x in ledger):
                raise ValueError('Blocked original Reader was dispatched')
        elif s:
            api=existing.PlainReplay(path/f"conv_{j['conversation']}/api",s['api_events'],context,indexed)
            result=error=None
            try:result=judge_answer(api,j['question'],j['reference'],j['answer'])
            except Exception as exc:error=exc
            api.assert_consumed()
            if s['status']=='complete':
                if error or result!=s['result']:raise ValueError('Strict success not reproduced')
            elif s['status']=='failed':
                kind=error.provider_error_type if isinstance(error,DeclaredTransportFailure) else type(error).__name__
                if error is None or kind!=s['error']['type']:raise ValueError('Strict failure not reproduced')
                if isinstance(error,InvalidJudgeLabel) and error.raw_response!=s['raw_invalid_label']:raise ValueError('Failed text changed')
            else:raise ValueError('Unknown strict state')
            if c and c.get('interpretation') is not None:
                text=result['raw'] if result else s['raw_invalid_label']
                expected=derive_label(text);proof=c['raw_evidence'];raw=runner.bound(proof['raw'])
                data=runner.shared_read(Path(proof['ledger']))[:proof['ledger_prefix_bytes']]
                import hashlib
                if hashlib.sha256(data).hexdigest()!=proof['ledger_prefix_sha256']:raise ValueError('Ledger proof changed')
                producer=json.loads(data.splitlines()[proof['physical_producer_line']-1])
                if producer!=proof['physical_producer'] or digest(producer)!=proof['physical_producer_sha256']:raise ValueError('Producer changed')
                if s['status']=='failed':
                    ok,reason=eligible_raw_response(raw['response'],text,http_status=producer['status'])
                    if not ok:expected={**expected,'status':'rejected','label':None,'correct':None,'rejection_reason':reason}
                if expected!=c['interpretation'] or expected['status']!=c['status']:raise ValueError('Compatibility text parsing changed')
        records[j['job_id']]={'job':j,'strict':s,'compat':c,'pointers':r}
    return {'manifest':m,'outcome':out,'records':records,'usage':measure_usage(ledger),'ledger_refs':ledger_refs,
            'ledger':ledger,'run':str(path)}

def strict_label(r):
    s=r and r.get('strict');return s['result']['correct'] if s and s['status']=='complete' else None

def compatible_label(r):
    if not r:return None
    return runner.resolve_pair(r['strict'],r['compat'])['correct']

def rows(jobs,labels,overrides=None,status_by_job=None):
    overrides=overrides or {};status_by_job=status_by_job or {};values=[]
    for j in jobs:
        f=overrides.get(j['job_id'],j)['F1'];label=labels.get(j['job_id'])
        if label is not None and type(label)is not bool:raise ValueError('Only boolean or null Judge allowed')
        ready=f['official_locomo_f1'] is not None
        reader_state='complete' if j['status']=='ready' else 'format_compatible' if ready else 'failed'
        judge_state='complete' if label is not None else status_by_job.get(j['job_id'],'failed' if ready else 'blocked_original_reader')
        values.append({'query_id':j['query_id'],'conversation':j['conversation'],'category':j['category'],'method':j['method'],
            'selection_status':'complete','reader_status':reader_state,
            'judge_status':judge_state,'qa_status':'complete' if label is not None else judge_state,
            'scores':{k:f[k] for k in ('official_locomo_f1','custom_token_f1')}|{'judge_correct':float(label) if label is not None else None},'errors':[]})
    return values

def aggregate(values):
    old=existing.aggregate
    f=FunctionType(old.__code__,{**old.__globals__,'ARMS':ARMS},old.__name__,old.__defaults__,old.__closure__)
    f.__kwdefaults__=dict(old.__kwdefaults__ or {})
    return f(values,denominator=1540,bootstrap_samples=2000)

def confusion(left,right):
    counts=Counter()
    for k in sorted(set(left)|set(right)):
        a,b=left.get(k),right.get(k)
        if a is None or b is None:counts['missing']+=1
        else:counts[('yes' if a else 'no')+'_to_'+('yes' if b else 'no')]+=1
    return dict(counts)

def summarize(firstpass,recovery=None,followup=None):
    first=phase(firstpass);src=sources.validate(first['manifest']['source_manifest']['path']);jobs=src['jobs']
    recovered=phase(recovery) if recovery else None;third=phase(followup) if followup else None
    if first['manifest']['phase_kind']!='firstpass':raise ValueError('First input must be the main firstpass')
    for child,parent in ((recovered,first),(third,recovered)):
        if child is None:continue
        if parent is None:raise ValueError('Supplement followup requires its recovery parent')
        binding=child['manifest']['parent']
        if (child['manifest']['source_manifest']!=first['manifest']['source_manifest']
            or binding['manifest']!=runner.pointer(Path(parent['run'])/'run_manifest.json')
            or binding['outcome']!=runner.pointer(Path(parent['run'])/'run_outcome.json')
            or Path(binding['run_dir']).resolve()!=Path(parent['run']).resolve()):
            raise ValueError('Recovery or followup belongs to another phase/source')
    stages=[x for x in (first,recovered,third) if x];labels={};overrides={};resolved={};supplement=[]
    for j in jobs:
        a=first['records'][j['job_id']];b=recovered and recovered['records'].get(j['job_id']);c=third and third['records'].get(j['job_id'])
        if c:
            if j['status']!='blocked_original_reader' or not b:raise ValueError('Only blocked-source first Judge failures get followup')
            final=runner.resolve_pair(b['strict'],b['compat'],c['strict'],c['compat'])
        else:final=runner.resolve_pair(a['strict'],a['compat'],b['strict'] if b else None,b['compat'] if b else None)
        if b and j['status']=='blocked_original_reader':overrides[j['job_id']]=b['job']
        labels[j['job_id']]=final['correct'];resolved[j['job_id']]=final
    main_keys={j['job_id'] for j in jobs}
    for stage in stages[1:]:
        for key,r in stage['records'].items():
            if key not in main_keys:supplement.append({'job_id':key,'job':r['job'],'result':runner.resolve_pair(r['strict'],r['compat']),'pointers':r['pointers']})
    old={j['job_id']:j['old_judge']['correct'] if j['old_judge'] else None for j in jobs}
    strict={k:strict_label(r) for k,r in first['records'].items()};compat={k:compatible_label(r) for k,r in first['records'].items()}
    same_answer_final={j['job_id']:labels[j['job_id']] if j['status']=='ready' else None for j in jobs}
    variants={'old_judge_fixed_original_answers':old,'new_strict_firstpass':strict,'new_compatible_firstpass':compat,
              'new_final_fixed_original_answers':same_answer_final,'new_final_with_reader_format_supplement':labels}
    missing_first={k:'missing' for k,r in first['records'].items() if r['strict'] is None}
    views={k:aggregate(rows(jobs,v,overrides if k.endswith('supplement') else None,
                           missing_first if k in ('new_strict_firstpass','new_compatible_firstpass') else None)) for k,v in variants.items()}
    old_view=views['old_judge_fixed_original_answers']
    for name,view in views.items():
        if name.endswith('supplement'):continue
        for arm in ARMS:
            for metric in ('official_locomo_f1','custom_token_f1'):
                if view['arms'][arm]['metrics'][metric]!=old_view['arms'][arm]['metrics'][metric]:
                    raise ValueError('Fixed-answer F1 changed under Judge-only evaluation')
        for metric in ('official_locomo_f1','custom_token_f1'):
            if view['paired_merged_minus_unmerged'][metric]!=old_view['paired_merged_minus_unmerged'][metric]:
                raise ValueError('Fixed-answer paired F1 changed under Judge-only evaluation')
    flips={name:{a:confusion({j['job_id']:old[j['job_id']] for j in jobs if j['method']==a},
                         {j['job_id']:values.get(j['job_id']) for j in jobs if j['method']==a}) for a in ARMS}
           for name,values in variants.items() if not name.startswith('old_')}
    method_pairs={name:confusion({j['query_id']:values.get(j['job_id']) for j in jobs if j['method']==ARMS[0]},
                                {j['query_id']:values.get(j['job_id']) for j in jobs if j['method']==ARMS[1]}) for name,values in variants.items()}
    retry_changes={a:confusion({j['job_id']:compat[j['job_id']] for j in jobs if j['method']==a and recovered and j['job_id'] in recovered['records']},
                              {j['job_id']:labels[j['job_id']] for j in jobs if j['method']==a and recovered and j['job_id'] in recovered['records']}) for a in ARMS}
    return {'created_utc':now(),'schema':'sdk20-judge-only-retest-summary-v1','planned_per_arm':1540,'planned_total':3080,
            'source_manifest':first['manifest']['source_manifest'],'views':views,'old_to_new_flips':flips,'method_pairs':method_pairs,
            'firstpass_compat_to_final_for_retested':retry_changes,'final_labels':labels,'resolved':resolved,'supplemental_external_jobs':supplement,
            'phase_counts':{s['run']:s['outcome']['counts'] for s in stages},'phase_usage':{s['run']:s['usage'] for s in stages},
            'total_current_usage':measure_usage([r for s in stages for r in s['ledger']]),
            'phase_sources':[{'manifest':runner.pointer(Path(s['run'])/'run_manifest.json'),'outcome':runner.pointer(Path(s['run'])/'run_outcome.json'),'ledgers':s['ledger_refs']} for s in stages],
            'fixed_answer_F1_unchanged':True,'model_calls_in_summary':0,'source_bindings_validated':True,
            'limitations':['Fixed-answer Judge comparison differs from regenerating Reader under the new plain Reader prompt.','Known cohort with development exposure; paired bootstrap is exploratory.','List-format supplements retain original list values but were not valid original Reader strings.','New logical technical retries can change labels; they are not mere offline parsing.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--firstpass',type=Path,required=True);p.add_argument('--recovery',type=Path);p.add_argument('--followup',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New summary path required')
    with sources.offline():report=summarize(a.firstpass,a.recovery,a.followup)
    save_new(a.output,report);print(json.dumps({'output':str(a.output),'model_calls':0}))

if __name__=='__main__':main()
