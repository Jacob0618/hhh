"""Read-only final seven-arm QA audit; no provider/native clients or dispatch.

Requires the fresh forty-bank source gate and a recorded whole-phase terminal
QA population. First accepted labels and technical recovery stay separate.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_gemini38_plain_qa_v1 as runner
from tools import summarize_sdk20_plain_qa_v1 as old
from tools import summarize_mem0_regroup_plain_qa_v1 as existing
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.summarize_sdk20_modules import InvalidArtifact
from tools.summarize_pilot400 import verified_official_metric
from tools.run_sdk20_modules import digest,save_new,now
from tools.run_sdk20_judge_retest_v2 import read,pointer,sha
from jev_mem import evaluation_gemini38_v1 as evaluator
from tools import sdk20_plain_qa_sources_v1 as historical_guard

SCHEMA='gemini38-seven-arm-plain-qa-summary-v1'
ARMS=runner.ARMS
METRICS=old.METRICS
CODE=('tools/summarize_gemini38_seven_arm_v1.py','tools/summarize_sdk20_plain_qa_v1.py',
      'tools/summarize_mem0_regroup_plain_qa_v1.py','tools/summarize_mem0_sparse1540.py',
      'tools/summarize_sdk20_modules.py','tools/summarize_pilot400.py')


def _clone(f,scope):
    result=FunctionType(f.__code__,{**f.__globals__,**scope},f.__name__,f.__defaults__,f.__closure__)
    result.__kwdefaults__=dict(f.__kwdefaults__ or {});return result


ledger=_clone(old.ledger,{'runner':runner,'evaluator':evaluator})
audit_stage=_clone(old.audit_stage,{'existing':SimpleNamespace(bound=existing.bound,audit_stage=runner._new_audit_stage)})


def _expected_compat(judge):
    interpretation=None;status='unavailable'
    if judge['status']=='complete':
        r=judge['result'];status='strict_valid'if r['parse_mode']=='strict'else'compat_valid'
        interpretation={'status':status,'label':r['label'],'correct':r['correct'],'parse_mode':r['parse_mode'],
                        'protocol':evaluator.JUDGE_PROTOCOL,'raw_text_sha256':r['raw_text_sha256'],'raw_modified':False}
    elif(judge.get('error')or{}).get('type')=='InvalidJudgeLabel':status='rejected'
    return status,interpretation


def intent_inventory(snapshot,api_dir,rows):
    """Count retained intents without ended physical proof; never infer no dispatch."""
    api_dir=Path(api_dir);unknown=[];success_without_raw=[];refs=[]
    for path in sorted((api_dir/'request_intents').glob('*.json')):
        request=snapshot.read(path);key=request.get('request_hash');stage=request.get('stage')
        if not isinstance(key,str)or path.stem!=key:raise InvalidArtifact('Request intent filename/hash differs')
        ref={'path':str(path),'sha256':snapshot.sha(path)};refs.append(ref)
        physical=[r for r in rows if r.get('request_hash')==key and r.get('stage')==stage and r.get('cache_hit')is not True]
        tail=physical[-1]if physical else None
        status=tail.get('status')if tail else None
        ended=tail is not None and isinstance(tail.get('timestamp_utc'),str)and bool(tail['timestamp_utc'])and(
            status=='transport_error'or type(status)is int and 200<=status<600)
        if not ended:unknown.append({'intent':ref,'request_hash':key,'stage':stage,
                                    'reason':'no_ended_physical_ledger_proof','raw_present':(api_dir/'api_cache'/(key+'.json')).is_file()})
        elif type(status)is int and 200<=status<300 and not(api_dir/'api_cache'/(key+'.json')).is_file():
            success_without_raw.append({'intent':ref,'request_hash':key,'stage':stage})
    return {'intent_sources':refs,'unknown_intents_without_ended_physical_proof':unknown,
            'unknown_intent_count':len(unknown),'ended_success_missing_raw_count':len(success_without_raw),
            'ended_success_missing_raw':success_without_raw,
            'interpretation':'An unlogged intent may precede dispatch or an unknown completed request; it is not measured zero.'}


def audit_phase(run_dir,source,*,parent=None):
    run=Path(run_dir).resolve();m,out=runner.terminal(run,_source=source)
    snapshot=Snapshot();snapshot.raw(run/'run_manifest.json');snapshot.raw(run/'run_outcome.json')
    logs,indexed,logrefs=ledger(snapshot,run,m);official,provenance=verified_official_metric()
    if provenance!=m['official_metric_provenance']:raise InvalidArtifact('Official metric binding differs')
    records={};errors=[]
    for j in m['jobs']:
        paths=runner.stage_files(run,j);ident=runner.identity(j,m);api_dir=run/f"conv_{j['conversation']}/api"
        record={'job':j,'paths':{k:str(v)for k,v in paths.items()},'reader':None,'judge':None,'compat':None,'qa':None,'errors':[]}
        for component in ('reader','judge','compat','qa'):
            try:
                if component=='reader':
                    if j.get('reader_source'):
                        if parent is None or j['job_id']not in parent['records']:raise InvalidArtifact('Inherited Reader has no parent slot')
                        prior=parent['records'][j['job_id']];pr=prior['reader'];rp=Path(prior['paths']['reader'])
                        if pr is None or pr['status']!='complete'or j['reader_source']!=pointer(rp):
                            raise InvalidArtifact('Reader inheritance is not the original valid artifact')
                        row=existing.bound(snapshot,paths['reader'],{**ident,'stage':'reader'});ctx=runner.stage_context(j,m,'reader')
                        if(row.get('status')!='complete'or row.get('result')!=pr['result']or row.get('inherited_from')!=j['reader_source']
                           or row.get('api_events')!=[]or row.get('model_calls')!=0 or row.get('error')is not None
                           or row.get('raw_invalid_label')is not None or paths['reader'].with_suffix('.intent.json').exists()
                           or any(r['context']==ctx for r in logs)):
                            raise InvalidArtifact('Successful Reader was changed or regenerated during recovery')
                    else:
                        row=audit_stage(snapshot,paths['reader'],{**ident,'stage':'reader'},api_dir,
                             runner.stage_context(j,m,'reader'),indexed,lambda api:runner.reader_call(api,j,official))
                elif component=='judge':
                    reader=record['reader'];reader_ok=reader is not None and reader['status']=='complete'
                    expected={**ident,'stage':'judge'}
                    if paths['reader'].exists():expected['reader_sha256']=snapshot.sha(paths['reader'])
                    row=audit_stage(snapshot,paths['judge'],expected,api_dir,runner.stage_context(j,m,'judge'),indexed,
                         lambda api:evaluator.judge_answer(api,j['question'],j['reference'],reader['result']['prediction']['answer']),
                         no_dispatch=not reader_ok)
                elif component=='compat':
                    row=existing.bound(snapshot,paths['compat'],{**ident,'stage':'compat'});judge=record['judge']
                    if judge is None:raise InvalidArtifact('Parser record has no audited Judge stage')
                    status,interpretation=_expected_compat(judge)
                    if(row.get('status')!=status or row.get('interpretation')!=interpretation or row.get('strict')!=pointer(paths['judge'])
                       or row.get('source_evidence')is not None or row.get('error')is not None or row.get('model_calls')!=0
                       or row.get('raw_or_strict_modified')is not False):
                        raise InvalidArtifact('Parse metadata does not match the first accepted/rejected raw Judge')
                else:
                    row=existing.bound(snapshot,paths['qa'],{**ident,'stage':'qa'})
                    reader,judge,compat=(record[k]for k in('reader','judge','compat'))
                    if any(v is None for v in(reader,judge,compat)):raise InvalidArtifact('QA record references an unaudited stage')
                    for stage in('reader','judge','compat'):
                        if row.get(stage)!=pointer(paths[stage]):raise InvalidArtifact('QA stage pointer changed')
                    reader_ok=reader['status']=='complete';resolved=runner.resolve_pair(judge,compat)
                    if j.get('reader_source'):
                        pr=parent['records'][j['job_id']];resolved=runner.resolve_pair(pr['judge'],pr['compat'],judge,compat)
                    elif m['phase_kind']=='technical_recovery':resolved=runner.resolve_pair(None,None,judge,compat)
                    if(row.get('reader_status')!=reader['status']or row.get('judge_status')!=judge['status']
                       or row.get('reader_metrics')!=(reader['result']['metrics']if reader_ok else None)
                       or row.get('judge_correct')!=(judge['result']['correct']if judge['status']=='complete'else None)
                       or row.get('resolved_judge')!=resolved
                       or row.get('status')!=('complete'if reader['status']==judge['status']=='complete'else'failed')):
                        raise InvalidArtifact('QA record changes separate metrics or accepted-label priority')
                record[component]=row
            except Exception as ex:
                record['errors'].append({'stage':component,'type':type(ex).__name__,'message':str(ex)[:1000]})
        records[j['job_id']]=record
        errors.extend({'job_id':j['job_id'],**e}for e in record['errors'])
    intents={str(ci):intent_inventory(snapshot,run/f'conv_{ci}/api',[r for r in logs if r['context']['conversation']==ci])
             for ci in{j['conversation']for j in m['jobs']}}
    if snapshot.unchanged():raise InvalidArtifact('QA phase source changed during replay')
    return {'run':str(run),'manifest':m,'outcome':out,'records':records,'ledger':logs,'ledger_refs':logrefs,
            'missing_ledgers':[str(run/f'conv_{ci}/api/api_calls.jsonl')for ci in{j['conversation']for j in m['jobs']}
                               if not(run/f'conv_{ci}/api/api_calls.jsonl').exists()],
            'intent_inventory':intents,
            'errors':errors,'input_sha256':{str(k):v for k,v in snapshot.files.items()}}


def _event_refs(row):
    return deepcopy(row.get('api_events',[]))if row else[]


def record_rows(jobs,first,second=None,*,final=False):
    rows=[]
    for j in jobs:
        a=first['records'][j['job_id']];b=second['records'].get(j['job_id'])if final and second else None
        reader_record=a if a['reader']and a['reader']['status']=='complete'else(b or a)
        reader=reader_record['reader'];reader_ok=reader is not None and reader['status']=='complete'
        resolution=runner.resolve_pair(a['judge'],a['compat'],b['judge']if b else None,b['compat']if b else None)
        judge_record=a if resolution['source']=='firstpass_accepted_label'else b if resolution['source']=='recovery_accepted_label'else None
        judge=judge_record['judge']if judge_record else None
        label=resolution['correct']
        if label is not None and not reader_ok:raise InvalidArtifact('Accepted label without a valid Reader')
        scores={name:reader['result']['metrics'][name]if reader_ok else None for name in METRICS[:2]}
        scores['judge_correct']=float(label)if label is not None else None
        answer=reader['result']['prediction']['answer']if reader_ok else None
        rows.append({'job_id':j['job_id'],'query_id':j['query_id'],'conversation':j['conversation'],'category':j['category'],'method':j['method'],
                     'question':j['question'],'reference':j['reference'],'reference_date':j['reference_date'],'selected_ids':j['selected_ids'],
                     'bank':j['bank'],'selection':j['selection'],'selection_status':'complete',
                     'reader_status':reader['status']if reader else'invalid','judge_status':'complete'if label is not None else'missing',
                     'qa_status':'complete'if label is not None else'failed','scores':scores,'reader_answer':answer,
                     'reader_answer_characters':len(answer)if answer is not None else None,
                     'reader_answer_whitespace_words':len(answer.split())if answer is not None else None,
                     'reader_artifact':pointer(reader_record['paths']['reader'])if reader_ok else None,
                     'reader_events':_event_refs(reader),'reader_inherited_from':reader.get('inherited_from')if reader else None,
                     'judge_resolution':resolution,'judge_artifact':pointer(judge_record['paths']['judge'])if judge else None,
                     'judge_events':_event_refs(judge),'judge_raw_text':judge['result']['raw']if judge else None,
                     'judge_parse_mode':judge['result']['parse_mode']if judge else None,
                     'errors':a['errors']+(b['errors']if b else[])})
    return rows


def aggregate(rows,*,bootstrap_samples=2000):
    scoped=_clone(existing.aggregate,{'ARMS':ARMS})
    result=scoped(rows,denominator=1540,bootstrap_samples=bootstrap_samples)
    first_pair=result.pop('paired_merged_minus_unmerged');comparisons={}
    baseline=ARMS[0]
    def valid(arm,name):
        return {r['query_id']:{'query_id':r['query_id'],'conversation':r['conversation'],name:r['scores'][name]}
                for r in rows if r['method']==arm and r['scores'][name]is not None}
    for arm in ARMS[1:]:
        pair=first_pair if arm==ARMS[1]else{name:existing.paired_metric(name,valid(baseline,name),valid(arm,name),
               samples=bootstrap_samples,denominator=1540)for name in METRICS}
        left={r['query_id']:r for r in rows if r['method']==baseline};right={r['query_id']:r for r in rows if r['method']==arm}
        transition=defaultdict(list)
        for qid in left:
            x,y=left[qid]['scores']['judge_correct'],right[qid]['scores']['judge_correct']
            transition['missing_either'if x is None or y is None else f'{int(x)}_to_{int(y)}'].append(qid)
        comparisons[arm+'_minus_'+baseline]={'left':baseline,'right':arm,'metrics':pair,
                'judge_transitions':{key:{'count':len(transition[key]),'query_ids':transition[key]}for key in('0_to_0','0_to_1','1_to_0','1_to_1','missing_either')}}
    result['paired_comparisons']=comparisons;return result


def measured_usage(rows,*,missing_ledgers=()):
    """Physical token observations, without applying old-model tariffs to 3.8."""
    physical=[r for r in rows if r.get('cache_hit')is not True];cache=[r for r in rows if r.get('cache_hit')is True]
    names={'prompt':['promptTokenCount','input_tokens'],'output':['candidatesTokenCount','output_tokens'],
           'thinking':['thoughtsTokenCount'],'total':['totalTokenCount','total_tokens']}
    fields={name:{'recorded_attempts':0,'missing_attempts':0,'recorded_token_sum':0}for name in names}
    statuses=Counter();models=Counter();unknown_usage=0
    for row in physical:
        if row.get('cache_hit')not in(None,False):raise InvalidArtifact('Invalid physical/cache flag')
        status=row.get('status');statuses['http_success'if type(status)is int and 200<=status<300 else'http_failure'if type(status)is int and 400<=status<600 else'transport_or_unknown']+=1
        models[row.get('model','unknown')]+=1;usage=row.get('usage')or{}
        unknown_usage+=not bool(usage)or row.get('usage_unknown')is True
        for name,keys in names.items():
            value=next((usage[k]for k in keys if k in usage),None)
            if value is None:fields[name]['missing_attempts']+=1;continue
            if type(value)not in(int,float)or not math.isfinite(value)or value<0:raise InvalidArtifact('Malformed actual token usage')
            fields[name]['recorded_attempts']+=1;fields[name]['recorded_token_sum']+=value
    return {'scope':'Actual supplied ledger rows only; cache usage is excluded from physical token sums',
            'available':not bool(missing_ledgers),'physical_attempts':len(physical),'cache_events':len(cache),'physical_status_counts':dict(statuses),
            'availability_semantics':'available means expected ledger files were present, not a proof of complete dispatch accounting; consult intent_inventory',
            'physical_attempts_by_model':dict(models),'usage_unknown_attempts':unknown_usage,'token_fields':fields,
            'missing_ledgers':list(missing_ledgers),'total_cost_usd':None,
            'pricing_status':'No new Gemini3.8/JEV/embedding tariff is adopted; recorded usage is not a paid invoice',
            'missing_thinking_is_not_zero':True}


def precomputation(snapshot,source):
    parent=Path(source['parent_run']);rows=[];refs=[];missing=[];intents={}
    for ci in range(10):
        paths=[parent/f'conv_{ci}'/arm/'native/api_calls.jsonl'for arm in('session','source_local_context','boundary_multiscale','boundary_neighbor')]
        paths += [parent/'retrieval'/f'conv_{ci}'/p/'api_calls.jsonl'for p in('tag_api','query_type_api')]
        for path in paths:
            local_rows=[]
            if not path.exists():
                missing.append(str(path));intents[str(path.parent)]=intent_inventory(snapshot,path.parent,[]);continue
            data=snapshot.raw(path);refs.append({'path':str(path),'sha256':snapshot.sha(path)})
            for line in data.decode('utf-8-sig').splitlines():
                if line.strip():
                    row=json.loads(line)
                    if row.get('model')not in('gemini-3.8-flash','jev-1.13.0','gemini-embedding-001'):
                        raise InvalidArtifact('New precomputation ledger contains another model')
                    if row.get('context',{}).get('conversation')!=ci:raise InvalidArtifact('Precomputation ledger conversation differs')
                    rows.append(row);local_rows.append(row)
            intents[str(path.parent)]=intent_inventory(snapshot,path.parent,local_rows)
    # SDK embedding reuse is stored in selection events, not a physical ledger.
    reuse=[];seen=set()
    for j in source['jobs']:
        path=j['selection']['path']
        if path in seen:continue
        seen.add(path);selection=snapshot.read(path)
        for event in selection.get('cache_events',[]):
            if event.get('cache_hit')is not True or event.get('model_calls')!=0:
                raise InvalidArtifact('Cache-only retrieval event declares fresh model usage')
            reuse.append(event)
    return {'usage':measured_usage(rows,missing_ledgers=missing),'ledger_sources':refs,'intent_inventory':intents,
            'by_model':{model:measured_usage([r for r in rows if r.get('model')==model])for model in sorted({r['model']for r in rows})},
            'embedding_pack':read(source['retrieval_manifest']['path'])['embedding_cache_pack'],
            'retrieval_embedding_cache_events':len(reuse),'retrieval_embedding_unique_request_hashes':len({e['request_hash']for e in reuse}),
            'note':'Writer/tag/query physical ledgers are separate from QA; inherited QUERY embedding cost is not measured as new usage.'}


def readiness(firstpass,recovery=None):
    """No scores: a diagnostic readiness checklist for absent or pending phases."""
    run=Path(firstpass);items=[]
    def check(name,call):
        try:detail=call();items.append({'check':name,'passed':True,'detail':detail})
        except Exception as ex:items.append({'check':name,'passed':False,'error_type':type(ex).__name__,'detail':str(ex)[:700]})
    check('new_firstpass_manifest',lambda:runner.manifest(run)['phase_kind'])
    check('whole_firstpass_terminal',lambda:runner.terminal(run)[1]['total'])
    if recovery:check('whole_single_recovery_terminal',lambda:runner.terminal(recovery)[1]['total'])
    return {'schema':SCHEMA+'-readiness','created_utc':now(),'firstpass':str(run),'checks':items,
            'ready_for_summary':all(i['passed']for i in items),'results_computed':False,'model_calls':0}


def summarize(firstpass,recovery=None,*,bootstrap_samples=2000):
    initial=read(Path(firstpass)/'run_manifest.json');source=runner.source_validate(initial['source_manifest']['path'])
    if len(source['jobs'])!=10780 or Counter(j['method']for j in source['jobs'])!={a:1540 for a in ARMS}:
        raise InvalidArtifact('Fresh seven-arm10780 source cohort required')
    first=audit_phase(firstpass,source);second=audit_phase(recovery,source,parent=first)if recovery else None
    if first['manifest']['phase_kind']!='firstpass':raise InvalidArtifact('Main phase is not firstpass')
    if second:
        p=second['manifest']['parent']
        if(p['manifest']!=pointer(Path(firstpass)/'run_manifest.json')or p['outcome']!=pointer(Path(firstpass)/'run_outcome.json')):
            raise InvalidArtifact('Recovery belongs to another firstpass')
    views={'accepted_firstpass':record_rows(source['jobs'],first),
           'final_with_uniform_technical_recovery':record_rows(source['jobs'],first,second,final=True)}
    phases=[first]+([second]if second else[]);logs=[r for p in phases for r in p['ledger']]
    errors=[{'phase':p['run'],**e}for p in phases for e in p['errors']]
    snapshot=Snapshot();snapshot.raw(initial['source_manifest']['path']);pre=precomputation(snapshot,source)
    if snapshot.unchanged():raise InvalidArtifact('Precomputation ledger/source changed during summary')
    runner.source_validate(initial['source_manifest']['path'])
    for p in phases:
        if any(not Path(path).exists()or sha(path)!=hashed for path,hashed in p['input_sha256'].items()):
            raise InvalidArtifact('Audited QA input changed before final summary publication')
    phase_slots={p['run']:{key:{'job_id':key,'paths':r['paths'],'errors':r['errors'],
         'stages':{stage:{'status':row.get('status'),'error':row.get('error'),'api_events':_event_refs(row),
                         'raw_invalid_label':row.get('raw_invalid_label'),
                         'reader_answer':row['result']['prediction']['answer']if stage=='reader'and row.get('status')=='complete'else None,
                         'metrics':row['result'].get('metrics')if stage=='reader'and row.get('status')=='complete'else None,
                         'judge_result':row.get('result')if stage=='judge'else None,
                         'inherited_from':row.get('inherited_from')}if row else None
                   for stage,row in((s,r[s])for s in('reader','judge','compat','qa'))}}
         for key,r in p['records'].items()}for p in phases}
    return {'schema':SCHEMA,'created_utc':now(),'declared_arms':list(ARMS),'planned_per_arm':1540,'planned_total':10780,
            'repetitions':1,'source_manifest':initial['source_manifest'],'new_execution_config':source['new_execution_config'],
            'evaluation_protocol':evaluator.PROTOCOL,'official_metric_provenance':source['official_metric_provenance'],
            'views':{name:aggregate(rows,bootstrap_samples=bootstrap_samples)for name,rows in views.items()},'slots_by_view':views,
            'audit_status':'no_invalid_artifacts'if not errors else'invalid_artifacts_retained_as_missing','audit_errors':errors,
            'phase_slot_audits':phase_slots,
            'phase_sources':[{'manifest':pointer(Path(p['run'])/'run_manifest.json'),'outcome':pointer(Path(p['run'])/'run_outcome.json'),
                              'ledgers':p['ledger_refs']}for p in phases],
            'phase_counts':{p['run']:p['outcome']['counts']for p in phases},'phase_usage':{p['run']:measured_usage(p['ledger'],missing_ledgers=p['missing_ledgers'])for p in phases},
            'phase_intent_inventory':{p['run']:p['intent_inventory']for p in phases},
            'qa_physical_usage':measured_usage(logs,missing_ledgers=[x for p in phases for x in p['missing_ledgers']]),'qa_usage_by_stage':{name:measured_usage([r for r in logs if r['context']['component']==name])for name in('reader','judge')},
            'qa_usage_by_producer_method':{arm:measured_usage([r for r in logs if r['context']['method']==arm])for arm in ARMS},
            'precomputation':pre,'input_sha256':{**{k:v for p in phases for k,v in p['input_sha256'].items()},**{str(k):v for k,v in snapshot.files.items()}},
            'summary_code_sha256':{p:sha(ROOT/p)for p in CODE},'model_calls_in_summary':0,'native_clients':0,
            'limitations':['One all-stage rebuild/run; no independent repetition SD or isolated Judge-prompt causal effect.',
              'New baseline comparisons share this3.8 Writer/Reader/Judge protocol; older differently judged scores are not causal controls.',
              'Official LoCoMo F1 and custom token F1 differ; full reference is retained for Judge and official category rules remain unchanged.',
              'First legal strict or unanimous terminal Judge label is accepted before recovery; no validYES/NO rejudging or best-score selection.',
              'Missing/invalid outcomes have null scores; fixed1540, metric-specific valid subsets and zero-imputation sensitivity are separate.',
              'Shared exact same-phase requests are correlated consumers, not independent model samples. Usage by producer method is not a causal marginal arm cost.',
              'Token fields are recorded observations; absent thoughtsTokenCount is unavailable, not zero; new Gemini3.8 tariff/billing is not inferred.',
              'Assistant source review is sampled semantic evidence, not fact-level human correctness gold. Bootstrap is exploratory on exposed10conversations.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--firstpass',type=Path,required=True);p.add_argument('--recovery',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--readiness',action='store_true');p.add_argument('--bootstrap-samples',type=int,default=2000);a=p.parse_args()
    if a.bootstrap_samples<1:raise ValueError('Positive bootstrap sample count required')
    if not a.readiness and a.output is None:p.error('--output required for summary')
    if a.output is not None and a.output.exists():raise FileExistsError('Fresh summary output required')
    with historical_guard.historical.offline():
        result=readiness(a.firstpass,a.recovery)if a.readiness else summarize(a.firstpass,a.recovery,bootstrap_samples=a.bootstrap_samples)
    if a.output:save_new(a.output,result)
    print(json.dumps(result if a.readiness else{'output':str(a.output),'audit_status':result['audit_status'],'planned_total':10780,'model_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
