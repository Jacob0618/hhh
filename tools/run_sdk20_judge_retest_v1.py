"""Judge-only fixed-Reader retest, with one separate postbatch technical phase.

prepare/check never construct an API or read credentials. Only explicit run
dispatches the existing plain Judge. Strict and compatibility records are
immutable and separate; no Reader/native/embedding work is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read
from tools.run_sdk20_modules import digest,save_new,now
from tools.run_pilot400_experiments_v2 import publish_json
from jev_mem.evaluation_plain_text_v1 import judge_answer,make_judge_prompt,InvalidJudgeLabel,JUDGE_MODEL,JUDGE_STAGE,JUDGE_PROTOCOL
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.judge_terminal_label_compat_v1 import derive_label,eligible_raw_response,PROTOCOL as COMPAT_PROTOCOL
from jev_mem.answer_format_compat import normalize_reader_prediction
from jev_mem.providers import ModelAPI

SCHEMA='sdk20-plain-judge-retest-run-v1'
SOURCE_SCHEMA='sdk20-plain-judge-retest-sources-v1'
POLICY={'judge_protocol':JUDGE_PROTOCOL,'judge_model':JUDGE_MODEL,'judge_stage':JUDGE_STAGE,
    'compat_protocol':COMPAT_PROTOCOL,'reader':'fixed_source_string_no_reader_call',
    'json_output':False,'max_output':2048,'transport_retries':2,'conversation_workers':8,'jobs_per_conversation':4,
    'cache':'same_phase_same_conversation_exact_request_only_no_external_roots',
    'phases':'firstpass_then_one_postbatch_recovery_then_only_deferred_reader_first_judge_failures_once',
    'retry_population':'all_firstpass_strict_failures_including_compat_valid_not_strict_yes_or_no',
    'final_priority':['firstpass_strict','retry_strict','firstpass_eligible_compat','retry_eligible_compat'],
    'incorrect_strict_label_retried':False,'blocked_reader':'main_null_then_source_bound_original_list_newline_only',
    'supplement_followup':'only_deferred_38_first_Judge_strict_failures_once_no_main_retry_or_global'}
CODE=('tools/run_sdk20_judge_retest_v1.py','tools/prepare_sdk20_judge_retest_sources_v1.py',
    'jev_mem/evaluation_plain_text_v1.py','jev_mem/plain_text_api_v1.py','jev_mem/judge_terminal_label_compat_v1.py',
    'jev_mem/providers.py','jev_mem/sdk20_cached_api.py','jev_mem/paced_embedding_api.py','jev_mem/answer_format_compat.py',
    'tools/run_sdk20_modules.py','tools/run_pilot400_experiments_v2.py')
SAFE=re.compile(r'^[A-Za-z0-9_-]+$')
VAULT=Path(r'C:\Users\86139\Desktop\jev\experiments\live\credential_store.py')


def read(path):return json.loads(shared_read(Path(path)).decode('utf-8-sig'))
def sha(path):return hashlib.sha256(shared_read(Path(path))).hexdigest()
def pointer(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def verify(path):
    value=read(path)
    if value.get('content_sha256')!=digest({k:v for k,v in value.items() if k!='content_sha256'}):raise ValueError('Content seal differs: '+str(path))
    return value
def bound(ref):
    if pointer(ref['path'])!=ref:raise ValueError('Source pointer changed')
    return read(ref['path'])
def source_validate(path):
    from tools.prepare_sdk20_judge_retest_sources_v1 import validate
    return validate(Path(path))
def check_refs(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value and isinstance(value['path'],str):
            if sha(value['path'])!=value['sha256']:raise ValueError('Referenced source changed')
        else:
            for x in value.values():check_refs(x)
    elif isinstance(value,list):
        for x in value:check_refs(x)
def job_key(job):return (job['conversation'],job['method'],job['query_id'])
def check_jobs(jobs):
    keys=[]
    for j in jobs:
        if (type(j.get('conversation')) is not int or j['conversation'] not in range(10)
                or not isinstance(j.get('method'),str) or not SAFE.fullmatch(j['method'])
                or not isinstance(j.get('query_id'),str) or not SAFE.fullmatch(j['query_id'])):raise ValueError('Invalid job identity')
        expected_id=j['method']+'/'+j['query_id']
        if j.get('main_cohort_member') is False and (j['conversation'],j['method'],j['query_id'])==(2,'dated_unmerged385','c2_q74'):
            expected_id='global_conflict/'+expected_id
        if j.get('job_id')!=expected_id:raise ValueError('Invalid scoped job identity')
        if not isinstance(j.get('question'),str) or not j['question'].strip() or not isinstance(j.get('reference'),str):raise ValueError('Fixed question/reference strings required')
        if j.get('status')=='ready':
            if (not isinstance(j.get('answer'),str) or not j['answer'].strip()
                    or j.get('answer_sha256')!=hashlib.sha256(j['answer'].encode()).hexdigest()):raise ValueError('Exact fixed Reader answer required')
        elif j.get('status')=='blocked_original_reader':
            if j.get('answer') is not None:raise ValueError('Blocked Reader cannot silently provide an answer')
        else:raise ValueError('Undeclared job status')
        keys.append(job_key(j))
    if len(set(keys))!=len(keys):raise ValueError('Duplicate conversation/method/query job')


def _new_manifest(run,source_path,source,*,phase_kind,phase_id,jobs,parent=None,resolutions=None):
    run=Path(run).resolve()
    if run.parent!=(ROOT/'runs').resolve() or run.exists():raise FileExistsError('A fresh direct child of runs is required')
    check_jobs(jobs)
    value={'schema':SCHEMA,'created_utc':now(),'policy':POLICY,'phase_kind':phase_kind,'phase_id':phase_id,
        'source_manifest':pointer(source_path),'jobs':jobs,'parent':parent,'blocked_resolutions':resolutions,
        'code_sha256':{p:sha(ROOT/p) for p in CODE},'source_schema':source['schema'],
        'model_calls_during_prepare':0,'counts':dict(Counter(j['status'] for j in jobs))}
    run.mkdir(parents=True,exist_ok=False)
    publish_json(run/'run_manifest.json',{**value,'protocol_sha256':digest(value)})
    return {'run_dir':str(run),'protocol_sha256':digest(value),'jobs':len(jobs),'counts':value['counts'],'model_calls':0}


def prepare(run_dir,source_path):
    source=source_validate(source_path)
    if source.get('schema')!=SOURCE_SCHEMA or source.get('phase_kind')!='firstpass':raise ValueError('Frozen firstpass sources required')
    jobs=copy.deepcopy(source['jobs'])
    if len(jobs)!=3080 or Counter(j['method'] for j in jobs)!={'baseline_sdk20':1540,'fact_types_sdk20':1540}:raise ValueError('Firstpass must preserve both1540-slot denominators')
    return _new_manifest(run_dir,source_path,source,phase_kind='firstpass',phase_id=source['phase_id'],jobs=jobs)


def result_paths(run,job):
    suffix=Path(f"conv_{job['conversation']}")/job['method']/(job['query_id']+'.json')
    return {stage:Path(run)/stage/suffix for stage in ('strict','compat')}


def terminal(run):
    run=Path(run).resolve();m=manifest(run);out=verify(run/'run_outcome.json')
    if (out.get('terminal') is not True or out.get('protocol_sha256')!=m['protocol_sha256']
            or out.get('total')!=len(m['jobs']) or len(out.get('results',[]))!=len(m['jobs'])):raise ValueError('Whole prior phase is not terminal')
    if {job_key(x) for x in out['results']}!={job_key(x) for x in m['jobs']}:raise ValueError('Terminal job population differs')
    bykey={job_key(j):j for j in m['jobs']}
    for row in out['results']:
        job=bykey[job_key(row)];expected=identity(job,m);paths=result_paths(run,job)
        if row['job_id']!=job['job_id']:raise ValueError('Terminal scoped job identity differs')
        for stage in ('strict','compat'):
            if row[stage] is not None:
                if row[stage]!=pointer(paths[stage]):raise ValueError('Terminal result points to another slot')
                value=verify(paths[stage])
                if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal artifact identity differs')
            elif paths[stage].exists():raise ValueError('Existing result cannot be declared unknown')
    return m,out


def deferred_job(old):
    """Use only the source validator's fixed original-list normalization and F1."""
    d=old.get('deferred_reader_resolution')
    if (not d or d.get('status')!='deferred_postbatch' or d.get('reader_resolution_authorized') is not True
            or d.get('source_choice')!='original_firstpass_raw_only_not_old_recovery'
            or d.get('reader_model_calls')!=0 or d.get('main_firstpass_metrics_remain_null') is not True):
        raise ValueError('Frozen deferred original-list proof required')
    normalized,proof=normalize_reader_prediction({'answer':old['pending_original_list']})
    if (d['normalization_proof']!=proof or d['normalized_answer']!=normalized['answer']
            or d['answer_sha256']!=hashlib.sha256(normalized['answer'].encode()).hexdigest()):
        raise ValueError('Deferred answer must be the exact original-list newline normalization')
    job=copy.deepcopy(old);job.update(status='ready',answer=d['normalized_answer'],answer_sha256=d['answer_sha256'],F1=copy.deepcopy(d['F1']))
    return job


def recovery_jobs(firstpass_run,blocked_resolutions=None):
    parent,out=terminal(firstpass_run)
    if parent['phase_kind']!='firstpass':raise ValueError('A recovery phase cannot itself be retried')
    source=source_validate(parent['source_manifest']['path']);bykey={job_key(j):j for j in parent['jobs']};jobs=[]
    for row in out['results']:
        original=bykey[job_key(row)]
        if original['status']=='blocked_original_reader':continue
        strict=bound(row['strict']) if row['strict'] else None
        if strict is not None and strict['status']=='complete':continue # yes and no both terminal
        job=copy.deepcopy(original);job['attempt_kind']='technical_retry_after_firstpass'
        job['parent_result']={'strict':row['strict'],'compat':row['compat'],'original_job_sha256':digest(original),'unknown':strict is None}
        jobs.append(job)
    for original in source.get('supplemental_jobs',[]):
        if original.get('status')!='deferred_postbatch':raise ValueError('Supplemental source must be explicitly deferred')
        job=copy.deepcopy(original);job['status']='ready';job['attempt_kind']='supplemental_global_retry';jobs.append(job)
    # The disclosed, source-sealed policy resolves all original lists only here.
    resolved=[deferred_job(j) for j in parent['jobs'] if j['status']=='blocked_original_reader']
    resolution_ref=None
    if blocked_resolutions is not None:
        resolutions=verify(blocked_resolutions);check_refs(resolutions)
        if (resolutions.get('schema')!='sdk20-judge-retest-blocked-reader-resolutions-v1'
                or resolutions.get('reader_resolution_authorized') is not True
                or resolutions.get('firstpass_source_manifest')!=parent['source_manifest']):raise ValueError('Explicit separately bound Reader resolution required')
        if resolutions['jobs']!=resolved:raise ValueError('External resolutions differ from source-bound original-list answers or F1')
        resolution_ref=pointer(blocked_resolutions)
    for job in resolved:
        job['attempt_kind']='first_judge_after_explicit_reader_resolution';jobs.append(job)
    check_jobs(jobs)
    return source,jobs,{'run_dir':str(Path(firstpass_run).resolve()),'manifest':pointer(Path(firstpass_run)/'run_manifest.json'),
        'outcome':pointer(Path(firstpass_run)/'run_outcome.json')},resolution_ref


def prepare_recovery(firstpass_run,blocked_resolutions=None):
    firstpass_run=Path(firstpass_run).resolve()
    source,jobs,parent,resolutions=recovery_jobs(firstpass_run,blocked_resolutions)
    run=firstpass_run.with_name(firstpass_run.name+'_recovery_v1')
    return _new_manifest(run,bound(parent['manifest'])['source_manifest']['path'],source,phase_kind='postbatch_recovery',
        phase_id=source['phase_id']+'_postbatch_recovery_v1',jobs=jobs,parent=parent,resolutions=resolutions)


def supplement_followup_jobs(recovery_run):
    parent,out=terminal(recovery_run)
    if parent['phase_kind']!='postbatch_recovery':raise ValueError('Only the first deferred Judge phase may have a followup')
    source=source_validate(parent['source_manifest']['path']);bykey={job_key(j):j for j in parent['jobs']};jobs=[]
    for row in out['results']:
        original=bykey[job_key(row)]
        if original.get('attempt_kind')!='first_judge_after_explicit_reader_resolution':continue
        strict=bound(row['strict']) if row['strict'] else None
        if strict is not None and strict['status']=='complete':continue
        job=copy.deepcopy(original);job['attempt_kind']='deferred_reader_second_and_final_judge'
        job['parent_result']={'strict':row['strict'],'compat':row['compat'],'original_job_sha256':digest(original),'unknown':strict is None}
        jobs.append(job)
    check_jobs(jobs)
    return source,jobs,{'run_dir':str(Path(recovery_run).resolve()),'manifest':pointer(Path(recovery_run)/'run_manifest.json'),
        'outcome':pointer(Path(recovery_run)/'run_outcome.json')}


def prepare_supplement_followup(recovery_run):
    recovery_run=Path(recovery_run).resolve();source,jobs,parent=supplement_followup_jobs(recovery_run)
    return _new_manifest(recovery_run.with_name(recovery_run.name+'_supplement_followup_v1'),
        bound(parent['manifest'])['source_manifest']['path'],source,phase_kind='supplement_followup',
        phase_id=source['phase_id']+'_supplement_followup_v1',jobs=jobs,parent=parent)


def manifest(run_dir):
    run=Path(run_dir).resolve();m=read(run/'run_manifest.json')
    if m.get('schema')!=SCHEMA or m.get('policy')!=POLICY or m['protocol_sha256']!=digest({k:v for k,v in m.items() if k!='protocol_sha256'}):raise ValueError('Run manifest/policy changed')
    if m['code_sha256']!={p:sha(ROOT/p) for p in CODE}:raise ValueError('Frozen Judge/source code changed')
    source=source_validate(m['source_manifest']['path']);bound(m['source_manifest'])
    if m['phase_kind']=='firstpass':
        if m['jobs']!=source['jobs'] or m['parent'] is not None or m['blocked_resolutions'] is not None:raise ValueError('Firstpass source jobs changed')
    elif m['phase_kind']=='postbatch_recovery':
        p=m['parent'];bound(p['manifest']);bound(p['outcome'])
        if run!=Path(p['run_dir']).with_name(Path(p['run_dir']).name+'_recovery_v1'):raise ValueError('Only one fixed fresh recovery namespace permitted')
        source2,jobs,parent,resolutions=recovery_jobs(p['run_dir'],m['blocked_resolutions']['path'] if m['blocked_resolutions'] else None)
        if jobs!=m['jobs'] or parent!=p or resolutions!=m['blocked_resolutions']:raise ValueError('Postbatch recovery population changed')
    elif m['phase_kind']=='supplement_followup':
        p=m['parent'];bound(p['manifest']);bound(p['outcome'])
        if run!=Path(p['run_dir']).with_name(Path(p['run_dir']).name+'_supplement_followup_v1'):raise ValueError('Only one fixed supplement followup namespace permitted')
        source2,jobs,parent=supplement_followup_jobs(p['run_dir'])
        if jobs!=m['jobs'] or parent!=p or m['blocked_resolutions'] is not None:raise ValueError('Supplement followup population changed')
    else:raise ValueError('Undeclared phase')
    check_jobs(m['jobs']);return m


def signature(job):
    return {'provider':'gemini','model':JUDGE_MODEL,'operation':'generateContent','body':{
        'contents':[{'role':'user','parts':[{'text':make_judge_prompt(job['question'],job['reference'],job['answer'])}]}],
        'generationConfig':{'temperature':0,'maxOutputTokens':2048}}}


def response_evidence(api_dir,job,m):
    sig=signature(job);key=hashlib.sha256(json.dumps(sig,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    path=Path(api_dir)/'api_cache'/(key+'.json');record=read(path)
    if record['request']!=sig or record['stage']!=JUDGE_STAGE:raise ValueError('Actual raw request differs from frozen Judge inputs')
    ledger=Path(api_dir)/'api_calls.jsonl';data=shared_read(ledger);data=data[:data.rfind(b'\n')+1]
    events=[json.loads(s) for s in data.splitlines() if s.strip()]
    matches=[(i+1,e) for i,e in enumerate(events) if e.get('request_hash')==key and e.get('stage')==JUDGE_STAGE and not e.get('cache_hit') and e.get('status')==200]
    if len(matches)!=1:raise ValueError('Exactly one current-phase physical success must produce the cached raw')
    line,event=matches[0]
    context=event.get('context',{})
    if (event.get('model')!=JUDGE_MODEL or event.get('operation')!='generateContent'
            or any(context.get(k)!=v for k,v in {'conversation':job['conversation'],'architecture':SCHEMA,
                'phase':'judge_retest','phase_id':m['phase_id']}.items())):raise ValueError('Raw producer identity differs')
    return record['response'],{'request_hash':key,'raw':pointer(path),'ledger':str(ledger.resolve()),
        'ledger_prefix_bytes':len(data),'ledger_prefix_sha256':hashlib.sha256(data).hexdigest(),
        'physical_producer_line':line,'physical_producer':event,'physical_producer_sha256':digest(event)}


def raw_text(response,job):
    """Replay the frozen ordinary-text provider parser without any transport."""
    class RawOnly:
        generate=ModelAPI.generate
        def request(self,provider,model,body,*,operation='generateContent',stage='unspecified'):
            if {'provider':provider,'model':model,'operation':operation,'body':body}!=signature(job) or stage!=JUDGE_STAGE:
                raise ValueError('Offline Judge replay request changed')
            return response
    return RawOnly().generate(make_judge_prompt(job['question'],job['reference'],job['answer']),
        model=JUDGE_MODEL,json_output=False,max_output=2048,stage=JUDGE_STAGE)


def identity(job,m):
    return {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],'phase_id':m['phase_id'],
        'job_id':job['job_id'],'conversation':job['conversation'],'method':job['method'],'query_id':job['query_id'],
        'source_job_sha256':digest(job),'source_manifest':m['source_manifest'],'answer_sha256':job.get('answer_sha256'),
        'attempt_kind':job.get('attempt_kind','first_judge_fixed_original_reader')}
def safe_error(exc,api=None):
    return {'type':type(exc).__name__,'message':api._redact(exc)[:1500] if api is not None else 'Local failure before an API redactor was available; inspect the exception type.'}
def durable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as h:json.dump(value,h,ensure_ascii=False);h.flush();os.fsync(h.fileno())


def one(run,job,m,api=None,initial_error=None):
    paths=result_paths(run,job);ident=identity(job,m)
    if any(p.exists() or p.with_suffix('.intent.json').exists() for p in paths.values()):raise RuntimeError('Existing strict/compat attempt retained; no implicit replay')
    if job['status']=='blocked_original_reader':
        save_new(paths['strict'],{**ident,'status':'blocked_original_reader','result':None,'error':None,'raw_invalid_label':None,'api_events':[],'raw_evidence':None})
        save_new(paths['compat'],{**ident,'status':'blocked_original_reader','interpretation':None,'strict':pointer(paths['strict']),'error':None})
        return
    scoped=api.scoped(method=job['method'],query_id=job['query_id'],job_id=job['job_id'],component='judge') if api is not None else None
    durable(paths['strict'].with_suffix('.intent.json'),{**ident,'created_utc':now(),'pid':os.getpid()})
    result=failure=raw_invalid=None
    try:
        if initial_error is not None:raise initial_error
        result=judge_answer(scoped,job['question'],job['reference'],job['answer'])
    except InvalidJudgeLabel as exc:failure=safe_error(exc,scoped);raw_invalid=exc.raw_response
    except Exception as exc:failure=safe_error(exc,scoped)
    strict={**ident,'status':'failed' if failure else 'complete','result':result,'error':failure,'raw_invalid_label':raw_invalid,
        'api_events':list(scoped.events) if scoped is not None else [],'raw_evidence':None}
    # Persist the strict result before doing any compatibility interpretation.
    save_new(paths['strict'],strict)
    interpretation=evidence=compat_error=None;status='unavailable'
    try:
        if strict['status']=='complete' or (failure or {}).get('type')=='InvalidJudgeLabel':
            text=result['raw'] if result else raw_invalid
            response,evidence=response_evidence(Path(run)/f"conv_{job['conversation']}"/'api',job,m)
            if raw_text(response,job)!=text:raise ValueError('Saved Judge text differs from original raw provider replay')
            interpretation=derive_label(text)
            if strict['status']=='complete':
                if interpretation['status']!='strict_valid' or interpretation['correct']!=result['correct']:raise ValueError('Strict result replay differs')
            else:
                ok,reason=eligible_raw_response(response,text,http_status=evidence['physical_producer']['status'])
                if not ok:interpretation={**interpretation,'status':'rejected','label':None,'correct':None,'rejection_reason':reason}
            status=interpretation['status']
    except Exception as exc:compat_error=safe_error(exc,scoped);status='unavailable';interpretation=None
    save_new(paths['compat'],{**ident,'status':status,'interpretation':interpretation,'strict':pointer(paths['strict']),
        'raw_evidence':evidence,'error':compat_error,'raw_or_strict_modified':False,'model_calls':0})


def resolve_pair(first_strict,first_compat,retry_strict=None,retry_compat=None):
    for name,row in (('firstpass_strict',first_strict),('retry_strict',retry_strict)):
        if row and row.get('status')=='complete':
            r=row['result'];return {'source':name,'correct':r['correct'],'label':r['label']}
    for name,row in (('firstpass_eligible_compat',first_compat),('retry_eligible_compat',retry_compat)):
        if row and row.get('status') in ('strict_valid','compat_valid'):
            r=row['interpretation'];return {'source':name,'correct':r['correct'],'label':r['label']}
    return {'source':None,'correct':None,'label':None}


def run(run_dir,*,api_factory=None):
    run_dir=Path(run_dir).resolve();m=manifest(run_dir)
    if (run_dir/'run_outcome.json').exists():return terminal(run_dir)[1]
    if api_factory is None and not VAULT.is_file():raise RuntimeError('DPAPI vault module required; environment fallback disabled')
    if (run_dir/'execution_intent.json').exists():raise RuntimeError('Prior phase failed/unknown is retained; use only the explicit postbatch recovery phase')
    durable(run_dir/'execution_intent.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'pid':os.getpid(),'created_utc':now()})
    def worker(ci,jobs):
        api=None;initial_error=None
        if any(j['status']=='ready' for j in jobs):
            try:
                if api_factory is None:
                    api=PlainTextEvaluationAPI(run_dir/f'conv_{ci}'/'api',cache_roots=(),retries=2,
                        context={'architecture':SCHEMA,'conversation':ci,'phase':'judge_retest','phase_id':m['phase_id']})
                else:api=api_factory(run_dir/f'conv_{ci}'/'api',ci,m)
            except Exception as exc:initial_error=exc
        errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(one,run_dir,j,m,api,initial_error):j for j in jobs}
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:errors.append({'job_id':futures[f]['job_id'],'error':safe_error(exc,api)})
        return {'conversation':ci,'jobs':len(jobs),'local_errors':errors}
    groups={ci:[j for j in m['jobs'] if j['conversation']==ci] for ci in range(10)};outcomes=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(worker,ci,jobs):ci for ci,jobs in groups.items() if jobs}
        for f in as_completed(futures):
            try:outcomes.append(f.result())
            except Exception as exc:outcomes.append({'conversation':futures[f],'local_errors':[safe_error(exc)]})
            publish_json(run_dir/'run_status.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'pid':os.getpid(),'updated_utc':now(),'terminal':False,'conversations_finished':len(outcomes),'outcomes':outcomes})
    results=[];counts={s:Counter() for s in ('strict','compat')}
    for job in m['jobs']:
        row={k:job[k] for k in ('job_id','conversation','method','query_id')}
        for stage,path in result_paths(run_dir,job).items():
            row[stage]=pointer(path) if path.exists() else None
            status=verify(path)['status'] if path.exists() else 'unknown';counts[stage][status]+=1
        results.append(row)
    final={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],'phase_id':m['phase_id'],
        'terminal':True,'finished_utc':now(),'total':len(m['jobs']),'counts':{s:dict(c) for s,c in counts.items()},
        'outcomes':outcomes,'results':results,'reader_calls':0,'external_cache_roots':[],
        'strict_results_preserved':True,'compat_protocol':COMPAT_PROTOCOL}
    save_new(run_dir/'run_outcome.json',final)
    publish_json(run_dir/'run_status.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'pid':os.getpid(),'updated_utc':now(),'terminal':True,'conversations_finished':len(outcomes),'counts':final['counts']})
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    a=sub.add_parser('prepare');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--source',type=Path,required=True)
    a=sub.add_parser('prepare-recovery');a.add_argument('--firstpass-run',type=Path,required=True);a.add_argument('--blocked-resolutions',type=Path)
    a=sub.add_parser('prepare-supplement-followup');a.add_argument('--recovery-run',type=Path,required=True)
    for name in ('check','run'):
        a=sub.add_parser(name);a.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='prepare':result=prepare(a.run_dir,a.source)
    elif a.phase=='prepare-recovery':result=prepare_recovery(a.firstpass_run,a.blocked_resolutions)
    elif a.phase=='prepare-supplement-followup':result=prepare_supplement_followup(a.recovery_run)
    elif a.phase=='check':m=manifest(a.run_dir);result={'protocol_sha256':m['protocol_sha256'],'jobs':len(m['jobs']),'model_calls':0}
    else:
        v=run(a.run_dir);result={k:v[k] for k in ('phase_kind','total','terminal','counts')}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
