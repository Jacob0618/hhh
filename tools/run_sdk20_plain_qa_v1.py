"""Fresh plain Reader/Judge on frozen session-baseline and typed top20 inputs.

prepare/check/recover are offline. recover prepares one fresh sibling phase;
only run may dispatch models. No native client, writer, retrieval, tagger, old
QA cache, or credential access occurs during preparation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
from datetime import date
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_sdk20_modules import digest,save_new,now
from tools.run_pilot400_experiments_v2 import publish_json
from tools.run_sdk20_judge_retest_v2 import read,sha,pointer,bound,verify,safe_error,resolve_pair
from tools.run_mem0_regroup_plain_qa_v1 import stage_once,durable_intent
from tools.replay_plain_judge_compat_v1 import replay_source
from tools.summarize_mem0_regroup_plain_qa_v1 import audit_stage,Snapshot
from tools.summarize_sdk20_modules import fingerprint
from tools.summarize_pilot400 import verified_official_metric
from tools.check_experiment_health import shared_read
from jev_mem.evaluation_plain_text_v1 import read_answer,judge_answer,reader_input,READER_MODEL,JUDGE_MODEL,PROTOCOL
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.judge_terminal_label_compat_v1 import PROTOCOL as COMPAT_PROTOCOL

SCHEMA='sdk20-plain-qa-run-v1'
SOURCE_SCHEMA='sdk20-plain-qa-sources-v1'
ARMS=('baseline_sdk20','fact_types_sdk20')
SAFE=re.compile(r'^[A-Za-z0-9_-]+$')
VAULT=Path(r'C:\Users\86139\Desktop\jev\experiments\live\credential_store.py')
POLICY={'arms':list(ARMS),'slots_per_arm':1540,'firstpass_slots':3080,'conversation_workers':8,
    'qa_threads_per_conversation':4,'evaluation_protocol':PROTOCOL,'reader_model':READER_MODEL,'judge_model':JUDGE_MODEL,
    'transport_retries':2,'model_output_json':False,'facts':20,'selection':'frozen_exact_order_no_new_search_or_typing',
    'reader_and_f1':'persist_before_judge','cache_roots':[],'cache':'same_phase_same_conversation_exact_request_only',
    'recovery':'once_only_after_whole_firstpass_terminal_all_Reader_or_strict_Judge_failures_including_compat_valid',
    'successful_reader':'inherit_exact_artifact_result_without_new_Reader',
    'valid_yes_and_no':'never_correctness_retry','unknown':'hard_reject_missing_or_unresolved_intents_pending_offline_review',
    'recovered_Reader_first_Judge_failure':'compat_or_missing_no_further_phase',
    'judge_compat_protocol':COMPAT_PROTOCOL,
    'final_priority':['firstpass_strict','retry_strict','firstpass_eligible_compat','retry_eligible_compat']}
CODE=('tools/run_sdk20_plain_qa_v1.py','tools/sdk20_plain_qa_sources_v1.py','tools/run_sdk20_judge_retest_v2.py',
    'tools/run_mem0_regroup_plain_qa_v1.py','tools/replay_plain_judge_compat_v1.py','tools/summarize_mem0_regroup_plain_qa_v1.py',
    'tools/summarize_sdk20_modules.py','tools/summarize_mem0_sparse1540.py','tools/summarize_pilot400.py',
    'tools/run_sdk20_modules.py','tools/run_pilot400_experiments_v2.py','tools/check_experiment_health.py',
    'jev_mem/evaluation_plain_text_v1.py','jev_mem/plain_text_api_v1.py','jev_mem/evaluation.py',
    'jev_mem/judge_terminal_label_compat_v1.py','jev_mem/providers.py','jev_mem/sdk20_cached_api.py','jev_mem/paced_embedding_api.py')


def source_validate(path):
    from tools.sdk20_plain_qa_sources_v1 import validate
    return validate(Path(path))


def job_key(j):return j['conversation'],j['method'],j['query_id']


def check_jobs(jobs):
    keys=[]
    for j in jobs:
        if (type(j.get('conversation'))is not int or j['conversation'] not in range(10) or j.get('method') not in ARMS
                or not isinstance(j.get('query_id'),str) or not SAFE.fullmatch(j['query_id'])
                or j.get('job_id')!=j['method']+'/'+j['query_id'] or j.get('status')!='ready'):
            raise ValueError('Invalid fixed source job identity')
        if (not isinstance(j.get('question'),str) or not j['question'].strip() or type(j.get('reference')) not in (str,int)
                or type(j.get('category'))is not int or j['category'] not in (1,2,3,4)):
            raise ValueError('Original question/reference/category types required')
        if not isinstance(j.get('reference_date'),str) or date.fromisoformat(j['reference_date']).isoformat()!=j['reference_date']:
            raise ValueError('Original conversation reference date required')
        ids=j['selected_ids'];facts=j['selected_facts']
        if len(ids)!=20 or len(set(ids))!=20 or [f['id'] for f in facts]!=ids:
            raise ValueError('Exact ordered20 unique selected facts required')
        if any(set(f)!={'id','text','observation_date'} for f in facts):raise ValueError('Only frozen Reader fact input fields permitted')
        reader_input(j['question'],facts,reference_date=j['reference_date'])
        keys.append(job_key(j))
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate job')


def stage_files(run,j):
    return {s:Path(run)/f"conv_{j['conversation']}"/s/j['method']/(j['query_id']+'.json') for s in ('reader','judge','compat','qa')}


def identity(j,m):
    return {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],'phase_id':m['phase_id'],
        **{k:j[k] for k in ('job_id','conversation','method','query_id','question','reference','category','reference_date','selected_ids')},
        'source_job_sha256':digest(j),'source_manifest':m['source_manifest']}


def stage_context(j,m,stage):
    return {'architecture':SCHEMA,'conversation':j['conversation'],'phase':'qa','phase_id':m['phase_id'],
        'query_id':j['query_id'],'method':j['method'],'component':stage}


def _new_manifest(run,source_path,source,*,jobs,phase_kind,parent=None):
    run=Path(run).resolve()
    if run.parent!=(ROOT/'runs').resolve() or run.exists():raise FileExistsError('Fresh direct runs namespace required')
    check_jobs(jobs);_,provenance=verified_official_metric()
    if source.get('official_metric_provenance')!=provenance:raise ValueError('Source official scorer differs')
    value={'schema':SCHEMA,'created_utc':now(),'policy':POLICY,'source_manifest':pointer(source_path),
        'phase_kind':phase_kind,'phase_id':'sdk20_plain_qa_'+phase_kind+'_v1','jobs':jobs,'parent':parent,
        'official_metric_provenance':provenance,'code_sha256':{p:sha(ROOT/p) for p in CODE},
        'model_calls_during_prepare':0,'cache_roots':[]}
    run.mkdir(parents=True,exist_ok=False)
    publish_json(run/'run_manifest.json',{**value,'protocol_sha256':digest(value)})
    return {'run_dir':str(run),'phase_kind':phase_kind,'jobs':len(jobs),'protocol_sha256':digest(value),'model_calls':0}


def prepare(run_dir,source_path):
    source=source_validate(source_path)
    if source.get('schema')!=SOURCE_SCHEMA:raise ValueError('Wrong source schema')
    jobs=copy.deepcopy(source['jobs'])
    if len(jobs)!=3080 or Counter(j['method'] for j in jobs)!={a:1540 for a in ARMS}:raise ValueError('Both1540-slot arms required')
    return _new_manifest(run_dir,source_path,source,jobs=jobs,phase_kind='firstpass')


def recovery_jobs(parent,out):
    if parent['phase_kind']!='firstpass':raise ValueError('Only one postbatch technical recovery is allowed')
    if any(r.get('local_errors') for r in out.get('outcomes',[])):
        raise ValueError('Local/controller errors require an offline audit before any recovery')
    originals={job_key(j):j for j in parent['jobs']};jobs=[]
    for row in out['results']:
        if any(row[s] is None for s in ('reader','judge','compat','qa')):
            raise ValueError('Missing/unknown stage requires an offline audit, not redispatch')
        j=originals[job_key(row)];reader=bound(row['reader']);judge=bound(row['judge']);compat=bound(row['compat'])
        if compat.get('error'):raise ValueError('Compatibility source verification error requires offline review')
        if reader['status'] not in ('complete','failed'):raise ValueError('Reader did not reach a known model outcome')
        if reader['status']=='failed':known_failure(j,parent,reader,row['reader'],'reader')
        elif judge['status']=='failed':known_failure(j,parent,judge,row['judge'],'judge')
        elif judge['status']!='complete':raise ValueError('Judge did not reach a known model outcome')
        if reader and reader['status']=='complete' and judge and judge['status']=='complete':continue
        candidate=copy.deepcopy(j)
        candidate['parent_results']={s:row[s] for s in ('reader','judge','compat','qa')}
        candidate['parent_job_sha256']=digest(j)
        candidate['reader_source']=row['reader'] if reader and reader['status']=='complete' else None
        candidate['attempt_kind']='judge_only_after_successful_reader' if candidate['reader_source'] else 'reader_retry_then_first_judge'
        jobs.append(candidate)
    check_jobs(jobs);return jobs


def known_failure(j,m,row,ref,stage):
    """Admit recorded model failures; never turn local/unknown work into a retry."""
    failure=row.get('error')or{};events=row.get('api_events',[])
    allowed={'RuntimeError','InvalidJudgeLabel'} if stage=='judge' else {'RuntimeError'}
    if failure.get('type') not in allowed or len(events)!=1:raise ValueError('Unresolved or local stage failure needs an offline audit')
    event=events[0];context=stage_context(j,m,stage)
    if event.get('context')!=context:raise ValueError('Failure event context differs')
    folder=Path(ref['path']).parents[2];api_dir=folder/'api';key=event['request_hash']
    rawpath=api_dir/'api_cache'/(key+'.json');intent=api_dir/'request_intents'/(key+'.json')
    if Path(event['cache_path']).resolve()!=rawpath.resolve() or not intent.is_file():raise ValueError('Failure source intent/raw location differs')
    request=read(intent)
    if request['request_hash']!=key or request['stage']!=event['stage'] or request['retries']!=2:
        raise ValueError('Failure request intent differs')
    ledger=[json.loads(line) for line in shared_read(api_dir/'api_calls.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
    physical=[r for r in ledger if r.get('request_hash')==key and r.get('stage')==event['stage'] and r.get('cache_hit')is not True]
    if (not physical or len(physical)>3 or any(r.get('context')!=request['context'] for r in physical)
            or [r.get('attempt') for r in physical]!=list(range(1,len(physical)+1))):
        raise ValueError('Failed request has no complete bounded physical sequence')
    last=physical[-1].get('status')
    if rawpath.exists():
        raw=read(rawpath)
        if (event.get('error_type')is not None or event.get('cache_sha256')!=sha(rawpath)
                or raw['request']!=request['signature'] or last!=200):raise ValueError('Raw generation failure provenance differs')
        if failure['type']!='InvalidJudgeLabel' and not failure.get('message','').startswith(('Empty generation','Generation missing or incomplete:')):
            raise ValueError('Successful raw with a local processing failure requires offline repair')
    elif not(type(last)is int and 400<=last<600 and event.get('error_type')is not None):
        raise ValueError('Unknown/no-raw or transport-only outcome requires offline audit before a new request')
    # Reuse the frozen generic stage replayer to reconstruct the exact request
    # from this job and verify provider/model/operation/raw/intent/consumer chain.
    indexed=defaultdict(list)
    for event_row in ledger:
        indexed[event_row['request_hash'],event_row['stage'],fingerprint(event_row['context'])].append(event_row)
    expected={**identity(j,m),'stage':stage};snapshot=Snapshot()
    if stage=='reader':
        official,_=verified_official_metric();callback=lambda api:reader_call(api,j,official)
    else:
        reader_path=folder/'reader'/j['method']/(j['query_id']+'.json');reader=verify(reader_path)
        expected['reader_sha256']=sha(reader_path)
        callback=lambda api:judge_answer(api,j['question'],j['reference'],reader['result']['prediction']['answer'])
    audit_stage(snapshot,Path(ref['path']),expected,api_dir,context,indexed,callback)
    if snapshot.unchanged():raise ValueError('Failed stage source changed while checking recovery')


def prepare_recovery(firstpass_run):
    firstpass_run=Path(firstpass_run).resolve();m,out=terminal(firstpass_run)
    jobs=recovery_jobs(m,out)
    parent={'run_dir':str(firstpass_run),'manifest':pointer(firstpass_run/'run_manifest.json'),'outcome':pointer(firstpass_run/'run_outcome.json')}
    return _new_manifest(firstpass_run.with_name(firstpass_run.name+'_recovery_v1'),m['source_manifest']['path'],
        read(m['source_manifest']['path']),jobs=jobs,phase_kind='technical_recovery',parent=parent)


def manifest(run_dir,*,_source=None):
    run=Path(run_dir).resolve();m=read(run/'run_manifest.json')
    if (m.get('schema')!=SCHEMA or m.get('policy')!=POLICY or m.get('cache_roots')!=[]
            or m['protocol_sha256']!=digest({k:v for k,v in m.items() if k!='protocol_sha256'})):
        raise ValueError('Run protocol/policy differs')
    if m['code_sha256']!={p:sha(ROOT/p) for p in CODE}:raise ValueError('Frozen run code changed')
    bound(m['source_manifest']);source=_source if _source is not None else source_validate(m['source_manifest']['path'])
    _,provenance=verified_official_metric()
    if provenance!=m['official_metric_provenance'] or provenance!=source['official_metric_provenance']:
        raise ValueError('Official metric provenance changed')
    if m['phase_kind']=='firstpass':
        if m['jobs']!=source['jobs'] or m['parent'] is not None:raise ValueError('Firstpass fixed inputs changed')
    elif m['phase_kind']=='technical_recovery':
        p=m['parent'];bound(p['manifest']);bound(p['outcome'])
        if run!=Path(p['run_dir']).with_name(Path(p['run_dir']).name+'_recovery_v1'):raise ValueError('Only one fixed recovery namespace allowed')
        pm,po=terminal(p['run_dir'],_source=source)
        if m['jobs']!=recovery_jobs(pm,po) or m['source_manifest']!=pm['source_manifest']:
            raise ValueError('Technical recovery population differs')
    else:raise ValueError('Unknown phase')
    check_jobs(m['jobs']);return m


def terminal(run_dir,*,_source=None):
    run=Path(run_dir).resolve();m=manifest(run,_source=_source);out=verify(run/'run_outcome.json')
    if (out.get('terminal')is not True or out.get('protocol_sha256')!=m['protocol_sha256']
            or out.get('total')!=len(m['jobs']) or len(out.get('results',[]))!=len(m['jobs'])):
        raise ValueError('Whole phase has not reached its recorded terminal population')
    jobs={job_key(j):j for j in m['jobs']}
    if {job_key(r) for r in out['results']}!=set(jobs):raise ValueError('Terminal slot population differs')
    for row in out['results']:
        j=jobs[job_key(row)];expected=identity(j,m)
        if row['job_id']!=j['job_id']:raise ValueError('Terminal job identity differs')
        for stage,path in stage_files(run,j).items():
            if row[stage] is None:
                if path.exists():raise ValueError('Existing stage cannot be recorded unknown')
                continue
            if row[stage]!=pointer(path):raise ValueError('Terminal stage pointer refers to another slot')
            value=verify(path)
            if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal stage input identity differs')
    return m,out


def reader_call(api,j,official):
    result=read_answer(api,j['question'],j['reference'],j['selected_facts'],reference_date=j['reference_date'])
    return {**result,'metrics':{'official_locomo_f1':official(result['prediction']['answer'],j['reference'],j['category']),
        'custom_token_f1':result['token_f1'],'custom_exact_match':result['exact_match']}}


def compatibility(run,j,m,reader,judge,path,api=None):
    ident=identity(j,m);strict_path=stage_files(run,j)['judge'];interpretation=evidence=failure=None;status='unavailable'
    if reader['status']=='complete' and (judge['status']=='complete' or (judge.get('error')or{}).get('type')=='InvalidJudgeLabel'):
        try:
            api_dir=Path(run)/f"conv_{j['conversation']}"/'api';ledger_path=api_dir/'api_calls.jsonl'
            data=shared_read(ledger_path);data=data[:data.rfind(b'\n')+1];indexed=defaultdict(list)
            for raw in data.decode('utf-8-sig').splitlines():
                if raw.strip():
                    row=json.loads(raw);indexed[row['request_hash'],row['stage'],fingerprint(row['context'])].append(row)
            replay=replay_source(api_dir,judge,stage_context(j,m,'judge'),indexed,question=j['question'],reference=j['reference'],
                reader_text=reader['result']['prediction']['answer'])
            interpretation=replay['interpretation'];status=interpretation['status'];evidence=replay['source_evidence']
        except Exception as exc:failure=safe_error(exc,api)
    save_new(path,{**ident,'stage':'compat','status':status,'interpretation':interpretation,'strict':pointer(strict_path),
        'source_evidence':evidence,'error':failure,'model_calls':0,'raw_or_strict_modified':False})
    return verify(path)


def one(run,j,m,api,official,initial_error=None):
    paths=stage_files(run,j);common=identity(j,m)
    # A controller rerun is never an implicit attempt; stage_once also retains
    # any interrupted intent and its exact prior success/failure if present.
    if paths['qa'].exists():raise RuntimeError('Existing terminal QA retained; no implicit replay')
    reader_api=api.scoped(**stage_context(j,m,'reader')) if api else None
    if j.get('reader_source'):
        source=bound(j['reader_source']);verify(j['reader_source']['path'])
        if source['status']!='complete':raise ValueError('Only a successful Reader may be inherited')
        save_new(paths['reader'],{**common,'stage':'reader','status':'complete','result':copy.deepcopy(source['result']),
            'error':None,'api_events':[],'raw_invalid_label':None,'inherited_from':j['reader_source'],'model_calls':0})
        reader=verify(paths['reader'])
    elif initial_error is not None:
        save_new(paths['reader'],{**common,'stage':'reader','status':'not_run','result':None,'api_events':[],
            'raw_invalid_label':None,'error':{'type':'APIInitializationFailure','message':'No dispatch; initialization raised '+type(initial_error).__name__}})
        reader=verify(paths['reader'])
    else:
        def call_reader():
            if initial_error is not None:raise RuntimeError('API initialization failed: '+type(initial_error).__name__)
            return reader_call(reader_api,j,official)
        reader=stage_once(paths['reader'],{**common,'stage':'reader'},call_reader,reader_api)
    expected={**common,'stage':'judge','reader_sha256':sha(paths['reader'])}
    if reader['status']=='complete' and initial_error is not None:
        save_new(paths['judge'],{**expected,'status':'not_run','result':None,'api_events':[],'raw_invalid_label':None,
            'error':{'type':'APIInitializationFailure','message':'No Judge dispatch; initialization raised '+type(initial_error).__name__}})
        judge=verify(paths['judge'])
    elif reader['status']=='complete':
        judge_api=api.scoped(**stage_context(j,m,'judge')) if api else None
        def call_judge():
            if initial_error is not None:raise RuntimeError('API initialization failed: '+type(initial_error).__name__)
            return judge_answer(judge_api,j['question'],j['reference'],reader['result']['prediction']['answer'])
        judge=stage_once(paths['judge'],expected,call_judge,judge_api)
    else:
        save_new(paths['judge'],{**expected,'status':'not_run','result':None,'api_events':[],'raw_invalid_label':None,
            'error':{'type':'ReaderFailed','message':'No Judge dispatch'}})
        judge=verify(paths['judge'])
    compat=compatibility(run,j,m,reader,judge,paths['compat'],api)
    if m['phase_kind']=='technical_recovery' and j.get('reader_source'):
        prior=j['parent_results'];first_judge=bound(prior['judge']) if prior['judge'] else None
        first_compat=bound(prior['compat']) if prior['compat'] else None
        resolved=resolve_pair(first_judge,first_compat,judge,compat)
    elif m['phase_kind']=='technical_recovery':resolved=resolve_pair(None,None,judge,compat)
    else:resolved=resolve_pair(judge,compat)
    save_new(paths['qa'],{**common,'stage':'qa','status':'complete' if reader['status']==judge['status']=='complete' else 'failed',
        'reader_status':reader['status'],'judge_status':judge['status'],'reader':pointer(paths['reader']),
        'judge':pointer(paths['judge']),'compat':pointer(paths['compat']),
        'reader_metrics':reader['result']['metrics'] if reader['status']=='complete' else None,
        'judge_correct':judge['result']['correct'] if judge['status']=='complete' else None,'resolved_judge':resolved})


def run(run_dir,*,api_factory=None):
    run=Path(run_dir).resolve();m=manifest(run)
    if (run/'run_outcome.json').exists():return terminal(run)[1]
    if (run/'execution_intent.json').exists():raise RuntimeError('Prior phase execution is retained; no implicit restart')
    if api_factory is None and not VAULT.is_file():raise RuntimeError('DPAPI vault module required; environment fallback disabled')
    official,provenance=verified_official_metric()
    durable_intent(run/'execution_intent.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'created_utc':now(),'pid':os.getpid()})
    def worker(ci,jobs):
        api=None;initial_error=None
        try:
            if api_factory is None:api=PlainTextEvaluationAPI(run/f'conv_{ci}'/'api',cache_roots=(),retries=2,
                context={'architecture':SCHEMA,'conversation':ci,'phase':'qa','phase_id':m['phase_id']})
            else:api=api_factory(run/f'conv_{ci}'/'api',ci,m)
        except Exception as exc:initial_error=exc
        errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(one,run,j,m,api,official,initial_error):j for j in jobs}
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:errors.append({'job_id':futures[f]['job_id'],'error':safe_error(exc,api)})
        return {'conversation':ci,'jobs':len(jobs),'terminal':True,'local_errors':errors}
    groups={ci:[j for j in m['jobs'] if j['conversation']==ci] for ci in range(10)};outcomes=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(worker,ci,jobs):ci for ci,jobs in groups.items() if jobs}
        for f in as_completed(futures):
            try:outcomes.append(f.result())
            except Exception as exc:outcomes.append({'conversation':futures[f],'terminal':True,'local_errors':[safe_error(exc)]})
            publish_json(run/'run_status.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],
                'phase_id':m['phase_id'],'terminal':False,'pid':os.getpid(),'updated_utc':now(),'conversations_finished':len(outcomes),'outcomes':outcomes})
    results=[];counts={s:Counter() for s in ('reader','judge','compat','qa')}
    for j in m['jobs']:
        row={k:j[k] for k in ('job_id','conversation','method','query_id')}
        for s,path in stage_files(run,j).items():
            row[s]=pointer(path) if path.exists() else None
            counts[s][verify(path)['status'] if path.exists() else 'unknown']+=1
        results.append(row)
    final={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],'phase_id':m['phase_id'],
        'terminal':True,'finished_utc':now(),'total':len(m['jobs']),'counts':{s:dict(v) for s,v in counts.items()},
        'outcomes':outcomes,'results':results,'native_clients':0,'external_cache_roots':[]}
    save_new(run/'run_outcome.json',final)
    publish_json(run/'run_status.json',{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'phase_kind':m['phase_kind'],
        'phase_id':m['phase_id'],'terminal':True,'pid':os.getpid(),'updated_utc':now(),'conversations_finished':len(outcomes),'counts':final['counts']})
    return verify(run/'run_outcome.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    a=sub.add_parser('prepare');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--source',type=Path,required=True)
    a=sub.add_parser('recover');a.add_argument('--firstpass-run',type=Path,required=True)
    for name in ('check','run'):
        a=sub.add_parser(name);a.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='prepare':result=prepare(a.run_dir,a.source)
    elif a.phase=='recover':result=prepare_recovery(a.firstpass_run)
    elif a.phase=='check':m=manifest(a.run_dir);result={'jobs':len(m['jobs']),'phase_kind':m['phase_kind'],'protocol_sha256':m['protocol_sha256'],'model_calls':0}
    else:
        out=run(a.run_dir);result={k:out[k] for k in ('phase_kind','total','terminal','counts')}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
