"""Bounded format-only recovery of the55 sealed SDK20-module failures.

prepare is strictly offline; execute alone reads the existing DPAPI vault.
Old code, raw responses and results are never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os,sys,threading,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_sdk20_modules as old
from tools.run_pilot400_experiments_v2 import publish_json
from jev_mem.evaluation import answer_and_judge
from jev_mem.providers import ModelAPI,load_credentials
from jev_mem.sdk20_cached_api import CachedLegacyAPI,check_legacy_request,digest as request_digest

SCHEMA='sdk20-modules-format-recovery-v1'
ARMS=old.ARMS
SOURCE_SUMMARY=ROOT/'research/sdk20_modules_results_2026-09-28_v1/summary.json'
SOURCE_SUMMARY_SHA='a66827f76d87f690a31e561f4ddc03cb43bb40802cfb04cd06fb1c81c1a9d776'
POLICY_FILE=ROOT/'research/sdk20_uniform_recovery_policy_2026-09-28.md'
EXPECTED_FAILURES={'baseline_sdk20':15,'fact_types_sdk20':23,'preseg_sdk20':17}
CODE=('tools/recover_sdk20_modules.py','jev_mem/evaluation.py','jev_mem/providers.py',
      'jev_mem/sdk20_cached_api.py','tools/run_sdk20_modules.py','tools/run_pilot400_experiments_v2.py')
POLICY={'logical_attempts_per_stage':3,'http_retries_per_logical_attempt':2,
    'conversation_workers':8,'qa_workers_per_conversation':4,
    'retry_condition':'only JSONDecodeError or original explicit Reader/Judge protocol ValueError with a successful raw response',
    'reader_lock':'first protocol-valid Reader permanently fixes response and Judge payload',
    'judge_lock':'first protocol-valid Judge is final regardless of correctness',
    'technical_policy':'network retry exhaustion, unresolved intent, incomplete generation and unknown errors terminate consumer; no next logical attempt',
    'cache_policy':'fresh globally shared per-stage per-logical-attempt cache, empty inherited cache roots',
    'response_schema':False,'reader_protocol':'brief_answer_local_evidence_ids_v2'}
FORMAT_ERRORS={'Reader did not return an answer string','Reader cited IDs outside its selected facts',
               'Judge did not return boolean correctness','Judge returned invalid completeness or reason'}
_STATUS_LOCK=threading.Lock()

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pointer(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify_pointer(p):
    if sha(p['path'])!=p['sha256']:raise ValueError('Source pointer changed: '+p['path'])
    return read(p['path'])
def save_new(p,v):
    p=Path(p)
    if p.exists():raise FileExistsError('Refusing to replace recovery artifact: '+str(p))
    publish_json(p,old.sealed(v))
def verify(p):return old.verify(p)
def fmt_error(exc):return isinstance(exc,json.JSONDecodeError) or (type(exc)is ValueError and str(exc) in FORMAT_ERRORS)
def err(exc):return {'type':type(exc).__name__,'message':str(exc)[:2000]}
def evaluate(api,job):
    return answer_and_judge(api,job['question'],job['reference'],deepcopy(job['selected_facts']),reference_date=job['reference_date'])

class CapturedRequest(Exception):
    def __init__(self,signature):self.signature=signature

class ProbeAPI:
    """Original generation parsing and evaluation, backed only by given responses."""
    generate=ModelAPI.generate
    def __init__(self,job,*,reader=None,judge=None,capture_reader=False):
        self.job=job;self.reader=reader;self.judge=judge;self.capture_reader=capture_reader
        self.seen=[]
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified'):
        check_legacy_request(model,body,stage)
        sig={'provider':provider,'model':model,'operation':operation,'body':body};self.seen.append(stage)
        if stage=='qa_reader':
            if self.capture_reader:raise CapturedRequest(sig)
            if sig!=self.job['expected_reader_signature']:raise ValueError('Recovery Reader payload drift')
            return deepcopy(self.reader)
        if stage!='strict_judge':raise ValueError('Unexpected recovery stage')
        if self.judge is None:raise CapturedRequest(sig)
        return deepcopy(self.judge)

def expected_reader(job):
    try:evaluate(ProbeAPI(job,capture_reader=True),job)
    except CapturedRequest as ex:return ex.signature
    raise AssertionError('Reader request not captured')

def probe_reader(job,raw_response):
    api=ProbeAPI(job,reader=raw_response)
    try:evaluate(api,job)
    except CapturedRequest as ex:
        if api.seen!=['qa_reader','strict_judge']:raise ValueError('Unexpected Reader probe path')
        return ex.signature
    raise AssertionError('Judge request not captured')

def probe_pair(job,reader_response,judge_response):
    api=ProbeAPI(job,reader=reader_response,judge=judge_response)
    result=evaluate(api,job)
    if api.seen!=['qa_reader','strict_judge']:raise ValueError('Unexpected Reader/Judge probe path')
    return result

def _old_raw(source_run,ci,arm,row,signature):
    key=request_digest(signature)
    if arm!=ARMS[0]:
        events=row['api_events']
        if len(events)!=1 or events[0]['stage']!='qa_reader' or events[0]['request_hash']!=key or events[0]['error_type']is not None:
            raise ValueError('Old failed QA lacks one exact successful Reader raw event')
        p=Path(events[0]['cache_path'])
        if p.resolve()!=source_run/f'conv_{ci}/api/api_cache/{key}.json' or sha(p)!=events[0]['cache_sha256']:
            raise ValueError('Old Reader event raw binding differs')
    else:
        candidates=[r/(key+'.json') for r in old.cache_roots(ci)[:2] if (r/(key+'.json')).exists()]
        if not candidates:raise ValueError('Historical baseline failed Reader raw is unavailable')
        p=candidates[0]
    raw=read(p)
    if raw.get('request')!=signature or raw.get('stage')!='qa_reader':raise ValueError('Old raw signature/stage differs')
    return pointer(p),raw['response']

def prepare(run_dir,source_summary=SOURCE_SUMMARY):
    run=Path(run_dir).resolve();sp=Path(source_summary).resolve()
    if run.exists():raise FileExistsError('Recovery requires a new directory')
    if sha(sp)!=SOURCE_SUMMARY_SHA:raise ValueError('Source is not the sealed complete SDK20-module summary')
    s=read(sp);source_run=Path(s['run_dir']).resolve();m=old.manifest(source_run)
    if run==source_run or source_run in run.parents or run in source_run.parents:raise ValueError('Recovery must be separate from original run')
    if (s['expected_slots']!=4620 or s['cohort_per_arm']!=1540 or s['all_terminal_slots_valid']is not True
        or s['integrity_errors'] or s['protocol_sha256']!=m['protocol_sha256']):raise ValueError('Source summary is not complete and validated')
    for p,h in s['input_hashes'].items():
        if sha(p)!=h:raise ValueError('Sealed source changed: '+p)
    failed=[r for r in s['outcomes'] if r['status']=='failed']
    if Counter(r['method'] for r in failed)!=EXPECTED_FAILURES:raise ValueError('Failure cohort differs from all55')
    jobs=[];source_hashes={**s['input_hashes'],str(sp):sha(sp)}
    for out in failed:
        row=verify_pointer(out['artifact']);old.verify(out['artifact']['path']);ci=row['conversation'];arm=row['method'];qid=row['query_id']
        if row['status']!='failed' or row['result']is not None or row['error']!={'type':'ValueError','message':'Reader did not return an answer string'}:
            raise ValueError('Source is not the declared nonstring Reader failure')
        selection_path=source_run/f'conv_{ci}/selection/{arm}/{qid}.json';sel=old.verify(selection_path)
        kind='preseg' if arm==ARMS[2] else 'session';bank,bhash=old.get_bank(source_run,ci,kind,m)
        if sel['status']!='complete' or sel['selected_ids']!=row['selected_ids'] or sel['bank_sha256']!=bhash:
            raise ValueError('Failed QA original selection binding differs')
        byid={f['id']:f for f in bank['facts']}
        if len(row['selected_ids'])!=20 or len(set(row['selected_ids']))!=20:raise ValueError('Recovery requires exact20 original facts')
        job={'schema':SCHEMA,'conversation':ci,'method':arm,'query_id':qid,'category':row['category'],
            'question':row['question'],'reference':row['reference'],'reference_date':row['reference_date'],
            'selected_ids':list(row['selected_ids']),
            'selected_facts':[{k:byid[i].get(k) for k in ('id','text','observation_date')} for i in row['selected_ids']],
            'source_qa':out['artifact'],'source_selection':pointer(selection_path),
            'source_bank':pointer(source_run/f'conv_{ci}/{kind}/bank.json'),'old_row':row}
        sig=expected_reader(job);job['expected_reader_signature']=sig;job['expected_reader_hash']=request_digest(sig)
        oldraw,response=_old_raw(source_run,ci,arm,row,sig);job['old_raw']=oldraw
        try:probe_reader(job,response)
        except ValueError as ex:
            if type(ex)is not ValueError or str(ex)!='Reader did not return an answer string':raise
        else:raise ValueError('Original Reader did not reproduce the exact nonstring failure')
        source_hashes[oldraw['path']]=oldraw['sha256'];jobs.append(job)
    if len({(j['method'],j['query_id']) for j in jobs})!=55:raise ValueError('Duplicate recovery consumers')
    run.mkdir(parents=True,exist_ok=False);job_pointers=[]
    # Jobs bind the source protocol; the recovery protocol seals their file hashes.
    for job in jobs:
        job['source_protocol_sha256']=m['protocol_sha256'];p=run/'jobs'/job['method']/(job['query_id']+'.json')
        save_new(p,job);job_pointers.append(pointer(p))
    config={'schema':SCHEMA,'created_utc':now(),'source_summary':pointer(sp),'source_run_manifest':pointer(source_run/'run_manifest.json'),
        'source_protocol_sha256':m['protocol_sha256'],'source_input_sha256':source_hashes,
        'source_failure_counts':EXPECTED_FAILURES,'source_failed_count':55,'jobs':job_pointers,
        'policy':POLICY,'policy_document':pointer(POLICY_FILE),'code_sha256':{p:sha(ROOT/p) for p in CODE},'model_calls_at_prepare':0}
    publish_json(run/'run_manifest.json',{**config,'protocol_sha256':old.digest(config)})
    return {'prepared':str(run),'protocol_sha256':old.digest(config),'jobs':55,'model_calls':0}

def manifest(run_dir,*,verify_sources=True):
    run=Path(run_dir).resolve();m=read(run/'run_manifest.json')
    if m.get('schema')!=SCHEMA or m.get('policy')!=POLICY or m.get('protocol_sha256')!=old.digest({k:v for k,v in m.items() if k!='protocol_sha256'}):
        raise ValueError('Recovery protocol differs')
    if m['source_summary']['sha256']!=SOURCE_SUMMARY_SHA:raise ValueError('Unapproved source summary')
    if sha(m['policy_document']['path'])!=m['policy_document']['sha256']:raise ValueError('Frozen recovery policy document changed')
    for p,h in m['code_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen recovery code changed: '+p)
    if verify_sources:
        for p,h in m['source_input_sha256'].items():
            if sha(p)!=h:raise ValueError('Recovery source changed: '+p)
    for p in m['jobs']:verify_pointer(p);verify(p['path'])
    return m

def identity(job,m):
    return {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'job_sha256':job['_sha256'],
        'conversation':job['conversation'],'method':job['method'],'query_id':job['query_id']}

def _raw_response(ptr,signature):
    raw=verify_pointer(ptr)
    if raw.get('request')!=signature:raise ValueError('Attempt raw request signature changed')
    if raw['response'].get('modelVersion',raw['response'].get('model')) not in (None,signature['model']):
        raise ValueError('Attempt resolved model differs')
    return raw['response']

def stage_status(folder,job,stage,attempt,status,**extra):
    with _STATUS_LOCK:
        publish_json(folder/'status.json',{'schema':SCHEMA,'updated_utc':now(),'pid':os.getpid(),
            'conversation':job['conversation'],'method':job['method'],'query_id':job['query_id'],
            'stage':stage,'logical_attempt':attempt,'status':status,**extra})

def check_record(record,job,m,stage,n,signature,reader_response=None):
    if any(record.get(k)!=v for k,v in identity(job,m).items()) or record.get('stage')!=stage or record.get('logical_attempt')!=n:
        raise ValueError('Existing consumer attempt identity changed')
    if record.get('signature')!=signature or record.get('request_hash')!=request_digest(signature):raise ValueError('Existing attempt request changed')
    status=record.get('status')
    if status=='technical_failed':return record
    if status not in ('format_failed','valid') or not record.get('raw'):raise ValueError('Existing attempt has invalid outcome')
    response=_raw_response(record['raw'],signature)
    try:value=probe_reader(job,response) if stage=='qa_reader' else probe_pair(job,reader_response,response)
    except Exception as ex:
        if status!='format_failed' or not fmt_error(ex) or record.get('error')!=err(ex):raise ValueError('Existing format outcome cannot replay') from ex
    else:
        if status!='valid' or record.get('error')is not None:raise ValueError('Existing valid outcome differs')
        key='judge_signature' if stage=='qa_reader' else 'result'
        if record.get(key)!=value:raise ValueError('Existing valid parsed payload differs')
    return record

def run_attempt(run,folder,job,m,stage,n,signature,*,reader_response=None,api_factory=CachedLegacyAPI,keys=None):
    path=folder/stage/f'attempt_{n}.json'
    if path.exists():return check_record(verify(path),job,m,stage,n,signature,reader_response),pointer(path)
    api_dir=Path(run)/'transport'/stage/f'attempt_{n}'
    context={'conversation':job['conversation'],'method':job['method'],'query_id':job['query_id'],
        'phase':'sdk20_recovery','stage':stage,'logical_attempt':n}
    record={**identity(job,m),'stage':stage,'logical_attempt':n,'signature':signature,
        'request_hash':request_digest(signature),'api_dir':str(api_dir.resolve()),'started_utc':now(),
        'status':None,'error':None,'raw':None,'transport_intent':None,'api_events':[],
        'judge_signature':None,'result':None}
    stage_status(folder,job,stage,n,'running')
    api=api_factory(api_dir,cache_roots=(),retries=2,context=context,keys=keys)
    try:
        response=api.request(signature['provider'],signature['model'],deepcopy(signature['body']),operation=signature['operation'],stage=stage)
        target=api_dir/'api_cache'/(record['request_hash']+'.json');record['raw']=pointer(target)
        if _raw_response(record['raw'],signature)!=response:raise ValueError('Returned response differs from durable cache')
        try:value=probe_reader(job,response) if stage=='qa_reader' else probe_pair(job,reader_response,response)
        except Exception as ex:
            if not fmt_error(ex):raise
            record['status']='format_failed';record['error']=err(ex)
        else:
            record['status']='valid';record['judge_signature' if stage=='qa_reader' else 'result']=value
    except Exception as ex:
        record['status']='technical_failed';record['error']=err(ex)
    finally:
        record['api_events']=deepcopy(api.events)
        target=api_dir/'api_cache'/(record['request_hash']+'.json');intent=api_dir/'request_intents'/(record['request_hash']+'.json')
        if target.exists():record['raw']=pointer(target)
        if intent.exists():record['transport_intent']=pointer(intent)
        record['finished_utc']=now()
    save_new(path,record);stage_status(folder,job,stage,n,record['status'])
    return record,pointer(path)

def execute_consumer(run,job,m,*,api_factory=CachedLegacyAPI,keys=None):
    run=Path(run).resolve();folder=run/'consumers'/job['method']/job['query_id'];rp=folder/'result.json'
    if rp.exists():
        result=verify(rp)
        if any(result.get(k)!=v for k,v in identity(job,m).items()):raise ValueError('Existing result identity differs')
        for p in result['attempt_records']:verify_pointer(p)
        if result['reader_lock']:verify_pointer(result['reader_lock'])
        return result
    attempts=[];lock=None;result=None;failure=None;reason=None
    expected=expected_reader(job)
    if expected!=job['expected_reader_signature'] or request_digest(expected)!=job['expected_reader_hash']:raise ValueError('Job Reader request reconstruction changed')
    for n in range(1,4):
        row,ptr=run_attempt(run,folder,job,m,'qa_reader',n,expected,api_factory=api_factory,keys=keys);attempts.append(ptr)
        if row['status']=='valid':
            lp=folder/'reader_lock.json';value={**identity(job,m),'logical_attempt':n,'attempt':ptr,'raw':row['raw'],
                'expected_reader_signature':expected,'judge_signature':row['judge_signature']}
            if lp.exists():
                existing=verify(lp)
                if {k:v for k,v in existing.items() if k!='content_sha256'}!=value:raise ValueError('First valid Reader lock differs')
            else:save_new(lp,value)
            lock=pointer(lp);reader_response=_raw_response(row['raw'],expected);judge_signature=row['judge_signature'];break
        if row['status']=='technical_failed':failure=row['error'];reason='reader_technical_failed';break
    else:failure=row['error'];reason='reader_format_attempts_exhausted'
    if lock:
        for n in range(1,4):
            row,ptr=run_attempt(run,folder,job,m,'strict_judge',n,judge_signature,reader_response=reader_response,api_factory=api_factory,keys=keys);attempts.append(ptr)
            if row['status']=='valid':result=row['result'];reason='first_valid_reader_and_judge';break
            if row['status']=='technical_failed':failure=row['error'];reason='judge_technical_failed';break
        else:failure=row['error'];reason='judge_format_attempts_exhausted'
    final={**identity(job,m),'status':'complete' if result is not None else 'failed','result':result,
        'status_reason':reason,'error':failure,'source_qa':job['source_qa'],'selected_ids':job['selected_ids'],
        'reader_lock':lock,'attempt_records':attempts,'finished_utc':now()}
    save_new(rp,final);stage_status(folder,job,'terminal',None,final['status'],status_reason=reason)
    return final

@contextmanager
def execution_lock(run):
    import msvcrt
    p=Path(run)/'execution.lock';f=p.open('a+b');f.seek(0,2)
    if f.tell()==0:f.write(b'0');f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise RuntimeError('Another recovery execute holds the run lock')
    try:yield
    finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()

def execute(run_dir):
    run=Path(run_dir).resolve()
    with execution_lock(run):
        m=manifest(run);vault=Path(r'C:\Users\86139\Desktop\jev\experiments\live\credential_store.py')
        if not vault.exists():raise RuntimeError('Existing DPAPI vault module is required; environment fallback is disabled')
        keys=load_credentials();jobs=defaultdict(list)
        for p in m['jobs']:
            job=verify(p['path']);job['_sha256']=p['sha256'];jobs[job['conversation']].append(job)
        started=now();all_results=[];status_path=run/'recovery_status.json'
        def progress(**extra):
            with _STATUS_LOCK:
                publish_json(status_path,{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'pid':os.getpid(),
                    'started_utc':started,'updated_utc':now(),'total':55,'terminal':False,
                    'completed':sum(r['status']=='complete' for r in all_results),
                    'failed':sum(r['status']=='failed' for r in all_results),'outcomes':len(all_results),**extra})
        def conv(ci,items):
            outputs=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(execute_consumer,run,j,m,keys=keys) for j in items]
                for future in as_completed(futures):
                    value=future.result();outputs.append(value);all_results.append(value);progress()
            return {'conversation':ci,'outcomes':len(outputs)}
        progress()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(conv,ci,items) for ci,items in jobs.items()]
                for f in as_completed(futures):f.result()
        except Exception as ex:progress(status='controller_failed',error=err(ex));raise
        if len(all_results)!=55:raise ValueError('Recovery consumer count does not close')
        progress(terminal=True,status='finished');return {'completed':sum(r['status']=='complete' for r in all_results),'failed':sum(r['status']=='failed' for r in all_results)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-summary',type=Path,default=SOURCE_SUMMARY)
    p=sub.add_parser('execute');p.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();value=prepare(args.run_dir,args.source_summary) if args.command=='prepare' else execute(args.run_dir)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
