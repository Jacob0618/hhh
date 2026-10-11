"""One exact JEV tag transport recovery in an independent namespace.

prepare/check are source-only. execute is an explicit, once-only dispatch with
retries=0. Original tags/intents/cache/status are never written. No other fact or
request can be recovered by this version; unknown chains and malformed raw fail
closed. Transport-attempt billing remains unknown when the provider reports none.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read
from tools.run_sdk20_modules import digest, now
from tools import prepare_gemini38_hybrid_effective40_v1 as base
from jev_mem.selection import tag_fact, COARSE_CRITERIA, FINE_CRITERIA, PROMPT_VERSION
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import CachedLegacyAPI

SCHEMA='gemini38-fact-tag-exact-recovery-v1'
FACT_ID='60d7f398-b6e7-48ef-88b3-a478318ee11c'
REQUEST_HASH='9b4b45484cc4ebeca262b730ada1d3a5a000bb7a784a11d44cfb76735ae76df5'
SOURCE_RUN=ROOT/'runs/gemini38_rebuild_v1_20261003'
PRODUCTION_NAMESPACE=ROOT/'runs/gemini38_fact_tag_recovery_v1_20261004'
CODE=('tools/recover_gemini38_fact_tag_v1.py','jev_mem/selection.py','jev_mem/providers.py','jev_mem/sdk20_cached_api.py','tools/prepare_gemini38_hybrid_effective40_v1.py','tools/prepare_gemini38_hybrid_v1.py')
POLICY={'retries':0,'one_dispatch':True,'cache_roots':[],'fact_count':1,'criteria_count':10,'model':'jev-1.13.0','stage':'jev_score','operation':'systemone'}
def require(value,message):
    if not value:raise ValueError(message)
def data(path):
    a=shared_read(Path(path));b=shared_read(Path(path));require(a==b,'Source changed while reading');return a
def pointer(path):return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(data(path)).hexdigest()}
def read(path):return json.loads(data(path).decode('utf-8-sig'))
def bound(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and pointer(ref['path'])==ref,'Source pointer changed');return read(ref['path'])
def verify(path):
    row=read(path);require(row.get('content_sha256')==digest({k:v for k,v in row.items()if k!='content_sha256'}),'Content seal changed');return row
def save_new(path,row):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8')as f:
        json.dump({**row,'content_sha256':digest(row)},f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def request_hash(signature):return hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def strict_result(result,fid):
    require(result.get('fact_id')==fid and result.get('prompt_version')==PROMPT_VERSION and result.get('criteria_evaluations')==10 and result.get('calibrated')is False,'Tag result identity/protocol differs')
    require(set(result.get('coarse_scores',{}))==set(COARSE_CRITERIA) and set(result.get('fine_scores',{}))==set(FINE_CRITERIA),'Exactly ten named criterion scores required')
    for v in (*result['coarse_scores'].values(),*result['fine_scores'].values()):
        require(type(v)in(int,float)and math.isfinite(v)and 0<=v<=1,'Finite numeric scores in[0,1] required')
class Capture:
    def score_many(self,state,criteria):
        self.signature={'provider':'jev','model':'jev-1.13.0','operation':'systemone','body':{'model':'jev-1.13.0','state':state,'questions':{k:{'type':'noul','instructions':v}for k,v in criteria.items()}}}
        return {k:0.0 for k in criteria}
def expected_request(fact,context):
    cap=Capture();expected=tag_fact({**fact,'source_context':context},cap);return cap.signature,expected['prompt_hash']
class ExactJudge:
    score_many=ModelAPI.score_many
    def __init__(self,signature,response=None,api=None):self.signature,self.response,self.api,self.calls=signature,response,api,0
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        require(self.calls==0 and stage=='jev_score' and cache and {'provider':provider,'model':model,'operation':operation,'body':body}==self.signature,'Only one exact original JEV request permitted')
        self.calls+=1
        return self.api.request(provider,model,body,operation=operation,stage=stage,cache=cache)if self.api is not None else self.response
def _json_lines(path):
    raw=data(path);require(raw.endswith(b'\n'),'Incomplete API ledger tail');return [json.loads(x)for x in raw.splitlines()if x.strip()]
def _ended(row):
    require(isinstance(row.get('timestamp_utc'),str),'Attempt has no ended log timestamp')
    require(datetime.fromisoformat(row['timestamp_utc']).tzinfo is not None,'Attempt timestamp has no timezone')
    v=row.get('latency_ms');require(type(v)in(int,float)and math.isfinite(v)and v>=0,'Attempt latency invalid')
def _stage_access(run,allow_retrieve=False):
    lock=Path(run)/'retrieval/active_stage.lock'
    if lock.exists():
        active=read(lock)
        require(allow_retrieve and active.get('stage')=='retrieve'and type(active.get('pid'))is int and active['pid']>0,'Recovery execution requires idle; validation only permits retrieve-stage read access')

def firstpass_snapshot(run,review_path,*,allow_retrieve=False):
    """Verify the entire1637 firstpass, not just the convenient failed row."""
    run=Path(run).resolve();review=read(review_path);rp=pointer(review_path)
    require(review.get('schema')=='gemini38-tag-firstpass-source-seal-61-v1' and review.get('status')=='passed_terminal_firstpass_with_one_bounded_transport_failure','Formal terminal firstpass audit required')
    require(Path(review['source_run']).resolve()==run and review['stage_terminal']is True and review['parent_dead']is True and review['all_worker_pids_dead']is True and review['active_stage_lock_gone']is True,'Tag stage/processes must be terminal')
    require((review['expected_tag_count'],review['complete_tag_count'],review['failed_tag_count'],review['unknown_request_count'])==(1637,1636,1,0),'Full1637 known firstpass required')
    m=bound(review['source_manifest']);require(review['source_manifest']==pointer(run/'retrieval/run_manifest.json')and m['schema']==base.SCHEMA and review['protocol_sha256']==m['protocol_sha256'],'Original effective40 retrieval manifest required')
    require(m['protocol_sha256']==digest({k:v for k,v in m.items()if k!='protocol_sha256'}),'Original retrieval protocol seal changed')
    require(m['code_sha256']=={p:pointer(ROOT/p)['sha256']for p in m['code_sha256']},'Frozen retrieval code changed')
    core=bound(review['core_audit']);require(core['status']=='sealed_firstpass_with_failed_tag' and core['stage_terminal']is True and not core['controller_alive'] and not core['changed_evidence_paths']and not core['binding_failures']and not core['requirements_unmet'],'Core transport audit incomplete')
    proofs={p:pointer(p)for p in core['evidence_sha256']}
    require(all(proofs[p]['sha256']==h for p,h in core['evidence_sha256'].items()),'Firstpass evidence changed')
    stage=bound(review['stage_status']);require(review['stage_status']==pointer(run/'retrieval/tag_status.json')and stage['terminal']is True,'Stage status not terminal')
    require({x['conversation']for x in stage['outcomes']}==set(range(10))and len(stage['outcomes'])==10,'All ten firstpass outcomes required')
    _stage_access(run,allow_retrieve)
    slots={(x['conversation'],x['fact_id']):x for x in review['tags']};require(len(slots)==len(review['tags'])==1637,'Duplicate/missing firstpass IDs')
    failures=[];banks={};fact=None;failed=None;contexts={}
    for ci in range(10):
        item=next(x for x in m['banks']if(x['conversation'],x['arm'])==(ci,'session'));bank=bound(item['bank']);banks[ci]=item['bank'];contexts[ci]=base.session_contexts(ci)
        ids={f['id']for f in bank['facts']};require(len(ids)==len(bank['facts'])and ids==set(review['expected_fact_ids_by_conversation'][str(ci)]),'Reviewed new-bank IDs differ')
        require({f for(c,f)in slots if c==ci}==ids,'Full firstpass tag coverage differs')
        state_path=run/'retrieval'/f'conv_{ci}'/'tag_status.json';state=read(state_path);proofs[str(state_path)]=pointer(state_path)
        require(state['terminal']is True and state['completed']==state['total']==len(ids)and state['protocol_sha256']==m['protocol_sha256'],'Worker status incomplete')
        actual_files={p.stem for p in (run/'retrieval'/f'conv_{ci}'/'tags').glob('*.json')};require(actual_files==ids,'Unexpected or missing tag files')
        failed_count=0
        for f in bank['facts']:
            reviewed=slots[(ci,f['id'])];tp=run/'retrieval'/f'conv_{ci}'/'tags'/(f['id']+'.json');row=verify(tp)
            require(reviewed['tag']==pointer(tp)and reviewed['status']==row['status'],'Audited tag pointer/status differs')
            context=contexts[ci][f['input_scope']['session_id']]
            require((row['conversation'],row['fact_id'],row['bank'],row['protocol_sha256'],row['fact_text_sha256'],row['source_context_sha256'])==(ci,f['id'],item['bank'],m['protocol_sha256'],digest(f['text']),digest(context)),'Tag current fact/context/bank binding differs')
            bound(reviewed['tag_intent'])
            if row['status']=='complete':strict_result(row['result'],f['id'])
            else:
                require(row['status']=='failed','Unknown terminal tag status');failed_count+=1;failures.append((ci,f['id']));fact=f;failed=row
        require(failed_count==state['failed'],'Worker failure count differs')
    require(failures==[(2,FACT_ID)],'Exactly the reviewed failed fact is permitted')
    require(review['all_failed']==[slots[(2,FACT_ID)]],'Reviewed failed set differs')
    event=failed['api_events'];require(len(event)==1 and event[0]['request_hash']==REQUEST_HASH and event[0]['cache_sha256']is None,'Original failed request event differs')
    original_api=run/'retrieval/conv_2/tag_api';intent_path=original_api/'request_intents'/(REQUEST_HASH+'.json');intent=read(intent_path)
    signature,prompt_hash=expected_request(fact,contexts[2][fact['input_scope']['session_id']])
    require(intent['signature']==signature and intent['request_hash']==request_hash(signature)==REQUEST_HASH and intent['retries']==2 and intent['context']==event[0]['context']and intent['stage']=='jev_score','Original complete intent/request differs')
    original_raw=original_api/'api_cache'/(REQUEST_HASH+'.json');require(not original_raw.exists(),'Original failed raw now exists; no resend')
    attempts=[r for r in _json_lines(original_api/'api_calls.jsonl')if r.get('request_hash')==REQUEST_HASH]
    require(len(attempts)==3 and [r.get('attempt')for r in attempts]==[1,2,3],'Exact three ended attempts required')
    for a in attempts:
        _ended(a);require(a.get('status')=='transport_error' and 'SSLEOFError' in a.get('error','')and a.get('cache_hit')is not True and a.get('context')==intent['context']and(a.get('provider'),a.get('model'),a.get('operation'),a.get('stage'))==('jev','jev-1.13.0','systemone','jev_score'),'Known exhausted TLS transport chain required')
    require(pointer(review_path)==rp,'Firstpass review changed')
    return {'review':rp,'source_manifest':review['source_manifest'],'original_protocol_sha256':m['protocol_sha256'],'snapshot_refs':list(proofs.values())+[pointer(review['source_manifest']['path']),pointer(review['stage_status']['path'])],
        'failed_tag':pointer(run/'retrieval/conv_2/tags'/(FACT_ID+'.json')),'failed_fact':{k:failed[k]for k in('conversation','fact_id','bank','fact_text_sha256','source_context_sha256','source_scope')},'original_request_intent':pointer(intent_path),'original_request_hash':REQUEST_HASH,'original_signature':signature,'original_context':intent['context'],'original_raw_absent_path':str(original_raw),'prompt_hash':prompt_hash},fact,contexts[2][fact['input_scope']['session_id']],review

def prepare(run,namespace,firstpass_review):
    run=Path(run).resolve();ns=Path(namespace).resolve();require(not ns.is_relative_to(run),'Recovery namespace must be outside original source run')
    require(run==SOURCE_RUN.resolve()and ns==PRODUCTION_NAMESPACE.resolve(),'Only the single authorized source/namespace may prepare')
    snapshot,_,_,_=firstpass_snapshot(run,firstpass_review);ns.mkdir(parents=True,exist_ok=False)
    m={'schema':SCHEMA,'created_utc':now(),'source_run':str(run),'namespace':str(ns),**snapshot,'policy':POLICY,'code_sha256':{p:pointer(ROOT/p)['sha256']for p in CODE}}
    m['protocol_sha256']=digest(m);save_new(ns/'recovery_manifest.json',m);return {'prepared':str(ns),'model_calls':0,'failed_fact_id':FACT_ID}
def check(namespace,*,allow_retrieve=False):
    ns=Path(namespace).resolve();m=verify(ns/'recovery_manifest.json')
    require(ns==PRODUCTION_NAMESPACE.resolve()and Path(m['source_run']).resolve()==SOURCE_RUN.resolve(),'Another recovery namespace/source is forbidden')
    require(m['schema']==SCHEMA and m['namespace']==str(ns)and m['policy']==POLICY,'Recovery protocol differs')
    require(m['protocol_sha256']==digest({k:v for k,v in m.items()if k not in('content_sha256','protocol_sha256')}),'Recovery protocol seal differs')
    require(m['code_sha256']=={p:pointer(ROOT/p)['sha256']for p in CODE},'Recovery source code changed')
    if allow_retrieve:
        terminal=verify(ns/'outcome.json');require(terminal['status']=='complete'and terminal['result']==pointer(ns/'result.json'),'Only completed recovery permits retrieve-stage validation')
    snapshot,fact,context,review=firstpass_snapshot(m['source_run'],m['review']['path'],allow_retrieve=allow_retrieve)
    for k,v in snapshot.items():require(m[k]==v,'Original firstpass snapshot changed: '+k)
    for ref in m['snapshot_refs']:require(pointer(ref['path'])==ref,'Firstpass snapshot bytes changed')
    require(not Path(m['original_raw_absent_path']).exists(),'Original raw appeared')
    return m,fact,context,review

def execute(namespace,api_factory=None):
    ns=Path(namespace).resolve();m,fact,context,_=check(ns)
    require(not(ns/'execute_intent.json').exists()and not(ns/'outcome.json').exists()and not(ns/'api').exists(),'Single recovery already started; no replay')
    mp=pointer(ns/'recovery_manifest.json');save_new(ns/'execute_intent.json',{'schema':SCHEMA,'created_utc':now(),'pid':os.getpid(),'recovery_manifest':mp,'original_request_hash':REQUEST_HASH,'policy':POLICY})
    api_root=ns/'api';api=None
    try:
        api=api_factory(api_root,m)if api_factory else CachedLegacyAPI(api_root,retries=0,cache_roots=(),context=m['original_context'])
        require(api.retries==0 and tuple(api.cache_roots)==()and api.context==m['original_context'],'Recovery API transport/context differs')
        guard=ExactJudge(m['original_signature'],api=api);result=tag_fact({**fact,'source_context':context},guard);strict_result(result,FACT_ID)
        require(result['prompt_hash']==m['prompt_hash'],'Tag prompt differs')
        newraw=pointer(api_root/'api_cache'/(REQUEST_HASH+'.json'));ledger=pointer(api_root/'api_calls.jsonl')
        record={'schema':base.SCHEMA,'protocol_sha256':m['original_protocol_sha256'],**m['failed_fact'],'status':'complete','result':result,'api_events':list(api.events)}
        row={'schema':SCHEMA,'status':'complete','created_utc':now(),'recovery_manifest':mp,'original_failed_tag':m['failed_tag'],'original_request_hash':REQUEST_HASH,'new_request_hash':REQUEST_HASH,'new_raw':newraw,'api_ledger':ledger,'new_api_root':str(api_root),'fact_tag_record':record}
        _validate_new(m,row,fact,context)
        save_new(ns/'result.json',row);save_new(ns/'outcome.json',{'schema':SCHEMA,'status':'complete','recovery_manifest':mp,'result':pointer(ns/'result.json'),'original_request_hash':REQUEST_HASH,'api_ledger':ledger})
    except Exception as ex:
        save_new(ns/'outcome.json',{'schema':SCHEMA,'status':'failed','recovery_manifest':mp,'error_type':type(ex).__name__,'error':'Once-only exact tag recovery failed; retained namespace cannot dispatch again','api_ledger':pointer(api_root/'api_calls.jsonl')if(api_root/'api_calls.jsonl').exists()else None,'unknown_cost_not_zero':True})
        raise
    return validate_result(ns)

def _validate_new(m,row,fact,context):
    ns=Path(m['namespace']);api_root=ns/'api';require(row['schema']==SCHEMA and row['status']=='complete'and row['recovery_manifest']==pointer(ns/'recovery_manifest.json'),'Recovery terminal result binding differs')
    require(row['original_failed_tag']==m['failed_tag']and row['original_request_hash']==row['new_request_hash']==REQUEST_HASH and Path(row['new_api_root']).resolve()==api_root,'Recovery original/new request differs')
    require(row['new_raw']==pointer(api_root/'api_cache'/(REQUEST_HASH+'.json'))and row['api_ledger']==pointer(api_root/'api_calls.jsonl'),'New raw/ledger namespace differs')
    raw=bound(row['new_raw']);require(raw['request']==m['original_signature']and raw['stage']=='jev_score','New full request differs')
    require(set(raw['response'].get('answers',{}))==set(COARSE_CRITERIA)|set(FINE_CRITERIA)and raw['response'].get('model')in(None,'jev-1.13.0'),'New raw criterion/model coverage differs')
    intent=read(api_root/'request_intents'/(REQUEST_HASH+'.json'));require(intent['signature']==m['original_signature']and intent['request_hash']==REQUEST_HASH and intent['retries']==0 and intent['context']==m['original_context']and intent['stage']=='jev_score','New API exclusive intent differs')
    require({p.name for p in(api_root/'request_intents').glob('*.json')}=={REQUEST_HASH+'.json'}and{p.name for p in(api_root/'api_cache').glob('*.json')}=={REQUEST_HASH+'.json'},'Extra recovery request/cache forbidden')
    require(not(api_root/'cache_lineage').exists(),'Inherited recovery cache forbidden')
    lines=_json_lines(row['api_ledger']['path']);require(len(lines)==1,'One physical attempt only')
    a=lines[0];_ended(a);require(a.get('cache_hit')is False and a.get('attempt')==1 and a.get('status')==200 and a.get('context')==m['original_context']and(a.get('provider'),a.get('model'),a.get('operation'),a.get('stage'),a.get('request_hash'))==('jev','jev-1.13.0','systemone','jev_score',REQUEST_HASH),'Exact once physical success required')
    expected=tag_fact({**fact,'source_context':context},ExactJudge(m['original_signature'],response=raw['response']));strict_result(expected,FACT_ID)
    rec=row['fact_tag_record'];require(rec['result']==expected and rec['status']=='complete'and rec['schema']==base.SCHEMA and rec['protocol_sha256']==m['original_protocol_sha256'],'Result must equal unchanged raw parser')
    require(all(rec[k]==v for k,v in m['failed_fact'].items()),'Tag original fact/context/bank metadata differs')
    ev=rec['api_events'];require(len(ev)==1 and ev[0]['source']=='new_request'and ev[0]['request_hash']==REQUEST_HASH and ev[0]['cache_sha256']==row['new_raw']['sha256']and ev[0]['error_type']is None and ev[0]['context']==m['original_context']and Path(ev[0]['cache_path']).resolve()==Path(row['new_raw']['path'])and(ev[0]['provider'],ev[0]['model'],ev[0]['operation'],ev[0]['stage'])==('jev','jev-1.13.0','systemone','jev_score'),'Recovery API event raw lineage differs')
    strict_result(rec['result'],FACT_ID)
def validate_recovery(namespace):
    ns=Path(namespace).resolve();m,fact,context,review=check(ns,allow_retrieve=True);row=verify(ns/'result.json');out=verify(ns/'outcome.json');intent=verify(ns/'execute_intent.json')
    require(out['status']=='complete'and out['schema']==SCHEMA and out['result']==pointer(ns/'result.json')and out['recovery_manifest']==pointer(ns/'recovery_manifest.json')and out['api_ledger']==row['api_ledger'],'Successful terminal recovery required')
    require(intent['schema']==SCHEMA and intent['recovery_manifest']==row['recovery_manifest']and intent['original_request_hash']==REQUEST_HASH and intent['policy']==POLICY,'Once-only execution intent changed')
    _validate_new(m,row,fact,context);return m,row,review
def validate_result(namespace):return validate_recovery(namespace)[1]
check_result=validate_result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check','execute'));p.add_argument('--namespace',type=Path,required=True);p.add_argument('--run',type=Path);p.add_argument('--firstpass-review',type=Path);a=p.parse_args()
    if a.stage=='prepare':require(a.run is not None and a.firstpass_review is not None,'prepare requires source run and firstpass-review');result=prepare(a.run,a.namespace,a.firstpass_review)
    elif a.stage=='check':m,_,_,_=check(a.namespace);result={'checked':str(a.namespace),'model_calls':0,'original_request_hash':m['original_request_hash']}
    else:r=execute(a.namespace);result={'status':r['status'],'namespace':str(a.namespace),'new_request_hash':r['new_request_hash']}
    print(json.dumps(result))
if __name__=='__main__':main()
