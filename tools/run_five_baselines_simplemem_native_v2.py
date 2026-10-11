"""Single-use, case-certified SimpleMem terminal-bank continuation.

All original results remain immutable. Original same-query calls replay in order;
the complete original SimpleMem request registry also forbids redrawing a seen
request in another query. Original QA functions and transport are unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import audit_simplemem_terminal_v2 as admission
from tools import baselines_simplemem_native_v2 as adapter
from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import BaselineAPI,BaselineFatal,read,sha,save_new,now
from tools.simplemem_prefix_api_v2 import PrefixPolicy,ContinuationAPI
from jev_mem.sdk20_cached_api import digest

ROOT=base.ROOT;OLD=base.RUN/'simplemem';NEW=base.RUN/'simplemem_native_v2'
SCHEMA='simplemem-native-continuation-entry-v2'
REVIEW_SCHEMA='simplemem-native-continuation-independent-review-v2'
DEFAULT_PROOF=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0603/simplemem_terminal_proof_v3.json'
CORE=('tools/baselines_simplemem_native_v2.py','tools/simplemem_prefix_api_v2.py',
      'tools/run_five_baselines_simplemem_native_v2.py','tools/audit_simplemem_terminal_v2.py',
      'tools/seal_simplemem_native_v2.py','tools/baseline_controller_guard_v3.py','tools/baseline_controller_guard_v2.py',
      'tools/archive_amem_continuation_v4.py','tools/check_experiment_health.py','research/sources/locomo_evaluation_cbfbc1.py')

def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def require(value,message):
    if not value:raise ValueError(message)
def code_hashes(parent):return {**parent['code_sha256'],**{p:sha(ROOT/p) for p in CORE}}
def check_review(path,proof_path,parent):
    path=adapter._safe_tree(path)
    require(path.is_relative_to(ROOT/'research'),'Review must be external research evidence')
    r=read(path)
    require(r.get('schema')==REVIEW_SCHEMA and r.get('status')=='passed' and r.get('scope')=='implementation_and_actual_terminal_admission'
        and r.get('proof')==binding(proof_path) and r.get('code_sha256')==code_hashes(parent)
        and r.get('upstream_sha256')==parent['upstream_sha256'] and r.get('archive_cleanup_downstream_authorized') is False,
        'Final-hash independent continuation review required')
    require(bool(r.get('evidence_sha256')),'Independent actual evidence required')
    for p,h in r['evidence_sha256'].items():require(sha(adapter._safe_tree(p))==h,'Independent evidence changed')
    return binding(path)
def expected_manifest(proof_path,proof,review_path):
    parent=read(proof['parent_manifest']['path'])
    require(binding(OLD/'run_manifest.json')==proof['parent_manifest'],'Original parent changed')
    return {'schema':SCHEMA,'method':'simplemem','self_root':str(NEW),'parent_root':str(OLD),
        'parent_manifest':proof['parent_manifest'],'proof':binding(proof_path),
        'independent_review':check_review(review_path,proof_path,parent),'questions':parent['questions'],
        'failed_query_ids':[c['query_id'] for c in proof['failed_retrievals']],
        'code_sha256':code_hashes(parent),'upstream_sha256':parent['upstream_sha256'],
        'scientific_configuration':{k:parent[k] for k in ('generation_model','thinkingBudget','temperature','internal_max_output',
            'reader_judge_max_output','evaluation_protocol','judge_prompt_sha256','metric','specification')},
        'continuation':{'original_same_query_ordered_prefix':True,'global_same_run_exact_cache':True,
            'seen_request_redraw_allowed':False,'new_unseen_suffix_first_requests_allowed':True,'bank_policy':'exact terminal clone',
            'qa_uniform_recovery_limit':1,'conversation_workers':10,'query_workers_per_conversation':4,
            'archive_cleanup_downstream_authorized':False},'adapter':adapter.describe()}
def prepare(review,proof_path=DEFAULT_PROOF):
    admission.assert_stopped();adapter._safe_tree(NEW)
    if NEW.exists():raise FileExistsError('Continuation namespace must be fresh')
    proof=admission.validate_proof(proof_path);m=expected_manifest(proof_path,proof,review)
    save_new(NEW/'run_manifest.json',m);base.status(NEW,'prepared',failed_retrievals=len(proof['failed_retrievals']))
    return {'status':'prepared','manifest':binding(NEW/'run_manifest.json'),'model_calls':0}
def manifest():
    admission.assert_stopped();adapter._safe_tree(NEW);m=read(NEW/'run_manifest.json')
    require(binding(m['proof']['path'])==m['proof'],'Proof bytes changed')
    p=admission.validate_proof(m['proof']['path'])
    require(m==expected_manifest(m['proof']['path'],p,m['independent_review']['path']),'Continuation manifest changed')
    for path,h in m['upstream_sha256'].items():require(sha(ROOT/path)==h,'Pinned upstream changed')
    return m,p

def _audit_functions(view):
    from tools import seal_simplemem_native_v2 as seal
    P=seal.paths(view);scope={**admission.__dict__,'Path':P,'shared_read':view.bytes,'read':view.read,'file_sha':view.sha,
        '_BOUND_INVENTORY':None,'_BINDING_CACHE':None,'_JSONL_CACHE':None}
    for name,fn in admission.__dict__.items():
        if isinstance(fn,FunctionType) and fn.__module__==admission.__name__:
            scope[name]=FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__);scope[name].__kwdefaults__=fn.__kwdefaults__
    scope.update(read=view.read,file_sha=view.sha,shared_read=view.bytes)
    return SimpleNamespace(**scope)

def audit_sidecar(m,proof,*,view=None):
    """Full native/raw/lineage replay; no database opener and no live validator."""
    from tools import seal_simplemem_native_v2 as seal
    view=view or seal.LiveView();P=seal.paths(view);a=_audit_functions(view)
    require(m['schema']==SCHEMA and m['self_root']==str(NEW) and m['parent_root']==str(OLD),'Wrong continuation identity')
    require(seal.bound(view,m['proof'])==proof,'Method proof changed')
    for path,h in {**m['code_sha256'],**m['upstream_sha256']}.items():require(view.sha(ROOT/path)==h,'Method scientific source changed')
    registry=proof['method_request_registry'];cases={c['query_id']:c for c in proof['failed_retrievals']}
    require(len(cases)==len(proof['failed_retrievals']) and set(m['failed_query_ids'])==set(cases),'Failed cohort differs')
    expected_captures={str(NEW/'results'/('c'+str(c['conversation']))/'retrieval'/(qid+'.json')) for qid,c in cases.items()}
    actual_captures={str(p) for p in P(NEW/'results').glob('c*/retrieval/*.json') if not p.name.endswith('.intent.json')}
    expected_intents={str(Path(p).with_suffix('.intent.json')) for p in expected_captures}
    require(actual_captures==expected_captures and {str(p) for p in P(NEW/'results').glob('c*/retrieval/*.intent.json')}==expected_intents,'Extra/missing continuation retrieval captures or intents')
    api=NEW/'work/api_method';events=a.jsonl(api/'logical_events.jsonl');ledger=a.jsonl(api/'api_calls.jsonl')
    consumers=defaultdict(list);physical=defaultdict(list);cached=[]
    for e in events:
        require(e.get('error_type') is None and e['context']['query_id'] in cases,'Unknown method consumer')
        require(e['context']=={'method':'simplemem','conversation':cases[e['context']['query_id']]['conversation'],'query_id':e['context']['query_id'],'phase':'retrieval'},'Consumer context differs')
        consumers[e['request_hash']].append(e)
    for r in ledger:
        if r.get('cache_hit'):cached.append(r)
        else:physical[r['request_hash']].append(r)
    keys=set(consumers);oldkeys=keys & registry.keys();newkeys=keys-oldkeys
    paths=lambda name:{p.stem:p for p in P(api/name).glob('*.json')}
    raws,intents,lineages=paths('api_cache'),paths('request_intents'),paths('cache_lineage')
    require(set(raws)==keys and set(intents)==newkeys==set(physical) and set(lineages)==oldkeys,'Exact method raw/intent/lineage/producer sets differ')
    records={};producer_report=[]
    for key in keys:
        raw=view.read(raws[key]);sig=raw['request'];rawsha=view.sha(raws[key])
        require(digest(sig)==key,'Raw signature differs')
        if key in oldkeys:
            origin=registry[key];require(sig==origin['signature'] and rawsha==origin['raw']['sha256'],'Seen raw/signature changed')
            require(view.sha(origin['raw']['path'])==rawsha and view.sha(origin['intent']['path'])==origin['intent']['sha256'],'Original raw/intent missing or changed')
            expect={'source_path':origin['raw']['path'],'source_sha256':rawsha,'target_path':str(raws[key]),'target_sha256':rawsha,'request_hash':key}
            require(view.read(lineages[key])==expect,'Exact original lineage differs')
            require(all(e['source']=='existing_run_cache' and e['stage'] in {x['stage'] for x in origin['logical_consumers']} for e in consumers[key]),'Seen request was redrawn or stage changed')
            intent=origin['intent']
        else:
            intent_value=view.read(intents[key]);cc=physical[key];new=[e for e in consumers[key] if e['source']=='new_request']
            require(intent_value['signature']==sig and intent_value['request_hash']==key and intent_value['retries']==2 and raw['stage']==intent_value['stage'],'New original-transport intent differs')
            require(len(new)==1 and [x['attempt'] for x in cc]==list(range(1,len(cc)+1)) and 1<=len(cc)<=3 and cc[-1]['status']==200
                and all(x['status'] in admission.RETRYABLE for x in cc[:-1]),'Unknown/unbounded physical method chain')
            ident={'context':intent_value['context'],'stage':intent_value['stage'],'request_hash':key,**{k:sig[k] for k in ('provider','model','operation')}}
            require(all(a.identity(x)==ident for x in [*cc,new[0]]),'New physical/consumer identity differs')
            require(cc[-1].get('usage')==raw['response'].get('usageMetadata',{}) and cc[-1].get('resolved_model')==raw['response'].get('modelVersion'),'Raw usage/model differs')
            times=[datetime.fromisoformat(intent_value['created_utc']),*[datetime.fromisoformat(x['timestamp_utc']) for x in cc],datetime.fromisoformat(new[0]['recorded_utc'])]
            require(times==sorted(times),'Method producer ordering differs')
            intent={'path':str(intents[key]),'sha256':view.sha(intents[key])}
            producer_report.append({'request_hash':key,'physical_attempts':len(cc),'raw_sha256':rawsha})
        for e in consumers[key]:
            require(e['source'] in ('existing_run_cache','new_request') and str(raws[key])==e['cache_path'] and e['cache_sha256']==rawsha
                and all(e[k]==sig[k] for k in ('provider','model','operation')),'Method consumer raw endpoint differs')
        records[key]={'signature':sig,'raw':{'path':str(raws[key]),'sha256':rawsha},'intent':intent,'raw_value':raw}
    require(Counter(digest(a.identity(e)) for e in events if e['source']=='existing_run_cache')==Counter(digest(a.identity(r)) for r in cached),'Method cache multiplicity differs')
    pacing=a.jsonl(api/'embedding_pacing_events.jsonl')
    require(Counter((x['request_hash'],x['physical_attempt']) for x in pacing if x['event']=='physical_embedding_admitted')==Counter((key,r['attempt']) for key,rr in physical.items() for r in rr if r['operation']=='batchEmbedContents'),'Physical embedding pacing differs')
    report=[];all_capture=[];all_logs=set();all_results=set();fmt=a.original_formatter()
    for qid,case in cases.items():
        context={'method':'simplemem','conversation':case['conversation'],'query_id':qid,'phase':'retrieval'}
        cp=NEW/'results'/('c'+str(case['conversation']))/'retrieval'/(qid+'.json');capture=view.read(cp)
        require(capture['status']=='complete' and view.read(cp.with_suffix('.intent.json'))['context']==context,'Incomplete retrieval or changed capture intent')
        logical=[e for e in events if e['context']==context]
        require(capture['events']==[a.normalized_event(e) for e in logical],'Ordered complete retrieval consumers differ')
        all_capture.extend(capture['events']);value=capture['value'];bank=NEW/'work/native'/('c'+str(case['conversation']))/'simplemem'
        log=Path(value['request_log']);result=bank/'retrieval'/(value['retrieval_id']+'.result.json')
        require(log==bank/'retrieval'/(value['retrieval_id']+'.requests.jsonl') and view.read(result)==value,'Native output path or full result differs')
        all_logs.add(str(log));all_results.add(str(result))
        native=a.audit_native(log,logical,records,context)
        require(len(native['prefix'])>=len(case['prefix']),'Native prefix truncated')
        for old,new in zip(case['prefix'],native['prefix']):
            require(all(old[k]==new[k] for k in ('signature','stage','context','native_request_payload_sha256','native_response_payload_sha256')),'Original full native prefix semantics differ')
        policy=PrefixPolicy(case,registry);journal=a.jsonl(NEW/'work/prefix'/(qid+'.jsonl'));pos=0
        for item in journal:
            if item['event']=='request_admission':
                require(pos<len(logical),'Extra request admission')
                e=logical[pos];expected=policy.before(records[e['request_hash']]['signature'],e['stage'])
                require(item=={'event':'request_admission',**expected},'Request policy journal differs');pos+=1
            elif item['event']=='native_compatibility':
                policy.authorize_native_compatibility(item['kind'],item['details'])
                require(item=={'event':'native_compatibility',**policy.compatibility[-1]},'Native compatibility journal differs')
            else:raise ValueError('Unknown native compatibility journal')
        require(pos==len(logical),'Missing method admissions');policy.report()
        state=view.read(bank/'state.json');original=proof['banks'][str(case['conversation'])]
        lineage=view.read(bank/'terminal_bank_lineage.json')
        old_db=Path(original['bank_dir'])/'lancedb'
        expected_copy={str(bank/'lancedb'/Path(path).relative_to(old_db)):h for path,h in original['lancedb_inventory'].items()}
        require(lineage['copied_database_inventory']==expected_copy,'Cloned database inventory rebasing differs')
        actual_copy={str(path):view.sha(path) for path in P(bank/'lancedb').rglob('*') if path.is_file()}
        require(actual_copy==expected_copy,'Cloned database bytes or exact file set changed')
        require(lineage['original_state']==original['state'] and lineage['original_database_inventory']==original['lancedb_inventory']
            and lineage['database_integrity']==original['database_integrity'] and lineage['rebased_state']=={'path':str(bank/'state.json'),'sha256':view.sha(bank/'state.json')},'Cloned terminal bank lineage differs')
        original_state=seal.bound(view,original['state']);expected=deepcopy(original_state)
        expected.update(work_dir=str(bank),memory_path=str(bank/'memory.json'),request_log=str(bank/'requests.jsonl'),export_paths=[str(bank/Path(p).name) for p in expected['export_paths']])
        require(state==expected,'Rebased terminal state changes scientific fields')
        memory=view.read(bank/'memory.json')
        for name,h in state['export_sha256'].items():require(view.sha(bank/name)==h,'Exact terminal export copy changed')
        memory_entries={x['entry_id']:x for x in memory['entries']};sources={x['dialogue_id']:x for x in memory['sources']};entries=[]
        for e in value['evidence']:
            eid=e['memory_id'];entry=memory_entries[eid]
            require(e=={'memory_id':eid,'memory':entry,'source_dia_ids':[sources[i]['dia_id'] for i in memory['window_provenance'][eid]],'attribution':'source window, not individually verified fact evidence'},'Complete native provenance differs')
            entries.append(SimpleNamespace(**entry))
        require(value['context']==fmt(None,entries) and value['count']==len(entries) and value['source_sha']==original_state['source_sha'] and value['source_hash']==original_state['source_hash'],'Original full context formatter differs')
        report.append({'query_id':qid,'original_prefix_calls':len(case['prefix']),'method_calls':len(logical),'compatibility':policy.compatibility,'capture':{'path':str(cp),'sha256':view.sha(cp)}})
    require(Counter(digest(x) for x in all_capture)==Counter(digest(a.normalized_event(x)) for x in events),'Unconsumed method consumers')
    require({str(p) for p in P(NEW/'work/native').rglob('*.requests.jsonl')}==all_logs and {str(p) for p in P(NEW/'work/native').rglob('*.result.json')}==all_results,'Orphan native logs/results')
    require({p.stem for p in P(NEW/'work/prefix').glob('*.jsonl')}==set(cases),'Extra/missing policy journals')
    return {'status':'passed','retrievals':report,'new_physical_requests':sorted(producer_report,key=lambda r:r['request_hash']),
        'original_same_query_ordered_prefix_calls':sum(len(c['prefix']) for c in cases.values()),'global_same_run_exact_cache_keys':sorted(oldkeys),
        'unknown_or_unresolved':[],'replay_scope':'full native requests/raw/embedding normalizations, exact prefix and compatibility, original formatter and provenance; terminal DB search was executed on the bound copy, not rerun by this text audit'}

def run():
    m,p=manifest()
    if base.BaselineAPI is not BaselineAPI:raise BaselineFatal('Frozen QA provider identity changed')
    if (NEW/'execution_intent.json').exists() or (NEW/'work').exists():raise FileExistsError('No implicit continuation reentry')
    save_new(NEW/'execution_intent.json',{'created_utc':now(),'manifest':binding(NEW/'run_manifest.json'),'failed_query_ids':m['failed_query_ids'],'seen_request_redraw_allowed':False})
    try:
        states={}
        for ci in sorted({c['conversation'] for c in p['failed_retrievals']}):
            bank=p['banks'][str(ci)]
            states[ci]=adapter.copy_terminal_bank(bank['state']['path'],NEW/'work/native'/('c'+str(ci))/'simplemem',state_sha256=bank['state']['sha256'],database_inventory=bank['lancedb_inventory'])
        admit={'run_dir':str(NEW/'work/api_method'),'query_ids':m['failed_query_ids'],'executing':True,
            'case_sha256':{c['query_id']:digest(c) for c in p['failed_retrievals']},'registry_sha256':digest(p['method_request_registry'])}
        qs={q['query_id']:q for q in p['questions']};registry=p['method_request_registry']
        def work(case):
            q=qs[case['query_id']];api=ContinuationAPI(NEW/'work/api_method',admission=admit,case=case,registry=registry,context={'method':'simplemem','conversation':q['conversation'],'query_id':q['query_id'],'phase':'retrieval'})
            result=base.capture_stage(NEW/'results'/('c'+str(q['conversation']))/'retrieval'/(q['query_id']+'.json'),lambda:adapter.retrieve(q['question'],states[q['conversation']],api),api)
            if result['status']!='complete':return {'query_id':q['query_id'],'status':'failed'}
            api.policy.report()
            base.qa_one(NEW,'simplemem',q,result['value'])
            return {'query_id':q['query_id'],'status':'complete'}
        def conversation(ci):
            with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(work,[c for c in p['failed_retrievals'] if c['conversation']==ci]))
        base.status(NEW,'retrieval_and_first_qa',failed_retrievals=len(p['failed_retrievals']))
        with ThreadPoolExecutor(max_workers=10) as pool:results=[r for group in pool.map(conversation,states) for r in group]
        save_new(NEW/'continuation_outcomes.json',results)
        require(admission.inventory(OLD)==p['original_inventory'],'Original SimpleMem run changed')
        require(len(results)==len(p['failed_retrievals']) and all(r['status']=='complete' for r in results),'Retrieval continuation incomplete; no union or QA recovery permitted')
        audit_sidecar(m,p)
        from tools import seal_simplemem_native_v2 as seal
        summary=seal.finish_run(m,p)
        require(admission.inventory(OLD)==p['original_inventory'],'Original evidence changed during seal')
        stage='complete_awaiting_result_review' if summary['audit_status']=='no_invalid_artifacts' else 'needs_review'
        base.status(NEW,stage,summary_sha256=sha(NEW/'results/summary.json'))
        return {'status':stage,'summary_sha256':sha(NEW/'results/summary.json')}
    except BaseException as exc:
        base.status(NEW,'needs_review',error=base.error_record(exc));raise
def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','run']);parser.add_argument('--proof',type=Path,default=DEFAULT_PROOF);parser.add_argument('--review',type=Path);parser.add_argument('--process-probe',action='store_true');args=parser.parse_args()
    if args.process_probe:
        require(args.command=='run','Probe needs actual run launcher shape');admission.assert_stopped();result={'status':'passed','scope':'process_guard_only','model_calls':0}
    elif args.command=='prepare':result=prepare(args.review,args.proof)
    elif args.command=='check':result={'status':'passed','manifest':manifest()[0]}
    else:result=run()
    print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__':main()
