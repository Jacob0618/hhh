"""Complete read-only admission for one terminal c2_q42 retrieval failure."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType
import numpy as np

from tools import run_five_baselines_amem_native_v4 as parent
from tools import baselines_amem_v4 as amem
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import read,sha,save_new
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,MODEL

ROOT=parent.ROOT
SOURCE=parent.NEW
FINAL=SOURCE/'retrieval_continuation_v1'
WORK_OWNER=SOURCE/'work/retrieval_continuation_v1'
QID='c2_q42'
KEY='1802b8f3f2aacae0514197993db94c2e22f597b25e4274b91cd15f23f5180761'
RAW_SHA='2f68277a42f3e276c227c6767972130959d652c2e89ef2f63509ece70a1b7949'
CONTEXT={'method':'amem','conversation':2,'phase':'retrieval','query_id':QID}
SCHEMA='amem-single-retrieval-terminal-admission-v1'
DEFAULT_QUALIFICATION=parent.previous.base.RESEARCH/'heartbeat_20261008T2147/independent_single_query_failure_qualification_v2.json'
DEFAULT_PROOF=parent.previous.base.RESEARCH/'amem_retrieval_continuation_v1/terminal_admission.json'
base=parent.previous.base


def clone(fn,**values):
    result=FunctionType(fn.__code__,{**fn.__globals__,**values},fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result


def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}


def bound(record,under=ROOT):
    p=Path(record['path']).resolve()
    if not p.is_relative_to(under.resolve()) or not p.is_file() or sha(p)!=record['sha256']:
        raise ValueError('Bound source evidence differs: '+str(p))
    return p


def source_inventory():
    """The two exact new trees are the only exclusions, including after prepare."""
    result={}
    for p in sorted(SOURCE.rglob('*')):
        if getattr(p.lstat(),'st_file_attributes',0)&0x400 or p.is_symlink():raise ValueError('Source reparse point')
        if p.is_relative_to(FINAL) or p.is_relative_to(WORK_OWNER):continue
        if p.is_file():result[str(p.resolve())]=sha(p)
    return result


def parent_manifest():
    def validate(path):
        previous=read(path);q=previous['failure_qualification'];bound(q)
        actual=clone(parent._collect,assert_source_stopped=guard.assert_stopped)(q['path'])
        if previous!=actual:raise ValueError('Full parent admission proof differs')
        return previous
    manifest,proof=clone(parent._private_manifest,validate_proof=validate)(return_proof=True)
    if manifest.get('self_root')!=str(SOURCE) or manifest.get('conversation_sources')!=proof['conversation_sources']:
        raise ValueError('Parent manifest ownership differs')
    return manifest,proof


def qualification(path):
    q=read(path)
    expected={'schema':'amem-single-query-failure-qualification-v2','status':'passed',
        'scope':'individual_known_terminal_technical_failure_only','kind':'known_terminal_truncated_keyword_json',
        'retry_limit':1,'method':'amem','conversation':2,'query_id':QID,'request_hash':KEY,'signature_sha256':KEY,
        'context':CONTEXT,'stage':'amem_query_keywords','raw_sha256':RAW_SHA,'unknown_or_inflight':[],
        'original_reader_judge_calls':0,'original_reader_judge_intents':0,'preserved_successful_qa':1539,
        'original_http_record_complete':True,'generation_complete':False,'finish_reason':'MAX_TOKENS',
        'lossless_local_repair_possible':False,'launch_admitted':False,'paid_request_authorized_by_certificate':False,
        'new_api_requests':0,'credential_reads':0,'run_files_modified':0,'source_run':str(SOURCE)}
    if any(q.get(k)!=v for k,v in expected.items()) or digest(q['signature'])!=KEY:
        raise ValueError('Exact one-use individual failure qualification required')
    for stem,folder in (('raw',SOURCE/'work/api_method/api_cache'),('native_call',SOURCE/'work/native/c2/calls'),
                        ('retrieval_capture',SOURCE/'results/c2/retrieval')):
        bound({'path':q[stem+'_path'],'sha256':q[stem+'_sha256']},folder)
    bound(q['request_intent'],SOURCE/'work/api_method/request_intents')
    report=read(bound({'path':q['review_path'],'sha256':q['review_sha256']}))
    if (report.get('schema')!='amem-v4-terminal-retrieval-gap-independent-review-v1' or report.get('status')!='passed'
            or report.get('unknown_terminal_logical_outcomes')!=0 or report.get('unresolved_terminal_request_intents')!=0
            or report.get('source_before_after_equal') is not True or report['qa']['successful_queries_replayed']!=1539
            or report.get('new_api_requests')!=0 or report.get('credential_reads')!=0 or report.get('production_files_written')!=0):
        raise ValueError('Complete independent terminal gap evidence required')
    for p,h in report['evidence_sha256'].items():bound({'path':p,'sha256':h})
    inventory=read(bound(q['source_inventory']))
    expected_inventory={str((SOURCE/p).resolve()):h for p,h in inventory['sha256'].items()}
    if (inventory.get('run')!=str(SOURCE) or inventory.get('before_after_equal') is not True
            or len(expected_inventory)!=q['source_inventory_files'] or source_inventory()!=expected_inventory):
        raise ValueError('Current source differs from complete independent terminal inventory')
    return q,report


def failed_slot(q,question,state):
    raw=read(q['raw_path']);call=read(q['native_call_path']);capture=read(q['retrieval_capture_path'])
    req=call['request'];sig={'provider':'gemini','model':MODEL,'operation':'generateContent',
        'body':generation_body(req['prompt'],system=req['system'],json_output=True,max_output=16384)}
    if (sig!=q['signature'] or raw['request']!=sig or digest(sig)!=KEY or call['request_sha256']!=amem._v1._sha(req)
            or call['status']!='failed' or call.get('raw_response') is not None or call.get('parsed_response') is not None
            or req['json_output'] is not True or req['max_output']!=16384 or req['stage']!='amem_query_keywords'
            or capture['status']!='failed' or len(capture['events'])!=1):
        raise ValueError('Exact original failed native/request/capture differs')
    event=capture['events'][0]
    if (event.get('context')!=CONTEXT or event.get('request_hash')!=KEY or event.get('stage')!='amem_query_keywords'
            or event.get('cache_sha256')!=RAW_SHA or event.get('cache_path')!=q['raw_path'] or event.get('error_type') is not None):
        raise ValueError('Original complete logical failure binding differs')
    response=raw['response'];candidates=response.get('candidates',[])
    if (len(candidates)!=1 or candidates[0].get('finishReason')!='MAX_TOKENS' or candidates[0].get('index')!=0
            or response.get('modelVersion') not in (MODEL,'models/'+MODEL)):
        raise ValueError('Known complete transport with terminal truncated generation required')
    content=candidates[0].get('content',{});parts=content.get('parts',[])
    if content.get('role')!='model' or len(parts)!=1 or set(parts[0])!={'text'}:raise ValueError('Plain truncated text required')
    try:json.loads(parts[0]['text'])
    except json.JSONDecodeError:pass
    else:raise ValueError('Known failure now parses; a new request is forbidden')
    key=amem._v1._sha({'question':question['question'],'memory_sha256':state['memory_sha256']})
    if (Path(q['native_call_path']).name!=f'query_{key}_amem_query_keywords.json'
            or (SOURCE/'work/native/c2/retrieval'/(key+'.json')).exists()
            or (SOURCE/'work/api_method/parser_lineage'/(KEY+'.json')).exists()
            or (SOURCE/'results/firstpass'/QID).exists()):
        raise ValueError('Failed query has unexpected successful or QA artifacts')
    return event


def native_audit(q,event,manifest):
    """Original strict audit with only the separately bound failed slot excluded."""
    frozen=parent.previous.FROZEN_PATH if hasattr(parent.previous,'FROZEN_PATH') else ROOT/'tools/run_five_baselines_amem_native_v3_admission_v2.py'
    data=frozen.read_bytes()
    if hashlib.sha256(data).hexdigest()!=parent.FROZEN_SHA:raise ValueError('Frozen audit source changed')
    source=data.decode();node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='audit_native_outputs')
    function=ast.get_source_segment(source,node)
    changes={
        'from tools import baselines_amem_v3 as amem':'from tools import baselines_amem_v4 as amem',
        '    event_counts=Counter(original._sha(event) for event in events)':'    events=_remove_exact_failure(events)\n    event_counts=Counter(original._sha(event) for event in events)',
        "for p in sorted((native/'calls').glob('*.json')):":"for p in sorted((native/'calls').glob('*.json')):\n            if str(p.resolve())==_failed_native:continue",
        "for q in (q for q in m['questions'] if q['conversation']==ci):":"for q in (q for q in m['questions'] if q['conversation']==ci and q['query_id']!=_failed_qid):",
        "if {p.stem for p in (NEW/'work/api_method/api_cache').glob('*.json')}!=parser_keys:":"if {p.stem for p in (NEW/'work/api_method/api_cache').glob('*.json')}!=parser_keys|{_failed_key}:"
    }
    for old,new in changes.items():
        if function.count(old)!=1:raise ValueError('Frozen audit substitution count differs')
        function=function.replace(old,new)
    def remove(values):
        selected=[value for value in values if value.get('request_hash')==KEY]
        if selected!=[event]:raise ValueError('Failed consumer is not exactly the qualified singleton')
        return [value for value in values if value.get('request_hash')!=KEY]
    scope={**parent.controller.audit_native_outputs.__globals__,'NEW':SOURCE,'_remove_exact_failure':remove,
           '_failed_native':str(Path(q['native_call_path']).resolve()),'_failed_qid':QID,'_failed_key':KEY}
    exec(compile(function,str(frozen),'exec'),scope)
    result=scope['audit_native_outputs'](manifest)
    result['known_failed_retrieval']={'query_id':QID,'request_hash':KEY,'raw_sha256':RAW_SHA,'native_call_sha256':q['native_call_sha256']}
    if len(result['generation_calls'])!=1477:raise ValueError('All 1477 successful native calls must replay')
    return result


def collect(qualification_path=DEFAULT_QUALIFICATION):
    guard.assert_stopped();before=source_inventory();m,parent_proof=parent_manifest()
    if read(SOURCE/'status.json').get('stage')!='needs_review':raise ValueError('Source is not the original terminal needs_review state')
    qualification_value,independent=qualification(qualification_path)
    question=next(q for q in m['questions'] if q['query_id']==QID)
    state_path=SOURCE/'work/native/c2/state.json';state=read(state_path);vector=state_path.parent/'cache/embeddings.npy'
    if (not state.get('build_complete') or state['completed_notes']!=663 or len(state['completed_sessions'])!=32
            or len(state['notes'])!=663 or sha(vector)!=state['embeddings_sha256']):
        raise ValueError('Immutable original terminal 663-note bank required')
    amem._runtime._checked_vectors(np.load(vector,allow_pickle=False),663)
    event=failed_slot(qualification_value,question,state)
    raw_audit=clone(parent.controller.audit_new_requests,NEW=SOURCE,_PROOF=parent_proof)()
    native=native_audit(qualification_value,event,m)
    routes=m['conversation_sources'];success=[]
    for q in m['questions']:
        if q['query_id']==QID:continue
        owner=Path(routes[str(q['conversation'])]);leaf=owner/'results/firstpass'/q['query_id'];ret=owner/'results'/f"c{q['conversation']}"/'retrieval'/(q['query_id']+'.json')
        outcome=read(leaf/'outcome.json');reader,judge=parent.controller.replay_leaf(q,leaf,owner,owner,leaf,ret)
        if (reader['status']!='complete' or judge['status']!='complete' or outcome['query_id']!=q['query_id']
                or outcome['conversation']!=q['conversation'] or outcome['method']!='amem' or outcome['phase']!='firstpass'
                or outcome['reader']!='complete' or outcome['judge']!='complete'
                or Path(outcome['reader_path']).resolve()!=leaf/'reader.json' or Path(outcome['judge_path']).resolve()!=leaf/'judge.json'):
            raise ValueError('Every other original QA must be complete and unchanged')
        success.append({'query_id':q['query_id'],'conversation':q['conversation'],'owner':str(owner),'leaf':str(leaf),'retrieval_path':str(ret),
                        'outcome':outcome,'bindings':{str(p):sha(p) for p in (ret,leaf/'reader.json',leaf/'judge.json',leaf/'outcome.json')}})
    if len(success)!=1539:raise ValueError('Exactly 1539 immutable successful original queries required')
    outcomes=read(SOURCE/'rebuild_firstpass_outcome.json')['outcomes']
    if len(outcomes)!=1 or outcomes[0]['conversation']!=2 or outcomes[0]['status']!='terminal':raise ValueError('Whole rebuild queue not terminal')
    values=outcomes[0]['results'];missing=[r for r in values if r.get('status')=='retrieval_failed']
    if len(values)!=152 or missing!=[{'query_id':QID,'status':'retrieval_failed'}]:raise ValueError('Only exact q42 retrieval may be missing')
    if any((SOURCE/p).exists() for p in ('qa_recovery_plan.json','qa_recovery_outcome.json','results/summary.json','results/firstpass_sources.json','results/recovery')):
        raise ValueError('Original unified recovery or seal unexpectedly started')
    if before!=source_inventory():raise ValueError('Source changed during admission')
    return {'schema':SCHEMA,'status':'passed','query_id':QID,'source_run':str(SOURCE),'final_root':str(FINAL),'work_owner':str(WORK_OWNER),
            'source_inventory':before,'source_inventory_exclusions':[str(FINAL),str(WORK_OWNER)],
            'parent_manifest':binding(SOURCE/'run_manifest.json'),'parent_admission':m['technical_continuation']['proof'],
            'original_routes':routes,'original_successes':success,'missing_question':question,
            'original_failure':{'retrieval':binding(qualification_value['retrieval_capture_path']),
                                'native_call':binding(qualification_value['native_call_path']),'raw':binding(qualification_value['raw_path'])},
            'replacement_certificate':binding(qualification_path),'replacement':qualification_value,
            'bank':{'state':binding(state_path),'embeddings':binding(vector),'memory_sha256':state['memory_sha256'],'notes':663,'sessions':32},
            'source_raw_audit':raw_audit,'source_native_audit':native,'model_calls_during_admission':0}


def audit(destination=DEFAULT_PROOF,qualification_path=DEFAULT_QUALIFICATION):
    if FINAL.exists() or WORK_OWNER.exists():raise FileExistsError('Sidecar namespace must be absent before admission')
    if Path(destination).exists():raise FileExistsError('Admission proof is immutable')
    proof=collect(qualification_path);save_new(destination,proof)
    return {'status':'passed','proof':binding(destination),'model_calls':0}


def validate_proof(path):
    saved=read(path)
    if saved.get('schema')!=SCHEMA or saved.get('status')!='passed':raise ValueError('Passed complete retrieval admission required')
    cert=saved['replacement_certificate'];bound(cert)
    if collect(cert['path'])!=saved:raise ValueError('Admission proof differs from complete current source replay')
    return saved
