"""Candidate live audit and deterministic offline native-topic replay.

This is new algorithm replay, not a pointer-only check. It grants no production
authority and never calls an API, encoder or database engine itself.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import types
import uuid

from tools import audit_hypermem_continuation_v3 as old
from tools import audit_hypermem_terminal_v4 as admission
from tools import baselines_hypermem_native_v4 as adapter
from tools import hypermem_prefix_api_v4 as prefix
from tools import audit_simplemem_terminal_v2 as common
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,response_text

require,read,sha,binding,jsonl=old.require,old.read,old.sha,old.binding,old.jsonl
STAGES=old.STAGES;math=old.math

def _replace(source,a,b):
    require(source.count(a)==1,'Frozen audit boundary changed: '+a[:70]);return source.replace(a,b)

def signature(request):
    return {'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
        'body':generation_body(request['prompt'],system=None,json_output=True,max_output=16384)}

def decoded_raw(rr,proof):
    key=digest(rr['signature']);text=response_text(rr['raw_value']['response']);cert=proof['compatibility_cases'].get(key)
    if cert is not None and cert['conversation'] in adapter.INSERTIONS:
        require(rr['raw']['sha256']==cert['raw']['sha256'],'Exact syntax source SHA differs')
        adapter.derive_exact_text(text,cert)
        return text
    return json.loads(text)

def derived_value(original,stage,local,rr,proof):
    key=digest(rr['signature']);cert=proof['compatibility_cases'].get(key)
    value=original
    if cert is not None:
        require(rr['raw']['sha256']==cert['raw']['sha256'] and cert['signature']==rr['signature'] and cert['stage']==stage,'Exact response certificate differs')
        if cert['conversation'] in adapter.INSERTIONS:value,_=adapter.derive_exact_text(original,cert)
        elif cert['conversation'] in (0,7):adapter.false_row_contract(value,local)
        else:raise ValueError('Unknown compatibility class')
    if stage=='hypermem_episode_roles':value,_=adapter.represent_episode_ids(value,local['simple_to_real'])
    return value

class _TopicStop(BaseException):pass

def replay_topic_callers(native,records,proof,*,allow_terminal_failure=False,terminal_stop=None):
    """Execute pinned topic pipeline with complete raw-derived in-memory answers.

    No topic is constructed from the journal being validated. Constructors,
    metadata rules, alias maps, formatter, order and deterministic UUID counter
    are those in the frozen private module. A coroutine suspension is forbidden.
    """
    admission.check_hash_environment(proof['hash_environment'])
    runtime=adapter._runtime;modules=runtime._load();stage2=modules['main/stage2_hypergraph_extraction.py']
    eps=stage2.load_episodes_from_json(str(native/'episodes.json'))
    checkpoint=read(native/'episode_stage.json');runtime._verify_seal(native/'episode_stage.json')
    require(checkpoint['artifacts']['episodes.json']==sha(native/'episodes.json'),'Topic replay episode checkpoint changed')
    ident=read(native/'identity.json');paths=sorted((native/'calls').glob('topic_*.json'))
    require([int(p.name.split('_')[1]) for p in paths]==list(range(1,len(paths)+1)),'Topic generation order is incomplete')
    results={};source_by_topic={};position=0;stopped=None
    context={'seed':ident['input_sha256'],'counter':checkpoint['uuid_counter']}
    token=runtime._UUID_CONTEXT.set(context)
    class Retained:
        async def generate(self,prompt,response_format=None,**kwargs):
            nonlocal position,stopped
            frame=inspect.currentframe().f_back;local=frame.f_locals;stage=runtime._GENERATION_STAGES.get(frame.f_code.co_name)
            require(position<len(paths),'Topic replay produced an unrecorded request')
            p=paths[position];position+=1;n=read(p)
            req={'prompt':prompt,'response_format':response_format,'stage':stage,'json_output':True,'max_output':16384,'system':None}
            require(n['request']==req and n['request_sha256']==runtime._sha(req),'Original topic replay full prompt/order differs')
            sig=signature(req);key=digest(sig);rr=records.get(key)
            if n['status']!='success':
                require(allow_terminal_failure and position==len(paths) and n['status']=='failed'
                    and n.get('error_type')=='ReviewedOfflineStop' and terminal_stop is not None
                    and terminal_stop['signature']==sig and terminal_stop['request_hash']==key and terminal_stop['stage']==stage,
                    'Unexpected failed native topic record; only an exact reviewed offline sentinel is admitted')
                stopped={'native_call':binding(p),'request_hash':key,'stage':stage};raise _TopicStop()
            require(rr is not None and rr['signature']==sig,'Topic replay has no exact raw producer')
            original=decoded_raw(rr,proof);value=derived_value(original,stage,local,rr,proof)
            require(n['raw_response']==original and n['parsed_response']==value,'Recorded successful topic value is not the exact raw-derived value')
            capture={'native_call':binding(p),'native_request':req,'request_hash':key,'stage':stage,
                'local':adapter._Generation._local_record(local),'derived_parsed_sha256':runtime._sha(value)}
            if stage=='hypermem_topic_create':
                # The next UUID call is the original constructor's own uuid4.
                tid='topic_'+str(uuid.uuid5(uuid.NAMESPACE_URL,f"hypermem:{context['seed']}:{context['counter']+1}"))
                source_by_topic[tid]=binding(p)
            elif stage=='hypermem_topic_update':source_by_topic[local['topic'].topic_id]=binding(p)
            elif stage=='hypermem_episode_roles':
                topic=local['topic'];ordered=local['episodes'];td=topic.to_dict()
                require(topic.topic_id in source_by_topic,'Actual caller-local topic has no preceding unique generation source')
                require(len({e.event_id for e in ordered})==len(ordered),'Role caller has duplicate episode identities')
                capture.update(local_topic=td,local_episodes=[e.to_dict() for e in ordered],
                    simple_to_real=dict(local['simple_to_real']),topic_generation_source=source_by_topic[topic.topic_id])
            results[p.name]=capture
            return json.dumps(value,ensure_ascii=False)
    coro=stage2.extract_topics_for_episodes(eps,Retained())
    try:
        try:yielded=coro.send(None)
        except StopIteration:pass
        except _TopicStop:pass
        else:raise ValueError('Offline topic replay unexpectedly suspended: '+type(yielded).__name__)
        require(position==len(paths),'Recorded topic calls were not fully consumed')
        require(stopped is None or allow_terminal_failure,'A final audit may not accept a stopped partial topic replay')
    finally:
        coro.close();runtime._UUID_CONTEXT.reset(token)
    return {'status':'passed_complete' if stopped is None else 'known_terminal_stop','calls':results,
        'terminal_stop':stopped,'raw_derived_only':True,'topic_journals_used_as_inputs':False,
        'model_calls':0,'index_or_encoder_calls':0}

def audit_replacements(folder,proof,records):
    m=read(folder/'run_manifest.json');prefix.require_human_grant(m.get('human_grant'),proof)
    paths={p.stem:p for p in (folder/'work/replacement_v4').glob('*.json')}
    require(set(paths)==set(proof['replacement_cases']),'Replacement consumption set differs')
    out=[]
    for key,cert in proof['replacement_cases'].items():
        p=paths[key];v=read(p);intent=read(folder/'work/api_method/request_intents'/f'{key}.json')
        expected={'schema':'hypermem-v4-exclusive-role-replacement-v1','created_utc':v['created_utc'],
            'manifest':binding(folder/'run_manifest.json'),'terminal_proof':m['terminal_proof'],'human_grant':m['human_grant'],
            'certificate':cert,'context':cert['context'],'signature':cert['signature'],'stage':cert['stage'],
            'request_hash':key,'prefix_ordinal':cert['prefix_ordinal'],'permission_consumed_before_transport':True,'logical_replacements':1}
        require(v==expected and datetime.fromisoformat(v['created_utc'])<=datetime.fromisoformat(intent['created_utc']), 'Replacement marker/transport order differs')
        require(p.stat().st_mtime<=datetime.fromisoformat(intent['created_utc']).timestamp()+0.001,'Replacement marker postdates transport intent')
        require(binding(cert['raw']['path'])==cert['raw'] and key in records and records[key]['signature']==cert['signature'],'Original replacement evidence differs')
        require(key!=admission.CONSUMED_OLD_KEY,'Consumed old c5 key cannot be replaced')
        out.append(binding(p))
    return out

_api_source=old._api_source.replace("key==proof['replacement']['request_hash']","key in proof['replacement_cases']")
_ns={**old.__dict__,**globals()};exec(compile(_api_source,'<v4-private-frozen-api-audit>','exec'),_ns);_raw_api=_ns['api_audit']
def api_audit(root,proof):
    report,records,events,calls=_raw_api(root,proof)
    for key,r in records.items():
        if key in proof['producer_registry']:
            oldrow=proof['producer_registry'][key]
            require(r['signature']==oldrow['signature'] and all(e['stage'] in oldrow['stages'] for e in r['events']),'Known full request/stage differs')
            if key not in proof['replacement_cases']:require(r['raw']['sha256']==oldrow['raw']['sha256'],'Known successful raw regenerated')
    if Path(root).name=='api_method':audit_replacements(Path(root).parent.parent,proof,records)
    return report,records,events,calls

_prefix_source=inspect.getsource(old.audit_prefixes).replace('prefix_v3','prefix_v4').replace('v3-prefix','v4-prefix')
_prefix_source=_replace(_prefix_source,"binding(folder/'work/replacement_v3/known_invalid_episode_once.json')","binding(folder/'work/replacement_v4'/(e['request_hash']+'.json'))")
_prefix_source=_prefix_source.replace('qualified_original_invalid_json_replacement','qualified_original_role_uuid_replacement')
_prefix_source=_prefix_source.replace("key==proof['replacement']['request_hash']","key in proof['replacement_cases']")
_prefix_source=_replace(_prefix_source,"elif e['mode']=='inherit':require(new['raw_response']==old['raw_response'],'Qualified failed raw changed')",
    "elif e['mode']=='inherit': verify_compatible_failed_prefix(new,old,e,proof)")

def verify_compatible_failed_prefix(new,oldvalue,e,proof):
    cert=proof['compatibility_cases'].get(e['request_hash']);require(cert is not None,'Unknown inherited failed prefix')
    raw=read(e['raw']['path']);text=response_text(raw['response'])
    if cert['conversation'] in adapter.INSERTIONS:
        value,_=adapter.derive_exact_text(text,cert);require(new['raw_response']==text and new['parsed_response']==value,'Failed syntax prefix derivation differs')
    else:require(new['raw_response']==oldvalue['raw_response']==new['parsed_response']==json.loads(text),'False-row prefix payload changed')

_ns={**old.__dict__,**globals()};exec(compile(_prefix_source,'<v4-private-ordered-prefix-audit>','exec'),_ns);audit_prefixes=_ns['audit_prefixes']

def verify_representation(native,p,value,parsed,rr,proof,ci):
    # A full deterministic caller replay is shared once per conversation audit.
    call=_TOPIC_REPLAY.get(str(native),{}).get(p.name)
    key=digest(rr['signature']);cert=proof['compatibility_cases'].get(key)
    m=read(native.parents[2]/'run_manifest.json')
    if cert is not None:
        require(call is not None,'Exact compatibility lacks original topic pipeline replay')
        jp=native/'compatibility_v4'/p.name;adapter._runtime._verify_seal(jp);j=read(jp)
        expected=derived_value(parsed,value['request']['stage'],call['local'],rr,proof)
        journal=adapter.derive_exact_text(parsed,cert)[1] if ci in adapter.INSERTIONS else adapter.false_row_contract(parsed,call['local'])
        evidence={'certificate':cert,'native_request':value['request'],'native_call_path':str(p),'local':call['local'],'derivation':journal}
        receipt={'schema':'hypermem-v4-exact-compatibility-authorization-v1','status':'authorized','manifest':binding(native.parents[2]/'run_manifest.json'),
            'terminal_proof':m['terminal_proof'],'request_hash':key,'context':{'method':'hypermem','conversation':ci,'phase':'build'},
            'stage':value['request']['stage'],'raw':rr['raw'],'evidence_sha256':adapter._runtime._sha(evidence),'model_regeneration_permitted':False}
        require(value['raw_response']==parsed and value['parsed_response']==expected and j=={**evidence,'authorization':receipt},'Exact syntax/consumer journal differs from native replay')
        require(not (native/'representations'/p.name).exists(),'Unexpected role journal for exact syntax/consumer response');return
    require(not (native/'compatibility_v4'/p.name).exists(),'Unknown compatibility journal')
    if value['parsed_response']==parsed:
        require(value['raw_response']==parsed and not (native/'representations'/p.name).exists(),'Unchanged response has extra representation');return
    require(value['request']['stage']=='hypermem_episode_roles' and call is not None and value['raw_response']==parsed,'Unadmitted response transformation')
    jp=native/'representations'/p.name;adapter._runtime._verify_seal(jp);j=read(jp)
    derived,pure=adapter.represent_episode_ids(parsed,call['simple_to_real'])
    require(pure is not None and derived==value['parsed_response'] and all(j.get(k)==v for k,v in pure.items()),'ID-only representation differs')
    require(j['native_request']==value['request'] and j['native_call_path']==str(p)
        and j['local_topic']==call['local_topic'] and j['local_episodes']==call['local_episodes']
        and j['simple_to_real']==call['simple_to_real'],'Complete actual caller-local objects/source/order differ')
    qualified=[r for r in proof['role_cases'] if r['request_hash']==key and r['conversation']==ci]
    scope='inherited_successful_role_response' if qualified else 'new_exact_bijection_response'
    if qualified:require(all(r['raw']['sha256']==rr['raw']['sha256'] and r['value']['local_topic']==j['local_topic'] and r['value']['local_episodes']==j['local_episodes'] for r in qualified),'Inherited role value/scope changed')
    else:require(key not in proof['producer_registry'] or key in proof['replacement_cases'],'Unknown old role transformation')
    evidence={k:v for k,v in j.items() if k!='authorization'}
    require(j['authorization']=={'schema':'hypermem-v4-role-representation-authorization-v1','status':'authorized','scope':scope,
        'manifest':binding(native.parents[2]/'run_manifest.json'),'terminal_proof':m['terminal_proof'],'request_hash':key,'signature_sha256':key,
        'context':{'method':'hypermem','conversation':ci,'phase':'build'},'stage':'hypermem_episode_roles','raw':rr['raw'],
        'native_request_sha256':adapter._runtime._sha(value['request']),'evidence_sha256':adapter._runtime._sha(evidence),
        'original_parsed_sha256':pure['original_parsed_sha256'],'derived_parsed_sha256':pure['derived_parsed_sha256'],'model_regeneration_permitted':False},'Role authorization receipt differs')

_TOPIC_REPLAY={}
_native_source=old._native_source
_native_source=_replace(_native_source,"parsed=json.loads(response_text(rr['raw_value']['response']))","parsed=decoded_raw(rr,proof)")
_native_source=_replace(_native_source,"expected_journals={p.name for p in (native/'calls').glob('*.json') if read(p)['parsed_response']!=read(p)['raw_response']}",
    "expected_journals={p.name for p in (native/'calls').glob('*hypermem_episode_roles.json') if read(p)['parsed_response']!=read(p)['raw_response']}")
_ns={**old.__dict__,**globals()};exec(compile(_native_source,'<v4-private-complete-native-audit>','exec'),_ns);_full_native=_ns['native_audit']
def native_audit(folder,proof,records,events):
    try:
        replays={}
        for ci in range(10):
            native=Path(folder)/f'work/native/c{ci}';r=replay_topic_callers(native,records,proof)
            _TOPIC_REPLAY[str(native)]=r['calls'];replays[str(ci)]=r
            expected={p.name for p in (native/'calls').glob('*.json') if digest(signature(read(p)['request'])) in proof['compatibility_cases']}
            require({p.name for p in (native/'compatibility_v4').glob('*.json')}==expected,'Exact compatibility journal set differs')
        report=_full_native(folder,proof,records,events)
        report['actual_topic_pipeline_replays']=replays
        report['historical_refusals_preserved']=proof['original_representation_auditor_refusals']
        return report
    finally:_TOPIC_REPLAY.clear()

_all_source=old._all_source
_start="require(admission.inventory(admission.ANCESTOR)==proof['original_inventory'] and admission.inventory(admission.OLD)==proof['current_inventory'],'Original/parent inventory changed')"
_all_source=_replace(_all_source,_start,"require(all(admission.inventory(p)==proof['run_inventories'][k] for k,p in proof['run_roots'].items()),'An immutable historical root changed')")
_a=_all_source.index('    old_calls=[c for r in proof')
_all_source=_all_source[:_a]+'''    historical=proof['corrected_historical_usage']
    prior=[]
    for name,path in proof['run_roots'].items():
        prior.extend(jsonl(Path(path)/'work/api_method/api_calls.jsonl'))
    require(admission.usage_presence(prior)==historical['combined'],'Historical corrected usage differs')
    return {'status':'passed','errors':[],'api':api,'native':native,'prefix':prefix_report,
        'historical_usage':historical,'new_usage':admission.usage_presence(all_calls),
        'combined_physical_usage':admission.usage_presence(prior+all_calls),
        'historical_inventories_sha256':digest(proof['run_inventories']),
        'source_bindings':proof['source_bindings'],'historical_refusals_waived':False}
'''
_ns={**old.__dict__,**globals()};exec(compile(_all_source,'<v4-private-original-qa-seal-audit>','exec'),_ns);audit_all=_ns['audit_all']
