"""Strict v3 producer/prefix/representation/native/QA audit.

The original v2 API/native/QA checks are compiled privately from exact locked
source. Explicit small substitutions add the multi-call prefix, single original
JSON replacement, and reversible ID evidence; index/vector checks stay live.
"""
from __future__ import annotations
import ast
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from tools import audit_hypermem_terminal_v3 as admission
from tools import audit_simplemem_terminal_v2 as common
from tools import baselines_hypermem_native_v3 as adapter
from tools import hypermem_prefix_api_v3 as transport
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,response_text,check_request,READER_STAGE,JUDGE_STAGE

FROZEN_AUDIT_SHA='5f30bdea08a823923f56c5fd9dd28df65f5b94043c75f0ef27630af8b713cd63'
require,read,sha,binding,jsonl=common.require,common.read,common.file_sha,common.binding,common.jsonl
STAGES=set(transport.STAGES)
_path=Path(__file__).with_name('audit_hypermem_continuation_v2.py')
require(sha(_path)==FROZEN_AUDIT_SHA,'Frozen continuation audit differs')
_text=_path.read_text('utf8');_tree=ast.parse(_text)


def _source(name):
    node=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(_text.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def _replace(source,old,new):
    require(source.count(old)==1,'Private audit substitution boundary changed: '+old[:90])
    return source.replace(old,new)


def audit_replacement(folder,proof,records):
    cert=proof['replacement']; key=cert['request_hash']
    marker=Path(folder)/'work/replacement_v3/known_invalid_episode_once.json'
    m=read(Path(folder)/'run_manifest.json')
    expected={'schema':'hypermem-v3-exclusive-json-replacement-v1',
        'manifest':binding(Path(folder)/'run_manifest.json'),'terminal_proof':m['terminal_proof'],
        'certificate':cert,'context':cert['context'],'signature':cert['signature'],
        'stage':cert['stage'],'request_hash':key,'prefix_ordinal':cert['prefix_ordinal'],
        'permission_consumed_before_transport':True,'logical_replacements':1}
    require(m['continuation_policy']['known_invalid_episode_replacement'] is True and read(marker)==expected,
        'Exact one-use JSON replacement authority differs')
    require({p.name for p in marker.parent.iterdir()}=={marker.name},'Extra replacement authority')
    intent=read(Path(folder)/'work/api_method/request_intents'/(key+'.json'))
    require(intent['signature']==cert['signature'] and intent['context']==cert['context'] and intent['stage']==cert['stage'],
        'Replacement full request differs')
    # Marker creation is an exclusive before-send action; filesystem times are
    # additionally checked against the durable transport intent creation time.
    require(marker.stat().st_mtime <= datetime.fromisoformat(intent['created_utc']).timestamp()+0.001,
        'Replacement marker postdates its transport intent')
    require(binding(cert['raw']['path'])==cert['raw'],'Original malformed raw changed')
    try:json.loads(response_text(read(cert['raw']['path'])['response']))
    except json.JSONDecodeError:pass
    else:raise ValueError('Original replacement no longer matches malformed JSON')
    require(key in records and records[key]['signature']==cert['signature'],'Replacement producer absent')
    json.loads(response_text(records[key]['raw_value']['response']))
    return binding(marker)


def api_audit(root,proof):
    # Frozen checks still account for every intent, physical attempt, raw,
    # logical consumer, cache use and embedding admission. Only original-key
    # membership and the qualified replacement differ from v2.
    report,records,events,calls=_api_audit(Path(root),proof)
    registry=proof['producer_registry']; badkey=proof['replacement']['request_hash']
    for key,r in records.items():
        if key in registry:
            original=registry[key]
            require(r['signature']==original['signature'] and all(e['stage'] in original['stages'] for e in r['events']),
                'Known signature/stage differs')
            if key!=badkey:
                require(r['raw']['sha256']==original['raw']['sha256'],'Known raw regenerated')
    if Path(root).name=='api_method':audit_replacement(Path(root).parent.parent,proof,records)
    return report,records,events,calls


def audit_prefixes(folder,proof,all_events):
    folder=Path(folder);base=folder/'work/prefix_v3';m=read(folder/'run_manifest.json')
    require({p.name for p in base.iterdir()}=={f'c{i}' for i in range(10)},'Prefix conversation set differs')
    reports=[]
    for ci,case in enumerate(proof['conversations']):
        prefix=case['prefix'];directory=base/f'c{ci}'
        expected_files={'complete.json','requests.jsonl'}|{f'{i:06d}.{kind}.json' for i in range(1,len(prefix)+1) for kind in ('intent','complete')}
        require({p.name for p in directory.iterdir()}==expected_files,'Prefix slots are missing/extra')
        require(read(directory/'complete.json')=={'conversation':ci,'prefix_count':len(prefix),
            'prefix_sha256':digest(prefix),'terminal_proof':m['terminal_proof']},'Incomplete original prefix')
        journal=jsonl(directory/'requests.jsonl'); require(len(journal)>=len(prefix),'Short prefix journal')
        expected=[]
        for i,e in enumerate(prefix):
            require(read(directory/f'{i+1:06d}.intent.json')=={'schema':'hypermem-v3-prefix-slot-intent-v1',
                'ordinal':i+1,'conversation':ci,'signature':e['signature'],'context':e['context'],
                'stage':e['stage'],'request_hash':e['request_hash'],'original_raw':e['raw'],
                'mode':e['mode'],'original_native_call':e['native_call']},'Prefix intent differs')
            raw=folder/'work/api_method/api_cache'/(e['request_hash']+'.json')
            require(read(directory/f'{i+1:06d}.complete.json')=={'schema':'hypermem-v3-prefix-slot-complete-v1',
                'ordinal':i+1,'conversation':ci,'request_hash':e['request_hash'],'mode':e['mode'],
                'original_raw':e['raw'],'current_raw':binding(raw)},'Prefix completion differs')
            if e['mode']=='inherit':require(sha(raw)==e['raw']['sha256'],'Prefix inherited bytes differ')
            expected.append({'event':'request_admitted','scope':'original_prefix','ordinal':i+1,
                'request_hash':e['request_hash'],'context':e['context'],'stage':e['stage'],'operation':e['operation'],
                'source':'qualified_original_exact_raw' if e['mode']=='inherit' else 'qualified_original_invalid_json_replacement',
                'replacement_marker':None if e['mode']=='inherit' else binding(folder/'work/replacement_v3/known_invalid_episode_once.json')})
            old=read(e['native_call']['path']);new=read(folder/f'work/native/c{ci}/calls'/Path(e['native_call']['path']).name)
            require(new['status']=='success' and new['request']==old['request'] and new['request_sha256']==old['request_sha256'],
                'Native original prefix order/request/parsed completion differs')
            if old['status']=='success':require(new==old,'Successful original native value changed')
            elif e['mode']=='inherit':require(new['raw_response']==old['raw_response'],'Qualified failed raw changed')
        require(journal[:len(prefix)]==expected,'Prefix journal order/multiplicity differs')
        ee=[e for e in all_events if e['context']['conversation']==ci]
        require(len(ee)==len(journal),'Admission/logical count differs across phases')
        first=ee[:len(prefix)]
        require([(e['request_hash'],e['stage'],e['context']) for e in first]==[(e['request_hash'],e['stage'],e['context']) for e in prefix],
                'Actual original logical prefix differs')
        expected_suffix=[]
        for e in ee[len(prefix):]:
            key=e['request_hash']
            source='new_namespace_request'
            if key in proof['producer_registry']:
                source='existing_replacement_cache' if key==proof['replacement']['request_hash'] else 'known_parent_exact_raw'
            expected_suffix.append({'event':'request_admitted','scope':'suffix','ordinal':None,'request_hash':key,
                'context':e['context'],'stage':e['stage'],'operation':e['operation'],'source':source,'replacement_marker':None})
        require(Counter(map(digest,journal[len(prefix):]))==Counter(map(digest,expected_suffix)),
                'Suffix/QA admission journal has extra or missing consumers')
        reports.append({'conversation':ci,'prefix_count':len(prefix),'logical_consumers':len(ee),'journal':binding(directory/'requests.jsonl')})
    return reports


def verify_representation(native,p,value,parsed,rr,proof,ci):
    journal_path=native/'representations'/p.name
    if value['parsed_response']==parsed:
        require(not journal_path.exists(),'Unexpected normalization journal')
        require(value['raw_response']==parsed,'Raw native value differs')
        return
    require(value['request']['stage']=='hypermem_episode_roles' and value['raw_response']==parsed,'Unadmitted value mutation')
    adapter._runtime._verify_seal(journal_path);j=read(journal_path)
    derived,pure=adapter.represent_episode_ids(parsed,j['simple_to_real'])
    require(pure and all(j.get(k)==v for k,v in pure.items()) and derived==value['parsed_response'], 'ID derivation differs')
    evidence={k:v for k,v in j.items() if k!='authorization'}
    require(j['native_call_path']==str(p) and j['native_request']==value['request'], 'Representation native request/owner differs')
    local=j['local_episodes'];mapping={f'episode_{i+1}':e['event_id'] for i,e in enumerate(local)}
    require(mapping==j['simple_to_real'],'Actual local ordered identities differ')
    episodes=read(native/'episodes.json');byid={e['event_id']:e for e in episodes}
    require(len(byid)==len(episodes) and all(byid.get(e['event_id'])==e for e in local),'Local episodes differ from real retained episode bank')
    modules=adapter._runtime._load();extractor=modules['extractors/topic_extractor.py'].TopicExtractor
    formatter=object.__new__(extractor)
    def obj(v):
        v=deepcopy(v)
        if isinstance(v.get('timestamp'),str):v['timestamp']=datetime.fromisoformat(v['timestamp'])
        return SimpleNamespace(**v)
    topic=obj(j['local_topic']);eps=[obj(e) for e in local]
    template=modules['prompts/topic_prompts.py'].EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT
    expected_prompt=template.format(topic_content=formatter._format_topic_display(topic),episodes=formatter._format_episode_list(eps,use_simple_ids=True))
    require(value['request']['prompt']==expected_prompt,'Role prompt does not describe the exact retained local evidence')
    ordinal=int(p.name.split('_')[1]);prior=list(p.parent.glob(f'topic_{ordinal-1:06d}_*.json'))
    require(len(prior)==1,'Role assignment lacks unique prior topic generation')
    topic_call=read(prior[0]);td=topic_call['parsed_response']
    require(topic_call['status']=='success' and topic_call['request']['stage'] in ('hypermem_topic_create','hypermem_topic_update')
        and all(j['local_topic'][k]==td[k] for k in ('title','summary','keywords')),'Current local topic differs from actual prior generation')
    key=digest(rr['signature']);m=read(native.parents[2]/'run_manifest.json')
    qualified=next((c for c in proof['role_cases'] if c['request_hash']==key),None)
    if qualified:
        require(qualified['conversation']==ci and qualified['raw']['sha256']==rr['raw']['sha256']
            and {e['simple_id']:e['event_id'] for e in qualified['mapping_from_complete_unique_native_display']}==mapping,
            'Original role qualification differs')
        scope='qualified_original_response' if Path(qualified['native_call']['path']).name==p.name else 'qualified_original_response_reused_suffix'
    else:
        require(m['continuation_policy']['prospective_role_representation'] is True and key not in proof['producer_registry'],
            'Future role representation not reviewed')
        scope='first_seen_suffix_representation'
    require(j['authorization']=={'schema':'hypermem-v3-role-representation-authorization-v1','status':'authorized','scope':scope,
        'manifest':binding(native.parents[2]/'run_manifest.json'),'terminal_proof':m['terminal_proof'],'role_qualification':proof['role_qualification'],
        'request_hash':key,'signature_sha256':digest(rr['signature']),'context':{'method':'hypermem','conversation':ci,'phase':'build'},
        'stage':'hypermem_episode_roles','raw':rr['raw'],'native_request_sha256':adapter._runtime._sha(value['request']),
        'evidence_sha256':adapter._runtime._sha(evidence),'original_parsed_sha256':pure['original_parsed_sha256'],
        'derived_parsed_sha256':pure['derived_parsed_sha256'],'model_regeneration_permitted':False},'Representation authority receipt differs')


# Private frozen checks, with exact source boundaries and published hashes.
_api_source=_source('api_audit')
_api_source=_api_source.replace("proof['registry']","proof['producer_registry']")
_api_source=_replace(_api_source,"and all(e['context']==origin['context'] for e in es)","and all(e['context'].get('phase')=='build' for e in es)")
_api_source=_replace(_api_source,"require(key not in proof['producer_registry'], 'Original physical request was regenerated')",
    "require(key not in proof['producer_registry'] or (root.name=='api_method' and key==proof['replacement']['request_hash']), 'Original physical request was regenerated')")
_namespace=dict(globals());exec(compile(_api_source,'<v3-private-frozen-api-audit>','exec'),_namespace);_api_audit=_namespace['api_audit']

_native_source=_source('native_audit')
_start="    require({p.name for p in (folder/'work/prefix').iterdir()}"
_a=_native_source.index(_start);_b=_native_source.index('    data=admission.chain.base.load_data()',_a)
_native_source=_native_source[:_a]+_native_source[_b:]
_a=_native_source.index("        case=proof['rows'][ci]");_b=_native_source.index('        conv=admission.chain.base.conversation_only',_a)
_native_source=_native_source[:_a]+"        case=proof['conversations'][ci]\n"+_native_source[_b:]
_native_source=_replace(_native_source,"require(rr['signature']==sig and value['parsed_response']==value['raw_response']==parsed,'Native value differs from exact raw')",
    "require(rr['signature']==sig,'Native signature differs from exact raw')\n            verify_representation(native,p,value,parsed,rr,proof,ci)")
_a=_native_source.index("            if p.name=='episode_000001_hypermem_boundary.json':");_b=_native_source.index('            native_bindings.append(binding(p))',_a)
_native_source=_native_source[:_a]+_native_source[_b:]
_native_source=_replace(_native_source,"        for p in sorted((native/'embedding_evidence').glob('*.json')):",
    "        expected_journals={p.name for p in (native/'calls').glob('*.json') if read(p)['parsed_response']!=read(p)['raw_response']}\n        require({p.name for p in (native/'representations').glob('*.json')}==expected_journals,'Extra/missing representation journal')\n        for p in sorted((native/'embedding_evidence').glob('*.json')):")
_namespace=dict(globals());exec(compile(_native_source,'<v3-private-frozen-native-audit>','exec'),_namespace);native_audit=_namespace['native_audit']

_all_source=_source('audit_all')
_all_source=_replace(_all_source,"require(admission.inventory(admission.OLD)==proof['original_inventory'],'Original parent inventory changed')",
    "require(admission.inventory(admission.ANCESTOR)==proof['original_inventory'] and admission.inventory(admission.OLD)==proof['current_inventory'],'Original/parent inventory changed')")
_all_source=_replace(_all_source,'api=[]; all_calls=[]; all_events=[]; method=None','api=[]; all_calls=[]; all_events=[]; method=None; phase_events=[]')
_all_source=_replace(_all_source,'report,records,events,calls=api_audit(root,proof);api.append(report);all_calls.extend(calls)',
    'report,records,events,calls=api_audit(root,proof);api.append(report);all_calls.extend(calls);phase_events.extend(events)')
_all_source=_replace(_all_source,'native=native_audit(folder,proof,*method)',
    "prefix_report=audit_prefixes(folder,proof,sorted(phase_events,key=lambda e:e['recorded_utc']))\n    native=native_audit(folder,proof,*method)")
_a=_all_source.index("    return {'status':'passed'")
_all_source=_all_source[:_a]+'''    old_calls=[c for r in proof['producer_registry'].values() for c in r['physical_attempts']]
    inherited_calls=[r['inherited_parent']['physical'] for r in proof['producer_registry'].values() if r.get('inherited_parent')]
    require(len(inherited_calls)==10,'Original ten physical producer set differs')
    return {'status':'passed','errors':[],'api':api,'native':native,'prefix':prefix_report,
        'ancestor_usage':admission.usage_presence(inherited_calls),'parent_usage':admission.usage_presence(old_calls),
        'new_usage':admission.usage_presence(all_calls),
        'combined_physical_usage':admission.usage_presence(inherited_calls+old_calls+all_calls),
        'ancestor_inventory_sha256':digest(proof['original_inventory']),
        'parent_inventory_sha256':digest(proof['current_inventory'])}
'''
_namespace=dict(globals());exec(compile(_all_source,'<v3-private-frozen-all-audit>','exec'),_namespace);audit_all=_namespace['audit_all']


def implementation():
    return {'frozen_v2_audit_sha256':FROZEN_AUDIT_SHA,
        'private_function_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in
            (('api_audit',_api_source),('native_audit',_native_source),('audit_all',_all_source))},
        'scope':'Original live index/vector/raw/native/QA checks plus complete ordered source prefix and separately qualified representation/replacement.'}
