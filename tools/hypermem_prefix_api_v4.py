"""Candidate ordered-prefix API; root-published human authorization is absent."""
from __future__ import annotations
from copy import deepcopy
import inspect
import json
from pathlib import Path
import textwrap
import threading
from types import MappingProxyType

from tools import hypermem_prefix_api_v3 as old
from tools import audit_hypermem_terminal_v4 as admission
from tools import baselines_hypermem_native_v4 as adapter
from tools.baselines_common_v1 import BaselineAPI,BaselineFatal,append,read,save_new,sha,now
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,response_text

require=old.require;binding=old.binding;safety=old.safety;check_endpoint=old.check_endpoint
STAGES=old.STAGES;PROOF_SCHEMA=admission.SCHEMA
# Operational trust boundary: only root may exclusively publish this file after
# a real explicit user instruction. Local JSON is not cryptographic proof of a
# human's identity. No candidate/implementation review is itself authorization.
HUMAN_GRANT_PATH=admission.ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v4/human_authorization_v1.json'
REVIEW_PATH=HUMAN_GRANT_PATH.with_name('independent_candidate_review_v1.json')
PROOF_PATH=HUMAN_GRANT_PATH.with_name('terminal_candidate_v1.json')

class _OfflineKeys:
    """Truthy to suppress credential loading; no provider key can be obtained."""
    def __bool__(self):return True
    def values(self):return ()
    def __getitem__(self,key):raise BaselineFatal('Research API has no credential or transport capability')

class _OfflineTransport:
    def post(self,*args,**kwargs):raise BaselineFatal('Research API transport is disabled')
    def request(self,*args,**kwargs):raise BaselineFatal('Research API transport is disabled')

class _OfflineLocal:
    # PacedEmbeddingAPI may temporarily wrap this session; that wrapper's
    # delegate still cannot send. It cannot replace the object's transport.
    @property
    def session(self):return _OfflineTransport()
    @session.setter
    def session(self,value):pass

def require_human_grant(grant,proof):
    require(HUMAN_GRANT_PATH.is_file(),'No explicit human authorization file; no write or credential access is permitted')
    actual=binding(HUMAN_GRANT_PATH)
    require(grant is None or grant==actual,'Only the single fixed root-published authorization path is accepted')
    value=read(HUMAN_GRANT_PATH)
    require(value.get('schema')=='hypermem-native-v4-explicit-human-grant-v1' and value.get('status')=='authorized_by_explicit_user_instruction'
        and value.get('issued_by')=='root_after_trusted_user_confirmation'
        and value.get('source_bindings')==admission.fixed_bindings()
        and type(value.get('logical_limit_per_key')) is int and value['logical_limit_per_key']==1
        and value.get('old_consumed_key')==admission.CONSUMED_OLD_KEY and value.get('old_consumed_key_permission') is False,
        'Human grant scope differs')
    confirmation=value.get('human_confirmation',{})
    require(confirmation.get('actor')=='user' and confirmation.get('source')=='direct_user_message'
        and isinstance(confirmation.get('message_id'),str) and bool(confirmation['message_id'])
        and isinstance(confirmation.get('text'),str) and bool(confirmation['text'].strip())
        and adapter.text_sha(confirmation['text'])==confirmation.get('text_sha256'),'Trusted user confirmation reference is incomplete')
    require(value.get('terminal_proof')==binding(PROOF_PATH) and value.get('implementation_review')==binding(REVIEW_PATH),
        'Authorization does not bind the final fixed proof and review')
    review=read(REVIEW_PATH)
    require(review.get('schema')=='hypermem-native-v4-independent-candidate-review-v1'
        and review.get('status')=='passed_implementation_and_zero_api_preflight_only'
        and review.get('production_launch_authorized') is False
        and value.get('code_sha256')==review.get('code_sha256') and bool(value['code_sha256']), 'Authorization has no final independent candidate review/code binding')
    for path,h in value['code_sha256'].items():require(sha(safety.safe(admission.ROOT/path))==h,'Authorized source changed')
    proposal_ref=admission.fixed_bindings()['proposal'];require(binding(proposal_ref['path'])==proposal_ref,'Exact request proposal changed')
    proposal=read(proposal_ref['path'])
    compatibility_keys=sorted(r['request_hash'] for r in proposal['targets']
        if r['conversation'] in admission.COMPATIBLE_CONVERSATIONS)
    require(value.get('accepted_compatibility_request_hashes')==compatibility_keys,
        'Human confirmation must explicitly accept the six exact non-regeneration compatibility cases')
    expected=[{'conversation':r['conversation'],'request_hash':r['request_hash'],'signature':r['signature'],
        'context':r['context'],'stage':r['stage'],'raw':{'path':r['original_raw']['source_path'],'sha256':r['original_raw']['observed_sha256']},
        'logical_limit':1} for r in proposal['targets'] if r['conversation'] in admission.REPLACEMENT_CONVERSATIONS]
    require(isinstance(value.get('replacements'),list)
        and all(isinstance(row,dict) and type(row.get('logical_limit')) is int and row['logical_limit']==1 for row in value['replacements'])
        and digest(value['replacements'])==digest(expected),'Authorization does not cover exactly four complete original requests')
    if proof is not None:
        require(proof['source_bindings']==value['source_bindings'] and set(proof['replacement_cases'])=={r['request_hash'] for r in expected}
            and sorted(proof['compatibility_cases'])==compatibility_keys,'Proof authorization key set differs')
        for row in expected:
            cert=proof['replacement_cases'][row['request_hash']]
            require(digest({k:cert[k] for k in row if k!='logical_limit'})==digest({k:v for k,v in row.items() if k!='logical_limit'}),
                'Proof full replacement certificate differs')
    return value

class PrefixAPI(old.PrefixAPI):
    def __init__(self,run_dir,*,context,cache_roots=(),**kwargs):
        a=type(self).admission
        require(a is not None,'No v4 admission')
        if not a['research_only']:
            checker=getattr(type(self),'_check_admitted_grant',None)
            if checker is None:require_human_grant(a['human_grant'],a['proof'])
            else:checker(a)
        else:
            require(Path(run_dir).absolute().is_relative_to(admission.ROOT/'research'),'Offline prefix must remain outside all production roots')
            # Even explicit calls to inherited parent request methods encounter
            # this object's key/transport barrier. Fake keys are not required.
            kwargs['keys']=_OfflineKeys()
        _original_init(self,run_dir,context=context,cache_roots=cache_roots,**kwargs)
        if a['research_only']:self.local=_OfflineLocal()

    def _replacement(self,entry,signature,stage):
        a=type(self).admission;key=entry['request_hash']
        require(not a['research_only'],'Offline prefix stops before replacement transport')
        require_human_grant(a['human_grant'],a['proof'])
        cert=a['replacements'].get(key)
        require(cert is not None and self.ci==cert['conversation'] in admission.REPLACEMENT_CONVERSATIONS
            and cert['kind']=='known_complete_invalid_role_uuid_set' and type(cert['retry_limit']) is int and cert['retry_limit']==1
            and entry['ordinal']==cert['prefix_ordinal'] and entry['raw']==cert['raw']
            and signature==cert['signature'] and self.context==cert['context'] and stage==cert['stage']=='hypermem_episode_roles'
            and key!=admission.CONSUMED_OLD_KEY,'Replacement is not one of the four exact qualified role requests')
        origin=self._check_known_raw(key,signature,stage);require(origin['raw']==cert['raw'],'Replacement source changed')
        json.loads(response_text(read(cert['raw']['path'])['response']))
        marker=a['new_run']/'work/replacement_v4'/(key+'.json')
        save_new(marker,{'schema':'hypermem-v4-exclusive-role-replacement-v1','created_utc':now(),
            'manifest':a['manifest'],'terminal_proof':a['proof_binding'],'human_grant':a['human_grant'],
            'certificate':cert,'context':self.context,'signature':signature,'stage':stage,'request_hash':key,
            'prefix_ordinal':entry['ordinal'],'permission_consumed_before_transport':True,'logical_replacements':1})
        return binding(marker)

    def generate(self,prompt,*,model='gemini-2.5-flash',system=None,json_output=False,max_output=16384,stage='unspecified'):
        body=generation_body(prompt,system=system,json_output=json_output,max_output=max_output)
        sig={'provider':'gemini','model':model,'operation':'generateContent','body':body};key=digest(sig)
        cert=type(self).admission['compatibility'].get(key)
        if cert is None or cert['conversation'] not in adapter.INSERTIONS:
            return BaselineAPI.generate(self,prompt,model=model,system=system,json_output=json_output,max_output=max_output,stage=stage)
        require(cert['signature']==sig and cert['stage']==stage and cert['context']==self.context,'Exact parser certificate context differs')
        raw=self.request('gemini',model,body,stage=stage)
        path=self.cache_dir/(key+'.json')
        require(binding(path)['sha256']==cert['raw']['sha256'],'Exact syntax raw changed')
        text=response_text(raw);require(adapter.text_sha(text)==cert['response_text_sha256'],'Exact original text changed')
        return text  # native raw_response is original text, never the derived object

    def compatibility_certificate(self,*,stage,request,native_call_path):
        a=type(self).admission;sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
            'body':generation_body(request['prompt'],system=None,json_output=True,max_output=16384)}
        cert=a['compatibility'].get(digest(sig))
        if cert is None:return None
        require(cert['signature']==sig and cert['context']==self.context and cert['stage']==stage
            and request=={'prompt':request['prompt'],'response_format':{'type':'json_object'},'stage':stage,'json_output':True,'max_output':16384,'system':None}
            and safety.safe(native_call_path).parent==a['new_run']/f'work/native/c{self.ci}/calls'
            and Path(native_call_path).name==Path(cert['native_call']['path']).name
            and self.case['prefix'][cert['prefix_ordinal']-1]['request_hash']==cert['request_hash'],'Exact compatibility owner/request/ordinal differs')
        require(self._check_known_raw(cert['request_hash'],sig,stage)['raw']==cert['raw'],'Exact original SDK raw changed')
        return deepcopy(cert)

    def _consumer(self,key,stage,*,cached_native):
        a=type(self).admission;path=self.cache_dir/(key+'.json')
        events=self.events
        if cached_native:
            require(a['progress'][self.ci]==len(self.case['prefix']) and a['inflight'][self.ci] is None,'Native cache cannot establish prefix authority')
            from tools.audit_simplemem_terminal_v2 import jsonl
            events=jsonl(self.run_dir/'logical_events.jsonl')
        require(any(e['request_hash']==key and e['stage']==stage and e['context']==self.context
            and e['cache_path']==str(path) and e['cache_sha256']==sha(path) and e['error_type'] is None for e in events),'No exact successful retained consumer')
        return path

    def authorize_exact_compatibility(self,cert,*,request,native_call_path,original,derived,local,journal,cached_native=False):
        actual=self.compatibility_certificate(stage=request['stage'],request=request,native_call_path=native_call_path)
        require(actual==cert,'Compatibility certificate is not authoritative')
        rawpath=self._consumer(cert['request_hash'],cert['stage'],cached_native=cached_native)
        require(binding(rawpath)['sha256']==cert['raw']['sha256'],'Compatibility source bytes changed')
        text=response_text(read(rawpath)['response'])
        require(adapter.text_sha(text)==cert['response_text_sha256'],'Compatibility original text differs')
        if cert['conversation'] in adapter.INSERTIONS:
            expected,j=adapter.derive_exact_text(text,cert)
            require(original==text and derived==expected and journal==j,'Exact syntax derivation differs')
        else:
            require(original==derived==json.loads(text),'False-row payload was modified')
            require(journal==adapter.false_row_contract(original,local),'False-row decision evidence differs')
        # Audit additionally reconstructs these objects from the original native
        # source. Runtime authority binds every full object and its order now.
        require('simple_to_real' in local if cert['stage']=='hypermem_topic_match' else 'topic' in local,'Missing actual caller-local scope')
        if cert['stage']=='hypermem_topic_match':
            require('topics_batch' in local and 'episode' in local
                and {f'topic_{i+1}':t['topic_id'] for i,t in enumerate(local['topics_batch'])}==local['simple_to_real'],'Actual topic batch identities differ')
        else:require('new_episode' in local,'Topic-update compatibility lacks its actual new_episode')
        evidence={'certificate':cert,'native_request':request,'native_call_path':native_call_path,'local':local,'derivation':journal}
        a=type(self).admission
        receipt={'schema':'hypermem-v4-exact-compatibility-authorization-v1','status':'authorized',
            'manifest':a['manifest'],'terminal_proof':a['proof_binding'],'request_hash':cert['request_hash'],
            'context':self.context,'stage':cert['stage'],'raw':binding(rawpath),'evidence_sha256':adapter._runtime._sha(evidence),
            'model_regeneration_permitted':False}
        if cached_native:
            jp=Path(native_call_path).parent.parent/'compatibility_v4'/Path(native_call_path).name
            adapter._runtime._verify_seal(jp)
            require(read(jp)=={**evidence,'authorization':receipt},'Cached compatibility journal differs')
        return receipt

    def authorize_episode_representation(self,evidence,*,original,derived,cached_native=False):
        a=type(self).admission;r=evidence['native_request']
        require(self.context=={'method':'hypermem','conversation':self.ci,'phase':'build'}
            and r=={'prompt':r['prompt'],'response_format':{'type':'json_object'},'stage':'hypermem_episode_roles','json_output':True,'max_output':16384,'system':None},'Role authority context differs')
        sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':generation_body(r['prompt'],system=None,json_output=True,max_output=16384)}
        key=digest(sig);rawpath=self._consumer(key,r['stage'],cached_native=cached_native)
        require(read(rawpath)['request']==sig and json.loads(response_text(read(rawpath)['response']))==original,'Role value differs from actual raw')
        path=safety.safe(evidence['native_call_path']);require(path.parent==a['new_run']/f'work/native/c{self.ci}/calls','Role native owner differs')
        expected,j=adapter.represent_episode_ids(original,evidence['simple_to_real'])
        require(j is not None and expected==derived and all(evidence.get(k)==v for k,v in j.items()),'Role mapping derivation differs')
        require({f'episode_{i+1}':e['event_id'] for i,e in enumerate(evidence['local_episodes'])}==evidence['simple_to_real'],'Role actual local order differs')
        matches=[c for c in a['role_cases'] if c['request_hash']==key and c['conversation']==self.ci]
        if matches:
            require(all(c['raw']['sha256']==sha(rawpath) and c['value']['simple_to_real']==evidence['simple_to_real']
                and c['value']['local_episodes']==evidence['local_episodes'] and c['value']['local_topic']==evidence['local_topic'] for c in matches),'Historical successful role scope differs')
            scope='inherited_successful_role_response'
        else:
            require(a['progress'][self.ci]==len(self.case['prefix']) and not a['research_only']
                and (key not in a['registry'] or key in a['replacements'])
                and (self.run_dir/'request_intents'/(key+'.json')).exists() and not (self.run_dir/'cache_lineage'/(key+'.json')).exists(),'Unknown role representation before complete prefix')
            scope='new_exact_bijection_response'
        receipt={'schema':'hypermem-v4-role-representation-authorization-v1','status':'authorized','scope':scope,
            'manifest':a['manifest'],'terminal_proof':a['proof_binding'],'request_hash':key,'signature_sha256':digest(sig),
            'context':self.context,'stage':r['stage'],'raw':binding(rawpath),'native_request_sha256':adapter._runtime._sha(r),
            'evidence_sha256':adapter._runtime._sha(evidence),'original_parsed_sha256':j['original_parsed_sha256'],
            'derived_parsed_sha256':j['derived_parsed_sha256'],'model_regeneration_permitted':False}
        if cached_native:
            jp=path.parent.parent/'representations'/path.name;adapter._runtime._verify_seal(jp)
            require(read(jp)=={**evidence,'authorization':receipt},'Cached role authorization differs')
        return receipt

_ns={**old.__dict__,**globals()}
_init=textwrap.dedent(inspect.getsource(old.PrefixAPI.__init__)).replace("'work/prefix_v3'","'work/prefix_v4'").replace('super().__init__(', 'BaselineAPI.__init__(self,')
exec(compile(_init,'<v4-private-prefix-constructor>','exec'),_ns);_original_init=_ns['__init__']
_request=textwrap.dedent(inspect.getsource(old.PrefixAPI.request)).replace('v3-prefix','v4-prefix').replace('super().request(','BaselineAPI.request(self,')
require(_request.count("cert = a['replacement']")==1,'Frozen suffix replacement dispatch changed')
_request=_request.replace("cert = a['replacement']","cert = a['replacements'].get(key)")
_request=_request.replace("source = 'new_namespace_request'","require(not a['research_only'], 'Offline prefix stops before unseen transport')\n                source = 'new_namespace_request'")
# The replacement branch is still a single exclusive last-prefix slot.
_request=_request.replace('qualified_original_invalid_json_replacement','qualified_original_role_uuid_replacement')
exec(compile(_request,'<v4-private-ordered-prefix-request>','exec'),_ns);PrefixAPI.request=_ns['request']

def admitted_api(proof,new_run,*,manifest,proof_binding,human_grant=None,research_only=False):
    require(proof.get('schema')==PROOF_SCHEMA and proof.get('status')=='candidate_evidence_only' and type(research_only) is bool,'Complete candidate proof required')
    path=Path(new_run).absolute()
    if research_only:require(path.is_relative_to(admission.ROOT/'research') and path!=admission.NEW,'Only isolated research prefix validation is permitted')
    else:
        require_human_grant(human_grant,proof);human_grant=binding(HUMAN_GRANT_PATH)
    # The large proof and complete code map are checked above once for this
    # admitted class, and again by prepare/run/ticket/final audit. Every scoped
    # API checks the small fixed grant/review bytes; it does not reread a huge
    # proof for each of the 1,540 questions.
    admitted_grant=deepcopy(human_grant)
    admitted_review=binding(REVIEW_PATH) if not research_only else None
    conversations=tuple(MappingProxyType({**c,'prefix':tuple(c['prefix'])}) for c in proof['conversations'])
    require([c['conversation'] for c in conversations]==list(range(10)) and all(c['prefix'] for c in conversations),'Whole ten-conversation prefix required')
    a={'executing':True,'research_only':research_only,'proof':proof,'human_grant':human_grant,
        'new_run':path,'manifest':deepcopy(manifest),'proof_binding':deepcopy(proof_binding),'conversations':conversations,
        'registry':MappingProxyType(proof['producer_registry']),'compatibility':MappingProxyType(proof['compatibility_cases']),
        'replacements':MappingProxyType(proof['replacement_cases']),'role_cases':tuple(proof['role_cases']),
        'questions':{q['query_id']:q['conversation'] for q in proof['questions']},
        'prefix_digests':{ci:digest(c['prefix']) for ci,c in enumerate(conversations)},
        'progress':{ci:0 for ci in range(10)},'inflight':{ci:None for ci in range(10)},
        'locks':{ci:threading.RLock() for ci in range(10)},'registry_lock':threading.RLock(),'key_locks':{}}
    def check_admitted(current):
        require(current is a and current['proof'] is proof and current['research_only'] is False
            and current['human_grant']==admitted_grant,'Admitted API authority changed')
        require(binding(HUMAN_GRANT_PATH)==admitted_grant and binding(REVIEW_PATH)==admitted_review,
            'Fixed human authorization or reviewed admission changed')
    return type('CandidateHyperMemV4API',(PrefixAPI,),{'admission':a,
        '_check_admitted_grant':staticmethod(check_admitted)})
