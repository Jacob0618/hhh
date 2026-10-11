"""Exact ordered original request replay and case-certified native compatibility.

This module makes no request on import. PrefixPolicy is also the offline audit
state machine; its decisions do not consult credentials or a database.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, append, read, sha, save_new
from jev_mem.sdk20_cached_api import digest, _LOCK, _HASH_LOCKS, _publish_bytes
from jev_mem.gemini25_ablation_provider_v1 import response_text
import threading

KINDS={'native_keyword_syntax_empty','native_keyword_field_empty','native_analysis_cached_default','native_empty_missing_queries'}
STAGES={'simplemem_requirements','simplemem_query_plan','simplemem_query_analysis','simplemem_completeness','simplemem_missing_queries'}

def text_sha(text):return hashlib.sha256(text.encode('utf8')).hexdigest()

class PrefixPolicy:
    def __init__(self,case,registry):
        self.case=deepcopy(case);self.registry=registry;self.cursor=0;self.requests=[];self.compatibility=[];self.last_generation=None
        self.context={'method':'simplemem','conversation':case['conversation'],'query_id':case['query_id'],'phase':'retrieval'}
        self.cert=case['certificate'];self.prefix=case['prefix'];self.analysis_attempts=0
        if (self.cert.get('kind') not in KINDS or not self.prefix or self.cert.get('request_hash')!=self.prefix[-1]['request_hash']
                or self.cert.get('retry_limit')!=0 or self.cert.get('paid_regeneration_permitted') is not False):
            raise BaselineFatal('Unknown exact native failure certificate')
        for i,p in enumerate(self.prefix):
            r=registry.get(p['request_hash'])
            if (p['ordinal']!=i or p['context']!=self.context or digest(p['signature'])!=p['request_hash'] or r is None
                    or p['signature']!=r['signature'] or p['raw']!=r['raw']):raise BaselineFatal('Incomplete or changed original prefix')
    @property
    def ready(self):
        return self.cursor==len(self.prefix) and bool(self.compatibility) and (self.cert['kind']!='native_analysis_cached_default' or self.analysis_attempts>=3 and self.analysis_attempts%3==0)
    def before(self,signature,stage):
        op=signature['operation'];key=digest(signature)
        if (op=='generateContent' and stage not in STAGES) or (op!='generateContent' and (op not in ('countTokens','batchEmbedContents') or stage not in ('embedding','embedding_length_check'))):
            raise BaselineFatal('Writer/undeclared method stage forbidden')
        if op=='batchEmbedContents' and any(r.get('taskType')!='RETRIEVAL_QUERY' for r in signature['body']['requests']):
            raise BaselineFatal('Corpus embedding forbidden')
        if self.cursor<len(self.prefix):
            p=self.prefix[self.cursor]
            if (signature!=p['signature'] or stage!=p['stage'] or p['context']!=self.context):raise BaselineFatal('Original native prefix order/signature/context differs')
            mode='original_prefix';self.cursor+=1
        elif not self.ready:
            if not (self.cert['kind']=='native_analysis_cached_default' and key==self.cert['request_hash'] and stage==self.cert['stage']
                    and self.analysis_attempts%3 in (1,2)):
                raise BaselineFatal('Unseen or reordered request before certified native boundary completed')
            mode='native_cached_analysis_retry'
        else:mode='seen_exact_cache' if key in self.registry else 'first_unseen_request'
        if key in self.registry and signature!=self.registry[key]['signature']:raise BaselineFatal('Seen signature collision')
        if key in self.registry and stage not in {e['stage'] for e in self.registry[key]['logical_consumers']}:
            raise BaselineFatal('Seen hash has no original same-stage consumer')
        decision={'ordinal':len(self.requests),'request_hash':key,'stage':stage,'context':deepcopy(self.context),'signature':deepcopy(signature),'mode':mode}
        self.requests.append(decision)
        if op=='generateContent':self.last_generation={'request_hash':key,'stage':stage,'raw':self.registry.get(key,{}).get('raw')}
        return decision
    def authorize_native_compatibility(self,kind,details):
        cert=self.cert;last=self.last_generation
        if self.cursor!=len(self.prefix) or last is None or any(last[k]!=cert[k] for k in ('request_hash','stage','raw')):
            raise BaselineFatal('Compatibility not at the exact admitted raw boundary')
        if kind=='keyword_search' and cert['kind'] in ('native_keyword_syntax_empty','native_keyword_field_empty'):
            expected=cert['exception']['message']
            prefix='SimpleMem retrieve: ValueError: '
            if not expected.startswith(prefix):raise BaselineFatal('Unexpected original keyword exception wrapper')
            if details!={'keywords':cert['keywords'],'exception':{'type':'ValueError','message':expected[len(prefix):]}}:
                raise BaselineFatal('Keyword error or original keywords differ from certificate')
        elif kind=='query_analysis_json' and cert['kind']=='native_analysis_cached_default':
            er=cert['json_error'];message=f"{er['message']}: line {er['line']} column {er['column']} (char {er['position']})"
            if (details['query']!=cert['analysis_query'] or details['response_text_sha256']!=cert['response_text_sha256']
                    or details['attempt']!=self.analysis_attempts%3 or details['exception']!={'type':'JSONDecodeError','message':message}):
                raise BaselineFatal('Original cached analysis retry differs')
            self.analysis_attempts+=1
        elif kind=='empty_missing_queries' and cert['kind']=='native_empty_missing_queries':
            if details!={'stage':cert['stage'],'response_text_sha256':cert['response_text_sha256'],'targeted_queries':[]}:
                raise BaselineFatal('Empty native missing-info case differs')
        else:raise BaselineFatal('Unqualified native fallback')
        self.compatibility.append({'ordinal':len(self.compatibility),'after_request_ordinal':len(self.requests)-1,'kind':kind,'details':deepcopy(details)})
        return True
    def report(self):
        if not self.ready:raise BaselineFatal('Original prefix or native compatibility point incomplete')
        return {'query_id':self.case['query_id'],'original_prefix_consumed':self.cursor,'requests':self.requests,'compatibility':self.compatibility}

class ContinuationAPI(BaselineAPI):
    def __init__(self,run_dir,*,admission,case,registry,context,cache_roots=(),**kwargs):
        expected=admission['run_dir'];real=Path(run_dir).absolute()
        if (str(real)!=expected or real.resolve()!=real or context!={'method':'simplemem','conversation':case['conversation'],'query_id':case['query_id'],'phase':'retrieval'}
                or case['query_id'] not in admission['query_ids'] or cache_roots or kwargs.get('retries',2)!=2
                or admission.get('executing') is not True or admission.get('case_sha256',{}).get(case['query_id'])!=digest(case)
                or admission.get('registry_sha256')!=digest(registry)):
            raise BaselineFatal('Continuation API requires admitted exact namespace, case, registry and query')
        self.policy=PrefixPolicy(case,registry);self.registry=registry
        self.journal=real.parent/'prefix'/ (case['query_id']+'.jsonl')
        super().__init__(run_dir,context=deepcopy(context),cache_roots=(),**kwargs)
    def scoped(self,**context):
        if {**self.context,**context}!=self.context:raise BaselineFatal('Native query context cannot change')
        return self
    def authorize_native_compatibility(self,kind,details):
        result=self.policy.authorize_native_compatibility(kind,details)
        append(self.journal,{'event':'native_compatibility',**self.policy.compatibility[-1]})
        return result
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        if not cache:raise BaselineFatal('Uncached continuation forbidden')
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        decision=self.policy.before(signature,stage);key=decision['request_hash']
        with _LOCK:lock=_HASH_LOCKS.setdefault((str(self.run_dir.resolve()),key),threading.RLock())
        with lock:
            origin=self.registry.get(key)
            if origin:
                source=Path(origin['raw']['path']);expected=origin['raw']['sha256']
                if not source.is_file() or sha(source)!=expected or read(source).get('request')!=signature:
                    raise BaselineFatal('Seen original raw is absent/changed; no redraw allowed')
                target=self.cache_dir/(key+'.json');lineage=self.run_dir/'cache_lineage'/(key+'.json')
                value={'source_path':str(source),'source_sha256':expected,'target_path':str(target.resolve()),'target_sha256':expected,'request_hash':key}
                if (self.run_dir/'request_intents'/(key+'.json')).exists():raise BaselineFatal('Seen original hash has a new physical intent')
                if not lineage.exists():save_new(lineage,value)
                elif read(lineage)!=value:raise BaselineFatal('Original cache lineage differs')
                if not target.exists():_publish_bytes(target,source.read_bytes())
                elif sha(target)!=expected:raise BaselineFatal('Imported raw differs')
            append(self.journal,{'event':'request_admission',**decision})
            return super().request(provider,model,body,operation=operation,stage=stage,cache=True)
