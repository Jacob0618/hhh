"""Ordered retained-prefix transport and separately authorized ID representation.

This class has no authority until a reviewed runner supplies a validated proof.
Original raw responses are imported exactly. A single, explicitly enabled and
qualified malformed-JSON replacement uses a durable before-send consumption
marker. No old producer is invented in this namespace.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import MappingProxyType

from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, append, read, save_new, sha
from tools import baselines_hypermem_native_v3 as adapter
from tools import archive_simplemem_native_v2 as safety
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body, response_text, READER_STAGE, JUDGE_STAGE, check_request

STAGES = frozenset(adapter._runtime._GENERATION_STAGES.values())
PROOF_SCHEMA = 'hypermem-native-v3-terminal-admission-v1'


def require(ok, message):
    if not ok:
        raise BaselineFatal(message)


def binding(path):
    path = safety.safe(path)
    return {'path':str(path), 'sha256':sha(path)}


def check_endpoint(context, owner, provider, model, body, operation, stage):
    require(provider == 'gemini', 'Only the frozen provider is admitted')
    if owner == 'api_method':
        if operation == 'generateContent':
            require(context['phase'] == 'build' and stage in STAGES and model == 'gemini-2.5-flash'
                and body.get('generationConfig') == {'maxOutputTokens':16384, 'responseMimeType':'application/json', 'thinkingConfig':{'thinkingBudget':0}},
                'Unknown method generation stage or configuration')
        else:
            require(model == 'gemini-embedding-001' and (operation,stage) in
                (('batchEmbedContents','embedding'),('countTokens','embedding_length_check')), 'Unknown embedding endpoint/stage')
    else:
        require(operation == 'generateContent' and stage in (READER_STAGE,JUDGE_STAGE), 'Unknown QA stage')
        check_request(provider, model, body, operation, stage)


class PrefixAPI(BaselineAPI):
    admission = None

    def __init__(self, run_dir, *, context, cache_roots=(), **kwargs):
        a = type(self).admission
        path = Path(run_dir).absolute()
        require(a and a.get('executing') is True and not cache_roots
            and type(kwargs.get('retries',2)) is int and kwargs.get('retries',2) == 2,
            'Missing exact continuation authority; credentials are not accessed')
        require(safety.safe(path) == path and path.parent == a['new_run']/'work'
            and path.name in ('api_method','api_firstpass','api_recovery'), 'Unadmitted continuation API path')
        ci = context.get('conversation')
        require(type(ci) is int and ci in range(10) and context.get('method') == 'hypermem', 'Unadmitted method/conversation')
        if path.name == 'api_method' and context.get('phase') == 'build':
            expected = {'method':'hypermem','conversation':ci,'phase':'build'}
        else:
            phase = 'retrieval' if path.name == 'api_method' else path.name[4:]
            qid = context.get('query_id')
            require(a['questions'].get(qid) == ci, 'Question is outside the fixed cohort')
            expected = {'method':'hypermem','conversation':ci,'phase':phase,'query_id':qid}
        require(context == expected, 'Context contains undeclared values')
        self.ci = ci
        self.case = a['conversations'][ci]
        self.prefix_root = a['new_run']/'work/prefix_v3'/f'c{ci}'
        super().__init__(run_dir,context=deepcopy(context),cache_roots=(),**kwargs)

    def _check_known_raw(self, key, signature, stage):
        origin = type(self).admission['registry'][key]
        require(origin['signature'] == signature and stage in origin['stages'], 'Known signature/stage differs')
        source = origin['raw']
        require(binding(source['path']) == source and read(source['path'])['request'] == signature,
            'Original raw changed; no redraw is permitted')
        return origin

    def _inherit(self, key, signature, stage):
        a = type(self).admission
        with a['registry_lock']:
            key_lock = a['key_locks'].setdefault(key,threading.RLock())
        with key_lock:
            return self._inherit_locked(key,signature,stage)

    def _inherit_locked(self, key, signature, stage):
        origin = self._check_known_raw(key,signature,stage)
        target = self.cache_dir/(key+'.json')
        require(not (self.run_dir/'request_intents'/(key+'.json')).exists(), 'Old key has an unexpected new producer')
        lineage = {'source_path':origin['raw']['path'],'source_sha256':origin['raw']['sha256'],
            'target_path':str(target.resolve()),'target_sha256':origin['raw']['sha256'],'request_hash':key}
        path = self.run_dir/'cache_lineage'/(key+'.json')
        if path.exists():
            require(read(path) == lineage, 'Original import lineage changed')
        else:
            save_new(path,lineage)
        # The frozen cache transport verifies/copies this exact lineage before
        # returning it and emits the actual inherited/cache consumer event.
        return origin

    def _replacement(self, entry, signature, stage):
        a = type(self).admission; cert = a['replacement']
        require(a['replacement_enabled'] is True and cert is not None
            and cert['kind'] == 'known_complete_invalid_episode_json'
            and type(cert['retry_limit']) is int and cert['retry_limit'] == 1
            and self.ci == cert['conversation'] == 5
            and cert['request_hash'] == entry['request_hash']
            and cert['signature'] == signature and cert['context'] == self.context
            and stage == cert['stage'] == 'hypermem_episode'
            and entry['ordinal'] == cert['prefix_ordinal']
            and entry['raw'] == cert['raw'], 'Unqualified original JSON replacement')
        original = self._check_known_raw(entry['request_hash'],signature,stage)
        require(original['raw'] == cert['raw'], 'Replacement source registry differs')
        response = read(cert['raw']['path'])['response']
        try:
            json.loads(response_text(response))
        except json.JSONDecodeError:
            pass
        else:
            raise BaselineFatal('Qualified replacement raw is not the same malformed JSON outcome')
        marker = a['new_run']/'work/replacement_v3/known_invalid_episode_once.json'
        value = {'schema':'hypermem-v3-exclusive-json-replacement-v1',
            'manifest':a['manifest'], 'terminal_proof':a['proof_binding'],
            'certificate':deepcopy(cert), 'context':deepcopy(self.context),
            'signature':deepcopy(signature), 'stage':stage,
            'request_hash':entry['request_hash'], 'prefix_ordinal':entry['ordinal'],
            'permission_consumed_before_transport':True, 'logical_replacements':1}
        # Always exclusive: failure after this marker never grants another send.
        save_new(marker,value)
        return binding(marker)

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        a = type(self).admission
        require(cache is True, 'Uncached continuation requests are forbidden')
        check_endpoint(self.context,self.run_dir.name,provider,model,body,operation,stage)
        signature = {'provider':provider,'model':model,'operation':operation,'body':body}
        key = digest(signature)
        ci = self.ci; prefix = self.case['prefix']; journal = self.prefix_root/'requests.jsonl'
        pending = None
        # Each conversation owns its ordered prefix. There is deliberately no
        # cross-conversation barrier, including for c5's replacement boundary.
        with a['locks'][ci]:
            ordinal = a['progress'][ci] + 1
            in_prefix = ordinal <= len(prefix)
            source = None; replacement_marker = None
            if in_prefix:
                require(a['inflight'][ci] is None, 'A prefix request is already in flight or unresolved')
                e = prefix[ordinal-1]
                require(self.run_dir.name == 'api_method' and self.context['phase'] == 'build'
                    and e['ordinal'] == ordinal and e['context'] == self.context
                    and e['signature'] == signature and e['stage'] == stage
                    and e['request_hash'] == key, 'Ordered original full prefix differs')
                start = {'schema':'hypermem-v3-prefix-slot-intent-v1','ordinal':ordinal,
                    'conversation':ci,'signature':deepcopy(signature),'context':deepcopy(self.context),
                    'stage':stage,'request_hash':key,'original_raw':e['raw'],
                    'mode':e['mode'],'original_native_call':e['native_call']}
                save_new(self.prefix_root/f'{ordinal:06d}.intent.json',start)
                if e['mode'] == 'inherit':
                    require(a['registry'][key]['raw'] == e['raw'], 'Prefix and all-seen registry raw differ')
                    self._inherit(key,signature,stage); source = 'qualified_original_exact_raw'
                elif e['mode'] == 'replace_once':
                    require(ordinal == len(prefix), 'Replacement must be the certified last failed prefix request')
                    replacement_marker = self._replacement(e,signature,stage)
                    source = 'qualified_original_invalid_json_replacement'
                else:
                    raise BaselineFatal('Unknown prefix permission')
                append(journal,{'event':'request_admitted','scope':'original_prefix','ordinal':ordinal,
                    'request_hash':key,'context':deepcopy(self.context),'stage':stage,'operation':operation,
                    'source':source,'replacement_marker':replacement_marker})
                a['inflight'][ci] = ordinal
                pending = (ordinal,e)
            else:
                require(read(self.prefix_root/'complete.json') == {'conversation':ci,'prefix_count':len(prefix),
                    'prefix_sha256':a['prefix_digests'][ci],'terminal_proof':a['proof_binding']}, 'Conversation prefix completion changed')
                if key in a['registry']:
                    require(self.run_dir.name == 'api_method', 'QA cannot consume method history')
                    cert = a['replacement']
                    if cert and key == cert['request_hash']:
                        require((self.run_dir/'request_intents'/(key+'.json')).exists()
                            and (self.cache_dir/(key+'.json')).exists()
                            and not (self.run_dir/'cache_lineage'/(key+'.json')).exists(),
                            'Replacement cannot be redrawn or fall back to original bad raw')
                        # Only its successful parsed new cache can be reused.
                        try: json.loads(response_text(read(self.cache_dir/(key+'.json'))['response']))
                        except Exception as exc: raise BaselineFatal('Existing replacement is not a valid parsed response') from exc
                        source = 'existing_replacement_cache'
                    else:
                        self._inherit(key,signature,stage); source = 'known_parent_exact_raw'
                else:
                    source = 'new_namespace_request'
                append(journal,{'event':'request_admitted','scope':'suffix','ordinal':None,
                    'request_hash':key,'context':deepcopy(self.context),'stage':stage,'operation':operation,
                    'source':source,'replacement_marker':None})
        if pending is not None:
            # Even the one c5 HTTP replacement is outside the per-conversation
            # lock. The durable exclusive slot and in-flight state forbid a
            # parallel request from advancing or re-sending this prefix.
            response = super().request(provider,model,body,operation=operation,stage=stage,cache=True)
            ordinal,e = pending
            with a['locks'][ci]:
                require(a['inflight'][ci] == ordinal and a['progress'][ci] == ordinal-1, 'Prefix completion order changed')
                complete = {'schema':'hypermem-v3-prefix-slot-complete-v1','ordinal':ordinal,
                    'conversation':ci,'request_hash':key,'mode':e['mode'],
                    'original_raw':e['raw'],'current_raw':binding(self.cache_dir/(key+'.json'))}
                if e['mode'] == 'inherit':
                    require(complete['current_raw']['sha256'] == e['raw']['sha256'], 'Inherited raw bytes changed')
                save_new(self.prefix_root/f'{ordinal:06d}.complete.json',complete)
                a['progress'][ci] = ordinal
                a['inflight'][ci] = None
                if ordinal == len(prefix):
                    save_new(self.prefix_root/'complete.json',{'conversation':ci,'prefix_count':len(prefix),
                        'prefix_sha256':a['prefix_digests'][ci],'terminal_proof':a['proof_binding']})
            return response
        # Prefix is complete for this conversation. Different requests may wait
        # in HTTP concurrently; frozen provider locks preserve same-key safety.
        return super().request(provider,model,body,operation=operation,stage=stage,cache=True)

    def authorize_episode_representation(self, evidence, *, original, derived, cached_native=False):
        a = type(self).admission
        require(self.run_dir.name == 'api_method' and self.context ==
            {'method':'hypermem','conversation':self.ci,'phase':'build'}, 'Representation outside build context')
        r = evidence['native_request']
        require(r == {'prompt':r['prompt'],'response_format':{'type':'json_object'},'stage':'hypermem_episode_roles',
            'json_output':True,'max_output':16384,'system':None}, 'Representation native request differs')
        signature = {'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
            'body':generation_body(r['prompt'],system=None,json_output=True,max_output=16384)}
        key = digest(signature); raw_path = self.cache_dir/(key+'.json')
        raw = read(raw_path)
        require(raw['request'] == signature and raw['stage'] == 'hypermem_episode_roles'
            and json.loads(response_text(raw['response'])) == original, 'Representation is not the actual retained HTTP response')
        path = safety.safe(evidence['native_call_path'])
        require(path.parent == a['new_run']/f'work/native/c{self.ci}/calls', 'Wrong native representation owner')
        expected, journal = adapter.represent_episode_ids(original,evidence['simple_to_real'])
        require(journal is not None and expected == derived
            and all(evidence.get(k) == v for k,v in journal.items()), 'Representation derivation changed')
        consumer_events = self.events
        if cached_native:
            require(a['progress'][self.ci] == len(self.case['prefix']) and a['inflight'][self.ci] is None,
                'Native cache cannot establish or resume prefix admission')
            journal_path = path.parent.parent/'representations'/path.name
            adapter._runtime._verify_seal(path)
            adapter._runtime._verify_seal(journal_path)
            retained = read(journal_path)
            require({k:v for k,v in retained.items() if k!='authorization'} == evidence,
                'Cached representation evidence differs')
            from tools.audit_simplemem_terminal_v2 import jsonl
            consumer_events = jsonl(self.run_dir/'logical_events.jsonl')
        require(any(e['request_hash'] == key and e['stage'] == 'hypermem_episode_roles'
            and e['context'] == self.context and e['cache_path'] == str(raw_path)
            and e['cache_sha256'] == sha(raw_path) and e['error_type'] is None for e in consumer_events),
            'Representation has no exact successful consumer in this API instance')
        qualified = a['role_cases'].get(key)
        if qualified:
            require(qualified['conversation'] == self.ci and qualified['raw']['sha256'] == sha(raw_path)
                and {x['simple_id']:x['event_id'] for x in qualified['mapping_from_complete_unique_native_display']} == evidence['simple_to_real']
                and qualified['original_parsed_sha256'] == journal['original_parsed_sha256']
                and qualified['derived_parsed_sha256'] == journal['derived_parsed_sha256'], 'Old role-ID qualification differs')
            require(binding(qualified['raw']['path']) == qualified['raw'] and binding(qualified['native_call']['path']) == qualified['native_call'], 'Qualified role source changed')
            scope = 'qualified_original_response'
            if Path(qualified['native_call']['path']).name != path.name:
                require(a['progress'][self.ci] == len(self.case['prefix']) and a['inflight'][self.ci] is None,
                    'A qualified response may use another native ordinal only after its complete original prefix')
                scope = 'qualified_original_response_reused_suffix'
        else:
            require(a['prospective_role_representation'] is True and key not in a['registry']
                and a['progress'][self.ci] == len(self.case['prefix'])
                and (self.run_dir/'request_intents'/(key+'.json')).exists()
                and not (self.run_dir/'cache_lineage'/(key+'.json')).exists(), 'Future ID representation is not explicitly admitted')
            scope = 'first_seen_suffix_representation'
        return {'schema':'hypermem-v3-role-representation-authorization-v1','status':'authorized','scope':scope,
            'manifest':a['manifest'],'terminal_proof':a['proof_binding'],'role_qualification':a['role_qualification'],
            'request_hash':key,'signature_sha256':digest(signature),'context':deepcopy(self.context),
            'stage':'hypermem_episode_roles','raw':binding(raw_path),
            'native_request_sha256':adapter._runtime._sha(r),'evidence_sha256':adapter._runtime._sha(evidence),
            'original_parsed_sha256':journal['original_parsed_sha256'],'derived_parsed_sha256':journal['derived_parsed_sha256'],
            'model_regeneration_permitted':False}


def admitted_api(proof, new_run, *, manifest, proof_binding,
                 replacement_enabled=False, prospective_role_representation=False):
    require(proof.get('schema') == PROOF_SCHEMA and proof.get('status') == 'passed'
        and type(replacement_enabled) is bool and type(prospective_role_representation) is bool,
        'Only a complete, explicitly reviewed terminal proof can construct an API')
    # One admitted proof is loaded once. Shared read-only indexes hold its
    # immutable source values; only cursor/lock state is mutable per class.
    conversations = tuple(MappingProxyType({**c,'prefix':tuple(c['prefix'])}) for c in proof['conversations'])
    require(len(conversations) == 10 and [c['conversation'] for c in conversations] == list(range(10))
        and all(c['prefix'] for c in conversations), 'Whole ten-conversation prefix required')
    authority = {'executing':True, 'new_run':Path(new_run).absolute(), 'manifest':deepcopy(manifest),
        'proof_binding':deepcopy(proof_binding),'conversations':conversations,
        'registry':MappingProxyType(proof['producer_registry']),
        'role_cases':MappingProxyType({c['request_hash']:c for c in proof['role_cases']}),
        'role_qualification':deepcopy(proof['role_qualification']),
        'replacement':deepcopy(proof.get('replacement')), 'replacement_enabled':replacement_enabled,
        'prospective_role_representation':prospective_role_representation,
        'questions':{q['query_id']:q['conversation'] for q in proof['questions']},
        'prefix_digests':{ci:digest(c['prefix']) for ci,c in enumerate(conversations)},
        'progress':{ci:0 for ci in range(10)}, 'inflight':{ci:None for ci in range(10)},
        'locks':{ci:threading.RLock() for ci in range(10)},'registry_lock':threading.RLock(),'key_locks':{}}
    return type('AdmittedHyperMemV3API',(PrefixAPI,),{'admission':authority})
