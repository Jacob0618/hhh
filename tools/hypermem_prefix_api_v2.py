"""Exact one-request original prefix per conversation, then unseen suffix only."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import threading
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, save_new, append, read, sha
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import READER_STAGE, JUDGE_STAGE

_LOCK = threading.RLock()
_LOCKS = {}


def lock(key):
    with _LOCK: return _LOCKS.setdefault(str(key), threading.RLock())


class PrefixAPI(BaselineAPI):
    admission = None

    def __init__(self, run_dir, *, context, cache_roots=(), **kwargs):
        a = type(self).admission
        path = Path(run_dir).absolute()
        if (not a or a.get('executing') is not True or cache_roots or type(kwargs.get('retries',2)) is not int or kwargs.get('retries', 2) != 2
                or path.resolve() != path or path.parent != Path(a['new_run']) / 'work'
                or path.name not in ('api_method', 'api_firstpass', 'api_recovery')
                or context.get('method') != 'hypermem' or type(context.get('conversation')) is not int
                or context['conversation'] not in range(10)):
            raise BaselineFatal('Unadmitted HyperMem namespace/context; no credential access')
        ci = context['conversation']
        if path.name == 'api_method':
            expected = {'method':'hypermem','conversation':ci,'phase':'build'}
            if context.get('phase') == 'retrieval':
                expected.update(phase='retrieval',query_id=context.get('query_id'))
                if expected['query_id'] not in a['questions'] or a['questions'][expected['query_id']] != ci:
                    raise BaselineFatal('Undeclared retrieval question')
            if context != expected: raise BaselineFatal('Method context differs')
        elif context != {'method':'hypermem','conversation':ci,'query_id':context.get('query_id'),'phase':path.name[4:]} or a['questions'].get(context.get('query_id')) != ci:
            raise BaselineFatal('Undeclared QA context')
        self.prefix_root = Path(a['new_run']) / 'work/prefix'
        self.case = a['rows'][ci]
        if self.case['conversation'] != ci: raise BaselineFatal('Prefix case ordering differs')
        super().__init__(run_dir, context=deepcopy(context), cache_roots=(), **kwargs)

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        a = type(self).admission
        signature = {'provider':provider,'model':model,'operation':operation,'body':body}
        key = digest(signature)
        ci = self.context['conversation']; case = self.case
        if self.run_dir.name == 'api_method':
            stages={'hypermem_boundary','hypermem_episode','hypermem_topic_create','hypermem_topic_update',
                    'hypermem_episode_roles','hypermem_topic_match','hypermem_fact_extract','hypermem_fact_roles'}
            if ((operation=='generateContent' and (stage not in stages or self.context['phase']!='build'))
                    or operation not in ('generateContent','batchEmbedContents','countTokens')
                    or (operation=='batchEmbedContents' and stage!='embedding')
                    or (operation=='countTokens' and stage!='embedding_length_check')):
                raise BaselineFatal('Unknown HyperMem method suffix stage')
        elif operation!='generateContent' or stage not in (READER_STAGE,JUDGE_STAGE):
            raise BaselineFatal('Unknown QA suffix stage')
        start_path = self.prefix_root / f'c{ci}.intent.json'
        done_path = self.prefix_root / f'c{ci}.complete.json'
        journal = self.prefix_root / f'c{ci}.jsonl'
        with lock(self.prefix_root / f'c{ci}'):
            if not cache: raise BaselineFatal('Continuation requires exact caching')
            origin = a['registry'].get(key)
            first = not done_path.exists()
            if first:
                if (self.run_dir.name != 'api_method' or self.context != case['context']
                        or signature != case['signature'] or stage != case['stage'] or key != case['request_hash']):
                    raise BaselineFatal('First full native request differs from admitted original prefix')
                save_new(start_path, {'conversation':ci,'request_hash':key,'signature':signature,
                                     'context':self.context,'stage':stage,'original_raw':case['raw']})
            else:
                expected_done = {'conversation':ci,'request_hash':case['request_hash'], 'original_raw':case['raw']}
                if read(done_path) != expected_done: raise BaselineFatal('Completed prefix proof changed')
            if origin:
                if (self.run_dir.name != 'api_method' or origin['context'] != self.context
                        or origin['stage'] != stage or origin['signature'] != signature):
                    raise BaselineFatal('Original key requested outside its exact admitted context')
                source = Path(origin['raw']['path']); h = origin['raw']['sha256']
                if sha(source) != h or read(source)['request'] != signature:
                    raise BaselineFatal('Original full raw changed; no redraw permitted')
                lineage_path = self.run_dir / 'cache_lineage' / (key+'.json')
                target = self.cache_dir / (key+'.json')
                lineage = {'source_path':str(source),'source_sha256':h,'target_path':str(target.resolve()),
                           'target_sha256':h,'request_hash':key}
                if (self.run_dir / 'request_intents' / (key+'.json')).exists():
                    raise BaselineFatal('Original key has a new producer intent')
                if lineage_path.exists():
                    if read(lineage_path) != lineage: raise BaselineFatal('Imported lineage differs')
                else: save_new(lineage_path, lineage)
            if self.run_dir.name == 'api_method':
                append(journal, {'event':'request_admitted','first_prefix':first,'request_hash':key,
                                 'context':deepcopy(self.context),'stage':stage,'operation':operation,
                                 'source':'original_exact_raw' if origin else 'new_namespace_request'})
            if first:
                value = super().request(provider, model, body, operation=operation, stage=stage, cache=True)
                save_new(done_path, {'conversation':ci,'request_hash':key,'original_raw':case['raw']})
                return value
        # The old prefix is complete. Do not serialize independent query HTTP
        # work: the frozen provider already owns per-request locks and logging.
        return super().request(provider, model, body, operation=operation, stage=stage, cache=True)


def admitted_api(proof, new_run):
    """Class-local authority for one fresh process; original module stays unchanged."""
    a = {'executing':True,'new_run':str(Path(new_run).resolve()),'rows':deepcopy(proof['rows']),
         'registry':deepcopy(proof['registry']),
         'questions':{q['query_id']:q['conversation'] for q in proof['questions']}}
    return type('AdmittedHyperMemAPI', (PrefixAPI,), {'admission':a})
