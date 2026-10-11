"""R07 native persistent supplements and SDK20 retrieval, in new namespaces only.

Import, prepare, audit, seal and validation are offline. Only the explicit
materialize command constructs an embedding-only paced client. Generation,
admission, QA and recovery are owned by other versioned tools. Native algorithms
are delegated unchanged; the entity-link observer records their actual targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_sdk20_modules import digest, now, Warnings
from tools.check_experiment_health import shared_read
from tools import run_gemini38_writer_v1 as writer
from tools import run_session_experiments as native
from tools.run_planned_retrieval_v1 import CacheOnlyEmbeddings
from jev_mem import qa_embedding_cache_pack_v1 as packs
from jev_mem.paced_embedding_api import PacedEmbeddingAPI, DEFAULT_GATE
from jev_mem.providers import Embeddings
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest as request_hash
from jev_mem.local_bm25_context_v2 import (local_bm25_context, validate_resources,
                                         compare_persisted_fact_vectors)
from jev_mem.sdk20_type_adapter import capture_native_pool, CAPTURE_SCHEMA
from tools import mem0_regroup_firstpass_review_v2 as idx

SCHEMA = 'write-omission-r07-native-materialization-v1'
SOURCE_SCHEMA = 'write-omission-r07-selected-inputs-v1'
ARMS = ('baseline', 'cheap', 'jev')
NEW_ARMS = ('cheap', 'jev')
SPEC_SHA = 'bcd1bfec608e23053b791c88064a35b249936351da0db8e4dbbd34a6250a36db'
CODE = tuple(dict.fromkeys((*writer.CODE, 'tools/materialize_write_omission_r07.py',
    'tools/run_planned_retrieval_v1.py', 'tools/mem0_regroup_firstpass_review_v2.py',
    'jev_mem/sdk20_type_adapter.py', 'jev_mem/qa_embedding_cache_pack_v1.py',
    'tools/check_experiment_health.py','upstream/mem0/mem0/utils/entity_extraction.py',
    'upstream/mem0/mem0/utils/scoring.py')))


def native_sdk_identity():
    import mem0
    base=Path(mem0.__file__).resolve().parent
    result={}
    for rel in ('memory/main.py','memory/storage.py','vector_stores/qdrant.py',
                'utils/entity_extraction.py','utils/scoring.py'):
        installed,upstream=base/rel,ROOT/'upstream/mem0/mem0'/rel
        require(sha(installed)==sha(upstream),'Actual installed native SDK differs from frozen upstream')
        result[str(installed)]=sha(installed)
    return result


class NativeGateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise NativeGateError(message)


def safe_error(ex,phase,candidate_id=None):
    return {'type':type(ex).__name__,'phase':phase,'candidate_id':candidate_id,
        'message':str(ex) if isinstance(ex,NativeGateError) else
            'Native stage failed; original intent/store/raw retained',
        'frames':[{'file':str(Path(f.filename).resolve()),'line':f.lineno,'function':f.name}
                  for f in traceback.extract_tb(ex.__traceback__)],
        'unknown_or_failed_no_automatic_reentry':True}


def sha(path):
    return hashlib.sha256(shared_read(Path(path).resolve())).hexdigest()


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(shared_read(Path(path).resolve()).decode('utf-8-sig'))


def bound(ref, sealed=False):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact pointer required')
    require(pointer(ref['path']) == ref, 'Bound artifact bytes differ')
    value = read(ref['path'])
    if sealed:
        require(value.get('content_sha256') == digest({k:v for k,v in value.items()
                if k != 'content_sha256'}), 'Content seal differs')
    return value


def save_new(path, value):
    """Durable exclusive publication; no old artifact is ever overwritten."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**value, 'content_sha256': digest(value)}
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return pointer(path)


def sealed_read(path):
    return bound(pointer(path), sealed=True)


def inventory(native_dir):
    """Persisted Qdrant and history only, with no global mutable hash registry."""
    root = Path(native_dir).resolve()
    require(root.is_dir(), 'Native directory missing')
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Native symlink refused')
        if path.is_file() and path.name.endswith(('-wal', '-journal')):
            require(path.stat().st_size == 0, 'Uncheckpointed native sidecar')
    for path in sorted([*(root/'qdrant').rglob('*'), root/'history.sqlite']):
        if path.is_file() and path.name != '.lock':
            result[path.relative_to(root).as_posix()] = sha(path)
    require('history.sqlite' in result and any(k.endswith('facts/storage.sqlite')
            for k in result), 'Native fact/history store missing')
    return result


def copy_closed(source, dest, expected):
    """Hold the original real Qdrant OS lock throughout the byte copy."""
    import portalocker
    source, dest = Path(source).resolve(), Path(dest).resolve()
    require(not dest.exists() and not dest.is_relative_to(source)
            and not source.is_relative_to(dest), 'Fresh disjoint native target required')
    require(inventory(source) == expected, 'Source native inventory differs')
    lock = source/'qdrant/.lock'
    require(lock.is_file() and not lock.is_symlink(), 'Existing native OS lock missing')
    with lock.open('rb') as handle:
        portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        try:
            require(inventory(source) == expected, 'Source changed before copy')
            for rel, expected_sha in expected.items():
                src, dst = (source/rel).resolve(), (dest/rel).resolve()
                require(src.is_relative_to(source) and dst.is_relative_to(dest),
                        'Native copy path escapes namespace')
                dst.parent.mkdir(parents=True, exist_ok=True)
                with src.open('rb') as h, dst.open('xb') as out:
                    shutil.copyfileobj(h, out)
                    out.flush(); os.fsync(out.fileno())
                require(sha(dst) == expected_sha, 'Copied native bytes differ')
            require(inventory(source) == expected and inventory(dest) == expected,
                    'Native inventory changed during copy')
        finally:
            portalocker.unlock(handle)


def primitive(value):
    if hasattr(value, 'model_dump'):
        return primitive(value.model_dump())
    if isinstance(value, dict):
        return {str(k): primitive(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(v) for v in value]
    return value


def sqlite_rows(path, table):
    require(table in ('points', 'history'), 'Unexpected native table')
    before = sha(path)
    con = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute('PRAGMA query_only=ON'); con.execute('BEGIN')
        require(con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok',
                'Native SQLite integrity failure')
        rows = [dict(r) for r in con.execute('SELECT * FROM '+table)]
        con.rollback()
    finally:
        con.close()
    require(sha(path) == before, 'Native SQLite changed during readonly audit')
    return rows


def snapshot(native_dir):
    native_dir = Path(native_dir)
    before = inventory(native_dir)
    result = {}
    for key, collection in (('facts', 'facts'), ('entities', 'facts_entities')):
        db = native_dir/f'qdrant/collection/{collection}/storage.sqlite'
        points = [pickle.loads(r['point']) for r in sqlite_rows(db, 'points')]
        result[key] = {str(p.id): {'payload': primitive(p.payload),
                                 'vector': primitive(p.vector)} for p in points}
        require(len(result[key]) == len(points), 'Duplicate native point ID')
    result['history'] = sorted(sqlite_rows(native_dir/'history.sqlite', 'history'), key=digest)
    require(inventory(native_dir) == before, 'Native changed during snapshot')
    return result


def dense(vector):
    return idx.dense(vector)


def sparse(vector, allow_empty=False):
    if not isinstance(vector, dict) or set(vector) != {'indices', 'values'}:
        return False
    from types import SimpleNamespace
    return idx.sparse(SimpleNamespace(**vector), allow_empty=allow_empty)


def fact_fields(point):
    p = point['payload']
    return {'id': p.get('id'), 'text': p.get('data'),
            'observation_date': p.get('jev_mem_observation_date')}


def validate_delta(before, after, additions, user_id):
    """Original records are an exact logical subset; only entity links may grow."""
    original_ids = set(before['facts'])
    new_by_id = {r['memory_id']: r for r in additions if r['status'] == 'added'}
    require(len(new_by_id) == sum(r['status'] == 'added' for r in additions),
            'Supplement UUID duplicated')
    require(set(after['facts']) == original_ids | set(new_by_id),
            'Missing or extra native facts')
    for mid in original_ids:
        require(after['facts'][mid] == before['facts'][mid], 'Original native fact changed')
    history_before = Counter(digest(h) for h in before['history'])
    history_after = Counter(digest(h) for h in after['history'])
    require(not history_before-history_after, 'Original native history changed')
    new_history = [h for h in after['history'] if h['memory_id'] not in original_ids]
    require(Counter(h['memory_id'] for h in new_history) == Counter(new_by_id.keys()),
            'Supplement history count differs')
    require(len(after['history']) == len(before['history']) + len(new_by_id),
            'Extra history event on original or supplement')
    for h in new_history:
        r = new_by_id[h['memory_id']]
        require(h['event'] == 'ADD' and not h['is_deleted'] and h['old_memory'] is None
                and h['new_memory'] == r['candidate']['statement'], 'Supplement ADD history differs')
    for mid, row in new_by_id.items():
        p, vec = after['facts'][mid]['payload'], after['facts'][mid]['vector']
        c = row['candidate']
        require(p['data'] == c['statement'] and p['user_id'] == user_id
                and p.get('jev_mem_session_id') == c['source_session_id']
                and p.get('jev_mem_observation_date') == c['observation_date']
                and p['hash'] == hashlib.md5(c['statement'].encode()).hexdigest(),
                'Supplement text/session/date/scope differs')
        require(dense(vec.get('')) and sparse(vec.get('bm25')), 'Supplement vectors invalid')
        require(p.get('created_at') and p.get('updated_at') == p['created_at'],
                'Supplement storage time/default metadata differs')
        for trace in row['entity_trace']:
            require(trace['memory_id'] == mid and trace.get('selected_entity_id') in after['entities'],
                    'Expected entity target missing')
            target = after['entities'][trace['selected_entity_id']]['payload']
            require(mid in target.get('linked_memory_ids', []) and target.get('user_id') == user_id,
                    'Explicit native entity link missing')
    for eid, old in before['entities'].items():
        require(eid in after['entities'], 'Original entity ID disappeared')
        new = after['entities'][eid]
        op, np = copy.deepcopy(old['payload']), copy.deepcopy(new['payload'])
        ol, nl = op.pop('linked_memory_ids'), np.pop('linked_memory_ids')
        require(op == np and old['vector'] == new['vector'] and set(ol) <= set(nl),
                'Original entity content/vector/link removed or changed')
    linked = set()
    for eid, point in after['entities'].items():
        p, v = point['payload'], point['vector']
        links = p.get('linked_memory_ids')
        require(isinstance(links, list) and links and len(links) == len(set(links))
                and set(links) <= set(after['facts']) and p.get('user_id') == user_id,
                'Entity links malformed or foreign')
        require(dense(v.get('')), 'Entity dense vector missing')
        # Entity sparse absence is an original native limitation, not a new
        # admission criterion. Facts require BM25; present entity sparse values
        # must be valid. Missing/empty entity sparse values are reported below.
        if v.get('bm25') is not None:
            require(sparse(v['bm25'], allow_empty=True), 'Entity BM25 vector malformed')
        linked.update(links)
    # Expected entities come from the unmodified native extractor, not a NER substitute.
    for row in additions:
        mid = row['memory_id']
        require(mid in after['facts'], 'Duplicate-skip existing fact missing')
        if row['status'] == 'duplicate_skipped':
            require(after['facts'][mid]['payload']['data'] == row['candidate']['statement'],
                    'Duplicate skip is not byte-identical')
        else:
            require(row['expected_entity_keys'] == [t['normalized_text'] for t in row['entity_trace']],
                    'Native swallowed entity operation or unexpected target')
            require(not row['expected_entity_keys'] or mid in linked, 'New orphan fact')
    return {'original_facts_unchanged': len(original_ids), 'added_facts': len(new_by_id),
            'duplicate_skips': sum(r['status'] == 'duplicate_skipped' for r in additions),
            'original_entity_ids_preserved': len(before['entities']),
            'entities': len(after['entities']), 'history_adds': len(new_history),
            'new_orphan_fact_ids': [], 'semantic_support_certified': False,
            'entity_sparse_missing': sorted(eid for eid,p in after['entities'].items()
                if p['vector'].get('bm25') is None),
            'entity_sparse_empty': sorted(eid for eid,p in after['entities'].items()
                if p['vector'].get('bm25') is not None and not p['vector']['bm25']['indices'])}


class EmbeddingOnlyAPI(PacedEmbeddingAPI):
    """The original paced durable embedding implementation, with no generation route."""
    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        require((provider, model, operation) == ('gemini', 'gemini-embedding-001', 'batchEmbedContents'),
                'Materialization permits only fixed ADD embeddings')
        require(stage == 'mem0_embedding_add' and isinstance(body, dict)
                and set(body) == {'requests'}, 'Unexpected materialization embedding stage/body')
        requests = body['requests']
        require(isinstance(requests, list) and 1 <= len(requests) <= 32, 'Embedding batch count differs')
        for r in requests:
            require(set(r) == {'model', 'content', 'taskType', 'outputDimensionality'}
                    and r['model'] == 'models/gemini-embedding-001'
                    and r['taskType'] == 'RETRIEVAL_DOCUMENT' and r['outputDimensionality'] == 768,
                    'Fixed ADD embedding signature differs')
        require(len(json.dumps({'provider':provider,'model':model,'operation':operation,'body':body},
                sort_keys=True, ensure_ascii=False).encode()) <= 131072, 'Embedding request byte cap')
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def expected_entities(memory, text):
    from mem0.memory.main import extract_entities
    keys = []
    for _, term in extract_entities(text):
        key = memory._normalize_entity_text(term)
        if key and key not in keys:
            keys.append(key)
    return keys


class OfflineAddReplay:
    """Reexecute the original embedding body/parser without a client or network path."""
    embed=ModelAPI.embed
    def __init__(self,response):
        self.response=response;self.signature=None
    def count_tokens(self,*args,**kwargs):
        raise ValueError('R07 bounded statement/entity unexpectedly requires countTokens')
    def request(self,provider,model,body,*,operation,stage):
        require(stage=='mem0_embedding_add','Original ADD embedding stage differs')
        require(self.signature is None,'Expected one bounded ADD embedding request')
        self.signature={'provider':provider,'model':model,'operation':operation,'body':body}
        return self.response


def add_signature(text):
    replay=OfflineAddReplay({'embeddings':[{'values':[1.]+[0.]*767}]})
    replay.embed([text],query=False,stage='mem0_embedding_add')
    return replay.signature


def embedding_ledger(api_dir,ci):
    """Physical attempts/cache consumers and local producer intents, never synthetic usage."""
    api_dir=Path(api_dir)
    path=api_dir/'api_calls.jsonl'
    if not path.exists():
        require(not list((api_dir/'request_intents').glob('*.json'))
                and not list((api_dir/'api_cache').glob('*.json')), 'Embedding intent without ledger')
        return []
    data=shared_read(path);rows=[json.loads(l) for l in data.decode('utf-8-sig').splitlines() if l.strip()]
    for row in rows:
        context=row.get('context',{})
        require(row.get('provider')=='gemini' and row.get('model')=='gemini-embedding-001'
                and row.get('operation')=='batchEmbedContents' and row.get('stage')=='mem0_embedding_add'
                and row.get('resolved_model') in (None,'gemini-embedding-001')
                and context.get('architecture')==SCHEMA and context.get('conversation')==ci
                and context.get('method') in NEW_ARMS and context.get('phase')=='write_supplement'
                and isinstance(context.get('candidate_id'),str) and isinstance(context.get('group_id'),str),
                'Embedding ledger identity/context/model differs')
        if row.get('cache_hit') is True:
            require('attempt' not in row and 'status' not in row and
                    row.get('usage_semantics')=='reused_response_no_new_model_usage',
                    'Malformed embedding cache consumer')
        else:
            require(type(row.get('attempt')) is int and 1<=row['attempt']<=3,
                    'Embedding physical attempt cap differs')
    require(shared_read(path)==data,'Embedding ledger changed during readonly audit')
    return rows


def audit_embedding(api_dir,text,context,ledger):
    signature=add_signature(text);key=request_hash(signature);api_dir=Path(api_dir)
    raw_ref=pointer(api_dir/f'api_cache/{key}.json');raw=bound(raw_ref)
    intent_ref=pointer(api_dir/f'request_intents/{key}.json');intent=bound(intent_ref)
    require(raw.get('request')==signature and raw.get('stage')=='mem0_embedding_add'
            and intent.get('signature')==signature and intent.get('request_hash')==key
            and intent.get('stage')=='mem0_embedding_add' and intent.get('retries')==2,
            'Original ADD raw/producer intent signature differs')
    require(not (api_dir/f'cache_lineage/{key}.json').exists(),
            'Undeclared external embedding cache import')
    producer=[r for r in ledger if r['request_hash']==key and r.get('context')==intent['context']
              and r.get('cache_hit') is not True]
    require(producer and [r['attempt'] for r in producer]==list(range(1,len(producer)+1))
            and producer[-1].get('status')==200
            and all(r.get('status') in ('transport_error',429,500,502,503,504,529)
                    for r in producer[:-1]), 'Embedding successful producer terminal/attempts missing')
    consumers=[r for r in ledger if r['request_hash']==key and r.get('context')==context]
    require(consumers and any(r.get('cache_hit') is True or r.get('status')==200 for r in consumers),
            'Embedding candidate consumer evidence missing')
    response=raw['response']
    require(response.get('modelVersion',response.get('model')) in (None,'gemini-embedding-001'),
            'Embedding raw explicitly returned a different model')
    replay=OfflineAddReplay(response);vector=replay.embed([text],query=False,stage='mem0_embedding_add')[0]
    require(replay.signature==signature,'Original embedding body/parser replay differs')
    return vector,{'request_hash':key,'raw':raw_ref,'producer_intent':intent_ref,
        'producer_context':intent['context'],'consumer_context':context,
        'resolved_model':response.get('modelVersion',response.get('model')),
        'resolved_model_available':response.get('modelVersion',response.get('model')) is not None}


def compare_embedding(stored,expected):
    require(dense(stored) and dense(expected),'Invalid persisted/replayed dense vector')
    error=max(abs(a-b) for a,b in zip(stored,expected))
    # Native Qdrant cosine storage normalizes float32; raw API norm uses float64.
    require(error<=2e-7,'Persisted dense vector differs from exact raw ADD response')
    return error


def compare_export_embedding(native_vector,export_vector):
    """Original effective-bank audit tolerance; the copied native vector stays exact."""
    require(dense(native_vector) and dense(export_vector), 'Original exported/native dense vector invalid')
    error=max(abs(a-b) for a,b in zip(native_vector,export_vector))
    require(error<=1e-6,'Original exported/native dense vector differs beyond original audit tolerance')
    return error


def audit_added_embeddings(api_dir,ci,arm,additions,native_snapshot):
    ledger=embedding_ledger(api_dir,ci);proofs=[]
    expected_occurrences=Counter()
    for row in additions:
        if row['status']!='added':continue
        c=row['candidate'];context={'architecture':SCHEMA,'conversation':ci,'method':arm,
            'phase':'write_supplement','candidate_id':c['candidate_id'],'group_id':c['source_session_id']}
        expected_occurrences[(request_hash(add_signature(c['statement'])),digest(context))]+=1
        v,proof=audit_embedding(api_dir,c['statement'],context,ledger)
        proof.update(memory_id=row['memory_id'],kind='fact',max_persistence_abs_error=
            compare_embedding(native_snapshot['facts'][row['memory_id']]['vector'][''],v))
        proofs.append(proof)
        for trace in row['entity_trace']:
            expected_occurrences[(request_hash(add_signature(trace['text'])),digest(context))]+=1
            v,proof=audit_embedding(api_dir,trace['text'],context,ledger)
            proof.update(memory_id=row['memory_id'],entity_id=trace['selected_entity_id'],
                         kind='entity_lookup_embedding',selection_kind=trace['selection_kind'])
            if trace['selection_kind']=='native_insert':
                proof['max_persistence_abs_error']=compare_embedding(
                    native_snapshot['entities'][trace['selected_entity_id']]['vector'][''],v)
                proof['kind']='entity_insert_embedding'
            # Exact/semantic dedup retains the existing canonical entity vector.
            proofs.append(proof)
    actual_occurrences=Counter((r['request_hash'],digest(r['context'])) for r in ledger
        if r['context']['method']==arm and (r.get('cache_hit') is True or r.get('status')==200))
    require(actual_occurrences==expected_occurrences,
            'Native embedding invocation occurrences differ from ADD/entity trace')
    return proofs


def audit_embedding_inventory(api_dir,ci,by_arm):
    """Close the whole conversation ledger: no orphan hash/context/physical producer."""
    api_dir=Path(api_dir);ledger=embedding_ledger(api_dir,ci)
    declared=Counter();texts={};contexts={}
    for arm in NEW_ARMS:
        for row in by_arm[arm]:
            if row['status']!='added':continue
            c=row['candidate'];context={'architecture':SCHEMA,'conversation':ci,'method':arm,
                'phase':'write_supplement','candidate_id':c['candidate_id'],'group_id':c['source_session_id']}
            cd=digest(context);contexts[cd]=context
            for text in [c['statement'],*[t['text'] for t in row['entity_trace']]]:
                key=request_hash(add_signature(text));texts[key]=text;declared[(key,cd)]+=1
    actual=Counter((r['request_hash'],digest(r['context'])) for r in ledger
        if r.get('cache_hit') is True or r.get('status')==200)
    require(actual==declared,'Embedding ledger contains orphan or missing declared invocation')
    require(all((r['request_hash'],digest(r['context'])) in declared for r in ledger),
            'Undeclared physical embedding context/hash')
    keys=set(texts)
    require({p.stem for p in (api_dir/'api_cache').glob('*.json')}==keys
            and {p.stem for p in (api_dir/'request_intents').glob('*.json')}==keys
            and not list((api_dir/'cache_lineage').glob('*.json')),
            'Orphan/missing embedding raw or unresolved producer intent')
    refs=[]
    for key,text in sorted(texts.items()):
        intent=read(api_dir/f'request_intents/{key}.json')
        physical=[r for r in ledger if r['request_hash']==key and r.get('cache_hit') is not True]
        require(physical and all(r['context']==intent['context'] for r in physical),
                'Multiple or undeclared same-hash embedding producers')
        context=contexts[digest(physical[0]['context'])]
        _,proof=audit_embedding(api_dir,text,context,ledger)
        producer_index=next(i for i,r in enumerate(ledger) if r in physical and r.get('status')==200)
        require(all(i>producer_index for i,r in enumerate(ledger)
                    if r['request_hash']==key and r.get('cache_hit') is True),
                'Embedding cache consumer preceded a successful producer')
        refs.append(proof)
    return {'conversation':ci,'declared_logical_invocations':sum(declared.values()),
        'unique_signatures':len(keys),'physical_attempts':sum(r.get('cache_hit') is not True for r in ledger),
        'cache_consumers':sum(r.get('cache_hit') is True for r in ledger),
        'known_usage_rows':sum(bool(r.get('usage')) for r in ledger if r.get('cache_hit') is not True),
        'usage_unavailable_not_zero':sum(not bool(r.get('usage')) for r in ledger if r.get('cache_hit') is not True),
        'ledger':pointer(api_dir/'api_calls.jsonl') if ledger else None,'raw_producer_bindings':refs}


@contextlib.contextmanager
def entity_observer(memory, expected_mid):
    """Observe delegated original upsert targets, including legal semantic merges."""
    traces = []
    original = memory._upsert_entity
    store = memory.entity_store
    old = {k: getattr(store, k) for k in ('search', 'insert', 'update')}
    lookup = memory._existing_entities_by_text
    active = None

    def existing(filters):
        rows = lookup(filters)
        if active is not None:
            match = rows.get(active['normalized_text'])
            if match is not None:
                active['selected_entity_id'] = str(match.id)
                active['selection_kind'] = 'native_exact'
        return rows

    def search(*args, **kwargs):
        rows = old['search'](*args, **kwargs)
        if active is not None and rows and rows[0].score >= .95:
            active['selected_entity_id'] = str(rows[0].id)
            active['selection_kind'] = 'native_semantic'
        return rows

    def insert(*args, **kwargs):
        result = old['insert'](*args, **kwargs)
        require(not args and len(kwargs.get('ids', [])) == 1, 'Native entity insert contract differs')
        active['selected_entity_id'] = str(kwargs['ids'][0])
        active['selection_kind'] = 'native_insert'
        return result

    def update(*args, **kwargs):
        result = old['update'](*args, **kwargs)
        require(not args and kwargs.get('vector') is None, 'Native entity update contract differs')
        require(active.get('selected_entity_id') == str(kwargs['vector_id']),
                'Native entity match/update target differs')
        return result

    def upsert(text, typ, mid, filters):
        nonlocal active
        require(mid == expected_mid and active is None, 'Unexpected concurrent native entity operation')
        active = {'memory_id': mid, 'entity_type': typ, 'text': text,
                  'normalized_text': memory._normalize_entity_text(text)}
        try:
            result = original(text, typ, mid, filters)
            require(active.get('selected_entity_id'), 'Native entity upsert swallowed failure')
            target = store.get(active['selected_entity_id'])
            require(target is not None and mid in (target.payload or {}).get('linked_memory_ids', []),
                    'Native entity target did not persist link')
            traces.append(copy.deepcopy(active))
            return result
        finally:
            active = None

    try:
        memory._upsert_entity = upsert
        memory._existing_entities_by_text = existing
        store.search, store.insert, store.update = search, insert, update
        yield traces
    finally:
        memory._upsert_entity = original
        memory._existing_entities_by_text = lookup
        for k, v in old.items():
            setattr(store, k, v)


def add_candidate(memory, candidate, user_id, existing_text_ids, on_add=None):
    text = candidate['statement']
    if text in existing_text_ids:
        return {'candidate': copy.deepcopy(candidate), 'memory_id': existing_text_ids[text],
                'status': 'duplicate_skipped', 'entity_trace': [], 'expected_entity_keys': []}
    expected = expected_entities(memory, text)
    result = memory.add(text, user_id=user_id, infer=False, metadata={
        'jev_mem_session_id': candidate['source_session_id'],
        'jev_mem_observation_date': candidate['observation_date']})
    rows = result.get('results', [])
    require(len(rows) == 1 and rows[0].get('event') == 'ADD'
            and rows[0].get('memory') == text and isinstance(rows[0].get('id'), str),
            'Native infer=False ADD returned unexpected result')
    mid = rows[0]['id']
    if on_add is not None:
        on_add(mid, result)  # Persist returned UUID before the fallible entity-link step.
    with entity_observer(memory, mid) as traces:
        memory._link_entities_for_memory(mid, text, {'user_id': user_id})
    require(expected == [t['normalized_text'] for t in traces],
            'Native entity extraction/link omitted or changed expected operations')
    existing_text_ids[text] = mid
    return {'candidate': copy.deepcopy(candidate), 'memory_id': mid, 'status': 'added',
            'native_result': copy.deepcopy(result), 'entity_trace': traces,
            'expected_entity_keys': expected}


def admission(path_or_run):
    from tools.prepare_write_omission_r07 import validate_admission
    return validate_admission(path_or_run)


def accepted(admit, ci, arm):
    """Derive ordered list from the complete signed item ledger, never manual selection."""
    rows = []
    for item in admit['items']:
        if item['conversation'] != ci:
            continue
        byid = {c['candidate_id']: c for c in item['candidates']}
        ids = item['accepted'][arm]
        require(len(ids) == len(set(ids)) and set(ids) <= set(byid), 'Admission IDs invalid')
        require(ids == [c['candidate_id'] for c in item['candidates'] if c['candidate_id'] in ids],
                'Admission proposer order differs')
        rows.extend(copy.deepcopy(byid[cid]) for cid in ids)
    return rows


def old_jobs(source):
    jobs = [j for j in source['jobs'] if j['method'] == 'baseline']
    require(len(jobs) == 1540 and len({j['query_id'] for j in jobs}) == 1540,
            'Fixed full baseline cohort required')
    return jobs


def prepare(run_dir):
    run = Path(run_dir).resolve()
    admit = admission(run)
    protocol = admit['protocol']
    require(protocol['research_protocol']['sha256'] == SPEC_SHA, 'R07 scientific protocol differs')
    bound(protocol['research_protocol'])
    out = run/'materialization'
    require(not out.exists(), 'Fresh materialization namespace required')
    source = bound(protocol['frozen_source'])
    jobs = old_jobs(source)
    retrieval = bound(source['retrieval_manifest'])
    resources = retrieval['resources']
    validate_resources(resources['snapshot'], resources['files_sha256'])
    pack = packs.verify_pack(retrieval['embedding_cache_pack'], check_sources=True)
    require(len(pack['full_queries']) == 1540, 'Frozen QUERY pack cohort differs')
    source_info = {}
    for ci in range(10):
        ref = protocol['bank_refs'][str(ci)]
        bank = bound(ref, sealed=True)
        require(bank['conversation'] == ci and bank['arm'] == 'session' and bank['complete']
                and not bank['native_index_audit']['entity_facts_without_any_link'],
                'Effective original N0 bank incomplete or orphaned')
        native_dir = Path(bank['native_dir']).resolve()
        state = read(Path(ref['path']).parent/'write_status.json')
        from tools.recover_gemini38_session_io_v1 import pid_alive
        require(state.get('status') == 'complete' and state.get('finished_utc')
                and not pid_alive(state['pid']), 'Original native owner not closed')
        inv = inventory(native_dir)
        declared = protocol.get('original_native_inventory_by_conversation', {}).get(str(ci))
        if declared is not None:
            require(inv == declared, 'Admission original native inventory differs')
        snap = snapshot(native_dir)
        require(set(snap['facts']) == {f['id'] for f in bank['facts']}, 'Source bank/native IDs differ')
        require(all(h['event'] == 'ADD' and not h['is_deleted'] for h in snap['history'])
                and Counter(h['memory_id'] for h in snap['history']) == Counter(snap['facts'].keys()),
                'Original native history is not exact ADD-only population')
        for f in bank['facts']:
            p = snap['facts'][f['id']]
            require(p['payload']['data'] == f['text'] and p['payload']['user_id'] == bank['user_id']
                    and p['payload'].get('jev_mem_observation_date') == f['observation_date'],
                    'Original fact projection differs')
            compare_export_embedding(p['vector'].get(''),f['embedding'])
        require(all(h['new_memory']==snap['facts'][h['memory_id']]['payload']['data']
                    and h['old_memory'] is None for h in snap['history']),
                'Original ADD-time text differs from canonical native fact')
        snap_ref = save_new(out/f'conv_{ci}/original_native_snapshot.json', snap)
        source_info[str(ci)] = {'bank': ref, 'native_dir': str(native_dir), 'inventory': inv,
                               'snapshot': snap_ref, 'user_id': bank['user_id']}
    value = {'schema': SCHEMA, 'created_utc': now(), 'admission': pointer(run/'admission_manifest.json'),
        'protocol': copy.deepcopy(protocol), 'source_banks': source_info,
        'resources': resources, 'embedding_cache_pack': retrieval['embedding_cache_pack'],
        'retrieval_manifest': source['retrieval_manifest'],
        'code_sha256': {p: sha(ROOT/p) for p in CODE},
        'native_sdk_sha256':native_sdk_identity(),
        'execution': {'materialization_concurrency': 8, 'capture_concurrency': 8,
                      'new_generation_allowed': False, 'automatic_rebuild': False},
        'baseline_slots': len(jobs), 'model_calls_at_prepare': 0}
    return save_new(out/'run_manifest.json', value)


def manifest(run_dir, validate_admission=True):
    run = Path(run_dir).resolve()
    m = sealed_read(run/'materialization/run_manifest.json')
    require(m['schema'] == SCHEMA and m['protocol']['research_protocol']['sha256'] == SPEC_SHA,
            'Materialization protocol differs')
    for path, expected in m['code_sha256'].items():
        require(sha(ROOT/path) == expected, 'Materialization code changed')
    require(m['native_sdk_sha256']==native_sdk_identity(),'Native SDK runtime changed')
    if validate_admission:
        require(admission(run) == bound(m['admission'], sealed=True), 'Admission changed')
    for ci, info in m['source_banks'].items():
        bound(info['bank'], sealed=True); bound(info['snapshot'], sealed=True)
        require(inventory(info['native_dir']) == info['inventory'], 'Original native bytes changed')
    return m


def bank_folder(run, ci, arm):
    require(type(ci) is int and ci in range(10) and arm in NEW_ARMS, 'Invalid new bank')
    return Path(run).resolve()/f'materialization/conv_{ci}/{arm}'


def materialize_bank(run_dir, ci, arm, m=None, admit=None):
    run = Path(run_dir).resolve(); m = m or manifest(run); admit = admit or admission(run)
    folder = bank_folder(run, ci, arm)
    require(not folder.exists(), 'Native bank reentry refused; preserve failed or unknown store')
    info = m['source_banks'][str(ci)]
    bank = bound(info['bank'], sealed=True)
    candidates = accepted(admit, ci, arm)
    save_new(folder/'materialization_intent.json', {'schema':SCHEMA, 'created_utc':now(),
        'pid':os.getpid(), 'conversation':ci,'arm':arm,'admission':m['admission'],
        'candidates_sha256':digest(candidates)})
    memory = None; rows = []; api = None; failure = None;phase='copy';active_candidate=None
    warnings = Warnings(); logger = logging.getLogger('mem0'); logger.addHandler(warnings)
    try:
        copy_closed(info['native_dir'], folder/'native', info['inventory'])
        before = bound(info['snapshot'], sealed=True)
        api = EmbeddingOnlyAPI(run/f'materialization/conv_{ci}/embedding_api', retries=2,
            gate_path=DEFAULT_GATE, context={'architecture':SCHEMA,'conversation':ci,
                                           'method':arm,'phase':'write_supplement'})
        with local_bm25_context(m['resources']['snapshot'], m['resources']['files_sha256']):
            phase='open_native'
            memory = writer.open_memory(folder/'native', api)
            from mem0.utils import spacy_models
            require(spacy_models.get_nlp_full() is not None and spacy_models.get_nlp_lemma() is not None
                    and memory.vector_store._get_bm25_encoder() is not None, 'Native NLP/BM25 unavailable')
            existing = {f['text']: f['id'] for f in bank['facts']}
            for index, c in enumerate(candidates):
                phase='add_and_explicit_entity_link';active_candidate=c['candidate_id']
                memory.embedding_model = Embeddings(api.scoped(candidate_id=c['candidate_id'],
                    group_id=c['source_session_id']))
                row = add_candidate(memory, c, bank['user_id'], existing,
                    on_add=lambda mid,result,index=index,c=c:save_new(folder/f'add_dispatch/{index:04d}.json',
                        {'candidate':c,'memory_id':mid,'native_result':result,
                         'status':'add_persisted_before_entity_link','created_utc':now()}))
                row['admission'] = m['admission']
                save_new(folder/f'additions/{index:04d}.json', row)
                rows.append(row)
            require(not warnings.items, 'Native materialization emitted warnings')
            native.close_memory(memory); memory = None
        phase='native_delta_audit';active_candidate=None
        after = snapshot(folder/'native')
        gate = validate_delta(before, after, rows, bank['user_id'])
        phase='dense_raw_intent_ledger_audit'
        dense_proof=audit_added_embeddings(run/f'materialization/conv_{ci}/embedding_api',ci,arm,rows,after)
        phase='bm25_cpu_replay'
        bm25 = compare_persisted_fact_vectors([folder/'native'], m['resources']['snapshot'],
                                             m['resources']['files_sha256'])
        require(bm25['facts_compared'] == bm25['exact_matches'] == len(after['facts']),
                'Persisted native BM25 values differ')
        facts = copy.deepcopy(bank['facts'])
        for row in rows:
            if row['status'] != 'added':
                continue
            c, mid = row['candidate'], row['memory_id']
            facts.append({'id':mid,'text':c['statement'],'event':'ADD','user_id':bank['user_id'],
                'observation_date':c['observation_date'], 'embedding':after['facts'][mid]['vector'][''],
                'input_scope':{'session_id':c['source_session_id'],
                    'source_turn_ids':list(dict.fromkeys(e['turn_id'] for e in c['evidence']))},
                'verified_support':[], 'support_status':'unverified_proposed_statement',
                'supplement_candidate_id':c['candidate_id'],'evidence':c['evidence'],
                'admission':m['admission']})
        b = {'schema':SCHEMA,'complete':True,'conversation':ci,'arm':arm,
            'protocol_sha256':digest(m['protocol']), 'user_id':bank['user_id'],
            'native_dir':str((folder/'native').resolve()),'facts':facts,'original_bank':info['bank'],
            'admission':m['admission'],'additions':rows,'gate':gate,'bm25':bm25,
            'dense_embedding_proof':dense_proof,
            'native_inventory':inventory(folder/'native'), 'created_utc':now()}
        phase='bank_publication'
        bank_ref = save_new(folder/'bank.json', b)
        require(inventory(info['native_dir']) == info['inventory'], 'Original native changed')
    except Exception as ex:
        failure = safe_error(ex,phase,active_candidate)
    finally:
        if memory is not None:
            try:
                native.close_memory(memory)
            except Exception as ex:
                failure = {'type':'NativeCloseFailure','cause_type':type(ex).__name__}
        logger.removeHandler(warnings)
    outcome = {'schema':SCHEMA,'conversation':ci,'arm':arm,'pid':os.getpid(),
        'finished_utc':now(),'status':'failed' if failure else 'complete',
        'materialization_intent':pointer(folder/'materialization_intent.json'),
        'bank':None if failure else bank_ref,'error':failure,'warnings':warnings.items,
        'embedding_api_dir':str((run/f'materialization/conv_{ci}/embedding_api').resolve())}
    save_new(folder/'materialization_outcome.json', outcome)
    return outcome


def materialize_conversation(run, ci, m, admit):
    rows=[]
    for arm in NEW_ARMS:
        row=materialize_bank(run,ci,arm,m,admit);rows.append(row)
        if row['status']!='complete':
            break
    return rows


def materialize(run_dir):
    """At most eight conversation processes, with sequential arms and one cache per conversation."""
    run = Path(run_dir).resolve(); m = manifest(run); admit = admission(run)
    intent = run/'materialization/controller_intent.json'
    save_new(intent, {'schema':SCHEMA,'pid':os.getpid(),'created_utc':now(),'recovery':False})
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(materialize_conversation,run,ci,m,admit):ci for ci in range(10)}
        for future in as_completed(tasks):
            ci=tasks[future]
            try:
                rows.extend(future.result())
            except Exception as ex:
                rows.append({'conversation':ci,'status':'failed','error_type':type(ex).__name__,
                             'retained_namespace':True})
    rows.sort(key=lambda r:(r['conversation'],r.get('arm','')))
    return save_new(run/'materialization/controller_outcome.json', {'schema':SCHEMA,
        'finished_utc':now(),'status':'complete' if len(rows)==20 and all(r['status']=='complete'
            for r in rows) else 'failed','banks':rows,'intent':pointer(intent)})


def audit_banks(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run); admit = admission(run)
    outcome = sealed_read(run/'materialization/controller_outcome.json')
    require(outcome['status'] == 'complete' and len(outcome['banks']) == 20,
            'All twenty banks must be terminal before admission')
    from tools.recover_gemini38_session_io_v1 import pid_alive
    require(not pid_alive(bound(outcome['intent'], sealed=True)['pid']),
            'Materialization controller still alive')
    refs = {};embedding_audits={}
    for ci in range(10):
        info = m['source_banks'][str(ci)]
        by_arm={}
        for arm in NEW_ARMS:
            folder = bank_folder(run, ci, arm)
            o = sealed_read(folder/'materialization_outcome.json')
            require(o['status'] == 'complete' and not pid_alive(o['pid']), 'Native owner not closed')
            b = bound(o['bank'], sealed=True)
            require(inventory(folder/'native') == b['native_inventory'], 'Frozen new native bytes differ')
            rows = b['additions']
            by_arm[arm]=rows
            require([r['candidate'] for r in rows] == accepted(admit, ci, arm),
                    'Native layer altered accepted supplement list')
            validate_delta(bound(info['snapshot'], sealed=True), snapshot(folder/'native'),
                           rows, info['user_id'])
            require(audit_added_embeddings(run/f'materialization/conv_{ci}/embedding_api',ci,arm,
                    rows,snapshot(folder/'native'))==b['dense_embedding_proof'],
                    'Bound ADD dense raw/intent/ledger replay differs')
            refs[f'{ci}/{arm}'] = o['bank']
        embedding_audits[str(ci)]=audit_embedding_inventory(
            run/f'materialization/conv_{ci}/embedding_api',ci,by_arm)
    return save_new(run/'materialization/index_admission.json', {'schema':SCHEMA,
        'status':'passed','admission':m['admission'],'banks':refs,'new_banks':20,
        'source_original_facts':1637,'semantic_support_certified':False,'model_calls':0,
        'embedding_audits':embedding_audits})


def replay_capture(capture, scorer=None):
    require(capture.get('schema') == CAPTURE_SCHEMA and capture.get('capture_sha256') ==
            digest({k:v for k,v in capture.items() if k != 'capture_sha256'}), 'Capture seal differs')
    if scorer is None:
        from mem0.memory.main import score_and_rank
        scorer = score_and_rank
    inputs = copy.deepcopy(capture['inputs'])
    require(inputs['top_k'] == 20 and len(inputs['semantic_results']) <= 80,
            'Native capture search budget differs')
    base = scorer(**inputs)
    inputs['top_k'] = len(inputs['semantic_results'])
    eligible = scorer(**inputs)
    require(base == capture['base20'] == eligible[:20] and eligible == capture['eligible']
            and [r['id'] for r in base] == capture['base20_ids']
            and len(base) == 20 and len({r['id'] for r in base}) == 20,
            'Original native SDK20 capture does not replay')
    return [r['id'] for r in base]


def selected_facts(bank, selected_ids):
    byid = {f['id']:f for f in bank['facts']}
    require(len(selected_ids)==20 and len(set(selected_ids))==20 and set(selected_ids)<=set(byid),
            'Native selection must contain twenty unique same-bank facts')
    return [{k:byid[mid].get(k) for k in ('id','text','observation_date')} for mid in selected_ids]


def validate_capture_bank(capture, bank, native_snapshot):
    ids=replay_capture(capture)
    require(set(capture['candidate_ids'])<=set(native_snapshot['facts']),
            'Capture candidate outside the frozen native bank')
    require(capture['candidate_ids']==[r['id'] for r in capture['inputs']['semantic_results']],
            'Captured candidate IDs differ from original scorer inputs')
    require(capture['eligible_ids']==[r['id'] for r in capture['eligible']]
            and capture['native_scores']=={r['id']:r['score'] for r in capture['eligible']},
            'Captured native score/eligible metadata differs')
    for row in capture['inputs']['semantic_results']:
        require(row.get('payload')==native_snapshot['facts'][row['id']]['payload'],
                'Captured semantic payload differs from frozen native fact')
    facts=selected_facts(bank,ids)
    for f in facts:
        payload=native_snapshot['facts'][f['id']]['payload']
        require(f['text']==payload['data'] and f['observation_date']==payload.get('jev_mem_observation_date'),
                'Bank Reader values differ from native payload')
    return ids


def retrieve_conversation(run,ci,m,gate,jobs,entries):
    """A dedicated process owns all native scorer captures for one conversation."""
    run=Path(run)
    rows = []
    for arm in NEW_ARMS:
            folder = bank_folder(run, ci, arm)
            ref = gate['banks'][f'{ci}/{arm}']; bank = bound(ref, sealed=True)
            require(inventory(folder/'native')==bank['native_inventory'], 'Frozen native changed before search')
            memory = None; before = snapshot(folder/'native'); fail = None; warnings = Warnings()
            logger = logging.getLogger('mem0'); logger.addHandler(warnings)
            api = CacheOnlyEmbeddings({e['request_hash']:e for e in entries})
            selections = []
            try:
                with local_bm25_context(m['resources']['snapshot'],m['resources']['files_sha256']):
                    memory = writer.open_memory(folder/'native', api)
                    for job in [j for j in jobs if j['conversation']==ci]:
                        scoped = api.scoped(conversation=ci,method=arm,query_id=job['query_id'],phase='retrieval')
                        memory.embedding_model = Embeddings(scoped)
                        with capture_native_pool() as capture:
                            result = memory.search(job['question'],top_k=20,
                                filters={'user_id':bank['user_id']},rerank=False)
                        ids = validate_capture_bank(capture,bank,before)
                        require(ids==[r['id'] for r in result['results']], 'Native returned order differs')
                        require(not warnings.items, 'Native retrieval emitted warnings')
                        value = {'schema':SCHEMA,'status':'complete','conversation':ci,'method':arm,
                            'query_id':job['query_id'],'original_job_sha256':digest(job),'bank':ref,
                            'capture':capture,'selected_ids':ids,'selected_facts':selected_facts(bank,ids),
                            'cache_events':scoped.events,'model_calls':0}
                        selections.append(save_new(folder/f'selections/{job["query_id"]}.json',value))
                    native.close_memory(memory); memory=None
                require(snapshot(folder/'native')==before, 'Native readonly search altered logical store')
                require(inventory(folder/'native')==bank['native_inventory'], 'Native search altered persisted bytes')
            except Exception as ex:
                fail=safe_error(ex,'native_readonly_retrieval',job['query_id'] if 'job' in locals() else None)
            finally:
                if memory is not None:
                    try:
                        native.close_memory(memory)
                    except Exception as ex:
                        fail={'type':'NativeCloseFailure','cause_type':type(ex).__name__}
                logger.removeHandler(warnings)
            row={'schema':SCHEMA,'conversation':ci,'arm':arm,'status':'failed' if fail else 'complete',
                 'pid':os.getpid(),'finished_utc':now(),'bank':ref,'selections':selections,
                 'error':fail,'warnings':warnings.items,'model_calls':0}
            save_new(folder/'retrieval_outcome.json',row); rows.append(row)
            if fail:
                break
    return rows


def retrieve(run_dir):
    """Eight process-isolated conversations; old QUERY cache has no HTTP fallback."""
    run = Path(run_dir).resolve(); m = manifest(run)
    gate = sealed_read(run/'materialization/index_admission.json')
    require(gate['status']=='passed' and len(gate['banks'])==20, 'Index admission missing')
    source = bound(m['protocol']['frozen_source']); jobs = old_jobs(source)
    pack = packs.verify_pack(m['embedding_cache_pack'], check_sources=True)
    save_new(run/'retrieval_execution_intent.json', {'schema':SCHEMA,'pid':os.getpid(),
                                                   'created_utc':now(),'model_calls':0})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(retrieve_conversation,run,ci,m,gate,
            [j for j in jobs if j['conversation']==ci],
            [e for e in pack['entries'] if e['conversation']==ci]):ci for ci in range(10)}
        for future in as_completed(tasks):
            ci=tasks[future]
            try:
                rows.extend(future.result())
            except Exception as ex:
                rows.append({'conversation':ci,'status':'failed','error_type':type(ex).__name__,
                             'retained_namespace':True,'model_calls':0})
    rows.sort(key=lambda r:(r['conversation'],r.get('arm','')))
    return save_new(run/'retrieval_outcome.json', {'schema':SCHEMA,'finished_utc':now(),
        'status':'complete' if len(rows)==20 and all(r['status']=='complete' for r in rows) else 'failed',
        'banks':rows,'model_calls':0})


def construct_jobs(run, m, gate, original_jobs):
    jobs=[]
    banks={k:bound(v,sealed=True) for k,v in gate['banks'].items()}
    native_snapshots={k:snapshot(v['native_dir']) for k,v in banks.items()}
    for arm in ARMS:
        for old in original_jobs:
            job=copy.deepcopy(old); job.update(method=arm,job_id=f'{arm}/{old["query_id"]}',
                source_job_sha256=digest(old),original_job=copy.deepcopy(old))
            if arm != 'baseline':
                folder=bank_folder(run,old['conversation'],arm)
                ref=pointer(folder/f'selections/{old["query_id"]}.json')
                selection=bound(ref,sealed=True)
                bank_ref=gate['banks'][f'{old["conversation"]}/{arm}']
                bank=banks[f'{old["conversation"]}/{arm}']
                ids=validate_capture_bank(selection['capture'],bank,
                    native_snapshots[f'{old["conversation"]}/{arm}'])
                require(selection['status']=='complete' and selection['query_id']==old['query_id']
                        and selection['original_job_sha256']==digest(old) and selection['bank']==bank_ref
                        and selection['selected_ids']==ids
                        and selection['selected_facts']==selected_facts(bank,ids), 'Selection lineage differs')
                job.update(selected_ids=ids,selected_facts=selection['selected_facts'],
                           bank=bank_ref,selection=ref)
                job['lineage']={**job.get('lineage',{}),'bank':bank_ref,'selection':ref,
                    'admission':m['admission'],'index_admission':pointer(Path(run)/'materialization/index_admission.json')}
            job['selected_sha256']=digest(job['selected_facts'])
            jobs.append(job)
    require(len(jobs)==4620,'Full three-arm cohort required')
    return jobs


def validate_retrieval_terminal(run,m,gate,original_jobs,require_dead=True):
    run=Path(run).resolve();outcome=sealed_read(run/'retrieval_outcome.json')
    require(outcome['schema']==SCHEMA and outcome['status']=='complete'
            and len(outcome['banks'])==20 and outcome.get('model_calls')==0,
            'Retrieval terminal stage is incomplete')
    from tools.recover_gemini38_session_io_v1 import pid_alive
    if require_dead:
        require(not pid_alive(sealed_read(run/'retrieval_execution_intent.json')['pid']),
                'Retrieval controller still alive')
    expected={(ci,arm) for ci in range(10) for arm in NEW_ARMS}
    require({(r['conversation'],r['arm']) for r in outcome['banks']}==expected,
            'Retrieval terminal bank identity/count differs')
    for row in outcome['banks']:
        ci,arm=row['conversation'],row['arm']
        folder=bank_folder(run,ci,arm)
        actual=sealed_read(folder/'retrieval_outcome.json')
        require({k:v for k,v in actual.items() if k!='content_sha256'}==row
                and row['status']=='complete' and row['schema']==SCHEMA and row['model_calls']==0
                and row['bank']==gate['banks'][f'{ci}/{arm}'] and not row['error'] and not row['warnings'],
                'Per-bank readonly retrieval terminal binding differs')
        if require_dead:
            require(not pid_alive(row['pid']),'Retrieval native owner still alive')
        qids=[j['query_id'] for j in original_jobs if j['conversation']==ci]
        expected_refs=[pointer(folder/f'selections/{qid}.json') for qid in qids]
        require(row['selections']==expected_refs,'Exact per-bank fixed cohort selection refs differ')
        paths={p.resolve() for p in (folder/'selections').glob('*.json')}
        require(paths=={Path(r['path']).resolve() for r in expected_refs},'Orphan retrieval selection file')
    return outcome


def seal(run_dir):
    run=Path(run_dir).resolve();m=manifest(run)
    gate=sealed_read(run/'materialization/index_admission.json')
    for key,ref in gate['banks'].items():
        bank=bound(ref,sealed=True)
        require(inventory(bank['native_dir'])==bank['native_inventory'],'Frozen bank bytes differ')
    source=bound(m['protocol']['frozen_source'])
    original_jobs=old_jobs(source)
    validate_retrieval_terminal(run,m,gate,original_jobs)
    jobs=construct_jobs(run,m,gate,original_jobs)
    p=m['protocol']
    value={'schema':SOURCE_SCHEMA,'created_utc':now(),'status':'ready','arms':list(ARMS),
        'jobs':jobs,'questions_per_arm':1540,'frozen_source':p['frozen_source'],
        'frozen_qa_run_manifest':p['frozen_qa_run_manifest'],
        'frozen_baseline_reuse_proof':p['frozen_baseline_reuse_proof'],
        'reuse_proof_qa_manifest':p['reuse_proof_qa_manifest'],
        'write_omission_protocol':p['research_protocol'],'selector_manifest':pointer(run/'run_manifest.json'),
        'admission_manifest':m['admission'],'materialization_manifest':pointer(run/'materialization/run_manifest.json'),
        'index_admission':pointer(run/'materialization/index_admission.json'),
        'retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'official_metric_provenance':source['official_metric_provenance'],
        'source_dataset_sha256':source['source_dataset_sha256'],
        'protocol_sha256':digest(m['protocol']),'code_sha256':m['code_sha256'],
        'selection_counts':{a:{'jobs':1540} for a in ARMS}}
    return save_new(run/'selected_inputs.json',value)


def validate_source(path, fresh=False):
    """Byte-bound source/index/admission/capture replay; IDs alone never prove reuse."""
    path=Path(path).resolve();before=sha(path);source=sealed_read(path)
    require(source['schema']==SOURCE_SCHEMA and source['arms']==list(ARMS) and source['status']=='ready',
            'R07 selected-input schema differs')
    run=path.parent;m=manifest(run)
    validate_source_pointers(source,run,m)
    for field in ('selector_manifest','admission_manifest','materialization_manifest',
                  'index_admission','retrieval_outcome','write_omission_protocol'):
        bound(source[field],sealed=field not in ('write_omission_protocol','selector_manifest'))
    require(source['write_omission_protocol']==m['protocol']['research_protocol']
            and source['protocol_sha256']==digest(m['protocol']) and source['code_sha256']==m['code_sha256'],
            'Selected source protocol/code differs')
    for field in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):
        require(source[field]==m['protocol'][field], 'Exact N0 proof lineage differs')
        bound(source[field])  # Published historical proof byte seal, not generic content seal.
    gate=bound(source['index_admission'],sealed=True)
    require(gate['status']=='passed' and len(gate['banks'])==20,'Index stage admission differs')
    admit=admission(run)
    original_source=bound(source['frozen_source']);original_jobs=old_jobs(original_source)
    require(source['source_dataset_sha256']==original_source['source_dataset_sha256'],
            'Frozen source dataset identity differs')
    validate_retrieval_terminal(run,m,gate,original_jobs)
    for key,ref in gate['banks'].items():
        ci,arm=key.split('/');ci=int(ci);b=bound(ref,sealed=True)
        require(b['conversation']==ci and b['arm']==arm and b['complete']
                and b['admission']==m['admission'],'New bank identity differs')
        require(inventory(b['native_dir'])==b['native_inventory'],'New native bytes changed')
        require([r['candidate'] for r in b['additions']]==accepted(admit,ci,arm),'Accepted list altered')
        validate_delta(bound(m['source_banks'][str(ci)]['snapshot'],sealed=True),
                       snapshot(b['native_dir']),b['additions'],b['user_id'])
        require(audit_added_embeddings(run/f'materialization/conv_{ci}/embedding_api',ci,arm,
                    b['additions'],snapshot(b['native_dir']))==b['dense_embedding_proof'],
                'New-bank dense embedding provenance changed')
    for ci in range(10):
        by_arm={a:bound(gate['banks'][f'{ci}/{a}'],sealed=True)['additions'] for a in NEW_ARMS}
        require(audit_embedding_inventory(run/f'materialization/conv_{ci}/embedding_api',ci,by_arm)
                ==gate['embedding_audits'][str(ci)], 'Frozen whole embedding ledger audit differs')
    expected=construct_jobs(run,m,gate,original_jobs)
    require(source['jobs']==expected and source['official_metric_provenance']==
            bound(source['frozen_source'])['official_metric_provenance'], 'Full selected-job replay differs')
    require(sha(path)==before,'Selected source changed during validation')
    return source


def validate_source_pointers(source,run,m):
    run=Path(run).resolve()
    expected={'selector_manifest':pointer(run/'run_manifest.json'),
        'admission_manifest':m['admission'],
        'materialization_manifest':pointer(run/'materialization/run_manifest.json'),
        'index_admission':pointer(run/'materialization/index_admission.json'),
        'retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'write_omission_protocol':m['protocol']['research_protocol']}
    require(all(source.get(k)==v for k,v in expected.items()),
            'Top-level selected source lineage points outside its actual run')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','materialize','audit','retrieve','seal','check'))
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    funcs={'prepare':prepare,'materialize':materialize,'audit':audit_banks,'retrieve':retrieve,'seal':seal,
           'check':lambda r:{'valid_jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),
                             'model_calls':0}}
    print(json.dumps(funcs[args.command](args.run_dir),ensure_ascii=False))


if __name__=='__main__':
    main()
