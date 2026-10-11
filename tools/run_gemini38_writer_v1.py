"""Fresh four-bank Gemini 3.8 native Writer; prepare/check are strictly offline.

Frozen session/group partitions and native extraction semantics are reused.
Only an explicit write stage dispatches requests. No unresolved store is replayed.
"""
from __future__ import annotations
import argparse
import copy
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_regroup_sdk20 as grouped
from tools import run_session_experiments as native
from tools import run_mem0_sparse1540 as sparse
from tools.run_mem0_regroup_remaining_v1 import resources
from tools.run_sdk20_modules import digest, verify, sha, now
from tools.run_pilot400_experiments_v2 import publish_json
from jev_mem.data import load_data, session_envelopes, SOURCE_HASH
from jev_mem.mem0_adapter import Mem0SessionAdapter
from jev_mem.providers import Embeddings
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem.paced_embedding_api import DEFAULT_GATE
from jev_mem.gemini38_provider_v1 import (
    Gemini38API, Gemini38ExtractionLLM, MODEL, THINKING_LEVEL, WRITER_MAX_OUTPUT,
)

SCHEMA = 'gemini38-fresh-native-writer-v1'
ARMS = ('session', 'source_local_context', 'boundary_multiscale', 'boundary_neighbor')
CURRENT_DATE = '2026-09-27'
SOURCE_DIRS = {
    'source_local_context': (ROOT/'runs/mem0_source_local_context_v1', 'source_local_context'),
    'boundary_multiscale': (ROOT/'runs/boundary_variants_v1_20261003/c1_multiscale/writer', 'c1_multiscale'),
    'boundary_neighbor': (ROOT/'runs/boundary_variants_v1_20261003/c2_neighbour/writer', 'c2_neighbour'),
}
CODE = tuple(dict.fromkeys((*grouped.CODE, *sparse.CODE_FILES,
    'tools/run_gemini38_writer_v1.py', 'jev_mem/gemini38_provider_v1.py',
    'jev_mem/source_local_group_context_v1.py', 'jev_mem/local_bm25_context_v2.py',
    'jev_mem/paced_embedding_api.py', 'jev_mem/sdk20_cached_api.py',
    'tools/run_mem0_regroup_remaining_v1.py')))


def prepare(run):
    run = Path(run).resolve()
    if run.exists():
        raise FileExistsError('Fresh independent namespace required')
    if sha(ROOT/'data/locomo10.json') != SOURCE_HASH:
        raise ValueError('Frozen dataset changed')
    data = load_data()
    prepared = []
    sources = {}
    for ci, item in enumerate(data):
        for arm in ARMS:
            if arm == 'session':
                envelopes = list(session_envelopes(item))
                source = grouped.pointer(ROOT/'data/locomo10.json')
                value = {'schema': SCHEMA, 'conversation': ci, 'arm': arm,
                         'envelopes': envelopes, 'source': source,
                         'dataset_sha256': SOURCE_HASH}
            else:
                directory, source_arm = SOURCE_DIRS[arm]
                path = directory/f'conv_{ci}'/source_arm/'group_inputs.json'
                old = verify(path)
                if (old['conversation'], old['arm'], old['dataset_sha256']) != (ci, source_arm, SOURCE_HASH):
                    raise ValueError('Historical group input identity differs')
                grouping = grouped.check_pointer(old['grouping'])
                envelopes = grouped.make_envelopes(item, grouping, 'dated_merged')
                if envelopes != old['envelopes']:
                    raise ValueError('Frozen source partition/rendering differs')
                if arm == 'source_local_context':
                    from jev_mem.source_local_group_context_v1 import build_context_plan
                    for envelope in envelopes:
                        build_context_plan(envelope, source_sessions=list(session_envelopes(item)),
                                           segments=grouping['segments'])
                source = grouped.pointer(path)
                sources[str(path.resolve())] = source['sha256']
                sources[old['grouping']['path']] = old['grouping']['sha256']
                value = {k: copy.deepcopy(v) for k, v in old.items() if k != 'content_sha256'}
                value.update(schema=SCHEMA, arm=arm, historical_source=source)
            prepared.append((ci, arm, value))
    if len(prepared) != 40 or sum(len(v['envelopes']) for ci,a,v in prepared if a == 'session') != 272:
        raise ValueError('Expected forty banks and 272 original sessions')
    resource = resources()
    inputs = []
    for ci, arm, value in prepared:
        path = run/f'conv_{ci}'/arm/'group_inputs.json'
        grouped.save_new(path, value)
        inputs.append({'conversation': ci, 'arm': arm, 'input': grouped.pointer(path)})
    m = {'schema': SCHEMA, 'created_utc': now(), 'arms': list(ARMS),
         'dataset_sha256': SOURCE_HASH, 'inputs': inputs, 'source_sha256': sources,
         'code_sha256': {p: sha(ROOT/p) for p in CODE},
         'dependencies': sparse.dependency_versions(), 'resources': resource,
         'current_date': CURRENT_DATE, 'models': {'writer': MODEL,
             'embedding': 'gemini-embedding-001', 'dimensions': 768},
         'generation_config': {'maxOutputTokens': WRITER_MAX_OUTPUT,
             'thinkingConfig': {'thinkingLevel': THINKING_LEVEL}, 'responseMimeType': 'application/json'},
         'policy': {'max_processes': 8, 'transport_retries': 2, 'external_cache_roots': [],
             'new_empty_banks': True, 'source_partitions_recomputed': False,
             'failed_or_unknown_write_reentry': False, 'development_conversations': [0, 1],
             'expansion_requires_bound_development_review': True},
         'gate_path': str(DEFAULT_GATE.resolve()), 'model_calls_at_prepare': 0,
         'qa_arms': {'N0': 'session', 'H0': 'session', 'H1': 'session', 'H2': 'session',
             'local_context': 'source_local_context', 'C1': 'boundary_multiscale', 'C2': 'boundary_neighbor'},
         'writer_semantics': 'Pinned native Memory.add and original ADDITIVE_EXTRACTION_PROMPT/parser; exact old partitions/dates/history; new generation model/config only'}
    publish_json(run/'run_manifest.json', {**m, 'protocol_sha256': digest(m)})
    return {'prepared': str(run), 'banks': 40, 'session_count': 272, 'model_calls': 0}


def manifest(run):
    run = Path(run).resolve()
    m = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    if (m.get('schema') != SCHEMA or m.get('protocol_sha256') != digest({k:v for k,v in m.items() if k != 'protocol_sha256'})
        or m.get('arms') != list(ARMS) or m.get('models', {}).get('writer') != MODEL
        or m.get('generation_config') != {'maxOutputTokens': WRITER_MAX_OUTPUT,
            'thinkingConfig': {'thinkingLevel': THINKING_LEVEL}, 'responseMimeType': 'application/json'}):
        raise ValueError('New Writer protocol identity/config changed')
    if sha(ROOT/'data/locomo10.json') != SOURCE_HASH or m['dependencies'] != sparse.dependency_versions():
        raise ValueError('Dataset/dependencies changed')
    if m['code_sha256'] != {p: sha(ROOT/p) for p in CODE}:
        raise ValueError('Frozen/new code changed after preparation')
    for p,h in m['source_sha256'].items():
        if sha(p) != h:
            raise ValueError('Source partition changed')
    expected = {(ci, a) for ci in range(10) for a in ARMS}
    if len(m['inputs']) != 40 or {(r['conversation'], r['arm']) for r in m['inputs']} != expected:
        raise ValueError('Forty unique bank input bindings required')
    for row in m['inputs']:
        value = grouped.check_pointer(row['input'])
        if (value['conversation'], value['arm'], value['dataset_sha256']) != (row['conversation'], row['arm'], SOURCE_HASH):
            raise ValueError('Input bank binding differs')
    validate_resources(m['resources']['snapshot'], m['resources']['files_sha256'])
    return m


def open_memory(folder, api):
    """Original native store/embed config; unused SDK generation client is 3.8 too."""
    from mem0 import Memory
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    memory = Memory.from_config({
        'vector_store': {'provider': 'qdrant', 'config': {'collection_name': 'facts',
            'embedding_model_dims': 768, 'path': str(folder/'qdrant'), 'on_disk': True}},
        'llm': {'provider': 'gemini', 'config': {'model': MODEL,
            'api_key': api.keys['GEMINI_API_KEY'], 'vertexai': False, 'max_tokens': WRITER_MAX_OUTPUT}},
        'embedder': {'provider': 'gemini', 'config': {'model': 'gemini-embedding-001',
            'api_key': api.keys['GEMINI_API_KEY'], 'embedding_dims': 768}},
        'history_db_path': str(folder/'history.sqlite')})
    memory.llm.client.close()
    memory.embedding_model.client.close()
    memory.llm = Gemini38ExtractionLLM(api)
    memory.embedding_model = Embeddings(api)
    return memory


def write_sessions(run, ci, m):
    folder = Path(run)/f'conv_{ci}'/'session'
    inp = verify(folder/'group_inputs.json')
    api = Gemini38API(folder/'native', retries=2, context={'architecture': SCHEMA,
        'conversation': ci, 'method': 'session', 'phase': 'write'}, gate_path=m['gate_path'])
    state = {'schema': SCHEMA, 'protocol_sha256': m['protocol_sha256'], 'conversation': ci,
        'arm': 'session', 'status': 'running', 'started_utc': now(), 'completed_groups': 0,
        'expected_groups':len(inp['envelopes']), 'facts':0, 'pid':os.getpid(), 'active_session':None}
    def status():
        state['updated_utc'] = now()
        publish_json(folder/'write_status.json', state)
    status()
    memory = None
    try:
        memory = open_memory(folder/'native', api)
        from mem0.utils import spacy_models
        if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None or memory.vector_store._get_bm25_encoder() is None:
            raise RuntimeError('Native NLP/BM25 unavailable')
        writer = Mem0SessionAdapter(memory, folder/'native/session_audit', current_date=CURRENT_DATE)
        uid = f'gemini38_c{ci}_session'
        records = []
        for envelope in inp['envelopes']:
            state['active_session'] = envelope['session_id']
            status()
            scoped = api.scoped(session_id=envelope['session_id'])
            memory.llm = Gemini38ExtractionLLM(scoped)
            memory.embedding_model = Embeddings(scoped)
            records.append(writer.add_session(user_id=uid, **envelope))
            state['completed_groups'] = len(records)
            state['facts'] = sum(len(r['fact_records']) for r in records)
            state['last_group'] = envelope['session_id']
            state['active_session'] = None
            status()
        facts = [f for r in records for f in r['fact_records']]
        for f in facts:
            point = memory.vector_store.client.retrieve(collection_name='facts', ids=[f['id']], with_vectors=True, with_payload=True)[0]
            if point.payload['data'] != f['text']:
                raise ValueError('Persisted native session fact changed')
            f['embedding'] = point.vector.get('') if isinstance(point.vector, dict) else point.vector
        sparse.vector_ok(facts)
        if len(facts) < 20:
            raise ValueError('Complete native bank requires twenty facts')
        index = sparse.sidecar_audit(memory, facts, uid)
        bank = {'schema': SCHEMA, 'protocol_sha256': m['protocol_sha256'], 'conversation': ci,
            'arm': 'session', 'complete': True, 'user_id': uid, 'facts': facts,
            'group_inputs': grouped.pointer(folder/'group_inputs.json'),
            'journal_files': [grouped.pointer(r['audit_path']) for r in records],
            'native_index_audit': index, 'native_dir': str((folder/'native').resolve()),
            'source_dataset_sha256': SOURCE_HASH, 'fact_support': 'unverified_session_input_scope'}
        grouped.save_new(folder/'bank.json', bank)
        state.update(status='complete', bank=grouped.pointer(folder/'bank.json'))
    except Exception as ex:
        state.update(status='failed', error={'type': type(ex).__name__, 'message': api._redact(ex)[:1500]})
    finally:
        if memory is not None:
            try:
                native.close_memory(memory)
            except Exception as ex:
                state.update(status='failed', close_error_type=type(ex).__name__,
                    error={'type':'NativeCloseFailure', 'message':'Native close failed; store retained for review'})
        state['finished_utc'] = now()
        status()
    return state


def development_review(run, m):
    review = verify(Path(run)/'development_review.json')
    expected = {(ci,a) for ci in (0,1) for a in ARMS}
    if review.get('status') != 'passed' or review.get('protocol_sha256') != m['protocol_sha256']:
        raise ValueError('Bound development source/index review required')
    if len(review.get('banks', [])) != 8:
        raise ValueError('Exactly eight development banks required')
    seen = set()
    for pointer in review['banks']:
        bank = grouped.check_pointer(pointer)
        seen.add((bank['conversation'], bank['arm']))
        if not bank.get('complete') or bank['protocol_sha256'] != m['protocol_sha256']:
            raise ValueError('Development bank incomplete or another protocol')
    if seen != expected or not review.get('source_date_audits') or not review.get('index_audits'):
        raise ValueError('Eight source/date/index audited banks required')
    for p in review['source_date_audits'] + review['index_audits']:
        grouped.check_pointer(p)


def worker(run, ci, arm):
    run = Path(run).resolve()
    m = manifest(run)
    if type(ci) is not int or ci not in range(10) or arm not in ARMS:
        raise ValueError('Invalid bank job')
    if ci > 1:
        development_review(run, m)
    folder = run/f'conv_{ci}'/arm
    outcome = folder/'write_outcome.json'
    if outcome.exists():
        old = verify(outcome)
        if old.get('status') != 'complete' or old.get('protocol_sha256') != m['protocol_sha256']:
            raise RuntimeError('Failed/unknown Writer outcome retained; no automatic rebuild')
        bank = grouped.check_pointer(old['bank'])
        if not bank.get('complete') or (bank['conversation'], bank['arm'], bank['protocol_sha256']) != (ci, arm, m['protocol_sha256']):
            raise ValueError('Completed bank identity differs')
        return old
    if any((folder/n).exists() for n in ('write_intent.json','native','write_status.json')):
        raise RuntimeError('Unresolved Writer store/intent retained; no redispatch')
    with (folder/'write_intent.json').open('x', encoding='utf-8') as h:
        json.dump({'protocol_sha256':m['protocol_sha256'], 'conversation':ci, 'arm':arm,
                   'pid':os.getpid(), 'created_utc':now()}, h)
        h.flush(); os.fsync(h.fileno())
    originals = (grouped.ModelAPI, grouped.Mem0GroupAdapter, native.open_memory, native.ExtractionLLM)
    state = None
    try:
        native.open_memory = open_memory
        native.ExtractionLLM = Gemini38ExtractionLLM
        grouped.ModelAPI = functools.partial(Gemini38API, gate_path=m['gate_path'])
        if arm == 'source_local_context':
            from jev_mem.source_local_group_context_v1 import SourceLocalMem0GroupAdapter
            inp = verify(folder/'group_inputs.json')
            grouping = grouped.check_pointer(inp['grouping'])
            grouped.Mem0GroupAdapter = functools.partial(SourceLocalMem0GroupAdapter,
                source_sessions=list(session_envelopes(load_data()[ci])), segments=grouping['segments'])
        with local_bm25_context(m['resources']['snapshot'], m['resources']['files_sha256']):
            state = write_sessions(run, ci, m) if arm == 'session' else grouped.write_arm(run, ci, arm, m)
    except Exception as ex:
        state = {'schema':SCHEMA, 'protocol_sha256':m['protocol_sha256'], 'conversation':ci,
            'arm':arm, 'status':'failed', 'pid':os.getpid(), 'finished_utc':now(),
            'error':{'type':type(ex).__name__, 'message':'Local Writer failure; retained intent/store requires audit'},
            'store_reentry_forbidden':True}
        status_path=folder/'write_status.json'
        if status_path.exists():
            state['retained_prior_status_sha256'] = sha(status_path)
            state['retained_prior_status_value'] = json.loads(status_path.read_text(encoding='utf-8'))
        publish_json(status_path,state)
    finally:
        grouped.ModelAPI, grouped.Mem0GroupAdapter, native.open_memory, native.ExtractionLLM = originals
    result = {'schema':SCHEMA, 'protocol_sha256':m['protocol_sha256'], 'conversation':ci,
        'arm':arm, 'status':state['status'], 'write_status':grouped.pointer(folder/'write_status.json'),
        'finished_utc':now(), 'external_cache_roots':[]}
    if state['status'] == 'complete':
        result['bank'] = state['bank']
    else:
        result['error'] = state.get('error')
    grouped.save_new(outcome, result)
    return result


def run_write(run, conversations, arms):
    run = Path(run).resolve()
    m = manifest(run)
    if (not conversations or len(set(conversations)) != len(conversations)
        or any(type(ci) is not int or ci not in range(10) for ci in conversations)
        or not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms)):
        raise ValueError('Unique declared conversation/arm jobs required')
    if any(ci > 1 for ci in conversations):
        development_review(run,m)
    jobs = [(ci,a) for ci in conversations for a in arms]
    rows = []
    publish_json(run/'controller_status.json', {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],
        'pid':os.getpid(),'terminal':False,'status':'running','started_utc':now(),
        'jobs':[{'conversation':ci,'arm':a} for ci,a in jobs], 'outcomes':[]})
    with grouped.writer_process_lock(run), ProcessPoolExecutor(max_workers=min(8,len(jobs))) as pool:
        tasks = {pool.submit(worker,run,ci,a):(ci,a) for ci,a in jobs}
        for future in as_completed(tasks):
            ci,a = tasks[future]
            try:
                result = future.result()
            except Exception as ex:
                result = {'conversation':ci, 'arm':a, 'status':'failed', 'error_type':type(ex).__name__}
            rows.append(result)
            publish_json(run/'controller_status.json', {'schema':SCHEMA, 'protocol_sha256':m['protocol_sha256'],
                'pid':os.getpid(),'updated_utc':now(),'terminal':len(rows)==len(jobs),
                'status':'complete' if len(rows)==len(jobs) and all(r['status']=='complete' for r in rows) else 'failed' if len(rows)==len(jobs) else 'running',
                'jobs':[{'conversation':ci,'arm':a} for ci,a in jobs], 'outcomes':rows})
    return {'terminal':True,'jobs':len(jobs),'complete':sum(r['status']=='complete' for r in rows),'outcomes':rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare','check','write'))
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--conversations', nargs='+', type=int, default=[0,1])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    a = p.parse_args()
    if a.stage == 'prepare': result = prepare(a.run_dir)
    elif a.stage == 'check': result = {'valid':bool(manifest(a.run_dir)), 'model_calls':0}
    else: result = run_write(a.run_dir,a.conversations,a.arms)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
