"""Offline exact-cache native replay, hybrid selection and full1540 admission.

Never initializes credentials or dispatches an API. Only new native copies are
opened. Missing embedding cache or any native replay mismatch blocks QA.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_sdk20_modules import digest, save_new, verify, sha, now, Warnings
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.freeze_sdk20_full1540_v1 import verify_lock, LOCK
from tools import run_sdk20_modules as legacy
from tools import run_session_experiments as native
from tools import mem0_regroup_plain_qa_sources_v1 as sources
from tools.run_mem0_regroup_remaining_v1 import resources
from tools.run_mem0_regroup_plain_qa_v1 import durable_intent
from tools.summarize_pilot400 import verified_official_metric
from jev_mem import qa_embedding_cache_pack_v1 as packs
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem.providers import ModelAPI, Embeddings
from jev_mem.sdk20_type_adapter import typed_select
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem.hybrid_native_capture_v1 import capture_hybrid_native, build_hybrid_pool, select_hybrid_arms

SCHEMA = 'planned-hybrid-retrieval-full1540-v1'
SOURCE_SCHEMA = 'planned-modules-selected-inputs-v1'
ARMS = ('hybrid_rank', 'hybrid_types', 'hybrid_types_no_relation')
OLD = ROOT / 'runs/sdk20_modules_v1'
CODE = tuple(dict.fromkeys((*legacy.CODE, 'tools/run_planned_retrieval_v1.py',
    'jev_mem/hybrid_native_capture_v1.py', 'jev_mem/qa_embedding_cache_pack_v1.py',
    'jev_mem/local_bm25_context_v2.py', 'tools/freeze_sdk20_full1540_v1.py')))


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(ref):
    if pointer(ref['path']) != ref:
        raise ValueError('Bound source bytes changed: ' + ref['path'])
    return Path(ref['path'])


def fact_projection(bank):
    # Embedding vectors and source-context metadata are not Reader value fields.
    # Keep the full bank file SHA separately, and never alter its contents.
    return [{k: f.get(k) for k in ('id', 'text', 'observation_date', 'user_id')} for f in bank['facts']]


def stable_selections(value):
    result = deepcopy(value)
    for selection in result.values():
        selection['trace'].pop('selector_cpu_seconds', None)
    return result


def verify_original_inventory(item):
    if sources.inventory(item['original_native']) != item['inventory']:
        raise ValueError('Original native inventory changed; source integrity gate failed')


class CacheOnlyEmbeddings:
    """SDK constructors get a noncredential placeholder; embed has no HTTP path."""
    embed = ModelAPI.embed
    keys = {'GEMINI_API_KEY': 'offline-cache-only-placeholder'}

    def __init__(self, entries, *, context=None):
        self.entries, self.context, self.events = entries, context or {}, []

    def scoped(self, **context):
        return type(self)(self.entries, context={**self.context, **context})

    def count_tokens(self, *args, **kwargs):
        raise RuntimeError('New countTokens prohibited in cache-only native replay')

    def request(self, provider, model, body, *, operation, stage):
        signature = {'provider': provider, 'model': model, 'operation': operation, 'body': body}
        key = request_digest(signature)
        row = self.entries.get(key)
        if row is None:
            raise RuntimeError('Missing frozen QUERY embedding; no network fallback: ' + key)
        raw_path = checked(row['target'])
        raw = raw_path.read_bytes()
        if packs.validate_raw(raw, key) != signature:
            raise ValueError('Exact embedding signature differs')
        self.events.append({'request_hash': key, 'stage': stage, 'context': self.context,
            'cache_hit': True, 'model_calls': 0, 'source': row['source'], 'target': row['target'],
            'usage_semantics': 'reused_response_no_new_model_usage', 'timestamp_utc': now()})
        return json.loads(raw)['response']


def prepare(run_dir):
    run = Path(run_dir).resolve()
    if run.parent != ROOT / 'runs' or run.exists():
        raise FileExistsError('A fresh direct runs namespace is required')
    lock = verify_lock(); old = legacy.manifest(OLD)
    qs, dates = sources.cohort()
    plan = packs.audit_sources()
    resource = resources(); inputs = {}
    for ci in range(10):
        folder = OLD / f'conv_{ci}'
        bank, oldhash = legacy.get_bank(OLD, ci, 'session', old)
        inventory = sources.inventory(bank['native_source'])
        if inventory != bank['source_inventory']:
            raise ValueError('Original terminal native inventory changed')
        refs = {'bank': pointer(folder / 'session/bank.json'), 'tags': {}, 'queries': {}}
        for f in bank['facts']:
            path = folder / 'tags' / (f['id'] + '.json'); tag = verify(path)
            if (tag['status'] != 'complete' or tag['fact_id'] != f['id'] or
                tag['conversation'] != ci or tag['bank_sha256'] != oldhash or
                tag['fact_text_sha256'] != digest(f['text'])):
                raise ValueError('Incomplete or mismatched original fact tags')
            refs['tags'][f['id']] = pointer(path)
        for q in qs[str(ci)]:
            entry = {}
            for name, rel in [('query_types', 'query_types'), ('retrieval', 'retrieval/session'),
                              ('old_typed', 'selection/fact_types_sdk20')]:
                path = folder / rel / (q['id'] + '.json'); value = verify(path)
                if value['status'] != 'complete' or any(value[k] != q[k2] for k, k2 in
                    [('query_id', 'id'), ('question', 'question'), ('reference', 'reference'), ('category', 'category')]):
                    raise ValueError('Original query identity/status differs')
                entry[name] = pointer(path)
            refs['queries'][q['id']] = entry
        inputs[str(ci)] = {'refs': refs, 'original_native': bank['native_source'], 'inventory': inventory}
    run.mkdir(parents=True)
    save_new(run / 'prepare_intent.json', {'schema': SCHEMA, 'model_calls': 0, 'created_utc': now()})
    for ci in range(10):
        item = inputs[str(ci)]; original = Path(item['original_native'])
        dest = run / f'conv_{ci}/native'
        for rel, expected in item['inventory'].items():
            src, dst = original / rel, dest / rel
            if not src.resolve().is_relative_to(original.resolve()) or not dst.resolve().is_relative_to(dest.resolve()):
                raise ValueError('Native inventory path escapes namespace')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists(): raise FileExistsError('Native copy target already exists')
            shutil.copyfile(src, dst)
            if sha(dst) != expected: raise ValueError('Native copied bytes differ')
        if sources.inventory(original) != item['inventory'] or sources.inventory(dest) != item['inventory']:
            raise ValueError('Source changed during native copy')
        item['native_copy'] = str(dest)
    cache = packs.create_pack(run, plan)
    _, official = verified_official_metric()
    value = {'schema': SCHEMA, 'created_utc': now(), 'arms': list(ARMS), 'questions': 1540,
        'full1540_lock': pointer(LOCK), 'original_manifest': pointer(OLD / 'run_manifest.json'),
        'source_inputs': inputs, 'cohort_sha256': digest(qs), 'reference_dates': dates,
        'resources': resource, 'embedding_cache_pack': cache,
        'official_metric_provenance': official, 'code_sha256': {p: sha(ROOT / p) for p in CODE},
        'baseline': lock['baseline'], 'replay': 'native20_and_original_typed20_order_required',
        'model_calls': 0, 'network_fallback': False, 'concurrent_conversations': 8}
    publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'run_dir': str(run), 'ready': True, 'facts': 2376, 'queries': 1540, 'model_calls': 0}


def manifest(run, *, conversation=None, full=False):
    run = Path(run).resolve(); m = read(run / 'run_manifest.json')
    verify_lock(); checked(m['full1540_lock']); checked(m['original_manifest'])
    if (m['schema'] != SCHEMA or m['arms'] != list(ARMS) or
        m['protocol_sha256'] != digest({k: v for k, v in m.items() if k != 'protocol_sha256'}) or
        m['code_sha256'] != {p: sha(ROOT / p) for p in CODE}):
        raise ValueError('Frozen new retrieval code/protocol changed')
    qs, dates = sources.cohort()
    if digest(qs) != m['cohort_sha256'] or dates != m['reference_dates']:
        raise ValueError('Full1540 cohort changed')
    validate_resources(m['resources']['snapshot'], m['resources']['files_sha256'])
    pack = packs.verify_pack(m['embedding_cache_pack'], conversation=conversation, check_sources=full)
    for ci in (range(10) if conversation is None else [conversation]):
        ref = m['source_inputs'][str(ci)]['refs']; checked(ref['bank'])
        for p in ref['tags'].values(): checked(p)
        for item in ref['queries'].values():
            for p in item.values(): checked(p)
    return m, pack


def retrieve_worker(run_dir, ci):
    run = Path(run_dir); m, pack = manifest(run, conversation=ci)
    folder = run / f'conv_{ci}'; inp = m['source_inputs'][str(ci)]
    durable_intent(folder / 'retrieval_execution_intent.json',
        {'created_utc': now(), 'pid': os.getpid(), 'conversation': ci, 'protocol_sha256': m['protocol_sha256']})
    refs = inp['refs']; bank = read(checked(refs['bank'])); facts = fact_projection(bank)
    tags = {fid: verify(checked(p))['result'] for fid, p in refs['tags'].items()}
    api = CacheOnlyEmbeddings({e['request_hash']: e for e in pack['entries'] if e['conversation'] == ci})
    dest = Path(inp['native_copy'])
    if sources.inventory(dest) != inp['inventory']:
        raise ValueError('New native copy changed before first search')
    mem = None; qs = sources.cohort()[0][str(ci)]; resource = m['resources']
    with local_bm25_context(resource['snapshot'], resource['files_sha256']):
        try:
            mem = native.open_memory(dest, api)
            from mem0.utils import spacy_models
            if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None or mem.vector_store._get_bm25_encoder() is None:
                raise RuntimeError('Native NLP/BM25 unavailable')
            for index, q in enumerate(qs):
                start = time.perf_counter(); cpu_start = time.process_time()
                previous = verify(checked(refs['queries'][q['id']]['retrieval']))
                qt = verify(checked(refs['queries'][q['id']]['query_types']))
                oldtyped = verify(checked(refs['queries'][q['id']]['old_typed']))
                if typed_select(previous['capture'], tags, qt['scores'])['selected_ids'] != oldtyped['selected_ids']:
                    raise ValueError('Original typed selection does not exactly replay')
                scoped = api.scoped(conversation=ci, query_id=q['id'], phase='retrieval')
                mem.embedding_model = Embeddings(scoped); audit = Warnings()
                logging.getLogger('mem0').addHandler(audit)
                captured = {}; failure = None
                try:
                    with capture_hybrid_native(mem, query=q['question'], filters={'user_id': bank['user_id']}, bank_facts=facts) as captured:
                        returned = mem.search(q['question'], top_k=20, filters={'user_id': bank['user_id']}, rerank=False)
                    if audit.items: raise RuntimeError('Native retrieval emitted warnings')
                    pool = build_hybrid_pool(captured, facts, bank_sha256=refs['bank']['sha256'],
                        expected_native20=previous['capture']['base20'], expected_native_capture=previous['capture'])
                    if [r['id'] for r in returned['results']] != previous['selected_ids']:
                        raise ValueError('Old native20 returned ID order differs')
                    choices = select_hybrid_arms(pool, tags, qt['scores'])
                    if set(choices) != set(ARMS): raise ValueError('Unexpected selector arms')
                    value = {'schema': SCHEMA, 'status': 'complete', 'conversation': ci, 'query_id': q['id'],
                        'protocol_sha256': m['protocol_sha256'], 'bank': refs['bank'], 'capture': captured,
                        'pool': pool, 'selections': choices, 'original_query_inputs': refs['queries'][q['id']],
                        'original_typed20_replay': True, 'cache_events': scoped.events, 'model_calls': 0,
                        'elapsed_seconds': time.perf_counter() - start, 'total_process_cpu_seconds': time.process_time() - cpu_start}
                    save_new(folder / 'selections' / (q['id'] + '.json'), value)
                except Exception as exc:
                    failure = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
                    save_new(folder / 'retrieval_failures' / (q['id'] + '.json'),
                        {'schema': SCHEMA, 'conversation': ci, 'query_id': q['id'], 'error': failure,
                         'capture': captured, 'warnings': audit.items, 'cache_events': scoped.events, 'model_calls': 0})
                    raise
                finally:
                    logging.getLogger('mem0').removeHandler(audit)
                publish_json(folder / 'retrieval_status.json', {'pid': os.getpid(), 'conversation': ci,
                    'updated_utc': now(), 'completed': index + 1, 'total': len(qs), 'last_query': q['id'],
                    'terminal': index + 1 == len(qs), 'model_calls': 0})
        finally:
            if mem is not None: native.close_memory(mem)
    verify_original_inventory(inp)
    save_new(folder / 'post_retrieval_inventory.json', {'inventory': sources.inventory(dest),
        'original_inventory_unchanged': True, 'original_inventory': inp['inventory'],
        'original_native': inp['original_native'], 'native_copy': str(dest), 'model_calls': 0})
    return {'conversation': ci, 'status': 'complete', 'queries': len(qs), 'model_calls': 0}


def retrieve(run_dir):
    run = Path(run_dir); m, _ = manifest(run, full=True)
    durable_intent(run / 'retrieval_execution_intent.json', {'pid': os.getpid(), 'created_utc': now(), 'protocol_sha256': m['protocol_sha256']})
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks = {pool.submit(retrieve_worker, str(run), ci): ci for ci in range(10)}
        for task in as_completed(tasks):
            try: row = task.result()
            except Exception as exc:
                row = {'conversation': tasks[task], 'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc)[:1500]}}
            results.append(row)
            publish_json(run / 'retrieval_status.json', {'pid': os.getpid(), 'updated_utc': now(), 'outcomes': results, 'terminal': False})
    result = {'schema': SCHEMA, 'protocol_sha256': m['protocol_sha256'], 'terminal': True, 'outcomes': results,
        'status': 'complete' if all(r['status'] == 'complete' for r in results) else 'failed', 'model_calls': 0, 'finished_utc': now()}
    save_new(run / 'retrieval_outcome.json', result)
    return result


def _jobs(run, m, *, replay=True):
    qs, dates = sources.cohort(); jobs = []
    for ci in range(10):
        refs = m['source_inputs'][str(ci)]['refs']; bank = read(checked(refs['bank']))
        inp = m['source_inputs'][str(ci)]; verify_original_inventory(inp)
        post = verify(Path(run) / f'conv_{ci}/post_retrieval_inventory.json')
        if (post.get('original_inventory_unchanged') is not True or post['original_inventory'] != inp['inventory'] or
            post['native_copy'] != inp['native_copy'] or post['original_native'] != inp['original_native'] or
            post['inventory'] != sources.inventory(inp['native_copy'])):
            raise ValueError('Post-retrieval native source/copy inventory differs')
        facts = fact_projection(bank)
        byid = {f['id']: f for f in bank['facts']}
        tags = {fid: verify(checked(p))['result'] for fid, p in refs['tags'].items()}
        for q in qs[str(ci)]:
            path = Path(run) / f"conv_{ci}/selections/{q['id']}.json"; selected = verify(path)
            if (selected['status'] != 'complete' or selected['query_id'] != q['id'] or selected['conversation'] != ci or
                selected['protocol_sha256'] != m['protocol_sha256'] or selected['bank'] != refs['bank']):
                raise ValueError('Selection identity changed')
            if replay:
                previous = verify(checked(refs['queries'][q['id']]['retrieval']))
                qt = verify(checked(refs['queries'][q['id']]['query_types']))
                expected = build_hybrid_pool(selected['capture'], facts, bank_sha256=refs['bank']['sha256'],
                    expected_native20=previous['capture']['base20'], expected_native_capture=previous['capture'])
                if (selected['pool'] != expected or
                    stable_selections(selected['selections']) != stable_selections(select_hybrid_arms(expected, tags, qt['scores']))):
                    raise ValueError('Frozen selector results no longer reproduce')
            for arm in ARMS:
                ids = selected['selections'][arm]['selected_ids']
                if len(ids) != 20 or len(set(ids)) != 20 or not set(ids) <= set(byid):
                    raise ValueError('Exactly20 original bank IDs required')
                jobs.append({'job_id': arm + '/' + q['id'], 'conversation': ci, 'method': arm, 'status': 'ready',
                    'query_id': q['id'], 'question': q['question'], 'reference': q['reference'], 'category': q['category'],
                    'reference_date': dates[str(ci)], 'selected_ids': ids,
                    'selected_facts': [{k: byid[fid].get(k) for k in ('id', 'text', 'observation_date')} for fid in ids],
                    'lineage': {'selection': pointer(path), 'bank': refs['bank'], 'pool_sha256': selected['pool']['pool_sha256']}})
    return jobs


def seal(run_dir):
    run = Path(run_dir); m, _ = manifest(run)
    outcome = verify(run / 'retrieval_outcome.json')
    if (outcome['status'] != 'complete' or outcome['protocol_sha256'] != m['protocol_sha256'] or
        len(outcome['outcomes']) != 10 or {r['conversation'] for r in outcome['outcomes']} != set(range(10))):
        raise ValueError('All ten conversations must pass replay before QA')
    jobs = _jobs(run, m)
    value = {'schema': SOURCE_SCHEMA, 'arms': list(ARMS), 'jobs': jobs,
        'retrieval_run': str(run.resolve()), 'retrieval_manifest': pointer(run / 'run_manifest.json'),
        'retrieval_outcome': pointer(run / 'retrieval_outcome.json'),
        'full1540_lock': m['full1540_lock'], 'official_metric_provenance': m['official_metric_provenance'],
        'precomputation_cost_lineage': {'original_manifest': m['original_manifest'], 'model_calls_in_new_retrieval': 0,
            'note': 'Original writer, embeddings and JEV labels reused; not a free cold start.'}, 'model_calls': 0}
    save_new(run / 'source_jobs.json', value)
    return {'source': str(run / 'source_jobs.json'), 'jobs': len(jobs), 'model_calls': 0}


def validate_source(path):
    source = verify(path)
    if source.get('schema') != SOURCE_SCHEMA or source.get('arms') != list(ARMS):
        raise ValueError('Only the frozen hybrid three-arm source is currently admitted')
    run = Path(source['retrieval_run']); checked(source['retrieval_manifest']); checked(source['retrieval_outcome'])
    m, _ = manifest(run)
    if (source['full1540_lock'] != m['full1540_lock'] or source['official_metric_provenance'] != m['official_metric_provenance'] or
        source['jobs'] != _jobs(run, m)):
        raise ValueError('Source inputs differ from recomputed sealed selection')
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'check', 'retrieve', 'seal'])
    p.add_argument('--run-dir', type=Path, required=True); a = p.parse_args()
    if a.stage == 'prepare': result = prepare(a.run_dir)
    elif a.stage == 'check': m, _ = manifest(a.run_dir); result = {'valid': True, 'protocol_sha256': m['protocol_sha256'], 'model_calls': 0}
    elif a.stage == 'retrieve': result = retrieve(a.run_dir)
    else: result = seal(a.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
