"""Fresh session-only Gemini 2.5 Writer; prepare/check are strictly offline.

The native write_sessions/open_memory functions retain their original code
objects in private namespaces. Original session partitions, dates, Memory.add,
extraction parser, persistence and source scopes are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import FunctionType
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_writer_v1 as frozen
from tools.run_sdk20_modules import digest, verify, sha, now
from tools.run_pilot400_experiments_v2 import publish_json
from jev_mem.data import load_data, session_envelopes, SOURCE_HASH
from jev_mem.gemini25_ablation_provider_v1 import (
    Gemini25API, Gemini25ExtractionLLM, MODEL, WRITER_MAX_OUTPUT,
    THINKING_BUDGET, generation_config,
)
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem.paced_embedding_api import DEFAULT_GATE

SCHEMA = 'gemini25-ablation-session-writer-v1'
ARMS = ('session',)
CURRENT_DATE = frozen.CURRENT_DATE
DEFAULT_RUN = ROOT/'runs/gemini25_ablation_writer_v1_20261007'
grouped, sparse, native = frozen.grouped, frozen.sparse, frozen.native
Embeddings = frozen.Embeddings
resources = frozen.resources
CODE = tuple(dict.fromkeys((*frozen.CODE,
    'tools/run_gemini25_ablation_writer_v1.py',
    'tools/audit_gemini25_ablation_banks_v1.py',
    'jev_mem/gemini25_ablation_provider_v1.py',
    'tools/audit_gemini38_banks_v1.py',
    'tools/publish_gemini38_bank_review_v1.py',
    'tools/audit_regroup_refinement_banks_v1.py',
    'tools/mem0_regroup_firstpass_review_v2.py',
    'jev_mem/source_append_r05.py')))


def clone(function, **updates):
    value = FunctionType(function.__code__, {**function.__globals__, **updates},
                         function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return value


open_memory = clone(frozen.open_memory, MODEL=MODEL, WRITER_MAX_OUTPUT=WRITER_MAX_OUTPUT,
                    Gemini38ExtractionLLM=Gemini25ExtractionLLM)
write_sessions = clone(frozen.write_sessions, SCHEMA=SCHEMA, CURRENT_DATE=CURRENT_DATE,
                       Gemini38API=Gemini25API, Gemini38ExtractionLLM=Gemini25ExtractionLLM,
                       open_memory=open_memory)


def codes():
    return {p: sha(ROOT/p) for p in CODE}


def policy():
    return {'max_processes': 8, 'transport_retries': 2, 'external_cache_roots': [],
            'new_empty_banks': True, 'failed_or_unknown_write_reentry': False,
            'development_conversations': [0, 1],
            'expansion_requires_bound_development_review': True,
            'embedding_gate': 'shared3sec_physical_start_429cooldown_at_least30sec'}


def prepare(run):
    run = Path(run).resolve()
    if run.parent != (ROOT/'runs').resolve() or not run.name.startswith('gemini25_ablation_writer_v1') or run.exists():
        raise FileExistsError('Fresh direct runs/gemini25_ablation_writer_v1* namespace required')
    if sha(ROOT/'data/locomo10.json') != SOURCE_HASH:
        raise ValueError('Frozen dataset changed')
    data = load_data()
    envelopes = [list(session_envelopes(item)) for item in data]
    if len(envelopes) != 10 or sum(map(len, envelopes)) != 272:
        raise ValueError('Exactly ten conversations and 272 original sessions required')
    resource, code, dependencies = resources(), codes(), sparse.dependency_versions()
    source = grouped.pointer(ROOT/'data/locomo10.json')
    inputs = []
    for ci, envs in enumerate(envelopes):
        path = run/f'conv_{ci}/session/group_inputs.json'
        grouped.save_new(path, {'schema': SCHEMA, 'conversation': ci, 'arm': 'session',
                               'envelopes': envs, 'source': source, 'dataset_sha256': SOURCE_HASH})
        inputs.append({'conversation': ci, 'arm': 'session', 'input': grouped.pointer(path)})
    value = {'schema': SCHEMA, 'created_utc': now(), 'arms': list(ARMS),
             'dataset_sha256': SOURCE_HASH, 'inputs': inputs, 'code_sha256': code,
             'dependencies': dependencies, 'resources': resource, 'current_date': CURRENT_DATE,
             'models': {'writer': MODEL, 'embedding': 'gemini-embedding-001', 'dimensions': 768},
             'generation_config': generation_config(WRITER_MAX_OUTPUT, True),
             'thinking_budget': THINKING_BUDGET, 'policy': policy(),
             'gate_path': str(DEFAULT_GATE.resolve()), 'model_calls_at_prepare': 0,
             'shared_qa_arms': ['baseline', 'r03_original', 'original_multi_source_closure'],
             'writer_semantics': 'Exact original session write_sessions/open_memory code objects; native Memory.add and ADDITIVE_EXTRACTION_PROMPT/parser unchanged',
             'local_user_id': 'gemini38_c{conversation}_session is an inherited opaque local scope; every native store and API cache is new',
             'source_support': 'Each fact retains input_scope; group_inputs bind exact source turn IDs/messages/dates; exposure is not verified support'}
    publish_json(run/'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'prepared': str(run), 'banks': 10, 'session_count': 272, 'model_calls': 0}


def manifest(run):
    run = Path(run).resolve()
    m = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    expected_config = generation_config(WRITER_MAX_OUTPUT, True)
    if (m.get('schema') != SCHEMA or m.get('protocol_sha256') != digest({k:v for k,v in m.items() if k != 'protocol_sha256'})
        or m.get('arms') != list(ARMS) or m.get('models') != {'writer': MODEL, 'embedding': 'gemini-embedding-001', 'dimensions': 768}
        or m.get('generation_config') != expected_config or m.get('current_date') != CURRENT_DATE
        or m.get('thinking_budget') != THINKING_BUDGET or m.get('policy') != policy()
        or m.get('gate_path') != str(DEFAULT_GATE.resolve())):
        raise ValueError('Session-only Writer protocol/config changed')
    if sha(ROOT/'data/locomo10.json') != SOURCE_HASH or m['dataset_sha256'] != SOURCE_HASH:
        raise ValueError('Dataset changed')
    if m['dependencies'] != sparse.dependency_versions() or m['code_sha256'] != codes():
        raise ValueError('Pinned dependencies/code changed after preparation')
    if (len(m['inputs']) != 10 or
        {(r['conversation'],r['arm']) for r in m['inputs']} != {(ci,'session') for ci in range(10)}):
        raise ValueError('Ten unique session bank inputs required')
    data = load_data()
    source = grouped.pointer(ROOT/'data/locomo10.json')
    for row in m['inputs']:
        ci = row['conversation']
        expected_path = run/f'conv_{ci}/session/group_inputs.json'
        if row['input'] != grouped.pointer(expected_path):
            raise ValueError('Bank input escaped current run')
        value = verify(expected_path)
        expected = {'schema': SCHEMA, 'conversation': ci, 'arm': 'session',
                    'envelopes': list(session_envelopes(data[ci])), 'source': source, 'dataset_sha256': SOURCE_HASH}
        if {k:v for k,v in value.items() if k != 'content_sha256'} != expected:
            raise ValueError('Original session/source/date rendering changed')
    if sum(len(list(session_envelopes(item))) for item in data) != 272:
        raise ValueError('Original session count changed')
    if m['resources'] != resources():
        raise ValueError('Pinned local BM25 resource binding changed')
    validate_resources(m['resources']['snapshot'], m['resources']['files_sha256'])
    return m


def development_review(run, m):
    from tools.audit_gemini25_ablation_banks_v1 import validate_review
    return validate_review(run, m, development=True)


def worker(run, ci, arm='session'):
    run = Path(run).resolve()
    m = manifest(run)
    if type(ci) is not int or ci not in range(10) or arm != 'session':
        raise ValueError('Only original session banks are admitted')
    if ci > 1:
        development_review(run, m)
    folder = run/f'conv_{ci}'/arm
    outcome = folder/'write_outcome.json'
    if outcome.exists():
        old = verify(outcome)
        if old.get('status') != 'complete' or old.get('protocol_sha256') != m['protocol_sha256']:
            raise RuntimeError('Failed/unknown Writer outcome retained; no automatic rebuild')
        bank = grouped.check_pointer(old['bank'])
        if not bank.get('complete') or (bank['conversation'],bank['arm'],bank['protocol_sha256']) != (ci,arm,m['protocol_sha256']):
            raise ValueError('Completed bank identity differs')
        return old
    if any((folder/n).exists() for n in ('write_intent.json', 'native', 'write_status.json')):
        raise RuntimeError('Unresolved Writer store/intent retained; no redispatch')
    with (folder/'write_intent.json').open('x', encoding='utf-8') as handle:
        json.dump({'protocol_sha256':m['protocol_sha256'], 'conversation':ci, 'arm':arm,
                   'pid':os.getpid(), 'created_utc':now()}, handle)
        handle.flush(); os.fsync(handle.fileno())
    try:
        with local_bm25_context(m['resources']['snapshot'],m['resources']['files_sha256']):
            state = write_sessions(run,ci,m)
    except Exception as exc:
        state = {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,
                 'status':'failed','pid':os.getpid(),'finished_utc':now(),
                 'error':{'type':type(exc).__name__,'message':'Local Writer failure; retained intent/store requires audit'},
                 'store_reentry_forbidden':True}
        status_path = folder/'write_status.json'
        if status_path.exists():
            state['retained_prior_status_sha256'] = sha(status_path)
            state['retained_prior_status_value'] = json.loads(status_path.read_text(encoding='utf-8'))
        publish_json(status_path,state)
    result = {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,
              'status':state['status'],'write_status':grouped.pointer(folder/'write_status.json'),
              'finished_utc':now(),'external_cache_roots':[]}
    if state['status'] == 'complete':
        result['bank'] = state['bank']
    else:
        result['error'] = state.get('error')
    grouped.save_new(outcome,result)
    return result


# Reuse the original bounded process controller without touching its globals.
run_write = clone(frozen.run_write, SCHEMA=SCHEMA, ARMS=ARMS,
                  manifest=manifest, development_review=development_review, worker=worker)


def source_support_inputs(run, ci):
    """Exact new-bank scope and source mapping for downstream pure SER planners."""
    from jev_mem.source_append_r05 import source_turns
    run = Path(run).resolve()
    m = manifest(run)
    if type(ci) is not int or ci not in range(10):
        raise ValueError('Invalid conversation')
    folder = run/f'conv_{ci}/session'
    bank = verify(folder/'bank.json')
    if bank.get('complete') is not True or (bank['conversation'],bank['arm'],bank['protocol_sha256']) != (ci,'session',m['protocol_sha256']):
        raise ValueError('Complete current session bank required')
    ref = grouped.pointer(folder/'group_inputs.json')
    if bank['group_inputs'] != ref:
        raise ValueError('Fact/source input binding changed')
    inp = verify(ref['path'])
    turns = source_turns(inp,ref)
    ids = {t['source_turn_id'] for t in turns}
    rawfacts = {f['id']:f for f in bank['facts']}
    if len(rawfacts) != len(bank['facts']):
        raise ValueError('Duplicate bank IDs')
    for f in rawfacts.values():
        scope = f.get('input_scope')
        if not isinstance(scope,dict) or not set(scope.get('source_turn_ids',[])) <= ids:
            raise ValueError('Fact current-source scope differs')
    return {'bank':grouped.pointer(folder/'bank.json'),'rawfacts':rawfacts,
            'group_inputs':ref,'turns':turns,'input_scope_is_verified_support':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','write'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--conversations',nargs='+',type=int,default=[0,1])
    args = parser.parse_args()
    if args.stage == 'prepare': result = prepare(args.run_dir)
    elif args.stage == 'check': result = {'valid':bool(manifest(args.run_dir)),'model_calls':0}
    else: result = run_write(args.run_dir,args.conversations,list(ARMS))
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
