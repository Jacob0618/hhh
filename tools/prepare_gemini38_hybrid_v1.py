"""Fresh-bank seven-arm retrieval seam; prepare/check/seal never call a model.

Explicit tag and retrieve can dispatch JEV fact/query scoring. Retrieval uses the sealed exact QUERY pack and
new immutable store copies. No old fact tags, UUID selections, or bank exceptions
are admitted. Conversations, not arms, own at most eight worker processes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_writer_v1 as writer
from tools.run_sdk20_modules import digest, sha, verify, save_new, now, Warnings
from tools.run_pilot400_experiments_v2 import publish_json
from tools.mem0_regroup_plain_qa_sources_v1 import cohort, inventory
from tools.summarize_pilot400 import verified_official_metric
from tools.freeze_sdk20_full1540_v1 import LOCK, verify_lock
from jev_mem.data import SOURCE_HASH, load_data, session_envelopes
from jev_mem.selection import tag_fact
from jev_mem.pilot400 import query_type_scores
from jev_mem.sdk20_type_adapter import capture_native_pool, _complete_fact_scores, _complete_query_scores
from jev_mem.hybrid_native_capture_v1 import capture_hybrid_native, build_hybrid_pool, select_hybrid_arms
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem import qa_embedding_cache_pack_v1 as packs
from tools.run_planned_retrieval_v1 import CacheOnlyEmbeddings, fact_projection, stable_selections

SCHEMA = 'gemini38-fresh-seven-arm-retrieval-v1'
SOURCE_SCHEMA = 'gemini38-seven-arm-selected-inputs-v1'
GATE_SCHEMA = 'gemini38-complete-bank-admission-v1'
NEW_CONFIG = ROOT/'configs/gemini38_seven_arm_full1540_v1.json'
NEW_CONFIG_SHA256 = '9efe4652267ee4c0ecab08503dd48562ed40d4993ac6500c704a41adea721498'
ARMS = ('baseline', 'source_local_context', 'hybrid_rank', 'hybrid_types',
        'hybrid_types_no_relation', 'boundary_multiscale', 'boundary_neighbor')
BANK_ARMS = writer.ARMS
H_ARMS = ('hybrid_rank', 'hybrid_types', 'hybrid_types_no_relation')
METHOD_BANK = {'baseline': 'session', **{a: 'session' for a in H_ARMS},
               **{a: a for a in BANK_ARMS if a != 'session'}}
CHECKS = ('source_dates', 'generation_chain', 'fact_dense', 'fact_bm25', 'entity_dense', 'entity_links')
CODE = tuple(dict.fromkeys((*writer.CODE, 'tools/prepare_gemini38_hybrid_v1.py',
    'tools/run_planned_retrieval_v1.py', 'jev_mem/hybrid_native_capture_v1.py',
    'jev_mem/sdk20_type_adapter.py', 'jev_mem/selection.py', 'jev_mem/routing.py',
    'jev_mem/pilot400.py', 'jev_mem/qa_embedding_cache_pack_v1.py',
    'tools/freeze_sdk20_full1540_v1.py', 'tools/summarize_pilot400.py',
    'tools/publish_gemini38_bank_review_v1.py','tools/audit_gemini38_banks_v1.py')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or pointer(ref['path']) != ref:
        raise ValueError('Source path/hash binding changed')
    return Path(ref['path'])


def _inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('Artifact outside new parent namespace')
    return path


def _fact_bank(bank, ci, arm, parent_protocol):
    schema = writer.SCHEMA if arm == 'session' else writer.grouped.SCHEMA
    if (bank.get('schema') != schema or (bank.get('conversation'), bank.get('arm'), bank.get('protocol_sha256')) != (ci, arm, parent_protocol)
        or bank.get('complete') is not True or bank.get('source_dataset_sha256') != SOURCE_HASH
        or not isinstance(bank.get('user_id'), str) or not bank['user_id']):
        raise ValueError('New bank identity/protocol/terminal state differs')
    facts = bank.get('facts', [])
    if len(facts) < 20 or len({f.get('id') for f in facts}) != len(facts):
        raise ValueError('Complete bank requires at least20 unique facts')
    for f in facts:
        if (not isinstance(f.get('id'), str) or not f['id'] or not isinstance(f.get('text'), str)
            or any(c in f['id'] for c in '/\\:') or f['id'] in ('.','..')
            or not f['text'].strip() or f.get('user_id') != bank['user_id']):
            raise ValueError('Bank fact text/ID/scope invalid')
    return facts


def verify_new_config(run, parent=None):
    """Historical lock is supplemental; the new all-stage protocol is mandatory."""
    run=Path(run).resolve();parent=parent or writer.manifest(run)
    if sha(NEW_CONFIG)!=NEW_CONFIG_SHA256:
        raise ValueError('Fresh seven-arm all-stage config bytes changed')
    config=verify(NEW_CONFIG)
    if (config.get('schema')!='gemini38-seven-arm-full1540-v1' or config.get('arms')!=list(ARMS)
        or config.get('repetitions')!=1 or config.get('questions_per_arm')!=1540 or config.get('total_slots')!=10780
        or config.get('physical_banks')!=40 or config.get('new_writer_bank_required') is not True
        or config.get('selected_facts')!=20 or config.get('qa_word_cap') is not None
        or config.get('writer_protocol_sha256')!=parent['protocol_sha256']
        or any(config.get('models',{}).get(k)!='gemini-3.8-flash'for k in ('writer','reader','judge'))
        or config['models'].get('jev')!='jev-1.13.0' or config['models'].get('embedding')!='gemini-embedding-001'
        or config['models'].get('embedding_dimensions')!=768):
        raise ValueError('Fresh all-stage config lineage/cohort/model protocol differs')
    if checked(config['writer_manifest'])!=(run/'run_manifest.json').resolve() or checked(config['historical_lock'])!=Path(LOCK).resolve():
        raise ValueError('Fresh config is bound to another Writer run/old lock')
    return config


def admit_gate(run, gate_path=None):
    """All forty banks require NEW source/date/index findings; no inherited waiver."""
    run = Path(run).resolve(); parent = writer.manifest(run);verify_new_config(run,parent)
    gp = _inside(gate_path or run/'all_bank_review.json', run); gate = verify(gp)
    if (gate.get('schema') != GATE_SCHEMA or gate.get('status') != 'passed'
        or gate.get('protocol_sha256') != parent['protocol_sha256'] or gate.get('exceptions') != []):
        raise ValueError('New forty-bank explicit review without inherited exceptions required')
    if checked(gate['writer_manifest']) != (run/'run_manifest.json').resolve():
        raise ValueError('Admission is for another Writer manifest')
    semantic=verify(_inside(checked(gate['semantic_review']), run))
    rows = gate.get('banks', [])
    expected = {(ci, a) for ci in range(10) for a in BANK_ARMS}
    if len(rows) != 40 or {(r.get('conversation'), r.get('arm')) for r in rows} != expected:
        raise ValueError('All forty unique bank reviews required')
    if (semantic.get('schema')!='gemini38-writer-source-semantic-review-v1'
        or semantic.get('status')!='reviewed_with_limitations' or semantic.get('qa_gold_used') is not False
        or semantic.get('protocol_sha256')!=parent['protocol_sha256']
        or len(semantic.get('banks',[]))!=40 or {(r['conversation'],r['arm'])for r in semantic['banks']}!=expected):
        raise ValueError('All forty source-only semantic reviews required')
    semantic_rows={(r['conversation'],r['arm']):r for r in semantic['banks']}
    audit_rows={}
    for kind in ('source_date_audits','index_audits'):
        refs=gate.get(kind)
        if not isinstance(refs,list)or not refs:raise ValueError('Bound strict source/index audit missing')
        for ref in refs:
            report=verify(_inside(checked(ref),run))
            if (report.get('schema')!='gemini38-terminal-source-index-audit-v1' or report.get('strict_passed') is not True
                or report.get('exceptions')!=[] or report.get('protocol_sha256')!=parent['protocol_sha256']
                or report.get('writer_manifest')!=gate['writer_manifest'] or report.get('conversations')!=list(range(10))
                or len(report.get('banks',[]))!=40 or {(r['conversation'],r['arm'])for r in report['banks']}!=expected):
                raise ValueError('Full new bound strict audit required')
            for src,expected_sha in report['source_file_hashes'].items():
                if sha(src)!=expected_sha:raise ValueError('Audited source file changed')
            audit_rows[ref['path']]={(r['conversation'],r['arm']):r for r in report['banks']}
    if semantic.get('structural_audit') not in gate['source_date_audits']:
        raise ValueError('Semantic review is bound to another strict structural audit')
    admitted = {}
    for row in rows:
        ci, arm = row['conversation'], row['arm']; folder = run/f'conv_{ci}'/arm
        if row.get('exceptions') != [] or any(row.get('checks', {}).get(k) is not True for k in CHECKS):
            raise ValueError('Every new bank must pass all source/date/index checks')
        for field, name in [('bank', 'bank.json'), ('write_status', 'write_status.json'), ('outcome', 'write_outcome.json')]:
            if checked(row[field]) != (folder/name).resolve():
                raise ValueError('Review references another bank lineage')
        bank = verify(row['bank']['path']); state = read(row['write_status']['path']); out = verify(row['outcome']['path'])
        _fact_bank(bank, ci, arm, parent['protocol_sha256'])
        sem=semantic_rows[(ci,arm)]
        if sem.get('bank')!=row['bank'] or not sem.get('notes') or sem.get('blocking_implementation_error') is not False:
            raise ValueError('Bank semantic review missing or blocking')
        if any((v.get('conversation'), v.get('arm'), v.get('protocol_sha256'), v.get('status')) !=
               (ci, arm, parent['protocol_sha256'], 'complete') for v in (state, out)):
            raise ValueError('Writer status/outcome not terminal for current protocol')
        if state.get('bank') != row['bank'] or out.get('bank') != row['bank'] or out.get('write_status') != row['write_status']:
            raise ValueError('Writer chain bank/status pointers differ')
        input_ref = next(r['input'] for r in parent['inputs'] if (r['conversation'], r['arm']) == (ci, arm))
        inp = verify(checked(input_ref)); count = len(inp['envelopes'])
        if bank.get('group_inputs') != input_ref or len(bank.get('journal_files', [])) != count:
            raise ValueError('Bank input/journal chain incomplete')
        if state.get('completed_groups') != count or state.get('expected_groups') != count or state.get('facts') != len(bank['facts']):
            raise ValueError('Writer group/fact totals incomplete')
        for ref in bank['journal_files']:
            checked(ref)
        for kind in ('source_date_audit', 'index_audit'):
            _inside(checked(row[kind]), run)
            if row[kind]not in gate[kind+'s']:
                raise ValueError('Bank review is outside declared strict audits')
            audited=audit_rows[row[kind]['path']][(ci,arm)]
            if (audited.get('bank')!=row['bank'] or audited.get('native_inventory')!=row.get('native_inventory')
                or audited.get('native_dir')!=row.get('native_dir') or audited.get('strict_passed')is not True
                or audited.get('source_generation_persistence_passed')is not True or audited.get('orphan_fact_ids')):
                raise ValueError('Current bank has no matching strict source/index findings')
            required={x['journal']['path']for x in audited.get('semantic_review_targets',[])}
            required.update(audited['journals'][i]['journal']['path']for i in (0,-1))
            journal_refs={ref['path']:ref for ref in bank['journal_files']}
            if not required<={ref['path']for ref in sem.get('reviewed_journals',[])}:
                raise ValueError('Mandatory source/date/empty/history review targets missing')
            if any(journal_refs.get(ref['path'])!=ref for ref in sem['reviewed_journals']):
                raise ValueError('Semantic review cites another bank journal')
        native = folder/'native'
        if Path(bank['native_dir']).resolve() != native.resolve() or inventory(native) != row.get('native_inventory'):
            raise ValueError('Reviewed new native inventory changed')
        admitted[(ci, arm)] = {'review': deepcopy(row), 'bank': bank, 'native': native}
    return parent, gate, admitted


def prepare(run, gate_path=None):
    run = Path(run).resolve(); dest = run/'retrieval'
    if dest.exists():
        raise FileExistsError('Fresh retrieval subnamespace required')
    parent, gate, sources = admit_gate(run, gate_path)
    verify_lock(); qs, dates = cohort(); resource = parent['resources']
    validate_resources(resource['snapshot'], resource['files_sha256'])
    cache_plan = packs.audit_sources()  # Public exact QUERY embeddings only, no native-bank inheritance.
    dest.mkdir(parents=True)
    save_new(dest/'prepare_intent.json', {'schema': SCHEMA, 'created_utc': now(), 'model_calls': 0})
    copied = []
    for ci in range(10):
        for arm in BANK_ARMS:
            source = sources[(ci, arm)]; origin = source['native'].resolve(); inv = source['review']['native_inventory']
            target = dest/f'conv_{ci}'/arm/'native'
            target.mkdir(parents=True, exist_ok=False)
            for rel, expected in inv.items():
                src = _inside(origin/rel, origin); dst = _inside(target/rel, target)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
                if sha(dst) != expected:
                    raise ValueError('Copied native bytes differ')
            if inventory(origin) != inv or inventory(target) != inv:
                raise ValueError('Native source/copy inventory changed')
            copied.append({'conversation': ci, 'arm': arm, 'bank': source['review']['bank'],
                           'native_source': str(origin), 'native_copy': str(target), 'native_inventory': inv})
    pack = packs.create_pack(dest, cache_plan)
    _, official = verified_official_metric()
    m = {'schema': SCHEMA, 'created_utc': now(), 'arms': list(ARMS), 'bank_arms': list(BANK_ARMS),
         'parent_manifest': pointer(run/'run_manifest.json'), 'parent_protocol_sha256': parent['protocol_sha256'],
         'new_execution_config':pointer(NEW_CONFIG),
         'source_review': pointer(gate_path or run/'all_bank_review.json'), 'full1540_lock': pointer(LOCK),
         'cohort_sha256': digest(qs), 'reference_dates': dates, 'questions': 1540,
         'banks': copied, 'resources': resource, 'embedding_cache_pack': pack,
         'official_metric_provenance': official, 'code_sha256': {p: sha(ROOT/p) for p in CODE},
         'models': {'writer': 'gemini-3.8-flash', 'jev': 'jev-1.13.0', 'embedding': 'gemini-embedding-001', 'dimensions': 768},
         'policy': {'max_processes': 8, 'native_top_k': 20, 'native_threshold': .1, 'rerank': False,
                    'hybrid_native80_lexical20_pool80': True, 'rrf_k': 60, 'fused_weight': 2, 'type_weight': 1,
                    'new_fact_labels': True, 'historical_fact_tag_read': False, 'native_gate': 'newbank_current_N0_samecapture',
                    'retrieval_gemini_generation_calls': 0, 'missing_embedding_blocks': True}}
    publish_json(dest/'run_manifest.json', {**m, 'protocol_sha256': digest(m)})
    return {'prepared': str(dest), 'banks': 40, 'questions': 1540, 'model_calls': 0}


def manifest(run, conversation=None, full=False):
    run = Path(run).resolve(); m = read(run/'retrieval/run_manifest.json')
    if (m.get('schema') != SCHEMA or m.get('arms') != list(ARMS)
        or m.get('protocol_sha256') != digest({k:v for k,v in m.items() if k != 'protocol_sha256'})
        or m.get('code_sha256') != {p: sha(ROOT/p) for p in CODE}):
        raise ValueError('Retrieval source code/protocol changed')
    parent = writer.manifest(run)
    verify_new_config(run,parent)
    if m.get('new_execution_config')!=pointer(NEW_CONFIG):raise ValueError('New execution config is missing/different')
    checked(m['parent_manifest']); checked(m['source_review']); checked(m['full1540_lock']); verify_lock()
    if m['parent_protocol_sha256'] != parent['protocol_sha256']:
        raise ValueError('Another Writer protocol')
    qs, dates = cohort()
    if digest(qs) != m['cohort_sha256'] or dates != m['reference_dates']:
        raise ValueError('Full query/category/gold/date cohort changed')
    validate_resources(m['resources']['snapshot'], m['resources']['files_sha256'])
    pack = packs.verify_pack(m['embedding_cache_pack'], conversation=conversation, check_sources=full)
    expected = {(ci,a) for ci in range(10) for a in BANK_ARMS}
    if len(m['banks']) != 40 or {(r['conversation'],r['arm']) for r in m['banks']} != expected:
        raise ValueError('Forty copied bank bindings required')
    for item in m['banks']:
        if conversation is None or item['conversation'] == conversation:
            bank = verify(checked(item['bank'])); _fact_bank(bank, item['conversation'], item['arm'], parent['protocol_sha256'])
            if inventory(item['native_source']) != item['native_inventory']:
                raise ValueError('Immutable Writer native source changed')
    if full:
        admit_gate(run, m['source_review']['path'])
    return m, pack


def session_contexts(ci):
    return {s['session_id']:'\n'.join(t['content']for t in s['messages'])
            for s in session_envelopes(load_data()[ci])}


def tag_worker(run, ci, api_factory=None):
    run = Path(run).resolve(); m, _ = manifest(run, ci); folder = run/'retrieval'/f'conv_{ci}'
    item = next(r for r in m['banks'] if (r['conversation'], r['arm']) == (ci, 'session'))
    bank = verify(checked(item['bank'])); contexts=session_contexts(ci)
    if api_factory is None:
        from jev_mem.gemini38_provider_v1 import Gemini38API
        api = Gemini38API(folder/'tag_api', retries=2, context={'architecture': SCHEMA, 'conversation': ci, 'phase': 'new_fact_typing'})
    else:
        api = api_factory(folder/'tag_api', ci, m)
    def one(fact):
        path = folder/'tags'/(fact['id']+'.json')
        session = fact.get('input_scope', {}).get('session_id')
        if session not in contexts:
            raise ValueError('New session fact lacks original source session scope')
        context=contexts[session]
        if path.exists():
            row = verify(path)
            if (row.get('status') != 'complete' or row.get('bank') != item['bank']
                or row.get('fact_text_sha256') != digest(fact['text']) or row.get('protocol_sha256') != m['protocol_sha256']
                or row.get('source_context_sha256')!=digest(context)):
                raise RuntimeError('Failed/stale fact tag retained; no implicit replay')
            if row['result'].get('fact_id')!=fact['id']:raise ValueError('Fact tag result uses another fact ID')
            _complete_fact_scores(row['result'], fact['id']); return row
        scoped = api.scoped(fact_id=fact['id'])
        intent = folder/'tag_intents'/(fact['id']+'.json')
        save_new(intent, {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'], 'created_utc':now(),
                          'fact_id':fact['id'],'bank':item['bank'],'pid':os.getpid()})
        row = {'schema': SCHEMA, 'protocol_sha256': m['protocol_sha256'], 'conversation': ci,
               'fact_id': fact['id'], 'bank': item['bank'], 'fact_text_sha256': digest(fact['text']),
               'source_context_sha256': digest(context), 'source_scope': 'whole_original_session_unverified_input_scope'}
        try:
            result = tag_fact({**fact, 'source_context': context}, scoped)
            if result.get('fact_id')!=fact['id']:raise ValueError('Fact typing result uses another fact ID')
            _complete_fact_scores(result, fact['id'])
            row.update(status='complete',result=result,api_events=list(scoped.events))
        except Exception as ex:
            row.update(status='failed',error_type=type(ex).__name__,
                       error='Fact typing failed; retained intent/cache requires audit',api_events=list(scoped.events))
            save_new(path,row)
            return row
        save_new(path,row); return row
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(one, f) for f in bank['facts']]):
            rows.append(fut.result())
            publish_json(folder/'tag_status.json', {'schema': SCHEMA, 'completed':len(rows), 'total':len(bank['facts']),
                         'failed':sum(r['status']!='complete'for r in rows),
                         'terminal':len(rows)==len(bank['facts']), 'protocol_sha256':m['protocol_sha256']})
    return {'conversation': ci, 'status': 'complete' if all(r['status']=='complete'for r in rows)else'failed',
            'facts':len(rows),'failed':sum(r['status']!='complete'for r in rows)}


def _load_tags(run, ci, bank, bank_ref, m):
    tags = {};contexts=session_contexts(ci)
    for f in bank['facts']:
        session=f.get('input_scope',{}).get('session_id')
        if session not in contexts:raise ValueError('Tagged new fact has no bound original session context')
        row = verify(Path(run)/'retrieval'/f'conv_{ci}'/'tags'/(f['id']+'.json'))
        if ((row.get('status'), row.get('conversation'), row.get('fact_id'), row.get('bank'), row.get('protocol_sha256')) !=
            ('complete', ci, f['id'], bank_ref, m['protocol_sha256']) or row.get('fact_text_sha256') != digest(f['text'])
            or row.get('source_context_sha256')!=digest(contexts[session])):
            raise ValueError('All actual new bank labels must match current fact bytes')
        if row['result'].get('fact_id')!=f['id']:raise ValueError('Tag result fact ID differs')
        _complete_fact_scores(row['result'], f['id']); tags[f['id']] = row['result']
    return tags


def validate_native_capture(cap, facts, user_id, *, _scorer_module=None):
    """Offline replay of captured scorer inputs with exact new-bank payload binding."""
    if cap.get('schema') != 'sdk20-native-pool-v1' or cap.get('capture_sha256') != digest({k:v for k,v in cap.items()if k!='capture_sha256'}):
        raise ValueError('Native scorer capture binding changed')
    inputs=cap['inputs']; rows=inputs['semantic_results']; byid={f['id']:f for f in facts}
    ids=[r['id']for r in rows]
    if (inputs.get('top_k')!=20 or inputs.get('threshold')!=.1 or inputs.get('explain') is not False
        or len(rows)>80 or len(set(ids))!=len(ids) or ids!=cap['candidate_ids']):
        raise ValueError('Native scorer candidate/budget protocol changed')
    for r in rows:
        if (r['id'] not in byid or r.get('payload',{}).get('user_id')!=user_id
            or r['payload'].get('data')!=byid[r['id']]['text']):
            raise ValueError('Captured native payload is outside the actual new bank')
    if _scorer_module is None:
        import importlib
        _scorer_module=importlib.import_module('mem0.memory.main')
    all_inputs=deepcopy(inputs);all_inputs['top_k']=len(rows)
    eligible=_scorer_module.score_and_rank(**all_inputs)
    base=_scorer_module.score_and_rank(**deepcopy(inputs))
    if (eligible!=cap['eligible'] or base!=cap['base20'] or base!=eligible[:20]
        or [r['id']for r in base]!=cap['base20_ids'] or [r['id']for r in eligible]!=cap['eligible_ids']
        or {r['id']:r['score']for r in eligible}!=cap['native_scores']):
        raise ValueError('Native scoring inputs do not reproduce captured ranking')
    return cap


def capture_current(memory, question, bank, bank_ref, tags=None, query_scores=None, *, _scorer_module=None):
    """One search creates both NEW N0 and H; no historical UUID replay assertion."""
    facts = fact_projection(bank); scope = {'user_id': bank['user_id']}
    if tags is not None:
        captured={}
        try:
            with capture_hybrid_native(memory, query=question, filters=scope, bank_facts=facts,_scorer_module=_scorer_module) as captured:
                returned = memory.search(question, top_k=20, filters=scope, rerank=False)
        except Exception as ex:
            ex.retrieval_capture=deepcopy(captured)
            raise
        native = captured['native_capture']
        pool = build_hybrid_pool(captured, facts, bank_sha256=bank_ref['sha256'],
                                 expected_native20=native['base20'], expected_native_capture=native)
        choices = select_hybrid_arms(pool, tags, query_scores)
        choices['baseline'] = {'selected_ids': list(native['base20_ids']), 'trace': {'method': 'native_sdk20_new_bank'}}
        value = {'capture': captured, 'pool': pool, 'selections': choices,
                 'native_replay_scope': 'current_newbank_same_N0_capture; not independent historical replay'}
    else:
        captured={}
        try:
            with capture_native_pool(_scorer_module=_scorer_module) as captured:
                returned = memory.search(question, top_k=20, filters=scope, rerank=False)
        except Exception as ex:
            ex.retrieval_capture=deepcopy(captured)
            raise
        choices = {'selected_ids': list(captured['base20_ids']), 'trace': {'method':'native_sdk20_new_bank'}}
        value = {'capture': captured, 'selection': choices}
    ids = [r['id'] for r in returned['results']]
    base = captured['native_capture']['base20_ids'] if tags is not None else captured['base20_ids']
    byid = {f['id']:f for f in facts}
    validate_native_capture(captured['native_capture'] if tags is not None else captured,facts,bank['user_id'],_scorer_module=_scorer_module)
    if ids != base or len(ids) != len(set(ids)) or len(ids) != 20 or not set(ids) <= set(byid):
        raise ValueError('Native returned20 differs or uses IDs outside new bank')
    for row in returned['results']:
        if 'memory' in row and row['memory'] != byid[row['id']]['text']:
            raise ValueError('Native public text differs from bank')
    value['native_search_request']={'query':question,'top_k':20,'filters':scope,'rerank':False,'threshold':.1,
                                    'explain':False,'reference_date':None,'show_expired':False}
    return value


def retrieve_worker(run, ci, api_factory=None):
    run = Path(run).resolve(); m, pack = manifest(run, ci)
    folder = run/'retrieval'/f'conv_{ci}'; qs = cohort()[0][str(ci)]
    intent = folder/'retrieval_intent.json'
    with intent.open('x', encoding='utf8') as h:
        json.dump({'pid':os.getpid(), 'created_utc':now(), 'protocol_sha256':m['protocol_sha256']},h)
        h.flush();os.fsync(h.fileno())
    emb = CacheOnlyEmbeddings({e['request_hash']:e for e in pack['entries'] if e['conversation']==ci})
    if api_factory is None:
        from jev_mem.gemini38_provider_v1 import Gemini38API
        types_api = Gemini38API(folder/'query_type_api', retries=2, context={'architecture':SCHEMA,'conversation':ci,'phase':'query_evidence_types'})
    else:
        types_api = api_factory(folder/'query_type_api',ci,m)
    resource = m['resources']; completed=0
    with local_bm25_context(resource['snapshot'], resource['files_sha256']):
        for arm in BANK_ARMS:
            item = next(r for r in m['banks'] if (r['conversation'],r['arm'])==(ci,arm)); bank = verify(checked(item['bank']))
            native_copy = Path(item['native_copy'])
            if inventory(native_copy) != item['native_inventory']:
                raise ValueError('New native copy changed before search')
            tags = _load_tags(run,ci,bank,item['bank'],m) if arm=='session' else None
            memory = None; q=None; scoped=None; qsapi=None; audit=None; value=None
            try:
                memory = writer.open_memory(native_copy, emb)
                from mem0.utils import spacy_models
                if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None or memory.vector_store._get_bm25_encoder() is None:
                    raise RuntimeError('Native NLP/BM25 unavailable')
                for q in qs:
                    qsapi=None;audit=None;value=None
                    scoped = emb.scoped(conversation=ci,query_id=q['id'],method=arm,phase='retrieval'); memory.embedding_model=writer.Embeddings(scoped)
                    query_scores = None; qt_ref = None
                    if arm=='session':
                        qp = folder/'query_types'/(q['id']+'.json'); qsapi = types_api.scoped(query_id=q['id'])
                        save_new(folder/'query_type_intents'/(q['id']+'.json'),{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],
                                 'conversation':ci,'query_id':q['id'],'question':q['question'],'created_utc':now(),'pid':os.getpid()})
                        try:
                            scores, metadata = query_type_scores(q['question'],qsapi); _complete_query_scores(scores)
                        except Exception as ex:
                            save_new(qp,{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,
                                     'query_id':q['id'],'question':q['question'],'status':'failed','error_type':type(ex).__name__,
                                     'api_events':list(qsapi.events)})
                            raise
                        save_new(qp,{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,
                                     'query_id':q['id'],'question':q['question'],'status':'complete','scores':scores,
                                     'metadata':metadata,'api_events':list(qsapi.events)})
                        query_scores=scores;qt_ref=pointer(qp)
                    audit = Warnings(); logging.getLogger('mem0').addHandler(audit)
                    try:
                        value = capture_current(memory,q['question'],bank,item['bank'],tags,query_scores)
                        if audit.items: raise RuntimeError('Native warnings require review')
                    finally:
                        logging.getLogger('mem0').removeHandler(audit)
                    p=folder/arm/'selections'/(q['id']+'.json')
                    save_new(p,{'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'status':'complete',
                                'conversation':ci,'bank_arm':arm,'query_id':q['id'],'question':q['question'],
                                'reference':q['reference'],'category':q['category'],'reference_date':m['reference_dates'][str(ci)],
                                'bank':item['bank'],'query_types':qt_ref,'cache_events':scoped.events,'gemini_generation_calls':0,**value})
                    completed+=1
                    publish_json(folder/'retrieval_status.json',{'completed':completed,'total':4*len(qs),'terminal':False,
                                 'last_query':q['id'],'bank_arm':arm,'protocol_sha256':m['protocol_sha256']})
            except Exception as ex:
                failed={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'bank_arm':arm,
                        'status':'failed','query_id':q['id']if q else None,'bank':item['bank'],'error_type':type(ex).__name__,
                        'error':'Retained retrieval intent/copy/cache requires audit; no implicit redispatch',
                        'cache_events':list(scoped.events)if scoped else [],'query_type_api_events':list(qsapi.events)if qsapi else [],
                        'native_warnings':audit.items if audit else [],'capture':getattr(ex,'retrieval_capture',value)}
                save_new(folder/arm/'retrieval_failure.json',failed)
                save_new(folder/'retrieval_outcome.json',{**failed,'queries':completed})
                raise
            finally:
                if memory is not None: writer.native.close_memory(memory)
            save_new(folder/arm/'post_native_inventory.json',{'inventory':inventory(native_copy),
                       'immutable_source_inventory_unchanged':inventory(item['native_source'])==item['native_inventory'],
                       'protocol_sha256':m['protocol_sha256']})
    result={'schema':SCHEMA,'conversation':ci,'status':'complete','queries':completed,'protocol_sha256':m['protocol_sha256']}
    save_new(folder/'retrieval_outcome.json',result);return result


def run_stage(run, stage, conversations=tuple(range(10)), workers=8, api_factory=None):
    run=Path(run).resolve();manifest(run,full=True)
    if (stage not in ('tag','retrieve') or not 1<=workers<=8 or not conversations or len(set(conversations))!=len(conversations)
        or any(type(ci)is not int or ci not in range(10) for ci in conversations)):
        raise ValueError('Unique conversation workers, maximum8, required')
    target=tag_worker if stage=='tag' else retrieve_worker;rows=[]
    lock=run/'retrieval/active_stage.lock'
    with lock.open('x',encoding='utf8') as h:json.dump({'pid':os.getpid(),'stage':stage,'created_utc':now()},h)
    try:
        with ProcessPoolExecutor(max_workers=min(workers,len(conversations))) as pool:
            futures={pool.submit(target,run,ci,api_factory):ci for ci in conversations}
            for f in as_completed(futures):
                try:r=f.result()
                except Exception as e:r={'conversation':futures[f],'status':'failed','error_type':type(e).__name__,
                                        'error':'Retained stage artifacts require audit; no automatic recovery'}
                rows.append(r);publish_json(run/'retrieval'/(stage+'_status.json'),{'stage':stage,'outcomes':rows,'terminal':len(rows)==len(conversations)})
    finally:lock.unlink()
    return {'stage':stage,'terminal':True,'outcomes':rows,'status':'complete' if all(r['status']=='complete'for r in rows)else'failed'}


def _selection_jobs(run, m, ci, arm):
    item=next(r for r in m['banks']if(r['conversation'],r['arm'])==(ci,arm));bank=verify(checked(item['bank']));facts=fact_projection(bank);byid={f['id']:f for f in facts}
    tags=_load_tags(run,ci,bank,item['bank'],m)if arm=='session'else None
    post=verify(Path(run)/'retrieval'/f'conv_{ci}'/arm/'post_native_inventory.json')
    if (post.get('immutable_source_inventory_unchanged') is not True or post.get('protocol_sha256')!=m['protocol_sha256']
        or post.get('inventory')!=inventory(item['native_copy'])):
        raise ValueError('Postsearch copy/source inventory proof differs')
    jobs=[]
    for q in cohort()[0][str(ci)]:
        p=Path(run)/'retrieval'/f'conv_{ci}'/arm/'selections'/(q['id']+'.json');row=verify(p)
        if (row.get('schema')!=SCHEMA or row.get('status')!='complete' or row.get('protocol_sha256')!=m['protocol_sha256']
            or row.get('bank')!=item['bank'] or row.get('conversation')!=ci or row.get('bank_arm')!=arm
            or any(row.get(k)!=q[v]for k,v in [('query_id','id'),('question','question'),('reference','reference'),('category','category')])
            or row.get('reference_date')!=m['reference_dates'][str(ci)]):raise ValueError('Selection question/gold/date/bank identity differs')
        if row.get('native_search_request')!={'query':q['question'],'top_k':20,'filters':{'user_id':bank['user_id']},
                    'rerank':False,'threshold':.1,'explain':False,'reference_date':None,'show_expired':False}:
            raise ValueError('Recorded native search request differs from current cohort/scope')
        if arm=='session':
            qt=verify(checked(row['query_types']));_complete_query_scores(qt['scores'])
            if (qt.get('status'),qt.get('query_id'),qt.get('question'),qt.get('protocol_sha256'))!=('complete',q['id'],q['question'],m['protocol_sha256']):raise ValueError('Query label identity differs')
            native=row['capture']['native_capture']
            if (row['capture'].get('query')!=q['question'] or row['capture'].get('effective_native_query')!=q['question'].strip()
                or row['capture'].get('filters')!={'user_id':bank['user_id']}):
                raise ValueError('Hybrid capture came from another query/scope')
            validate_native_capture(native,facts,bank['user_id'])
            pool=build_hybrid_pool(row['capture'],facts,bank_sha256=item['bank']['sha256'],expected_native20=native['base20'],expected_native_capture=native)
            choices=select_hybrid_arms(pool,tags,qt['scores']);choices['baseline']={'selected_ids':native['base20_ids'],'trace':{'method':'native_sdk20_new_bank'}}
            if row.get('pool')!=pool or stable_selections(row['selections'])!=stable_selections(choices):raise ValueError('Current-bank H selection no longer reproduces')
        else:
            cap=row['capture'];base=cap['base20_ids']
            validate_native_capture(cap,facts,bank['user_id'])
            choices={arm:{'selected_ids':base}}
            if row['selection']['selected_ids']!=base:raise ValueError('Native selected20 differs')
        for method,choice in choices.items():
            ids=choice['selected_ids']
            if len(ids)!=20 or len(set(ids))!=20 or not set(ids)<=set(byid):raise ValueError('Exactly20 new bank facts required')
            jobs.append({'job_id':method+'/'+q['id'],'method':method,'status':'ready','conversation':ci,
                         'query_id':q['id'],'question':q['question'],'reference':q['reference'],'category':q['category'],
                         'reference_date':m['reference_dates'][str(ci)],'selected_ids':list(ids),
                         'selected_facts':[{k:byid[fid].get(k)for k in ('id','text','observation_date')}for fid in ids],
                         'bank':item['bank'],'selection':pointer(p),'protocol_sha256':m['protocol_sha256'],
                         'lineage':{'bank':item['bank'],'selection':pointer(p)}})
    return jobs


def seal(run):
    run=Path(run).resolve();m,_=manifest(run,full=True);jobs=[]
    for ci in range(10):
        out=verify(run/'retrieval'/f'conv_{ci}'/'retrieval_outcome.json')
        if out.get('status')!='complete' or out.get('protocol_sha256')!=m['protocol_sha256'] or out.get('queries')!=4*len(cohort()[0][str(ci)]):raise ValueError('All four new bank retrievals must complete per conversation')
        for arm in BANK_ARMS:jobs.extend(_selection_jobs(run,m,ci,arm))
    if len(jobs)!=10780 or len({j['job_id']for j in jobs})!=10780:raise ValueError('Seven methods each1540 required')
    value={'schema':SOURCE_SCHEMA,'created_utc':now(),'status':'sealed','arms':list(ARMS),'jobs':jobs,
           'parent_run':str(run),'retrieval_manifest':pointer(run/'retrieval/run_manifest.json'),
           'new_execution_config':m['new_execution_config'],
           'protocol_sha256':m['protocol_sha256'],'source_dataset_sha256':SOURCE_HASH,
           'questions_per_arm':1540,'official_metric_provenance':m['official_metric_provenance'],
           'code_sha256':m['code_sha256'],'model_calls_at_seal':0}
    path=run/'retrieval/selected_inputs.json';save_new(path,value);validate_source(path);return {'source':pointer(path),'jobs':10780,'model_calls':0}


def validate_source(path):
    """Read-only full cohort + current bank/selection/code/hash admission for QA."""
    source=verify(path)
    if (source.get('schema')!=SOURCE_SCHEMA or source.get('status')!='sealed' or source.get('arms')!=list(ARMS)
        or source.get('source_dataset_sha256')!=SOURCE_HASH or source.get('questions_per_arm')!=1540):raise ValueError('New seven-arm sealed source required')
    run=Path(source['parent_run']).resolve();m,_=manifest(run,full=True)
    if (checked(source['retrieval_manifest'])!=(run/'retrieval/run_manifest.json').resolve()
        or source.get('new_execution_config')!=m['new_execution_config']
        or source['protocol_sha256']!=m['protocol_sha256'] or source['code_sha256']!=m['code_sha256']):raise ValueError('Selected source uses another protocol')
    _,official=verified_official_metric()
    if source['official_metric_provenance']!=official:raise ValueError('Official metric source changed')
    expected=[]
    for ci in range(10):
        out=verify(run/'retrieval'/f'conv_{ci}'/'retrieval_outcome.json')
        if (out.get('status')!='complete' or out.get('conversation')!=ci or out.get('protocol_sha256')!=m['protocol_sha256']
            or out.get('queries')!=4*len(cohort()[0][str(ci)])):
            raise ValueError('Selected source has no complete four-bank retrieval outcome')
        for arm in BANK_ARMS:expected.extend(_selection_jobs(run,m,ci,arm))
    if source['jobs']!=expected or len(expected)!=10780 or len({j['job_id']for j in expected})!=10780:raise ValueError('Full selected20/source/question/gold/date lineage differs')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check','tag','retrieve','seal','validate-source'))
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gate',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--conversations',nargs='+',type=int,default=list(range(10)));p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='prepare':result=prepare(a.run_dir,a.gate)
    elif a.stage=='check':result={'valid':bool(manifest(a.run_dir,full=True)),'model_calls':0}
    elif a.stage=='seal':result=seal(a.run_dir)
    elif a.stage=='validate-source':result={'valid':bool(validate_source(a.source or a.run_dir/'retrieval/selected_inputs.json')),'model_calls':0}
    else:result=run_stage(a.run_dir,a.stage,tuple(a.conversations),a.workers)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
