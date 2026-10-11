"""Offline source/raw/ADD/index audit for terminal Gemini 3.8 rebuilt banks.

Reuse the existing strict grouped inspector and SQLite/BM25 helpers. Native
entity deficits still block admission; no old exception is implicitly reused.
Semantic sampling is a separate artifact, never inferred from structural pass.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_regroup_refinement_banks_v1 as strict
from tools import mem0_regroup_firstpass_review_v2 as idx
from tools import run_gemini38_writer_v1 as writer
from tools.run_sdk20_modules import save_new, now
from jev_mem.gemini38_provider_v1 import MODEL, generation_body, response_text, WRITER_MAX_OUTPUT
from jev_mem.local_bm25_context_v2 import forbid_network, compare_persisted_fact_vectors

require = strict.require
SCHEMA = 'gemini38-terminal-source-index-audit-v1'


def check_generation(journal, ledger, native, prompt, env, ci, arm):
    require(journal['builder_call_count'] == 1 and len(journal['llm_calls']) == 1, 'Exactly one extraction required')
    call = journal['llm_calls'][0]
    messages = [{'role': 'system', 'content': env['ADDITIVE_EXTRACTION_PROMPT']}, {'role': 'user', 'content': prompt}]
    require(call['kwargs']['messages'] == messages, 'Actual extraction input differs from original reconstructed prompt')
    key = 'session_id' if arm == 'session' else 'group_id'
    events = [r for r in ledger if r.get('stage') == 'mem0_extract' and r.get('context', {}).get(key) == journal[key]
              and (r.get('cache_hit') or r.get('status') == 200)]
    require(len(events) == 1, 'Exactly one original extraction response required')
    event = events[0]
    require((event.get('provider'), event.get('model'), event.get('operation')) == ('gemini', MODEL, 'generateContent'), 'Extraction model/operation differs')
    require(event['context'].get('conversation') == ci and event['context'].get('method') == arm, 'Wrong extraction context')
    path = native/'api_cache'/(event['request_hash']+'.json')
    raw = idx.read(path)
    signature = {'provider': 'gemini', 'model': MODEL, 'operation': 'generateContent', 'body':
                 generation_body(prompt, system=env['ADDITIVE_EXTRACTION_PROMPT'], json_output=True, max_output=WRITER_MAX_OUTPUT)}
    require(raw['request'] == signature and raw['stage'] == 'mem0_extract', 'Original 3.8 request differs')
    require(hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == event['request_hash'], 'Request hash mismatch')
    parsed = json.loads(response_text(raw['response']))
    require(parsed == json.loads(call['response']) and parsed['memory'] == call['parsed_memory'], 'Raw/parser chain differs')
    require(isinstance(parsed['memory'], list) and all(isinstance(x, dict) and isinstance(x.get('text'), str) and x['text'].strip() for x in parsed['memory']), 'Invalid extracted records')
    return parsed['memory'], idx.pointer(path)


# Independent globals: the old module/functions remain untouched.
_f = strict.inspect_bank
_group = FunctionType(_f.__code__, {**vars(strict), 'check_generation': check_generation}, _f.__name__, _f.__defaults__, _f.__closure__)


def entity_audit(bank, facts, entities, extractor, nlp):
    linked, missing, empty = set(), [], []
    for eid, point in entities.items():
        vectors = point.vector if isinstance(point.vector, dict) else {'': point.vector}
        require(idx.dense(vectors.get('')) and point.payload['user_id'] == bank['user_id'], 'Entity dense/scope invalid')
        links = point.payload.get('linked_memory_ids', [])
        require(links and len(set(links)) == len(links) and set(links) <= set(facts), 'Invalid entity links')
        linked.update(links)
        if vectors.get('bm25') is None:
            missing.append(eid)
        else:
            require(idx.sparse(vectors['bm25'], allow_empty=True), 'Malformed entity sparse')
            if not vectors['bm25'].indices:
                empty.append(eid)
    expected = {f['id'] for f, doc in zip(bank['facts'], nlp.pipe([f['text'] for f in bank['facts']], batch_size=32)) if extractor._extract_entities_from_doc(doc)}
    orphans = sorted(expected - linked)
    declared = bank['native_index_audit']
    require(declared['facts_with_local_entities'] == len(expected) and declared['entity_facts_without_any_link'] == orphans
            and set(declared['linked_memory_ids']) == linked and declared['entities'] == len(entities), 'Declared entity audit differs')
    return orphans, missing, empty


def inspect_session(target, manifest, conversation, env, extractor, nlp):
    ci, arm, folder, state, outcome, bank = target
    native = folder/'native'
    before = idx.native_inventory(native)
    inp = idx.verify(folder/'group_inputs.json')
    envelopes = list(env['session_envelopes']({'conversation': conversation}))
    require(inp['envelopes'] == envelopes and idx.bound(bank['group_inputs']) == inp, 'Original whole-session inputs differ')
    require(len(bank['journal_files']) == len(envelopes), 'Session journal coverage differs')
    facts = idx.points(native/'qdrant/collection/facts/storage.sqlite')
    entities = idx.points(native/'qdrant/collection/facts_entities/storage.sqlite')
    histories = idx.rows(native/'history.sqlite', 'history')
    exported = {f['id']: f for f in bank['facts']}
    require(len(exported) == len(bank['facts']) and set(exported) == set(facts), 'Fact identity mismatch')
    require(Counter(h['memory_id'] for h in histories) == Counter(exported.keys()), 'History ADD count mismatch')
    for h in histories:
        require(h['event'] == 'ADD' and not h['is_deleted'] and h['old_memory'] is None and h['new_memory'] == exported[h['memory_id']]['text'], 'History content mismatch')
    ledger = [json.loads(x) for x in idx.shared_read(native/'api_calls.jsonl').decode('utf-8').splitlines()]
    idx.sha(native/'api_calls.jsonl')
    emitted, history, journals, quality = [], [], [], []
    for seq, (envelope, ref) in enumerate(zip(envelopes, bank['journal_files'])):
        j = idx.bound(ref)
        require(j['status'] == 'success' and j['sequence'] == seq and all(j[k] == v for k,v in envelope.items()), 'Session journal/source differs')
        require(j['current_date'] == manifest['current_date'] and j['history_context'] == history[-10:], 'Session date/history differs')
        request_keys = ('user_id', 'session_id', 'messages', 'observation_date', 'current_date', 'source_turn_ids', 'metadata')
        require(j['request_sha256'] == idx.digest({k:j[k] for k in request_keys}), 'Session request fingerprint differs')
        prior = {f['id']:f for f in emitted}
        for e in j['existing_memory_context']:
            require(e['id'] in prior and e['text'] == prior[e['id']]['text'] and e['hash'] == hashlib.md5(e['text'].encode()).hexdigest(), 'Existing memory context differs')
        prompt = env['generate_additive_extraction_prompt'](
            existing_memories=[{'id':str(i), 'text':f['text']} for i,f in enumerate(j['existing_memory_context'])],
            new_messages=env['parse_messages'](j['messages']), last_k_messages=[x['message'] for x in history[-10:]],
            current_date=manifest['current_date'], timestamp=j['observation_date'], custom_instructions=None)
        extracted, raw = check_generation(j, ledger, native, prompt, env, ci, arm)
        existing = {e['hash'] for e in j['existing_memory_context']}
        expected = {f['text'] for f in extracted if hashlib.md5(f['text'].encode()).hexdigest() not in existing}
        require({a['memory'] for a in j['results']} == expected and len(j['results']) == len(expected) == len(j['fact_records'])
                and j['sdk_response']['results'] == j['results'], 'Session parsed-to-ADD mismatch')
        require(not j['persistence_check']['missing_texts'] and not j['persistence_check']['unexpected_texts'], 'Session persistence incomplete')
        scope = {'session_id':j['session_id'], 'source_turn_ids':j['source_turn_ids'],
                 'historical_turn_ids':[x['source_turn_id'] for x in history[-10:]],
                 'existing_memory_ids':[x['id'] for x in j['existing_memory_context']]}
        for f,a in zip(j['fact_records'], j['results']):
            require((f['id'],f['text'],f['event']) == (a['id'],a['memory'],'ADD') and a['event'] == 'ADD', 'Session ADD identity mismatch')
            require(f['input_scope'] == scope and f['observation_date'] == j['observation_date'] and f['verified_support'] == [] and f['support_status'] == 'unverified', 'Session support/date scope changed')
            point = facts[f['id']]; payload = point.payload; vectors = point.vector
            require(payload['data'] == f['text'] and payload['hash'] == hashlib.md5(f['text'].encode()).hexdigest()
                    and payload['user_id'] == bank['user_id'] and payload['jev_mem_session_id'] == j['session_id']
                    and payload['jev_mem_observation_date'] == j['observation_date'], 'Session native payload changed')
            require(payload.get('attributed_to') == next(x.get('attributed_to') for x in extracted if x['text'] == f['text']), 'Attribution changed before ADD')
            require(isinstance(vectors, dict) and idx.dense(vectors.get('')) and idx.sparse(vectors.get('bm25')), 'Fact dense/sparse missing')
            require(idx.dense(exported[f['id']]['embedding']) and max(abs(x-y) for x,y in zip(vectors[''], exported[f['id']]['embedding'])) <= 1e-6, 'Dense export mismatch')
        journals.append({'session_id':j['session_id'], 'journal':ref, 'raw':raw, 'facts':len(j['fact_records'])})
        if not extracted or seq in (0, len(envelopes)-1):
            quality.append({'session_id':j['session_id'], 'journal':ref, 'raw':raw, 'empty_extraction':not extracted})
        emitted.extend(j['fact_records'])
        history.extend({'message':msg, 'session_id':j['session_id'], 'source_turn_id':tid} for msg,tid in zip(j['messages'],j['source_turn_ids']))
    require(emitted == [{k:v for k,v in f.items() if k != 'embedding'} for f in bank['facts']]
            and state['completed_groups'] == len(envelopes) and state['facts'] == len(facts), 'Session bank/status count mismatch')
    orphans, missing, empty = entity_audit(bank, facts, entities, extractor, nlp)
    require(idx.native_inventory(native) == before, 'Native inventory changed during audit')
    return {'conversation':ci, 'arm':arm, 'status':'strict_pass' if not orphans else 'blocked_entity_links',
            'strict_passed':not orphans, 'source_generation_persistence_passed':True, 'bank':idx.pointer(folder/'bank.json'),
            'outcome':idx.pointer(folder/'write_outcome.json'), 'inputs':idx.pointer(folder/'group_inputs.json'),
            'native_dir':str(native), 'native_inventory':before, 'groups':len(envelopes), 'facts':len(facts),
            'orphan_fact_ids':orphans, 'unused_entity_bm25_missing_ids':missing, 'unused_entity_bm25_empty_ids':empty,
            'journals':journals, 'semantic_review_targets':quality, 'semantic_correctness_verified':False}


def audit(run, conversations):
    run = Path(run).resolve(); idx.HASHES.clear()
    require(conversations in ([0,1], list(range(10))), 'Only development8 or full40 bank review')
    # The same OS lock as the controller must be available; never delete it.
    with writer.grouped.writer_process_lock(run), forbid_network() as guards:
        m = writer.manifest(run)
        targets, failures = [], []
        for ci in conversations:
            for arm in writer.ARMS:
                folder = run/f'conv_{ci}'/arm
                s = idx.read(folder/'write_status.json'); o = idx.verify(folder/'write_outcome.json')
                require(s['status'] in ('complete','failed') and s.get('finished_utc') and o['status'] == s['status'], 'Bank not terminal')
                require(o['write_status'] == idx.pointer(folder/'write_status.json') and o['protocol_sha256'] == m['protocol_sha256'], 'Outcome binding differs')
                if s['status'] != 'complete':
                    failures.append({'conversation':ci, 'arm':arm, 'status':s['status'], 'write_status':idx.pointer(folder/'write_status.json'), 'outcome':idx.pointer(folder/'write_outcome.json')})
                    continue
                b = idx.bound(o['bank'])
                require(b['complete'] is True and (b['conversation'],b['arm'],b['protocol_sha256']) == (ci,arm,m['protocol_sha256']) and s['bank'] == o['bank'], 'Bank binding differs')
                for p in (folder/'native').rglob('*'):
                    require(not p.is_symlink(), 'Symlink native source refused')
                    require(not (p.is_file() and p.name.endswith(('-wal','-journal')) and p.stat().st_size), 'Live database refused')
                targets.append((ci,arm,folder,s,o,b))
        if failures:
            return {'schema':SCHEMA, 'status':'blocked_terminal_writer_failure', 'protocol_sha256':m['protocol_sha256'], 'failures':failures,
                    'strict_passed':False, 'model_calls':0, 'created_utc':now(), 'banks':[]}
        env = strict.prompt_environment(); extractor,nlp = idx.load_extractor()
        data = idx.read(ROOT/'data/locomo10.json')
        rows = [(inspect_session if t[1] == 'session' else _group)(t,m,data[t[0]]['conversation'],env,extractor,nlp) for t in targets]
        r = m['resources']
        bm25 = compare_persisted_fact_vectors([Path(x['native_dir']) for x in rows], r['snapshot'], r['files_sha256'])
        require(bm25['facts_compared'] == bm25['exact_matches'] == sum(x['facts'] for x in rows), 'Fact BM25 equivalence incomplete')
        require(not any(g.call_count for g in guards), 'Offline audit attempted network')
        for p,h in list(idx.HASHES.items()):
            require(idx.sha(p) == h, 'Audit source mutated')
        return {'schema':SCHEMA, 'created_utc':now(), 'status':'strict_source_index_passed_semantic_review_pending' if all(x['strict_passed'] for x in rows) else 'blocked_entity_links',
                'protocol_sha256':m['protocol_sha256'], 'writer_manifest':idx.pointer(run/'run_manifest.json'), 'conversations':conversations,
                'strict_passed':all(x['strict_passed'] for x in rows), 'banks':rows, 'offline_bm25_equivalence':bm25,
                'source_file_hashes':dict(idx.HASHES), 'model_calls':0, 'semantic_correctness_verified':False, 'exceptions':[],
                'limitations':['Input scope is not proof every generated fact is semantically correct.', 'Entity check detects orphan facts, not every missing partial edge.', 'Unused entity sparse gaps are reported, not repaired.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--conversations', nargs='+', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new audit output')
    result = audit(a.run_dir, a.conversations)
    save_new(a.output, result)
    print(json.dumps({'output':str(a.output), 'status':result['status'], 'banks':len(result['banks']), 'model_calls':0}))


if __name__ == '__main__':
    main()
