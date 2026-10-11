"""Versioned, offline effective40 source review. Original strict gates stay unchanged.

Two exact terminal entity-link deficits may be exported as metadata in a new
namespace. No native client, repair, generation or original-file write occurs.
Audit and admission are separate; this module never publishes an admission gate.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_gemini38_banks_v1 as old
from tools import prepare_gemini38_hybrid_v1 as retrieval
from tools import recover_gemini38_session_io_v1 as recovery
from tools import mem0_regroup_firstpass_review_v2 as idx
from tools import run_gemini38_writer_v1 as writer
from tools.run_sdk20_modules import now, save_new
from jev_mem.data import SOURCE_HASH
from jev_mem.local_bm25_context_v2 import forbid_network, compare_persisted_fact_vectors

SCHEMA = 'gemini38-effective40-source-index-review-v1'
GATE_SCHEMA = 'gemini38-effective40-bank-admission-v1'
EXPORT_SCHEMA = 'gemini38-reviewed-native-bank-export-v1'
KNOWN = {
    (3, 'boundary_multiscale'): ('17b9d52c-2942-4aab-b8fa-4604f1a65562',
        'Nate unwinds at home by playing video games and watching movies.', 33, 177),
    (3, 'source_local_context'): ('742f94d0-bae6-4a11-b3f9-690151b42177',
        'Nate unwinds by playing video games and watching movies at home.', 36, 185),
}
FAILURES = set(KNOWN) | {(3, 'session')}
EXPECTED = {(ci, arm) for ci in range(10) for arm in writer.ARMS}
ERROR = {'type': 'ValueError', 'message': 'Native entity linking omitted extracted entity-bearing facts'}
require = old.require


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact pointer required')
    require(idx.pointer(ref['path']) == ref, 'Bound artifact changed')
    return Path(ref['path']).resolve()


def inside(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    require(p.is_relative_to(r), 'Artifact outside permitted namespace')
    return p


def full_native_inventory(native):
    """All persisted native evidence, including raw/ledger/journal files, no lock bytes."""
    native = Path(native).resolve()
    require(native.is_dir(), 'Native directory missing')
    answer = {}
    for p in sorted(native.rglob('*')):
        require(not p.is_symlink(), 'Symlink native source refused')
        if p.is_file():
            require(not (p.name.endswith(('-wal', '-journal')) and p.stat().st_size), 'Nonempty SQLite WAL/journal refused')
            if p.name != '.lock':
                answer[p.relative_to(native).as_posix()] = idx.sha(p)
    require(answer and 'history.sqlite' in answer, 'Incomplete native inventory')
    return answer


def inventory_normalized(value):
    require(isinstance(value, dict), 'Native inventory missing')
    answer = {k.replace('\\', '/'): v for k, v in value.items()}
    require(len(answer) == len(value), 'Duplicate normalized native path')
    return answer


@contextlib.contextmanager
def closed_native(native, state):
    """Real OS lock, dead recorded PID and no WAL; no native client is instantiated."""
    import portalocker
    require(state.get('status') in ('complete', 'failed') and state.get('finished_utc'), 'Native owner not terminal')
    require(not recovery.pid_alive(state['pid']), 'Native owner PID alive (reuse conservatively blocks)')
    native = Path(native).resolve()
    before = full_native_inventory(native)
    lock = native/'qdrant/.lock'
    require(lock.is_file() and not lock.is_symlink(), 'Existing Qdrant OS lock required')
    with lock.open('rb') as h:
        portalocker.lock(h, portalocker.LOCK_EX | portalocker.LOCK_NB)
        try:
            require(full_native_inventory(native) == before, 'Native changed while acquiring lock')
            yield before
            require(full_native_inventory(native) == before, 'Native changed during readonly review')
        finally:
            portalocker.unlock(h)


def validate_fact_bank(bank, ci, arm, parent_protocol):
    """Content/identity validation only; admit_gate checks the full provenance."""
    if bank.get('schema') != EXPORT_SCHEMA:
        return retrieval._fact_bank(bank, ci, arm, parent_protocol)
    require((ci, arm) in KNOWN, 'Export outside exact deficit allowlist')
    require((bank.get('conversation'), bank.get('arm'), bank.get('protocol_sha256')) == (ci, arm, parent_protocol)
            and bank.get('source_dataset_sha256') == SOURCE_HASH, 'Export scientific identity differs')
    require('complete' not in bank and bank.get('original_write_status') == 'failed'
            and bank.get('all_planned_groups_persisted') is True
            and bank.get('reviewed_export_status') == 'reviewed_with_native_limitations'
            and bank.get('entity_links_complete') is False and bank.get('strict_passed') is False,
            'Export must retain failed status/actual deficit; no complete or strict-pass fiction')
    uid = bank.get('user_id'); facts = bank.get('facts', [])
    require(uid == f'dated_regroup_c{ci}_{arm}' and len(facts) == KNOWN[(ci, arm)][3]
            and len({f.get('id') for f in facts}) == len(facts), 'Export fact count/ID/user scope differs')
    for f in facts:
        require(isinstance(f.get('id'), str) and f['id'] and not any(c in f['id'] for c in '/\\:')
                and f['id'] not in ('.', '..') and isinstance(f.get('text'), str) and f['text'].strip()
                and f.get('user_id') == uid and idx.dense(f.get('embedding')), 'Export fact malformed')
    mid, text, count, _ = KNOWN[(ci, arm)]
    require(bank.get('orphan_fact_ids') == [mid] and next((f['text'] for f in facts if f['id'] == mid), None) == text
            and len(bank.get('journal_files', [])) == count, 'Export exact known defect/journal coverage differs')
    return facts


def failure_contract(run, failure_review):
    r = idx.bound(failure_review)
    items = r.get('failures', [])
    require(len(items) == 3 and {(x['conversation'], x['arm']) for x in items} == FAILURES,
            'Only the exact three firstpass failures are supported')
    session = next(x for x in items if x['arm'] == 'session')
    rr, m, parent, _ = recovery.validate_review(failure_review, session['recovery_namespace'])
    require(parent == Path(run).resolve(), 'Failure review belongs to another parent')
    require(idx.pointer(parent/'run_manifest.json') == r['parent_manifest'], 'Scientific parent changed')
    return rr, m, { (x['conversation'], x['arm']): x for x in items }


def validate_known_defect(ci, arm, state, outcome, proof, native_inventory, parent_ref, input_ref):
    slot = (ci, arm); require(slot in KNOWN, 'Unknown failed export')
    mid, text, groups, facts = KNOWN[slot]
    require(state.get('status') == outcome.get('status') == 'failed'
            and state.get('error') == outcome.get('error') == ERROR
            and state.get('completed_groups') == state.get('expected_groups') == groups
            and state.get('facts') == facts, 'Failed bank is not the exact fully written entity defect')
    require(proof.get('schema') == 'gemini38-terminal-c3-entity-overwrite-readonly-audit-v1'
            and proof.get('state') == state and proof.get('outcome') == outcome
            and proof.get('parent_manifest') == parent_ref and proof.get('inputs') == input_ref
            and proof.get('source_sha_unchanged') is True and proof.get('source_mutations') == 0
            and proof.get('model_calls') == 0 and proof.get('bank_json_exists') is False,
            'Diagnosis binding or readonly provenance differs')
    require(proof.get('native_inventory_before') == proof.get('native_inventory_after')
            and inventory_normalized(proof['native_inventory_before']) == native_inventory,
            'Native diagnosis inventory drift')
    orphans = proof.get('orphans', [])
    require(len(orphans) == 1 and orphans[0]['fact']['id'] == mid and orphans[0]['fact']['text'] == text,
            'Diagnosis does not identify the exact known fact')
    chain = proof['source_generation_add_chain']
    require(chain['groups_verified'] == groups and chain['facts_verified'] == facts
            and proof['fact_bm25_exact']['compared'] == proof['fact_bm25_exact']['exact_matches'] == facts,
            'Diagnosis did not bind a full persisted bank')
    for row in chain['raw_refs']:
        checked(row['journal']); checked(row['raw'])


def build_export(run, ci, arm, parent, item, state, outcome, extractor, nlp, native_inventory):
    """Pure metadata builder over CLOSED readonly native data; never changes facts."""
    folder = Path(run)/f'conv_{ci}'/arm; native = folder/'native'
    require(not (folder/'bank.json').exists(), 'Failed original unexpectedly has a bank export')
    proof = idx.bound(item['diagnosis'])
    input_ref = next(x['input'] for x in parent['inputs'] if (x['conversation'],x['arm']) == (ci,arm))
    validate_known_defect(ci,arm,state,outcome,proof,native_inventory,idx.pointer(Path(run)/'run_manifest.json'),input_ref)
    refs = [x['journal'] for x in proof['source_generation_add_chain']['raw_refs']]
    journals = [idx.bound(ref) for ref in refs]
    require([j['sequence'] for j in journals] == list(range(KNOWN[(ci,arm)][2]))
            and all(j['status'] == 'success' for j in journals), 'Incomplete failed-bank journal sequence')
    facts = copy.deepcopy([f for j in journals for f in j['fact_records']])
    points = idx.points(native/'qdrant/collection/facts/storage.sqlite')
    entities = idx.points(native/'qdrant/collection/facts_entities/storage.sqlite')
    history = idx.rows(native/'history.sqlite','history')
    idx.validate_records(facts,points,history)
    for f in facts:
        f['embedding'] = list(points[f['id']].vector[''])
    linked = set()
    for point in entities.values():
        linked.update(point.payload.get('linked_memory_ids', []))
    expected = {f['id'] for f,doc in zip(facts,nlp.pipe([f['text'] for f in facts],batch_size=32))
                if extractor._extract_entities_from_doc(doc)}
    index = {'history_adds_verified':len(facts),'entities':len(entities), 'linked_memory_ids':sorted(linked),
             'facts_with_local_entities':len(expected),'entity_facts_without_any_link':sorted(expected-linked),
             'scope':'Native store consistency only, not factual semantic support.'}
    b = {'schema':EXPORT_SCHEMA,'protocol_sha256':parent['protocol_sha256'],'conversation':ci,'arm':arm,
         'user_id':f'dated_regroup_c{ci}_{arm}','facts':facts,'group_inputs':input_ref,
         'fact_support':'unverified_group_input_scope','journal_files':refs,'native_index_audit':index,
         'explicit_empty_group_ids':[j['group_id'] for j in journals if j.get('explicit_empty_extraction')],
         'chunk_embedding_group_ids':[j['group_id'] for j in journals if j.get('group_context_embedding',{}).get('triggered')],
         'native_dir':str(native.resolve()),'source_dataset_sha256':SOURCE_HASH,
         'all_planned_groups_persisted':True,'reviewed_export_status':'reviewed_with_native_limitations',
         'original_write_status':'failed','entity_links_complete':False,'strict_passed':False,
         'orphan_fact_ids':sorted(expected-linked),'native_inventory':native_inventory,
         'scientific_parent_manifest':idx.pointer(Path(run)/'run_manifest.json'),
         'original_status_pointer':idx.pointer(folder/'write_status.json'),
         'original_outcome_pointer':idx.pointer(folder/'write_outcome.json'),'diagnosis':item['diagnosis']}
    validate_fact_bank(b,ci,arm,parent['protocol_sha256'])
    return b


def inspect_effective_bank(target, parent, conversation, env, extractor, nlp, *, bank_pointer=None):
    """Reuse the frozen inspector code object; only exact exported metadata pointer differs."""
    ci,arm,folder,state,outcome,bank = target
    if bank_pointer is None:
        return (old.inspect_session if arm == 'session' else old._group)(target,parent,conversation,env,extractor,nlp)
    require(arm != 'session' and bank.get('schema') == EXPORT_SCHEMA, 'Pointer seam only for explicit export')
    exact = (folder/'bank.json').resolve()
    def pointer(path):
        return bank_pointer if Path(path).resolve() == exact else idx.pointer(path)
    proxy = SimpleNamespace(**{**vars(idx), 'pointer':pointer})
    f = old.strict.inspect_bank
    cloned = FunctionType(f.__code__,{**vars(old.strict),'check_generation':old.check_generation,
                                    'index_audit':proxy},f.__name__,f.__defaults__,f.__closure__)
    return cloned(target,parent,conversation,env,extractor,nlp)


def _recovery_source(namespace, parent, failure_review):
    namespace = Path(namespace).resolve()
    m,pm,original = recovery.check(namespace)
    require(pm == parent and m['failure_review'] == failure_review, 'Recovery is from another reviewed firstpass')
    folder = namespace/'conv_3/session'
    intent = idx.read(namespace/'recovery_intent.json'); out = idx.read(folder/'recovery_outcome.json')
    require(intent.get('once_only') is True and not recovery.pid_alive(intent['pid'])
            and intent['execution_recovery_sha256'] == out.get('execution_recovery_sha256') == m['execution_recovery_sha256']
            and out.get('status') == 'complete' and out.get('prefix_requests_consumed') == out.get('prefix_requests_expected') == 18,
            'Recovery incomplete, active or wrong exact-prefix execution')
    state = idx.read(folder/'write_status.json'); scientific = idx.verify(folder/'write_outcome.json')
    require(state.get('status') == scientific.get('status') == 'complete' and scientific['write_status'] == idx.pointer(folder/'write_status.json')
            and state.get('bank') == scientific.get('bank') == out.get('bank'), 'Recovery terminal bank chain differs')
    bank = idx.bound(out['bank']); validate_fact_bank(bank,3,'session',parent['protocol_sha256'])
    require(state['completed_groups'] == state['expected_groups'] == len(idx.bound(bank['group_inputs'])['envelopes']) == 29,
            'Recovery requires every original session, not the prefix probe')
    for p,h in m['original_evidence_sha256'].items():
        # prefix_sources currently exports {path: SHA}, not reconstructed parsed text.
        require(idx.sha(p) == h, 'Original recovery evidence changed')
    validate_recovery_prefix(folder, m)
    return folder,state,scientific,bank,{
        'recovery_manifest':idx.pointer(namespace/'recovery_manifest.json'),
        'recovery_intent':idx.pointer(namespace/'recovery_intent.json'),
        'recovery_outcome':idx.pointer(folder/'recovery_outcome.json')}


def validate_recovery_prefix(folder, manifest):
    """All18 complete request signatures, original raw bytes and consumer ledger."""
    folder=Path(folder); native=folder/'native'; prefix=manifest['prefix_requests']
    require(len(prefix)==18 and len({x['request_hash'] for x in prefix})==18,'Exact18 unique inherited requests required')
    ledger=[json.loads(x) for x in idx.shared_read(native/'api_calls.jsonl').splitlines() if x.strip()]
    idx.sha(native/'api_calls.jsonl')
    events=[json.loads(x) for x in idx.shared_read(folder/'recovery_request_events.jsonl').splitlines() if x.strip()]
    idx.sha(folder/'recovery_request_events.jsonl')
    inherited=[x for x in events if x.get('source')=='inherited_exact_request' and x.get('event')=='request_returned']
    require([x['request_hash'] for x in inherited]==[x['request_hash'] for x in prefix], 'Inherited request order incomplete/drifted')
    require(not any(x.get('event') in ('hard_guard_rejection','request_failed') for x in events),'Recovery guard failure recorded')
    for row,event in zip(prefix,inherited):
        key=row['request_hash']; raw_path=native/'api_cache'/f'{key}.json'
        require(idx.sha(raw_path)==row['raw']['sha256'] and idx.read(raw_path)['request']==row['signature'], 'Inherited complete request/raw drift')
        lineage=idx.read(native/'cache_lineage'/f'{key}.json')
        require(lineage['source_path']==row['raw']['path'] and lineage['source_sha256']==row['raw']['sha256']
                and lineage['target_path']==str(raw_path.resolve()) and lineage['target_sha256']==row['raw']['sha256']
                and lineage['request_hash']==key and lineage['original_intent']==row['intent']
                and lineage['original_physical_attempt']==row['original_attempt'],'Inherited raw provenance drift')
        matches=[x for x in ledger if x.get('request_hash')==key]
        require(len(matches)==1 and matches[0].get('cache_hit') is True
                and matches[0].get('context')==row['context'] and matches[0].get('stage')==row['stage']
                and (matches[0].get('provider'),matches[0].get('model'),matches[0].get('operation'))==
                    (row['signature']['provider'],row['signature']['model'],row['signature']['operation']),
                'Inherited ledger consumer lacks exact own context/stage or was freshly called')
        require(event['context']==row['context'] and event['stage']==row['stage'] and event['original_raw']==row['raw']
                and event['consumed_raw']==idx.pointer(raw_path), 'Inherited recovery event changed')


def audit_effective40(run, failure_review, *, recovery_run=None, export_dir=None):
    """Explicit root invocation only: readonly audit and NEW metadata-only exports.

    A missing recovery is pending; no partial bank is exported or admitted.
    The existing old strict audit remains blocked and is never overwritten.
    """
    run=Path(run).resolve(); idx.HASHES.clear()
    export_dir=inside(export_dir or run/'reviews/effective40_v1/exports',run/'reviews/effective40_v1')
    with recovery.closed_parent_lock(run), forbid_network() as guards:
        review,parent,items=failure_contract(run,failure_review)
        env=old.strict.prompt_environment(); extractor,nlp=idx.load_extractor()
        require(idx.sha(ROOT/'data/locomo10.json') == SOURCE_HASH,'Original dataset changed')
        data=idx.read(ROOT/'data/locomo10.json'); rows=[]; pending=[]
        originals={(b['conversation'],b['arm']):b for b in review['banks']}
        for ci,arm in sorted(EXPECTED):
            original=originals[(ci,arm)]; folder=run/f'conv_{ci}'/arm
            provenance={'scientific_parent_manifest':review['parent_manifest'],'policy':review['policy'],
                        'failure_review':failure_review}
            if (ci,arm)==(3,'session'):
                if recovery_run is None:
                    pending.append({'conversation':ci,'arm':arm,'reason':'Full29-session new recovery not supplied'})
                    continue
                folder,state,outcome,bank,extra=_recovery_source(recovery_run,parent,failure_review)
                provenance.update(extra); kind='io_recovery_complete'; bp=state['bank']
            else:
                state=idx.bound(original['write_status']); outcome=idx.verify(checked(original['outcome']))
                kind='reviewed_native_export' if (ci,arm) in KNOWN else 'original_complete'
                bank=None if kind=='reviewed_native_export' else idx.bound(outcome['bank'])
                bp=None if bank is None else outcome['bank']
            native=folder/'native'
            with closed_native(native,state) as inventory:
                if bank is None:
                    item=items[(ci,arm)]
                    require(item['action']=='readonly_native_export_with_declared_link_deficit','Wrong export policy action')
                    bank=build_export(run,ci,arm,parent,item,state,outcome,extractor,nlp,inventory)
                    candidate_path=export_dir/f'conv_{ci}'/arm/'bank_export.json'
                    # The inspector receives an explicit unpublished candidate pointer;
                    # the metadata is saved only after its full source/ADD checks pass.
                    sealed={**bank,'content_sha256':idx.digest(bank)}
                    bp={'path':str(candidate_path.resolve()),'sha256':None}
                    row=inspect_effective_bank((ci,arm,folder,state,outcome,bank),parent,data[ci]['conversation'],env,extractor,nlp,bank_pointer=bp)
                    require(row['orphan_fact_ids']==[KNOWN[(ci,arm)][0]] and row['strict_passed'] is False,'Unknown entity deficit')
                    if candidate_path.exists():
                        require(idx.verify(candidate_path)==sealed,'Existing export differs; no overwrite')
                    else:
                        save_new(candidate_path,bank)
                    bp=idx.pointer(candidate_path); row['bank']=bp
                    provenance['diagnosis']=item['diagnosis']
                else:
                    validate_fact_bank(bank,ci,arm,parent['protocol_sha256'])
                    row=inspect_effective_bank((ci,arm,folder,state,outcome,bank),parent,data[ci]['conversation'],env,extractor,nlp)
                    require(row['strict_passed'] is True and row['orphan_fact_ids']==[],'New non-allowlisted native deficit')
                row.update(source_kind=kind,bank=bp,native_store_inventory=row['native_inventory'],native_inventory=inventory,
                    original_write_status=original['write_status'],original_outcome=original['outcome'],
                    original_status=original['original_status'],write_status=idx.pointer(folder/'write_status.json'),
                    outcome=idx.pointer(folder/'write_outcome.json'),effective_status='reviewed_with_native_limitations' if kind=='reviewed_native_export' else 'complete',
                    provenance=provenance,checks={k:True for k in ('source_dates','generation_chain','fact_dense','fact_bm25','entity_dense')},
                    exceptions=[{'type':'declared_entity_link_deficit','fact_id':KNOWN[(ci,arm)][0]}] if kind=='reviewed_native_export' else [])
                row['checks']['entity_links']=kind!='reviewed_native_export'; rows.append(row)
        resources=parent['resources']
        bm25=compare_persisted_fact_vectors([Path(x['native_dir']) for x in rows],resources['snapshot'],resources['files_sha256'])
        require(bm25['facts_compared']==bm25['exact_matches']==sum(x['facts'] for x in rows),'Uniform fact BM25 audit incomplete')
        require(not any(g.call_count for g in guards),'Offline review attempted network')
        for p,h in list(idx.HASHES.items()): require(idx.sha(p)==h,'Audit source changed')
        return {'schema':SCHEMA,'status':'pending_recovery' if pending else 'effective40_structural_reviewed_semantic_pending',
                'created_utc':now(),'writer_manifest':review['parent_manifest'],'protocol_sha256':parent['protocol_sha256'],
                'failure_review':failure_review,'policy':review['policy'],'banks':rows,'pending':pending,
                'strict_passed':False,'model_calls':0,'network_attempts':0,'qa_gold_used':False,
                'offline_bm25_equivalence':bm25,'source_file_hashes':dict(idx.HASHES),
                'review_code':idx.pointer(__file__),'semantic_correctness_verified':False,
                'limitations':['Exact two declared entity deficits remain unrepaired.',
                    'Raw/source/date/ADD consistency does not certify fact semantics or every missing partial entity edge.']}


def _slots(rows, message):
    require(isinstance(rows,list) and len(rows)==40 and {(x.get('conversation'),x.get('arm')) for x in rows}==EXPECTED,message)
    return {(r['conversation'],r['arm']):r for r in rows}


def admit_gate(run, gate_path=None):
    """Read-only, fail-closed new gate; caller must supply published source review."""
    run=Path(run).resolve(); idx.HASHES.clear()
    with recovery.closed_parent_lock(run):
        parent=writer.manifest(run); retrieval.verify_new_config(run,parent)
        gate_path=inside(gate_path or run/'reviews/effective40_v1/all_bank_review.json',run/'reviews/effective40_v1')
        gate=idx.verify(gate_path)
        require(gate.get('schema')==GATE_SCHEMA and gate.get('status')=='passed'
                and gate.get('qa_gold_used') is False and gate.get('protocol_sha256')==parent['protocol_sha256']
                and gate.get('writer_manifest')==idx.pointer(run/'run_manifest.json'),'New effective40 bound gate required')
        review,_,items=failure_contract(run,gate['failure_review'])
        require(gate.get('policy')==review['policy'],'Gate policy binding differs')
        ap=inside(checked(gate['structural_audit']),run/'reviews/effective40_v1'); audit=idx.verify(ap)
        require(audit.get('schema')==SCHEMA and audit.get('status')=='effective40_structural_reviewed_semantic_pending'
                and audit.get('pending')==[] and audit.get('failure_review')==gate['failure_review']
                and audit.get('writer_manifest')==gate['writer_manifest'] and audit.get('protocol_sha256')==parent['protocol_sha256']
                and audit.get('model_calls')==0 and audit.get('network_attempts')==0 and audit.get('qa_gold_used') is False
                and audit.get('review_code')==idx.pointer(__file__),'Current complete structural provenance required')
        for p,h in audit['source_file_hashes'].items(): require(idx.sha(p)==h,'Audited source changed')
        ar=_slots(audit['banks'],'Full40 structural rows required'); gr=_slots(gate['banks'],'Full40 gate rows required')
        semantic=idx.verify(inside(checked(gate['semantic_review']),run))
        require(semantic.get('schema')=='gemini38-writer-source-semantic-review-v1'
                and semantic.get('status')=='reviewed_with_limitations' and semantic.get('qa_gold_used') is False
                and semantic.get('protocol_sha256')==parent['protocol_sha256']
                and semantic.get('structural_audit')==gate['structural_audit'],'New source-only semantic review required')
        sr=_slots(semantic['banks'],'Full40 semantic rows required'); admitted={}
        for slot in sorted(EXPECTED):
            ci,arm=slot; row=gr[slot]; a=ar[slot]; sem=sr[slot]
            require(row==a,'Gate row must preserve exact audited row (no silent rewriting)')
            bank=idx.verify(checked(row['bank'])); validate_fact_bank(bank,ci,arm,parent['protocol_sha256'])
            state=idx.bound(row['write_status']); outcome=idx.verify(checked(row['outcome']))
            require(outcome['write_status']==row['write_status'] and (state.get('conversation'),state.get('arm'),state.get('protocol_sha256'))==(*slot,parent['protocol_sha256'])
                    and state.get('status')==outcome.get('status') and not recovery.pid_alive(state['pid']),'Effective terminal provenance changed')
            checked(row['original_write_status']); checked(row['original_outcome'])
            require(row['provenance']['failure_review']==gate['failure_review']
                    and row['provenance']['scientific_parent_manifest']==gate['writer_manifest']
                    and row['provenance']['policy']==gate['policy'],'Per-bank scientific provenance differs')
            require(all(row['checks'].get(k) is True for k in ('source_dates','generation_chain','fact_dense','fact_bm25','entity_dense'))
                    and row.get('source_generation_persistence_passed') is True,'Required source/index checks missing')
            native=Path(row['native_dir']).resolve()
            with closed_native(native,state) as inventory:
                require(inventory==row['native_inventory'] and idx.native_inventory(native)==row['native_store_inventory'],'Audited native inventory changed')
            kind=row['source_kind']; original=run/f'conv_{ci}'/arm
            require(row['original_write_status']==idx.pointer(original/'write_status.json')
                    and row['original_outcome']==idx.pointer(original/'write_outcome.json'),'Original failure/status lineage differs')
            if kind=='reviewed_native_export':
                require(slot in KNOWN and row['strict_passed'] is False and row['checks']['entity_links'] is False
                        and row['orphan_fact_ids']==[KNOWN[slot][0]] and row['exceptions']==[{'type':'declared_entity_link_deficit','fact_id':KNOWN[slot][0]}]
                        and bank['native_inventory']==inventory and Path(bank['native_dir']).resolve()==native
                        and native==(original/'native').resolve(),'Exact retained native deficit/export required')
                require(bank['original_status_pointer']==row['original_write_status'] and bank['original_outcome_pointer']==row['original_outcome']
                        and bank['diagnosis']==items[slot]['diagnosis']==row['provenance']['diagnosis'],'Export provenance differs')
                validate_known_defect(ci,arm,state,outcome,idx.bound(items[slot]['diagnosis']),inventory,gate['writer_manifest'],bank['group_inputs'])
                inside(checked(row['bank']),run/'reviews/effective40_v1/exports')
            else:
                require(row['strict_passed'] is True and row['checks']['entity_links'] is True and row['orphan_fact_ids']==[]
                        and row['exceptions']==[] and state['status']=='complete' and state['bank']==outcome['bank']==row['bank'],
                        'Complete strict source required outside exact exports')
                if kind=='io_recovery_complete':
                    require(slot==(3,'session'),'Unknown recovery slot')
                    folder,rs,ro,rb,refs=_recovery_source(items[slot]['recovery_namespace'],parent,gate['failure_review'])
                    require(rs==state and ro==outcome and rb==bank and (folder/'native').resolve()==native
                            and all(row['provenance'].get(k)==v for k,v in refs.items()),'Recovery execution provenance differs')
                else:
                    require(kind=='original_complete' and slot not in FAILURES and native==(original/'native').resolve()
                            and row['bank']==idx.pointer(original/'bank.json'),'Unexpected effective source kind')
            require(sem.get('bank')==row['bank'] and sem.get('blocking_implementation_error') is False and sem.get('notes'),
                    'Semantic bank review missing or blocking')
            required={x['journal']['path'] for x in row.get('semantic_review_targets',[])}
            required.update(row['journals'][i]['journal']['path'] for i in (0,-1))
            refs={x['path']:x for x in bank['journal_files']}; reviewed=sem.get('reviewed_journals',[])
            require(required<={x['path'] for x in reviewed},'Required source/date/history/empty targets unreviewed')
            for ref in reviewed: require(refs.get(ref['path'])==ref,'Review cites another bank journal'); checked(ref)
            require(state['completed_groups']==state['expected_groups']==len(bank['journal_files']) and state['facts']==len(bank['facts']),
                    'Effective full journal/fact totals differ')
            admitted[slot]={'review':row,'bank':bank,'native':native}
        return parent,gate,admitted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--failure-review',required=True,type=Path)
    p.add_argument('--recovery-run',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); require(not a.output.exists(),'New output required; never overwrite an audit')
    result=audit_effective40(a.run_dir,idx.pointer(a.failure_review),recovery_run=a.recovery_run)
    save_new(a.output,result)
    print(json.dumps({'output':str(a.output),'status':result['status'],'banks':len(result['banks']),'pending':result['pending'],'model_calls':0}))


if __name__=='__main__':main()
