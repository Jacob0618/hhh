"""Exact, offline compatibility replay for original R07 entity vectors.

This is not a Memory.add or nearest-neighbor decision replay. The caller must
first bind the additions and their original raw/intent/ledger provenance. New
entity inserts are omitted: only native searches and persistence of old entity
payloads are replayed, on a closed temporary copy. A mismatch is never relaxed
with a tolerance. This module never opens a source with a writable Qdrant client.
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import tempfile

from tools import materialize_write_omission_r07 as original

SCHEMA = 'r07-original-entity-native-search-update-replay-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def _core(value):
    return {key: value[key] for key in ('facts', 'entities', 'history')}


def _plan(before, after, additions, user_id):
    require(isinstance(user_id, str) and bool(user_id), 'Replay user scope missing')
    require(isinstance(additions, list), 'Ordered additions required')
    for mid, old in before['facts'].items():
        require(after['facts'].get(mid) == old, 'Original fact differs')
    payloads = {}
    changed = []
    for eid, old in before['entities'].items():
        require(eid in after['entities'], 'Original entity missing')
        new = after['entities'][eid]
        op, np = copy.deepcopy(old['payload']), copy.deepcopy(new['payload'])
        ol, nl = op.pop('linked_memory_ids', None), np.pop('linked_memory_ids', None)
        require(isinstance(ol, list) and isinstance(nl, list)
                and len(ol) == len(set(ol)) and len(nl) == len(set(nl)), 'Entity links malformed')
        require(op == np and op.get('user_id') == user_id and set(ol) <= set(nl),
                'Original entity payload/scope/link changed')
        require(set(old['vector']) == set(new['vector']) and '' in old['vector'],
                'Original vector slots differ')
        require({k:v for k,v in old['vector'].items() if k != ''}
                == {k:v for k,v in new['vector'].items() if k != ''},
                'Original sparse or other vector changed')
        payloads[eid] = copy.deepcopy(old['payload'])
        if old['vector'][''] != new['vector']['']:
            changed.append(eid)
    seen_ids, seen_candidates, entities = set(), set(), set(before['entities'])
    plan = []
    for row in additions:
        require(isinstance(row, dict) and row.get('status') in ('added', 'duplicate_skipped'),
                'Unfinished/unknown addition')
        candidate = row.get('candidate', {})
        cid, mid = candidate.get('candidate_id'), row.get('memory_id')
        require(isinstance(cid, str) and cid and cid not in seen_candidates, 'Candidate identity duplicated')
        seen_candidates.add(cid)
        require(isinstance(mid, str) and mid in after['facts'], 'Addition memory missing')
        if row['status'] == 'duplicate_skipped':
            require(not row.get('entity_trace') and not row.get('expected_entity_keys'),
                    'Duplicate skip cannot have entity operations')
            require(after['facts'][mid]['payload'].get('data') == candidate.get('statement'),
                    'Duplicate skip statement differs')
            continue
        require(mid not in before['facts'] and mid not in seen_ids, 'New memory identity reused')
        seen_ids.add(mid)
        traces = row.get('entity_trace')
        require(isinstance(traces, list) and row.get('expected_entity_keys')
                == [t.get('normalized_text') for t in traces], 'Entity trace order/count differs')
        require(len(row['expected_entity_keys']) == len(set(row['expected_entity_keys'])),
                'Duplicate normalized entity operation')
        for trace in traces:
            require(isinstance(trace, dict) and trace.get('memory_id') == mid,
                    'Entity trace memory context differs')
            eid, kind = trace.get('selected_entity_id'), trace.get('selection_kind')
            require(isinstance(eid, str) and eid in after['entities']
                    and kind in ('native_exact', 'native_semantic', 'native_insert'),
                    'Unknown entity trace kind/target')
            require(isinstance(trace.get('text'), str) and bool(trace['text'])
                    and isinstance(trace.get('normalized_text'), str) and bool(trace['normalized_text']),
                    'Entity trace text missing')
            require(after['entities'][eid]['payload'].get('user_id') == user_id
                    and mid in after['entities'][eid]['payload'].get('linked_memory_ids', []),
                    'Entity trace link or scope missing')
            if kind == 'native_insert':
                require(eid not in entities, 'Insert targets preexisting entity')
                entities.add(eid)
            else:
                require(eid in entities, 'Match targets entity before its insertion')
            update_payload = None
            if eid in payloads:
                links = payloads[eid]['linked_memory_ids']
                if mid not in links:
                    links.append(mid)
                    update_payload = copy.deepcopy(payloads[eid])
            plan.append({'search': kind != 'native_exact', 'text': trace['text'],
                         'target': eid, 'payload': update_payload})
    for eid, expected in payloads.items():
        require(expected == after['entities'][eid]['payload'],
                'Original entity final payload not explained by ordered traces')
    require(set(after['facts']) == set(before['facts']) | seen_ids, 'Extra or missing fact outside additions')
    require(set(after['entities']) == entities, 'Extra or missing entity outside traces')
    return plan, changed


def replay_original_entities(original_native_path, before, after, additions, user_id):
    """Return an immutable-input-bound exact replay report, or raise ValueError.

    The report does not replace the caller's complete new-fact/history/raw/API
    audit. It does not establish source entailment or semantic match correctness.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.local import distances, local_collection, persistence
    from mem0.vector_stores.qdrant import Qdrant

    plan, changed = _plan(before, after, additions, user_id)
    source = Path(original_native_path).resolve()
    inventory = original.inventory(source)
    require(original.snapshot(source) == _core(before), 'Before snapshot differs from original source')
    code = {str(Path(inspect.getfile(obj)).resolve()): original.sha(inspect.getfile(obj))
            for obj in (Qdrant, QdrantClient, distances, local_collection, persistence)}
    code[str(Path(__file__).resolve())] = original.sha(__file__)
    code[str(Path(original.__file__).resolve())] = original.sha(original.__file__)
    with tempfile.TemporaryDirectory(prefix='r07-native-exact-') as temp:
        target = Path(temp)/'native'
        original.copy_closed(source, target, inventory)
        client = None
        try:
            client = QdrantClient(path=str(target/'qdrant'))
            store = Qdrant.__new__(Qdrant)
            store.client = client
            store.collection_name = 'facts_entities'
            if before['entities']:
                # Cosine normalization operates on the complete dense matrix,
                # independently of query values. Never use a hand-written norm.
                query = list(next(iter(before['entities'].values()))['vector'][''])
                for operation in plan:
                    if operation['search']:
                        store.search(query=operation['text'], vectors=list(query), top_k=1,
                                     filters={'user_id': user_id})
                    if operation['payload'] is not None:
                        store.update(vector_id=operation['target'], vector=None,
                                     payload=operation['payload'])
        finally:
            if client is not None:
                client.close()
        replay = original.snapshot(target)
        for eid in before['entities']:
            require(replay['entities'][eid] == after['entities'][eid],
                    'Original entity differs from exact native search/update replay: '+eid)
        require(replay['facts'] == before['facts'] and replay['history'] == before['history'],
                'Numeric diagnostic modified original fact/history copy')
    require(original.inventory(source) == inventory, 'Original source bytes changed during replay')
    require(all(original.sha(path) == sha for path, sha in code.items()), 'Replay code changed during audit')
    return {'schema': SCHEMA, 'native_search_update_replay_exact': True,
            'original_entity_count': len(before['entities']), 'changed_dense_entity_ids': sorted(changed),
            'trace_operations': len(plan), 'semantic_search_operations': sum(x['search'] for x in plan),
            'old_entity_payload_updates': sum(x['payload'] is not None for x in plan),
            'original_native_path': str(source), 'original_native_inventory': inventory,
            'before_snapshot_sha256': original.digest(before), 'after_snapshot_sha256': original.digest(after),
            'ordered_additions_sha256': original.digest(additions), 'user_id': user_id,
            'native_code_sha256': code, 'new_model_calls': 0, 'epsilon_used': False,
            'full_memory_add_replayed': False, 'new_entity_insert_replayed': False,
            'source_semantics_certified': False, 'semantic_match_decisions_recomputed': False,
            'provenance_requirement': 'Caller binds original trace/addition/raw/intent/ledger bytes and fully audits all new records.'}
