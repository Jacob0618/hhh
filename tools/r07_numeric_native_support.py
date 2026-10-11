"""R07 technical continuation helpers: isolated frozen functions and exact cache lineage.

No clients, model calls, source writes or numerical epsilon admission on import.
"""
from __future__ import annotations
import copy
from collections import Counter
from pathlib import Path
import types
from tools import materialize_write_omission_r07 as old


def isolated(overrides=None):
    """Clone function globals, including calls between frozen functions, without mutation."""
    space = dict(vars(old))
    for name, value in list(space.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(old):
            fn = types.FunctionType(value.__code__, space, name, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            space[name] = fn
    space.update(overrides or {})
    return types.SimpleNamespace(**space)


def numeric_delta(original_native, before, after, additions, user_id):
    from tools.r07_native_numeric_replay import replay_original_entities
    report = replay_original_entities(original_native, before, after, additions, user_id)
    old.require(report.get('native_search_update_replay_exact') is True,
                'Exact native numeric trajectory was not certified')
    comparison = copy.deepcopy(after)
    # Only the audit object is projected. The actual native store is never modified.
    for eid, point in before['entities'].items():
        comparison['entities'][eid]['vector'][''] = copy.deepcopy(point['vector'][''])
    result = old.validate_delta(before, comparison, additions, user_id)
    return {**result, 'numeric_trajectory': report,
            'actual_storage_vectors_replaced': False, 'epsilon_admission': False}


def audit_import(api_dir, source_api_dir, text, context, ledger):
    """Imported target has no local intent/physical producer, only a declared consumer."""
    api_dir, source_api_dir = Path(api_dir).resolve(), Path(source_api_dir).resolve()
    signature = old.add_signature(text); key = old.request_hash(signature)
    lineage_path = api_dir/f'cache_lineage/{key}.json'
    if not lineage_path.exists():
        return old.audit_embedding(api_dir, text, context, ledger)
    lineage_ref = old.pointer(lineage_path); lineage = old.bound(lineage_ref)
    raw_ref = old.pointer(api_dir/f'api_cache/{key}.json')
    source_ref = old.pointer(source_api_dir/f'api_cache/{key}.json')
    old.require(lineage == {'source_path': source_ref['path'], 'source_sha256': source_ref['sha256'],
        'target_path': raw_ref['path'], 'target_sha256': raw_ref['sha256'], 'request_hash': key}
        and raw_ref['sha256'] == source_ref['sha256']
        and not (api_dir/f'request_intents/{key}.json').exists(),
        'Imported embedding target/source/lineage conflicts')
    rows = [r for r in ledger if r['request_hash'] == key]
    old.require(rows and all(r.get('cache_hit') is True for r in rows)
        and any(r['context'] == context for r in rows), 'Imported request has local dispatch or no consumer')
    source_ledger = old.embedding_ledger(source_api_dir, context['conversation'])
    intent = old.read(source_api_dir/f'request_intents/{key}.json')
    vector, proof = old.audit_embedding(source_api_dir, text, intent['context'], source_ledger)
    target = old.bound(raw_ref)
    old.require(target == old.bound(source_ref) and target['request'] == signature,
                'Imported raw response differs from exact original producer')
    return vector, {**proof, 'raw': raw_ref, 'original_raw': source_ref,
        'cache_lineage': lineage_ref, 'consumer_context': context,
        'imported_exact_request': True, 'new_physical_attempts': 0}


def audit_inventory(api_dir, source_api_dir, ci, by_arm):
    api_dir = Path(api_dir).resolve(); ledger = old.embedding_ledger(api_dir, ci)
    declared = Counter(); texts = {}; contexts = {}
    for arm in old.NEW_ARMS:
        for row in by_arm.get(arm, []):
            if row['status'] != 'added': continue
            c = row['candidate']; context = {'architecture': old.SCHEMA, 'conversation': ci,
                'method': arm, 'phase': 'write_supplement', 'candidate_id': c['candidate_id'],
                'group_id': c['source_session_id']}
            cd = old.digest(context); contexts[cd] = context
            for text in [c['statement'], *[t['text'] for t in row['entity_trace']]]:
                key = old.request_hash(old.add_signature(text)); texts[key] = text
                declared[(key, cd)] += 1
    actual = Counter((r['request_hash'], old.digest(r['context'])) for r in ledger
                     if r.get('cache_hit') is True or r.get('status') == 200)
    old.require(actual == declared and all((r['request_hash'], old.digest(r['context'])) in declared
        for r in ledger), 'New ledger has orphan/missing invocation or physical context')
    keys = set(texts)
    intents = {p.stem for p in (api_dir/'request_intents').glob('*.json')}
    imports = {p.stem for p in (api_dir/'cache_lineage').glob('*.json')}
    old.require(intents.isdisjoint(imports) and intents | imports == keys
        and {p.stem for p in (api_dir/'api_cache').glob('*.json')} == keys,
        'New raw/intent/import inventory is incomplete or orphaned')
    proofs = []
    for key, text in sorted(texts.items()):
        rows = [r for r in ledger if r['request_hash'] == key]
        physical = [r for r in rows if r.get('cache_hit') is not True]
        if key in intents:
            intent = old.read(api_dir/f'request_intents/{key}.json')
            old.require(physical and all(r['context'] == intent['context'] for r in physical),
                        'New same-hash physical producer is not unique')
            producer_index = next((i for i, r in enumerate(ledger) if r in physical and r.get('status') == 200), None)
            old.require(producer_index is not None and all(i > producer_index for i, r in enumerate(ledger)
                if r['request_hash'] == key and r.get('cache_hit') is True),
                'Local cache consumer precedes successful producer')
        for cd in sorted({old.digest(r['context']) for r in rows}):
            _, proof = audit_import(api_dir, source_api_dir, text, contexts[cd], ledger)
            proofs.append(proof)
    return {'conversation': ci, 'declared_logical_invocations': sum(declared.values()),
        'unique_signatures': len(keys), 'physical_attempts': sum(r.get('cache_hit') is not True for r in ledger),
        'cache_consumers': sum(r.get('cache_hit') is True for r in ledger),
        'imported_signatures': len(imports), 'local_signatures': len(intents),
        'known_usage_rows': sum(bool(r.get('usage')) for r in ledger if r.get('cache_hit') is not True),
        'usage_unavailable_not_zero': sum(not bool(r.get('usage')) for r in ledger if r.get('cache_hit') is not True),
        'ledger': old.pointer(api_dir/'api_calls.jsonl') if ledger else None,
        'raw_producer_bindings': proofs, 'original_physical_rows_copied': False}
