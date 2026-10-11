"""Authenticated envelope projection and UPDATE cache lineage; no clients on import."""
from __future__ import annotations
import copy,json,shutil,tempfile,types
from pathlib import Path
from tools import materialize_qualifier_repair_r09 as original
from tools import r09_native_update_audit as audit
old=original.old
require=old.require
ROOT=old.ROOT

def isolated(module,overrides=None):
    g=dict(vars(module))
    for name,value in list(g.items()):
        if isinstance(value,types.FunctionType) and value.__globals__ is vars(module):
            f=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
            f.__kwdefaults__=copy.copy(value.__kwdefaults__);g[name]=f
    g.update(overrides or {});return types.SimpleNamespace(**g)

def snapshot_body(value):
    require(set(value)=={'content_sha256','facts','entities','history'},'Snapshot envelope keys differ')
    body={k:v for k,v in value.items() if k!='content_sha256'}
    require(old.digest(body)==value['content_sha256'] and isinstance(body['facts'],dict)
        and isinstance(body['entities'],dict) and isinstance(body['history'],list),
        'Snapshot envelope seal or body differs')
    return copy.deepcopy(body)

def replay_exact(native,before,after,rows,resources):
    body=snapshot_body(before)
    return audit.replay_exact(native,body,after,rows,resources)

def byte_ref(ref):
    require(old.pointer(ref['path'])==ref,'Bound input bytes changed')

def audit_imported_embeddings(api_dir,source_api_dir,ci,new_rows,source_rows,source_input_refs):
    """Replay authenticated source producer plus actual new consumer occurrences offline.

    Temporary audit projection joins ledgers in time order, never copies physical events
    into the new run or its cost report. Frozen occurrence/raw/normalization rules apply.
    """
    api_dir,source_api_dir=Path(api_dir).resolve(),Path(source_api_dir).resolve()
    require(source_api_dir==ROOT/f'runs/qualifier_repair_r09_v1/materialization/conv_{ci}/embedding_api',
            'Foreign conversation import root')
    for ref in source_input_refs:byte_ref(ref)
    source_proof=audit.audit_embeddings(source_api_dir,ci,{'cheap':source_rows})
    def ledger(path):
        p=path/'api_calls.jsonl';data=old.shared_read(p) if p.exists() else b''
        return data,[json.loads(x) for x in data.decode('utf-8-sig').splitlines() if x.strip()]
    source_bytes,source_ledger=ledger(source_api_dir);new_bytes,new_ledger=ledger(api_dir)
    oldkeys={p.stem for p in (source_api_dir/'api_cache').glob('*.json')}
    keys={p.stem for p in (api_dir/'api_cache').glob('*.json')}
    intents={p.stem for p in (api_dir/'request_intents').glob('*.json')}
    imports={p.stem for p in (api_dir/'cache_lineage').glob('*.json')}
    require(intents.isdisjoint(imports) and intents|imports==keys and imports==keys&oldkeys,
        'New intent/import/raw inventory differs')
    lineage=[]
    for key in sorted(imports):
        target=old.pointer(api_dir/f'api_cache/{key}.json');source=old.pointer(source_api_dir/f'api_cache/{key}.json')
        lr=old.pointer(api_dir/f'cache_lineage/{key}.json');v=old.bound(lr)
        require(v=={'source_path':source['path'],'source_sha256':source['sha256'],
            'target_path':target['path'],'target_sha256':target['sha256'],'request_hash':key}
            and target['sha256']==source['sha256']
            and all(r.get('cache_hit') is True for r in new_ledger if r.get('request_hash')==key),
            'Imported raw bytes/lineage or zero-dispatch claim differs')
        lineage.append({'request_hash':key,'raw':target,'original_raw':source,'cache_lineage':lr,
            'original_intent':old.pointer(source_api_dir/f'request_intents/{key}.json')})
    with tempfile.TemporaryDirectory(prefix='r09-import-proof-') as td:
        temp=Path(td)
        for sub in ('api_cache','request_intents'):
            (temp/sub).mkdir()
            for p in (source_api_dir/sub).glob('*.json'):shutil.copyfile(p,temp/sub/p.name)
            for p in (api_dir/sub).glob('*.json'):
                target=temp/sub/p.name
                require(not target.exists() or old.sha(target)==old.sha(p),'Same-hash producer bytes conflict')
                shutil.copyfile(p,target)
        (temp/'api_calls.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in
            [*source_ledger,*new_ledger])+'\n',encoding='utf-8')
        joined=audit.audit_embeddings(temp,ci,{'cheap':source_rows,'jev':new_rows})
    consumers=[]
    for c in joined['consumers']:
        if c['context']['method']!='jev':continue
        consumers.append({**c,'raw':old.pointer(api_dir/f"api_cache/{c['hash']}.json")})
    producers=[]
    for p in joined['producers']:
        key=p['request_hash']
        if key not in keys:continue
        origin=source_api_dir if key in imports else api_dir
        producers.append({**p,'intent':old.pointer(origin/f'request_intents/{key}.json'),
            'imported':key in imports})
    require((old.shared_read(api_dir/'api_calls.jsonl') if (api_dir/'api_calls.jsonl').exists() else b'')==new_bytes
        and (old.shared_read(source_api_dir/'api_calls.jsonl') if (source_api_dir/'api_calls.jsonl').exists() else b'')==source_bytes,
        'UPDATE ledgers changed while auditing')
    for ref in source_input_refs:byte_ref(ref)
    return {'conversation':ci,'logical_invocations':len(consumers),'unique_signatures':len(keys),
        'physical_attempts':sum(r.get('cache_hit') is not True for r in new_ledger),
        'cache_consumers':sum(r.get('cache_hit') is True for r in new_ledger),
        'unknown_usage_rows':sum(not r.get('usage') for r in new_ledger if r.get('cache_hit') is not True),
        'ledger':old.pointer(api_dir/'api_calls.jsonl') if new_ledger else None,
        'producers':producers,'consumers':consumers,'imports':lineage,
        'original_physical_rows_copied':False,'original_proof':source_proof,
        'temporary_join_used_only_for_semantic_replay':True}
