"""Independent, read-only admission of the complete twenty-row recovery QA map.

Consumes a draft without its input_audit pointer (avoids a circular hash). Never
publishes the input manifest, calls a model, or opens a Qdrant client. SQLite
point/history snapshots are deserialized into in-memory read connections only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mem0_sparse1540 as base
from tools import run_mem0_sparse1540_recovered_qa_v2 as qa
from tools import audit_mem0_sparse1540_retrieval as retrieval_audit
from tools.check_experiment_health import shared_read

POLICY=qa.POLICY
OUT=qa.OUT
NINE_PATH=ROOT/'research/mem0_sparse1540_retrieval_nine_conv_audit_2026-09-28.json'
NINE_SHA='b867b41f3448212e87672cc866825eea7a2c4ea43eee9b293b7041efd0bf69fc'
COVERAGE_PATH=ROOT/'research/mem0_sparse1540_twenty_bank_entity_coverage_2026-09-28.json'
COVERAGE_SHA='f7845228541049d940ec2ac90e0dbd23eeb4af09fa787a2c65869680d46eee06'
POLICY_PATH=ROOT/'research/mem0_sparse1540_index_recovery_policy_v2_2026-09-28.json'
POLICY_SHA='0321499a5abfc1573f6f894fd40ebcbebc0b5027477eb17f1da241c1ecbdb2b0'
_AUDITED_FILES={}


def captured(path):
    path=Path(path).resolve();raw=shared_read(path);digest=hashlib.sha256(raw).hexdigest()
    if str(path) in _AUDITED_FILES and _AUDITED_FILES[str(path)]!=digest:
        raise ValueError('Audited file changed during audit: '+str(path))
    _AUDITED_FILES[str(path)]=digest
    return raw
def sha(path):return hashlib.sha256(captured(path)).hexdigest()
def read(path):return json.loads(captured(path).decode('utf-8-sig'))
def bound(pointer):
    if sha(pointer['path'])!=pointer['sha256']:raise ValueError('Bound file changed: '+str(pointer['path']))
    return read(pointer['path'])
def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def equal_path(left,right):return Path(left).resolve()==Path(right).resolve()


def validate_map_shape(manifest,questions):
    if manifest.get('policy')!=POLICY:raise ValueError('Wrong recovery policy')
    if manifest.get('question_ids')!=[q['id'] for q in questions] or len(questions)!=1540:
        raise ValueError('Fixed1540 question cohort changed')
    rows=manifest.get('rows',[])
    if (len(rows)!=20 or any(type(r.get('conversation')) is not int for r in rows)
            or {(r['conversation'],r['arm']) for r in rows}!={(i,a) for i in range(10) for a in base.ARMS}):
        raise ValueError('Need exactly twenty unique arm/conversation rows')
    for row in rows:
        if (row.get('index_recovery') is not None)!=(row['conversation']==3):
            raise ValueError('Only both conv3 rows must be recovered')
    for key,value in {'provider_retries':2,'conversation_workers':8,'query_workers':4,'final_unique_facts':20}.items():
        if type(manifest.get(key)) is not int or manifest[key]!=value:raise ValueError('Resource/protocol budget changed')
    return rows


def canonical_points(path):
    """Reproduce Qdrant's loaded Record shape, without opening its local client."""
    from qdrant_client.http.models import Record
    raw=captured(path);db=sqlite3.connect(':memory:');db.deserialize(raw)
    try:
        output={}
        for (blob,) in db.execute('SELECT point FROM points'):
            point=pickle.loads(blob)  # Own locally generated Qdrant PointStruct.
            vector=point.vector
            if isinstance(vector,dict) and set(vector)=={''}:vector=vector['']
            record=Record(id=point.id,payload=point.payload,vector=vector).model_dump(mode='json')
            if str(point.id) in output:raise ValueError('Duplicate stored point')
            output[str(point.id)]=record
        return output,hashlib.sha256(raw).hexdigest()
    finally:db.close()


def validate_exact_union(oldfacts,newfacts,oldentities,newentities,fid,eid):
    if oldfacts!=newfacts:raise ValueError('Fact/vector/payload store changed')
    if any(fid in p['payload'].get('linked_memory_ids',[]) for p in oldentities.values()):
        raise ValueError('Target was not fully orphaned before repair')
    candidates=[i for i,p in oldentities.items() if ' '.join(p['payload']['data'].strip().lower().split())=='video games']
    if candidates!=[eid]:raise ValueError('Repair destination is not unique exact original entity')
    expected=deepcopy(oldentities)
    expected[eid]['payload']['linked_memory_ids']=sorted(set(expected[eid]['payload'].get('linked_memory_ids',[]))|{fid})
    if newentities!=expected:raise ValueError('Entity mutation exceeds one declared link union')
    return expected


def validate_bank_vector(exported,persisted,*,historical_export=False):
    if (len(exported)!=768 or len(persisted)!=768 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in exported+persisted)):
        raise ValueError('Invalid bank/store768 vectors')
    norms=[math.sqrt(sum(x*x for x in v)) for v in (exported,persisted)]
    difference=max(abs(x-y) for x,y in zip(exported,persisted))
    if any(abs(n-1)>1e-5 for n in norms) or (difference>1e-6 if historical_export else exported!=persisted):
        raise ValueError('Bank vector differs from allowed historical representation')
    return difference


def audit_repair(row,bank,coverage_row,policy,extract):
    arm=row['arm'];folder=OUT/'recovered'/arm/'conv_3'
    pointer=row['index_recovery']
    if not equal_path(pointer['path'],folder/'repair_result.json'):raise ValueError('Wrong repair result pointer')
    result=bound(pointer);intent=bound({'path':result['path'],'sha256':result['sha256']})
    if not equal_path(result['path'],folder/'repair_intent.json'):raise ValueError('Wrong repair intent pointer')
    target=next(x for x in policy['recovery_targets'] if x['arm']==arm and x['conversation']==3)
    fid,eid=target['orphan_fact_id'],target['exact_entity_id']
    if (intent['policy']!=POLICY or intent['arm']!=arm or intent['conversation']!=3 or
            intent['fact_id']!=fid or intent['entity_id']!=eid or intent['model_calls']!=0 or
            result['status']!='recovered_on_copy' or result['original_unchanged'] is not True):
        raise ValueError('Repair identity/status mismatch')
    if (result.get('arm')!=arm or result.get('conversation')!=3 or result.get('model_calls')!=0 or
            result.get('bank')!={'path':str(Path(row['bank_path']).resolve()),'sha256':row['bank_sha256']} or
            not equal_path(result['native_dir'],folder/'native')):
        raise ValueError('Repair result bank/native pointer differs from admitted row')
    if coverage_row['orphan_fact_ids']!=[fid]:raise ValueError('Repair not covered by pre-QA uniform orphan policy')
    if (intent['policy_binding']!={'path':str(POLICY_PATH),'sha256':POLICY_SHA} or
            intent['coverage']!={'path':str(COVERAGE_PATH),'sha256':COVERAGE_SHA}):
        raise ValueError('Repair intent policy/coverage binding changed')
    source=base.BASE/'conv_3' if arm=='session_rank' else base.conv_path(3)/'native'
    snapshot=folder/'original_snapshot';native=folder/'native'
    if not equal_path(intent['source_path'],source) or not equal_path(intent['original_snapshot'],snapshot):
        raise ValueError('Repair source/snapshot redirected')
    # Recheck every declared original and snapshot file; ignores only SDK lock.
    for name,expected in intent['source_hashes'].items():
        if isinstance(expected,dict):
            original={str(p.relative_to(source/name)):sha(p) for p in (source/name).rglob('*') if p.is_file() and p.name!='.lock'}
            copied={str(p.relative_to(snapshot/name)):sha(p) for p in (snapshot/name).rglob('*') if p.is_file() and p.name!='.lock'}
            if original!=expected or copied!=expected:raise ValueError('Original/snapshot inventory changed')
        elif sha(source/name)!=expected or sha(snapshot/name)!=expected:
            raise ValueError('Original/snapshot history changed')
    oldfacts,ofh=canonical_points(snapshot/'qdrant/collection/facts/storage.sqlite')
    newfacts,nfh=canonical_points(native/'qdrant/collection/facts/storage.sqlite')
    oldentities,oeh=canonical_points(snapshot/'qdrant/collection/facts_entities/storage.sqlite')
    newentities,neh=canonical_points(native/'qdrant/collection/facts_entities/storage.sqlite')
    expected=validate_exact_union(oldfacts,newfacts,oldentities,newentities,fid,eid)
    if (base.fingerprint(oldfacts)!=intent['fact_store_sha256'] or
            base.fingerprint(oldentities)!=intent['entity_store_before_sha256'] or
            base.fingerprint(expected)!=intent['entity_store_after_sha256'] or
            intent['before']!=oldentities[eid] or intent['intended_after']!=expected[eid]):
        raise ValueError('Repair intent store fingerprints differ from persisted bytes')
    if sha(native/'history.sqlite')!=sha(snapshot/'history.sqlite'):
        raise ValueError('History bytes changed in repair/retrieval')
    db=sqlite3.connect(':memory:');db.deserialize(captured(native/'history.sqlite'))
    try:
        for fact in bank['facts']:
            got=db.execute('SELECT new_memory,event,is_deleted FROM history WHERE memory_id=?',(fact['id'],)).fetchall()
            if got!=[(fact['text'],'ADD',0)]:raise ValueError('History ADD mismatch')
    finally:db.close()
    if {f['id'] for f in bank['facts']}!=set(oldfacts):raise ValueError('Bank facts differ from original store')
    max_export_difference=0.0
    if arm=='session_rank':
        source_manifest=read(base.OUT/'manifest.json')
        if (not equal_path(row['bank_path'],base.conv_path(3)/'session_bank.json') or
                row['bank_sha256']!=source_manifest['conversations'][3]['baseline_bank_sha256']):
            raise ValueError('A3 must retain the original baseline bank bytes')
    for f in bank['facts']:
        p=oldfacts[f['id']];v=p['vector'].get('') if isinstance(p['vector'],dict) else p['vector']
        if p['payload']['data']!=f['text']:raise ValueError('Bank fact text changed')
        difference=validate_bank_vector(f['embedding'],v,historical_export=arm=='session_rank')
        max_export_difference=max(max_export_difference,difference)
    if base.fingerprint(next(f for f in bank['facts'] if f['id']==fid))!=intent['fact_sha256']:
        raise ValueError('Repair target fact body changed')
    if arm=='regroup_rank':
        inputs=read(base.conv_path(3)/'group_inputs.json')['envelopes'];journals={}
        for path in (source/'session_audit').rglob('*.json'):
            record=read(path)
            if record.get('status')=='success':
                if record['session_id'] in journals:raise ValueError('Duplicate success journal')
                journals[record['session_id']]=(record,binding(path))
        if set(journals)!={x['session_id'] for x in inputs}:raise ValueError('Journal partition changed')
        original_facts=[];journal_bindings=[]
        for envelope in inputs:
            record,pointer=journals[envelope['session_id']]
            if any(record[k]!=v for k,v in envelope.items()):raise ValueError('Journal source/date changed')
            for fact in record['fact_records']:
                v=oldfacts[fact['id']]['vector'];v=v.get('') if isinstance(v,dict) else v
                original_facts.append({**fact,'embedding':v})
            journal_bindings.append(pointer)
        if bank['facts']!=original_facts or bank['journal_files']!=journal_bindings:
            raise ValueError('Recovered C bank is not exact ordered original journal facts')
    linked={mid for e in newentities.values() for mid in e['payload'].get('linked_memory_ids',[])}
    expected_ids={f['id'] for f in bank['facts'] if extract(f['text'])}
    if expected_ids-linked or linked-set(oldfacts):raise ValueError('Recovered orphan/scope gate failed')
    gate=result['native_index_audit']
    if (gate['history_adds_verified']!=len(oldfacts) or gate['entities']!=len(newentities) or
            gate['linked_memory_ids']!=sorted(linked) or gate['facts_with_local_entities']!=len(expected_ids) or
            gate['entity_facts_without_any_link']!=[]):raise ValueError('Reported sidecar gate not reproduced')
    return {'arm':arm,'conversation':3,'fact_id':fid,'entity_id':eid,'repair_result':binding(folder/'repair_result.json'),
        'repair_intent':binding(folder/'repair_intent.json'),'facts_unchanged':True,'history_unchanged':True,
        'only_declared_entity_union':True,'orphan_gate_passed':True,'facts':len(oldfacts),'entities':len(newentities),
        'historical_bank_export_max_absolute_difference':max_export_difference,
        'vector_comparison_scope':'Store before/after exact; only immutable A historical export vs persistent representation permits declared1e-6.',
        'store_bytes':{'old_facts':ofh,'new_facts':nfh,'old_entities':oeh,'new_entities':neh},'model_calls':0}


def recovery_consumer_audit(rows,arm,qid,expected,loader,seeds):
    """Reuse the reviewed validator with an explicit, lossless namespace mapping.

    Only exact matching v2/arm/conv3/retrieve contexts receive its legacy key.
    Every other context is marked foreign and cannot become a query consumer.
    Original ledger bytes/hashes and original context hashes remain in evidence.
    """
    mapped=[]
    for row in rows:
        new=deepcopy(row);ctx=new.get('context',{})
        matches=(ctx.get('architecture')==POLICY and ctx.get('method')==arm and
                 ctx.get('conversation')==3 and ctx.get('phase')=='retrieve')
        ctx['architecture']=base.POLICY if matches else 'foreign_original_context'
        ctx['method']='regroup_rank' if matches else ctx.get('method')
        new['context']=ctx;mapped.append(new)
    report=retrieval_audit.audit_consumers(mapped,3,qid,expected,loader,seeds)
    report['original_context']={'architecture':POLICY,'conversation':3,'method':arm,'phase':'retrieve','query_id':qid}
    report['original_context_rows_sha256']=base.fingerprint([r for r in rows if r.get('context')==report['original_context']])
    return report


def audit_recovered_retrieval(row,qs,extract):
    arm=row['arm'];folder=OUT/'recovered'/arm/'conv_3'
    status=read(folder/'retrieve_status.json')
    if status.get('status')!='complete' or status.get('done')!=199 or status.get('expected')!=199:
        raise ValueError('Recovered retrieval not complete')
    ledger_rows,ledger=retrieval_audit.ledger_rows(row['native_ledger_path'])
    if ledger['malformed_lines']:raise ValueError('Malformed recovery API ledger')
    qids={q['id'] for q in qs};specs={q['id']:retrieval_audit.native_expected_requests(q['question'],extract) for q in qs}
    expected_hashes={base.request_digest(s['signature']) for ss in specs.values() for s in ss}
    for r in ledger_rows:
        c=r.get('context',{})
        if (c.get('architecture')!=POLICY or c.get('conversation')!=3 or c.get('method')!=arm or
                c.get('phase')!='retrieve' or c.get('query_id') not in qids):raise ValueError('Unexpected recovery ledger context')
    lineage=read(folder/'cache_seed_lineage.json');source=base.BASE/'conv_3' if arm=='session_rank' else base.conv_path(3)/'native'
    if not equal_path(lineage['original_source_dir'],source):raise ValueError('Wrong seed origin')
    seeds=set()
    for seed in lineage['entries']:
        h=seed['request_hash']
        if (h in seeds or h not in expected_hashes or not equal_path(seed['source']['path'],source/'api_cache'/(h+'.json'))
                or not equal_path(seed['target']['path'],folder/'native/api_cache'/(h+'.json'))):raise ValueError('Invalid seed identity/path')
        original=bound(seed['source']);cached=bound(seed['target'])
        if original!=cached or seed['source']['sha256']!=seed['target']['sha256'] or base.request_digest(cached['request'])!=h:
            raise ValueError('Seed request/body/file differs')
        seeds.add(h)
    memo={}
    def loader(h):
        if h not in memo:
            p=Path(row['native_cache_dir'])/(h+'.json');memo[h]=(read(p),{'path':str(p),'file_sha256':sha(p)})
        return memo[h]
    reports=[]
    for q in qs:
        r=recovery_consumer_audit(ledger_rows,arm,q['id'],specs[q['id']],loader,seeds)
        reports.append({'query_id':q['id'],**r})
    if any(not r['admitted'] for r in reports):raise ValueError('Recovered embedding consumers not admitted')
    if sha(row['native_ledger_path'])!=ledger['sha256']:raise ValueError('Recovery ledger changed during audit')
    return {'arm':arm,'queries':reports,'ledger':ledger,'seed_lineage':binding(folder/'cache_seed_lineage.json'),
            'retrieve_status':binding(folder/'retrieve_status.json'),'admitted':True}


def audit(draft_path):
    _AUDITED_FILES.clear()
    draft_sha=sha(draft_path);manifest=read(draft_path);questions=base.questions()
    rows=validate_map_shape(manifest,questions);source=bound(manifest['source_manifest'])
    if not equal_path(manifest['source_manifest']['path'],base.OUT/'manifest.json') or source!=base.verify_manifest():
        raise ValueError('Frozen source protocol changed')
    if (not equal_path(manifest['recovery_policy']['path'],POLICY_PATH) or manifest['recovery_policy']['sha256']!=POLICY_SHA):
        raise ValueError('Draft does not reference reviewed symmetric recovery policy')
    policy=bound(manifest['recovery_policy']);coverage=bound({'path':str(COVERAGE_PATH),'sha256':COVERAGE_SHA})
    if policy['schema']!='mem0-sparse1540-orphan-index-recovery-policy-v2' or manifest['models']!=source['models']:
        raise ValueError('Recovery policy/models changed')
    codes=manifest['code_sha256']
    required={*qa.REQUIRED_CODE,'tools/audit_mem0_sparse1540_inputs_v2.py','tools/audit_mem0_sparse1540_retrieval.py',
              'tools/recover_mem0_sparse1540_index_v2.py','tools/recover_mem0_sparse1540_index_v3.py'}
    if not required<=set(codes):raise ValueError('Missing audit/recovery frozen code dependency')
    for relative,digest in codes.items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Current recovery code differs from draft')
    if not equal_path(policy['uniform_audit']['path'],COVERAGE_PATH) or policy['uniform_audit']['sha256']!=COVERAGE_SHA:
        raise ValueError('Uniform orphan coverage policy changed')
    coverage_rows={(r['conversation'],r['arm']):r for r in coverage['rows']}
    for row in coverage['rows']:
        for f in row['source_files']:
            if sha(f['source'])!=f['sha256'] or sha(f['copy'])!=f['sha256']:raise ValueError('Pre-repair original/copy bytes changed')
    nine=bound({'path':str(NINE_PATH),'sha256':NINE_SHA})
    if nine['admitted'] is not True or nine['manifest_sha256']!=manifest['source_manifest']['sha256']:
        raise ValueError('Original nine-conversation retrieval audit not admitted')
    # Revalidate every byte cited by the admitted request/record evidence.
    seen={}
    for conv in nine['conversations']:
        seen[conv['conversation']]=conv
        if sha(conv['ledger']['path'])!=conv['ledger']['sha256']:raise ValueError('Original retrieval ledger changed')
        for q in conv['queries']:
            for arm in base.ARMS:
                pointer=q[arm]
                if sha(pointer['path'])!=pointer['file_sha256']:raise ValueError('Original retrieval record changed')
            for r in q['consumers']['requests']:
                p=r['cache']
                if sha(p['path'])!=p['file_sha256']:raise ValueError('Original retrieval cache changed')
    if set(seen)!=set(range(10))-{3}:raise ValueError('Nine-conversation audit coverage changed')
    for p in nine['stderr']['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Original retrieval stderr changed')
    extract,extractor=retrieval_audit.entity_replayer();loaded=[];repairs=[];consumer_reports=[]
    recovery_manifest=read(OUT/'recovery_manifest.json')
    if (recovery_manifest['policy']!=POLICY or recovery_manifest['recovery_policy']!=manifest['recovery_policy']
            or recovery_manifest['source_manifest']!=manifest['source_manifest'] or recovery_manifest.get('model_calls')!=0
            or recovery_manifest['coverage']!={'path':str(COVERAGE_PATH),'sha256':COVERAGE_SHA}):
        raise ValueError('Recovery completion manifest binding differs')
    for row in rows:
        ci,arm=row['conversation'],row['arm'];qs=[q for q in questions if q['conversation']==ci]
        value=qa.validate_row(row,qs)
        if sha(row['bank_path'])!=row['bank_sha256']:raise ValueError('Bank changed during row validation')
        for qid,expected_hash in row['retrieval_files'].items():
            if sha(Path(row['retrieval_dir'])/(qid+'.json'))!=expected_hash:
                raise ValueError('Retrieval changed during row validation')
        if ci==3:
            repairs.append(audit_repair(row,value['bank'],coverage_rows[(ci,arm)],policy,extract))
            consumer_reports.append(audit_recovered_retrieval(row,qs,extract))
            if bound(row['index_recovery']) not in recovery_manifest['repairs']:raise ValueError('Repair absent from completion manifest')
            for retrieval in value['retrievals'].values():
                if retrieval['source'].get('index_recovery')!=row['index_recovery']:
                    raise ValueError('Recovered retrieval points to another index recovery')
        else:
            if coverage_rows[(ci,arm)]['orphan_fact_ids']:raise ValueError('Unrepaired orphan in reused input')
            original=seen[ci]
            expected_bank=original['baseline_bank_sha256'] if arm=='session_rank' else original['bank_sha256']
            if row['bank_sha256']!=expected_bank:raise ValueError('Reused bank differs from original admitted one')
        loaded.append({'conversation':ci,'arm':arm,'bank':{'path':row['bank_path'],'sha256':row['bank_sha256']},
            'questions':len(qs),'retrieval_files_sha256':base.fingerprint(row['retrieval_files']),
            'selection_count':20,'index_recovery':row['index_recovery'],'admitted':True})
    stderr=retrieval_audit.stderr_audit([OUT/'retrieve_recovered.stderr.log'])
    if not stderr['admitted']:raise ValueError('Recovery entity boost stderr not admitted')
    for p in stderr['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Recovery stderr changed during audit')
    if sha(draft_path)!=draft_sha:raise ValueError('Draft changed during input audit')
    return {'policy':POLICY,'admitted':True,'source_manifest_sha256':manifest['source_manifest']['sha256'],
        'recovery_policy_sha256':manifest['recovery_policy']['sha256'],'rows_sha256':base.fingerprint(rows),
        'question_ids_sha256':base.fingerprint(manifest['question_ids']),'model_calls':0,
        'captured_utc':datetime.now(timezone.utc).isoformat(),'draft':binding(draft_path),
        'code_sha256':codes,'audit_code_sha256':sha(__file__),'row_count':20,'question_arm_slots':3080,
        'rows':loaded,'repairs':repairs,'recovery_consumers':consumer_reports,'stderr':stderr,'extractor':extractor,
        'prior_retrieval_audit':{'path':str(NINE_PATH),'sha256':NINE_SHA},
        'coverage':{'path':str(COVERAGE_PATH),'sha256':COVERAGE_SHA},
        'recovery_manifest':binding(OUT/'recovery_manifest.json'),'qdrant_opened':False,'credentials_loaded':False,
        'audited_files':dict(sorted(_AUDITED_FILES.items())),
        'scope':'Twenty-row fixed cohort closed before QA. Both native conv3 indexes receive only the declared orphan union; '
                'other partial links remain unchanged. Raw embedding consumers are verified, not local entity boost scores. '
                'Supplied recovery stderr is bound to its current captured bytes; launcher capture provenance is separately recorded by the coordinator.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT/'runs'):parser.error('Use a new audit path outside runs')
    report=audit(args.draft)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'admitted':report['admitted'],'rows':20,'question_arm_slots':3080,'output':str(out),'sha256':sha(out),'model_calls':0}))


if __name__=='__main__':main()
