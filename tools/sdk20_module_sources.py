"""Thin, offline source adapter for experiments retaining the original SDK20 QA.

Never opens an original Qdrant client, constructs ModelAPI, or copies ledgers.
prepare_conv writes fresh copies only; it refuses any pre-existing conv directory.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.data import SOURCE_HASH,load_data,queries,session_envelopes
from jev_mem.evaluation import lexical_metrics
from tools import run_session_experiments as old
from tools import run_mem0_sparse1540 as transfer

BASE=ROOT/'runs/session_top20_v1'
TRANSFER=ROOT/'runs/mem0_sparse1540_v1'
RECOVERY=ROOT/'runs/technical_recovery_v1'
SEAL=ROOT/'research/main_batch_firstpass_seal_v1/seal.json'
SEAL_SHA='c6bae930c7d2bde85d5ae273ca816d8d96bc2692fbdfdd702b52a5e489e0a146'
SUMMARY=ROOT/'research/heartbeat_checks/20260927T2241/recovery_final/summary.json'
SUMMARY_SHA='35485ca600b851b6bfdf751dc4bad99664ccc49f4d4c71841bfc804434def540'
AUDIT=ROOT/'research/mem0_sparse1540_inputs_v2_audit_2026-09-28.json'
AUDIT_SHA='492e2d255dcf8a5ace2b21668cc40ae356bbbe463ee0442ffa4517132530d24f'
COVERAGE=ROOT/'research/mem0_sparse1540_twenty_bank_entity_coverage_2026-09-28.json'
COVERAGE_SHA='f7845228541049d940ec2ac90e0dbd23eeb4af09fa787a2c65869680d46eee06'
MANIFEST=TRANSFER/'manifest.json'
MANIFEST_SHA='c94e20e11baa278febdf722712bae915c75431656e7507996d988616f90813c4'
SCHEMA='sdk20-module-source-inputs-v1'
RESULT_KEYS=('prediction','judgment','token_f1','exact_match','selected_count','evidence_characters',
 'reader_protocol','reader_model','judge_model','source_evidence_metrics','source_metrics_note')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pointer(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def checked(path,expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Frozen source hash differs: '+str(path))
    return json.loads(raw)
def _dump_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

@lru_cache(maxsize=1)
def _metadata():
    seal=checked(SEAL,SEAL_SHA);summary=checked(SUMMARY,SUMMARY_SHA)
    audit=checked(AUDIT,AUDIT_SHA);coverage=checked(COVERAGE,COVERAGE_SHA)
    manifest=checked(MANIFEST,MANIFEST_SHA)
    if sha(ROOT/'data/locomo10.json')!=SOURCE_HASH or seal['source_dataset_sha256']!=SOURCE_HASH:
        raise ValueError('Dataset changed')
    for source in (seal,manifest):
        for name,digest in source['code_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen source code changed: '+name)
    if not audit['admitted'] or audit['row_count']!=20 or not coverage['all_twenty_scanned']:
        raise ValueError('Previous complete-bank admission is unavailable')
    entry=next(r for r in seal['runs'] if r['architecture']=='mem0')
    invpath=SEAL.parent/entry['inventory_path'];raw=invpath.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['inventory_sha256']:raise ValueError('Old inventory changed')
    inventory={r['path']:r['sha256'] for r in map(json.loads,raw.splitlines()) if r}
    return {'seal':seal,'summary':summary,'audit':audit,'coverage':coverage,'manifest':manifest,'old_inventory':inventory}

def _old_read(path):
    path=Path(path);rel=path.relative_to(BASE).as_posix()
    expected=_metadata()['old_inventory'].get(rel)
    if expected is None:raise ValueError('Original file absent from seal: '+rel)
    return checked(path,expected)

@lru_cache(maxsize=10)
def _old_conv(ci):
    if type(ci)is not int or ci not in range(10):raise ValueError('Invalid conversation')
    data=load_data()[ci]
    folder=BASE/f'conv_{ci}';facts=_old_read(folder/'facts.json');summary=_old_read(folder/'ingestion_summary.json')
    if not summary['complete'] or summary['facts']!=len(facts):raise ValueError('Old complete bank missing')
    return facts,summary,{q['id']:q for q in queries(data,ci)},list(session_envelopes(data))[-1]['observation_date']

def baseline_row(ci,q):
    """Return bound historical SDK20 success or explicit failed/null outcome."""
    facts,summary,questions,date=_old_conv(ci)
    expected=questions.get(q.get('id'))
    if expected is None or any(q.get(k)!=expected[k] for k in ('id','question','reference','category')):
        raise ValueError('Question differs from frozen cohort')
    qid=q['id'];folder=BASE/f'conv_{ci}';retpath=folder/'retrieval'/f'{qid}.json'
    retrieval=_old_read(retpath)
    old._verify_retrieval(retrieval,q,summary,_metadata()['old_inventory'][f'conv_{ci}/facts.json'])
    ids=[f['id'] for f in retrieval['sdk20']]
    if len(ids)!=20 or len(set(ids))!=20:raise ValueError('Original SDK20 budget differs')
    first=folder/'qa/sdk20'/f'{qid}.json'
    overlay=RECOVERY/f'execute/consumers/mem0/conv_{ci}/sdk20/{qid}.json'
    if first.exists():
        if overlay.exists():raise ValueError('Overlay must not replace old success')
        path=first;row=_old_read(path);kind='firstpass';status='complete'
        if row['request_fingerprint']!=old._qa_fingerprint(q,'sdk20',retrieval,date):
            raise ValueError('Old QA protocol mismatch')
    else:
        path=overlay;rel=path.relative_to(RECOVERY).as_posix()
        row=checked(path,_metadata()['summary']['overlay_file_sha256'][rel]);kind='technical_recovery';status=row['status']
        if row['seal_sha256']!=SEAL_SHA or row['architecture']!='mem0':raise ValueError('Wrong overlay source')
    if (row['query_id']!=qid or row['conversation']!=ci or row['method']!='sdk20'
        or any(row[k]!=q[k] for k in ('question','reference','category')) or row['selected_ids']!=ids):
        raise ValueError('Old result identity, question, or SDK20 selection differs')
    if status=='complete':
        result={k:deepcopy(row[k]) for k in RESULT_KEYS}
        if (not isinstance(result['prediction']['answer'],str) or type(result['judgment']['correct'])is not bool
            or result['reader_protocol']!='brief_answer_local_evidence_ids_v2'
            or result['reader_model']!='gemini-2.5-flash' or result['judge_model']!='gemini-3.5-flash-lite'
            or result['selected_count']!=20):raise ValueError('Old result contract differs')
        lexical=lexical_metrics(result['prediction']['answer'],q['reference'])
        if any(abs(lexical[k]-result[k])>1e-12 for k in ('token_f1','exact_match')):
            raise ValueError('Old lexical scores differ')
        error=None
    elif status=='failed':
        if row.get('error_type')!='ValueError' or row.get('error')!='Reader did not return an answer string':
            raise ValueError('Unreviewed baseline failure')
        result=None;error={'type':row['error_type'],'message':row['error']}
    else:raise ValueError('Unexpected baseline status')
    return {'query_id':qid,'conversation':ci,'status':status,'result':result,'selected_ids':ids,'error':error,
            'source':{**pointer(path),'kind':kind},'retrieval_source':pointer(retpath),'reference_date':date}

def _history_valid(path,facts):
    """Read bytes into memory so even SQLite never opens the source database."""
    con=sqlite3.connect(':memory:');con.deserialize(Path(path).read_bytes())
    try:
        rows=con.execute('SELECT memory_id,new_memory,event,is_deleted FROM history').fetchall()
        expected=sorted((f['id'],f['text'],'ADD',0) for f in facts)
        if sorted(rows)!=expected:raise ValueError('Native ADD history differs from fact bank')
    finally:con.close()

def load_banks(ci):
    """Load validated old session facts and split facts, original unrepaired stores."""
    facts,summary,_,_=_old_conv(ci);meta=_metadata();answer={}
    for kind,arm in (('session','session_rank'),('preseg','regroup_rank')):
        admitted=next(r for r in meta['audit']['rows'] if r['conversation']==ci and r['arm']==arm)
        bank=checked(admitted['bank']['path'],admitted['bank']['sha256'])
        if bank['conversation']!=ci or bank['arm']!=arm or bank['source_dataset_sha256']!=SOURCE_HASH or not bank['complete']:
            raise ValueError('Admitted bank identity differs')
        native=(BASE/f'conv_{ci}') if kind=='session' else (TRANSFER/f'conv_{ci}/native')
        if kind=='session':
            if bank['facts']!=facts:raise ValueError('Session bank no longer equals old Mem0 facts')
            user_id=summary['user_id'];group_source=None;journals=[]
        else:
            user_id=bank['user_id'];group_source=TRANSFER/f'conv_{ci}/group_inputs.json'
            gp=checked(group_source,bank['group_inputs_sha256'])
            journals=bank['journal_files'];flat=[]
            for envelope,p in zip(gp['envelopes'],journals):
                j=checked(p['path'],p['sha256'])
                if j['status']!='success' or any(j[k]!=v for k,v in envelope.items()):
                    raise ValueError('Sparse ADD source/date/input journal differs')
                flat.extend(j['fact_records'])
            if len(journals)!=len(gp['envelopes']):raise ValueError('Sparse journal count differs')
            bare=[{k:v for k,v in f.items() if k!='embedding'} for f in bank['facts']]
            if flat!=bare:raise ValueError('Sparse facts differ from original ADD journals')
        coverage=next(r for r in meta['coverage']['rows'] if r['conversation']==ci and r['arm']==arm)
        for point_file in coverage['source_files']:
            if sha(point_file['source'])!=point_file['sha256']:
                raise ValueError('Original unrepaired native store differs from prior coverage')
        _history_valid(native/'history.sqlite',bank['facts'])
        transfer.vector_ok(bank['facts'])
        answer[kind]={'schema':SCHEMA+'-bank','conversation':ci,'kind':kind,'facts':deepcopy(bank['facts']),
         'user_id':user_id,'source':deepcopy(admitted['bank']),'native_source':str(native),
         'native_state':'original_unrepaired','fact_support':'input_scope_unverified_not_gold',
         'source_dataset_sha256':SOURCE_HASH,'prior_admission':pointer(AUDIT),'prior_coverage':pointer(COVERAGE),
         'group_source':pointer(group_source) if group_source else None,'journal_files':journals,
         'original_orphan_fact_ids':coverage['orphan_fact_ids'],
         'fact_export_note':'conv3 preseg bank was exported during recovery; only unchanged facts are reused, never its repaired index' if ci==3 and kind=='preseg' else None}
    return answer

def _native_inventory(native):
    native=Path(native);out={}
    for p in (native/'qdrant').rglob('*'):
        if p.is_symlink():raise ValueError('Symlinks forbidden in source copy')
        if p.is_file() and p.name!='.lock':out[p.relative_to(native).as_posix()]=sha(p)
    hp=native/'history.sqlite'
    if hp.is_symlink():raise ValueError('Symlink history forbidden')
    out['history.sqlite']=sha(hp)
    return out

def prepare_conv(dest,ci):
    """Create dest/conv_i/{session,preseg}/native once, preserving source hashes."""
    dest=Path(dest).resolve();folder=dest/f'conv_{ci}'
    for oldroot in (BASE,TRANSFER,RECOVERY,ROOT/'runs/mem0_sparse1540_index_recovery_v2'):
        if dest==oldroot or oldroot in dest.parents or dest in oldroot.parents:
            raise ValueError('New run must be separate from immutable old runs')
    if folder.exists():raise FileExistsError('Conversation destination already exists; never overwrite or recopy')
    banks=load_banks(ci)
    inventories={kind:_native_inventory(b['native_source']) for kind,b in banks.items()}
    folder.mkdir(parents=True,exist_ok=False)
    _dump_new(folder/'prepare_intent.json',{'schema':SCHEMA,'conversation':ci,'source_inventories':inventories,'model_calls':0})
    result={'schema':SCHEMA,'conversation':ci,'source_dataset_sha256':SOURCE_HASH,'native_state':'original_unrepaired',
            'prior_input_admission':pointer(AUDIT),'baseline_seal':pointer(SEAL),'banks':{},'model_calls':0,'inherited_ledger_bytes':0}
    for kind,bank in banks.items():
        native=folder/kind/'native';source=Path(bank['native_source']);native.mkdir(parents=True)
        for rel,expected in inventories[kind].items():
            src=source/rel;dst=native/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():raise FileExistsError('Copy destination exists')
            shutil.copyfile(src,dst)
            if sha(dst)!=expected:raise ValueError('Native copy differs')
        if _native_inventory(source)!=inventories[kind] or _native_inventory(native)!=inventories[kind]:
            raise ValueError('Source changed during copy or copy incomplete')
        bank={**bank,'native_dir':str(native),'source_inventory':inventories[kind]}
        bankpath=folder/kind/'bank.json';_dump_new(bankpath,bank)
        result['banks'][kind]={'bank':pointer(bankpath),'native_dir':str(native),'user_id':bank['user_id'],
                               'facts':len(bank['facts']),'source_native':str(source),'source_inventory':inventories[kind]}
    _dump_new(folder/'source_manifest.json',result)
    return result

def readiness():
    rows=[baseline_row(ci,q) for ci,item in enumerate(load_data()) for q in queries(item,ci)]
    counts=Counter((r['source']['kind'],r['status']) for r in rows)
    expected={('firstpass','complete'):1518,('technical_recovery','complete'):7,('technical_recovery','failed'):15}
    if counts!=expected:raise ValueError('Frozen SDK20 source population changed')
    return {'schema':SCHEMA+'-readiness','questions':1540,'successful':1525,'failed':15,'model_calls':0,
            'counts':{a+':'+b:n for (a,b),n in counts.items()},
            'failed_query_ids':[r['query_id'] for r in rows if r['status']=='failed'],
            'source_summary':pointer(SUMMARY),'sources':[{'query_id':r['query_id'],'status':r['status'],'source':r['source']} for r in rows]}
