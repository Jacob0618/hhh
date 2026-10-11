"""Offline R07 source/prefix preparation, score compilation and admission replay."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.write_omission_r07 import (PROTOCOL_SHA256,ARMS,POLICY,digest,request_hash,
    serialized_signature,validate_state,build_proposer_request,parse_proposal_response,
    stable_candidates,build_request,parse_scores,admit_candidates)
from jev_mem.data import session_envelopes
from tools import prepare_evidence_selection_r03 as prior
from tools.prepare_evidence_exchange_r01 import _baseline_jobs
from tools import prepare_evidence_coverage_r06 as coverage
from tools import mem0_regroup_firstpass_review_v2 as index
Reader,save_new,file_sha,now=prior.Reader,prior.save_new,prior.file_sha,prior.now
_canonical_ref=prior._canonical_ref
DEFAULT_SOURCE=prior.DEFAULT_SOURCE
RESEARCH_PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R07_write_omission_protocol_final_v1.json'
SCHEMA='write-omission-r07-selector-manifest-v1'
COMPILED_SCHEMA='write-omission-r07-score-manifest-v1'
ADMISSION_SCHEMA='write-omission-r07-admission-manifest-v1'
INPUT_SCHEMA='write-omission-r07-source-input-v1'
REQUEST_SCHEMA='write-omission-r07-request-v1'
SCORE_SCHEMA='write-omission-r07-score-result-v1'
AUDIT_SCHEMA='write-omission-r07-terminal-audit-v1'
SOURCE_SCHEMA='write-omission-r07-selected-inputs-v1'
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/write_omission_r07.py',
 'tools/prepare_write_omission_r07.py','tools/score_write_omission_r07.py',
 'tools/snapshot_write_omission_r07.py','tools/materialize_write_omission_r07.py',
 'tools/run_write_omission_qa_r07.py','tools/summarize_write_omission_r07.py',
 'jev_mem/data.py','jev_mem/mem0_adapter.py','tools/run_gemini38_writer_v1.py',
 'tools/prepare_gemini38_hybrid_effective40_v1.py','tools/run_planned_retrieval_v1.py',
 'tools/mem0_regroup_firstpass_review_v2.py','tools/review_gemini38_effective40_v1.py',
 'tools/prepare_evidence_coverage_r06.py','jev_mem/evidence_coverage_r06.py',
 'jev_mem/sdk20_type_adapter.py','jev_mem/providers.py',
 'upstream/mem0/mem0/memory/main.py','upstream/mem0/mem0/utils/entity_extraction.py',
 'upstream/mem0/mem0/utils/scoring.py')))


def _body(value):return {k:v for k,v in value.items()if k!='content_sha256'}
def _codes():return {p:file_sha(ROOT/p)for p in CODE}
def _inside(ref,run):return prior._inside(ref,run)


def _spec(reader):
    value=reader.read(RESEARCH_PROTOCOL,sealed=False);ref=reader.refs[str(RESEARCH_PROTOCOL.resolve())]
    if ref['sha256']!=PROTOCOL_SHA256:raise ValueError('Three-voted R07 protocol changed')
    for pointer in value['input_refs'].values():reader.raw_pointer(pointer['path'],ref=_canonical_ref(pointer))
    reuse=reader.bound(_canonical_ref(value['input_refs']['reuse_producer_qa_manifest']),sealed=False)
    prior._validate_qa_manifest(reuse)
    if _canonical_ref(reuse['baseline_reuse_proof'])!=_canonical_ref(value['input_refs']['baseline_reuse_proof']):
        raise ValueError('Original R01 proof producer binding changed')
    return value,deepcopy(ref)


def _dataset(reader,ref):
    """One byte buffer, one parse; QA fields never leave this offline lineage loader."""
    ref=_canonical_ref(ref);key=ref['path']
    if key not in reader.rows:
        raw=Path(key).read_bytes();actual={'path':key,'sha256':hashlib.sha256(raw).hexdigest()}
        if actual!=ref:raise ValueError('Frozen dataset changed')
        reader.rows[key]=json.loads(raw);reader.refs[key]=actual
    if reader.refs[key]!=ref or not isinstance(reader.rows[key],list)or len(reader.rows[key])!=10:
        raise ValueError('Exact ten-conversation source dataset required')
    return reader.rows[key]


def _history(reader,bank):
    path=Path(bank['native_dir'])/'history.sqlite';ref=reader.raw_pointer(path)
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        connection.row_factory=sqlite3.Row;connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
        if connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Native history integrity failed')
        rows=[dict(row)for row in connection.execute('SELECT * FROM history')];connection.rollback()
    finally:connection.close()
    if file_sha(path)!=ref['sha256']:raise ValueError('Native history changed during readonly audit')
    facts={f['id']:f for f in bank['facts']}
    if Counter(row['memory_id']for row in rows)!=Counter(facts.keys()):raise ValueError('Original history must have one ADD per fact')
    for row in rows:
        if (row['event']!='ADD'or row['is_deleted']or row['old_memory']is not None
            or row['new_memory']!=facts[row['memory_id']]['text']):raise ValueError('Original history contains rewrite/delete or wrong ADD')
    return ref


def _projection(turn,session,date):
    value={'turn_id':turn['dia_id'],'speaker':turn['speaker'],'session_id':session,
           'observation_date':date,'text':turn.get('text','')}
    if turn.get('blip_caption'):value['blip_caption']=turn['blip_caption']
    return value


def session_units(conversation,bank,groups,journals):
    """Validate exact original births and last-ten context before source projection."""
    envelopes=list(session_envelopes({'conversation':conversation}))
    if (groups.get('arm')!='session'or groups.get('conversation')!=bank['conversation']
        or groups['envelopes']!=envelopes or len(journals)!=len(envelopes)):
        raise ValueError('Original group envelopes/source identity differs')
    final={f['id']:f for f in bank['facts']}
    if len(final)!=len(bank['facts']):raise ValueError('Duplicate original canonical fact ID')
    projection={};turn_messages={};session_sequences={};history=[];birth={};units=[]
    for seq,(env,journal)in enumerate(zip(envelopes,journals)):
        sid=env['session_id'];date=env['observation_date'];session_sequences[sid]=seq
        if (journal.get('status')!='success'or journal.get('sequence')!=seq
            or journal.get('user_id')!=bank['user_id']or any(journal.get(k)!=v for k,v in env.items())
            or journal.get('history_context')!=history[-10:]):raise ValueError('Exact journal session/date/history differs')
        turns=conversation[sid]
        if len(turns)!=len(env['messages']):raise ValueError('Turn/message count differs')
        current=[]
        for turn,message in zip(turns,env['messages']):
            tid=turn['dia_id']
            if tid in projection:raise ValueError('Source turn ID is not globally unique within conversation')
            projection[tid]=_projection(turn,sid,date);turn_messages[tid]=message;current.append(projection[tid])
        previous=[]
        for h in journal['history_context']:
            tid=h['source_turn_id']
            if (tid not in projection or projection[tid]['session_id']!=h['session_id']
                or session_sequences[h['session_id']]>=seq or turn_messages[tid]!=h['message']):
                raise ValueError('Historical source turn missing/future/message-mismatched')
            previous.append(projection[tid])
        expected_ids=[h['source_turn_id']for h in journal['history_context']]
        for fact in journal['fact_records']:
            mid=fact['id'];scope=fact['input_scope']
            if (mid in birth or mid not in final or fact.get('event')!='ADD'
                or fact!={k:v for k,v in final[mid].items()if k!='embedding'}
                or scope['session_id']!=sid or scope['source_turn_ids']!=env['source_turn_ids']
                or scope['historical_turn_ids']!=expected_ids or fact['observation_date']!=date):
                raise ValueError('Unique original ADD birth/current/history/payload differs')
            birth[mid]=seq
        prefix=[{k:f[k]for k in ('id','text','observation_date')}for f in sorted(
            (f for f in bank['facts']if f['id']in birth),key=lambda f:(birth[f['id']],list(final).index(f['id'])))]
        state={'session_id':sid,'observation_date':date,'current_source':deepcopy(current),
               'historical_source':deepcopy(previous),'baseline_prefix':prefix}
        validate_state(state);units.append({'session_id':sid,'observation_date':date,'state':state,'sequence':seq})
        history.extend({'message':m,'session_id':sid,'source_turn_id':tid}for m,tid in zip(env['messages'],env['source_turn_ids']))
    if set(birth)!=set(final):raise ValueError('Not every final fact has an original ADD birth')
    return units


def _inputs(reader,spec):
    ref=_canonical_ref(spec['input_refs']['original_selected']);source=reader.bound(ref);jobs=_baseline_jobs(source)
    data=_dataset(reader,spec['input_refs']['dataset']);banks={};units=[];histories={};inventories={};groups_refs={}
    for old in jobs:
        ci=old['conversation'];bref=_canonical_ref(old['bank'])
        if ci in banks:
            if banks[ci]['ref']!=bref:raise ValueError('Conversation switched original bank')
            continue
        bank=reader.bound(bref)
        if (bank.get('complete')is not True or bank.get('arm')!='session'or bank.get('conversation')!=ci
            or bank['source_dataset_sha256']!=spec['input_refs']['dataset']['sha256']
            or bank['native_index_audit']['entity_facts_without_any_link']!=[]):raise ValueError('Wrong effective N0 bank/orphan scope')
        gref=_canonical_ref(bank['group_inputs']);groups=reader.bound(gref)
        if groups['dataset_sha256']!=spec['input_refs']['dataset']['sha256']:raise ValueError('Group dataset changed')
        jrefs=[_canonical_ref(r)for r in bank['journal_files']];journals=[reader.bound(j,sealed=False)for j in jrefs]
        current=session_units(data[ci]['conversation'],bank,groups,journals)
        for unit,jref in zip(current,jrefs):
            qid=f'c{ci}_s{unit["sequence"]+1:03d}'
            units.append({**unit,'query_id':qid,'conversation':ci,'bank':bref,'group_inputs':gref,'journal':jref})
        banks[ci]={'ref':bref,'bank':bank};groups_refs[str(ci)]=gref;histories[str(ci)]=_history(reader,bank)
        inventory=index.native_inventory(bank['native_dir'])
        for relative,sha in inventory.items():reader.raw_pointer(Path(bank['native_dir'])/relative,ref={'path':str((Path(bank['native_dir'])/relative).resolve()),'sha256':sha})
        inventories[str(ci)]=inventory
    if len(banks)!=10 or sum(len(v['bank']['facts'])for v in banks.values())!=1637 or len(units)!=272:
        raise ValueError('Expected original 10 banks, 1637 facts and 272 sessions')
    for old in jobs:
        byid={f['id']:f for f in banks[old['conversation']]['bank']['facts']}
        expected=[{k:byid[mid][k]for k in ('id','text','observation_date')}for mid in old['selected_ids']]
        if expected!=old['selected_facts']:raise ValueError('Original N0 selected facts changed')
    return source,ref,banks,units,groups_refs,histories,inventories


def _protocol(reader,spec,spec_ref,inputs):
    source,sref,banks,units,groups,histories,inventories=inputs
    retrieval_ref=_canonical_ref(source['retrieval_manifest']);retrieval=reader.bound(retrieval_ref,sealed=False)
    if retrieval.get('protocol_sha256')!=digest({k:v for k,v in retrieval.items()if k!='protocol_sha256'}):
        raise ValueError('Original retrieval manifest protocol digest changed')
    return {'schema':SCHEMA,'policy':deepcopy(POLICY),'research_protocol':spec_ref,
        'frozen_source':sref,'frozen_retrieval_manifest':retrieval_ref,
        'frozen_qa_run_manifest':reader.raw_pointer(prior.FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof':_canonical_ref(spec['input_refs']['baseline_reuse_proof']),
        'reuse_proof_qa_manifest':_canonical_ref(spec['input_refs']['reuse_producer_qa_manifest']),
        'source_dataset_sha256':source['source_dataset_sha256'],'frozen_protocol_sha256':source['protocol_sha256'],
        'official_metric_provenance':source['official_metric_provenance'],'bank_refs':{str(ci):v['ref']for ci,v in banks.items()},
        'group_input_refs':groups,'native_history_refs':histories,'original_native_inventory_by_conversation':inventories,
        'code_sha256':_codes(),'arms':list(ARMS),'questions_per_arm':1540,'session_count':272,'fact_count':1637}


def _request_value(unit,spec,psha,*,candidates=None):
    is_score=candidates is not None;state=unit['state']
    signature=build_request(state,candidates,spec)if is_score else build_proposer_request(state,spec)
    size=len(serialized_signature(signature));value={'schema':REQUEST_SCHEMA,'request_id':unit['query_id'],
        'query_id':unit['query_id'],'conversation':unit['conversation'],'session_id':unit['session_id'],
        'protocol_sha256':psha,'state':deepcopy(state),'signature':signature,'request_hash':request_hash(signature),
        'signature_utf8_bytes':size,'signature_utf8_byte_cap':131072,'dispatch_eligible':size<=131072,
        'logical_request_cap':1,'max_physical_attempts_per_phase':3,'input_tokens':None}
    if is_score:value.update({'candidates':deepcopy(candidates),'logical_noul_count':2*len(candidates)})
    return value


def _input_value(unit,psha):
    return {'schema':INPUT_SCHEMA,'protocol_sha256':psha,**deepcopy(unit),'state_sha256':digest(unit['state'])}


def prepare(run_dir):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve()or not run.name.startswith('write_omission_r07')or run.exists():
        raise FileExistsError('Fresh direct runs/write_omission_r07* namespace required')
    reader=Reader();spec,sref=_spec(reader);inputs=_inputs(reader,spec);protocol=_protocol(reader,spec,sref,inputs);psha=digest(protocol)
    units=inputs[3];prepared=[(u,_request_value(u,spec,psha))for u in units]
    run.mkdir(parents=True,exist_ok=False);items=[]
    for unit,request in prepared:
        qid=unit['query_id'];iref=save_new(run/'source_inputs'/f'{qid}.json',_input_value(unit,psha))
        reqref=save_new(run/'proposer_requests'/f'{qid}.json',request)
        items.append({'query_id':qid,'conversation':unit['conversation'],'session_id':unit['session_id'],
                      'source_input':iref,'request':reqref})
    ref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'phase':'proposer','status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':psha,'items':items,'request_refs':[i['request']for i in items],
        'summary':{'sessions':272,'oversize':sum(not r['dispatch_eligible']for u,r in prepared)},'model_calls_during_prepare':0})
    return {'manifest':ref,'sessions':272,'model_calls':0}


def _initial(reader,run):
    m=reader.read(run/'run_manifest.json');spec,sref=_spec(reader);inputs=_inputs(reader,spec)
    p=_protocol(reader,spec,sref,inputs)
    if (m.get('schema')!=SCHEMA or m.get('status')!='prepared'or m.get('phase')!='proposer'
        or m['protocol']!=p or m['protocol_sha256']!=digest(p)or len(m['items'])!=272):raise ValueError('Immutable R07 source/code/protocol differs')
    expected=[]
    for unit,item in zip(inputs[3],m['items']):
        qid=unit['query_id']
        if (item['query_id']!=qid or item['conversation']!=unit['conversation']or item['session_id']!=unit['session_id']
            or _inside(item['request'],run)!=run/'proposer_requests'/f'{qid}.json'
            or _inside(item['source_input'],run)!=run/'source_inputs'/f'{qid}.json'
            or _body(reader.bound(item['source_input']))!=_input_value(unit,m['protocol_sha256'])
            or _body(reader.bound(item['request']))!=_request_value(unit,spec,m['protocol_sha256'])):
            raise ValueError('Original source/prefix/request replay differs')
        expected.append(item['request'])
    if m.get('request_refs')!=expected:raise ValueError('Canonical proposer request refs differ')
    _no_orphans(run/'proposer_requests',expected);_no_orphans(run/'source_inputs',[i['source_input']for i in m['items']])
    return m,spec


def _no_orphans(directory,refs):return coverage._no_orphans(directory,refs)


def _audit_gate(reader,run,phase,manifest_ref,requests):
    path=run/'reviews'/f'{phase}_terminal_audit_v1.json';audit=reader.read(path)
    auditor=ROOT/'tools/score_write_omission_r07.py'
    if (audit.get('schema')!=AUDIT_SCHEMA or audit.get('phase')!=phase or audit.get('status')!='passed'
        or audit.get('manifest')!=manifest_ref or audit.get('unknown')!=[] or audit.get('requests_count')!=len(requests)
        or audit.get('codepins')!={'path':str(auditor.resolve()),'sha256':_codes()['tools/score_write_omission_r07.py']}):
        raise ValueError('Exact passed R07 terminal audit required')
    outcome=audit['outcome'];_inside(outcome,run);reader.bound(outcome)
    refs=audit.get('input_refs')
    if not isinstance(refs,list)or len({r['path']for r in refs})!=len(refs):raise ValueError('Complete unique audit input refs required')
    for ref in refs:
        if _inside(ref,run)==run/'api_calls.jsonl':raise ValueError('Ledger must be phase/hash projected')
        reader.bound(ref,sealed=False)
    hashes=sorted({reader.bound(ref)['request_hash']for ref in requests})
    stage='write_omission_proposer_r07'if phase=='proposer'else'write_omission_score_r07'
    if audit.get('ledger_projection')!=coverage.ledger_projection(reader,run/'api_calls.jsonl',stage,hashes):
        raise ValueError('Terminal audit ledger projection became stale')
    directory='proposer_responses'if phase=='proposer'else'responses';expected=[]
    for reqref in requests:
        req=reader.bound(reqref);ref=reader.pointer(run/directory/f'{req["request_id"]}.json')
        expected.append({**ref,'request_id':req['request_id'],'query_id':req['query_id'],'request_ref':reqref})
    if audit.get('response_refs')!=expected:raise ValueError('Every audited response must match canonical request sequence')
    controller=run/('execution_intent_proposer.json'if phase=='proposer'else'execution_intent_score_jev.json')
    required={manifest_ref['path'],outcome['path'],str(controller.resolve()),*[r['path']for r in requests],*[r['path']for r in expected]}
    if not required<={r['path']for r in refs}:raise ValueError('Audit omits required terminal inputs')
    failures=audit.get('known_terminal_failures')
    if not isinstance(failures,list)or any(f.get('technical_recovery_eligible')for f in failures):raise ValueError('Unreviewed technical recovery blocks next phase')
    return reader.pointer(path)


def _result(reader,run,reqref,phase,spec):
    req=reader.bound(reqref);directory='proposer_responses'if phase=='proposer'else'responses'
    path=run/directory/f'{req["request_id"]}.json';value=reader.read(path)
    if (value.get('schema')!=SCORE_SCHEMA or value.get('phase')!=phase
        or any(value.get(k)!=req[k]for k in ('request_id','query_id','conversation','session_id','protocol_sha256','request_hash'))):
        raise ValueError('Terminal R07 result identity differs')
    status=value.get('status');key='proposal_parse'if phase=='proposer'else'scores'
    if status not in ('complete','failed','not_dispatched'):raise ValueError('Unknown/nonterminal R07 state blocks')
    if value.get('actual_request_hash')not in (None,req['request_hash']):raise ValueError('Actual signature hash differs')
    rawref=value.get('raw');raw=None
    if rawref is not None:
        _inside(rawref,run);raw=reader.bound(rawref,sealed=False)
        if raw.get('request')!=req['signature']:raise ValueError('Raw signature differs')
    if status=='complete':
        if raw is None or not req['dispatch_eligible']:raise ValueError('Complete response needs dispatched raw')
        parsed=parse_proposal_response(raw['response'],req['state'],spec)if phase=='proposer'else parse_scores(raw['response'],req['signature'])
        if value.get(key)!=parsed:raise ValueError('Parsed terminal output differs from actual raw')
    else:
        if value.get(key)is not None or not value.get('reason'):raise ValueError('Unavailable parse must be explicit, never zero')
        if status=='not_dispatched'and(req['dispatch_eligible']or raw is not None):raise ValueError('Only cap overflow can skip nonempty requests')
        parsed=None
    return {'status':status,'parsed':parsed,'response':reader.pointer(path),'raw_response':rawref}


def _compiled(reader,run,m,spec):
    requests=[i['request']for i in m['items']]
    audit_ref=_audit_gate(reader,run,'proposer',reader.pointer(run/'run_manifest.json'),requests)
    rows=[]
    for item in m['items']:
        source=reader.bound(item['source_input']);result=_result(reader,run,item['request'],'proposer',spec)
        candidates=stable_candidates(result['parsed'],item['query_id'],source['state'])if result['parsed']is not None else []
        rows.append({**deepcopy(item),'proposer_request':item['request'],'proposer_result':result,
            'observation_date':source['observation_date'],'candidates':candidates})
    return rows,audit_ref


def compile_score(run_dir):
    run=Path(run_dir).resolve()
    if (run/'score_manifest.json').exists()or(run/'requests').exists():raise FileExistsError('Immutable compiled score namespace already exists')
    reader=Reader();m,spec=_initial(reader,run);rows,aref=_compiled(reader,run,m,spec)
    # Full source/audit/response replay completes before creating score artifacts.
    refs=[];items=[];counts=Counter()
    for row in rows:
        reqref=None
        if row['candidates']:
            source=reader.bound(row['source_input']);request=_request_value(source,spec,m['protocol_sha256'],candidates=row['candidates'])
            reqref=save_new(run/'requests'/f'{row["query_id"]}.json',request);refs.append(reqref)
            counts['logical_nouls']+=request['logical_noul_count'];counts['oversize']+=not request['dispatch_eligible']
        items.append({**row,'request':reqref});counts['candidates']+=len(row['candidates'])
    ref=save_new(run/'score_manifest.json',{'schema':COMPILED_SCHEMA,'phase':'jev','status':'compiled','created_utc':now(),
        'protocol':deepcopy(m['protocol']),'protocol_sha256':m['protocol_sha256'],'initial_manifest':reader.pointer(run/'run_manifest.json'),
        'proposer_terminal_audit':aref,'items':items,'request_refs':refs,'summary':dict(counts),'model_calls_during_compile':0})
    return {'manifest':ref,'sessions':272,'requests':len(refs),'summary':dict(counts),'model_calls':0}


def _score_manifest(reader,run,m,spec):
    value=reader.read(run/'score_manifest.json');rows,aref=_compiled(reader,run,m,spec);refs=[]
    if (value.get('schema')!=COMPILED_SCHEMA or value.get('phase')!='jev'or value.get('status')!='compiled'
        or value.get('protocol')!=m['protocol']or value.get('protocol_sha256')!=m['protocol_sha256']
        or value.get('initial_manifest')!=reader.pointer(run/'run_manifest.json')or value.get('proposer_terminal_audit')!=aref
        or len(value['items'])!=272):raise ValueError('R07 immutable score manifest differs')
    for row,item in zip(rows,value['items']):
        reqref=item['request']
        if _body(item)!={**row,'request':reqref}:raise ValueError('Compiled proposal lineage differs')
        if bool(row['candidates'])!=(reqref is not None):raise ValueError('Exactly one request per nonempty proposal required')
        if reqref is not None:
            if _inside(reqref,run)!=run/'requests'/f'{row["query_id"]}.json':raise ValueError('Score request namespace differs')
            source=reader.bound(row['source_input'])
            if _body(reader.bound(reqref))!=_request_value(source,spec,m['protocol_sha256'],candidates=row['candidates']):
                raise ValueError('JEV full source/candidate request replay differs')
            refs.append(reqref)
    if value['request_refs']!=refs:raise ValueError('Canonical JEV request refs differ')
    _no_orphans(run/'requests',refs)
    return value


def validate_prepared(run_dir,stage='proposer'):
    reader=Reader();run=Path(run_dir).resolve();m,spec=_initial(reader,run)
    if stage=='proposer':return m,[i['request']for i in m['items']]
    if stage=='score':
        value=_score_manifest(reader,run,m,spec);return value,value['request_refs']
    raise ValueError('Only proposer or score prepared phases')


def _admission_value(reader,run,m,spec,score):
    audit_ref=_audit_gate(reader,run,'jev',reader.pointer(run/'score_manifest.json'),score['request_refs'])
    rows=[];accepted={arm:{str(ci):[]for ci in range(10)}for arm in ('cheap','jev')}
    for item in score['items']:
        candidates=item['candidates'];result=_result(reader,run,item['request'],'jev',spec)if item['request']else None
        decision=admit_candidates(candidates,None if result is None else result['parsed'],status='complete'if result is None else result['status'])
        cheap=[c['candidate_id']for c in candidates];jev=decision['accepted_ids'];ci=str(item['conversation'])
        for arm,ids in [('cheap',cheap),('jev',jev)]:accepted[arm][ci].extend(deepcopy(c)for c in candidates if c['candidate_id']in ids)
        rows.append({**deepcopy(item),'score_request':item['request'],'score_result':result,
            'accepted':{'cheap':cheap,'jev':jev},'decision':decision})
    return {'schema':ADMISSION_SCHEMA,'status':'admitted','protocol':deepcopy(m['protocol']),
        'protocol_sha256':m['protocol_sha256'],'initial_manifest':reader.pointer(run/'run_manifest.json'),
        'score_manifest':reader.pointer(run/'score_manifest.json'),'jev_terminal_audit':audit_ref,
        'items':rows,'accepted_by_conversation':accepted,
        'summary':{arm:sum(len(v)for v in accepted[arm].values())for arm in ('cheap','jev')},'model_calls_during_admit':0}


def admit(run_dir):
    run=Path(run_dir).resolve();reader=Reader();m,spec=_initial(reader,run);score=_score_manifest(reader,run,m,spec)
    value=_admission_value(reader,run,m,spec,score);ref=save_new(run/'admission_manifest.json',value)
    return {'manifest':ref,'summary':value['summary'],'model_calls':0}


def validate_admission(path_or_run):
    path=Path(path_or_run).resolve();run=path if path.is_dir()else path.parent
    reader=Reader();m,spec=_initial(reader,run);score=_score_manifest(reader,run,m,spec);value=reader.read(run/'admission_manifest.json')
    if _body(value)!=_admission_value(reader,run,m,spec,score):raise ValueError('Frozen admission plan differs from source/raw gates')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','check','compile-score','check-score','admit','check-admission'));parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    if args.mode=='prepare':value=prepare(args.run_dir)
    elif args.mode=='compile-score':value=compile_score(args.run_dir)
    elif args.mode=='admit':value=admit(args.run_dir)
    elif args.mode=='check-admission':value={'status':'passed','sessions':len(validate_admission(args.run_dir)['items']),'model_calls':0}
    else:
        m,refs=validate_prepared(args.run_dir,'score'if args.mode=='check-score'else'proposer');value={'status':'passed','requests':len(refs),'model_calls':0}
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
