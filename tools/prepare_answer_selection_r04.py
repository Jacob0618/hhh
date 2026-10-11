"""Offline R04 extraction, exact Reader provenance replay, Choice sealing."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.answer_selection_r04 import (ARMS,ROUTES,POLICY,PROTOCOL_SHA256,digest,
    request_hash,serialized_signature,build_request,presentation,union_records,
    cheap_choice,parse_choice,choose_jev)
from tools import prepare_evidence_selection_r03 as prior
from tools import run_evidence_selection_qa_r03 as qa
from tools import summarize_evidence_selection_r03 as audit
from jev_mem.gemini38_provider_v1 import generation_body,response_text
from jev_mem.evaluation_gemini38_v1 import reader_input,READER_SYSTEM,MODEL,QA_MAX_OUTPUT,READER_STAGE
from tools.summarize_sdk20_modules import fingerprint
READER_LEDGER=audit._scope['ledger']

SCHEMA='answer-selection-r04-selector-manifest-v1'
REQUEST_SCHEMA='answer-selection-r04-request-v1'
SCORE_SCHEMA='answer-selection-r04-score-result-v1'
SOURCE_SCHEMA='answer-selection-r04-selected-answers-v1'
DECISION_SCHEMA='answer-selection-r04-decision-v1'
RESEARCH_PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R04_answer_selection_protocol_final_v1.json'
# Evaluation/summary is deliberately not a dependency of the scoring process.
CODE=tuple(dict.fromkeys((*prior.CODE,
    'tools/summarize_gemini38_seven_arm_v1.py','tools/summarize_sdk20_plain_qa_v1.py',
    'tools/summarize_mem0_regroup_plain_qa_v1.py','tools/summarize_sdk20_modules.py',
    'tools/summarize_evidence_exchange_r01.py','tools/summarize_evidence_selection_r03.py',
    'jev_mem/answer_selection_r04.py',
    'tools/prepare_answer_selection_r04.py','tools/score_answer_selection_r04.py',
    'tools/snapshot_answer_selection_r04.py')))
save_new=prior.save_new
now=prior.now
file_sha=prior.file_sha

class Reader(prior.Reader):
    """One byte read per path, shared by source validation and Reader replay."""
    def __init__(self):
        super().__init__();self.bytes={};self.sealed_checked=set()
    def raw(self,path):
        path=str(Path(path).resolve())
        if path not in self.bytes:
            b=Path(path).read_bytes();self.bytes[path]=b
            self.refs[path]={'path':path,'sha256':hashlib.sha256(b).hexdigest()}
        return self.bytes[path]
    def read(self,path,*,ref=None,sealed=True):
        key=str(Path(path).resolve());b=self.raw(key)
        if key not in self.rows:
            value=json.loads(b)
            if not isinstance(value,dict):raise ValueError('Object artifact required')
            self.rows[key]=value
        value=self.rows[key]
        if sealed and key not in self.sealed_checked:
            if value.get('content_sha256')!=digest({k:v for k,v in value.items() if k!='content_sha256'}):
                raise ValueError('Content digest changed: '+key)
            self.sealed_checked.add(key)
        if ref is not None and self.refs[key]!=ref:raise ValueError('Pinned path/SHA changed: '+key)
        return value
    def raw_pointer(self,path,*,ref=None):
        key=str(Path(path).resolve());self.raw(key)
        if ref is not None and self.refs[key]!=ref:raise ValueError('Pinned raw path/SHA changed: '+key)
        return deepcopy(self.refs[key])

class Snapshot:
    """File-only replay adapter using the same byte cache as the source reader."""
    def __init__(self,reader):self.reader=reader;self.files={}
    def raw(self,path):
        p=Path(path).resolve();b=self.reader.raw(p);self.files[p]=self.reader.refs[str(p)]['sha256'];return b
    def read(self,path):
        self.raw(path);return self.reader.read(path,sealed=False)
    def sha(self,path):self.raw(path);return self.files[Path(path).resolve()]

def _codes():return {p:file_sha(ROOT/p) for p in CODE}

def _spec(reader):
    spec=reader.read(RESEARCH_PROTOCOL,sealed=False);ref=reader.raw_pointer(RESEARCH_PROTOCOL)
    if ref['sha256']!=PROTOCOL_SHA256:raise ValueError('Three-voted R04 protocol changed')
    return spec,ref

def _reader_signature(job):
    # No gold/reference answer is read by this builder, or sent to a model.
    payload=reader_input(job['question'],job['selected_facts'],reference_date=job['reference_date'])
    body=generation_body(json.dumps(payload,ensure_ascii=False),system=READER_SYSTEM,
                         max_output=QA_MAX_OUTPUT)
    return {'provider':'gemini','model':MODEL,'operation':'generateContent','body':body}

def _validate_parent_source(reader,source_ref):
    source=reader.bound(source_ref);run=Path(source['parent_run']).resolve()
    manifest=prior._manifest(reader,run)
    if (source.get('schema')!=prior.SOURCE_SCHEMA or source.get('status')!='sealed'
        or source.get('selector_manifest')!=reader.pointer(run/'run_manifest.json')
        or source.get('protocol_sha256')!=manifest['protocol_sha256']
        or source.get('selection_protocol')!=manifest['protocol']['research_protocol']
        or any(source[k]!=manifest['protocol'][k] for k in ('frozen_source','frozen_qa_run_manifest',
            'frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256',
            'official_metric_provenance','code_sha256'))):
        raise ValueError('Frozen R03 source/manifest differs')
    expected=[];pools={}
    for old,method,value in prior._expected_records(reader,run,manifest):
        sp=run/'selections'/method/f'{old["query_id"]}.json';actual=reader.read(sp)
        if {k:v for k,v in actual.items() if k!='content_sha256'}!=value:
            raise ValueError('Frozen R03 selection replay differs')
        expected.append(prior._job(old,method,value,reader.pointer(sp),manifest['protocol_sha256']))
        if method=='baseline':pools[old['query_id']]=reader.bound(value['proposal'])['frozen_pool']
    if source['jobs']!=expected or Counter(j['method'] for j in expected)!={a:1540 for a in prior.ARMS}:
        raise ValueError('Frozen R03 source cohort differs')
    return source,pools

def _qa_parent(reader,ref,source_ref):
    m=reader.bound(ref,sealed=False)
    if (m.get('schema')!=qa.SCHEMA or m.get('phase_kind')!='firstpass'
        or m.get('source_manifest')!=source_ref
        or m.get('protocol_sha256')!=digest({k:v for k,v in m.items() if k!='protocol_sha256'})
        or len(m.get('jobs',[]))!=4620 or len({j['job_id'] for j in m['jobs']})!=4620
        or any(file_sha(ROOT/p)!=s for p,s in m['code_sha256'].items())):
        raise ValueError('Frozen complete R03 QA identity/code differs')
    out=reader.read(Path(ref['path']).parent/'run_outcome.json')
    if out.get('terminal') is not True or out.get('protocol_sha256')!=m['protocol_sha256'] or out.get('schema')!=m['schema']:
        raise ValueError('R03 must be recorded whole-phase terminal')
    return m

def _published_reuse_proof(reader,m):
    # qa.proof_for -> prior._load_proof -> sdk bound authenticates exact file SHA.
    # This published R03 proof copied an old content_sha before adding revalidation;
    # its authoritative seal is the immutable QA manifest pointer, not that stale
    # inherited field. This narrow exception never changes general sealed reads.
    proof=reader.bound(m['baseline_reuse_proof'],sealed=False)
    cache=getattr(reader,'_published_proof_cache',{})
    key=m['baseline_reuse_proof']['sha256']
    if key in cache:return cache[key]
    rv=proof.get('reuse_revalidation',{})
    if (proof.get('schema')!=qa.SCHEMA+'-reuse-proof' or proof.get('model_calls_during_proof')!=0
        or rv.get('new_model_calls')!=0 or rv.get('current_reader_and_judge_signatures')!=1540
        or rv.get('proof')!=reader.raw_pointer(prior.FROZEN_REUSE_PROOF)
        or rv.get('producer_manifest')!=reader.raw_pointer(prior.REUSE_QA_MANIFEST)):
        raise ValueError('Published Reader proof revalidation lineage differs')
    original=reader.bound(rv['proof'],sealed=False)
    if (proof['entries']!=original['entries'] or proof['frozen_manifest']!=original['frozen_manifest']
        or proof['frozen_outcome']!=original['frozen_outcome']
        or any(proof['input_sha256'].get(p)!=s for p,s in original['input_sha256'].items())):
        raise ValueError('Published proof no longer carries the exact original raw replay')
    producer=reader.bound(rv['producer_manifest'],sealed=False)
    prior._validate_qa_manifest(producer)
    outref=rv['producer_outcome'];out=reader.bound(outref)
    if (producer['baseline_reuse_proof']!=rv['proof'] or out.get('terminal') is not True
        or out.get('protocol_sha256')!=producer['protocol_sha256']):
        raise ValueError('Proof producer is not the bound original complete phase')
    cache[key]=proof;reader._published_proof_cache=cache
    return proof

def _candidate(reader,snapshot,run,m,job,source_job,indexed,proof_done):
    if job!=source_job:raise ValueError('QA job differs from selected source job')
    path=qa.stage_files(run,job)['reader'];signature=_reader_signature(job)
    row=audit.original.existing.bound(snapshot,path,{**qa.identity(job,m),'stage':'reader'})
    if row.get('status')!='complete' or row.get('error') is not None or row.get('raw_invalid_label') is not None:
        raise ValueError('Both source Readers must be valid complete')
    answer=row['result']['prediction']['answer']
    if not isinstance(answer,str) or not answer.strip() or row['result']['prediction']['raw']!=answer:
        raise ValueError('Nonempty exact actual Reader string required')
    ctx=qa.stage_context(job,m,'reader');events=row.get('api_events')
    if row.get('inheritance_kind')=='offline_verified_frozen_exact_request':
        pref=m['baseline_reuse_proof'];proof=_published_reuse_proof(reader,m);entry=proof['entries'][job['query_id']]
        if (row.get('reuse_proof')!=pref or row.get('inherited_from')!=entry['reader']
            or m['reuse_plans'][job['job_id']]['reader']!=entry['reader']
            or row.get('model_calls')!=0 or events!=[] or path.with_suffix('.intent.json').exists()
            or indexed.get((request_hash(signature),READER_STAGE,fingerprint(ctx)))
            or signature!=entry['signatures']['reader']
            or answer!=entry['reader_result']['prediction']['answer']):
            raise ValueError('Inherited Reader is not exact original proven reuse')
        if pref['sha256'] not in proof_done:
            # Reuse the audited raw proof only while all of its source bytes remain.
            if proof.get('model_calls_during_proof')!=0:raise ValueError('Original proof is not offline')
            for p,s in proof['input_sha256'].items():
                if snapshot.sha(p)!=s:raise ValueError('Original raw replay proof binding changed')
            for p,s in proof.get('proof_code_sha256',{}).items():
                if file_sha(ROOT/p)!=s:raise ValueError('Original raw replay implementation changed')
            proof_done.add(pref['sha256'])
        raw_lineage={'kind':'exact_verified_reader_inheritance','proof':pref,'producer_reader':entry['reader']}
    else:
        intent=snapshot.read(path.with_suffix('.intent.json'))
        if any(intent.get(k)!=v for k,v in {**qa.identity(job,m),'stage':'reader'}.items()):
            raise ValueError('Reader stage intent identity differs')
        replay=qa.Gemini38Replay(run/f'conv_{job["conversation"]}/api',events,ctx,indexed,snapshot=snapshot)
        response=replay.request(signature['provider'],signature['model'],signature['body'],
            operation=signature['operation'],stage=READER_STAGE)
        replay.assert_consumed()
        if response_text(response)!=answer:raise ValueError('Reader answer differs from raw response')
        raw_lineage={'kind':'raw_reader_replay','api_events':deepcopy(events)}
    return {'answer':answer,'reader':reader.raw_pointer(path),
        'source_job_sha256':digest(job),'selected_source_job_sha256':digest(source_job),
        'reader_request_sha256':request_hash(signature),'reader_request_signature':signature,
        'selected_ids':deepcopy(job['selected_ids']),'selected_facts':deepcopy(job['selected_facts']),
        'raw_lineage':raw_lineage}

def _inputs(reader,spec):
    refs=spec['source_refs'];sref=prior._canonical_ref(refs['r03_selected_inputs'])
    source,pools=_validate_parent_source(reader,sref)
    if source['selector_manifest']!=prior._canonical_ref(refs['r03_selection_manifest']):
        raise ValueError('Frozen pool source manifest changed')
    qref=prior._canonical_ref(refs['r03_qa_manifest']);m=_qa_parent(reader,qref,sref)
    run=Path(qref['path']).parent;snapshot=Snapshot(reader)
    # Existing ledger validator and replay bind consumer and physical producer contexts.
    logs,indexed,ledger_refs=READER_LEDGER(snapshot,run,m)
    jobs={j['job_id']:j for j in m['jobs']};sjobs={j['job_id']:j for j in source['jobs']}
    if jobs!=sjobs:raise ValueError('Actual QA jobs differ from pinned selected inputs')
    result=[];proof_done=set()
    for b in (j for j in source['jobs'] if j['method']=='baseline'):
        qid=b['query_id'];j=jobs['jev/'+qid]
        if any(b[k]!=j[k] for k in ('conversation','query_id','question','reference_date')):
            raise ValueError('Two routes differ in frozen query/date')
        candidates={route:_candidate(reader,snapshot,run,m,job,sjobs[job['job_id']],indexed,proof_done)
            for route,job in (('baseline',b),('jev_r03',j))}
        pool=pools[qid];memories=union_records(pool['candidate_records'],b['selected_facts'],j['selected_facts'])
        signature=build_request(b['question'],b['reference_date'],memories,candidates,qid,spec)
        identical=candidates['baseline']['answer']==candidates['jev_r03']['answer'];size=len(serialized_signature(signature))
        result.append({'query_id':qid,'qid':qid,'conversation':b['conversation'],'state':signature['body']['state'],
            'presentation':presentation(qid),'candidates':candidates,'union':memories,
            'pool_ref':pool['pool_ref'],'pool_sha256':pool['pool_sha256'],
            'request_signature':signature,'signature':signature,'request_hash':request_hash(signature),
            'identical':identical,'dispatch_eligible':not identical and size<=POLICY['signature_utf8_byte_cap'],
            'signature_utf8_bytes':size,'signature_utf8_byte_cap':POLICY['signature_utf8_byte_cap'],
            'logical_request_cap':0 if identical else 1,'max_physical_attempts_per_phase':3,
            'input_tokens':None,'gold_or_labels_in_model_state':False})
    if len(result)!=1540 or len({r['query_id'] for r in result})!=1540:raise ValueError('Full1540 required')
    return result,source,ledger_refs

def prepare(run_dir):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.name.startswith('answer_selection_r04') or run.exists():
        raise FileExistsError('Fresh direct runs/answer_selection_r04* namespace required')
    reader=Reader();spec,ref=_spec(reader);records,source,ledgers=_inputs(reader,spec)
    protocol={'research_protocol':ref,'policy':deepcopy(POLICY),'source_refs':{k:v for k,v in spec['source_refs'].items() if 'OFFLINE' not in k},
        'code_sha256':_codes(),'arms':list(ARMS),'questions_per_arm':1540,
        'source_dataset_sha256':source['source_dataset_sha256'],'official_metric_provenance':source['official_metric_provenance']}
    psha=digest(protocol)
    # All pinned inputs and code must validate before creating the namespace.
    bindings={p:ref['sha256'] for p,ref in reader.refs.items()}
    if any(file_sha(p)!=s for p,s in bindings.items()):raise ValueError('Input changed during offline preparation')
    run.mkdir();items=[];counts=Counter()
    for r in records:
        rec={**r,'schema':REQUEST_SCHEMA,'protocol_sha256':psha}
        req=save_new(run/'requests'/f'{r["query_id"]}.json',rec)
        items.append({'query_id':r['query_id'],'conversation':r['conversation'],'request':req})
        counts['queries']+=1;counts['identical_answer_no_dispatch']+=r['identical']
        counts['dispatch_eligible']+=r['dispatch_eligible']
        counts['oversize_not_dispatched']+=not r['identical'] and not r['dispatch_eligible']
        counts['cheap_r03_choices']+=cheap_choice(r)=='jev_r03'
    proof=save_new(run/'reader_source_proof.json',{'schema':'answer-selection-r04-reader-only-proof-v1',
        'protocol_sha256':psha,'input_sha256':bindings,'ledger_refs':ledgers,
        'candidate_readers_verified':3080,'model_calls':0,'judge_labels_read_for_selection':False,
        'semantics':'Exact Reader signature/raw text and provenance only; metrics and labels are not selection inputs.'})
    mref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':psha,'items':items,'reader_source_proof':proof,
        'summary':dict(counts),'model_calls_during_prepare':0})
    return {'manifest':mref,'summary':dict(counts),'model_calls':0}

def _manifest(reader,run):
    m=reader.read(run/'run_manifest.json');spec,ref=_spec(reader);p=m['protocol']
    if (m.get('schema')!=SCHEMA or m.get('status')!='prepared' or m['protocol_sha256']!=digest(p)
        or p['research_protocol']!=ref or p['policy']!=POLICY or p['code_sha256']!=_codes()
        or p['source_refs']!={k:v for k,v in spec['source_refs'].items() if 'OFFLINE' not in k}
        or p['arms']!=list(ARMS) or p['questions_per_arm']!=1540
        or len(m['items'])!=1540 or len({i['query_id'] for i in m['items']})!=1540):
        raise ValueError('Frozen R04 protocol/code/cohort changed')
    return m,spec

def validate_prepared(run_dir):
    run=Path(run_dir).resolve();reader=Reader();m,spec=_manifest(reader,run)
    expected,source,_=_inputs(reader,spec)
    proof=reader.bound(m['reader_source_proof'])
    if proof.get('protocol_sha256')!=m['protocol_sha256'] or proof.get('candidate_readers_verified')!=3080 or proof.get('model_calls')!=0:
        raise ValueError('Reader-only preparation proof identity differs')
    for p,s in proof['input_sha256'].items():
        if reader.raw_pointer(p)['sha256']!=s:raise ValueError('Prepared Reader source bytes changed')
    if any(m['protocol'][k]!=source[k] for k in ('source_dataset_sha256','official_metric_provenance')):
        raise ValueError('Dataset/metric source differs')
    byid={r['query_id']:r for r in expected};records=[]
    for item in m['items']:
        qid=item['query_id'];ref=item['request'];prior._inside(ref,run)
        if Path(ref['path']).resolve()!=run/'requests'/f'{qid}.json':raise ValueError('Query filename differs')
        rec=reader.bound(ref);value={**byid[qid],'schema':REQUEST_SCHEMA,'protocol_sha256':m['protocol_sha256']}
        if {k:v for k,v in rec.items() if k!='content_sha256'}!=value or item['conversation']!=value['conversation']:
            raise ValueError('Prepared Choice signature/candidates/union cannot replay')
        records.append(rec)
    if {Path(i['request']['path']).resolve() for i in m['items']}!={p.resolve() for p in (run/'requests').glob('*.json')}:
        raise ValueError('Missing/orphan request record')
    return m,records

def _result(reader,run,m,record):
    path=run/'responses'/f'{record["query_id"]}.json';result=reader.read(path);ref=reader.pointer(path)
    if (result.get('schema')!=SCORE_SCHEMA or result.get('query_id')!=record['query_id']
        or result.get('protocol_sha256')!=m['protocol_sha256'] or result.get('request_hash')!=record['request_hash']):
        raise ValueError('Choice result identity differs')
    status=result.get('status');parsed=result.get('choice_result');rawref=result.get('raw')
    if status=='identical_answer_no_dispatch':
        if not record['identical'] or parsed is not None or rawref is not None:raise ValueError('Invalid identical offline result')
    elif status=='complete':
        if not record['dispatch_eligible']:raise ValueError('Ineligible signature was dispatched')
        raw=reader.bound(rawref,sealed=False)
        if raw.get('request')!=record['signature'] or parse_choice(raw['response'],record['signature'])!=parsed:
            raise ValueError('Choice raw/signature/parser differs')
    elif status in ('failed','unknown','not_dispatched'):
        if parsed is not None:raise ValueError('Unavailable Choice cannot fill probabilities')
        if rawref is not None and reader.bound(rawref,sealed=False).get('request')!=record['signature']:
            raise ValueError('Unavailable raw must still bind approved signature')
    else:raise ValueError('Explicit terminal Choice status required')
    decision=choose_jev(record,parsed,status=status)
    if result.get('selected_route')!=decision['route'] or result.get('selection_reason')!=decision['reason']:
        raise ValueError('Wrapper action differs from pure rule')
    candidate=record['candidates'][decision['route']]
    if (result.get('selected_original_reader')!=candidate['reader'] or result.get('selected_exact_answer')!=candidate['answer']
        or result.get('presentation')!=record['presentation'] or result.get('cheap_route')!=cheap_choice(record)
        or result.get('union_sha256')!=digest(record['union'])):
        raise ValueError('Wrapper changed the selected original answer or context')
    return decision,ref,rawref,status

def _decisions(reader,run,m,records):
    if {p.stem for p in (run/'responses').glob('*.json')}!={r['query_id'] for r in records}:
        raise ValueError('Exactly one terminal wrapper for every prepared query required')
    refs={i['query_id']:i['request'] for i in m['items']}
    values=[]
    for rec in records:
        jdecision,response,raw,status=_result(reader,run,m,rec)
        for method in ARMS:
            d={'route':'baseline','reason':'baseline_anchor'} if method=='baseline' else ({'route':cheap_choice(rec),'reason':'fixed_exact_abstention_heuristic'} if method=='cheap' else jdecision)
            candidate=rec['candidates'][d['route']]
            values.append({'schema':DECISION_SCHEMA,'protocol_sha256':m['protocol_sha256'],'query_id':rec['query_id'],
                'conversation':rec['conversation'],'method':method,**d,'answer':candidate['answer'],'reader':candidate['reader'],
                'request':refs[rec['query_id']],
                'response':response if method=='jev' else None,'raw':raw if method=='jev' else None,
                'score_status':status if method=='jev' else 'offline','source_job_sha256':candidate['source_job_sha256']})
    return values

def _job(value,decision_ref):
    return {**{k:value[k] for k in ('query_id','conversation','method','route','answer','reader','source_job_sha256')},
            'job_id':value['method']+'/'+value['query_id'],'decision':decision_ref}

def seal(run_dir):
    run=Path(run_dir).resolve()
    if (run/'selected_answers.json').exists():raise FileExistsError('Never overwrite selected answers')
    m,records=validate_prepared(run);reader=Reader();jobs=[];counts=Counter()
    values=_decisions(reader,run,m,records)
    # All result wrappers must pass before publishing any decisions/source.
    for value in values:
        method=value['method'];dp=save_new(run/'decisions'/method/f'{value["query_id"]}.json',value)
        jobs.append(_job(value,dp));counts[method+'_route_'+value['route']]+=1
    source=save_new(run/'selected_answers.json',{'schema':SOURCE_SCHEMA,'status':'sealed','parent_run':str(run),
        'protocol_sha256':m['protocol_sha256'],'selector_manifest':prior._canonical_ref({'path':str(run/'run_manifest.json'),'sha256':file_sha(run/'run_manifest.json')}),
        'answer_protocol':m['protocol']['research_protocol'],'frozen_r03_qa_manifest':m['protocol']['source_refs']['r03_qa_manifest'],
        'frozen_r03_selected_inputs':m['protocol']['source_refs']['r03_selected_inputs'],
        'arms':list(ARMS),'questions_per_arm':1540,'jobs':jobs,'selection_counts':dict(counts),
        'reader_source_proof':m['reader_source_proof'],'model_calls_during_seal':0})
    return {'source':source,'jobs':len(jobs),'selection_counts':dict(counts),'model_calls':0}

def validate_source(path,*,fresh=False):
    path=Path(path).resolve();reader=Reader();source=reader.read(path);run=Path(source['parent_run']).resolve()
    if path!=run/'selected_answers.json' or source.get('schema')!=SOURCE_SCHEMA or source.get('status')!='sealed':
        raise ValueError('Sealed R04 exact-answer source required')
    m,records=validate_prepared(run)
    if (source['protocol_sha256']!=m['protocol_sha256'] or source['selector_manifest']!=reader.pointer(run/'run_manifest.json')
        or source['answer_protocol']!=m['protocol']['research_protocol'] or source['arms']!=list(ARMS)
        or source['questions_per_arm']!=1540 or len(source['jobs'])!=4620
        or source['reader_source_proof']!=m['reader_source_proof']
        or source['frozen_r03_qa_manifest']!=m['protocol']['source_refs']['r03_qa_manifest']
        or source['frozen_r03_selected_inputs']!=m['protocol']['source_refs']['r03_selected_inputs']):
        raise ValueError('Source protocol/cohort differs')
    expected=[];counts=Counter()
    for value in _decisions(reader,run,m,records):
        path=run/'decisions'/value['method']/f'{value["query_id"]}.json';actual=reader.read(path)
        if {k:v for k,v in actual.items() if k!='content_sha256'}!=value:raise ValueError('Decision full replay differs')
        expected.append(_job(value,reader.pointer(path)));counts[value['method']+'_route_'+value['route']]+=1
    if source['jobs']!=expected or source['selection_counts']!=dict(counts):raise ValueError('Full4620 answer cohort/counts differ')
    return source

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check','seal','validate-source'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':result=prepare(a.run_dir)
    elif a.stage=='check':
        m,recs=validate_prepared(a.run_dir);result={'valid':True,'queries':len(recs),'model_calls':0,'summary':m['summary']}
    elif a.stage=='seal':result=seal(a.run_dir)
    else:result={'valid':bool(validate_source(a.run_dir/'selected_answers.json',fresh=True)),'model_calls':0}
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
