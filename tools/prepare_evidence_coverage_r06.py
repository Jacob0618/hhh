"""Offline immutable R06 stages: prepare, compile-score, seal and replay validation."""
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
from jev_mem.evidence_coverage_r06 import (ARMS,POLICY,PROTOCOL_SHA256,digest,request_hash,
 serialized_signature,build_facet_request,parse_facet_response,build_request,parse_scores,
 choose_baseline,choose_cheap,choose_jev,selected_facts)
from tools import prepare_evidence_selection_r03 as prior
Reader=prior.Reader
save_new=prior.save_new
file_sha=prior.file_sha
now=prior.now
DEFAULT_SOURCE=prior.DEFAULT_SOURCE
FROZEN_QA_MANIFEST=prior.FROZEN_QA_MANIFEST
FROZEN_REUSE_PROOF=prior.FROZEN_REUSE_PROOF
REUSE_QA_MANIFEST=prior.REUSE_QA_MANIFEST
RESEARCH_PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R06_coverage_protocol_final_v1.json'
SCHEMA='evidence-coverage-r06-selector-manifest-v1'
COMPILED_SCHEMA='evidence-coverage-r06-score-manifest-v1'
SOURCE_SCHEMA='evidence-coverage-r06-selected-inputs-v1'
PROPOSAL_SCHEMA='evidence-coverage-r06-pool-input-v1'
FACET_REQUEST_SCHEMA='evidence-coverage-r06-facet-request-v1'
REQUEST_SCHEMA='evidence-coverage-r06-request-v1'
FACET_SCORE_SCHEMA='evidence-coverage-r06-score-result-v1'
SCORE_SCHEMA='evidence-coverage-r06-score-result-v1'
SELECTION_SCHEMA='evidence-coverage-r06-selection-v1'
AUDIT_SCHEMA='evidence-coverage-r06-terminal-audit-v1'
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/evidence_coverage_r06.py',
 'tools/prepare_evidence_coverage_r06.py','tools/score_evidence_coverage_r06.py',
 'tools/snapshot_evidence_coverage_r06.py','tools/run_evidence_coverage_qa_r06.py',
 'tools/summarize_evidence_coverage_r06.py')))
_VALIDATION_CACHE={}


def _body(value):
    return {k:v for k,v in value.items()if k!='content_sha256'}


def _codes():
    return {p:file_sha(ROOT/p)for p in CODE}


def _research_protocol(reader):
    spec=reader.read(RESEARCH_PROTOCOL,sealed=False)
    ref=deepcopy(reader.refs[str(RESEARCH_PROTOCOL.resolve())])
    if ref['sha256']!=PROTOCOL_SHA256:raise ValueError('Exact three-voted R06 protocol changed')
    for pointer in spec['input_refs'].values():
        reader.raw_pointer(pointer['path'],ref=prior._canonical_ref(pointer))
    return spec,ref


def _inputs(reader,source_path,spec):
    # Reuse already audited R03 extraction/first20/samebank/proposal pool validation.
    return prior._inputs(reader,source_path,{'basis':{'r01_pool_manifest':spec['input_refs']['r01_pool_manifest']}})


def _protocol(reader,source,source_ref,banks,spec,spec_ref):
    reuse=reader.read(REUSE_QA_MANIFEST,sealed=False)
    prior._validate_qa_manifest(reuse)
    if reuse['baseline_reuse_proof']!=reader.raw_pointer(FROZEN_REUSE_PROOF):
        raise ValueError('Original R01 proof producer mismatch')
    codes=_codes()
    if (codes['tools/pilot400_recovery_rank_v1.py']!=spec['input_refs']['bm25']['sha256']
        or codes['jev_mem/routing.py']!=spec['input_refs']['tokenizer']['sha256']):
        raise ValueError('Frozen BM25 or tokenizer changed')
    return {'schema':SCHEMA,'policy':deepcopy(POLICY),'research_protocol':spec_ref,
        'scientific_basis':deepcopy(spec['input_refs']),'frozen_source':source_ref,
        'frozen_qa_run_manifest':reader.raw_pointer(FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof':reader.raw_pointer(FROZEN_REUSE_PROOF),
        'reuse_proof_qa_manifest':reader.raw_pointer(REUSE_QA_MANIFEST),
        'source_dataset_sha256':source['source_dataset_sha256'],
        'frozen_protocol_sha256':source['protocol_sha256'],'code_sha256':codes,
        'questions_per_arm':1540,'arms':list(ARMS),'fact_count':1637,
        'bank_refs':{str(ci):banks[ci]['ref']for ci in sorted(banks)},
        'official_metric_provenance':source['official_metric_provenance']}


def _request_value(signature,old,protocol_sha,*,facet_index=None,requirement=None):
    qid=old['query_id'];request_id=qid if facet_index is None else f'{qid}_f{facet_index:02d}'
    size=len(serialized_signature(signature))
    value={'schema':FACET_REQUEST_SCHEMA if facet_index is None else REQUEST_SCHEMA,
        'request_id':request_id,'query_id':qid,'conversation':old['conversation'],
        'question':old['question'],'reference_date':old['reference_date'],
        'protocol_sha256':protocol_sha,'signature':signature,'request_hash':request_hash(signature),
        'logical_request_cap':1,'max_physical_attempts_per_phase':3,
        'signature_utf8_bytes':size,'signature_utf8_byte_cap':131072,
        'dispatch_eligible':size<=131072,'input_tokens':None,
        'input_tokens_note':'Serialization bytes are not token counts or billing.'}
    if facet_index is not None:
        value.update({'facet_index':facet_index,'requirement':requirement,
                      'logical_noul_count':len(signature['body']['questions'])})
    return value


def _prepared_value(old,source_ref,realized,req_ref,protocol_sha):
    return {'schema':PROPOSAL_SCHEMA,'protocol_sha256':protocol_sha,
        'query_id':old['query_id'],'conversation':old['conversation'],'frozen_source':source_ref,
        'original_job':deepcopy(old),'source_job_sha256':digest(old),'bank':old['bank'],
        'original_selection':old['selection'],'facet_request':req_ref,
        'baseline_selected_sha256':digest(old['selected_facts']),
        'frozen_pool':deepcopy(realized),'candidate_records_sha256':digest(realized['candidate_records']),
        'gold_not_used_by_selection':True,'lineage_job_contains_offline_reference':True}


def prepare(run_dir,source_path=DEFAULT_SOURCE):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve()or not run.name.startswith('evidence_coverage_r06')or run.exists():
        raise FileExistsError('Fresh direct runs/evidence_coverage_r06* namespace required')
    reader=Reader();spec,spec_ref=_research_protocol(reader)
    source,sref,jobs,banks,pools=_inputs(reader,source_path,spec)
    protocol=_protocol(reader,source,sref,banks,spec,spec_ref);psha=digest(protocol)
    run.mkdir(parents=True,exist_ok=False);items=[];counts=Counter()
    for old in jobs:
        qid=old['query_id'];signature=build_facet_request(old['question'],old['reference_date'],spec)
        req=save_new(run/'facet_requests'/f'{qid}.json',_request_value(signature,old,psha))
        prop=save_new(run/'proposals'/f'{qid}.json',_prepared_value(old,sref,pools[qid],req,psha))
        items.append({'query_id':qid,'conversation':old['conversation'],'proposal':prop,'request':req})
        counts['facet_requests']+=1;counts['baseline_first20_verified']+=1
        counts['candidate_records']+=len(pools[qid]['candidate_records'])
        counts['oversize_facet_requests']+=len(serialized_signature(signature))>131072
    mref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'phase':'facet','status':'prepared',
        'created_utc':now(),'protocol':protocol,'protocol_sha256':psha,'items':items,
        'summary':dict(counts),'model_calls_during_prepare':0})
    return {'manifest':mref,'queries':1540,'summary':dict(counts),'model_calls':0}


def _manifest(reader,run):
    m=reader.read(run/'run_manifest.json');spec,sref=_research_protocol(reader)
    source,source_ref,jobs,banks,pools=_inputs(reader,m['protocol']['frozen_source']['path'],spec)
    expected=_protocol(reader,source,source_ref,banks,spec,sref)
    if (m.get('schema')!=SCHEMA or m.get('phase')!='facet'or m.get('status')!='prepared'
        or m['protocol']!=expected or m['protocol_sha256']!=digest(expected)
        or len(m['items'])!=1540 or len({i['query_id']for i in m['items']})!=1540):
        raise ValueError('R06 immutable initial protocol/code/cohort differs')
    byid={j['query_id']:j for j in jobs}
    if set(byid)!={i['query_id']for i in m['items']}:raise ValueError('Prepared cohort changed')
    for item in m['items']:
        qid=item['query_id'];old=byid[qid];realized=pools[qid]
        for ref in (item['request'],item['proposal']):prior._inside(ref,run)
        if (Path(item['request']['path']).resolve()!=run/'facet_requests'/f'{qid}.json'
            or Path(item['proposal']['path']).resolve()!=run/'proposals'/f'{qid}.json'
            or item['conversation']!=old['conversation']):raise ValueError('Initial query namespace changed')
        if _body(reader.bound(item['proposal']))!=_prepared_value(old,source_ref,realized,item['request'],m['protocol_sha256']):
            raise ValueError('Original pool/proposal exact replay differs')
        sig=build_facet_request(old['question'],old['reference_date'],spec)
        if _body(reader.bound(item['request']))!=_request_value(sig,old,m['protocol_sha256']):
            raise ValueError('Facet request exact replay differs')
    _no_orphans(run/'facet_requests',[i['request']for i in m['items']])
    return m,spec,byid,pools


def _no_orphans(directory,refs):
    if {Path(r['path']).resolve()for r in refs}!={p.resolve()for p in directory.glob('*.json')}:
        raise ValueError('Unmanifested or missing request file')


def _audit_gate(reader,run,phase,manifest_ref,requests):
    path=run/'reviews'/f'{phase}_terminal_audit_v1.json';audit=reader.read(path)
    auditor=ROOT/'tools/score_evidence_coverage_r06.py'
    if (audit.get('schema')!=AUDIT_SCHEMA or audit.get('phase')!=phase
        or audit.get('status')!='passed'or audit.get('manifest')!=manifest_ref
        or audit.get('unknown')!=[] or audit.get('requests_count')!=len(requests)
        or audit.get('codepins')!={'path':str(auditor.resolve()),'sha256':_codes()['tools/score_evidence_coverage_r06.py']}):
        raise ValueError('Exact passed terminal audit required')
    outcome=audit.get('outcome');prior._inside(outcome,run);reader.raw_pointer(outcome['path'],ref=outcome)
    input_refs=audit.get('input_refs')
    if not isinstance(input_refs,list)or len({r.get('path')for r in input_refs})!=len(input_refs):
        raise ValueError('Complete unique immutable terminal audit input refs required')
    ledger_path=(run/'api_calls.jsonl').resolve()
    for ref in input_refs:
        p=prior._inside(ref,run)
        if p==ledger_path:raise ValueError('Shared ledger needs the immutable phase projection, not old fullfile SHA')
        reader.bound(ref,sealed=False)
    hashes=sorted({reader.bound(ref)['request_hash']for ref in requests})
    stage='evidence_coverage_facet_r06'if phase=='facet'else'evidence_coverage_score_r06'
    current=ledger_projection(reader,ledger_path,stage,hashes)
    if audit.get('ledger_projection')!=current:
        raise ValueError('Terminal ledger phase/hash projection changed after audit')
    expected=[]
    directory='facet_responses'if phase=='facet'else'responses'
    for reqref in requests:
        req=reader.bound(reqref);result_path=run/directory/f'{req["request_id"]}.json'
        ref=reader.pointer(result_path)
        expected.append({**ref,'request_id':req['request_id'],'query_id':req['query_id'],'request_ref':reqref})
    if audit.get('response_refs')!=expected:raise ValueError('Audit does not bind every exact terminal response')
    controller=run/('execution_intent_facet.json'if phase=='facet'else'execution_intent_score_jev.json')
    required={manifest_ref['path'],outcome['path'],str(controller.resolve()),
              *[r['path']for r in requests],*[r['path']for r in expected]}
    if not required<={r['path']for r in input_refs}:raise ValueError('Audit input refs omit required terminal artifacts')
    if (not isinstance(audit.get('known_terminal_failures'),list)
        or any(x.get('technical_recovery_eligible') for x in audit['known_terminal_failures'])):
        raise ValueError('Known failures require explicit list and no pending eligible recovery')
    return reader.pointer(path),audit


def ledger_projection(reader,path,stage,request_hashes):
    """Read current ledger once; permit new other-phase rows, bind all relevant rows."""
    path=Path(path).resolve();key=str(path)
    if not hasattr(reader,'ledger_rows'):reader.ledger_rows={}
    if key not in reader.ledger_rows:
        raw=path.read_bytes()if path.exists()else b''
        if raw and not raw.endswith(b'\n'):raise ValueError('Unfinished JSONL ledger line is unknown')
        rows=[json.loads(line)for line in raw.splitlines()if line.strip()]
        if any(not isinstance(row,dict)for row in rows):raise ValueError('Ledger events must be objects')
        reader.ledger_rows[key]=rows
        if path.exists():reader.refs[key]={'path':key,'sha256':hashlib.sha256(raw).hexdigest()}
    keys=set(request_hashes)
    rows=[row for row in reader.ledger_rows[key]if row.get('stage')==stage or row.get('request_hash')in keys]
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return {'path':key,'stage':stage,'request_hashes':sorted(keys),
        'selected_row_count':len(rows),'selected_rows_sha256':hashlib.sha256(raw).hexdigest()}


def _result(reader,run,reqref,*,phase,spec):
    req=reader.bound(reqref)
    path=run/('facet_responses'if phase=='facet'else'responses')/f'{req["request_id"]}.json'
    value=reader.read(path);status=value.get('status')
    if (value.get('schema')!=(FACET_SCORE_SCHEMA if phase=='facet'else SCORE_SCHEMA)
        or value.get('phase')!=phase
        or any(value.get(k)!=req[k]for k in ('request_id','query_id','conversation','protocol_sha256','request_hash'))
        or (phase=='jev'and value.get('facet_index')!=req['facet_index'])):
        raise ValueError('Terminal response identity differs')
    if status=='unknown':raise ValueError('Unknown phase state blocks compile/seal')
    if status not in ('complete','failed','not_dispatched'):raise ValueError('Explicit terminal status required')
    parsed_key='facet_parse'if phase=='facet'else'scores'
    rawref=value.get('raw');raw=None
    if rawref is not None:
        prior._inside(rawref,run);raw=reader.bound(rawref,sealed=False)
        if raw.get('request')!=req['signature']:raise ValueError('Terminal raw signature differs')
    if value.get('actual_request_hash')not in (None,req['request_hash']):
        raise ValueError('Actual request hash differs')
    if status=='complete':
        if not req['dispatch_eligible']or raw is None:raise ValueError('Complete result needs approved full raw')
        parsed=parse_facet_response(raw['response'],req['question'],spec)if phase=='facet'else parse_scores(raw['response'],req['signature'])
        if value.get(parsed_key)!=parsed:raise ValueError('Stored parser output differs from bound raw')
    else:
        if value.get(parsed_key)is not None:raise ValueError('Unavailable parse must be absent, never zeros')
        if not value.get('reason'):raise ValueError('Terminal unavailable result must explain reason')
        if status=='not_dispatched'and(req['dispatch_eligible']or raw is not None):
            raise ValueError('Not dispatched only for the full signature cap')
        parsed=None
    return {'status':status,'parsed':parsed,'response':reader.pointer(path),'raw_response':rawref}


def _facet_results(reader,run,m,spec):
    refs=[i['request']for i in m['items']]
    audit_ref,_=_audit_gate(reader,run,'facet',reader.pointer(run/'run_manifest.json'),refs)
    results={i['query_id']:_result(reader,run,i['request'],phase='facet',spec=spec)for i in m['items']}
    return results,audit_ref


def _compiled_item(old,item,realized,facet_result,requests):
    parsed=facet_result['parsed'];facets=None if parsed is None else parsed['facets']
    return {'query_id':old['query_id'],'conversation':old['conversation'],'proposal':item['proposal'],
        'facet_request':item['request'],'facet_result':deepcopy(facet_result),'facets':facets,
        'requests':requests,'cheap_decision':choose_cheap(old['question'],old['reference_date'],
          realized['candidate_records'],facets,facet_status=facet_result['status'])}


def compile_score(run_dir):
    run=Path(run_dir).resolve()
    if (run/'score_manifest.json').exists()or(run/'requests').exists():
        raise FileExistsError('Never overwrite compiled score namespace')
    reader=Reader();m,spec,byid,pools=_manifest(reader,run)
    facets,aref=_facet_results(reader,run,m,spec);items=[];refs=[];counts=Counter()
    # Every audit/response replay above completes before creating any new namespace.
    for initial in m['items']:
        qid=initial['query_id'];old=byid[qid];r=facets[qid];requests=[]
        if r['status']=='complete':
            for j,requirement in enumerate(r['parsed']['facets']):
                signature=build_request(old['question'],old['reference_date'],requirement,pools[qid]['candidate_records'],spec)
                ref=save_new(run/'requests'/f'{qid}_f{j:02d}.json',
                    _request_value(signature,old,m['protocol_sha256'],facet_index=j,requirement=requirement))
                requests.append(ref);refs.append(ref);counts['jev_requests']+=1
                counts['logical_nouls']+=len(pools[qid]['candidate_records'])
                counts['oversize_not_dispatched']+=len(serialized_signature(signature))>131072
            counts['facet_count_'+str(r['parsed']['m'])]+=1
        else:counts['generator_fallback_queries']+=1
        items.append(_compiled_item(old,initial,pools[qid],r,requests))
    ref=save_new(run/'score_manifest.json',{'schema':COMPILED_SCHEMA,'phase':'jev','status':'compiled',
        'created_utc':now(),'protocol':deepcopy(m['protocol']),'protocol_sha256':m['protocol_sha256'],
        'initial_manifest':reader.pointer(run/'run_manifest.json'),'facet_terminal_audit':aref,
        'items':items,'request_refs':refs,'summary':dict(counts),'model_calls_during_compile':0})
    return {'manifest':ref,'queries':1540,'requests':len(refs),'summary':dict(counts),'model_calls':0}


def _compiled(reader,run,m,spec,byid,pools):
    cm=reader.read(run/'score_manifest.json');facets,aref=_facet_results(reader,run,m,spec)
    if (cm.get('schema')!=COMPILED_SCHEMA or cm.get('phase')!='jev'or cm.get('status')!='compiled'
        or cm.get('protocol')!=m['protocol']or cm.get('protocol_sha256')!=m['protocol_sha256']
        or cm.get('initial_manifest')!=reader.pointer(run/'run_manifest.json')
        or cm.get('facet_terminal_audit')!=aref or len(cm['items'])!=1540):
        raise ValueError('Compiled immutable matrix manifest differs')
    expected=[];allrefs=[]
    for initial in m['items']:
        qid=initial['query_id'];old=byid[qid];result=facets[qid];refs=[]
        if result['status']=='complete':
            for j,requirement in enumerate(result['parsed']['facets']):
                path=run/'requests'/f'{qid}_f{j:02d}.json';ref=reader.pointer(path)
                signature=build_request(old['question'],old['reference_date'],requirement,pools[qid]['candidate_records'],spec)
                if _body(reader.bound(ref))!=_request_value(signature,old,m['protocol_sha256'],facet_index=j,requirement=requirement):
                    raise ValueError('Full facet matrix request exact replay differs')
                refs.append(ref);allrefs.append(ref)
        expected.append(_compiled_item(old,initial,pools[qid],result,refs))
    if cm['items']!=expected or cm['request_refs']!=allrefs:raise ValueError('Compiled matrix/facets/cheap decision changed')
    _no_orphans(run/'requests',allrefs)
    return cm


def validate_prepared(run_dir,stage='facet'):
    run=Path(run_dir).resolve();reader=Reader();m,spec,jobs,pools=_manifest(reader,run)
    if stage=='facet':return m,[deepcopy(i['request'])for i in m['items']]
    if stage!='score':raise ValueError('Prepared stage must be facet or score')
    cm=_compiled(reader,run,m,spec,jobs,pools)
    return cm,deepcopy(cm['request_refs'])


def _expected_records(reader,run,m,spec,byid,pools):
    cm=_compiled(reader,run,m,spec,byid,pools)
    jref,_=_audit_gate(reader,run,'jev',reader.pointer(run/'score_manifest.json'),cm['request_refs'])
    records=[]
    for item in cm['items']:
        qid=item['query_id'];old=byid[qid];realized=pools[qid];candidates=realized['candidate_records']
        results=[_result(reader,run,r,phase='jev',spec=spec)for r in item['requests']]
        statuses=[r['status']for r in results]
        if item['facet_result']['status']!='complete':
            jd=choose_baseline(candidates,score_status=item['facet_result']['status'],reason='facet_unavailable_keep_baseline')
        elif all(s=='complete'for s in statuses)and len(results)==len(item['facets']):
            jd=choose_jev(candidates,[r['parsed']for r in results])
        else:
            jd=choose_jev(candidates,None,score_status='failed'if'failed'in statuses else'not_dispatched')
        decisions={'baseline':choose_baseline(candidates),'cheap':item['cheap_decision'],'jev':jd}
        native=set(realized['original_native80_ids']);base=set(old['selected_ids'])
        for method in ARMS:
            d=decisions[method];selected=selected_facts(candidates,d['pool_selected_indices'])
            ids=[f['id']for f in selected];added=set(ids)-base;removed=base-set(ids)
            if(set(ids)==base)!=(selected==old['selected_facts']):raise ValueError('Same members must be exact ordered N0')
            records.append((old,method,{'schema':SELECTION_SCHEMA,'protocol_sha256':m['protocol_sha256'],
                'query_id':qid,'conversation':old['conversation'],'method':method,'bank':old['bank'],
                'original_selection':old['selection'],'proposal':item['proposal'],
                'facet_request':item['facet_request'],'facet_result':item['facet_result'],
                'facets':item['facets'],'requests':item['requests'],
                'results':results if method=='jev'else None,'decision':d,'selected_ids':ids,
                'selected_facts':selected,'selected_sha256':digest(selected),
                'baseline_selected_sha256':digest(old['selected_facts']),
                'unchanged_ordered_reader_facts':selected==old['selected_facts'],
                'source_job_sha256':digest(old),'frozen_source':m['protocol']['frozen_source'],
                'frozen_pool':realized['pool_ref'],'pool_sha256':realized['pool_sha256'],
                'pool_selected_indices':d['pool_selected_indices'],'added_ids':sorted(added),
                'removed_ids':sorted(removed),'added_outside_original_native80_ids':sorted(added-native),
                'same_membership_as_baseline':set(ids)==base,'jev_terminal_audit':jref}))
    return cm,records,jref


def _job(old,method,record,selection_ref,protocol_sha):
    return {'job_id':method+'/'+old['query_id'],'method':method,'status':'ready',
        **{k:old[k]for k in ('conversation','query_id','question','reference','category','reference_date')},
        'selected_ids':record['selected_ids'],'selected_facts':record['selected_facts'],
        'selected_sha256':record['selected_sha256'],'bank':old['bank'],'selection':selection_ref,
        'protocol_sha256':protocol_sha,'source_job_sha256':digest(old),
        'unchanged_ordered_reader_facts':record['unchanged_ordered_reader_facts'],
        'original_baseline_job_id':old['job_id'],'original_protocol_sha256':old['protocol_sha256'],
        'lineage':{k:record[k]for k in ('frozen_source','bank','original_selection','proposal',
          'facet_request','facet_result','requests','jev_terminal_audit')}}


def _count(records):
    counts=Counter()
    for old,method,r in records:
        d=r['decision'];counts[f'{method}_changed_queries']+=not r['unchanged_ordered_reader_facts']
        counts[f'{method}_changed_membership_queries']+=not r['same_membership_as_baseline']
        counts[f'{method}_retained_baseline_fact_occurrences']+=d['retained_baseline_count']
        counts[f'{method}_added_fact_occurrences']+=len(r['added_ids'])
        counts[f'{method}_added_outside_original_native80_occurrences']+=len(r['added_outside_original_native80_ids'])
        counts[f'{method}_score_status_'+d['score_status']]+=1
        if d['coverage']is not None:counts[f'{method}_zero_fill_slots']+=d['coverage']['zero_fill_count']
    return dict(counts)


def seal(run_dir):
    run=Path(run_dir).resolve()
    if(run/'selected_inputs.json').exists()or(run/'selections').exists():raise FileExistsError('Never overwrite sealed selection namespace')
    reader=Reader();m,spec,byid,pools=_manifest(reader,run)
    cm,records,jref=_expected_records(reader,run,m,spec,byid,pools);jobs=[]
    for old,method,r in records:
        ref=save_new(run/'selections'/method/f'{old["query_id"]}.json',r)
        jobs.append(_job(old,method,r,ref,m['protocol_sha256']))
    value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),
        'questions_per_arm':1540,'jobs':jobs,'parent_run':str(run),
        'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'score_manifest':reader.pointer(run/'score_manifest.json'),
        'facet_terminal_audit':cm['facet_terminal_audit'],'jev_terminal_audit':jref,
        'protocol_sha256':m['protocol_sha256'],'coverage_protocol':m['protocol']['research_protocol'],
        **{k:m['protocol'][k]for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof',
          'reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256')},
        'selection_counts':_count(records),'model_calls_during_seal':0}
    return {'source':save_new(run/'selected_inputs.json',value),'jobs':len(jobs),
        'selection_counts':value['selection_counts'],'model_calls':0}


def validate_source(path,*,fresh=False):
    path=Path(path).resolve();key=str(path);cached=_VALIDATION_CACHE.get(key)
    if cached and not fresh and all(Path(p).is_file()and(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==mark
        for p,mark in cached['stats'].items())and all(not Path(p).exists()for p in cached['absent_paths']):
        return deepcopy(cached['value'])
    reader=Reader();source=reader.read(path);run=Path(source['parent_run']).resolve()
    if(path!=run/'selected_inputs.json'or source.get('schema')!=SOURCE_SCHEMA
        or source.get('status')!='sealed'or source.get('arms')!=list(ARMS)or source.get('questions_per_arm')!=1540):
        raise ValueError('Exact R06 sealed source required')
    m,spec,byid,pools=_manifest(reader,run);cm,records,jref=_expected_records(reader,run,m,spec,byid,pools)
    if(source['protocol_sha256']!=m['protocol_sha256']or source['coverage_protocol']!=m['protocol']['research_protocol']
        or source['selector_manifest']!=reader.pointer(run/'run_manifest.json')
        or source['score_manifest']!=reader.pointer(run/'score_manifest.json')
        or source['facet_terminal_audit']!=cm['facet_terminal_audit']or source['jev_terminal_audit']!=jref
        or any(source[k]!=m['protocol'][k]for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof',
          'reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'))):
        raise ValueError('Sealed R06 scientific/source/code lineage differs')
    expected=[]
    for old,method,r in records:
        p=run/'selections'/method/f'{old["query_id"]}.json'
        if _body(reader.read(p))!=r:raise ValueError('Selected20 cannot be exactly reproduced')
        expected.append(_job(old,method,r,reader.pointer(p),m['protocol_sha256']))
    if (source['jobs']!=expected or Counter(j['method']for j in expected)!={a:1540 for a in ARMS}
        or source['selection_counts']!=_count(records)):raise ValueError('Full4620 jobs/hash/query/selection differs')
    stats={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)for p in reader.refs}
    for name in CODE:
        p=str((ROOT/name).resolve());stats[p]=(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)
    absent=[p for p in getattr(reader,'ledger_rows',{})if not Path(p).exists()]
    _VALIDATION_CACHE[key]={'stats':stats,'absent_paths':absent,'value':deepcopy(source)}
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','compile-score','check-score','seal','validate-source'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    args=parser.parse_args()
    if args.stage=='prepare':result=prepare(args.run_dir,args.source)
    elif args.stage=='compile-score':result=compile_score(args.run_dir)
    elif args.stage=='seal':result=seal(args.run_dir)
    elif args.stage=='validate-source':result={'valid':True,'jobs':len(validate_source(args.run_dir/'selected_inputs.json',fresh=True)['jobs']),'model_calls':0}
    else:
        m,refs=validate_prepared(args.run_dir,stage='facet'if args.stage=='check'else'score')
        result={'valid':True,'queries':len(m['items']),'requests':len(refs),'model_calls':0}
    import json
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
