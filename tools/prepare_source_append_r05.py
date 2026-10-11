"""Offline R05 source projection extraction, exact replay and sealing. No API calls."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.source_append_r05 import (ARMS,POLICY,PROTOCOL_SHA256,WRAPPER,digest,request_hash,
    serialized_signature,parse_scores,source_turns,build_candidates,build_request,
    choose_baseline,choose_cheap,choose_jev,selected_facts)
from tools.prepare_evidence_exchange_r01 import _load_baseline
from tools import prepare_evidence_selection_r03 as prior
Reader,save_new,file_sha,now=prior.Reader,prior.save_new,prior.file_sha,prior.now
_canonical_ref,_validate_qa_manifest=prior._canonical_ref,prior._validate_qa_manifest
DEFAULT_SOURCE=prior.DEFAULT_SOURCE
FROZEN_QA_MANIFEST=prior.FROZEN_QA_MANIFEST
FROZEN_REUSE_PROOF=prior.FROZEN_REUSE_PROOF
REUSE_QA_MANIFEST=prior.REUSE_QA_MANIFEST
RESEARCH_PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R05_source_append_protocol_final_v1.json'
SCHEMA='source-append-r05-selector-manifest-v1'
SOURCE_SCHEMA='source-append-r05-selected-inputs-v1'
PROPOSAL_SCHEMA='source-append-r05-source-input-v1'
REQUEST_SCHEMA='source-append-r05-request-v1'
SCORE_SCHEMA='source-append-r05-score-result-v1'
SELECTION_SCHEMA='source-append-r05-selection-v1'
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/source_append_r05.py','tools/prepare_source_append_r05.py',
    'tools/score_source_append_r05.py','tools/snapshot_source_append_r05.py',
    'tools/run_source_append_qa_r05.py','tools/summarize_source_append_r05.py','jev_mem/data.py')))
_VALIDATION_CACHE={}


def _research_protocol(reader):
    spec=reader.read(RESEARCH_PROTOCOL,sealed=False);ref=deepcopy(reader.refs[str(RESEARCH_PROTOCOL.resolve())])
    if ref['sha256']!=PROTOCOL_SHA256 or spec['reader_representation']['wrapper']!=WRAPPER:
        raise ValueError('Three-voted protocol/template bytes changed')
    for value in spec['input_refs'].values(): reader.raw_pointer(value['path'],ref=_canonical_ref(value))
    return spec,ref


def _proof_pointers(reader):
    manifest=reader.read(REUSE_QA_MANIFEST,sealed=False);_validate_qa_manifest(manifest)
    proofref=reader.raw_pointer(FROZEN_REUSE_PROOF)
    if manifest['baseline_reuse_proof']!=proofref: raise ValueError('Frozen R01 replay proof binding differs')
    return {'frozen_qa_run_manifest':reader.raw_pointer(FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof':proofref,'reuse_proof_qa_manifest':reader.raw_pointer(REUSE_QA_MANIFEST)}


def _inputs(reader,source_path,spec):
    source,source_ref,jobs,banks=_load_baseline(reader,source_path)
    parent=reader.bound(_canonical_ref(spec['input_refs']['n0_lineage']))
    parent_manifest=reader.bound(_canonical_ref(spec['input_refs']['n0_manifest']))
    if (parent['frozen_source']!=source_ref or parent['protocol_sha256']!=parent_manifest['protocol_sha256']
        or parent_manifest['protocol']['frozen_source']!=source_ref
        or parent_manifest['protocol_sha256']!=digest(parent_manifest['protocol'])):
        raise ValueError('N0 source not bound to frozen R01 lineage')
    baseline={j['query_id']:j for j in parent['jobs'] if j['method']=='baseline'}
    if set(baseline)!={j['query_id'] for j in jobs}: raise ValueError('Different baseline cohort')
    for old in jobs:
        prev=baseline[old['query_id']]
        if any(prev[k]!=old[k] for k in ('question','reference','category','reference_date','selected_ids','selected_facts','bank')):
            raise ValueError('Exact old baseline identity/text/date/order differs')
    allow=reader.bound(_canonical_ref(spec['input_refs']['session_source_allowlist']),sealed=False)
    entries={v['conversation']:v for v in allow['conversations']}
    if set(entries)!=set(range(10)) or len(allow['conversations'])!=10: raise ValueError('Ten authoritative bank/input pairs required')
    sources={}
    for ci,b in banks.items():
        e=entries[ci];bankref=_canonical_ref(e['bank']);groupref=_canonical_ref(e['group_inputs'])
        if bankref!=b['ref']: raise ValueError('Wrong authoritative same-conversation bank')
        bank=reader.bound(bankref);groups=reader.bound(groupref)
        if (_canonical_ref(bank['group_inputs'])!=groupref or groups['conversation']!=ci or groups['arm']!='session'
            or groups['dataset_sha256']!=source['source_dataset_sha256']): raise ValueError('Bank/source envelope lineage differs')
        turns=source_turns(groups,groupref)
        rawfacts={f['id']:f for f in bank['facts']}
        if len(rawfacts)!=len(bank['facts']): raise ValueError('Duplicate canonical fact')
        sources[ci]={'group_ref':groupref,'turns':turns,'rawfacts':rawfacts,'bm25_cache':{}}
    return source,source_ref,jobs,banks,sources


def _locator(old,sources):
    source=sources[old['conversation']]
    scopes=[deepcopy(source['rawfacts'][f['id']].get('input_scope')) for f in old['selected_facts']]
    return build_candidates(old['question'],old['selected_facts'],scopes,source['turns'],
        conversation=old['conversation'],score_cache=source['bm25_cache'])


def _codes(): return {path:file_sha(ROOT/path) for path in CODE}


def _protocol(reader,source,source_ref,banks,sources,spec,spec_ref):
    codes=_codes()
    for name in ('bm25_implementation','bm25_tokenizer_constants','writer_media_rendering_reference_only'):
        ref=spec['input_refs'][name]
        relative=Path(ref['path']).resolve().relative_to(ROOT).as_posix()
        if codes[relative]!=ref['sha256']: raise ValueError('Frozen source rendering/BM25 dependency changed')
    return {'schema':SCHEMA,'policy':deepcopy(POLICY),'research_protocol':spec_ref,
        'frozen_source':source_ref,**_proof_pointers(reader),'source_dataset_sha256':source['source_dataset_sha256'],
        'frozen_protocol_sha256':source['protocol_sha256'],'code_sha256':codes,'arms':list(ARMS),
        'questions_per_arm':1540,'fact_count':1637,'bank_refs':{str(ci):b['ref'] for ci,b in sorted(banks.items())},
        'group_input_refs':{str(ci):v['group_ref'] for ci,v in sorted(sources.items())},
        'official_metric_provenance':source['official_metric_provenance']}


def _prepared_value(old,source_ref,locator,request_ref,psha):
    return {'schema':PROPOSAL_SCHEMA,'protocol_sha256':psha,'query_id':old['query_id'],
        'conversation':old['conversation'],'frozen_source':source_ref,'original_job':deepcopy(old),
        'source_job_sha256':digest(old),'bank':old['bank'],'original_selection':old['selection'],
        'baseline_selected_sha256':digest(old['selected_facts']),'locator':locator,
        'locator_sha256':digest(locator),'request':request_ref,
        'cheap_decision':choose_cheap(old['selected_facts'],locator['candidates']),
        'gold_not_used_by_selection':True,'lineage_job_contains_offline_reference':True}


def _request_value(old,locator,spec,psha):
    candidates=locator['candidates'];n=len(candidates)
    signature=build_request(old['question'],old['reference_date'],old['selected_facts'],candidates,spec)
    size=len(serialized_signature(signature));eligible=n>=3 and size<=POLICY['signature_utf8_byte_cap']
    return {'schema':REQUEST_SCHEMA,'query_id':old['query_id'],'conversation':old['conversation'],
        'protocol_sha256':psha,'signature':signature,'request_hash':request_hash(signature),
        'logical_noul_count':n,'logical_request_cap':1,'max_physical_attempts_per_phase':3,
        'signature_utf8_bytes':size,'signature_utf8_byte_cap':POLICY['signature_utf8_byte_cap'],
        'dispatch_eligible':eligible,'skip_reason':'no_ranking_choice' if n<=2 else ('signature_oversize' if not eligible else None),
        'input_tokens':None,'input_tokens_note':'UTF8bytes are not tokens or actual billing.'}


def prepare(run_dir,source_path=DEFAULT_SOURCE):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.name.startswith('source_append_r05') or run.exists():
        raise FileExistsError('Fresh direct runs/source_append_r05* namespace required')
    reader=Reader();spec,sref=_research_protocol(reader)
    source,sourceref,jobs,banks,sources=_inputs(reader,source_path,spec)
    protocol=_protocol(reader,source,sourceref,banks,sources,spec,sref);psha=digest(protocol)
    # Fully replay source/proposer requests before creating any output namespace.
    prepared=[(old,_locator(old,sources)) for old in jobs]
    run.mkdir(parents=True,exist_ok=False);items=[];counts=Counter()
    for old,locator in prepared:
        qid=old['query_id'];req=_request_value(old,locator,spec,psha)
        reqref=save_new(run/'requests'/f'{qid}.json',req)
        pref=save_new(run/'proposals'/f'{qid}.json',_prepared_value(old,sourceref,locator,reqref,psha))
        items.append({'query_id':qid,'conversation':old['conversation'],'request':reqref,'proposal':pref})
        counts[f'candidate_count_{len(locator["candidates"])}']+=1
        counts['dispatch_eligible_queries']+=req['dispatch_eligible'];counts['logical_nouls_if_dispatched']+=len(locator['candidates']) if req['dispatch_eligible'] else 0
        counts['source_turns_scanned_occurrences']+=locator['source_turn_count']
    ref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':psha,'items':items,'summary':dict(counts),
        'model_calls_during_prepare':0,'original_source_paths_read_once':len(reader.rows)})
    return {'manifest':ref,'queries':1540,'model_calls':0,'summary':dict(counts)}


def _manifest(reader,run):
    m=reader.read(Path(run)/'run_manifest.json');spec,sref=_research_protocol(reader);p=m['protocol']
    if (m.get('schema')!=SCHEMA or m.get('status')!='prepared' or m['protocol_sha256']!=digest(p)
        or p['research_protocol']!=sref or p['policy']!=POLICY or p['code_sha256']!=_codes()
        or p['arms']!=list(ARMS) or p['questions_per_arm']!=1540 or p['fact_count']!=1637
        or len(m['items'])!=1540 or len({i['query_id'] for i in m['items']})!=1540
        or any(p[k]!=v for k,v in _proof_pointers(reader).items())):
        raise ValueError('R05 protocol/code/cohort/proof identity changed')
    return m


def _inside(ref,run): return prior._inside(ref,run)


# Isolated namespace: reuse frozen raw score replay, never change prior globals.
_score_replay=FunctionType(prior._score.__code__,{**prior._score.__globals__,'SCORE_SCHEMA':SCORE_SCHEMA},'r05_raw_score_replay')


def _score(reader,run,proposal):
    req=reader.bound(proposal['request'])
    scores,status,resultref,rawref=_score_replay(reader,run,proposal)
    if req['logical_noul_count']<=2:
        result=reader.bound(resultref)
        if status!='not_dispatched' or scores is not None or rawref is not None or result.get('reason')!='no_ranking_choice':
            raise ValueError('No-ranking queries must have terminal skip with no raw/API scores')
    return scores,status,resultref,rawref


def _expected_records(reader,run,m,*,require_scores=True):
    p=m['protocol'];spec,_=_research_protocol(reader)
    source,sourceref,jobs,banks,sources=_inputs(reader,p['frozen_source']['path'],spec)
    if _protocol(reader,source,sourceref,banks,sources,spec,p['research_protocol'])!=p:
        raise ValueError('Frozen source/protocol/group-input identities changed')
    byid={j['query_id']:j for j in jobs}
    if set(byid)!={i['query_id'] for i in m['items']}: raise ValueError('Cohort changed')
    output=[]
    for item in m['items']:
        qid=item['query_id'];old=byid[qid];locator=_locator(old,sources)
        for key,dirname in [('request','requests'),('proposal','proposals')]:
            if _inside(item[key],run)!=Path(run)/dirname/f'{qid}.json': raise ValueError('Query namespace identity differs')
        if item['conversation']!=old['conversation']: raise ValueError('Conversation differs')
        actual=reader.bound(item['proposal']);expected=_prepared_value(old,sourceref,locator,item['request'],m['protocol_sha256'])
        if {k:v for k,v in actual.items() if k!='content_sha256'}!=expected: raise ValueError('Proposer/source-offset replay differs')
        req=reader.bound(item['request'])
        if {k:v for k,v in req.items() if k!='content_sha256'}!=_request_value(old,locator,spec,m['protocol_sha256']): raise ValueError('Full source request replay differs')
        if not require_scores: continue
        scores,status,response_ref,raw_ref=_score(reader,run,actual);candidates=locator['candidates']
        decisions={'baseline':choose_baseline(old['selected_facts'],candidates),'cheap':actual['cheap_decision'],
            'jev':choose_jev(old['selected_facts'],candidates,scores,score_status=status)}
        for method,decision in decisions.items():
            facts=selected_facts(old['selected_facts'],candidates,decision)
            output.append((old,method,{'schema':SELECTION_SCHEMA,'protocol_sha256':m['protocol_sha256'],
                'query_id':qid,'conversation':old['conversation'],'method':method,'bank':old['bank'],
                'original_selection':old['selection'],'proposal':item['proposal'],'request':item['request'],
                'response':response_ref if method=='jev' else None,'raw_response':raw_ref if method=='jev' else None,
                'raw_scores':scores if method=='jev' else None,'decision':decision,
                'selected_ids':old['selected_ids'],'selected_facts':facts,'selected_sha256':digest(facts),
                'baseline_selected_sha256':digest(old['selected_facts']),
                'unchanged_ordered_reader_facts':facts==old['selected_facts'],
                'source_job_sha256':digest(old),'frozen_source':sourceref,'locator_sha256':digest(locator),
                'same_membership_as_baseline':True,'derived_view_not_new_bank_fact':True,
                'attachments':[deepcopy(candidates[i]) for i in decision['candidate_indices']]}))
    return output


def validate_prepared(run_dir):
    run=Path(run_dir).resolve();reader=Reader();m=_manifest(reader,run)
    _expected_records(reader,run,m,require_scores=False)
    refs=[deepcopy(i['request']) for i in m['items']]
    if {Path(r['path']).resolve() for r in refs}!={p.resolve() for p in (run/'requests').glob('*.json')}:
        raise ValueError('Unmanifested/missing request file')
    return m,refs


def _job(old,method,record,selection_ref,psha):
    # The mature job shape and unchanged gold/date lineage are reused, not rewritten.
    return prior._job(old,method,record,selection_ref,psha)


def seal(run_dir):
    run=Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists(): raise FileExistsError('Cannot overwrite sealed source')
    reader=Reader();m=_manifest(reader,run);rows=_expected_records(reader,run,m)
    jobs=[];counts=Counter()
    for old,method,record in rows:
        sref=save_new(run/'selections'/method/f'{old["query_id"]}.json',record)
        jobs.append(_job(old,method,record,sref,m['protocol_sha256']))
        counts[f'{method}_changed_queries']+=not record['unchanged_ordered_reader_facts']
        counts[f'{method}_attachment_occurrences']+=len(record['attachments'])
        counts[f'{method}_added_utf8_bytes']+=record['decision']['added_utf8_bytes']
        if method=='jev': counts['jev_score_status_'+record['decision']['score_status']]+=1
    p=m['protocol']
    value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),
        'questions_per_arm':1540,'jobs':jobs,'parent_run':str(run),'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'],'source_append_protocol':p['research_protocol'],
        **{k:p[k] for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof',
            'reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256')},
        'selection_counts':dict(counts),'model_calls_during_seal':0}
    ref=save_new(run/'selected_inputs.json',value)
    return {'source':ref,'jobs':len(jobs),'model_calls':0,'selection_counts':dict(counts)}


def validate_source(path,*,fresh=False):
    path=Path(path).resolve();key=str(path);cached=_VALIDATION_CACHE.get(key)
    if cached and not fresh and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==mark for p,mark in cached['stats'].items()):
        return deepcopy(cached['value'])
    reader=Reader();source=reader.read(path);run=Path(source['parent_run']).resolve()
    if (path!=run/'selected_inputs.json' or source.get('schema')!=SOURCE_SCHEMA or source.get('status')!='sealed'
        or source.get('arms')!=list(ARMS) or source.get('questions_per_arm')!=1540): raise ValueError('Sealed R05 full cohort required')
    m=_manifest(reader,run);p=m['protocol']
    if (source['protocol_sha256']!=m['protocol_sha256'] or source['source_append_protocol']!=p['research_protocol']
        or source['selector_manifest']!=reader.pointer(run/'run_manifest.json')
        or any(source[k]!=p[k] for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'))):
        raise ValueError('Selected source uses different frozen lineage')
    expected=[]
    for old,method,value in _expected_records(reader,run,m):
        sp=run/'selections'/method/f'{old["query_id"]}.json';actual=reader.read(sp)
        if {k:v for k,v in actual.items() if k!='content_sha256'}!=value: raise ValueError('Derived Reader view cannot replay from exact source offsets/scores')
        expected.append(_job(old,method,value,reader.pointer(sp),m['protocol_sha256']))
    if source['jobs']!=expected or Counter(j['method'] for j in expected)!={a:1540 for a in ARMS}:
        raise ValueError('Full derived selected20/question/gold/date/hash lineage differs')
    stats={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in reader.refs}
    for name in CODE:
        p=str((ROOT/name).resolve());stats[p]=(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)
    _VALIDATION_CACHE[key]={'stats':stats,'value':deepcopy(source)}
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','check','seal','validate-source'))
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    args=parser.parse_args()
    if args.stage=='prepare': result=prepare(args.run_dir,args.source)
    elif args.stage=='check':
        m,refs=validate_prepared(args.run_dir);result={'valid':True,'queries':len(m['items']),'requests':len(refs),'model_calls':0}
    elif args.stage=='seal': result=seal(args.run_dir)
    else: result={'valid':True,'jobs':len(validate_source(args.run_dir/'selected_inputs.json',fresh=True)['jobs']),'model_calls':0}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
