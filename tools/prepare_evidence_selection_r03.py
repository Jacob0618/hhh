"""Offline R03 frozenpool extraction/sealing. No retrieval or model dispatch."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.evidence_selection_r03 import (ARMS,POLICY,PROTOCOL_SHA256,digest,build_request,
 request_hash,serialized_signature,parse_scores,choose_baseline,choose_cheap,choose_jev,selected_facts)
from tools.prepare_evidence_exchange_r01 import _load_baseline
from tools.prepare_evidence_ordering_r02 import CODE as R02_CODE
SCHEMA='evidence-selection-r03-selector-manifest-v1'
SOURCE_SCHEMA='evidence-selection-r03-selected-inputs-v1'
PROPOSAL_SCHEMA='evidence-selection-r03-pool-input-v1'
REQUEST_SCHEMA='evidence-selection-r03-request-v1'
SCORE_SCHEMA='evidence-selection-r03-score-result-v1'
SELECTION_SCHEMA='evidence-selection-r03-selection-v1'
DEFAULT_SOURCE=ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/selected_inputs_tag_overlay_v1.json'
FROZEN_QA_MANIFEST=ROOT/'runs/gemini38_seven_arm_qa_full1540_v1/run_manifest.json'
FROZEN_REUSE_PROOF=ROOT/'runs/evidence_exchange_qa_r01_v1/baseline_reuse_proofs.json'
REUSE_QA_MANIFEST=ROOT/'runs/evidence_exchange_qa_r01_v1/run_manifest.json'
RESEARCH_PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json'
CODE=tuple(dict.fromkeys((*R02_CODE,'jev_mem/evidence_selection_r03.py','tools/prepare_evidence_selection_r03.py',
 'tools/score_evidence_selection_r03.py','tools/snapshot_evidence_selection_r03.py',
 'tools/run_evidence_selection_qa_r03.py','tools/summarize_evidence_selection_r03.py')))
_VALIDATION_CACHE={}


def _research_protocol(reader):
    # Read/hash identical bytes through one read(), including the unsealed research protocol.
    spec=reader.read(RESEARCH_PROTOCOL,sealed=False)
    ref=deepcopy(reader.refs[str(RESEARCH_PROTOCOL.resolve())])
    if ref['sha256']!=PROTOCOL_SHA256:raise ValueError('Three-voted R03 protocol bytes changed')
    return spec,ref


def _basis_checks(reader,spec):
    for ref in spec['basis'].values():
        reader.raw_pointer(ref['path'],ref=_canonical_ref(ref))
    r02=reader.read(spec['basis']['r02_predicate_protocol']['path'],sealed=False)
    if spec['jev']['question_instructions_template']!=r02['jev']['question_instructions_template']:
        raise ValueError('Exact R02 pointwise predicate changed')
    reuse=reader.read(REUSE_QA_MANIFEST,sealed=False)
    _validate_qa_manifest(reuse)
    if reuse['baseline_reuse_proof']!=reader.raw_pointer(FROZEN_REUSE_PROOF):
        raise ValueError('Pinned R01 actual replay proof changed')
    return reuse


def _inputs(reader,source_path,spec):
    source,source_ref,jobs,banks=_load_baseline(reader,source_path)
    pool_ref=_canonical_ref(spec['basis']['r01_pool_manifest'])
    parent=reader.bound(pool_ref)
    if (parent['protocol']['frozen_source']!=source_ref
        or parent['protocol_sha256']!=digest(parent['protocol'])
        or len(parent['items'])!=1540 or len({i['query_id']for i in parent['items']})!=1540):
        raise ValueError('Frozen R01 realizedpool manifest/source changed')
    items={i['query_id']:i for i in parent['items']}
    if set(items)!={j['query_id']for j in jobs}:raise ValueError('R01 pool and N0 cohorts differ')
    realized={}
    for old in jobs:
        item=items[old['query_id']];v=reader.bound(item['proposal']);pool=v['pool']
        if (v['schema']!='evidence-exchange-r01-proposal-v1'
            or v['protocol_sha256']!=parent['protocol_sha256']or v['original_job']!=old
            or v['source_job_sha256']!=digest(old)or v['bank']!=old['bank']
            or v['frozen_source']!=source_ref or v['original_selection']!=old['selection']
            or v['baseline_selected_sha256']!=digest(old['selected_facts'])
            or v['pool_sha256']!=digest(pool)or pool['baseline_ids']!=old['selected_ids']
            or not isinstance(pool['pool_ids'],list)or not 20<=len(pool['pool_ids'])<=80
            or len(set(pool['pool_ids']))!=len(pool['pool_ids'])
            or pool['pool_ids'][:20]!=old['selected_ids']or item['conversation']!=old['conversation']):
            raise ValueError('Frozen realizedpool/first20/query/source binding differs')
        facts={f['id']:f for f in banks[old['conversation']]['facts']}
        if not set(pool['pool_ids'])<=set(facts):raise ValueError('Candidate outside pinned same bank')
        candidates=[deepcopy(facts[fid])for fid in pool['pool_ids']]
        if candidates[:20]!=old['selected_facts']:raise ValueError('Baseline record text/date changed')
        original=reader.bound(old['selection'])['capture']['native_capture']
        native=original['candidate_ids']
        if pool['original_native80_ids']!=native:raise ValueError('Native80 source binding differs')
        if not set(pool['independent_ids']).isdisjoint(native):raise ValueError('Outside means original entire native80')
        realized[old['query_id']]={'pool_ref':item['proposal'],'pool_sha256':v['pool_sha256'],
            'pool_ids':pool['pool_ids'],'candidate_records':candidates,'original_native80_ids':native,
            'independent_ids':pool['independent_ids'],'native_other_ids':pool['native_other_ids'],
            'native_fill_ids':pool['native_fill_ids']}
    return source,source_ref,jobs,banks,realized


def _protocol(reader,source,source_ref,banks,spec,spec_ref):
    _basis_checks(reader,spec)
    codes=_codes()
    if codes['tools/pilot400_recovery_rank_v1.py']!=spec['cheap']['scorer_sha256']or codes['jev_mem/routing.py']!=spec['cheap']['tokenizer_dependency_sha256']:
        raise ValueError('Frozen BM25/tokenizer changed')
    return {'schema':SCHEMA,'policy':deepcopy(POLICY),'research_protocol':spec_ref,
        'scientific_basis':deepcopy(spec['basis']),'frozen_source':source_ref,
        'frozen_qa_run_manifest':reader.raw_pointer(FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof':reader.raw_pointer(FROZEN_REUSE_PROOF),
        'reuse_proof_qa_manifest':reader.raw_pointer(REUSE_QA_MANIFEST),
        'source_dataset_sha256':source['source_dataset_sha256'],
        'frozen_protocol_sha256':source['protocol_sha256'],'code_sha256':codes,
        'questions_per_arm':1540,'arms':list(ARMS),'fact_count':1637,
        'bank_refs':{str(ci):banks[ci]['ref']for ci in sorted(banks)},
        'official_metric_provenance':source['official_metric_provenance']}


def _prepared_value(old,source_ref,realized,req_ref,protocol_sha):
    records=realized['candidate_records']
    return {'schema':PROPOSAL_SCHEMA,'protocol_sha256':protocol_sha,'query_id':old['query_id'],
        'conversation':old['conversation'],'frozen_source':source_ref,'original_job':deepcopy(old),
        'source_job_sha256':digest(old),'bank':old['bank'],'original_selection':old['selection'],
        'request':req_ref,'baseline_selected_sha256':digest(old['selected_facts']),
        'frozen_pool':deepcopy(realized),'candidate_records_sha256':digest(records),
        'cheap_decision':choose_cheap(old['question'],old['reference_date'],records),
        'gold_not_used_by_selection':True,'lineage_job_contains_offline_reference':True}


def prepare(run_dir,source_path=DEFAULT_SOURCE):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve()or not run.name.startswith('evidence_selection_r03')or run.exists():
        raise FileExistsError('Fresh direct runs/evidence_selection_r03* namespace required')
    reader=Reader();spec,spec_ref=_research_protocol(reader)
    source,source_ref,jobs,banks,pools=_inputs(reader,source_path,spec)
    protocol=_protocol(reader,source,source_ref,banks,spec,spec_ref);psha=digest(protocol)
    run.mkdir(parents=True,exist_ok=False);items=[];summary=Counter()
    for old in jobs:
        qid=old['query_id'];realized=pools[qid];records=realized['candidate_records']
        signature=build_request(old['question'],old['reference_date'],records,spec)
        count=len(serialized_signature(signature))
        req_ref=save_new(run/'requests'/f'{qid}.json',{'schema':REQUEST_SCHEMA,'query_id':qid,
            'conversation':old['conversation'],'protocol_sha256':psha,'signature':signature,
            'request_hash':request_hash(signature),'logical_noul_count':len(records),
            'logical_request_cap':1,'max_physical_attempts_per_phase':3,'signature_utf8_bytes':count,
            'signature_utf8_byte_cap':POLICY['signature_utf8_byte_cap'],
            'dispatch_eligible':count<=POLICY['signature_utf8_byte_cap'],
            'input_tokens':None,'input_tokens_note':'UTF8 bytes are not tokens or billing.'})
        value=_prepared_value(old,source_ref,realized,req_ref,psha)
        proposal_ref=save_new(run/'proposals'/f'{qid}.json',value)
        items.append({'query_id':qid,'conversation':old['conversation'],'proposal':proposal_ref,'request':req_ref})
        summary['requests_prepared']+=1;summary['logical_nouls']+=len(records)
        summary['cheap_changed_membership_queries']+=value['cheap_decision']['changed_membership']
        summary['baseline_exact_first20_queries']+=1;summary['candidate_records']+=len(records)
        summary['oversize_not_dispatch_eligible']+=count>POLICY['signature_utf8_byte_cap']
    ref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':psha,'items':items,'summary':dict(summary),
        'model_calls_during_prepare':0,'original_source_paths_read_once':len(reader.rows)})
    return {'manifest':ref,'queries':1540,'model_calls':0,'summary':dict(summary)}


def _manifest(reader,run):
    m=reader.read(Path(run)/'run_manifest.json');spec,spec_ref=_research_protocol(reader);p=m['protocol']
    if (m.get('schema')!=SCHEMA or m.get('status')!='prepared'or m['protocol_sha256']!=digest(p)
        or p['policy']!=POLICY or p['research_protocol']!=spec_ref or p['scientific_basis']!=spec['basis']
        or p['code_sha256']!=_codes()or p['arms']!=list(ARMS)or p['questions_per_arm']!=1540
        or p['fact_count']!=1637 or len(m['items'])!=1540 or len({i['query_id']for i in m['items']})!=1540):
        raise ValueError('R03 fixed protocol/code/cohort differs')
    _basis_checks(reader,spec)
    for path,key in ((FROZEN_QA_MANIFEST,'frozen_qa_run_manifest'),(FROZEN_REUSE_PROOF,'frozen_baseline_reuse_proof'),(REUSE_QA_MANIFEST,'reuse_proof_qa_manifest')):
        reader.raw_pointer(path,ref=p[key])
    return m


def _expected_records(reader,run,m,*,require_scores=True):
    p=m['protocol'];spec,_=_research_protocol(reader)
    source,source_ref,originals,banks,pools=_inputs(reader,p['frozen_source']['path'],spec)
    if (source_ref!=p['frozen_source']or p['frozen_protocol_sha256']!=source['protocol_sha256']
        or p['source_dataset_sha256']!=source['source_dataset_sha256']
        or p['official_metric_provenance']!=source['official_metric_provenance']
        or p['bank_refs']!={str(ci):banks[ci]['ref']for ci in sorted(banks)}):
        raise ValueError('Frozen source/bank/metric changed')
    byid={j['query_id']:j for j in originals}
    if set(byid)!={i['query_id']for i in m['items']}:raise ValueError('Prepared cohort differs')
    records=[]
    for item in m['items']:
        qid=item['query_id'];old=byid[qid];realized=pools[qid];candidates=realized['candidate_records']
        _inside(item['proposal'],run);_inside(item['request'],run)
        if item['conversation']!=old['conversation']or Path(item['request']['path']).resolve()!=Path(run)/'requests'/f'{qid}.json'or Path(item['proposal']['path']).resolve()!=Path(run)/'proposals'/f'{qid}.json':
            raise ValueError('Per-query namespace/conversation differs')
        value=reader.bound(item['proposal']);expected=_prepared_value(old,source_ref,realized,item['request'],m['protocol_sha256'])
        if {k:v for k,v in value.items()if k!='content_sha256'}!=expected:raise ValueError('Prepared realized pool/record/cheap replay differs')
        signature=build_request(old['question'],old['reference_date'],candidates,spec);req=reader.bound(item['request'])
        size=len(serialized_signature(signature))
        if(req['schema']!=REQUEST_SCHEMA or req['signature']!=signature or req['query_id']!=qid
            or req['conversation']!=old['conversation']or req['protocol_sha256']!=m['protocol_sha256']
            or req['request_hash']!=request_hash(signature)or req['logical_noul_count']!=len(candidates)
            or req['logical_request_cap']!=1 or req['max_physical_attempts_per_phase']!=3
            or req['signature_utf8_bytes']!=size or req['signature_utf8_byte_cap']!=POLICY['signature_utf8_byte_cap']
            or req['dispatch_eligible']!=(size<=POLICY['signature_utf8_byte_cap'])):
            raise ValueError('Complete pool request failed exact replay')
        if not require_scores:continue
        scores,status,response_ref,raw_ref=_score(reader,run,value)
        decisions={'baseline':choose_baseline(candidates),'cheap':value['cheap_decision'],
            'jev':choose_jev(candidates,scores,score_status=status)}
        native=set(realized['original_native80_ids']);base=set(old['selected_ids'])
        for method in ARMS:
            d=decisions[method];selected=selected_facts(candidates,d['pool_selected_indices'])
            selectedids=d['selected_ids'];added=set(selectedids)-base;removed=base-set(selectedids)
            if (set(selectedids)==base)!=(selected==old['selected_facts']):
                raise ValueError('Same membership must preserve exact N0 order/records')
            records.append((old,method,{'schema':SELECTION_SCHEMA,'protocol_sha256':m['protocol_sha256'],
                'query_id':qid,'conversation':old['conversation'],'method':method,'bank':old['bank'],
                'original_selection':old['selection'],'proposal':item['proposal'],'request':item['request'],
                'response':response_ref if method=='jev'else None,'raw_response':raw_ref if method=='jev'else None,
                'raw_scores':scores if method=='jev'else None,'decision':d,'selected_ids':selectedids,
                'selected_facts':selected,'selected_sha256':digest(selected),
                'baseline_selected_sha256':digest(old['selected_facts']),
                'unchanged_ordered_reader_facts':selected==old['selected_facts'],
                'source_job_sha256':digest(old),'frozen_source':source_ref,
                'frozen_pool':realized['pool_ref'],'pool_sha256':realized['pool_sha256'],
                'pool_selected_indices':d['pool_selected_indices'],'added_ids':sorted(added),
                'removed_ids':sorted(removed),'added_outside_original_native80_ids':sorted(added-native),
                'same_membership_as_baseline':set(selectedids)==base}))
    return records


def validate_prepared(run_dir):
    run=Path(run_dir).resolve();reader=Reader();m=_manifest(reader,run)
    _expected_records(reader,run,m,require_scores=False)
    refs=[deepcopy(i['request'])for i in m['items']]
    if {Path(r['path']).resolve()for r in refs}!={p.resolve()for p in (run/'requests').glob('*.json')}:
        raise ValueError('Unmanifested/missing request present')
    return m,refs


def _canonical_ref(ref):
    return {'path':str(Path(ref['path']).resolve()),'sha256':ref['sha256']}

def _validate_qa_manifest(value):
    if value.get('protocol_sha256') != digest({k:v for k,v in value.items() if k != 'protocol_sha256'}):
        raise ValueError('Frozen R01 QA manifest protocol digest changed')

def now():
    return datetime.now(timezone.utc).isoformat()

class Reader:
    """Read/hash the identical byte buffer exactly once per unique input path."""
    def __init__(self):
        self.rows, self.refs = {}, {}

    def read(self, path, *, ref=None, sealed=True):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.rows:
            raw = path.read_bytes()
            self.refs[key] = {'path': key, 'sha256': hashlib.sha256(raw).hexdigest()}
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('JSON object artifact required')
            if sealed and value.get('content_sha256') != digest({k:v for k,v in value.items() if k != 'content_sha256'}):
                raise ValueError('Content digest changed: '+key)
            self.rows[key] = value
        if ref is not None and self.refs[key] != ref:
            raise ValueError('File source SHA/path changed: '+key)
        return self.rows[key]

    def bound(self, ref, *, sealed=True):
        if not isinstance(ref, dict) or set(ref) != {'path','sha256'}:
            raise ValueError('Exact file path/SHA pointer required')
        return self.read(ref['path'], ref=ref, sealed=sealed)

    def pointer(self, path):
        self.read(path)
        return deepcopy(self.refs[str(Path(path).resolve())])

    def raw_pointer(self, path, *, ref=None):
        key = str(Path(path).resolve())
        if key not in self.refs:
            self.refs[key] = {'path':key,'sha256':hashlib.sha256(Path(key).read_bytes()).hexdigest()}
        if ref is not None and self.refs[key] != ref:
            raise ValueError('Raw immutable pointer changed: '+key)
        return deepcopy(self.refs[key])

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k:v for k,v in value.items() if k != 'content_sha256'}
    row = {**body, 'content_sha256': digest(body)}
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {'path': str(path.resolve()), 'sha256': file_sha(path)}

def _inside(ref, run):
    path = Path(ref['path']).resolve()
    if not path.is_relative_to(Path(run).resolve()):
        raise ValueError('New artifact is outside the R03 run namespace')
    return path

def _codes():
    return {p:file_sha(ROOT/p) for p in CODE}

def _score(reader, run, proposal):
    if proposal['request'] is None:
        return None, 'not_needed', None, None
    req = reader.bound(proposal['request'])
    _inside(proposal['request'], run)
    path = Path(run)/'responses'/f'{proposal["query_id"]}.json'
    if not path.exists():
        raise ValueError('Missing result is not terminal unknown: '+proposal['query_id'])
    result = reader.read(path)
    result_ref = reader.refs[str(path.resolve())]
    if (result.get('schema') != SCORE_SCHEMA or result.get('query_id') != proposal['query_id']
        or result.get('protocol_sha256') != proposal['protocol_sha256']
        or result.get('request_hash') != req['request_hash']
        or req['request_hash'] != request_hash(req['signature'])):
        raise ValueError('Score request/protocol identity differs')
    status = result.get('status')
    if status == 'complete':
        if not req['dispatch_eligible']:
            raise ValueError('Oversize signature must not be dispatched')
        raw_ref = result.get('raw')
        raw = reader.bound(raw_ref, sealed=False)
        if raw.get('request') != req['signature']:
            raise ValueError('Full dispatched request differs from the one approved signature')
        scores = parse_scores(raw['response'], req['signature'])
        if result.get('scores') is not None and result['scores'] != scores:
            raise ValueError('Stored parsed scores differ from byte-bound raw response')
        return scores, status, result_ref, raw_ref
    if status not in ('failed','unknown','not_dispatched') or result.get('scores') is not None:
        raise ValueError('Known unavailable result requires explicit status and no synthetic scores')
    if status == 'failed' and not (result.get('error') or (result.get('error_type') and result.get('reason'))):
        raise ValueError('A failed result must document the error')
    if result.get('raw') is not None:
        raw = reader.bound(result['raw'],sealed=False)
        if raw.get('request') != req['signature']:
            raise ValueError('Unavailable result raw still must bind the exact approved signature')
    return None, status, result_ref, result.get('raw')

def _job(old, method, record, selection_ref, protocol_sha):
    return {'job_id':method+'/'+old['query_id'], 'method':method, 'status':'ready',
        **{k:old[k] for k in ('conversation','query_id','question','reference','category','reference_date')},
        'selected_ids':record['selected_ids'], 'selected_facts':record['selected_facts'],
        'selected_sha256':record['selected_sha256'], 'bank':old['bank'], 'selection':selection_ref,
        'protocol_sha256':protocol_sha, 'source_job_sha256':digest(old),
        'unchanged_ordered_reader_facts':record['unchanged_ordered_reader_facts'],
        'original_baseline_job_id':old['job_id'], 'original_protocol_sha256':old['protocol_sha256'],
        'lineage':{k:record[k] for k in ('frozen_source','bank','original_selection','proposal','request','response','raw_response')}}

def seal(run_dir):
    run = Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists():
        raise FileExistsError('Never overwrite an already sealed selected source')
    reader = Reader()
    m = _manifest(reader,run)
    records = _expected_records(reader,run,m)
    jobs, counts = [], Counter()
    for old, method, record in records:
        selection_ref = save_new(run/'selections'/method/f'{old["query_id"]}.json',record)
        jobs.append(_job(old,method,record,selection_ref,m['protocol_sha256']))
        counts[f'{method}_changed_queries'] += not record['unchanged_ordered_reader_facts']
        counts[f'{method}_changed_membership_queries'] += not record['same_membership_as_baseline']
        counts[f'{method}_retained_baseline_fact_occurrences'] += record['decision']['retained_baseline_count']
        counts[f'{method}_added_fact_occurrences'] += len(record['added_ids'])
        counts[f'{method}_added_outside_original_native80_occurrences'] += len(record['added_outside_original_native80_ids'])
        if method == 'jev':
            counts['jev_score_status_'+record['decision']['score_status']] += 1
    source_ref = save_new(run/'selected_inputs.json',{'schema':SOURCE_SCHEMA,'status':'sealed',
        'created_utc':now(), 'arms':list(ARMS), 'questions_per_arm':1540,
        'jobs':jobs, 'parent_run':str(run), 'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'], 'frozen_source':m['protocol']['frozen_source'],
        'frozen_qa_run_manifest':m['protocol']['frozen_qa_run_manifest'],
        'selection_protocol':m['protocol']['research_protocol'],
        'frozen_baseline_reuse_proof':m['protocol']['frozen_baseline_reuse_proof'],
        'reuse_proof_qa_manifest':m['protocol']['reuse_proof_qa_manifest'],
        'source_dataset_sha256':m['protocol']['source_dataset_sha256'],
        'official_metric_provenance':m['protocol']['official_metric_provenance'],
        'code_sha256':m['protocol']['code_sha256'], 'selection_counts':dict(counts),
        'model_calls_during_seal':0})
    return {'source':source_ref,'jobs':len(jobs),'model_calls':0,'selection_counts':dict(counts)}

def validate_source(path, *, fresh=False):
    """Exact offline replay; cached validation still checks all bound file stat marks.

    fresh=True forces byte/hash reads. Within a worker process unchanged files are
    not reread 1540 times; stat changes invalidate the entire cached validation.
    """
    path = Path(path).resolve()
    key = str(path)
    cached = _VALIDATION_CACHE.get(key)
    if cached and not fresh and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns) == mark
        for p,mark in cached['stats'].items()):
        return deepcopy(cached['value'])
    reader = Reader()
    source = reader.read(path)
    run = Path(source['parent_run']).resolve()
    if (path != run/'selected_inputs.json' or source.get('schema') != SOURCE_SCHEMA
        or source.get('status') != 'sealed' or source.get('arms') != list(ARMS)
        or source.get('questions_per_arm') != 1540):
        raise ValueError('New sealed R03 three-arm source required')
    m = _manifest(reader,run)
    if (source['protocol_sha256'] != m['protocol_sha256']
        or source.get('selection_protocol') != m['protocol']['research_protocol']
        or source['selector_manifest'] != reader.pointer(run/'run_manifest.json')
        or any(source[k] != m['protocol'][k] for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'))):
        raise ValueError('Sealed source is bound to a different R03 protocol')
    records = _expected_records(reader,run,m)
    expected = []
    for old,method,value in records:
        selection_path = run/'selections'/method/f'{old["query_id"]}.json'
        actual = reader.read(selection_path)
        if {k:v for k,v in actual.items() if k != 'content_sha256'} != value:
            raise ValueError('Selection cannot be reproduced from its exact scores and original bank')
        expected.append(_job(old,method,value,reader.pointer(selection_path),m['protocol_sha256']))
    if source['jobs'] != expected or Counter(j['method'] for j in expected) != {a:1540 for a in ARMS}:
        raise ValueError('Full selected20/question/gold/date/hash lineage differs')
    stats = {p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in reader.refs}
    for name in CODE:
        p = str((ROOT/name).resolve())
        stats[p] = (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)
    _VALIDATION_CACHE[key] = {'stats':stats,'value':deepcopy(source)}
    return source

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','seal','validate-source'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.run_dir,args.source)
    elif args.stage == 'check':
        m, requests = validate_prepared(args.run_dir)
        result = {'valid':True,'queries':len(m['items']),'requests':len(requests),'model_calls':0}
    elif args.stage == 'seal':
        result = seal(args.run_dir)
    else:
        value = validate_source(args.run_dir/'selected_inputs.json',fresh=True)
        result = {'valid':True,'jobs':len(value['jobs']),'model_calls':0}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
