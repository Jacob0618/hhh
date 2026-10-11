"""Offline R02 exact20 ordering preparation, replay and sealing. No dispatch."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem.evidence_ordering_r02 import (ARMS,POLICY,PROTOCOL_SHA256,digest,build_request,
    request_hash,serialized_signature,parse_scores,choose_baseline,choose_cheap,choose_jev,selected_facts)
from tools.prepare_evidence_exchange_r01 import _load_baseline
SCHEMA = 'evidence-ordering-r02-selector-manifest-v1'
SOURCE_SCHEMA = 'evidence-ordering-r02-selected-inputs-v1'
PROPOSAL_SCHEMA = 'evidence-ordering-r02-order-input-v1'
REQUEST_SCHEMA = 'evidence-ordering-r02-request-v1'
SCORE_SCHEMA = 'evidence-ordering-r02-score-result-v1'
SELECTION_SCHEMA = 'evidence-ordering-r02-selection-v1'
DEFAULT_SOURCE = ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/selected_inputs_tag_overlay_v1.json'
FROZEN_QA_MANIFEST = ROOT/'runs/gemini38_seven_arm_qa_full1540_v1/run_manifest.json'
FROZEN_REUSE_PROOF = ROOT/'runs/evidence_exchange_qa_r01_v1/baseline_reuse_proofs.json'
REUSE_QA_MANIFEST = ROOT/'runs/evidence_exchange_qa_r01_v1/run_manifest.json'
RESEARCH_PROTOCOL = ROOT/'research/jev_autoresearch_2026-10-04/R02_ordering_protocol.json'
CODE = ('jev_mem/evidence_ordering_r02.py','tools/prepare_evidence_ordering_r02.py',
 'jev_mem/evidence_exchange_r01.py','tools/prepare_evidence_exchange_r01.py',
 'tools/pilot400_recovery_rank_v1.py','jev_mem/routing.py','jev_mem/pilot400.py','jev_mem/selection.py',
 'tools/score_evidence_ordering_r02.py','tools/snapshot_evidence_ordering_r02.py',
 'tools/run_evidence_ordering_qa_r02.py','tools/summarize_evidence_ordering_r02.py',
 'tools/run_evidence_exchange_qa_r01.py','tools/summarize_evidence_exchange_r01.py',
 'tools/score_evidence_exchange_r01.py','jev_mem/gemini38_provider_v1.py','jev_mem/providers.py','jev_mem/paced_embedding_api.py',
 'jev_mem/sdk20_cached_api.py','tools/check_experiment_health.py')
_VALIDATION_CACHE = {}


def _research_protocol(reader):
    ref = reader.raw_pointer(RESEARCH_PROTOCOL)
    if ref['sha256'] != PROTOCOL_SHA256:
        raise ValueError('The three-voted R02 protocol bytes changed')
    return reader.bound(ref,sealed=False),ref


def _canonical_ref(ref):
    return {'path':str(Path(ref['path']).resolve()),'sha256':ref['sha256']}


def _validate_qa_manifest(value):
    if value.get('protocol_sha256') != digest({k:v for k,v in value.items() if k != 'protocol_sha256'}):
        raise ValueError('Frozen R01 QA manifest protocol digest changed')


def prepare(run_dir, source_path=DEFAULT_SOURCE):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT/'runs').resolve() or not run.name.startswith('evidence_ordering_r02') or run.exists():
        raise FileExistsError('Fresh direct runs/evidence_ordering_r02* namespace required')
    reader = Reader()
    spec,spec_ref = _research_protocol(reader)
    source,source_ref,jobs,banks = _load_baseline(reader,source_path)
    codes = _codes()
    basis = spec['source_basis']
    for name in ('n0_selector_manifest','n0_qa_manifest'):
        reader.raw_pointer(basis[name]['path'],ref=_canonical_ref(basis[name]))
    original_r01 = reader.bound(_canonical_ref(basis['n0_selector_manifest']))
    r01_qa = reader.bound(_canonical_ref(basis['n0_qa_manifest']),sealed=False)
    _validate_qa_manifest(r01_qa)
    proof_ref = reader.raw_pointer(FROZEN_REUSE_PROOF)
    if r01_qa['baseline_reuse_proof'] != proof_ref:
        raise ValueError('Baseline replay proof differs from pinned R01 QA manifest')
    if original_r01['protocol']['frozen_source'] != source_ref:
        raise ValueError('R02 baseline must be exactly the independently audited R01 source')
    if codes['tools/pilot400_recovery_rank_v1.py'] != spec['cheap']['scorer_sha256'] or codes['jev_mem/routing.py'] != spec['cheap']['tokenizer_dependency_sha256']:
        raise ValueError('Frozen BM25/tokenizer changed')
    protocol = {'schema':SCHEMA,'policy':deepcopy(POLICY),'research_protocol':spec_ref,
        'source_basis':deepcopy(basis),'frozen_source':source_ref,
        'frozen_qa_run_manifest':reader.raw_pointer(FROZEN_QA_MANIFEST),
        'frozen_baseline_reuse_proof':reader.raw_pointer(FROZEN_REUSE_PROOF),
        'reuse_proof_qa_manifest':reader.raw_pointer(REUSE_QA_MANIFEST,ref=_canonical_ref(basis['n0_qa_manifest'])),
        'source_dataset_sha256':source['source_dataset_sha256'],
        'frozen_protocol_sha256':source['protocol_sha256'],'code_sha256':codes,
        'questions_per_arm':1540,'arms':list(ARMS),'fact_count':1637,
        'bank_refs':{str(ci):banks[ci]['ref'] for ci in sorted(banks)},
        'official_metric_provenance':source['official_metric_provenance']}
    protocol_sha = digest(protocol)
    run.mkdir(parents=True,exist_ok=False)
    items,summary = [],Counter()
    for old in jobs:
        qid,ci = old['query_id'],old['conversation']
        signature = build_request(old['question'],old['reference_date'],old['selected_facts'],spec)
        byte_count = len(serialized_signature(signature))
        req_ref = save_new(run/'requests'/f'{qid}.json',{'schema':REQUEST_SCHEMA,
            'query_id':qid,'conversation':ci,'protocol_sha256':protocol_sha,
            'signature':signature,'request_hash':request_hash(signature),'logical_noul_count':20,
            'logical_request_cap':1,'max_physical_attempts_per_phase':3,'signature_utf8_bytes':byte_count,
            'signature_utf8_byte_cap':POLICY['signature_utf8_byte_cap'],
            'dispatch_eligible':byte_count <= POLICY['signature_utf8_byte_cap'],
            'input_tokens':None,'input_tokens_note':'Not measured; UTF8 bytes are not tokens or billing.'})
        cheap = choose_cheap(old['question'],old['reference_date'],old['selected_facts'])
        order_ref = save_new(run/'proposals'/f'{qid}.json',{'schema':PROPOSAL_SCHEMA,
            'protocol_sha256':protocol_sha,'query_id':qid,'conversation':ci,
            'frozen_source':source_ref,'original_job':deepcopy(old),'source_job_sha256':digest(old),
            'bank':old['bank'],'original_selection':old['selection'],'request':req_ref,
            'baseline_selected_sha256':digest(old['selected_facts']),'cheap_decision':cheap,
            'gold_not_used_by_ordering':True,'lineage_job_contains_offline_reference':True})
        items.append({'query_id':qid,'conversation':ci,'proposal':order_ref,'request':req_ref})
        summary['requests_prepared'] += 1
        summary['logical_nouls'] += 20
        summary['cheap_changed_queries'] += cheap['changed_order']
        summary['oversize_not_dispatch_eligible'] += byte_count > POLICY['signature_utf8_byte_cap']
    ref = save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':protocol_sha,'items':items,'summary':dict(summary),
        'model_calls_during_prepare':0,'original_source_paths_read_once':len(reader.rows)})
    return {'manifest':ref,'queries':1540,'model_calls':0,'summary':dict(summary)}


def _manifest(reader,run):
    m = reader.read(Path(run)/'run_manifest.json')
    spec,spec_ref = _research_protocol(reader)
    p = m.get('protocol',{})
    if (m.get('schema') != SCHEMA or m.get('status') != 'prepared'
        or m.get('protocol_sha256') != digest(p) or p.get('policy') != POLICY
        or p.get('research_protocol') != spec_ref or p.get('code_sha256') != _codes()
        or p.get('source_basis') != spec['source_basis']
        or p.get('questions_per_arm') != 1540 or p.get('arms') != list(ARMS)
        or p.get('fact_count') != 1637 or len(m.get('items',[])) != 1540
        or len({x['query_id'] for x in m['items']}) != 1540):
        raise ValueError('R02 pinned protocol/code/cohort differs')
    for path,key in ((FROZEN_QA_MANIFEST,'frozen_qa_run_manifest'),(FROZEN_REUSE_PROOF,'frozen_baseline_reuse_proof'),(REUSE_QA_MANIFEST,'reuse_proof_qa_manifest')):
        reader.raw_pointer(path,ref=p[key])
    for ref in spec['source_basis'].values():
        if isinstance(ref,dict) and set(ref) == {'path','sha256'}:
            reader.raw_pointer(ref['path'],ref=_canonical_ref(ref))
    if p['reuse_proof_qa_manifest'] != _canonical_ref(spec['source_basis']['n0_qa_manifest']):
        raise ValueError('Replay-proof source QA manifest changed')
    r01 = reader.bound(_canonical_ref(spec['source_basis']['n0_selector_manifest']))
    r01_qa = reader.bound(_canonical_ref(spec['source_basis']['n0_qa_manifest']),sealed=False)
    _validate_qa_manifest(r01_qa)
    if r01_qa['baseline_reuse_proof'] != p['frozen_baseline_reuse_proof']:
        raise ValueError('Replay proof is not the originally pinned proof')
    if r01['protocol']['frozen_source'] != p['frozen_source']:
        raise ValueError('R01 baseline source changed')
    return m


def _expected_records(reader,run,m,*,require_scores=True):
    p = m['protocol']
    spec,_ = _research_protocol(reader)
    source,source_ref,originals,banks = _load_baseline(reader,p['frozen_source']['path'])
    if (source_ref != p['frozen_source'] or source['protocol_sha256'] != p['frozen_protocol_sha256']
        or p['bank_refs'] != {str(ci):banks[ci]['ref'] for ci in sorted(banks)}
        or p['source_dataset_sha256'] != source['source_dataset_sha256']
        or p['official_metric_provenance'] != source['official_metric_provenance']):
        raise ValueError('Frozen baseline/source/bank/metric changed')
    jobs_by_id = {j['query_id']:j for j in originals}
    if set(jobs_by_id) != {i['query_id'] for i in m['items']}:
        raise ValueError('Prepared cohort differs from exact frozen1540')
    records = []
    for item in m['items']:
        qid = item['query_id']
        old = jobs_by_id[qid]
        if item['conversation'] != old['conversation']:
            raise ValueError('Query conversation changed')
        _inside(item['proposal'],run); _inside(item['request'],run)
        if Path(item['request']['path']).resolve() != Path(run)/'requests'/f'{qid}.json' or Path(item['proposal']['path']).resolve() != Path(run)/'proposals'/f'{qid}.json':
            raise ValueError('Per-query request/proposal namespace changed')
        v = reader.bound(item['proposal'])
        signature = build_request(old['question'],old['reference_date'],old['selected_facts'],spec)
        cheap = choose_cheap(old['question'],old['reference_date'],old['selected_facts'])
        expected = {'schema':PROPOSAL_SCHEMA,'protocol_sha256':m['protocol_sha256'],
            'query_id':qid,'conversation':old['conversation'],'frozen_source':source_ref,
            'original_job':old,'source_job_sha256':digest(old),'bank':old['bank'],
            'original_selection':old['selection'],'request':item['request'],
            'baseline_selected_sha256':digest(old['selected_facts']),'cheap_decision':cheap,
            'gold_not_used_by_ordering':True,'lineage_job_contains_offline_reference':True}
        if {k:x for k,x in v.items() if k != 'content_sha256'} != expected:
            raise ValueError('Prepared baseline/order data failed exact replay')
        req = reader.bound(item['request'])
        byte_count = len(serialized_signature(signature))
        if (req.get('schema') != REQUEST_SCHEMA or req.get('signature') != signature
            or req.get('protocol_sha256') != m['protocol_sha256']
            or req.get('request_hash') != request_hash(signature) or req.get('query_id') != qid
            or req.get('conversation') != old['conversation'] or req.get('logical_noul_count') != 20
            or req.get('logical_request_cap') != 1 or req.get('max_physical_attempts_per_phase') != 3
            or req.get('signature_utf8_bytes') != byte_count
            or req.get('signature_utf8_byte_cap') != POLICY['signature_utf8_byte_cap']
            or req.get('dispatch_eligible') != (byte_count <= POLICY['signature_utf8_byte_cap'])):
            raise ValueError('Prepared complete20 request failed exact replay')
        if not require_scores:
            continue
        scores,status,response_ref,raw_ref = _score(reader,run,v)
        decisions = {'baseline':choose_baseline(old['selected_facts']),'cheap':cheap,
            'jev':choose_jev(old['selected_facts'],scores,score_status=status)}
        for method in ARMS:
            d = decisions[method]
            selected = selected_facts(old['selected_facts'],d['permutation'])
            records.append((old,method,{'schema':SELECTION_SCHEMA,'protocol_sha256':m['protocol_sha256'],
                'query_id':qid,'conversation':old['conversation'],'method':method,'bank':old['bank'],
                'original_selection':old['selection'],'proposal':item['proposal'],'request':item['request'],
                'response':response_ref if method == 'jev' else None,'raw_response':raw_ref if method == 'jev' else None,
                'raw_scores':scores if method == 'jev' else None,'decision':d,'selected_ids':d['selected_ids'],
                'selected_facts':selected,'selected_sha256':digest(selected),
                'baseline_selected_sha256':digest(old['selected_facts']),
                'unchanged_ordered_reader_facts':selected == old['selected_facts'],
                'source_job_sha256':digest(old),'frozen_source':source_ref,
                'record_multiset_equal_baseline':True}))
    return records


def validate_prepared(run_dir):
    run = Path(run_dir).resolve()
    reader = Reader()
    m = _manifest(reader,run)
    _expected_records(reader,run,m,require_scores=False)
    refs = [deepcopy(item['request']) for item in m['items']]
    if {Path(r['path']).resolve() for r in refs} != {p.resolve() for p in (run/'requests').glob('*.json')}:
        raise ValueError('Unmanifested or missing request present')
    return m,refs


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
        raise ValueError('New artifact is outside the R02 run namespace')
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
        if method == 'jev':
            counts['jev_score_status_'+record['decision']['score_status']] += 1
    source_ref = save_new(run/'selected_inputs.json',{'schema':SOURCE_SCHEMA,'status':'sealed',
        'created_utc':now(), 'arms':list(ARMS), 'questions_per_arm':1540,
        'jobs':jobs, 'parent_run':str(run), 'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'], 'frozen_source':m['protocol']['frozen_source'],
        'frozen_qa_run_manifest':m['protocol']['frozen_qa_run_manifest'],
        'ordering_protocol':m['protocol']['research_protocol'],
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
        raise ValueError('New sealed R02 three-arm source required')
    m = _manifest(reader,run)
    if (source['protocol_sha256'] != m['protocol_sha256']
        or source.get('ordering_protocol') != m['protocol']['research_protocol']
        or source['selector_manifest'] != reader.pointer(run/'run_manifest.json')
        or any(source[k] != m['protocol'][k] for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'))):
        raise ValueError('Sealed source is bound to a different R02 protocol')
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


if __name__ == '__main__':
    main()
