"""Offline extraction/sealing for R01. This module NEVER dispatches an API.

prepare reads the frozen sealed source, saved native80 captures and 10 session
banks once per path. seal consumes externally dispatched byte-bound raw results.
No native index, credential, provider client, embedding or frozen writer opens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.evidence_exchange_r01 import (ARMS, POLICY, build_pool, propose_exchanges,
    build_request, request_hash, parse_scores, choose_cheap, choose_jev, selected_facts,
    fact_index, digest, pilot_ids, serialized_signature)

SCHEMA = 'evidence-exchange-r01-selector-manifest-v1'
SOURCE_SCHEMA = 'evidence-exchange-r01-selected-inputs-v1'
PROPOSAL_SCHEMA = 'evidence-exchange-r01-proposal-v1'
REQUEST_SCHEMA = 'evidence-exchange-r01-request-v1'
SCORE_SCHEMA = 'evidence-exchange-r01-score-result-v1'
SELECTION_SCHEMA = 'evidence-exchange-r01-selection-v1'
DEFAULT_SOURCE = ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/selected_inputs_tag_overlay_v1.json'
FROZEN_QA_MANIFEST = ROOT/'runs/gemini38_seven_arm_qa_full1540_v1/run_manifest.json'
CODE = ('jev_mem/evidence_exchange_r01.py', 'tools/prepare_evidence_exchange_r01.py',
    'tools/pilot400_recovery_rank_v1.py', 'jev_mem/routing.py', 'jev_mem/pilot400.py',
    'jev_mem/selection.py', 'tools/score_evidence_exchange_r01.py',
    'jev_mem/gemini38_provider_v1.py', 'jev_mem/paced_embedding_api.py',
    'jev_mem/sdk20_cached_api.py', 'tools/check_experiment_health.py')
FROZEN_SCHEMAS = {'gemini38-effective40-tag-overlay-seven-arm-selected-inputs-v1'}
_VALIDATION_CACHE = {}


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
        raise ValueError('New artifact is outside the R01 run namespace')
    return path


def _codes():
    return {p:file_sha(ROOT/p) for p in CODE}


def _baseline_jobs(source):
    jobs = [j for j in source.get('jobs',[]) if j.get('method') == 'baseline']
    if (source.get('schema') not in FROZEN_SCHEMAS or source.get('status') != 'sealed'
        or source.get('questions_per_arm') != 1540 or len(jobs) != 1540
        or len({j.get('query_id') for j in jobs}) != 1540
        or set(j.get('conversation') for j in jobs) != set(range(10))):
        raise ValueError('Frozen sealed 10-conversation baseline1540 required')
    for j in jobs:
        ci, qid = j['conversation'], j['query_id']
        if (type(ci) is not int or not re.fullmatch(rf'c{ci}_q\d+', qid)
            or j.get('job_id') != 'baseline/'+qid or j.get('status') != 'ready'
            or len(j.get('selected_ids',[])) != 20 or len(set(j['selected_ids'])) != 20
            or [f['id'] for f in j['selected_facts']] != j['selected_ids']
            or any(set(f) != {'id','text','observation_date'} for f in j['selected_facts'])):
            raise ValueError('Original fixed baseline job shape differs')
    return sorted(jobs, key=lambda j:(j['conversation'], int(j['query_id'].split('_q')[1])))


def _load_baseline(reader, source_path):
    source = reader.read(source_path)
    source_ref = reader.refs[str(Path(source_path).resolve())]
    jobs = _baseline_jobs(source)
    banks = {}
    for job in jobs:
        ci = job['conversation']
        bank = reader.bound(job['bank'])
        if (bank.get('complete') is not True or bank.get('arm') != 'session'
            or bank.get('conversation') != ci or bank.get('source_dataset_sha256') != source['source_dataset_sha256']):
            raise ValueError('Only the frozen same-conversation session bank is admitted')
        if ci in banks and banks[ci]['ref'] != job['bank']:
            raise ValueError('A conversation may not switch logical banks')
        by_id = fact_index(bank['facts'])
        if selected_facts(bank['facts'], job['selected_ids']) != job['selected_facts']:
            raise ValueError('Original selected fact text/date differs from frozen bank')
        banks[ci] = {'ref': job['bank'], 'facts': list(by_id.values())}
        old = reader.bound(job['selection'])
        native = old['capture']['native_capture']
        if (old.get('status') != 'complete' or old.get('bank_arm') != 'session'
            or old.get('conversation') != ci or old.get('query_id') != job['query_id']
            or old.get('bank') != job['bank'] or old.get('question') != job['question']
            or old.get('reference') != job['reference'] or old.get('category') != job['category']
            or old.get('reference_date') != job['reference_date']
            or native.get('base20_ids') != job['selected_ids']
            or native.get('capture_sha256') != digest({k:v for k,v in native.items() if k != 'capture_sha256'})):
            raise ValueError('Saved native capture/question identity changed')
    if len(banks) != 10 or sum(len(b['facts']) for b in banks.values()) != 1637:
        raise ValueError('Expected the unchanged10 logical session banks /1637 facts')
    return source, source_ref, jobs, banks


def prepare(run_dir, source_path=DEFAULT_SOURCE):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT/'runs').resolve() or not run.name.startswith('evidence_exchange_r01') or run.exists():
        raise FileExistsError('Fresh direct runs/evidence_exchange_r01* namespace required')
    reader = Reader()
    source, source_ref, jobs, banks = _load_baseline(reader, source_path)
    codes = _codes()
    protocol = {'schema': SCHEMA, 'policy': deepcopy(POLICY), 'frozen_source': source_ref,
        'frozen_qa_run_manifest':reader.raw_pointer(FROZEN_QA_MANIFEST),
        'source_dataset_sha256': source['source_dataset_sha256'],
        'frozen_protocol_sha256': source['protocol_sha256'], 'code_sha256': codes,
        'questions_per_arm':1540, 'arms':list(ARMS),
        'pilot_ids':pilot_ids(jobs), 'fact_count':1637,
        'bank_refs': {str(ci): banks[ci]['ref'] for ci in sorted(banks)},
        'official_metric_provenance':source['official_metric_provenance']}
    protocol_sha = digest(protocol)
    run.mkdir(parents=True, exist_ok=False)
    rows, pool_summary = [], Counter()
    for job in jobs:
        qid, ci = job['query_id'], job['conversation']
        old = reader.bound(job['selection'])
        native = old['capture']['native_capture']
        facts = banks[ci]['facts']
        pool = build_pool(job['question'], facts, job['selected_ids'], native['candidate_ids'])
        proposals = propose_exchanges(pool)
        signature = build_request(job['question'], job['reference_date'], facts, pool, proposals)
        req_ref = None
        if signature is not None:
            req_ref = save_new(run/'requests'/f'{qid}.json', {'schema': REQUEST_SCHEMA,
                'query_id':qid, 'conversation':ci, 'protocol_sha256':protocol_sha,
                'signature':signature, 'request_hash':request_hash(signature),
                'logical_noul_count':len(signature['body']['questions']), 'physical_request_cap':1,
                'signature_utf8_bytes':len(serialized_signature(signature)),
                'signature_utf8_byte_cap':POLICY['signature_utf8_byte_cap'],
                'dispatch_eligible':len(serialized_signature(signature)) <= POLICY['signature_utf8_byte_cap'],
                'input_tokens':None, 'input_tokens_note':'Not measured; byte count is not tokenizer or bill.'})
        proposal_ref = save_new(run/'proposals'/f'{qid}.json', {'schema':PROPOSAL_SCHEMA,
            'protocol_sha256':protocol_sha, 'query_id':qid, 'conversation':ci,
            'frozen_source':source_ref, 'original_job':deepcopy(job), 'source_job_sha256':digest(job),
            'bank':job['bank'], 'original_selection':job['selection'],
            'original_native_capture_sha256':native['capture_sha256'],
            'original_eligible_ids_audit_only':native['eligible_ids'],
            'pool':pool, 'pool_sha256':digest(pool), 'proposals':proposals,
            'proposals_sha256':digest(proposals), 'request':req_ref,
            'baseline_selected_sha256':digest(job['selected_facts']),
            'cheap_decision':choose_cheap(pool,proposals),
            'no_gold_or_answers_in_selector':True})
        rows.append({'query_id':qid, 'conversation':ci, 'proposal':proposal_ref, 'request':req_ref})
        pool_summary['bank_fact_documents_scanned'] += pool['bank_scan_count']
        pool_summary['independent_queries'] += 1
        pool_summary['independent_candidates_returned'] += len(pool['independent_ids'])
        pool_summary['logical_nouls'] += 3*len(proposals)
        pool_summary['requests_prepared'] += bool(req_ref)
        pool_summary['cheap_changed_queries'] += bool(proposals and choose_cheap(pool,proposals)['swaps'])
    pilot_ref = save_new(run/'pilot_manifest.json', {'schema':'evidence-exchange-r01-pilot-manifest-v1',
        'protocol_sha256':protocol_sha,'query_ids':protocol['pilot_ids'],
        'sampling':'sha256(r01-pilot-v1|conversation|query_id)_lowest8_per_conversation',
        'qa_outcomes_used':False,'provider_model':'gemini-3.8-flash',
        'thinking_level':'low','temperature':'omitted','max_output_tokens':8192})
    manifest_ref = save_new(run/'run_manifest.json', {'schema':SCHEMA, 'status':'prepared',
        'created_utc':now(), 'protocol':protocol, 'protocol_sha256':protocol_sha,
        'items':rows, 'pilot_manifest':pilot_ref, 'summary':dict(pool_summary), 'model_calls_during_prepare':0,
        'original_source_paths_read_once':len(reader.rows)})
    return {'manifest':manifest_ref, 'queries':1540, 'model_calls':0, 'summary':dict(pool_summary)}


def _manifest(reader, run):
    path = Path(run)/'run_manifest.json'
    m = reader.read(path)
    if (m.get('schema') != SCHEMA or m.get('status') != 'prepared'
        or m.get('protocol_sha256') != digest(m.get('protocol'))
        or m['protocol'].get('policy') != POLICY or m['protocol'].get('code_sha256') != _codes()
        or len(m.get('items',[])) != 1540 or len({x['query_id'] for x in m['items']}) != 1540):
        raise ValueError('R01 fixed protocol/code/complete cohort differs')
    reader.raw_pointer(FROZEN_QA_MANIFEST,ref=m['protocol']['frozen_qa_run_manifest'])
    _inside(m['pilot_manifest'],run)
    pilot = reader.bound(m['pilot_manifest'])
    if pilot.get('query_ids') != m['protocol']['pilot_ids'] or pilot.get('protocol_sha256') != m['protocol_sha256']:
        raise ValueError('Fixed80 pilot manifest changed')
    return m


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


def _expected_records(reader, run, m, *, require_scores=True):
    p = m['protocol']
    source, source_ref, originals, banks = _load_baseline(reader, p['frozen_source']['path'])
    if source_ref != p['frozen_source'] or source['protocol_sha256'] != p['frozen_protocol_sha256']:
        raise ValueError('Frozen source changed since R01 preparation')
    jobs_by_id = {j['query_id']:j for j in originals}
    if set(jobs_by_id) != {i['query_id'] for i in m['items']}:
        raise ValueError('Prepared cohort is not the frozen1540')
    records = []
    for item in m['items']:
        _inside(item['proposal'], run)
        v = reader.bound(item['proposal'])
        old = jobs_by_id[item['query_id']]
        facts = banks[item['conversation']]['facts']
        capture = reader.bound(old['selection'])['capture']['native_capture']
        pool = build_pool(old['question'], facts, old['selected_ids'], capture['candidate_ids'])
        proposals = propose_exchanges(pool)
        signature = build_request(old['question'], old['reference_date'], facts, pool, proposals)
        if (v.get('schema') != PROPOSAL_SCHEMA or v.get('protocol_sha256') != m['protocol_sha256']
            or v.get('original_job') != old or v.get('source_job_sha256') != digest(old)
            or v.get('pool') != pool or v.get('pool_sha256') != digest(pool)
            or v.get('proposals') != proposals or v.get('proposals_sha256') != digest(proposals)
            or v.get('request') != item['request'] or v.get('bank') != old['bank']
            or v.get('original_selection') != old['selection'] or v.get('frozen_source') != source_ref
            or v.get('baseline_selected_sha256') != digest(old['selected_facts'])
            or v.get('original_native_capture_sha256') != capture['capture_sha256']
            or v.get('cheap_decision') != choose_cheap(pool,proposals)):
            raise ValueError('Prepared pool/proposal/baseline failed exact replay')
        if signature is not None:
            req = reader.bound(item['request'])
            if (req.get('signature') != signature or req.get('schema') != REQUEST_SCHEMA
                or req.get('protocol_sha256') != m['protocol_sha256']
                or req.get('request_hash') != request_hash(signature)
                or req.get('query_id') != old['query_id'] or req.get('conversation') != old['conversation']
                or req.get('signature_utf8_bytes') != len(serialized_signature(signature))
                or req.get('signature_utf8_byte_cap') != POLICY['signature_utf8_byte_cap']
                or req.get('dispatch_eligible') != (len(serialized_signature(signature)) <= POLICY['signature_utf8_byte_cap'])):
                raise ValueError('Prepared request failed exact replay')
        elif item['request'] is not None:
            raise ValueError('No-proposal query must not dispatch a request')
        if not require_scores:
            continue
        scores, score_status, response_ref, raw_ref = _score(reader,run,v)
        decisions = {'baseline':{'selected_ids':old['selected_ids'], 'accepted_proposal_id':None,
            'score_status':'inherited', 'reason':'unchanged_native_baseline', 'swaps':0,
            'calibration':POLICY['calibration']},
            'cheap':choose_cheap(pool,proposals),
            'jev':choose_jev(pool,proposals,scores,score_status=score_status)}
        for method in ARMS:
            d = decisions[method]
            selected = selected_facts(facts,d['selected_ids'])
            record = {'schema':SELECTION_SCHEMA, 'protocol_sha256':m['protocol_sha256'],
                'query_id':old['query_id'], 'conversation':old['conversation'], 'method':method,
                'bank':old['bank'], 'original_selection':old['selection'],
                'proposal':item['proposal'], 'request':item['request'],
                'response':response_ref if method == 'jev' else None,
                'raw_response':raw_ref if method == 'jev' else None,
                'raw_scores':scores if method == 'jev' else None,
                'decision':d, 'selected_ids':list(d['selected_ids']),
                'selected_facts':selected, 'selected_sha256':digest(selected),
                'baseline_selected_sha256':digest(old['selected_facts']),
                'unchanged_ordered_reader_facts':selected == old['selected_facts'],
                'source_job_sha256':digest(old), 'frozen_source':source_ref}
            records.append((old,method,record))
    return records


def validate_prepared(run_dir):
    """Full1540/source/code/pool/request offline check before external dispatch.

    Returns (manifest, request_pointer_list); no responses are required here.
    """
    run = Path(run_dir).resolve()
    reader = Reader()
    m = _manifest(reader,run)
    _expected_records(reader,run,m,require_scores=False)
    return m, [deepcopy(item['request']) for item in m['items'] if item['request'] is not None]


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
        raise ValueError('New sealed R01 three-arm source required')
    m = _manifest(reader,run)
    if (source['protocol_sha256'] != m['protocol_sha256']
        or source['selector_manifest'] != reader.pointer(run/'run_manifest.json')
        or any(source[k] != m['protocol'][k] for k in ('frozen_source','frozen_qa_run_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'))):
        raise ValueError('Sealed source is bound to a different R01 protocol')
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
