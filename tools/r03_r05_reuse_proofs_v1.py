"""Offline exact-request reuse proofs from the completed module-stack QA run.

No API construction, cache import, credential access, selection, or generation.
The existing stage auditor/evaluator bytecode replays every historical stage.
The new matrix is deliberately absent: consumers are supplied by the source.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_module_stack_qa_v1 as prior
from tools import summarize_module_stack_v1 as prior_summary
from tools.run_sdk20_modules import digest, now
from tools.run_sdk20_judge_retest_v2 import bound, verify
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.summarize_sdk20_modules import fingerprint
from jev_mem.sdk20_cached_api import digest as request_hash

SCHEMA = 'r03-r05-refinement-exact-qa-reuse-proof-v1'
REF_KEYS = ('source_manifest', 'run_manifest', 'run_outcome', 'summary')
IDENTITY = ('conversation', 'query_id', 'question', 'reference', 'category', 'reference_date')
PROOF_KEYS = ('frozen_source', 'frozen_qa_run_manifest', 'frozen_baseline_reuse_proof',
              'reuse_proof_qa_manifest', 'source_dataset_sha256', 'official_metric_provenance')
CODE = tuple(dict.fromkeys((*prior_summary.CODE, 'tools/r03_r05_reuse_proofs_v1.py')))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _bound(snapshot, ref, *, sealed=False):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact file pointer required')
    path = Path(ref['path']).resolve()
    value = snapshot.read(path)
    _require(str(path) == ref['path'] and snapshot.files[path] == ref['sha256'], 'Pinned producer bytes differ')
    if sealed:
        _require(value.get('content_sha256') == digest({k: v for k, v in value.items() if k != 'content_sha256'}),
                 'Pinned producer content seal differs')
    return value


def _scope(job, stage, signature):
    return _scope_hash(job, stage, request_hash(signature))


def _scope_hash(job, stage, hashed):
    return f"{job['conversation']}/{job['query_id']}/{stage}/{hashed}"


def _consumer(job):
    return {**{k: deepcopy(job[k]) for k in IDENTITY}, 'job_id': job['job_id'],
            'method': job['method'], 'selected_ids': deepcopy(job['selected_ids']),
            'reader_request_hash': request_hash(prior.request_signature(job, 'reader'))}


def _validate_consumers(jobs, originals):
    _require(isinstance(jobs, list) and bool(jobs), 'Nonempty consumer jobs required')
    methods = tuple(dict.fromkeys(j['method'] for j in jobs))
    check = prior.clone(prior.check_jobs, {**prior.check_jobs.__globals__, 'ARMS': methods})
    check(jobs)
    _require({j['query_id'] for j in jobs} == set(originals), 'Consumer query cohort differs from producer')
    for j in jobs:
        old = originals[j['query_id']]
        _require(all(j[k] == old[k] for k in IDENTITY), 'Consumer original QA identity changed')


def _load_inputs(source):
    refs = source.get('prior_qa_reuse')
    _require(isinstance(refs, dict) and set(refs) == set(REF_KEYS), 'prior_qa_reuse requires exactly four pointers')
    snapshot = Snapshot()
    old_source = _bound(snapshot, refs['source_manifest'], sealed=True)
    m = _bound(snapshot, refs['run_manifest'])
    outcome = _bound(snapshot, refs['run_outcome'], sealed=True)
    summary = _bound(snapshot, refs['summary'], sealed=True)
    run = Path(refs['run_manifest']['path']).parent
    _require(Path(refs['run_outcome']['path']) == run / 'run_outcome.json', 'Producer outcome namespace differs')
    _require(m.get('source_manifest') == refs['source_manifest'] and m.get('phase_kind') == 'firstpass'
             and m.get('schema') == prior.SCHEMA, 'Only original module-stack firstpass is admitted')
    _require(old_source.get('schema') == prior.SOURCE_SCHEMA and old_source.get('status') == 'sealed',
             'Module-stack source is not sealed')
    _require(all(source.get(k) == old_source.get(k) for k in PROOF_KEYS), 'Original N0/source/metric bindings differ')
    _require(summary.get('schema') == prior_summary.SCHEMA and summary.get('audit_status') == 'no_invalid_artifacts'
             and summary.get('audit_errors') == [] and summary.get('source_manifest') == refs['source_manifest']
             and summary.get('model_calls_in_summary') == 0 and summary.get('native_clients') == 0,
             'Completed raw-audited producer summary required')
    phases = summary.get('phase_sources', [])
    _require(len(phases) == 1 and phases[0].get('manifest') == refs['run_manifest']
             and phases[0].get('outcome') == refs['run_outcome'], 'Summary binds another producer phase')
    _require(summary.get('declared_arms') == list(prior.ARMS) and summary.get('planned_per_arm') == prior.PER_ARM
             and summary.get('planned_total') == prior.TOTAL, 'Historical producer cardinality differs')
    _require(summary.get('summary_code_sha256') == {p: _file_sha(ROOT / p) for p in prior_summary.CODE},
             'Historical raw auditor code changed')
    hashes = summary.get('input_sha256')
    _require(isinstance(hashes, dict) and bool(hashes), 'Summary raw input bindings missing')
    for ref in (refs['run_manifest'], refs['run_outcome']):
        _require(hashes.get(ref['path']) == ref['sha256'], 'Summary omits producer terminal binding')
    def check(item):
        filename, expected = item
        _require(_file_sha(filename) == expected, 'Historical audited input changed: ' + filename)
        return Path(filename).resolve(), expected
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, hashed in pool.map(check, hashes.items()):
            previous = snapshot.files.setdefault(path, hashed)
            _require(previous == hashed, 'Conflicting historical input binding')
    prior.check_full_cohort(m['jobs'])
    originals = {j['query_id']: j for j in m['jobs'] if j['method'] == 'baseline'}
    _validate_consumers(source['jobs'], originals)
    _require(outcome.get('terminal') is True and outcome.get('total') == prior.TOTAL
             and not any(row.get('local_errors') for row in outcome.get('outcomes', [])),
             'Whole producer terminal without controller errors required')
    return snapshot, run, old_source, m


def _projection(snapshot, path, expected, api_dir, context, indexed, callback, *, no_dispatch=False):
    """Select complete same-request producer rows, retaining every consumer row.

    The full ledger was already validated by the unchanged ledger function.
    Projection removes unrelated requests from quadratic old auditor scans;
    it never removes this context's undeclared activity or a matching producer.
    """
    maps = getattr(snapshot, '_refinement_ledger_projection', None)
    if maps is None:
        contexts, requests = defaultdict(dict), defaultdict(dict)
        for key, rows in indexed.items():
            for row in rows:
                ci = row['context']['conversation']
                contexts[fingerprint(row['context'])][key] = rows
                requests[ci, key[0], key[1]][key] = rows
        maps = contexts, requests
        snapshot._refinement_ledger_projection = maps
    contexts, requests = maps
    own = contexts.get(fingerprint(context), {})
    row = snapshot.read(path)
    events = row.get('api_events', [])
    _require(isinstance(events, list), 'Stage API events missing')
    allowed = {(e.get('request_hash'), e.get('stage')) for e in events}
    _require({(k[0], k[1]) for k in own} == allowed, 'Stage and full ledger context coverage differ')
    subset = dict(own)
    for hashed, stage in allowed:
        subset.update(requests.get((context['conversation'], hashed, stage), {}))
    return prior_summary.audit_stage(snapshot, path, expected, api_dir, context, subset, callback,
                                     no_dispatch=no_dispatch)


def _inventory(snapshot, api_dir, rows):
    """Strict finite terminal inventory; no inferred successful or zero-cost work."""
    api_dir = Path(api_dir)
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['request_hash']].append(row)
    intents = {p.stem: p for p in (api_dir / 'request_intents').glob('*.json')}
    raw = {p.stem: p for p in (api_dir / 'api_cache').glob('*.json')}
    _require(all(p.is_file() and p.suffix == '.json' for folder in ('request_intents', 'api_cache')
                 for p in (api_dir / folder).glob('*')), 'Unknown partial producer cache/intent file')
    _require(set(intents) == set(raw) == set(by_hash), 'Missing/orphan/unknown request intent or raw cache')
    _require(not list((api_dir / 'cache_lineage').glob('*')), 'Imported QA cache roots are not admitted')
    refs = []
    for hashed, path in sorted(intents.items()):
        intent = snapshot.read(path)
        _require(intent.get('request_hash') == hashed == request_hash(intent.get('signature'))
                 and intent.get('retries') == 2, 'Request signature/hash/retries differ')
        physical = [r for r in by_hash[hashed] if r.get('cache_hit') is not True]
        _require(physical and len(physical) <= 3
                 and [r.get('attempt') for r in physical] == list(range(1, len(physical) + 1))
                 and all(r.get('context') == intent.get('context') and r.get('stage') == intent.get('stage')
                         and isinstance(r.get('timestamp_utc'), str) and r['timestamp_utc'] for r in physical)
                 and type(physical[-1].get('status')) is int and 200 <= physical[-1]['status'] < 300
                 and all(prior.Gemini38Replay._retry_status(r.get('status')) for r in physical[:-1]),
                 'Request lacks a complete ended successful physical sequence')
        snapshot.raw(raw[hashed])
        refs.append({'path': str(path.resolve()), 'sha256': snapshot.files[path.resolve()]})
    inventories = getattr(snapshot, '_refinement_inventories', {})
    inventories[str(api_dir.resolve())] = sorted(str(p.resolve()) for folder in ('api_cache', 'request_intents', 'cache_lineage')
                                               for p in (api_dir / folder).glob('*'))
    snapshot._refinement_inventories = inventories
    return {'intent_sources': refs, 'unknown_intents_without_ended_physical_proof': [], 'unknown_intent_count': 0,
            'ended_success_missing_raw_count': 0, 'ended_success_missing_raw': [],
            'interpretation': 'Exact intent/raw/ledger inventory and bounded physical success sequences verified.'}


def _producer_inventory(run, conversations):
    return sorted(str(p.resolve()) for ci in conversations
                  for folder in ('api', 'reader', 'judge', 'compat', 'qa')
                  for p in (Path(run) / f'conv_{ci}' / folder).rglob('*') if p.is_file())


def _audit(snapshot, run, old_source):
    # Every stage/evaluator/replayer remains original bytecode in isolated globals.
    scope = {**prior_summary.audit_phase.__globals__, 'Snapshot': lambda: snapshot,
             'audit_stage': _projection, 'intent_inventory': _inventory}
    audit = prior.clone(prior_summary.audit_phase, scope)
    with prior_summary.original.historical_guard.historical.offline():
        phase = audit(run, old_source)
    _require(phase['errors'] == [], 'Historical stages failed original raw replay')
    for root, before in snapshot._refinement_inventories.items():
        after = sorted(str(p.resolve()) for folder in ('api_cache', 'request_intents', 'cache_lineage')
                       for p in (Path(root) / folder).glob('*'))
        _require(before == after, 'Producer cache inventory changed during replay')
    expected = set()
    for record in phase['records'].values():
        for stage, path in record['paths'].items():
            path = Path(path).resolve()
            expected.add(str(path))
            _require(record[stage] is not None, 'Missing producer stage')
            if stage in ('reader', 'judge'):
                _require(record[stage]['status'] == 'complete', 'Incomplete producer stage is not inheritable')
                if record[stage].get('api_events'):
                    expected.add(str(path.with_suffix('.intent.json')))
    actual = {str(p.resolve()) for ci in {j['conversation'] for j in phase['manifest']['jobs']}
              for stage in ('reader', 'judge', 'compat', 'qa')
              for p in (run / f'conv_{ci}' / stage).rglob('*.json')}
    _require(actual == expected, 'Unexpected/missing durable stage artifact or intent')
    return phase


def _entry(job, record, stage, signature):
    return {'stage': stage, 'source_job': {k: deepcopy(job[k]) for k in (*IDENTITY, 'method', 'job_id')},
            'source_job_sha256': digest(job), 'reader': {'path': record['paths']['reader'],
                'sha256': _file_sha(record['paths']['reader'])},
            'judge': {'path': record['paths']['judge'], 'sha256': _file_sha(record['paths']['judge'])},
            'reader_result': deepcopy(record['reader']['result']), 'judge_result': deepcopy(record['judge']['result']),
            'signatures': {stage: deepcopy(signature)}, 'provider_request_hash': request_hash(signature)}


def build_reuse_proofs(source):
    """Verify the prior producer and freeze compact deterministic lookup entries.

    Only Reader signatures needed by this source are retained. All distinct
    same-query Judge signatures are retained for later actual-answer lookup.
    The old 174 MB summary/result rows are never copied into the new proof.
    """
    snapshot, run, old_source, manifest = _load_inputs(source)
    conversations = sorted({j['conversation'] for j in manifest['jobs']})
    inventory = _producer_inventory(run, conversations)
    phase = _audit(snapshot, run, old_source)
    consumers = {j['job_id']: _consumer(j) for j in source['jobs']}
    needed = {_scope_hash(j, 'reader', consumers[j['job_id']]['reader_request_hash']) for j in source['jobs']}
    entries = {}
    order = {arm: i for i, arm in enumerate(prior.ARMS)}
    for job in sorted(manifest['jobs'], key=lambda j: (order[j['method']], j['job_id'])):
        record = phase['records'][job['job_id']]
        answer = record['reader']['result']['prediction']['answer']
        for stage in ('reader', 'judge'):
            signature = prior.request_signature(job, stage, answer)
            key = _scope(job, stage, signature)
            if (stage == 'reader' and key not in needed) or key in entries:
                continue
            entries[key] = _entry(job, record, stage, signature)
    plans = {job_id: (_scope_hash(j, 'reader', j['reader_request_hash'])
                     if _scope_hash(j, 'reader', j['reader_request_hash']) in entries else None)
             for job_id, j in consumers.items()}
    _require(not snapshot.unchanged(), 'Producer input changed after raw replay')
    _require(inventory == _producer_inventory(run, conversations), 'Producer API/stage inventory changed after replay')
    _require(set(inventory) <= {str(p) for p in snapshot.files}, 'Unbound producer API/stage file')
    return {'schema': SCHEMA, 'created_utc': now(), 'prior_qa_reuse': deepcopy(source['prior_qa_reuse']),
            'evaluation_protocol': prior.evaluator.PROTOCOL, 'official_metric_provenance': source['official_metric_provenance'],
            'consumer_jobs': consumers, 'reader_matches': plans, 'entries': entries,
            'candidate_precedence': list(prior.ARMS), 'choice_semantics': 'first frozen arm/job for exact same-query full request; never label or metric based',
            'historical_jobs_replayed': len(manifest['jobs']), 'historical_reader_judge_stages_replayed': 2 * len(manifest['jobs']),
            'historical_current_run_usage': prior_summary.original.measured_usage(phase['ledger']),
            'historical_inherited_N0_usage': deepcopy(prior.proof_for(manifest)['inherited_physical_usage']),
            'historical_usage_is_new_cost': False, 'input_sha256': {str(k): v for k, v in snapshot.files.items()},
            'producer_inventory': {'run': str(run.resolve()), 'conversations': conversations, 'files': inventory},
            'proof_code_sha256': {p: _file_sha(ROOT / p) for p in CODE},
            'model_calls_during_proof': 0, 'not_independent_samples': True}


def _check_lookup(proof, job):
    _require(proof.get('schema') == SCHEMA and proof.get('model_calls_during_proof') == 0,
             'Verified refinement reuse proof required')
    _require(proof.get('consumer_jobs', {}).get(job['job_id']) == _consumer(job), 'Consumer job differs from prepared proof')


def _lookup(proof, job, stage, signature):
    _check_lookup(proof, job)
    key = _scope(job, stage, signature)
    entry = proof['entries'].get(key)
    if entry is None:
        return None
    _require(entry.get('stage') == stage and entry.get('signatures', {}).get(stage) == signature
             and entry.get('provider_request_hash') == request_hash(signature)
             and all(entry['source_job'][k] == job[k] for k in IDENTITY), 'Exact inherited request or original identity differs')
    return deepcopy(entry)


def lookup_reader(proof, job):
    signature = prior.request_signature(job, 'reader')
    entry = _lookup(proof, job, 'reader', signature)
    expected = _scope(job, 'reader', signature) if entry is not None else None
    _require(proof['reader_matches'].get(job['job_id']) == expected, 'Prepared Reader reuse plan differs')
    return entry


def lookup_judge(proof, job, reader_ref):
    """Require a byte-bound actual durable Reader before dynamic Judge reuse."""
    _check_lookup(proof, job)
    reader = bound(reader_ref)
    verify(reader_ref['path'])
    _require(reader.get('stage') == 'reader' and reader.get('status') == 'complete'
             and reader.get('source_job_sha256') == digest(job)
             and all(reader.get(k) == job[k] for k in (*IDENTITY, 'method', 'job_id', 'selected_ids')),
             'Judge reuse requires this job\'s actual complete durable Reader')
    answer = reader['result']['prediction']['answer']
    return _lookup(proof, job, 'judge', prior.request_signature(job, 'judge', answer))


def revalidate(proof, source):
    """Check saved proof and every raw binding without repeating evaluator work."""
    _require(proof.get('schema') == SCHEMA and proof.get('prior_qa_reuse') == source.get('prior_qa_reuse')
             and proof.get('model_calls_during_proof') == 0 and proof.get('not_independent_samples') is True,
             'Prior proof source/schema differs')
    _require(proof.get('proof_code_sha256') == {p: _file_sha(ROOT / p) for p in CODE}, 'Reuse proof code changed')
    _require(proof.get('consumer_jobs') == {j['job_id']: _consumer(j) for j in source['jobs']}, 'Proof consumer cohort changed')
    inventory = proof.get('producer_inventory')
    _require(isinstance(inventory, dict)
             and inventory.get('run') == str(Path(proof['prior_qa_reuse']['run_manifest']['path']).parent.resolve())
             and inventory.get('conversations') == list(range(10))
             and inventory.get('files') == _producer_inventory(inventory['run'], inventory['conversations']),
             'Producer API/stage inventory changed or missing')
    for path, hashed in proof['input_sha256'].items():
        _require(_file_sha(path) == hashed, 'Inherited raw proof input changed: ' + path)
    for job in source['jobs']:
        lookup_reader(proof, job)
    _require(inventory['files'] == _producer_inventory(inventory['run'], inventory['conversations']),
             'Producer API/stage inventory changed during revalidation')
    return {'model_calls': 0, 'bindings': len(proof['input_sha256']), 'consumer_jobs': len(source['jobs'])}
