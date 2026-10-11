"""R01 thin durable QA adapter. Only `run` can dispatch Gemini requests.

The inherited baseline is reconstructed from raw responses through the frozen
evaluator, not from a summary. Exact request reuse is provenance, not a new draw.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_plain_qa_tag_overlay_v1 as frozen
from tools import summarize_gemini38_seven_arm_v1 as audit
from tools.run_sdk20_modules import digest, save_new, now
from tools.run_sdk20_judge_retest_v2 import read, pointer, sha, bound, verify
from tools.run_mem0_regroup_plain_qa_v1 import publish_json
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.summarize_pilot400 import verified_official_metric
from jev_mem import evaluation_gemini38_v1 as evaluator
from jev_mem.gemini38_provider_v1 import Gemini38API, check_request
from tools import prepare_evidence_exchange_r01 as retrieval

SCHEMA = 'evidence-exchange-r01-plain-qa-run-v1'
SOURCE_SCHEMA = 'evidence-exchange-r01-selected-inputs-v1'
ARMS = ('baseline', 'cheap', 'jev')
POLICY = {**copy.deepcopy(frozen.POLICY), 'arms': list(ARMS), 'firstpass_slots': 4620,
          'cache': 'same_phase_same_conversation_exact_request_or_offline_verified_frozen_baseline',
          'frozen_baseline_reuse': 'exact_request_raw_intent_ledger_replay_no_summary_score_copy',
          'judge_reuse': 'actual_persisted_reader_answer_exact_full_request_against_one_prebound_old_candidate_both_phases',
          'inherited_cost': 'zero_new_dispatch_not_zero_historical_or_cold_start_cost'}
CODE = tuple(dict.fromkeys((*frozen.CODE, *retrieval.CODE,
                           'tools/run_evidence_exchange_qa_r01.py')))


def clone(f, scope, name=None):
    name = name or f.__name__
    result = FunctionType(f.__code__, scope, name, f.__defaults__, f.__closure__)
    result.__kwdefaults__ = dict(f.__kwdefaults__ or {})
    result.__module__ = __name__
    result.__qualname__ = name
    return result


def source_validate(path):
    from tools.prepare_evidence_exchange_r01 import validate_source
    source = validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('R01 sealed selector source required')
    return source


class SignatureCapture:
    """Exercise the actual evaluator and provider serializer without a client."""
    generate = Gemini38API.generate

    def __init__(self, text):
        self.text = text
        self.signatures = []

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        check_request(provider, model, body, operation, stage)
        self.signatures.append({'provider': provider, 'model': model, 'operation': operation, 'body': body})
        return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': self.text}]}}]}


def request_signature(j, stage, answer='capture only'):
    api = SignatureCapture(answer if stage == 'reader' else '[[yes]]')
    if stage == 'reader':
        evaluator.read_answer(api, j['question'], j['reference'], j['selected_facts'], reference_date=j['reference_date'])
    elif stage == 'judge':
        evaluator.judge_answer(api, j['question'], j['reference'], answer)
    else:
        raise ValueError('Reader or Judge required')
    if len(api.signatures) != 1:
        raise ValueError('Expected exactly one frozen evaluator request')
    return api.signatures[0]


def build_reuse_proofs(source):
    """One offline pass over 1540 old baseline requests, raw and stage intents."""
    ref = source.get('frozen_qa_run_manifest')
    if not isinstance(ref, dict):
        raise ValueError('Explicit frozen QA run-manifest pointer required')
    bound(ref)
    old_run = Path(ref['path']).parent
    old_manifest = read(ref['path'])
    old_source = bound(old_manifest['source_manifest'])
    if source.get('frozen_source') is not None and source['frozen_source'] != old_manifest['source_manifest']:
        raise ValueError('R01 and old QA do not bind the identical frozen source')
    # Skip recalculating unrelated old seven-arm retrieval/bank admission. The
    # original manifest still checks every code/policy/protocol/scorer/source-job
    # binding; every baseline generation is independently raw-replayed below.
    old_m = frozen.manifest(old_run, _source=old_source)
    old_out = verify(old_run / 'run_outcome.json')
    old_jobs_by_key = {frozen.job_key(j): j for j in old_m['jobs']}
    if (old_out.get('terminal') is not True or old_out.get('schema') != old_m['schema']
        or old_out.get('protocol_sha256') != old_m['protocol_sha256']
        or old_out.get('total') != 10780 or len(old_m['jobs']) != 10780
        or len(old_out.get('results', [])) != 10780
        or {frozen.job_key(j) for j in old_out['results']} != set(old_jobs_by_key)
        or any(r.get('job_id') != old_jobs_by_key[frozen.job_key(r)]['job_id'] for r in old_out['results'])):
        raise ValueError('Frozen whole10780 terminal identity/outcome differs')
    old_baseline_out = {r['query_id']: r for r in old_out['results'] if r['method'] == 'baseline'}
    if ref != pointer(old_run / 'run_manifest.json') or old_m['phase_kind'] != 'firstpass':
        raise ValueError('Only frozen complete firstpass baseline is reusable')
    old_jobs = {j['query_id']: j for j in old_m['jobs'] if j['method'] == 'baseline'}
    new_jobs = {j['query_id']: j for j in source['jobs'] if j['method'] == 'baseline'}
    if len(old_jobs) != 1540 or set(new_jobs) != set(old_jobs):
        raise ValueError('Exact 1540 baseline cohort required')
    snapshot = Snapshot()
    snapshot.raw(ref['path']); snapshot.raw(old_run / 'run_outcome.json')
    snapshot.raw(old_m['source_manifest']['path'])
    old_ledger = audit._clone(audit.old.ledger, {'runner': frozen, 'evaluator': evaluator})
    logs, indexed, refs = old_ledger(snapshot, old_run, old_m)
    # A raw cache is conversation-local. Restrict the existing replayer's index
    # to that conversation so its producer lookup does not scan ten caches.
    by_conversation = {ci: {key: rows for key, rows in indexed.items()
                           if rows and rows[0]['context']['conversation'] == ci} for ci in range(10)}
    official, provenance = verified_official_metric()
    if provenance != old_m['official_metric_provenance'] or provenance != source['official_metric_provenance']:
        raise ValueError('Official metric provenance differs')
    entries = {}
    fields = ('conversation', 'query_id', 'question', 'reference', 'category', 'reference_date', 'selected_ids', 'selected_facts')
    for qid, old_j in old_jobs.items():
        if any(new_jobs[qid].get(k) != old_j.get(k) for k in fields):
            raise ValueError('Baseline frozen input differs: ' + qid)
        paths = frozen.stage_files(old_run, old_j)
        if any(old_baseline_out[qid].get(s) != pointer(path) for s, path in paths.items()):
            raise ValueError('Frozen baseline terminal stage pointer differs: ' + qid)
        ident = frozen.identity(old_j, old_m)
        api_dir = old_run / f"conv_{old_j['conversation']}/api"
        local_index = by_conversation[old_j['conversation']]
        reader = frozen._new_audit_stage(snapshot, paths['reader'], {**ident, 'stage': 'reader'}, api_dir,
                  frozen.stage_context(old_j, old_m, 'reader'), local_index,
                  lambda api, j=old_j: frozen.reader_call(api, j, official))
        if reader['status'] != 'complete':
            raise ValueError('Frozen baseline Reader is not accepted: ' + qid)
        answer = reader['result']['prediction']['answer']
        judge = frozen._new_audit_stage(snapshot, paths['judge'],
                  {**ident, 'stage': 'judge', 'reader_sha256': snapshot.sha(paths['reader'])}, api_dir,
                  frozen.stage_context(old_j, old_m, 'judge'), local_index,
                  lambda api, j=old_j, a=answer: evaluator.judge_answer(api, j['question'], j['reference'], a))
        if judge['status'] != 'complete':
            raise ValueError('Frozen baseline Judge is not accepted: ' + qid)
        signatures = {s: request_signature(old_j, s, answer) for s in ('reader', 'judge')}
        # Replay above binds generated requests to the original event/raw/intent;
        # store those signatures so every new consumer is compared by value.
        entries[qid] = {'source_job': copy.deepcopy(old_j),
             'reader': pointer(paths['reader']), 'judge': pointer(paths['judge']),
             'reader_result': reader['result'], 'judge_result': judge['result'],
             'signatures': signatures, 'request_sha256': {s: digest(v) for s, v in signatures.items()}}
    if snapshot.unchanged():
        raise ValueError('Frozen baseline changed during offline replay')
    return {'schema': SCHEMA + '-reuse-proof', 'created_utc': now(), 'frozen_manifest': ref,
            'frozen_outcome': pointer(old_run / 'run_outcome.json'), 'entries': entries,
            'input_sha256': {str(k): v for k, v in snapshot.files.items()}, 'ledger_sources': refs,
            'inherited_physical_usage': audit.measured_usage([r for r in logs if r['context']['method'] == 'baseline']),
            'model_calls_during_proof': 0, 'not_independent_samples': True}


def reuse_plan(j, proof):
    entry = proof['entries'][j['query_id']]
    old = entry['source_job']
    # Gold/category influence metrics even though not all occur in Reader input.
    if any(j[k] != old[k] for k in ('conversation', 'question', 'reference', 'category', 'reference_date')):
        raise ValueError('Original QA identity changed')
    reader_same = request_signature(j, 'reader') == entry['signatures']['reader']
    answer = entry['reader_result']['prediction']['answer']
    judge_same = request_signature(j, 'judge', answer) == entry['signatures']['judge']
    if j['method'] == 'baseline' and not (reader_same and judge_same):
        raise ValueError('Baseline cannot dispatch a fresh request')
    return {'reader': entry['reader'] if reader_same else None,
            'judge_candidate': entry['judge']}


@lru_cache(maxsize=4)
def _load_proof(path, file_sha256):
    return bound({'path': path, 'sha256': file_sha256})


def proof_for(m):
    ref = m['baseline_reuse_proof']
    return _load_proof(ref['path'], ref['sha256'])


def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run = Path(run).resolve()
    if run.parent != (ROOT / 'runs').resolve() or run.exists():
        raise FileExistsError('Fresh direct runs namespace required')
    check_jobs(jobs)
    _, provenance = verified_official_metric()
    if source.get('official_metric_provenance') != provenance:
        raise ValueError('Official scorer differs')
    proof = build_reuse_proofs(source) if phase_kind == 'firstpass' else bound(read(parent['manifest']['path'])['baseline_reuse_proof'])
    plans = {j['job_id']: (reuse_plan(j, proof) if phase_kind == 'firstpass' else
             {'reader': None, 'judge_candidate': proof['entries'][j['query_id']]['judge']}) for j in jobs}
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / 'baseline_reuse_proofs.json', proof)
    value = {'schema': SCHEMA, 'created_utc': now(), 'policy': POLICY, 'source_manifest': pointer(source_path),
       'phase_kind': phase_kind, 'phase_id': 'evidence_exchange_r01_qa_' + phase_kind,
       'jobs': jobs, 'parent': parent, 'official_metric_provenance': provenance,
       'code_sha256': {p: sha(ROOT / p) for p in CODE}, 'model_calls_during_prepare': 0,
       'cache_roots': [], 'baseline_reuse_proof': pointer(run / 'baseline_reuse_proofs.json'), 'reuse_plans': plans}
    publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs), 'protocol_sha256': digest(value), 'model_calls': 0}


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    jobs = copy.deepcopy(source['jobs'])
    if len(jobs) != 4620 or Counter(j['method'] for j in jobs) != {a: 1540 for a in ARMS}:
        raise ValueError('Three full 1540-slot arms required')
    return _new_manifest(run_dir, source_path, source, jobs=jobs, phase_kind='firstpass')


def manifest(run_dir, *, _source=None):
    m = _ordinary_manifest(run_dir, _source=_source)
    proof = proof_for(m)
    for path, hashed in proof['input_sha256'].items():
        if not Path(path).is_file() or sha(path) != hashed:
            raise ValueError('Frozen inherited input changed: ' + path)
    expected = {j['job_id']: (reuse_plan(j, proof) if m['phase_kind'] == 'firstpass' else
             {'reader': None, 'judge_candidate': proof['entries'][j['query_id']]['judge']}) for j in m['jobs']}
    if m.get('reuse_plans') != expected:
        raise ValueError('Exact-request reuse plan differs')
    return m


def inherited_stage(path, common, entry, stage, proof_ref):
    if Path(path).exists() or Path(path).with_suffix('.intent.json').exists():
        raise RuntimeError('Existing stage or unresolved intent retained; no implicit inheritance/restart')
    original_reader = {'original_reader': entry['reader'], 'original_judge_reader_sha256': entry['reader']['sha256']} if stage == 'judge' else {}
    save_new(path, {**common, **original_reader, 'status': 'complete', 'result': copy.deepcopy(entry[stage + '_result']),
           'error': None, 'api_events': [], 'raw_invalid_label': None, 'inherited_from': entry[stage],
           'inheritance_kind': 'offline_verified_frozen_exact_request', 'reuse_proof': proof_ref, 'model_calls': 0})
    return verify(path)


def one(run, j, m, api, official, initial_error=None):
    """Frozen durable stage semantics with two explicit, proven inheritance branches."""
    paths = stage_files(run, j); common = identity(j, m)
    if paths['qa'].exists():
        raise RuntimeError('Existing terminal QA retained; no implicit replay')
    plan = m['reuse_plans'][j['job_id']]
    entry = proof_for(m)['entries'][j['query_id']]
    if j.get('reader_source'):
        source = bound(j['reader_source']); verify(j['reader_source']['path'])
        if source['status'] != 'complete': raise ValueError('Only a successful Reader may be inherited')
        if paths['reader'].exists() or paths['reader'].with_suffix('.intent.json').exists():
            raise RuntimeError('Existing stage or unresolved intent retained; no implicit recovery restart')
        save_new(paths['reader'], {**common, 'stage':'reader', 'status':'complete', 'result':copy.deepcopy(source['result']),
              'error':None, 'api_events':[], 'raw_invalid_label':None, 'inherited_from':j['reader_source'], 'model_calls':0})
        reader = verify(paths['reader'])
    elif plan['reader']:
        reader = inherited_stage(paths['reader'], {**common, 'stage': 'reader'}, entry, 'reader', m['baseline_reuse_proof'])
    elif initial_error is not None:
        save_new(paths['reader'], {**common, 'stage': 'reader', 'status': 'not_run', 'result': None,
           'api_events': [], 'raw_invalid_label': None, 'error': {'type': 'APIInitializationFailure',
           'message': 'No dispatch; initialization raised ' + type(initial_error).__name__}})
        reader = verify(paths['reader'])
    else:
        scoped = api.scoped(**stage_context(j, m, 'reader')) if api else None
        reader = _impl['stage_once'](paths['reader'], {**common, 'stage': 'reader'}, lambda: reader_call(scoped, j, official), scoped)
    expected = {**common, 'stage': 'judge', 'reader_sha256': sha(paths['reader'])}
    # Judge serialization uses the ACTUAL durable Reader answer, independently
    # of whether the Reader request/facts matched. There is one prebound source,
    # never a search among outcomes. Both old YES and old NO are eligible.
    judge_exact = reader['status'] == 'complete' and request_signature(j, 'judge',
                  reader['result']['prediction']['answer']) == entry['signatures']['judge']
    if plan['judge_candidate'] and judge_exact:
        judge = inherited_stage(paths['judge'], expected, entry, 'judge', m['baseline_reuse_proof'])
    elif reader['status'] == 'complete' and initial_error is None:
        scoped = api.scoped(**stage_context(j, m, 'judge')) if api else None
        judge = _impl['stage_once'](paths['judge'], expected,
                lambda: evaluator.judge_answer(scoped, j['question'], j['reference'], reader['result']['prediction']['answer']), scoped)
    else:
        error = {'type': 'ReaderFailed', 'message': 'No Judge dispatch'} if reader['status'] != 'complete' else {
                'type': 'APIInitializationFailure', 'message': 'No Judge dispatch; initialization raised ' + type(initial_error).__name__}
        save_new(paths['judge'], {**expected, 'status': 'not_run', 'result': None, 'api_events': [], 'raw_invalid_label': None, 'error': error})
        judge = verify(paths['judge'])
    compat = compatibility(run, j, m, reader, judge, paths['compat'], api)
    if m['phase_kind'] == 'technical_recovery' and j.get('reader_source'):
        prior = j['parent_results']; first_judge = bound(prior['judge']) if prior['judge'] else None
        first_compat = bound(prior['compat']) if prior['compat'] else None
        resolved = resolve_pair(first_judge, first_compat, judge, compat)
    elif m['phase_kind'] == 'technical_recovery': resolved = resolve_pair(None, None, judge, compat)
    else: resolved = resolve_pair(judge, compat)
    save_new(paths['qa'], {**common, 'stage': 'qa', 'status': 'complete' if reader['status'] == judge['status'] == 'complete' else 'failed',
       'reader_status': reader['status'], 'judge_status': judge['status'], 'reader': pointer(paths['reader']),
       'judge': pointer(paths['judge']), 'compat': pointer(paths['compat']),
       'reader_metrics': reader['result']['metrics'] if reader['status'] == 'complete' else None,
       'judge_correct': judge['result']['correct'] if judge['status'] == 'complete' else None,
       'resolved_judge': resolved})


_impl = {**vars(frozen), **frozen._impl}
for _name, _function in tuple(frozen._impl.items()):
    if isinstance(_function, FunctionType) and _function.__globals__ is frozen._impl:
        _impl[_name] = clone(_function, _impl, _name)
_ordinary_manifest = _impl['manifest']
_ordinary_one = _impl['one']
_impl.update(_impl=_impl, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA, ARMS=ARMS, POLICY=POLICY,
             CODE=CODE, source_validate=source_validate, prepare=prepare, _new_manifest=_new_manifest,
             manifest=manifest, one=one)
for _name in ('job_key', 'check_jobs', 'stage_files', 'identity', 'stage_context', 'recovery_jobs',
              'prepare_recovery', 'terminal', 'reader_call', 'compatibility', 'resolve_pair', 'known_failure', 'run'):
    globals()[_name] = _impl[_name]
Gemini38Replay = frozen.Gemini38Replay
_new_audit_stage = frozen._new_audit_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    for name in ('run', 'check', 'recover'):
        p = sub.add_parser(name); p.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'prepare': result = prepare(args.run_dir, args.source)
    elif args.phase == 'recover': result = prepare_recovery(args.run_dir)
    elif args.phase == 'run': result = run(args.run_dir)
    else: result = {'phase_kind': manifest(args.run_dir)['phase_kind'], 'model_calls': 0}
    import json
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
