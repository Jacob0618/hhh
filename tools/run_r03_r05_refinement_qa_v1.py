"""Seven-arm refinement QA; unchanged evaluator/durable transport/recovery.

Only run dispatches. The two reference arms must inherit audited exact raw
requests. New proof routing is explicit; original module globals are isolated.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from functools import lru_cache
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_module_stack_qa_v1 as prior
from tools import r03_r05_reuse_proofs_v1 as reuse
from tools.run_sdk20_modules import digest, save_new, now
from tools.run_sdk20_judge_retest_v2 import read, pointer, sha, bound, verify
from tools.run_mem0_regroup_plain_qa_v1 import publish_json
from tools.summarize_pilot400 import verified_official_metric

SCHEMA = 'r03-r05-refinement-plain-qa-run-v1'
SOURCE_SCHEMA = 'r03-r05-refinement-selected-inputs-v1'
ARMS = ('baseline', 'r03_original', 'r03_refined', 'original_source_closure',
        'refined_source_closure', 'original_multi_source_closure', 'refined_multi_source_closure')
REFERENCE_ARMS = ('baseline', 'r03_original')
PER_ARM = 1540
TOTAL = PER_ARM * len(ARMS)
POLICY = copy.deepcopy(prior.POLICY)
POLICY.update(arms=list(ARMS), firstpass_slots=TOTAL,
    selection='sealed_r03_r05_refinement_selected20',
    baseline_proof='audited_module_stack_complete_exact_request_raw_proof',
    frozen_baseline_reuse='two_reference_arms_inherit_only_no_new_reference_dispatch',
    judge_reuse='actual_durable_reader_answer_exact_request_against_prebound_all_prior_signatures_yes_and_no',
    cache='same_phase_same_conversation_exact_request_or_verified_prior_request_proof')
CODE = tuple(dict.fromkeys((*prior.CODE, *reuse.CODE,
    'tools/prepare_r03_r05_refinement_v1.py', 'tools/run_r03_r05_refinement_qa_v1.py')))
clone = prior.clone
evaluator = prior.evaluator
SignatureCapture = prior.SignatureCapture
request_signature = prior.request_signature
Gemini38API = prior.Gemini38API
Gemini38Replay = prior.Gemini38Replay
_new_audit_stage = prior._new_audit_stage
inherited_stage = prior.inherited_stage


def check_full_cohort(jobs):
    check_jobs(jobs)
    if len(jobs) != TOTAL or Counter(j['method'] for j in jobs) != {a: PER_ARM for a in ARMS}:
        raise ValueError('Seven full fixed1540 arms required')
    baseline = {j['query_id']: j for j in jobs if j['method'] == 'baseline'}
    for arm in ARMS:
        arm_jobs = {j['query_id']: j for j in jobs if j['method'] == arm}
        if len(arm_jobs) != PER_ARM or set(arm_jobs) != set(baseline):
            raise ValueError('Every arm must use the same original1540 query cohort')
        for qid, job in arm_jobs.items():
            if any(job[k] != baseline[qid][k] for k in reuse.IDENTITY):
                raise ValueError('Cross-arm original QA identity differs')


def source_validate(path):
    from tools.prepare_r03_r05_refinement_v1 import validate_source, SOURCE_SCHEMA as prep_schema, ARMS as prep_arms
    if prep_schema != SOURCE_SCHEMA or tuple(prep_arms) != ARMS:
        raise ValueError('Refinement prep/QA source contract differs')
    source = validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Sealed refinement source required')
    check_full_cohort(source['jobs'])
    return source


@lru_cache(maxsize=4)
def _load_proof(path, hashed):
    proof = bound({'path': path, 'sha256': hashed})
    verify(path)
    return proof


def proof_for(manifest):
    ref = manifest['prior_qa_reuse_proof']
    return _load_proof(ref['path'], ref['sha256'])


def _plans(jobs, proof, phase_kind):
    plans = {}
    for j in jobs:
        candidate = reuse.lookup_reader(proof, j) if phase_kind == 'firstpass' else None
        if j['method'] in REFERENCE_ARMS:
            if phase_kind != 'firstpass' or candidate is None:
                raise ValueError('Reference arms require exact inherited Reader; no fresh dispatch')
            signature = request_signature(j, 'judge', candidate['reader_result']['prediction']['answer'])
            if reuse._lookup(proof, j, 'judge', signature) is None:
                raise ValueError('Reference arms require exact inherited Judge; no fresh dispatch')
        plans[j['job_id']] = {'reader': candidate['reader'] if candidate else None,
                             'judge': 'actual_durable_reader_exact_request_lookup'}
    return plans


def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run = Path(run).resolve()
    if run.parent != (ROOT / 'runs').resolve() or run.exists():
        raise FileExistsError('Fresh direct runs namespace required')
    check_jobs(jobs)
    if phase_kind == 'firstpass':
        check_full_cohort(jobs)
    _, metric = verified_official_metric()
    if source['official_metric_provenance'] != metric:
        raise ValueError('Official scorer differs')
    if phase_kind == 'firstpass':
        proof = reuse.build_reuse_proofs(source)
        proof_ref = None
    else:
        parent_manifest = bound(parent['manifest'])
        proof_ref = parent_manifest['prior_qa_reuse_proof']
        proof = bound(proof_ref)
        reuse.revalidate(proof, source)
    plans = _plans(jobs, proof, phase_kind)
    run.mkdir(parents=True, exist_ok=False)
    if proof_ref is None:
        proof_ref = save_new(run / 'prior_qa_reuse_proof.json', proof)
    value = {'schema': SCHEMA, 'created_utc': now(), 'policy': POLICY,
        'source_manifest': pointer(source_path), 'phase_kind': phase_kind,
        'phase_id': 'r03_r05_refinement_qa_' + phase_kind + '_v1', 'jobs': jobs, 'parent': parent,
        'official_metric_provenance': metric, 'code_sha256': {p: sha(ROOT / p) for p in CODE},
        'model_calls_during_prepare': 0, 'cache_roots': [],
        'prior_qa_reuse_proof': proof_ref, 'reuse_plans': plans}
    publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs),
            'protocol_sha256': digest(value), 'model_calls': 0,
            'inherited_reader_candidates': sum(p['reader'] is not None for p in plans.values())}


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    return _new_manifest(run_dir, source_path, source, jobs=copy.deepcopy(source['jobs']), phase_kind='firstpass')


def manifest(run_dir, *, _source=None):
    m = _ordinary_manifest(run_dir, _source=_source)
    source = _source if _source is not None else source_validate(m['source_manifest']['path'])
    # Revalidate the proof file once per manifest admission, even on an LRU hit.
    # Per-slot lookups may then share the same already-admitted immutable object.
    disk_proof = bound(m['prior_qa_reuse_proof'])
    verify(m['prior_qa_reuse_proof']['path'])
    proof = proof_for(m)
    if proof != disk_proof:
        raise ValueError('Cached reuse proof differs from current bound proof bytes')
    reuse.revalidate(proof, source)
    if m.get('reuse_plans') != _plans(m['jobs'], proof, m['phase_kind']):
        raise ValueError('Exact-request reuse plans differ')
    if m['phase_kind'] == 'firstpass':
        check_full_cohort(m['jobs'])
    return m


def one(run, j, m, api, official, initial_error=None):
    """Original stage lifecycle; the new seam only selects exact prior entries."""
    paths = stage_files(run, j)
    common = identity(j, m)
    if paths['qa'].exists():
        raise RuntimeError('Existing terminal QA retained; no implicit replay')
    plan = m['reuse_plans'][j['job_id']]
    proof = proof_for(m)
    if j.get('reader_source'):
        source = bound(j['reader_source']); verify(j['reader_source']['path'])
        if source['status'] != 'complete':
            raise ValueError('Only a successful Reader may be inherited')
        if paths['reader'].exists() or paths['reader'].with_suffix('.intent.json').exists():
            raise RuntimeError('Existing stage or unresolved intent retained; no implicit recovery restart')
        save_new(paths['reader'], {**common, 'stage': 'reader', 'status': 'complete', 'result': copy.deepcopy(source['result']),
            'error': None, 'api_events': [], 'raw_invalid_label': None, 'inherited_from': j['reader_source'], 'model_calls': 0})
        reader = verify(paths['reader'])
    elif plan['reader']:
        entry = reuse.lookup_reader(proof, j)
        if entry is None or entry['reader'] != plan['reader']:
            raise ValueError('Frozen Reader reuse plan changed')
        reader = inherited_stage(paths['reader'], {**common, 'stage': 'reader'}, entry, 'reader', m['prior_qa_reuse_proof'])
    elif j['method'] in REFERENCE_ARMS:
        raise ValueError('Reference Reader may not dispatch')
    elif initial_error is not None:
        save_new(paths['reader'], {**common, 'stage': 'reader', 'status': 'not_run', 'result': None,
            'api_events': [], 'raw_invalid_label': None, 'error': {'type': 'APIInitializationFailure',
                'message': 'No dispatch; initialization raised ' + type(initial_error).__name__}})
        reader = verify(paths['reader'])
    else:
        scoped = api.scoped(**stage_context(j, m, 'reader')) if api else None
        reader = _impl['stage_once'](paths['reader'], {**common, 'stage': 'reader'}, lambda: reader_call(scoped, j, official), scoped)
    expected = {**common, 'stage': 'judge', 'reader_sha256': sha(paths['reader'])}
    entry = reuse.lookup_judge(proof, j, pointer(paths['reader'])) if reader['status'] == 'complete' else None
    if entry is not None:
        judge = inherited_stage(paths['judge'], expected, entry, 'judge', m['prior_qa_reuse_proof'])
    elif j['method'] in REFERENCE_ARMS:
        raise ValueError('Reference Judge may not dispatch')
    elif reader['status'] == 'complete' and initial_error is None:
        scoped = api.scoped(**stage_context(j, m, 'judge')) if api else None
        judge = _impl['stage_once'](paths['judge'], expected,
            lambda: evaluator.judge_answer(scoped, j['question'], j['reference'], reader['result']['prediction']['answer']), scoped)
    else:
        error = {'type': 'ReaderFailed', 'message': 'No Judge dispatch'} if reader['status'] != 'complete' else {
            'type': 'APIInitializationFailure', 'message': 'No Judge dispatch; initialization raised ' + type(initial_error).__name__}
        save_new(paths['judge'], {**expected, 'status': 'not_run', 'result': None, 'api_events': [],
            'raw_invalid_label': None, 'error': error})
        judge = verify(paths['judge'])
    compat = compatibility(run, j, m, reader, judge, paths['compat'], api)
    if m['phase_kind'] == 'technical_recovery' and j.get('reader_source'):
        parent = j['parent_results']
        resolved = resolve_pair(bound(parent['judge']) if parent['judge'] else None,
            bound(parent['compat']) if parent['compat'] else None, judge, compat)
    elif m['phase_kind'] == 'technical_recovery':
        resolved = resolve_pair(None, None, judge, compat)
    else:
        resolved = resolve_pair(judge, compat)
    save_new(paths['qa'], {**common, 'stage': 'qa', 'status': 'complete' if reader['status'] == judge['status'] == 'complete' else 'failed',
        'reader_status': reader['status'], 'judge_status': judge['status'], 'reader': pointer(paths['reader']),
        'judge': pointer(paths['judge']), 'compat': pointer(paths['compat']),
        'reader_metrics': reader['result']['metrics'] if reader['status'] == 'complete' else None,
        'judge_correct': judge['result']['correct'] if judge['status'] == 'complete' else None, 'resolved_judge': resolved})


_impl = dict(prior._impl)
for name, f in tuple(_impl.items()):
    if isinstance(f, FunctionType) and f.__globals__ is prior._impl:
        _impl[name] = clone(f, _impl, name)
_ordinary_manifest = clone(prior._ordinary_manifest, _impl)
_impl.update(_impl=_impl, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA, ARMS=ARMS, POLICY=POLICY, CODE=CODE,
    source_validate=source_validate, prepare=prepare, _new_manifest=_new_manifest, manifest=manifest, one=one)
for name in ('job_key', 'check_jobs', 'stage_files', 'identity', 'stage_context', 'recovery_jobs',
             'prepare_recovery', 'terminal', 'reader_call', 'compatibility', 'resolve_pair', 'known_failure', 'run'):
    globals()[name] = _impl[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    for name in ('run', 'check', 'recover'):
        p = commands.add_parser(name); p.add_argument('--run-dir', type=Path, required=True)
    a = parser.parse_args()
    if a.phase == 'prepare': result = prepare(a.run_dir, a.source)
    elif a.phase == 'run': result = run(a.run_dir)
    elif a.phase == 'recover': result = prepare_recovery(a.run_dir)
    else: result = {'phase_kind': manifest(a.run_dir)['phase_kind'], 'model_calls': 0}
    import json
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
