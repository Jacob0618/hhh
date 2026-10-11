"""Nine-arm module-stack QA adapter; only ``run`` dispatches model requests.

The Reader, Judge, durable stages, raw replay and recovery admission are the
original Gemini 3.8 implementation in isolated function namespaces. Only the
sealed input cohort and run identity differ. Archived R03/R08 scores are never
request caches or inherited QA producers.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_evidence_exchange_qa_r01 as prior
from tools import run_evidence_selection_qa_r03 as proof_template
from tools.run_sdk20_modules import digest, save_new, now
from tools.run_sdk20_judge_retest_v2 import read, pointer, sha, bound, verify
from tools.run_mem0_regroup_plain_qa_v1 import publish_json
from tools.summarize_pilot400 import verified_official_metric

SCHEMA = 'module-stack-plain-qa-run-v1'
SOURCE_SCHEMA = 'module-stack-source-selected-inputs-v1'
ARMS = ('baseline', 'pool_old_bm25', 'pool_old_jev', 'pool_new_bm25',
        'pool_new_jev', 'stack_source', 'stack_support', 'stack_closure',
        'stack_support_closure')
PER_ARM = 1540
TOTAL = PER_ARM * len(ARMS)
ORIGINAL_PROOF_BINDINGS = 12289
POLICY = copy.deepcopy(prior.POLICY)
POLICY.update(arms=list(ARMS), firstpass_slots=TOTAL,
    selection='sealed_pool_inputs_or_R08_JEV20_old_R05_local_source_support_closure_stack',
    baseline_proof='original_R01_raw_replay_12289_bindings_and_current_exact_requests',
    archive_qa_scores='diagnostic_only_never_cache_or_inherited_production')
CODE = tuple(dict.fromkeys((*prior.CODE,
    'tools/run_evidence_selection_qa_r03.py',
    'jev_mem/source_append_r05.py', 'jev_mem/source_support_stack_v1.py',
    'tools/prepare_module_stack_v1.py',
    'tools/run_module_stack_qa_v1.py')))
clone = proof_template.clone


def check_full_cohort(jobs):
    """Reject missing/duplicate arms or a different query/identity population."""
    check_jobs(jobs)
    if len(jobs) != TOTAL or Counter(j['method'] for j in jobs) != {a: PER_ARM for a in ARMS}:
        raise ValueError('Nine full fixed 1540-slot arms required')
    baseline = {j['query_id']: j for j in jobs if j['method'] == 'baseline'}
    if len(baseline) != PER_ARM:
        raise ValueError('Unique baseline query identities required')
    fields = ('conversation', 'question', 'reference', 'category', 'reference_date')
    for arm in ARMS:
        arm_jobs = {j['query_id']: j for j in jobs if j['method'] == arm}
        if len(arm_jobs) != PER_ARM or set(arm_jobs) != set(baseline):
            raise ValueError('Every arm must use the identical fixed 1540-query cohort')
        for qid, job in arm_jobs.items():
            if any(job.get(k) != baseline[qid].get(k) for k in fields):
                raise ValueError('Cross-arm original QA identity differs: ' + job['job_id'])


def source_validate(path):
    from tools.prepare_module_stack_v1 import validate_source
    source = validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Sealed module-stack source required')
    check_full_cohort(source['jobs'])
    return source


SignatureCapture = prior.SignatureCapture
request_signature = prior.request_signature
reuse_plan = prior.reuse_plan
proof_for = prior.proof_for
inherited_stage = prior.inherited_stage
evaluator = prior.evaluator
Gemini38API = prior.Gemini38API
Gemini38Replay = prior.Gemini38Replay
_new_audit_stage = prior._new_audit_stage
# Historical producer totals deliberately remain 4620/10780 in this clone.
_proof_scope = {**vars(proof_template), 'SCHEMA': SCHEMA}
_revalidate_original_proof = clone(proof_template.build_reuse_proofs, _proof_scope)


def build_reuse_proofs(source):
    ref = source.get('frozen_baseline_reuse_proof')
    if not isinstance(ref, dict) or not isinstance(source.get('reuse_proof_qa_manifest'), dict):
        raise ValueError('Explicit prior raw-proof and QA-manifest bindings required')
    proof = bound(ref)
    if len(proof.get('input_sha256', {})) != ORIGINAL_PROOF_BINDINGS:
        raise ValueError('Original genuine 12289 raw-proof input bindings required')
    # This verifies all input bytes, original producer policy/code/terminals,
    # baseline facts and complete actual evaluator request signatures.
    return _revalidate_original_proof(source)


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
    proof = (build_reuse_proofs(source) if phase_kind == 'firstpass'
             else bound(read(parent['manifest']['path'])['baseline_reuse_proof']))
    plans = {j['job_id']: (reuse_plan(j, proof) if phase_kind == 'firstpass' else
             {'reader': None, 'judge_candidate': proof['entries'][j['query_id']]['judge']}) for j in jobs}
    code = {p: sha(ROOT / p) for p in CODE}
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / 'baseline_reuse_proofs.json', proof)
    value = {'schema': SCHEMA, 'created_utc': now(), 'policy': POLICY,
        'source_manifest': pointer(source_path), 'phase_kind': phase_kind,
        'phase_id': 'module_stack_qa_' + phase_kind + '_v1', 'jobs': jobs, 'parent': parent,
        'official_metric_provenance': metric, 'code_sha256': code,
        'model_calls_during_prepare': 0, 'cache_roots': [],
        'baseline_reuse_proof': pointer(run / 'baseline_reuse_proofs.json'), 'reuse_plans': plans}
    publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs),
            'protocol_sha256': digest(value), 'model_calls': 0}


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    jobs = copy.deepcopy(source['jobs'])
    check_full_cohort(jobs)
    return _new_manifest(run_dir, source_path, source, jobs=jobs, phase_kind='firstpass')


# Namespace isolation preserves both old module state and implementation bytes.
_impl = dict(prior._impl)
for name, f in tuple(_impl.items()):
    if isinstance(f, FunctionType) and f.__globals__ is prior._impl:
        _impl[name] = clone(f, _impl, name)
_ordinary_manifest = clone(prior._ordinary_manifest, _impl)
_helpers = {**vars(prior), 'SCHEMA': SCHEMA, 'SOURCE_SCHEMA': SOURCE_SCHEMA,
    'ARMS': ARMS, 'POLICY': POLICY, 'CODE': CODE, '_impl': _impl,
    '_ordinary_manifest': _ordinary_manifest, 'source_validate': source_validate,
    'build_reuse_proofs': build_reuse_proofs, '_new_manifest': _new_manifest}
_bound_manifest = clone(prior.manifest, _helpers)


def manifest(run_dir, *, _source=None):
    m = _bound_manifest(run_dir, _source=_source)
    if m['phase_kind'] == 'firstpass':
        check_full_cohort(m['jobs'])
    return m


one = clone(prior.one, _helpers)
_helpers.update(prepare=prepare, manifest=manifest, one=one)
_impl.update(_impl=_impl, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA, ARMS=ARMS,
    POLICY=POLICY, CODE=CODE, source_validate=source_validate, prepare=prepare,
    _new_manifest=_new_manifest, manifest=manifest, one=one)
for name in ('job_key', 'check_jobs', 'stage_files', 'identity', 'stage_context',
             'recovery_jobs', 'prepare_recovery', 'terminal', 'reader_call',
             'compatibility', 'resolve_pair', 'known_failure', 'run'):
    globals()[name] = _impl[name]
    _helpers[name] = _impl[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    for name in ('run', 'check', 'recover'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'prepare': result = prepare(args.run_dir, args.source)
    elif args.phase == 'recover': result = prepare_recovery(args.run_dir)
    elif args.phase == 'run': result = run(args.run_dir)
    else: result = {'phase_kind': manifest(args.run_dir)['phase_kind'], 'model_calls': 0}
    import json
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
