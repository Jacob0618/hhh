"""R07 isolated QA adapter for persistent write-omission repair.

The sealed source owns complete proposal/admission, native-bank/index and SDK20 replay. This adapter
uses unchanged R01 Reader/Judge/stage semantics and full request inheritance.
Only run can dispatch; prepare/check are offline. No old namespaces are mutated.
"""
from __future__ import annotations
import argparse
import copy
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

SCHEMA = 'write-omission-r07-plain-qa-run-v1'
SOURCE_SCHEMA = 'write-omission-r07-selected-inputs-v1'
ARMS = prior.ARMS
POLICY = copy.deepcopy(prior.POLICY)
POLICY.update(selection='original_native_sdk20_on_persistent_supplemented_bank',
              baseline_proof='prior_raw_replay_exact_bytes_all_bindings_current_requests_revalidated')
CODE = tuple(dict.fromkeys((*prior.CODE,
    'tools/run_evidence_selection_qa_r03.py',
    'jev_mem/write_omission_r07.py', 'tools/prepare_write_omission_r07.py',
    'tools/score_write_omission_r07.py', 'tools/materialize_write_omission_r07.py', 'tools/snapshot_write_omission_r07.py',
    'tools/run_write_omission_qa_r07.py')))
clone = proof_template.clone


def source_validate(path):
    from tools.materialize_write_omission_r07 import validate_source
    source = validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('R07 sealed native write-repair source required')
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
# Prior raw-proof bytes and all original input hashes/terminal identities remain
# mandatory. The historical proof's bound byte SHA is not a new generic digest.
_proof_scope = {**vars(proof_template), 'SCHEMA': SCHEMA}
build_reuse_proofs = clone(proof_template.build_reuse_proofs, _proof_scope)


def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run = Path(run).resolve()
    if run.parent != (ROOT / 'runs').resolve() or run.exists():
        raise FileExistsError('Fresh direct runs namespace required')
    check_jobs(jobs)
    _, metric = verified_official_metric()
    if source['official_metric_provenance'] != metric:
        raise ValueError('Official scorer differs')
    proof = (build_reuse_proofs(source) if phase_kind == 'firstpass'
             else bound(read(parent['manifest']['path'])['baseline_reuse_proof']))
    plans = {j['job_id']: (reuse_plan(j, proof) if phase_kind == 'firstpass' else
             {'reader': None, 'judge_candidate': proof['entries'][j['query_id']]['judge']}) for j in jobs}
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / 'baseline_reuse_proofs.json', proof)
    value = {'schema': SCHEMA, 'created_utc': now(), 'policy': POLICY,
        'source_manifest': pointer(source_path), 'phase_kind': phase_kind,
        'phase_id': 'write_omission_r07_qa_' + phase_kind, 'jobs': jobs, 'parent': parent,
        'official_metric_provenance': metric, 'code_sha256': {p: sha(ROOT / p) for p in CODE},
        'model_calls_during_prepare': 0, 'cache_roots': [],
        'baseline_reuse_proof': pointer(run / 'baseline_reuse_proofs.json'), 'reuse_plans': plans}
    publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs),
            'protocol_sha256': digest(value), 'model_calls': 0}


_impl = dict(prior._impl)
for name, f in tuple(_impl.items()):
    if isinstance(f, FunctionType) and f.__globals__ is prior._impl:
        _impl[name] = clone(f, _impl, name)
_ordinary_manifest = clone(prior._ordinary_manifest, _impl)
_helpers = {**vars(prior), 'SCHEMA': SCHEMA, 'SOURCE_SCHEMA': SOURCE_SCHEMA,
    'ARMS': ARMS, 'POLICY': POLICY, 'CODE': CODE, '_impl': _impl,
    '_ordinary_manifest': _ordinary_manifest, 'source_validate': source_validate,
    'build_reuse_proofs': build_reuse_proofs, '_new_manifest': _new_manifest}
prepare = clone(prior.prepare, _helpers)
manifest = clone(prior.manifest, _helpers)
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
