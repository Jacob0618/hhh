"""R09 zero-update locked-copy technical adapter; original scientific QA unchanged.

The sealed source owns complete source/revision, native UPDATE/index and SDK20 replay. This adapter
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

SCHEMA = 'qualifier-repair-r09-retrieval-plain-qa-run-v1'
SOURCE_SCHEMA = 'qualifier-repair-r09-selected-inputs-v1'
ARMS = prior.ARMS
POLICY = copy.deepcopy(prior.POLICY)
POLICY.update(selection='original_native_sdk20_on_persistent_qualifier_repaired_bank',
              baseline_proof='prior_raw_replay_exact_bytes_all_bindings_current_requests_revalidated',
              recovery='once_after_whole_phase_only_proven_exhausted_transport_or_retryable_http',
              ineligible_recovery='known_malformed_http200_nonretryable_http_and_valid_labels_preserved',
              native_gate='authenticated_effective20_and_zero_update_locked_copy_exact_retrieval',
              retrieval_technical_protocol_sha256='ce3c86a30b531dee4c4085f17818acf6c367eea9f2f24b1ddcd03aeafcbae8c2',
              snapshot_technical_protocol_sha256='4836f4b2a9bb8684bdc89534cf635721e3fc2411408d9857ec5b7c317ce10321')
CODE = tuple(dict.fromkeys((*prior.CODE,
    'tools/run_evidence_selection_qa_r03.py',
    'jev_mem/qualifier_repair_r09.py', 'tools/prepare_qualifier_repair_r09.py',
    'tools/score_qualifier_repair_r09.py', 'tools/materialize_qualifier_repair_r09.py',
    'tools/run_qualifier_repair_qa_r09.py', 'tools/r09_native_update_audit.py',
    'tools/materialize_qualifier_repair_snapshot_r09.py', 'tools/r09_snapshot_native_support.py',
    'tools/run_qualifier_repair_snapshot_qa_r09.py',
    'tools/retrieve_qualifier_repair_compatible_r09.py',
    'tools/run_qualifier_repair_retrieval_qa_r09.py')))
clone = proof_template.clone


def technical_bindings(source):
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('R09 sealed native qualifier-repair source required')
    for name in ('snapshot_compatibility_protocol', 'effective_bank_gate', 'technical_manifest',
                 'retrieval_compatibility_protocol', 'retrieval_manifest', 'retrieval_gate'):
        ref = source.get(name)
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('Explicit authenticated snapshot source binding required: ' + name)
        bound(ref)
    if source['snapshot_compatibility_protocol']['sha256'] != POLICY['snapshot_technical_protocol_sha256']:
        raise ValueError('Snapshot technical protocol differs')
    if source['retrieval_compatibility_protocol']['sha256'] != POLICY['retrieval_technical_protocol_sha256']:
        raise ValueError('Retrieval technical protocol differs')
    if source.get('qualifier_repair_protocol', {}).get('sha256') != '59908bea9a269f5c79bb30a40bf30df165362bc5f5e291101f3e63b8731efc4b':
        raise ValueError('Frozen R09 science differs')
    return source


def source_validate(path):
    from tools.retrieve_qualifier_repair_compatible_r09 import validate_source
    return technical_bindings(validate_source(Path(path)))


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
        'phase_id': 'qualifier_repair_r09_retrieval_qa_' + phase_kind, 'jobs': jobs, 'parent': parent,
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


# The broad inherited verifier establishes raw/producer/consumer/intent/ledger
# provenance for every failure first. The new scientific protocol is narrower:
# successful HTTP but invalid text is retained without another model draw.
_broad_recovery_jobs = recovery_jobs
_RETRYABLE_TERMINALS = {'transport_error', 429, 500, 502, 503, 504, 529}


def recovery_jobs(parent, out):
    candidates = _broad_recovery_jobs(parent, out)
    admitted = []
    for candidate in candidates:
        stage = 'judge' if candidate['reader_source'] else 'reader'
        ref = candidate['parent_results'][stage]
        record = bound(ref)
        events = record.get('api_events', [])
        if len(events) != 1:
            raise ValueError('Narrow recovery needs the audited single request event')
        event = events[0]
        api_dir = Path(ref['path']).parents[2] / 'api'
        raw_path = api_dir / 'api_cache' / (event['request_hash'] + '.json')
        if Path(event['cache_path']).resolve() != raw_path.resolve():
            raise ValueError('Narrow recovery raw path differs from audited event')
        if raw_path.exists():
            continue  # HTTP200/schema/label failure is known, never retryable here.
        intent = read(api_dir / 'request_intents' / (event['request_hash'] + '.json'))
        import json
        from tools.check_experiment_health import shared_read
        data = shared_read(api_dir / 'api_calls.jsonl')
        if data and not data.endswith(b'\n'):
            raise ValueError('Incomplete ledger remains unknown, never redispatched')
        rows = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
        physical = [r for r in rows if r.get('request_hash') == event['request_hash']
                    and r.get('stage') == event['stage'] and not r.get('cache_hit')]
        if (not physical or len(physical) > 3
            or [r.get('attempt') for r in physical] != list(range(1, len(physical)+1))
            or any(r.get('context') != intent['context'] for r in physical)):
            raise ValueError('Narrow recovery producer sequence differs')
        tail = physical[-1].get('status')
        if tail in _RETRYABLE_TERMINALS:
            if len(physical) != 3 or any(r.get('status') not in _RETRYABLE_TERMINALS for r in physical):
                raise ValueError('Retryable failure has no complete three-attempt exhaustion')
            admitted.append(candidate)
        elif type(tail) is int and 400 <= tail < 600:
            continue  # Proven nonretryable HTTP remains an explicit missing outcome.
        else:
            raise ValueError('Narrow recovery encountered an unknown terminal status')
    check_jobs(admitted)
    return admitted


_impl['recovery_jobs'] = recovery_jobs
_helpers['recovery_jobs'] = recovery_jobs


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
