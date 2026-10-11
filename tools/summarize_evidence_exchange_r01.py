"""Read-only R01 QA audit with explicit frozen exact-request inheritance."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_evidence_exchange_qa_r01 as runner
from tools import summarize_gemini38_seven_arm_v1 as original
from tools.run_sdk20_modules import save_new, now
from tools.run_sdk20_judge_retest_v2 import read, sha, pointer, bound

SCHEMA = 'evidence-exchange-r01-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*original.CODE, *runner.CODE, 'tools/summarize_evidence_exchange_r01.py')))


def _cached_snapshot_json(snapshot, path):
    """A snapshot owns immutable bytes; do not reread a large proof per slot."""
    key = str(Path(path).resolve())
    cache = getattr(snapshot, '_evidence_exchange_json_cache', None)
    if cache is None:
        cache = {}; snapshot._evidence_exchange_json_cache = cache
    if key not in cache:
        cache[key] = snapshot.read(path)
    return cache[key]


def audit_stage(snapshot, path, expected, api_dir, context, indexed, callback, *, no_dispatch=False):
    path = Path(path)
    raw = snapshot.read(path) if path.exists() else None
    if not raw or raw.get('inheritance_kind') != 'offline_verified_frozen_exact_request':
        return _ordinary_audit_stage(snapshot, path, expected, api_dir, context, indexed, callback, no_dispatch=no_dispatch)
    row = original.existing.bound(snapshot, path, expected)
    manifest_path = path.parents[3] / 'run_manifest.json'
    m = _cached_snapshot_json(snapshot, manifest_path)
    if row.get('reuse_proof') != m.get('baseline_reuse_proof'):
        raise original.InvalidArtifact('Inherited stage proof is not this manifest binding')
    ref = row['reuse_proof']; proof = _cached_snapshot_json(snapshot, ref['path'])
    if snapshot.files[Path(ref['path']).resolve()] != ref['sha256']:
        raise original.InvalidArtifact('Inherited proof bytes differ from manifest pointer')
    stage = expected['stage']; entry = proof['entries'][expected['query_id']]
    plan = m['reuse_plans'][expected['job_id']]
    candidate = plan.get('reader') if stage == 'reader' else plan.get('judge_candidate')
    if (m['phase_kind'] not in ('firstpass', 'technical_recovery') or stage not in ('reader', 'judge') or candidate != entry[stage]
        or row.get('inherited_from') != entry[stage] or row.get('status') != 'complete'
        or row.get('api_events') != [] or row.get('model_calls') != 0 or row.get('error') is not None
        or row.get('raw_invalid_label') is not None or path.with_suffix('.intent.json').exists()
        or any(key[2] == original.old.fingerprint(context) and rows for key, rows in indexed.items())):
        raise original.InvalidArtifact('Inherited stage was dispatched, changed or is not exact baseline reuse')
    if stage == 'judge' and (row.get('original_reader') != entry['reader'] or
                             row.get('original_judge_reader_sha256') != entry['reader']['sha256']):
        raise original.InvalidArtifact('Original Judge Reader provenance differs')
    text = entry['reader_result']['prediction']['answer'] if stage == 'reader' else entry['judge_result']['raw']
    capture = runner.SignatureCapture(text)
    generated = callback(capture)
    if capture.signatures != [entry['signatures'][stage]] or generated != entry[stage + '_result'] or row['result'] != generated:
        raise original.InvalidArtifact('Inherited callback/request/result differs from original replay')
    done = getattr(snapshot, '_evidence_exchange_proof_bindings', set())
    if ref['sha256'] not in done:
        for filename, hashed in proof['input_sha256'].items():
            # manifest() checked each byte binding; record them without copying
            # old physical ledger rows into the new usage denominator.
            snapshot.files.setdefault(Path(filename).resolve(), hashed)
        done.add(ref['sha256']); snapshot._evidence_exchange_proof_bindings = done
    return row


_ordinary_audit_stage = original._clone(original.old.audit_stage, {
    'existing': original.SimpleNamespace(bound=original.existing.bound, audit_stage=runner._new_audit_stage)})
_scope = {**vars(original), 'runner': runner, 'ARMS': ARMS, 'SCHEMA': SCHEMA, 'CODE': CODE,
          'ledger': original._clone(original.old.ledger, {'runner': runner, 'evaluator': original.evaluator}),
          'audit_stage': audit_stage}
for _name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    _scope[_name] = original._clone(getattr(original, _name), _scope)
for _name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    globals()[_name] = _scope[_name]


def stopping_result(rows):
    counts = {arm: sum(r['scores']['judge_correct'] == 1 for r in rows if r['method'] == arm) for arm in ARMS}
    missing = {arm: sum(r['scores']['judge_correct'] is None for r in rows if r['method'] == arm) for arm in ARMS}
    baseline = 1153
    eligible = counts['baseline'] == baseline and missing['baseline'] == 0 and sum(r['method'] == 'baseline' for r in rows) == 1540
    return {'denominator': 1540, 'baseline_correct': baseline, 'audited_baseline_correct': counts['baseline'],
        'threshold_eligible': eligible, 'correct': counts, 'missing': missing,
        'minimum_strict_gt_3pp': 1200,
        'strict_gt_3pp_fixed_denominator': {arm: eligible and (counts[arm] - baseline) * 100 > 3 * 1540 for arm in ARMS[1:]},
        'missing_scores_retained_null_zero_imputation_only_for_fixed_denominator': True,
        'independent_component_count': 1, 'cheap_and_jev_share_selector_component': True,
        'next_round_requires_actual_three_agent_consensus': True}


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    initial = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(initial['source_manifest']['path'])
    if len(source['jobs']) != 4620 or Counter(j['method'] for j in source['jobs']) != {a: 1540 for a in ARMS}:
        raise original.InvalidArtifact('Full R01 three-arm fixed cohort required')
    first = audit_phase(firstpass, source)
    second = audit_phase(recovery, source, parent=first) if recovery else None
    if first['manifest']['phase_kind'] != 'firstpass':
        raise original.InvalidArtifact('Main phase must be firstpass')
    if second and (second['manifest']['parent']['manifest'] != pointer(Path(firstpass) / 'run_manifest.json')
                   or second['manifest']['parent']['outcome'] != pointer(Path(firstpass) / 'run_outcome.json')):
        raise original.InvalidArtifact('Recovery belongs to another firstpass')
    views = {'accepted_firstpass': record_rows(source['jobs'], first),
             'final_with_uniform_technical_recovery': record_rows(source['jobs'], first, second, final=True)}
    phases = [first] + ([second] if second else [])
    errors = [{'phase': p['run'], **e} for p in phases for e in p['errors']]
    logs = [r for p in phases for r in p['ledger']]
    runner.source_validate(initial['source_manifest']['path'])
    for p in phases:
        if any(not Path(path).is_file() or sha(path) != hashed for path, hashed in p['input_sha256'].items()):
            raise original.InvalidArtifact('Audited QA inputs changed before summary')
    lineage = {j['job_id']: {'lineage': j.get('lineage'), 'selection': j['selection'],
                  'reuse_plan': initial['reuse_plans'].get(j['job_id'])} for j in source['jobs']}
    return {'schema': SCHEMA, 'created_utc': now(), 'declared_arms': list(ARMS), 'planned_per_arm': 1540,
       'planned_total': 4620, 'source_manifest': initial['source_manifest'],
       'evaluation_protocol': original.evaluator.PROTOCOL, 'official_metric_provenance': source['official_metric_provenance'],
       'views': {name: aggregate(rows, bootstrap_samples=bootstrap_samples) for name, rows in views.items()},
       'slots_by_view': views, 'stopping_by_view': {name: stopping_result(rows) for name, rows in views.items()},
       'audit_status': 'no_invalid_artifacts' if not errors else 'invalid_artifacts_retained_as_missing',
       'audit_errors': errors, 'selector_lineage': lineage, 'selector_manifest': source['selector_manifest'],
       'selection_counts': source.get('selection_counts'), 'baseline_reuse_proof': initial['baseline_reuse_proof'],
       'inherited_usage': runner.proof_for(initial)['inherited_physical_usage'],
       'inheritance_usage_semantics': 'historical producer ledger cost; not new dispatch and not full cold-start marginal cost',
       'qa_physical_usage': original.measured_usage(logs, missing_ledgers=[x for p in phases for x in p['missing_ledgers']]),
       'qa_usage_by_stage': {stage: original.measured_usage([r for r in logs if r['context']['component'] == stage]) for stage in ('reader', 'judge')},
       'qa_usage_by_producer_method': {arm: original.measured_usage([r for r in logs if r['context']['method'] == arm]) for arm in ARMS},
       'phase_counts': {p['run']: p['outcome']['counts'] for p in phases},
       'phase_intent_inventory': {p['run']: p['intent_inventory'] for p in phases},
       'phase_sources': [{'manifest': pointer(Path(p['run']) / 'run_manifest.json'), 'outcome': pointer(Path(p['run']) / 'run_outcome.json'), 'ledgers': p['ledger_refs']} for p in phases],
       'input_sha256': {k: v for p in phases for k, v in p['input_sha256'].items()},
       'summary_code_sha256': {p: sha(ROOT / p) for p in CODE}, 'model_calls_in_summary': 0, 'native_clients': 0,
       'limitations': ['Fixed 20 facts; token-matched Reader sensitivity is deferred. Bytes/characters are not measured tokens.',
          'Predicate probabilities and thresholds are uncalibrated pilot settings, not semantic truth.',
          'Inherited exact requests are correlated consumers with zero new calls, not independent repetitions.',
          'Selector technical failure/unknown remains in lineage even when unchanged facts legally reuse QA.',
          'Judge flips are model labels, not independently established human correctness.',
          'Summary does not authorize another round; all three reviewers must discuss and agree.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstpass', type=Path, required=True); parser.add_argument('--recovery', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1: raise ValueError('Positive bootstrap count required')
    if not args.readiness and args.output is None: parser.error('--output required')
    with original.historical_guard.historical.offline():
        result = readiness(args.firstpass, args.recovery) if args.readiness else summarize(args.firstpass, args.recovery, bootstrap_samples=args.bootstrap_samples)
    if args.output: save_new(args.output, result)
    import json
    print(json.dumps(result if args.readiness else {'output': str(args.output), 'audit_status': result['audit_status'], 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
