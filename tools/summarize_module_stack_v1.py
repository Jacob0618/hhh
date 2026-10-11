"""Offline nine-arm module-stack audit using the original raw QA replayer."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_evidence_exchange_r01 as prior
from tools import run_module_stack_qa_v1 as runner
from tools.run_sdk20_modules import save_new, now
from tools.run_sdk20_judge_retest_v2 import read, pointer, sha, bound

SCHEMA = 'module-stack-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_module_stack_v1.py')))
original = prior.original
_cached_snapshot_json = prior._cached_snapshot_json
_ordinary_audit_stage = prior._ordinary_audit_stage
_helpers = {**vars(prior), 'runner': runner, 'SCHEMA': SCHEMA, 'ARMS': ARMS, 'CODE': CODE}
audit_stage = runner.clone(prior.audit_stage, _helpers)
_helpers['audit_stage'] = audit_stage
_scope = {**prior._scope, 'runner': runner, 'ARMS': ARMS, 'SCHEMA': SCHEMA, 'CODE': CODE,
    'ledger': original._clone(original.old.ledger, {'runner': runner, 'evaluator': original.evaluator}),
    'audit_stage': audit_stage}
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    _scope[name] = original._clone(getattr(original, name), _scope)
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    globals()[name] = _scope[name]

CONTRASTS = (
    ('pool_old_bm25', 'pool_old_jev'),
    ('pool_new_bm25', 'pool_new_jev'),
    ('pool_old_bm25', 'pool_new_bm25'),
    ('pool_old_jev', 'pool_new_jev'),
    ('pool_new_jev', 'stack_source'),
    ('stack_source', 'stack_support'),
    ('stack_source', 'stack_closure'),
    ('stack_source', 'stack_support_closure'),
    ('stack_support', 'stack_support_closure'),
    ('stack_closure', 'stack_support_closure'))


def stack_contrasts(rows, *, bootstrap_samples=2000):
    """Predeclared component contrasts retain the same paired missing rules."""
    def valid(arm, name):
        return {r['query_id']: {'query_id': r['query_id'], 'conversation': r['conversation'], name: r['scores'][name]}
                for r in rows if r['method'] == arm and r['scores'][name] is not None}
    return {right + '_minus_' + left: {'left': left, 'right': right,
        'metrics': {name: original.existing.paired_metric(name, valid(left, name), valid(right, name),
            samples=bootstrap_samples, denominator=runner.PER_ARM) for name in original.METRICS}}
        for left, right in CONTRASTS}


def stopping_result(rows):
    counts = {a: sum(r['scores']['judge_correct'] == 1 for r in rows if r['method'] == a) for a in ARMS}
    missing = {a: sum(r['scores']['judge_correct'] is None for r in rows if r['method'] == a) for a in ARMS}
    eligible = counts['baseline'] == 1153 and missing['baseline'] == 0 and all(
        sum(r['method'] == arm for r in rows) == runner.PER_ARM for arm in ARMS)
    return {'denominator': runner.PER_ARM, 'baseline_correct': 1153,
        'audited_baseline_correct': counts['baseline'], 'threshold_eligible': eligible,
        'correct': counts, 'missing': missing, 'minimum_strict_gt_3pp': 1200,
        'strict_gt_3pp_fixed_denominator': {a: eligible and (counts[a] - 1153) * 100 > 3 * runner.PER_ARM for a in ARMS[1:]},
        'missing_scores_retained_null_zero_imputation_only_for_fixed_denominator': True,
        'component_success_requires_planned_contrasts_not_arm_count': True,
        'no_additional_round_authorized_by_summary': True}


def augmentation_audit(source, protocol):
    """Inspect stack prefixes; sealed validator owns exact source/offset replay."""
    jobs = source['jobs']
    runner.check_full_cohort(jobs)
    by_key = {(j['method'], j['query_id']): j for j in jobs}
    from jev_mem.source_append_r05 import WRAPPER
    left, right = WRAPPER.split('{rendered_packet}')
    if '\n' not in left or '\\n' in left:
        raise ValueError('Actual LF attachment wrapper required')
    distributions = {arm: Counter() for arm in ARMS}
    rows = []
    for job in jobs:
        origin_job = by_key[('pool_new_jev', job['query_id'])] if job['method'] in ARMS[5:] else job
        if job['selected_ids'] != origin_job['selected_ids']:
            raise ValueError('Stack must preserve R08 JEV ordered20 IDs')
        changed = []
        for slot, (fact, origin) in enumerate(zip(job['selected_facts'], origin_job['selected_facts'])):
            if (fact['id'] != origin['id'] or fact['observation_date'] != origin['observation_date']
                    or not fact['text'].startswith(origin['text'])):
                raise ValueError('Stack must preserve R08 JEV ID/date/text prefix')
            suffix = fact['text'][len(origin['text']):]
            if not suffix:
                continue
            if not suffix.startswith(left) or not suffix.endswith(right):
                raise ValueError('Complete attachment wrapper required')
            rendered = suffix[len(left):len(suffix) - len(right)]
            packet = json.loads(rendered)
            if (set(packet) != {'session_id', 'source_observation_date', 'turns'}
                    or json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(',', ':')) != rendered
                    or len(suffix.encode('utf-8')) > 4096):
                raise ValueError('Canonical bounded packet required')
            # Closure can include up to five source turns. Old R05's three-turn
            # window is not a valid upper bound for this new sealed protocol.
            if not isinstance(packet['turns'], list) or not 1 <= len(packet['turns']) <= 5:
                raise ValueError('At most five source turns required')
            for turn in packet['turns']:
                if (set(turn) != {'source_turn_id', 'speaker_text', 'machine_captions'}
                        or not isinstance(turn['speaker_text'], str)
                        or not isinstance(turn['machine_captions'], list)
                        or not all(isinstance(c, str) for c in turn['machine_captions'])):
                    raise ValueError('Restricted speaker/caption projection required')
            changed.append({'slot': slot, 'added_utf8_bytes': len(suffix.encode('utf-8')),
                            'source_turn_count': len(packet['turns'])})
        total = sum(c['added_utf8_bytes'] for c in changed)
        if len(changed) > 2 or total > 8192:
            raise ValueError('At most two source slots and 8192 total suffix bytes required')
        distributions[job['method']][len(changed)] += 1
        rows.append({'method': job['method'], 'query_id': job['query_id'],
            'attached_slots': [c['slot'] for c in changed], 'added_utf8_bytes': total,
            'attachments': changed})
    return {'stack_R08_JEV20_order_ids_dates_prefixes_preserved': True,
        'first_five_unappended_inputs_and_offsets_checked_by_sealed_validator': True,
        'slot_count_is_not_equal_information_or_tokens': True,
        'attachment_count_distribution': {a: dict(sorted(v.items())) for a, v in distributions.items()},
        'added_utf8_bytes_by_arm': {a: sum(r['added_utf8_bytes'] for r in rows if r['method'] == a) for a in ARMS},
        'per_job': rows, 'measured_tokens_from_bytes': None}


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    initial = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(initial['source_manifest']['path'])
    runner.check_full_cohort(source['jobs'])
    protocol = bound(source['module_stack_protocol'])
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
    aggregates = {name: aggregate(rows, bootstrap_samples=bootstrap_samples) for name, rows in views.items()}
    for name, rows in views.items():
        aggregates[name]['planned_stack_contrasts'] = stack_contrasts(rows, bootstrap_samples=bootstrap_samples)
    return {'schema': SCHEMA, 'created_utc': now(), 'declared_arms': list(ARMS),
       'planned_per_arm': runner.PER_ARM, 'planned_total': runner.TOTAL,
       'source_manifest': initial['source_manifest'], 'module_stack_protocol': source['module_stack_protocol'],
       'evaluation_protocol': original.evaluator.PROTOCOL, 'official_metric_provenance': source['official_metric_provenance'],
       'views': aggregates, 'slots_by_view': views,
       'stopping_by_view': {name: stopping_result(rows) for name, rows in views.items()},
       'audit_status': 'no_invalid_artifacts' if not errors else 'invalid_artifacts_retained_as_missing',
       'audit_errors': errors, 'selector_lineage': lineage, 'selector_manifest': source['selector_manifest'],
       'selection_counts': source.get('selection_counts'), 'baseline_reuse_proof': initial['baseline_reuse_proof'],
       'prior_raw_proof_lineage': runner.proof_for(initial).get('reuse_revalidation'),
       'source_append_contract_audit': augmentation_audit(source, protocol),
       'inherited_usage': runner.proof_for(initial)['inherited_physical_usage'],
       'inheritance_usage_semantics': 'historical N0 producer ledger cost; not new dispatch and not full cold-start marginal cost',
       'qa_physical_usage': original.measured_usage(logs, missing_ledgers=[x for p in phases for x in p['missing_ledgers']]),
       'qa_usage_by_stage': {stage: original.measured_usage([r for r in logs if r['context']['component'] == stage]) for stage in ('reader', 'judge')},
       'qa_usage_by_producer_method': {arm: original.measured_usage([r for r in logs if r['context']['method'] == arm]) for arm in ARMS},
       'phase_counts': {p['run']: p['outcome']['counts'] for p in phases},
       'phase_intent_inventory': {p['run']: p['intent_inventory'] for p in phases},
       'phase_sources': [{'manifest': pointer(Path(p['run']) / 'run_manifest.json'), 'outcome': pointer(Path(p['run']) / 'run_outcome.json'), 'ledgers': p['ledger_refs']} for p in phases],
       'input_sha256': {k: v for p in phases for k, v in p['input_sha256'].items()},
       'summary_code_sha256': {p: sha(ROOT / p) for p in CODE}, 'model_calls_in_summary': 0, 'native_clients': 0,
       'limitations': [
          'All arms retain 20 slots; source attachment length and realized information differ. Bytes are not measured tokens.',
          'Missing outcomes remain null; complete-only denominators and fixed1540 zero-imputed sensitivity are reported separately.',
          'Inherited exact N0 requests and same-run exact cache consumers are correlated, not independent repetitions.',
          'Archived R03/R08 QA scores are diagnostic only and never inherited production or request caches.',
          'Support and closure rules are retrieval interventions, not independently verified semantic support or causal truth.',
          'Source observation dates are not event dates; a copied source does not establish a bridge, plan, or first occurrence.',
          'Judge labels are model observations, not independently established human correctness.',
          'Direction was informed by prior diagnostics; this is a fixed-cohort module-stacking comparison.',
          'Historical producer usage is separate from new physical usage and is not a complete deployment invoice.',
          'Arm count does not establish independent successful components; use the planned component contrasts.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstpass', type=Path, required=True)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1: raise ValueError('Positive bootstrap count required')
    if not args.readiness and args.output is None: parser.error('--output required')
    with original.historical_guard.historical.offline():
        result = (readiness(args.firstpass, args.recovery) if args.readiness else
                  summarize(args.firstpass, args.recovery, bootstrap_samples=args.bootstrap_samples))
    if args.output: save_new(args.output, result)
    print(json.dumps(result if args.readiness else {'output': str(args.output),
        'audit_status': result['audit_status'], 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
