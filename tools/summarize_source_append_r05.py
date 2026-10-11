"""Offline R05 raw audit: preserved N0 prefixes plus bounded source suffixes."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_evidence_exchange_r01 as prior
from tools import run_source_append_qa_r05 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read, bound

SCHEMA = 'source-append-r05-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_source_append_r05.py')))
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


def stopping_result(rows):
    result = prior.stopping_result(rows)
    result.pop('independent_component_count', None)
    result.pop('cheap_and_jev_share_selector_component', None)
    result.update(component_family='bounded_source_append_independent_of_r03_r04',
        candidate_component_family_count=1,
        successful_jev_components_counted_here=int(result['strict_gt_3pp_fixed_denominator']['jev']),
        cheap_and_jev_share_source_append_component=True,
        root_stop_count_requires_actual_three_reviewer_result_consensus=True)
    return result


_helpers.update(audit_phase=audit_phase, record_rows=record_rows, aggregate=aggregate,
                readiness=readiness, stopping_result=stopping_result)
_summarize = runner.clone(prior.summarize, _helpers)


def augmentation_audit(source, protocol):
    """Independent structural check; exact source offsets replay in validator.

    This check never claims prefix or literal substring proves semantic support.
    Canonical source packets contain only the protocol's restricted projection.
    """
    jobs = source['jobs']
    baseline = {j['query_id']: j for j in jobs if j['method'] == 'baseline'}
    if len(baseline) != 1540 or len(jobs) != 4620:
        raise ValueError('Fixed R05 full cohort required')
    left, right = protocol['reader_representation']['wrapper'].split('{rendered_packet}')
    if '\n' not in left or '\\n' in left:
        raise ValueError('Actual LF wrapper required')
    distributions = {arm: Counter() for arm in ARMS}
    rows = []
    for job in jobs:
        old = baseline[job['query_id']]
        if (len(job['selected_facts']) != 20 or job['selected_ids'] != old['selected_ids']
            or len(set(job['selected_ids'])) != 20):
            raise ValueError('Original20 ordered IDs required')
        changed = []
        for slot, (fact, origin) in enumerate(zip(job['selected_facts'], old['selected_facts'])):
            if (fact['id'] != origin['id'] or fact['observation_date'] != origin['observation_date']
                or not fact['text'].startswith(origin['text'])):
                raise ValueError('Original ID/date/text prefix differs')
            suffix = fact['text'][len(origin['text']):]
            if not suffix: continue
            if not suffix.startswith(left) or not suffix.endswith(right):
                raise ValueError('Fixed complete attachment wrapper required')
            rendered = suffix[len(left):len(suffix)-len(right)]
            packet = json.loads(rendered)
            if (set(packet) != {'session_id', 'source_observation_date', 'turns'}
                or json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(',', ':')) != rendered
                or len(suffix.encode('utf-8')) > 4096):
                raise ValueError('Canonical complete bounded packet required')
            if not isinstance(packet['turns'], list) or not 1 <= len(packet['turns']) <= 3:
                raise ValueError('Bounded source window required')
            for turn in packet['turns']:
                if (set(turn) != {'source_turn_id', 'speaker_text', 'machine_captions'}
                    or not isinstance(turn['speaker_text'], str)
                    or not isinstance(turn['machine_captions'], list)
                    or not all(isinstance(c, str) for c in turn['machine_captions'])):
                    raise ValueError('Restricted speaker/caption packet projection required')
            changed.append({'slot': slot, 'added_utf8_bytes': len(suffix.encode('utf-8'))})
        total = sum(c['added_utf8_bytes'] for c in changed)
        if len(changed) > 2 or total > 8192 or (job['method'] == 'baseline' and changed):
            raise ValueError('At most2 source slots, unchanged baseline required')
        distributions[job['method']][len(changed)] += 1
        rows.append({'method': job['method'], 'query_id': job['query_id'],
            'attached_slots': [c['slot'] for c in changed], 'added_utf8_bytes': total})
    job_index = {(j['method'], j['query_id']): j for j in jobs}
    if len(job_index) != 4620:
        raise ValueError('Unique method/query identities required')
    if any(sum(r['method'] == arm for r in rows) != 1540 for arm in ARMS):
        raise ValueError('Each arm1540 required')
    return {'original20_order_ids_dates_prefixes_preserved': True,
        'source_offsets_replayed_by_sealed_validator': True,
        'slot_count_is_not_equal_information_or_tokens': True,
        'attachment_count_distribution': {a: dict(sorted(v.items())) for a, v in distributions.items()},
        'added_utf8_bytes_by_arm': {a: sum(r['added_utf8_bytes'] for r in rows if r['method'] == a) for a in ARMS},
        'cheap_jev_same_derived_reader_facts': sum(
            job_index[('cheap', q)]['selected_facts'] == job_index[('jev', q)]['selected_facts'] for q in baseline),
        'per_job': rows, 'measured_tokens_from_bytes': None}


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    result = _summarize(firstpass, recovery, bootstrap_samples=bootstrap_samples)
    manifest = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(manifest['source_manifest']['path'])
    protocol = bound(source['source_append_protocol'])
    result.update(source_append_protocol=source['source_append_protocol'],
        selection_counts=source.get('selection_counts'),
        source_append_contract_audit=augmentation_audit(source, protocol),
        evidence_contract='Original N0 ordered20 IDs/date/text prefixes retained; at most2 slots carry complete restricted verbatim source packets. Offset replay is not semantic support certification.',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        limitations=[
            'Preserved20 slots and unchanged memory prefixes are not equal information or tokens; attachment bytes are not measured tokens.',
            'Cheap and JEV share candidate scope/projection/caps, but may select different source contents and realized lengths.',
            'Current session input scope is source exposure, not verified fact support. Local windows and lexical misses do not certify source absence.',
            'Caption-only restricted projection excludes image queries seen by original Writer; it is not byte-identical full Writer input.',
            'Source observation date is not event date; copied text does not create bridge/plan/first-time truth.',
            'Pointwise local usefulness is uncalibrated, not whole-set completeness or nondegradation. All-low valid scores still rank.',
            'Inherited exact requests are correlated consumers, not independent repetitions or free cold-start cost.',
            'Inherited producer usage includes historical evaluation. It is not a complete deployment cold invoice.',
            'Judge labels are model observations, not independently established human correctness.',
            'Direction was informed by previous QA diagnoses; locator eligibility and parameter fitting do not consume per-query gold.',
            'This source-append module is separate from R03/R04. Success and stop counts require actual thresholds and three-reviewer result discussion.'])
    return result


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
