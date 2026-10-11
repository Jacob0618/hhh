"""Offline full1540 extension audit, retaining original400 and new1140 lineage.

Reuse the frozen raw Reader/Judge replay and aggregation. This module never
prepares jobs, opens native banks, retrieves credentials, or calls a model.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_regroup_refinement_qa_v1 as old
from tools import summarize_sdk20_plain_qa_v1 as sdk
from tools import compare_regroup_refinement400_v1 as comparison
from tools import freeze_sdk20_full1540_v1 as locktool
from tools.run_sdk20_modules import save_new, now
from tools.summarize_sdk20_modules import InvalidArtifact
from jev_mem.local_bm25_context_v2 import forbid_network

ARMS = old.ARMS
N = 1540
EXTENSION_N = 1140
FINAL = 'final_with_uniform_technical_recovery'
VIEWS = ('strict_firstpass', 'compatible_firstpass', 'final')
METRICS = sdk.METRICS
EXTENSION_SCHEMA = 'regroup-refinement-plain-extension1140-v1'
EXTENSION_MODULE = 'tools.run_regroup_refinement_extension1140_v1'
PINS = {
    'source_local_context': (
        'research/heartbeat_checks/20260930T1253/qa_context_final_summary_v1.json',
        'acfbd8bb106541e8890025528b9b1d9bd572b6216bf776a5921cf533e13f9196'),
    'jev_refined': (
        'research/heartbeat_checks/20260930T1253/qa_jev_final_summary_v1.json',
        'eb42a8d95f47fe8af47c41cdb6f74032927d41a8e325bbeadce1af06f4d93fed'),
}
CODE = ('tools/summarize_regroup_refinement_full1540_v1.py', *old.CODE,
        'tools/compare_regroup_refinement400_v1.py', 'tools/freeze_sdk20_full1540_v1.py')


def checked(ref):
    actual = old.plain.pointer(ref['path'])
    if any(ref.get(k) != actual[k] for k in ('path', 'sha256')):
        raise InvalidArtifact('Bound summary/phase source drift')
    return Path(actual['path'])


def canonical(lock):
    rows = lock['dataset']['questions']
    expected = {q['query_id']: q for q in rows}
    if len(rows) != N or len(expected) != N or {q['category'] for q in rows} != {1, 2, 3, 4}:
        raise InvalidArtifact('Frozen full1540 population differs')
    return expected


def validate_slots(rows, arm, expected):
    if (len(rows) != len(expected) or len({r['query_id'] for r in rows}) != len(expected)
            or {r['query_id'] for r in rows} != set(expected)):
        raise InvalidArtifact('All planned distinct question IDs, including failures, are required')
    for row in rows:
        q = expected[row['query_id']]
        if (row['method'] != arm or row['conversation'] != q['conversation']
                or row['category'] != q['category'] or set(row['scores']) != set(METRICS)):
            raise InvalidArtifact('Question identity or metric identity differs')
        if row['scores']['official_locomo_f1'] is None and row['scores']['judge_correct'] is not None:
            raise InvalidArtifact('Judge cannot be valid without a valid Reader')


def validate_jobs(report, expected, arm):
    source = old.plain.verify(checked(report['source_manifest']))
    jobs = source['jobs']
    if len(jobs) != len(expected) or {j['query_id'] for j in jobs} != set(expected):
        raise InvalidArtifact('Question source differs from the locked cohort')
    for j in jobs:
        q = expected[j['query_id']]
        if j['method'] != arm or any(j[k] != q[k] for k in
                ('query_id', 'conversation', 'category', 'question', 'reference', 'reference_date')):
            raise InvalidArtifact('Question/reference/date or arm changed')


def aggregate(rows, arm, denominator=N):
    fn = old.scoped(old.aggregate, ARMS=('baseline_sdk20', 'fact_types_sdk20',
                    'dated_unmerged385', 'dated_merged', *ARMS), DENOMINATOR=denominator)
    return fn(rows, arm)['arms'][arm]


def extension_driver(run):
    m = old.plain.read(Path(run) / 'run_manifest.json')
    if m.get('schema') != EXTENSION_SCHEMA:
        raise InvalidArtifact('Only the frozen complementary1140 driver is supported')
    return importlib.import_module(EXTENSION_MODULE)


def summarize_extension(firstpass, recovery=None):
    """Privately bind the new namespace and denominator; raw replay stays old."""
    lock = locktool.verify_lock()
    driver = extension_driver(firstpass)
    arm = old.plain.read(Path(firstpass) / 'run_manifest.json')['arm']
    if arm not in ARMS:
        raise InvalidArtifact('Unrecognized refinement arm')
    replay = old.scoped(old.audit_phase, driver_for=extension_driver)
    aggregation = old.scoped(old.aggregate, DENOMINATOR=EXTENSION_N)
    fn = old.scoped(old.summarize, DENOMINATOR=EXTENSION_N, driver_for=extension_driver,
                    audit_phase=replay, aggregate=aggregation)
    result = fn(firstpass, recovery)
    expected = canonical(lock)
    # The extension's cohort function validates the fixed400 complement without
    # looking at scores, answers, or historical call success.
    complement, _ = driver.sources.cohort()
    ids = {q['id'] for qs in complement.values() for q in qs}
    if len(ids) != EXTENSION_N or not ids <= set(expected):
        raise InvalidArtifact('Extension is not the fixed full1540-minus400 cohort')
    subset = {qid: expected[qid] for qid in ids}
    validate_jobs(result, subset, arm)
    for rows in result['slots_by_view'].values():
        validate_slots(rows, arm, subset)
    result['schema'] = 'regroup-refinement-extension1140-qa-summary-v1'
    result['full1540_lock'] = old.plain.pointer(locktool.LOCK)
    result['summary_code_sha256'] = {p: old.plain.sha(ROOT / p) for p in CODE}
    result['limitations'] = [s.replace('fixed400', 'fixed1140 complementary subset') for s in result['limitations']]
    result['limitations'].append('This report alone is not a full1540 comparison; original400 must be re-audited and joined without overlap.')
    return result


def phase_paths(report):
    phases = report['phase_sources']
    if len(phases) not in (1, 2):
        raise InvalidArtifact('At most one bounded technical recovery is permitted')
    paths = []
    for p in phases:
        manifest = checked(p['manifest'])
        if checked(p['outcome']).parent != manifest.parent:
            raise InvalidArtifact('Phase manifest and terminal outcome belong to different runs')
        for ledger in p['ledgers']:
            checked(ledger)
        paths.append(manifest.parent)
    return paths[0], paths[1] if len(paths) == 2 else None


def compare_replayed(stored, fresh):
    for key in ('slots_by_view', 'audit_status', 'audit_errors', 'phase_sources', 'source_manifest'):
        if stored[key] != fresh[key]:
            raise InvalidArtifact('Stored report does not reproduce original raw replay: ' + key)


def replay_original400(arm):
    path, expected_sha = PINS[arm]
    path = ROOT / path
    if old.plain.sha(path) != expected_sha:
        raise InvalidArtifact('The original400 summary was changed')
    stored = old.plain.verify(path)
    first, recovery = phase_paths(stored)
    fresh = old.summarize(first, recovery)
    compare_replayed(stored, fresh)
    if fresh['planned_per_arm'] != 400 or fresh['arm'] != arm or not fresh['recovery_supplied']:
        raise InvalidArtifact('Expected the final original400 with its one completed recovery')
    return fresh, old.plain.pointer(path)


def join_views(first, extension, arm, expected):
    """Require exact disjoint coverage; no averaging of400 and1140 averages."""
    if set(first['slots_by_view']) != set(VIEWS) or set(extension['slots_by_view']) != set(VIEWS):
        raise InvalidArtifact('Strict, compatible, and final views must remain separate')
    output = {}
    for view in VIEWS:
        a, b = first['slots_by_view'][view], extension['slots_by_view'][view]
        ai, bi = {r['query_id'] for r in a}, {r['query_id'] for r in b}
        if len(a) != 400 or len(b) != EXTENSION_N or ai & bi or ai | bi != set(expected):
            raise InvalidArtifact('Original400 and complementary1140 overlap or do not cover full1540')
        joined = {r['query_id']: r for r in [*a, *b]}
        rows = [joined[qid] for qid in expected]
        validate_slots(rows, arm, expected)
        output[view] = rows
    return output


def combine_arm(extension_path, arm, lock):
    stored = old.plain.verify(extension_path)
    if (stored.get('schema') != 'regroup-refinement-extension1140-qa-summary-v1'
            or stored.get('arm') != arm or stored.get('planned_per_arm') != EXTENSION_N
            or stored.get('full1540_lock') != old.plain.pointer(locktool.LOCK)):
        raise InvalidArtifact('Expected a locked complementary1140 summary for this arm')
    firstpass, recovery = phase_paths(stored)
    extension = summarize_extension(firstpass, recovery)
    compare_replayed(stored, extension)
    first, original_ref = replay_original400(arm)
    expected = canonical(lock)
    first_ids = {r['query_id'] for r in first['slots_by_view']['final']}
    validate_jobs(first, {qid: expected[qid] for qid in first_ids}, arm)
    rows = join_views(first, extension, arm, expected)
    # Compare scientific protocol bindings, excluding namespaces/row population.
    a = old.plain.read(first['phase_sources'][0]['manifest']['path'])
    b = old.plain.read(extension['phase_sources'][0]['manifest']['path'])
    for key in ('gate', 'plain_text_config', 'evaluation_protocol', 'dataset_sha256',
                'official_metric_provenance', 'reference_dates', 'sdk_version', 'sdk_sha256', 'runtime_sha256'):
        if a[key] != b[key]:
            raise InvalidArtifact('Extension changed original400 protocol/source binding: ' + key)
    if not extension['recovery_supplied']:
        driver = extension_driver(firstpass)
        m, out = driver.terminal(firstpass)
        if driver.recovery_jobs(m, out):
            raise InvalidArtifact('Uniform technical recovery is still pending; no final1540 claim')
    return {'views': rows, 'aggregates': {v: aggregate(r, arm) for v, r in rows.items()},
            'sources': {'original400': original_ref, 'complementary1140': old.plain.pointer(extension_path)},
            'audit_errors': first['audit_errors'] + extension['audit_errors'],
            'usage': {'inherited400': first['current_qa_plus_retrieval_usage'],
                      'new1140': extension['current_qa_plus_retrieval_usage'],
                      'note': 'Scopes are reported separately; inherited computation is not newly billed or free cold start.'}}


def replay_baseline(lock):
    stored = old.plain.verify(checked(lock['baseline']['summary']))
    firstpass, recovery = phase_paths(stored)
    fresh = sdk.summarize(firstpass, recovery)
    compare_replayed(stored, fresh)
    expected = canonical(lock)
    output = {}
    for arm in ('baseline_sdk20', 'fact_types_sdk20'):
        rows = [r for r in fresh['slots_by_view'][FINAL] if r['method'] == arm]
        validate_slots(rows, arm, expected)
        output[arm] = {'rows': rows, 'aggregate': aggregate(rows, arm)}
    base = output['baseline_sdk20']['aggregate']['metrics']
    for name, percent in (('official_locomo_f1', lock['baseline']['official_f1_percent']),
                          ('judge_correct', lock['baseline']['judge_percent'])):
        if base[name]['valid_denominator'] != N or abs(100 * base[name]['complete_only_mean'] - percent) > 1e-10:
            raise InvalidArtifact('User-selected54.610/66.039 baseline drift')
    return output, {'summary': lock['baseline']['summary'], 'view': FINAL,
                    'raw_replayed': True, 'usage': fresh['total_current_usage']}


def pair(left, right, *, bootstrap_samples=2000):
    return old.scoped(comparison.pair, N=N)(left, right, bootstrap_samples=bootstrap_samples)


def combine(context_extension, jev_extension, *, dated_summary=None):
    lock = locktool.verify_lock()
    refs, base_source = replay_baseline(lock)
    arms = {arm: combine_arm(path, arm, lock) for arm, path in
            zip(ARMS, (context_extension, jev_extension))}
    rows = {a: v['rows'] for a, v in refs.items()}
    rows.update({a: v['views']['final'] for a, v in arms.items()})
    if dated_summary is not None:
        # The dated adapter owns its distinct original raw/compatibility and
        # pre-existing c2_q74 retry lineage; no new generic recovery is inferred.
        module = importlib.import_module('tools.run_dated_judge_uniform_recovery_v1')
        stored_dated = old.plain.verify(dated_summary)
        dated = module.summarize(stored_dated['run_dir'])
        if stored_dated['slots_by_view'] != dated['slots_by_view']:
            raise InvalidArtifact('Dated final summary does not reproduce original raw replay')
        for arm in ('dated_unmerged385', 'dated_merged'):
            value = [r for r in dated['slots_by_view'][FINAL] if r['method'] == arm]
            validate_slots(value, arm, canonical(lock))
            rows[arm] = value
    else:
        dated = None
    table = {arm: aggregate(value, arm) for arm, value in rows.items()}
    pairs = {arm + '_minus_baseline_sdk20': pair(rows['baseline_sdk20'], value)
             for arm, value in rows.items() if arm != 'baseline_sdk20'}
    if dated:
        pairs['dated_merged_minus_dated_unmerged385'] = pair(rows['dated_unmerged385'], rows['dated_merged'])
    return {'schema': 'regroup-refinement-full1540-comparison-v1', 'created_utc': now(),
            'full1540_lock': old.plain.pointer(locktool.LOCK), 'planned_per_method': N,
            'main_view': FINAL, 'baseline': 'baseline_sdk20', 'table': table, 'paired': pairs,
            'slots': [r for rs in rows.values() for r in rs],
            'historical_views_preserved': {a: {'strict_firstpass': v['aggregates']['strict_firstpass'],
                                           'compatible_firstpass': v['aggregates']['compatible_firstpass']}
                                          for a, v in arms.items()},
            'sources': {'baseline_and_types': base_source, 'extensions': {a: v['sources'] for a, v in arms.items()},
                        'dated_final': old.plain.pointer(dated_summary) if dated_summary else None},
            'comparison_complete': dated is not None,
            'missing_comparison_reason': None if dated else 'Dated217 uniform technical recovery and replay not supplied; table is not the complete six-method comparison.',
            'audit_errors': {a: v['audit_errors'] for a, v in arms.items()},
            'usage': {a: v['usage'] for a, v in arms.items()},
            'summary_code_sha256': {p: old.plain.sha(ROOT / p) for p in CODE},
            'model_calls_in_summary': 0, 'native_clients': 0,
            'limitations': ['All means use independently reported valid denominators and planned1540; missing-as-zero is only a sensitivity.',
                           'Original400 and complementary1140 use the same frozen cohort/protocol and bank, but were generated at different times.',
                           'Ten conversations are previously exposed, including development examples; this is not a blind test.',
                           'Paired cluster intervals are exploratory and do not establish causality or practical importance.',
                           'Only declared writer/type modules differ; twenty facts do not impose identical token counts.',
                           'Historical recovery and new1140 costs remain separate; unknown usage is not zero.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    extension = sub.add_parser('extension')
    extension.add_argument('--firstpass', type=Path, required=True)
    extension.add_argument('--recovery', type=Path)
    extension.add_argument('--output', type=Path, required=True)
    combined = sub.add_parser('combine')
    combined.add_argument('--context-extension', type=Path, required=True)
    combined.add_argument('--jev-extension', type=Path, required=True)
    combined.add_argument('--dated-summary', type=Path)
    combined.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh summary output required; never overwrite immutable reports')
    with forbid_network():
        value = summarize_extension(args.firstpass, args.recovery) if args.action == 'extension' else combine(
            args.context_extension, args.jev_extension, dated_summary=args.dated_summary)
    save_new(args.output, value)
    print(json.dumps({'output': str(args.output), 'schema': value['schema'], 'model_calls': 0}))


if __name__ == '__main__':
    main()
