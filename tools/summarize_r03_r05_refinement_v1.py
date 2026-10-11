"""Offline refinement summary with exact inherited/raw audits and fixed1540.

No ModelAPI construction or transport. Old outcomes/cost remain separately
bound historical observations; only this source run's ledgers are new cost.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_module_stack_v1 as prior
from tools import run_r03_r05_refinement_qa_v1 as runner
from tools import r03_r05_reuse_proofs_v1 as reuse
from tools.run_sdk20_modules import save_new, now
from tools.run_sdk20_judge_retest_v2 import read, pointer, sha, bound

SCHEMA = 'r03-r05-refinement-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_r03_r05_refinement_v1.py')))
original = prior.original
offline = original.historical_guard.historical.offline
CONTRASTS = (
    ('baseline', 'r03_original'),
    ('r03_original', 'r03_refined'),
    ('r03_original', 'original_source_closure'),
    ('r03_refined', 'refined_source_closure'),
    ('original_source_closure', 'refined_source_closure'),
    ('original_source_closure', 'original_multi_source_closure'),
    ('refined_source_closure', 'refined_multi_source_closure'),
    ('original_multi_source_closure', 'refined_multi_source_closure'))


def _raw_audit_stage(snapshot, path, expected, api_dir, context, indexed, callback, *, no_dispatch=False):
    path = Path(path)
    raw = snapshot.read(path) if path.exists() else None
    if not raw or raw.get('inheritance_kind') != 'offline_verified_frozen_exact_request':
        return prior._ordinary_audit_stage(snapshot, path, expected, api_dir, context, indexed, callback,
                                           no_dispatch=no_dispatch)
    if no_dispatch:
        raise original.InvalidArtifact('Inherited stage lacks a successful prerequisite')
    row = original.existing.bound(snapshot, path, expected)
    manifest_path = path.parents[3] / 'run_manifest.json'
    m = prior._cached_snapshot_json(snapshot, manifest_path)
    ref = m['prior_qa_reuse_proof']
    proof = prior._cached_snapshot_json(snapshot, ref['path'])
    if snapshot.files[Path(ref['path']).resolve()] != ref['sha256'] or row.get('reuse_proof') != ref:
        raise original.InvalidArtifact('Inherited stage proof bytes/binding differ')
    jobs = getattr(snapshot, '_refinement_summary_jobs', {})
    if str(manifest_path) not in jobs:
        jobs[str(manifest_path)] = {j['job_id']: j for j in m['jobs']}
        snapshot._refinement_summary_jobs = jobs
    job = jobs[str(manifest_path)][expected['job_id']]
    stage = expected['stage']
    if stage == 'reader':
        entry = reuse.lookup_reader(proof, job)
        candidate = m['reuse_plans'][job['job_id']]['reader']
    elif stage == 'judge':
        rp = runner.stage_files(manifest_path.parent, job)['reader']
        snapshot.raw(rp)
        entry = reuse.lookup_judge(proof, job, {'path': str(rp.resolve()), 'sha256': snapshot.files[rp.resolve()]})
        candidate = entry['judge'] if entry else None
    else:
        raise original.InvalidArtifact('Only Reader/Judge may inherit QA proof')
    if (entry is None or candidate != entry[stage] or row.get('inherited_from') != entry[stage]
        or row.get('status') != 'complete' or row.get('api_events') != [] or row.get('model_calls') != 0
        or row.get('error') is not None or row.get('raw_invalid_label') is not None
        or path.with_suffix('.intent.json').exists()
        or any(r.get('context') == context for rows in indexed.values() for r in rows)):
        raise original.InvalidArtifact('Inherited stage has changed or has dispatch activity')
    if stage == 'judge' and (row.get('original_reader') != entry['reader']
                            or row.get('original_judge_reader_sha256') != entry['reader']['sha256']):
        raise original.InvalidArtifact('Inherited Judge original Reader provenance differs')
    text = entry['reader_result']['prediction']['answer'] if stage == 'reader' else entry['judge_result']['raw']
    capture = runner.SignatureCapture(text)
    generated = callback(capture)
    if capture.signatures != [entry['signatures'][stage]] or generated != entry[stage + '_result'] or row['result'] != generated:
        raise original.InvalidArtifact('Inherited callback/full request/result differs from original replay')
    done = getattr(snapshot, '_refinement_proof_inputs', set())
    if ref['sha256'] not in done:
        for filename, hashed in proof['input_sha256'].items():
            previous = snapshot.files.setdefault(Path(filename).resolve(), hashed)
            if previous != hashed:
                raise original.InvalidArtifact('Conflicting inherited input binding')
        done.add(ref['sha256']); snapshot._refinement_proof_inputs = done
    return row


# The projection preserves the original replayer and complete relevant producer
# chain. Unlike the old global scans it does not hash every unrelated context.
audit_stage = runner.clone(reuse._projection,
    {**reuse._projection.__globals__, 'prior_summary': SimpleNamespace(audit_stage=_raw_audit_stage)})
_scope = {**prior._scope, 'runner': runner, 'ARMS': ARMS, 'SCHEMA': SCHEMA, 'CODE': CODE,
    'ledger': original._clone(original.old.ledger, {'runner': runner, 'evaluator': runner.evaluator}),
    'audit_stage': audit_stage}
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    _scope[name] = original._clone(getattr(original, name), _scope)
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    globals()[name] = _scope[name]


def component_contrasts(rows, *, bootstrap_samples=2000):
    def valid(arm, metric):
        return {r['query_id']: {'query_id': r['query_id'], 'conversation': r['conversation'], metric: r['scores'][metric]}
                for r in rows if r['method'] == arm and r['scores'][metric] is not None}
    return {right + '_minus_' + left: {'left': left, 'right': right,
        'metrics': {name: original.existing.paired_metric(name, valid(left, name), valid(right, name),
            samples=bootstrap_samples, denominator=runner.PER_ARM) for name in original.METRICS}}
        for left, right in CONTRASTS}


def factorial_interaction(rows, *, bootstrap_samples=2000):
    names = ('original_source_closure', 'refined_source_closure',
             'original_multi_source_closure', 'refined_multi_source_closure')
    indexed = {a: {r['query_id']: r for r in rows if r['method'] == a} for a in names}
    result = {}
    for metric in original.METRICS:
        complete = set.intersection(*(set(q for q, r in indexed[a].items() if r['scores'][metric] is not None) for a in names))
        left, right = {}, {}
        for qid in sorted(complete):
            a, b, c, d = (indexed[name][qid] for name in names)
            if len({r['conversation'] for r in (a, b, c, d)}) != 1:
                raise ValueError('Factorial rows differ in conversation identity')
            delta = d['scores'][metric] - b['scores'][metric] - c['scores'][metric] + a['scores'][metric]
            ident = {'query_id': qid, 'conversation': a['conversation']}
            left[qid] = {**ident, metric: 0.0}; right[qid] = {**ident, metric: delta}
        result[metric] = {'common_valid_four_arm_count': len(complete),
            'planned_denominator': runner.PER_ARM,
            'contrast': original.existing.paired_metric(metric, left, right, samples=bootstrap_samples, denominator=runner.PER_ARM),
            'by_conversation': {str(ci): {'n': sum(r['conversation'] == ci for r in right.values()),
                'sum_interaction': sum(r[metric] for r in right.values() if r['conversation'] == ci)} for ci in range(10)}}
    return {'formula': 'refined_multi - refined_single - original_multi + original_single; closure held fixed',
            'arms': list(names), 'metrics': result,
            'interpretation': 'Signed per-query difference of differences on the four-arm common-valid subset; exploratory same-cohort paired cluster bootstrap, not independent causal replication.'}


def selected_input_audit(source):
    runner.check_full_cohort(source['jobs'])
    by_key = {(j['method'], j['query_id']): j for j in source['jobs']}
    bases = {'original_source_closure': 'r03_original', 'original_multi_source_closure': 'r03_original',
             'refined_source_closure': 'r03_refined', 'refined_multi_source_closure': 'r03_refined'}
    counts = {a: Counter() for a in ARMS}; sizes = {a: 0 for a in ARMS}
    for job in source['jobs']:
        base = by_key[(bases[job['method']], job['query_id'])] if job['method'] in bases else job
        if job['selected_ids'] != base['selected_ids']:
            raise ValueError('Source variant changed declared base20 IDs/order')
        changed = 0
        for fact, origin in zip(job['selected_facts'], base['selected_facts']):
            if (fact['id'] != origin['id'] or fact['observation_date'] != origin['observation_date']
                or not fact['text'].startswith(origin['text'])):
                raise ValueError('Source variant changed original fact/date/text prefix')
            suffix = fact['text'][len(origin['text']):]
            changed += bool(suffix); sizes[job['method']] += len(suffix.encode('utf-8'))
        counts[job['method']][changed] += 1
    return {'all_arms_exact20': True, 'source_variants_preserve_declared_base20': True,
            'source_offsets_packets_caps_replayed_by_sealed_source_validator': True,
            'changed_slots_by_arm': {a: dict(sorted(v.items())) for a, v in counts.items()},
            'added_utf8_bytes_by_arm': sizes, 'bytes_are_measured_tokens': False}


def source_usage(source):
    from tools import prepare_r03_r05_refinement_v1 as prep
    run = Path(source['parent_run']).resolve()
    if run.parent != (ROOT / 'runs').resolve() or not run.name.startswith('r03_r05_refinement'):
        raise ValueError('Current refinement source run required for new cost')
    reader = prep.Reader()
    manifest = reader.bound(source['selector_manifest'])
    if manifest.get('schema') != prep.SCHEMA or Path(source['selector_manifest']['path']).parent != run:
        raise ValueError('Current source cost namespace differs')
    rows, refs, phase_counts = [], [], {}
    for phase in prep.PHASES:
        results = prep.phase_results(reader, run, phase)
        phase_counts[phase] = dict(Counter(r['status'] for r in results.values()))
        for part in ('firstpass', 'recovery_v1'):
            folder = run / 'phases' / phase / part
            if not folder.exists():
                continue
            ledger = folder / 'api_calls.jsonl'
            if not ledger.exists():
                # A fully oversize/non-dispatched phase can have no ledger.
                continue
            data = ledger.read_bytes()
            if data and not data.endswith(b'\n'):
                raise ValueError('Unterminated source ledger row')
            rows.extend(json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip())
            refs.append(pointer(ledger))
    return {'physical_usage': original.measured_usage(rows), 'ledger_sources': refs,
            'phase_terminal_counts': phase_counts, 'historical_source_ledgers_included': False}


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('Positive bootstrap sample count required')
    initial = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(initial['source_manifest']['path'])
    bound(source['r03_r05_protocol'])
    first = audit_phase(firstpass, source)
    second = audit_phase(recovery, source, parent=first) if recovery else None
    if first['manifest']['phase_kind'] != 'firstpass':
        raise original.InvalidArtifact('Main phase must be firstpass')
    if second and (second['manifest']['parent']['manifest'] != pointer(Path(firstpass) / 'run_manifest.json')
                   or second['manifest']['parent']['outcome'] != pointer(Path(firstpass) / 'run_outcome.json')):
        raise original.InvalidArtifact('Recovery belongs to another firstpass')
    phases = [first] + ([second] if second else [])
    errors = [{'phase': p['run'], **e} for p in phases for e in p['errors']]
    if errors:
        raise original.InvalidArtifact('Raw audit failed; no aggregate summary is admissible')
    for phase in phases:
        for inventory in phase['intent_inventory'].values():
            if inventory['unknown_intent_count'] or inventory['ended_success_missing_raw_count']:
                raise original.InvalidArtifact('Unknown intent or successful producer missing raw blocks summary')
    views = {'accepted_firstpass': record_rows(source['jobs'], first),
             'final_with_uniform_technical_recovery': record_rows(source['jobs'], first, second, final=True)}
    result_views = {name: aggregate(rows, bootstrap_samples=bootstrap_samples) for name, rows in views.items()}
    final = views['final_with_uniform_technical_recovery']
    contrasts = component_contrasts(final, bootstrap_samples=bootstrap_samples)
    interaction = factorial_interaction(final, bootstrap_samples=bootstrap_samples)
    selector_cost = source_usage(source)
    runner.source_validate(initial['source_manifest']['path'])
    for phase in phases:
        if any(not Path(path).is_file() or sha(path) != hashed for path, hashed in phase['input_sha256'].items()):
            raise original.InvalidArtifact('Audited QA inputs changed before summary')
    if any(pointer(ref['path']) != ref for ref in selector_cost['ledger_sources']):
        raise original.InvalidArtifact('Current source ledger changed during summary')
    logs = [r for phase in phases for r in phase['ledger']]
    proof = runner.proof_for(initial)
    return {'schema': SCHEMA, 'created_utc': now(), 'declared_arms': list(ARMS),
        'planned_per_arm': runner.PER_ARM, 'planned_total': runner.TOTAL, 'audit_status': 'no_invalid_artifacts',
        'errors': [], 'audit_errors': [], 'rows': final, 'by_method': result_views['final_with_uniform_technical_recovery']['arms'],
        'views': result_views, 'contrasts': contrasts, 'factorial_interaction_same_closure': interaction,
        'source_manifest': initial['source_manifest'], 'r03_r05_protocol': source['r03_r05_protocol'],
        'selector_manifest': source['selector_manifest'], 'prior_qa_reuse': source['prior_qa_reuse'],
        'prior_qa_reuse_proof': initial['prior_qa_reuse_proof'], 'evaluation_protocol': runner.evaluator.PROTOCOL,
        'official_metric_provenance': source['official_metric_provenance'], 'selected_input_audit': selected_input_audit(source),
        'qa_physical_usage': original.measured_usage(logs, missing_ledgers=[p for phase in phases for p in phase['missing_ledgers']]),
        'qa_usage_by_stage': {s: original.measured_usage([r for r in logs if r['context']['component'] == s]) for s in ('reader', 'judge')},
        'source_current_run_usage': selector_cost,
        'historical_usage': {'module_stack_qa': proof['historical_current_run_usage'], 'original_N0': proof['historical_inherited_N0_usage'],
            'included_in_new_cost': False, 'source_cost_comparison': 'See the bound prior report; no old source ledger is added to current cost.'},
        'phase_counts': {p['run']: p['outcome']['counts'] for p in phases},
        'phase_intent_inventory': {p['run']: p['intent_inventory'] for p in phases},
        'phase_sources': [{'manifest': pointer(Path(p['run']) / 'run_manifest.json'),
            'outcome': pointer(Path(p['run']) / 'run_outcome.json'), 'ledgers': p['ledger_refs']} for p in phases],
        'input_sha256': {k: v for p in phases for k, v in p['input_sha256'].items()},
        'summary_code_sha256': {p: sha(ROOT / p) for p in CODE}, 'model_calls_in_summary': 0, 'native_clients': 0,
        'limitations': ['Fixed previously exposed1540 cohort; exploratory paired evidence, not blind generalization.',
            'Reference arms and exact-signature cache/inheritance are correlated reuse, not independent samples.',
            'Missing scores remain null; complete-only estimates and fixed1540 zero-imputed sensitivity are separate.',
            'Equal20 slots do not imply equal tokens or information. Missing transport/token usage is unknown, not zero.',
            'Frozen archived numeric scores are selector inputs only; deleted ZIP HTTP caches were not reconstructed.',
            'Valid NO outcomes do not enter technical recovery.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firstpass', type=Path, required=True); p.add_argument('--recovery', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--bootstrap-samples', type=int, default=2000)
    a = p.parse_args()
    with offline():
        result = summarize(a.firstpass, a.recovery, bootstrap_samples=a.bootstrap_samples)
    save_new(a.output, result)
    print(json.dumps({'output': str(a.output), 'audit_status': result['audit_status'], 'model_calls': 0}))
