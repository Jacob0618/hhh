"""Offline raw-response audit and three-arm Gemini 2.5 ablation summary.

The scientific metrics and raw replay are the frozen QA implementation, rebound
to the new model and source contract. No writer-bank audit or provider is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import json
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_gemini38_seven_arm_v1 as original
from tools import run_gemini25_ablation_qa_v1 as runner
from jev_mem import evaluation_gemini25_ablation_v1 as evaluator

SCHEMA = 'gemini25-ablation-plain-qa-summary-v1'
ARMS = runner.ARMS
PER_ARM = runner.PER_ARM
TOTAL = runner.TOTAL
METRICS = original.METRICS
Snapshot = original.Snapshot
InvalidArtifact = original.InvalidArtifact
CODE = tuple(dict.fromkeys((*original.CODE, *runner.CODE,
                           'tools/summarize_gemini25_ablation_v1.py')))


def clone(function, scope):
    result = FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result.__module__ = __name__
    return result


_scope = dict(vars(original))
for _name, _function in vars(original).items():
    if isinstance(_function, FunctionType) and _function.__module__ == original.__name__:
        _scope[_name] = clone(_function, _scope)
_scope.update(runner=runner, evaluator=evaluator, SCHEMA=SCHEMA, ARMS=ARMS, CODE=CODE)
_scope['audit_stage'] = clone(original.old.audit_stage, {
    **original.old.audit_stage.__globals__,
    'existing': SimpleNamespace(bound=original.existing.bound, audit_stage=runner._new_audit_stage),
})
for _name in ('audit_phase', 'record_rows', 'readiness', 'intent_inventory', '_event_refs'):
    globals()[_name] = _scope[_name]


def ledger(snapshot, run, manifest):
    """Original ledger identity checks, admitting only exact provider aliases."""
    rows, refs, indexed = [], [], defaultdict(list)
    jobs = {(j['conversation'], j['method'], j['query_id']): j for j in manifest['jobs']}
    for ci in range(10):
        path = run / f'conv_{ci}/api/api_calls.jsonl'
        if not path.exists():
            continue
        data = snapshot.raw(path)
        refs.append({'path': str(path), 'sha256': snapshot.sha(path)})
        for line in data.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get('context', {})
            key = (ci, context.get('method'), context.get('query_id'))
            if key not in jobs:
                raise InvalidArtifact('Ledger job is outside phase population')
            stage = context.get('component')
            expected = {'reader': (evaluator.READER_STAGE, evaluator.READER_MODEL),
                        'judge': (evaluator.JUDGE_STAGE, evaluator.JUDGE_MODEL)}.get(stage)
            if (context != runner.stage_context(jobs[key], manifest, stage)
                or expected != (row.get('stage'), row.get('model'))
                or row.get('provider') != 'gemini' or row.get('operation') != 'generateContent'
                or row.get('resolved_model') not in (None, evaluator.MODEL, 'models/' + evaluator.MODEL)):
                raise InvalidArtifact('Unexpected QA model, stage, operation, or context')
            if row.get('cache_hit') is True:
                if 'status' in row or 'attempt' in row or row.get('usage_semantics') != 'reused_response_no_new_model_usage':
                    raise InvalidArtifact('Cache event incorrectly declares physical usage')
            elif type(row.get('attempt')) is not int or not 1 <= row['attempt'] <= 3:
                raise InvalidArtifact('Physical attempt exceeds frozen provider bound')
            rows.append(row)
            indexed[row['request_hash'], row['stage'], original.old.fingerprint(context)].append(row)
    return rows, indexed, refs


_scope['ledger'] = ledger


def measured_usage(rows, *, missing_ledgers=()):
    value = original.measured_usage(rows, missing_ledgers=missing_ledgers)
    value['pricing_status'] = 'No unverified tariff is applied; recorded Gemini 2.5 usage is not a paid invoice'
    return value


def aggregate(rows, *, bootstrap_samples=2000):
    """Same F1/Judge statistics, plus all three paired comparisons by category."""
    result = _scope['aggregate'](rows, bootstrap_samples=bootstrap_samples)
    comparisons_out = {}
    for left_arm, right_arm in combinations(ARMS, 2):
        left = {r['query_id']: r for r in rows if r['method'] == left_arm}
        right = {r['query_id']: r for r in rows if r['method'] == right_arm}
        if set(left) != set(right):
            raise InvalidArtifact('Paired arm query populations differ')

        def pair_for(qids, denominator):
            transitions = defaultdict(list)
            for qid in qids:
                a, b = left[qid], right[qid]
                if (a['conversation'], a['category']) != (b['conversation'], b['category']):
                    raise InvalidArtifact('Paired question conversation/category differs')
                x, y = a['scores']['judge_correct'], b['scores']['judge_correct']
                transitions['missing_either' if x is None or y is None else f'{int(x)}_to_{int(y)}'].append(qid)
            metrics = {}
            for name in METRICS:
                def valid(side):
                    return {qid: {'query_id': qid, 'conversation': side[qid]['conversation'], name: side[qid]['scores'][name]}
                            for qid in qids if side[qid]['scores'][name] is not None}
                metrics[name] = original.existing.paired_metric(
                    name, valid(left), valid(right), samples=bootstrap_samples, denominator=denominator)
            return {'fixed_denominator': denominator, 'metrics': metrics,
                    'judge_transitions': {key: {'count': len(transitions[key]), 'query_ids': transitions[key]}
                        for key in ('0_to_0', '0_to_1', '1_to_0', '1_to_1', 'missing_either')}}

        value = {'left': left_arm, 'right': right_arm, **pair_for(list(left), PER_ARM)}
        value['by_category'] = {}
        for category in (1, 2, 3, 4):
            qids = [qid for qid, row in left.items() if row['category'] == category]
            value['by_category'][str(category)] = pair_for(qids, len(qids))
        comparisons_out[right_arm + '_minus_' + left_arm] = value
    result['paired_comparisons'] = comparisons_out
    return result


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    if bootstrap_samples < 1:
        raise ValueError('Positive bootstrap sample count required')
    initial = original.read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(initial['source_manifest']['path'])
    if len(source['jobs']) != TOTAL or Counter(j['method'] for j in source['jobs']) != {a: PER_ARM for a in ARMS}:
        raise InvalidArtifact('Fresh three-arm 4620-source cohort required')
    first = audit_phase(firstpass, source)
    second = audit_phase(recovery, source, parent=first) if recovery else None
    if first['manifest']['phase_kind'] != 'firstpass':
        raise InvalidArtifact('Main phase is not firstpass')
    if second:
        parent = second['manifest']['parent']
        if (second['manifest']['phase_kind'] != 'technical_recovery'
            or parent['manifest'] != original.pointer(Path(firstpass) / 'run_manifest.json')
            or parent['outcome'] != original.pointer(Path(firstpass) / 'run_outcome.json')):
            raise InvalidArtifact('Recovery belongs to another firstpass')
    views = {
        'accepted_firstpass': record_rows(source['jobs'], first),
        'final_with_uniform_technical_recovery': record_rows(source['jobs'], first, second, final=True),
    }
    phases = [first] + ([second] if second else [])
    logs = [row for phase in phases for row in phase['ledger']]
    errors = [{'phase': phase['run'], **error} for phase in phases for error in phase['errors']]
    snapshot = Snapshot()
    snapshot.raw(initial['source_manifest']['path'])
    if snapshot.sha(initial['source_manifest']['path']) != initial['source_manifest']['sha256']:
        raise InvalidArtifact('Sealed source pointer changed')
    # The source validator owns bank/retrieval admission. QA only consumes this
    # freshly sealed contract; it never invokes any old writer-bank audit.
    runner.source_validate(initial['source_manifest']['path'])
    for phase in phases:
        if any(not Path(path).is_file() or original.sha(path) != hashed
               for path, hashed in phase['input_sha256'].items()):
            raise InvalidArtifact('Audited QA input changed before summary')
    if snapshot.unchanged():
        raise InvalidArtifact('Source changed during summary')
    return {
        'schema': SCHEMA, 'created_utc': original.now(), 'declared_arms': list(ARMS),
        'planned_per_arm': PER_ARM, 'planned_total': TOTAL, 'repetitions': 1,
        'source_manifest': initial['source_manifest'], 'evaluation_protocol': evaluator.PROTOCOL,
        'official_metric_provenance': source['official_metric_provenance'],
        'views': {name: aggregate(rows, bootstrap_samples=bootstrap_samples) for name, rows in views.items()},
        'slots_by_view': views, 'audit_status': 'no_invalid_artifacts' if not errors else 'invalid_artifacts_retained_as_missing',
        'audit_errors': errors,
        'phase_sources': [{'manifest': original.pointer(Path(p['run']) / 'run_manifest.json'),
                           'outcome': original.pointer(Path(p['run']) / 'run_outcome.json'),
                           'ledgers': p['ledger_refs']} for p in phases],
        'phase_counts': {p['run']: p['outcome']['counts'] for p in phases},
        'phase_usage': {p['run']: measured_usage(p['ledger'], missing_ledgers=p['missing_ledgers']) for p in phases},
        'phase_intent_inventory': {p['run']: p['intent_inventory'] for p in phases},
        'qa_physical_usage': measured_usage(logs, missing_ledgers=[x for p in phases for x in p['missing_ledgers']]),
        'qa_usage_by_stage': {stage: measured_usage([r for r in logs if r['context']['component'] == stage]) for stage in ('reader', 'judge')},
        'qa_usage_by_producer_method': {arm: measured_usage([r for r in logs if r['context']['method'] == arm]) for arm in ARMS},
        'precomputation': {'status': 'separate_source_accounting', 'source_manifest': initial['source_manifest'],
                          'note': 'Writer, retrieval, JEV and embedding usage are accounted from source ledgers separately; QA adds no estimated fees.'},
        'input_sha256': {**{k: v for p in phases for k, v in p['input_sha256'].items()},
                         **{str(k): v for k, v in snapshot.files.items()}},
        'summary_code_sha256': {p: original.sha(ROOT / p) for p in CODE},
        'model_calls_in_summary': 0, 'native_clients': 0,
        'limitations': [
            'One all-stage rebuild/run; no independent repetition standard deviation.',
            'All three arms use fresh Gemini 2.5 Reader/Judge; historical 3.8 scores are not substituted.',
            'Official LoCoMo F1 and custom token F1 differ; Judge receives the complete original reference.',
            'First legal strict or unanimous terminal label is accepted; valid YES and NO never trigger a correctness retry.',
            'Missing outcomes remain null; valid denominators, fixed1540 means and zero-imputation sensitivity are separate.',
            'Same-phase identical requests may share responses; consumers are correlated and producer-method cost is not marginal arm cost.',
            'Missing token fields remain unavailable. Prices and total paid fees are not inferred.',
            'Exploratory paired bootstrap uses the exposed ten conversations.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstpass', type=Path, default=runner.DEFAULT_RUN)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    if not args.readiness and args.output is None:
        parser.error('--output required for summary')
    if args.output is not None and args.output.exists():
        raise FileExistsError('Fresh summary output required')
    with original.historical_guard.historical.offline():
        result = readiness(args.firstpass, args.recovery) if args.readiness else summarize(
            args.firstpass, args.recovery, bootstrap_samples=args.bootstrap_samples)
    if args.output:
        original.save_new(args.output, result)
    print(json.dumps(result if args.readiness else {
        'output': str(args.output), 'audit_status': result['audit_status'], 'planned_total': TOTAL, 'model_calls': 0,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
