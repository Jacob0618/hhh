"""Offline full-cohort union and one bounded QA recovery for SimpleMem v2.

Scientific Reader/Judge/metrics/provider functions are reused without changes.
Original failures remain in the firstpass view. Original successful generations
are immutable references, not newly charged independent samples.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import read, sha, save_new, now
from jev_mem.sdk20_cached_api import digest

ROOT = base.ROOT
OLD = base.RUN / 'simplemem'
NEW = base.RUN / 'simplemem_native_v2'
METHOD = 'simplemem'
SCHEMA = 'simplemem-native-continuation-union-v2'


class LiveView:
    read = staticmethod(read)
    sha = staticmethod(sha)
    bytes = staticmethod(lambda p: Path(p).read_bytes())


def clone(fn, **changes):
    result = FunctionType(fn.__code__, {**fn.__globals__, **changes}, fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def paths(view):
    if isinstance(view, LiveView):
        return Path
    # An archive resolver must provide bytes/read/sha/exists/glob with no live
    # fallback. The shared path facade only redirects filesystem access.
    from tools.archive_amem_continuation_v4 import archived_path_class
    if not hasattr(view, '_simplemem_path_type'):
        view._simplemem_path_type = archived_path_class(view)
    return view._simplemem_path_type


def bound(view, record):
    if view.sha(record['path']) != record['sha256']:
        raise ValueError('Bound SimpleMem evidence changed: ' + record['path'])
    return view.read(record['path'])


def api_functions(view):
    P = paths(view)
    class Replay(base.RawReplay):
        generate = clone(base.RawReplay.generate, read=view.read, sha=view.sha, Path=P)
    known = clone(base.known_recoverable, RawReplay=Replay, read=view.read, sha=view.sha, Path=P)
    return P, Replay, known


def contract(manifest, proof, view):
    parent = bound(view, manifest['parent_manifest'])
    if Path(manifest['parent_manifest']['path']) != OLD / 'run_manifest.json':
        raise ValueError('Exact original SimpleMem parent required')
    if manifest['questions'] != parent['questions'] or proof['questions'] != parent['questions']:
        raise ValueError('Fixed question/gold/date cohort differs')
    qs = parent['questions']
    if len(qs) != 1540 or len({q['query_id'] for q in qs}) != 1540:
        raise ValueError('Full unique 1540 cohort required')
    if bound(view, manifest['proof']) != proof:
        raise ValueError('Actual terminal proof differs')
    failed = {x['query_id'] for x in proof['failed_retrievals']}
    if len(failed) != len(proof['failed_retrievals']) or not failed <= {q['query_id'] for q in qs}:
        raise ValueError('Invalid original failed retrieval cohort')
    for p, h in manifest['code_sha256'].items():
        if view.sha(ROOT / p) != h:
            raise ValueError('Frozen SimpleMem union source changed: ' + p)
    return qs, failed


def source(q, owner, phase='firstpass'):
    return (owner / 'results' / phase / q['query_id'],
            owner / 'results' / ('c' + str(q['conversation'])) / 'retrieval' / (q['query_id'] + '.json'))


def replay(view, q, owner, leaf, ret, *, original_owner=None, original_leaf=None):
    P, Replay, known = api_functions(view)
    reader, judge = view.read(leaf / 'reader.json'), view.read(leaf / 'judge.json')
    retrieval = view.read(ret)
    if retrieval['status'] != 'complete':
        raise ValueError('QA lacks complete native retrieval')
    r_owner, r_leaf = owner, leaf
    if leaf.parent.name == 'recovery':
        if original_owner is None or original_leaf is None:
            raise ValueError('Recovery requires its exact original firstpass source')
        original = view.read(original_leaf / 'reader.json')
        if original['status'] not in ('complete', 'failed'):
            raise ValueError('Unclassified original Reader status')
        if bool(reader.get('inherited')) != (original['status'] == 'complete'):
            raise ValueError('Successful Reader must be inherited; failed Reader cannot be inherited')
    if reader.get('inherited'):
        if leaf.parent.name != 'recovery' or original_owner is None or original_leaf is None:
            raise ValueError('Reader inheritance only in the once-only recovery')
        original = view.read(original_leaf / 'reader.json')
        if original['status'] != 'complete' or {k: v for k, v in reader.items() if k != 'inherited'} != original:
            raise ValueError('Inherited successful Reader differs')
        r_owner, r_leaf = original_owner, original_leaf
    if judge.get('inherited'):
        raise ValueError('Judge cannot be inherited or voted')
    for kind, stage, stage_owner, stage_leaf in (
        ('reader', reader, r_owner, r_leaf), ('judge', judge, owner, leaf)
    ):
        intent = stage_leaf / (kind + '.intent.json')
        context = {'method': METHOD, 'conversation': q['conversation'], 'query_id': q['query_id'], 'phase': stage_leaf.parent.name}
        if stage['status'] in ('complete', 'failed'):
            if view.read(intent).get('context') != context or len(stage.get('events', [])) != 1:
                raise ValueError('QA stage intent/consumer cardinality differs')
            if stage['events'][0].get('context') != context:
                raise ValueError('QA stage consumer context differs')
        elif stage['status'] != 'not_run' or P(intent).exists() or stage.get('events'):
            raise ValueError('Undeclared QA stage/intent')
    if reader['status'] == 'complete':
        expected = base.evaluator.read_answer(Replay(reader['events'], P(r_owner), METHOD, q['query_id']),
            q['question'], q['reference'], [{'id': 'native_context', 'text': retrieval['value']['context'], 'observation_date': None}],
            reference_date=q['reference_date'])
        official, _ = base.verified_official_metric()
        expected['official_f1'] = official(expected['prediction']['answer'], q['reference'], q['category'])
        if expected != reader['value']:
            raise ValueError('Reader/full answer/official and custom F1 raw replay differs')
        if judge['status'] == 'complete':
            expected_j = base.evaluator.judge_answer(Replay(judge['events'], P(owner), METHOD, q['query_id']),
                q['question'], q['reference'], expected['prediction']['answer'])
            if expected_j != judge['value']:
                raise ValueError('Judge raw replay differs')
        elif judge['status'] != 'failed' or not known(judge, P(owner), METHOD, q, retrieval['value'], kind='judge', reader=reader):
            raise ValueError('Unclassified failed Judge')
    elif reader['status'] != 'failed' or judge['status'] != 'not_run' or not known(reader, P(owner), METHOD, q, retrieval['value'], kind='reader'):
        raise ValueError('Unclassified failed Reader')
    expected_outcome = {'query_id': q['query_id'], 'conversation': q['conversation'], 'method': METHOD,
        'phase': leaf.parent.name, 'reader': reader['status'], 'judge': judge['status'],
        'reader_path': str(leaf / 'reader.json'), 'judge_path': str(leaf / 'judge.json')}
    if view.read(leaf / 'outcome.json') != expected_outcome:
        raise ValueError('QA outcome identity/status/paths differ')
    return reader, judge


def row(q, reader=None, judge=None, *, phase, owner, leaf, ret):
    answer = reader['value']['prediction']['answer'] if reader and reader['status'] == 'complete' else None
    official, _ = base.verified_official_metric()
    return {**q, 'answer': answer,
        'official_f1': official(answer, q['reference'], q['category']) if answer is not None else None,
        'custom_metrics': base.evaluator.lexical_metrics(answer, q['reference']) if answer is not None else None,
        'judge_correct': judge['value']['correct'] if judge and judge['status'] == 'complete' else None,
        'reader_status': reader['status'] if reader else 'not_run', 'judge_status': judge['status'] if judge else 'not_run',
        'retrieval_status': 'complete' if reader else 'failed', 'phase': phase,
        'source_owner': str(owner), 'source_leaf': str(leaf) if reader else None, 'retrieval_source': str(ret)}


def aggregate(rows):
    n = len(rows)
    f1 = [r['official_f1'] for r in rows if r['official_f1'] is not None]
    labels = [r['judge_correct'] for r in rows if r['judge_correct'] is not None]
    custom = [r['custom_metrics']['token_f1'] for r in rows if r['custom_metrics'] is not None]
    return {'slots': n, 'reader_valid': len(f1), 'judge_valid': len(labels),
        'missing_reader': n - len(f1), 'missing_judge': n - len(labels), 'missing_imputed': False,
        'official_f1_valid_percent': 100 * math.fsum(f1) / len(f1) if f1 else None,
        'custom_f1_valid_percent': 100 * math.fsum(custom) / len(custom) if custom else None,
        'judge_valid_percent': 100 * sum(labels) / len(labels) if labels else None,
        'official_f1_full_cohort_percent': 100 * math.fsum(f1) / n if len(f1) == n else None,
        'judge_full_cohort_percent': 100 * sum(labels) / n if len(labels) == n else None,
        'official_f1_missing_bounds_percent': [100 * math.fsum(f1) / n, 100 * (math.fsum(f1) + n - len(f1)) / n],
        'judge_missing_bounds_percent': [100 * sum(labels) / n, 100 * (sum(labels) + n - len(labels)) / n]}


def pre_recovery(manifest, proof, view):
    qs, failed = contract(manifest, proof, view)
    P = paths(view)
    original, completed, sources, jobs = [], [], [], []
    for q in qs:
        old_leaf, old_ret = source(q, OLD)
        if q['query_id'] in failed:
            if view.read(old_ret)['status'] != 'failed' or P(old_leaf).exists():
                raise ValueError('Original failed retrieval/absent QA changed')
            original.append(row(q, phase='firstpass', owner=OLD, leaf=old_leaf, ret=old_ret))
            owner = NEW
        else:
            owner = OLD
        leaf, ret = source(q, owner)
        r, j = replay(view, q, owner, leaf, ret)
        if owner == OLD:
            original.append(row(q, r, j, phase='firstpass', owner=owner, leaf=leaf, ret=ret))
        completed.append(row(q, r, j, phase='completed_firstpass', owner=owner, leaf=leaf, ret=ret))
        record = {'query_id': q['query_id'], 'conversation': q['conversation'], 'owner': str(owner),
            'leaf': str(leaf), 'retrieval_path': str(ret),
            'bindings': {str(p): view.sha(p) for p in (ret, leaf / 'reader.json', leaf / 'judge.json', leaf / 'outcome.json')}}
        sources.append(record)
        if r['status'] != 'complete' or j['status'] != 'complete':
            jobs.append({'query_id': q['query_id'], 'conversation': q['conversation'],
                'stage': 'reader' if r['status'] == 'failed' else 'judge', 'source': record})
    if {p.name for p in P(NEW / 'results/firstpass').glob('*') if p.is_dir()} != failed:
        raise ValueError('Only admitted failed retrievals may have new firstpass QA')
    return original, completed, sources, jobs


def audit_qa(view, sources, jobs, *, recovered):
    P = paths(view)
    report = []
    for phase in ('firstpass', 'recovery') if recovered else ('firstpass',):
        selected = [s for s in sources if Path(s['owner']) == NEW] if phase == 'firstpass' else [j['source'] for j in jobs]
        api = NEW / 'work' / ('api_' + phase)
        if not selected:
            if P(api).exists():
                raise ValueError('Unexpected empty QA API namespace')
            continue
        events = []
        for s in selected:
            leaf = NEW / 'results' / phase / s['query_id']
            for kind in ('reader', 'judge'):
                stage = view.read(leaf / (kind + '.json'))
                if not stage.get('inherited'):
                    events.extend(stage.get('events', []))
        lines = lambda p: [json.loads(s) for s in view.bytes(p).splitlines() if s.strip()]
        recorded = lines(api / 'logical_events.jsonl')
        clean = lambda e: {k: v for k, v in e.items() if k != 'recorded_utc'}
        if Counter(digest(clean(e)) for e in recorded) != Counter(digest(e) for e in events):
            raise ValueError('All QA logical consumers must match executed stages exactly')
        keys = {e['request_hash'] for e in events}
        if {p.stem for p in P(api / 'request_intents').glob('*.json')} != keys or {p.stem for p in P(api / 'api_cache').glob('*.json')} != keys:
            raise ValueError('QA raw/intent set differs from consumers')
        ledger = lines(api / 'api_calls.jsonl')
        if any(r.get('request_hash') not in keys for r in ledger):
            raise ValueError('Unbound QA physical or cache request')
        for key in keys:
            rawpath = api / 'api_cache' / (key + '.json')
            raw = view.read(rawpath)
            intent = view.read(api / 'request_intents' / (key + '.json'))
            sig = intent['signature']
            consumers = [e for e in events if e['request_hash'] == key]
            physical = [r for r in ledger if r['request_hash'] == key and not r.get('cache_hit')]
            cached = [r for r in ledger if r['request_hash'] == key and r.get('cache_hit')]
            if (digest(sig) != key or raw.get('request') != sig or intent.get('request_hash') != key or intent.get('retries') != 2
                or not 1 <= len(physical) <= 3 or [r.get('attempt') for r in physical] != list(range(1, len(physical) + 1))
                or physical[-1].get('status') != 200
                or any(r.get('status') not in (429, 500, 502, 503, 504, 529, 'transport_error') for r in physical[:-1])):
                raise ValueError('Full QA signature/intent/bounded physical producer differs')
            for r in physical:
                if r.get('context') != intent['context'] or r.get('stage') != intent['stage'] or any(r.get(k) != sig[k] for k in ('provider', 'model', 'operation')):
                    raise ValueError('QA physical producer context differs')
            if physical[-1].get('usage') != raw['response'].get('usageMetadata', {}) or physical[-1].get('resolved_model') != raw['response'].get('modelVersion'):
                raise ValueError('QA usage or resolved model differs from raw')
            new = [e for e in consumers if e.get('source') == 'new_request']
            # Exact frozen API vocabulary; historical cache inheritance is not
            # permitted for newly issued Reader/Judge requests.
            hits = [e for e in consumers if e.get('source') == 'existing_run_cache']
            if len(new) != 1 or len(hits) != len(cached) or len(new) + len(hits) != len(consumers):
                raise ValueError('QA producer/cache consumer multiplicity differs')
            if new[0]['context'] != intent['context'] or new[0]['stage'] != intent['stage']:
                raise ValueError('QA generating consumer differs from intent')
            producer_time = [e['recorded_utc'] for e in recorded if e['request_hash'] == key and e['source'] == 'new_request']
            times = [datetime.fromisoformat(intent['created_utc']),
                *[datetime.fromisoformat(r['timestamp_utc']) for r in physical],
                *[datetime.fromisoformat(t) for t in producer_time]]
            if len(producer_time) != 1 or times != sorted(times) or raw.get('stage') != intent['stage']:
                raise ValueError('QA producer intent/attempt/consumer time order or raw stage differs')
            identity = lambda r: digest({k: r.get(k) for k in ('context', 'stage', 'provider', 'model', 'operation', 'request_hash')})
            if Counter(identity(e) for e in hits) != Counter(identity(r) for r in cached):
                raise ValueError('QA exact cache consumer ledger differs')
            for e in consumers:
                if e.get('error_type') is not None or Path(e['cache_path']) != rawpath or e.get('cache_sha256') != view.sha(rawpath) or any(e.get(k) != sig[k] for k in ('provider', 'model', 'operation')):
                    raise ValueError('QA logical response lineage differs')
            report.append({'phase': phase, 'request_hash': key, 'physical_attempts': len(physical), 'cache_hits': len(cached), 'raw_sha256': view.sha(rawpath)})
    return {'status': 'passed', 'requests': sorted(report, key=lambda r: (r['phase'], r['request_hash']))}


def usage(view):
    P = paths(view)
    owners = {}
    for name, folder in (('original', OLD), ('continuation', NEW)):
        rows, ledgers = [], {}
        for p in P(folder / 'work').glob('api_*/api_calls.jsonl'):
            rows.extend(json.loads(s) for s in view.bytes(p).splitlines() if s.strip())
            ledgers[str(p)] = view.sha(p)
        physical = [r for r in rows if not r.get('cache_hit')]
        fields = ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'totalTokenCount')
        owners[name] = {'physical_attempts': len(physical), 'exact_cache_hits': len(rows) - len(physical),
            'status_counts': dict(Counter(str(r.get('status')) for r in physical)), 'ledger_sha256': ledgers,
            'usage_unknown_or_missing': sum(bool(r.get('usage_unknown')) or not r.get('usage') for r in physical),
            'tokens': {k: {'reported_sum': sum(r['usage'][k] for r in physical if isinstance(r.get('usage'), dict) and k in r['usage'])
                if any(isinstance(r.get('usage'), dict) and k in r['usage'] for r in physical) else None,
                'reported_attempts': sum(isinstance(r.get('usage'), dict) and k in r['usage'] for r in physical)} for k in fields},
            'cost_usd': None}
    return {'owners': owners, 'physical_attempts': sum(r['physical_attempts'] for r in owners.values()),
        'exact_cache_hits': sum(r['exact_cache_hits'] for r in owners.values()), 'cost_usd': None,
        'note': 'Original and continuation physical calls counted once. Prefix reuse and inherited QA are not independent repeats. Unreported usage/cost is unknown.'}


def reconstruct(manifest, proof, *, view=None, created_utc=None):
    view = view or LiveView()
    qs, failed = contract(manifest, proof, view)
    first, completed, sources, jobs = pre_recovery(manifest, proof, view)
    if view.read(NEW / 'firstpass_sources.json') != sources:
        raise ValueError('Full completed-firstpass source table differs')
    plan = view.read(NEW / 'qa_recovery_plan.json')
    outcome = view.read(NEW / 'qa_recovery_outcome.json')
    if plan != {'round': 1, 'full_cohort': 1540, 'jobs': jobs, 'firstpass_sources_sha256': view.sha(NEW / 'firstpass_sources.json')}:
        raise ValueError('Uniform once-only QA recovery plan differs')
    if outcome.get('plan_sha256') != view.sha(NEW / 'qa_recovery_plan.json'):
        raise ValueError('QA recovery outcome plan binding differs')
    P = paths(view)
    ids = {j['query_id'] for j in jobs}
    if {p.name for p in P(NEW / 'results/recovery').glob('*') if p.is_dir()} != ids:
        raise ValueError('Only all eligible QA failures may be recovered once')
    effective = []
    recovered = []
    source_by_id = {s['query_id']: s for s in sources}
    for q, existing in zip(qs, completed):
        if q['query_id'] not in ids:
            effective.append({**existing, 'phase': 'effective'})
            continue
        origin = source_by_id[q['query_id']]
        leaf, _ = source(q, NEW, 'recovery')
        ret = Path(origin['retrieval_path'])
        r, j = replay(view, q, NEW, leaf, ret, original_owner=Path(origin['owner']), original_leaf=Path(origin['leaf']))
        effective.append(row(q, r, j, phase='effective', owner=NEW, leaf=leaf, ret=ret))
        recovered.append(view.read(leaf / 'outcome.json'))
    if sorted(outcome.get('results', []), key=lambda r: r['query_id']) != sorted(recovered, key=lambda r: r['query_id']):
        raise ValueError('Actual once-only recovery outcomes differ')
    qa_audit = audit_qa(view, sources, jobs, recovered=bool(jobs))
    from tools import run_five_baselines_simplemem_native_v2 as controller
    native_audit = controller.audit_sidecar(manifest, proof, view=view)
    if native_audit.get('status') != 'passed':
        raise ValueError('Full continuation native/request audit failed')
    valid = all(r['reader_status'] == r['judge_status'] == 'complete' for r in effective)
    summary = {'schema': SCHEMA, 'method': METHOD, 'self_root': str(NEW), 'parent_root': str(OLD),
        'audit_status': 'no_invalid_artifacts' if valid else 'known_technical_failure_retained',
        'created_utc': created_utc or now(), 'firstpass': aggregate(first),
        'completed_firstpass': aggregate(completed), 'effective': aggregate(effective),
        'qa_recovery_jobs': len(jobs), 'retrieval_continued': len(failed), 'inherited_original_qa': 1540 - len(failed),
        'by_category': {str(c): aggregate([r for r in effective if r['category'] == c]) for c in sorted({r['category'] for r in effective})},
        'usage': usage(view), 'parent_manifest': manifest['parent_manifest'], 'proof': manifest['proof'],
        'limitations': 'Gemini embedding replacement; native turn/window/planning/reflection and full context retained. Original 70 retrieval failures retained; independently qualified native compatibility continuation. No equal token budget claim.',
        'audit_content_sha256': {name: hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for name, v in (('raw_request_audit.json', qa_audit), ('method_request_audit.json', native_audit))}}
    return {'firstpass_per_query.json': first, 'completed_firstpass_per_query.json': completed,
        'per_query.json': effective, 'raw_request_audit.json': qa_audit, 'method_request_audit.json': native_audit, 'summary.json': summary}


def finish_run(manifest, proof):
    view = LiveView()
    _, _, sources, jobs = pre_recovery(manifest, proof, view)
    audit_qa(view, sources, [], recovered=False)
    from tools import run_five_baselines_simplemem_native_v2 as controller
    if controller.audit_sidecar(manifest, proof, view=view).get('status') != 'passed':
        raise ValueError('Native retrieval audit required before any QA recovery')
    save_new(NEW / 'firstpass_sources.json', sources)
    save_new(NEW / 'qa_recovery_plan.json', {'round': 1, 'full_cohort': 1540, 'jobs': jobs,
        'firstpass_sources_sha256': sha(NEW / 'firstpass_sources.json')})
    base.status(NEW, 'qa_recovery', jobs=len(jobs))
    questions = {q['query_id']: q for q in manifest['questions']}
    def recover_conv(ci):
        selected = [j for j in jobs if j['conversation'] == ci]
        def one(job):
            q = questions[job['query_id']]
            s = job['source']
            r = read(Path(s['leaf']) / 'reader.json')
            return base.qa_one(NEW, METHOD, q, read(s['retrieval_path'])['value'], phase='recovery',
                inherited_reader=r if r['status'] == 'complete' else None)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(one, selected))
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for rows in pool.map(recover_conv, range(10)):
            results.extend(rows)
    save_new(NEW / 'qa_recovery_outcome.json', {'plan_sha256': sha(NEW / 'qa_recovery_plan.json'), 'results': results})
    base.status(NEW, 'offline_audit')
    outputs = reconstruct(manifest, proof)
    for name, value in outputs.items():
        save_new(NEW / 'results' / name, value)
    return outputs['summary.json']
