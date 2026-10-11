"""One method at a time, ten independent conversations, durable stage evidence.

This controller does not implement memory algorithms. It calls reviewed adapters
around the pinned upstream methods and the existing Gemini 2.5 evaluation code.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from tools.baselines_common_v1 import (
    BaselineAPI, BaselineFatal, METHODS, now, sha, read, save_new, dump_json, error_record,
)
from jev_mem.data import load_data
from jev_mem import evaluation_gemini25_ablation_v1 as evaluator
from tools.summarize_pilot400 import verified_official_metric

RUN = ROOT / 'runs/five_baselines_gemini25_v1_20261008'
RESEARCH = ROOT / 'research/baseline_execution_2026-10-08'
LOCK = ROOT / 'configs/sdk20_full1540_locked_v1.json'
DATA_SHA = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
SHARED_FILES = (
    'tools/run_five_baselines_gemini25_v1.py', 'tools/baselines_common_v1.py',
    'jev_mem/providers.py', 'jev_mem/sdk20_cached_api.py', 'jev_mem/paced_embedding_api.py',
    'jev_mem/gemini25_ablation_provider_v1.py', 'jev_mem/gemini38_provider_v1.py',
    'jev_mem/evaluation_gemini25_ablation_v1.py', 'jev_mem/evaluation_gemini38_v1.py',
    'jev_mem/evaluation_plain_text_v1.py', 'jev_mem/evaluation.py',
    'jev_mem/locomo_official_metrics.py', 'jev_mem/judge_terminal_label_compat_v1.py',
    'jev_mem/data.py', 'tools/summarize_pilot400.py', 'tools/run_pilot400_experiments_v2.py',
    'tools/cleanup_reviewed_baseline_v1.ps1',
    'research/gemini38_retest_2026-10-03/mem0_paper_era_plain_label_FORMAT_VARIANT_61.txt',
)


def adapter(method):
    return importlib.import_module('tools.baselines_' + method + '_v1')


def questions():
    locked = read(LOCK)['dataset']
    if locked['source_sha256'] != DATA_SHA or sha(ROOT / 'data/locomo10.json') != DATA_SHA:
        raise ValueError('Dataset differs')
    qs = locked['questions']
    if len(qs) != 1540 or len({q['query_id'] for q in qs}) != 1540:
        raise ValueError('Fixed 1540 identities differ')
    return qs


def paths(method):
    if method not in METHODS:
        raise ValueError('Undeclared method')
    return RUN / method


def source_inventory(method):
    base = ROOT / 'upstream/baselines_v1' / method
    files = [p for p in base.rglob('*') if p.is_file() and '.git' not in p.parts
             and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def prepare(method, review):
    folder = paths(method)
    if folder.exists():
        raise FileExistsError('Fresh per-method namespace required')
    checked = read(review)
    required = {*SHARED_FILES, 'tools/baselines_' + method + '_v1.py'}
    code = {p: sha(ROOT / p) for p in sorted(required)}
    if checked.get('status') != 'passed' or any(checked.get('code_sha256', {}).get(p) != h for p, h in code.items()):
        raise ValueError('Independent preflight review must bind current shared and method code')
    # Adapters may reuse another adapter's helper. Preserve every independently
    # reviewed dependency, instead of silently dropping the additional bindings.
    for p, h in checked['code_sha256'].items():
        target = (ROOT / p).resolve()
        if not target.is_relative_to(ROOT) or sha(target) != h:
            raise ValueError('Reviewed extra dependency changed or escaped workspace')
        code[p] = h
    upstream = source_inventory(method)
    if not upstream or checked.get('upstream_sha256') != upstream:
        raise ValueError('Review must bind all pinned upstream sources')
    _, metric = verified_official_metric()
    spec = {
        'schema': 'five-baselines-gemini25-one-method-v1', 'created_utc': now(),
        'method': method, 'method_order': list(METHODS), 'questions': questions(),
        'dataset_sha256': DATA_SHA, 'generation_model': 'gemini-2.5-flash',
        'thinkingBudget': 0, 'temperature': 'omitted',
        'internal_max_output': 16384, 'reader_judge_max_output': 8192,
        'evaluation_protocol': evaluator.PROTOCOL, 'judge_prompt_sha256': evaluator.JUDGE_TEMPLATE_SHA256,
        'conversation_workers': 10, 'query_workers_per_conversation': 4,
        'ingestion': 'chronological sessions with native internal granularity',
        'reader_context': 'complete native context in one evidence container; no top20 truncation',
        'historical_cache': 'forbidden', 'transport_retries': 2,
        'qa_recovery': 'one uniform round after all firstpass slots terminal; valid labels final',
        'code_sha256': code, 'upstream_sha256': upstream,
        'review': {'path': str(Path(review).resolve()), 'sha256': sha(review)},
        'metric': metric, 'specification': checked.get('specification'),
    }
    save_new(folder / 'run_manifest.json', spec)
    return {'method': method, 'manifest_sha256': sha(folder / 'run_manifest.json'), 'model_calls': 0}


def manifest(method):
    m = read(paths(method) / 'run_manifest.json')
    if m['questions'] != questions():
        raise ValueError('Question set changed')
    for p, h in {**m['code_sha256'], **m['upstream_sha256']}.items():
        if sha(ROOT / p) != h:
            raise ValueError('Frozen source changed: ' + p)
    if sha(m['review']['path']) != m['review']['sha256']:
        raise ValueError('Review changed')
    return m


def status(folder, stage, **extra):
    dump_json(Path(folder) / 'status.json', {'updated_utc': now(), 'pid': os.getpid(), 'stage': stage, **extra})


def conversation_only(item):
    from jev_mem.data import date_value
    import re
    conv = item['conversation']
    result = {k: deepcopy(conv[k]) for k in ('speaker_a', 'speaker_b')}
    keys = [k for k in conv if re.fullmatch(r'session_\d+', k)]
    keys.sort(key=lambda k: (date_value(conv[k + '_date_time']), int(k.split('_')[1])))
    for k in keys:
        result[k + '_date_time'] = conv[k + '_date_time']
        result[k] = deepcopy(conv[k])
    return result


def persist_memory(state, native, result_dir):
    exports = state.get('export_paths', [])
    if not exports:
        raise ValueError('Adapter must export inspectable memory')
    bound = []
    for index, source in enumerate(exports):
        p = Path(source)
        if not p.is_absolute():
            p = native / p
        p = p.resolve()
        if not p.is_relative_to(native.resolve()) or p.suffix not in ('.json', '.jsonl', '.txt', '.md') or not p.is_file():
            raise ValueError('Memory export must be a text result within this native directory')
        target = result_dir / 'memory' / (str(index) + '_' + p.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(p, target)
        if sha(p) != sha(target):
            raise ValueError('Memory export copy mismatch')
        bound.append({'path': str(target), 'sha256': sha(target)})
    return bound


def capture_stage(path, callback, api):
    save_new(path.with_suffix('.intent.json'), {'created_utc': now(), 'context': api.context})
    start = len(api.events)
    try:
        value = {'status': 'complete', 'value': callback()}
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        value = {'status': 'failed', 'error': error_record(exc, api)}
    value.update(completed_utc=now(), events=api.events[start:])
    save_new(path, value)
    return value


def qa_one(folder, method, q, retrieval, *, phase='firstpass', inherited_reader=None):
    leaf = folder / 'results' / phase / q['query_id']
    leaf.mkdir(parents=True, exist_ok=False)
    api = BaselineAPI(folder / 'work' / ('api_' + phase), context={
        'method': method, 'conversation': q['conversation'], 'query_id': q['query_id'], 'phase': phase})
    official, _ = verified_official_metric()
    if inherited_reader is not None:
        reader = deepcopy(inherited_reader)
        reader['inherited'] = True
        save_new(leaf / 'reader.json', reader)
    else:
        def do_reader():
            result = evaluator.read_answer(api, q['question'], q['reference'],
                [{'id': 'native_context', 'text': retrieval['context'], 'observation_date': None}],
                reference_date=q['reference_date'])
            result['official_f1'] = official(result['prediction']['answer'], q['reference'], q['category'])
            return result
        reader = capture_stage(leaf / 'reader.json', do_reader, api)
    if reader['status'] == 'complete':
        judge = capture_stage(leaf / 'judge.json', lambda: evaluator.judge_answer(
            api, q['question'], q['reference'], reader['value']['prediction']['answer']), api)
    else:
        judge = {'status': 'not_run', 'reason': 'reader_failed'}
        save_new(leaf / 'judge.json', judge)
    row = {'query_id': q['query_id'], 'conversation': q['conversation'], 'method': method,
           'phase': phase, 'reader': reader['status'], 'judge': judge['status'],
           'reader_path': str(leaf / 'reader.json'), 'judge_path': str(leaf / 'judge.json')}
    save_new(leaf / 'outcome.json', row)
    return row


def conv_worker(folder, method, ci, qs, item):
    native = folder / 'work' / 'native' / ('c' + str(ci))
    native.mkdir(parents=True, exist_ok=False)
    out = folder / 'results' / ('c' + str(ci))
    out.mkdir(parents=True, exist_ok=False)
    api = BaselineAPI(folder / 'work/api_method', context={
        'method': method, 'conversation': ci, 'phase': 'build'})
    save_new(out / 'build.intent.json', {'created_utc': now(), 'conversation': ci})
    try:
        status(out, 'build')
        module = adapter(method)
        state = module.build(conversation_only(item), native, api)
        memory = persist_memory(state, native, out)
        save_new(out / 'build.json', {'status': 'complete', 'memory': memory, 'completed_utc': now()})
        status(out, 'retrieve_and_qa')
        def query(q):
            query_api = api.scoped(query_id=q['query_id'], phase='retrieval')
            target = out / 'retrieval' / (q['query_id'] + '.json')
            retrieved = capture_stage(target, lambda: module.retrieve(q['question'], state, query_api), query_api)
            if retrieved['status'] != 'complete':
                return {'query_id': q['query_id'], 'status': 'retrieval_failed'}
            value = retrieved['value']
            if not isinstance(value.get('context'), str):
                raise ValueError('Adapter did not return a text context')
            return qa_one(folder, method, q, value)
        outcomes = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(query, q) for q in qs]):
                outcomes.append(future.result())
        save_new(out / 'outcome.json', {'status': 'terminal', 'results': outcomes, 'completed_utc': now()})
        status(out, 'terminal', queries=len(outcomes))
        return {'conversation': ci, 'status': 'terminal', 'results': outcomes}
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        failure = {'conversation': ci, 'status': 'needs_review', 'error': error_record(exc, api)}
        save_new(out / 'failure.json', failure)
        status(out, 'needs_review', error=failure['error'])
        return failure


def known_recoverable(stage, folder, method, q, retrieval, *, kind, reader=None):
    """A raw replay must reproduce a model-output failure, never a local bug."""
    if stage['status'] != 'failed':
        return False
    ev = stage.get('events', [])
    if not ev or not all(e.get('cache_sha256') and not e.get('error_type') for e in ev):
        return False
    replay = RawReplay(ev, folder, method, q['query_id'])
    try:
        if kind == 'reader':
            evaluator.read_answer(replay, q['question'], q['reference'],
                [{'id': 'native_context', 'text': retrieval['context'], 'observation_date': None}],
                reference_date=q['reference_date'])
        else:
            evaluator.judge_answer(replay, q['question'], q['reference'], reader['value']['prediction']['answer'])
    except (RawResponseInvalid, evaluator.InvalidJudgeLabel):
        return True
    # A complete original answer must be recovered offline after local-bug review,
    # not regenerated in search of a different answer.
    raise ValueError('Failed stage replays to a valid result; local review required, no redispatch')


def run(method):
    m = manifest(method)
    folder = paths(method)
    for prior in METHODS[:METHODS.index(method)]:
        previous = paths(prior)
        if (not (previous / 'results/cleanup_receipt.json').exists()
                or read(previous / 'results/summary.json').get('audit_status') != 'no_invalid_artifacts'):
            raise ValueError('Previous method must finish result review and cleanup before next method')
    save_new(folder / 'execution_intent.json', {'created_utc': now(), 'pid': os.getpid(),
              'manifest_sha256': sha(folder / 'run_manifest.json')})
    status(folder, 'firstpass')
    data = load_data()
    outcomes = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(conv_worker, folder, method, ci,
                   [q for q in m['questions'] if q['conversation'] == ci], item) for ci, item in enumerate(data)]
        for f in as_completed(futures):
            outcomes.append(f.result())
            status(folder, 'firstpass', conversations_terminal=len(outcomes), outcomes=outcomes)
    save_new(folder / 'firstpass_outcome.json', {'completed_utc': now(), 'outcomes': outcomes})
    if any(x['status'] != 'terminal' or any(r.get('status') == 'retrieval_failed' for r in x['results']) for x in outcomes):
        status(folder, 'needs_review', reason='build/retrieval failure; original evidence retained')
        return {'status': 'needs_review'}
    rows = [r for c in outcomes for r in c['results']]
    if len(rows) != 1540 or {r['query_id'] for r in rows} != {q['query_id'] for q in m['questions']}:
        raise ValueError('Firstpass identities incomplete')
    recover = []
    for row in rows:
        reader, judge = read(row['reader_path']), read(row['judge_path'])
        if reader['status'] == 'complete' and judge['status'] == 'complete':
            continue
        failed = reader if reader['status'] == 'failed' else judge
        q = next(q for q in m['questions'] if q['query_id'] == row['query_id'])
        retrieval = read(folder / 'results' / ('c' + str(q['conversation'])) / 'retrieval' / (q['query_id'] + '.json'))['value']
        if not known_recoverable(failed, folder, method, q, retrieval,
                                 kind='reader' if reader['status'] == 'failed' else 'judge', reader=reader):
            status(folder, 'needs_review', reason='QA failure requires request/ledger review before recovery')
            return {'status': 'needs_review'}
        recover.append(row)
    save_new(folder / 'qa_recovery_plan.json', {'created_utc': now(), 'jobs': recover})
    status(folder, 'qa_recovery', jobs=len(recover))
    recovered = []
    # ten conversation groups retain the same per-conversation query limit.
    def recover_conv(ci):
        targets = [r for r in recover if r['conversation'] == ci]
        def one(row):
            q = next(q for q in m['questions'] if q['query_id'] == row['query_id'])
            ret = read(folder / 'results' / ('c' + str(ci)) / 'retrieval' / (q['query_id'] + '.json'))['value']
            reader = read(row['reader_path'])
            return qa_one(folder, method, q, ret, phase='recovery',
                          inherited_reader=reader if reader['status'] == 'complete' else None)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(one, targets))
    with ThreadPoolExecutor(max_workers=10) as pool:
        for result in pool.map(recover_conv, range(10)):
            recovered.extend(result)
    save_new(folder / 'qa_recovery_outcome.json', {'completed_utc': now(), 'results': recovered})
    status(folder, 'offline_audit')
    summary = seal(method)
    status(folder, 'completed_awaiting_result_review', summary_sha256=sha(folder / 'results/summary.json'))
    return summary


def audit_requests(folder):
    inventory, errors = [], []
    for root in sorted((folder / 'work').glob('api_*')):
        ledger = [json.loads(x) for x in (root / 'api_calls.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (root / 'api_calls.jsonl').exists() else []
        producers = {}
        for row in ledger:
            if not row.get('cache_hit'):
                producers.setdefault(row.get('request_hash'), []).append(row)
        for p in sorted((root / 'request_intents').glob('*.json')):
            intent = read(p)
            key = p.stem
            from jev_mem.sdk20_cached_api import digest
            if digest(intent['signature']) != key:
                errors.append('Intent hash differs: ' + key)
            raw_path = root / 'api_cache' / p.name
            calls = producers.get(key, [])
            if not raw_path.exists():
                errors.append('Missing raw response requires manual failure audit: ' + key)
                continue
            raw = read(raw_path)
            if raw.get('request') != intent['signature'] or not any(type(r.get('status')) is int and 200 <= r['status'] < 300 for r in calls):
                errors.append('Raw/producer differs: ' + key)
            inventory.append({'api': root.name, 'request_hash': key, 'raw_sha256': sha(raw_path),
                              'intent_sha256': sha(p), 'operation': intent['signature']['operation']})
        intents = {p.stem for p in (root / 'request_intents').glob('*.json')}
        if {p.stem for p in (root / 'api_cache').glob('*.json')} - intents:
            errors.append('Orphan raw without intent: ' + root.name)
    return {'status': 'passed' if not errors else 'needs_review', 'errors': errors, 'inventory': inventory}


class RawResponseInvalid(RuntimeError):
    pass


class RawReplay:
    """Reconstruct the complete QA request; return only its original raw output."""
    def __init__(self, events, folder, method, query_id):
        self.events, self.folder, self.method, self.query_id = events, folder, method, query_id
        self.used = 0

    def generate(self, prompt, *, model, system=None, json_output=False, max_output, stage):
        from jev_mem.gemini25_ablation_provider_v1 import generation_body, response_text
        from jev_mem.sdk20_cached_api import digest
        if len(self.events) != 1 or self.used:
            raise ValueError('QA stage must contain exactly one logical generation')
        event = self.events[0]
        signature = {'provider': 'gemini', 'model': model, 'operation': 'generateContent',
                     'body': generation_body(prompt, system=system, json_output=json_output, max_output=max_output)}
        key = digest(signature)
        p = Path(event['cache_path']).resolve()
        if (not p.is_relative_to((self.folder / 'work').resolve()) or p.name != key + '.json'
                or event.get('request_hash') != key or event.get('stage') != stage
                or event.get('context', {}).get('method') != self.method
                or event.get('context', {}).get('query_id') != self.query_id
                or event.get('cache_sha256') != sha(p)):
            raise ValueError('Reconstructed QA request/event binding differs')
        raw = read(p)
        if raw.get('request') != signature:
            raise ValueError('Exact full QA request differs')
        if raw['response'].get('modelVersion') not in (None, model, 'models/' + model):
            raise ValueError('Unapproved resolved model requires review, never a parser retry')
        self.used += 1
        try:
            return response_text(raw['response'])
        except RuntimeError as exc:
            raise RawResponseInvalid(str(exc)) from None


def replay_qa_leaf(folder, method, q, leaf):
    reader, judge = read(leaf / 'reader.json'), read(leaf / 'judge.json')
    ret = read(folder / 'results' / ('c' + str(q['conversation'])) / 'retrieval' / (q['query_id'] + '.json'))
    if ret['status'] != 'complete':
        raise ValueError('QA lacks a complete retrieval')
    if reader.get('inherited'):
        original = read(folder / 'results/firstpass' / q['query_id'] / 'reader.json')
        if {k: v for k, v in reader.items() if k != 'inherited'} != original:
            raise ValueError('Inherited successful Reader differs')
    if reader['status'] == 'complete':
        replay = RawReplay(reader['events'], folder, method, q['query_id'])
        expected = evaluator.read_answer(replay, q['question'], q['reference'],
            [{'id': 'native_context', 'text': ret['value']['context'], 'observation_date': None}],
            reference_date=q['reference_date'])
        official, _ = verified_official_metric()
        expected['official_f1'] = official(expected['prediction']['answer'], q['reference'], q['category'])
        if reader['value'] != expected:
            raise ValueError('Persisted Reader/F1 differs from original request/raw replay')
        if judge['status'] == 'complete':
            jr = RawReplay(judge['events'], folder, method, q['query_id'])
            expected_j = evaluator.judge_answer(jr, q['question'], q['reference'], expected['prediction']['answer'])
            if judge['value'] != expected_j:
                raise ValueError('Persisted Judge differs from original request/raw replay')
        elif judge['status'] == 'failed':
            if not known_recoverable(judge, folder, method, q, ret['value'], kind='judge', reader=reader):
                raise ValueError('Unresolved failed Judge requires explicit review')
    elif reader['status'] == 'failed':
        if judge['status'] != 'not_run' or not known_recoverable(reader, folder, method, q, ret['value'], kind='reader'):
            raise ValueError('Unresolved failed Reader requires explicit review')
    else:
        raise ValueError('Undeclared Reader status')
    return reader, judge


def seal(method):
    m = manifest(method)
    folder = paths(method)
    official, _ = verified_official_metric()
    raw_audit = audit_requests(folder)
    if raw_audit['errors']:
        save_new(folder / 'raw_audit_failure.json', raw_audit)
        raise ValueError('Raw request audit failed')
    rows, firstpass = [], []
    for q in m['questions']:
        for phase, out in [('firstpass', firstpass), ('effective', rows)]:
            leaf = folder / 'results/firstpass' / q['query_id']
            if phase == 'effective' and (folder / 'results/recovery' / q['query_id']).exists():
                leaf = folder / 'results/recovery' / q['query_id']
            reader, judge = replay_qa_leaf(folder, method, q, leaf)
            ans = reader['value']['prediction']['answer'] if reader['status'] == 'complete' else None
            f1 = official(ans, q['reference'], q['category']) if ans is not None else None
            custom = evaluator.lexical_metrics(ans, q['reference']) if ans is not None else None
            if ans is not None and abs(f1 - reader['value']['official_f1']) > 1e-12:
                raise ValueError('Official F1 differs from persisted answer')
            label = evaluator.parse_judge_label(judge['value']['raw']) if judge['status'] == 'complete' else None
            if label and label['correct'] != judge['value']['correct']:
                raise ValueError('Judge differs from raw text')
            out.append({**q, 'answer': ans, 'official_f1': f1, 'custom_metrics': custom,
                'judge_correct': label['correct'] if label else None,
                'reader_status': reader['status'], 'judge_status': judge['status'], 'phase': phase})
    def aggregate(values):
        f1 = [r['official_f1'] for r in values if r['official_f1'] is not None]
        labels = [r['judge_correct'] for r in values if r['judge_correct'] is not None]
        custom = [r['custom_metrics'].get('token_f1', r['custom_metrics'].get('f1'))
                  for r in values if r['custom_metrics'] is not None]
        return {'slots': 1540, 'reader_valid': len(f1), 'judge_valid': len(labels),
                'custom_f1_valid_percent': 100 * sum(custom) / len(custom) if custom else None,
                'official_f1_valid_percent': 100 * sum(f1) / len(f1) if f1 else None,
                'official_f1_fixed_missing_zero_percent': 100 * sum(f1) / 1540,
                'judge_valid_percent': 100 * sum(labels) / len(labels) if labels else None,
                'judge_fixed_missing_zero_percent': 100 * sum(labels) / 1540,
                'missing_reader': 1540 - len(f1), 'missing_judge': 1540 - len(labels)}
    all_ledger = []
    for p in (folder / 'work').glob('api_*/api_calls.jsonl'):
        all_ledger.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip())
    physical = [r for r in all_ledger if not r.get('cache_hit')]
    usage = {'physical_attempts': len(physical), 'exact_cache_hits': len(all_ledger)-len(physical),
             'status_counts': dict(Counter(str(r.get('status')) for r in physical)),
             'usage_unknown_or_missing': sum(bool(r.get('usage_unknown')) or not r.get('usage') for r in physical),
             'known_partial_tokens': {k: sum(r.get('usage', {}).get(k, 0) for r in physical) for k in
                 ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'totalTokenCount')},
             'cost_usd': None, 'cost_note': 'Unreported usage/cost is unknown, not zero.'}
    save_new(folder / 'results/raw_request_audit.json', raw_audit)
    save_new(folder / 'results/per_query.json', rows)
    save_new(folder / 'results/firstpass_per_query.json', firstpass)
    result = {'schema': 'five-baseline-result-v1', 'method': method,
              'audit_status': 'no_invalid_artifacts', 'firstpass': aggregate(firstpass),
              'effective': aggregate(rows), 'usage': usage, 'created_utc': now()}
    save_new(folder / 'results/summary.json', result)
    return result


def archive_results(method, review):
    """Retain generation evidence once, remove no files; emit a bounded cleanup plan."""
    folder = paths(method)
    r = read(review)
    summary = folder / 'results/summary.json'
    if r.get('status') != 'passed' or r.get('summary_sha256') != sha(summary):
        raise ValueError('Independent numerical/raw result review required')
    target = folder / 'results/model_evidence.zip'
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root in sorted((folder / 'work').glob('api_*')):
            for p in sorted(root.rglob('*')):
                if not p.is_file():
                    continue
                if p.parent.name == 'api_cache':
                    value = read(p)
                    # Query/document vectors are reproducible work; full generation
                    # requests and responses remain evidence, never an active cache.
                    if value['request']['operation'] != 'generateContent':
                        continue
                z.write(p, str(p.relative_to(folder / 'work')))
    with zipfile.ZipFile(target) as z:
        bad = z.testzip()
        if bad:
            raise ValueError('Archive CRC failed: ' + bad)
    keep = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
            if p.is_file() and not p.is_relative_to(folder / 'work')}
    plan = {'schema': 'baseline-reviewed-cleanup-plan-v1', 'method': method,
            'created_utc': now(), 'result_review': {'path': str(review), 'sha256': sha(review)},
            'workspace': str(ROOT), 'allowed_delete_roots': [str((folder / 'work').resolve())],
            'keep_sha256': keep, 'archive_sha256': sha(target), 'archive_crc_verified': True,
            'deletion_executed': False}
    save_new(folder / 'cleanup_plan.json', plan)
    return {'cleanup_plan': str(folder / 'cleanup_plan.json'), 'archive_bytes': target.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'archive'])
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--review', type=Path)
    args = p.parse_args()
    if args.command in ('prepare', 'archive') and not args.review:
        p.error('--review required')
    try:
        result = (prepare(args.method, args.review) if args.command == 'prepare' else
                  archive_results(args.method, args.review) if args.command == 'archive' else run(args.method))
        print(json.dumps(result, ensure_ascii=True))
    except BaseException as exc:
        if args.command == 'run':
            status(paths(args.method), 'needs_review', error=error_record(exc))
        raise


if __name__ == '__main__':
    main()
