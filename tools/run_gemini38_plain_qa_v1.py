"""Seven-arm QA on newly rebuilt, sealed Gemini 3.8 selected inputs.

Thin isolated rebinding of the durable stage driver. No old module globals or
old artifacts are changed. prepare/check/recover are offline; only run dispatches.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_sdk20_plain_qa_v1 as _base
from tools import summarize_mem0_regroup_plain_qa_v1 as _audit
from jev_mem import evaluation_gemini38_v1 as evaluator
from jev_mem.gemini38_provider_v1 import Gemini38API, check_request, THINKING_LEVEL, QA_MAX_OUTPUT, WRITER_MAX_OUTPUT

SCHEMA = 'gemini38-seven-arm-plain-qa-run-v1'
SOURCE_SCHEMA = 'gemini38-seven-arm-selected-inputs-v1'
ARMS = ('baseline', 'source_local_context', 'hybrid_rank', 'hybrid_types',
        'hybrid_types_no_relation', 'boundary_multiscale', 'boundary_neighbor')
POLICY = {**copy.deepcopy(_base.POLICY), 'arms': list(ARMS), 'firstpass_slots': 10780,
          'evaluation_protocol': evaluator.PROTOCOL, 'reader_model': evaluator.MODEL,
          'judge_model': evaluator.MODEL, 'thinking_level': THINKING_LEVEL,
          'temperature': 'omitted', 'writer_max_output_tokens': WRITER_MAX_OUTPUT,
          'qa_max_output_tokens': QA_MAX_OUTPUT, 'word_limit': None,
          'judge_prompt_sha256': evaluator.JUDGE_TEMPLATE_SHA256,
          'judge_source_commit': evaluator.JUDGE_SOURCE_COMMIT,
          'judge_format_variant': True, 'judge_reference': 'full_original_reference_unchanged',
          'recovery': 'once_only_after_whole_firstpass_terminal_technical_failures_or_no_legal_label',
          'judge_parse': 'exact_or_unanimous_exact_terminal_label_once_before_recovery',
          'bounded_transport_recovery': 'terminal_ledger_1_to_3_contiguous_attempts_final_transport_error_timestamp_no_raw_once',
          'judge_compat_protocol': evaluator.JUDGE_PROTOCOL,
          'final_priority': ['firstpass_accepted_label', 'recovery_accepted_label'],
          'recovered_Reader_first_Judge_failure': 'retained_technical_failure_no_further_phase'}
CODE = tuple(dict.fromkeys((*_base.CODE,
    'tools/run_gemini38_plain_qa_v1.py', 'tools/prepare_gemini38_hybrid_v1.py',
    'jev_mem/evaluation_gemini38_v1.py', 'jev_mem/gemini38_provider_v1.py',
    'research/gemini38_retest_2026-10-03/mem0_paper_era_plain_label_FORMAT_VARIANT_61.txt')))


def _clone(function, scope):
    value = FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return value


def _replay_guard(provider, model, body, *, operation='generateContent', stage='unspecified'):
    check_request(provider, model, body, operation, stage)


class Gemini38Replay(_audit.PlainReplay):
    """File-only replay with exact new body construction, never a provider client."""
    generate = Gemini38API.generate
    request = _clone(_audit.PlainReplay.request,
                     {**_audit.PlainReplay.request.__globals__, 'check_plain_text_request': _replay_guard})


_new_audit_stage = _clone(_audit.audit_stage,
                         {**_audit.audit_stage.__globals__, 'PlainReplay': Gemini38Replay, 'evaluator': evaluator})


def source_validate(path):
    from tools.prepare_gemini38_hybrid_v1 import validate_source
    value = validate_source(Path(path))
    if value.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Only newly rebuilt seven-arm Gemini 3.8 selected sources are allowed')
    return value


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    jobs = copy.deepcopy(source['jobs'])
    if len(jobs) != 10780 or Counter(j['method'] for j in jobs) != {a: 1540 for a in ARMS}:
        raise ValueError('All seven fixed 1540-slot arms required')
    return _impl['_new_manifest'](run_dir, source_path, source, jobs=jobs, phase_kind='firstpass')


def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run = Path(run).resolve()
    if run.parent != (ROOT / 'runs').resolve() or run.exists():
        raise FileExistsError('Fresh direct runs namespace required')
    _impl['check_jobs'](jobs)
    _, provenance = _base.verified_official_metric()
    if source.get('official_metric_provenance') != provenance:
        raise ValueError('Source official scorer differs')
    value = {'schema': SCHEMA, 'created_utc': _base.now(), 'policy': POLICY,
        'source_manifest': _base.pointer(source_path), 'phase_kind': phase_kind,
        'phase_id': 'gemini38_plain_qa_' + phase_kind + '_v1', 'jobs': jobs, 'parent': parent,
        'official_metric_provenance': provenance, 'code_sha256': {p: _base.sha(ROOT / p) for p in CODE},
        'model_calls_during_prepare': 0, 'cache_roots': []}
    run.mkdir(parents=True, exist_ok=False)
    _base.publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': _base.digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs),
            'protocol_sha256': _base.digest(value), 'model_calls': 0}


def compatibility(run, j, m, reader, judge, path, api=None):
    """Record the already-completed parser mode; no second interpretation/call."""
    interpretation = None
    status = 'unavailable'
    if judge['status'] == 'complete':
        r = judge['result']
        status = 'strict_valid' if r['parse_mode'] == 'strict' else 'compat_valid'
        interpretation = {'status': status, 'label': r['label'], 'correct': r['correct'],
                          'parse_mode': r['parse_mode'], 'protocol': evaluator.JUDGE_PROTOCOL,
                          'raw_text_sha256': r['raw_text_sha256'], 'raw_modified': False}
    elif (judge.get('error') or {}).get('type') == 'InvalidJudgeLabel':
        status = 'rejected'
    _base.save_new(path, {**_impl['identity'](j, m), 'stage': 'compat', 'status': status,
        'interpretation': interpretation, 'strict': _base.pointer(_impl['stage_files'](run, j)['judge']),
        'source_evidence': None, 'error': None, 'model_calls': 0, 'raw_or_strict_modified': False,
        'note': 'Parser is part of the first Judge outcome; terminal valid output never causes rejudging.'})
    return _base.verify(path)


def resolve_pair(first, first_compat, retry=None, retry_compat=None):
    for name, row in (('firstpass_accepted_label', first), ('recovery_accepted_label', retry)):
        if row and row.get('status') == 'complete':
            r = row['result']
            return {'source': name, 'correct': r['correct'], 'label': r['label'], 'parse_mode': r['parse_mode']}
    return {'source': None, 'correct': None, 'label': None, 'parse_mode': None}


def known_failure(j, m, row, ref, stage):
    """Bounded recorded failures, after whole-phase terminal; unknowns rejected.

    Reuse the original gate's hash/context/intent/ledger and callback raw replay.
    This new protocol additionally admits fully recorded exhausted transport.
    """
    failure = row.get('error') or {}; events = row.get('api_events', [])
    allowed = {'RuntimeError', 'InvalidJudgeLabel'} if stage == 'judge' else {'RuntimeError'}
    if failure.get('type') not in allowed or len(events) != 1:
        raise ValueError('Unknown/local stage failure needs an offline audit')
    event = events[0]; context = _impl['stage_context'](j, m, stage)
    if event.get('context') != context:
        raise ValueError('Failure event context differs')
    folder = Path(ref['path']).parents[2]; api_dir = folder / 'api'; key = event['request_hash']
    rawpath = api_dir / 'api_cache' / (key + '.json'); intent = api_dir / 'request_intents' / (key + '.json')
    if Path(event['cache_path']).resolve() != rawpath.resolve() or not intent.is_file():
        raise ValueError('Failure intent/raw location differs')
    request = _base.read(intent)
    if (request['request_hash'] != key or request['stage'] != event['stage'] or request['retries'] != 2
        or not isinstance(request.get('context'), dict)):
        raise ValueError('Failure request intent differs')
    producer_context = request['context']; producer_job = j
    if producer_context != context:
        if event.get('source') != 'existing_run_cache' or not rawpath.exists():
            raise ValueError('Different producer allowed only for a bound same-phase existing raw cache consumer')
        candidates = [pj for pj in m['jobs'] if (pj['conversation'], pj['method'], pj['query_id']) ==
                      (producer_context.get('conversation'), producer_context.get('method'), producer_context.get('query_id'))]
        if len(candidates) != 1:
            raise ValueError('Cache producer is outside the sealed phase cohort')
        producer_job = candidates[0]
        if (_impl['stage_context'](producer_job, m, stage) != producer_context
            or producer_job['conversation'] != j['conversation']):
            raise ValueError('Cache producer phase/conversation/component differs')
        producer_path = folder / stage / producer_job['method'] / (producer_job['query_id'] + '.json')
        producer = _base.verify(producer_path)
        if (producer.get('status') != 'failed' or any(producer.get(k) != v for k,v in
             {**_impl['identity'](producer_job,m), 'stage':stage}.items())
            or len(producer.get('api_events', [])) != 1
            or producer['api_events'][0].get('request_hash') != key
            or producer['api_events'][0].get('context') != producer_context
            or producer['api_events'][0].get('source') != 'new_request'):
            raise ValueError('Sealed cache producer failure/event binding differs')
    snapshot = _base.Snapshot()
    ledger = [json.loads(line) for line in snapshot.raw(api_dir / 'api_calls.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
    physical = [r for r in ledger if r.get('request_hash') == key and r.get('stage') == event['stage'] and r.get('cache_hit') is not True]
    if (not physical or len(physical) > 3 or any(r.get('context') != producer_context for r in physical)
        or [r.get('attempt') for r in physical] != list(range(1, len(physical) + 1))
        or any(not isinstance(r.get('timestamp_utc'), str) or not r['timestamp_utc'] for r in physical)):
        raise ValueError('Failed request lacks complete bounded ended physical sequence')
    last = physical[-1].get('status')
    if last in ('transport_error', 429, 500, 502, 503, 504, 529) and len(physical) != 3:
        raise ValueError('Retryable terminal status lacks the complete exhausted three-attempt tail')
    if rawpath.exists():
        raw = _base.read(rawpath)
        if (event.get('error_type') is not None or event.get('cache_sha256') != _base.sha(rawpath)
            or raw['request'] != request['signature'] or last != 200):
            raise ValueError('Raw generation failure provenance differs')
        prefixes = ('Empty generation', 'Generation missing or incomplete:',
            'Missing, multiple, or incomplete generation candidates', 'Resolved Gemini model differs',
            'Unexpected tool output is retained and never executed', 'Empty ordinary generation text')
        if failure['type'] != 'InvalidJudgeLabel' and not failure.get('message', '').startswith(prefixes):
            raise ValueError('Successful raw with local failure requires offline repair')
    elif not ((last == 'transport_error' or type(last) is int and 400 <= last < 600)
              and event.get('error_type') is not None):
        raise ValueError('Unknown/no-ended raw request cannot justify redispatch')
    indexed = defaultdict(list)
    for r in ledger:
        indexed[r['request_hash'], r['stage'], _base.fingerprint(r['context'])].append(r)
    expected = {**_impl['identity'](j, m), 'stage': stage}
    if stage == 'reader':
        official, _ = _base.verified_official_metric()
        callback = lambda api: _impl['reader_call'](api, j, official)
    else:
        reader_path = folder / 'reader' / j['method'] / (j['query_id'] + '.json')
        reader = _base.verify(reader_path); expected['reader_sha256'] = _base.sha(reader_path)
        callback = lambda api: evaluator.judge_answer(api, j['question'], j['reference'], reader['result']['prediction']['answer'])
    _new_audit_stage(snapshot, Path(ref['path']), expected, api_dir, context, indexed, callback)
    if producer_context != context:
        producer_expected = {**_impl['identity'](producer_job,m), 'stage':stage}
        if stage == 'reader':
            official, _ = _base.verified_official_metric()
            producer_callback = lambda api: _impl['reader_call'](api, producer_job, official)
        else:
            prp = folder / 'reader' / producer_job['method'] / (producer_job['query_id'] + '.json')
            pr = _base.verify(prp); producer_expected['reader_sha256'] = _base.sha(prp)
            producer_callback = lambda api: evaluator.judge_answer(api, producer_job['question'], producer_job['reference'], pr['result']['prediction']['answer'])
        _new_audit_stage(snapshot, producer_path, producer_expected, api_dir, producer_context, indexed, producer_callback)
    if snapshot.unchanged():
        raise ValueError('Failed stage source changed while checking recovery')


# Clone each base-defined function into an isolated namespace. Its nested worker
# functions therefore see the new evaluator/API/policy, without mutating _base.
_impl = dict(vars(_base))
for _name, _value in vars(_base).items():
    if isinstance(_value, FunctionType) and _value.__module__ == _base.__name__:
        _impl[_name] = _clone(_value, _impl)
_impl.update({'SCHEMA': SCHEMA, 'SOURCE_SCHEMA': SOURCE_SCHEMA, 'ARMS': ARMS, 'POLICY': POLICY, 'CODE': CODE,
              'source_validate': source_validate, 'prepare': prepare, '_new_manifest': _new_manifest, 'compatibility': compatibility,
              'resolve_pair': resolve_pair, 'known_failure': known_failure, 'audit_stage': _new_audit_stage,
              'PlainTextEvaluationAPI': Gemini38API, 'read_answer': evaluator.read_answer,
              'judge_answer': evaluator.judge_answer, 'reader_input': evaluator.reader_input,
              'READER_MODEL': evaluator.MODEL, 'JUDGE_MODEL': evaluator.MODEL,
              'PROTOCOL': evaluator.PROTOCOL, 'COMPAT_PROTOCOL': evaluator.JUDGE_PROTOCOL})

for _name in ('job_key', 'check_jobs', 'stage_files', 'identity', 'stage_context', '_new_manifest',
              'recovery_jobs', 'prepare_recovery', 'manifest', 'terminal', 'reader_call', 'one', 'run'):
    globals()[_name] = _impl[_name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--run-dir', type=Path, required=True); a.add_argument('--source', type=Path, required=True)
    a = sub.add_parser('recover'); a.add_argument('--firstpass-run', type=Path, required=True)
    for name in ('check', 'run'):
        a = sub.add_parser(name); a.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    if a.phase == 'prepare': result = prepare(a.run_dir, a.source)
    elif a.phase == 'recover': result = prepare_recovery(a.firstpass_run)
    elif a.phase == 'check':
        m = manifest(a.run_dir); result = {'jobs': len(m['jobs']), 'phase_kind': m['phase_kind'], 'protocol_sha256': m['protocol_sha256'], 'model_calls': 0}
    else:
        out = run(a.run_dir); result = {k: out[k] for k in ('phase_kind', 'total', 'terminal', 'counts')}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
