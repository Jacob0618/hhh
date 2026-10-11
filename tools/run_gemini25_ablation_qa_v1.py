"""Fresh three-arm Gemini 2.5 QA, using the frozen durable QA stage engine.

prepare/check/recover/seal are offline. Only run may dispatch models. Historical
QA responses are never cache inputs; same-phase exact requests may share a raw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_plain_qa_v1 as original
from tools import summarize_mem0_regroup_plain_qa_v1 as _audit
from jev_mem import evaluation_gemini25_ablation_v1 as evaluator
from jev_mem.gemini25_ablation_provider_v1 import (
    Gemini25API, check_request, THINKING_BUDGET, QA_MAX_OUTPUT, WRITER_MAX_OUTPUT,
)

_base = original._base
SCHEMA = 'gemini25-ablation-plain-qa-run-v1'
SOURCE_SCHEMA = 'gemini25-ablation-selected-inputs-v1'
ARMS = ('baseline', 'r03_original', 'original_multi_source_closure')
PER_ARM = 1540
TOTAL = len(ARMS) * PER_ARM
RUN_NAME = 'gemini25_ablation_qa_v1_20261007'
DEFAULT_RUN = ROOT / 'runs' / RUN_NAME
POLICY = {
    **deepcopy(original.POLICY), 'arms': list(ARMS), 'firstpass_slots': TOTAL,
    'evaluation_protocol': evaluator.PROTOCOL, 'reader_model': evaluator.MODEL,
    'judge_model': evaluator.MODEL, 'thinking_budget': THINKING_BUDGET,
    'temperature': 'omitted', 'writer_max_output_tokens': WRITER_MAX_OUTPUT,
    'qa_max_output_tokens': QA_MAX_OUTPUT,
    'judge_prompt_sha256': evaluator.JUDGE_TEMPLATE_SHA256,
    'judge_source_commit': evaluator.JUDGE_SOURCE_COMMIT,
    'judge_compat_protocol': evaluator.JUDGE_PROTOCOL,
    'all_reader_and_judge_fresh': True,
    'historical_qa_cache': 'forbidden',
}
POLICY.pop('thinking_level', None)
CODE = tuple(dict.fromkeys((*_base.CODE,
    'tools/run_gemini38_plain_qa_v1.py', 'tools/run_gemini25_ablation_qa_v1.py',
    'tools/prepare_gemini25_ablation_v1.py',
    'jev_mem/evaluation_gemini38_v1.py', 'jev_mem/gemini38_provider_v1.py',
    'jev_mem/evaluation_gemini25_ablation_v1.py', 'jev_mem/gemini25_ablation_provider_v1.py',
    'research/gemini38_retest_2026-10-03/mem0_paper_era_plain_label_FORMAT_VARIANT_61.txt')))


def clone(function, scope, name=None):
    result = FunctionType(function.__code__, scope, name or function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result.__module__ = __name__
    result.__qualname__ = name or function.__qualname__
    return result


def _replay_guard(provider, model, body, *, operation='generateContent', stage='unspecified'):
    check_request(provider, model, body, operation, stage)


class Gemini25Replay(_audit.PlainReplay):
    """File-only replay: exact Gemini 2.5 body, without provider construction."""
    generate = Gemini25API.generate
    _unresolved_request = clone(_audit.PlainReplay.request,
                               {**_audit.PlainReplay.request.__globals__, 'check_plain_text_request': _replay_guard})

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', **kwargs):
        """Frozen request/producer proof with the provider's exact model aliases.

        This specializes only the request audit seam. Raw response and ledger
        values remain unmodified and must match each other byte-for-byte in
        their JSON values; no broad model-name normalization is performed.
        """
        _replay_guard(provider, model, body, operation=operation, stage=stage)
        if self.unresolved_guard:
            return self._unresolved_request(provider, model, body, operation=operation, stage=stage, **kwargs)
        signature = {'provider': provider, 'model': model, 'operation': operation, 'body': body}
        key = _audit.request_digest(signature)
        target = self.api_dir / 'api_cache' / (key + '.json')
        intent_path = self.api_dir / 'request_intents' / (key + '.json')
        if (self.api_dir / 'cache_lineage' / (key + '.json')).exists():
            raise _audit.InvalidArtifact('Historical QA cache lineage is forbidden')
        if self.snapshot is not None:
            for path in (target, intent_path):
                if path.exists():
                    self.snapshot.raw(path)
        matches = [e for e in self.events if e.get('request_hash') == key and e.get('stage') == stage]
        if len(matches) != 1 or (key, stage) in self.seen:
            raise _audit.InvalidArtifact('Missing/duplicate reconstructed request event')
        event = matches[0]
        if (event.get('provider') != provider or event.get('model') != model
            or event.get('operation') != operation or event.get('context') != self.context):
            raise _audit.InvalidArtifact('Event request/context differs')
        if event.get('source') not in ('existing_run_cache', 'new_request'):
            raise _audit.InvalidArtifact('Historical or unknown request source policy')
        self.seen.append((key, stage))
        if Path(event.get('cache_path', '')).resolve() != target:
            raise _audit.InvalidArtifact('Event cache path is outside fixed API cache')
        logs = self.ledger.get((key, stage, _audit.fingerprint(self.context)), [])
        self._logs_identity(logs, signature, key, stage)
        if target.exists():
            if event.get('error_type') is not None or event.get('cache_sha256') != _base.sha(target):
                raise _audit.InvalidArtifact('Cached response event disagrees')
            raw = _base.read(target)
            if raw.get('request') != signature or 'response' not in raw:
                raise _audit.InvalidArtifact('Raw request signature differs')
            response = raw['response']
            usage = response.get('usageMetadata', response.get('usage', {}))
            resolved = response.get('modelVersion', response.get('model'))
            if resolved not in (None, model, 'models/' + model):
                raise _audit.InvalidArtifact('Unapproved resolved model differs from frozen request model')
            good = [r for r in logs if r.get('cache_hit') is True
                    or (type(r.get('status')) is int and 200 <= r['status'] < 300)]
            if not good:
                raise _audit.InvalidArtifact('Response lacks same-context successful consumer')
            if any(r.get('usage') != usage or r.get('resolved_model') != resolved for r in good):
                raise _audit.InvalidArtifact('Consumer usage/model does not match signed raw response')
            producer = self._physical(key, stage, signature, success=True)[-1]
            if producer.get('usage') != usage or producer.get('resolved_model') != resolved:
                raise _audit.InvalidArtifact('Producer usage/model does not match raw response')
            return deepcopy(response)
        if event.get('cache_sha256') is not None or event.get('error_type') is None:
            raise _audit.InvalidArtifact('Missing raw response not declared failed')
        if not intent_path.exists():
            raise _audit.InvalidArtifact('Transport failure lacks dispatch intent')
        intent = _base.read(intent_path)
        if (intent.get('signature') != signature or intent.get('context') != self.context
            or intent.get('stage') != stage or intent.get('retries') != 2):
            raise _audit.InvalidArtifact('Transport failure intent differs')
        self._physical(key, stage, signature, success=False)
        raise _audit.DeclaredTransportFailure(event['error_type'])


_new_audit_stage = clone(_audit.audit_stage,
                        {**_audit.audit_stage.__globals__, 'PlainReplay': Gemini25Replay, 'evaluator': evaluator})


def source_validate(path):
    from tools.prepare_gemini25_ablation_v1 import validate_source
    source = validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Only the newly sealed Gemini 2.5 three-arm sources are allowed')
    return source


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    jobs = deepcopy(source['jobs'])
    if len(jobs) != TOTAL or Counter(j['method'] for j in jobs) != {a: PER_ARM for a in ARMS}:
        raise ValueError('All three fixed 1540-slot arms are required')
    if any(any(k in j for k in ('reader_source', 'parent_results', 'parent_job_sha256', 'attempt_kind')) for j in jobs):
        raise ValueError('Firstpass cannot inherit a historical Reader or QA result')
    return _impl['_new_manifest'](run_dir, source_path, source, jobs=jobs, phase_kind='firstpass')


def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run = Path(run).resolve()
    expected_name = RUN_NAME if phase_kind == 'firstpass' else RUN_NAME + '_recovery_v1'
    if phase_kind not in ('firstpass', 'technical_recovery'):
        raise ValueError('Unknown phase')
    if run.parent != (ROOT / 'runs').resolve() or run.name != expected_name or run.exists():
        raise FileExistsError('Fresh dedicated Gemini 2.5 ablation namespace required')
    _impl['check_jobs'](jobs)
    _, provenance = _base.verified_official_metric()
    if source.get('official_metric_provenance') != provenance:
        raise ValueError('Source official scorer differs')
    value = {
        'schema': SCHEMA, 'created_utc': _base.now(), 'policy': POLICY,
        'source_manifest': _base.pointer(source_path), 'phase_kind': phase_kind,
        'phase_id': 'gemini25_ablation_qa_' + phase_kind + '_v1', 'jobs': jobs, 'parent': parent,
        'official_metric_provenance': provenance, 'code_sha256': {p: _base.sha(ROOT / p) for p in CODE},
        'model_calls_during_prepare': 0, 'cache_roots': [],
    }
    run.mkdir(parents=True, exist_ok=False)
    _base.publish_json(run / 'run_manifest.json', {**value, 'protocol_sha256': _base.digest(value)})
    return {'run_dir': str(run), 'phase_kind': phase_kind, 'jobs': len(jobs),
            'protocol_sha256': _base.digest(value), 'model_calls': 0}


# Clone both the SDK durable stage functions and its frozen 3.8 overrides. Their
# nested workers resolve only this scope; importing this adapter changes neither
# historical module's globals, policies, code hashes nor result files.
_impl = {**vars(original), **original._impl}
for _name, _function in original._impl.items():
    if isinstance(_function, FunctionType) and _function.__module__ in (original.__name__, _base.__name__):
        _impl[_name] = clone(_function, _impl, _name)
_impl.update(
    _impl=_impl, ROOT=ROOT, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA,
    ARMS=ARMS, POLICY=POLICY, CODE=CODE, evaluator=evaluator,
    source_validate=source_validate, prepare=prepare, _new_manifest=_new_manifest,
    PlainTextEvaluationAPI=Gemini25API, read_answer=evaluator.read_answer,
    judge_answer=evaluator.judge_answer, reader_input=evaluator.reader_input,
    READER_MODEL=evaluator.MODEL, JUDGE_MODEL=evaluator.MODEL,
    PROTOCOL=evaluator.PROTOCOL, COMPAT_PROTOCOL=evaluator.JUDGE_PROTOCOL,
    audit_stage=_new_audit_stage, _new_audit_stage=_new_audit_stage,
)
for _name, _function in tuple(_impl.items()):
    if isinstance(_function, FunctionType) and _function.__globals__ is _impl:
        globals()[_name] = _function


def _resolved_phase(run_dir, *, source=None, parent=None):
    """Offline raw audit, including unresolved intents outside a scored leaf."""
    from tools import summarize_gemini25_ablation_v1 as summary
    run = Path(run_dir).resolve()
    m, outcome = terminal(run, _source=source)
    if any(row.get('local_errors') for row in outcome.get('outcomes', [])):
        raise ValueError('Local/controller errors require offline review')
    for row in outcome['results']:
        if any(row[stage] is None for stage in ('reader', 'judge', 'compat', 'qa')):
            raise ValueError('Unknown stage population cannot be sealed or recovered')
        reader = _base.bound(row['reader'])
        judge = _base.bound(row['judge'])
        if reader.get('status') not in ('complete', 'failed'):
            raise ValueError('Reader has an unknown/local outcome')
        if reader['status'] == 'complete' and judge.get('status') not in ('complete', 'failed'):
            raise ValueError('Judge has an unknown/local outcome')
        for stage, value in (('reader', reader), ('judge', judge)):
            if (value.get('error') or {}).get('type') in ('UnresolvedStageIntent', 'APIInitializationFailure'):
                raise ValueError('Unknown/local stage intent requires offline review')
    report = summary.audit_phase(run, source or source_validate(m['source_manifest']['path']), parent=parent)
    if report['errors']:
        raise ValueError('Invalid raw/stage/metric bindings require offline review')
    if any(v['unknown_intent_count'] or v['ended_success_missing_raw_count']
           for v in report['intent_inventory'].values()):
        raise ValueError('Unknown request intents or missing successful raw responses require offline review')
    return report


def prepare_recovery(firstpass_run):
    """Retain the original once-only population gate, after offline raw audit."""
    run = Path(firstpass_run).resolve()
    m, outcome = terminal(run)
    if m['phase_kind'] != 'firstpass':
        raise ValueError('Only one postbatch technical recovery is allowed')
    _resolved_phase(run)
    jobs = recovery_jobs(m, outcome)
    parent = {'run_dir': str(run), 'manifest': _base.pointer(run / 'run_manifest.json'),
              'outcome': _base.pointer(run / 'run_outcome.json')}
    return _new_manifest(run.with_name(run.name + '_recovery_v1'), m['source_manifest']['path'],
                         _base.read(m['source_manifest']['path']), jobs=jobs,
                         phase_kind='technical_recovery', parent=parent)


def seal(run_dir):
    """Publish an offline raw-replayed audit seal, never a leaf-count seal."""
    from tools import summarize_gemini25_ablation_v1 as summary
    run = Path(run_dir).resolve()
    m = manifest(run)
    firstpass = Path(m['parent']['run_dir']) if m['phase_kind'] == 'technical_recovery' else run
    first = _resolved_phase(firstpass)
    if m['phase_kind'] == 'technical_recovery':
        _resolved_phase(run, parent=first)
    result = summary.summarize(firstpass, run if m['phase_kind'] == 'technical_recovery' else None,
                               bootstrap_samples=1)
    if result['audit_errors']:
        raise ValueError('Invalid raw/stage/metric bindings cannot be sealed')
    inventories = [v for phase in result['phase_intent_inventory'].values() for v in phase.values()]
    if any(v['unknown_intent_count'] or v['ended_success_missing_raw_count'] for v in inventories):
        raise ValueError('Unknown request intents or missing successful raw responses cannot be sealed')
    value = {
        'schema': SCHEMA + '-raw-audit-seal', 'created_utc': _base.now(),
        'run_manifest': _base.pointer(run / 'run_manifest.json'),
        'run_outcome': _base.pointer(run / 'run_outcome.json'),
        'source_manifest': m['source_manifest'], 'phase_sources': result['phase_sources'],
        'input_sha256': result['input_sha256'], 'summary_code_sha256': result['summary_code_sha256'],
        'audit_status': result['audit_status'], 'raw_replayed': True,
        'official_f1_recomputed': True, 'custom_f1_recomputed': True,
        'model_calls': 0,
    }
    path = run / 'qa_raw_audit_seal.json'
    _base.save_new(path, value)
    return {'seal': _base.pointer(path), 'audit_status': result['audit_status'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    item = sub.add_parser('prepare')
    item.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    item.add_argument('--source', type=Path, required=True)
    item = sub.add_parser('recover')
    item.add_argument('--firstpass-run', type=Path, default=DEFAULT_RUN)
    for name in ('check', 'run', 'seal'):
        item = sub.add_parser(name)
        item.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.phase == 'prepare':
        result = prepare(args.run_dir, args.source)
    elif args.phase == 'recover':
        result = prepare_recovery(args.firstpass_run)
    elif args.phase == 'seal':
        result = seal(args.run_dir)
    elif args.phase == 'check':
        m = manifest(args.run_dir)
        result = {'jobs': len(m['jobs']), 'phase_kind': m['phase_kind'],
                  'protocol_sha256': m['protocol_sha256'], 'model_calls': 0}
    else:
        outcome = run(args.run_dir)
        result = {k: outcome[k] for k in ('phase_kind', 'total', 'terminal', 'counts')}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
