"""Frozen seven-arm scoring with explicit original/recovery Writer accounting."""
from __future__ import annotations
from pathlib import Path
from types import FunctionType
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_gemini38_seven_arm_v1 as original
from tools import run_gemini38_plain_qa_effective40_v1 as runner
from tools import prepare_gemini38_hybrid_effective40_v1 as retrieval

SCHEMA = 'gemini38-effective40-seven-arm-qa-summary-v1'
CODE = tuple(dict.fromkeys((*original.CODE, *runner.CODE,
                           'tools/summarize_gemini38_effective40_v1.py')))


def clone(function, scope):
    out = FunctionType(function.__code__, scope, function.__name__,
                       function.__defaults__, function.__closure__)
    out.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return out


_scope = dict(vars(original))
for _name, _function in vars(original).items():
    if isinstance(_function, FunctionType) and _function.__module__ == original.__name__:
        _scope[_name] = clone(_function, _scope)
_scope.update(runner=runner, SCHEMA=SCHEMA, CODE=CODE)
_scope['ledger'] = clone(original.ledger, {**original.ledger.__globals__, 'runner': runner})


def _rows(snapshot, paths):
    result = []
    for path in paths:
        for line in snapshot.raw(path).decode('utf-8-sig').splitlines():
            if line.strip():
                result.append(json.loads(line))
    return result


def precomputation(snapshot, source):
    prior = original.precomputation(snapshot, source)
    manifest = snapshot.read(source['retrieval_manifest']['path'])
    review_path = manifest['source_review']['path']
    _, gate, admitted = retrieval.admission.admit_gate(source['parent_run'], review_path)
    if original.sha(review_path) != manifest['source_review']['sha256']:
        raise original.InvalidArtifact('Reviewed source gate changed during accounting')
    snapshot.raw(review_path)
    origin_paths = [x['path'] for x in prior['ledger_sources']]
    recovery_paths = []
    recovery_refs = []
    recovery_intents = {}
    for row in gate['banks']:
        if row['source_kind'] != 'io_recovery_complete':
            continue
        native = admitted[(row['conversation'], row['arm'])]['native']
        path = Path(native)/'api_calls.jsonl'
        if str(path) in origin_paths or path in recovery_paths or not path.is_file():
            raise original.InvalidArtifact('Recovery ledger missing/duplicate/original alias')
        rows = _rows(snapshot, [path])
        for event in rows:
            if (event.get('model') not in ('gemini-3.8-flash', 'gemini-embedding-001')
                or event.get('context', {}).get('conversation') != row['conversation']):
                raise original.InvalidArtifact('Recovery Writer ledger model/conversation differs')
        recovery_paths.append(path)
        recovery_refs.append({'path': str(path), 'sha256': snapshot.sha(path)})
        recovery_intents[str(path.parent)] = original.intent_inventory(snapshot, path.parent, rows)
    recovery_rows = _rows(snapshot, recovery_paths)
    combined_rows = _rows(snapshot, origin_paths) + recovery_rows
    prior['original_writer_tag_query_usage'] = prior['usage']
    prior['writer_recovery'] = {
        'ledger_sources': recovery_refs,
        'usage': original.measured_usage(recovery_rows),
        'intent_inventory': recovery_intents,
        'inherited_cache_events': sum(r.get('cache_hit') is True for r in recovery_rows),
        'source_review': manifest['source_review'],
        'note': 'Only executed recovery ledger is added; research probes and export metadata are not model calls.'}
    prior['usage'] = original.measured_usage(combined_rows,
        missing_ledgers=prior['original_writer_tag_query_usage']['missing_ledgers'])
    prior['by_model'] = {model: original.measured_usage(
        [r for r in combined_rows if r.get('model') == model])
        for model in sorted({r['model'] for r in combined_rows})}
    prior['effective_source_treatments'] = [
        {k: r.get(k) for k in ('conversation', 'arm', 'source_kind',
                             'original_status', 'orphan_fact_ids', 'strict_passed')}
        for r in gate['banks']]
    prior['note'] += (' Original failed Writer calls remain in original usage. '
        'Recovery imported cache is reuse, not new physical usage; missing billing stays unknown. '
        'Reviewed native exports preserve original entity-link deficits, not algorithmic repairs.')
    return prior


_scope['precomputation'] = precomputation
for _name, _function in tuple(_scope.items()):
    if isinstance(_function, FunctionType) and _function.__globals__ is _scope:
        globals()[_name] = _function

if __name__ == '__main__':
    main()
