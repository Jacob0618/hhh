"""Same seven-arm scoring; separately account for the one-fact tag recovery."""
from pathlib import Path
from types import FunctionType
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_gemini38_effective40_v1 as original
from tools import run_gemini38_plain_qa_tag_overlay_v1 as runner
from tools import recover_gemini38_fact_tag_v1 as tag_recovery

SCHEMA = 'gemini38-tag-recovered-seven-arm-qa-summary-v1'
CODE = tuple(dict.fromkeys((*original.CODE, *runner.CODE,
                           'tools/summarize_gemini38_tag_overlay_v1.py')))


def clone(function, scope):
    result = FunctionType(function.__code__, scope, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


_scope = dict(original._scope)
for _name, _function in original._scope.items():
    if isinstance(_function, FunctionType) and _function.__globals__ is original._scope:
        _scope[_name] = clone(_function, _scope)
_scope.update(runner=runner, SCHEMA=SCHEMA, CODE=CODE)
_scope['ledger'] = clone(original.original.ledger,
                        {**original.original.ledger.__globals__, 'runner': runner})


def precomputation(snapshot, source):
    prior = original.precomputation(snapshot, source)
    ref = source['tag_recovery_overlay']
    if snapshot.sha(ref['path']) != ref['sha256']:
        raise original.original.InvalidArtifact('Tag recovery overlay changed')
    overlay = snapshot.read(ref['path'])
    manifest, result, _ = tag_recovery.validate_recovery(overlay['recovery_namespace'])
    result_ref = overlay['recovery_result']
    if snapshot.sha(result_ref['path']) != result_ref['sha256'] or snapshot.read(result_ref['path']) != result:
        raise original.original.InvalidArtifact('Recovered tag result provenance differs')
    ledger_ref = result['api_ledger']
    failed_ref = result['original_failed_tag']
    if snapshot.sha(failed_ref['path']) != failed_ref['sha256']:
        raise original.original.InvalidArtifact('Original failed tag changed')
    failed_tag = snapshot.read(failed_ref['path'])
    failed_event = failed_tag['api_events'][0]
    path = Path(ledger_ref['path']).resolve()
    if snapshot.sha(path) != ledger_ref['sha256']:
        raise original.original.InvalidArtifact('Recovered tag ledger changed')
    refs = [*prior['ledger_sources'], *prior['writer_recovery']['ledger_sources']]
    paths = [Path(r['path']).resolve() for r in refs]
    normalized = [str(p).casefold() for p in paths]
    if len(normalized) != len(set(normalized)) or str(path).casefold() in normalized:
        raise original.original.InvalidArtifact('Duplicate or aliased tag recovery ledger')
    rows = original._rows(snapshot, [path])
    for row in rows:
        if (row.get('provider'), row.get('model'), row.get('operation'), row.get('stage')) != (
                'jev', 'jev-1.13.0', 'systemone', 'jev_score'):
            raise original.original.InvalidArtifact('Unexpected tag recovery API operation')
        if row.get('context') != failed_event['context'] or row.get('request_hash') != failed_event['request_hash']:
            raise original.original.InvalidArtifact('Tag recovery does not match the original request context/hash')
    combined = original._rows(snapshot, paths) + rows
    prior['before_fact_tag_recovery_usage'] = prior['usage']
    prior['fact_tag_recovery'] = {
        'source_overlay': ref, 'recovery_manifest': overlay['recovery_manifest'],
        'recovery_result': result_ref, 'ledger_sources': [ledger_ref],
        'usage': original.original.measured_usage(rows),
        'intent_inventory': original.original.intent_inventory(snapshot, path.parent, rows),
        'note': 'Original exhausted attempts and the failed fact record remain first-pass evidence; this independent once-only ledger is added once.'}
    prior['usage'] = original.original.measured_usage(combined,
        missing_ledgers=prior['before_fact_tag_recovery_usage']['missing_ledgers'])
    prior['by_model'] = {model: original.original.measured_usage(
        [r for r in combined if r.get('model') == model])
        for model in sorted({r['model'] for r in combined})}
    prior['note'] += ' One fact tag was recovered under a separate reviewed execution; the first-pass tag phase was not entirely successful.'
    return prior


_scope['precomputation'] = precomputation
for _name, _function in tuple(_scope.items()):
    if isinstance(_function, FunctionType) and _function.__globals__ is _scope:
        globals()[_name] = _function

if __name__ == '__main__':
    main()
