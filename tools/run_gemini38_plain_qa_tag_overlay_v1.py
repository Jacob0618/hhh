"""Unchanged QA stages, with explicit one-fact technical tag recovery lineage."""
from pathlib import Path
from types import FunctionType
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_plain_qa_effective40_v1 as original
from tools import prepare_gemini38_hybrid_tag_overlay_v1 as retrieval

SCHEMA = 'gemini38-tag-recovered-seven-arm-plain-qa-run-v1'
SOURCE_SCHEMA = retrieval.SOURCE_SCHEMA
ARMS = original.ARMS
POLICY = original.POLICY
CODE = tuple(dict.fromkeys((*original.CODE, *retrieval.CODE,
                           'tools/run_gemini38_plain_qa_tag_overlay_v1.py')))


def clone(function, scope, name=None):
    name = name or function.__name__
    result = FunctionType(function.__code__, scope, name,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result.__module__ = __name__
    result.__qualname__ = name
    return result


def source_validate(path):
    source = retrieval.validate_source(Path(path))
    if source.get('schema') != SOURCE_SCHEMA or not source.get('tag_recovery_overlay'):
        raise ValueError('Explicit reviewed tag-recovery source required')
    return source


_impl = {**vars(original), **original._impl}
for _name, _function in original._impl.items():
    if isinstance(_function, FunctionType) and _function.__module__ == original.__name__:
        _impl[_name] = clone(_function, _impl, _name)
_impl.update(_impl=_impl, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA,
             CODE=CODE, source_validate=source_validate)
for _name, _function in tuple(_impl.items()):
    if isinstance(_function, FunctionType) and _function.__module__ == __name__:
        globals()[_name] = _function

Gemini38Replay = original.Gemini38Replay
_new_audit_stage = original._new_audit_stage
main = clone(original.main, {**original.main.__globals__, **globals()}, 'main')

if __name__ == '__main__':
    main()
