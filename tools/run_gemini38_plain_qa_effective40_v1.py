"""Use the unchanged 3.8 Reader/Judge on explicitly reviewed effective40 sources."""
from __future__ import annotations
from pathlib import Path
from types import FunctionType
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_plain_qa_v1 as original
from tools import prepare_gemini38_hybrid_effective40_v1 as retrieval

SCHEMA = 'gemini38-effective40-seven-arm-plain-qa-run-v1'
SOURCE_SCHEMA = retrieval.SOURCE_SCHEMA
ARMS = original.ARMS
POLICY = original.POLICY
CODE = tuple(dict.fromkeys((*original.CODE, *retrieval.CODE,
                           'tools/run_gemini38_plain_qa_effective40_v1.py')))


def clone(function, scope, name=None):
    out = FunctionType(function.__code__, scope, name or function.__name__,
                       function.__defaults__, function.__closure__)
    out.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    out.__module__ = __name__
    out.__qualname__ = name or function.__qualname__
    return out


def source_validate(path):
    value = retrieval.validate_source(Path(path))
    if value.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Reviewed effective40 selected-input schema required')
    return value


# original._impl already contains the frozen legacy stage code plus the 3.8
# provider/evaluator substitutions. Clone that scope, not the legacy defaults.
_impl = {**vars(original), **original._impl}
for _name, _function in original._impl.items():
    if isinstance(_function, FunctionType) and _function.__module__ in (
            original.__name__, original._base.__name__):
        _impl[_name] = clone(_function, _impl, _name)
_impl.update(_impl=_impl, SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA,
             CODE=CODE, source_validate=source_validate)
for _name, _function in tuple(_impl.items()):
    if isinstance(_function, FunctionType) and _function.__module__ == __name__:
        globals()[_name] = _function

# Offline raw replay is unchanged; summary imports these names explicitly.
Gemini38Replay = original.Gemini38Replay
_new_audit_stage = original._new_audit_stage
main = clone(original.main, {**vars(original), **globals()}, 'main')

if __name__ == '__main__':
    main()
