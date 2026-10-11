"""V2 source admission with exactly the V1 QA function bodies.

Functions are rebound into this module's independent globals, not patched into
V1 or a writer. Only sources, schema and frozen code bindings change. Top-level
names remain importable/pickleable for Windows ProcessPool spawn.
"""
from pathlib import Path as _Path
import sys as _sys
from types import FunctionType as _FunctionType

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tools import run_regroup_refinement_qa_v1 as _v1
from tools import regroup_refinement_qa_sources_v2 as _v2_sources

# Keep canonical native retrieve/copy and plain.one objects unchanged. Copy
# names, never mutate V1's globals or any imported module object.
for _name, _value in vars(_v1).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

sources = _v2_sources
SCHEMA = 'regroup-refinement-plain-qa400-v2'
CODE = tuple(dict.fromkeys((*_v1.CODE, 'tools/run_regroup_refinement_qa_v2.py',
                           'tools/regroup_refinement_qa_sources_v2.py',
                           'tools/review_regroup_refinement_banks_v2.py',
                           'tools/mem0_regroup_plain_qa_sources_v2.py',
                           'tools/audit_regroup_refinement_banks_v1.py')))
REBOUND_FUNCTIONS = tuple(name for name, value in vars(_v1).items()
                          if isinstance(value, _FunctionType) and value.__module__ == _v1.__name__)
for _name in REBOUND_FUNCTIONS:
    _original = getattr(_v1, _name)
    _bound = _FunctionType(_original.__code__, globals(), _original.__name__,
                          _original.__defaults__, _original.__closure__)
    _bound.__kwdefaults__ = _original.__kwdefaults__
    _bound.__annotations__ = dict(_original.__annotations__)
    _bound.__doc__ = _original.__doc__
    _bound.__module__ = __name__
    _bound.__qualname__ = _original.__qualname__
    globals()[_name] = _bound


if __name__ == '__main__':
    main()
