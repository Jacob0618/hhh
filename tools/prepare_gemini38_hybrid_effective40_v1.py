"""Source-admission seam only; frozen seven-arm typing/retrieval code is reused."""
from __future__ import annotations
from pathlib import Path
from types import FunctionType
from types import SimpleNamespace
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_gemini38_hybrid_v1 as original
from tools import review_gemini38_effective40_v1 as admission

SCHEMA = 'gemini38-effective40-seven-arm-retrieval-v1'
SOURCE_SCHEMA = 'gemini38-effective40-seven-arm-selected-inputs-v1'
CODE = tuple(dict.fromkeys((*original.CODE,
    'tools/prepare_gemini38_hybrid_effective40_v1.py',
    'tools/review_gemini38_effective40_v1.py',
    'tools/recover_gemini38_session_io_v1.py')))


def clone(function, scope, name=None):
    name = name or function.__name__
    out = FunctionType(function.__code__, scope, name,
                       function.__defaults__, function.__closure__)
    out.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    out.__module__ = __name__
    out.__qualname__ = name
    return out


def snapshot_sha(path):
    # Source immutability is checked against its sealed inventory by the
    # unchanged core. A working QA copy can legitimately change on open/close;
    # do not put those snapshot bytes into the audit's immutable HASHES map.
    return hashlib.sha256(admission.idx.shared_read(path)).hexdigest()


_inventory_idx = SimpleNamespace(**{**vars(admission.idx), 'sha': snapshot_sha})
inventory = clone(admission.full_native_inventory,
                  {**admission.full_native_inventory.__globals__, 'idx': _inventory_idx},
                  'inventory')


_scope = dict(vars(original))
for _name, _function in vars(original).items():
    if isinstance(_function, FunctionType) and _function.__module__ == original.__name__:
        _scope[_name] = clone(_function, _scope, _name)
_scope.update(SCHEMA=SCHEMA, SOURCE_SCHEMA=SOURCE_SCHEMA, CODE=CODE,
              admit_gate=admission.admit_gate,
              _fact_bank=admission.validate_fact_bank,
              inventory=inventory)
for _name, _function in tuple(_scope.items()):
    if isinstance(_function, FunctionType) and _function.__module__ == __name__:
        globals()[_name] = _function

_prepare_core = _scope['prepare']


def prepare(run, gate_path=None):
    # The original prepare records the explicitly supplied gate path in its
    # manifest. Never let its historical all_bank_review default name be used.
    gate = Path(gate_path) if gate_path else Path(run)/'reviews/effective40_v1/all_bank_review.json'
    return _prepare_core(run, gate)


_scope['prepare'] = prepare

if __name__ == '__main__':
    main()
