"""Fixed1540 new input arms; reuse every original plain QA/stage/recovery body.

Only admission, population and artifact namespaces differ. No prompt, provider,
Reader, Judge, scoring, stage order or technical-recovery rule is reimplemented.
"""
from pathlib import Path as _Path
import sys as _sys
from types import FunctionType as _FunctionType
from collections import Counter as _Counter

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tools import run_sdk20_plain_qa_v1 as _old

for _name, _value in vars(_old).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

SCHEMA = 'planned-modules-plain-full1540-v1'
SOURCE_SCHEMA = 'planned-modules-selected-inputs-v1'
ARMS = ('hybrid_rank', 'hybrid_types', 'hybrid_types_no_relation',
        'boundary_multiscale', 'boundary_neighbor')
POLICY = {**_old.POLICY, 'arms': list(ARMS),
          'firstpass_slots': '1540_per_declared_source_arm',
          'selection': 'new_sealed_module_inputs_unchanged_original_plain_evaluator',
          'comparison': 'historical_locked_N0_preserved_separate_new_arms',
          'correctness_tuning': False}
CODE = tuple(dict.fromkeys((*_old.CODE,
    'tools/run_planned_plain_qa_v1.py', 'tools/run_planned_retrieval_v1.py',
    'jev_mem/hybrid_native_capture_v1.py', 'tools/freeze_sdk20_full1540_v1.py')))

# Scope independent function globals; never mutate any old module or its code.
REBOUND_FUNCTIONS = tuple(n for n, f in vars(_old).items()
    if isinstance(f, _FunctionType) and f.__module__ == _old.__name__)
for _name in REBOUND_FUNCTIONS:
    _f = getattr(_old, _name)
    _bound = _FunctionType(_f.__code__, globals(), _f.__name__, _f.__defaults__, _f.__closure__)
    _bound.__kwdefaults__ = _f.__kwdefaults__
    _bound.__annotations__ = dict(_f.__annotations__)
    _bound.__module__, _bound.__qualname__, _bound.__doc__ = __name__, _f.__qualname__, _f.__doc__
    globals()[_name] = _bound


def source_validate(path):
    from tools.run_planned_retrieval_v1 import validate_source
    return validate_source(_Path(path))


def prepare(run_dir, source_path):
    source = source_validate(source_path)
    if source.get('schema') != SOURCE_SCHEMA:
        raise ValueError('Wrong sealed source schema')
    arms = source['arms']
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError('Undeclared experiment arms')
    jobs = copy.deepcopy(source['jobs'])
    if _Counter(j['method'] for j in jobs) != {arm: 1540 for arm in arms}:
        raise ValueError('Every admitted new arm requires all1540 questions')
    return _new_manifest(run_dir, source_path, source, jobs=jobs, phase_kind='firstpass')


if __name__ == '__main__':
    main()
