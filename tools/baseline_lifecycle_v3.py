"""Single-root lifecycle with the separately reviewed query-overlay chain gate."""
from pathlib import Path
import hashlib
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FROZEN='tools/baseline_lifecycle_v2.py'
FROZEN_SHA='9347c7913858dd3c70b8cf10fac39a338efac6dfd0fc9e3b9196dac6fb7fd65c'


def _load():
    raw=(ROOT/FROZEN).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FROZEN_SHA:raise ValueError('Frozen single-root lifecycle changed')
    source=raw.decode('utf-8')
    for old,new in (
        ('run_five_baselines_reviewed_chain_v2','run_five_baselines_reviewed_chain_v3'),
        ('baseline_lifecycle_v2','baseline_lifecycle_v3'),
        ('cleanup_reviewed_chain_v2','cleanup_reviewed_chain_v3'),
        ('archive_amem_continuation_v4','archive_amem_retrieval_continuation_v1'),
        ('cleanup_amem_continuation_v4','cleanup_amem_retrieval_continuation_v1'),
        ('baseline_controller_guard_v2','baseline_controller_guard_v3'),
        ('baseline-single-root-v2','baseline-single-root-v3'),
        ('cleanup_plan_chain_v2.json','cleanup_plan_chain_v3.json'),
    ):
        if old not in source:raise ValueError('Frozen lifecycle naming boundary differs: '+old)
        source=source.replace(old,new)
    module=types.ModuleType('_jev_single_root_chain_v3');module.__file__=str(ROOT/FROZEN)
    exec(compile(source,str(ROOT/FROZEN),'exec'),module.__dict__)
    module.TOOL_FILES=tuple(dict.fromkeys((*module.TOOL_FILES,FROZEN,
        'tools/archive_amem_continuation_v4.py','tools/baseline_controller_guard_v2.py',
        'tools/seal_amem_retrieval_continuation_v1.py','tools/run_five_baselines_amem_retrieval_v1.py',
        'tools/audit_amem_retrieval_admission_v1.py','tools/baselines_amem_retrieval_v1.py')))
    return module


engine=_load()
for _name,_value in vars(engine).items():
    if not _name.startswith('__'):globals()[_name]=_value


if __name__=='__main__':main()
