"""Exact new retrieval/chain entry names on the frozen narrow process guard."""
from pathlib import Path
import hashlib
from types import FunctionType
from tools import baseline_controller_guard_v2 as frozen

FROZEN_SHA='5127497b3e7e6cb6319425ce9f1e4f6e659e1a7c87820dec147957807bcd6c26'
if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()!=FROZEN_SHA:
    raise ValueError('Frozen process guard changed')
LAUNCHERS=(*frozen.LAUNCHERS,'run_five_baselines_amem_retrieval_v1','run_five_baselines_reviewed_chain_v3')
active_controllers=FunctionType(frozen.active_controllers.__code__,
    {**frozen.active_controllers.__globals__,'LAUNCHERS':LAUNCHERS},'active_controllers',frozen.active_controllers.__defaults__)
assert_stopped=FunctionType(frozen.assert_stopped.__code__,
    {**frozen.assert_stopped.__globals__,'active_controllers':active_controllers},'assert_stopped',frozen.assert_stopped.__defaults__)
