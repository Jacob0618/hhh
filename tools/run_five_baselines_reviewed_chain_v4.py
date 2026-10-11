"""Route the frozen later-method dispatcher through the metric-complete ZIP."""
from contextlib import contextmanager
from types import FunctionType
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_five_baselines_reviewed_chain_v3 as frozen
from tools import archive_amem_metric_closure_v2 as archive
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import read,sha

FROZEN_CHAIN_SHA='59ec9049a82fce24cc737c80b2003b307a55bf1faaa0c367bdd313416f09b814'
FROZEN_GUARD_SHA='22671305b4288897e71e1d00b7e3e914b46987d374644222b0585559c13962bb'
LAUNCHER='run_five_baselines_reviewed_chain_v4'
active_controllers=FunctionType(guard.active_controllers.__code__,
    {**guard.active_controllers.__globals__,'LAUNCHERS':(*guard.LAUNCHERS,LAUNCHER)},
    'active_controllers',guard.active_controllers.__defaults__)
assert_stopped=FunctionType(guard.assert_stopped.__code__,
    {**guard.assert_stopped.__globals__,'active_controllers':active_controllers},
    'assert_stopped',guard.assert_stopped.__defaults__)


def verify_code_review(path):
    value=read(path)
    if value.get('status')!='passed' or any(value.get('code_sha256',{}).get(p)!=h for p,h in archive.code_hashes().items()):
        raise ValueError('Supplement must retain original bindings plus three metric-archive tools and official metric source')


def verify_predecessors(method):
    assert_stopped()
    if archive.verify_completion().get('status')!='passed':raise ValueError('Metric-complete A archive required')
    for prior in frozen.lifecycle.METHODS[:frozen.lifecycle.METHODS.index(method)]:
        if frozen.lifecycle.verify_completion(prior).get('status')!='passed':raise ValueError('Earlier method not completed')


@contextmanager
def private_context():
    if sha(ROOT/'tools/run_five_baselines_reviewed_chain_v3.py')!=FROZEN_CHAIN_SHA or sha(ROOT/'tools/baseline_controller_guard_v3.py')!=FROZEN_GUARD_SHA:
        raise ValueError('Frozen chain or guard changed')
    changes=((guard,'active_controllers',active_controllers),(guard,'assert_stopped',assert_stopped),
        (frozen.amem,'DEFAULT_PLAN',archive.DEFAULT_PLAN),(frozen,'verify_code_review',verify_code_review),
        (frozen,'verify_predecessors',verify_predecessors))
    previous=[(module,key,getattr(module,key)) for module,key,_ in changes]
    try:
        for module,key,value in changes:setattr(module,key,value)
        yield
    finally:
        for module,key,value in reversed(previous):setattr(module,key,value)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('command',choices=('prepare','check','run'))
    parser.add_argument('--method',choices=frozen.lifecycle.METHODS,required=True)
    args,_=parser.parse_known_args()
    with private_context():
        if args.command=='prepare':verify_predecessors(args.method)
        frozen.main()


if __name__=='__main__':main()
