"""Strong v4 four-root completion gate for the original later-method runner."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import baseline_lifecycle_v2 as lifecycle
from tools import archive_amem_continuation_v4 as amem
from tools.baselines_common_v1 import read,sha

FROZEN_ENTRY='tools/run_five_baselines_amem_native_v3_admission_v2.py'
FROZEN_SHA='73cfff288099e72c5e20e87bbe504f1c835ec144e81140f18719b868613d9ac1'


def private_dispatcher():
    raw=(ROOT/FROZEN_ENTRY).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FROZEN_SHA:raise ValueError('Frozen dispatcher changed')
    module=types.ModuleType('_jev_later_methods_v4_gate');module.__file__=str(ROOT/FROZEN_ENTRY)
    exec(compile(raw,str(ROOT/FROZEN_ENTRY),'exec'),module.__dict__)
    # Only the predecessor path/gate changes. The non-A-Mem branch never
    # installs NativeAPI and retains the original adapter/QA/transport functions.
    module.NEW=amem.ROOTS['v4']
    return module


frozen=private_dispatcher()


def verify_code_review(review):
    value=read(review)
    if value.get('status')!='passed' or any(value.get('code_sha256',{}).get(p)!=h for p,h in lifecycle.code_hashes().items()):
        raise ValueError('Passed supplementary review must bind actual v2 launcher and all v4 lifecycle helpers')


def verify_predecessors(method):
    lifecycle.stopped()
    if amem.verify_completion(amem.ROOTS['v4']/'cleanup_plan_native_v4.json').get('status')!='passed':
        raise ValueError('Four-root A-Mem completion verification failed')
    for prior in lifecycle.METHODS[:lifecycle.METHODS.index(method)]:
        if lifecycle.verify_completion(prior).get('status')!='passed':raise ValueError('Earlier method completion verification failed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','check','run'))
    parser.add_argument('--method',choices=lifecycle.METHODS,required=True)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    if sha(ROOT/FROZEN_ENTRY)!=FROZEN_SHA:raise ValueError('Frozen actual dispatcher changed')
    if frozen.base.BaselineAPI is not frozen.BaselineAPI or frozen.base.adapter is not frozen._adapter:
        raise ValueError('Fresh CLI process required; provider/adapter globals changed')
    if args.command=='prepare':
        if args.review is None:parser.error('--review required for prepare')
        verify_code_review(args.review)
    else:verify_code_review(frozen.base.manifest(args.method)['review']['path'])
    # Frozen non-A-Mem run calls this before base.run; it now enforces the
    # accepted v4 archive plus each earlier later-method archive exactly once.
    frozen.require_reviewed_predecessor=lambda:verify_predecessors(args.method)
    frozen.main()


if __name__=='__main__':main()
