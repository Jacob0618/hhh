"""Strong read-only predecessor gate before the frozen later-method dispatcher."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import baseline_lifecycle_v1 as lifecycle
from tools import run_five_baselines_amem_native_v3_admission_v2 as frozen
from tools.baselines_common_v1 import read, sha

FROZEN_ENTRY = 'tools/run_five_baselines_amem_native_v3_admission_v2.py'
FROZEN_SHA = '73cfff288099e72c5e20e87bbe504f1c835ec144e81140f18719b868613d9ac1'


def verify_code_review(review):
    value = read(review)
    if value.get('status') != 'passed' or any(value.get('code_sha256', {}).get(p) != h
                                             for p, h in lifecycle.code_hashes().items()):
        raise ValueError('Passed supplementary review must bind the actual dispatcher and all lifecycle helpers')


def verify_predecessors(method):
    lifecycle.stopped()
    from tools import archive_amem_continuation_v3 as amem
    result = amem.verify_completion(frozen.NEW / 'cleanup_plan_native_v3.json')
    if result.get('status') != 'passed':
        raise ValueError('A-Mem strong completion verification failed')
    for prior in lifecycle.METHODS[:lifecycle.METHODS.index(method)]:
        if lifecycle.verify_completion(prior).get('status') != 'passed':
            raise ValueError('Earlier method completion verification failed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'check', 'run'))
    p.add_argument('--method', choices=lifecycle.METHODS, required=True)
    p.add_argument('--review', type=Path)
    args = p.parse_args()
    if sha(ROOT / FROZEN_ENTRY) != FROZEN_SHA:
        raise ValueError('Frozen actual dispatcher changed')
    if frozen.base.BaselineAPI is not frozen.BaselineAPI or frozen.base.adapter is not frozen._adapter:
        raise ValueError('Fresh CLI process required; shared dispatcher globals were previously modified')
    if args.command == 'prepare':
        if args.review is None:
            p.error('--review required for prepare')
        verify_code_review(args.review)
    else:
        manifest = frozen.base.manifest(args.method)
        verify_code_review(manifest['review']['path'])
    if args.command == 'run':
        verify_predecessors(args.method)
    # The frozen entry retains its routing, method order, manifest and QA gates.
    # sys.argv already has the exact arguments accepted by that entry.
    frozen.main()


if __name__ == '__main__':
    main()
