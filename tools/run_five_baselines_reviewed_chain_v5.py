"""Later-method launcher using the separately accepted two-root SimpleMem union."""
from contextlib import contextmanager
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_five_baselines_reviewed_chain_v4 as previous
from tools import archive_simplemem_native_v2 as simple
from tools.baselines_common_v1 import read,sha

FROZEN_CHAIN_SHA='b0a8a7a3ce37ff91b78babdbd7d458ee11de00c518a521b19386645750f377f3'

def verify_code_review(path):
    value=read(path)
    if value.get('status')!='passed' or any(value.get('code_sha256',{}).get(p)!=h for p,h in simple.code_hashes().items()):
        raise ValueError('New supplement must retain all scientific bindings plus exact two-root lifecycle/launcher/official metric code')

def verify_predecessors(method):
    simple.stopped()
    if previous.archive.verify_completion().get('status')!='passed':raise ValueError('Actual A completion required')
    for prior in previous.frozen.lifecycle.METHODS[:previous.frozen.lifecycle.METHODS.index(method)]:
        outcome=simple.verify_completion() if prior=='simplemem' else previous.frozen.lifecycle.verify_completion(prior)
        if outcome.get('status')!='passed':raise ValueError('Every earlier method must be complete and cleaned')

@contextmanager
def private_context():
    if sha(ROOT/'tools/run_five_baselines_reviewed_chain_v4.py')!=FROZEN_CHAIN_SHA:raise ValueError('Frozen chain4 changed')
    changes={'active_controllers':simple.active_controllers,'assert_stopped':simple.stopped,
        'verify_code_review':verify_code_review,'verify_predecessors':verify_predecessors}
    before={k:getattr(previous,k) for k in changes}
    try:
        for k,v in changes.items():setattr(previous,k,v)
        yield
    finally:
        for k,v in before.items():setattr(previous,k,v)

def main():
    with private_context():previous.main()
if __name__=='__main__':main()
