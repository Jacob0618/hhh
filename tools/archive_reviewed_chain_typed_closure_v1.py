"""Archive-only compatibility entry. Existing readers/cleanup/launchers stay frozen.

The two extra implementation sources enter each archive and permanent keep map
through the required independently reviewed code SHA table. Existing archive
schemas and exact runtime verifier code tables are deliberately unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import archive_amem_retrieval_continuation_v1 as amem
from tools import baseline_lifecycle_v3 as lifecycle
from tools import baseline_evidence_closure_v1 as typed
from tools.baselines_common_v1 import read,sha

EXTRA=('tools/baseline_evidence_closure_v1.py','tools/archive_reviewed_chain_typed_closure_v1.py')
REVIEW_SCHEMA='single-query-overlay-typed-closure-independent-review-v1'


def code_hashes():
    return {**lifecycle.code_hashes(),**{p:sha(ROOT/p) for p in EXTRA}}


def verify_review(path):
    path=amem.safe_path(path);value=read(path)
    if (value.get('schema')!=REVIEW_SCHEMA or value.get('status')!='passed'
            or value.get('scope')!='archive_generation_typed_closure_compatibility_only'
            or any(value.get('code_sha256',{}).get(p)!=h for p,h in code_hashes().items())):
        raise ValueError('Independent typed-closure review must bind all original sixteen and two added sources')
    if not value.get('evidence_sha256'):raise ValueError('Typed-closure review must retain independent actual closure evidence')
    for p,h in value['evidence_sha256'].items():
        if sha(amem.safe_path(ROOT/p))!=h:raise ValueError('Typed-closure review evidence changed')
    return value


def evidence_closure(initial, *, annotations=None):
    return typed.collect(initial,root=ROOT,safe_path=amem.safe_path,read=read,sha=sha,annotations=annotations)


@contextmanager
def compatibility(method):
    """Only the archive builder's process-local closure hook is replaced."""
    if method=='amem':
        module=amem.engine;attribute='closure';original=getattr(module,attribute)
        def adapted(initial):
            return evidence_closure([*initial,(str(amem.FINAL/'run_manifest.json'),sha(amem.FINAL/'run_manifest.json'))])
    elif method in lifecycle.METHODS:
        module=lifecycle.engine;attribute='evidence_closure';original=getattr(module,attribute)
        adapted=evidence_closure
    else:raise ValueError('Only the fixed five methods are accepted')
    setattr(module,attribute,adapted)
    try:yield
    finally:setattr(module,attribute,original)


def archive(method,result_review,tool_review):
    # The unchanged builders enforce stopped controllers, result/raw replay,
    # exact inventories, one-use archive output, and immutable review bindings.
    # Validate the added source bindings BEFORE entering either archive builder.
    verify_review(tool_review)
    with compatibility(method):
        result=amem.archive(result_review,tool_review) if method=='amem' else lifecycle.archive(method,result_review,tool_review)
    # Original verify/complete functions never call an evidence closure. Their
    # exact code tables stay unchanged; all eighteen reviewed source bytes are
    # instead in their existing permanent keep_sha256 and ZIP evidence maps.
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('archive',))
    p.add_argument('--method',choices=('amem',*lifecycle.METHODS),required=True)
    p.add_argument('--result-review',type=Path,required=True);p.add_argument('--tool-review',type=Path,required=True)
    args=p.parse_args();print(json.dumps(archive(args.method,args.result_review,args.tool_review),ensure_ascii=True))


if __name__=='__main__':main()
