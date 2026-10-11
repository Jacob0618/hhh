"""Offline R02 summary: inherited R01 raw audit, same20/order-only lineage."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import summarize_evidence_exchange_r01 as prior
from tools import run_evidence_ordering_qa_r02 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read,pointer,sha,bound

SCHEMA='evidence-ordering-r02-plain-qa-summary-v1'
ARMS=runner.ARMS
CODE=tuple(dict.fromkeys((*prior.CODE,*runner.CODE,'tools/summarize_evidence_ordering_r02.py')))
original=prior.original
_cached_snapshot_json=prior._cached_snapshot_json
_ordinary_audit_stage=prior._ordinary_audit_stage
_helpers={**vars(prior),'runner':runner,'SCHEMA':SCHEMA,'ARMS':ARMS,'CODE':CODE}
audit_stage=runner.clone(prior.audit_stage,_helpers)
_helpers['audit_stage']=audit_stage
_scope={**prior._scope,'runner':runner,'ARMS':ARMS,'SCHEMA':SCHEMA,'CODE':CODE,
    'ledger':original._clone(original.old.ledger,{'runner':runner,'evaluator':original.evaluator}),
    'audit_stage':audit_stage}
for name in ('audit_phase','record_rows','aggregate','readiness'):
    _scope[name]=original._clone(getattr(original,name),_scope)
for name in ('audit_phase','record_rows','aggregate','readiness'): globals()[name]=_scope[name]
stopping_result=runner.clone(prior.stopping_result,_helpers)
_helpers.update(audit_phase=audit_phase,record_rows=record_rows,aggregate=aggregate,readiness=readiness,
                stopping_result=stopping_result)
_summarize=runner.clone(prior.summarize,_helpers)

def summarize(firstpass,recovery=None,*,bootstrap_samples=2000):
    result=_summarize(firstpass,recovery,bootstrap_samples=bootstrap_samples)
    manifest=read(Path(firstpass)/'run_manifest.json')
    source=runner.source_validate(manifest['source_manifest']['path'])
    result.update(ordering_protocol=source.get('ordering_protocol'),
        ordering_counts=source.get('ordering_counts',source.get('selection_counts')),
        unchanged_evidence_contract='exact baseline20 ID/text/date records; only order may vary',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        limitations=['Same20 evidence content is fixed; tokenizer counts can change with positional serialization/BPE. No character-to-token estimates.',
            'Single-record scores estimate fact-relative direct usefulness, not source truth or whole-answer sufficiency.',
            'Observation date is metadata and never automatically an event date.',
            'Any technical unknown retains whole original order with explicit selector lineage.',
            'Inherited exact requests are correlated consumers, not independent new draws or free cold-start cost.',
            'Judge labels are model observations, not independently established human correctness.',
            'Direction was informed by prior QA diagnoses on a seen dataset; no per-query gold ordering or parameter fitting.',
            'R01 exchange and R02 ordering cannot automatically count as two independent successful components.',
            'A complete audited summary still requires actual three-reviewer result discussion before another round.'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firstpass',type=Path,required=True); p.add_argument('--recovery',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--readiness',action='store_true')
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap_samples<1: raise ValueError('Positive bootstrap count required')
    if not a.readiness and a.output is None:p.error('--output required')
    with original.historical_guard.historical.offline():
        result=readiness(a.firstpass,a.recovery) if a.readiness else summarize(a.firstpass,a.recovery,bootstrap_samples=a.bootstrap_samples)
    if a.output:save_new(a.output,result)
    import json
    print(json.dumps(result if a.readiness else {'output':str(a.output),'audit_status':result['audit_status'],'model_calls':0},ensure_ascii=False))

if __name__=='__main__': main()
