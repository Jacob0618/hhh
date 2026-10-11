"""VMem/PRISM dispatch after independently archived HyperMem three-root results.

Reuses the exact reviewed chain6 single-use gate and original scientific run.
No old Hyper request or cache is eligible in either new method.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_five_baselines_reviewed_chain_v6 as foundation
from tools import archive_hypermem_native_v3 as hyper
from tools.baselines_common_v1 import read,sha,save_new,now

base=foundation.base;previous=foundation.previous;guard=foundation.guard
METHODS=('vmem','prism')
ENTRY='tools/run_five_baselines_reviewed_chain_v8.py'
REVIEW_SCHEMA='baseline-reviewed-chain-v8-preparation-supplement-v1'
FROZEN_CHAIN_SHA='a9634bafcd35184c17ba8a24e631f444cb32096f8b0a3427db25503d52e65beb'
_ISSUER=object()
LEGACY_LOOP=foundation.LEGACY_LOOP;BRIDGE_LINE=foundation.BRIDGE_LINE
active_controllers=hyper.active_controllers
assert_stopped=hyper.stopped
require=hyper.require;digest=hyper.digest;binding=hyper.binding


def code_hashes():return {**foundation.code_hashes(),**hyper.code_hashes()}


def frozen_sources():
    require(sha(ROOT/'tools/run_five_baselines_reviewed_chain_v6.py')==FROZEN_CHAIN_SHA,'Frozen strong-gate foundation changed')
    foundation.frozen_sources()


def sidecar(method):
    require(method in METHODS,'Only fresh later methods permitted')
    return base.paths(method)/'technical_dispatch_v8'


@contextmanager
def guard_context():
    changes=((guard,'active_controllers',active_controllers),(guard,'assert_stopped',assert_stopped),
             (previous.simple,'_stop',assert_stopped))
    before=[(m,k,getattr(m,k)) for m,k,_ in changes]
    try:
        for m,k,v in changes:setattr(m,k,v)
        yield
    finally:
        for m,k,v in reversed(before):setattr(m,k,v)


def check_review(method,path):
    require(method in METHODS,'Only VMem/PRISM continuation of the method queue permitted')
    value=read(hyper.safe(path))
    require(value.get('schema')==REVIEW_SCHEMA and value.get('status')=='passed'
            and value.get('scope')=='implementation_preparation_supplement' and value.get('method')==method,'Exact new preparation supplement required')
    require(all(value.get('code_sha256',{}).get(p)==h for p,h in code_hashes().items()),'New lifecycle/dispatcher/scientific source omitted')
    for p,h in value['code_sha256'].items():require(sha(hyper.safe(ROOT/p))==h,'Preparation code binding changed')
    require(value.get('upstream_sha256')==base.source_inventory(method),'Original upstream differs')
    scientific=ROOT/foundation.PRIOR_REVIEW_ROOT/(method+'_preparation_review_v1.json')
    expected=foundation.PRIOR_REVIEW_SHA[method]
    require(sha(scientific)==expected,'Prior full scientific review differs')
    evidence=value.get('evidence_sha256',{})
    require(any(Path(ROOT/p)==scientific and h==expected for p,h in evidence.items()),'Prior complete scientific review not retained')
    require(value.get('specification')==read(scientific)['specification'] and
            all(value['specification'].get(k)==v for k,v in base.adapter(method).describe().items()),'Full scientific protocol changed')
    for p,h in evidence.items():require(sha(hyper.safe(ROOT/p))==h,'Preparation evidence changed')
    return value


def completion_paths(method):
    require(method in METHODS,'Wrong later method')
    result=foundation.completion_paths('hypermem')
    result['hypermem']={'completion':hyper.OUTPUT/'predecessor_completion.json','receipt':hyper.OUTPUT/'cleanup_receipt.json','plan':hyper.DEFAULT_PLAN}
    if method=='prism':
        life=previous.previous.frozen.lifecycle;r=life.folder('vmem')
        result['vmem']={'completion':r/'results/predecessor_completion.json','receipt':r/'results/cleanup_receipt.json','plan':life.plan_path('vmem')}
    return result


ATTESTATION = hyper.PARENT/'continuation_execution_intent.json'
ATTESTATION_SHA = 'b74d0f73160b048f044f4a70c5b0ad9e2dd6a5066b2ef31a23ce93f102b9611f'


def attested_predecessors():
    require(sha(hyper.safe(ATTESTATION))==ATTESTATION_SHA,'Prior runtime acceptance changed')
    ticket=read(ATTESTATION)
    require(ticket.get('schema')=='hypermem-native-v2-execution-intent-v1'
        and ticket.get('manifest',{}).get('sha256')=='474a62558f4f3905b3704e717cae3ac93de61aaa683baf41ffc9d24b5a53ffd0',
        'Wrong prior independently qualified runtime acceptance')
    accepted=ticket.get('predecessor_acceptance',{})
    require(accepted.get('schema')=='runtime-predecessor-acceptance-record-v1'
        and accepted.get('new_scientific_input') is False and accepted.get('recursive_rearchive_claimed') is False
        and set(accepted.get('verified_objects',{}))=={'amem','simplemem'},'Prior acceptance scope differs')
    paths=foundation.completion_paths('hypermem')
    observed={name:{k:binding(p) for k,p in pp.items()} for name,pp in paths.items()}
    require({name:{kind+'_sha256':record['sha256'] for kind,record in records.items()}
        for name,records in observed.items()}==accepted['verified_objects'],'Previously accepted completion/receipt/plan changed')
    for name in ('amem','amem_format_v2','amem_native_v3','amem_native_v4','simplemem','simplemem_native_v2'):
        require(not hyper.safe(base.RUN/name/'work').exists(),'Previously removed prerequisite work reappeared')
    return observed


def strong_predecessors(method):
    wanted=completion_paths(method)
    before={n:{k:binding(p) for k,p in pp.items()} for n,pp in wanted.items()}
    assert_stopped()
    accepted=attested_predecessors()
    require(all(before[n]==value for n,value in accepted.items()),'Exact prior acceptance objects differ')
    require(hyper.verify_completion().get('status')=='passed','Actual reviewed HyperMem three-root completion required')
    if method=='prism':require(previous.previous.frozen.lifecycle.verify_completion('vmem').get('status')=='passed','Actual reviewed VMem completion required')
    after={n:{k:binding(p) for k,p in pp.items()} for n,pp in wanted.items()}
    require(before==after and attested_predecessors()==accepted,'Predecessor evidence changed while verified')
    return after


def pristine(method,*,prepared=False):
    require(method in METHODS and prepared is False,'No legacy Hyper resume permission')
    return foundation.pristine(method,prepared=False)


def prepare(method,review):
    frozen_sources();check_review(method,review)
    require(not hyper.safe(base.paths(method)).exists(),'Fresh per-method namespace required')
    strong_predecessors(method)
    return base.prepare(method,review)


def manifest(method):
    frozen_sources();m=base.manifest(method);check_review(method,m['review']['path'])
    return m,None


def bridge_description():return foundation.bridge_description()


# The already independently reviewed ticket retains complete paths only while
# checking them live. Its persisted acceptance explicitly says old experiments
# are prerequisites, not current model inputs or recursive archive dependencies.
_source=(ROOT/'tools/run_five_baselines_reviewed_chain_v6.py').read_text(encoding='utf8')
require(sha(ROOT/'tools/run_five_baselines_reviewed_chain_v6.py')==FROZEN_CHAIN_SHA,'Frozen ticket source changed')
_tree=ast.parse(_source)
for _name in ('_GateTicket','private_run','run'):
    _node=next(n for n in _tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==_name)
    _text='\n'.join(_source.splitlines()[_node.lineno-1:_node.end_lineno])+'\n'
    # Labels only; the token's validation, exact original run bridge and all
    # science remain identical. New global routes point at actual Hyper union.
    _text=_text.replace('reviewed-chain-v6','reviewed-chain-v8')
    _text=_text.replace('Live completion, receipt, plan, ZIP replay and work-removal gates passed for this exact dispatch; earlier archived experiments are not new model inputs or current archive dependencies.', 'Current HyperMem and applicable VMem strong completion gates passed; A-Mem and SimpleMem retain their exact previously verified completion, receipt and plan SHAs and absent work roots. Earlier archived experiments are not new model inputs or recursively replayed here.')
    exec(compile(_text,'<reviewed-chain-v8-private-frozen-ticket>','exec'),globals())

_original_ticket_consume=_GateTicket.consume
def _consume_with_current_evidence(self,method):
    require(method in METHODS and method==self.method and not self.used,'Wrong/reused predecessor ticket')
    # Long predecessor ZIP verification must not leave a window for an
    # implementation evidence file to change before this run owns its intent.
    check_review(method,self.value['review']['path'])
    return _original_ticket_consume(self,method)
_GateTicket.consume=_consume_with_current_evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check','run'))
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--review',type=Path);a=p.parse_args()
    if a.command=='prepare' and not a.review:p.error('--review required')
    with guard_context():
        assert_stopped()
        if a.command=='prepare':result=prepare(a.method,a.review)
        elif a.command=='check':manifest(a.method);result={'status':'passed','method':a.method}
        else:result=run(a.method)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
