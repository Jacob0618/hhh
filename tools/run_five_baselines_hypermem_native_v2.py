"""Fresh HyperMem continuation with exact inherited ten-request prefix.

All scientific orchestration, QA prompts, uniform recovery, and score replay
are private copies of the frozen original functions. Only namespace, adapter,
explicit prefix transport, predecessor gate, and stronger audit are supplied.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import audit_hypermem_terminal_v2 as admission
from tools import audit_hypermem_continuation_v2 as audit
from tools import baselines_hypermem_native_v2 as adapter
from tools.hypermem_prefix_api_v2 import admitted_api
from tools.baselines_common_v1 import read,sha,save_new,now,error_record

chain=admission.chain
OLD,NEW=admission.OLD,admission.NEW
ENTRY='tools/run_five_baselines_hypermem_native_v2.py'
CORE=(ENTRY,'tools/baselines_hypermem_native_v2.py','tools/hypermem_prefix_api_v2.py',
      'tools/audit_hypermem_terminal_v2.py','tools/audit_hypermem_continuation_v2.py',admission.AUDIT_SOURCE,
      'research/sources/locomo_evaluation_cbfbc1.py',
      'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/launch_hypermem_native_v2.ps1')
SCHEMA='hypermem-native-v2-continuation-manifest-v1'
REVIEW_SCHEMA='hypermem-native-v2-independent-continuation-review-v1'
DEFAULT_PROOF=ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v2/terminal_admission_v1.json'
require=admission.require
binding=admission.binding
_OWNED_EXECUTION=False


def code_hashes(parent):
    return {**parent['code_sha256'],**read(OLD/'technical_dispatch_v6/manifest.json')['code_sha256'],
            **{p:sha(ROOT/p) for p in CORE}}


def check_review(path,proof_path,proof,parent):
    path=chain.previous.simple.safe(path);review=read(path)
    require(review.get('schema')==REVIEW_SCHEMA and review.get('status')=='passed','Independent continuation review required')
    require(review.get('proof_sha256')==sha(proof_path) and review.get('specification')==parent['specification'], 'Review proof/full scientific specification differs')
    required=code_hashes(parent);code=review.get('code_sha256',{})
    require(all(code.get(p)==h for p,h in required.items()),'Review lacks current continuation code')
    for p,h in code.items():
        target=chain.previous.simple.safe(ROOT/p)
        require(target.is_relative_to(ROOT) and sha(target)==h,'Reviewed code changed/escaped workspace')
    evidence=review.get('evidence_sha256')
    require(isinstance(evidence,dict) and bool(evidence),'Independent supporting evidence bindings required')
    for p,h in evidence.items():
        target=chain.previous.simple.safe(ROOT/p)
        require(target.is_relative_to(ROOT) and sha(target)==h,'Independent supporting evidence changed/escaped workspace')
    require(review.get('upstream_sha256')==parent['upstream_sha256']==chain.base.source_inventory('hypermem'), 'Upstream review differs')
    require(all(parent['specification'].get(k)==v for k,v in adapter.describe().items()), 'Scientific adapter specification differs')
    return review


def expected_manifest(proof_path,proof,review_path):
    parent=read(OLD/'run_manifest.json');review=check_review(review_path,proof_path,proof,parent)
    m=deepcopy(parent)
    m.update(schema=SCHEMA,original_run=str(OLD),new_run=str(NEW),parent_manifest=binding(OLD/'run_manifest.json'),
             terminal_proof=binding(proof_path),review=binding(review_path),code_sha256=review['code_sha256'],
             historical_cache='Only the ten exact original HyperMem boundary raw responses in the terminal proof; no other historical reuse.',
             technical_compatibility=adapter.COMPATIBILITY,
             scientific_orchestration=chain.bridge_description(),
             parent_physical_usage=proof['original_physical_usage'],
             lifecycle={'effective_result_root':str(NEW),'original_evidence_root':str(OLD),
                        'future_cleanup_roots':[str(OLD/'work'),str(NEW/'work')],
                        'cleanup_admitted':False,'note':'Separate two-root result/archive/cleanup review required; existing single-root completion is not sufficient.'})
    return m


def pristine(prepared):
    require(chain.previous.simple.safe(NEW)==NEW and chain.previous.simple.safe(OLD)==OLD,
            'Original/continuation path is aliased or contains a reparse ancestor')
    if not prepared:
        require(not NEW.exists(),'Fresh HyperMem continuation namespace required')
    else:
        require(NEW.is_dir() and sorted(p.name for p in NEW.iterdir())==['run_manifest.json'], 'Continuation already started or undeclared artifact present')


def prepare(review,proof_path=DEFAULT_PROOF):
    pristine(False)
    proof_path=Path(proof_path).resolve();review=Path(review).resolve()
    proof=admission.validate_proof(proof_path)
    m=expected_manifest(proof_path,proof,review)
    chain.strong_predecessors('hypermem')
    pristine(False)
    save_new(NEW/'run_manifest.json',m)
    return {'status':'prepared','manifest':binding(NEW/'run_manifest.json'),'model_calls':0}


def manifest():
    m=read(NEW/'run_manifest.json')
    proof_path=Path(m['terminal_proof']['path']);require(binding(proof_path)==m['terminal_proof'],'Terminal proof changed')
    proof=admission.validate_proof(proof_path)
    require(m==expected_manifest(proof_path,proof,Path(m['review']['path'])),'Continuation manifest differs')
    return m,proof


class GateTicket:
    def __init__(self,m,proof,completions):
        self.used=False;self.m=m;self.proof=proof;self.completions=completions
        self.manifest_binding=binding(NEW/'run_manifest.json')

    def consume(self,method):
        require(method=='hypermem' and not self.used,'Wrong/reused predecessor ticket')
        require(binding(NEW/'run_manifest.json')==self.manifest_binding,'Prepared manifest changed')
        require(binding(self.m['review']['path'])==self.m['review'] and binding(self.m['terminal_proof']['path'])==self.m['terminal_proof'],'Review/proof changed before execution')
        check_review(self.m['review']['path'],self.m['terminal_proof']['path'],self.proof,read(OLD/'run_manifest.json'))
        for p,h in {**self.m['code_sha256'],**self.m['upstream_sha256']}.items():require(sha(ROOT/p)==h,'Code changed before execution')
        expected={n:{k:binding(p) for k,p in pp.items()} for n,pp in chain.completion_paths('hypermem').items()}
        require(expected==self.completions,'Strong predecessor changed before ticket use')
        pristine(True);self.used=True
        save_new(NEW/'continuation_execution_intent.json',{'schema':'hypermem-native-v2-execution-intent-v1','created_utc':now(),'pid':os.getpid(),
            'manifest':self.manifest_binding,'terminal_proof':self.m['terminal_proof'],'review':self.m['review'],
            'predecessor_acceptance':{'schema':'runtime-predecessor-acceptance-record-v1',
                'scope':'Actual current strong ZIP, receipt, completion and deleted-work gates passed. Earlier archived experiments are not current model inputs or recursively rearchived dependencies.',
                'new_scientific_input':False,'recursive_rearchive_claimed':False,
                'verified_objects':{n:{k+'_sha256':r['sha256'] for k,r in rr.items()} for n,rr in self.completions.items()}}})


def private_controller(m,proof,ticket):
    path=ROOT/'tools/run_five_baselines_gemini25_v1.py'
    require(sha(path)==chain.FROZEN_BASE_SHA,'Frozen base changed')
    text=path.read_text(encoding='utf8')
    private=types.ModuleType('_hypermem_native_v2_private_controller');private.__file__=str(path)
    exec(compile(text,str(path),'exec'),private.__dict__)
    original_manifest=private.manifest
    private.paths=lambda method: NEW if method=='hypermem' else (_ for _ in ()).throw(ValueError('Only HyperMem admitted'))
    def checked_manifest(method):
        require(method=='hypermem','Wrong method');require(read(NEW/'run_manifest.json')==m,'Manifest changed during run')
        for p,h in {**m['code_sha256'],**m['upstream_sha256']}.items(): require(sha(ROOT/p)==h,'Code changed during run')
        require(binding(m['review']['path'])==m['review'] and binding(m['terminal_proof']['path'])==m['terminal_proof'],'Run admission changed')
        check_review(m['review']['path'],m['terminal_proof']['path'],proof,read(OLD/'run_manifest.json'))
        return m
    private.manifest=checked_manifest
    private.adapter=lambda method: adapter if method=='hypermem' else (_ for _ in ()).throw(ValueError('Wrong adapter'))
    private.BaselineAPI=admitted_api(proof,NEW)
    latest_audit={}
    def complete_audit(folder):
        require(Path(folder)==NEW,'Wrong audit root')
        value=audit.audit_all(NEW,proof);latest_audit['value']=value
        return value
    private.audit_requests=complete_audit
    original_save=private.save_new
    def result_save(path,value):
        global _OWNED_EXECUTION
        if Path(path)==NEW/'results/summary.json':
            checked=latest_audit['value']
            value['usage']={'parent_original':checked['parent_usage'],'continuation':checked['new_usage'],
                            'combined_physical':checked['combined_physical_usage'],
                            'cost_usd':None,'note':'Inherited exact caches have no new model usage. Missing per-field telemetry is unknown.'}
            value['technical_continuation']={'parent_manifest':m['parent_manifest'],'terminal_proof':m['terminal_proof'],
                'original_run':str(OLD),'effective_result_root':str(NEW),'compatibility':adapter.COMPATIBILITY,
                'raw_native_audit':binding(NEW/'results/raw_request_audit.json')}
        saved=original_save(path,value)
        if Path(path)==NEW/'execution_intent.json': _OWNED_EXECUTION=True
        return saved
    private.save_new=result_save
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    require(source.count(chain.LEGACY_LOOP)==1,'Original legacy loop boundary changed')
    derived=source.replace(chain.LEGACY_LOOP,chain.BRIDGE_LINE)
    require(hashlib.sha256(derived.encode()).hexdigest()==chain.bridge_description()['derived_function_sha256'],'Scientific run body changed')
    private._consume_reviewed_predecessors=ticket.consume
    exec(compile(derived,'<hypermem-native-v2-exact-predecessor-bridge>','exec'),private.__dict__)
    return private


def run():
    m,proof=manifest();pristine(True)
    completions=chain.strong_predecessors('hypermem')
    ticket=GateTicket(m,proof,completions)
    return private_controller(m,proof,ticket).run('hypermem')


def main():
    global _OWNED_EXECUTION
    _OWNED_EXECUTION=False
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check','run'))
    p.add_argument('--review',type=Path);p.add_argument('--proof',type=Path,default=DEFAULT_PROOF);a=p.parse_args()
    if a.command=='prepare' and not a.review:p.error('--review required')
    with admission.guard_context():
        admission.assert_stopped()
        try:
            result=prepare(a.review,a.proof) if a.command=='prepare' else ({'status':'passed','manifest':binding(NEW/'run_manifest.json')} if a.command=='check' and manifest() else run())
            print(json.dumps(result,ensure_ascii=True))
        except BaseException as exc:
            if a.command=='run' and _OWNED_EXECUTION:
                chain.base.status(NEW,'needs_review',error=error_record(exc))
            raise

if __name__=='__main__':main()
