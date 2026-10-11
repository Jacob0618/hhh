"""Fresh HyperMem continuation with complete ten-conversation retained prefix.

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
from tools import audit_hypermem_terminal_v3 as admission
from tools import audit_hypermem_continuation_v3 as audit
from tools import baselines_hypermem_native_v3 as adapter
from tools.hypermem_prefix_api_v3 import admitted_api
from tools.baselines_common_v1 import read,sha,save_new,now,error_record

chain=admission.chain
ANCESTOR,OLD,NEW=admission.ANCESTOR,admission.OLD,admission.NEW
ENTRY='tools/run_five_baselines_hypermem_native_v3.py'
CORE=(ENTRY,'tools/baselines_hypermem_native_v3.py','tools/hypermem_prefix_api_v3.py',
      'tools/audit_hypermem_terminal_v3.py','tools/audit_hypermem_continuation_v3.py',admission.COLLECTOR,
      'research/sources/locomo_evaluation_cbfbc1.py',
      'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/launch_hypermem_native_v3.ps1')
SCHEMA='hypermem-native-v3-continuation-manifest-v1'
REVIEW_SCHEMA='hypermem-native-v3-independent-continuation-review-v1'
DEFAULT_PROOF=ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v3/terminal_admission_v1.json'
PREFLIGHT_HARNESS='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_native_v3_prefix_author_v2.py'
require=admission.require
binding=admission.binding
_OWNED_EXECUTION=False


def policy(proof,prospective=False):
    require(type(prospective) is bool,'Prospective representation must be an explicit boolean')
    return {'schema':'hypermem-native-v3-reviewed-compatibility-policy-v1',
        'known_invalid_episode_replacement':True,
        'replacement_request_hash':proof['replacement']['request_hash'],
        'replacement_conversation':5,'replacement_logical_limit':1,
        'prospective_role_representation':prospective,
        'representation_kind':adapter.KIND,'representation_stage':'hypermem_episode_roles',
        'representation_scope':'Only the actual caller-local complete unique canonical UUID-to-native-alias bijection; only episode_id values may change; original validator for every other value; original HTTP bytes retained.',
        'per_conversation_complete_prefix_before_suffix':True,'global_runtime_prefix_barrier':False,
        'full_ten_offline_prefix_preflight_required':True,'other_old_raw_regeneration_permitted':False,
        'new_representation_generation_quota':0}


def check_preflight(value,proof):
    path=chain.previous.simple.safe(value['path']);require(binding(path)==value,'Full-prefix author evidence changed')
    report=read(path)
    require(report.get('schema')=='hypermem-native-v3-author-full-prefix-preflight-v1'
        and report.get('status')=='passed' and report.get('role')=='author_validation_not_independent_review'
        and report.get('production_launch_authorized') is False and report.get('actual_replacement_executed') is False
        and type(report.get('model_calls')) is int and report['model_calls']==0
        and type(report.get('credential_reads')) is int and report['credential_reads']==0
        and report.get('source_report')==proof['source_report']
        and report.get('hash_environment')==admission.hash_environment()
        and report.get('original_inventory_before_after_equal') is True and report.get('parent_inventory_before_after_equal') is True
        and report.get('qualified_role_boundaries_completed')==9 and report.get('stopped_before_replacement')==1
        and report.get('stopped_before_first_unseen_suffix')==9 and not any(report.get('blocked_operation_counts',{'missing':1}).values()),
        'All ten actual offline prefix boundaries were not validated')
    require([r['conversation'] for r in report['cases']]==list(range(10)),'Preflight cohort differs')
    required_code={str(ROOT/p):sha(ROOT/p) for p in ('tools/baselines_hypermem_native_v3.py',
        'tools/hypermem_prefix_api_v3.py','tools/audit_hypermem_terminal_v3.py',PREFLIGHT_HARNESS)}
    require(report['code_sha256']==required_code,'Preflight full candidate/harness code bindings differ')
    projection=report['projection'];require(binding(chain.previous.simple.safe(projection['path']))==projection,'Preflight projection changed')
    from jev_mem.sdk20_cached_api import digest
    require(read(projection['path'])=={'source_report':proof['source_report'],'normalized_value_sha256':digest(proof),
        'original_value_sha256':digest(admission.restore_cohort(proof)),
        'scope':'author_only_projection_not_production_admission'},'Preflight complete proof projection differs')
    for row,case in zip(report['cases'],proof['conversations']):
        ci=case['conversation'];n=len(case['prefix'])
        require(row['prefix_total']==n and row['inherited']==n-(ci==5)
            and row['source_messages_equal'] is True and row['episodes_equal'] is (ci!=5)
            and row['boundary']['kind']==('before_certified_replacement_transport' if ci==5 else 'after_original_prefix_before_first_unseen_transport'),
            'Actual preflight boundary differs')
        bank=path.parent/f'isolated_run/work/native/c{ci}'
        require(row['native_inventory']==admission.inventory(bank),'Preflight native evidence inventory changed')
        expected=[]
        for entry in case['prefix'][:n-(ci==5)]:
            native=bank/'calls'/Path(entry['native_call']['path']).name
            expected.append({'ordinal':entry['ordinal'],**binding(native)})
        require(row['native_records']==expected and row['completed_original_native_payloads_equal']==sum(e['native_status']=='success' for e in case['prefix'][:n-(ci==5)]),
            'Preflight native ordinal/count/binding differs')
    return report


def code_hashes(parent):
    return {**parent['code_sha256'],
            **{p:sha(ROOT/p) for p in CORE}}


def check_review(path,proof_path,proof,parent):
    path=chain.previous.simple.safe(path);review=read(path)
    require(review.get('schema')==REVIEW_SCHEMA and review.get('status')=='passed','Independent continuation review required')
    require(review.get('proof_sha256')==sha(proof_path) and review.get('specification')==parent['specification'], 'Review proof/full scientific specification differs')
    admission.check_hash_environment(review.get('hash_environment'))
    require(review.get('hash_environment')==admission.hash_environment(),'Explicit reviewed hash environment required')
    selected=review.get('continuation_policy',{})
    require(type(selected.get('prospective_role_representation')) is bool
        and selected==policy(proof,selected['prospective_role_representation']),'Review compatibility policy differs')
    preflight=review.get('prefix_preflight');require(isinstance(preflight,dict) and set(preflight)=={'path','sha256'},'Exact author full-prefix evidence required')
    check_preflight(preflight,proof)
    required=code_hashes(parent);code=review.get('code_sha256',{})
    require(all(code.get(p)==h for p,h in required.items()),'Review lacks current continuation code')
    for p,h in code.items():
        target=chain.previous.simple.safe(ROOT/p)
        require(target.is_relative_to(ROOT) and sha(target)==h,'Reviewed code changed/escaped workspace')
    evidence=review.get('evidence_sha256')
    require(isinstance(evidence,dict) and bool(evidence),'Independent supporting evidence bindings required')
    require(evidence.get(preflight['path'],evidence.get(str(Path(preflight['path']).relative_to(ROOT)).replace('\\','/')))==preflight['sha256'],
        'Review evidence does not include full-prefix validation')
    for p,h in evidence.items():
        target=chain.previous.simple.safe(ROOT/p)
        require(target.is_relative_to(ROOT) and sha(target)==h,'Independent supporting evidence changed/escaped workspace')
    require(review.get('upstream_sha256')==parent['upstream_sha256']==chain.base.source_inventory('hypermem'), 'Upstream review differs')
    require(all(parent['specification'].get(k)==v for k,v in adapter.describe().items()), 'Scientific adapter specification differs')
    return review


def expected_manifest(proof_path,proof,review_path):
    parent=read(OLD/'run_manifest.json');review=check_review(review_path,proof_path,proof,parent)
    m=deepcopy(parent)
    m.update(schema=SCHEMA,original_run=str(ANCESTOR),parent_run=str(OLD),new_run=str(NEW),original_manifest=binding(ANCESTOR/'run_manifest.json'),parent_manifest=binding(OLD/'run_manifest.json'),
             terminal_proof=binding(proof_path),review=binding(review_path),code_sha256=review['code_sha256'],
             historical_cache='Complete exact native-v2 per-conversation prefix and all-seen registry, including original ten boundary ancestors; only the certified c5 malformed JSON receives one exclusive replacement.',
             continuation_policy=review['continuation_policy'],
             hash_environment=review['hash_environment'],
             technical_compatibility=adapter.COMPATIBILITY,
             scientific_orchestration=chain.bridge_description(),
             ancestor_physical_usage=proof['original_usage'],parent_physical_usage=proof['new_usage'],
             lifecycle={'effective_result_root':str(NEW),'original_evidence_root':str(ANCESTOR),'parent_evidence_root':str(OLD),
                        'future_cleanup_roots':[str(ANCESTOR/'work'),str(OLD/'work'),str(NEW/'work')],
                        'cleanup_admitted':False,'note':'Separate three-root result/archive/cleanup review required; no earlier completion or lifecycle review grants cleanup.'})
    return m


def pristine(prepared):
    require(chain.previous.simple.safe(NEW)==NEW and chain.previous.simple.safe(OLD)==OLD,
            'Original/continuation path is aliased or contains a reparse ancestor')
    if not prepared:
        require(not NEW.exists(),'Fresh HyperMem continuation namespace required')
    else:
        require(NEW.is_dir() and sorted(p.name for p in NEW.iterdir())==['run_manifest.json'], 'Continuation already started or undeclared artifact present')


def prepare(review,proof_path=DEFAULT_PROOF):
    admission.check_hash_environment()
    pristine(False)
    proof_path=chain.previous.simple.safe(proof_path);review=chain.previous.simple.safe(review)
    proof=admission.validate_proof(proof_path)
    m=expected_manifest(proof_path,proof,review)
    chain.strong_predecessors('hypermem')
    pristine(False)
    save_new(NEW/'run_manifest.json',m)
    return {'status':'prepared','manifest':binding(NEW/'run_manifest.json'),'model_calls':0}


def manifest():
    admission.check_hash_environment()
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
        admission.check_hash_environment(self.m['hash_environment'])
        require(method=='hypermem' and not self.used,'Wrong/reused predecessor ticket')
        require(binding(NEW/'run_manifest.json')==self.manifest_binding,'Prepared manifest changed')
        require(binding(self.m['review']['path'])==self.m['review'] and binding(self.m['terminal_proof']['path'])==self.m['terminal_proof'],'Review/proof changed before execution')
        check_review(self.m['review']['path'],self.m['terminal_proof']['path'],self.proof,read(OLD/'run_manifest.json'))
        for p,h in {**self.m['code_sha256'],**self.m['upstream_sha256']}.items():require(sha(ROOT/p)==h,'Code changed before execution')
        expected={n:{k:binding(p) for k,p in pp.items()} for n,pp in chain.completion_paths('hypermem').items()}
        require(expected==self.completions,'Strong predecessor changed before ticket use')
        pristine(True);self.used=True
        save_new(NEW/'continuation_execution_intent.json',{'schema':'hypermem-native-v3-execution-intent-v1','created_utc':now(),'pid':os.getpid(),
            'manifest':self.manifest_binding,'terminal_proof':self.m['terminal_proof'],'review':self.m['review'],
            'predecessor_acceptance':{'schema':'runtime-predecessor-acceptance-record-v1',
                'scope':'Actual current strong ZIP, receipt, completion and deleted-work gates passed. Earlier archived experiments are not current model inputs or recursively rearchived dependencies.',
                'new_scientific_input':False,'recursive_rearchive_claimed':False,
                'verified_objects':{n:{k+'_sha256':r['sha256'] for k,r in rr.items()} for n,rr in self.completions.items()}}})


def private_controller(m,proof,ticket):
    path=ROOT/'tools/run_five_baselines_gemini25_v1.py'
    require(sha(path)==chain.FROZEN_BASE_SHA,'Frozen base changed')
    text=path.read_text(encoding='utf8')
    private=types.ModuleType('_hypermem_native_v3_private_controller');private.__file__=str(path)
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
    private.BaselineAPI=admitted_api(proof,NEW,manifest=binding(NEW/'run_manifest.json'),proof_binding=m['terminal_proof'],
        replacement_enabled=m['continuation_policy']['known_invalid_episode_replacement'],
        prospective_role_representation=m['continuation_policy']['prospective_role_representation'])
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
            value['usage']={'ancestor_original':checked['ancestor_usage'],'parent_native_v2':checked['parent_usage'],'continuation':checked['new_usage'],
                            'combined_physical':checked['combined_physical_usage'],
                            'cost_usd':None,'note':'Inherited exact caches have no new model usage. Missing per-field telemetry is unknown.'}
            value['technical_continuation']={'parent_manifest':m['parent_manifest'],'terminal_proof':m['terminal_proof'],
                'original_run':str(ANCESTOR),'parent_run':str(OLD),'effective_result_root':str(NEW),'compatibility':adapter.COMPATIBILITY,
                'continuation_policy':m['continuation_policy'],
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
    exec(compile(derived,'<hypermem-native-v3-exact-predecessor-bridge>','exec'),private.__dict__)
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
