"""Unactivated v4 candidate. No production action is currently authorized.

The fixed root-published human grant file is absent. prepare/check/run reject before any
directory, intent, status, provider or credential operation. Research fixtures
may inspect the candidate manifest and unchanged private scientific controller.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import types

from tools import run_five_baselines_hypermem_native_v3 as original
from tools import audit_hypermem_terminal_v4 as admission
from tools import audit_hypermem_continuation_v4 as audit
from tools import baselines_hypermem_native_v4 as adapter
from tools import hypermem_prefix_api_v4 as prefix
from tools.baselines_common_v1 import BaselineAPI,read,sha,save_new,now,error_record

ROOT=admission.ROOT;chain=admission.chain
ANCESTOR,PARENT,OLD,NEW=admission.ANCESTOR,admission.PARENT,admission.OLD,admission.NEW
ENTRY='tools/run_five_baselines_hypermem_native_v4.py'
CORE=(ENTRY,'tools/baselines_hypermem_native_v4.py','tools/hypermem_prefix_api_v4.py',
    'tools/audit_hypermem_terminal_v4.py','tools/audit_hypermem_continuation_v4.py',
    'research/sources/locomo_evaluation_cbfbc1.py',
    'research/baseline_execution_2026-10-08/heartbeat_20261009T1152/launch_hypermem_native_v4.ps1')
SCHEMA='hypermem-native-v4-continuation-candidate-manifest-v1'
REVIEW_SCHEMA='hypermem-native-v4-independent-candidate-review-v1'
DEFAULT_PROOF=ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v4/terminal_candidate_v1.json'
PREFLIGHT_HARNESS='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_native_v4_prefix_candidate_v1.py'
require=admission.require;binding=admission.binding
admitted_api=prefix.admitted_api
_OWNED_EXECUTION=False
_RUN_STATE={'owned':False}

def require_activation(grant,proof=None):
    # This is deliberately before even reading the production manifest.
    prefix.require_human_grant(grant,proof)
    return binding(prefix.HUMAN_GRANT_PATH)

def policy(proof):
    return {'schema':'hypermem-native-v4-exact-candidate-policy-v1','source_bindings':proof['source_bindings'],
        'six_exact_response_keys':sorted(proof['compatibility_cases']),
        'four_proposed_replacement_keys':sorted(proof['replacement_cases']),
        'logical_limit_per_replacement_key':1,'human_authorization_required':True,
        'old_consumed_key':admission.CONSUMED_OLD_KEY,'old_consumed_key_permission':False,
        'all_original_success_payloads_preserved':6835,'whole_original_ordered_prefix':6845,
        'prospective_role_representation':'unchanged_v3_complete_actual_local_bijection_only',
        'future_unknown_compatibility_expansion':False,'qa':'unchanged_original_1540_once_uniform_recovery',
        'cleanup_or_downstream_authorized':False}

def code_hashes(parent):
    return {**parent['code_sha256'],**{p:sha(ROOT/p) for p in (*CORE,PREFLIGHT_HARNESS)}}

def check_preflight(ref,proof):
    admission.exact(ref);r=read(ref['path'])
    require(r.get('schema')=='hypermem-native-v4-author-zero-api-prefix-v1' and r.get('status')=='passed'
        and r.get('source_bindings')==proof['source_bindings'] and r.get('hash_environment')==admission.hash_environment()
        and r.get('inherited_successful_payloads_equal')==6835 and r.get('four_replacements_not_sent')==4
        and r.get('six_compatibility_boundaries_completed')==6 and r.get('six_stopped_before_first_unseen_request')==6
        and r.get('production_launch_authorized') is False and r.get('all_three_original_inventories_unchanged') is True,
        'Actual full-prefix research evidence is incomplete')
    require(all(type(r.get(k)) is int and r[k]==0 for k in ('model_calls','credential_reads','network_attempts')),'Preflight was not zero-provider')
    expected={str(ROOT/p):sha(ROOT/p) for p in (*CORE,PREFLIGHT_HARNESS)}
    require(r.get('code_sha256')==expected,'Preflight candidate/harness code changed')
    parent=read(OLD/'run_manifest.json')
    dependencies={str(ROOT/p):h for p,h in {**parent['code_sha256'],**parent['upstream_sha256']}.items()}
    require(r.get('frozen_dependencies')=={'parent_manifest':binding(OLD/'run_manifest.json'),'files_sha256':dependencies},
        'Preflight original scientific dependency table differs')
    require([x['conversation'] for x in r['cases']]==list(range(10)),'Actual preflight cohort differs')
    for row,case in zip(r['cases'],proof['conversations']):
        ci=case['conversation'];stop=ci in admission.REPLACEMENT_CONVERSATIONS
        require(row['inherited']==len(case['prefix'])-int(stop)
            and row['stop_kind']==('before_exact_ungranted_replacement' if stop else 'after_exact_prefix_before_unseen')
            and row['successful_payloads_equal']==sum(e['native_status']=='success' for e in case['prefix']), 'Preflight original source boundary differs')
        bank=Path(ref['path']).parent/'isolated_run/work/native'/f'c{ci}'
        require(admission.inventory(bank)==row['native_inventory'],'Preflight actual native artifacts changed')
    return r

def check_review(path,proof_path,proof,parent):
    r=read(path)
    require(r.get('schema')==REVIEW_SCHEMA and r.get('status')=='passed_implementation_and_zero_api_preflight_only'
        and r.get('production_launch_authorized') is False,'Independent candidate review required; it is not human authorization')
    require(r.get('proof_sha256')==sha(proof_path) and r.get('source_bindings')==proof['source_bindings']
        and r.get('specification')==parent['specification'] and r.get('continuation_policy')==policy(proof),'Reviewed candidate scope differs')
    require(r.get('code_sha256')==code_hashes(parent) and r.get('upstream_sha256')==parent['upstream_sha256']==chain.base.source_inventory('hypermem'),'Candidate scientific/code binding differs')
    for p,h in {**r['code_sha256'],**r['upstream_sha256'],**r['evidence_sha256']}.items():
        pp=chain.previous.simple.safe(ROOT/p);require(pp.is_relative_to(ROOT) and sha(pp)==h,'Reviewed source/evidence changed')
    admission.check_hash_environment(r.get('hash_environment'))
    check_preflight(r['prefix_preflight'],proof)
    require(r['evidence_sha256'].get(r['prefix_preflight']['path'])==r['prefix_preflight']['sha256'],'Review omits actual preflight evidence')
    return r

def expected_manifest(proof_path,proof,review_path,*,human_grant=None):
    parent=read(OLD/'run_manifest.json');r=check_review(review_path,proof_path,proof,parent)
    m=deepcopy(parent)
    m.update(schema=SCHEMA,original_run=str(ANCESTOR),parent_run=str(OLD),new_run=str(NEW),
        original_manifest=binding(ANCESTOR/'run_manifest.json'),parent_manifest=binding(OLD/'run_manifest.json'),
        terminal_proof=binding(proof_path),review=binding(review_path),source_bindings=proof['source_bindings'],
        code_sha256=r['code_sha256'],continuation_policy=policy(proof),human_grant=human_grant,
        hash_environment=r['hash_environment'],technical_compatibility=adapter.COMPATIBILITY,
        historical_usage=proof['corrected_historical_usage'],historical_cache='All6845 current ordered native requests;6835 successful payloads unchanged; exact raw inheritance including consumed oldc5 successful raw; four future replacement keys require separate human grant.',
        lifecycle={'cleanup_admitted':False,'downstream_admitted':False,'note':'No four-root lifecycle implementation or actual result acceptance is supplied by this candidate.'})
    return m

def pristine(prepared):
    require(chain.previous.simple.safe(NEW)==NEW,'Continuation root is aliased or reparsed')
    require((NEW.is_dir() and sorted(p.name for p in NEW.iterdir())==['run_manifest.json']) if prepared else not NEW.exists(),'Fresh unstarted candidate namespace required')

def prepare(review,proof_path=DEFAULT_PROOF,*,human_grant=None):
    human_grant=require_activation(human_grant);admission.assert_stopped();pristine(False)
    proof=admission.validate_proof(proof_path);require_activation(human_grant,proof)
    m=expected_manifest(proof_path,proof,review,human_grant=human_grant)
    chain.strong_predecessors('hypermem');pristine(False)
    save_new(NEW/'run_manifest.json',m)
    return {'status':'prepared','manifest':binding(NEW/'run_manifest.json'),'model_calls':0}

def manifest(*,human_grant=None):
    human_grant=require_activation(human_grant);admission.assert_stopped()
    m=read(NEW/'run_manifest.json');require(m['human_grant']==human_grant,'Prepared human authorization differs')
    proof=admission.validate_proof(m['terminal_proof']['path']);require_activation(human_grant,proof)
    require(m==expected_manifest(m['terminal_proof']['path'],proof,m['review']['path'],human_grant=human_grant),'Candidate manifest differs')
    return m,proof

class GateTicket:
    def __init__(self,m,proof,completions):
        self.used=False;self.m=m;self.proof=proof;self.completions=completions;self.manifest_binding=binding(NEW/'run_manifest.json')
    def consume(self,method):
        require_activation(self.m['human_grant'],self.proof)
        require(method=='hypermem' and not self.used,'Wrong/reused continuation ticket')
        require(binding(NEW/'run_manifest.json')==self.manifest_binding,'Manifest changed before first request')
        check_review(self.m['review']['path'],self.m['terminal_proof']['path'],self.proof,read(OLD/'run_manifest.json'))
        admission.check_hash_environment(self.m['hash_environment'])
        require({n:{k:binding(p) for k,p in pp.items()} for n,pp in chain.completion_paths('hypermem').items()}==self.completions,'Accepted predecessor changed')
        pristine(True);self.used=True
        save_new(NEW/'continuation_execution_intent.json',{'schema':'hypermem-native-v4-execution-intent-v1',
            'created_utc':now(),'pid':os.getpid(),'manifest':self.manifest_binding,'terminal_proof':self.m['terminal_proof'],
            'review':self.m['review'],'human_grant':self.m['human_grant'],
            'predecessor_acceptance':{'schema':'runtime-predecessor-acceptance-record-v1',
                'scope':'Actual current strong ZIP, receipt, completion and deleted-work gates passed. Earlier archived experiments are not current model inputs or recursively rearchived dependencies.',
                'new_scientific_input':False,'recursive_rearchive_claimed':False,
                'verified_objects':{n:{k+'_sha256':r['sha256'] for k,r in rr.items()} for n,rr in self.completions.items()}}})

# Private scientific controller function: same original run/qa_one/recovery/
# seal functions and the already reviewed predecessor bridge. Only namespace,
# adapter/API and evidence audit/summary fields are supplied here.
_private_source=textwrap.dedent(inspect.getsource(original.private_controller)).replace('_hypermem_native_v3_private_controller','_hypermem_native_v4_private_controller')
require(_private_source.count("_OWNED_EXECUTION=True")==1,'Frozen execution ownership write changed')
_private_source=_private_source.replace('_OWNED_EXECUTION=True',"_RUN_STATE['owned']=True")
_a=_private_source.index('    private.BaselineAPI=admitted_api(');_b=_private_source.index('    latest_audit={}',_a)
_private_source=_private_source[:_a]+"    private.BaselineAPI=admitted_api(proof,NEW,manifest=binding(NEW/'run_manifest.json'),proof_binding=m['terminal_proof'],human_grant=m['human_grant'])\n"+_private_source[_b:]
_a=_private_source.index("            value['usage']={");_b=_private_source.index("            value['technical_continuation']",_a)
_private_source=_private_source[:_a]+"            value['usage']={'historical':checked['historical_usage'],'continuation':checked['new_usage'],'combined_physical':checked['combined_physical_usage'],'cost_usd':None,'note':'Missing telemetry remains unknown; caches are not new model usage.'}\n"+_private_source[_b:]
_ns={**original.__dict__,**globals()};exec(compile(_private_source,'<candidate-v4-private-original-scientific-controller>','exec'),_ns);private_controller=_ns['private_controller']

def run(*,human_grant=None):
    human_grant=require_activation(human_grant)
    require(chain.base.BaselineAPI is BaselineAPI,'Shared provider identity changed before credentials')
    m,proof=manifest(human_grant=human_grant);pristine(True)
    ticket=GateTicket(m,proof,chain.strong_predecessors('hypermem'))
    return private_controller(m,proof,ticket).run('hypermem')

def main():
    _RUN_STATE['owned']=False
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check','run'))
    p.add_argument('--review',type=Path);p.add_argument('--proof',type=Path,default=DEFAULT_PROOF)
    args=p.parse_args()
    # Exact same early barrier for every production CLI branch.
    grant=require_activation(None)
    if args.command=='prepare' and args.review is None:p.error('--review required')
    with admission.guard_context():
        try:
            if args.command=='prepare':result=prepare(args.review,args.proof,human_grant=grant)
            elif args.command=='check':
                manifest(human_grant=grant);result={'status':'passed','manifest':binding(NEW/'run_manifest.json')}
            else:result=run(human_grant=grant)
        except BaseException as exc:
            if args.command=='run' and _RUN_STATE['owned']:chain.base.status(NEW,'needs_review',error=error_record(exc))
            raise
    print(json.dumps(result))

if __name__=='__main__':main()
