"""Candidate evidence projection. No production authority or namespace creation."""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
import json

from tools import audit_hypermem_terminal_v3 as previous
from tools import audit_simplemem_terminal_v2 as common
from tools import baselines_hypermem_native_v4 as adapter
from jev_mem.sdk20_cached_api import digest

ROOT=Path(__file__).resolve().parents[1]
chain=previous.chain;OLD=previous.NEW;NEW=OLD.with_name('hypermem_native_v4')
ANCESTOR=previous.ANCESTOR;PARENT=previous.OLD
SCHEMA='hypermem-native-v4-terminal-admission-candidate-v1'
H=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0949'
FIXED={
 'cohort':(H/'native_v3_terminal_cohort_v1.json','deae91998b5823fa552abb9b199b7be0076f3afdb9b4ae4327715125f0fbc25a'),
 'usage_correction':(H/'native_v3_terminal_usage_correction_v1.json','f01ec07ab207b67e6d856836007fb5a81134d75208cc95899e33a8a7ac91ffa9'),
 'independent_review':(H/'independent_native_v3_terminal_review_v1.json','cda85cfe0f9dd5c82f9d7c835b027f06e50e08993a3c55260d84df3b95b07a6e'),
 'readiness':(H/'new_suffix_diagnosis_v1/plan_readiness_v1.json','867b56a2c33284177ae37ff51eef375710b9455d088bdc3bf1f60f069c66fa62'),
 'proposal':(H/'new_suffix_diagnosis_v1/minimal_continuation_proposal_v1.json','06540738a8cc0028743eb5f309756f1d95f06675c8ca2a05437c0a7723a8f6c8')}
REPLACEMENT_CONVERSATIONS=(2,3,6,9)
COMPATIBLE_CONVERSATIONS=(0,1,4,5,7,8)
CONSUMED_OLD_KEY='6758bd8d8dc914d7b627506008d436505d2cdcf597bf3ab15539eae1dd91fb6e'
require,read,sha,binding,inventory=common.require,common.read,common.file_sha,common.binding,common.inventory
usage_presence=previous.usage_presence
hash_environment=previous.hash_environment
check_hash_environment=previous.check_hash_environment
_active=previous.active_controllers
active_controllers=FunctionType(_active.__code__,{**_active.__globals__,
    'LAUNCHERS':(*_active.__globals__['LAUNCHERS'],'run_five_baselines_hypermem_native_v4')},_active.__name__,_active.__defaults__)
_stop=previous.assert_stopped
assert_stopped=FunctionType(_stop.__code__,{**_stop.__globals__,'active_controllers':active_controllers},_stop.__name__,_stop.__defaults__)

@contextmanager
def guard_context():
    """Exact current launcher allowance, restored even if a deep gate fails."""
    require(sha(chain.__file__)==previous.previous.CHAIN_SHA,'Frozen predecessor bridge differs')
    changes=((chain.guard,'active_controllers',active_controllers),
        (chain.guard,'assert_stopped',assert_stopped),(chain.previous.simple,'_stop',assert_stopped))
    before=[(m,k,getattr(m,k)) for m,k,_ in changes]
    try:
        for m,k,v in changes:setattr(m,k,v)
        yield
    finally:
        for m,k,v in reversed(before):setattr(m,k,v)

def fixed_bindings():return {k:{'path':str(p),'sha256':h} for k,(p,h) in FIXED.items()}

def exact(pathbind):require(binding(pathbind['path'])==pathbind,'Bound v4 evidence changed')

def normalize(cohort,correction,review,readiness,proposal):
    """Pure projection; keeps every original ordered row and full source object."""
    refs=fixed_bindings()
    require(cohort['status']=='passed' and cohort['unknown_or_inflight']==[] and cohort['before_equals_after'] is True,'Incomplete terminal accounting')
    require(review['status']=='passed' and review['base_report']==refs['cohort'] and review['required_usage_correction']==refs['usage_correction']
        and review['all_refusals_unwaived'] is True,'Complete terminal independent review is absent')
    require(correction['base_report']==refs['cohort'] and correction['all_other_usage_fields_equal'] is True
        and correction['corrected_usage']['combined']['cache_events']==6223,'Mandatory usage correction differs')
    require(readiness['inputs']['author_terminal_report']==refs['cohort'] and readiness['inputs']['original_proposal']==refs['proposal'], 'Plan readiness is for another source')
    require(proposal['production_prepare_authorized'] is False and proposal['model_request_authorized'] is False,'An offline proposal cannot confer execution authority')
    targets={x['conversation']:x for x in proposal['targets']}
    require(sorted(targets)==list(range(10)) and len(proposal['targets'])==10,'Exact target cohort differs')
    cases=[];compat={};replacements={};roles=[]
    for ci,case in enumerate(cohort['conversations']):
        require(case['conversation']==ci and len(case['native_calls'])>0,'Original native order differs')
        target=targets[ci];last=case['native_calls'][-1]
        require(last==case['failure'] and target['request_hash']==last['request_hash'] and target['signature']==last['signature']
            and target['context']==last['context'] and target['stage']==last['stage']
            and target['original_raw']['observed_sha256']==last['raw']['sha256'],'Target is not the exact terminal original request')
        prefix=[]
        for row in case['native_calls']:
            item=deepcopy(row);item['mode']='replace_once' if ci in REPLACEMENT_CONVERSATIONS and row is case['native_calls'][-1] else 'inherit'
            prefix.append(item)
        # Identity is not used to classify the last row after a JSON roundtrip.
        prefix[-1]['mode']='replace_once' if ci in REPLACEMENT_CONVERSATIONS else 'inherit'
        cert={'conversation':ci,'request_hash':last['request_hash'],'signature':last['signature'],'context':last['context'],
            'stage':last['stage'],'raw':last['raw'],'native_call':last['native_call'],'prefix_ordinal':last['ordinal'],
            'response_text_sha256':last['response_text_sha256'],'source_case_class':target['class']}
        if ci in REPLACEMENT_CONVERSATIONS:
            require(last['stage']=='hypermem_episode_roles' and last['parse_error'] is None,'Replacement kind differs')
            replacements[last['request_hash']]={**cert,'kind':'known_complete_invalid_role_uuid_set','retry_limit':1,
                'human_authorized':False}
        else:
            edits=[] if target['candidate_evidence'] is None else target['candidate_evidence']['edits']
            cert.update(kind='native_false_row_consumer' if ci in (0,7) else 'exact_syntax_representation',edits=edits)
            compat[last['request_hash']]=cert
        cases.append({'conversation':ci,'prefix':prefix,'source_messages':case['source_messages'],
            'source_identity':case['source_identity'],'source_conversation_sha256':case['source_conversation_sha256']})
    require(sum(len(c['prefix']) for c in cases)==6845 and sum(r['native_status']=='success' for c in cases for r in c['prefix'])==6835,'Complete native prefix differs')
    require(CONSUMED_OLD_KEY not in replacements and cohort['remaining_original_c5_replacement_quota']==0,'Consumed c5 quota cannot revive')
    registry=deepcopy(cohort['producer_registry'])
    require(registry[CONSUMED_OLD_KEY]['raw']['sha256']=='a22193622b6b9488de2d68f9186559da9c8f99523fa5ab041cabb2a0f2546393','Old successful replacement raw differs')
    return {'schema':SCHEMA,'status':'candidate_evidence_only','source_bindings':refs,
        'cohort_value_sha256':digest(cohort),'conversations':cases,'producer_registry':registry,
        'compatibility_cases':compat,'replacement_cases':replacements,'questions':cohort['questions'],
        'run_roots':cohort['run_roots'],'run_inventories':cohort['run_inventories'],
        'original_representation_auditor_refusals':deepcopy(cohort['original_representation_auditor_refusals']),
        'corrected_historical_usage':correction['corrected_usage'],'parent_manifest':cohort['manifest'],
        'role_cases':roles,'hash_environment':hash_environment(),
        'human_grant':None,'launch_admitted':False,'model_calls':0,'new_retry_authorized':False}

def collect(*,verify_inventories=True):
    """Read-only and research use only; no output or namespace is created."""
    assert_stopped();refs=fixed_bindings()
    for r in refs.values():exact(r)
    values={k:read(v['path']) for k,v in refs.items()}
    proof=normalize(values['cohort'],values['usage_correction'],values['independent_review'],values['readiness'],values['proposal'])
    proposal=values['proposal'];cohort=values['cohort']
    for target in proposal['targets']:
        ci=target['conversation'];key=target['request_hash']
        if ci in adapter.INSERTIONS:
            er=target['candidate_evidence']['report'];exact(er);sr=read(er['path'])
            row=sr if ci==5 else next(v for v in sr['rows'] if v['conversation']==ci)
            require(row['request_hash']==key and row['original_response_text_sha256']==proof['compatibility_cases'][key]['response_text_sha256'],'Exact syntax qualification differs')
            proof['compatibility_cases'][key].update(derived_parsed_sha256=row['derived_parsed_sha256'],syntax_evidence=er)
    for case in cohort['conversations']:
        for row in case['native_calls']:
            if row['native_status']!='success':continue
            native=read(row['native_call']['path'])
            if native['parsed_response']!=native['raw_response']:
                p=Path(row['native_call']['path']);jp=p.parent.parent/'representations'/p.name
                j=read(jp);require(sha(jp)==cohort['run_inventories']['current'][str(jp)],'Historical representation journal differs')
                proof['role_cases'].append({'conversation':case['conversation'],'request_hash':row['request_hash'],'raw':row['raw'],
                    'native_call':row['native_call'],'journal':binding(jp),'value':j})
    if verify_inventories:
        for name,root in proof['run_roots'].items():require(inventory(root)==proof['run_inventories'][name],'Terminal source inventory changed')
    for r in refs.values():exact(r)
    return proof

def validate_proof(path):
    value=read(path)
    require(value==collect(),'Candidate proof no longer equals complete fixed evidence')
    return value
