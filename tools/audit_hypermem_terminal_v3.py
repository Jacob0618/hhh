"""Exact all-ten native-v2 terminal evidence, normalized for a new namespace.

The immutable independent collector is executed privately and compared with its
published complete report. This proof is evidence admission, not permission to
send; code review, full-prefix preflight and the runner's one-use gate remain
separate requirements.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import audit_hypermem_terminal_v2 as previous
from tools import audit_simplemem_terminal_v2 as common
from tools.baselines_common_v1 import save_new

chain = previous.chain
ANCESTOR = previous.OLD
OLD = previous.NEW
NEW = OLD.with_name('hypermem_native_v3')
SCHEMA = 'hypermem-native-v3-terminal-admission-v1'
COLLECTOR = 'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/audit_hypermem_terminal_cohort_v1.py'
COLLECTOR_SHA = '3a21f31e19d430dd64f28050405294144164ca4b997903ee7fc4706ffcad6f28'
SOURCE_REPORT = 'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/terminal_cohort_v1.json'
SOURCE_REPORT_SHA = '3762f95f51e65bba3778798a5d926ba7603c43a0164a6af8fa48ef3b7cd50b47'
ROLE_REVIEW = 'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/independent_episode_role_bijection_v1.json'
ROLE_REVIEW_SHA = 'c9bfcfbdcdba7b516aa89c28907863205a1e210e1ad3491cee32967e762f4f18'
SEED_REPORT='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/hash_seed_candidate_v1/report.json'
SEED_REPORT_SHA='fbe12d36872c79e2e80dd3bb9dd86c20a50d6eb5f0ec04c2a0e363173e3e5f7a'
SEED_REVIEW='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/independent_seed_candidate_v1.json'
SEED_REVIEW_SHA='28b88cae8a05a4d9c8963181e4d7a9bc1f4dbd1723cf665cc56142e9e51070bb'
require,read,sha,inventory,binding = common.require,common.read,common.file_sha,common.inventory,common.binding
usage_presence = previous.usage_presence


def hash_environment():
    path=chain.previous.simple.safe(ROOT/SEED_REPORT)
    require(sha(path)==SEED_REPORT_SHA,'Observed-set seed candidate evidence changed')
    report=read(path);selected=report['selected']
    require(report['status']=='candidate_found_not_launch_admitted' and report['old_seed_recovered'] is False
        and report['selection_used_gold_or_qa'] is False and report['production_launch_authorized'] is False
        and report['model_calls']==0 and report['credential_reads']==0
        and selected['matches'] is True and selected['failures']==[] and selected['seed']=='780',
        'Seed evidence scope differs')
    independent=chain.previous.simple.safe(ROOT/SEED_REVIEW)
    require(sha(independent)==SEED_REVIEW_SHA,'Independent observed-set qualification changed')
    review=read(independent)
    require(review['schema']=='independent-hypermem-observed-hash-seed-qualification-v1'
        and review['status']=='passed_observed_representation_only' and review['candidate_report']==binding(path)
        and review['runtime']==selected and review['base_interpreter']==report['base_interpreter']
        and review['source_hashes_before_after_equal'] is True and review['full_prefix_replay_completed'] is False
        and review['production_launch_authorized'] is False,'Independent seed scope differs')
    return {'seed':selected['seed'],'qualification':binding(independent),'candidate_report':binding(path),'base_interpreter':report['base_interpreter'],
        'fingerprint':{k:selected[k] for k in ('implementation','version','cache_tag','sentinel')},
        'scope':'Observed native unordered-list representation matches; full-prefix replay is separately mandatory. Original random seed is unknown; no gold or QA used.'}


def check_hash_environment(expected=None):
    contract=hash_environment()
    require(expected is None or expected==contract,'Reviewed hash environment differs')
    require(os.environ.get('PYTHONHASHSEED')==contract['seed'],'Hash seed must be set before Python process startup')
    actual={'implementation':platform.python_implementation(),'version':platform.python_version(),
        'cache_tag':sys.implementation.cache_tag,'sentinel':{s:hash(s) for s in contract['fingerprint']['sentinel']}}
    require(actual==contract['fingerprint'] and binding(chain.previous.simple.safe(sys._base_executable))==contract['base_interpreter'],
        'Actual interpreter/hash sentinel differs; changing environment after startup is insufficient')
    return contract
_active = previous.active_controllers
active_controllers = FunctionType(_active.__code__,{**_active.__globals__,
    'LAUNCHERS':(*_active.__globals__['LAUNCHERS'],'run_five_baselines_hypermem_native_v3')},
    'active_controllers',_active.__defaults__)
_stop = previous.assert_stopped
assert_stopped = FunctionType(_stop.__code__,{**_stop.__globals__,'active_controllers':active_controllers},
    'assert_stopped',_stop.__defaults__)


@contextmanager
def guard_context():
    require(sha(chain.__file__) == previous.CHAIN_SHA,'Frozen predecessor bridge differs')
    changes=((chain.guard,'active_controllers',active_controllers),
        (chain.guard,'assert_stopped',assert_stopped),(chain.previous.simple,'_stop',assert_stopped))
    before=[(m,k,getattr(m,k)) for m,k,_ in changes]
    try:
        for m,k,v in changes:setattr(m,k,v)
        yield
    finally:
        for m,k,v in reversed(before):setattr(m,k,v)


def locked_collector():
    path=ROOT/COLLECTOR
    require(sha(path)==COLLECTOR_SHA,'Independent complete terminal collector changed')
    module=ModuleType('_hypermem_v3_qualified_terminal_collector')
    module.__file__=str(path)
    exec(compile(path.read_text('utf8'),str(path),'exec'),module.__dict__)
    # Only the existing controller probe's exact new self-parent allowance is
    # supplied. All inventory/producer/native/schema code is unchanged.
    module.admission=SimpleNamespace(**{**vars(previous),'assert_stopped':assert_stopped})
    return module


def normalize(result):
    """Consumes one newly collected object, avoiding a second 118-MB copy."""
    require(result['schema']=='hypermem-native-v2-independent-terminal-cohort-v1'
        and result['status']=='passed' and result['launch_admitted'] is False
        and result['cleanup_authorized'] is False and result['unknown_or_inflight']==[]
        and result['before_equals_after'] is True,'Unqualified terminal cohort')
    require(result['manifest']['path']==str(OLD/'run_manifest.json')
        and result['parent_root']==str(ANCESTOR) and result['run_root']==str(OLD), 'Wrong historical roots')
    require(result['role_qualification']=={'path':str(ROOT/ROLE_REVIEW),'sha256':ROLE_REVIEW_SHA}
        and sha(ROOT/ROLE_REVIEW)==ROLE_REVIEW_SHA,'Role classification changed')
    role_report=read(ROOT/ROLE_REVIEW)
    require(role_report['status']=='passed' and role_report['production_resume_authorized'] is False
        and role_report['normalization_implemented_in_production'] is False,'Role report is not the qualified individual evidence')
    cases=role_report['rows']
    require(len(cases)==9 and {r['conversation'] for r in cases}==set(range(10))-{5},'Exact nine role cases required')
    require(result['questions']==chain.base.questions() and len(result['questions'])==1540,'Fixed QA cohort differs')
    require([r['conversation'] for r in result['conversations']]==list(range(10)),'Ten ordered conversations required')
    replacement=None
    for conv in result['conversations']:
        ci=conv['conversation']; entries=conv['prefix']; failure=conv['failure']
        require(entries and entries[-1]['request_hash']==failure['request_hash']
            and all(e['ordinal']==i+1 for i,e in enumerate(entries)), 'Original prefix/failure boundary differs')
        for entry in entries:
            entry['source_mode']=entry['mode']
            entry['mode']='inherit'
        if ci==5:
            cert=failure['certificate']
            require(cert['kind']=='known_complete_invalid_episode_json' and type(cert['retry_limit']) is int
                and cert['retry_limit']==1 and cert['retry_permission_active'] is False,
                'Wrong malformed JSON candidate qualification')
            entries[-1]['mode']='replace_once'
            replacement={**cert, **{k:failure[k] for k in ('conversation','request_hash','signature','context','stage','raw','native_call')},
                'prefix_ordinal':len(entries),'response_text_sha256':failure['response_text_sha256']}
        else:
            case=next(r for r in cases if r['conversation']==ci)
            require(case['request_hash']==failure['request_hash'] and case['raw']==failure['raw']
                and case['native_call']==failure['native_call'],'Role qualification does not bind the exact last failed request')
    require(replacement is not None,'Missing single replacement boundary')
    result.update(schema=SCHEMA,source_scope=result['scope'],source_auditor=result['auditor'],scope='complete_original_evidence_and_explicit_candidate_boundaries',
        source_report={'path':str(ROOT/SOURCE_REPORT),'sha256':SOURCE_REPORT_SHA},
        collector={'path':str(ROOT/COLLECTOR),'sha256':COLLECTOR_SHA},auditor=binding(Path(__file__)),
        original_run=str(ANCESTOR),parent_run=str(OLD),new_run=str(NEW),
        role_cases=cases,replacement=replacement,
        prospective_role_representation_enabled=False,
        new_generation_replacement_permission_enabled=False)
    # These false flags remain evidence semantics. Only an exact implementation
    # review and its run manifest may explicitly enable either runtime policy.
    return result


def restore_cohort(proof):
    """Restore the published collector value without retaining another full copy.

    Outer objects and changed prefix entries are shallow-copied; immutable large
    signatures/registries remain shared. The published report is also an exact
    independently bound artifact in the archive closure.
    """
    value=dict(proof)
    value['schema']='hypermem-native-v2-independent-terminal-cohort-v1'
    value['scope']=value.pop('source_scope')
    value['auditor']=value.pop('source_auditor')
    for name in ('source_report','collector','original_run','parent_run','new_run',
                 'role_cases','replacement','prospective_role_representation_enabled',
                 'new_generation_replacement_permission_enabled'):
        value.pop(name)
    value['conversations']=[]
    for conv in proof['conversations']:
        c=dict(conv);c['prefix']=[]
        for entry in conv['prefix']:
            e=dict(entry);e['mode']=e.pop('source_mode');c['prefix'].append(e)
        value['conversations'].append(c)
    return value


def collect(progress=None):
    assert_stopped()
    require(sha(ROOT/SOURCE_REPORT)==SOURCE_REPORT_SHA,'Published full terminal report changed')
    expected=read(ROOT/SOURCE_REPORT)
    current=locked_collector().collect(progress=progress)
    require(current==expected,'Current complete audit differs from immutable independent terminal report')
    del expected
    value=normalize(current)
    assert_stopped()
    return value


def validate_proof(path):
    path=chain.previous.simple.safe(path)
    value=read(path)
    require(value==collect(),'Terminal admission is not the complete current qualified audit')
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('create','verify'))
    parser.add_argument('--proof',type=Path,required=True)
    args=parser.parse_args()
    target=chain.previous.simple.safe(args.proof)
    require(target.is_relative_to(ROOT/'research/baseline_execution_2026-10-08'), 'Research-only proof publication required')
    with guard_context():
        if args.command=='create':
            require(not target.exists(),'Proof publication is append-only')
            print(json.dumps(save_new(target,collect(progress=lambda x:print(json.dumps(x),flush=True)))))
        else:
            validate_proof(target)
            print(json.dumps({'status':'passed','proof':binding(target),'launch_admitted':False}))


if __name__=='__main__':main()
