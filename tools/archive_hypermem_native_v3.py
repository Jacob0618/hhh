"""Three-root HyperMem lifecycle; separate from production continuation authority.

The reviewed v2 lifecycle is loaded into a private namespace. Generic archive,
CRC, blob, path, keep, deletion receipt and completion gates are reused there.
Only current-result routing, ancestor replay and the v3 native audit are new.
No operation is admitted without the separate exact implementation/result/bank
reviews. Importing this candidate cannot produce a plan or grant completion.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from copy import deepcopy
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.baselines_common_v1 import read,sha,save_new,now
from tools import archive_hypermem_native_v2 as frozen

FROZEN_ENGINE='tools/archive_hypermem_native_v2.py'
FROZEN_ENGINE_SHA='76da29410880a70a5e2b90316659ade665470d78df8942f872c9e05ccb0501a3'
COLLECTOR='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/audit_hypermem_terminal_cohort_v1.py'
COLLECTOR_SHA='3a21f31e19d430dd64f28050405294144164ca4b997903ee7fc4706ffcad6f28'
COHORT_REVIEW='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/terminal_cohort_v1.json'
COHORT_REVIEW_SHA='3762f95f51e65bba3778798a5d926ba7603c43a0164a6af8fa48ef3b7cd50b47'
FAILED_PREFIX_REPORT='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/hypermem_native_v3_prefix_author_v1/report.json'
FAILED_PREFIX_REPORT_SHA='da914f2cf61bd8d96a71f9bd1bda0e0ab449eeccbccfc21b7e966f9da550e338'
FAILED_SOURCE_LINEAGE='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/hypermem_native_v3_prefix_author_v1/code_snapshot_lineage_v1.json'
FAILED_SOURCE_LINEAGE_SHA='ae75865f7b0e5cb7997cbc83bd585bb2422ee5c69d7a231f59af7150beac93bb'
OBSERVER_REPORT='research/baseline_execution_2026-10-08/launch_hypermem_native_v2/health_v1/native_snapshot05.json'
OBSERVER_REPORT_SHA='5ff77ccd15e49f6193d548d6c664eb4600ae9d5fd5a87015173e47ba62bd08f2'
OBSERVER_LINEAGE='research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_observer_capsules_v1/lineage.json'
OBSERVER_LINEAGE_SHA='47160707f1563d7cf21dad069e5e04e6a2eb8878de733c37367e81a517e430b7'
OBSERVER_REGISTRIES={
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_observer_capsules_v4/native_snapshot05_5ff77ccd15e4/registry.json':
        '735f77397c48efa638a60509bfde1ab75a1ba942812e08e251b56eb8a57609e6',
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_observer_capsules_v4/native_snapshot04_c8edf6e33cf8/registry.json':
        'c861bf2597e3d69595fa9ef65cd3d314136705d7a80efcfc3cec4190ec70e6c6',
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_observer_capsules_v4/heartbeat0949_native01_0d676a78a932/registry.json':
        '80586e60b6f897a9f4f3d303f62be91b3c0cd4a02a94834ec0e0a17cec248c70',
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_observer_capsules_v4/heartbeat0949_native02_5cc0a46f5f97/registry.json':
        'f2a95562ba0bcf1b5409e3e34a0add902b4fec61a363537a9fb6dffb3be14198',
}
STARTED_HARNESS='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_historical_started_capsules_v1.py'
STARTED_HARNESS_SHA='a02730023e54f1f302649cde3baf7839cd81f8d3ac7bd45d06565ed0419ce036'
STARTED_REGISTRIES={
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_started_capsules_v1/native_snapshot05_5ff77ccd15e4/registry.json':
        'ba0964822ff0129928a42e13057572a71a406a3360613e8ca17292103dbb8529',
    'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_started_capsules_v1/native_snapshot04_c8edf6e33cf8/registry.json':
        '93ed3bd7b9891751e71c60d9b66be4960d8fd3d6f97cf33c6fec0194be4b3285',
}
STATUS_HARNESS='research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_historical_status_capsules_v1.py'
STATUS_HARNESS_SHA='e9d74ab0860d1bfb79ef81cbf593a745981d6afc6add19391017aff9fe6f9913'
STATUS_REGISTRY='research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_status_capsules_v1/native_snapshot05_5ff77ccd15e4/registry.json'
STATUS_REGISTRY_SHA='1cff5255ea155213f9409dce4cad1863224239d90d62af6473b11567662c1d27'
STATUS_VERSIONS={
    OBSERVER_REPORT:(OBSERVER_REPORT_SHA,STATUS_REGISTRY,STATUS_REGISTRY_SHA,STATUS_HARNESS,STATUS_HARNESS_SHA),
    'research/baseline_execution_2026-10-08/launch_hypermem_native_v2/health_v1/native_snapshot04.json':(
        'c8edf6e33cf861b1f958bc1bc12dc402d6d8ba15734e09e9b242c40a753b5b3a',
        'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_status_capsules_v2/native_snapshot04_c8edf6e33cf8/registry.json',
        'f4ce834da8da6fa1b21ba8335cdd397edb44b92ed39c1c20eeb84836e3f870a3',
        'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_historical_status_capsules_v2.py',
        '77d512e34a6531d55a05899154b8656251bae2977fbb7428659fef1aca6ec616'),
    'research/baseline_execution_2026-10-08/launch_hypermem_native_v2/health_v1/heartbeat0949_native01.json':(
        '0d676a78a9325831a4dae3e33026cbc7c88df91d8a9cb97ab5fe6d32da8c1c45',
        'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_status_capsules_v3/heartbeat0949_native01_0d676a78a932/registry.json',
        'cb3dcecce01706cd4afc51c62f3a5244a74dabe6a76363b8bbe84ba07e4bb5ce',
        'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_historical_status_capsules_v3.py',
        '40abc01c97f74fa8cc55fcc3c53dd097949f832bbcc4aae5c93ba2e3934d1335'),
    'research/baseline_execution_2026-10-08/launch_hypermem_native_v2/health_v1/heartbeat0949_native02.json':(
        '5cc0a46f5f97248f8d0ce7da0ae784614c453c15bb4147b18793f262879dda7e',
        'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_status_capsules_v3/heartbeat0949_native02_5cc0a46f5f97/registry.json',
        '0f1ac97bd1b478252db39c3233847b9f8378959bab2176d24158ed837086ca46',
        'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_historical_status_capsules_v3.py',
        '40abc01c97f74fa8cc55fcc3c53dd097949f832bbcc4aae5c93ba2e3934d1335'),
}
EMPTY_SEAL_LINEAGE='research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle/historical_empty_seal_v1/lineage.json'
EMPTY_SEAL_LINEAGE_SHA='c7830cf0ec4336014f3a1f9fdc7e5633d7f8705f7434c296d51a610304c1a0c4'
EMPTY_SEAL_OBSERVER='research/baseline_execution_2026-10-08/launch_hypermem_native_v2/health_v1/native_snapshot04.json'
EMPTY_SEAL_OBSERVER_SHA='c8edf6e33cf861b1f958bc1bc12dc402d6d8ba15734e09e9b242c40a753b5b3a'
EMPTY_SEAL_SOURCE='runs/five_baselines_gemini25_v1_20261008/hypermem_native_v2/work/native/c8/calls/.integrity/episode_000014_hypermem_boundary.json.sha256'
ROOTS_BASE=ROOT/'runs/five_baselines_gemini25_v1_20261008'
OLD=ROOTS_BASE/'hypermem'
PARENT=ROOTS_BASE/'hypermem_native_v2'
NEW=ROOTS_BASE/'hypermem_native_v3'
OUTPUT=ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v3_lifecycle'
DEFAULT_PLAN=OUTPUT/'cleanup_plan.json'
SCHEMA='hypermem-native-v3-three-root-archive-v1'
PLAN_SCHEMA='hypermem-native-v3-three-root-cleanup-plan-v1'
REVIEW_SCHEMA='hypermem-native-v3-lifecycle-implementation-review-v1'
RESULT_SCHEMA='hypermem-native-v3-result-review-v1'
CERT_SCHEMA='hypermem-native-v3-predelete-bank-validation-v1'
BANK_REVIEW_SCHEMA='hypermem-native-v3-bank-validation-review-v1'
SCOPE=frozen.SCOPE+'; all three historical roots, reversible role-ID journals and the exact single malformed-JSON replacement are retained and replayed'
EXTRA=('tools/archive_hypermem_native_v3.py','tools/cleanup_hypermem_native_v3.ps1','tools/run_five_baselines_reviewed_chain_v8.py')
V3_CORE=('tools/baselines_hypermem_native_v3.py','tools/hypermem_prefix_api_v3.py',
    'tools/run_five_baselines_hypermem_native_v3.py','tools/audit_hypermem_terminal_v3.py','tools/audit_hypermem_continuation_v3.py')
OUTPUTS=frozen.OUTPUTS
INDEXES=frozen.INDEXES
require=frozen.require
digest=frozen.digest
_READ_CONTEXT=ContextVar('hypermem_v3_lifecycle_read_context',default=None)

require(sha(ROOT/FROZEN_ENGINE)==FROZEN_ENGINE_SHA,'Reviewed two-root lifecycle changed')
_source=(ROOT/FROZEN_ENGINE).read_text(encoding='utf8')
_engine=ModuleType('_hypermem_v3_private_lifecycle')
_engine.__file__=str(ROOT/FROZEN_ENGINE)
# These replacements change lifecycle labels only, never scientific code,
# native prompts, response parsing, output scores, or transport behavior.
_labels={'hypermem-native-v2':'hypermem-native-v3','two-root':'three-root','two_root':'three_root'}
_derived=_source
for _old,_new in _labels.items():_derived=_derived.replace(_old,_new)
exec(compile(_derived,'<private-reviewed-hypermem-three-root-engine>','exec'),_engine.__dict__)
ENGINE_DERIVATION={'source_sha256':FROZEN_ENGINE_SHA,
    'derived_source_sha256':hashlib.sha256(_derived.encode()).hexdigest(),'label_replacements':_labels,
    'scope':'Private generic lifecycle reuse; explicitly overridden v3 admission/audit routes are declared separately.'}


def roots():return {'original':OLD,'parent':PARENT,'continuation':NEW}
def work_roots():return [r/'work' for r in roots().values()]
def safe(path):return frozen.single.clone(frozen.single.safe_path,ROOT=ROOT)(path)
def binding(path):
    p=safe(path);return {'path':str(p),'sha256':sha(p)}
def bound(record):
    p=safe(record['path']);require(sha(p)==record['sha256'],'Bound evidence changed: '+str(p));return p
def inventory(root):return frozen.single.clone(frozen.single.inventory,ROOT=ROOT,safe_path=safe)(root)
def inventories():return {k:inventory(r) for k,r in roots().items()}
def hashes(inv):return {p:r['sha256'] for p,r in inv.items()}


_active=frozen.active_controllers
active_controllers=FunctionType(_active.__code__,{**_active.__globals__,
    'LAUNCHERS':(*_active.__globals__['LAUNCHERS'],'run_five_baselines_hypermem_native_v3','run_five_baselines_reviewed_chain_v8')},
    'active_controllers',_active.__defaults__)
_stop=frozen._stop
stopped=FunctionType(_stop.__code__,{**_stop.__globals__,'active_controllers':active_controllers},'stopped',_stop.__defaults__)


def _core():
    require(all((ROOT/p).is_file() for p in V3_CORE),'All separately reviewed v3 core files must exist')
    return importlib.import_module('tools.run_five_baselines_hypermem_native_v3')


def code_hashes():
    require(sha(ROOT/FROZEN_ENGINE)==FROZEN_ENGINE_SHA and sha(ROOT/COLLECTOR)==COLLECTOR_SHA,'Frozen lifecycle/terminal auditor changed')
    require(all((ROOT/p).is_file() for p in V3_CORE),'V3 core is not ready')
    core=_core()
    return {**frozen.code_hashes(),**{p:sha(ROOT/p) for p in (*core.CORE,*EXTRA,COLLECTOR)}}


def _sync():
    for name in ('ROOT','OLD','NEW','OUTPUT','DEFAULT_PLAN','SCHEMA','PLAN_SCHEMA','REVIEW_SCHEMA','RESULT_SCHEMA',
                 'CERT_SCHEMA','BANK_REVIEW_SCHEMA','SCOPE','EXTRA','roots','work_roots','safe','binding','bound',
                 'inventory','inventories','hashes','stopped','code_hashes','result_review','banks_for','certify',
                 'bank_structure','reconstruct','replay_all','archived_auditors','evidence_closure'):
        _engine.__dict__[name]=globals()[name]


def disposition(path):
    _sync();return _engine.disposition(path)


def historical_failed_source_projection(value,document,annotations=None):
    """Only one exact historical report resolves its four old source versions.

    This is a closure interpretation, not a change to the archived report or
    runnable source. All non-code evidence still traverses the frozen walker.
    The complete original report and four exact source capsules are retained.
    """
    document=safe(document)
    require(document==ROOT/FAILED_PREFIX_REPORT and sha(document)==FAILED_PREFIX_REPORT_SHA,
        'Unknown historical failed report or changed bytes')
    lineage_path=safe(ROOT/FAILED_SOURCE_LINEAGE)
    require(sha(lineage_path)==FAILED_SOURCE_LINEAGE_SHA,'Exact historical source lineage changed')
    lineage=read(lineage_path)
    require(lineage.get('schema')=='historical-failed-author-source-snapshot-lineage-v1'
        and lineage.get('report')=={'path':str(document),'sha256':FAILED_PREFIX_REPORT_SHA}
        and lineage.get('current_source_claimed_unchanged') is False,'Historical source scope differs')
    rows=lineage.get('code_snapshots',[])
    require(len(rows)==4 and len({x['original_path'] for x in rows})==4,'Exactly four historical source versions required')
    expected={x['original_path']:x['original_sha256'] for x in rows}
    require(value==read(document) and value.get('schema')=='hypermem-native-v3-author-full-prefix-preflight-v1'
        and value.get('status')=='failed' and value.get('role')=='author_validation_not_independent_review'
        and value.get('production_launch_authorized') is False and value.get('actual_replacement_executed') is False
        and value.get('code_sha256')==expected,'Failed report code map or scope differs')
    capsules={}
    for row in rows:
        original=safe(row['original_path']);snapshot=safe(row['snapshot']['path'])
        require(original.is_relative_to(ROOT) and snapshot.parent==document.parent/'code_snapshot'
            and snapshot.name==original.name and row['original_sha256']==row['snapshot']['sha256']
            and sha(snapshot)==row['original_sha256'],'Historical source capsule differs')
        capsules[str(snapshot)]=row['original_sha256']
    require(len(capsules)==4,'Historical source capsules collide')
    if annotations is not None:
        annotations.append({'document':str(document),'document_sha256':FAILED_PREFIX_REPORT_SHA,
            'json_pointer':'$.code_sha256','treatment':'Exact historical source-version binding; original report bytes unchanged, four original versions retained at explicit capsule paths; no current-source equivalence claimed.',
            'lineage':{'path':str(lineage_path),'sha256':FAILED_SOURCE_LINEAGE_SHA},'versions':rows})
    return {**value,'code_sha256':capsules,
        'historical_source_snapshot_lineage':{'path':str(lineage_path),'sha256':FAILED_SOURCE_LINEAGE_SHA}}


def evidence_closure(initial,*,annotations=None):
    require(sha(ROOT/FROZEN_ENGINE)==FROZEN_ENGINE_SHA
        and sha(ROOT/'tools/archive_simplemem_native_v2.py')==frozen.FROZEN_SIMPLE_SHA,'Reviewed closure foundations changed')
    def closure_read(path):
        value=read(path)
        if Path(path)==ROOT/FAILED_PREFIX_REPORT:
            return historical_failed_source_projection(value,path,annotations)
        projected=value
        if any(Path(path)==Path(r['report']['path']) for r in observer_registries()):
            projected=historical_observer_projection(value,path,annotations)
        if any(Path(path)==Path(r['report']['path']) for r in started_registries()):
            started=historical_started_projection(value,path,annotations)
            projected=deepcopy(projected)
            for registry in started_registries():
                if Path(registry['report']['path'])!=Path(path):continue
                for row in registry['records']:
                    group,key=row['field_path']
                    require(projected[group][key]==value[group][key],'Historical capsule rules overlap')
                    projected[group][key]=started[group][key]
            projected['historical_started_capsule_registry']=started['historical_started_capsule_registry']
        if any(Path(path)==ROOT/p for p in STATUS_VERSIONS):
            statuses=historical_status_projection(value,path,annotations);projected=deepcopy(projected)
            for row in status_registry(path)['records']:
                group,key=row['field_path'];require(projected[group][key]==value[group][key],'Historical capsule rules overlap')
                projected[group][key]=statuses[group][key]
            projected['historical_status_capsule_registry']=statuses['historical_status_capsule_registry']
        if Path(path) in (ROOT/EMPTY_SEAL_OBSERVER,ROOT/EMPTY_SEAL_LINEAGE):
            empty=historical_empty_seal_projection(value,path,annotations)
            if Path(path)==ROOT/EMPTY_SEAL_OBSERVER:
                key=str(ROOT/EMPTY_SEAL_SOURCE);require(projected['files'][key]==value['files'][key],'Historical capsule rules overlap')
                projected['files'][key]=empty['files'][key]
                projected['historical_empty_read_capsule_lineage']=empty['historical_empty_read_capsule_lineage']
            else:projected=empty
        return projected
    original=frozen.simple.evidence_closure
    # Preserve its exact typed traversal, prior numeric/reference contracts and
    # one deliberately invalid test classification. Only the single pinned
    # historical document gets the declared four-source interpretation above.
    fn=FunctionType(original.__code__,{**original.__globals__,'read':closure_read},original.__name__,original.__defaults__)
    with _read_context():return fn(initial,annotations=annotations)


def observer_registries():
    cache=_READ_CONTEXT.get();key=('exact_observer_registries',tuple(OBSERVER_REGISTRIES.items()))
    if cache is not None and key in cache:return cache[key]
    values=[]
    for path,h in OBSERVER_REGISTRIES.items():
        p=bound({'path':str(ROOT/path),'sha256':h});v=read(p)
        require(v.get('schema')=='hypermem-exact-observer-byte-version-candidate-registry-v1' and v.get('status')=='passed'
            and v.get('source_before_after_equal') is True and v.get('archive_authorized') is False and v.get('cleanup_authorized') is False,
            'Exact historical byte-version registry differs')
        values.append({**v,'registry_binding':{'path':str(p),'sha256':h}})
    if cache is not None:cache[key]=values
    return values


def historical_observer_projection(value,document,annotations=None):
    """Only individually qualified exact byte versions; all else stays strict."""
    document=safe(document)
    selected=[r for r in observer_registries() if Path(r['report']['path'])==document]
    require(len(selected)==1,'Unknown historical observer document')
    registry=selected[0];require(binding(document)==registry['report'] and value==read(document),'Exact observer bytes differ')
    cache=_READ_CONTEXT.get();key=('live',COHORT_REVIEW_SHA)
    if cache is not None and key in cache:cohort=cache[key]
    else:
        require(sha(ROOT/COHORT_REVIEW)==COHORT_REVIEW_SHA,'Full terminal qualification changed')
        cohort=read(ROOT/COHORT_REVIEW)
        if cache is not None:cache[key]=cohort
    projected=deepcopy(value);seen=set()
    for row in registry['records']:
        source=safe(row['original_source_path']);field=row['field_path']
        require(row['report']==registry['report'] and len(field)==2 and field[0] in ('files','binding_and_prefix_files') and field[1]==str(source)
            and ((source.parent==PARENT/'work/prefix' and frozen.re.fullmatch(r'c[0-9]\.jsonl',source.name))
                 or (source.parent==PARENT/'work/api_method' and source.name in ('api_calls.jsonl','logical_events.jsonl')))
            and tuple(field) not in seen,'Historical record outside exact qualified JSONL field or repeated')
        seen.add(tuple(field));lp=bound(row['lineage']);lineage=read(lp);record=value[field[0]][str(source)]
        require(lineage.get('schema')=='hypermem-exact-historical-observer-prefix-capsule-v1' and lineage.get('status')=='passed'
            and lineage.get('observer')==registry['report'] and lineage.get('field_path')==field
            and lineage.get('source_qualification')=={'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA}
            and lineage.get('original_source_path')==str(source) and lineage.get('historical_prefix_length')==row['observed_length']
            and lineage.get('historical_prefix_sha256')==row['observed_sha256'] and lineage.get('capsule')==row['capsule']
            and lineage.get('terminal_source')==row['terminal_source'] and lineage.get('prefix_byte_equality') is True
            and lineage.get('source_before_after_equal') is True and lineage.get('cleanup_authorized') is False and lineage.get('archive_authorized') is False
            and lineage.get('observer_metadata_changed_during_read') is row['observer_metadata_changed_during_read']
            and lineage.get('historical_observation_claimed_atomic') is False,
            'Individual historical prefix qualification differs')
        require(record['status']=='read' and record['path']==str(source) and record['bytes']==row['observed_length']
            and record['sha256']==row['observed_sha256'] and type(row['observer_metadata_changed_during_read']) is bool
            and record['size_or_mtime_changed_during_read'] is row['observer_metadata_changed_during_read'],
            'Exact individually qualified observer field differs')
        capsule=bound(row['capsule']);terminal=bound(row['terminal_source']);data=capsule.read_bytes();whole=terminal.read_bytes()
        require(terminal==source and cohort['current_inventory'][str(source)]==row['terminal_source']['sha256']
            and len(data)==row['observed_length'] and data==whole[:len(data)] and len(whole)==lineage['terminal_source_length'],
            'Historical capsule is not the exact observed terminal prefix')
        projected[field[0]][str(source)]={**record,'path':str(capsule)}
        if annotations is not None:annotations.append({'document':str(document),'document_sha256':registry['report']['sha256'],
            'field_path':field,'original_source_path':str(source),'historical_length':row['observed_length'],
            'historical_sha256':record['sha256'],'capsule':row['capsule'],'lineage':row['lineage'],
            'original_metadata_changed_during_read':row['observer_metadata_changed_during_read'],'historical_observation_claimed_atomic':False,
            'treatment':'This exact historical hot-read byte version resolves to its verified capsule; all other fields remain recursive and strict; original snapshot bytes retained unchanged.'})
    require(seen,'Empty historical byte-version qualification')
    projected['historical_prefix_capsule_registry']=registry['registry_binding']
    return projected


def started_registries():
    cache=_READ_CONTEXT.get();key=('exact_started_registries',tuple(STARTED_REGISTRIES.items()))
    if cache is not None and key in cache:return cache[key]
    values=[]
    for path,h in STARTED_REGISTRIES.items():
        p=bound({'path':str(ROOT/path),'sha256':h});v=read(p)
        require(v.get('schema')=='hypermem-exact-observer-started-version-candidate-registry-v1' and v.get('status')=='passed'
            and v.get('source_before_after_equal') is True and v.get('unmatched')==[]
            and v.get('archive_authorized') is False and v.get('cleanup_authorized') is False
            and v.get('script')=={'path':str(ROOT/STARTED_HARNESS),'sha256':STARTED_HARNESS_SHA},
            'Exact historical started-version registry differs')
        values.append({**v,'registry_binding':{'path':str(p),'sha256':h}})
    if cache is not None:cache[key]=values
    return values


def historical_started_projection(value,document,annotations=None):
    """A fixed whitelist of byte-identical frozen started-state reconstructions."""
    document=safe(document);selected=[r for r in started_registries() if Path(r['report']['path'])==document]
    require(len(selected)==1,'Unknown historical started-state document')
    registry=selected[0]
    require(binding(document)==registry['report'] and value==read(document),'Exact started-state observer bytes differ')
    require(sha(ROOT/STARTED_HARNESS)==STARTED_HARNESS_SHA,'Qualified started-state constructor changed')
    cache=_READ_CONTEXT.get();key=('started_constructor',STARTED_HARNESS_SHA)
    if cache is not None and key in cache:derive,construction=cache[key]
    else:
        spec=importlib.util.spec_from_file_location('_exact_historical_started_constructor',ROOT/STARTED_HARNESS)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        derive,construction=module.constructor()
        if cache is not None:cache[key]=(derive,construction)
    require(construction==registry['construction'],'Frozen native constructor/writer qualification differs')
    key=('live',COHORT_REVIEW_SHA)
    if cache is not None and key in cache:cohort=cache[key]
    else:
        require(sha(ROOT/COHORT_REVIEW)==COHORT_REVIEW_SHA,'Full terminal qualification changed')
        cohort=read(ROOT/COHORT_REVIEW)
        if cache is not None:cache[key]=cohort
    projected=deepcopy(value);seen=set()
    for row in registry['records']:
        source=safe(row['original_source_path']);field=row['field_path']
        require(row['report']==registry['report'] and len(field)==2 and field[0] in ('files','binding_and_prefix_files')
            and field[1]==str(source) and source.parent.name=='calls'
            and frozen.re.fullmatch(r'c[0-9]',source.parent.parent.name) and source.parent.parent.parent==PARENT/'work/native'
            and frozen.re.fullmatch(r'(episode|topic|fact)_[0-9]{6}_hypermem_[a-z_]+\.json',source.name)
            and tuple(field) not in seen,'Historical started record outside exact qualified field')
        seen.add(tuple(field));lineage=read(bound(row['lineage']));record=value[field[0]][str(source)]
        require(lineage.get('schema')=='hypermem-exact-historical-started-capsule-v1' and lineage.get('status')=='passed'
            and lineage.get('observer')==registry['report'] and lineage.get('field_path')==field
            and lineage.get('source_qualification')=={'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA}
            and lineage.get('original_source_path')==str(source) and lineage.get('construction')==construction
            and lineage.get('historical_length')==row['observed_length'] and lineage.get('historical_sha256')==row['observed_sha256']
            and lineage.get('capsule')==row['capsule'] and lineage.get('terminal_source')==row['terminal_source']
            and lineage.get('deterministic_reconstruction') is True and lineage.get('historical_file_read_today') is False
            and lineage.get('observed_hash_and_length_equal') is True and lineage.get('source_before_after_equal') is True
            and lineage.get('observer_metadata_changed_during_read') is row['observer_metadata_changed_during_read']
            and lineage.get('historical_observation_claimed_atomic') is False
            and lineage.get('archive_authorized') is False and lineage.get('cleanup_authorized') is False,
            'Individual historical started-state qualification differs')
        require(record.get('status')=='parsed' and record.get('path')==str(source)
            and record.get('bytes')==row['observed_length'] and record.get('sha256')==row['observed_sha256']
            and type(row['observer_metadata_changed_during_read']) is bool
            and record.get('size_or_mtime_changed_during_read') is row['observer_metadata_changed_during_read'],
            'Exact reconstructed observer descriptor differs')
        capsule=bound(row['capsule']);terminal=bound(row['terminal_source']);whole=terminal.read_bytes();data=capsule.read_bytes()
        require(terminal==source and cohort['current_inventory'][str(source)]==row['terminal_source']['sha256']
            and len(whole)==lineage['terminal_source_length'] and len(data)==row['observed_length'] and data==derive(json.loads(whole)),
            'Started capsule is not the exact frozen deterministic reconstruction')
        projected[field[0]][str(source)]={**record,'path':str(capsule)}
        if annotations is not None:annotations.append({'document':str(document),'document_sha256':registry['report']['sha256'],
            'field_path':field,'original_source_path':str(source),'historical_length':len(data),'historical_sha256':record['sha256'],
            'capsule':row['capsule'],'lineage':row['lineage'],'construction':construction,
            'treatment':'Exact frozen started-state reconstruction, not a current read of a historical file; original observer and final native bytes both retained. No other JSON state is qualified.'})
    require(seen,'Empty historical started-state qualification')
    projected['historical_started_capsule_registry']=registry['registry_binding']
    return projected


def status_registry(document=None):
    document=safe(document or ROOT/OBSERVER_REPORT)
    selected=[(p,v) for p,v in STATUS_VERSIONS.items() if ROOT/p==document]
    require(len(selected)==1,'Unknown historical status observer')
    _,(report_sha,registry,registry_sha,harness,harness_sha)=selected[0]
    cache=_READ_CONTEXT.get();key=('exact_status_registry',registry_sha)
    if cache is not None and key in cache:return cache[key]
    p=bound({'path':str(ROOT/registry),'sha256':registry_sha});v=read(p)
    require(v.get('schema')=='hypermem-exact-observer-status-version-candidate-registry-v1' and v.get('status')=='passed'
        and v.get('report')=={'path':str(document),'sha256':report_sha}
        and v.get('source_before_after_equal') is True and v.get('archive_authorized') is False and v.get('cleanup_authorized') is False
        and v.get('script')=={'path':str(ROOT/harness),'sha256':harness_sha},'Exact historical status registry differs')
    v={**v,'registry_binding':{'path':str(p),'sha256':registry_sha}}
    if cache is not None:cache[key]=v
    return v


def historical_status_projection(value,document,annotations=None):
    """Only eleven whole status values already saved in the exact observer."""
    document=safe(document);registry=status_registry(document)
    require(binding(document)==registry['report'] and value==read(document),'Unknown or changed historical status observer')
    harness=bound(registry['script'])
    spec=importlib.util.spec_from_file_location('_exact_historical_status_constructor',harness)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);derive,construction=module.constructor()
    require(construction==registry['construction'],'Complete-status constructor differs')
    cache=_READ_CONTEXT.get();key=('live',COHORT_REVIEW_SHA)
    if cache is not None and key in cache:cohort=cache[key]
    else:
        require(sha(ROOT/COHORT_REVIEW)==COHORT_REVIEW_SHA,'Full terminal qualification changed');cohort=read(ROOT/COHORT_REVIEW)
        if cache is not None:cache[key]=cohort
    expected={str(PARENT/'status.json'):['controller_status'],**{
        str(PARENT/f'results/c{ci}/status.json'):['conversations',str(ci),'controller_status'] for ci in range(10)}}
    unchanged=registry.get('unchanged_terminal_statuses',{})
    selected={r['original_source_path']:r['value_field_path'] for r in registry['records']}
    require(isinstance(unchanged,dict) and len(selected)==len(registry['records']) and not set(selected).intersection(unchanged)
        and set(selected)|set(unchanged)==set(expected) and all(expected[k]==v for k,v in selected.items()),
        'All eleven whole statuses must be exact historical versions or unchanged terminal bytes')
    for path,h in unchanged.items():
        record=value['files'][path];payload=value
        for segment in expected[path]:payload=payload[segment]
        require(sha(path)==h==cohort['current_inventory'][path]==record['sha256'] and len(Path(path).read_bytes())==record['bytes']
            and read(path)==payload,'Already terminal observed status differs')
    projected=deepcopy(value)
    for row in registry['records']:
        source=safe(row['original_source_path']);field=row['field_path'];value_field=row['value_field_path']
        require(row['report']==registry['report'] and field==['files',str(source)],'Status descriptor outside exact field')
        record=value['files'][str(source)];payload=value
        for segment in value_field:payload=payload[segment]
        lineage=read(bound(row['lineage']))
        require(lineage.get('schema')=='hypermem-exact-historical-status-capsule-v1' and lineage.get('status')=='passed'
            and lineage.get('observer')==registry['report'] and lineage.get('field_path')==field and lineage.get('value_field_path')==value_field
            and lineage.get('embedded_value')==payload and lineage.get('construction')==construction
            and lineage.get('source_qualification')=={'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA}
            and lineage.get('original_source_path')==str(source) and lineage.get('historical_length')==row['observed_length']
            and lineage.get('historical_sha256')==row['observed_sha256'] and lineage.get('terminal_source')==row['terminal_source']
            and lineage.get('capsule')==row['capsule'] and lineage.get('deterministic_reconstruction') is True
            and lineage.get('historical_file_read_today') is False and lineage.get('complete_embedded_value_unchanged') is True
            and lineage.get('observed_hash_and_length_equal') is True and lineage.get('source_before_after_equal') is True
            and lineage.get('observer_metadata_changed_during_read') is row['observer_metadata_changed_during_read']
            and lineage.get('historical_observation_claimed_atomic') is False
            and lineage.get('archive_authorized') is False and lineage.get('cleanup_authorized') is False,
            'Exact individual complete-status qualification differs')
        require(record.get('status')=='parsed' and record.get('path')==str(source) and record.get('bytes')==row['observed_length']
            and record.get('sha256')==row['observed_sha256'] and type(row['observer_metadata_changed_during_read']) is bool
            and record.get('size_or_mtime_changed_during_read') is row['observer_metadata_changed_during_read'],'Exact status descriptor differs')
        capsule=bound(row['capsule']);terminal=bound(row['terminal_source']);data=capsule.read_bytes();whole=terminal.read_bytes()
        require(terminal==source and cohort['current_inventory'][str(source)]==row['terminal_source']['sha256']
            and len(whole)==lineage['terminal_source_length'] and len(data)==row['observed_length'] and data==derive(payload),
            'Status capsule differs from the complete original observed value')
        projected['files'][str(source)]={**record,'path':str(capsule)}
        if annotations is not None:annotations.append({'document':str(document),'document_sha256':registry['report']['sha256'],
            'field_path':field,'value_field_path':value_field,'capsule':row['capsule'],'lineage':row['lineage'],
            'treatment':'Exact complete embedded status value with frozen serialization; reconstructed historical bytes, not a historical file read today; full terminal status also retained.'})
    projected['historical_status_capsule_registry']=registry['registry_binding']
    return projected


def historical_empty_seal_projection(value,document,annotations=None):
    """One exact zero-byte historical read; never an accepted native seal."""
    document=safe(document);observer=ROOT/EMPTY_SEAL_OBSERVER;source=ROOT/EMPTY_SEAL_SOURCE;lp=ROOT/EMPTY_SEAL_LINEAGE
    require(document in (observer,lp) and sha(observer)==EMPTY_SEAL_OBSERVER_SHA and sha(lp)==EMPTY_SEAL_LINEAGE_SHA,
        'Unknown or changed historical empty-read evidence')
    require(value==read(document),'Historical empty-read source bytes differ')
    lineage=read(lp);observed=read(observer)['files'][str(source)];zero_sha=hashlib.sha256(b'').hexdigest()
    require(lineage.get('schema')=='hypermem-exact-historical-empty-seal-capsule-v1' and lineage.get('status')=='passed'
        and lineage.get('observer')=={'path':str(observer),'sha256':EMPTY_SEAL_OBSERVER_SHA}
        and lineage.get('field_path')==['files',str(source)] and lineage.get('observed_descriptor')==observed
        and lineage.get('historical_length')==0 and lineage.get('historical_sha256')==zero_sha
        and lineage.get('historical_seal_valid') is False and lineage.get('terminal_seal_verified') is True
        and lineage.get('observed_hash_and_length_equal') is True and lineage.get('source_before_after_equal') is True
        and lineage.get('historical_file_read_today') is False and lineage.get('historical_observation_claimed_atomic') is False
        and lineage.get('archive_authorized') is False and lineage.get('cleanup_authorized') is False
        and lineage.get('source_qualification')=={'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA},
        'Exact empty-read qualification differs')
    require(observed.get('status')=='read' and observed.get('path')==str(source) and type(observed.get('bytes')) is int
        and observed['bytes']==0 and observed.get('sha256')==zero_sha and observed.get('size_or_mtime_changed_during_read') is False,
        'Only the exact zero-byte observation is qualified')
    ctor=bound(lineage['constructor']);require(ctor==ROOT/'tools/baselines_hypermem_v1.py'
        and lineage['constructor']['sha256']=='5b4d73650f9c4d69e714468bc31f4da5accf8ce83d9e83bf2112ff51b1643df3','Original seal constructor differs')
    cache=_READ_CONTEXT.get();key=('live',COHORT_REVIEW_SHA)
    if cache is not None and key in cache:cohort=cache[key]
    else:
        require(sha(ROOT/COHORT_REVIEW)==COHORT_REVIEW_SHA,'Full terminal qualification changed');cohort=read(ROOT/COHORT_REVIEW)
        if cache is not None:cache[key]=cohort
    terminal=bound(lineage['terminal_source']);native=bound(lineage['terminal_native']);capsule=bound(lineage['capsule'])
    require(terminal==source and native==source.parent.parent/'episode_000014_hypermem_boundary.json'
        and cohort['current_inventory'][str(terminal)]==lineage['terminal_source']['sha256']
        and cohort['current_inventory'][str(native)]==lineage['terminal_native']['sha256']
        and len(terminal.read_bytes())==64 and terminal.read_bytes().decode('ascii')==sha(native)
        and capsule.read_bytes()==b'','Terminal valid seal/native or historical empty bytes differ')
    projected=deepcopy(value)
    if document==observer:
        projected['files'][str(source)]={**observed,'path':str(capsule)}
        projected['historical_empty_read_capsule_lineage']={'path':str(lp),'sha256':EMPTY_SEAL_LINEAGE_SHA}
    else:
        # The lineage repeats the same descriptor as evidence. Its original
        # bytes stay immutable; that one duplicate observation resolves to
        # the identical capsule, while terminal_source remains the valid seal.
        projected['observed_descriptor']={**observed,'path':str(capsule)}
    if annotations is not None:annotations.append({'document':str(document),'document_sha256':sha(document),
        'field_path':['files',str(source)] if document==observer else ['observed_descriptor'],
        'capsule':lineage['capsule'],'lineage':{'path':str(lp),'sha256':EMPTY_SEAL_LINEAGE_SHA},
        'historical_seal_valid':False,'terminal_seal_verified':True,
        'treatment':'One exact empty byte observation, explicitly not a valid native seal. The full terminal 64-byte seal and native artifact remain original and independently verified.'})
    return projected


class Resolver(_engine.Resolver):
    def __init__(self,*args,**kwargs):_sync();super().__init__(*args,**kwargs)


@contextmanager
def _read_context():
    if _READ_CONTEXT.get() is not None:
        yield;return
    token=_READ_CONTEXT.set({})
    try:yield
    finally:_READ_CONTEXT.reset(token)


def _cohort(proof,view=None):
    # The 118 MB source qualification exists once, not as a second embedded
    # proof copy. Decode it once per live operation or per immutable ZIP view.
    record=proof.get('source_report')
    require(record=={'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA},'Exact full source qualification required')
    view=view or _engine.seal.LiveView()
    owner='live' if isinstance(view,_engine.seal.LiveView) else view
    cache=_READ_CONTEXT.get();key=(owner,COHORT_REVIEW_SHA)
    if cache is not None and key in cache:cohort=cache[key]
    else:
        require(view.sha(record['path'])==record['sha256'],'Original terminal qualification changed')
        cohort=view.read(record['path'])
        if cache is not None:cache[key]=cohort
    core=_core()
    require(core.admission.restore_cohort(proof)==cohort,'Admission is not a reversible normalization of the full original cohort')
    require(cohort.get('schema')=='hypermem-native-v2-independent-terminal-cohort-v1' and cohort.get('status')=='passed'
        and cohort.get('launch_admitted') is False and cohort.get('unknown_or_inflight')==[], 'Ancestor cohort qualification differs')
    return cohort


def result_review(path,view=None,*,live=False):
    _sync();core=_core();view=view or _engine.seal.LiveView();v=view.read(path)
    require(not any(safe(path).is_relative_to(r) for r in roots().values()),'Result review must be outside all three runs')
    require(v.get('schema')==RESULT_SCHEMA and v.get('status')=='passed' and v.get('method')=='hypermem','Independent v3 result review required')
    inv=v.get('reviewed_run_inventories',{});require(set(inv)==set(roots()),'All three complete reviewed inventories required')
    for owner,r in roots().items():
        _engine.check_inventory(inv[owner],r)
        if live:require(hashes(inventory(r))==inv[owner],'Independently reviewed run bytes changed')
    m=view.read(NEW/'run_manifest.json');p=_engine.seal.bound(view,m['terminal_proof']);s=view.read(NEW/'results/summary.json')
    require(m.get('schema')=='hypermem-native-v3-continuation-manifest-v1' and m.get('new_run')==str(NEW),'Wrong v3 result manifest')
    require(m['parent_manifest']=={'path':str(PARENT/'run_manifest.json'),'sha256':view.sha(PARENT/'run_manifest.json')}
        and m['original_manifest']=={'path':str(OLD/'run_manifest.json'),'sha256':view.sha(OLD/'run_manifest.json')}, 'Three generation manifest lineage differs')
    cohort=_cohort(p,view)
    require(inv['original']==cohort['original_inventory'] and inv['parent']==cohort['current_inventory'],'Historical full inventories changed')
    require(view.sha(NEW/'results/summary.json')==v.get('summary_sha256') and s.get('schema')=='five-baseline-result-v1'
        and s.get('method')=='hypermem' and s.get('audit_status')=='no_invalid_artifacts','Incomplete effective HyperMem result')
    for phase in ('firstpass','effective'):
        count=s.get(phase,{})
        require(count.get('slots')==1540 and all(type(count.get(k)) is int and 0<=count[k]<=1540 for k in ('slots','reader_valid','judge_valid','missing_reader','missing_judge'))
            and count.get('missing_reader')==1540-count['reader_valid'] and count.get('missing_judge')==1540-count['judge_valid'],'Fixed denominator/valid accounting differs')
    state=view.read(NEW/'status.json')
    require(state.get('stage')=='completed_awaiting_result_review' and state.get('summary_sha256')==v['summary_sha256'],'Successful current terminal status required')
    require(all(view.read(r/'status.json').get('stage')=='needs_review' for r in (OLD,PARENT)),'Original terminal failure was overwritten')
    require(m['questions']==p['questions']==cohort['questions'] and len(m['questions'])==1540,'Original fixed cohort differs')
    for source,h in {**m['code_sha256'],**m['upstream_sha256']}.items():require(view.sha(ROOT/source)==h,'Scientific source changed')
    cr=_engine.seal.bound(view,m['review'])
    require(cr.get('schema')==core.REVIEW_SCHEMA and cr.get('status')=='passed' and cr.get('proof_sha256')==m['terminal_proof']['sha256']
        and cr.get('code_sha256')==m['code_sha256'] and cr.get('upstream_sha256')==m['upstream_sha256']
        and cr.get('specification')==m['specification'],'V3 core independent admission changed')
    for source,h in cr.get('evidence_sha256',{}).items():require(view.sha(ROOT/source)==h,'Core review evidence changed')
    return m,p,s,v


def banks_for(proof,invs):
    _sync();core=_core();cohort=_cohort(proof)
    require(hashes(invs['original'])==cohort['original_inventory'] and hashes(invs['parent'])==cohort['current_inventory'],'Historical banks changed')
    # Original function still checks each final native state, actual index bytes,
    # every embedding identity/seal and exact normalized-vector digest.
    f=FunctionType(frozen.banks_for.__code__,{**_engine.__dict__,'core':core},'banks_for',frozen.banks_for.__defaults__)
    return f({'original_inventory':cohort['original_inventory']},invs)


def bank_structure(view,proof,cert):
    _sync();f=FunctionType(frozen.bank_structure.__code__,{**_engine.__dict__,'core':_core()},'bank_structure',frozen.bank_structure.__defaults__)
    return f(view,proof,cert)


def _certify(result_path,tool_path):
    _sync();core=_core();audit=importlib.import_module('tools.audit_hypermem_continuation_v3')
    result_path=safe(result_path);tool_path=safe(tool_path);stopped();_engine.tool_review(tool_path)
    require(not (OUTPUT/'bank_validation.json').exists(),'Certificate already exists')
    m,p,s,_=result_review(result_path,live=True);before=inventories()
    require(m==core.expected_manifest(m['terminal_proof']['path'],p,m['review']['path']),'Complete core manifest gate differs')
    require(core.admission.validate_proof(m['terminal_proof']['path'])==p,'Whole terminal admission changed')
    raw=audit.audit_all(NEW,p)
    require(raw==read(NEW/'results/raw_request_audit.json'),'Actual predelete full native/request audit differs')
    outputs=reconstruct(_engine.seal.LiveView(),m,p,created_utc=s['created_utc'],raw_audit=raw)
    for name,value in outputs.items():require(value==read(NEW/'results'/name),'Live four-output replay differs')
    banks=banks_for(p,before)
    marker=NEW/'work/replacement_v3/known_invalid_episode_once.json'
    intent=NEW/'work/api_method/request_intents'/(p['replacement']['request_hash']+'.json')
    stat=marker.stat();intent_value=read(intent)
    require(stat.st_mtime<=frozen.datetime.fromisoformat(intent_value['created_utc']).timestamp()+0.001,'Actual marker postdates transport intent')
    replacement_time={'marker':binding(marker),'marker_mtime_ns':stat.st_mtime_ns,'marker_mtime':stat.st_mtime,
        'transport_intent':binding(intent),'intent_created_utc':intent_value['created_utc'],
        'scope':'Exact predelete physical timestamp observation; original before-send assertion retained during archive replay.'}
    require(inventories()==before,'Run changed during bank certification');stopped()
    value={'schema':CERT_SCHEMA,'status':'passed','created_utc':now(),'scope':'actual_predelete_bytes_and_unmodified_frozen_full_audits',
        'result_review':binding(result_path),'tool_review':binding(tool_path),'manifest':binding(NEW/'run_manifest.json'),'proof':m['terminal_proof'],
        'run_inventories':before,'banks':banks,'replacement_time':replacement_time,'output_sha256':{str(NEW/'results'/n):sha(NEW/'results'/n) for n in OUTPUTS},
        'code_sha256':code_hashes(),'inventory_before_after_equal':True,'database_engine_opened':False,'retrieval_search_reexecuted':False,
        'model_calls':0,'credential_reads':0,'postdelete_scope':SCOPE,
        'embedding_certificate_semantics':'Only exact identity, original byte SHA, seal and normalized-vector digest retained. No vector values or runnable index are packed as cleanup evidence.'}
    save_new(OUTPUT/'bank_validation.json',value)
    return {'status':'certificate_created_requires_independent_review','certificate':binding(OUTPUT/'bank_validation.json')}


def reconstruct(view,m,proof,*,created_utc,raw_audit):
    """V3 original four-output save hook; final core implementation required."""
    core=_core();_sync()
    ns=_engine.qa_namespace(view,m);outputs={}
    def capture(path,value):
        require(Path(path).parent==NEW/'results' and Path(path).name in OUTPUTS and Path(path).name not in outputs,'Unexpected/duplicate seal output')
        outputs[Path(path).name]=value
    p=ROOT/'tools/run_five_baselines_hypermem_native_v3.py'
    require(view.sha(p)==m['code_sha256']['tools/run_five_baselines_hypermem_native_v3.py'],'V3 controller source differs')
    source=view.bytes(p).decode('utf8');tree=ast.parse(source)
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='private_controller')
    hook=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='result_save')
    hookns={**vars(core),'NEW':NEW,'OLD':PARENT,'ANCESTOR':OLD,'m':m,'proof':proof,
        'original_save':capture,'latest_audit':{'value':raw_audit},'binding':lambda p:{'path':str(p),'sha256':view.sha(p)}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[hook],type_ignores=[])),'<original-v3-summary-projection-memory-sink>','exec'),hookns)
    ns.update(save_new=hookns['result_save'],now=lambda:created_utc,audit_requests=lambda folder:raw_audit)
    runseal=FunctionType(core.chain.base.seal.__code__,ns,'seal',core.chain.base.seal.__defaults__)
    runseal('hypermem');require(set(outputs)==set(OUTPUTS),'Incomplete original four-output seal');return outputs


def audit_ancestors(view,proof):
    """Replay the frozen full terminal collector against retained ZIP bytes."""
    _sync();expected=_cohort(proof,view)
    require(view.sha(ROOT/COLLECTOR)==COLLECTOR_SHA and view.sha(ROOT/COHORT_REVIEW)==COHORT_REVIEW_SHA,'Frozen ancestor qualification changed')
    require(view.read(ROOT/COHORT_REVIEW)==expected,'Proof dropped or changed full independent cohort')
    P,bind,lines,inv=_engine.view_functions(view)
    cm=SimpleNamespace(**{**vars(frozen.audit.common),'Path':P,'read':view.read,'file_sha':view.sha,'binding':bind,'jsonl':lines,
        'shared_read':view.bytes,'inventory':inv})
    base=frozen.terminal.chain.base
    bs={**vars(base),'Path':P,'ROOT':P(ROOT),'LOCK':P(base.LOCK),'read':view.read,'sha':view.sha,'load_data':lambda:view.read(ROOT/'data/locomo10.json')}
    for name in ('questions','conversation_only'):
        f=getattr(base,name);bs[name]=FunctionType(f.__code__,bs,f.__name__,f.__defaults__,f.__closure__)
    adm=SimpleNamespace(**{**vars(frozen.terminal),'OLD':P(OLD),'NEW':P(PARENT),'inventory':inv,'assert_stopped':lambda:None,
        'chain':SimpleNamespace(base=SimpleNamespace(**bs))})
    ans={**vars(frozen.audit),'Path':P,'read':view.read,'sha':view.sha,'binding':bind,'jsonl':lines,'common':cm}
    ans['api_audit']=FunctionType(frozen.audit.api_audit.__code__,ans,'api_audit',frozen.audit.api_audit.__defaults__)
    src=view.bytes(ROOT/COLLECTOR).decode('utf8');tree=ast.parse(src)
    # Definitions execute unchanged; imports/constants are already explicit ZIP
    # facades. Current controller checks are performed by the enclosing command.
    namespace={'__file__':str(ROOT/COLLECTOR),'ROOT':P(ROOT),'HERE':P((ROOT/COLLECTOR).parent),'RUN':P(PARENT),'OLD':P(OLD),
        'SCHEMA':expected['schema'],'MANIFEST_SHA':expected['manifest']['sha256'],'PROOF_SHA':expected['original_terminal_proof']['sha256'],
        'ROLE_REVIEW':P(expected['role_qualification']['path']),'ROLE_SHA':expected['role_qualification']['sha256'],
        'DIAGNOSIS':P(expected['malformed_json_diagnosis']['path']),'DIAGNOSIS_SHA':expected['malformed_json_diagnosis']['sha256'],
        'CORE':tuple(expected['current_core_sha256']),'Path':P,'read':view.read,'sha':view.sha,'binding':bind,'inventory':inv,'require':require,'jsonl':lines,
        'common':cm,'admission':adm,'complete_audit':SimpleNamespace(**ans),'adapter':frozen.core.adapter,
        'Counter':Counter,'defaultdict':frozen.defaultdict,'ExitStack':ExitStack,'datetime':frozen.datetime,'hashlib':hashlib,'json':json,'re':frozen.re,
        'digest':frozen.audit.digest,'generation_body':frozen.audit.generation_body,'response_text':frozen.audit.response_text,
        'socket':__import__('socket'),'patch':__import__('unittest.mock',fromlist=['patch']).patch}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='main']
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<unchanged-full-terminal-cohort-zip-facade>','exec'),namespace)
    actual=namespace['collect']();require(actual==expected,'Full ancestor native/API/source qualification differs')
    return {'status':'passed','cohort_review':{'path':str(ROOT/COHORT_REVIEW),'sha256':COHORT_REVIEW_SHA},
        'original_file_count':len(actual['original_inventory']),'parent_file_count':len(actual['current_inventory']),
        'native_calls':actual['counts']['native_generation_calls'],'unknown_or_inflight':actual['unknown_or_inflight'],
        'scope':'Frozen exact terminal collector through archive paths; its OS controller checks delegated to current outer guard; no DB/model.'}


def _archived_formatters(view,m):
    """Compile only the unchanged pure formatters actually used by the audit."""
    base=ROOT/'upstream/baselines_v1/hypermem/hypermem'
    paths={name:base/name for name in ('extractors/topic_extractor.py','prompts/topic_prompts.py','main/stage4_hypergraph_retrieval.py')}
    expected={str(ROOT/p):h for p,h in m['upstream_sha256'].items()}
    for p in paths.values():require(view.sha(p)==expected[str(p)],'Pinned pure formatter source differs')
    def compile_nodes(nodes,name):
        env={'console':SimpleNamespace(print=lambda *args,**kwargs:None)}
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),name,'exec'),env);return env
    tree=ast.parse(view.bytes(paths['extractors/topic_extractor.py']).decode('utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TopicExtractor')
    names=('_format_episode_display','_format_episode_list','_format_topic_display')
    methods=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(methods)==3,'Exact three display methods required')
    local=compile_nodes([ast.ClassDef(name='TopicExtractor',bases=[],keywords=[],body=methods,decorator_list=[])],'<unchanged-pinned-role-display-methods>')
    tree=ast.parse(view.bytes(paths['main/stage4_hypergraph_retrieval.py']).decode('utf-8-sig'))
    funcs=[deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_retrieval_template','format_hierarchical_results')]
    require(len(funcs)==2,'Exact retrieval formatter functions required')
    retrieval=compile_nodes(funcs,'<unchanged-pinned-complete-retrieval-formatter>')
    tree=ast.parse(view.bytes(paths['prompts/topic_prompts.py']).decode('utf-8-sig'))
    template=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT' for t in n.targets)]
    require(len(template)==1 and isinstance(template[0],str),'Exact native role prompt required')
    return {'extractors/topic_extractor.py':SimpleNamespace(TopicExtractor=local['TopicExtractor']),
        'main/stage4_hypergraph_retrieval.py':SimpleNamespace(format_hierarchical_results=retrieval['format_hierarchical_results']),
        'prompts/topic_prompts.py':SimpleNamespace(EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT=template[0])}


def archived_auditors(view,cert,m,proof=None):
    require(view is not None and isinstance(cert,dict) and isinstance(m,dict),'Exact archive view, independent bank certificate and manifest required')
    _sync();core=_core();audit=importlib.import_module('tools.audit_hypermem_continuation_v3')
    p=ROOT/'tools/audit_hypermem_continuation_v3.py'
    require(view.sha(p)==m['code_sha256']['tools/audit_hypermem_continuation_v3.py'],'Final v3 audit source differs')
    P0,bind0,lines0,inv0=_engine.view_functions(view)
    timing=cert.get('replacement_time',{})
    expected_marker=NEW/'work/replacement_v3/known_invalid_episode_once.json'
    if proof is None:proof=view.read(m['terminal_proof']['path'])
    expected_intent=NEW/'work/api_method/request_intents'/(proof['replacement']['request_hash']+'.json')
    require(timing.get('marker')=={'path':str(expected_marker),'sha256':view.sha(expected_marker)}
        and timing.get('transport_intent')=={'path':str(expected_intent),'sha256':view.sha(expected_intent)}
        and timing.get('intent_created_utc')==view.read(expected_intent)['created_utc']
        and type(timing.get('marker_mtime_ns')) is int and timing['marker_mtime_ns']>=0
        and isinstance(timing.get('marker_mtime'),(int,float)) and not isinstance(timing['marker_mtime'],bool)
        and abs(timing['marker_mtime']-timing['marker_mtime_ns']/1_000_000_000)<0.000001,'Certified exact marker metadata differs')
    class ArchivePath(P0):
        def stat(self,*args,**kwargs):
            require(Path(self)==expected_marker,'Uncertified filesystem metadata request: '+str(self))
            require(view.sha(self)==timing['marker']['sha256'],'Exact marker bytes changed')
            return SimpleNamespace(st_mtime=timing['marker_mtime'],st_mtime_ns=timing['marker_mtime_ns'])
    P=ArchivePath
    def bind(p):return {'path':str(P(p).resolve()),'sha256':view.sha(p)}
    def lines(p):
        if not P(p).exists():return []
        b=view.bytes(p);require(not b or b.endswith(b'\n'),'Incomplete terminal JSONL');return [json.loads(x) for x in b.splitlines() if x.strip()]
    def inv(folder):return {str(p):view.sha(p) for p in sorted(P(folder).rglob('*')) if p.is_file()}
    runtime=_engine.private_runtime(view)
    # This loader contains no search/model/DB code and reads every scientific
    # formatter/prompt from its exact retained ZIP source.
    modules=_archived_formatters(view,m);runtime._load=lambda:modules
    qa=_engine.qa_namespace(view,m)
    adm=SimpleNamespace(**{**vars(audit.admission),'ANCESTOR':P(OLD),'OLD':P(PARENT),'NEW':P(NEW),'inventory':inv,
        'chain':SimpleNamespace(base=SimpleNamespace(**qa))})
    cm=SimpleNamespace(**{**vars(audit.common),'Path':P,'read':view.read,'file_sha':view.sha,'binding':bind,'jsonl':lines})
    checked=[]
    def bank(native):return cert['banks'][str(native)]
    def documents(native,state):
        require(bank(native)['document_embedding_sha256']==state['document_embedding_sha256'],'Certified document map differs');checked.append(('documents',str(native)))
    def index(native,name,filename,state):
        require(bank(native)['index_files'][str(native/'cache'/filename)]['sha256']==state['cache_sha256'][name],'Certified index digest differs')
        checked.append(('index',str(native/'cache'/filename)))
    def embedding_paths(native):return [P(p) for p in sorted(bank(native)['embeddings'])]
    def embedding(p):
        e=bank(p.parent.parent)['embeddings'][str(p)]
        require(view.bytes(e['seal']['path']).decode('ascii')==e['original_binding']['sha256'],'Embedding seal differs')
        checked.append(('embedding',str(p)));return e
    text=audit._native_source
    changes=[("        runtime._verify_artifacts(native,state['document_embedding_sha256'])","        verify_certified_documents(native,state)"),
        ("            require(sha(native/'cache'/filename)==state['cache_sha256'][name],'Native index changed')","            verify_certified_index(native,name,filename,state)"),
        ("        for p in sorted((native/'embedding_evidence').glob('*.json')):\n            runtime._verify_seal(p); value=read(p); ident=value['identity']",
         "        for p in certified_embedding_paths(native):\n            value=certified_embedding(p); ident=value['identity']"),
        ("            require(value['vectors']==vectors,'Original normalized native embedding values differ')\n            native_bindings.append(binding(p))",
         "            require(vector_digest(vectors)==value['normalized_vectors_sha256'],'Original normalized native embedding digest differs')\n            native_bindings.append(value['original_binding'])")]
    altered=text
    for before,after in changes:
        require(altered.count(before)==1,'Exact v3 archival cache substitution boundary differs');altered=altered.replace(before,after)
    ns={**vars(audit),'Path':P,'read':view.read,'sha':view.sha,'binding':bind,'jsonl':lines,'admission':adm,'common':cm,
        'adapter':SimpleNamespace(**{**vars(core.adapter),'_runtime':runtime}),'verify_certified_documents':documents,
        'verify_certified_index':index,'certified_embedding_paths':embedding_paths,'certified_embedding':embedding,'vector_digest':digest}
    for name in ('_api_audit','audit_replacement','api_audit','audit_prefixes','verify_representation','audit_all'):
        f=getattr(audit,name);ns[name]=FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
    exec(compile(altered,'<v3-native-audit-exact-certified-cache-substitutions>','exec'),ns)
    transforms={'v3_audit_source_sha256':view.sha(p),'original_native_function_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'archival_native_function_sha256':hashlib.sha256(altered.encode()).hexdigest(),'exact_substitutions':[{'before':a,'after':b} for a,b in changes],
        'scope':SCOPE,'representation_replacement_prefix_api_and_qa_functions_unchanged':True,
        'sole_stat_substitution':{'path':str(expected_marker),'certificate_field':'replacement_time','original_before_send_comparison_retained':True},
        'native_load_substitution':'Only unchanged pinned role-display and complete retrieval formatters/prompt are compiled from ZIP; no graph/index search or model loader.',
        'historical_scope_interpretation':'The reproduced native.scope describes its original predelete audit; archive replay validates certified disposed bytes and does not claim a fresh search.'}
    return ns['audit_all'],transforms,checked


def replay_all(view):
    _sync();am=view.manifest;m,p,s,review=result_review(am['result_review']['path'],view)
    for record in am['bindings'].values():_engine.seal.bound(view,record)
    cert=_engine.verify_certificate(view,am['bank_review']['path'],am['result_review']['path'],am['tool_review']['path'])
    require(cert['manifest']=={'path':str(NEW/'run_manifest.json'),'sha256':view.sha(NEW/'run_manifest.json')}
        and cert['proof']==m['terminal_proof'],'Certificate source differs')
    for owner,r in roots().items():
        actual={x['path']:x['sha256'] for x in am['records'] if Path(x['path']).is_relative_to(r)}
        require(actual==review['reviewed_run_inventories'][owner]==hashes(cert['run_inventories'][owner]),'Three-root archived inventory differs')
    bank_structure(view,p,cert);ancestors=audit_ancestors(view,p)
    auditor,transforms,checked=archived_auditors(view,cert,m,p);raw=auditor(_engine.virtual_paths(view)(NEW),p)
    outputs=reconstruct(view,m,p,created_utc=s['created_utc'],raw_audit=raw)
    for name,value in outputs.items():require(value==view.read(NEW/'results'/name),'Original four-output ZIP replay differs: '+name)
    expected=Counter([('documents',str(NEW/f'work/native/c{ci}')) for ci in range(10)])
    for bank in cert['banks'].values():
        expected.update(('index',p) for p in bank['index_files']);expected.update(('embedding',p) for p in bank['embeddings'])
    require(Counter(checked)==expected,'Incomplete/extra certified-cache consumption')
    return {'status':'passed','scope':SCOPE,'ancestor_audit':ancestors,'replayed_rows':{n:len(outputs[n]) for n in OUTPUTS[:2]},
        'output_content_sha256':{n:digest(v) for n,v in outputs.items()},'explicit_private_transformations':transforms,
        'bank_certificate':am['bindings']['bank_certificate'],'bank_byte_checks_after_delete':0,'database_searches':0,'model_calls':0}


def certify(*args,**kwargs):
    with _read_context():return _certify(*args,**kwargs)
def archive(*args,**kwargs):
    with _read_context():
        _sync();require(not (OUTPUT/'removal_parent.json').exists(),'Historical parent removal already began')
        return _engine.archive(*args,**kwargs)
def verify_archive(*args,**kwargs):
    with _read_context():_sync();return _engine.verify_archive(*args,**kwargs)
def complete(*args,**kwargs):
    with _read_context():_sync();return _engine.complete(*args,**kwargs)
def verify_completion(*args,**kwargs):
    with _read_context():_sync();return _engine.verify_completion(*args,**kwargs)
def strict_replay(*args,**kwargs):
    with _read_context():_sync();return _engine.strict_replay(*args,**kwargs)
def main():_sync();return _engine.main()


_sync()
if __name__=='__main__':main()
