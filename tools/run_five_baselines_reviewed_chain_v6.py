"""Technical bridge from reviewed predecessor unions to the frozen later run.

The Hyper scientific manifest remains untouched. Its zero-call chain5 failure
receives a separate admission manifest. Only the exact obsolete predecessor
loop is replaced in a private run function; all experiment operations remain
the original source. No permission to resume an intent or adopt a cache exists.
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
from tools import run_five_baselines_reviewed_chain_v5 as previous
from tools import run_five_baselines_gemini25_v1 as base
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import BaselineAPI,read,sha,save_new,now

ENTRY='tools/run_five_baselines_reviewed_chain_v6.py'
METHODS=('hypermem','vmem','prism')
FROZEN_BASE_SHA='ba70d44f8b74be6f4f48b0ca57dac98d242bf0470fbb92837ac15e038a3e531d'
FROZEN_CHAIN_SHA='e21007b617be145986a3b4fdacc1cdb24867e8246197693180ee67dd7cfb4ca7'
ORIGINAL_MANIFEST_SHA='587e091852902ced9fab3081b9f44771e6993d736938f5f21eba0c13071e7b5d'
SCHEMA='hypermem-zero-call-technical-dispatch-v6'
REVIEW_SCHEMA='baseline-reviewed-chain-v6-preparation-supplement-v1'
PRIOR_REVIEW_ROOT='research/baseline_execution_2026-10-08/downstream_dispatch_simplemem_native_v2'
PRIOR_REVIEW_SHA={'hypermem':'b6cc071975f80e619f01ee1e11d18786ad5212119fb72a78a72610204f2eb320',
    'vmem':'24548de17bf7f49f9d58ef2992444d273ae80a045ff5a15797703e8f84ed8591',
    'prism':'8dea55d44e8c7c2283edb6b6b0db233605752657de5e0999129637d9ecf7d09c'}
LAUNCHER='research/baseline_execution_2026-10-08/heartbeat_20261009T0603/launch_hypermem_reviewed_chain_v6.ps1'
LAUNCHER_SHA='c9d6c73589aa59022f8bbf62adfba317c9cda4d3df852e104f6da27efd0a3a98'
QUALIFICATION='research/baseline_execution_2026-10-08/heartbeat_20261009T0603/independent_hypermem_zero_call_qualification_v2.json'
QUALIFICATION_SHA='379488c0d3c7ff95d04340cb5cf19d755c25f5d0b734731ee8d4b737b06e27df'
FAILURE_ROOT='research/baseline_execution_2026-10-08/launch_hypermem_reviewed_chain_v5'
FAILURE_FILES={'launch_intent.json':'a81fff60225ac6be6f69d7a060a79f73444580a0dbc87f31b2217fbe6876969c',
    'receipt.json':'4552d42b2243cafd0b479c8cc154470ba6f869d80c9f81ccd374da278c5bd19a',
    'stderr.log':'dda46287f668b112bef9ad4bafd4f00bdf8ab7e7f21615841f31ae5d0b7a2b72',
    'stdout.log':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
LEGACY_LOOP="""    for prior in METHODS[:METHODS.index(method)]:
        previous = paths(prior)
        if (not (previous / 'results/cleanup_receipt.json').exists()
                or read(previous / 'results/summary.json').get('audit_status') != 'no_invalid_artifacts'):
            raise ValueError('Previous method must finish result review and cleanup before next method')"""
BRIDGE_LINE='    _consume_reviewed_predecessors(method)'
_ORIGINAL_ADAPTER=base.adapter
_ORIGINAL_RUN=base.run
_ACTIVE=guard.active_controllers
active_controllers=FunctionType(_ACTIVE.__code__,{**_ACTIVE.__globals__,
    'LAUNCHERS':(*_ACTIVE.__globals__['LAUNCHERS'],'run_five_baselines_reviewed_chain_v4',
                 'run_five_baselines_reviewed_chain_v5','run_five_baselines_reviewed_chain_v6')},
    'active_controllers',_ACTIVE.__defaults__)
_STOP=guard.assert_stopped
assert_stopped=FunctionType(_STOP.__code__,{**_STOP.__globals__,'active_controllers':active_controllers},
    'assert_stopped',_STOP.__defaults__)
_ISSUER=object()

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def binding(path):
    p=previous.simple.safe(path);return {'path':str(p),'sha256':sha(p)}
def sidecar(method):return base.paths(method)/'technical_dispatch_v6'
def code_hashes():return {**previous.simple.code_hashes(),ENTRY:sha(ROOT/ENTRY),LAUNCHER:LAUNCHER_SHA}
def frozen_sources():
    require(sha(ROOT/'tools/run_five_baselines_gemini25_v1.py')==FROZEN_BASE_SHA,'Frozen experiment runner changed')
    require(sha(ROOT/'tools/run_five_baselines_reviewed_chain_v5.py')==FROZEN_CHAIN_SHA,'Frozen strong predecessor implementation changed')
    require(sha(ROOT/LAUNCHER)==LAUNCHER_SHA,'Reviewed chain6 launcher changed')
    require(base.BaselineAPI is BaselineAPI and base.adapter is _ORIGINAL_ADAPTER and base.run is _ORIGINAL_RUN,
            'Fresh original science/provider process required')

@contextmanager
def guard_context():
    # The same original guard checks every foreign controller. Only the precise
    # verified current venv parent may be ignored for this additional basename.
    changes=((guard,'active_controllers',active_controllers),(guard,'assert_stopped',assert_stopped),
             (previous.simple,'_stop',assert_stopped))
    before=[(m,k,getattr(m,k)) for m,k,_ in changes]
    try:
        for m,k,v in changes:setattr(m,k,v)
        yield
    finally:
        for m,k,v in reversed(before):setattr(m,k,v)

def check_review(method,path):
    path=previous.simple.safe(path);v=read(path)
    require(v.get('schema')==REVIEW_SCHEMA and v.get('status')=='passed'
        and v.get('scope')=='implementation_preparation_supplement' and v.get('method')==method,
        'New independent preparation supplement required')
    require(all(v.get('code_sha256',{}).get(p)==h for p,h in code_hashes().items()),'Review omits the actual entry or inherited source')
    for p,h in v['code_sha256'].items():require(sha(previous.simple.safe(ROOT/p))==h,'Reviewed code changed')
    require(v.get('upstream_sha256')==base.source_inventory(method),'Pinned method upstream differs')
    old_path=ROOT/PRIOR_REVIEW_ROOT/(method+'_preparation_review_v1.json')
    require(sha(old_path)==PRIOR_REVIEW_SHA[method],'Prior full scientific review changed')
    require(any(Path(ROOT/p)==old_path and h==PRIOR_REVIEW_SHA[method] for p,h in v.get('evidence_sha256',{}).items()),
            'Supplement must explicitly retain the prior full scientific review')
    specification=v.get('specification')
    require(specification==read(old_path)['specification'] and
        all(specification.get(k)==value for k,value in base.adapter(method).describe().items()),'Scientific method specification differs')
    for p,h in v.get('evidence_sha256',{}).items():require(sha(previous.simple.safe(ROOT/p))==h,'Preparation evidence changed')
    return v

def failure_admission():
    root=base.paths('hypermem');original=binding(root/'run_manifest.json')
    require(original['sha256']==ORIGINAL_MANIFEST_SHA,'Original zero-call scientific manifest changed')
    qualifier=ROOT/QUALIFICATION
    require(sha(qualifier)==QUALIFICATION_SHA,'Independent zero-call failure qualification changed')
    q=read(qualifier)
    require(q.get('schema')=='hypermem-chain5-zero-dispatch-independent-diagnosis-v1'
        and q.get('status')=='confirmed_local_gate_failure' and q.get('namespace_before_after_equal') is True
        and q.get('namespace_inventory')=={original['path']:ORIGINAL_MANIFEST_SHA}
        and all(type(q.get(k)) is int and q[k]==0 for k in ('producer_count','logical_consumer_count','provider_intents_count','native_artifacts_count')),
        'Failure qualification is not the exact zero-call prefix')
    records={name:binding(ROOT/FAILURE_ROOT/name) for name in FAILURE_FILES}
    require(all(records[name]['sha256']==h for name,h in FAILURE_FILES.items()),'Original failed launch evidence changed')
    return {'qualification':binding(qualifier),'original_inventory':q['namespace_inventory'],'failed_launch':records,
            'producer_count':0,'cache_reuse_permitted':False,'native_prefix_replay_permitted':False}

def pristine(method,*,prepared):
    root=previous.simple.safe(base.paths(method))
    allowed={root/'run_manifest.json'}
    if method=='hypermem' and prepared:allowed.add(sidecar(method)/'manifest.json')
    actual=set()
    for p in root.rglob('*'):
        previous.simple.safe(p)
        if p.is_file():actual.add(p)
        else:require(method=='hypermem' and prepared and p==sidecar(method),'Existing work/results/cache or other directory forbids dispatch')
    require(actual==allowed,'Fresh zero-call namespace required; no resume or orphan adoption')

def completion_paths(method):
    paths={'amem':{
        'completion':previous.previous.archive.old.FINAL/'results/predecessor_completion.json',
        'receipt':previous.previous.archive.old.FINAL/'results/cleanup_receipt.json',
        'plan':previous.previous.archive.DEFAULT_PLAN}}
    for name in previous.previous.frozen.lifecycle.METHODS[:previous.previous.frozen.lifecycle.METHODS.index(method)]:
        if name=='simplemem':
            paths[name]={'completion':previous.simple.OUTPUT/'predecessor_completion.json',
                         'receipt':previous.simple.OUTPUT/'cleanup_receipt.json','plan':previous.simple.DEFAULT_PLAN}
        else:
            root=previous.previous.frozen.lifecycle.folder(name)
            paths[name]={'completion':root/'results/predecessor_completion.json','receipt':root/'results/cleanup_receipt.json',
                         'plan':previous.previous.frozen.lifecycle.plan_path(name)}
    return paths

def strong_predecessors(method):
    wanted=completion_paths(method)
    before={name:{k:binding(p) for k,p in files.items()} for name,files in wanted.items()}
    previous.verify_predecessors(method)  # unchanged actual ZIP/receipt/deletion verification
    after={name:{k:binding(p) for k,p in files.items()} for name,files in wanted.items()}
    require(before==after,'Strong predecessor completion changed while being verified')
    return after

def technical_value(review,m,admission):
    return {'schema':SCHEMA,'method':'hypermem','run_root':str(base.paths('hypermem')),
        'original_manifest':binding(base.paths('hypermem')/'run_manifest.json'),'zero_call_admission':admission,
        'review':binding(review),'code_sha256':code_hashes(),'upstream_sha256':m['upstream_sha256'],
        'specification':m['specification'],'questions_sha256':digest(m['questions']),
        'science_contract':'Original manifest, adapter, transport, prompts, concurrency, native processing and QA are unchanged.',
        'bridge':bridge_description(),'historical_cache':'forbidden','resume_existing_execution':'forbidden'}

def prepare(method,review):
    frozen_sources();checked=check_review(method,review)
    if method!='hypermem':
        require(not base.paths(method).exists(),'Fresh per-method namespace required')
        strong_predecessors(method)
        return base.prepare(method,review)
    pristine(method,prepared=False);m=base.manifest(method);admission=failure_admission()
    require(all(checked['code_sha256'].get(p)==h for p,h in m['code_sha256'].items()),'Old scientific/technical bindings must all be retained')
    require(checked['specification']==m['specification'] and checked['upstream_sha256']==m['upstream_sha256'],'Prepared science contract changed')
    strong_predecessors(method)
    value={'created_utc':now(),**technical_value(review,m,admission)}
    save_new(sidecar(method)/'manifest.json',value)
    return {'method':method,'original_manifest':value['original_manifest'],'technical_manifest':binding(sidecar(method)/'manifest.json'),'model_calls':0}

def manifest(method):
    frozen_sources();m=base.manifest(method)
    if method!='hypermem':check_review(method,m['review']['path']);return m,None
    t=read(sidecar(method)/'manifest.json');check_review(method,t['review']['path'])
    expected=technical_value(t['review']['path'],m,failure_admission())
    require({k:v for k,v in t.items() if k!='created_utc'}==expected and isinstance(t.get('created_utc'),str),'Technical dispatch manifest or source binding differs')
    require(all(read(t['review']['path'])['code_sha256'].get(p)==h for p,h in m['code_sha256'].items()),'Original manifest dependencies were dropped')
    return m,t

def bridge_description():
    frozen_sources();text=(ROOT/'tools/run_five_baselines_gemini25_v1.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    original='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    require(original.count(LEGACY_LOOP)==1,'Exact obsolete predecessor loop boundary differs')
    derived=original.replace(LEGACY_LOOP,BRIDGE_LINE)
    a=ast.parse(original).body[0];b=ast.parse(derived).body[0]
    require(isinstance(a.body[2],ast.For) and isinstance(b.body[2],ast.Expr),'Unexpected run AST boundary')
    a.body[2:3]=[];b.body[2:3]=[]
    require(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Scientific run AST differs outside the sole gate')
    return {'source_sha256':FROZEN_BASE_SHA,'original_function_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'derived_function_sha256':hashlib.sha256(derived.encode()).hexdigest(),
            'removed_loop_sha256':hashlib.sha256(LEGACY_LOOP.encode()).hexdigest(),
            'replacement':BRIDGE_LINE.strip(),'scientific_ast_outside_gate_equal':True}

class _GateTicket:
    def __init__(self,issuer,method,m,t,completions):
        require(issuer is _ISSUER,'Only the verified run entry can issue a bridge ticket')
        review=t['review'] if t else m['review']
        self.method=method;self.used=False
        self.value={'schema':'reviewed-chain-v6-single-use-predecessor-ticket','method':method,'nonce':secrets.token_hex(16),
            'original_manifest':binding(base.paths(method)/'run_manifest.json'),
            'technical_manifest':binding(sidecar(method)/'manifest.json') if t else None,
            'review':review,'code_sha256':code_hashes(),'strong_predecessors':completions,'bridge':bridge_description()}
    def consume(self,method):
        frozen_sources()
        require(not self.used and method==self.method,'Predecessor ticket reused or wrong method')
        require(self.value['code_sha256']==code_hashes(),'Entry code changed after predecessor verification')
        for record in (self.value['original_manifest'],self.value['technical_manifest'],self.value['review']):
            if record is not None:require(binding(record['path'])==record,'Manifest or preparation review changed after verification')
        require(set(self.value['strong_predecessors'])==set(completion_paths(method)),'Predecessor method sequence differs')
        for name,records in self.value['strong_predecessors'].items():
            require(records=={k:binding(p) for k,p in completion_paths(method)[name].items()},'Strong predecessor evidence changed before consumption')
        pristine(method,prepared=(method=='hypermem'))
        self.used=True
        # Prior lifecycle objects have already been independently archived and
        # verified live above. This attests that runtime prerequisite; it does
        # not claim their deleted work is a new experiment input or recursively
        # rearchive those earlier experiments. The full paths exist only in the
        # in-memory single-use ticket and are checked immediately before this.
        persisted={k:v for k,v in self.value.items() if k!='strong_predecessors'}
        persisted['predecessor_acceptance']={'schema':'runtime-predecessor-acceptance-record-v1',
            'scope':'Live completion, receipt, plan, ZIP replay and work-removal gates passed for this exact dispatch; earlier archived experiments are not new model inputs or current archive dependencies.',
            'new_scientific_input':False,'recursive_rearchive_claimed':False,
            'verified_objects':{name:{kind+'_sha256':record['sha256'] for kind,record in records.items()}
                                for name,records in self.value['strong_predecessors'].items()}}
        save_new(sidecar(method)/'execution_intent.json',{'created_utc':now(),'pid':os.getpid(),'ticket':persisted})

def private_run(ticket):
    description=bridge_description();text=(ROOT/'tools/run_five_baselines_gemini25_v1.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    source=source.replace(LEGACY_LOOP,BRIDGE_LINE)
    require(hashlib.sha256(source.encode()).hexdigest()==description['derived_function_sha256'],'Private gate function changed')
    namespace={**vars(base),'_consume_reviewed_predecessors':ticket.consume}
    exec(compile(source,'<reviewed-chain-v6-exact-predecessor-bridge>','exec'),namespace)
    return namespace['run']

def run(method):
    m,t=manifest(method);pristine(method,prepared=(method=='hypermem'))
    completions=strong_predecessors(method)
    ticket=_GateTicket(_ISSUER,method,m,t,completions)
    return private_run(ticket)(method)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check','run'))
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--review',type=Path)
    args=p.parse_args()
    if args.command=='prepare' and args.review is None:p.error('--review required')
    with guard_context():
        assert_stopped()
        if args.command=='prepare':value=prepare(args.method,args.review)
        elif args.command=='check':
            m,t=manifest(args.method);value={'status':'passed','method':args.method,'original_manifest':binding(base.paths(args.method)/'run_manifest.json'),
                'technical_manifest':binding(sidecar(args.method)/'manifest.json') if t else None}
        else:value=run(args.method)
    print(json.dumps(value,ensure_ascii=True))

if __name__=='__main__':main()
