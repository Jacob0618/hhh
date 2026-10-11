"""R02 isolated QA adapter: frozen R01 stage semantics, same20 ordering source.

Only run dispatches. Preparation revalidates the prior baseline raw-replay proof
by its exact bytes, source bindings and current complete request signatures.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from pathlib import Path
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_evidence_exchange_qa_r01 as prior
from tools.run_sdk20_modules import digest, save_new, now
from tools.run_sdk20_judge_retest_v2 import read,pointer,sha,bound,verify
from tools.run_mem0_regroup_plain_qa_v1 import publish_json
from tools.summarize_pilot400 import verified_official_metric

SCHEMA='evidence-ordering-r02-plain-qa-run-v1'
SOURCE_SCHEMA='evidence-ordering-r02-selected-inputs-v1'
ARMS=prior.ARMS
POLICY=copy.deepcopy(prior.POLICY)
POLICY.update(ordering='same_original20_records_only_order_changes',
              baseline_proof='prior_raw_replay_exact_bytes_all_bindings_current_requests_revalidated')
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/evidence_ordering_r02.py',
    'tools/prepare_evidence_ordering_r02.py','tools/score_evidence_ordering_r02.py',
    'tools/run_evidence_ordering_qa_r02.py')))

def clone(f, scope, name=None):
    r=FunctionType(f.__code__,scope,name or f.__name__,f.__defaults__,f.__closure__)
    r.__kwdefaults__=dict(f.__kwdefaults__ or {}); r.__module__=__name__
    return r

def source_validate(path):
    from tools.prepare_evidence_ordering_r02 import validate_source
    source=validate_source(Path(path))
    if source.get('schema')!=SOURCE_SCHEMA: raise ValueError('R02 sealed ordering source required')
    return source

SignatureCapture=prior.SignatureCapture
request_signature=prior.request_signature
reuse_plan=prior.reuse_plan
proof_for=prior.proof_for
inherited_stage=prior.inherited_stage
evaluator=prior.evaluator
Gemini38API=prior.Gemini38API
Gemini38Replay=prior.Gemini38Replay
_new_audit_stage=prior._new_audit_stage

def build_reuse_proofs(source):
    """Reuse proven raw replay only if every original byte and identity remains."""
    ref=source.get('frozen_baseline_reuse_proof')
    manifest_ref=source.get('reuse_proof_qa_manifest')
    if not isinstance(ref,dict) or not isinstance(manifest_ref,dict):
        raise ValueError('Explicit prior raw-proof and QA-manifest bindings required')
    proof=bound(ref); prior_m=bound(manifest_ref)
    if (prior_m.get('schema')!=prior.SCHEMA or prior_m.get('phase_kind')!='firstpass'
        or prior_m.get('baseline_reuse_proof')!=ref
        or proof.get('schema')!=prior.SCHEMA+'-reuse-proof'
        or proof.get('model_calls_during_proof')!=0
        or proof.get('frozen_manifest')!=source.get('frozen_qa_run_manifest')):
        raise ValueError('Prior proof lineage differs')
    producer_run=Path(manifest_ref['path']).parent
    producer_out=verify(producer_run/'run_outcome.json')
    producer_jobs={prior.job_key(j):j for j in prior_m['jobs']}
    producer_rows=producer_out.get('results',[])
    if (producer_out.get('terminal') is not True or producer_out.get('total')!=4620
        or producer_out.get('schema')!=prior_m['schema'] or producer_out.get('protocol_sha256')!=prior_m['protocol_sha256']
        or len(producer_jobs)!=4620 or len(producer_rows)!=4620
        or {prior.job_key(j) for j in producer_rows}!=set(producer_jobs)
        or any(r.get('job_id')!=producer_jobs[prior.job_key(r)]['job_id'] for r in producer_rows)):
        raise ValueError('Prior proof-producing QA terminal identity differs')
    # All prior raw replay inputs are hashed again, including raw/stages/intents/
    # ledger and original whole terminal. No summary score enters this function.
    for path,hashed in proof['input_sha256'].items():
        if not Path(path).is_file() or sha(path)!=hashed: raise ValueError('Prior raw-proof input changed: '+path)
    for path,hashed in prior_m['code_sha256'].items():
        if sha(ROOT/path)!=hashed: raise ValueError('Prior proof implementation changed: '+path)
    if prior_m.get('policy')!=prior.POLICY: raise ValueError('Prior QA policy differs')
    old_m=bound(proof['frozen_manifest'])
    if old_m['source_manifest']!=pointer(old_m['source_manifest']['path']): raise ValueError('Original source bytes differ')
    if source.get('frozen_source')!=old_m['source_manifest']: raise ValueError('Original N0 source differs')
    if prior_m['source_manifest']!=pointer(prior_m['source_manifest']['path']): raise ValueError('Prior source bytes differ')
    old_out=bound(proof['frozen_outcome'])
    oldjobs={prior.frozen.job_key(j):j for j in old_m['jobs']}
    oldrows=old_out.get('results',[])
    if (old_out.get('terminal') is not True or old_out.get('total')!=10780
        or old_out.get('schema')!=old_m['schema'] or old_out.get('protocol_sha256')!=old_m['protocol_sha256']
        or len(oldjobs)!=10780 or len(oldrows)!=10780
        or {prior.frozen.job_key(j) for j in oldrows}!=set(oldjobs)
        or any(r.get('job_id')!=oldjobs[prior.frozen.job_key(r)]['job_id'] for r in oldrows)):
        raise ValueError('Original whole terminal identity differs')
    if prior_m.get('protocol_sha256')!=digest({k:v for k,v in prior_m.items() if k!='protocol_sha256'}):
        raise ValueError('Prior QA manifest protocol differs')
    if old_m.get('protocol_sha256')!=digest({k:v for k,v in old_m.items() if k!='protocol_sha256'}):
        raise ValueError('Original QA manifest protocol differs')
    _,metric=verified_official_metric()
    if source['official_metric_provenance']!=metric or old_m['official_metric_provenance']!=metric:
        raise ValueError('Official metric provenance differs')
    entries=proof['entries']; jobs={j['query_id']:j for j in source['jobs'] if j['method']=='baseline'}
    original={j['query_id']:j for j in old_m['jobs'] if j['method']=='baseline'}
    if len(entries)!=1540 or set(entries)!=set(jobs) or set(entries)!=set(original):
        raise ValueError('Full original1540 proof cohort required')
    fields=('conversation','query_id','question','reference','category','reference_date','selected_ids','selected_facts')
    for qid,entry in entries.items():
        if any(jobs[qid].get(k)!=entry['source_job'].get(k) or original[qid].get(k)!=entry['source_job'].get(k) for k in fields):
            raise ValueError('Current/original baseline job differs: '+qid)
        answer=entry['reader_result']['prediction']['answer']
        if (request_signature(jobs[qid],'reader')!=entry['signatures']['reader']
            or request_signature(jobs[qid],'judge',answer)!=entry['signatures']['judge']
            or any(digest(entry['signatures'][s])!=entry['request_sha256'][s] for s in ('reader','judge'))):
            raise ValueError('Current baseline full request differs: '+qid)
    result=copy.deepcopy(proof)
    result.update(schema=SCHEMA+'-reuse-proof',created_utc=now(),
        reuse_revalidation={'proof':ref,'producer_manifest':manifest_ref,
            'producer_outcome':pointer(producer_run/'run_outcome.json'),
            'validated_input_bindings':len(proof['input_sha256']),
            'current_reader_and_judge_signatures':1540,'raw_replay_repeated':False,
            'basis':'previous exact raw replay; all proof and original input bytes unchanged',
            'new_model_calls':0})
    result['input_sha256'].update({ref['path']:ref['sha256'],manifest_ref['path']:manifest_ref['sha256'],
        prior_m['source_manifest']['path']:prior_m['source_manifest']['sha256'],
        str((producer_run/'run_outcome.json').resolve()):sha(producer_run/'run_outcome.json')})
    return result

def _new_manifest(run, source_path, source, *, jobs, phase_kind, parent=None):
    run=Path(run).resolve()
    if run.parent!=(ROOT/'runs').resolve() or run.exists(): raise FileExistsError('Fresh direct runs namespace required')
    check_jobs(jobs); _,metric=verified_official_metric()
    if source['official_metric_provenance']!=metric: raise ValueError('Official scorer differs')
    proof=build_reuse_proofs(source) if phase_kind=='firstpass' else bound(read(parent['manifest']['path'])['baseline_reuse_proof'])
    plans={j['job_id']:(reuse_plan(j,proof) if phase_kind=='firstpass' else
        {'reader':None,'judge_candidate':proof['entries'][j['query_id']]['judge']}) for j in jobs}
    run.mkdir(parents=True,exist_ok=False); save_new(run/'baseline_reuse_proofs.json',proof)
    value={'schema':SCHEMA,'created_utc':now(),'policy':POLICY,'source_manifest':pointer(source_path),
        'phase_kind':phase_kind,'phase_id':'evidence_ordering_r02_qa_'+phase_kind,'jobs':jobs,'parent':parent,
        'official_metric_provenance':metric,'code_sha256':{p:sha(ROOT/p) for p in CODE},
        'model_calls_during_prepare':0,'cache_roots':[],
        'baseline_reuse_proof':pointer(run/'baseline_reuse_proofs.json'),'reuse_plans':plans}
    publish_json(run/'run_manifest.json',{**value,'protocol_sha256':digest(value)})
    return {'run_dir':str(run),'phase_kind':phase_kind,'jobs':len(jobs),'protocol_sha256':digest(value),'model_calls':0}

# Every inherited function executes inside fresh dictionaries. Never mutate the
# R01 module/_impl; its manifest code hashes and globals remain unchanged.
_impl=dict(prior._impl)
for name,f in tuple(_impl.items()):
    if isinstance(f,FunctionType) and f.__globals__ is prior._impl: _impl[name]=clone(f,_impl,name)
_ordinary_manifest=clone(prior._ordinary_manifest,_impl)
_helpers={**vars(prior),'SCHEMA':SCHEMA,'SOURCE_SCHEMA':SOURCE_SCHEMA,'ARMS':ARMS,'POLICY':POLICY,'CODE':CODE,
    '_impl':_impl,'_ordinary_manifest':_ordinary_manifest,'source_validate':source_validate,
    'build_reuse_proofs':build_reuse_proofs,'_new_manifest':_new_manifest}
prepare=clone(prior.prepare,_helpers)
manifest=clone(prior.manifest,_helpers)
one=clone(prior.one,_helpers)
_helpers.update(prepare=prepare,manifest=manifest,one=one)
_impl.update(_impl=_impl,SCHEMA=SCHEMA,SOURCE_SCHEMA=SOURCE_SCHEMA,ARMS=ARMS,POLICY=POLICY,CODE=CODE,
    source_validate=source_validate,prepare=prepare,_new_manifest=_new_manifest,manifest=manifest,one=one)
for name in ('job_key','check_jobs','stage_files','identity','stage_context','recovery_jobs','prepare_recovery',
             'terminal','reader_call','compatibility','resolve_pair','known_failure','run'):
    globals()[name]=_impl[name]
    _helpers[name]=_impl[name]

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='phase',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--run-dir',type=Path,required=True); q.add_argument('--source',type=Path,required=True)
    for name in ('run','check','recover'):
        q=sub.add_parser(name); q.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='prepare': result=prepare(a.run_dir,a.source)
    elif a.phase=='recover':result=prepare_recovery(a.run_dir)
    elif a.phase=='run':result=run(a.run_dir)
    else:result={'phase_kind':manifest(a.run_dir)['phase_kind'],'model_calls':0}
    import json
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
