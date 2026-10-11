"""Reviewed finite QA-only continuation after the preserved preparation defect."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import r03_r05_qa_prepare_pointer_adapter_v2 as fix
from tools import prepare_r03_r05_refinement_v1 as prep
from tools.score_evidence_exchange_r01 import sha,now,publish,status_write,exclusive_intent
BASE=ROOT/'research/r03_r05_refinement_2026-10-06'
SOURCE=ROOT/'runs/r03_r05_refinement_v1_20261006/selected_inputs.json'
OLD=ROOT/'runs/r03_r05_refinement_qa_v1_20261006'
NEW=ROOT/'runs/r03_r05_refinement_qa_v2_20261006'
CONT=BASE/'billing_resume_v1/live_continuation_v1'


def bindings():
    files=[SOURCE,OLD/'run_manifest.json',OLD/'prior_qa_reuse_proof.json',CONT/'outcome.json',
           BASE/'finalreview_v1.json',ROOT/'tools/r03_r05_qa_prepare_pointer_adapter_v2.py',
           Path(__file__),ROOT/'tools/test_r03_r05_qa_prepare_pointer_adapter_v2.py']
    return {str(p.resolve()):sha(p) for p in files}


def admission(review_path,audit_dir):
    audit_dir=Path(audit_dir).resolve()
    if audit_dir.exists() or not audit_dir.is_relative_to(BASE.resolve()) or audit_dir==BASE.resolve():
        raise ValueError('Fresh experiment audit namespace required')
    if NEW.exists() or NEW.with_name(NEW.name+'_recovery_v1').exists():raise FileExistsError('Fresh QA v2 required')
    if {p.relative_to(OLD).as_posix() for p in OLD.rglob('*') if p.is_file()} != {'run_manifest.json','prior_qa_reuse_proof.json'}:
        raise ValueError('Old QA must contain only preserved preparation artifacts; no API/intents/stages')
    m=json.loads((OLD/'run_manifest.json').read_text(encoding='utf-8'))
    if (m.get('prior_qa_reuse_proof') is not None or m.get('phase_kind')!='firstpass'
        or m.get('source_manifest')!=fix.frozen.pointer(SOURCE)
        or m.get('protocol_sha256')!=fix.frozen.digest({k:v for k,v in m.items() if k!='protocol_sha256'})
        or m.get('policy')!=fix.frozen.POLICY or m.get('code_sha256')!={p:sha(ROOT/p) for p in fix.frozen.CODE}):
        raise ValueError('Exact known preparation defect required')
    stopped=json.loads((CONT/'outcome.json').read_text(encoding='utf-8'))
    if stopped.get('status')!='needs_review' or stopped.get('failed_stage')!='qa_firstpass':
        raise ValueError('Preserved QA-only boundary required')
    final=prep.Reader().read(BASE/'finalreview_v1.json')
    if (sha(BASE/'finalreview_v1.json')!='82152c39ea56f0196879bc91e4abbc0c1fb6874826f3bb30ed4523d6f45d8e7c'
        or final['code_sha256']!={**prep.codes(),'tools/run_r03_r05_refinement_pipeline_v1.py':sha(ROOT/'tools/run_r03_r05_refinement_pipeline_v1.py')}
        or final['protocol_sha256']!=sha(prep.SPEC)):
        raise ValueError('Scientific freeze differs')
    reviewed=json.loads(Path(review_path).read_text(encoding='utf-8'))
    pinned=bindings()
    if (reviewed.get('schema')!='r03-r05-qa-prepare-fix-review-v2' or reviewed.get('status')!='ready'
        or reviewed.get('bindings')!=pinned or reviewed.get('models_dispatched_before_review')!=0):
        raise ValueError('Independent exact-hash ready review required')
    evidence=reviewed.get('independent_review')
    if (not isinstance(evidence,dict) or set(evidence)!={'path','sha256'}
        or Path(evidence['path']).resolve()==Path(review_path).resolve() or sha(evidence['path'])!=evidence['sha256']):
        raise ValueError('Distinct independent review artifact required')
    return pinned


def run(review_path,audit_dir):
    pinned=admission(review_path,audit_dir)
    audit_dir=Path(audit_dir).resolve();audit_dir.mkdir(parents=True,exist_ok=False)
    exclusive_intent(audit_dir/'intent.json',{'schema':'r03-r05-qa-prepare-fix-v2','pid':os.getpid(),
        'created_utc':now(),'review':{'path':str(Path(review_path).resolve()),'sha256':sha(review_path)},'bindings':pinned})
    state={'schema':'r03-r05-qa-prepare-fix-v2','pid':os.getpid(),'created_utc':now(),'status':'running','stages':[]}
    def step(name,fn):
        state.update(stage=name,updated_utc=now());status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'start','stage':name,'utc':now()}),flush=True)
        value=fn();ref=prep.save_new(audit_dir/(name+'.json'),value)
        state['stages'].append({'stage':name,'result':ref,'finished_utc':now()});status_write(audit_dir/'status.json',state)
        return value
    qa=fix.frozen
    from tools import summarize_r03_r05_refinement_v1 as summary
    try:
        step('qa_prepare_v2',lambda:fix.prepare(NEW,OLD,technical_bindings=pinned))
        def firstpass():
            if bindings()!=pinned:raise ValueError('Pinned old evidence changed before original QA admission')
            return qa.run(NEW)
        step('qa_firstpass',firstpass)
        qm,qo=qa.terminal(NEW);recovery=None
        if qa.recovery_jobs(qm,qo):
            step('qa_recovery_prepare',lambda:qa.prepare_recovery(NEW))
            recovery=NEW.with_name(NEW.name+'_recovery_v1')
            step('qa_recovery',lambda:qa.run(recovery));qa.terminal(recovery)
        def finish():
            with summary.offline():result=summary.summarize(NEW,recovery,bootstrap_samples=2000)
            ref=prep.save_new(audit_dir/'final_summary_v1.json',result)
            if result['audit_status']!='no_invalid_artifacts':raise ValueError('Final raw audit failed')
            return {'summary':ref,'audit_status':result['audit_status'],'model_calls':0}
        step('summary',finish)
        state.update(status='completed_awaiting_research_discussion',finished_utc=now())
    except Exception as exc:
        state.update(status='needs_review',error_type=type(exc).__name__,failed_stage=state.get('stage'),updated_utc=now())
        status_write(audit_dir/'status.json',state);publish(audit_dir/'outcome.json',state);raise
    state.update(updated_utc=now());status_write(audit_dir/'status.json',state);publish(audit_dir/'outcome.json',state)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.review,a.audit_dir),ensure_ascii=False))
