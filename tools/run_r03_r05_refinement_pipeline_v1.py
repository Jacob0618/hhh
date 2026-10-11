"""Finite refinement orchestration, following the existing module-stack gates."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_r03_r05_refinement_v1 as prep
from tools import score_r03_r05_refinement_v1 as score
from tools.score_evidence_exchange_r01 import exclusive_intent,status_write,publish,now,sha


def run(run_dir,qa_run,review_path,audit_dir):
    from tools import run_r03_r05_refinement_qa_v1 as qa
    from tools import summarize_r03_r05_refinement_v1 as summary
    run_dir,qa_run,audit_dir=(Path(p).resolve() for p in (run_dir,qa_run,audit_dir))
    if run_dir.exists() or qa_run.exists() or audit_dir.exists():raise FileExistsError('Fresh namespaces required; no implicit resume')
    if any(p.parent!=(ROOT/'runs').resolve() for p in (run_dir,qa_run)):raise ValueError('Direct runs paths required')
    if not audit_dir.is_relative_to(ROOT/'research'/'r03_r05_refinement_2026-10-06'):
        raise ValueError('Audit outside the current finite experiment')
    review=prep.Reader().read(review_path)
    expected={**prep.codes(),'tools/run_r03_r05_refinement_pipeline_v1.py':sha(__file__)}
    if (review.get('status')!='ready' or review.get('code_sha256')!=expected
        or review.get('protocol_sha256')!=sha(prep.SPEC) or review.get('models_dispatched_before_review')!=0):
        raise ValueError('Reviewed exact scientific code/protocol required before dispatch')
    audit_dir.mkdir(parents=True,exist_ok=False)
    exclusive_intent(audit_dir/'intent.json',{'pid':os.getpid(),'created_utc':now(),
        'schema':'r03-r05-refinement-pipeline-v1','review':{'path':str(Path(review_path).resolve()),'sha256':sha(review_path)},
        'run':str(run_dir),'qa_run':str(qa_run),'code_sha256':expected})
    state={'schema':'r03-r05-refinement-pipeline-v1','pid':os.getpid(),'created_utc':now(),
        'status':'running','run':str(run_dir),'qa_run':str(qa_run),'stages':[]}
    def step(name,fn):
        state.update(stage=name,updated_utc=now());status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'start','stage':name,'utc':now()}),flush=True)
        value=fn();ref=prep.save_new(audit_dir/(name+'.json'),value)
        state['stages'].append({'stage':name,'result':ref,'finished_utc':now()})
        state.update(updated_utc=now());status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'complete','stage':name,'utc':now()}),flush=True)
        return value
    try:
        step('prepare',lambda:prep.prepare(run_dir))
        for phase in prep.PHASES:
            step(phase+'_prepare',lambda phase=phase:prep.prepare_phase(run_dir,phase))
            result=step(phase+'_score',lambda phase=phase:score.run_phase(run_dir,phase))
            if result['status']!='terminal':raise ValueError('Scoring phase needs review')
            first=prep.phase_results(prep.Reader(),run_dir,phase,allow_recovery=False)
            if any(row['status']=='failed' for row in first.values()):
                recovered=step(phase+'_recover',lambda phase=phase:score.run_phase(run_dir,phase,recovery=True))
                if recovered['status']!='terminal':raise ValueError('Recovery phase needs review')
            effective=prep.phase_results(prep.Reader(),run_dir,phase)
            step(phase+'_terminal_audit',lambda effective=effective:{'requests':len(effective),
                'effective_counts':dict(Counter(row['status'] for row in effective.values())),
                'unknown':0,'all_raw_and_producer_consumer_verified':True,'model_calls':0})
        step('source_seal',lambda:prep.seal(run_dir))
        step('qa_prepare',lambda:qa.prepare(qa_run,run_dir/'selected_inputs.json'))
        step('qa_firstpass',lambda:qa.run(qa_run))
        qm,qo=qa.terminal(qa_run);recovery=None
        if qa.recovery_jobs(qm,qo):
            step('qa_recovery_prepare',lambda:qa.prepare_recovery(qa_run))
            recovery=qa_run.with_name(qa_run.name+'_recovery_v1')
            step('qa_recovery',lambda:qa.run(recovery));qa.terminal(recovery)
        def final_summary():
            with summary.offline():result=summary.summarize(qa_run,recovery,bootstrap_samples=2000)
            ref=prep.save_new(audit_dir/'final_summary_v1.json',result)
            if result['audit_status']!='no_invalid_artifacts':raise ValueError('Final raw audit failed')
            return {'summary':ref,'audit_status':result['audit_status'],'model_calls':0}
        step('summary',final_summary)
        state.update(status='completed_awaiting_research_discussion',finished_utc=now(),updated_utc=now())
    except Exception as exc:
        state.update(status='needs_review',error_type=type(exc).__name__,failed_stage=state.get('stage'),updated_utc=now())
        status_write(audit_dir/'status.json',state);publish(audit_dir/'outcome.json',state)
        raise
    status_write(audit_dir/'status.json',state);publish(audit_dir/'outcome.json',state)
    return {'status':state['status'],'stages':len(state['stages']),'run':str(run_dir),'qa_run':str(qa_run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--qa-run',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.run_dir,a.qa_run,a.review,a.audit_dir),ensure_ascii=False))
