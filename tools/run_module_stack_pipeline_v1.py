"""Finite reviewed module-stack orchestration; no implicit resume or new policy."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_module_stack_v1 as prep
from tools import score_module_stack_v1 as score
from tools import run_module_stack_qa_v1 as qa
from tools import summarize_module_stack_v1 as summary
from tools.score_evidence_exchange_r01 import exclusive_intent,status_write,publish,now,sha


def run(run_dir,qa_run,review_path,audit_dir):
    run_dir,qa_run,audit_dir=(Path(p).resolve() for p in (run_dir,qa_run,audit_dir))
    if run_dir.exists() or qa_run.exists() or audit_dir.exists():
        raise FileExistsError('All pipeline output namespaces must be fresh; no implicit restart')
    for path in (run_dir,qa_run):
        if path.parent!=(ROOT/'runs').resolve():raise ValueError('Direct workspace runs only')
    if not audit_dir.is_relative_to(ROOT/'research'/'module_stack_2026-10-05'):
        raise ValueError('Audit directory outside this reviewed research task')
    review=prep.Reader().read(review_path)
    expected={**prep.codes(),'tools/run_module_stack_pipeline_v1.py':sha(__file__)}
    if (review.get('status')!='ready' or review.get('code_sha256')!=expected
        or review.get('protocol_sha256')!=sha(prep.SPEC)
        or review.get('models_dispatched_before_review')!=0):
        raise ValueError('Exact reviewed code/protocol must precede all model dispatch')
    audit_dir.mkdir(parents=True,exist_ok=False)
    exclusive_intent(audit_dir/'intent.json',{'pid':os.getpid(),'created_utc':now(),'schema':'module-stack-pipeline-v1',
        'review':{'path':str(Path(review_path).resolve()),'sha256':sha(review_path)},
        'run':str(run_dir),'qa_run':str(qa_run),'code_sha256':expected})
    state={'schema':'module-stack-pipeline-v1','pid':os.getpid(),'created_utc':now(),'status':'running',
        'run':str(run_dir),'qa_run':str(qa_run),'stages':[]}
    def step(name,fn):
        state.update(stage=name,updated_utc=now());status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'start','stage':name,'utc':now()},ensure_ascii=False),flush=True)
        value=fn()
        ref=prep.save_new(audit_dir/(name+'.json'),value)
        state['stages'].append({'stage':name,'result':ref,'finished_utc':now()})
        state.update(updated_utc=now());status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'complete','stage':name,'utc':now()},ensure_ascii=False),flush=True)
        return value
    try:
        step('prepare',lambda:prep.prepare(run_dir))
        for phase in ('support','closure','rank'):
            step(phase+'_prepare',lambda phase=phase:prep.prepare_phase(run_dir,phase))
            result=step(phase+'_score',lambda phase=phase:score.run_phase(run_dir,phase))
            if result['status']!='terminal':raise ValueError('Scoring phase requires review: '+phase)
            first=prep.phase_results(prep.Reader(),run_dir,phase,allow_recovery=False)
            if any(r['status']=='failed' for r in first.values()):
                recovered=step(phase+'_recover',lambda phase=phase:score.run_phase(run_dir,phase,recovery=True))
                if recovered['status']!='terminal':raise ValueError('Recovery requires review: '+phase)
            effective=prep.phase_results(prep.Reader(),run_dir,phase)
            step(phase+'_terminal_audit',lambda effective=effective:{'requests':len(effective),
                'effective_counts':dict(Counter(r['status'] for r in effective.values())),
                'unknown':0,'all_raw_and_producer_consumer_verified':True,'model_calls':0})
        step('source_seal',lambda:prep.seal(run_dir))
        step('qa_prepare',lambda:qa.prepare(qa_run,run_dir/'selected_inputs.json'))
        step('qa_firstpass',lambda:qa.run(qa_run))
        qm,qo=qa.terminal(qa_run)
        jobs=qa.recovery_jobs(qm,qo)
        recovery=None
        if jobs:
            step('qa_recovery_prepare',lambda:qa.prepare_recovery(qa_run))
            recovery=qa_run.with_name(qa_run.name+'_recovery_v1')
            step('qa_recovery',lambda:qa.run(recovery))
            qa.terminal(recovery)
        def final_summary():
            with summary.original.historical_guard.historical.offline():
                result=summary.summarize(qa_run,recovery,bootstrap_samples=2000)
            target=audit_dir/'final_summary_v1.json'
            ref=prep.save_new(target,result)
            if result['audit_status']!='no_invalid_artifacts':raise ValueError('Final raw audit retained invalid artifacts')
            return {'summary':ref,'audit_status':result['audit_status'],'model_calls':0}
        step('summary',final_summary)
        state.update(status='completed_awaiting_research_discussion',finished_utc=now(),updated_utc=now())
    except Exception as exc:
        # Do not serialize exception text, which may originate in a transport.
        state.update(status='needs_review',error_type=type(exc).__name__,failed_stage=state.get('stage'),updated_utc=now())
        status_write(audit_dir/'status.json',state)
        publish(audit_dir/'outcome.json',state)
        raise
    status_write(audit_dir/'status.json',state);publish(audit_dir/'outcome.json',state)
    return {'status':state['status'],'stages':len(state['stages']),'run':str(run_dir),'qa_run':str(qa_run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--qa-run',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.run_dir,a.qa_run,a.review,a.audit_dir),ensure_ascii=False))
