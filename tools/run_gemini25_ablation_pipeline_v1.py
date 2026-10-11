"""Finite downstream workflow; starts only after the actual ten-bank review.

No Writer, review, repair or automatic restart is implemented here. Scientific
stages remain the existing adapters and frozen selector/evaluation functions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_gemini25_ablation_v1 as source
from tools import score_gemini25_ablation_v1 as score
from tools import run_gemini25_ablation_qa_v1 as qa
from tools import summarize_gemini25_ablation_v1 as report
from tools.prepare_evidence_selection_r03 import Reader, save_new, file_sha
from tools.run_sdk20_modules import now
from tools.run_pilot400_experiments_v2 import publish_json
from tools.check_experiment_health import shared_read

WRITER=ROOT/'runs/gemini25_ablation_writer_v1_20261007'
SOURCE=ROOT/'runs/gemini25_ablation_source_v1_20261007'
QA=ROOT/'runs/gemini25_ablation_qa_v1_20261007'
LIVE=ROOT/'research/gemini25_ablation_2026-10-07/live_pipeline_v1'


def precomputation_usage(writer_run, source_run):
    """Count physical attempts once; cache events stay separate, prices unknown."""
    paths=[]
    for ci in range(10):
        paths.append(('writer',writer_run/f'conv_{ci}/session/native/api_calls.jsonl'))
        paths.append(('retrieval',source_run/f'conv_{ci}/retrieval_api/api_calls.jsonl'))
    for phase in source.PHASES:
        for attempt in ('firstpass','recovery_v1'):
            p=source_run/'phases'/phase/attempt/'api_calls.jsonl'
            if p.exists():paths.append((phase+'/'+attempt,p))
    refs=[];rows=[];by_stage={}
    for stage,p in paths:
        if not p.exists():raise ValueError('Missing expected physical ledger: '+str(p))
        raw=shared_read(p)
        if raw and not raw.endswith(b'\n'):raise ValueError('Truncated physical ledger')
        records=[json.loads(line) for line in raw.splitlines()]
        rows.extend(records);by_stage[stage]=by_stage.get(stage,[])+records
        refs.append({'stage':stage,'path':str(p.resolve()),'sha256':file_sha(p)})
    return {'ledger_refs':refs,'total':report.measured_usage(rows),
            'by_stage':{k:report.measured_usage(v) for k,v in by_stage.items()},
            'historical_cache_inherited':False,'prices_inferred':False,
            'note':'Physical ledger only. Current-run exact-request cache reuse is separate; unavailable fields remain unknown.'}


def run(writer_run=WRITER,source_run=SOURCE,qa_run=QA,live=LIVE):
    writer_run,source_run,qa_run,live=map(lambda p:Path(p).resolve(),(writer_run,source_run,qa_run,live))
    from tools.audit_gemini25_ablation_banks_v1 import admit_review
    admission=admit_review(writer_run,conversations=tuple(range(10)))
    if source_run.exists() or qa_run.exists() or live.exists():
        raise FileExistsError('New workflow only; existing/failed workflow requires explicit technical review')
    live.mkdir(parents=True,exist_ok=False)
    save_new(live/'execution_intent.json',{'pid':os.getpid(),'created_utc':now(),
        'writer_gate':source.pointer(writer_run/'all_bank_review.json'),
        'source_run':str(source_run),'qa_run':str(qa_run),
        'code_sha256':{p:file_sha(ROOT/p) for p in dict.fromkeys((*source.CODE,*qa.CODE,'tools/run_gemini25_ablation_pipeline_v1.py','tools/summarize_gemini25_ablation_v1.py'))}})
    state={'pid':os.getpid(),'status':'running','stage':'source_prepare','started_utc':now(),'completed_stages':[]}
    def stage(name,fn):
        state.update(stage=name,updated_utc=now());publish_json(live/'status.json',state)
        print(json.dumps({'stage':name,'event':'start','utc':now()}),flush=True)
        value=fn()
        # Store output without answers/raw payload in stdout.
        save_new(live/(name+'.json'),value if isinstance(value,dict) else {'result':value})
        state['completed_stages'].append(name);state['updated_utc']=now();publish_json(live/'status.json',state)
        return value
    try:
        stage('source_prepare',lambda:source.prepare(source_run,writer_run,writer_run/'all_bank_review.json'))
        stage('retrieve',lambda:source.retrieve(source_run))
        for phase in source.PHASES:
            stage(phase+'_prepare',lambda p=phase:source.prepare_phase(source_run,p))
            stage(phase+'_firstpass',lambda p=phase:score.run_phase(source_run,p))
            rows=score.phase_results(Reader(),source_run,phase,allow_recovery=False)
            if any(r['status']=='failed' for r in rows.values()):
                stage(phase+'_recovery',lambda p=phase:score.run_phase(source_run,p,recovery=True))
            rows=score.phase_results(Reader(),source_run,phase)
            stage(phase+'_raw_audit',lambda r=rows:{'terminal':len(r),'counts':dict(Counter(x['status'] for x in r.values())),'model_calls':0})
        stage('source_seal',lambda:source.seal(source_run))
        stage('qa_prepare',lambda:qa.prepare(qa_run,source_run/'selected_inputs.json'))
        stage('qa_firstpass',lambda:qa.run(qa_run))
        # This audited preparation rejects unknown/IO/unfinished intent before recovery.
        stage('qa_recovery_prepare',lambda:qa.prepare_recovery(qa_run))
        recovery=qa_run.with_name(qa_run.name+'_recovery_v1')
        stage('qa_recovery',lambda:qa.run(recovery))
        # The raw seal also rejects orphan/unfinished intents, which are
        # reported separately from invalid-artifact counts in the summary.
        stage('qa_raw_audit',lambda:qa.seal(recovery))
        def summary():
            with report.original.historical_guard.historical.offline():
                value=report.summarize(qa_run,recovery)
                if value['audit_status']!='no_invalid_artifacts':raise ValueError('Final raw audit did not pass')
                value['precomputation_usage']=precomputation_usage(writer_run,source_run)
                return value
        stage('final_summary_v1',summary)
        state.update(status='completed_awaiting_research_discussion',finished_utc=now())
        publish_json(live/'status.json',state)
    except Exception as exc:
        state.update(status='needs_review',finished_utc=now(),error_type=type(exc).__name__)
        publish_json(live/'status.json',state)
        raise
    return {'status':state['status'],'summary':str(live/'final_summary_v1.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run',))
    a=p.parse_args();print(json.dumps(run(),ensure_ascii=False))
