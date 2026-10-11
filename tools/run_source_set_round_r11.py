"""One authorized R11 round. Reuse frozen stage tools; stop on unknown failures."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.score_evidence_exchange_r01 import exclusive_intent,status_write,now
from tools.run_sdk20_judge_retest_v2 import read,pointer
from tools import score_source_set_r11 as scoring
from tools import prepare_source_set_r11 as selection
from tools import run_source_set_qa_r11 as qa
from tools import summarize_source_set_r11 as summary
from tools import compare_source_set_r11 as compare
from tools.run_sdk20_modules import save_new

RUN=ROOT/'runs/source_set_r11_v1'
QA=ROOT/'runs/source_set_qa_r11_v1'
REPORT=ROOT/'research/module_consolidation_2026-10-05/R11_live'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    state={'schema':'source-set-r11-single-round-pipeline-v1','pid':os.getpid(),'started_utc':now(),'stage':'check',
        'selection_run':str(RUN),'qa_run':str(QA),'status':'running'}
    exclusive_intent(REPORT/'execution_intent.json',{**state,'manifest':pointer(RUN/'run_manifest.json')})
    def mark(stage,**extra):
        state.update(stage=stage,updated_utc=now(),**extra);status_write(REPORT/'pipeline_status.json',state)
        print(json.dumps({'stage':stage,'status':state['status'],'utc':state['updated_utc']},ensure_ascii=False),flush=True)
    try:
        mark('score');out=scoring.run_phase(RUN)
        if out['status']!='terminal' or out.get('local_errors'): raise RuntimeError('Score terminal requires investigation')
        failures=[read(p) for p in (RUN/'responses').glob('*/*.json') if read(p)['status']=='failed']
        if failures:
            # Review actual failure population before any new recovery or QA.
            save_new(REPORT/'score_failure_population.json',failures)
            raise RuntimeError('Score failures require bounded recovery review before QA')
        mark('seal');selection.seal(RUN)
        mark('qa_prepare');qa.prepare(QA,RUN/'selected_inputs.json')
        mark('qa_firstpass');first=qa.run(QA)
        _,terminal=qa.terminal(QA)
        m=qa.manifest(QA);jobs=qa.recovery_jobs(m,terminal)
        recovery=None
        if jobs:
            mark('qa_recovery_prepare',technical_recovery_slots=len(jobs))
            qa.prepare_recovery(QA);recovery=QA.with_name(QA.name+'_recovery_v1')
            mark('qa_recovery');qa.run(recovery);qa.terminal(recovery)
        mark('raw_audit')
        with summary.original.historical_guard.historical.offline():
            result=summary.summarize(QA,recovery,bootstrap_samples=2000)
        sp=REPORT/'raw_summary_v1.json';save_new(sp,result)
        if result['audit_status']!='no_invalid_artifacts': raise RuntimeError('Offline raw audit rejected artifacts')
        mark('five_arm_comparison')
        compare.make(sp,REPORT/'five_arm_comparison_v1.json',
            ROOT/'research/cleanup_2026-10-05/r05_postcleanup_raw_audit_v1.json')
        mark('awaiting_result_discussion',status='complete',summary=pointer(sp),comparison=pointer(REPORT/'five_arm_comparison_v1.json'))
    except Exception as exc:
        # Error text is local stage errors only; do not print request/client representations.
        state.update(status='needs_review',error_type=type(exc).__name__,updated_utc=now())
        status_write(REPORT/'pipeline_status.json',state)
        print(json.dumps({'status':'needs_review','stage':state['stage'],'error_type':type(exc).__name__},ensure_ascii=False),flush=True)
        raise RuntimeError(state['stage']+': needs review; '+type(exc).__name__) from None

if __name__=='__main__': main()
