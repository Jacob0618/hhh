"""Reviewed R11 continuation after all scoring ended; no JEV dispatch exists here."""
from __future__ import annotations
import json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.score_evidence_exchange_r01 import exclusive_intent,status_write,now
from tools.run_sdk20_judge_retest_v2 import read,pointer,verify
from tools.run_sdk20_modules import save_new
from tools import prepare_source_set_rounding_r11 as source
from tools import run_source_set_rounding_qa_r11 as qa
from tools import summarize_source_set_rounding_r11 as summary
from tools.project_source_set_strict_r11 import project
from tools.compare_source_set_r11 import make

VIEW=ROOT/'runs/source_set_r11_rounding_view_v1'
QA=ROOT/'runs/source_set_rounding_qa_r11_v1'
REPORT=ROOT/'research/module_consolidation_2026-10-05/R11_live'

def main():
    state={'schema':'r11-reviewed-rounded-sensitivity-continuation-v1','pid':os.getpid(),'started_utc':now(),
        'status':'running','stage':'validate_prior_terminal','selection_run':str(VIEW),'qa_run':str(QA),
        'strict_primary_retained':True,'new_JEV_calls':0}
    old=read(REPORT/'pipeline_status.json')
    score=verify(ROOT/'runs/source_set_r11_v1/score_jev_outcome.json')
    if old.get('status')!='needs_review' or score['status']!='terminal' or score['terminal_requests']!=3080 or score.get('local_errors'):
        raise ValueError('Only fully ended and reviewed first scoring phase admitted')
    if not source.STRICT_SOURCE.is_file():raise ValueError('Strict source must be sealed before derived preparation')
    exclusive_intent(REPORT/'reviewed_continuation_intent.json',{**state,'strict_source':pointer(source.STRICT_SOURCE),
        'original_pipeline_terminal':pointer(REPORT/'pipeline_status.json'),'score_outcome':pointer(ROOT/'runs/source_set_r11_v1/score_jev_outcome.json')})
    def mark(stage,**fields):
        state.update(stage=stage,updated_utc=now(),**fields);status_write(REPORT/'reviewed_status.json',state)
        print(json.dumps({'stage':stage,'status':state['status'],'utc':state['updated_utc']},ensure_ascii=False),flush=True)
    try:
        mark('rounding_view_prepare');source.prepare(VIEW)
        mark('qa_prepare');qa.prepare(QA,VIEW/'selected_inputs.json')
        mark('qa_firstpass');qa.run(QA)
        m,out=qa.terminal(QA);jobs=qa.recovery_jobs(m,out);recovery=None
        if jobs:
            mark('qa_recovery_prepare',technical_recovery_slots=len(jobs));qa.prepare_recovery(QA)
            recovery=QA.with_name(QA.name+'_recovery_v1');mark('qa_recovery');qa.run(recovery);qa.terminal(recovery)
        mark('raw_audit')
        with summary.original.historical_guard.historical.offline():result=summary.summarize(QA,recovery,bootstrap_samples=2000)
        compat_path=REPORT/'rounded_raw_summary_v1.json';save_new(compat_path,result)
        if result['audit_status']!='no_invalid_artifacts':raise ValueError('Raw audit rejected artifacts')
        mark('strict_exact_projection');strict_path=REPORT/'strict_primary_summary_v1.json';project(compat_path,strict_path)
        mark('comparisons')
        oldpath=ROOT/'research/cleanup_2026-10-05/r05_postcleanup_raw_audit_v1.json'
        make(strict_path,REPORT/'strict_five_arm_comparison_v1.json',oldpath)
        make(compat_path,REPORT/'rounded_five_arm_comparison_v1.json',oldpath)
        mark('awaiting_result_discussion',status='complete',strict_summary=pointer(strict_path),rounded_summary=pointer(compat_path))
    except Exception as exc:
        mark(state['stage'],status='needs_review',error_type=type(exc).__name__)
        raise RuntimeError(state['stage']+': needs review; '+type(exc).__name__) from None

if __name__=='__main__': main()
