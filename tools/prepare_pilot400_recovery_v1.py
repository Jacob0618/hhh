"""Offline freeze of the separate user-authorized recovery, no credentials."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pilot400_recovery_writer_v1 import OLD,DEFAULT,read,sha,save_once
from tools.audit_pilot400_recorded_qa_v1 import verify_terminal_seal

def main():
    seal=ROOT/'research/heartbeat_checks/20260928T0014/qa_terminal/seal.json'
    verified=verify_terminal_seal(OLD,read(seal))
    dependencies=[seal, ROOT/'configs/pilot400_manifest_v1.json',
        ROOT/'research/pilot400_bank_failure_review_v2.json',
        ROOT/'research/heartbeat_checks/20260928T0014/qa_failure_audit.json',
        ROOT/'research/pilot400_recovery_policy_2026-09-28.md']
    for ci in (1,7):
        dependencies += [OLD/'conversations'/('conv_'+str(ci))/'banks/regroup'/name
            for name in ('group_fact_bank.json','pending_group.json','grouping.json','raw_source.json','bank_summary.json')]
    code=['tools/pilot400_recovery_writer_v1.py','tools/pilot400_recovery_rank_v1.py',
          'jev_mem/group_writer_v2.py','jev_mem/providers.py','jev_mem/pilot400.py',
          'jev_mem/routing.py','jev_mem/evaluation_pilot.py','jev_mem/evaluation.py']
    plan={'schema':'pilot400-recovery-plan-v1','original_run':str(OLD),'output_run':str(DEFAULT),
        'authorization':'User requested rescue of C74 on 2026-09-28; uniform format recovery across all three arms.',
        'fixed_questions_per_arm':400,'eligible_slots':82,'bank_dependent_C':71,'ordinary_format_consumers':11,
        'original_successes_frozen':1118,'old_terminal_files_verified':len(verified),
        'dependencies':[{'path':str(p),'sha256':sha(p)} for p in dependencies],
        'source_files':[{'path':str(ROOT/p),'sha256':sha(ROOT/p)} for p in code],
        'new_writer_fallback_per_group':1,'new_qa_generation_attempts_per_unique_request':1,
        'max_workers':8,'qa_after_complete_bank_and_semantic_review':True,
        'no_new_jev_requests':True,'original_inputs_not_rewritten':True}
    save_once(DEFAULT/'recovery_plan.json',plan)
    print({'old_terminal_files_verified':len(verified),'recovery_plan':str(DEFAULT/'recovery_plan.json')})

if __name__=='__main__':main()
