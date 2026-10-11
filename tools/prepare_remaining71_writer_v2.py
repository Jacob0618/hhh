"""Freeze source-only recovery before new model calls; no credential access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pilot400_members_only_writer_v2 import OUT,OLD,PREVIOUS,read,sha,save_once
from tools.audit_pilot400_recorded_qa_v1 import verify_terminal_seal
from tools.pilot400_recovery_qa_v1 import TERMINAL_SEAL

def main():
    original=verify_terminal_seal(OLD,read(TERMINAL_SEAL))
    prior=read(PREVIOUS/'recovery_terminal.json')
    for item in prior['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Prior recovery changed')
    paths=[TERMINAL_SEAL,PREVIOUS/'recovery_terminal.json',PREVIOUS/'banks/conv_1/bank_overlay.json',
        ROOT/'research/pilot400_remaining71_policy_v2_2026-09-28.md',
        ROOT/'research/pilot400_remaining71_independent_policy_review_2026-09-28.json',
        ROOT/'configs/pilot400_manifest_v1.json']
    for ci in (1,7):
        paths += [OLD/'conversations'/('conv_'+str(ci))/'banks/regroup'/name
            for name in ('grouping.json','raw_source.json','group_fact_bank.json','pending_group.json')]
    paths += [ROOT/name for name in ('tools/pilot400_members_only_writer_v2.py',
        'jev_mem/group_writer_v2.py','jev_mem/providers.py','jev_mem/data.py',
        'jev_mem/topic_grouping.py','jev_mem/pilot400.py')]
    plan={'schema':'remaining71-writer-plan-v2','explicit_authorization':'User requested restoration of remaining71 on 2026-09-28.',
        'new_groups':14,'reused_ordinary_v2_groups':63,'new_questions_after_bank_review':71,
        'new_jev_requests':0,'old_terminal_files_verified':len(original),
        'writer_workers':{'conversations':2,'groups_per_conversation':4},
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in paths]}
    save_once(OUT/'writer_plan.json',plan)
    print({'plan':str(OUT/'writer_plan.json'),'old_files_verified':len(original),'new_groups':14})

if __name__=='__main__':main()
