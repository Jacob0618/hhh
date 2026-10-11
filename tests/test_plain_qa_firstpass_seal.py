import copy
from datetime import datetime,timezone,timedelta
import unittest
from pathlib import Path
from tools.seal_mem0_plain_qa_firstpass_v1 import check_snapshot_pair,terminal_evidence,reject_live_qa

class TerminalSealGuards(unittest.TestCase):
    def evidence(self):
        run=Path('test_qa_run').resolve();outcomes=[];conversations=[]
        for ci in range(10):
            q={'conversation':ci,'completed':308,'total':308,'reader_complete':308,'judge_complete':306,'failed':2,'terminal':True}
            outcomes.append(q)
            arms={a:{stage:{'output_files':154,'counts':({'complete':153,'failed':1} if stage in ('judge','qa') else {'complete':154})}
                     for stage in ('selection','reader','judge','qa')} for a in ('dated_unmerged385','dated_merged')}
            conversations.append({'conversation':ci,'expected_per_arm':154,'states':{'qa_status.json':q},'arms':arms,'api':{'malformed_line_numbers':[]}})
        return {'run_dir':str(run),'protocol_sha256':'p','expected_slots':3080,'processes':[],
                'captured_utc':'2026-09-28T18:40:00+00:00','controller':{'run_status.json':{
                    'terminal':True,'conversations_terminal':10,'expected_qa':3080,'failures':[],'outcomes':outcomes}},'conversations':conversations}
    def test_terminal_strict_failures_are_valid_evidence(self):
        a=self.evidence();b=copy.deepcopy(a);b['captured_utc']='2026-09-28T18:40:30+00:00'
        gap,_=check_snapshot_pair(a,b,a['run_dir'],'p');self.assertEqual(gap,30)
    def test_gap_and_changed_counts_refused(self):
        a=self.evidence();b=copy.deepcopy(a);b['captured_utc']='2026-09-28T18:40:29+00:00'
        with self.assertRaises(ValueError):check_snapshot_pair(a,b,a['run_dir'],'p')
        b['captured_utc']='2026-09-28T18:40:30+00:00';b['conversations'][0]['arms']['dated_merged']['judge']['counts']={'complete':154}
        with self.assertRaises(ValueError):check_snapshot_pair(a,b,a['run_dir'],'p')
    def test_nonterminal_or_missing_stage_refused(self):
        for modification in ('controller','worker','slot','process'):
            a=self.evidence()
            if modification=='controller':a['controller']['run_status.json']['terminal']=False
            elif modification=='worker':a['conversations'][0]['states']['qa_status.json']['terminal']=False
            elif modification=='slot':a['conversations'][0]['arms']['dated_merged']['reader']['output_files']=153
            else:a['processes']=[{'ProcessId':123}]
            with self.subTest(modification=modification),self.assertRaises(ValueError):terminal_evidence(a,a['run_dir'],'p')
    def test_recorded_pid_or_named_driver_refused(self):
        reject_live_qa([{'ProcessId':12,'CommandLine':'python unrelated.py'}],{13})
        for p in [{'ProcessId':13,'CommandLine':'python child'}, {'ProcessId':12,'CommandLine':'python tools/run_mem0_regroup_plain_qa_v2.py run'}]:
            with self.assertRaises(ValueError):reject_live_qa([p],{13})

if __name__=='__main__':unittest.main()
