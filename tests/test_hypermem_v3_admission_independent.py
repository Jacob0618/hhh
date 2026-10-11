"""Independent pure preflight-artifact and real startup-hash gates.

Synthetic ten-case evidence validates the gate mechanics, not the scientific
preflight. Actual all-prefix evidence is reviewed separately.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools import run_five_baselines_hypermem_native_v3 as r
from tools.baselines_common_v1 import save_new,sha
from jev_mem.sdk20_cached_api import digest


class PreflightGate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=r.ROOT/'research');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.report_path=self.root/'report.json'
        self.proof={'schema':r.admission.SCHEMA,'source_scope':'fixture','source_auditor':{'fixture':True},
          'auditor':{'fixture':True},'source_report':{'fixture':True},'collector':{},'original_run':'fixture',
          'parent_run':'fixture','new_run':'fixture','role_cases':[],'replacement':{},
          'prospective_role_representation_enabled':False,'new_generation_replacement_permission_enabled':False,
          'conversations':[]}
        rows=[]
        for ci in range(10):
            entry={'ordinal':1,'native_call':{'path':f'fixture/c{ci}/calls/topic_000001.json'},
                   'native_status':'success' if ci!=5 else 'failed','mode':'inherit','source_mode':'old'}
            self.proof['conversations'].append({'conversation':ci,'prefix':[entry]})
            bank=self.root/f'isolated_run/work/native/c{ci}';save_new(bank/'source_messages.json',{'fixture':True})
            records=[]
            if ci!=5:
                p=bank/'calls/topic_000001.json';save_new(p,{'fixture':True,'conversation':ci})
                records=[{'ordinal':1,**r.binding(p)}]
            rows.append({'conversation':ci,'prefix_total':1,'inherited':int(ci!=5),'source_messages_equal':True,
                         'episodes_equal':ci!=5,'boundary':{'kind':'before_certified_replacement_transport' if ci==5 else 'after_original_prefix_before_first_unseen_transport'},
                         'native_records':records,'native_inventory':r.admission.inventory(bank),
                         'completed_original_native_payloads_equal':int(ci!=5)})
        projection=self.root/'projection.json'
        save_new(projection,{'source_report':self.proof['source_report'],'normalized_value_sha256':digest(self.proof),
          'original_value_sha256':digest(r.admission.restore_cohort(self.proof)),'scope':'author_only_projection_not_production_admission'})
        self.report={'schema':'hypermem-native-v3-author-full-prefix-preflight-v1','status':'passed',
          'role':'author_validation_not_independent_review','production_launch_authorized':False,
          'actual_replacement_executed':False,'model_calls':0,'credential_reads':0,
          'source_report':self.proof['source_report'],'hash_environment':r.admission.hash_environment(),
          'original_inventory_before_after_equal':True,'parent_inventory_before_after_equal':True,
          'qualified_role_boundaries_completed':9,'stopped_before_replacement':1,'stopped_before_first_unseen_suffix':9,
          'blocked_operation_counts':{},'cases':rows,'projection':r.binding(projection),
          'code_sha256':{str(r.ROOT/p):sha(r.ROOT/p) for p in ('tools/baselines_hypermem_native_v3.py',
             'tools/hypermem_prefix_api_v3.py','tools/audit_hypermem_terminal_v3.py',r.PREFLIGHT_HARNESS)}}

    def check(self,report=None):
        self.report_path.write_text(json.dumps(report or self.report),encoding='utf8')
        return r.check_preflight(r.binding(self.report_path),self.proof)

    def test_complete_synthetic_gate_passes_without_any_production_action(self):
        self.assertEqual(self.check(),self.report)

    def test_missing_short_failed_paid_and_fake_code_reports_rejected(self):
        changes=[lambda x:x.update(status='failed'),lambda x:x.update(code_sha256={}),
                 lambda x:x.update(cases=x['cases'][:-1]),lambda x:x.update(model_calls=1),
                 lambda x:x.update(actual_replacement_executed=True),
                 lambda x:x['cases'][0].update(inherited=0),
                 lambda x:x.update(blocked_operation_counts={'network':1}),
                 lambda x:x.pop('hash_environment')]
        for change in changes:
            report=deepcopy(self.report);change(report)
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):
                self.check(report)

    def test_changed_native_projection_or_ordinal_cannot_hide_behind_passed_summary(self):
        self.check()
        path=Path(self.report['cases'][0]['native_records'][0]['path']);before=path.read_bytes()
        path.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'native evidence inventory'):self.check()
        path.write_bytes(before)
        projection=Path(self.report['projection']['path']);before=projection.read_bytes()
        projection.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'projection'):self.check()
        projection.write_bytes(before)
        report=deepcopy(self.report);report['cases'][0]['native_records'][0]['ordinal']=2
        with self.assertRaisesRegex(ValueError,'ordinal'):self.check(report)


class HashStartup(unittest.TestCase):
    def test_actual_startup_seed_required_even_if_environment_is_later_forged(self):
        text="import os; os.environ['PYTHONHASHSEED']='780'; from tools import audit_hypermem_terminal_v3 as a; a.check_hash_environment(); print('HASH_GATE_PASSED')"
        for seed,expected in [('780',True),('779',False)]:
            env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONDONTWRITEBYTECODE':'1'}
            p=subprocess.run([sys.executable,'-B','-c',text],cwd=r.ROOT,env=env,text=True,capture_output=True,timeout=45)
            with self.subTest(seed=seed):
                self.assertEqual(p.returncode==0,expected,p.stderr)
                if expected:self.assertIn('HASH_GATE_PASSED',p.stdout)
                else:self.assertIn('Actual interpreter/hash sentinel differs',p.stderr)


if __name__=='__main__':unittest.main()
