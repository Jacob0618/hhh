"""Pure/temp-file tests only; never read the production v3 run inventory."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('readonly_v3_terminal',ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/audit_hypermem_native_v3_terminal_v1.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class TerminalCollector(unittest.TestCase):
    def test_terminal_aggregate_exact_members_values_and_time_order(self):
        failures=[{'conversation':i,'status':'needs_review','error':{'type':'ValueError','message':str(i)}} for i in range(10)]
        outcome={'completed_utc':'2026-10-09T12:01:00+00:00','outcomes':list(reversed(failures))}
        def read(path):
            if path.name=='firstpass_outcome.json':return outcome
            if path==c.RUN/'status.json':return {'updated_utc':'2026-10-09T12:02:00+00:00'}
            ci=int(path.parent.name[1:])
            return failures[ci] if path.name=='failure.json' else {'stage':'needs_review','error':failures[ci]['error'],'updated_utc':'2026-10-09T12:00:00+00:00'}
        with patch.object(c,'read',side_effect=read),patch.object(c,'binding',side_effect=lambda p:{'path':str(p),'sha256':'a'*64}):
            self.assertEqual(c.terminal_aggregate()['completion_order'],list(reversed(range(10))))
            outcome['outcomes'][0]=failures[0]
            with self.assertRaises(ValueError):c.terminal_aggregate()
            outcome['outcomes']=failures;outcome['completed_utc']='2026-10-09T11:00:00+00:00'
            with self.assertRaises(ValueError):c.terminal_aggregate()
            outcome['completed_utc']='2026-10-09T13:00:00+00:00'
            with self.assertRaises(ValueError):c.terminal_aggregate()
    def test_live_controller_rejects_before_read(self):
        with patch.object(c.admission,'assert_stopped',side_effect=ValueError('active')),patch.object(c,'read') as read:
            with self.assertRaisesRegex(ValueError,'active'):c.terminal_gate()
            read.assert_not_called()
    def test_all_ten_terminal_required(self):
        with patch.object(c.admission,'assert_stopped'):
            for status in ('firstpass','build','completed_awaiting_result_review'):
                with patch.object(c,'read',return_value={'stage':status}):
                    with self.assertRaises(ValueError):c.terminal_gate()
            with patch.object(c,'read',side_effect=[{'stage':'needs_review'}]*10+[{'stage':'build'}]):
                with self.assertRaises(ValueError):c.terminal_gate()
            with patch.object(c,'read',return_value={'stage':'needs_review'}):c.terminal_gate()
    def test_malformed_complete_raw_preserves_exact_parse_failure(self):
        text,parsed,error=c.parsed_raw({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"x": "broken "quote"}'}]}}]})
        self.assertIsNone(parsed);self.assertEqual(error['type'],'JSONDecodeError');self.assertIn('quote',text)
    def test_unknown_native_filename_rejects(self):
        for name in ('topic_000001_other.json','unknown_000001_hypermem_episode_roles.json','topic_1_hypermem_topic_create.json'):
            with self.assertRaises(ValueError):c.native_key(Path(name))
        self.assertEqual(c.native_key(Path('topic_000002_hypermem_episode_roles.json'))[:2],(1,2))
    def fixture(self,directory,*,status='success',different=False):
        p=directory/'calls/topic_000002_hypermem_episode_roles.json';p.parent.mkdir(parents=True)
        req={'prompt':'fixture prompt','response_format':{'type':'json_object'},'stage':'hypermem_episode_roles','json_output':True,'max_output':16384,'system':None}
        parsed={'episode_roles':[{'episode_id':'uuid','role':'background','weight':0.2}]}
        derived={'episode_roles':[{'episode_id':'episode_1','role':'background','weight':0.2}]} if different else parsed
        n={'request':req,'request_sha256':c.compact(req),'status':status}
        if status=='success':n.update(raw_response=parsed,parsed_response=derived)
        else:n.update(error_type='HyperMemError',error='Episode role assignment IDs are invalid',raw_response=parsed)
        p.write_text(json.dumps(n),encoding='utf8');inv={str(p):c.sha(p)}
        if status=='success':
            s=p.parent/'.integrity'/(p.name+'.sha256');s.parent.mkdir();s.write_text(c.sha(p),encoding='ascii');inv[str(s)]=c.sha(s)
            if different:
                j=directory/'representations'/p.name;j.parent.mkdir();j.write_text('{}',encoding='utf8')
        sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':c.generation_body(req['prompt'],system=None,json_output=True,max_output=16384)}
        context={'method':'hypermem','conversation':0,'phase':'build'};event={'request_hash':c.digest(sig),'stage':req['stage'],'context':context}
        rr={'signature':sig,'raw':{'path':'fixture raw','sha256':'a'*64},'raw_value':{'response':{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(parsed)}]}}]}}}
        return p,event,rr,context,inv
    def test_success_and_failed_native_have_different_seal_contracts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research') as tmp:
            p,e,rr,ctx,inv=self.fixture(Path(tmp)/'success')
            row=c.audit_one_native(p,e,rr,ctx,inv,{},0,lambda *a:None)
            self.assertEqual(row['native_status'],'success')
            with self.assertRaises(ValueError):c.audit_one_native(p,{**e,'stage':'hypermem_topic_match'},rr,ctx,inv,{},0,lambda *a:None)
            bad=dict(inv);bad.pop(next(x for x in inv if '.integrity' in x))
            with self.assertRaises(ValueError):c.audit_one_native(p,e,rr,ctx,bad,{},0,lambda *a:None)
            p,e,rr,ctx,inv=self.fixture(Path(tmp)/'failure',status='failed')
            row=c.audit_one_native(p,e,rr,ctx,inv,{},0,lambda *a:None);self.assertEqual(row['native_status'],'failed')
            inv[str(p.parent/'.integrity'/(p.name+'.sha256'))]='b'*64
            with self.assertRaises(ValueError):c.audit_one_native(p,e,rr,ctx,inv,{},0,lambda *a:None)
    def test_original_known_refusal_retained_and_unknown_refusal_blocks(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research') as tmp:
            p,e,rr,ctx,inv=self.fixture(Path(tmp),different=True)
            def reject(*args):raise ValueError('Current local topic differs from actual prior generation')
            with patch.object(c,'authority_after_rejection',return_value={'native_acceptance_or_new_permission':False}) as authority:
                row=c.audit_one_native(p,e,rr,ctx,inv,{},0,reject)
                self.assertFalse(row['original_representation_auditor_refusal']['original_validator_passed'])
                self.assertFalse(row['original_representation_auditor_refusal']['waived']);authority.assert_called_once()
            def unknown(*args):raise ValueError('ID derivation differs')
            with self.assertRaisesRegex(ValueError,'Unexpected'):c.audit_one_native(p,e,rr,ctx,inv,{},0,unknown)
    def test_offline_guard_denies_provider_and_socket(self):
        from contextlib import ExitStack
        import socket
        from tools.baselines_common_v1 import BaselineAPI
        attempts=[]
        with ExitStack() as stack:
            c.offline(stack,attempts)
            with self.assertRaises(BaseException):BaselineAPI(ROOT/'research/not_created')
            with self.assertRaises(BaseException):socket.create_connection(('example.invalid',443))
        self.assertEqual(len(attempts),2)
    def test_pure_formatter_loader_checks_all_source_hashes(self):
        from tools import run_five_baselines_gemini25_v1 as base
        upstream=base.source_inventory('hypermem')
        modules=c.pure_role_modules({'upstream_sha256':upstream})
        self.assertEqual(set(modules),{'extractors/topic_extractor.py','prompts/topic_prompts.py'})
        bad=dict(upstream);key=next(k for k in bad if str(k).replace('\\','/').endswith('extractors/topic_extractor.py'));bad[key]='0'*64
        with self.assertRaises(ValueError):c.pure_role_modules({'upstream_sha256':bad})

if __name__=='__main__':unittest.main()
