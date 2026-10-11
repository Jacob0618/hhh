"""Independent bounded diagnostics tests; no production run/inventory read."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/audit_hypermem_native_v3_terminal_v1.py'
spec=importlib.util.spec_from_file_location('independent_terminal_collector_candidate',SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class NativeRecordBoundary(unittest.TestCase):
    def fixture(self,*,raw_text,success=False):
        path=ROOT/'research/synthetic_terminal_native/calls/topic_000001_hypermem_topic_create.json'
        stage='hypermem_topic_create';context={'method':'hypermem','conversation':0,'phase':'build'}
        request={'prompt':'fixture no model','response_format':{'type':'json_object'},'stage':stage,'json_output':True,'max_output':16384,'system':None}
        signature={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
            'body':a.generation_body(request['prompt'],system=None,json_output=True,max_output=16384)}
        event={'request_hash':a.digest(signature),'stage':stage,'context':context}
        record={'request':request,'request_sha256':a.compact(request),'status':'success' if success else 'failed'}
        if not success:record.update(error_type='BaselineFatal',error='Retained generation cannot be parsed: JSONDecodeError')
        rr={'signature':signature,'raw':{'path':str(path.with_name('raw.json')),'sha256':'b'*64},
            'raw_value':{'response':{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':raw_text}]}}]}}}
        return path,context,event,record,rr

    def audit(self,values,checker=lambda *args:None):
        path,context,event,record,rr=values
        with patch.object(a,'read',return_value=record),patch.object(a,'binding',side_effect=lambda p:{'path':str(p),'sha256':'a'*64}),patch.object(a,'seal'):
            return a.audit_one_native(path,event,rr,context,{}, {},0,checker)

    def test_bad_json_without_native_raw_is_accounted_not_accepted(self):
        row=self.audit(self.fixture(raw_text='{"a":'))
        self.assertEqual(row['native_status'],'failed')
        self.assertEqual(row['parse_error']['type'],'JSONDecodeError')
        self.assertEqual(row['native_error']['type'],'BaselineFatal')
        self.assertIsNone(row['parsed_response_sha256'])

    def test_valid_raw_with_missing_native_response_cannot_impersonate_parse_failure(self):
        with self.assertRaises(ValueError):self.audit(self.fixture(raw_text='{"a":1}'))

    def test_unrelated_original_representation_failure_is_not_ordinal_diagnostic(self):
        values=self.fixture(raw_text='{"episode_roles":[{"episode_id":"uuid"}]}',success=True)
        values[3]['raw_response']=json.loads('{"episode_roles":[{"episode_id":"uuid"}]}')
        values[3]['parsed_response']={'episode_roles':[{'episode_id':'episode_1'}]}
        def fail(*args):raise ValueError('Role prompt does not describe the exact retained local evidence')
        with patch.object(a,'authority_after_rejection') as authority:
            with self.assertRaisesRegex(ValueError,'Unexpected original representation audit refusal'):self.audit(values,fail)
            authority.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
