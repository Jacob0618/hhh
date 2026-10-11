import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools import baselines_simplemem_native_v2 as adapter
from tools import simplemem_prefix_api_v2 as prefix
from tools import run_five_baselines_simplemem_native_v2 as entry
from tools.baselines_common_v1 import BaselineAPI,BaselineFatal,read,sha,save_new
from jev_mem import providers
from jev_mem.gemini25_ablation_provider_v1 import generation_body
from jev_mem.sdk20_cached_api import digest

class TransportTests(unittest.TestCase):
    def fixture(self,root):
        stage='simplemem_query_analysis';prompt='original question';body=generation_body(prompt,system=None,json_output=False,max_output=16384)
        sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':body};key=digest(sig)
        context={'method':'simplemem','conversation':0,'query_id':'c0_q0','phase':'retrieval'}
        text='{broken';raw=root/'old'/ (key+'.json');save_new(raw,{'request':sig,'stage':stage,'response':{'candidates':[{'content':{'parts':[{'text':text}]},'finishReason':'STOP'}],'usageMetadata':{}}})
        rb={'path':str(raw),'sha256':sha(raw)}
        try:json.loads(text)
        except json.JSONDecodeError as e:err={'message':e.msg,'line':e.lineno,'column':e.colno,'position':e.pos};message=str(e)
        reg={key:{'signature':sig,'raw':rb,'logical_consumers':[{'stage':stage}]}}
        case={'query_id':'c0_q0','conversation':0,'prefix':[{'ordinal':0,'context':context,'signature':sig,'request_hash':key,'stage':stage,'raw':rb}],
            'certificate':{'kind':'native_analysis_cached_default','request_hash':key,'stage':stage,'raw':rb,'retry_limit':0,'paid_regeneration_permitted':False,
                'analysis_query':'original question','response_text_sha256':prefix.text_sha(text),'json_error':err}}
        directory=root/'new/api_method';admit={'run_dir':str(directory),'query_ids':['c0_q0'],'executing':True,'case_sha256':{'c0_q0':digest(case)},'registry_sha256':digest(reg)}
        return directory,case,reg,admit,context,prompt,text,message
    def test_real_original_cache_import_three_consumers_zero_transport(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'research') as tmp:
            d,c,r,a,ctx,prompt,text,message=self.fixture(Path(tmp))
            with patch.object(providers,'load_credentials',side_effect=AssertionError('vault')),patch.object(socket.socket,'connect',side_effect=AssertionError('network')):
                api=prefix.ContinuationAPI(d,admission=a,case=c,registry=r,context=ctx,keys={'offline':'unused-fixture'})
                for attempt in range(3):
                    self.assertEqual(api.generate(prompt,stage='simplemem_query_analysis'),text)
                    api.authorize_native_compatibility('query_analysis_json',{'query':prompt,'response_text_sha256':prefix.text_sha(text),'attempt':attempt,'exception':{'type':'JSONDecodeError','message':message}})
            self.assertTrue(api.policy.ready);self.assertEqual(len(api.events),3)
            self.assertEqual([e['source'] for e in api.events],['existing_run_cache']*3)
            self.assertEqual(len(list((d/'request_intents').glob('*.json'))),0)
            ledger=[json.loads(x) for x in (d/'api_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(ledger),3);self.assertTrue(all(x['cache_hit'] for x in ledger))
            key=next(iter(r));self.assertEqual(sha(d/'api_cache'/(key+'.json')),r[key]['raw']['sha256'])
    def test_deleted_original_raw_cannot_be_regenerated(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'research') as tmp:
            d,c,r,a,ctx,prompt,text,msg=self.fixture(Path(tmp))
            Path(next(iter(r.values()))['raw']['path']).unlink()
            with patch.object(providers,'load_credentials',side_effect=AssertionError('vault')):
                api=prefix.ContinuationAPI(d,admission=a,case=c,registry=r,context=ctx,keys={'offline':'fixture'})
                with self.assertRaises(BaselineFatal):api.generate(prompt,stage='simplemem_query_analysis')
            self.assertFalse((d/'request_intents').exists())
    def test_manifest_binds_original_science_and_official_metric(self):
        self.assertIn('research/sources/locomo_evaluation_cbfbc1.py',entry.CORE)
        self.assertIs(entry.base.BaselineAPI,BaselineAPI)
        self.assertIn('tools/seal_simplemem_native_v2.py',entry.CORE)
    def test_safe_path_rejects_traversal(self):
        with self.assertRaises(BaselineFatal):adapter._safe_tree(adapter.ROOT/'research'/'..'/'runs')

class BankTests(unittest.TestCase):
    def test_actual_native_bank_copied_without_rebuilding_or_original_writes(self):
        import lancedb
        sys.path.insert(0,str(adapter.ROOT/'tests'))
        from test_baselines_simplemem_v1 import FakeAPI,conversation
        from tools import baselines_simplemem_v1 as frozen
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'research') as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fake=FakeAPI()
            with patch.object(providers,'load_credentials',side_effect=AssertionError('vault')),patch.object(socket.socket,'connect',side_effect=AssertionError('network')):
                state=frozen.build(conversation(2),root/'original',fake)
                original=Path(state['work_dir']);before=adapter._inventory(original);calls=len(fake.calls);embeds=len(fake.embeddings)
                copied=adapter.copy_terminal_bank(original/'state.json',root/'copy',state_sha256=sha(original/'state.json'),database_inventory=adapter._inventory(original/'lancedb'))
                self.assertEqual(adapter._inventory(original),before)
                self.assertEqual(len(fake.calls),calls);self.assertEqual(len(fake.embeddings),embeds)
                self.assertEqual(copied['database_integrity'],state['database_integrity'])
                self.assertEqual(sha(root/'copy/memory.json'),sha(original/'memory.json'))
                with self.assertRaises(FileExistsError):adapter.copy_terminal_bank(original/'state.json',root/'copy',state_sha256=sha(original/'state.json'),database_inventory=adapter._inventory(original/'lancedb'))

if __name__=='__main__':unittest.main()
