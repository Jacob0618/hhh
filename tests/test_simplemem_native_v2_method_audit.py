"""Real native retrieval + frozen transport cache + pure sidecar audit fixture."""
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
from tools import run_five_baselines_simplemem_native_v2 as controller
from tools import baselines_simplemem_native_v2 as adapter
from tools import audit_simplemem_terminal_v2 as audit
from tools import simplemem_prefix_api_v2 as prefix
from tools import baselines_simplemem_v1 as frozen
from tools.baselines_common_v1 import BaselineAPI,read,sha,save_new,append,now
from jev_mem import providers
from jev_mem.sdk20_cached_api import digest

class MethodFixture(unittest.TestCase):
    def test_real_native_transport_cache_and_full_text_audit(self):
        import lancedb
        sys.path.insert(0,str(adapter.ROOT/'tests'))
        from test_baselines_simplemem_v1 import FakeAPI,conversation
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'research') as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);old=root/'old';new=root/'new';context={'method':'simplemem','conversation':0,'query_id':'c0_q0','phase':'retrieval'}
            state=frozen.build(conversation(2),old/'work/native/c0',FakeAPI());bank=Path(state['work_dir'])
            original_request=providers.ModelAPI.request
            def fake_request(api,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
                sig={'provider':provider,'model':model,'operation':operation,'body':body};key=digest(sig);target=api.cache_dir/(key+'.json')
                if target.exists():return original_request(api,provider,model,body,operation=operation,stage=stage,cache=cache)
                if operation=='batchEmbedContents':response={'embeddings':[{'values':[2.0]+[0.0]*767} for r in body['requests']]}
                else:
                    value={'simplemem_requirements':{'required_info':[],'minimal_queries_needed':1},'simplemem_query_plan':{'queries':['Alice']},
                        'simplemem_query_analysis':{'keywords':["John's family"],'persons':[],'entities':[],'time_expression':None,'location':None},
                        'simplemem_completeness':{'assessment':'complete'}}[stage]
                    response={'candidates':[{'content':{'parts':[{'text':json.dumps(value)}]},'finishReason':'STOP'}],'usageMetadata':{}}
                save_new(target,{'request':sig,'stage':stage,'response':response})
                append(api.run_dir/'api_calls.jsonl',{'timestamp_utc':now(),'request_hash':key,'stage':stage,'context':api.context,
                    'provider':provider,'model':model,'operation':operation,'attempt':1,'status':200,'cache_hit':False,
                    'usage':response.get('usageMetadata',{}),'resolved_model':response.get('modelVersion')})
                return response
            with (patch.object(providers,'load_credentials',side_effect=AssertionError('vault')),patch.object(socket.socket,'connect',side_effect=AssertionError('network')),
                 patch.object(providers.ModelAPI,'request',fake_request)):
                api=BaselineAPI(old/'work/api_method',context=context,keys={'fixture':'offline'})
                capture=controller.base.capture_stage(old/'results/c0/retrieval/c0_q0.json',lambda:frozen.retrieve('What about Alice?',state,api),api)
                self.assertEqual(capture['status'],'failed');self.assertIn('Syntax Error',capture['error']['message'])
                log=next((bank/'retrieval').glob('*.requests.jsonl'));records={}
                for rawpath in (old/'work/api_method/api_cache').glob('*.json'):
                    key=rawpath.stem;raw=read(rawpath);intent=old/'work/api_method/request_intents'/(key+'.json')
                    records[key]={'signature':raw['request'],'raw':controller.binding(rawpath),'intent':controller.binding(intent),'raw_value':raw,
                        'logical_consumers':[e for e in api.events if e['request_hash']==key]}
                native=audit.audit_native(log,api.events,records,context)
                q={'query_id':'c0_q0','conversation':0,'question':'What about Alice?'}
                cert=audit.failure_certificate(q,capture,native['prefix'],records)
                case={'query_id':'c0_q0','conversation':0,'prefix':native['prefix'],'certificate':cert,'native_log':controller.binding(log)}
                registry={key:{k:r[k] for k in ('signature','raw','intent','logical_consumers')} for key,r in records.items()}
                oldinv=adapter._inventory(bank/'lancedb');copied=adapter.copy_terminal_bank(bank/'state.json',new/'work/native/c0/simplemem',state_sha256=sha(bank/'state.json'),database_inventory=oldinv)
                proof={'failed_retrievals':[case],'method_request_registry':registry,'banks':{'0':{'bank_dir':str(bank),'state':controller.binding(bank/'state.json'),'database_integrity':state['database_integrity'],'lancedb_inventory':oldinv}}}
                proof_path=root/'proof.json';save_new(proof_path,proof)
                admitted={'run_dir':str(new/'work/api_method'),'query_ids':['c0_q0'],'executing':True,'case_sha256':{'c0_q0':digest(case)},'registry_sha256':digest(registry)}
                api2=prefix.ContinuationAPI(new/'work/api_method',admission=admitted,case=case,registry=registry,context=context,keys={'fixture':'offline'})
                cp=new/'results/c0/retrieval/c0_q0.json'
                result=controller.base.capture_stage(cp,lambda:adapter.retrieve('What about Alice?',copied,api2),api2)
                self.assertEqual(result['status'],'complete',result.get('error'))
                m={'schema':controller.SCHEMA,'self_root':str(new),'parent_root':str(old),'proof':controller.binding(proof_path),'code_sha256':{},'upstream_sha256':{},'failed_query_ids':['c0_q0']}
                with patch.object(controller,'NEW',new),patch.object(controller,'OLD',old):
                    report=controller.audit_sidecar(m,proof)
                    self.assertEqual(report['status'],'passed');self.assertEqual(len(report['new_physical_requests']),1)
                    # FTS or any copied DB byte change must fail before accepting text outputs.
                    item=next(p for p in (new/'work/native/c0/simplemem/lancedb').rglob('*') if p.is_file());before=item.read_bytes();item.write_bytes(before+b'changed')
                    with self.assertRaisesRegex(ValueError,'database bytes'):controller.audit_sidecar(m,proof)

if __name__=='__main__':unittest.main()
