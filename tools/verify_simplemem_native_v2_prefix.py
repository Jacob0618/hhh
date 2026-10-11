"""Author preflight: real copied terminal banks, every old prefix, no new API.

Stops each native query at its first genuinely unseen request. All copied banks
and native logs stay in a fresh research directory for independent inspection.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import socket
import requests
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import run_five_baselines_simplemem_native_v2 as controller
from tools import baselines_simplemem_native_v2 as adapter
from tools import audit_simplemem_terminal_v2 as admission
from tools.simplemem_prefix_api_v2 import PrefixPolicy
from tools.baselines_common_v1 import BaselineAPI,BaselineFatal,read,sha,save_new
from jev_mem import providers
from jev_mem.sdk20_cached_api import digest

class FirstUnseen(BaseException):pass
class OfflineAPI(BaselineAPI):
    def __init__(self,case,registry):
        self.policy=PrefixPolicy(case,registry);self.registry=registry;self.events=[];self.context=self.policy.context;self.keys={};self.next_request=None
    def scoped(self,**context):
        if {**self.context,**context}!=self.context:raise BaselineFatal('Offline context changed')
        return self
    def authorize_native_compatibility(self,kind,details):return self.policy.authorize_native_compatibility(kind,details)
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        decision=self.policy.before(signature,stage);key=decision['request_hash']
        if decision['mode']=='first_unseen_request':
            self.next_request=decision;raise FirstUnseen(key)
        origin=self.registry[key];raw=read(origin['raw']['path'])
        if sha(origin['raw']['path'])!=origin['raw']['sha256'] or raw['request']!=signature:raise AssertionError('Bound old raw changed')
        self.events.append({'request_hash':key,'stage':stage,**{k:signature[k] for k in ('provider','model','operation')},'context':self.context,
            'source':'existing_run_cache','cache_path':origin['raw']['path'],'cache_sha256':origin['raw']['sha256'],'error_type':None})
        return raw['response']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proof',type=Path,default=controller.DEFAULT_PROOF);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=adapter._safe_tree(args.output)
    if not out.is_relative_to(controller.ROOT/'research') or out.exists():raise ValueError('Fresh research-only output required')
    proof=read(args.proof);before=admission.inventory(controller.OLD)
    if before!=proof['original_inventory']:raise ValueError('Actual terminal source inventory changed')
    started=time.monotonic();counts={'network_attempts':0,'credential_reads':0}
    # LanceDB creates the Windows asyncio self-pipe during import. Initialize
    # only that declared runtime before blocking all network APIs; no DB opened.
    import lancedb
    def no_network(*args,**kwargs):counts['network_attempts']+=1;raise AssertionError('Network blocked')
    def no_vault(*args,**kwargs):counts['credential_reads']+=1;raise AssertionError('Credentials blocked')
    qs={q['query_id']:q for q in proof['questions']};results=[];states={};code=controller.code_hashes(read(proof['parent_manifest']['path']))
    with patch.object(socket.socket,'connect',no_network),patch.object(socket.socket,'connect_ex',no_network),patch.object(socket,'create_connection',no_network),patch.object(requests.Session,'request',no_network),patch.object(providers,'load_credentials',no_vault):
        for ci in sorted({c['conversation'] for c in proof['failed_retrievals']}):
            bank=proof['banks'][str(ci)]
            states[ci]=adapter.copy_terminal_bank(bank['state']['path'],out/'native'/('c'+str(ci))/'simplemem',state_sha256=bank['state']['sha256'],database_inventory=bank['lancedb_inventory'])
        for case in proof['failed_retrievals']:
            qid=case['query_id'];api=OfflineAPI(case,proof['method_request_registry']);stream=io.StringIO();state=states[case['conversation']]
            try:
                with redirect_stdout(stream):result=adapter.retrieve(qs[qid]['question'],state,api)
                status='completed_with_only_old_raw';result_sha=digest(result)
            except FirstUnseen:
                status='stopped_before_first_unseen_request';result_sha=None
            except BaseException as exc:
                save_new(out/'failure.json',{'status':'failed','query_id':qid,'error':{'type':type(exc).__name__,'message':str(exc)},'policy':api.policy.__dict__,'native_stdout':stream.getvalue()})
                raise
            policy=api.policy.report()
            logs=sorted((Path(state['work_dir'])/'retrieval').glob('*.requests.jsonl'))
            native_log=next(p for p in logs if p.stem.startswith(adapter._load_private()._hash(qs[qid]['question'])+'-') and str(p) not in {r['native_log']['path'] for r in results})
            native=admission.jsonl(native_log)
            original=admission.jsonl(case['native_log']['path'])
            if [{k:v for k,v in x.items() if k!='utc'} for x in native[:len(original)]]!=[{k:v for k,v in x.items() if k!='utc'} for x in original]:
                raise AssertionError('Actual complete original native request/response prefix differs')
            record={'query_id':qid,'status':status,'policy':policy,'next_unseen_request':api.next_request,
                'native_log':{'path':str(native_log),'sha256':sha(native_log)},'native_prefix_full_semantics_equal':True,
                'completed_result_sha256':result_sha,'native_stdout':stream.getvalue()}
            results.append(record);print(qid+': '+status,flush=True)
        copied=[]
        for ci,state in states.items():
            path=Path(state['work_dir']);lineage=read(path/'terminal_bank_lineage.json')
            actual=adapter._inventory(path/'lancedb')
            if actual!=lineage['copied_database_inventory']:raise AssertionError('Native retrieval mutated copied database')
            copied.append({'conversation':ci,'lineage':{'path':str(path/'terminal_bank_lineage.json'),'sha256':sha(path/'terminal_bank_lineage.json')},'database_before_after_equal':True,'database_files':len(actual)})
    after=admission.inventory(controller.OLD)
    if after!=before or counts!={'network_attempts':0,'credential_reads':0}:raise AssertionError('Offline source immutability/network contract failed')
    if code!=controller.code_hashes(read(proof['parent_manifest']['path'])):raise AssertionError('Candidate changed during actual preflight')
    report={'schema':'simplemem-native-v2-actual-prefix-author-preflight-v1','status':'passed','scope':'author_real_native_prefix_preflight_not_independent_approval',
        'proof':{'path':str(args.proof.resolve()),'sha256':sha(args.proof)},'code_sha256':code,'script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'queries':results,'cohort':len(results),'failure_kinds':dict(Counter(c['certificate']['kind'] for c in proof['failed_retrievals'])),
        'copied_banks':copied,'original_inventory_sha256':digest(before),'original_files':len(before),'original_inventory_before_after_equal':True,
        'new_model_calls':0,'new_corpus_embedding_calls':0,**counts,'production_writes':0,
        'runtime_initialization':'lancedb imported before hard network block only for Windows asyncio self-pipe; no DB opened before guard',
        'elapsed_seconds':time.monotonic()-started}
    save_new(out/'report.json',report);print(json.dumps({'status':'passed','report':str(out/'report.json'),'queries':len(results),'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main()
