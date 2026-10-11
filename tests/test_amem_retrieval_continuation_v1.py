"""Offline one-request contract fixtures. No real encoder, network or credential read."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,FunctionType
import unittest
from unittest.mock import patch

from tools import run_five_baselines_amem_retrieval_v1 as entry
from tools.baselines_common_v1 import BaselineFatal,read,sha,save_new


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.final=self.root/'final';self.owner=self.root/'owner'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry,'FINAL',self.final));self.stack.enter_context(patch.object(entry,'WORK_OWNER',self.owner))
        self.stack.enter_context(patch('jev_mem.providers.load_credentials',side_effect=AssertionError('credential access forbidden')))
        self.stack.enter_context(patch('requests.sessions.Session.post',side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch('jev_mem.providers.time.sleep'))
        cert=entry.admission.DEFAULT_QUALIFICATION;qualification=read(cert)
        state=entry.SOURCE/'work/native/c2/state.json';self.state=read(state)
        parent=read(entry.SOURCE/'run_manifest.json');self.q=next(q for q in parent['questions'] if q['query_id']==entry.QID)
        self.original_call=read(qualification['native_call_path'])
        self.proof={'query_id':entry.QID,'replacement':qualification,'replacement_certificate':entry.admission.binding(cert),
                    'original_failure':{'raw':{'path':qualification['raw_path'],'sha256':qualification['raw_sha256']},
                                        'native_call':{'path':qualification['native_call_path'],'sha256':qualification['native_call_sha256']}},
                    'missing_question':self.q,'bank':{'state':entry.admission.binding(state),
                        'embeddings':entry.admission.binding(state.parent/'cache/embeddings.npy'),
                        'memory_sha256':self.state['memory_sha256'],'notes':663,'sessions':32}}
        self.proof_path=self.root/'proof.json';save_new(self.proof_path,self.proof)
        self.m={'query_id':entry.QID,'work_owner':str(self.owner),'proof':entry.admission.binding(self.proof_path)}
        self.count=0

    def api(self):
        return entry.ReplacementAPI(self.owner/'work/api_method',proof=self.proof,context=entry.CONTEXT,
                                    keys={'GEMINI_API_KEY':'offline-test-secret-value'})

    def response(self,status=200,text='{"keywords":"flood cosmos area"}'):
        return SimpleNamespace(status_code=status,ok=status==200,headers={},json=lambda:{
            'modelVersion':'gemini-2.5-flash','candidates':[{'index':0,'finishReason':'STOP',
              'content':{'role':'model','parts':[{'text':text}]}}],
            'usageMetadata':{'promptTokenCount':80,'candidatesTokenCount':8,'totalTokenCount':88}})

    def post(self,*args,**kwargs):
        self.assertTrue((self.owner/'work/one_use_retrieval_replacement.json').exists())
        self.count+=1
        return self.response()

    def generate(self,api):
        req=self.original_call['request']
        return api.generate(req['prompt'],system=req['system'],json_output=True,max_output=16384,stage=req['stage'])

    def complete(self,*,retry_status=None):
        api=self.api()
        def post(*args,**kwargs):
            self.assertTrue((self.owner/'work/one_use_retrieval_replacement.json').exists());self.count+=1
            return self.response(retry_status if retry_status is not None and self.count==1 else 200)
        api.local.session=SimpleNamespace(post=post)
        native=self.owner/'native/c2';native.mkdir(parents=True)
        state=deepcopy(self.state);state['work_dir']=str(native)
        controller=entry.amem._Controller(api,native)
        classes=entry.amem._runtime._classes(None,controller);system=classes['AgenticMemorySystem']()
        for note in self.state['notes']:
            obj=classes['MemoryNote'].__new__(classes['MemoryNote']);obj.__dict__.update(deepcopy(note));system.memories[obj.id]=obj
        system.retriever.search=lambda query,k:list(range(10))
        fn=entry.amem._runtime._retrieve
        derived=FunctionType(fn.__code__,{**fn.__globals__,'_restore':lambda *args:(system,controller,classes)},fn.__name__,fn.__defaults__,fn.__closure__)
        key=entry.amem._v1._sha({'question':self.q['question'],'memory_sha256':state['memory_sha256']})
        capture=self.owner/'results/c2/retrieval'/(entry.QID+'.json')
        result=entry.base.capture_stage(capture,lambda:derived(self.q['question'],state,api,key),api)
        self.assertEqual(result['status'],'complete',result.get('error'))
        return api

    def audit(self):return entry.audit_sidecar(self.m,self.proof,self.final,self.owner)

    def test_complete_original_native_and_full_audit(self):
        self.complete();report=self.audit()
        self.assertEqual(report['logical_replacements'],1);self.assertEqual(report['physical_attempts'],1)
        self.assertEqual(self.count,1)

    def test_original_529_retry_then_success_is_one_logical_request(self):
        self.complete(retry_status=529);self.assertEqual(self.audit()['physical_attempts'],2)

    def test_marker_precedes_send_and_prevents_new_instance_or_second_request(self):
        api=self.api();api.local.session=SimpleNamespace(post=self.post)
        self.generate(api)
        with self.assertRaises(BaselineFatal):self.generate(api)
        with self.assertRaises(BaselineFatal):self.api()
        self.assertEqual(self.count,1)

    def test_changed_signature_rejected_before_marker_or_send(self):
        api=self.api();api.local.session=SimpleNamespace(post=self.post)
        req=self.original_call['request']
        with self.assertRaises(BaselineFatal):api.generate(req['prompt']+' ',system=req['system'],json_output=True,stage=req['stage'])
        self.assertEqual(self.count,0);self.assertFalse(api.marker.exists())

    def test_wrong_scope_directory_or_retry_policy_before_credential(self):
        for kw in ({'run_dir':self.root/'other'},{'context':{**entry.CONTEXT,'query_id':'c2_q41'}},{'retries':3}):
            args={'run_dir':self.owner/'work/api_method','proof':self.proof,'context':entry.CONTEXT,**kw}
            with self.assertRaises(BaselineFatal):entry.ReplacementAPI(**args)
        self.assertFalse(self.owner.exists())

    def test_invalid_response_cannot_retry_logical_request(self):
        api=self.api()
        def post(*args,**kwargs):self.count+=1;return self.response(text='{"keywords":"unterminated')
        api.local.session=SimpleNamespace(post=post)
        with self.assertRaises(BaseException):self.generate(api)
        with self.assertRaises(BaselineFatal):self.generate(api)
        self.assertEqual(self.count,1);self.assertTrue(api.marker.exists())

    def test_unknown_transport_keeps_quota_and_original_three_physical_attempts(self):
        import requests
        api=self.api()
        def post(*args,**kwargs):self.count+=1;raise requests.Timeout('offline known test timeout')
        api.local.session=SimpleNamespace(post=post)
        with self.assertRaises(BaselineFatal):self.generate(api)
        with self.assertRaises(BaselineFatal):self.api()
        self.assertEqual(self.count,3)

    def mutate(self,path,change):
        original=path.read_bytes();value=read(path);change(value);path.write_text(json.dumps(value),encoding='utf8')
        try:
            with self.assertRaises((ValueError,BaselineFatal)):self.audit()
        finally:path.write_bytes(original)

    def test_changed_marker_parser_native_or_result_rejected(self):
        self.complete();api=self.owner/'work/api_method';native=self.owner/'native/c2'
        self.mutate(self.owner/'work/one_use_retrieval_replacement.json',lambda v:v.update(logical_limit=2))
        self.mutate(api/'parser_lineage'/(entry.KEY+'.json'),lambda v:v.update(parsed_payload_sha256='0'*64))
        self.mutate(next((native/'calls').glob('*.json')),lambda v:v['parsed_response'].update(keywords='altered'))
        self.mutate(next((native/'retrieval').glob('*.json')),lambda v:v.update(context='altered'))
        self.mutate(next((native/'retrieval').glob('*.json')),lambda v:v['evidence'][0].update(via='altered'))
        self.assertEqual(self.audit()['status'],'passed')

    def test_duplicate_consumer_and_extra_raw_are_rejected(self):
        self.complete();api=self.owner/'work/api_method';log=api/'logical_events.jsonl';original=log.read_bytes()
        log.write_bytes(original+original)
        with self.assertRaises(ValueError):self.audit()
        log.write_bytes(original);save_new(api/'api_cache/extra.json',{})
        with self.assertRaises(ValueError):self.audit()

    def test_physical_identity_and_terminal_metadata_must_match_raw(self):
        self.complete();path=self.owner/'work/api_method/api_calls.jsonl';original=path.read_bytes()
        for change in (lambda row:row.update(context={**entry.CONTEXT,'query_id':'c2_q41'}),
                       lambda row:row.update(attempt=2),lambda row:row.update(resolved_model='other'),
                       lambda row:row.update(usage={}),lambda row:row.update(status=503)):
            value=json.loads(original);change(value);path.write_text(json.dumps(value)+'\n',encoding='utf8')
            with self.assertRaises(ValueError):self.audit()
            path.write_bytes(original)

    def test_shared_qa_provider_identity_precedes_execution_or_credentials(self):
        with patch.object(entry,'manifest',return_value=({},{})),patch.object(entry.base,'BaselineAPI',object()):
            with self.assertRaisesRegex(ValueError,'QA provider binding'):entry.run()
        self.assertFalse(self.owner.exists());self.assertFalse(self.final.exists())

    def test_inventory_excludes_only_exact_two_new_subtrees(self):
        source=self.root/'source';final=source/'retrieval_continuation_v1';work=source/'work/retrieval_continuation_v1'
        for path in (source/'original.json',source/'retrieval_continuation_v10/not-excluded.json',final/'new.json',work/'new.json'):
            save_new(path,{'value':1})
        with patch.multiple(entry.admission,SOURCE=source,FINAL=final,WORK_OWNER=work):
            inventory=entry.admission.source_inventory()
            self.assertEqual(set(inventory),{str((source/'original.json').resolve()),str((source/'retrieval_continuation_v10/not-excluded.json').resolve())})
            (source/'original.json').write_text('{}',encoding='utf8')
            self.assertNotEqual(entry.admission.source_inventory(),inventory)

    def test_manifest_review_must_bind_actual_core_hashes_and_proof(self):
        parent={'code_sha256':{},'upstream_sha256':{}}
        value={'schema':entry.REVIEW_SCHEMA,'status':'passed','scope':'single_query_core_implementation_and_actual_admission',
               'proof':entry.admission.binding(self.proof_path),'code_sha256':entry.code_hashes(parent),
               'upstream_sha256':{},'archive_cleanup_downstream_authorized':False,
               'evidence_sha256':{str(self.proof_path):sha(self.proof_path)}}
        review=self.root/'review.json';save_new(review,value)
        with patch.object(entry,'ROOT',self.root),patch.object(entry,'code_hashes',return_value=value['code_sha256']),patch.object(entry.admission,'bound',side_effect=lambda v:Path(v['path'])):
            self.assertEqual(entry.check_review(review,self.proof_path,parent)['sha256'],sha(review))
            value['code_sha256']={};review.write_text(json.dumps(value),encoding='utf8')
            with self.assertRaises(ValueError):entry.check_review(review,self.proof_path,parent)

    def test_full1540_union_with_real_method_audit_and_real_qa_functions(self):
        from tests.test_amem_retrieval_union_v1 import UnionTests
        from tools import seal_amem_retrieval_continuation_v1 as union
        fixture=UnionTests();fixture.setUp()
        try:
            # Replace only the earlier factory's method stub with the real entry.
            # All replay, method audit, QA audit, formatter and aggregation execute.
            fixture.entry_patch.stop()
            self.owner=fixture.work;self.final=fixture.final
            with patch.object(entry,'WORK_OWNER',self.owner),patch.object(entry,'FINAL',self.final):
                self.complete()
                ret_path=self.owner/'results/c2/retrieval'/(entry.QID+'.json');ret=read(ret_path)
                count=[0]
                def qa_api(folder,**kwargs):
                    api=entry.BaselineAPI(folder,keys={'GEMINI_API_KEY':'offline-test-secret-value'},**kwargs)
                    def post(*args,**kw):
                        count[0]+=1
                        return self.response(text=str(self.q['reference']) if count[0]==1 else '[[yes]]')
                    api.local.session=SimpleNamespace(post=post)
                    return api
                with patch.object(entry.base,'BaselineAPI',side_effect=qa_api):
                    entry.base.qa_one(self.owner,'amem',self.q,ret['value'],phase='firstpass')
                self.assertEqual(count[0],2)
                union.copy_qa('firstpass')
                for p in (ret_path,ret_path.with_suffix('.intent.json')):entry.copy_new(p,self.final/'results/technical_retrieval'/p.name)
                fixture.q.clear();fixture.q.update(self.q)
                fixture.proof.update({k:deepcopy(v) for k,v in self.proof.items() if k!='original_failure'})
                fixture.proof['original_failure'].update(self.proof['original_failure'])
                for name in list(fixture.docs):
                    if Path(name).is_relative_to(fixture.work) or Path(name).is_relative_to(fixture.final):del fixture.docs[name]
                for root in (self.owner,self.final):
                    for p in root.rglob('*'):
                        if p.is_file():fixture.docs[str(p)]=p.read_bytes()
                for record in (self.proof['replacement_certificate'],self.proof['bank']['state'],
                               self.proof['original_failure']['native_call'],self.proof['original_failure']['raw']):
                    fixture.docs[record['path']]=Path(record['path']).read_bytes()
                fixture.put(fixture.root/'proof.json',fixture.proof)
                fixture.manifest['proof']['sha256']=fixture.hash(fixture.root/'proof.json')
                fixture.put(fixture.parent/'run_manifest.json',{'questions':fixture.qs,'conversation_sources':fixture.routes})
                fixture.manifest['parent_manifest']['sha256']=fixture.hash(fixture.parent/'run_manifest.json')
                fixture.finalize_plan()
                result=union.reconstruct(fixture.manifest,fixture.proof,view=fixture.view(),created_utc='fixture')
                self.assertEqual(result['method_request_audit.json']['status'],'passed')
                self.assertEqual(result['method_request_audit.json']['physical_attempts'],1)
                self.assertEqual(len(result['raw_request_audit.json']['requests']),2)
                self.assertEqual(result['summary.json']['firstpass']['reader_valid'],1539)
                self.assertEqual(result['summary.json']['effective']['reader_valid'],1540)
                row=next(v for v in result['firstpass_per_query.json'] if v['query_id']==entry.QID)
                self.assertEqual((row['retrieval_status'],row['reader_status'],row['judge_status']),('failed','not_run','not_run'))
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
