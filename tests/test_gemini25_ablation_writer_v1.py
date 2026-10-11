"""Focused offline tests: no providers, credentials or production artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gemini25_ablation_writer_v1 as writer
from tools import audit_gemini25_ablation_banks_v1 as audit
from jev_mem.local_bm25_context_v2 import forbid_network
from jev_mem.gemini25_ablation_provider_v1 import MODEL,generation_body,WRITER_MAX_OUTPUT


class WriterTests(unittest.TestCase):
    def test_original_code_objects_and_globals_unchanged(self):
        self.assertIs(writer.write_sessions.__code__,writer.frozen.write_sessions.__code__)
        self.assertIs(writer.open_memory.__code__,writer.frozen.open_memory.__code__)
        self.assertIs(writer.run_write.__code__,writer.frozen.run_write.__code__)
        self.assertIsNot(writer.write_sessions.__globals__,writer.frozen.write_sessions.__globals__)
        self.assertIs(writer.write_sessions.__globals__['Gemini38API'],writer.Gemini25API)
        self.assertIs(writer.write_sessions.__globals__['open_memory'],writer.open_memory)
        self.assertEqual(writer.frozen.open_memory.__globals__['MODEL'],'gemini-3.8-flash')
        self.assertEqual(writer.open_memory.__globals__['MODEL'],'gemini-2.5-flash')
        self.assertIs(audit.check_generation.__code__,audit.frozen.check_generation.__code__)
        self.assertIs(audit.inspect_session.__code__,audit.frozen.inspect_session.__code__)
        self.assertIs(audit.audit.__code__,audit.frozen.audit.__code__)
        self.assertEqual(audit.publish.__code__.co_code,audit.publisher.publish.__code__.co_code)
        self.assertEqual(audit.frozen.check_generation.__globals__['MODEL'],'gemini-3.8-flash')

    def test_prepare_ten_banks_272_sessions_and_reject_resealed_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'data').mkdir()
            shutil.copyfile(writer.ROOT/'data/locomo10.json',root/'data/locomo10.json')
            snapshot=root/'bm25';snapshot.mkdir()
            english=snapshot/'english.txt';english.write_text('the\nand\n',encoding='utf8')
            resource={'snapshot':str(snapshot),'files_sha256':{str(english):writer.sha(english)},'probe':{'fixture':True}}
            run=root/'runs/gemini25_ablation_writer_v1_test'
            with patch.object(writer,'ROOT',root),patch.object(writer,'codes',return_value={'fixture':'code'}),\
                 patch.object(writer,'resources',return_value=resource),\
                 patch.object(writer.sparse,'dependency_versions',return_value={'fixture':'version'}),\
                 patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credentials')),\
                 forbid_network() as guards:
                result=writer.prepare(run)
                self.assertEqual((result['banks'],result['session_count'],result['model_calls']),(10,272,0))
                m=writer.manifest(run)
                self.assertEqual(len(m['inputs']),10)
                self.assertEqual(m['arms'],['session'])
                self.assertEqual(m['generation_config']['thinkingConfig'],{'thinkingBudget':0})
                self.assertEqual(m['shared_qa_arms'],['baseline','r03_original','original_multi_source_closure'])
                self.assertEqual(sum(g.call_count for g in guards),0)
                with self.assertRaises(FileExistsError):writer.prepare(run)
                item=m['inputs'][0];p=Path(item['input']['path'])
                value=writer.verify(p)
                value['envelopes'][0]['observation_date']='2001-01-01'
                value.pop('content_sha256')
                value['content_sha256']=writer.digest(value)
                p.write_text(json.dumps(value),encoding='utf8')
                item['input']=writer.grouped.pointer(p)
                m.pop('protocol_sha256');m['protocol_sha256']=writer.digest(m)
                (run/'run_manifest.json').write_text(json.dumps(m),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'source/date rendering'):
                    writer.manifest(run)

    def test_unresolved_store_is_never_reentered(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);folder=run/'conv_0/session';folder.mkdir(parents=True)
            (folder/'write_intent.json').write_text('{}')
            with patch.object(writer,'manifest',return_value={'protocol_sha256':'fixture'}),\
                 patch.object(writer,'write_sessions',side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(RuntimeError,'Unresolved Writer'):
                    writer.worker(run,0)

    def test_expansion_requires_review_before_any_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            with patch.object(writer,'manifest',return_value={'protocol_sha256':'fixture'}),\
                 patch.object(writer,'development_review',side_effect=ValueError('missing review')):
                with self.assertRaisesRegex(ValueError,'missing review'):writer.worker(run,2)
                self.assertFalse((run/'conv_2/session/write_intent.json').exists())

    def test_raw_audit_accepts_exact_new_model_rejects_old_model(self):
        with tempfile.TemporaryDirectory() as directory:
            native=Path(directory);(native/'api_cache').mkdir()
            env={'ADDITIVE_EXTRACTION_PROMPT':'Fixture original system'}
            prompt='Fixture source session'
            parsed={'memory':[]}
            body=generation_body(prompt,system=env['ADDITIVE_EXTRACTION_PROMPT'],json_output=True,max_output=WRITER_MAX_OUTPUT)
            signature={'provider':'gemini','model':MODEL,'operation':'generateContent','body':body}
            key=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            path=native/'api_cache'/(key+'.json')
            raw={'request':signature,'stage':'mem0_extract','response':{'modelVersion':MODEL,'candidates':[{
                'finishReason':'STOP','content':{'parts':[{'text':json.dumps(parsed)}]}}]}}
            path.write_text(json.dumps(raw),encoding='utf8')
            journal={'builder_call_count':1,'session_id':'session_1','llm_calls':[{'kwargs':{'messages':[
                {'role':'system','content':env['ADDITIVE_EXTRACTION_PROMPT']},{'role':'user','content':prompt}]},
                'response':json.dumps(parsed),'parsed_memory':[]}]}
            event={'stage':'mem0_extract','status':200,'provider':'gemini','model':MODEL,'operation':'generateContent',
                   'request_hash':key,'context':{'session_id':'session_1','conversation':0,'method':'session'}}
            with forbid_network() as guards:
                extracted,ref=audit.check_generation(journal,[event],native,prompt,env,0,'session')
                self.assertEqual(extracted,[])
                self.assertEqual(Path(ref['path']),path)
                event['model']='gemini-3.8-flash'
                with self.assertRaisesRegex(ValueError,'model/operation differs'):
                    audit.check_generation(journal,[event],native,prompt,env,0,'session')
                self.assertEqual(sum(g.call_count for g in guards),0)

    def test_bank_scope_maps_exact_source_turns_and_rejects_outside_id(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);folder=run/'conv_0/session'
            envs=list(writer.session_envelopes(writer.load_data()[0]))
            writer.grouped.save_new(folder/'group_inputs.json',{'envelopes':envs})
            ref=writer.grouped.pointer(folder/'group_inputs.json')
            fact={'id':'fixture','text':'fixture fact','input_scope':{'source_turn_ids':envs[0]['source_turn_ids']}}
            bank={'complete':True,'conversation':0,'arm':'session','protocol_sha256':'fixture',
                  'group_inputs':ref,'facts':[fact]}
            writer.grouped.save_new(folder/'bank.json',bank)
            with patch.object(writer,'manifest',return_value={'protocol_sha256':'fixture'}):
                sources=writer.source_support_inputs(run,0)
                self.assertEqual(sources['rawfacts']['fixture']['input_scope'],fact['input_scope'])
                self.assertEqual(len(sources['turns']),sum(len(e['source_turn_ids']) for e in envs))
                self.assertEqual(sources['group_inputs'],ref)
                bad=copy.deepcopy(writer.verify(folder/'bank.json'));bad.pop('content_sha256')
                bad['facts'][0]['input_scope']['source_turn_ids']=['outside-current-conversation']
                bad['content_sha256']=writer.digest(bad)
                (folder/'bank.json').write_text(json.dumps(bad),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'current-source scope differs'):
                    writer.source_support_inputs(run,0)


if __name__=='__main__':unittest.main()
