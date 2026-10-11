"""Pure offline tag-overlay tests; no real prepare/recovery/retrieval/API."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import prepare_gemini38_hybrid_tag_overlay_v1 as n
from jev_mem.selection import tag_fact

class Capture:
    def score_many(self,state,criteria):
        self.signature={'provider':'jev','model':'jev-1.13.0','operation':'systemone',
            'body':{'model':'jev-1.13.0','state':state,'questions':{k:{'type':'noul','instructions':v}for k,v in criteria.items()}}}
        self.scores={k:.5 for k in criteria};return self.scores

class Overlay(unittest.TestCase):
    def setUp(self):
        n.idx.HASHES.clear();self.tmp=tempfile.TemporaryDirectory(prefix='jev_tag_overlay61_')
        self.root=Path(self.tmp.name);self.run=self.root/'parent';self.ns=self.root/'recovery';self.ns.mkdir()
        (self.run/'retrieval').mkdir(parents=True)
        self.context='Frozen original source conversation.'
        self.fact={'id':n.FAILED_FACT_ID,'text':'A recovered-tag target fact.','input_scope':{'session_id':'session_1'}}
        bank=self.run/'bank.json';n.save_new(bank,{'facts':[self.fact]});self.bank_ref=n.idx.pointer(bank)
        cap=Capture();self.tag=tag_fact({**self.fact,'source_context':self.context},cap)
        self.signature=cap.signature;self.hash=hashlib.sha256(json.dumps(self.signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        n.save_new(self.run/'retrieval/run_manifest.json',{'scientific':'frozen'})
        n.save_new(self.run/'retrieval/tag_status.json',{'status':'failed','terminal':True})
        self.scientific={'protocol_sha256':'original','banks':[{'conversation':2,'arm':'session','bank':self.bank_ref}]}
        self.failed={'schema':n.SCHEMA,'protocol_sha256':'original','conversation':2,'fact_id':n.FAILED_FACT_ID,'bank':self.bank_ref,
            'fact_text_sha256':n.digest(self.fact['text']),'source_context_sha256':n.digest(self.context),
            'source_scope':'whole_original_session_unverified_input_scope','status':'failed','error_type':'RuntimeError'}
        n.save_new(n.failed_path(self.run),self.failed)
        self.intent=self.root/'old_intent.json';self.intent.write_text(json.dumps({'signature':self.signature}))
        self.raw=self.ns/'raw.json';self.raw.write_text(json.dumps({'request':self.signature,'response':{'answers':{k:{'noul':v}for k,v in cap.scores.items()}}}))
        self.manifest={'schema':'gemini38-fact-tag-exact-recovery-v1','original_protocol_sha256':'original',
            'source_manifest':n.idx.pointer(self.run/'retrieval/run_manifest.json'),
            'failed_tag':n.idx.pointer(n.failed_path(self.run)),'original_request_hash':self.hash,
            'original_request_intent':n.idx.pointer(self.intent),'review':{'path':'boundreview','sha256':'r'},
            'policy':{'retries':0,'one_dispatch':True,'cache_roots':[]}}
        row={**self.failed,'status':'complete','result':self.tag}
        row.pop('error_type')
        self.terminal={'schema':self.manifest['schema'],'status':'complete','original_failed_tag':self.manifest['failed_tag'],
            'original_request_hash':self.hash,'new_request_hash':self.hash,'new_raw':n.idx.pointer(self.raw),'fact_tag_record':row}
        n.save_new(self.ns/'recovery_manifest.json',self.manifest);n.save_new(self.ns/'result.json',self.terminal)
        self.patches=[patch.object(n,'REQUEST_HASH',self.hash),patch.object(n,'_recovery_validate',side_effect=lambda ns:(self.manifest,self.terminal,{})),
            patch.object(n,'_execution_codes',return_value={'fake':'sha'}),patch.object(n.base,'session_contexts',return_value={'session_1':self.context}),
            patch.object(n.base,'manifest',return_value=(self.scientific,{'pack':True})),
            patch.dict(n._scope,{'session_contexts':lambda ci:{'session_1':self.context}})]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def test_prepare_check_preserve_original_failed_bytes(self):
        old=n.failed_path(self.run).read_bytes();m=n.prepare(self.run,self.ns)
        side,row=n.validate_overlay(self.run,self.scientific)
        self.assertEqual(row['status'],'complete');self.assertEqual(n.failed_path(self.run).read_bytes(),old)
        self.assertFalse(n.idx.verify(n.failed_path(self.run))['status']=='complete')
        self.assertEqual(side['recovery_namespace'],str(self.ns));self.assertEqual(m['model_calls'],0)
        with self.assertRaises(ValueError):n.prepare(self.run,self.ns)
    def test_incomplete_recovery_wrong_policy_identity_and_scores_refused(self):
        for which in ('status','hash','fact','policy','score','protocol'):
            oldm,oldt=copy.deepcopy((self.manifest,self.terminal))
            if which=='status':self.terminal['status']='failed'
            elif which=='hash':self.terminal['new_request_hash']='different'
            elif which=='fact':self.terminal['fact_tag_record']['fact_id']='other'
            elif which=='policy':self.manifest['policy']['retries']=2
            elif which=='score':self.terminal['fact_tag_record']['result']['coarse_scores']['episodic']=.6
            else:self.manifest['original_protocol_sha256']='other'
            with self.subTest(which=which),self.assertRaises(ValueError):n.recovery_record(self.run,self.scientific,self.ns)
            self.manifest,self.terminal=oldm,oldt
    def test_same_result_different_complete_request_refused(self):
        self.terminal['new_raw']=n.idx.pointer(self.raw)
        original=json.loads(self.raw.read_text());original['request']['body']['state']='different source context'
        altered=self.ns/'different_raw.json';altered.write_text(json.dumps(original));self.terminal['new_raw']=n.idx.pointer(altered)
        with self.assertRaises(ValueError):n.recovery_record(self.run,self.scientific,self.ns)
    def test_load_tags_only_exact_original_path_uses_original_core(self):
        n.prepare(self.run,self.ns);before=n.failed_path(self.run).read_bytes()
        tags=n._load_tags(self.run,2,{'facts':[self.fact]},self.bank_ref,self.scientific)
        self.assertEqual(tags[self.fact['id']],self.tag);self.assertEqual(before,n.failed_path(self.run).read_bytes())
        other={'id':'another_failed','text':'Different fact.','input_scope':{'session_id':'session_1'}}
        path=self.run/'retrieval/conv_2/tags/another_failed.json'
        n.save_new(path,{**self.failed,'fact_id':other['id'],'fact_text_sha256':n.digest(other['text'])})
        with self.assertRaises(ValueError):n._load_tags(self.run,2,{'facts':[other]},self.bank_ref,self.scientific)
    def test_no_firstpass_retag_stage_and_missing_sidecar_rejected(self):
        with self.assertRaises(ValueError):n.run_stage(self.run,'tag')
        with self.assertRaises(FileNotFoundError):n.manifest(self.run)
    def test_scientific_codeobjects_scope_and_pickle_targets(self):
        for name in ('retrieve_worker','capture_current','validate_native_capture','_selection_jobs'):
            self.assertIs(getattr(n,name).__code__,getattr(n.base,name).__code__)
            self.assertIs(getattr(n,name).__globals__,n._scope)
        self.assertIs(pickle.loads(pickle.dumps(n.retrieve_worker)),n.retrieve_worker)
        self.assertIsNot(n.base._scope,n._scope)
        self.assertIsNot(n.base._scope['_load_tags'],n._load_tags)
    def test_new_selected_source_disclosure_required_no_legacy_fallback(self):
        n.prepare(self.run,self.ns);side,_=n.validate_overlay(self.run,self.scientific)
        good={'schema':n.SOURCE_SCHEMA,'parent_run':str(self.run),'tag_recovery_overlay':n.idx.pointer(n.sidecar_path(self.run)),
            'tag_recovery_execution_protocol_sha256':side['execution_protocol_sha256'],'tag_firstpass_complete':False,'tag_firstpass_failed_tags':1}
        with patch.object(n,'_validate_source_core',return_value={'validated':'scientific_core'})as core:
            n.save_new(n.source_path(self.run),good);self.assertEqual(n.validate_source(n.source_path(self.run)),{'validated':'scientific_core'});core.assert_called_once()
        legacy=self.run/'retrieval/selected_inputs.json';n.save_new(legacy,good)
        with self.assertRaises(ValueError):n.validate_source(legacy)
    def test_original_seal_code_saves_new_filename_with_explicit_overlay(self):
        self.scientific.update(new_execution_config={'path':'config','sha256':'c'},official_metric_provenance={'official':True},code_sha256={'original':'science'})
        n.prepare(self.run,self.ns)
        counts={str(ci):[{}]*154 for ci in range(10)}
        def jobs(run,m,ci,arm):
            return [{'job_id':f'{ci}/{arm}/{j}'}for j in range(616 if arm=='session' else 154)]
        def outcome(path):return {'status':'complete','protocol_sha256':'original','queries':616}
        with patch.dict(n._scope,{'cohort':lambda:(counts,{}),'_selection_jobs':jobs,'verify':outcome}), \
             patch.object(n,'_validate_source_core',side_effect=lambda path:n.idx.verify(path)):
            result=n.seal(self.run)
        self.assertEqual(result['jobs'],10780);self.assertEqual(result['source'],n.idx.pointer(n.source_path(self.run)))
        self.assertFalse((self.run/'retrieval/selected_inputs.json').exists())
        source=n.idx.verify(n.source_path(self.run));self.assertEqual(source['schema'],n.SOURCE_SCHEMA)
        self.assertEqual(source['tag_recovery_overlay'],n.idx.pointer(n.sidecar_path(self.run)))
        self.assertFalse(source['tag_firstpass_complete']);self.assertEqual(source['tag_firstpass_failed_tags'],1)
        with self.assertRaises(ValueError):n.seal(self.run)

if __name__=='__main__':unittest.main()
