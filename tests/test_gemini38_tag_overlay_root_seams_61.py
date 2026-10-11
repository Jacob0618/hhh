"""Independent offline QA/accounting thin-seam tests; all ledgers are fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gemini38_plain_qa_tag_overlay_v1 as qa
from tools import summarize_gemini38_tag_overlay_v1 as summary


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ptr(p): return {'path':str(p),'sha256':sha(p)}
def save(p,obj): p.write_text(json.dumps(obj),encoding='utf8');return ptr(p)


class Snapshot:
    def raw(self,p):return Path(p).read_bytes()
    def read(self,p):return json.loads(self.raw(p))
    def sha(self,p):return sha(p)


class SeamTests(unittest.TestCase):
    def setUp(self):
        self.credentials=patch('jev_mem.providers.load_credentials',side_effect=AssertionError('No credentials allowed'))
        self.credentials.start(); self.addCleanup(self.credentials.stop)

    def test_qa_codeobjects_evaluator_and_original_scope_unchanged(self):
        original=qa.original
        self.assertIsNot(qa._impl,original._impl)
        self.assertIs(qa._impl['_impl'],qa._impl)
        self.assertIs(original._impl['_impl'],original._impl)
        self.assertEqual(original._impl['SCHEMA'],original.SCHEMA)
        self.assertIs(qa._impl['source_validate'],qa.source_validate)
        for name in ('one','run','manifest','reader_call','known_failure','recovery_jobs','prepare_recovery','prepare','_new_manifest'):
            self.assertIs(qa._impl[name].__code__,original._impl[name].__code__)
            self.assertIs(qa._impl[name].__globals__,qa._impl)
        for name in ('read_answer','judge_answer','PlainTextEvaluationAPI','audit_stage'):
            self.assertIs(qa._impl[name],original._impl[name])
        self.assertIs(qa.Gemini38Replay,original.Gemini38Replay)
        self.assertIs(qa._new_audit_stage,original._new_audit_stage)
        self.assertEqual(qa.ARMS,original.ARMS);self.assertEqual(qa.POLICY,original.POLICY)
        self.assertIs(qa.main.__code__,original.main.__code__)
        self.assertIs(qa.main.__globals__['prepare'],qa.prepare)
        self.assertIs(qa.main.__globals__['run'],qa.run)
        self.assertEqual(qa.POLICY['word_limit'],None)
        self.assertEqual(qa.POLICY['qa_max_output_tokens'],8192)
        self.assertEqual(qa.POLICY['thinking_level'],'low')

    def test_source_schema_overlay_and10780_cohort(self):
        for source in ({'schema':qa.original.SOURCE_SCHEMA,'tag_recovery_overlay':{'path':'p'}},
                       {'schema':qa.SOURCE_SCHEMA}, {'schema':qa.SOURCE_SCHEMA,'tag_recovery_overlay':{}}):
            with self.subTest(source=source),patch.object(qa.retrieval,'validate_source',return_value=source):
                with self.assertRaisesRegex(ValueError,'reviewed tag-recovery'):qa.source_validate('fixture')
        jobs=[{'method':arm,'query_id':f'q{i}'}for arm in qa.ARMS for i in range(1540)]
        source={'schema':qa.SOURCE_SCHEMA,'tag_recovery_overlay':{'path':'p','sha256':'s'},'jobs':jobs}
        with patch.object(qa.retrieval,'validate_source',return_value=source),patch.dict(qa._impl,{'_new_manifest':lambda *a,**kw:kw}):
            result=qa.prepare('mock-new','mock-source')
            self.assertEqual(len(result['jobs']),10780);self.assertEqual(result['phase_kind'],'firstpass')
        with patch.object(qa.retrieval,'validate_source',return_value={**source,'jobs':jobs[:-1]}):
            with self.assertRaisesRegex(ValueError,'1540'):qa.prepare('mock-new','mock-source')

    def test_summary_core_globals_scoring_and_new_runner(self):
        for name in ('audit_phase','record_rows','aggregate','readiness','summarize','main'):
            self.assertIs(summary._scope[name].__code__,summary.original._scope[name].__code__)
            self.assertIs(summary._scope[name].__globals__,summary._scope)
            self.assertIs(summary._scope[name].__globals__['runner'],qa)
        self.assertIs(summary._scope['precomputation'],summary.precomputation)
        self.assertIs(summary._scope['ledger'].__globals__['runner'],qa)
        self.assertIsNot(summary._scope,summary.original._scope)
        self.assertIs(summary.original._scope['runner'],qa.original)

    def fixture(self,root):
        ctx={'architecture':'fixture','conversation':2,'phase':'new_fact_typing','fact_id':'failed-fact'}
        common={'provider':'jev','model':'jev-1.13.0','operation':'systemone','stage':'jev_score','context':ctx,'request_hash':'h'}
        good={**common,'cache_hit':False,'status':200,'usage':{'input_tokens':100,'output_tokens':10}}
        unknown={**common,'cache_hit':False,'status':'transport_error','usage_unknown':True}
        oldrows=[good]+[{**unknown,'attempt':i%3+1}for i in range(61)]
        origin=root/'origin.jsonl';origin.write_text(''.join(json.dumps(r)+'\n'for r in oldrows))
        io={**common,'model':'gemini-3.8-flash','context':{'conversation':3,'method':'session'},'request_hash':'io','status':200,'cache_hit':False,'usage':{'promptTokenCount':20,'candidatesTokenCount':2,'totalTokenCount':22}}
        cache={**io,'cache_hit':True,'usage_semantics':'reused_response_no_new_model_usage'}
        ioledger=root/'io.jsonl';ioledger.write_text(''.join(json.dumps(r)+'\n'for r in (io,cache)))
        new={**common,'cache_hit':False,'attempt':1,'status':200,'usage':{'input_tokens':50,'output_tokens':4}}
        newledger=root/'tag_once.jsonl';newledger.write_text(json.dumps(new)+'\n')
        failed=root/'original_failed.json';failedref=save(failed,{'status':'failed','api_events':[common]})
        result={'api_ledger':ptr(newledger),'original_failed_tag':failedref}
        resultpath=root/'result.json';resultref=save(resultpath,result)
        overlay={'recovery_namespace':str(root/'recovery'),'recovery_manifest':{'path':'mock-manifest','sha256':'m'},'recovery_result':resultref}
        overlaypath=root/'overlay.json';overlayref=save(overlaypath,overlay)
        prior={'usage':summary.original.original.measured_usage(oldrows+[io,cache]),'ledger_sources':[ptr(origin)],
            'writer_recovery':{'ledger_sources':[ptr(ioledger)]},'by_model':{},'note':'original61transportunknown and failedtag retained'}
        return {'ctx':ctx,'new':new,'ledger':newledger,'failed':failed,'result':result,'resultpath':resultpath,
                'overlay':overlay,'overlaypath':overlaypath,'source':{'tag_recovery_overlay':overlayref},'prior':prior}

    def execute_fixture(self,f):
        with patch.object(summary.original,'precomputation',return_value=copy.deepcopy(f['prior'])), \
             patch.object(summary.tag_recovery,'validate_recovery',return_value=({},f['result'],[])), \
             patch.object(summary.original.original,'intent_inventory',return_value={'unknown_intent_count':0}):
            return summary.precomputation(Snapshot(),f['source'])

    def test_usage_add_once_writerio_cache_excluded_and61unknown_retained(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));result=self.execute_fixture(f);u=result['usage']
            self.assertEqual(u['physical_attempts'],64);self.assertEqual(u['cache_events'],1)
            self.assertEqual(u['usage_unknown_attempts'],61)
            self.assertEqual(u['token_fields']['prompt']['recorded_token_sum'],170)
            self.assertEqual(u['token_fields']['thinking']['missing_attempts'],64)
            self.assertIsNone(u['total_cost_usd'])
            self.assertEqual(result['fact_tag_recovery']['usage']['physical_attempts'],1)
            self.assertEqual(result['before_fact_tag_recovery_usage']['physical_attempts'],63)
            self.assertEqual(result['writer_recovery'],f['prior']['writer_recovery'])
            self.assertIn('first-pass tag phase was not entirely successful',result['note'])
            self.assertEqual(result['by_model']['jev-1.13.0']['usage_unknown_attempts'],61)

    def test_ledger_duplicate_alias_and_exactidentity_reject(self):
        for kind in ('duplicate_original','tag_alias','wrong_context','wrong_hash','wrong_model','wrong_provider','wrong_operation','wrong_stage'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d))
                if kind=='duplicate_original':f['prior']['ledger_sources']*=2
                elif kind=='tag_alias':f['prior']['ledger_sources'].append(ptr(f['ledger']))
                else:
                    key='request_hash' if kind=='wrong_hash' else kind[6:]
                    bad=copy.deepcopy(f['new']);bad[key]={'conversation':3}if key=='context'else'wrong'
                    f['ledger'].write_text(json.dumps(bad)+'\n');f['result']['api_ledger']=ptr(f['ledger'])
                    f['overlay']['recovery_result']=save(f['resultpath'],f['result'])
                    f['source']['tag_recovery_overlay']=save(f['overlaypath'],f['overlay'])
                with self.assertRaises(summary.original.original.InvalidArtifact):self.execute_fixture(f)

    def test_overlay_result_failedtag_ledger_byte_changes_reject(self):
        for key in ('overlaypath','resultpath','failed','ledger'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));f[key].write_text('{"changed":true}')
                with self.assertRaises(summary.original.original.InvalidArtifact):self.execute_fixture(f)


if __name__=='__main__':unittest.main()
