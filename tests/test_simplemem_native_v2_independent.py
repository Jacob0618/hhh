"""Independent native-scope adversarial fixtures, never opening any database."""
import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace, CodeType
import unittest
from unittest.mock import patch

from tools import baselines_simplemem_native_v2 as candidate
from tools import baselines_simplemem_v1 as original
from tools.baselines_common_v1 import BaselineFatal
from tools import simplemem_prefix_api_v2 as prefix
from jev_mem.sdk20_cached_api import digest


class Audit:
    def __init__(self): self.rows=[]
    def record(self,v): self.rows.append(v)


class Gate:
    def __init__(self,kind,accept):self.kind,self.accept=kind,accept;self.calls=[]
    def authorize_native_compatibility(self,kind,details):
        self.calls.append((kind,deepcopy(details)))
        return kind==self.kind and self.accept(details)


@contextmanager
def active(gate):
    token=candidate._ACTIVE.set(gate)
    try:yield
    finally:candidate._ACTIVE.reset(token)


def fingerprint(code):
    if not isinstance(code,CodeType):return code
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
        tuple(fingerprint(x) for x in code.co_consts),code.co_argcount,code.co_kwonlyargcount)


def nested_codes(code):
    result={}
    for c in code.co_consts:
        if isinstance(c,CodeType):
            result[c.co_name]=c
            for k,v in nested_codes(c).items():result[c.co_name+'.'+k]=v
    return result


class NativeScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen_before=(original._NATIVE,original._SourceBridge,original._LLMBridge.extract_json)
        cls.module=candidate._load_private();cls.native=cls.module._load_native()

    def test_frozen_module_objects_not_mutated_and_success_code_unchanged(self):
        self.assertEqual(self.frozen_before,(original._NATIVE,original._SourceBridge,original._LLMBridge.extract_json))
        changed=[];checked=0
        for relative in ('database/vector_store','hybrid_retriever','memory_builder','answer_generator'):
            path=original.SOURCE_ROOT/('simplemem/core/'+relative+'.py')
            baseline=ast.fix_missing_locations(original._SourceBridge().visit(ast.parse(path.read_text('utf8'))))
            private=ast.fix_missing_locations(self.module._SourceBridge().visit(ast.parse(path.read_text('utf8'))))
            left=nested_codes(compile(baseline,str(path),'exec'));right=nested_codes(compile(private,str(path),'exec'))
            self.assertEqual(set(left),set(right))
            for name in left:
                # Class objects contain the differing method, so check every leaf method/code.
                if any(isinstance(x,CodeType) for x in left[name].co_consts):continue
                checked+=1
                if fingerprint(left[name])!=fingerprint(right[name]):changed.append(relative+':'+name)
        self.assertGreater(checked,45)
        self.assertEqual(sorted(changed),['database/vector_store:VectorStore.keyword_search','hybrid_retriever:HybridRetriever._analyze_query'])

    def test_keyword_exact_gate_retains_native_empty_and_rejects_unknown(self):
        fn=self.native.vector_store.VectorStore.keyword_search
        for message in ("Syntax Error: John's family", "Field does not exist: 'CS'"):
            with self.subTest(message=message):
                gate=Gate('keyword_search',lambda d:d['exception']=={'type':'ValueError','message':message})
                def fail(*args,**kw):raise ValueError(message)
                store=SimpleNamespace(backend=SimpleNamespace(count=lambda:1,keyword_search=fail),_results_to_entries=lambda x:x)
                with active(gate),redirect_stdout(io.StringIO()):self.assertEqual(fn(store,["John's family"],5),[])
                self.assertEqual(len(gate.calls),1)
                def io_fail(*args,**kw):raise OSError('disk unavailable')
                store.backend.keyword_search=io_fail
                with active(gate),self.assertRaises(BaselineFatal):fn(store,["John's family"],5)
        with self.assertRaises(BaselineFatal):candidate._keyword_exception(ValueError('arbitrary'),['x'])

    def test_query_analysis_three_original_attempts_then_default(self):
        responses=[];raw='{"keywords": [broken]}'
        class API:
            def generate(self,prompt,**kw):responses.append((prompt,kw));return raw
        gate=Gate('query_analysis_json',lambda d:d['response_text_sha256']==hashlib.sha256(raw.encode()).hexdigest() and d['attempt'] in range(3))
        bridge=self.module._LLMBridge(API(),Audit())
        with active(gate),redirect_stdout(io.StringIO()):
            value=self.native.hybrid_retriever.HybridRetriever._analyze_query(SimpleNamespace(llm_client=bridge),'Which day?')
        self.assertEqual(value,{'keywords':['Which day?'],'persons':[],'time_expression':None,'location':None,'entities':[]})
        self.assertEqual(len(responses),3);self.assertTrue(all(x==responses[0] for x in responses))
        self.assertEqual([d['attempt'] for _,d in gate.calls],[0,1,2])

    def test_query_analysis_non_json_schema_failure_not_defaulted(self):
        class API:
            def generate(self,*a,**kw):return '{"keywords":42,"persons":[],"entities":[],"location":null,"time_expression":null}'
        gate=Gate('query_analysis_json',lambda d:True)
        bridge=self.module._LLMBridge(API(),Audit())
        with active(gate),self.assertRaisesRegex(ValueError,'Invalid query-analysis keywords'):
            self.native.hybrid_retriever.HybridRetriever._analyze_query(SimpleNamespace(llm_client=bridge),'Which day?')
        self.assertEqual(gate.calls,[])

    def test_empty_only_exact_missing_stage_while_other_shapes_fail(self):
        bridge=self.module._LLMBridge(None,Audit())
        def _generate_missing_info_queries(text):return bridge.extract_json(text)
        def _generate_targeted_queries(text):return bridge.extract_json(text)
        gate=Gate('empty_missing_queries',lambda d:d['targeted_queries']==[])
        with active(gate):
            self.assertEqual(_generate_missing_info_queries('{"targeted_queries":[]}'),{'targeted_queries':[]})
            for text in ('{}','{"targeted_queries":null}','{"targeted_queries":""}','{"targeted_queries":[""]}','{"targeted_queries":[1]}'):
                with self.subTest(text=text),self.assertRaises(ValueError):_generate_missing_info_queries(text)
            with self.assertRaises(ValueError):_generate_targeted_queries('{"queries":[]}')
        with self.assertRaises(BaselineFatal):_generate_missing_info_queries('{"targeted_queries":[]}')

    def test_writer_and_nested_retrieval_rejected_before_science(self):
        with self.assertRaises(BaselineFatal):candidate.build({},None,None)
        with active(object()),self.assertRaises(BaselineFatal):candidate.retrieve('question',{},None)


class PrefixAuthorizationTests(unittest.TestCase):
    def fixture(self,kind='native_keyword_syntax_empty'):
        stage='simplemem_query_analysis'
        signature={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':{'contents':[{'parts':[{'text':'fixture'}]}]}}
        key=digest(signature);context={'method':'simplemem','conversation':0,'query_id':'c0_q0','phase':'retrieval'}
        raw={'path':str(candidate.ROOT/'research/fixture/raw.json'),'sha256':'a'*64}
        reg={key:{'signature':signature,'raw':raw,'logical_consumers':[{'stage':stage}]}}
        cert={'kind':kind,'request_hash':key,'stage':stage,'raw':raw,'retry_limit':0,'paid_regeneration_permitted':False,
            'keywords':["John's family"],'exception':{'type':'BaselineFatal','message':"SimpleMem retrieve: ValueError: Syntax Error: John's family"},
            'analysis_query':'Question?','response_text_sha256':'b'*64,
            'json_error':{'message':'Expecting value','line':1,'column':2,'position':1}}
        case={'query_id':'c0_q0','conversation':0,'certificate':cert,
            'prefix':[{'ordinal':0,'context':context,'signature':signature,'request_hash':key,'stage':stage,'raw':raw}]}
        return case,reg,signature
    def complete_keyword(self,p,signature):
        self.assertEqual(p.before(signature,'simplemem_query_analysis')['mode'],'original_prefix')
        p.authorize_native_compatibility('keyword_search',{'keywords':["John's family"],'exception':{'type':'ValueError','message':"Syntax Error: John's family"}})
    def test_prefix_order_and_unqualified_boundary_block_new_requests(self):
        c,r,s=self.fixture();p=prefix.PrefixPolicy(c,r);new=deepcopy(s);new['body']['nonce']='different'
        with self.assertRaises(BaselineFatal):p.before(new,'simplemem_query_analysis')
        p=prefix.PrefixPolicy(c,r);p.before(s,'simplemem_query_analysis')
        with self.assertRaises(BaselineFatal):p.before(new,'simplemem_completeness')
        with self.assertRaises(BaselineFatal):p.authorize_native_compatibility('keyword_search',{'keywords':['altered'],'exception':{'type':'ValueError','message':'Syntax Error: altered'}})
    def test_same_run_registry_reused_but_changed_stage_rejected(self):
        c,r,s=self.fixture();other=deepcopy(s);other['body']['other_query']=True;k=digest(other)
        r[k]={'signature':other,'raw':{'path':'retained-other-raw','sha256':'c'*64},'logical_consumers':[{'stage':'simplemem_completeness'}]}
        p=prefix.PrefixPolicy(c,r);self.complete_keyword(p,s)
        self.assertEqual(p.before(other,'simplemem_completeness')['mode'],'seen_exact_cache')
        p=prefix.PrefixPolicy(c,r);self.complete_keyword(p,s)
        with self.assertRaises(BaselineFatal):p.before(other,'simplemem_missing_queries')
    def test_three_json_attempts_and_repeat_group_keep_identical_raw(self):
        c,r,s=self.fixture('native_analysis_cached_default');p=prefix.PrefixPolicy(c,r)
        details={'query':'Question?','response_text_sha256':'b'*64,'exception':{'type':'JSONDecodeError','message':'Expecting value: line 1 column 2 (char 1)'}}
        for group in range(2):
            for attempt in range(3):
                d=p.before(s,'simplemem_query_analysis')
                p.authorize_native_compatibility('query_analysis_json',{**details,'attempt':attempt})
                self.assertEqual(p.ready,attempt==2)
        self.assertEqual(p.analysis_attempts,6);self.assertEqual(len(p.report()['requests']),6)
    def test_changed_json_error_or_hash_and_unfinished_attempt_reject(self):
        c,r,s=self.fixture('native_analysis_cached_default');p=prefix.PrefixPolicy(c,r);p.before(s,'simplemem_query_analysis')
        details={'query':'Question?','response_text_sha256':'b'*64,'attempt':0,'exception':{'type':'JSONDecodeError','message':'Expecting value: line 1 column 2 (char 1)'}}
        for changed in ({**details,'response_text_sha256':'c'*64},{**details,'exception':{'type':'JSONDecodeError','message':'different'}},{**details,'attempt':2}):
            with self.assertRaises(BaselineFatal):p.authorize_native_compatibility('query_analysis_json',changed)
        p.authorize_native_compatibility('query_analysis_json',details)
        new=deepcopy(s);new['body']['fresh']=True
        with self.assertRaises(BaselineFatal):p.before(new,'simplemem_completeness')
        with self.assertRaises(BaselineFatal):p.report()
    def test_case_registry_tampering_rejected_before_provider_initialization(self):
        c,r,s=self.fixture();directory=candidate.ROOT/'research/nonexistent_prefix_fixture/api_method'
        admission={'run_dir':str(directory),'executing':True,'query_ids':['c0_q0'],'case_sha256':{'c0_q0':digest(c)},'registry_sha256':digest(r)}
        context=c['prefix'][0]['context']
        with patch.object(prefix.BaselineAPI,'__init__',side_effect=AssertionError('must not initialize provider')) as init:
            changed=deepcopy(c);changed['certificate']['keywords']=['changed']
            with self.assertRaises(BaselineFatal):prefix.ContinuationAPI(directory,admission=admission,case=changed,registry=r,context=context)
            changed=deepcopy(r);changed[next(iter(r))]['raw']['sha256']='d'*64
            with self.assertRaises(BaselineFatal):prefix.ContinuationAPI(directory,admission=admission,case=c,registry=changed,context=context)
            with self.assertRaises(BaselineFatal):prefix.ContinuationAPI(directory,admission=admission,case=c,registry=r,context=context,cache_roots=[candidate.ROOT])
            self.assertEqual(init.call_count,0)


if __name__=='__main__':unittest.main()
