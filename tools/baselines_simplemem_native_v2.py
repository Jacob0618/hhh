"""Private SimpleMem retrieval continuation over an exact copied terminal bank.

No writer, new embedding of the corpus, prompt rewrite, keyword escaping or
general exception fallback. Only case-certified original native behavior is
restored; all other handlers retain the frozen v1 hard-stop policy.
"""
from __future__ import annotations
import ast
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import textwrap
import threading

from tools.baselines_common_v1 import BaselineFatal,read,sha,save_new

ROOT=Path(__file__).resolve().parents[1]
FROZEN_PATH=ROOT/'tools/baselines_simplemem_v1.py'
FROZEN_SHA='616bf5c1feea32193fbde29e3ca665c3ea8b89d9e5a71d18e58a78fe95696b4b'
SCHEMA='simplemem-case-certified-native-continuation-v2'
_ACTIVE=ContextVar('simplemem_native_v2_active_query',default=None)
_LOCK=threading.RLock()
_PRIVATE=None


def _text_sha(value):return hashlib.sha256(value.encode('utf8')).hexdigest()


def _gate(kind,details):
    api=_ACTIVE.get()
    if api is None or not hasattr(api,'authorize_native_compatibility'):
        raise BaselineFatal('Native exception policy requires an admitted exact query certificate')
    if api.authorize_native_compatibility(kind,details) is not True:
        raise BaselineFatal('Unqualified SimpleMem native exception/empty-list compatibility')
    return True


def _keyword_exception(error,keywords):
    return _gate('keyword_search',{'keywords':list(keywords),
        'exception':{'type':type(error).__name__,'message':str(error)}})


def _analysis_exception(error,query,response,attempt):
    if not isinstance(error,json.JSONDecodeError):return False
    return _gate('query_analysis_json',{'query':query,'response_text_sha256':_text_sha(response),
        'attempt':attempt,'exception':{'type':type(error).__name__,'message':str(error)}})


def _empty_targeted(caller,response,value):
    if caller!='_generate_missing_info_queries' or value.get('targeted_queries')!=[]:return False
    return _gate('empty_missing_queries',{'stage':'simplemem_missing_queries',
        'response_text_sha256':_text_sha(response),'targeted_queries':[]})


def _load_private():
    global _PRIVATE
    with _LOCK:
        if sha(FROZEN_PATH)!=FROZEN_SHA:raise BaselineFatal('Frozen SimpleMem adapter changed')
        if _PRIVATE is not None:return _PRIVATE
        spec=importlib.util.spec_from_file_location('_jev_simplemem_native_v2_adapter',FROZEN_PATH)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module._PREFIX='_jev_simplemem_native_v2_upstream'
        module._NATIVE=None
        original_bridge=module._SourceBridge
        class NarrowSourceBridge(original_bridge):
            def __init__(self):self.functions=[]
            def visit_FunctionDef(self,node):
                self.functions.append(node.name)
                try:return self.generic_visit(node)
                finally:self.functions.pop()
            def visit_ExceptHandler(self,node):
                function=self.functions[-1] if self.functions else None
                if function=='keyword_search' and node.name=='error':
                    gate=ast.Call(func=ast.Name(id='_native_v2_keyword_exception',ctx=ast.Load()),
                        args=[ast.Name(id='error',ctx=ast.Load()),ast.Name(id='keywords',ctx=ast.Load())],keywords=[])
                elif function=='_analyze_query' and node.name=='e':
                    gate=ast.Call(func=ast.Name(id='_native_v2_analysis_exception',ctx=ast.Load()),
                        args=[ast.Name(id=x,ctx=ast.Load()) for x in ('e','query','response','attempt')],keywords=[])
                else:return super().visit_ExceptHandler(node)
                node.body=[ast.If(test=ast.UnaryOp(op=ast.Not(),operand=gate),
                    body=[ast.Raise(exc=None,cause=None)],orelse=[]),*node.body]
                return node
        module._SourceBridge=NarrowSourceBridge
        original_load=module._load_native
        def load_native():
            native=original_load()
            for value in vars(native).values():
                value.__dict__.update(_native_v2_keyword_exception=_keyword_exception,
                    _native_v2_analysis_exception=_analysis_exception)
            return native
        module._load_native=load_native
        # Change one extra v1 schema condition. The upstream caller frame and
        # every other parse/schema check remain the original function body.
        tree=ast.parse(textwrap.dedent(inspect.getsource(module._LLMBridge.extract_json)))
        class EmptyOnly(ast.NodeTransformer):
            count=0
            def visit_UnaryOp(self,node):
                if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Name) and node.operand.id=='queries':
                    self.count+=1
                    return ast.BoolOp(op=ast.And(),values=[node,ast.UnaryOp(op=ast.Not(),operand=ast.Call(
                        func=ast.Name(id='_native_v2_empty_targeted',ctx=ast.Load()),
                        args=[ast.Name(id=x,ctx=ast.Load()) for x in ('caller','response','value')],keywords=[]))])
                return self.generic_visit(node)
        change=EmptyOnly();tree=ast.fix_missing_locations(change.visit(tree))
        if change.count!=1:raise BaselineFatal('Frozen empty-query schema patch no longer has one exact site')
        scope={**module.__dict__,'_native_v2_empty_targeted':_empty_targeted}
        exec(compile(tree,str(FROZEN_PATH)+'::private-empty-targeted-v2','exec'),scope)
        module._LLMBridge.extract_json=scope['extract_json']
        _PRIVATE=module
        return module


def describe():
    frozen=_load_private()
    return {'schema':SCHEMA,'frozen_adapter_sha256':FROZEN_SHA,'scientific_specification':frozen.describe(),
        'writer_allowed':False,'corpus_reembedding_allowed':False,'keyword_rewriting_allowed':False,
        'state_policy':'Exact terminal database copy, original exports and rebased state in new namespace; original bank never opened natively',
        'compatibility':'Only admitted per-query raw/signature/stage certificates: original keyword empty branch, three cached bad-analysis parses/default, explicit empty targeted_queries',
        'other_exceptions':'Frozen v1 hard-stop policy; no unqualified fallback or fresh redraw of any original request'}


def _safe_tree(path):
    p=Path(path).absolute()
    if '..' in p.parts or not p.is_relative_to(ROOT):raise BaselineFatal('Foreign bank path')
    for part in (p,*p.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(),'st_file_attributes',0)&0x400):
            raise BaselineFatal('Reparse bank path forbidden')
        if part==ROOT:break
    resolved=p.resolve()
    if resolved!=p or not resolved.is_relative_to(ROOT):raise BaselineFatal('Noncanonical bank path')
    return resolved


def _inventory(directory):
    root=_safe_tree(directory)
    result={}
    for p in sorted(root.rglob('*')):
        p=_safe_tree(p)
        if p.is_file():result[str(p)]=sha(p)
    return result


def copy_terminal_bank(state_path,destination,*,state_sha256,database_inventory):
    """Only copies existing bytes, without invoking any native database opener."""
    frozen=_load_private();state_path=_safe_tree(state_path);destination=_safe_tree(destination)
    if sha(state_path)!=state_sha256:raise BaselineFatal('Terminal state changed')
    state=read(state_path);source=_safe_tree(state['work_dir']);database=source/'lancedb'
    if state_path!=source/'state.json' or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise BaselineFatal('Terminal source and new bank copy must be distinct')
    frozen._verify_state(state)
    expected={str(_safe_tree(p)):h for p,h in database_inventory.items()}
    if not expected or any(not Path(p).is_relative_to(database) for p in expected):
        raise BaselineFatal('Complete exact terminal database inventory required')
    if _inventory(database)!=expected:raise BaselineFatal('Terminal LanceDB inventory differs')
    if destination.exists():raise FileExistsError('Continuation bank is never implicitly reused')
    destination.mkdir(parents=True,exist_ok=False)
    for name,h in state['export_sha256'].items():
        target=destination/name;shutil.copyfile(source/name,target)
        if sha(target)!=h:raise BaselineFatal('Export copy changed bytes')
    copied={}
    for path,h in expected.items():
        target=destination/'lancedb'/Path(path).relative_to(database)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if sha(target)!=h:raise BaselineFatal('Terminal database copy changed bytes')
        copied[str(target)]=h
    if _inventory(database)!=expected or sha(state_path)!=state_sha256:
        raise BaselineFatal('Original bank changed during exact copy')
    result=deepcopy(state);result.update(work_dir=str(destination),memory_path=str(destination/'memory.json'),
        request_log=str(destination/'requests.jsonl'),export_paths=[str(destination/Path(p).name) for p in state['export_paths']])
    frozen._verify_state(result)
    save_new(destination/'state.json',result)
    save_new(destination/'terminal_bank_lineage.json',{'schema':SCHEMA,'original_state':{'path':str(state_path),'sha256':state_sha256},
        'original_database_inventory':expected,'copied_database_inventory':copied,
        'rebased_state':{'path':str(destination/'state.json'),'sha256':sha(destination/'state.json')},
        'database_integrity':state['database_integrity'],'new_writer_calls':0,'new_corpus_embedding_calls':0})
    return result


def retrieve(question,state,api):
    frozen=_load_private()
    if _ACTIVE.get() is not None:raise BaselineFatal('Nested private retrieval context forbidden')
    token=_ACTIVE.set(api)
    try:return frozen.retrieve(question,state,api)
    finally:_ACTIVE.reset(token)


def build(*args,**kwargs):raise BaselineFatal('This technical continuation cannot rebuild memory')
