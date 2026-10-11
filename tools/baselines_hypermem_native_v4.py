"""Candidate only: six exact historical responses; unchanged private v3 science.

No authority is conferred by this module. A separately admitted API must bind
every compatibility operation to its original raw, full request and local
caller. Unknown responses retain the frozen strict behavior.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import types

V3_SHA256='348c38d970fe45c829373335d1ae5b50c26026b2b806cfd447e3f80fec9237a3'
_path=Path(__file__).with_name('baselines_hypermem_native_v3.py')
_source=_path.read_bytes()
if hashlib.sha256(_source).hexdigest()!=V3_SHA256:raise RuntimeError('Frozen native-v3 adapter changed')
_text=_source.decode('utf-8-sig')
# A private v3 module, including its private upstream package, avoids mutation
# of any frozen module already imported by an independent reviewer.
_text=_text.replace("old.replace('_v2', '_v3')","old.replace('_v2', '_v4')")
_text=_text.replace('_jev_hypermem_v3_private_v2','_jev_hypermem_v4_private_v2')
_v3=types.ModuleType('_jev_hypermem_v4_private_v3');_v3.__file__=str(_path)
exec(compile(_text,str(_path),'exec'),_v3.__dict__)
_runtime=_v3._runtime
HyperMemError=_v3.HyperMemError
BaselineFatal=_runtime.BaselineFatal
represent_episode_ids=_v3.represent_episode_ids
KIND=_v3.KIND
FALSE_CONVERSATIONS=frozenset((0,7))
INSERTIONS={1:((206,'"topic_id": '),(276,'"topic_id": ')),4:((206,'"topic_id": '),),
    5:((7017,'\\'),(7035,'\\'),(7184,'\\'),(7215,'\\')),8:((1322,'\\'),(1332,'\\'))}

def require(ok,message):
    if not ok:raise BaselineFatal(message)

def text_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def strict_json(text):
    def object_pairs(pairs):
        value={}
        for k,v in pairs:
            require(k not in value,'Duplicate JSON member is outside compatibility scope');value[k]=v
        return value
    return json.loads(text,object_pairs_hook=object_pairs,parse_constant=lambda x:(_ for _ in ()).throw(BaselineFatal('Nonfinite JSON constant')))

def derive_exact_text(text,certificate):
    """Pure exact-byte insertion plus inverse. Never searches or guesses edits."""
    ci=certificate['conversation'];edits=INSERTIONS.get(ci)
    require(edits is not None and text_sha(text)==certificate['response_text_sha256'], 'Unknown exact malformed response')
    expected=[{'offset':n,'insert':v} for n,v in edits]
    require(certificate.get('edits')==expected,'Candidate edit certificate differs')
    try:strict_json(text)
    except json.JSONDecodeError:pass
    else:raise BaselineFatal('Exact syntax compatibility requires the original invalid JSON')
    derived=text
    for offset,value in reversed(edits):
        require(0<=offset<len(text),'Insertion offset escapes response')
        require(text[offset]=='"','Insertion must precede the reviewed existing quote')
        derived=derived[:offset]+value+derived[offset:]
    undo=derived;shift=0;positions=[]
    for offset,value in edits:positions.append((offset+shift,value));shift+=len(value)
    for offset,value in reversed(positions):
        require(undo[offset:offset+len(value)]==value,'Insertion inverse differs')
        undo=undo[:offset]+undo[offset+len(value):]
    require(undo.encode('utf-8')==text.encode('utf-8'),'Compatibility is not exactly invertible')
    value=strict_json(derived)
    require(isinstance(value,dict) and _runtime._sha(value)==certificate['derived_parsed_sha256'],'Derived exact value differs')
    return value,{'schema':'hypermem-exact-syntax-representation-v4','conversation':ci,
        'kind':'fixed_topic_id_label_insertion' if ci in (1,4) else 'fixed_summary_quote_escapes',
        'original_text_sha256':text_sha(text),'derived_text_sha256':text_sha(derived),
        'original_text':text,'derived_text':derived,'derived_parsed_sha256':_runtime._sha(value),
        'edits':expected,'inverse_utf8_equal':True,'raw_bytes_mutated':False}

def false_row_contract(value,local):
    """Preserve the payload and the exact pinned match:false consumer branch."""
    mapping=local.get('simple_to_real');rows=value.get('results') if isinstance(value,dict) else None
    require(isinstance(mapping,dict) and mapping and isinstance(rows,list) and len(rows)==len(mapping),'Exact false-row local shape differs')
    require(all(isinstance(row,dict) and type(row.get('match')) is bool for row in rows),'Topic decisions are not explicit booleans')
    positive=[row['topic_id'] for row in rows if row['match']]
    require(len(set(positive))==len(positive) and all(x in mapping for x in positive),'True topic IDs must be original valid unique aliases')
    # No key is added, dropped, renamed or coerced; native code ignores false IDs.
    return {'simple_to_real':dict(mapping),'true_aliases':positive,'payload_sha256':_runtime._sha(value),
        'payload_changed':False,'native_false_rows_ignored':True}

class _Generation(_v3._Generation):
    def _certificate(self,stage,request,path):
        lookup=getattr(self.api,'compatibility_certificate',None)
        return None if lookup is None else lookup(stage=stage,request=request,native_call_path=str(path))

    def _store_compatibility(self,path,record,*,cached=False):
        output=self.work_dir/'compatibility_v4'/Path(path).name
        if output.exists():
            _runtime._verify_seal(output);require(_runtime._read_json(output)==record,'Compatibility journal changed')
        else:
            require(not cached and not _runtime._seal_path(output).exists(),'Missing sealed compatibility journal')
            _runtime._write_json(output,record);_runtime._seal(output)

    def _decode_exact(self,raw,stage,local,path,request,*,cached_native=False):
        cert=self._certificate(stage,request,path)
        if cert is None or cert['conversation'] not in INSERTIONS:
            return raw if isinstance(raw,dict) else json.loads(raw)
        require(type(raw) is str,'Malformed original response must remain original text')
        value,journal=derive_exact_text(raw,cert)
        local_record=self._local_record(local)
        receipt=self.api.authorize_exact_compatibility(cert,request=request,native_call_path=str(path),
            original=raw,derived=value,local=local_record,journal=journal,cached_native=cached_native)
        require(receipt.get('status')=='authorized','No exact syntax compatibility authority')
        self._store_compatibility(path,{'certificate':cert,'native_request':request,'native_call_path':str(path),
            'local':local_record,'derivation':journal,'authorization':receipt},cached=cached_native)
        return value

    @staticmethod
    def _local_record(local):
        return {k:([x.to_dict() for x in v] if isinstance(v,list) else v.to_dict() if hasattr(v,'to_dict') else copy.deepcopy(v))
            for k,v in local.items() if k in ('simple_to_real','topics','topics_batch','topic','episode','new_episode')}

    def _validate_qualified(self,value,stage,local,path,request,*,cached_native=False):
        cert=self._certificate(stage,request,path)
        if cached_native and cert is not None and cert['conversation'] in INSERTIONS:
            retained=_runtime._read_json(path)
            require(retained['request']==request,'Cached exact syntax request changed')
            expected=self._decode_exact(retained['raw_response'],stage,local,path,request,cached_native=True)
            require(value==expected,'Cached exact syntax value changed')
        if cert is not None and cert['conversation'] in FALSE_CONVERSATIONS:
            require(stage=='hypermem_topic_match','False-row exception used outside topic match')
            journal=false_row_contract(value,local);local_record=self._local_record(local)
            receipt=self.api.authorize_exact_compatibility(cert,request=request,native_call_path=str(path),
                original=value,derived=value,local=local_record,journal=journal,cached_native=cached_native)
            require(receipt.get('status')=='authorized','No exact false-row compatibility authority')
            self._store_compatibility(path,{'certificate':cert,'native_request':request,'native_call_path':str(path),
                'local':local_record,'derivation':journal,'authorization':receipt},cached=cached_native)
            return
        return _v3._Generation._validate(value,stage,local)

_derived_source=_v3._derived_source
_decode='value = raw if isinstance(raw, dict) else json.loads(raw)'
require(_derived_source.count(_decode)==1,'Frozen decode boundary changed')
_derived_source=_derived_source.replace(_decode,'value = self._decode_exact(raw, stage, frame.f_locals, path, request)')
_validate='self._validate(value, stage, frame.f_locals)'
require(_derived_source.count(_validate)==2,'Frozen validation boundaries changed')
_derived_source=_derived_source.replace(_validate,"self._validate_qualified(value, stage, frame.f_locals, path, request, cached_native='cached' in locals())")
_namespace=dict(_runtime.__dict__)
exec(compile(_derived_source,'<hypermem-v4-exact-historical-response-boundaries>','exec'),_namespace)
_Generation.generate=_namespace['generate'];_runtime._Generation=_Generation
COMPATIBILITY={**copy.deepcopy(_v3.COMPATIBILITY),'schema':'hypermem-native-v4-technical-compatibility-candidate-v1',
    'six_exact_historical_responses':{'false_row_consumer':[0,7],'fixed_label_insertions':[1,4],'fixed_quote_escapes':[5,8]},
    'future_response_scope_expanded':False,'production_authority':False}
build,retrieve,describe=_runtime.build,_runtime.retrieve,_runtime.describe
UPSTREAM_REVISION,CONFIG,SCHEMA=_runtime.UPSTREAM_REVISION,_runtime.CONFIG,_runtime.SCHEMA

def implementation():
    return {'frozen_v3_sha256':hashlib.sha256(_source).hexdigest(),'compatibility':copy.deepcopy(COMPATIBILITY),
        'derived_generation_sha256':text_sha(_derived_source),'scientific_specification':describe()}
