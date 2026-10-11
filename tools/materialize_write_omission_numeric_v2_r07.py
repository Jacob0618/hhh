"""R07 prelaunch JSONL binding erratum: format-neutral input SHA checks only.

All dedicated document, raw, producer, ledger, exact native and QA checks delegate
to the frozen numeric v1 adapter in isolated function globals. No old edits.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import types
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import materialize_write_omission_numeric_r07 as v1

ERRATUM = ROOT/'research/jev_autoresearch_2026-10-04/R07_numeric_io_erratum_protocol_v2.json'
ERRATUM_SHA = '176aec771add5d2506afdec41f5edc78b3ecbfbcef815821d0978780fac7e34d'
SOURCE_SCHEMA = v1.SOURCE_SCHEMA
CODE = tuple(dict.fromkeys((*v1.CODE,'tools/materialize_write_omission_numeric_v2_r07.py',
    'tools/run_write_omission_numeric_v2_qa_r07.py','tools/summarize_write_omission_numeric_v2_r07.py')))


def erratum():
    v1.require(v1.old.sha(ERRATUM)==ERRATUM_SHA,'Three-voted IO erratum changed')
    e = v1.sealed_read(ERRATUM)
    for field in ('scientific_protocol','numeric_compatibility_protocol','prepared_v1_selector_manifest',
                  'prepared_v1_native_manifest','prelaunch_failure'):
        v1.bound(e[field])
    v1.require(e['numeric_compatibility_protocol']==v1.pointer(v1.TECH_SPEC), 'Numeric rules changed')
    return e


def spec():
    return {**v1.spec(),'namespace':erratum()['namespace']}


def validate_input_bytes(refs):
    """No assumptions about serialization; semantic loaders are stage-specific."""
    for ref in refs:
        v1.require(isinstance(ref,dict) and set(ref)=={'path','sha256'}
            and v1.pointer(ref['path'])==ref,'Bound input bytes differ')


def _publish(path,value):
    if Path(path).name=='run_manifest.json':
        value={**value,'numeric_io_erratum':v1.pointer(ERRATUM)}
    return v1.save_new(path,value)


def _context():
    space=dict(vars(v1))
    for name,value in list(space.items()):
        if isinstance(value,types.FunctionType) and value.__globals__ is vars(v1):
            f=types.FunctionType(value.__code__,space,name,value.__defaults__,value.__closure__)
            f.__kwdefaults__=value.__kwdefaults__;space[name]=f
    space['_unversioned_seal_value']=space['_seal_value']
    space.update(spec=spec,manifest=manifest,CODE=CODE,save_new=_publish,
        materialize_first_attempt=materialize_first_attempt,retrieve_conversation=retrieve_conversation,
        _seal_value=_seal_value)
    return types.SimpleNamespace(**space)


def manifest(run_dir,validate_admission=True):
    """Frozen v1 manifest invariants; only input_refs use a byte-level loader."""
    run=Path(run_dir).resolve();s=spec();m=v1.sealed_read(run/'materialization/run_manifest.json')
    v1.require(str(run)==s['namespace'] and m['schema']==v1.SCHEMA
        and m['numeric_compatibility_protocol']==v1.pointer(v1.TECH_SPEC)
        and m['admission']==s['original_admission'],'Numeric manifest identity differs')
    v1.require(m.get('numeric_io_erratum')==v1.pointer(ERRATUM),'IO erratum manifest binding differs')
    original=v1.bound(s['original_native_manifest'],sealed=True)
    for field in ('protocol','source_banks','resources','embedding_cache_pack','retrieval_manifest','native_sdk_sha256'):
        v1.require(m[field]==original[field],'Inherited original native/source/resource field changed')
    for path,value in m['code_sha256'].items():
        v1.require(v1.old.sha(ROOT/path)==value,'Numeric implementation changed')
    for path,value in m['numeric_runtime_code_sha256'].items():
        v1.require(v1.old.sha(path)==value,'Pinned native numeric runtime changed')
    v1.require(m['native_sdk_sha256']==v1.old.native_sdk_identity(),'Native SDK changed')
    validate_input_bytes(m['input_refs'])
    for info in [*m['source_banks'].values(),*m['original_native_inventory'].values()]:
        v1.require(v1.old.inventory(info['native_dir'])==info['inventory'],'Immutable original native bytes changed')
    for info in m['source_banks'].values():
        v1.bound(info['bank'],sealed=True);v1.bound(info['snapshot'],sealed=True)
    if validate_admission:
        v1.require(v1.old.admission(s['original_run'])==v1.bound(m['admission'],sealed=True),'Scientific admission changed')
    selector=v1.sealed_read(run/'run_manifest.json')
    v1.require(selector['original_selector_manifest']==s['original_selector_manifest']
        and selector['numeric_compatibility_protocol']==m['numeric_compatibility_protocol']
        and selector['code_sha256']==m['code_sha256']
        and selector.get('numeric_io_erratum')==m['numeric_io_erratum'],'Numeric selector/code lineage differs')
    return m


def prepare(run_dir):return _context().prepare(run_dir)
def materialize(run_dir):return _context().materialize(run_dir)
def effective_audit(run_dir,publish=False):return _context().effective_audit(run_dir,publish=publish)
def retrieve(run_dir):return _context().retrieve(run_dir)
def seal(run_dir):return _context().seal(run_dir)
def validate_source(path,fresh=False):return _context().validate_source(path,fresh=fresh)


def materialize_first_attempt(run,ci,m):
    ctx=_context()
    v1.require(ci in (4,5,6,7),'Only never-started four conversations may dispatch')
    return ctx._native_context(run,ci,m).materialize_bank(run,ci,'jev',
        m,v1.old.admission(spec()['original_run']))


def retrieve_conversation(run,ci,m,gate,jobs,entries):
    return _context()._retrieval_context(run,m).retrieve_conversation(run,ci,m,gate,jobs,entries)


def _seal_value(run):
    return {**_context()._unversioned_seal_value(run),'numeric_io_erratum':v1.pointer(ERRATUM)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','materialize','audit','retrieve','seal','check'))
    parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    funcs={'prepare':prepare,'materialize':materialize,'audit':lambda r:effective_audit(r,publish=True),
        'retrieve':retrieve,'seal':seal,'check':lambda r:{'valid_jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),
                                                     'model_calls':0}}
    print(json.dumps(funcs[args.command](args.run_dir),ensure_ascii=False))


if __name__=='__main__':main()
