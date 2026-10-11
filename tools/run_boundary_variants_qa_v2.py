"""Independent C reviewed-source v2; exact v1 full1540/native SDK20/plain.one bodies."""
from pathlib import Path as _Path
from types import FunctionType as _FunctionType
from copy import deepcopy as _deepcopy
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from tools import run_boundary_variants_qa_v1 as _v2_template
from tools import boundary_variants_qa_sources_v2 as _v2_sources
for _name,_value in vars(_v2_template).items():
    if not _name.startswith('__'):globals()[_name]=_value
sources=_v2_sources
SCHEMA='boundary-variants-plain-qa-full1540-v2'
POLICY=_deepcopy(_v2_template.POLICY)
POLICY.update(source_gate='uniform20_original_complete_facts_reviewed_metadata_original_native_unrepaired')
CODE=tuple(dict.fromkeys((*_v2_template.CODE,*_v2_sources.review.DEPENDENCIES,
    'tools/boundary_variants_qa_sources_v2.py','tools/run_boundary_variants_qa_v2.py',
    'tools/mem0_regroup_plain_qa_sources_v2.py')))
for _name,_value in vars(_v2_template).items():
    if isinstance(_value,_FunctionType) and _value.__module__==_v2_template.__name__:
        _fn=_FunctionType(_value.__code__,globals(),_name,_value.__defaults__,_value.__closure__)
        _fn.__kwdefaults__=_value.__kwdefaults__;_fn.__module__=__name__;globals()[_name]=_fn
_prepare_v1_body=prepare
def prepare(run_dir,gate_path):
    gate=sources.verify(gate_path);review=sources.verify(sources.checked_pointer(gate['uniform_twenty_bank_review']))
    with sources.review._locks(review['runs']):return _prepare_v1_body(run_dir,gate_path)
if __name__=='__main__':main()
