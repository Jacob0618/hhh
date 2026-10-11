"""Explicit reviewed-source v2 admission, unchanged offline C summary bodies.

Original failed Writer state and reviewed metadata export are separate evidence.
No model/Reader/Judge, source repair, export, or old-summary mutation occurs here.
"""
from pathlib import Path as _Path
import sys as _sys
from types import FunctionType as _FunctionType
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from tools import summarize_boundary_variants_full1540_v1 as _v1
from tools import run_boundary_variants_qa_v2 as _driver

for _name,_value in vars(_v1).items():
    if not _name.startswith('__'):globals()[_name]=_value

driver=_driver
SCHEMA='boundary-variants-single-arm-full1540-summary-v2'
COMPARISON_SCHEMA='boundary-variants-full1540-comparison-v2'
CODE=tuple(dict.fromkeys((*_v1.CODE,*_driver.CODE,'tools/summarize_boundary_variants_full1540_v2.py',
    'tools/run_boundary_variants_qa_v2.py','tools/boundary_variants_qa_sources_v2.py',
    'tools/review_boundary_variants_banks_v2.py')))
REBOUND_FUNCTIONS=tuple(name for name,value in vars(_v1).items()
    if isinstance(value,_FunctionType) and value.__module__==_v1.__name__)
for _name in REBOUND_FUNCTIONS:
    _original=getattr(_v1,_name)
    _fn=_FunctionType(_original.__code__,globals(),_name,
        _original.__defaults__,_original.__closure__)
    _fn.__kwdefaults__=_original.__kwdefaults__
    _fn.__module__=__name__;_fn.__qualname__=_name;_fn.__doc__=_original.__doc__
    globals()[_name]=_fn

_summarize_core=summarize
_compare_core=compare


def source_admission_evidence(firstpass):
    """Re-admit exactly the v2 gate; never silently accept it as legacy strict."""
    m=old.plain.read(Path(firstpass)/'run_manifest.json')
    if m.get('schema')!=driver.SCHEMA:
        raise InvalidArtifact('Explicit reviewed-source v2 QA schema required')
    gate_path=full.checked(m['gate'])
    gate,admitted,_=driver.sources.admit_gate(gate_path)
    if (gate.get('schema')!='boundary-variants-full1540-qa-gate-v2'
        or gate.get('arm')!=m['arm'] or len(admitted)!=10):
        raise InvalidArtifact('Explicit full10 v2 source gate required')
    rows=copy.deepcopy(gate['banks'])
    for row in rows:
        expected={'strict_complete':'complete','reviewed_export':'failed'}.get(row.get('source_kind'))
        if expected is None or row.get('original_write_status')!=expected:
            raise InvalidArtifact('Original Writer status cannot be erased by reviewed metadata export')
    return {'gate':old.plain.pointer(gate_path),'gate_schema':gate['schema'],
        'original_writer_manifest':gate['source_manifest'],
        'source_rows':rows,'original_failed_sources':[r for r in rows if r['original_write_status']=='failed'],
        'reviewed_export_sources':[r for r in rows if r['source_kind']=='reviewed_export'],
        'original_strict_completion_not_implied':True,
        'export_is_metadata_admission_not_native_repair':True,
        'bound_review_and_export_fields':{k:copy.deepcopy(v) for k,v in gate.items()
            if k in ('uniform_twenty_bank_review','export_manifest','limitations','checks','content_quality_reviews')},
        'note':'Original failed status remains failed. Reviewed metadata admits complete original facts with disclosed native defects; no facts/index/Writer output regenerated.'}


def summarize(firstpass,recovery=None,global_accounting_audit=None):
    evidence=source_admission_evidence(firstpass)
    result=_summarize_core(firstpass,recovery,global_accounting_audit)
    result['source_admission_evidence']=evidence
    result['limitations'] += [
        'Explicit v2 reviewed-source admission includes any retained original Writer failures; it does not turn them into original strict-complete banks.',
        'Metadata export does not repair original facts or the original native index. Source defects and v2 packaging are additional interpretation factors.',
        'Unrepaired entity-edge omissions may affect entity boost, ranking and evidence available to Reader; metadata admission does not establish no performance effect.']
    return result


def compare(c1_path,c2_path,*,bootstrap_samples=2000):
    result=_compare_core(c1_path,c2_path,bootstrap_samples=bootstrap_samples)
    # The unchanged comparison core already replayed both reports. Re-read only
    # its exact sealed summary pointers, without doing a second expensive replay.
    reports=[old.plain.verify(full.checked(result['sources'][key])) for key in ('c1','c2')]
    result['source_admission_evidence']={r['arm']:r['source_admission_evidence'] for r in reports}
    result['limitations'] += ['Both C inputs explicitly use reviewed-source v2 admission; preserved Writer failures are disclosed separately from QA validity.']
    return result


if __name__=='__main__':main()
