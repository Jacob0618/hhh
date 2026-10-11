"""Offline v3: hydrate sealed firstpass jobs for summary bookkeeping only.

Frozen v1/v2 scoring, admission, costs, denominators and recovery predicates
remain unchanged. No seal/prepare/recovery execution or native client occurs.
"""
from pathlib import Path as _Path
from types import FunctionType as _FunctionType, SimpleNamespace as _SimpleNamespace
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from tools import summarize_boundary_variants_full1540_v2 as _v2
from tools import run_boundary_variants_qa_v2 as _qa

for _name,_value in vars(_v2).items():
    if not _name.startswith('__'):globals()[_name]=_value
SCHEMA='boundary-variants-single-arm-full1540-summary-v3'
COMPARISON_SCHEMA='boundary-variants-full1540-comparison-v3'
CODE=tuple(dict.fromkeys((*_v2.CODE,'tools/summarize_boundary_variants_full1540_v3.py')))


def _rebind(namespace,source_namespace):
    for name,value in list(source_namespace.items()):
        if isinstance(value,_FunctionType) and value.__globals__ is source_namespace:
            fn=_FunctionType(value.__code__,namespace,name,value.__defaults__,value.__closure__)
            fn.__kwdefaults__=value.__kwdefaults__
            fn.__module__=__name__;fn.__qualname__=name;fn.__doc__=value.__doc__
            namespace[name]=fn


_rebind(globals(),vars(_v2))
_summarize_v2=summarize


def sealed_firstpass_jobs(firstpass):
    """Read existing seals only; never call execution_manifest/seal_jobs."""
    run=Path(firstpass).resolve()
    require_closed(run)
    m=_qa.manifest(run)  # Exact original policy/code/runtime/cohort/SDK admission.
    if m.get('phase_kind')!='firstpass':
        raise InvalidArtifact('Bookkeeping hydration requires the original firstpass')
    seal=old.plain.verify(run/'firstpass_seal.json')
    source_ref=old.plain.pointer(run/'source_jobs.json')
    if (seal.get('protocol_sha256')!=m['protocol_sha256'] or seal['source_jobs']!=source_ref):
        raise InvalidArtifact('Sealed firstpass source identity differs')
    source=old.plain.verify(full.checked(seal['source_jobs']))
    if (source.get('schema')!=_qa.SCHEMA+'-sources' or source.get('arm')!=m['arm']
        or source.get('protocol_sha256')!=m['protocol_sha256'] or source.get('model_calls')!=0
        or source.get('official_metric_provenance')!=m['official_metric_provenance']
        or source.get('retrieval_outcome')!=old.plain.pointer(run/'retrieval_outcome.json')):
        raise InvalidArtifact('Sealed source_jobs protocol/arm/provenance differs')
    full.validate_jobs({'source_manifest':source_ref},full.canonical(locktool.verify_lock()),m['arm'])
    jobs=source['jobs']
    if len(jobs)!=N or len({j['job_id'] for j in jobs})!=N:
        raise InvalidArtifact('Sealed source_jobs must retain the complete population')
    for job in jobs:_qa.check_job(job,m['arm'])
    hydrated={**m,'jobs':jobs,'source_manifest':source_ref}
    evidence={'kind':'sealed_firstpass_jobs_summary_bookkeeping_only',
        'manifest':old.plain.pointer(run/'run_manifest.json'),
        'firstpass_seal':old.plain.pointer(run/'firstpass_seal.json'),
        'source_jobs':source_ref,'protocol_sha256':m['protocol_sha256'],
        'jobs':len(jobs),'job_list_sha256':_qa.digest(jobs),
        'original_recovery_predicate_unchanged':True,'model_calls':0,
        'source_writes':0,'recovery_executed':False}
    return m,hydrated,old.plain.verify(run/'run_outcome.json'),evidence


def bookkeeping_driver(firstpass):
    """Per-invocation facade: the original QA module is never patched."""
    raw,hydrated,outcome,evidence=sealed_firstpass_jobs(firstpass)
    facade=_SimpleNamespace(**vars(_qa))
    def recovery_jobs(m,out):
        # Preserve the original rejection of a second technical recovery. Its
        # subset jobs are never overwritten with the whole firstpass population.
        if m.get('phase_kind')!='firstpass':return _qa.recovery_jobs(m,out)
        without_jobs={k:v for k,v in m.items() if k not in ('jobs','source_manifest')}
        if (without_jobs!=raw or out!=outcome
            or 'jobs' in m and m['jobs']!=hydrated['jobs']
            or 'source_manifest' in m and m['source_manifest']!=hydrated['source_manifest']):
            raise InvalidArtifact('Summary recovery bookkeeping lineage/jobs differs')
        return _qa.recovery_jobs(hydrated,out)
    facade.recovery_jobs=recovery_jobs
    return facade,evidence


def summarize(firstpass,recovery=None,global_accounting_audit=None):
    facade,evidence=bookkeeping_driver(firstpass)
    namespace=dict(globals());namespace['driver']=facade
    _rebind(namespace,globals())
    result=namespace['_summarize_v2'](firstpass,recovery,global_accounting_audit)
    result['summary_bookkeeping_hydration']=evidence
    return result


if __name__=='__main__':main()
