"""Offline manifest-only repair; all frozen QA runtime functions remain original."""
from __future__ import annotations
import copy
from pathlib import Path
from tools import run_r03_r05_refinement_qa_v1 as frozen


def prepare(new_run, old_run, *, technical_bindings):
    new_run,old_run=Path(new_run).resolve(),Path(old_run).resolve()
    if new_run.parent != (frozen.ROOT/'runs').resolve() or new_run.exists():
        raise FileExistsError('Fresh direct runs namespace required')
    if {p.relative_to(old_run).as_posix() for p in old_run.rglob('*') if p.is_file()} != {'run_manifest.json','prior_qa_reuse_proof.json'}:
        raise ValueError('Old failed run must have only two preparation files')
    old=frozen.read(old_run/'run_manifest.json');proof_ref=frozen.pointer(old_run/'prior_qa_reuse_proof.json')
    if (old.get('prior_qa_reuse_proof') is not None or old.get('phase_kind')!='firstpass'
        or old.get('protocol_sha256')!=frozen.digest({k:v for k,v in old.items() if k!='protocol_sha256'})):
        raise ValueError('Exact preparation-only null-pointer defect required')
    source=frozen.bound(old['source_manifest'])
    if source.get('schema')!=frozen.SOURCE_SCHEMA:
        raise ValueError('Sealed refinement source schema required')
    _,metric=frozen.verified_official_metric()
    if (old.get('schema')!=frozen.SCHEMA or old.get('policy')!=frozen.POLICY
        or old.get('cache_roots')!=[] or old.get('parent') is not None
        or old.get('jobs')!=source['jobs'] or old['source_manifest']!=frozen.pointer(old['source_manifest']['path'])
        or old.get('official_metric_provenance')!=metric or metric!=source['official_metric_provenance']
        or old.get('code_sha256')!={p:frozen.sha(frozen.ROOT/p) for p in frozen.CODE}):
        raise ValueError('Original frozen manifest/source admission mismatch')
    frozen.check_full_cohort(old['jobs'])
    proof=frozen.bound(proof_ref);frozen.verify(proof_ref['path'])
    if old.get('reuse_plans') != frozen._plans(old['jobs'],proof,'firstpass'):
        raise ValueError('Exact inherited plans differ')
    value=copy.deepcopy(old)
    value.pop('protocol_sha256')
    value.update(prior_qa_reuse_proof=proof_ref,created_utc=frozen.now())
    # Lightweight original structural admission follows publication. Full source
    # and historical raw proof admission remains pending at original qa.run.
    new_run.mkdir(parents=True,exist_ok=False)
    frozen.publish_json(new_run/'run_manifest.json',{**value,'protocol_sha256':frozen.digest(value)})
    frozen.save_new(new_run/'prepare_fix_lineage_v2.json',{
        'schema':'r03-r05-qa-manifest-pointer-fix-lineage-v2','created_utc':frozen.now(),
        'old_manifest':frozen.pointer(old_run/'run_manifest.json'),'preserved_proof':proof_ref,
        'new_manifest':frozen.pointer(new_run/'run_manifest.json'),'technical_bindings':technical_bindings,
        'changed_manifest_fields':['prior_qa_reuse_proof','created_utc','protocol_sha256'],
        'full_raw_admission':'pending_original_qa_run_manifest','model_calls':0})
    admitted=frozen._ordinary_manifest(new_run,_source=source)
    return {'run_dir':str(new_run),'jobs':len(admitted['jobs']),'proof':proof_ref,
            'protocol_sha256':admitted['protocol_sha256'],'model_calls':0,
            'full_raw_admission':'pending_original_qa_run_manifest'}
