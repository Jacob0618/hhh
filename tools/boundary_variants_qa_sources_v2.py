"""Explicit C uniform20 reviewed metadata admission; original native bytes retained."""
from __future__ import annotations
import copy
import argparse
from pathlib import Path
from tools import boundary_variants_qa_sources_v1 as legacy
from tools import review_boundary_variants_banks_v2 as review
from tools import mem0_regroup_plain_qa_sources_v2 as common

ROOT=legacy.ROOT
ARMS=legacy.ARMS
GATE_SCHEMA='boundary-variants-full1540-qa-gate-v2'
CHECKS=('twenty_terminal_banks_reviewed','twenty_complete_fact_banks',
        'source_dates_and_generation_chain_verified','current_sdk20_index_dependencies_reviewed',
        'native_index_limitations_disclosed','content_quality_findings_disclosed')
CONFIG,COHORT=legacy.CONFIG,legacy.COHORT
pointer,checked_pointer=legacy.pointer,legacy.checked_pointer
verify,digest,sha=legacy.verify,legacy.digest,legacy.sha
copy_bank,inventory,protocol_config,cohort=legacy.copy_bank,legacy.inventory,legacy.protocol_config,legacy.cohort
# Compatibility name for the existing independent C QA lock wrapper.
audit=legacy.audit

def population(review_path):
    value=review.check_review(review_path);review._population(value['banks'])
    rows={(r['conversation'],r['arm']):r for r in value['banks']}
    for r in rows.values():
        state={'strict_complete':'complete','reviewed_export':'failed'}.get(r.get('source_kind'))
        if (state is None or r.get('original_write_status')!=state or r.get('groups')!=r.get('expected_groups')
                or r.get('groups',0)<=0 or r.get('facts',0)<20
                or not r.get('native_index_limitations') or (state=='failed' and not r.get('proof'))):
            raise ValueError('Partial/unknown/unproved failure cannot enter reviewed C source')
    return value,rows

def export_population(export_path,review_ref):
    outcome=verify(export_path)
    if (outcome.get('schema')!=review.EXPORT_SCHEMA or outcome.get('status')!='reviewed_metadata_exports_complete'
            or outcome.get('source_review')!=review_ref or outcome.get('source_mutations')!=0
            or outcome.get('native_bytes_copied')!=0 or outcome.get('model_calls')!=0):
        raise ValueError('Uniform20 metadata export outcome required')
    review._population(outcome['exports'])
    return {(r['conversation'],r['arm']):r for r in outcome['exports']}

def assemble_rows(review_path,source_arm,export_manifest):
    if source_arm not in review.ARMS:raise ValueError('Current C source arm required')
    _,pop=population(review_path);ref=pointer(review_path)
    exports=export_population(export_manifest,ref);rows=[]
    for ci in range(10):
        r=pop[ci,source_arm];e=exports[ci,source_arm];loaded=review.check_export_bank(e['reviewed_export'])
        if loaded['uniform_row']!=r or loaded['review']['bank']!=e['bank']:
            raise ValueError('Metadata export row differs from uniform20 review')
        rows.append({'conversation':ci,'arm':legacy.audit.ARM_METHODS[source_arm],'source_arm':source_arm,
            'source_kind':r['source_kind'],'original_write_status':r['original_write_status'],
            'reviewed_export_status':'complete','review_row_sha256':digest(r),'bank':e['bank'],
            'reviewed_export':e['reviewed_export'],'audits':copy.deepcopy(r['audits']),
            'native_index_limitations':copy.deepcopy(r['native_index_limitations']),
            'write_status':r['write_status'],'source_manifest':r['source_manifest'],
            'group_inputs':r['group_inputs'],'native_inventory':r['native_inventory'],'proof':r['proof']})
    return rows

def admit_gate(path):
    gate=verify(path);source_arm=gate.get('source_arm')
    if (gate.get('schema')!=GATE_SCHEMA or gate.get('status')!='passed'
            or legacy.audit.ARM_METHODS.get(source_arm)!=gate.get('arm')
            or any(gate.get('checks',{}).get(k) is not True for k in CHECKS)
            or not gate.get('limitations') or not gate.get('content_quality_reviews')):
        raise ValueError('Explicit v2 uniform20 structural/source-quality gate required')
    for ref in gate['content_quality_reviews']:checked_pointer(ref)
    review_path=checked_pointer(gate['uniform_twenty_bank_review'])
    value=verify(review_path)
    with review._locks(value['runs']):
        _,pop=population(review_path)
        export_path=checked_pointer(gate['export_manifest'])
        expected=assemble_rows(review_path,source_arm,export_path)
        if gate.get('banks')!=expected:raise ValueError('Gate rows differ from full20 reviewed metadata export')
        writer_manifest=checked_pointer(gate['source_manifest']);origin=writer_manifest.parent
        parent=legacy.writer.manifest(origin)
        if (value['runs'][source_arm]!=str(origin.resolve()) or parent['arms']!=[source_arm]
                or parent['boundary_variant']['arm']!=source_arm):
            raise ValueError('Original Writer manifest/arm lineage differs')
        admitted={}
        for row in expected:
            ci=row['conversation'];r=pop[ci,source_arm]
            if row['source_manifest']!=gate['source_manifest']:raise ValueError('Mixed original writer manifests')
            loaded=review.check_export_bank(row['reviewed_export']);bank=loaded['bank'];native=loaded['native']
            status=legacy.old.read(checked_pointer(r['write_status']))
            if (status.get('status')!=row['original_write_status']
                    or status.get('protocol_sha256')!=parent['protocol_sha256']
                    or (status.get('conversation'),status.get('arm'))!=(ci,source_arm)):
                raise ValueError('Original failed/successful Writer state changed')
            common._common_chain(r,bank,status,native,parent)
            provenance={**copy.deepcopy(row),'original_review':copy.deepcopy(r),
                'uniform_twenty_bank_review':gate['uniform_twenty_bank_review'],'export_manifest':gate['export_manifest']}
            admitted[ci,gate['arm']]={'review':provenance,'bank':bank,'native':native}
        resource=parent['boundary_variant']['resources']
        legacy.old.validate_resources(resource['snapshot'],resource['files_sha256'])
        return gate,admitted,resource

def write_gate(review_path,export_manifest,source_arm,quality_reviews,output):
    """Explicit offline gate publication, only after independently completed whole20 review/export."""
    if not quality_reviews:raise ValueError('Separately bound semantic findings required')
    for ref in quality_reviews:checked_pointer(ref)
    declared=verify(review_path)
    with review._locks(declared['runs']):
        value,_=population(review_path)
        rows=assemble_rows(review_path,source_arm,export_manifest)
        gate={'schema':GATE_SCHEMA,'status':'passed','arm':legacy.audit.ARM_METHODS[source_arm],
            'source_arm':source_arm,'source_manifest':pointer(Path(value['runs'][source_arm])/'run_manifest.json'),
            'uniform_twenty_bank_review':pointer(review_path),'export_manifest':pointer(export_manifest),
            'checks':{k:True for k in CHECKS},'content_quality_reviews':quality_reviews,'banks':rows,
            'limitations':value['limitations']+['Original strict failure remains failed; reviewed metadata is a separate policy.'],
            'model_calls':0,'source_mutations':0}
        legacy.old.save_new(output,gate);admit_gate(output);return pointer(output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('gate','check'))
    p.add_argument('--review',type=Path);p.add_argument('--export-manifest',type=Path)
    p.add_argument('--source-arm',choices=review.ARMS);p.add_argument('--quality-review',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='check':print({'valid':bool(admit_gate(a.output)),'model_calls':0})
    else:print(write_gate(a.review,a.export_manifest,a.source_arm,[pointer(x) for x in a.quality_review],a.output))

if __name__=='__main__':main()
