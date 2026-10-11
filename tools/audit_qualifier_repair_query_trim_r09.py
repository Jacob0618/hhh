"""R09 derived offline audit: frozen SDK QUERY preprocessing, unchanged retrieval."""
from __future__ import annotations
import argparse,copy,inspect,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import retrieve_qualifier_repair_compatible_r09 as parent
from mem0.memory.main import Memory,_validate_and_trim_search_query
from jev_mem.local_bm25_context_v2 import forbid_network
require,pointer,bound,sealed_read,save_new,sha,digest,body=parent.require,parent.pointer,parent.bound,parent.sealed_read,parent.save_new,parent.sha,parent.digest,parent.body
SOURCE_SCHEMA=parent.SOURCE_SCHEMA
ERRATUM=ROOT/'research/jev_autoresearch_2026-10-04/R09_query_preprocessing_audit_erratum_v2.json'
ERRATUM_SHA='276d4eb25190c8770f39e06505c86d66dcad0e20b969c200a3e0c2dd43735ea7'
CODE=tuple(dict.fromkeys((*parent.CODE,'tools/audit_qualifier_repair_query_trim_r09.py',
    'tools/run_qualifier_repair_query_trim_qa_r09.py','tools/summarize_qualifier_repair_query_trim_r09.py')))
MANIFEST_SCHEMA='qualifier-repair-r09-query-event-audit-manifest-v1'

def erratum():
    require(sha(ERRATUM)==ERRATUM_SHA,'QUERY preprocessing erratum changed')
    spec=sealed_read(ERRATUM)
    for ref in spec['refs'].values():require(pointer(ref['path'])==ref,'QUERY erratum input bytes changed')
    require(pointer(inspect.getfile(Memory))['sha256']==spec['refs']['frozen_sdk']['sha256']
        and inspect.getfile(_validate_and_trim_search_query)==inspect.getfile(Memory),
        'Imported frozen SDK query helper differs')
    return spec

def completed_run(spec):return Path(spec['refs']['retrieval_manifest']['path']).resolve().parent
def audit_dir(spec):return Path(spec['outputs']['directory']).resolve()

def validate_cache_events(selection,job,entries,arm):
    # The original Reader/source question remains byte-for-byte unchanged.
    derived=copy.deepcopy(job)
    derived['question']=_validate_and_trim_search_query(job['question'])
    return parent.validate_cache_events(selection,derived,entries,arm)

def completed_input_refs(run):
    folder=Path(run)/'retrieval'
    return [pointer(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and 'native' not in p.relative_to(folder).parts]

def _scope(run,derived_gate=None):
    run=Path(run).resolve();alias=run/'retrieval_gate.json'
    def audit_pointer(path):
        if derived_gate is not None and Path(path).resolve()==alias:
            require(pointer(derived_gate['path'])==derived_gate,'Derived QUERY gate changed')
            return derived_gate
        return pointer(path)
    return parent.parent.support.isolated(parent,{'validate_cache_events':validate_cache_events,'pointer':audit_pointer})

def prepare(run):
    spec=erratum();run=Path(run).resolve();out=audit_dir(spec)
    require(run==completed_run(spec) and not out.exists(),'Fresh derived audit directory required')
    m=parent.manifest(run)
    require(pointer(run/'run_manifest.json')==spec['refs']['retrieval_manifest']
        and pointer(run/'retrieval_outcome.json')==spec['refs']['complete_retrieval_outcome'],
        'Completed retrieval identity changed')
    outcome=sealed_read(run/'retrieval_outcome.json')
    require(outcome['status']=='complete' and len(outcome['banks'])==20 and outcome['model_calls']==0,
        'Completed retrieval stage required')
    value={'schema':MANIFEST_SCHEMA,'query_event_audit_erratum':pointer(ERRATUM),
        'completed_retrieval_run':str(run),'retrieval_manifest':pointer(run/'run_manifest.json'),
        'completed_retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'failed_offline_audit':spec['refs']['failed_offline_audit_stderr'],
        'completed_input_refs':completed_input_refs(run),'code_sha256':{p:sha(ROOT/p) for p in CODE},
        'parent_code_sha256':m['code_sha256'],'frozen_query_helper':pointer(inspect.getfile(Memory)),
        'new_model_calls':0,'retrieval_rerun':False,'scientific_changes':False}
    return save_new(out/'manifest.json',value)

def manifest(path=None):
    spec=erratum();expected=audit_dir(spec)/'manifest.json'
    require(path is None or Path(path).resolve()==expected,'Foreign QUERY audit manifest')
    m=sealed_read(expected);run=completed_run(spec)
    require(m['schema']==MANIFEST_SCHEMA and m['query_event_audit_erratum']==pointer(ERRATUM)
        and m['completed_retrieval_run']==str(run)
        and m['retrieval_manifest']==spec['refs']['retrieval_manifest']
        and m['completed_retrieval_outcome']==spec['refs']['complete_retrieval_outcome']
        and m['failed_offline_audit']==spec['refs']['failed_offline_audit_stderr']
        and m['frozen_query_helper']==pointer(inspect.getfile(Memory))
        and m['code_sha256']=={p:sha(ROOT/p) for p in CODE}
        and m['completed_input_refs']==completed_input_refs(run),
        'Derived QUERY audit manifest/code/selection bytes differ')
    original=parent.manifest(run)
    require(m['parent_code_sha256']==original['code_sha256'],'Original47 code closure differs')
    return m,original,run

def gate_value(m,original,run):
    with forbid_network():value=_scope(run).audit(run,publish=False)
    value.update(query_event_audit_erratum=pointer(ERRATUM),
        query_event_audit_manifest=pointer(audit_dir(erratum())/'manifest.json'),
        query_event_audit_code_sha256=m['code_sha256'],
        original_failed_offline_audit=m['failed_offline_audit'],
        query_preprocessing='Frozen SDK _validate_and_trim_search_query, expected QUERY events only',
        original_job_and_reader_question_modified=False,retrieval_rerun=False)
    return value

def audit(run,publish=True):
    m,original,actual=manifest();require(Path(run).resolve()==actual,'Foreign completed retrieval run')
    value=gate_value(m,original,actual)
    return save_new(audit_dir(erratum())/'retrieval_gate.json',value) if publish else value

def source_value(m,original,run):
    spec=erratum();gate_ref=pointer(audit_dir(spec)/'retrieval_gate.json')
    # Only the assembler's parent gate pointer is redirected. Original jobs,
    # original47 code identity and all original source values remain unchanged.
    value=_scope(run,gate_ref)._source_value(run,original)
    value.update(query_event_audit_erratum=pointer(ERRATUM),
        query_event_audit_manifest=pointer(audit_dir(spec)/'manifest.json'),
        query_event_audit_code_sha256=m['code_sha256'])
    return value

def seal(run):
    m,original,actual=manifest();require(Path(run).resolve()==actual,'Foreign completed retrieval run')
    expected=gate_value(m,original,actual);gate=sealed_read(audit_dir(erratum())/'retrieval_gate.json')
    require(body(gate)==expected,'Derived QUERY event gate replay differs')
    return save_new(audit_dir(erratum())/'selected_inputs.json',source_value(m,original,actual))

def validate_source(path,fresh=False):
    path=Path(path).resolve();before=sha(path);source=sealed_read(path);spec=erratum()
    require(path==audit_dir(spec)/'selected_inputs.json','Foreign derived QUERY source')
    require(source['query_event_audit_erratum']==pointer(ERRATUM)
        and source['query_event_audit_manifest']==pointer(audit_dir(spec)/'manifest.json'),
        'Derived QUERY source audit refs differ')
    m,original,run=manifest(source['query_event_audit_manifest']['path'])
    if fresh:
        require(parent.parent.audit_banks(parent.APPROVED,publish=False)==
            body(bound(original['approved_effective20_gate'],sealed=True)),
            'Approved native UPDATE full closure changed')
    require(body(sealed_read(audit_dir(spec)/'retrieval_gate.json'))==gate_value(m,original,run)
        and body(source)==source_value(m,original,run),'Derived QUERY source/gate exact replay differs')
    for field in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):
        bound(source[field])
    require(sha(path)==before,'Derived QUERY source changed during validation');return source

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','audit','seal','check'));parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();functions={'prepare':prepare,'audit':audit,'seal':seal,
        'check':lambda run:{'jobs':len(validate_source(audit_dir(erratum())/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(functions[args.command](args.run_dir),ensure_ascii=False))
if __name__=='__main__':main()
