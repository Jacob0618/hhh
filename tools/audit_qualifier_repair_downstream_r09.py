"""Explicit posthoc R09 downstream exact admission; historical numeric parity failed."""
from __future__ import annotations
import argparse,copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import audit_qualifier_repair_query_trim_r09 as query
from tools import run_evidence_exchange_qa_r01 as qa
from jev_mem.sdk20_cached_api import digest as request_hash
from jev_mem.local_bm25_context_v2 import forbid_network
parent=query.parent
require,pointer,bound,sealed_read,save_new,sha,digest,body=query.require,query.pointer,query.bound,query.sealed_read,query.save_new,query.sha,query.digest,query.body
SOURCE_SCHEMA=query.SOURCE_SCHEMA
PROTOCOL=ROOT/'research/jev_autoresearch_2026-10-04/R09_downstream_exact_admission_protocol_v1.json'
PROTOCOL_SHA='a9c0589e19324564e8680f727f1010686b5c90137c507ac06ac5f7e675ae49f5'
MANIFEST_SCHEMA='qualifier-repair-r09-downstream-exact-manifest-v1'
CODE=tuple(dict.fromkeys((*query.CODE,'tools/audit_qualifier_repair_downstream_r09.py',
    'tools/run_qualifier_repair_downstream_qa_r09.py','tools/summarize_qualifier_repair_downstream_r09.py')))

def protocol():
    require(sha(PROTOCOL)==PROTOCOL_SHA,'Downstream admission protocol changed')
    spec=sealed_read(PROTOCOL)
    for ref in spec['refs'].values():require(pointer(ref['path'])==ref,'Downstream protocol proof bytes changed')
    require(spec['refs']['query_event_erratum']==pointer(query.ERRATUM),'Frozen QUERY erratum differs')
    query.erratum();return spec
def output_dir(spec):return Path(spec['implementation']['outputs_directory']).resolve()
def completed_run():return query.completed_run(query.erratum())

def projection(selection):
    """Specific structural comparison only; every capture is also self-replayed."""
    value=parent.selection_body(selection);capture=value['capture']
    def floats(values):
        for key,v in values.items():
            require(type(v) is float and math.isfinite(v),'Permitted score must be finite float')
            values[key]='PERMITTED_RECORDED_FLOAT_VALUE'
    floats(capture['inputs']['entity_boosts']);floats(capture['native_scores'])
    for field in ('eligible','base20'):
        for row in capture[field]:
            require('score' in row,'Derived score field missing')
            score={'score':row['score']};floats(score);row['score']=score['score']
    require(type(capture['capture_sha256']) is str and len(capture['capture_sha256'])==64,'Capture hash shape differs')
    capture['capture_sha256']='PERMITTED_DERIVED_CAPTURE_HASH'
    return value

def compare_pair(old,new,job,bank,snapshot):
    expected={'schema':parent.SCHEMA,'status':'complete','conversation':job['conversation'],
        'query_id':job['query_id'],'model_calls':0}
    require(all(old[k]==v==new[k] for k,v in expected.items()) and
        old['method']==new['method'] and old['method'] in ('cheap','jev'),'Bound pair selection identity differs')
    require(projection(old)==projection(new),'Unexplained historical/current field difference')
    for s in (old,new):
        require(parent.old.validate_capture_bank(s['capture'],bank,snapshot)==s['selected_ids']
            and s['selected_facts']==parent.old.selected_facts(bank,s['selected_ids'])
            and s['original_job_sha256']==digest(job),'Independent capture/payload/source replay differs')
    signatures=[];serialized=[]
    for s in (old,new):
        actual=copy.deepcopy(job);actual.update(method=s['method'],selected_ids=s['selected_ids'],selected_facts=s['selected_facts'])
        signature=qa.request_signature(actual,'reader');signatures.append(signature)
        serialized.append(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode('utf-8'))
    require(signatures[0]==signatures[1] and serialized[0]==serialized[1],'Complete Reader signature differs')
    return {'reader_request_sha256':request_hash(signatures[0]),'serialized_signature_bytes':len(serialized[0]),
        'independent_own_boost_capture_replay':True,'full_reader_signature_and_canonical_utf8_exact':True,
        'original_strict_capture_equal':parent.selection_body(old)==parent.selection_body(new)}

def pairs(spec,original):
    proof=bound(spec['refs']['actual_full_reader_proof'],sealed=True)
    census=bound(spec['refs']['observed_difference_census'])
    require(proof['denominator']==proof['equal_full_requests']==1900 and proof['different_full_requests']==0
        and census['n']==1900 and census['exact_match_slots']==604 and census['mismatch_slots']==1296,
        'Bound actual1900 proof population differs')
    rows=proof['rows'];identities=[r['identity'] for r in rows]
    require(len(rows)==1900 and len(set(identities))==1900 and set(identities)==
        set(original['historical_retrieval']['selections']),'Bound historical pair identity set differs')
    run=completed_run()
    for row in rows:
        arm,qid=row['identity'].split('/');ci=int(qid.split('_')[0][1:])
        require(row['original_selection']==original['historical_retrieval']['selections'][row['identity']]
            and row['current_selection']==pointer(parent.retrieval_folder(run,ci,arm)/f'selections/{qid}.json'),
            'Immutable bound historical pair differs')
        for ref in (row['original_selection'],row['current_selection']):
            require(pointer(ref['path'])==ref and census['input_refs'].get(ref['path'])==ref['sha256'],
                'Reader/census pair bytes disagree')
    return rows

def prepare(run):
    spec=protocol();run=Path(run).resolve();out=output_dir(spec)
    require(run==completed_run() and not out.exists(),'Fresh downstream audit directory required')
    original=parent.manifest(run);rows=pairs(spec,original)
    value={'schema':MANIFEST_SCHEMA,'downstream_admission_protocol':pointer(PROTOCOL),
        'query_event_audit_erratum':pointer(query.ERRATUM),'completed_retrieval_run':str(run),
        'retrieval_manifest':pointer(run/'run_manifest.json'),'completed_retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'completed_input_refs':query.completed_input_refs(run),'historical_pairs_sha256':digest(rows),
        'code_sha256':{p:sha(ROOT/p) for p in CODE},'parent_code_sha256':original['code_sha256'],
        'original_strict_parity':spec['original_strict_parity'],'new_model_calls':0,'retrieval_rerun':False,
        'posthoc_technical_criterion':True,'historical_numeric_identity_claimed':False}
    return save_new(out/'manifest.json',value)

def manifest(path=None):
    spec=protocol();expected=output_dir(spec)/'manifest.json'
    require(path is None or Path(path).resolve()==expected,'Foreign downstream audit manifest')
    m=sealed_read(expected);run=completed_run();original=parent.manifest(run)
    require(m['schema']==MANIFEST_SCHEMA and m['completed_retrieval_run']==str(run)
        and m['downstream_admission_protocol']==pointer(PROTOCOL)
        and m['query_event_audit_erratum']==pointer(query.ERRATUM)
        and m['retrieval_manifest']==pointer(run/'run_manifest.json')
        and m['completed_retrieval_outcome']==pointer(run/'retrieval_outcome.json')
        and m['completed_input_refs']==query.completed_input_refs(run)
        and m['historical_pairs_sha256']==digest(pairs(spec,original))
        and m['parent_code_sha256']==original['code_sha256']
        and m['code_sha256']=={p:sha(ROOT/p) for p in CODE}
        and m['original_strict_parity']==spec['original_strict_parity']
        and m['new_model_calls']==0 and m['retrieval_rerun'] is False
        and m['posthoc_technical_criterion'] is True and m['historical_numeric_identity_claimed'] is False,
        'Downstream manifest/code/actual pair binding differs')
    return m,original,run

def gate_value(m,original,run):
    spec=protocol();rows=pairs(spec,original);gate=bound(original['approved_effective20_gate'],sealed=True)
    jobs={j['query_id']:j for j in parent.old.old_jobs(bound(original['protocol']['frozen_source']))}
    snapshots={};banks={};reviewed=[]
    with forbid_network():
        for row in rows:
            arm,qid=row['identity'].split('/');job=jobs[qid];key=f'{job["conversation"]}/{arm}'
            if key not in banks:
                banks[key]=bound(gate['banks'][key],sealed=True)
                snapshots[key]=parent.old.snapshot(banks[key]['native_dir'])
            result=compare_pair(bound(row['original_selection'],sealed=True),bound(row['current_selection'],sealed=True),
                job,banks[key],snapshots[key])
            require(result['reader_request_sha256']==row['original_reader_request_sha256']==row['current_reader_request_sha256'],
                'Actual Reader signature differs from bound1900 proof')
            reviewed.append({**row,**result})
        private=parent.parent.support.isolated(parent,{'validate_cache_events':query.validate_cache_events,
            'selection_body':projection})
        value=private.audit(run,publish=False)
    require(sum(r['original_strict_capture_equal'] for r in reviewed)==604,'Recorded strict parity census changed')
    value.pop('historical_parity_slots');value.pop('historical_parity')
    value.update(original_strict_parity={k:spec['original_strict_parity'][k] for k in ('status','total','exact','different')},historical_pair_reviews=reviewed,
        downstream_reader_exact={'total':1900,'equal':1900,'different':0},
        capture_self_replay={'historical':1900,'current':3080},
        query_event_audit_erratum=pointer(query.ERRATUM),query_event_audit_manifest=pointer(output_dir(spec)/'manifest.json'),
        downstream_admission_protocol=pointer(PROTOCOL),downstream_admission_manifest=pointer(output_dir(spec)/'manifest.json'),
        downstream_admission_code_sha256=m['code_sha256'],
        observed_difference_census=spec['refs']['observed_difference_census'],actual_full_reader_proof=spec['refs']['actual_full_reader_proof'],
        criterion='Posthoc downstream exact evidence; original strict numeric capture parity FAILED',
        entity_boosts_independently_regenerated=False,historical_thread_trace_reconstructed=False)
    return value

def audit(run,publish=True):
    m,original,actual=manifest();require(Path(run).resolve()==actual,'Foreign completed retrieval run')
    value=gate_value(m,original,actual)
    return save_new(output_dir(protocol())/'retrieval_gate.json',value) if publish else value

def source_value(m,original,run):
    spec=protocol();gate_ref=pointer(output_dir(spec)/'retrieval_gate.json')
    value=query._scope(run,gate_ref)._source_value(run,original)
    value.update(query_event_audit_erratum=pointer(query.ERRATUM),
        query_event_audit_manifest=pointer(output_dir(spec)/'manifest.json'),
        downstream_admission_protocol=pointer(PROTOCOL),downstream_admission_manifest=pointer(output_dir(spec)/'manifest.json'),
        downstream_admission_code_sha256=m['code_sha256'])
    return value

def seal(run):
    m,original,actual=manifest();require(Path(run).resolve()==actual,'Foreign completed retrieval run')
    require(body(sealed_read(output_dir(protocol())/'retrieval_gate.json'))==gate_value(m,original,actual),
        'Downstream exact gate replay differs')
    return save_new(output_dir(protocol())/'selected_inputs.json',source_value(m,original,actual))

def validate_source(path,fresh=False):
    path=Path(path).resolve();before=sha(path);source=sealed_read(path);spec=protocol()
    require(path==output_dir(spec)/'selected_inputs.json' and source['downstream_admission_protocol']==pointer(PROTOCOL)
        and source['downstream_admission_manifest']==pointer(output_dir(spec)/'manifest.json')
        and source['query_event_audit_erratum']==pointer(query.ERRATUM)
        and source['query_event_audit_manifest']==source['downstream_admission_manifest'],'Downstream source refs differ')
    m,original,run=manifest(source['downstream_admission_manifest']['path'])
    if fresh:require(parent.parent.audit_banks(parent.APPROVED,publish=False)==
        body(bound(original['approved_effective20_gate'],sealed=True)),'Approved native full closure differs')
    require(body(sealed_read(output_dir(spec)/'retrieval_gate.json'))==gate_value(m,original,run)
        and body(source)==source_value(m,original,run),'Downstream source/gate exact replay differs')
    for field in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):bound(source[field])
    require(sha(path)==before,'Downstream source changed during audit');return source

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','audit','seal','check'))
    parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    functions={'prepare':prepare,'audit':audit,'seal':seal,'check':lambda run:{'jobs':len(validate_source(output_dir(protocol())/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(functions[args.command](args.run_dir),ensure_ascii=False))
if __name__=='__main__':main()
