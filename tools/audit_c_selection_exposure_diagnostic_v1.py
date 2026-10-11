"""Read existing C selections only; no query execution, model, native DB or gold."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_sdk20_modules import digest,save_new,now
from tools.check_experiment_health import shared_read
from jev_mem.local_bm25_context_v2 import forbid_network
INPUTS={}
def sha(p):
    p=Path(p).resolve();h=hashlib.sha256(shared_read(p)).hexdigest()
    if str(p) in INPUTS:assert INPUTS[str(p)]==h
    INPUTS[str(p)]=h;return h
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):sha(p);return json.loads(shared_read(p))
def verify(p):
    x=read(p);assert x['content_sha256']==digest({k:v for k,v in x.items() if k!='content_sha256'});return x
def main():
    output=ROOT/'research/planned_execution_2026-10-03/c_selection_exposure_diagnostic_v1.json';assert not output.exists()
    checks=ROOT/'research/planned_execution_2026-10-03/checks/20261003T0658'
    targets=[]
    for rawarm,qa in (('c1_multiscale','boundary_multiscale'),('c2_neighbour','boundary_neighbor')):
        p=checks/('c1_conv3_failure_review.json' if rawarm=='c1_multiscale' else 'c2_conv3_failure_review.json');v=verify(p);cause=v['cause_evidence']
        targets.append({'kind':'original_unrepaired_orphan_entity_edge','source_arm':rawarm,'qa_arm':qa,'conversation':3,'fact_id':cause['fact_id'],
            'fact_text':cause['fact_text'],'source_group':cause['group_id'],'source_review':ref(p),'journal':cause['journal'],
            'entity_embedding_raws':cause['entity_embedding_raws'],'source_warning':'Original Writer failed due to missing entity link; metadata admission preserved native defect.'})
    semantic=verify(checks/'c_remaining_semantic_review.json')
    for finding in semantic['concrete_findings']:
        targets.append({'kind':'reviewed_source_date_error','source_arm':finding['arm'],'qa_arm':{'c1_multiscale':'boundary_multiscale','c2_neighbour':'boundary_neighbor'}[finding['arm']],
            'conversation':finding['conversation'],'fact_id':finding['fact_id'],'fact_text':finding['fact_text'],'source_group':finding['group_id'],
            'source_review':ref(checks/'c_remaining_semantic_review.json'),'journal':finding['journal'],'writer_extraction_raw':finding['raw'],
            'source_turn_id':finding['source_turn_id'],'finding':finding['finding'],'expected_date_or_scope':finding['calendar_or_scope_expectation']})
    runbindings=[]
    for qa,name in (('boundary_multiscale','boundary_multiscale_qa_full1540_v2_20261003'),('boundary_neighbor','boundary_neighbor_qa_full1540_v2_20261003')):
        run=ROOT/'runs'/name;m=read(run/'run_manifest.json');source=verify(run/'source_jobs.json');jobs=source['jobs']
        assert m['arm']==qa and source['arm']==qa and len(jobs)==1540 and len({j['query_id'] for j in jobs})==1540
        local=[t for t in targets if t['qa_arm']==qa];counts=Counter(j['conversation'] for j in jobs)
        for t in local:
            bp=m['banks'][str(t['conversation'])][qa];assert ref(bp['path'])==bp;b=verify(bp['path']);fact=next(f for f in b['facts'] if f['id']==t['fact_id'])
            assert fact['text']==t['fact_text'] and fact['input_scope']['group_id']==t['source_group']
            jp=t['journal'];assert ref(jp['path'])==jp;journal=read(jp['path']);assert any(f['id']==t['fact_id'] and f['text']==t['fact_text'] for f in journal['fact_records'])
            t.update(bank=bp,source_fact_sha256=digest(fact),source_scope=fact['input_scope'],source_observation_dates=fact['source_observation_dates'],
                original_write_status=b.get('original_write_status'),source_kind=b.get('source_kind'),arm_question_denominator=1540,conversation_question_denominator=counts[t['conversation']],
                semantic_candidate_hits=[],native_eligible_hits=[],final20_hits=[])
        for j in jobs:
            binding=j['lineage']['selection'];assert ref(binding['path'])==binding;s=verify(binding['path']);cap=s['capture']
            assert s['protocol_sha256']==m['protocol_sha256'] and s['status']=='complete' and s['error'] is None
            assert (s['conversation'],s['method'],s['query_id'])==(j['conversation'],qa,j['query_id'])
            assert s['bank_sha256']==j['lineage']['bank']['sha256']
            assert cap['capture_sha256']==digest({k:v for k,v in cap.items() if k!='capture_sha256'})
            assert cap['schema']=='sdk20-native-pool-v1' and cap['candidate_cap']==80 and len(cap['candidate_ids'])<=80
            assert s['selected_ids']==cap['base20_ids']==j['selected_ids'] and len(s['selected_ids'])==20 and len(set(s['selected_ids']))==20
            assert set(cap['eligible_ids'])<=set(cap['candidate_ids']) and set(s['selected_ids'])<=set(cap['eligible_ids'])
            assert [f['id'] for f in j['selected_facts']]==s['selected_ids']
            for t in local:
                fid=t['fact_id'];hit={'query_id':j['query_id'],'conversation':j['conversation'],'selection':binding,'source_job_sha256':digest(j)}
                if fid in cap['candidate_ids']:
                    assert j['conversation']==t['conversation'];t['semantic_candidate_hits'].append({**hit,'rank':cap['candidate_ids'].index(fid)+1})
                if fid in cap['eligible_ids']:
                    t['native_eligible_hits'].append({**hit,'rank':cap['eligible_ids'].index(fid)+1,'native_score':cap['native_scores'][fid],
                        'entity_boost_input':cap['inputs']['entity_boosts'].get(fid),'entity_boost_key_present':fid in cap['inputs']['entity_boosts']})
                if fid in s['selected_ids']:
                    sf=next(f for f in j['selected_facts'] if f['id']==fid);assert sf['text']==t['fact_text']
                    t['final20_hits'].append({**hit,'rank':s['selected_ids'].index(fid)+1,'selected_fact_text_exact':True})
        runbindings.append({'qa_arm':qa,'run_dir':str(run),'manifest':ref(run/'run_manifest.json'),'source_jobs':ref(run/'source_jobs.json'),
            'selections_checked':1540,'conversation_question_counts':dict(counts)})
    for t in targets:
        for key in ('semantic_candidate','native_eligible','final20'):
            t[key+'_count']=len(t[key+'_hits']);t[key+'_query_ids']=[h['query_id'] for h in t[key+'_hits']]
    for p,h in INPUTS.items():assert hashlib.sha256(shared_read(p)).hexdigest()==h
    v={'schema':'c-existing-selection-exposure-diagnostic-v1','created_utc':now(),'status':'five_known_facts_existing_selection_exposure_quantified',
        'runs':runbindings,'targets':targets,'selection_field_definitions':{'semantic_candidate':'capture.candidate_ids: original semantic pool <=80; not the physical union of all backend returns',
            'native_eligible':'capture.eligible_ids: original full scored eligible ranking after native threshold',
            'final20':'selection.selected_ids == capture.base20_ids == source_jobs selected_ids, exact order; supplied selected_facts text verified'},
        'source_file_hashes':INPUTS,'model_calls':0,'retrieval_calls':0,'sqlite_connections':0,'source_mutations':0,'qa_reference_or_correctness_used':False,
        'limitations':['Existing-selection exposure only, not a repaired-index counterfactual or evidence of actual QA correctness impact.',
            'Not selected does not prove no entity-edge effect: a missing boost can alter ranking or exclude this fact, and other candidates/Reader evidence can change.',
            'Selected fact exposure does not prove Reader used it or generated an incorrect answer. No Reader/Judge/reference/gold comparisons were performed.',
            'The three date-error facts were frozen source-reviewed examples, not selected based on answer correctness; no samples or answers changed.'],
        'auditor':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    save_new(output,v);print(json.dumps({'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'targets':[{'kind':t['kind'],'arm':t['qa_arm'],'fact':t['fact_id'],'semantic':t['semantic_candidate_count'],'eligible':t['native_eligible_count'],'final20':t['final20_count']} for t in targets]}))
if __name__=='__main__':
    with forbid_network() as guards:main();assert not any(g.call_count for g in guards)
