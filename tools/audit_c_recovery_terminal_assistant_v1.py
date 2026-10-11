"""Closed C recovery QA files only; no source locks/native DB/summary/API."""
import argparse,json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_sdk20_plain_qa_v1 as plain
from tools import run_boundary_variants_qa_v2 as driver
from tools import summarize_sdk20_plain_qa_v1 as replay
from tools import run_boundary_variants_parallel_v1 as processes
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.run_sdk20_modules import digest,save_new,now
from jev_mem.local_bm25_context_v2 import forbid_network
ADMISSION=ROOT/'research/planned_execution_2026-10-03/five_arm_firstpass_recovery_admission_v1.json'
ADMISSION_SHA='05f16ee5187ef2d82edf9d77ed8b7b0316def902296fb5073462478d8a9963f8'
RUNS={'c1':('boundary_multiscale','boundary_multiscale_qa_full1540_v2_20261003'),
      'c2':('boundary_neighbor','boundary_neighbor_qa_full1540_v2_20261003')}
def verify(s,p):
    v=s.read(p);assert v['content_sha256']==digest({k:x for k,x in v.items() if k!='content_sha256'});return v
def ptr(s,p):return {'path':str(Path(p).resolve()),'sha256':s.sha(p)}
def bound(s,p):assert ptr(s,p['path'])==p;return verify(s,p['path'])
def audit(which):
    qa,name=RUNS[which];parent=ROOT/'runs'/name;run=parent.with_name(parent.name+'_recovery_v1');snap=Snapshot()
    admission=verify(snap,ADMISSION);assert snap.sha(ADMISSION)==ADMISSION_SHA
    prior=next(r for r in admission['runs'] if r['run_dir']==str(parent));expected={c['job_id']:c for c in prior['candidates']}
    m=snap.read(run/'run_manifest.json');out=verify(snap,run/'run_outcome.json')
    assert m['schema']==driver.SCHEMA and m['phase_kind']=='technical_recovery' and m['arm']==qa and m['policy']==driver.POLICY and m['cache_roots']==[]
    assert m['protocol_sha256']==digest({k:v for k,v in m.items() if k!='protocol_sha256'})
    assert all(snap.sha(ROOT/p)==h for p,h in m['code_sha256'].items())
    assert m['parent']=={'run_dir':str(parent),'manifest':prior['manifest'],'outcome':prior['outcome'],'firstpass_seal':prior['firstpass_seal']}
    assert m['source_manifest']==prior['source_jobs']
    for ref in (prior['manifest'],prior['outcome'],prior['source_jobs'],prior['firstpass_seal']):assert ptr(snap,ref['path'])==ref
    assert digest(m['jobs'])==prior['candidate_jobs_sha256'] and {j['job_id'] for j in m['jobs']}==set(expected)
    assert out['terminal'] is True and out['protocol_sha256']==m['protocol_sha256'] and out['phase_kind']=='technical_recovery'
    assert out['total']==len(m['jobs'])==len(out['results']) and out['planned_per_arm']==1540 and out['finished_utc']
    assert not any(r.get('local_errors') for r in out['outcomes'])
    rows={r['job_id']:r for r in out['results']};assert len(rows)==len(expected) and set(rows)==set(expected)
    logs,indexed,ledgerrefs=replay.ledger(snap,run,m)
    assert all(r['context']['component']=='judge' and r['stage']==replay.evaluator.JUDGE_STAGE and r['model']==replay.evaluator.JUDGE_MODEL and r['operation']=='generateContent' for r in logs)
    assert not list(run.rglob('selection/*.json')) and not (run/'retrieval_outcome.json').exists()
    evidence=[];counts=Counter();resolved=Counter();compat_labels=Counter();failures=Counter()
    for j in m['jobs']:
        exp=expected[j['job_id']];assert digest(j)==exp['candidate_job_sha256'] and j['parent_results']==exp['parent_results'] and j['reader_source']==exp['reader_source']
        assert j['attempt_kind']=='judge_only_after_successful_reader'
        ident=plain.identity(j,m);paths=plain.stage_files(run,j);values={}
        for stage,p in paths.items():
            assert rows[j['job_id']][stage]==ptr(snap,p);v=verify(snap,p);assert all(v.get(k)==x for k,x in ident.items());values[stage]=v;counts[stage+'_'+v['status']]+=1
        reader=values['reader'];original_reader=bound(snap,j['reader_source']);first_judge=bound(snap,j['parent_results']['judge']);first_compat=bound(snap,j['parent_results']['compat'])
        assert original_reader['status']=='complete' and reader['status']=='complete' and reader['result']==original_reader['result']
        assert reader['inherited_from']==j['reader_source'] and reader['model_calls']==0 and reader['api_events']==[] and reader['error'] is None and reader['raw_invalid_label'] is None
        assert digest(reader['result'])==exp['reader_result_sha256'] and not paths['reader'].with_suffix('.intent.json').exists()
        api=run/f"conv_{j['conversation']}/api";ctx=plain.stage_context(j,m,'judge')
        judge=replay.audit_stage(snap,paths['judge'],{**ident,'stage':'judge','reader_sha256':snap.sha(paths['reader'])},api,ctx,indexed,
            lambda r,j=j,reader=reader:plain.judge_answer(r,j['question'],j['reference'],reader['result']['prediction']['answer']))
        assert judge==values['judge'] and judge['status'] in ('complete','failed')
        if judge['status']=='failed':
            plain.known_failure(j,m,judge,ptr(snap,paths['judge']),'judge');failures[judge['error']['type']]+=1
        ev=judge['api_events'][0];assert ev['request_hash']==first_judge['api_events'][0]['request_hash']==exp['request_hash'] and ev['context']==ctx
        ip=api/'request_intents'/(ev['request_hash']+'.json');intent=snap.read(ip)
        assert intent['request_hash']==ev['request_hash'] and intent['retries']==2 and intent['stage']==ev['stage']
        oldintent=snap.read(exp['request_intent']['path']);assert ptr(snap,exp['request_intent']['path'])==exp['request_intent'] and intent['signature']==oldintent['signature']
        physical=[r for r in logs if r['request_hash']==ev['request_hash'] and r['stage']==ev['stage'] and r.get('cache_hit') is not True]
        assert 1<=len(physical)<=3 and [r['attempt'] for r in physical]==list(range(1,len(physical)+1)) and all(r['context']==intent['context'] for r in physical)
        rp=api/'api_cache'/(ev['request_hash']+'.json');raw=snap.read(rp);assert ptr(snap,rp)['sha256']==ev['cache_sha256'] and raw['request']==intent['signature'] and physical[-1]['status']==200
        compatibility=plain.replay_source(api,judge,ctx,indexed,question=j['question'],reference=j['reference'],reader_text=reader['result']['prediction']['answer'])
        cp=values['compat'];assert cp['status']==compatibility['interpretation']['status'] and cp['interpretation']==compatibility['interpretation'] and cp['source_evidence']==compatibility['source_evidence']
        assert cp['strict']==ptr(snap,paths['judge']) and cp['model_calls']==0 and cp['raw_or_strict_modified'] is False and cp['error'] is None
        qa_row=values['qa'];resolution=plain.resolve_pair(first_judge,first_compat,judge,cp)
        assert all(qa_row[st]==ptr(snap,paths[st]) for st in ('reader','judge','compat'))
        assert qa_row['reader_metrics']==original_reader['result']['metrics'] and qa_row['reader_status']=='complete' and qa_row['judge_status']==judge['status']
        assert qa_row['status']==('complete' if judge['status']=='complete' else 'failed') and qa_row['judge_correct']==(judge['result']['correct'] if judge['status']=='complete' else None) and qa_row['resolved_judge']==resolution
        resolved[resolution['source']]+=1;compat_labels[cp['interpretation']['label']]+=1
        evidence.append({'job_id':j['job_id'],'candidate_job_sha256':digest(j),'parent_job_sha256':j['parent_job_sha256'],'parent_results':j['parent_results'],
            'reader_source':j['reader_source'],'reader':ptr(snap,paths['reader']),'reader_result_sha256':digest(reader['result']),'reader_metrics_sha256':digest(reader['result']['metrics']),
            'reader_complete_inherited_exact':True,'reader_model_calls':0,'judge':ptr(snap,paths['judge']),'compat':ptr(snap,paths['compat']),'qa':ptr(snap,paths['qa']),
            'judge_status':judge['status'],'judge_error':judge['error'],'compat_status':cp['status'],'resolved_judge':resolution,
            'judge_request_hash':ev['request_hash'],'same_original_request_hash':True,'request_intent':ptr(snap,ip),'raw':ptr(snap,rp),
            'candidate_event_context':ev['context'],'physical_owner_context':intent['context'],'physical_statuses':[r['status'] for r in physical],
            'physical_attempt_count':len(physical),'final_physical_status':200})
    changed=snap.unchanged();assert not changed,changed[:3]
    intentp=run/'qa_execution_intent.json';ei=snap.read(intentp);assert ei['protocol_sha256']==m['protocol_sha256']
    try:live=processes.process_identity(ei['pid'])
    except RuntimeError:live=None
    assert live is None,'Wait for original QA process to exit; do not interrupt or redispatch'
    physical=[r for r in logs if r.get('cache_hit') is not True]
    return {'schema':'closed-c-recovery-terminal-assistant-check-v1','created_utc':now(),'status':'closed_candidates_inherited_reader_and_judge_chain_verified','qa_arm':qa,
        'run_dir':str(run),'manifest':ptr(snap,run/'run_manifest.json'),'outcome':ptr(snap,run/'run_outcome.json'),'joint_firstpass_admission':ptr(snap,ADMISSION),
        'candidate_count':len(expected),'candidate_ids_sha256':digest(sorted(expected)),'candidate_jobs_sha256':digest(m['jobs']),
        'status_counts':dict(counts),'failed_judge_error_counts':dict(failures),'compat_label_counts':dict(compat_labels),'resolution_source_counts':dict(resolved),
        'reader_inheritance_count':len(expected),'reader_model_requests':0,'retrieval_or_writer_or_jev_requests':0,'all_judge_same_original_signature_and_hash':True,
        'ledger':{'physical_attempts':len(physical),'cache_hit_rows':len(logs)-len(physical),'physical_status_counts':dict(Counter(str(r.get('status')) for r in physical)),
            'distinct_physical_request_hashes':len({r['request_hash'] for r in physical}),'refs':ledgerrefs},
        'execution_intent':ptr(snap,intentp),'original_execution_pid':ei['pid'],'original_pid_absent':True,
        'checks':{'candidate_population_exact':True,'all_parent_pointers_unchanged':True,'all_reader_result_and_f1_inherited_exact':True,'reader_zero_api_and_no_intent':True,
            'judge_raw_intent_context_and_bounded_attempts_replayed':True,'compat_original_raw_replayed':True,'four_stage_consumer_fields_exact':True,'no_local_or_unknown_artifacts':True,'source_hashes_unchanged':True},
        'candidate_evidence':evidence,'source_file_hashes':{str(p):h for p,h in snap.files.items()},'blockers':[],
        'model_calls_in_audit':0,'source_mutations':0,'sqlite_connections':0,'source_lock_acquisitions':0,'summary_runs':0,'second_recovery_prepared':False,
        'limitations':['Closed QA recovery-file audit only, not a full1540 summary or renewed full20 source/native audit.',
            'Original strict failed artifacts remain failed even where compatible labels provide resolved final scores. Policy closed does not certify strict success or source/answer truth.',
            'Shared exact-signature cache in this recovery phase is one physical judgment, not independent votes; inherited Reader result is not regenerated.',
            'One uniform recovery completed; no second recovery/namespace/retry is permitted by this audit.'],
        'auditor':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=RUNS,required=True);a=p.parse_args()
    path=ROOT/'research/planned_execution_2026-10-03'/f'{a.arm}_recovery_terminal_assistant_check_v1.json';assert not path.exists()
    with forbid_network() as guards:r=audit(a.arm);assert not any(g.call_count for g in guards)
    save_new(path,r);print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':r['candidate_count'],'status_counts':r['status_counts'],'ledger':r['ledger'],'resolution':r['resolution_source_counts']}))
if __name__=='__main__':main()
