"""Read-only joint firstpass candidate admission; no seal/recover/API action."""
import json,hashlib,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import audit_five_arm_pre_recovery_h_c1_v1 as pretool
from tools.run_sdk20_modules import digest,save_new,now
from tools.check_experiment_health import shared_read
from tools import run_boundary_variants_parallel_v1 as processes
from jev_mem.local_bm25_context_v2 import forbid_network
PRE=ROOT/'research/planned_execution_2026-10-03/five_arm_pre_recovery_h_c1_audit.json'
PRE_SHA='64712a50f7cdbb03b95d2134cae632a60a4d112a5a8d9eedc5204cffed68d813'
C2=ROOT/'runs/boundary_neighbor_qa_full1540_v2_20261003'
def sha(p):return hashlib.sha256(shared_read(p)).hexdigest()
def pointer(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(shared_read(p))
def sealed(p):
    v=read(p);assert v['content_sha256']==digest({k:x for k,x in v.items() if k!='content_sha256'});return v
def pre_recheck():
    assert sha(PRE)==PRE_SHA;v=sealed(PRE)
    assert v['schema']=='five-arm-partial-pre-recovery-independent-audit-v1' and v['candidate_total']==417
    assert pointer(v['auditor']['path'])==v['auditor']
    for r in v['runs']:
        for p,h in r['source_file_hashes'].items():assert sha(p)==h,p
    cohorts=[]
    for r in v['runs']:
        m=read(r['manifest']['path']);jobs=m.get('jobs')
        if jobs is None:jobs=sealed(r['source_jobs']['path'])['jobs']
        for arm in r['status_counts_by_arm']:
            cohorts.append({j['query_id']:{k:j[k] for k in ('conversation','question','reference','category','reference_date')} for j in jobs if j['method']==arm})
    assert all(digest(c)==v['cohort_sha256'] and len(c)==1540 for c in cohorts)
    return v,cohorts[0]
def finish_seal(r):
    p=C2/'firstpass_seal.json'
    while not p.exists():time.sleep(15)
    v=sealed(p);o=sealed(C2/'run_outcome.json')
    assert v['schema']==pretool.cdriver.SCHEMA+'-firstpass-seal' and v['protocol_sha256']==r['protocol_sha256'] and v['total']==1540
    assert v['manifest']==r['manifest'] and v['outcome']==r['outcome'] and v['source_jobs']==r['source_jobs'] and v['stage_inventory']==o['results']
    assert v['model_calls']==0 and v['unknown_policy']=='unknown artifacts retained; never implicit redispatch'
    r['firstpass_seal']=pointer(p);r['source_file_hashes'][str(p.resolve())]=sha(p)
    r['blockers']=[b for b in r['blockers'] if b!='C1 auto firstpass seal not yet published; no seal helper invoked by this audit']
    assert not r['blockers']
def phase_worker(r):
    run=Path(r['run_dir']);p=run/('execution_intent.json' if r['total']==4620 else 'qa_execution_intent.json');v=read(p)
    assert v['protocol_sha256']==r['protocol_sha256']
    try: identity=processes.process_identity(v['pid'])
    except RuntimeError:identity=None
    # Allow the original auto seal process to finish. Never kill or restart it.
    while identity is not None:
        time.sleep(10)
        try:identity=processes.process_identity(v['pid'])
        except RuntimeError:identity=None
    return {'execution_intent':pointer(p),'pid':v['pid'],'intent_created_utc':v['created_utc'],'current_pid_absent':True,
            'closure_basis':'Original outcome emitted after worker futures joined; auto firstpass seal exists for C; original launcher PID no longer present.'}
def main():
    out=ROOT/'research/planned_execution_2026-10-03/five_arm_firstpass_recovery_admission_v1.json';assert not out.exists()
    assert (C2/'run_outcome.json').exists(),'Wait for actual C2 outcome; never synthesize or seal'
    with forbid_network() as guards:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fp=pool.submit(pre_recheck);fc=pool.submit(pretool.one,C2,pretool.cdriver,True)
            pre,cohort=fp.result();c2=fc.result()
        assert not any(g.call_count for g in guards)
        finish_seal(c2)
        actual=c2.pop('cohort_by_arm');assert set(actual)=={'boundary_neighbor'} and actual['boundary_neighbor']==cohort
        reports=pre['runs']+[c2]
        for r in reports:
            for p,h in r['source_file_hashes'].items():assert sha(p)==h,p
            r['worker_closure']=phase_worker(r)
        counts={a:n for r in reports for a,n in r['status_counts_by_arm'].items()}
        assert set(counts)=={'hybrid_rank','hybrid_types','hybrid_types_no_relation','boundary_multiscale','boundary_neighbor'}
        assert all(v['reader_complete']==1540 for v in counts.values()) and sum(r['total'] for r in reports)==7700
        candidate_ids=[c['job_id'] for r in reports for c in r['candidates']];assert len(set(candidate_ids))==len(candidate_ids)
        assert all(c['known_failure_passed'] and c['successful_reader_raw_replay_passed'] and c['compat_raw_replay_passed'] and c['attempt_kind']=='judge_only_after_successful_reader' for r in reports for c in r['candidates'])
        assert not any(r['blockers'] for r in reports)
        targets=[str(Path(r['run_dir']).with_name(Path(r['run_dir']).name+'_recovery_v1')) for r in reports]
        assert not any(Path(p).exists() for p in targets),'Existing fixed recovery namespace requires review; do not overwrite'
        for r in reports:
            for p,h in r['source_file_hashes'].items():assert sha(p)==h,p
        assert not any(g.call_count for g in guards)
    summaries=[]
    for p in (ROOT/'research/planned_execution_2026-10-03/hybrid_firstpass_summary.json',):
        summaries.append({'pointer':pointer(p),'role':'Previously verified H full raw summary; candidates independently rechecked in bound partial audit'})
    value={'schema':'five-arm-firstpass-recovery-admission-v1','created_utc':now(),'status':'five_firstpass_arms_closed_known_failure_candidates_verified',
        'firstpass_candidate_admission_passed':True,'recovery_preparation_allowed':True,'recovery_api_execution_allowed':False,
        'api_execution_requires':'Root must prepare the three fixed manifests, compare exact candidates/parent_results/Reader inheritance against this admission, then separately authorize execution.',
        'reused_frozen_pre_audit':pointer(PRE),'reviewed_firstpass_slots':7700,'planned_per_arm':1540,'exact_cohort_sha256':digest(cohort),
        'runs':reports,'candidate_total':len(candidate_ids),'candidate_ids_sha256':digest(sorted(candidate_ids)),
        'candidate_counts_by_arm':{a:v['strict_failure'] for r in reports for a,v in r['candidate_counts_by_arm'].items()},
        'strict_valid_no_excluded_total':sum(r['strict_valid_no_excluded'] for r in reports),
        'all_successful_reader_inheritance_only':True,'blockers':[],'fixed_future_recovery_paths':targets,'future_recovery_namespaces_exist':False,
        'existing_summary_evidence':summaries,'C_full_firstpass_summary_status':'Not asserted complete. Existing summary v2 has rawmanifest missing-jobs bookkeeping issue; separate offline v3 fix/review is pending. Candidate audit uses frozen recovery_jobs on exact hydrated execution contract, without changing QA policy or code.',
        'verification_scope':['All7700 four-stage pointers/seals/identity/job/facts-order/cohort and combined QA consumer fields checked; all candidate known_failure raw/signature/context/intent/physical sequences passed.',
            'Every candidate successful Reader raw/result/official metrics and compatible Judge interpretation replayed; valid strict NO excluded, compatible YES/NO candidates retained uniformly.',
            'H/C1 partial source hash inventory reread unchanged, C2 current read inventory unchanged. Does not claim all7700 Reader/Judge raw outputs independently replayed by a complete new metric summary.',
            'Known source semantic date errors/original c3 entity defects remain untouched and outside QA technical recovery. No correctness-based retry or second recovery.'],
        'model_calls':0,'source_mutations':0,'recovery_namespaces_created':0,'seal_helpers_invoked':0,
        'auditor':pointer(Path(__file__)),'candidate_auditor':pointer(Path(pretool.__file__))}
    save_new(out,value);print(json.dumps({'output':str(out),'sha256':sha(out),'candidate_total':len(candidate_ids),'candidate_counts':value['candidate_counts_by_arm'],'blockers':[]}))
if __name__=='__main__':main()
