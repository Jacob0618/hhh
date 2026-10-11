"""Read-only pre-recovery candidate audit. Never prepares recovery or seals a run."""
from pathlib import Path
import sys,json,hashlib,copy
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_planned_plain_qa_v1 as hdriver
from tools import run_boundary_variants_qa_v2 as cdriver
from tools import run_sdk20_plain_qa_v1 as plain
from tools import summarize_sdk20_plain_qa_v1 as audit
from tools.summarize_mem0_sparse1540 import Snapshot
from tools.run_sdk20_modules import digest,save_new,now
from jev_mem.local_bm25_context_v2 import forbid_network

def verify(snapshot,p):
    value=snapshot.read(p)
    assert value['content_sha256']==digest({k:v for k,v in value.items() if k!='content_sha256'}),str(p)
    return value
def ptr(snapshot,p):return {'path':str(Path(p).resolve()),'sha256':snapshot.sha(p)}
def jobhash(ids):return digest(sorted(ids))
def one(run,driver,is_c):
    evaluator=plain if is_c else driver
    snap=Snapshot();mp=run/'run_manifest.json';op=run/'run_outcome.json';m=snap.read(mp);out=verify(snap,op)
    assert m['schema']==driver.SCHEMA and m['policy']==driver.POLICY and m['phase_kind']=='firstpass'
    if is_c:
        assert m['cache_roots']=={str(ci):[str((run/f'embedding_cache_pack/conv_{ci}/api_cache').resolve())] for ci in range(10)}
    else:assert m['cache_roots']==[]
    assert m['protocol_sha256']==digest({k:v for k,v in m.items() if k!='protocol_sha256'})
    assert all(snap.sha(ROOT/name)==h for name,h in m['code_sha256'].items())
    if is_c:
        source=verify(snap,run/'source_jobs.json');assert source['schema']==driver.SCHEMA+'-sources' and source['arm']==m['arm']
        m={**m,'jobs':source['jobs'],'source_manifest':ptr(snap,run/'source_jobs.json')}
    else:
        source=verify(snap,m['source_manifest']['path']);assert ptr(snap,m['source_manifest']['path'])==m['source_manifest'] and source['jobs']==m['jobs']
    jobs={plain.job_key(j):j for j in m['jobs']};total=1540 if is_c else 4620
    assert len(jobs)==len(m['jobs'])==total and out['terminal'] is True and out['finished_utc'] and out['protocol_sha256']==m['protocol_sha256']
    assert out['total']==len(out['results'])==total and {plain.job_key(r) for r in out['results']}==set(jobs)
    assert not any(o.get('local_errors') for o in out['outcomes'])
    snapshot_counts=defaultdict(Counter);strict_valid_no=[];failed_keys=[];stages={};blockers=[]
    for row in out['results']:
        j=jobs[plain.job_key(row)];ident=evaluator.identity(j,m);assert row['job_id']==j['job_id']
        values={}
        for name,p in plain.stage_files(run,j).items():
            if row[name] is None:raise ValueError('Unknown pointer '+j['job_id']+'/'+name)
            assert row[name]==ptr(snap,p);v=verify(snap,p);assert all(v.get(k)==x for k,x in ident.items())
            values[name]=v;snapshot_counts[j['method']][name+'_'+v['status']]+=1
        r,g,cp,qa=(values[s] for s in ('reader','judge','compat','qa'))
        assert r['status']=='complete' and not cp.get('error')
        assert all(qa[name]==row[name] for name in ('reader','judge','compat'))
        assert qa['reader_metrics']==r['result']['metrics'] and qa['reader_status']==r['status'] and qa['judge_status']==g['status']
        assert qa['resolved_judge']==plain.resolve_pair(g,cp)
        if g['status']=='failed':failed_keys.append(j['job_id'])
        elif g['status']=='complete':
            assert type(g['result']['correct']) is bool
            if not g['result']['correct']:strict_valid_no.append(j['job_id'])
        else:raise ValueError('Unknown Judge state '+j['job_id'])
        stages[j['job_id']]=(row,values)
    # Original pure candidate function invokes the exact frozen known_failure
    # replayer for every strict failure; it never constructs ModelAPI or writes.
    candidates=driver.recovery_jobs(m,out)
    assert {j['job_id'] for j in candidates}==set(failed_keys) and not set(strict_valid_no)&set(failed_keys)
    official,provenance=audit.verified_official_metric();assert provenance==m['official_metric_provenance']
    ledgerfn=FunctionType(audit.ledger.__code__,dict(audit.ledger.__globals__,runner=evaluator),'private_ledger',audit.ledger.__defaults__)
    logs,indexed,ledgerrefs=ledgerfn(snap,run,m)
    items=[];candidate_counts=defaultdict(Counter)
    for j in candidates:
        jid=j['job_id'];row,vs=stages[jid];original=jobs[plain.job_key(j)]
        assert j['parent_job_sha256']==digest(original) and j['parent_results']=={s:row[s] for s in ('reader','judge','compat','qa')}
        assert j['attempt_kind']=='judge_only_after_successful_reader' and j['reader_source']==row['reader']
        paths=plain.stage_files(run,j);api=run/f"conv_{j['conversation']}/api";ident=evaluator.identity(original,m)
        # Independently replay each successful Reader selected for inheritance.
        reader=audit.audit_stage(snap,paths['reader'],{**ident,'stage':'reader'},api,evaluator.stage_context(original,m,'reader'),indexed,lambda replay,j=original:evaluator.reader_call(replay,j,official))
        assert reader['status']=='complete' and reader['result']==vs['reader']['result']
        event=vs['judge']['api_events'][0];key=event['request_hash'];ip=api/'request_intents'/(key+'.json');request=snap.read(ip)
        assert event['context']==evaluator.stage_context(original,m,'judge') and request['retries']==2 and request['stage']==event['stage']
        assert key==hashlib.sha256(json.dumps(request['signature'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        physical=[e for e in logs if e.get('request_hash')==key and e.get('stage')==event['stage'] and e.get('cache_hit') is not True]
        assert [e['attempt'] for e in physical]==list(range(1,len(physical)+1)) and 1<=len(physical)<=3
        assert all(e['context']==request['context'] for e in physical)
        rawpath=api/'api_cache'/(key+'.json');rawref=None
        if rawpath.exists():
            raw=snap.read(rawpath);rawref=ptr(snap,rawpath)
            assert event['cache_sha256']==rawref['sha256'] and raw['request']==request['signature'] and physical[-1]['status']==200
        for stage in ('reader','judge'):snap.read(paths[stage].with_suffix('.intent.json'))
        # Compatibility label interpretation is also independently replayed;
        # valid compat YES/NO remain strict-failure candidates alike.
        replay=plain.replay_source(api,vs['judge'],evaluator.stage_context(original,m,'judge'),indexed,question=j['question'],reference=j['reference'],reader_text=reader['result']['prediction']['answer'])
        assert vs['compat']['interpretation']==replay['interpretation'] and vs['compat']['source_evidence']==replay['source_evidence'] and vs['compat']['status']==replay['interpretation']['status']
        candidate_counts[j['method']].update({'strict_failure':1,'reader_inherited':1,'compat_'+vs['compat']['status']:1})
        label=(vs['compat'].get('interpretation')or{}).get('label')
        if label is not None:candidate_counts[j['method']]['compat_label_'+str(label)]+=1
        items.append({'job_id':jid,'method':j['method'],'query_id':j['query_id'],'conversation':j['conversation'],
            'parent_job_sha256':j['parent_job_sha256'],'candidate_job_sha256':digest(j),'parent_results':j['parent_results'],'reader_source':j['reader_source'],
            'selected_ids':original['selected_ids'],'selected_facts_sha256':digest(original['selected_facts']),
            'reader_result_sha256':digest(reader['result']),'reader_unchanged_on_recovery_contract':True,'attempt_kind':j['attempt_kind'],
            'known_failure_passed':True,'successful_reader_raw_replay_passed':True,'compat_raw_replay_passed':True,'compat_status':vs['compat']['status'],'compat_label':label,
            'strict_error':vs['judge']['error'],'request_hash':key,'request_intent':ptr(snap,ip),'raw':rawref,
            'physical_attempt_count':len(physical),'attempt_numbers':[e['attempt'] for e in physical],'physical_statuses':[e.get('status') for e in physical],
            'request_context':request['context'],'stage_event_context':event['context'],'shared_exact_signature_physical_owner':request['context']!=event['context'],
            'stage_intents':{s:ptr(snap,paths[s].with_suffix('.intent.json')) for s in ('reader','judge')}})
    seal=None
    if is_c:
        p=run/'firstpass_seal.json'
        if p.exists():
            value=verify(snap,p)
            assert value['schema']==driver.SCHEMA+'-firstpass-seal' and value['protocol_sha256']==m['protocol_sha256'] and value['total']==1540
            assert value['manifest']==ptr(snap,mp) and value['outcome']==ptr(snap,op) and value['source_jobs']==m['source_manifest'] and value['stage_inventory']==out['results']
            seal=ptr(snap,p)
        else:blockers.append('C1 auto firstpass seal not yet published; no seal helper invoked by this audit')
    changed=snap.unchanged();assert not changed,changed[:3]
    cohort={arm:{q['query_id']:{k:q[k] for k in ('conversation','question','reference','category','reference_date')} for q in m['jobs'] if q['method']==arm} for arm in {j['method'] for j in m['jobs']}}
    return {'run_dir':str(run),'schema':m['schema'],'phase_kind':'firstpass','phase_id':m['phase_id'],'protocol_sha256':m['protocol_sha256'],
        'manifest':ptr(snap,mp),'outcome':ptr(snap,op),'source_jobs':m['source_manifest'],'firstpass_seal':seal,'terminal':True,'finished_utc':out['finished_utc'],'total':total,
        'status_counts_by_arm':dict(snapshot_counts),'candidate_counts_by_arm':dict(candidate_counts),'candidates':items,'candidate_count':len(items),
        'candidate_ids_sha256':jobhash(failed_keys),'candidate_jobs_sha256':digest(candidates),'strict_valid_no_excluded':len(strict_valid_no),'strict_valid_no_ids_sha256':jobhash(strict_valid_no),
        'blockers':blockers,'source_file_hashes':{str(p):h for p,h in snap.files.items()},'ledger_refs':ledgerrefs,'cohort_by_arm':cohort}

def main():
    out=ROOT/'research/planned_execution_2026-10-03/five_arm_pre_recovery_h_c1_audit.json'
    assert not out.exists()
    with forbid_network() as guards:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(one,ROOT/'runs/planned_hybrid_qa_v1',hdriver,False),pool.submit(one,ROOT/'runs/boundary_multiscale_qa_full1540_v2_20261003',cdriver,True)]
            reports=[f.result() for f in futures]
        assert not any(g.call_count for g in guards)
    canonical=None
    for report in reports:
        cohorts=report.pop('cohort_by_arm')
        for arm,c in cohorts.items():
            assert len(c)==1540
            if canonical is None:canonical=c
            else:assert c==canonical,arm
    hsummary=ROOT/'research/planned_execution_2026-10-03/hybrid_firstpass_summary.json';summary=json.loads(hsummary.read_text(encoding='utf-8'))
    assert summary['audit_status']=='no_invalid_artifacts' and not summary['audit_errors']
    value={'schema':'five-arm-partial-pre-recovery-independent-audit-v1','created_utc':now(),'status':'H_and_C1_known_failure_checked_C2_pending','overall_recovery_allowed':False,
        'runs':reports,'reviewed_total_firstpass_slots':6160,'reviewed_arms':list(reports[0]['status_counts_by_arm'])+list(reports[1]['status_counts_by_arm']),
        'candidate_total':sum(r['candidate_count'] for r in reports),'cohort_sha256':digest(canonical),
        'prior_verified_H_summary':{'path':str(hsummary),'sha256':hashlib.sha256(hsummary.read_bytes()).hexdigest(),'audit_status':summary['audit_status'],
            'use':'Existing full H raw audit provenance only; current candidates and their successful Readers replayed independently; not claiming all prior input hashes re-read here.'},
        'blockers':['C2 boundary_neighbor firstpass outcome/seal/raw audit not yet reviewed; five-arm joint permission forbidden']+[b for r in reports for b in r['blockers']],
        'policy_notes':['Frozen pure recovery_jobs/known_failure functions invoked without recovery preparation or API construction.',
            'Every strict Judge failure is a candidate regardless of compatible YES/NO; valid strict NO excluded. Successful Reader must be inherited unchanged, with zero new Reader request.',
            'This partial audit is not a recovery execution/preparation, full C1 metric summary, source truth certification or joint five-arm record.'],
        'auditor':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'model_calls':0,'source_mutations':0,'recovery_namespaces_created':0,'seal_helpers_invoked':0}
    save_new(out,value);print(json.dumps({'output':str(out),'candidate_counts':[r['candidate_count'] for r in reports],'blockers':value['blockers'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
