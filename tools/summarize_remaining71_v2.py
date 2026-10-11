"""Offline exact input/response audit and full400 overlay of the remaining71."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_pilot400_remaining71_v2 as driver
from tools import pilot400_recovery_qa_v1 as qa
from tools import recover_remaining71_transport_v3 as transport
from tools.audit_remaining71_ledger_v2 import audit_qa_ledger
from tools.pilot400_recovery_writer_v1 import read,sha,save_once
from tools.audit_pilot400_recorded_qa_v1 import verify_terminal_seal
from tools.summarize_pilot400 import verified_official_metric,paired_comparison,METRICS
from tools.summarize_pilot400_v2 import ledger_usage
from jev_mem.evaluation import lexical_metrics
from jev_mem.pilot400 import ARMS,fingerprint,load_cohort
from jev_mem.data import load_data
OUT=driver.DEFAULT_OUTPUT
OLD=driver.ORIGINAL_RUN
PREV=driver.PREVIOUS_RECOVERY

class Replay:
    def __init__(self,events):self.events=list(events)
    def generate(self,prompt,*,model,system,json_output,max_output,stage):
        e=self.events.pop(0)
        legacy=e['source']=='original_valid_reader_cache'
        schema=None if legacy else (qa.READER_SCHEMA if stage==qa.evaluation.READER_STAGE else qa.JUDGE_SCHEMA)
        sig=qa.generation_signature(prompt,model=model,system=system,json_output=json_output,max_output=max_output,schema=schema)
        if e['stage']!=stage or e['request_hash']!=qa.request_hash(sig):raise ValueError('Raw request replay mismatch')
        p=Path(e['path'] if legacy else e['cache_path'])
        if sha(p)!=(e['file_sha256'] if legacy else e['cache_sha256']):raise ValueError('Raw response file changed')
        cache=read(p)
        if cache['request']!=sig or fingerprint(cache['response'])!=e['response_sha256']:raise ValueError('Response signature changed')
        if not legacy:
            intent=read(p.parent.parent/'qa_request_intents'/(e['request_hash']+'.json'))
            if intent['request']!=sig:raise ValueError('Request intent changed')
        return qa.decode_response(cache['response'])

def consumers_for(q,events):
    rows=[]
    for event in events:
        consumer={'conversation':q['conversation'],'query_id':q['id'],'method':driver.ARM,'event':event}
        if event['source'] in ('new_response','recovery_cache'):
            cache=read(event['cache_path']);response=cache['response']
            consumer.update(response_usage=response.get('usageMetadata',response.get('usage',{})),
                            resolved_model=response.get('modelVersion',response.get('model')))
        rows.append(consumer)
    return rows

def main():
    status=read(OUT/'qa_remaining71_status.json')
    if (status['status'],status['completed'],status['failed'],status['done'])!=('requires_review',70,1,71):
        raise ValueError('Reviewed v2 terminal population changed')
    transport_status=read(transport.OUT/'status.json')
    if transport_status['status']!='complete' or transport_status['completed']!=1:
        raise ValueError('Separate transport recovery not complete')
    _,_,transport_plan=transport.prepare()
    if read(transport.OUT/'plan.json')!=transport_plan:
        raise ValueError('Transport recovery inputs or policy changed')
    v2_seal_path=ROOT/'research/pilot400_remaining71_v2_terminal_seal.json'
    v2_seal=read(v2_seal_path)
    if Path(v2_seal['run_dir']).resolve()!=OUT.resolve():raise ValueError('Wrong v2 seal run')
    for item in v2_seal['files']:
        if sha(OUT/item['path'])!=item['sha256']:raise ValueError('Sealed v2 file changed')
    old_files=verify_terminal_seal(OLD,read(qa.TERMINAL_SEAL))
    for item in read(PREV/'recovery_terminal.json')['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Previous recovery changed')
    records=driver.collect(OUT)
    result_map={};new_result_map={};validated=[];fallback_slots=0;selected_chars=[];qa_consumers=[];transport_consumers=[]
    for r in records:
        retrieval_path,path=driver._paths(OUT,r)
        retrieval=qa._verify_sealed(read(retrieval_path))
        if retrieval!=driver.retrieval_for(r):raise ValueError('New stable retrieval replay mismatch')
        row=qa._verify_sealed(read(path));selected=retrieval['selected'];q=r['q']
        qa_consumers.extend(consumers_for(q,row['api_events']))
        if row['status']=='failed':
            if q['id']!='c1_q43':raise ValueError('Unreviewed failure in combined overlay')
            path=transport.OUT/'conversations/conv_1/qa'/driver.ARM/'c1_q43.json'
            row=qa._verify_sealed(read(path))
            if (sha(path)!=transport_status['qa_sha256'] or
                    row['technical_recovery_plan_sha256']!=sha(transport.OUT/'plan.json')):
                raise ValueError('Transport recovery artifact changed')
            transport_consumers.extend(consumers_for(q,row['api_events']))
        expected_binding={**r['binding'],'retrieval_sha256':sha(retrieval_path),'selected_sha256':fingerprint(selected)}
        if row['binding']!=expected_binding or row['selected_ids']!=[f['id'] for f in selected]:raise ValueError('QA binding mismatch')
        api=Replay(row['api_events'])
        result=qa.evaluation.answer_and_judge(api,q['question'],q['reference'],selected,reference_date=r['binding']['reference_date'])
        if row['status']!='complete' or result!=row['result'] or api.events:raise ValueError('QA raw result replay mismatch')
        result_map[(driver.ARM,q['id'])]=row['result'];new_result_map[q['id']]=row['result']
        fallback_slots+=retrieval['trace']['raw_member_excerpt_selected']
        selected_chars.append(result['evidence_characters'])
        validated.append({'query_id':q['id'],'conversation':q['conversation'],'qa_sha256':sha(path),
            'retrieval_sha256':sha(retrieval_path),'selected_count':len(selected),'new_response_events':len(row['api_events'])})
    if len(list(OUT.glob('conversations/conv_*/qa/*/*.json')))!=71:raise ValueError('Unexpected new QA population')
    for r in qa.collect_existing_failures():
        q=r['q'];path=PREV/'conversations'/('conv_'+str(q['conversation']))/'qa'/r['arm']/(q['id']+'.json')
        row=qa._verify_sealed(read(path));api=Replay(row['api_events'])
        expected=qa.evaluation.answer_and_judge(api,q['question'],q['reference'],r['selected'],reference_date=r['reference_date'])
        if row['result']!=expected or api.events:raise ValueError('Previous recovery replay changed')
        result_map[(r['arm'],q['id'])]=row['result']
    if len(new_result_map)!=71 or len(result_map)!=82:
        raise ValueError('Expected exactly71 new and11 previous recovered outcomes')
    _,questions=load_cohort(ROOT/'configs/pilot400_manifest_v1.json',load_data())
    official,provenance=verified_official_metric();metrics={a:{} for a in ARMS};new_metrics=[];old_successes=0
    for q in questions:
        for arm in ARMS:
            original=read(OLD/'conversations'/('conv_'+str(q['conversation']))/'qa'/arm/(q['id']+'.json'))
            key=(arm,q['id'])
            if original['status']=='complete':
                old_successes+=1
                if key in result_map:raise ValueError('Existing successful QA must not be replaced')
                result=original['result']
            else:result=result_map[key]
            scored={'query_id':q['id'],'conversation':q['conversation'],'category':q['category'],
                'official_locomo_f1':official(result['answer'],q['reference'],q['category']),
                'custom_token_f1':lexical_metrics(result['answer'],q['reference'])['token_f1'],
                'judge_correct':float(result['judge']['correct']),'judge_completeness':result['judge']['completeness']}
            if scored['custom_token_f1']!=result['token_f1']:raise ValueError('Stored custom F1 differs')
            metrics[arm][q['id']]=scored
            if arm==driver.ARM and q['id'] in new_result_map:new_metrics.append(scored)
    if old_successes!=1118 or len(new_metrics)!=71 or any(len(rows)!=400 for rows in metrics.values()):
        raise ValueError('Full400 overlay population differs')
    def aggregate(rows):
        return {'completed':len(rows),'fixed_denominator':len(rows),'failed':0,
            'judge_correct_count':sum(r['judge_correct'] for r in rows),
            'means':{m:sum(r[m] for r in rows)/len(rows) for m in METRICS}}
    ledger=[json.loads(s) for s in (OUT/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    physical=[r for r in ledger if not r.get('cache_hit')]
    qa_physical=[r for r in physical if r['stage'] in (qa.evaluation.READER_STAGE,qa.evaluation.JUDGE_STAGE)]
    if any(r['provider']=='jev' for r in ledger):raise ValueError('Unexpected JEV traffic')
    hashes=Counter(r['request_hash'] for r in qa_physical)
    if any(n!=1 for n in hashes.values()):raise ValueError('Unexpected repeated attempt within original v2 run')
    qa_failed=[r for r in qa_physical if r['status']!=200]
    if (len(qa_failed)!=1 or qa_failed[0]['request_hash']!=transport_plan['original_request_hash']
            or qa_failed[0]['status']!='transport_error' or not qa_failed[0].get('usage_unknown')):
        raise ValueError('Reviewed original v2 transport failure changed')
    transport_ledger=[json.loads(s) for s in (transport.OUT/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    transport_physical=[r for r in transport_ledger if not r.get('cache_hit')]
    if len(transport_physical)!=2 or any(r['status']!=200 or r['provider']!='gemini' for r in transport_physical):
        raise ValueError('Expected exact separate Reader/Judge transport recovery')
    consumer_audits={'v2':audit_qa_ledger(qa_consumers,ledger),
        'transport_v3':audit_qa_ledger(transport_consumers,transport_ledger,expected_queries=[(1,'c1_q43')])}
    bank_stats={};admission=read(OUT/'bank_admission.json')
    for ci in (1,7):
        generated_path=OUT/'banks'/('conv_'+str(ci))/'bank_overlay.json'
        bank_path=OUT/admission['banks'][str(ci)]['bank_path']
        generated=read(generated_path);bank=read(bank_path)
        rejected={entry['fact']['id'] for entry in bank['source_review_quarantine']}
        if (bank['facts']!=[f for f in generated['facts'] if f['id'] not in rejected]
                or bank['groups']!=generated['groups']):
            raise ValueError('Source-reviewed bank changed beyond explicit exclusion')
        active_ids={f['id'] for f in bank['facts']}
        fresh=[g['record'] for g in bank['groups'] if g['kind']=='members-only-quoted-fact-isolation-v2']
        bank_stats[str(ci)]={'total_groups':len(bank['groups']),'facts_or_memory_items':len(bank['facts']),
            'generated_fact_count_before_assistant_review':len(generated['facts']),
            'new_groups':len(fresh),'new_modes':dict(Counter(g['mode'] for g in fresh)),
            'new_model_admitted_before_assistant_review':sum(len(g['facts']) for g in fresh),
            'new_admitted_items':sum(f['id'] in active_ids for g in fresh for f in g['facts']),
            'program_quarantined_candidates':sum(len(g['quarantined']) for g in fresh),
            'assistant_source_review_excluded':len(rejected),
            'raw_fallback_items':sum(f.get('memory_kind')=='raw_member_excerpt' for f in bank['facts']),
            'bank_sha256':sha(bank_path),'generated_bank_sha256':sha(generated_path)}
    report={'schema':'remaining71-full400-overlay-summary-v2','original_completed_frozen':1118,
        'previous_recovery_completed_frozen':11,'newly_completed':71,'original_terminal_files_unchanged':len(old_files),
        'arms':{a:aggregate(list(rows.values())) for a,rows in metrics.items()},'new71_only':aggregate(new_metrics),
        'paired_differences':{a:paired_comparison(metrics[ARMS[0]],metrics[a],samples=2000,seed=0) for a in ARMS[1:]},
        'per_question_metrics':metrics,'bank_statistics':bank_stats,'qa_audit':validated,
        'raw_fallback_selected_slots':fallback_slots,'mean_new_evidence_characters':sum(selected_chars)/len(selected_chars),
        'qa_physical_attempts':len(qa_physical)+len(transport_physical),
        'v2_qa_physical_attempts':len(qa_physical),'transport_v3_qa_physical_attempts':len(transport_physical),
        'v2_completed':70,'v2_transport_failed_unknown':1,'transport_v3_completed':1,
        'qa_stages':dict(Counter(r['stage'] for r in qa_physical+transport_physical)),
        'qa_consumer_ledger_audits':consumer_audits,
        'usage':ledger_usage(OUT/'api_calls.jsonl'),'official_metric_provenance':provenance,
        'transport_v3_usage':ledger_usage(transport.OUT/'api_calls.jsonl'),
        'v2_terminal_seal_sha256':sha(v2_seal_path),
        'bank_admission_sha256':sha(OUT/'bank_admission.json'),
        'assistant_source_review_sha256':admission['review_sha256'],
        'assistant_source_review_scope':'All113 new generated facts only;106 retained,7 excluded. Inherited641 facts unchanged. Not human gold.',
        'original_retrieval_replay_gate_passed':False,'new71_stable_retrieval_exact_replay_passed':True,
        'limitations':['Post-hoc members-only/source-validated fault-isolation writing recovery for two C conversations.',
            'Seven of113 new claims were excluded by independent assistant source-only review, in addition to the automatic writer check; this is not a fully automatic unchanged C pipeline.',
            'One Reader TLS failure was recovered by an explicit separate identical-payload attempt; the original failure and unknown server usage remain recorded.',
            'Original and prior results are preserved; new full400 overlay is not an untouched original C replication.',
            'No new QA/gold in writing; source annotations are still proposed rather than retrieval gold.',
            'Only ten conversation clusters; development conversations already inspected.']}
    dest=ROOT/'research/pilot400_remaining71_results_v2_2026-09-28.json';save_once(dest,report)
    lines=['# 剩余71题恢复结果','','固定71题全部恢复；三组现均400/400。原始及第一次恢复产物保持不变。',
        '本400题模块实验采用独立GroupFactWriter架构，不是Mem0原生流程。','',
        '| 组 | 完成 | 官方LoCoMo F1 | Custom F1 | Judge判对数/400 | Judge判对率 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,item in report['arms'].items():
        m=item['means'];lines.append(f"| {arm} | 400/400 | {m['official_locomo_f1']*100:.3f}% | {m['custom_token_f1']*100:.3f}% | {int(item['judge_correct_count'])}/400 | {m['judge_correct']*100:.3f}% |")
    lines += ['', '配对差值使用同一400题，区间按10个conversation整群重采样2000次；是探索性估计。', '',
        '| 相对A | 官方F1差值（百分点，95%区间） | Judge差值（百分点，95%区间） |',
        '|---|---:|---:|']
    for arm,paired in report['paired_differences'].items():
        cells=[]
        for key in ('official_locomo_f1','judge_correct'):
            metric=paired['metrics'][key];ci=metric['cluster_bootstrap_95_percentile_interval']
            if paired['common_success_count']!=400 or ci is None:
                raise ValueError('Full400 paired interval is not available')
            cells.append(f"{metric['other_minus_baseline']*100:+.3f} [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]")
        lines.append(f"| {arm} − A | {cells[0]} | {cells[1]} |")
    new=report['new71_only'];lines += ['',f"新恢复71题中Judge判对{int(new['judge_correct_count'])}题。原C的189/329与新结果使用不同分母，不能把均值变化当作模块提升。",'',
        '两个库共复用63个有效组，14个新组采用完整成员组抽取、逐字引文与逐事实支持判断。新生成113条事实经独立助手原文审查后隔离7条，保留106条；旧641条事实保持不变，最终两个活跃库分别291、456条。此审查不接触QA/gold，也不等同于人工真值；自动检查仍可能放过来源不足的内容。',
        f"实际原文后备进入QA的槽位数：{fallback_slots}。各库新模式、隔离项和长度见JSON；若使用原文项，它不等同于一条原子事实。",'',
        '这是有明确写入容错改动的事后恢复结果，原C实验的失败仍有效且保留。不能当作纯原协议复现。',
        f"全部71题的新检索与Reader/Judge原始响应已离线精确重放。首遍70成功、1次Reader TLS失败；该题另行原样请求一次后成功。QA共{len(qa_physical)+len(transport_physical)}次物理尝试，原TLS的服务端用量未知，无新增JEV调用。",'',
        '调用费用仅为配置单价已知部分估算；embedding等未知费用单列，完整账单未知。具体调用/缓存/源审查与配对区间见同名JSON。']
    md=dest.with_suffix('.md')
    if md.exists():raise ValueError('Do not overwrite final findings')
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('arms','new71_only','bank_statistics','qa_physical_attempts','raw_fallback_selected_slots')},indent=2))

if __name__=='__main__':main()
