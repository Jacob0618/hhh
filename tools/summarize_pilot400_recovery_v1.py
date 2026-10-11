"""Offline raw-response audit and separate completed-only/fixed400 recovery table."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import pilot400_recovery_qa_v1 as qa
from tools.pilot400_recovery_writer_v1 import read,sha,save_once,DEFAULT,OLD
from tools.audit_pilot400_recorded_qa_v1 import verify_terminal_seal
from tools.summarize_pilot400 import verified_official_metric,paired_comparison,METRICS
from tools.summarize_pilot400_v2 import ledger_usage
from jev_mem.evaluation import lexical_metrics
from jev_mem.pilot400 import ARMS,fingerprint,load_cohort
from jev_mem.data import load_data

class OfflineReplay:
    def __init__(self,events):self.events=list(events)
    def generate(self,prompt,*,model,system,json_output,max_output,stage):
        event=self.events.pop(0)
        schema=None if event['source']=='original_valid_reader_cache' else (
            qa.READER_SCHEMA if stage==qa.evaluation.READER_STAGE else qa.JUDGE_SCHEMA)
        signature=qa.generation_signature(prompt,model=model,system=system,
            json_output=json_output,max_output=max_output,schema=schema)
        if qa.request_hash(signature)!=event['request_hash'] or event['stage']!=stage:
            raise ValueError('Recovery request/event mismatch')
        path=Path(event['path'] if schema is None else event['cache_path'])
        expected_sha=event['file_sha256'] if schema is None else event['cache_sha256']
        if sha(path)!=expected_sha:raise ValueError('Recovery cache bytes changed')
        cached=read(path)
        if cached['request']!=signature or fingerprint(cached['response'])!=event['response_sha256']:
            raise ValueError('Exact request/raw response binding mismatch')
        if schema is not None:
            intent=read(DEFAULT/'qa_request_intents'/(event['request_hash']+'.json'))
            if intent['request']!=signature:raise ValueError('Intent mismatch')
        return qa.decode_response(cached['response'])

def main():
    review_path=ROOT/'research/pilot400_recovery_semantic_review_2026-09-28.json'
    review=read(review_path)
    if review['bank_admission']!={'1':False,'7':False} or review['qa_authorized'] is not False:
        raise ValueError('This summary admits only the actual rejected-bank recovery disposition')
    for ci in (1,7):
        if read(DEFAULT/'banks'/('conv_'+str(ci))/'status.json')['status']=='running':
            raise ValueError('Writer still active; no final usage summary yet')
    if read(DEFAULT/'qa_recovery_status.json')['status']!='complete':
        raise ValueError('QA recovery not complete')
    sealed=verify_terminal_seal(OLD,read(qa.TERMINAL_SEAL))
    admitted=qa.collect_existing_failures()
    overlays={}
    for item in admitted:
        q,arm=item['q'],item['arm']
        path=DEFAULT/'conversations'/('conv_'+str(q['conversation']))/'qa'/arm/(q['id']+'.json')
        row=qa._verify_sealed(read(path))
        if row['binding']!=item['binding'] or row['selected_ids']!=[f['id'] for f in item['selected']]:
            raise ValueError('Recovery original selection/binding differs')
        api=OfflineReplay(row['api_events'])
        expected=qa.evaluation.answer_and_judge(api,q['question'],q['reference'],item['selected'],reference_date=item['reference_date'])
        if row['status']!='complete' or row['result']!=expected or api.events:
            raise ValueError('Recovery result is not exact raw-response replay')
        overlays[(arm,q['id'])]=row
    all_new=list(DEFAULT.glob('conversations/conv_*/qa/*/*.json'))
    if len(all_new)!=11 or len(overlays)!=11:
        raise ValueError('Unexpected additional QA results')
    _,questions=load_cohort(ROOT/'configs/pilot400_manifest_v1.json',load_data())
    official,metric_provenance=verified_official_metric()
    def score(q,r):
        return {'query_id':q['id'],'conversation':q['conversation'],'category':q['category'],
            'official_locomo_f1':official(r['answer'],q['reference'],q['category']),
            'custom_token_f1':lexical_metrics(r['answer'],q['reference'])['token_f1'],
            'judge_correct':float(r['judge']['correct']),'judge_completeness':r['judge']['completeness']}
    old_scores={a:{} for a in ARMS}; new_scores={a:{} for a in ARMS}
    for q in questions:
        for arm in ARMS:
            p=OLD/'conversations'/('conv_'+str(q['conversation']))/'qa'/arm/(q['id']+'.json')
            row=read(p)
            if row['status']=='complete':
                old_scores[arm][q['id']]=score(q,row['result'])
                new_scores[arm][q['id']]=old_scores[arm][q['id']]
            elif (arm,q['id']) in overlays:
                new_scores[arm][q['id']]=score(q,overlays[(arm,q['id'])]['result'])
    def aggregate(scores):
        return {arm:{'completed':len(rows),'failed':400-len(rows),'fixed_denominator':400,
            'judge_correct_count':sum(r['judge_correct'] for r in rows.values()),
            'completed_only':{m:sum(r[m] for r in rows.values())/len(rows) for m in METRICS},
            'missing_as_zero_sensitivity':{m:sum(r[m] for r in rows.values())/400 for m in METRICS}}
            for arm,rows in scores.items()}
    ledger=[json.loads(line) for line in (DEFAULT/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    http=[r for r in ledger if not r.get('cache_hit')]
    qa_http=[r for r in http if r['stage'] in (qa.evaluation.READER_STAGE,qa.evaluation.JUDGE_STAGE)]
    if len(qa_http)!=19 or len({r['request_hash'] for r in qa_http})!=19 or any(r['status']!=200 for r in qa_http):
        raise ValueError('Unexpected QA physical-attempt accounting')
    report={'schema':'pilot400-recovery-summary-v1','original_run':str(OLD),'recovery_run':str(DEFAULT),
        'old_terminal_files_unchanged':len(sealed),'original':aggregate(old_scores),'after_recovery':aggregate(new_scores),
        'recovered_slots':11,'recovered_C_slots':3,'remaining_failed_C_slots':71,
        'missing_is_not_observed_zero':True,'original_results_overwritten':False,
        'bank_admission':review['bank_admission'],'semantic_review':{'path':str(review_path),'sha256':sha(review_path)},
        'official_metric_provenance':metric_provenance,
        'paired_after_recovery':{arm:paired_comparison(new_scores[ARMS[0]],new_scores[arm],samples=2000,seed=0) for arm in ARMS[1:]},
        'qa_physical_attempts':len(qa_http),'qa_stages':dict(Counter(r['stage'] for r in qa_http)),
        'usage':ledger_usage(DEFAULT/'api_calls.jsonl'),
        'scope':'Post-hoc technical recovery of all failures, not a new untouched benchmark. Bank-dependent71 remain unscored.',
        'strict_old_retrieval_replay_passed':False}
    destination=ROOT/'research/pilot400_recovery_results_2026-09-28.json'
    save_once(destination,report)
    lines=['# Fixed400：独立架构技术恢复结果','','原始试验及失败全部保留；本报告只追加用户新授权的技术恢复。',
        '400题A/B/C均为独立GroupFactWriter架构，不是native Mem0，也不是旧1540题使用的完全同一writer/reader。','',
        '| 组 | 原有效数 | 恢复后有效数 | 原Judge（完成分母） | 恢复后Judge（完成分母） | 恢复后Judge（固定400、失败计零） |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a,b=report['original'][arm],report['after_recovery'][arm]
        lines.append(f"| {arm} | {a['completed']} | {b['completed']} | {a['completed_only']['judge_correct']*100:.3f}% | {b['completed_only']['judge_correct']*100:.3f}% | {b['missing_as_zero_sensitivity']['judge_correct']*100:.3f}% |")
    lines += ['', 'C原57.055%=186/326；固定400下的失败计零值为46.500%。71个库依赖失败原来未调用Reader/Judge，另3个QA格式失败本次已补齐。',
        'A/B全部同类格式失败也按相同规则恢复。19次新QA物理请求（9Reader+10Judge），两次合法旧Reader复用；共享Judge不重复物理调用。',
        '', '两个库的额外提取均未通过最终原文支持检查：conv7仍为纯上下文事实；conv1虽然引用结构合法，但两条具体内容只有上下文支持。71题不使用部分库、不补造答案、不移出400分母。',
        '写入剩余组所产生的新记录保留为未获QA准入的恢复产物，不能声称该事实库已恢复成功。',
        '', '技术恢复仅检验输出/来源合同，模型和选题不变；既有完整结果1118个不重跑。原BM25跨进程确定性问题仍保留，恢复QA精确沿用记录的20事实和顺序。',
        '', '完整原始请求与费用未知项见同名JSON；费用为配置单价估算，不是供应商账单。']
    path=destination.with_suffix('.md')
    if path.exists():raise ValueError('Do not overwrite recovery findings')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'original':report['original'],'after_recovery':report['after_recovery'],'qa_http':len(qa_http),
        'known_cost_usd':report['usage']['known_cost_component_usd'],'unknown_cost_attempts':report['usage']['unknown_cost_attempts']},indent=2))

if __name__=='__main__':main()
