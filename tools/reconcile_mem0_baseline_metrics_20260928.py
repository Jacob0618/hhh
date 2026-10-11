"""Read-only reconstruction of old sdk20/rank and new session_rank scores.

Writes only new research artifacts; no clients, APIs, model execution or run edits.
"""
from pathlib import Path
import json, hashlib, statistics, sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries
from jev_mem.evaluation import lexical_metrics
from tools.summarize_pilot400 import verified_official_metric

METRICS = ('official_locomo_f1', 'judge_correct', 'custom_token_f1')

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def means(rows, ids):
    ids = sorted(ids)
    return {'n': len(ids), **{k: 100 * sum(rows[i][k] for i in ids) / len(ids) if ids else None for k in METRICS},
            'judge_correct_count': sum(rows[i]['judge_correct'] for i in ids),
            'answer_words_mean': statistics.mean(rows[i]['answer_words'] for i in ids) if ids else None,
            'answer_words_median': statistics.median(rows[i]['answer_words'] for i in ids) if ids else None}

def main():
    old_path = ROOT/'research/heartbeat_checks/20260927T2241/recovery_final/summary.json'
    new_path = ROOT/'research/mem0_sparse1540_final_summary_2026-09-28/summary.json'
    oldsum, newsum = read(old_path), read(new_path)
    official, provenance = verified_official_metric()
    assert provenance == newsum['official_metric_provenance']
    questions = {q['id']: {**q, 'conversation': ci} for ci,item in enumerate(load_data()) for q in queries(item, ci)}
    assert len(questions) == 1540
    source_hashes = {str(p):sha(p) for p in (old_path,new_path)}
    rows, raw, count, failures = {}, {}, {}, {}
    for method in ('sdk20','rank'):
        rows[method], raw[method], failures[method] = {}, {}, []
        counts = Counter()
        expected = {r['query_id']:r for r in oldsum['runs']['mem0']['per_query_metrics_after_recovery'][method]}
        for qid,q in questions.items():
            p = ROOT/f'runs/session_top20_v1/conv_{q["conversation"]}/qa/{method}/{qid}.json'
            overlay = ROOT/f'runs/technical_recovery_v1/execute/consumers/mem0/conv_{q["conversation"]}/{method}/{qid}.json'
            if p.exists():
                assert not overlay.exists(), f'duplicate firstpass and overlay {qid}'
                value=read(p); counts['firstpass_complete']+=1
            elif overlay.exists():
                p=overlay; value=read(p)
                if value['status']!='complete':
                    assert qid not in expected
                    failures[method].append({'query_id':qid,'conversation':q['conversation'],'category':q['category'],
                                             'status':value['status'],'error_type':value.get('error_type'),'error':value.get('error'),
                                             'path':str(p),'sha256':sha(p)})
                    counts['overlay_'+value['status']]+=1;continue
                counts['overlay_complete']+=1
            else: raise ValueError(f'unaccounted {method}:{qid}')
            assert value['question']==q['question'] and value['reference']==q['reference']
            assert value['query_id']==qid and value['category']==q['category']
            answer=value['prediction']['answer']; lexical=lexical_metrics(answer,q['reference'])
            r={'query_id':qid,'conversation':q['conversation'],'category':q['category'],
               'official_locomo_f1':official(answer,q['reference'],q['category']),
               'judge_correct':int(value['judgment']['correct']), 'custom_token_f1':lexical['token_f1'],
               'answer_words':len(answer.split())}
            ref=expected[qid]
            for k in METRICS:
                kold='strict_correct' if k=='judge_correct' else k
                assert abs(r[k]-ref[kold])<1e-12,(method,qid,k)
            assert abs(lexical['token_f1']-value['token_f1'])<1e-12
            source_hashes[str(p)]=sha(p); rows[method][qid]=r;raw[method][qid]=value
        assert set(rows[method])==set(expected)
        count[method]=dict(counts)
    rows['new'],raw['new']={},{}
    expected={x['query_id']:x['scores'] for x in newsum['outcomes'] if x['arm']=='session_rank'}
    for qid,q in questions.items():
        p=ROOT/f'runs/mem0_sparse1540_index_recovery_v2/conv_{q["conversation"]}/qa/session_rank/{qid}.json'
        outer=read(p);assert outer['status']=='complete'
        value=outer['result'];answer=value['prediction']['answer'];lexical=lexical_metrics(answer,q['reference'])
        r={'query_id':qid,'conversation':q['conversation'],'category':q['category'],
           'official_locomo_f1':official(answer,q['reference'],q['category']),
           'judge_correct':int(value['judgment']['correct']),'custom_token_f1':lexical['token_f1'],
           'answer_words':len(answer.split())}
        for k in METRICS: assert abs(r[k]-expected[qid][k])<1e-12
        assert abs(lexical['token_f1']-value['token_f1'])<1e-12
        source_hashes[str(p)]=sha(p); rows['new'][qid]=r;raw['new'][qid]={**value,'selected_ids':outer['selected_ids']}
    common=set(rows['sdk20'])&set(rows['rank'])&set(rows['new'])
    groups=[('old_sdk20_own','sdk20',set(rows['sdk20'])),('old_sdk20_common','sdk20',common),
            ('old_rank_common','rank',common),('new_session_rank_common','new',common),('new_session_rank_all','new',set(rows['new']))]
    chain=[{'name':name,**means(rows[method],ids)} for name,method,ids in groups]
    steps=[{'from':a['name'],'to':b['name'],'delta_pp':{k:b[k]-a[k] for k in METRICS}} for a,b in zip(chain,chain[1:])]
    alignments={}
    for method in ('sdk20','rank'):
        ids=set(rows[method])
        same_order={qid for qid in ids if raw[method][qid]['selected_ids']==raw['new'][qid]['selected_ids']}
        same_set={qid for qid in ids if set(raw[method][qid]['selected_ids'])==set(raw['new'][qid]['selected_ids'])}
        same_answer={qid for qid in ids if raw[method][qid]['prediction']['answer']==raw['new'][qid]['prediction']['answer']}
        subsets={'all_old_valid':ids,'same_selected_order':same_order,'same_selected_set':same_set,
                 'different_selected_set':ids-same_set,'same_answer_exact':same_answer,'different_answer_exact':ids-same_answer}
        alignments[method]={}
        for label,sub in subsets.items():
            a,b=means(rows[method],sub),means(rows['new'],sub)
            alignments[method][label]={'old':a,'new':b,'delta_pp':{k:b[k]-a[k] if sub else None for k in METRICS},
                'judge_transition':dict(Counter(f"{rows[method][i]['judge_correct']}->{rows['new'][i]['judge_correct']}" for i in sub))}
    breakdown={}
    for field,vals in [('conversation',range(10)),('category',range(1,5))]:
        breakdown[field]=[]
        for val in vals:
            sub={qid for qid in common if questions[qid][field]==val}
            breakdown[field].append({field:val,'common_n':len(sub),
                **{label:means(rows[label],sub) for label in ('sdk20','rank','new')},
                'new_minus_old_rank_pp':{k:means(rows['new'],sub)[k]-means(rows['rank'],sub)[k] for k in METRICS}})
    result={'schema':'mem0-baseline-reconciliation-metrics-v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'model_calls':0,'source_dataset_sha256':sha(ROOT/'data/locomo10.json'),'official_metric_provenance':provenance,
      'raw_scores_recomputed_and_match_both_published_summaries':True,'source_counts':count,'failures':failures,
      'own_success_means':{label:means(rs,set(rs)) for label,rs in rows.items()},'descriptive_chain':chain,
      'descriptive_chain_steps':steps,'common_query_ids':sorted(common),'alignments':alignments,'breakdown':breakdown,
      'per_query_metrics':rows,'source_file_sha256':source_hashes,
      'interpretation':'Descriptive decomposition, not causal identification. Metrics computed from existing actual answers, no missing imputation, no API or run edits. Selected ID identity does not by itself prove complete HTTP payload identity.'}
    target=ROOT/'research/baseline_reconciliation_2026-09-28_metrics.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Mem0 基线口径：逐题离线复算','',f'核算时间：{result["created_utc"]}。零 API；旧首遍、恢复 overlay、新运行全部保持不变。',
      '', '## 核验结论', '', '逐个读取旧 sdk20、旧 rank 和新 session_rank 的成功 QA 文件，使用同一 SHA256 锁定的官方 LoCoMo F1 适配器重新计分，全部逐题分数与原汇总一致；不是把 custom F1 混入官方 F1。',
      '', '| 组别 | 有效分母 | 官方 F1 (%) | Judge (%) | Judge 正确题数 |', '|---|---:|---:|---:|---:|']
    for name,r in result['own_success_means'].items():lines.append(f'| {name} | {r["n"]} | {r["official_locomo_f1"]:.6f} | {r["judge_correct"]:.6f} | {r["judge_correct_count"]} |')
    lines+=['','旧 sdk20 剩余 15 题、rank 剩余 14 题，均为 Reader 非字符串格式失败，未当作得零答案。新基线 1540 题全部有效。',
      '', '## 同题分解', '', '三方共同成功题为 1519。以下路径仅为描述性的算术分解，不是不同因素的因果效应。',
      '', '| 口径 | 分母 | 官方 F1 (%) | Judge (%) |', '|---|---:|---:|---:|']
    for r in chain:lines.append(f'| {r["name"]} | {r["n"]} | {r["official_locomo_f1"]:.6f} | {r["judge_correct"]:.6f} |')
    lines+=['','| 变化 | F1 差值（百分点） | Judge 差值（百分点） |','|---|---:|---:|']
    for s in steps:lines.append(f'| {s["from"]} → {s["to"]} | {s["delta_pp"]["official_locomo_f1"]:+.6f} | {s["delta_pp"]["judge_correct"]:+.6f} |')
    lines+=['','因此，缺失题分母与 sdk20→rank 方法变化不能解释主要差距；在相同 1519 题上，旧 rank 到新 session_rank 仍然有明显得分漂移。',
      '', '## 相同证据 ID 顺序的子集', '', '以下仅证明选中事实 ID 与顺序相同，不单独证明完整请求、日期、元数据、生成参数完全相同。',
      '', '| 旧方法 | 同顺序题数 | 旧 F1 | 新 F1 | F1 差值 | 旧 Judge | 新 Judge | Judge 差值 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in ('sdk20','rank'):
        d=alignments[m]['same_selected_order'];a,b=d['old'],d['new']
        lines.append(f'| {m} | {a["n"]} | {a["official_locomo_f1"]:.6f} | {b["official_locomo_f1"]:.6f} | {d["delta_pp"]["official_locomo_f1"]:+.6f} | {a["judge_correct"]:.6f} | {b["judge_correct"]:.6f} | {d["delta_pp"]["judge_correct"]:+.6f} |')
    d=alignments['rank']['same_answer_exact'];a,b=d['old'],d['new']
    lines+=['', '## 相同回答下的 Judge 漂移', '',
      f'旧 rank 与新 A 共有 {a["n"]} 题回答逐字符相同，官方 F1 当然完全相同；Judge 正确题数却从 {a["judge_correct_count"]} 变为 {b["judge_correct_count"]}。其中 23 题由错转对、3 题由对转错，净增 20。',
      '', '在旧 rank 的 1526 个有效题上，新 A 的 Judge 共净增 45 个正确判断；20 个来自相同回答的重新判分，另 25 个来自回答改变的题。这是算术分解，不等于证明 Judge schema 是唯一原因；两次真实请求配置及时间也须检查。',
      '', '共同 1519 题中，旧 rank 平均回答 15.943 词、中位 12 词，新 A 为 13.786 词、中位 9 词。更短回答与 F1 改变同时出现，不能仅凭该相关性认定内容正确性提高。']
    for field in ('conversation','category'):
        lines+=['',f'## 按 {field} 分组（三方共同题）','','| 分组 | n | 旧 sdk20 F1/Judge | 旧 rank F1/Judge | 新 A F1/Judge | 新−旧 rank F1/Judge（百分点） |','|---|---:|---:|---:|---:|---:|']
        for r in breakdown[field]:
            parts=[f'{r[m]["official_locomo_f1"]:.3f} / {r[m]["judge_correct"]:.3f}' for m in ('sdk20','rank','new')]
            delta=r['new_minus_old_rank_pp'];lines.append(f'| {r[field]} | {r["common_n"]} | '+ ' | '.join(parts)+f' | {delta["official_locomo_f1"]:+.3f} / {delta["judge_correct"]:+.3f} |')
    lines+=['','完整逐题分数、失败定位、全部读取文件 SHA256、相同答案与证据集合子集见同名 JSON。解释回答差异需要另外核查实际请求；本报告不把离线相关性解释成 schema 的独立因果效应。']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'files':[str(target),str(target.with_suffix('.md'))], 'chain':chain,'steps':steps,'alignments':alignments},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
