"""Offline selection-only audit: deliberately never reads QA artifacts or scores."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,statistics,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_sdk20_modules import verify,digest
from jev_mem.data import load_data,session_envelopes,SOURCE_HASH

RUN=ROOT/'runs/sdk20_modules_v1'
DEST=ROOT/'research/sdk20_modules_selection_audit_2026-09-28'
ARMS=('baseline_sdk20','fact_types_sdk20','preseg_sdk20')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def stats(values):
    v=sorted(values)
    return {'n':len(v),'mean':statistics.mean(v) if v else None,'median':statistics.median(v) if v else None,
            'min':min(v) if v else None,'max':max(v) if v else None,'histogram':dict(sorted(Counter(v).items()))}
def summary(rows):
    return {'questions':len(rows),'B_vs_A':{
        'identical_order':sum(r['B']['same_order'] for r in rows),
        'identical_set':sum(r['B']['same_set'] for r in rows),
        'same_set_but_reordered':sum(r['B']['same_set'] and not r['B']['same_order'] for r in rows),
        'changed_set':sum(not r['B']['same_set'] for r in rows),
        'replaced_facts_out_of_20':stats([r['B']['entered_count'] for r in rows]),
        'same_rank_positions_out_of_20':stats([r['B']['unchanged_positions'] for r in rows]),
        'candidate_count':stats([r['B']['candidate_count'] for r in rows]),
        'eligible_count':stats([r['B']['eligible_count'] for r in rows]),
        'exact_distinct_affinities':stats([r['B']['distinct_affinities'] for r in rows]),
        'queries_with_any_exact_affinity_tie':sum(r['B']['tied_candidate_count']>0 for r in rows),
        'candidates_in_tie_groups':stats([r['B']['tied_candidate_count'] for r in rows]),
        'zero_query_weight_count':sum(r['B']['zero_query_weights'] for r in rows),
        'fallback_reasons':dict(Counter(r['B']['fallback_reason'] or 'none' for r in rows))},
        'C_vs_A':{'exact_text_date_multiset_overlap_out_of_20':stats([r['C']['overlap_count'] for r in rows]),
          'mean_overlap_fraction':statistics.mean(r['C']['overlap_count']/20 for r in rows),
          'identical_pair_multiset':sum(r['C']['same_multiset'] for r in rows),
          'identical_pair_order':sum(r['C']['same_order'] for r in rows),
          'eligible_count':stats([r['C']['eligible_count'] for r in rows]),
          'queries_with_duplicate_text_date_pairs_A':sum(r['C']['duplicate_pairs_A']>0 for r in rows),
          'queries_with_duplicate_text_date_pairs_C':sum(r['C']['duplicate_pairs_C']>0 for r in rows)}}

def main():
    manifest=read(RUN/'run_manifest.json');protocol=manifest['protocol_sha256']
    assert digest({k:v for k,v in manifest.items() if k!='protocol_sha256'})==protocol
    assert manifest['dataset_sha256']==SOURCE_HASH and manifest['questions']==1540
    hashes={str(RUN/'run_manifest.json'):sha(RUN/'run_manifest.json')}
    rows=[];bank_rows=[];data=load_data()
    for ci in range(10):
        folder=RUN/f'conv_{ci}';sm=folder/'source_manifest.json'
        assert sha(sm)==manifest['source_manifests'][str(ci)]
        smdoc=read(sm);hashes[str(sm)]=sha(sm);banks={}
        for kind in ('session','preseg'):
            p=folder/kind/'bank.json';b=read(p)
            assert sha(p)==smdoc['banks'][kind]['bank']['sha256']
            assert b['native_state']=='original_unrepaired' and b['conversation']==ci
            banks[kind]=b;hashes[str(p)]=sha(p)
        sessions=list(session_envelopes(data[ci]));source_sessions=len(sessions)
        gs=banks['preseg']['group_source'];gp=Path(gs['path']);assert sha(gp)==gs['sha256']
        grouped=read(gp);hashes[str(gp)]=sha(gp)
        gpath=Path(grouped['grouping_path']);assert sha(gpath)==grouped['grouping_sha256']
        grouping=read(gpath);hashes[str(gpath)]=sha(gpath)
        groups=grouping['groups'];segments=grouping['segments']
        assert len(groups)==len(grouped['envelopes'])
        cross_date=sum(len(set(g['observation_dates']))>1 for g in groups)
        cross_session=sum(len(set(g['original_session_ids']))>1 for g in groups)
        merged=sum(len(g['segment_ids'])>1 for g in groups)
        assert all(g['cross_date']==(len(set(g['observation_dates']))>1) for g in groups)
        bank_rows.append({'conversation':ci,'A_B_facts':len(banks['session']['facts']),
            'C_facts':len(banks['preseg']['facts']),'original_sessions':source_sessions,
            'segments':len(segments),'groups':len(groups),'groups_merging_segments':merged,
            'cross_session_groups':cross_session,'cross_date_groups':cross_date,
            'unique_text_date_pairs_A':len({(f['text'],f['observation_date']) for f in banks['session']['facts']}),
            'unique_text_date_pairs_C':len({(f['text'],f['observation_date']) for f in banks['preseg']['facts']})})
        byid={k:{f['id']:f for f in b['facts']} for k,b in banks.items()}
        for qid in manifest['query_ids'][str(ci)]:
            sels={}
            for arm in ARMS:
                p=folder/'selection'/arm/(qid+'.json');r=verify(p);hashes[str(p)]=sha(p)
                assert r['protocol_sha256']==protocol and r['conversation']==ci and r['query_id']==qid
                assert r['method']==arm and r['status']=='complete'
                assert len(r['selected_ids'])==len(set(r['selected_ids']))==20
                kind='preseg' if arm==ARMS[2] else 'session'
                assert r['bank_sha256']==hashes[str(folder/kind/'bank.json')]
                assert set(r['selected_ids'])<=byid[kind].keys();sels[arm]=r
            a,b,c=[sels[arm] for arm in ARMS];idsa,idsb,idsc=[r['selected_ids'] for r in (a,b,c)]
            tr=b['trace'];rank=tr['ranking_in_native_order'];counts=Counter(r['type_affinity'] for r in rank if r['type_affinity']is not None)
            assert tr['base20_ids']==idsa and tr['selected_ids']==idsb
            assert len(rank)==tr['eligible_count'] and len(set(r['id'] for r in rank))==len(rank)
            assert set(tr['entered_ids'])==set(idsb)-set(idsa) and set(tr['dropped_ids'])==set(idsa)-set(idsb)
            querysum=sum(tr['query_type_scores'].values())
            if querysum==0:assert tr['fallback_reason']=='zero_query_weights'
            elif len(counts)<=1:assert tr['fallback_reason']=='equal_type_affinities'
            else:assert tr['fallback_reason']is None
            for kind,sel in [('session',b),('preseg',c)]:
                rp=folder/'retrieval'/kind/(qid+'.json');ret=verify(rp);hashes[str(rp)]=sha(rp)
                assert sel['retrieval_sha256']==sha(rp) and ret['protocol_sha256']==protocol
                assert ret['capture']['capture_sha256']==digest({k:v for k,v in ret['capture'].items() if k!='capture_sha256'})
                if kind=='session':assert ret['capture']['eligible_ids']==[r['id'] for r in rank]
                else:
                    assert ret['selected_ids']==idsc
                    c_eligible=len(ret['capture']['eligible'])
            pairsa=[(byid['session'][i]['text'],byid['session'][i]['observation_date']) for i in idsa]
            pairsc=[(byid['preseg'][i]['text'],byid['preseg'][i]['observation_date']) for i in idsc]
            pa,pc=Counter(pairsa),Counter(pairsc);overlap=sum((pa&pc).values())
            rows.append({'conversation':ci,'query_id':qid,'category':a['category'],'B':{
                'same_order':idsa==idsb,'same_set':set(idsa)==set(idsb),'entered_count':len(set(idsb)-set(idsa)),
                'unchanged_positions':sum(x==y for x,y in zip(idsa,idsb)),
                'candidate_count':tr['candidate_count'],'eligible_count':tr['eligible_count'],
                'distinct_affinities':len(counts),'tied_candidate_count':sum(v for v in counts.values() if v>1),
                'largest_tie_group':max(counts.values(),default=0),'zero_query_weights':querysum==0,
                'fallback_reason':tr['fallback_reason']},
                'C':{'overlap_count':overlap,'same_multiset':pa==pc,'same_order':pairsa==pairsc,
                'duplicate_pairs_A':20-len(pa),'duplicate_pairs_C':20-len(pc),'eligible_count':c_eligible}})
    assert len(rows)==1540 and sum(b['A_B_facts'] for b in bank_rows)==2376 and sum(b['C_facts'] for b in bank_rows)==2527
    assert sum(b['original_sessions'] for b in bank_rows)==272 and sum(b['groups'] for b in bank_rows)==385
    assert all(b['groups_merging_segments']==b['cross_session_groups']==b['cross_date_groups']==0 for b in bank_rows)
    assert all(sha(p)==h for p,h in hashes.items())
    report={'schema':'sdk20-modules-selection-audit-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':protocol,'run_dir':str(RUN),'questions':1540,'selection_files_complete':4620,
        'new_module_selection_files_complete':3080,'QA_files_read':0,'model_calls':0,
        'overall':summary(rows),'by_conversation':{str(ci):summary([r for r in rows if r['conversation']==ci]) for ci in range(10)},
        'by_category':{str(cat):summary([r for r in rows if r['category']==cat]) for cat in range(1,5)},
        'banks':bank_rows,'per_query':rows,'input_sha256':hashes,'generator':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'definitions':{'replacement':'Count of B-selected fact IDs absent from A; each arm selects20 unique IDs.',
            'affinity_ties':'Exact float equality in the frozen trace, matching the existing selector tie policy; no new tolerance.',
            'C_overlap':'Exact unnormalized (fact text, observation_date) multiset intersection, counting multiplicity; divided by20. Not semantic similarity or retrieval quality.',
            'no_QA':'Only manifests, selection/retrieval traces, fact banks, grouping and original session structure read. No QA file, answer score or correctness read.'},
        'limitations':['Selection change alone is not evidence of accuracy improvement.',
            'Exact text/date overlap can be low because of paraphrasing; it does not measure evidence recall.',
            '385 groups each contain one segment; these artifacts establish splitting, not cross-session/date topic merging.',
            'Input scopes are recorded provenance, not manually verified support for every fact.']}
    assert not DEST.with_suffix('.json').exists() and not DEST.with_suffix('.md').exists()
    DEST.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    bs=report['overall']['B_vs_A'];cs=report['overall']['C_vs_A']
    text=['# SDK20 两模块：仅证据选择诊断','',f'固定1540题；三臂4620份选择产物全部complete，其中两个新模块3080份。未读取QA答案或评分，未调用模型，未改参数。协议：`{protocol}`。','',
      f'B（逐事实类型软排序）与A使用同一2376条事实。20条顺序完全相同：{bs["identical_order"]}/1540；集合相同：{bs["identical_set"]}/1540，其中仅顺序改变{bs["same_set_but_reordered"]}题；集合发生变化{bs["changed_set"]}题。',
      f'每题平均替换{bs["replaced_facts_out_of_20"]["mean"]:.4f}/20条，中位数{bs["replaced_facts_out_of_20"]["median"]:g}，范围{bs["replaced_facts_out_of_20"]["min"]}–{bs["replaced_facts_out_of_20"]["max"]}。候选数均值{bs["candidate_count"]["mean"]:.4f}，eligible均值{bs["eligible_count"]["mean"]:.4f}（范围{bs["eligible_count"]["min"]}–{bs["eligible_count"]["max"]}）。','',
      f'存在至少一组完全相同type-affinity的题目：{bs["queries_with_any_exact_affinity_tie"]}/1540；零query权重：{bs["zero_query_weight_count"]}；fallback分布：`{json.dumps(bs["fallback_reasons"],ensure_ascii=False)}`。同分按原实现的精确浮点比较统计，不另设容差。','',
      f'C（稀疏切分后写入）事实库从2376增至2527条；272个原session变为385个segment和385个group，合并多segment、跨session、跨日期group均为0。因此本轮C是切分实验，不能称为跨日期话题归组已经生效。',
      f'C与A按精确（事实文本，observation_date）多重集合比较，每题平均重合{cs["exact_text_date_multiset_overlap_out_of_20"]["mean"]:.4f}/20条（{cs["mean_overlap_fraction"]*100:.3f}%）；范围{cs["exact_text_date_multiset_overlap_out_of_20"]["min"]}–{cs["exact_text_date_multiset_overlap_out_of_20"]["max"]}。集合完全相同{cs["identical_pair_multiset"]}题，顺序也相同{cs["identical_pair_order"]}题。这个指标只衡量原文表述及日期的精确重合，不衡量语义等价或证据召回。','',
      '| conv | 题数 | B集合改变 | B平均替换/20 | C平均精确重合/20 | A/B事实 | C事实 | session→group |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for b in bank_rows:
        ci=b['conversation'];ss=report['by_conversation'][str(ci)]
        text.append(f'| {ci} | {ss["questions"]} | {ss["B_vs_A"]["changed_set"]} | {ss["B_vs_A"]["replaced_facts_out_of_20"]["mean"]:.3f} | {ss["C_vs_A"]["exact_text_date_multiset_overlap_out_of_20"]["mean"]:.3f} | {b["A_B_facts"]} | {b["C_facts"]} | {b["original_sessions"]}→{b["groups"]} |')
    text+=['','证据选择发生变化并不表示QA质量提升；需等冻结QA实验完成后再按既定分母判断。逐题统计、按类别汇总和输入SHA见同名JSON。']
    DEST.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'output':str(DEST),'overall':report['overall'],'banks':bank_rows},ensure_ascii=False))

if __name__=='__main__':main()
