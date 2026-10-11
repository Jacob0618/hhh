"""One-off offline preservation. Never deletes files or calls any model."""
import hashlib
import json
from pathlib import Path
import zipfile
from collections import Counter, defaultdict

ROOT = Path(r'D:\jev mem')
RESEARCH = ROOT / 'research/jev_autoresearch_2026-10-04'
OUT = ROOT / 'archives/retained_rounds_2026-10-05'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = OUT / 'R03_R04_R06_R08_review_v3.zip'
assert not TARGET.exists()
ROUNDS = (3, 4, 6, 8)
RUNS = {3:'evidence_selection_r03_v1',4:'answer_selection_r04_v1',6:'evidence_coverage_r06_v1',8:'facet_discovery_r08_v1'}
index = {}
expected = {}
summaries = {}
sources = {}
banks = {}
checked = {}

def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

def read_ref(ref):
    path = str(Path(ref['path']))
    if path not in checked:
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if ref.get('sha256'):
            assert sha == ref['sha256'], path
        checked[path] = json.loads(raw)
        index[path] = {'sha256':sha}
    return checked[path]

def strip_vectors(value):
    if isinstance(value, dict):
        return {k:strip_vectors(v) for k,v in value.items() if k not in ('embedding','embeddings','vectors','vector')}
    if isinstance(value, list):
        return [strip_vectors(v) for v in value]
    return value

with zipfile.ZipFile(TARGET, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as z:
    def doc(name, value):
        z.writestr(name, compact(value))

    def original(path, name):
        raw = path.read_bytes(); z.writestr(name, raw)
        index[str(path)] = {'sha256':hashlib.sha256(raw).hexdigest(),'entry':name,'preserved_original_bytes':True}

    original(ROOT/'data/locomo10.json', 'data/locomo10.json')
    original(ROOT/'configs/gemini38_seven_arm_full1540_v1.json', 'config/frozen.json')
    for r in ROUNDS:
        p = RESEARCH/f'r{r:02d}_final_summary.json'
        s = json.loads(p.read_text(encoding='utf-8'))
        assert s['audit_status'] == 'no_invalid_artifacts'
        summaries[r] = s
        original(p, f'R{r:02d}/formal_summary.json')
        src = read_ref(s['source_manifest'])
        sources[r] = {j['job_id']:j for j in src['jobs']}
        doc(f'R{r:02d}/source_metadata.json', {k:v for k,v in src.items() if k!='jobs'})
        manifest = ROOT/'runs'/RUNS[r]/'run_manifest.json'
        m = json.loads(manifest.read_text(encoding='utf-8'))
        doc(f'R{r:02d}/protocol.json', m['protocol'])
        scientific=m['protocol'].get('research_protocol')
        if isinstance(scientific,dict) and 'path' in scientific:
            doc(f'R{r:02d}/scientific_protocol.json',read_ref(scientific))
        rows = s['slots_by_view']['final_with_uniform_technical_recovery']
        assert len(rows)==4620 and len({x['job_id'] for x in rows})==4620
        expected[r] = {'counts':dict(Counter(x['method'] for x in rows)),
                       'judge_sums':{a:sum(x['scores']['judge_correct'] for x in rows if x['method']==a) for a in ('baseline','cheap','jev')},
                       'f1_sums':{a:sum(x['scores']['official_locomo_f1'] for x in rows if x['method']==a) for a in ('baseline','cheap','jev')}}
        for source in (sources[3] if r==4 else sources[r]).values():
            conv=source['conversation']
            if conv not in banks:
                bank=read_ref(source['bank']);banks[conv]=strip_vectors(bank)
                doc(f'facts/conv_{conv}.json',banks[conv])
                group=bank.get('group_inputs')
                if isinstance(group,dict) and 'path' in group:
                    doc(f'facts/conv_{conv}_group_inputs.json',read_ref(group))
        with z.open(f'R{r:02d}/qa_and_selected20.jsonl','w') as out:
            for row in rows:
                if r == 4:
                    source = sources[3][row['original_qa_method']+'/'+row['query_id']]
                else:
                    source = sources[r][row['job_id']]
                facts = source['selected_facts']
                assert [f['id'] for f in facts]==row['selected_ids'] and len(facts)==20 and len(set(row['selected_ids']))==20
                conv=row['conversation']
                reader=read_ref(row.get('selected_original_reader',row['reader_artifact']))
                judge=read_ref(row.get('selected_original_judge',row['judge_artifact']))
                pred=reader.get('result',{}).get('prediction',{})
                assert pred.get('answer')==row['reader_answer'],row['job_id']
                result={k:row.get(k) for k in ('job_id','query_id','conversation','category','method','question','reference','reference_date','selection_status','reader_status','judge_status','qa_status','scores','reader_answer','judge_raw_text','judge_parse_mode','errors','original_route','answer_selection')}
                result.update(selected_facts=facts,reader_raw=pred.get('raw'),judge_result=judge.get('result'),
                              original_reader_ref=row.get('selected_original_reader',row['reader_artifact']),
                              original_judge_ref=row.get('selected_original_judge',row['judge_artifact']))
                out.write(compact(result)+b'\n')
        with z.open(f'R{r:02d}/retrieval_or_choice.jsonl','w') as out:
            for row in rows:
                ref=row['answer_selection'] if r==4 else sources[r][row['job_id']]['selection']
                selection=read_ref(ref)
                if r in (4,8):
                    response=read_ref(selection['response']) if selection.get('response') else None
                    selection={**selection,'preserved_score_response':strip_vectors(response)}
                out.write(compact({'job_id':row['job_id'],'original_ref':ref,'result':strip_vectors(selection)})+b'\n')
        print(f'R{r:02d} QA4620 + decisions preserved',flush=True)
    with z.open('shared/original_candidate_pools.jsonl','w') as out:
        for q in sorted({row['query_id'] for row in summaries[3]['slots_by_view']['final_with_uniform_technical_recovery']}):
            p=ROOT/'runs/evidence_exchange_r01_v1/proposals'/f'{q}.json'
            obj=json.loads(p.read_text(encoding='utf-8'))
            out.write(compact({'query_id':q,'conversation':obj['conversation'],'pool':obj['pool']})+b'\n')
    with z.open('R04/two_answer_candidates.jsonl','w') as out:
        for p in sorted((ROOT/'runs/answer_selection_r04_v1/requests').glob('*.json')):
            d=json.loads(p.read_text(encoding='utf-8'))
            assert len(d['candidates'])==2
            out.write(compact(d)+b'\n')
    doc('archive_index.json',index)
    doc('expected_metrics.json',expected)
    z.writestr('README.txt', 'Offline QA/retrieval review archive; no API response caches or embedding vectors.\nEach round has 4620 final three-arm cases and original formal summary with first-pass/final views.\nFacts, source dialogue, exact selected20, Reader/Judge text, decisions, scores, R03/R06 original candidate pools and R04 both answer candidates are self-contained. R08 discovered IDs resolve through common per-conversation facts.\nOld absolute paths are historical provenance, not live dependencies. Native stores and HTTP cache are intentionally removed; old online/raw-cache validators cannot run without the deleted artifacts. No claim of full online reproducibility.\n')

with zipfile.ZipFile(TARGET) as z:
    assert z.testzip() is None
    lookup={}
    for c in range(10):
        b=json.loads(z.read(f'facts/conv_{c}.json'));lookup[c]={f['id']:f for f in b['facts']}
    for r in ROUNDS:
        rows=[json.loads(x) for x in z.read(f'R{r:02d}/qa_and_selected20.jsonl').splitlines()]
        assert len(rows)==4620 and len({x['job_id'] for x in rows})==4620
        for x in rows:
            assert len(x['selected_facts'])==20
            for f in x['selected_facts']:
                assert lookup[x['conversation']][f['id']]['text']==f['text']
                assert lookup[x['conversation']][f['id']].get('observation_date')==f.get('observation_date')
            assert x['reader_raw'] is not None and x['judge_result']['raw']==x['judge_raw_text']
        for a in ('baseline','cheap','jev'):
            group=[x for x in rows if x['method']==a]
            assert len(group)==1540
            assert sum(x['scores']['judge_correct'] for x in group)==expected[r]['judge_sums'][a]
            assert abs(sum(x['scores']['official_locomo_f1'] for x in group)-expected[r]['f1_sums'][a])<1e-9
    assert len(z.read('R04/two_answer_candidates.jsonl').splitlines())==1540
result={'status':'verified','archive':str(TARGET),'sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'bytes':TARGET.stat().st_size,'rounds':[3,4,6,8],'valid_cases_per_round':4620,'shared_fact_count':sum(len(v) for v in lookup.values()),'api_calls':0,'cache_files_preserved_for_these_rounds':0,'metrics':expected}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='metrics'},ensure_ascii=False),flush=True)
