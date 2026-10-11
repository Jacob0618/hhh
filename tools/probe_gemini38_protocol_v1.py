"""One synthetic connectivity/format probe; never a LoCoMo effect measurement."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.gemini38_provider_v1 import Gemini38API, Gemini38ExtractionLLM, MODEL
from jev_mem.evaluation_gemini38_v1 import read_answer, judge_answer
from tools.run_sdk20_modules import save_new, now, sha


def main():
    run = ROOT/'research/gemini38_retest_2026-10-03/live_synthetic_probe_v1'
    if run.exists():
        raise FileExistsError('One probe only; retained attempts require explicit review')
    run.mkdir(parents=True)
    api = Gemini38API(run, retries=2, context={'purpose':'synthetic_3_8_transport_format_smoke_not_locomo'})
    result = {'created_utc':now(), 'model':MODEL, 'benchmark_questions':0, 'purpose':'transport/schema check, not quality or stability evidence', 'stages':[]}
    try:
        from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
        prompt = generate_additive_extraction_prompt(existing_memories=[], new_messages='user: Rina collects blue stones.',
            last_k_messages=[], current_date='2026-09-27', timestamp='2026-09-27', custom_instructions=None)
        extraction = Gemini38ExtractionLLM(api.scoped(component='writer')).generate_response([
            {'role':'system','content':ADDITIVE_EXTRACTION_PROMPT}, {'role':'user','content':prompt}])
        result['stages'].append({'stage':'writer', 'status':'complete', 'memory_count':len(json.loads(extraction)['memory']), 'parsed':json.loads(extraction)})
        facts = [{'id':'synthetic_f1','text':'Rina collects blue stones.','observation_date':'2026-09-27'}]
        facts += [{'id':f'synthetic_f{i}', 'text':f'Unrelated inventory item number {i} is in a box.', 'observation_date':'2026-09-27'} for i in range(2,21)]
        reader = read_answer(api.scoped(component='reader'), 'What does Rina collect?', 'blue stones', facts, reference_date='2026-09-27')
        save_new(run/'reader.json',reader)
        result['stages'].append({'stage':'reader', 'status':'complete', 'answer':reader['prediction']['answer']})
        judge = judge_answer(api.scoped(component='judge'), 'What does Rina collect?', 'blue stones', reader['prediction']['answer'])
        save_new(run/'judge.json',judge)
        result['stages'].append({'stage':'judge','status':'complete','label':judge['label'],'parse_mode':judge['parse_mode']})
        result['status']='complete'
    except Exception as exc:
        result['status']='failed'
        result['error']={'type':type(exc).__name__,'message':api._redact(exc)[:1000]}
    result['finished_utc']=now()
    result['code_sha256']={p:sha(ROOT/p) for p in ('tools/probe_gemini38_protocol_v1.py','jev_mem/gemini38_provider_v1.py','jev_mem/evaluation_gemini38_v1.py')}
    save_new(run/'result.json',result)
    print(json.dumps({'status':result['status'],'stages':[{k:v for k,v in s.items() if k not in ('parsed',)} for s in result['stages']], 'result':str(run/'result.json'), 'error':result.get('error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
