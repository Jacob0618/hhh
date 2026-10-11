"""Frozen full-conversation Gemini 3.8 comparator; no writer or QA on this path."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import threading
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import fullconv_llm_segmentation_v1 as core
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from tools.run_global_segment_grouping import (read, publish_json, pointer, checked_pointer,
    save_once, verify_seal, sha, now, exclusive_driver, load_plan, frozen)

CODE = ('jev_mem/fullconv_llm_segmentation_v1.py', 'tools/run_fullconv_llm_segmentation_v1.py',
        'jev_mem/sdk20_cached_api.py', 'jev_mem/providers.py')

def prepare(run, source):
    if run.exists(): raise FileExistsError('New run directory required')
    parent = frozen(source)
    jobs = []
    for ci in range(10):
        plan = load_plan(parent, ci)
        raw = [t for s in plan['segments'] for t in s['turns']]
        path = run / f'conv_{ci}' / 'input.json'
        save_once(path, {'conversation':ci, 'raw_turns':raw, 'request_body':core.request_body(raw),
                        'source_plan':parent['plans'][str(ci)]})
        jobs.append({'conversation':ci, 'input':pointer(path)})
    save_once(run/'manifest.json', {'schema':core.VERSION, 'created_utc':now(), 'jobs':jobs,
        'source_manifest':pointer(source/'manifest.json'), 'code_sha256':{f:sha(ROOT/f) for f in CODE},
        'model':core.MODEL, 'generation':core.GENERATION, 'conversation_workers':8,
        'failure_policy':'retain_firstpass_all_then_at_most_one_separate_technical_recovery_no_correctness_retry',
        'llm_is_gold':False, 'qa_gold_or_existing_partition_in_prompt':False})
    return {'prepared':str(run),'conversations':10,'model_calls':0}

def check(run):
    m=verify_seal(read(run/'manifest.json'))
    if m['code_sha256']!={f:sha(ROOT/f) for f in CODE}: raise ValueError('Frozen code changed')
    for j in m['jobs']:
        row=verify_seal(checked_pointer(j['input']))
        if row['request_body']!=core.request_body(row['raw_turns']): raise ValueError('Request changed')
        checked_pointer(row['source_plan'])
    return m

def execute(run):
    m=check(run)
    state={'schema':core.VERSION,'pid':os.getpid(),'status':'running','started_utc':now(),
           'conversations':{str(i):{'status':'pending'} for i in range(10)}}
    lock=threading.Lock()
    def update(ci=None, **kw):
        with lock:
            (state if ci is None else state['conversations'][str(ci)]).update(kw)
            state['updated_utc']=now()
            publish_json(run/'status.json',state)
    with exclusive_driver(run):
        api=CachedLegacyAPI(run/'api',retries=2,timeout=300,cache_roots=(),context={'phase':core.VERSION})
        def work(j):
            ci=j['conversation']; path=run/f'conv_{ci}'/'grouping.json'
            if path.exists():
                result=verify_seal(read(path))
                if result['status']!='complete': raise RuntimeError('Prior terminal failure retained')
                update(ci,status='complete',reused_terminal=True,**result['trace']); return
            row=verify_seal(checked_pointer(j['input']))
            a=api.scoped(conversation=ci)
            update(ci,status='requesting',turns=len(row['raw_turns']))
            result={'schema':core.VERSION,'conversation':ci,'input':j['input'],'started_utc':now()}
            try:
                response=a.request('gemini',core.MODEL,row['request_body'],stage='fullconv_segment_and_group')
                text=core.response_text(response)
                result['response_text']=text
                result.update(core.parse_partition(text,row['raw_turns']),status='complete',error=None)
            except Exception as e:
                result.update(status='failed',error={'type':type(e).__name__,'message':a._redact(e)})
            result.update(finished_utc=now(),api_events=a.events)
            save_once(path,result)
            update(ci,status=result['status'],**result.get('trace',{}),error=result.get('error'))
        update()
        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks={pool.submit(work,j):j['conversation'] for j in m['jobs']}
            for f in as_completed(tasks):
                try:f.result()
                except Exception as e:update(tasks[f],status='failed',error={'type':type(e).__name__,'message':api._redact(e)})
        update(status='complete' if all(v['status']=='complete' for v in state['conversations'].values()) else 'failed',finished_utc=now())
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','check','run'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,default=ROOT/'runs/global_segment_grouping_v1')
    a=p.parse_args();run=a.run_dir.resolve()
    result=prepare(run,a.source_dir.resolve()) if a.stage=='prepare' else execute(run) if a.stage=='run' else {'valid':bool(check(run))}
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
