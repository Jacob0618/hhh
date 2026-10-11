"""One uniform post-batch recovery for nine reviewed technical failures.

Output budget changes (16384 -> 65536), with explicit function-calling NONE.
First responses remain frozen. No returned tool call is ever executed.
This version refuses other error populations and cannot be recursively recovered.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_fullconv_llm_segmentation_v1 as first
from jev_mem import fullconv_llm_segmentation_v1 as core
from tools.run_global_segment_grouping import read,verify_seal,pointer,checked_pointer,sha,save_once

VERSION='fullconv-llm-output-budget-recovery-v1'
SOURCE=ROOT/'runs/fullconv_llm_segmentation_v1'
ORIGINAL_BODY=core.request_body
ORIGINAL_GENERATION=copy.deepcopy(core.GENERATION)
EXPECTED_FAILURES={0:'MAX_TOKENS',1:'MALFORMED_FUNCTION_CALL',2:'MAX_TOKENS',3:'MAX_TOKENS',
                   4:'MAX_TOKENS',5:'MAX_TOKENS',6:'MAX_TOKENS',7:'MALFORMED_FUNCTION_CALL',8:'STOP_NON_TEXT_FUNCTION_CALL'}

def sources():
    m=first.check(SOURCE);status=read(SOURCE/'status.json')
    if set(status['conversations'])!={str(i) for i in range(10)}:raise ValueError('Ten original conversations required')
    if status['status']!='failed' or any(x['status'] not in ('failed','complete') for x in status['conversations'].values()):
        raise ValueError('This bounded recovery requires all ten first-pass conversations terminal')
    rows=[];inherited=[]
    for ci in range(10):
        p=SOURCE/f'conv_{ci}'/'grouping.json';r=verify_seal(read(p))
        if r['status'] not in ('failed','complete') or len(r['api_events'])!=1:raise ValueError('Expected one logical response per conversation')
        e=r['api_events'][0]
        if e['error_type'] is not None:raise ValueError('Unknown/transport terminal failure needs a different review')
        raw=checked_pointer({'path':e['cache_path'],'sha256':e['cache_sha256']})
        cs=raw['response'].get('candidates',[])
        inp=verify_seal(checked_pointer(next(j['input'] for j in m['jobs'] if j['conversation']==ci)))
        from jev_mem.sdk20_cached_api import digest as request_digest
        if (raw['request']['body']!=inp['request_body'] or raw['request']['model']!=core.MODEL
            or raw['request'].get('provider')!='gemini' or raw['request'].get('operation')!='generateContent'
            or request_digest(raw['request'])!=e['request_hash']
            or e['context'].get('conversation')!=ci or raw['response'].get('modelVersion')!=core.MODEL):
            raise ValueError('Only exact original requests are admitted')
        if r['status']=='complete':
            if ci!=9:raise ValueError('The reviewed complete population changed')
            replay=core.parse_partition(core.response_text(raw['response']),inp['raw_turns'])
            if any(replay[k]!=r[k] for k in ('segments','groups','trace')):raise ValueError('Completed source differs from raw replay')
            inherited.append({'conversation':ci,'result':pointer(p),'generation':inp['request_body']['generationConfig']})
            continue
        if len(cs)!=1:raise ValueError('One original candidate required')
        finish=cs[0].get('finishReason');parts=cs[0].get('content',{}).get('parts',[])
        failure_kind=('STOP_NON_TEXT_FUNCTION_CALL' if finish=='STOP' and len(parts)==1
                      and 'functionCall' in parts[0] and not parts[0].get('text') else finish)
        if failure_kind!=EXPECTED_FAILURES.get(ci):raise ValueError('Unreviewed failure type or conversation')
        rows.append({'conversation':ci,'failure':pointer(p),'input':next(j['input'] for j in m['jobs'] if j['conversation']==ci),
                     'raw':{'path':e['cache_path'],'sha256':e['cache_sha256']},'failure_kind':failure_kind,
                     'usage':raw['response'].get('usageMetadata',{})})
    return {'schema':VERSION,'source_manifest':pointer(SOURCE/'manifest.json'),'source_status':pointer(SOURCE/'status.json'),
            'failure_population':rows,'inherited_complete':inherited,
            'changed_field':'generationConfig.maxOutputTokens','old':16384,'new':65536,
            'additional_change':{'toolConfig':{'functionCallingConfig':{'mode':'NONE'}}},
            'failure_counts':{'MAX_TOKENS':6,'MALFORMED_FUNCTION_CALL':2,'STOP_NON_TEXT_FUNCTION_CALL':1},
            'tool_outputs_never_executed':True,'budget_change_not_proven_cure_for_tool_failures':True,
            'unchanged':['model','thinkingLevel','temperature','prompt','all_source_turns','partition_parser'],
            'logical_recovery_attempts_per_conversation':1,'subsequent_recovery_allowed':False}

@contextmanager
def binding():
    code=first.CODE;generation=core.GENERATION;body=core.request_body
    core.GENERATION={**ORIGINAL_GENERATION,'maxOutputTokens':65536}
    def text_only_body(raw):
        value=ORIGINAL_BODY(raw)
        value['toolConfig']={'functionCallingConfig':{'mode':'NONE'}}
        return value
    core.request_body=text_only_body
    first.CODE=(*code,'tools/recover_fullconv_llm_segmentation_v1.py')
    try:yield
    finally:first.CODE=code;core.GENERATION=generation;core.request_body=body

def prepare(run):
    review=sources()
    with binding():result=first.prepare(run,ROOT/'runs/global_segment_grouping_v1')
    save_once(run/'recovery_review.json',review)
    # Preserve successful first-pass partition and its ORIGINAL input/raw pointers.
    # The 64K prospective input is never executed for these explicitly inherited rows.
    for row in review['inherited_complete']:
        source=verify_seal(checked_pointer(row['result']))
        value={k:v for k,v in source.items() if k!='content_sha256'}
        value.update(inherited_firstpass=True,source_result=row['result'],
                     effective_generation=row['generation'],new_model_calls=0)
        save_once(run/f"conv_{row['conversation']}"/'grouping.json',value)
    return result

def check(run):
    review=verify_seal(read(run/'recovery_review.json'))
    if {k:v for k,v in review.items() if k!='content_sha256'}!=sources():raise ValueError('Recovery population changed')
    with binding():m=first.check(run)
    if m['generation']['maxOutputTokens']!=65536:raise ValueError('Recovery budget changed')
    for row in review['inherited_complete']:
        old=verify_seal(checked_pointer(row['result']))
        new=verify_seal(read(run/f"conv_{row['conversation']}"/'grouping.json'))
        if any(new[k]!=v for k,v in old.items() if k!='content_sha256') or not new.get('inherited_firstpass'):
            raise ValueError('Inherited partition/raw/source pointers changed')
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check','run'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
    if a.stage=='prepare':result=prepare(run)
    else:
        check(run)
        if a.stage=='run':
            with binding():result=first.execute(run)
        else:result={'valid':True,'calls':0}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
