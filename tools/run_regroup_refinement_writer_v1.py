"""New grouped inputs, frozen native Mem0 writer, separate source-local context arm.

Prepare/check never call models. One independent store per conversation/arm;
failed or uncertain writes cannot be re-entered. QA requires a later source audit.
"""
from __future__ import annotations
import argparse
import copy
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mem0_regroup_sdk20 as frozen
from tools.run_mem0_regroup_remaining_v1 import resources
from tools.run_sdk20_modules import digest, verify, sha, now
from tools.run_pilot400_experiments_v2 import read,publish_json
from jev_mem.paced_embedding_api import PacedEmbeddingAPI,DEFAULT_GATE
from jev_mem.local_bm25_context_v2 import local_bm25_context,validate_resources

VERSION='mem0-regroup-refinement-writer-v1'
BASE=ROOT/'runs/mem0_global_regroup_v1/run_manifest.json'
ARMS=('llm_fullconv','jev_refined','source_local_context')
EXTRA=('tools/run_regroup_refinement_writer_v1.py','jev_mem/paced_embedding_api.py',
       'jev_mem/local_bm25_context_v2.py','jev_mem/sdk20_cached_api.py')

def prepare(run,grouping_dir,arm,context_local=False):
    run=Path(run).resolve();grouping_dir=Path(grouping_dir).resolve()
    if run.exists() or arm not in ARMS:raise ValueError('New independent namespace and declared arm required')
    if context_local!=(arm=='source_local_context'):raise ValueError('Only declared context arm may alter writer context')
    parent=frozen.manifest(BASE.parent)
    data=frozen.load_data();rows=[];source_hashes={};prepared=[]
    if len(data)!=10:raise ValueError('The fixed ten-conversation source is required')
    for ci,item in enumerate(data):
        gp=grouping_dir/f'conv_{ci}'/'grouping.json'
        if arm=='jev_refined' and not gp.exists():
            gp=grouping_dir/f'conv_{ci}'/'identity_and_bounded_refinement'/'grouping.json'
        if not gp.exists():gp=grouping_dir/'conversations'/f'conv_{ci}'/'grouping.json'
        grouping=verify(gp)
        if grouping.get('conversation')!=ci or grouping.get('status')!='complete':raise ValueError('All ten complete groupings required')
        envelopes=frozen.make_envelopes(item,grouping,'dated_merged')
        prepared.append((ci,{'schema':frozen.SCHEMA,'conversation':ci,'arm':arm,
            'grouping':frozen.pointer(gp),'envelopes':envelopes,
            'turn_count':sum(len(x['messages']) for x in envelopes),'dataset_sha256':frozen.SOURCE_HASH}))
        source_hashes[str(gp.resolve())]=sha(gp)
        if context_local:
            from jev_mem.source_local_group_context_v1 import build_context_plan
            sessions=list(frozen.session_envelopes(item))
            for envelope in envelopes:
                build_context_plan(envelope,source_sessions=sessions,segments=grouping['segments'])
    # All source files and the optional writer-context plans pass before the new
    # namespace is created. A late incomplete grouping cannot poison preparation.
    resource_bindings=resources()
    for ci,value in prepared:
        path=run/f'conv_{ci}'/arm/'group_inputs.json'
        frozen.save_new(path,value)
        rows.append({'conversation':ci,'arm':arm,'input':frozen.pointer(path)})
    extra=list(EXTRA)
    if context_local:extra.append('jev_mem/source_local_group_context_v1.py')
    value=copy.deepcopy(parent);value.pop('protocol_sha256')
    value.update(created_utc=now(),arms=[arm],inputs=rows,grouping_source_sha256=source_hashes,
        downstream='canonical plain-text SDK20; fixed pilot400; separate source/index audit required',
        refinement={'schema':VERSION,'arm':arm,'source_local_context':context_local,
                    'base_manifest':frozen.pointer(BASE),'resources':resource_bindings,
                    'code_sha256':{f:sha(ROOT/f) for f in extra},
                    'policy':{'max_workers':8,'api_cache_roots':[],
                              'empty_new_banks':True,'failed_or_unknown_reentry':False,
                              'development_conversations':[0,1],
                              'remaining_requires_source_review':True,
                              'generation_models_unchanged':True},
                    'gate_path':str(DEFAULT_GATE.resolve())})
    if context_local:value['protocol_changes']=[*value['protocol_changes'],'source_local_preceding_two_turns_and_explicit_segment_boundaries']
    publish_json(run/'run_manifest.json',{**value,'protocol_sha256':digest(value)})
    return {'prepared':str(run),'arm':arm,'groups':sum(len(verify(x['input']['path'])['envelopes']) for x in rows),'calls':0}

def manifest(run):
    m=frozen.manifest(run);x=m.get('refinement',{})
    if (x.get('schema')!=VERSION or x.get('arm') not in ARMS or m['arms']!=[x['arm']]
            or x.get('source_local_context')!=(x['arm']=='source_local_context')):raise ValueError('Refinement identity changed')
    frozen.check_pointer(x['base_manifest'])
    if len(m['inputs'])!=10 or {(r['conversation'],r['arm']) for r in m['inputs']}!={(ci,x['arm']) for ci in range(10)}:
        raise ValueError('Exactly ten bound conversation inputs are required')
    for row in m['inputs']:
        inp=frozen.check_pointer(row['input'])
        if (inp.get('conversation'),inp.get('arm'),inp.get('dataset_sha256'))!=(row['conversation'],x['arm'],frozen.SOURCE_HASH):
            raise ValueError('Bound group input identity differs')
        gp=inp['grouping'];grouping=frozen.check_pointer(gp)
        if m['grouping_source_sha256'].get(str(Path(gp['path']).resolve()))!=gp['sha256']:
            raise ValueError('Input grouping differs from manifest source binding')
        if grouping.get('conversation')!=row['conversation'] or grouping.get('status')!='complete' or grouping.get('qa_visible')is not False:
            raise ValueError('Writer source is not a complete QA-blind conversation grouping')
    for f,h in x['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen adapter changed: '+f)
    r=x['resources'];validate_resources(r['snapshot'],r['files_sha256'])
    return m

def admit_expansion(run,m):
    run=Path(run).resolve()
    review=verify(run/'development_review.json')
    if review.get('status')!='passed' or review.get('protocol_sha256')!=m['protocol_sha256']:
        raise ValueError('Both complete development banks require source/index review')
    arm=m['arms'][0]
    if len(review.get('banks',[]))!=2 or {r['path'] for r in review.get('banks',[])}!={str((run/f'conv_{ci}'/arm/'bank.json').resolve()) for ci in (0,1)}:
        raise ValueError('Development review must bind exactly both banks')
    for p in review['banks']:
        bank=frozen.check_pointer(p)
        expected_ci=next(ci for ci in (0,1) if p['path']==str((run/f'conv_{ci}'/arm/'bank.json').resolve()))
        if (bank.get('complete') is not True or bank.get('protocol_sha256')!=m['protocol_sha256']
                or (bank.get('conversation'),bank.get('arm'))!=(expected_ci,arm)):
            raise ValueError('Incomplete or mismatched reviewed bank')
    if not review.get('source_date_audits') or not review.get('index_audits'):raise ValueError('Audit evidence required')
    for p in review['source_date_audits']+review['index_audits']:frozen.check_pointer(p)

def worker(run,ci):
    run=Path(run);m=manifest(run);arm=m['arms'][0];folder=run/f'conv_{ci}'/arm
    if type(ci)is not int or ci not in range(10):raise ValueError('Invalid conversation')
    if ci>1:admit_expansion(run,m)
    outcome=folder/'write_outcome.json';intent=folder/'write_intent.json'
    if outcome.exists():
        r=verify(outcome)
        if (r.get('status')!='complete' or r.get('protocol_sha256')!=m['protocol_sha256']
                or (r.get('conversation'),r.get('arm'))!=(ci,arm)):raise RuntimeError('Failed or mismatched outcome retained')
        bank=frozen.check_pointer(r['bank'])
        if (bank.get('complete') is not True or bank.get('protocol_sha256')!=m['protocol_sha256']
                or (bank.get('conversation'),bank.get('arm'))!=(ci,arm)):
            raise RuntimeError('Completed outcome bank identity differs')
        return r
    if intent.exists() or (folder/'native').exists() or (folder/'write_status.json').exists():raise RuntimeError('Unresolved write attempt retained')
    with intent.open('x',encoding='utf-8') as h:
        json.dump({'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,'pid':os.getpid(),'created_utc':now()},h)
        h.flush();os.fsync(h.fileno())
    resource=m['refinement']['resources']
    api_original,adapter_original=frozen.ModelAPI,frozen.Mem0GroupAdapter
    def api(path,**kwargs):return PacedEmbeddingAPI(path,cache_roots=(),gate_path=m['refinement']['gate_path'],**kwargs)
    state=None
    try:
        frozen.ModelAPI=api
        if m['refinement']['source_local_context']:
            from jev_mem.source_local_group_context_v1 import SourceLocalMem0GroupAdapter
            inp=verify(folder/'group_inputs.json');grouping=verify(inp['grouping']['path'])
            sessions=list(frozen.session_envelopes(frozen.load_data()[ci]))
            frozen.Mem0GroupAdapter=functools.partial(SourceLocalMem0GroupAdapter,source_sessions=sessions,segments=grouping['segments'])
        with local_bm25_context(resource['snapshot'],resource['files_sha256']):
            state=frozen.write_arm(run,ci,arm,m)
    finally:frozen.ModelAPI,frozen.Mem0GroupAdapter=api_original,adapter_original
    result={'schema':VERSION,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,
            'status':state['status'],'write_status':frozen.pointer(folder/'write_status.json'),'finished_utc':now(),'cache_roots':[]}
    if state['status']=='complete':result['bank']=state['bank']
    else:result['error']=state.get('error')
    frozen.save_new(outcome,result)
    return result

def run_write(run,convs):
    m=manifest(run)
    if not convs or len(set(convs))!=len(convs) or any(ci not in range(10) for ci in convs):raise ValueError('Invalid conversation list')
    if any(ci>1 for ci in convs):admit_expansion(run,m)
    state={'schema':VERSION,'status':'running','pid':os.getpid(),'protocol_sha256':m['protocol_sha256'],
           'started_utc':now(),'requested_conversations':convs,'outcomes':{}}
    with frozen.writer_process_lock(run),ProcessPoolExecutor(max_workers=min(8,len(convs))) as pool:
        publish_json(run/'controller_status.json',state)
        tasks={pool.submit(worker,run,ci):ci for ci in convs}
        for f in as_completed(tasks):
            ci=tasks[f]
            try:result=f.result()
            except Exception as e:result={'status':'failed','conversation':ci,'error_type':type(e).__name__,'message':str(e)[:1200]}
            state['outcomes'][str(ci)]=result;state['updated_utc']=now();publish_json(run/'controller_status.json',state)
        state.update(status='complete' if all(x['status']=='complete' for x in state['outcomes'].values()) else 'failed',finished_utc=now())
        publish_json(run/'controller_status.json',state)
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check','write'))
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--grouping-dir',type=Path)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--source-local-context',action='store_true')
    p.add_argument('--conversations',nargs='+',type=int,default=[0,1]);a=p.parse_args();run=a.run_dir.resolve()
    if a.stage=='prepare':
        if not a.grouping_dir or not a.arm:p.error('prepare requires grouping-dir and arm')
        result=prepare(run,a.grouping_dir.resolve(),a.arm,a.source_local_context)
    elif a.stage=='check':result={'valid':bool(manifest(run)),'calls':0}
    else:result=run_write(run,a.conversations)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
