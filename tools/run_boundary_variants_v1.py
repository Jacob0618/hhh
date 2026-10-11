"""Prepare/check offline; boundary/global-score/write are explicit paid stages.

Reuse frozen boundary predicates, global grouping and native write_arm. New
namespaces only; uncertain attempts never silently redispatch. Full1540 QA is
admitted separately after complete new native banks and their source audit.
"""
from __future__ import annotations
import argparse
import copy
import functools
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import boundary_variants_v1 as variants
from jev_mem import global_segment_grouping as grouping
from jev_mem.topic_grouping import raw_turns_from_locomo, verify_partition
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from tools import run_global_segment_grouping as global_runner
from tools import run_mem0_regroup_sdk20 as writer
from tools.run_sdk20_modules import digest, verify, sha, now

SOURCE = ROOT / 'runs/pilot400_modules_v2'
GLOBAL_SOURCE = ROOT / 'runs/global_segment_grouping_v1'
LOCK = ROOT / 'configs/sdk20_full1540_locked_v1.json'
FILES = ('jev_mem/boundary_variants_v1.py', 'tools/run_boundary_variants_v1.py',
         'jev_mem/topic_boundary_budget.py', 'jev_mem/topic_boundaries.py',
         'jev_mem/sparse_topic_pipeline.py', 'jev_mem/global_segment_grouping.py',
         'tools/run_global_segment_grouping.py', 'tools/run_mem0_regroup_sdk20.py',
         'jev_mem/sdk20_cached_api.py', 'jev_mem/providers.py',
         'jev_mem/paced_embedding_api.py', 'jev_mem/local_bm25_context_v2.py',
         'tools/run_mem0_regroup_remaining_v1.py')
POLICY = {'windows': list(variants.WINDOWS), 'priority': variants.PRIORITY,
          'c2_policy': variants.C2_POLICY, 'boundary_cache_roots': [],
          'transport_retries': 2, 'boundary_budget_unit': 'logical_score_many_requests',
          'max_boundary_logical_per_session': 6, 'qa_size': 1540,
          'qa_implementation': 'original_SDK20_plain_Reader_and_Judge_separate_root_admission',
          'global_parameters': global_runner.PARAMETERS}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    global_runner.save_once(path, value)


def pointer(path):
    return global_runner.pointer(path)


def checked(value):
    return global_runner.verify_seal(global_runner.checked_pointer(value))


def prepare(run):
    run = Path(run).resolve()
    if run.exists():
        raise FileExistsError('Prepare requires a new namespace')
    source_manifest = read(SOURCE/'run_manifest.json')
    frozen_params = source_manifest['protocol']['parameters']
    names = ('cut_threshold', 'keep_threshold', 'dependency_threshold')
    parameters = {n: frozen_params[n] for n in names}
    if (frozen_params['initial_boundary_budget'], frozen_params['refinement_budget'],
        frozen_params['initial_window'], frozen_params['expanded_window'], frozen_params['provider_retries']) != (4,2,2,4,2):
        raise ValueError('Source budget/window/transport policy changed')
    old_global = global_runner.frozen(GLOBAL_SOURCE)
    dataset = ROOT/'data/locomo10.json'
    if sha(dataset) != writer.SOURCE_HASH:
        raise ValueError('Source dataset differs')
    data = read(dataset)
    plans = []
    comparison = []
    for ci in range(10):
        old_plan = global_runner.load_plan(old_global, ci)
        gp, ep = old_plan['source_grouping'], old_plan['source_embeddings']
        original, embedding = checked(gp), checked(ep)
        raw = raw_turns_from_locomo(data[ci]['conversation'])
        if original['status'] != 'complete' or embedding['status'] != 'complete':
            raise ValueError('Completed inherited source required')
        if original['binding']['conversation'] != ci or embedding['binding']['conversation'] != ci:
            raise ValueError('Source conversation differs')
        verify_partition(raw, original['segments'])
        if [t for s in original['segments'] for t in s['turns']] != raw:
            raise ValueError('Original segment source order changed')
        if any(x['binding']['raw_source_sha256'] != grouping.fingerprint(raw) for x in (original, embedding)):
            raise ValueError('Source embedding provenance differs')
        proposal = variants.propose_multiscale_boundaries(raw, variants.old._vectors(embedding['vectors'],len(raw)))
        old_proposal = original['trace']['boundary']['proposer']
        old_gaps = {r['right_source_index'] for s in old_proposal['sessions'] for r in s['initial_requests']}
        new_gaps = {r['right_source_index'] for s in proposal['sessions'] for r in s['initial_requests']}
        plans.append({'conversation': ci, 'source_grouping': gp, 'source_embeddings': ep,
            'raw_source_sha256': grouping.fingerprint(raw), 'proposal': proposal,
            'old_initial_cut_indices': original['trace']['boundary']['semantic_cut_indices'],
            'qa_visible': False})
        comparison.append({'conversation':ci, 'old_segments':len(original['segments']),
            'initial_logical_requests':proposal['initial_logical_requests'],
            'same_initial_gaps':len(old_gaps & new_gaps), 'old_only_gaps':sorted(old_gaps-new_gaps),
            'c1_c2_only_gaps':sorted(new_gaps-old_gaps)})
    if sum(x['old_segments'] for x in comparison) != 385:
        raise ValueError('Frozen old initial segmentation cohort must have 385 segments')
    sources = {str(p):sha(p) for p in (SOURCE/'run_manifest.json', GLOBAL_SOURCE/'manifest.json', LOCK, dataset)}
    plan_pointers = {}
    for plan in plans:
        path = run/'plans'/f"conv_{plan['conversation']}.json"
        save(path, plan)
        plan_pointers[str(plan['conversation'])] = pointer(path)
    m = {'schema':variants.VERSION, 'created_utc':now(), 'arms':list(variants.ARMS),
         'parameters':parameters, 'policy':POLICY, 'plans':plan_pointers,
         'source_sha256':sources, 'code_sha256':{p:sha(ROOT/p) for p in FILES},
         'source_parameters_pointer':pointer(SOURCE/'run_manifest.json'),
         'official_lock':pointer(LOCK), 'baseline':{'official_f1':54.6095082528163,'judge_percent':66.03896103896105},
         'comparators':['old_initial_385_then_unchanged_global_grouping','c1_multiscale','c2_neighbour'],
         'old_refined_408_is_not_primary_boundary_control':True,
         'candidate_comparison':comparison, 'model_calls_at_prepare':0, 'new_embedding_calls':0}
    save(run/'manifest.json', m)
    return {'prepared':str(run), 'calls':0, 'candidate_comparison':comparison}


def manifest(run):
    m = global_runner.verify_seal(read(Path(run)/'manifest.json'))
    if m['schema'] != variants.VERSION or m['policy'] != POLICY or m['arms'] != list(variants.ARMS):
        raise ValueError('Frozen variant policy changed')
    for path, value in m['source_sha256'].items():
        if sha(path) != value:
            raise ValueError('Frozen source changed: '+path)
    if m['code_sha256'] != {p:sha(ROOT/p) for p in FILES}:
        raise ValueError('Frozen implementation changed')
    source = global_runner.checked_pointer(m['source_parameters_pointer'])
    if m['parameters'] != {k:source['protocol']['parameters'][k] for k in m['parameters']}:
        raise ValueError('Boundary thresholds changed')
    if set(m['plans']) != {str(ci) for ci in range(10)}:
        raise ValueError('Ten plans required')
    for ci, ptr in m['plans'].items():
        p = checked(ptr)
        if p['conversation'] != int(ci):
            raise ValueError('Plan binding differs')
        checked(p['source_grouping']); checked(p['source_embeddings'])
    return m


def source_for(m, ci):
    p = checked(m['plans'][str(ci)])
    original = checked(p['source_grouping'])
    raw = [t for s in original['segments'] for t in s['turns']]
    return p, raw, checked(p['source_embeddings'])['vectors']


def boundary(run, arm, convs):
    m = manifest(run)
    from jev_mem.providers import load_credentials
    keys = load_credentials()  # only this explicit paid stage constructs a client
    results=[]
    with global_runner.exclusive_driver(Path(run)):
        for ci in convs:
            p, raw, vectors = source_for(m, ci)
            path = Path(run)/arm/'boundary'/f'conv_{ci}'/'grouping.json'
            if path.exists():
                old = checked(pointer(path))
                if old['status'] != 'complete' or old['parent_manifest_sha256'] != sha(Path(run)/'manifest.json'):
                    raise ValueError('Failed/changed boundary outcome retained')
                results.append({'conversation':ci,'reused_terminal':True}); continue
            folder = path.parent
            api = CachedLegacyAPI(folder/'api', keys=keys, retries=2, timeout=150,
                cache_roots=(), context={'phase':'boundary_variants','conversation':ci,'arm':arm})
            try:
                result = variants.segment_boundaries(raw, arm=arm, judge=api, embeddings=vectors, **m['parameters'])
                result.update(status='complete', error=None)
            except Exception as exc:
                result={'status':'failed', 'error':{'type':type(exc).__name__,'message':api._redact(exc)},
                        'trace':getattr(exc,'trace',{})}
            result.update(conversation=ci, binding={'conversation':ci,'raw_source_sha256':p['raw_source_sha256']},
                parent_manifest_sha256=sha(Path(run)/'manifest.json'),
                source_plan=m['plans'][str(ci)], api_events=api.events,
                provider_ledger=str((folder/'api/api_calls.jsonl').resolve()),
                token_policy='actual_usage_from_raw_response_or_ledger_missing_is_unknown',
                qa_visible=False, new_embedding_calls=0)
            save(path,result)
            results.append({'conversation':ci,'status':result['status'],
                            'logical_requests':result['trace'].get('logical_judge_calls')})
            if result['status'] != 'complete':
                raise RuntimeError('Failed boundary retained; separate recovery version required')
    return results


def global_prepare(run, arm):
    m = manifest(run)
    folder = Path(run)/arm/'global'
    if folder.exists():
        raise FileExistsError('New global namespace required')
    plans=[]
    for ci in range(10):
        p, raw, vectors=source_for(m,ci)
        bp=Path(run)/arm/'boundary'/f'conv_{ci}'/'grouping.json'
        b=checked(pointer(bp))
        if b['status'] != 'complete' or b['parent_manifest_sha256'] != sha(Path(run)/'manifest.json'):
            raise ValueError('Ten completed bound segmentations required')
        verify_partition(raw,b['segments'])
        plan=grouping.prepare_candidates(ci,b['segments'],vectors,
            top_k=global_runner.PARAMETERS['top_k'],max_pair_characters=global_runner.PARAMETERS['max_pair_characters'])
        plan.update(source_grouping=pointer(bp),source_embeddings=p['source_embeddings'],
            source_dataset=pointer(ROOT/'data/locomo10.json'), source_order_checked_against_raw_conversation=True,
            initial_judgments_started=False,qa_or_gold_used_or_forwarded=False)
        plans.append(plan)
    ptrs={}
    for plan in plans:
        path=folder/'conversations'/f"conv_{plan['conversation']}"/'plan.json'
        save(path,plan);ptrs[str(plan['conversation'])]=pointer(path)
    value={'schema':grouping.VERSION,'created_utc':now(),'parameters':global_runner.PARAMETERS,
        'criterion':grouping.CRITERION,'plans':ptrs,'code_sha256':{p:sha(ROOT/p) for p in global_runner.FILES},
        'all_segments':sum(len(p['segments']) for p in plans),'cross_conversation_pairs':0,
        'new_embedding_calls':0,'model_calls_at_prepare':0,'manual_labels_used_to_select_or_score':False,
        'boundary_variant':arm,'boundary_parent':pointer(Path(run)/'manifest.json')}
    save(folder/'manifest.json',value)
    return {'prepared':str(folder),'calls':0,'segments':value['all_segments']}


def global_manifest(run):
    m=global_runner.frozen(run)  # unchanged parameters, criterion and code check
    parent=Path(m['boundary_parent']['path']).parent
    if sha(m['boundary_parent']['path']) != m['boundary_parent']['sha256']:
        raise ValueError('Boundary parent changed')
    manifest(parent)
    if m['boundary_variant'] not in variants.ARMS:
        raise ValueError('Boundary global arm differs')
    return m


def global_score(run,arm,convs):
    # All original function bodies share new independent globals; no module patch.
    namespace=dict(vars(global_runner))
    for name,value in vars(global_runner).items():
        if isinstance(value,FunctionType) and value.__module__ == global_runner.__name__:
            fn=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            fn.__kwdefaults__=value.__kwdefaults__;namespace[name]=fn
    namespace['frozen']=global_manifest
    return namespace['score_run'](Path(run)/arm/'global',convs)


def build(run,arm):
    m=manifest(run);gf=Path(run)/arm/'global';global_manifest(gf)
    destination=Path(run)/arm/'writer'
    if destination.exists():
        raise FileExistsError('New writer namespace required')
    parent=writer.manifest(ROOT/'runs/mem0_global_regroup_v1')
    data=writer.load_data();prepared=[];sources={}
    for ci,item in enumerate(data):
        gp=gf/'conversations'/f'conv_{ci}'/'grouping.json'
        value=verify(gp)
        if value.get('status') != 'complete' or value.get('conversation') != ci or value.get('qa_visible') is not False:
            raise ValueError('Ten completed QA-blind grouping sources required')
        plan=checked(global_manifest(gf)['plans'][str(ci)])
        if value['plan_sha256'] != sha(gf/'conversations'/f'conv_{ci}'/'plan.json'):
            raise ValueError('Global outcome plan differs')
        raw=[t for s in plan['segments'] for t in s['turns']]
        verify_partition(raw,value['groups'])
        env=writer.make_envelopes(item,value,'dated_merged')
        prepared.append((ci,{'schema':writer.SCHEMA,'conversation':ci,'arm':arm,'grouping':pointer(gp),
            'envelopes':env,'turn_count':sum(len(x['messages']) for x in env),'dataset_sha256':writer.SOURCE_HASH}))
        sources[str(gp.resolve())]=sha(gp)
    from tools.run_mem0_regroup_remaining_v1 import resources
    resource_bindings=resources()
    rows=[]
    for ci,value in prepared:
        path=destination/f'conv_{ci}'/arm/'group_inputs.json'
        writer.save_new(path,value);rows.append({'conversation':ci,'arm':arm,'input':pointer(path)})
    value=copy.deepcopy(parent);value.pop('protocol_sha256')
    value.update(created_utc=now(),arms=[arm],inputs=rows,grouping_source_sha256=sources,
        code_sha256={**parent['code_sha256'],**m['code_sha256']},
        boundary_variant={'schema':variants.VERSION,'arm':arm,'parent':pointer(Path(run)/'manifest.json'),
            'global':pointer(gf/'manifest.json'),'resources':resource_bindings,'api_cache_roots':[]},
        downstream='new native bank then separate audited original SDK20 full1540; no QA executed at build')
    from tools.run_pilot400_experiments_v2 import publish_json
    publish_json(destination/'run_manifest.json',{**value,'protocol_sha256':digest(value)})
    writer.manifest(destination)
    ready={'schema':variants.VERSION+'-source-ready','status':'source_ready_for_writer',
        'arm':arm,'writer_manifest':pointer(destination/'run_manifest.json'),
        'writer_group_count':sum(len(v['envelopes']) for _,v in prepared),
        'full1540_qa_run_suggested':str((Path(run)/arm/'qa_full1540').resolve()),
        'qa_admission':'pending_complete_banks_and_source_index_audit','models_unchanged':True,
        'writer_and_qa_calls_at_build':0}
    save(Path(run)/arm/'source_ready.json',ready)
    return ready


def write(run,arm,convs):
    manifest(run);folder=Path(run)/arm/'writer';m=writer.manifest(folder)
    from jev_mem.paced_embedding_api import PacedEmbeddingAPI, DEFAULT_GATE
    from jev_mem.local_bm25_context_v2 import local_bm25_context
    resources=m['boundary_variant']['resources']
    namespace=dict(vars(writer),ModelAPI=functools.partial(PacedEmbeddingAPI,
        cache_roots=(),gate_path=DEFAULT_GATE))
    fn=FunctionType(writer.write_arm.__code__,namespace,writer.write_arm.__name__,writer.write_arm.__defaults__)
    results=[]
    with writer.writer_process_lock(folder),local_bm25_context(resources['snapshot'],resources['files_sha256']):
        for ci in convs:
            target=folder/f'conv_{ci}'/arm
            status=target/'write_status.json'
            if status.exists():
                old=read(status)
                if old.get('status') != 'complete' or old.get('protocol_sha256') != m['protocol_sha256']:
                    raise RuntimeError('Failed/uncertain writer attempt retained')
            elif (target/'native').exists():
                raise RuntimeError('Unresolved native attempt retained')
            results.append(fn(folder,ci,arm,m))
            if results[-1]['status'] != 'complete':
                raise RuntimeError('Writer failure retained; separate recovery version required')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','check','boundary','global-prepare','global-score','build','writer-check','write'))
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--arm',choices=variants.ARMS)
    p.add_argument('--conversations',nargs='+',type=int,default=list(range(10)))
    a=p.parse_args();run=a.run_dir.resolve()
    if len(a.conversations)!=len(set(a.conversations)) or any(ci not in range(10) for ci in a.conversations):
        p.error('Unique conversation indices 0..9 required')
    if a.stage not in ('prepare','check') and not a.arm:p.error('stage requires --arm')
    if a.stage=='prepare':result=prepare(run)
    elif a.stage=='check':result={'valid':bool(manifest(run)),'calls':0}
    elif a.stage=='boundary':result=boundary(run,a.arm,a.conversations)
    elif a.stage=='global-prepare':result=global_prepare(run,a.arm)
    elif a.stage=='global-score':result=global_score(run,a.arm,a.conversations)
    elif a.stage=='build':result=build(run,a.arm)
    elif a.stage=='writer-check':result={'valid':bool(writer.manifest(run/a.arm/'writer')),'calls':0}
    else:result=write(run,a.arm,a.conversations)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
