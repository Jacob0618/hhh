"""Fresh-bank three-arm adapter. Scientific selectors are imported unchanged.

prepare/retrieve/check/seal have distinct entry points. Retrieval may request
only QUERY embeddings; score dispatch lives in score_gemini25_ablation_v1.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_evidence_selection_r03 import Reader, save_new, file_sha
from tools.run_sdk20_modules import digest, now, verify
from tools.run_sdk20_judge_retest_v2 import pointer
from tools.run_pilot400_experiments_v2 import publish_json
from tools.mem0_regroup_plain_qa_sources_v1 import cohort, inventory
from tools.summarize_pilot400 import verified_official_metric
from tools.prepare_gemini38_hybrid_v1 import capture_current, validate_native_capture
from tools.run_planned_retrieval_v1 import fact_projection
from tools.prepare_module_stack_v1 import merged_scores
from jev_mem import evidence_exchange_r01 as pool_core
from jev_mem import evidence_selection_r03 as r03
from jev_mem import source_append_r05 as source_core
from jev_mem import source_support_stack_v1 as stack
from jev_mem import r05_refinement_v1 as r05
from jev_mem.data import SOURCE_HASH
from jev_mem.local_bm25_context_v2 import local_bm25_context
from jev_mem.providers import Embeddings
from tools import run_gemini25_ablation_writer_v1 as writer

SCHEMA = 'gemini25-ablation-source-manifest-v1'
SOURCE_SCHEMA = 'gemini25-ablation-selected-inputs-v1'
ARMS = ('baseline', 'r03_original', 'original_multi_source_closure')
PHASES = ('r03', 'support', 'multi_compare', 'closure', 'rank')
R03_SPEC = ROOT/'research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json'
CODE = ('tools/prepare_gemini25_ablation_v1.py', 'tools/score_gemini25_ablation_v1.py',
        'jev_mem/evidence_exchange_r01.py', 'jev_mem/evidence_selection_r03.py',
        'jev_mem/source_append_r05.py', 'jev_mem/source_support_stack_v1.py',
        'jev_mem/r05_refinement_v1.py', 'tools/pilot400_recovery_rank_v1.py',
        'jev_mem/routing.py', 'tools/prepare_gemini38_hybrid_v1.py',
        'jev_mem/sdk20_type_adapter.py', 'tools/run_planned_retrieval_v1.py',
        'tools/score_module_stack_v1.py', 'tools/score_source_append_r05.py',
        'tools/score_evidence_exchange_r01.py', 'tools/prepare_evidence_selection_r03.py',
        'tools/prepare_module_stack_v1.py', 'jev_mem/gemini25_ablation_provider_v1.py',
        'jev_mem/providers.py', 'jev_mem/sdk20_cached_api.py', 'jev_mem/paced_embedding_api.py')
CODE = tuple(dict.fromkeys((*CODE, *writer.CODE,
        'tools/audit_gemini25_ablation_retrieval_v1.py',
        'tools/mem0_regroup_plain_qa_sources_v1.py',
        'jev_mem/qa_embedding_cache_pack_v1.py', 'jev_mem/hybrid_native_capture_v1.py')))
PREDECESSORS = {'r03': (), 'support': (), 'multi_compare': ('r03', 'support'),
                'closure': ('multi_compare',), 'rank': ('closure',)}


def fixed_protocol():
    return {'arms':list(ARMS), 'questions_per_arm':1540, 'selected_count':20,
        'writer':'gemini-2.5-flash', 'embedding':'gemini-embedding-001', 'dimensions':768,
        'jev':'jev-1.13.0', 'pool':pool_core.POLICY, 'r03':r03.POLICY,
        'r03_spec':pointer(R03_SPEC), 'source_support':stack.PROTOCOL, 'r05':r05.PROTOCOL,
        'reuse':'new banks and scores; unchanged scientific functions; no historical QA inheritance',
        'native_search':{'top_k':20,'rerank':False,'threshold':.1},
        'failure':'one uniform recovery after terminal stage; unknown blocks; valid NO never retried'}


def spec():
    if file_sha(R03_SPEC) != r03.PROTOCOL_SHA256:
        raise ValueError('Original R03 predicate protocol changed')
    return json.loads(R03_SPEC.read_text(encoding='utf8'))


def canonical(f):
    return {k: deepcopy(f.get(k)) for k in ('id', 'text', 'observation_date')}


def validate_sdk():
    from mem0.memory import main as sdk
    if file_sha(Path(sdk.__file__)) != file_sha(ROOT/'upstream/mem0/mem0/memory/main.py'):
        raise ValueError('Installed Memory source differs from the frozen local SDK')


def manifest(run):
    validate_sdk()
    run = Path(run).resolve()
    m = verify(run/'run_manifest.json')
    if (m['schema'] != SCHEMA or m['arms'] != list(ARMS)
            or m['code_sha256'] != {p:file_sha(ROOT/p) for p in CODE}
            or m['dataset_sha256'] != SOURCE_HASH or file_sha(ROOT/'data/locomo10.json') != SOURCE_HASH
            or m['r03_spec'] != pointer(R03_SPEC)):
        raise ValueError('Frozen source model/code/dataset/protocol changed')
    qs, dates = cohort()
    if m['cohort_sha256'] != digest(qs) or m['reference_dates'] != dates:
        raise ValueError('Question/gold/category/reference-date cohort changed')
    if m['protocol'] != fixed_protocol() or m['protocol_sha256'] != digest(m['protocol']):
        raise ValueError('Scientific protocol binding changed')
    from tools.audit_gemini25_ablation_banks_v1 import admit_review
    parent_path = Path(m['writer_manifest']['path']).resolve()
    if parent_path.name != 'run_manifest.json' or pointer(parent_path) != m['writer_manifest']:
        raise ValueError('Writer manifest pointer changed')
    if pointer(m['bank_review']['path']) != m['bank_review']:
        raise ValueError('Bank review pointer changed')
    admission = admit_review(parent_path.parent, m['bank_review']['path'], conversations=tuple(range(10)))
    if m['resources'] != admission['manifest']['resources']:
        raise ValueError('Writer resources changed')
    expected = []
    for row in admission['banks']:
        ci = row['conversation']; bank = verify(row['bank']['path'])
        expected.append({'conversation':ci,'bank':row['bank'],
                         'native_source':str(Path(bank['native_dir']).resolve()),
                         'native_copy':str(run/f'conv_{ci}/native'),
                         'native_inventory':row['native_inventory']})
    if m['banks'] != sorted(expected, key=lambda r:r['conversation']) or len(expected) != 10:
        raise ValueError('Exactly ten freshly reviewed session banks required')
    _, metric = verified_official_metric()
    if m['official_metric_provenance'] != metric:
        raise ValueError('Official metric provenance changed')
    return m


def prepare(run, writer_run, review_path):
    validate_sdk()
    from tools import run_gemini25_ablation_writer_v1 as writer
    run, writer_run = Path(run).resolve(), Path(writer_run).resolve()
    if run.exists() or run.parent != ROOT/'runs':
        raise FileExistsError('New direct runs namespace required')
    from tools.audit_gemini25_ablation_banks_v1 import admit_review
    admission = admit_review(writer_run, review_path, conversations=tuple(range(10)))
    parent, gate = admission['manifest'], admission['review']
    admitted = {row['conversation']:row for row in admission['banks']}
    qs, dates = cohort()
    _, metric = verified_official_metric()
    protocol = fixed_protocol()
    spec()
    run.mkdir(parents=True)
    save_new(run/'prepare_intent.json', {'created_utc':now(),'pid':os.getpid(),'model_calls':0})
    banks = []
    for ci in range(10):
        row = admitted[ci]
        bank_ref = row['bank']
        bank = verify(bank_ref['path'])
        origin = Path(bank['native_dir']).resolve()
        inv = row['native_inventory']
        if inventory(origin) != inv:
            raise ValueError('Reviewed original native store changed')
        target = run/f'conv_{ci}/native'
        target.mkdir(parents=True, exist_ok=False)
        for rel, hashed in inv.items():
            src, dst = (origin/rel).resolve(), (target/rel).resolve()
            if not src.is_relative_to(origin) or not dst.is_relative_to(target):
                raise ValueError('Native inventory escapes declared directory')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src,dst)
            if file_sha(dst) != hashed:
                raise ValueError('Native copy differs')
        if inventory(origin) != inv or inventory(target) != inv:
            raise ValueError('Original/copy native inventory changed during preparation')
        banks.append({'conversation':ci,'bank':bank_ref,'native_source':str(origin),
                      'native_copy':str(target),'native_inventory':inv})
    value = {'schema':SCHEMA,'created_utc':now(),'arms':list(ARMS),'protocol':protocol,
        'protocol_sha256':digest(protocol),'writer_manifest':pointer(writer_run/'run_manifest.json'),
        'bank_review':pointer(review_path),'banks':banks,'resources':parent['resources'],
        'dataset_sha256':SOURCE_HASH,'cohort_sha256':digest(qs),'reference_dates':dates,
        'official_metric_provenance':metric,'r03_spec':pointer(R03_SPEC),
        'code_sha256':{p:file_sha(ROOT/p) for p in CODE},'model_calls_during_prepare':0}
    save_new(run/'run_manifest.json',value)
    return {'banks':len(banks),'model_calls':0,'run_dir':str(run)}


def retrieve_worker(run, ci):
    from tools import run_gemini25_ablation_writer_v1 as writer
    from jev_mem.gemini25_ablation_provider_v1 import Gemini25API
    run = Path(run).resolve(); m = manifest(run)
    item = next(x for x in m['banks'] if x['conversation']==ci)
    bank = verify(item['bank']['path'])
    if pointer(item['bank']['path']) != item['bank']:
        raise ValueError('Bank bytes changed')
    folder=run/f'conv_{ci}'
    save_new(folder/'retrieval_intent.json',{'created_utc':now(),'pid':os.getpid(),
        'protocol_sha256':m['protocol_sha256']})
    if inventory(item['native_source']) != item['native_inventory'] or inventory(item['native_copy']) != item['native_inventory']:
        raise ValueError('Native copy/source changed before retrieval')
    api = Gemini25API(folder/'retrieval_api',retries=2,context={'conversation':ci,'phase':'retrieval'})
    qs=cohort()[0][str(ci)]; resource=m['resources']; memory=None
    state={'status':'running','pid':os.getpid(),'conversation':ci,'completed':0,'expected':len(qs),'started_utc':now()}
    publish_json(folder/'retrieval_status.json',state)
    try:
        with local_bm25_context(resource['snapshot'],resource['files_sha256']):
            memory=writer.open_memory(Path(item['native_copy']),api)
            facts=[canonical(f) for f in bank['facts']]
            for q in qs:
                scoped=api.scoped(query_id=q['id']);memory.embedding_model=Embeddings(scoped)
                value=capture_current(memory,q['question'],bank,item['bank'])
                cap=value['capture']; pool=pool_core.build_pool(q['question'],facts,cap['base20_ids'],cap['candidate_ids'])
                save_new(folder/'retrieval'/(q['id']+'.json'),{'schema':SCHEMA,'status':'complete',
                    'conversation':ci,'query_id':q['id'],'question':q['question'], 'bank':item['bank'],
                    'protocol_sha256':m['protocol_sha256'],'capture':value,'pool':pool,'api_events':list(scoped.events)})
                state.update(completed=state['completed']+1,last_query_id=q['id'],updated_utc=now())
                publish_json(folder/'retrieval_status.json',state)
    except Exception as exc:
        state.update(status='needs_review',error_type=type(exc).__name__,error=api._redact(exc)[:1000])
        raise
    finally:
        if memory is not None:
            writer.native.close_memory(memory)
        if state['completed']==len(qs):state['status']='complete'
        state['finished_utc']=now();publish_json(folder/'retrieval_status.json',state)
    if inventory(item['native_source']) != item['native_inventory']:
        raise ValueError('Immutable writer store changed')
    save_new(folder/'post_retrieval.json',{'status':'complete','protocol_sha256':m['protocol_sha256'],
        'source_unchanged':True,'native_inventory':inventory(item['native_copy'])})
    return state


def retrieve(run, conversations=tuple(range(10)), workers=8):
    run=Path(run).resolve();manifest(run)
    if not 1<=workers<=8 or len(set(conversations))!=len(conversations):raise ValueError('Bounded distinct conversations required')
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(retrieve_worker,run,ci) for ci in conversations]):rows.append(f.result())
    save_new(run/'retrieval_outcome.json',{'created_utc':now(),'conversations':rows,
        'status':'complete' if len(rows)==10 and all(r['status']=='complete' for r in rows) else 'partial'})
    return rows


def inputs(reader, run):
    run=Path(run).resolve()
    cache = getattr(reader, '_gemini25_inputs', {})
    if str(run) in cache:
        return cache[str(run)]
    m=manifest(run);qs,dates=cohort();sources={};queries=[]
    from tools.audit_gemini25_ablation_retrieval_v1 import audit_retrieval
    for item in m['banks']:
        ci=item['conversation'];bank=reader.bound(item['bank']);inp=reader.bound(bank['group_inputs'])
        audit_retrieval(reader,run,m,item,bank,qs[str(ci)])
        turns=source_core.source_turns(inp,bank['group_inputs']);rawfacts={f['id']:f for f in bank['facts']}
        sources[ci]={'bank':item['bank'],'rawfacts':rawfacts,'turns':turns}
        post=reader.read(run/f'conv_{ci}/post_retrieval.json')
        if post.get('status')!='complete' or post.get('source_unchanged') is not True:
            raise ValueError('Retrieval has not completed')
        for q in qs[str(ci)]:
            ref=reader.pointer(run/f'conv_{ci}/retrieval'/(q['id']+'.json'));r=reader.bound(ref)
            if (r['query_id'],r['question'],r['conversation'],r['bank'],r['status'],r['protocol_sha256']) != (q['id'],q['question'],ci,item['bank'],'complete',m['protocol_sha256']):
                raise ValueError('Native query/bank identity differs')
            cap=r['capture']['capture'];validate_native_capture(cap,fact_projection(bank),bank['user_id'])
            facts=[canonical(f) for f in bank['facts']]
            expected=pool_core.build_pool(q['question'],facts,cap['base20_ids'],cap['candidate_ids'])
            if r['pool']!=expected:raise ValueError('Original R03 candidate constructor does not replay')
            records=[canonical(rawfacts[fid]) for fid in expected['pool_ids']]
            queries.append({'conversation':ci,'query_id':q['id'],'question':q['question'],'reference':q['reference'],
                'category':q['category'],'reference_date':dates[str(ci)],'bank':item['bank'],'retrieval':ref,'records':records})
    cache[str(run)] = (m,sources,queries)
    reader._gemini25_inputs = cache
    return m,sources,queries


def request_value(sig,rid,q,ci,phase,psha):
    from tools.score_gemini25_ablation_v1 import validate_request
    validate_request(sig,allow_oversize=True)
    size=len(pool_core.serialized_signature(sig))
    return {'schema':'gemini25-ablation-request-v1','request_id':rid,'query_id':q,'conversation':ci,
        'phase':phase,'protocol_sha256':psha,'signature':sig,'request_hash':pool_core.request_hash(sig),
        'signature_utf8_bytes':size,'dispatch_eligible':size<=131072,'skip_reason':None if size<=131072 else 'signature_oversize'}


def requests(plan,prefix,q,ci,phase,psha):
    return [request_value(r['signature'],f'{prefix}_b{i}',q,ci,phase,psha) for i,r in enumerate(plan['requests'])]


def units(reader,run,phase):
    pm=reader.read(Path(run)/'phases'/phase/'manifest.json')
    return [(ref,reader.bound(ref)) for ref in pm['units']]


def phase_results(reader,run,phase,*,allow_recovery=True):
    from tools.score_gemini25_ablation_v1 import phase_results as results
    return results(reader,Path(run),phase,allow_recovery=allow_recovery)


def phase_values(reader,run,phase):
    m,sources,qs=inputs(reader,run);psha=m['protocol_sha256']
    if phase=='r03':
        for q in qs:
            sig=r03.build_request(q['question'],q['reference_date'],q['records'],spec())
            req=request_value(sig,q['query_id'],q['query_id'],q['conversation'],phase,psha)
            yield {'unit_id':q['query_id'],'conversation':q['conversation'],'records':q['records'],'request_ids':[req['request_id']]},[req]
        return
    if phase=='support':
        for ci,s in sources.items():
            for fid,f in s['rawfacts'].items():
                plan=stack.build_support_plan(canonical(f),f.get('input_scope'),s['turns'],conversation=ci)
                uid=f'c{ci}_fact_{fid}';reqs=requests(plan,uid,uid,ci,phase,psha)
                yield {'unit_id':uid,'conversation':ci,'plan':plan,'request_ids':[r['request_id'] for r in reqs]},reqs
        return
    if phase=='multi_compare':
        scores=phase_results(reader,run,'r03');support_scores=phase_results(reader,run,'support');supports={ci:{} for ci in range(10)}
        for _,u in units(reader,run,'support'):
            values,status=merged_scores(u['request_ids'],support_scores)
            supports[u['conversation']][u['plan']['fact']['id']]={'plan':u['plan'],'scores':values,'status':status}
        for q in qs:
            ci,qid=q['conversation'],q['query_id'];res=scores[qid]
            decision=r03.choose_jev(q['records'],res['scores'],score_status=res['status'])
            facts=r03.selected_facts(q['records'],decision['pool_selected_indices']);s=sources[ci]
            scopes=[s['rawfacts'][f['id']].get('input_scope') for f in facts]
            plan=r05.build_candidate_variants(q['question'],facts,scopes,s['turns'],conversation=ci,
                support_results=supports[ci],reference_date=q['reference_date'])['multi_plan']
            reqs=requests(plan,qid,qid,ci,phase,psha)
            yield {'unit_id':qid,'conversation':ci,'records':facts,'r03_decision':decision,'plan':plan,'request_ids':[r['request_id'] for r in reqs]},reqs
        return
    before='multi_compare' if phase=='closure' else 'closure'
    values=phase_results(reader,run,before);previous={u['unit_id']:(ref,u) for ref,u in units(reader,run,before)}
    for q in qs:
        ci,qid=q['conversation'],q['query_id'];ref,u=previous[qid];facts=u['records'];reqs=[]
        if phase=='closure':
            scores,status=merged_scores(u['request_ids'],values)
            loc=r05.resolve_multi(u['plan'],scores,score_status=status)['locator'];plans=[];rids=[]
            for ix,candidate in enumerate(loc['candidates']):
                scope=sources[ci]['rawfacts'][facts[candidate['origin']['anchor_slot']]['id']].get('input_scope')
                p=stack.build_closure_plan(q['question'],q['reference_date'],facts,candidate,scope,sources[ci]['turns'],conversation=ci)
                these=requests(p,f'{qid}_c{ix}',qid,ci,phase,psha);reqs+=these;plans.append(p);rids.append([r['request_id'] for r in these])
            out={'unit_id':qid,'conversation':ci,'records':facts,'locator':loc,'plans':plans,'request_ids':rids,'prior_unit':ref}
        elif phase=='rank':
            resolved=[]
            for plan,rids in zip(u['plans'],u['request_ids']):
                scores,status=merged_scores(rids,values);resolved.append(stack.resolve_closure(plan,scores,score_status=status))
            loc=stack.apply_closures(u['locator'],resolved,facts,closure_plans=u['plans'])
            sig=stack.build_rank_request(q['question'],q['reference_date'],facts,loc['candidates'])
            if sig is not None:reqs=[request_value(sig,qid,qid,ci,phase,psha)]
            out={'unit_id':qid,'conversation':ci,'records':facts,'locator':loc,'request_ids':[r['request_id'] for r in reqs],'prior_unit':ref}
        else:raise ValueError('Unknown phase')
        yield out,reqs


def prepare_phase(run,phase):
    if phase not in PHASES:raise ValueError('Unknown phase')
    run=Path(run).resolve();m=manifest(run);dest=run/'phases'/phase
    if dest.exists():raise FileExistsError('Immutable phase')
    reader=Reader();checked=set()
    for parent in PREDECESSORS[phase]:
        validate_phase(run,parent,reader=reader,checked=checked)
    dest.mkdir(parents=True);us=[];rs=[]
    for u,reqs in phase_values(reader,run,phase):
        us.append(save_new(dest/'units'/(u['unit_id']+'.json'),u))
        rs.extend(save_new(dest/'requests'/(r['request_id']+'.json'),r) for r in reqs)
    save_new(dest/'manifest.json',{'schema':'gemini25-ablation-phase-v1','phase':phase,'created_utc':now(),
        'protocol_sha256':m['protocol_sha256'],'parent_manifest':pointer(run/'run_manifest.json'),'units':us,'requests':rs})
    return {'units':len(us),'requests':len(rs),'model_calls':0}


def validate_phase(run,phase,*,reader=None,checked=None):
    if phase not in PHASES:raise ValueError('Unknown phase')
    run=Path(run);reader=reader or Reader();checked=set() if checked is None else checked
    if phase in checked:
        pm=reader.read(run/'phases'/phase/'manifest.json')
        return pm,pm['requests']
    for parent in PREDECESSORS[phase]:
        validate_phase(run,parent,reader=reader,checked=checked)
    m=inputs(reader,run)[0];pm=reader.read(run/'phases'/phase/'manifest.json')
    if pm['schema']!='gemini25-ablation-phase-v1' or pm['phase']!=phase or pm['protocol_sha256']!=m['protocol_sha256'] or pm['parent_manifest']!=pointer(run/'run_manifest.json'):
        raise ValueError('Phase manifest binding changed')
    expected_units=[];expected_requests=[]
    for u,reqs in phase_values(reader,run,phase):
        p=run/'phases'/phase/'units'/(u['unit_id']+'.json');ref=reader.pointer(p)
        if {k:v for k,v in reader.bound(ref).items() if k!='content_sha256'} != u:raise ValueError('Scientific unit does not replay')
        expected_units.append(ref)
        for r in reqs:
            p=run/'phases'/phase/'requests'/(r['request_id']+'.json');ref=reader.pointer(p)
            if {k:v for k,v in reader.bound(ref).items() if k!='content_sha256'}!=r:raise ValueError('Scientific request does not replay')
            expected_requests.append(ref)
    if expected_units!=pm['units'] or expected_requests!=pm['requests']:raise ValueError('Phase input coverage differs')
    for folder,refs in (('units',expected_units),('requests',expected_requests)):
        if {Path(r['path']).resolve() for r in refs}!={p.resolve() for p in (run/'phases'/phase/folder).glob('*.json')}:
            raise ValueError('Unmanifested phase inputs')
    checked.add(phase)
    return pm,pm['requests']


def selections(reader,run):
    m,_,qs=inputs(reader,run);r03rows=phase_results(reader,run,'r03');rankrows=phase_results(reader,run,'rank')
    ranks={u['unit_id']:(ref,u) for ref,u in units(reader,run,'rank')}
    for q in qs:
        qid=q['query_id'];rr=r03rows[qid];d=r03.choose_jev(q['records'],rr['scores'],score_status=rr['status'])
        chosen=r03.selected_facts(q['records'],d['pool_selected_indices']);rank_ref,u=ranks[qid]
        if chosen!=u['records']:raise ValueError('Source recovery uses different R03 facts')
        sc,st=merged_scores(u['request_ids'],rankrows)
        # No ranking request is needed when <=2 packets remain; original chooser handles it.
        if not u['request_ids']:sc,st=None,'not_dispatched'
        rd=stack.choose_rank(chosen,u['locator']['candidates'],sc,score_status=st)
        full=stack.selected_facts(chosen,u['locator']['candidates'],rd)
        for arm,facts in zip(ARMS,(q['records'][:20],chosen,full)):
            value={'schema':'gemini25-ablation-selection-v1','conversation':q['conversation'],'query_id':qid,
                'method':arm,'bank':q['bank'],'retrieval':q['retrieval'],'selected_ids':[f['id'] for f in facts],
                'selected_facts':facts,'r03_decision':d if arm!='baseline' else None,
                'rank_decision':rd if arm==ARMS[2] else None,'rank_unit':rank_ref if arm==ARMS[2] else None,
                'protocol_sha256':m['protocol_sha256']}
            yield q,arm,value


def seal(run):
    run=Path(run);m=manifest(run)
    reader=Reader();checked=set();jobs=[]
    for phase in PHASES:validate_phase(run,phase,reader=reader,checked=checked)
    for q,arm,selection in selections(reader,run):
        ref=save_new(run/'selections'/arm/(q['query_id']+'.json'),selection)
        jobs.append({'job_id':arm+'/'+q['query_id'],'method':arm,'status':'ready',
            **{k:q[k] for k in ('conversation','query_id','question','reference','category','reference_date','bank')},
            'selected_ids':selection['selected_ids'],'selected_facts':selection['selected_facts'],
            'selection':ref,'protocol_sha256':m['protocol_sha256'],'lineage':{'bank':q['bank'],'selection':ref}})
    value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),'questions_per_arm':1540,
        'jobs':jobs,'parent_run':str(run.resolve()),'selector_manifest':pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'],'official_metric_provenance':m['official_metric_provenance'],
        'audited_bindings':list(reader.refs.values()),'code_sha256':m['code_sha256']}
    return save_new(run/'selected_inputs.json',value)


def validate_source(path):
    reader=Reader();value=reader.read(path);run=Path(value['parent_run']);m=manifest(run)
    if value['schema']!=SOURCE_SCHEMA or value['status']!='sealed' or value['arms']!=list(ARMS) or value['selector_manifest']!=pointer(run/'run_manifest.json'):
        raise ValueError('Sealed three-arm source required')
    for ref in value['audited_bindings']:reader.raw_pointer(ref['path'],ref=ref)
    checked=set()
    for phase in PHASES:validate_phase(run,phase,reader=reader,checked=checked)
    expected=[]
    for q,arm,rec in selections(reader,run):
        p=run/'selections'/arm/(q['query_id']+'.json');sr=reader.pointer(p)
        if {k:v for k,v in reader.bound(sr).items() if k!='content_sha256'}!=rec:raise ValueError('Final selection does not replay')
        expected.append({'job_id':arm+'/'+q['query_id'],'method':arm,'status':'ready',
            **{k:q[k] for k in ('conversation','query_id','question','reference','category','reference_date','bank')},
            'selected_ids':rec['selected_ids'],'selected_facts':rec['selected_facts'],'selection':sr,
            'protocol_sha256':m['protocol_sha256'],'lineage':{'bank':q['bank'],'selection':sr}})
    if value['jobs']!=expected or len(expected)!=4620 or Counter(j['method'] for j in expected)!={a:1540 for a in ARMS}:
        raise ValueError('Full original three-arm cohort or selected facts changed')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--run-dir',type=Path,required=True);s.add_argument('--writer-run',type=Path,required=True);s.add_argument('--review',type=Path,required=True)
    for command in ('retrieve','seal','check'):
        s=sub.add_parser(command);s.add_argument('--run-dir',type=Path,required=True)
    for command in ('prepare-phase','check-phase'):
        s=sub.add_parser(command);s.add_argument('--run-dir',type=Path,required=True);s.add_argument('--phase',choices=PHASES,required=True)
    a=p.parse_args()
    if a.command=='prepare':out=prepare(a.run_dir,a.writer_run,a.review)
    elif a.command=='retrieve':out=retrieve(a.run_dir)
    elif a.command=='seal':out=seal(a.run_dir)
    elif a.command=='prepare-phase':out=prepare_phase(a.run_dir,a.phase)
    elif a.command=='check-phase':pm,rs=validate_phase(a.run_dir,a.phase);out={'requests':len(rs),'model_calls':0}
    else:out={'schema':manifest(a.run_dir)['schema'],'model_calls':0}
    print(json.dumps(out,ensure_ascii=False))
