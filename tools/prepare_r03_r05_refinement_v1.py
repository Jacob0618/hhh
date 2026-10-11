"""Offline, versioned preparation of one finite R03/R05 refinement comparison.

Uses the existing Mem0 bank/source loader and the completed module-stack
cohort. Archived pools are input records, never reconstructed HTTP caches.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_module_stack_v1 as prior
from jev_mem import r03_r05_request_envelope_v1 as wire
from jev_mem.evidence_exchange_r01 import digest,request_hash,serialized_signature

Reader,save_new,file_sha,now=prior.Reader,prior.save_new,prior.file_sha,prior.now
ARMS=('baseline','r03_original','r03_refined','original_source_closure',
    'refined_source_closure','original_multi_source_closure','refined_multi_source_closure')
SOURCE_ARMS=ARMS[3:]
PHASES=('precision','complement','sets','multi_compare','closure','rank')
SCHEMA='r03-r05-refinement-selector-manifest-v1'
SOURCE_SCHEMA='r03-r05-refinement-selected-inputs-v1'
REQUEST_SCHEMA='r03-r05-refinement-request-v1'
SPEC=ROOT/'research/r03_r05_refinement_2026-10-06/protocol_v1.json'
PRIOR_RUN=ROOT/'runs/module_stack_v1_final_20261005'
PRIOR_QA=ROOT/'runs/module_stack_qa_v1_20261005'
PRIOR_SUMMARY=ROOT/'research/module_stack_2026-10-05/live_v1/final_summary_v1.json'
PRIOR_SUMMARY_SHA='ee1039384934b84b6da183dbe4c13904ca9355fdcdf3d2b633c8813b80123686'
PROOF_KEYS=prior.PROOF_KEYS
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/r03_refinement_v1.py','jev_mem/r05_refinement_v1.py',
    'jev_mem/r03_r05_request_envelope_v1.py','tools/prepare_r03_r05_refinement_v1.py',
    'tools/score_r03_r05_refinement_v1.py','tools/r03_r05_reuse_proofs_v1.py',
    'tools/run_r03_r05_refinement_qa_v1.py','tools/summarize_r03_r05_refinement_v1.py')))
_PHASE_CACHE={}
_SOURCE_CACHE={}
_SUPPORT_CACHE=None


def codes():return {p:file_sha(ROOT/p) for p in CODE}
def plain(value):return {k:v for k,v in value.items() if k!='content_sha256'}


def _prior_refs(reader):
    refs={'source_manifest':reader.pointer(PRIOR_RUN/'selected_inputs.json'),
        'run_manifest':reader.raw_pointer(PRIOR_QA/'run_manifest.json'),
        'run_outcome':reader.raw_pointer(PRIOR_QA/'run_outcome.json'),
        'summary':reader.raw_pointer(PRIOR_SUMMARY)}
    if refs['summary']['sha256']!=PRIOR_SUMMARY_SHA:raise ValueError('Pinned completed summary changed')
    return refs


def _pools(reader,original_jobs,sources,imported):
    from tools.module_stack_archive_v1 import ARCHIVE_SHA256
    reader.raw_pointer(prior.ARCHIVE,ref={'path':str(prior.ARCHIVE.resolve()),'sha256':ARCHIVE_SHA256})
    expected={j['query_id']:j for j in original_jobs};out={}
    with zipfile.ZipFile(prior.ARCHIVE) as z:
        for line in z.read('shared/original_candidate_pools.jsonl').splitlines():
            row=json.loads(line);q=row['query_id'];ci=row['conversation']
            if q not in expected or q in out or ci!=expected[q]['conversation']:raise ValueError('Pool identity differs')
            ids=row['pool']['pool_ids']
            if len(ids)!=80 or len(set(ids))!=80 or ids[:20]!=expected[q]['selected_ids']:
                raise ValueError('Original 80 pool or protected prefix changed')
            records=[prior.canonical_fact(sources[ci]['rawfacts'][fid]) for fid in ids]
            oj=imported['arms']['pool_old_jev'][q]['selected_facts']
            if len(oj)!=20 or any(f not in records for f in oj):raise ValueError('Original R03 is outside original pool')
            out[q]={'conversation':ci,'records':records,'original_r03':deepcopy(oj),
                'archive_pool_line_sha256':__import__('hashlib').sha256(line).hexdigest()}
    if set(out)!=set(expected):raise ValueError('1540 fixed pools required')
    return out


def _spec(reader):
    from jev_mem import r03_refinement_v1 as r03
    from jev_mem import r05_refinement_v1 as r05
    spec=reader.read(SPEC,sealed=False)
    if (spec['arms']!=list(ARMS) or spec['questions_per_arm']!=1540
        or spec['r03']!=r03.PROTOCOL or spec['r05']!=r05.PROTOCOL):
        raise ValueError('Research protocol differs from reviewed pure planners')
    return spec


def prepare(run_dir):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.name.startswith('r03_r05_refinement_v1') or run.exists():
        raise FileExistsError('Fresh direct refinement namespace required')
    reader=Reader();spec=_spec(reader)
    pm,pb,sources=prior.base(reader,PRIOR_RUN,replay_archive=True)
    pools=_pools(reader,pb['original_jobs'],sources,pb['imported'])
    p={'schema':SCHEMA,'arms':list(ARMS),'questions_per_arm':1540,'phases':list(PHASES),
        'research_protocol':reader.raw_pointer(SPEC),'prior_selector_manifest':reader.pointer(PRIOR_RUN/'run_manifest.json'),
        'prior_base_inputs':pm['base_inputs'],'prior_qa_reuse':_prior_refs(reader),
        'code_sha256':codes(),**{k:pm['protocol'][k] for k in PROOF_KEYS},
        'r03':spec['r03'],'r05':spec['r05'],'bank_refs':pm['protocol']['bank_refs'],
        'group_input_refs':pm['protocol']['group_input_refs'],
        'known_failure_recovery':'one_uniform_phase_after_terminal_then_explicit_fallback'}
    psha=digest(p)
    run.mkdir(parents=True,exist_ok=False)
    pref=save_new(run/'pool_inputs.json',{'schema':'r03-r05-pool-inputs-v1','protocol_sha256':psha,'queries':pools})
    ref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':p,'protocol_sha256':psha,'pool_inputs':pref,'model_calls_during_prepare':0})
    return {'manifest':ref,'questions':len(pools),'arms':len(ARMS),'model_calls':0}


def base(reader,run,*,replay_archive=False):
    m=reader.read(run/'run_manifest.json');p=m['protocol'];inputs=reader.bound(m['pool_inputs'])
    spec=_spec(reader)
    if (m['schema']!=SCHEMA or m['status']!='prepared' or m['protocol_sha256']!=digest(p)
        or p['arms']!=list(ARMS) or p['phases']!=list(PHASES) or p['code_sha256']!=codes()
        or inputs['protocol_sha256']!=m['protocol_sha256'] or p['research_protocol']!=reader.raw_pointer(SPEC)
        or p['r03']!=spec['r03'] or p['r05']!=spec['r05']):raise ValueError('Frozen refinement contract changed')
    pm,pb,sources=prior.base(reader,PRIOR_RUN,replay_archive=replay_archive)
    if (p['prior_selector_manifest']!=reader.pointer(PRIOR_RUN/'run_manifest.json')
        or p['prior_base_inputs']!=pm['base_inputs'] or p['prior_qa_reuse']!=_prior_refs(reader)
        or any(p[k]!=pm['protocol'][k] for k in PROOF_KEYS)
        or p['bank_refs']!=pm['protocol']['bank_refs'] or p['group_input_refs']!=pm['protocol']['group_input_refs']):
        raise ValueError('Prior fixed bank/source/request lineage changed')
    if replay_archive and inputs['queries']!=_pools(reader,pb['original_jobs'],sources,pb['imported']):
        raise ValueError('Archived source pools do not replay')
    return m,{'original_jobs':pb['original_jobs'],'pools':inputs['queries']},sources


def request_value(sig,rid,q,ci,phase,psha):
    wire.validate_request(sig,allow_oversize=True);size=len(serialized_signature(sig))
    return {'schema':REQUEST_SCHEMA,'request_id':rid,'query_id':q,'conversation':ci,'phase':phase,
        'protocol_sha256':psha,'signature':sig,'request_hash':request_hash(sig),
        'signature_utf8_bytes':size,'dispatch_eligible':size<=131072,
        'skip_reason':None if size<=131072 else 'signature_oversize'}


def units(reader,run,phase):
    return [(ref,reader.bound(ref)) for ref in reader.read(run/'phases'/phase/'manifest.json')['units']]


def phase_results(reader,run,phase,*,allow_recovery=True):
    from tools import score_r03_r05_refinement_v1 as dispatch
    from tools.check_experiment_health import shared_read
    pm=reader.read(run/'phases'/phase/'manifest.json');requests=pm['requests']
    def producer_results(producer,refs):
        outcome=reader.read(producer/'outcome.json');intent=reader.read(producer/'execution_intent.json',sealed=False)
        if (outcome['schema']!=dispatch.SCHEMA or outcome['status']!='terminal' or outcome.get('local_errors')
            or outcome['expected_requests']!=len(refs) or outcome['terminal_requests']!=len(refs)
            or intent['requests']!=refs or intent['transport_retries']!=2
            or intent['dispatch_code_sha256']!=file_sha(ROOT/'tools/score_r03_r05_refinement_v1.py')
            or intent['manifest_sha256']!=reader.pointer(run/'phases'/phase/'manifest.json')['sha256']):
            raise ValueError('Complete phase terminal and producer proof required')
        result={};allowed=set();eligible=set()
        for ref in refs:
            req=reader.bound(ref);rid=req['request_id'];rp=producer/'responses'/(rid+'.json');rref=reader.pointer(rp)
            row=dispatch.verify_result(reader,rref,ref,producer,2)
            result[rid]={**deepcopy(row),'result_ref':rref}
            if req['dispatch_eligible']:
                eligible.add(req['request_hash']);allowed.add((req['request_hash'],digest(dispatch.context(req))))
        if {Path(r['result_ref']['path']).resolve() for r in result.values()}!={p.resolve() for p in (producer/'responses').glob('*.json')}:
            raise ValueError('Foreign or missing terminal response')
        if {p.stem for p in (producer/'request_intents').glob('*.json')}!=eligible:raise ValueError('Foreign or missing intent')
        rawkeys={r['request_hash'] for r in result.values() if r.get('raw')}
        if {p.stem for p in (producer/'api_cache').glob('*.json')}!=rawkeys:raise ValueError('Foreign or missing raw')
        for ip in (producer/'request_intents').glob('*.json'):
            iv=reader.read(ip,sealed=False)
            if (ip.stem,digest(iv.get('context'))) not in allowed or request_hash(iv['signature'])!=ip.stem:
                raise ValueError('Unexpected intent identity')
        lp=producer/'api_calls.jsonl'
        if lp.exists():
            data=shared_read(lp);reader.raw_pointer(lp)
            if data and not data.endswith(b'\n'):raise ValueError('Partial API ledger')
            for line in data.splitlines():
                event=json.loads(line)
                if ((event.get('request_hash'),digest(event.get('context'))) not in allowed
                    or (event.get('stage'),event.get('provider'),event.get('model'),event.get('operation'))
                    !=(dispatch.STAGE,'jev','jev-1.13.0','systemone')):
                    raise ValueError('Unexpected ledger event')
        return result
    rows=producer_results(run/'phases'/phase/'firstpass',requests)
    recovery=run/'phases'/phase/'recovery_v1'
    failed=[ref for ref in requests if rows[reader.bound(ref)['request_id']]['status']=='failed']
    if allow_recovery and failed and not recovery.exists():raise ValueError('Uniform failure recovery must precede fallback')
    if allow_recovery and recovery.exists():
        if not failed:raise ValueError('Unnecessary recovery producer')
        for rid,row in producer_results(recovery,failed).items():
            rows[rid]={**row,'firstpass_result_ref':rows[rid]['result_ref']}
    return rows


merged_scores=prior.merged_scores


def _request_rows(plan,q,ci,phase,psha,*,prefix=None):
    return [request_value(r['signature'],f'{prefix or q}_b{i}',q,ci,phase,psha)
        for i,r in enumerate(plan['requests'])]


def _resolved_r03(reader,run):
    from jev_mem import r03_refinement_v1 as r03
    results=phase_results(reader,run,'sets');out={}
    for ref,u in units(reader,run,'sets'):
        scores,status=merged_scores(u['request_ids'],results)
        decision=r03.choose_final(u['plan'],scores,score_status=status)
        out[u['unit_id']]={'decision':decision,'records':r03.selected_facts(u['plan']['base']['pool'],decision),
            'set_unit':ref,'score_refs':[results[rid]['result_ref'] for rid in u['request_ids']]}
    return out


def phase_values(reader,run,phase):
    from jev_mem import r03_refinement_v1 as r03
    m,b,sources=base(reader,run);psha=m['protocol_sha256']
    jobs=b['original_jobs'];pools=b['pools']
    if phase=='precision':
        for j in jobs:
            q,ci=j['query_id'],j['conversation'];p=pools[q]
            plan=r03.build_precision_plan(j['question'],j['reference_date'],p['records'],p['original_r03'])
            reqs=_request_rows(plan,q,ci,phase,psha)
            yield {'unit_id':q,'conversation':ci,'plan':plan,'request_ids':[r['request_id'] for r in reqs]},reqs
    elif phase in ('complement','sets'):
        previous='precision' if phase=='complement' else 'complement'
        results=phase_results(reader,run,previous)
        for ref,u in units(reader,run,previous):
            q,ci=u['unit_id'],u['conversation'];scores,status=merged_scores(u['request_ids'],results)
            if phase=='complement':
                resolved=r03.resolve_precision(u['plan'],scores,score_status=status)
                plan=r03.build_complement_plan(u['plan'],resolved)
            else:
                resolved=r03.resolve_complement(u['plan'],scores,score_status=status)
                plan=r03.build_set_plan(u['plan'],resolved)
            reqs=_request_rows(plan,q,ci,phase,psha)
            yield {'unit_id':q,'conversation':ci,'plan':plan,'prior_unit':ref,
                'request_ids':[r['request_id'] for r in reqs]},reqs
    else:
        yield from source_phase_values(reader,run,phase,m,b,sources)


def phase_input_inventory(run,phase):
    files=[p for d in ('units','requests') for p in (run/'phases'/phase/d).glob('*.json')]
    for previous in PHASES[:PHASES.index(phase)]:
        files.extend(p for p in (run/'phases'/previous).rglob('*') if p.is_file())
    return sorted([str(p.resolve()) for p in files]+support_inventory())


def support_inventory():
    return sorted(str(p.resolve()) for p in (PRIOR_RUN/'phases/support').rglob('*') if p.is_file())


def prepare_phase(run_dir,phase):
    run=Path(run_dir).resolve()
    if phase not in PHASES:raise ValueError('Unknown frozen phase')
    if PHASES.index(phase):validate_phase(run,PHASES[PHASES.index(phase)-1])
    reader=Reader();m,_,_=base(reader,run)
    destination=run/'phases'/phase
    if destination.exists():raise FileExistsError('Prepared phase is immutable')
    destination.mkdir(parents=True,exist_ok=False)
    urefs=[];rrefs=[];counts=Counter()
    for u,reqs in phase_values(reader,run,phase):
        urefs.append(save_new(destination/'units'/(u['unit_id']+'.json'),u))
        for r in reqs:
            rrefs.append(save_new(destination/'requests'/(r['request_id']+'.json'),r))
            counts['dispatch_eligible']+=r['dispatch_eligible']
            counts['noul_questions']+=len(r['signature']['body']['questions'])
    ref=save_new(destination/'manifest.json',{'schema':'r03-r05-refinement-phase-manifest-v1','phase':phase,
        'protocol_sha256':m['protocol_sha256'],'parent_manifest':reader.pointer(run/'run_manifest.json'),
        'units':urefs,'requests':rrefs,'counts':dict(counts),'created_utc':now(),'model_calls':0})
    return {'manifest':ref,'units':len(urefs),'requests':len(rrefs),'counts':dict(counts),'model_calls':0}


def validate_phase(run_dir,phase):
    run=Path(run_dir).resolve()
    if phase not in PHASES:raise ValueError('Unknown frozen phase')
    key=(str(run),phase);cached=_PHASE_CACHE.get(key)
    if (cached and cached['inventory']==phase_input_inventory(run,phase)
        and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s
            for p,s in cached['stats'].items())):
        return deepcopy(cached['manifest']),deepcopy(cached['requests'])
    if PHASES.index(phase):validate_phase(run,PHASES[PHASES.index(phase)-1])
    reader=Reader();m,_,_=base(reader,run)
    pm=reader.read(run/'phases'/phase/'manifest.json')
    if (pm['schema']!='r03-r05-refinement-phase-manifest-v1' or pm['phase']!=phase
        or pm['protocol_sha256']!=m['protocol_sha256'] or pm['parent_manifest']!=reader.pointer(run/'run_manifest.json')):
        raise ValueError('Phase manifest binding differs')
    urefs=[];rrefs=[];counts=Counter()
    for u,reqs in phase_values(reader,run,phase):
        up=run/'phases'/phase/'units'/(u['unit_id']+'.json')
        if plain(reader.read(up))!=u:raise ValueError('Gold-free source plan does not replay: '+u['unit_id'])
        urefs.append(reader.pointer(up))
        for r in reqs:
            rp=run/'phases'/phase/'requests'/(r['request_id']+'.json')
            if plain(reader.read(rp))!=r:raise ValueError('Exact scientific request does not replay')
            rrefs.append(reader.pointer(rp));counts['dispatch_eligible']+=r['dispatch_eligible']
            counts['noul_questions']+=len(r['signature']['body']['questions'])
    if pm['units']!=urefs or pm['requests']!=rrefs or pm['counts']!=dict(counts):raise ValueError('Phase cohort/order/count differs')
    for d,refs in (('units',urefs),('requests',rrefs)):
        if {Path(r['path']).resolve() for r in refs}!={p.resolve() for p in (run/'phases'/phase/d).glob('*.json')}:
            raise ValueError('Unexpected phase file inventory')
    paths=set(reader.refs)|{str(ROOT/p) for p in CODE}|set(phase_input_inventory(run,phase))
    _PHASE_CACHE[key]={'manifest':deepcopy(pm),'requests':deepcopy(rrefs),'inventory':phase_input_inventory(run,phase),
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}}
    return pm,rrefs


def support_results(reader):
    """Reuse preserved, validated partial-support scores; no recreated cache."""
    global _SUPPORT_CACHE
    if _SUPPORT_CACHE and _SUPPORT_CACHE['inventory']==support_inventory() and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s
        for p,s in _SUPPORT_CACHE['stats'].items()):
        for p,ref in _SUPPORT_CACHE['refs'].items():reader.refs[p]=ref
        return deepcopy(_SUPPORT_CACHE['value'])
    rows=prior.phase_results(reader,PRIOR_RUN,'support');out={ci:{} for ci in range(10)}
    for ref,u in prior.units(reader,PRIOR_RUN,'support'):
        scores,status=merged_scores(u['request_ids'],rows)
        out[u['conversation']][u['plan']['fact']['id']]={'plan':u['plan'],'scores':scores,'status':status,
            'unit_ref':ref,'result_refs':[rows[rid]['result_ref'] for rid in u['request_ids']]}
    _SUPPORT_CACHE={'value':deepcopy(out),'refs':deepcopy(reader.refs),'inventory':support_inventory(),
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in reader.refs}}
    return out


def source_phase_values(reader,run,phase,m,b,sources):
    from jev_mem import r05_refinement_v1 as core
    psha=m['protocol_sha256'];jobs=b['original_jobs']
    if phase=='multi_compare':
        refined=_resolved_r03(reader,run);supports=support_results(reader)
        for j in jobs:
            q,ci=j['query_id'],j['conversation'];s=sources[ci]
            records={'original':b['pools'][q]['original_r03'],'refined':refined[q]['records']}
            plans={};locators={};ids={};reqs=[]
            for key,facts in records.items():
                scopes=[s['rawfacts'][f['id']].get('input_scope') for f in facts]
                scores={f['id']:{k:supports[ci][f['id']][k] for k in ('plan','scores','status')} for f in facts}
                built=core.build_candidate_variants(j['question'],facts,scopes,s['turns'],conversation=ci,
                    support_results=scores,reference_date=j['reference_date'])
                plans[key]=built['multi_plan'];locators[key]=built['original_locator']
                these=_request_rows(plans[key],q,ci,phase,psha,prefix=f'{q}_{key}')
                reqs.extend(these);ids[key]=[r['request_id'] for r in these]
            yield {'unit_id':q,'conversation':ci,'records':records,'plans':plans,'locators':locators,
                'request_ids':ids,'r03_lineage':refined[q]},reqs
    elif phase=='closure':
        results=phase_results(reader,run,'multi_compare')
        byq={u['unit_id']:(ref,u) for ref,u in units(reader,run,'multi_compare')}
        for j in jobs:
            q,ci=j['query_id'],j['conversation'];s=sources[ci];prior_ref,u=byq[q]
            resolved={}
            for key in ('original','refined'):
                scores,status=merged_scores(u['request_ids'][key],results)
                resolved[key]=core.resolve_multi(u['plans'][key],scores,score_status=status)
            locators={};plans={};ids={};reqs=[]
            for arm,key,multi in source_arm_specs():
                facts=u['records'][key]
                loc=resolved[key]['locator'] if multi else u['locators'][key]
                locators[arm]=loc;plans[arm]=[];ids[arm]=[]
                for ix,c in enumerate(loc['candidates']):
                    scope=s['rawfacts'][facts[c['origin']['anchor_slot']]['id']].get('input_scope')
                    plan=core.build_closure_plan(j['question'],j['reference_date'],facts,c,scope,s['turns'],conversation=ci)
                    plans[arm].append(plan)
                    these=_request_rows(plan,q,ci,phase,psha,prefix=f'{q}_{arm}_{ix}')
                    reqs.extend(these);ids[arm].append([r['request_id'] for r in these])
            yield {'unit_id':q,'conversation':ci,'records':u['records'],'locators':locators,'plans':plans,
                'request_ids':ids,'multi_resolved':resolved,'prior_unit':prior_ref},reqs
    elif phase=='rank':
        results=phase_results(reader,run,'closure')
        byq={u['unit_id']:(ref,u) for ref,u in units(reader,run,'closure')}
        for j in jobs:
            q,ci=j['query_id'],j['conversation'];prior_ref,u=byq[q]
            locators={};reqs=[];ids={}
            for arm,key,_ in source_arm_specs():
                facts=u['records'][key];resolved=[]
                for plan,rids in zip(u['plans'][arm],u['request_ids'][arm]):
                    scores,status=merged_scores(rids,results)
                    resolved.append(core.resolve_closure(plan,scores,score_status=status))
                loc=core.apply_closures(u['locators'][arm],resolved,facts,closure_plans=u['plans'][arm])
                locators[arm]=loc
                sig=core.build_rank_request(j['question'],j['reference_date'],facts,loc['candidates'])
                rid=f'{q}_{arm}' if sig is not None else None;ids[arm]=rid
                if sig is not None:reqs.append(request_value(sig,rid,q,ci,phase,psha))
            yield {'unit_id':q,'conversation':ci,'records':u['records'],'locators':locators,
                'request_ids':ids,'prior_unit':prior_ref},reqs
    else:raise ValueError('Unknown source refinement phase')


def source_arm_specs():
    return (('original_source_closure','original',False),('refined_source_closure','refined',False),
        ('original_multi_source_closure','original',True),('refined_multi_source_closure','refined',True))


def expected_selections(reader,run):
    from jev_mem import r05_refinement_v1 as core
    m,b,_=base(reader,run);results=phase_results(reader,run,'rank');refined=_resolved_r03(reader,run)
    ranking={u['unit_id']:u for _,u in units(reader,run,'rank')}
    for oldjob in b['original_jobs']:
        q=oldjob['query_id'];p=b['pools'][q]
        for arm in ARMS:
            attachments=[];decision=None
            if arm=='baseline':
                facts=deepcopy(oldjob['selected_facts']);lineage={'source':'frozen_original_baseline'}
            elif arm=='r03_original':
                facts=deepcopy(p['original_r03']);lineage={'source':'frozen_original_R03',
                    'pool_inputs':m['pool_inputs'],'selected_sha256':digest(facts)}
            elif arm=='r03_refined':
                facts=deepcopy(refined[q]['records']);lineage=deepcopy(refined[q]);decision=lineage['decision']
            else:
                key=next(k for a,k,_ in source_arm_specs() if a==arm)
                ru=ranking[q];upstream=ru['records'][key];candidates=ru['locators'][arm]['candidates']
                rid=ru['request_ids'][arm]
                if rid is None:scores,status,score_ref=None,'not_dispatched',None
                else:
                    row=results[rid];scores,status,score_ref=row['scores'],row['status'],row['result_ref']
                decision=core.choose_rank(upstream,candidates,scores,score_status=status)
                facts=core.selected_facts(upstream,candidates,decision)
                attachments=[deepcopy(candidates[i]) for i in decision['candidate_indices']]
                lineage={'upstream_method':'r03_'+key,'upstream_selected_sha256':digest(upstream),
                    'rank_unit':reader.pointer(run/'phases/rank/units'/(q+'.json')),'score_result':score_ref,
                    'closure_enabled':True,'multi_enabled':'multi' in arm}
            if len(facts)!=20 or len({f['id'] for f in facts})!=20:raise ValueError('Exactly 20 unique facts required')
            rec={'schema':'r03-r05-refinement-selection-v1','protocol_sha256':m['protocol_sha256'],
                'query_id':q,'conversation':oldjob['conversation'],'method':arm,'bank':oldjob['bank'],
                'original_selection':oldjob['selection'],'selected_ids':[f['id'] for f in facts],
                'selected_facts':facts,'selected_sha256':digest(facts),
                'baseline_selected_sha256':digest(oldjob['selected_facts']),
                'unchanged_ordered_reader_facts':facts==oldjob['selected_facts'],'source_job_sha256':digest(oldjob),
                'frozen_source':m['protocol']['frozen_source'],'proposal':None,'request':None,'response':None,
                'raw_response':None,'attachments':attachments,'decision':decision,'lineage':lineage,
                'derived_view_not_new_bank_fact':arm in SOURCE_ARMS}
            yield oldjob,arm,rec


def seal(run_dir):
    run=Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists():raise FileExistsError('Sealed source cannot be overwritten')
    validate_phase(run,'rank')
    reader=Reader();m,_,_=base(reader,run,replay_archive=True);jobs=[];counts=Counter()
    for oldjob,arm,rec in expected_selections(reader,run):
        ref=save_new(run/'selections'/arm/(oldjob['query_id']+'.json'),rec)
        jobs.append(prior.old._job(oldjob,arm,rec,ref,m['protocol_sha256']))
        counts[arm+'_changed_queries']+=not rec['unchanged_ordered_reader_facts']
        counts[arm+'_attachments']+=len(rec['attachments'])
    p=m['protocol']
    value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),
        'questions_per_arm':1540,'jobs':jobs,'parent_run':str(run),'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'],'r03_r05_protocol':p['research_protocol'],
        **{k:p[k] for k in PROOF_KEYS},'prior_qa_reuse':p['prior_qa_reuse'],'code_sha256':p['code_sha256'],
        'selection_counts':dict(counts),'model_calls_during_seal':0}
    return {'source':save_new(run/'selected_inputs.json',value),'jobs':len(jobs),'selection_counts':dict(counts),'model_calls':0}


def validate_source(path,*,fresh=False):
    path=Path(path).resolve();cached=_SOURCE_CACHE.get(str(path))
    inventory=(sorted([str(p.resolve()) for p in Path(cached['run']).rglob('*') if p.is_file()]+support_inventory()) if cached else None)
    if (cached and not fresh and inventory==cached['inventory'] and all(Path(p).is_file()
        and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in cached['stats'].items())):
        return deepcopy(cached['source'])
    reader=Reader();src=reader.read(path);run=Path(src['parent_run']).resolve()
    if (src['schema']!=SOURCE_SCHEMA or src['status']!='sealed' or src['arms']!=list(ARMS)
        or src['questions_per_arm']!=1540 or len(src['jobs'])!=10780 or path!=run/'selected_inputs.json'):
        raise ValueError('Exact seven-arm full-cohort source required')
    validate_phase(run,'rank');m,_,_=base(reader,run,replay_archive=True);p=m['protocol']
    if (src['selector_manifest']!=reader.pointer(run/'run_manifest.json') or src['protocol_sha256']!=m['protocol_sha256']
        or src['r03_r05_protocol']!=p['research_protocol'] or src['code_sha256']!=p['code_sha256']
        or src['prior_qa_reuse']!=p['prior_qa_reuse'] or any(src[k]!=p[k] for k in PROOF_KEYS)):
        raise ValueError('Sealed source lineage changed')
    jobs=[];counts=Counter();selection_paths=set()
    for oldjob,arm,rec in expected_selections(reader,run):
        rp=run/'selections'/arm/(oldjob['query_id']+'.json');selection_paths.add(rp.resolve())
        if plain(reader.read(rp))!=rec:raise ValueError('Selection does not replay')
        jobs.append(prior.old._job(oldjob,arm,rec,reader.pointer(rp),m['protocol_sha256']))
        counts[arm+'_changed_queries']+=not rec['unchanged_ordered_reader_facts']
        counts[arm+'_attachments']+=len(rec['attachments'])
    if (jobs!=src['jobs'] or dict(counts)!=src['selection_counts'] or selection_paths!=
        {p.resolve() for p in (run/'selections').rglob('*.json')}):raise ValueError('Final cohort/count/inventory differs')
    paths=set(reader.refs)|{str(ROOT/p) for p in CODE}|set(support_inventory())
    paths.update(str(p.resolve()) for p in run.rglob('*') if p.is_file())
    _SOURCE_CACHE[str(path)]={'source':deepcopy(src),'run':str(run),
        'inventory':sorted([str(p.resolve()) for p in run.rglob('*') if p.is_file()]+support_inventory()),
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}}
    return src


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','prepare-phase','check-phase','seal','check-source'])
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--phase',choices=PHASES)
    a=parser.parse_args()
    if a.action=='prepare':result=prepare(a.run_dir)
    elif a.action=='prepare-phase':result=prepare_phase(a.run_dir,a.phase)
    elif a.action=='check-phase':
        m,refs=validate_phase(a.run_dir,a.phase);result={'phase':a.phase,'units':len(m['units']),'requests':len(refs),'model_calls':0}
    elif a.action=='seal':result=seal(a.run_dir)
    else:
        src=validate_source(a.run_dir/'selected_inputs.json',fresh=True);result={'jobs':len(src['jobs']),'model_calls':0,'valid':True}
    print(json.dumps(result,ensure_ascii=False,indent=2))
