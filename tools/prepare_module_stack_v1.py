"""Versioned, offline binding and replay for upstream modules then source S x C.

No model client is constructed here.  Archived selections are frozen scientific
inputs, never reconstructed API caches or historical QA producers.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_source_append_r05 as old
from jev_mem import source_support_stack_v1 as core
from jev_mem.evidence_exchange_r01 import request_hash,serialized_signature,digest

Reader,save_new,file_sha,now=old.Reader,old.save_new,old.file_sha,old.now
ARMS=('baseline','pool_old_bm25','pool_old_jev','pool_new_bm25','pool_new_jev',
    'stack_source','stack_support','stack_closure','stack_support_closure')
SOURCE_ARMS=ARMS[5:]
ARCHIVE=ROOT/'archives/retained_rounds_2026-10-05/R03_R04_R06_R08_review_v3.zip'
SPEC=ROOT/'research/module_stack_2026-10-05/protocol_v1.json'
SCHEMA='module-stack-selector-manifest-v1'
SOURCE_SCHEMA='module-stack-source-selected-inputs-v1'
REQUEST_SCHEMA='module-stack-request-v1'
CODE=tuple(dict.fromkeys((*old.CODE,'jev_mem/source_support_stack_v1.py',
    'tools/module_stack_archive_v1.py','tools/prepare_module_stack_v1.py',
    'tools/score_module_stack_v1.py','tools/run_module_stack_qa_v1.py',
    'tools/summarize_module_stack_v1.py')))
PROOF_KEYS=('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof',
    'reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance')
_CACHE={}
_PHASE_CACHE={}


def codes():return {p:file_sha(ROOT/p) for p in CODE}
def plain(row):return {k:v for k,v in row.items() if k!='content_sha256'}
def canonical_fact(f):return {k:deepcopy(f[k]) for k in ('id','text','observation_date')}
def compact_locator(loc):
    return {k:deepcopy(loc[k]) for k in ('candidates','support_locator','source_turn_count',
        'per_fact_candidate_count','deduplicated_candidate_count') if k in loc} | {'full_locator_sha256':digest(loc)}


def phase_input_inventory(run,phase):
    current=run/'phases'/phase
    files=[p for d in ('units','requests') for p in (current/d).glob('*.json')]
    for previous in ('support','closure','rank')[:('support','closure','rank').index(phase)]:
        files.extend(p for p in (run/'phases'/previous).rglob('*') if p.is_file())
    return sorted(str(p.resolve()) for p in files)


def actual_inputs(reader):
    spec,sref=old._research_protocol(reader)
    source,source_ref,jobs,banks,sources=old._inputs(reader,old.DEFAULT_SOURCE,spec)
    if core.LOCAL_PREDICATE!=spec['jev']['predicate']:
        raise ValueError('Original R05 local predicate must remain byte-identical')
    return source,source_ref,jobs,banks,sources,sref


def prepare(run_dir):
    from tools.module_stack_archive_v1 import load_archive
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.name.startswith('module_stack_v1') or run.exists():
        raise FileExistsError('Fresh direct runs/module_stack_v1 namespace required')
    reader=Reader();spec=reader.read(SPEC,sealed=False)
    if spec['arms']!=list(ARMS) or spec['source_core']!=core.PROTOCOL or spec['questions_per_arm']!=1540:
        raise ValueError('Reviewed research protocol differs from frozen matrix/core')
    source,sref,jobs,banks,sources,priorref=actual_inputs(reader)
    imported=load_archive(ARCHIVE,jobs,sources)
    proof={'frozen_source':sref,**old._proof_pointers(reader),
        'source_dataset_sha256':source['source_dataset_sha256'],'official_metric_provenance':source['official_metric_provenance']}
    protocol={'schema':SCHEMA,'arms':list(ARMS),'questions_per_arm':1540,'source_core':core.PROTOCOL,
        'research_protocol':reader.raw_pointer(SPEC),'r05_scientific_protocol':priorref,
        'archive':imported['archive_ref'],'archive_members':imported['members'],
        'bank_refs':{str(ci):b['ref'] for ci,b in banks.items()},
        'group_input_refs':{str(ci):s['group_ref'] for ci,s in sources.items()},
        'code_sha256':codes(),**proof,'baseline_policy':'bound_full_raw_exact_request_inheritance',
        'upstream_qa_policy':'fresh_generation_from_archived_frozen_selection_inputs',
        'technical_recovery':'one_uniform_known_failure_pass_per_phase_then_explicit_fallback',
        'historical_selected_inputs_not_raw_producer':True}
    psha=digest(protocol)
    run.mkdir(parents=True,exist_ok=False)
    inputref=save_new(run/'base_inputs.json',{'schema':'module-stack-base-inputs-v1',
        'original_jobs':jobs,'imported':imported,'protocol_sha256':psha})
    ref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'status':'prepared','created_utc':now(),
        'protocol':protocol,'protocol_sha256':psha,'base_inputs':inputref,'model_calls_during_prepare':0})
    return {'manifest':ref,'model_calls':0,'archive_audit':imported['audit']}


def base(reader,run,*,replay_archive=False):
    m=reader.read(run/'run_manifest.json');p=m['protocol'];b=reader.bound(m['base_inputs'])
    if (m['schema']!=SCHEMA or m['status']!='prepared' or m['protocol_sha256']!=digest(p)
        or p['arms']!=list(ARMS) or p['source_core']!=core.PROTOCOL or p['code_sha256']!=codes()
        or b['protocol_sha256']!=m['protocol_sha256']):
        raise ValueError('Frozen module-stack protocol/code changed')
    spec=reader.bound(p['research_protocol'],sealed=False)
    if spec['source_core']!=core.PROTOCOL or spec['arms']!=list(ARMS) or spec['questions_per_arm']!=1540:
        raise ValueError('Research protocol changed')
    source,sref,jobs,banks,sources,priorref=actual_inputs(reader)
    if (b['original_jobs']!=jobs or p['frozen_source']!=sref or p['r05_scientific_protocol']!=priorref
        or any(p[k]!=v for k,v in old._proof_pointers(reader).items())
        or p['official_metric_provenance']!=source['official_metric_provenance']
        or p['source_dataset_sha256']!=source['source_dataset_sha256']
        or p['bank_refs']!={str(ci):v['ref'] for ci,v in banks.items()}
        or p['group_input_refs']!={str(ci):s['group_ref'] for ci,s in sources.items()}):
        raise ValueError('Original baseline/source/date/bank/proof identity differs')
    reader.raw_pointer(p['archive']['path'],ref=p['archive'])
    if replay_archive:
        from tools.module_stack_archive_v1 import load_archive
        if load_archive(Path(p['archive']['path']),jobs,sources)!=b['imported']:
            raise ValueError('Derived archived selections do not replay')
    return m,b,sources


def request_value(signature,request_id,query_id,ci,phase,psha):
    core.validate_request(signature,allow_oversize=True)
    size=len(serialized_signature(signature))
    return {'schema':REQUEST_SCHEMA,'request_id':request_id,'query_id':query_id,'conversation':ci,
        'phase':phase,'protocol_sha256':psha,'signature':signature,'request_hash':request_hash(signature),
        'signature_utf8_bytes':size,'dispatch_eligible':size<=131072,
        'skip_reason':None if size<=131072 else 'signature_oversize'}


def phase_results(reader,run,phase,*,allow_recovery=True):
    from tools.score_module_stack_v1 import verify_result,SCHEMA as DISPATCH_SCHEMA
    pm=reader.read(run/'phases'/phase/'manifest.json');requests=pm['requests']
    def producer_results(producer,refs,retries):
        outcome=reader.read(producer/'outcome.json')
        intent=reader.read(producer/'execution_intent.json',sealed=False)
        if (outcome['schema']!=DISPATCH_SCHEMA or outcome['status']!='terminal' or outcome.get('local_errors')
            or outcome['expected_requests']!=len(refs) or outcome['terminal_requests']!=len(refs)
            or intent['requests']!=refs or intent['transport_retries']!=retries
            or intent['dispatch_code_sha256']!=file_sha(ROOT/'tools/score_module_stack_v1.py')
            or intent['manifest_sha256']!=reader.pointer(run/'phases'/phase/'manifest.json')['sha256']):
            raise ValueError('Whole phase/producer terminal gate differs')
        result={}
        for ref in refs:
            req=reader.bound(ref);rid=req['request_id'];rp=producer/'responses'/(rid+'.json')
            rref=reader.pointer(rp)
            row=verify_result(reader,rref,ref,producer,retries)
            result[rid]={**deepcopy(row),'result_ref':rref}
        if {Path(r['result_ref']['path']).resolve() for r in result.values()}!={p.resolve() for p in (producer/'responses').glob('*.json')}:
            raise ValueError('Unmanifested/missing score responses')
        eligible={reader.bound(ref)['request_hash'] for ref in refs if reader.bound(ref)['dispatch_eligible']}
        actual_intents={p.stem for p in (producer/'request_intents').glob('*.json')}
        if actual_intents!=eligible:
            raise ValueError('Unmanifested/missing request intent; unresolved intents block the whole phase')
        raw_hashes={r['request_hash'] for r in result.values() if r.get('raw')}
        if {p.stem for p in (producer/'api_cache').glob('*.json')}!=raw_hashes:
            raise ValueError('Orphan or missing API raw response')
        from tools.score_module_stack_v1 import STAGE,context
        allowed_pairs={(reader.bound(ref)['request_hash'],digest(context(reader.bound(ref))))
            for ref in refs if reader.bound(ref)['dispatch_eligible']}
        for ip in (producer/'request_intents').glob('*.json'):
            iv=reader.read(ip,sealed=False)
            if (ip.stem,digest(iv.get('context'))) not in allowed_pairs or iv.get('request_hash')!=ip.stem or request_hash(iv['signature'])!=ip.stem:
                raise ValueError('Foreign producer intent context')
        from tools.check_experiment_health import shared_read
        lp=producer/'api_calls.jsonl'
        if lp.exists():
            data=shared_read(lp);reader.raw_pointer(lp)
            if data and not data.endswith(b'\n'):raise ValueError('Unterminated ledger line')
            for line in data.splitlines():
                event=json.loads(line)
                if ((event.get('request_hash'),digest(event.get('context'))) not in allowed_pairs
                    or (event.get('stage'),event.get('provider'),event.get('model'),event.get('operation'))
                        !=(STAGE,'jev','jev-1.13.0','systemone')):
                    raise ValueError('Unmanifested ledger request/context/stage')
        return result
    results=producer_results(run/'phases'/phase/'firstpass',requests,2)
    rp=run/'phases'/phase/'recovery_v1'
    if allow_recovery and any(r['status']=='failed' for r in results.values()) and not rp.exists():
        raise ValueError('Known failures require the uniform recovery phase before downstream fallback')
    if allow_recovery and rp.exists():
        failed=[r for r in requests if results[reader.bound(r)['request_id']]['status']=='failed']
        recovery=producer_results(rp,failed,2)
        for rid,row in recovery.items():
            results[rid]={**row,'firstpass_result_ref':results[rid]['result_ref']}
    return results


def merged_scores(request_ids,results):
    if not request_ids:return {},'complete'
    rows=[results[rid] for rid in request_ids]
    if any(r['status']=='unknown' for r in rows):raise ValueError('Unknown model intent cannot be silently retried or consumed')
    if any(r['status']!='complete' for r in rows):
        return None,'failed' if any(r['status']=='failed' for r in rows) else 'not_dispatched'
    combined={}
    for r in rows:
        if set(combined)&set(r['scores']):raise ValueError('Overlapping score keys across support batches')
        combined.update(r['scores'])
    return combined,'complete'


def units(reader,run,phase):
    pm=reader.read(run/'phases'/phase/'manifest.json')
    return [(ref,reader.bound(ref)) for ref in pm['units']]


def support_sidecars(reader,run):
    results=phase_results(reader,run,'support');sides={ci:{} for ci in range(10)}
    for ref,u in units(reader,run,'support'):
        scores,status=merged_scores(u['request_ids'],results)
        sides[u['conversation']][u['plan']['fact']['id']]=core.resolve_support(u['plan'],scores,score_status=status)
    return sides


def phase_values(reader,run,phase):
    m,b,sources=base(reader,run);psha=m['protocol_sha256']
    jobs=b['original_jobs'];imports=b['imported']['arms']
    if phase=='support':
        for ci in range(10):
            for fid,rawfact in sources[ci]['rawfacts'].items():
                plan=core.build_support_plan(canonical_fact(rawfact),rawfact.get('input_scope'),sources[ci]['turns'],conversation=ci)
                uid=f'c{ci}_fact_{fid}'
                reqs=[request_value(r['signature'],uid+f'_b{i}',uid,ci,'support',psha) for i,r in enumerate(plan['requests'])]
                yield {'unit_id':uid,'conversation':ci,'plan':plan,'request_ids':[r['request_id'] for r in reqs]},reqs
    elif phase=='closure':
        sides=support_sidecars(reader,run)
        for j in jobs:
            ci,q=j['conversation'],j['query_id'];s=sources[ci]
            records=imports['pool_new_jev'][q]['selected_facts']
            scopes=[s['rawfacts'][f['id']].get('input_scope') for f in records]
            locators={};plans={};reqs=[];ids={}
            for key,enabled in (('original',False),('support',True)):
                loc=compact_locator(core.build_candidates(j['question'],records,scopes,s['turns'],conversation=ci,
                    enable_support=enabled,support_sidecars=sides[ci] if enabled else None))
                locators[key]=loc;plans[key]=[];ids[key]=[]
                for ix,c in enumerate(loc['candidates']):
                    plan=core.build_closure_plan(j['question'],j['reference_date'],records,c,
                        scopes[c['origin']['anchor_slot']],s['turns'],conversation=ci)
                    plans[key].append(plan);this=[]
                    for bi,r in enumerate(plan['requests']):
                        req=request_value(r['signature'],f'{q}_{key}_{ix}_b{bi}',q,ci,'closure_'+key,psha)
                        reqs.append(req);this.append(req['request_id'])
                    ids[key].append(this)
            yield {'unit_id':q,'conversation':ci,'locators':locators,'plans':plans,'request_ids':ids},reqs
    elif phase=='rank':
        results=phase_results(reader,run,'closure')
        byq={u['unit_id']:u for _,u in units(reader,run,'closure')}
        for j in jobs:
            ci,q=j['conversation'],j['query_id'];u=byq[q]
            records=imports['pool_new_jev'][q]['selected_facts'];locators={};reqs=[];ids={}
            for arm,key,close in (('stack_source','original',False),('stack_support','support',False),
                ('stack_closure','original',True),('stack_support_closure','support',True)):
                loc=u['locators'][key]
                if close:
                    resolved=[]
                    for plan,rids in zip(u['plans'][key],u['request_ids'][key]):
                        scores,status=merged_scores(rids,results)
                        resolved.append(core.resolve_closure(plan,scores,score_status=status))
                    loc=core.apply_closures(loc,resolved,records,closure_plans=u['plans'][key])
                locators[arm]=loc
                sig=core.build_rank_request(j['question'],j['reference_date'],records,loc['candidates'])
                rid=f'{q}_{arm}' if sig is not None else None
                ids[arm]=rid
                if sig is not None:reqs.append(request_value(sig,rid,q,ci,'rank_'+arm,psha))
            yield {'unit_id':q,'conversation':ci,'locators':locators,'request_ids':ids},reqs
    else:raise ValueError('Unknown module-stack phase')


def prepare_phase(run_dir,phase):
    run=Path(run_dir).resolve()
    if phase=='closure':validate_phase(run,'support')
    elif phase=='rank':validate_phase(run,'closure')
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
    ref=save_new(destination/'manifest.json',{'schema':'module-stack-phase-manifest-v1','phase':phase,
        'protocol_sha256':m['protocol_sha256'],'parent_manifest':reader.pointer(run/'run_manifest.json'),
        'units':urefs,'requests':rrefs,'counts':dict(counts),'created_utc':now(),'model_calls':0})
    return {'manifest':ref,'units':len(urefs),'requests':len(rrefs),'counts':dict(counts),'model_calls':0}


def validate_phase(run_dir,phase):
    run=Path(run_dir).resolve()
    key=(str(run),phase);cached=_PHASE_CACHE.get(key)
    if (cached and cached['inventory']==phase_input_inventory(run,phase)
        and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in cached['stats'].items())):
        return deepcopy(cached['manifest']),deepcopy(cached['requests'])
    if phase=='closure':validate_phase(run,'support')
    elif phase=='rank':validate_phase(run,'closure')
    reader=Reader();m,_,_=base(reader,run)
    pm=reader.read(run/'phases'/phase/'manifest.json')
    if (pm['schema']!='module-stack-phase-manifest-v1' or pm['phase']!=phase
        or pm['protocol_sha256']!=m['protocol_sha256'] or pm['parent_manifest']!=reader.pointer(run/'run_manifest.json')):
        raise ValueError('Phase manifest binding changed')
    expected_units=[];expected_requests=[];counts=Counter()
    for u,reqs in phase_values(reader,run,phase):
        up=run/'phases'/phase/'units'/(u['unit_id']+'.json')
        if plain(reader.read(up))!=u:raise ValueError('Pure source plan replay differs: '+u['unit_id'])
        expected_units.append(reader.pointer(up))
        for req in reqs:
            rp=run/'phases'/phase/'requests'/(req['request_id']+'.json')
            if plain(reader.read(rp))!=req:raise ValueError('Full gold-free request replay differs')
            expected_requests.append(reader.pointer(rp));counts['dispatch_eligible']+=req['dispatch_eligible']
            counts['noul_questions']+=len(req['signature']['body']['questions'])
    if pm['units']!=expected_units or pm['requests']!=expected_requests or pm['counts']!=dict(counts):
        raise ValueError('Missing/reordered/unmanifested phase inputs')
    for dirname,refs in (('units',expected_units),('requests',expected_requests)):
        if {Path(r['path']).resolve() for r in refs}!={p.resolve() for p in (run/'phases'/phase/dirname).glob('*.json')}:
            raise ValueError('Phase inventory differs')
    paths=set(reader.refs)|{str(ROOT/p) for p in CODE}|set(phase_input_inventory(run,phase))
    _PHASE_CACHE[key]={'manifest':deepcopy(pm),'requests':deepcopy(expected_requests),
        'inventory':phase_input_inventory(run,phase),
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}}
    return pm,expected_requests


def expected_selections(reader,run):
    m,b,_=base(reader,run);results=phase_results(reader,run,'rank');imports=b['imported']['arms']
    ranking={u['unit_id']:u for _,u in units(reader,run,'rank')}
    for oldjob in b['original_jobs']:
        q=oldjob['query_id'];upstream=imports['pool_new_jev'][q]['selected_facts']
        for arm in ARMS:
            attachments=[];decision=None;lineage=None
            if arm=='baseline':facts=deepcopy(oldjob['selected_facts']);lineage={'source':'frozen_original_baseline'}
            elif arm in ARMS[1:5]:
                imported=imports[arm][q];facts=deepcopy(imported['selected_facts'])
                lineage=deepcopy(imported)
            else:
                ru=ranking[q];candidates=ru['locators'][arm]['candidates'];rid=ru['request_ids'][arm]
                if rid is None:scores,status=None,'not_dispatched';score_ref=None
                else:
                    result=results[rid];scores,status=result['scores'],result['status'];score_ref=result['result_ref']
                decision=core.choose_rank(upstream,candidates,scores,score_status=status)
                facts=core.selected_facts(upstream,candidates,decision)
                attachments=[deepcopy(candidates[i]) for i in decision['candidate_indices']]
                lineage={'upstream_method':'pool_new_jev','upstream_selected_sha256':digest(upstream),
                    'rank_unit':reader.pointer(run/'phases'/'rank'/'units'/(q+'.json')),
                    'score_result':score_ref,'historical_qa_not_imported':True}
            rec={'schema':'module-stack-selection-v1','protocol_sha256':m['protocol_sha256'],
                'query_id':q,'conversation':oldjob['conversation'],'method':arm,'bank':oldjob['bank'],
                'original_selection':oldjob['selection'],'selected_ids':[f['id'] for f in facts],
                'selected_facts':facts,'selected_sha256':digest(facts),
                'baseline_selected_sha256':digest(oldjob['selected_facts']),
                'unchanged_ordered_reader_facts':facts==oldjob['selected_facts'],
                'source_job_sha256':digest(oldjob),'frozen_source':m['protocol']['frozen_source'],
                'proposal':None,'request':None,'response':None,'raw_response':None,
                'attachments':attachments,'decision':decision,'lineage':lineage,
                'derived_view_not_new_bank_fact':arm in SOURCE_ARMS}
            yield oldjob,arm,rec


def seal(run_dir):
    run=Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists():raise FileExistsError('Sealed source cannot be overwritten')
    # A sealed source can only follow complete pure replays of every prior stage.
    validate_phase(run,'rank')
    reader=Reader();m,_,_=base(reader,run,replay_archive=True);jobs=[];counts=Counter()
    for oldjob,arm,rec in expected_selections(reader,run):
        ref=save_new(run/'selections'/arm/(oldjob['query_id']+'.json'),rec)
        jobs.append(old._job(oldjob,arm,rec,ref,m['protocol_sha256']))
        counts[arm+'_changed_queries']+=not rec['unchanged_ordered_reader_facts']
        counts[arm+'_attachments']+=len(rec['attachments'])
    p=m['protocol']
    value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),
        'questions_per_arm':1540,'jobs':jobs,'parent_run':str(run),'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'],'module_stack_protocol':p['research_protocol'],
        **{k:p[k] for k in PROOF_KEYS},'code_sha256':p['code_sha256'],'selection_counts':dict(counts),
        'model_calls_during_seal':0}
    return {'source':save_new(run/'selected_inputs.json',value),'jobs':len(jobs),'selection_counts':dict(counts),'model_calls':0}


def validate_source(path,*,fresh=False):
    path=Path(path).resolve();cached=_CACHE.get(str(path))
    current_inventory=(sorted(str(p.resolve()) for p in Path(cached['run']).rglob('*') if p.is_file()) if cached else None)
    if (cached and not fresh and current_inventory==cached['inventory']
        and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in cached['stats'].items())):
        return deepcopy(cached['source'])
    reader=Reader();src=reader.read(path);run=Path(src['parent_run']).resolve()
    if (src['schema']!=SOURCE_SCHEMA or src['status']!='sealed' or src['arms']!=list(ARMS)
        or src['questions_per_arm']!=1540 or len(src['jobs'])!=13860 or path!=run/'selected_inputs.json'):
        raise ValueError('Exact nine-arm full-cohort source required')
    validate_phase(run,'rank')
    m,_,_=base(reader,run,replay_archive=True);p=m['protocol']
    if (src['selector_manifest']!=reader.pointer(run/'run_manifest.json') or src['protocol_sha256']!=m['protocol_sha256']
        or src['module_stack_protocol']!=p['research_protocol'] or src['code_sha256']!=p['code_sha256']
        or any(src[k]!=p[k] for k in PROOF_KEYS)):
        raise ValueError('Sealed source lineage changed')
    jobs=[];counts=Counter()
    for oldjob,arm,rec in expected_selections(reader,run):
        rp=run/'selections'/arm/(oldjob['query_id']+'.json')
        if plain(reader.read(rp))!=rec:raise ValueError('Selection not reproduced by frozen source/core/scores')
        jobs.append(old._job(oldjob,arm,rec,reader.pointer(rp),m['protocol_sha256']))
        counts[arm+'_changed_queries']+=not rec['unchanged_ordered_reader_facts']
        counts[arm+'_attachments']+=len(rec['attachments'])
    if jobs!=src['jobs'] or dict(counts)!=src['selection_counts']:
        raise ValueError('Final QA inputs/counts differ from exact replay')
    paths=set(reader.refs)|{str(ROOT/p) for p in CODE}
    for phase in ('support','closure','rank'):
        paths.update(str(p.resolve()) for p in (run/'phases'/phase).rglob('*.json'))
        paths.update(str(p.resolve()) for p in (run/'phases'/phase).rglob('*.jsonl'))
    _CACHE[str(path)]={'source':deepcopy(src),'run':str(run),
        'inventory':sorted(str(p.resolve()) for p in run.rglob('*') if p.is_file()),
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}}
    return src


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','prepare-phase','check-phase','seal','check-source'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['support','closure','rank'])
    a=parser.parse_args()
    if a.action=='prepare':result=prepare(a.run_dir)
    elif a.action=='prepare-phase':result=prepare_phase(a.run_dir,a.phase)
    elif a.action=='check-phase':
        m,refs=validate_phase(a.run_dir,a.phase);result={'phase':a.phase,'units':len(m['units']),'requests':len(refs),'model_calls':0}
    elif a.action=='seal':result=seal(a.run_dir)
    else:
        src=validate_source(a.run_dir/'selected_inputs.json',fresh=True);result={'jobs':len(src['jobs']),'model_calls':0,'valid':True}
    print(json.dumps(result,ensure_ascii=False,indent=2))
