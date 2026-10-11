"""R11 offline binding of original R05 candidates to full-set G/P selectors."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import prepare_source_append_r05 as old
from jev_mem import source_set_r11 as core
Reader,save_new,file_sha,now=old.Reader,old.save_new,old.file_sha,old.now
digest,request_hash,serialized_signature=old.digest,old.request_hash,old.serialized_signature
ARMS=('baseline','g','p')
SCHEMA='source-set-r11-selector-manifest-v1'
SOURCE_SCHEMA='source-set-r11-selected-inputs-v1'
REQUEST_SCHEMA='source-set-r11-request-v1'
SCORE_SCHEMA='source-set-r11-score-result-v1'
SELECTION_SCHEMA='source-set-r11-selection-v1'
DEFAULT_SOURCE=ROOT/'runs/source_append_r05_v1/selected_inputs.json'
SPEC=ROOT/'research/module_consolidation_2026-10-05/R11_protocol_v1.json'
CODE=tuple(dict.fromkeys((*old.CODE,'jev_mem/source_set_r11.py','tools/prepare_source_set_r11.py',
    'tools/score_source_set_r11.py','tools/run_source_set_qa_r11.py')))
_VALIDATION_CACHE={}


def codes(): return {p:file_sha(ROOT/p) for p in CODE}


def inputs(reader,source_path):
    source=old.validate_source(source_path)
    source_ref=reader.raw_pointer(source_path)
    manifest=reader.bound(source['selector_manifest'])
    baseline={j['query_id']:j for j in source['jobs'] if j['method']=='baseline'}
    if len(baseline)!=1540 or len(manifest['items'])!=1540: raise ValueError('1540 original queries required')
    rows=[]
    for item in manifest['items']:
        proposal=reader.bound(item['proposal']);job=proposal['original_job'];q=job['query_id']
        if q!=item['query_id'] or q not in baseline: raise ValueError('Original query binding differs')
        fields=('question','reference','category','reference_date','selected_ids','selected_facts','bank','conversation')
        if any(job[k]!=baseline[q][k] for k in fields): raise ValueError('N0 identity differs')
        cs=proposal['locator']['candidates']
        if (len(cs)!=4 or len({c['packet_id'] for c in cs})!=4
            or len({c['origin']['anchor_slot'] for c in cs})!=4):
            raise ValueError('Exactly four distinct original packets and anchors required')
        rows.append((item['proposal'],proposal))
    return source,source_ref,rows


def request_value(job,cs,spec,psha,arm):
    signature=(core.build_g_request(job['question'],job['reference_date'],job['selected_facts'],cs,spec)
        if arm=='g' else core.build_p_request(job['query_id'],job['question'],job['reference_date'],job['selected_facts'],cs,spec))
    size=len(serialized_signature(signature));eligible=size<=131072
    return {'schema':REQUEST_SCHEMA,'arm':arm,'query_id':job['query_id'],'conversation':job['conversation'],
        'protocol_sha256':psha,'signature':signature,'request_hash':request_hash(signature),
        'signature_utf8_bytes':size,'dispatch_eligible':eligible,'skip_reason':None if eligible else 'signature_oversize',
        'logical_request_cap':1,'max_physical_attempts_per_phase':3}


def prepare(run_dir,source_path=DEFAULT_SOURCE):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.name.startswith('source_set_r11') or run.exists():
        raise FileExistsError('Fresh direct runs/source_set_r11* namespace required')
    reader=Reader();spec=reader.read(SPEC,sealed=False);sref=reader.raw_pointer(SPEC)
    source,sourceref,rows=inputs(reader,source_path)
    p={'schema':SCHEMA,'policy':core.POLICY,'research_protocol':sref,'r05_source':sourceref,
        'r05_selector_manifest':source['selector_manifest'],'arms':list(ARMS),'questions_per_arm':1540,
        'code_sha256':codes(),**{k:source[k] for k in ('frozen_source','frozen_qa_run_manifest',
            'frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance')}}
    psha=digest(p)
    # All request construction occurs before creation; no partial scientific input set.
    prepared=[(ref,prop,{a:request_value(prop['original_job'],prop['locator']['candidates'],spec,psha,a) for a in ('g','p')}) for ref,prop in rows]
    run.mkdir(parents=True,exist_ok=False);items=[];counts=Counter()
    for ref,prop,requests in prepared:
        q=prop['query_id'];refs={a:save_new(run/'requests'/a/f'{q}.json',v) for a,v in requests.items()}
        items.append({'query_id':q,'conversation':prop['conversation'],'proposal':ref,'requests':refs})
        for a,v in requests.items(): counts[a+'_dispatch_eligible']+=v['dispatch_eligible']
    m={'schema':SCHEMA,'status':'prepared','created_utc':now(),'protocol':p,'protocol_sha256':psha,
       'items':items,'summary':dict(counts),'model_calls_during_prepare':0}
    ref=save_new(run/'run_manifest.json',m)
    return {'manifest':ref,'queries':1540,'requests':3080,'summary':dict(counts),'model_calls':0}


def validated(reader,run):
    m=reader.read(run/'run_manifest.json');p=m['protocol'];spec=reader.bound(p['research_protocol'],sealed=False)
    if (m['schema']!=SCHEMA or m['status']!='prepared' or m['protocol_sha256']!=digest(p)
        or p['code_sha256']!=codes() or p['policy']!=core.POLICY or p['arms']!=list(ARMS)
        or p['research_protocol']!=reader.raw_pointer(SPEC) or p['questions_per_arm']!=1540):
        raise ValueError('Frozen R11 protocol/code differs')
    source,sref,rows=inputs(reader,p['r05_source']['path'])
    if sref!=p['r05_source'] or p['r05_selector_manifest']!=source['selector_manifest']: raise ValueError('R05 producer changed')
    for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance'):
        if p[k]!=source[k]: raise ValueError('Original source binding changed: '+k)
    if len(m['items'])!=1540: raise ValueError('Full cohort required')
    refs=[]
    for item,(ref,prop) in zip(m['items'],rows):
        j=prop['original_job'];q=j['query_id']
        if (item['query_id']!=q or item['conversation']!=j['conversation'] or item['proposal']!=ref
            or set(item['requests'])!={'g','p'}): raise ValueError('R11 original proposal identity changed')
        for a in ('g','p'):
            reqref=item['requests'][a];req=reader.bound(reqref)
            if Path(reqref['path']).resolve()!=run/'requests'/a/f'{q}.json': raise ValueError('Request namespace differs')
            if {k:v for k,v in req.items() if k!='content_sha256'}!=request_value(j,prop['locator']['candidates'],spec,m['protocol_sha256'],a):
                raise ValueError('Full gold-free request replay differs')
            refs.append(reqref)
    if {Path(r['path']).resolve() for r in refs}!={x.resolve() for x in (run/'requests').rglob('*.json')}:
        raise ValueError('Unmanifested/missing requests')
    return m,rows,refs


def validate_prepared(run_dir):
    m,_,refs=validated(Reader(),Path(run_dir).resolve());return m,refs


def score(reader,run,item,arm):
    req=reader.bound(item['requests'][arm]);path=run/'responses'/arm/f'{item["query_id"]}.json'
    result=reader.read(path);ref=reader.pointer(path)
    if (result.get('schema')!=SCORE_SCHEMA or result.get('arm')!=arm or result.get('query_id')!=item['query_id']
        or result.get('conversation')!=item['conversation'] or result.get('actual_request_hash')!=req['request_hash']
        or result.get('request_path')!=item['requests'][arm]['path'] or result.get('request_sha256')!=item['requests'][arm]['sha256']
        or result.get('request_hash')!=req['request_hash'] or result.get('protocol_sha256')!=req['protocol_sha256']):
        raise ValueError('R11 score identity mismatch')
    status=result['status'];rawref=result.get('raw');parsed=None
    if rawref and Path(rawref['path']).resolve()!=run/'api_cache'/f'{req["request_hash"]}.json': raise ValueError('Foreign score raw')
    raw=reader.bound(rawref,sealed=False) if rawref else None
    if raw is not None and raw.get('request')!=req['signature']: raise ValueError('Raw signature differs')
    if not req['dispatch_eligible']:
        if (status!='not_dispatched' or raw is not None or result.get('scores') is not None
            or result.get('reason')!='serialized_signature_byte_cap_exceeded'
            or (run/'request_intents'/f'{req["request_hash"]}.json').exists()
            or (run/'api_cache'/f'{req["request_hash"]}.json').exists()):
            raise ValueError('Oversize request must remain undispatched without scores or raw')
        lp=run/'api_calls.jsonl'
        if lp.exists():
            reader.raw_pointer(lp)
            if any(json.loads(line).get('request_hash')==req['request_hash'] for line in lp.read_bytes().splitlines() if line):
                raise ValueError('Oversize request was dispatched')
    else:
        verify_dispatch(reader,run,req,result,raw,arm)
    if status=='complete':
        if raw is None: raise ValueError('Complete score requires raw')
        parsed=(core.parse_g if arm=='g' else core.parse_p)(raw['response'],req['signature'])
        if result.get('scores')!=parsed: raise ValueError('Parsed scores differ from raw')
    elif status=='unknown': raise ValueError('Unresolved request blocks seal')
    elif status=='not_dispatched':
        if req['dispatch_eligible'] or raw is not None: raise ValueError('Invalid skip')
    elif status!='failed' or not (result.get('error_type') and result.get('reason')):
        raise ValueError('Failed score needs explicit terminal reason')
    if status!='complete' and result.get('scores') is not None: raise ValueError('No synthetic scores')
    return parsed,status,ref,rawref,req


def verify_dispatch(reader,run,req,result,raw,arm):
    """Terminal evidence from the actual producer, independent of status strings."""
    from tools.score_source_set_r11 import classify_failure,STAGES
    stage=STAGES[arm];key=req['request_hash'];ip=run/'request_intents'/f'{key}.json'
    intent=reader.read(ip,sealed=False)
    if (intent.get('request_hash')!=key or intent.get('signature')!=req['signature']
        or intent.get('stage')!=stage or intent.get('retries')!=2): raise ValueError('Producer intent mismatch')
    if not hasattr(reader,'_r11_events'):
        lp=run/'api_calls.jsonl';reader.raw_pointer(lp)
        data=lp.read_bytes()
        if data and not data.endswith(b'\n'): raise ValueError('Unfinished API ledger line')
        events={}
        for line in data.splitlines():
            event=json.loads(line);events.setdefault(event.get('request_hash'),[]).append(event)
        reader._r11_events=events
    events=reader._r11_events.get(key,[])
    rows=[e for e in events if not e.get('cache_hit') and e.get('context')==intent['context']]
    if (not rows or len(rows)>3 or [r.get('attempt') for r in rows]!=list(range(1,len(rows)+1))
        or any((r.get('stage'),r.get('provider'),r.get('model'),r.get('operation'))!=(stage,'jev','jev-1.13.0','systemone') for r in rows)
        or any(not r.get('timestamp_utc') for r in rows)):
        raise ValueError('Incomplete/foreign physical attempt sequence')
    retryable={429,500,502,503,504,529,'transport_error'}
    if any(r['status'] not in retryable for r in rows[:-1]): raise ValueError('Nonretryable attempt repeated')
    context={'conversation':req['conversation'],'query_id':req['query_id'],'method':'source_set_r11_'+arm,'phase':arm}
    if not any(e.get('context')==context and e.get('stage')==stage for e in events): raise ValueError('Missing consumer event')
    if result['status']=='complete':
        if rows[-1]['status']!=200 or raw is None: raise ValueError('Complete score lacks successful physical producer')
    elif result['status']=='failed':
        if raw is not None:
            if rows[-1]['status']!=200: raise ValueError('Raw with no HTTP200')
            try: (core.parse_g if arm=='g' else core.parse_p)(raw['response'],req['signature'])
            except (ValueError,TypeError,KeyError): pass
            else: raise ValueError('Valid response cannot be marked failed')
        api=SimpleNamespace(run_dir=run,context=context)
        state,reason,evidence=classify_failure(api,req['signature'],run/'api_cache'/f'{key}.json',arm)
        if (state!='failed' or result.get('reason')!=reason or result.get('failure_evidence')!=evidence):
            raise ValueError('Failure producer/consumer/terminal proof differs')
    else: raise ValueError('Eligible nonterminal/unknown score blocks seal')


def records(reader,run,m,rows):
    outcome=reader.read(run/'score_jev_outcome.json',sealed=False)
    if (outcome.get('schema')!='source-set-r11-dispatch-v1' or outcome.get('status')!='terminal'
        or outcome.get('terminal_requests')!=3080 or outcome.get('expected_requests')!=3080
        or outcome.get('local_errors')): raise ValueError('Whole score phase must be terminal without local errors')
    intent=reader.read(run/'execution_intent_score_jev.json',sealed=False)
    if (intent.get('manifest_sha256')!=reader.pointer(run/'run_manifest.json')['sha256']
        or intent.get('dispatch_code_sha256')!=file_sha(ROOT/'tools/score_source_set_r11.py')):
        raise ValueError('Execution intent/code binding differs')
    expected_paths={run/'responses'/a/f'{i["query_id"]}.json' for i in m['items'] for a in ('g','p')}
    if expected_paths!={x.resolve() for x in (run/'responses').glob('*/*.json')}:
        raise ValueError('Response inventory mismatch')
    output=[]
    for item,(pref,prop) in zip(m['items'],rows):
        j=prop['original_job'];cs=prop['locator']['candidates'];fs=j['selected_facts']
        for a in ARMS:
            parsed=status=response=raw=req=None
            if a=='baseline': decision=old.choose_baseline(fs,cs);facts=deepcopy(fs)
            else:
                parsed,status,response,raw,req=score(reader,run,item,a)
                decision=(core.choose_g(fs,cs,parsed,score_status=status) if a=='g' else
                    core.choose_p(fs,cs,parsed,signature=req['signature'],score_status=status))
                facts=core.selected_facts(fs,cs,decision)
            rec={'schema':SELECTION_SCHEMA,'protocol_sha256':m['protocol_sha256'],'query_id':j['query_id'],
                'conversation':j['conversation'],'method':a,'bank':j['bank'],'original_selection':j['selection'],
                'proposal':pref,'request':item['requests'][a] if a!='baseline' else None,
                'response':response,'raw_response':raw,'raw_scores':parsed,'decision':decision,
                'selected_ids':j['selected_ids'],'selected_facts':facts,'selected_sha256':digest(facts),
                'baseline_selected_sha256':digest(fs),'unchanged_ordered_reader_facts':facts==fs,
                'source_job_sha256':digest(j),'frozen_source':m['protocol']['frozen_source'],
                'derived_view_not_new_bank_fact':True,'attachments':[deepcopy(cs[i]) for i in decision['candidate_indices']]}
            output.append((j,a,rec))
    return output


def seal(run_dir):
    run=Path(run_dir).resolve()
    if (run/'selected_inputs.json').exists(): raise FileExistsError('Sealed source exists')
    reader=Reader();m,rs,_=validated(reader,run);rows=records(reader,run,m,rs);jobs=[];counts=Counter()
    for j,a,rec in rows:
        ref=save_new(run/'selections'/a/f'{j["query_id"]}.json',rec)
        jobs.append(old._job(j,a,rec,ref,m['protocol_sha256']))
        counts[a+'_changed_queries']+=not rec['unchanged_ordered_reader_facts']
        counts[a+'_attachments']+=len(rec['attachments'])
        counts[a+'_score_'+str(rec['decision']['score_status'])]+=1
    p=m['protocol'];value={'schema':SOURCE_SCHEMA,'status':'sealed','created_utc':now(),'arms':list(ARMS),
        'questions_per_arm':1540,'jobs':jobs,'parent_run':str(run),'selector_manifest':reader.pointer(run/'run_manifest.json'),
        'protocol_sha256':m['protocol_sha256'],'source_set_protocol':p['research_protocol'],'r05_source':p['r05_source'],
        **{k:p[k] for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof',
            'reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256')},
        'selection_counts':dict(counts),'model_calls_during_seal':0}
    return {'source':save_new(run/'selected_inputs.json',value),'jobs':len(jobs),'selection_counts':dict(counts),'model_calls':0}


def validate_source(path,*,fresh=False):
    path=Path(path).resolve();key=str(path);cached=_VALIDATION_CACHE.get(key)
    if cached and not fresh and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in cached['stats'].items()):
        return deepcopy(cached['value'])
    reader=Reader();src=reader.read(path);run=Path(src['parent_run']).resolve()
    if path!=run/'selected_inputs.json' or src['schema']!=SOURCE_SCHEMA or src['status']!='sealed' or src['arms']!=list(ARMS):
        raise ValueError('Sealed R11 source required')
    m,rs,_=validated(reader,run);p=m['protocol']
    if src['protocol_sha256']!=m['protocol_sha256'] or src['selector_manifest']!=reader.pointer(run/'run_manifest.json') or src['source_set_protocol']!=p['research_protocol']:
        raise ValueError('R11 source/protocol binding differs')
    for k in ('r05_source','frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest','source_dataset_sha256','official_metric_provenance','code_sha256'):
        if src[k]!=p[k]: raise ValueError('Source lineage differs: '+k)
    jobs=[]
    for j,a,rec in records(reader,run,m,rs):
        sp=run/'selections'/a/f'{j["query_id"]}.json';actual=reader.read(sp)
        if {k:v for k,v in actual.items() if k!='content_sha256'}!=rec: raise ValueError('Selected view replay differs')
        jobs.append(old._job(j,a,rec,reader.pointer(sp),m['protocol_sha256']))
    if src['jobs']!=jobs or Counter(j['method'] for j in jobs)!={a:1540 for a in ARMS}: raise ValueError('Full cohort differs')
    stats={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in reader.refs}
    stats.update(old._VALIDATION_CACHE[str(Path(p['r05_source']['path']).resolve())]['stats'])
    for rel in CODE:
        fp=str((ROOT/rel).resolve());stats[fp]=(Path(fp).stat().st_size,Path(fp).stat().st_mtime_ns)
    _VALIDATION_CACHE[key]={'stats':stats,'value':deepcopy(src)}
    return src


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('prepare','check','seal','validate-source'))
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE);args=ap.parse_args()
    if args.stage=='prepare': r=prepare(args.run_dir,args.source)
    elif args.stage=='check':
        m,refs=validate_prepared(args.run_dir);r={'valid':True,'requests':len(refs),'queries':len(m['items']),'model_calls':0}
    elif args.stage=='seal': r=seal(args.run_dir)
    else: r={'valid':True,'jobs':len(validate_source(args.run_dir/'selected_inputs.json',fresh=True)['jobs']),'model_calls':0}
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__': main()
