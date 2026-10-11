"""Offline R04 publication: replay prior QA, then inherit selected exact answers.

This module never creates an API client or generates a Reader/Judge response.
R04 request construction and scoring contain no gold; only this reporting
process loads the original references and labels after decisions are persisted.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib,json,math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import summarize_evidence_selection_r03 as prior
from tools import answer_selection_inheritance_r04 as inheritance
from tools.run_sdk20_modules import digest,save_new,now
from tools.run_sdk20_judge_retest_v2 import pointer,sha
from jev_mem.answer_selection_r04 import (parse_choice,choose_jev,cheap_choice,
    normalize_abstention,POLICY,request_hash,serialized_signature)

SCHEMA='answer-selection-r04-offline-summary-v1'
ARMS=inheritance.ARMS
STAGE='answer_selection_score_r04'
CODE=tuple(dict.fromkeys((*prior.CODE,'jev_mem/answer_selection_r04.py',
    'tools/prepare_answer_selection_r04.py','tools/score_answer_selection_r04.py',
    'tools/answer_selection_inheritance_r04.py','tools/summarize_answer_selection_r04.py')))
require=inheritance.require

class Snapshot:
    def __init__(self):self.files={};self.values={};self.bytes={}
    def raw(self,path):
        path=str(Path(path).resolve())
        if path not in self.bytes:
            self.bytes[path]=Path(path).read_bytes()
            self.files[path]=hashlib.sha256(self.bytes[path]).hexdigest()
        return self.bytes[path]
    def read(self,path,*,sealed=False):
        path=str(Path(path).resolve())
        if path not in self.values:self.values[path]=json.loads(self.raw(path))
        value=self.values[path]
        if sealed:require(value.get('content_sha256')==digest({k:v for k,v in value.items() if k!='content_sha256'}),'Content SHA differs: '+path)
        return value
    def pointer(self,path):self.raw(path);path=str(Path(path).resolve());return {'path':path,'sha256':self.files[path]}
    def sha(self,path):return self.pointer(path)['sha256']
    def bound(self,ref,*,sealed=False):
        require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'Exact file pointer required')
        value=self.read(ref['path'],sealed=sealed)
        require(self.pointer(ref['path'])==ref,'File pointer SHA differs: '+ref['path'])
        return value
    def ledger(self,path):
        raw=self.raw(path)
        require(not raw or raw.endswith(b'\n'),'Unterminated ledger line')
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    def unchanged(self):
        require(all(Path(p).is_file() and sha(p)==h for p,h in self.files.items()),'Inputs changed during offline summary')

def context(qid):return {'conversation':int(qid.split('_')[0][1:]),'query_id':qid,'method':'jev_answer_choice','phase':'jev'}

def rejected_response_diagnostic(response,parser_error):
    """Describe rejected raw values; never normalize or override the frozen parser."""
    answers=response.get('answers',{}) if isinstance(response,dict) else {}
    answer=answers.get('answer_choice',{}) if isinstance(answers,dict) else {}
    if not isinstance(answer,dict):answer={}
    values=answer.get('probabilities')
    labels=('first_better','second_better','no_clear_preference','neither_supported')
    numeric=isinstance(values,dict) and set(values)==set(labels) and all(
        isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1
        for v in values.values())
    total=sum(values.values()) if numeric else None
    maxima=[k for k,v in values.items() if abs(v-max(values.values()))<=1e-12] if numeric else []
    category=('probability_sum_outside_frozen_tolerance' if numeric and abs(total-1)>1e-6 else
        'returned_choice_not_maximum' if numeric and answer.get('choice') not in maxima else
        'other_schema_validation_failure')
    return {'category':category,'frozen_parser_error':str(parser_error),
        'raw_probabilities':values,'raw_probability_sum':total,
        'raw_probability_sum_distance_from_one':abs(total-1) if numeric else None,
        'returned_choice':answer.get('choice'),'maximum_probability_labels':maxima,
        'raw_confidence':answer.get('confidence'),'reclassified_as_valid':False,
        'probabilities_renormalized':False,'qa_missing_or_answer_error':False}

def choice_validity_report(audits):
    rejected=[{'query_id':q,'raw':a['raw'],**a['schema_rejection']} for q,a in sorted(audits.items())
              if a.get('schema_rejection') is not None]
    return {'schema_rejections':rejected,'schema_rejection_count':len(rejected),
        'schema_rejection_categories':dict(Counter(r['category'] for r in rejected)),
        'status_counts':dict(Counter(a['status'] for a in audits.values())),
        'valid_choice_semantic_fallback_count':sum(a['status']=='complete' and a['selected_route']=='baseline'
            for a in audits.values()),
        'schema_failure_inherits_original_valid_qa':True,
        'schema_failure_is_not_qa_missing_or_human_answer_error':True}

def audit_wrapper(snapshot,run,rec,wrapper,events):
    """Independent raw/signature/producer/consumer audit, with no semantic retry."""
    run=Path(run).resolve();qid=rec['query_id'];sig=rec['signature'];key=request_hash(sig)
    ref=snapshot.pointer(run/'requests'/f'{qid}.json')
    require(wrapper.get('schema')=='answer-selection-r04-score-result-v1'
        and wrapper.get('query_id')==qid and wrapper.get('protocol_sha256')==rec['protocol_sha256']
        and wrapper.get('request_hash')==key==rec['request_hash']
        and wrapper.get('request_path')==ref['path'] and wrapper.get('request_sha256')==ref['sha256'],
        'Wrapper request/protocol/query binding differs')
    require(wrapper.get('presentation')==rec['presentation'] and wrapper.get('union_sha256')==digest(rec['state']['memories'])
        and wrapper.get('serialized_signature_utf8_bytes')==len(serialized_signature(sig)),
        'Wrapper presentation/union/byte signature differs')
    require(datetime.fromisoformat(wrapper['started_utc'])<=datetime.fromisoformat(wrapper['finished_utc']),
            'Wrapper interval differs')
    own=[e for e in events if e.get('context')==context(qid)]
    status=wrapper['status'];parsed=None;eligibility=False;schema_rejection=None
    if status in ('identical_answer_no_dispatch','not_dispatched'):
        require(wrapper.get('actual_request_hash') is None and wrapper.get('raw') is None
            and wrapper.get('choice_result') is None and not events
            and not (run/'request_intents'/f'{key}.json').exists(), 'Offline fallback was dispatched')
        require((status=='identical_answer_no_dispatch' and rec['identical']) or
            (status=='not_dispatched' and not rec['identical'] and len(serialized_signature(sig))>131072),
            'Offline fallback eligibility differs')
    else:
        require(status in ('complete','unknown','failed') and not rec['identical'] and rec['dispatch_eligible']
            and wrapper.get('actual_request_hash')==key,'Invalid dispatch status or eligibility')
        intent_path=run/'request_intents'/f'{key}.json'
        intent=snapshot.read(intent_path) if intent_path.exists() else None
        if intent is not None:
            require(intent.get('request_hash')==key and intent.get('signature')==sig and intent.get('stage')==STAGE
                and intent.get('retries')==2,'Intent full signature/stage/retry differs')
            producer=intent.get('context')
            require(isinstance(producer,dict) and producer==context(producer.get('query_id','')), 'Unexpected producer context')
            physical=[e for e in events if e.get('cache_hit') is not True]
            require(all(e.get('context')==producer for e in physical),'Unrelated context supplies physical producer')
            require(len(physical)<=3 and [e.get('attempt') for e in physical]==list(range(1,len(physical)+1)),
                    'Physical attempt sequence/cap differs')
            for e in physical:
                require(datetime.fromisoformat(e['timestamp_utc'])>=datetime.fromisoformat(intent['created_utc']),
                        'Physical attempt precedes intent')
            require(all(e.get('status') in (429,500,502,503,504,529,'transport_error') for e in physical[:-1]),
                    'A semantic/successful outcome was retried')
        else:physical=[]
        if wrapper.get('raw'):
            raw_ref=wrapper['raw']
            require(Path(raw_ref['path']).resolve()==run/'api_cache'/f'{key}.json','Raw escaped current namespace')
            raw=snapshot.bound(raw_ref)
            require(raw.get('request')==sig and raw.get('stage')==STAGE,'Actual raw full signature/stage differs')
        else:raw=None
        if status=='complete':
            require(intent is not None and raw is not None and physical and own,'Complete consumer has no actual lineage')
            require(physical[-1].get('status')==200 and physical[-1].get('resolved_model')=='jev-1.13.0',
                    'Complete raw has no pinned-model successful producer')
            actual_usage=raw['response'].get('usageMetadata',raw['response'].get('usage',{}))
            require(physical[-1].get('usage')==actual_usage,'Ledger/raw usage differs')
            parsed=parse_choice(raw['response'],sig)
            require(parsed==wrapper.get('choice_result'),'Wrapper Choice differs from raw parse')
        else:
            require(wrapper.get('choice_result') is None,'Unknown cannot be synthetic zero probabilities')
            evidence=wrapper.get('failure_evidence')
            require(isinstance(evidence,dict) and evidence.get('consumer_context')==context(qid),
                    'Unavailable wrapper lacks retained actual failure evidence')
            if evidence.get('intent') is not None:
                snapshot.bound(evidence['intent'])
                require(evidence['intent']==snapshot.pointer(intent_path) and evidence.get('producer_context')==intent['context'],
                        'Failure evidence producer/intent differs')
                actual=[{k:e.get(k) for k in ('attempt','status','timestamp_utc')} for e in physical]
                captured=evidence.get('terminal_attempts',[])
                # A frozen unknown consumer is not promoted by later same-hash
                # success. Its captured evidence must be an actual prefix.
                require(actual[:len(captured)]==captured,'Captured terminal evidence is not the actual producer prefix')
            else:require(not own,'Failure with no intent nevertheless has own ended event')
            if status=='failed' and wrapper.get('reason')=='response_validation_failed':
                require(raw is not None,'Malformed successful response needs exact original raw')
                require(physical and physical[-1].get('status')==200
                    and physical[-1].get('resolved_model')=='jev-1.13.0',
                    'Malformed raw lacks pinned successful HTTP producer')
                require(physical[-1].get('usage')==raw['response'].get('usageMetadata',raw['response'].get('usage',{})),
                        'Malformed raw/ledger usage differs')
                try:parse_choice(raw['response'],sig)
                except (ValueError,TypeError,KeyError) as exc:schema_rejection=rejected_response_diagnostic(raw['response'],exc)
                else:raise ValueError('Valid raw was marked malformed for semantic retry')
            if status=='failed' and wrapper.get('reason') in ('transport_exhausted_three_attempts','http_error_terminal'):
                require(physical and (physical[-1].get('status')=='transport_error' and len(physical)==3
                    or type(physical[-1].get('status')) is int and physical[-1]['status']>=400),
                    'Known technical failure lacks actual terminal evidence')
                eligibility=physical[-1].get('status') in (429,500,502,503,504,529,'transport_error')
    replay=choose_jev(rec,parsed,status=status)
    require(wrapper.get('selected_route')==replay['route'] and wrapper.get('selection_reason')==replay['reason']
        and wrapper.get('cheap_route')==cheap_choice(rec),'Persisted route policy differs')
    candidate=rec['candidates'][replay['route']]
    require(wrapper.get('selected_original_reader')==candidate['reader']
        and wrapper.get('selected_exact_answer')==candidate['answer'],'Selected exact original answer/pointer differs')
    return {'technical_recovery_eligible':eligibility,'status':status,'selected_route':replay['route'],
        'selection_reason':replay['reason'],'raw':wrapper.get('raw'),'schema_rejection':schema_rejection,
        'new_reader_calls':0,'new_judge_calls':0}

def audit_score_phase(run,manifest,records,snapshot):
    run=Path(run).resolve()
    outcome=snapshot.read(run/'score_jev_outcome.json',sealed=True)
    controller=snapshot.read(run/'execution_intent_score_jev.json',sealed=True)
    require(outcome.get('status')=='terminal' and outcome.get('local_errors')==[]
        and outcome.get('terminal_requests')==1540 and outcome.get('expected_requests')==1540,
        'R04 Choice phase is not wholly terminal')
    require(controller.get('manifest_sha256')==snapshot.pointer(run/'run_manifest.json')['sha256']
        and controller.get('dispatch_code_sha256')==sha(ROOT/'tools/score_answer_selection_r04.py')
        and controller.get('phase')=='jev' and controller.get('pid')==outcome.get('pid')
        and controller.get('transport_retries')==2 and controller.get('signature_byte_cap')==131072,
        'Controller actual code/manifest/producer/budget differs')
    expected_refs=[snapshot.pointer(run/'requests'/f'{r["query_id"]}.json') for r in records]
    require(controller.get('request_files')==expected_refs,'Controller request set/order differs')
    requests={r['query_id']:r for r in records}
    require(len(requests)==1540 and len(records)==1540,'Full unique1540 prepared requests required')
    require({p.stem for p in (run/'responses').glob('*.json')}==set(requests),'Unmanifested or absent response')
    ledger_path=run/'api_calls.jsonl';logs=snapshot.ledger(ledger_path) if ledger_path.exists() else []
    byhash=defaultdict(list);qhash=defaultdict(set)
    for r in records:qhash[r['request_hash']].add(r['query_id'])
    for e in logs:
        require(e.get('request_hash') in qhash and e.get('stage')==STAGE
            and (e.get('provider'),e.get('model'),e.get('operation'))==('jev','jev-1.13.0','systemone')
            and e.get('context')==context(e.get('context',{}).get('query_id',''))
            and e['context']['query_id'] in qhash[e['request_hash']], 'Ledger outside exact R04 cohort/stage')
        byhash[e['request_hash']].append(e)
    wrappers={};audits={}
    for qid,rec in requests.items():
        wrapper=snapshot.read(run/'responses'/f'{qid}.json',sealed=True)
        wrappers[qid]=wrapper;audits[qid]=audit_wrapper(snapshot,run,rec,wrapper,byhash[rec['request_hash']])
    require(dict(Counter(w['status'] for w in wrappers.values()))==outcome.get('counts'),'Outcome/wrapper counts differ')
    intents=prior.original.intent_inventory(snapshot,run,logs)
    return {'wrappers':wrappers,'audits':audits,'ledger':logs,
        'usage':prior.original.measured_usage(logs,missing_ledgers=[] if ledger_path.exists() else [str(ledger_path)]),
        'intent_inventory':intents,'eligible_qids':[q for q,a in audits.items() if a['technical_recovery_eligible']],
        'phase_sources':{'manifest':snapshot.pointer(run/'run_manifest.json'),
            'outcome':snapshot.pointer(run/'score_jev_outcome.json'),'execution_intent':snapshot.pointer(run/'execution_intent_score_jev.json'),
            'ledger':snapshot.pointer(ledger_path) if ledger_path.exists() else None}}

def audit_original_candidates(spec,snapshot):
    """Execute frozen R03 raw replay, never import labels as choice features."""
    refs=spec['source_refs'];qa_m=snapshot.bound(refs['r03_qa_manifest'])
    # R04 validate_source has just replayed the complete frozen R03 source.
    # Rebind identical bytes here, then give them to the existing raw QA audit;
    # do not repeat the unrelated bank/selection preparation a second time.
    original_source=snapshot.bound(refs['r03_selected_inputs'],sealed=True)
    require(qa_m['source_manifest']==refs['r03_selected_inputs'],'R03 QA source pointer differs')
    phase=prior.audit_phase(Path(refs['r03_qa_manifest']['path']).parent,original_source)
    require(not phase['errors'],'Original raw replay has invalid Reader/Judge/compat/QA artifacts')
    rows=prior.record_rows(original_source['jobs'],phase)
    index=inheritance.candidate_index(rows)
    qa_jobs={(j['method'],j['query_id']):j for j in qa_m['jobs']}
    source_jobs={(j['method'],j['query_id']):j for j in original_source['jobs']}
    return phase,rows,index,qa_jobs,source_jobs

def cold_accounting(spec,phase,index,qa_jobs,snapshot):
    """Measured producers needed for two routes, excluding cheap-only/Judge work."""
    keys={request_hash(prior.runner.request_signature(qa_jobs[key],'reader')) for key in index}
    baseline_keys={request_hash(prior.runner.request_signature(qa_jobs[key],'reader')) for key in index if key[0]=='baseline'}
    r03_keys={request_hash(prior.runner.request_signature(qa_jobs[key],'reader')) for key in index if key[0]=='jev'}
    proof=snapshot.bound(phase['manifest']['baseline_reuse_proof'])
    legacy=[];tagged=[]
    for ref in proof['ledger_sources']:
        require(snapshot.pointer(ref['path'])==ref,'Historical producer ledger SHA differs')
        local=snapshot.ledger(ref['path']);legacy.extend(local)
        tagged.extend((ref['path'],e) for e in local)
    old_readers=[e for e in legacy if e.get('request_hash') in keys and e.get('context',{}).get('component')=='reader']
    new_readers=[e for e in phase['ledger'] if e.get('request_hash') in keys and e.get('context',{}).get('component')=='reader']
    all_readers=old_readers+new_readers
    # A cold exact-signature deduplication must not sum two historical physical
    # producers of the same request. Choose earliest recorded producer chain,
    # never lowest tokens or the most favorable answer/label.
    tagged.extend((str(Path(phase['run'])/f"conv_{e['context']['conversation']}/api/api_calls.jsonl"),e)
                  for e in phase['ledger'])
    chains=defaultdict(list)
    for path,e in tagged:
        if e.get('request_hash') in keys and e.get('context',{}).get('component')=='reader' and e.get('cache_hit') is not True:
            chains[(e['request_hash'],path,digest(e['context']))].append(e)
    selected_chains={}
    for key in sorted(keys):
        options=[(chain,events) for chain,events in chains.items() if chain[0]==key]
        if options:
            chain,events=min(options,key=lambda pair:(pair[1][0]['timestamp_utc'],pair[0][1],pair[0][2]))
            selected_chains[key]=(chain,events)
    deduplicated=[e for _,events in selected_chains.values() for e in events]
    known=set(selected_chains)
    scorer_run=Path(spec['source_refs']['r03_selection_manifest']['path']).parent
    scorer=snapshot.ledger(scorer_run/'api_calls.jsonl')
    require(all(e.get('provider')=='jev' and e.get('model')=='jev-1.13.0'
        and e.get('stage')=='evidence_selection_score_r03' for e in scorer),'R03 upstream scoring ledger differs')
    terminal_counts=Counter(str(events[-1].get('status')) for _,events in selected_chains.values())
    has_full_producer_coverage=not (keys-known) and all(events[-1].get('status')==200 for _,events in selected_chains.values())
    reader_usage=prior.original.measured_usage(deduplicated)
    reader_usage['available']=reader_usage['available'] and has_full_producer_coverage
    return {'scope':'Historical observed producers required by N0 and R03J Reader signatures only, plus necessary R03 selector. Not an exact counterfactual paid invoice.',
        'logical_reader_consumers':3080,'unique_reader_request_signatures':len(keys),
        'exact_reader_signatures_shared_by_routes':len(baseline_keys&r03_keys),
        'reader_usage':reader_usage,'selected_producer_chain_terminal_counts':dict(terminal_counts),
        'required_producer_coverage_complete':has_full_producer_coverage,
        'earliest_failed_chain_is_partial_not_full_cold_cost':not has_full_producer_coverage,
        'reader_usage_deduplication':'Earliest timestamp recorded producer chain per complete Reader signature; path/context ties deterministic; not selected by outcome or token cost.',
        'all_route_reader_historical_observations_NOT_deduplicated_cold_cost':prior.original.measured_usage(all_readers),
        'baseline_reader_historical_usage':prior.original.measured_usage([e for e in old_readers if e['request_hash'] in baseline_keys]),
        'r03_route_reader_historical_usage':prior.original.measured_usage([e for e in all_readers if e['request_hash'] in r03_keys]),
        'route_usage_overlaps_not_additive':True,'required_reader_hashes_without_observed_physical_producer':sorted(keys-known),
        'reader_signature_producer_refs':[{'method':m,'query_id':q,'reader':r['reader_artifact'],
            'request_hash':request_hash(prior.runner.request_signature(qa_jobs[(m,q)],'reader'))} for (m,q),r in sorted(index.items())],
        'r03_upstream_selection_usage':prior.original.measured_usage(scorer),
        'new_r04_choice_usage_added_separately':True,'judge_evaluation_excluded_from_deployment_cost':True,
        'r03_all_experiment_qa_usage_NOT_cold_cost':prior.original.measured_usage(phase['ledger']),
        'original_baseline_reader_judge_historical_usage_NOT_cold_cost':proof['inherited_physical_usage'],
        'total_cost_usd':None,'missing_or_unbilled_is_not_zero':True}

def grouped(rows,field):
    result={}
    for value in sorted({r[field] for r in rows},key=str):
        group=[r for r in rows if r[field]==value];denom=sum(r['method']=='baseline' for r in group)
        result[str(value)]={a:{'denominator':denom,'correct':sum(r['scores']['judge_correct']==1 for r in group if r['method']==a),
            **{m:sum(r['scores'][m] or 0 for r in group if r['method']==a)/denom for m in inheritance.METRICS}}
            for a in ARMS}
    return result

def summarize(firstpass,*,source_path=None,bootstrap_samples=2000):
    from tools.prepare_answer_selection_r04 import validate_source
    run=Path(firstpass).resolve();snapshot=Snapshot()
    source_path=Path(source_path) if source_path else run/'selected_answers.json'
    source=validate_source(source_path)
    require(snapshot.read(source_path,sealed=True)==source,'Selected source changed before snapshot')
    require(source['selector_manifest']==snapshot.pointer(run/'run_manifest.json'),'Selected answers belong to another run')
    # validate_source already executed validate_prepared and full source replay.
    # Admit the same bound request bytes, avoiding a redundant slow replay.
    manifest=snapshot.bound(source['selector_manifest'],sealed=True)
    records=[snapshot.bound(item['request'],sealed=True) for item in manifest['items']]
    spec=snapshot.bound(source['answer_protocol'])
    require(source['answer_protocol']['sha256']=='90d05e6e62783e16bcb25bc000e638444ef132b12d17e9adef97f69669fe1b4f',
            'Scientific R04 protocol differs')
    score=audit_score_phase(run,manifest,records,snapshot)
    phase,old_rows,index,qa_jobs,source_jobs=audit_original_candidates(spec,snapshot)
    requests={r['query_id']:r for r in records}
    for rec in records:
        for route,method in inheritance.ROUTES.items():
            candidate=rec['candidates'][route]
            signature=prior.runner.request_signature(qa_jobs[(method,rec['query_id'])],'reader')
            require(candidate['reader_request_signature']==signature and candidate['reader_request_sha256']==request_hash(signature),
                    'Original candidate full Reader request signature differs')
    rows=inheritance.inherit_jobs(source['jobs'],requests,index,qa_jobs,source_jobs)
    for job in source['jobs']:
        decision=snapshot.bound(job['decision'],sealed=True);qid=job['query_id']
        require(decision['route']==job['route'] and decision['answer']==job['answer']
            and decision['reader']==job['reader'],'Decision/source original response differs')
        expected='baseline' if job['method']=='baseline' else cheap_choice(requests[qid]) if job['method']=='cheap' else score['wrappers'][qid]['selected_route']
        require(job['route']==expected,'Sealed selected route cannot be reproduced')
    aggregate=prior.runner.clone(prior.original.aggregate,{**vars(prior.original),'ARMS':ARMS})
    reference=[deepcopy(r) for (m,q),r in index.items() if m=='jev']
    baseline=[r for r in rows if r['method']=='baseline']
    baseline_byid={r['query_id']:r for r in baseline}
    reference_byid={r['query_id']:r for r in reference}
    oracle=Counter(f"{int(baseline_byid[q]['scores']['judge_correct'])}_to_{int(reference_byid[q]['scores']['judge_correct'])}"
                   for q in baseline_byid)
    views={'accepted_firstpass':rows,'final_with_uniform_technical_recovery':deepcopy(rows)}
    comparisons={}
    for name,left in [('baseline',[r for r in rows if r['method']=='baseline']),('r03_reference',reference),('cheap',[r for r in rows if r['method']=='cheap'])]:
        for arm in ('cheap','jev'):
            if name==arm:continue
            comparisons[arm+'_minus_'+name]=inheritance.transitions(left,[r for r in rows if r['method']==arm])
    abstention=lambda text:normalize_abstention(text) in POLICY['abstention_exact_set']
    strata=inheritance.descriptive_strata(rows,requests,score['wrappers'],abstention)
    cold=cold_accounting(spec,phase,index,qa_jobs,snapshot)
    snapshot.unchanged()
    require(all(Path(p).is_file() and sha(p)==h for p,h in phase['input_sha256'].items()),'Original audited QA changed')
    return {'schema':SCHEMA,'created_utc':now(),'declared_arms':list(ARMS),'planned_per_arm':1540,'planned_total':4620,
        'source_manifest':snapshot.pointer(source_path),'answer_protocol':source['answer_protocol'],
        'audit_status':'no_invalid_artifacts','audit_errors':[],'views':{n:aggregate(v,bootstrap_samples=bootstrap_samples) for n,v in views.items()},
        'slots_by_view':views,'r03_reference_slots':reference,'paired_comparisons':comparisons,
        'r03_reference_metrics':{'denominator':1540,**{m:sum(r['scores'][m] for r in reference)/1540 for m in inheritance.METRICS},
            'judge_correct_count':sum(r['scores']['judge_correct']==1 for r in reference)},
        'oracle_OFFLINE_DIAGNOSTIC_ONLY':{'correct':1540-oracle['0_to_0'],'both_correct':oracle['1_to_1'],
            'one_correct':oracle['0_to_1']+oracle['1_to_0'],'both_wrong':oracle['0_to_0'],
            'selection_input':False,'not_a_success_guarantee':True},
        'group_metrics':{f:grouped(rows,f) for f in ('category','conversation')},
        'stopping_by_view':{n:inheritance.stopping(v) for n,v in views.items()},
        **strata,'choice_response_validity':choice_validity_report(score['audits']),
        'new_choice_usage':score['usage'],'choice_intent_inventory':score['intent_inventory'],
        'cold_upstream_accounting':cold,'technical_recovery':{'eligible_qids':score['eligible_qids'],
            'eligible_count':len(score['eligible_qids']),'performed':False,'phase_count':1,
            'no_eligible_recovery':not score['eligible_qids'],
            'final_view_semantics':'No recovery overlay supplied; retained firstpass fallback. If eligible failures actually require recovery, use a separately versioned reviewed adapter.'},
        'phase_sources':score['phase_sources'],'original_raw_audit':{'manifest':spec['source_refs']['r03_qa_manifest'],
            'audited_slots':len(old_rows),'source_candidate_slots':len(index),'new_model_calls':0,
            'replay':'frozen R03 Reader+Judge+compat+QA signatures/raw/intent/ledger and inherited exact-request proof chains'},
        'input_sha256':{**phase['input_sha256'],**snapshot.files},'summary_code_sha256':{p:sha(ROOT/p) for p in CODE},
        'model_calls_in_summary':0,'native_clients':0,'new_reader_calls':0,'new_judge_calls':0,
        'limitations':['Memory-relative Choice is not source truth or calibrated correctness.',
            'Original actual Reader artifacts are referenced; no answer was generated from the shared union40.',
            'Finite cheap exact-abstention heuristic is not a full abstention detector.',
            'Reader/selector historical usage is required upstream cost, not free inheritance or equal-budget N0.',
            'R03+answer choice is a composite and cannot count as second independent successful component.',
            'Judge flips are official model labels; same-semantic strings can be judged inconsistently.',
            'Oracle1200 is a posthoc bound, never a selector input or guarantee.',
            'Another round requires actual three-reviewer post-result discussion.']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firstpass','--run-dir',dest='firstpass',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    a=p.parse_args();require(a.bootstrap_samples>0,'Positive bootstrap count required')
    with prior.original.historical_guard.historical.offline():
        value=summarize(a.firstpass,source_path=a.source,bootstrap_samples=a.bootstrap_samples)
    save_new(a.output,value)
    print(json.dumps({'output':str(a.output),'audit_status':value['audit_status'],'model_calls':0}))

if __name__=='__main__':main()
