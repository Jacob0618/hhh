"""Pure source support locator S x query-dependent closure C. No I/O/API/gold.

S uses one frozen fact scope, never the question or a query's20-scope corpus.
C acts after the original top4 admission; final ranking retains R05 semantics.
Raw Noul scores are uncalibrated; partial_support_locator is not certification.
"""
from copy import deepcopy
import math
from collections.abc import Mapping
from jev_mem import source_append_r05 as prior
from jev_mem.evidence_exchange_r01 import digest, serialized_signature, parse_scores

VERSION = 'source-support-stack-v1'
MODEL = 'jev-1.13.0'
ARMS = ('s0c0', 's1c0', 's0c1', 's1c1')
POLICY = {'version': VERSION, 'support_center_cap': 8, 'questions_per_request': 4,
    'support_threshold_strict_gt': .5, 'closure_threshold_strict_gt': .5,
    'closure_extra_left': 1, 'closure_extra_right': 1, 'closure_turn_cap': 5,
    'candidate_cap': 4, 'attachment_cap': 2, 'attachment_utf8_byte_cap': 4096,
    'signature_utf8_byte_cap': 131072, 'support_scope': 'single_fact_current_input_scope',
    'query_corpus': 'full_original20_current_scope_union',
    'support_tie': 'original_center_position', 'closure_tie': 'fewest_new_turns_then_variant_order',
    'closure_collision': 'earliest_original_candidate_merge_aliases_no_refill',
    'support_label': 'partial_support_locator', 'rank_failure': 'unchanged_original20',
    'calibration': 'uncalibrated_operational_majority_threshold_not_truth_probability'}
SUPPORT_PREDICATE = (
    'Does this quoted source window directly support at least one substantive atomic claim '
    'in the displayed original memory, with the same subject, relation and material '
    'conditions? Preserve speaker attribution, negation, quantities, planned versus '
    'completed status and temporal wording; do not invent a bridge between events. '
    'A related topic or a question without its answer is insufficient. Judge only this '
    'memory and source window. The source observation date is not necessarily the event '
    'date, and machine captions are annotations rather than speaker assertions. This '
    'judgment locates possible partial support; it does not certify the whole memory.')
CLOSURE_PREDICATE = (
    'Compared with the displayed base source window, do the added complete adjacent turns '
    'supply an explicit answer, referent, speaker attribution, condition or temporal '
    'relationship needed to interpret evidence requested by the question and missing or '
    'unresolved in the base window? Added text must close a concrete evidence relation, '
    'not merely mention a related topic. Judge only the displayed original memory, base '
    'window and proposed extension. Preserve negation and planned versus completed status; '
    'do not invent event bridges. Source observation date is not event date and machine '
    'captions are annotations. A larger window alone is not a benefit.')
LOCAL_PREDICATE = (
    'Does this quoted source context provide an explicit detail, relation, condition, '
    'speaker attribution, or temporal wording requested by the question that is absent '
    'or less explicit in this one original memory? A partial requested list item may '
    'count. Judge only the displayed source and this memory; do not assume completeness '
    'of the other19 memories or full conversation. Do not invent a bridge or treat source '
    'observation date or machine caption as a speaker assertion/event date. Related topic '
    'alone is insufficient.')
PROTOCOL = {'schema': VERSION+'-protocol', 'policy': POLICY,
    'support_predicate': SUPPORT_PREDICATE, 'closure_predicate': CLOSURE_PREDICATE,
    'local_predicate': LOCAL_PREDICATE}


def _protocol(protocol):
    p = PROTOCOL if protocol is None else protocol
    if p != PROTOCOL:
        raise ValueError('Frozen source-stack protocol differs')
    return p


def _fact(fact):
    if (not isinstance(fact, dict) or set(fact) != {'id','text','observation_date'}
        or not isinstance(fact['id'],str) or not fact['id']
        or not isinstance(fact['text'],str) or not fact['text'].strip()
        or (fact['observation_date'] is not None and not isinstance(fact['observation_date'],str))):
        raise ValueError('Exact canonical fact only; no QA/gold fields')
    return fact


def _turns(turns):
    if not isinstance(turns,list): raise ValueError('Source turn list required')
    byid={}
    for i,t in enumerate(turns):
        if not isinstance(t,dict):raise ValueError('Source turn mapping required')
        v=t.get('view',{})
        if (not isinstance(t,dict) or not isinstance(t.get('source_turn_id'),str)
            or not t['source_turn_id'] or t['source_turn_id'] in byid
            or type(t.get('global_index')) is not int or t['global_index']!=i
            or not isinstance(t.get('session_id'),str) or not t['session_id']
            or not isinstance(t.get('observation_date'),str)
            or set(v)!= {'source_turn_id','speaker_text','machine_captions'}
            or v['source_turn_id']!=t['source_turn_id'] or not isinstance(v['speaker_text'],str)
            or not isinstance(v['machine_captions'],list)
            or any(not isinstance(c,str) for c in v['machine_captions'])
            or not all(k in t for k in ('source','json_pointer','content_sha256','projection'))
            or t['projection'].get('speaker_text')!=v['speaker_text']
            or t['projection'].get('machine_captions')!=v['machine_captions']):
            raise ValueError('Exact ordered projected turns with provenance required')
        byid[t['source_turn_id']]=t
    return byid


def _scope(scope,byid):
    if scope is None:return None
    if not isinstance(scope,Mapping): raise ValueError('Scope mapping required')
    session=scope.get('session_id');ids=scope.get('source_turn_ids')
    if (not isinstance(session,str) or not isinstance(ids,list)
        or any(not isinstance(t,str) or t not in byid for t in ids) or len(set(ids))!=len(ids)
        or any(byid[t]['session_id']!=session for t in ids)):
        raise ValueError('Current scope must bind unique turns in its declared session')
    # Historical/existing-memory metadata is never a permission expansion.
    return {'session_id':session,'source_turn_ids':list(ids)}


def _source_binding(scope,turns):
    allowed=set(scope['source_turn_ids']) if scope else set()
    return digest([{k:deepcopy(t[k]) for k in ('source_turn_id','session_id','observation_date',
        'global_index','view','source','json_pointer','content_sha256','projection')}
        for t in turns if t['source_turn_id'] in allowed])


def _packet(fact,slot,center,included,conversation,*,qscore=0.,fscore=0.,boundaries=None):
    view={'session_id':center['session_id'],'source_observation_date':center['observation_date'],
        'turns':[deepcopy(t['view']) for t in included]}
    text=prior.attachment(view)
    origin={'anchor_slot':slot,'anchor_id':fact['id'],'center_turn_id':center['source_turn_id'],
        'center_global_index':center['global_index'],'query_bm25':qscore,'anchor_bm25':fscore}
    return {'packet_id':digest({'conversation':conversation,'view':view}),'packet_view':view,
        'attachment':text,'attachment_utf8_bytes':len(text.encode('utf-8')),
        'origin':origin,'origin_aliases':[deepcopy(origin)],'boundary':deepcopy(boundaries or {}),
        'turn_provenance':[{k:deepcopy(t[k]) for k in ('source_turn_id','source','json_pointer',
            'content_sha256','projection')} for t in included]}


def _base_packet(fact,slot,center,scope,turns,conversation,**scores):
    included=[];markers={};allowed=set(scope['source_turn_ids'])
    for side,offset in (('left',-1),('center',0),('right',1)):
        i=center['global_index']+offset
        t=turns[i] if 0<=i<len(turns) else None
        if t is None or t['session_id']!=center['session_id']:
            markers[side]='boundary_missing'
        elif t['source_turn_id'] not in allowed: markers[side]='scope_excluded'
        else: included.append(t);markers[side]='included'
    return _packet(fact,slot,center,included,conversation,boundaries=markers,**scores)


def _request(state,questions,mapping,batch):
    signature={'provider':'jev','model':MODEL,'operation':'systemone',
        'body':{'model':MODEL,'state':deepcopy(state),'questions':deepcopy(questions)}}
    size=len(serialized_signature(signature))
    return {'batch_index':batch,'signature':signature,'key_to_window_index':dict(mapping),
        'signature_sha256':digest(signature),'signature_utf8_bytes':size,
        'dispatch_eligible':size<=POLICY['signature_utf8_byte_cap'],
        'skip_reason':None if size<=POLICY['signature_utf8_byte_cap'] else 'signature_oversize'}


def validate_request(signature,*,allow_oversize=False):
    if (not isinstance(signature,dict) or set(signature)!= {'provider','model','operation','body'}
        or signature['provider']!='jev' or signature['model']!=MODEL or signature['operation']!='systemone'):
        raise ValueError('Pinned pure systemone signature required')
    body=signature['body']
    if (not isinstance(body,dict) or set(body)!= {'model','state','questions'} or body['model']!=MODEL
        or not isinstance(body['state'],dict) or not isinstance(body['questions'],dict)
        or not 1<=len(body['questions'])<=4):raise ValueError('Complete bounded Noul body required')
    state=body['state'];questions=body['questions'];prefix={k[0] for k in questions if isinstance(k,str)}
    if len(prefix)!=1 or next(iter(prefix)) not in ('s','c','f'):raise ValueError('Known question keys only')
    stage=next(iter(prefix));expected_state={'s':{'original_memory'},'c':{'query','original_memory','base_source_attachment'},'f':{'query'}}[stage]
    if set(state)!=expected_state:raise ValueError('Model state whitelist differs; no gold/QA')
    if 'original_memory' in state:
        m=state['original_memory']
        if (not isinstance(m,dict) or set(m)!= {'text','observation_date'} or not isinstance(m['text'],str)
            or (m['observation_date'] is not None and not isinstance(m['observation_date'],str))):
            raise ValueError('Memory projection whitelist differs')
    if 'query' in state:
        q=state['query']
        if (not isinstance(q,dict) or set(q)!= {'text','reference_date'} or not isinstance(q['text'],str)
            or not q['text'].strip() or (q['reference_date'] is not None and not isinstance(q['reference_date'],str))):
            raise ValueError('Question projection whitelist differs')
    if stage=='c' and not isinstance(state['base_source_attachment'],str):raise ValueError('Base source text required')
    predicate={'s':SUPPORT_PREDICATE,'c':CLOSURE_PREDICATE,'f':LOCAL_PREDICATE}[stage]
    validkeys={f'{stage}{i:02d}' for i in range(8 if stage=='s' else 3 if stage=='c' else 4)}
    if not set(questions)<=validkeys:raise ValueError('Question key outside frozen bound')
    for row in questions.values():
        if (not isinstance(row,dict) or set(row)!= {'type','instructions'} or row['type']!='noul'
            or not isinstance(row['instructions'],str) or not row['instructions'].startswith(predicate+'\n')):
            raise ValueError('Frozen predicate and Noul fields required')
    if not allow_oversize and len(serialized_signature(signature))>131072:raise ValueError('Signature oversize; never truncate')
    return stage


def parse_response(response,signature):
    validate_request(signature)
    return parse_scores(response,signature)


def build_support_plan(fact,scope,turns,*,conversation,protocol=None):
    _protocol(protocol);_fact(fact);byid=_turns(turns);scope=_scope(scope,byid)
    corpus=[t for t in turns if scope and t['source_turn_id'] in set(scope['source_turn_ids'])]
    scores=prior.stable_bm25(fact['text'],[prior.document(t) for t in corpus]) if corpus else []
    choices=[];excluded=[]
    for t,score in zip(corpus,scores):
        c=_base_packet(fact,0,t,scope,turns,conversation,fscore=score)
        if c['attachment_utf8_bytes']>4096:excluded.append({'center':t['source_turn_id'],'reason':'whole_attachment_oversize'})
        else:choices.append(c)
    choices.sort(key=lambda c:(-c['origin']['anchor_bm25'],c['origin']['center_global_index']))
    windows=choices[:8];requests=[]
    for start in range(0,len(windows),4):
        mapping={f's{i:02d}':i for i in range(start,min(start+4,len(windows)))}
        questions={k:{'type':'noul','instructions':SUPPORT_PREDICATE+'\nDisplayed source window:\n'+prior.canonical({'source_attachment':windows[i]['attachment']})} for k,i in mapping.items()}
        requests.append(_request({'original_memory':{k:fact[k] for k in ('text','observation_date')}},questions,mapping,len(requests)))
    return {'schema':VERSION+'-support-plan','conversation':conversation,'fact':deepcopy(fact),'scope':deepcopy(scope),
        'fact_sha256':digest(fact),'scope_sha256':digest(scope),'source_sha256':_source_binding(scope,turns),
        'protocol_sha256':digest(PROTOCOL),'windows':windows,'requests':requests,'exclusions':excluded,
        'eligible_window_count_before_cap':len(choices),'support_granularity':'at_least_one_substantive_atomic_claim',
        'scope_semantics':'input_exposure_is_not_verified_fact_support'}


def _scores(scores,keys,status,requests):
    if status not in ('complete','failed','unknown','not_dispatched'):raise ValueError('Explicit score status required')
    if status!='complete':
        if scores is not None:raise ValueError('Unavailable scores must not contain partial or invented values')
        return None
    if not all(r['dispatch_eligible'] for r in requests):raise ValueError('Ineligible plan cannot have complete scores')
    if not isinstance(scores,Mapping) or set(scores)!=set(keys):raise ValueError('Exactly all batch scores required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise ValueError('Finite nonboolean probabilities in[0,1] required')
    return dict(scores)


def resolve_support(plan,scores,*,score_status='complete'):
    for r in plan['requests']:validate_request(r['signature'],allow_oversize=True)
    keys=[k for r in plan['requests'] for k in r['key_to_window_index']]
    values=_scores(scores,keys,score_status,plan['requests'])
    winner=None
    if values and max(values.values())>.5:
        winner=min(range(len(plan['windows'])),key=lambda i:(-values[f's{i:02d}'],plan['windows'][i]['origin']['center_global_index']))
    reason=('partial_support_locator' if winner is not None else 'no_eligible_support_windows' if not keys
        else 'unavailable_support_scores' if values is None else 'no_support_score_strictly_above_half')
    return {'schema':VERSION+'-support-sidecar','plan_sha256':digest(plan),'fact_sha256':plan['fact_sha256'],
        'scope_sha256':plan['scope_sha256'],'source_sha256':plan['source_sha256'],'protocol_sha256':plan['protocol_sha256'],
        'score_status':score_status,'scores':values,'chosen_window_index':winner,
        'center_turn_id':plan['windows'][winner]['origin']['center_turn_id'] if winner is not None else None,
        'reason':reason,'support_label':'partial_support_locator' if winner is not None else 'unknown',
        'support_granularity':plan['support_granularity'],'certified_fact_support':False,
        'input_exposure_is_not_verified_support':True,'unknown_does_not_delete_fact':True}


def _validate_sidecar(sidecar,plan):
    if not isinstance(sidecar,dict) or sidecar!=resolve_support(plan,sidecar.get('scores'),score_status=sidecar.get('score_status')):
        raise ValueError('Support sidecar does not replay from current exact fact/scope/source plan')


def build_candidates(question,records,scopes,turns,*,conversation,enable_support=False,support_sidecars=None,protocol=None):
    _protocol(protocol);prior.validate_original(records);byid=_turns(turns)
    if len(scopes)!=20 or type(enable_support)is not bool:raise ValueError('Twenty scopes and explicit support switch required')
    canonical_scopes=[_scope(s,byid) for s in scopes]
    old=prior.build_candidates(question,records,canonical_scopes,turns,conversation=conversation)
    diagnostics=[]
    if not enable_support:
        result=deepcopy(old);result['support_locator']={'enabled':False,'per_fact':[]};return result
    if support_sidecars is None:support_sidecars={}
    if not isinstance(support_sidecars,Mapping):raise ValueError('Fact ID to support sidecar mapping required')
    corpus=[t for t in turns if t['source_turn_id'] in set(old['corpus_turn_ids'])]
    ci={t['source_turn_id']:i for i,t in enumerate(corpus)};per_fact=[];excluded=deepcopy(old['exclusions'])
    for slot,(fact,scope) in enumerate(zip(records,canonical_scopes)):
        allowed=set(scope['source_turn_ids']) if scope else set();options=[]
        for t in corpus:
            if t['source_turn_id'] not in allowed:continue
            idx=ci[t['source_turn_id']];q=old['query_bm25_scores'][idx];f=old['anchor_bm25_scores'][slot][idx]
            if q<=0 or f<=0:continue
            c=_base_packet(fact,slot,t,scope,turns,conversation,qscore=q,fscore=f)
            if c['attachment_utf8_bytes']<=4096:options.append(c)
        options.sort(key=lambda c:(-c['origin']['query_bm25'],-c['origin']['anchor_bm25'],c['origin']['center_global_index']))
        fallback=options[0] if options else None;chosen=fallback;sidecar=support_sidecars.get(fact['id'])
        reason='support_sidecar_absent_keep_original_locator';used=False
        if sidecar is not None:
            plan=build_support_plan(fact,scope,turns,conversation=conversation);_validate_sidecar(sidecar,plan)
            tid=sidecar['center_turn_id']
            if tid is None:reason='support_unknown_keep_original_locator'
            else:
                match=next((c for c in options if c['origin']['center_turn_id']==tid),None)
                if match is None:reason='support_center_not_dual_positive_or_oversize_keep_original_locator'
                else:chosen=match;used=True;reason='partial_support_locator_dual_positive'
        diagnostics.append({'slot':slot,'fact_id':fact['id'],'reason':reason,'support_used':used,
            'chosen_center':chosen['origin']['center_turn_id'] if chosen else None,
            'original_locator_center':fallback['origin']['center_turn_id'] if fallback else None})
        if chosen:per_fact.append(chosen)
    groups={}
    for c in per_fact:groups.setdefault(c['packet_id'],[]).append(c)
    candidates=[]
    for entries in groups.values():
        entries.sort(key=lambda c:(-c['origin']['anchor_bm25'],c['origin']['anchor_slot'],c['origin']['center_global_index']))
        c=deepcopy(entries[0]);c['origin_aliases']=[deepcopy(e['origin']) for e in entries];candidates.append(c)
    candidates.sort(key=lambda c:(-c['origin']['query_bm25'],-c['origin']['anchor_bm25'],c['origin']['anchor_slot'],c['origin']['center_global_index']))
    result={**deepcopy(old),'candidates':candidates[:4],'per_fact_candidate_count':len(per_fact),
        'deduplicated_candidate_count':len(groups),'exclusions':excluded,
        'support_locator':{'enabled':True,'per_fact':diagnostics,'support_used_count':sum(d['support_used'] for d in diagnostics),
            'fallback_count':sum(not d['support_used'] for d in diagnostics),
            'candidate_universe_can_change':True,'full_query_corpus_unchanged':True}}
    prior.validate_candidates(result['candidates'],records)
    return result


def build_closure_plan(question,reference_date,records,candidate,scope,turns,*,conversation,protocol=None):
    _protocol(protocol);prior.validate_original(records);prior.validate_candidates([candidate],records)
    byid=_turns(turns);scope=_scope(scope,byid);origin=candidate['origin'];fact=records[origin['anchor_slot']]
    if scope is None:raise ValueError('Candidate closure requires its current scope')
    allowed=set(scope['source_turn_ids']);ids=[t['source_turn_id'] for t in candidate['packet_view']['turns']]
    if (not ids or any(t not in allowed for t in ids) or ids!=sorted(ids,key=lambda tid:byid[tid]['global_index'])):
        raise ValueError('Base packet must be ordered in the consuming anchor current scope')
    center=byid.get(origin['center_turn_id'])
    expected=_base_packet(fact,origin['anchor_slot'],center,scope,turns,conversation,
        qscore=origin['query_bm25'],fscore=origin['anchor_bm25']) if center else None
    if expected is None or any(candidate[k]!=expected[k] for k in ('packet_view','attachment','attachment_utf8_bytes','packet_id','turn_provenance')):
        raise ValueError('Closure base must replay exact original bounded window')
    first=byid[ids[0]]['global_index'];last=byid[ids[-1]]['global_index'];variants=[];excluded=[]
    for order,(name,left,right) in enumerate((('left',1,0),('right',0,1),('both',1,1))):
        indices=list(range(first-left,last+right+1))
        if any(i<0 or i>=len(turns) or turns[i]['session_id']!=scope['session_id'] or turns[i]['source_turn_id'] not in allowed for i in indices):
            excluded.append({'variant':name,'reason':'session_boundary_or_scope_excluded'});continue
        included=[turns[i] for i in indices]
        if len(included)>5:raise ValueError('Frozen closure at most five complete turns')
        c=_packet(fact,origin['anchor_slot'],center,included,conversation,qscore=origin['query_bm25'],
            fscore=origin['anchor_bm25'],boundaries=candidate.get('boundary'))
        c['origin_aliases']=deepcopy(candidate['origin_aliases'])
        if c['attachment_utf8_bytes']>4096:
            excluded.append({'variant':name,'reason':'whole_attachment_oversize','utf8_bytes':c['attachment_utf8_bytes']});continue
        variants.append({'name':name,'variant_order':order,'new_turn_count':len(included)-len(ids),'candidate':c})
    mapping={f'c{i:02d}':i for i in range(len(variants))}
    state={'query':{'text':question,'reference_date':reference_date},
        'original_memory':{k:fact[k] for k in ('text','observation_date')},'base_source_attachment':candidate['attachment']}
    questions={k:{'type':'noul','instructions':CLOSURE_PREDICATE+'\nProposed complete extension:\n'+prior.canonical({'source_attachment':variants[i]['candidate']['attachment']})} for k,i in mapping.items()}
    requests=[_request(state,questions,mapping,0)] if questions else []
    for r in requests:validate_request(r['signature'],allow_oversize=True)
    return {'schema':VERSION+'-closure-plan','conversation':conversation,'base':deepcopy(candidate),
        'variants':variants,'requests':requests,'exclusions':excluded,'protocol_sha256':digest(PROTOCOL),
        'scope_sha256':digest(scope),'source_sha256':_source_binding(scope,turns)}


def resolve_closure(plan,scores,*,score_status='complete'):
    for r in plan['requests']:validate_request(r['signature'],allow_oversize=True)
    keys=[k for r in plan['requests'] for k in r['key_to_window_index']];values=_scores(scores,keys,score_status,plan['requests'])
    winner=None
    if values and max(values.values())>.5:
        winner=min(range(len(plan['variants'])),key=lambda i:(-values[f'c{i:02d}'],plan['variants'][i]['new_turn_count'],plan['variants'][i]['variant_order']))
    candidate=deepcopy(plan['base'] if winner is None else plan['variants'][winner]['candidate'])
    return {'schema':VERSION+'-closure-result','plan_sha256':digest(plan),'candidate':candidate,
        'diagnostic':{'score_status':score_status,'scores':values,'chosen_variant_index':winner,
            'reason':'query_adjacent_relation_closure' if winner is not None else 'no_eligible_extensions' if not keys
            else 'unavailable_closure_scores_keep_base' if values is None else 'no_closure_score_strictly_above_half_keep_base',
            'added_turn_count':0 if winner is None else plan['variants'][winner]['new_turn_count'],
            'base_packet_id':plan['base']['packet_id'],'effective_packet_id':candidate['packet_id'],
            'semantic_support_certified':False}}


def apply_closures(locator,resolved,records,*,closure_plans):
    prior.validate_original(records);prior.validate_candidates(locator['candidates'],records)
    if (not isinstance(resolved,list) or not isinstance(closure_plans,list)
        or len(resolved)!=len(locator['candidates']) or len(closure_plans)!=len(resolved)):
        raise ValueError('One closure plan/result per admitted candidate required')
    candidates=[];byid={};collisions=[]
    for i,(base,result) in enumerate(zip(locator['candidates'],resolved)):
        plan=closure_plans[i]
        if (plan['base']!=base or result!=resolve_closure(plan,result['diagnostic']['scores'],
            score_status=result['diagnostic']['score_status'])):
            raise ValueError('Complete closure result must replay from its bound plan and scores')
        c=deepcopy(result['candidate'])
        if (result['schema']!=VERSION+'-closure-result' or result['diagnostic']['base_packet_id']!=base['packet_id']
            or c['origin']!=base['origin']):raise ValueError('Closure may not reorder origins or admit new centers')
        prior.validate_candidates([c],records)
        if c['packet_id'] in byid:
            first=byid[c['packet_id']];kept=candidates[first]
            for alias in c['origin_aliases']:
                if alias not in kept['origin_aliases']:kept['origin_aliases'].append(deepcopy(alias))
            collisions.append({'dropped_original_index':i,'kept_effective_index':first,'duplicate_attachment_utf8_bytes':c['attachment_utf8_bytes']})
        else:byid[c['packet_id']]=len(candidates);candidates.append(c)
    prior.validate_candidates(candidates,records)
    return {**deepcopy(locator),'candidates':candidates,'closure':{'enabled':True,
        'pre_closure_candidate_count':len(resolved),'unique_candidate_count':len(candidates),
        'collisions':collisions,'per_candidate':[deepcopy(r['diagnostic']) for r in resolved],
        'top4_admission_not_repeated':True,'no_outside_refill':True}}


def build_rank_request(question,reference_date,records,candidates,protocol=None):
    _protocol(protocol)
    if len(candidates)<=2:return None
    signature=prior.build_request(question,reference_date,records,candidates,{'jev':{'predicate':LOCAL_PREDICATE}})
    validate_request(signature,allow_oversize=True)
    return signature


def choose_rank(records,candidates,scores,*,score_status='complete'):
    return prior.choose_jev(records,candidates,scores,score_status=score_status)


def selected_facts(records,candidates,decision):
    return prior.selected_facts(records,candidates,decision)
