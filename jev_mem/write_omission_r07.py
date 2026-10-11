"""Pure bounded write-opportunity builders, parsers and admission. No API clients."""
from collections.abc import Mapping
from copy import deepcopy
import json
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores
from jev_mem.gemini38_provider_v1 import generation_body, response_text

PROTOCOL_SHA256='bcd1bfec608e23053b791c88064a35b249936351da0db8e4dbbd34a6250a36db'
ARMS=('baseline','cheap','jev')
POLICY={'version':'write-omission-r07-v1','population':1540,'sessions':272,
 'original_facts':1637,'candidate_cap':2,'statement_utf8_byte_cap':768,
 'citation_cap':4,'quote_utf8_byte_cap':1024,'proposal_utf8_byte_cap':16384,
 'signature_utf8_byte_cap':131072,'support_min':.80,'novelty_min':.50,
 'inherited_entity_orphan_allowlist':[],'calibration':'raw_uncalibrated_AND',
 'proposer_feedback':False,'query_conditioned':False}
STATE_FIELDS={'session_id','observation_date','current_source','historical_source','baseline_prefix'}
TURN_FIELDS={'turn_id','speaker','session_id','observation_date','text','blip_caption'}


def canonical_json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _nonempty(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError('Nonempty '+name+' required')


def validate_state(state):
    if not isinstance(state,dict) or set(state)!=STATE_FIELDS:raise ValueError('Exact source state whitelist required')
    _nonempty(state['session_id'],'session ID');_nonempty(state['observation_date'],'observation date')
    byid={}
    for key in ('current_source','historical_source'):
        if not isinstance(state[key],list):raise ValueError('Explicit source list required')
        for turn in state[key]:
            if (not isinstance(turn,dict)or not {'turn_id','speaker','session_id','observation_date','text'}<=set(turn)
                or not set(turn)<=TURN_FIELDS or any(not isinstance(v,str) for v in turn.values())):
                raise ValueError('Exact source projection fields required')
            for name in ('turn_id','speaker','session_id','observation_date'):_nonempty(turn[name],name)
            if turn['turn_id'] in byid:raise ValueError('Duplicate source turn ID')
            if key=='current_source'and(turn['session_id']!=state['session_id']or turn['observation_date']!=state['observation_date']):
                raise ValueError('Current session/date differs')
            if key=='historical_source'and turn['session_id']==state['session_id']:raise ValueError('Historical turn cannot be current')
            byid[turn['turn_id']]=turn
    if not state['current_source']:raise ValueError('Current source must exist')
    prefix=state['baseline_prefix']
    if not isinstance(prefix,list):raise ValueError('Explicit baseline prefix required')
    ids=[]
    for fact in prefix:
        if not isinstance(fact,dict)or set(fact)!={'id','text','observation_date'}:raise ValueError('Exact prefix whitelist required')
        _nonempty(fact['id'],'fact ID');_nonempty(fact['text'],'fact text')
        if not isinstance(fact['observation_date'],str):raise ValueError('Original prefix date required')
        ids.append(fact['id'])
    if len(set(ids))!=len(ids):raise ValueError('Unique prefix IDs required')
    return byid


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def parse_proposals(text,state,protocol=None):
    byid=validate_state(state)
    if not isinstance(text,str):raise ValueError('Actual Gemini response text required')
    stripped=text.strip()
    if len(stripped.encode('utf8'))>16384:raise ValueError('Proposal response byte cap exceeded')
    raw=json.loads(stripped,object_pairs_hook=_unique_object,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(raw,dict)or set(raw)!={'memories'}or not isinstance(raw['memories'],list)or len(raw['memories'])>2:
        raise ValueError('Exact zero-to-two memories schema required')
    validated=[];current={t['turn_id']for t in state['current_source']}
    # All objects validate before deduplication. Never cherry-pick a malformed batch.
    for i,item in enumerate(raw['memories']):
        if not isinstance(item,dict)or set(item)!={'statement','evidence'}:raise ValueError('Exact memory fields required')
        _nonempty(item['statement'],'statement');statement=item['statement'].strip()
        if len(statement.encode('utf8'))>768:raise ValueError('Statement byte cap exceeded')
        evidence=item['evidence']
        if not isinstance(evidence,list)or not 1<=len(evidence)<=4:raise ValueError('One-to-four exact citations required')
        seen=set();has_current=False
        for cite in evidence:
            if not isinstance(cite,dict)or set(cite)!={'turn_id','field','quote'}:raise ValueError('Exact citation fields required')
            if any(not isinstance(v,str)for v in cite.values()):raise ValueError('Citation fields must be strings')
            if cite['field'] not in ('text','blip_caption')or cite['turn_id']not in byid:raise ValueError('Located allowed source required')
            _nonempty(cite['quote'],'quote')
            if len(cite['quote'].encode('utf8'))>1024 or cite['quote']not in byid[cite['turn_id']].get(cite['field'],''):
                raise ValueError('Exact bounded source substring required')
            key=canonical_json(cite)
            if key in seen:raise ValueError('Duplicate citation object')
            seen.add(key);has_current|=cite['turn_id']in current
        if not has_current:raise ValueError('Current-session citation required')
        validated.append({'statement':statement,'evidence':deepcopy(evidence),'proposal_index':i})
    prefix={f['text']for f in state['baseline_prefix']};seen=set();candidates=[];exclusions=[]
    for item in validated:
        text=item['statement'];reason='duplicate_statement'if text in seen else 'literal_redundant'if text in prefix else None
        seen.add(text)
        if reason:exclusions.append({'proposal_index':item['proposal_index'],'statement':text,'reason':reason})
        else:candidates.append(item)
    return {'candidates':candidates,'exclusions':exclusions,'raw_count':len(validated),'retained_count':len(candidates)}


def parse_proposal_response(response,state,protocol=None):
    return parse_proposals(response_text(response),state,protocol)


def stable_candidates(parsed,query_id,state):
    return [{**deepcopy(c),'candidate_id':f'{query_id}_m{c["proposal_index"]:02d}',
             'source_session_id':state['session_id'],'observation_date':state['observation_date']}
            for c in parsed['candidates']]


def build_proposer_request(state,protocol):
    validate_state(state)
    prompt=protocol['proposer']['prompt'].replace('{input_json}',canonical_json(state))
    return {'provider':'gemini','model':'gemini-3.8-flash','operation':'generateContent',
            'body':generation_body(prompt,system=None,json_output=False,max_output=8192)}


def expected_keys(candidates):
    if not isinstance(candidates,list)or not 1<=len(candidates)<=2:raise ValueError('One-to-two candidates required')
    return {f'c{i:02d}_{kind}'for i in range(len(candidates))for kind in ('support','novelty')}


def build_request(state,candidates,protocol):
    validate_state(state);expected_keys(candidates);questions={}
    for i,c in enumerate(candidates):
        candidate_json=canonical_json({'statement':c['statement'],'evidence':c['evidence']})
        for kind in ('support','novelty'):
            questions[f'c{i:02d}_{kind}']={'type':'noul','instructions':protocol['jev'][kind+'_prompt'].replace('{candidate_json}',candidate_json)}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone',
            'body':{'model':'jev-1.13.0','state':deepcopy(state),'questions':questions}}


def admit_candidates(candidates,scores,*,status='complete'):
    if not candidates:return {'accepted_ids':[],'reason':'no_candidates','gate_trace':[]}
    keys=expected_keys(candidates)
    if status=='unknown':raise ValueError('Unknown admission state blocks')
    if status not in ('complete','failed','not_dispatched'):raise ValueError('Explicit terminal status required')
    if status!='complete':
        if scores is not None:raise ValueError('Unavailable scores cannot contain numbers')
        return {'accepted_ids':[],'reason':'terminal_unavailable','gate_trace':[]}
    import math
    if (not isinstance(scores,Mapping)or set(scores)!=keys or any(isinstance(p,bool)or not isinstance(p,(float,int))
        or not math.isfinite(p)or not 0<=p<=1 for p in scores.values())):raise ValueError('Complete bounded nonboolean scores required')
    accepted=[];trace=[]
    for i,c in enumerate(candidates):
        s,n=scores[f'c{i:02d}_support'],scores[f'c{i:02d}_novelty'];yes=s>=.80 and n>=.50
        trace.append({'candidate_id':c['candidate_id'],'support':s,'novelty':n,'support_pass':s>=.80,'novelty_pass':n>=.50,'accepted':yes})
        if yes:accepted.append(c['candidate_id'])
    return {'accepted_ids':accepted,'reason':'frozen_raw_AND','gate_trace':trace}
