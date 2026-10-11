"""Pure R09 existing-ID qualifier repair: bounded source, strict parsing, no APIs."""
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores
from jev_mem.gemini38_provider_v1 import generation_body, response_text
from jev_mem import write_omission_r07 as original

PROTOCOL_SHA256='59908bea9a269f5c79bb30a40bf30df165362bc5f5e291101f3e63b8731efc4b'
ARMS=('baseline','cheap','jev')
POLICY={'version':'qualifier-repair-r09-v1','population':1540,'sessions':272,
 'original_facts':1637,'candidate_cap':2,'replacement_utf8_byte_cap':1536,
 'anchor_cap':4,'citation_cap':6,'quote_utf8_byte_cap':1024,
 'proposal_utf8_byte_cap':24576,'signature_utf8_byte_cap':131072,
 'support_min':.80,'preservation_min':.80,'inherited_entity_orphan_allowlist':[],
 'calibration':'raw_uncalibrated_AND','proposer_feedback':False,'query_conditioned':False}
STATE_FIELDS={'session_id','observation_date','current_source','historical_source','current_facts'}
REVISION_FIELDS={'fact_id','replacement_text','edit_kind','anchor_turn_ids','evidence'}
canonical_json=original.canonical_json
_nonempty=original._nonempty
_unique_object=original._unique_object


def validate_state(state):
    if not isinstance(state,dict) or set(state)!=STATE_FIELDS:
        raise ValueError('Exact five-field R09 source state required')
    # Reuse only the original strict source projection validator. The translated
    # prefix is local validation data and never enters a model request.
    translated={k:deepcopy(v) for k,v in state.items() if k!='current_facts'}
    translated['baseline_prefix']=deepcopy(state['current_facts'])
    byid=original.validate_state(translated)
    for fact in state['current_facts']:
        if fact['observation_date']!=state['observation_date']:
            raise ValueError('Current-birth fact observation date differs')
    return byid


def parse_proposals(text,state,protocol=None):
    byid=validate_state(state)
    if not isinstance(text,str):raise ValueError('Actual Gemini text required')
    stripped=text.strip()
    if len(stripped.encode('utf8'))>24576:raise ValueError('Proposal response byte cap exceeded')
    raw=json.loads(stripped,object_pairs_hook=_unique_object,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if (not isinstance(raw,dict) or set(raw)!={'revisions'}
        or not isinstance(raw['revisions'],list) or len(raw['revisions'])>2):
        raise ValueError('Exact zero-to-two revisions schema required')
    facts={f['id']:f for f in state['current_facts']}
    ordinal={f['id']:i for i,f in enumerate(state['current_facts'])}
    current=[t['turn_id'] for t in state['current_source']]
    seen=set();validated=[]
    # Validate the WHOLE response before filtering no-ops or assigning cNN.
    for i,item in enumerate(raw['revisions']):
        if not isinstance(item,dict) or set(item)!=REVISION_FIELDS:
            raise ValueError('Exact revision fields required')
        target=item['fact_id']
        if not isinstance(target,str) or target not in facts or target in seen:
            raise ValueError('Unique current-birth target ID required')
        seen.add(target)
        value=item['replacement_text'];_nonempty(value,'replacement text')
        if len(value.encode('utf8'))>1536:raise ValueError('Replacement byte cap exceeded')
        if item['edit_kind'] not in ('clarify','correct'):raise ValueError('Exact edit_kind required')
        anchor=item['anchor_turn_ids']
        if (not isinstance(anchor,list) or not 1<=len(anchor)<=4
            or any(not isinstance(t,str) or t not in current for t in anchor)):
            raise ValueError('One-to-four current anchor IDs required')
        start=current.index(anchor[0])
        if anchor!=current[start:start+len(anchor)]:
            raise ValueError('Anchor must be contiguous in actual current source order')
        evidence=item['evidence']
        if not isinstance(evidence,list) or not 1<=len(evidence)<=6:
            raise ValueError('One-to-six exact citations required')
        cited_anchor=False
        for cite in evidence:
            if (not isinstance(cite,dict) or set(cite)!={'turn_id','field','quote'}
                or any(not isinstance(v,str) for v in cite.values())):
                raise ValueError('Exact citation fields required')
            if cite['turn_id'] not in byid or cite['field'] not in ('text','blip_caption'):
                raise ValueError('Located allowed source field required')
            _nonempty(cite['quote'],'quote')
            if (len(cite['quote'].encode('utf8'))>1024
                or cite['quote'] not in byid[cite['turn_id']].get(cite['field'],'')):
                raise ValueError('Exact bounded source substring required')
            cited_anchor|=cite['turn_id'] in anchor
        if not cited_anchor:raise ValueError('At least one cited anchor turn required')
        validated.append({**deepcopy(item),'proposal_index':i})
    candidates=[];exclusions=[]
    for item in sorted(validated,key=lambda c:ordinal[c['fact_id']]):
        if item['replacement_text']==facts[item['fact_id']]['text']:
            exclusions.append({**item,'reason':'literal_noop'})
        else:candidates.append(item)
    return {'candidates':candidates,'exclusions':exclusions,
            'raw_count':len(validated),'retained_count':len(candidates)}


def parse_proposal_response(response,state,protocol=None):
    return parse_proposals(response_text(response),state,protocol)


def stable_candidates(parsed,query_id,state):
    validate_state(state)
    facts={f['id']:f for f in state['current_facts']}
    return [{**deepcopy(c),'candidate_id':c['fact_id'],'candidate_key':f'c{i:02d}',
             'original_fact':deepcopy(facts[c['fact_id']]),
             'source_session_id':state['session_id'],'observation_date':state['observation_date']}
            for i,c in enumerate(parsed['candidates'])]


def candidate_payload(candidate):
    return {k:deepcopy(candidate[k]) for k in sorted(REVISION_FIELDS)}


def validate_candidates(state,candidates):
    expected_keys(candidates)
    # Reparse the actual model-content subset to prohibit accidental mutation or
    # no-op reintroduction between proposer audit and scoring.
    parsed=parse_proposals(canonical_json({'revisions':[candidate_payload(c) for c in candidates]}),state)
    replay=stable_candidates(parsed,'unused',state)
    indices=[c.get('proposal_index') for c in candidates]
    if any(type(i) is not int or i not in (0,1) for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('Original proposal ordinals must be unique and bounded')
    for expected,actual in zip(replay,candidates):
        # proposal_index is original generation order; no later replay can infer it.
        expected['proposal_index']=actual.get('proposal_index')
        if expected!=actual:raise ValueError('Canonical target/proposal provenance differs')
    if len(replay)!=len(candidates):raise ValueError('No-op or duplicate candidate reintroduced')


def prompt_template(text):
    """Protocol template escape notation becomes actual LF before interpolation."""
    return text.replace('\\n','\n')


def build_proposer_request(state,protocol):
    validate_state(state)
    prompt=prompt_template(protocol['proposer']['prompt']).replace('{input_json}',canonical_json(state))
    return {'provider':'gemini','model':'gemini-3.8-flash','operation':'generateContent',
            'body':generation_body(prompt,system=None,json_output=False,max_output=8192)}


def expected_keys(candidates):
    if not isinstance(candidates,list) or not 1<=len(candidates)<=2:
        raise ValueError('One-to-two candidates required')
    return {f'c{i:02d}_{axis}' for i in range(len(candidates)) for axis in ('support','preservation')}


def build_request(state,candidates,protocol):
    validate_state(state);validate_candidates(state,candidates);questions={}
    for i,c in enumerate(candidates):
        candidate_json=canonical_json(candidate_payload(c))
        original_json=canonical_json(c['original_fact'])
        for kind in ('support','preservation'):
            prompt=prompt_template(protocol['jev'][kind+'_prompt']).replace('{original_fact_json}',original_json).replace('{candidate_json}',candidate_json)
            questions[f'c{i:02d}_{kind}']={'type':'noul','instructions':prompt}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone',
            'body':{'model':'jev-1.13.0','state':deepcopy(state),'questions':questions}}


def admit_candidates(candidates,scores,*,status='complete'):
    if status=='unknown':raise ValueError('Unknown admission state blocks')
    if status not in ('complete','failed','not_dispatched'):raise ValueError('Explicit terminal status required')
    if not candidates:
        if scores is not None:raise ValueError('No candidates cannot carry scores')
        return {'accepted_ids':[],'reason':'no_candidates','gate_trace':[]}
    keys=expected_keys(candidates)
    if status not in ('complete','failed','not_dispatched'):raise ValueError('Explicit terminal status required')
    if status!='complete':
        if scores is not None:raise ValueError('Unavailable scores cannot contain numbers')
        return {'accepted_ids':[],'reason':'terminal_unavailable','gate_trace':[]}
    if (not isinstance(scores,Mapping) or set(scores)!=keys or any(isinstance(p,bool)
        or not isinstance(p,(float,int)) or not math.isfinite(p) or not 0<=p<=1 for p in scores.values())):
        raise ValueError('Complete bounded nonboolean scores required')
    accepted=[];trace=[]
    for i,c in enumerate(candidates):
        s,p=scores[f'c{i:02d}_support'],scores[f'c{i:02d}_preservation'];yes=s>=.80 and p>=.80
        trace.append({'candidate_id':c['fact_id'],'fact_id':c['fact_id'],'support':s,'preservation':p,
                      'support_pass':s>=.80,'preservation_pass':p>=.80,'accepted':yes})
        if yes:accepted.append(c['fact_id'])
    return {'accepted_ids':accepted,'reason':'frozen_raw_AND','gate_trace':trace}
