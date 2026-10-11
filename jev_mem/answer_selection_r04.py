"""Pure selection of one of two unchanged, persisted Reader answers."""
from copy import deepcopy
import hashlib
import math
import unicodedata
from collections.abc import Mapping
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature

PROTOCOL_SHA256='90d05e6e62783e16bcb25bc000e638444ef132b12d17e9adef97f69669fe1b4f'
ARMS=('baseline','cheap','jev')
ROUTES=('baseline','jev_r03')
CHOICES=('first_better','second_better','no_clear_preference','neither_supported')
POLICY={'version':'answer-selection-r04-v1','signature_utf8_byte_cap':131072,
        'model':'jev-1.13.0','presentation_seed':'r04-answer-order-v1',
        'abstention_exact_set':['insufficient evidence','evidence is insufficient','the evidence is insufficient'],
        'probability_sum_tolerance':1e-6,'maximum_tie_tolerance':1e-12,
        'new_reader_calls':0,'new_answers':False,'support':'memory_relative_not_source_truth'}

def validate_candidates(candidates):
    if not isinstance(candidates,Mapping) or set(candidates)!=set(ROUTES):
        raise ValueError('Exactly the two frozen candidate routes required')
    for c in candidates.values():
        if not isinstance(c,Mapping) or not isinstance(c.get('answer'),str) or not c['answer'].strip():
            raise ValueError('Actual nonempty original answer required, never regenerate')

def presentation(query_id):
    if not isinstance(query_id,str) or not query_id:raise ValueError('Global query ID required')
    bit=hashlib.sha256((POLICY['presentation_seed']+'|'+query_id).encode('utf-8')).digest()[0]%2
    routes=ROUTES if bit==0 else ROUTES[::-1]
    return {'first':routes[0],'second':routes[1],'hash_bit':bit,'seed':POLICY['presentation_seed']}

def union_records(pool,baseline,selected):
    from jev_mem.evidence_selection_r03 import validate_records
    validate_records(pool)
    if baseline!=pool[:20]:raise ValueError('Frozen S0 must be exact pool first20')
    byid={f['id']:f for f in pool}
    for records in (baseline,selected):
        if len(records)!=20 or len({f['id'] for f in records})!=20 or any(byid.get(f['id'])!=f for f in records):
            raise ValueError('Both routes require20 exact canonical pool records')
    ids={f['id'] for f in baseline+selected}
    return [deepcopy(f) for f in pool if f['id'] in ids]

def build_request(question,reference_date,memories,candidates,query_id,protocol):
    validate_candidates(candidates)
    if not isinstance(question,str) or not question.strip():raise ValueError('Nonempty question required')
    from jev_mem.evidence_selection_r03 import validate_records
    validate_records(memories)
    if len(memories)>40:raise ValueError('Union cannot exceed40 facts')
    mapping=presentation(query_id)
    state={'question':question,'reference_date':reference_date,'memories':deepcopy(memories),
           'candidate_first':candidates[mapping['first']]['answer'],
           'candidate_second':candidates[mapping['second']]['answer']}
    question_spec=deepcopy(protocol['jev']['request']['questions'])
    if set(question_spec)!= {'answer_choice'} or question_spec['answer_choice']['type']!='choice' or set(question_spec['answer_choice']['criteria'])!=set(CHOICES):
        raise ValueError('Exact four-way Choice contract required')
    return {'provider':'jev','model':POLICY['model'],'operation':'systemone',
            'body':{'model':POLICY['model'],'state':state,'questions':question_spec}}

def normalize_abstention(answer):
    if not isinstance(answer,str) or not answer.strip():raise ValueError('Nonempty actual answer required')
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in answer.casefold()).split())

def cheap_choice(record):
    candidates=record['candidates'];validate_candidates(candidates)
    abstain=set(POLICY['abstention_exact_set'])
    b=normalize_abstention(candidates['baseline']['answer']) in abstain
    j=normalize_abstention(candidates['jev_r03']['answer']) in abstain
    return 'jev_r03' if b and not j else 'baseline'

def _number(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1

def parse_choice(response,signature):
    """Reject malformed outputs; caller records explicit unknown, never zeros."""
    if signature.get('model')!=POLICY['model'] or signature.get('body',{}).get('model')!=POLICY['model']:
        raise ValueError('Unexpected approved request model')
    questions=signature.get('body',{}).get('questions',{})
    if (set(questions)!={'answer_choice'} or questions['answer_choice'].get('type')!='choice'
        or set(questions['answer_choice'].get('criteria',{}))!=set(CHOICES)):
        raise ValueError('Unexpected Choice question/options signature')
    if not isinstance(response,Mapping) or response.get('model')!=POLICY['model']:
        raise ValueError('Unexpected response model')
    answers=response.get('answers')
    if not isinstance(answers,Mapping) or set(answers)!={'answer_choice'}:raise ValueError('Exact one answer key required')
    answer=answers['answer_choice']
    if not isinstance(answer,Mapping) or answer.get('type')!='choice':raise ValueError('Choice answer type required')
    probs=answer.get('probabilities')
    if not isinstance(probs,Mapping) or set(probs)!=set(CHOICES) or any(not _number(v) for v in probs.values()):
        raise ValueError('All four finite nonboolean probabilities required')
    if abs(sum(probs.values())-1)>POLICY['probability_sum_tolerance'] or not _number(answer.get('confidence')):
        raise ValueError('Probability sum or confidence invalid')
    choice=answer.get('choice');maximum=max(probs.values());eps=POLICY['maximum_tie_tolerance']
    if choice not in CHOICES or abs(probs[choice]-maximum)>eps:raise ValueError('Returned choice must maximize probabilities')
    return {'choice':choice,'probabilities':dict(probs),'confidence':answer['confidence'],
            'tied_max':sum(abs(v-maximum)<=eps for v in probs.values())>1}

parse_scores=parse_choice

def choose_jev(record,parsed=None,*,status='complete',score_status=None):
    if score_status is not None:status=score_status
    validate_candidates(record['candidates'])
    same=record['candidates']['baseline']['answer']==record['candidates']['jev_r03']['answer']
    if bool(record['identical'])!=same:raise ValueError('Identical marker differs from exact strings')
    if same:
        if parsed is not None:raise ValueError('Identical answers must not have new Choice scores')
        return {'route':'baseline','reason':'identical_answer_no_dispatch'}
    if status!='complete':
        if status not in ('failed','unknown','not_dispatched') or parsed is not None:
            raise ValueError('Unavailable output requires explicit status and no synthetic probabilities')
        return {'route':'baseline','reason':status+'_fallback_baseline'}
    # Revalidate the stored parser object, so a forged tied_max cannot choose a route.
    response={'model':POLICY['model'],'answers':{'answer_choice':{'type':'choice',**(parsed or {})}}}
    checked=parse_choice(response,record.get('signature',record.get('request_signature',{})))
    if checked!=parsed:raise ValueError('Parsed Choice object cannot be reproduced')
    if checked['tied_max']:return {'route':'baseline','reason':'probability_maximum_tie_fallback'}
    choice=checked['choice']
    if choice in ('no_clear_preference','neither_supported'):return {'route':'baseline','reason':choice+'_fallback'}
    mapping=presentation(record['query_id'])
    if record['presentation']!=mapping:raise ValueError('Hash presentation mapping changed')
    return {'route':mapping['first' if choice=='first_better' else 'second'],'reason':choice}
