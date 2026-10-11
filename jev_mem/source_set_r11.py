"""Pure R11 full-memory conditional source selection; no API or gold access."""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import combinations
import math
from jev_mem.source_append_r05 import (canonical, validate_original, validate_candidates,
    digest, request_hash, serialized_signature, parse_scores)

ARMS = ('g', 'p')
MODEL = 'jev-1.13.0'
PAIR_KEYS = tuple(f'p{i:02d}' for i in range(6))
PAIRS = tuple(combinations(range(4), 2))
POLICY = {'version': 'source-set-r11-v1', 'selected_count': 20,
    'candidate_count': 4, 'attachment_count': 2, 'attachment_utf8_byte_cap': 4096,
    'signature_utf8_byte_cap': 131072, 'model': MODEL,
    'presentation_seed': 'r11-pair-order-v1', 'probability_sum_tolerance': 1e-6,
    'maximum_tie_tolerance': 1e-12, 'threshold': None,
    'failure_action': 'entire_original_N0', 'pair_tie_action': 'entire_original_N0'}


def validate_frozen_input(records, candidates):
    validate_original(records)
    validate_candidates(candidates, records)
    if len(candidates) != 4:
        raise ValueError('R11 requires exactly the four frozen R05 candidates')


def pair_options(query_id):
    if not isinstance(query_id, str) or not query_id:
        raise ValueError('Frozen query identity required for presentation')
    ordered = sorted(PAIRS, key=lambda pair: (hashlib.sha256(
        f'{POLICY["presentation_seed"]}|{query_id}|{pair[0]}{pair[1]}'.encode('utf-8')).hexdigest(), pair))
    return {key: list(pair) for key, pair in zip(PAIR_KEYS, ordered)}


def _state(question, reference_date, records):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Exact nonempty question required')
    if reference_date is not None and not isinstance(reference_date, str):
        raise ValueError('Original date must remain a string or null')
    return {'query': {'text': question, 'reference_date': reference_date},
        'memories': [{k: deepcopy(f[k]) for k in ('text', 'observation_date')} for f in records]}


def _candidate(i, candidate):
    return {'candidate_index': i, 'anchor_slot': candidate['origin']['anchor_slot'],
        'source_attachment': candidate['attachment']}


def _signature(state, questions):
    return {'provider': 'jev', 'model': MODEL, 'operation': 'systemone',
        'body': {'model': MODEL, 'state': state, 'questions': questions}}


def build_g_request(question, reference_date, records, candidates, protocol):
    validate_frozen_input(records, candidates)
    predicate = protocol['g']['predicate']
    if not isinstance(predicate, str) or not predicate.strip():
        raise ValueError('Frozen G predicate required')
    questions = {f'f{i:02d}': {'type': 'noul', 'instructions': predicate +
        '\nDisplayed candidate source attachment (JSON):\n' + canonical(_candidate(i, c))}
        for i, c in enumerate(candidates)}
    return _signature(_state(question, reference_date, records), questions)


def build_p_request(query_id, question, reference_date, records, candidates, protocol):
    validate_frozen_input(records, candidates)
    spec = protocol['p']
    if spec['presentation_seed'] != POLICY['presentation_seed']:
        raise ValueError('Frozen presentation seed differs')
    if not isinstance(spec['instructions'], str) or not spec['instructions'].strip():
        raise ValueError('Frozen P predicate required')
    template = spec['criteria_template']
    if (not isinstance(template, str) or '{first}' not in template or '{second}' not in template):
        raise ValueError('Symmetric pair criterion with both indices required')
    options = pair_options(query_id)
    state = _state(question, reference_date, records)
    state.update(candidates=[_candidate(i, c) for i, c in enumerate(candidates)],
        bundle_options=options)
    criteria = {key: template.format(first=pair[0], second=pair[1]) for key, pair in options.items()}
    return _signature(state, {'pair_choice': {'type': 'choice',
        'instructions': spec['instructions'], 'criteria': criteria}})


def validate_signature(arm, signature):
    if (arm not in ARMS or not isinstance(signature, Mapping)
        or set(signature) != {'provider', 'model', 'operation', 'body'}
        or (signature['provider'], signature['model'], signature['operation']) != ('jev', MODEL, 'systemone')):
        raise ValueError('Only approved R11 systemone signature allowed')
    body = signature['body']
    if not isinstance(body, Mapping) or set(body) != {'model', 'state', 'questions'} or body['model'] != MODEL:
        raise ValueError('Approved body model/state/questions required')
    state = body['state']
    expected_state = {'query', 'memories'} | ({'candidates', 'bundle_options'} if arm == 'p' else set())
    if not isinstance(state, Mapping) or set(state) != expected_state:
        raise ValueError('Only whitelisted model state fields allowed')
    query = state['query']
    if (not isinstance(query, Mapping) or set(query) != {'text', 'reference_date'}
        or not isinstance(query['text'], str) or not query['text'].strip()
        or (query['reference_date'] is not None and not isinstance(query['reference_date'], str))):
        raise ValueError('Original query/date shape required')
    memories = state['memories']
    if (not isinstance(memories, list) or len(memories) != 20
        or any(not isinstance(f, Mapping) or set(f) != {'text', 'observation_date'}
            or not isinstance(f['text'], str) or not f['text'].strip()
            or (f['observation_date'] is not None and not isinstance(f['observation_date'], str)) for f in memories)):
        raise ValueError('All twenty exact memory views required')
    questions = body['questions']
    if not isinstance(questions, Mapping):
        raise ValueError('Prepared question mapping required')
    if arm == 'g':
        if set(questions) != {f'f{i:02d}' for i in range(4)}:
            raise ValueError('Exactly four contiguous G questions required')
        if any(not isinstance(q, Mapping) or set(q) != {'type', 'instructions'}
            or q['type'] != 'noul' or not isinstance(q['instructions'], str) for q in questions.values()):
            raise ValueError('Only full prepared G Noul questions allowed')
    else:
        if set(questions) != {'pair_choice'}:
            raise ValueError('Exact one P Choice required')
        q = questions['pair_choice']
        if (not isinstance(q, Mapping) or set(q) != {'type', 'instructions', 'criteria'}
            or q['type'] != 'choice' or not isinstance(q['instructions'], str)
            or not isinstance(q['criteria'], Mapping) or set(q['criteria']) != set(PAIR_KEYS)
            or any(not isinstance(v, str) or not v for v in q['criteria'].values())):
            raise ValueError('Exact six P criteria required')
        options = state['bundle_options']
        if (not isinstance(options, Mapping) or set(options) != set(PAIR_KEYS)
            or any(not isinstance(v, list) or len(v) != 2 or any(type(i) is not int for i in v)
                or tuple(v) not in PAIRS for v in options.values())
            or {tuple(v) for v in options.values()} != set(PAIRS)):
            raise ValueError('Complete unique six original pairs required')
        cs = state['candidates']
        if (not isinstance(cs, list) or len(cs) != 4 or any(not isinstance(c, Mapping)
            or set(c) != {'candidate_index', 'anchor_slot', 'source_attachment'}
            or type(c['candidate_index']) is not int or c['candidate_index'] != i
            or type(c['anchor_slot']) is not int or not 0 <= c['anchor_slot'] < 20
            or not isinstance(c['source_attachment'], str)
            or len(c['source_attachment'].encode('utf-8')) > 4096 for i, c in enumerate(cs))
            or len({c['anchor_slot'] for c in cs}) != 4):
            raise ValueError('Exact four distinct bounded candidate views required')
    return signature


def parse_g(response, signature):
    validate_signature('g', signature)
    return parse_scores(response, signature)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def parse_p(response, signature):
    validate_signature('p', signature)
    if not isinstance(response, Mapping) or response.get('model') != MODEL:
        raise ValueError('Exact response model required')
    answers = response.get('answers')
    if not isinstance(answers, Mapping) or set(answers) != {'pair_choice'}:
        raise ValueError('Exact one pair_choice answer required')
    answer = answers['pair_choice']
    if not isinstance(answer, Mapping) or answer.get('type') != 'choice':
        raise ValueError('Choice response required')
    probs = answer.get('probabilities')
    if (not isinstance(probs, Mapping) or set(probs) != set(PAIR_KEYS)
        or any(not _number(v) for v in probs.values())
        or abs(math.fsum(probs.values()) - 1) > POLICY['probability_sum_tolerance']
        or not _number(answer.get('confidence'))):
        raise ValueError('All six normalized finite probabilities and confidence required')
    choice = answer.get('choice')
    maximum, epsilon = max(probs.values()), POLICY['maximum_tie_tolerance']
    if choice not in PAIR_KEYS or abs(probs[choice] - maximum) > epsilon:
        raise ValueError('Returned choice must maximize probabilities')
    return {'choice': choice, 'probabilities': dict(probs), 'confidence': answer['confidence'],
        'tied_max': sum(abs(v - maximum) <= epsilon for v in probs.values()) > 1,
        'candidate_indices': deepcopy(signature['body']['state']['bundle_options'][choice])}


def _decision(arm, indices, candidates, status, reason):
    return {'candidate_indices': list(indices),
        'attached_slot_indices': [candidates[i]['origin']['anchor_slot'] for i in indices],
        'packet_ids': [candidates[i]['packet_id'] for i in indices],
        'added_utf8_bytes': sum(candidates[i]['attachment_utf8_bytes'] for i in indices),
        'score_status': status, 'reason': reason, 'changed_text': bool(indices),
        'calibration': 'uncalibrated_full_set_conditional_usefulness' if arm == 'g'
            else 'uncalibrated_comparative_pair_choice', 'arm': arm}


def _unavailable(arm, candidates, parsed, status):
    if status not in ('failed', 'unknown', 'not_dispatched') or parsed is not None:
        raise ValueError('Explicit unavailable status and no synthetic probabilities required')
    return _decision(arm, [], candidates, status, 'unavailable_scores_keep_baseline')


def choose_g(records, candidates, scores, *, score_status='complete'):
    validate_frozen_input(records, candidates)
    if score_status != 'complete':
        return _unavailable('g', candidates, scores, score_status)
    keys = {f'f{i:02d}' for i in range(4)}
    if not isinstance(scores, Mapping) or set(scores) != keys or any(not _number(v) for v in scores.values()):
        raise ValueError('All four valid probabilities required')
    indices = sorted(range(4), key=lambda i: (-scores[f'f{i:02d}'], i))[:2]
    return _decision('g', indices, candidates, 'complete', 'full_set_conditional_top2')


def choose_p(records, candidates, parsed, *, signature=None, score_status='complete'):
    validate_frozen_input(records, candidates)
    if score_status != 'complete':
        return _unavailable('p', candidates, parsed, score_status)
    if not isinstance(parsed, Mapping) or signature is None:
        raise ValueError('Complete parsed Choice bound to its signature required')
    validate_signature('p', signature)
    state = signature['body']['state']
    if (state['memories'] != [{k: deepcopy(f[k]) for k in ('text', 'observation_date')} for f in records]
        or state['candidates'] != [_candidate(i, c) for i, c in enumerate(candidates)]):
        raise ValueError('P signature does not bind the exact frozen memories and candidates')
    raw = {'model': MODEL, 'answers': {'pair_choice': {'type': 'choice',
        **{k: parsed.get(k) for k in ('choice', 'probabilities', 'confidence')}}}}
    checked = parse_p(raw, signature)
    if checked != parsed:
        raise ValueError('Parsed Choice and pair mapping must replay exactly')
    if checked['tied_max']:
        return _decision('p', [], candidates, 'complete', 'probability_maximum_tie_fallback')
    return _decision('p', checked['candidate_indices'], candidates, 'complete', 'full_set_pair_choice')


def selected_facts(records, candidates, decision):
    validate_frozen_input(records, candidates)
    if not isinstance(decision, Mapping) or decision.get('arm') not in ARMS:
        raise ValueError('Exact R11 decision required')
    indices = decision.get('candidate_indices')
    if (not isinstance(indices, list) or len(indices) not in (0, 2)
        or any(type(i) is not int or not 0 <= i < 4 for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError('Either baseline fallback or exactly two unique candidates required')
    if decision != _decision(decision['arm'], indices, candidates, decision['score_status'], decision['reason']):
        raise ValueError('Attachment decision accounting differs')
    result = deepcopy(records)
    for i in indices:
        c = candidates[i]
        result[c['origin']['anchor_slot']]['text'] += c['attachment']
    return result
