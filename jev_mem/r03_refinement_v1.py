"""Pure, answer-blind precision/complement/set refinement of frozen R03 inputs."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math

from jev_mem.evidence_selection_r03 import validate_records, selected_facts as pool_facts
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores

MODEL = 'jev-1.13.0'
VERSION = 'r03-refinement-v1'
POLICY = {
    'version': VERSION, 'pool_cap': 80, 'selected_count': 20,
    'questions_per_batch': 32, 'signature_utf8_byte_cap': 131072,
    'swap_counts': [1, 2, 4], 'set_proposal_cap': 4,
    'set_acceptance_strict_gt': 0.5, 'set_maximum_tie_tolerance': 1e-12,
    'pointwise_threshold': None, 'complement_threshold': None,
    'ranking_tie': 'original_pool_index',
    'donor_order': 'reverse_precision_ranking_within_preliminary20',
    'reader_order': 'original_pool_order',
    'failure_action': 'whole_query_original_OJ20',
    'unknown_action': 'raise_needs_review',
    'oversize_action': 'no_dispatch_no_truncation_no_adaptive_batch',
    'calibration': 'uncalibrated_operational_scores_not_truth_probabilities',
}
PRECISION = (
    'Does this memory explicitly supply at least one fact or relation required by the original '
    'question, with the requested subject, relation direction, event identity and material '
    'conditions? Partial requested list members or necessary reasoning links may count; the '
    'memory need not answer the whole question. Topic or actor overlap alone is insufficient. '
    'Preserve negation, planned versus completed status, and the question temporal scope. '
    'Do not invent an unstated bridge or infer an event date solely from observation_date. '
    'A memory need not repeat every question word when its relation is explicit. Assess '
    'memory-relative usefulness, not source truth.'
)
COMPLEMENT = (
    'Compared with all selected_memories in state, does this candidate explicitly add a '
    'distinct fact, requested list member, necessary reasoning link, or clarification of a '
    'subject, relation, event, condition or time that is needed by the original question '
    'and absent or less explicit in that set? Mere paraphrase, shared topic or a different '
    'event is insufficient. Partial requested list members may count; do not invent unknown '
    'members or require the candidate to complete the entire answer. Preserve negation, '
    'planned versus completed status and temporal scope. Observation date is not by itself '
    'event date. Assess only the displayed memories; this does not certify source truth '
    'or completeness of the resulting answer.'
)
SET_PREDICATE = (
    'Compare proposed_memories with original_memories for the original question. Does the '
    'proposed set provide a strict net improvement in explicit question-required evidence, '
    'through additional necessary coverage or clearer subject, relation, event, condition '
    'or time, without losing a necessary list member or reasoning link supplied by the '
    'original set? Evaluate each set using only the twenty keys assigned to that set in '
    'original_memories or PROPOSED_MEMORY_KEYS_JSON. Other fact_table entries and keys '
    'belonging only to another set must not fill a gap in the set being evaluated. '
    'Read each assigned memory from fact_table. More topical text, redundancy '
    'or length alone is not improvement. Keep speaker/subject and relation direction, '
    'negation, planned versus completed status, and temporal scope; do not invent bridges '
    'or turn observation dates into event dates. If the sets are equally useful, or the '
    'proposed set trades away required evidence without a clear net gain, answer NO. '
    'Assess memory-relative evidence only, not source truth or guaranteed Reader accuracy.'
)
PROTOCOL = {'schema': VERSION + '-protocol', 'policy': POLICY,
    'precision_predicate': PRECISION, 'complement_predicate': COMPLEMENT,
    'set_predicate': SET_PREDICATE}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _view(f):
    return {k: deepcopy(f[k]) for k in ('text', 'observation_date')}


def _base(question, reference_date, pool, original):
    validate_records(pool)
    validate_records(original)
    if len(original) != 20 or not isinstance(question, str) or not question.strip():
        raise ValueError('Exact question and twenty original OJ facts required')
    if reference_date is not None and not isinstance(reference_date, str):
        raise ValueError('Original reference date string/null required')
    pos = {f['id']: i for i, f in enumerate(pool)}
    if any(f['id'] not in pos or pool[pos[f['id']]] != f for f in original):
        raise ValueError('Original OJ20 must be canonical members of the same frozen pool')
    indices = [pos[f['id']] for f in original]
    if indices != sorted(indices):
        raise ValueError('Original OJ Reader order must be original pool order')
    return {'query': {'text': question, 'reference_date': reference_date},
        'pool': deepcopy(pool), 'original_indices': indices}


def _restore_base(base):
    return _base(base['query']['text'], base['query']['reference_date'], base['pool'],
        pool_facts(base['pool'], base['original_indices']))


def _signature(state, questions):
    return {'provider': 'jev', 'model': MODEL, 'operation': 'systemone',
        'body': {'model': MODEL, 'state': state, 'questions': questions}}


def _requests(base, indices, stage, extra=None):
    out = []
    predicate, prefix = (PRECISION, 'f') if stage == 'precision' else (COMPLEMENT, 'c')
    for start in range(0, len(indices), POLICY['questions_per_batch']):
        batch = indices[start:start + POLICY['questions_per_batch']]
        questions = {f'{prefix}{i:02d}': {'type': 'noul', 'instructions': predicate +
            '\nMEMORY_JSON:\n' + _canonical(_view(base['pool'][i]))} for i in batch}
        state = {'query': deepcopy(base['query']), **deepcopy(extra or {})}
        sig = _signature(state, questions)
        validate_request(sig, allow_oversize=True)
        out.append({'signature': sig, 'key_to_pool_index': {f'{prefix}{i:02d}': i for i in batch}})
    return out


def _scores(scores, keys, status):
    if status == 'unknown':
        raise ValueError('Unknown execution requires review; it is not a fallback outcome')
    if status not in ('complete', 'failed', 'not_dispatched'):
        raise ValueError('Unknown score status')
    if status != 'complete':
        if scores is not None:
            raise ValueError('Failed phase cannot retain partial or invented scores')
        return None
    if not isinstance(scores, Mapping) or set(scores) != set(keys):
        raise ValueError('Complete exact score map required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Finite nonboolean probabilities required')
    return dict(scores)


def build_precision_plan(question, reference_date, pool_records, original_OJ20):
    base = _base(question, reference_date, pool_records, original_OJ20)
    return {'schema': VERSION + '-precision', 'base': base,
        'requests': _requests(base, list(range(len(base['pool']))), 'precision')}


def resolve_precision(plan, scores, *, score_status='complete'):
    b = _restore_base(plan['base'])
    if plan != build_precision_plan(b['query']['text'], b['query']['reference_date'],
            b['pool'], pool_facts(b['pool'], b['original_indices'])):
        raise ValueError('Precision plan does not replay')
    vals = _scores(scores, [f'f{i:02d}' for i in range(len(b['pool']))], score_status)
    ranking = sorted(range(len(b['pool'])), key=lambda i: (-vals[f'f{i:02d}'], i)) if vals is not None else []
    return {'status': score_status, 'scores': vals, 'ranked_indices': ranking,
        'preliminary_indices': sorted(ranking[:20]) if vals is not None else None}


def build_complement_plan(precision_plan, precision_result):
    if precision_result != resolve_precision(precision_plan, precision_result['scores'],
            score_status=precision_result['status']):
        raise ValueError('Precision result does not replay')
    b = deepcopy(precision_plan['base']); p = precision_result['preliminary_indices']
    indices = [i for i in range(len(b['pool'])) if i not in set(p)] if p is not None else []
    return {'schema': VERSION + '-complement', 'base': b,
        'precision': deepcopy(precision_result), 'candidate_indices': indices,
        'requests': _requests(b, indices, 'complement',
            {'selected_memories': [_view(b['pool'][i]) for i in p]}) if p is not None else []}


def _precision_plan(base):
    return build_precision_plan(base['query']['text'], base['query']['reference_date'],
        base['pool'], pool_facts(base['pool'], base['original_indices']))


def resolve_complement(plan, scores, *, score_status='complete'):
    if plan != build_complement_plan(_precision_plan(plan['base']), plan['precision']):
        raise ValueError('Complement plan does not replay')
    if plan['precision']['status'] != 'complete':
        if score_status not in ('complete', 'failed', 'not_dispatched') or scores not in (None, {}):
            raise ValueError('Unexecuted dependent stage cannot supply scores or unknown state')
        return {'status': plan['precision']['status'], 'scores': None, 'ranked_indices': []}
    vals = _scores(scores, [f'c{i:02d}' for i in plan['candidate_indices']], score_status)
    ranking = sorted(plan['candidate_indices'], key=lambda i: (-vals[f'c{i:02d}'], i)) if vals is not None else []
    return {'status': score_status, 'scores': vals, 'ranked_indices': ranking}


def build_set_plan(complement_plan, complement_result):
    if complement_result != resolve_complement(complement_plan, complement_result['scores'],
            score_status=complement_result['status']):
        raise ValueError('Complement result does not replay')
    b = deepcopy(complement_plan['base']); precision = complement_plan['precision']
    proposals = []; seen = {tuple(b['original_indices'])}; fallback = None
    if precision['status'] != 'complete': fallback = 'precision_unavailable_keep_OJ'
    elif complement_result['status'] != 'complete': fallback = 'complement_unavailable_keep_OJ'
    else:
        p = precision['preliminary_indices']; donors = list(reversed(precision['ranked_indices'][:20]))
        candidates = [(0, p)]
        for k in POLICY['swap_counts']:
            if len(complement_result['ranked_indices']) >= k:
                candidates.append((k, sorted((set(p) - set(donors[:k])) | set(complement_result['ranked_indices'][:k]))))
        for k, indices in candidates:
            if tuple(indices) in seen: continue
            seen.add(tuple(indices)); pool_facts(b['pool'], indices)
            proposals.append({'key': f'p{len(proposals):02d}', 'swap_count_from_P': k, 'selected_indices': indices})
    requests = []
    if proposals:
        state = {'query': deepcopy(b['query']),
            'fact_table': {f'f{i:02d}': _view(f) for i, f in enumerate(b['pool'])},
            'original_memories': [f'f{i:02d}' for i in b['original_indices']]}
        questions = {p['key']: {'type': 'noul', 'instructions': SET_PREDICATE +
            '\nPROPOSED_MEMORY_KEYS_JSON:\n' + _canonical([f'f{i:02d}' for i in p['selected_indices']])} for p in proposals}
        sig = _signature(state, questions); validate_request(sig, allow_oversize=True)
        requests = [{'signature': sig, 'key_to_proposal_index': {p['key']: i for i, p in enumerate(proposals)}}]
    return {'schema': VERSION + '-set', 'base': b, 'precision': deepcopy(precision),
        'complement': deepcopy(complement_result), 'proposals': proposals,
        'fallback_reason': fallback, 'requests': requests}


def choose_final(plan, scores, *, score_status='complete'):
    pp = _precision_plan(plan['base']); cp = build_complement_plan(pp, plan['precision'])
    if plan != build_set_plan(cp, plan['complement']):
        raise ValueError('Set plan does not replay')
    winner = None; reason = plan['fallback_reason']; b = plan['base']
    if score_status not in ('complete', 'failed', 'not_dispatched'):
        raise ValueError('Unknown set execution requires review')
    if reason or not plan['proposals']:
        if scores not in (None, {}): raise ValueError('No set request may supply scores')
        reason = reason or 'all_proposals_equal_OJ'
    else:
        vals = _scores(scores, [p['key'] for p in plan['proposals']], score_status)
        if vals is None: reason = 'set_scores_unavailable_keep_OJ'
        else:
            best = max(vals.values())
            tied = [k for k, v in vals.items() if abs(v - best) <= POLICY['set_maximum_tie_tolerance']]
            if best <= POLICY['set_acceptance_strict_gt']: reason = 'no_strict_majority_improvement_keep_OJ'
            elif len(tied) != 1: reason = 'maximum_tie_keep_OJ'
            else:
                winner = next(p for p in plan['proposals'] if p['key'] == tied[0])
                reason = 'unique_majority_set_improvement'
    indices = winner['selected_indices'] if winner else b['original_indices']
    effective_status = (plan['precision']['status'] if plan['precision']['status'] != 'complete'
        else plan['complement']['status'] if plan['complement']['status'] != 'complete' else score_status)
    return {'selected_indices': deepcopy(indices), 'selected_ids': [b['pool'][i]['id'] for i in indices],
        'accepted_proposal_key': winner['key'] if winner else None, 'reason': reason,
        'score_status': effective_status, 'calibration': POLICY['calibration']}


def selected_facts(pool_records, decision):
    facts = pool_facts(pool_records, decision['selected_indices'])
    if [f['id'] for f in facts] != decision['selected_ids']: raise ValueError('Final selection ID/index mismatch')
    return facts


def _valid_view(value):
    return (isinstance(value, Mapping) and set(value) == {'text', 'observation_date'}
        and isinstance(value['text'], str) and bool(value['text'].strip())
        and (value['observation_date'] is None or isinstance(value['observation_date'], str)))


def validate_request(signature, *, allow_oversize=False):
    if not isinstance(signature, Mapping) or set(signature) != {'provider','model','operation','body'}:
        raise ValueError('Exact JEV signature fields required')
    if (signature['provider'],signature['model'],signature['operation']) != ('jev', MODEL, 'systemone'):
        raise ValueError('Frozen JEV model/operation required')
    body = signature['body']
    if not isinstance(body, Mapping) or set(body) != {'model','state','questions'} or body['model'] != MODEL:
        raise ValueError('Frozen body fields required')
    state, questions = body['state'], body['questions']
    if not isinstance(state, Mapping) or not isinstance(questions, Mapping) or not questions:
        raise ValueError('Nonempty bounded request required')
    query = state.get('query')
    if (not isinstance(query, Mapping) or set(query) != {'text','reference_date'}
        or not isinstance(query['text'], str) or not query['text'].strip()
        or (query['reference_date'] is not None and not isinstance(query['reference_date'], str))):
        raise ValueError('Only exact question/date admitted')
    if any(not isinstance(key, str) or not key for key in questions):
        raise ValueError('String bookkeeping keys required')
    prefix = next(iter(questions))[0]
    if prefix == 'f': stage, predicate, marker, extra = 'precision', PRECISION, '\nMEMORY_JSON:\n', set()
    elif prefix == 'c': stage, predicate, marker, extra = 'complement', COMPLEMENT, '\nMEMORY_JSON:\n', {'selected_memories'}
    elif prefix == 'p': stage, predicate, marker, extra = 'set', SET_PREDICATE, '\nPROPOSED_MEMORY_KEYS_JSON:\n', {'fact_table','original_memories'}
    else: raise ValueError('Unknown stage prefix')
    if set(state) != {'query'} | extra or len(questions) > (4 if stage == 'set' else 32):
        raise ValueError('Stage whitelist/cap differs')
    if stage == 'complement' and (not isinstance(state['selected_memories'], list)
            or len(state['selected_memories']) != 20 or not all(map(_valid_view,state['selected_memories']))):
        raise ValueError('Exactly twenty full memory views required')
    if stage == 'set':
        table = state['fact_table']; original = state['original_memories']
        if (not isinstance(table, Mapping) or not 20 <= len(table) <= 80
            or set(table) != {f'f{i:02d}' for i in range(len(table))} or not all(map(_valid_view,table.values()))
            or not isinstance(original, list) or len(original) != 20 or len(set(original)) != 20 or not set(original) <= set(table)):
            raise ValueError('Bounded canonical table and original20 required')
    for key, item in questions.items():
        if not isinstance(key,str) or len(key)!=3 or key[0]!=prefix or not key[1:].isdigit() or int(key[1:]) >= (4 if stage=='set' else 80):
            raise ValueError('Bookkeeping key invalid')
        if (not isinstance(item, Mapping) or set(item) != {'type','instructions'} or item['type'] != 'noul'
            or not isinstance(item['instructions'],str) or not item['instructions'].startswith(predicate+marker)):
            raise ValueError('Exact frozen predicate/type required')
        payload = json.loads(item['instructions'][len(predicate+marker):])
        if stage != 'set' and not _valid_view(payload): raise ValueError('Only memory text/date allowed')
        if stage == 'set' and (not isinstance(payload,list) or len(payload)!=20 or len(set(payload))!=20 or not set(payload)<=set(table)):
            raise ValueError('Exactly twenty proposal memory keys required')
        if item['instructions'] != predicate+marker+_canonical(payload): raise ValueError('Canonical full payload required')
    if not allow_oversize and len(serialized_signature(signature)) > POLICY['signature_utf8_byte_cap']:
        raise ValueError('Signature oversize; no model dispatch allowed')
    return stage


def parse_response(response, signature):
    validate_request(signature)
    return parse_scores(response, signature)
