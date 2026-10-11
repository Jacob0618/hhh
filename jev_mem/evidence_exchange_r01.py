"""Pure, bounded R01 evidence exchange. No providers, stores or API dispatch.

All retrieval inputs are frozen facts and the question, never QA gold/labels.
Noul probabilities are uncalibrated heuristics, not semantic support certificates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Mapping

from tools.pilot400_recovery_rank_v1 import stable_bm25

VERSION = 'evidence-exchange-r01-v1'
ARMS = ('baseline', 'cheap', 'jev')
POLICY = {
    'version': VERSION, 'selected_count': 20, 'pool_cap': 80,
    'native_other_cap': 40, 'independent_cap': 20, 'donor_tail': 4,
    'proposal_cap': 4, 'g_min': .80, 'l_max': .10, 'x_max': .10,
    'calibration': 'raw_probabilities_uncalibrated_heuristics',
    'native_order': 'capture.native_capture.candidate_ids',
    'native_eligible_ids': 'audit_only_not_pool_filter',
    'bm25': 'tools.pilot400_recovery_rank_v1.stable_bm25',
    'bm25_k1': 1.2, 'bm25_b': .75,
    'tokenizer': 'unicode_word_casefold_no_stopwords_sorted_unique_query_terms',
    'independent_retrieval': 'one_query_full_same_bank_scan_excluding_entire_original_native80',
    'challengers': 'positive_bm25_desc_bank_index_id_first4',
    'donors': 'baseline_tail4_bm25_asc_later_baseline_index_first',
    'proposals': 'zip_challengers_and_donors_no_sequential_acceptance',
    'cheap_gate': 'maximum_strictly_positive_bm25_delta_tie_proposal_order',
    'jev_gate': 'first_complete_proposal_g_ge_0.80_l_le_0.10_x_le_0.10',
    'request_cap': 1, 'request_cap_semantics': 'one_logical_signature_one_shared_state_batch',
    'noul_cap': 12, 'signature_utf8_byte_cap': 131072,
    'oversize_action': 'explicit_not_dispatched_keep_baseline_never_truncate',
    'response_model': 'jev-1.13.0', 'response_answer_type': 'noul_required',
    'source_context': 'facts_only_memory_relative',
    'unknown_action': 'unchanged_baseline_without_zero_imputation',
    'pilot_hash_prefix': 'r01-pilot-v1', 'pilot_per_conversation': 8,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def reader_fact(fact):
    return {key: fact.get(key) for key in ('id', 'text', 'observation_date')}


def fact_index(facts):
    result = {}
    for fact in facts:
        if (not isinstance(fact, dict) or not isinstance(fact.get('id'), str)
            or not fact['id'] or not isinstance(fact.get('text'), str) or not fact['text'].strip()
            or fact['id'] in result):
            raise ValueError('Unique canonical bank ID and nonempty text required')
        result[fact['id']] = reader_fact(fact)
    return result


def validate_ids(ids, by_id, *, exact=None, cap=None):
    if (not isinstance(ids, list) or any(not isinstance(x, str) for x in ids)
        or len(ids) != len(set(ids)) or not set(ids) <= set(by_id)
        or (exact is not None and len(ids) != exact) or (cap is not None and len(ids) > cap)):
        raise ValueError('IDs must be unique, in the same bank, and within the declared count')


def build_pool(question, facts, baseline_ids, native80_ids):
    """One full bank BM25 scan, one bounded pool; original native80 stays intact."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Nonempty query required')
    by_id = fact_index(facts)
    validate_ids(baseline_ids, by_id, exact=20)
    validate_ids(native80_ids, by_id, cap=80)
    if not set(baseline_ids) <= set(native80_ids):
        raise ValueError('Baseline20 must come from the original captured native80')
    bank_order = {fid: i for i, fid in enumerate(by_id)}
    scores = dict(zip(by_id, stable_bm25(question, [f['text'] for f in by_id.values()])))
    if any(not math.isfinite(x) or x < 0 for x in scores.values()):
        raise ValueError('Invalid BM25 result')
    baseline_set, native_set = set(baseline_ids), set(native80_ids)
    native_others = [fid for fid in native80_ids if fid not in baseline_set][:40]
    lexical = sorted((fid for fid in by_id if fid not in native_set and scores[fid] > 0),
        key=lambda fid: (-scores[fid], bank_order[fid], fid))[:20]
    pool = [*baseline_ids, *native_others, *lexical]
    seen = set(pool)
    filler = [fid for fid in native80_ids if fid not in seen][:80 - len(pool)]
    pool.extend(filler)
    validate_ids(pool, by_id, cap=80)
    return {'schema': VERSION + '-pool', 'pool_ids': pool, 'baseline_ids': list(baseline_ids),
        'original_native80_ids': list(native80_ids), 'native_other_ids': native_others,
        'independent_ids': lexical, 'native_fill_ids': filler,
        'independent_positive_outside_count': sum(fid not in native_set and scores[fid] > 0 for fid in by_id),
        'bank_scan_count': len(by_id), 'bm25_query_count': 1, 'bm25_scores': scores,
        'bank_order': list(by_id), 'policy_sha256': digest(POLICY)}


def propose_exchanges(pool):
    base, scores = pool['baseline_ids'], pool['bm25_scores']
    rank = {fid: i for i, fid in enumerate(pool['bank_order'])}
    donors = sorted(base[-4:], key=lambda fid: (scores[fid], -base.index(fid), fid))
    challengers = sorted((fid for fid in pool['pool_ids'] if fid not in set(base) and scores[fid] > 0),
        key=lambda fid: (-scores[fid], rank[fid], fid))[:4]
    proposals = []
    for i, (added, removed) in enumerate(zip(challengers, donors)):
        selected = [added if fid == removed else fid for fid in base]
        proposals.append({'proposal_id': f'p{i}', 'added_id': added, 'removed_id': removed,
            'slot': base.index(removed), 'selected_ids': selected,
            'bm25_added': scores[added], 'bm25_removed': scores[removed],
            'bm25_delta': scores[added] - scores[removed]})
    return proposals


def validate_proposals(pool, proposals):
    if proposals != propose_exchanges(pool):
        raise ValueError('Full proposals differ from the frozen deterministic proposer')


def choose_cheap(pool, proposals, *, eligible_proposal_ids=None):
    validate_proposals(pool, proposals)
    allowed = set(eligible_proposal_ids) if eligible_proposal_ids is not None else {p['proposal_id'] for p in proposals}
    winners = [p for p in proposals if p['proposal_id'] in allowed and p['bm25_delta'] > 0]
    winner = max(winners, key=lambda p: (p['bm25_delta'], -int(p['proposal_id'][1:]))) if winners else None
    return _decision(pool, winner, 'complete', 'positive_lexical_delta' if winner else 'no_positive_lexical_delta')


def _decision(pool, winner, score_status, reason):
    return {'selected_ids': list(winner['selected_ids'] if winner else pool['baseline_ids']),
        'accepted_proposal_id': winner['proposal_id'] if winner else None,
        'score_status': score_status, 'reason': reason,
        'calibration': POLICY['calibration'], 'swaps': 1 if winner else 0}


def build_request(question, reference_date, facts, pool, proposals):
    """Returns the complete signature, matching ModelAPI's original request format."""
    validate_proposals(pool, proposals)
    if not proposals:
        return None
    by_id = fact_index(facts)
    base = pool['baseline_ids']
    fact_ids = list(base) + [p['added_id'] for p in proposals]
    index = {fid: i for i, fid in enumerate(fact_ids)}
    state = {'query': {'text': question, 'reference_date': reference_date},
        'facts': [by_id[fid] for fid in fact_ids],
        'baseline_indices': [index[fid] for fid in base],
        'proposals': [{'proposal_id': p['proposal_id'], 'removed_index': index[p['removed_id']],
            'added_index': index[p['added_id']], 'selected_indices': [index[fid] for fid in p['selected_ids']]}
            for p in proposals],
        'scope_note': 'Only stored facts are supplied. Judge memory-relative support, not dialogue truth or eventual Reader accuracy.'}
    questions = {}
    for i, proposal in enumerate(proposals):
        prefix = (f'Evaluate state.proposals[{i}] against state.baseline_indices for state.query.text. '
            'The replacement set is the COMPLETE 20 facts indexed by selected_indices. '
            'Do not generate an answer or invent facts. Same topic/person alone is not necessary support. '
            'Judge only the supplied memory world, with explicit roles, reasons, list members and time constraints. ')
        questions[f'{proposal["proposal_id"]}_G'] = {'type': 'noul', 'instructions': prefix +
            'Does the replacement add specific information needed by the query that the baseline lacks, '
            'including an explicitly readable bridge? This does not mean all answer parts are covered.'}
        questions[f'{proposal["proposal_id"]}_L'] = {'type': 'noul', 'instructions': prefix +
            'Does the replacement lose specific query-necessary support from the removed fact that '
            'neither remaining facts nor the added fact replace? Consider each reason/list member/time/role.'}
        questions[f'{proposal["proposal_id"]}_X'] = {'type': 'noul', 'instructions': prefix +
            'Does the added fact introduce a relevant contradiction not explained by explicit different '
            'events, dates, or a stated correction/update? A later observation date alone does not justify overwriting.'}
    return {'provider': 'jev', 'model': 'jev-1.13.0', 'operation': 'systemone',
        'body': {'model': 'jev-1.13.0', 'state': state, 'questions': questions}}


def request_hash(signature):
    # This intentionally matches the frozen ModelAPI.request serializer, including spaces.
    return hashlib.sha256(serialized_signature(signature)).hexdigest()


def serialized_signature(signature):
    return json.dumps(signature, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode('utf-8')


def parse_scores(response, signature):
    keys = signature['body']['questions']
    if not isinstance(response, dict) or response.get('model') != 'jev-1.13.0':
        raise ValueError('Response must explicitly resolve to the pinned jev-1.13.0 model')
    if not isinstance(response.get('answers'), dict):
        raise ValueError('Response answers missing; unavailable is not zero')
    if set(response['answers']) != set(keys):
        raise ValueError('Exactly all requested Noul answers required')
    scores = {}
    for key in keys:
        row = response['answers'][key]
        if not isinstance(row, dict) or row.get('type') != 'noul':
            raise ValueError('Every answer must explicitly be type noul')
        value = row.get('noul') if isinstance(row, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Missing, unknown, nonfinite or invalid Noul; never impute zero')
        scores[key] = float(value)
    return scores


def choose_jev(pool, proposals, scores, *, score_status='complete', eligible_proposal_ids=None):
    validate_proposals(pool, proposals)
    if not proposals:
        return _decision(pool, None, 'not_needed', 'no_positive_challengers')
    if score_status != 'complete':
        if score_status not in ('failed', 'unknown', 'not_dispatched'):
            raise ValueError('Explicit terminal/unavailable score status required')
        if scores is not None:
            raise ValueError('Unavailable score record must not contain invented values')
        return _decision(pool, None, score_status, 'unavailable_scores_keep_baseline')
    expected = {f'{p["proposal_id"]}_{label}' for p in proposals for label in ('G', 'L', 'X')}
    if not isinstance(scores, Mapping) or set(scores) != expected:
        raise ValueError('All 3 scores per proposal required; no partial/zero-filled decisions')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Finite non-boolean probabilities required')
    allowed = set(eligible_proposal_ids) if eligible_proposal_ids is not None else {p['proposal_id'] for p in proposals}
    for proposal in proposals:
        p = proposal['proposal_id']
        if p in allowed and scores[p+'_G'] >= .8 and scores[p+'_L'] <= .1 and scores[p+'_X'] <= .1:
            return _decision(pool, proposal, score_status, 'raw_g_l_x_and_gate')
    return _decision(pool, None, score_status, 'no_proposal_passes_all_gates')


def selected_facts(facts, ids):
    by_id = fact_index(facts)
    validate_ids(ids, by_id, exact=20)
    return [deepcopy(by_id[fid]) for fid in ids]


def token_eligible_proposals(question, reference_date, facts, pool, proposals, token_counter, reader_serializer):
    """Same-state sensitivity: filter final actions only; no JEV re-request."""
    validate_proposals(pool, proposals)
    base = token_counter(reader_serializer(question, selected_facts(facts, pool['baseline_ids']), reference_date))
    if type(base) is not int or base < 0:
        raise ValueError('Token counter must return a nonnegative integer')
    counts, allowed = {}, []
    for p in proposals:
        count = token_counter(reader_serializer(question, selected_facts(facts, p['selected_ids']), reference_date))
        if type(count) is not int or count < 0:
            raise ValueError('Invalid token count')
        counts[p['proposal_id']] = count
        if count <= base:
            allowed.append(p['proposal_id'])
    return {'baseline_tokens': base, 'proposal_tokens': counts, 'eligible_proposal_ids': allowed,
        'state_changed': False, 'new_jev_requests': 0}


def pilot_ids(jobs):
    out = []
    for ci in range(10):
        rows = [j for j in jobs if j['conversation'] == ci]
        ranked = sorted(rows, key=lambda j: (hashlib.sha256(
            f'r01-pilot-v1|{ci}|{j["query_id"]}'.encode()).hexdigest(), j['query_id']))
        if len(ranked) < 8:
            raise ValueError('Pilot requires at least8 slots per conversation')
        out.extend(j['query_id'] for j in ranked[:8])
    return out
