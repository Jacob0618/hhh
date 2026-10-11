"""Pure R02 permutations of exact frozen20 records; no API, truth or recall claim."""
from copy import deepcopy
import json
import math
from collections.abc import Mapping

from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores
from tools.pilot400_recovery_rank_v1 import stable_bm25

VERSION = 'evidence-ordering-r02-v1'
ARMS = ('baseline', 'cheap', 'jev')
PROTOCOL_SHA256 = '42c6e8030908dd1ad16406947d15986dba1f72e8189878210f7bd88613c9353d'
POLICY = {'version':VERSION, 'selected_count':20, 'signature_utf8_byte_cap':131072,
    'response_model':'jev-1.13.0', 'response_answer_type':'noul_required',
    'calibration':'raw_probabilities_uncalibrated_ranking_feature',
    'unknown_action':'any_unavailable_score_entire_original_order',
    'bm25_k1':1.2, 'bm25_b':.75, 'bm25_corpus':'original20',
    'tie_break':'original_index', 'threshold':None, 'record_changes_permitted':False}


def validate_records(records):
    if (not isinstance(records, list) or len(records) != 20
        or any(not isinstance(f, dict) or set(f) != {'id','text','observation_date'}
            or not isinstance(f['id'], str) or not f['id']
            or not isinstance(f['text'], str) or not f['text'].strip()
            or (f['observation_date'] is not None and not isinstance(f['observation_date'], str)) for f in records)
        or len({f['id'] for f in records}) != 20):
        raise ValueError('Exact20 unique canonical Reader records required')


def date_value(value):
    if value is not None and not isinstance(value, str):
        raise ValueError('Date must be original string or null')
    return value if value is not None else ''


def validate_permutation(records, permutation):
    validate_records(records)
    if (not isinstance(permutation, list) or len(permutation) != 20
        or any(type(i) is not int for i in permutation) or set(permutation) != set(range(20))):
        raise ValueError('A complete permutation of original20 is required')


def selected_facts(records, permutation):
    validate_permutation(records, permutation)
    return [deepcopy(records[i]) for i in permutation]


def _decision(records, permutation, status, reason, scores=None):
    selected = selected_facts(records, permutation)
    return {'selected_ids':[f['id'] for f in selected], 'permutation':permutation,
        'score_status':status, 'reason':reason, 'changed_order':selected != records,
        'calibration':POLICY['calibration'], 'accepted_proposal_id':None, 'swaps':0,
        'stable_tie_pairs':sum(scores[i] == scores[j] for i in range(20) for j in range(i+1,20)) if scores is not None else None}


def choose_baseline(records):
    return _decision(records, list(range(20)), 'inherited', 'unchanged_native_baseline')


def choose_cheap(question, reference_date, records):
    validate_records(records)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Nonempty query required')
    query = question+'\nReference date: '+date_value(reference_date)
    docs = [f['text']+'\nObservation date: '+date_value(f['observation_date']) for f in records]
    scores = stable_bm25(query, docs)
    if len(scores) != 20 or any(not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError('Invalid BM25 result')
    decision = _decision(records, sorted(range(20),key=lambda i:(-scores[i],i)),
        'complete','stable_same20_bm25', scores)
    decision['bm25_scores'] = scores
    return decision


def build_request(question, reference_date, records, protocol):
    validate_records(records)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Nonempty query required')
    template = protocol['jev']['question_instructions_template']
    questions = {}
    for i, fact in enumerate(records):
        payload = {'text':fact['text'], 'observation_date':date_value(fact['observation_date'])}
        literal = json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        questions[f'f{i:02d}'] = {'type':'noul','instructions':template.replace('{record_json}',literal)}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone',
        'body':{'model':'jev-1.13.0', 'state':{'query':{'text':question,'reference_date':date_value(reference_date)}},
            'questions':questions}}


def choose_jev(records, scores, *, score_status='complete'):
    validate_records(records)
    if score_status != 'complete':
        if score_status not in ('failed','unknown','not_dispatched') or scores is not None:
            raise ValueError('Unavailable scores must be explicit with no invented values')
        return _decision(records,list(range(20)),score_status,'unavailable_scores_keep_baseline')
    expected = {f'f{i:02d}' for i in range(20)}
    if not isinstance(scores, Mapping) or set(scores) != expected:
        raise ValueError('All20 scores required; unavailable is not zero')
    values = [scores[f'f{i:02d}'] for i in range(20)]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('All20 finite nonboolean Noul probabilities required')
    return _decision(records, sorted(range(20),key=lambda i:(-values[i],i)),
        'complete','stable_raw_explicit_usefulness',values)
