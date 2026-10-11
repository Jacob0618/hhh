"""Pure R03 membership selection from an unchanged frozen20..80 candidatepool."""
from copy import deepcopy
from collections.abc import Mapping
import json
import math
from jev_mem.evidence_exchange_r01 import digest,request_hash,serialized_signature,parse_scores
from jev_mem.evidence_ordering_r02 import date_value
from tools.pilot400_recovery_rank_v1 import stable_bm25

VERSION='evidence-selection-r03-v1'
ARMS=('baseline','cheap','jev')
PROTOCOL_SHA256='fb123e30f7a4f9177565ca830169fecee9adfd4bb6a7af70d1439d4192c7e2bb'
POLICY={'version':VERSION,'selected_count':20,'pool_min':20,'pool_cap':80,
    'signature_utf8_byte_cap':131072,'response_model':'jev-1.13.0',
    'calibration':'raw_probabilities_uncalibrated_ranking_feature','threshold':None,
    'reader_order':'frozen_pool_order','tie_break':'frozen_pool_index',
    'unknown_action':'any_unavailable_score_full_originalS0','bm25_corpus':'frozen20..80pool',
    'bm25_k1':1.2,'bm25_b':.75,'new_retrieval':False,'prior_per_fact_score_reuse':False}


def validate_records(records):
    if (not isinstance(records,list) or not 20<=len(records)<=80
        or any(not isinstance(f,dict)or set(f)!={'id','text','observation_date'}
            or not isinstance(f['id'],str)or not f['id']
            or not isinstance(f['text'],str)or not f['text'].strip()
            or(f['observation_date']is not None and not isinstance(f['observation_date'],str))for f in records)
        or len({f['id']for f in records})!=len(records)):
        raise ValueError('Frozen20..80 unique canonical Reader records required')


def selected_facts(records,indices):
    validate_records(records)
    if (not isinstance(indices,list)or len(indices)!=20 or any(type(i)is not int for i in indices)
        or indices!=sorted(set(indices))or not set(indices)<=set(range(len(records)))):
        raise ValueError('Exactly20 unique indices restored to frozenpool order required')
    return [deepcopy(records[i])for i in indices]


def _decision(records,ranked_indices,status,reason,scores=None):
    members=ranked_indices[:20];indices=sorted(members);selected=selected_facts(records,indices)
    baseline=records[:20]
    return {'selected_ids':[f['id']for f in selected],'pool_selected_indices':indices,
        'score_ranked_member_indices':members,'score_status':status,'reason':reason,
        'changed_membership':set(indices)!=set(range(20)),
        'changed_order':selected!=baseline,'retained_baseline_count':len(set(indices)&set(range(20))),
        'calibration':POLICY['calibration'],'accepted_proposal_id':None,'swaps':None,
        'stable_tie_pairs':sum(scores[i]==scores[j]for i in range(len(records))for j in range(i+1,len(records)))if scores is not None else None}


def choose_baseline(records):
    return _decision(records,list(range(20)),'inherited','unchanged_native_baseline')


def choose_cheap(question,reference_date,records):
    validate_records(records)
    if not isinstance(question,str)or not question.strip():raise ValueError('Nonempty query required')
    query=question+'\nReference date: '+date_value(reference_date)
    docs=[f['text']+'\nObservation date: '+date_value(f['observation_date'])for f in records]
    scores=stable_bm25(query,docs)
    if len(scores)!=len(records)or any(not math.isfinite(v)or v<0 for v in scores):
        raise ValueError('Invalid fresh samepool BM25 scores')
    decision=_decision(records,sorted(range(len(records)),key=lambda i:(-scores[i],i)),
        'complete','stable_top20_bm25_members_restored_poolorder',scores)
    decision['bm25_scores']=scores
    return decision


def build_request(question,reference_date,records,protocol):
    validate_records(records)
    if not isinstance(question,str)or not question.strip():raise ValueError('Nonempty query required')
    template=protocol['jev']['question_instructions_template'];questions={}
    for i,f in enumerate(records):
        payload={'text':f['text'],'observation_date':date_value(f['observation_date'])}
        literal=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        questions[f'f{i:02d}']={'type':'noul','instructions':template.replace('{record_json}',literal)}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone',
        'body':{'model':'jev-1.13.0','state':{'query':{'text':question,'reference_date':date_value(reference_date)}},'questions':questions}}


def choose_jev(records,scores,*,score_status='complete'):
    validate_records(records)
    if score_status!='complete':
        if score_status not in ('failed','unknown','not_dispatched')or scores is not None:
            raise ValueError('Unavailable scores must be explicit with no invented values')
        return _decision(records,list(range(20)),score_status,'unavailable_scores_keep_baseline')
    keys={f'f{i:02d}'for i in range(len(records))}
    if not isinstance(scores,Mapping)or set(scores)!=keys:raise ValueError('Alln scores required; no partial/zero fill')
    values=[scores[f'f{i:02d}']for i in range(len(records))]
    if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 0<=v<=1 for v in values):
        raise ValueError('Finite nonboolean probabilities required for alln')
    return _decision(records,sorted(range(len(records)),key=lambda i:(-values[i],i)),
        'complete','stable_top20_raw_usefulness_members_restored_poolorder',values)
