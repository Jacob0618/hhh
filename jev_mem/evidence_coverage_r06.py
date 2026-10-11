"""Pure fixed-pool coverage selection; builders and parsers never dispatch models."""
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores
from jev_mem.evidence_ordering_r02 import date_value
from jev_mem.evidence_selection_r03 import validate_records, selected_facts, _decision
from jev_mem.gemini38_provider_v1 import generation_body, response_text
from tools.pilot400_recovery_rank_v1 import stable_bm25

VERSION='evidence-coverage-r06-v1'
ARMS=('baseline','cheap','jev')
PROTOCOL_SHA256='83e7a9456fae1dac2fd2177957c4b4a36965c22de67c4df65f1d7c7b58872ffa'
POLICY={'version':VERSION,'selected_count':20,'pool_min':20,'pool_cap':80,
 'facet_min':1,'facet_cap':4,'subrequirement_cap':3,'subrequirement_utf8_byte_cap':512,
 'facet_response_utf8_byte_cap':2048,'signature_utf8_byte_cap':131072,
 'response_model':'jev-1.13.0','facet_model':'gemini-3.8-flash',
 'calibration':'raw_pyes_uncalibrated_empirical_max_coverage','threshold':None,
 'reader_order':'frozen_pool_order','unknown_action':'full_originalS0',
 'bm25_k1':1.2,'bm25_b':.75,'new_retrieval':False,'source_append':False}


def _question(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Exact nonempty question required')
    return value


def _key(value):
    return ' '.join(value.split()).casefold()


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON object key')
        result[key]=value
    return result


def parse_facets(text,question,protocol=None):
    """Only syntactic validation; valid strings are not semantic certificates."""
    _question(question)
    if not isinstance(text,str):raise ValueError('Actual ordinary Gemini text required')
    stripped=text.strip()
    if len(stripped.encode('utf-8'))>2048:raise ValueError('Facet response byte cap exceeded')
    value=json.loads(stripped,object_pairs_hook=_unique_object)
    if not isinstance(value,dict) or set(value)!={'subrequirements'}:
        raise ValueError('Exactly one subrequirements object required')
    raw=value['subrequirements']
    if not isinstance(raw,list) or len(raw)>3:raise ValueError('Zero to three subrequirements required')
    retained=[];seen={_key(question)};duplicates=0
    for requirement in raw:
        if not isinstance(requirement,str) or not requirement.strip():
            raise ValueError('Nonempty requirement strings required')
        requirement=requirement.strip()
        if len(requirement.encode('utf-8'))>512:raise ValueError('Requirement byte cap exceeded')
        key=_key(requirement)
        if key in seen:duplicates+=1;continue
        retained.append(requirement);seen.add(key)
    return {'subrequirements':retained,'facets':[question,*retained],
            'raw_count':len(raw),'duplicate_count':duplicates,'m':len(retained)+1}


def parse_facet_response(response,question,protocol=None):
    return parse_facets(response_text(response),question,protocol)


def build_facet_request(question,reference_date,protocol):
    payload=json.dumps({'question':_question(question),'reference_date':date_value(reference_date)},
        ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    prompt=protocol['facet_generation']['prompt'].replace('{input_json}',payload)
    return {'provider':'gemini','model':'gemini-3.8-flash','operation':'generateContent',
            'body':generation_body(prompt,system=None,json_output=False,max_output=8192)}


def validate_facets(facets,question=None):
    if (not isinstance(facets,list) or not 1<=len(facets)<=4
        or any(not isinstance(f,str) or not f.strip() for f in facets)
        or len({_key(f) for f in facets})!=len(facets)
        or any(len(f.encode('utf-8'))>512 for f in facets[1:])):
        raise ValueError('Canonical originalQ plus <=3 bounded distinct requirements required')
    if question is not None and facets[0]!=question:raise ValueError('Facet0 must be exact originalQ')


def build_request(question,reference_date,requirement,records,protocol):
    validate_records(records);_question(question);_question(requirement)
    template=protocol['jev']['instruction_template']
    questions={f'f{i:02d}':{'type':'noul','instructions':template.format(
        text=f['text'],observation_date=date_value(f['observation_date']))}
        for i,f in enumerate(records)}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone',
        'body':{'model':'jev-1.13.0','state':{'question':question,
          'reference_date':date_value(reference_date),'requirement':requirement},'questions':questions}}


def choose_coverage(matrix):
    """Optimize the frozen empirical feature objective, not answer probability."""
    if not isinstance(matrix,list) or not 20<=len(matrix)<=80:
        raise ValueError('20..80 matrix rows required')
    if not all(isinstance(row,list) for row in matrix):raise ValueError('Explicit row lists required')
    m=len(matrix[0])
    if not 1<=m<=4 or any(len(row)!=m for row in matrix):raise ValueError('Complete1..4-column matrix required')
    if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 0<=v<=1
           for row in matrix for v in row):raise ValueError('Finite nonboolean unit-range scores required')
    covered=[0.0]*m;chosen=[];steps=[]
    for _ in range(20):
        options=[]
        for i,row in enumerate(matrix):
            if i in chosen:continue
            delta=math.fsum(max(0.0,row[j]-covered[j]) for j in range(m))
            options.append(((-delta,-row[0],i),i,delta))
        _,i,delta=min(options)
        chosen.append(i);covered=[max(covered[j],matrix[i][j])for j in range(m)]
        steps.append({'pool_index':i,'delta_sum':delta,'delta_mean':delta/m,'covered':deepcopy(covered)})
    return {'indices':sorted(chosen),'greedy_indices':chosen,'marginal_gains':steps,
        'covered':covered,'objective_value':math.fsum(covered)/m,
        'zero_fill_count':sum(x['delta_sum']==0.0 for x in steps)}


def _from_matrix(records,matrix,reason):
    trace=choose_coverage(matrix)
    result=_decision(records,trace['greedy_indices'],'complete',reason)
    result.update({'coverage':trace,'matrix':deepcopy(matrix),'facet_count':len(matrix[0]),
                   'calibration':POLICY['calibration']})
    return result


def choose_baseline(records,*,score_status='inherited',reason='unchanged_native_baseline'):
    validate_records(records)
    result=_decision(records,list(range(20)),score_status,reason)
    result.update({'coverage':None,'matrix':None,'facet_count':None,'calibration':POLICY['calibration']})
    return result


def cheap_matrix(question,reference_date,records,facets):
    validate_records(records);validate_facets(facets,_question(question))
    documents=[f['text']+'\nObservation date: '+date_value(f['observation_date'])for f in records]
    columns=[];raw_columns=[]
    for facet in facets:
        scores=stable_bm25(facet+'\nReference date: '+date_value(reference_date),documents)
        if len(scores)!=len(records)or any(isinstance(v,bool)or not isinstance(v,(int,float))
            or not math.isfinite(v)or v<0 for v in scores):raise ValueError('Invalid samepool BM25 column')
        maximum=max(scores);raw_columns.append(scores)
        columns.append([v/maximum if maximum>0 else 0.0 for v in scores])
    return [[column[i]for column in columns]for i in range(len(records))],raw_columns


def choose_cheap(question,reference_date,records,facets,*,facet_status='complete'):
    if facet_status!='complete':
        if facet_status not in ('failed','not_dispatched')or facets is not None:
            raise ValueError('Failed generator requires explicit absent facets; unknown blocks seal')
        return choose_baseline(records,score_status=facet_status,reason='facet_unavailable_keep_baseline')
    matrix,raw=cheap_matrix(question,reference_date,records,facets)
    decision=_from_matrix(records,matrix,'normalized_bm25_facet_coverage_poolorder')
    decision['bm25_raw_columns']=raw
    return decision


def choose_jev(records,score_columns,*,score_status='complete'):
    validate_records(records)
    if score_status!='complete':
        if score_status not in ('failed','not_dispatched')or score_columns is not None:
            raise ValueError('Unavailable full matrix must be absent; unknown cannot seal')
        return choose_baseline(records,score_status=score_status,reason='unavailable_column_keep_baseline')
    if not isinstance(score_columns,list)or not 1<=len(score_columns)<=4:
        raise ValueError('Every facet column required')
    keys={f'f{i:02d}'for i in range(len(records))}
    if any(not isinstance(column,Mapping)or set(column)!=keys for column in score_columns):
        raise ValueError('All exact candidate scores required in all columns')
    matrix=[[column[f'f{i:02d}']for column in score_columns]for i in range(len(records))]
    return _from_matrix(records,matrix,'raw_noul_facet_coverage_poolorder')
