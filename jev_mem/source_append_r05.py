"""Pure R05 bounded current-source projection and append; no provider or QA data."""
from __future__ import annotations
from copy import deepcopy
from collections.abc import Mapping
import json
import math
import re
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores
from tools.pilot400_recovery_rank_v1 import stable_bm25

ARMS = ('baseline', 'cheap', 'jev')
PROTOCOL_SHA256 = '5b9e319b66872eab6f40e71a472bcb2d93e907ff23678e079039898280c913fa'
POLICY = {'version':'source-append-r05-v1', 'selected_count':20, 'candidate_cap':4,
    'attachment_cap':2, 'attachment_utf8_byte_cap':4096, 'signature_utf8_byte_cap':131072,
    'bm25_k1':1.2, 'bm25_b':.75, 'bm25_dates':False, 'threshold':None,
    'unknown_action':'entire_original_N0', 'source_scope':'current_input_scope_only',
    'response_model':'jev-1.13.0', 'response_type':'noul', 'calibration':'uncalibrated_local_incremental_usefulness'}
WRAPPER = '\n\n[Source input context; source observation date is not necessarily event date; machine captions are annotations]\n{rendered_packet}\n[End source input context]'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def project_content(content):
    """Remove rendered image metadata; copy exact non-media/caption substrings and offsets."""
    if not isinstance(content,str): raise ValueError('Source content must be exact text')
    speaker_spans, captions, omitted = [], [], []
    cursor = 0
    # An unterminated annotation is omitted through end, never leaked into speaker text.
    for match in re.finditer(r'\[Sharing image[^\]]*(?:\]|$)',content):
        if cursor < match.start():
            speaker_spans.append({'start':cursor,'end':match.start(),'text':content[cursor:match.start()]})
        payload = match.group(); cap_start = None
        if payload.endswith(']'):
            prefix = '[Sharing image that shows: '
            marker = '. The image shows: '
            if payload.startswith(prefix): cap_start = match.start()+len(prefix)
            elif payload.startswith('[Sharing image - query: ') and marker in payload:
                # Use the rendering delimiter; ambiguous multiple delimiters are unknown.
                if payload.count(marker)==1: cap_start=match.start()+payload.index(marker)+len(marker)
        if cap_start is not None:
            cap_end=match.end()-1
            captions.append({'start':cap_start,'end':cap_end,'text':content[cap_start:cap_end]})
            omitted.append({'start':match.start(),'end':match.end(),'reason':'rendered_media_markup_and_query_removed'})
        else:
            omitted.append({'start':match.start(),'end':match.end(),'reason':'unresolved_media'})
        cursor=match.end()
    if cursor < len(content): speaker_spans.append({'start':cursor,'end':len(content),'text':content[cursor:]})
    return {'speaker_text':''.join(s['text'] for s in speaker_spans),
        'machine_captions':[s['text'] for s in captions], 'speaker_spans':speaker_spans,
        'caption_spans':captions, 'omitted_media':omitted}


def source_turns(group_inputs, group_ref):
    """Index only admitted envelopes, with exact original order and JSON pointers."""
    envs=group_inputs.get('envelopes')
    if not isinstance(envs,list): raise ValueError('Source envelopes list required')
    turns=[]; seen=set(); sessions=set()
    for ei,env in enumerate(envs):
        session=env.get('session_id'); ids=env.get('source_turn_ids'); messages=env.get('messages')
        if (not isinstance(session,str) or not session or session in sessions
            or not isinstance(ids,list) or not isinstance(messages,list) or len(ids)!=len(messages)
            or any(not isinstance(t,str) or not t for t in ids) or len(set(ids))!=len(ids)):
            raise ValueError('Unique sessions/turns and exact ID/message zip required')
        sessions.add(session)
        date=env.get('observation_date')
        if not isinstance(date,str): raise ValueError('Exact source observation date required')
        for ti,(tid,msg) in enumerate(zip(ids,messages)):
            if tid in seen or not isinstance(msg,dict) or not isinstance(msg.get('content'),str):
                raise ValueError('Duplicate or malformed source message')
            seen.add(tid); projection=project_content(msg['content'])
            turns.append({'source_turn_id':tid,'session_id':session,'observation_date':date,
                'session_index':ei,'turn_index':ti,'global_index':len(turns),
                'view':{'source_turn_id':tid,'speaker_text':projection['speaker_text'],
                    'machine_captions':projection['machine_captions']},
                'projection':projection,'source':deepcopy(group_ref),
                'json_pointer':f'/envelopes/{ei}/messages/{ti}/content',
                'content_sha256':digest(msg['content'])})
    return turns


def validate_original(records):
    if (not isinstance(records,list) or len(records)!=20 or len({f.get('id') for f in records if isinstance(f,dict)})!=20
        or any(not isinstance(f,dict) or set(f)!= {'id','text','observation_date'}
            or not isinstance(f['id'],str) or not f['id'] or not isinstance(f['text'],str) or not f['text'].strip()
            or (f['observation_date'] is not None and not isinstance(f['observation_date'],str)) for f in records)):
        raise ValueError('Original twenty exact canonical Reader records required')


def attachment(view):
    return WRAPPER.replace('{rendered_packet}',canonical(view))


def document(turn):
    v=turn['view']
    return v['speaker_text']+''.join('\nMachine caption: '+cap for cap in v['machine_captions'])


def build_candidates(question, records, scopes, turns, *, conversation, score_cache=None):
    validate_original(records)
    if not isinstance(question,str) or not question.strip() or len(scopes)!=20: raise ValueError('Question/scopes required')
    byid={t['source_turn_id']:t for t in turns}
    if len(byid)!=len(turns): raise ValueError('Duplicate source turns')
    allowed=[]; exclusions=[]
    for slot,scope in enumerate(scopes):
        if scope is None or not scope.get('source_turn_ids'):
            allowed.append([]);exclusions.append({'slot':slot,'reason':'absent_or_empty_current_scope'});continue
        ids=scope['source_turn_ids'];session=scope.get('session_id')
        if (not isinstance(ids,list) or len(set(ids))!=len(ids)
            or any(tid not in byid or byid[tid]['session_id']!=session for tid in ids)):
            raise ValueError('Current scope IDs must exist uniquely in declared session')
        allowed.append(ids)
    population=set(tid for ids in allowed for tid in ids)
    corpus=[t for t in turns if t['source_turn_id'] in population]
    docs=[document(t) for t in corpus];qscores=stable_bm25(question,docs)
    corpus_digest=digest(docs) if score_cache is not None else None
    corpus_index={t['source_turn_id']:i for i,t in enumerate(corpus)}
    per_fact=[]; all_anchor_scores=[]
    for slot,(fact,ids) in enumerate(zip(records,allowed)):
        cache_key=(corpus_digest,fact['text'])
        if score_cache is not None and cache_key in score_cache:
            fscores=score_cache[cache_key]
        else:
            fscores=stable_bm25(fact['text'],docs)
            if score_cache is not None: score_cache[cache_key]=fscores
        all_anchor_scores.append(fscores)
        options=[];permitted=set(ids)
        for tid in ids:
            center=byid[tid];ci=corpus_index[tid];qscore=qscores[ci];fscore=fscores[ci]
            if qscore<=0 or fscore<=0: continue
            included=[];boundaries={}
            for side,offset in [('left',-1),('center',0),('right',1)]:
                index=center['global_index']+offset
                neighbour=turns[index] if 0<=index<len(turns) else None
                if neighbour is None or neighbour['session_id']!=center['session_id']:
                    boundaries[side]='boundary_missing';continue
                if neighbour['source_turn_id'] not in permitted:
                    boundaries[side]='scope_excluded';continue
                included.append(neighbour);boundaries[side]='included'
            view={'session_id':center['session_id'],'source_observation_date':center['observation_date'],
                'turns':[deepcopy(t['view']) for t in included]}
            text=attachment(view);size=len(text.encode('utf-8'))
            if size>POLICY['attachment_utf8_byte_cap']:
                exclusions.append({'slot':slot,'center':tid,'reason':'whole_attachment_oversize','utf8_bytes':size});continue
            origin={'anchor_slot':slot,'anchor_id':fact['id'],'center_turn_id':tid,
                'center_global_index':center['global_index'],'query_bm25':qscore,'anchor_bm25':fscore}
            options.append({'packet_id':digest({'conversation':conversation,'view':view}), 'packet_view':view,
                'attachment':text,'attachment_utf8_bytes':size,'origin':origin,
                'boundary':boundaries,'turn_provenance':[{k:deepcopy(t[k]) for k in
                    ('source_turn_id','source','json_pointer','content_sha256','projection')} for t in included]})
        options.sort(key=lambda c:(-c['origin']['query_bm25'],-c['origin']['anchor_bm25'],c['origin']['center_global_index']))
        if options: per_fact.append(options[0])
        elif ids: exclusions.append({'slot':slot,'reason':'no_positive_eligible_center','not_full_source_absence':True})
    groups={}
    for c in per_fact: groups.setdefault(c['packet_id'],[]).append(c)
    candidates=[]
    for entries in groups.values():
        entries.sort(key=lambda c:(-c['origin']['anchor_bm25'],c['origin']['anchor_slot'],c['origin']['center_global_index']))
        c=deepcopy(entries[0]);c['origin_aliases']=[deepcopy(x['origin']) for x in entries];candidates.append(c)
    candidates.sort(key=lambda c:(-c['origin']['query_bm25'],-c['origin']['anchor_bm25'],c['origin']['anchor_slot'],c['origin']['center_global_index']))
    candidates=candidates[:POLICY['candidate_cap']]
    if len({c['origin']['anchor_slot'] for c in candidates})!=len(candidates): raise ValueError('Canonical anchor slots must be distinct')
    return {'candidates':candidates,'corpus_turn_ids':[t['source_turn_id'] for t in corpus],
        'query_bm25_scores':qscores,'anchor_bm25_scores':all_anchor_scores,
        'exclusions':exclusions,'per_fact_candidate_count':len(per_fact),'deduplicated_candidate_count':len(groups),
        'source_turn_count':len(corpus),'source_document_utf8_bytes':sum(len(d.encode('utf-8')) for d in docs),
        'source_scope_semantics':'input exposure boundary, not verified semantic provenance'}


def build_request(question, reference_date, records, candidates, protocol):
    validate_original(records);validate_candidates(candidates,records)
    questions={}
    for i,c in enumerate(candidates):
        anchor=records[c['origin']['anchor_slot']]
        literal=canonical({'original_memory':{'text':anchor['text'],'observation_date':anchor['observation_date']},
            'source_attachment':c['attachment']})
        questions[f'f{i:02d}']={'type':'noul','instructions':protocol['jev']['predicate']+'\nDisplayed memory and quoted source context:\n'+literal}
    return {'provider':'jev','model':'jev-1.13.0','operation':'systemone','body':{'model':'jev-1.13.0',
        'state':{'query':{'text':question,'reference_date':reference_date}},'questions':questions}}


def validate_candidates(candidates,records):
    if not isinstance(candidates,list) or len(candidates)>4: raise ValueError('At most four candidates')
    slots=[];ids=[]
    for c in candidates:
        slot=c['origin']['anchor_slot'];slots.append(slot);ids.append(c['packet_id'])
        if (type(slot)is not int or not 0<=slot<20 or c['origin']['anchor_id']!=records[slot]['id']
            or c['attachment']!=attachment(c['packet_view'])
            or c['attachment_utf8_bytes']!=len(c['attachment'].encode('utf-8'))
            or c['attachment_utf8_bytes']>4096): raise ValueError('Candidate anchor/complete rendering differs')
    if len(set(slots))!=len(slots) or len(set(ids))!=len(ids): raise ValueError('No duplicate packet/anchor slots')


def _decision(indices,candidates,status,reason):
    return {'candidate_indices':indices,'attached_slot_indices':[candidates[i]['origin']['anchor_slot'] for i in indices],
        'packet_ids':[candidates[i]['packet_id'] for i in indices],'added_utf8_bytes':sum(candidates[i]['attachment_utf8_bytes'] for i in indices),
        'score_status':status,'reason':reason,'changed_text':bool(indices),'calibration':POLICY['calibration']}


def choose_baseline(records,candidates):
    validate_original(records);validate_candidates(candidates,records)
    return _decision([],candidates,'inherited','unchanged_native_baseline')


def choose_cheap(records,candidates):
    validate_original(records);validate_candidates(candidates,records)
    return _decision(list(range(min(2,len(candidates)))),candidates,'not_needed' if len(candidates)<=2 else 'complete','shared_lexical_top2')


def choose_jev(records,candidates,scores,*,score_status='complete'):
    validate_original(records);validate_candidates(candidates,records);n=len(candidates)
    if n<=2:
        if scores is not None or score_status not in ('not_dispatched','not_needed'): raise ValueError('No-ranking queries must not be dispatched/scored')
        return _decision(list(range(n)),candidates,'not_dispatched','no_ranking_choice')
    if score_status!='complete':
        if score_status not in ('failed','unknown','not_dispatched') or scores is not None: raise ValueError('Unavailable scores are explicit, not zero')
        return _decision([],candidates,score_status,'unavailable_scores_keep_baseline')
    keys={f'f{i:02d}' for i in range(n)}
    if not isinstance(scores,Mapping) or set(scores)!=keys: raise ValueError('All candidate probabilities required')
    values=[scores[f'f{i:02d}'] for i in range(n)]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values): raise ValueError('Finite nonboolean pYES required')
    return _decision(sorted(range(n),key=lambda i:(-values[i],i))[:2],candidates,'complete','raw_local_incremental_usefulness_top2')


def selected_facts(records,candidates,decision):
    validate_original(records);validate_candidates(candidates,records)
    indices=decision['candidate_indices']
    if (not isinstance(indices,list) or len(indices)>2 or len(set(indices))!=len(indices)
        or any(type(i)is not int or not 0<=i<len(candidates) for i in indices)): raise ValueError('At most two unique candidate indices')
    expected=_decision(indices,candidates,decision['score_status'],decision['reason'])
    if decision!=expected: raise ValueError('Decision attachment accounting differs')
    result=deepcopy(records)
    for i in indices:
        c=candidates[i];result[c['origin']['anchor_slot']]['text']+=c['attachment']
    return result
