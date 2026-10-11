"""Pure conservative R05 window refinement. No I/O, API, gold or generated text.

Original top4 anchors remain admitted. Complete query-independent S8 judgments
propose one alternative per anchor; a Q-conditioned comparison decides replacement.
Frozen closure and local R05 ranking operate only after this comparison.
"""
from copy import deepcopy
from collections.abc import Mapping
import json

from jev_mem import source_append_r05 as prior
from jev_mem import source_support_stack_v1 as stack
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature, parse_scores

VERSION = 'r05-refinement-v1'
MODEL = stack.MODEL
POLICY = {'version': VERSION, 'original_anchor_cap': 4, 'alternative_per_anchor': 1,
    'variant_cap': 8, 'support_threshold_strict_gt': .5,
    'comparison_threshold_strict_gt': .5, 'questions_per_request': 4,
    'attachment_cap': 2, 'attachment_utf8_byte_cap': 4096,
    'signature_utf8_byte_cap': 131072,
    'alternative_order': 'whole_window_query_bm25_then_support_probability_then_center_position',
    'query_corpus': 'all_eligible_S8_windows_before_support_filter',
    'collision': 'earliest_original_candidate_merge_aliases_no_refill',
    'unknown': 'raise_and_stop_review', 'failed': 'keep_all_original_packets',
    'calibration': 'operational_uncalibrated_half_threshold_not_truth_probability'}
COMPARISON_PREDICATE = (
    'Compared with the original source attachment, does the alternative source '
    'attachment add or clarify an explicit detail, relation, condition, speaker '
    'attribution or temporal wording requested by the question, while retaining '
    'the original attachment\'s support needed to answer that question? Assess only '
    'this displayed original memory and these two alternative windows. Preserve '
    'subject, negation, quantities, planned versus completed status and event '
    'relationships. A related topic, a larger window, or support for a different '
    'claim in the memory alone is insufficient. Do not invent bridges or assume '
    'completeness of other memories or the conversation. Source observation date '
    'is not event date; machine captions are annotations, not speaker assertions.')
DISPLAY_PREFIX = COMPARISON_PREDICATE + '\nDisplayed comparison:\n'
PROTOCOL = {'schema': VERSION+'-protocol', 'policy': POLICY,
    'comparison_predicate': COMPARISON_PREDICATE,
    'support_protocol_sha256': digest(stack.PROTOCOL),
    'closure_predicate': stack.CLOSURE_PREDICATE, 'local_predicate': stack.LOCAL_PREDICATE}


def _query(question, reference_date):
    if (not isinstance(question, str) or not question.strip()
        or (reference_date is not None and not isinstance(reference_date, str))):
        raise ValueError('Question/date whitelist required')
    return {'text': question, 'reference_date': reference_date}


def validate_request(signature, *, allow_oversize=False):
    if (not isinstance(signature, dict) or set(signature) != {'provider','model','operation','body'}
        or signature['provider'] != 'jev' or signature['model'] != MODEL
        or signature['operation'] != 'systemone'):
        raise ValueError('Pinned JEV systemone signature required')
    b = signature['body']
    if (not isinstance(b, dict) or set(b) != {'model','state','questions'}
        or b['model'] != MODEL or not isinstance(b['state'],dict)
        or set(b['state']) != {'query'} or not isinstance(b['questions'],dict)
        or not 1 <= len(b['questions']) <= 4):
        raise ValueError('Bounded comparison body whitelist required')
    q = b['state']['query']
    if not isinstance(q,dict) or set(q) != {'text','reference_date'}:
        raise ValueError('No gold/QA fields in query')
    _query(q['text'],q['reference_date'])
    if not set(b['questions']) <= {f'm{i:02d}' for i in range(4)}:
        raise ValueError('Frozen comparison keys only')
    for row in b['questions'].values():
        if (not isinstance(row,dict) or set(row) != {'type','instructions'}
            or row['type'] != 'noul' or not isinstance(row['instructions'],str)
            or not row['instructions'].startswith(DISPLAY_PREFIX)):
            raise ValueError('Frozen comparative Noul predicate required')
        payload = json.loads(row['instructions'][len(DISPLAY_PREFIX):])
        if (not isinstance(payload,dict) or set(payload) != {'original_memory',
            'original_source_attachment','alternative_source_attachment'}):
            raise ValueError('Comparison payload whitelist differs')
        m = payload['original_memory']
        if (not isinstance(m,dict) or set(m) != {'text','observation_date'}
            or not isinstance(m['text'],str)
            or (m['observation_date'] is not None and not isinstance(m['observation_date'],str))
            or any(not isinstance(payload[k],str) for k in
                ('original_source_attachment','alternative_source_attachment'))):
            raise ValueError('Memory/source projection whitelist differs')
        if row['instructions'] != DISPLAY_PREFIX + prior.canonical(payload):
            raise ValueError('Canonical exact comparison instructions required')
    if not allow_oversize and len(serialized_signature(signature)) > 131072:
        raise ValueError('Oversize signature cannot dispatch or truncate')
    return 'multi_compare'


def parse_response(response, signature):
    validate_request(signature)
    return parse_scores(response,signature)


def build_candidate_variants(question, records, scopes, turns, *, conversation,
                             support_results, reference_date=None, protocol=None):
    """Return exact original locator and unscored multi_plan, not an eager replacement.

    support_results[fid] must be {plan, scores, status}; the plan is rebuilt from
    this current fact/scope/source and every S batch must be present. Missing or
    known failed support retains the original packet, never deletes its fact.
    """
    if protocol is not None and protocol != PROTOCOL:
        raise ValueError('Frozen refinement protocol differs')
    query = _query(question,reference_date)
    prior.validate_original(records)
    byid = stack._turns(turns)
    if len(scopes) != 20 or not isinstance(support_results,Mapping):
        raise ValueError('Twenty scopes and support-result mapping required')
    canonical_scopes = [stack._scope(s,byid) for s in scopes]
    original = prior.build_candidates(question,records,canonical_scopes,turns,conversation=conversation)
    pairs = []; diagnostics = []
    for i,old in enumerate(original['candidates']):
        slot = old['origin']['anchor_slot']; fact = records[slot]
        supplied = support_results.get(fact['id']); alternative = None
        diag = {'candidate_index': i, 'anchor_slot': slot, 'fact_id': fact['id'],
            'old_packet_id': old['packet_id'], 'reason': 'support_absent_keep_original',
            'support_plan_sha256': None, 'support_status': None, 'window_query_bm25': [],
            'chosen_support_window_index': None, 'chosen_support_probability': None}
        if supplied is not None:
            if not isinstance(supplied,dict) or set(supplied) != {'plan','scores','status'}:
                raise ValueError('Exact support plan/scores/status required')
            plan = stack.build_support_plan(fact,canonical_scopes[slot],turns,conversation=conversation)
            if supplied['plan'] != plan:
                raise ValueError('Support plan differs from current exact fact/scope/source')
            if supplied['status'] == 'unknown':
                raise ValueError('Unknown support requires terminal review')
            result = stack.resolve_support(plan,supplied['scores'],score_status=supplied['status'])
            diag.update(support_plan_sha256=digest(plan),support_status=supplied['status'])
            windows = plan['windows']
            # IDF is fixed by every eligible window, before support filtering.
            docs = ['\n'.join(prior.document(byid[t['source_turn_id']])
                for t in w['packet_view']['turns']) for w in windows]
            qs = prior.stable_bm25(question,docs) if docs else []
            diag['window_query_bm25'] = qs
            options = []
            if result['scores'] is not None:
                for j,w in enumerate(windows):
                    p = result['scores'][f's{j:02d}']
                    if p > .5 and w['packet_id'] != old['packet_id']:
                        options.append((j,w,p))
            if options:
                j,w,p = min(options,key=lambda x:(-qs[x[0]],-x[2],x[1]['origin']['center_global_index']))
                center = byid[w['origin']['center_turn_id']]
                alternative = stack._base_packet(fact,slot,center,canonical_scopes[slot],turns,
                    conversation,qscore=qs[j],fscore=w['origin']['anchor_bm25'])
                prior.validate_candidates([alternative],records)
                diag.update(reason='partial_support_alternative_requires_question_comparison',
                    chosen_support_window_index=j,chosen_support_probability=p)
                pairs.append({'key':f'm{i:02d}','candidate_index':i,'anchor_slot':slot,
                    'original':deepcopy(old),'alternative':alternative,
                    'support_probability':p,'whole_window_query_bm25':qs[j],
                    'support_window_index':j,'support_plan_sha256':digest(plan)})
            else:
                diag['reason'] = ('unavailable_support_keep_original' if result['scores'] is None
                    else 'no_distinct_supported_alternative_keep_original')
        diagnostics.append(diag)
    requests = []
    if pairs:
        questions = {}
        for pair in pairs:
            fact = records[pair['anchor_slot']]
            payload = {'original_memory':{k:fact[k] for k in ('text','observation_date')},
                'original_source_attachment':pair['original']['attachment'],
                'alternative_source_attachment':pair['alternative']['attachment']}
            questions[pair['key']] = {'type':'noul','instructions':DISPLAY_PREFIX+prior.canonical(payload)}
        sig = {'provider':'jev','model':MODEL,'operation':'systemone',
            'body':{'model':MODEL,'state':{'query':query},'questions':questions}}
        validate_request(sig,allow_oversize=True)
        size = len(serialized_signature(sig))
        requests.append({'batch_index':0,'signature':sig,'request_hash':request_hash(sig),
            'signature_sha256':digest(sig),'signature_utf8_bytes':size,
            'key_to_candidate_index':{p['key']:p['candidate_index'] for p in pairs},
            'key_to_pair_index':{p['key']:j for j,p in enumerate(pairs)},
            'dispatch_eligible':size <= 131072,
            'skip_reason':None if size <= 131072 else 'signature_oversize'})
    multi = {'schema':VERSION+'-multi-plan','conversation':conversation,
        'query':query,'records':deepcopy(records),'original_locator':deepcopy(original),
        'protocol_sha256':digest(PROTOCOL),'pairs':pairs,'requests':requests,
        'diagnostics':diagnostics,'variant_count':len(original['candidates'])+len(pairs),
        'top4_admission_unchanged':True,'candidate_universe': 'original_anchors_with_window_alternatives'}
    return {'original_locator':original,'multi_plan':multi}


def resolve_multi(plan, scores, *, score_status='complete'):
    """Return {locator, decisions}; unknown is a blocking error, never a fallback success.

    Durable callers must rebuild and compare the complete plan before accepting
    scores. A known failed/oversize single batch retains all original windows.
    """
    if plan['schema'] != VERSION+'-multi-plan' or plan['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Current frozen multi plan required')
    if score_status == 'unknown':
        raise ValueError('Unknown multi comparison requires terminal review')
    for r in plan['requests']:
        validate_request(r['signature'],allow_oversize=True)
        if r['request_hash'] != request_hash(r['signature']):
            raise ValueError('Request hash differs')
    keys = [p['key'] for p in plan['pairs']]
    values = stack._scores(scores,keys,score_status,plan['requests'])
    records = plan['records']; original = plan['original_locator']
    prior.validate_original(records); prior.validate_candidates(original['candidates'],records)
    candidates = deepcopy(original['candidates']); decisions = []
    for pair in plan['pairs']:
        i = pair['candidate_index']
        if candidates[i] != pair['original'] or pair['anchor_slot'] != candidates[i]['origin']['anchor_slot']:
            raise ValueError('Comparison must bind original admitted anchor')
        prior.validate_candidates([pair['alternative']],records)
        if pair['alternative']['origin']['anchor_slot'] != pair['anchor_slot']:
            raise ValueError('Alternative may not change consuming anchor')
        p = None if values is None else values[pair['key']]
        use = p is not None and p > .5
        if use: candidates[i] = deepcopy(pair['alternative'])
        decisions.append({'key':pair['key'],'candidate_index':i,'anchor_slot':pair['anchor_slot'],
            'score_status':score_status,'comparison_probability':p,'replaced':use,
            'old_packet_id':pair['original']['packet_id'],
            'alternative_packet_id':pair['alternative']['packet_id'],
            'reason':'comparative_question_support_strictly_above_half' if use
                else 'known_unavailable_keep_original' if p is None else 'comparison_not_above_half_keep_original'})
    unique = []; seen = {}; collisions = []
    for i,c in enumerate(candidates):
        if c['packet_id'] in seen:
            k = seen[c['packet_id']]
            for alias in c['origin_aliases']:
                if alias not in unique[k]['origin_aliases']:unique[k]['origin_aliases'].append(deepcopy(alias))
            collisions.append({'dropped_original_index':i,'kept_effective_index':k,
                'duplicate_attachment_utf8_bytes':c['attachment_utf8_bytes']})
        else: seen[c['packet_id']] = len(unique); unique.append(c)
    prior.validate_candidates(unique,records)
    locator = {**deepcopy(original),'candidates':unique,
        'multi_refinement':{'enabled':True,'plan_sha256':digest(plan),'score_status':score_status,
            'scores':values,'decisions':deepcopy(decisions),'collisions':collisions,
            'original_candidate_count':len(candidates),'unique_candidate_count':len(unique),
            'variant_count':plan['variant_count'],'top4_admission_not_repeated':True,'no_outside_refill':True}}
    return {'locator':locator,'decisions':decisions}


# Exact frozen closure/local rendering, reused after multi_compare resolution.
build_closure_plan = stack.build_closure_plan
resolve_closure = stack.resolve_closure
apply_closures = stack.apply_closures
build_rank_request = stack.build_rank_request
choose_rank = stack.choose_rank
selected_facts = stack.selected_facts
