"""Pure reconciliation of the fixed remaining71 QA consumers and HTTP ledger.

No filesystem, credentials, QA text, answers, gold or provider calls are needed.
Consumers: {conversation, query_id, method, event, [response_usage],
            [resolved_model]}. Events are the preserved ResponseSchemaAPI events.
The caller may pass the full current run ledger; unrelated writer rows are ignored.
Raw request/response replay is a separate gate. This gate proves ledger coverage
and a prior successful physical source for each consumed cache response.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

PHASE = 'pilot_remaining71_qa'
METHOD = 'regroup_untyped_rank'
MODELS = {'pilot_qa_reader': 'gemini-2.5-flash',
          'pilot_strict_judge': 'gemini-3.5-flash-lite'}
COUNTS = {1: 21, 7: 50}


def _time(value):
    if not isinstance(value, str):
        raise ValueError('QA ledger timestamp must be an ISO string')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid QA ledger timestamp') from error
    if result.tzinfo is None:
        raise ValueError('QA ledger timestamp must include timezone')
    return result


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Invalid SHA256 in ' + name)
    return value


def audit_qa_ledger(consumers, ledger, *, phase=PHASE, expected_queries=None):
    """Hard fail on any missing, extra, repeated or wrongly scoped QA event.

    By default 21 conv1 and 50 conv7 questions are required. Each has one Reader;
    a successful Reader requires one Judge, while a failed Reader forbids Judge.
    A separate bounded retry namespace may supply its explicit expected_queries
    [(conversation, query_id), ...]. There is one matching row per consumer.
    Physical request hashes occur once with an owning success or failure event.
    A recovery_cache event needs a same-stage/model/hash
    physical source earlier in both ledger order and timestamp; the response
    fingerprint and usage must agree. Reuse across query contexts is allowed.
    """
    if not isinstance(consumers, list) or not isinstance(ledger, list):
        raise ValueError('Consumers and ledger must be lists')
    if not isinstance(phase, str) or not phase:
        raise ValueError('QA phase must be explicit')
    expected_ids = None
    if expected_queries is not None:
        if not isinstance(expected_queries, (list, tuple)) or not expected_queries:
            raise ValueError('Explicit recovery subset cannot be empty')
        expected_ids = {tuple(x) for x in expected_queries}
        if len(expected_ids) != len(expected_queries):
            raise ValueError('Duplicate explicit recovery question')
        for ci, qid in expected_ids:
            if (type(ci) is not int or ci not in COUNTS or not isinstance(qid, str)
                    or not qid.startswith(f'c{ci}_q') or not qid[len(f'c{ci}_q'):].isdigit()):
                raise ValueError('Explicit recovery question is outside source conversations')
    expected = {}
    qstages = {}
    for consumer in consumers:
        if not isinstance(consumer, dict):
            raise ValueError('Invalid QA consumer')
        ci, qid, method = (consumer.get(k) for k in ('conversation', 'query_id', 'method'))
        event = consumer.get('event')
        if (type(ci) is not int or ci not in COUNTS or method != METHOD
                or not isinstance(qid, str) or not qid.startswith(f'c{ci}_q')
                or not qid[len(f'c{ci}_q'):].isdigit() or not isinstance(event, dict)):
            raise ValueError('Consumer is outside fixed remaining71 identity')
        stage = event.get('stage')
        if stage not in MODELS or event.get('source') not in ('new_response', 'recovery_cache', 'failed_or_unknown_response'):
            raise ValueError('Unexpected QA consumer stage or response source')
        physical = event['source'] != 'recovery_cache'
        if type(event.get('new_physical_attempt')) is not bool or event['new_physical_attempt'] != physical:
            raise ValueError('Consumer physical/cache source flags differ')
        request_hash = _digest(event.get('request_hash'), 'consumer request_hash')
        if event['source'] == 'failed_or_unknown_response':
            if event.get('response_sha256') is not None:
                raise ValueError('Failed or unknown response cannot claim a successful fingerprint')
        else:
            _digest(event.get('response_sha256'), 'consumer response_sha256')
        key = (ci, qid, method, stage, request_hash, not physical)
        if key in expected:
            raise ValueError('Duplicate QA consumer')
        expected[key] = consumer
        stages = qstages.setdefault((ci, qid), {})
        if stage in stages:
            raise ValueError('Duplicate stage for one QA question')
        stages[stage] = event['source']
    if expected_ids is None:
        if len(qstages) != 71 or Counter(ci for ci, _ in qstages) != Counter(COUNTS):
            raise ValueError('Fixed remaining71 requires 21+50 questions')
    elif set(qstages) != expected_ids:
        raise ValueError('Consumed questions differ from explicit recovery subset')
    for stages in qstages.values():
        if 'pilot_qa_reader' not in stages:
            raise ValueError('Every terminal question requires a Reader event')
        if stages['pilot_qa_reader'] == 'failed_or_unknown_response':
            if set(stages) != {'pilot_qa_reader'}:
                raise ValueError('A failed Reader must not have a Judge event')
        elif set(stages) != set(MODELS):
            raise ValueError('A successful Reader requires a terminal Judge event')

    found = {}
    physical_by_hash = {}
    cache_rows = []
    failed_rows = []
    ignored = 0
    for index, row in enumerate(ledger):
        if not isinstance(row, dict):
            raise ValueError('Malformed ledger row')
        context = row.get('context') or {}
        if not isinstance(context, dict):
            raise ValueError('Malformed ledger context')
        stage = row.get('stage')
        is_qa = (stage in MODELS or context.get('phase') == phase
                 or (context.get('query_id') is not None and context.get('method') == METHOD))
        if not is_qa:
            ignored += 1
            continue
        if (stage not in MODELS or context.get('phase') != phase
                or row.get('provider') != 'gemini' or row.get('model') != MODELS[stage]
                or row.get('operation') != 'generateContent'):
            raise ValueError('QA ledger stage/model/provider/phase mismatch')
        ci, qid, method = (context.get(k) for k in ('conversation', 'query_id', 'method'))
        if type(ci) is not int or ci not in COUNTS or method != METHOD:
            raise ValueError('QA ledger conversation/method mismatch')
        cached = row.get('cache_hit')
        if cached is None and row.get('status') == 'transport_error' and row.get('usage_unknown') is True:
            cached = False  # ModelAPI transport-error rows omit cache_hit.
        if type(cached) is not bool:
            raise ValueError('QA ledger must explicitly identify physical versus cache')
        request_hash = _digest(row.get('request_hash'), 'ledger request_hash')
        key = (ci, qid, method, stage, request_hash, cached)
        if key not in expected:
            raise ValueError('Unexpected QA ledger consumer/context/source')
        if key in found:
            raise ValueError('Extra or repeated QA ledger row for one consumer')
        consumer = expected[key]
        timestamp = _time(row.get('timestamp_utc'))
        usage = row.get('usage', {})
        if not isinstance(usage, dict):
            raise ValueError('QA ledger usage must be an object, including when unknown')
        if 'response_usage' in consumer and usage != consumer['response_usage']:
            raise ValueError('Raw response usage differs from ledger usage')
        if 'resolved_model' in consumer and row.get('resolved_model') != consumer['resolved_model']:
            raise ValueError('Resolved model differs from raw response')
        entry = {'index': index, 'timestamp': timestamp, 'stage': stage, 'consumer': consumer, 'row': row}
        found[key] = entry
        if cached:
            if row.get('usage_semantics') != 'reused_response_no_new_model_usage':
                raise ValueError('Cache row has unexpected usage semantics')
            if row.get('status') not in (None, 200):
                raise ValueError('Cached QA response has an error status')
            cache_rows.append(entry)
        else:
            if type(row.get('attempt')) is not int or row['attempt'] != 1:
                raise ValueError('New QA request is not a once-only HTTP200 or failure attempt')
            if request_hash in physical_by_hash:
                raise ValueError('Duplicate physical request hash in remaining71')
            failed = consumer['event']['source'] == 'failed_or_unknown_response'
            if failed:
                transport = row.get('status') == 'transport_error' and row.get('usage_unknown') is True
                http_error = type(row.get('status')) is int and 400 <= row['status'] < 600
                if not (transport or http_error):
                    raise ValueError('Failed consumer has no corresponding HTTP or unknown transport failure')
                failed_rows.append(entry)
            elif type(row.get('status')) is not int or row['status'] != 200:
                raise ValueError('New QA request is not a once-only HTTP200 attempt')
            physical_by_hash[request_hash] = entry
    if set(found) != set(expected):
        raise ValueError('Missing QA ledger rows for preserved consumer events')

    reuse = []
    for entry in cache_rows:
        event = entry['consumer']['event']
        source = physical_by_hash.get(event['request_hash'])
        if source is None:
            raise ValueError('Cache consumer has no successful physical source in current ledger')
        if source['consumer']['event']['source'] != 'new_response' or source['row'].get('status') != 200:
            raise ValueError('Cache consumer cannot use a failed or unknown physical source')
        if source['index'] >= entry['index'] or source['timestamp'] > entry['timestamp']:
            raise ValueError('Cache source does not precede consumption')
        if (source['stage'] != entry['stage']
                or source['consumer']['event']['response_sha256'] != event['response_sha256']
                or source['row']['usage'] != entry['row']['usage']
                or source['row'].get('resolved_model') != entry['row'].get('resolved_model')):
            raise ValueError('Cache consumer differs from its physical response source')
        reuse.append({'query_id': entry['consumer']['query_id'],
                      'conversation': entry['consumer']['conversation'], 'stage': entry['stage'],
                      'request_hash': event['request_hash'],
                      'source_query_id': source['consumer']['query_id'],
                      'source_conversation': source['consumer']['conversation']})
    return {'schema': 'remaining71-consumer-ledger-closure-v2', 'passed': True,
            'consumers': len(consumers), 'questions': len(qstages),
            'questions_by_conversation': {str(ci): count for ci, count in Counter(ci for ci, _ in qstages).items()},
            'phase': phase, 'explicit_retry_subset': expected_ids is not None,
            'matched_qa_ledger_rows': len(found), 'physical_attempts': len(physical_by_hash),
            'physical_http200': len(physical_by_hash) - len(failed_rows),
            'failed_or_unknown_consumers': len(failed_rows),
            'unknown_usage_attempts': sum(e['row'].get('usage_unknown') is True for e in failed_rows),
            'failed_contexts': [{'conversation': e['consumer']['conversation'],
                                 'query_id': e['consumer']['query_id'], 'stage': e['stage'],
                                 'request_hash': e['consumer']['event']['request_hash'],
                                 'status': e['row'].get('status')}
                                for e in failed_rows],
            'cache_consumers': len(cache_rows), 'cache_reuse': reuse,
            'ignored_non_qa_ledger_rows': ignored,
            'response_usage_bindings_checked': sum('response_usage' in c for c in consumers),
            'scope': 'Physical/cache ledger coverage only; raw request/response replay is a separate required gate.'}
