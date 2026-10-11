"""Offline SimpleMem original terminal proof. No DB/model/credential initialization.

All original bytes are shared-read, inventories are checked before and after, and
the original QA functions replay their exact requests and retained responses.
This author audit is not independent launch approval.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace, FunctionType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read

OLD = ROOT / 'runs/five_baselines_gemini25_v1_20261008/simplemem'
SCHEMA = 'simplemem-native-v2-terminal-proof-v1'
METHOD = 'simplemem'
RETRYABLE = {429, 500, 502, 503, 504, 529, 'transport_error'}
ALLOWED_STAGES = {'simplemem_write', 'simplemem_requirements', 'simplemem_query_plan',
                  'simplemem_query_analysis', 'simplemem_completeness', 'simplemem_missing_queries'}
GUARD_SOURCE_SHA = '22671305b4288897e71e1d00b7e3e914b46987d374644222b0585559c13962bb'
_BOUND_INVENTORY = None
_BINDING_CACHE = None
_JSONL_CACHE = None

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest(value, compact=False):
    options = dict(ensure_ascii=False, sort_keys=True, allow_nan=False)
    if compact:
        options['separators'] = (',', ':')
    return sha_bytes(json.dumps(value, **options).encode())

def read(path):
    path = Path(path)
    raw = shared_read(path)
    if _BOUND_INVENTORY is not None and str(path) in _BOUND_INVENTORY:
        require(sha_bytes(raw) == _BOUND_INVENTORY[str(path)], 'Observed original bytes changed: ' + str(path))
    return json.loads(raw.decode('utf-8-sig'))

def file_sha(path):
    return sha_bytes(shared_read(Path(path)))

def binding(path):
    key = str(path)
    if _BINDING_CACHE is not None and key in _BINDING_CACHE:
        return _BINDING_CACHE[key]
    path = Path(path).resolve()
    expected = _BOUND_INVENTORY.get(str(path)) if _BOUND_INVENTORY is not None else None
    value = {'path': str(path), 'sha256': expected or file_sha(path)}
    if _BINDING_CACHE is not None:
        _BINDING_CACHE[key] = value
    return value

def jsonl(path):
    path = Path(path)
    if _JSONL_CACHE is not None and str(path) in _JSONL_CACHE:
        return _JSONL_CACHE[str(path)]
    if not path.exists():
        return []
    raw = shared_read(path)
    if _BOUND_INVENTORY is not None and str(path) in _BOUND_INVENTORY:
        require(sha_bytes(raw) == _BOUND_INVENTORY[str(path)], 'Original JSONL bytes changed')
    require(not raw or raw.endswith(b'\n'), 'Incomplete terminal JSONL: ' + str(path))
    value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if _JSONL_CACHE is not None:
        _JSONL_CACHE[str(path)] = value
    return value

def inventory(root):
    root = Path(root).resolve()
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            require(not p.is_symlink() and not getattr(p.lstat(), 'st_file_attributes', 0) & 1024,
                    'Reparse point in original run: ' + str(p))
        for name in sorted(files):
            p = Path(directory) / name
            require(p.is_relative_to(root), 'Inventory path escaped root')
            result[str(p)] = file_sha(p)
    return dict(sorted(result.items()))

def assert_stopped():
    from tools import baseline_controller_guard_v3 as frozen
    require(file_sha(frozen.__file__) == GUARD_SOURCE_SHA, 'Original process guard changed')
    launchers = (*frozen.LAUNCHERS, 'run_five_baselines_reviewed_chain_v4', 'run_five_baselines_simplemem_native_v2')
    active = FunctionType(frozen.active_controllers.__code__, {**frozen.active_controllers.__globals__, 'LAUNCHERS': launchers},
                          'active_controllers', frozen.active_controllers.__defaults__)
    guard = FunctionType(frozen.assert_stopped.__code__, {**frozen.assert_stopped.__globals__, 'active_controllers': active},
                         'assert_stopped', frozen.assert_stopped.__defaults__)
    guard()

@contextmanager
def offline():
    from jev_mem import providers
    def forbidden(*args, **kwargs):
        raise AssertionError('Terminal audit forbids network or credential access')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
         patch.object(providers, 'load_credentials', forbidden):
        yield

def normalized_event(event):
    return {k: v for k, v in event.items() if k != 'recorded_utc'}

def identity(event):
    return {k: event.get(k) for k in ('context', 'stage', 'provider', 'model', 'operation', 'request_hash')}

def context_key(context):
    return digest(context)

def audit_api(root):
    """Exact local producer, bounded physical attempts, cache consumers, raw binding."""
    root = Path(root).resolve()
    calls = jsonl(root / 'api_calls.jsonl')
    events = jsonl(root / 'logical_events.jsonl')
    physical, cached, consumers = defaultdict(list), [], defaultdict(list)
    for call in calls:
        if call.get('cache_hit'):
            cached.append(call)
        else:
            physical[call['request_hash']].append(call)
    for event in events:
        require(event.get('source') in ('new_request', 'existing_run_cache') and event.get('error_type') is None,
                'Unknown/failed/imported logical consumer')
        consumers[event['request_hash']].append(event)
    intents = {p.stem: p for p in (root / 'request_intents').glob('*.json')}
    raw_paths = {p.stem: p for p in (root / 'api_cache').glob('*.json')}
    require(set(intents) == set(raw_paths) == set(physical) == set(consumers), 'API producer/raw/consumer key sets differ')
    require(not any((root / 'cache_lineage').glob('*.json')), 'Historical cache lineage in original cold run')
    records, retry_chains = {}, []
    for key in sorted(intents):
        intent = read(intents[key]); raw = read(raw_paths[key]); signature = intent['signature']
        require(type(intent.get('retries')) is int and intent['retries'] == 2, 'Frozen transport retry limit must be exactly two')
        require(digest(signature) == key == intent['request_hash'] and raw.get('request') == signature, 'Intent/raw/full signature mismatch')
        require(signature['provider'] == 'gemini', 'Undeclared provider')
        operation = signature['operation']
        if operation == 'generateContent':
            require(signature['model'] == 'gemini-2.5-flash', 'Wrong generation model')
            cfg = signature['body']['generationConfig']
            require(cfg.get('thinkingConfig') == {'thinkingBudget': 0} and 'temperature' not in cfg, 'Generation protocol differs')
        else:
            require(operation in ('batchEmbedContents', 'countTokens') and signature['model'] == 'gemini-embedding-001', 'Undeclared method endpoint')
        cc = physical[key]
        require([c.get('attempt') for c in cc] == list(range(1, len(cc) + 1)) and len(cc) <= intent['retries'] + 1,
                'Physical attempt order/limit differs')
        require(cc[-1].get('status') == 200 and all(c.get('status') in RETRYABLE for c in cc[:-1]), 'Unknown/unresolved physical result')
        times = [datetime.fromisoformat(c['timestamp_utc']) for c in cc]
        require(times == sorted(times) and datetime.fromisoformat(intent['created_utc']) <= times[0], 'Physical/intent timestamps out of order')
        response = raw['response']
        require(cc[-1].get('usage') == response.get('usageMetadata', response.get('usage', {})) and
                cc[-1].get('resolved_model') == response.get('modelVersion', response.get('model')), 'Physical success usage/model differs from raw')
        expected_identity = {'context': intent['context'], 'stage': intent['stage'], 'request_hash': key,
                             **{k: signature[k] for k in ('provider', 'model', 'operation')}}
        require(all(identity(c) == expected_identity for c in cc), 'Physical retry identity differs')
        require(sum(e['source'] == 'new_request' for e in consumers[key]) == 1, 'Must have exactly one local logical producer')
        producer = next(e for e in consumers[key] if e['source'] == 'new_request')
        require(identity(producer) == expected_identity, 'Logical producer differs from intent')
        require(datetime.fromisoformat(producer['recorded_utc']) >= times[-1], 'Consumer recorded before physical success')
        raw_sha = file_sha(raw_paths[key])
        for e in consumers[key]:
            require(Path(e['cache_path']).resolve() == raw_paths[key].resolve() and e['cache_sha256'] == raw_sha, 'Consumer raw differs')
            require(all(e.get(k) == signature[k] for k in ('provider', 'model', 'operation')), 'Consumer endpoint differs')
        records[key] = {'request_hash': key, 'signature': signature, 'raw': binding(raw_paths[key]), 'intent': binding(intents[key]),
                        'intent_value': intent, 'raw_value': raw, 'physical_calls': cc, 'logical_consumers': consumers[key]}
        if len(cc) > 1:
            retry_chains.append({'request_hash': key, 'identity': expected_identity, 'attempts': cc, 'raw': records[key]['raw']})
    expect_cache = Counter(digest(identity(e)) for e in events if e['source'] == 'existing_run_cache')
    require(expect_cache == Counter(digest(identity(c)) for c in cached), 'Cache physical-log/consumer multiplicity differs')
    for c in cached:
        response = records[c['request_hash']]['raw_value']['response']
        require(c.get('usage') == response.get('usageMetadata', response.get('usage', {})) and
                c.get('resolved_model') == response.get('modelVersion', response.get('model')) and
                c.get('usage_semantics') == 'reused_response_no_new_model_usage', 'Cached usage/model semantics differ')
    pacing = jsonl(root / 'embedding_pacing_events.jsonl')
    actual_admissions = Counter((e['request_hash'], e['physical_attempt']) for e in pacing if e['event'] == 'physical_embedding_admitted')
    expect_admissions = Counter((k, c['attempt']) for k, rr in physical.items() for c in rr if c['operation'] == 'batchEmbedContents')
    require(actual_admissions == expect_admissions, 'Embedding physical admissions differ')
    summary = {'api_root': str(root), 'unique_requests': len(records), 'physical_calls': sum(map(len, physical.values())),
               'logical_calls': len(events), 'cache_consumers': len(cached), 'status_counts': dict(Counter(str(c['status']) for rr in physical.values() for c in rr)),
               'retry_chains': retry_chains, 'intent_raw_records': [{k: v[k] for k in ('request_hash', 'raw', 'intent')} for v in records.values()],
               'logical_events': binding(root / 'logical_events.jsonl'), 'physical_ledger': binding(root / 'api_calls.jsonl')}
    return summary, records, events

def generation_signature(event):
    from jev_mem.gemini25_ablation_provider_v1 import generation_body
    require(event.get('max_output') == 16384 and event.get('temperature_sent') is False and event['stage'] in ALLOWED_STAGES,
            'Native generation protocol changed')
    require(event['request_id'] == digest({'stage': event['stage'], 'prompt': event['prompt'], 'system': event['system']}, True),
            'Native generation request_id differs')
    return {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
            'body': generation_body(event['prompt'], system=event['system'], json_output=False, max_output=16384)}

def embedding_signatures(event):
    require(event['model'] == 'gemini-embedding-001' and event['dimensions'] == 768 and type(event['query']) is bool,
            'Native embedding protocol changed')
    result = []
    for text in event['texts']:
        if len(text.encode('utf8')) > 1800:
            result.append(('embedding_length_check', {'provider': 'gemini', 'model': 'gemini-embedding-001', 'operation': 'countTokens',
                'body': {'contents': [{'parts': [{'text': text}]}]}}))
    for start in range(0, len(event['texts']), 32):
        result.append(('embedding', {'provider': 'gemini', 'model': 'gemini-embedding-001', 'operation': 'batchEmbedContents',
            'body': {'requests': [{'model': 'models/gemini-embedding-001', 'content': {'parts': [{'text': text}]},
                        'taskType': 'RETRIEVAL_QUERY' if event['query'] else 'RETRIEVAL_DOCUMENT', 'outputDimensionality': 768}
                       for text in event['texts'][start:start+32]]}}))
    return result

def audit_native(log, logical, records, context, *, build=False):
    import numpy as np
    from jev_mem.gemini25_ablation_provider_v1 import response_text
    entries = jsonl(log); pos = 0; prefix = []; generation_count = 0; embedding_count = 0
    ii = 0; fts = []
    while ii < len(entries):
        e = entries[ii]
        if e['event'] == 'final_fts_rebuild':
            require(build and ii == len(entries)-1 and e.get('completed') is True and e['backend'] == 'Tantivy' and e['tokenizer_name'] == 'en_stem', 'Unexpected FTS audit event')
            fts.append(e); ii += 1; continue
        require(e['event'] in ('generation_request', 'embedding_request') and ii+1 < len(entries), 'Incomplete/unknown native sequence')
        answer = entries[ii+1]
        signature_stages = [(e['stage'], generation_signature(e))] if e['event'] == 'generation_request' else embedding_signatures(e)
        vectors = []
        for stage, signature in signature_stages:
            require(pos < len(logical), 'Native request lacks logical event')
            event = logical[pos]; key = digest(signature); record = records[key]
            require(event['context'] == context and event['stage'] == stage and event['request_hash'] == key and record['signature'] == signature,
                    'Ordered native/logical/full request signature differs')
            item = {'ordinal': pos, 'event': normalized_event(event), 'event_sha256': digest(normalized_event(event)),
                    'signature': signature, 'raw': record['raw'], 'intent': record['intent'], 'operation': signature['operation'],
                    'stage': stage, 'context': context, 'request_hash': key, 'native_log': binding(log), 'native_request_ordinal': ii,
                    'native_request_sha256': digest(e), 'native_response_sha256': digest(answer), 'mode': 'inherit'}
            item['native_request_payload_sha256'] = digest({k: v for k, v in e.items() if k != 'utc'})
            item['native_response_payload_sha256'] = digest({k: v for k, v in answer.items() if k != 'utc'})
            response = record['raw_value']['response']
            if signature['operation'] == 'generateContent':
                text = response_text(response)
                require(answer['event'] == 'generation_response' and answer['stage'] == stage and answer['request_id'] == e['request_id'] and answer['response'] == text,
                        'Native generation text differs from original raw')
                item['native_request_id'] = e['request_id']; item['response_text_sha256'] = sha_bytes(text.encode('utf8')); generation_count += 1
            elif signature['operation'] == 'countTokens':
                require(type(response.get('totalTokens')) is int and response['totalTokens'] <= 2048, 'Invalid embedding token count')
            else:
                rows = response.get('embeddings', [])
                require(len(rows) == len(signature['body']['requests']), 'Embedding count differs')
                for row in rows:
                    vector = row['values']; norm = math.sqrt(sum(x*x for x in vector))
                    require(len(vector) == 768 and math.isfinite(norm) and norm > 0, 'Invalid raw embedding')
                    vectors.append([x/norm for x in vector])
            prefix.append(item); pos += 1
        if e['event'] == 'embedding_request':
            values = np.asarray(vectors, dtype=np.float64)
            require(values.shape == (len(e['texts']), 768) and np.isfinite(values).all(), 'Native embedding matrix differs')
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            require(not (norms == 0).any(), 'Zero native embedding')
            normalized = values / norms
            require(answer['event'] == 'embedding_response' and answer['query'] == e['query'] and answer['count'] == len(e['texts']) and answer['vectors_sha256'] == digest(normalized.tolist(), True),
                    'Exact normalized embedding digest differs')
            for item in prefix[-len(signature_stages):]:
                item['native_embedding_vectors_sha256'] = answer['vectors_sha256']
            embedding_count += 1
        ii += 2
    require(pos == len(logical), 'Extra logical request absent from native audit')
    require(not build or len(fts) == 1, 'Missing final FTS audit')
    return {'log': binding(log), 'generation_responses': generation_count, 'embedding_responses': embedding_count,
            'logical_calls': pos, 'prefix': prefix, 'final_fts': fts}

def parse_native_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)[:-3].strip()
    return json.loads(text)

def failure_certificate(q, capture, prefix, records):
    from jev_mem.gemini25_ablation_provider_v1 import response_text
    require(prefix and capture['error']['type'] == 'BaselineFatal', 'Unknown failure boundary')
    last = prefix[-1]; record = records[last['request_hash']]; raw = record['raw_value']['response']; text = response_text(raw)
    message = capture['error']['message']; stage = last['stage']
    cert = {'query_id': q['query_id'], 'conversation': q['conversation'], 'stage': stage, 'request_hash': last['request_hash'],
            'raw': last['raw'], 'response_text_sha256': sha_bytes(text.encode()), 'exception': capture['error'],
            'paid_regeneration_permitted': False, 'retry_limit': 0, 'prefix_ordinal': last['ordinal']}
    if stage == 'simplemem_query_analysis':
        # Prove the actual upstream argument against the complete original prompt;
        # do not infer it from a qid or substitute a decomposed semantic subquery.
        messages = analysis_messages(q['question'])
        from jev_mem.gemini25_ablation_provider_v1 import generation_body
        expected_body = generation_body(messages[1]['content'], system=messages[0]['content'], json_output=False, max_output=16384)
        require(record['signature']['body'] == expected_body, 'Native analysis argument/prompt differs from original question')
        cert['analysis_query_verified_against_full_native_prompt'] = True
    try:
        value = parse_native_json(text)
    except json.JSONDecodeError as exc:
        require(stage == 'simplemem_query_analysis' and 'JSONDecodeError' in message, 'Unqualified invalid JSON')
        cert.update(kind='native_analysis_cached_default', analysis_query=q['question'],
                    json_error={'message': exc.msg, 'line': exc.lineno, 'column': exc.colno, 'position': exc.pos})
        return cert
    require(isinstance(value, dict), 'Failure response is not a JSON object')
    if stage == 'simplemem_missing_queries':
        require(value.get('targeted_queries') == [] and type(value.get('targeted_queries')) is list and message == 'SimpleMem retrieve: ValueError: Invalid targeted_queries',
                'Unqualified missing-info schema error')
        cert.update(kind='native_empty_missing_queries', targeted_queries=[])
    elif stage == 'simplemem_query_analysis':
        keywords = value.get('keywords')
        require(isinstance(keywords, list) and all(type(k) is str for k in keywords), 'Invalid keywords require separate review')
        joined = ' '.join(keywords)
        if message == 'SimpleMem retrieve: ValueError: Syntax Error: ' + joined:
            kind = 'native_keyword_syntax_empty'
        else:
            match = re.fullmatch(r"SimpleMem retrieve: ValueError: Field does not exist: '([^']+)'", message)
            require(match is not None and match.group(1)+':' in joined, 'Unqualified keyword error')
            kind = 'native_keyword_field_empty'; cert['missing_field'] = match.group(1)
        cert.update(kind=kind, keywords=keywords, joined_keywords_sha256=sha_bytes(joined.encode()), analysis_query=q['question'])
    else:
        raise ValueError('Unknown failed method stage: ' + stage)
    return cert

def analysis_messages(query):
    path = ROOT / 'upstream/baselines_v1/simplemem/simplemem/core/hybrid_retriever.py'
    tree = ast.parse(shared_read(path).decode('utf8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'HybridRetriever')
    fun = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_analyze_query')
    for arg in fun.args.args:
        arg.annotation = None
    fun.returns = None
    class Captured(BaseException):
        pass
    result = []
    def capture(messages, **kwargs):
        result.extend(messages)
        raise Captured()
    ns = {'config': SimpleNamespace(USE_JSON_FORMAT=False)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[])), str(path), 'exec'), ns)
    try:
        ns['_analyze_query'](SimpleNamespace(llm_client=SimpleNamespace(chat_completion=capture)), query)
    except Captured:
        pass
    require(len(result) == 2, 'Could not reconstruct original query-analysis messages')
    return result

def original_formatter():
    path = ROOT / 'upstream/baselines_v1/simplemem/simplemem/core/answer_generator.py'
    tree = ast.parse(shared_read(path).decode('utf8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'AnswerGenerator')
    fun = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_format_contexts')
    for arg in fun.args.args:
        arg.annotation = None
    fun.returns = None
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[])), str(path), 'exec'), ns)
    return ns['_format_contexts']

def collect():
    global _BOUND_INVENTORY, _BINDING_CACHE, _JSONL_CACHE
    _BOUND_INVENTORY = None; _BINDING_CACHE = {}; _JSONL_CACHE = {}
    assert_stopped()
    print('terminal-audit: inventory before', flush=True)
    before = inventory(OLD)
    _BOUND_INVENTORY = before
    print('terminal-audit: inventory files=' + str(len(before)) + '; API producers/consumers', flush=True)
    from tools import run_five_baselines_gemini25_v1 as base
    from tools import baselines_simplemem_v1 as adapter
    with patch.object(base, 'read', read), patch.object(base, 'sha', file_sha):
        manifest = base.manifest(METHOD)
        require(manifest['specification'] == adapter.describe(), 'Scientific specification differs')
        require(read(manifest['review']['path'])['status'] == 'passed', 'Original implementation review not passed')
        qs = base.questions()
        require(len(qs) == 1540 and manifest['questions'] == qs, 'Fixed cohort mismatch')
        state = read(OLD/'status.json')
        require(state['stage'] == 'needs_review' and state['reason'] == 'build/retrieval failure; original evidence retained', 'Original method not expected terminal')
        require(not (OLD/'results/recovery').exists() and not (OLD/'qa_recovery_plan.json').exists(), 'Original recovery already exists')
        firstpass = read(OLD/'firstpass_outcome.json')
        require(len(firstpass['outcomes']) == 10 and {x['conversation'] for x in firstpass['outcomes']} == set(range(10)), 'Original firstpass outcome set differs')
        firstpass_by_ci = {x['conversation']: x for x in firstpass['outcomes']}
        api_dirs = {p.name for p in (OLD/'work').glob('api_*') if p.is_dir()}
        require(api_dirs == {'api_method', 'api_firstpass'}, 'Unexpected original API namespace')
        method_audit, records, events = audit_api(OLD/'work/api_method')
        qa_audit, qa_records, qa_events = audit_api(OLD/'work/api_firstpass')
        print('terminal-audit: API audit complete; native + original QA replay', flush=True)
        dataset = read(ROOT/'data/locomo10.json')
        by_context = defaultdict(list)
        for e in events:
            by_context[context_key(e['context'])].append(e)
        grouped_used = set(); all_capture_events = []; all_qa_events = []; failed = []; qa_sources = {}; qa_failures = []; cohort = []
        native_audits = {}; banks = {}; known_logs = set(); known_native_results = set(); fmt = original_formatter()
        for ci in range(10):
            out = OLD/f'results/c{ci}'; bank = OLD/f'work/native/c{ci}/simplemem'
            status = read(out/'status.json'); build = read(out/'build.json'); outcomes = read(out/'outcome.json')
            expected = [q['query_id'] for q in qs if q['conversation'] == ci]
            require(status['stage'] == outcomes['status'] == 'terminal' and status['queries'] == len(expected) and build['status'] == 'complete', 'Conversation is not fully terminal')
            require({r['query_id'] for r in outcomes['results']} == set(expected) and len(outcomes['results']) == len(expected), 'Conversation outcomes do not cover cohort')
            require(firstpass_by_ci[ci] == {'conversation': ci, 'status': 'terminal', 'results': outcomes['results']}, 'Root/conv firstpass outcome differs')
            require(not (out/'failure.json').exists(), 'Original build/controller failure')
            sv = read(bank/'state.json'); memory = read(bank/'memory.json')
            source = read(bank/'source.json'); original_conversation = base.conversation_only(dataset[ci])
            require(source['conversation'] == original_conversation and sv['source_hash'] == adapter._hash(original_conversation)
                    and memory['source_hash'] == sv['source_hash'] and sv['protocol_hash'] == adapter._hash(adapter.describe())
                    and memory['protocol'] == adapter.describe() and sv['source_sha'] == adapter.SOURCE_SHA,
                    'Bank source/protocol is not the original dataset conversation')
            with patch.object(adapter, '_load_native', lambda: SimpleNamespace(memory_entry=SimpleNamespace(Dialogue=SimpleNamespace))):
                dialogues, exact_sources = adapter._dialogues(original_conversation)
            require(source['sources'] == memory['sources'] == exact_sources and sv['turn_count'] == len(dialogues), 'Source dates/captions/turns differ')
            require(sv['entry_count'] == len(memory['entries']) == sv['database_integrity']['row_count'] and
                    adapter._entry_digest(memory['entries']) == sv['database_integrity']['entries_sha256'], 'Export entry integrity differs')
            require(Path(sv['work_dir']).resolve() == bank.resolve() and sv['database_integrity'] == memory['database_integrity'], 'State/export integrity differs')
            require(set(sv['export_sha256']) == {'memory.json', 'memory.txt', 'source.json'}, 'Export set differs')
            for name, expected_sha in sv['export_sha256'].items():
                require(file_sha(bank/name) == expected_sha, 'Native export differs')
            for b in build['memory']:
                require(file_sha(b['path']) == b['sha256'], 'Permanent export differs')
            database_inventory = {p: h for p, h in before.items() if Path(p).is_relative_to((bank/'lancedb').resolve())}
            require(database_inventory, 'Empty native DB byte inventory')
            banks[str(ci)] = {'bank_dir': str(bank), 'state': binding(bank/'state.json'), 'state_value': sv,
                'database_integrity': sv['database_integrity'], 'lancedb_inventory': database_inventory,
                'source_conversation_sha256': adapter._hash(original_conversation), 'source': binding(bank/'source.json'),
                'exports': [binding(bank/name) for name in sv['export_sha256']], 'permanent_exports': build['memory'],
                'database_validation_scope': 'exact original DB file bytes only; no DB engine opened during this audit'}
            ctx = {'method': METHOD, 'conversation': ci, 'phase': 'build'}; ck = context_key(ctx); grouped_used.add(ck)
            build_log = bank/'requests.jsonl'; known_logs.add(str(build_log.resolve()))
            nb = audit_native(build_log, by_context[ck], records, ctx, build=True)
            require(nb['final_fts'][0]['entry_count'] == sv['entry_count'], 'Final FTS recorded entry count differs')
            native_audits['build_c'+str(ci)] = {k: v for k, v in nb.items() if k != 'prefix'}
            lookup = {entry['entry_id']: entry for entry in memory['entries']}
            sources = {x['dialogue_id']: x for x in memory['sources']}
            verified_outcomes = []
            for q in [q for q in qs if q['conversation'] == ci]:
                qid = q['query_id']; rp = out/'retrieval'/(qid+'.json'); ret = read(rp); ctx = {'method': METHOD, 'conversation': ci, 'phase': 'retrieval', 'query_id': qid}
                require(read(rp.with_suffix('.intent.json'))['context'] == ctx, 'Retrieval capture intent context differs')
                ck = context_key(ctx); grouped_used.add(ck)
                require(ret['events'] == [normalized_event(e) for e in by_context[ck]], 'Capture/native logical sequence differs')
                all_capture_events.extend(ret['events'])
                if ret['status'] == 'complete':
                    log = Path(ret['value']['request_log'])
                else:
                    candidates = list((bank/'retrieval').glob(digest(q['question'], True)+'-*.requests.jsonl'))
                    # Duplicate question strings may have multiple UUID logs. Match the captured request sequence.
                    expected_first = ret['events'][0]['request_hash']
                    candidates = [p for p in candidates if jsonl(p) and digest(generation_signature(jsonl(p)[0])) == expected_first]
                    if len(candidates) > 1:
                        t0 = read(rp.with_suffix('.intent.json'))['created_utc']; t1 = ret['completed_utc']
                        candidates = [p for p in candidates if t0 <= jsonl(p)[0]['utc'] <= t1]
                    require(len(candidates) == 1, 'Ambiguous/missing failed native log')
                    log = candidates[0]
                require(log.resolve().is_relative_to((bank/'retrieval').resolve()), 'Native log outside bank')
                known_logs.add(str(log.resolve()))
                na = audit_native(log, ret['events'], records, ctx)
                native_audits[qid] = {k: v for k, v in na.items() if k != 'prefix'}
                leaf = OLD/'results/firstpass'/qid
                row = {'query_id': qid, 'conversation': ci, 'retrieval': binding(rp), 'retrieval_status': ret['status']}
                if ret['status'] == 'failed':
                    require(not leaf.exists(), 'Failed retrieval unexpectedly has QA')
                    cert = failure_certificate(q, ret, na['prefix'], records)
                    failed.append({'query_id': qid, 'conversation': ci, 'capture': binding(rp), 'capture_value': ret,
                                   'native_log': binding(log), 'prefix': na['prefix'], 'certificate': cert})
                    row.update(partition='retrieval_failed', reader='not_run', judge='not_run')
                    verified_outcomes.append({'query_id': qid, 'status': 'retrieval_failed'})
                else:
                    require(ret['status'] == 'complete', 'Unknown retrieval status')
                    value = ret['value']; result_path = log.with_name(log.name.replace('.requests.jsonl', '.result.json'))
                    known_native_results.add(str(result_path.resolve()))
                    require(read(result_path) == value and value['count'] == len(value['evidence']), 'Native result/capture differs')
                    entries = []
                    for evidence in value['evidence']:
                        entry = evidence['memory']; eid = evidence['memory_id']
                        require(eid == entry['entry_id'] and entry == lookup[eid], 'Retrieved memory differs from bank export')
                        require(evidence['source_dia_ids'] == [sources[i]['dia_id'] for i in memory['window_provenance'][eid]], 'Retrieved provenance differs')
                        entries.append(SimpleNamespace(**entry))
                    require(fmt(None, entries) == value['context'], 'Original full context formatter differs')
                    reader, judge = base.replay_qa_leaf(OLD, METHOD, q, leaf)
                    outcome = read(leaf/'outcome.json')
                    expected_outcome = {'query_id': qid, 'conversation': ci, 'method': METHOD, 'phase': 'firstpass',
                        'reader': reader['status'], 'judge': judge['status'], 'reader_path': str(leaf/'reader.json'), 'judge_path': str(leaf/'judge.json')}
                    require(outcome == expected_outcome, 'QA outcome metadata differs')
                    verified_outcomes.append(expected_outcome)
                    for stage, v in [('reader', reader), ('judge', judge)]:
                        all_qa_events.extend(v.get('events', []))
                        if v['status'] == 'failed':
                            qa_failures.append({'query_id': qid, 'conversation': ci, 'stage': stage, 'capture': binding(leaf/(stage+'.json')),
                                'error': v['error'], 'events': v['events'], 'successful_reader': binding(leaf/'reader.json') if reader['status'] == 'complete' else None,
                                'qualification': 'original frozen known_recoverable replay returned true; no retry executed'})
                    qa_sources[qid] = {'query_id': qid, 'conversation': ci, 'owner_root': str(OLD), 'leaf': str(leaf),
                        'retrieval': binding(rp), 'reader': binding(leaf/'reader.json'), 'judge': binding(leaf/'judge.json'),
                        'outcome': binding(leaf/'outcome.json'), 'reader_status': reader['status'], 'judge_status': judge['status']}
                    row.update(partition='qa_complete' if reader['status'] == judge['status'] == 'complete' else 'qa_failed', reader=reader['status'], judge=judge['status'])
                cohort.append(row)
            require({p.stem for p in (out/'retrieval').glob('*.json') if not p.name.endswith('.intent.json')} == set(expected), 'Extra/missing retrieval capture')
            require(Counter(digest(x) for x in verified_outcomes) == Counter(digest(x) for x in outcomes['results']), 'Firstpass outcome statuses/paths differ from actual captures')
            require({p.name[:-12] for p in (out/'retrieval').glob('*.intent.json')} == set(expected), 'Extra/missing retrieval intent')
            print('terminal-audit: c' + str(ci) + ' full native/QA complete', flush=True)
        require(grouped_used == set(by_context), 'Unconsumed method context')
        require(Counter(digest(e) for e in all_qa_events) == Counter(digest(normalized_event(e)) for e in qa_events), 'Extra/missing QA logical consumers')
        require({p.name for p in (OLD/'results/firstpass').iterdir()} == set(qa_sources), 'Extra/missing QA leaf')
        require({str(p.resolve()) for p in (OLD/'work/native').rglob('*.requests.jsonl')} | {str(p.resolve()) for p in (OLD/'work/native').glob('c*/simplemem/requests.jsonl')} == known_logs,
                'Unconsumed native request log')
        require({str(p.resolve()) for p in (OLD/'work/native').rglob('*.result.json')} == known_native_results, 'Orphan/missing native result')
        require(len(cohort) == 1540 and len({r['query_id'] for r in cohort}) == 1540, 'Cohort partition not complete')
        print('terminal-audit: inventory after', flush=True)
        after = inventory(OLD)
        require(before == after, 'Original inventory changed during audit')
        assert_stopped()
        proof = {'schema': SCHEMA, 'status': 'passed', 'scope': 'author_terminal_evidence_audit_not_independent_launch_approval',
            'launch_admitted': False, 'paid_regeneration_permissions': [], 'parent_root': str(OLD), 'parent_manifest': binding(OLD/'run_manifest.json'),
            'parent_firstpass_outcome': binding(OLD/'firstpass_outcome.json'),
            'dataset': binding(ROOT/'data/locomo10.json'), 'question_lock': binding(base.LOCK),
            'questions': qs, 'original_inventory': before, 'original_inventory_sha256': digest(before), 'inventory_before_after_equal': True,
            'source_sha256': {str((ROOT/p).resolve()): h for p, h in {**manifest['code_sha256'], **manifest['upstream_sha256']}.items()},
            'audit_tool': binding(__file__), 'cohort': sorted(cohort, key=lambda r: next(i for i,q in enumerate(qs) if q['query_id']==r['query_id'])),
            'partition_counts': dict(Counter(r['partition'] for r in cohort)), 'banks': banks, 'failed_retrievals': failed,
            'failure_kind_counts': dict(Counter(f['certificate']['kind'] for f in failed)), 'original_qa_sources': qa_sources, 'qa_failures': qa_failures,
            'api_audits': {'api_method': method_audit, 'api_firstpass': qa_audit}, 'native_audits': native_audits,
            'method_request_registry': {key: {'request_hash': key, 'signature': value['signature'], 'raw': value['raw'], 'intent': value['intent'],
                'producer_context': value['intent_value']['context'], 'producer_stage': value['intent_value']['stage'],
                'logical_consumers': value['logical_consumers']} for key, value in records.items()},
            'process_guard_source': binding(ROOT/'tools/baseline_controller_guard_v3.py'),
            'process_guard_added_launchers': ['run_five_baselines_reviewed_chain_v4', 'run_five_baselines_simplemem_native_v2'],
            'all_controller_stopped_before_after': True, 'model_calls': 0, 'credential_reads': 0, 'production_writes': 0,
            'database_engine_initialized': False, 'qa_original_function_replay_count': len(qa_sources),
            'qa_replay_scope': 'full original Reader and Judge request/raw/value/F1/label; failed Judge replays original known_recoverable exactly',
            'native_replay_scope': 'original native audit request sequence, all full prompts/signatures, generation raw text, both embedding normalization steps and full result formatter/provenance; not a fresh vector/FTS search'}
        _BOUND_INVENTORY = None; _BINDING_CACHE = None; _JSONL_CACHE = None
        return proof

def validate_proof(path):
    value = read(path)
    require(value.get('schema') == SCHEMA and value.get('status') == 'passed' and value.get('launch_admitted') is False,
            'Unknown terminal proof contract')
    with offline():
        actual = collect()
    require(actual == value, 'Terminal proof differs from complete fresh audit')
    return value

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('create').add_argument('--output', required=True)
    sub.add_parser('validate').add_argument('--proof', required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        p = validate_proof(args.proof)
    else:
        out = Path(args.output).resolve()
        require(out.is_relative_to(ROOT/'research') and not out.exists(), 'Proof output must be fresh research file')
        with offline():
            p = collect()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf8') as f:
            json.dump(p, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': 'passed', 'partitions': p['partition_counts'], 'failure_kinds': p['failure_kind_counts'],
                      'inventory_files': len(p['original_inventory']), 'qa_replayed': p['qa_original_function_replay_count']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
