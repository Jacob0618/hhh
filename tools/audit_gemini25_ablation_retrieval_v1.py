"""Offline fresh-query embedding and native-capture admission for ablation QA.

No API, credential or native store client is constructed. Exact SDK query and
entity texts are replayed through the unchanged embedding request builder.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest as request_hash
from jev_mem.qa_embedding_cache_pack_v1 import validate_raw
from tools.mem0_regroup_plain_qa_sources_v1 import inventory
from tools.prepare_gemini38_hybrid_v1 import validate_native_capture
from tools.run_planned_retrieval_v1 import fact_projection

MODEL = 'gemini-embedding-001'
STAGE = 'mem0_embedding_search'
LENGTH_STAGE = STAGE + '_length_check'
RETRYABLE = {429, 500, 502, 503, 504, 529, 'transport_error'}


def _bytes(reader, path):
    path = Path(path).resolve()
    value = path.read_bytes()
    reader.raw_pointer(path, ref={'path': str(path), 'sha256': hashlib.sha256(value).hexdigest()})
    return value


def query_batches(reader, question):
    """Use the exact installed SDK normalization and ordered entity extraction."""
    import mem0.memory.main as sdk
    import mem0.utils.entity_extraction as entity
    import mem0.utils.spacy_models as spacy_models
    import spacy
    if not spacy.util.is_package('en_core_web_sm'):
        raise ValueError('Pinned local spaCy model required; retrieval audit never downloads')
    for module in (sdk, entity, spacy_models):
        reader.raw_pointer(module.__file__)
    text = sdk._validate_and_trim_search_query(question)
    entities = sdk.extract_entities(text)
    if spacy_models._load_failed_full or spacy_models._nlp_full is None:
        raise ValueError('Local entity extraction unavailable; no silent empty-entity audit')
    seen, texts = set(), []
    for _, entity_text in entities[:8]:
        key = sdk.Memory._normalize_entity_text(entity_text)
        if key and key not in seen:
            seen.add(key)
            texts.append(entity_text)
    return [[text]] + ([texts] if texts else [])


class EmbeddingProof:
    """Unchanged provider embedding builder/parser over only recorded raw JSON."""
    embed = ModelAPI.embed
    count_tokens = ModelAPI.count_tokens

    def __init__(self, reader, api_dir, events, context):
        self.reader, self.api_dir = reader, Path(api_dir).resolve()
        self.events, self.context = events, context
        self.position = 0
        self.requests = []

    def request(self, provider, model, body, *, operation, stage):
        signature = {'provider': provider, 'model': model, 'operation': operation, 'body': body}
        key = request_hash(signature)
        if (provider, model) != ('gemini', MODEL) or (operation, stage) not in (
            ('batchEmbedContents', STAGE), ('countTokens', LENGTH_STAGE)):
            raise ValueError('Retrieval may consume only exact QUERY embedding or length-check requests')
        if self.position >= len(self.events):
            raise ValueError('Missing expected query/entity embedding event')
        event = self.events[self.position]
        self.position += 1
        expected = {'provider': provider, 'model': model, 'operation': operation,
                    'request_hash': key, 'stage': stage, 'context': self.context, 'error_type': None}
        if any(event.get(k) != v for k, v in expected.items()):
            raise ValueError('Retrieval event does not bind the exact SDK query/entity request')
        raw_path = self.api_dir / 'api_cache' / (key + '.json')
        if (event.get('source') not in ('new_request', 'existing_run_cache')
            or Path(event.get('cache_path', '')).resolve() != raw_path):
            raise ValueError('Historical, foreign or unknown retrieval cache source')
        raw_ref = {'path': str(raw_path), 'sha256': event.get('cache_sha256')}
        raw = self.reader.bound(raw_ref, sealed=False)
        if raw.get('request') != signature or raw.get('stage') != stage or not isinstance(raw.get('response'), dict):
            raise ValueError('Retrieval raw request/stage differs')
        response = raw['response']
        if response.get('modelVersion', response.get('model')) not in (None, MODEL, 'models/' + MODEL):
            raise ValueError('Retrieval raw resolved model differs')
        if operation == 'batchEmbedContents':
            texts = [entry['content']['parts'][0]['text'] for entry in body['requests']]
            vectors = response.get('embeddings', [])
            if len(vectors) != len(texts):
                raise ValueError('Raw QUERY embedding batch count differs')
            for item in vectors:
                vector = item.get('values')
                if (not isinstance(vector, list) or len(vector) != 768
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)
                    or not math.isfinite(sum(x * x for x in vector)) or sum(x * x for x in vector) <= 0):
                    raise ValueError('Each raw QUERY vector must have 768 finite nonboolean values and nonzero norm')
            # The original pack helper verifies short exact QUERY768 signatures
            # and replays the same provider vector parser; no old pack is read.
            if all(len(text.encode('utf-8')) <= 1800 for text in texts):
                validate_raw(_bytes(self.reader, raw_path), key)
        elif type(response.get('totalTokens')) is not int or response['totalTokens'] < 0:
            raise ValueError('Embedding token-count response is not a nonnegative integer')
        self.requests.append({'hash': key, 'signature': signature, 'stage': stage,
                              'event': deepcopy(event), 'response': response})
        return deepcopy(response)

    def finish(self):
        if self.position != len(self.events):
            raise ValueError('Unconsumed or duplicate query/entity embedding event')


def _inventory(reader, folder, expected):
    folder = Path(folder).resolve()
    if not folder.is_dir() or inventory(folder) != expected:
        raise ValueError('Closed native inventory changed')
    for rel, hashed in expected.items():
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Native inventory path escapes its declared directory')
        reader.raw_pointer(path, ref={'path': str(path), 'sha256': hashed})


def audit_retrieval(reader, run, manifest, item, bank, queries):
    """Admit a whole conversation's new retrieval, raw producers and consumers."""
    run = Path(run).resolve()
    ci = item['conversation']
    folder, api_dir = run / f'conv_{ci}', run / f'conv_{ci}/retrieval_api'
    if type(ci) is not int or ci not in range(10):
        raise ValueError('Declared conversation required')
    if (Path(item['native_copy']).resolve() != folder / 'native'
        or Path(item['native_source']).resolve() != Path(bank['native_dir']).resolve()
        or reader.bound(item['bank']) != bank):
        raise ValueError('Retrieval native copy/source/bank binding differs')
    _inventory(reader, item['native_source'], item['native_inventory'])
    post = reader.read(folder / 'post_retrieval.json')
    if (post.get('status') != 'complete' or post.get('source_unchanged') is not True
        or post.get('protocol_sha256') != manifest['protocol_sha256']):
        raise ValueError('Closed post-retrieval protocol is incomplete or differs')
    _inventory(reader, item['native_copy'], post['native_inventory'])
    intent = reader.read(folder / 'retrieval_intent.json')
    status = reader.read(folder / 'retrieval_status.json', sealed=False)
    if (intent.get('protocol_sha256') != manifest['protocol_sha256']
        or status.get('status') != 'complete' or status.get('conversation') != ci
        or status.get('completed') != len(queries) or status.get('expected') != len(queries)
        or not status.get('finished_utc')):
        raise ValueError('Whole retrieval conversation has not reached a declared terminal state')
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Unique fixed query population required')
    expected_paths = {folder / 'retrieval' / (q['id'] + '.json') for q in queries}
    if {p.resolve() for p in (folder / 'retrieval').glob('*.json')} != expected_paths:
        raise ValueError('Missing or extra retrieval query artifacts')
    observed = []
    for query in queries:
        qid, question = query['id'], query['question']
        row = reader.read(folder / 'retrieval' / (qid + '.json'))
        if (row.get('conversation'), row.get('query_id'), row.get('question'), row.get('bank'),
            row.get('protocol_sha256'), row.get('status')) != (
            ci, qid, question, item['bank'], manifest['protocol_sha256'], 'complete'):
            raise ValueError('Retrieval query/current bank binding differs')
        search = {'query': question, 'top_k': 20, 'filters': {'user_id': bank['user_id']},
                  'rerank': False, 'threshold': .1, 'explain': False,
                  'reference_date': None, 'show_expired': False}
        capture = row['capture']
        if capture.get('native_search_request') != search:
            raise ValueError('Captured native search request differs from this exact query')
        native = capture['capture']
        validate_native_capture(native, fact_projection(bank), bank['user_id'])
        if (capture.get('selection', {}).get('selected_ids') != native['base20_ids']
            or len(native['base20_ids']) != 20):
            raise ValueError('Native selection does not retain its exact captured twenty facts')
        events = row.get('api_events')
        if not isinstance(events, list):
            raise ValueError('Missing retrieval API events')
        context = {'conversation': ci, 'phase': 'retrieval', 'query_id': qid}
        replay = EmbeddingProof(reader, api_dir, events, context)
        for batch in query_batches(reader, question):
            replay.embed(batch, query=True, stage=STAGE)
        replay.finish()
        observed.extend(replay.requests)
    keys = {r['hash'] for r in observed}
    for name in ('api_cache', 'request_intents'):
        if {p.stem for p in (api_dir / name).glob('*.json')} != keys:
            raise ValueError('Foreign, missing or unknown retrieval raw/intent inventory')
    if any((api_dir / 'cache_lineage').glob('*.json')):
        raise ValueError('Historical retrieval cache lineage is forbidden in this rebuild')
    data = _bytes(reader, api_dir / 'api_calls.jsonl')
    if data and not data.endswith(b'\n'):
        raise ValueError('Incomplete retrieval API ledger')
    ledger = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    by_key = defaultdict(list)
    allowed = {(r['hash'], request_hash(r['event']['context'])) for r in observed}
    for row in ledger:
        if (row.get('request_hash'), request_hash(row.get('context'))) not in allowed:
            raise ValueError('Foreign retrieval ledger request/context pair')
        if (row.get('provider'), row.get('model'), row.get('operation'), row.get('stage')) not in (
            ('gemini', MODEL, 'batchEmbedContents', STAGE), ('gemini', MODEL, 'countTokens', LENGTH_STAGE)):
            raise ValueError('Unexpected retrieval ledger provider/model/operation/stage')
        if ((row.get('cache_hit') is not None and type(row.get('cache_hit')) is not bool)
            or not isinstance(row.get('timestamp_utc'), str) or not row['timestamp_utc']):
            raise ValueError('Unended or malformed retrieval ledger entry')
        by_key[row['request_hash']].append(row)
    for key in keys:
        consumers = [r for r in observed if r['hash'] == key]
        sample = consumers[0]
        if any(r['signature'] != sample['signature'] or r['stage'] != sample['stage'] for r in consumers):
            raise ValueError('Ambiguous same-hash retrieval request')
        intent = reader.read(api_dir / 'request_intents' / (key + '.json'), sealed=False)
        if (intent.get('signature') != sample['signature'] or intent.get('request_hash') != key
            or intent.get('stage') != sample['stage'] or intent.get('retries') != 2
            or (key, request_hash(intent.get('context'))) not in allowed):
            raise ValueError('Retrieval producer intent binding differs')
        logs = by_key[key]
        physical = [r for r in logs if r.get('cache_hit') is not True]
        cached = [r for r in logs if r.get('cache_hit') is True]
        if (not 1 <= len(physical) <= 3 or any(type(r.get('attempt')) is not int for r in physical)
            or [r.get('attempt') for r in physical] != list(range(1, len(physical) + 1))
            or any(r.get('context') != intent['context'] for r in physical)
            or physical[-1].get('status') != 200
            or any(r.get('status') not in RETRYABLE for r in physical[:-1])):
            raise ValueError('Retrieval lacks a bounded ended successful physical producer')
        if any('status' in r or 'attempt' in r or r.get('usage_semantics') != 'reused_response_no_new_model_usage' for r in cached):
            raise ValueError('Malformed retrieval cache consumer ledger')
        produced = [r for r in consumers if r['event']['source'] == 'new_request']
        if len(produced) != 1 or produced[0]['event']['context'] != intent['context']:
            raise ValueError('Physical producer is not the unique admitted new request event')
        expected_cache = Counter(request_hash(r['event']['context']) for r in consumers if r['event']['source'] == 'existing_run_cache')
        if Counter(request_hash(r['context']) for r in cached) != expected_cache:
            raise ValueError('Retrieval cache consumers do not match recorded query events')
        raw_response = sample['response']
        usage = raw_response.get('usageMetadata', raw_response.get('usage', {}))
        resolved = raw_response.get('modelVersion', raw_response.get('model'))
        if any(r.get('usage') != usage or r.get('resolved_model') != resolved for r in [physical[-1], *cached]):
            raise ValueError('Retrieval raw usage/model differs from producer or consumer ledger')
        if any((r.get('provider'), r.get('model'), r.get('operation'), r.get('stage')) != (
            'gemini', MODEL, sample['signature']['operation'], sample['stage']) for r in logs):
            raise ValueError('Retrieval ledger request operation/stage differs from its hash')
    return True
