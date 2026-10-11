"""Pure R08 discovery over copied original-bank vectors; no I/O or dispatch."""
from __future__ import annotations
from copy import deepcopy
import math
from jev_mem.evidence_selection_r03 import (ARMS, digest, build_request, request_hash,
    serialized_signature, parse_scores, choose_baseline, choose_cheap, choose_jev,
    selected_facts, validate_records)

PROTOCOL_SHA256 = '22b54e4bac04abcfadca7a8a1d9ab2aa4b3b5472523ef8c9df866d4d046ef197'
MAX_BYTES = 131072


def normalize_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 768:
        raise ValueError('Exactly768 vector components required')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Finite nonboolean vector components required')
    copied = [float(x) for x in values]
    total = 0.0
    for x in copied:
        total += x * x
    norm = math.sqrt(total)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Positive finite vector norm required')
    return [x / norm for x in copied]


def rank_facets(facts, query_vectors):
    if not isinstance(facts, list) or len({f['id'] for f in facts}) != len(facts):
        raise ValueError('Unique ordered original-bank facts required')
    vectors = [normalize_vector(f['embedding']) for f in facts]
    result = []
    for values in query_vectors:
        query = normalize_vector(values)
        rows = []
        for ordinal, (fact, vector) in enumerate(zip(facts, vectors)):
            score = 0.0
            for q, d in zip(query, vector):
                score += q * d
            if not math.isfinite(score):
                raise ValueError('Nonfinite cosine')
            rows.append({'id': fact['id'], 'cosine': score, 'bank_ordinal': ordinal})
        rows.sort(key=lambda r: (-r['cosine'], r['bank_ordinal']))
        result.append([{**r, 'rank': i + 1} for i, r in enumerate(rows)])
    return result


def round_robin(rankings, native80_ids, prefix_ids, limit=20):
    if type(limit) is not int or not 0 <= limit <= 20:
        raise ValueError('Frozen discovery quota0..20')
    excluded = set(native80_ids) | set(prefix_ids)
    seen, positions, result = set(excluded), [0] * len(rankings), []
    while len(result) < limit:
        progressed = False
        for facet_index, rows in enumerate(rankings, 1):
            pos = positions[facet_index - 1]
            while pos < len(rows) and rows[pos]['id'] in seen:
                pos += 1
            if pos < len(rows):
                row = rows[pos]
                result.append({**deepcopy(row), 'facet_index': facet_index})
                seen.add(row['id']); pos += 1; progressed = True
            positions[facet_index - 1] = pos
            if len(result) == limit:
                break
        if not progressed:
            break
    return result


def build_pool(original_pool_ids, baseline_ids, native_other_ids, native80_ids, discoveries):
    if (len(original_pool_ids) != 80 or len(set(original_pool_ids)) != 80
            or len(baseline_ids) != 20 or len(set(baseline_ids)) != 20
            or len(native_other_ids) != 40 or len(set(native_other_ids)) != 40
            or len(native80_ids) != 80 or len(set(native80_ids)) != 80
            or original_pool_ids[:60] != list(baseline_ids) + list(native_other_ids)
            or not set(native_other_ids).issubset(native80_ids)
            or len(discoveries) > 20):
        raise ValueError('Frozen80/20+40 prefix invariant differs')
    ids = [d['id'] if isinstance(d, dict) else d for d in discoveries]
    if len(set(ids)) != len(ids) or set(ids) & (set(native80_ids) | set(original_pool_ids[:60])):
        raise ValueError('Discovery must be unique and strictly outside native80')
    pool = list(original_pool_ids[:60]) + ids
    filled = 0
    for fid in original_pool_ids:
        if len(pool) == 80:
            break
        if fid not in pool:
            pool.append(fid); filled += 1
    if len(pool) != 80 or len(set(pool)) != 80:
        raise ValueError('Exact shared80 required')
    return {'pool_ids': pool, 'added_ids': sorted(set(pool) - set(original_pool_ids)),
            'displaced_ids': sorted(set(original_pool_ids) - set(pool)),
            'fill_count': filled, 'discovery_count': len(ids),
            'equal_original_pool': pool == list(original_pool_ids)}


def embedding_signature(texts):
    if not isinstance(texts, list) or not 1 <= len(texts) <= 32:
        raise ValueError('Ordered batch1..32 required')
    if any(not isinstance(t, str) or not t or len(t.encode('utf8')) > 512 for t in texts):
        raise ValueError('Imported derived facet byte bound differs')
    return {'provider': 'gemini', 'model': 'gemini-embedding-001',
            'operation': 'batchEmbedContents', 'body': {'requests': [
                {'model': 'models/gemini-embedding-001', 'content': {'parts': [{'text': t}]},
                 'taskType': 'RETRIEVAL_QUERY', 'outputDimensionality': 768} for t in texts]}}


def parse_embeddings(response, count):
    if not isinstance(response, dict) or not isinstance(response.get('embeddings'), list):
        raise ValueError('Embedding list required')
    if len(response['embeddings']) != count:
        raise ValueError('Embedding batch count mismatch')
    return [normalize_vector(item['values']) for item in response['embeddings']]


def pack_batches(items):
    """Stable conversation-local exact-text dedup, preserving consumer mappings."""
    ordered = sorted(items, key=lambda x: (x['conversation'], int(x['query_id'].split('_q')[1])))
    byconv = {}
    for item in ordered:
        status = item['status']
        if status == 'unknown':
            raise ValueError('Unknown imported facet state blocks preparation')
        if status != 'complete':
            if status not in ('failed', 'not_dispatched'):
                raise ValueError('Explicit terminal facet status required')
            continue
        facets = item['facets']
        if not isinstance(facets, list) or not 1 <= len(facets) <= 4:
            raise ValueError('Original facet plus0..3 derived required')
        own = byconv.setdefault(item['conversation'], {})
        for index, text in enumerate(facets[1:], 1):
            own.setdefault(text, []).append({'query_id': item['query_id'], 'facet_index': index})
    batches = []
    for ci in sorted(byconv):
        texts = list(byconv[ci])
        for start in range(0, len(texts), 32):
            chunk = texts[start:start + 32]
            batches.append({'request_id': f'c{ci}_b{start // 32:03d}', 'conversation': ci,
                            'texts': chunk, 'consumers': [deepcopy(byconv[ci][t]) for t in chunk]})
    return batches


def same_score_provenance(current_ids, producer_ids):
    if list(current_ids) != list(producer_ids):
        raise ValueError('HTTP identity alone cannot substitute ordered fact IDs')
    return True
