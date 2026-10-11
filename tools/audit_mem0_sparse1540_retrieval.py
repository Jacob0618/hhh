"""Offline admission evidence for frozen native Mem0 retrieval (no APIs/Qdrant).

Replays sealed rank20 records and exact query/entity embedding request signatures.
An error is recovered only by a later success consumed in the SAME query context.
Entity-store searches are local and not in the API ledger; captured stderr is a
separate required gate, with conservative run-wide scope for unattributed errors.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
from tools import run_mem0_sparse1540 as driver
from jev_mem.pilot400 import fingerprint

SCHEMA = 'mem0-sparse1540-retrieval-admission-v1'
MODEL = 'gemini-embedding-001'
STAGE = 'mem0_embedding_search'
WARNING_PATTERNS = (
    r'entity boost computation failed', r'entity boost search failed',
    r'skipping entity boost', r'failed to load spacy',
    r'(?:keyword|bm25).*fail',
)


def read(path):
    return json.loads(shared_read(path).decode('utf-8-sig'))


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest_bytes(shared_read(path))


def signature(texts, operation='batchEmbedContents'):
    if operation == 'countTokens':
        body = {'contents': [{'parts': [{'text': texts[0]}]}]}
    else:
        body = {'requests': [{'model': 'models/' + MODEL,
            'content': {'parts': [{'text': text}]}, 'taskType': 'RETRIEVAL_QUERY',
            'outputDimensionality': 768} for text in texts]}
    return {'provider': 'gemini', 'model': MODEL, 'operation': operation, 'body': body}


def expected_requests(question, entities):
    """Mirror pinned SDK first-eight-then-normalized-dedup and ModelAPI.embed."""
    seen, entity_texts = set(), []
    for _, text in entities[:8]:
        key = ' '.join(text.strip().lower().split())
        if key and key not in seen:
            seen.add(key)
            entity_texts.append(text)
    output = []
    for role, texts in [('query', [question]), ('entity', entity_texts)]:
        if not texts:
            continue
        for text in texts:
            if len(text.encode('utf-8')) > 1800:
                output.append({'role': role + '_length_check', 'stage': STAGE + '_length_check',
                    'signature': signature([text], 'countTokens')})
        output.append({'role': role, 'stage': STAGE, 'signature': signature(texts)})
    return output


def native_expected_requests(question, extract):
    """Mirror Memory.search's strip before NLP/embed, not sealed QA identity."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('native_search_query_empty_or_invalid')
    query_text = question.strip()
    return expected_requests(query_text, extract(query_text))


def validate_cache(cache, expected):
    if cache.get('request') != expected or cache.get('stage') not in (STAGE, STAGE + '_length_check'):
        raise ValueError('cache_request_or_creation_stage_mismatch')
    response = cache['response']
    if expected['operation'] == 'countTokens':
        total = response.get('totalTokens')
        if type(total) is not int or not 0 <= total <= 2048:
            raise ValueError('invalid_or_oversized_token_count')
        return {'response_sha256': fingerprint(response), 'total_tokens': total}
    embeddings = response.get('embeddings')
    if not isinstance(embeddings, list) or len(embeddings) != len(expected['body']['requests']):
        raise ValueError('embedding_batch_count_mismatch')
    normalized = []
    for row in embeddings:
        vector = row.get('values')
        if (not isinstance(vector, list) or len(vector) != 768 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError('invalid_768_embedding')
        norm = math.sqrt(sum(x * x for x in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('invalid_embedding_norm')
        normalized.append([x / norm for x in vector])
    return {'response_sha256': fingerprint(response), 'vectors': len(normalized),
            'dimensions': 768, 'normalized_vectors_sha256': fingerprint(normalized)}


def same_context(row, ci, qid):
    context = row.get('context', {})
    return (context.get('architecture') == driver.POLICY and context.get('phase') == 'retrieve'
            and context.get('method') == 'regroup_rank' and context.get('conversation') == ci
            and context.get('query_id') == qid)


def ledger_rows(path):
    raw = shared_read(path)
    rows, malformed = [], []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
            rows.append({**row, '_line': number})
        except (ValueError, UnicodeError):
            malformed.append(number)
    return rows, {'path': str(path), 'sha256': digest_bytes(raw), 'bytes': len(raw),
                  'malformed_lines': malformed, 'rows': len(rows)}


def audit_consumers(rows, ci, qid, expected, cache_loader, seed_hashes=None):
    """Pure request-consumer audit; cache existence alone never repairs a failure."""
    expected_map, required = {}, Counter()
    for task in expected:
        sig = task['signature']
        key = (task['stage'], driver.request_digest(sig))
        expected_map[key] = sig
        required[key] += 1
    relevant = [r for r in rows if same_context(r, ci, qid)]
    by_key = defaultdict(list)
    errors, output = [], []
    for row in relevant:
        key = (row.get('stage'), row.get('request_hash'))
        by_key[key].append(row)
        if key not in required:
            errors.append('unexpected_request_in_query_context')
    for key, count in required.items():
        stage, request_hash = key
        consumers = by_key[key]
        evidence = {'stage': stage, 'request_hash': request_hash, 'expected_consumptions': count,
                    'ledger_lines': [r.get('_line') for r in consumers]}
        problem = []
        if not consumers:
            problem.append('missing_request_consumer')
        successes = [r for r in consumers if
                     (r.get('cache_hit') is True and r.get('status') in (None, 200)) or
                     (r.get('status') == 200 and r.get('cache_hit') in (None, False))]
        physical_errors = [r for r in consumers if r not in successes]
        if len(successes) != count:
            problem.append('successful_consumption_count_mismatch')
        if consumers and consumers[-1] not in successes:
            problem.append('unrecovered_same_context_request_error')
        physical_count, expected_attempt = 0, 1
        for row in consumers:
            sig = expected_map[key]
            if any(row.get(k) != sig[k] for k in ('provider', 'model', 'operation')):
                problem.append('ledger_request_metadata_mismatch')
            if row.get('cache_hit') is not None and type(row.get('cache_hit')) is not bool:
                problem.append('invalid_cache_hit_flag')
            if row.get('cache_hit') is True:
                if row.get('status') not in (None, 200):
                    problem.append('contradictory_cache_status')
                expected_attempt = 1
            else:
                physical_count += 1
                if type(row.get('attempt')) is not int or not 1 <= row['attempt'] <= 3:
                    problem.append('physical_attempt_outside_uniform_budget')
                if row.get('attempt') != expected_attempt:
                    problem.append('physical_attempt_sequence_invalid')
                expected_attempt = 1 if row.get('status') == 200 else expected_attempt + 1
        if physical_count > 3 * count:
            problem.append('physical_attempt_count_exceeds_uniform_budget')
        try:
            cached, cache_meta = cache_loader(request_hash)
            evidence.update(cache=cache_meta, replay=validate_cache(cached, expected_map[key]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problem.append('raw_cache_invalid_or_missing:' + type(exc).__name__)
        for success in successes:
            if success.get('cache_hit') is True:
                # A valid earlier producer can be any context. Recovery, however,
                # still needs this actual same-context consumer (above).
                earlier = [r for r in rows if r.get('_line', 0) < success.get('_line', 0)
                    and r.get('request_hash') == request_hash and r.get('status') == 200
                    and not r.get('cache_hit') and all(r.get(k) == expected_map[key][k]
                                                      for k in ('provider', 'model', 'operation'))]
                if request_hash not in (seed_hashes or set()) and not earlier:
                    problem.append('cache_consumer_has_no_prior_valid_producer_or_seed')
        evidence.update(physical_success=sum(not r.get('cache_hit') for r in successes),
            cache_hits=sum(r.get('cache_hit') is True for r in successes),
            physical_errors=len(physical_errors),
            recovered_errors=len(physical_errors) if consumers and consumers[-1] in successes else 0,
            admitted=not problem, problems=sorted(set(problem)))
        errors.extend(problem)
        output.append(evidence)
    return {'admitted': not errors, 'problems': sorted(set(errors)), 'requests': output,
            'context_consumers': len(relevant)}


def stderr_audit(paths):
    files, warnings = [], []
    for path in paths:
        raw = shared_read(path)
        item = {'path': str(Path(path).resolve()), 'sha256': digest_bytes(raw), 'bytes': len(raw)}
        files.append(item)
        for number, line in enumerate(raw.decode('utf-8-sig', errors='replace').splitlines(), 1):
            for pattern in WARNING_PATTERNS:
                if re.search(pattern, line, re.I):
                    warnings.append({'path': item['path'], 'line': number, 'pattern': pattern,
                                     'line_sha256': digest_bytes(line.encode())})
                    break
    return {'admitted': bool(files) and not warnings, 'files': files, 'warnings': warnings,
        'coverage_supplied': bool(files),
        'scope': 'All selected conversations conservatively share the captured retrieval stderr scope. '
                 'No per-query attribution exists for SDK local entity-store warnings. Absence of a warning '
                 'is not independent proof of the computed entity boost values.'}


def entity_replayer():
    # Load the installed local package directly: never invoke SDK auto-download.
    import importlib.util
    if importlib.util.find_spec('en_core_web_sm') is None:
        raise ValueError('local_spacy_model_unavailable_no_download_attempted')
    import spacy
    from mem0.utils import entity_extraction
    source = Path(entity_extraction.__file__).resolve()
    if file_sha(source) != file_sha(ROOT / 'upstream/mem0/mem0/utils/entity_extraction.py'):
        raise ValueError('installed_entity_extraction_differs_from_pinned_checkout')
    nlp = spacy.load('en_core_web_sm')
    return lambda text: entity_extraction._extract_entities_from_doc(nlp(text)), {
        'source_path': str(source), 'source_sha256': file_sha(source),
        'spacy_version': spacy.__version__, 'model_version': nlp.meta['version'],
        'normalization': 'first_eight_then_nonempty_whitespace_lower_dedup',
        'scope': 'Local deterministic NLP replay only; no native vector store opened.'}


def audit(conversations, stderr_paths):
    manifest = driver.verify_manifest()
    extract, extractor = entity_replayer()
    stderr = stderr_audit(stderr_paths)
    all_questions = driver.questions()
    reports = []
    for ci in conversations:
        folder = driver.conv_path(ci)
        qs = [q for q in all_questions if q['conversation'] == ci]
        problems = []
        state_path = folder / 'retrieve_status.json'
        state = read(state_path) if state_path.exists() else {'status': 'not_started'}
        if state.get('status') != 'complete' or state.get('done') != len(qs):
            problems.append('retrieval_stage_not_complete')
        try:
            bank, bank_hash = driver.load_bank(ci, 'regroup_rank', manifest)
            base_bank, base_hash = driver.load_bank(ci, 'session_rank', manifest)
        except (OSError, ValueError, KeyError):
            reports.append({'conversation': ci, 'admitted': False, 'problems': ['banks_not_admitted'],
                            'expected_queries': len(qs), 'queries': []})
            continue
        ledger_path = folder / 'native/api_calls.jsonl'
        rows, ledger = ledger_rows(ledger_path)
        if ledger['malformed_lines']:
            problems.append('malformed_api_ledger')
        expected_ids = {q['id'] for q in qs}
        for row in rows:
            ctx = row.get('context', {})
            if ctx.get('phase') == 'retrieve' and (ctx.get('conversation') != ci or
                    ctx.get('query_id') not in expected_ids or ctx.get('method') != 'regroup_rank'
                    or ctx.get('architecture') != driver.POLICY):
                problems.append('unexpected_retrieval_context')
        seeds = set()
        for seed in read(folder / 'query_embedding_seed_lineage.json')['entries']:
            target = driver.OUT / seed['target_relative']
            if file_sha(target) != seed['source_sha256']:
                problems.append('seed_cache_changed')
            else:
                seeds.add(seed['request_hash'])
        memo = {}
        def cache_loader(request_hash):
            if request_hash not in memo:
                path = folder / 'native/api_cache' / (request_hash + '.json')
                raw = shared_read(path)
                memo[request_hash] = (json.loads(raw.decode('utf-8-sig')),
                                     {'path': str(path), 'file_sha256': digest_bytes(raw)})
            return memo[request_hash]
        qreports = []
        for q in qs:
            entry = {'query_id': q['id'], 'problems': []}
            for arm, b, bh in [('session_rank', base_bank, base_hash), ('regroup_rank', bank, bank_hash)]:
                path = folder / 'retrieval' / arm / (q['id'] + '.json')
                try:
                    row = read(path)
                    driver.validate_retrieval(row, q, arm, b, bh)
                    entry[arm] = {'path': str(path), 'file_sha256': file_sha(path),
                        'pool_count': len(row['pool_ids']), 'selected_count': len(row['selected']),
                        'selected_sha256': fingerprint(row['selected'])}
                except (OSError, ValueError, KeyError, TypeError):
                    entry['problems'].append(arm + '_record_missing_or_invalid')
            expected = native_expected_requests(q['question'], extract)
            consumers = audit_consumers(rows, ci, q['id'], expected, cache_loader, seeds)
            entry['consumers'] = consumers
            entry['admitted'] = not entry['problems'] and consumers['admitted']
            qreports.append(entry)
        ledger['unchanged_during_audit'] = file_sha(ledger_path) == ledger['sha256']
        if not ledger['unchanged_during_audit']:
            problems.append('ledger_changed_during_audit')
        reports.append({'conversation': ci, 'admitted': not problems and all(x['admitted'] for x in qreports),
            'problems': sorted(set(problems)), 'expected_queries': len(qs),
            'retrieval_state': state, 'retrieval_state_sha256': file_sha(state_path) if state_path.exists() else None,
            'bank_sha256': bank_hash, 'baseline_bank_sha256': base_hash, 'ledger': ledger, 'queries': qreports})
    stderr['unchanged_during_audit'] = all(file_sha(x['path']) == x['sha256'] for x in stderr['files'])
    if not stderr['unchanged_during_audit']:
        stderr['admitted'] = False
    return {'schema': SCHEMA, 'captured_utc': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(driver.OUT), 'manifest_sha256': file_sha(driver.OUT / 'manifest.json'),
        'audit_code_sha256': file_sha(__file__), 'conversations': reports, 'extractor': extractor,
        'stderr': stderr, 'admitted': stderr['admitted'] and all(x['admitted'] for x in reports),
        'model_calls': 0, 'credentials_loaded': False, 'qdrant_opened': False,
        'scope': 'Both arms sealed candidate order/scores and rank20 replay; new regroup arm query/entity '
                 'embedding consumers only. Baseline prior-run request failures are outside this audit. '
                 'No candidate search is rerun. Cache response replay proves recorded embedding values; '
                 'local entity-store scores/boosts were not separately persisted and are not independently reconstructed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversations', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stderr', type=Path, action='append', help='Captured retrieval stderr, repeatable; otherwise discover run-root retrieve*.stderr.log')
    args = parser.parse_args()
    cis = [int(x) for x in args.conversations.split(',')]
    if not cis or len(set(cis)) != len(cis) or any(x not in range(10) for x in cis):
        parser.error('Conversations must be unique 0..9')
    output = args.output.resolve()
    if output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        parser.error('Use a new report path outside experimental runs')
    paths = args.stderr if args.stderr is not None else sorted(driver.OUT.glob('*retriev*.stderr.log'))
    report = audit(cis, paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'admitted': report['admitted'], 'output': str(output),
        'conversations': [{'conversation': x['conversation'], 'admitted': x['admitted'],
            'valid_queries': sum(q['admitted'] for q in x['queries']), 'expected': x['expected_queries'],
            'problems': x['problems']} for x in report['conversations']],
        'stderr_admitted': report['stderr']['admitted'], 'model_calls': 0}))


if __name__ == '__main__':
    main()
