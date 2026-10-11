"""R05 source-append dispatcher derived from frozen R03; transport/executor reused.

The only allowed model operation is the prebound 3..4-predicate JEV request; 0..2 candidates skip dispatch.
No Writer, Reader, Judge, bank, or gold is accessed by the per-query dispatcher.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.evidence_exchange_r01 import parse_scores, request_hash, serialized_signature
from jev_mem.gemini38_provider_v1 import Gemini38API
from tools.score_evidence_exchange_r01 import now, read, sha, publish, exclusive_intent, status_write
from tools.check_experiment_health import shared_read

SCHEMA = 'source-append-r05-dispatch-v1'
RESULT_SCHEMA = 'source-append-r05-score-result-v1'
JEV_STAGE = 'source_append_score_r05'
MAX_BYTES = 131072
LOCK = threading.Lock()
def expected_keys(body):
    questions = body.get('questions')
    if not isinstance(questions, dict) or not 0 <= len(questions) <= 4:
        raise ValueError('Exactly 0..4 contiguous prepared Noul questions required')
    expected = {f'f{i:02d}' for i in range(len(questions))}
    if set(questions) != expected:
        raise ValueError('Noncontiguous R05 candidate keys')
    if any(not isinstance(v, dict) or v.get('type') != 'noul' for v in questions.values()):
        raise ValueError('Only Noul questions permitted')
    return expected


class GateAPI(Gemini38API):
    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        if ((provider, model, operation, stage) != ('jev', 'jev-1.13.0', 'systemone', JEV_STAGE)
                or body.get('model') != model or not 3 <= len(body.get('questions', {})) <= 4
                or set(body.get('questions', {})) != expected_keys(body)
                or any(v.get('type') != 'noul' for v in body['questions'].values())):
            raise ValueError('Only exact predeclared R05 JEV scoring permitted')
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def classify_failure(api, signature, raw_path):
    """Bind terminal transport to its actual producer, not an unrelated context.

An exact-hash consumer can share the producer's failed request but must record
that lineage. An incomplete intent, absent raw after HTTP200, or partial retry
sequence remains unknown and is not automatic permission to redispatch.
"""
    key = request_hash(signature)
    intent_path = api.run_dir / 'request_intents' / (key + '.json')
    evidence = {'consumer_context': dict(api.context), 'producer_context': None,
                'intent': None, 'terminal_attempts': []}
    if not intent_path.exists():
        return 'unknown', 'no_bound_request_intent', evidence
    intent = read(intent_path)
    if (intent.get('request_hash') != key or intent.get('signature') != signature
            or intent.get('stage') != JEV_STAGE or intent.get('retries') != 2):
        return 'unknown', 'request_intent_binding_differs', evidence
    evidence['intent'] = {'path': str(intent_path.resolve()), 'sha256': sha(intent_path)}
    evidence['producer_context'] = intent['context']
    if raw_path.exists():
        saved = read(raw_path)
        if saved.get('request') == signature and 'response' in saved:
            return 'failed', 'response_validation_failed', evidence
        return 'unknown', 'raw_binding_differs', evidence
    ledger = api.run_dir / 'api_calls.jsonl'
    if not ledger.exists():
        return 'unknown', 'no_complete_terminal_dispatch_evidence', evidence
    data = shared_read(ledger)
    rows = []
    for line in data[:data.rfind(b'\n') + 1].splitlines():
        row = json.loads(line)
        if (row.get('request_hash') == key and not row.get('cache_hit')
                and row.get('context') == intent['context'] and row.get('stage') == JEV_STAGE
                and row.get('provider') == 'jev' and row.get('model') == 'jev-1.13.0'
                and row.get('operation') == 'systemone'):
            rows.append(row)
    evidence['terminal_attempts'] = [{k: r.get(k) for k in ('attempt', 'status', 'timestamp_utc')} for r in rows]
    if not rows or [r.get('attempt') for r in rows] != list(range(1, len(rows) + 1)) or len(rows) > 3:
        return 'unknown', 'incomplete_or_duplicate_attempt_sequence', evidence
    retryable = {429, 500, 502, 503, 504, 529, 'transport_error'}
    if any(r.get('status') not in retryable for r in rows[:-1]):
        return 'unknown', 'inconsistent_retry_sequence', evidence
    tail = rows[-1]
    if tail.get('status') == 'transport_error' and len(rows) == 3:
        return 'failed', 'transport_exhausted_three_attempts', evidence
    status = tail.get('status')
    if type(status) is int and status >= 400 and (status not in retryable or len(rows) == 3):
        return 'failed', 'http_error_terminal', evidence
    return 'unknown', 'no_complete_terminal_dispatch_evidence', evidence


def one(run, request_path, api):
    run = Path(run)
    rec = read(request_path)
    sig, qid = rec['signature'], rec['query_id']
    destination = run / 'responses' / (qid + '.json')
    if destination.exists():
        raise FileExistsError('Previous terminal result retained')
    if (request_hash(sig) != rec['request_hash'] or sig['provider'] != 'jev'
            or sig['model'] != 'jev-1.13.0' or sig['operation'] != 'systemone'
            or set(sig['body']['questions']) != expected_keys(sig['body'])):
        raise ValueError('Frozen R05 request binding differs')
    n = len(sig['body']['questions'])
    measured = len(serialized_signature(sig))
    if (rec.get('logical_noul_count') != n or rec.get('dispatch_eligible') != (n >= 3 and measured <= MAX_BYTES)
            or rec.get('signature_utf8_bytes') != measured
            or (n <= 2 and rec.get('skip_reason') != 'no_ranking_choice')):
        raise ValueError('Prepared dispatch eligibility or source candidate count differs')
    row = {'schema': RESULT_SCHEMA, 'query_id': qid, 'protocol_sha256': rec['protocol_sha256'],
           'request_hash': rec['request_hash'], 'actual_request_hash': rec['request_hash'],
           'request_path': str(Path(request_path).resolve()), 'request_sha256': sha(request_path),
           'started_utc': now(), 'scores': None, 'raw': None,
           'serialized_signature_utf8_bytes': len(serialized_signature(sig)), 'input_token_count': None}
    scoped = api.scoped(conversation=int(qid.split('_')[0][1:]), query_id=qid,
                        method='jev_source_append', phase='jev')
    raw_path = scoped.cache_dir / (rec['request_hash'] + '.json')
    if n <= 2:
        row.update(status='not_dispatched', reason='no_ranking_choice')
    elif row['serialized_signature_utf8_bytes'] > MAX_BYTES:
        row.update(status='not_dispatched', reason='serialized_signature_byte_cap_exceeded')
    else:
        try:
            response = scoped.request(sig['provider'], sig['model'], sig['body'],
                                      operation=sig['operation'], stage=JEV_STAGE)
            saved = read(raw_path)
            if saved.get('request') != sig or saved.get('response') != response:
                raise ValueError('Persisted response differs')
            row['scores'] = parse_scores(response, sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason, evidence = classify_failure(scoped, sig, raw_path)
            row.update(status=state, reason=reason, error_type=type(exc).__name__, failure_evidence=evidence)
    if raw_path.exists():
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    row['finished_utc'] = now()
    publish(destination, row)
    return row


def run_phase(run):
    from tools.prepare_source_append_r05 import validate_prepared
    run = Path(run).resolve()
    manifest, refs = validate_prepared(run)
    files = [Path(r['path']).resolve() for r in refs]
    if len(files) != 1540 or len(files) != len(set(files)):
        raise ValueError('Exactly 1540 unique prepared queries required')
    for ref, path in zip(refs, files):
        if sha(path) != ref['sha256'] or path.parent != (run / 'requests').resolve():
            raise ValueError('Dispatch pointer escaped or changed')
    if set(files) != {p.resolve() for p in (run / 'requests').glob('*.json')}:
        raise ValueError('Unmanifested request present')
    if any((run / 'responses').glob('*.json')):
        raise ValueError('Existing score products require explicit reviewed recovery')
    exclusive_intent(run / 'execution_intent_score_jev.json', {
        'schema': SCHEMA, 'phase': 'jev', 'pid': os.getpid(), 'created_utc': now(),
        'dispatch_code_sha256': sha(__file__), 'manifest_sha256': sha(run / 'run_manifest.json'),
        'request_files': refs, 'transport_retries': 2, 'conversations_max': 8,
        'workers_per_conversation': 4, 'technical_recovery_phases_max': 1,
        'signature_byte_cap': MAX_BYTES, 'input_token_count_limit_claim': None})
    api = GateAPI(run, retries=2)
    status = {'schema': SCHEMA, 'phase': 'jev', 'pid': os.getpid(), 'started_utc': now(),
              'status': 'running', 'expected_requests': 1540, 'terminal_requests': 0,
              'counts': {}, 'conversations': {}}
    status_path = run / 'score_jev_status.json'
    status_write(status_path, status)
    results, local_errors = [], []

    def conv(ci):
        own = [p for p in files if p.stem.startswith(f'c{ci}_')]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one, run, p, api): p for p in own}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    with LOCK:
                        local_errors.append({'query_id': futures[future].stem, 'error_type': type(exc).__name__})
                    continue
                with LOCK:
                    results.append(row)
                    status['terminal_requests'] = len(results)
                    status['counts'] = dict(Counter(x['status'] for x in results))
                    status['conversations'][str(ci)] = {
                        'terminal': sum(x['query_id'].startswith(f'c{ci}_') for x in results),
                        'expected': len(own), 'last_query_id': row['query_id'], 'last_activity_utc': now()}
                    status['updated_utc'] = now()
                    status_write(status_path, status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(conv, i) for i in range(10)]):
            future.result()
    status.update(status='needs_review' if local_errors or len(results) != 1540 else 'terminal',
                  finished_utc=now(), local_errors=local_errors)
    publish(run / 'score_jev_outcome.json', status)
    status_write(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_phase(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
