"""R04: score two immutable actual answers; no Reader/Judge/gold access."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.evidence_exchange_r01 import digest, request_hash, serialized_signature
from jev_mem.gemini38_provider_v1 import Gemini38API
from tools.score_evidence_exchange_r01 import now, read, sha, publish, exclusive_intent, status_write
from tools import score_evidence_selection_r03 as previous

SCHEMA = 'answer-selection-r04-dispatch-v1'
RESULT_SCHEMA = 'answer-selection-r04-score-result-v1'
JEV_STAGE = 'answer_selection_score_r04'
MAX_BYTES = 131072
LOCK = threading.Lock()
# Reuse exact context-bound R03 failure classification, with only stage changed.
_failure_globals = dict(previous.classify_failure.__globals__, JEV_STAGE=JEV_STAGE)
classify_failure = types.FunctionType(previous.classify_failure.__code__, _failure_globals,
                                     'classify_failure')


def expected_keys(body):
    questions = body.get('questions')
    if (not isinstance(questions, dict) or set(questions) != {'answer_choice'}
            or not isinstance(questions['answer_choice'], dict)
            or questions['answer_choice'].get('type') != 'choice'):
        raise ValueError('Only the frozen answer_choice Choice question permitted')
    return {'answer_choice'}


class GateAPI(Gemini38API):
    def request(self, provider, model, body, *, operation='generateContent',
                stage='unspecified', cache=True):
        if ((provider, model, operation, stage) != ('jev', 'jev-1.13.0', 'systemone', JEV_STAGE)
                or body.get('model') != model or set(body.get('questions', {})) != expected_keys(body)):
            raise ValueError('Only the prepared R04 JEV Choice operation permitted')
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def one(run, request_path, api):
    from jev_mem.answer_selection_r04 import parse_choice, choose_jev, cheap_choice
    run = Path(run)
    rec = read(request_path)
    sig, qid = rec['signature'], rec['query_id']
    destination = run / 'responses' / (qid + '.json')
    if destination.exists():
        raise FileExistsError('Previous terminal result retained')
    if (request_hash(sig) != rec['request_hash'] or sig['provider'] != 'jev'
            or sig['model'] != 'jev-1.13.0' or sig['operation'] != 'systemone'
            or set(sig['body']['questions']) != expected_keys(sig['body'])):
        raise ValueError('Frozen R04 request binding differs')
    row = {'schema': RESULT_SCHEMA, 'query_id': qid, 'protocol_sha256': rec['protocol_sha256'],
           'request_hash': rec['request_hash'], 'actual_request_hash': None,
           'request_path': str(Path(request_path).resolve()), 'request_sha256': sha(request_path),
           'started_utc': now(), 'choice_result': None, 'raw': None,
           'serialized_signature_utf8_bytes': len(serialized_signature(sig)),
           'input_token_count': None, 'presentation': rec['presentation']}
    scoped = api.scoped(conversation=int(qid.split('_')[0][1:]), query_id=qid,
                        method='jev_answer_choice', phase='jev')
    raw_path = scoped.cache_dir / (rec['request_hash'] + '.json')
    if rec['identical']:
        row.update(status='identical_answer_no_dispatch', reason='exact_identical_answers')
    elif row['serialized_signature_utf8_bytes'] > MAX_BYTES:
        row.update(status='not_dispatched', reason='serialized_signature_byte_cap_exceeded')
    else:
        try:
            row['actual_request_hash'] = rec['request_hash']
            response = scoped.request(sig['provider'], sig['model'], sig['body'],
                                      operation=sig['operation'], stage=JEV_STAGE)
            saved = read(raw_path)
            if saved.get('request') != sig or saved.get('response') != response:
                raise ValueError('Persisted response differs')
            row['choice_result'] = parse_choice(response, sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason, evidence = classify_failure(scoped, sig, raw_path)
            row.update(status=state, reason=reason, error_type=type(exc).__name__,
                       failure_evidence=evidence)
    if raw_path.exists():
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    choice = choose_jev(rec, row['choice_result'], status=row['status'])
    route = choice['route']
    candidate = rec['candidates'][route]
    row.update(selected_route=route, selection_reason=choice['reason'],
               cheap_route=cheap_choice(rec), selected_original_reader=candidate['reader'],
               selected_exact_answer=candidate['answer'],
               union_sha256=digest(rec['state']['memories']), finished_utc=now())
    publish(destination, row)
    return row


def run_phase(run):
    from tools.prepare_answer_selection_r04 import validate_prepared
    run = Path(run).resolve()
    manifest, records = validate_prepared(run)
    files = [run / 'requests' / (r['query_id'] + '.json') for r in records]
    if len(files) != 1540 or len(files) != len(set(files)):
        raise ValueError('Exactly 1540 unique prepared queries required')
    if set(files) != set((run / 'requests').glob('*.json')):
        raise ValueError('Unmanifested request present')
    for rec, path in zip(records, files):
        if read(path) != rec:
            raise ValueError('Prepared artifact changed after validation')
    if any((run / 'responses').glob('*.json')):
        raise ValueError('Existing results require explicit reviewed recovery')
    exclusive_intent(run / 'execution_intent_score_jev.json', {
        'schema': SCHEMA, 'phase': 'jev', 'pid': os.getpid(), 'created_utc': now(),
        'dispatch_code_sha256': sha(__file__), 'manifest_sha256': sha(run / 'run_manifest.json'),
        'request_files': [{'path': str(p), 'sha256': sha(p)} for p in files],
        'transport_retries': 2, 'conversations_max': 8, 'workers_per_conversation': 4,
        'technical_recovery_phases_max': 1, 'signature_byte_cap': MAX_BYTES,
        'input_token_count_limit_claim': None})
    api = GateAPI(run, retries=2)
    status = {'schema': SCHEMA, 'phase': 'jev', 'pid': os.getpid(), 'started_utc': now(),
              'status': 'running', 'expected_requests': 1540,
              'dispatch_eligible': sum(r['dispatch_eligible'] for r in records),
              'terminal_requests': 0, 'counts': {}, 'conversations': {}}
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
                        local_errors.append({'query_id': futures[future].stem,
                                             'error_type': type(exc).__name__})
                    continue
                with LOCK:
                    results.append(row)
                    status['terminal_requests'] = len(results)
                    status['counts'] = dict(Counter(x['status'] for x in results))
                    status['conversations'][str(ci)] = {
                        'terminal': sum(x['query_id'].startswith(f'c{ci}_') for x in results),
                        'expected': len(own), 'last_query_id': row['query_id'],
                        'last_activity_utc': now()}
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
