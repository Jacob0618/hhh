"""R11 dispatcher: only the two prebound G/P JEV contracts can dispatch."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import source_set_r11 as pure
from jev_mem.gemini38_provider_v1 import Gemini38API
from tools.score_evidence_exchange_r01 import now, read, sha, publish, exclusive_intent, status_write
from tools.score_source_append_r05 import classify_failure as _prior_classify

SCHEMA = 'source-set-r11-dispatch-v1'
RESULT_SCHEMA = 'source-set-r11-score-result-v1'
STAGES = {'g': 'source_set_r11_g', 'p': 'source_set_r11_p'}
MAX_BYTES = pure.POLICY['signature_utf8_byte_cap']
LOCK = threading.Lock()


def classify_failure(api, signature, raw_path, arm):
    # Reuse the frozen producer/intent/attempt classifier with an isolated stage
    # namespace; old R05 globals and source bytes are never mutated.
    scope = {**_prior_classify.__globals__, 'JEV_STAGE': STAGES[arm]}
    cloned = FunctionType(_prior_classify.__code__, scope,
        _prior_classify.__name__, _prior_classify.__defaults__, _prior_classify.__closure__)
    return cloned(api, signature, raw_path)


class GateAPI(Gemini38API):
    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        arm = next((a for a, s in STAGES.items() if stage == s), None)
        if arm is None:
            raise ValueError('Only predeclared R11 G/P JEV stages permitted')
        pure.validate_signature(arm, {'provider': provider, 'model': model,
            'operation': operation, 'body': body})
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def one(run, request_path, api):
    run = Path(run)
    rec = read(request_path)
    arm, sig, qid = rec['arm'], rec['signature'], rec['query_id']
    pure.validate_signature(arm, sig)
    destination = run / 'responses' / arm / (qid + '.json')
    if destination.exists():
        raise FileExistsError('Previous terminal result retained')
    measured = len(pure.serialized_signature(sig))
    if (pure.request_hash(sig) != rec['request_hash']
        or rec['signature_utf8_bytes'] != measured
        or rec['dispatch_eligible'] != (measured <= MAX_BYTES)
        or rec.get('skip_reason') != (None if measured <= MAX_BYTES else 'signature_oversize')):
        raise ValueError('Prepared R11 signature or eligibility differs')
    row = {'schema': RESULT_SCHEMA, 'arm': arm, 'query_id': qid,
        'conversation': rec['conversation'], 'protocol_sha256': rec['protocol_sha256'],
        'request_hash': rec['request_hash'], 'actual_request_hash': rec['request_hash'],
        'request_path': str(Path(request_path).resolve()), 'request_sha256': sha(request_path),
        'started_utc': now(), 'scores': None, 'raw': None,
        'serialized_signature_utf8_bytes': measured, 'input_token_count': None}
    scoped = api.scoped(conversation=rec['conversation'], query_id=qid,
        method='source_set_r11_' + arm, phase=arm)
    raw_path = scoped.cache_dir / (rec['request_hash'] + '.json')
    if measured > MAX_BYTES:
        row.update(status='not_dispatched', reason='serialized_signature_byte_cap_exceeded')
    else:
        try:
            response = scoped.request(sig['provider'], sig['model'], sig['body'],
                operation=sig['operation'], stage=STAGES[arm])
            saved = read(raw_path)
            if saved.get('request') != sig or saved.get('response') != response:
                raise ValueError('Persisted actual response binding differs')
            row['scores'] = (pure.parse_g if arm == 'g' else pure.parse_p)(response, sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason, evidence = classify_failure(scoped, sig, raw_path, arm)
            row.update(status=state, reason=reason, error_type=type(exc).__name__,
                failure_evidence=evidence)
    if raw_path.exists():
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    row['finished_utc'] = now()
    publish(destination, row)
    return row


def run_phase(run):
    from tools.prepare_source_set_r11 import validate_prepared
    run = Path(run).resolve()
    manifest, refs = validate_prepared(run)
    files = [Path(r['path']).resolve() for r in refs]
    if len(files) != 3080 or len(set(files)) != 3080:
        raise ValueError('Exactly 1540 G and 1540 P prepared requests required')
    records = []
    for ref, path in zip(refs, files):
        if sha(path) != ref['sha256'] or path.parent.parent != (run / 'requests').resolve():
            raise ValueError('Dispatch pointer escaped or changed')
        rec = read(path)
        if path != run / 'requests' / rec['arm'] / (rec['query_id'] + '.json'):
            raise ValueError('Arm/query request namespace differs')
        if rec['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Manifest/request protocol differs')
        records.append(rec)
    if Counter(r['arm'] for r in records) != {'g': 1540, 'p': 1540}:
        raise ValueError('Two complete arms required')
    identities = {(r['arm'], r['query_id']) for r in records}
    if len(identities) != 3080 or {q for a, q in identities if a == 'g'} != {q for a, q in identities if a == 'p'}:
        raise ValueError('Exact common 1540 identities required')
    if set(files) != {p.resolve() for p in (run / 'requests').glob('*/*.json')}:
        raise ValueError('Unmanifested request present')
    if any((run / 'responses').glob('*/*.json')):
        raise ValueError('Existing products require explicit reviewed recovery')
    exclusive_intent(run / 'execution_intent_score_jev.json', {'schema': SCHEMA,
        'pid': os.getpid(), 'created_utc': now(), 'dispatch_code_sha256': sha(__file__),
        'manifest_sha256': sha(run / 'run_manifest.json'), 'request_files': refs,
        'transport_retries': 2, 'conversations_max': 8, 'workers_per_conversation': 4,
        'technical_recovery_phases_max': 1, 'signature_byte_cap': MAX_BYTES})
    api = GateAPI(run, retries=2)
    status = {'schema': SCHEMA, 'pid': os.getpid(), 'started_utc': now(),
        'status': 'running', 'expected_requests': 3080, 'terminal_requests': 0,
        'counts': {}, 'arm_counts': {}, 'conversations': {}}
    status_path = run / 'score_jev_status.json'
    status_write(status_path, status)
    results, errors = [], []

    def conv(ci):
        own = [path for path, rec in zip(files, records) if rec['conversation'] == ci]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one, run, path, api): path for path in own}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    with LOCK:
                        errors.append({'request_path': str(futures[future]), 'error_type': type(exc).__name__})
                    continue
                with LOCK:
                    results.append(row)
                    status.update(terminal_requests=len(results),
                        counts=dict(Counter(r['status'] for r in results)),
                        arm_counts=dict(Counter(r['arm'] + '/' + r['status'] for r in results)),
                        updated_utc=now())
                    status['conversations'][str(ci)] = {'terminal': sum(r['conversation'] == ci for r in results),
                        'expected': len(own), 'last_query_id': row['query_id'], 'last_arm': row['arm'],
                        'last_activity_utc': now()}
                    status_write(status_path, status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(conv, ci) for ci in range(10)]):
            future.result()
    status.update(status='needs_review' if errors or len(results) != 3080
        or any(r['status'] == 'unknown' for r in results) else 'terminal',
        finished_utc=now(), local_errors=errors)
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
