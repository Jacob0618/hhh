"""R01 dispatch only: exact frozen JEV requests and fixed80 LLM gate diagnostic.

No selector, Reader, Judge, metric or original source is modified. Commands refuse
re-entry; an unfinished intent is evidence requiring review, not permission to resend.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.evidence_exchange_r01 import digest, parse_scores, request_hash
from jev_mem.gemini38_provider_v1 import Gemini38API, generation_body, response_text
from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from jev_mem.sdk20_cached_api import _publish_bytes
from tools.check_experiment_health import shared_read

SCHEMA = 'evidence-exchange-r01-dispatch-v1'
JEV_STAGE = 'evidence_exchange_score_r01'
PILOT_STAGE = 'evidence_exchange_llm_pilot_r01'
MAX_BYTES = 131072
LOCK = threading.Lock()
PILOT_INSTRUCTION = '''Evaluate the given memory-relative evidence exchanges. The shared state and
the exact predicates are below. Do not answer the user query. For each requested
predicate return exactly one line KEY=yes, KEY=no, or KEY=unknown. Use unknown
when the supplied facts do not permit a decision. Include all requested keys
once, in the given order. No JSON, explanations, probabilities or extra lines.
The predicates G, L and X mean added support, lost necessary support and
unexplained contradiction respectively; apply each question's full instructions.
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(shared_read(Path(path)).decode('utf-8-sig'))


def sha(path):
    return hashlib.sha256(shared_read(Path(path))).hexdigest()


def publish(path, value):
    body = dict(value)
    body['content_sha256'] = digest(body)
    _publish_bytes(Path(path), (json.dumps(body, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n').encode('utf-8'))
    return body


def exclusive_intent(path, value):
    body = dict(value)
    body['content_sha256'] = digest(body)
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return body


def status_write(path, value):
    # Only this controller replaces this status; readers use FILE_SHARE_DELETE.
    import time
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def pilot_body(signature):
    payload = {k: signature['body'][k] for k in ('state', 'questions')}
    return generation_body(PILOT_INSTRUCTION + '\n' + json.dumps(payload, ensure_ascii=False), max_output=8192)


def parse_pilot(text, signature):
    expected = list(signature['body']['questions'])
    lines = text.strip().splitlines()
    parsed = {}
    if len(lines) != len(expected):
        raise ValueError('Pilot predicate count differs')
    for key, line in zip(expected, lines):
        match = re.fullmatch(re.escape(key) + r'=(yes|no|unknown)', line.strip())
        if not match:
            raise ValueError('Pilot requires ordered exact yes/no/unknown predicate lines')
        parsed[key] = match.group(1)
    return parsed


class GateAPI(Gemini38API):
    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        if stage == PILOT_STAGE:
            if (provider != 'gemini' or model != 'gemini-3.8-flash' or operation != 'generateContent'
                    or body.get('generationConfig') != generation_body('x')['generationConfig']
                    or set(body) != {'contents', 'generationConfig'}
                    or not body['contents'][0]['parts'][0]['text'].startswith(PILOT_INSTRUCTION)):
                raise ValueError('Pilot configuration differs')
            return PacedEmbeddingAPI.request(self, provider, model, body,
                operation=operation, stage=stage, cache=cache)
        if stage != JEV_STAGE or provider != 'jev' or model != 'jev-1.13.0' or operation != 'systemone':
            raise ValueError('Only declared R01 scoring stages allowed')
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def classify_failure(api, key, raw_path):
    """Only complete local evidence classifies known failures; default unknown."""
    if raw_path.exists():
        return 'failed', 'response_validation_failed'
    ledger = api.run_dir / 'api_calls.jsonl'
    if ledger.exists():
        complete = shared_read(ledger).splitlines()
        rows = []
        for line in complete:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('request_hash') == key and not row.get('cache_hit'):
                rows.append(row)
        if rows:
            tail = rows[-1]
            if tail.get('status') == 'transport_error' and tail.get('attempt') == 3:
                return 'failed', 'transport_exhausted_three_attempts'
            if isinstance(tail.get('status'), int) and tail['status'] >= 400:
                return 'failed', 'http_error_terminal'
    return 'unknown', 'no_complete_terminal_dispatch_evidence'


def one(run, request_path, api, phase):
    rec = read(request_path)
    sig, qid = rec['signature'], rec['query_id']
    if request_hash(sig) != rec['request_hash']:
        raise ValueError('Frozen request hash differs')
    if sig['provider'] != 'jev' or sig['model'] != 'jev-1.13.0' or sig['operation'] != 'systemone':
        raise ValueError('Unexpected frozen JEV signature')
    encoded = json.dumps(sig, ensure_ascii=False, sort_keys=True).encode('utf-8')
    row = {'schema': 'evidence-exchange-r01-score-result-v1' if phase == 'jev' else 'evidence-exchange-r01-pilot-result-v1',
        'query_id': qid, 'protocol_sha256': rec['protocol_sha256'], 'request_hash': rec['request_hash'],
        'request_path': str(Path(request_path).resolve()), 'request_sha256': sha(request_path),
        'started_utc': now(), 'scores': None, 'raw': None,
        'serialized_signature_utf8_bytes': len(encoded), 'input_token_count': None}
    scoped = api.scoped(conversation=int(qid.split('_')[0][1:]), query_id=qid,
        method='jev_exchange' if phase == 'jev' else 'same_state_llm_diagnostic', phase=phase)
    actual = sig if phase == 'jev' else {'provider': 'gemini', 'model': 'gemini-3.8-flash',
        'operation': 'generateContent', 'body': pilot_body(sig)}
    key = request_hash(actual)
    raw_path = scoped.cache_dir / (key + '.json')
    row['actual_request_hash'] = key
    row['actual_signature_utf8_bytes'] = len(json.dumps(actual, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    if row['actual_signature_utf8_bytes'] > MAX_BYTES:
        row.update(status='not_dispatched', reason='serialized_signature_byte_cap_exceeded')
    else:
        try:
            response = scoped.request(actual['provider'], actual['model'], actual['body'],
                operation=actual['operation'], stage=JEV_STAGE if phase == 'jev' else PILOT_STAGE)
            saved = read(raw_path)
            if saved.get('request') != actual or saved.get('response') != response:
                raise ValueError('Persisted raw request or response differs')
            if phase == 'jev':
                row['scores'] = parse_scores(response, sig)
            else:
                row['labels'] = parse_pilot(response_text(response), sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason = classify_failure(scoped, key, raw_path)
            row.update(status=state, reason=reason, error_type=type(exc).__name__)
    if raw_path.exists():
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    row['finished_utc'] = now()
    destination = run / ('responses' if phase == 'jev' else 'pilot_responses') / (qid + '.json')
    publish(destination, row)
    return row


def run_phase(run, phase):
    from tools.prepare_evidence_exchange_r01 import validate_prepared
    run = Path(run).resolve()
    # This validates source bindings and frozen code before any credentials/API.
    manifest, request_refs = validate_prepared(run)
    files = [Path(ref['path']).resolve() for ref in request_refs]
    if len(files) != len(set(files)):
        raise ValueError('Duplicate request source pointers')
    for ref, path in zip(request_refs, files):
        if sha(path) != ref['sha256'] or path.parent != (run / 'requests').resolve():
            raise ValueError('Dispatch pointer escaped or changed after validation')
    if set(files) != {p.resolve() for p in (run / 'requests').glob('*.json')}:
        raise ValueError('Unmanifested request file present; no dispatch')
    if not files:
        raise ValueError('No prepared requests; cannot call models')
    if phase == 'pilot':
        pilot = read(run / 'pilot_manifest.json')
        ids = set(pilot['query_ids'])
        files = [p for p in files if p.stem in ids]
        # Some sampled qids may have no positive challenger; report them, no fabricated calls.
    manifest_path = run / 'run_manifest.json'
    intent = {'schema': SCHEMA, 'phase': phase, 'pid': os.getpid(), 'created_utc': now(),
        'dispatch_code_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path),
        'request_files': [{'path': str(p), 'sha256': sha(p)} for p in files],
        'transport_retries': 2, 'attempts_per_logical_phase_max': 3, 'technical_recovery_phases_max': 1,
        'total_attempts_across_both_phases_max': 6, 'unreviewed_restart_max': 0, 'reviewed_crash_restart_max': 1,
        'conversations_max': 8, 'workers_per_conversation': 4, 'signature_byte_cap': MAX_BYTES,
        'input_token_count_limit_claim': None}
    exclusive_intent(run / f'execution_intent_score_{phase}.json', intent)
    # No successful/failed score output is silently replaced on restart.
    api = GateAPI(run if phase == 'jev' else run / 'pilot_api', retries=2)
    status = {'schema': SCHEMA, 'phase': phase, 'pid': os.getpid(), 'started_utc': now(),
        'status': 'running', 'expected_requests': len(files), 'terminal_requests': 0,
        'counts': {}, 'conversations': {}}
    status_path = run / f'score_{phase}_status.json'
    status_write(status_path, status)
    results, local_errors = [], []

    def conv(ci):
        own = [p for p in files if p.stem.startswith(f'c{ci}_')]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one, run, path, api, phase): path for path in own}
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
                    status['conversations'][str(ci)] = {'terminal': sum(x['query_id'].startswith(f'c{ci}_') for x in results),
                        'expected': len(own), 'last_query_id': row['query_id'], 'last_activity_utc': now()}
                    status['updated_utc'] = now()
                    status_write(status_path, status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(conv, ci) for ci in range(10)]
        for future in as_completed(futures):
            future.result()
    status.update(status='needs_review' if local_errors or len(results) != len(files) else 'terminal',
        finished_utc=now(), local_errors=local_errors)
    publish(run / f'score_{phase}_outcome.json', status)
    status_write(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['jev', 'pilot'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_phase(args.run_dir, args.phase), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
