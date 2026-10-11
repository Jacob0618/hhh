"""Durable phase dispatcher for the fixed module-stack experiment.

Reuses the original provider and R05 terminal-attempt classifier.  This file
does not choose evidence or evaluate answers.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import source_support_stack_v1 as core
from jev_mem.evidence_exchange_r01 import request_hash
from jev_mem.gemini38_provider_v1 import Gemini38API
from tools.score_evidence_exchange_r01 import now, read, sha, publish, exclusive_intent, status_write
from tools.score_source_append_r05 import classify_failure as prior_classify

STAGE = 'module_stack_jev_v1'
SCHEMA = 'module-stack-dispatch-v1'
RESULT_SCHEMA = 'module-stack-score-result-v1'
MAX_BYTES = 131072
LOCK = threading.Lock()


def classify_failure(api, signature, raw_path):
    scope = {**prior_classify.__globals__, 'JEV_STAGE': STAGE}
    fn = FunctionType(prior_classify.__code__, scope, prior_classify.__name__,
        prior_classify.__defaults__, prior_classify.__closure__)
    return fn(api, signature, raw_path)


class GateAPI(Gemini38API):
    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        if stage != STAGE:
            raise ValueError('Only the frozen module-stack JEV stage is permitted')
        core.validate_request({'provider': provider, 'model': model, 'operation': operation, 'body': body})
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)


def context(rec):
    return {'conversation': rec['conversation'], 'query_id': rec['query_id'],
        'method': STAGE, 'phase': rec['phase'], 'request_id': rec['request_id']}


def one(producer, ref, api, protocol_sha):
    rec = read(ref['path'])
    if sha(ref['path']) != ref['sha256'] or rec['protocol_sha256'] != protocol_sha:
        raise ValueError('Prepared request changed')
    sig = rec['signature']
    core.validate_request(sig, allow_oversize=True)
    size = len(core.serialized_signature(sig))
    if size != rec['signature_utf8_bytes'] or request_hash(sig) != rec['request_hash']:
        raise ValueError('Signature binding differs')
    destination = producer / 'responses' / (rec['request_id'] + '.json')
    if destination.exists():
        raise FileExistsError('Terminal results are immutable')
    row = {'schema': RESULT_SCHEMA, 'request_id': rec['request_id'], 'query_id': rec['query_id'],
        'phase': rec['phase'], 'conversation': rec['conversation'], 'protocol_sha256': protocol_sha,
        'request': ref, 'request_hash': rec['request_hash'], 'actual_request_hash': rec['request_hash'],
        'producer_dir': str(producer), 'started_utc': now(), 'raw': None, 'scores': None,
        'serialized_signature_utf8_bytes': size}
    scoped = api.scoped(**context(rec))
    raw_path = scoped.cache_dir / (rec['request_hash'] + '.json')
    if not rec['dispatch_eligible']:
        row.update(status='not_dispatched', reason=rec['skip_reason'])
    else:
        if size > MAX_BYTES:
            raise ValueError('Oversize request was incorrectly marked eligible')
        try:
            response = scoped.request(sig['provider'], sig['model'], sig['body'],
                operation=sig['operation'], stage=STAGE)
            saved = read(raw_path)
            if saved.get('request') != sig or saved.get('response') != response:
                raise ValueError('Saved raw request/response binding differs')
            row['scores'] = core.parse_response(response, sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason, evidence = classify_failure(scoped, sig, raw_path)
            row.update(status=state, reason=reason, error_type=type(exc).__name__, failure_evidence=evidence)
    if raw_path.exists():
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    row['finished_utc'] = now()
    publish(destination, row)
    return row


def verify_result(reader, ref, request_ref, producer, retries):
    """Verify raw scores and physical producer/consumer evidence, not status alone."""
    producer = Path(producer).resolve()
    req, result = reader.bound(request_ref), reader.bound(ref)
    if (result['schema'] != RESULT_SCHEMA or result['request'] != request_ref
        or result['producer_dir'] != str(producer)
        or any(result[k] != req[k] for k in ('request_id','query_id','phase','conversation','protocol_sha256','request_hash'))
        or result['actual_request_hash'] != req['request_hash']
        or Path(ref['path']).resolve() != producer/'responses'/(req['request_id']+'.json')):
        raise ValueError('Score record identity mismatch')
    key, sig = req['request_hash'], req['signature']
    rawref = result.get('raw')
    raw = reader.bound(rawref, sealed=False) if rawref else None
    if rawref and (Path(rawref['path']).resolve() != producer/'api_cache'/(key+'.json') or raw['request'] != sig):
        raise ValueError('Foreign raw response')
    if not hasattr(reader, '_stack_ledger'):
        reader._stack_ledger = {}
    lp = producer/'api_calls.jsonl'
    if str(lp) not in reader._stack_ledger:
        events = {}
        if lp.exists():
            from tools.check_experiment_health import shared_read
            data = shared_read(lp)
            reader.raw_pointer(lp)
            if data and not data.endswith(b'\n'):
                raise ValueError('Unfinished API ledger')
            for line in data.splitlines():
                event = json.loads(line)
                events.setdefault(event.get('request_hash'), []).append(event)
        reader._stack_ledger[str(lp)] = events
    events = reader._stack_ledger[str(lp)].get(key, [])
    ip = producer/'request_intents'/(key+'.json')
    if not req['dispatch_eligible']:
        if (result['status'] != 'not_dispatched' or result['reason'] != req['skip_reason']
            or raw or result['scores'] is not None or ip.exists() or events):
            raise ValueError('Skipped request dispatched or contains synthetic scores')
        return result
    intent = reader.read(ip, sealed=False)
    if intent.get('signature') != sig or intent.get('request_hash') != key or intent.get('stage') != STAGE or intent.get('retries') != retries:
        raise ValueError('Producer intent differs')
    physical = [e for e in events if not e.get('cache_hit') and e.get('context') == intent['context']]
    if (not physical or len(physical) > retries+1
        or [e.get('attempt') for e in physical] != list(range(1,len(physical)+1))
        or any((e.get('stage'),e.get('provider'),e.get('model'),e.get('operation')) != (STAGE,'jev','jev-1.13.0','systemone') for e in physical)):
        raise ValueError('Missing contiguous producer attempts')
    consumer_event=any(e.get('context') == context(req) and e.get('stage') == STAGE for e in events)
    # The frozen provider rejects an exact-hash consumer before logging when its
    # producer exhausted transport without a raw response.  Such a failed
    # consumer is proved below by its bound terminal producer and classifier.
    if not consumer_event and not (result['status']=='failed' and raw is None):
        raise ValueError('Missing consumer event for a successful/raw response')
    retryable = {429,500,502,503,504,529,'transport_error'}
    if any(e['status'] not in retryable for e in physical[:-1]):
        raise ValueError('Nonretryable response was repeated')
    if result['status'] == 'complete':
        if not raw or physical[-1]['status'] != 200 or core.parse_response(raw['response'],sig) != result['scores']:
            raise ValueError('Complete score raw proof differs')
    elif result['status'] == 'failed':
        if result['scores'] is not None:
            raise ValueError('Synthetic failure scores')
        if raw:
            if physical[-1]['status'] != 200:
                raise ValueError('Raw has no successful producer')
            try:
                core.parse_response(raw['response'],sig)
            except (ValueError,TypeError,KeyError):
                pass
            else:
                raise ValueError('Valid score marked failed')
        stub = SimpleNamespace(run_dir=producer, context=context(req))
        state, reason, evidence = classify_failure(stub,sig,producer/'api_cache'/(key+'.json'))
        if state != 'failed' or reason != result['reason'] or evidence != result['failure_evidence']:
            raise ValueError('Failure terminal proof differs')
    else:
        raise ValueError('Unknown result needs review before any dependent phase')
    return result


def run_phase(run_dir, phase, *, recovery=False):
    from tools.prepare_module_stack_v1 import validate_phase, phase_results, Reader
    run = Path(run_dir).resolve()
    manifest, refs = validate_phase(run,phase)
    if recovery:
        first = phase_results(Reader(),run,phase,allow_recovery=False)
        refs = [ref for ref in refs if first[read(ref['path'])['request_id']]['status'] == 'failed']
        if not refs:
            raise ValueError('No technical failures to recover')
    producer = run/'phases'/phase/('recovery_v1' if recovery else 'firstpass')
    producer.mkdir(parents=True,exist_ok=True)
    if any((producer/'responses').glob('*.json')):
        raise FileExistsError('Cannot re-enter an existing scoring producer')
    # Reuse the frozen provider/classifier retry contract in both phases.
    # There is at most one recovery phase; successful results are never retried.
    retries = 2
    exclusive_intent(producer/'execution_intent.json',{'schema':SCHEMA,'pid':os.getpid(),
        'created_utc':now(),'phase':phase,'recovery':recovery,'transport_retries':retries,
        'manifest_sha256':sha(run/'phases'/phase/'manifest.json'),
        'dispatch_code_sha256':sha(__file__),'requests':refs})
    api = GateAPI(producer,retries=retries)
    status = {'schema':SCHEMA,'phase':phase,'pid':os.getpid(),'started_utc':now(),
        'status':'running','expected_requests':len(refs),'terminal_requests':0,
        'counts':{},'conversations':{},'recovery':recovery}
    sp = producer/'status.json'
    status_write(sp,status)
    rows, errors = [], []
    byconv = {ci:[] for ci in range(10)}
    for ref in refs:
        byconv[read(ref['path'])['conversation']].append(ref)
    def conv(ci):
        own = byconv[ci]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one,producer,ref,api,manifest['protocol_sha256']):ref for ref in own}
            for f in as_completed(futures):
                with LOCK:
                    try:
                        row = f.result()
                        rows.append(row)
                        status['conversations'][str(ci)] = {'terminal':sum(r['conversation']==ci for r in rows),
                            'expected':len(own),'last_request_id':row['request_id'],'last_activity_utc':now()}
                    except Exception as exc:
                        errors.append({'request':futures[f],'error_type':type(exc).__name__})
                    status.update(terminal_requests=len(rows),updated_utc=now(),
                        counts=dict(Counter(r['status'] for r in rows)),local_errors=deepcopy(errors))
                    status_write(sp,status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(conv,ci) for ci in range(10)]):
            f.result()
    status.update(status='needs_review' if errors or len(rows)!=len(refs) or any(r['status']=='unknown' for r in rows) else 'terminal',
        finished_utc=now(),local_errors=errors)
    publish(producer/'outcome.json',status)
    status_write(sp,status)
    return status


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['support','closure','rank'],required=True)
    p.add_argument('--recover',action='store_true')
    a=p.parse_args()
    print(json.dumps(run_phase(a.run_dir,a.phase,recovery=a.recover),ensure_ascii=False,indent=2))
