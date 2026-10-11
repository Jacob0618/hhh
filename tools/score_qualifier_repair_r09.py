"""R09 phase dispatcher and offline terminal audit; reuse frozen transport.

The two model phases are separated by a complete proposal audit and immutable
admission-score manifest. Original R06 bounded dispatch and raw audit reused.
This file never generates QA answers or changes a native memory bank.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.evidence_exchange_r01 import digest, parse_scores, request_hash, serialized_signature
from jev_mem.gemini38_provider_v1 import Gemini38API, generation_body, response_text
from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from tools.score_evidence_exchange_r01 import now, read, sha, publish, exclusive_intent, status_write
from tools.check_experiment_health import shared_read

SPEC_PATH = ROOT / 'research/jev_autoresearch_2026-10-04/R09_qualifier_repair_protocol_final_v1.json'
SPEC_SHA = '59908bea9a269f5c79bb30a40bf30df165362bc5f5e291101f3e63b8731efc4b'
SCHEMA = 'qualifier-repair-r09-dispatch-v1'
RESULT_SCHEMA = 'qualifier-repair-r09-score-result-v1'
AUDIT_SCHEMA = 'qualifier-repair-r09-terminal-audit-v1'
PROPOSER_STAGE = 'qualifier_repair_proposer_r09'
JEV_STAGE = 'qualifier_repair_score_r09'
MAX_BYTES = 131072
RETRYABLE = {429, 500, 502, 503, 504, 529}
LOCK = threading.Lock()
JEV_INFLIGHT = threading.BoundedSemaphore(15)


def spec():
    if sha(SPEC_PATH) != SPEC_SHA:
        raise ValueError('Three-voted R09 protocol changed')
    return read(SPEC_PATH)


def phase_paths(run, phase):
    if phase not in ('proposer', 'jev'):
        raise ValueError('Unknown phase')
    return {
        'requests': run / ('proposer_requests' if phase == 'proposer' else 'requests'),
        'responses': run / ('proposer_responses' if phase == 'proposer' else 'responses'),
        'manifest': run / ('run_manifest.json' if phase == 'proposer' else 'score_manifest.json'),
        'status': run / ('proposer_status.json' if phase == 'proposer' else 'score_jev_status.json'),
        'outcome': run / ('proposer_outcome.json' if phase == 'proposer' else 'score_jev_outcome.json'),
        'intent': run / ('execution_intent_proposer.json' if phase == 'proposer' else 'execution_intent_score_jev.json'),
        'stage': PROPOSER_STAGE if phase == 'proposer' else JEV_STAGE,
    }


def expected_keys(body):
    questions=body.get('questions')
    if not isinstance(questions,dict) or len(questions) not in (2,4):
        raise ValueError('One or two candidates with exactly two Nouls each required')
    expected={f'c{i:02d}_{axis}' for i in range(len(questions)//2) for axis in ('support','preservation')}
    if set(questions)!=expected or any(not isinstance(v,dict) or set(v)!={'type','instructions'}
        or v['type']!='noul' or not isinstance(v['instructions'],str) for v in questions.values()):
        raise ValueError('Wrong admission keys or primitive')
    return expected


class GateAPI(Gemini38API):
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        from jev_mem.qualifier_repair_r09 import (build_proposer_request,build_request,
            parse_proposals,stable_candidates,canonical_json,prompt_template)
        signature={'provider':provider,'model':model,'operation':operation,'body':body}
        policy=spec()
        if len(serialized_signature(signature))>MAX_BYTES:raise ValueError('Actual full signature byte cap exceeded')
        if stage==PROPOSER_STAGE:
            prefix=prompt_template(policy['proposer']['prompt']).split('{input_json}')[0]
            if set(body)!={'contents','generationConfig'} or body.get('generationConfig')!=generation_body('x')['generationConfig']:
                raise ValueError('Frozen proposer body differs')
            contents=body.get('contents')
            if (not isinstance(contents,list) or len(contents)!=1 or set(contents[0])!={'role','parts'}
                or contents[0]['role']!='user' or len(contents[0]['parts'])!=1 or set(contents[0]['parts'][0])!={'text'}):
                raise ValueError('One plain text user part required')
            prompt=contents[0]['parts'][0]['text']
            if not isinstance(prompt,str) or not prompt.startswith(prefix):raise ValueError('Proposer prompt differs')
            state=json.loads(prompt[len(prefix):])
            if not state['current_facts']:raise ValueError('Empty current inventory forbids dispatch')
            if signature!=build_proposer_request(state,policy):raise ValueError('Noncanonical proposer full signature')
            # Explicit authorization of this new stage, without changing the old
            # provider whitelist or impersonating its Writer stage.
            return PacedEmbeddingAPI.request(self,provider,model,body,operation=operation,stage=stage,cache=cache)
        if stage!=JEV_STAGE or (provider,model,operation)!=('jev','jev-1.13.0','systemone'):
            raise ValueError('Wrong admission backend/stage')
        keys=expected_keys(body)
        if set(body)!={'model','state','questions'} or body['model']!=model:raise ValueError('Admission body differs')
        revisions=[]
        for i in range(len(keys)//2):
            instructions=body['questions'][f'c{i:02d}_support']['instructions']
            marker='\nREVISION_JSON:\n'
            if marker not in instructions:raise ValueError('Missing complete revision literal')
            revisions.append(json.loads(instructions.rsplit(marker,1)[1]))
        parsed=parse_proposals(canonical_json({'revisions':revisions}),body['state'],policy)
        candidates=stable_candidates(parsed,'transport',body['state'])
        if signature!=build_request(body['state'],candidates,policy):
            raise ValueError('Noncanonical full original-fact/revision predicates')
        with JEV_INFLIGHT:
            return super().request(provider,model,body,operation=operation,stage=stage,cache=cache)


def _events(run):
    path = run / 'api_calls.jsonl'
    if not path.exists():
        return []
    raw = shared_read(path)
    return [json.loads(x) for x in raw[:raw.rfind(b'\n') + 1].splitlines()]


def classify_failure(api, signature, raw_path, stage, events=None):
    key = request_hash(signature)
    ip = api.run_dir / 'request_intents' / (key + '.json')
    evidence = {'consumer_context': dict(api.context), 'producer_context': None,
                'intent': None, 'terminal_attempts': []}
    if not ip.exists():
        return 'unknown', 'no_bound_request_intent', evidence
    intent = read(ip)
    if (intent.get('request_hash') != key or intent.get('signature') != signature
        or intent.get('stage') != stage or intent.get('retries') != 2):
        return 'unknown', 'request_intent_binding_differs', evidence
    evidence['intent'] = {'path': str(ip.resolve()), 'sha256': sha(ip)}
    evidence['producer_context'] = intent['context']
    if raw_path.exists():
        saved = read(raw_path)
        if saved.get('request') == signature and 'response' in saved:
            return 'failed', 'response_validation_failed', evidence
        return 'unknown', 'raw_binding_differs', evidence
    all_events = _events(api.run_dir) if events is None else events
    rows = [r for r in all_events if r.get('request_hash') == key and not r.get('cache_hit')
            and r.get('context') == intent['context'] and r.get('stage') == stage
            and (r.get('provider'), r.get('model'), r.get('operation')) ==
                (signature['provider'], signature['model'], signature['operation'])]
    evidence['terminal_attempts'] = [{k: r.get(k) for k in ('attempt', 'status', 'timestamp_utc')} for r in rows]
    if (not rows or len(rows) > 3
        or [r.get('attempt') for r in rows] != list(range(1, len(rows) + 1))
        or any(r.get('status') not in RETRYABLE | {'transport_error'} for r in rows[:-1])):
        return 'unknown', 'incomplete_or_inconsistent_attempt_sequence', evidence
    tail = rows[-1].get('status')
    if tail == 'transport_error' and len(rows) == 3:
        return 'failed', 'transport_exhausted_three_attempts', evidence
    if type(tail) is int and tail >= 400 and (tail not in RETRYABLE or len(rows) == 3):
        return 'failed', 'retryable_http_exhausted' if tail in RETRYABLE else 'nonretryable_http_terminal', evidence
    return 'unknown', 'no_complete_terminal_dispatch_evidence', evidence


def validate_record(rec,request_path,phase):
    from jev_mem.qualifier_repair_r09 import build_proposer_request,build_request
    sig=rec['signature']
    if request_hash(sig)!=rec['request_hash'] or rec['request_id']!=request_path.stem or rec['query_id']!=rec['request_id']:
        raise ValueError('Prepared identity changed')
    if rec['conversation']!=int(rec['query_id'].split('_')[0][1:]) or rec['state']['session_id']!=rec['session_id']:
        raise ValueError('Session identity differs')
    size=len(serialized_signature(sig))
    empty=phase=='proposer' and not rec['state']['current_facts']
    reason='zero_current_facts' if empty else 'serialized_signature_byte_cap_exceeded' if size>MAX_BYTES else None
    if (rec['signature_utf8_bytes']!=size or rec['dispatch_eligible']!=(size<=MAX_BYTES and not empty)
        or rec.get('dispatch_skip_reason')!=reason):raise ValueError('Byte/empty-inventory dispatch binding differs')
    exact=build_proposer_request(rec['state'],spec()) if phase=='proposer' else build_request(rec['state'],rec['candidates'],spec())
    if sig!=exact: raise ValueError('Canonical full request differs')
    if phase=='jev': expected_keys(sig['body'])
    return size


def request_context(rec,phase):
    return {'conversation':rec['conversation'],'query_id':rec['query_id'],'session_id':rec['session_id'],
        'method':'qualifier_repair_proposer' if phase=='proposer' else 'qualifier_repair_admission','phase':phase}


def one(run, request_path, api, phase):
    run, request_path = Path(run), Path(request_path)
    paths, rec = phase_paths(run, phase), read(request_path)
    measured = validate_record(rec, request_path, phase)
    sig, qid = rec['signature'], rec['query_id']
    dst = paths['responses'] / (request_path.stem + '.json')
    if dst.exists():
        raise FileExistsError('Existing result is immutable')
    row = {'schema': RESULT_SCHEMA, 'phase': phase, 'request_id': request_path.stem,
           'query_id': qid, 'conversation': rec['conversation'], 'session_id': rec['session_id'],
           'protocol_sha256': rec['protocol_sha256'], 'request_hash': rec['request_hash'],
           'actual_request_hash': rec['request_hash'], 'request_path': str(request_path.resolve()),
           'request_sha256': sha(request_path), 'started_utc': now(),
           'scores': None, 'proposal_parse': None, 'raw': None, 'resolved_model_version': None,
           'serialized_signature_utf8_bytes': measured, 'input_token_count': None}
    context = request_context(rec, phase)
    raw_path = run / 'api_cache' / (rec['request_hash'] + '.json')
    if not rec['dispatch_eligible']:
        row.update(status='not_dispatched',reason=rec['dispatch_skip_reason'])
    else:
        scoped=api.scoped(**context)
        if raw_path.resolve()!=(scoped.cache_dir/(rec['request_hash']+'.json')).resolve():raise ValueError('Current cache namespace differs')
        try:
            response = scoped.request(sig['provider'], sig['model'], sig['body'],
                operation=sig['operation'], stage=paths['stage'])
            saved = read(raw_path)
            if saved.get('request') != sig or saved.get('response') != response:
                raise ValueError('Persisted raw response differs')
            row['resolved_model_version'] = response.get('modelVersion') if phase == 'proposer' else response.get('model')
            if phase == 'proposer':
                from jev_mem.qualifier_repair_r09 import parse_proposals
                row['proposal_parse'] = parse_proposals(response_text(response), rec['state'], spec())
            else:
                row['scores'] = parse_scores(response, sig)
            row.update(status='complete', reason=None)
        except Exception as exc:
            state, reason, evidence = classify_failure(scoped, sig, raw_path, paths['stage'])
            row.update(status=state, reason=reason, error_type=type(exc).__name__, failure_evidence=evidence)
    if raw_path.exists():
        raw_response=read(raw_path).get('response',{})
        row['resolved_model_version']=(raw_response.get('modelVersion') if phase=='proposer' else raw_response.get('model')) if isinstance(raw_response,dict) else None
        row['raw'] = {'path': str(raw_path.resolve()), 'sha256': sha(raw_path)}
    row['finished_utc'] = now()
    publish(dst, row)
    return row


def run_phase(run, phase):
    from tools.prepare_qualifier_repair_r09 import validate_prepared
    run = Path(run).resolve()
    paths = phase_paths(run, phase)
    manifest, refs = validate_prepared(run, stage='proposer' if phase == 'proposer' else 'score')
    files = [Path(r['path']).resolve() for r in refs]
    if (len(files) != len(set(files)) or (phase == 'proposer' and len(files) != 272)
        or (phase == 'jev' and len(files) > 272)):
        raise ValueError('Phase cohort size differs')
    for ref, path in zip(refs, files):
        if sha(path) != ref['sha256'] or path.parent != paths['requests']:
            raise ValueError('Manifest request pointer escaped/changed')
        validate_record(read(path), path, phase)
    if set(files) != {p.resolve() for p in paths['requests'].glob('*.json')}:
        raise ValueError('Unmanifested request present')
    if any(paths['responses'].glob('*.json')):
        raise ValueError('Existing phase requires explicitly reviewed recovery')
    exclusive_intent(paths['intent'], {'schema': SCHEMA, 'phase': phase, 'pid': os.getpid(),
        'created_utc': now(), 'dispatch_code_sha256': sha(__file__),
        'manifest_sha256': sha(paths['manifest']), 'request_files': refs,
        'transport_retries': 2, 'conversations_max': 8, 'workers_per_conversation': 4,
        'jev_inflight_max': 15, 'technical_recovery_phases_max': 1, 'signature_byte_cap': MAX_BYTES})
    api = GateAPI(run,retries=2) if any(read(p)['dispatch_eligible'] for p in files) else None
    status = {'schema': SCHEMA, 'phase': phase, 'pid': os.getpid(), 'started_utc': now(),
        'status': 'running', 'expected_requests': len(files), 'terminal_requests': 0,
        'counts': {}, 'conversations': {}}
    status_write(paths['status'], status)
    results, errors = [], []
    byconv = defaultdict(list)
    for path in files:
        byconv[read(path)['conversation']].append(path)

    def conv(ci):
        own = byconv[ci]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one, run, p, api, phase): p for p in own}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    with LOCK:
                        errors.append({'request_id': futures[future].stem, 'error_type': type(exc).__name__})
                    continue
                with LOCK:
                    results.append(row)
                    status['terminal_requests'] = len(results)
                    status['counts'] = dict(Counter(x['status'] for x in results))
                    prior = status['conversations'].get(str(ci), {})
                    status['conversations'][str(ci)] = {'terminal': prior.get('terminal', 0) + 1,
                        'expected': len(own), 'last_request_id': row['request_id'], 'last_activity_utc': now()}
                    status['updated_utc'] = now()
                    status_write(paths['status'], status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(conv, i) for i in range(10)]):
            future.result()
    status.update(status='needs_review' if errors or len(results) != len(files) else 'terminal',
        finished_utc=now(), local_errors=errors)
    publish(paths['outcome'], status)
    status_write(paths['status'], status)
    return status


def audit_replay(run,phase,refs):
    """Offline complete raw/intent/ledger replay; no client or credentials."""
    run=Path(run).resolve()
    paths=phase_paths(run,phase)
    out = read(paths['outcome'])
    if (out['status'] != 'terminal' or out.get('local_errors')
        or out['expected_requests'] != len(refs) or out['terminal_requests'] != len(refs)):
        raise ValueError('Full phase terminal cohort required')
    rows, failures, unknown, response_refs = [], [], [], []
    events = _events(run)
    indexed = defaultdict(list)
    for event in events:
        indexed[event.get('request_hash')].append(event)
    pins = {}

    def pinned(p):
        p = Path(p).resolve()
        h = sha(p)
        if str(p) in pins and pins[str(p)] != h:
            raise ValueError('Audit input changed')
        pins[str(p)] = h
        return read(p)

    pinned(paths['manifest'])
    if pinned(paths['outcome']) != out:
        raise ValueError('Outcome changed')
    execution = pinned(paths['intent'])
    if (execution.get('phase') != phase or execution.get('request_files') != refs
        or execution.get('manifest_sha256') != sha(paths['manifest'])
        or execution.get('dispatch_code_sha256') != sha(__file__)
        or execution.get('transport_retries') != 2):
        raise ValueError('Execution intent differs')
    ledgerp = run / 'api_calls.jsonl'
    if ledgerp.exists():
        ledger = shared_read(ledgerp)
        if ledger and not ledger.endswith(b'\n'):
            raise ValueError('Incomplete terminal API ledger')
        pins[str(ledgerp)] = hashlib.sha256(ledger).hexdigest()
        if [json.loads(x) for x in ledger.splitlines()] != events:
            raise ValueError('Ledger changed during audit initialization')

    # Establish every legitimate consumer before examining physical producers.
    # A matching request hash alone does not authorize an extra logical call.
    prepared = {}
    contexts, signatures = defaultdict(set), {}
    for ref in refs:
        rp = Path(ref['path']).resolve()
        if sha(rp) != ref['sha256']:
            raise ValueError('Request changed')
        rec = pinned(rp)
        validate_record(rec, rp, phase)
        key = rec['request_hash']
        prepared[str(rp)] = rec
        contexts[key].add(digest(request_context(rec, phase)))
        if key in signatures and signatures[key] != rec['signature']:
            raise ValueError('Request hash has conflicting signatures')
        signatures[key] = rec['signature']
    for ip in sorted((run/'request_intents').glob('*.json')):
        possible=read(ip);key=possible.get('request_hash')
        if possible.get('stage')==paths['stage'] or key in contexts:
            intent=pinned(ip)
            if (key not in contexts or ip.name!=key+'.json' or intent.get('signature')!=signatures[key]
                or intent.get('stage')!=paths['stage'] or intent.get('retries')!=2
                or digest(intent.get('context')) not in contexts[key]):
                raise ValueError('Unmanifested or mismatched current-phase request intent')
    projection = [e for e in events if e.get('stage') == paths['stage']
                  or e.get('request_hash') in contexts]
    for event in projection:
        key = event.get('request_hash')
        if key not in contexts:
            raise ValueError('Unmanifested phase API event')
        sig = signatures[key]
        if (event.get('stage') != paths['stage']
            or digest(event.get('context')) not in contexts[key]
            or (event.get('provider'), event.get('model'), event.get('operation')) !=
               (sig['provider'], sig['model'], sig['operation'])):
            raise ValueError('API event has unauthorized stage/context/backend')
    for key, allowed in contexts.items():
        ip = run / 'request_intents' / (key + '.json')
        own_events = indexed.get(key, [])
        if not ip.exists():
            if own_events:
                raise ValueError('API activity lacks producer intent')
            continue
        intent = pinned(ip)
        if (intent.get('signature') != signatures[key] or intent.get('request_hash') != key
            or intent.get('stage') != paths['stage'] or intent.get('retries') != 2
            or digest(intent.get('context')) not in allowed):
            raise ValueError('Producer intent is not an authorized prepared consumer')
        producer = intent['context']
        caches = Counter()
        for event in own_events:
            if event.get('cache_hit'):
                consumer = digest(event['context'])
                caches[consumer] += 1
                if event['context'] == producer or caches[consumer] > 1:
                    raise ValueError('Duplicate consumer or producer cache replay')
            elif event['context'] != producer:
                raise ValueError('More than one physical producer for exact request')

    def successful_raw(rec, row):
        sig, key = rec['signature'], rec['request_hash']
        rawref = row['raw']; rawpath = Path(rawref['path']).resolve()
        if rawpath != (run / 'api_cache' / (key + '.json')).resolve():
            raise ValueError('Raw response outside current phase namespace')
        if sha(rawpath) != rawref['sha256']:
            raise ValueError('Raw SHA differs')
        if (run / 'cache_lineage' / (key + '.json')).exists():
            raise ValueError('Cross-run selector inheritance is undeclared')
        raw = pinned(rawpath)
        if raw['request'] != sig:
            raise ValueError('Raw signature differs')
        intent = pinned(run / 'request_intents' / (key + '.json'))
        if (intent['signature'] != sig or intent['request_hash'] != key
            or intent['stage'] != paths['stage'] or intent['retries'] != 2):
            raise ValueError('Producer intent differs')
        physical = [e for e in indexed[key] if not e.get('cache_hit')
            and e.get('context') == intent['context'] and e.get('stage') == paths['stage']]
        if (not physical or len(physical) > 3
            or [e.get('attempt') for e in physical] != list(range(1, len(physical) + 1))
            or physical[-1].get('status') != 200
            or any(e.get('status') not in RETRYABLE | {'transport_error'} for e in physical[:-1])
            or any((e.get('provider'), e.get('model'), e.get('operation')) !=
                   (sig['provider'], sig['model'], sig['operation']) for e in physical)):
            raise ValueError('Successful producer lacks bounded completed attempts')
        own = [e for e in indexed[key] if e.get('context') == request_context(rec, phase)
            and e.get('stage') == paths['stage']
            and (e.get('provider'), e.get('model'), e.get('operation')) ==
                (sig['provider'], sig['model'], sig['operation'])]
        if not own or not (own[-1].get('cache_hit') or own[-1].get('status') == 200):
            raise ValueError('No successful exact consumer event')
        raw_response=raw['response']
        resolved=(raw_response.get('modelVersion') if phase=='proposer' else raw_response.get('model')) if isinstance(raw_response,dict) else None
        if row.get('resolved_model_version')!=resolved: raise ValueError('Resolved-model metadata differs')
        return raw['response']

    def parsed_response(response, rec):
        if phase == 'proposer':
            from jev_mem.qualifier_repair_r09 import parse_proposals
            return parse_proposals(response_text(response), rec['state'], spec())
        return parse_scores(response, rec['signature'])

    for ref in refs:
        rp = Path(ref['path']).resolve()
        if sha(rp) != ref['sha256']:
            raise ValueError('Request changed')
        rec = prepared[str(rp)]
        p = paths['responses'] / (rp.stem + '.json')
        row = pinned(p)
        if (row['content_sha256'] != digest({k: v for k, v in row.items() if k != 'content_sha256'})
            or row['schema'] != RESULT_SCHEMA or row['phase'] != phase
            or row['request_id'] != rp.stem or row['query_id'] != rec['query_id']
            or row['conversation'] != rec['conversation'] or row.get('session_id') != rec['session_id'] or row['protocol_sha256'] != rec['protocol_sha256']
            or row['request_hash'] != rec['request_hash'] or row['actual_request_hash'] != rec['request_hash']
            or Path(row['request_path']).resolve() != rp or row['request_sha256'] != ref['sha256']
            or row.get('serialized_signature_utf8_bytes') != rec['signature_utf8_bytes']):
            raise ValueError('Response/request lineage differs')
        sig, key = rec['signature'], rec['request_hash']
        if row['status'] == 'not_dispatched':
            if (rec['dispatch_eligible'] or row.get('raw') or row.get('scores') is not None
                or row.get('proposal_parse') is not None or indexed.get(key)
                or row.get('reason') != rec['dispatch_skip_reason']
                or (run / 'request_intents' / (key + '.json')).exists()
                or (run / 'api_cache' / (key + '.json')).exists()):
                raise ValueError('Undispatched request has model activity')
        elif row['status'] == 'complete':
            parsed = parsed_response(successful_raw(rec, row), rec)
            if phase == 'proposer':
                if row['proposal_parse'] != parsed or row['scores'] is not None:
                    raise ValueError('Proposer parser replay differs')
            elif row['scores'] != parsed or row['proposal_parse'] is not None:
                raise ValueError('JEV parser replay differs')
        elif row['status'] in ('failed', 'unknown'):
            # Classification replay needs no credentialed provider instance.
            from types import SimpleNamespace
            context = row.get('failure_evidence', {}).get('consumer_context', {})
            if context != request_context(rec, phase):
                raise ValueError('Failed consumer context differs')
            api = SimpleNamespace(run_dir=run, context=context)
            rawpath = run / 'api_cache' / (key + '.json')
            state, reason, evidence = classify_failure(api, sig, rawpath, paths['stage'], indexed[key])
            if (state, reason, evidence) != (row['status'], row['reason'], row['failure_evidence']):
                raise ValueError('Failed stage replay differs')
            if evidence.get('intent'):
                pinned(evidence['intent']['path'])
            if row.get('scores') is not None or row.get('proposal_parse') is not None:
                raise ValueError('Failed response cannot carry partial parsed values')
            if reason == 'response_validation_failed':
                response = successful_raw(rec, row)
                try:
                    parsed_response(response, rec)
                except (ValueError, RuntimeError, KeyError, TypeError, AttributeError):
                    pass
                else:
                    raise ValueError('Valid response mislabeled as invalid')
            if state == 'unknown':
                unknown.append(row['request_id'])
            else:
                failures.append({'request_id': row['request_id'], 'query_id': row['query_id'],
                    'reason': reason, 'technical_recovery_eligible': reason in
                    ('transport_exhausted_three_attempts', 'retryable_http_exhausted')})
        else:
            raise ValueError('Undeclared terminal status')
        rows.append(row)
        response_refs.append({'path': str(p), 'sha256': pins[str(p)], 'request_id': rp.stem,
            'query_id': row['query_id'], 'request_ref': {'path': str(rp), 'sha256': ref['sha256']}})
    if {p.stem for p in paths['responses'].glob('*.json')} != {r['request_id'] for r in rows}:
        raise ValueError('Extra phase response artifacts')
    counts = dict(Counter(r['status'] for r in rows))
    if counts != out['counts']:
        raise ValueError('Terminal status counts differ')
    hashes = {r['request_hash'] for r in rows}
    if any(e.get('stage') == paths['stage'] and e.get('request_hash') not in hashes for e in events):
        raise ValueError('Unmanifested phase API event')
    for path, h in pins.items():
        if sha(path) != h:
            raise ValueError('Audit input changed before completion')
    eligible = [x for x in failures if x['technical_recovery_eligible']]
    result = {'schema': AUDIT_SCHEMA, 'phase': phase,
        'status': 'needs_review' if unknown else 'needs_recovery_review' if eligible else 'passed',
        'created_utc': now(), 'run': str(run), 'manifest': {'path': str(paths['manifest']), 'sha256': sha(paths['manifest'])},
        'outcome': {'path': str(paths['outcome']), 'sha256': sha(paths['outcome'])},
        'response_refs': response_refs, 'requests_count': len(refs), 'counts': counts,
        'input_refs': [{'path': p, 'sha256': h} for p, h in sorted(pins.items())
                       if p != str(ledgerp)],
        'ledger_projection': {'path': str(ledgerp), 'stage': paths['stage'],
            'request_hashes': sorted(contexts), 'selected_row_count': len(projection),
            'selected_rows_sha256': digest(projection)},
        'unknown': unknown, 'known_terminal_failures': failures,
        'codepins': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'new_model_calls': 0, 'input_file_count': len(pins)}
    return result


def audit_phase(run,phase,output):
    from tools.prepare_qualifier_repair_r09 import validate_prepared
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh audit path required')
    _,refs=validate_prepared(run,stage='proposer' if phase=='proposer' else 'score')
    result=audit_replay(run,phase,refs);publish(output,result);return result


def recover_phase(run,phase):
    raise RuntimeError('R09 recovery is HARD BLOCKED: eligible failures require a separately reviewed exact-signature adapter; no automatic bypass or redispatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['proposer','jev','audit','recover'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['proposer', 'jev'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'audit':
        if not args.phase or not args.output:
            parser.error('Audit requires --phase and --output')
        result = audit_phase(args.run_dir, args.phase, args.output)
    elif args.action=='recover':
        if not args.phase:parser.error('Recovery requires --phase')
        result=recover_phase(args.run_dir,args.phase)
    else:
        result = run_phase(args.run_dir, args.action)
    print(json.dumps({k: result[k] for k in ('status', 'phase', 'counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
