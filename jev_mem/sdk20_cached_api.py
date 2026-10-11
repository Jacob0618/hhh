"""Cache/source bookkeeping around unchanged ModelAPI request construction.

No generation, embedding, JEV prompt or response parsing is reimplemented here.
The inherited generate/score_many/embed methods are exactly the legacy methods.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone

from jev_mem.providers import ModelAPI
from jev_mem.evaluation import READER_SYSTEM, JUDGE_SYSTEM
from tools.run_pilot400_experiments_v2 import read

_LOCK = threading.Lock()
_HASH_LOCKS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _publish_bytes(path, raw):
    """Flush new bytes before atomic publication; never replace an existing cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError('Cache publication destination already exists: ' + str(path))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_record(raw, signature):
    saved = json.loads(raw.decode('utf-8-sig'))
    if saved.get('request') != signature or 'response' not in saved:
        raise ValueError('Source cache request/hash mismatch')
    # Response content is deliberately not validated here. The legacy consumer
    # must replay its failure too; a later root is never tried for a better answer.
    return saved


def _bound_lineage(path, target, key):
    lineage = read(path)
    source = Path(lineage['source_path'])
    expected = lineage.get('source_sha256')
    if (lineage.get('request_hash') != key or Path(lineage['target_path']).resolve() != target.resolve()
            or source.name != key + '.json' or not isinstance(expected, str) or len(expected) != 64
            or lineage.get('target_sha256') != expected):
        raise ValueError('Cache lineage binding differs')
    return lineage


def _bound_intent(path, signature, key):
    intent = read(path)
    if intent.get('request_hash') != key or intent.get('signature') != signature:
        raise ValueError('Local request intent binding differs')


def check_legacy_request(model, body, stage):
    if stage not in ('qa_reader', 'strict_judge'):
        return
    reader = stage == 'qa_reader'
    expected_model = 'gemini-2.5-flash' if reader else 'gemini-3.5-flash-lite'
    config = {'temperature': 0, 'maxOutputTokens': 2048, 'responseMimeType': 'application/json'}
    if reader:
        config['thinkingConfig'] = {'thinkingBudget': 0}
    expected_system = READER_SYSTEM if reader else JUDGE_SYSTEM
    if model != expected_model or body.get('generationConfig') != config:
        raise ValueError('QA request differs from the fixed legacy SDK20 generation configuration')
    if body.get('systemInstruction') != {'parts': [{'text': expected_system}]}:
        raise ValueError('QA system prompt differs from the legacy baseline')
    if set(body) != {'contents', 'generationConfig', 'systemInstruction'}:
        raise ValueError('QA request contains undeclared fields')


class CachedLegacyAPI(ModelAPI):
    """Reuse exact raw sources; freeze unknown logical attempts; no new schema."""
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        super().__init__(run_dir, **kwargs)
        self.cache_roots = tuple(Path(p).resolve() for p in cache_roots)
        self.events = []

    def scoped(self, **context):
        return type(self)(self.run_dir, keys=self.keys, retries=self.retries, timeout=self.timeout,
                          context={**self.context, **context}, cache_roots=self.cache_roots)

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        if not cache:
            raise ValueError('This fixed run requires auditable request caching')
        check_legacy_request(model, body, stage)
        signature = {'provider': provider, 'model': model, 'operation': operation, 'body': body}
        key = digest(signature)
        target = self.cache_dir / (key + '.json')
        intent = self.run_dir / 'request_intents' / (key + '.json')
        lineage_path = self.run_dir / 'cache_lineage' / (key + '.json')
        with _LOCK:
            lock = _HASH_LOCKS.setdefault((str(self.run_dir.resolve()), key), threading.RLock())
        with lock:
            source = 'existing_run_cache'
            if lineage_path.exists() and intent.exists():
                raise ValueError('Cache has conflicting imported and local request provenance')
            if target.exists():
                if lineage_path.exists():
                    lineage = _bound_lineage(lineage_path, target, key)
                    if sha(target) != lineage['target_sha256']:
                        raise ValueError('Current run cache differs from imported source hash')
                elif intent.exists():
                    _bound_intent(intent, signature, key)
                else:
                    raise ValueError('Orphan current run cache has no lineage or local request intent')
            elif lineage_path.exists():
                # An interrupted import is pinned to its already recorded source,
                # even if another root now contains a successful same-hash answer.
                lineage = _bound_lineage(lineage_path, target, key)
                raw = Path(lineage['source_path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != lineage['source_sha256']:
                    raise ValueError('Interrupted import source hash differs')
                _source_record(raw, signature)
                _publish_bytes(target, raw)
                source = 'inherited_exact_request'
            elif intent.exists():
                _bound_intent(intent, signature, key)
                raise RuntimeError('Prior unresolved logical request is retained; no implicit redispatch: ' + key)
            else:
                for root in self.cache_roots:
                    candidate = root / (key + '.json')
                    if not candidate.exists():
                        continue
                    raw = candidate.read_bytes()
                    _source_record(raw, signature)
                    raw_sha = hashlib.sha256(raw).hexdigest()
                    lineage = {
                        'source_path': str(candidate), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                        'target_path': str(target.resolve()), 'target_sha256': raw_sha, 'request_hash': key}
                    # The provenance is durable before any target can be reused.
                    _publish_bytes(lineage_path, (json.dumps(lineage, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
                    _publish_bytes(target, raw)
                    source = 'inherited_exact_request'
                    break
            if target.exists():
                if read(target).get('request') != signature:
                    raise ValueError('Current run raw request cache differs')
            else:
                intent.parent.mkdir(parents=True, exist_ok=True)
                with intent.open('x', encoding='utf-8') as handle:
                    json.dump({'request_hash': key, 'signature': signature, 'context': self.context,
                               'stage': stage, 'retries': self.retries,
                               'created_utc': datetime.now(timezone.utc).isoformat()}, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                source = 'new_request'
            event = {'request_hash': key, 'stage': stage, 'provider': provider, 'model': model,
                     'operation': operation, 'source': source, 'context': self.context,
                     'cache_path': str(target), 'cache_sha256': None, 'error_type': None}
            try:
                return super().request(provider, model, body, operation=operation, stage=stage, cache=True)
            except Exception as exc:
                event['error_type'] = type(exc).__name__
                raise
            finally:
                if target.exists():
                    event['cache_sha256'] = sha(target)
                self.events.append(event)
