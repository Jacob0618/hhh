"""Transport and durable evidence for the five adapted baseline comparison.

No credentials or network on import. Scientific algorithms live in pinned
upstream modules. This file only handles transport, lifecycle, and accounting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone

from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from jev_mem.gemini25_ablation_provider_v1 import (
    generation_body, response_text, check_request, MODEL, READER_STAGE, JUDGE_STAGE,
)
from jev_mem.providers import dump_json
from jev_mem.sdk20_cached_api import digest

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('amem', 'simplemem', 'hypermem', 'vmem', 'prism')
_EVENT_LOCK = threading.Lock()


class BaselineFatal(BaseException):
    """A method must not silently turn a provider/schema failure into memory.

    BaseException is deliberate: upstream broad `except Exception` fallbacks
    must not manufacture empty memory after an exhausted/unknown request.
    The experiment worker catches this only at its terminal boundary.
    """


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _EVENT_LOCK, path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


class BaselineAPI(PacedEmbeddingAPI):
    """Unchanged durable raw requests/retries, new declared generation stages."""
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        if cache_roots:
            raise ValueError('Historical response reuse is forbidden')
        super().__init__(run_dir, cache_roots=(), **kwargs)

    def request(self, provider, model, body, *, operation='generateContent',
                stage='unspecified', cache=True):
        if operation == 'generateContent':
            if stage in (READER_STAGE, JUDGE_STAGE):
                check_request(provider, model, body, operation, stage)
            else:
                if provider != 'gemini' or model != MODEL:
                    raise BaselineFatal('Only Gemini 2.5 Flash generation is admitted')
                if not any(stage.startswith(m + '_') for m in METHODS):
                    raise BaselineFatal('Method generation stage must be declared')
                cfg = body.get('generationConfig', {})
                if (cfg.get('thinkingConfig') != {'thinkingBudget': 0}
                        or cfg.get('maxOutputTokens') != 16384
                        or set(cfg) - {'thinkingConfig', 'maxOutputTokens', 'responseMimeType'}
                        or cfg.get('responseMimeType') not in (None, 'application/json')):
                    raise BaselineFatal('Frozen internal generation configuration differs')
        elif provider != 'gemini' or model != 'gemini-embedding-001':
            raise BaselineFatal('Only the frozen Gemini embedding/countTokens endpoint is admitted')
        start = len(self.events)
        try:
            return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)
        except Exception as exc:
            raise BaselineFatal(self._redact(str(exc))) from None
        finally:
            for event in self.events[start:]:
                append(self.run_dir / 'logical_events.jsonl', {'recorded_utc': now(), **event})

    def generate(self, prompt, *, model=MODEL, system=None, json_output=False,
                 max_output=16384, stage='unspecified'):
        body = generation_body(prompt, system=system, json_output=json_output,
                               max_output=max_output)
        raw = self.request('gemini', model, body, stage=stage)
        try:
            text = response_text(raw)
            return json.loads(text) if json_output else text
        except Exception as exc:
            raise BaselineFatal('Retained generation cannot be parsed: ' + type(exc).__name__) from None

    def embed(self, texts, *, query=False, stage='embedding'):
        try:
            return super().embed(texts, query=query, stage=stage)
        except Exception as exc:
            raise BaselineFatal(self._redact(str(exc))) from None


def error_record(exc, api=None):
    return {'type': type(exc).__name__, 'message': api._redact(str(exc)) if api else str(exc)}
