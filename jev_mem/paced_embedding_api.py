"""Physical-attempt pacing for the unchanged legacy Gemini embedding request.

This is an opt-in transport adapter, not a retrying controller. It preserves the
legacy body, signature, parser, cache and two-retry policy. A SQLite gate is
shared by processes; cached responses never reach the gate. No work runs on
import and no credentials, URL, headers or prompt text are written by this file.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from jev_mem.providers import ROOT
from jev_mem.sdk20_cached_api import CachedLegacyAPI, digest

PROTOCOL = "legacy-gemini-embedding-physical-pacing-v1"
MODEL = "gemini-embedding-001"
OPERATION = "batchEmbedContents"
DEFAULT_GATE = ROOT / "runtime" / "embedding_physical_gate_v1.sqlite"
_LOG_LOCK = threading.Lock()


def _is_embedding_url(url):
    value = urlsplit(url)
    return (value.scheme == "https"
            and value.netloc == "generativelanguage.googleapis.com"
            and value.path == "/v1beta/models/gemini-embedding-001:batchEmbedContents"
            and not value.query and not value.fragment)


def _retry_after_seconds(response):
    # The original provider also interprets numeric Retry-After. Invalid/header
    # dates leave the fixed thirty-second shared cooldown in effect.
    try:
        value = float(response.headers.get("retry-after", 0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value >= 0 else 0.0


class SharedEmbeddingGate:
    """Admit one physical start per interval, rechecking new shared cooldowns.

    Future slots are never pre-reserved: a worker waiting when another receives
    429 must see the newly published cooldown before it can dispatch.
    """
    def __init__(self, path=DEFAULT_GATE, *, interval_seconds=3.0,
                 cooldown_seconds=30.0, clock=None, sleeper=None):
        if (not math.isfinite(interval_seconds) or interval_seconds <= 0
                or not math.isfinite(cooldown_seconds) or cooldown_seconds < 30):
            raise ValueError("Invalid shared embedding pacing policy")
        self.path = Path(path).resolve()
        self.interval_seconds = float(interval_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.clock = clock or time.time
        self.sleeper = sleeper or time.sleep

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS embedding_gate "
                     "(name TEXT PRIMARY KEY, next_at REAL NOT NULL, "
                     "cooldown_until REAL NOT NULL, interval_seconds REAL NOT NULL, "
                     "cooldown_seconds REAL NOT NULL)")
        return conn

    def _state(self, conn):
        row = conn.execute("SELECT next_at,cooldown_until,interval_seconds,cooldown_seconds "
                           "FROM embedding_gate WHERE name=?", (MODEL,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO embedding_gate VALUES (?,?,?,?,?)",
                         (MODEL, 0.0, 0.0, self.interval_seconds, self.cooldown_seconds))
            return 0.0, 0.0
        if row[2:] != (self.interval_seconds, self.cooldown_seconds):
            raise ValueError("Shared embedding gate policy differs; do not bypass the active gate")
        return row[0], row[1]

    def acquire(self):
        began = self.clock()
        while True:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                next_at, cooldown_until = self._state(conn)
                current = self.clock()
                wait = max(next_at, cooldown_until) - current
                if wait <= 0:
                    conn.execute("UPDATE embedding_gate SET next_at=? WHERE name=?",
                                 (current + self.interval_seconds, MODEL))
                    conn.commit()
                    return {"admitted_at_unix": current,
                            "waited_seconds": max(0.0, current - began),
                            "interval_seconds": self.interval_seconds,
                            "cooldown_until_unix": cooldown_until}
                conn.commit()
            finally:
                conn.close()
            # Recheck after bounded waits instead of honoring stale reservations.
            self.sleeper(min(wait, 1.0))

    def note_429(self, retry_after_seconds=0.0):
        if not math.isfinite(retry_after_seconds) or retry_after_seconds < 0:
            raise ValueError("Invalid numeric Retry-After")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            _, previous = self._state(conn)
            current = self.clock()
            until = max(previous, current + self.cooldown_seconds,
                        current + retry_after_seconds)
            conn.execute("UPDATE embedding_gate SET cooldown_until=? WHERE name=?",
                         (until, MODEL))
            conn.commit()
            return {"received_at_unix": current, "cooldown_until_unix": until,
                    "cooldown_seconds": self.cooldown_seconds,
                    "numeric_retry_after_seconds": retry_after_seconds}
        finally:
            conn.close()


class _PacedSession:
    """Narrow post proxy, installed on this API's thread-local Session only."""
    def __init__(self, delegate, gate, request_hash, emit):
        self.delegate, self.gate = delegate, gate
        self.request_hash, self.emit = request_hash, emit
        self.physical_attempt = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def post(self, url, *args, **kwargs):
        if not _is_embedding_url(url):
            return self.delegate.post(url, *args, **kwargs)
        self.physical_attempt += 1
        grant = self.gate.acquire()
        common = {"request_hash": self.request_hash,
                  "physical_attempt": self.physical_attempt}
        self.emit({**common, "event": "physical_embedding_admitted", **grant})
        response = self.delegate.post(url, *args, **kwargs)
        if response.status_code == 429:
            try:
                cooldown = self.gate.note_429(_retry_after_seconds(response))
            except Exception as error:
                # Do not retry a request because local pacing persistence failed.
                # Preserve that a physical HTTP429 was seen even if the original
                # provider cannot receive this response after the local failure.
                self.emit({**common, "event": "cooldown_persistence_failed",
                           "observed_http_status": 429, "error_type": type(error).__name__})
                raise
            self.emit({**common, "event": "shared_embedding_429_cooldown",
                       "observed_http_status": 429, **cooldown})
        return response


class PacedEmbeddingAPI(CachedLegacyAPI):
    """Legacy exact-cache API with shared pacing of every embedding POST.

    `retries=2` stays fixed. CountTokens, generation and JEV are not paced here;
    JEV retains its existing shared gate. Failed/unknown logical requests retain
    CachedLegacyAPI's durable intent and cannot be implicitly redispatched.
    """
    def __init__(self, run_dir, *, gate_path=DEFAULT_GATE, interval_seconds=3.0,
                 cooldown_seconds=30.0, **kwargs):
        if kwargs.get("retries", 2) != 2:
            raise ValueError("Pacing preserves the original retries=2 policy")
        super().__init__(run_dir, **kwargs)
        self.gate_path = Path(gate_path).resolve()
        self.interval_seconds = float(interval_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.embedding_gate = SharedEmbeddingGate(self.gate_path,
            interval_seconds=self.interval_seconds, cooldown_seconds=self.cooldown_seconds)

    def scoped(self, **context):
        return type(self)(self.run_dir, keys=self.keys, retries=self.retries,
                         timeout=self.timeout, context={**self.context, **context},
                         cache_roots=self.cache_roots, gate_path=self.gate_path,
                         interval_seconds=self.interval_seconds,
                         cooldown_seconds=self.cooldown_seconds)

    def _pacing_log(self, row):
        # A separate scheduling ledger: it is not a billable API-call ledger.
        # Keep attribution bounded and avoid serializing arbitrary context text.
        context = {key: value for key, value in self.context.items()
                   if key in ("architecture", "conversation", "method", "phase", "group_id")
                   and isinstance(value, (str, int))}
        value = {"protocol": PROTOCOL, "model": MODEL, "operation": OPERATION,
                 "context": context, **row}
        with _LOG_LOCK:
            with (self.run_dir / "embedding_pacing_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(self._redact(json.dumps(value, ensure_ascii=False)) + "\n")

    def request(self, provider, model, body, *, operation="generateContent",
                stage="unspecified", cache=True):
        if (provider, model, operation) != ("gemini", MODEL, OPERATION):
            return super().request(provider, model, body, operation=operation,
                                   stage=stage, cache=cache)
        # The unchanged legacy provider loops over Session.post for each physical
        # attempt. Install a scoped proxy there, not outside its retry loop.
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        original_session = self.local.session
        signature = {"provider": provider, "model": model,
                     "operation": operation, "body": body}
        self.local.session = _PacedSession(original_session, self.embedding_gate,
                                          digest(signature), self._pacing_log)
        try:
            return super().request(provider, model, body, operation=operation,
                                   stage=stage, cache=cache)
        finally:
            self.local.session = original_session
