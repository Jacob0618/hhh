"""Audited model clients. Credentials and request headers are never logged."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import threading
import time
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()


def _pace_jev():
    """15 HTTP attempts/s shared by this workspace's processes; cache is free."""
    path = ROOT / "runtime" / "jev_rate_gate.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS gate (name TEXT PRIMARY KEY, next_at REAL)")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT next_at FROM gate WHERE name='jev'").fetchone()
        now = time.time()
        scheduled = max(now, row[0] if row else now)
        connection.execute("INSERT OR REPLACE INTO gate VALUES ('jev', ?)", (scheduled + 1 / 15,))
        connection.commit()
    finally:
        connection.close()
    wait = scheduled - time.time()
    if wait > 0:
        time.sleep(wait)


def load_credentials():
    # Existing user-authorized current-user DPAPI vault; do not copy its secrets.
    vault_module = Path(r"C:\Users\86139\Desktop\jev\experiments\live\credential_store.py")
    if vault_module.exists():
        spec = importlib.util.spec_from_file_location("jev_dpapi_credentials", vault_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = module.load()
        return {key: values[key] for key in ("GEMINI_API_KEY", "TYPESAFE_API_KEY")}
    values = {key: os.environ.get(key) for key in ("GEMINI_API_KEY", "TYPESAFE_API_KEY")}
    if not all(values.values()):
        raise RuntimeError("Required credentials missing; no credential values logged")
    return values


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class ModelAPI:
    def __init__(self, run_dir, *, keys=None, retries=2, timeout=150, context=None):
        self.keys = keys or load_credentials()
        self.run_dir = Path(run_dir)
        self.cache_dir = self.run_dir / "api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.timeout = timeout
        self.local = threading.local()
        self.context = dict(context or {})

    def scoped(self, **context):
        """Bind audit attribution, including requests from nested worker threads."""
        return ModelAPI(self.run_dir, keys=self.keys, retries=self.retries,
                        timeout=self.timeout, context={**self.context, **context})

    def _redact(self, value):
        text = str(value)
        for key in self.keys.values():
            text = text.replace(key, "<redacted>")
        return text

    def _log(self, row):
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "context": self.context, **row}
        with _LOCK:
            with (self.run_dir / "api_calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(self._redact(json.dumps(row, ensure_ascii=False)) + "\n")

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        request_hash = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / (request_hash + ".json")
        if cache and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            self._log({"stage": stage, "provider": provider, "model": model,
                       "operation": operation, "request_hash": request_hash,
                       "cache_hit": True, "latency_ms": 0,
                       "usage": record["response"].get("usageMetadata", record["response"].get("usage", {})),
                       "usage_semantics": "reused_response_no_new_model_usage",
                       "resolved_model": record["response"].get("modelVersion", record["response"].get("model"))})
            return record["response"]
        if provider == "jev":
            url = "https://api.typesafe.ai/v1/systemone"
            headers = {"Authorization": "Bearer " + self.keys["TYPESAFE_API_KEY"]}
        elif provider == "gemini":
            if not model.startswith("gemini-") or any(c in model for c in "/?&#"):
                raise ValueError("Invalid model ID")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{operation}"
            headers = {"x-goog-api-key": self.keys["GEMINI_API_KEY"]}
        else:
            raise ValueError("Unknown provider")
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        for attempt in range(self.retries + 1):
            if provider == "jev":
                _pace_jev()
            start = time.perf_counter()
            retry_delay = min(2 ** (attempt + 1), 15)
            try:
                response = self.local.session.post(url, headers=headers, json=body, timeout=(20, self.timeout))
                try:
                    data = response.json()
                except ValueError:
                    data = {"error": self._redact(response.text[:1500])}
                row = {"stage": stage, "provider": provider, "model": model, "operation": operation,
                       "request_hash": request_hash, "attempt": attempt + 1, "status": response.status_code,
                       "latency_ms": (time.perf_counter() - start) * 1000,
                       "usage": data.get("usageMetadata", data.get("usage", {})),
                       "resolved_model": data.get("modelVersion", data.get("model")),
                       "request_id": response.headers.get("x-request-id"), "cache_hit": False}
                if not response.ok:
                    row["error"] = self._redact(data.get("error", data))[:1500]
                self._log(row)
                if response.ok:
                    dump_json(path, {"request": signature, "response": data, "stage": stage})
                    return data
                if response.status_code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries:
                    raise RuntimeError(f"{provider}/{model} HTTP {response.status_code}: {row['error']}")
                try:
                    retry_delay = max(retry_delay, float(response.headers.get("retry-after", 0)))
                except ValueError:
                    pass
            except requests.RequestException as exc:
                self._log({"stage": stage, "provider": provider, "model": model, "operation": operation,
                           "request_hash": request_hash, "attempt": attempt + 1, "status": "transport_error",
                           "latency_ms": (time.perf_counter() - start) * 1000,
                           "error": self._redact(exc)[:1500], "usage_unknown": True})
                if attempt == self.retries:
                    raise RuntimeError(self._redact(exc)) from None
            time.sleep(retry_delay)
        raise RuntimeError("Request did not produce a response")

    def generate(self, prompt, *, model="gemini-2.5-flash", system=None, json_output=False,
                 max_output=8192, stage="generation"):
        config = {"temperature": 0, "maxOutputTokens": max_output}
        if model == "gemini-2.5-flash":
            config["thinkingConfig"] = {"thinkingBudget": 0}
        if json_output:
            config["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        response = self.request("gemini", model, body, stage=stage)
        candidates = response.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            raise RuntimeError("Generation missing or incomplete: " + str([c.get("finishReason") for c in candidates]))
        text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                       if not p.get("thought"))
        if not text.strip():
            raise RuntimeError("Empty generation")
        return json.loads(text) if json_output else text

    def score_many(self, state, criteria, *, stage="jev_score"):
        model = "jev-1.13.0"
        body = {"model": model, "state": state,
                "questions": {key: {"type": "noul", "instructions": value} for key, value in criteria.items()}}
        response = self.request("jev", model, body, operation="systemone", stage=stage)
        scores = {key: response["answers"][key]["noul"] for key in criteria}
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in scores.values()):
            raise RuntimeError("Invalid Noul scores")
        return scores

    def embed(self, texts, *, query=False, stage="embedding"):
        model = "gemini-embedding-001"
        for text in texts:
            # Conservative byte bound for short inputs; tokenize longer strings
            # with the exact embedding model before submitting, never truncate.
            if len(text.encode("utf-8")) > 1800:
                count = self.count_tokens(text, model=model, stage=stage + "_length_check")
                if count > 2048:
                    raise ValueError(f"Embedding input has {count} tokens; limit is 2048. No truncation applied.")
        result = []
        for start in range(0, len(texts), 32):
            chunk = texts[start:start + 32]
            body = {"requests": [{"model": "models/" + model,
                                   "content": {"parts": [{"text": text}]},
                                   "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                                   "outputDimensionality": 768} for text in chunk]}
            response = self.request("gemini", model, body, operation="batchEmbedContents", stage=stage)
            embeddings = response.get("embeddings", [])
            if len(embeddings) != len(chunk):
                raise RuntimeError("Embedding batch count mismatch")
            for embedding in embeddings:
                vector = embedding["values"]
                norm = math.sqrt(sum(x * x for x in vector))
                if len(vector) != 768 or not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Invalid embedding vector")
                result.append([x / norm for x in vector])
        return result

    def count_tokens(self, text, *, model="gemini-embedding-001", stage="token_count"):
        body = {"contents": [{"parts": [{"text": text}]}]}
        result = self.request("gemini", model, body, operation="countTokens", stage=stage)
        return int(result["totalTokens"])


class ExtractionLLM:
    def __init__(self, api):
        self.api = api
        self.config = type("Config", (), {"model": "gemini-2.5-flash"})()

    def generate_response(self, messages, **kwargs):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        data = self.api.generate(prompt, system=system, json_output=True, max_output=16384, stage="mem0_extract")
        if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
            raise RuntimeError("Invalid extraction JSON: memory list required")
        if any(not isinstance(m, dict) or not isinstance(m.get("text"), str) or not m["text"].strip()
               for m in data["memory"]):
            raise RuntimeError("Invalid extracted memory record")
        return json.dumps(data, ensure_ascii=False)


class Embeddings:
    def __init__(self, api):
        self.api = api
        self.config = type("Config", (), {"model": "gemini-embedding-001", "embedding_dims": 768})()

    def embed(self, text, memory_action="add"):
        return self.api.embed([text], query=memory_action == "search", stage="mem0_embedding_" + str(memory_action))[0]

    def embed_batch(self, texts, memory_action="add"):
        return self.api.embed(texts, query=memory_action == "search", stage="mem0_embedding_" + str(memory_action))
