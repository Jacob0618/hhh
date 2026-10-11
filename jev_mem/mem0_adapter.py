"""Audited session ingestion through the pinned, otherwise unchanged Mem0 SDK.

This adapter does not implement an alternative extractor.  Each successful
session invokes ``Memory.add`` exactly once with all input turns.  Only the
existing prompt builder's observation date and last-k message input are adapted.
The current date is frozen at adapter construction (UTC today by default).

The audit directory and the Memory store are one inseparable run.  A failed or
interrupted write is deliberately not retried automatically: Mem0 persistence is
not transactional across its stores.  Rebuild the affected conversation in a
fresh run/scope, or explicitly reconcile its records before continuing.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PINNED_MEM0_VERSION = "2.2.1"
_INGEST_LOCK = threading.RLock()
_BUILDER_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "jev_mem_session_prompt_context", default=None
)


class SessionRecoveryRequired(RuntimeError):
    """A prior attempt may have written data and cannot be blindly repeated."""


class IncompleteSessionError(RuntimeError):
    """The SDK did not persist all nonduplicate facts returned by extraction."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_name(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:56] or "item"
    return stem + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _validate_date(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Dates must be explicit YYYY-MM-DD strings")
    return value


class _CaptureLLM:
    def __init__(self, delegate: Any, record: dict, flush):
        self.delegate, self.record, self.flush = delegate, record, flush

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate_response(self, *args, **kwargs):
        from mem0.memory.utils import extract_json, remove_code_blocks

        row = {"args": copy.deepcopy(args), "kwargs": copy.deepcopy(kwargs), "started_at": _utc_now()}
        self.record["llm_calls"].append(row)
        self.flush()
        try:
            response = self.delegate.generate_response(*args, **kwargs)
            row["response"] = response
            row["finished_at"] = _utc_now()
            # Mem0 silently converts malformed JSON into an empty extraction.
            # Validate the same parsed object before returning it to that parser.
            # This is a fail-closed audit check, not a prompt or output rewrite.
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Extraction did not return a nonempty JSON string")
            cleaned = remove_code_blocks(response)
            try:
                decoded = json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                decoded = json.loads(extract_json(cleaned), strict=False)
            if not isinstance(decoded, dict) or not isinstance(decoded.get("memory"), list):
                raise ValueError("Extraction must contain a memory list (an empty list is valid)")
            if any(not isinstance(item, dict) or not isinstance(item.get("text"), str)
                   or not item["text"].strip() for item in decoded["memory"]):
                raise ValueError("Each extracted memory must have a nonempty text string")
            row["parsed_memory"] = decoded["memory"]
            self.flush()
            return response
        except Exception as error:
            # Provider errors may contain URLs/keys.  Never copy their text into
            # the adapter's disk audit; provider clients own sanitized logging.
            row["error_type"] = type(error).__name__
            row["finished_at"] = _utc_now()
            self.flush()
            raise


class _CaptureVectorStore:
    def __init__(self, delegate: Any, record: dict):
        self.delegate, self.record = delegate, record

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def search(self, *args, **kwargs):
        results = self.delegate.search(*args, **kwargs)
        self.record["existing_memory_context"] = [
            {"id": str(item.id), "text": (item.payload or {}).get("data", ""),
             "hash": (item.payload or {}).get("hash")}
            for item in results
        ]
        return results


class _WarningAudit(logging.Handler):
    def __init__(self, record: dict):
        super().__init__(logging.WARNING)
        self.record = record

    def emit(self, event):
        # Even msg may already be an f-string containing a provider URL/key.
        self.record["sdk_warnings"].append({"level": event.levelname, "function": event.funcName,
                                           "source_line": event.lineno})


@contextlib.contextmanager
def _adapt_builder(record: dict):
    from mem0.memory import main as module

    original = module.generate_additive_extraction_prompt
    token = _BUILDER_CONTEXT.set(record)

    def contextual_builder(*args, **kwargs):
        active = _BUILDER_CONTEXT.get()
        if active is not None:
            kwargs["timestamp"] = active["observation_date"]
            kwargs["current_date"] = active["current_date"]
            kwargs["last_k_messages"] = [row["message"] for row in active["history_context"]]
            active["builder_call_count"] += 1
        return original(*args, **kwargs)

    module.generate_additive_extraction_prompt = contextual_builder
    try:
        yield
    finally:
        module.generate_additive_extraction_prompt = original
        _BUILDER_CONTEXT.reset(token)


class Mem0SessionAdapter:
    """Sequential whole-session writer for a dedicated synchronous Memory.

    Root code injects its real LLM and embedding providers into ``memory`` before
    constructing this adapter.  No credentials are accepted or inspected here.
    Use a single adapter/audit directory for a given Memory store, and do not
    write to that store outside the adapter.  Calls within this process are
    serialized; the run directory must have one process owner.
    """

    def __init__(self, memory: Any, audit_dir: str | Path, current_date: str | None = None):
        from mem0.memory import main as module

        if importlib.metadata.version("mem0ai") != PINNED_MEM0_VERSION:
            raise RuntimeError("Adapter protocol requires the pinned Mem0 2.2.1 SDK")
        checkout_source = Path(__file__).resolve().parents[1] / "upstream/mem0/mem0/memory/main.py"
        if checkout_source.exists() and hashlib.sha256(Path(module.__file__).read_bytes()).digest() != hashlib.sha256(
            checkout_source.read_bytes()
        ).digest():
            raise RuntimeError("Installed Memory source differs from the pinned local checkout")
        self.memory = memory
        self.audit_dir = Path(audit_dir).resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = _validate_date(current_date or datetime.now(timezone.utc).date().isoformat())

    def _records(self, user_id: str) -> list[dict]:
        scope_dir = self.audit_dir / _safe_name(user_id)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in scope_dir.glob("*.json")]
        return sorted(records, key=lambda item: item["sequence"])

    def add_session(
        self, *, user_id: str, session_id: str, messages: list[dict], observation_date: str,
        source_turn_ids: list[str] | None = None, metadata: dict | None = None,
    ) -> dict:
        """Submit all turns once; return audit plus actual SDK ``results``.

        ``source_turn_ids`` identify the input scope, never verified support.
        Existing success checkpoints are returned without a repeated model call,
        after checking the input fingerprint and persisted memory IDs.  Empty
        extraction is a valid success only for an explicit JSON memory: [] result.
        """
        from mem0.memory import main as module

        if not user_id or not session_id:
            raise ValueError("user_id and session_id must be nonempty")
        _validate_date(observation_date)
        if not isinstance(messages, list) or not messages:
            raise ValueError("A session requires a nonempty message list")
        if any(not isinstance(row, dict) or row.get("role") not in {"user", "assistant"}
               or not isinstance(row.get("content"), str) or not row["content"].strip()
               or set(row) - {"role", "content", "name"} for row in messages):
            raise ValueError("Session messages must contain only role/content/optional name; no QA fields")
        if source_turn_ids is None:
            source_turn_ids = [f"{session_id}:turn:{index}" for index in range(len(messages))]
        if len(source_turn_ids) != len(messages) or len(set(source_turn_ids)) != len(source_turn_ids):
            raise ValueError("One unique source_turn_id is required for each message")
        if any(not isinstance(item, str) or not item for item in source_turn_ids):
            raise ValueError("source_turn_ids must be nonempty strings")
        metadata = copy.deepcopy(metadata or {})
        reserved = {"user_id", "agent_id", "run_id", "data", "hash", "created_at", "updated_at",
                    "jev_mem_session_id", "jev_mem_observation_date"}
        if reserved & set(metadata):
            raise ValueError("Caller metadata cannot override SDK scope, payload, or adapter fields")
        request = {"user_id": user_id, "session_id": session_id, "messages": copy.deepcopy(messages),
                   "observation_date": observation_date, "current_date": self.current_date,
                   "source_turn_ids": list(source_turn_ids), "metadata": metadata}
        fingerprint = _json_hash(request)
        path = self.audit_dir / _safe_name(user_id) / (_safe_name(session_id) + ".json")

        with _INGEST_LOCK:
            previous = self._records(user_id)
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["request_sha256"] != fingerprint:
                    raise SessionRecoveryRequired("This session ID already has a different recorded request")
                if saved["status"] != "success":
                    raise SessionRecoveryRequired("Earlier session attempt failed or was interrupted; reconcile/rebuild")
                for fact in saved["results"]:
                    persisted = self.memory.vector_store.get(vector_id=fact["id"])
                    if persisted is None or (persisted.payload or {}).get("data") != fact["memory"]:
                        raise SessionRecoveryRequired("Success checkpoint does not match the attached Memory store")
                saved["replayed"] = True
                return saved
            if any(row["status"] != "success" for row in previous):
                raise SessionRecoveryRequired("Conversation has an unresolved write; cannot ingest later sessions")
            if previous and observation_date < previous[-1]["observation_date"]:
                raise ValueError("Sessions must be submitted in chronological order")
            history = []
            for row in previous:
                history.extend({"message": message, "session_id": row["session_id"], "source_turn_id": turn_id}
                               for message, turn_id in zip(row["messages"], row["source_turn_ids"]))
            record = {**request, "status": "in_progress", "request_sha256": fingerprint,
                      "sequence": len(previous), "started_at": _utc_now(), "replayed": False,
                      "history_context": history[-10:], "builder_call_count": 0,
                      "existing_memory_context": [], "llm_calls": [], "results": [], "fact_records": [],
                      "sdk_warnings": [], "audit_path": str(path),
                      "adapter_protocol": "mem0_2.2.1_session_date_ordered_history_v1",
                      "patch_scope": ["builder.timestamp", "builder.current_date_frozen",
                                      "builder.last_k_messages", "fail_closed_output_and_persistence_checks"],
                      "system_prompt_changed": False, "extraction_fact_limit": None}
            flush = lambda: _atomic_json(path, record)
            flush()  # Written before any provider or store mutation.
            old_llm, old_store = self.memory.llm, self.memory.vector_store
            handler = _WarningAudit(record)
            module.logger.addHandler(handler)
            self.memory.llm = _CaptureLLM(old_llm, record, flush)
            self.memory.vector_store = _CaptureVectorStore(old_store, record)
            try:
                add_metadata = {**metadata, "jev_mem_session_id": session_id,
                                "jev_mem_observation_date": observation_date}
                with _adapt_builder(record):
                    result = self.memory.add(copy.deepcopy(messages), user_id=user_id,
                                             metadata=add_metadata, infer=True)
                record["sdk_response"] = result
                if record["builder_call_count"] != 1 or len(record["llm_calls"]) != 1:
                    raise IncompleteSessionError("Expected one SDK extraction for the whole session")
                rows = result.get("results") if isinstance(result, dict) else None
                if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get("event") != "ADD"
                                                     or not row.get("id") or not row.get("memory") for row in rows):
                    raise IncompleteSessionError("Unexpected SDK return shape or a non-ADD operation")
                record["results"] = rows
                extracted = record["llm_calls"][0]["parsed_memory"]
                existing_hashes = {row["hash"] for row in record["existing_memory_context"] if row["hash"]}
                expected_texts = {row["text"] for row in extracted
                                  if hashlib.md5(row["text"].encode()).hexdigest() not in existing_hashes}
                actual_texts = {row["memory"] for row in rows}
                record["persistence_check"] = {"extracted_count": len(extracted),
                                               "expected_unique_nonduplicate_count": len(expected_texts),
                                               "returned_add_count": len(rows),
                                               "missing_texts": sorted(expected_texts - actual_texts),
                                               "unexpected_texts": sorted(actual_texts - expected_texts)}
                if actual_texts != expected_texts:
                    raise IncompleteSessionError("Some nonduplicate extracted facts were not returned as persisted ADDs")
                for row in rows:
                    stored = old_store.get(vector_id=row["id"])
                    if stored is None or (stored.payload or {}).get("data") != row["memory"]:
                        raise IncompleteSessionError("Returned ADD is missing or different in the vector store")
                input_scope = {"session_id": session_id, "source_turn_ids": list(source_turn_ids),
                               "historical_turn_ids": [row["source_turn_id"] for row in history[-10:]],
                               "existing_memory_ids": [row["id"] for row in record["existing_memory_context"]]}
                record["fact_records"] = [
                    {"id": row["id"], "text": row["memory"], "event": "ADD", "user_id": user_id,
                     "observation_date": observation_date, "input_scope": copy.deepcopy(input_scope),
                     "verified_support": [], "support_status": "unverified"} for row in rows
                ]
                record["status"] = "success"
                record["finished_at"] = _utc_now()
                flush()
                return record
            except BaseException as error:
                record["status"] = "requires_recovery"
                record["error_type"] = type(error).__name__
                record["finished_at"] = _utc_now()
                flush()
                raise
            finally:
                self.memory.llm, self.memory.vector_store = old_llm, old_store
                module.logger.removeHandler(handler)
