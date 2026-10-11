"""Structured transport for the unchanged Mem0 short-answer evaluation.

The original evaluation.py builds both prompts and results. This wrapper adds
only responseSchema and durable request/event accounting. It has no credentials
or transport constructor; callers supply their scoped ModelAPI (retries 0..2).
An unknown prior logical attempt is never silently resent by this entry point.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import unicodedata
import uuid

from jev_mem import evaluation
from jev_mem.pilot400 import fingerprint
from tools.pilot400_recovery_qa_v1 import (
    generation_signature, request_hash, decode_response, READER_SCHEMA, JUDGE_SCHEMA)
from tools.run_pilot400_experiments_v2 import read, file_sha

PROTOCOL = "mem0-transfer-structured-qa-v1"
_LOCK = threading.Lock()
_HASH_LOCKS = {}
_CONTEXT_KEYS = {"architecture", "conversation", "query_id", "method", "arm", "phase",
                 "driver_stage", "group_id", "recovery_version"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_decoded(value, stage):
    if stage == "qa_reader":
        if (not isinstance(value, dict) or set(value) != {"answer", "used_fact_ids"}
                or not isinstance(value["answer"], str)
                or not isinstance(value["used_fact_ids"], list)
                or any(not isinstance(x, str) for x in value["used_fact_ids"])
                or len(value["used_fact_ids"]) > 20
                or len(set(value["used_fact_ids"])) != len(value["used_fact_ids"])):
            raise ValueError("Reader violates the declared string-answer/evidence-ID schema")
    else:
        if (not isinstance(value, dict) or set(value) != {"correct", "completeness", "reason"}
                or type(value["correct"]) is not bool or not isinstance(value["reason"], str)
                or isinstance(value["completeness"], bool)
                or not isinstance(value["completeness"], (int, float))
                or not math.isfinite(value["completeness"]) or not 0 <= value["completeness"] <= 1):
            raise ValueError("Judge violates the declared correctness/completeness schema")
    return value


class StructuredMem0QAAPI:
    """Wrap one scoped provider; events contain hashes/metadata, never prompt text."""
    def __init__(self, base):
        if type(base.retries) is not int or not 0 <= base.retries <= 2:
            raise ValueError("Declare 0..2 provider-internal technical retries")
        self.base = base
        self.run_dir = Path(base.run_dir).resolve()
        self.events = []

    def scoped(self, **context):
        return type(self)(self.base.scoped(**context))

    def _event(self, value):
        context = {k: v for k, v in self.base.context.items() if k in _CONTEXT_KEYS}
        event_id = uuid.uuid4().hex
        path = self.run_dir / "mem0_transfer_qa_events" / fingerprint(context) / (event_id + ".json")
        row = {"protocol": PROTOCOL, "event_id": event_id, "event_path": str(path),
               "recorded_utc": _now(), "context": context, **value}
        # Durable before exposing a successful stage to its consumer.
        _write_once(path, row)
        self.events.append(row)

    def generate(self, prompt, *, model="gemini-2.5-flash", system=None,
                 json_output=False, max_output=8192, stage="generation"):
        if stage == "qa_reader":
            expected = ("gemini-2.5-flash", evaluation.READER_SYSTEM, READER_SCHEMA)
        elif stage == "strict_judge":
            expected = ("gemini-3.5-flash-lite", evaluation.JUDGE_SYSTEM, JUDGE_SCHEMA)
        else:
            raise ValueError("Mem0 transfer QA permits only qa_reader and strict_judge")
        if (not isinstance(prompt, str) or not prompt or (model, system, json_output, max_output)
                != (expected[0], expected[1], True, 2048)):
            raise ValueError("Cannot change the frozen QA model/system/token settings")
        signature = generation_signature(prompt, model=model, system=system, json_output=True,
            max_output=2048, schema=expected[2])
        legacy = generation_signature(prompt, model=model, system=system, json_output=True, max_output=2048)
        digest = request_hash(signature)
        cache_path = self.run_dir / "api_cache" / (digest + ".json")
        intent_path = self.run_dir / "mem0_transfer_qa_intents" / (digest + ".json")
        expected_intent = {"protocol": PROTOCOL, "stage": stage, "provider": "gemini", "model": model,
            "operation": "generateContent", "request_hash": digest, "body_sha256": fingerprint(signature["body"]),
            "provider_retries": self.base.retries, "max_http_attempts": self.base.retries + 1}
        metadata = {"stage": stage, "provider": "gemini", "model": model, "operation": "generateContent",
            "request_hash": digest, "legacy_request_hash": request_hash(legacy), "intent_path": str(intent_path),
            "cache_path": str(cache_path), "cache_sha256": None, "response_sha256": None,
            "provider_retry_budget": self.base.retries, "max_http_attempts": self.base.retries + 1,
            "new_logical_dispatch": False, "cache_source": "not_dispatched", "error_type": None}
        with _LOCK:
            lock = _HASH_LOCKS.setdefault((str(self.run_dir), digest), threading.Lock())
        with lock:
            try:
                if intent_path.exists():
                    intent = read(intent_path)
                    if any(intent.get(k) != v for k, v in expected_intent.items()):
                        raise ValueError("Existing QA intent differs from frozen request/retry policy")
                    if not cache_path.exists():
                        metadata["cache_source"] = "prior_unknown_no_redispatch"
                        raise RuntimeError("Prior QA logical attempt has no durable response; review before a new version")
                else:
                    if cache_path.exists():
                        raise RuntimeError("QA cache has no attributable durable intent")
                    context = {k: v for k, v in self.base.context.items() if k in _CONTEXT_KEYS}
                    _write_once(intent_path, {**expected_intent, "created_utc": _now(), "pid": os.getpid(), "context": context})
                existed = cache_path.exists()
                metadata["cache_source"] = "validated_cache" if existed else "new_http"
                if existed:
                    cached = read(cache_path)
                    if cached.get("request") != signature or cached.get("stage") != stage:
                        metadata["cache_source"] = "invalid_existing_cache"
                        raise ValueError("Existing raw response request/stage binding differs")
                    _validate_decoded(decode_response(cached["response"]), stage)
                metadata["new_logical_dispatch"] = not existed
                # This is one logical dispatch. Only ModelAPI's predeclared retry
                # budget can produce more than one physical HTTP attempt.
                response = self.base.request("gemini", model, signature["body"], stage=stage)
                if not cache_path.exists():
                    raise RuntimeError("Provider returned without a durable response cache")
                cached = read(cache_path)
                metadata.update(cache_sha256=file_sha(cache_path), response_sha256=fingerprint(response))
                if (cached.get("request") != signature or cached.get("stage") != stage
                        or cached.get("response") != response):
                    raise ValueError("Returned HTTP response differs from its exact durable cache")
                decoded = _validate_decoded(decode_response(response), stage)
            except Exception as error:
                metadata["error_type"] = type(error).__name__
                if cache_path.exists():
                    metadata["cache_sha256"] = file_sha(cache_path)
                    metadata["cache_source"] = ("new_http_invalid_response" if metadata["new_logical_dispatch"]
                                                  else "invalid_existing_cache")
                elif metadata["new_logical_dispatch"]:
                    metadata["cache_source"] = "failed_or_unknown_response"
                self._event(metadata)
                raise
            self._event(metadata)
            return copy.deepcopy(decoded)


class Mem0TransferQAError(RuntimeError):
    def __init__(self, cause, events):
        super().__init__("Mem0 transfer QA failed: " + type(cause).__name__)
        self.cause_type = type(cause).__name__
        self.api_events = copy.deepcopy(events)


def evaluate(api, question, reference, selected, reference_date):
    """Keep original evaluator's prompts and fields; enforce exactly20 evidence."""
    if (not isinstance(selected, list) or len(selected) != 20
            or any(not isinstance(f, dict) or not isinstance(f.get("id"), str) or not f["id"]
                   or not isinstance(f.get("text"), str) or not f["text"].strip() for f in selected)
            or len({f["id"] for f in selected}) != 20
            or len({" ".join(unicodedata.normalize("NFC", f["text"]).split()) for f in selected}) != 20):
        raise ValueError("Mem0 transfer QA requires exactly20 unique IDs and normalized fact texts")
    wrapper = api if isinstance(api, StructuredMem0QAAPI) else StructuredMem0QAAPI(api)
    before = len(wrapper.events)
    try:
        result = evaluation.answer_and_judge(wrapper, question, reference, selected, reference_date=reference_date)
    except Exception as error:
        raise Mem0TransferQAError(error, wrapper.events[before:]) from error
    return {"result": result, "api_events": copy.deepcopy(wrapper.events[before:])}
