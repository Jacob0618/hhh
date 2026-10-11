"""Separate, bounded structured-output recovery. Never mutates the original pilot.

Same prompts/models/evidence; only generationConfig.responseSchema is added.
No network or credentials on import or in --check. Every new request hash gets
one physical attempt, with durable intent before dispatch and no blind retries.
Google REST schema: https://ai.google.dev/api/generate-content#v1beta.GenerationConfig
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import evaluation_pilot as evaluation
from jev_mem.data import load_data
from jev_mem.pilot400 import ARMS, fingerprint, load_cohort, reader_fact
from jev_mem.providers import ModelAPI
from jev_mem.topic_grouping import raw_turns_from_locomo
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now

ORIGINAL_RUN = ROOT / "runs/pilot400_modules_v2"
DEFAULT_OUTPUT = ROOT / "runs/pilot400_recovery_v1"
POLICY = "pilot400-structured-qa-recovery-v1"
ORIGINAL_PROTOCOL = "34817b88f064dd0b12c650608dd4087c279681458aa6cfb9079deba893a389f5"
TERMINAL_SEAL = ROOT / "research/heartbeat_checks/20260928T0014/qa_terminal/seal.json"
READER_SCHEMA = {"type": "OBJECT", "properties": {
    "answer": {"type": "STRING"},
    "used_fact_ids": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": 20}},
    "required": ["answer", "used_fact_ids"], "propertyOrdering": ["answer", "used_fact_ids"]}
JUDGE_SCHEMA = {"type": "OBJECT", "properties": {
    "correct": {"type": "BOOLEAN"},
    "completeness": {"type": "NUMBER", "minimum": 0, "maximum": 1},
    "reason": {"type": "STRING"}},
    "required": ["correct", "completeness", "reason"],
    "propertyOrdering": ["correct", "completeness", "reason"]}
EXPECTED = {(2, ARMS[0], "c2_q50"), (4, ARMS[0], "c4_q1"), (7, ARMS[0], "c7_q20"),
            (3, ARMS[0], "c3_q197"), (4, ARMS[1], "c4_q1"), (5, ARMS[1], "c5_q36"),
            (7, ARMS[1], "c7_q20"), (7, ARMS[1], "c7_q85"),
            (4, ARMS[2], "c4_q1"), (8, ARMS[2], "c8_q34"), (3, ARMS[2], "c3_q197")}
_LOCK = threading.Lock()
_HASH_LOCKS = {}


def request_hash(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generation_signature(prompt, *, model, system, json_output, max_output, schema=None):
    config = {"temperature": 0, "maxOutputTokens": max_output}
    if model == "gemini-2.5-flash":
        config["thinkingConfig"] = {"thinkingBudget": 0}
    if json_output:
        config["responseMimeType"] = "application/json"
    if schema is not None:
        config["responseSchema"] = copy.deepcopy(schema)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return {"provider": "gemini", "model": model, "operation": "generateContent", "body": body}


def decode_response(response):
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Generation missing or incomplete")
    value = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                    if not p.get("thought"))
    if not value.strip():
        raise ValueError("Empty generation")
    return json.loads(value)


def _safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Recovery output must be a dedicated directory inside workspace runs")
    for name in ("pilot400_modules_v2", "pilot400_modules_v1", "pilot400_writer_recovery_v1",
                 "session_top20_v1", "independent_top20_v1", "technical_recovery_v1"):
        old = (ROOT / "runs" / name).resolve()
        if path == old or path in old.parents or old in path.parents:
            raise ValueError("Recovery output overlaps frozen original data")
    return path


def _sealed(row):
    return {**row, "content_sha256": fingerprint(row)}


def _verify_sealed(row):
    if row.get("content_sha256") != fingerprint({k: v for k, v in row.items() if k != "content_sha256"}):
        raise ValueError("Artifact content seal mismatch")
    return row


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ResponseSchemaAPI:
    """ModelAPI-compatible generation wrapper; structured requests have one attempt.

    original_reader permits only a hash-verified old reader response. It cannot
    reuse judges or invalid reader arrays. A shared hash mutex prevents A/C's
    identical judge request racing; durable request intent also blocks reruns
    after an unknown/crashed attempt. Raw responses stay in the new api_cache.
    """
    def __init__(self, base, *, original_reader=None):
        if base.retries != 0:
            raise ValueError("Recovery ModelAPI must use retries=0")
        self.base = base
        self.run_dir = Path(base.run_dir)
        self.original_reader = copy.deepcopy(original_reader)
        self.events = []

    def _redact(self, text):
        return self.base._redact(text)

    def scoped(self, **context):
        return ResponseSchemaAPI(self.base.scoped(**context), original_reader=self.original_reader)

    def generate(self, prompt, *, model="gemini-2.5-flash", system=None, json_output=False,
                 max_output=8192, stage="generation"):
        if not json_output or stage not in (evaluation.READER_STAGE, evaluation.JUDGE_STAGE):
            raise ValueError("Recovery wrapper supports only frozen pilot reader/judge JSON stages")
        expected = ((evaluation.READER_MODEL, evaluation.READER_SYSTEM, READER_SCHEMA)
                    if stage == evaluation.READER_STAGE else
                    (evaluation.JUDGE_MODEL, evaluation.JUDGE_SYSTEM, JUDGE_SCHEMA))
        if (model, system, max_output) != (expected[0], expected[1], evaluation.MAX_OUTPUT):
            raise ValueError("Recovery cannot change the frozen model/prompt/token settings")
        legacy = generation_signature(prompt, model=model, system=system, json_output=True, max_output=max_output)
        if stage == evaluation.READER_STAGE and self.original_reader is not None:
            source = self.original_reader
            if source["request"] != legacy or source["request_hash"] != request_hash(legacy):
                raise ValueError("Old reader reuse signature mismatch")
            path = Path(source["path"])
            if file_sha(path) != source["file_sha256"]:
                raise ValueError("Old reader response bytes changed")
            cached = read(path)
            if cached["request"] != legacy:
                raise ValueError("Old reader cache request differs")
            value = decode_response(cached["response"])
            # Complete evidence-ID validation happens again inside answer_question.
            if (not isinstance(value, dict) or set(value) != {"answer", "used_fact_ids"}
                    or not isinstance(value["answer"], str)):
                raise ValueError("Only an already valid string reader can be reused")
            event = {"stage": stage, "source": "original_valid_reader_cache", **source,
                     "response_sha256": fingerprint(cached["response"]), "new_physical_attempt": False}
            self.events.append(event)
            self.base._log({"stage": stage, "provider": "gemini", "model": model,
                "operation": "generateContent", "request_hash": source["request_hash"],
                "cache_hit": True, "latency_ms": 0, "usage": cached["response"].get("usageMetadata", {}),
                "usage_semantics": "original_valid_response_reuse_no_new_model_usage",
                "original_cache_path": str(path), "original_cache_sha256": source["file_sha256"]})
            return value
        signature = generation_signature(prompt, model=model, system=system, json_output=True,
                                         max_output=max_output, schema=expected[2])
        digest = request_hash(signature)
        key = (str(self.run_dir.resolve()), digest)
        with _LOCK:
            lock = _HASH_LOCKS.setdefault(key, threading.Lock())
        with lock:
            cache_path = self.run_dir / "api_cache" / (digest + ".json")
            intent_path = self.run_dir / "qa_request_intents" / (digest + ".json")
            if intent_path.exists():
                intent = read(intent_path)
                if intent["request"] != signature or intent["policy"] != POLICY:
                    raise ValueError("Recovery intent identity changed")
                if not cache_path.exists():
                    raise RuntimeError("Prior request has no durable response; unknown outcome is not resent")
            else:
                if cache_path.exists():
                    raise RuntimeError("Unattributed recovery cache without durable intent")
                intent_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects independent process owners as well as threads.
                with intent_path.open("x", encoding="utf-8") as handle:
                    json.dump({"policy": POLICY, "request_hash": digest, "request": signature,
                               "created_utc": now(), "pid": os.getpid(), "context": self.base.context}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            existed = cache_path.exists()
            if existed and read(cache_path).get("request") != signature:
                raise ValueError("Recovery cache signature differs")
            try:
                response = self.base.request("gemini", model, signature["body"], stage=stage)
            except Exception as error:
                self.events.append({"stage": stage, "source": "failed_or_unknown_response",
                    "request_hash": digest, "legacy_request_hash": request_hash(legacy),
                    "intent_path": str(intent_path), "response_sha256": None,
                    "new_physical_attempt": not existed,
                    "error": self._redact(str(error))[:1800]})
                raise
            record = read(cache_path)
            if record.get("request") != signature or record.get("response") != response:
                raise ValueError("Recovery response was not durably cached against its request")
            self.events.append({"stage": stage, "source": "recovery_cache" if existed else "new_response",
                                "request_hash": digest, "legacy_request_hash": request_hash(legacy),
                                "response_sha256": fingerprint(response), "cache_sha256": file_sha(cache_path),
                                "cache_path": str(cache_path), "new_physical_attempt": not existed})
            return decode_response(response)


def evaluate_new(q, selected, reference_date, api=None, *, run_dir=None, context=None):
    """Evaluate a newly completed bank without extra retrieval/JEV calls.

    Returns result only; caller owns the immutable QA artifact and bank binding.
    Pass a ResponseSchemaAPI to retain events, or run_dir to construct one.
    """
    if len(selected) != 20 or len({f["id"] for f in selected}) != 20:
        raise ValueError("Recovery uses exactly 20 unique recorded facts")
    if api is None:
        api = ResponseSchemaAPI(ModelAPI(_safe_output(run_dir), retries=0, context=context))
    if not isinstance(api, ResponseSchemaAPI):
        raise TypeError("Explicit structured-output recovery wrapper is required")
    return evaluation.answer_and_judge(api, q["question"], q["reference"], selected, reference_date=reference_date)


def _original_reader(q, selected, reference_date, original):
    prepared = evaluation.prepare_reader_input(q["question"], selected, reference_date=reference_date)
    signature = generation_signature(_json(prepared["payload"]), model=evaluation.READER_MODEL,
        system=evaluation.READER_SYSTEM, json_output=True, max_output=evaluation.MAX_OUTPUT)
    digest = request_hash(signature)
    path = original / "api_cache" / (digest + ".json")
    cached = read(path)
    if cached.get("request") != signature:
        raise ValueError("Original reader cache signature mismatch")
    decoded = decode_response(cached["response"])
    if (set(decoded) != {"answer", "used_fact_ids"} or not isinstance(decoded["answer"], str)
            or not isinstance(decoded["used_fact_ids"], list)
            or len(set(decoded["used_fact_ids"])) != len(decoded["used_fact_ids"])
            or not set(decoded["used_fact_ids"]) <= set(prepared["local_to_actual"])):
        raise ValueError("Original reader is not reusable")
    return {"path": str(path), "file_sha256": file_sha(path), "request_hash": digest, "request": signature}


def collect_existing_failures(original=ORIGINAL_RUN):
    """Read-only admission of exactly 11 reviewed technical failures, no score filter."""
    original = Path(original).resolve()
    data = load_data()
    _, questions = load_cohort(ROOT / "configs/pilot400_manifest_v1.json", data)
    by_id = {q["id"]: q for q in questions}
    seal = read(TERMINAL_SEAL)
    if Path(seal["run_dir"]).resolve() != original:
        raise ValueError("Original terminal seal belongs to another namespace")
    sealed = {item["path"]: item["sha256"] for item in seal["files"]}
    manifest = read(original / "run_manifest.json")
    if manifest["protocol_sha256"] != ORIGINAL_PROTOCOL:
        raise ValueError("Unexpected original pilot protocol")
    found, records = set(), []
    for path in sorted(original.glob("conversations/conv_*/qa/*/*.json")):
        row = _verify_sealed(read(path))
        relative = path.relative_to(original).as_posix()
        if file_sha(path) != sealed.get(relative):
            raise ValueError("Original QA differs from terminal seal")
        if row["status"] != "failed":
            continue
        q, arm, ci = by_id[row["query_id"]], row["arm"], row["conversation"]
        retrieved_path = path.parents[2] / "retrieval" / arm / path.name
        if not retrieved_path.exists():
            continue  # 71 reviewed bank dependencies, handled by separate writer recovery.
        found.add((ci, arm, q["id"]))
        retrieved = _verify_sealed(read(retrieved_path))
        if (retrieved["status"] != "complete" or file_sha(retrieved_path) !=
                sealed.get(retrieved_path.relative_to(original).as_posix())):
            raise ValueError("Original retrieval is not a sealed complete artifact")
        selected = retrieved["selected"]
        if selected != [reader_fact(f) for f in retrieved["pool"][:20]]:
            raise ValueError("Original selected sequence differs from exact saved pool top twenty")
        bank_path = original / row["bank_path"]
        if file_sha(bank_path) != row["bank_sha256"] or row["bank_sha256"] != retrieved["binding"]["bank_sha256"]:
            raise ValueError("Original bank binding differs")
        if (q["conversation"] != ci or retrieved["binding"]["query_id"] != q["id"]
                or retrieved["binding"]["question_sha256"] != fingerprint(q["question"])
                or row["binding"]["reference_sha256"] != fingerprint(q["reference"])):
            raise ValueError("Original question/reference binding differs")
        raw = raw_turns_from_locomo(data[ci]["conversation"])
        reference_date = max(t["observation_date"] for t in raw)
        evaluation.prepare_reader_input(q["question"], selected, reference_date=reference_date)
        reused = _original_reader(q, selected, reference_date, original) if q["id"] == "c3_q197" else None
        records.append({"q": q, "arm": arm, "conversation": ci, "selected": selected,
            "reference_date": reference_date, "original_reader": reused,
            "binding": {"policy": POLICY, "original_qa_path": str(path), "original_qa_sha256": file_sha(path),
                "original_retrieval_path": str(retrieved_path), "original_retrieval_sha256": file_sha(retrieved_path),
                "bank_path": str(bank_path), "bank_sha256": file_sha(bank_path),
                "selected_sha256": fingerprint(selected), "query_id": q["id"], "arm": arm,
                "question_sha256": fingerprint(q["question"]), "reference_sha256": fingerprint(q["reference"]),
                "reference_date": reference_date, "original_reader": reused}})
    if found != EXPECTED:
        raise ValueError("The uniform reviewed 11-failure population changed")
    return records


def recover_existing_qa(output_dir=DEFAULT_OUTPUT, workers=8):
    """One uniform recovery pass over all eleven reviewed A/B/C technical failures."""
    output_dir = _safe_output(output_dir)
    if not 1 <= workers <= 8:
        raise ValueError("Recovery permits 1..8 workers")
    records = collect_existing_failures()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"policy": POLICY, "reader_schema": READER_SCHEMA, "judge_schema": JUDGE_SCHEMA,
            "source_code_sha256": file_sha(__file__), "provider_retries": 0,
            "entries": [record["binding"] for record in records]}
    plan_path = output_dir / "qa_recovery_plan.json"
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError("Recovery plan/code changed; use a new namespace")
    else:
        publish_json(plan_path, plan)
    base = ModelAPI(output_dir, retries=0, context={"architecture": "pilot400_recovery_v1", "driver_stage": "qa"})
    def one(record):
        q, arm = record["q"], record["arm"]
        path = output_dir / "conversations" / ("conv_" + str(record["conversation"])) / "qa" / arm / (q["id"] + ".json")
        if path.exists():
            saved = _verify_sealed(read(path))
            if saved["binding"] != record["binding"]:
                raise ValueError("Recovery QA artifact binding changed")
            return saved
        api = ResponseSchemaAPI(base.scoped(conversation=record["conversation"], query_id=q["id"],
                    method=arm, phase="pilot_recovery_qa"), original_reader=record["original_reader"])
        start = now()
        result, error = None, None
        try:
            result = evaluate_new(q, record["selected"], record["reference_date"], api)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": api._redact(str(exc))[:1800]}
        saved = _sealed({"schema": POLICY, "arm": arm, "query_id": q["id"],
            "conversation": q["conversation"], "category": q["category"], "binding": record["binding"],
            "status": "complete" if result is not None else "failed", "result": result, "error": error,
            "selected_ids": [f["id"] for f in record["selected"]], "api_events": api.events,
            "started_utc": start, "finished_utc": now()})
        if path.exists():
            raise RuntimeError("Concurrent owner created QA artifact")
        publish_json(path, saved)
        return saved
    status_path = output_dir / "qa_recovery_status.json"
    state = {"policy": POLICY, "status": "running", "pid": os.getpid(), "started_utc": now(),
             "expected": len(records), "done": 0, "completed": 0, "failed": 0, "workers": workers}
    publish_json(status_path, state)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, record) for record in records]):
            row = future.result()
            state["done"] += 1
            state["completed" if row["status"] == "complete" else "failed"] += 1
            state["updated_utc"] = now()
            publish_json(status_path, state)
    state.update(status="complete" if state["failed"] == 0 else "requires_review", finished_utc=now())
    publish_json(status_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="Offline validate all original eleven failure inputs; no credentials/API")
    args = parser.parse_args()
    if args.check:
        rows = collect_existing_failures()
        print(json.dumps({"policy": POLICY, "failure_count": len(rows),
            "reusable_readers": sum(row["original_reader"] is not None for row in rows),
            "entries": [{"conversation": row["conversation"], "arm": row["arm"], "query_id": row["q"]["id"]} for row in rows],
            "model_calls": 0, "credentials_loaded": False}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(recover_existing_qa(args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
