"""Single-attempt technical recovery. Original artifacts are read-only.

No ModelAPI is constructed here. Network access is an injected, execute-only
transport, called after a durable exclusive attempt marker. Frozen validators
are exercised, including their original permissive/strict schema boundaries.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from jev_mem import data, evaluation, selection
from jev_mem.providers import ModelAPI

ROOT = Path(__file__).resolve().parents[1]
POLICY = "technical-recovery-network-extension-v1"
NEW_CODE = ("jev_mem/recovery_v1.py", "tools/run_technical_recovery_v1.py",
            "tests/test_technical_recovery_v1.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")).encode())


def request_hash(signature):
    # Exact frozen provider algorithm: spaces are significant.
    return sha(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity(row):
    return (row["architecture"], int(row["conversation"]), row["query_id"], row["method"])


class RecoveryStop(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class FixedResponse:
    def __init__(self, signature, response):
        self.signature, self.response = signature, response

    def request(self, provider, model, body, *, operation="generateContent", **_):
        if {"provider": provider, "model": model, "body": body, "operation": operation} != self.signature:
            raise ValueError("Frozen reconstruction changed request signature")
        return copy.deepcopy(self.response)


def validate_response(signature, stage, response):
    """Exercise provider and evaluation validators, never semantic correctness.

    The other evaluation stage uses a local legal sentinel ONLY inside this
    validation harness. No sentinel becomes a cache, result, or model input.
    """
    proxy = FixedResponse(signature, response)
    body = signature["body"]
    if signature["provider"] == "jev":
        if stage != "jev_score":
            raise ValueError("Recovery only permits frozen QA scoring")
        criteria = {k: v["instructions"] for k, v in body["questions"].items()}
        scores = ModelAPI.score_many(proxy, body["state"], criteria, stage=stage)
        # Provider and selector both check these original Noul fields.
        if set(criteria) not in ({"relevant"}, {"new_evidence"}):
            raise ValueError("Unexpected frozen selector criterion")
        class ScoreResponse:
            def score_many(self, *_):
                return scores
        selection._validated_scores(ScoreResponse(), body["state"], criteria)
        return
    if stage not in ("qa_reader", "strict_judge"):
        raise ValueError("Recovery cannot perform other generation stages")
    prompt = body["contents"][0]["parts"][0]["text"]
    system = body.get("systemInstruction", {}).get("parts", [{}])[0].get("text")
    parsed = ModelAPI.generate(proxy, prompt, model=signature["model"], system=system,
                               json_output=True, max_output=2048, stage=stage)
    payload = json.loads(prompt)

    class ValidationStages:
        def generate(self, _prompt, *, stage, **kwargs):
            if stage == "qa_reader":
                return copy.deepcopy(parsed) if target == "qa_reader" else {
                    "answer": payload["predicted_answer"], "used_fact_ids": []}
            return copy.deepcopy(parsed) if target == "strict_judge" else {
                "correct": False, "completeness": 0, "reason": "local schema sentinel"}

    target = stage
    facts = [{"id": x["id"], "text": x["fact"], "observation_date": x.get("observation_date")}
             for x in payload["facts"]] if stage == "qa_reader" else []
    evaluation.answer_and_judge(ValidationStages(), payload["question"],
                               payload.get("reference_answer", ""), facts,
                               reference_date=payload.get("conversation_reference_date"))


class SealStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.seal_sha256 = sha(self.path.read_bytes())
        self.seal = read_json(self.path)
        if self.seal.get("schema") != "jev-mem-technical-recovery-seal-v1" or self.seal.get("status") != "sealed_after_main_terminal":
            raise ValueError("First-pass terminal seal required")
        for key, expected in (("per_hash_max_new_http_attempts", 1), ("max_consumer_workers", 8),
                              ("frozen_dependency_completion_allowed", True)):
            if self.seal.get("policy", self.seal).get(key) != expected:
                raise ValueError("Recovery policy mismatch: " + key)
        if self.seal["source_dataset_sha256"] != data.SOURCE_HASH:
            raise ValueError("Pinned source hash differs")
        required = {"jev_mem/providers.py", "jev_mem/selection.py", "jev_mem/evaluation.py",
                    "jev_mem/data.py", "tools/run_session_experiments.py", "tools/run_independent_experiments.py"}
        if not required <= self.seal["code_sha256"].keys():
            raise ValueError("Seal missing frozen code")
        for name, expected in self.seal["code_sha256"].items():
            if sha((ROOT / name).read_bytes()) != expected:
                raise ValueError("Frozen source changed: " + name)
        self.runs = {r["architecture"]: r for r in self.seal["runs"]}
        if set(self.runs) != {"mem0", "independent"}:
            raise ValueError("Both original architectures required")
        self.index = {}
        self.cache_locations = defaultdict(list)
        for run in self.seal["runs"]:  # Frozen seal order then POSIX relative path.
            inv = self.path.parent / run["inventory_path"]
            raw = inv.read_bytes()
            if sha(raw) != run["inventory_sha256"]:
                raise ValueError("Seal inventory checksum mismatch")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            idx = {}
            for row in sorted(rows, key=lambda r: r["path"]):
                rel = row["path"]
                if Path(rel).is_absolute() or ".." in Path(rel).parts or rel in idx:
                    raise ValueError("Invalid/duplicate inventory path")
                idx[rel] = row
                if "/api_cache/" in rel and rel.endswith(".json"):
                    self.cache_locations[Path(rel).stem].append((run["architecture"], rel))
            self.index[run["architecture"]] = idx
        failure_path = self.path.parent / self.seal["failures_path"]
        raw = failure_path.read_bytes()
        if sha(raw) != self.seal["failures_sha256"]:
            raise ValueError("Failure inventory changed")
        self.failures = json.loads(raw)
        if len({identity(f) for f in self.failures}) != len(self.failures):
            raise ValueError("Duplicate failed consumer")
        if len(self.failures) != sum(r["failed_evaluations"] for r in self.seal["runs"]):
            raise ValueError("Failure count differs from sealed full denominator")
        self.dataset = data.load_data()
        self.questions = {ci: {q["id"]: q for q in data.queries(item, ci)} for ci, item in enumerate(self.dataset)}
        self.denominators = {}
        for architecture, run in self.runs.items():
            if run["conversations"] != list(range(10)) or run["requested_questions"] != 1540:
                raise ValueError("Seal does not cover fixed 1540 cohort")
            expected_methods = ["sdk20", "rank", "mmr", "jev_single", "conditional"] if architecture == "mem0" else ["rank", "mmr", "jev_single", "conditional"]
            if run["methods"] != expected_methods:
                raise ValueError("Seal methods differ")
            self.denominators[architecture] = {m: 1540 for m in run["methods"]}
            if (run["requested_evaluations"] != 1540 * len(expected_methods)
                    or run["successful_evaluations"] + run["failed_evaluations"] != run["requested_evaluations"]):
                raise ValueError("Full evaluation denominator mismatch")
        self.logs = defaultdict(lambda: defaultdict(list))
        failed_keys = {identity(f) for f in self.failures}
        for failure in self.failures:
            arch, ci, qid, method = identity(failure)
            if qid not in self.questions[ci] or method not in self.runs[arch]["methods"]:
                raise ValueError("Failure outside fixed cohort")
            if Path(failure["run_dir"]).resolve() != Path(self.runs[arch]["run_dir"]).resolve():
                raise ValueError("Failure run differs")
            if f"conv_{ci}/qa/{method}/{qid}.json" in self.index[arch]:
                raise ValueError("A first-pass success cannot be retried")
        for arch, run in self.runs.items():
            expected = {(arch, ci, qid, method) for ci in run["conversations"]
                        for qid in self.questions[ci] for method in run["methods"]}
            successes = set()
            for relative in self.index[arch]:
                parts = relative.split("/")
                if len(parts) == 4 and parts[0].startswith("conv_") and parts[1] == "qa" and parts[3].endswith(".json"):
                    key = (arch, int(parts[0][5:]), parts[3][:-5], parts[2])
                    if key not in expected:
                        raise ValueError("Unexpected QA artifact outside fixed cohort")
                    successes.add(key)
            if ({key for key in failed_keys if key[0] == arch} != expected - successes
                    or len(successes) != run["successful_evaluations"]):
                raise ValueError("Seal failures must exactly equal full cohort minus first-pass successes")
        for arch, run in self.runs.items():
            for ci in run["conversations"]:
                raw = self.read_bytes(arch, f"conv_{ci}/api_calls.jsonl")
                for line_no, line in enumerate(raw.splitlines(), 1):
                    row = json.loads(line)
                    ctx = row.get("context", {})
                    key = (arch, ctx.get("conversation"), ctx.get("query_id"), ctx.get("method"))
                    if key in failed_keys:
                        self.logs[key][row["request_hash"]].append({**row, "source_line": line_no})

    def read_bytes(self, architecture, relative):
        record = self.index[architecture].get(relative)
        if record is None:
            raise ValueError("File absent from seal: " + relative)
        raw = (Path(self.runs[architecture]["run_dir"]) / relative).read_bytes()
        if len(raw) != record["bytes"] or sha(raw) != record["sha256"]:
            raise ValueError("Original artifact changed: " + relative)
        return raw

    def read(self, architecture, relative):
        return json.loads(self.read_bytes(architecture, relative))

    def cached(self, signature, stage):
        h = request_hash(signature)
        sources, valid = [], None
        for arch, path in self.cache_locations.get(h, []):
            record = self.read(arch, path)
            if record.get("request") != signature or request_hash(record["request"]) != h:
                raise ValueError("Sealed cached request differs from frozen reconstruction")
            source = {"architecture": arch, "path": path,
                      "file_sha256": self.index[arch][path]["sha256"],
                      "response_sha256": digest(record.get("response"))}
            try:
                validate_response(signature, stage, record["response"])
                source["validation"] = "valid"
                if valid is None:
                    valid = (record["response"], len(sources))
            except Exception as error:
                source.update(validation="invalid", error=str(error)[:600])
            sources.append(source)
        return valid, sources

    def inputs(self, failure):
        arch, ci, qid, _ = identity(failure)
        q = self.questions[ci][qid]
        base = f"conv_{ci}/"
        summary = self.read(arch, base + "ingestion_summary.json")
        if not summary.get("complete"):
            raise ValueError("Full ingestion required")
        facts_raw = self.read_bytes(arch, base + "facts.json")
        retrieval = self.read(arch, base + f"retrieval/{qid}.json")
        # Import only pure binding helpers; constructors and ensure_run are not used.
        if arch == "mem0":
            from tools.run_session_experiments import _verify_retrieval
            _verify_retrieval(retrieval, q, summary, sha(facts_raw))
        else:
            from tools.run_independent_experiments import _verify_retrieval, _retrieval_binding
            manifest = self.read(arch, "run_manifest.json")
            if manifest["mode"] != "full" or summary["facts_sha256"] != digest(json.loads(facts_raw)):
                raise ValueError("Independent fact bank binding mismatch")
            bank = self.read(arch, base + "independent_store/fact_bank.json")
            if digest(bank["facts"]) != summary["facts_sha256"]:
                raise ValueError("Independent stored fact bank differs")
            _verify_retrieval(retrieval, _retrieval_binding(q, summary, manifest))
        if not isinstance(retrieval.get("pool60"), list) or len(retrieval["pool60"]) > 60:
            raise ValueError("Frozen candidate budget exceeded")
        envelopes = list(data.session_envelopes(self.dataset[ci]))
        if arch == "independent" and set(summary["session_ids"]) != {x["session_id"] for x in envelopes}:
            raise ValueError("Unseen or incomplete session date scope")
        reference_date = envelopes[-1]["observation_date"]
        return q, retrieval, reference_date, {"facts_file_sha256": sha(facts_raw),
                  "retrieval_sha256": digest(retrieval), "reference_date": reference_date}


class AttemptStore:
    def __init__(self, output, seal_sha256, transport=None):
        self.output, self.seal_sha256, self.transport = Path(output), seal_sha256, transport
        self.lock = threading.Lock()
        self.locks = {}

    def _lock(self, h):
        with self.lock:
            return self.locks.setdefault(h, threading.Lock())

    def obtain(self, signature, stage, proof, *, execute):
        h = request_hash(signature)
        directory = self.output / "attempts" / h
        with self._lock(h):
            planned = {"schema": "jev-mem-recovery-request-v1", "seal_sha256": self.seal_sha256,
                       "request_hash": h, "request": signature, "stage": stage, "proof": proof,
                       "policy": POLICY, "max_new_http_attempts": 1}
            path = directory / "planned.json"
            try:
                exclusive(path, planned)
            except FileExistsError:
                previous = read_json(path)
                if previous["request"] != signature or previous["seal_sha256"] != self.seal_sha256:
                    raise ValueError("Attempt ledger binding differs")
            response_file = directory / "response.json"
            if response_file.exists():
                response = read_json(response_file)
                if response.get("request") != signature or response.get("seal_sha256") != self.seal_sha256:
                    raise ValueError("Overlay response binding differs")
                validate_response(signature, stage, response["response"])
                return response["response"], "overlay_response"
            # ModelAPI writes raw response before returning. Salvage only if an
            # attempt marker exists; this does not make another HTTP request.
            started = directory / "attempt_started.json"
            transport_cache = self.output / "transport/api_cache" / (h + ".json")
            if started.exists() and transport_cache.exists():
                raw = read_json(transport_cache)
                if raw.get("request") != signature:
                    raise ValueError("Transport response has wrong signature")
                exclusive(response_file, {**raw, "seal_sha256": self.seal_sha256,
                          "request_hash": h, "persisted_utc": utc(), "salvaged": True})
                validate_response(signature, stage, raw["response"])
                return raw["response"], "overlay_response"
            if started.exists():
                outcome = read_json(directory / "outcome.json") if (directory / "outcome.json").exists() else {}
                status = "failed" if outcome.get("status") == "failed" else "unknown"
                raise RecoveryStop(status, "An attempt already started; no second HTTP permitted: " + h)
            if not execute:
                raise RecoveryStop("blocked", "Missing legal cached response: " + h)
            if self.transport is None:
                raise ValueError("Execute transport was not configured")
            try:
                exclusive(started, {"request_hash": h, "seal_sha256": self.seal_sha256,
                          "started_utc": utc(), "pid": os.getpid(), "proof": proof})
            except FileExistsError:
                raise RecoveryStop("unknown", "Another process owns this attempt; do not resend")
            try:
                response = self.transport(signature, stage, proof)
                exclusive(response_file, {"request_hash": h, "request": signature,
                          "response": response, "stage": stage, "seal_sha256": self.seal_sha256,
                          "persisted_utc": utc()})
            except Exception as error:
                status = "failed" if isinstance(error, RecoveryStop) and error.status == "failed" else "unknown"
                publish(directory / "outcome.json", {"status": status, "error": str(error)[:1800],
                        "finished_utc": utc(), "request_hash": h})
                raise RecoveryStop(status, str(error)) from error
            try:
                validate_response(signature, stage, response)
            except Exception as error:
                publish(directory / "outcome.json", {"status": "failed", "error": str(error)[:1800],
                        "finished_utc": utc(), "request_hash": h})
                raise
            publish(directory / "outcome.json", {"status": "complete", "request_hash": h,
                    "finished_utc": utc()})
            return response, "new_http"


class ReplayAPI:
    generate = ModelAPI.generate
    score_many = ModelAPI.score_many

    def __init__(self, store, attempts, failure, execute=False):
        self.store, self.attempts, self.failure, self.execute = store, attempts, failure, execute
        self.key = identity(failure)
        self.events = []
        self.lock = threading.Lock()
        self.counts = Counter()

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", **_):
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        h = request_hash(signature)
        with self.lock:
            self.counts[stage] += 1
            limits = {"jev_score": 124 if self.failure["method"] == "conditional" else 60,
                      "qa_reader": 1, "strict_judge": 1}
            if stage not in limits or self.counts[stage] > limits[stage]:
                raise ValueError("Frozen request bound exceeded")
            predecessors = sorted({e["request_hash"] for e in self.events if e.get("status") == "valid"})
        old_logs = self.store.logs[self.key]
        matching = [r for r in old_logs.get(h, []) if r.get("stage") == stage
                    and r.get("provider") == provider and r.get("model") == model
                    and r.get("operation") == operation]
        if h in old_logs and not matching:
            raise RecoveryStop("unbound", "Original request hash context/stage mismatch")
        # Reader/judge each have one original signature. A different signature
        # must not be laundered through a cache belonging to another consumer.
        if stage in ("qa_reader", "strict_judge") and not matching and any(
                row.get("stage") == stage for entries in old_logs.values() for row in entries):
            raise RecoveryStop("unbound", "Original single-generation stage exists with a different request hash")
        valid, sources = self.store.cached(signature, stage)
        event = {"request_hash": h, "stage": stage, "sources": sources,
                 "original_log_lines": [r["source_line"] for r in matching]}
        try:
            if valid is not None:
                response, chosen = valid
                event.update(kind="cache_completion", source="sealed_cache", chosen_source=chosen)
            else:
                network = any(not r.get("cache_hit") and (r.get("status") == "transport_error"
                              or isinstance(r.get("status"), int) and r["status"] >= 400) for r in matching)
                if matching and sources:
                    kind = "same_request_recovery"
                    basis = "original_stage_format_invalid"
                elif network:
                    kind = "same_request_recovery"
                    basis = "original_network_failure"
                elif not matching and (predecessors or any(
                        not r.get("cache_hit") and (r.get("status") == "transport_error" or
                        isinstance(r.get("status"), int) and r["status"] >= 400)
                        for entries in old_logs.values() for r in entries)):
                    kind = "first_time_dependency_completion"
                    basis = "frozen_algorithm_reached_previously_unlogged_step"
                else:
                    raise RecoveryStop("unbound", "Cannot bind missing request to original failure/dependency")
                proof = {"consumer": dict(zip(("architecture", "conversation", "query_id", "method"), self.key)),
                         "kind": kind, "basis": basis, "original_log_lines": event["original_log_lines"],
                         "predecessor_request_hashes": predecessors, "sources": sources,
                         "frozen_stage_count": dict(self.counts)}
                event.update(kind=kind, basis=basis)
                response, source = self.attempts.obtain(signature, stage, proof, execute=self.execute)
                event["source"] = source
            event["status"] = "valid"
            event["response_sha256"] = digest(response)
            return copy.deepcopy(response)
        except Exception as error:
            event.update(status=getattr(error, "status", "failed"), error=str(error)[:1800])
            raise
        finally:
            with self.lock:
                self.events.append(event)


def replay_consumer(store, attempts, failure, *, execute=False):
    start = time.perf_counter()
    api = ReplayAPI(store, attempts, failure, execute)
    row = {"schema": "jev-mem-technical-recovery-result-v1", "seal_sha256": store.seal_sha256,
           "policy": POLICY, "original_failure": failure,
           **dict(zip(("architecture", "conversation", "query_id", "method"), identity(failure)))}
    try:
        q, retrieval, reference_date, bindings = store.inputs(failure)
        row.update(question=q["question"], reference=q["reference"], category=q["category"], input_bindings=bindings)
        if failure["method"] == "sdk20":
            selected, trace = retrieval["sdk20"], {"method": "sdk20"}
        else:
            result = selection.select(q["question"], retrieval["pool60"], api, method=failure["method"],
                      top_k=20, seed_count=8, round_sizes=(6, 6), shortlist_size=32,
                      conditional_threshold=.5, mmr_weight=.7, max_workers=8)
            selected, trace = result.selected, result.trace
        row.update(selected_ids=[f["id"] for f in selected], selection_trace=trace)
        row.update(evaluation.answer_and_judge(api, q["question"], q["reference"], selected,
                                            reference_date=reference_date))
        row["status"] = "complete"
    except Exception as error:
        row.update(status=getattr(error, "status", "failed"), error_type=type(error).__name__, error=str(error)[:1800])
    row["request_events"] = sorted(api.events, key=lambda x: (x["stage"], x["request_hash"]))
    row["recovery_elapsed_seconds"] = time.perf_counter() - start
    row["finished_utc"] = utc()
    row["row_sha256"] = digest(row)
    return row


def code_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in NEW_CODE}


def verify_review(path, store, output):
    review = read_json(path)
    plan_path = Path(output) / "plan/plan.json"
    plan = read_json(plan_path)
    if (review.get("schema") != "jev-mem-technical-recovery-review-v1" or review.get("approved") is not True
            or review.get("seal_sha256") != store.seal_sha256
            or review.get("plan_sha256") != sha(plan_path.read_bytes())
            or review.get("recovery_code_sha256") != code_hashes()
            or plan.get("seal_sha256") != store.seal_sha256
            or plan.get("recovery_code_sha256") != code_hashes()
            or plan.get("output_dir") != str(Path(output).resolve())
            or plan.get("consumers") != [list(identity(f)) for f in store.failures]):
        raise ValueError("Execute requires reviewed exact seal/plan/current recovery code")


class NamespaceLock:
    """OS-released lock prevents two supervisors exceeding eight consumers."""
    def __init__(self, output):
        self.path = Path(output) / "supervisor.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another recovery supervisor owns this namespace") from None
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def run(store, output, mode, workers=8, *, transport=None):
    if mode not in ("plan", "cache-only", "execute") or not 1 <= workers <= 8:
        raise ValueError("Mode and consumer workers must satisfy fixed policy")
    output = Path(output).resolve()
    for old in [store.path.parent, *(Path(r["run_dir"]).resolve() for r in store.runs.values())]:
        if output == old or output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError("Recovery namespace must not contain/overwrite original artifacts")
    output.mkdir(parents=True, exist_ok=True)
    with NamespaceLock(output):
        return _run_locked(store, output, mode, workers, transport=transport)


def _run_locked(store, output, mode, workers, *, transport):
    binding = {"seal_sha256": store.seal_sha256, "policy": POLICY, "denominators": store.denominators}
    bind_path = output / "binding.json"
    if bind_path.exists():
        if read_json(bind_path) != binding:
            raise ValueError("Recovery namespace already bound to another seal")
    else:
        exclusive(bind_path, binding)
    folder = output / ("cache_only" if mode == "cache-only" else mode)
    if mode == "plan" and (folder / "plan.json").exists():
        raise ValueError("Plan already sealed; do not overwrite reviewed plan")
    attempts = AttemptStore(output, store.seal_sha256, transport)
    lock = threading.Lock()
    state = {"schema": "jev-mem-technical-recovery-status-v1", **binding, "mode": mode,
             "status": "running", "pid": os.getpid(), "started_utc": utc(), "updated_utc": utc(),
             "expected_consumers": len(store.failures), "done": 0, "by_status": {},
             "api_log_path": str(output / "transport/api_calls.jsonl")}
    rows = []

    def status():
        dirs = list((output / "attempts").glob("*")) if (output / "attempts").exists() else []
        outcomes = [read_json(p / "outcome.json") for p in dirs if (p / "outcome.json").exists()]
        unresolved = sum((p / "attempt_started.json").exists() and not (p / "response.json").exists()
                         and not (p / "outcome.json").exists() for p in dirs)
        state.update(physical_attempts_started=sum((p / "attempt_started.json").exists() for p in dirs),
                     raw_responses_persisted=sum((p / "response.json").exists() for p in dirs),
                     started_without_response=sum((p / "attempt_started.json").exists() and not (p / "response.json").exists() for p in dirs),
                     known_failed_attempts=sum(x.get("status") == "failed" for x in outcomes),
                     unknown_attempts=sum(x.get("status") == "unknown" for x in outcomes) + (unresolved if state["status"] != "running" else 0),
                     inflight_or_crash_unknown=unresolved if state["status"] == "running" else 0,
                     updated_utc=utc())
        publish(folder / "status.json", state)

    def consume(failure):
        arch, ci, qid, method = identity(failure)
        target = folder / "consumers" / arch / f"conv_{ci}" / method / (qid + ".json")
        if target.exists():
            row = read_json(target)
            if (row.get("seal_sha256") != store.seal_sha256 or identity(row) != identity(failure)
                    or row.get("row_sha256") != digest({k: v for k, v in row.items() if k != "row_sha256"})):
                raise ValueError("Existing recovery result changed")
            # A completed/failed/unknown execution is immutable. No second pass.
        else:
            row = replay_consumer(store, attempts, failure, execute=mode == "execute")
            publish(target, row)
        with lock:
            rows.append(row)
            state["done"] = len(rows)
            state["by_status"] = dict(Counter(r["status"] for r in rows))
            status()
        return row

    status()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(consume, store.failures))
        state["status"] = "complete" if all(r["status"] == "complete" for r in rows) else "requires_review"
        if mode == "plan":
            plan = {"schema": "jev-mem-technical-recovery-plan-v1", **binding,
                    "output_dir": str(output),
                    "recovery_code_sha256": code_hashes(), "created_utc": utc(),
                    "consumers": [list(identity(f)) for f in store.failures],
                    "by_status": state["by_status"],
                    "frontier_requests": sorted(p.parent.name for p in (output / "attempts").glob("*/planned.json")),
                    "dynamic_dependencies": "Only frozen 60+32+32 selector and one reader/one judge per consumer",
                    "max_consumer_workers": 8, "per_hash_max_new_http_attempts": 1}
            exclusive(folder / "plan.json", plan)
    except BaseException as error:
        state.update(status="failed", error=str(error)[:1800])
        raise
    finally:
        state["finished_utc"] = utc()
        status()
    return state
