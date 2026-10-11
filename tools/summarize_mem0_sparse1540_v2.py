"""Offline v2 explicit-map QA/raw-response audit and fixed1540 metrics.

The immutable v1 metric/cost helpers are reused; QA replay is scoped to the new
policy without modifying any imported module globals. This never opens stores,
constructs provider clients, invokes APIs, or writes to any experiment run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_sparse1540 as runner
from tools import run_mem0_sparse1540_recovered_qa_v2 as qa
from tools import summarize_mem0_sparse1540 as core
from tools import summarize_pilot400 as metric
from jev_mem import mem0_transfer_qa as transport
from jev_mem import evaluation

SCHEMA = "mem0-native-sparse1540-recovered-summary-v2"
CORE_SHA256 = "406353916f0daecbb91402b95b8c1cf1acfebac8ea8baaa251fca62821dac98f"
ARMS, METRICS = runner.ARMS, core.METRICS
Snapshot, ledger_rows, sealed = core.Snapshot, core.ledger_rows, core.sealed
DeclaredProviderFailure, ReplayAuditError = core.DeclaredProviderFailure, core.ReplayAuditError

def scoped_context(ci, arm, qid):
    return {"architecture": qa.POLICY, "conversation": ci, "phase": "qa", "query_id": qid, "method": arm}


class ReplayQA:
    """The evaluator calls this pure file-only implementation, never the wrapper."""
    def __init__(self, folder, context, events, ledger, snapshot, expected_ids):
        self.folder, self.context, self.events = Path(folder).resolve(), context, events
        self.ledger, self.snapshot, self.expected_ids = ledger, snapshot, set(expected_ids)
        self.used = 0
        self.event_paths = []

    def generate(self, *args, **kwargs):
        try:
            return self._generate(*args, **kwargs)
        except DeclaredProviderFailure:
            raise
        except Exception as exc:
            raise ReplayAuditError(str(exc)) from exc

    def _generate(self, prompt, *, model="gemini-2.5-flash", system=None, json_output=False,
                 max_output=8192, stage="generation"):
        settings = {"qa_reader": ("gemini-2.5-flash", evaluation.READER_SYSTEM, transport.READER_SCHEMA),
                    "strict_judge": ("gemini-3.5-flash-lite", evaluation.JUDGE_SYSTEM, transport.JUDGE_SCHEMA)}
        if stage not in settings or (model, system, json_output, max_output) != (*settings[stage][:2], True, 2048):
            raise ValueError("Unexpected frozen reader/judge settings")
        if self.used >= len(self.events):
            raise ValueError("Missing durable consumer event")
        event = self.events[self.used]
        self.used += 1
        signature = transport.generation_signature(prompt, model=model, system=system, json_output=True,
                                                     max_output=2048, schema=settings[stage][2])
        digest = transport.request_hash(signature)
        legacy = transport.request_hash(transport.generation_signature(prompt, model=model, system=system,
                                                                        json_output=True, max_output=2048))
        cache_path = self.folder / "api_cache" / (digest + ".json")
        intent_path = self.folder / "mem0_transfer_qa_intents" / (digest + ".json")
        expected = {"protocol": transport.PROTOCOL, "context": self.context, "stage": stage,
                    "provider": "gemini", "model": model, "operation": "generateContent", "request_hash": digest,
                    "legacy_request_hash": legacy, "provider_retry_budget": 2, "max_http_attempts": 3,
                    "cache_path": str(cache_path), "intent_path": str(intent_path)}
        if any(event.get(k) != v for k, v in expected.items()):
            raise ValueError("Consumer event identity/request/settings differ")
        if not isinstance(event.get("event_id"), str) or len(event["event_id"]) != 32:
            raise ValueError("Invalid durable event ID")
        event_path = self.folder / "mem0_transfer_qa_events" / runner.fingerprint(self.context) / (event["event_id"] + ".json")
        if Path(event.get("event_path", "")).resolve() != event_path or self.snapshot.read(event_path) != event:
            raise ValueError("Consumer event pointer/body differs from durable event")
        self.event_paths.append(event_path)
        intent = self.snapshot.read(intent_path)
        intent_expected = {"protocol": transport.PROTOCOL, "stage": stage, "provider": "gemini", "model": model,
                           "operation": "generateContent", "request_hash": digest,
                           "body_sha256": runner.fingerprint(signature["body"]), "provider_retries": 2, "max_http_attempts": 3}
        if any(intent.get(k) != v for k, v in intent_expected.items()):
            raise ValueError("Durable intent differs from exact structured request")
        creator = intent.get("context", {})
        if (creator.get("architecture") != qa.POLICY or creator.get("conversation") != self.context["conversation"]
                or creator.get("phase") != "qa" or creator.get("method") not in ARMS
                or creator.get("query_id") not in self.expected_ids):
            raise ValueError("Intent creator is outside the declared cohort/scope")
        request_rows = [r for r in self.ledger if r.get("request_hash") == digest]
        matched = [r for r in request_rows if r.get("context") == self.context]
        physical = [r for r in request_rows if r.get("cache_hit") is not True]
        cache_events = [r for r in request_rows if r.get("cache_hit") is True]
        own_cache = [r for r in cache_events if r.get("context") == self.context]
        for r in request_rows:
            if any(r.get(k) != v for k, v in {"stage": stage, "provider": "gemini", "model": model,
                                               "operation": "generateContent"}.items()):
                raise ValueError("Request ledger provider/model/stage differs")
            if "cache_hit" in r and type(r["cache_hit"]) is not bool:
                raise ValueError("Request ledger cache flag is not boolean")
        for r in cache_events:
            # Frozen ModelAPI emits no HTTP status/attempt for a cache read.
            if ("status" in r or "attempt" in r or r.get("latency_ms") != 0
                    or r.get("usage_semantics") != "reused_response_no_new_model_usage"
                    or r.get("error") or r.get("usage_unknown")):
                raise ValueError("Malformed cache consumer cannot stand in for an HTTP success")
        if (len(physical) > 3 or any(type(r.get("attempt")) is not int or not 1 <= r["attempt"] <= 3 for r in physical)
                or sorted(r["attempt"] for r in physical) != list(range(1, len(physical) + 1))
                or any(r.get("context") != creator for r in physical)):
            raise ValueError("Physical requests exceed frozen retry budget")
        successful = [r for r in physical if type(r.get("status")) is int and 200 <= r["status"] < 300]
        failures = [r for r in physical if r not in successful]
        for r in failures:
            status = r.get("status")
            if (not (type(status) is int and 400 <= status < 600 or status == "transport_error")
                    or not isinstance(r.get("error"), str) or not r["error"]
                    or status == "transport_error" and r.get("usage_unknown") is not True):
                raise ValueError("Physical failure lacks HTTP/transport error evidence")
        if successful and (len(successful) != 1 or successful[0]["attempt"] != len(physical)
                           or successful[0].get("error") or successful[0].get("usage_unknown")):
            raise ValueError("Raw response requires a unique final successful physical producer")
        cache = None
        if cache_path.exists():
            cache = self.snapshot.read(cache_path)
            if (cache.get("request") != signature or cache.get("stage") != stage
                    or event.get("cache_sha256") != self.snapshot.sha(cache_path)):
                raise ValueError("Raw HTTP cache signature/stage/bytes differ")
            if event.get("response_sha256") is not None and event["response_sha256"] != runner.fingerprint(cache["response"]):
                raise ValueError("Raw HTTP response hash differs")
        elif event.get("cache_sha256") is not None:
            raise ValueError("Event claims an absent raw response")
        if cache is not None:
            if not successful:
                raise ValueError("Raw response has no successful physical producer")
            response = cache["response"]
            usage = response.get("usageMetadata", response.get("usage", {}))
            resolved_model = response.get("modelVersion", response.get("model"))
            for r in successful + cache_events:
                if r.get("usage") != usage or r.get("resolved_model") != resolved_model:
                    raise ValueError("Ledger usage/model differs from the raw HTTP response")
        source, dispatch = event.get("cache_source"), event.get("new_logical_dispatch")
        if event.get("error_type") is not None:
            if not isinstance(event["error_type"], str) or not event["error_type"]:
                raise ValueError("Invalid declared provider failure")
            if source in {"failed_or_unknown_response", "prior_unknown_no_redispatch"}:
                if (cache is not None or successful or not failures or cache_events or event.get("response_sha256") is not None
                        or event["error_type"] != "RuntimeError"
                        or dispatch is not (source == "failed_or_unknown_response")
                        or source == "failed_or_unknown_response" and creator != self.context
                        or source == "prior_unknown_no_redispatch" and matched):
                    raise ValueError("Failed/unknown consumer lacks attributable failed physical evidence")
            elif source in {"new_http_invalid_response", "invalid_existing_cache"}:
                if (cache is None or not successful or own_cache
                        or dispatch is not (source == "new_http_invalid_response")
                        or source == "new_http_invalid_response" and creator != self.context
                        or source == "new_http_invalid_response" and event.get("response_sha256") != runner.fingerprint(cache["response"])
                        or source == "invalid_existing_cache" and event.get("response_sha256") is not None):
                    raise ValueError("Invalid-response failure lacks its exact dispatch/raw provenance")
                try:
                    transport._validate_decoded(transport.decode_response(cache["response"]), stage)
                except Exception as exc:
                    if type(exc).__name__ != event["error_type"]:
                        raise ValueError("Raw response replays a different validation failure") from exc
                else:
                    raise ValueError("Declared invalid raw response replays successfully")
            else:
                raise ValueError("Unproven consumer failure source requires separate review")
            raise DeclaredProviderFailure(event["error_type"])
        if (cache is None or not successful or event.get("response_sha256") != runner.fingerprint(cache["response"])
                or source not in {"new_http", "validated_cache"} or dispatch is not (source == "new_http")
                or source == "new_http" and (creator != self.context or own_cache)
                or source == "validated_cache" and len(own_cache) != 1):
            raise ValueError("Successful consumer lacks valid raw response/attributed successful attempt")
        return transport._validate_decoded(transport.decode_response(cache["response"]), stage)


def validate_qa(row, q, arm, selected, binding, qa_folder, ledger, snapshot, official, expected_ids):
    sealed(row)
    identity = {"schema": qa.POLICY + "-qa", "query_id": q["id"], "conversation": q["conversation"],
                "category": q["category"], "arm": arm, "binding": binding,
                "selected_ids": [f["id"] for f in selected]}
    if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in identity.items()):
        raise ValueError("QA identity/input binding differs")
    if (len(selected) != 20 or len({f["id"] for f in selected}) != 20
            or len({" ".join(f["text"].split()) for f in selected}) != 20):
        raise ValueError("Both arms require twenty unique selected facts")
    events = row.get("api_events")
    if not isinstance(events, list) or not 1 <= len(events) <= 2:
        raise ValueError("QA requires one or two durable consumer events")
    replay = ReplayQA(qa_folder, scoped_context(q["conversation"], arm, q["id"]), events, ledger, snapshot, expected_ids)
    declared_requests = {(e.get("stage"), e.get("request_hash")) for e in events}
    if any((r.get("stage"), r.get("request_hash")) not in declared_requests for r in ledger if r.get("context") == replay.context):
        raise ValueError("QA has physical/cache events outside its durable consumer requests")
    status = row.get("status")
    if status not in {"complete", "failed"}:
        raise ValueError("QA is not terminal")
    if status == "failed":
        err = row.get("error")
        if row.get("result") is not None or not isinstance(err, dict) or any(not isinstance(err.get(k), str) or not err[k] for k in ("type", "message")):
            raise ValueError("Failed QA must retain an error and a null result")
    elif row.get("error") is not None:
        raise ValueError("Complete QA carries an error")
    try:
        value = evaluation.answer_and_judge(replay, q["question"], q["reference"], selected,
                                             reference_date=binding["reference_date"])
    except DeclaredProviderFailure as exc:
        if status != "failed" or row["error"]["type"] != str(exc):
            raise ValueError("Consumer failure differs from terminal QA") from exc
        value = None
    except ReplayAuditError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        # This comes from the unchanged evaluator after the cache/event checks,
        # e.g. a structurally valid reader returning an unknown local evidence ID.
        if status != "failed" or row["error"]["type"] != type(exc).__name__:
            raise ValueError("Original evaluator rejection differs from terminal QA") from exc
        value = None
    if replay.used != len(events):
        raise ValueError("QA retains extra unconsumed events")
    if status == "failed":
        if value is not None:
            raise ValueError("Declared failure replays as complete; requires review")
        return None, replay.event_paths
    if value != row.get("result"):
        raise ValueError("Result/metrics differ from exact raw reader/judge replay")
    scored = {"query_id": q["id"], "conversation": q["conversation"], "category": q["category"],
              "official_locomo_f1": official(value["prediction"]["answer"], q["reference"], q["category"]),
              "custom_token_f1": value["token_f1"], "judge_correct": float(value["judgment"]["correct"]),
              "judge_completeness": value["judgment"]["completeness"]}
    for key in METRICS:
        if type(scored[key]) not in (int, float) or not math.isfinite(scored[key]) or not 0 <= scored[key] <= 1:
            raise ValueError("Invalid offline score")
    return scored, replay.event_paths


def bind_pointer(snapshot, pointer):
    if snapshot.sha(pointer["path"]) != pointer["sha256"]:
        raise ValueError("Signed audit/recovery pointer changed")
    return snapshot.read(pointer["path"])


def recovery_usage(snapshot, entry):
    """Copied source ledger bytes are inherited work, never newly billed work."""
    recovery = bind_pointer(snapshot, entry["index_recovery"])
    path = Path(entry["native_ledger_path"])
    if not path.exists():
        return core.measure_usage(None)
    raw = snapshot.raw(path)
    prefix = recovery.get("inherited_ledger_prefix", recovery.get("source_ledger"))
    if prefix is not None:
        original = snapshot.raw(prefix["path"])
        if (snapshot.sha(prefix["path"]) != prefix["sha256"] or len(original) != prefix["bytes"]
                or not raw.startswith(original) or original and not original.endswith(b"\n")):
            raise ValueError("Recovered ledger does not retain the exact declared source prefix")
        fresh = raw[len(original):]
    elif recovery.get("inherited_ledger_bytes") == 0 and type(recovery.get("inherited_ledger_bytes")) is int:
        fresh = raw
    else:
        return {**core.measure_usage(None), "reason": "Undeclared inherited ledger prefix; cannot attribute new cost"}
    rows = [json.loads(line) for line in fresh.decode("utf-8-sig").splitlines() if line.strip()]
    for event in rows:
        ctx = event.get("context", {})
        if (ctx.get("architecture") != qa.POLICY or ctx.get("conversation") != entry["conversation"]
                or ctx.get("method") != entry["arm"] or ctx.get("phase") != "retrieve"):
            raise ValueError("Recovery appended ledger contains undeclared native calls")
    return {**core.measure_usage(rows), "source_ledger_path": str(path), "source_ledger_sha256": snapshot.sha(path),
            "inherited_prefix_bytes": len(raw) - len(fresh),
            "stages": {stage: core.measure_usage([x for x in rows if x["stage"] == stage])
                       for stage in sorted({x["stage"] for x in rows})}}


def usage_report(snapshot, manifest):
    inherited = core.costs(snapshot)
    rows, sources, missing = [], [], []
    for ci in range(10):
        path = qa.conv_path(ci) / "qa_api/api_calls.jsonl"
        events = ledger_rows(snapshot, path)
        if events is None:
            missing.append(str(path)); continue
        sources.append({"path": str(path), "sha256": snapshot.sha(path)})
        rows.extend(events)
    new_qa = core.measure_usage(rows)
    new_qa.update(source_ledgers=sources, missing_ledgers=missing, scope_complete=not missing,
        stages={stage: core.measure_usage([x for x in rows if x["stage"] == stage])
                for stage in sorted({x["stage"] for x in rows})})
    if missing:
        new_qa["total_cost_usd"] = None
    recovered = [{"conversation": row["conversation"], "arm": row["arm"],
                  **recovery_usage(snapshot, row)} for row in manifest["rows"] if row["index_recovery"] is not None]
    return {"new_v2_qa": new_qa, "new_v2_native_retrieval_after_index_recovery": recovered,
        "original_v1_transfer_work": {p: inherited["new_namespace"]["phases"][p] for p in ("write", "retrieve")},
        "inherited_baseline_native_write_and_retrieval": inherited["inherited_baseline_native_write_and_retrieval"],
        "inherited_sparse_grouping_work": inherited["inherited_sparse_grouping_work"],
        "prices_usd_per_million": inherited["prices_usd_per_million"],
        "interpretation": "Disjoint provenance scopes, not a combined all-in cold-cost estimate. Original v1 native attempts remain original work even if a stage failed. Copied ledger prefixes and cache responses are not newly billed. New physical attempts include technical retries and recorded thought tokens. Unpriced/missing/unknown usage stays unknown. HTTP latency sums are not wall-clock duration."}


def summarize(*, bootstrap_samples=2000):
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("Positive integer bootstrap count required")
    if runner.sha(core.__file__) != CORE_SHA256:
        raise ValueError("Reviewed v1 pure helper implementation changed")
    manifest, digest, loaded = qa.verify_inputs()
    snapshot = Snapshot()
    if snapshot.sha(qa.OUT / "input_manifest.json") != digest:
        raise ValueError("Input manifest changed after admission")
    source = bind_pointer(snapshot, manifest["source_manifest"])
    bind_pointer(snapshot, manifest["recovery_policy"])
    admission = bind_pointer(snapshot, manifest["input_audit"])
    for path, expected in admission["audited_files"].items():
        if snapshot.sha(path) != expected:
            raise ValueError("Independent input audit evidence changed")
    # These inputs are explicit immutable files, never relocated v1 module globals.
    for row in manifest["rows"]:
        if snapshot.sha(row["bank_path"]) != row["bank_sha256"]:
            raise ValueError("Mapped bank changed after admission")
        for qid, expected in row["retrieval_files"].items():
            if snapshot.sha(Path(row["retrieval_dir"]) / (qid + ".json")) != expected:
                raise ValueError("Mapped retrieval changed after admission")
        if row["index_recovery"] is not None:
            bind_pointer(snapshot, row["index_recovery"])
    for path, expected in source["source_files"].items():
        if snapshot.sha(path) != expected:
            raise ValueError("Original inherited source bytes changed")
    official, official_provenance = metric.verified_official_metric()
    qs, data = runner.questions(), runner.load_data()
    slots, errors, consumed, expected_paths = {}, [], set(), set()
    banks = [{"conversation": ci, "arm": arm, "facts": len(entry["bank"]["facts"]),
        "sha256": entry["row"]["bank_sha256"], "path": entry["row"]["bank_path"],
        "index_recovered": entry["row"]["index_recovery"] is not None, "status": "complete"}
        for (ci, arm), entry in loaded.items()]
    for ci in range(10):
        folder = qa.conv_path(ci)
        questions = [q for q in qs if q["conversation"] == ci]
        expected_ids = [q["id"] for q in questions]
        reference_date = list(runner.session_envelopes(data[ci]))[-1]["observation_date"]
        ledger = ledger_rows(snapshot, folder / "qa_api/api_calls.jsonl") or []
        contexts = [scoped_context(ci, arm, qid) for arm in ARMS for qid in expected_ids]
        for event in ledger:
            if (event.get("context") not in contexts or event.get("stage") not in {"qa_reader", "strict_judge"}
                    or event.get("provider") != "gemini" or event.get("operation") != "generateContent"
                    or event.get("model") != ("gemini-2.5-flash" if event.get("stage") == "qa_reader" else "gemini-3.5-flash-lite")):
                raise ValueError("New QA ledger scope/model/stage differs")
        for arm in ARMS:
            entry = loaded[(ci, arm)]
            for q in questions:
                path = folder / "qa" / arm / (q["id"] + ".json")
                expected_paths.add(path.resolve())
                item = {"conversation": ci, "category": q["category"], "query_id": q["id"], "arm": arm,
                        "status": "missing", "scores": None}
                slots[(arm, q["id"])] = item
                if not path.exists():
                    continue
                try:
                    retrieval = entry["retrievals"][q["id"]]
                    binding = qa.input_binding(manifest, digest, entry["row"], q, retrieval, reference_date)
                    row = snapshot.read(path)
                    scores, paths = validate_qa(row, q, arm, retrieval["selected"], binding, folder / "qa_api",
                                               ledger, snapshot, official, expected_ids)
                    consumed.update(paths)
                    item.update(status=row["status"], scores=scores, error=row.get("error"), artifact_sha256=snapshot.sha(path))
                except Exception as exc:
                    item.update(status="invalid", error={"type": type(exc).__name__, "message": str(exc)})
        sp = folder / "qa_status.json"
        if not sp.exists():
            errors.append({"conversation": ci, "error": "Missing QA state"}); continue
        state = snapshot.read(sp)
        observed = {arm: {"completed": sum(slots[(arm, qid)]["status"] == "complete" for qid in expected_ids),
                          "failed": sum(slots[(arm, qid)]["status"] == "failed" for qid in expected_ids)} for arm in ARMS}
        expected = {"policy": qa.POLICY, "stage": "qa", "conversation": ci, "input_manifest_sha256": digest,
            "expected": 2 * len(questions), "done": sum(x["completed"] + x["failed"] for x in observed.values()),
            "completed": sum(x["completed"] for x in observed.values()), "failed": sum(x["failed"] for x in observed.values()), "arms": observed}
        if any(type(state.get(k)) is not type(v) or state.get(k) != v for k, v in expected.items()):
            errors.append({"conversation": ci, "error": "QA state differs from verified outcomes/binding"})
        if state.get("status") != ("requires_review" if expected["failed"] else "complete") or state.get("done") != expected["expected"]:
            errors.append({"conversation": ci, "error": "QA state is not correctly closed"})
    extra = {p.resolve() for p in qa.OUT.glob("conv_*/qa/*/*.json")} - expected_paths
    if extra:
        errors.append({"error": "Unexpected QA files", "paths": sorted(map(str, extra))})
    orphaned = {p.resolve() for p in qa.OUT.glob("conv_*/qa_api/mem0_transfer_qa_events/*/*.json")} - consumed
    if orphaned:
        errors.append({"error": "Unconsumed durable consumer events", "paths": sorted(map(str, orphaned))})
    usage = usage_report(snapshot, manifest)
    coordinator_path = qa.OUT / "qa_coordinator_status.json"
    if not coordinator_path.exists():
        errors.append({"error": "Missing coordinator status"})
    else:
        coordinator = snapshot.read(coordinator_path)
        failed_cis = sorted({v["conversation"] for v in slots.values() if v["status"] != "complete"})
        if (coordinator.get("input_manifest_sha256") != digest or coordinator.get("policy") != qa.POLICY
                or sorted(coordinator.get("finished", [])) != list(range(10))
                or sorted(coordinator.get("failed", [])) != failed_cis
                or coordinator.get("status") != ("requires_review" if failed_cis else "complete")
                or coordinator.get("expected") != 3080 or coordinator.get("workers") != 8 or coordinator.get("query_workers") != 4):
            errors.append({"error": "Coordinator does not close the exact ten-conversation run"})
    original_states = {str(ci): snapshot.read(runner.conv_path(ci) / "write_status.json") for ci in range(10)}
    changed = snapshot.unchanged()
    if changed:
        errors.append({"error": "Inputs changed during summary", "paths": changed})
    arms, paired = core.summarize_rows(qs, slots, bootstrap_samples=bootstrap_samples)
    closed = len(banks) == 20 and len(slots) == 3080 and all(v["status"] in {"complete", "failed"} for v in slots.values()) and not errors
    return {"schema": SCHEMA, "policy": qa.POLICY, "snapshot_utc": runner.now(), "run_dir": str(qa.OUT),
        "input_manifest_sha256": digest, "source_manifest_sha256": manifest["source_manifest"]["sha256"],
        "source_dataset_sha256": runner.SOURCE_HASH, "cohort_per_arm": 1540, "expected_arm_question_slots": 3080,
        "all_twenty_banks_and_3080_terminal_slots_verified": closed,
        "all_3080_answers_complete": closed and all(v["status"] == "complete" for v in slots.values()),
        "banks": banks, "arms": arms, "paired_regroup_minus_session": paired, "outcomes": list(slots.values()),
        "official_metric_provenance": official_provenance, "independent_input_admission": admission,
        "integrity_errors": errors, "usage": usage, "original_v1_write_states_unchanged": original_states,
        "input_hashes": {str(p): h for p, h in snapshot.files.items()},
        "interpretation": "As-run explicit index-recovery experiment, not an original first-pass success. Both conv3 arms restored only reviewed orphan entity links and repeated native retrieval; facts/dates/vectors remain frozen. Other18 mapped inputs are reused. Original v1 failed gates remain failed. All data previously inspected; model service-date drift and offline grouping limit causal/blind-test claims. Rank20 and actual structured Reader/Judge requests/responses are replayed; native ANN search itself is not reconstructed. Missing/failed/invalid remain in each1540 denominator; completed-only results and zero-imputation sensitivity are distinct."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    runs = (ROOT / "runs").resolve()
    if output == runs or output.is_relative_to(runs) or runs.is_relative_to(output):
        raise ValueError("Use a new independent output directory outside experiment runs")
    report = summarize(bootstrap_samples=args.bootstrap_samples)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Native Mem0 recovered sparse1540", "", "Integrity closed: " + str(report["all_twenty_banks_and_3080_terminal_slots_verified"]), "",
        "| Arm | Complete | Failed | Missing | Invalid | Official F1 (completed only) |", "|---|---:|---:|---:|---:|---:|"]
    for arm, value in report["arms"].items():
        c = value["status_counts"]
        lines.append(f"| {arm} | {c['complete']} | {c['failed']} | {c['missing']} | {c['invalid']} | {value['completed_only_metrics']['official_locomo_f1']} |")
    lines.extend(["", report["interpretation"], "", "Full fixed-denominator sensitivities, paired ten-conversation bootstrap, provenance and separate costs are in summary.json."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.json"), "integrity_closed": report["all_twenty_banks_and_3080_terminal_slots_verified"]}))
    return 0 if report["all_twenty_banks_and_3080_terminal_slots_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

