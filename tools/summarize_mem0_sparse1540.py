"""Offline native Mem0 transfer integrity, fixed1540 metrics and separate costs.

No ModelAPI/Memory constructors, network, retries or writes to experiment inputs.
Missing/failed/invalid outcomes never disappear from the 1540-per-arm denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_sparse1540 as runner
from tools import summarize_pilot400 as metric
from tools.summarize_experiments import usage_cost, PRICES_USD_PER_MILLION
from jev_mem import mem0_transfer_qa as transport
from jev_mem import evaluation

SCHEMA = "mem0-native-sparse1540-summary-v1"
ARMS = runner.ARMS
METRICS = metric.METRICS


class Snapshot:
    def __init__(self):
        self.files = {}

    def raw(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in self.files and self.files[path] != digest:
            raise ValueError("Input changed during audit: " + str(path))
        self.files[path] = digest
        return raw

    def read(self, path):
        return json.loads(self.raw(path))

    def sha(self, path):
        self.raw(path)
        return self.files[Path(path).resolve()]

    def unchanged(self):
        return [str(path) for path, digest in self.files.items()
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest]


def sealed(value):
    return runner._verify_sealed(value)


def ledger_rows(snapshot, path):
    path = Path(path)
    if not path.exists():
        return None
    return [json.loads(line) for line in snapshot.raw(path).decode("utf-8-sig").splitlines() if line.strip()]


def scoped_context(ci, arm, qid):
    return {"architecture": runner.POLICY, "conversation": ci, "phase": "qa", "query_id": qid, "method": arm}


def inside(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("Audit pointer leaves its designated directory")
    return path


class DeclaredProviderFailure(RuntimeError):
    """A durable failed consumer event; does not infer the remote outcome."""


class ReplayAuditError(ValueError):
    """An integrity error, distinct from the original evaluator's rejection."""


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
        if (creator.get("architecture") != runner.POLICY or creator.get("conversation") != self.context["conversation"]
                or creator.get("phase") != "qa" or creator.get("method") not in ARMS
                or creator.get("query_id") not in self.expected_ids):
            raise ValueError("Intent creator is outside the declared cohort/scope")
        matched = [r for r in self.ledger if r.get("stage") == stage and r.get("request_hash") == digest
                   and r.get("context") == self.context]
        physical = [r for r in self.ledger if r.get("request_hash") == digest and r.get("cache_hit") is not True]
        if (len(physical) > 3 or any(type(r.get("attempt")) is not int or not 1 <= r["attempt"] <= 3 for r in physical)
                or sorted(r["attempt"] for r in physical) != list(range(1, len(physical) + 1))
                or any(r.get("context") != creator for r in physical)):
            raise ValueError("Physical requests exceed frozen retry budget")
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
        if event.get("error_type") is not None:
            if not isinstance(event["error_type"], str) or not event["error_type"]:
                raise ValueError("Invalid declared provider failure")
            raise DeclaredProviderFailure(event["error_type"])
        if (cache is None or not physical or event.get("response_sha256") != runner.fingerprint(cache["response"])
                or event.get("cache_source") not in {"new_http", "validated_cache"}
                or event.get("new_logical_dispatch") is not (event["cache_source"] == "new_http")
                or not any(r.get("cache_hit") is True or type(r.get("status")) is int and 200 <= r["status"] < 300 for r in matched)):
            raise ValueError("Successful consumer lacks valid raw response/attributed successful attempt")
        return transport._validate_decoded(transport.decode_response(cache["response"]), stage)


def validate_qa(row, q, arm, selected, binding, qa_folder, ledger, snapshot, official, expected_ids):
    sealed(row)
    identity = {"schema": runner.POLICY + "-qa", "query_id": q["id"], "conversation": q["conversation"],
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


def audit_bank(ci, arm, manifest, snapshot):
    bank, digest = runner.load_bank(ci, arm, manifest)
    folder = runner.conv_path(ci)
    path = folder / ("session_bank.json" if arm == ARMS[0] else "regroup_bank.json")
    if snapshot.sha(path) != digest or bank.get("schema") != runner.POLICY + "-bank":
        raise ValueError("Bank schema/bytes differ")
    for fact in bank["facts"]:
        if (not isinstance(fact["text"], str) or not fact["text"].strip()
                or date.fromisoformat(fact["observation_date"]).isoformat() != fact["observation_date"]
                or fact.get("event") != "ADD" or fact.get("support_status") != "unverified"):
            raise ValueError("Native fact text/date/ADD provenance differs")
    if arm == ARMS[1]:
        inp = snapshot.read(folder / "group_inputs.json")
        journals = bank["journal_files"]
        state = snapshot.read(folder / "write_status.json")
        if (len(journals) != len(inp["envelopes"]) or state.get("done") != len(journals)
                or state.get("expected") != len(journals) or state.get("failed") != 0
                or state.get("facts") != len(bank["facts"])):
            raise ValueError("Native write stage/group count does not close")
        flattened, history = [], []
        for index, (pointer, envelope) in enumerate(zip(journals, inp["envelopes"])):
            jp = inside(pointer["path"], folder / "native/session_audit")
            r = snapshot.read(jp)
            if snapshot.sha(jp) != pointer["sha256"] or any(r.get(k) != v for k, v in envelope.items()):
                raise ValueError("Native group journal differs from exact prepared source messages")
            if (r.get("status") != "success" or r.get("sequence") != index or r.get("current_date") != runner.CURRENT_DATE
                    or r.get("builder_call_count") != 1 or len(r.get("llm_calls", [])) != 1
                    or r.get("history_context") != history[-10:] or r.get("system_prompt_changed") is not False
                    or r.get("persistence_check", {}).get("missing_texts") != []
                    or r.get("persistence_check", {}).get("unexpected_texts") != []):
                raise ValueError("Native group date/history/extraction/persistence audit fails")
            flattened.extend(r["fact_records"])
            history.extend({"message": message, "session_id": envelope["session_id"], "source_turn_id": tid}
                           for message, tid in zip(envelope["messages"], envelope["source_turn_ids"]))
        if flattened != [{k: v for k, v in f.items() if k != "embedding"} for f in bank["facts"]]:
            raise ValueError("Native committed facts differ from complete group journals")
        indexes = bank["native_index_audit"]
        if indexes.get("history_adds_verified") != len(flattened) or indexes.get("entity_facts_without_any_link") != []:
            raise ValueError("Native history/entity index audit incomplete")
    return bank, digest


def measure_usage(rows):
    totals = {"physical_attempts": 0, "cache_events": 0, "http_successes": 0, "http_failures": 0,
              "transport_or_unknown": 0, "unknown_cost_attempts": 0, "known_cost_component_usd": 0.,
              "recorded_input_tokens": 0, "recorded_output_tokens": 0, "recorded_thinking_tokens": 0}
    if rows is None:
        return {"available": False, "total_cost_usd": None, "reason": "Missing ledger is not measured zero"}
    for row in rows:
        if row.get("cache_hit") is True:
            totals["cache_events"] += 1
            continue
        if row.get("cache_hit") not in (None, False):
            raise ValueError("Invalid cache flag")
        totals["physical_attempts"] += 1
        status = row.get("status")
        category = ("http_successes" if type(status) is int and 200 <= status < 300 else
                    "http_failures" if type(status) is int and 400 <= status < 600 else "transport_or_unknown")
        totals[category] += 1
        cost = usage_cost(row)
        vals = [cost[k + "_tokens"] for k in ("input", "output", "thinking")]
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0) for v in vals):
            raise ValueError("Malformed recorded usage")
        totals["known_cost_component_usd"] += cost["known_component_usd"]
        totals["unknown_cost_attempts"] += bool(cost["unknown"] or row.get("usage_unknown") or category == "transport_or_unknown")
        for name, value in zip(("input", "output", "thinking"), vals):
            if value is not None:
                totals["recorded_" + name + "_tokens"] += value
    return {"available": True, "rows": len(rows), **totals,
            "total_cost_usd": None if totals["unknown_cost_attempts"] else totals["known_cost_component_usd"]}


def costs(snapshot):
    new_rows, missing, sources = [], [], []
    for ci in range(10):
        for directory in ("native", "qa_api"):
            path = runner.conv_path(ci) / directory / "api_calls.jsonl"
            rows = ledger_rows(snapshot, path)
            if rows is None:
                missing.append(str(path)); continue
            sources.append({"path": str(path), "sha256": snapshot.sha(path)})
            for row in rows:
                context = row.get("context", {})
                if (context.get("architecture") != runner.POLICY or context.get("conversation") != ci
                        or context.get("phase") not in {"write", "retrieve", "qa"}):
                    raise ValueError("New physical ledger has an unexpected architecture/conversation/phase")
            new_rows.extend(rows)
    phases = {p: measure_usage([r for r in new_rows if r["context"]["phase"] == p]) for p in ("write", "retrieve", "qa")}
    for phase, item in phases.items():
        phase_missing = [p for p in missing if ("qa_api" in Path(p).parts) == (phase == "qa")]
        item["missing_ledgers"] = phase_missing
        item["scope_complete"] = not phase_missing
        if phase_missing:
            item["total_cost_usd"] = None
    stages = {stage: measure_usage([r for r in new_rows if r["stage"] == stage]) for stage in sorted({r["stage"] for r in new_rows})}
    baseline, baseline_missing, baseline_sources = [], [], []
    for ci in range(10):
        path = runner.BASE / ("conv_" + str(ci)) / "api_calls.jsonl"
        rows = ledger_rows(snapshot, path)
        if rows is None:
            baseline_missing.append(str(path)); continue
        baseline_sources.append({"path": str(path), "sha256": snapshot.sha(path)})
        baseline.extend(r for r in rows if r.get("context", {}).get("phase") in {"ingestion", "retrieval"})
    grouping_run = runner.GROUPS.parent
    group_paths = {grouping_run / "api_calls.jsonl"}
    lineage_path = grouping_run / "cache_seed_lineage.json"
    lineage_missing = not lineage_path.exists()
    if not lineage_missing:
        lineage = snapshot.read(lineage_path)
        path = Path(lineage["source_ledger_path"])
        if snapshot.sha(path) != lineage["source_ledger_sha256"]:
            raise ValueError("Inherited grouping source ledger changed")
        group_paths.add(path)
    grouping, group_missing, group_sources = [], [], []
    for path in sorted(group_paths):
        rows = ledger_rows(snapshot, path)
        if rows is None:
            group_missing.append(str(path)); continue
        group_sources.append({"path": str(path), "sha256": snapshot.sha(path)})
        grouping.extend(r for r in rows if r.get("context", {}).get("method") == "regroup"
                        and r.get("context", {}).get("phase") == "pilot_write"
                        and (r.get("stage", "").startswith("sparse_topic_")
                             or r.get("stage") in {"pilot_raw_turn_embedding", "pilot_group_token_count"}))
    def scoped(rows, absent, paths):
        result = measure_usage(rows)
        result.update(source_ledgers=paths, missing_ledgers=absent)
        if absent:
            result["total_cost_usd"] = None
            result["scope_complete"] = False
        else:
            result["scope_complete"] = True
        return result
    return {"new_namespace": {**scoped(new_rows, missing, sources), "phases": phases, "stages": stages},
            "inherited_baseline_native_write_and_retrieval": scoped(baseline, baseline_missing, baseline_sources),
            "inherited_sparse_grouping_work": scoped(grouping, group_missing + (["cache_seed_lineage.json"] if lineage_missing else []), group_sources),
            "prices_usd_per_million": PRICES_USD_PER_MILLION,
            "interpretation": "Configured price estimates, not invoices. Each physical ledger row counted once, including retries. Cache reuse adds zero new usage; its source work is listed separately. Inherited scopes exclude old QA, independent writers and pilot fact extraction. No combined all-in cold cost is inferred; absent lineage, unpriced embeddings and unknown transport usage remain unknown."}


def summarize_rows(questions, slots, *, bootstrap_samples=2000):
    denominator = len(questions)
    arms, complete = {}, {}
    for arm in ARMS:
        values = [slots[(arm, q["id"])] for q in questions]
        counts = {s: sum(v["status"] == s for v in values) for s in ("complete", "failed", "missing", "invalid")}
        records = [v["scores"] for v in values if v["status"] == "complete"]
        complete[arm] = {r["query_id"]: r for r in records}
        if sum(counts.values()) != denominator:
            raise ValueError("Arm denominator does not close")
        conditional = {m: sum(r[m] for r in records) / len(records) if records else None for m in METRICS}
        arms[arm] = {"cohort_denominator": denominator, "status_counts": counts,
            "completion_fraction": counts["complete"] / denominator,
            "completed_only_metrics": conditional,
            "full_cohort_observed_metrics": conditional if len(records) == denominator else {m: None for m in METRICS},
            "missing_and_failure_zero_sensitivity": {"is_imputation_not_observed_scoring": True, "denominator": denominator,
                "imputed_count": denominator - len(records), "means": {m: sum(r[m] for r in records) / denominator for m in METRICS}},
            "scored_questions": records,
            "by_conversation": {str(ci): {s: sum(v["status"] == s and v["conversation"] == ci for v in values) for s in counts} for ci in range(10)},
            "by_category": {str(cat): {s: sum(v["status"] == s and v["category"] == cat for v in values) for s in counts} for cat in (1, 2, 3, 4)}}
    paired = metric.paired_comparison(complete[ARMS[0]], complete[ARMS[1]], samples=bootstrap_samples, seed=0, denominator=denominator)
    return arms, paired


def summarize(run_dir=runner.OUT, *, bootstrap_samples=2000):
    if Path(run_dir).resolve() != runner.OUT.resolve():
        raise ValueError("This frozen protocol is bound to runs/mem0_sparse1540_v1")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("Positive integer bootstrap sample count required")
    snapshot = Snapshot()
    manifest = runner.verify_manifest()
    manifest_sha = snapshot.sha(runner.OUT / "manifest.json")
    qs = runner.questions()
    if len(qs) != 1540 or manifest["arms"] != list(ARMS) or manifest["question_ids"] != [q["id"] for q in qs]:
        raise ValueError("Frozen 1540/two-arm manifest differs")
    # Use the independently reviewed full native query/entity embedding gate;
    # do not substitute a weaker reconstruction of the rank20 adapter alone.
    from tools import audit_mem0_sparse1540_retrieval as retrieval_auditor
    admission = retrieval_auditor.audit(list(range(10)), sorted(runner.OUT.glob("*retriev*.stderr.log")))
    if (not admission.get("admitted") or admission.get("manifest_sha256") != manifest_sha
            or [c["conversation"] for c in admission["conversations"]] != list(range(10))):
        raise ValueError("Independent native retrieval admission has not passed for all ten conversations; use its standalone report for preflight")
    def bind_admission(value):
        if isinstance(value, dict):
            if "path" in value:
                digest = value.get("file_sha256", value.get("sha256"))
                if digest and snapshot.sha(value["path"]) != digest:
                    raise ValueError("Admitted retrieval input changed")
            for child in value.values():
                bind_admission(child)
        elif isinstance(value, list):
            for child in value:
                bind_admission(child)
    bind_admission(admission)
    official, official_provenance = metric.verified_official_metric()
    for path, digest in manifest["source_files"].items():
        if snapshot.sha(path) != digest:
            raise ValueError("Inherited bank/grouping/retrieval source bytes changed")
    data = runner.load_data()
    slots, banks_report, errors, expected_qa, expected_retrieval, consumed = {}, [], [], set(), set(), set()
    for ci in range(10):
        folder = runner.conv_path(ci)
        questions = [q for q in qs if q["conversation"] == ci]
        expected_ids = [q["id"] for q in questions]
        ledger = ledger_rows(snapshot, folder / "qa_api/api_calls.jsonl") or []
        allowed_contexts = [scoped_context(ci, arm, qid) for arm in ARMS for qid in expected_ids]
        for event in ledger:
            if (event.get("context") not in allowed_contexts or event.get("stage") not in {"qa_reader", "strict_judge"}
                    or event.get("provider") != "gemini" or event.get("operation") != "generateContent"
                    or event.get("model") != ("gemini-2.5-flash" if event.get("stage") == "qa_reader" else "gemini-3.5-flash-lite")):
                raise ValueError("QA physical ledger event lacks valid frozen scope")
        reference_date = list(runner.session_envelopes(data[ci]))[-1]["observation_date"]
        for arm in ARMS:
            bank = None
            try:
                bank, digest = audit_bank(ci, arm, manifest, snapshot)
                banks_report.append({"conversation": ci, "arm": arm, "status": "complete", "facts": len(bank["facts"]), "sha256": digest})
            except Exception as exc:
                banks_report.append({"conversation": ci, "arm": arm, "status": "missing" if isinstance(exc, FileNotFoundError) else "invalid", "error": str(exc)})
            for q in questions:
                rp, qp = folder / "retrieval" / arm / (q["id"] + ".json"), folder / "qa" / arm / (q["id"] + ".json")
                expected_qa.add(qp.resolve()); expected_retrieval.add(rp.resolve())
                item = {"conversation": ci, "category": q["category"], "query_id": q["id"], "arm": arm,
                        "status": "missing", "scores": None}
                slots[(arm, q["id"])] = item
                try:
                    if rp.exists():
                        if bank is None:
                            raise ValueError("Retrieval exists without a validated complete bank")
                        retrieval = runner.validate_retrieval(snapshot.read(rp), q, arm, bank, digest)
                        if arm == ARMS[0]:
                            source = retrieval["source"]
                            if manifest["source_files"].get(source["path"]) != source["sha256"] or source.get("reused_native_search") is not True:
                                raise ValueError("Session retrieval source lineage differs")
                        elif retrieval["source"] != {"new_native_search": True, "top_k": 60, "threshold": .1, "rerank": False}:
                            raise ValueError("Regroup retrieval is not the declared native search")
                    else:
                        retrieval = None
                    if not qp.exists():
                        continue
                    if retrieval is None:
                        raise ValueError("QA exists without a validated retrieval")
                    selected = retrieval["selected"]
                    binding = {"manifest_sha256": manifest_sha, "bank_sha256": digest, "retrieval_sha256": snapshot.sha(rp),
                               "selected_sha256": runner.fingerprint(selected), "question_sha256": runner.fingerprint(q["question"]),
                               "reference_sha256": runner.fingerprint(q["reference"]), "reference_date": reference_date}
                    row = snapshot.read(qp)
                    scored, paths = validate_qa(row, q, arm, selected, binding, folder / "qa_api", ledger, snapshot, official, expected_ids)
                    consumed.update(paths)
                    item.update(status=row["status"], scores=scored, error=row.get("error"), artifact_sha256=snapshot.sha(qp))
                except Exception as exc:
                    item.update(status="invalid", error={"type": type(exc).__name__, "message": str(exc)})
        sp = folder / "qa_status.json"
        if sp.exists():
            state = snapshot.read(sp)
            observed = {arm: {"completed": sum(slots[(arm, qid)]["status"] == "complete" for qid in expected_ids),
                              "failed": sum(slots[(arm, qid)]["status"] == "failed" for qid in expected_ids)} for arm in ARMS}
            if (state.get("policy") != runner.POLICY or state.get("conversation") != ci or state.get("stage") != "qa"
                    or state.get("expected") != 2 * len(questions) or state.get("arms") != observed
                    or state.get("done") != sum(v["completed"] + v["failed"] for v in observed.values())
                    or state.get("completed") != sum(v["completed"] for v in observed.values())
                    or state.get("failed") != sum(v["failed"] for v in observed.values())):
                errors.append({"conversation": ci, "error": "QA state differs from verified artifacts"})
            if state.get("status") in {"complete", "requires_review"} and state["status"] != ("requires_review" if state.get("failed") else "complete"):
                errors.append({"conversation": ci, "error": "QA terminal label differs from failure count"})
            if state.get("status") not in {"complete", "requires_review"} or state.get("done") != 2 * len(questions):
                errors.append({"conversation": ci, "error": "QA stage not terminal for all expected slots"})
        else:
            errors.append({"conversation": ci, "error": "Missing QA stage status"})
    actual_qa = {p.resolve() for p in runner.OUT.glob("conv_*/qa/*/*.json")}
    actual_retrieval = {p.resolve() for p in runner.OUT.glob("conv_*/retrieval/*/*.json")}
    if actual_qa - expected_qa or actual_retrieval - expected_retrieval:
        errors.append({"error": "Unexpected QA or retrieval paths", "paths": sorted(map(str, (actual_qa - expected_qa) | (actual_retrieval - expected_retrieval)))})
    event_files = {p.resolve() for p in runner.OUT.glob("conv_*/qa_api/mem0_transfer_qa_events/*/*.json")}
    orphaned = event_files - consumed
    if orphaned:
        errors.append({"error": "Unconsumed durable QA events", "paths": sorted(map(str, orphaned))})
    usage = costs(snapshot)
    changed = snapshot.unchanged()
    if changed:
        errors.append({"error": "Inputs changed during summary", "paths": changed})
    arms, paired = summarize_rows(qs, slots, bootstrap_samples=bootstrap_samples)
    closed = (len(banks_report) == 20 and all(b["status"] == "complete" for b in banks_report)
              and all(s["status"] in {"complete", "failed"} for s in slots.values()) and not errors)
    return {"schema": SCHEMA, "snapshot_utc": runner.now(), "run_dir": str(runner.OUT),
            "manifest_sha256": manifest_sha, "source_dataset_sha256": runner.SOURCE_HASH,
            "cohort_per_arm": 1540, "expected_arm_question_slots": 3080, "banks": banks_report,
            "all_twenty_banks_and_3080_terminal_slots_verified": closed,
            "all_3080_answers_complete": closed and all(s["status"] == "complete" for s in slots.values()),
            "arms": arms, "paired_regroup_minus_session": paired, "official_metric_provenance": official_provenance,
            "independent_native_retrieval_admission": admission,
            "outcomes": list(slots.values()), "integrity_errors": errors, "usage": usage,
            "input_hashes": {str(k): v for k, v in snapshot.files.items()},
            "interpretation": "As-run native Mem0 sparse-splitting transfer, including previously inspected development data; not a blind test. Cached baseline native pools are preserved; this audits rank20 selection from recorded native pools, not native-search reconstruction. Native input scopes are unverified support. Failed/missing/invalid outcomes have no observed score and remain in each arm's1540 denominator. Inherited and new costs are separate; no unknown cost is a measured zero."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=runner.OUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    output, run = args.output_dir.resolve(), args.run_dir.resolve()
    if output == run or run in output.parents or output in run.parents or (ROOT / "runs").resolve() in output.parents:
        raise ValueError("Use a new independent output directory outside all runs")
    report = summarize(run, bootstrap_samples=args.bootstrap_samples)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Native Mem0 sparse1540", "", "Integrity closed: " + str(report["all_twenty_banks_and_3080_terminal_slots_verified"]), "",
             "| Arm | Complete | Failed | Missing | Invalid | Official F1 (completed only) |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in report["arms"].items():
        c = row["status_counts"]
        lines.append(f"| {arm} | {c['complete']} | {c['failed']} | {c['missing']} | {c['invalid']} | {row['completed_only_metrics']['official_locomo_f1']} |")
    lines += ["", report["interpretation"], "", "See summary.json for paired bootstrap, individual outcomes, zero-imputation sensitivity, input hashes and separate inherited/new cost estimates."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.json"), "integrity_closed": report["all_twenty_banks_and_3080_terminal_slots_verified"]}))
    return 0 if report["all_twenty_banks_and_3080_terminal_slots_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
