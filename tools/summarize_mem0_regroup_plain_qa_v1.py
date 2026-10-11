"""Offline3080-slot plain-text QA audit with separate Reader and Judge denominators.

Reuses the frozen metric, paired cluster bootstrap, raw-request replay and usage
helpers. No ModelAPI/Memory constructor, credential access or model call occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_regroup_plain_qa_v1 as driver
from tools import summarize_pilot400 as metrics
from tools.summarize_mem0_sparse1540 import Snapshot, measure_usage
from tools.summarize_sdk20_modules import ReplayAPI, DeclaredTransportFailure, InvalidArtifact, fingerprint
from tools.run_sdk20_modules import digest, sha, now, verify, save_new
from jev_mem import evaluation_plain_text_v1 as evaluator
from jev_mem.plain_text_api_v1 import check_plain_text_request
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem import qa_embedding_cache_pack_v1 as cache_pack
from tools.summarize_experiments import PRICES_USD_PER_MILLION

SCHEMA = "mem0-dated-plain-qa-summary-v1"
ARMS = driver.ARMS
METRICS = ("official_locomo_f1", "custom_token_f1", "judge_correct")


class PlainReplay(ReplayAPI):
    """The original generator and physical-request proof, with new stage guard."""
    def __init__(self, *args, snapshot=None, unresolved_guard=False, unresolved_hash=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot, self.unresolved_guard, self.unresolved_hash = snapshot, unresolved_guard, unresolved_hash

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", **kwargs):
        check_plain_text_request(provider, model, body, operation=operation, stage=stage)
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        key = request_digest(signature)
        if (self.api_dir / "cache_lineage" / (key + ".json")).exists():
            raise InvalidArtifact("Plain Reader/Judge may not inherit old QA responses from embedding-only sources")
        target = self.api_dir / "api_cache" / (key + ".json")
        intent_path = self.api_dir / "request_intents" / (key + ".json")
        if self.snapshot is not None:
            for path in (target, intent_path):
                if path.exists():
                    self.snapshot.raw(path)
        if self.unresolved_guard:
            if self.events or target.exists() or not intent_path.exists() or self.unresolved_hash not in (None, key):
                raise InvalidArtifact("Prior unresolved request guard has contradictory dispatch/raw evidence")
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if (intent.get("signature") != signature or intent.get("request_hash") != key
                    or intent.get("stage") != stage or intent.get("retries") != 2
                    or intent.get("context", {}).get("conversation") != self.context["conversation"]
                    or self.ledger.get((key, stage, fingerprint(self.context)), [])):
                raise InvalidArtifact("Unresolved guard does not bind a prior same-conversation request")
            # CachedLegacyAPI raises before constructing this consumer's event.
            # An earlier producer may have failed or remained unknown. This is
            # a retained technical failure, never a generated negative answer.
            raise DeclaredTransportFailure("RuntimeError")
        return super().request(provider, model, body, operation=operation, stage=stage, **kwargs)


def bound(snapshot, path, expected):
    value = snapshot.read(path)
    if value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}):
        raise InvalidArtifact("Artifact content hash differs")
    if any(value.get(k) != v for k, v in expected.items()):
        raise InvalidArtifact("Artifact question, inputs or protocol binding differs")
    return value


def ledger_index(snapshot, run, cohort=None):
    rows, paths, missing = [], [], []
    for ci in range(10):
        path = run / f"conv_{ci}/api/api_calls.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        for line in snapshot.raw(path).decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get("context", {})
            if (context.get("conversation") != ci or context.get("architecture") != driver.SCHEMA
                    or context.get("phase") not in ("qa", "retrieval") or context.get("method") not in ARMS):
                raise InvalidArtifact("New ledger context is outside the two fixed QA arms")
            if cohort is not None and context.get("query_id") not in {q["id"] for q in cohort[str(ci)]}:
                raise InvalidArtifact("Ledger query is outside the frozen cohort")
            if context["phase"] == "qa":
                stages = {"reader": (evaluator.READER_STAGE, evaluator.READER_MODEL),
                          "judge": (evaluator.JUDGE_STAGE, evaluator.JUDGE_MODEL)}
                expected = stages.get(context.get("component"))
                if expected != (row.get("stage"), row.get("model")) or row.get("operation") != "generateContent":
                    raise InvalidArtifact("QA ledger uses another model/stage/operation")
            elif (row.get("model") != "gemini-embedding-001"
                  or row.get("operation") not in ("batchEmbedContents", "countTokens")
                  or row.get("stage") not in ("mem0_embedding_search", "mem0_embedding_search_length_check")):
                raise InvalidArtifact("Retrieval ledger contains undeclared computation")
            if row.get("provider") != "gemini" or row.get("resolved_model") not in (None, row.get("model")):
                raise InvalidArtifact("Ledger provider/resolved model differs")
            if row.get("cache_hit") is True:
                if "status" in row or "attempt" in row or row.get("usage_semantics") != "reused_response_no_new_model_usage":
                    raise InvalidArtifact("Malformed cache consumer ledger")
            elif type(row.get("attempt")) is not int or not 1 <= row["attempt"] <= 3:
                raise InvalidArtifact("Physical attempt is outside the original bounded transport")
            rows.append(row)
        paths.append({"path": str(path), "sha256": snapshot.sha(path)})
    indexed = defaultdict(list)
    for row in rows:
        indexed[(row["request_hash"], row["stage"], fingerprint(row["context"]))].append(row)
    return rows, indexed, paths, missing


def selected(snapshot, run, ci, arm, q, m, bank, bank_hash):
    path = run / f"conv_{ci}/selection" / arm / (q["id"] + ".json")
    ident = driver.identity(ci, arm, q, m)
    row = bound(snapshot, path, {**ident, "bank_sha256": bank_hash})
    if row.get("status") == "failed":
        if row.get("selected_ids") != [] or not row.get("error"):
            raise InvalidArtifact("Failed retrieval must not supply partial evidence")
        return row, path
    if row.get("status") != "complete" or row.get("error") is not None:
        raise InvalidArtifact("Selection is not a valid terminal result")
    capture = row.get("capture", {})
    if capture.get("capture_sha256") != digest({k: v for k, v in capture.items() if k != "capture_sha256"}):
        raise InvalidArtifact("Native selection capture differs")
    ids = row.get("selected_ids")
    if (not isinstance(ids, list) or len(ids) != 20 or len(set(ids)) != 20
            or ids != capture.get("base20_ids") or not set(ids) <= {f["id"] for f in bank["facts"]}
            or capture.get("candidate_cap") != 80 or capture.get("inputs", {}).get("top_k") != 20
            or len(capture.get("candidate_ids", [])) > 80):
        raise InvalidArtifact("Native top20 identity/order/budget differs")
    from mem0.utils.scoring import score_and_rank
    if score_and_rank(**capture["inputs"]) != capture["base20"]:
        raise InvalidArtifact("Native scorer does not reproduce the captured20")
    return row, path


def stage_context(ci, arm, qid, name):
    return {"conversation": ci, "architecture": driver.SCHEMA, "query_id": qid,
            "method": arm, "phase": "qa", "component": name}


def audit_stage(snapshot, path, expected, api_dir, context, indexed, callback, *, no_dispatch=False):
    if not path.exists():
        return {"status": "missing", "result": None, "error": None}
    row = bound(snapshot, path, expected)
    state = row.get("status")
    logs = [r for rs in indexed.values() for r in rs if r.get("context") == context]
    if state == "not_run":
        if not no_dispatch or row.get("result") is not None or not row.get("error") or row.get("api_events") != [] or logs:
            raise InvalidArtifact("Not-run stage has downstream activity or lacks its prerequisite failure")
        return row
    if no_dispatch or state not in ("complete", "failed"):
        raise InvalidArtifact("Stage executed without a successful prerequisite")
    intent = snapshot.read(path.with_suffix(".intent.json"))
    if any(intent.get(k) != v for k, v in expected.items()):
        raise InvalidArtifact("Stage dispatch intent differs")
    failure_info = row.get("error") or {}
    if failure_info.get("type") == "UnresolvedStageIntent":
        if (state != "failed" or row.get("result") is not None or row.get("api_events") != []
                or failure_info.get("message") != "Prior stage is retained; no implicit redispatch"
                or row.get("raw_invalid_label") is not None):
            raise InvalidArtifact("Interrupted-stage marker is inconsistent")
        return row
    unresolved = (state == "failed" and failure_info.get("type") == "RuntimeError"
                  and failure_info.get("message", "").startswith("Prior unresolved logical request is retained; no implicit redispatch: "))
    replay = PlainReplay(api_dir, row.get("api_events"), context, indexed, snapshot=snapshot, unresolved_guard=unresolved,
                         unresolved_hash=failure_info["message"].rsplit(": ", 1)[-1] if unresolved else None)
    generated, failure = None, None
    try:
        generated = callback(replay)
    except InvalidArtifact:
        raise
    except Exception as exc:
        failure = exc
    replay.assert_consumed()
    if state == "complete":
        if failure is not None or row.get("error") is not None or generated != row.get("result"):
            raise InvalidArtifact("Successful stage does not replay its raw result and metric")
    else:
        if failure is None or row.get("result") is not None or not row.get("error"):
            raise InvalidArtifact("Declared stage failure cannot be reproduced")
        expected_type = failure.provider_error_type if isinstance(failure, DeclaredTransportFailure) else type(failure).__name__
        if row["error"].get("type") != expected_type:
            raise InvalidArtifact("Declared failure type differs from original replay")
        invalid_raw = failure.raw_response if isinstance(failure, evaluator.InvalidJudgeLabel) else None
        if row.get("raw_invalid_label") != invalid_raw:
            raise InvalidArtifact("Invalid Judge label raw differs")
    return row


def audit_slot(snapshot, run, ci, arm, q, m, bank, bank_hash, indexed, official):
    result = {"query_id": q["id"], "conversation": ci, "category": q["category"], "method": arm,
              "selection_status": "missing", "reader_status": "missing", "judge_status": "missing",
              "qa_status": "missing", "scores": {key: None for key in METRICS}, "errors": []}
    try:
        selection, sp = selected(snapshot, run, ci, arm, q, m, bank, bank_hash)
        result["selection_status"] = selection["status"]
    except FileNotFoundError:
        return result
    except Exception as exc:
        result["selection_status"] = "invalid"
        result["errors"].append({"stage": "selection", "type": type(exc).__name__, "message": str(exc)[:1000]})
        return result
    common = {**driver.identity(ci, arm, q, m), "selection_sha256": snapshot.sha(sp), "bank_sha256": bank_hash,
              "selected_ids": selection["selected_ids"], "reference_date": m["reference_dates"][str(ci)]}
    api_dir = run / f"conv_{ci}/api"
    rp = run / f"conv_{ci}/reader" / arm / (q["id"] + ".json")
    jp = run / f"conv_{ci}/judge" / arm / (q["id"] + ".json")
    byid = {f["id"]: f for f in bank["facts"]}
    def reader_call(api):
        answer = evaluator.read_answer(api, q["question"], q["reference"], [byid[i] for i in selection["selected_ids"]],
                                       reference_date=common["reference_date"])
        return {**answer, "metrics": {"official_locomo_f1": official(answer["prediction"]["answer"], q["reference"], q["category"]),
                                      "custom_token_f1": answer["token_f1"], "custom_exact_match": answer["exact_match"]}}
    reader = None
    reader_success = False
    try:
        reader = audit_stage(snapshot, rp, {**common, "stage": "reader"}, api_dir,
                             stage_context(ci, arm, q["id"], "reader"), indexed, reader_call,
                             no_dispatch=selection["status"] != "complete")
        result["reader_status"] = reader["status"]
        if reader["status"] == "complete":
            for name in ("official_locomo_f1", "custom_token_f1"):
                result["scores"][name] = reader["result"]["metrics"][name]
    except Exception as exc:
        result["reader_status"] = "invalid"
        result["errors"].append({"stage": "reader", "type": type(exc).__name__, "message": str(exc)[:1000]})
    judge_expected = {**common, "stage": "judge"}
    if selection["status"] == "complete" and rp.exists():
        judge_expected["reader_sha256"] = snapshot.sha(rp)
    try:
        reader_success = reader is not None and reader["status"] == "complete"
        judge = audit_stage(snapshot, jp, judge_expected, api_dir, stage_context(ci, arm, q["id"], "judge"), indexed,
                            lambda api: evaluator.judge_answer(api, q["question"], q["reference"], reader["result"]["prediction"]["answer"]),
                            no_dispatch=not reader_success)
        result["judge_status"] = judge["status"]
        if judge["status"] == "complete":
            result["scores"]["judge_correct"] = float(judge["result"]["correct"])
    except Exception as exc:
        result["judge_status"] = "invalid"
        result["errors"].append({"stage": "judge", "type": type(exc).__name__, "message": str(exc)[:1000]})
    qp = run / f"conv_{ci}/qa" / arm / (q["id"] + ".json")
    if qp.exists():
        try:
            final = bound(snapshot, qp, common)
            if (result["reader_status"] == "invalid" or result["judge_status"] == "invalid"
                    or final.get("reader") != {"path": str(rp), "sha256": snapshot.sha(rp)}
                    or final.get("judge") != {"path": str(jp), "sha256": snapshot.sha(jp)}
                    or final.get("reader_status") != result["reader_status"] or final.get("judge_status") != result["judge_status"]
                    or final.get("reader_metrics") != (reader["result"]["metrics"] if reader_success else None)
                    or final.get("judge_correct") != (bool(result["scores"]["judge_correct"]) if result["judge_status"] == "complete" else None)):
                raise InvalidArtifact("Final combined artifact differs from the independent stages")
            expected_status = "complete" if result["reader_status"] == result["judge_status"] == "complete" else "failed"
            if final.get("status") != expected_status:
                raise InvalidArtifact("Combined terminal state differs")
            result["qa_status"] = expected_status
        except Exception as exc:
            result["qa_status"] = "invalid"
            result["errors"].append({"stage": "qa", "type": type(exc).__name__, "message": str(exc)[:1000]})
    return result


def paired_metric(name, left, right, *, samples, denominator):
    # Reuse the exact existing estimator/eligibility rule without inventing the
    # old JSON Judge's nonexistent completeness field or mutating module globals.
    original = metrics.paired_comparison
    scoped = FunctionType(original.__code__, {**original.__globals__, "METRICS": (name,)},
                          original.__name__, original.__defaults__, original.__closure__)
    scoped.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return scoped(left, right, samples=samples, seed=0, denominator=denominator)


def aggregate(slots, *, denominator=1540, bootstrap_samples=2000):
    arms, valid = {}, {arm: {key: {} for key in METRICS} for arm in ARMS}
    for arm in ARMS:
        values = [v for v in slots if v["method"] == arm]
        if len(values) != denominator:
            raise ValueError("Full planned denominator must include every missing or failed slot")
        if len({v["query_id"] for v in values}) != denominator:
            raise ValueError("Planned slots must contain distinct query identities")
        for value in values:
            for name in METRICS:
                score = value["scores"][name]
                if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("Metric is not a finite unit interval score")
        status_counts = {stage: dict(Counter(v[stage + "_status"] for v in values)) for stage in ("selection", "reader", "judge", "qa")}
        report = {}
        for key in METRICS:
            records = [{"query_id": v["query_id"], "conversation": v["conversation"], key: v["scores"][key]}
                       for v in values if v["scores"][key] is not None]
            valid[arm][key] = {v["query_id"]: v for v in records}
            count = len(records)
            total = sum(v[key] for v in records)
            report[key] = {"valid_denominator": count, "planned_denominator": denominator,
                           "complete_only_mean": total / count if count else None,
                           "full_cohort_observed_mean": total / denominator if count == denominator else None,
                           "zero_imputed_sensitivity": {"mean": total / denominator, "imputed_slots": denominator - count,
                                                        "is_imputation_not_observed_score": True}}
        arms[arm] = {"planned_denominator": denominator, "statuses": status_counts, "metrics": report,
                     "by_conversation": {}, "by_category": {}}
        for dimension, keys in (("conversation", range(10)), ("category", range(1, 5))):
            for key in keys:
                sub = [v for v in values if v[dimension] == key]
                arms[arm]["by_" + dimension][str(key)] = {"planned_denominator": len(sub), "metrics": {
                    name: {"valid_denominator": sum(v["scores"][name] is not None for v in sub),
                           "complete_only_mean": (sum(v["scores"][name] for v in sub if v["scores"][name] is not None)
                                                  / sum(v["scores"][name] is not None for v in sub))
                           if any(v["scores"][name] is not None for v in sub) else None} for name in METRICS}}
    paired = {key: paired_metric(key, valid[ARMS[0]][key], valid[ARMS[1]][key], samples=bootstrap_samples,
                                denominator=denominator) for key in METRICS}
    return {"arms": arms, "paired_merged_minus_unmerged": paired,
            "interpretation": "Previously exposed10-conversation cohort; each metric uses its own common-valid subset. Missingness can bias pairs. Bootstrap is exploratory, not blind validation or a causal claim."}


def usage_and_lineage(snapshot, run, m, rows, paths, missing):
    packed = cache_pack.verify_pack(m["embedding_cache_pack"], check_sources=True)
    pack_rows = {(e["conversation"], e["request_hash"]): e for e in packed["entries"]}
    inherited, unresolved = [], []
    for ci in range(10):
        api = run / f"conv_{ci}/api"
        for path in (api / "request_intents").glob("*.json"):
            intent = snapshot.read(path)
            key = intent["request_hash"]
            if key != path.stem or request_digest(intent["signature"]) != key or intent["context"].get("conversation") != ci:
                raise InvalidArtifact("Current request intent identity differs")
            if not (api / "api_cache" / (key + ".json")).exists():
                physical = [r for r in rows if r["context"].get("conversation") == ci
                            and r["request_hash"] == key and r.get("cache_hit") is not True]
                unresolved.append({"conversation": ci, "request_hash": key, "intent": {"path": str(path), "sha256": snapshot.sha(path)},
                                   "recorded_attempts": len(physical), "unlogged_dispatch_possible": not physical})
        for path in (api / "cache_lineage").glob("*.json"):
            line = snapshot.read(path)
            key = line["request_hash"]
            if (ci, key) not in pack_rows:
                raise InvalidArtifact("QA lineage is outside the embedding-only pack")
            source = pack_rows[(ci, key)]
            target = api / "api_cache" / (key + ".json")
            if (Path(line["source_path"]).resolve() != Path(source["target"]["path"])
                    or Path(line["target_path"]).resolve() != target
                    or line["source_sha256"] != source["source"]["sha256"]
                    or line["target_sha256"] != snapshot.sha(target) or line["target_sha256"] != line["source_sha256"]
                    or (api / "request_intents" / (key + ".json")).exists()):
                raise InvalidArtifact("Embedding cache original→pack→consumer lineage differs")
            inherited.append({"conversation": ci, "request_hash": key, "original": source["source"],
                              "pack": source["target"], "consumer": {"path": str(target), "sha256": snapshot.sha(target)}})
    total = measure_usage(rows)
    if missing or any(v["unlogged_dispatch_possible"] for v in unresolved):
        total["total_cost_usd"] = None
    return {"new_namespace_only": total, "missing_ledgers": missing, "scope_complete": not missing,
            "ledger_sources": paths, "by_stage": {stage: measure_usage([r for r in rows if r["stage"] == stage]) for stage in sorted({r["stage"] for r in rows})},
            "prices_usd_per_million": PRICES_USD_PER_MILLION, "inherited_embedding_requests": len(inherited),
            "embedding_lineage": inherited, "requests_without_raw": unresolved,
            "note": "Physical attempts only are billed here. Cache events reuse previously paid computation; old raw usage is not billed again. Missing prices or usage make total unknown; this is not free cold-start cost."}


def summarize(run_dir, *, bootstrap_samples=2000):
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("Positive bootstrap samples required")
    run = Path(run_dir).resolve()
    initial = driver.read(run / "run_manifest.json")
    gate = verify(initial["gate"]["path"])
    if gate.get("schema") == "mem0-dated-regroup-qa-gate-v2":
        from tools import run_mem0_regroup_plain_qa_v2 as runner
    else:
        runner = driver
    m = runner.manifest(run, recheck_sources=True)
    snapshot = Snapshot()
    snapshot.raw(run / "run_manifest.json")
    terminal = snapshot.read(run / "run_status.json")
    if terminal.get("terminal") is not True:
        raise ValueError("Use the health checker while active; this summary requires a terminal controller")
    official, provenance = metrics.verified_official_metric()
    questions = driver.sources.cohort()[0]
    rows, indexed, paths, missing = ledger_index(snapshot, run, questions)
    expected_paths = {run / f"conv_{ci}" / stage / arm / (q["id"] + suffix)
                      for ci in range(10) for stage in ("selection", "reader", "judge", "qa")
                      for arm in ARMS for q in questions[str(ci)]
                      for suffix in ((".json", ".intent.json") if stage in ("reader", "judge") else (".json",))}
    observed_paths = {p for ci in range(10) for stage in ("selection", "reader", "judge", "qa")
                      for p in (run / f"conv_{ci}" / stage).rglob("*.json")}
    if observed_paths - expected_paths:
        raise InvalidArtifact("Unexpected query/stage artifacts outside the3080 planned slots")
    slots = []
    for ci in range(10):
        for arm in ARMS:
            bank, bank_hash = driver.get_bank(run, ci, arm, m)
            snapshot.raw(m["banks"][str(ci)][arm]["path"])
            slots.extend(audit_slot(snapshot, run, ci, arm, q, m, bank, bank_hash, indexed, official) for q in questions[str(ci)])
    report = aggregate(slots, bootstrap_samples=bootstrap_samples)
    usage = usage_and_lineage(snapshot, run, m, rows, paths, missing)
    changed = snapshot.unchanged()
    if changed:
        raise ValueError("Inputs changed during offline audit; no mixed snapshot accepted")
    return {"schema": SCHEMA, "created_utc": now(), "run": str(run), "protocol_sha256": m["protocol_sha256"],
            "planned_total": 3080, "official_metric_provenance": provenance, **report, "slots": slots, "usage": usage,
            "audit_status": "invalid_artifacts_present" if any(v["errors"] for v in slots) else "no_invalid_artifacts",
            "invalid_slot_count": sum(bool(v["errors"]) for v in slots),
            "source_limits": m["limitations"], "input_sha256": {str(p): h for p, h in snapshot.files.items()},
            "model_calls": 0, "native_clients": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Offline report output must be new")
    report = summarize(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    save_new(args.output_dir / "summary.json", report)
    lines = ["# Mem0两臂纯文本QA离线审核", "", "各臂计划1540题；Reader/F1与Judge有效分母分别报告。", "",
             "|臂|Reader有效|Judge有效|官方F1（有效题）|custom F1（有效题）|Judge（有效题）|", "|---|---:|---:|---:|---:|---:|"]
    def fmt(value): return "—" if value is None else f"{100 * value:.3f}"
    for arm in ARMS:
        a = report["arms"][arm]["metrics"]
        lines.append(f"|{arm}|{a['official_locomo_f1']['valid_denominator']}/1540|{a['judge_correct']['valid_denominator']}/1540|"
                     f"{fmt(a['official_locomo_f1']['complete_only_mean'])}|{fmt(a['custom_token_f1']['complete_only_mean'])}|{fmt(a['judge_correct']['complete_only_mean'])}|")
    lines += ["", report["interpretation"], "", "零填充仅是缺失敏感度分析，不是实际生成评分。费用未知项不会填零，原缓存不重复计费。"]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir), "model_calls": 0}))


if __name__ == "__main__":
    main()
