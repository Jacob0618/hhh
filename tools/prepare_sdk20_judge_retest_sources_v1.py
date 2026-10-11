"""Offline fixed-Reader sources for a new Judge-only comparison of old SDK20.

First-pass A/B only. Existing later Reader recoveries are historical references,
never mixed into these3042 strings. No native client, API or credential access.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_sdk20_modules as legacy
from tools import sdk20_module_sources as sources
from tools import recover_sdk20_modules as recovery
from tools import summarize_sdk20_modules as summary_tool
from tools.run_sdk20_modules import save_new, verify, digest, now
from tools.recover_mem0_plain_judge_labels_v1 import offline, pointer, check_pointer
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem import evaluation_plain_text_v1 as plain
from jev_mem.judge_terminal_label_compat_v1 import derive_label
from jev_mem.answer_format_compat import normalize_reader_prediction
from jev_mem.evaluation import lexical_metrics
from tools.summarize_pilot400 import verified_official_metric

SCHEMA = "sdk20-plain-judge-retest-sources-v1"
ARMS = ("baseline_sdk20", "fact_types_sdk20")
SUMMARY = ROOT / "research/sdk20_modules_results_2026-09-28_v1/summary.json"
SUMMARY_SHA = "a66827f76d87f690a31e561f4ddc03cb43bb40802cfb04cd06fb1c81c1a9d776"
HISTORICAL_RECOVERY = ROOT / "research/sdk20_modules_recovery_results_2026-09-28_v1/summary.json"
HISTORICAL_RECOVERY_SHA = "f6641cb32523314e19b9f05b0d4972bf0fabce206dfb9facc7cffb366aee3802"
CONFIG = ROOT / "configs/plain_text_qa_v1.json"
GLOBAL_COMPAT = ROOT / "runs/mem0_plain_judge_label_compat_v1/summary.json"
CODE = ("tools/prepare_sdk20_judge_retest_sources_v1.py", "tools/sdk20_module_sources.py",
        "tools/run_sdk20_modules.py", "tools/recover_sdk20_modules.py", "tools/summarize_sdk20_modules.py",
        "tools/recover_mem0_plain_judge_labels_v1.py", "jev_mem/providers.py", "jev_mem/evaluation.py",
        "jev_mem/evaluation_plain_text_v1.py", "jev_mem/judge_terminal_label_compat_v1.py", "jev_mem/data.py",
        "jev_mem/answer_format_compat.py", "jev_mem/locomo_official_metrics.py", "tools/summarize_pilot400.py")
EXPECTED = {ARMS[0]: {"ready": 1525, "blocked_original_reader": 15},
            ARMS[1]: {"ready": 1517, "blocked_original_reader": 23}}


def text_sha(value): return hashlib.sha256(value.encode("utf-8")).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Capture:
    generate = ModelAPI.generate
    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified"):
        raise recovery.CapturedRequest({"provider": provider, "model": model, "operation": operation, "body": body})


def judge_signature(question, reference, answer):
    try: plain.judge_answer(Capture(), question, reference, answer)
    except recovery.CapturedRequest as exc: return exc.signature
    raise AssertionError("Judge signature not captured")


class RawOnly:
    generate = ModelAPI.generate
    def __init__(self, raw): self.raw = raw
    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified"):
        if self.raw["request"] != {"provider": provider, "model": model, "operation": operation, "body": body}:
            raise ValueError("Original Reader request changed")
        if self.raw.get("stage") != "qa_reader" or stage != "qa_reader":
            raise ValueError("Not an original JSON Reader response")
        return self.raw["response"]


def reader_prediction(raw):
    sig = raw["request"]; body = sig["body"]
    prompt = body["contents"][0]["parts"][0]["text"]
    system = body["systemInstruction"]["parts"][0]["text"]
    return RawOnly(raw).generate(prompt, model=sig["model"], system=system, json_output=True,
                                 max_output=2048, stage="qa_reader")


def answer_fields(row, outcome, prediction):
    """No list normalization here, including already recovered historical cases."""
    if row["status"] == "complete":
        answer = row["result"]["prediction"]["answer"]
        if not isinstance(answer, str) or not answer.strip() or prediction.get("answer") != answer:
            raise ValueError("Saved successful string differs from its original Reader raw")
        return {"status": "ready", "reader_status": "complete", "answer": answer,
                "answer_sha256": text_sha(answer), "pending_original_list": None,
                "F1": {"official_locomo_f1": outcome["scores"]["official_locomo_f1"],
                       "custom_token_f1": outcome["scores"]["custom_token_f1"],
                       "custom_exact_match": row["result"]["exact_match"]},
                "old_judge": deepcopy(row["result"]["judgment"])}
    if (row["status"] != "failed" or row.get("result") is not None
            or row.get("error") != {"type": "ValueError", "message": "Reader did not return an answer string"}
            or outcome.get("scores") is not None):
        raise ValueError("Only the reviewed original nonstring Reader failure is admitted")
    value = prediction.get("answer")
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("Pending original Reader must be a nonempty list of nonempty strings")
    return {"status": "blocked_original_reader", "reader_status": "failed", "answer": None,
            "answer_sha256": None, "pending_original_list": deepcopy(value),
            "F1": {"official_locomo_f1": None, "custom_token_f1": None, "custom_exact_match": None},
            "old_judge": None}


def deferred_resolution(prediction, q, official):
    normalized, proof = normalize_reader_prediction(prediction)
    ids = prediction.get("used_fact_ids", [])
    if not isinstance(ids, list) or any(not isinstance(v, str) for v in ids) or not set(ids) <= {f"f{i}" for i in range(1, 21)}:
        raise ValueError("Original list Reader failed an additional citation contract")
    answer = normalized["answer"]; custom = lexical_metrics(answer, q["reference"])
    return {"status": "deferred_postbatch", "reader_resolution_authorized": True,
            "source_choice": "original_firstpass_raw_only_not_old_recovery",
            "normalized_answer": answer, "answer_sha256": text_sha(answer), "normalization_proof": proof,
            "F1": {"official_locomo_f1": official(answer, q["reference"], q["category"]),
                   "custom_token_f1": custom["token_f1"], "custom_exact_match": custom["exact_match"]},
            "reader_model_calls": 0, "main_firstpass_metrics_remain_null": True}


def get_reader_raw(ci, arm, row, expected, capture):
    key = request_digest(expected)
    if row["status"] == "failed":
        jp = ROOT / "runs/sdk20_modules_recovery_v1/jobs" / arm / (row["query_id"] + ".json")
        job = capture(jp, sealed=True)
        if (job["source_qa"]["sha256"] != legacy.sha(row["_path"])
                or job["expected_reader_signature"] != expected or job["expected_reader_hash"] != key):
            raise ValueError("Prior capture does not bind this original failed Reader")
        raw = capture(job["old_raw"]["path"], expected=job["old_raw"]["sha256"])
        return Path(job["old_raw"]["path"]), raw, {"prior_original_raw_capture": pointer(jp)}
    if arm == ARMS[1]:
        events = [e for e in row["api_events"] if e["stage"] == "qa_reader"]
        if len(events) != 1 or events[0]["request_hash"] != key or events[0].get("error_type") is not None:
            raise ValueError("Typed Reader lacks one exact successful raw event")
        event = events[0]; path = Path(event["cache_path"])
        if path.resolve() != ROOT / f"runs/sdk20_modules_v1/conv_{ci}/api/api_cache/{key}.json":
            raise ValueError("Typed Reader raw path is outside its original conversation")
        return path, capture(path, expected=event["cache_sha256"]), {"reader_event": deepcopy(event)}
    origin = row["source"]; prior = capture(origin["path"], expected=origin["sha256"])
    if origin["kind"] == "firstpass":
        path = sources.BASE / f"conv_{ci}/api_cache/{key}.json"
        raw = sources._old_read(path)
        capture(path)
    elif origin["kind"] == "technical_recovery":
        event = [e for e in prior["request_events"] if e["stage"] == "qa_reader"]
        if len(event) != 1 or event[0]["request_hash"] != key or event[0].get("status") != "valid":
            raise ValueError("Historical SDK20 recovery Reader event is not exact")
        path = sources.RECOVERY / f"transport/api_cache/{key}.json"; raw = capture(path)
        if digest(raw["response"]) != event[0]["response_sha256"]:
            raise ValueError("Historical recovered Reader response changed")
    else: raise ValueError("Unreviewed historical SDK20 source kind")
    return path, raw, {"historical_qa": deepcopy(origin)}


def supplemental(capture):
    """One prior global conflict, deferred until the new comparison batch ends."""
    comp = capture(GLOBAL_COMPAT, sealed=True)
    pending = comp["retained_failures"]
    if len(pending) != 1 or (pending[0]["conversation"], pending[0]["method"], pending[0]["query_id"]) != (2, "dated_unmerged385", "c2_q74"):
        raise ValueError("Global supplemental population differs from the one retained conflict")
    side = capture(pending[0]["sidecar"]["path"], expected=pending[0]["sidecar"]["sha256"], sealed=True)
    refs = side["original"]["artifacts"]
    reader = capture(refs["reader"]["path"], expected=refs["reader"]["sha256"], sealed=True)
    judge = capture(refs["judge"]["path"], expected=refs["judge"]["sha256"], sealed=True)
    rawref = side["source_replay"]["source_evidence"]["raw"]
    raw = capture(rawref["path"], expected=rawref["sha256"])
    answer = reader["result"]["prediction"]["answer"]
    if (reader["status"] != "complete" or judge["status"] != "failed"
            or judge["error"]["type"] != "InvalidJudgeLabel" or not isinstance(answer, str)
            or side["status"] != "retained_failure"
            or derive_label(judge["raw_invalid_label"])["rejection_reason"] != "conflicting_exact_decision_label_anywhere"
            or raw["request"] != judge_signature(reader["question"], reader["reference"], answer)):
        raise ValueError("Global conflict or its fixed Reader/Judge signature changed")
    return {"schema": SCHEMA + "-job", "job_id": "global_conflict/dated_unmerged385/c2_q74",
            "phase_kind": "postbatch_supplement", "status": "deferred_postbatch", "conversation": 2,
            "method": "dated_unmerged385", "query_id": "c2_q74", "category": reader["category"],
            "question": reader["question"], "reference": reader["reference"], "answer": answer,
            "answer_sha256": text_sha(answer), "reader_status": "complete", "pending_original_list": None,
            "F1": deepcopy(reader["result"]["metrics"]), "old_judge": None,
            "source": {"original_qa": refs["qa"], "reader": refs["reader"], "original_judge": refs["judge"],
                       "original_judge_raw": rawref, "compat_sidecar": pending[0]["sidecar"],
                       "compat_summary": pointer(GLOBAL_COMPAT)},
            "old_judge_failure": deepcopy(judge["error"]), "main_cohort_member": False}


def collect():
    tracked = {}
    def capture(path, *, expected=None, sealed=False):
        path = Path(path).resolve(); ref = pointer(path)
        if expected is not None and ref["sha256"] != expected: raise ValueError("Historical source changed: " + str(path))
        if str(path) in tracked and tracked[str(path)] != ref["sha256"]: raise ValueError("Source changed while preparing")
        tracked[str(path)] = ref["sha256"]
        return verify(path) if sealed else read(path)
    first = capture(SUMMARY, expected=SUMMARY_SHA)
    history = capture(HISTORICAL_RECOVERY, expected=HISTORICAL_RECOVERY_SHA)
    config = capture(CONFIG, sealed=True)
    run = Path(first["run_dir"]); m = legacy.manifest(run); capture(run / "run_manifest.json")
    if first["expected_slots"] != 4620 or not first["all_terminal_slots_valid"] or first["integrity_errors"] or first["protocol_sha256"] != m["protocol_sha256"]:
        raise ValueError("Original complete first-pass summary is not admitted")
    # Reuse the sealed summary but re-read only this two-arm source population;
    # no C-arm QA, native indexes or irrelevant historical experiment is rerun.
    official, official_provenance = verified_official_metric()
    questions = [{**q, "conversation": ci} for ci, conv in enumerate(legacy.load_data()) for q in legacy.queries(conv, ci)]
    outcomes = {(r["method"], r["query_id"]): r for r in first["outcomes"]}
    banks, jobs = {}, []
    for ci in range(10):
        bank, bh = legacy.get_bank(run, ci, "session", m)
        capture(run / f"conv_{ci}/session/bank.json", expected=bh)
        capture(run / f"conv_{ci}/source_manifest.json", expected=m["source_manifests"][str(ci)])
        banks[ci] = bank, bh, {f["id"]: f for f in bank["facts"]}
    for q in questions:
        ci = q["conversation"]; bank, bh, byid = banks[ci]
        for arm in ARMS:
            outcome = outcomes[arm, q["id"]]; path = Path(outcome["artifact"]["path"])
            row = capture(path, expected=outcome["artifact"]["sha256"], sealed=True)
            summary_tool.validate_row(row, q, arm, m["protocol_sha256"])
            sp = run / f"conv_{ci}/selection/{arm}/{q['id']}.json"; selection = capture(sp, expected=first["input_hashes"][str(sp)], sealed=True)
            if selection["status"] != "complete" or selection["selected_ids"] != row["selected_ids"] or selection["bank_sha256"] != bh:
                raise ValueError("Original20-fact selection differs")
            if arm == ARMS[0]:
                original = sources.baseline_row(ci, q)
                if any(row[k] != original[k] for k in ("status", "result", "selected_ids", "error", "source", "reference_date")):
                    raise ValueError("Historical baseline outcome changed")
            facts = [{k: byid[i].get(k) for k in ("id", "text", "observation_date")} for i in row["selected_ids"]]
            request_job = {"question": q["question"], "reference": q["reference"], "reference_date": row["reference_date"], "selected_facts": facts}
            sig = recovery.expected_reader(request_job)
            rawpath, raw, lineage = get_reader_raw(ci, arm, {**row, "_path": str(path)}, sig, capture)
            if raw["request"] != sig: raise ValueError("Original complete Reader payload differs")
            prediction = reader_prediction(raw)
            fields = answer_fields(row, outcome, prediction)
            jobs.append({"schema": SCHEMA + "-job", "job_id": arm + "/" + q["id"], "phase_kind": "firstpass",
                "conversation": ci, "method": arm, "query_id": q["id"], "category": q["category"],
                "question": q["question"], "reference": q["reference"], **fields,
                "source": {"original_qa": outcome["artifact"], "original_reader_raw": pointer(rawpath),
                           "source_selection": pointer(sp), "source_bank": pointer(run / f"conv_{ci}/session/bank.json"),
                           "reader_request_hash": request_digest(sig), "reader_signature": sig, **lineage},
                "original_error": deepcopy(row["error"]), "selected_ids": deepcopy(row["selected_ids"]),
                "reference_date": row["reference_date"], "main_cohort_member": True,
                "deferred_reader_resolution": deferred_resolution(prediction, q, official) if fields["status"] == "blocked_original_reader" else None})
    supplement = supplemental(capture)
    counts = {arm: dict(Counter(j["status"] for j in jobs if j["method"] == arm)) for arm in ARMS}
    if counts != EXPECTED or len(jobs) != 3080: raise ValueError("Main first-pass source population changed")
    common = sorted(set(j["query_id"] for j in jobs if j["method"] == ARMS[0] and j["status"] == "ready") &
                    set(j["query_id"] for j in jobs if j["method"] == ARMS[1] and j["status"] == "ready"))
    if len(common) != 1512: raise ValueError("Historical paired string population changed")
    if any(legacy.sha(p) != h for p, h in tracked.items()): raise ValueError("Historical inputs changed during source capture")
    return {"schema": SCHEMA, "created_utc": now(), "phase_id": "sdk20_firstpass_fixed_reader_v1", "phase_kind": "firstpass",
            "source_summary": pointer(SUMMARY), "source_protocol_sha256": m["protocol_sha256"],
            "historical_recovery_summary": pointer(HISTORICAL_RECOVERY), "historical_recovery_excluded_from_main": True,
            "historical_recovery_complete_counts": {a: history["arms"][a]["status_counts"]["complete"] for a in ARMS},
            "plain_text_config": pointer(CONFIG), "judge_prompt": config["prompts"]["judge_template"],
            "user_override": "Latest user request fixes historical Reader answers and changes only Judge; overrides the old config sentence requiring Reader regeneration for old baselines.",
            "arms": list(ARMS), "cohort_per_arm": 1540, "expected_slots": 3080,
            "status_counts": counts, "ready_judges": 3042, "blocked_original_readers": 38,
            "common_ready_query_ids": common, "jobs": jobs, "supplemental_jobs": [supplement],
            "pending_list_policy": "Main remains blocked/null. Postbatch uses the original firstpass list only, joined with one newline under the disclosed default; no old recovery3 or new Reader generation.",
            "official_metric_provenance": official_provenance,
            "input_sha256": tracked, "code_sha256": {p: legacy.sha(ROOT / p) for p in CODE},
            "model_calls": 0, "native_clients": 0, "original_mutations": 0}


def validate_main_jobs(value, original_outcomes):
    keys = {(j["method"], j["query_id"]) for j in value["jobs"]}
    expected_keys = {k for k in original_outcomes if k[0] in ARMS}
    actual_counts = {arm: dict(Counter(j["status"] for j in value["jobs"] if j["method"] == arm)) for arm in ARMS}
    if (len(keys) != 3080 or len(value["jobs"]) != 3080 or keys != expected_keys or actual_counts != EXPECTED
            or value["ready_judges"] != 3042 or value["blocked_original_readers"] != 38):
        raise ValueError("Job identities or first-pass status counts differ")


def validate(path, *, check_sources=True, conversation=None):
    value = verify(path)
    if value.get("schema") != SCHEMA or value.get("phase_kind") != "firstpass" or value.get("expected_slots") != 3080 or value.get("status_counts") != EXPECTED:
        raise ValueError("Not the frozen main two-arm first-pass source set")
    if value["source_summary"]["sha256"] != SUMMARY_SHA or value["historical_recovery_summary"]["sha256"] != HISTORICAL_RECOVERY_SHA:
        raise ValueError("Historical source summary changed")
    for p, h in value["code_sha256"].items(): check_pointer({"path": ROOT / p, "sha256": h})
    if check_sources:
        for p, h in value["input_sha256"].items(): check_pointer({"path": p, "sha256": h})
    first = read(check_pointer(value["source_summary"]))
    original_outcomes = {(r["method"], r["query_id"]): r for r in first["outcomes"]}
    official, provenance = verified_official_metric()
    if provenance != value["official_metric_provenance"]: raise ValueError("Official metric provenance differs")
    validate_main_jobs(value, original_outcomes)
    for job in value["jobs"]:
        if conversation is not None and job["conversation"] != conversation: continue
        refs = job["source"]
        original = verify(check_pointer(refs["original_qa"]))
        raw = read(check_pointer(refs["original_reader_raw"]))
        if raw["request"] != refs["reader_signature"] or request_digest(raw["request"]) != refs["reader_request_hash"]:
            raise ValueError("Job source Reader signature differs")
        if (job.get("job_id") != job["method"] + "/" + job["query_id"] or job.get("phase_kind") != "firstpass"
                or job.get("main_cohort_member") is not True
                or any(job[k] != original[k] for k in ("question", "reference", "conversation", "method", "query_id", "category", "selected_ids", "reference_date"))):
            raise ValueError("Job identity or evidence order differs from source")
        outcome = original_outcomes[job["method"], job["query_id"]]
        if refs["original_qa"] != outcome["artifact"]: raise ValueError("Job source is not the original first-pass outcome")
        prediction = reader_prediction(raw)
        expected = answer_fields(original, outcome, prediction)
        if any(job[k] != v for k, v in expected.items()): raise ValueError("Job answer, blocked list or original scores changed")
        normalized = deferred_resolution(prediction, job, official) if job["status"] == "blocked_original_reader" else None
        if job.get("deferred_reader_resolution") != normalized: raise ValueError("Deferred list normalization changed")
    def checked_capture(p, *, expected=None, sealed=False):
        ref = {"path": p, "sha256": expected or value["input_sha256"][str(Path(p).resolve())]}
        p = check_pointer(ref)
        return verify(p) if sealed else read(p)
    if value["supplemental_jobs"] != [supplemental(checked_capture)]: raise ValueError("Global deferred conflict changed")
    return value


def prepare(output_dir):
    output = Path(output_dir).resolve()
    if output.parent != ROOT / "runs" or output.exists(): raise ValueError("Use a new direct runs namespace")
    with offline(): value = collect()
    save_new(output / "source_manifest.json", value)
    return {"source_manifest": pointer(output / "source_manifest.json"), "ready": 3042, "blocked": 38, "supplemental_deferred": 1, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "check")); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.output_dir is None: parser.error("prepare requires --output-dir")
        result = prepare(args.output_dir)
    else:
        if args.source_manifest is None: parser.error("check requires --source-manifest")
        with offline(): value = validate(args.source_manifest)
        result = {"valid": True, "ready": value["ready_judges"], "blocked": value["blocked_original_readers"], "model_calls": 0}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
