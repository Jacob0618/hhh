"""Frozen original Mem0 evidence for a fresh plain Reader/Judge comparison.

No native store, retrieval, embedding, model call, or old answer cache is used.
Every old selection (including38 Reader failures) becomes a ready fresh job.
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
from tools import prepare_sdk20_judge_retest_sources_v1 as historical
from tools.run_sdk20_modules import digest, save_new, verify, now
from tools.check_experiment_health import shared_read
from tools.summarize_pilot400 import verified_official_metric
from jev_mem.data import load_data, queries, session_envelopes
from jev_mem.evaluation_plain_text_v1 import reader_input

SCHEMA = "sdk20-plain-qa-sources-v1"
SOURCE = ROOT / "runs/sdk20_judge_retest_sources_v1/source_manifest.json"
SOURCE_SHA = "676fa8dc6c4f67f555bed07ae252974dc3bf96141b5eaf15291867684ba1af25"
OLD_RUN = ROOT / "runs/sdk20_modules_v1"
ARMS = ("baseline_sdk20", "fact_types_sdk20")
FACT_FIELDS = ("id", "text", "observation_date")
CODE = ("tools/sdk20_plain_qa_sources_v1.py", "tools/prepare_sdk20_judge_retest_sources_v1.py",
        "tools/run_sdk20_modules.py", "jev_mem/data.py", "jev_mem/evaluation_plain_text_v1.py",
        "jev_mem/evaluation.py", "jev_mem/locomo_official_metrics.py", "tools/summarize_pilot400.py")
POLICY = {"old_answer_reused": False, "old_qa_cache_roots": [],
          "retrieval": "reuse_exact_original_selected_ids_and_order_no_search",
          "population": "both_original1540_questions_including_all38_original_Reader_failures",
          "new_reader": "brief_answer_plain_text_v1", "new_judge": "user_reference_contains_binary_v1",
          "evidence_budget": 20, "fact_bank": "original272_sessions2376_facts_shared_by_both_arms",
          "typing": "existing_frozen_fact_tags_and_query_types_no_new_JEV"}


def sha(path):
    return hashlib.sha256(shared_read(Path(path))).hexdigest()


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    return json.loads(shared_read(Path(path)))


def _check_seal(value):
    if value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}):
        raise ValueError("Original sealed artifact changed")


class BoundInputs:
    def __init__(self, summary):
        self.allowed = summary["input_hashes"]
        self.hashes = {}
        self.cache = {}

    def get(self, path, *, expected=None, sealed=False):
        path = Path(path).resolve(); key = str(path)
        bound = self.allowed.get(key)
        if expected is None:
            expected = bound
        if expected is None or (bound is not None and bound != expected):
            raise ValueError("Source not bound by original summary/manifest")
        if key not in self.cache:
            raw = shared_read(path)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Frozen source bytes changed: " + key)
            self.cache[key] = json.loads(raw); self.hashes[key] = expected
        elif self.hashes[key] != expected:
            raise ValueError("Conflicting source hash")
        result = self.cache[key]
        if sealed:
            _check_seal(result)
        return result

    def ref(self, path):
        path = str(Path(path).resolve())
        return {"path": path, "sha256": self.hashes[path]}


def job_from_selection(old_job, bank, bank_ref, selection, selection_ref, query, *, lineage):
    """Pure projection: old answers and metrics cannot enter the new job."""
    ci, arm, qid = old_job["conversation"], old_job["method"], old_job["query_id"]
    if arm not in ARMS or old_job["job_id"] != arm + "/" + qid:
        raise ValueError("Wrong original arm/job identity")
    if any(old_job[k] != query[qk] for k, qk in (("question", "question"), ("reference", "reference"),
                                               ("category", "category"), ("query_id", "id"))):
        raise ValueError("Question/gold/category changed")
    if type(old_job["reference"]) not in (str, int):
        raise ValueError("Original reference type changed")
    if any(selection[k] != old_job[k] for k in ("conversation", "method", "query_id", "question", "reference", "category")):
        raise ValueError("Selection identity differs")
    ids = selection["selected_ids"]
    if (selection["status"] != "complete" or ids != old_job["selected_ids"] or len(ids) != 20
            or len(set(ids)) != 20 or selection["bank_sha256"] != bank_ref["sha256"]):
        raise ValueError("Original20-fact selection/order changed")
    byid = {f["id"]: f for f in bank["facts"]}
    if len(byid) != len(bank["facts"]) or not set(ids) <= byid.keys():
        raise ValueError("Fact bank IDs missing/duplicate")
    facts = [{k: byid[i].get(k) for k in FACT_FIELDS} for i in ids]
    payload = reader_input(old_job["question"], facts, reference_date=old_job["reference_date"])
    original_payload = json.loads(old_job["source"]["reader_signature"]["body"]["contents"][0]["parts"][0]["text"])
    if payload != original_payload:
        raise ValueError("Fresh Reader evidence input differs from original exact payload")
    return {"schema": SCHEMA + "-job", "job_id": arm + "/" + qid, "status": "ready",
            "conversation": ci, "method": arm, "query_id": qid, "category": old_job["category"],
            "question": old_job["question"], "reference": old_job["reference"],
            "reference_date": old_job["reference_date"], "selected_ids": deepcopy(ids),
            "selected_facts": facts, "reader_input_sha256": digest(payload),
            "source": {"original_job_sha256": digest(old_job), "source_selection": selection_ref,
                       "source_bank": bank_ref, **deepcopy(lineage)}}


def build_source_payload(*, check_sources=True, conversation=None):
    if conversation is not None and (type(conversation) is not int or conversation not in range(10)):
        raise ValueError("Invalid conversation")
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Frozen original source manifest changed")
    parent = historical.validate(SOURCE, check_sources=check_sources, conversation=conversation)
    if parent["arms"] != list(ARMS) or len(parent["jobs"]) != 3080:
        raise ValueError("Original two-arm population changed")
    summary = read(parent["source_summary"]["path"])
    bound = BoundInputs(summary)
    m = bound.get(OLD_RUN / "run_manifest.json", expected=parent["input_sha256"][str(OLD_RUN / "run_manifest.json")])
    if m["protocol_sha256"] != parent["source_protocol_sha256"]:
        raise ValueError("Original SDK20 protocol differs")
    data = load_data(); questions = {}; envelopes = {}
    for ci, item in enumerate(data):
        envelopes[ci] = list(session_envelopes(item))
        questions.update({q["id"]: {**q, "conversation": ci} for q in queries(item, ci)})
    if len(questions) != 1540 or sum(map(len, envelopes.values())) != 272:
        raise ValueError("Original1540 questions/272 sessions changed")
    banks, bank_info = {}, {}
    for ci in range(10):
        bp = OLD_RUN / f"conv_{ci}/session/bank.json"
        bank = bound.get(bp); bh = bound.ref(bp)
        if bank["conversation"] != ci or bank["kind"] != "session":
            raise ValueError("Original session bank identity differs")
        tags = {}
        for fact in bank["facts"]:
            tp = OLD_RUN / f"conv_{ci}/tags/{fact['id']}.json"
            tag = bound.get(tp, sealed=True)
            if (tag["status"] != "complete" or tag["conversation"] != ci or tag["fact_id"] != fact["id"]
                    or tag["protocol_sha256"] != m["protocol_sha256"] or tag["bank_sha256"] != bh["sha256"]
                    or tag["fact_text_sha256"] != digest(fact["text"])):
                raise ValueError("Original fact tag provenance differs")
            tags[fact["id"]] = bound.ref(tp)
        banks[ci] = bank, bh
        bank_info[str(ci)] = {"bank": bh, "facts": len(bank["facts"]), "sessions": len(envelopes[ci]),
                              "fact_tags": tags, "fact_tags_sha256": digest(tags)}
    if sum(x["facts"] for x in bank_info.values()) != 2376:
        raise ValueError("Original2376 shared facts changed")
    jobs = []
    for old in parent["jobs"]:
        ci, arm, qid = old["conversation"], old["method"], old["query_id"]
        q = questions[qid]
        if ci != q["conversation"] or old["reference_date"] != envelopes[ci][-1]["observation_date"]:
            raise ValueError("Original conversation/date changed")
        bank, bh = banks[ci]; sp = Path(old["source"]["source_selection"]["path"])
        selection = bound.get(sp, expected=old["source"]["source_selection"]["sha256"], sealed=True)
        if selection["protocol_sha256"] != m["protocol_sha256"]:
            raise ValueError("Selection protocol differs")
        rp = OLD_RUN / f"conv_{ci}/retrieval/session/{qid}.json"
        retrieval = bound.get(rp, sealed=True)
        if (retrieval["status"] != "complete" or retrieval["protocol_sha256"] != m["protocol_sha256"]
                or retrieval["bank_sha256"] != bh["sha256"] or not retrieval["baseline_replay"]["matched"]):
            raise ValueError("Frozen native retrieval capture differs")
        if arm == ARMS[0]:
            if (selection["trace"] != {"method": "original_sdk20"}
                    or selection["selected_ids"] != retrieval["selected_ids"]
                    or selection["source"] != retrieval["baseline_replay"]["source"]):
                raise ValueError("Native baseline20 order differs")
            lineage = {"retrieval": bound.ref(rp), "query_types": None, "fact_tags_sha256": None}
        else:
            qp = OLD_RUN / f"conv_{ci}/query_types/{qid}.json"; qt = bound.get(qp, sealed=True)
            if (qt["status"] != "complete" or qt["protocol_sha256"] != m["protocol_sha256"]
                    or qt["conversation"] != ci or qt["query_id"] != qid or qt["question"] != old["question"]
                    or selection["retrieval_sha256"] != bound.ref(rp)["sha256"]
                    or selection["trace"]["query_type_scores"] != qt["scores"]
                    or selection["trace"]["selected_ids"] != selection["selected_ids"]
                    or selection["trace"]["base20_ids"] != retrieval["selected_ids"]):
                raise ValueError("Frozen type selection/query provenance differs")
            lineage = {"retrieval": bound.ref(rp), "query_types": bound.ref(qp),
                       "fact_tags_sha256": bank_info[str(ci)]["fact_tags_sha256"]}
        jobs.append(job_from_selection(old, bank, bh, selection, bound.ref(sp), q, lineage=lineage))
    expected_keys = {(a, qid) for a in ARMS for qid in questions}
    if len(jobs) != 3080 or {(j["method"], j["query_id"]) for j in jobs} != expected_keys:
        raise ValueError("Full two-arm1540 population differs")
    _, metric = verified_official_metric()
    if metric != parent["official_metric_provenance"]:
        raise ValueError("Official metric provenance changed")
    return {"schema": SCHEMA, "phase_kind": "fresh_plain_reader_and_judge", "policy": POLICY,
            "source_manifest": {"path": str(SOURCE), "sha256": SOURCE_SHA},
            "source_protocol_sha256": m["protocol_sha256"], "arms": list(ARMS),
            "cohort_per_arm": 1540, "expected_slots": 3080, "ready_jobs": 3080,
            "old_failure_slots_included": dict(Counter(j["method"] for j in parent["jobs"] if j["status"] != "ready")),
            "original_sessions": 272, "original_shared_facts": 2376, "banks": bank_info,
            "jobs": jobs, "official_metric_provenance": metric,
            "precomputation_cost_lineage": {
                "original_summary": deepcopy(parent["source_summary"]),
                "original_run_manifest": bound.ref(OLD_RUN / "run_manifest.json"),
                "policy": "Existing writing, embedding, fact typing and query selection are reused; zero new calls here does not imply zero cold-start cost. Do not recount old ledger usage as new QA cost.",
                "prior_qa_usage_not_in_new_run": True},
            "input_sha256": bound.hashes, "code_sha256": {p: sha(ROOT / p) for p in CODE},
            "model_calls": 0, "native_clients": 0, "original_mutations": 0, "old_qa_cache_roots": []}


def validate_rebuild(path, *, check_sources=True, conversation=None):
    """Explicit slow reproducibility check; never the routine runtime validator."""
    value = verify(path)
    if value.get("schema") != SCHEMA or value.get("policy") != POLICY:
        raise ValueError("Wrong fresh plain source schema/policy")
    expected = build_source_payload(check_sources=check_sources, conversation=conversation)
    if {k: v for k, v in value.items() if k not in ("created_utc", "content_sha256")} != expected:
        raise ValueError("Fresh source projection/lineage differs")
    return value


def validate(path, *, check_sources=True, conversation=None):
    """Validate the prepared projection without replaying old Reader/F1/captures.

    The immutable parent and original summary anchor all projected inputs. Byte
    hashes remain checked, but historical raw responses and banks are not parsed.
    """
    if conversation is not None and (type(conversation) is not int or conversation not in range(10)):
        raise ValueError("Invalid conversation")
    value = verify(path)
    if (value.get("schema") != SCHEMA or value.get("policy") != POLICY
            or value.get("phase_kind") != "fresh_plain_reader_and_judge"
            or value.get("source_manifest") != {"path": str(SOURCE), "sha256": SOURCE_SHA}
            or value.get("code_sha256") != {p: sha(ROOT / p) for p in CODE}):
        raise ValueError("Fresh source protocol/code differs")
    raw_parent = shared_read(SOURCE)
    if hashlib.sha256(raw_parent).hexdigest() != SOURCE_SHA:
        raise ValueError("Frozen original source manifest changed")
    parent = json.loads(raw_parent); _check_seal(parent)
    summary_path = Path(parent["source_summary"]["path"])
    raw_summary = shared_read(summary_path)
    if (summary_path != historical.SUMMARY or hashlib.sha256(raw_summary).hexdigest() != historical.SUMMARY_SHA
            or parent["source_summary"]["sha256"] != historical.SUMMARY_SHA):
        raise ValueError("Frozen original summary differs")
    summary = json.loads(raw_summary)
    required = {}; expected_jobs = []
    def ref(p):
        key = str(Path(p).resolve())
        expected = summary["input_hashes"].get(key, parent["input_sha256"].get(key))
        if expected is None:
            raise ValueError("Projection input lacks a frozen original hash")
        required[key] = expected
        return {"path": key, "sha256": expected}
    mr = ref(OLD_RUN / "run_manifest.json")
    questions = {}; envelopes = {}
    for ci, item in enumerate(load_data()):
        envelopes[ci] = list(session_envelopes(item))
        questions.update({q["id"]: {**q, "conversation": ci} for q in queries(item, ci)})
    if len(questions) != 1540 or sum(map(len, envelopes.values())) != 272:
        raise ValueError("Frozen cohort/session count differs")
    banks = {}
    for ci in range(10):
        tag_dir = OLD_RUN / f"conv_{ci}/tags"
        tag_paths = [Path(p) for p in summary["input_hashes"] if Path(p).parent == tag_dir]
        tags = {p.stem: ref(p) for p in tag_paths}
        n = summary["native_tag_counts"][str(ci)]
        if len(tags) != n:
            raise ValueError("Original fact tag population differs")
        banks[str(ci)] = {"bank": ref(OLD_RUN / f"conv_{ci}/session/bank.json"), "facts": n,
                          "sessions": len(envelopes[ci]), "fact_tags": tags, "fact_tags_sha256": digest(tags)}
    for old in parent["jobs"]:
        ci, arm, qid = old["conversation"], old["method"], old["query_id"]
        if arm not in ARMS or qid not in questions or questions[qid]["conversation"] != ci:
            raise ValueError("Original job identity differs")
        q = questions[qid]
        if any(old[k] != q[k] for k in ("question", "reference", "category")):
            raise ValueError("Original question/gold/category differs")
        if type(old["reference"]) not in (str, int) or old["reference_date"] != envelopes[ci][-1]["observation_date"]:
            raise ValueError("Original reference type/date differs")
        ids = old["selected_ids"]
        payload = json.loads(old["source"]["reader_signature"]["body"]["contents"][0]["parts"][0]["text"])
        if (len(ids) != 20 or len(set(ids)) != 20 or len(payload["facts"]) != 20
                or not set(ids) <= banks[str(ci)]["fact_tags"].keys()):
            raise ValueError("Original20-fact identity/budget differs")
        facts = [{"id": fid, "text": f["fact"], "observation_date": f["observation_date"]}
                 for fid, f in zip(ids, payload["facts"])]
        if reader_input(old["question"], facts, reference_date=old["reference_date"]) != payload:
            raise ValueError("Original Reader input protocol differs")
        sp = ref(OLD_RUN / f"conv_{ci}/selection/{arm}/{qid}.json")
        bp = banks[str(ci)]["bank"]
        if any(old["source"]["source_selection"][k] != sp[k] for k in ("path", "sha256")):
            raise ValueError("Original selection source differs")
        if any(old["source"]["source_bank"][k] != bp[k] for k in ("path", "sha256")):
            raise ValueError("Original shared bank source differs")
        lineage = {"retrieval": ref(OLD_RUN / f"conv_{ci}/retrieval/session/{qid}.json"),
                   "query_types": ref(OLD_RUN / f"conv_{ci}/query_types/{qid}.json") if arm == ARMS[1] else None,
                   "fact_tags_sha256": banks[str(ci)]["fact_tags_sha256"] if arm == ARMS[1] else None}
        expected_jobs.append({"schema": SCHEMA + "-job", "job_id": arm + "/" + qid, "status": "ready",
            "conversation": ci, "method": arm, "query_id": qid, "category": old["category"],
            "question": old["question"], "reference": old["reference"], "reference_date": old["reference_date"],
            "selected_ids": ids, "selected_facts": facts, "reader_input_sha256": digest(payload),
            "source": {"original_job_sha256": digest(old), "source_selection": sp, "source_bank": bp, **lineage}})
    if (len(expected_jobs) != 3080 or {(j["method"], j["query_id"]) for j in expected_jobs}
            != {(a, qid) for a in ARMS for qid in questions}):
        raise ValueError("Full3080-slot population differs")
    _, metric = verified_official_metric()
    expected = {"schema": SCHEMA, "phase_kind": "fresh_plain_reader_and_judge", "policy": POLICY,
        "source_manifest": {"path": str(SOURCE), "sha256": SOURCE_SHA},
        "source_protocol_sha256": parent["source_protocol_sha256"], "arms": list(ARMS),
        "cohort_per_arm": 1540, "expected_slots": 3080, "ready_jobs": 3080,
        "old_failure_slots_included": dict(Counter(j["method"] for j in parent["jobs"] if j["status"] != "ready")),
        "original_sessions": 272, "original_shared_facts": 2376, "banks": banks, "jobs": expected_jobs,
        "official_metric_provenance": metric, "precomputation_cost_lineage": {
            "original_summary": deepcopy(parent["source_summary"]), "original_run_manifest": mr,
            "policy": "Existing writing, embedding, fact typing and query selection are reused; zero new calls here does not imply zero cold-start cost. Do not recount old ledger usage as new QA cost.",
            "prior_qa_usage_not_in_new_run": True},
        "input_sha256": required, "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "model_calls": 0, "native_clients": 0, "original_mutations": 0, "old_qa_cache_roots": []}
    if (sum(b["facts"] for b in banks.values()) != 2376
            or metric != parent["official_metric_provenance"]
            or {k: v for k, v in value.items() if k not in ("created_utc", "content_sha256")} != expected):
        raise ValueError("Prepared projection/lineage differs from frozen original evidence")
    if check_sources:
        for p, h in required.items():
            if sha(p) != h:
                raise ValueError("Frozen source bytes changed: " + p)
    return value


def prepare(output_dir):
    output = Path(output_dir).resolve()
    if output.parent != (ROOT / "runs").resolve() or output.exists():
        raise ValueError("Use a fresh direct child of runs")
    with historical.offline():
        value = build_source_payload()
    output.mkdir(parents=False, exist_ok=False)
    save_new(output / "source_manifest.json", {"created_utc": now(), **value})
    return {"source_manifest": pointer(output / "source_manifest.json"), "ready": 3080,
            "old_failure_slots_included": value["old_failure_slots_included"], "model_calls": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "check", "rebuild-check"))
    p.add_argument("--output-dir", type=Path); p.add_argument("--source-manifest", type=Path)
    a = p.parse_args()
    if a.phase == "prepare":
        if a.output_dir is None: p.error("prepare requires --output-dir")
        out = prepare(a.output_dir)
    else:
        if a.source_manifest is None: p.error("check requires --source-manifest")
        with historical.offline():
            value = (validate_rebuild if a.phase == "rebuild-check" else validate)(a.source_manifest)
        out = {"valid": True, "ready": len(value["jobs"]), "model_calls": 0}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
