"""Thin native SDK20 + unchanged plain Reader/Judge runner for the fixed400.

prepare/check/recover are offline. retrieve invokes only native retrieval and
QUERY embeddings; qa invokes the unchanged Reader/Judge implementation. Each arm
has its own admitted ten-bank namespace. No historical Reader/Judge is inherited.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import regroup_refinement_qa_sources_v1 as sources
from tools import run_mem0_regroup_plain_qa_v1 as native_qa
from tools import run_sdk20_plain_qa_v1 as plain
from tools.run_sdk20_modules import digest, save_new, sha, now, verify
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.summarize_pilot400 import verified_official_metric
from jev_mem import qa_embedding_cache_pack_v1 as embedding_pack
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.evaluation_plain_text_v1 import reader_input
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem.paced_embedding_api import DEFAULT_GATE
from jev_mem.data import SOURCE_HASH

SCHEMA = "regroup-refinement-plain-qa400-v1"
POLICY = {"slots_per_arm": 400, "conversation_workers": 8, "qa_threads_per_conversation": 4,
          "selection": "unchanged_native_SDK20_top_k20_rerank_false_80_pool",
          "retrieval_within_process": "sequential", "facts": 20,
          "reader_judge_implementation": "run_sdk20_plain_qa_v1.one_unchanged",
          "reader_and_f1": "persist_before_judge", "json_output": False,
          "failure_policy": "all_firstpass_terminal_then_one_uniform_technical_recovery",
          "unknown_policy": "hard_refuse_implicit_redispatch",
          "cache_policy": "only_exact_query_embedding_pack_no_historical_reader_or_judge",
          "reference_comparison": "historical_global_same400_firstpass_compat_not_mixed_final_recovery",
          "no_resampling_or_answer_based_retry": True}
CODE = tuple(dict.fromkeys(("tools/run_regroup_refinement_qa_v1.py", "tools/regroup_refinement_qa_sources_v1.py",
                           "tools/run_regroup_refinement_writer_v1.py", "jev_mem/pilot400.py",
                           *native_qa.CODE, *plain.CODE)))
RUNTIME = (".venv/Lib/site-packages/fastembed/sparse/bm25.py",
           ".venv/Lib/site-packages/fastembed/sparse/sparse_text_embedding.py",
           ".venv/Lib/site-packages/qdrant_client/fastembed_common.py",
           ".venv/Lib/site-packages/qdrant_client/qdrant_fastembed.py")


def prepare(run_dir, gate_path):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT / "runs").resolve() or run.exists():
        raise FileExistsError("Fresh direct runs namespace required")
    gate, admitted, resource = sources.admit_gate(gate_path)
    config = sources.protocol_config()
    rows, dates = sources.cohort()
    _, provenance = verified_official_metric()
    plan = embedding_pack.audit_sources()
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / "prepare_intent.json", {"schema": SCHEMA, "gate": sources.pointer(gate_path), "model_calls": 0})
    arm = gate["arm"]
    banks = {str(ci): {arm: sources.copy_bank(run, ci, arm, admitted[(ci, arm)])} for ci in range(10)}
    cache = embedding_pack.create_pack(run, plan)
    sdkroot = Path(importlib.metadata.distribution("mem0ai").locate_file("mem0"))
    value = {"schema": SCHEMA, "created_utc": now(), "policy": POLICY, "arm": arm,
             "phase_kind": "firstpass", "phase_id": SCHEMA + "_" + arm + "_firstpass",
             "dataset_sha256": SOURCE_HASH, "cohort_manifest": sources.pointer(sources.COHORT),
             "cohort_sha256": digest(rows), "reference_dates": dates,
             "query_ids": {ci: [q["id"] for q in qs] for ci, qs in rows.items()},
             "banks": banks, "gate": sources.pointer(gate_path), "plain_text_config": sources.pointer(sources.CONFIG),
             "evaluation_protocol": config["evaluation_protocol"], "resources": resource,
             "embedding_gate": str(DEFAULT_GATE.resolve()), "embedding_cache_pack": cache,
             "cache_roots": cache["cache_roots"], "official_metric_provenance": provenance,
             "code_sha256": {p: sha(ROOT / p) for p in CODE},
             "runtime_sha256": {p: sha(ROOT / p) for p in RUNTIME},
             "sdk_version": importlib.metadata.version("mem0ai"),
             "sdk_sha256": {str(p): sha(p) for p in sdkroot.rglob("*.py")},
             "limitations": gate["limitations"], "model_calls_during_prepare": 0}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    return {"run_dir": str(run), "arm": arm, "banks": 10, "qa_total": 400, "model_calls": 0}


def manifest(run_dir, *, conversation=None, recheck_sources=False):
    run = Path(run_dir).resolve()
    m = read(run / "run_manifest.json")
    if (m.get("schema") != SCHEMA or m.get("policy") != POLICY or m.get("arm") not in sources.ARMS
            or m.get("protocol_sha256") != digest({k: v for k, v in m.items() if k != "protocol_sha256"})):
        raise ValueError("Frozen refinement QA protocol changed")
    if m["code_sha256"] != {p: sha(ROOT / p) for p in CODE}:
        raise ValueError("Frozen QA code changed")
    for path, expected in {**m["runtime_sha256"], **m["sdk_sha256"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen SDK/runtime changed")
    if importlib.metadata.version("mem0ai") != m["sdk_version"]:
        raise ValueError("Native SDK version changed")
    sources.checked_pointer(m["cohort_manifest"])
    sources.checked_pointer(m["plain_text_config"])
    sources.protocol_config(m["plain_text_config"]["path"])
    rows, dates = sources.cohort()
    if (m["dataset_sha256"] != SOURCE_HASH or digest(rows) != m["cohort_sha256"] or dates != m["reference_dates"]
            or m["query_ids"] != {ci: [q["id"] for q in qs] for ci, qs in rows.items()}):
        raise ValueError("Fixed400 question/date identity changed")
    _, provenance = verified_official_metric()
    if provenance != m["official_metric_provenance"]:
        raise ValueError("Official metric changed")
    sources.checked_pointer(m["gate"])
    validate_resources(m["resources"]["snapshot"], m["resources"]["files_sha256"])
    if m["phase_kind"] == "firstpass":
        embedding_pack.verify_pack(m["embedding_cache_pack"], conversation=conversation, check_sources=recheck_sources)
        if m["cache_roots"] != m["embedding_cache_pack"]["cache_roots"]:
            raise ValueError("Only the frozen QUERY embedding pack can be inherited")
        for ci in range(10):
            native_qa.get_bank(run, ci, m["arm"], m)
    elif m["phase_kind"] == "technical_recovery":
        if m["cache_roots"] != [] or m.get("parent") is None:
            raise ValueError("Recovery must use a fresh phase with no external API cache")
        p = m["parent"]
        sources.checked_pointer(p["manifest"]); sources.checked_pointer(p["outcome"])
        if run != Path(p["run_dir"]).with_name(Path(p["run_dir"]).name + "_recovery_v1"):
            raise ValueError("Only one separate bounded recovery namespace is permitted")
    else:
        raise ValueError("Unknown phase kind")
    if recheck_sources:
        gate, _, resource = sources.admit_gate(m["gate"]["path"])
        if gate["limitations"] != m["limitations"] or resource != m["resources"]:
            raise ValueError("Source audit binding changed")
    return m


def retrieve_worker(run_dir, ci):
    run = Path(run_dir)
    m = manifest(run, conversation=ci)
    folder = run / f"conv_{ci}"
    native_qa.durable_intent(folder / "retrieval_execution_intent.json",
                             {"protocol_sha256": m["protocol_sha256"], "conversation": ci,
                              "created_utc": now(), "pid": os.getpid()})
    if not plain.VAULT.is_file():
        raise RuntimeError("DPAPI vault module required; no environment fallback")
    api = PlainTextEvaluationAPI(folder / "api", cache_roots=m["cache_roots"][str(ci)],
                                gate_path=m["embedding_gate"], retries=2,
                                context={"conversation": ci, "architecture": SCHEMA, "phase": "retrieval"})
    qs = sources.cohort()[0][str(ci)]
    resource = m["resources"]
    with local_bm25_context(resource["snapshot"], resource["files_sha256"]):
        native_qa.retrieve_arm(run, ci, m["arm"], qs, m, api)
    selections = [verify(folder / "selection" / m["arm"] / (q["id"] + ".json")) for q in qs]
    return {"conversation": ci, "terminal": True, "total": len(qs),
            "counts": dict(Counter(row["status"] for row in selections))}


def retrieve(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run, recheck_sources=True)
    if m["phase_kind"] != "firstpass":
        raise ValueError("No new retrieval in QA technical recovery")
    path = run / "retrieval_outcome.json"
    if path.exists():
        return verify(path)
    native_qa.durable_intent(run / "retrieval_execution_intent.json",
                             {"protocol_sha256": m["protocol_sha256"], "pid": os.getpid(), "created_utc": now()})
    outcomes, errors = [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks = {pool.submit(retrieve_worker, str(run), ci): ci for ci in range(10)}
        for future in as_completed(tasks):
            try:
                outcomes.append(future.result())
            except Exception as exc:
                errors.append({"conversation": tasks[future], "error": plain.safe_error(exc)})
            publish_json(run / "retrieval_status.json", {"schema": SCHEMA, "pid": os.getpid(), "terminal": False,
                         "updated_utc": now(), "outcomes": outcomes, "errors": errors})
    result = {"schema": SCHEMA, "protocol_sha256": m["protocol_sha256"], "terminal": True,
              "outcomes": outcomes, "errors": errors, "finished_utc": now(), "expected_slots": 400}
    save_new(path, result)
    publish_json(run / "retrieval_status.json", result)
    return verify(path)


def check_job(job, arm):
    if (job.get("method") != arm or job.get("status") != "ready"
            or job.get("job_id") != arm + "/" + job.get("query_id", "")):
        raise ValueError("Invalid refinement QA job")
    ids, facts = job["selected_ids"], job["selected_facts"]
    if len(ids) != 20 or len(set(ids)) != 20 or [f["id"] for f in facts] != ids:
        raise ValueError("Exactly twenty ordered unique facts required")
    if any(set(f) != {"id", "text", "observation_date"} for f in facts):
        raise ValueError("Only original plain Reader fact fields allowed")
    reader_input(job["question"], facts, reference_date=job["reference_date"])


def job_from_selection(run, ci, arm, q, m):
    bank, bank_hash = native_qa.get_bank(run, ci, arm, m)
    path = Path(run) / f"conv_{ci}/selection/{arm}/{q['id']}.json"
    expected = {**native_qa.identity(ci, arm, q, m), "bank_sha256": bank_hash}
    selection = native_qa.existing(path, expected)
    if selection is None or selection.get("status") != "complete":
        raise ValueError("Missing/failed selection retained for separate review; no partial400 dispatch")
    ids = selection["selected_ids"]; byid = {f["id"]: f for f in bank["facts"]}
    if not set(ids) <= set(byid) or ids != selection.get("capture", {}).get("base20_ids"):
        raise ValueError("Selection IDs/order no longer match original native SDK20")
    job = {"job_id": arm + "/" + q["id"], "conversation": ci, "method": arm,
           "query_id": q["id"], "status": "ready", "question": q["question"],
           "reference": q["reference"], "category": q["category"], "reference_date": m["reference_dates"][str(ci)],
           "selected_ids": ids, "selected_facts": [{k: byid[i].get(k) for k in ("id", "text", "observation_date")} for i in ids],
           "lineage": {"selection": sources.pointer(path), "bank": m["banks"][str(ci)][arm]}}
    check_job(job, arm)
    return job


def seal_jobs(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run)
    if m["phase_kind"] != "firstpass":
        raise ValueError("Recovery jobs are bound at recovery preparation")
    outcome = verify(run / "retrieval_outcome.json")
    if (outcome.get("terminal") is not True or outcome.get("protocol_sha256") != m["protocol_sha256"]
            or outcome.get("errors") or len(outcome.get("outcomes", [])) != 10
            or {r["conversation"] for r in outcome["outcomes"]} != set(range(10))):
        raise ValueError("Whole retrieval batch must finish and have no unresolved controller errors")
    rows, _ = sources.cohort()
    jobs = [job_from_selection(run, ci, m["arm"], q, m) for ci in range(10) for q in rows[str(ci)]]
    value = {"schema": SCHEMA + "-sources", "arm": m["arm"], "jobs": jobs,
             "protocol_sha256": m["protocol_sha256"], "retrieval_outcome": sources.pointer(run / "retrieval_outcome.json"),
             "official_metric_provenance": m["official_metric_provenance"], "model_calls": 0}
    path = run / "source_jobs.json"
    if path.exists():
        old = verify(path)
        if {k: v for k, v in old.items() if k != "content_sha256"} != value:
            raise ValueError("Frozen selected inputs changed")
    else:
        save_new(path, value)
    return verify(path)


def execution_manifest(run_dir):
    run = Path(run_dir).resolve(); m = manifest(run)
    if m["phase_kind"] == "firstpass":
        source = seal_jobs(run)
        return {**m, "jobs": source["jobs"], "source_manifest": sources.pointer(run / "source_jobs.json")}
    source = verify(sources.checked_pointer(m["source_manifest"]))
    parent_m, parent_out = terminal(m["parent"]["run_dir"])
    if m["jobs"] != recovery_jobs(parent_m, parent_out):
        raise ValueError("Uniform recovery population changed")
    if m["source_manifest"] != parent_m["source_manifest"] or source["official_metric_provenance"] != m["official_metric_provenance"]:
        raise ValueError("Recovery source lineage changed")
    return m


def qa(run_dir, *, api_factory=None):
    run = Path(run_dir).resolve(); m = execution_manifest(run)
    if (run / "run_outcome.json").exists():
        return terminal(run)[1]
    if (run / "qa_execution_intent.json").exists():
        raise RuntimeError("Prior phase intent retained; no implicit restart")
    if api_factory is None and not plain.VAULT.is_file():
        raise RuntimeError("DPAPI vault required; no environment fallback")
    official, _ = verified_official_metric()
    native_qa.durable_intent(run / "qa_execution_intent.json",
                             {"protocol_sha256": m["protocol_sha256"], "pid": os.getpid(), "created_utc": now()})
    def worker(ci, jobs):
        api = initial_error = None
        try:
            # QA gets no embedding roots or old answers, only its own fresh phase cache.
            api = (api_factory(run / f"conv_{ci}/api", ci, m) if api_factory else
                   PlainTextEvaluationAPI(run / f"conv_{ci}/api", cache_roots=(), retries=2,
                       context={"architecture": plain.SCHEMA, "conversation": ci, "phase": "qa", "phase_id": m["phase_id"]}))
        except Exception as exc:
            initial_error = exc
        errors, completed = [], 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = {pool.submit(plain.one, run, j, m, api, official, initial_error): j for j in jobs}
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"job_id": tasks[future]["job_id"], "error": plain.safe_error(exc, api)})
                completed += 1
                publish_json(run / f"conv_{ci}/qa_progress.json", {"schema": SCHEMA, "pid": os.getpid(),
                             "conversation": ci, "phase_kind": m["phase_kind"], "updated_utc": now(),
                             "completed_slots": completed, "expected_slots": len(jobs),
                             "local_error_count": len(errors), "terminal": completed == len(jobs)})
        return {"conversation": ci, "jobs": len(jobs), "terminal": True, "local_errors": errors}
    outcomes = []
    byconv = {ci: [j for j in m["jobs"] if j["conversation"] == ci] for ci in range(10)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(worker, ci, jobs): ci for ci, jobs in byconv.items() if jobs}
        for future in as_completed(futures):
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append({"conversation": futures[future], "terminal": True, "local_errors": [plain.safe_error(exc)]})
            publish_json(run / "qa_status.json", {"schema": SCHEMA, "pid": os.getpid(), "terminal": False,
                         "updated_utc": now(), "phase_kind": m["phase_kind"], "outcomes": outcomes})
    results, counts = [], {stage: Counter() for stage in ("reader", "judge", "compat", "qa")}
    for j in m["jobs"]:
        row = {k: j[k] for k in ("job_id", "conversation", "method", "query_id")}
        for stage, path in plain.stage_files(run, j).items():
            row[stage] = sources.pointer(path) if path.exists() else None
            counts[stage][verify(path)["status"] if path.exists() else "unknown"] += 1
        results.append(row)
    result = {"schema": SCHEMA, "protocol_sha256": m["protocol_sha256"], "phase_kind": m["phase_kind"],
              "phase_id": m["phase_id"], "arm": m["arm"], "terminal": True, "finished_utc": now(),
              "total": len(m["jobs"]), "planned_per_arm": 400, "counts": {s: dict(v) for s, v in counts.items()},
              "outcomes": outcomes, "results": results, "new_native_clients_during_qa": 0}
    save_new(run / "run_outcome.json", result)
    publish_json(run / "qa_status.json", {**result, "pid": os.getpid()})
    return verify(run / "run_outcome.json")


def terminal(run_dir):
    run = Path(run_dir).resolve(); m = execution_manifest(run); out = verify(run / "run_outcome.json")
    if (out.get("terminal") is not True or out.get("protocol_sha256") != m["protocol_sha256"]
            or out.get("total") != len(m["jobs"]) or len(out.get("results", [])) != len(m["jobs"])):
        raise ValueError("Whole QA phase is not terminal")
    jobs = {plain.job_key(j): j for j in m["jobs"]}
    if len(jobs) != len(m["jobs"]) or {plain.job_key(r) for r in out["results"]} != set(jobs):
        raise ValueError("Terminal QA population differs")
    for row in out["results"]:
        j = jobs[plain.job_key(row)]; expected = plain.identity(j, m)
        for stage, path in plain.stage_files(run, j).items():
            if row[stage] is None:
                if path.exists():
                    raise ValueError("An existing artifact was recorded unknown")
                continue
            if row[stage] != sources.pointer(path):
                raise ValueError("Terminal artifact points to another slot")
            value = verify(path)
            if any(value.get(k) != v for k, v in expected.items()):
                raise ValueError("Terminal artifact changed frozen job identity")
    return m, out


def recovery_jobs(m, out):
    if m["phase_kind"] != "firstpass":
        raise ValueError("Only one postbatch technical recovery allowed")
    if any(r.get("local_errors") for r in out["outcomes"]):
        raise ValueError("Local errors require offline review before recovery")
    originals = {plain.job_key(j): j for j in m["jobs"]}; jobs = []
    for row in out["results"]:
        if any(row[s] is None for s in ("reader", "judge", "compat", "qa")):
            raise ValueError("Unknown stage cannot be implicitly redispatched")
        j = originals[plain.job_key(row)]
        reader, judge, compat = (verify(sources.checked_pointer(row[s])) for s in ("reader", "judge", "compat"))
        if compat.get("error"):
            raise ValueError("Compatibility provenance failure requires review")
        if reader["status"] == "failed":
            plain.known_failure(j, m, reader, row["reader"], "reader")
        elif reader["status"] != "complete":
            raise ValueError("Unknown Reader state cannot be retried")
        elif judge["status"] == "failed":
            plain.known_failure(j, m, judge, row["judge"], "judge")
        elif judge["status"] != "complete":
            raise ValueError("Unknown Judge state cannot be retried")
        else:
            continue  # Includes valid NO; correctness is never a retry criterion.
        candidate = copy.deepcopy(j)
        candidate.update(parent_results={s: row[s] for s in ("reader", "judge", "compat", "qa")},
                         parent_job_sha256=digest(j), reader_source=row["reader"] if reader["status"] == "complete" else None)
        candidate["attempt_kind"] = "judge_only_after_successful_reader" if candidate["reader_source"] else "reader_retry_then_first_judge"
        check_job(candidate, m["arm"]); jobs.append(candidate)
    return jobs


def prepare_recovery(firstpass_run):
    source_run = Path(firstpass_run).resolve(); m, out = terminal(source_run)
    jobs = recovery_jobs(m, out)
    run = source_run.with_name(source_run.name + "_recovery_v1")
    if run.exists():
        raise FileExistsError("One fixed recovery namespace; never overwrite or retry again")
    value = {k: copy.deepcopy(v) for k, v in m.items() if k not in {"protocol_sha256", "jobs"}}
    value.update(created_utc=now(), phase_kind="technical_recovery", phase_id=SCHEMA + "_" + m["arm"] + "_technical_recovery",
                 jobs=jobs, cache_roots=[], parent={"run_dir": str(source_run),
                    "manifest": sources.pointer(source_run / "run_manifest.json"), "outcome": sources.pointer(source_run / "run_outcome.json")})
    run.mkdir(parents=True, exist_ok=False)
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    return {"run_dir": str(run), "jobs": len(jobs), "model_calls": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "check", "retrieve", "seal", "qa", "run", "recover"))
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--gate", type=Path)
    a = p.parse_args(); run = a.run_dir.resolve()
    if a.stage == "prepare":
        if a.gate is None:
            p.error("prepare requires --gate")
        result = prepare(run, a.gate)
    elif a.stage == "check":
        m = manifest(run, recheck_sources=True); result = {"valid": True, "arm": m["arm"], "phase_kind": m["phase_kind"], "model_calls": 0}
    elif a.stage == "retrieve": result = retrieve(run)
    elif a.stage == "seal": result = {"jobs": len(seal_jobs(run)["jobs"]), "model_calls": 0}
    elif a.stage == "qa": result = qa(run)
    elif a.stage == "recover": result = prepare_recovery(run)
    else:
        m = manifest(run)
        if m["phase_kind"] == "firstpass": retrieve(run)
        result = qa(run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
