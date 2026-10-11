"""Two dated Mem0 arms, unchanged SDK20 search, independent plain-text QA.

Prepare/check are offline and require the final twenty-bank audit. Run alone can
dispatch models. Original native stores are never opened, and old QA outputs are
never imported. No task runs on import.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import mem0_regroup_plain_qa_sources_v1 as sources
from tools import run_session_experiments as native
from tools import run_mem0_regroup_sdk20 as writer
from tools.run_sdk20_modules import digest, sha, now, verify, save_new, Warnings
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.summarize_pilot400 import verified_official_metric
from jev_mem.data import SOURCE_HASH
from jev_mem.evaluation_plain_text_v1 import read_answer, judge_answer, InvalidJudgeLabel
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.providers import Embeddings
from jev_mem.paced_embedding_api import DEFAULT_GATE
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources
from jev_mem.sdk20_type_adapter import capture_native_pool
from jev_mem import qa_embedding_cache_pack_v1 as embedding_pack

SCHEMA = "mem0-dated-regroup-plain-qa-v1"
ARMS = sources.ARMS
CODE = ("tools/run_mem0_regroup_plain_qa_v1.py", "tools/mem0_regroup_plain_qa_sources_v1.py",
        "tools/run_session_experiments.py", "tools/run_sdk20_modules.py", "tools/sdk20_module_sources.py",
        "tools/run_pilot400_experiments_v2.py", "tools/summarize_pilot400.py",
        "jev_mem/sdk20_type_adapter.py", "jev_mem/plain_text_api_v1.py", "jev_mem/evaluation_plain_text_v1.py",
        "jev_mem/providers.py", "jev_mem/sdk20_cached_api.py", "jev_mem/paced_embedding_api.py",
        "jev_mem/local_bm25_context_v2.py", "jev_mem/data.py", "jev_mem/evaluation.py",
        "jev_mem/qa_embedding_cache_pack_v1.py")
POLICY = {"arms": list(ARMS), "questions_per_arm": 1540, "total": 3080,
          "workers": 8, "qa_threads_per_conversation": 4, "retrieval_within_process": "sequential",
          "retrieval": {"top_k": 20, "rerank": False, "native_candidate_cap": 80},
          "reference_date": "last_original_session_envelopes_observation_date",
          "reader_fact_dates": "unchanged_including_null", "qa_outputs": "plain_text_v1_no_old_QA_inheritance",
          "failure_policy": "retain_first_failure_or_unknown_without_logical_redispatch",
          "reader_and_judge": "persist_reader_and_both_F1_metrics_before_judge",
          "cache_policy": "embedding_only_pack_same_conversation_original_session_then_sdk20_exact_first_source_no_old_QA",
          "native_source": "byte_copy_only_never_open_original", "inherited_api_ledger_bytes": 0}


def prepare(run_dir, gate_path, config_path=sources.CONFIG):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT / "runs").resolve():
        raise ValueError("QA namespace must be a direct new child of workspace runs")
    if run.exists():
        raise FileExistsError("A new QA namespace is required")
    config = sources.protocol_config(config_path)
    gate, admitted, resource = sources.admit_gate(gate_path)
    cache_plan = embedding_pack.audit_sources()
    rows, dates = sources.cohort()
    _, metric_provenance = verified_official_metric()
    for origin in (sources.DEVELOPMENT, sources.FOURTH, sources.REMAINING):
        origin = origin.resolve()
        if run == origin or run in origin.parents or origin in run.parents:
            raise ValueError("QA namespace must be disjoint from frozen writer runs")
    # Nothing is created until every input and review has been admitted.
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / "prepare_intent.json", {"schema": SCHEMA, "gate": sources.pointer(gate_path), "model_calls": 0})
    banks = {}
    for ci in range(10):
        banks[str(ci)] = {arm: sources.copy_bank(run, ci, arm, admitted[(ci, arm)]) for arm in ARMS}
    cache_binding = embedding_pack.create_pack(run, cache_plan)
    sdkroot = Path(importlib.metadata.distribution("mem0ai").locate_file("mem0"))
    runtime = (".venv/Lib/site-packages/fastembed/sparse/bm25.py",
               ".venv/Lib/site-packages/fastembed/sparse/sparse_text_embedding.py",
               ".venv/Lib/site-packages/qdrant_client/fastembed_common.py",
               ".venv/Lib/site-packages/qdrant_client/qdrant_fastembed.py")
    value = {"schema": SCHEMA, "created_utc": now(), "policy": POLICY,
             "dataset_sha256": SOURCE_HASH, "cohort_sha256": digest(rows), "reference_dates": dates,
             "query_ids": {ci: [q["id"] for q in qs] for ci, qs in rows.items()},
             "banks": banks, "gate": sources.pointer(gate_path), "plain_text_config": sources.pointer(config_path),
             "evaluation_protocol": config["evaluation_protocol"], "resources": resource,
             "embedding_gate": str(DEFAULT_GATE.resolve()), "official_metric_provenance": metric_provenance,
             "embedding_cache_pack": cache_binding, "cache_roots": cache_binding["cache_roots"],
             "code_sha256": {path: sha(ROOT / path) for path in CODE},
             "runtime_sha256": {path: sha(ROOT / path) for path in runtime},
             "sdk_version": importlib.metadata.version("mem0ai"),
             "sdk_sha256": {str(p): sha(p) for p in sdkroot.rglob("*.py")},
             "limitations": gate["limitations"], "model_calls_during_prepare": 0}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    return {"prepared": str(run), "protocol_sha256": digest(value), "banks": 20, "qa_total": 3080, "model_calls": 0}


def manifest(run_dir, *, recheck_sources=False, conversation=None):
    run = Path(run_dir).resolve()
    value = read(run / "run_manifest.json")
    if (value.get("schema") != SCHEMA or value.get("policy") != POLICY
            or value.get("protocol_sha256") != digest({k: v for k, v in value.items() if k != "protocol_sha256"})):
        raise ValueError("QA manifest identity or policy changed")
    for path, expected in {**value["code_sha256"], **value["runtime_sha256"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("QA frozen source changed: " + path)
    if importlib.metadata.version("mem0ai") != value["sdk_version"]:
        raise ValueError("Native Mem0 version changed")
    for path, expected in value["sdk_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Frozen native SDK source changed")
    sources.checked_pointer(value["plain_text_config"])
    sources.protocol_config(value["plain_text_config"]["path"])
    sources.checked_pointer(value["gate"])
    rows, dates = sources.cohort()
    if (value["dataset_sha256"] != SOURCE_HASH or digest(rows) != value["cohort_sha256"]
            or dates != value["reference_dates"]
            or value["query_ids"] != {ci: [q["id"] for q in qs] for ci, qs in rows.items()}):
        raise ValueError("Cohort or original conversation reference dates changed")
    _, provenance = verified_official_metric()
    if provenance != value["official_metric_provenance"]:
        raise ValueError("Official metric implementation changed")
    validate_resources(value["resources"]["snapshot"], value["resources"]["files_sha256"])
    embedding_pack.verify_pack(value["embedding_cache_pack"], conversation=conversation, check_sources=recheck_sources)
    if value["cache_roots"] != value["embedding_cache_pack"]["cache_roots"]:
        raise ValueError("QA cache roots differ from the bound embedding-only pack")
    for ci in range(10):
        for arm in ARMS:
            get_bank(run, ci, arm, value)
    if recheck_sources:
        gate, admitted, resource = sources.admit_gate(value["gate"]["path"])
        if resource != value["resources"] or gate["limitations"] != value["limitations"]:
            raise ValueError("Source resource/quality review binding changed")
    return value


def get_bank(run, ci, arm, m):
    binding = m["banks"][str(ci)][arm]
    path = sources.checked_pointer(binding)
    expected = (Path(run) / f"conv_{ci}" / arm / "bank.json").resolve()
    if path.resolve() != expected:
        raise ValueError("QA bank is not the bound local copy")
    bank = verify(path)
    if ((bank["conversation"], bank["arm"]) != (ci, arm)
            or Path(bank["native_dir"]).resolve() != expected.parent / "native"
            or Path(bank["native_dir"]).resolve() == Path(bank["native_source"]).resolve()):
        raise ValueError("QA attempted to open an original source or another conversation")
    return bank, binding["sha256"]


def identity(ci, arm, q, m):
    return {"schema": SCHEMA, "protocol_sha256": m["protocol_sha256"], "conversation": ci,
            "method": arm, "query_id": q["id"], "question": q["question"],
            "reference": q["reference"], "category": q["category"]}


def existing(path, expected):
    if not Path(path).exists():
        return None
    value = verify(path)
    if any(value.get(key) != val for key, val in expected.items()):
        raise ValueError("Existing immutable QA artifact changed identity or inputs")
    return value


def error(exc, api=None):
    text = api._redact(exc) if api else str(exc)
    return {"type": type(exc).__name__, "message": text[:1500]}


def durable_intent(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())


def status(folder, stage, **value):
    publish_json(Path(folder) / (stage + "_status.json"),
                 {"schema": SCHEMA, "pid": os.getpid(), "updated_utc": now(), "stage": stage, **value})


def retrieve_arm(run, ci, arm, qs, m, api):
    folder = Path(run) / f"conv_{ci}"
    bank, bank_hash = get_bank(run, ci, arm, m)
    byid = {f["id"]: f for f in bank["facts"]}
    memory, startup_error = None, None
    native_copy = Path(bank["native_dir"])
    if sources.inventory(native_copy) != bank["source_inventory"]:
        raise ValueError("QA copy changed before its first native search")
    try:
        memory = native.open_memory(native_copy, api)
        from mem0.utils import spacy_models
        if (spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None
                or memory.vector_store._get_bm25_encoder() is None):
            raise RuntimeError("Native NLP/facts BM25 unavailable; retrieval must not silently fall back")
    except Exception as exc:
        startup_error = error(exc, api)
    try:
        for index, q in enumerate(qs):
            ident = identity(ci, arm, q, m)
            path = folder / "selection" / arm / (q["id"] + ".json")
            if path.exists():
                raise RuntimeError("Selection exists before a new conversation attempt")
            scoped = api.scoped(query_id=q["id"], method=arm, phase="retrieval")
            audit, failure, ids, captured = Warnings(), startup_error, [], None
            if failure is None:
                memory.embedding_model = Embeddings(scoped)
                logging.getLogger("mem0").addHandler(audit)
                try:
                    with capture_native_pool() as capture:
                        result = memory.search(q["question"], top_k=20, filters={"user_id": bank["user_id"]}, rerank=False)
                    ids = [row["id"] for row in result["results"]]
                    if (ids != capture["base20_ids"] or len(ids) != 20 or len(set(ids)) != 20
                            or not set(capture["eligible_ids"]) <= set(byid) or not set(ids) <= set(byid)):
                        raise ValueError("SDK20 selection identity, order or budget differs")
                    if audit.items:
                        raise RuntimeError("Native search warning requires review")
                    captured = capture
                except Exception as exc:
                    failure, ids = error(exc, scoped), []
                finally:
                    logging.getLogger("mem0").removeHandler(audit)
            save_new(path, {**ident, "status": "failed" if failure else "complete", "bank_sha256": bank_hash,
                            "selected_ids": ids, "capture": captured, "error": failure,
                            "sdk_warnings": audit.items, "api_events": scoped.events})
            status(folder, "retrieval", method=arm, completed=index + 1, total=len(qs), terminal=False, last_query=q["id"])
    finally:
        if memory is not None:
            native.close_memory(memory)
    save_new(folder / arm / "post_retrieval_inventory.json", {"schema": SCHEMA,
             "protocol_sha256": m["protocol_sha256"], "native_copy": str(native_copy),
             "inventory": sources.inventory(native_copy), "source_native_opened": False})


def stage_once(path, expected, callback, api):
    old = existing(path, expected)
    if old is not None:
        return old
    intent = Path(path).with_suffix(".intent.json")
    if intent.exists():
        # No new logical attempt after interrupted dispatch/persistence. The raw
        # cache remains available for an explicitly reviewed recovery later.
        value = {**expected, "status": "failed", "result": None, "api_events": [],
                 "error": {"type": "UnresolvedStageIntent", "message": "Prior stage is retained; no implicit redispatch"},
                 "raw_invalid_label": None}
    else:
        durable_intent(intent, {**expected, "created_utc": now(), "pid": os.getpid()})
        failure, raw_invalid, result = None, None, None
        try:
            result = callback()
        except Exception as exc:
            failure = error(exc, api)
            if isinstance(exc, InvalidJudgeLabel):
                raw_invalid = exc.raw_response
        value = {**expected, "status": "failed" if failure else "complete", "result": result,
                 "error": failure, "raw_invalid_label": raw_invalid, "api_events": list(api.events)}
    save_new(path, value)
    return verify(path)


def qa_one(run, ci, arm, q, m, bank, bank_hash, api, official):
    folder = Path(run) / f"conv_{ci}"
    ident = identity(ci, arm, q, m)
    sp = folder / "selection" / arm / (q["id"] + ".json")
    selection = existing(sp, {**ident, "bank_sha256": bank_hash})
    if selection is None:
        raise ValueError("A terminal selection record is required")
    ids = selection["selected_ids"]
    common = {**ident, "selection_sha256": sha(sp), "bank_sha256": bank_hash,
              "selected_ids": ids, "reference_date": m["reference_dates"][str(ci)]}
    final_path = folder / "qa" / arm / (q["id"] + ".json")
    done = existing(final_path, common)
    if done is not None:
        return done
    rp, jp = (folder / name / arm / (q["id"] + ".json") for name in ("reader", "judge"))
    if selection["status"] != "complete":
        for path, stage in ((rp, "reader"), (jp, "judge")):
            if existing(path, common) is None:
                save_new(path, {**common, "stage": stage, "status": "not_run", "result": None,
                                "error": {"type": "SelectionFailed", "message": "No downstream dispatch"}, "api_events": []})
    else:
        byid = {fact["id"]: fact for fact in bank["facts"]}
        if len(ids) != 20 or len(set(ids)) != 20 or not set(ids) <= set(byid):
            raise ValueError("Bound selection does not contain20 unique bank facts")
        reader_api = api.scoped(query_id=q["id"], method=arm, phase="qa", component="reader")
        def call_reader():
            result = read_answer(reader_api, q["question"], q["reference"], [byid[i] for i in ids],
                                 reference_date=common["reference_date"])
            return {**result, "metrics": {
                "official_locomo_f1": official(result["prediction"]["answer"], q["reference"], q["category"]),
                "custom_token_f1": result["token_f1"], "custom_exact_match": result["exact_match"]}}
        reader = stage_once(rp, {**common, "stage": "reader"}, call_reader, reader_api)
        # The reader artifact has been flushed/published before Judge is invoked.
        if reader["status"] == "complete":
            judge_api = api.scoped(query_id=q["id"], method=arm, phase="qa", component="judge")
            stage_once(jp, {**common, "stage": "judge", "reader_sha256": sha(rp)},
                       lambda: judge_answer(judge_api, q["question"], q["reference"], reader["result"]["prediction"]["answer"]), judge_api)
        elif existing(jp, common) is None:
            save_new(jp, {**common, "stage": "judge", "reader_sha256": sha(rp), "status": "not_run", "result": None,
                          "error": {"type": "ReaderFailed", "message": "No Judge dispatch"}, "api_events": []})
    reader, judge = verify(rp), verify(jp)
    value = {**common, "status": "complete" if reader["status"] == judge["status"] == "complete" else "failed",
             "reader_status": reader["status"], "judge_status": judge["status"],
             "reader": sources.pointer(rp), "judge": sources.pointer(jp),
             "reader_metrics": reader["result"]["metrics"] if reader["status"] == "complete" else None,
             "judge_correct": judge["result"]["correct"] if judge["status"] == "complete" else None}
    save_new(final_path, value)
    return verify(final_path)


def worker(run_dir, ci):
    run, m = Path(run_dir), manifest(run_dir, conversation=ci)
    folder = run / f"conv_{ci}"
    durable_intent(folder / "execution_intent.json", {"protocol_sha256": m["protocol_sha256"], "conversation": ci,
                                                    "pid": os.getpid(), "created_utc": now()})
    vault = Path(r"C:\Users\86139\Desktop\jev\experiments\live\credential_store.py")
    if not vault.exists():
        raise RuntimeError("DPAPI vault module required; environment fallback disabled")
    api = PlainTextEvaluationAPI(folder / "api", cache_roots=m["cache_roots"][str(ci)],
                                gate_path=m["embedding_gate"], context={"conversation": ci, "architecture": SCHEMA})
    qs = sources.cohort()[0][str(ci)]
    resource = m["resources"]
    with local_bm25_context(resource["snapshot"], resource["files_sha256"]):
        for arm in ARMS:
            retrieve_arm(run, ci, arm, qs, m, api)
    banks = {arm: get_bank(run, ci, arm, m) for arm in ARMS}
    official, _ = verified_official_metric()
    completed = reader_ok = judge_ok = failed = 0
    def one(q):
        return [qa_one(run, ci, arm, q, m, *banks[arm], api, official) for arm in ARMS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, q) for q in qs]):
            for row in future.result():
                completed += 1
                reader_ok += row["reader_status"] == "complete"
                judge_ok += row["judge_status"] == "complete"
                failed += row["status"] != "complete"
            status(folder, "qa", completed=completed, total=2 * len(qs), reader_complete=reader_ok,
                   judge_complete=judge_ok, failed=failed, terminal=False)
    result = {"conversation": ci, "completed": completed, "total": 2 * len(qs),
              "reader_complete": reader_ok, "judge_complete": judge_ok, "failed": failed, "terminal": True}
    status(folder, "qa", **result)
    return result


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    m = manifest(run_dir, recheck_sources=True)
    with writer.writer_process_lock(run_dir):
        durable_intent(run_dir / "execution_intent.json", {"protocol_sha256": m["protocol_sha256"],
                                                         "pid": os.getpid(), "created_utc": now()})
        outcomes, failures = [], []
        with ProcessPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(worker, str(run_dir), ci): ci for ci in range(10)}
            for future in as_completed(pending):
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    failures.append({"conversation": pending[future], "error": error(exc)})
                status(run_dir, "run", terminal=False, conversations_terminal=len(outcomes) + len(failures),
                       outcomes=outcomes, failures=failures, expected_qa=3080)
        status(run_dir, "run", terminal=True, conversations_terminal=10, outcomes=outcomes, failures=failures, expected_qa=3080)
    return {"outcomes": outcomes, "failures": failures, "expected_qa": 3080}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "check", "run"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.gate is None:
            parser.error("prepare requires --gate with a passed20-bank review")
        result = prepare(args.run_dir, args.gate)
    elif args.phase == "check":
        result = {"valid": True, "protocol_sha256": manifest(args.run_dir, recheck_sources=True)["protocol_sha256"], "model_calls": 0}
    else:
        result = run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
