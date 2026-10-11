"""Explicit live stages for C-off/C-JEV; imports and --help make no API calls.

Smoke and full artifacts cannot share a run directory. Every cached result is
bound to its fact-bank snapshot, protocol and input. Only requested records enter
summaries; technical errors remain visible in the full requested denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_mem.data import SOURCE_HASH, load_data, queries, session_envelopes
from jev_mem.evaluation import answer_and_judge
from jev_mem.independent import IndependentMemory
from jev_mem.providers import ModelAPI, dump_json
from jev_mem.selection import select

METHODS = ("rank", "mmr", "jev_single", "conditional")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(name + " must be a positive integer")


def _protocol():
    files = ["jev_mem/independent.py", "jev_mem/data.py", "jev_mem/providers.py",
             "jev_mem/selection.py", "jev_mem/evaluation.py", "tools/run_independent_experiments.py"]
    return {"name": "independent_session_top20_v1", "dataset_sha256": SOURCE_HASH,
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
            "extractor_reader": "gemini-2.5-flash", "judge": "gemini-3.5-flash-lite",
            "embedding": "gemini-embedding-001", "dimensions": 768, "jev": "jev-1.13.0",
            "current_date": "2026-09-27", "input": "one_complete_session_per_add",
            "write_fact_cap": None, "candidate_pool": 60, "qa_fact_count": 20,
            "retrieval": "independent_bm25_dense_union_rrf", "rrf_k": 60,
            "selection": {"seed": 8, "rounds": [6, 6], "shortlist": 32, "conditional_threshold": .5,
                          "mmr_weight": .7, "selector_internal_workers": 8}}


def ensure_run(run_path, *, allow_partial=False, limit=None):
    _positive(limit, "Smoke limit")
    if limit is not None and not allow_partial:
        raise ValueError("Sample limits require --allow-partial-smoke and a separate smoke run directory")
    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "run_manifest.json"
    manifest = {"schema": "independent-live-run-v1", "mode": "smoke" if allow_partial else "full",
                "protocol": _protocol()}
    manifest["protocol_sha256"] = _hash(manifest["protocol"])
    if manifest_path.exists():
        if _read(manifest_path) != manifest:
            raise RuntimeError("Run mode/protocol changed; use a new directory, never mix smoke/full or stale artifacts")
    else:
        if any(run_path.glob("conv_*")):
            raise RuntimeError("Unmanifested existing conversation artifacts cannot be adopted as a new run")
        dump_json(manifest_path, manifest)
    return manifest


def _conversation(ci):
    data = load_data()
    if isinstance(ci, bool) or not isinstance(ci, int) or not 0 <= ci < len(data):
        raise ValueError("Conversation index is outside the pinned dataset")
    return data[ci]


def _error(api, error):
    redact = getattr(api, "_redact", None)
    message = redact(error) if redact else str(error)
    return {"error_type": type(error).__name__, "error": message[:1800]}


def _emit(row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def ingest_one(run_path, ci, max_sessions=None, allow_partial=False):
    manifest = ensure_run(run_path, allow_partial=allow_partial, limit=max_sessions)
    item = _conversation(ci)
    all_envelopes = list(session_envelopes(item))
    requested = all_envelopes[:max_sessions] if max_sessions is not None else all_envelopes
    conv_dir = Path(run_path) / f"conv_{ci}"
    api = ModelAPI(conv_dir)
    memory = IndependentMemory(api, conv_dir / "independent_store")
    user_id = f"locomo_c{ci}_independent_session_v1"
    errors, attempted_records = [], []
    start = time.perf_counter()
    try:
        for index, envelope in enumerate(requested):
            try:
                memory.api = api.scoped(conversation=ci, session_id=envelope["session_id"], architecture="independent")
                record = memory.add_session(user_id=user_id, **envelope, current_date="2026-09-27")
                attempted_records.append(record)
                _emit({"stage": "ingest", "architecture": "independent", "conversation": ci,
                       "session": envelope["session_id"], "session_index": index + 1,
                       "sessions_in_stage": len(requested), "facts_added": len(record["fact_records"]),
                       "replayed": record["replayed"]})
            except Exception as error:
                errors.append({"session_id": envelope["session_id"], **_error(api, error)})
                # Later writes cannot skip a missing session or pending transaction.
                break
        facts = memory.get_facts()
        committed = _read(memory.bank_path)["sessions"] if memory.bank_path.exists() else []
        committed_ids = [row["session_id"] for row in committed]
        expected_ids = [row["session_id"] for row in all_envelopes]
        if committed_ids != expected_ids[:len(committed_ids)]:
            raise RuntimeError("Stored sessions are not a chronological prefix of the source conversation")
        requested_ids = [row["session_id"] for row in requested]
        completed_requested = sum(session_id in committed_ids for session_id in requested_ids)
        summary = {"conversation": ci, "architecture": "independent", "user_id": user_id,
                   "sessions_requested": len(requested), "sessions_completed": len(committed),
                   "requested_sessions_completed": completed_requested, "sessions_total": len(all_envelopes),
                   "session_ids": committed_ids, "uncompleted_requested_session_ids": [sid for sid in requested_ids if sid not in committed_ids],
                   "facts": len(facts), "facts_sha256": _hash(facts), "complete": committed_ids == expected_ids,
                   "errors": errors, "elapsed_seconds": time.perf_counter() - start,
                   "smoke_run": manifest["mode"] == "smoke", "source_support_verified": False,
                   "protocol_sha256": manifest["protocol_sha256"], "input_unit": "complete_session",
                   "date_and_history_adapter": "independent_explicit_dates_last_10_max300chars"}
        dump_json(conv_dir / "facts.json", facts)
        dump_json(conv_dir / "ingestion_summary.json", summary)
        return summary
    finally:
        memory.close()


def _ingestion(conv_dir, manifest, allow_partial):
    summary = _read(conv_dir / "ingestion_summary.json")
    facts = _read(conv_dir / "facts.json")
    if summary.get("architecture") != "independent" or summary.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise RuntimeError("Ingestion artifact protocol/architecture does not match this run")
    if summary.get("smoke_run") != (manifest["mode"] == "smoke"):
        raise RuntimeError("Ingestion artifact smoke/full mode does not match this run")
    if _hash(facts) != summary.get("facts_sha256"):
        raise RuntimeError("facts.json no longer matches its ingestion snapshot")
    if not summary["complete"] and not allow_partial:
        raise RuntimeError("Full evaluation requires complete conversation ingestion")
    bank = _read(conv_dir / "independent_store" / "fact_bank.json")
    if _hash(bank["facts"]) != summary["facts_sha256"]:
        raise RuntimeError("Fact bank changed after facts.json export; rerun ingestion export")
    return summary, facts


def _query_scope(item, ci, max_questions):
    all_queries = list(queries(item, ci))
    requested = all_queries[:max_questions] if max_questions is not None else all_queries
    if len({q["id"] for q in requested}) != len(requested):
        raise RuntimeError("QA IDs are not unique")
    return all_queries, requested


def _retrieval_binding(q, summary, manifest):
    return {"query_id": q["id"], "question": q["question"], "architecture": "independent",
            "facts_sha256": summary["facts_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "smoke_run": manifest["mode"] == "smoke", "partial_conversation_smoke": not summary["complete"],
            "pool_protocol": "independent_bm25_dense_union_rrf_top60"}


def _check_binding(row, expected, kind):
    if row.get("status") != "success" or any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError(kind + " cache has stale input, protocol, snapshot or smoke/full metadata")


def _verify_retrieval(row, expected):
    _check_binding(row, expected, "Retrieval")
    if not isinstance(row.get("pool60"), list) or row.get("content_sha256") != _hash({"pool60": row["pool60"]}):
        raise RuntimeError("Retrieval candidate content checksum changed; use a new run directory")


def _verify_qa_content(row):
    if row.get("result_sha256") != _hash({key: value for key, value in row.items() if key != "result_sha256"}):
        raise RuntimeError("QA cached result checksum changed; use a new run directory")


def retrieve_one(run_path, ci, max_questions=None, allow_partial=False, workers=4):
    _positive(workers, "workers")
    manifest = ensure_run(run_path, allow_partial=allow_partial, limit=max_questions)
    conv_dir = Path(run_path) / f"conv_{ci}"
    summary, _ = _ingestion(conv_dir, manifest, allow_partial)
    all_queries, requested = _query_scope(_conversation(ci), ci, max_questions)
    api = ModelAPI(conv_dir)
    memory = IndependentMemory(api, conv_dir / "independent_store")

    def retrieve(q):
        path = conv_dir / "retrieval" / (q["id"] + ".json")
        expected = _retrieval_binding(q, summary, manifest)
        if path.exists():
            _verify_retrieval(_read(path), expected)
            return {"query_id": q["id"], "cached": True}
        start = time.perf_counter()
        scoped = api.scoped(conversation=ci, query_id=q["id"], architecture="independent", method="shared_retrieval")
        query_memory = IndependentMemory(scoped, conv_dir / "independent_store")
        try:
            candidates = query_memory.search(q["question"], top_k=60)
        finally:
            query_memory.close()
        dump_json(path, {**expected, "status": "success", "pool60": candidates,
                         "content_sha256": _hash({"pool60": candidates}),
                         "retrieval_seconds": time.perf_counter() - start})
        _emit({"stage": "retrieve", "conversation": ci, "query": q["id"], "candidates": len(candidates)})
        return {"query_id": q["id"], "cached": False}

    rows, errors = [], []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(retrieve, q): q["id"] for q in requested}
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as error:
                    entry = {"query_id": futures[future], **_error(api, error)}
                    errors.append(entry)
                    dump_json(conv_dir / "retrieval_errors" / (entry["query_id"] + ".json"), entry)
        report = {"conversation": ci, "architecture": "independent", "requested_queries": len(requested),
                  "conversation_queries": len(all_queries), "requested_query_ids": [q["id"] for q in requested],
                  "completed": len(rows), "cached": sum(row["cached"] for row in rows), "errors": errors,
                  "complete": len(rows) == len(requested), "full_conversation_qa_scope": len(requested) == len(all_queries),
                  "smoke_run": manifest["mode"] == "smoke", "facts_sha256": summary["facts_sha256"],
                  "partial_conversation_smoke": not summary["complete"], "protocol_sha256": manifest["protocol_sha256"]}
        dump_json(conv_dir / "retrieval_summary.json", report)
        return report
    finally:
        memory.close()


def qa_one(run_path, ci, methods=METHODS, max_questions=None, workers=4, allow_partial=False):
    _positive(workers, "workers")
    methods = list(methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("Unique methods must be chosen from rank,mmr,jev_single,conditional")
    manifest = ensure_run(run_path, allow_partial=allow_partial, limit=max_questions)
    conv_dir = Path(run_path) / f"conv_{ci}"
    summary, _ = _ingestion(conv_dir, manifest, allow_partial)
    item = _conversation(ci)
    all_queries, requested = _query_scope(item, ci, max_questions)
    envelopes = list(session_envelopes(item))
    committed_ids = set(summary["session_ids"])
    committed_dates = [row["observation_date"] for row in envelopes if row["session_id"] in committed_ids]
    if not committed_dates:
        raise RuntimeError("No committed session is available for QA")
    reference_date = committed_dates[-1]  # Never use an unseen future date in partial smoke.
    api = ModelAPI(conv_dir)

    def evaluate(q):
        successful, errors = [], []
        for method in methods:
            try:
                retrieval = _read(conv_dir / "retrieval" / (q["id"] + ".json"))
                _verify_retrieval(retrieval, _retrieval_binding(q, summary, manifest))
                expected = {**_retrieval_binding(q, summary, manifest), "method": method,
                            "reference_sha256": _hash(q["reference"]), "retrieval_sha256": _hash(retrieval),
                            "reference_date": reference_date}
                path = conv_dir / "qa" / method / (q["id"] + ".json")
                if path.exists():
                    row = _read(path)
                    _check_binding(row, expected, "QA")
                    _verify_qa_content(row)
                else:
                    start = time.perf_counter()
                    method_api = api.scoped(conversation=ci, query_id=q["id"], method=method, architecture="independent")
                    selection = select(q["question"], retrieval["pool60"], method_api, method=method, top_k=20,
                                       seed_count=8, round_sizes=(6, 6), shortlist_size=32,
                                       conditional_threshold=.5, mmr_weight=.7, max_workers=8)
                    selected = selection.selected
                    scoring = answer_and_judge(method_api, q["question"], q["reference"], selected,
                                               reference_date=reference_date)
                    row = {**expected, "status": "success", "conversation": ci, "category": q["category"],
                           "reference": q["reference"], "selected_ids": [f["id"] for f in selected],
                           "selection_trace": selection.trace, **scoring,
                           "elapsed_seconds": time.perf_counter() - start}
                    _validate_scoring(row)
                    row["result_sha256"] = _hash(row)
                    dump_json(path, row)
                    _emit({"stage": "qa", "conversation": ci, "query": q["id"], "method": method,
                           "correct": row["judgment"]["correct"], "f1": row["token_f1"]})
                _validate_scoring(row)
                successful.append(row)
            except Exception as error:
                entry = {"query_id": q["id"], "method": method, **_error(api, error)}
                errors.append(entry)
                dump_json(conv_dir / "qa_errors" / method / (q["id"] + ".json"), entry)
        return successful, errors

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate, q): q["id"] for q in requested}
        for future in as_completed(futures):
            try:
                successful, failures = future.result()
                rows.extend(successful)
                errors.extend(failures)
            except Exception as error:
                # A bookkeeping failure must also retain every method denominator.
                for method in methods:
                    errors.append({"query_id": futures[future], "method": method, **_error(api, error)})
    report = {"conversation": ci, "architecture": "independent", "requested_queries": len(requested),
              "conversation_queries": len(all_queries), "requested_query_ids": [q["id"] for q in requested],
              "requested_evaluations": len(requested) * len(methods), "errors": errors, "methods": {},
              "smoke_run": manifest["mode"] == "smoke", "partial_conversation_smoke": not summary["complete"],
              "full_conversation_qa_scope": len(requested) == len(all_queries), "facts_sha256": summary["facts_sha256"],
              "protocol_sha256": manifest["protocol_sha256"], "arm_mapping": {"C-off": "rank", "C-JEV": "conditional"}}
    denominator = len(requested)
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        correct = sum(row["judgment"]["correct"] for row in group)
        f1 = sum(row["token_f1"] for row in group)
        report["methods"][method] = {"completed": len(group), "requested": denominator,
            "failed": denominator - len(group), "strict_correct": correct,
            "accuracy_all_requested": correct / denominator if denominator else None,
            "f1_all_requested": f1 / denominator if denominator else None,
            "accuracy_successful_only": correct / len(group) if group else None,
            "f1_successful_only": f1 / len(group) if group else None,
            "technical_failure_rate": (denominator - len(group)) / denominator if denominator else None}
    report["complete"] = all(row["completed"] == denominator for row in report["methods"].values())
    dump_json(conv_dir / "qa_summary.json", report)
    return report


def _validate_scoring(row):
    if not isinstance(row.get("judgment", {}).get("correct"), bool):
        raise ValueError("Expected a boolean strict judgment")
    f1 = row.get("token_f1")
    if isinstance(f1, bool) or not isinstance(f1, (int, float)) or not math.isfinite(f1) or not 0 <= f1 <= 1:
        raise ValueError("Invalid token F1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["ingest", "retrieve", "qa"])
    parser.add_argument("--run-dir", default=str(ROOT / "runs/independent_top20_v1"))
    parser.add_argument("--conversations", default="0", help="Comma-separated original conversation indices")
    parser.add_argument("--max-sessions", type=int, help="Ingest smoke only; requires --allow-partial-smoke")
    parser.add_argument("--max-questions", type=int, help="Retrieve/QA smoke only; requires --allow-partial-smoke")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-partial-smoke", action="store_true", help="Permanently labels this run directory as smoke")
    args = parser.parse_args(argv)
    if args.stage != "ingest" and args.max_sessions is not None:
        parser.error("--max-sessions is only valid for ingest")
    if args.stage == "ingest" and args.max_questions is not None:
        parser.error("--max-questions is only valid for retrieve/qa")
    if (args.max_sessions is not None or args.max_questions is not None) and not args.allow_partial_smoke:
        parser.error("Sample limits require --allow-partial-smoke and a separate run directory")
    try:
        cis = [int(x.strip()) for x in args.conversations.split(",")]
        if not cis or len(set(cis)) != len(cis):
            raise ValueError("Conversation indices must be unique")
        _positive(args.workers, "workers")
        _positive(args.max_sessions, "max-sessions")
        _positive(args.max_questions, "max-questions")
        for ci in cis:
            _conversation(ci)
        methods = [method.strip() for method in args.methods.split(",")]
        if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
            raise ValueError("Invalid --methods")
    except ValueError as error:
        parser.error(str(error))
    run_path = Path(args.run_dir)
    outputs = []
    for ci in cis:
        if args.stage == "ingest":
            result = ingest_one(run_path, ci, args.max_sessions, args.allow_partial_smoke)
        elif args.stage == "retrieve":
            result = retrieve_one(run_path, ci, args.max_questions, args.allow_partial_smoke, args.workers)
        else:
            result = qa_one(run_path, ci, methods, args.max_questions, args.workers, args.allow_partial_smoke)
        outputs.append(result)
        dump_json(run_path / ("last_" + args.stage + "_summary.json"), outputs)
        _emit(result)
    # A full ingestion failure or requested query failure propagates to orchestration.
    failed = any(row.get("errors") or (row["requested_sessions_completed"] != row["sessions_requested"]
                                       if args.stage == "ingest" else not row.get("complete", False))
                 for row in outputs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
