"""Explicit live stages with durable artifacts. No stage starts on import."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["MEM0_DIR"] = str(ROOT / "runtime/live_mem0")
os.environ["FASTEMBED_CACHE_PATH"] = str(ROOT / "cache/fastembed")
os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from jev_mem.data import load_data, session_envelopes, queries
from jev_mem.evaluation import answer_and_judge, READER_SYSTEM, JUDGE_SYSTEM
from jev_mem.providers import ModelAPI, ExtractionLLM, Embeddings, dump_json
from jev_mem.mem0_adapter import Mem0SessionAdapter
from jev_mem.selection import select, tag_fact, PROMPT_VERSION


CACHE_SCHEMA = "session-top20-artifacts-v2"


class CacheProtocolMismatch(RuntimeError):
    """A durable artifact belongs to different inputs/protocol; do not overwrite."""


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _retrieval_fingerprint(q, summary, facts_hash):
    return _fingerprint({"schema": CACHE_SCHEMA, "query_id": q["id"], "question": q["question"],
                         "facts_sha256": facts_hash, "user_id": summary["user_id"],
                         "sessions_completed": summary["sessions_completed"],
                         "sessions_total": summary["sessions_total"], "complete": summary["complete"],
                         "retrieval": {"mem0_version": "2.2.1", "top_k": [20, 60], "threshold": 0.1,
                                       "rerank": False, "embedder": "gemini-embedding-001", "dimensions": 768}})


def _verify_retrieval(retrieval, q, summary, facts_hash):
    expected = _retrieval_fingerprint(q, summary, facts_hash)
    if (retrieval.get("cache_schema") != CACHE_SCHEMA or retrieval.get("request_fingerprint") != expected
            or retrieval.get("partial_conversation_smoke") != (not summary["complete"])
            or retrieval.get("query_id") != q["id"] or retrieval.get("question") != q["question"]):
        raise CacheProtocolMismatch("Retrieval cache does not match current facts/question/protocol; use a fresh run directory")
    content = {key: retrieval.get(key) for key in ("sdk20", "pool60")}
    if retrieval.get("content_fingerprint") != _fingerprint(content):
        raise CacheProtocolMismatch("Retrieval cache content checksum differs; use a fresh run directory")
    return expected


def _qa_fingerprint(q, method, retrieval, reference_date):
    return _fingerprint({"schema": CACHE_SCHEMA, "query_id": q["id"], "question": q["question"],
                         "reference": q["reference"], "reference_date": reference_date, "method": method,
                         "reader_model": "gemini-2.5-flash", "judge_model": "gemini-3.5-flash-lite",
                         "jev_model": "jev-1.13.0", "reader_system": READER_SYSTEM, "judge_system": JUDGE_SYSTEM,
                         "selector_prompt_version": PROMPT_VERSION,
                         "selector_code_sha256": hashlib.sha256((ROOT / "jev_mem/selection.py").read_bytes()).hexdigest(),
                         "qa_code_sha256": hashlib.sha256((ROOT / "jev_mem/evaluation.py").read_bytes()).hexdigest(),
                         "top_k": 20, "retrieval_request": retrieval["request_fingerprint"],
                         "retrieval_content": retrieval["content_fingerprint"],
                         "partial_conversation_smoke": retrieval["partial_conversation_smoke"]})


def summarize_qa_artifacts(conv_dir, ci, qs, methods, *, expected_fingerprints, errors=(), allow_partial=False):
    """Re-read each durable method result, including successes before a later failure.

    Every method uses the same complete requested-query denominator. Failed,
    absent, stale, or invalid rows remain missing (zero contribution), and are
    explicitly counted; these are conservative incomplete-run metrics.
    """
    query_ids = [q["id"] for q in qs]
    if len(set(query_ids)) != len(query_ids) or len(set(methods)) != len(methods):
        raise ValueError("Duplicate requested query IDs or methods")
    report = {"conversation": ci, "requested_queries": len(qs), "requested_query_ids": query_ids,
              "errors": list(errors), "methods": {}, "smoke_only": bool(allow_partial),
              "denominator_policy": "all_requested_queries_including_failed_or_missing",
              "api_accounting": "api_calls.jsonl: context attributes method/query; cache hits incur no new model usage"}
    for method in methods:
        group, missing_ids = [], []
        for q in qs:
            path = Path(conv_dir) / "qa" / method / (q["id"] + ".json")
            if not path.exists():
                missing_ids.append(q["id"])
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                expected = expected_fingerprints.get((q["id"], method))
                if (not expected or row.get("cache_schema") != CACHE_SCHEMA
                        or row.get("request_fingerprint") != expected or row.get("query_id") != q["id"]
                        or row.get("method") != method or row.get("conversation") != ci):
                    raise CacheProtocolMismatch("QA artifact identity/protocol fingerprint mismatch")
                if not isinstance(row.get("partial_conversation_smoke"), bool):
                    raise ValueError("Missing explicit partial-conversation marker")
                if row["partial_conversation_smoke"] and not allow_partial:
                    raise CacheProtocolMismatch("Partial-conversation result is not eligible for formal aggregation")
                correct = row["judgment"]["correct"]
                f1 = row["token_f1"]
                if not isinstance(correct, bool) or isinstance(f1, bool) or not isinstance(f1, (int, float)):
                    raise ValueError("Invalid correctness/F1 types")
                if not math.isfinite(f1) or not 0 <= f1 <= 1:
                    raise ValueError("F1 must be finite and between zero and one")
                report["smoke_only"] |= row["partial_conversation_smoke"]
                group.append(row)
            except (KeyError, ValueError, TypeError, OSError, CacheProtocolMismatch) as error:
                missing_ids.append(q["id"])
                report["errors"].append({"query_id": q["id"], "method": method,
                                         "error_type": type(error).__name__, "stage": "summary_artifact_validation"})
        correct_total = sum(row["judgment"]["correct"] for row in group)
        f1_total = sum(row["token_f1"] for row in group)
        report["methods"][method] = {"completed": len(group), "requested": len(qs), "missing": len(missing_ids),
            "missing_query_ids": missing_ids, "strict_correct": correct_total, "token_f1_sum": f1_total,
            "accuracy_all_requested": correct_total / len(qs) if qs else None,
            "f1_all_requested": f1_total / len(qs) if qs else None,
            "complete": len(group) == len(qs)}
    report["evaluation_complete"] = bool(qs) and all(row["complete"] for row in report["methods"].values())
    return report


def aggregate_qa_reports(reports, methods, expected_benchmark_ids=None):
    """Pool requested cohorts without averaging conversation-level percentages."""
    if expected_benchmark_ids is None:
        expected_benchmark_ids = {q["id"] for ci, item in enumerate(load_data()) for q in queries(item, ci)}
    requested_ids = [qid for report in reports for qid in report["requested_query_ids"]]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Overlapping conversation/query cohorts cannot be pooled")
    if not set(requested_ids) <= set(expected_benchmark_ids):
        raise ValueError("Unexpected query IDs in summary")
    count = len(requested_ids)
    smoke = any(report.get("smoke_only", False) for report in reports)
    output = {"requested_queries": count, "benchmark_queries": len(expected_benchmark_ids),
              "conversations": [report["conversation"] for report in reports], "smoke_only": smoke,
              "full_benchmark_coverage": set(requested_ids) == set(expected_benchmark_ids),
              "report_scope": "smoke_only" if smoke else "requested_cohort",
              "denominator_policy": "same_all_requested_queries_for_every_method", "methods": {},
              "development_conversations": [0, 1],
              "interpretation": "Development conversations 0/1; full 1540 includes development and is not a clean holdout."}
    for method in methods:
        groups = []
        for report in reports:
            group = report["methods"].get(method)
            if group is None or group["requested"] != report["requested_queries"]:
                raise ValueError("Every pooled method must use the identical requested cohort")
            groups.append(group)
        done = sum(group["completed"] for group in groups)
        correct = sum(group["strict_correct"] for group in groups)
        f1 = sum(group["token_f1_sum"] for group in groups)
        output["methods"][method] = {"completed": done, "requested": count, "missing": count - done,
            "strict_correct": correct, "token_f1_sum": f1,
            "accuracy_all_requested": correct / count if count else None,
            "f1_all_requested": f1 / count if count else None}
    output["evaluation_complete"] = bool(count) and all(group["missing"] == 0 for group in output["methods"].values())
    output["formal_full_benchmark_ready"] = output["full_benchmark_coverage"] and output["evaluation_complete"] and not smoke
    if output["formal_full_benchmark_ready"]:
        output["report_scope"] = "full_1540_including_development"
    return output


def open_memory(conv_dir, api):
    from mem0 import Memory
    conv_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "facts",
            "embedding_model_dims": 768, "path": str(conv_dir / "qdrant"), "on_disk": True}},
        "llm": {"provider": "gemini", "config": {"model": "gemini-2.5-flash",
            "api_key": api.keys["GEMINI_API_KEY"], "vertexai": False, "max_tokens": 16384, "temperature": 0}},
        "embedder": {"provider": "gemini", "config": {"model": "gemini-embedding-001",
            "api_key": api.keys["GEMINI_API_KEY"], "embedding_dims": 768}},
        "history_db_path": str(conv_dir / "history.sqlite")}
    memory = Memory.from_config(config)
    memory.llm.client.close()
    memory.embedding_model.client.close()
    memory.llm = ExtractionLLM(api)
    memory.embedding_model = Embeddings(api)
    return memory


def close_memory(memory):
    memory.close()
    memory.vector_store.client.close()


def ingest_one(run_path, ci, max_sessions=None):
    from mem0.utils import spacy_models
    conv_dir = Path(run_path) / f"conv_{ci}"
    api = ModelAPI(conv_dir).scoped(phase="ingestion", conversation=ci)
    item = load_data()[ci]
    envelopes = list(session_envelopes(item))
    if max_sessions:
        envelopes = envelopes[:max_sessions]
    prior_summary = conv_dir / "ingestion_summary.json"
    if prior_summary.exists():
        earlier = json.loads(prior_summary.read_text(encoding="utf-8"))
        if earlier.get("sessions_completed", 0) > len(envelopes):
            raise CacheProtocolMismatch("Cannot replace a larger ingested fact snapshot with a smaller smoke prefix")
    user_id = f"locomo_c{ci}_session_v1"
    memory = open_memory(conv_dir, api)
    writer = Mem0SessionAdapter(memory, conv_dir / "session_audit", current_date="2026-09-27")
    records = []
    started = time.perf_counter()
    try:
        # Require genuine NLP/BM25 readiness, never evaluate a silent fallback.
        if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None:
            raise RuntimeError("spaCy dependency/model unavailable; refusing degraded baseline")
        if memory.vector_store._get_bm25_encoder() is None:
            raise RuntimeError("BM25 encoder unavailable; refusing degraded baseline")
        for index, envelope in enumerate(envelopes):
            record = writer.add_session(user_id=user_id, **envelope)
            records.append(record)
            print(json.dumps({"stage": "ingest", "conversation": ci, "session": envelope["session_id"],
                              "session_index": index + 1, "sessions_in_stage": len(envelopes),
                              "facts_added": len(record["results"]), "replayed": record["replayed"]}), flush=True)
        facts = [f for record in records for f in record["fact_records"]]
        # Vectors are retrieved from the real persisted store, not re-embedded.
        for fact in facts:
            point = memory.vector_store.client.retrieve(collection_name="facts", ids=[fact["id"]],
                                                        with_vectors=True, with_payload=False)[0]
            fact["embedding"] = point.vector.get("") if isinstance(point.vector, dict) else point.vector
        dump_json(conv_dir / "facts.json", facts)
        summary = {"conversation": ci, "sessions_completed": len(records), "sessions_total": len(list(session_envelopes(item))),
                   "facts": len(facts), "complete": len(records) == len(list(session_envelopes(item))),
                   "elapsed_seconds": time.perf_counter() - started, "user_id": user_id,
                   "date_and_history_adapter": True, "source_support_verified": False}
        dump_json(conv_dir / "ingestion_summary.json", summary)
        return summary
    finally:
        close_memory(memory)


def retrieve_one(run_path, ci, max_questions=None, allow_partial=False):
    conv_dir = Path(run_path) / f"conv_{ci}"
    summary = json.loads((conv_dir / "ingestion_summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] and not allow_partial:
        raise RuntimeError("QA retrieval requires the full conversation ingestion")
    facts = json.loads((conv_dir / "facts.json").read_text(encoding="utf-8"))
    facts_hash = hashlib.sha256((conv_dir / "facts.json").read_bytes()).hexdigest()
    by_id = {f["id"]: f for f in facts}
    api = ModelAPI(conv_dir).scoped(phase="retrieval", conversation=ci)
    memory = open_memory(conv_dir, api)
    qs = list(queries(load_data()[ci], ci))
    if max_questions:
        qs = qs[:max_questions]
    try:
        for q in qs:
            path = conv_dir / "retrieval" / (q["id"] + ".json")
            if path.exists():
                _verify_retrieval(json.loads(path.read_text(encoding="utf-8")), q, summary, facts_hash)
                continue
            memory.embedding_model = Embeddings(api.scoped(query_id=q["id"]))
            start = time.perf_counter()
            r20 = memory.search(q["question"], top_k=20, filters={"user_id": summary["user_id"]}, rerank=False)
            r60 = memory.search(q["question"], top_k=60, filters={"user_id": summary["user_id"]}, rerank=False)
            def enrich(rows):
                return [{**by_id[row["id"]], "score": row.get("score"), "sources": ["mem0_hybrid"]}
                        for row in rows["results"]]
            row = {"cache_schema": CACHE_SCHEMA, "request_fingerprint": _retrieval_fingerprint(q, summary, facts_hash),
                   "query_id": q["id"], "question": q["question"], "sdk20": enrich(r20), "pool60": enrich(r60),
                   "retrieval_seconds": time.perf_counter() - start, "facts_sha256": facts_hash,
                   "partial_conversation_smoke": not summary["complete"]}
            row["content_fingerprint"] = _fingerprint({key: row[key] for key in ("sdk20", "pool60")})
            dump_json(path, row)
        return {"conversation": ci, "retrieval_queries": len(qs)}
    finally:
        close_memory(memory)


def qa_one(run_path, ci, methods, max_questions=None, workers=4, allow_partial=False):
    conv_dir = Path(run_path) / f"conv_{ci}"
    api = ModelAPI(conv_dir)
    summary = json.loads((conv_dir / "ingestion_summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] and not allow_partial:
        raise RuntimeError("QA requires full conversation ingestion outside explicit smoke")
    facts_hash = hashlib.sha256((conv_dir / "facts.json").read_bytes()).hexdigest()
    item = load_data()[ci]
    reference_date = list(session_envelopes(item))[-1]["observation_date"]
    qs = list(queries(item, ci))
    if max_questions:
        qs = qs[:max_questions]
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("Methods must be nonempty and unique")
    expected_fingerprints = {}
    def evaluate(q):
        retrieval = json.loads((conv_dir / "retrieval" / (q["id"] + ".json")).read_text(encoding="utf-8"))
        _verify_retrieval(retrieval, q, summary, facts_hash)
        if retrieval["partial_conversation_smoke"] and not allow_partial:
            raise RuntimeError("Refusing partial-conversation QA outside smoke")
        for method in methods:
            expected_fingerprints[(q["id"], method)] = _qa_fingerprint(q, method, retrieval, reference_date)
        output = []
        def run_method(method):
            fingerprint = expected_fingerprints[(q["id"], method)]
            path = conv_dir / "qa" / method / (q["id"] + ".json")
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("cache_schema") != CACHE_SCHEMA or row.get("request_fingerprint") != fingerprint:
                    raise CacheProtocolMismatch("QA cache has different inputs/prompts/selection; use a fresh run directory")
                return row
            start = time.perf_counter()
            method_api = api.scoped(phase="qa", conversation=ci, query_id=q["id"], method=method)
            if method == "sdk20":
                selected, trace = retrieval["sdk20"], {"method": "sdk20"}
            else:
                selection = select(q["question"], retrieval["pool60"], method_api, method=method, top_k=20, max_workers=8)
                selected, trace = selection.selected, selection.trace
            scoring = answer_and_judge(method_api, q["question"], q["reference"], selected, reference_date=reference_date)
            row = {"cache_schema": CACHE_SCHEMA, "request_fingerprint": fingerprint,
                   "query_id": q["id"], "conversation": ci, "category": q["category"],
                   "question": q["question"], "reference": q["reference"], "method": method,
                   "selected_ids": [f["id"] for f in selected], "selection_trace": trace, **scoring,
                   "elapsed_seconds": time.perf_counter() - start,
                   "partial_conversation_smoke": retrieval["partial_conversation_smoke"]}
            dump_json(path, row)
            print(json.dumps({"stage": "qa", "query": q["id"], "method": method,
                              "correct": row["judgment"]["correct"], "f1": row["token_f1"]}), flush=True)
            return row
        for method in methods:
            try:
                output.append(run_method(method))
            except Exception as error:
                errors.append({"query_id": q["id"], "method": method,
                               "error": api._redact(error)[:1800]})
        return output
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate, q): q["id"] for q in qs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append({"query_id": futures[future], "error": api._redact(error)[:1800]})
    report = summarize_qa_artifacts(conv_dir, ci, qs, methods, expected_fingerprints=expected_fingerprints,
                                    errors=errors, allow_partial=allow_partial)
    dump_json(conv_dir / "qa_summary.json", report)
    return report


def tag_one(run_path, ci, limit=None):
    conv_dir = Path(run_path) / f"conv_{ci}"
    api = ModelAPI(conv_dir)
    facts = json.loads((conv_dir / "facts.json").read_text(encoding="utf-8"))
    if limit:
        facts = facts[:limit]
    sessions = {s["session_id"]: s for s in session_envelopes(load_data()[ci])}
    def process(fact):
        path = conv_dir / "tags" / (fact["id"] + ".json")
        sid = fact["input_scope"]["session_id"]
        # Context is write-time visible and not asserted to be verified support.
        context = "\n".join(m["content"] for m in sessions[sid]["messages"])
        fingerprint = _fingerprint({"schema": CACHE_SCHEMA, "fact": fact, "context": context,
                                    "prompt_version": PROMPT_VERSION, "model": "jev-1.13.0",
                                    "selector_code_sha256": hashlib.sha256((ROOT / "jev_mem/selection.py").read_bytes()).hexdigest()})
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("cache_schema") != CACHE_SCHEMA or saved.get("request_fingerprint") != fingerprint:
                raise CacheProtocolMismatch("Fact typing cache has changed inputs/protocol; use a fresh run directory")
            return
        tagged = tag_fact({**fact, "source_context": context},
                          api.scoped(phase="write_typing", conversation=ci, fact_id=fact["id"]))
        dump_json(path, {"cache_schema": CACHE_SCHEMA, "request_fingerprint": fingerprint,
                         "fact_id": fact["id"], "source_context_status": "whole_session_input_scope_not_verified_support", **tagged})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(process, facts))
    return {"conversation": ci, "tagged_facts": len(facts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "retrieve", "qa", "tag"])
    parser.add_argument("--run-dir", default=str(ROOT / "runs/session_top20_v1"))
    parser.add_argument("--conversations", default="0")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--methods", default="sdk20,rank,mmr,jev_single,conditional")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-partial-smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.max_sessions is not None and args.max_sessions < 1) or (
        args.max_questions is not None and args.max_questions < 1
    ):
        parser.error("workers and optional limits must be positive")
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    cis = [int(x) for x in args.conversations.split(",")]
    outputs = []
    for ci in cis:
        if args.stage == "ingest":
            result = ingest_one(run_path, ci, args.max_sessions)
        elif args.stage == "retrieve":
            result = retrieve_one(run_path, ci, args.max_questions, args.allow_partial_smoke)
        elif args.stage == "tag":
            result = tag_one(run_path, ci)
        else:
            result = qa_one(run_path, ci, args.methods.split(","), args.max_questions, args.workers, args.allow_partial_smoke)
        outputs.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    dump_json(run_path / ("last_" + args.stage + "_summary.json"), outputs)
    if args.stage == "qa":
        aggregate = aggregate_qa_reports(outputs, args.methods.split(","))
        dump_json(run_path / "last_qa_aggregate.json", aggregate)


if __name__ == "__main__":
    main()
