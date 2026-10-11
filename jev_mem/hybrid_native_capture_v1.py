"""One native SDK20 search, two captured channels, one immutable hybrid pool.

No providers, files, stores or requests are opened on import. The caller must use
a NEW store copy in a dedicated process. H0/H1/H2 never rewrite fact content.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import importlib
import math
import os
import threading
import time

from .sdk20_type_adapter import (capture_native_pool, _digest,
                               _complete_query_scores, _complete_fact_scores,
                               CAPTURE_SCHEMA as NATIVE_CAPTURE_SCHEMA)

CAPTURE_SCHEMA = "hybrid-native-capture-v1"
POOL_SCHEMA = "hybrid-native-pool-v1"
SELECT_SCHEMA = "hybrid-fused-type-selection-v1"
ARMS = {"H0": "hybrid_rank", "H1": "hybrid_types", "H2": "hybrid_types_no_relation"}
_ACTIVE = False


def _bank(facts):
    result = {}
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str) or not fact["id"]:
            raise ValueError("bank requires nonempty string IDs")
        if fact["id"] in result or not isinstance(fact.get("text"), str):
            raise ValueError("duplicate bank ID or invalid text")
        result[fact["id"]] = fact
    return result


def _snapshot(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("backend must return a materialized list")
    output = []
    for row in rows:
        get = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
        score = get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("backend score must be finite")
        payload = get("payload")
        if not isinstance(payload, dict):
            raise ValueError("backend payload unavailable")
        output.append({"id": str(get("id")), "score": score, "payload": deepcopy(payload)})
    if len(output) > 80 or len({row["id"] for row in output}) != len(output):
        raise ValueError("backend exceeds 80 or contains duplicate IDs")
    return output


def _fingerprint(record, field):
    return _digest({key: value for key, value in record.items() if key != field})


def _check(record, schema, field):
    if record.get("schema") != schema or record.get(field) != _fingerprint(record, field):
        raise ValueError("schema/fingerprint mismatch")


@contextmanager
def capture_hybrid_native(memory, *, query, filters, bank_facts, _scorer_module=None):
    """Capture EXACTLY one unmodified Memory.search on the supplied new copy.

    Both vector-store queries are forwarded once, returning their exact original
    objects. None/unavailable, empty, raised errors and invalid rows are distinct.
    A failure retains a status/error-type trace in the yielded dictionary.
    Only simple user/agent/run equality filters are supported in this protocol.
    """
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("hybrid capture cannot nest or run concurrently")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if (not isinstance(filters, dict) or not filters or
        any(k not in {"user_id", "agent_id", "run_id"} or
            not isinstance(v, str) or not v or v != v.strip() for k, v in filters.items())):
        raise ValueError("only nonempty simple scope equality filters supported")
    by_id = _bank(bank_facts)
    module = _scorer_module or importlib.import_module("mem0.memory.main")
    store = memory.vector_store
    # Instance attributes must be restored, including absence (bound class methods).
    targets = [(memory, "search"), (store, "search"), (store, "keyword_search")]
    originals = [(obj, name, name in vars(obj), vars(obj).get(name), getattr(obj, name))
                 for obj, name in targets]
    search_original, dense_original, lexical_original = [row[4] for row in originals]
    owner = (os.getpid(), threading.get_ident())
    captured = {"schema": CAPTURE_SCHEMA, "status": "pending", "query": query,
                "effective_native_query": query.strip(),
                "filters": deepcopy(filters), "bank_content_sha256": _digest(bank_facts),
                "channels": {}, "call_counts": {"memory_search": 0, "dense": 0, "keyword": 0},
                "additional_backend_queries": 0, "snapshot_validation_cpu_seconds": 0.0}
    _ACTIVE = True

    def guard(name):
        if owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("capture must stay in its owning process/thread")
        captured["call_counts"][name] += 1
        if captured["call_counts"][name] != 1:
            raise RuntimeError("exactly one call per native search channel allowed")

    def backend(name, original, query_arg, top_k, filter_arg, extra):
        guard(name)
        if top_k != 80 or isinstance(top_k, bool) or filter_arg != filters:
            raise ValueError("native backend depth/scope drift")
        if name == "dense" and query_arg != captured["effective_native_query"]:
            raise ValueError("dense query drift")
        info = {"query": query_arg, "top_k": top_k, "filters": deepcopy(filter_arg)}
        captured["channels"][name] = info
        try:
            result = original(query=query_arg, top_k=top_k, filters=filter_arg, **extra)
        except Exception as exc:
            info.update(status="error", error_type=type(exc).__name__)
            raise
        started = time.process_time()
        try:
            if result is None:
                info.update(status="unavailable", rows=None, returned_count=0)
            else:
                rows = _snapshot(result)
                info.update(status="empty" if not rows else "complete", rows=rows, returned_count=len(rows))
                for row in rows:
                    if row["id"] not in by_id:
                        raise ValueError("backend ID outside bound bank")
                    if any(row["payload"].get(k) != v for k, v in filters.items()):
                        raise ValueError("backend payload scope mismatch")
                    if row["payload"].get("data") != by_id[row["id"]]["text"]:
                        raise ValueError("backend payload text differs from bound bank")
        except Exception as exc:
            info.update(status="invalid", error_type=type(exc).__name__)
            raise
        finally:
            captured["snapshot_validation_cpu_seconds"] += time.process_time() - started
        return result

    def dense(query, vectors, top_k=5, filters=None):
        captured["query_vector_sha256"] = _digest(vectors)
        captured["query_vector_dimensions"] = len(vectors)
        return backend("dense", dense_original, query, top_k, filters, {"vectors": vectors})

    def lexical(query, top_k=5, filters=None):
        return backend("keyword", lexical_original, query, top_k, filters, {})

    def search(query, *, top_k=20, filters=None, threshold=.1, rerank=False,
               explain=False, reference_date=None, show_expired=False, **kwargs):
        guard("memory_search")
        if (query != captured["query"] or filters != captured["filters"] or
            top_k != 20 or isinstance(top_k, bool) or threshold != .1 or
            rerank is not False or explain is not False or reference_date is not None or
            show_expired is not False or kwargs):
            raise ValueError("frozen native search arguments drift")
        result = search_original(query, top_k=top_k, filters=filters, threshold=threshold,
                                 rerank=rerank, explain=explain,
                                 reference_date=reference_date, show_expired=show_expired)
        captured["public_native20"] = deepcopy(result["results"])
        return result

    try:
        memory.search, store.search, store.keyword_search = search, dense, lexical
        with capture_native_pool(_scorer_module=module) as native:
            yield captured
        if captured["call_counts"] != {"memory_search": 1, "dense": 1, "keyword": 1}:
            raise RuntimeError("missing channel in native search")
        if captured["channels"]["keyword"]["status"] == "unavailable":
            raise RuntimeError("native keyword channel unavailable; no fallback")
        if captured["channels"]["dense"]["status"] == "unavailable":
            raise RuntimeError("native dense channel unavailable")
        if len(native["eligible_ids"]) < 20:
            raise ValueError("native returned fewer than the required 20 Reader facts")
        public = captured["public_native20"]
        if [row["id"] for row in public] != native["base20_ids"]:
            raise ValueError("native public20 differs from exact scorer replay")
        for row, scored in zip(public, native["base20"]):
            if row["score"] != scored["score"]:
                raise ValueError("native public score differs from scorer replay")
        captured["native_capture"] = deepcopy(native)
        rows = captured["channels"]["keyword"]["rows"]
        legal = [row for row in rows if row["score"] > 0 and not module._payload_is_expired(row["payload"])]
        captured["lexical_legal_ids"] = [row["id"] for row in legal[:20]]
        captured["lexical_excluded"] = [{"id": row["id"], "reason": "nonpositive" if row["score"] <= 0 else "expired"}
                                         for row in rows if row not in legal]
        captured["status"] = "complete"
    except Exception as exc:
        captured.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        for obj, name, had_instance, instance_value, _ in reversed(originals):
            if had_instance:
                setattr(obj, name, instance_value)
            else:
                delattr(obj, name)
        _ACTIVE = False
        captured["capture_sha256"] = _fingerprint(captured, "capture_sha256")


def compare_native20(captured, expected_rows, bank_facts):
    """Compare historical ordered20 score/text/date; no native re-query."""
    _check(captured, CAPTURE_SCHEMA, "capture_sha256")
    by_id = _bank(bank_facts)
    actual = captured["native_capture"]["base20"]
    if len(expected_rows) != 20 or [r["id"] for r in expected_rows] != [r["id"] for r in actual]:
        raise ValueError("historical ordered native20 did not replay")
    for new, old in zip(actual, expected_rows):
        fact = by_id[new["id"]]
        if abs(new["score"] - old["score"]) > 1e-7:
            raise ValueError("historical native score differs")
        if "text" in old and (old["text"] != fact["text"] or old.get("observation_date") != fact.get("observation_date")):
            raise ValueError("historical native fact text/date differs")
        if "payload" in old and old["payload"] != new["payload"]:
            raise ValueError("historical native payload differs")
    return {"matched": True, "ordered_ids": [r["id"] for r in actual], "expected_rows_sha256": _digest(expected_rows)}


def build_hybrid_pool(captured, bank_facts, *, bank_sha256, expected_native20, expected_native_capture):
    """ID-only equal RRF60. Historical replay is a mandatory admission gate."""
    _check(captured, CAPTURE_SCHEMA, "capture_sha256")
    if captured["status"] != "complete" or captured["bank_content_sha256"] != _digest(bank_facts):
        raise ValueError("capture incomplete or bound bank changed")
    if not isinstance(bank_sha256, str) or len(bank_sha256) != 64 or any(c not in "0123456789abcdef" for c in bank_sha256):
        raise ValueError("bank file SHA256 required")
    by_id = _bank(bank_facts)
    replay = compare_native20(captured, expected_native20, bank_facts)
    native = captured["native_capture"]
    _check(expected_native_capture, NATIVE_CAPTURE_SCHEMA, "capture_sha256")
    if expected_native_capture.get("eligible_ids") != native["eligible_ids"]:
        raise ValueError("historical complete native ordered pool did not replay")
    expected_eligible = expected_native_capture.get("eligible", [])
    if len(expected_eligible) != len(native["eligible"]):
        raise ValueError("historical native pool rows unavailable")
    for old, new in zip(expected_eligible, native["eligible"]):
        if old["id"] != new["id"] or abs(old["score"] - new["score"]) > 1e-7 or old["payload"] != new["payload"]:
            raise ValueError("historical complete native row differs")
    n = {fact_id: rank for rank, fact_id in enumerate(native["eligible_ids"], 1)}
    l = {fact_id: rank for rank, fact_id in enumerate(captured["lexical_legal_ids"], 1)}
    scores = {fid: (Fraction(1, 60 + n[fid]) if fid in n else Fraction(0)) +
                   (Fraction(1, 60 + l[fid]) if fid in l else Fraction(0)) for fid in n.keys() | l.keys()}
    order = sorted(scores, key=lambda fid: (-scores[fid], n.get(fid, math.inf), l.get(fid, math.inf), fid))
    ids = order[:80]
    if len(ids) != min(80, len(order)) or not set(native["base20_ids"] + list(l)).issubset(ids):
        raise ValueError("hybrid pool budget/retention invariant failed")
    output = {"schema": POOL_SCHEMA, "capture_sha256": captured["capture_sha256"],
              "bank_sha256": bank_sha256, "bank_content_sha256": _digest(bank_facts),
              "query": captured["query"], "filters": captured["filters"],
              "pool_ids": ids, "facts": [deepcopy(by_id[fid]) for fid in ids],
              "native20_replay": replay, "native20_ids": list(native["base20_ids"]),
              "native_pool_replay": {"matched": True, "expected_capture_sha256": _digest(expected_native_capture)},
              "ranking": [{"id": fid, "native_rank": n.get(fid), "lexical_rank": l.get(fid),
                           "fused_rank": rank, "fused_score": float(scores[fid]),
                           "fused_score_fraction": str(scores[fid]),
                           "outside_native80": fid not in n} for rank, fid in enumerate(order, 1)],
              "budget": {"native_backend_limit": 80, "keyword_backend_limit": 80,
                         "native_backend_returned": captured["channels"]["dense"]["returned_count"],
                         "keyword_backend_returned": captured["channels"]["keyword"]["returned_count"],
                         "fusion_retained_slots": len(n) + len(l), "union_unique_ids": len(order),
                         "pool_unique_ids": len(ids), "reader_fact_cap": 20,
                         "additional_backend_queries": 0,
                         "entity_backend_work": "native path unchanged; external API ledger required"}}
    output["pool_sha256"] = _fingerprint(output, "pool_sha256")
    return output


def select_hybrid_arms(pool, tags, query_type_scores):
    """H0/H1/H2 share the same pool/hash; H2 only masks relation's weight.

    All ten labels remain mandatory and are retained verbatim. No missing-label
    prior, tagger or new taxonomy is used. This is NOT an SDK-native capture.
    """
    _check(pool, POOL_SCHEMA, "pool_sha256")
    if not 20 <= len(pool["pool_ids"]) <= 80 or len(set(pool["pool_ids"])) != len(pool["pool_ids"]):
        raise ValueError("expected 20..80 unique pool IDs")
    q = _complete_query_scores(query_type_scores)
    values = {fid: _complete_fact_scores(tags.get(fid), fid) for fid in pool["pool_ids"]}
    result = {}
    for arm in ("H0", "H1", "H2"):
        started = time.process_time()
        weights = dict(q)
        if arm == "H2":
            weights["explicit_relation"] = 0.0
        total = sum(weights.values()) if arm != "H0" else 0
        affinities = {fid: sum(w * values[fid][key] for key, w in weights.items()) / total
                      for fid in pool["pool_ids"]} if total else {}
        counts = Counter(affinities.values())
        midranks, next_rank = {}, 1
        for affinity in sorted(counts, reverse=True):
            midranks[affinity] = next_rank + (counts[affinity] - 1) / 2
            next_rank += counts[affinity]
        ranking = [{"id": fid, "fused_rank": r, "type_affinity": affinities.get(fid),
                    "type_rank": midranks.get(affinities.get(fid)),
                    "soft_score": 2 / (60 + r) +
                    (1 / (60 + midranks[affinities[fid]]) if total else 0)}
                   for r, fid in enumerate(pool["pool_ids"], 1)]
        fallback = "no_type_selector" if arm == "H0" else "zero_query_weights" if not total else "equal_type_affinities" if len(counts) <= 1 else None
        chosen = (pool["pool_ids"][:20] if fallback else
                  [row["id"] for row in sorted(ranking, key=lambda row: (-row["soft_score"], row["fused_rank"]))[:20]])
        result[ARMS[arm]] = {"selected_ids": list(chosen), "trace": {
            "schema": SELECT_SCHEMA, "arm": ARMS[arm], "design_arm": arm, "pool_sha256": pool["pool_sha256"],
            "pool_ids": list(pool["pool_ids"]), "original_query_scores": dict(q),
            "effective_query_weights": weights if arm != "H0" else None,
            "routing_mask": {"explicit_relation": 0.0} if arm == "H2" else {},
            "fact_scores": deepcopy(values), "tag_scores_sha256": _digest(values),
            "ranking_in_fused_order": ranking, "fallback_reason": fallback,
            "fused_weight": 2, "type_weight": 1, "rrf_k": 60,
            "selected_ids": list(chosen), "selector_cpu_seconds": time.process_time() - started}}
    return result
