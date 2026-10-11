"""B's only retrieval change: fixed type soft-ranking of SDK20's own pool.

Run each conversation in a separate process. Capture wraps the module binding
used by native ``Memory.search(top_k=20, rerank=False)`` and returns its original
20 results unchanged. It neither searches again nor rewrites the SDK score
formula: a second, local call to the original scorer exposes all eligible rows
from the same at-most-80 semantic candidates, with the same semantic threshold.
Reader/Judge, facts, payloads, native scalar scores and the SDK source are untouched.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
import threading
from typing import Any, Mapping

from .routing import RRF_K, TYPE_KEYS, _flatten_tags, _type_scores

CAPTURE_SCHEMA = "sdk20-native-pool-v1"
SELECTION_SCHEMA = "sdk20-type-soft-selection-v1"
TOP_K = 20
CANDIDATE_CAP = 80
NATIVE_WEIGHT = 2
TYPE_WEIGHT = 1
_ACTIVE_CAPTURE = False


def _digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _ids(rows: list[dict]) -> list[str]:
    ids = [row["id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("native candidate/result IDs must be unique nonempty strings")
    return ids


@contextmanager
def capture_native_pool(*, _scorer_module: Any = None):
    """Yield a JSON-ready capture populated by exactly one native SDK20 search.

    Use ``with capture_native_pool() as captured: result = memory.search(...)``.
    ``result`` is unchanged; call ``typed_select(captured, tags, query_scores)``
    after exiting. The private module argument supports network-free tests.
    A process-global patch is intentionally restricted to one thread, one search
    and one non-nested context; the driver must isolate conversations by process.
    """
    global _ACTIVE_CAPTURE
    if _ACTIVE_CAPTURE:
        raise RuntimeError("native pool capture cannot be nested or concurrent")
    _ACTIVE_CAPTURE = True
    module = None
    original = None
    captured: dict[str, Any] = {}
    owner = (os.getpid(), threading.get_ident())
    called = False
    try:
        module = _scorer_module if _scorer_module is not None else importlib.import_module("mem0.memory.main")
        original = module.score_and_rank

        def hook(semantic_results, bm25_scores, entity_boosts, threshold, top_k, explain=False):
            nonlocal called
            if (os.getpid(), threading.get_ident()) != owner:
                raise RuntimeError("native pool capture requires a dedicated process and its owning thread")
            if called:
                raise RuntimeError("native pool capture permits exactly one search")
            called = True
            if top_k != TOP_K or isinstance(top_k, bool):
                raise ValueError("capture requires the original search top_k=20")
            if len(semantic_results) > CANDIDATE_CAP:
                raise ValueError("SDK20 semantic candidate pool exceeds 80")
            inputs = deepcopy({
                "semantic_results": semantic_results,
                "bm25_scores": bm25_scores,
                "entity_boosts": entity_boosts,
                "threshold": threshold,
                "top_k": top_k,
                "explain": explain,
            })
            candidate_ids = _ids(inputs["semantic_results"])
            # Return this exact original result object to native formatting.
            base20 = original(semantic_results, bm25_scores, entity_boosts, threshold, top_k, explain)
            all_inputs = deepcopy(inputs)
            all_inputs["top_k"] = len(semantic_results)
            eligible = original(**all_inputs)
            eligible_ids = _ids(eligible)
            if not set(eligible_ids).issubset(candidate_ids):
                raise ValueError("original scorer returned an ID outside the captured semantic pool")
            if base20 != eligible[:TOP_K]:
                raise ValueError("original scorer's complete ranking does not reproduce its native20")
            captured.update({
                "schema": CAPTURE_SCHEMA,
                "candidate_cap": CANDIDATE_CAP,
                "inputs": inputs,
                "candidate_ids": candidate_ids,
                "base20": deepcopy(base20),
                "base20_ids": _ids(base20),
                "eligible": deepcopy(eligible),
                "eligible_ids": eligible_ids,
                "native_scores": {row["id"]: row["score"] for row in eligible},
            })
            captured["capture_sha256"] = _digest(captured)
            return base20

        module.score_and_rank = hook
        yield captured
        if not captured:
            raise RuntimeError("native pool capture completed without a successful SDK20 search")
    finally:
        if module is not None and original is not None:
            module.score_and_rank = original
        _ACTIVE_CAPTURE = False


def _complete_query_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    values = _type_scores(scores)
    if set(values) != TYPE_KEYS:
        raise ValueError("query type scores require all 10 finite labels in [0, 1]")
    return values


def _complete_fact_scores(tag: Mapping[str, Any] | None, fact_id: str) -> dict[str, float]:
    try:
        values = _flatten_tags(tag)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"invalid type label record for {fact_id}") from exc
    if set(values) != TYPE_KEYS:
        raise ValueError(f"fact {fact_id} requires all 10 finite labels in [0, 1]")
    return values


def typed_select(captured: Mapping[str, Any], tags: Mapping[str, Mapping[str, Any]],
                 query_type_scores: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``selected_ids`` and an auditable trace; never mutate input records.

    Every eligible fact and the query require all ten valid type scores. No
    missing-label prior, origin-ID fallback, additional retrieval or deduplication
    is used. Fixed fusion is 2/(60+native_rank) + 1/(60+type_midrank).
    Zero query weights or equal affinities return the exact native20 ID order.
    """
    if captured.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unsupported native-pool capture schema")
    unhashed = {key: value for key, value in captured.items() if key != "capture_sha256"}
    if _digest(unhashed) != captured.get("capture_sha256"):
        raise ValueError("native-pool capture fingerprint mismatch")
    if not isinstance(tags, Mapping):
        raise ValueError("tags must map native fact IDs to complete type records")
    weights = _complete_query_scores(query_type_scores)
    eligible = captured["eligible"]
    native_ids = captured["eligible_ids"]
    values = {fact_id: _complete_fact_scores(tags.get(fact_id), fact_id) for fact_id in native_ids}
    total_weight = sum(weights.values())
    affinities = {
        fact_id: sum(weight * values[fact_id][key] for key, weight in weights.items()) / total_weight
        for fact_id in native_ids
    } if total_weight else {}
    counts = Counter(affinities.values())
    midranks = {}
    next_rank = 1
    for affinity in sorted(counts, reverse=True):
        midranks[affinity] = next_rank + (counts[affinity] - 1) / 2
        next_rank += counts[affinity]
    ranking = []
    for rank, row in enumerate(eligible, 1):
        affinity = affinities.get(row["id"])
        type_rank = midranks[affinity] if affinity is not None else None
        ranking.append({
            "id": row["id"], "native_rank": rank, "native_score": row["score"],
            "type_affinity": affinity, "type_rank": type_rank,
            "soft_score": NATIVE_WEIGHT / (RRF_K + rank)
            + (TYPE_WEIGHT / (RRF_K + type_rank) if type_rank is not None else 0.0),
        })
    reason = "zero_query_weights" if not total_weight else "equal_type_affinities" if len(counts) <= 1 else None
    if reason is not None:
        selected_ids = list(captured["base20_ids"])
    else:
        selected_ids = [row["id"] for row in sorted(ranking, key=lambda row: (-row["soft_score"], row["native_rank"]))[:TOP_K]]
    baseline = captured["base20_ids"]
    return {"selected_ids": selected_ids, "trace": {
        "schema": SELECTION_SCHEMA,
        "retrieval_change": "B_only_fixed_type_ranking_inside_original_sdk20_semantic_pool",
        "capture_sha256": captured["capture_sha256"],
        "top_k": TOP_K, "candidate_cap": CANDIDATE_CAP,
        "candidate_count": len(captured["candidate_ids"]), "eligible_count": len(eligible),
        "threshold": captured["inputs"]["threshold"],
        "native_weight": NATIVE_WEIGHT, "type_weight": TYPE_WEIGHT, "rrf_k": RRF_K,
        "query_type_scores": weights, "fallback_reason": reason,
        "base20_ids": list(baseline), "selected_ids": list(selected_ids),
        "entered_ids": [fact_id for fact_id in selected_ids if fact_id not in baseline],
        "dropped_ids": [fact_id for fact_id in baseline if fact_id not in selected_ids],
        "ranking_in_native_order": ranking,
    }}
