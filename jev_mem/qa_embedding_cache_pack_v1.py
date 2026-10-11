"""Exact, sealed legacy retrieval-query embeddings only; no model dispatch.

The SDK's query normalization and old retrieval ledger identify full-question and
entity-query requests. Missing/invalid first sources stop admission; they never
become a fresh request or a reason to try a better response from another root.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from jev_mem.data import load_data, queries
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest as request_digest
from tools.run_sdk20_modules import ROOT, digest, manifest as sdk_manifest, save_new, sha, verify

SCHEMA = "sealed-query-embedding-cache-pack-v1"
SEAL = ROOT / "research/main_batch_firstpass_seal_v1/seal.json"
SEAL_SHA = "c6bae930c7d2bde85d5ae273ca816d8d96bc2692fbdfdd702b52a5e489e0a146"
BASE = ROOT / "runs/session_top20_v1"
SDK = ROOT / "runs/sdk20_modules_v1"
MODEL = "gemini-embedding-001"
OPERATION = "batchEmbedContents"
PRECEDENCE = ["session_top20_v1/conv_{ci}/api_cache", "sdk20_modules_v1/conv_{ci}/api/api_cache"]


def pointer(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked_pointer(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or pointer(value["path"]) != value:
        raise ValueError("Embedding cache artifact pointer changed")
    return Path(value["path"])


class _ReplayEmbed:
    embed = ModelAPI.embed

    def __init__(self, response=None):
        self.response = response if response is not None else {"embeddings": [{"values": [1.0] + [0.0] * 767}]}
        self.signatures = []

    def request(self, provider, model, body, *, operation, stage):
        self.signatures.append({"provider": provider, "model": model, "operation": operation, "body": body})
        return self.response

    def count_tokens(self, *args, **kwargs):
        raise ValueError("Long query requires a separately reviewed countTokens cache; no dispatch")


def validate_raw(raw, expected_hash):
    """Replay the original vector parser; never rewrite a response's bytes."""
    saved = json.loads(raw)
    signature = saved.get("request")
    if (not isinstance(signature, dict) or set(signature) != {"provider", "model", "operation", "body"}
            or signature["provider"] != "gemini" or signature["model"] != MODEL
            or signature["operation"] != OPERATION or request_digest(signature) != expected_hash):
        raise ValueError("Only exact Gemini001 query embedding requests are admitted")
    body = signature["body"]
    if not isinstance(body, dict) or set(body) != {"requests"}:
        raise ValueError("Embedding request body differs")
    requests = body["requests"]
    if not isinstance(requests, list) or not 1 <= len(requests) <= 32:
        raise ValueError("Embedding batch size differs")
    texts = []
    for item in requests:
        if (not isinstance(item, dict) or set(item) != {"model", "content", "taskType", "outputDimensionality"}
                or item["model"] != "models/" + MODEL or item["taskType"] != "RETRIEVAL_QUERY"
                or item["outputDimensionality"] != 768 or not isinstance(item["content"], dict)
                or set(item["content"]) != {"parts"}):
            raise ValueError("Only RETRIEVAL_QUERY768 embeddings are admitted")
        parts = item["content"]["parts"]
        if (not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict)
                or set(parts[0]) != {"text"} or not isinstance(parts[0]["text"], str)):
            raise ValueError("Embedding text shape differs")
        texts.append(parts[0]["text"])
    replay = _ReplayEmbed(saved["response"])
    vectors = replay.embed(texts, query=True, stage="mem0_embedding_search")
    if replay.signatures != [signature] or len(vectors) != len(texts):
        raise ValueError("Embedding request does not replay the original provider")
    return signature


def _first_source(ci, key, inventory):
    roots = [BASE / f"conv_{ci}/api_cache", SDK / f"conv_{ci}/api/api_cache"]
    chosen = next((root / (key + ".json") for root in roots if (root / (key + ".json")).exists()), None)
    if chosen is None:
        raise ValueError(f"No completed frozen query embedding source for conversation{ci}/{key}; unknown/failed requires review")
    raw = chosen.read_bytes()
    signature = validate_raw(raw, key)  # Failure deliberately never falls through.
    expected = inventory.get(f"conv_{ci}/api_cache/{key}.json")
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("First embedding response is not the original sealed bytes; separate review required")
    return {"conversation": ci, "request_hash": key, "signature": signature,
            "source": pointer(chosen), "sealed_original_relative_path": f"conv_{ci}/api_cache/{key}.json",
            "sealed_original_sha256": expected, "source_rank": roots.index(chosen.parent)}


def _validate_plan(plan):
    if plan.get("schema") != SCHEMA + "-plan" or plan.get("source_precedence") != PRECEDENCE:
        raise ValueError("Unsupported embedding admission plan")
    entries, queries_ = plan.get("entries", []), plan.get("full_queries", [])
    keys = {(r["conversation"], r["request_hash"]) for r in entries}
    if (len(keys) != len(entries) or {ci for ci, _ in keys} != set(range(10))
            or any(type(ci) is not int or ci not in range(10) for ci, _ in keys)
            or len(queries_) != 1540 or len({r["query_id"] for r in queries_}) != 1540
            or any((r["conversation"], r["request_hash"]) not in keys for r in queries_)):
        raise ValueError("Embedding plan request/query coverage is incomplete or duplicated")
    counts = {"questions": len(queries_), "query_or_entity_requests": len(entries),
              "per_conversation": dict(Counter(str(x["conversation"]) for x in entries))}
    if plan.get("counts") != counts or plan.get("model_calls") != 0 or plan.get("native_clients") != 0:
        raise ValueError("Embedding plan counts or no-dispatch declaration differs")
    for entry in entries:
        ci, key, rank = entry["conversation"], entry["request_hash"], entry["source_rank"]
        if (not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                or type(rank) is not int or rank not in (0, 1)):
            raise ValueError("Invalid embedding source rank or request hash")
        roots = [BASE / f"conv_{ci}/api_cache", SDK / f"conv_{ci}/api/api_cache"]
        if (Path(entry["source"]["path"]).resolve() != (roots[rank] / (key + ".json")).resolve()
                or entry["source"]["sha256"] != entry["sealed_original_sha256"]
                or entry["sealed_original_relative_path"] != f"conv_{ci}/api_cache/{key}.json"
                or request_digest(entry["signature"]) != key):
            raise ValueError("Embedding plan source path/signature/seal lineage differs")


def audit_sources():
    """Read-only plan, including exact full-query coverage and entity embeddings."""
    sdk = sdk_manifest(SDK)
    if sha(SEAL) != SEAL_SHA:
        raise ValueError("Original cache seal changed")
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    entry = next(row for row in seal["runs"] if row["architecture"] == "mem0")
    inventory_path = SEAL.parent / entry["inventory_path"]
    raw_inventory = inventory_path.read_bytes()
    if hashlib.sha256(raw_inventory).hexdigest() != entry["inventory_sha256"]:
        raise ValueError("Original cache inventory changed")
    inventory = {row["path"]: row["sha256"] for row in map(json.loads, raw_inventory.splitlines())}
    # Execute only this pinned, pure SDK validation function; do not import/open
    # a native Memory/Qdrant store just to normalize an embedding-cache key.
    sdk_main = next(Path(p) for p in sdk["sdk_sha256"] if Path(p).as_posix().endswith("mem0/memory/main.py"))
    tree = ast.parse(sdk_main.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_and_trim_search_query")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(sdk_main), "exec"), namespace)
    normalize = namespace[function.name]
    entries, full_queries, ledgers = [], [], []
    for ci, item in enumerate(load_data()):
        qs = list(queries(item, ci))
        allowed = {q["id"] for q in qs}
        ledger = SDK / f"conv_{ci}/api/api_calls.jsonl"
        ledger_bytes = ledger.read_bytes()
        seen = {}
        for event in map(json.loads, ledger_bytes.splitlines()):
            context = event.get("context", {})
            if context.get("phase") != "retrieval" or event.get("operation") != OPERATION:
                continue
            if (event.get("provider") != "gemini" or event.get("model") != MODEL
                    or event.get("stage") != "mem0_embedding_search"
                    or context.get("conversation") != ci or context.get("method") not in ("session", "preseg")
                    or context.get("query_id") not in allowed):
                raise ValueError("Legacy retrieval embedding context differs")
            key = event["request_hash"]
            seen.setdefault(key, set()).add(context["query_id"])
        for q in qs:
            probe = _ReplayEmbed()
            probe.embed([normalize(q["question"])], query=True, stage="mem0_embedding_search")
            key = request_digest(probe.signatures[0])
            if q["id"] not in seen.get(key, set()):
                raise ValueError("Actual normalized SDK query is absent from its completed retrieval ledger")
            full_queries.append({"conversation": ci, "query_id": q["id"], "request_hash": key,
                                 "sdk_trimmed": normalize(q["question"]) != q["question"]})
        for key in sorted(seen):
            source = _first_source(ci, key, inventory)
            entries.append({**source, "query_ids": sorted(seen[key])})
        if ledger.read_bytes() != ledger_bytes:
            raise ValueError("Old retrieval ledger changed during admission")
        ledgers.append(pointer(ledger))
    if len(full_queries) != 1540 or len({r["query_id"] for r in full_queries}) != 1540:
        raise ValueError("Expected exactly1540 fixed original questions")
    return {"schema": SCHEMA + "-plan", "source_precedence": PRECEDENCE,
            "original_seal": pointer(SEAL), "original_inventory": pointer(inventory_path),
            "sdk_manifest": pointer(SDK / "run_manifest.json"), "sdk_search_code": pointer(sdk_main),
            "retrieval_ledgers": ledgers, "entries": entries, "full_queries": full_queries,
            "counts": {"questions": len(full_queries), "query_or_entity_requests": len(entries),
                       "per_conversation": dict(Counter(str(x["conversation"]) for x in entries))},
            "model_calls": 0, "native_clients": 0,
            "policy": "first existing source, sealed bytes, QUERY768 only, no missing/invalid logical request redispatch"}


def create_pack(run_root, plan):
    """Copy a reviewed plan into a fresh embedding-only pack, preserving bytes."""
    run = Path(run_root).resolve()
    dest = run / "embedding_cache_pack"
    _validate_plan(plan)
    if dest.exists():
        raise FileExistsError("Embedding pack requires a new namespace")
    for root in (BASE, SDK):
        if run == root.resolve() or root.resolve() in run.parents or run in root.resolve().parents:
            raise ValueError("Embedding pack must not modify original source runs")
    for binding in [plan["original_seal"], plan["original_inventory"], plan["sdk_manifest"],
                    plan["sdk_search_code"], *plan["retrieval_ledgers"]]:
        checked_pointer(binding)
    entries = []
    for entry in plan["entries"]:
        source = checked_pointer(entry["source"])
        roots = [BASE / f"conv_{entry['conversation']}/api_cache", SDK / f"conv_{entry['conversation']}/api/api_cache"]
        first = next((r / (entry["request_hash"] + ".json") for r in roots
                      if (r / (entry["request_hash"] + ".json")).exists()), None)
        if first is None or source.resolve() != first.resolve():
            raise ValueError("Fixed first-source precedence changed after audit")
        raw = source.read_bytes()
        if validate_raw(raw, entry["request_hash"]) != entry["signature"]:
            raise ValueError("Embedding signature changed before copy")
        target = dest / f"conv_{entry['conversation']}/api_cache/{entry['request_hash']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
        if sha(target) != entry["source"]["sha256"] or sha(source) != entry["source"]["sha256"]:
            raise ValueError("Embedding raw changed during copy")
        entries.append({**entry, "target": pointer(target)})
    value = {**plan, "schema": SCHEMA, "entries": entries, "plan_sha256": digest(plan),
             "pack_root": str(dest), "helper": pointer(__file__), "raw_bytes_rewritten": False}
    save_new(dest / "manifest.json", value)
    return {"manifest": pointer(dest / "manifest.json"),
            "cache_roots": {str(ci): [str(dest / f"conv_{ci}/api_cache")] for ci in range(10)}}


def verify_pack(binding, conversation=None, check_sources=False):
    """Validate a frozen pack; a worker may check just its own conversation."""
    path = checked_pointer(binding["manifest"])
    value = verify(path)
    dest = path.parent
    if (value.get("schema") != SCHEMA or value.get("source_precedence") != PRECEDENCE
            or value.get("pack_root") != str(dest) or value.get("raw_bytes_rewritten") is not False):
        raise ValueError("Embedding pack metadata differs")
    checked_pointer(value["helper"])
    plan = {k: v for k, v in value.items() if k not in
            ("content_sha256", "plan_sha256", "pack_root", "helper", "raw_bytes_rewritten")}
    plan["schema"] = SCHEMA + "-plan"
    plan["entries"] = [{k: v for k, v in row.items() if k != "target"} for row in value["entries"]]
    _validate_plan(plan)
    if digest(plan) != value.get("plan_sha256"):
        raise ValueError("Embedding pack differs from its source admission plan")
    expected_roots = {str(ci): [str(dest / f"conv_{ci}/api_cache")] for ci in range(10)}
    if binding.get("cache_roots") != expected_roots:
        raise ValueError("Embedding cache roots differ from the isolated pack")
    if conversation is not None and (type(conversation) is not int or conversation not in range(10)):
        raise ValueError("Invalid embedding pack conversation")
    chosen = [r for r in value["entries"] if conversation is None or r["conversation"] == conversation]
    expected_paths = set()
    for entry in chosen:
        target = checked_pointer(entry["target"])
        expected = dest / f"conv_{entry['conversation']}/api_cache/{entry['request_hash']}.json"
        if target != expected or entry["target"]["sha256"] != entry["source"]["sha256"]:
            raise ValueError("Embedding pack target or raw-source lineage differs")
        if validate_raw(target.read_bytes(), entry["request_hash"]) != entry["signature"]:
            raise ValueError("Embedding pack raw request changed")
        expected_paths.add(target)
        if check_sources:
            checked_pointer(entry["source"])
    observed = {p.resolve() for ci in (range(10) if conversation is None else [conversation])
                for p in (dest / f"conv_{ci}/api_cache").glob("*.json")}
    if observed != expected_paths or len(expected_paths) != len(chosen):
        raise ValueError("Unexpected/missing/duplicate file in embedding-only cache")
    if check_sources:
        for pointer_ in [value["original_seal"], value["original_inventory"], value["sdk_manifest"],
                         value["sdk_search_code"], *value["retrieval_ledgers"]]:
            checked_pointer(pointer_)
    return value
