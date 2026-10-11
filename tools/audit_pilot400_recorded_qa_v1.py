"""Audit sealed AS-RUN QA inputs and raw responses; strict retrieval replay remains failed.

No model client, credentials, network, new ranking, answer repair or score filtering.
The pinned diagnostic policy fixes both sensitivity sets before effect inspection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_pilot400_qa_v1 as core
from tools.summarize_pilot400_reviewed_v1 import validate_bank_failure_review
from jev_mem import pilot400, routing
from jev_mem.selection import deduplicate_candidates

driver, evaluation = core.driver, core.evaluation
SCHEMA = "jev-mem-pilot400-recorded-input-raw-response-audit-v1"
POLICY_SHA256 = "390021383045338c6131389794debe80a3f26ada9f3ad374662113a97baa19eb"
CORE_SHA256 = "8f7ff7c5152b18d5c5f250b392c17b6d712ab14f5b668ca84776d1fee18438ac"
DEFAULT_POLICY = ROOT / "research/heartbeat_checks/20260928T0014/recorded_input_reporting_policy_v1.json"


def verify_policy(path, run_dir, manifest):
    raw = driver.shared_read(path)
    if core.sha(raw) != POLICY_SHA256 or driver.file_sha(ROOT / "tools/audit_pilot400_qa_v1.py") != CORE_SHA256:
        raise ValueError("The fixed reporting policy or immutable core audit changed")
    policy = json.loads(raw)
    if (policy["schema"] != "jev-mem-pilot400-recorded-input-reporting-policy-v1"
            or Path(policy["run_dir"]).resolve() != Path(run_dir).resolve()
            or policy["protocol_sha256"] != manifest["protocol_sha256"]
            or policy["strict_retrieval_gate_status"] != "failed_preserved_not_reclassified"):
        raise ValueError("Recorded-input policy identity differs")
    for item in policy["artifacts"].values():
        artifact = (ROOT / item["path"]).resolve()
        if not artifact.is_relative_to(ROOT) or driver.file_sha(artifact) != item["sha256"]:
            raise ValueError("Policy diagnostic/seal artifact changed")
    return policy


def verify_terminal_seal(run_dir, seal):
    run_dir = Path(run_dir).resolve()
    if Path(seal["run_dir"]).resolve() != run_dir:
        raise ValueError("Terminal seal belongs to another run")
    result = {}
    for item in seal["files"]:
        relative = item["path"]
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or relative in result:
            raise ValueError("Invalid/duplicate sealed artifact path")
        raw = driver.shared_read(path)
        if len(raw) != item["bytes"] or core.sha(raw) != item["sha256"]:
            raise ValueError("Original terminal artifact bytes changed: " + relative)
        result[relative] = item["sha256"]
    return result


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Recorded routing value is not finite: " + name)


def validate_recorded_retrieval(run_dir, q, arm, raw, bank, canonical, query_bound, route_bound, *, sealed):
    """Validate the sealed original pool/selection without recomputing its order."""
    folder = driver.conv_dir(run_dir, q["conversation"])
    path = folder / "retrieval" / arm / (q["id"] + ".json")
    relative = path.relative_to(Path(run_dir)).as_posix()
    if sealed.get(relative) != driver.file_sha(path):
        raise ValueError("Recorded retrieval is not the original terminal sealed file")
    saved = driver.require(driver._verified(path, driver.ARTIFACT_SCHEMA, route_bound))
    if saved.get("binding") != route_bound:
        raise ValueError("Recorded retrieval explicit binding differs")
    vector_path = folder / "query_embeddings" / (q["id"] + ".json")
    vector_row = driver.require(driver._verified(vector_path, driver.ARTIFACT_SCHEMA, query_bound))
    if vector_row.get("binding") != query_bound or sealed.get(vector_path.relative_to(Path(run_dir)).as_posix()) != driver.file_sha(vector_path):
        raise ValueError("Query embedding identity/seal differs")
    driver._vectors([vector_row["vector"]], 1)
    needs = None
    if arm == driver.ARMS[1]:
        type_path = folder / "query_types" / (q["id"] + ".json")
        types = driver.require(driver._verified(type_path, driver.ARTIFACT_SCHEMA, query_bound))
        if types.get("binding") != query_bound or sealed.get(type_path.relative_to(Path(run_dir)).as_posix()) != driver.file_sha(type_path):
            raise ValueError("Query type identity/seal differs")
        needs = pilot400._scores(types["scores"])
    weights = routing._type_scores(needs)
    query_hash = routing._hash({"query": q["question"], "type_scores": needs, "version": routing.ROUTING_VERSION})
    channel_names = ["bm25", "dense"] + (["typed"] if sum(weights.values()) else [])
    pool = saved.get("pool")
    if not isinstance(pool, list) or len(pool) != min(60, len(canonical)) or not 20 <= len(pool) <= 60:
        raise ValueError("Recorded pool has an invalid frozen budget")
    ids = [f.get("id") for f in pool]
    if any(not isinstance(fid, str) or fid not in canonical for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Recorded pool contains duplicate or noncanonical IDs")
    for fact in pool:
        original = canonical[fact["id"]]
        content = {k: v for k, v in fact.items() if k not in {"routing", "channel_provenance"}}
        if content != {k: v for k, v in original.items() if k != "channel_provenance"}:
            raise ValueError("Recorded pool differs from complete canonical bank content/source/date/vector")
        route = fact.get("routing")
        fields = {"version", "query_hash", "channels", "channel_ranks", "channel_raw_scores", "rrf_score",
                  "base_rrf_score", "backfill", "typed_enabled", "missing_type_prior", "query_type_scores"}
        if not isinstance(route, dict) or set(route) != fields:
            raise ValueError("Recorded routing metadata schema differs")
        members, ranks, scores = route["channels"], route["channel_ranks"], route["channel_raw_scores"]
        if (route["version"] != routing.ROUTING_VERSION or route["query_hash"] != query_hash
                or type(route["typed_enabled"]) is not bool or route["typed_enabled"] != ("typed" in channel_names)
                or route["missing_type_prior"] != 0.5 or route["query_type_scores"] != weights
                or not isinstance(ranks, dict) or set(ranks) != set(channel_names)
                or not isinstance(scores, dict) or set(scores) != set(channel_names)
                or not isinstance(members, list) or members != [name for name in channel_names if name in members]
                or type(route["backfill"]) is not bool or route["backfill"] != (not members)):
            raise ValueError("Recorded channel/type/query metadata differs")
        for name in channel_names:
            if type(ranks[name]) is not int or not 1 <= ranks[name] <= len(canonical):
                raise ValueError("Recorded channel rank is invalid")
            _finite(scores[name], name)
        if scores["bm25"] < 0 or not -1 <= scores["dense"] <= 1 or "typed" in scores and not 0 <= scores["typed"] <= 1:
            raise ValueError("Recorded raw score range is invalid")
        if members != [name for name in channel_names if ranks[name] <= 20]:
            raise ValueError("Recorded channel membership disagrees with recorded ranks")
        expected_rrf = sum(1 / (routing.RRF_K + ranks[name]) for name in channel_names if name in members)
        expected_base = sum(1 / (routing.RRF_K + ranks[name]) for name in ["bm25", "dense"])
        if route["rrf_score"] != expected_rrf or route["base_rrf_score"] != expected_base:
            raise ValueError("Recorded RRF is inconsistent with recorded channel ranks")
        provenance = list(original.get("channel_provenance", []))
        if members:
            provenance += [{"channel": name, "rank": ranks[name], "query_hash": query_hash} for name in members]
            if fact.get("channel_provenance") != provenance:
                raise ValueError("Recorded channel provenance differs")
        else:
            recorded = fact.get("channel_provenance")
            if (not isinstance(recorded, list) or len(recorded) != len(provenance) + 1 or recorded[:-1] != provenance
                    or set(recorded[-1]) != {"channel", "rank", "query_hash"}
                    or recorded[-1]["channel"] != "generic_backfill" or recorded[-1]["query_hash"] != query_hash
                    or type(recorded[-1]["rank"]) is not int or not 1 <= recorded[-1]["rank"] <= len(canonical)):
                raise ValueError("Recorded backfill provenance differs")
    selected = [pilot400.reader_fact(f) for f in pool[:20]]
    trace = {"version": pilot400.VERSION, "selector": "rank", "candidate_count": len(pool),
             "selected_ids": [f["id"] for f in selected], "typed": needs is not None,
             "model_calls_made_by_retrieval": 0, "max_pool": 60, "top_per_channel": 20, "final_fact_count": 20}
    if saved.get("selected") != selected or saved.get("trace") != trace:
        raise ValueError("Recorded selection/order/trace differs from its exact pool first twenty")
    return saved, {"path": relative, "sha256": driver.file_sha(path), "pool_count": len(pool),
                   "selected_count": len(selected), "verification": "Sealed as-run content and internal metadata; no deterministic rank recomputation"}


def replay_recorded(run_dir, row, q, raw, saved, proof):
    selected = saved["selected"]
    if (row.get("selected_ids") != [f["id"] for f in selected] or row.get("retrieval_path") != proof["path"]
            or row.get("retrieval_sha256") != proof["sha256"]):
        raise ValueError("QA references a different recorded retrieval/order")
    cache_api = core.CacheOnlyAPI(Path(run_dir) / "api_cache")
    result = evaluation.answer_and_judge(cache_api, q["question"], q["reference"], selected,
                                         reference_date=max(t["observation_date"] for t in raw))
    if result != row.get("result"):
        raise ValueError("QA differs from its recorded-order exact reader/judge HTTP replay")
    if [p["stage"] for p in cache_api.proofs] != [evaluation.READER_STAGE, evaluation.JUDGE_STAGE]:
        raise ValueError("Missing logical reader/judge raw replay")
    return cache_api.proofs


def audit_recorded(run_dir, review_path, policy_path=DEFAULT_POLICY):
    run_dir = Path(run_dir).resolve()
    manifest = driver.ensure_manifest(run_dir, create=False)
    policy = verify_policy(policy_path, run_dir, manifest)
    seal = driver.read(ROOT / policy["artifacts"]["qa_terminal/seal.json"]["path"])
    sealed = verify_terminal_seal(run_dir, seal)
    admission = validate_bank_failure_review(run_dir, review_path, require_failed_qa=True)
    failed_banks = {(x["conversation"], x["bank"]) for x in admission["failed_banks"]}
    data = driver.load_data(); _, questions = driver.load_cohort(driver.COHORT_PATH, data)
    if len(questions) != 400 or len({q["id"] for q in questions}) != 400:
        raise ValueError("Frozen cohort is not exactly 400 unique questions")
    ledger_raw = driver.shared_read(run_dir / "api_calls.jsonl")
    events, by_key = core.ledger_index(ledger_raw, {(q["conversation"], arm, q["id"]) for q in questions for arm in driver.ARMS})
    reports, conversations, seen_qa, seen_retrieval, immutable = [], [], set(), set(), {}
    for ci in range(10):
        raw = driver.raw_turns_from_locomo(data[ci]["conversation"])
        banks = {name: None if (ci, name) in failed_banks else driver.load_bank(run_dir, ci, name, raw, manifest)
                 for name in ("session", "regroup")}
        canonical = {name: {f["id"]: f for f in deduplicate_candidates(bank["facts"])} if bank else {}
                     for name, bank in banks.items()}
        for bank in banks.values():
            if bank: immutable[bank["path"]] = bank["sha256"]
        tags, tags_sha = driver.load_tags(run_dir, ci, raw, manifest, banks["session"]) if banks["session"] else (None, None)
        if tags_sha: immutable[f"conversations/conv_{ci}/tag_summary.json"] = tags_sha
        qs = [q for q in questions if q["conversation"] == ci]; artifacts = []
        for q in qs:
            for arm in driver.ARMS:
                path = driver.conv_dir(run_dir, ci) / "qa" / arm / (q["id"] + ".json")
                relative = path.relative_to(run_dir).as_posix(); seen_qa.add(relative)
                if sealed.get(relative) != driver.file_sha(path): raise ValueError("QA not in original terminal seal")
                name = "regroup" if arm == driver.ARMS[2] else "session"; bank = banks[name]
                row, qb, rb = core.verified_qa(path, q, arm, raw, manifest, bank, tags_sha); artifacts.append(row)
                matched = by_key.get((ci, arm, q["id"]), [])
                item = {"conversation": ci, "query_id": q["id"], "arm": arm, "status": row["status"],
                        "artifact_sha256": driver.file_sha(path), "error": row.get("error"), "ledger_lines": [e["line"] for e in matched]}
                retrieval_path = driver.conv_dir(run_dir, ci) / "retrieval" / arm / (q["id"] + ".json")
                saved = proof = None
                if retrieval_path.exists():
                    if bank is None: raise ValueError("Failed bank has a recorded retrieval")
                    saved, proof = validate_recorded_retrieval(run_dir, q, arm, raw, bank, canonical[name], qb, rb, sealed=sealed)
                    seen_retrieval.add(proof["path"]); item["recorded_retrieval"] = proof
                if row["status"] == "complete":
                    if saved is None: raise ValueError("Complete QA has no recorded retrieval")
                    proofs = replay_recorded(run_dir, row, q, raw, saved, proof)
                    core.validate_complete_events(proofs, matched); item["raw_response_replays"] = proofs
                    for p in proofs: immutable["api_cache/" + p["request_hash"] + ".json"] = p["cache_sha256"]
                elif bank is None and matched:
                    raise ValueError("Failed bank has reader/judge events")
                reports.append(item)
        counts = core.validate_terminal_status(driver.conv_dir(run_dir, ci) / "qa_status.json", ci, qs, manifest, artifacts)
        conversations.append({"conversation": ci, "methods": counts})
    actual_qa = {p.relative_to(run_dir).as_posix() for p in run_dir.glob("conversations/conv_*/qa/*/*.json")}
    actual_retrieval = {p.relative_to(run_dir).as_posix() for p in run_dir.glob("conversations/conv_*/retrieval/*/*.json")}
    if len(reports) != 1200 or actual_qa != seen_qa or actual_retrieval != seen_retrieval:
        raise ValueError("Expected exactly 1200 QA and every existing recorded retrieval")
    verify_policy(policy_path, run_dir, manifest); verify_terminal_seal(run_dir, seal)
    for path, digest in immutable.items():
        if driver.file_sha(run_dir / path) != digest: raise ValueError("Bank/tag/HTTP cache changed during recorded-input audit")
    return {"schema": SCHEMA, "snapshot_utc": driver.now(), "run_dir": str(run_dir),
        "recorded_input_raw_response_valid": True, "strict_deterministic_retrieval_audit_passed": False,
        "reporting_policy_sha256": POLICY_SHA256, "policy": policy, "review_sha256": admission["review_sha256"],
        "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": driver.file_sha(run_dir / "run_manifest.json"),
        "terminal_artifacts": len(reports), "recorded_retrievals_checked": len(seen_retrieval),
        "status_counts": dict(Counter(r["status"] for r in reports)), "conversations": conversations, "artifacts": reports,
        "ledger_sha256": core.sha(ledger_raw), "reader_judge_events": events,
        "scope": "Sealed as-run pool/fact/source/date/vector/order and raw reader/judge replay. No ranking recomputation, no answers repaired, no API calls. Strict cross-process retrieval reproduction remains failed. Unknown usage/cost remains separate; semantic support and classifier accuracy are not certified."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--review", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv); run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output == run or run in output.parents or output in run.parents: raise ValueError("Output overlaps run")
    report = audit_recorded(run, args.review, args.policy)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "audit.json"), "recorded_input_raw_response_valid": True,
                      "strict_deterministic_retrieval_audit_passed": False, "status_counts": report["status_counts"]}))
    return 0


if __name__ == "__main__": main()
