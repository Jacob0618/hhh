"""Pure offline provenance audit for the two new recovered banks.

Reconstructs mechanical quotations, model admission decisions, quarantines,
explicit empty groups and source fallbacks. Model support assessments remain
model judgments, not semantic ground truth. No credentials or API construction.
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pilot400_members_only_writer_v2 as writer
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now
from jev_mem.data import SOURCE_HASH, load_data
from jev_mem.pilot400 import fingerprint
from jev_mem.topic_grouping import raw_turns_from_locomo, verify_partition
w = writer.w


def checked_cache(run, signature, stage, *, require_intent=False):
    digest = w.provider_request_hash(signature)
    path = Path(run) / "api_cache" / (digest + ".json")
    cached = read(path)
    if cached.get("request") != signature or cached.get("stage") != stage:
        raise ValueError("HTTP cache request/stage differs: " + digest)
    if require_intent:
        intent = read(Path(run) / "writer_request_intents" / (digest + ".json"))
        if intent.get("request") != signature or intent.get("request_hash") != digest or intent.get("stage") != stage:
            raise ValueError("Writer durable intent binding differs")
    return cached["response"], {"request_hash": digest, "cache_path": str(path), "cache_sha256": file_sha(path)}


def verify_embeddings(facts, journal, cache_run, *, pending=None):
    texts = [f["text"] for f in facts]
    signatures = [{"provider": "gemini", "model": "gemini-embedding-001", "operation": "batchEmbedContents",
        "body": {"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT", "outputDimensionality": 768} for text in texts[start:start + 32]]}}
        for start in range(0, len(texts), 32)]
    expected = {"stage": "group_fact_embedding", "texts_sha256": fingerprint(texts), "requests": signatures,
        "request_hashes": [w.provider_request_hash(s) for s in signatures], "max_logical_calls": 1}
    if any(journal.get(k) != v for k, v in expected.items()):
        raise ValueError("Embedding journal differs from admitted exact fact texts")
    single, double = [], []
    for sig in signatures:
        raw, _ = checked_cache(cache_run, sig, "group_fact_embedding")
        rows = raw.get("embeddings", [])
        if len(rows) != len(sig["body"]["requests"]):
            raise ValueError("Embedding HTTP batch size differs")
        for row in rows:
            values = row["values"]
            norm = math.sqrt(sum(x * x for x in values))
            normalized = [x / norm for x in values]
            single.append(w._vector(values))
            double.append(w._vector(normalized))
    actual = [f["embedding"] for f in facts]
    if journal.get("status") == "completed_from_all_batch_caches":
        base = single
    elif journal.get("status") == "completed":
        base = double
    else:
        raise ValueError("Embedding transaction has not completed")
    resumed = [w._vector(v) for v in base]
    if actual not in (base, resumed):
        raise ValueError("Stored embeddings differ from exact HTTP normalization chain")
    if pending is not None and pending.get("embeddings") != base:
        raise ValueError("Embedding checkpoint differs from HTTP normalization chain")
    return {"http_batches": len(signatures), "facts": len(facts), "journal_status": journal["status"],
            "extra_checkpoint_normalization": actual != base}


def audit_new_group(ci, group, raw, record, run):
    request = {"members": group["turns"], "read_only_context": [],
        "member_context_ids": {t["source_turn_id"]: [] for t in group["turns"]},
        "member_source_turn_ids": [t["source_turn_id"] for t in group["turns"]], "context_source_turn_ids": [],
        "user_id": "pilot400_conv_" + str(ci), "group_id": group["group_id"], "grouping_method": "jev_topics"}
    if (record.get("schema") != writer.VERSION or record.get("conversation") != ci
            or record.get("members") != group["turns"] or record.get("source_sha256") != fingerprint(raw)
            or record.get("group_id") != group["group_id"] or record.get("source_only") is not True
            or record.get("qa_visible") is not False):
        raise ValueError("New group identity/raw-source binding differs")
    sig = writer.extraction_request(ci, group)
    if record["extraction_signature"] != sig:
        raise ValueError("Extraction signature differs from source-only request")
    events, candidates, quarantine, accepted, verdicts = [], [], [], [], {}
    decoded, failure = None, None
    try:
        response, event = checked_cache(run, sig, "remaining71_members_only_extract", require_intent=True)
        decoded = w.decode_http_response(response)
        events.append(event)
        if not isinstance(decoded, dict) or set(decoded) != {"facts"} or not isinstance(decoded["facts"], list):
            raise ValueError("Invalid extraction envelope")
        for index, item in enumerate(decoded["facts"]):
            try:
                fact = writer.parse_quoted(item, request, w.provider_request_hash(sig), index)
                candidates.append({"candidate_index": index, "fact": fact, "proposed": item})
            except (ValueError, w.GroupWriteError, TypeError, KeyError) as error:
                quarantine.append({"candidate_index": index, "reason": str(error), "proposed": item, "stage": "mechanical"})
        if candidates:
            review_sig = writer.validator_request(group["turns"], [{"candidate_index": c["candidate_index"],
                "proposed": c["proposed"]} for c in candidates])
            response, event = checked_cache(run, review_sig, "remaining71_member_support_check", require_intent=True)
            value = w.decode_http_response(response)
            events.append(event)
            verdicts = writer.checked_assessments(value, [c["candidate_index"] for c in candidates])
            for c in candidates:
                verdict = verdicts[c["candidate_index"]]
                if all(verdict[k] for k in ("supported", "attribution_ok", "temporal_ok")):
                    accepted.append(c["fact"])
                else:
                    quarantine.append({"candidate_index": c["candidate_index"], "stage": "model_support",
                        "proposed": c["proposed"], "reason": verdict["reason"], "assessment": verdict})
    except FileNotFoundError:
        # No raw response cannot be scientifically reconstructed by this audit.
        raise ValueError("Missing raw HTTP evidence requires separate unknown-attempt review") from None
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    mode = "generated_facts" if accepted else ("model_empty" if decoded == {"facts": []} and failure is None else "raw_member_excerpt_fallback")
    if mode == "raw_member_excerpt_fallback":
        accepted = writer.excerpt_fallback(request)
    expected = {"raw_response": decoded, "events": events, "candidates": candidates,
                "assessments": {str(k): v for k, v in verdicts.items()}, "quarantined": quarantine,
                "failure": failure, "mode": mode}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("New group reconstructed " + key + " differs: " + group["group_id"])
    if [{k: v for k, v in f.items() if k != "embedding"} for f in record["facts"]] != accepted:
        raise ValueError("Committed facts differ from reconstructed admission/fallback")
    pending_path = Path(record["embedding_path"])
    if not pending_path.resolve().is_relative_to(Path(run).resolve()) or file_sha(pending_path) != record["embedding_sha256"]:
        raise ValueError("Embedding checkpoint path/hash differs")
    pending = read(pending_path)
    embedding = verify_embeddings(record["facts"], pending["embedding_attempt"], run, pending=pending)
    return {"group_id": group["group_id"], "mode": mode, "facts": len(accepted),
        "candidates": len(candidates), "quarantined": len(quarantine), "failure": failure,
        "embedding": embedding, "semantic_support_verified": False}


def audit(run=writer.OUT):
    run = Path(run).resolve()
    data = load_data()
    results = []
    for ci in (1, 7):
        old_root = writer.OLD / "conversations" / ("conv_" + str(ci)) / "banks/regroup"
        raw = raw_turns_from_locomo(data[ci]["conversation"])
        archive = read(old_root / "raw_source.json")
        if archive["raw_turns"] != raw or archive["source_sha256"] != fingerprint(raw):
            raise ValueError("Original raw archive/source hash changed")
        groups = read(old_root / "grouping.json")["groups"]
        verify_partition(raw, groups)
        sources = {g["group_id"]: ("original_v2", g, writer.OLD) for g in read(old_root / "group_fact_bank.json")["groups"]}
        if ci == 1:
            for g in read(writer.PREVIOUS / "banks/conv_1/bank_overlay.json")["groups"]:
                if g["kind"] == "new_v2":
                    sources[g["group_id"]] = ("previous_recovery_ordinary_v2", g["record"], writer.PREVIOUS)
        path = run / "banks" / ("conv_" + str(ci)) / "bank_overlay.json"
        bank = read(path)
        if (bank.get("schema") != "pilot400-fault-isolated-bank-v2" or bank.get("conversation") != ci
                or bank.get("source_sha256") != fingerprint(raw) or bank.get("all_source_turns_processed") != len(raw)
                or bank.get("original_grouping_sha256") != file_sha(old_root / "grouping.json")
                or bank.get("raw_source_sha256") != file_sha(old_root / "raw_source.json")
                or [g["group_id"] for g in bank["groups"]] != [g["group_id"] for g in groups]):
            raise ValueError("Bank whole-source/group partition/hash differs")
        rows, flattened = [], []
        for group, saved in zip(groups, bank["groups"]):
            gid, record = group["group_id"], saved["record"]
            if gid in sources:
                kind, original, source_run = sources[gid]
                if saved["kind"] != kind or record != original or record["members"] != group["turns"]:
                    raise ValueError("Reused group differs from exact prior successful group")
                provenance = w.validate_generation_provenance(record)
                expected_facts = record["fact_records"]
                embedding = verify_embeddings(expected_facts, record["embedding_attempt"], source_run)
                row = {"group_id": gid, "kind": kind, "facts": len(expected_facts), "generation": provenance,
                       "embedding": embedding, "semantic_support_verified": False}
            else:
                if saved["kind"] != writer.VERSION:
                    raise ValueError("Unexpected new group protocol")
                record_path = run / "banks" / ("conv_" + str(ci)) / "groups" / gid / "record.json"
                if read(record_path) != record:
                    raise ValueError("Standalone new group differs from bank copy")
                row = {"kind": writer.VERSION, **audit_new_group(ci, group, raw, record, run)}
                expected_facts = record["facts"]
            if saved["facts"] != expected_facts:
                raise ValueError("Bank group facts alias differs")
            flattened.extend(expected_facts)
            rows.append(row)
        if bank["facts"] != flattened or len({f["id"] for f in flattened}) != len(flattened):
            raise ValueError("Whole bank differs from exact admitted group concatenation")
        results.append({"conversation": ci, "bank_path": str(path), "bank_sha256": file_sha(path),
            "groups": rows, "group_count": len(rows), "source_turns": len(raw), "facts": len(flattened),
            "kinds": dict(collections.Counter(r["kind"] for r in rows)),
            "new_modes": dict(collections.Counter(r["mode"] for r in rows if "mode" in r)),
            "memory_kinds": dict(collections.Counter(f.get("memory_kind", "legacy_generated_fact") for f in flattened))})
    return {"schema": "pilot400-members-only-bank-provenance-audit-v2", "status": "passed",
        "run_dir": str(run), "source_dataset_sha256": SOURCE_HASH, "created_utc": now(),
        "source_code_sha256": file_sha(__file__), "writer_code_sha256": file_sha(writer.__file__),
        "model_calls": 0, "credentials_loaded": False, "semantic_support_verified": False,
        "limitation": "Proves recorded generation/admission/embedding lineage, not factual completeness or model-judgment correctness.",
        "banks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=writer.OUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Audit output is immutable; use a new report path")
    report = audit(args.run)
    publish_json(args.out, report)
    print(json.dumps({"status": report["status"], "banks": [{"conversation": b["conversation"],
        "facts": b["facts"], "group_count": b["group_count"], "new_modes": b["new_modes"]} for b in report["banks"]],
        "model_calls": 0, "report_sha256": file_sha(args.out)}, indent=2))


if __name__ == "__main__":
    main()
