"""Read-only terminal source/date/raw/persistence/index audit for new grouped arms.

No Memory/Qdrant/model client is constructed. SQLite helpers and BM25 comparison
are reused from existing audits. Semantic correctness and QA admission remain
separate reviews; an index orphan blocks the strict result and is never repaired.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import mem0_regroup_firstpass_review_v2 as index_audit
from jev_mem.local_bm25_context_v2 import compare_persisted_fact_vectors, forbid_network
from jev_mem.topic_grouping import raw_turns_from_locomo, verify_partition

VERSION = "regroup-refinement-terminal-bank-audit-v1"
ARMS = {"llm_fullconv", "jev_refined", "source_local_context"}
GENERATION = {"temperature": 0, "maxOutputTokens": 16384,
              "thinkingConfig": {"thinkingBudget": 0}, "responseMimeType": "application/json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pure_functions(path, names, environment):
    """Load only named pure definitions, never execute the original runner."""
    index_audit.sha(path)
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    found = {n.name for n in selected if isinstance(n, ast.FunctionDef)}
    found.update(t.id for n in selected if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name))
    require(set(names) <= found, "Requested audit definitions missing")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), environment)
    return environment


def prompt_environment():
    env = {"json": json, "copy": copy, "datetime": datetime, "date": date,
           "timezone": timezone, "re": re}
    pure_functions(ROOT / "jev_mem/data.py", {"date_value", "session_envelopes"}, env)
    pure_functions(ROOT / "upstream/mem0/mem0/memory/utils.py", {"parse_messages"}, env)
    pure_functions(ROOT / "upstream/mem0/mem0/configs/prompts.py", {
        "ADDITIVE_EXTRACTION_PROMPT", "PAST_MESSAGE_TRUNCATION_LIMIT", "_truncate_content",
        "_format_summary", "_format_conversation_history", "_serialize_memories",
        "_format_new_messages", "_resolve_dates", "generate_additive_extraction_prompt"}, env)
    pure_functions(ROOT / "tools/run_mem0_regroup_sdk20.py", {"MULTIPLE_DATES", "render_source"}, env)
    def valid_date(value):
        require(type(value) is str and date.fromisoformat(value).isoformat() == value, "Invalid source date")
        return value
    env["frozen"] = SimpleNamespace(digest=index_audit.digest, render_source=env["render_source"],
                                   original=SimpleNamespace(_validate_date=valid_date))
    pure_functions(ROOT / "jev_mem/source_local_group_context_v1.py",
                   {"VERSION", "CONTEXT_ONLY", "MEMBER_SCOPE", "_source_index", "build_context_plan"}, env)
    return env


def terminal_preflight(run, conversations):
    """All requested banks must be closed complete before any SQLite is opened."""
    run = Path(run).resolve()
    m = index_audit.read(run / "run_manifest.json")
    require(m["protocol_sha256"] == index_audit.digest({k: v for k, v in m.items() if k != "protocol_sha256"}),
            "Writer manifest seal mismatch")
    refinement = m.get("refinement", {})
    arm = refinement.get("arm")
    require(refinement.get("schema") == "mem0-regroup-refinement-writer-v1" and arm in ARMS
            and m["arms"] == [arm], "Unsupported grouped writer protocol")
    require(refinement.get("source_local_context") == (arm == "source_local_context"), "Context arm mismatch")
    require(conversations and len(set(conversations)) == len(conversations)
            and all(type(ci) is int and ci in range(10) for ci in conversations), "Invalid conversation cohort")
    targets = []
    for ci in conversations:
        folder = run / f"conv_{ci}" / arm
        state = index_audit.read(folder / "write_status.json")
        require(state.get("status") == "complete" and state.get("finished_utc")
                and state.get("completed_groups") == state.get("expected_groups"),
                f"c{ci}/{arm} is not closed complete; no SQLite audit permitted")
        outcome = index_audit.verify(folder / "write_outcome.json")
        bank = index_audit.bound(state["bank"])
        require(outcome.get("status") == "complete" and bank.get("complete") is True
                and outcome.get("cache_roots") == [] and outcome.get("bank") == state["bank"]
                and outcome.get("write_status") == index_audit.pointer(folder / "write_status.json"),
                "Writer terminal/bank/outcome chain incomplete")
        require(all(x.get("protocol_sha256") == m["protocol_sha256"]
                    and (x.get("conversation"), x.get("arm")) == (ci, arm)
                    for x in (state, outcome, bank)), "Terminal protocol/identity mismatch")
        require(index_audit.verify(folder / "bank.json") == bank, "Bank seal mismatch")
        for p in (folder / "native").rglob("*"):
            require(not p.is_symlink(), "Native symlink forbidden")
            if p.is_file() and p.name.endswith(("-wal", "-journal")):
                require(p.stat().st_size == 0, "Uncheckpointed native database; audit refused")
        targets.append((ci, arm, folder, state, outcome, bank))
    for name, expected in {**m["code_sha256"], **refinement["code_sha256"]}.items():
        require(index_audit.sha(ROOT / name) == expected, "Frozen writer/adapter code changed: " + name)
    for name, expected in {**m["upstream_source_sha256"], **m["grouping_source_sha256"]}.items():
        require(index_audit.sha(name) == expected, "Frozen upstream/grouping source changed")
    index_audit.bound(refinement["base_manifest"])
    return m, targets


def effective_inputs(journal, envelope, history, grouping, sessions, local, env):
    """Reconstruct native and effective prompt inputs without trusting labels."""
    native_messages = [env["render_source"](m, s) for m, s in zip(envelope["messages"], envelope["turn_sources"])]
    require(journal["writer_messages"] == native_messages, "Frozen native source rendering differs")
    request_keys = ("user_id", "group_id", "segment_ids", "messages", "writer_messages", "turn_sources",
                    "source_turn_ids", "source_order_indices", "source_observation_dates", "observation_date", "current_date")
    require(journal["request_sha256"] == index_audit.digest({k: journal[k] for k in request_keys}),
            "Native group request fingerprint mismatch")
    if local:
        context = env["build_context_plan"](envelope, source_sessions=sessions, segments=grouping["segments"])
        require(journal.get("source_local_context") == context
                and journal["history_context"] == context["history_context"], "Effective source-local context differs")
        require(journal.get("effective_request_sha256") == index_audit.digest({
            "native_request_sha256": journal["request_sha256"], "source_local_context_sha256": context["sha256"]}),
            "Effective extraction fingerprint mismatch")
        return context["writer_messages"], context["history_context"], context
    require("source_local_context" not in journal and journal["history_context"] == history[-10:],
            "Ordinary grouped history differs")
    return native_messages, history[-10:], None


def check_generation(journal, ledger, native, prompt, env, ci, arm):
    calls = journal["llm_calls"]
    require(journal["builder_call_count"] == 1 and len(calls) == 1, "Exactly one native extraction required")
    call = calls[0]
    messages = [{"role": "system", "content": env["ADDITIVE_EXTRACTION_PROMPT"]}, {"role": "user", "content": prompt}]
    require(call["kwargs"]["messages"] == messages, "Actual captured writer messages differ from reconstructed prompt")
    events = [e for e in ledger if e.get("stage") == "mem0_extract"
              and e.get("context", {}).get("group_id") == journal["group_id"]]
    success = [e for e in events if e.get("cache_hit") or e.get("status") == 200]
    require(len(success) == 1, "Extraction success ledger must identify one original raw")
    event = success[0]
    require(event.get("provider") == "gemini" and event.get("model") == "gemini-2.5-flash"
            and event.get("operation") == "generateContent"
            and event["context"].get("conversation") == ci and event["context"].get("method") == arm,
            "Extraction event provider/model/context differs")
    path = native / "api_cache" / (event["request_hash"] + ".json")
    raw = index_audit.read(path)
    expected = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "body": {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": env["ADDITIVE_EXTRACTION_PROMPT"]}]}, "generationConfig": GENERATION}}
    require(raw["request"] == expected and raw["stage"] == "mem0_extract", "Raw original extraction request differs")
    require(hashlib.sha256(json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == event["request_hash"],
            "Raw request hash differs")
    response = raw["response"]
    require(response.get("modelVersion") in (None, "gemini-2.5-flash"), "Resolved writer model differs")
    candidates = response.get("candidates", [])
    require(len(candidates) == 1 and candidates[0].get("finishReason") == "STOP", "Incomplete writer response")
    parsed = json.loads("".join(p.get("text", "") for p in candidates[0]["content"]["parts"] if not p.get("thought")))
    require(parsed == json.loads(call["response"]) and parsed["memory"] == call["parsed_memory"], "Raw-to-parser lineage differs")
    require(isinstance(parsed["memory"], list) and all(isinstance(x, dict) and isinstance(x.get("text"), str)
            and x["text"].strip() for x in parsed["memory"]), "Malformed fact records")
    return parsed["memory"], index_audit.pointer(path)


def inspect_bank(target, manifest, conversation, env, extractor, nlp):
    ci, arm, folder, state, outcome, bank = target
    native = folder / "native"
    before = index_audit.native_inventory(native)
    inp = index_audit.verify(folder / "group_inputs.json")
    declared = next(x["input"] for x in manifest["inputs"] if (x["conversation"], x["arm"]) == (ci, arm))
    require(index_audit.bound(declared) == inp and index_audit.bound(bank["group_inputs"]) == inp, "Group inputs changed")
    grouping = index_audit.bound(inp["grouping"])
    require(grouping.get("status") == "complete" and grouping.get("qa_visible") is False
            and grouping.get("conversation") == ci, "Grouping source incomplete or wrong conversation")
    raw = raw_turns_from_locomo(conversation)
    verify_partition(raw, grouping["segments"])
    verify_partition(raw, grouping["groups"])
    sessions = list(env["session_envelopes"]({"conversation": conversation}))
    source = {}
    for session in sessions:
        for tid, message in zip(session["source_turn_ids"], session["messages"]):
            source[tid] = ({"source_turn_id": tid, "session_id": session["session_id"],
                            "observation_date": session["observation_date"], "source_order_index": len(source)}, message)
    segments = {s["segment_id"]: s for s in grouping["segments"]}
    require(len(segments) == len(grouping["segments"]) and Counter(sid for g in grouping["groups"] for sid in g["segment_ids"])
            == Counter(segments.keys()), "Source segment group assignment differs")
    envelopes = []
    for group in sorted(grouping["groups"], key=lambda g: min(source[t][0]["source_order_index"] for t in g["source_turn_ids"])):
        ids = group["source_turn_ids"]
        require(ids == sorted([t for sid in group["segment_ids"] for t in segments[sid]["source_turn_ids"]],
                              key=lambda t: source[t][0]["source_order_index"]), "Group is not exact full segment union")
        members = [source[t] for t in ids]
        dates = sorted({s["observation_date"] for s, m in members})
        envelopes.append({"group_id": group["group_id"], "segment_ids": group["segment_ids"],
            "messages": [m for s, m in members], "turn_sources": [s for s, m in members], "source_turn_ids": ids,
            "source_order_indices": [source[t][0]["source_order_index"] for t in ids], "source_observation_dates": dates,
            "observation_date": dates[0] if len(dates) == 1 else None})
    require(inp["envelopes"] == envelopes and inp["turn_count"] == len(source)
            and len(bank["journal_files"]) == len(envelopes), "Writer source envelopes/coverage differ")
    facts = index_audit.points(native / "qdrant/collection/facts/storage.sqlite")
    entities = index_audit.points(native / "qdrant/collection/facts_entities/storage.sqlite")
    history_rows = index_audit.rows(native / "history.sqlite", "history")
    index_audit.validate_records(bank["facts"], facts, history_rows)
    exported = {f["id"]: f for f in bank["facts"]}
    ledger_path = native / "api_calls.jsonl"
    index_audit.sha(ledger_path)
    ledger = [json.loads(line) for line in index_audit.shared_read(ledger_path).decode("utf-8").splitlines()]
    history, emitted, journals, quality = [], [], [], []
    max_difference = 0.
    for seq, (envelope, ref) in enumerate(zip(envelopes, bank["journal_files"])):
        journal = index_audit.bound(ref)
        require(journal.get("status") == "success" and journal["sequence"] == seq
                and all(journal[k] == value for k, value in envelope.items()), "Journal/source sequence differs")
        require(journal["current_date"] == manifest["current_date"], "Frozen current date differs")
        writer_messages, effective_history, context = effective_inputs(journal, envelope, history, grouping, sessions,
                                                                       arm == "source_local_context", env)
        prior = {f["id"]: f for f in emitted}
        for existing in journal["existing_memory_context"]:
            require(existing["id"] in prior and existing["text"] == prior[existing["id"]]["text"]
                    and existing["hash"] == hashlib.md5(existing["text"].encode()).hexdigest(), "Existing fact context differs")
        prompt = env["generate_additive_extraction_prompt"](
            existing_memories=[{"id": str(i), "text": f["text"]} for i, f in enumerate(journal["existing_memory_context"])],
            new_messages=env["parse_messages"](writer_messages),
            last_k_messages=[x["writer_message"] for x in effective_history], current_date=manifest["current_date"],
            timestamp=envelope["observation_date"] or env["MULTIPLE_DATES"], custom_instructions=None)
        extracted, raw_ref = check_generation(journal, ledger, native, prompt, env, ci, arm)
        existing_hashes = {f["hash"] for f in journal["existing_memory_context"]}
        expected = {f["text"] for f in extracted if hashlib.md5(f["text"].encode()).hexdigest() not in existing_hashes}
        require({r["memory"] for r in journal["results"]} == expected
                and len(journal["results"]) == len(expected) == len(journal["fact_records"])
                and all(r["event"] == "ADD" for r in journal["results"])
                and journal["sdk_response"]["results"] == journal["results"], "Parsed-to-native ADD chain differs")
        require(not journal["persistence_check"]["missing_texts"] and not journal["persistence_check"]["unexpected_texts"]
                and journal["explicit_empty_extraction"] == (not extracted), "Incomplete persistence or empty extraction marker")
        for fact, added in zip(journal["fact_records"], journal["results"]):
            require((fact["id"], fact["text"], fact["event"]) == (added["id"], added["memory"], "ADD")
                    and fact["verified_support"] == [] and fact["support_status"] == "unverified_group_input_scope", "Fact provenance changed")
            scope = {"group_id": journal["group_id"], "segment_ids": journal["segment_ids"],
                     "source_turn_ids": journal["source_turn_ids"], "turn_sources": journal["turn_sources"],
                     "historical_turn_ids": [x["source_turn_id"] for x in effective_history],
                     "existing_memory_ids": [x["id"] for x in journal["existing_memory_context"]]}
            if context:
                scope.update(history_policy=env["VERSION"], history_context_only=True)
            require(fact["input_scope"] == scope and fact["observation_date"] == envelope["observation_date"]
                    and fact["source_observation_dates"] == envelope["source_observation_dates"], "Effective fact source/date scope differs")
            point = facts[fact["id"]]
            payload = point.payload
            require(payload["user_id"] == bank["user_id"] and payload["jev_mem_segment_ids"] == journal["segment_ids"]
                    and payload["jev_mem_date_scope"] == fact["observation_date_scope"] ==
                    ("single_date_group" if len(envelope["source_observation_dates"]) == 1 else "mixed_dates_unattributed"),
                    "Native payload scope differs")
            original_fact = next(f for f in extracted if f["text"] == fact["text"])
            require(payload.get("attributed_to") == original_fact.get("attributed_to"), "Extracted attribution changed before persistence")
            vector = exported[fact["id"]]["embedding"]
            require(index_audit.dense(vector), "Invalid exported dense vector")
            difference = max(abs(x-y) for x, y in zip(point.vector[""], vector))
            require(difference <= 1e-6 and all(abs(math.sqrt(sum(x*x for x in v))-1) <= 1e-5
                    for v in (point.vector[""], vector)), "Dense export/persistence mismatch")
            max_difference = max(max_difference, difference)
        if len(envelope["source_observation_dates"]) > 1 or not extracted or context:
            quality.append({"group_id": journal["group_id"], "journal": ref, "raw": raw_ref,
                            "mixed_dates": len(envelope["source_observation_dates"]) > 1,
                            "empty_extraction": not extracted, "effective_history_turn_ids": [x["source_turn_id"] for x in effective_history]})
        journals.append({"group_id": journal["group_id"], "journal": ref, "raw": raw_ref,
                         "facts": len(journal["fact_records"]), "context_only_turns": len(effective_history) if context else 0})
        emitted.extend(journal["fact_records"])
        history.extend({**s, "message": m, "writer_message": wm, "group_id": journal["group_id"]}
                       for s, m, wm in zip(journal["turn_sources"], journal["messages"], journal["writer_messages"]))
    require(emitted == [{k: v for k, v in fact.items() if k != "embedding"} for fact in bank["facts"]]
            and state["facts"] == len(facts) and state["completed_groups"] == len(envelopes), "Bank/journal/status fact counts differ")
    linked, missing_sparse, empty_sparse = set(), [], []
    for eid, point in entities.items():
        vectors = point.vector if isinstance(point.vector, dict) else {"": point.vector}
        require(index_audit.dense(vectors.get("")) and point.payload["user_id"] == bank["user_id"], "Invalid/foreign entity dense")
        links = point.payload.get("linked_memory_ids", [])
        require(links and len(set(links)) == len(links) and set(links) <= set(facts), "Entity links missing/duplicate/foreign")
        linked.update(links)
        sparse = vectors.get("bm25")
        if sparse is None:
            missing_sparse.append(eid)
        else:
            require(index_audit.sparse(sparse, allow_empty=True), "Malformed entity sparse")
            if not sparse.indices:
                empty_sparse.append(eid)
    docs = nlp.pipe([f["text"] for f in bank["facts"]], batch_size=32)
    expected_entities = {f["id"] for f, doc in zip(bank["facts"], docs) if extractor._extract_entities_from_doc(doc)}
    orphans = sorted(expected_entities - linked)
    declared_index = bank["native_index_audit"]
    require(declared_index["facts_with_local_entities"] == len(expected_entities)
            and declared_index["entity_facts_without_any_link"] == orphans
            and set(declared_index["linked_memory_ids"]) == linked and declared_index["entities"] == len(entities),
            "Declared native index audit differs from offline recomputation")
    require(index_audit.native_inventory(native) == before, "Native source changed during audit")
    return {"conversation": ci, "arm": arm, "status": "strict_pass" if not orphans else "blocked_entity_links",
            "strict_passed": not orphans, "source_generation_persistence_passed": True,
            "bank": index_audit.pointer(folder / "bank.json"), "inputs": index_audit.pointer(folder / "group_inputs.json"),
            "outcome": index_audit.pointer(folder / "write_outcome.json"), "native_dir": str(native),
            "groups": len(envelopes), "source_turns": len(source), "facts": len(facts),
            "history_adds": len(history_rows), "entity_points": len(entities), "orphan_fact_ids": orphans,
            "unused_entity_bm25_missing_ids": missing_sparse, "unused_entity_bm25_empty_ids": empty_sparse,
            "dense_export_max_difference": max_difference, "journals": journals, "semantic_review_targets": quality,
            "native_inventory": before, "writer_prompt_reconstructed_and_raw_verified": True,
            "source_local_context_effective_inputs_verified": arm == "source_local_context",
            "semantic_correctness_verified": False, "qa_gate_passed": False}


def audit(run, conversations):
    index_audit.HASHES.clear()
    run = Path(run).resolve()
    with forbid_network() as guards:
        manifest, targets = terminal_preflight(run, conversations)
        dataset = ROOT / "data/locomo10.json"
        require(index_audit.sha(dataset) == manifest["dataset_sha256"], "Dataset changed")
        conversations_data = [item["conversation"] for item in index_audit.read(dataset)]
        env = prompt_environment()
        extractor, nlp = index_audit.load_extractor()
        banks = [inspect_bank(target, manifest, conversations_data[target[0]], env, extractor, nlp) for target in targets]
        resources = manifest["refinement"]["resources"]
        bm25 = compare_persisted_fact_vectors([Path(b["native_dir"]) for b in banks], resources["snapshot"], resources["files_sha256"])
        require(bm25["facts_compared"] == bm25["exact_matches"] == sum(b["facts"] for b in banks), "BM25 coverage differs")
        require(not any(g.call_count for g in guards), "Network was attempted")
        for path, expected in list(index_audit.HASHES.items()):
            require(index_audit.sha(path) == expected, "Audit source changed")
    passed = all(b["strict_passed"] for b in banks)
    return {"schema": VERSION, "status": "strict_source_index_passed_semantic_review_pending" if passed else "blocked_entity_links",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
            "protocol_sha256": manifest["protocol_sha256"], "manifest": index_audit.pointer(run / "run_manifest.json"),
            "banks": banks, "strict_passed": passed, "offline_bm25_equivalence": bm25,
            "model_calls": 0, "network_attempts": 0, "memory_clients": 0, "qdrant_clients": 0,
            "source_mutations": 0, "qa_gold_used": False, "source_file_hashes": dict(index_audit.HASHES),
            "audit_tool": index_audit.pointer(__file__), "qa_gate_passed": False,
            "limitations": ["Source input scope is not per-fact semantic support.",
                            "Unused entity sparse omissions are disclosed, not repaired.",
                            "Local entity extraction checks facts with no links, not every expected partial entity edge."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--conversations", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Never overwrite an audit report")
    try:
        result = audit(args.run_dir, args.conversations)
    except Exception as error:
        result = {"schema": VERSION, "status": "refused_or_failed", "strict_passed": False,
                  "run": str(args.run_dir.resolve()), "conversations": args.conversations,
                  "error_type": type(error).__name__, "error": str(error), "qa_gate_passed": False,
                  "model_calls": 0, "source_mutations": 0}
    ref = index_audit.save(args.output, result)
    print(json.dumps({"report": ref, "status": result["status"], "strict_passed": result["strict_passed"],
                      "banks": [{k: b[k] for k in ("conversation", "arm", "groups", "facts", "orphan_fact_ids")}
                                for b in result.get("banks", [])]}, ensure_ascii=False, indent=2))
    return 0 if result["strict_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
