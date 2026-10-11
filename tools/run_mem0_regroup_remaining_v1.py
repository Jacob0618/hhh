"""Gated 8-conversation dated Mem0 writing using the frozen native writer.

Only conv2-9 receive new empty stores. Four reviewed development banks are
immutable references. No API call occurs on import, prepare, or check.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_regroup_sdk20 as frozen
from tools import recover_mem0_regroup_sdk20 as once
from tools import recover_mem0_regroup_bm25_v2 as class_recovery
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.run_sdk20_modules import digest, sha, now, verify
from jev_mem.local_bm25_context_v2 import local_bm25_context, validate_resources, compare_persisted_fact_vectors
from jev_mem.paced_embedding_api import PacedEmbeddingAPI, DEFAULT_GATE

SCHEMA = "dated-mem0-remaining-eight-conversations-v1"
GATE_SCHEMA = "mem0-dated-regroup-development-gate-v1"
SOURCE = ROOT / "runs/mem0_global_regroup_v1"
PROBE = ROOT / "research/mem0_global_regroup_execution_2026-09-28/bm25_local_snapshot_probe.json"
DEVELOPMENT = ROOT / "runs/mem0_global_regroup_recovery_v1"
DEVELOPMENT_FOURTH = ROOT / "runs/mem0_global_regroup_bm25_recovery_v2"
CODE = ("tools/run_mem0_regroup_remaining_v1.py", "jev_mem/local_bm25_context_v2.py",
        "jev_mem/paced_embedding_api.py", "jev_mem/sdk20_cached_api.py",
        "tools/recover_mem0_regroup_sdk20.py", "tools/recover_mem0_regroup_bm25_v2.py")
CHECKS = ("four_complete_banks", "source_dates_and_generation_chain_verified",
          "current_sdk20_index_dependencies_valid", "content_quality_findings_disclosed")
WRITER_FIELDS = ("schema", "dataset_sha256", "writer", "models", "current_date",
    "provider_retries", "protocol_changes", "embedding_policy", "empty_extraction_policy",
    "downstream", "dependencies", "code_sha256", "upstream_source_sha256", "grouping_source_sha256")
POLICY = {"schema": SCHEMA, "conversations": list(range(2, 10)), "arms": list(frozen.ARMS),
    "workers": 8, "arms_within_conversation": "sequential", "provider_retries": 2,
    "embedding_gate": "shared3sec_physical_start_429cooldown_at_least30sec",
    "api_cache_roots": [], "bm25_context": "class_preserving_fixed_local_snapshot_v2",
    "bank_attempt": "once_new_namespace_no_reenter_failed_or_unknown",
    "development_banks": "four_reviewed_references_never_rewritten",
    "writer": "frozen_original_write_arm_no_model_prompt_body_change",
    "qa": "not_started_by_this_tool"}


def file_pointer(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Reviewed artifact pointer changed")
    return Path(value["path"])


def development_gate(path, parent):
    gate = verify(path)
    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "passed":
        raise ValueError("A passed four-bank development review is required")
    if any(gate.get("checks", {}).get(k) is not True for k in CHECKS):
        raise ValueError("Development source/date/index review checks are incomplete")
    if not isinstance(gate.get("limitations"), list) or not gate["limitations"] or any(not isinstance(s, str) or not s.strip() for s in gate["limitations"]):
        raise ValueError("Known model/content limitations must be disclosed")
    audits = gate.get("audits", {})
    for kind in ("source_date", "index"):
        if not isinstance(audits.get(kind), list) or not audits[kind]:
            raise ValueError("Missing development review artifact references")
        for artifact in audits[kind]:
            file_pointer(artifact)
    rows = gate.get("banks", [])
    if len(rows) != 4 or {(r["conversation"], r["arm"]) for r in rows} != {(c, a) for c in (0, 1) for a in frozen.ARMS}:
        raise ValueError("Gate must bind exactly four development banks")
    # The reviewed artifacts must belong to the still-frozen execution sources,
    # including each source's transport and local-resource recovery adapters.
    source_manifests = {DEVELOPMENT.resolve(): once.manifest(DEVELOPMENT),
                        DEVELOPMENT_FOURTH.resolve(): class_recovery.manifest(DEVELOPMENT_FOURTH)}
    for source_manifest in source_manifests.values():
        if any(field not in parent or field not in source_manifest or source_manifest[field] != parent[field] for field in WRITER_FIELDS):
            raise ValueError("Development writer/model/input protocol differs from expansion")
    for row in rows:
        ci, arm = row["conversation"], row["arm"]
        source = DEVELOPMENT_FOURTH if (ci, arm) == (1, "dated_merged") else DEVELOPMENT
        folder = source / f"conv_{ci}" / arm
        for key, name in (("bank", "bank.json"), ("write_status", "write_status.json"), ("outcome", "recovery_outcome.json")):
            if file_pointer(row[key]).resolve() != (folder / name).resolve():
                raise ValueError("Development bank source differs from the reviewed lineage")
        bank = verify(row["bank"]["path"])
        status = read(row["write_status"]["path"])
        outcome = verify(row["outcome"]["path"])
        if (bank.get("complete") is not True or status.get("status") != "complete" or outcome.get("status") != "complete"
                or (bank.get("conversation"), bank.get("arm")) != (ci, arm)
                or status.get("bank") != row["bank"] or outcome.get("bank") != row["bank"]
                or outcome.get("write_status") != row["write_status"]
                or any(artifact.get("protocol_sha256") != source_manifests[source.resolve()]["protocol_sha256"]
                       for artifact in (bank, status, outcome))
                or status.get("facts") != len(bank["facts"])
                or len({f["id"] for f in bank["facts"]}) != len(bank["facts"])):
            raise ValueError("Referenced development bank is incomplete or inconsistent")
        original = next(x for x in parent["inputs"] if (x["conversation"], x["arm"]) == (ci, arm))
        frozen.check_pointer(bank["group_inputs"])
        if bank["group_inputs"]["sha256"] != original["input"]["sha256"]:
            raise ValueError("Development bank used a different group input")
        if status["completed_groups"] != status["expected_groups"] or status["expected_groups"] != len(read(bank["group_inputs"]["path"])["envelopes"]):
            raise ValueError("Development groups are incomplete")
    return gate


def resources():
    probe = read(PROBE)
    value = {"snapshot": probe["snapshot"], "files_sha256": probe["snapshot_files_sha256"], "probe": frozen.pointer(PROBE)}
    validate_resources(value["snapshot"], value["files_sha256"])
    return value


def offline_acceptance(gate, resource):
    # The same three frozen banks used in the original equivalence audit remain
    # the fixed algorithm comparison. The fourth bank is admitted separately.
    natives = [str(Path(r["bank"]["path"]).parent / "native") for r in gate["banks"] if (r["conversation"], r["arm"]) != (1, "dated_merged")]
    proof = compare_persisted_fact_vectors(natives, resource["snapshot"], resource["files_sha256"])
    if proof["facts_compared"] != 518:
        raise ValueError("Fixed development sparse equivalence population changed")
    return {"vectors": proof, "native": class_recovery.fresh_native_smoke(resource)}


def prepare(run_dir, gate_path, source_run=SOURCE):
    run = Path(run_dir).resolve()
    source = Path(source_run).resolve()
    if source != SOURCE.resolve():
        raise ValueError("Expansion must use the frozen original grouped inputs")
    if run.exists() or source in run.parents:
        raise FileExistsError("Expansion requires a new independent namespace")
    parent = frozen.manifest(source)
    gate = development_gate(gate_path, parent)
    resource = resources()
    proof = offline_acceptance(gate, resource)
    replacements = {}
    for item in parent["inputs"]:
        ci, arm = item["conversation"], item["arm"]
        if ci not in range(2, 10):
            continue
        src = file_pointer(item["input"])
        dst = run / f"conv_{ci}" / arm / "group_inputs.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("xb") as handle:
            handle.write(src.read_bytes())
        if sha(src) != sha(dst):
            raise ValueError("Frozen remaining conversation input copy changed")
        replacements[(ci, arm)] = {**item, "input": frozen.pointer(dst)}
    if len(replacements) != 16:
        raise ValueError("Expected sixteen new conversation-arm inputs")
    value = copy.deepcopy(parent)
    value.pop("protocol_sha256")
    value["created_utc"] = now()
    value["inputs"] = [replacements.get((x["conversation"], x["arm"]), x) for x in parent["inputs"]]
    value["remaining_expansion"] = {"schema": SCHEMA, "policy": POLICY,
        "source_manifest": frozen.pointer(source / "run_manifest.json"),
        "development_gate": frozen.pointer(gate_path), "development_banks": gate["banks"],
        "resources": resource, "offline_acceptance": proof,
        "code_sha256": {p: sha(ROOT / p) for p in CODE}, "gate_path": str(DEFAULT_GATE.resolve()),
        "runtime_source_sha256": {p: sha(ROOT / p) for p in (
            ".venv/Lib/site-packages/fastembed/sparse/bm25.py",
            ".venv/Lib/site-packages/fastembed/sparse/sparse_text_embedding.py",
            ".venv/Lib/site-packages/fastembed/common/model_management.py",
            ".venv/Lib/site-packages/qdrant_client/fastembed_common.py",
            ".venv/Lib/site-packages/qdrant_client/qdrant_fastembed.py")},
        "model_calls_during_prepare": 0}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    frozen.save_new(run / "offline_dependency_acceptance.json", proof)
    return {"prepared": str(run), "protocol_sha256": digest(value), "new_banks": 16,
            "development_references": 4, "model_calls": 0}


def manifest(run):
    value = frozen.manifest(run)
    info = value.get("remaining_expansion", {})
    if info.get("schema") != SCHEMA or info.get("policy") != POLICY:
        raise ValueError("Remaining expansion policy changed")
    for path, expected in {**info["code_sha256"], **info["runtime_source_sha256"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen remaining runtime changed: " + path)
    parent = frozen.check_pointer(info["source_manifest"])
    file_pointer(info["development_gate"])
    gate = development_gate(info["development_gate"]["path"], parent)
    if gate["banks"] != info["development_banks"] or resources() != info["resources"]:
        raise ValueError("Development gate/resources changed")
    if verify(Path(run) / "offline_dependency_acceptance.json") != {**info["offline_acceptance"], "content_sha256": digest(info["offline_acceptance"])}:
        raise ValueError("Offline dependency acceptance changed")
    return value


def write_arm_once(run, ci, arm, value):
    if ci not in range(2, 10) or arm not in frozen.ARMS:
        raise ValueError("Only the sixteen non-development banks are admitted")
    folder = Path(run) / f"conv_{ci}" / arm
    outcome_path = folder / "write_outcome.json"
    if outcome_path.exists():
        outcome = verify(outcome_path)
        if outcome.get("protocol_sha256") != value["protocol_sha256"] or outcome.get("status") != "complete":
            raise RuntimeError("Failed expansion bank is retained; no implicit retry")
        frozen.check_pointer(outcome["write_status"])
        frozen.check_pointer(outcome["bank"])
        return outcome
    intent = folder / "write_intent.json"
    if intent.exists() or (folder / "write_status.json").exists() or (folder / "native").exists():
        raise RuntimeError("Prior nontransactional expansion attempt is retained")
    once._write_intent(intent, {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"],
        "conversation": ci, "arm": arm, "pid": os.getpid(), "created_utc": now(),
        "input": frozen.pointer(folder / "group_inputs.json"), "cache_roots": []})
    original_api = frozen.ModelAPI
    def constructor(path, **kwargs):
        return PacedEmbeddingAPI(path, cache_roots=(), gate_path=value["remaining_expansion"]["gate_path"], **kwargs)
    try:
        frozen.ModelAPI = constructor
        state = frozen.write_arm(run, ci, arm, value)
    finally:
        frozen.ModelAPI = original_api
    outcome = {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"],
        "conversation": ci, "arm": arm, "status": state["status"], "finished_utc": now(),
        "write_status": frozen.pointer(folder / "write_status.json"), "cache_roots": []}
    if state["status"] == "complete":
        outcome["bank"] = state["bank"]
    else:
        outcome["error"] = state.get("error")
    frozen.save_new(outcome_path, outcome)
    return outcome


def write_conv(run, ci):
    if ci not in range(2, 10):
        raise ValueError("Development conversations may not be written by this runner")
    value = manifest(run)
    resource = value["remaining_expansion"]["resources"]
    with local_bm25_context(resource["snapshot"], resource["files_sha256"]):
        return [write_arm_once(run, ci, arm, value) for arm in frozen.ARMS]


def write(run):
    value = manifest(run)
    info = value["remaining_expansion"]
    gate = development_gate(info["development_gate"]["path"], frozen.check_pointer(info["source_manifest"]))
    offline_acceptance(gate, info["resources"])
    with frozen.writer_process_lock(run), ProcessPoolExecutor(max_workers=8) as pool:
        return [f.result() for f in as_completed([pool.submit(write_conv, run, ci) for ci in range(2, 10)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--development-gate", type=Path, required=True)
    p.add_argument("--source-run", type=Path, default=SOURCE)
    for name in ("check", "write"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_dir, args.development_gate, args.source_run)
    elif args.command == "check":
        value = manifest(args.run_dir)
        result = {"protocol_sha256": value["protocol_sha256"], "model_calls": 0}
    else:
        result = write(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
