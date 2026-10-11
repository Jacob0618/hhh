"""One new namespace for the pre-model conv1 merged BM25 startup failure.

The three complete banks are references, never rewritten. The original writer,
first-extraction pin, API body and physical embedding pacing are reused.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_mem0_regroup_sdk20 as legacy
from tools import run_mem0_regroup_sdk20 as frozen
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.run_sdk20_modules import digest, sha, now, verify
from jev_mem.local_bm25_context import PROTOCOL as BM25_PROTOCOL, local_bm25_context, validate_resources, compare_persisted_fact_vectors

SCHEMA = "dated-mem0-c1-merged-local-bm25-recovery-v1"
TARGET = (1, "dated_merged")
AUDIT = ROOT / "research/mem0_global_regroup_execution_2026-09-28/bm25_startup_failure_terminal_audit.json"
PROBE = ROOT / "research/mem0_global_regroup_execution_2026-09-28/bm25_local_snapshot_probe.json"
CODE = ("tools/recover_mem0_regroup_bm25_v1.py", "jev_mem/local_bm25_context.py",
        "tools/recover_mem0_regroup_sdk20.py")
POLICY = {"schema": SCHEMA, "target": list(TARGET), "resource_policy": BM25_PROTOCOL,
    "reuse_successful_banks": "reference_only_no_new_write", "successful_banks": 3,
    "failed_bank": "startup_before_group_or_model_request_zero_facts",
    "new_attempt": "one_new_namespace_no_reenter_after_failed_or_unknown",
    "first_extraction": "same_original_conv1_merged_raw_23facts",
    "embedding_transport": "unchanged_PinnedRecoveryAPI_shared3s_429cooldown30s_retries2",
    "bm25": "existing_fixed_snapshot_specific_model_path_local_files_only_defaults_unchanged",
    "qa_or_other_conversations": "not_started_by_this_tool"}


def source_admission(source_run):
    source = Path(source_run).resolve()
    parent = legacy.manifest(source)
    audit = read(AUDIT)
    probe = read(PROBE)
    if audit.get("source_manifest") != frozen.pointer(source / "run_manifest.json"):
        raise ValueError("Terminal failure audit does not bind this source run")
    if audit.get("bm25_offline_probe") != frozen.pointer(PROBE):
        raise ValueError("Offline probe differs from the reviewed source audit")
    if (probe.get("run") != str(source) or probe.get("facts_compared") != 518
            or probe.get("exact_sparse_term_value_match") is not True
            or probe.get("network_calls") != 0 or probe.get("mismatches")):
        raise ValueError("Reviewed 518-fact offline equivalence probe is required")
    rows = audit["states"]
    if len(rows) != 4 or {(x["conversation"], x["arm"]) for x in rows} != {(c, a) for c in (0, 1) for a in frozen.ARMS}:
        raise ValueError("Expected the four terminal development banks")
    reused = []
    failed = None
    for row in rows:
        ci, arm = row["conversation"], row["arm"]
        folder = source / f"conv_{ci}" / arm
        state = frozen.check_pointer(row["status_pointer"])
        outcome = frozen.check_pointer(row["outcome_pointer"])
        verify(row["outcome_pointer"]["path"])
        if outcome["status"] != state["status"] or state["protocol_sha256"] != parent["protocol_sha256"]:
            raise ValueError("Terminal source state differs")
        native = folder / "native"
        if {str(p.relative_to(native)): sha(p) for p in native.rglob("*") if p.is_file()} != row["native_inventory"]:
            raise ValueError("Terminal source native files changed")
        if (ci, arm) == TARGET:
            if state["status"] != "failed" or state["completed_groups"] != 0 or state["facts"] != 0 or state.get("error") != {"type": "RuntimeError", "message": "Native NLP/BM25 unavailable"}:
                raise ValueError("Only the reviewed pre-model dependency failure is admitted")
            if ((native / "api_calls.jsonl").exists() or list((native / "api_cache").glob("*.json"))
                    or list((native / "request_intents").glob("*.json")) or list((native / "group_audit").rglob("*.json"))):
                raise ValueError("Failed bank had model activity; not the admitted startup failure")
            failed = {"conversation": ci, "arm": arm, "status": row["status_pointer"],
                      "outcome": row["outcome_pointer"], "zero_model_calls": True}
        else:
            bank = frozen.check_pointer(row["bank_pointer"])
            verify(row["bank_pointer"]["path"])
            if state["status"] != "complete" or bank.get("complete") is not True or len(bank["facts"]) != state["facts"]:
                raise ValueError("A referenced completed bank is incomplete")
            reused.append({"conversation": ci, "arm": arm, "bank": row["bank_pointer"],
                           "status": row["status_pointer"], "outcome": row["outcome_pointer"],
                           "native_dir": str(native), "facts": len(bank["facts"]), "action": "reference_only"})
    if failed is None or len(reused) != 3 or sum(x["facts"] for x in reused) != 518:
        raise ValueError("Expected three preserved banks and one startup failure")
    pin = next(r for r in parent["recovery"]["source_bindings"] if (r["conversation"], r["arm"]) == TARGET)
    if len(pin["raw_first_extraction"]["parsed_memory"]) != 23:
        raise ValueError("Original first-extraction 23-fact pin changed")
    resources = {"snapshot": probe["snapshot"], "files_sha256": probe["snapshot_files_sha256"], "probe": frozen.pointer(PROBE)}
    validate_resources(resources["snapshot"], resources["files_sha256"])
    return parent, reused, failed, resources


def offline_acceptance(reused, resources):
    report = compare_persisted_fact_vectors([r["native_dir"] for r in reused], resources["snapshot"], resources["files_sha256"])
    if report["facts_compared"] != 518 or report["exact_matches"] != 518:
        raise ValueError("The fixed 518-fact comparison did not pass")
    return report


def prepare(run_dir, source_run):
    run = Path(run_dir).resolve()
    source = Path(source_run).resolve()
    if run.exists() or run == source or source in run.parents:
        raise FileExistsError("A new separate recovery namespace is required")
    parent, reused, failed, resources = source_admission(source)
    acceptance = offline_acceptance(reused, resources)
    target_input = next(x for x in parent["inputs"] if (x["conversation"], x["arm"]) == TARGET)
    src = Path(target_input["input"]["path"])
    frozen.check_pointer(target_input["input"])
    dst = run / "conv_1/dated_merged/group_inputs.json"
    dst.parent.mkdir(parents=True)
    with dst.open("xb") as handle:
        handle.write(src.read_bytes())
    if sha(dst) != sha(src):
        raise ValueError("Target input changed while copying")
    value = copy.deepcopy(parent)
    value.pop("protocol_sha256")
    value["created_utc"] = now()
    value["inputs"] = [({**x, "input": frozen.pointer(dst)} if (x["conversation"], x["arm"]) == TARGET else x) for x in parent["inputs"]]
    value["local_bm25_recovery"] = {"schema": SCHEMA, "policy": POLICY,
        "source_manifest": frozen.pointer(source / "run_manifest.json"), "source_run": str(source),
        "terminal_audit": frozen.pointer(AUDIT), "reused_complete_banks": reused,
        "failed_source": failed, "resources": resources, "offline_acceptance": acceptance,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "runtime_source_sha256": {p: sha(ROOT / p) for p in (
            ".venv/Lib/site-packages/fastembed/sparse/bm25.py",
            ".venv/Lib/site-packages/fastembed/sparse/sparse_text_embedding.py",
            ".venv/Lib/site-packages/fastembed/common/model_management.py")},
        "source_date_index_gate_passed": False, "model_calls_during_prepare": 0}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    frozen.save_new(run / "offline_bm25_acceptance.json", acceptance)
    return {"prepared": str(run), "protocol_sha256": digest(value), "reused_banks": 3,
            "new_writer_banks": 1, "offline_exact_fact_vectors": 518, "model_calls": 0}


def manifest(run):
    value = legacy.manifest(run)
    info = value.get("local_bm25_recovery", {})
    if info.get("schema") != SCHEMA or info.get("policy") != POLICY:
        raise ValueError("Local BM25 recovery policy differs")
    for path, expected in {**info["code_sha256"], **info["runtime_source_sha256"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen local recovery code changed: " + path)
    frozen.check_pointer(info["source_manifest"])
    frozen.check_pointer(info["terminal_audit"])
    _, reused, failed, resources = source_admission(info["source_run"])
    if (reused, failed, resources) != (info["reused_complete_banks"], info["failed_source"], info["resources"]):
        raise ValueError("Local BM25 recovery source changed")
    if verify(Path(run) / "offline_bm25_acceptance.json") != {**info["offline_acceptance"], "content_sha256": digest(info["offline_acceptance"])}:
        raise ValueError("Offline acceptance artifact changed")
    return value


def write(run):
    value = manifest(run)
    info = value["local_bm25_recovery"]
    offline_acceptance(info["reused_complete_banks"], info["resources"])
    with frozen.writer_process_lock(run), local_bm25_context(info["resources"]["snapshot"], info["resources"]["files_sha256"]):
        # Legacy run_arm owns the durable bank intent and never re-enters a
        # failed/unknown nontransactional store. It calls frozen.write_arm.
        return legacy.run_arm(run, TARGET[0], TARGET[1], value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    for command in ("check", "write"):
        check_parser = sub.add_parser(command)
        check_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_dir, args.source_run)
    elif args.command == "check":
        value = manifest(args.run_dir)
        info = value["local_bm25_recovery"]
        result = {"protocol_sha256": value["protocol_sha256"],
            "offline_acceptance": offline_acceptance(info["reused_complete_banks"], info["resources"]), "model_calls": 0}
    else:
        result = write(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
