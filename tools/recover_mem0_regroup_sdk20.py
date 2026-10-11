"""Rebuild the four failed dated Mem0 development banks with transport pacing.

No model calls on import/prepare/check. The frozen writer is reused unmodified;
only its child-process API constructor is temporarily replaced. Failed or
uncertain nontransactional stores are never reopened by this runner.
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
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.run_sdk20_modules import sealed, verify, sha, digest, now
from jev_mem.paced_embedding_api import PacedEmbeddingAPI, DEFAULT_GATE, PROTOCOL as PACE_PROTOCOL
from jev_mem.sdk20_cached_api import digest as request_digest

SCHEMA = "dated-mem0-development-rebuild-v1"
AUDIT = ROOT / "research/mem0_global_regroup_dev429_failure_audit_2026-09-28.json"
DIAGNOSTIC = ROOT / "runs/embedding_429_probe_20260928_v1"
CODE = ("tools/recover_mem0_regroup_sdk20.py", "jev_mem/paced_embedding_api.py",
        "jev_mem/sdk20_cached_api.py")
POLICY = {"schema": SCHEMA, "transport": PACE_PROTOCOL, "retries": 2,
          "physical_embedding_start_interval_seconds": 3.0,
          "shared_429_cooldown_seconds": 30.0,
          "cache_policy": "only_original_same_conversation_same_arm_successful_exact_raw",
          "diagnostic_cache_imported": False, "stores": "empty_rebuild_no_partial_store_copy",
          "generation": "pin_each_failed_first_group_first_successful_raw_response",
          "attempt_policy": "one_rebuild_per_bank_no_retry_failed_or_uncertain_bank",
          "conversations": [0, 1], "max_processes": 2,
          "expansion": "not_supported_by_this_runner_requires_four_bank_source_date_index_review"}


def diagnostic_evidence():
    result = read(DIAGNOSTIC / "probe_result.json")
    plan = read(DIAGNOSTIC / "probe_plan.json")
    if (result.get("status") != "complete" or result.get("returned_vectors") != 7
            or result.get("dimensions") != 768 or plan.get("bank_mutations") is not False
            or result.get("source_failed_request_hash") != plan.get("source_failed_request_hash")):
        raise ValueError("Independent successful embedding diagnostic is required")
    paths = [DIAGNOSTIC / name for name in ("probe_result.json", "probe_plan.json", "api_calls.jsonl", "embedding_pacing_events.jsonl")]
    paths.append(DIAGNOSTIC / "api_cache" / (result["source_failed_request_hash"] + ".json"))
    return {"purpose": "recovery_admission_evidence_only_not_a_cache_source", "artifacts": [frozen.pointer(p) for p in paths]}


def source_bindings(source, audit_path):
    """Read-only admission of the frozen four failures and their own raw caches."""
    source = Path(source).resolve()
    parent = frozen.manifest(source)
    audit = read(audit_path)
    if (audit.get("schema") != "dated-mem0-dev429-terminal-audit-v1"
            or audit.get("source_run") != str(source)
            or audit.get("all_four_terminal_failed") is not True
            or audit.get("source_protocol_sha256") != parent["protocol_sha256"]):
        raise ValueError("Failure audit does not bind this source run")
    frozen.check_pointer(audit["source_manifest"])
    rows = audit["rows"]
    if {(r["conversation"], r["arm"]) for r in rows} != {(c, a) for c in (0, 1) for a in frozen.ARMS} or len(rows) != 4:
        raise ValueError("Recovery requires exactly four reviewed development failures")
    for row in rows:
        ci, arm = row["conversation"], row["arm"]
        folder = source / f"conv_{ci}" / arm
        status = frozen.check_pointer(row["status"])
        journal = frozen.check_pointer(row["journal"])
        if (status["status"] != "failed" or status["completed_groups"] != 0
                or journal["status"] != "requires_recovery" or len(journal["llm_calls"]) != 1
                or (folder / "bank.json").exists()):
            raise ValueError("Only the reviewed terminal first-group failures are admitted")
        for key in ("input", "api_ledger", "history"):
            if sha(row[key]["path"]) != row[key]["sha256"]:
                raise ValueError("Failed source artifact changed")
        native = folder / "native"
        inventory = {str(p.relative_to(native)): sha(p) for p in native.rglob("*") if p.is_file()}
        if inventory != row["source_native_inventory"]:
            raise ValueError("Failed source store/cache changed after audit")
        own_cache = native / "api_cache"
        if Path(row["success_cache_root"]) != own_cache:
            raise ValueError("Cache root must be the original same-arm cache")
        expected_files = set()
        for cache in row["success_cache_inventory"]:
            p = Path(cache["path"])
            raw = frozen.check_pointer(cache)
            if p.parent != own_cache or p.stem != request_digest(raw["request"]):
                raise ValueError("Source cache is not an exact same-arm raw request")
            expected_files.add(p)
        if set(own_cache.glob("*.json")) != expected_files:
            raise ValueError("Source cache inventory differs")
        pin = row["raw_first_extraction"]
        raw = frozen.check_pointer(pin)
        if (Path(pin["path"]).parent != own_cache or raw["stage"] != "mem0_extract"
                or pin["request_hash"] != request_digest(raw["request"])
                or pin["parsed_memory"] != journal["llm_calls"][0]["parsed_memory"]):
            raise ValueError("First extraction pin differs from the reviewed journal")
    return parent, rows


def prepare(run_dir, source_run, audit_path=AUDIT):
    run = Path(run_dir).resolve()
    source = Path(source_run).resolve()
    if run.exists() or run == source or source in run.parents:
        raise FileExistsError("Recovery requires a new independent directory")
    parent, rows = source_bindings(source, audit_path)
    diagnostic = diagnostic_evidence()
    run.mkdir(parents=True)
    bindings = []
    for item in parent["inputs"]:
        src = Path(item["input"]["path"])
        frozen.check_pointer(item["input"])
        dst = run / f"conv_{item['conversation']}" / item["arm"] / "group_inputs.json"
        dst.parent.mkdir(parents=True)
        with dst.open("xb") as handle:
            handle.write(src.read_bytes())
        if sha(src) != sha(dst):
            raise ValueError("Frozen group input copy differs")
        bindings.append({**item, "input": frozen.pointer(dst)})
    manifest = copy.deepcopy(parent)
    manifest.pop("protocol_sha256")
    manifest["created_utc"] = now()
    manifest["inputs"] = bindings
    manifest["recovery"] = {"schema": SCHEMA, "policy": POLICY,
        "source_manifest": frozen.pointer(source / "run_manifest.json"),
        "source_audit": frozen.pointer(audit_path), "source_run": str(source),
        "source_bindings": rows, "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "successful_diagnostic": diagnostic,
        "gate_path": str(DEFAULT_GATE.resolve()), "original_writer_reused": True,
        "model_calls_during_prepare": 0}
    publish_json(run / "run_manifest.json", {**manifest, "protocol_sha256": digest(manifest)})
    return {"prepared": str(run), "protocol_sha256": digest(manifest), "model_calls": 0}


def manifest(run):
    value = frozen.manifest(run)
    recovery = value.get("recovery", {})
    if recovery.get("schema") != SCHEMA or recovery.get("policy") != POLICY:
        raise ValueError("Recovery policy differs")
    for path, expected in recovery["code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen recovery source changed: " + path)
    frozen.check_pointer(recovery["source_manifest"])
    frozen.check_pointer(recovery["source_audit"])
    if recovery["successful_diagnostic"] != diagnostic_evidence():
        raise ValueError("Successful diagnostic evidence changed")
    _, rows = source_bindings(recovery["source_run"], recovery["source_audit"]["path"])
    if rows != recovery["source_bindings"]:
        raise ValueError("Reviewed source bindings changed")
    return value


class PinnedRecoveryAPI(PacedEmbeddingAPI):
    """The first group must consume that same arm's already successful raw."""
    def __init__(self, run_dir, *, source_binding, **kwargs):
        self.source_binding = source_binding
        roots = (Path(source_binding["success_cache_root"]).resolve(),)
        if "cache_roots" in kwargs and tuple(Path(p).resolve() for p in kwargs["cache_roots"]) != roots:
            raise ValueError("Cross-arm or diagnostic cache is not admitted")
        kwargs["cache_roots"] = roots
        super().__init__(run_dir, **kwargs)

    def scoped(self, **context):
        return type(self)(self.run_dir, source_binding=self.source_binding,
            keys=self.keys, retries=self.retries, timeout=self.timeout,
            context={**self.context, **context}, gate_path=self.gate_path,
            interval_seconds=self.interval_seconds, cooldown_seconds=self.cooldown_seconds)

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        pin = self.source_binding["raw_first_extraction"]
        original_journal = frozen.check_pointer(self.source_binding["journal"])
        pinned = stage == "mem0_extract" and self.context.get("group_id") == original_journal["group_id"]
        if pinned:
            signature = {"provider": provider, "model": model, "operation": operation, "body": body}
            if request_digest(signature) != pin["request_hash"]:
                raise ValueError("First-group extraction request changed; new generation is forbidden")
            if sha(pin["path"]) != pin["sha256"]:
                raise ValueError("Pinned first extraction raw changed")
        result = super().request(provider, model, body, operation=operation, stage=stage, cache=cache)
        if pinned and sha(self.cache_dir / (pin["request_hash"] + ".json")) != pin["sha256"]:
            raise ValueError("First extraction did not reuse its own exact raw")
        return result


def _write_intent(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def run_arm(run, ci, arm, value):
    run = Path(run)
    folder = run / f"conv_{ci}" / arm
    binding = next(r for r in value["recovery"]["source_bindings"] if (r["conversation"], r["arm"]) == (ci, arm))
    outcome_path = folder / "recovery_outcome.json"
    intent_path = folder / "rebuild_intent.json"
    if outcome_path.exists():
        outcome = verify(outcome_path)
        if (outcome["protocol_sha256"] != value["protocol_sha256"] or outcome["status"] != "complete"):
            raise RuntimeError("Failed rebuild is retained; no implicit second rebuild")
        frozen.check_pointer(outcome["write_status"])
        frozen.check_pointer(outcome["bank"])
        return outcome
    if intent_path.exists() or (folder / "write_status.json").exists() or (folder / "native").exists():
        raise RuntimeError("Prior nontransactional rebuild is retained; do not resume or redispatch")
    _write_intent(intent_path, {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"],
        "conversation": ci, "arm": arm, "pid": os.getpid(), "created_utc": now(),
        "source_journal": binding["journal"], "source_first_extraction": binding["raw_first_extraction"]})
    original_api = frozen.ModelAPI
    def constructor(path, **kwargs):
        return PinnedRecoveryAPI(path, source_binding=binding,
            gate_path=value["recovery"]["gate_path"], **kwargs)
    try:
        frozen.ModelAPI = constructor
        state = frozen.write_arm(run, ci, arm, value)
    finally:
        frozen.ModelAPI = original_api
    outcome = {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"],
        "conversation": ci, "arm": arm, "status": state["status"], "finished_utc": now(),
        "write_status": frozen.pointer(folder / "write_status.json"), "source_journal": binding["journal"],
        "source_first_extraction": binding["raw_first_extraction"], "first_extraction_verified": False}
    journals = sorted((folder / "native/group_audit").rglob("*.json"))
    initial = [p for p in journals if read(p).get("sequence") == 0]
    if not initial and journals:
        initial = [min(journals, key=lambda p: read(p)["sequence"])]
    if initial:
        journal = read(initial[0])
        calls = journal.get("llm_calls", [])
        if calls:
            if len(calls) != 1 or calls[0].get("parsed_memory") != binding["raw_first_extraction"]["parsed_memory"]:
                raise ValueError("Rebuilt first extraction differs from the frozen raw")
            outcome["first_extraction_verified"] = True
            outcome["first_group_journal"] = frozen.pointer(initial[0])
    if state["status"] == "complete":
        if not outcome["first_extraction_verified"]:
            raise ValueError("Complete bank lacks pinned first extraction evidence")
        outcome["bank"] = state["bank"]
    else:
        outcome["error"] = state.get("error")
    frozen.save_new(outcome_path, outcome)
    return outcome


def run_conv(run, ci):
    if ci not in (0, 1):
        raise ValueError("This recovery runner is restricted to the four development banks")
    value = manifest(run)
    return [run_arm(run, ci, arm, value) for arm in frozen.ARMS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--audit", type=Path, default=AUDIT)
    for name in ("check", "write"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_dir, args.source_run, args.audit)
    elif args.command == "check":
        result = {"protocol_sha256": manifest(args.run_dir)["protocol_sha256"], "model_calls": 0}
    else:
        value = manifest(args.run_dir)
        with frozen.writer_process_lock(args.run_dir):
            with ProcessPoolExecutor(max_workers=2) as pool:
                result = [f.result() for f in as_completed([pool.submit(run_conv, args.run_dir, ci) for ci in (0, 1)])]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
