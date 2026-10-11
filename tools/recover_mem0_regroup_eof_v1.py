"""One empty-store reconstruction of conv8, with exact original 14 raw pins.

prepare/check/probe are offline. Only explicit run dispatches model requests.
The frozen writer and native SDK are reused, with a scoped response adapter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_regroup_sdk20 as frozen
from tools import run_mem0_regroup_remaining_v1 as remaining
from tools import recover_mem0_regroup_sdk20 as once
from tools.run_pilot400_experiments_v2 import read, publish_json
from tools.run_sdk20_modules import sha, digest, now, verify
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem.paced_embedding_api import PacedEmbeddingAPI, DEFAULT_GATE
from jev_mem.local_bm25_context_v2 import local_bm25_context, forbid_network
from jev_mem.writer_eof_compat_v1 import derive_response, PROTOCOL
from tools.check_experiment_health import shared_read

SCHEMA = "dated-mem0-eof-empty-reconstruction-v1"
SOURCE = ROOT / "runs/mem0_global_regroup_remaining_v1"
DEFAULT_RUN = ROOT / "runs/mem0_global_regroup_eof_recovery_v1"
AUDIT = ROOT / "research/mem0_regroup_conv8_merged_json_failure_audit_2026-09-29.json"
CASE_PROOF = ROOT / "research/mem0_regroup_conv8_eof_root_close_offline_proof_2026-09-29.json"
POLICY_DOC = ROOT / "research/mem0_regroup_failure_followup_policy_2026-09-29.md"
TERMINAL_SEAL = ROOT / "research/mem0_regroup_remaining_firstpass_seal_2026-09-29.json"
CODE = ("tools/recover_mem0_regroup_eof_v1.py", "jev_mem/writer_eof_compat_v1.py",
        "jev_mem/paced_embedding_api.py", "jev_mem/sdk20_cached_api.py",
        "jev_mem/local_bm25_context_v2.py", "tools/check_experiment_health.py")
CI, ARM = 8, "dated_merged"
POLICY = {"schema": SCHEMA, "compatibility": PROTOCOL, "conversation": CI, "arm": ARM,
    "expected_groups": 29, "pinned_extractions": 14, "successful_prefix_groups": 13,
    "source_caches": "same_conversation_same_arm_exact_raw_only",
    "store": "new_empty_store_no_partial_copy", "retries": 2,
    "pinned_payload_drift": "hard_stop_no_dispatch", "bank_attempts": 1,
    "compat_scope": "all_extractions_uniform_without_qa_or_correctness",
    "raw_cache": "never_overwrite_derived_response_in_memory_only",
    "new_groups": "first_normal_requests_only_no_extra_regeneration",
    "qa": "not_started_by_this_tool"}


def inventory(folder):
    # Same non-content Qdrant lock exclusion as the frozen failure audit.
    return {str(p.resolve()): sha(p) for p in Path(folder).rglob("*") if p.is_file() and p.name != ".lock"}


def _ledger(path):
    data = shared_read(path)
    data = data[:data.rfind(b"\n") + 1]
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]


def successful_http(path, key, group_id, response):
    data = shared_read(path)
    data = data[:data.rfind(b"\n") + 1]
    all_rows = [json.loads(x) for x in data.decode("utf-8").splitlines()]
    rows = [r for r in all_rows if r.get("request_hash") == key
            and r.get("stage") == "mem0_extract" and not r.get("cache_hit")
            and r.get("context", {}).get("conversation") == CI
            and r.get("context", {}).get("method") == ARM
            and r.get("context", {}).get("group_id") == group_id]
    ok = [r for r in rows if r.get("status") == 200]
    if (len(ok) != 1 or rows[-1] != ok[0] or ok[0].get("provider") != "gemini"
            or ok[0].get("model") != "gemini-2.5-flash"
            or ok[0].get("operation") != "generateContent"
            or ok[0].get("usage") != response.get("usageMetadata", {})
            or ok[0].get("resolved_model") != response.get("modelVersion")):
        raise ValueError("Original HTTP200 provenance differs")
    return {"path": str(Path(path).resolve()), "prefix_bytes": len(data),
        "prefix_sha256": hashlib.sha256(data).hexdigest(),
        "physical_event_line": all_rows.index(ok[0]) + 1,
        "physical_event_sha256": digest(ok[0]), "physical_event": ok[0]}


def verify_http_evidence(evidence, key, group_id, response):
    data = shared_read(evidence["path"])
    size = evidence["prefix_bytes"]
    if not isinstance(size, int) or size <= 0 or len(data) < size or hashlib.sha256(data[:size]).hexdigest() != evidence["prefix_sha256"]:
        raise ValueError("HTTP ledger prefix changed")
    rows = [json.loads(x) for x in data[:size].decode("utf-8").splitlines()]
    event = rows[evidence["physical_event_line"] - 1]
    if event != evidence["physical_event"] or digest(event) != evidence["physical_event_sha256"]:
        raise ValueError("HTTP event proof changed")
    current = successful_http(evidence["path"], key, group_id, response)
    if current["physical_event"] != event:
        raise ValueError("Original matching HTTP event differs")


def terminal_seal():
    seal = verify(TERMINAL_SEAL)
    if (seal.get("schema") != "mem0-regroup-remaining-firstpass-terminal-seal-v1"
            or seal.get("status") != "all16_terminal_original_failures_preserved"
            or Path(seal["run_dir"]).resolve() != SOURCE.resolve()
            or seal.get("counts", {}).get("banks") != 16
            or len(seal.get("process_snapshots", [])) != 2):
        raise ValueError("Original sixteen-bank terminal seal is required")
    for pointer in seal["process_snapshots"]:
        snapshot = frozen.check_pointer(pointer)
        if snapshot.get("processes") != [] or Path(snapshot["run_dir"]).resolve() != SOURCE.resolve():
            raise ValueError("Terminal seal does not show two empty worker snapshots")
    for pointer in [seal["source_manifest"], *seal["original_code"], *seal["file_inventory"]]:
        if sha(pointer["path"]) != pointer["sha256"] or Path(pointer["path"]).stat().st_size != pointer["bytes"]:
            raise ValueError("Sealed original file changed: " + pointer["path"])
    expected = {str(Path(x["path"]).resolve()) for x in seal["file_inventory"]}
    actual = {str(p.resolve()) for ci in range(2, 10) for p in (SOURCE / f"conv_{ci}").rglob("*")
              if p.is_file() and p.name != ".lock"}
    if expected != actual:
        raise ValueError("Original terminal file inventory changed")
    command = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    processes = json.loads(result.stdout.strip() or "[]")
    if isinstance(processes, dict):
        processes = [processes]
    if any("run_mem0_regroup_remaining_v1.py" in (p.get("CommandLine") or "") for p in processes):
        raise RuntimeError("Original writer is still active")
    return frozen.pointer(TERMINAL_SEAL)


def source_binding():
    parent = remaining.manifest(SOURCE)
    audit = read(AUDIT)
    folder = SOURCE / f"conv_{CI}" / ARM
    expected = {"expected_groups": 29, "completed_groups": 13,
                "persisted_facts": 114, "failed_group_adds": 0}
    if any(audit.get("counts", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("The reviewed source failure population changed")
    if inventory(folder) != audit["target_inventory_before"]:
        raise ValueError("Original failed bank inventory changed")
    status = read(folder / "write_status.json")
    outcome = verify(folder / "write_outcome.json")
    if status["status"] != "failed" or outcome["status"] != "failed" or (folder / "bank.json").exists():
        raise ValueError("Only the reviewed incomplete failed bank is admitted")
    item = next(x for x in parent["inputs"] if (x["conversation"], x["arm"]) == (CI, ARM))
    inp = frozen.check_pointer(item["input"])
    if len(inp["envelopes"]) != 29:
        raise ValueError("Frozen group count differs")
    proof = read(CASE_PROOF)
    rows = audit["source_prefix_audit"]["successful_groups"] + [{
        "group_id": "global_event_0013", "journal": proof["failed_journal"], "raw": proof["original_raw"]}]
    pins = []
    for i, row in enumerate(rows):
        journal = frozen.check_pointer(row["journal"])
        raw = frozen.check_pointer(row["raw"])
        key = request_digest(raw["request"])
        if (row["group_id"] != inp["envelopes"][i]["group_id"] or journal["sequence"] != i
                or len(journal["llm_calls"]) != 1 or raw["stage"] != "mem0_extract"
                or Path(row["raw"]["path"]) != folder / "native/api_cache" / (key + ".json")):
            raise ValueError("Pinned extraction is not its own ordered source response")
        ledger = successful_http(folder / "native/api_calls.jsonl", key, row["group_id"], raw["response"])
        derived, parse = derive_response(raw["response"], http_status=200)
        memory = json.loads(derived["candidates"][0]["content"]["parts"][0]["text"])["memory"]
        if i < 13 and (parse["action"] != "unchanged" or journal["llm_calls"][0]["parsed_memory"] != memory):
            raise ValueError("Successful original extraction differs")
        if i == 13 and (parse["action"] != "append_single_root_brace"
                         or memory != proof["memory_records_unmodified"]):
            raise ValueError("Reviewed EOF extraction differs")
        pins.append({"group_id": row["group_id"], "sequence": i, "raw": row["raw"],
            "journal": row["journal"], "request_hash": key, "signature": raw["request"],
            "parsed_memory": memory, "parse_proof": parse, "physical_ledger": ledger})
    if len(pins) != 14:
        raise ValueError("Exactly fourteen original responses must be pinned")
    binding = {"source_run": str(SOURCE.resolve()), "source_manifest": frozen.pointer(SOURCE / "run_manifest.json"),
        "source_audit": frozen.pointer(AUDIT), "case_proof": frozen.pointer(CASE_PROOF),
        "source_inventory": audit["target_inventory_before"],
        "source_input": item["input"], "cache_root": str((folder / "native/api_cache").resolve()),
        "pins": pins}
    return parent, binding


class PinnedEOFAPI(PacedEmbeddingAPI):
    def __init__(self, run_dir, *, binding, offline_stop=False, **kwargs):
        self.binding, self.offline_stop = binding, offline_stop
        roots = (Path(binding["cache_root"]).resolve(),)
        if "cache_roots" in kwargs and tuple(Path(p).resolve() for p in kwargs["cache_roots"]) != roots:
            raise ValueError("Only the original same-arm cache is admitted")
        kwargs["cache_roots"] = roots
        super().__init__(run_dir, **kwargs)

    def scoped(self, **context):
        return type(self)(self.run_dir, binding=self.binding, offline_stop=self.offline_stop,
            keys=self.keys, retries=self.retries, timeout=self.timeout,
            context={**self.context, **context}, gate_path=self.gate_path,
            interval_seconds=self.interval_seconds, cooldown_seconds=self.cooldown_seconds)

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        key = request_digest(signature)
        gid = self.context.get("group_id")
        pin = next((x for x in self.binding["pins"] if x["group_id"] == gid), None)
        source = Path(self.binding["cache_root"]) / (key + ".json")
        if stage == "mem0_extract":
            if gid is None or (provider, model, operation) != ("gemini", "gemini-2.5-flash", "generateContent"):
                raise ValueError("Extraction scope/model changed")
            if pin and (signature != pin["signature"] or key != pin["request_hash"]):
                raise ValueError("Pinned existing-memory/request chain changed; dispatch forbidden: " + str(gid))
            if pin and sha(source) != pin["raw"]["sha256"]:
                raise ValueError("Pinned original raw changed")
        if source.exists():
            expected = self.binding["source_inventory"].get(str(source.resolve()))
            if expected is None or sha(source) != expected:
                raise ValueError("Source cache is not in the sealed original inventory")
        if self.offline_stop and not source.exists():
            raise ValueError("Offline prefix replay cannot dispatch an uncached request: " + stage + ":" + key)
        response = super().request(provider, model, body, operation=operation, stage=stage, cache=cache)
        if stage != "mem0_extract":
            return response
        target = self.cache_dir / (key + ".json")
        if pin and sha(target) != pin["raw"]["sha256"]:
            raise ValueError("Pinned response was not copied byte for byte")
        ledger = Path(pin["physical_ledger"]["path"]) if pin else self.run_dir / "api_calls.jsonl"
        evidence = successful_http(ledger, key, gid, response)
        derived, proof = derive_response(response, http_status=200)
        lineage = {"schema": SCHEMA, "group_id": gid, "request_hash": key,
            "original_raw": frozen.pointer(target), "source_pin": pin["raw"] if pin else None,
            "physical_ledger": evidence, "parse_proof": proof,
            "compat_source_sha256": sha(ROOT / "jev_mem/writer_eof_compat_v1.py"),
            "derived_response_persisted_to_cache": False}
        dest = self.run_dir / "eof_parse_lineage" / (gid + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if verify(dest) != {**lineage, "content_sha256": digest(lineage)}:
                raise ValueError("Existing parse lineage differs")
        else:
            frozen.save_new(dest, lineage)
        if self.offline_stop and pin and pin["sequence"] == 13:
            raise RuntimeError("OFFLINE_PREFIX_ALL14_EXACT_STOP_BEFORE_ADD")
        return derived


def _copy_input(run, parent, binding):
    source = Path(binding["source_input"]["path"])
    target = run / f"conv_{CI}" / ARM / "group_inputs.json"
    target.parent.mkdir(parents=True)
    with target.open("xb") as f:
        f.write(source.read_bytes())
    value = copy.deepcopy(parent)
    value.pop("protocol_sha256")
    value["created_utc"] = now()
    value["inputs"] = [{**x, "input": frozen.pointer(target)} if (x["conversation"], x["arm"]) == (CI, ARM) else x for x in value["inputs"]]
    return value


def prefix_probe():
    """Real native empty-store replay; all network and credential reads forbidden."""
    parent, binding = source_binding()
    resource = parent["remaining_expansion"]["resources"]
    with tempfile.TemporaryDirectory(prefix="mem0-eof-prefix-") as directory:
        run = Path(directory)
        value = _copy_input(run, parent, binding)
        value["protocol_sha256"] = digest(value)
        original = frozen.ModelAPI
        def constructor(path, **kwargs):
            return PinnedEOFAPI(path, binding=binding, offline_stop=True,
                keys={"GEMINI_API_KEY": "offline-test-gemini", "TYPESAFE_API_KEY": "offline-test-jev"}, **kwargs)
        try:
            frozen.ModelAPI = constructor
            with forbid_network() as guards, patch("jev_mem.providers.load_credentials", side_effect=AssertionError("No credentials in probe")), local_bm25_context(resource["snapshot"], resource["files_sha256"]):
                state = frozen.write_arm(run, CI, ARM, value)
                network_attempts = sum(x.call_count for x in guards)
        finally:
            frozen.ModelAPI = original
        native = run / f"conv_{CI}" / ARM / "native"
        lineages = sorted((native / "eof_parse_lineage").glob("*.json"))
        successful = [read(p) for p in (native / "group_audit").rglob("*.json") if read(p).get("status") == "complete"]
        # Adapter success is persisted as 'committed' in some frozen versions;
        # source journals are authoritative for the exact status spelling.
        source_success_status = read(binding["pins"][0]["journal"]["path"])["status"]
        successful = [read(p) for p in (native / "group_audit").rglob("*.json") if read(p).get("status") == source_success_status]
        hashes = [verify(p)["request_hash"] for p in lineages]
        passed = (state["status"] == "failed" and state["completed_groups"] == 13
            and state["facts"] == 114 and len(successful) == 13 and len(lineages) == 14
            and set(hashes) == {x["request_hash"] for x in binding["pins"]} and network_attempts == 0
            and all(r.get("cache_hit") is True for r in _ledger(native / "api_calls.jsonl")))
        proof = {"schema": SCHEMA + "-prefix-probe", "created_utc": now(), "status": "passed" if passed else "failed",
            "source_binding_sha256": digest(binding), "code_sha256": {p: sha(ROOT / p) for p in CODE},
            "successful_groups": len(successful), "completed_facts": state["facts"], "pinned_generation_requests": len(lineages),
            "network_attempts": network_attempts, "model_calls": 0, "temporary_store_removed": True,
            "state": state, "parse_proofs": [verify(p)["parse_proof"] for p in lineages],
            "request_chain_exact": passed, "uuid_semantics": "fresh_UUIDs_only_original_enumerated_id_text_prompts_exactly_matched"}
    if inventory(SOURCE / f"conv_{CI}" / ARM) != binding["source_inventory"]:
        raise ValueError("Offline probe changed source inventory")
    return proof


def prepare(run_dir, probe_path):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT / "runs").resolve() or run.exists():
        raise FileExistsError("A new direct child of workspace runs is required")
    parent, binding = source_binding()
    seal = terminal_seal()
    probe = verify(probe_path)
    code = {p: sha(ROOT / p) for p in CODE}
    if (probe.get("status") != "passed" or probe.get("source_binding_sha256") != digest(binding)
            or probe.get("code_sha256") != code or probe.get("network_attempts") != 0
            or probe.get("pinned_generation_requests") != 14):
        raise ValueError("Exact fourteen-request empty-store offline probe must pass for frozen code")
    value = _copy_input(run, parent, binding)
    value["eof_recovery"] = {"schema": SCHEMA, "policy": POLICY, "binding": binding,
        "code_sha256": code, "prefix_probe": frozen.pointer(probe_path), "policy_document": frozen.pointer(POLICY_DOC),
        "terminal_seal": seal,
        "resources": parent["remaining_expansion"]["resources"], "gate_path": str(DEFAULT_GATE.resolve())}
    publish_json(run / "run_manifest.json", {**value, "protocol_sha256": digest(value)})
    return {"prepared": str(run), "protocol_sha256": digest(value), "model_calls": 0, "new_bank_count": 1}


def manifest(run):
    value = frozen.manifest(run)
    info = value.get("eof_recovery", {})
    if info.get("schema") != SCHEMA or info.get("policy") != POLICY:
        raise ValueError("EOF recovery policy changed")
    if info["code_sha256"] != {p: sha(ROOT / p) for p in CODE}:
        raise ValueError("Frozen EOF recovery code changed")
    parent, binding = source_binding()
    if binding != info["binding"] or info["resources"] != parent["remaining_expansion"]["resources"]:
        raise ValueError("EOF original source or resources changed")
    if info["terminal_seal"] != terminal_seal():
        raise ValueError("Original terminal seal changed")
    probe = frozen.check_pointer(info["prefix_probe"])
    frozen.check_pointer(info["policy_document"])
    if probe.get("status") != "passed" or probe["code_sha256"] != info["code_sha256"] or probe["source_binding_sha256"] != digest(binding):
        raise ValueError("Offline exact-prefix proof changed")
    return value


def validate_complete(run, value=None):
    run = Path(run)
    value = value or manifest(run)
    folder = run / f"conv_{CI}" / ARM
    outcome = verify(folder / "eof_recovery_outcome.json")
    status = frozen.check_pointer(outcome["write_status"])
    bank = frozen.check_pointer(outcome["bank"])
    if (outcome["status"] != "complete" or bank.get("complete") is not True
            or status["status"] != "complete" or status["completed_groups"] != 29
            or status["expected_groups"] != 29 or len(bank["journal_files"]) != 29
            or any(x["protocol_sha256"] != value["protocol_sha256"] for x in (outcome, status, bank))):
        raise ValueError("EOF reconstruction is not a complete twenty-nine-group bank")
    verify_lineages(folder, value["eof_recovery"]["binding"], bank["journal_files"])
    return outcome


def verify_lineages(folder, binding, journals):
    folder = Path(folder)
    result = []
    for pointer in journals:
        j = frozen.check_pointer(pointer)
        gid = j["group_id"]
        record = verify(folder / "native/eof_parse_lineage" / (gid + ".json"))
        raw = frozen.check_pointer(record["original_raw"])
        pin = next((p for p in binding["pins"] if p["group_id"] == gid), None)
        if (record["request_hash"] != request_digest(raw["request"])
                or record["compat_source_sha256"] != sha(ROOT / "jev_mem/writer_eof_compat_v1.py")
                or record["derived_response_persisted_to_cache"] is not False
                or (pin and (raw["request"] != pin["signature"] or record["original_raw"]["sha256"] != pin["raw"]["sha256"]))):
            raise ValueError("EOF raw pin/lineage differs")
        verify_http_evidence(record["physical_ledger"], record["request_hash"], gid, raw["response"])
        derived, proof = derive_response(raw["response"], http_status=200)
        calls = j["llm_calls"]
        data = json.loads(derived["candidates"][0]["content"]["parts"][0]["text"])
        if (proof != record["parse_proof"] or len(calls) != 1 or calls[0].get("parsed_memory") != data["memory"]
                or json.loads(calls[0]["response"]) != data):
            raise ValueError("Derived raw and actual writer extraction differ")
        # Use the exact original ModelAPI body builder and record validator.
        from jev_mem.providers import ModelAPI, ExtractionLLM
        class Replay:
            generate = ModelAPI.generate
            def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
                if stage != "mem0_extract" or {"provider": provider, "model": model, "operation": operation, "body": body} != raw["request"]:
                    raise ValueError("Writer journal does not reproduce the actual request")
                return derived
        replay = ExtractionLLM(Replay()).generate_response(*calls[0]["args"], **calls[0]["kwargs"])
        if json.loads(replay) != data:
            raise ValueError("Original ExtractionLLM validator differs")
        result.append(frozen.pointer(folder / "native/eof_parse_lineage" / (gid + ".json")))
    if not {p["group_id"] for p in binding["pins"]}.issubset({frozen.check_pointer(j)["group_id"] for j in journals}):
        raise ValueError("Complete reconstruction lacks original pinned prefix")
    return result


def run_once(run):
    run = Path(run).resolve()
    value = manifest(run)
    folder = run / f"conv_{CI}" / ARM
    if (folder / "eof_recovery_outcome.json").exists():
        return validate_complete(run, value)
    if any((folder / name).exists() for name in ("eof_rebuild_intent.json", "write_status.json", "native")):
        raise RuntimeError("Existing failed or unknown reconstruction is retained; no reentry")
    info = value["eof_recovery"]
    once._write_intent(folder / "eof_rebuild_intent.json", {"schema": SCHEMA,
        "protocol_sha256": value["protocol_sha256"], "created_utc": now(), "binding_sha256": digest(info["binding"])})
    original = frozen.ModelAPI
    def constructor(path, **kwargs):
        return PinnedEOFAPI(path, binding=info["binding"], gate_path=info["gate_path"], **kwargs)
    try:
        frozen.ModelAPI = constructor
        with local_bm25_context(info["resources"]["snapshot"], info["resources"]["files_sha256"]):
            state = frozen.write_arm(run, CI, ARM, value)
    finally:
        frozen.ModelAPI = original
    outcome = {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"], "conversation": CI, "arm": ARM,
        "status": state["status"], "finished_utc": now(), "write_status": frozen.pointer(folder / "write_status.json"),
        "original_source_status": "failed", "source_partial_store_copied": False,
        "source_binding_sha256": digest(info["binding"]), "compatibility_protocol": PROTOCOL}
    if state["status"] == "complete":
        bank = frozen.check_pointer(state["bank"])
        outcome["parse_lineages"] = verify_lineages(folder, info["binding"], bank["journal_files"])
        outcome["bank"] = state["bank"]
    frozen.save_new(folder / "eof_recovery_outcome.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare"); p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN); p.add_argument("--probe", type=Path, required=True)
    for command in ("check", "run"):
        p = sub.add_parser(command); p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.command == "probe":
        result = prefix_probe(); frozen.save_new(args.output, result)
        result = {k: v for k, v in result.items() if k not in ("parse_proofs", "state")}
    elif args.command == "prepare":
        result = prepare(args.run_dir, args.probe)
    elif args.command == "check":
        result = {"protocol_sha256": manifest(args.run_dir)["protocol_sha256"], "model_calls": 0}
    else:
        with frozen.writer_process_lock(args.run_dir):
            result = run_once(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
