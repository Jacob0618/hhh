"""Versioned EOF recovery entry: validate Markdown as bytes, never as JSON.

The prepared v1 namespace is immutable and has never run. Its response adapter,
native writer, exact-prefix probe and terminal verifier are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_mem0_regroup_eof_v1 as v1

SCHEMA = "dated-mem0-eof-empty-reconstruction-entry-v2"
DEFAULT_RUN = ROOT / "runs/mem0_global_regroup_eof_recovery_v2"
PREPARED_V1 = ROOT / "runs/mem0_global_regroup_eof_recovery_v1"
PREPARED_V1_PROTOCOL = "7304728753b1e78e61bb36329564fb1e21c540b5c02b94470f19444a9300b876"
CODE = (*v1.CODE, "tools/recover_mem0_regroup_eof_v2.py")
POLICY = {**v1.POLICY, "schema": SCHEMA,
    "entry_change": "Markdown_policy_pointer_verified_as_bytes_not_JSON",
    "prior_v1": "prepared_only_check_failed_zero_bank_intent_zero_native_zero_API",
    "generation_and_compatibility_changed": False}


def file_pointer(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": v1.sha(path), "bytes": path.stat().st_size}


def check_file_pointer(pointer):
    """Validate a file without assuming its contents are JSON."""
    if (not isinstance(pointer, dict) or set(pointer) != {"path", "sha256", "bytes"}
            or file_pointer(pointer["path"]) != pointer):
        raise ValueError("Immutable file bytes/hash differ")
    return Path(pointer["path"])


def prior_prepared_binding():
    # frozen.manifest handles JSON only and verifies all original writer hashes.
    # Do not call the v1 entry manifest, whose known bug JSON-decodes Markdown.
    prior = v1.frozen.manifest(PREPARED_V1)
    expected = {str((PREPARED_V1 / name).resolve()) for name in
                ("run_manifest.json", "conv_8/dated_merged/group_inputs.json")}
    actual = {str(p.resolve()) for p in PREPARED_V1.rglob("*") if p.is_file()}
    if prior["protocol_sha256"] != PREPARED_V1_PROTOCOL or actual != expected:
        raise ValueError("Prepared v1 changed or contains an attempted writer/API")
    info = prior["eof_recovery"]
    if info["schema"] != v1.SCHEMA or info["policy"] != v1.POLICY or info["code_sha256"] != {p: v1.sha(ROOT / p) for p in v1.CODE}:
        raise ValueError("Prepared v1 frozen code or policy changed")
    policy = info["policy_document"]
    if Path(policy["path"]).resolve() != v1.POLICY_DOC.resolve() or v1.sha(policy["path"]) != policy["sha256"]:
        raise ValueError("Original Markdown policy changed")
    return {"run_dir": str(PREPARED_V1.resolve()), "protocol_sha256": prior["protocol_sha256"],
        "files": [file_pointer(p) for p in sorted(expected)],
        "failure": {"phase": "check_before_run", "type": "JSONDecodeError",
                    "cause": "JSON_loading_Markdown_policy_pointer", "bank_intents": 0, "native_stores": 0, "model_calls": 0}}


def prefix_probe():
    prior = prior_prepared_binding()
    proof = v1.prefix_probe()
    proof["schema"] = SCHEMA + "-prefix-probe"
    proof["code_sha256"] = {p: v1.sha(ROOT / p) for p in CODE}
    proof["prepared_v1_binding_sha256"] = v1.digest(prior)
    return proof


def prepare(run_dir, probe_path):
    run = Path(run_dir).resolve()
    if run.parent != (ROOT / "runs").resolve() or run.exists():
        raise FileExistsError("A new direct child of workspace runs is required")
    parent, binding = v1.source_binding()
    seal = v1.terminal_seal()
    prior = prior_prepared_binding()
    probe = v1.verify(probe_path)
    code = {p: v1.sha(ROOT / p) for p in CODE}
    if (probe.get("schema") != SCHEMA + "-prefix-probe" or probe.get("status") != "passed"
            or probe.get("source_binding_sha256") != v1.digest(binding)
            or probe.get("code_sha256") != code or probe.get("network_attempts") != 0
            or probe.get("pinned_generation_requests") != 14
            or probe.get("prepared_v1_binding_sha256") != v1.digest(prior)):
        raise ValueError("Frozen v2 exact-prefix offline proof must pass")
    value = v1._copy_input(run, parent, binding)
    value["eof_recovery"] = {"schema": SCHEMA, "policy": POLICY, "binding": binding,
        "code_sha256": code, "prefix_probe": v1.frozen.pointer(probe_path),
        "policy_document": file_pointer(v1.POLICY_DOC), "terminal_seal": seal,
        "prepared_v1": prior, "resources": parent["remaining_expansion"]["resources"],
        "gate_path": str(v1.DEFAULT_GATE.resolve())}
    v1.publish_json(run / "run_manifest.json", {**value, "protocol_sha256": v1.digest(value)})
    return {"prepared": str(run), "protocol_sha256": v1.digest(value), "model_calls": 0, "new_bank_count": 1}


def manifest(run):
    value = v1.frozen.manifest(run)
    info = value.get("eof_recovery", {})
    if info.get("schema") != SCHEMA or info.get("policy") != POLICY:
        raise ValueError("EOF v2 entry policy changed")
    if info["code_sha256"] != {p: v1.sha(ROOT / p) for p in CODE}:
        raise ValueError("Frozen EOF v2 code changed")
    parent, binding = v1.source_binding()
    if binding != info["binding"] or info["resources"] != parent["remaining_expansion"]["resources"]:
        raise ValueError("Original EOF source or resources changed")
    if info["terminal_seal"] != v1.terminal_seal() or info["prepared_v1"] != prior_prepared_binding():
        raise ValueError("Original terminal seal or prepared v1 changed")
    if check_file_pointer(info["policy_document"]).resolve() != v1.POLICY_DOC.resolve():
        raise ValueError("Policy document path changed")
    probe = v1.frozen.check_pointer(info["prefix_probe"])
    if (probe.get("schema") != SCHEMA + "-prefix-probe" or probe.get("status") != "passed"
            or probe["code_sha256"] != info["code_sha256"]
            or probe["source_binding_sha256"] != v1.digest(binding)
            or probe["prepared_v1_binding_sha256"] != v1.digest(info["prepared_v1"])):
        raise ValueError("Exact-prefix proof binding differs")
    return value


def validate_complete(run, value=None):
    value = value or manifest(run)
    outcome = v1.validate_complete(run, value)
    if outcome["schema"] != SCHEMA:
        raise ValueError("EOF v2 outcome schema differs")
    return outcome


def run_once(run):
    run = Path(run).resolve()
    value = manifest(run)
    folder = run / f"conv_{v1.CI}" / v1.ARM
    if (folder / "eof_recovery_outcome.json").exists():
        return validate_complete(run, value)
    if any((folder / name).exists() for name in ("eof_rebuild_intent.json", "write_status.json", "native")):
        raise RuntimeError("Existing failed or unknown reconstruction is retained; no reentry")
    info = value["eof_recovery"]
    v1.once._write_intent(folder / "eof_rebuild_intent.json", {"schema": SCHEMA,
        "protocol_sha256": value["protocol_sha256"], "created_utc": v1.now(), "binding_sha256": v1.digest(info["binding"])})
    original = v1.frozen.ModelAPI
    def constructor(path, **kwargs):
        return v1.PinnedEOFAPI(path, binding=info["binding"], gate_path=info["gate_path"], **kwargs)
    try:
        v1.frozen.ModelAPI = constructor
        with v1.local_bm25_context(info["resources"]["snapshot"], info["resources"]["files_sha256"]):
            state = v1.frozen.write_arm(run, v1.CI, v1.ARM, value)
    finally:
        v1.frozen.ModelAPI = original
    outcome = {"schema": SCHEMA, "protocol_sha256": value["protocol_sha256"], "conversation": v1.CI, "arm": v1.ARM,
        "status": state["status"], "finished_utc": v1.now(), "write_status": v1.frozen.pointer(folder / "write_status.json"),
        "original_source_status": "failed", "source_partial_store_copied": False,
        "source_binding_sha256": v1.digest(info["binding"]), "compatibility_protocol": v1.PROTOCOL}
    if state["status"] == "complete":
        bank = v1.frozen.check_pointer(state["bank"])
        outcome["parse_lineages"] = v1.verify_lineages(folder, info["binding"], bank["journal_files"])
        outcome["bank"] = state["bank"]
    v1.frozen.save_new(folder / "eof_recovery_outcome.json", outcome)
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
        result = prefix_probe(); v1.frozen.save_new(args.output, result)
        result = {k: v for k, v in result.items() if k not in ("parse_proofs", "state")}
    elif args.command == "prepare":
        result = prepare(args.run_dir, args.probe)
    elif args.command == "check":
        result = {"protocol_sha256": manifest(args.run_dir)["protocol_sha256"], "model_calls": 0}
    else:
        with v1.frozen.writer_process_lock(args.run_dir):
            result = run_once(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
