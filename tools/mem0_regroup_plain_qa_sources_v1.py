"""Read-only admission and byte copying of twenty reviewed dated Mem0 banks.

No Mem0 store is opened here. A four-bank development gate is insufficient.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from tools import run_mem0_regroup_sdk20 as writer
from tools import recover_mem0_regroup_sdk20 as recovery
from tools import recover_mem0_regroup_bm25_v2 as bm25_recovery
from tools import run_mem0_regroup_remaining_v1 as remaining
from tools.sdk20_module_sources import _native_inventory
from tools.run_sdk20_modules import ROOT, digest, sha, verify, save_new
from tools.run_pilot400_experiments_v2 import read
from jev_mem.data import SOURCE_HASH, load_data, queries, session_envelopes
from jev_mem import evaluation_plain_text_v1 as evaluation
from jev_mem.plain_text_api_v1 import PROTOCOL as PROVIDER_PROTOCOL
from jev_mem.local_bm25_context_v2 import validate_resources

ARMS = writer.ARMS
GATE_SCHEMA = "mem0-dated-regroup-qa-gate-v1"
CHECKS = ("twenty_complete_banks", "source_dates_and_generation_chain_verified",
          "current_sdk20_index_dependencies_valid", "content_quality_findings_disclosed")
CONFIG = ROOT / "configs/plain_text_qa_v1.json"
DEVELOPMENT = ROOT / "runs/mem0_global_regroup_recovery_v1"
FOURTH = ROOT / "runs/mem0_global_regroup_bm25_recovery_v2"
REMAINING = ROOT / "runs/mem0_global_regroup_remaining_v1"


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def checked_pointer(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Expected a path/hash artifact binding")
    if pointer(value["path"]) != value:
        raise ValueError("Reviewed artifact path or hash differs")
    return Path(value["path"])


def protocol_config(path=CONFIG):
    value = verify(path)
    expected = {"schema": "plain-text-qa-protocol-v1", "evaluation_protocol": evaluation.PROTOCOL,
                "reader_protocol": evaluation.READER_PROTOCOL, "judge_protocol": evaluation.JUDGE_PROTOCOL,
                "provider_protocol": PROVIDER_PROTOCOL, "reader_stage": evaluation.READER_STAGE,
                "judge_stage": evaluation.JUDGE_STAGE,
                "models": {"reader": evaluation.READER_MODEL, "judge": evaluation.JUDGE_MODEL},
                "generation": {"temperature": 0, "maxOutputTokens": 2048, "reader_thinkingBudget": 0,
                               "json_output": False, "responseMimeType": None, "responseSchema": None},
                "prompts": {"reader_system": evaluation.READER_SYSTEM, "judge_template": evaluation.JUDGE_TEMPLATE}}
    if any(value.get(k) != v for k, v in expected.items()) or value["selected_facts"]["budget"] != 20:
        raise ValueError("Frozen plain-text protocol differs from implementation")
    for path, expected_hash in value["source_sha256"].items():
        if sha(ROOT / path) != expected_hash:
            raise ValueError("Frozen plain-text source changed: " + path)
    return value


def cohort():
    rows, dates = {}, {}
    for ci, item in enumerate(load_data()):
        rows[str(ci)] = list(queries(item, ci))
        dates[str(ci)] = list(session_envelopes(item))[-1]["observation_date"]
    ids = [q["id"] for qs in rows.values() for q in qs]
    if len(ids) != 1540 or len(set(ids)) != 1540:
        raise ValueError("The fixed category1-4 cohort must contain exactly1540 unique questions")
    return rows, dates


def expected_source(ci, arm):
    if ci in (0, 1):
        return FOURTH if (ci, arm) == (1, "dated_merged") else DEVELOPMENT
    return REMAINING


def source_manifests():
    return {str(DEVELOPMENT.resolve()): recovery.manifest(DEVELOPMENT),
            str(FOURTH.resolve()): bm25_recovery.manifest(FOURTH),
            str(REMAINING.resolve()): remaining.manifest(REMAINING)}


def inventory(native):
    # A closed history database needs no uncheckpointed WAL/journal. Never copy
    # a live SQLite main file and silently omit its committed WAL contents.
    native = Path(native)
    for p in native.rglob("*"):
        if p.is_file() and p.name.endswith(("-wal", "-journal")) and p.stat().st_size:
            raise ValueError("Uncheckpointed SQLite sidecar requires terminal review")
    return _native_inventory(native)


def admit_gate(path):
    gate = verify(path)
    if (gate.get("schema") != GATE_SCHEMA or gate.get("status") != "passed"
            or any(gate.get("checks", {}).get(k) is not True for k in CHECKS)):
        raise ValueError("All twenty terminal banks require an explicit source/date/index QA review")
    if not isinstance(gate.get("limitations"), list) or not gate["limitations"]:
        raise ValueError("Known model/source-support limitations must be disclosed")
    rows = gate.get("banks", [])
    if (len(rows) != 20 or {(r.get("conversation"), r.get("arm")) for r in rows}
            != {(c, a) for c in range(10) for a in ARMS}):
        raise ValueError("QA admission requires exactly20 distinct conversation-arm banks")
    parents = source_manifests()
    admitted = {}
    for row in rows:
        ci, arm = row["conversation"], row["arm"]
        origin = expected_source(ci, arm).resolve()
        parent = parents[str(origin)]
        folder = origin / f"conv_{ci}" / arm
        outcome_name = "write_outcome.json" if ci >= 2 else "recovery_outcome.json"
        for field, expected in (("bank", folder / "bank.json"), ("write_status", folder / "write_status.json"),
                                ("outcome", folder / outcome_name), ("source_manifest", origin / "run_manifest.json")):
            if checked_pointer(row[field]).resolve() != expected:
                raise ValueError("Gate uses another bank lineage")
        for kind in ("source_date", "index"):
            refs = row.get("audits", {}).get(kind)
            if not isinstance(refs, list) or not refs:
                raise ValueError("Every bank needs source/date and current-SDK20 index review references")
            for ref in refs:
                checked_pointer(ref)
        bank, status, outcome = verify(row["bank"]["path"]), read(row["write_status"]["path"]), verify(row["outcome"]["path"])
        if (bank.get("complete") is not True or status.get("status") != "complete" or outcome.get("status") != "complete"
                or any(v.get("protocol_sha256") != parent["protocol_sha256"] for v in (bank, status, outcome))
                or any((v.get("conversation"), v.get("arm")) != (ci, arm) for v in (bank, status, outcome))
                or bank.get("source_dataset_sha256") != SOURCE_HASH
                or status.get("bank") != row["bank"] or outcome.get("bank") != row["bank"]
                or outcome.get("write_status") != row["write_status"]):
            raise ValueError("Source bank terminal state or protocol binding differs")
        expected_input = next(v["input"] for v in parent["inputs"] if (v["conversation"], v["arm"]) == (ci, arm))
        checked_pointer(bank["group_inputs"])
        if bank["group_inputs"] != expected_input:
            raise ValueError("Bank group input differs from its frozen writer manifest")
        for journal in bank["journal_files"]:
            checked_pointer(journal)
        group_count = len(verify(bank["group_inputs"]["path"])["envelopes"])
        if (status.get("completed_groups") != group_count or status.get("expected_groups") != group_count
                or len(bank["journal_files"]) != group_count or status.get("facts") != len(bank["facts"])
                or len(bank["facts"]) < 20 or len({f["id"] for f in bank["facts"]}) != len(bank["facts"])):
            raise ValueError("Source groups or facts are incomplete")
        native = folder / "native"
        if Path(bank["native_dir"]).resolve() != native or inventory(native) != row.get("native_inventory"):
            raise ValueError("Terminal native source inventory differs from the audited snapshot")
        admitted[(ci, arm)] = {"review": copy.deepcopy(row), "bank": bank, "native": native}
    resource = parents[str(REMAINING.resolve())]["remaining_expansion"]["resources"]
    validate_resources(resource["snapshot"], resource["files_sha256"])
    return gate, admitted, resource


def copy_bank(dest, ci, arm, source):
    """Copy only complete native search/history files, with before/after hashes."""
    dest = Path(dest).resolve()
    folder, native_source = dest / f"conv_{ci}" / arm, source["native"].resolve()
    if folder.exists() or dest == native_source or native_source in dest.parents or dest in native_source.parents:
        raise FileExistsError("QA copy requires a new namespace outside the immutable source")
    initial = source["review"]["native_inventory"]
    if inventory(native_source) != initial:
        raise ValueError("Native source changed before copying")
    native_copy = folder / "native"
    native_copy.mkdir(parents=True, exist_ok=False)
    for name, expected in initial.items():
        src, dst = native_source / name, native_copy / name
        if not src.resolve().is_relative_to(native_source) or not dst.resolve().is_relative_to(native_copy):
            raise ValueError("Inventory path escapes its native directory")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != expected:
            raise ValueError("QA native copy differs")
    if inventory(native_source) != initial or inventory(native_copy) != initial:
        raise ValueError("Source changed during byte copy or copy is incomplete")
    original = source["bank"]
    bank = {"schema": "mem0-dated-plain-qa-bank-v1", "conversation": ci, "arm": arm,
            "user_id": original["user_id"], "facts": copy.deepcopy(original["facts"]),
            "native_dir": str(native_copy), "native_source": str(native_source), "source_inventory": initial,
            "source_bank": source["review"]["bank"], "source_review": source["review"],
            "source_dataset_sha256": SOURCE_HASH, "fact_support": original["fact_support"],
            "inherited_api_ledger_bytes": 0}
    save_new(folder / "bank.json", bank)
    return pointer(folder / "bank.json")
