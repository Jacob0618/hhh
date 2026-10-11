"""Offline admission for one ten-bank refinement arm and the fixed 400 QA IDs."""
from __future__ import annotations
import copy
from pathlib import Path

from tools import mem0_regroup_plain_qa_sources_v1 as old
from tools import run_regroup_refinement_writer_v1 as writer
from tools.run_sdk20_modules import ROOT, verify, digest, sha
from tools.run_pilot400_experiments_v2 import read
from jev_mem.data import SOURCE_HASH, load_data, session_envelopes
from jev_mem.pilot400 import load_cohort

ARMS = writer.ARMS
GATE_SCHEMA = "regroup-refinement-qa-gate-v1"
CHECKS = ("ten_complete_banks", "source_dates_and_generation_chain_verified",
          "current_sdk20_index_dependencies_valid", "content_quality_findings_disclosed")
COHORT = ROOT / "configs/pilot400_manifest_v1.json"
CONFIG = old.CONFIG
pointer, checked_pointer = old.pointer, old.checked_pointer
copy_bank, inventory, protocol_config = old.copy_bank, old.inventory, old.protocol_config


def cohort():
    data = load_data()
    _, selected = load_cohort(COHORT, data)
    rows = {str(ci): [q for q in selected if q["conversation"] == ci] for ci in range(10)}
    dates = {str(ci): list(session_envelopes(item))[-1]["observation_date"] for ci, item in enumerate(data)}
    if sum(map(len, rows.values())) != 400 or any(not qs for qs in rows.values()):
        raise ValueError("The original stratified400 cohort is required")
    return rows, dates


def admit_gate(path):
    gate = verify(path)
    if (gate.get("schema") != GATE_SCHEMA or gate.get("status") != "passed"
            or gate.get("arm") not in ARMS or any(gate.get("checks", {}).get(k) is not True for k in CHECKS)):
        raise ValueError("An explicit complete ten-bank source/date/index review is required")
    if not isinstance(gate.get("limitations"), list) or not gate["limitations"]:
        raise ValueError("Known source/model limitations must be disclosed")
    source_manifest = checked_pointer(gate["source_manifest"])
    source = source_manifest.parent.resolve()
    if source_manifest.name != "run_manifest.json":
        raise ValueError("Writer manifest path required")
    parent = writer.manifest(source)
    arm = gate["arm"]
    if parent["arms"] != [arm]:
        raise ValueError("Writer arm differs from gate")
    rows = gate.get("banks", [])
    if len(rows) != 10 or {(r.get("conversation"), r.get("arm")) for r in rows} != {(c, arm) for c in range(10)}:
        raise ValueError("Exactly ten distinct complete conversation banks required")
    admitted = {}
    for row in rows:
        ci = row["conversation"]
        folder = source / f"conv_{ci}" / arm
        for field, expected in (("bank", folder / "bank.json"), ("write_status", folder / "write_status.json"),
                                ("outcome", folder / "write_outcome.json"), ("source_manifest", source_manifest)):
            if checked_pointer(row[field]).resolve() != expected:
                raise ValueError("Gate points to another bank lineage")
        for kind in ("source_date", "index"):
            refs = row.get("audits", {}).get(kind)
            if not isinstance(refs, list) or not refs:
                raise ValueError("Each bank needs source/date and index review evidence")
            for ref in refs:
                checked_pointer(ref)
        bank, status, outcome = verify(row["bank"]["path"]), read(row["write_status"]["path"]), verify(row["outcome"]["path"])
        if (bank.get("complete") is not True or status.get("status") != "complete" or outcome.get("status") != "complete"
                or any(v.get("protocol_sha256") != parent["protocol_sha256"] for v in (bank, status, outcome))
                or any((v.get("conversation"), v.get("arm")) != (ci, arm) for v in (bank, status, outcome))
                or bank.get("source_dataset_sha256") != SOURCE_HASH or status.get("bank") != row["bank"]
                or outcome.get("bank") != row["bank"] or outcome.get("write_status") != row["write_status"]):
            raise ValueError("Bank terminal outcome/source protocol differs")
        expected_input = next(r["input"] for r in parent["inputs"] if (r["conversation"], r["arm"]) == (ci, arm))
        checked_pointer(bank["group_inputs"])
        if bank["group_inputs"] != expected_input:
            raise ValueError("Bank group inputs changed")
        for journal in bank["journal_files"]:
            checked_pointer(journal)
        group_count = len(verify(bank["group_inputs"]["path"])["envelopes"])
        if (status.get("completed_groups") != group_count or status.get("expected_groups") != group_count
                or len(bank["journal_files"]) != group_count or status.get("facts") != len(bank["facts"])
                or len(bank["facts"]) < 20 or len({f["id"] for f in bank["facts"]}) != len(bank["facts"])):
            raise ValueError("Group/fact count incomplete")
        native = folder / "native"
        if Path(bank["native_dir"]).resolve() != native or inventory(native) != row.get("native_inventory"):
            raise ValueError("Native source inventory differs from terminal audit")
        admitted[(ci, arm)] = {"review": copy.deepcopy(row), "bank": bank, "native": native}
    resource = parent["refinement"]["resources"]
    old.validate_resources(resource["snapshot"], resource["files_sha256"])
    return gate, admitted, resource
