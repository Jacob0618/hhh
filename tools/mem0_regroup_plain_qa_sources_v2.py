"""Versioned admission of 17 strict banks, two reviewed exports and one rebuild.

The frozen byte-copy/QA path is reused. Original failed states are never renamed
complete, and an incomplete original bank is never a source of QA facts.
"""
from __future__ import annotations

import copy
from pathlib import Path

from tools import mem0_regroup_plain_qa_sources_v1 as legacy
from tools import mem0_regroup_firstpass_review_v2 as exports
from tools import recover_mem0_regroup_eof_v2 as eof

GATE_SCHEMA = "mem0-dated-regroup-qa-gate-v2"
CHECKS = ("twenty_fact_complete_banks", "source_dates_and_generation_chain_verified",
          "current_sdk20_index_dependencies_reviewed", "native_index_limitations_disclosed")
EXPORT_ROOT = legacy.ROOT / "runs/mem0_regroup_native_metadata_export_v2"
EOF_ROOT = legacy.ROOT / "runs/mem0_global_regroup_eof_recovery_v2"
VALIDATOR_CODE = ("tools/mem0_regroup_firstpass_review_v2.py",
                  "tools/recover_mem0_regroup_eof_v1.py", "tools/recover_mem0_regroup_eof_v2.py",
                  "jev_mem/writer_eof_compat_v1.py")


def __getattr__(name):
    return getattr(legacy, name)


def expected_kind(ci, arm):
    if ci == 3:
        return "reviewed_export"
    return "eof_recovery" if (ci, arm) == (8, "dated_merged") else "strict_complete"


def _path(row, field, expected):
    if legacy.checked_pointer(row[field]).resolve() != Path(expected).resolve():
        raise ValueError("Gate source pointer uses another lineage: " + field)


def _complete(row, origin, parent, *, outcome_name):
    ci, arm = row["conversation"], row["arm"]
    folder = origin / f"conv_{ci}" / arm
    for field, expected in (("bank", folder / "bank.json"), ("write_status", folder / "write_status.json"),
                            ("outcome", folder / outcome_name), ("source_manifest", origin / "run_manifest.json")):
        _path(row, field, expected)
    bank = legacy.verify(row["bank"]["path"])
    status, outcome = legacy.read(row["write_status"]["path"]), legacy.verify(row["outcome"]["path"])
    if (bank.get("complete") is not True or status.get("status") != "complete" or outcome.get("status") != "complete"
            or any(v.get("protocol_sha256") != parent["protocol_sha256"] for v in (bank, status, outcome))
            or any((v.get("conversation"), v.get("arm")) != (ci, arm) for v in (bank, status, outcome))
            or status.get("bank") != row["bank"] or outcome.get("bank") != row["bank"]
            or outcome.get("write_status") != row["write_status"]):
        raise ValueError("Complete bank/status/outcome protocol or identity differs")
    return bank, status, folder / "native"


def _common_chain(row, bank, status, native, parent):
    ci, arm = row["conversation"], row["arm"]
    inp = next(v["input"] for v in parent["inputs"] if (v["conversation"], v["arm"]) == (ci, arm))
    legacy.checked_pointer(bank["group_inputs"])
    if bank["group_inputs"] != inp or bank.get("source_dataset_sha256") != legacy.SOURCE_HASH:
        raise ValueError("Bank dataset or frozen group inputs changed")
    for journal in bank["journal_files"]:
        legacy.checked_pointer(journal)
    count = len(legacy.verify(inp["path"])["envelopes"])
    if (status.get("completed_groups") != count or status.get("expected_groups") != count
            or len(bank["journal_files"]) != count or status.get("facts") != len(bank["facts"])
            or len(bank["facts"]) < 20 or len({f["id"] for f in bank["facts"]}) != len(bank["facts"])):
        raise ValueError("Complete input/group/fact chain is required")
    if Path(bank["native_dir"]).resolve() != native.resolve() or legacy.inventory(native) != row["native_inventory"]:
        raise ValueError("Native source changed after terminal audit")


def assemble_rows(audits_by_bank, limitations_by_bank):
    """Return source rows only, after terminal validation; never publish a gate.

    Both arguments use ``(conversation, arm)`` keys. Audits must supply nonempty
    source_date/index pointer lists and limitations must disclose native defects.
    Root independently combines the twenty current reviews and decides admission.
    """
    population = {(ci, arm) for ci in range(10) for arm in legacy.ARMS}
    if set(audits_by_bank) != population or set(limitations_by_bank) != population:
        raise ValueError("Exactly20 explicit source reviews and native limitations are required")
    parents = legacy.source_manifests()
    recovered = eof.manifest(EOF_ROOT)
    eof.validate_complete(EOF_ROOT, recovered)
    rows = []
    for ci, arm in sorted(population):
        kind = expected_kind(ci, arm)
        if kind == "reviewed_export":
            loaded = exports.check_export_bank(EXPORT_ROOT / f"conv_{ci}" / arm)
            row = {"conversation": ci, "arm": arm, **copy.deepcopy(loaded["review"])}
            parent = parents[str(legacy.REMAINING.resolve())]
            bank, native = loaded["bank"], loaded["native"]
            status = legacy.read(row["original_write_status"]["path"])
        else:
            origin = EOF_ROOT.resolve() if kind == "eof_recovery" else legacy.expected_source(ci, arm).resolve()
            parent = recovered if kind == "eof_recovery" else parents[str(origin)]
            folder = origin / f"conv_{ci}" / arm
            outcome_name = ("eof_recovery_outcome.json" if kind == "eof_recovery" else
                            "write_outcome.json" if ci >= 2 else "recovery_outcome.json")
            row = {"conversation": ci, "arm": arm, "source_kind": kind,
                   "bank": legacy.pointer(folder / "bank.json"), "write_status": legacy.pointer(folder / "write_status.json"),
                   "outcome": legacy.pointer(folder / outcome_name), "source_manifest": legacy.pointer(origin / "run_manifest.json"),
                   "native_inventory": legacy.inventory(folder / "native")}
            if kind == "eof_recovery":
                old = legacy.REMAINING / f"conv_{ci}" / arm
                row.update(original_write_status=legacy.pointer(old / "write_status.json"),
                           original_outcome=legacy.pointer(old / "write_outcome.json"))
            bank, status, native = _complete(row, origin, parent, outcome_name=outcome_name)
        row["audits"] = copy.deepcopy(audits_by_bank[ci, arm])
        row["native_index_limitations"] = copy.deepcopy(limitations_by_bank[ci, arm])
        if not isinstance(row["native_index_limitations"], list) or (ci == 3 and not row["native_index_limitations"]):
            raise ValueError("Explicit native defect disclosure is required")
        for name in ("source_date", "index"):
            refs = row["audits"].get(name)
            if not isinstance(refs, list) or not refs:
                raise ValueError("Per-bank audit pointers are required")
            for ref in refs:
                legacy.checked_pointer(ref)
        _common_chain(row, bank, status, native, parent)
        rows.append(row)
    return rows


def admit_gate(path):
    gate = legacy.verify(path)
    if (gate.get("schema") != GATE_SCHEMA or gate.get("status") != "passed"
            or any(gate.get("checks", {}).get(k) is not True for k in CHECKS)
            or not isinstance(gate.get("limitations"), list) or not gate["limitations"]):
        raise ValueError("The explicit revised twenty-bank review has not passed")
    rows = gate.get("banks", [])
    population = {(ci, arm) for ci in range(10) for arm in legacy.ARMS}
    if len(rows) != 20 or {(r.get("conversation"), r.get("arm")) for r in rows} != population:
        raise ValueError("Exactly20 reviewed conversation-arm banks are required")
    uniform = legacy.verify(legacy.checked_pointer(gate["uniform_twenty_bank_review"]))
    covered = uniform.get("banks", [])
    if len(covered) != 20 or {(r.get("conversation"), r.get("arm")) for r in covered} != population:
        raise ValueError("Uniform review must cover all20 current source banks")
    for row in rows:
        match = next(v for v in covered if (v["conversation"], v["arm"]) == (row["conversation"], row["arm"]))
        if any(match.get(k) != row.get(k) for k in ("bank", "native_inventory", "source_kind")):
            raise ValueError("Uniform review covers another source bank/inventory")
        if (row.get("source_kind") != expected_kind(row["conversation"], row["arm"])
                or not isinstance(row.get("native_index_limitations"), list)):
            raise ValueError("Unreviewed source exception or missing native index disclosure")
        for kind in ("source_date", "index"):
            refs = row.get("audits", {}).get(kind)
            if not isinstance(refs, list) or not refs:
                raise ValueError("Every source needs date/chain and index audit references")
            for ref in refs:
                legacy.checked_pointer(ref)
    parents = legacy.source_manifests()
    recovered_parent = eof.manifest(EOF_ROOT)
    eof.validate_complete(EOF_ROOT, recovered_parent)
    admitted = {}
    for row in rows:
        ci, arm, kind = row["conversation"], row["arm"], row["source_kind"]
        origin = legacy.expected_source(ci, arm).resolve()
        parent = parents[str(origin)]
        if kind == "reviewed_export":
            _path(row, "reviewed_export", EXPORT_ROOT / f"conv_{ci}" / arm / "reviewed_export.json")
            result = exports.check_export_bank(row["reviewed_export"])
            for key, value in result["review"].items():
                if row.get(key) != value:
                    raise ValueError("Reviewed metadata export provenance differs: " + key)
            if row.get("write_status") is not None or row.get("outcome") is not None:
                raise ValueError("Metadata export must preserve original failed status separately")
            if not row["native_index_limitations"]:
                raise ValueError("Known native missing link must remain disclosed")
            bank, native = result["bank"], result["native"]
            status = legacy.read(row["original_write_status"]["path"])
        else:
            if kind == "eof_recovery":
                old = origin / f"conv_{ci}" / arm
                for field, p in (("original_write_status", old / "write_status.json"),
                                 ("original_outcome", old / "write_outcome.json")):
                    _path(row, field, p)
                    if legacy.read(p).get("status") != "failed":
                        raise ValueError("Original partial failure must remain failed")
                origin, parent = EOF_ROOT.resolve(), recovered_parent
                outcome_name = "eof_recovery_outcome.json"
            else:
                outcome_name = "write_outcome.json" if ci >= 2 else "recovery_outcome.json"
            bank, status, native = _complete(row, origin, parent, outcome_name=outcome_name)
        _common_chain(row, bank, status, native, parent)
        admitted[(ci, arm)] = {"review": copy.deepcopy(row), "bank": bank, "native": native}
    resource = parents[str(legacy.REMAINING.resolve())]["remaining_expansion"]["resources"]
    if recovered_parent["eof_recovery"]["resources"] != resource:
        raise ValueError("Recovered bank uses another local embedding/NLP resource binding")
    legacy.validate_resources(resource["snapshot"], resource["files_sha256"])
    return gate, admitted, resource
