"""Independent full1540 admission for one audited complete C writer arm."""
from __future__ import annotations
import copy
from pathlib import Path
from tools import mem0_regroup_plain_qa_sources_v1 as old
from tools import run_mem0_regroup_sdk20 as writer
from tools import audit_boundary_variants_banks_v1 as audit
from tools.freeze_sdk20_full1540_v1 import verify_lock, LOCK
from tools.run_sdk20_modules import ROOT, verify, digest, sha
from jev_mem.data import SOURCE_HASH

ARMS=tuple(audit.ARM_METHODS.values())
GATE_SCHEMA=audit.GATE_SCHEMA
COHORT=LOCK
CONFIG=old.CONFIG
pointer,checked_pointer=old.pointer,old.checked_pointer
copy_bank,inventory=old.copy_bank,old.inventory


def protocol_config(path=CONFIG):
    verify_lock()
    return old.protocol_config(path)


def cohort():
    lock=verify_lock();rows,dates=old.cohort()
    frozen=lock["dataset"]["questions"]
    actual=[{"conversation":ci,"query_id":q["id"],"source_qa_index":q["source_qa_index"],
             "question":q["question"],"reference":q["reference"],"category":q["category"],"reference_date":dates[str(ci)]}
            for ci in range(10) for q in rows[str(ci)]]
    if len(frozen)!=1540 or frozen!=actual:
        raise ValueError("Original frozen1540 full question/reference/category/date/order differs")
    return rows,dates


def admit_gate(path):
    gate=verify(path)
    required=("ten_complete_banks","source_dates_and_generation_chain_verified",
              "current_sdk20_index_dependencies_valid","content_quality_findings_disclosed")
    raw_arm=gate.get("source_arm");arm=gate.get("arm")
    if (gate.get("schema")!=GATE_SCHEMA or gate.get("status")!="passed" or
        audit.ARM_METHODS.get(raw_arm)!=arm or any(gate.get("checks",{}).get(k) is not True for k in required)):
        raise ValueError("A complete independently audited ten-bank gate required")
    report_path=checked_pointer(gate["audit"]);report=verify(report_path)
    if (report.get("schema")!=audit.SCHEMA or report.get("strict_passed") is not True or
        report.get("conversations")!=list(range(10)) or report.get("arm")!=raw_arm or
        report.get("writer_manifest")!=gate["source_manifest"] or report.get("limitations")!=gate.get("limitations")):
        raise ValueError("Development/partial/failed audit cannot admit full1540")
    if report["code_sha256"]!={p:sha(ROOT/p) for p in audit.CODE}:
        raise ValueError("Audit code binding changed")
    if report.get("audit_tool")!=pointer(audit.__file__):
        raise ValueError("Independent audit tool binding differs")
    for p,h in report["source_file_hashes"].items():
        if sha(p)!=h: raise ValueError("Audited source changed")
    source_manifest=checked_pointer(gate["source_manifest"])
    source=source_manifest.parent.resolve();parent=writer.manifest(source)
    if parent["arms"]!=[raw_arm] or parent.get("boundary_variant",{}).get("arm")!=raw_arm:
        raise ValueError("Gate source writer arm differs")
    if audit.closed_controllers(source,raw_arm,list(range(10)))!=report.get("writer_scheduling"):
        raise ValueError("Writer controller evidence changed")
    if len(report.get("rows",[]))!=10 or {(r.get("conversation"),r.get("arm")) for r in report["rows"]}!={(ci,raw_arm) for ci in range(10)}:
        raise ValueError("Audit must cover all10 distinct source identities")
    rows=gate.get("banks",[])
    if len(rows)!=10 or {(r.get("conversation"),r.get("arm")) for r in rows}!={(ci,arm) for ci in range(10)}:
        raise ValueError("Exactly10 distinct banks required")
    admitted={}
    for row in rows:
        ci=row["conversation"];folder=source/f"conv_{ci}"/raw_arm
        inspected=next(r for r in report["rows"] if (r["conversation"],r["arm"])==(ci,raw_arm))
        if row.get("source_kind")!="strict_complete" or row.get("original_write_status")!="complete" or row.get("source_arm")!=raw_arm:
            raise ValueError("Unknown failure/export exception refused; original failure cannot be erased")
        for key,expected in (("bank",folder/"bank.json"),("write_status",folder/"write_status.json"),
                             ("group_inputs",folder/"group_inputs.json"),("source_manifest",source_manifest)):
            if checked_pointer(row[key]).resolve()!=expected or (key in inspected and row[key]!=inspected[key]):
                raise ValueError("Gate source lineage differs")
        if inspected.get("strict_passed") is not True or inspected.get("source_generation_persistence_passed") is not True:
            raise ValueError("Bank lacks full raw/source/persistence/index proof")
        if set(row.get("audits",{}))!={"source_date","index"}:
            raise ValueError("Both complete source/date and index audits required")
        for refs in row["audits"].values():
            if refs!=[gate["audit"]]:raise ValueError("Audit reference differs")
        state=old.read(folder/"write_status.json");bank=verify(folder/"bank.json")
        if (state.get("status")!="complete" or not state.get("finished_utc") or bank.get("complete") is not True or
            state.get("bank")!=row["bank"] or bank.get("source_dataset_sha256")!=SOURCE_HASH or
            any((v.get("conversation"),v.get("arm"),v.get("protocol_sha256"))!=(ci,raw_arm,parent["protocol_sha256"]) for v in (state,bank))):
            raise ValueError("Original terminal/bank identity differs")
        expected_input=next(r["input"] for r in parent["inputs"] if (r["conversation"],r["arm"])==(ci,raw_arm))
        if bank["group_inputs"]!=expected_input or bank["group_inputs"]!=row["group_inputs"]:
            raise ValueError("Group input differs")
        groups=len(verify(checked_pointer(expected_input))["envelopes"])
        if (state["completed_groups"]!=state["expected_groups"] or state["expected_groups"]!=groups or
            len(bank["journal_files"])!=groups or state["facts"]!=len(bank["facts"]) or
            len(bank["facts"])<20 or len({f["id"] for f in bank["facts"]})!=len(bank["facts"])):
            raise ValueError("Partial/group/fact source cannot be admitted")
        for journal in bank["journal_files"]: checked_pointer(journal)
        native=folder/"native"
        if Path(bank["native_dir"]).resolve()!=native or inventory(native)!=row["native_inventory"] or row["native_inventory"]!=inspected["native_inventory"]:
            raise ValueError("Closed native inventory changed")
        admitted[ci,arm]={"review":copy.deepcopy(row),"bank":bank,"native":native}
    resource=parent["boundary_variant"]["resources"]
    old.validate_resources(resource["snapshot"],resource["files_sha256"])
    return gate,admitted,resource
