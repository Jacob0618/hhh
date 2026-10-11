"""Combine actual terminal/source/index/quality reviews; never run a model.

check uses only temporary provisional documents. publish is a separate explicit
operation writing two new research artifacts; no QA namespace is prepared.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import mem0_regroup_plain_qa_sources_v2 as sources
from tools.run_sdk20_modules import save_new, digest, now
from tools.summarize_mem0_sparse1540 import Snapshot

SCHEMA = "mem0-dated-regroup-final-twenty-source-review-v2"
DEFAULTS = {"development": ROOT / "research/mem0_regroup_four_bank_development_gate_2026-09-29.json",
    "firstpass": ROOT / "research/mem0_regroup_uniform20_firstpass_review_v2_2026-09-29.json",
    "core": ROOT / "research/mem0_regroup_firstpass_complete_facts_core_v2_2026-09-29.json",
    "quality": ROOT / "research/mem0_regroup_remaining_first_pass_source_quality_2026-09-29.json",
    "eof_core": ROOT / "research/mem0_regroup_eof_independent_audit_v1_2026-09-29.json",
    "eof_quality": ROOT / "research/mem0_regroup_eof_source_quality_2026-09-29.json"}
CORE_CHECKS = ("selected_banks_all_groups_complete", "source_partition_and_exact_messages", "source_dates_and_roles",
    "native_prompt_exact_with_declared_date_adapter", "new_first_generation_lineage_verified", "raw_to_parsed_to_add_to_bank",
    "native_history_adds", "facts_dense768_and_bm25", "entities_dense768", "existing_entity_links_valid_and_native_orphans_disclosed",
    "mixed_date_metadata_null", "verified_support_not_fabricated", "source_artifacts_unchanged", "original_writer_statuses_preserved")
EOF_CHECKS = ("all_29_original_or_derived_raw_independently_verified", "original_14_raw_pins_byte_identical_no_redispatch",
    "original_partial_failure_preserved", "facts_bm25_exact_offline_reencoding", "source_dates_and_generation_chain_verified",
    "current_sdk20_index_dependencies_valid")
LIMITATIONS = [
    "This is an explicit post-failure source-admission v2: original strict outcomes remain17complete/3failed. Two complete c3 fact banks are separately reviewed metadata exports; c8 is a separately documented EOF compatibility reconstruction.",
    "Native stores and indexes remain unrepaired. The two c3 entity-bearing orphan facts remain searchable as facts but lack the original entity link; complete semantic entity-edge coverage is not claimed.",
    "Facts dense/BM25, existing entity dense and valid referenced links are the current SDK20 dependencies. Unused entity sparse omissions/empty vectors are disclosed, not silently repaired.",
    "Qualitative source reviews disclose native mistakes, omissions, role attribution and approximate/ambiguous dates; they are assistant inspections, not independent human gold or measured factual accuracy. No facts were rewritten or regenerated based on quality.",
    "Cross-date groups retain per-turn provenance and null fact observation_date where appropriate; group input scope is not per-fact verified support. Noncontiguous grouping may expose future-dated prior history, so this is offline memory, not an online causal result.",
    "Both arms use the same dated writer/context embedding adapter and original SDK20 selection; development conversations0/1 were inspected. This1540 cohort is not previously untouched blind evaluation.",
    "Both Reader and Judge now output plain text under the fixed new protocol. Historical Judge65.639 and previous JSON evaluations are references only and cannot be mixed into this comparison or attributed to grouping alone.",
    "QA has not been run by this gate assembler. All1540 questions per arm, independent Reader/F1 and Judge missing denominators, and technical failures must remain explicit."]


def key(row):
    return row["conversation"], row["arm"]


def unique_rows(rows, expected):
    result = {key(row): row for row in rows}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError("Review has duplicate, missing or unexpected conversation-arm rows")
    return result


def pointer(snapshot, path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": snapshot.sha(path)}


def check_pointer(snapshot, ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or pointer(snapshot, ref["path"]) != ref:
        raise ValueError("Review evidence pointer changed")


def load(snapshot, path):
    value = snapshot.read(path)
    if "content_sha256" in value and value["content_sha256"] != digest({k:v for k,v in value.items() if k != "content_sha256"}):
        raise ValueError("Review content seal differs")
    return value


def quality_targets(core, *, include_eof=False):
    result = {}
    for bank in core["banks"]:
        for row in bank["qualitative_targets"]:
            result[(*key(bank), row["group_id"])] = row["journal"]
        if include_eof:
            # Compatibility changes only syntax, but its recovered facts still
            # receive an explicit original-source quality review.
            row = next(j for j in bank["journals"] if j["group_id"] == "global_event_0013")
            result[(*key(bank), row["group_id"])] = row["journal"]
    return result


def validate_quality(snapshot, quality, expected, *, eof=False, bank=None, core_ref=None):
    schema = "mem0-regroup-eof-source-quality-v1" if eof else "mem0-regroup-first-pass-source-quality-v2"
    status = "source_quality_review_complete" if eof else "all_first_pass_successful_target_groups_reviewed_recovery_still_pending"
    if (quality.get("schema") != schema or quality.get("status") != status or quality.get("qa_gold_used") is not False
            or quality.get("api_calls") != 0 or quality.get("source_mutations") != 0
            or not isinstance(quality.get("limitations"), list) or not quality["limitations"]):
        raise ValueError("Required source-quality disclosure is not complete or uses another protocol")
    if eof and (quality.get("bank") != bank or quality.get("structural_audit") != core_ref):
        raise ValueError("EOF quality review describes another recovered bank or structural audit")
    groups = quality.get("groups", [])
    actual = {(*key(g), g["group_id"]): g for g in groups}
    if len(actual) != len(groups) or set(actual) != set(expected):
        raise ValueError("Every targeted cross-date/empty/EOF group needs source review")
    for identity, row in actual.items():
        quality_read = (row.get("all_turns_read") is True and row.get("all_extracted_records_read") is True) if eof else row.get("quality_only_no_bank_mutations") is True
        if (row.get("source_review_completed") is not True or not quality_read
                or row.get("journal") != expected[identity] or not isinstance(row.get("findings"), list)):
            raise ValueError("Quality group journal or explicit completed review differs")
        check_pointer(snapshot, row["journal"])


def inspect_reviews(paths, snapshot):
    # Reject all missing prerequisites before expensive source hashing or any publication.
    if set(paths) != set(DEFAULTS) or any(not Path(p).is_file() for p in paths.values()):
        raise FileNotFoundError("All six actual reviews, including terminal EOF and its source quality, are required")
    values = {name: load(snapshot, path) for name, path in paths.items()}
    refs = {name: pointer(snapshot, path) for name, path in paths.items()}
    dev, old, core, eof = (values[k] for k in ("development", "firstpass", "core", "eof_core"))
    if (dev.get("schema") != "mem0-dated-regroup-development-gate-v1" or dev.get("status") != "passed"
            or any(dev.get("checks", {}).get(k) is not True for k in
                   ("four_complete_banks", "source_dates_and_generation_chain_verified", "current_sdk20_index_dependencies_valid", "content_quality_findings_disclosed"))):
        raise ValueError("Development four-bank source/index/quality review is required")
    population = {(c,a) for c in range(10) for a in sources.ARMS}
    devrows = unique_rows(dev["banks"], {(c,a) for c in (0,1) for a in sources.ARMS})
    oldrows = unique_rows(old["banks"], population)
    if (old.get("schema") != "mem0-regroup-firstpass-uniform20-review-v2"
            or old.get("status") != "reviewed_with_three_original_failures"
            or old.get("counts", {}).get("original_complete") != 17 or old["counts"].get("original_failed") != 3
            or old.get("checks", {}).get("all20_complete_fact_banks") is not False
            or any(old.get("checks", {}).get(k) is not True for k in ("uniform20_index_and_orphan_review", "original_sources_unchanged"))):
        raise ValueError("Original strict failures must remain explicit; first-pass is not the final gate")
    for name, report in (("core",core),("eof_core",eof)):
        if (report.get("structural_integrity_passed") is not True or report.get("all_read_artifact_hashes_unchanged") is not True
                or report.get("native_index_policy") != "original_unrepaired"
                or any(report.get("checks", {}).get(k) is not True for k in CORE_CHECKS)):
            raise ValueError("Complete source/date/history/current-index core review is missing: " + name)
    corekeys = {(c,a) for c in range(2,10) for a in sources.ARMS} - {(8,"dated_merged")}
    corerows = unique_rows(core["banks"], corekeys)
    corerefs = unique_rows(core["allbanks"], corekeys)
    eofrow = unique_rows(eof["banks"], {(8,"dated_merged")})[8,"dated_merged"]
    eofref = unique_rows(eof["allbanks"], {(8,"dated_merged")})[8,"dated_merged"]
    if (eof.get("recovered_bank_structurally_complete") is not True or eofrow.get("groups") != 29
            or eofrow.get("recovery_write_status") != "complete" or eofrow.get("source_kind") != "eof_recovery"
            or any(eof.get("checks", {}).get(k) is not True for k in EOF_CHECKS)):
        raise ValueError("EOF recovery has not passed the full29-group independent review")
    validate_quality(snapshot, values["quality"], quality_targets(core))
    validate_quality(snapshot, values["eof_quality"], quality_targets(eof,include_eof=True), eof=True,
                     bank=eofrow["bank"], core_ref=refs["eof_core"])
    for report in (old, core, eof):
        for p,h in report.get("source_file_hashes", report.get("audited_file_hashes", {})).items():
            if snapshot.sha(p) != h:
                raise ValueError("Previously audited evidence bytes changed")
    for name in ("source_date","index"):
        for p in dev["audits"][name]:check_pointer(snapshot,p)
    return values,refs,devrows,oldrows,corerows,corerefs,eofrow,eofref


def build(paths=None):
    paths = dict(DEFAULTS if paths is None else paths)
    snapshot = Snapshot()
    values,refs,devrows,oldrows,core,corerefs,eofrow,eofref = inspect_reviews(paths,snapshot)
    audits, limitations = {}, {}
    for identity, old in oldrows.items():
        ci,arm = identity
        if ci < 2:
            audits[identity] = {k:[*values["development"]["audits"][k],refs["firstpass"]] for k in ("source_date","index")}
        elif identity == (8,"dated_merged"):
            audits[identity] = {"source_date":[refs["eof_core"],refs["eof_quality"]], "index":[refs["eof_core"],refs["firstpass"]]}
        else:
            audits[identity] = {"source_date":[refs["core"],refs["quality"]], "index":[refs["core"],refs["firstpass"]]}
        index = eofrow if identity == (8,"dated_merged") else old
        limitations[identity] = [{"kind":"native_orphan_fact_ids", "ids": index.get("orphan_fact_ids",[])},
            {"kind":"unused_entity_sparse_missing", "count":index.get("entity_bm25_missing",len(index.get("entity_sparse_missing",[])))},
            {"kind":"unused_entity_sparse_empty", "items":index.get("entity_bm25_empty",index.get("entity_sparse_empty",[]))},
            {"kind":"not_exhaustive_semantic_edge_validation", "disclosed":True}]
    rows = sources.assemble_rows(audits,limitations)
    if Counter(r["source_kind"] for r in rows) != {"strict_complete":17,"reviewed_export":2,"eof_recovery":1}:
        raise ValueError("Final source composition changed")
    totals = Counter(); current_details = []
    for row in rows:
        identity=key(row);ci,arm=identity;old=oldrows[identity]
        index = eofrow if identity==(8,"dated_merged") else old
        expected = eofref if identity==(8,"dated_merged") else corerefs[identity] if ci>=2 else old
        if row["native_inventory"] != expected["native_inventory"]:
            raise ValueError("Current native bytes differ from the actual index review")
        for field in ("bank","source_manifest"):
            if field=="bank" and ci==3:continue
            if row[field] != expected[field]:raise ValueError("Audited current bank identity differs")
        if ci<2:
            if any(row[k]!=devrows[identity][k] for k in ("bank","write_status","outcome")):
                raise ValueError("Development bank review no longer matches")
        if row["source_kind"]=="reviewed_export":
            if row["original_write_status"]!=old["write_status"] or row["original_outcome"]!=old["outcome"]:
                raise ValueError("Export hides the original failed provenance")
        elif identity==(8,"dated_merged"):
            if row["original_write_status"]!=old["write_status"] or row["original_outcome"]!=old["outcome"]:
                raise ValueError("EOF recovery hides the original partial provenance")
        bank = load(snapshot,row["bank"]["path"])
        if len(bank["facts"])!=index["facts"] or len(bank["journal_files"])!=index["groups"]:
            raise ValueError("Reviewed current fact/group counts differ")
        totals.update(facts=len(bank["facts"]),groups=len(bank["journal_files"]))
        current_details.append({"conversation":ci,"arm":arm,"source_kind":row["source_kind"],
            "original_write_status":old["original_write_status"],"reviewed_export_status":"complete" if ci==3 else None,
            "recovery_write_status":"complete" if identity==(8,"dated_merged") else None,
            "facts":len(bank["facts"]),"groups":len(bank["journal_files"]),"orphan_fact_ids":index.get("orphan_fact_ids",[]),
            "future_dated_history_groups":(eofrow if identity==(8,"dated_merged") else core.get(identity,{})).get("future_dated_history_groups",[]),
            "native_index_policy":"original_unrepaired"})
    for row in rows:
        for group in row["audits"].values():
            for ref in group:check_pointer(snapshot,ref)
    if snapshot.unchanged():raise ValueError("Review sources changed while assembling")
    report={"schema":SCHEMA,"status":"twenty_current_fact_banks_reviewed","created_utc":now(),"banks":rows,
        "source_reviews":refs,"counts":{"banks":20,"strict_complete":17,"reviewed_export":2,"eof_recovery":1,**totals},
        "current_bank_details":current_details,"checks":{k:True for k in sources.CHECKS},"limitations":LIMITATIONS,
        "original_strict_outcomes":{"complete":17,"failed":3,"rewritten":False},
        "quality_scope":"Bound development review, all23 first-pass cross-date/empty targets, plus all recovered EOF cross-date/empty targets and compatibility group. Not all-fact semantic certification.",
        "input_sha256":{str(p):h for p,h in snapshot.files.items()},"assembler":sources.pointer(__file__),
        "model_api_calls":0,"native_clients":0,"qa_started":False}
    gate={"schema":sources.GATE_SCHEMA,"status":"passed","created_utc":report["created_utc"],"banks":copy.deepcopy(rows),
          "checks":{k:True for k in sources.CHECKS},"limitations":LIMITATIONS,"assembled_by":sources.pointer(__file__),
          "scope":"Only revised-source admission for the frozen two-arm plain-text1540 QA; original strict failures remain preserved."}
    return report,gate


def preflight(report,gate):
    with tempfile.TemporaryDirectory(prefix="mem0-final20-gate-check-") as directory:
        rp,gp=Path(directory)/"uniform.json",Path(directory)/"gate.json"
        save_new(rp,report);save_new(gp,{**gate,"uniform_twenty_bank_review":sources.pointer(rp)})
        sources.admit_gate(gp)


def publish(report_path,gate_path,paths=None):
    rp,gp=Path(report_path).resolve(),Path(gate_path).resolve()
    research=(ROOT/"research").resolve()
    if rp==gp or rp.exists() or gp.exists() or not rp.is_relative_to(research) or not gp.is_relative_to(research):
        raise ValueError("Publication requires two distinct new research artifact paths")
    report,gate=build(paths);preflight(report,gate)
    for p,h in report["input_sha256"].items():
        if sources.sha(p)!=h:raise ValueError("Evidence changed before publication")
    save_new(rp,report)
    save_new(gp,{**gate,"uniform_twenty_bank_review":sources.pointer(rp)})
    return {"uniform_twenty_bank_review":sources.pointer(rp),"gate":sources.pointer(gp),"model_api_calls":0,"qa_started":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("check","publish"))
    parser.add_argument("--report",type=Path);parser.add_argument("--gate",type=Path)
    for name,path in DEFAULTS.items():parser.add_argument("--"+name.replace("_","-"),type=Path,default=path)
    args=parser.parse_args();paths={k:getattr(args,k) for k in DEFAULTS}
    if args.phase=="publish":
        if args.report is None or args.gate is None:parser.error("publish requires --report and --gate")
        value=publish(args.report,args.gate,paths)
    else:
        report,gate=build(paths);preflight(report,gate)
        value={"ready":True,"counts":report["counts"],"persistent_gate_published":False,"model_api_calls":0}
    print(json.dumps(value,ensure_ascii=False))


if __name__=="__main__":main()
