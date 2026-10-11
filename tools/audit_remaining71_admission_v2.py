"""Independently verify the active subset; never edits a bank or calls models."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now
from tools.audit_pilot400_members_only_v2 import audit as audit_generated
from tools.pilot400_members_only_writer_v2 import OUT, VERSION
from jev_mem.pilot400 import fingerprint, COHORT_SHA256
from jev_mem.data import SOURCE_HASH


def audit():
    review_path = ROOT / "research/pilot400_remaining71_semantic_review_v2_2026-09-28.json"
    review = read(review_path)
    gate_path = OUT / "bank_admission.json"
    gate = read(gate_path)
    if (gate["schema"] != "pilot400-recovery-v2-bank-admission" or gate["source_dataset_sha256"] != SOURCE_HASH
            or gate["cohort_sha256"] != COHORT_SHA256 or gate["review_sha256"] != file_sha(review_path)):
        raise ValueError("Admission gate source/review differs")
    if any(review["scope"][k] for k in ("qa_read", "gold_read", "scores_read", "model_api_calls", "experiment_data_modified")):
        raise ValueError("Review was not source-only")
    for entry in review["source_bindings"]:
        if file_sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Source review input bytes changed")
    generated_audit = audit_generated(OUT)
    rejected, accepted = set(review["excluded_fact_ids"]), set(review["accepted_fact_ids"])
    by_id = {r["fact_id"]: r for r in review["fact_reviews"]}
    if len(rejected) != 7 or len(accepted) != 106 or rejected & accepted or set(by_id) != rejected | accepted:
        raise ValueError("Review population differs")
    all_new, seen_quarantine, banks = set(), set(), []
    for ci, expected_active in ((1, 291), (7, 456)):
        generated_path = OUT / "banks" / ("conv_" + str(ci)) / "bank_overlay.json"
        active_path = generated_path.with_name("bank_overlay_admitted.json")
        generated, active = read(generated_path), read(active_path)
        if gate["banks"][str(ci)]["bank_path"] != active_path.relative_to(OUT).as_posix():
            raise ValueError("Gate names another active bank")
        if (file_sha(active_path) != gate["banks"][str(ci)]["bank_sha256"]
                or file_sha(generated_path) != active["generated_bank_sha256"]
                or active["source_review_sha256"] != file_sha(review_path)):
            raise ValueError("Active/generated/review hash differs")
        expected = [f for f in generated["facts"] if f["id"] not in rejected]
        if active["facts"] != expected or len(expected) != expected_active:
            raise ValueError("Active facts are not the exact order-preserving subset")
        if active["groups"] != generated["groups"]:
            raise ValueError("Generated groups or raw responses changed")
        for key, value in generated.items():
            if key not in ("schema", "facts", "requires_independent_review") and active.get(key) != value:
                raise ValueError("Preserved generated metadata changed: " + key)
        quarantine = [f for f in generated["facts"] if f["id"] in rejected]
        if active["source_review_quarantine"] != [{"fact": f, "review": by_id[f["id"]]} for f in quarantine]:
            raise ValueError("Quarantined original claims/reasons changed or disappeared")
        for group in generated["groups"]:
            if group["kind"] == VERSION:
                all_new.update(f["id"] for f in group["facts"])
                record_path = OUT / "banks" / ("conv_" + str(ci)) / "groups" / group["group_id"] / "record.json"
                for i, fact in enumerate(group["facts"]):
                    decision = by_id[fact["id"]]
                    if (decision["index"] != i or decision["conversation"] != ci
                            or decision["group_id"] != group["group_id"] or decision["record_sha256"] != file_sha(record_path)
                            or decision["fact_text_unchanged"] != fact["text"] or decision["proposed_source_ids"] != fact["sources"]
                            or decision["temporal_status_unchanged"] != fact["temporal_status"]):
                        raise ValueError("Review does not bind exact unedited generated fact")
            elif any(f["id"] in rejected for f in group["facts"]):
                raise ValueError("Review removed an inherited fact outside scope")
        seen_quarantine.update(f["id"] for f in quarantine)
        banks.append({"conversation": ci, "generated_facts": len(generated["facts"]), "active_facts": len(expected),
            "excluded": len(quarantine), "groups": len(active["groups"]), "exact_subset": True,
            "generated_records_preserved": True, "fact_text_date_source_embedding_unchanged": True,
            "generated_bank_sha256": file_sha(generated_path), "active_bank_sha256": file_sha(active_path),
            "active_fact_body_fingerprint": fingerprint(expected), "quarantined_fact_ids": [f["id"] for f in quarantine]})
    if all_new != accepted | rejected or seen_quarantine != rejected:
        raise ValueError("Global exact review/quarantine coverage differs")
    return {"schema": "remaining71-independent-admission-subset-audit-v2", "status": "passed", "created_utc": now(),
        "source_code_sha256": file_sha(__file__), "bank_admission_sha256": file_sha(gate_path),
        "source_review_sha256": file_sha(review_path), "generated_provenance_revalidated": generated_audit["status"],
        "reviewed_new_facts": len(all_new), "admitted_new_facts": len(accepted), "excluded_new_facts": len(rejected),
        "banks": banks, "model_calls": 0, "credentials_loaded": False,
        "limitations": "Exact extraction/admission lineage only; source review is assistant review, not human semantic gold."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Audit output already exists; choose a new immutable path")
    result = audit()
    publish_json(args.out, result)
    print({"status": result["status"], "active_facts": [b["active_facts"] for b in result["banks"]],
           "excluded": result["excluded_new_facts"], "report_sha256": file_sha(args.out), "model_calls": 0})
